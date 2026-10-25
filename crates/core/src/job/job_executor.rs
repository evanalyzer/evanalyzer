//! # job_executor
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-06-05
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.

use crate::pipeline::scope::Scope;
use crate::{
    ImageContainer, ImageInfo, ZProjection,
    algos::Connectivity,
    dataset::{DatasetAlgorithm, DatasetContext},
    image::{ImageReader, ImageTile, PixelSizes, ReadMode},
    pipeline::{
        image_cache::ImageCache,
        object_cache::ObjectCache,
        pipeline::{Pipeline, PipelineImageMeta},
        pipeline_cache::{CacheAddress, ChannelVariant, GlobalImageMeta, GlobalPipelineCache},
    },
    storage::{ImageSummary, PipelineResultExporter},
    utils::resources::MAX_TILE_SIZE,
};
use evanalyzer_cfg::{
    core_types::{
        BreakpointMode, BreakpointSettings, ImageAddress, InternalErrors, ObjectClass, PipelineId,
    },
    settings::{
        images_settings::{
            GlobalImageSettings, ImageEntry, TStackHandling, ZStackHandling, ZStackSettings,
        },
        object_settings::ObjectMetricSettings,
    },
};
use indexmap::IndexMap;
use kornia_image::ImageSize;
use log::{debug, info, warn};
use rayon::prelude::*;
use std::{
    collections::{HashMap, HashSet},
    ops::RangeInclusive,
    path::PathBuf,
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, AtomicUsize, Ordering},
        mpsc::{Receiver, Sender},
    },
    thread::JoinHandle,
    time::Instant,
};

pub enum ProgressEvent {
    Started {
        total: usize,
    },
    /// Emitted once the tile list is known, before any tile starts processing.
    /// Allows the UI to show the correct total immediately.
    TilesScheduled {
        total_tiles: usize,
    },
    /// Emitted after each tile completes when processing a single image in parallel.
    /// Carries the ROIs found in that tile so callers can update previews incrementally.
    TileCompleted {
        tile_index: usize,
        total_tiles: usize,
        objects: Vec<ObjectMetricSettings>,
    },
    /// Emitted once the whole-image-scoped phase (tile-merge, plus any
    /// `WholeImage`-scoped commands like Voronoi, plus the DB export) for
    /// one (t, z) stack completes - always *after* every tile of that stack
    /// has already reported `TileCompleted`. `total_tiles` (see
    /// `TilesScheduled`) reserves one unit per (t, z) stack for exactly this
    /// event, so a progress bar driven by `TileCompleted`/this event
    /// together never claims 100% before the pipeline is actually done -
    /// this phase can itself take a long time (e.g. Voronoi across a
    /// whole-slide image's full object set), and without a dedicated event
    /// there was nothing to report while it ran.
    WholeImagePhaseCompleted {
        completed: usize,
        total_tiles: usize,
    },
    /// Emitted when an image starts. With several images in parallel more
    /// than one is in work; each ends with `ImageCompleted` or `ImageFailed`.
    ImageStarted {
        path: PathBuf,
    },
    ImageCompleted {
        index: usize,
        total: usize,
        path: PathBuf,
    },
    ImageFailed {
        path: PathBuf,
    },
    Finished,
    /// Emitted when the pipeline stops at a breakpoint.  Carries the
    /// intermediate image so the UI can display it in the viewport.
    BreakpointReached {
        image: crate::image::ImageContainer,
        /// The segmentation/instance label maps captured at the same step,
        /// if the pipeline had produced them by then (`None` before
        /// `Threshold`/`ConnectedComponents`/`Watershed` respectively).
        segmentation: Option<crate::image::ImageContainer>,
        instances: Option<crate::image::ImageContainer>,
        /// Tile origin in image-pixel coordinates.
        tile_offset_x: usize,
        tile_offset_y: usize,
        tile_width: usize,
        tile_height: usize,
        /// Original image bit depth (e.g. 8, 12, 16) — used for the
        /// pixel-value HUD so values are scaled to the real range.
        nr_bits: u16,
        /// The channel the breakpointed pipeline actually started from
        /// (`ImageAddress::Channel(n)`), so the UI can look up *that*
        /// channel's histogram/LUT settings instead of guessing. `None`
        /// when the pipeline starts from something other than a plain
        /// channel address (e.g. scratchpad or a memory slot).
        channel_idx: Option<i32>,
    },
}

/// Controls tile selection when running a preview on a single image.
///
/// Only meaningful for `analyze_image`'s interactive calling context (see its
/// doc comment); a batch run's `self.preview_tile_settings` is always `None`,
/// so this setting has no effect there and every tile is always processed.
pub struct PreviewTileSettings {
    /// Current pan offset (screen pixels from the image's top-left corner).
    pub offset_x: f32,
    pub offset_y: f32,
    /// Viewport dimensions in screen pixels.
    pub viewport_width: f32,
    pub viewport_height: f32,
    /// Current zoom level (1.0 = 100 %).
    pub zoom: f32,
    /// When `false` (default) only tiles intersecting the viewport are processed.
    /// When `true`  visible tiles run first, then the remaining tiles follow in a
    /// second parallel batch - useful for an exhaustive preview that still gives
    /// fast feedback for the area the user is looking at.
    pub process_all_tiles: bool,
}

impl PreviewTileSettings {
    fn is_tile_visible(&self, tile: &ImageTile) -> bool {
        let x1 = tile.offset_x as f32 * self.zoom + self.offset_x;
        let y1 = tile.offset_y as f32 * self.zoom + self.offset_y;
        let x2 = (tile.offset_x + tile.width) as f32 * self.zoom + self.offset_x;
        let y2 = (tile.offset_y + tile.height) as f32 * self.zoom + self.offset_y;
        x1 < self.viewport_width && x2 > 0.0 && y1 < self.viewport_height && y2 > 0.0
    }
}

/// The one place `evanalyzer_cfg::settings::project_settings::TileMergeConnectivity`
/// (project settings) gets converted into `algos::Connectivity` (the algorithm's
/// own parameter type) - `TileMerge` itself never references project settings
/// or `job_executor` at all; this is where that boundary gets crossed, right
/// where `TileMerge` actually gets constructed below.
impl From<evanalyzer_cfg::settings::project_settings::TileMergeConnectivity> for Connectivity {
    fn from(value: evanalyzer_cfg::settings::project_settings::TileMergeConnectivity) -> Self {
        match value {
            evanalyzer_cfg::settings::project_settings::TileMergeConnectivity::FourConnected => {
                Connectivity::FourConnected
            }
            evanalyzer_cfg::settings::project_settings::TileMergeConnectivity::EightConnected => {
                Connectivity::EightConnected
            }
        }
    }
}

/// The `PipelineId` `job_generator` assigns to the system-inserted `TileMerge`
/// post-process pipeline (see its own doc comment: not a user-pickable
/// command, prepended by `job_generator` from project-level settings). Needs
/// to be a name both `job_generator` (constructs it) and `job_executor`
/// (must run it before any other post-process pipeline - `get_execution_order`
/// alone doesn't guarantee that, see `analyze_image`) can reference, rather
/// than each hand-writing the same magic sentinel value.
pub(crate) const TILE_MERGE_PIPELINE_ID: PipelineId = PipelineId(0xFFFFFFFF);

pub struct JobExecutor {
    pub project_path: PathBuf,
    pub output_path: PathBuf,

    // The job_generator splits the user created pipelines in two parts.
    // The pipelines steps which can be executed in parallel and thos which has to be executed
    // after the preprocessing has been finished
    pub pipelines_pre_process: IndexMap<PipelineId, Pipeline>,
    pub pipelines_post_process: IndexMap<PipelineId, Pipeline>,
    /// Dataset steps, in the order the pipelines and their steps are
    /// authored. They run once after every image has been analysed.
    pub dataset_steps: Vec<Box<dyn DatasetAlgorithm>>,
    /// Run only the dataset steps, on the objects already in the results
    /// storage (a copy of an earlier analysis's) - no image is analysed.
    pub skip_image_phase: bool,

    pub image_base_path: PathBuf,
    pub images: IndexMap<PathBuf, ImageEntry>,
    pub global_image_settings: GlobalImageSettings,
    pub result_storage: Arc<dyn PipelineResultExporter>,
    pub override_pixel_sizes: Option<PixelSizes>,
    /// When set, tile selection in single-image preview runs is guided by the
    /// viewport position.  `None` means process all tiles (normal full run).
    pub preview_tile_settings: Option<PreviewTileSettings>,

    /// Debugging settings, if set the pipeline stops at this point and returns the actual image
    pub breakpoint: Option<BreakpointSettings>,
    /// How many tiles send their breakpoint image: the visible ones nearest
    /// the viewport centre (at least 1).
    pub breakpoint_tiles: usize,

    /// The project's class colors (`0xRRGGBB`), handed to every image's pipeline cache
    pub class_colors: Arc<HashMap<ObjectClass, u32>>,
}

impl<'a> JobExecutor {
    pub fn new(
        project_path: PathBuf,
        output_path: PathBuf,
        images: IndexMap<PathBuf, ImageEntry>,
        image_base_path: PathBuf,
        global_image_settings: GlobalImageSettings,
        result_storage: Arc<dyn PipelineResultExporter>,
        override_pixel_sizes: Option<PixelSizes>,
        class_colors: Arc<HashMap<ObjectClass, u32>>,
    ) -> Self {
        Self {
            pipelines_pre_process: IndexMap::new(),
            pipelines_post_process: IndexMap::new(),
            dataset_steps: Vec::new(),
            skip_image_phase: false,
            project_path,
            output_path,
            image_base_path,
            images,
            global_image_settings,
            result_storage,
            override_pixel_sizes,
            preview_tile_settings: None,
            breakpoint: None,
            breakpoint_tiles: 1,
            class_colors,
        }
    }

    pub fn add_pre_process_pipeline(&mut self, p: Pipeline) {
        self.pipelines_pre_process.insert(p.id, p);
    }

    pub fn add_dataset_step(&mut self, step: Box<dyn DatasetAlgorithm>) {
        self.dataset_steps.push(step);
    }

    /// Runs the dataset steps on the exported objects of every image - once
    /// the image phase has finished.
    fn run_dataset_steps(
        &self,
        progress: &Sender<ProgressEvent>,
        cancel: &Arc<AtomicBool>,
    ) -> Result<(), InternalErrors> {
        if self.dataset_steps.is_empty() {
            return Ok(());
        }
        let store = self.result_storage.dataset_store().ok_or_else(|| {
            InternalErrors::Generic(
                "These results can't be read back, so the dataset steps can't run on them".into(),
            )
        })?;
        let mut ctx = DatasetContext {
            store,
            progress: Some(progress.clone()),
            cancel: cancel.clone(),
        };
        crate::dataset::run_dataset_steps(&self.dataset_steps, &mut ctx)
    }

    pub fn add_post_process_pipeline(&mut self, p: Pipeline) {
        self.pipelines_post_process.insert(p.id, p);
    }

    /// Runs all images through the configured pipelines, blocking until complete.
    ///
    /// Images are processed in parallel up to `parallelism` threads. Progress events
    /// are sent on `progress` as each image completes or fails. Returns the first
    /// error encountered; remaining in-flight images are abandoned.
    ///
    /// Prefer [`run_async`](Self::run_async) when calling from a GUI or CLI that
    /// needs to stay responsive while the job runs.
    ///
    /// # Arguments
    /// * `parallelism` - Maximum number of images to analyze concurrently
    /// * `progress` - Sender to receive [`ProgressEvent`]s during execution
    ///
    /// # Example
    /// ```no_run
    /// use std::sync::mpsc;
    /// use evanalyzer_core::{generate_job_from_project_settings, ProgressEvent};
    ///
    /// let job = generate_job_from_project_settings(&config)?;
    /// let (tx, rx) = mpsc::channel();
    ///
    /// std::thread::spawn(move || job.run(4, tx));
    ///
    /// for event in rx {
    ///     if let ProgressEvent::ImageCompleted { index, total, .. } = event {
    ///         println!("{index}/{total}");
    ///     }
    /// }
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn run(
        &self,
        parallelism: usize,
        progress: Sender<ProgressEvent>,
        cancel: Arc<AtomicBool>,
    ) -> Result<(), InternalErrors> {
        info!("Starting pipeline with {} parallel threads", parallelism);
        let start = Instant::now();

        let ram_budget = self.estimate_ram_budget();
        let image_cache_bytes = crate::utils::resources::recommended_image_cache_bytes(
            parallelism,
            ram_budget.working_set_bytes + ram_budget.object_cache_margin_bytes,
            ram_budget.min_image_cache_bytes,
        );
        info!(
            "Image cache capacity: {} bytes/worker ({}) x {} worker(s)",
            image_cache_bytes,
            crate::utils::resources::format_binary_bytes(image_cache_bytes),
            parallelism
        );

        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(parallelism)
            .build()
            .unwrap();

        let order = self.get_execution_order();
        let total = if self.skip_image_phase {
            0
        } else {
            self.images.len()
        };
        let completed = AtomicUsize::new(0);

        progress.send(ProgressEvent::Started { total }).ok();

        let result = pool.install(|| {
            if self.skip_image_phase {
                info!("Image phase skipped: running the dataset steps only");
                Ok(())
            } else if total == 1 {
                // Single image: parallelize over tiles instead of images
                let (rel_path, image_info) = self.images.iter().next().unwrap();
                let abs_path = self.image_base_path.join(rel_path);
                progress
                    .send(ProgressEvent::ImageStarted {
                        path: rel_path.clone(),
                    })
                    .ok();
                match self.analyze_image(
                    rel_path,
                    &abs_path,
                    image_info,
                    &order,
                    self.result_storage.clone(),
                    cancel.clone(),
                    Some(progress.clone()),
                    image_cache_bytes,
                ) {
                    Ok(()) => {
                        progress
                            .send(ProgressEvent::ImageCompleted {
                                index: 1,
                                total,
                                path: rel_path.clone(),
                            })
                            .ok();
                        Ok(())
                    }
                    // A cancel is not a failure of this image - pass it on
                    // unwrapped so callers can tell "cancelled by user" from
                    // a real error, same as the multi-image branch below.
                    Err(InternalErrors::Cancelled) => Err(InternalErrors::Cancelled),
                    Err(e) => {
                        progress
                            .send(ProgressEvent::ImageFailed {
                                path: rel_path.clone(),
                            })
                            .ok();
                        // Name the failing image in the propagated error - the
                        // caller (GUI/CLI) only sees this string, not the
                        // ImageFailed event above, so without the path the
                        // user has no way to tell which file broke a batch.
                        Err(InternalErrors::Internal(format!(
                            "{}: {e}",
                            rel_path.display()
                        )))
                    }
                }
            } else {
                // Multiple images: parallelize over images. Every image is
                // always attempted, regardless of whether an earlier one
                // failed - a single malformed file must not silently stop
                // the rest of a large batch from being processed (unlike
                // `try_for_each`, which aborts remaining work on the first
                // `Err`). `cancel` is still checked per-item so an explicit
                // cancel request stops new work from starting, same as before.
                let failures: Mutex<Vec<PathBuf>> = Mutex::new(Vec::new());
                self.images.par_iter().for_each(|(rel_path, image_info)| {
                    if cancel.load(Ordering::Relaxed) {
                        return;
                    }
                    let abs_path = self.image_base_path.join(rel_path);
                    progress
                        .send(ProgressEvent::ImageStarted {
                            path: rel_path.clone(),
                        })
                        .ok();
                    match self.analyze_image(
                        rel_path,
                        &abs_path,
                        image_info,
                        &order,
                        self.result_storage.clone(),
                        cancel.clone(),
                        None,
                        image_cache_bytes,
                    ) {
                        Ok(()) => {
                            let index = completed.fetch_add(1, Ordering::Relaxed) + 1;
                            progress
                                .send(ProgressEvent::ImageCompleted {
                                    index,
                                    total,
                                    path: rel_path.clone(),
                                })
                                .ok();
                        }
                        Err(e) => {
                            progress
                                .send(ProgressEvent::ImageFailed {
                                    path: rel_path.clone(),
                                })
                                .ok();
                            warn!("{}: {e}", rel_path.display());
                            failures
                                .lock()
                                .expect("failures mutex poisoned")
                                .push(rel_path.clone());
                        }
                    }
                });

                let failures = failures.into_inner().expect("failures mutex poisoned");
                if cancel.load(Ordering::Relaxed) {
                    Err(InternalErrors::Cancelled)
                } else if failures.is_empty() {
                    Ok(())
                } else {
                    Err(InternalErrors::Internal(format!(
                        "{} of {total} image(s) failed: {}",
                        failures.len(),
                        failures
                            .iter()
                            .map(|p| p.display().to_string())
                            .collect::<Vec<_>>()
                            .join(", ")
                    )))
                }
            }
        });

        // Only on complete results: a step over all images must not see a
        // partial set (a failed or cancelled image).
        let result = result.and_then(|()| self.run_dataset_steps(&progress, &cancel));

        info!("Pipeline completed in {:?}", start.elapsed());
        // Before `Finished`: whoever reads the results next sees the outcome.
        if let Err(e) = self.result_storage.finish_run(&result) {
            warn!("Could not record how the run ended: {e}");
        }
        progress.send(ProgressEvent::Finished).ok();
        result
    }

    /// Spawns the job on a background thread and returns immediately.
    ///
    /// Returns a [`JoinHandle`] to wait for completion and a [`Receiver`] to
    /// observe [`ProgressEvent`]s. The receiver acts as a natural backpressure
    /// point: the background thread blocks on send if the caller stops draining it.
    ///
    /// This is the preferred entry point for GUI and CLI consumers that need to
    /// remain responsive while images are being processed.
    ///
    /// # Arguments
    /// * `parallelism` - Maximum number of images to analyze concurrently
    ///
    /// # Example
    /// ```no_run
    /// use evanalyzer_core::{generate_job_from_project_settings, ProgressEvent};
    ///
    /// let job = generate_job_from_project_settings(&config)?;
    /// let (handle, rx) = job.run_async(4);
    ///
    /// for event in rx {
    ///     match event {
    ///         ProgressEvent::Started { total } => println!("Processing {total} images"),
    ///         ProgressEvent::ImageCompleted { index, total, path } => {
    ///             println!("[{index}/{total}] {}", path.display());
    ///         }
    ///         ProgressEvent::ImageFailed { path } => {
    ///             eprintln!("Failed: {}", path.display());
    ///         }
    ///         ProgressEvent::Finished => break,
    ///         _ => {}
    ///     }
    /// }
    ///
    /// handle.join().unwrap()?;
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn run_async(
        self,
        parallelism: usize,
    ) -> (
        JoinHandle<Result<(), InternalErrors>>,
        Receiver<ProgressEvent>,
        Arc<AtomicBool>,
    ) {
        let (tx, rx) = std::sync::mpsc::channel();
        let cancel = Arc::new(AtomicBool::new(false));
        let cancel_clone = Arc::clone(&cancel);
        let handle = std::thread::spawn(move || self.run(parallelism, tx, cancel_clone));
        (handle, rx, cancel)
    }

    /// Analyzes one whole image, including all configured time and z-stacks,
    /// splitting it into tiles and running every tile's full pipeline chain
    /// through rayon's `par_iter`.
    ///
    /// Tile processing **always** runs in parallel, regardless of `progress` -
    /// it shares the same rayon pool that [`run`](Self::run) uses to
    /// parallelize over images (nested `par_iter` calls share one pool via
    /// work stealing: when there are fewer images than cores, the otherwise-idle
    /// cores pick up tile work for the images that *are* running, instead of
    /// sitting idle). There is no "single-threaded tiles" mode - what differs
    /// between the two calling contexts below is only which optional,
    /// UI-facing behaviors are active, not whether tiles run in parallel.
    ///
    /// `progress` distinguishes those two contexts:
    /// - `None` - a plain batch run ([`run`](Self::run)'s multi-image branch,
    ///   which already reports progress per whole image once this call
    ///   returns). Tiles use the full analysis size ([`MAX_TILE_SIZE`]), in
    ///   one single pass, with no breakpoint/preview events sent.
    /// - `Some(sender)` - an interactive single-image run ([`run`](Self::run)'s
    ///   single-image branch). Additionally: tiles are sized for live preview
    ///   (see [`preview_tile_size`](Self::preview_tile_size)), the viewport's
    ///   visible tiles are processed before the rest so first results arrive
    ///   fast, and per-tile [`ProgressEvent::TileCompleted`]/[`ProgressEvent::BreakpointReached`]
    ///   events are sent on `sender`.
    ///
    /// # Where "every tile is done" actually is
    /// The tile loop below (`first_pass`/`second_pass`) is the only place
    /// tiles are processed; `tile_result` holds its combined outcome, and by
    /// the time `writer_handle.join()` returns, every tile's pipeline output
    /// has been handed to `exporter`. The line below marked
    /// `// -- ALL TILES FINISHED --` is the single point (reached by both
    /// calling contexts) where every tile of *this* image is guaranteed
    /// done and exported - the place to add anything that must run exactly
    /// once per image after all its tiles finish (e.g. merging objects that
    /// spanned a tile boundary), without duplicating that logic for both
    /// contexts.
    fn analyze_image(
        &self,
        image_rel_path: &PathBuf,
        image_path: &PathBuf,
        image_entry: &ImageEntry,
        order: &[PipelineId],
        exporter: Arc<dyn PipelineResultExporter>,
        cancel: Arc<AtomicBool>,
        progress: Option<Sender<ProgressEvent>>,
        image_cache_bytes: u64,
    ) -> Result<(), InternalErrors> {
        let start_image = Instant::now();
        const RES_IDX: i32 = 0;
        let tile_size = match &progress {
            Some(_) => self.preview_tile_size(),
            None => MAX_TILE_SIZE,
        };

        // One reader for the whole image: it is thread-safe, every tile
        // below reads through it in parallel.
        let start = Instant::now();
        let reader = ImageReader::new(image_path, ReadMode::Default)?;
        log::debug!(
            "Prepare image reader {:?} {:?}",
            image_rel_path,
            start.elapsed()
        );

        let (
            full_size,
            z_proj,
            z_handling,
            z_range,
            t_stacks,
            is_rgb,
            nr_bits,
            pixel_sizes,
            nr_c_stacks,
            nr_z_stacks,
            nr_t_stacks,
            frame_interval_s,
        ) = {
            let series_info = reader
                .image_meta
                .series
                .get(&self.active_series(image_entry))
                .ok_or_else(|| InternalErrors::ImageReadError("Series not found".into()))?;

            let py_meta = series_info
                .resolutions
                .get(&RES_IDX)
                .ok_or_else(|| InternalErrors::ImageReadError("Resolution not found".into()))?;

            let full_size = ImageSize {
                width: py_meta.width as usize,
                height: py_meta.height as usize,
            };
            let (z_proj, z_handling, z_range) =
                self.prepare_z_stack_iterator(series_info, image_entry);
            let t_stacks: Vec<i32> = self
                .prepare_t_stack_iterator(series_info, image_entry)
                .collect();
            let pixel_sizes = match &self.override_pixel_sizes {
                Some(from_user) => from_user.clone(),
                None => PixelSizes {
                    px_size_x: series_info.pixel_sizes.px_size_x,
                    px_size_y: series_info.pixel_sizes.px_size_y,
                    px_size_z: series_info.pixel_sizes.px_size_z,
                },
            };
            (
                full_size,
                z_proj,
                z_handling,
                z_range,
                t_stacks,
                py_meta.is_rgb,
                py_meta.nr_bits,
                pixel_sizes,
                series_info.nr_c_stacks.max(0) as u32,
                series_info.nr_z_stacks.max(0) as u32,
                series_info.nr_t_stacks.max(0) as u32,
                // The project's frame interval, else the image's own.
                self.global_image_settings
                    .t_stack
                    .as_ref()
                    .and_then(|t| t.frame_interval)
                    .filter(|s| s.is_finite() && *s > 0.0)
                    .or(series_info
                        .frame_interval
                        .filter(|s| s.is_finite() && *s > 0.0))
                    .map(f64::from),
            )
        };

        let tiles: Vec<ImageTile> = self
            .prepare_tile_iterator(full_size.width, full_size.height, tile_size)
            .collect();
        let z_stacks: Vec<i32> = z_range.collect();

        // Load Image Channel steps with their own projection or z-plane: read
        // beside the channels (`add_channel_variants`), their z-plane checked
        // before any tile is read.
        let channel_variants = self.channel_variants();
        check_channel_variant_planes(&channel_variants, nr_z_stacks)?;
        if !channel_variants.is_empty() && z_stacks.len() > 1 {
            warn!(
                "{image_rel_path:?}: a Load Image Channel step sets its own z-projection, but the \
                 project analyzes {} z-planes one by one - its pipeline runs once per plane, so \
                 its objects repeat for every plane",
                z_stacks.len()
            );
        }

        // Pre-select the breakpoint target tile before building work items so
        // we can mark exactly one tile as the event sender deterministically.
        // Only meaningful when `progress.is_some()` (interactive run); in a
        // batch run `self.breakpoint` is never set, so this is always `None`
        // there.
        //
        // Strategy: among visible tiles, pick the one whose centre is closest
        // to the viewport centre (in image-pixel space).  When no viewport
        // settings are available, fall back to the geometrically middle tile.
        let breakpoint_targets: Vec<(usize, usize)> = if self.breakpoint.is_some() {
            Self::breakpoint_target_tiles(
                &tiles,
                self.preview_tile_settings.as_ref(),
                self.breakpoint_tiles,
            )
        } else {
            Vec::new()
        };

        // Visible/hidden only depends on tile geometry (x, y), never on which
        // (t, z) stack a tile belongs to - compute the split once and reuse it
        // for every (t, z) group below, so the viewport area is processed
        // first for every plane, not just the first one processed.
        let (visible_tiles, hidden_tiles): (Vec<ImageTile>, Vec<ImageTile>) =
            match &self.preview_tile_settings {
                Some(settings) => {
                    let (visible, hidden): (Vec<_>, Vec<_>) = tiles
                        .iter()
                        .copied()
                        .partition(|t| settings.is_tile_visible(t));
                    let hidden = if settings.process_all_tiles {
                        hidden
                    } else {
                        vec![]
                    };
                    (visible, hidden)
                }
                None => (tiles.clone(), vec![]),
            };

        // One `WholeImagePhaseCompleted` unit per whole-image-scoped pipeline
        // that actually has commands to run (TileMerge, plus any user
        // pipeline's WholeImage-scoped steps, e.g. Voronoi or
        // Colocalization), plus one more for the final export - matches
        // exactly what the loop below ticks off, so a long-running whole-image
        // command (Voronoi across many objects, say) shows up as one of
        // *several* visible steps instead of the bar freezing on a single
        // step for however long the whole phase takes. `.max(1)`: even if
        // every whole-image pipeline is empty, the "object cache empty"
        // early-exit path (see below) still needs its own single reserved
        // unit.
        let whole_image_units_per_stack = self.count_whole_image_progress_units(order);

        // Recalculate so progress events reflect only the tiles actually being processed.
        let total_tiles =
            (visible_tiles.len() + hidden_tiles.len()) * z_stacks.len() * t_stacks.len()
                + whole_image_units_per_stack * z_stacks.len() * t_stacks.len();
        let completed = Arc::new(AtomicUsize::new(0));
        if let Some(progress) = &progress {
            progress
                .send(ProgressEvent::TilesScheduled { total_tiles })
                .ok();
        }

        // Images from different t/z stacks are never used together, so each
        // (t, z) combination gets its own `global_cache`, merged from that
        // stack's tiles alone, and runs its own whole-image phase and export -
        // never mixed with another stack's objects.
        let mut analyze_result: Result<(), InternalErrors> = Ok(());

        // `image_entry` is the same value for every tile/stack of this image -
        // built once here and `Arc::clone()`d (a refcount bump) per tile
        // below, instead of the tile closure itself doing `Arc::new(image_entry.clone())`
        // (a full deep clone, including its `BTreeMap<i32, SeriesSettings>`)
        // on every single tile.
        let image_entry_arc = Arc::new(image_entry.clone());

        // The user's hand-annotated objects of this image, for
        // `LoadAnnotatedObjects` - only gathered when a pipeline uses them.
        // A measurement step measures every image, also one without objects.
        let measures = self
            .pipelines_post_process
            .values()
            .any(|p| p.commands.iter().any(|(_, _, c)| c.measures()));
        let loads_annotations = self.pipelines_post_process.values().any(|p| {
            p.commands
                .iter()
                .any(|(_, _, c)| c.uses_annotated_objects())
        });
        let series_annotations: &[ObjectMetricSettings] = match loads_annotations {
            true => image_entry
                .series
                .get(&self.active_series(image_entry))
                .map(|series| series.objects.as_slice())
                .unwrap_or(&[]),
            false => &[],
        };
        // With a z-projection the stacks loop sees one projected image, not
        // individual z-planes, so annotations then match on t alone.
        let per_z_plane = matches!(
            z_handling,
            ZStackHandling::AllStacks | ZStackHandling::SingleStack
        );

        // "Exclude Image": the planes that excluded the image, with why.
        let mut exclusions: Vec<(i32, i32, String)> = Vec::new();
        'stacks: for &t in &t_stacks {
            // A channel variant is the same image on every z-plane of this
            // t: with several planes, read once per tile and reused.
            let variant_store: VariantStore = Mutex::new(HashMap::new());
            let reuse_variants = (z_stacks.len() > 1).then_some(&variant_store);
            for &z in &z_stacks {
                let plane_annotations: Arc<Vec<ObjectMetricSettings>> = Arc::new(
                    series_annotations
                        .iter()
                        .filter(|a| a.t_stack == t && (!per_z_plane || a.z_stack == z))
                        .cloned()
                        .collect(),
                );
                let global_cache = match self.prepare_global_image_cache(
                    full_size,
                    is_rgb,
                    image_rel_path,
                    nr_bits,
                    pixel_sizes.clone(),
                    image_cache_bytes,
                ) {
                    Ok(cache) => cache,
                    Err(e) => {
                        analyze_result = Err(e);
                        break 'stacks;
                    }
                };

                let z_range_in = matches!(
                    z_handling,
                    ZStackHandling::AllStacks | ZStackHandling::SingleStack
                )
                .then(|| z..=z);

                // Processes one tile against its own copy of `global_cache` and
                // returns it - `try_reduce` below folds every tile's copy back
                // into one cache for this (t, z) stack (rayon's `try_for_each`
                // can't be used here since it discards each call's output).
                let run_tile = |tile: ImageTile,
                                sender: Option<Sender<ProgressEvent>>|
                 -> Result<GlobalPipelineCache, InternalErrors> {
                    if cancel.load(Ordering::Relaxed) {
                        return Err(InternalErrors::Cancelled);
                    }

                    let is_bp_target = breakpoint_targets.contains(&(tile.offset_x, tile.offset_y));

                    let tile_start = Instant::now();

                    let mut cache = self.prepare_pipeline_cache(
                        global_cache.clone(),
                        &reader,
                        image_entry_arc.clone(),
                        &tile,
                        t,
                        &z_proj,
                        &z_range_in,
                        RES_IDX,
                    )?;
                    self.add_channel_variants(
                        &mut cache,
                        &reader,
                        &image_entry_arc,
                        &tile,
                        t,
                        z_proj,
                        RES_IDX,
                        &channel_variants,
                        nr_c_stacks,
                        reuse_variants,
                    )?;
                    let read_s = tile_start.elapsed().as_secs_f64();

                    let mut stop_capture: Option<crate::pipeline::pipeline::BreakpointCapture> =
                        None;
                    let mut snapshot_capture: Option<crate::pipeline::pipeline::BreakpointCapture> =
                        None;
                    // The channel the breakpointed pipeline actually started from, so
                    // the UI can look up that channel's histogram/LUT settings
                    // instead of guessing (previously hardcoded to channel 0 on the
                    // GUI side, which showed the wrong - often black, since it read
                    // an unrelated channel's histogram range - image whenever the
                    // pipeline didn't start from channel 0).
                    let mut breakpoint_channel_idx: Option<i32> = None;
                    for pipe_id in order {
                        if stop_capture.is_some() {
                            break;
                        }
                        if let Some(p) = self.pipelines_pre_process.get(pipe_id) {
                            let (bp_step, snapshot_mode) = self
                                .breakpoint
                                .as_ref()
                                .filter(|b| b.pipeline_id == *pipe_id)
                                .map(|b| (Some(b.step_id), b.mode == BreakpointMode::Snapshot))
                                .unwrap_or((None, false));
                            let result = p.run_commands(
                                self.output_path.clone(),
                                Some(tile),
                                cache,
                                bp_step,
                                snapshot_mode,
                            )?;
                            if result.breakpoint_hit {
                                stop_capture = result.breakpoint_capture;
                                if let ImageAddress::Channel(idx) = p.settings.start_image {
                                    breakpoint_channel_idx = Some(idx);
                                }
                            } else if let Some(capture) = result.breakpoint_capture {
                                snapshot_capture = Some(capture);
                                if let ImageAddress::Channel(idx) = p.settings.start_image {
                                    breakpoint_channel_idx = Some(idx);
                                }
                            }
                            cache = result.cache;
                        }
                    }
                    // Every pipeline ran on this tile: its shared slots are done.
                    cache.clear_shared_slots();
                    debug!(
                        "tile ({}, {}) {}x{}: read {read_s:.2} s, all {:.2} s",
                        tile.offset_x,
                        tile.offset_y,
                        tile.width,
                        tile.height,
                        tile_start.elapsed().as_secs_f64()
                    );

                    // Snapshot: send the captured buffers but continue merging normally.
                    if let Some(capture) = snapshot_capture {
                        if is_bp_target && let Some(sender) = &sender {
                            sender
                                .send(ProgressEvent::BreakpointReached {
                                    image: (*capture.image).clone(),
                                    segmentation: capture.segmentation.map(|s| (*s).clone()),
                                    instances: capture.instances.map(|i| (*i).clone()),
                                    tile_offset_x: tile.offset_x,
                                    tile_offset_y: tile.offset_y,
                                    tile_width: tile.width,
                                    tile_height: tile.height,
                                    nr_bits,
                                    channel_idx: breakpoint_channel_idx,
                                })
                                .ok();
                        }
                        // fall through — the full pipeline ran, merge results normally.
                    }

                    // Stop: send the buffers, discard this tile's (partial) output
                    // instead of merging it in - an empty copy of `global_cache`
                    // contributes nothing when folded together below.
                    if let Some(capture) = stop_capture {
                        if is_bp_target && let Some(sender) = &sender {
                            sender
                                .send(ProgressEvent::BreakpointReached {
                                    image: (*capture.image).clone(),
                                    segmentation: capture.segmentation.map(|s| (*s).clone()),
                                    instances: capture.instances.map(|i| (*i).clone()),
                                    tile_offset_x: tile.offset_x,
                                    tile_offset_y: tile.offset_y,
                                    tile_width: tile.width,
                                    tile_height: tile.height,
                                    nr_bits,
                                    channel_idx: breakpoint_channel_idx,
                                })
                                .ok();
                        }
                        return Ok(global_cache.clone());
                    }

                    // Only interactive runs consume per-tile objects (for the
                    // TileCompleted event below) - skip the clone/collect otherwise.
                    let tile_objects: Vec<ObjectMetricSettings> = if sender.is_some() {
                        cache
                            .object_cache
                            .values()
                            // Modules' internal objects aren't shown.
                            .filter(|r| r.scope() == Scope::Global)
                            .map(|r| r.to_object_settings())
                            .collect()
                    } else {
                        Vec::new()
                    };

                    if let Some(sender) = &sender {
                        let tile_index = completed.fetch_add(1, Ordering::Relaxed) + 1;
                        sender
                            .send(ProgressEvent::TileCompleted {
                                tile_index,
                                total_tiles,
                                objects: tile_objects,
                            })
                            .ok();
                    }

                    Ok(cache)
                };

                // Merge cache clouser
                let merge_caches = |mut a: GlobalPipelineCache,
                                    b: GlobalPipelineCache|
                 -> Result<GlobalPipelineCache, InternalErrors> {
                    a.object_cache.extend(b.object_cache);
                    a.image_cache.extend(b.image_cache);
                    Ok(a)
                };

                // Execute the tiles in prallel
                let tiles_start = Instant::now();
                let stack_result = visible_tiles
                    .par_iter()
                    .map(|&tile| run_tile(tile, progress.clone()))
                    .try_reduce(|| global_cache.clone(), merge_caches)
                    .and_then(|acc| {
                        if hidden_tiles.is_empty() {
                            return Ok(acc);
                        }
                        hidden_tiles
                            .par_iter()
                            .map(|&tile| run_tile(tile, progress.clone()))
                            .try_reduce(|| acc.clone(), merge_caches)
                    });

                debug!(
                    "{image_rel_path:?} t={t} z={z}: {} tiles in {:.2} s",
                    visible_tiles.len() + hidden_tiles.len(),
                    tiles_start.elapsed().as_secs_f64()
                );
                let stack_merge_result = stack_result.and_then(|mut merged_cache| {
                    merged_cache.annotated_objects = plane_annotations;
                    merged_cache.measurements.z_stack = z;
                    merged_cache.measurements.t_stack = t;
                    if merged_cache.object_cache.is_empty()
                        && merged_cache.annotated_objects.is_empty()
                        && !measures
                    {
                        // Object buffer is empty - no whole-image-scoped
                        // commands or export to run, but this stack's
                        // reserved progress unit (see `total_tiles` above)
                        // still needs to be accounted for, or the bar would
                        // sit permanently one unit short of 100%.
                        if let Some(sender) = &progress {
                            let idx = completed.fetch_add(1, Ordering::Relaxed) + 1;
                            sender
                                .send(ProgressEvent::WholeImagePhaseCompleted {
                                    completed: idx,
                                    total_tiles,
                                })
                                .ok();
                        }
                        return Ok(());
                    }

                    // Run every pipeline's whole-image-scoped commands. `order`
                    // (from `get_execution_order`) already guarantees TileMerge
                    // comes first.
                    for pipe_id in order {
                        let Some(p) = self.pipelines_post_process.get(pipe_id) else {
                            continue;
                        };

                        if p.commands.is_empty() {
                            continue;
                        }
                        let (bp_step, snapshot_mode) = self
                            .breakpoint
                            .as_ref()
                            .filter(|b| b.pipeline_id == *pipe_id)
                            .map(|b| (Some(b.step_id), b.mode == BreakpointMode::Snapshot))
                            .unwrap_or((None, false));
                        let pipeline_start = Instant::now();
                        let result = p.run_commands(
                            self.output_path.clone(),
                            None,
                            merged_cache,
                            bp_step,
                            snapshot_mode,
                        )?;
                        merged_cache = result.cache;
                        debug!(
                            "{image_rel_path:?} t={t} z={z}: whole-image pipeline {pipe_id} in {:.2} s",
                            pipeline_start.elapsed().as_secs_f64()
                        );

                        // One reserved unit per non-empty whole-image
                        // pipeline (see `whole_image_units_per_stack`) - a
                        // long-running command here (Voronoi, say) now shows
                        // up as one of several visible progress steps
                        // instead of the bar freezing on a single step.
                        if let Some(sender) = &progress {
                            let idx = completed.fetch_add(1, Ordering::Relaxed) + 1;
                            sender
                                .send(ProgressEvent::WholeImagePhaseCompleted {
                                    completed: idx,
                                    total_tiles,
                                })
                                .ok();
                        }

                        if result.breakpoint_hit {

                            info!(
                                "Whole-image breakpoint hit for pipeline {} on image {:?} (t={}, z={})",
                                pipe_id, image_rel_path, t, z
                            );
                            break;
                        }
                    }

                    // Modules' internal objects never reach the results.
                    // `EndScope` after each instance's last command normally
                    // removed them already (with the references to them) -
                    // this catches a run stopped early by a breakpoint.
                    merged_cache
                        .object_cache
                        .retain(|_, object| object.scope() == Scope::Global);
                    let export_start = Instant::now();
                    exporter.export(&merged_cache)?;
                    debug!(
                        "{image_rel_path:?} t={t} z={z}: export of {} objects in {:.2} s",
                        merged_cache.object_cache.len(),
                        export_start.elapsed().as_secs_f64()
                    );
                    if let Some(reason) = &merged_cache.measurements.excluded {
                        exclusions.push((t, z, reason.clone()));
                    }

                    if let Some(sender) = &progress {
                        let idx = completed.fetch_add(1, Ordering::Relaxed) + 1;
                        sender
                            .send(ProgressEvent::WholeImagePhaseCompleted {
                                completed: idx,
                                total_tiles,
                            })
                            .ok();
                    }
                    Ok(())
                });

                if let Err(e) = stack_merge_result {
                    analyze_result = Err(e);
                    break 'stacks;
                }
            }
        }

        // Record the image was processed even if it produced zero objects -
        // run regardless of a partial failure so a partially-failed image
        // still shows up rather than vanishing entirely. The error (if any)
        // is passed through so it's recorded as failed rather than looking
        // identical to a genuinely complete image.
        let combined_error = analyze_result.as_ref().err().map(|e| e.to_string());
        // The planes the loop above ran (a projection's one z) are recorded
        // as analysed - also where nothing was found.
        let finalize_result = exporter.finalize_image(
            image_rel_path,
            &ImageSummary {
                width: full_size.width as u32,
                height: full_size.height as u32,
                nr_c_stacks,
                nr_z_stacks,
                nr_t_stacks,
                t_stacks_analyzed: &t_stacks,
                z_stacks_analyzed: &z_stacks,
                frame_interval_s,
                excluded: exclusion_reason(&exclusions, t_stacks.len() * z_stacks.len()).as_deref(),
            },
            combined_error.as_deref(),
        );

        let duration = start_image.elapsed();
        log::debug!("Executed image pipeline in {:?}", duration);

        analyze_result.and(finalize_result)
    }

    /// Generates an iterator over image tiles for processing large images.
    ///
    /// This method divides a full-sized image into smaller, manageable tiles based on the
    /// specified tile size. It's particularly useful for processing large whole-slide images
    /// that cannot fit entirely in memory. Each tile is positioned with its offset coordinates
    /// and dimensions, ensuring complete coverage of the full image.
    ///
    /// # Arguments
    /// * `full_width` - The total width of the full image in pixels
    /// * `full_height` - The total height of the full image in pixels
    /// * `tile_size` - The desired size of each tile in pixels (e.g., 4096)
    ///
    /// # Returns
    /// An iterator that yields `ImageTile` structs containing offset and dimension information
    /// for each tile in row-major order (left-to-right, top-to-bottom).
    fn prepare_tile_iterator(
        &self,
        full_width: usize,
        full_height: usize,
        tile_size: usize,
    ) -> impl Iterator<Item = ImageTile> {
        let x_steps = full_width.div_ceil(tile_size);
        let y_steps = full_height.div_ceil(tile_size);

        (0..y_steps).flat_map(move |y| {
            (0..x_steps).map(move |x| {
                let offset_x = x * tile_size;
                let offset_y = y * tile_size;

                ImageTile {
                    offset_x,
                    offset_y,
                    width: (full_width - offset_x).min(tile_size),
                    height: (full_height - offset_y).min(tile_size),
                }
            })
        })
    }

    /// Generates a range of time stack indices based on project settings.
    ///
    /// This method determines which time frames to process based on the configured
    /// T-stack handling mode. It can either return a single time frame or a range
    /// covering all available time points in the image.
    ///
    /// Resolves settings as **per-series override -> global setting -> default**
    /// (matching the equivalent resolution the GUI itself uses, in
    /// `evanalyzer_app`'s `project_ext.rs`). A series entry always
    /// exists once an image is added to a project, but its `t_stack` field is only
    /// ever populated by an explicit per-series override - so checking *whether
    /// the field is set* (not whether the entry exists) is what makes the global
    /// setting reachable.
    ///
    /// # Arguments
    /// * `project` - The project containing global and image-specific settings
    /// * `image_info` - Metadata about the image, including the total number of time stacks
    /// * `image_entry` - The specific image entry with its configured T-stack settings
    ///
    /// # Returns
    /// A `RangeInclusive<i32>` representing the time frame indices to process:
    /// - For `SingleStack` mode: returns only the configured time index
    /// - For `AllStacks` mode: returns the range from 0 to `nr_t_stacks - 1`
    fn prepare_t_stack_iterator(
        &self,
        image_info: &ImageInfo,
        image_entry: &ImageEntry,
    ) -> RangeInclusive<i32> {
        let local = image_entry
            .series
            .get(&self.active_series(image_entry))
            .and_then(|s| s.t_stack.clone());
        let settings = local
            .or_else(|| self.global_image_settings.t_stack.clone())
            .unwrap_or_default();

        match settings.stack_handling {
            TStackHandling::SingleStack => settings.t_stack..=settings.t_stack,
            TStackHandling::AllStacks => 0..=image_info.nr_t_stacks - 1,
        }
    }

    /// Resolves the active Z-stack settings as **per-series override -> global
    /// setting -> default**. See [`prepare_t_stack_iterator`](Self::prepare_t_stack_iterator)
    /// for why checking field presence (not series-entry presence) matters here:
    /// every image has a series entry with `z_stack: None` unless explicitly
    /// overridden, so the previous entry-existence check meant a project-wide
    /// projection choice (e.g. Maximum Intensity) was silently discarded in favor
    /// of the `SingleStack` default for every image.
    /// The series that counts for `image_entry` - see
    /// [`crate::job::job_generator::active_series`].
    fn active_series(&self, image_entry: &ImageEntry) -> i32 {
        crate::job::job_generator::active_series(
            image_entry,
            self.global_image_settings.selected_series,
        )
    }

    fn get_z_stack_settings(&self, image_entry: &ImageEntry) -> ZStackSettings {
        let local = image_entry
            .series
            .get(&self.active_series(image_entry))
            .and_then(|s| s.z_stack.clone());

        local
            .or_else(|| self.global_image_settings.z_stack.clone())
            .unwrap_or_default()
    }

    /// Prepares Z-stack projection settings and generates a range of Z indices.
    ///
    /// This method determines how to handle Z-stack data based on project settings.
    /// It can apply various projection methods (max, min, average, sum intensity, etc.)
    /// or process individual Z-slices. The method returns the projection type, handling mode,
    /// and the range of Z indices to process.
    ///
    /// # Arguments
    /// * `project` - The project containing global and image-specific settings
    /// * `image_info` - Metadata about the image, including the total number of Z-stacks
    /// * `image_entry` - The specific image entry with its configured Z-stack settings
    ///
    /// # Returns
    /// A tuple containing:
    /// - `ZProjection` - The projection method to apply (None, MaxIntensity, MinIntensity, etc.)
    /// - `ZStackHandling` - The handling mode (SingleStack, AllStacks, or a projection type)
    /// - `RangeInclusive<i32>` - The Z indices to process:
    ///   - For projection methods: returns 0..=0 (single projected output)
    ///   - For `SingleStack`: returns the configured Z range
    ///   - For `AllStacks`: returns 0 to `nr_z_stacks - 1`
    fn prepare_z_stack_iterator(
        &self,
        image_info: &ImageInfo,
        image_entry: &ImageEntry,
    ) -> (ZProjection, ZStackHandling, RangeInclusive<i32>) {
        let settings = self.get_z_stack_settings(image_entry);
        let handling = settings.z_projection.clone();

        let (projection, range) = match handling {
            ZStackHandling::SingleStack => {
                (ZProjection::None, settings.z_range.clone().unwrap_or(0..=0))
            }
            ZStackHandling::AllStacks => {
                (ZProjection::None, 0..=(image_info.nr_z_stacks as i32 - 1))
            }
            ZStackHandling::MaxIntensity => (ZProjection::MaxIntensity, 0..=0),
            ZStackHandling::MinIntensity => (ZProjection::MinIntensity, 0..=0),
            ZStackHandling::AvgIntensity => (ZProjection::AvgIntensity, 0..=0),
            ZStackHandling::SumIntensity => (ZProjection::SumIntensity, 0..=0),
            ZStackHandling::TakeTheMiddle => (ZProjection::TakeTheMiddle, 0..=0),
        };

        (projection, handling, range)
    }

    fn prepare_global_image_cache(
        &self,
        full_image_width: ImageSize,
        is_rgb: bool,
        image_rel_path: &PathBuf,
        nr_of_bits: u16,
        pixel_sizes: PixelSizes,
        image_cache_bytes: u64,
    ) -> Result<GlobalPipelineCache, InternalErrors> {
        Ok(GlobalPipelineCache {
            image_cache: ImageCache::with_capacity_bytes(image_cache_bytes)
                .map_err(|e| InternalErrors::Io(format!("Failed to create image cache: {e}")))?,
            image_meta: GlobalImageMeta {
                full_image_width,
                is_rgb,
                nr_of_bits,
                pixel_sizes,
            },
            object_cache: ObjectCache::default(),
            measurements: Default::default(),
            image_rel_path: image_rel_path.into(),
            annotated_objects: Default::default(),
            class_colors: Arc::clone(&self.class_colors),
        })
    }

    /// Prepares the pipeline cache
    ///
    /// Loads the selected image plane from the image and inits the
    /// cache with the loaded image planes and returns the cache.
    /// This cache can now be used for processing the pipelines of the image
    fn prepare_pipeline_cache(
        &self,
        mut global_cache: GlobalPipelineCache,
        image_reader: &ImageReader,
        image_entry: Arc<ImageEntry>,
        image_tile: &ImageTile,
        t_stack: i32,
        z_projection: &ZProjection,
        z_range: &Option<RangeInclusive<i32>>,
        resolution_index: i32,
    ) -> Result<GlobalPipelineCache, InternalErrors> {
        let loaded_channels = image_reader.read_image_tile_combined(
            self.active_series(&image_entry),
            resolution_index,
            z_projection.clone(),
            z_range,
            t_stack,
            None,
            image_tile,
        )?;

        // Get size from the first channel if it exists
        let loaded_size = loaded_channels
            .first()
            .map(|img| img.image.size())
            .unwrap_or(ImageSize {
                width: 0,
                height: 0,
            });

        // Store image meta
        let image_meta = PipelineImageMeta {
            image_tile_info: ImageTile {
                width: loaded_size.width,
                height: loaded_size.height,
                ..*image_tile // Copies offset_x and offset_y from image_tile
            },
            full_image_width: global_cache.image_meta.full_image_width,
            is_rgb: global_cache.image_meta.is_rgb,
            nr_of_bits: global_cache.image_meta.nr_of_bits,
            pixel_sizes: global_cache.image_meta.pixel_sizes.clone(),
        };

        for img in loaded_channels {
            global_cache.image_cache.insert(
                CacheAddress::Channel((img.c_stack, image_meta.image_tile_info.clone())),
                img.image,
            );
        }

        Ok(global_cache)
    }

    /// The channels the pipelines read with their own projection or z-plane
    /// (Load Image Channel steps not on Auto), each once.
    fn channel_variants(&self) -> Vec<ChannelVariant> {
        let mut variants: Vec<ChannelVariant> = Vec::new();
        for variant in self
            .pipelines_pre_process
            .values()
            .chain(self.pipelines_post_process.values())
            .flat_map(|p| {
                p.commands
                    .iter()
                    .filter_map(|(_, _, c)| c.channel_variant())
            })
        {
            if !variants.contains(&variant) {
                variants.push(variant);
            }
        }
        variants
    }

    /// Reads `variants` - channels with their own projection or z-plane (Load
    /// Image Channel steps not on Auto) - for `tile` into `cache`, beside the
    /// channels [`Self::prepare_pipeline_cache`] read. One with the projection
    /// those channels already have is theirs, not read again. With `reuse`
    /// (several z-planes of one t) each is read once per tile, the planes
    /// after share it. A channel the image doesn't have is skipped: the step
    /// reports it.
    #[allow(clippy::too_many_arguments)]
    fn add_channel_variants(
        &self,
        cache: &mut GlobalPipelineCache,
        reader: &ImageReader,
        image_entry: &ImageEntry,
        tile: &ImageTile,
        t_stack: i32,
        z_projection: ZProjection,
        resolution_index: i32,
        variants: &[ChannelVariant],
        nr_c_stacks: u32,
        reuse: Option<&VariantStore>,
    ) -> Result<(), InternalErrors> {
        for &variant in variants {
            if variant.channel < 0 || variant.channel as u32 >= nr_c_stacks {
                continue;
            }
            if variant.projection != ZProjection::None && variant.projection == z_projection {
                let read = cache.channel_keys().find(|(c, t)| {
                    *c == variant.channel
                        && t.offset_x == tile.offset_x
                        && t.offset_y == tile.offset_y
                });
                if let Some((channel, loaded_tile)) = read
                    && let Some(image) = cache
                        .image_cache
                        .get(&CacheAddress::Channel((channel, loaded_tile)))
                {
                    cache
                        .image_cache
                        .insert(CacheAddress::ChannelVariant((variant, loaded_tile)), image);
                    continue;
                }
            }

            let key = (variant, tile.offset_x, tile.offset_y);
            let stored = reuse.and_then(|store| store.lock().expect("Poisoned").get(&key).cloned());
            let image = match stored {
                Some(image) => image,
                None => {
                    let z_range =
                        (variant.projection == ZProjection::None).then(|| variant.z..=variant.z);
                    let Some(read) = reader
                        .read_image_tile_combined(
                            self.active_series(image_entry),
                            resolution_index,
                            variant.projection,
                            &z_range,
                            t_stack,
                            Some(&vec![variant.channel]),
                            tile,
                        )?
                        .into_iter()
                        .next()
                    else {
                        continue;
                    };
                    if let Some(store) = reuse {
                        store
                            .lock()
                            .expect("Poisoned")
                            .insert(key, read.image.clone());
                    }
                    read.image
                }
            };
            let size = image.size();
            let loaded_tile = ImageTile {
                width: size.width,
                height: size.height,
                ..*tile
            };
            cache
                .image_cache
                .insert(CacheAddress::ChannelVariant((variant, loaded_tile)), image);
        }
        Ok(())
    }

    /// Pure tile-count math, factored out of [`count_preview_visible_tiles`] so it
    /// can be unit-tested without needing a real image file on disk.
    fn count_visible_tiles_for_image(&self, full_width: usize, full_height: usize) -> usize {
        let tile_size = self.preview_tile_size();
        let tiles = self.prepare_tile_iterator(full_width, full_height, tile_size);
        match &self.preview_tile_settings {
            Some(settings) if !settings.process_all_tiles => {
                tiles.filter(|t| settings.is_tile_visible(t)).count()
            }
            _ => tiles.count(),
        }
    }

    /// Counts how many tiles a preview run would actually process, without running
    /// the pipeline. Mirrors the tile-size and viewport-visibility logic
    /// `analyze_image`'s interactive calling context uses (same `preview_tile_size`/
    /// `prepare_tile_iterator`/`is_tile_visible` calls), so callers can gate on the
    /// same number the real run would use before paying for a full pipeline dispatch.
    ///
    /// Only reads image metadata (dimensions), not pixel data, so this is cheap
    /// enough to call synchronously before starting a preview.
    pub fn count_preview_visible_tiles(&self) -> Result<usize, InternalErrors> {
        const RES_IDX: i32 = 0;
        let mut total = 0usize;
        for (rel_path, image_entry) in &self.images {
            let abs_path = self.image_base_path.join(rel_path);
            let reader = ImageReader::new(&abs_path, ReadMode::Default)?;
            let series_info = reader
                .image_meta
                .series
                .get(&self.active_series(image_entry))
                .ok_or_else(|| InternalErrors::ImageReadError("Series not found".into()))?;
            let py_meta = series_info
                .resolutions
                .get(&RES_IDX)
                .ok_or_else(|| InternalErrors::ImageReadError("Resolution not found".into()))?;

            total +=
                self.count_visible_tiles_for_image(py_meta.width as usize, py_meta.height as usize);
        }
        Ok(total)
    }

    /// Determines the correct order to run pipelines based on dependencies
    fn get_execution_order(&self) -> Vec<PipelineId> {
        let mut order = Vec::new();
        let mut visited = HashSet::new();
        let mut temp_visited = HashSet::new();

        fn visit(
            name: &PipelineId,
            pre_pipelines: &IndexMap<PipelineId, Pipeline>,
            post_pipelines: &IndexMap<PipelineId, Pipeline>,
            visited: &mut HashSet<PipelineId>,
            temp_visited: &mut HashSet<PipelineId>,
            order: &mut Vec<PipelineId>,
        ) {
            if temp_visited.contains(name) {
                panic!("Circular dependency detected!");
            }
            if !visited.contains(name) {
                temp_visited.insert(name.clone());
                if let Some(p) = pre_pipelines.get(name).or_else(|| post_pipelines.get(name)) {
                    for dep in &p.dependencies {
                        visit(
                            dep,
                            pre_pipelines,
                            post_pipelines,
                            visited,
                            temp_visited,
                            order,
                        );
                    }
                }
                temp_visited.remove(name);
                visited.insert(name.clone());
                order.push(name.clone());
            }
        }

        // Both maps share one dependency graph - a post-process pipeline can
        // depend on a pre-process one finishing first (and vice versa isn't
        // ruled out by the type system either), so the order must be computed
        // over their union, not just `pipelines_post_process` alone.
        for name in self
            .pipelines_pre_process
            .keys()
            .chain(self.pipelines_post_process.keys())
        {
            visit(
                name,
                &self.pipelines_pre_process,
                &self.pipelines_post_process,
                &mut visited,
                &mut temp_visited,
                &mut order,
            );
        }

        // TileMerge must run before any other whole-image command
        if let Some(pos) = order.iter().position(|id| *id == TILE_MERGE_PIPELINE_ID) {
            let tile_merge_id = order.remove(pos);
            order.insert(0, tile_merge_id);
        }

        order
    }

    /// Number of `ProgressEvent::WholeImagePhaseCompleted` units one (t, z)
    /// stack's whole-image phase will emit: one per pipeline in `order`
    /// that's registered in `pipelines_post_process` and has at least one
    /// command, plus one more for the final export - matching exactly what
    /// `analyze_image`'s whole-image loop ticks off, so a progress bar
    /// driven by these events reaches 100% exactly when that phase actually
    /// finishes. `.max(1)`: even when every whole-image pipeline is empty,
    /// the "object cache empty" early-exit path still emits its own single
    /// unit.
    fn count_whole_image_progress_units(&self, order: &[PipelineId]) -> usize {
        order
            .iter()
            .filter(|&pipe_id| {
                self.pipelines_post_process
                    .get(pipe_id)
                    .is_some_and(|p| !p.commands.is_empty())
            })
            .count()
            .saturating_add(1)
            .max(1)
    }

    /// Picks the tile size for a single-image preview run.
    ///
    /// At higher zoom the viewport covers a smaller area of the full image, so
    /// using the same fixed 4096px tile means re-reading and re-analyzing a lot
    /// of pixels the user can't see just to get the small visible patch. Shrinking
    /// the tile as zoom increases keeps the analyzed area closer to what's on
    /// screen, so feedback for that area arrives faster.
    /// The tiles a breakpoint captures (their offsets), at most `count`
    /// (at least 1): among the tiles visible in the viewport (all tiles if
    /// none is), the ones whose centres are closest to the viewport centre;
    /// without viewport settings the middle tiles.
    fn breakpoint_target_tiles(
        tiles: &[ImageTile],
        settings: Option<&PreviewTileSettings>,
        count: usize,
    ) -> Vec<(usize, usize)> {
        let count = count.max(1);
        let candidates: Vec<&ImageTile> = match settings {
            Some(settings) => {
                let visible: Vec<&ImageTile> = tiles
                    .iter()
                    .filter(|t| settings.is_tile_visible(t))
                    .collect();
                if visible.is_empty() {
                    tiles.iter().collect()
                } else {
                    visible
                }
            }
            None => tiles.iter().collect(),
        };
        // The centre: the viewport's in image pixels
        // (screen = img * zoom + offset), else the middle tile's.
        let centre = match settings {
            Some(settings) => Some((
                (settings.viewport_width / 2.0 - settings.offset_x) / settings.zoom,
                (settings.viewport_height / 2.0 - settings.offset_y) / settings.zoom,
            )),
            None => tiles.get(tiles.len() / 2).map(|t| {
                (
                    t.offset_x as f32 + t.width as f32 / 2.0,
                    t.offset_y as f32 + t.height as f32 / 2.0,
                )
            }),
        };
        let Some((cx, cy)) = centre else {
            return Vec::new();
        };
        let mut by_distance: Vec<(&ImageTile, f32)> = candidates
            .into_iter()
            .map(|t| {
                let tx = t.offset_x as f32 + t.width as f32 / 2.0;
                let ty = t.offset_y as f32 + t.height as f32 / 2.0;
                (t, (tx - cx).powi(2) + (ty - cy).powi(2))
            })
            .collect();
        // Stable: equally near tiles keep their order.
        by_distance.sort_by(|(_, da), (_, db)| da.total_cmp(db));
        by_distance
            .into_iter()
            .take(count)
            .map(|(t, _)| (t.offset_x, t.offset_y))
            .collect()
    }

    fn preview_tile_size(&self) -> usize {
        // All tiles: as big as they get - fewer seams to merge, and the
        // limit on a whole-image preview doesn't depend on the zoom.
        let Some(zoom) = self
            .preview_tile_settings
            .as_ref()
            .filter(|s| !s.process_all_tiles)
            .map(|s| s.zoom)
        else {
            return MAX_TILE_SIZE;
        };
        if zoom >= 8.0 {
            512
        } else if zoom >= 4.0 {
            1024
        } else if zoom >= 2.0 {
            2048
        } else {
            MAX_TILE_SIZE
        }
    }

    /// Estimates the peak per-worker RAM breakdown for this job, from the
    /// images actually being analyzed - see [`RamBudget`](crate::resources::RamBudget)
    /// for what each component covers, and [`RamBudget::total_bytes`] for
    /// [`recommended_parallelism`](crate::recommended_parallelism)'s flat
    /// input, which either over-commits on a large/many-channel image or
    /// needlessly caps a small one down to a single thread on a low-RAM
    /// machine if left as a flat guess instead.
    ///
    /// Takes the worst case *total* across every image in the job (not
    /// necessarily the image with the largest tile or the most channels
    /// individually - see [`RamBudget::total_bytes`]), since parallelism
    /// (and the resulting image-cache budget, see `run`) are decided once
    /// up front for the whole batch, not per image. Each image's own
    /// dimensions/channel count come from already-scanned metadata
    /// (`ImageEntry`/`SeriesSettings`) - this never reopens a file, so it's
    /// cheap to call before starting a run.
    pub fn estimate_ram_budget(&self) -> crate::utils::resources::RamBudget {
        // A channel read with its own projection is one more channel image.
        let variants = self.channel_variants();
        self.images
            .values()
            .filter_map(|entry| entry.series.get(&self.active_series(entry)))
            .map(|series| {
                let tile_width = (series.image_width as usize).min(MAX_TILE_SIZE);
                let tile_height = (series.image_height as usize).min(MAX_TILE_SIZE);
                crate::utils::resources::estimate_ram_budget(
                    tile_width,
                    tile_height,
                    series.channels.len() + variants.len(),
                )
            })
            .max_by_key(|budget| budget.total_bytes())
            .unwrap_or_else(|| {
                crate::utils::resources::estimate_ram_budget(MAX_TILE_SIZE, MAX_TILE_SIZE, 1)
            })
    }

    /// Flat total of [`estimate_ram_budget`](Self::estimate_ram_budget) -
    /// what [`recommended_parallelism`](crate::recommended_parallelism)
    /// actually wants.
    pub fn estimate_ram_per_worker_bytes(&self) -> u64 {
        self.estimate_ram_budget().total_bytes()
    }
}

/// Channel variants read for one t, by (variant, tile x, tile y) - shared by
/// its z-planes (see `JobExecutor::add_channel_variants`).
type VariantStore = Mutex<HashMap<(ChannelVariant, usize, usize), Arc<ImageContainer>>>;

/// A Load Image Channel step's z-plane the image doesn't have - checked
/// before any tile is read, for a clear message instead of a reader error.
fn check_channel_variant_planes(
    variants: &[ChannelVariant],
    nr_z_stacks: u32,
) -> Result<(), InternalErrors> {
    let planes = nr_z_stacks.max(1) as i32;
    match variants
        .iter()
        .find(|v| v.projection == ZProjection::None && (v.z < 0 || v.z >= planes))
    {
        Some(v) => Err(InternalErrors::InvalidArgument(format!(
            "a Load Image Channel step loads z-plane {} of channel {}, but this image has {} \
             - check the z-plane of the pipeline's Load Image Channel steps",
            v.z,
            v.channel,
            match planes {
                1 => "only z-plane 0".to_string(),
                n => format!("z-planes 0–{}", n - 1),
            }
        ))),
        None => Ok(()),
    }
}

/// Why the image is excluded (`None`: it isn't): the first plane's reason -
/// with its plane and how many more planes, when there are several.
fn exclusion_reason(exclusions: &[(i32, i32, String)], planes: usize) -> Option<String> {
    let (t, z, reason) = exclusions.first()?;
    Some(match (planes > 1, exclusions.len() - 1) {
        (false, _) => reason.clone(),
        (true, 0) => format!("t {t}, z {z}: {reason}"),
        (true, more) => format!("t {t}, z {z}: {reason} (and {more} more planes)"),
    })
}

#[cfg(test)]
mod z_t_stack_precedence_tests {
    use super::*;
    use crate::storage::memory::MemoryExporter;
    use evanalyzer_cfg::settings::images_settings::{SeriesSettings, TStackSettings};
    use std::collections::BTreeMap;
    use std::sync::{Arc, Mutex};

    fn make_job_executor(
        global_z: Option<ZStackSettings>,
        global_t: Option<TStackSettings>,
    ) -> JobExecutor {
        let global_image_settings = GlobalImageSettings {
            z_stack: global_z,
            t_stack: global_t,
            ..Default::default()
        };

        JobExecutor::new(
            std::path::PathBuf::new(),
            std::path::PathBuf::new(),
            IndexMap::new(),
            std::path::PathBuf::new(),
            global_image_settings,
            Arc::new(MemoryExporter {
                out_objects: Arc::new(Mutex::new(Vec::new())),
                ..Default::default()
            }),
            None,
            Default::default(),
        )
    }

    /// Mirrors what `add_image_to_list` actually produces for every image: a
    /// series entry exists, but its `z_stack`/`t_stack` fields are unset unless
    /// the (currently unreachable from the GUI) per-series override is used.
    fn make_image_entry_with_default_series() -> ImageEntry {
        let mut series = BTreeMap::new();
        series.insert(0, SeriesSettings::default());
        ImageEntry {
            rel_path: std::path::PathBuf::new(),
            file_size: 0,
            selected_series: 0,
            series,
        }
    }

    #[test]
    fn global_z_projection_is_honored_when_no_per_series_override_exists() {
        let job = make_job_executor(
            Some(ZStackSettings {
                z_projection: ZStackHandling::MaxIntensity,
                z_range: None,
            }),
            None,
        );
        let entry = make_image_entry_with_default_series();

        let resolved = job.get_z_stack_settings(&entry);

        assert_eq!(resolved.z_projection, ZStackHandling::MaxIntensity);
    }

    #[test]
    fn per_series_z_override_takes_precedence_over_global() {
        let job = make_job_executor(
            Some(ZStackSettings {
                z_projection: ZStackHandling::MaxIntensity,
                z_range: None,
            }),
            None,
        );
        let mut entry = make_image_entry_with_default_series();
        entry.series.get_mut(&0).unwrap().z_stack = Some(ZStackSettings {
            z_projection: ZStackHandling::SumIntensity,
            z_range: None,
        });

        let resolved = job.get_z_stack_settings(&entry);

        assert_eq!(resolved.z_projection, ZStackHandling::SumIntensity);
    }

    #[test]
    fn z_settings_fall_back_to_default_when_neither_local_nor_global_is_set() {
        let job = make_job_executor(None, None);
        let entry = make_image_entry_with_default_series();

        let resolved = job.get_z_stack_settings(&entry);

        assert_eq!(resolved.z_projection, ZStackHandling::SingleStack);
    }

    #[test]
    fn global_t_all_stacks_is_honored_when_no_per_series_override_exists() {
        let job = make_job_executor(
            None,
            Some(TStackSettings {
                stack_handling: TStackHandling::AllStacks,
                playback_speed: 1.0,
                t_stack: 0,
                frame_interval: None,
            }),
        );
        let entry = make_image_entry_with_default_series();
        let image_info = ImageInfo {
            nr_t_stacks: 5,
            ..Default::default()
        };

        let range = job.prepare_t_stack_iterator(&image_info, &entry);

        assert_eq!(range, 0..=4);
    }

    #[test]
    fn per_series_t_override_takes_precedence_over_global() {
        let job = make_job_executor(
            None,
            Some(TStackSettings {
                stack_handling: TStackHandling::AllStacks,
                playback_speed: 1.0,
                t_stack: 0,
                frame_interval: None,
            }),
        );
        let mut entry = make_image_entry_with_default_series();
        entry.series.get_mut(&0).unwrap().t_stack = Some(TStackSettings {
            stack_handling: TStackHandling::SingleStack,
            playback_speed: 1.0,
            t_stack: 3,
            frame_interval: None,
        });
        let image_info = ImageInfo {
            nr_t_stacks: 5,
            ..Default::default()
        };

        let range = job.prepare_t_stack_iterator(&image_info, &entry);

        assert_eq!(range, 3..=3);
    }

    #[test]
    fn t_all_stacks_with_single_frame_image_yields_single_element_range() {
        let job = make_job_executor(
            None,
            Some(TStackSettings {
                stack_handling: TStackHandling::AllStacks,
                playback_speed: 1.0,
                t_stack: 0,
                frame_interval: None,
            }),
        );
        let entry = make_image_entry_with_default_series();
        let image_info = ImageInfo {
            nr_t_stacks: 1,
            ..Default::default()
        };

        let range = job.prepare_t_stack_iterator(&image_info, &entry);

        assert_eq!(range, 0..=0);
    }

    #[test]
    fn t_single_stack_ignores_total_frame_count() {
        let job = make_job_executor(
            None,
            Some(TStackSettings {
                stack_handling: TStackHandling::SingleStack,
                playback_speed: 1.0,
                t_stack: 7,
                frame_interval: None,
            }),
        );
        let entry = make_image_entry_with_default_series();
        let image_info = ImageInfo {
            nr_t_stacks: 100,
            ..Default::default()
        };

        let range = job.prepare_t_stack_iterator(&image_info, &entry);

        assert_eq!(range, 7..=7);
    }

    #[test]
    fn missing_series_entry_falls_back_to_global_z_setting() {
        let job = make_job_executor(
            Some(ZStackSettings {
                z_projection: ZStackHandling::MaxIntensity,
                z_range: None,
            }),
            None,
        );
        let entry = ImageEntry {
            rel_path: std::path::PathBuf::new(),
            file_size: 0,
            selected_series: 0,
            series: BTreeMap::new(),
        };

        let resolved = job.get_z_stack_settings(&entry);

        assert_eq!(resolved.z_projection, ZStackHandling::MaxIntensity);
    }

    #[test]
    fn missing_series_entry_falls_back_to_global_t_setting() {
        let job = make_job_executor(
            None,
            Some(TStackSettings {
                stack_handling: TStackHandling::AllStacks,
                playback_speed: 1.0,
                t_stack: 0,
                frame_interval: None,
            }),
        );
        let entry = ImageEntry {
            rel_path: std::path::PathBuf::new(),
            file_size: 0,
            selected_series: 0,
            series: BTreeMap::new(),
        };
        let image_info = ImageInfo {
            nr_t_stacks: 4,
            ..Default::default()
        };

        let range = job.prepare_t_stack_iterator(&image_info, &entry);

        assert_eq!(range, 0..=3);
    }
}

#[cfg(test)]
mod preview_visible_tile_count_tests {
    use super::*;
    use crate::storage::memory::MemoryExporter;
    use std::sync::{Arc, Mutex};

    fn make_job(preview_tile_settings: Option<PreviewTileSettings>) -> JobExecutor {
        let mut job = JobExecutor::new(
            std::path::PathBuf::new(),
            std::path::PathBuf::new(),
            IndexMap::new(),
            std::path::PathBuf::new(),
            GlobalImageSettings::default(),
            Arc::new(MemoryExporter {
                out_objects: Arc::new(Mutex::new(Vec::new())),
                ..Default::default()
            }),
            None,
            Default::default(),
        );
        job.preview_tile_settings = preview_tile_settings;
        job
    }

    /// No preview settings means a full (non-preview) run: every tile counts.
    #[test]
    fn without_preview_settings_counts_every_tile() {
        let job = make_job(None);
        // 4096px base tile size, 10000x10000 image -> 3x3 = 9 tiles.
        assert_eq!(job.count_visible_tiles_for_image(10_000, 10_000), 9);
    }

    /// Zoomed out so far that the whole 10000x10000 image fits the viewport: every
    /// tile in the (fixed 4096px, since zoom < 2.0) grid intersects the viewport.
    #[test]
    fn zoomed_out_to_fit_the_whole_slide_counts_every_tile() {
        let job = make_job(Some(PreviewTileSettings {
            offset_x: 0.0,
            offset_y: 0.0,
            viewport_width: 1000.0,
            viewport_height: 1000.0,
            zoom: 0.1,
            process_all_tiles: false,
        }));
        assert_eq!(job.count_visible_tiles_for_image(10_000, 10_000), 9);
    }

    /// Zoomed in enough that the viewport only covers a small image-space patch:
    /// only the tiles actually under the viewport should count, not the whole grid.
    #[test]
    fn zoomed_in_on_one_tile_counts_only_that_tile() {
        let job = make_job(Some(PreviewTileSettings {
            offset_x: 0.0,
            offset_y: 0.0,
            viewport_width: 1000.0,
            viewport_height: 1000.0,
            // zoom >= 2.0 shrinks the tile size to 2048px (see `preview_tile_size`),
            // and the viewport (1000px screen / 2.0 zoom = 500 image px) sits
            // entirely inside the first 2048px tile.
            zoom: 2.0,
            process_all_tiles: false,
        }));
        assert_eq!(job.count_visible_tiles_for_image(10_000, 10_000), 1);
    }

    /// Sanity check for the gate's exact use case: a whole-slide image (e.g.
    /// 40000x30000, comparable to a real WSI) viewed fully zoomed out covers far
    /// more than a handful of 4096px tiles - this is the scenario the GUI's
    /// preview-rejection threshold (`MAX_PREVIEW_VISIBLE_TILES` in
    /// `pipeline_worker.rs`) is meant to catch.
    #[test]
    fn whole_slide_zoomed_out_exceeds_a_small_tile_budget() {
        let job = make_job(Some(PreviewTileSettings {
            offset_x: 0.0,
            offset_y: 0.0,
            viewport_width: 1200.0,
            viewport_height: 900.0,
            zoom: 0.03,
            process_all_tiles: false,
        }));
        let visible = job.count_visible_tiles_for_image(40_000, 30_000);
        assert!(
            visible > 4,
            "expected a fully zoomed-out whole-slide view to exceed a 4-tile budget, got {visible}"
        );
    }
}

#[cfg(test)]
mod preview_tile_size_tests {
    use super::*;
    use crate::storage::memory::MemoryExporter;
    use std::sync::{Arc, Mutex};

    fn make_job(zoom: Option<f32>) -> JobExecutor {
        let mut job = JobExecutor::new(
            std::path::PathBuf::new(),
            std::path::PathBuf::new(),
            IndexMap::new(),
            std::path::PathBuf::new(),
            GlobalImageSettings::default(),
            Arc::new(MemoryExporter {
                out_objects: Arc::new(Mutex::new(Vec::new())),
                ..Default::default()
            }),
            None,
            Default::default(),
        );
        job.preview_tile_settings = zoom.map(|zoom| PreviewTileSettings {
            offset_x: 0.0,
            offset_y: 0.0,
            viewport_width: 0.0,
            viewport_height: 0.0,
            zoom,
            process_all_tiles: false,
        });
        job
    }

    #[test]
    fn no_preview_settings_uses_base_tile_size() {
        assert_eq!(make_job(None).preview_tile_size(), 4096);
    }

    #[test]
    fn zoom_just_below_2_uses_base_tile_size() {
        assert_eq!(make_job(Some(1.99)).preview_tile_size(), 4096);
    }

    #[test]
    fn zoom_exactly_2_shrinks_tile_to_2048() {
        assert_eq!(make_job(Some(2.0)).preview_tile_size(), 2048);
    }

    #[test]
    fn zoom_just_below_4_stays_at_2048() {
        assert_eq!(make_job(Some(3.99)).preview_tile_size(), 2048);
    }

    #[test]
    fn zoom_exactly_4_shrinks_tile_to_1024() {
        assert_eq!(make_job(Some(4.0)).preview_tile_size(), 1024);
    }

    #[test]
    fn zoom_just_below_8_stays_at_1024() {
        assert_eq!(make_job(Some(7.99)).preview_tile_size(), 1024);
    }

    #[test]
    fn zoom_exactly_8_shrinks_tile_to_512() {
        assert_eq!(make_job(Some(8.0)).preview_tile_size(), 512);
    }

    #[test]
    fn zoom_far_beyond_8_still_clamps_to_512() {
        assert_eq!(make_job(Some(100.0)).preview_tile_size(), 512);
    }
}

#[cfg(test)]
mod tile_iterator_tests {
    use super::*;
    use crate::storage::memory::MemoryExporter;
    use std::sync::{Arc, Mutex};

    fn make_job() -> JobExecutor {
        JobExecutor::new(
            std::path::PathBuf::new(),
            std::path::PathBuf::new(),
            IndexMap::new(),
            std::path::PathBuf::new(),
            GlobalImageSettings::default(),
            Arc::new(MemoryExporter {
                out_objects: Arc::new(Mutex::new(Vec::new())),
                ..Default::default()
            }),
            None,
            Default::default(),
        )
    }

    #[test]
    fn image_smaller_than_tile_size_yields_single_full_size_tile() {
        let job = make_job();
        let tiles: Vec<_> = job.prepare_tile_iterator(100, 200, 4096).collect();

        assert_eq!(tiles.len(), 1);
        assert_eq!(tiles[0].offset_x, 0);
        assert_eq!(tiles[0].offset_y, 0);
        assert_eq!(tiles[0].width, 100);
        assert_eq!(tiles[0].height, 200);
    }

    #[test]
    fn image_exact_multiple_of_tile_size_yields_uniform_tiles() {
        let job = make_job();
        let tiles: Vec<_> = job.prepare_tile_iterator(8192, 8192, 4096).collect();

        assert_eq!(tiles.len(), 4);
        assert!(tiles.iter().all(|t| t.width == 4096 && t.height == 4096));
    }

    #[test]
    fn image_not_multiple_of_tile_size_yields_remainder_tile() {
        let job = make_job();
        let tiles: Vec<_> = job.prepare_tile_iterator(5000, 3000, 4096).collect();

        assert_eq!(tiles.len(), 2);
        assert_eq!(tiles[0].width, 4096);
        assert_eq!(tiles[1].offset_x, 4096);
        assert_eq!(tiles[1].width, 5000 - 4096);
        assert!(tiles.iter().all(|t| t.height == 3000));
    }

    #[test]
    fn tiles_are_generated_in_row_major_order() {
        let job = make_job();
        let tiles: Vec<_> = job.prepare_tile_iterator(9000, 9000, 4096).collect();
        let offsets: Vec<(usize, usize)> = tiles.iter().map(|t| (t.offset_x, t.offset_y)).collect();

        assert_eq!(
            offsets,
            vec![
                (0, 0),
                (4096, 0),
                (8192, 0),
                (0, 4096),
                (4096, 4096),
                (8192, 4096),
                (0, 8192),
                (4096, 8192),
                (8192, 8192),
            ]
        );
    }

    #[test]
    fn zero_sized_image_yields_no_tiles() {
        let job = make_job();
        let tiles: Vec<_> = job.prepare_tile_iterator(0, 0, 4096).collect();

        assert!(tiles.is_empty());
    }
}

#[cfg(test)]
mod tile_visibility_tests {
    use super::*;

    fn tile(offset_x: usize, offset_y: usize, width: usize, height: usize) -> ImageTile {
        ImageTile {
            offset_x,
            offset_y,
            width,
            height,
        }
    }

    fn settings(offset_x: f32, offset_y: f32, zoom: f32) -> PreviewTileSettings {
        PreviewTileSettings {
            offset_x,
            offset_y,
            viewport_width: 1000.0,
            viewport_height: 1000.0,
            zoom,
            process_all_tiles: false,
        }
    }

    #[test]
    fn tile_fully_inside_viewport_is_visible() {
        let s = settings(0.0, 0.0, 1.0);
        assert!(s.is_tile_visible(&tile(0, 0, 500, 500)));
    }

    #[test]
    fn tile_starting_exactly_at_viewport_right_edge_is_not_visible() {
        let s = settings(0.0, 0.0, 1.0);
        assert!(!s.is_tile_visible(&tile(1000, 0, 500, 500)));
    }

    #[test]
    fn tile_ending_exactly_at_viewport_left_edge_is_not_visible() {
        // Panned so the tile's right edge lands exactly at screen x = 0:
        // x2 = 500 * 1.0 + (-500.0) = 0.0, and the check requires x2 > 0.0.
        let s = settings(-500.0, 0.0, 1.0);
        assert!(!s.is_tile_visible(&tile(0, 0, 500, 500)));
    }

    #[test]
    fn tile_starting_exactly_at_viewport_bottom_edge_is_not_visible() {
        let s = settings(0.0, 0.0, 1.0);
        assert!(!s.is_tile_visible(&tile(0, 1000, 500, 500)));
    }

    #[test]
    fn tile_ending_exactly_at_viewport_top_edge_is_not_visible() {
        let s = settings(0.0, -500.0, 1.0);
        assert!(!s.is_tile_visible(&tile(0, 0, 500, 500)));
    }

    #[test]
    fn tile_partially_overlapping_viewport_is_visible() {
        let s = settings(-800.0, 0.0, 1.0);
        assert!(s.is_tile_visible(&tile(0, 0, 1000, 500)));
    }

    #[test]
    fn tile_far_outside_viewport_is_not_visible() {
        let s = settings(0.0, 0.0, 1.0);
        assert!(!s.is_tile_visible(&tile(50_000, 50_000, 500, 500)));
    }
}

#[cfg(test)]
mod z_stack_iterator_tests {
    use super::*;
    use crate::storage::memory::MemoryExporter;
    use evanalyzer_cfg::settings::images_settings::SeriesSettings;
    use std::collections::BTreeMap;
    use std::sync::{Arc, Mutex};

    fn make_job_executor(global_z: Option<ZStackSettings>) -> JobExecutor {
        let global_image_settings = GlobalImageSettings {
            z_stack: global_z,
            ..Default::default()
        };

        JobExecutor::new(
            std::path::PathBuf::new(),
            std::path::PathBuf::new(),
            IndexMap::new(),
            std::path::PathBuf::new(),
            global_image_settings,
            Arc::new(MemoryExporter {
                out_objects: Arc::new(Mutex::new(Vec::new())),
                ..Default::default()
            }),
            None,
            Default::default(),
        )
    }

    fn make_image_entry() -> ImageEntry {
        let mut series = BTreeMap::new();
        series.insert(0, SeriesSettings::default());
        ImageEntry {
            rel_path: std::path::PathBuf::new(),
            file_size: 0,
            selected_series: 0,
            series,
        }
    }

    #[test]
    fn single_stack_without_explicit_range_defaults_to_first_slice() {
        let job = make_job_executor(Some(ZStackSettings {
            z_projection: ZStackHandling::SingleStack,
            z_range: None,
        }));
        let entry = make_image_entry();
        let image_info = ImageInfo {
            nr_z_stacks: 10,
            ..Default::default()
        };

        let (projection, handling, range) = job.prepare_z_stack_iterator(&image_info, &entry);

        assert_eq!(projection, ZProjection::None);
        assert_eq!(handling, ZStackHandling::SingleStack);
        assert_eq!(range, 0..=0);
    }

    #[test]
    fn single_stack_with_explicit_range_is_honored() {
        let job = make_job_executor(Some(ZStackSettings {
            z_projection: ZStackHandling::SingleStack,
            z_range: Some(2..=5),
        }));
        let entry = make_image_entry();
        let image_info = ImageInfo {
            nr_z_stacks: 10,
            ..Default::default()
        };

        let (_, _, range) = job.prepare_z_stack_iterator(&image_info, &entry);

        assert_eq!(range, 2..=5);
    }

    #[test]
    fn all_stacks_covers_full_range() {
        let job = make_job_executor(Some(ZStackSettings {
            z_projection: ZStackHandling::AllStacks,
            z_range: None,
        }));
        let entry = make_image_entry();
        let image_info = ImageInfo {
            nr_z_stacks: 7,
            ..Default::default()
        };

        let (projection, handling, range) = job.prepare_z_stack_iterator(&image_info, &entry);

        assert_eq!(projection, ZProjection::None);
        assert_eq!(handling, ZStackHandling::AllStacks);
        assert_eq!(range, 0..=6);
    }

    #[test]
    fn all_stacks_with_single_z_stack_yields_single_element_range() {
        let job = make_job_executor(Some(ZStackSettings {
            z_projection: ZStackHandling::AllStacks,
            z_range: None,
        }));
        let entry = make_image_entry();
        let image_info = ImageInfo {
            nr_z_stacks: 1,
            ..Default::default()
        };

        let (_, _, range) = job.prepare_z_stack_iterator(&image_info, &entry);

        assert_eq!(range, 0..=0);
    }

    #[test]
    fn each_projection_mode_maps_to_matching_z_projection_and_collapses_to_single_slice() {
        let cases = [
            (ZStackHandling::MaxIntensity, ZProjection::MaxIntensity),
            (ZStackHandling::MinIntensity, ZProjection::MinIntensity),
            (ZStackHandling::AvgIntensity, ZProjection::AvgIntensity),
            (ZStackHandling::SumIntensity, ZProjection::SumIntensity),
            (ZStackHandling::TakeTheMiddle, ZProjection::TakeTheMiddle),
        ];

        for (handling, expected_projection) in cases {
            let job = make_job_executor(Some(ZStackSettings {
                z_projection: handling.clone(),
                z_range: None,
            }));
            let entry = make_image_entry();
            let image_info = ImageInfo {
                nr_z_stacks: 10,
                ..Default::default()
            };

            let (projection, out_handling, range) =
                job.prepare_z_stack_iterator(&image_info, &entry);

            assert_eq!(
                projection, expected_projection,
                "wrong projection for handling {handling:?}"
            );
            assert_eq!(out_handling, handling);
            assert_eq!(range, 0..=0);
        }
    }
}

#[cfg(test)]
mod execution_order_tests {
    use super::*;
    use crate::algos::{ExecutionScope, ImageAlgorithm};
    use crate::pipeline::pipeline::CorePipelineSettings;
    use crate::pipeline::pipeline_context::PipelineContext;
    use crate::storage::memory::MemoryExporter;
    use evanalyzer_cfg::core_types::CitationMetadata;
    use std::sync::{Arc, Mutex};

    fn make_job_executor() -> JobExecutor {
        JobExecutor::new(
            std::path::PathBuf::new(),
            std::path::PathBuf::new(),
            IndexMap::new(),
            std::path::PathBuf::new(),
            GlobalImageSettings::default(),
            Arc::new(MemoryExporter {
                out_objects: Arc::new(Mutex::new(Vec::new())),
                ..Default::default()
            }),
            None,
            Default::default(),
        )
    }

    fn make_pipeline(id: u32, dependencies: &[u32]) -> Pipeline {
        let mut p = Pipeline::new(
            PipelineId(id),
            CorePipelineSettings {
                start_image: ImageAddress::Channel(0),
            },
        );
        for dep in dependencies {
            p.add_dependency(PipelineId(*dep));
        }
        p
    }

    #[test]
    fn pipelines_without_dependencies_preserve_insertion_order() {
        let mut job = make_job_executor();
        job.add_pre_process_pipeline(make_pipeline(1, &[]));
        job.add_pre_process_pipeline(make_pipeline(2, &[]));
        job.add_pre_process_pipeline(make_pipeline(3, &[]));

        let order = job.get_execution_order();

        assert_eq!(order, vec![PipelineId(1), PipelineId(2), PipelineId(3)]);
    }

    #[test]
    fn dependency_runs_before_dependent() {
        let mut job = make_job_executor();
        job.add_pre_process_pipeline(make_pipeline(1, &[2]));
        job.add_pre_process_pipeline(make_pipeline(2, &[]));

        let order = job.get_execution_order();

        assert_eq!(order, vec![PipelineId(2), PipelineId(1)]);
    }

    #[test]
    fn transitive_dependency_chain_is_fully_ordered() {
        let mut job = make_job_executor();
        job.add_pre_process_pipeline(make_pipeline(1, &[2]));
        job.add_pre_process_pipeline(make_pipeline(2, &[3]));
        job.add_pre_process_pipeline(make_pipeline(3, &[]));

        let order = job.get_execution_order();

        assert_eq!(order, vec![PipelineId(3), PipelineId(2), PipelineId(1)]);
    }

    #[test]
    fn diamond_dependency_orders_shared_dependency_first() {
        let mut job = make_job_executor();
        job.add_pre_process_pipeline(make_pipeline(1, &[])); // A
        job.add_pre_process_pipeline(make_pipeline(2, &[1])); // B depends on A
        job.add_pre_process_pipeline(make_pipeline(3, &[1])); // C depends on A
        job.add_pre_process_pipeline(make_pipeline(4, &[2, 3])); // D depends on B, C

        let order = job.get_execution_order();
        let pos = |id: u32| order.iter().position(|x| *x == PipelineId(id)).unwrap();

        assert!(pos(1) < pos(2));
        assert!(pos(1) < pos(3));
        assert!(pos(2) < pos(4));
        assert!(pos(3) < pos(4));
    }

    #[test]
    fn dependency_on_unregistered_pipeline_is_still_included_in_order() {
        let mut job = make_job_executor();
        job.add_pre_process_pipeline(make_pipeline(1, &[99]));

        let order = job.get_execution_order();

        assert_eq!(order, vec![PipelineId(99), PipelineId(1)]);
    }

    #[test]
    fn job_with_no_pipelines_yields_empty_order() {
        let job = make_job_executor();

        assert!(job.get_execution_order().is_empty());
    }

    #[test]
    #[should_panic(expected = "Circular dependency detected!")]
    fn direct_circular_dependency_panics() {
        let mut job = make_job_executor();
        job.add_pre_process_pipeline(make_pipeline(1, &[2]));
        job.add_pre_process_pipeline(make_pipeline(2, &[1]));

        job.get_execution_order();
    }

    #[test]
    #[should_panic(expected = "Circular dependency detected!")]
    fn self_dependency_panics() {
        let mut job = make_job_executor();
        job.add_pre_process_pipeline(make_pipeline(1, &[1]));

        job.get_execution_order();
    }

    #[test]
    #[should_panic(expected = "Circular dependency detected!")]
    fn transitive_circular_dependency_panics() {
        let mut job = make_job_executor();
        job.add_pre_process_pipeline(make_pipeline(1, &[2]));
        job.add_pre_process_pipeline(make_pipeline(2, &[3]));
        job.add_pre_process_pipeline(make_pipeline(3, &[1]));

        job.get_execution_order();
    }

    // ---- count_whole_image_progress_units ----

    struct NoopCommand;
    impl ImageAlgorithm for NoopCommand {
        fn execute(
            &self,
            _ctx: &mut PipelineContext,
            _cache: &mut GlobalPipelineCache,
        ) -> Result<(), InternalErrors> {
            Ok(())
        }
        fn name(&self) -> &'static str {
            "Noop"
        }
        fn cite(&self) -> Vec<&'static CitationMetadata> {
            Vec::new()
        }
        fn execution_scope(&self) -> ExecutionScope {
            ExecutionScope::WholeImage
        }
    }

    /// Like `make_pipeline`, but with one real command, so
    /// `commands.is_empty()` is `false` - what actually distinguishes a
    /// whole-image pipeline that contributes a progress unit from one that
    /// doesn't.
    fn make_pipeline_with_command(id: u32) -> Pipeline {
        let mut p = make_pipeline(id, &[]);
        p.add_command(Box::new(NoopCommand));
        p
    }

    #[test]
    fn no_post_process_pipelines_still_reserves_one_unit_for_the_empty_object_cache_exit() {
        let job = make_job_executor();
        assert_eq!(job.count_whole_image_progress_units(&[]), 1);
    }

    #[test]
    fn empty_post_process_pipelines_in_order_do_not_add_units() {
        let mut job = make_job_executor();
        job.add_post_process_pipeline(make_pipeline(1, &[])); // no commands

        let order = vec![PipelineId(1)];
        assert_eq!(job.count_whole_image_progress_units(&order), 1);
    }

    #[test]
    fn one_unit_per_non_empty_post_process_pipeline_plus_one_for_export() {
        let mut job = make_job_executor();
        job.add_post_process_pipeline(make_pipeline_with_command(1));
        job.add_post_process_pipeline(make_pipeline_with_command(2));

        let order = vec![PipelineId(1), PipelineId(2)];
        assert_eq!(job.count_whole_image_progress_units(&order), 3);
    }

    #[test]
    fn a_pipeline_id_in_order_but_not_registered_is_not_counted() {
        let mut job = make_job_executor();
        job.add_post_process_pipeline(make_pipeline_with_command(1));

        // `order` can name a pipeline id that was never registered as a
        // post-process pipeline (e.g. a pre-process-only id, or a stale
        // dependency) - `count_whole_image_progress_units` must skip it,
        // same as the real whole-image loop's own `let Some(p) = ... else
        // { continue }`.
        let order = vec![PipelineId(1), PipelineId(99)];
        assert_eq!(job.count_whole_image_progress_units(&order), 2);
    }
}

/// End-to-end coverage for `run`/`analyze_image`/`prepare_pipeline_cache` -
/// the paths above only exercise the pure helper
/// methods (tile math, precedence resolution, execution ordering), never the
/// actual image-loading-through-DB-export flow. Uses the same real image
/// fixture as `image_reader`'s tests.
#[cfg(test)]
mod full_run_integration_tests {
    use super::*;
    use crate::algos::{
        ConnectedComponents, ExtractObjects, Threshold, ThresholdEntry, ThresholdMethod,
        ThresholdValueSource,
    };
    use crate::pipeline::pipeline::CorePipelineSettings;
    use crate::storage::memory::MemoryExporter;
    use evanalyzer_cfg::core_types::{ObjectClass, ObjectId, PixelUnits, SegmentationClass};
    use evanalyzer_cfg::settings::images_settings::{SeriesSettings, TStackSettings};
    use std::collections::BTreeMap;
    use std::sync::atomic::AtomicBool;
    use std::sync::{Arc, Mutex};

    fn threshold_connected_components_extract_pipeline() -> Pipeline {
        let mut pipeline = Pipeline::new(
            PipelineId(1),
            CorePipelineSettings {
                start_image: ImageAddress::Channel(0),
            },
        );
        // Manual threshold spanning the full 8-bit range: every pixel is
        // assigned to the same segmentation class, so the fixture image
        // (known non-empty from image_reader's tests on the same file)
        // reliably produces at least one connected component.
        // A project step (id 1): what the breakpoint tests target.
        pipeline.add_step(
            evanalyzer_cfg::core_types::StepId(1),
            Scope::Global,
            Box::new(Threshold {
                thresholds: vec![ThresholdEntry {
                    method: ThresholdMethod::Manual,
                    min_threshold: 0.0,
                    max_threshold: 255.0,
                    unit: PixelUnits::Bit,
                    segmentation_class: SegmentationClass(1),
                    value_source: ThresholdValueSource::ActualImage,
                }],
            }),
        );
        pipeline.add_command(Box::new(ConnectedComponents { min_size: 0 }));
        pipeline.add_command(Box::new(ExtractObjects {
            max_objects_before_fail: 100_000,
            class_mapping: vec![],
        }));
        pipeline
    }

    /// Remembers the planes the run reports analysed (`finalize_image`),
    /// as (t, z) pairs, and the frame interval.
    #[derive(Default)]
    struct PlanesRecorder(Mutex<Vec<(i32, i32)>>, Mutex<Option<f64>>);

    impl PipelineResultExporter for PlanesRecorder {
        fn export(&self, _cache: &GlobalPipelineCache) -> Result<(), InternalErrors> {
            Ok(())
        }

        fn finalize_image(
            &self,
            _image_rel_path: &std::path::Path,
            summary: &ImageSummary,
            _error: Option<&str>,
        ) -> Result<(), InternalErrors> {
            let mut planes = self.0.lock().unwrap();
            for &t in summary.t_stacks_analyzed {
                planes.extend(summary.z_stacks_analyzed.iter().map(|&z| (t, z)));
            }
            *self.1.lock().unwrap() = summary.frame_interval_s;
            Ok(())
        }
    }

    /// The (t, z) planes a run of the fixture image with `z_stack` reports.
    fn analysed_planes(z_stack: Option<ZStackSettings>) -> Vec<(i32, i32)> {
        let recorder = Arc::new(PlanesRecorder::default());
        let mut job = make_single_image_job(Arc::new(Mutex::new(Vec::new())));
        job.result_storage = recorder.clone();
        job.global_image_settings.z_stack = z_stack;
        let (tx, _rx) = std::sync::mpsc::channel();
        job.run(1, tx, Arc::new(AtomicBool::new(false))).unwrap();
        recorder.0.lock().unwrap().clone()
    }

    /// The project's frame interval is recorded with the image - over the
    /// image's own; an invalid one counts as unset.
    #[test]
    fn the_run_records_the_projects_frame_interval() {
        let interval = |seconds: Option<f32>| {
            let recorder = Arc::new(PlanesRecorder::default());
            let mut job = make_single_image_job(Arc::new(Mutex::new(Vec::new())));
            job.result_storage = recorder.clone();
            job.global_image_settings.t_stack = Some(TStackSettings {
                frame_interval: seconds,
                ..Default::default()
            });
            let (tx, _rx) = std::sync::mpsc::channel();
            job.run(1, tx, Arc::new(AtomicBool::new(false))).unwrap();
            *recorder.1.lock().unwrap()
        };
        assert_eq!(interval(Some(30.0)), Some(30.0));
        let from_the_image = interval(None);
        assert_eq!(
            interval(Some(0.0)),
            from_the_image,
            "0 is unset: the image's own"
        );
        assert_eq!(interval(Some(f32::NAN)), from_the_image);
    }

    /// Every analysed plane is reported - each z-plane analysed one by one,
    /// a projection's one z - whether objects were found there or not.
    #[test]
    fn the_run_reports_every_plane_it_analysed() {
        let one_by_one = analysed_planes(Some(ZStackSettings {
            z_projection: ZStackHandling::AllStacks,
            z_range: None,
        }));
        let projected = analysed_planes(Some(ZStackSettings {
            z_projection: ZStackHandling::MaxIntensity,
            z_range: None,
        }));
        let z_planes: HashSet<i32> = one_by_one.iter().map(|(_, z)| *z).collect();
        assert!(
            z_planes.len() > 2,
            "the fixture has several z-planes: {one_by_one:?}"
        );
        assert!(projected.iter().all(|(_, z)| *z == 0), "{projected:?}");
        let times = |planes: &[(i32, i32)]| planes.iter().map(|(t, _)| *t).collect::<HashSet<_>>();
        assert_eq!(
            times(&projected),
            times(&one_by_one),
            "the same time points"
        );
        assert_eq!(
            projected.len(),
            times(&projected).len(),
            "one plane per time point"
        );
    }

    fn make_single_image_job(out_objects: Arc<Mutex<Vec<ObjectMetricSettings>>>) -> JobExecutor {
        let rel_path = PathBuf::from("multi-channel-4D-series.ome.tif");
        let mut series = BTreeMap::new();
        series.insert(0, SeriesSettings::default());
        let mut images = IndexMap::new();
        images.insert(
            rel_path.clone(),
            ImageEntry {
                rel_path,
                file_size: 0,
                selected_series: 0,
                series,
            },
        );

        let mut job = JobExecutor::new(
            PathBuf::new(),
            std::env::temp_dir(),
            images,
            PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR"), "/tests")),
            GlobalImageSettings::default(),
            Arc::new(MemoryExporter {
                out_objects,
                ..Default::default()
            }),
            None,
            Default::default(),
        );
        job.add_pre_process_pipeline(threshold_connected_components_extract_pipeline());
        job
    }

    /// What reaches the exporter: per plane, its (t, z) and the values of
    /// its measurements.
    struct MeasurementRecorder(Mutex<Vec<((i32, i32), Vec<f64>)>>);

    impl PipelineResultExporter for MeasurementRecorder {
        fn export(&self, cache: &GlobalPipelineCache) -> Result<(), InternalErrors> {
            use crate::pipeline::pipeline_measure::MeasurementValues;
            // The Pearson value of Measure Colocalization's.
            let values = cache
                .measurements
                .measurements
                .iter()
                .filter(|m| m.info.key.ends_with("/pearson"))
                .flat_map(|m| match &m.values {
                    MeasurementValues::Image(values) => values.iter().map(|(_, v)| *v).collect(),
                    MeasurementValues::Objects(_) => Vec::new(),
                })
                .collect();
            let plane = (cache.measurements.t_stack, cache.measurements.z_stack);
            self.0.lock().unwrap().push((plane, values));
            Ok(())
        }
    }

    /// A measurement step measures every plane of an image without any
    /// objects too - the whole-image phase isn't skipped for it.
    #[test]
    fn an_image_without_objects_is_measured_on_every_plane() {
        let recorder = Arc::new(MeasurementRecorder(Mutex::new(Vec::new())));
        let mut job = make_single_image_job(Arc::new(Mutex::new(Vec::new())));
        job.result_storage = recorder.clone();
        job.pipelines_pre_process.clear();
        let mut measure = Pipeline::new(
            PipelineId(1),
            CorePipelineSettings {
                start_image: ImageAddress::Scratchpad,
            },
        );
        measure.add_command(Box::new(crate::algos::MeasureColocalization {
            id: "pearson_01".into(),
            channel_a: evanalyzer_cfg::core_types::ImageChannelIdx(0),
            channel_b: evanalyzer_cfg::core_types::ImageChannelIdx(1),
            threshold_percent: 15.0,
            costes: crate::algos::CostesMethod::Off,
            objects: ObjectClass::Unset,
        }));
        job.add_post_process_pipeline(measure);

        let (tx, _rx) = std::sync::mpsc::channel();
        job.run(1, tx, Arc::new(AtomicBool::new(false))).unwrap();

        let planes = recorder.0.lock().unwrap();
        assert!(!planes.is_empty(), "no plane measured");
        let mut seen = HashSet::new();
        for (plane, values) in planes.iter() {
            assert!(seen.insert(*plane), "plane {plane:?} twice");
            let [r] = values[..] else {
                panic!("plane {plane:?}: one value, got {values:?}");
            };
            assert!((-1.0..=1.0).contains(&r), "plane {plane:?}: {r}");
        }
    }

    /// Per plane: the measurements' values, the area of its objects and of
    /// the image.
    struct AreaRecorder(Mutex<Vec<(Vec<f64>, usize, usize)>>);

    impl PipelineResultExporter for AreaRecorder {
        fn export(&self, cache: &GlobalPipelineCache) -> Result<(), InternalErrors> {
            use crate::pipeline::pipeline_measure::MeasurementValues;
            let values = cache
                .measurements
                .measurements
                .iter()
                .flat_map(|m| match &m.values {
                    MeasurementValues::Image(values) => values.iter().map(|(_, v)| *v).collect(),
                    MeasurementValues::Objects(_) => Vec::new(),
                })
                .collect();
            let area = cache.object_cache.values().map(|o| o.area).sum();
            let size = &cache.image_meta.full_image_width;
            let image_area = size.width * size.height;
            self.0.lock().unwrap().push((values, area, image_area));
            Ok(())
        }
    }

    /// In a real run a measurement step sees the objects the pipelines
    /// before it found (the threshold's lower bound is exclusive: the
    /// fixture's black first row isn't covered).
    #[test]
    fn area_occupied_sees_the_objects_of_the_run() {
        let recorder = Arc::new(AreaRecorder(Mutex::new(Vec::new())));
        let mut job = make_single_image_job(Arc::new(Mutex::new(Vec::new())));
        job.result_storage = recorder.clone();
        let mut measure = Pipeline::new(
            PipelineId(2),
            CorePipelineSettings {
                start_image: ImageAddress::Scratchpad,
            },
        );
        measure.add_command(Box::new(crate::algos::AreaOccupied {
            id: "area_occupied_01".into(),
            object_class: ObjectClass::Global(1),
        }));
        job.add_post_process_pipeline(measure);

        let (tx, _rx) = std::sync::mpsc::channel();
        job.run(1, tx, Arc::new(AtomicBool::new(false))).unwrap();

        let planes = recorder.0.lock().unwrap();
        assert!(!planes.is_empty(), "no plane measured");
        for (values, objects_area, image_area) in planes.iter() {
            // area_px, area_um2 (if the image has a pixel size), fraction
            let (area_px, fraction) = (values[0], values[values.len() - 1]);
            assert!(*objects_area > 0);
            assert_eq!(area_px, *objects_area as f64);
            let expected = *objects_area as f64 / *image_area as f64;
            assert!(
                (fraction - expected).abs() < 1e-12,
                "{fraction} != {expected}"
            );
        }
    }

    /// Several planes: the first one's reason with its plane, and how many
    /// more excluded it.
    #[test]
    fn an_exclusion_names_its_plane_when_there_are_several() {
        let one = vec![(0, 0, "a".to_string())];
        assert_eq!(super::exclusion_reason(&one, 1).as_deref(), Some("a"));
        assert_eq!(
            super::exclusion_reason(&one, 4).as_deref(),
            Some("t 0, z 0: a")
        );
        let three = vec![
            (2, 1, "b".to_string()),
            (3, 1, "c".to_string()),
            (4, 1, "d".to_string()),
        ];
        assert_eq!(
            super::exclusion_reason(&three, 10).as_deref(),
            Some("t 2, z 1: b (and 2 more planes)")
        );
        assert_eq!(super::exclusion_reason(&[], 10), None);
    }

    /// What `finalize_image` was told about exclusion.
    #[derive(Default)]
    struct ExclusionRecorder(Mutex<Vec<Option<String>>>);

    impl PipelineResultExporter for ExclusionRecorder {
        fn export(&self, _cache: &GlobalPipelineCache) -> Result<(), InternalErrors> {
            Ok(())
        }

        fn finalize_image(
            &self,
            _image_rel_path: &std::path::Path,
            summary: &ImageSummary,
            _error: Option<&str>,
        ) -> Result<(), InternalErrors> {
            self.0
                .lock()
                .unwrap()
                .push(summary.excluded.map(str::to_string));
            Ok(())
        }
    }

    /// Image Quality, then Exclude Image: a rule the plane meets excludes
    /// the image, one it doesn't meet doesn't - the image is analysed either
    /// way.
    #[test]
    fn exclude_image_excludes_the_image_in_a_run() {
        use crate::algos::{ExcludeImage, ExcludeRule, ExcludeWhen, ImageQuality, RuleComparison};
        let run = |comparison: RuleComparison| {
            let recorder = Arc::new(ExclusionRecorder::default());
            let mut job = make_single_image_job(Arc::new(Mutex::new(Vec::new())));
            job.result_storage = recorder.clone();
            job.pipelines_pre_process.clear();
            let mut check = Pipeline::new(
                PipelineId(1),
                CorePipelineSettings {
                    start_image: ImageAddress::Scratchpad,
                },
            );
            check.add_command(Box::new(ImageQuality {
                id: "quality_01".into(),
                channel: evanalyzer_cfg::core_types::ImageChannelIdx(0),
            }));
            check.add_command(Box::new(ExcludeImage {
                exclude_when: ExcludeWhen::AnyRule,
                rules: vec![ExcludeRule {
                    measurement: "quality_01/percent_maximal".into(),
                    comparison,
                    // Some pixel is always at the maximum: above 0 %, never below.
                    value: 0.0,
                }],
            }));
            job.add_post_process_pipeline(check);
            let (tx, _rx) = std::sync::mpsc::channel();
            job.run(1, tx, Arc::new(AtomicBool::new(false))).unwrap();
            let excluded = recorder.0.lock().unwrap().clone();
            excluded
        };

        let [excluded] = &run(RuleComparison::Above)[..] else {
            panic!("one image");
        };
        let reason = excluded.as_deref().expect("excluded");
        // One plane analysed: no plane in the reason.
        assert!(
            reason.starts_with("Pixels at the maximum (channel 0) of quality_01 = ")
                && reason.ends_with(", above 0"),
            "{reason}"
        );

        assert_eq!(run(RuleComparison::Below), [None], "not excluded");
    }

    /// In a real run Propagation grows the objects the pipelines before it
    /// found: each region holds its seed, is linked to it and has the
    /// output class.
    #[test]
    fn propagation_grows_the_objects_of_the_run() {
        let out_objects = Arc::new(Mutex::new(Vec::new()));
        let mut job = make_single_image_job(out_objects.clone());
        let mut grow = Pipeline::new(
            PipelineId(2),
            CorePipelineSettings {
                start_image: ImageAddress::Scratchpad,
            },
        );
        grow.add_command(Box::new(crate::algos::Propagation {
            seeds: ObjectClass::Global(1),
            growth_channel: evanalyzer_cfg::core_types::ImageChannelIdx(1),
            mask: ObjectClass::Unset,
            regularization: 0.05,
            output_class: ObjectClass::Global(2),
            exclude_at_edges: false,
        }));
        job.add_post_process_pipeline(grow);

        let (tx, _rx) = std::sync::mpsc::channel();
        job.run(1, tx, Arc::new(AtomicBool::new(false))).unwrap();

        let objects = out_objects.lock().unwrap();
        let has = |o: &ObjectMetricSettings, class: u32| {
            o.object_class.contains(&ObjectClass::Global(class))
        };
        let seeds: Vec<_> = objects.iter().filter(|o| has(o, 1)).collect();
        let grown: Vec<_> = objects.iter().filter(|o| has(o, 2)).collect();
        assert!(!seeds.is_empty(), "the run found seeds");
        assert_eq!(grown.len(), seeds.len(), "one region per seed");
        for region in &grown {
            let inside = |s: &&&ObjectMetricSettings| {
                let ([sx0, sy0, sx1, sy1], [rx0, ry0, rx1, ry1]) = (s.bbox, region.bbox);
                sx0 >= rx0 && sy0 >= ry0 && sx1 <= rx1 && sy1 <= ry1
            };
            let seed = seeds.iter().find(inside).expect("a seed inside it");
            assert!(region.area >= seed.area, "holds its seed");
        }
    }

    #[test]
    fn a_pipeline_reading_a_channel_the_image_lacks_fails_with_a_clear_message() {
        // What the pre-start check catches for images with stored channel
        // info - here it reaches the run itself.
        let out_objects = Arc::new(Mutex::new(Vec::new()));
        let mut job = make_single_image_job(out_objects);
        let mut pipeline = threshold_connected_components_extract_pipeline();
        pipeline.settings.start_image = ImageAddress::Channel(9);
        job.pipelines_pre_process.clear();
        job.add_pre_process_pipeline(pipeline);

        let (tx, _rx) = std::sync::mpsc::channel();
        let err = job
            .run(1, tx, Arc::new(AtomicBool::new(false)))
            .unwrap_err();
        let message = format!("{err}");
        assert!(message.contains("reads image channel 9"), "{message}");
        assert!(
            message.contains("check the pipeline's image source"),
            "{message}"
        );
    }

    #[test]
    fn a_modules_internal_objects_never_reach_the_results() {
        use crate::algos::{ClassifyMatchHandling, ClassifyObjects};
        use crate::pipeline::scope::ScopeId;

        // Every extracted object is reclassified into a local class by a
        // command running in a module instance's scope.
        let run_with_reclassify_in = |scope: Scope| {
            let out_objects = Arc::new(Mutex::new(Vec::new()));
            let mut job = make_single_image_job(out_objects.clone());
            let mut post = Pipeline::new(
                PipelineId(1),
                CorePipelineSettings {
                    start_image: ImageAddress::Scratchpad,
                },
            );
            post.add_step(
                evanalyzer_cfg::core_types::StepId::UNASSIGNED,
                scope,
                Box::new(ClassifyObjects {
                    match_handling: ClassifyMatchHandling::ReclassifyIfMatch,
                    output_class: ObjectClass::Local(1),
                    ..Default::default()
                }),
            );
            job.add_post_process_pipeline(post);
            let (tx, _rx) = std::sync::mpsc::channel();
            job.run(1, tx, Arc::new(AtomicBool::new(false))).unwrap();
            out_objects.lock().unwrap().len()
        };

        assert_eq!(
            run_with_reclassify_in(Scope::Scoped(ScopeId(1))),
            0,
            "local objects are internal"
        );
        // Control: the same run without the module's scope exports them (a
        // local class outside a module is otherwise refused at expansion).
        assert!(run_with_reclassify_in(Scope::Global) > 0);
    }

    #[test]
    fn a_pipeline_without_an_image_source_loads_its_channel_like_a_channel_source() {
        use crate::algos::LoadChannel;
        use evanalyzer_cfg::core_types::ImageChannelIdx;
        let run = |source_none: bool| {
            let out_objects = Arc::new(Mutex::new(Vec::new()));
            let mut job = make_single_image_job(out_objects.clone());
            let mut pipeline = threshold_connected_components_extract_pipeline();
            if source_none {
                pipeline.settings.start_image = ImageAddress::None;
                pipeline.commands.insert(
                    0,
                    (
                        evanalyzer_cfg::core_types::StepId::UNASSIGNED,
                        Scope::Global,
                        Box::new(LoadChannel {
                            channel: ImageChannelIdx(0),
                            projection: crate::algos::ChannelZProjection::Auto,
                            z_index: 0,
                        }),
                    ),
                );
            }
            job.pipelines_pre_process.clear();
            job.add_pre_process_pipeline(pipeline);
            let (tx, _rx) = std::sync::mpsc::channel();
            job.run(1, tx, Arc::new(AtomicBool::new(false))).unwrap();
            let mut areas: Vec<usize> =
                out_objects.lock().unwrap().iter().map(|o| o.area).collect();
            areas.sort_unstable();
            areas
        };
        let from_channel = run(false);
        assert!(!from_channel.is_empty(), "the fixture image has objects");
        assert_eq!(run(true), from_channel, "the same objects");
    }

    /// A run whose pipeline starts with `load` (no image source) - its
    /// objects' areas, sorted, or the run's error.
    fn run_loading(
        load: crate::algos::LoadChannel,
        z_stack: Option<ZStackSettings>,
    ) -> Result<Vec<usize>, InternalErrors> {
        let out_objects = Arc::new(Mutex::new(Vec::new()));
        let mut job = make_single_image_job(out_objects.clone());
        job.global_image_settings.z_stack = z_stack;
        let mut pipeline = threshold_connected_components_extract_pipeline();
        pipeline.settings.start_image = ImageAddress::None;
        pipeline.commands.insert(
            0,
            (
                evanalyzer_cfg::core_types::StepId::UNASSIGNED,
                Scope::Global,
                Box::new(load),
            ),
        );
        job.pipelines_pre_process.clear();
        job.add_pre_process_pipeline(pipeline);
        let (tx, _rx) = std::sync::mpsc::channel();
        job.run(1, tx, Arc::new(AtomicBool::new(false)))?;
        let mut areas: Vec<usize> = out_objects.lock().unwrap().iter().map(|o| o.area).collect();
        areas.sort_unstable();
        Ok(areas)
    }

    fn load(
        projection: crate::algos::ChannelZProjection,
        z_index: u32,
    ) -> crate::algos::LoadChannel {
        crate::algos::LoadChannel {
            channel: evanalyzer_cfg::core_types::ImageChannelIdx(0),
            projection,
            z_index,
        }
    }

    /// A Load Image Channel step with its own projection or z-plane reads
    /// its channel so (the run finds what the step looks up) - also while the
    /// project analyzes every z-plane.
    #[test]
    fn a_load_channel_step_with_its_own_projection_or_plane_runs() {
        use crate::algos::ChannelZProjection;
        let all_planes = Some(ZStackSettings {
            z_projection: ZStackHandling::AllStacks,
            z_range: None,
        });
        for z_stack in [None, all_planes] {
            for step in [
                load(ChannelZProjection::MaxIntensity, 0),
                load(ChannelZProjection::None, 1),
            ] {
                let areas = run_loading(step, z_stack.clone()).unwrap();
                assert!(!areas.is_empty(), "the fixture image has objects");
            }
        }
    }

    /// A z-plane the image doesn't have stops the run before any tile is
    /// read, saying which plane and what the image has.
    #[test]
    fn a_load_channel_step_with_a_plane_the_image_lacks_fails_clearly() {
        let err = run_loading(load(crate::algos::ChannelZProjection::None, 999), None).unwrap_err();
        let message = format!("{err}");
        assert!(
            message.contains("loads z-plane 999 of channel 0"),
            "{message}"
        );
        assert!(message.contains("z-planes 0–"), "{message}");
    }

    fn fixture_reader() -> (ImageReader, i32) {
        let path = PathBuf::from(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/multi-channel-4D-series.ome.tif"
        ));
        let reader = ImageReader::new(&path, ReadMode::Default).unwrap();
        let channels = reader.get_image_meta().series[&0].nr_c_stacks;
        (reader, channels)
    }

    fn whole_image() -> ImageTile {
        ImageTile {
            offset_x: 0,
            offset_y: 0,
            width: 0,
            height: 0,
        }
    }

    fn pixels(image: &ImageContainer) -> Vec<f32> {
        match image {
            ImageContainer::F32Gray(image) => image.as_slice().to_vec(),
            other => panic!("expected a gray image, got {other:?}"),
        }
    }

    /// The variants in `cache`, by their projection.
    fn variant_images(cache: &GlobalPipelineCache) -> Vec<(ChannelVariant, Arc<ImageContainer>)> {
        cache
            .image_cache
            .keys()
            .filter_map(|key| match key {
                CacheAddress::ChannelVariant((variant, _)) => {
                    Some((variant, cache.image_cache.get(&key)?))
                }
                _ => None,
            })
            .collect()
    }

    /// A variant is the channel read with the variant's projection or plane.
    #[test]
    fn a_channel_variant_is_the_channel_read_with_its_projection() {
        let job = make_single_image_job(Arc::new(Mutex::new(Vec::new())));
        let (reader, channels) = fixture_reader();
        let entry = job.images.values().next().unwrap().clone();
        let read = |projection: ZProjection, z: Option<RangeInclusive<i32>>| {
            reader
                .read_image_tile_combined(0, 0, projection, &z, 0, Some(&vec![0]), &whole_image())
                .unwrap()
                .remove(0)
                .image
        };
        let max = ChannelVariant {
            channel: 0,
            projection: ZProjection::MaxIntensity,
            z: 0,
        };
        let plane = ChannelVariant {
            channel: 0,
            projection: ZProjection::None,
            z: 1,
        };
        let mut cache = GlobalPipelineCache::default();
        job.add_channel_variants(
            &mut cache,
            &reader,
            &entry,
            &whole_image(),
            0,
            ZProjection::None,
            0,
            &[max, plane],
            channels as u32,
            None,
        )
        .unwrap();
        let images = variant_images(&cache);
        let image_of = |v: ChannelVariant| {
            pixels(
                &images
                    .iter()
                    .find(|(variant, _)| *variant == v)
                    .expect("read")
                    .1,
            )
        };
        assert_eq!(
            image_of(max),
            pixels(&read(ZProjection::MaxIntensity, None))
        );
        assert_eq!(
            image_of(plane),
            pixels(&read(ZProjection::None, Some(1..=1)))
        );
        assert_ne!(
            image_of(max),
            image_of(plane),
            "the fixture's planes differ"
        );
    }

    /// Not read again: a variant already read for this tile on an earlier
    /// z-plane (`reuse`), or one with the projection the channels have.
    #[test]
    fn a_channel_variant_is_not_read_twice() {
        let job = make_single_image_job(Arc::new(Mutex::new(Vec::new())));
        let (reader, channels) = fixture_reader();
        let entry = job.images.values().next().unwrap().clone();
        let tile = whole_image();
        let max = ChannelVariant {
            channel: 0,
            projection: ZProjection::MaxIntensity,
            z: 0,
        };
        let marker = |value: f32| {
            Arc::new(ImageContainer::new_f32_gray_from_image_test(
                kornia_image::Image::new(
                    ImageSize {
                        width: 1,
                        height: 1,
                    },
                    vec![value],
                )
                .unwrap(),
            ))
        };

        // Read on an earlier plane: taken from the store.
        let store: VariantStore = Mutex::new(HashMap::new());
        store.lock().unwrap().insert((max, 0, 0), marker(42.0));
        let mut cache = GlobalPipelineCache::default();
        job.add_channel_variants(
            &mut cache,
            &reader,
            &entry,
            &tile,
            0,
            ZProjection::None,
            0,
            &[max],
            channels as u32,
            Some(&store),
        )
        .unwrap();
        assert_eq!(pixels(&variant_images(&cache)[0].1), [42.0]);

        // The channels are max projections already: the variant is theirs.
        let mut cache = GlobalPipelineCache::default();
        let channel = marker(7.0);
        cache.add_to_channel_cache(
            channel.clone(),
            0,
            ImageTile {
                width: 1,
                height: 1,
                ..tile
            },
        );
        job.add_channel_variants(
            &mut cache,
            &reader,
            &entry,
            &tile,
            0,
            ZProjection::MaxIntensity,
            0,
            &[max],
            channels as u32,
            None,
        )
        .unwrap();
        assert!(Arc::ptr_eq(&variant_images(&cache)[0].1, &channel));

        // First read with a store: kept for the planes after.
        let store: VariantStore = Mutex::new(HashMap::new());
        let mut cache = GlobalPipelineCache::default();
        job.add_channel_variants(
            &mut cache,
            &reader,
            &entry,
            &tile,
            0,
            ZProjection::None,
            0,
            &[max],
            channels as u32,
            Some(&store),
        )
        .unwrap();
        assert!(store.lock().unwrap().contains_key(&(max, 0, 0)));
    }

    #[test]
    fn a_plane_outside_the_image_is_refused_others_and_projections_pass() {
        let variant = |projection, z| ChannelVariant {
            channel: 2,
            projection,
            z,
        };
        assert!(check_channel_variant_planes(&[variant(ZProjection::None, 4)], 5).is_ok());
        assert!(check_channel_variant_planes(&[variant(ZProjection::MaxIntensity, 0)], 1).is_ok());
        let err = check_channel_variant_planes(&[variant(ZProjection::None, 5)], 5).unwrap_err();
        assert!(
            format!("{err}")
                .contains("loads z-plane 5 of channel 2, but this image has z-planes 0–4"),
            "{err}"
        );
        let err = check_channel_variant_planes(&[variant(ZProjection::None, 1)], 1).unwrap_err();
        assert!(format!("{err}").contains("has only z-plane 0"), "{err}");
    }

    /// Pipeline 1 stores its image in a shared slot, pipeline 2 starts from
    /// it - also when pipeline 2's source is a pipeline slot, as projects saved
    /// before chose. (Every slot used to be dropped after each pipeline, so no
    /// image could get from one pipeline to the next.)
    #[test]
    fn a_pipeline_starts_from_the_image_an_earlier_pipeline_stored() {
        use crate::algos::{ImageCache, ImageCacheMode};
        use evanalyzer_cfg::core_types::MemoryId;
        for source in [MemoryId::ProjectCache(1), MemoryId::PipelineContext(1)] {
            let out_objects = Arc::new(Mutex::new(Vec::new()));
            let mut job = make_single_image_job(out_objects.clone());
            job.pipelines_pre_process.clear();
            let mut store = Pipeline::new(
                PipelineId(1),
                CorePipelineSettings {
                    start_image: ImageAddress::Channel(0),
                },
            );
            store.add_command(Box::new(ImageCache {
                mode: ImageCacheMode::Store,
                address: ImageAddress::Memory(MemoryId::ProjectCache(1)),
            }));
            job.add_pre_process_pipeline(store);
            let mut detect = threshold_connected_components_extract_pipeline();
            detect.id = PipelineId(2);
            detect.settings.start_image = ImageAddress::Memory(source);
            job.add_pre_process_pipeline(detect);

            let (tx, _rx) = std::sync::mpsc::channel();
            job.run(1, tx, Arc::new(AtomicBool::new(false)))
                .unwrap_or_else(|e| panic!("{source:?}: {e}"));
            assert!(!out_objects.lock().unwrap().is_empty(), "{source:?}");
        }
    }

    /// A synthetic 16-bit image made of ties: 3 x 3 spots (2900) on a grid
    /// whose spacing alternates 40 / 41 px, so Voronoi cells have pixels
    /// exactly as far from two spots (x = 40, 121) - and bars (4000) lying
    /// across two cells: at x 77..=84 exactly half in each (Coloc / Relate
    /// ties, three of them), at x 36..=44 across a column of equidistant pixels.
    fn write_tie_image(dir: &std::path::Path) -> PathBuf {
        let (width, height) = (170u32, 120u32);
        let mut image = image::ImageBuffer::<image::Luma<u16>, Vec<u16>>::from_pixel(
            width,
            height,
            image::Luma([200]),
        );
        for cy in [20u32, 60, 100] {
            for cx in [20u32, 60, 101, 141] {
                for y in cy - 1..=cy + 1 {
                    for x in cx - 1..=cx + 1 {
                        image.put_pixel(x, y, image::Luma([2900]));
                    }
                }
            }
        }
        for (x0, x1, y) in [
            (77u32, 84u32, 20u32),
            (77, 84, 60),
            (77, 84, 100),
            (36, 44, 60),
            (118, 124, 100),
        ] {
            for x in x0..=x1 {
                image.put_pixel(x, y, image::Luma([4000]));
            }
        }
        let path = dir.join("ties.tif");
        image.save(&path).unwrap();
        path
    }

    /// Threshold `min..=max` of channel 0 → objects of class `class`.
    fn objects_pipeline(id: u32, min: f32, max: f32, class: u32) -> Pipeline {
        let mut pipeline = Pipeline::new(
            PipelineId(id),
            CorePipelineSettings {
                start_image: ImageAddress::Channel(0),
            },
        );
        pipeline.add_command(Box::new(Threshold {
            thresholds: vec![ThresholdEntry {
                method: ThresholdMethod::Manual,
                min_threshold: min,
                max_threshold: max,
                unit: PixelUnits::Bit,
                segmentation_class: SegmentationClass(class),
                value_source: ThresholdValueSource::ActualImage,
            }],
        }));
        pipeline.add_command(Box::new(ConnectedComponents { min_size: 0 }));
        pipeline.add_command(Box::new(ExtractObjects {
            max_objects_before_fail: 100_000,
            class_mapping: vec![],
        }));
        pipeline
    }

    /// Spots, bars, Voronoi cells around the spots, the bars colocalized
    /// with one cell each (their overlap as objects) and related to one -
    /// run with `threads`; every exported object as text without ids
    /// (partners and parents by their box), sorted.
    fn tie_run(dir: &std::path::Path, threads: usize) -> Vec<String> {
        let image_path = write_tie_image(dir);
        let rel_path = PathBuf::from(image_path.file_name().unwrap());
        let mut series = BTreeMap::new();
        series.insert(0, SeriesSettings::default());
        let mut images = IndexMap::new();
        images.insert(
            rel_path.clone(),
            ImageEntry {
                rel_path,
                file_size: 0,
                selected_series: 0,
                series,
            },
        );
        let out_objects = Arc::new(Mutex::new(Vec::new()));
        let mut job = JobExecutor::new(
            PathBuf::new(),
            std::env::temp_dir(),
            images,
            dir.to_path_buf(),
            GlobalImageSettings::default(),
            Arc::new(MemoryExporter {
                out_objects: out_objects.clone(),
                ..Default::default()
            }),
            None,
            Default::default(),
        );
        let (spot, cell, bar, overlap) = (
            ObjectClass::Global(1),
            ObjectClass::Global(2),
            ObjectClass::Global(3),
            ObjectClass::Global(4),
        );
        job.add_pre_process_pipeline(objects_pipeline(1, 2500.0, 3500.0, 1));
        job.add_pre_process_pipeline(objects_pipeline(2, 3600.0, 65535.0, 3));
        let mut whole = Pipeline::new(
            PipelineId(3),
            CorePipelineSettings {
                start_image: ImageAddress::Scratchpad,
            },
        );
        whole.add_command(Box::new(crate::algos::Voronoi {
            centers: spot,
            center_filter_classes: vec![],
            mask: ObjectClass::Unset,
            mask_filter_classes: vec![],
            output_class: cell,
            unit: evanalyzer_cfg::core_types::SizeUnits::Pixels,
            max_radius: 0.0,
            exclude_areas_at_the_edges: false,
            exclude_areas_with_no_center: false,
        }));
        whole.add_command(Box::new(crate::algos::Colocalization {
            classes_to_coloc: vec![cell, bar],
            all_combinations: false,
            exclude_classes: vec![],
            min_coloc_area: 0.0,
            size_unit: evanalyzer_cfg::core_types::SizeUnitsRel::Pixels,
            multiplicity: crate::algos::ColocMultiplicity::OneToOne,
            class_for_overlapping_areas: overlap,
            mark_objects_of: ObjectClass::Unset,
            mark_as: ObjectClass::Unset,
            filter_classes: vec![],
        }));
        whole.add_command(Box::new(crate::algos::RelateObjects {
            child_classes: vec![bar],
            parent_classes: vec![cell],
            rule: crate::algos::RelateRule::LargestOverlap,
            remove_without_parent: false,
        }));
        job.add_post_process_pipeline(whole);
        let (tx, _rx) = std::sync::mpsc::channel();
        job.run(threads, tx, Arc::new(AtomicBool::new(false)))
            .unwrap();

        let objects = out_objects.lock().unwrap();
        let boxes: std::collections::HashMap<ObjectId, [u32; 4]> =
            objects.iter().map(|o| (o.id.clone(), o.bbox)).collect();
        let mut described: Vec<String> = objects
            .iter()
            .map(|o| {
                let mut classes: Vec<String> =
                    o.object_class.iter().map(|c| format!("{c:?}")).collect();
                classes.sort();
                let mut partners: Vec<String> = o
                    .colocalized_with
                    .iter()
                    .map(|(class, ids)| {
                        let mut at: Vec<_> = ids.iter().map(|id| boxes.get(id)).collect();
                        at.sort();
                        format!("{class:?} {at:?}")
                    })
                    .collect();
                partners.sort();
                let mut parents: Vec<String> = o
                    .parents
                    .iter()
                    .map(|(class, id)| format!("{class:?} {:?}", boxes.get(id)))
                    .collect();
                parents.sort();
                let mut intensities: Vec<String> = o
                    .intensities
                    .iter()
                    .map(|(channel, i)| format!("{channel} {i:?}"))
                    .collect();
                intensities.sort();
                let mask: String = o
                    .mask_data
                    .iter()
                    .map(|bit| if *bit { '1' } else { '0' })
                    .collect();
                format!(
                    "{classes:?} {:?} {} {mask} {partners:?} {parents:?} {intensities:?}",
                    o.bbox, o.area
                )
            })
            .collect();
        described.sort();
        described
    }

    /// The same image and project give the very same objects in every run
    /// and with any number of threads - also where steps meet ties (object
    /// ids are random, so nothing may depend on them).
    #[test]
    fn a_run_gives_the_same_objects_every_time_and_with_any_number_of_threads() {
        let dir = tempfile::tempdir().unwrap();
        let first = tie_run(dir.path(), 1);
        assert!(
            first
                .iter()
                .filter(|o| o.starts_with("[\"Global(2)\"]"))
                .count()
                == 12,
            "one Voronoi cell per spot"
        );
        assert!(
            first.iter().any(|o| o.starts_with("[\"Global(4)\"]")),
            "the bars' overlaps with their cells"
        );
        for (run, threads) in [4, 4, 1, 8].into_iter().enumerate() {
            assert_eq!(
                tie_run(dir.path(), threads),
                first,
                "run {} with {threads} threads",
                run + 2
            );
        }
    }

    /// `make_single_image_job` plus a [`RecordingStep`] as dataset step.
    fn single_image_job_with_dataset_step(
        fail: bool,
    ) -> (
        JobExecutor,
        Arc<Mutex<Vec<ObjectMetricSettings>>>,
        Arc<Mutex<Vec<usize>>>,
    ) {
        let out_objects = Arc::new(Mutex::new(Vec::new()));
        let mut job = make_single_image_job(out_objects.clone());
        let seen = Arc::new(Mutex::new(Vec::new()));
        job.add_dataset_step(Box::new(crate::dataset::test_support::RecordingStep {
            seen: seen.clone(),
            fail,
        }));
        (job, out_objects, seen)
    }

    #[test]
    fn dataset_steps_run_once_after_the_images_on_every_exported_object() {
        let (job, out_objects, seen) = single_image_job_with_dataset_step(false);
        let (tx, _rx) = std::sync::mpsc::channel();
        job.run(1, tx, Arc::new(AtomicBool::new(false))).unwrap();

        let exported = out_objects.lock().unwrap().len();
        assert!(exported > 0, "the fixture image has objects");
        assert_eq!(*seen.lock().unwrap(), [exported]);
    }

    #[test]
    fn a_failing_dataset_step_fails_the_run() {
        let (job, _out_objects, _seen) = single_image_job_with_dataset_step(true);
        let (tx, _rx) = std::sync::mpsc::channel();
        let err = job
            .run(1, tx, Arc::new(AtomicBool::new(false)))
            .unwrap_err();
        assert!(
            format!("{err}").contains("the dataset step failed"),
            "{err}"
        );
    }

    #[test]
    fn a_cancelled_run_skips_the_dataset_steps() {
        let (job, _out_objects, seen) = single_image_job_with_dataset_step(false);
        let (tx, _rx) = std::sync::mpsc::channel();
        assert!(job.run(1, tx, Arc::new(AtomicBool::new(true))).is_err());
        assert!(seen.lock().unwrap().is_empty(), "no partial results");
    }

    #[test]
    fn run_on_a_real_fixture_image_writes_extracted_objects_through_the_exporter() {
        let out_objects = Arc::new(Mutex::new(Vec::new()));
        let job = make_single_image_job(out_objects.clone());

        let (tx, rx) = std::sync::mpsc::channel();
        let result = job.run(1, tx, Arc::new(AtomicBool::new(false)));
        let events: Vec<ProgressEvent> = rx.into_iter().collect();

        result.expect("threshold -> connected components -> extract pipeline should succeed");
        assert!(
            events
                .iter()
                .any(|e| matches!(e, ProgressEvent::ImageCompleted { .. })),
            "expected an ImageCompleted event for the single processed image"
        );
        assert!(
            !events
                .iter()
                .any(|e| matches!(e, ProgressEvent::ImageFailed { .. })),
            "no image should have failed"
        );
        let exported = out_objects.lock().unwrap();
        assert!(
            !exported.is_empty(),
            "thresholding the whole 8-bit range should extract at least one object"
        );
        // Regression guard: each tile's objects used to be exported
        // immediately *and* re-exported again via the whole-image cache
        // after the whole-image phase - every object doubled, and anything
        // `TileMerge` actually merged left its stale pre-merge fragments
        // behind too. There must be exactly one exported row per id now.
        let unique_ids: std::collections::HashSet<_> =
            exported.iter().map(|o| o.id.clone()).collect();
        assert_eq!(
            unique_ids.len(),
            exported.len(),
            "every exported object id must appear exactly once, not be duplicated \
             across the per-tile and whole-image export paths"
        );
    }

    fn make_multi_image_job(
        out_objects: Arc<Mutex<Vec<ObjectMetricSettings>>>,
        rel_paths: Vec<PathBuf>,
    ) -> JobExecutor {
        let mut series = BTreeMap::new();
        series.insert(0, SeriesSettings::default());
        let mut images = IndexMap::new();
        for rel_path in rel_paths {
            images.insert(
                rel_path.clone(),
                ImageEntry {
                    rel_path,
                    file_size: 0,
                    selected_series: 0,
                    series: series.clone(),
                },
            );
        }

        let mut job = JobExecutor::new(
            PathBuf::new(),
            std::env::temp_dir(),
            images,
            PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR"), "/tests")),
            GlobalImageSettings::default(),
            Arc::new(MemoryExporter {
                out_objects,
                ..Default::default()
            }),
            None,
            Default::default(),
        );
        job.add_pre_process_pipeline(threshold_connected_components_extract_pipeline());
        job
    }

    #[test]
    fn run_still_processes_the_rest_of_the_batch_after_one_image_fails() {
        let out_objects = Arc::new(Mutex::new(Vec::new()));
        let valid = PathBuf::from("multi-channel-4D-series.ome.tif");
        let invalid = PathBuf::from("does-not-exist.ome.tif");
        let job = make_multi_image_job(out_objects.clone(), vec![valid.clone(), invalid.clone()]);

        let (tx, rx) = std::sync::mpsc::channel();
        // Single-threaded so the valid image can't just happen to finish
        // before the invalid one is even scheduled - this forces both to be
        // attempted regardless of order.
        let result = job.run(1, tx, Arc::new(AtomicBool::new(false)));
        let events: Vec<ProgressEvent> = rx.into_iter().collect();

        let err = result.expect_err("expected an error since one of the two images failed");
        let err_msg = err.to_string();
        assert!(
            err_msg.contains("does-not-exist.ome.tif"),
            "error should name the failing image, got: {err_msg}"
        );

        assert!(
            events
                .iter()
                .any(|e| matches!(e, ProgressEvent::ImageCompleted { path, .. } if path == &valid)),
            "the valid image must still complete even though another image in the batch failed"
        );
        assert!(
            events
                .iter()
                .any(|e| matches!(e, ProgressEvent::ImageFailed { path } if path == &invalid)),
            "the invalid image must be reported as failed"
        );
        assert!(
            !out_objects.lock().unwrap().is_empty(),
            "the valid image's objects should still have been exported, not dropped because \
             another image in the batch failed"
        );
    }

    #[test]
    fn run_on_a_single_invalid_image_names_it_in_the_propagated_error() {
        // Mirrors `run_on_a_real_fixture_image_writes_extracted_objects_through_the_exporter`
        // but with a nonexistent path, exercising the single-image branch's
        // *error* arm (the success arm is already covered by that test).
        let out_objects = Arc::new(Mutex::new(Vec::new()));
        let job = make_multi_image_job(
            out_objects.clone(),
            vec![PathBuf::from("does-not-exist.ome.tif")],
        );

        let (tx, rx) = std::sync::mpsc::channel();
        let result = job.run(1, tx, Arc::new(AtomicBool::new(false)));
        let events: Vec<ProgressEvent> = rx.into_iter().collect();

        let err = result.expect_err("the only image failed, so the whole run must fail");
        assert!(
            err.to_string().contains("does-not-exist.ome.tif"),
            "error should name the failing image, got: {err}"
        );
        assert!(
            events
                .iter()
                .any(|e| matches!(e, ProgressEvent::ImageFailed { .. }))
        );
        assert!(out_objects.lock().unwrap().is_empty());
    }

    #[test]
    fn run_on_multiple_valid_images_succeeds_and_exports_every_images_objects() {
        // `run_still_processes_the_rest_of_the_batch_after_one_image_fails`
        // covers the multi-image branch's failure-aggregation path; this
        // covers its `Ok(())` path (every image in the batch succeeding).
        let out_objects = Arc::new(Mutex::new(Vec::new()));
        let a = PathBuf::from("multi-channel-4D-series.ome.tif");
        let b = PathBuf::from("slice_Z0_C0_T0.tif");
        let job = make_multi_image_job(out_objects.clone(), vec![a.clone(), b.clone()]);

        let (tx, rx) = std::sync::mpsc::channel();
        let result = job.run(2, tx, Arc::new(AtomicBool::new(false)));
        let events: Vec<ProgressEvent> = rx.into_iter().collect();

        result.expect("both images are valid, the whole batch should succeed");
        for path in [&a, &b] {
            assert!(
                events.iter().any(
                    |e| matches!(e, ProgressEvent::ImageCompleted { path: p, .. } if p == path)
                ),
                "{path:?} should have completed"
            );
        }
        assert!(
            !events
                .iter()
                .any(|e| matches!(e, ProgressEvent::ImageFailed { .. }))
        );
        assert!(!out_objects.lock().unwrap().is_empty());
    }

    // -- Breakpoint handling through the batch (multi-image, `analyze_image`'s
    // `progress: None`) path -- `analyze_image` used to have its own simpler
    // `bp_hit`-only breakpoint check before it was unified with the
    // interactive path's richer stop/snapshot capture handling; these two
    // tests are the regression guard that the unification didn't change
    // either mode's observable behavior for a batch run, since `Pipeline::run`
    // guarantees `breakpoint_hit == true` only for `BreakpointMode::Stop`
    // (`pipeline.rs`'s early-return branch) - `Snapshot` always falls through
    // to the normal completion path with `breakpoint_hit: false`.

    // ---- breakpoint capture ----

    fn tile(x: usize, y: usize) -> ImageTile {
        ImageTile {
            offset_x: x,
            offset_y: y,
            width: 512,
            height: 512,
        }
    }

    fn viewport(offset_x: f32, offset_y: f32) -> PreviewTileSettings {
        PreviewTileSettings {
            offset_x,
            offset_y,
            viewport_width: 400.0,
            viewport_height: 400.0,
            zoom: 1.0,
            process_all_tiles: false,
        }
    }

    #[test]
    fn a_breakpoint_on_several_tiles_takes_the_nearest_visible_ones_first() {
        let tiles = [tile(0, 0), tile(512, 0), tile(0, 512), tile(512, 512)];
        // Centre at (700, 650): bottom-right nearest, then top-right
        // (its centre 106 px further), then bottom-left.
        let settings = viewport(200.0 - 700.0, 200.0 - 650.0);
        assert_eq!(
            JobExecutor::breakpoint_target_tiles(&tiles, Some(&settings), 3),
            [(512, 512), (512, 0), (0, 512)]
        );
        // More wanted than there are: all of them, once.
        assert_eq!(
            JobExecutor::breakpoint_target_tiles(&tiles, Some(&settings), 9).len(),
            4
        );
        // 0 means 1.
        assert_eq!(
            JobExecutor::breakpoint_target_tiles(&tiles, Some(&settings), 0).len(),
            1
        );
    }

    #[test]
    fn the_breakpoint_captures_the_visible_tile_closest_to_the_viewport_centre() {
        let tiles = [tile(0, 0), tile(512, 0), tile(0, 512), tile(512, 512)];
        // The viewport (400x400 at zoom 1) panned so its centre lies at image
        // pixel (700, 650): over the bottom-right tile, with others visible too.
        let settings = viewport(200.0 - 700.0, 200.0 - 650.0);
        assert_eq!(
            JobExecutor::breakpoint_target_tiles(&tiles, Some(&settings), 1)
                .first()
                .copied(),
            Some((512, 512))
        );
    }

    #[test]
    fn the_breakpoint_falls_back_sensibly_without_a_usable_viewport() {
        let tiles = [tile(0, 0), tile(512, 0), tile(0, 512), tile(512, 512)];
        // No viewport settings: the middle tile of the list.
        assert_eq!(
            JobExecutor::breakpoint_target_tiles(&tiles, None, 1)
                .first()
                .copied(),
            Some((0, 512))
        );
        // Viewport far away from the image: the closest of all tiles.
        let away = viewport(-5000.0, -5000.0);
        assert_eq!(
            JobExecutor::breakpoint_target_tiles(&tiles, Some(&away), 1)
                .first()
                .copied(),
            Some((512, 512))
        );
        // A single visible tile is taken as is.
        let only_first = viewport(0.0, 0.0);
        assert_eq!(
            JobExecutor::breakpoint_target_tiles(&tiles[..1], Some(&only_first), 1)
                .first()
                .copied(),
            Some((0, 0))
        );
        assert_eq!(
            JobExecutor::breakpoint_target_tiles(&[], None, 1)
                .first()
                .copied(),
            None
        );
    }

    /// The id of pipeline 1's first step (the Threshold of
    /// `threshold_connected_components_extract_pipeline`).
    fn threshold_step(job: &JobExecutor) -> evanalyzer_cfg::core_types::StepId {
        job.pipelines_pre_process[&PipelineId(1)].commands[0].0
    }

    fn breakpoint_events(mode: BreakpointMode) -> (Vec<ProgressEvent>, usize) {
        let out_objects = Arc::new(Mutex::new(Vec::new()));
        let mut job = make_single_image_job(out_objects.clone());
        job.breakpoint = Some(BreakpointSettings {
            pipeline_id: PipelineId(1),
            step_id: threshold_step(&job), // right after Threshold
            mode,
        });
        let (tx, rx) = std::sync::mpsc::channel();
        job.run(1, tx, Arc::new(AtomicBool::new(false)))
            .expect("a breakpoint is not an error");
        let events = rx.into_iter().collect();
        let written = out_objects.lock().unwrap().len();
        (events, written)
    }

    fn reached(events: &[ProgressEvent]) -> Vec<&ProgressEvent> {
        events
            .iter()
            .filter(|e| matches!(e, ProgressEvent::BreakpointReached { .. }))
            .collect()
    }

    #[test]
    fn a_stop_breakpoint_reports_the_captured_tile_and_writes_nothing() {
        let (events, written) = breakpoint_events(BreakpointMode::Stop);
        let reached = reached(&events);
        assert_eq!(reached.len(), 1, "exactly one tile sends the capture");
        let ProgressEvent::BreakpointReached {
            segmentation,
            tile_offset_x,
            tile_offset_y,
            tile_width,
            tile_height,
            ..
        } = reached[0]
        else {
            unreachable!()
        };
        assert!(segmentation.is_some(), "captured after Threshold");
        assert_eq!((*tile_offset_x, *tile_offset_y), (0, 0));
        assert_eq!((*tile_width, *tile_height), (439, 167));
        assert_eq!(written, 0, "the pipeline stopped before Extract Objects");
    }

    #[test]
    fn a_snapshot_breakpoint_reports_the_capture_and_still_writes_the_objects() {
        let (events, written) = breakpoint_events(BreakpointMode::Snapshot);
        assert_eq!(reached(&events).len(), 1);
        assert!(written > 0, "a snapshot lets the pipeline run to the end");
    }

    #[test]
    fn breakpoint_stop_during_a_batch_run_skips_the_write_but_the_image_still_completes() {
        let out_objects = Arc::new(Mutex::new(Vec::new()));
        let a = PathBuf::from("multi-channel-4D-series.ome.tif");
        let b = PathBuf::from("slice_Z0_C0_T0.tif");
        let mut job = make_multi_image_job(out_objects.clone(), vec![a.clone(), b.clone()]);
        job.breakpoint = Some(BreakpointSettings {
            pipeline_id: PipelineId(1),
            step_id: threshold_step(&job), // right after Threshold, before ConnectedComponents/ExtractObjects
            mode: BreakpointMode::Stop,
        });

        let (tx, rx) = std::sync::mpsc::channel();
        // 2 images -> total > 1 -> the batch path, not the single-image
        // interactive path that already had this behavior.
        let result = job.run(2, tx, Arc::new(AtomicBool::new(false)));
        let events: Vec<ProgressEvent> = rx.into_iter().collect();

        result.expect("a breakpoint hit must not surface as a batch run error");
        assert!(
            out_objects.lock().unwrap().is_empty(),
            "the pipeline stopped before ExtractObjects ever ran, so nothing should have been written"
        );
        for path in [&a, &b] {
            assert!(
                events.iter().any(
                    |e| matches!(e, ProgressEvent::ImageCompleted { path: p, .. } if p == path)
                ),
                "{path:?} should still report as completed, not failed, when it only hit a Stop breakpoint"
            );
        }
    }

    #[test]
    fn breakpoint_snapshot_during_a_batch_run_does_not_skip_the_write() {
        let out_objects = Arc::new(Mutex::new(Vec::new()));
        let a = PathBuf::from("multi-channel-4D-series.ome.tif");
        let b = PathBuf::from("slice_Z0_C0_T0.tif");
        let mut job = make_multi_image_job(out_objects.clone(), vec![a, b]);
        job.breakpoint = Some(BreakpointSettings {
            pipeline_id: PipelineId(1),
            step_id: threshold_step(&job),
            mode: BreakpointMode::Snapshot,
        });

        let (tx, rx) = std::sync::mpsc::channel();
        let result = job.run(2, tx, Arc::new(AtomicBool::new(false)));
        let _events: Vec<ProgressEvent> = rx.into_iter().collect();

        result.expect("snapshot breakpoints must not fail the run");
        assert!(
            !out_objects.lock().unwrap().is_empty(),
            "snapshot mode must still run the pipeline to completion and write results, \
             unlike stop mode"
        );
    }

    #[test]
    fn run_on_multiple_images_stops_early_when_already_cancelled() {
        // Setting `cancel` before the run starts must stop every per-image
        // task from doing any work at all (the multi-image branch's
        // per-item `if cancel.load(...) { return; }` check), and the run
        // must report `Cancelled` rather than treating "nothing failed" as
        // success.
        let out_objects = Arc::new(Mutex::new(Vec::new()));
        let a = PathBuf::from("multi-channel-4D-series.ome.tif");
        let b = PathBuf::from("slice_Z0_C0_T0.tif");
        let job = make_multi_image_job(out_objects.clone(), vec![a, b]);

        let (tx, rx) = std::sync::mpsc::channel();
        let cancel = Arc::new(AtomicBool::new(true));
        let result = job.run(2, tx, cancel);
        let events: Vec<ProgressEvent> = rx.into_iter().collect();

        assert!(matches!(result, Err(InternalErrors::Cancelled)));
        assert!(
            !events
                .iter()
                .any(|e| matches!(e, ProgressEvent::ImageCompleted { .. })
                    || matches!(e, ProgressEvent::ImageFailed { .. })),
            "a pre-cancelled run must not process any image"
        );
        assert!(out_objects.lock().unwrap().is_empty());
    }

    #[test]
    fn run_on_a_single_image_reports_cancelled_unwrapped() {
        // The single-image branch used to wrap every error, `Cancelled`
        // included, into `Internal("<path>: Cancelled")` - so cancelling a
        // preview showed up as a failure instead of "cancelled by user".
        let job = make_single_image_job(Arc::new(Mutex::new(Vec::new())));

        let (tx, rx) = std::sync::mpsc::channel();
        let result = job.run(1, tx, Arc::new(AtomicBool::new(true)));
        let events: Vec<ProgressEvent> = rx.into_iter().collect();

        assert!(
            matches!(result, Err(InternalErrors::Cancelled)),
            "{result:?}"
        );
        assert!(
            !events
                .iter()
                .any(|e| matches!(e, ProgressEvent::ImageFailed { .. })),
            "a cancelled image must not be reported as failed"
        );
    }

    #[test]
    fn count_preview_visible_tiles_matches_the_real_fixture_images_single_tile_grid() {
        let job = make_single_image_job(Arc::new(Mutex::new(Vec::new())));

        // The fixture image is far smaller than the 4096px base tile size
        // used when no preview settings are set, so it must resolve to
        // exactly one tile.
        let count = job.count_preview_visible_tiles().unwrap();
        assert_eq!(count, 1);
    }

    /// A filled 3×3 annotation of the plane (`z`, `t`), carrying `class`.
    fn annotation_on_plane(z: i32, t: i32, class: ObjectClass) -> ObjectMetricSettings {
        let mut object = crate::object::Object::new(crate::object::ObjectInit {
            id: ObjectId(800_000 + (z * 10 + t) as u128),
            bbox: [1, 1, 3, 3],
            mask_data: bitvec::bitvec![u64, bitvec::order::Lsb0; 1; 9],
            area: 9,
            ..Default::default()
        });
        object.add_object_class(class, Scope::Global).unwrap();
        let mut settings = object.to_object_settings();
        settings.z_stack = z;
        settings.t_stack = t;
        settings
    }

    #[test]
    fn load_annotated_objects_runs_without_segmentation_and_only_loads_the_analyzed_plane() {
        use crate::algos::LoadAnnotatedObjects;
        const ANALYZED: ObjectClass = ObjectClass::Global(1);
        const OTHER_Z: ObjectClass = ObjectClass::Global(2);
        const OTHER_T: ObjectClass = ObjectClass::Global(3);
        const LOADED: ObjectClass = ObjectClass::Global(4);

        let out_objects = Arc::new(Mutex::new(Vec::new()));
        let rel_path = PathBuf::from("multi-channel-4D-series.ome.tif");
        let mut series = BTreeMap::new();
        series.insert(
            0,
            SeriesSettings {
                // Default stack settings analyze z = 0, t = 0 only.
                objects: vec![
                    annotation_on_plane(0, 0, ANALYZED),
                    annotation_on_plane(1, 0, OTHER_Z),
                    annotation_on_plane(0, 1, OTHER_T),
                ],
                ..Default::default()
            },
        );
        let mut images = IndexMap::new();
        images.insert(
            rel_path.clone(),
            ImageEntry {
                rel_path,
                file_size: 0,
                selected_series: 0,
                series,
            },
        );
        let mut job = JobExecutor::new(
            PathBuf::new(),
            std::env::temp_dir(),
            images,
            PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR"), "/tests")),
            GlobalImageSettings::default(),
            Arc::new(MemoryExporter {
                out_objects: out_objects.clone(),
                ..Default::default()
            }),
            None,
            Default::default(),
        );
        // Built like `job_generator` does for a pipeline holding only this
        // command: an empty tile pipeline (nothing is segmented) plus the
        // whole-image one. The annotations alone must be enough for the
        // whole-image phase to run.
        job.add_pre_process_pipeline(Pipeline::new(
            PipelineId(1),
            CorePipelineSettings {
                start_image: ImageAddress::Channel(0),
            },
        ));
        let mut pipeline = Pipeline::new(
            PipelineId(1),
            CorePipelineSettings {
                start_image: ImageAddress::Scratchpad,
            },
        );
        pipeline.add_command(Box::new(LoadAnnotatedObjects {
            input_classes: vec![],
            output_class: LOADED,
            keep_annotated_classes: true,
        }));
        job.add_post_process_pipeline(pipeline);

        let (tx, _rx) = std::sync::mpsc::channel();
        job.run(1, tx, Arc::new(AtomicBool::new(false)))
            .expect("loading annotations should succeed");

        let exported = out_objects.lock().unwrap();
        assert_eq!(
            exported.len(),
            1,
            "only the z = 0, t = 0 annotation is loaded"
        );
        let object = &exported[0];
        assert!(object.object_class.contains(&ANALYZED));
        assert!(object.object_class.contains(&LOADED));
        assert_eq!(
            object.segmentation_class,
            SegmentationClass::MANUAL_ANNOTATED
        );
        assert_eq!(object.area, 9);
        assert!(
            !object.intensities.is_empty(),
            "the loaded annotation is measured on the image"
        );
    }
}

/// The plan's strongest correctness check for `docs/tile_merge_plan.md`:
/// a synthetic object deliberately split across a tile boundary must produce,
/// after buffering + merging, the exact same area/bbox/intensity as running
/// the identical pipeline on the same object *without* tiling at all.
///
/// Runs the real `Threshold -> ConnectedComponents -> ExtractObjects` chain
/// (via `Pipeline::run`, bypassing file I/O) so this exercises the actual
/// extraction path's `touches_edge`/bbox/intensity output feeding into
/// `tile_merge`, not just hand-built `Object`s like `tile_merge`'s own unit
/// tests. Doesn't go through `JobExecutor::analyze_image` itself (that needs
/// a real multi-tile-sized image file on disk) - this instead replicates
/// exactly what `analyze_image`'s tile loop does: run each tile's pipeline,
/// split out tile-edge-touching fragments, buffer them, and merge once every
/// tile is done.
#[cfg(test)]
mod tile_merge_end_to_end_tests {
    use super::*;
    use crate::ImagePlane;
    use crate::Object;
    use crate::algos::TileMerge;
    use crate::algos::Voronoi;
    use crate::algos::touches_tile_edge;
    use crate::algos::{
        ConnectedComponents, Connectivity, ExtractObjects, Threshold, ThresholdEntry,
        ThresholdMethod, ThresholdValueSource,
    };
    use crate::image::Point2d;
    use crate::image::{ImageContainer, ManagedImage};
    use crate::pipeline::image_cache::ImageCache;
    use crate::pipeline::pipeline::CorePipelineSettings;
    use evanalyzer_cfg::core_types::{ObjectClass, PixelUnits, SegmentationClass, SizeUnits};
    use kornia_image::Image;

    fn threshold_connected_components_extract_pipeline() -> Pipeline {
        let mut pipeline = Pipeline::new(
            PipelineId(1),
            CorePipelineSettings {
                start_image: ImageAddress::Channel(0),
            },
        );
        // A project step (id 1): what the breakpoint tests target.
        pipeline.add_step(
            evanalyzer_cfg::core_types::StepId(1),
            Scope::Global,
            Box::new(Threshold {
                thresholds: vec![ThresholdEntry {
                    method: ThresholdMethod::Manual,
                    min_threshold: 0.0,
                    max_threshold: 255.0,
                    unit: PixelUnits::Bit,
                    segmentation_class: SegmentationClass(1),
                    value_source: ThresholdValueSource::ActualImage,
                }],
            }),
        );
        pipeline.add_command(Box::new(ConnectedComponents { min_size: 0 }));
        pipeline.add_command(Box::new(ExtractObjects {
            max_objects_before_fail: 100_000,
            class_mapping: vec![],
        }));
        pipeline
    }

    /// A `PipelineCache` seeded with a single-channel gray image, as if it
    /// were tile `(tile_offset_x, 0)` of a `(full_width, full_height)` image
    /// - everything `Threshold`/`ConnectedComponents`/`ExtractObjects` need
    /// to run exactly as `job_executor` would run them on a real tile.
    fn gray_tile_cache(
        data: Vec<f32>,
        tile_width: usize,
        tile_height: usize,
        tile_offset_x: usize,
        full_width: usize,
        full_height: usize,
    ) -> GlobalPipelineCache {
        let image = Image::<f32, 1>::new(
            ImageSize {
                width: tile_width,
                height: tile_height,
            },
            data,
        )
        .unwrap();
        let container = Arc::new(ImageContainer::F32Gray(ManagedImage {
            data: image,
            tile_offset: Point2d {
                x: tile_offset_x,
                y: 0,
            },
            plane: Some(ImagePlane { z: 0, c: 0, t: 0 }),
        }));
        let mut images: ImageCache = ImageCache::new().unwrap();
        images.insert(
            CacheAddress::Channel((
                0,
                ImageTile {
                    offset_x: tile_offset_x,
                    offset_y: 0,
                    width: tile_width,
                    height: tile_height,
                },
            )),
            container,
        );
        GlobalPipelineCache {
            image_cache: images,

            object_cache: Default::default(),
            measurements: Default::default(),
            image_rel_path: PathBuf::new(),
            annotated_objects: Default::default(),
            class_colors: Default::default(),
            image_meta: GlobalImageMeta {
                full_image_width: ImageSize {
                    width: full_width,
                    height: full_height,
                },
                is_rgb: false,
                nr_of_bits: 8,
                pixel_sizes: PixelSizes::default(),
            },
        }
    }

    /// `tile` must match the `ImageTile` key `gray_tile_cache` stored the
    /// channel image under - `run_commands`'s `None` fallback assumes the
    /// tile covers the *whole* image (`cache.image_meta.full_image_width`),
    /// which is only true for the untiled reference run, not for a real
    /// sub-tile like `tile_a`/`tile_b` below.
    fn run_pipeline(cache: GlobalPipelineCache, tile: ImageTile) -> GlobalPipelineCache {
        threshold_connected_components_extract_pipeline()
            .run_commands(PathBuf::new(), Some(tile), cache, None, false)
            .expect("pipeline must run successfully")
            .cache
    }

    /// Splits every tile-edge-touching, merge-eligible object out of `cache`
    /// - the same filter `analyze_image`'s tile loop applies before
    /// `cache_tx.try_send`. Every class merges unless it's on
    /// `classes_to_not_merge`. Each `Object` already carries its own
    /// `source_tile` (set by `ExtractObjects`), so unlike the old
    /// `PendingFragment` wrapper this just returns plain objects.
    fn split_fragments(
        cache: &mut GlobalPipelineCache,
        tile: &ImageTile,
        classes_to_not_merge: &[ObjectClass],
    ) -> Vec<Object> {
        let ids: Vec<_> = cache
            .object_cache
            .iter()
            .filter(|(_, object)| {
                !object.touches_edge
                    && touches_tile_edge(object.bbox, tile)
                    && !object
                        .object_classes()
                        .iter()
                        .any(|c| classes_to_not_merge.contains(c))
            })
            .map(|(id, _)| id.clone())
            .collect();
        ids.into_iter()
            .filter_map(|id| cache.object_cache.remove(&id))
            .collect()
    }

    /// 6x3 image, one horizontal 4-pixel-wide object at row y=1, columns
    /// x=1..=4 - entirely interior to the full image (doesn't touch its
    /// edges), but split by a tile boundary at x=3 into a 2px piece in each
    /// of two 3-wide tiles. Reference: process the whole image as one tile.
    fn whole_image_data() -> Vec<f32> {
        #[rustfmt::skip]
        let data = vec![
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 1.0, 1.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ];
        data
    }

    #[test]
    fn tiled_merge_reproduces_the_untiled_reference_object() {
        // Reference: single tile covering the whole image.
        let reference_cache = run_pipeline(
            gray_tile_cache(whole_image_data(), 6, 3, 0, 6, 3),
            ImageTile {
                offset_x: 0,
                offset_y: 0,
                width: 6,
                height: 3,
            },
        );
        let reference_objects: Vec<_> = reference_cache.object_cache.values().collect();
        assert_eq!(
            reference_objects.len(),
            1,
            "the untiled reference must find exactly one object"
        );
        let reference = reference_objects[0];
        assert_eq!(reference.bbox, [1, 1, 4, 1]);
        assert_eq!(reference.area, 4);
        let reference_sum = reference.intensities.get(&0).unwrap().sum_intensity;

        // Tiled: split the same image into two 3-wide tiles at x=3, process
        // each independently (exactly like two rayon workers would).
        let full = whole_image_data();
        let mut tile_a_data = Vec::with_capacity(9);
        let mut tile_b_data = Vec::with_capacity(9);
        for y in 0..3 {
            let row = &full[y * 6..(y + 1) * 6];
            tile_a_data.extend_from_slice(&row[0..3]);
            tile_b_data.extend_from_slice(&row[3..6]);
        }
        let tile_a = ImageTile {
            offset_x: 0,
            offset_y: 0,
            width: 3,
            height: 3,
        };
        let tile_b = ImageTile {
            offset_x: 3,
            offset_y: 0,
            width: 3,
            height: 3,
        };
        let mut cache_a = run_pipeline(gray_tile_cache(tile_a_data, 3, 3, 0, 6, 3), tile_a);
        let mut cache_b = run_pipeline(gray_tile_cache(tile_b_data, 3, 3, 3, 6, 3), tile_b);

        assert_eq!(
            cache_a.object_cache.len(),
            1,
            "tile A must find one fragment"
        );
        assert_eq!(
            cache_b.object_cache.len(),
            1,
            "tile B must find one fragment"
        );
        let frag_a_bbox = cache_a.object_cache.values().next().unwrap().bbox;
        let frag_b_bbox = cache_b.object_cache.values().next().unwrap().bbox;
        assert_eq!(
            frag_a_bbox,
            [1, 1, 2, 1],
            "tile A's fragment is its half of the blob"
        );
        assert_eq!(
            frag_b_bbox,
            [3, 1, 4, 1],
            "tile B's fragment is its half of the blob"
        );

        // No `classes_to_not_merge` entries - every class merges by default.
        let classes_to_not_merge: [ObjectClass; 0] = [];
        let mut pending = split_fragments(&mut cache_a, &tile_a, &classes_to_not_merge);
        pending.extend(split_fragments(
            &mut cache_b,
            &tile_b,
            &classes_to_not_merge,
        ));
        assert_eq!(
            pending.len(),
            2,
            "both tile-edge fragments must have been buffered, not exported per-tile"
        );
        assert!(
            cache_a.object_cache.is_empty() && cache_b.object_cache.is_empty(),
            "buffered fragments must be removed from their tile's own export"
        );

        let mut whole_image_cache = GlobalPipelineCache {
            object_cache: pending.into_iter().map(|o| (o.id.clone(), o)).collect(),
            ..Default::default()
        };
        // The tiles' channels too, as `analyze_image`'s `merge_caches` folds
        // them in - Tile Merge measures the merged object on them.
        whole_image_cache.image_cache.extend(cache_a.image_cache);
        whole_image_cache.image_cache.extend(cache_b.image_cache);
        let mut tile_merge_pipeline = Pipeline::new(
            PipelineId(0),
            CorePipelineSettings {
                start_image: ImageAddress::Scratchpad,
            },
        );
        tile_merge_pipeline.add_command(Box::new(TileMerge {
            classes_to_not_merge: Vec::new(),
            connectivity: Connectivity::EightConnected,
            max_fragments_per_group: 100,
        }));
        whole_image_cache = tile_merge_pipeline
            .run_commands(PathBuf::new(), None, whole_image_cache, None, false)
            .unwrap()
            .cache;

        let merged: Vec<_> = whole_image_cache.object_cache.values().collect();
        assert_eq!(
            merged.len(),
            1,
            "the two fragments must merge into one object"
        );
        assert_eq!(
            merged[0].bbox, reference.bbox,
            "merged bbox must match the untiled reference exactly"
        );
        assert_eq!(
            merged[0].area, reference.area,
            "merged area must match the untiled reference exactly"
        );
        assert_eq!(
            merged[0].intensities.get(&0).unwrap().sum_intensity,
            reference_sum,
            "merged intensity sum must match the untiled reference exactly"
        );
        let spread = |i: &crate::object::Intensity| {
            [
                i.std_intensity,
                i.median_intensity,
                i.lower_quartile_intensity,
                i.upper_quartile_intensity,
                i.mad_intensity,
            ]
        };
        let (got, want) = (
            merged[0].intensities.get(&0).unwrap(),
            reference.intensities.get(&0).unwrap(),
        );
        assert!(got.has_spread());
        assert_eq!(
            spread(got),
            spread(want),
            "merged intensity spread (median, quartiles, ...) must match the untiled reference"
        );
    }

    /// Runs the real whole-image chain `TileMerge -> Voronoi` (in that
    /// order, matching `get_execution_order`'s system-inserted placement of
    /// `TileMerge` first) against `cache`.
    fn run_tile_merge_and_voronoi(
        cache: GlobalPipelineCache,
        voronoi: Voronoi,
    ) -> GlobalPipelineCache {
        let mut whole_image_pipeline = Pipeline::new(
            PipelineId(0),
            CorePipelineSettings {
                start_image: ImageAddress::Scratchpad,
            },
        );
        whole_image_pipeline.add_command(Box::new(TileMerge {
            classes_to_not_merge: Vec::new(),
            connectivity: Connectivity::EightConnected,
            max_fragments_per_group: 100,
        }));
        whole_image_pipeline.add_command(Box::new(voronoi));
        whole_image_pipeline
            .run_commands(PathBuf::new(), None, cache, None, false)
            .unwrap()
            .cache
    }

    const CENTER_CLASS: ObjectClass = ObjectClass::Global(1);
    const VORONOI_CLASS: ObjectClass = ObjectClass::Global(2);

    fn default_voronoi() -> Voronoi {
        Voronoi {
            centers: CENTER_CLASS,
            center_filter_classes: Vec::new(),
            mask: ObjectClass::Unset,
            mask_filter_classes: Vec::new(),
            output_class: VORONOI_CLASS,
            unit: SizeUnits::Pixels,
            max_radius: 0.0,
            exclude_areas_at_the_edges: false,
            exclude_areas_with_no_center: false,
        }
    }

    /// 12x6 image with two objects: a 4px blob at row y=1 that a tile split
    /// at x=6 cuts into two 2px fragments (like `whole_image_data` above,
    /// just wider so there's room for a second, untouched object), and a
    /// separate 2px blob at row y=4 entirely inside the first tile - a
    /// second Voronoi seed the tessellation has to correctly divide space
    /// with once the first object is properly reconstructed.
    fn two_object_image_data() -> Vec<f32> {
        // Centers land at (5.5, 1) and (1.5, 4) - chosen so their
        // perpendicular bisector never crosses an exact integer pixel
        // coordinate, which would otherwise make a pixel's nearest-center
        // assignment an exact tie. `CenterKdTree`'s tie-break is "lowest
        // index in the `centers` Vec wins", and that index depends on
        // `cache.object_cache`'s (an `IndexMap`) iteration order - which
        // differs between the untiled reference (objects inserted directly)
        // and the tiled run (fragments buffered, merged, then folded back
        // in), even though both produce geometrically identical objects. An
        // exact tie is therefore genuinely ambiguous, not a correctness bug
        // in either `TileMerge` or `Voronoi` - avoiding one entirely keeps
        // this test about the actual claim under test.
        #[rustfmt::skip]
        let data = vec![
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ];
        data
    }

    /// End-to-end proof of the paper-facing claim that Voronoi tessellation
    /// is computed *correctly* for whole-slide (tiled) images, not merely
    /// that it runs once whole-image-scoped: drives the real
    /// `Threshold -> ConnectedComponents -> ExtractObjects` per-tile
    /// extraction, `TileMerge`, and `Voronoi` exactly as `job_executor`
    /// would for a real analysis run, and checks the resulting regions
    /// against an untiled reference of the same full chain - not just that
    /// the merged *object* matches (already proven by
    /// `tiled_merge_reproduces_the_untiled_reference_object`), but that a
    /// downstream whole-image algorithm consuming it produces identical
    /// output too.
    #[test]
    fn tiled_pipeline_produces_the_same_voronoi_tessellation_as_the_untiled_reference() {
        // Reference: single tile covering the whole image.
        let reference_objects_cache = run_pipeline(
            gray_tile_cache(two_object_image_data(), 12, 6, 0, 12, 6),
            ImageTile {
                offset_x: 0,
                offset_y: 0,
                width: 12,
                height: 6,
            },
        );
        assert_eq!(
            reference_objects_cache.object_cache.len(),
            2,
            "the untiled reference must find exactly two objects"
        );
        let reference_cache =
            run_tile_merge_and_voronoi(reference_objects_cache, default_voronoi());
        let mut reference_regions: Vec<_> = reference_cache
            .object_cache
            .values()
            .filter(|o| o.is_of(&VORONOI_CLASS, Scope::Global))
            .map(|o| (o.bbox, o.area))
            .collect();
        reference_regions.sort();
        assert_eq!(
            reference_regions.len(),
            2,
            "the untiled reference must tessellate into exactly two Voronoi regions"
        );

        // Tiled: split into two 6-wide tiles at x=6, process each
        // independently (exactly like two rayon workers would), then merge.
        let full = two_object_image_data();
        let mut tile_a_data = Vec::with_capacity(36);
        let mut tile_b_data = Vec::with_capacity(36);
        for y in 0..6 {
            let row = &full[y * 12..(y + 1) * 12];
            tile_a_data.extend_from_slice(&row[0..6]);
            tile_b_data.extend_from_slice(&row[6..12]);
        }
        let tile_a = ImageTile {
            offset_x: 0,
            offset_y: 0,
            width: 6,
            height: 6,
        };
        let tile_b = ImageTile {
            offset_x: 6,
            offset_y: 0,
            width: 6,
            height: 6,
        };
        let mut cache_a = run_pipeline(gray_tile_cache(tile_a_data, 6, 6, 0, 12, 6), tile_a);
        let mut cache_b = run_pipeline(gray_tile_cache(tile_b_data, 6, 6, 6, 12, 6), tile_b);
        assert_eq!(
            cache_a.object_cache.len() + cache_b.object_cache.len(),
            3,
            "tile A must find the untouched blob plus a tile-edge fragment, tile B just the fragment"
        );

        let classes_to_not_merge: [ObjectClass; 0] = [];
        let mut pending = split_fragments(&mut cache_a, &tile_a, &classes_to_not_merge);
        pending.extend(split_fragments(
            &mut cache_b,
            &tile_b,
            &classes_to_not_merge,
        ));
        assert_eq!(
            pending.len(),
            2,
            "both tile-edge fragments must have been buffered"
        );

        let mut whole_image_cache = GlobalPipelineCache {
            object_cache: pending.into_iter().map(|o| (o.id.clone(), o)).collect(),
            image_meta: GlobalImageMeta {
                full_image_width: ImageSize {
                    width: 12,
                    height: 6,
                },
                is_rgb: false,
                nr_of_bits: 8,
                pixel_sizes: PixelSizes::default(),
            },
            ..Default::default()
        };
        // The untouched blob (interior to tile A, never buffered) merges
        // back in exactly as `analyze_image`'s own `merge_caches` closure
        // folds each tile's leftover cache into the shared whole-image one.
        whole_image_cache.object_cache.extend(cache_a.object_cache);
        whole_image_cache.object_cache.extend(cache_b.object_cache);
        whole_image_cache.image_cache.extend(cache_a.image_cache);
        whole_image_cache.image_cache.extend(cache_b.image_cache);
        assert_eq!(whole_image_cache.object_cache.len(), 3);

        let tiled_cache = run_tile_merge_and_voronoi(whole_image_cache, default_voronoi());
        let mut tiled_regions: Vec<_> = tiled_cache
            .object_cache
            .values()
            .filter(|o| o.is_of(&VORONOI_CLASS, Scope::Global))
            .map(|o| (o.bbox, o.area))
            .collect();
        tiled_regions.sort();

        assert_eq!(
            tiled_regions, reference_regions,
            "the tiled pipeline's Voronoi tessellation must exactly match the untiled reference"
        );
    }
}

#[cfg(test)]
mod estimate_ram_per_worker_bytes_tests {
    use super::*;
    use crate::storage::memory::MemoryExporter;
    use evanalyzer_cfg::settings::images_settings::{ChannelSettings, SeriesSettings};
    use std::collections::BTreeMap;

    fn image_entry(width: u64, height: u64, channel_count: usize) -> ImageEntry {
        let channels = (0..channel_count as i32)
            .map(|c| (c, ChannelSettings::default()))
            .collect();
        let mut series = BTreeMap::new();
        series.insert(
            0,
            SeriesSettings {
                image_width: width,
                image_height: height,
                channels,
                ..Default::default()
            },
        );
        ImageEntry {
            selected_series: 0,
            series,
            ..Default::default()
        }
    }

    fn job_with(images: IndexMap<PathBuf, ImageEntry>) -> JobExecutor {
        JobExecutor::new(
            PathBuf::new(),
            PathBuf::new(),
            images,
            PathBuf::new(),
            GlobalImageSettings::default(),
            Arc::new(MemoryExporter {
                out_objects: Arc::new(Mutex::new(Vec::new())),
                ..Default::default()
            }),
            None,
            Default::default(),
        )
    }

    #[test]
    fn matches_the_largest_image_in_the_job_not_the_first_or_smallest() {
        let mut images = IndexMap::new();
        images.insert(PathBuf::from("small.tif"), image_entry(512, 512, 1));
        images.insert(PathBuf::from("large.tif"), image_entry(2048, 2048, 4));
        let job = job_with(images);

        let expected = crate::utils::resources::estimate_ram_budget(2048, 2048, 4).total_bytes();
        assert_eq!(job.estimate_ram_per_worker_bytes(), expected);
    }

    #[test]
    fn caps_tile_dimensions_to_the_max_analysis_tile_size() {
        // A whole-slide image far larger than one analysis tile - the
        // estimate must reflect one 4096x4096 tile's memory, not the whole
        // image's, since a worker never holds more than one tile at a time
        // regardless of the source image's total size.
        let mut images = IndexMap::new();
        images.insert(PathBuf::from("huge.tif"), image_entry(20_000, 20_000, 2));
        let job = job_with(images);

        let expected = crate::utils::resources::estimate_ram_budget(4096, 4096, 2).total_bytes();
        assert_eq!(job.estimate_ram_per_worker_bytes(), expected);
    }

    #[test]
    fn falls_back_sanely_when_the_job_has_no_images() {
        let job = job_with(IndexMap::new());
        assert!(job.estimate_ram_per_worker_bytes() > 0);
    }
}

/*


#[cfg(test)]
mod tests {
    use env_logger::Env;
    use evanalyzer_cfg::{
        core_types::SegmentationClass,
        settings::{images_settings::TStackSettings, project_settings::ProjectSettings},
    };

    use super::*;
    use crate::{
        algos::{
            Blur, ConnectedComponents, ExtractObjects, ImageSource, SaveImage, Threshold,
            ThresholdEntry, ThresholdMethod,
        },
        pipeline::pipeline::CorePipelineSettings,
    };

    #[test]
    fn simple_pipeline() -> Result<(), InternalErrors> {
        ////////////////
        env_logger::Builder::from_env(Env::default().default_filter_or("debug")).init();

        // First pipeline
        let mut pipeline = Pipeline::new(
            PipelineId(1),
            CorePipelineSettings {
                start_image: ImageAddress::Channel(0),
            },
        );

        let saver = SaveImage {
            path: concat!(env!("CARGO_MANIFEST_DIR"), "/tests/project_test/output/start.jpg")
                .into(),
            source: ImageSource::Image,
        };
        pipeline.add_command(Box::new(saver));

        let blur = Blur { kernel_size: 3 };
        pipeline.add_command(Box::new(blur));

        let saver = SaveImage {
            path: concat!(env!("CARGO_MANIFEST_DIR"), "/tests/project_test/output/after_blur.jpg")
                .into(),
            source: ImageSource::Image,
        };
        pipeline.add_command(Box::new(saver));

        let threshold = Threshold {
            thresholds: vec![
                ThresholdEntry {
                    method: ThresholdMethod::Manual,
                    min_threshold: 0.0,
                    max_threshold: 0.3,
                    segmentation_class: SegmentationClass(0),
                },
                ThresholdEntry {
                    method: ThresholdMethod::Manual,
                    min_threshold: 0.3,
                    max_threshold: 0.5,
                    segmentation_class: SegmentationClass(2),
                },
                ThresholdEntry {
                    method: ThresholdMethod::Manual,
                    min_threshold: 0.5,
                    max_threshold: 0.7,
                    segmentation_class: SegmentationClass(3),
                },
                ThresholdEntry {
                    method: ThresholdMethod::Manual,
                    min_threshold: 0.7,
                    max_threshold: 1.0,
                    segmentation_class: SegmentationClass(4),
                },
            ],
        };
        pipeline.add_command(Box::new(threshold));

        let saver = SaveImage {
            path: concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/tests/project_test/output/after_threshold.jpg"
            )
            .into(),
            source: ImageSource::SegmentationMask,
        };
        pipeline.add_command(Box::new(saver));

        let cco = ConnectedComponents;
        pipeline.add_command(Box::new(cco));

        let saver = SaveImage {
            path: concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/tests/project_test/output/instance_map.jpg"
            )
            .into(),
            source: ImageSource::InstanceMap,
        };
        pipeline.add_command(Box::new(saver));

        let extract_objects = ExtractObjects;
        pipeline.add_command(Box::new(extract_objects));

        // Prepare the project
        let mut project = ProjectSettings::default();
        project.images.settings.z_stack = Some(ZStackSettings {
            z_projection: ZStackHandling::AllStacks,
            z_range: None,
        });

        project.images.settings.t_stack = Some(TStackSettings {
            stack_handling: TStackHandling::AllStacks,
            playback_speed: 0.0,
            t_stack: 0,
            frame_interval: None,
        });

        project.add_image();

        // Create analyze job
        let mut analyze_job = JobExecutor::new(
            concat!(env!("CARGO_MANIFEST_DIR"), "/tests/project_test").into(),
            project.images.list,
            project.images.root.expect("No image root path set"),
            project.images.settings,
        );
        analyze_job.add_pipeline(pipeline);

        let (handle, rx) = analyze_job.run_async(1);
        for event in rx {
            if let ProgressEvent::ImageFailed { path } = event {
                println!("Failed: {}", path.display());
            }
        }
        handle.join().unwrap()?;

        Ok(())
    }
}

*/
