use crate::{
    ImageInfo, ZProjection, algos::Connectivity, image::{ImageReader, ImageTile, PixelSizes, ReadMode}, pipeline::{
        image_cache::ImageCache, object_cache::ObjectCache, pipeline::{ Pipeline, PipelineImageMeta}, pipeline_cache::{CacheAddress, GlobalImageMeta, GlobalPipelineCache},
    }, resources::MAX_TILE_SIZE, storage::PipelineResultExporter,
};
use evanalyzer_cfg::{
    core_types::{ImageAddress, InternalErrors, PipelineId},
    settings::{
        images_settings::{
            GlobalImageSettings, ImageEntry, TStackHandling, ZStackHandling, ZStackSettings,
        },
        object_settings::ObjectMetricSettings,
    },
};
use indexmap::IndexMap;
use kornia_image::ImageSize;
use log::{info, warn};
use rayon::prelude::*;
use std::{
    collections::HashSet,
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

/// Controls pipeline behaviour when a breakpoint step is reached.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum BreakpointMode {
    /// Stop the pipeline at this step and return the intermediate image.
    Stop,
    /// Capture the image at this step, then continue running the pipeline
    /// to completion.  The final results (ROIs, DB write) are produced
    /// normally; the captured image is sent as a side-channel preview.
    Snapshot,
}

pub struct BreakpointSettings {
    pub pipeline_id: PipelineId,
    pub pipeline_step_id: i32,
    pub mode: BreakpointMode,
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


    pub image_base_path: PathBuf,
    pub images: IndexMap<PathBuf, ImageEntry>,
    pub global_image_settings: GlobalImageSettings,
    pub result_storage: Arc<Mutex<dyn PipelineResultExporter>>,
    pub override_pixel_sizes: Option<PixelSizes>,
    /// When set, tile selection in single-image preview runs is guided by the
    /// viewport position.  `None` means process all tiles (normal full run).
    pub preview_tile_settings: Option<PreviewTileSettings>,

    /// Debugging settings, if set the pipeline stops at this point and returns the actual image
    pub breakpoint: Option<BreakpointSettings>,
}

impl<'a> JobExecutor {
    pub fn new(
        project_path: PathBuf,
        output_path: PathBuf,
        images: IndexMap<PathBuf, ImageEntry>,
        image_base_path: PathBuf,
        global_image_settings: GlobalImageSettings,
        result_storage: Arc<Mutex<dyn PipelineResultExporter>>,
        override_pixel_sizes: Option<PixelSizes>,
    ) -> Self {
        Self {
            pipelines_pre_process: IndexMap::new(),
            pipelines_post_process: IndexMap::new(),
            project_path,
            output_path,
            image_base_path,
            images,
            global_image_settings,
            result_storage,
            override_pixel_sizes,
            preview_tile_settings: None,
            breakpoint: None,
        }
    }

    pub fn add_pre_process_pipeline(&mut self, p: Pipeline) {
        self.pipelines_pre_process.insert(p.id, p);
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
        let image_cache_bytes = crate::resources::recommended_image_cache_bytes(
            parallelism,
            ram_budget.working_set_bytes + ram_budget.object_cache_margin_bytes,
            ram_budget.min_image_cache_bytes,
        );
        info!(
            "Image cache capacity: {} bytes/worker ({}) x {} worker(s)",
            image_cache_bytes,
            crate::resources::format_binary_bytes(image_cache_bytes),
            parallelism
        );

        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(parallelism)
            .build()
            .unwrap();

        let order = self.get_execution_order();
        let total = self.images.len();
        let completed = AtomicUsize::new(0);

        progress.send(ProgressEvent::Started { total }).ok();

        let result = pool.install(|| {
            if total == 1 {
                // Single image: parallelize over tiles instead of images
                let (rel_path, image_info) = self.images.iter().next().unwrap();
                let abs_path = self.image_base_path.join(rel_path);
                match self.analyze_image(
                    rel_path,
                    &abs_path,
                    image_info,
                    &order,
                    self.result_storage.clone(),
                    cancel,
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

        info!("Pipeline completed in {:?}", start.elapsed());
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
        exporter: Arc<Mutex<dyn PipelineResultExporter>>,
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

        // Extract everything needed from the reader in a scoped block so the
        // borrow ends before tiles are processed in parallel - each tile work
        // item below opens its own reader so concurrent threads never share
        // mutable file-handle state.
        let (full_size, z_proj, z_handling, z_range, t_stacks, is_rgb, nr_bits, pixel_sizes) = {
            let start = Instant::now();
            let reader = ImageReader::new(image_path, ReadMode::Default)?;
            let duration = start.elapsed();
            info!("Prepare image reader {:?} {:?}", image_rel_path, duration);

            let series_info = reader
                .image_meta
                .series
                .get(&image_entry.selected_series)
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
            )
        };

        let tiles: Vec<ImageTile> = self
            .prepare_tile_iterator(full_size.width, full_size.height, tile_size)
            .collect();
        let z_stacks: Vec<i32> = z_range.collect();

        // Pre-select the breakpoint target tile before building work items so
        // we can mark exactly one tile as the event sender deterministically.
        // Only meaningful when `progress.is_some()` (interactive run); in a
        // batch run `self.breakpoint` is never set, so this is always `None`
        // there.
        //
        // Strategy: among visible tiles, pick the one whose centre is closest
        // to the viewport centre (in image-pixel space).  When no viewport
        // settings are available, fall back to the geometrically middle tile.
        let breakpoint_target: Option<(usize, usize)> = if self.breakpoint.is_some() {
            match &self.preview_tile_settings {
                Some(settings) => {
                    let visible: Vec<&ImageTile> = tiles
                        .iter()
                        .filter(|t| settings.is_tile_visible(t))
                        .collect();
                    let candidates = if visible.is_empty() {
                        tiles.iter().collect::<Vec<_>>()
                    } else {
                        visible
                    };
                    if candidates.len() == 1 {
                        candidates.first().map(|t| (t.offset_x, t.offset_y))
                    } else {
                        // Viewport centre in image-pixel coordinates:
                        //   screen_x = img_x * zoom + offset_x  →  img_x = (screen_x - offset_x) / zoom
                        let cx =
                            (settings.viewport_width / 2.0 - settings.offset_x) / settings.zoom;
                        let cy =
                            (settings.viewport_height / 2.0 - settings.offset_y) / settings.zoom;
                        candidates
                            .iter()
                            .map(|t| {
                                let tx = t.offset_x as f32 + t.width as f32 / 2.0;
                                let ty = t.offset_y as f32 + t.height as f32 / 2.0;
                                (t, (tx - cx).powi(2) + (ty - cy).powi(2))
                            })
                            .min_by(|(_, da), (_, db)| da.total_cmp(db))
                            .map(|(t, _)| (t.offset_x, t.offset_y))
                    }
                }
                None => tiles.get(tiles.len() / 2).map(|t| (t.offset_x, t.offset_y)),
            }
        } else {
            None
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
        let total_tiles = (visible_tiles.len() + hidden_tiles.len()) * z_stacks.len() * t_stacks.len()
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

        'stacks: for &t in &t_stacks {
            for &z in &z_stacks {
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

                    let is_bp_target = breakpoint_target
                        .map(|(ox, oy)| tile.offset_x == ox && tile.offset_y == oy)
                        .unwrap_or(false);

                    // A fresh reader per tile, not pooled: measured to be
                    // faster in practice than sharing a small mutex-guarded
                    // pool across many concurrent tiles (see
                    // `prepare_pipeline_cache`'s doc comment) - the pool's
                    // per-tile lock contention cost more wall-clock time
                    // than the reader construction it was meant to save.
                    let reader = ImageReader::new(image_path, ReadMode::Default)?;

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
                                .map(|b| {
                                    (Some(b.pipeline_step_id), b.mode == BreakpointMode::Snapshot)
                                })
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

                let stack_merge_result = stack_result.and_then(|mut merged_cache| {
                    if merged_cache.object_cache.is_empty() {
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
                            .map(|b| (Some(b.pipeline_step_id), b.mode == BreakpointMode::Snapshot))
                            .unwrap_or((None, false));
                        let result = p.run_commands(
                            self.output_path.clone(),
                            None,
                            merged_cache,
                            bp_step,
                            snapshot_mode,
                        )?;
                        merged_cache = result.cache;

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

                    exporter
                        .lock()
                        .unwrap_or_else(|e| e.into_inner())
                        .export(&merged_cache)?;

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
        let finalize_result = exporter
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .finalize_image(image_rel_path, combined_error.as_deref());

        let duration = start_image.elapsed();
        info!("Executed image pipeline in {:?}", duration);

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
    /// `evanalyzer_app`'s `images_ext.rs`/`project_ext.rs`). A series entry always
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
            .get(&image_entry.selected_series)
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
    fn get_z_stack_settings(&self, image_entry: &ImageEntry) -> ZStackSettings {
        let local = image_entry
            .series
            .get(&image_entry.selected_series)
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
            image_rel_path: image_rel_path.into(),
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
            image_entry.selected_series,
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

    /// Pure tile-count math, factored out of [`count_preview_visible_tiles`] so it
    /// can be unit-tested without needing a real image file on disk.
    fn count_visible_tiles_for_image(&self, full_width: usize, full_height: usize) -> usize {
        let tile_size = self.preview_tile_size();
        let tiles = self.prepare_tile_iterator(full_width, full_height, tile_size);
        match &self.preview_tile_settings {
            Some(settings) => tiles.filter(|t| settings.is_tile_visible(t)).count(),
            None => tiles.count(),
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
                .get(&image_entry.selected_series)
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
                        visit(dep, pre_pipelines, post_pipelines, visited, temp_visited, order);
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
    fn preview_tile_size(&self) -> usize {
        let Some(zoom) = self.preview_tile_settings.as_ref().map(|s| s.zoom) else {
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
    pub fn estimate_ram_budget(&self) -> crate::resources::RamBudget {
        self.images
            .values()
            .filter_map(|entry| entry.series.get(&entry.selected_series))
            .map(|series| {
                let tile_width = (series.image_width as usize).min(MAX_TILE_SIZE);
                let tile_height = (series.image_height as usize).min(MAX_TILE_SIZE);
                crate::resources::estimate_ram_budget(tile_width, tile_height, series.channels.len())
            })
            .max_by_key(|budget| budget.total_bytes())
            .unwrap_or_else(|| {
                crate::resources::estimate_ram_budget(MAX_TILE_SIZE, MAX_TILE_SIZE, 1)
            })
    }

    /// Flat total of [`estimate_ram_budget`](Self::estimate_ram_budget) -
    /// what [`recommended_parallelism`](crate::recommended_parallelism)
    /// actually wants.
    pub fn estimate_ram_per_worker_bytes(&self) -> u64 {
        self.estimate_ram_budget().total_bytes()
    }
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
            Arc::new(Mutex::new(MemoryExporter {
                out_objects: Arc::new(Mutex::new(Vec::new())),
            })),
            None,
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
            }),
        );
        let mut entry = make_image_entry_with_default_series();
        entry.series.get_mut(&0).unwrap().t_stack = Some(TStackSettings {
            stack_handling: TStackHandling::SingleStack,
            playback_speed: 1.0,
            t_stack: 3,
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
            Arc::new(Mutex::new(MemoryExporter {
                out_objects: Arc::new(Mutex::new(Vec::new())),
            })),
            None,
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
            Arc::new(Mutex::new(MemoryExporter {
                out_objects: Arc::new(Mutex::new(Vec::new())),
            })),
            None,
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
            Arc::new(Mutex::new(MemoryExporter {
                out_objects: Arc::new(Mutex::new(Vec::new())),
            })),
            None,
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
            Arc::new(Mutex::new(MemoryExporter {
                out_objects: Arc::new(Mutex::new(Vec::new())),
            })),
            None,
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
            Arc::new(Mutex::new(MemoryExporter {
                out_objects: Arc::new(Mutex::new(Vec::new())),
            })),
            None,
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
        fn cite(&self) -> Option<&'static CitationMetadata> {
            None
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
    use evanalyzer_cfg::core_types::{PixelUnits, SegmentationClass};
    use evanalyzer_cfg::settings::images_settings::SeriesSettings;
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
        pipeline.add_command(Box::new(Threshold {
            thresholds: vec![ThresholdEntry {
                method: ThresholdMethod::Manual,
                min_threshold: 0.0,
                max_threshold: 255.0,
                unit: PixelUnits::Bit,
                object_class_id: SegmentationClass(1),
                value_source: ThresholdValueSource::ActualImage,
            }],
        }));
        pipeline.add_command(Box::new(ConnectedComponents { min_size: 0 }));
        pipeline.add_command(Box::new(ExtractObjects {
            max_objects_before_fail: 100_000,
        }));
        pipeline
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
            Arc::new(Mutex::new(MemoryExporter { out_objects })),
            None,
        );
        job.add_pre_process_pipeline(threshold_connected_components_extract_pipeline());
        job
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
            Arc::new(Mutex::new(MemoryExporter { out_objects })),
            None,
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

    #[test]
    fn breakpoint_stop_during_a_batch_run_skips_the_write_but_the_image_still_completes() {
        let out_objects = Arc::new(Mutex::new(Vec::new()));
        let a = PathBuf::from("multi-channel-4D-series.ome.tif");
        let b = PathBuf::from("slice_Z0_C0_T0.tif");
        let mut job = make_multi_image_job(out_objects.clone(), vec![a.clone(), b.clone()]);
        job.breakpoint = Some(BreakpointSettings {
            pipeline_id: PipelineId(1),
            pipeline_step_id: 0, // right after Threshold, before ConnectedComponents/ExtractObjects
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
            pipeline_step_id: 0,
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
    fn count_preview_visible_tiles_matches_the_real_fixture_images_single_tile_grid() {
        let job = make_single_image_job(Arc::new(Mutex::new(Vec::new())));

        // The fixture image is far smaller than the 4096px base tile size
        // used when no preview settings are set, so it must resolve to
        // exactly one tile.
        let count = job.count_preview_visible_tiles().unwrap();
        assert_eq!(count, 1);
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
use crate::algos::touches_tile_edge;
    use crate::algos::{
        ConnectedComponents, Connectivity, ExtractObjects, Threshold, ThresholdEntry,
        ThresholdMethod, ThresholdValueSource,
    };
    use crate::image::{ImageContainer, ManagedImage};
    use crate::pipeline::image_cache::ImageCache;
use crate::pipeline::pipeline::CorePipelineSettings;
    use evanalyzer_cfg::core_types::{ObjectClass, PixelUnits, SegmentationClass};
    use kornia_apriltag::utils::Point2d;
    use kornia_image::Image;
    use kornia_tensor::CpuAllocator;

    fn threshold_connected_components_extract_pipeline() -> Pipeline {
        let mut pipeline = Pipeline::new(
            PipelineId(1),
            CorePipelineSettings {
                start_image: ImageAddress::Channel(0),
            },
        );
        pipeline.add_command(Box::new(Threshold {
            thresholds: vec![ThresholdEntry {
                method: ThresholdMethod::Manual,
                min_threshold: 0.0,
                max_threshold: 255.0,
                unit: PixelUnits::Bit,
                object_class_id: SegmentationClass(1),
                value_source: ThresholdValueSource::ActualImage,
            }],
        }));
        pipeline.add_command(Box::new(ConnectedComponents { min_size: 0 }));
        pipeline.add_command(Box::new(ExtractObjects {
            max_objects_before_fail: 100_000,
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
        let image = Image::<f32, 1, CpuAllocator>::new(
            ImageSize {
                width: tile_width,
                height: tile_height,
            },
            data,
            CpuAllocator,
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
            image_rel_path: PathBuf::new(),
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
                        .object_class
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
            .run_commands(PathBuf::new(),None, whole_image_cache, None, false)
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
            Arc::new(Mutex::new(MemoryExporter {
                out_objects: Arc::new(Mutex::new(Vec::new())),
            })),
            None,
        )
    }

    #[test]
    fn matches_the_largest_image_in_the_job_not_the_first_or_smallest() {
        let mut images = IndexMap::new();
        images.insert(PathBuf::from("small.tif"), image_entry(512, 512, 1));
        images.insert(PathBuf::from("large.tif"), image_entry(2048, 2048, 4));
        let job = job_with(images);

        let expected = crate::resources::estimate_ram_budget(2048, 2048, 4).total_bytes();
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

        let expected = crate::resources::estimate_ram_budget(4096, 4096, 2).total_bytes();
        assert_eq!(job.estimate_ram_per_worker_bytes(), expected);
    }

    #[test]
    fn falls_back_sanely_when_the_job_has_no_images() {
        let job = job_with(IndexMap::new());
        assert!(job.estimate_ram_per_worker_bytes() > 0);
    }
}

#[cfg(test)]
mod exporter_poison_tests {
    use crate::storage::PipelineResultExporter;
    use crate::storage::memory::MemoryExporter;
    use std::sync::{Arc, Mutex};

    #[test]
    fn locking_the_shared_exporter_recovers_from_poison_instead_of_panicking() {
        // `exporter` (`Arc<Mutex<dyn PipelineResultExporter>>`) is shared
        // across every concurrently-running image's writer thread - see
        // `analyze_image`. One image's writer panicking while holding this
        // lock must not crash every other image's export/finalize_image
        // call too.
        let exporter: Arc<Mutex<dyn PipelineResultExporter>> =
            Arc::new(Mutex::new(MemoryExporter {
                out_objects: Arc::new(Mutex::new(Vec::new())),
            }));

        let exporter_for_panic = exporter.clone();
        let panicked = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _guard = exporter_for_panic.lock().unwrap();
            panic!("simulated writer-thread panic while holding the exporter lock");
        }));
        assert!(panicked.is_err(), "the panic should have propagated");

        // Same recovery pattern used at every real call site in this module.
        let recovered = exporter.lock().unwrap_or_else(|e| e.into_inner());
        assert!(
            recovered
                .finalize_image(std::path::Path::new("after-poison.tif"), None)
                .is_ok(),
            "the exporter must still be usable after recovering from poison"
        );
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
                    object_class_id: SegmentationClass(0),
                },
                ThresholdEntry {
                    method: ThresholdMethod::Manual,
                    min_threshold: 0.3,
                    max_threshold: 0.5,
                    object_class_id: SegmentationClass(2),
                },
                ThresholdEntry {
                    method: ThresholdMethod::Manual,
                    min_threshold: 0.5,
                    max_threshold: 0.7,
                    object_class_id: SegmentationClass(3),
                },
                ThresholdEntry {
                    method: ThresholdMethod::Manual,
                    min_threshold: 0.7,
                    max_threshold: 1.0,
                    object_class_id: SegmentationClass(4),
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
