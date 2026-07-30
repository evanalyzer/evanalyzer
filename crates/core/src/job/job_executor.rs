use crate::{
    ImageInfo, ZProjection,
    image::{ImageReader, ImageTile, PixelSizes, ReadMode},
    pipeline::{
        pipeline::{Pipeline, PipelineImageMeta},
        pipeline_cache::{ImageCache, ImageMap, PipelineCache},
    },
    storage::PipelineResultExporter,
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
    collections::{BTreeMap, HashSet},
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
        nr_bits: u8,
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
/// Only used by `analyze_image_tiles_parallel`; full multi-image runs always
/// process every tile regardless of this setting.
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

pub struct JobExecutor {
    pub project_path: PathBuf,
    pub output_path: PathBuf,
    pub pipelines: IndexMap<PipelineId, Pipeline>,
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
            pipelines: IndexMap::new(),
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
                match self.analyze_image_tiles_parallel(
                    rel_path,
                    &abs_path,
                    image_info,
                    &order,
                    self.result_storage.clone(),
                    progress.clone(),
                    cancel,
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
                // Multiple images: parallelize over images
                self.images
                    .par_iter()
                    .try_for_each(|(rel_path, image_info)| {
                        if cancel.load(Ordering::Relaxed) {
                            return Err(InternalErrors::Cancelled);
                        }
                        let abs_path = self.image_base_path.join(rel_path);
                        match self.analyze_image(
                            &rel_path,
                            &abs_path,
                            image_info,
                            &order,
                            self.result_storage.clone(),
                            cancel.clone(),
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
                                Ok(())
                            }
                            Err(e) => {
                                progress
                                    .send(ProgressEvent::ImageFailed {
                                        path: rel_path.clone(),
                                    })
                                    .ok();
                                // See the single-image branch above: name the
                                // failing image so the caller's error message
                                // doesn't just say *something* broke.
                                Err(InternalErrors::Internal(format!(
                                    "{}: {e}",
                                    rel_path.display()
                                )))
                            }
                        }
                    })
            }
        });

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

    pub fn add_pipeline(&mut self, p: Pipeline) {
        self.pipelines.insert(p.id, p);
    }

    /// Picks the tile size for a single-image preview run.
    ///
    /// At higher zoom the viewport covers a smaller area of the full image, so
    /// using the same fixed 4096px tile means re-reading and re-analyzing a lot
    /// of pixels the user can't see just to get the small visible patch. Shrinking
    /// the tile as zoom increases keeps the analyzed area closer to what's on
    /// screen, so feedback for that area arrives faster.
    fn preview_tile_size(&self) -> usize {
        const BASE_TILE_SIZE: usize = 4096;
        let Some(zoom) = self.preview_tile_settings.as_ref().map(|s| s.zoom) else {
            return BASE_TILE_SIZE;
        };
        if zoom >= 8.0 {
            512
        } else if zoom >= 4.0 {
            1024
        } else if zoom >= 2.0 {
            2048
        } else {
            BASE_TILE_SIZE
        }
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
    /// `analyze_image_tiles_parallel` uses (same `preview_tile_size`/`prepare_tile_iterator`/
    /// `is_tile_visible` calls), so callers can gate on the same number the real run
    /// would use before paying for a full pipeline dispatch.
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

    /// Analyze one image
    ///
    /// This function analyzes one whole image, including all configured time and z-stacks.
    /// If this is whole slide image, which is too big to load to RAM at once, the function
    /// splits the image into tiles and analyzes tiles in parallel.
    ///
    /// Tile processing runs on the same rayon pool that [`run`](Self::run) uses to
    /// parallelize over images. Nested `par_iter` calls share that pool via work
    /// stealing, so when there are fewer images than cores the otherwise-idle
    /// cores pick up tile work for the images that *are* running instead of
    /// sitting idle - no manual thread-budget split between "image parallelism"
    /// and "tile parallelism" is needed.
    fn analyze_image(
        &self,
        image_rel_path: &PathBuf,
        image_path: &PathBuf,
        image_entry: &ImageEntry,
        order: &[PipelineId],
        exporter: Arc<Mutex<dyn PipelineResultExporter>>,
        cancel: Arc<AtomicBool>,
    ) -> Result<(), InternalErrors> {
        const RES_IDX: i32 = 0;
        const TILE_SIZE: usize = 4096;
        let start_image = Instant::now();

        // Extract everything needed from the reader in a scoped block so the
        // borrow ends before tiles are processed in parallel - each tile work
        // item below opens its own reader so concurrent threads never share
        // mutable file-handle state (see analyze_image_tiles_parallel).
        let (full_size, z_proj, z_handling, z_stacks, t_stacks, is_rgb, nr_bits, pixel_sizes) = {
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
            let z_stacks: Vec<i32> = z_range.collect();
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
                z_stacks,
                t_stacks,
                py_meta.is_rgb,
                py_meta.nr_bits,
                pixel_sizes,
            )
        };

        let tiles: Vec<ImageTile> = self
            .prepare_tile_iterator(full_size.width, full_size.height, TILE_SIZE)
            .collect();

        // Flat (t, z, tile) work list, processed in parallel below.
        let mut all_work: Vec<(i32, i32, ImageTile)> =
            Vec::with_capacity(t_stacks.len() * z_stacks.len() * tiles.len());
        for &t in &t_stacks {
            for &z in &z_stacks {
                for tile in &tiles {
                    all_work.push((t, z, tile.clone()));
                }
            }
        }

        // Spawn a dedicated DB writer thread so one tile's image loading and
        // pipeline execution can overlap with another tile's DuckDB insert.
        let (cache_tx, cache_rx) = std::sync::mpsc::sync_channel::<PipelineCache>(4);
        let writer_handle = {
            let exporter = exporter.clone();
            std::thread::spawn(move || -> Result<(), InternalErrors> {
                for cache in cache_rx {
                    let t0 = Instant::now();
                    exporter.lock().expect("Poisoned").export(&cache)?;
                    info!("DB write: {:.1?}", t0.elapsed());
                }
                Ok(())
            })
        };

        let tile_result =
            all_work
                .into_par_iter()
                .try_for_each(|(t, z, tile)| -> Result<(), InternalErrors> {
                    if cancel.load(Ordering::Relaxed) {
                        return Err(InternalErrors::Cancelled);
                    }

                    let reader = ImageReader::new(image_path, ReadMode::Default)?;
                    let z_range_in = matches!(
                        z_handling,
                        ZStackHandling::AllStacks | ZStackHandling::SingleStack
                    )
                    .then(|| z..=z);

                    let mut cache = self.prepare_pipeline_cache(
                        &reader,
                        Arc::new(image_entry.clone()),
                        &tile,
                        t,
                        &z_proj,
                        &z_range_in,
                        RES_IDX,
                        full_size,
                        is_rgb,
                        image_rel_path,
                        nr_bits,
                        pixel_sizes.clone(),
                    )?;

                    let mut bp_hit = false;
                    for pipe_id in order {
                        if bp_hit {
                            break;
                        }
                        if let Some(p) = self.pipelines.get(pipe_id) {
                            let (bp_step, snapshot_mode) = self
                                .breakpoint
                                .as_ref()
                                .filter(|b| b.pipeline_id == *pipe_id)
                                .map(|b| {
                                    (Some(b.pipeline_step_id), b.mode == BreakpointMode::Snapshot)
                                })
                                .unwrap_or((None, false));
                            let result =
                                p.run(self.output_path.clone(), cache, bp_step, snapshot_mode)?;
                            bp_hit = result.breakpoint_hit;
                            cache = result.cache;
                        }
                    }
                    if bp_hit {
                        // Skip DB write for a Stop breakpoint run.
                        return Ok(());
                    }

                    match cache_tx.try_send(cache) {
                        Ok(()) => {}
                        Err(std::sync::mpsc::TrySendError::Full(cache)) => {
                            warn!("DB writer backpressure: channel full, tile stalling");
                            cache_tx.send(cache).map_err(|e| {
                                InternalErrors::Io(format!("DB writer exited: {e}"))
                            })?;
                        }
                        Err(std::sync::mpsc::TrySendError::Disconnected(_)) => {
                            return Err(InternalErrors::Io(
                                "DB writer thread exited unexpectedly".into(),
                            ));
                        }
                    }
                    Ok(())
                });

        drop(cache_tx);
        let writer_result = writer_handle.join().expect("DB writer thread panicked");

        let duration = start_image.elapsed();
        info!("Executed image pipeline in {:?}", duration);

        tile_result.and(writer_result)
    }

    /// Like [`analyze_image`] but processes tiles in parallel.
    ///
    /// Used when only a single image is being processed so that parallelism is
    /// applied across tiles rather than across images.  A fresh [`ImageReader`]
    /// is created per work item so that concurrent threads do not share mutable
    /// file-handle state.
    ///
    /// After each tile completes a [`ProgressEvent::TileCompleted`] event is sent on
    /// `progress` carrying the ROIs found in that tile, allowing callers to update
    /// an incremental preview without waiting for all tiles to finish.
    fn analyze_image_tiles_parallel(
        &self,
        image_rel_path: &PathBuf,
        image_path: &PathBuf,
        image_entry: &ImageEntry,
        order: &[PipelineId],
        exporter: Arc<Mutex<dyn PipelineResultExporter>>,
        progress: Sender<ProgressEvent>,
        cancel: Arc<AtomicBool>,
    ) -> Result<(), InternalErrors> {
        const RES_IDX: i32 = 0;
        let tile_size = self.preview_tile_size();

        // Extract everything we need from the reader in a scoped block so the
        // borrow of `reader` ends before we enter the parallel section.
        let (full_size, z_proj, z_handling, z_range, t_stacks, is_rgb, nr_bits, pixel_sizes) = {
            let reader = ImageReader::new(image_path, ReadMode::Default)?;
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

        // Build a flat list of every (t, z, tile) combination.
        // Each work item carries its own Sender clone because Sender is not Sync.
        // The fifth element marks the single tile that should emit BreakpointReached.
        let total_tiles = tiles.len() * z_stacks.len() * t_stacks.len();
        let completed = Arc::new(AtomicUsize::new(0));

        let mut all_work: Vec<(i32, i32, ImageTile, Sender<ProgressEvent>, bool)> =
            Vec::with_capacity(total_tiles);
        for &t in &t_stacks {
            for &z in &z_stacks {
                for tile in &tiles {
                    let is_bp_target = breakpoint_target
                        .map(|(ox, oy)| tile.offset_x == ox && tile.offset_y == oy)
                        .unwrap_or(false);
                    all_work.push((t, z, tile.clone(), progress.clone(), is_bp_target));
                }
            }
        }

        // When preview tile settings are present, split into visible / hidden
        // so the viewport area is processed first, giving fast first results.
        let (first_pass, second_pass) = match &self.preview_tile_settings {
            Some(settings) => {
                let (visible, hidden): (Vec<_>, Vec<_>) = all_work
                    .into_iter()
                    .partition(|(_, _, tile, _, _)| settings.is_tile_visible(tile));
                let hidden = if settings.process_all_tiles {
                    hidden
                } else {
                    vec![]
                };
                (visible, hidden)
            }
            None => (all_work, vec![]),
        };

        // Recalculate so progress events reflect only the tiles actually being processed.
        let total_tiles = first_pass.len() + second_pass.len();
        progress
            .send(ProgressEvent::TilesScheduled { total_tiles })
            .ok();

        // Spawn a dedicated DB writer thread. Rayon workers send their completed
        // caches through a bounded channel instead of locking a mutex — they
        // block only when the channel is full (backpressure), not for the
        // entire duration of a DuckDB insert.
        let (cache_tx, cache_rx) = std::sync::mpsc::sync_channel::<PipelineCache>(4);
        let writer_handle = {
            let exporter = exporter.clone();
            std::thread::spawn(move || -> Result<(), InternalErrors> {
                for cache in cache_rx {
                    let t0 = Instant::now();
                    exporter.lock().expect("Poisoned").export(&cache)?;
                    info!("DB write: {:.1?}", t0.elapsed());
                }
                Ok(())
            })
        };

        // Closure that processes one (t, z, tile, sender, is_bp_target) work item.
        // `cache_tx: SyncSender` is Sync, so the closure is Fn + Send + Sync
        // and can be shared across all Rayon workers.
        let run_tile = |(t, z, tile, sender, is_bp_target): (
            i32,
            i32,
            ImageTile,
            Sender<ProgressEvent>,
            bool,
        )|
         -> Result<(), InternalErrors> {
            if cancel.load(Ordering::Relaxed) {
                return Err(InternalErrors::Cancelled);
            }

            let reader = ImageReader::new(image_path, ReadMode::Default)?;

            let z_range_in = matches!(
                z_handling,
                ZStackHandling::AllStacks | ZStackHandling::SingleStack
            )
            .then(|| z..=z);

            let mut cache = self.prepare_pipeline_cache(
                &reader,
                Arc::new(image_entry.clone()),
                &tile,
                t,
                &z_proj,
                &z_range_in,
                RES_IDX,
                full_size,
                is_rgb,
                image_rel_path,
                nr_bits,
                pixel_sizes.clone(),
            )?;

            let mut stop_capture: Option<crate::pipeline::pipeline::BreakpointCapture> = None;
            let mut snapshot_capture: Option<crate::pipeline::pipeline::BreakpointCapture> = None;
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
                if let Some(p) = self.pipelines.get(pipe_id) {
                    let (bp_step, snapshot_mode) = self
                        .breakpoint
                        .as_ref()
                        .filter(|b| b.pipeline_id == *pipe_id)
                        .map(|b| (Some(b.pipeline_step_id), b.mode == BreakpointMode::Snapshot))
                        .unwrap_or((None, false));
                    let result = p.run(self.output_path.clone(), cache, bp_step, snapshot_mode)?;
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

            // Snapshot: send the captured buffers but continue to DB write.
            if let Some(capture) = snapshot_capture {
                if is_bp_target {
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
                // fall through — the full pipeline ran, write results normally.
            }

            // Stop: send the buffers and skip DB write.
            if let Some(capture) = stop_capture {
                if is_bp_target {
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
                return Ok(());
            }

            let tile_objects: Vec<ObjectMetricSettings> = cache
                .object_cache
                .values()
                .map(|r| r.to_object_settings())
                .collect();

            match cache_tx.try_send(cache) {
                Ok(()) => {}
                Err(std::sync::mpsc::TrySendError::Full(cache)) => {
                    warn!("DB writer backpressure: channel full, tile stalling");
                    cache_tx.send(cache).map_err(|_| {
                        InternalErrors::Io("DB writer thread exited unexpectedly".into())
                    })?;
                }
                Err(std::sync::mpsc::TrySendError::Disconnected(_)) => {
                    return Err(InternalErrors::Io(
                        "DB writer thread exited unexpectedly".into(),
                    ));
                }
            }

            let tile_index = completed.fetch_add(1, Ordering::Relaxed) + 1;
            sender
                .send(ProgressEvent::TileCompleted {
                    tile_index,
                    total_tiles,
                    objects: tile_objects,
                })
                .ok();

            Ok(())
        };

        // Run both passes; collect the combined result before touching the
        // channel so the writer always has a chance to drain cleanly.
        let pass_result = first_pass
            .into_par_iter()
            .try_for_each(&run_tile)
            .and_then(|()| second_pass.into_par_iter().try_for_each(&run_tile));

        // Dropping cache_tx closes the channel so the writer thread's loop exits.
        // The closure borrows &cache_tx by reference (it's Copy); NLL ends that
        // borrow at the last try_for_each call above, so drop(cache_tx) is valid.
        drop(cache_tx);

        let writer_result = writer_handle.join().expect("DB writer thread panicked");

        pass_result.and(writer_result)
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

    /// Prepares the pipeline cache
    ///
    /// Loads the selected image plane from the image and inits the
    /// cache with the loaded image planes and returns the cache.
    /// This cache can now be used for processing the pipelines of the image
    fn prepare_pipeline_cache(
        &self,
        image_reader: &ImageReader,
        image_entry: Arc<ImageEntry>,
        image_tile: &ImageTile,
        t_stack: i32,
        z_projection: &ZProjection,
        z_range: &Option<RangeInclusive<i32>>,
        resolution_index: i32,
        full_image_width: ImageSize,
        is_rgb: bool,
        image_rel_path: &PathBuf,
        nr_of_bits: u8,
        pixel_sizes: PixelSizes,
    ) -> Result<PipelineCache, InternalErrors> {
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

        // Collect Vec into HashMap automatically
        let image_cache_map: ImageMap = loaded_channels
            .into_iter()
            .map(|img| (ImageAddress::Channel(img.c_stack), img.image))
            .collect();

        let image_meta = PipelineImageMeta {
            image_tile_info: ImageTile {
                width: loaded_size.width,
                height: loaded_size.height,
                ..*image_tile // Copies offset_x and offset_y from image_tile
            },
            full_image_width,
            is_rgb,
            nr_of_bits,
            pixel_sizes,
        };

        Ok(PipelineCache {
            image_cache: ImageCache {
                image_meta: image_meta,
                images: image_cache_map,
            },
            object_cache: BTreeMap::new(),
            image_rel_path: image_rel_path.clone(),
        })
    }

    /// Determines the correct order to run pipelines based on dependencies
    fn get_execution_order(&self) -> Vec<PipelineId> {
        let mut order = Vec::new();
        let mut visited = HashSet::new();
        let mut temp_visited = HashSet::new();

        fn visit(
            name: &PipelineId,
            pipelines: &IndexMap<PipelineId, Pipeline>,
            visited: &mut HashSet<PipelineId>,
            temp_visited: &mut HashSet<PipelineId>,
            order: &mut Vec<PipelineId>,
        ) {
            if temp_visited.contains(name) {
                panic!("Circular dependency detected!");
            }
            if !visited.contains(name) {
                temp_visited.insert(name.clone());
                if let Some(p) = pipelines.get(name) {
                    for dep in &p.dependencies {
                        visit(dep, pipelines, visited, temp_visited, order);
                    }
                }
                temp_visited.remove(name);
                visited.insert(name.clone());
                order.push(name.clone());
            }
        }

        for name in self.pipelines.keys() {
            visit(
                name,
                &self.pipelines,
                &mut visited,
                &mut temp_visited,
                &mut order,
            );
        }
        order
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
    use crate::pipeline::pipeline::CorePipelineSettings;
    use crate::storage::memory::MemoryExporter;
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
        job.add_pipeline(make_pipeline(1, &[]));
        job.add_pipeline(make_pipeline(2, &[]));
        job.add_pipeline(make_pipeline(3, &[]));

        let order = job.get_execution_order();

        assert_eq!(order, vec![PipelineId(1), PipelineId(2), PipelineId(3)]);
    }

    #[test]
    fn dependency_runs_before_dependent() {
        let mut job = make_job_executor();
        job.add_pipeline(make_pipeline(1, &[2]));
        job.add_pipeline(make_pipeline(2, &[]));

        let order = job.get_execution_order();

        assert_eq!(order, vec![PipelineId(2), PipelineId(1)]);
    }

    #[test]
    fn transitive_dependency_chain_is_fully_ordered() {
        let mut job = make_job_executor();
        job.add_pipeline(make_pipeline(1, &[2]));
        job.add_pipeline(make_pipeline(2, &[3]));
        job.add_pipeline(make_pipeline(3, &[]));

        let order = job.get_execution_order();

        assert_eq!(order, vec![PipelineId(3), PipelineId(2), PipelineId(1)]);
    }

    #[test]
    fn diamond_dependency_orders_shared_dependency_first() {
        let mut job = make_job_executor();
        job.add_pipeline(make_pipeline(1, &[])); // A
        job.add_pipeline(make_pipeline(2, &[1])); // B depends on A
        job.add_pipeline(make_pipeline(3, &[1])); // C depends on A
        job.add_pipeline(make_pipeline(4, &[2, 3])); // D depends on B, C

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
        job.add_pipeline(make_pipeline(1, &[99]));

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
        job.add_pipeline(make_pipeline(1, &[2]));
        job.add_pipeline(make_pipeline(2, &[1]));

        job.get_execution_order();
    }

    #[test]
    #[should_panic(expected = "Circular dependency detected!")]
    fn self_dependency_panics() {
        let mut job = make_job_executor();
        job.add_pipeline(make_pipeline(1, &[1]));

        job.get_execution_order();
    }

    #[test]
    #[should_panic(expected = "Circular dependency detected!")]
    fn transitive_circular_dependency_panics() {
        let mut job = make_job_executor();
        job.add_pipeline(make_pipeline(1, &[2]));
        job.add_pipeline(make_pipeline(2, &[3]));
        job.add_pipeline(make_pipeline(3, &[1]));

        job.get_execution_order();
    }
}

/// End-to-end coverage for `run`/`analyze_image_tiles_parallel`/
/// `prepare_pipeline_cache` - the paths above only exercise the pure helper
/// methods (tile math, precedence resolution, execution ordering), never the
/// actual image-loading-through-DB-export flow. Uses the same real BioFormats
/// fixture and JVM setup as `image_reader`'s tests.
#[cfg(test)]
mod full_run_integration_tests {
    use super::*;
    use crate::algos::{ConnectedComponents, ExtractObjects, Threshold, ThresholdEntry, ThresholdMethod};
    use crate::init_java_wrapper;
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
        job.add_pipeline(threshold_connected_components_extract_pipeline());
        job
    }

    #[test]
    fn run_on_a_real_fixture_image_writes_extracted_objects_through_the_exporter() {
        init_java_wrapper(1_000_000_000).unwrap();
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
            !events.iter().any(|e| matches!(e, ProgressEvent::ImageFailed { .. })),
            "no image should have failed"
        );
        assert!(
            !out_objects.lock().unwrap().is_empty(),
            "thresholding the whole 8-bit range should extract at least one object"
        );
    }

    #[test]
    fn count_preview_visible_tiles_matches_the_real_fixture_images_single_tile_grid() {
        init_java_wrapper(1_000_000_000).unwrap();
        let job = make_single_image_job(Arc::new(Mutex::new(Vec::new())));

        // The fixture image is far smaller than the 4096px base tile size
        // used when no preview settings are set, so it must resolve to
        // exactly one tile.
        let count = job.count_preview_visible_tiles().unwrap();
        assert_eq!(count, 1);
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
        init_java_wrapper,
        pipeline::pipeline::CorePipelineSettings,
    };

    #[test]
    fn simple_pipeline() -> Result<(), InternalErrors> {
        ////////////////
        env_logger::Builder::from_env(Env::default().default_filter_or("debug")).init();

        init_java_wrapper(1000000000).expect("Can not init JAVA");

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
