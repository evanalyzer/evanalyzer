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
        roi_settings::RoiSettings,
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
        rois: Vec<RoiSettings>,
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
        /// Tile origin in image-pixel coordinates.
        tile_offset_x: usize,
        tile_offset_y: usize,
        tile_width: usize,
        tile_height: usize,
        /// Original image bit depth (e.g. 8, 12, 16) — used for the
        /// pixel-value HUD so values are scaled to the real range.
        nr_bits: u8,
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
        images: IndexMap<PathBuf, ImageEntry>,
        image_base_path: PathBuf,
        global_image_settings: GlobalImageSettings,
        result_storage: Arc<Mutex<dyn PipelineResultExporter>>,
        override_pixel_sizes: Option<PixelSizes>,
    ) -> Self {
        Self {
            pipelines: IndexMap::new(),
            project_path,
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
                        Err(e)
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
                                Err(e)
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

    /// Analyze one image
    ///
    /// This function analyzes one whole image, including all configured time and z-stacks.
    /// If this is whole slide image, which is too big to load to RAM at once, the function
    /// splits the image into tiles and analyze tile per tile
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
            self.prepare_z_stack_iterator(series_info, &image_entry);

        let t_stack_iter = self.prepare_t_stack_iterator(series_info, &image_entry);

        // Spawn a dedicated DB writer thread so tile N+1's image loading and
        // pipeline execution can overlap with tile N's DuckDB insert.
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

        let pixel_sizes = match &self.override_pixel_sizes {
            Some(from_user) => from_user.clone(),
            None => PixelSizes {
                px_size_x: series_info.pixel_sizes.px_size_x,
                px_size_y: series_info.pixel_sizes.px_size_y,
                px_size_z: series_info.pixel_sizes.px_size_z,
            },
        };

        let mut tile_result: Result<(), InternalErrors> = Ok(());
        'outer: for t in t_stack_iter {
            for z in z_range.clone() {
                for tile in self.prepare_tile_iterator(full_size.width, full_size.height, TILE_SIZE)
                {
                    if cancel.load(Ordering::Relaxed) {
                        tile_result = Err(InternalErrors::Cancelled);
                        break 'outer;
                    }
                    let z_range_in = matches!(
                        z_handling,
                        ZStackHandling::AllStacks | ZStackHandling::SingleStack
                    )
                    .then(|| z..=z);

                    let start = Instant::now();
                    let mut cache = self.prepare_pipeline_cache(
                        &reader,
                        Arc::new(image_entry.clone()),
                        &tile,
                        t,
                        &z_proj,
                        &z_range_in,
                        RES_IDX,
                        full_size,
                        py_meta.is_rgb,
                        image_rel_path,
                        py_meta.nr_bits,
                        pixel_sizes.clone(),
                    )?;
                    let duration = start.elapsed();
                    info!(
                        "Read image to pipeline cache {:?} {:?}",
                        image_rel_path, duration
                    );

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
                            let result = p.run(cache, bp_step, snapshot_mode)?;
                            bp_hit = result.breakpoint_hit;
                            cache = result.cache;
                        }
                    }
                    if bp_hit {
                        // Skip DB write for a Stop breakpoint run.
                        continue;
                    }

                    match cache_tx.try_send(cache) {
                        Ok(()) => {}
                        Err(std::sync::mpsc::TrySendError::Full(cache)) => {
                            warn!("DB writer backpressure: channel full, tile stalling");
                            if let Err(e) = cache_tx.send(cache) {
                                tile_result =
                                    Err(InternalErrors::Io(format!("DB writer exited: {e}")));
                                break 'outer;
                            }
                        }
                        Err(std::sync::mpsc::TrySendError::Disconnected(_)) => {
                            tile_result = Err(InternalErrors::Io(
                                "DB writer thread exited unexpectedly".into(),
                            ));
                            break 'outer;
                        }
                    }
                }
            }
        }

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
        const TILE_SIZE: usize = 4096;

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
            .prepare_tile_iterator(full_size.width, full_size.height, TILE_SIZE)
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

            let mut stop_image: Option<crate::image::ImageContainer> = None;
            let mut snapshot_image: Option<crate::image::ImageContainer> = None;
            for pipe_id in order {
                if stop_image.is_some() {
                    break;
                }
                if let Some(p) = self.pipelines.get(pipe_id) {
                    let (bp_step, snapshot_mode) = self
                        .breakpoint
                        .as_ref()
                        .filter(|b| b.pipeline_id == *pipe_id)
                        .map(|b| (Some(b.pipeline_step_id), b.mode == BreakpointMode::Snapshot))
                        .unwrap_or((None, false));
                    let result = p.run(cache, bp_step, snapshot_mode)?;
                    if result.breakpoint_hit {
                        stop_image = Some(result.image);
                    } else if let Some(snap) = result.breakpoint_snapshot {
                        snapshot_image = Some(snap);
                    }
                    cache = result.cache;
                }
            }

            // Snapshot: send the captured image but continue to DB write.
            if let Some(image) = snapshot_image {
                if is_bp_target {
                    sender
                        .send(ProgressEvent::BreakpointReached {
                            image,
                            tile_offset_x: tile.offset_x,
                            tile_offset_y: tile.offset_y,
                            tile_width: tile.width,
                            tile_height: tile.height,
                            nr_bits,
                        })
                        .ok();
                }
                // fall through — the full pipeline ran, write results normally.
            }

            // Stop: send the image and skip DB write.
            if let Some(image) = stop_image {
                if is_bp_target {
                    sender
                        .send(ProgressEvent::BreakpointReached {
                            image,
                            tile_offset_x: tile.offset_x,
                            tile_offset_y: tile.offset_y,
                            tile_width: tile.width,
                            tile_height: tile.height,
                            nr_bits,
                        })
                        .ok();
                }
                return Ok(());
            }

            let tile_rois: Vec<RoiSettings> = cache
                .roi_cache
                .values()
                .map(|r| r.to_roi_settings())
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
                    rois: tile_rois,
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
        let t_stack_settings = match image_entry.series.get(&image_entry.selected_series) {
            Some(t_stack_settings_image) => &t_stack_settings_image.t_stack,
            None => &self.global_image_settings.t_stack,
        };

        if let Some(t_stack_settings_some) = t_stack_settings {
            match t_stack_settings_some.stack_handling {
                TStackHandling::SingleStack => {
                    return t_stack_settings_some.t_stack..=t_stack_settings_some.t_stack;
                }
                TStackHandling::AllStacks => {
                    return 0..=image_info.nr_t_stacks - 1;
                }
            };
        } else {
            return 0..=image_info.nr_t_stacks - 1;
        }
    }

    fn get_z_stack_settings(&self, image_entry: &ImageEntry) -> ZStackSettings {
        let z_stack_settings = match image_entry.series.get(&image_entry.selected_series) {
            Some(z_stack_settings) => &z_stack_settings.z_stack,
            None => &self.global_image_settings.z_stack,
        };

        if let Some(t_stack_settings_some) = z_stack_settings {
            return t_stack_settings_some.clone();
        } else {
            return ZStackSettings::default();
        }
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
            roi_cache: BTreeMap::new(),
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
            Blur, ConnectedComponents, ExtractRois, ImageSource, SaveImage, Threshold,
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
            path: "/workspaces/evanalyzer/crates/core/tests/project_test/output/start.jpg".into(),
            source: ImageSource::Image,
        };
        pipeline.add_command(Box::new(saver));

        let blur = Blur { kernel_size: 3 };
        pipeline.add_command(Box::new(blur));

        let saver = SaveImage {
            path: "/workspaces/evanalyzer/crates/core/tests/project_test/output/after_blur.jpg"
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
            path:
                "/workspaces/evanalyzer/crates/core/tests/project_test/output/after_threshold.jpg"
                    .into(),
            source: ImageSource::SegmentationMask,
        };
        pipeline.add_command(Box::new(saver));

        let cco = ConnectedComponents;
        pipeline.add_command(Box::new(cco));

        let saver = SaveImage {
            path: "/workspaces/evanalyzer/crates/core/tests/project_test/output/instance_map.jpg"
                .into(),
            source: ImageSource::InstanceMap,
        };
        pipeline.add_command(Box::new(saver));

        let extract_rois = ExtractRois;
        pipeline.add_command(Box::new(extract_rois));

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
            "/workspaces/evanalyzer/crates/core/tests/project_test".into(),
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
