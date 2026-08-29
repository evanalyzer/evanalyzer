use crate::{
    ImageTile, ManagedImage,
    algos::ImageAlgorithm,
    image::{ImageContainer, PixelSizes},
    pipeline::{
        pipeline_cache::{CacheAddress, GlobalPipelineCache},
        pipeline_context::PipelineContext,
    },
};
use evanalyzer_cfg::core_types::{ImageAddress, InternalErrors, PipelineId};
use kornia_image::ImageSize;
use log::info;
use std::sync::Arc;
use std::{path::PathBuf, time::Instant};

/// The buffers captured at a breakpoint, all from the same pipeline step so
/// the UI can switch between views instantly with no pipeline re-run.
#[derive(Clone)]
pub struct BreakpointCapture {
    pub image: Arc<ImageContainer>,
    /// `None` if this pipeline step runs before segmentation (e.g. before
    /// `Threshold`).
    pub segmentation: Option<Arc<ImageContainer>>,
    /// `None` if this pipeline step runs before instance labeling (e.g.
    /// before `ConnectedComponents`/`Watershed`).
    pub instances: Option<Arc<ImageContainer>>,
}

pub struct PipelineResult {
    pub image: Arc<ImageContainer>,
    pub cache: GlobalPipelineCache,
    /// True when the pipeline stopped early due to a Stop breakpoint.
    pub breakpoint_hit: bool,
    /// Populated when a Stop or Snapshot breakpoint was reached: the image
    /// plus segmentation/instance maps captured at that step.
    pub breakpoint_capture: Option<BreakpointCapture>,
}

pub struct CorePipelineSettings {
    pub(crate) start_image: ImageAddress,
}

#[derive(Clone)]
pub struct PipelineImageMeta {
    /// Tile information of the image
    pub image_tile_info: ImageTile,
    /// The size of the original image (not the tile)
    pub full_image_width: ImageSize,
    /// True if this is a RGB image
    pub is_rgb: bool,
    /// Image bit depth: 8, 16, 32
    pub nr_of_bits: u16,
    /// Sizes of the image pixels in nm
    pub pixel_sizes: PixelSizes,
}

pub struct Pipeline {
    pub id: PipelineId,
    pub dependencies: Vec<PipelineId>,
    pub settings: CorePipelineSettings,
    /// `ExecutionScope::Tile` commands, each paired with its original
    /// as-authored step index (see `add_command`) so a breakpoint targeting
    /// a specific step still resolves correctly even though `Tile` and
    /// `WholeImage` commands are split into two separately-run lists.
    pub commands: Vec<(usize, Box<dyn ImageAlgorithm>)>,
    /// Next as-authored step index to assign in `add_command`. Counts every
    /// command added regardless of which scope it lands in, so the index
    /// reflects the pipeline's authored order (what a breakpoint targets),
    /// not either scope's own position.
    next_step_index: usize,
}

/// Pipeline execution pipeline implementation
///
/// # Returns
///
/// - `Self` - Describe the return value.
///
/// # Examples
///
/// ```
/// use crate::...;
///
/// let _ = new();
/// ```
impl Pipeline {
    pub fn new(id: PipelineId, settings: CorePipelineSettings) -> Self {
        Self {
            id,
            dependencies: Vec::new(),
            settings,
            commands: Vec::new(),
            next_step_index: 0,
        }
    }

    /// Add a new command to the end of the pipeline, routing it to the
    /// `Tile` or `WholeImage` list per its `execution_scope()` while
    /// preserving its as-authored step index (see `next_step_index`).
    pub fn add_command(&mut self, command: Box<dyn ImageAlgorithm>) {
        let step_index = self.next_step_index;
        self.next_step_index += 1;
        self.commands.push((step_index, command));
    }

    // Add dependency
    pub fn add_dependency(&mut self, pipeline_id: PipelineId) {
        if !self.dependencies.contains(&pipeline_id) {
            self.dependencies.push(pipeline_id);
        }
    }

    /// Execute this pipeline's per-tile (`ExecutionScope::Tile`) commands.
    ///
    /// Called once per tile, before tile-merge has reconciled the image's
    /// full object set - `ExecutionScope::WholeImage` commands never run
    /// here, see `run_whole_image`.
    ///
    /// `breakpoint_step` identifies a step's *as-authored* index (see
    /// `add_command`) at which to act; a step index belonging to a
    /// `WholeImage` command never matches here. `snapshot_mode`:
    ///   - `false` (Stop) — stop execution at that step and return early.
    ///   - `true`  (Snapshot) — capture the buffers at that step, then
    ///     continue to completion; the capture is returned in
    ///     `breakpoint_capture`.
    pub fn run_commands(
        &self,
        output_path: PathBuf,
        tile: Option<ImageTile>,
        mut cache: GlobalPipelineCache,
        breakpoint_step: Option<i32>,
        snapshot_mode: bool,
    ) -> Result<PipelineResult, InternalErrors> {
        let tile = match tile {
            Some(data) => data,
            None => ImageTile {
                offset_x: 0,
                offset_y: 0,
                width: cache.image_meta.full_image_width.width,
                height: cache.image_meta.full_image_width.height,
            },
        };

        let cache_idx: CacheAddress = match self.settings.start_image {
            ImageAddress::Scratchpad => CacheAddress::Scratchpad,
            ImageAddress::Memory(memory_id) => CacheAddress::Memory(memory_id),
            ImageAddress::Channel(channel_idx) => CacheAddress::Channel((channel_idx, tile)),
        };

        let Some(initial_image) = cache.get_image_from_cache(&cache_idx, tile) else {
            return Err(InternalErrors::CacheMiss("Image not found in cache".into()));
        };

        let mut ctx = PipelineContext::new_from_image(
            output_path,
            PipelineImageMeta {
                image_tile_info: tile,
                full_image_width: cache.image_meta.full_image_width,
                is_rgb: cache.image_meta.is_rgb,
                nr_of_bits: cache.image_meta.nr_of_bits,
                pixel_sizes: cache.image_meta.pixel_sizes.clone(),
            },
            initial_image,
        )?;
        let start = Instant::now();
        let mut breakpoint_capture: Option<BreakpointCapture> = None;

        for (step_index, command) in &self.commands {
            let step_index = *step_index;
            let step_start = Instant::now();
            command.execute(&mut ctx, &mut cache)?;
            let duration = step_start.elapsed();
            info!("Executed {} in {:?}", command.name(), duration);

            // Only capture at the exact breakpoint step - `breakpoint_step`
            // is `None` whenever no breakpoint is set, so this never runs
            // (and never clones segmentation/instance buffers) for a normal
            // run. Comparing against the command's as-authored `step_index`
            // (not its position in `commands`) is what lets a breakpoint
            // resolve correctly regardless of which scope's list it's run
            // through.
            if breakpoint_step == Some(step_index as i32) {
                let capture = BreakpointCapture {
                    image: ctx.image.clone(),
                    segmentation: ctx.segmentation_map.as_ref().map(|s| {
                        Arc::new(ImageContainer::U32(ManagedImage {
                            data: s.clone(),
                            tile_offset: ctx.image.tile_offset(),
                            plane: ctx.image.plane(),
                        }))
                    }),
                    instances: ctx.instance_map.as_ref().map(|i| {
                        Arc::new(ImageContainer::U32(ManagedImage {
                            data: i.clone(),
                            tile_offset: ctx.image.tile_offset(),
                            plane: ctx.image.plane(),
                        }))
                    }),
                };
                if snapshot_mode {
                    // Snapshot: capture but continue running.
                    breakpoint_capture = Some(capture);
                } else {
                    // Stop: return immediately with the intermediate image.
                    cache.clear_pipeline_context();
                    info!(
                        "Breakpoint (stop) at step {} of pipeline {} in {:?}",
                        step_index,
                        self.id,
                        start.elapsed()
                    );
                    return Ok(PipelineResult {
                        image: ctx.image,
                        cache,
                        breakpoint_hit: true,
                        breakpoint_capture: Some(capture),
                    });
                }
            }
        }

        cache.clear_pipeline_context();
        info!(
            "Executed pipeline steps {} in {:?}",
            self.id,
            start.elapsed()
        );
        Ok(PipelineResult {
            image: ctx.image,
            cache,
            breakpoint_hit: false,
            breakpoint_capture,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        ManagedImage, Object,
        algos::{ExecutionScope, ExtractObjects, Voronoi},
        pipeline::pipeline_cache::GlobalImageMeta,
    };
    use evanalyzer_cfg::core_types::{CitationMetadata, ImageAddress, ObjectClass, SizeUnits};
    use kornia_apriltag::utils::Point2d;
    use kornia_image::{Image, ImageSize};
    use kornia_tensor::CpuAllocator;

    /// Test-only stand-in for real segmentation (Threshold + ConnectedComponents):
    /// stamps one rectangular object - at tile-local coordinates - directly into
    /// `ctx.segmentation_map`/`ctx.instance_map`, so the test can focus on whether
    /// the *tile-awareness* of downstream steps (ExtractObjects, Voronoi) holds, not on
    /// thresholding behaviour.
    struct FakeSegmenter {
        rect: [usize; 4], // [x_min, y_min, x_max, y_max], tile-local, inclusive
    }

    impl ImageAlgorithm for FakeSegmenter {
        fn execute(
            &self,
            ctx: &mut PipelineContext,
            _cache: &mut GlobalPipelineCache,
        ) -> Result<(), InternalErrors> {
            let size = ctx.get_image_size();
            let (w, h) = (size.width, size.height);
            let mut seg = vec![0u32; w * h];
            let mut inst = vec![0u32; w * h];
            let [x_min, y_min, x_max, y_max] = self.rect;
            for y in y_min..=y_max {
                for x in x_min..=x_max {
                    seg[y * w + x] = 1;
                    inst[y * w + x] = 1;
                }
            }
            ctx.segmentation_map =
                Some(Image::<u32, 1, CpuAllocator>::new(size, seg, CpuAllocator).unwrap());
            ctx.instance_map =
                Some(Image::<u32, 1, CpuAllocator>::new(size, inst, CpuAllocator).unwrap());
            Ok(())
        }

        fn name(&self) -> &'static str {
            "FakeSegmenter"
        }

        fn cite(&self) -> Option<&'static CitationMetadata> {
            None
        }

        fn execution_scope(&self) -> ExecutionScope {
            ExecutionScope::Tile
        }
    }

    /// Runs stage 1 (`FakeSegmenter` + `ExtractObjects`, both `Tile`-scoped)
    /// independently for each `(tile_offset, tile_size, object_rect)` tile via
    /// `run_tile` - exactly as `JobExecutor` runs a tile's first pipeline -
    /// then combines every tile's extracted objects into one whole-image
    /// cache and runs stage 2 (`Voronoi`, `WholeImage`-scoped) exactly once
    /// via `run_whole_image`, simulating `JobExecutor` running it once per
    /// image, after tile-merge, rather than once per tile.
    fn run_pipeline_across_tiles(
        full_w: usize,
        full_h: usize,
        tiles: &[((usize, usize), (usize, usize), [usize; 4])],
    ) -> GlobalPipelineCache {
        let mut whole_image_cache = GlobalPipelineCache::default();
        whole_image_cache.image_meta = GlobalImageMeta {
            full_image_width: ImageSize {
                width: full_w,
                height: full_h,
            },
            is_rgb: false,
            nr_of_bits: 8,
            pixel_sizes: crate::image::PixelSizes {
                px_size_x: 1.0,
                px_size_y: 1.0,
                px_size_z: 1.0,
            },
        };

        for &(tile_offset, tile_size, object_rect) in tiles {
            let meta = GlobalImageMeta {
                full_image_width: ImageSize {
                    width: full_w,
                    height: full_h,
                },
                is_rgb: false,
                nr_of_bits: 8,
                pixel_sizes: crate::image::PixelSizes {
                    px_size_x: 1.0,
                    px_size_y: 1.0,
                    px_size_z: 1.0,
                },
            };

            let mut cache = GlobalPipelineCache::default();
            cache.image_meta = meta;
            let channel = Image::<f32, 1, CpuAllocator>::new(
                ImageSize {
                    width: tile_size.0,
                    height: tile_size.1,
                },
                vec![2.0f32; tile_size.0 * tile_size.1],
                CpuAllocator,
            )
            .unwrap();
            cache.add_to_channel_cache(
                std::sync::Arc::new(ImageContainer::F32Gray(ManagedImage {
                    data: channel,
                    tile_offset: Point2d {
                        x: tile_offset.0,
                        y: tile_offset.1,
                    },
                    plane: None,
                })),
                0,
                crate::ImageTile {
                    offset_x: tile_offset.0,
                    offset_y: tile_offset.1,
                    width: tile_size.0,
                    height: tile_size.1,
                },
            );

            let mut stage1 = Pipeline::new(
                PipelineId(1),
                CorePipelineSettings {
                    start_image: ImageAddress::Channel(0),
                },
            );
            stage1.add_command(Box::new(FakeSegmenter { rect: object_rect }));
            stage1.add_command(Box::new(ExtractObjects {
                max_objects_before_fail: 100_000,
            }));
            let result = stage1
                .run_commands(
                    PathBuf::default(),
                    Some(crate::ImageTile {
                        offset_x: tile_offset.0,
                        offset_y: tile_offset.1,
                        width: tile_size.0,
                        height: tile_size.1,
                    }),
                    cache,
                    None,
                    false,
                )
                .expect("tile stage must not fail");

            // Stand-in for tile-merge: union every tile's objects into one
            // whole-image object set. Real `JobExecutor` also reconciles
            // edge-touching fragments here (`merge_pending_fragments`), but
            // these test objects never touch their tile's edge, so a plain
            // union already matches what tile-merge would hand onward.
            whole_image_cache
                .object_cache
                .extend(result.cache.object_cache);
        }

        // Stage 2: Voronoi, sourced from Scratchpad - the recommended setup
        // for a pure object-manipulation step with no pixel input of its
        // own - run exactly once against every tile's combined objects, not
        // once per tile.
        let mut stage2 = Pipeline::new(
            PipelineId(2),
            CorePipelineSettings {
                start_image: ImageAddress::Scratchpad,
            },
        );
        stage2.add_command(Box::new(Voronoi {
            centers: ObjectClass::Valid(1),
            center_filter_classes: vec![],
            mask: ObjectClass::Unset,
            mask_filter_classes: vec![],
            output_class: ObjectClass::Valid(99),
            unit: SizeUnits::Pixels,
            max_radius: 0.0,
            exclude_areas_at_the_edges: false,
            exclude_areas_with_no_center: false,
        }));
        let result = stage2
            .run_commands(PathBuf::default(), None, whole_image_cache, None, false)
            .expect("whole-image stage must not fail");

        result.cache
    }

    #[test]
    fn voronoi_regions_are_computed_once_across_the_whole_image_not_per_tile() {
        // Regression test for the whole bug class: a per-tile algorithm computing
        // pixel coordinates from the *full* image instead of the *current tile*
        // silently corrupts results (or, once intensity sampling reads from the
        // tile-local channel buffer, panics outright) for any image split into more
        // than one tile. Two seed objects placed symmetrically around the tile
        // boundary drive the real ExtractObjects -> Voronoi pipeline chain,
        // exactly as JobExecutor does: `ExtractObjects` per tile, `Voronoi` once
        // for the whole image after both tiles' objects are combined - so each
        // region is bounded by the *other* seed, not by its own tile's edge.
        let full_w = 40;
        let full_h = 20;
        let tile_size = (20, 20);

        let cache = run_pipeline_across_tiles(
            full_w,
            full_h,
            &[
                ((0, 0), tile_size, [8, 8, 11, 11]),
                ((20, 0), tile_size, [8, 8, 11, 11]),
            ],
        );

        let mut regions: Vec<&Object> = cache
            .object_cache
            .values()
            .filter(|r| r.has_object_class(&ObjectClass::Valid(99)))
            .collect();
        regions.sort_by_key(|r| r.bbox[0]);
        assert_eq!(
            regions.len(),
            2,
            "one Voronoi region per seed, computed together over the whole image"
        );

        // The two seeds sit at absolute x-centers 9.5 and 29.5 - symmetric
        // around the tile boundary at x=20 - so a correct whole-image
        // tessellation splits the canvas into two equal halves at that
        // midline, each seed's region bounded by the *other* seed rather
        // than by its own tile's edge.
        let total_area: usize = regions.iter().map(|r| r.area).sum();
        assert_eq!(
            total_area,
            full_w * full_h,
            "the two regions must tile the whole image with no gaps or overlap"
        );
        assert_eq!(regions[0].bbox, [0, 0, 19, 19]);
        assert_eq!(regions[1].bbox, [20, 0, 39, 19]);
    }

    fn default_image_meta(size: ImageSize) -> GlobalImageMeta {
        GlobalImageMeta {
            full_image_width: size,
            is_rgb: false,
            nr_of_bits: 8,
            pixel_sizes: crate::image::PixelSizes {
                px_size_x: 1.0,
                px_size_y: 1.0,
                px_size_z: 1.0,
            },
        }
    }

    /// Mutates a single pixel in place via [`PipelineContext::get_f32_gray_image_mut`],
    /// the copy-on-write boundary. Used to test that boundary directly rather
    /// than through a full algorithm's math.
    struct SetFirstPixel {
        value: f32,
    }

    impl ImageAlgorithm for SetFirstPixel {
        fn execute(
            &self,
            ctx: &mut PipelineContext,
            _cache: &mut GlobalPipelineCache,
        ) -> Result<(), InternalErrors> {
            let img = ctx.get_f32_gray_image_mut()?;
            img.as_slice_mut()[0] = self.value;
            Ok(())
        }

        fn name(&self) -> &'static str {
            "SetFirstPixel"
        }

        fn cite(&self) -> Option<&'static CitationMetadata> {
            None
        }

        fn execution_scope(&self) -> ExecutionScope {
            ExecutionScope::Tile
        }
    }

    #[test]
    fn read_only_pipeline_shares_the_cached_image_without_cloning() {
        let size = ImageSize {
            width: 4,
            height: 4,
        };
        let channel_image = Arc::new(ImageContainer::F32Gray(ManagedImage {
            data: Image::<f32, 1, CpuAllocator>::new(size, vec![2.0f32; 16], CpuAllocator).unwrap(),
            tile_offset: Point2d { x: 0, y: 0 },
            plane: None,
        }));

        let mut cache = GlobalPipelineCache::default();
        cache.image_meta = default_image_meta(size);
        cache.add_to_channel_cache(
            Arc::clone(&channel_image),
            0,
            crate::ImageTile {
                offset_x: 0,
                offset_y: 0,
                width: size.width,
                height: size.height,
            },
        );

        // FakeSegmenter only ever touches segmentation_map/instance_map, never
        // ctx.image - exactly the shape of a real segmentation+measurement
        // pipeline (e.g. Cellpose -> ExtractObjects).
        let mut pipeline = Pipeline::new(
            PipelineId(1),
            CorePipelineSettings {
                start_image: ImageAddress::Channel(0),
            },
        );
        pipeline.add_command(Box::new(FakeSegmenter { rect: [0, 0, 1, 1] }));

        let result = pipeline
            .run_commands(
                PathBuf::default(),
                Some(crate::ImageTile {
                    offset_x: 0,
                    offset_y: 0,
                    width: size.width,
                    height: size.height,
                }),
                cache,
                None,
                false,
            )
            .expect("run must not fail");

        assert!(
            Arc::ptr_eq(&result.image, &channel_image),
            "a pipeline that never mutates the image must share the cache's Arc, not copy it"
        );
    }

    #[test]
    fn mutating_command_clones_before_writing_leaving_the_cached_image_untouched() {
        let size = ImageSize {
            width: 2,
            height: 2,
        };
        let channel_image = Arc::new(ImageContainer::F32Gray(ManagedImage {
            data: Image::<f32, 1, CpuAllocator>::new(size, vec![1.0f32; 4], CpuAllocator).unwrap(),
            tile_offset: Point2d { x: 0, y: 0 },
            plane: None,
        }));

        let mut cache = GlobalPipelineCache::default();
        cache.image_meta = default_image_meta(size);
        cache.add_to_channel_cache(
            Arc::clone(&channel_image),
            0,
            crate::ImageTile {
                offset_x: 0,
                offset_y: 0,
                width: size.width,
                height: size.height,
            },
        );

        let mut pipeline = Pipeline::new(
            PipelineId(1),
            CorePipelineSettings {
                start_image: ImageAddress::Channel(0),
            },
        );
        pipeline.add_command(Box::new(SetFirstPixel { value: 9.0 }));

        let result = pipeline
            .run_commands(
                PathBuf::default(),
                Some(crate::ImageTile {
                    offset_x: 0,
                    offset_y: 0,
                    width: size.width,
                    height: size.height,
                }),
                cache,
                None,
                false,
            )
            .expect("run must not fail");

        // The pipeline's own output reflects the mutation...
        match result.image.as_ref() {
            ImageContainer::F32Gray(img) => assert_eq!(img.as_slice()[0], 9.0),
            other => panic!("expected F32Gray, got {other:?}"),
        }

        // ...but the cache's original Arc must never observe it: make_mut has
        // to clone before writing, not mutate the shared buffer in place.
        match channel_image.as_ref() {
            ImageContainer::F32Gray(img) => {
                assert_eq!(img.as_slice()[0], 1.0, "cached image must remain unmutated")
            }
            other => panic!("expected F32Gray, got {other:?}"),
        }
        assert!(
            !Arc::ptr_eq(&result.image, &channel_image),
            "mutation must have produced a distinct buffer"
        );
    }
}
