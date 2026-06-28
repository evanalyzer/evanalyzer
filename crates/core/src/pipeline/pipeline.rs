use crate::{
    ImageTile,
    algos::ImageAlgorithm,
    image::{ImageContainer, PixelSizes},
    pipeline::{pipeline_cache::PipelineCache, pipeline_context::PipelineContext},
};
use evanalyzer_cfg::core_types::{ImageAddress, InternalErrors, PipelineId};
use kornia_image::ImageSize;
use log::info;
use std::{path::PathBuf, time::Instant};

pub struct PipelineResult {
    pub image: ImageContainer,
    pub cache: PipelineCache,
    /// True when the pipeline stopped early due to a Stop breakpoint.
    pub breakpoint_hit: bool,
    /// Populated when a Snapshot breakpoint was reached: the image captured
    /// at that step while the pipeline continued to run to completion.
    pub breakpoint_snapshot: Option<ImageContainer>,
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
    pub nr_of_bits: u8,
    /// Sizes of the image pixels in nm
    pub pixel_sizes: PixelSizes,
}

pub struct Pipeline {
    pub id: PipelineId,
    pub dependencies: Vec<PipelineId>,
    pub settings: CorePipelineSettings,
    pub commands: Vec<Box<dyn ImageAlgorithm>>,
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
        }
    }

    /// Add a new command to the end of the pipeline
    pub fn add_command(&mut self, command: Box<dyn ImageAlgorithm>) {
        self.commands.push(command);
    }

    // Add dependency
    pub fn add_dependency(&mut self, pipeline_id: PipelineId) {
        if !self.dependencies.contains(&pipeline_id) {
            self.dependencies.push(pipeline_id);
        }
    }

    /// Execute all commands in sequence.
    ///
    /// `breakpoint_step` identifies a step index (0-based) at which to act.
    /// `snapshot_mode`:
    ///   - `false` (Stop) — stop execution at that step and return early.
    ///   - `true`  (Snapshot) — clone the image at that step, then continue
    ///     to completion; the clone is returned in `breakpoint_snapshot`.
    pub fn run(
        &self,
        output_path: PathBuf,
        mut cache: PipelineCache,
        breakpoint_step: Option<i32>,
        snapshot_mode: bool,
    ) -> Result<PipelineResult, InternalErrors> {
        let Some(initial_image) = cache
            .image_cache
            .get_image_from_cache(&self.settings.start_image)
        else {
            return Err(InternalErrors::CacheMiss("Image not found in cache".into()));
        };

        let mut ctx = PipelineContext::new_from_image(
            output_path,
            cache.image_cache.image_meta.clone(),
            initial_image.as_ref().clone(),
        )?;
        let start = Instant::now();
        let mut breakpoint_snapshot: Option<ImageContainer> = None;

        for (idx, command) in self.commands.iter().enumerate() {
            let step_start = Instant::now();
            command.execute(&mut ctx, &mut cache)?;
            let duration = step_start.elapsed();
            info!("Executed {} in {:?}", command.name(), duration);

            if breakpoint_step == Some(idx as i32) {
                if snapshot_mode {
                    // Snapshot: capture but continue running.
                    breakpoint_snapshot = Some(ctx.image.clone());
                } else {
                    // Stop: return immediately with the intermediate image.
                    cache.image_cache.clear_pipeline_context();
                    info!(
                        "Breakpoint (stop) at step {} of pipeline {} in {:?}",
                        idx,
                        self.id,
                        start.elapsed()
                    );
                    return Ok(PipelineResult {
                        image: ctx.image,
                        cache,
                        breakpoint_hit: true,
                        breakpoint_snapshot: None,
                    });
                }
            }
        }

        cache.image_cache.clear_pipeline_context();
        info!(
            "Executed pipeline steps {} in {:?}",
            self.id,
            start.elapsed()
        );
        Ok(PipelineResult {
            image: ctx.image,
            cache,
            breakpoint_hit: false,
            breakpoint_snapshot,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        ManagedImage, Roi,
        algos::{ExtractRois, Voronoi},
    };
    use evanalyzer_cfg::core_types::{ImageAddress, ObjectClass, SizeUnits};
    use kornia_apriltag::utils::Point2d;
    use kornia_image::{Image, ImageSize};
    use kornia_tensor::CpuAllocator;

    /// Test-only stand-in for real segmentation (Threshold + ConnectedComponents):
    /// stamps one rectangular object - at tile-local coordinates - directly into
    /// `ctx.segmentation_map`/`ctx.instance_map`, so the test can focus on whether
    /// the *tile-awareness* of downstream steps (ExtractRois, Voronoi) holds, not on
    /// thresholding behaviour.
    struct FakeSegmenter {
        rect: [usize; 4], // [x_min, y_min, x_max, y_max], tile-local, inclusive
    }

    impl ImageAlgorithm for FakeSegmenter {
        fn execute(
            &self,
            ctx: &mut PipelineContext,
            _cache: &mut PipelineCache,
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
    }

    fn run_two_stage_pipeline_on_tile(
        full_w: usize,
        full_h: usize,
        tile_offset: (usize, usize),
        tile_size: (usize, usize),
        object_rect: [usize; 4],
    ) -> PipelineCache {
        let meta = PipelineImageMeta {
            image_tile_info: crate::ImageTile {
                offset_x: tile_offset.0,
                offset_y: tile_offset.1,
                width: tile_size.0,
                height: tile_size.1,
            },
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

        let mut cache = PipelineCache::default();
        cache.image_cache.image_meta = meta;
        let channel = Image::<f32, 1, CpuAllocator>::new(
            ImageSize {
                width: tile_size.0,
                height: tile_size.1,
            },
            vec![2.0f32; tile_size.0 * tile_size.1],
            CpuAllocator,
        )
        .unwrap();
        cache.image_cache.add_to_channel_cache(
            std::sync::Arc::new(ImageContainer::F32Gray(ManagedImage {
                data: channel,
                tile_offset: Point2d {
                    x: tile_offset.0,
                    y: tile_offset.1,
                },
                plane: None,
            })),
            0,
        );

        // Stage 1: segment + extract ROIs from real per-tile pixel data, exactly the
        // way JobExecutor runs a tile's first pipeline.
        let mut stage1 = Pipeline::new(
            PipelineId(1),
            CorePipelineSettings {
                start_image: ImageAddress::Channel(0),
            },
        );
        stage1.add_command(Box::new(FakeSegmenter { rect: object_rect }));
        stage1.add_command(Box::new(ExtractRois {
            max_objects_before_fail: 100_000,
        }));
        let result1 = stage1
            .run(PathBuf::default(), cache, None, false)
            .expect("stage 1 must not fail");

        // Stage 2: Voronoi, sourced from Scratchpad - the recommended setup for a
        // pure object-manipulation step with no pixel input of its own - reading the
        // center ROI stage 1 just produced via the cache that's threaded between
        // both pipelines on this same tile.
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
        let result2 = stage2
            .run(PathBuf::default(), result1.cache, None, false)
            .expect("stage 2 must not fail");

        result2.cache
    }

    #[test]
    fn voronoi_regions_stay_within_their_own_tile_across_a_multi_tile_image() {
        // Regression test for the whole bug class: a per-tile algorithm computing
        // pixel coordinates from the *full* image instead of the *current tile*
        // silently corrupts results (or, once intensity sampling reads from the
        // tile-local channel buffer, panics outright) for any image split into more
        // than one tile. This drives the real ExtractRois -> Voronoi pipeline chain,
        // exactly as JobExecutor does per tile, against two adjacent tiles cut from
        // one larger image, and checks neither tile's results leak past its own
        // absolute bounds.
        let full_w = 40;
        let full_h = 20;
        let tile_size = (20, 20);

        let cache_a = run_two_stage_pipeline_on_tile(
            full_w,
            full_h,
            (0, 0),
            tile_size,
            [8, 8, 11, 11],
        );
        let cache_b = run_two_stage_pipeline_on_tile(
            full_w,
            full_h,
            (20, 0),
            tile_size,
            [8, 8, 11, 11],
        );

        for (cache, lo_x, hi_x) in [(&cache_a, 0u32, 20u32), (&cache_b, 20u32, 40u32)] {
            let regions: Vec<&Roi> = cache
                .roi_cache
                .values()
                .filter(|r| r.has_object_class(&ObjectClass::Valid(99)))
                .collect();
            assert_eq!(regions.len(), 1, "exactly one Voronoi region per tile");
            let [x_min, _, x_max, y_max] = regions[0].bbox;
            assert!(
                x_min >= lo_x && x_max < hi_x,
                "region bbox {:?} must stay within this tile's own x-range [{lo_x},{hi_x})",
                regions[0].bbox
            );
            assert!(y_max < 20);
            assert_eq!(regions[0].area, tile_size.0 * tile_size.1);
            assert_eq!(
                regions[0]
                    .intensities
                    .get(&0)
                    .expect("Voronoi region must have measured intensities")
                    .sum_intensity,
                (tile_size.0 * tile_size.1) as f64 * 2.0
            );
        }
    }
}
