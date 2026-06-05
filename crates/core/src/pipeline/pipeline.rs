use crate::{
    ImageTile,
    algos::ImageAlgorithm,
    image::{ImageContainer, PixelSizes},
    pipeline::{pipeline_cache::PipelineCache, pipeline_context::PipelineContext},
};
use evanalyzer_cfg::core_types::{ImageAddress, InternalErrors, PipelineId};
use kornia_image::ImageSize;
use log::info;
use std::time::Instant;

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
        info!("Executed pipeline steps {} in {:?}", self.id, start.elapsed());
        Ok(PipelineResult {
            image: ctx.image,
            cache,
            breakpoint_hit: false,
            breakpoint_snapshot,
        })
    }
}
