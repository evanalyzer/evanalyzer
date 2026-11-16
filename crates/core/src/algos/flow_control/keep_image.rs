//! # keep_image
//!
//! Keeps an image of the tile pass - the pipeline's image at a step, a
//! memory slot - for a whole-image command that reads it.
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-11-16
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.

use crate::algos::{ExecutionScope, ImageAlgorithm};
use crate::pipeline::{
    pipeline_cache::{CacheAddress, GlobalPipelineCache, KeptImage, KeptSource},
    pipeline_context::PipelineContext,
};
use evanalyzer_cfg::core_types::{CitationMetadata, ImageTile, InternalErrors, MemoryId};

/// Keeps `source` of each tile under [`CacheAddress::Kept`] - added by the
/// job generator to the tile commands at the place of a whole-image command
/// that reads it (`ImageAlgorithm::kept_image`): whole-image commands run
/// after every tile command, when the pipeline's image and its memory slots
/// are gone. The command frees the kept tiles when it's done.
///
/// `pub(crate)` - internal: a fully `pub` command here would become a
/// pipeline command the user can add.
pub(crate) struct KeepImage {
    pub id: KeptImage,
    pub source: KeptSource,
    /// The whole-image command it keeps the image for (in errors).
    pub command: &'static str,
}

impl ImageAlgorithm for KeepImage {
    fn execute(
        &self,
        ctx: &mut PipelineContext,
        cache: &mut GlobalPipelineCache,
    ) -> Result<(), InternalErrors> {
        // No copy: shares the buffer (a later step changing the pipeline's
        // image copies it then).
        let image = match self.source {
            KeptSource::PipelineStep => ctx.image.clone(),
            KeptSource::Memory(id) => cache
                .image_cache
                .get(&CacheAddress::memory(id, ctx.scope)?)
                .ok_or_else(|| {
                    InternalErrors::CacheMiss(format!(
                        "{}: memory slot {} is empty at this step - store an image there \
                         (Image Cache) before it",
                        self.command,
                        slot_number(id)
                    ))
                })?,
        };
        let size = image.size();
        let tile = ImageTile {
            offset_x: ctx.image_meta.image_tile_info.offset_x,
            offset_y: ctx.image_meta.image_tile_info.offset_y,
            width: size.width,
            height: size.height,
        };
        cache
            .image_cache
            .insert(CacheAddress::Kept((self.id, tile)), image);
        Ok(())
    }

    fn name(&self) -> &'static str {
        "Keep Image"
    }

    fn cite(&self) -> Vec<&'static CitationMetadata> {
        Vec::new()
    }

    fn execution_scope(&self) -> ExecutionScope {
        ExecutionScope::Tile
    }

    /// The kept image's own copy when a later step changes the pipeline's
    /// image (f32 gray).
    fn ram_per_tile_pixel(&self) -> f64 {
        match self.source {
            KeptSource::PipelineStep => 4.0,
            KeptSource::Memory(_) => 0.0,
        }
    }
}

fn slot_number(id: MemoryId) -> u32 {
    match id {
        MemoryId::PipelineContext(n) | MemoryId::ProjectCache(n) | MemoryId::ModuleContext(n) => n,
    }
}
