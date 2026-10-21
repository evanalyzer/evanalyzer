//! # load_channel
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-10-14
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **AGPL-3.0**.

use crate::algos::{ExecutionScope, GlobalPipelineCache, ImageAlgorithm, PipelineContext};
use crate::pipeline::pipeline_cache::{CacheAddress, ChannelVariant};
use evanalyzer_cfg::core_types::{CitationMetadata, ImageChannelIdx, InternalErrors, ZProjection};
use macros::CommandsMeta;

#[derive(CommandsMeta)]
pub enum ChannelZProjection {
    Auto,
    None,
    MaxIntensity,
    MinIntensity,
    AvgIntensity,
    SumIntensity,
    TakeTheMiddle,
}

/// Loads an image channel as the image the following steps work on.
///
/// Replaces the current image. Use it as the first step of a pipeline
/// without an image source, or to switch channels within a pipeline - e.g.
/// a module that detects nuclei in one channel and cells in another.
///
/// Projection: Auto loads the channel as the project's z-stack setting does.
/// Any other choice loads it with its own z-projection, or one z-plane (None).
/// Use it with a project that analyzes one projection or one plane: if the project analyzes the z-planes one by one, the pipeline runs once per plane on the same image and its objects repeat for every plane.
/// Objects found on another projection are still drawn over the project's image, and annotations are matched to the project's z-planes.
#[derive(CommandsMeta)]
#[cmdsmeta(category = "Preprocessing")]
pub struct LoadChannel {
    /// The image channel to load.
    #[cmdsmeta(summary = true)]
    pub channel: ImageChannelIdx,

    /// Optional z projection setting
    ///
    /// If set to Auto the global z-projection setting from the project is used.
    /// If not Auto the image channel is loaded with the given z-projection settings.
    #[cmdsmeta(default = ChannelZProjection::Auto,optional = true)]
    pub projection: ChannelZProjection,

    /// With projection None: the z-plane loaded (0 = the first).
    #[cmdsmeta(default = 0, optional = true)]
    pub z_index: u32,
}

impl ImageAlgorithm for LoadChannel {
    fn execute(
        &self,
        ctx: &mut PipelineContext,
        cache: &mut GlobalPipelineCache,
    ) -> Result<(), InternalErrors> {
        let tile = ctx.image_meta.image_tile_info.clone();
        let channel = self.channel.0 as i32;
        let address = match self.channel_variant() {
            Some(variant) => CacheAddress::ChannelVariant((variant, tile)),
            None => CacheAddress::Channel((channel, tile)),
        };
        // Cheap Arc clone: the cache and the context share the buffer.
        ctx.image = cache
            .image_cache
            .get(&address)
            .ok_or_else(|| missing_channel(cache, channel, tile))?;
        Ok(())
    }

    /// Not on Auto: the channel with this step's projection (or z-plane).
    fn channel_variant(&self) -> Option<ChannelVariant> {
        let projection = match self.projection {
            ChannelZProjection::Auto => return None,
            ChannelZProjection::None => ZProjection::None,
            ChannelZProjection::MaxIntensity => ZProjection::MaxIntensity,
            ChannelZProjection::MinIntensity => ZProjection::MinIntensity,
            ChannelZProjection::AvgIntensity => ZProjection::AvgIntensity,
            ChannelZProjection::SumIntensity => ZProjection::SumIntensity,
            ChannelZProjection::TakeTheMiddle => ZProjection::TakeTheMiddle,
        };
        Some(ChannelVariant {
            channel: self.channel.0 as i32,
            projection,
            z: match projection {
                ZProjection::None => self.z_index as i32,
                _ => 0,
            },
        })
    }

    fn name(&self) -> &'static str {
        "Load Image Channel"
    }

    fn cite(&self) -> Vec<&'static CitationMetadata> {
        vec![&CitationMetadata::DANMAYR]
    }

    fn execution_scope(&self) -> ExecutionScope {
        ExecutionScope::Tile
    }

    /// The scratch pad isn't used - the step only swaps the image.
    fn scratch_is_workspace(&self) -> bool {
        false
    }
}

/// The error for reading `channel`, which the image doesn't have - naming the
/// channels it does have, like the check at the pipeline start.
pub(crate) fn missing_channel(
    cache: &GlobalPipelineCache,
    channel: i32,
    tile: crate::ImageTile,
) -> InternalErrors {
    let mut available: Vec<i32> = cache
        .channel_keys()
        .filter(|(_, t)| *t == tile)
        .map(|(c, _)| c)
        .collect();
    available.sort_unstable();
    available.dedup();
    let available = if available.is_empty() {
        "none".to_string()
    } else {
        available
            .iter()
            .map(|c| c.to_string())
            .collect::<Vec<_>>()
            .join(", ")
    };
    InternalErrors::CacheMiss(format!(
        "reads image channel {channel}, which this image doesn't have (it has channels \
         {available}) - check the channel of the pipeline's load steps"
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::image::ImageContainer;
    use kornia_image::{Image, ImageSize};
    use std::sync::Arc;

    fn gray(value: f32) -> Arc<ImageContainer> {
        Arc::new(ImageContainer::new_f32_gray_from_image_test(
            Image::new(
                ImageSize {
                    width: 1,
                    height: 1,
                },
                vec![value],
            )
            .unwrap(),
        ))
    }

    fn ctx() -> PipelineContext {
        PipelineContext::new_from_image_test(
            Image::new(
                ImageSize {
                    width: 1,
                    height: 1,
                },
                vec![0.0],
            )
            .unwrap(),
        )
        .unwrap()
    }

    #[test]
    fn the_channel_becomes_the_current_image() {
        let mut ctx = ctx();
        let tile = ctx.image_meta.image_tile_info.clone();
        let mut cache = GlobalPipelineCache::default();
        cache.add_to_channel_cache(gray(5.0), 0, tile);
        cache.add_to_channel_cache(gray(7.0), 1, tile);

        LoadChannel {
            channel: ImageChannelIdx(1),
            projection: ChannelZProjection::Auto,
            z_index: 0,
        }
        .execute(&mut ctx, &mut cache)
        .unwrap();

        let ImageContainer::F32Gray(image) = ctx.image.as_ref() else {
            panic!("expected a gray image");
        };
        assert_eq!(image.as_slice(), [7.0]);
    }

    #[test]
    fn a_missing_channel_names_the_ones_the_image_has() {
        let mut ctx = ctx();
        let tile = ctx.image_meta.image_tile_info.clone();
        let mut cache = GlobalPipelineCache::default();
        cache.add_to_channel_cache(gray(5.0), 0, tile);
        cache.add_to_channel_cache(gray(7.0), 2, tile);

        let error = LoadChannel {
            channel: ImageChannelIdx(9),
            projection: ChannelZProjection::Auto,
            z_index: 0,
        }
        .execute(&mut ctx, &mut cache)
        .unwrap_err()
        .to_string();
        assert!(
            error.contains("reads image channel 9") && error.contains("channels 0, 2"),
            "{error}"
        );
    }
}
