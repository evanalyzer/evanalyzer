//! # image_cache
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-02-01
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **AGPL-3.0**.

use crate::algos::{ImageAlgorithm, PipelineCache, PipelineContext};
use evanalyzer_cfg::core_types::ImageAddress;
use evanalyzer_cfg::core_types::InternalErrors;
use macros::CommandsMeta;
use std::sync::Arc;

/// Defines the interaction type with the persistent image storage.
pub enum ImageCacheMode {
    /// Captures the current image from the pipeline and writes it to the cache.
    ///
    /// Used for "checkpointing" results at a specific stage in the pipeline
    /// for later comparison or retrieval.
    Store,

    /// Retrieves a previously stored image from the cache and injects it
    /// into the current pipeline context.
    ///
    /// This effectively replaces the current working image with the cached version.
    Load,
}

/// A filter that acts as a synchronization point between the pipeline and a storage backend.
///
/// `ImageCache` allows the pipeline to branch or "undo" operations by saving
/// states to a named address and reloading them as needed.
///
/// # Examples
///
/// ```
/// use imagec::backend::core::context::{ImageCache, ImageCacheMode, ImageAddress};
/// let checkpoint = ImageCache {
///     mode: ImageCacheMode::Store,
///     address: ImageAddress::from("pre_processed_state"),
/// };
/// ```
#[derive(CommandsMeta)]
#[cmdsmeta(category = "Preprocessing")]
pub struct ImageCache {
    /// Whether to save the current state to the cache or load a state from it.
    pub mode: ImageCacheMode,

    /// The unique identifier or memory slot where the image is stored.
    pub address: ImageAddress,
}

impl ImageAlgorithm for ImageCache {
    /// Transfers image data between the active pipeline context and the persistent cache.
    ///
    /// This method allows for non-linear pipelines by creating "save points" or
    /// restoring the image to a previous state before heavy processing was applied.
    ///
    /// # Behavior
    /// - **Load**: Attempts to find an image at the specified `address`. If found, it
    ///   replaces the current image in the context.
    /// - **Store**: Clones the current image from the context and saves it into
    ///   the cache at the specified `address`, overwriting any existing data there.
    ///
    /// # Errors
    ///
    /// Returns [`InternalErrors::CacheMiss`] if a `Load` operation is requested for
    /// an address that has not been initialized in the cache.
    fn execute(
        &self,
        ctx: &mut PipelineContext,
        cache: &mut PipelineCache,
    ) -> Result<(), InternalErrors> {
        match self.mode {
            ImageCacheMode::Load => {
                ctx.image = cache
                    .image_cache
                    .images
                    .get(&self.address)
                    .ok_or(InternalErrors::CacheMiss("".to_string()))?
                    .as_ref()
                    .clone();
                Ok(())
            }
            ImageCacheMode::Store => {
                cache
                    .image_cache
                    .images
                    .insert(self.address.clone(), Arc::new(ctx.image.clone()));
                Ok(())
            }
        }
    }

    fn name(&self) -> &'static str {
        "ImageCache"
    }
}

// --- Test ------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        F32Gray,
        image::{ImageContainer, PixelSizes},
        pipeline::pipeline::PipelineImageMeta,
    };
    use evanalyzer_cfg::core_types::{MemoryId, MemorySlot};
    use kornia_image::{Image, ImageSize};
    use kornia_tensor::CpuAllocator;

    #[test]
    fn test_image_cache_store_and_load() -> Result<(), Box<dyn std::error::Error>> {
        // 1. Setup: Create a dummy image
        let size = ImageSize {
            width: 2,
            height: 2,
        };
        let data = vec![1.0f32, 2.0, 3.0, 4.0];
        let test_image = Image::<f32, 1, CpuAllocator>::new(size, data, CpuAllocator)?;

        // 2. Setup: Context and Cache
        // Assuming PipelineContext has an 'image' and 'scratch_pad'
        let mut ctx = PipelineContext::new_from_image(
            PipelineImageMeta {
                image_tile_info: crate::ImageTile {
                    offset_x: 0,
                    offset_y: 0,
                    width: 2,
                    height: 2,
                },
                full_image_width: ImageSize {
                    width: 2,
                    height: 2,
                },
                is_rgb: false,
                nr_of_bits: 8,
                pixel_sizes: PixelSizes {
                    px_size_x: 1.0,
                    px_size_y: 1.0,
                    px_size_z: 1.0,
                },
            },
            ImageContainer::new_f32_gray_from_image_test(test_image),
        )?;

        let mut cache = PipelineCache::default();

        // Define an address (Memory slot M2)
        let address = ImageAddress::Memory(MemoryId::PipelineContext(2));

        // 3. Test STORE mode
        let store_algo = ImageCache {
            address: address.clone(),
            mode: ImageCacheMode::Store,
        };

        store_algo.execute(&mut ctx, &mut cache)?;

        // Verify it exists in the cache
        assert!(cache.image_cache.images.contains_key(&address));

        // 4. Modify context image so we can prove 'Load' actually changes it
        let empty_data = vec![0.0f32; 4];
        ctx.image = ImageContainer::new_f32_gray_from_image_test(Image::new(
            size,
            empty_data,
            CpuAllocator,
        )?);

        // 5. Test LOAD mode
        let load_algo = ImageCache {
            address: address.clone(),
            mode: ImageCacheMode::Load,
        };

        load_algo.execute(&mut ctx, &mut cache)?;

        // 6. Final Assertions
        if let ImageContainer::F32Gray(result_img) = ctx.image {
            let result_slice = result_img.as_slice();
            assert_eq!(result_slice[0], 1.0);
            assert_eq!(result_slice[3], 4.0);
            assert_eq!(result_img.size().width, 2);
        } else {
            panic!("Loaded image is not F32Gray");
        }

        Ok(())
    }

    #[test]
    fn test_load_non_existent_address_fails() {
        let size = ImageSize {
            width: 1,
            height: 1,
        };
        let mut ctx = PipelineContext::new_test::<F32Gray>(size).unwrap();
        let mut cache = PipelineCache::default();

        let address = ImageAddress::Memory(MemoryId::PipelineContext(8));
        let load_algo = ImageCache {
            address,
            mode: ImageCacheMode::Load,
        };

        let result = load_algo.execute(&mut ctx, &mut cache);
        assert!(result.is_err());
        // Verify it's a CacheMiss error
        match result {
            Err(InternalErrors::CacheMiss(_)) => (),
            _ => panic!("Expected CacheMiss error"),
        }
    }

    #[test]
    fn test_image_cache_name() {
        let algo = ImageCache {
            mode: ImageCacheMode::Store,
            address: ImageAddress::Memory(MemoryId::PipelineContext(1)),
        };
        assert_eq!(algo.name(), "ImageCache");
    }
}
