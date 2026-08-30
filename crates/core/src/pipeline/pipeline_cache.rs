use crate::{
    ImageTile,
    image::{ImageContainer, PixelSizes},
    pipeline::{object_cache::ObjectCache, pipeline::PipelineImageMeta},
};
use evanalyzer_cfg::core_types::MemoryId;
use kornia_apriltag::utils::Point2d;
use kornia_image::{Image, ImageSize};
use kornia_tensor::CpuAllocator;
use std::{collections::HashMap, path::PathBuf, sync::Arc};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CacheAddress {
    Scratchpad,
    Memory(MemoryId),
    Channel((i32, ImageTile)),
}

/// This is a map which stores an image to the RAM.
/// The image can be addressed either by Image plane or a memory ID
pub type ImageMap = HashMap<CacheAddress, Arc<ImageContainer>>;

#[derive(Clone)]
pub struct GlobalImageMeta {
    /// The size of the original image (not the tile)
    pub full_image_width: ImageSize,
    /// True if this is a RGB image
    pub is_rgb: bool,
    /// Image bit depth: 8, 16, 32
    pub nr_of_bits: u16,
    /// Sizes of the image pixels in nm
    pub pixel_sizes: PixelSizes,
}

impl Default for GlobalImageMeta {
    fn default() -> Self {
        Self {
            full_image_width: ImageSize {
                width: 0,
                height: 0,
            },
            is_rgb: Default::default(),
            nr_of_bits: 16,
            pixel_sizes: Default::default(),
        }
    }
}

#[derive(Default, Clone)]
pub struct GlobalPipelineCache {
    pub image_cache: ImageMap,
    pub image_meta: GlobalImageMeta,
    pub object_cache: ObjectCache,
    pub image_rel_path: PathBuf,
}

mod tests {
    use crate::{image::PixelSizes, pipeline::pipeline_cache::PipelineImageMeta};
    use kornia_image::ImageSize;

    impl Default for PipelineImageMeta {
        fn default() -> Self {
            Self {
                image_tile_info: Default::default(),
                full_image_width: ImageSize {
                    width: 0,
                    height: 0,
                },
                is_rgb: Default::default(),
                nr_of_bits: 16,
                pixel_sizes: PixelSizes {
                    px_size_x: 1.0,
                    px_size_y: 1.0,
                    px_size_z: 1.0,
                },
            }
        }
    }
}

impl GlobalPipelineCache {
    pub fn clear_pipeline_context(&mut self) {
        self.image_cache.retain(|key, _| {
            match key {
                // Always keep Channel types
                CacheAddress::Channel(_) => true,
                // If it is a Memory(PipelineContext), return false to remove it
                CacheAddress::Memory(MemoryId::PipelineContext(_)) => false,
                _ => false,
            }
        });
    }

    pub fn add_to_channel_cache(
        &mut self,
        image: Arc<ImageContainer>,
        channel_idx: i32,
        image_tile_info: ImageTile,
    ) {
        self.image_cache
            .insert(CacheAddress::Channel((channel_idx, image_tile_info)), image);
    }

    pub fn get_image_from_channel_cache(
        &self,
        channel_idx: i32,
        image_tile_info: ImageTile,
    ) -> Option<Arc<ImageContainer>> {
        self.image_cache
            .get(&CacheAddress::Channel((channel_idx, image_tile_info)))
            .cloned()
    }

    pub fn get_image_from_channel_cache_for_tile(
        &self,
        channel_idx: (i32, ImageTile),
    ) -> Option<Arc<ImageContainer>> {
        self.image_cache
            .get(&CacheAddress::Channel(channel_idx))
            .cloned()
    }

    pub fn get_image_from_memory_cache(&self, memory_id: MemoryId) -> Option<Arc<ImageContainer>> {
        self.image_cache
            .get(&CacheAddress::Memory(memory_id.clone()))
            .cloned()
    }

    pub fn get_image_from_cache(
        &self,
        cache_slot: &CacheAddress,
        image_tile_info: ImageTile,
    ) -> Option<Arc<ImageContainer>> {
        match cache_slot {
            CacheAddress::Scratchpad => match self.image_meta.is_rgb {
                true => Some(Arc::new(ImageContainer::F32Rgb(crate::ManagedImage {
                    data: Image::<f32, 3, CpuAllocator>::new(
                        kornia_image::ImageSize {
                            width: image_tile_info.width,
                            height: image_tile_info.height,
                        },
                        vec![0f32; image_tile_info.width * image_tile_info.height * 3],
                        CpuAllocator,
                    )
                    .expect("Could not allocate memory for image scratchpad"),
                    tile_offset: Point2d {
                        x: image_tile_info.offset_x,
                        y: image_tile_info.offset_y,
                    },
                    plane: None,
                }))),
                false => Some(Arc::new(ImageContainer::F32Gray(crate::ManagedImage {
                    data: Image::<f32, 1, CpuAllocator>::new(
                        kornia_image::ImageSize {
                            width: image_tile_info.width,
                            height: image_tile_info.height,
                        },
                        vec![0f32; image_tile_info.width * image_tile_info.height],
                        CpuAllocator,
                    )
                    .expect("Could not allocate memory for image scratchpad"),
                    tile_offset: Point2d {
                        x: image_tile_info.offset_x,
                        y: image_tile_info.offset_y,
                    },
                    plane: None,
                }))),
            },
            CacheAddress::Memory(memory_id) => self.get_image_from_memory_cache(memory_id.clone()),
            CacheAddress::Channel(channel_idx) => {
                self.get_image_from_channel_cache_for_tile(channel_idx.clone())
            }
        }
    }

    /// Iterates over all Channel images in the cache.
    /// Returns an iterator of (channel_index, &ImageContainer)
    pub fn iter_channels(&self) -> impl Iterator<Item = ((i32, ImageTile), &ImageContainer)> {
        self.image_cache.iter().filter_map(|(key, container)| {
            if let CacheAddress::Channel(index) = key {
                // Convert the &Arc<ImageContainer> into &ImageContainer
                Some((*index, container.as_ref()))
            } else {
                None
            }
        })
    }

    /// Resolves the registered channel images into flat pixel slices ready for
    /// direct-index sampling, paired with whether each is RGB (needing luminance
    /// conversion via [`sample_channel_pixel`]). Shared by every algorithm that
    /// samples per-channel intensities for a set of pixels (`ExtractObjects`,
    /// colocalization intersection ROIs, ...) so the channel resolution and the
    /// RGB-luminance formula live in exactly one place.
    pub fn resolve_channel_views(&self) -> Vec<((i32, ImageTile), bool, &[f32])> {
        self.iter_channels()
            .filter_map(|(idx, container)| match container {
                ImageContainer::F32Gray(img) => Some((idx, false, img.as_slice())),
                ImageContainer::F32Rgb(img) => Some((idx, true, img.as_slice())),
                ImageContainer::U32(_) => None,
            })
            .collect()
    }
}

/// Samples one pixel from a channel view resolved by [`ImageCache::resolve_channel_views`],
/// converting RGB to perceptual luminance (BT.709) when `is_rgb` is set.
pub fn sample_channel_pixel(is_rgb: bool, slice: &[f32], sample: usize) -> f32 {
    if is_rgb {
        let idx = sample * 3;
        (0.2126 * slice[idx] + 0.7152 * slice[idx + 1] + 0.0722 * slice[idx + 2]).max(0.0)
    } else {
        slice[sample]
    }
}

#[cfg(test)]
mod cache_tests {
    use super::*;
    use crate::ManagedImage;
    use kornia_apriltag::utils::Point2d;
    use kornia_image::{Image, ImageSize};
    use kornia_tensor::CpuAllocator;

    fn gray_container(width: usize, height: usize, data: Vec<f32>) -> Arc<ImageContainer> {
        Arc::new(ImageContainer::F32Gray(ManagedImage {
            data: Image::<f32, 1, CpuAllocator>::new(
                ImageSize { width, height },
                data,
                CpuAllocator,
            )
            .unwrap(),
            tile_offset: Point2d { x: 0, y: 0 },
            plane: None,
        }))
    }

    fn rgb_container(width: usize, height: usize, data: Vec<f32>) -> Arc<ImageContainer> {
        Arc::new(ImageContainer::F32Rgb(ManagedImage {
            data: Image::<f32, 3, CpuAllocator>::new(
                ImageSize { width, height },
                data,
                CpuAllocator,
            )
            .unwrap(),
            tile_offset: Point2d { x: 0, y: 0 },
            plane: None,
        }))
    }

    fn u32_container(width: usize, height: usize, data: Vec<u32>) -> Arc<ImageContainer> {
        Arc::new(ImageContainer::U32(ManagedImage {
            data: Image::<u32, 1, CpuAllocator>::new(
                ImageSize { width, height },
                data,
                CpuAllocator,
            )
            .unwrap(),
            tile_offset: Point2d { x: 0, y: 0 },
            plane: None,
        }))
    }

    // ---- clear_pipeline_context ----

    #[test]
    fn clear_pipeline_context_keeps_channel_entries() {
        let mut cache = GlobalPipelineCache::default();
        cache.add_to_channel_cache(gray_container(1, 1, vec![1.0]), 0, ImageTile::default());

        cache.clear_pipeline_context();

        assert!(
            cache
                .get_image_from_channel_cache(0, ImageTile::default())
                .is_some()
        );
    }

    #[test]
    fn clear_pipeline_context_drops_pipeline_context_memory_entries() {
        let mut cache = GlobalPipelineCache::default();
        cache.image_cache.insert(
            CacheAddress::Memory(MemoryId::PipelineContext(1)),
            gray_container(1, 1, vec![1.0]),
        );

        cache.clear_pipeline_context();

        assert!(
            cache
                .get_image_from_memory_cache(MemoryId::PipelineContext(1))
                .is_none()
        );
    }

    /// Pins down the current (possibly surprising) behavior of the catch-all `_ => false`
    /// arm: the doc comment on `clear_pipeline_context` only mentions dropping
    /// `Memory(PipelineContext(_))`, but the match's fallback arm actually also drops
    /// every other non-Channel `ImageAddress` variant -- including `Scratchpad` and
    /// `Memory(ProjectCache(_))`. This test documents that reality; it is not asserting
    /// that this is the *intended* behavior.
    #[test]
    fn clear_pipeline_context_also_drops_scratchpad_and_project_cache_via_catch_all() {
        let mut cache = GlobalPipelineCache::default();
        cache
            .image_cache
            .insert(CacheAddress::Scratchpad, gray_container(1, 1, vec![2.0]));
        cache.image_cache.insert(
            CacheAddress::Memory(MemoryId::ProjectCache(7)),
            gray_container(1, 1, vec![3.0]),
        );
        cache.add_to_channel_cache(gray_container(1, 1, vec![4.0]), 5, ImageTile::default());

        cache.clear_pipeline_context();

        // Channel entry survives.
        assert!(
            cache
                .get_image_from_channel_cache(5, ImageTile::default())
                .is_some()
        );
        // Scratchpad and ProjectCache entries are both gone too, via the `_ => false` arm,
        // even though the doc comment only calls out PipelineContext memory slots.
        assert!(!cache.image_cache.contains_key(&CacheAddress::Scratchpad));
        assert!(
            !cache
                .image_cache
                .contains_key(&CacheAddress::Memory(MemoryId::ProjectCache(7)))
        );
        assert_eq!(cache.image_cache.len(), 1);
    }

    // ---- add_to_channel_cache / get_image_from_channel_cache ----

    #[test]
    fn channel_cache_round_trip() {
        let mut cache = GlobalPipelineCache::default();
        let image = gray_container(1, 1, vec![42.0]);

        cache.add_to_channel_cache(image.clone(), 3, ImageTile::default());

        let fetched = cache
            .get_image_from_channel_cache(3, ImageTile::default())
            .unwrap();
        assert!(Arc::ptr_eq(&fetched, &image));
    }

    #[test]
    fn channel_cache_miss_returns_none() {
        let cache = GlobalPipelineCache::default();
        assert!(
            cache
                .get_image_from_channel_cache(99, ImageTile::default())
                .is_none()
        );
    }

    // ---- get_image_from_memory_cache ----

    #[test]
    fn memory_cache_round_trip() {
        let mut cache = GlobalPipelineCache::default();
        let image = gray_container(1, 1, vec![7.0]);
        let id = MemoryId::PipelineContext(2);
        cache
            .image_cache
            .insert(CacheAddress::Memory(id), image.clone());

        let fetched = cache.get_image_from_memory_cache(id).unwrap();
        assert!(Arc::ptr_eq(&fetched, &image));
    }

    #[test]
    fn memory_cache_miss_returns_none() {
        let cache = GlobalPipelineCache::default();
        assert!(
            cache
                .get_image_from_memory_cache(MemoryId::PipelineContext(1))
                .is_none()
        );
    }

    // ---- get_image_from_cache ----

    #[test]
    fn get_image_from_cache_scratchpad_gray_is_zeroed_and_sized() {
        let mut cache = GlobalPipelineCache::default();
        cache.image_meta.is_rgb = false;

        let result = cache
            .get_image_from_cache(
                &CacheAddress::Scratchpad,
                ImageTile {
                    width: 3,
                    height: 2,
                    ..Default::default()
                },
            )
            .unwrap();

        match result.as_ref() {
            ImageContainer::F32Gray(img) => {
                assert_eq!(img.data.width(), 3);
                assert_eq!(img.data.height(), 2);
                assert!(img.as_slice().iter().all(|&v| v == 0.0));
            }
            other => panic!("expected F32Gray, got {other:?}"),
        }
    }

    #[test]
    fn get_image_from_cache_scratchpad_rgb_is_zeroed_and_sized() {
        let mut cache = GlobalPipelineCache::default();
        cache.image_meta.is_rgb = true;

        let result = cache
            .get_image_from_cache(
                &CacheAddress::Scratchpad,
                ImageTile {
                    width: 2,
                    height: 2,
                    ..Default::default()
                },
            )
            .unwrap();

        match result.as_ref() {
            ImageContainer::F32Rgb(img) => {
                assert_eq!(img.data.width(), 2);
                assert_eq!(img.data.height(), 2);
                assert!(img.as_slice().iter().all(|&v| v == 0.0));
            }
            other => panic!("expected F32Rgb, got {other:?}"),
        }
    }

    #[test]
    fn get_image_from_cache_memory_delegates() {
        let mut cache = GlobalPipelineCache::default();
        let image = gray_container(1, 1, vec![9.0]);
        let id = MemoryId::PipelineContext(4);
        cache
            .image_cache
            .insert(CacheAddress::Memory(id), image.clone());

        let fetched = cache
            .get_image_from_cache(&CacheAddress::Memory(id), ImageTile::default())
            .unwrap();
        assert!(Arc::ptr_eq(&fetched, &image));
    }

    #[test]
    fn get_image_from_cache_channel_delegates() {
        let mut cache = GlobalPipelineCache::default();
        let image = gray_container(1, 1, vec![11.0]);
        cache.add_to_channel_cache(image.clone(), 6, ImageTile::default());

        let fetched = cache
            .get_image_from_cache(
                &CacheAddress::Channel((6, ImageTile::default())),
                ImageTile::default(),
            )
            .unwrap();
        assert!(Arc::ptr_eq(&fetched, &image));
    }

    // ---- iter_channels ----

    #[test]
    fn iter_channels_only_yields_channel_entries() {
        let mut cache = GlobalPipelineCache::default();
        cache.add_to_channel_cache(gray_container(1, 1, vec![1.0]), 0, ImageTile::default());
        cache.add_to_channel_cache(gray_container(1, 1, vec![2.0]), 1, ImageTile::default());
        cache.image_cache.insert(
            CacheAddress::Memory(MemoryId::PipelineContext(1)),
            gray_container(1, 1, vec![3.0]),
        );
        cache
            .image_cache
            .insert(CacheAddress::Scratchpad, gray_container(1, 1, vec![4.0]));

        let mut indices: Vec<(i32, ImageTile)> =
            cache.iter_channels().map(|(idx, _)| idx).collect();
        indices.sort();

        assert_eq!(
            indices,
            vec![(0, ImageTile::default()), (1, ImageTile::default())]
        );
    }

    // ---- resolve_channel_views ----

    #[test]
    fn resolve_channel_views_returns_gray_and_rgb_but_skips_u32() {
        let mut cache = GlobalPipelineCache::default();
        cache.add_to_channel_cache(gray_container(1, 1, vec![5.0]), 0, ImageTile::default());
        cache.add_to_channel_cache(
            rgb_container(1, 1, vec![1.0, 2.0, 3.0]),
            1,
            ImageTile::default(),
        );
        cache.add_to_channel_cache(u32_container(1, 1, vec![7]), 2, ImageTile::default());

        let mut views = cache.resolve_channel_views();
        views.sort_by_key(|(idx, _, _)| *idx);

        assert_eq!(views.len(), 2);

        let (idx0, is_rgb0, slice0) = views[0];
        assert_eq!(idx0.0, 0);
        assert!(!is_rgb0);
        assert_eq!(slice0, &[5.0]);

        let (idx1, is_rgb1, slice1) = views[1];
        assert_eq!(idx1.0, 1);
        assert!(is_rgb1);
        assert_eq!(slice1, &[1.0, 2.0, 3.0]);
    }

    // ---- sample_channel_pixel ----

    #[test]
    fn sample_channel_pixel_rgb_uses_bt709_luminance() {
        let slice = [10.0f32, 20.0, 30.0];
        let expected = 0.2126 * 10.0 + 0.7152 * 20.0 + 0.0722 * 30.0;

        let result = sample_channel_pixel(true, &slice, 0);

        assert!((result - expected).abs() < 1e-6);
    }

    #[test]
    fn sample_channel_pixel_gray_is_passthrough() {
        let slice = [1.0f32, 2.0, 3.0];

        assert_eq!(sample_channel_pixel(false, &slice, 2), 3.0);
    }
}
