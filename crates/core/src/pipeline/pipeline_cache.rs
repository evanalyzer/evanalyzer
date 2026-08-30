use crate::{
    ImageTile,
    image::{ImageContainer, PixelSizes},
    pipeline::{image_cache::ImageCache, object_cache::ObjectCache, pipeline::PipelineImageMeta},
};
use evanalyzer_cfg::core_types::MemoryId;
use kornia_apriltag::utils::Point2d;
use kornia_image::{Image, ImageSize};
use kornia_tensor::CpuAllocator;
use std::{path::PathBuf, sync::Arc};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CacheAddress {
    Scratchpad,
    Memory(MemoryId),
    Channel((i32, ImageTile)),
}

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
    pub image_cache: ImageCache,
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
        self.image_cache.retain(|key| {
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
    }

    pub fn get_image_from_channel_cache_for_tile(
        &self,
        channel_idx: (i32, ImageTile),
    ) -> Option<Arc<ImageContainer>> {
        self.image_cache.get(&CacheAddress::Channel(channel_idx))
    }

    pub fn get_image_from_memory_cache(&self, memory_id: MemoryId) -> Option<Arc<ImageContainer>> {
        self.image_cache
            .get(&CacheAddress::Memory(memory_id.clone()))
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

    /// Registered `(channel, tile)` keys, without loading any pixel data -
    /// see `ImageCache::keys`. Cheap: only reads the always-in-memory index,
    /// never touches the hot cache or disk.
    pub fn channel_keys(&self) -> impl Iterator<Item = (i32, ImageTile)> + '_ {
        self.image_cache.keys().filter_map(|key| match key {
            CacheAddress::Channel(index) => Some(*index),
            _ => None,
        })
    }

    /// Iterates over all Channel images in the cache, loading each from disk
    /// (or the hot cache) as needed - see `ImageCache::get`. Returns an
    /// owned `Arc<ImageContainer>` per entry, not `&ImageContainer`: `get`
    /// can't hand back a borrow once `hot_cache` sits behind an `RwLock`
    /// (see its own doc comment), so this is the same trade for the same
    /// reason - cloning an `Arc` is a cheap refcount bump, not a data copy.
    ///
    /// Loads *every* registered channel entry - for a cache holding more
    /// than the current tile's own channels (e.g. the whole-image phase,
    /// once tiles are merged across the image), prefer
    /// [`resolve_channel_views_for_bbox`](Self::resolve_channel_views_for_bbox)
    /// instead, so a query only pays for the tiles it actually needs.
    pub fn iter_channels(
        &self,
    ) -> impl Iterator<Item = ((i32, ImageTile), Arc<ImageContainer>)> + '_ {
        self.channel_keys()
            .filter_map(|index| Some((index, self.image_cache.get(&CacheAddress::Channel(index))?)))
    }

    /// Resolves every registered channel image into flat pixel slices ready
    /// for direct-index sampling, paired with whether each is RGB (needing
    /// luminance conversion via [`sample_channel_pixel`]). Used by
    /// `ExtractObjects`, which only ever has one tile's worth of channels
    /// loaded at a time (see `prepare_pipeline_cache`) - "every registered
    /// entry" and "everything relevant" are the same set there, so there's
    /// no bbox to scope by. For anything that can see a larger cache (the
    /// whole-image phase), use
    /// [`resolve_channel_views_for_bbox`](Self::resolve_channel_views_for_bbox).
    pub fn resolve_channel_views(&self) -> Vec<((i32, ImageTile), bool, Arc<ImageContainer>)> {
        self.iter_channels()
            .filter_map(|(idx, container)| {
                let is_rgb = channel_is_rgb(&container)?;
                Some((idx, is_rgb, container))
            })
            .collect()
    }

    /// Like [`resolve_channel_views`](Self::resolve_channel_views), but only
    /// loads the channel entries whose tile bounds actually intersect
    /// `bbox` ([x_min, y_min, x_max, y_max], inclusive - matching
    /// `Object::bbox`), instead of every registered entry.
    ///
    /// `ImageCache` is disk-backed (see its own doc comments): once the
    /// whole-image phase merges every tile's channel images into one cache,
    /// `resolve_channel_views` would force a disk load for *every* tile in
    /// the image just to measure one object's intensities, even though a
    /// single object's mask - the only thing `Object::measure_intensities`
    /// actually samples - typically only touches a handful of tiles.
    /// Filtering by key (cheap - see `channel_keys`) before ever calling
    /// `get` (which may hit disk) keeps that query proportional to the
    /// object's own size, not the whole image's.
    pub fn resolve_channel_views_for_bbox(
        &self,
        bbox: [u32; 4],
    ) -> Vec<((i32, ImageTile), bool, Arc<ImageContainer>)> {
        let [x_min, y_min, x_max, y_max] = bbox;
        self.channel_keys()
            .filter(|(_, tile)| tile_intersects_bbox(tile, x_min, y_min, x_max, y_max))
            .filter_map(|index| {
                let container = self.image_cache.get(&CacheAddress::Channel(index))?;
                let is_rgb = channel_is_rgb(&container)?;
                Some((index, is_rgb, container))
            })
            .collect()
    }
}

/// Whether a tile ([offset_x, offset_x + width) x [offset_y, offset_y +
/// height), i.e. exclusive upper bounds) overlaps an inclusive pixel bbox
/// ([x_min, x_max] x [y_min, y_max], matching `Object::bbox`'s convention).
fn tile_intersects_bbox(tile: &ImageTile, x_min: u32, y_min: u32, x_max: u32, y_max: u32) -> bool {
    let tile_x_min = tile.offset_x as u32;
    let tile_y_min = tile.offset_y as u32;
    let tile_x_max = tile_x_min + tile.width as u32;
    let tile_y_max = tile_y_min + tile.height as u32;
    tile_x_min <= x_max && x_min < tile_x_max && tile_y_min <= y_max && y_min < tile_y_max
}

/// Whether a channel container carries `f32` pixel data sampleable by
/// [`sample_channel_pixel`] - `Some(is_rgb)` for `F32Gray`/`F32Rgb`, `None`
/// for `U32` (label/instance maps, never intensity data).
fn channel_is_rgb(container: &ImageContainer) -> Option<bool> {
    match container {
        ImageContainer::F32Gray(_) => Some(false),
        ImageContainer::F32Rgb(_) => Some(true),
        ImageContainer::U32(_) => None,
    }
}

/// Extracts the flat pixel slice from a channel container resolved by
/// [`GlobalPipelineCache::resolve_channel_views`]/
/// [`resolve_channel_views_for_bbox`](GlobalPipelineCache::resolve_channel_views_for_bbox),
/// for use with [`sample_channel_pixel`]. `None` for `U32` - both resolvers
/// already filter those out via `channel_is_rgb`, so this should never
/// actually return `None` for anything they returned.
pub fn channel_pixel_slice(container: &ImageContainer) -> Option<&[f32]> {
    match container {
        ImageContainer::F32Gray(img) => Some(img.as_slice()),
        ImageContainer::F32Rgb(img) => Some(img.as_slice()),
        ImageContainer::U32(_) => None,
    }
}

/// Samples one pixel from a channel view resolved by [`GlobalPipelineCache::resolve_channel_views`],
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
        // Not `Arc::ptr_eq`: `ImageCache` is disk-backed, so a fetched image may be
        // a freshly reconstructed `Arc` (same content, different allocation) rather
        // than the exact one inserted - see `ImageCache::get`'s own doc comment.
        assert_eq!(channel_pixel_slice(&fetched), channel_pixel_slice(&image));
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
        // Not `Arc::ptr_eq` - see `channel_cache_round_trip`'s comment.
        assert_eq!(channel_pixel_slice(&fetched), channel_pixel_slice(&image));
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
        // Not `Arc::ptr_eq` - see `channel_cache_round_trip`'s comment.
        assert_eq!(channel_pixel_slice(&fetched), channel_pixel_slice(&image));
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
        // Not `Arc::ptr_eq` - see `channel_cache_round_trip`'s comment.
        assert_eq!(channel_pixel_slice(&fetched), channel_pixel_slice(&image));
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

        let (idx0, is_rgb0, container0) = &views[0];
        assert_eq!(idx0.0, 0);
        assert!(!is_rgb0);
        assert_eq!(channel_pixel_slice(container0), Some([5.0].as_slice()));

        let (idx1, is_rgb1, container1) = &views[1];
        assert_eq!(idx1.0, 1);
        assert!(is_rgb1);
        assert_eq!(
            channel_pixel_slice(container1),
            Some([1.0, 2.0, 3.0].as_slice())
        );
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
