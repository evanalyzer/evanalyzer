use crate::{image::ImageContainer, pipeline::pipeline::PipelineImageMeta, roi::Roi};
use evanalyzer_cfg::core_types::{ImageAddress, MemoryId, ObjectId};
use kornia_apriltag::utils::Point2d;
use kornia_image::Image;
use kornia_tensor::CpuAllocator;
use std::{
    collections::{BTreeMap, HashMap},
    path::PathBuf,
    sync::Arc,
};

/// This is a map which stores an image to the RAM.
/// The image can be addressed either by Image plane or a memory ID
pub type ImageMap = HashMap<ImageAddress, Arc<ImageContainer>>;

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

#[derive(Default)]
pub struct ImageCache {
    pub image_meta: PipelineImageMeta,
    pub images: ImageMap,
}

impl ImageCache {
    pub fn clear_pipeline_context(&mut self) {
        self.images.retain(|key, _| {
            match key {
                // Always keep Channel types
                ImageAddress::Channel(_) => true,
                // If it is a Memory(PipelineContext), return false to remove it
                ImageAddress::Memory(MemoryId::PipelineContext(_)) => false,
                _ => false,
            }
        });
    }

    pub fn add_to_channel_cache(&mut self, image: Arc<ImageContainer>, channel_idx: i32) {
        self.images
            .insert(ImageAddress::Channel(channel_idx), image);
    }

    pub fn get_image_from_channel_cache(&self, channel_idx: i32) -> Option<Arc<ImageContainer>> {
        self.images
            .get(&ImageAddress::Channel(channel_idx))
            .cloned()
    }

    pub fn get_image_from_memory_cache(&self, memory_id: MemoryId) -> Option<Arc<ImageContainer>> {
        self.images
            .get(&ImageAddress::Memory(memory_id.clone()))
            .cloned()
    }

    pub fn get_image_from_cache(&self, cache_slot: &ImageAddress) -> Option<Arc<ImageContainer>> {
        match cache_slot {
            ImageAddress::Scratchpad => match self.image_meta.is_rgb {
                true => Some(Arc::new(ImageContainer::F32Rgb(crate::ManagedImage {
                    data: Image::<f32, 3, CpuAllocator>::new(
                        kornia_image::ImageSize {
                            width: self.image_meta.image_tile_info.width,
                            height: self.image_meta.image_tile_info.height,
                        },
                        vec![
                            0f32;
                            self.image_meta.image_tile_info.width
                                * self.image_meta.image_tile_info.height
                        ],
                        CpuAllocator,
                    )
                    .expect("Could not allocate memory for image scratchpad"),
                    tile_offset: Point2d {
                        x: self.image_meta.image_tile_info.offset_x,
                        y: self.image_meta.image_tile_info.offset_y,
                    },
                    plane: None,
                }))),
                false => Some(Arc::new(ImageContainer::F32Gray(crate::ManagedImage {
                    data: Image::<f32, 1, CpuAllocator>::new(
                        kornia_image::ImageSize {
                            width: self.image_meta.image_tile_info.width,
                            height: self.image_meta.image_tile_info.height,
                        },
                        vec![
                            0f32;
                            self.image_meta.image_tile_info.width
                                * self.image_meta.image_tile_info.height
                        ],
                        CpuAllocator,
                    )
                    .expect("Could not allocate memory for image scratchpad"),
                    tile_offset: Point2d {
                        x: self.image_meta.image_tile_info.offset_x,
                        y: self.image_meta.image_tile_info.offset_y,
                    },
                    plane: None,
                }))),
            },
            ImageAddress::Memory(memory_id) => self.get_image_from_memory_cache(memory_id.clone()),
            ImageAddress::Channel(channel_idx) => {
                self.get_image_from_channel_cache(channel_idx.clone())
            }
        }
    }

    /// Iterates over all Channel images in the cache.
    /// Returns an iterator of (channel_index, &ImageContainer)
    pub fn iter_channels(&self) -> impl Iterator<Item = (i32, &ImageContainer)> {
        self.images.iter().filter_map(|(key, container)| {
            if let ImageAddress::Channel(index) = key {
                // Convert the &Arc<ImageContainer> into &ImageContainer
                Some((*index, container.as_ref()))
            } else {
                None
            }
        })
    }

    /// Returns a snapshot of all Channel images as a Vector of (index, reference).
    pub fn get_channel_slice(&self) -> Vec<(i32, Arc<ImageContainer>)> {
        self.images
            .iter()
            .filter_map(|(key, container)| {
                if let ImageAddress::Channel(index) = key {
                    // Convert &Arc<ImageContainer> to &ImageContainer
                    Some((*index, container.clone()))
                } else {
                    None
                }
            })
            .collect() // Collects into a Vec
    }
}

#[derive(Default)]
pub struct PipelineCache {
    pub image_cache: ImageCache,
    pub roi_cache: BTreeMap<ObjectId, Roi>,
    pub image_rel_path: PathBuf,
}
