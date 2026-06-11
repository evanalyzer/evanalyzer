//! # roi
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-02-03
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **AGPL-3.0**.

use crate::{
    ImageContainer, ImagePlane,
    algos::ImageAlgorithm,
    pipeline::{pipeline_cache::PipelineCache, pipeline_context::PipelineContext},
    roi::{Intensity, Roi, RoiInit},
};
use bitvec::{order::Lsb0, vec::BitVec};
use evanalyzer_cfg::core_types::{InternalErrors, ObjectClass, ObjectId, SegmentationClass};
use indexmap::IndexMap;
use kornia_apriltag::utils::Point2d;
use kornia_image::ImageSize;
use macros::CommandsMeta;
use std::sync::Arc;

/// Represents a bounded region of interest extracted from a labeled image.

/// A command to extract spatial statistics and bounding boxes from labeled objects.
#[derive(CommandsMeta)]
#[cmdsmeta(category = "measure")]
pub struct ExtractRois {
    /// Maximum allowed ROIs to extract.
    ///
    /// If this limit is exceeded the pipeline fails.
    /// This is a protection against memory overload.
    #[cmdsmeta(default = 100000, min = 100000, max = 100000, step = 1, optional = true)]
    pub max_objects_before_fail: i32,
}

impl ImageAlgorithm for ExtractRois {
    /// Extracts ROI data from a U32Label image.
    ///
    /// The process performs a single-pass scan ($O(N)$) to minimize CPU overhead:
    /// 1. **Extent Tracking**: For every non-zero pixel, the algorithm updates the
    ///    running min/max coordinates for that specific Label ID.
    /// 2. **Area Calculation**: Increments the pixel count for each unique ID.
    /// 3. **ROI Finalization**: Converts tracked extents into width/height dimensions.
    ///
    /// # Errors
    /// Returns [`InternalErrors::FormatMismatch`] if the input image is not U32Label.
    fn execute(
        &self,
        ctx: &mut PipelineContext,
        cache: &mut PipelineCache,
    ) -> Result<(), InternalErrors> {
        let size = ctx.get_image_size();
        let (w, h) = (size.width as usize, size.height as usize);

        // Semantic Labels (Pixel Class which defines which pixel belongs to which class)
        let segmentation_slice = ctx.get_segmentation_map()?.as_slice();

        // Instance IDs (Each individual object gets its own instance id)
        let instance_map_slice = ctx.get_instance_map()?.as_slice();

        // ConnectedComponents emits dense, contiguous instance ids (1..=max_id), so every
        // per-object accumulator can be indexed by id with a flat Vec. This avoids the
        // per-pixel HashMap lookups and id bookkeeping the previous version paid for.
        let max_id = instance_map_slice.iter().copied().max().unwrap_or(0) as usize;
        if max_id == 0 {
            return Ok(()); // no objects in this tile
        }

        // Dense re-indexing guarantees every id in 1..=max_id has at least one pixel.
        let object_count = max_id;
        if object_count > self.max_objects_before_fail as usize {
            return Err(InternalErrors::TooManyObjects(format!(
                "Detected {} objects, limit is {}. Adjust upstream filter parameters to reduce noise.",
                object_count, self.max_objects_before_fail
            )));
        }

        // Hoist loop-invariant lookups out of the per-pixel hot path. In particular
        // `get_channel_slice()` allocates a Vec and bumps Arc refcounts on every call,
        // so calling it once per pixel dominated the runtime.
        let tile_offset = ctx.get_image_tile_offset();
        let full_image_size = ctx.full_image_size();
        let channels = cache.image_cache.get_channel_slice();
        let plane = ctx
            .get_image_plane()
            .unwrap_or(ImagePlane { z: -1, c: -1, t: -1 });

        // Resolve each channel's pixel slice and sampling geometry once. The previous
        // per-pixel path re-matched the container type and recomputed zoom for every
        // pixel — all loop-invariant per channel. `is_rgb` selects the luminance vs.
        // direct-sample branch below.
        let origin_width = size.width;
        let zoom_x = size.width / full_image_size.width;
        let zoom_y = size.height / full_image_size.height;
        let channel_views: Vec<(i32, bool, &[f32])> = channels
            .iter()
            .filter_map(|(idx, container)| match container.as_ref() {
                ImageContainer::F32Gray(img) => Some((*idx, false, img.as_slice())),
                ImageContainer::F32Rgb(img) => Some((*idx, true, img.as_slice())),
                ImageContainer::U32(_) => None, // no intensity contribution
            })
            .collect();
        let n_ch = channel_views.len();
        let n_obj = max_id + 1;

        // Struct-of-arrays accumulators. Pass 1 only ever touches these small, contiguous
        // arrays (indexed by instance id) instead of the large `Roi` struct and its
        // per-ROI intensity IndexMap — far better cache behaviour on the hot per-pixel
        // path. The full `Roi`s are assembled once, in pass 2.
        let mut area = vec![0usize; n_obj];
        let mut bbox = vec![[u32::MAX, u32::MAX, 0u32, 0u32]; n_obj];
        let mut sum_x = vec![0u64; n_obj];
        let mut sum_y = vec![0u64; n_obj];
        let mut sum_x2 = vec![0u64; n_obj];
        let mut sum_y2 = vec![0u64; n_obj];
        let mut sum_xy = vec![0u64; n_obj];
        let mut seg_class = vec![0u32; n_obj];
        let mut touches_edge = vec![false; n_obj];

        // Per-(object, channel) intensity accumulators, indexed `id * n_ch + ci`.
        // Only sum / min / max are kept; mean (avg) is derived as sum / area in pass 2.
        // No per-pixel value storage — that (for median/stddev) was the bottleneck.
        let mut i_sum = vec![0f64; n_obj * n_ch];
        let mut i_min = vec![f32::MAX; n_obj * n_ch];
        let mut i_max = vec![f32::MIN; n_obj * n_ch];

        // --- Pass 1: accumulate per-object area, moments, intensities and bbox ---
        let edge_x = full_image_size.width.saturating_sub(1);
        let edge_y = full_image_size.height.saturating_sub(1);
        for y in 0..h {
            let row = y * w;
            let sample_row = (y * zoom_y) * origin_width;
            for x in 0..w {
                let id = instance_map_slice[row + x] as usize;
                if id == 0 {
                    continue;
                }

                let x_abs = x + tile_offset.x;
                let y_abs = y + tile_offset.y;

                seg_class[id] = segmentation_slice[row + x];

                // Sample intensity for each pre-resolved channel.
                let sample = sample_row + x * zoom_x;
                let base = id * n_ch;
                for (ci, (_, is_rgb, slice)) in channel_views.iter().enumerate() {
                    let val = if *is_rgb {
                        let idx = sample * 3;
                        // Perceptual luminance (BT.709); background_level = 0.
                        (0.2126 * slice[idx] + 0.7152 * slice[idx + 1] + 0.0722 * slice[idx + 2])
                            .max(0.0)
                    } else {
                        slice[sample]
                    };
                    let k = base + ci;
                    i_sum[k] += val as f64;
                    if val < i_min[k] {
                        i_min[k] = val;
                    }
                    if val > i_max[k] {
                        i_max[k] = val;
                    }
                }

                // Moments (for centroid / ellipse) use absolute coordinates.
                area[id] += 1;
                sum_x[id] += x_abs as u64;
                sum_y[id] += y_abs as u64;
                sum_x2[id] += (x_abs * x_abs) as u64;
                sum_y2[id] += (y_abs * y_abs) as u64;
                sum_xy[id] += (x_abs * y_abs) as u64;

                if x_abs == 0 || y_abs == 0 || x_abs == edge_x || y_abs == edge_y {
                    touches_edge[id] = true;
                }

                let b = &mut bbox[id];
                b[0] = b[0].min(x_abs as u32); // x_min
                b[1] = b[1].min(y_abs as u32); // y_min
                b[2] = b[2].max(x_abs as u32); // x_max
                b[3] = b[3].max(y_abs as u32); // y_max
            }
        }

        // --- Pass 2: assemble each ROI (mask, intensities, geometry) ---
        // Kept single-threaded on purpose: the pipeline parallelizes across images (one
        // core per image), so adding threads here would oversubscribe the CPU.
        cache.roi_cache.extend((1..=max_id).filter(|&id| area[id] != 0).map(|id| {
            let bb = bbox[id];
            let rw = (bb[2] - bb[0] + 1) as usize;
            let rh = (bb[3] - bb[1] + 1) as usize;

            // Build the bbox-relative bitmask by re-scanning only the bbox window.
            let mut mask = BitVec::<u64, Lsb0>::repeat(false, rw * rh);
            for ry in 0..rh {
                for rx in 0..rw {
                    // Convert bbox-local coords to tile-local for slice indexing.
                    let tile_x = rx + bb[0] as usize - tile_offset.x;
                    let tile_y = ry + bb[1] as usize - tile_offset.y;
                    if instance_map_slice[tile_y * w + tile_x] == id as u32 {
                        mask.set(ry * rw + rx, true);
                    }
                }
            }

            // Per-channel intensity stats from the SoA accumulators. Mean is sum / area
            // (every pixel contributes one sample per channel); no per-pixel storage.
            let base = id * n_ch;
            let n = area[id] as f64;
            let mut intensities = IndexMap::with_capacity(n_ch);
            for (ci, (ch_idx, _, _)) in channel_views.iter().enumerate() {
                let k = base + ci;
                intensities.insert(
                    *ch_idx,
                    Intensity {
                        sum_intensity: i_sum[k],
                        min_intensity: i_min[k],
                        max_intensity: i_max[k],
                        avg_intensity: (i_sum[k] / n) as f32,
                        pixel_values: Vec::new(),
                    },
                );
            }

            // `Roi::new` finalizes geometry (perimeter/ellipse) from the mask and moments.
            let mut roi = Roi::new(RoiInit {
                id: ObjectId::next(),
                segmentation_class: SegmentationClass(seg_class[id]),
                bbox: bb,
                mask_data: mask,
                area: area[id],
                touches_edge: touches_edge[id],
                sum_x: sum_x[id],
                sum_y: sum_y[id],
                sum_x2: sum_x2[id],
                sum_y2: sum_y2[id],
                sum_xy: sum_xy[id],
                intensities,
                plane,
                ..Default::default()
            });
            // Assign the segmentation class as default first object class so classify ROI is not mandatory.
            roi.add_object_class(ObjectClass::from_segmentation_class(roi.segmentation_class));

            (roi.id.clone(), roi)
        }));

        Ok(())
    }

    fn name(&self) -> &'static str {
        "ExtractRois"
    }
}

impl Roi {
    pub fn from_mask(
        full_image_size: &ImageSize,
        mask: BitVec<u64, Lsb0>,
        bbox: [u32; 4],
        origin_image: &crate::ImageContainer,
        images: &[(i32, Arc<crate::ImageContainer>)],
        object_class: ObjectClass,
    ) -> Self {
        let mut roi = Roi::new(RoiInit {
            id: ObjectId::next(),
            segmentation_class: SegmentationClass::MANUAL_ANNOTATED,
            intensities: IndexMap::new(),
            bbox: bbox,
            mask_data: mask,
            ..Default::default()
        });

        roi.add_object_class(object_class);
        roi.plane = match origin_image.plane() {
            Some(plane) => plane,
            None => ImagePlane {
                z: -1,
                c: -1,
                t: -1,
            },
        };

        let x1 = bbox[0] as usize;
        let y1 = bbox[1] as usize;
        let x2 = bbox[2] as usize;
        let y2 = bbox[3] as usize;

        for y in y1..=y2 {
            for x in x1..=x2 {
                roi.update_roi_metrics(full_image_size, x, y, origin_image, images);
            }
        }
        roi.finalize_intensity_statistics();
        // Re-finalize geometry now that the mask, area and moments are fully
        // accumulated — the eager finalize in Roi::new only saw the empty skeleton.
        roi.finalize_geometry();

        roi
    }

    /// Updates ROI metrics (sum / min / max intensity, moments, edge flag) for one pixel.
    ///
    /// This method accumulates intensity data from each channel for every pixel in the ROI.
    /// For grayscale images, the raw intensity is used.
    /// For RGB images, the perceptual luminance (BT.709) is calculated with optional background correction.
    /// The per-channel average is derived later in [`finalize_intensity_statistics`].
    ///
    /// # Arguments
    /// * `x` - Absolute X coordinate in the full image
    /// * `y` - Absolute Y coordinate in the full image
    /// * `origin_image` - The original image container with tile and zoom information
    /// * `images` - Array of image containers indexed by channel ID
    pub fn update_roi_metrics(
        &mut self,
        full_image_size: &ImageSize,
        x: usize,
        y: usize,
        origin_image: &crate::ImageContainer,
        images: &[(i32, Arc<crate::ImageContainer>)],
    ) {
        let zoom_x = origin_image.size().width / full_image_size.width;
        let zoom_y = origin_image.size().height / full_image_size.height;

        let x_rel = (x - origin_image.tile_offset().x) * zoom_x;
        let y_rel = (y - origin_image.tile_offset().y) * zoom_y;

        // Measure intensity for each channel
        for (index, image) in images {
            match image.as_ref() {
                crate::ImageContainer::F32Gray(image) => {
                    let intensity_slice = image.as_slice();
                    let val = intensity_slice[y_rel * origin_image.size().width + x_rel];
                    let channel_intensity =
                        self.intensities.entry(*index).or_insert_with(new_intensity_acc);
                    channel_intensity.sum_intensity += val as f64;
                    channel_intensity.max_intensity = channel_intensity.max_intensity.max(val);
                    channel_intensity.min_intensity = channel_intensity.min_intensity.min(val);
                }
                crate::ImageContainer::F32Rgb(image) => {
                    let rgb_slice = image.as_slice();
                    // Assuming 3 floats per pixel (RGB)
                    let idx = (y_rel * origin_image.size().width + x_rel) * 3;
                    let r = rgb_slice[idx];
                    let g = rgb_slice[idx + 1];
                    let b = rgb_slice[idx + 2];

                    // Biological Best Practice: Perceptual Luminance (BT.709)
                    // This provides a consistent brightness metric regardless of dye color.
                    let raw_val = 0.2126 * r + 0.7152 * g + 0.0722 * b;

                    // Background Correction: Subtracting background noise (CTCF)
                    // Ensure you have a 'background_level' derived from a non-sample area of the image.
                    let background_level: f32 = 0.0;
                    let corrected_val = (raw_val - background_level).max(0.0);

                    let channel_intensity =
                        self.intensities.entry(*index).or_insert_with(new_intensity_acc);

                    channel_intensity.sum_intensity += corrected_val as f64;
                    channel_intensity.max_intensity =
                        channel_intensity.max_intensity.max(corrected_val);
                    channel_intensity.min_intensity =
                        channel_intensity.min_intensity.min(corrected_val);
                }
                crate::ImageContainer::U32(_image) => {}
            }
        }

        self.area += 1;
        // Update Moments (for Ellipse/Centroid)
        self.sum_x += x as u64;
        self.sum_y += y as u64;
        self.sum_x2 += (x * x) as u64;
        self.sum_y2 += (y * y) as u64;
        self.sum_xy += (x * y) as u64;

        // Edge Detection
        if x == 0 || x == full_image_size.width - 1 || y == 0 || y == full_image_size.height - 1 {
            self.touches_edge = true;
        }
    }

    /// Finalizes per-channel intensity statistics — derives the average from the
    /// accumulated sum and the ROI's pixel `area`. Call once after all pixels have been
    /// processed via [`update_roi_metrics`](Self::update_roi_metrics).
    pub fn finalize_intensity_statistics(&mut self) {
        let n = self.area.max(1) as f64;
        for (_channel_id, intensity) in self.intensities.iter_mut() {
            intensity.avg_intensity = (intensity.sum_intensity / n) as f32;
        }
    }
}

/// A fresh per-channel intensity accumulator: sum/avg start at 0, and min/max start at
/// the opposite sentinels so the first real pixel always replaces them.
fn new_intensity_acc() -> Intensity {
    Intensity {
        min_intensity: f32::MAX,
        max_intensity: f32::MIN,
        ..Intensity::default()
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::{
        F32Gray, ImagePlane,
        image::{ManagedImage, PixelSizes},
    };
    use bitvec::slice::BitSlice;
    use kornia_image::{Image, ImageSize};
    use kornia_tensor::CpuAllocator;
    // Adjust imports based on your internal project structure

    #[test]
    fn test_extract_rois_full_validation() {
        let size = ImageSize {
            width: 10,
            height: 10,
        };
        let mut ctx = PipelineContext::new_test::<F32Gray>(size).unwrap();
        let mut cache = PipelineCache::default();

        // 1. Setup Mock Data
        // Create an Intensity Image (f32)
        let mut intensity = vec![0.0f32; 100];
        // Create a Semantic Label Image (u32)
        let mut labels = vec![0u32; 100];
        // Create an Instance Class Image (u32) - This is what ExtractRois iterates on
        let mut classes = vec![0u32; 100];

        // Define Object 1: A 2x2 square in the middle
        // Positions: (4,4), (4,5), (5,4), (5,5)
        for y in 4..6 {
            for x in 4..6 {
                let idx = y * 10 + x;
                intensity[idx] = 10.0; // Mean should be 10.0
                labels[idx] = 1; // Semantic: Type A
                classes[idx] = 1; // Instance ID: 1
            }
        }

        // Define Object 2: A single pixel touching the edge
        intensity[0] = 5.0;
        labels[0] = 2; // Semantic: Type B
        classes[0] = 2; // Instance ID: 2

        // Load data into context (assuming your API allows this)
        ctx.image = crate::image::ImageContainer::F32Gray(ManagedImage {
            data: Image::<f32, 1, CpuAllocator>::from_size_slice(
                ImageSize {
                    width: 10,
                    height: 10,
                },
                &intensity,
                CpuAllocator,
            )
            .expect("Failed to create test image"),
            tile_offset: Point2d { x: 0, y: 0 },
            plane: Some(ImagePlane { z: 0, c: 0, t: 0 }),
        });

        ctx.segmentation_map = Some(
            Image::<u32, 1, CpuAllocator>::from_size_slice(
                ImageSize {
                    width: 10,
                    height: 10,
                },
                &labels,
                CpuAllocator,
            )
            .expect("Failed to create test image"),
        );

        ctx.instance_map = Some(
            Image::<u32, 1, CpuAllocator>::from_size_slice(
                ImageSize {
                    width: 10,
                    height: 10,
                },
                &classes,
                CpuAllocator,
            )
            .expect("Failed to create test image"),
        );

        cache
            .image_cache
            .add_to_channel_cache(Arc::new(ctx.image.clone()), 0);

        // 2. Execute Algorithm
        let extractor = ExtractRois { max_objects_before_fail: 100_000 };
        extractor
            .execute(&mut ctx, &mut cache)
            .expect("Extraction failed");

        let mut rois: Vec<&Roi> = cache.roi_cache.values().collect();
        // Sort by area (ascending) so the single-pixel edge object is [0] and
        // the 2×2 square is [1]. Sorting by ObjectId is unsafe because UUIDv7
        // is time-based and interleaves with IDs from parallel tests.
        rois.sort_by_key(|r| r.area);
        assert_eq!(rois.len(), 2);

        // 3. Assertions for Object 1 (The 2x2 Square) — larger area, index 1
        let roi1 = rois[1];

        assert_eq!(roi1.area, 4);
        assert_eq!(roi1.bbox, [4, 4, 5, 5]); // [min_x, min_y, max_x, max_y]
        assert_eq!(roi1.intensities.get(&0).unwrap().sum_intensity, 40.0);
        assert_eq!(roi1.intensities.get(&0).unwrap().max_intensity, 10.0);
        assert_eq!(roi1.segmentation_class, SegmentationClass(1));
        assert!(!roi1.touches_edge);

        // Test Centroid
        let (cx, cy) = roi1.get_centroid();
        assert_eq!(cx, 4.5);
        assert_eq!(cy, 4.5);

        // Test Ellipse (A square should have equal major/minor)
        let ellipse = roi1.get_ellipse();
        assert!(ellipse.major > 0.0);
        assert!((ellipse.major - ellipse.minor).abs() < 0.001); // Symmetry check
        assert_eq!(ellipse.eccentricity, 0.0); // Square is circle-like in moments

        // 4. Assertions for Object 2 (The Edge Pixel) - lower ObjectId, index 0
        let roi2 = rois[0];
        assert!(roi2.touches_edge);
        assert_eq!(roi2.area, 1);

        // 5. Test Compressed Mask for Object 1
        // ROI 1 is 2x2. Mask data should have bits set for a 2x2 area.
        // bitset_size = (2*2 + 63) / 64 = 1.
        assert_eq!(roi1.mask_data.len(), 4);
        assert!(roi1.mask_data[0]);
        assert!(roi1.mask_data[1]);
        assert!(roi1.mask_data[2]);
        assert!(roi1.mask_data[3]);
    }

    #[test]
    fn test_extract_rois_with_tile_offset() {
        // Setup: Full image 50x60, Tile 15x20 at offset 10x15
        let full_size = ImageSize {
            width: 50,
            height: 60,
        };
        let tile_size = ImageSize {
            width: 15,
            height: 20,
        };
        let offset = Point2d { x: 10, y: 15 };

        let mut ctx =
            PipelineContext::new_test_with_offset::<F32Gray>(tile_size, full_size, offset).unwrap();
        let mut cache = PipelineCache::default();

        // Define a 1x1 object at the very top-left of this tile (local 0,0)
        // Global coordinates should be (10+0, 15+0) = (10, 15)
        let mut classes = vec![0u32; 15 * 20];
        classes[0] = 3; // Instance ID 3

        let mut intensity = vec![1.0f32; 15 * 20];
        intensity[0] = 50.0; // Value at local (0,0)

        // Mock ImageContainer in Context
        ctx.image = crate::image::ImageContainer::F32Gray(ManagedImage {
            data: Image::<f32, 1, CpuAllocator>::from_size_slice(
                tile_size,
                &intensity,
                CpuAllocator,
            )
            .unwrap(),
            tile_offset: offset,
            plane: Some(ImagePlane { z: 0, c: 0, t: 0 }),
        });

        ctx.instance_map = Some(
            Image::<u32, 1, CpuAllocator>::from_size_slice(tile_size, &classes, CpuAllocator)
                .unwrap(),
        );

        // Semantic labels are required by the logic
        let labels = vec![1u32; 15 * 20];
        ctx.segmentation_map = Some(
            Image::<u32, 1, CpuAllocator>::from_size_slice(tile_size, &labels, CpuAllocator)
                .unwrap(),
        );

        cache
            .image_cache
            .add_to_channel_cache(Arc::new(ctx.image.clone()), 0);

        // Execute
        let extractor = ExtractRois { max_objects_before_fail: 100_000 };
        extractor
            .execute(&mut ctx, &mut cache)
            .expect("Extraction failed");

        // Assertions - only one ROI was created
        assert_eq!(cache.roi_cache.len(), 1);
        let roi = cache.roi_cache.values().next().expect("ROI not found");

        // The bounding box should be in GLOBAL coordinates
        assert_eq!(roi.bbox, [10, 15, 10, 15]);

        // Centroid should be the global coordinate (10.0, 15.0)
        let (cx, cy) = roi.get_centroid();
        assert_eq!(cx, 10.0);
        assert_eq!(cy, 15.0);

        // Intensity check
        assert_eq!(roi.intensities.get(&0).unwrap().sum_intensity, 50.0);
    }

    #[test]
    fn test_roi_from_mask_no_offset() {
        let mask = BitVec::<u64, Lsb0>::repeat(true, 4); // 2x2 block
        let bbox = [0, 0, 1, 1];
        let image_size = ImageSize {
            width: 2,
            height: 2,
        };

        // Mock image: 2x2, all 10.0
        let img_data = vec![10.0f32; 4];
        let container = Arc::new(crate::image::ImageContainer::F32Gray(ManagedImage {
            data: Image::<f32, 1, CpuAllocator>::from_size_slice(
                image_size,
                &img_data,
                CpuAllocator,
            )
            .unwrap(),
            tile_offset: Point2d { x: 0, y: 0 },
            plane: Some(ImagePlane { z: 0, c: 0, t: 0 }),
        }));

        let roi = Roi::from_mask(
            &image_size,
            mask,
            bbox,
            &container,
            &[(0, container.clone())],
            ObjectClass::Unset,
        );

        assert_eq!(roi.area, 4);
        assert_eq!(roi.intensities.get(&0).unwrap().sum_intensity, 40.0);
    }

    #[test]
    fn test_roi_from_mask_with_offset() {
        // Mask is 2x2, but positioned at global (10, 10)
        let mask = BitVec::<u64, Lsb0>::repeat(true, 4);
        let bbox = [10, 10, 11, 11];
        let full_size = ImageSize {
            width: 20,
            height: 20,
        };
        let tile_size = ImageSize {
            width: 2,
            height: 2,
        };
        let offset = Point2d { x: 10, y: 10 };

        let img_data = vec![5.0f32; 4];
        let container = Arc::new(crate::image::ImageContainer::F32Gray(ManagedImage {
            data: Image::<f32, 1, CpuAllocator>::from_size_slice(
                tile_size,
                &img_data,
                CpuAllocator,
            )
            .unwrap(),
            tile_offset: offset,
            plane: Some(ImagePlane { z: 0, c: 0, t: 0 }),
        }));

        let roi = Roi::from_mask(
            &full_size,
            mask,
            bbox,
            &container,
            &[(0, container.clone())],
            ObjectClass::Unset,
        );

        // Centroid should be in global space: (10.5, 10.5)
        let (cx, cy) = roi.get_centroid();
        assert_eq!(cx, 10.5);
        assert_eq!(cy, 10.5);
    }

    #[test]
    fn test_roi_from_mask_rgb() {
        let mask = BitVec::<u64, Lsb0>::repeat(true, 1); // 1 pixel
        let bbox = [0, 0, 0, 0];
        let size = ImageSize {
            width: 1,
            height: 1,
        };

        // R=1.0, G=0.0, B=0.0 -> Luminance = 0.2126
        let rgb_data = vec![1.0f32, 0.0f32, 0.0f32];
        let container = Arc::new(crate::image::ImageContainer::F32Rgb(ManagedImage {
            data: Image::<f32, 3, CpuAllocator>::from_size_slice(size, &rgb_data, CpuAllocator)
                .unwrap(),
            tile_offset: Point2d { x: 0, y: 0 },
            plane: Some(ImagePlane { z: 0, c: 0, t: 0 }),
        }));

        let roi = Roi::from_mask(
            &size,
            mask,
            bbox,
            &container,
            &[(1, container.clone())],
            ObjectClass::Unset,
        );

        let intensity = roi.intensities.get(&1).unwrap();
        assert!((intensity.sum_intensity - 0.2126).abs() < 1e-4);
    }

    #[test]
    fn test_name() {
        let extractor = ExtractRois { max_objects_before_fail: 100_000 };
        let name = extractor.name();
        assert_eq!(name, "ExtractRois");
    }
}
