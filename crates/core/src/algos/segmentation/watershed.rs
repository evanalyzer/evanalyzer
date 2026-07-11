//! # watershed
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-02-01
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **AGPL-3.0**.

use crate::{
    algos::{
        ImageAlgorithm, segmentation::maximum_finder::watershed_segment_edm,
        spartial_transform::edm::DistanceTransform,
    },
    image::ImageContainer,
    pipeline::{pipeline_cache::PipelineCache, pipeline_context::PipelineContext},
};
use evanalyzer_cfg::core_types::InternalErrors;
use kornia_image::Image;
use kornia_imgproc::filter::gaussian_blur;
use kornia_tensor::CpuAllocator;
use macros::CommandsMeta;
use std::sync::Arc;

/// A morphological segmentation algorithm that splits touching objects using distance topography.
#[derive(CommandsMeta)]
#[cmdsmeta(category = "object", next = "measure")]
///
/// This is a faithful port of ImageJ's `Process > Binary > Watershed`
/// (`MaximumFinder` applied to the Euclidean distance map). Touching objects that
/// `ConnectedComponents` merged into a single blob are split at their "necks":
/// the distance map's local maxima are the seeds, maxima protruding less than
/// `maximum_finder_tolerance` above the ridge connecting them to a higher maximum
/// are merged, and a constrained flood draws 1-pixel watershed lines between the
/// surviving basins. The split blob is then re-labeled into separate instances.
#[derive(CommandsMeta)]
pub struct Watershed {
    /// Prominence tolerance for the maximum finder, in pixels of distance.
    ///
    /// A local maximum of the distance map is treated as a separate object only
    /// if it protrudes more than this value above the ridge connecting it to a
    /// higher maximum. This is ImageJ's "prominence"/"noise tolerance" parameter.
    ///
    /// * **Low values**: more sensitive; may over-segment ragged objects.
    /// * **High values**: more robust; may fail to split genuinely touching objects.
    ///
    /// ImageJ's default of `0.5` works well for most distance maps; raise it if a
    /// single object is being split into several pieces.
    #[cmdsmeta(default = 0.5, min = 0.1, max = 20.0, step = 0.5)]
    pub maximum_finder_tolerance: f32,

    /// Standard deviation (px) of an optional Gaussian blur applied to the
    /// distance map *before* the maximum finder. `0` disables it.
    ///
    /// ImageJ's `trueEdmHeight` correction already handles ordinary ragged mask
    /// boundaries, so this is rarely needed; for extremely noisy AI masks a value
    /// of `1.0`–`2.0` can further suppress spurious maxima.
    #[cmdsmeta(default = 0.0, min = 0.0, max = 10.0, step = 0.5)]
    pub smoothing_sigma: f32,

    /// Minimum object size, in pixels. After segmentation, any object smaller than
    /// this is removed (its pixels become background). `0` disables the filter.
    ///
    /// Use it to drop tiny fragments left by very ragged masks.
    #[cmdsmeta(default = 0, min = 0, max = 100000, step = 1)]
    pub min_object_size: i32,
}

impl ImageAlgorithm for Watershed {
    /// Splits touching objects via ImageJ's EDM watershed.
    ///
    /// 1. **Topography**: a Euclidean Distance Map (EDM) is computed from the
    ///    binary footprint of the current instance map (any instance > 0 is foreground).
    /// 2. **Watershed**: [`watershed_segment_edm`] runs ImageJ's `MaximumFinder`
    ///    on the EDM, producing a mask in which particles are `255` and 1-pixel
    ///    watershed lines (plus background) are `0`.
    /// 3. **Re-labeling**: the cut foreground is re-labeled into separate
    ///    instances with 8-connectivity, constrained so two pixels are only joined
    ///    if they shared the same original instance (already-separated objects are
    ///    never merged).
    ///
    /// # Errors
    /// Returns an error if the instance map is missing or the underlying
    /// `DistanceTransform` fails.
    fn execute(
        &self,
        ctx: &mut PipelineContext,
        cache: &mut PipelineCache,
    ) -> Result<(), InternalErrors> {
        if self.maximum_finder_tolerance <= 0.0 {
            return Ok(());
        }

        // Get references to labels. We clone seed_labels because DistanceTransform
        // will overwrite the context's 'image' slot.
        let seed_instances = ctx.get_instance_map()?.clone();

        // Reuse the scratch_pad to create the F32 input for DistanceTransform
        // This avoids the 'f32_data' Vec allocation.
        ctx.prepare_f32_gray_scratch()?;
        if let ImageContainer::F32Gray(scratch) = Arc::make_mut(&mut ctx.scratch_pad) {
            let scratch_slice = scratch.as_slice_mut();
            let label_slice = seed_instances.as_slice();

            // Manual loop is extremely fast and zero-allocation
            for (f, &l) in scratch_slice.iter_mut().zip(label_slice.iter()) {
                *f = if l > 0 { 1.0 } else { 0.0 };
            }
        }

        // Swap scratch into main image so DistanceTransform can see it
        // This effectively "moves" our prepared mask into the input slot
        ctx.swap()?;

        // Run the transform
        let transform = DistanceTransform {
            threshold: 0.0,
            edges_are_background: true,
        };
        transform.execute(ctx, cache)?;

        // Get the EDM result (which is now in ctx.image after transform.execute)
        let edm_image = ctx.get_f32_gray_image()?;
        let width = edm_image.width();
        let height = edm_image.height();

        // Optionally smooth the distance map to further suppress spurious maxima
        // from very ragged masks (ImageJ's trueEdmHeight already handles ordinary
        // roughness, so this is usually unnecessary).
        let smoothed;
        let edm: &[f32] = if self.smoothing_sigma > 0.0 {
            smoothed = Self::smooth_edm(&edm_image, self.smoothing_sigma)?;
            smoothed.as_slice()
        } else {
            edm_image.as_slice()
        };

        // ImageJ MaximumFinder watershed: particles = 255, watershed lines = 0.
        let mask = watershed_segment_edm(edm, width, height, self.maximum_finder_tolerance);

        // Re-label the cut foreground into separate instances.
        let final_instances = self.label_cut_foreground(
            &mask,
            seed_instances.as_slice(),
            width,
            height,
            self.min_object_size.max(0) as usize,
        );

        let (_, classes) = ctx.get_segmentation_and_instances_mut(false)?;
        classes.as_slice_mut().copy_from_slice(&final_instances);

        Ok(())
    }

    fn name(&self) -> &'static str {
        "Watershed"
    }
}

// --- Internal Logic ---

impl Watershed {
    /// Gaussian-blurs the distance map into a fresh buffer. The kernel size is
    /// derived from `sigma` (`2·round(3σ)+1`, forced odd, min 3).
    fn smooth_edm(
        edm: &Image<f32, 1, CpuAllocator>,
        sigma: f32,
    ) -> Result<Image<f32, 1, CpuAllocator>, InternalErrors> {
        let radius = (3.0 * sigma).round().max(1.0) as usize;
        let ksize = 2 * radius + 1;
        let size = edm.size();
        let mut out = Image::<f32, 1, CpuAllocator>::new(
            size,
            vec![0.0; size.width * size.height],
            CpuAllocator,
        )
        .map_err(InternalErrors::from_kornia)?;
        gaussian_blur(edm, &mut out, (ksize, ksize), (sigma, sigma))
            .map_err(InternalErrors::from_kornia)?;
        Ok(out)
    }

    /// Labels the watershed-cut foreground (`mask == 255`) into separate instances
    /// using 8-connectivity. Two pixels are only joined if they belonged to the
    /// same original instance, so distinct upstream objects are never merged.
    /// Objects smaller than `min_object_size` (when > 1) are removed and the
    /// remaining ids are compacted to a contiguous `1..=n` range.
    fn label_cut_foreground(
        &self,
        mask: &[u8],
        seed: &[u32],
        width: usize,
        height: usize,
        min_object_size: usize,
    ) -> Vec<u32> {
        let n = width * height;
        let mut labels = vec![0u32; n];
        let mut next_id = 1u32;
        let mut stack: Vec<usize> = Vec::new();

        for start in 0..n {
            if mask[start] != 255 || labels[start] != 0 || seed[start] == 0 {
                continue;
            }
            let instance = seed[start];
            labels[start] = next_id;
            stack.push(start);
            while let Some(p) = stack.pop() {
                let x = (p % width) as i32;
                let y = (p / width) as i32;
                for dy in -1..=1 {
                    for dx in -1..=1 {
                        if dx == 0 && dy == 0 {
                            continue;
                        }
                        let nx = x + dx;
                        let ny = y + dy;
                        if nx < 0 || nx >= width as i32 || ny < 0 || ny >= height as i32 {
                            continue;
                        }
                        let q = ny as usize * width + nx as usize;
                        if mask[q] == 255 && labels[q] == 0 && seed[q] == instance {
                            labels[q] = next_id;
                            stack.push(q);
                        }
                    }
                }
            }
            next_id += 1;
        }

        if min_object_size > 1 {
            let mut sizes = vec![0usize; next_id as usize];
            for &l in &labels {
                sizes[l as usize] += 1;
            }
            let mut remap = vec![0u32; next_id as usize];
            let mut compact = 1u32;
            for (id, remap_id) in remap.iter_mut().enumerate().skip(1) {
                if sizes[id] >= min_object_size {
                    *remap_id = compact;
                    compact += 1;
                }
            }
            for l in labels.iter_mut() {
                *l = remap[*l as usize];
            }
        }

        labels
    }
}
#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use super::*;
    use crate::{F32Gray, image::ImageDebugExt};
    use kornia_image::ImageSize;

    #[test]
    fn test_watershed_multi_class_boundaries() {
        let size = ImageSize {
            width: 25,
            height: 10,
        };
        let mut class_data = vec![0u32; 25 * 10];

        for y in 0..10 {
            for x in 0..25 {
                let dist_a = (x as i32 - 4).abs() + (y as i32 - 5).abs();
                let dist_b = (x as i32 - 10).abs() + (y as i32 - 5).abs();
                let dist_c = (x as i32 - 17).abs() + (y as i32 - 5).abs();

                if dist_a <= 3 || dist_b <= 3 {
                    class_data[y * 25 + x] = 1;
                } else if dist_c <= 3 {
                    class_data[y * 25 + x] = 2;
                }
            }
        }

        let mut ctx = PipelineContext::new_test::<F32Gray>(ImageSize {
            width: 25,
            height: 10,
        })
        .unwrap();

        ctx.instance_map = Some(
            Image::<u32, 1, CpuAllocator>::new(size, class_data.clone(), CpuAllocator).unwrap(),
        );

        ctx.instance_map
            .as_ref()
            .expect("No classes")
            .print_window();

        let mut cache = PipelineCache::default();

        let watershed = Watershed {
            maximum_finder_tolerance: 0.1,
            smoothing_sigma: 0.0,
            min_object_size: 0,
        };
        watershed
            .execute(&mut ctx, &mut cache)
            .expect("Watershed failed");

        let result_labels = ctx.instance_map.expect("No classes");
        result_labels.print_window();
        let label_slice = result_labels.as_slice();

        // Three diamonds: A=(4,5) and B=(10,5) (class 1, touching), C=(17,5) (class 2).
        // Watershed must split A|B; the constrained re-labeling keeps C separate.
        // So the three centres must carry three distinct, non-zero instance ids.
        let a = label_slice[5 * 25 + 4];
        let b = label_slice[5 * 25 + 10];
        let c = label_slice[5 * 25 + 17];

        assert!(a > 0, "diamond A centre should be labeled");
        assert!(b > 0, "diamond B centre should be labeled");
        assert!(c > 0, "diamond C centre should be labeled");
        assert_ne!(a, b, "touching same-class diamonds A and B must be split");
        assert_ne!(a, c, "A and C must be distinct instances");
        assert_ne!(b, c, "B and C must be distinct instances");

        let unique: HashSet<u32> = label_slice.iter().copied().filter(|&v| v > 0).collect();
        assert_eq!(
            unique.len(),
            3,
            "expected exactly 3 instances, found {}",
            unique.len()
        );
    }

    #[test]
    fn test_waterseh_complex_topology() {
        let size = ImageSize {
            width: 12,
            height: 12,
        };
        let mut data = vec![0u32; (size.width * size.height) as usize];
        let w = size.width as usize;

        // --- 1. U-Shape (Equivalence / Merger Test) ---
        // Two vertical legs connected by a horizontal bar at the bottom
        for y in 2..5 {
            data[y * w + 2] = 1; // Left leg
            data[y * w + 4] = 1; // Right leg
        }
        data[5 * w + 2] = 1;
        data[5 * w + 3] = 1;
        data[5 * w + 4] = 1; // Bottom connection

        // --- 2. Diagonal (8-Connectivity Test) ---
        // These two pixels only touch at a corner (7,7) and (8,8)
        data[7 * w + 7] = 1;
        data[8 * w + 8] = 1;

        // --- 3. Nested Object (The Donut / Hole Test) ---
        // Outer Square (From x=7,y=1 to x=11,y=5)
        for i in 7..12 {
            data[1 * w + i] = 1; // Top
            data[5 * w + i] = 1; // Bottom
        }
        for i in 1..6 {
            data[i * w + 7] = 1; // Left
            data[i * w + 11] = 1; // Right
        }
        // Inner "Core" at (9,3)
        // Separated by at least one pixel of '0' from all sides (including diagonals)
        data[3 * w + 9] = 1;

        // Setup Context
        let mut ctx = PipelineContext::new_test::<F32Gray>(size).unwrap();
        ctx.instance_map =
            Some(Image::<u32, 1, CpuAllocator>::new(size, data, CpuAllocator).unwrap());

        println!("--- Input Mask ---");
        ctx.get_instance_map().unwrap().print_window();

        let mut cache = PipelineCache::default();
        let watershed = Watershed {
            maximum_finder_tolerance: 0.8,
            smoothing_sigma: 0.0,
            min_object_size: 0,
        };

        // Execute CCL
        watershed
            .execute(&mut ctx, &mut cache)
            .expect("CCL Execution failed");

        // Get Results
        let output = ctx.get_instance_map().unwrap();
        println!("--- Labeled Output ---");
        output.print_window();
        let out_slice = output.as_slice();

        // --- ASSERTIONS ---

        // 1. U-Shape: a single-pixel-wide skeleton has a perfectly flat EDM
        // (every foreground pixel is exactly 1.0 px from the background, with
        // zero prominence anywhere) — there is no real peak to split on, so
        // both legs must stay one object.
        let id_u_left = out_slice[2 * w + 2];
        let id_u_right = out_slice[2 * w + 4];

        assert!(id_u_left > 0, "Left tip of U-shape should be labeled");
        assert!(id_u_right > 0, "Right tip of U-shape should be labeled");

        assert_eq!(
            id_u_left, id_u_right,
            "a flat-EDM skeleton has no prominence to split on; both legs should stay one object"
        );

        // 2. 8-Connectivity: (7,7) and (8,8) must have the same ID
        let id_diag_1 = out_slice[7 * w + 7];
        let id_diag_2 = out_slice[8 * w + 8];
        assert!(id_diag_1 > 0);
        assert_eq!(
            id_diag_1, id_diag_2,
            "8-connectivity failed to merge diagonal pixels"
        );

        // 3. Nested Donut: Outer Ring (7,1) and Inner Core (9,3) must be DIFFERENT
        let id_ring = out_slice[1 * w + 7];
        let id_core = out_slice[3 * w + 9];
        let moat_pixel = out_slice[2 * w + 8]; // This should be 0

        assert_eq!(moat_pixel, 0, "Moat background was corrupted");
        assert!(id_ring > 0 && id_core > 0);
        assert_ne!(
            id_ring, id_core,
            "Nested core incorrectly merged with outer ring"
        );

        // 4. Global Discrepancy: All 3 main structures must be different from each other
        assert_ne!(id_u_left, id_diag_1, "U-shape merged with Diagonal");
        assert_ne!(id_u_left, id_ring, "U-shape merged with Donut");
        assert_ne!(id_diag_1, id_ring, "Diagonal merged with Donut");

        // 5. Count Objects: Should be exactly 4 unique IDs (U, Diag, Ring, Core)
        use std::collections::HashSet;
        let unique_ids: HashSet<_> = out_slice.iter().filter(|&&x| x > 0).collect();
        assert_eq!(
            unique_ids.len(),
            4,
            "Found {} objects, expected 4",
            unique_ids.len()
        );
    }

    /* GRAYSCALE test
    #[test]
    fn test_watershed_splitting_overlapping_shapes() {
        let size = ImageSize {
            width: 25,
            height: 10,
        };
        // Initialize background as 0.0
        let mut data = vec![0f32; 25 * 10];

        // Create two overlapping "diamonds"
        // Diamond 1 Center: (7, 7)
        // Diamond 2 Center: (17, 7)
        // They meet exactly at x=12.
        // for y in 0..15 {
        //     for x in 0..30 {
        //         let dist1 = (x as i32 - 7).abs() + (y as i32 - 7).abs();
        //         let dist2 = (x as i32 - 17).abs() + (y as i32 - 7).abs();
        //         // Union of the two shapes
        //         if dist1 <= 5 || dist2 <= 5 {
        //             data[y * 30 + x] = 1.0;
        //         }
        //     }
        // }

        for y in 0..10 {
            for x in 0..25 {
                let dist_a = (x as i32 - 4).abs() + (y as i32 - 5).abs();
                let dist_b = (x as i32 - 10).abs() + (y as i32 - 5).abs();
                let dist_c = (x as i32 - 17).abs() + (y as i32 - 5).abs();

                if dist_a <= 3 || dist_b <= 3 {
                    data[y * 25 + x] = 1.0;
                } else if dist_c <= 3 {
                    data[y * 25 + x] = 2.0;
                }
            }
        }

        let input_img = ImageContainer::F32Gray(
            Image::<f32, 1, CpuAllocator>::new(size, data, CpuAllocator).unwrap(),
        );

        let mut ctx = PipelineContext::new_from_image(input_img).unwrap();
        let mut cache = PipelineCache::default();

        ctx.get_f32_gray_image().unwrap().print_window();
        let watershed = Watershed {
            maximum_finder_tolerance: 0.1, // Small tolerance to force splitting deep valleys
        };

        // 1. Execute
        watershed
            .execute(&mut ctx, &mut cache)
            .expect("Watershed execution failed");

        // 2. Verify Result is U32Label
        let label_slice = ctx.labels.as_slice();

        ctx.labels.print_window();

        // 3. Verify object count
        let max_label = label_slice.iter().max().unwrap_or(&0);
        assert!(
            *max_label >= 2,
            "Watershed should have found at least 2 distinct objects, found {}",
            max_label
        );

        // 4. Verify the split line
        // The shapes overlap at x=12. We expect a line of 0s there.
        // We also want to verify that x=11 and x=13 are NOT 0 (they are the objects).
        let center_y = 5;
        let center_x = 7; // The mathematical intersection point

        let val_left = label_slice[center_y * 25 + (center_x - 1)];
        let val_split = label_slice[center_y * 25 + center_x];
        let val_right = label_slice[center_y * 25 + (center_x + 1)];

        // A. The split line itself must be 0 (background/watershed)
        assert_eq!(
            val_split, 1,
            "Pixel at separation point (5, 7) should be 1 (first element)"
        );

        // B. The pixels to the left and right must be valid objects
        assert_eq!(
            val_left, 1,
            "Pixel to the left of split should be an object"
        );
        assert_eq!(
            val_right, 2,
            "Pixel to the right of split should the second object"
        );

        // C. The split must separate different instances
        assert_ne!(
            val_left, val_right,
            "The split line did not separate two different labels!"
        );

        let center_y = 5;
        let center_x = 14; // The mathematical intersection point

        let val_left = label_slice[center_y * 25 + (center_x - 1)];
        let val_split = label_slice[center_y * 25 + center_x];
        let val_right = label_slice[center_y * 25 + (center_x + 1)];

        // A. The split line itself must be 0 (background/watershed)
        assert_eq!(
            val_split, 2,
            "Pixel at separation point (5, 14) should be 1 (first element)"
        );

        // B. The pixels to the left and right must be valid objects
        assert_eq!(
            val_left, 2,
            "Pixel to the left of split should be an object"
        );
        assert_eq!(
            val_right, 3,
            "Pixel to the right of split should the third object"
        );

        // C. The split must separate different instances
        assert_ne!(
            val_left, val_right,
            "The split line did not separate two different labels!"
        );

    }*/

    /// Verifies ImageJ `Process > Binary > Watershed` behaviour on the canonical
    /// two-touching-discs test case.
    ///
    /// Two discs (r=4, centres 6 px apart) are merged into one instance label
    /// before watershed runs - exactly as they would appear after a CCL step
    /// that failed to separate touching blobs.  The algorithm must find the two
    /// EDM peaks independently and flood-fill them into distinct labels.
    ///
    /// This test would have failed under the old PQ-based flood, which assigned
    /// a brand-new seed to every raw local maximum with no prominence check —
    /// a noisy/asymmetric EDM landscape could shatter a single disc into extra
    /// labels, or (with this exact symmetric geometry) just barely survive.
    /// `watershed_labels`'s tolerance-based peak merging makes the result robust.
    #[test]
    fn test_watershed_matches_imagej_two_touching_discs() {
        let width = 17usize;
        let height = 13usize;
        let size = ImageSize { width, height };

        // Two discs of radius 4 whose centres are 6 px apart.
        // sum-of-radii (8) > distance (6) → they overlap and form one
        // connected blob when labelled as a single instance.
        let c1 = (5i32, 6i32);
        let c2 = (11i32, 6i32);
        let r: i32 = 4;

        let mut data = vec![0u32; width * height];
        for y in 0..height as i32 {
            for x in 0..width as i32 {
                let in1 = (x - c1.0).pow(2) + (y - c1.1).pow(2) <= r * r;
                let in2 = (x - c2.0).pow(2) + (y - c2.1).pow(2) <= r * r;
                if in1 || in2 {
                    data[y as usize * width + x as usize] = 1;
                }
            }
        }

        let mut ctx = PipelineContext::new_test::<F32Gray>(size).unwrap();
        ctx.instance_map =
            Some(Image::<u32, 1, CpuAllocator>::new(size, data, CpuAllocator).unwrap());

        let mut cache = PipelineCache::default();
        let watershed = Watershed {
            maximum_finder_tolerance: 0.5,
            smoothing_sigma: 0.0,
            min_object_size: 0,
        };
        watershed
            .execute(&mut ctx, &mut cache)
            .expect("Watershed failed");

        let result = ctx.instance_map.expect("No instance map after watershed");
        let out = result.as_slice();

        let label_c1 = out[c1.1 as usize * width + c1.0 as usize];
        let label_c2 = out[c2.1 as usize * width + c2.0 as usize];

        assert_ne!(label_c1, 0, "centre of disc 1 is unlabelled");
        assert_ne!(label_c2, 0, "centre of disc 2 is unlabelled");
        assert_ne!(
            label_c1, label_c2,
            "watershed did not split the two discs into separate instances"
        );

        // No phantom objects: exactly two non-zero labels in the output.
        let unique: std::collections::HashSet<u32> =
            out.iter().copied().filter(|&v| v > 0).collect();
        assert_eq!(
            unique.len(),
            2,
            "expected exactly 2 instances after split, found {}",
            unique.len()
        );
    }

    /// Same touching-discs geometry as above, but with a `maximum_finder_tolerance`
    /// large enough to exceed the prominence of the saddle between the two peaks.
    ///
    /// Before the tolerance-based peak merging fix, `Watershed` always created a
    /// brand-new seed for every detected local maximum regardless of `tolerance`
    /// (the parameter only ever loosened a per-pixel neighbor check), so two
    /// touching, similarly-shaped objects — or one irregular/jagged object with
    /// more than one bump in its distance map — were reliably over-segmented no
    /// matter how high `tolerance` was set. This is the bug reported against
    /// real AI-segmented nuclei masks: single nuclei were being sliced in half.
    #[test]
    fn test_watershed_merges_shallow_peaks_within_tolerance() {
        let width = 17usize;
        let height = 13usize;
        let size = ImageSize { width, height };

        let c1 = (5i32, 6i32);
        let c2 = (11i32, 6i32);
        let r: i32 = 4;

        let mut data = vec![0u32; width * height];
        for y in 0..height as i32 {
            for x in 0..width as i32 {
                let in1 = (x - c1.0).pow(2) + (y - c1.1).pow(2) <= r * r;
                let in2 = (x - c2.0).pow(2) + (y - c2.1).pow(2) <= r * r;
                if in1 || in2 {
                    data[y as usize * width + x as usize] = 1;
                }
            }
        }

        let mut ctx = PipelineContext::new_test::<F32Gray>(size).unwrap();
        ctx.instance_map =
            Some(Image::<u32, 1, CpuAllocator>::new(size, data, CpuAllocator).unwrap());

        let mut cache = PipelineCache::default();
        // High enough to swallow the shallow saddle between the two disc peaks
        // (which split correctly at tolerance = 0.5 in the test above).
        let watershed = Watershed {
            maximum_finder_tolerance: 3.0,
            smoothing_sigma: 0.0,
            min_object_size: 0,
        };
        watershed
            .execute(&mut ctx, &mut cache)
            .expect("Watershed failed");

        let result = ctx.instance_map.expect("No instance map after watershed");
        let out = result.as_slice();

        let unique: std::collections::HashSet<u32> =
            out.iter().copied().filter(|&v| v > 0).collect();
        assert_eq!(
            unique.len(),
            1,
            "a high tolerance should merge the two shallow-saddle peaks back into \
             one object, found {} instances",
            unique.len()
        );
    }

    /// A small, low (radius 2) bump touching a much taller (radius 6) dominant
    /// disc — modeling a tiny jagged-boundary noise blob attached to a real
    /// nucleus. The bump's own peak (~2.24) is barely above the saddle (~2.0)
    /// where it meets the big disc, so it should be absorbed regardless of how
    /// much taller the dominant peak is.
    ///
    /// Before fixing the merge criterion to compare against the *weakest*
    /// competing peak, the check required every competing peak (including the
    /// big disc's ~6.08) to be within `tolerance` of the saddle. That can never
    /// hold at any sane tolerance, so the tiny bump survived forever as a
    /// permanent one-object "island" — exactly the bug reported against real
    /// AI-segmented nuclei with jagged mask boundaries.
    #[test]
    fn test_watershed_absorbs_low_prominence_bump_into_dominant_peak() {
        let width = 20usize;
        let height = 20usize;
        let size = ImageSize { width, height };

        let c1 = (6i32, 10i32); // dominant disc
        let r1: i32 = 6;
        let c2 = (13i32, 10i32); // small bump, touching the dominant disc
        let r2: i32 = 2;

        let mut data = vec![0u32; width * height];
        for y in 0..height as i32 {
            for x in 0..width as i32 {
                let in1 = (x - c1.0).pow(2) + (y - c1.1).pow(2) <= r1 * r1;
                let in2 = (x - c2.0).pow(2) + (y - c2.1).pow(2) <= r2 * r2;
                if in1 || in2 {
                    data[y as usize * width + x as usize] = 1;
                }
            }
        }

        let mut ctx = PipelineContext::new_test::<F32Gray>(size).unwrap();
        ctx.instance_map =
            Some(Image::<u32, 1, CpuAllocator>::new(size, data, CpuAllocator).unwrap());

        let mut cache = PipelineCache::default();
        // Comfortably above the bump's own prominence (~0.24) but far below
        // the dominant peak's height above the saddle (~4.08) — a tolerance
        // a user would plausibly pick for real nuclei, not an extreme value.
        let watershed = Watershed {
            maximum_finder_tolerance: 1.0,
            smoothing_sigma: 0.0,
            min_object_size: 0,
        };
        watershed
            .execute(&mut ctx, &mut cache)
            .expect("Watershed failed");

        let result = ctx.instance_map.expect("No instance map after watershed");
        let out = result.as_slice();

        let unique: std::collections::HashSet<u32> =
            out.iter().copied().filter(|&v| v > 0).collect();
        assert_eq!(
            unique.len(),
            1,
            "the low-prominence bump should be absorbed into the dominant peak's \
             basin instead of surviving as its own instance, found {} instances",
            unique.len()
        );
    }

    /// `min_object_size` deterministically dissolves a tiny spurious inner basin
    /// (the "island" artifact) into the larger object surrounding it.
    ///
    /// Geometry: a dominant disc (r=6) touching a small bump (r=2). At a *low*
    /// prominence tolerance the bump survives as a 1-px basin nested inside the
    /// dominant object — exactly the inner-island artifact seen on real nuclei.
    /// With `min_object_size` set above the island size it is merged away,
    /// leaving a single clean object.
    #[test]
    fn test_watershed_min_object_size_dissolves_inner_island() {
        let width = 20usize;
        let height = 20usize;
        let size = ImageSize { width, height };

        let c1 = (6i32, 10i32);
        let r1: i32 = 6;
        let c2 = (13i32, 10i32);
        let r2: i32 = 2;

        let build = || {
            let mut data = vec![0u32; width * height];
            for y in 0..height as i32 {
                for x in 0..width as i32 {
                    let in1 = (x - c1.0).pow(2) + (y - c1.1).pow(2) <= r1 * r1;
                    let in2 = (x - c2.0).pow(2) + (y - c2.1).pow(2) <= r2 * r2;
                    if in1 || in2 {
                        data[y as usize * width + x as usize] = 1;
                    }
                }
            }
            data
        };

        let count_instances = |min_object_size: i32| -> usize {
            let mut ctx = PipelineContext::new_test::<F32Gray>(size).unwrap();
            ctx.instance_map =
                Some(Image::<u32, 1, CpuAllocator>::new(size, build(), CpuAllocator).unwrap());
            let mut cache = PipelineCache::default();
            Watershed {
                // Low enough that the bump's ~0.24px prominence is NOT merged,
                // so it survives as a separate basin without the size filter.
                maximum_finder_tolerance: 0.1,
                smoothing_sigma: 0.0,
                min_object_size,
            }
            .execute(&mut ctx, &mut cache)
            .expect("Watershed failed");
            let result = ctx.instance_map.expect("No instance map after watershed");
            result
                .as_slice()
                .iter()
                .copied()
                .filter(|&v| v > 0)
                .collect::<std::collections::HashSet<u32>>()
                .len()
        };

        assert_eq!(
            count_instances(0),
            2,
            "without the size filter the spurious bump basin should survive as an inner island"
        );
        assert_eq!(
            count_instances(20),
            1,
            "min_object_size should dissolve the inner island into the dominant object"
        );
    }

    /// Smoothing the distance map must not corrupt a clean split: two well-separated
    /// touching discs still resolve to exactly two instances with `smoothing_sigma`
    /// enabled, and the object footprint stays within the original mask (no pixel
    /// outside the mask gets labeled despite the blur bleeding values outward).
    #[test]
    fn test_watershed_smoothing_preserves_split_and_footprint() {
        let width = 17usize;
        let height = 13usize;
        let size = ImageSize { width, height };

        let c1 = (5i32, 6i32);
        let c2 = (11i32, 6i32);
        let r: i32 = 4;

        let mut data = vec![0u32; width * height];
        for y in 0..height as i32 {
            for x in 0..width as i32 {
                let in1 = (x - c1.0).pow(2) + (y - c1.1).pow(2) <= r * r;
                let in2 = (x - c2.0).pow(2) + (y - c2.1).pow(2) <= r * r;
                if in1 || in2 {
                    data[y as usize * width + x as usize] = 1;
                }
            }
        }
        let footprint = data.clone();

        let mut ctx = PipelineContext::new_test::<F32Gray>(size).unwrap();
        ctx.instance_map =
            Some(Image::<u32, 1, CpuAllocator>::new(size, data, CpuAllocator).unwrap());
        let mut cache = PipelineCache::default();
        Watershed {
            maximum_finder_tolerance: 0.5,
            smoothing_sigma: 1.0,
            min_object_size: 0,
        }
        .execute(&mut ctx, &mut cache)
        .expect("Watershed failed");

        let result = ctx.instance_map.expect("No instance map after watershed");
        let out = result.as_slice();

        // No label may fall outside the original mask (the blur bleeds EDM values
        // into the background, but the segmentation must stay within the footprint).
        // Note: ImageJ removes the 1-px watershed line, so not every foreground
        // pixel stays labeled — we only assert the no-leak direction.
        for (i, (&label, &fg)) in out.iter().zip(footprint.iter()).enumerate() {
            if label > 0 {
                assert!(fg > 0, "label leaked outside the mask at pixel {i}");
            }
        }

        let unique: std::collections::HashSet<u32> =
            out.iter().copied().filter(|&v| v > 0).collect();
        assert_eq!(
            unique.len(),
            2,
            "smoothing should still split the two touching discs, found {}",
            unique.len()
        );
    }

    #[test]
    fn test_watershed_matches_original_imagej_logic() {
        // Two overlapping discs (radius 4, centres 6 px apart) merged into one
        // instance label, simulating a fused binary object as ImageJ sees it.
        let width = 17usize;
        let height = 13usize;
        let size = ImageSize { width, height };

        let c1 = (5i32, 6i32);
        let c2 = (11i32, 6i32);
        let r: i32 = 4;

        let mut input_labels = vec![0u32; width * height];
        for y in 0..height as i32 {
            for x in 0..width as i32 {
                let in_disc1 = (x - c1.0).pow(2) + (y - c1.1).pow(2) <= r * r;
                let in_disc2 = (x - c2.0).pow(2) + (y - c2.1).pow(2) <= r * r;
                if in_disc1 || in_disc2 {
                    input_labels[(y as usize) * width + (x as usize)] = 1; // merged as one object
                }
            }
        }

        let mut ctx = PipelineContext::new_test::<F32Gray>(size).unwrap();
        ctx.instance_map =
            Some(Image::<u32, 1, CpuAllocator>::new(size, input_labels, CpuAllocator).unwrap());

        let mut cache = PipelineCache::default();

        let watershed = Watershed {
            maximum_finder_tolerance: 0.5,
            smoothing_sigma: 0.0,
            min_object_size: 0,
        };

        watershed
            .execute(&mut ctx, &mut cache)
            .expect("Watershed crashed");

        let result = ctx.instance_map.expect("No instance map after watershed");
        let output_slice = result.as_slice();

        // Check A: disc centres must not be erased (label 0)
        let idx_c1 = (c1.1 as usize) * width + (c1.0 as usize);
        let idx_c2 = (c2.1 as usize) * width + (c2.0 as usize);

        let label_c1 = output_slice[idx_c1];
        let label_c2 = output_slice[idx_c2];

        assert_ne!(label_c1, 0, "centre of disc 1 was erased (label 0)");
        assert_ne!(label_c2, 0, "centre of disc 2 was erased (label 0)");

        // Check B: the two discs must have been split into different instances
        assert_ne!(
            label_c1, label_c2,
            "the two overlapping discs were not separated (both have id {})",
            label_c1
        );

        // Check C: exactly 2 non-zero labels - no over-segmentation
        let unique_labels: HashSet<u32> = output_slice.iter().copied().filter(|&v| v > 0).collect();

        assert_eq!(
            unique_labels.len(),
            2,
            "expected exactly 2 instances, found {}",
            unique_labels.len()
        );

        // Check D: the midpoint (x=8, y=6) must belong to one of the two instances
        // or be a watershed boundary (0) - not some unexpected third label.
        let mid_x = (c1.0 + c2.0) / 2; // = 8
        let idx_mid = (c1.1 as usize) * width + (mid_x as usize);
        let label_mid = output_slice[idx_mid];

        assert!(
            label_mid == 0 || label_mid == label_c1 || label_mid == label_c2,
            "unexpected label at watershed boundary: {}",
            label_mid
        );
    }

    /// Regression for the reported failure: an elongated "peanut" nucleus (two
    /// fused lobes with a waist) must split into exactly two clean instances at
    /// the default tolerance — with NO spurious tiny inner-island objects, which
    /// is what the previous prominence-based implementation produced.
    #[test]
    fn test_watershed_splits_peanut_without_islands() {
        let width = 24usize;
        let height = 40usize;
        let size = ImageSize { width, height };

        // Two vertically-stacked lobes (r=8) whose centres are 13 px apart, so
        // they overlap into one connected blob with a narrow waist — the shape of
        // the middle object in the report.
        let c1 = (12i32, 13i32);
        let c2 = (12i32, 26i32);
        let r: i32 = 8;

        let mut data = vec![0u32; width * height];
        for y in 0..height as i32 {
            for x in 0..width as i32 {
                let in1 = (x - c1.0).pow(2) + (y - c1.1).pow(2) <= r * r;
                let in2 = (x - c2.0).pow(2) + (y - c2.1).pow(2) <= r * r;
                if in1 || in2 {
                    data[y as usize * width + x as usize] = 1;
                }
            }
        }

        let mut ctx = PipelineContext::new_test::<F32Gray>(size).unwrap();
        ctx.instance_map =
            Some(Image::<u32, 1, CpuAllocator>::new(size, data, CpuAllocator).unwrap());
        let mut cache = PipelineCache::default();

        Watershed {
            maximum_finder_tolerance: 0.5, // ImageJ default
            smoothing_sigma: 0.0,
            min_object_size: 0,
        }
        .execute(&mut ctx, &mut cache)
        .expect("Watershed failed");

        let result = ctx.instance_map.expect("No instance map after watershed");
        let out = result.as_slice();

        let top = out[c1.1 as usize * width + c1.0 as usize];
        let bottom = out[c2.1 as usize * width + c2.0 as usize];
        assert!(top > 0 && bottom > 0, "both lobe centres must be labeled");
        assert_ne!(top, bottom, "the peanut must be split into two lobes");

        // Exactly two instances — no spurious inner islands.
        let unique: HashSet<u32> = out.iter().copied().filter(|&v| v > 0).collect();
        assert_eq!(
            unique.len(),
            2,
            "expected exactly 2 instances with no inner islands, found {}",
            unique.len()
        );
    }
    // temp debug
}
