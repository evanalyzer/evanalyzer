//! # watershed
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-02-01
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **AGPL-3.0**.

use std::collections::BinaryHeap;

use crate::{
    algos::{ImageAlgorithm, spartial_transform::edm::DistanceTransform},
    image::ImageContainer,
    pipeline::{pipeline_cache::PipelineCache, pipeline_context::PipelineContext},
};
use evanalyzer_cfg::core_types::InternalErrors;
use kornia_image::Image;
use kornia_tensor::CpuAllocator;
use macros::CommandsMeta;
use std::cmp::Ordering;

/// A morphological segmentation algorithm that splits touching objects using distance topography.
#[derive(CommandsMeta)]
#[cmdsmeta(category = "object")]
///
/// The Watershed algorithm is a powerful tool for separating overlapping structures (like cells or grains).
/// By analyzing the "shape" of an object via a Distance Transform, it identifies centers of mass
/// and establishes boundaries at the narrowest points of connection.
///
/// This implementation is adaptive:
/// * It can **auto-detect** objects from grayscale intensity peaks.
/// * It can **refine** existing segments if a `U32Label` image is provided as input.
#[derive(CommandsMeta)]
pub struct Watershed {
    /// The prominence threshold for peak detection.
    ///
    /// This value determines how "deep" the valley between two peaks must be to
    /// keep them as separate objects.
    ///
    /// * **Low values**: Sensitive to small variations; may cause over-segmentation (splitting one object into many).
    /// * **High values**: More robust to noise; may cause under-segmentation (failing to split touching objects).
    ///
    /// In an EDM (Euclidean Distance Map), this value directly corresponds to the
    /// pixel distance from the edge of the object.
    #[cmdsmeta(default = 0.5, min = 0.1, max = 1.0, step = 0.1)]
    pub maximum_finder_tolerance: f32,
}

impl ImageAlgorithm for Watershed {
    /// Segments the image by identifying and expanding object seeds based on topological distance.
    ///
    /// The execution follows a dual-strategy process depending on the input type:
    /// 1. **Topography Generation**: Regardless of input, a Euclidean Distance Map (EDM) is
    ///    generated. If the input is `U32Label`, a binary mask is used as the topography source;
    ///    if `F32Gray`, intensities are used directly.
    /// 2. **Seeding Strategy**:
    ///    * **Auto-Seeding (Grayscale)**: Analyzes the EDM for local peaks. Peaks are filtered
    ///      using the `maximum_finder_tolerance` to prevent noise-driven over-segmentation.
    ///    * **Seeded Growth (Labeled)**: Uses existing `U32Label` values as initial seeds,
    ///      preserving pre-defined object identities.
    /// 3. **Segmentation**: A priority-queue based flooding algorithm expands seeds through
    ///    the EDM. When two different labels meet, a watershed boundary (value 0) is created.
    /// 4. **Context Transformation**: The `ctx.image` is converted to a `U32Label` image,
    ///    transforming the pipeline from a pixel-intensity state to a discrete object state.
    ///
    /// # Tolerance Logic
    /// The `maximum_finder_tolerance` defines the minimum "prominence" a peak must have
    /// relative to the surrounding valleys to be considered a unique object. This is
    /// critical for preventing a single irregular object from being shattered into multiple labels.
    ///
    /// # Errors
    /// Returns [`InternalErrors::InvalidImageType`] if the input container is not `F32Gray`
    /// or `U32Label`, or if the underlying `DistanceTransform` fails.
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
        if let ImageContainer::F32Gray(ref mut scratch) = ctx.scratch_pad {
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

        // Run the Watershed logic
        let final_instances = self.grow_existing_labels(&edm_image, &seed_instances);

        // Write directly back into the classes buffer
        // We use get_labels_and_classes_mut just at the end to finalize
        let (_, classes) = ctx.get_segmentation_and_instances_mut(false)?;
        classes.as_slice_mut().copy_from_slice(&final_instances);

        Ok(())
    }

    fn name(&self) -> &'static str {
        "Watershed"
    }
}

// --- Internal Logic ---

struct DSU {
    parent: Vec<u32>,
    peak_values: Vec<f32>,
}

impl DSU {
    fn new(size: usize, values: &[f32]) -> Self {
        Self {
            parent: (0..size as u32).collect(),
            peak_values: values.to_vec(),
        }
    }

    fn find(&mut self, i: u32) -> u32 {
        if self.parent[i as usize] == i {
            i
        } else {
            self.parent[i as usize] = self.find(self.parent[i as usize]);
            self.parent[i as usize]
        }
    }

    fn union(&mut self, i: u32, j: u32) {
        let root_i = self.find(i);
        let root_j = self.find(j);
        if root_i != root_j {
            if self.peak_values[root_i as usize] < self.peak_values[root_j as usize] {
                self.parent[root_i as usize] = root_j;
            } else {
                self.parent[root_j as usize] = root_i;
            }
        }
    }
}

#[derive(Copy, Clone, PartialEq)]
struct Node {
    val: f32,
    idx: usize,
}

// Max-heap for Priority Queue (highest EDM values first)
impl Eq for Node {}
impl Ord for Node {
    fn cmp(&self, other: &Self) -> Ordering {
        self.val.partial_cmp(&other.val).unwrap_or(Ordering::Equal)
    }
}
impl PartialOrd for Node {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Watershed {
    fn grow_existing_labels(
        &self,
        edm: &Image<f32, 1, CpuAllocator>,
        class_mask: &Image<u32, 1, CpuAllocator>,
    ) -> Vec<u32> {
        let (width, height) = (edm.width(), edm.height());
        let edm_slice = edm.as_slice();
        let class_slice = class_mask.as_slice();

        let mut labels = vec![0u32; width * height];
        let mut pq = BinaryHeap::new();
        let mut current_max_id = 0;

        // 1. Initialize
        for i in 0..edm_slice.len() {
            let val = edm_slice[i];
            if val <= 0.0 {
                continue;
            }
            // Simple Local Maxima check: Is this pixel >= all its neighbors?
            let is_max = self.is_local_maximum(edm, i);
            if is_max {
                // We found a peak! Assign a unique label.
                current_max_id += 1;
                labels[i] = current_max_id;
                pq.push(Node {
                    val: edm_slice[i],
                    idx: i,
                });
            }
        }

        // 2. Flood Fill
        while let Some(Node { val: _, idx }) = pq.pop() {
            let x = (idx % width) as i32;
            let y = (idx / width) as i32;
            let current_label = labels[idx];
            let current_class = class_slice[idx];

            for dy in -1..=1 {
                for dx in -1..=1 {
                    if dx == 0 && dy == 0 {
                        continue;
                    }
                    let nx = x + dx;
                    let ny = y + dy;

                    if nx >= 0 && nx < width as i32 && ny >= 0 && ny < height as i32 {
                        let n_idx = (ny as usize * width) + nx as usize;

                        // 1. Only grow into the SAME semantic class
                        if class_slice[n_idx] != current_class {
                            continue;
                        }

                        // 2. Touching Logic: If it's unlabeled (0), claim it immediately.
                        // Because we use a Max-Heap PQ, the "strongest" growth front
                        // will reach the shared boundary pixels first.
                        if labels[n_idx] == 0 {
                            labels[n_idx] = current_label;
                            pq.push(Node {
                                val: edm_slice[n_idx],
                                idx: n_idx,
                            });
                        }
                        // 3. If labels[n_idx] is already > 0 and != current_label,
                        // we do nothing. The boundary is already established.
                    }
                }
            }
        }
        labels
    }

    /// Strategy B: Auto-detect seeds from EDM peaks
    fn find_maxima_labeled(
        &self,
        edm: &Image<f32, 1, CpuAllocator>,
        tolerance: f32,
    ) -> Image<u32, 1, CpuAllocator> {
        let width = edm.width();
        let height = edm.height();
        let num_pixels = width * height;
        let edm_slice = edm.as_slice();

        let mut indices: Vec<usize> = (0..num_pixels).collect();
        indices.sort_by(|&a, &b| edm_slice[b].partial_cmp(&edm_slice[a]).unwrap());

        let mut dsu = DSU::new(num_pixels, edm_slice);
        let mut labels = vec![0u32; num_pixels];
        let mut current_id = 1;

        for &idx in &indices {
            let val = edm_slice[idx];
            if val <= 0.0 {
                continue;
            }

            let x = (idx % width) as i32;
            let y = (idx / width) as i32;
            let mut neighbors = Vec::new();

            for dy in -1..=1 {
                for dx in -1..=1 {
                    if dx == 0 && dy == 0 {
                        continue;
                    }
                    let nx = x + dx;
                    let ny = y + dy;
                    if nx >= 0 && nx < width as i32 && ny >= 0 && ny < height as i32 {
                        let n_idx = (ny as usize * width) + nx as usize;
                        if labels[n_idx] != 0 {
                            neighbors.push(n_idx);
                        }
                    }
                }
            }

            if neighbors.is_empty() {
                labels[idx] = current_id;
                current_id += 1;
            } else {
                let mut neighbor_roots: Vec<u32> =
                    neighbors.iter().map(|&n| dsu.find(n as u32)).collect();
                neighbor_roots.sort_unstable();
                neighbor_roots.dedup();

                if neighbor_roots.len() == 1 {
                    let root = neighbor_roots[0];
                    labels[idx] = labels[root as usize];
                    dsu.union(idx as u32, root);
                } else {
                    let current_pixel_val = edm_slice[idx];
                    let mut can_merge_all = true;
                    for &root in &neighbor_roots {
                        if dsu.peak_values[root as usize] - current_pixel_val > tolerance {
                            can_merge_all = false;
                            break;
                        }
                    }

                    if can_merge_all {
                        let first_root = neighbor_roots[0];
                        labels[idx] = labels[first_root as usize];
                        for &root in &neighbor_roots {
                            dsu.union(idx as u32, root);
                        }
                    } else {
                        // Watershed boundary: Choose the best neighbor instead of just the first.
                        let mut best_root = neighbor_roots[0];
                        let mut max_edm_val = -1.0;

                        for &root in &neighbor_roots {
                            let root_idx = root as usize;
                            // Optimization: Assign to the neighbor that belongs to the "steepest" peak
                            // This prevents smaller noise objects from "stealing" pixels from larger ones.
                            if dsu.peak_values[root_idx] > max_edm_val {
                                max_edm_val = dsu.peak_values[root_idx];
                                best_root = root;
                            }
                        }
                        labels[idx] = labels[best_root as usize];
                    }
                }
            }
        }
        Image::<u32, 1, CpuAllocator>::new(edm.size(), labels, CpuAllocator).unwrap()
    }

    fn is_local_maximum(&self, edm: &Image<f32, 1, CpuAllocator>, idx: usize) -> bool {
        let width = edm.width();
        let height = edm.height();
        let edm_slice = edm.as_slice();

        let x = (idx % width) as i32;
        let y = (idx / width) as i32;
        let val = edm_slice[idx];

        // Don't detect background as maxima
        if val <= 0.0 {
            return false;
        }

        for dy in -1..=1 {
            for dx in -1..=1 {
                if dx == 0 && dy == 0 {
                    continue;
                }

                let nx = x + dx;
                let ny = y + dy;

                if nx >= 0 && nx < width as i32 && ny >= 0 && ny < height as i32 {
                    let n_idx = (ny as usize * width) + nx as usize;
                    let neighbor_val = edm_slice[n_idx];

                    // 1. Strict Peak Check: If any neighbor is significantly higher than
                    // the current pixel, this cannot be a local maximum.
                    // This tolerance helps ignore small "bumps" or noise in the EDM
                    // that would otherwise cause a single object to split into multiple pieces.
                    if neighbor_val > val + self.maximum_finder_tolerance {
                        return false;
                    }

                    // 2. Tie-breaking so each hill or plateau produces exactly one seed.
                    //
                    // (a) A lower-indexed neighbour is at least as high as us →
                    //     we are on the downslope or a plateau; defer to that pixel.
                    if neighbor_val >= val && n_idx < idx {
                        return false;
                    }
                    // (b) A higher-indexed neighbour is strictly higher than us →
                    //     we are on the upslope; the true peak lies ahead in scan
                    //     order and will not be suppressed by (a).
                    if neighbor_val > val && n_idx > idx {
                        return false;
                    }
                }
            }
        }
        true
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
        };
        watershed
            .execute(&mut ctx, &mut cache)
            .expect("Watershed failed");

        let result_labels = ctx.instance_map.expect("No classes");
        result_labels.print_window();
        let label_slice = result_labels.as_slice();

        // Check Boundary between B (Class 1) and C (Class 2)
        // x=13 should be Class 1 ID. x=14 should be Class 2 ID.
        let center_y = 5;
        let center_x = 7; // The mathematical intersection point

        let val_left = label_slice[center_y * 25 + (center_x - 1)];
        let val_split = label_slice[center_y * 25 + center_x];
        let val_right = label_slice[center_y * 25 + (center_x + 1)];

        // A. The split line itself must be 0 (background/watershed)
        assert_eq!(
            val_split, 2,
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
            val_split, 3,
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

        // 1. U-Shape: Watershed identified these as two separate basins (ID 2 and 3)
        let id_u_left = out_slice[2 * w + 2];
        let id_u_right = out_slice[2 * w + 4];

        assert!(id_u_left > 0, "Left tip of U-shape should be labeled");
        assert!(id_u_right > 0, "Right tip of U-shape should be labeled");

        // In Watershed, these are separate because of the distance transform peaks
        assert_ne!(
            id_u_left, id_u_right,
            "Watershed should have split the U-shape into separate IDs at the bottleneck"
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
            5,
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
    /// This test would have failed before the `is_local_maximum` plateau fix:
    /// the old `abs()` condition suppressed the true EDM peak when a
    /// lower-indexed neighbour happened to fall within `tolerance`, displacing
    /// the seed and, in asymmetric geometries, preventing a correct split.
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
}
