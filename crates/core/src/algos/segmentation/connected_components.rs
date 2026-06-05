//! # connected_components
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-02-06
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **AGPL-3.0**.

use evanalyzer_cfg::core_types::InternalErrors;
use macros::CommandsMeta;

use crate::{
    algos::ImageAlgorithm,
    pipeline::{pipeline_cache::PipelineCache, pipeline_context::PipelineContext},
};

/// Identifies and labels discrete objects within a binary or multi-class image.
#[derive(CommandsMeta)]
#[cmdsmeta(category = "object")]
pub struct ConnectedComponents;

impl ImageAlgorithm for ConnectedComponents {
    /// Identifies and labels discrete objects within a binary or multi-class image.
    ///
    /// The execution follows a classic two-pass Union-Find approach:
    /// 1. **Initial Labeling & Equivalence Recording**: The algorithm performs a raster scan
    ///    of the image. When a foreground pixel is found, it checks its 8-neighbors (NW, N, NE, W).
    ///    - If no neighbors are labeled, a new unique ID is assigned.
    ///    - If neighbors are labeled, the current pixel takes the smallest neighbor ID,
    ///      and an equivalence is recorded in the Disjoint Set Union (DSU) structure to
    ///      mark these IDs as belonging to the same object.
    /// 2. **Resolution & Compaction**: A second pass resolves all recorded equivalences
    ///    using path compression. This ensures that every pixel in a single connected
    ///    component shares the same final ID.
    /// 3. **ID Re-indexing**: The resulting IDs are re-mapped to a contiguous range
    ///    (1, 2, 3...) to remove gaps caused by merges.
    ///
    /// # Connectivity
    /// This implementation uses **8-connectivity**, meaning pixels touching diagonally
    /// are considered part of the same object. This is critical for preventing
    /// narrow structures from being artificially fragmented.
    ///
    /// # Errors
    /// Returns [`InternalErrors::FormatMismatch`] if the input is not a `U32Label` image.
    /// Thresholding should typically be performed before this command to provide the
    /// necessary class IDs.
    fn execute(
        &self,
        ctx: &mut PipelineContext,
        _cache: &mut PipelineCache,
    ) -> Result<(), InternalErrors> {
        let (labels, classes) = ctx.get_segmentation_and_instances_mut(false)?;

        self.compute_ccl(
            labels.as_slice(),
            classes.as_slice_mut(),
            labels.size().width,
            labels.size().height,
        );

        ctx.swap()?;
        Ok(())
    }

    fn name(&self) -> &'static str {
        "ConnectedComponents"
    }
}
impl ConnectedComponents {
    fn compute_ccl(&self, input: &[u32], output: &mut [u32], width: usize, height: usize) {
        let mut parent = Vec::with_capacity(1024);
        parent.push(0); // Label 0 is background

        let mut next_label = 1;

        for y in 0..height {
            for x in 0..width {
                let idx = y * width + x;
                let class_id = input[idx];
                if class_id == 0 {
                    continue;
                }

                // Collect labels of 8-neighbors (only those already processed: West, NW, N, NE)
                let mut neighbor_labels = [0u32; 4];
                let mut count = 0;

                // West
                if x > 0 && input[idx - 1] == class_id {
                    neighbor_labels[count] = output[idx - 1];
                    count += 1;
                }
                if y > 0 {
                    // North
                    if input[idx - width] == class_id {
                        neighbor_labels[count] = output[idx - width];
                        count += 1;
                    }
                    // North-West
                    if x > 0 && input[idx - width - 1] == class_id {
                        neighbor_labels[count] = output[idx - width - 1];
                        count += 1;
                    }
                    // North-East
                    if x + 1 < width && input[idx - width + 1] == class_id {
                        neighbor_labels[count] = output[idx - width + 1];
                        count += 1;
                    }
                }

                if count == 0 {
                    output[idx] = next_label;
                    parent.push(next_label);
                    next_label += 1;
                } else {
                    // Find the absolute minimum root among neighbors
                    let mut min_l = u32::MAX;
                    for i in 0..count {
                        let root = self.find_root(&mut parent, neighbor_labels[i]);
                        if root < min_l {
                            min_l = root;
                        }
                    }

                    output[idx] = min_l;

                    // Union all neighbor roots to the min_l
                    for i in 0..count {
                        self.union_roots(&mut parent, neighbor_labels[i], min_l);
                    }
                }
            }
        }

        // Final Pass: Flattening and Re-indexing
        let mut lookup = vec![0u32; parent.len()];
        let mut final_id = 1;
        for i in 1..parent.len() {
            let root = self.find_root(&mut parent, i as u32);
            if (i as u32) == root {
                lookup[i] = final_id;
                final_id += 1;
            } else {
                lookup[i] = lookup[root as usize];
            }
        }

        for val in output.iter_mut() {
            if *val > 0 {
                *val = lookup[*val as usize];
            }
        }
    }

    fn find_root(&self, parent: &mut Vec<u32>, mut i: u32) -> u32 {
        while parent[i as usize] != i {
            parent[i as usize] = parent[parent[i as usize] as usize]; // Path compression
            i = parent[i as usize];
        }
        i
    }

    fn union_roots(&self, parent: &mut Vec<u32>, i: u32, j: u32) {
        let root_i = self.find_root(parent, i);
        let root_j = self.find_root(parent, j);
        if root_i != root_j {
            if root_i < root_j {
                parent[root_j as usize] = root_i;
            } else {
                parent[root_i as usize] = root_j;
            }
        }
    }
}

// --- Tests ------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{F32Gray, image::ImageDebugExt};
    use kornia_image::{Image, ImageSize};
    use kornia_tensor::CpuAllocator;

    #[test]
    fn test_ccl_equivalence_and_8_connectivity() {
        let size = ImageSize {
            width: 10,
            height: 10,
        };
        let mut data = vec![0u32; 100];

        // Create a "U-shape" that connects at the bottom
        // If equivalence isn't handled, the two legs will have different IDs
        data[2 * 10 + 2] = 1; // Left leg top
        data[3 * 10 + 2] = 1;
        data[4 * 10 + 2] = 1;

        data[4 * 10 + 3] = 1; // Connection
        data[4 * 10 + 4] = 1;

        data[2 * 10 + 4] = 1; // Right leg top
        data[3 * 10 + 4] = 1;

        // Create a diagonal-only connection (8-connectivity test)
        // (6,6) and (7,7) should be the same object
        data[6 * 10 + 6] = 1;
        data[7 * 10 + 7] = 1;

        // Create a diagonal-only connection (8-connectivity test)
        // (6,6) and (7,7) should be the same object
        data[0 * 10 + 7] = 2;
        data[0 * 10 + 8] = 2;

        let mut ctx = PipelineContext::new_test::<F32Gray>(size).unwrap();
        ctx.segmentation_map =
            Some(Image::<u32, 1, CpuAllocator>::new(size, data, CpuAllocator).unwrap());

        ctx.get_segmentation_map().unwrap().print_window();

        let mut cache = PipelineCache::default();
        let labeling = ConnectedComponents;

        // Execute
        let result = labeling.execute(&mut ctx, &mut cache);
        assert!(result.is_ok());

        // Get results from ctx.image (because of ctx.swap())
        let output = ctx.get_instance_map().unwrap();
        output.print_window();
        let out_slice = output.as_slice();

        // 1. Verify U-Shape merger: (2,2) and (2,4) must have the same ID
        let id_left_leg = out_slice[2 * 10 + 2];
        let id_right_leg = out_slice[2 * 10 + 4];
        assert!(id_left_leg > 0, "Left leg should be labeled");
        assert_eq!(
            id_left_leg, id_right_leg,
            "Equivalence failed: U-shape should be one object"
        );

        // 2. Verify 8-Connectivity: (6,6) and (7,7) must have the same ID
        let id_diag_1 = out_slice[6 * 10 + 6];
        let id_diag_2 = out_slice[7 * 10 + 7];
        assert!(id_diag_1 > 0);
        assert_eq!(
            id_diag_1, id_diag_2,
            "8-connectivity failed: Diagonals should be one object"
        );

        // 3. Verify they are two DIFFERENT objects
        assert_ne!(
            id_left_leg, id_diag_1,
            "Distinct objects were incorrectly merged"
        );

        // 4. Verify contiguous IDs (max label should be 2)
        let max_id = out_slice.iter().max().unwrap();
        assert_eq!(*max_id, 3, "IDs should be contiguous starting from 1");
    }

    #[test]
    fn test_ccl_complex_topology() {
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
        ctx.segmentation_map =
            Some(Image::<u32, 1, CpuAllocator>::new(size, data, CpuAllocator).unwrap());

        println!("--- Input Mask ---");
        ctx.get_segmentation_map().unwrap().print_window();

        let mut cache = PipelineCache::default();
        let labeling = ConnectedComponents;

        // Execute CCL
        labeling
            .execute(&mut ctx, &mut cache)
            .expect("CCL Execution failed");

        // Get Results
        let output = ctx.get_instance_map().unwrap();
        println!("--- Labeled Output ---");
        output.print_window();
        let out_slice = output.as_slice();

        // --- ASSERTIONS ---

        // 1. U-Shape: (2,2) and (2,4) must have the same ID
        let id_u_left = out_slice[2 * w + 2];
        let id_u_right = out_slice[2 * w + 4];
        assert!(id_u_left > 0, "U-shape should be labeled");
        assert_eq!(
            id_u_left, id_u_right,
            "U-shape legs failed to merge equivalence"
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

    #[test]
    fn test_ccl_preserves_distinct_input_labels() {
        let size = ImageSize {
            width: 10,
            height: 10,
        };
        let mut data = vec![0u32; 100];
        let w = size.width as usize;

        // --- 1. Create two touching objects with DIFFERENT labels ---
        // Object A (Label 1) - A 2x2 square
        data[2 * w + 2] = 1;
        data[2 * w + 3] = 1;
        data[3 * w + 2] = 1;
        data[3 * w + 3] = 1;

        // Object B (Label 2) - A 2x2 square touching Object A at the boundary
        // Touching at (2,4) and (3,4)
        data[2 * w + 4] = 2;
        data[2 * w + 5] = 2;
        data[3 * w + 4] = 2;
        data[3 * w + 5] = 2;

        // --- 2. Create two touching objects with the SAME label ---
        // These SHOULD be merged into one Class ID
        data[6 * w + 2] = 3;
        data[6 * w + 3] = 3; // Touching (6,2)

        let mut ctx = PipelineContext::new_test::<F32Gray>(size).unwrap();
        ctx.segmentation_map =
            Some(Image::<u32, 1, CpuAllocator>::new(size, data, CpuAllocator).unwrap());

        println!("--- Input Mask ---");
        ctx.get_segmentation_map().unwrap().print_window();

        let mut cache = PipelineCache::default();
        let labeling = ConnectedComponents;

        // Execute
        labeling.execute(&mut ctx, &mut cache).expect("CCL Failed");

        let output = ctx.get_instance_map().unwrap();
        println!("--- Labeled Output ---");
        output.print_window();

        let out_slice = output.as_slice();

        // --- ASSERTIONS ---

        // 1. Check touching different labels
        let id_obj_a = out_slice[2 * w + 2];
        let id_obj_b = out_slice[2 * w + 4];

        assert!(id_obj_a > 0 && id_obj_b > 0);
        assert_ne!(
            id_obj_a, id_obj_b,
            "CCL merged two different input labels (1 and 2) just because they touched!"
        );

        // 2. Check touching same labels
        let id_obj_c_part1 = out_slice[6 * w + 2];
        let id_obj_c_part2 = out_slice[6 * w + 3];
        assert_eq!(
            id_obj_c_part1, id_obj_c_part2,
            "CCL failed to merge pixels with the same label (3) that are touching"
        );
    }
}
