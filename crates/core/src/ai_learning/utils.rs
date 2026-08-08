use crate::ZProjection;
use crate::ai_learning::model::{Classifier, KnnModel, mlp::predict_mlp};
use crate::image::ImageTile;
use crate::object::Object;
use evanalyzer_cfg::core_types::InternalErrors;
use evanalyzer_cfg::settings::images_settings::ZStackHandling;
use smartcore::linalg::basic::matrix::DenseMatrix;
use std::ops::RangeInclusive;

/// Matches `JobExecutor::analyze_image`'s fixed tile size
/// (`crates/core/src/job/job_executor.rs`) - not reused directly (that
/// constant is private to `JobExecutor`), but kept identical for consistency.
pub const TILE_SIZE: usize = 4096;

// -- Shared training helpers, backend-agnostic over how `rows`/`labels` were
// gathered (pixel samples or object feature vectors) --------------------

pub fn to_dense_matrix(rows: &[Vec<f32>]) -> Result<DenseMatrix<f32>, InternalErrors> {
    let row_refs: Vec<&[f32]> = rows.iter().map(|r| r.as_slice()).collect();
    DenseMatrix::from_2d_array(&row_refs).map_err(|e| InternalErrors::Internal(e.to_string()))
}

/// Shared by every backend's `fit_*` so they all validate identically before
/// touching smartcore/burn. `labels` are dense indices into the owning job's
/// `class_labels` (not raw `SegmentationClass`/`ObjectClass` values) - see
/// `training::pixel`/`training::object` for where that mapping happens, and
/// why it matters (an MLP's output layer is sized off `class_labels.len()`,
/// not the raw label values, which aren't guaranteed contiguous from zero).
pub(crate) fn validate_training_data(
    rows: &[Vec<f32>],
    labels: &[usize],
) -> Result<(), InternalErrors> {
    if rows.len() != labels.len() {
        return Err(InternalErrors::Internal(
            "sample rows and labels must have the same length".to_string(),
        ));
    }
    if rows.is_empty() {
        return Err(InternalErrors::Internal(
            "cannot train on zero samples".to_string(),
        ));
    }
    Ok(())
}

impl Classifier {
    /// Predicts a class index (into the owning `SavedClassifier::settings`'s
    /// nested `class_labels`) for each row of `features`.
    pub fn predict(&self, features: &[Vec<f32>]) -> Result<Vec<usize>, InternalErrors> {
        if features.is_empty() {
            return Ok(Vec::new());
        }
        match self {
            Classifier::RandomForest(model) => {
                let x = to_dense_matrix(features)?;
                model
                    .predict(&x)
                    .map_err(|e| InternalErrors::Internal(e.to_string()))
            }
            Classifier::Knn(model) => {
                let x = to_dense_matrix(features)?;
                match model {
                    KnnModel::Euclidean(model) => model.predict(&x),
                    KnnModel::Manhattan(model) => model.predict(&x),
                    KnnModel::Cosine(model) => model.predict(&x),
                    KnnModel::Hamming(model) => model.predict(&x),
                    KnnModel::Minkowski(model) => model.predict(&x),
                }
                .map_err(|e| InternalErrors::Internal(e.to_string()))
            }
            Classifier::Mlp {
                architecture,
                weights,
            } => predict_mlp(architecture, weights, features),
        }
    }
}

// -- Tiling / z-stack helpers, shared by pixel training (object training
// needs neither - object metrics are already computed, no image I/O) -----

pub fn tile_grid(full_width: usize, full_height: usize, tile_size: usize) -> Vec<ImageTile> {
    let x_steps = full_width.div_ceil(tile_size);
    let y_steps = full_height.div_ceil(tile_size);
    let mut tiles = Vec::with_capacity(x_steps * y_steps);
    for y in 0..y_steps {
        for x in 0..x_steps {
            let offset_x = x * tile_size;
            let offset_y = y * tile_size;
            tiles.push(ImageTile {
                offset_x,
                offset_y,
                width: (full_width - offset_x).min(tile_size),
                height: (full_height - offset_y).min(tile_size),
            });
        }
    }
    tiles
}

pub fn bbox_overlaps_tile(bbox: [u32; 4], tile: &ImageTile) -> bool {
    let [x_min, y_min, x_max, y_max] = bbox;
    let tile_x_min = tile.offset_x as u32;
    let tile_y_min = tile.offset_y as u32;
    let tile_x_max = tile.offset_x as u32 + tile.width as u32 - 1;
    let tile_y_max = tile.offset_y as u32 + tile.height as u32 - 1;
    x_min <= tile_x_max && x_max >= tile_x_min && y_min <= tile_y_max && y_max >= tile_y_min
}

/// (x, y) coordinates (in the image's full-resolution grid, `usize`) of every
/// pixel in `object`'s mask that falls within `tile`'s bounds.
pub fn masked_pixels_in_tile(object: &Object, tile: &ImageTile) -> Vec<(usize, usize)> {
    let [x_min, y_min, x_max, y_max] = object.bbox;
    let tile_x_min = tile.offset_x as u32;
    let tile_y_min = tile.offset_y as u32;
    let tile_x_max = tile.offset_x as u32 + tile.width as u32 - 1;
    let tile_y_max = tile.offset_y as u32 + tile.height as u32 - 1;

    let ix_min = x_min.max(tile_x_min);
    let iy_min = y_min.max(tile_y_min);
    let ix_max = x_max.min(tile_x_max);
    let iy_max = y_max.min(tile_y_max);

    if ix_min > ix_max || iy_min > iy_max {
        return Vec::new();
    }

    let mut samples = Vec::new();
    for y in iy_min..=iy_max {
        for x in ix_min..=ix_max {
            if object.is_part_of(x, y) {
                samples.push((x as usize, y as usize));
            }
        }
    }
    samples
}

/// Mirrors `JobExecutor::prepare_z_stack_iterator`'s handling table
/// (`crates/core/src/job/job_executor.rs`), minus the project-configurable
/// z-range `SingleStack` normally supports there - this job has no per-image
/// `ZStackSettings`, so `SingleStack` here always means "just the first
/// z-plane."
pub fn resolve_z_projection(
    handling: &ZStackHandling,
    nr_z_stacks: i32,
) -> (ZProjection, RangeInclusive<i32>) {
    match handling {
        ZStackHandling::SingleStack => (ZProjection::None, 0..=0),
        ZStackHandling::AllStacks => (ZProjection::None, 0..=(nr_z_stacks - 1)),
        ZStackHandling::MaxIntensity => (ZProjection::MaxIntensity, 0..=0),
        ZStackHandling::MinIntensity => (ZProjection::MinIntensity, 0..=0),
        ZStackHandling::AvgIntensity => (ZProjection::AvgIntensity, 0..=0),
        ZStackHandling::SumIntensity => (ZProjection::SumIntensity, 0..=0),
        ZStackHandling::TakeTheMiddle => (ZProjection::TakeTheMiddle, 0..=0),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::object::{Object, ObjectInit};
    use bitvec::prelude::*;
    use smartcore::linalg::basic::arrays::Array;

    // -- to_dense_matrix / validate_training_data --------------------------

    #[test]
    fn to_dense_matrix_preserves_shape_and_values() {
        let rows = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]];
        let matrix = to_dense_matrix(&rows).unwrap();
        assert_eq!(matrix.shape(), (2, 3));
    }

    #[test]
    fn validate_training_data_rejects_mismatched_lengths() {
        let rows = vec![vec![1.0], vec![2.0]];
        let labels = [0usize];
        let err = validate_training_data(&rows, &labels).unwrap_err();
        assert!(matches!(err, InternalErrors::Internal(_)));
    }

    #[test]
    fn validate_training_data_rejects_zero_samples() {
        let err = validate_training_data(&[], &[]).unwrap_err();
        let InternalErrors::Internal(msg) = err else {
            panic!("expected Internal, got a different variant");
        };
        assert!(msg.contains("zero samples"));
    }

    #[test]
    fn validate_training_data_accepts_matching_nonempty_input() {
        let rows = vec![vec![1.0], vec![2.0]];
        let labels = [0usize, 1usize];
        assert!(validate_training_data(&rows, &labels).is_ok());
    }

    // -- tile_grid -----------------------------------------------------------

    fn tile_fields(t: &ImageTile) -> (usize, usize, usize, usize) {
        (t.offset_x, t.offset_y, t.width, t.height)
    }

    #[test]
    fn tile_grid_splits_evenly_divisible_dimensions_into_equal_tiles() {
        let tiles = tile_grid(20, 10, 10);
        assert_eq!(tiles.len(), 2);
        assert_eq!(tile_fields(&tiles[0]), (0, 0, 10, 10));
        assert_eq!(tile_fields(&tiles[1]), (10, 0, 10, 10));
    }

    #[test]
    fn tile_grid_shrinks_the_last_tile_for_non_divisible_dimensions() {
        let tiles = tile_grid(15, 8, 10);
        assert_eq!(tiles.len(), 2);
        assert_eq!(tiles[0].width, 10);
        assert_eq!(tiles[1].width, 5);
        assert_eq!(tiles[0].height, 8);
    }

    #[test]
    fn tile_grid_of_a_size_smaller_than_one_tile_returns_a_single_tile() {
        let tiles = tile_grid(5, 5, 100);
        assert_eq!(tiles.len(), 1);
        assert_eq!(tiles[0].width, 5);
        assert_eq!(tiles[0].height, 5);
    }

    // -- bbox_overlaps_tile ----------------------------------------------------

    fn tile(offset_x: usize, offset_y: usize, width: usize, height: usize) -> ImageTile {
        ImageTile {
            offset_x,
            offset_y,
            width,
            height,
        }
    }

    #[test]
    fn bbox_overlaps_tile_true_when_fully_inside() {
        assert!(bbox_overlaps_tile([2, 2, 5, 5], &tile(0, 0, 10, 10)));
    }

    #[test]
    fn bbox_overlaps_tile_true_when_partially_overlapping_at_an_edge() {
        // bbox spans x 8..12, tile covers x 0..9 (inclusive) - overlaps at x=8.
        assert!(bbox_overlaps_tile([8, 0, 12, 5], &tile(0, 0, 10, 10)));
    }

    #[test]
    fn bbox_overlaps_tile_false_when_entirely_to_the_right() {
        assert!(!bbox_overlaps_tile([20, 0, 25, 5], &tile(0, 0, 10, 10)));
    }

    #[test]
    fn bbox_overlaps_tile_false_when_entirely_below() {
        assert!(!bbox_overlaps_tile([0, 20, 5, 25], &tile(0, 0, 10, 10)));
    }

    // -- masked_pixels_in_tile ------------------------------------------------

    fn square_object(x_min: u32, y_min: u32, side: u32) -> Object {
        let x_max = x_min + side - 1;
        let y_max = y_min + side - 1;
        let mut mask = bitvec![u64, Lsb0; 1; (side * side) as usize];
        mask.set(0, true); // no-op, keeps mask fully filled for clarity
        Object::new(ObjectInit {
            bbox: [x_min, y_min, x_max, y_max],
            mask_data: mask,
            area: (side * side) as usize,
            ..Default::default()
        })
    }

    #[test]
    fn masked_pixels_in_tile_returns_every_mask_pixel_when_the_tile_fully_covers_it() {
        let object = square_object(0, 0, 2); // pixels (0,0),(1,0),(0,1),(1,1)
        let pixels = masked_pixels_in_tile(&object, &tile(0, 0, 10, 10));
        let mut pixels = pixels;
        pixels.sort();
        assert_eq!(pixels, vec![(0, 0), (0, 1), (1, 0), (1, 1)]);
    }

    #[test]
    fn masked_pixels_in_tile_clips_to_the_tiles_bounds() {
        let object = square_object(0, 0, 4); // 4x4 object at origin
        // Tile only covers the left half (x: 0..=1).
        let pixels = masked_pixels_in_tile(&object, &tile(0, 0, 2, 4));
        assert!(pixels.iter().all(|&(x, _)| x < 2));
        assert_eq!(pixels.len(), 8); // 2 columns x 4 rows
    }

    #[test]
    fn masked_pixels_in_tile_is_empty_when_the_tile_does_not_overlap_the_object() {
        let object = square_object(0, 0, 2);
        let pixels = masked_pixels_in_tile(&object, &tile(100, 100, 10, 10));
        assert!(pixels.is_empty());
    }

    // -- resolve_z_projection ---------------------------------------------------

    #[test]
    fn resolve_z_projection_maps_every_handling_variant() {
        assert_eq!(
            resolve_z_projection(&ZStackHandling::SingleStack, 5),
            (ZProjection::None, 0..=0)
        );
        assert_eq!(
            resolve_z_projection(&ZStackHandling::AllStacks, 5),
            (ZProjection::None, 0..=4)
        );
        assert_eq!(
            resolve_z_projection(&ZStackHandling::MaxIntensity, 5),
            (ZProjection::MaxIntensity, 0..=0)
        );
        assert_eq!(
            resolve_z_projection(&ZStackHandling::MinIntensity, 5),
            (ZProjection::MinIntensity, 0..=0)
        );
        assert_eq!(
            resolve_z_projection(&ZStackHandling::AvgIntensity, 5),
            (ZProjection::AvgIntensity, 0..=0)
        );
        assert_eq!(
            resolve_z_projection(&ZStackHandling::SumIntensity, 5),
            (ZProjection::SumIntensity, 0..=0)
        );
        assert_eq!(
            resolve_z_projection(&ZStackHandling::TakeTheMiddle, 5),
            (ZProjection::TakeTheMiddle, 0..=0)
        );
    }

    // `Classifier::predict` itself has no cheap default variant to construct
    // here (RandomForest/Knn/Mlp all wrap a real fitted model) - it's
    // exercised end-to-end by `model::random_forest`/`knn`/`mlp`'s own
    // fit+predict tests instead, which cover every variant with a real (if
    // tiny) fitted model.
}
