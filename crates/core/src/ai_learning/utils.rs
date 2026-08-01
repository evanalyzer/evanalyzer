use crate::ZProjection;
use crate::ai_learning::model::{Classifier, mlp::predict_mlp};
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
                model
                    .predict(&x)
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
