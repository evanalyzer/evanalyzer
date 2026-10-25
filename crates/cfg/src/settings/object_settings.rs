// # object_settings
//
// **Author:** Joachim Danmayr
// **Date:** 2026-02-06
//
// ## License
// Copyright 2026 Joachim Danmayr.
// Licensed under the **PolyForm Noncommercial License 1.0.0**.

use crate::types::{
    classes::{ObjectClass, SegmentationClass},
    ids::{ObjectId, TrackId},
    image::ImageDimensionality,
};
use bitvec::prelude::*;
use indexmap::IndexMap;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;

#[allow(dead_code)]
#[derive(Debug, Default, Serialize, Deserialize, Clone, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct IntensitySettings {
    /// Sum of all pixel intensities in the object
    pub sum_intensity: f64,
    /// Minimum pixel intensity in the object
    pub min_intensity: f64,
    /// Maximum pixel intensity in the object
    pub max_intensity: f64,
    /// Average pixel intensity in the object
    #[serde(default)]
    pub avg_intensity: f64,
    /// How the pixel values spread (CellProfiler's MeasureObjectIntensity) -
    /// none where not measured (an annotation stored before 2026-10-24).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub std_intensity: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub median_intensity: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub lower_quartile_intensity: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub upper_quartile_intensity: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mad_intensity: Option<f64>,
}

#[derive(Debug, Default, Serialize, Deserialize, Clone, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct TrackSettings {
    pub id: TrackId,
    pub object_ids: Vec<ObjectId>,     // Ordered list of ROIs over time
    pub parent_track: Option<TrackId>, // If created by division
}

#[derive(Debug, Default, Serialize, Deserialize, Clone, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct ObjectMetricSettings {
    // Global unique object ID
    pub id: ObjectId,

    // Semantic class after threshold
    pub segmentation_class: SegmentationClass,

    // Dedicated class after classify object
    pub object_class: HashSet<ObjectClass>,

    // Excludes this object from AI classifier training even though it has an
    // `object_class` assigned (e.g. a bad/outlier annotation) - unlike
    // leaving `object_class` empty, the label is kept for display/manual
    // classification purposes, just not used as a training sample.
    #[serde(default)]
    pub exclude_from_training: bool,

    // Colocalization
    pub colocalized_with: IndexMap<ObjectClass, Vec<ObjectId>>,

    /// The objects this one belongs to, at most one per class: parent
    /// class -> parent object (set by Relate Objects).
    #[serde(default)]
    pub parents: IndexMap<ObjectClass, ObjectId>,

    // Tracking
    pub track: TrackSettings,

    // Are size
    pub area: usize,

    // Bounding box
    pub bbox: [u32; 4], // x_min, y_min, x_max, y_max

    #[schemars(with = "Vec<bool>")]
    pub mask_data: BitVec<u64, Lsb0>, // The mask data (relative to BBox)

    // True if it touches the edge
    pub touches_edge: bool,

    // Accumulators for moments & intensity
    pub sum_x: u64,
    pub sum_y: u64,
    pub sum_x2: u64,
    pub sum_y2: u64,
    pub sum_xy: u64,

    // Intensities
    pub intensities: IndexMap<i32, IntensitySettings>, // Intensity values for each image channel

    // Precomputed perimeter (pixels), filled from Object::get_perimeter so consumers
    // (e.g. the GUI object list) don't recompute the O(bbox area) boundary walk.
    #[serde(default)]
    pub perimeter: f32,

    // Image plane information
    pub z_stack: i32,
    pub c_stack: i32,
    pub t_stack: i32,

    /// One plane (`z_stack`) or a volume of `depth` planes from `z_stack` on -
    /// then `bbox` is the box of every plane and `mask_data` covers them one
    /// after the other (z-major).
    #[serde(default)]
    pub dimensionality: ImageDimensionality,

    /// z moments of a volume, z counted from `z_stack` (0 for one plane).
    #[serde(default, skip_serializing_if = "is_zero")]
    pub sum_z: u64,
    #[serde(default, skip_serializing_if = "is_zero")]
    pub sum_z2: u64,
    #[serde(default, skip_serializing_if = "is_zero")]
    pub sum_xz: u64,
    #[serde(default, skip_serializing_if = "is_zero")]
    pub sum_yz: u64,
}

fn is_zero(value: &u64) -> bool {
    *value == 0
}
