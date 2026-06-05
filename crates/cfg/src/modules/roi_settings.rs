// # roi
//
// **Author:** Joachim Danmayr
// **Date:** 2026-02-06
//
// ## License
// Copyright 2026 Joachim Danmayr.
// Licensed under the **AGPL-3.0**.

use crate::types::{
    classes::{ObjectClass, SegmentationClass},
    ids::{ObjectId, TrackId},
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
    /// Sum of all pixel intensities in the ROI
    pub sum_intensity: f64,
    /// Minimum pixel intensity in the ROI
    pub min_intensity: f32,
    /// Maximum pixel intensity in the ROI
    pub max_intensity: f32,
    /// Median pixel intensity in the ROI
    pub median_intensity: Option<f32>,
    /// Standard deviation of pixel intensities
    pub std_dev: Option<f32>,
    /// All pixel values (used for computing median and std_dev)
    #[serde(skip)]
    pub pixel_values: Vec<f32>,
}

#[derive(Debug, Default, Serialize, Deserialize, Clone, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct TrackSettings {
    pub id: TrackId,
    pub roi_ids: Vec<ObjectId>,        // Ordered list of ROIs over time
    pub parent_track: Option<TrackId>, // If created by division
}

#[derive(Debug, Default, Serialize, Deserialize, Clone, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct RoiSettings {
    // Global unique object ID
    pub id: ObjectId,

    // Semantic class after threshold
    pub segmentation_class: SegmentationClass,

    // Dedicated class after classify roi
    pub object_class: HashSet<ObjectClass>,

    // Colocalization
    pub colocalized_with: IndexMap<ObjectClass, Vec<ObjectId>>,

    // Relation
    pub parent_id: Option<ObjectId>, // Who owns me?
    pub children: Vec<ObjectId>,     // Who is part of me?

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

    // Image plane information
    pub z_stack: i32,
    pub c_stack: i32,
    pub t_stack: i32,
}
