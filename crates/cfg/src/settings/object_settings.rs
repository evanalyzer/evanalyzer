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
    image::{ImageDimensionality, ZProjection},
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

    // Image plane information. A manual annotation marks the object in every
    // channel: its `c_stack` is 0 and not used.
    pub z_stack: i32,
    pub c_stack: i32,
    pub t_stack: i32,

    /// The z-projection it was drawn on (a Max, Avg, ... view), measured on
    /// it - `None`: of plane `z_stack`. One drawn on a projection belongs to
    /// the projections of frame `t_stack` (`z_stack` is 0 and not used): it
    /// is shown in projection views and used by analyses on a z-projection
    /// only. An object of one plane is shown on that plane and in projection
    /// views, and used by analyses of that plane and on a z-projection.
    #[serde(default, skip_serializing_if = "is_one_plane")]
    pub z_projection: ZProjection,

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

impl ObjectMetricSettings {
    /// Whether this object (of one plane, or drawn on a z-projection - not a
    /// volume's) belongs to plane `z` of frame `t`, or (`z: None`) to the
    /// z-projection of frame `t`: what an analysis of that image uses, and
    /// what the viewer shows on it (see [`Self::z_projection`]).
    pub fn belongs_to(&self, z: Option<i32>, t: i32) -> bool {
        self.t_stack == t
            && match z {
                None => true,
                Some(z) => self.z_projection == ZProjection::None && self.z_stack == z,
            }
    }
}

fn is_zero(value: &u64) -> bool {
    *value == 0
}

fn is_one_plane(value: &ZProjection) -> bool {
    *value == ZProjection::None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_planes_object_belongs_to_its_plane_and_the_projections_of_its_frame() {
        let object = ObjectMetricSettings {
            z_stack: 3,
            t_stack: 1,
            ..Default::default()
        };
        assert!(object.belongs_to(Some(3), 1));
        assert!(object.belongs_to(None, 1));
        assert!(!object.belongs_to(Some(2), 1), "another plane");
        assert!(!object.belongs_to(Some(3), 0), "another frame");
        assert!(!object.belongs_to(None, 0), "a projection of another frame");
    }

    #[test]
    fn an_object_drawn_on_a_projection_belongs_to_the_projections_of_its_frame_only() {
        let object = ObjectMetricSettings {
            t_stack: 1,
            z_projection: ZProjection::MaxIntensity,
            ..Default::default()
        };
        assert!(object.belongs_to(None, 1));
        assert!(!object.belongs_to(Some(0), 1), "not plane 0, its unused z");
        assert!(!object.belongs_to(None, 2));
    }

    #[test]
    fn an_object_of_one_plane_is_saved_without_a_projection() {
        let json = serde_json::to_value(ObjectMetricSettings::default()).unwrap();
        assert!(json.get("zProjection").is_none());
        let projected = ObjectMetricSettings {
            z_projection: ZProjection::AvgIntensity,
            ..Default::default()
        };
        let json = serde_json::to_value(&projected).unwrap();
        assert_eq!(json["zProjection"], "AvgIntensity");
        let back: ObjectMetricSettings = serde_json::from_value(json).unwrap();
        assert_eq!(back.z_projection, ZProjection::AvgIntensity);
    }
}
