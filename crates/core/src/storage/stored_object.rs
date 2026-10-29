//! # stored_object
//!
//! An analysed object as read back from the results storage, and the filter
//! to read objects with.
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-10-15
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.

use bitvec::{order::Lsb0, vec::BitVec};
use evanalyzer_cfg::core_types::{ObjectClass, ObjectId};

/// Which stored objects to read. `None` matches all.
#[derive(Debug, Clone, Default)]
pub struct ObjectFilter {
    pub image_rel_path: Option<String>,
    pub t_stack: Option<i32>,
    /// Objects that carry this class (among others). `Unset` matches none:
    /// it is never stored as an object's class.
    pub object_class: Option<ObjectClass>,
    /// Also read the objects' masks ([`StoredObject::mask`]) - off by
    /// default, they are by far the biggest part of an object.
    pub with_masks: bool,
}

impl ObjectFilter {
    /// Whether an object with `classes` passes the class filter.
    pub fn matches_classes(&self, classes: &[ObjectClass]) -> bool {
        self.object_class
            .is_none_or(|class| classes.contains(&class))
    }
}

/// An analysed object as the results storage keeps it - the `objects`
/// table's values. Read back by whoever works on finished results, e.g.
/// dataset steps.
#[derive(Debug, Clone, PartialEq)]
pub struct StoredObject {
    pub id: ObjectId,
    pub image_rel_path: String,
    pub t_stack: i32,
    pub z_stack: i32,
    /// Centre of the bounding box, in pixels.
    pub centroid_px: (f64, f64),
    /// Size of a pixel in x, in nm - `None` if unknown.
    pub pixel_size_nm: Option<f64>,
    /// The object's track - `None` if it isn't part of one.
    pub track_id: Option<u64>,
    /// x_min, y_min, x_max, y_max in pixels.
    pub bbox_px: [u32; 4],
    pub area_px: u64,
    pub object_classes: Vec<ObjectClass>,
    /// Mean intensity per channel (index = channel), normalized to 0..1;
    /// `None` for a channel that wasn't measured.
    pub intensity_mean: Vec<Option<f64>>,
    /// The mask, a bit per pixel of `bbox_px` (inclusive) row by row -
    /// `None` unless [`ObjectFilter::with_masks`] asked for it, or when the
    /// results keep no masks.
    pub mask: Option<BitVec<u64, Lsb0>>,
}

/// The track an object belongs to, as a dataset step sets it.
#[derive(Debug, Clone, PartialEq)]
pub struct TrackAssignment {
    pub object_id: ObjectId,
    /// `0`: none.
    pub track_id: u64,
    /// The tracks this one came from: one for a daughter of a division,
    /// two (or more) for the track two objects merged into - empty for
    /// most. The same for every object of the track.
    pub parent_track_ids: Vec<u64>,
}
