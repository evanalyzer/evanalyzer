//! # Track Overlap
//!
//! Links the objects of a time-lapse into tracks by how much their masks
//! overlap - for large objects that move less than their size from frame
//! to frame (e.g. cells), where no distance needs to be chosen.
//!
//! ## Overview
//! A link costs `1 - IoU` (intersection over union of the two masks); it
//! is possible from an IoU of `min_overlap` on. Merges and splits compare
//! the smaller object with the bigger one instead - a daughter covers at
//! most half of its mother, but nearly all of it lies inside her: their
//! cost is `1 - intersection / smaller area`, times the amount penalty of
//! Track LAP. The linking steps are those of
//! [`crate::dataset::libs::linking`]. Without stored masks, the bounding
//! boxes overlap instead.
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-10-29
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.

use crate::dataset::libs::linking::{
    self, GapOptions, JAQAMAN_2008, JONKER_1987, Linker, MergeSplitAmount,
};
use crate::dataset::{DatasetAlgorithm, DatasetContext};
use crate::storage::stored_object::StoredObject;
use evanalyzer_cfg::core_types::{CitationMetadata, ImageChannelIdx, InternalErrors, ObjectClass};
use macros::CommandsMeta;

/// Links objects into tracks by the globally best assignment of
/// overlapping masks.
///
/// Objects of `object_class` in two consecutive frames of the same image and
/// z-plane are linked so that they overlap the most in total, each object
/// once - two objects can be linked if their intersection over union is
/// at least `min_overlap`. Then track ends are linked to starts up to
/// `max_gap_frames` later, and - if switched on - track ends into other
/// tracks (merge) and starts out of them (split). A merge or split starts
/// new tracks, with the old ones as parents.
#[derive(CommandsMeta)]
#[cmdsmeta(category = "dataset", display_name = "Track Overlap")]
pub struct TrackOverlap {
    /// Object class to track.
    pub object_class: ObjectClass,

    /// How much two objects must overlap to be linked: intersection over
    /// union, from 0 to 1 (1: the same mask). For a merge or split, the
    /// part of the smaller object inside the bigger one.
    #[cmdsmeta(default = 0.3, min = 0.01, max = 1.0, step = 0.05)]
    pub min_overlap: f32,

    /// How many frames in a row an object may be missing and its track
    /// still continue. 0: no gap closing.
    #[cmdsmeta(default = 2, min = 0, max = 100)]
    pub max_gap_frames: usize,

    /// Detect two objects merging into one.
    pub merging: bool,

    /// Detect one object splitting into two (e.g. cell division).
    pub splitting: bool,

    /// What a merge adds up and a split divides.
    #[cmdsmeta(default = MergeSplitAmount::Area, visibility = Advanced)]
    pub merge_split_amount: MergeSplitAmount,

    /// The channel of the intensity, if `merge_split_amount` is the
    /// intensity.
    #[cmdsmeta(visibility = Advanced)]
    pub channel: ImageChannelIdx,
}

impl DatasetAlgorithm for TrackOverlap {
    fn execute(&self, ctx: &mut DatasetContext) -> Result<(), InternalErrors> {
        if !(self.min_overlap > 0.0 && self.min_overlap <= 1.0) {
            return Err(InternalErrors::InvalidArgument(
                "Track Overlap: the minimum overlap must be greater than 0 and at most 1".into(),
            ));
        }
        let options = GapOptions {
            max_gap_frames: self.max_gap_frames as i32,
            merging: self.merging,
            splitting: self.splitting,
            amount: self.merge_split_amount.of(self.channel),
        };
        let linker = Overlap {
            min_overlap: self.min_overlap as f64,
        };
        linking::run(ctx, self.object_class, true, &options, |_| Ok(linker))
    }

    fn name(&self) -> &'static str {
        "Track Overlap"
    }

    fn cite(&self) -> Vec<&'static CitationMetadata> {
        vec![&JAQAMAN_2008, &JONKER_1987]
    }
}

/// Costs by overlap: `1 - IoU`.
#[derive(Clone, Copy)]
struct Overlap {
    min_overlap: f64,
}

impl Linker for Overlap {
    type State = ();

    fn start(&self, _: &StoredObject) {}
    fn extend(&self, _: &mut (), _: &StoredObject) {}

    fn search(&self, _: &(), last: &StoredObject, _: i32, _: bool) -> ((f64, f64), f64) {
        (last.centroid_px, self.extent(last))
    }

    fn cost(
        &self,
        _: &(),
        last: &StoredObject,
        _: i32,
        object: &StoredObject,
        _: bool,
    ) -> Option<f64> {
        let intersection = intersection(last, object) as f64;
        let union = area(last) + area(object) - intersection;
        let iou = if union > 0.0 {
            intersection / union
        } else {
            0.0
        };
        (iou >= self.min_overlap).then_some(1.0 - iou)
    }

    fn max_cost(&self, _: bool) -> f64 {
        1.0 - self.min_overlap
    }

    fn pair_cost(&self, a: &StoredObject, b: &StoredObject) -> Option<f64> {
        let smaller = area(a).min(area(b));
        let covered = if smaller > 0.0 {
            intersection(a, b) as f64 / smaller
        } else {
            0.0
        };
        (covered >= self.min_overlap).then_some(1.0 - covered)
    }

    fn pair_radius(&self, a: &StoredObject) -> f64 {
        self.extent(a)
    }

    /// From the centroid to the farthest corner of the box: every pixel of
    /// the object is that close.
    fn extent(&self, object: &StoredObject) -> f64 {
        let [x0, y0, x1, y1] = object.bbox_px;
        let (cx, cy) = object.centroid_px;
        let dx = (cx - x0 as f64).abs().max((x1 as f64 + 1.0 - cx).abs());
        let dy = (cy - y0 as f64).abs().max((y1 as f64 + 1.0 - cy).abs());
        dx.hypot(dy)
    }
}

/// The object's pixels: its mask's, else its box's.
fn area(object: &StoredObject) -> f64 {
    match &object.mask {
        Some(mask) => mask.count_ones() as f64,
        None => {
            let [x0, y0, x1, y1] = object.bbox_px;
            ((x1 - x0 + 1) as f64) * ((y1 - y0 + 1) as f64)
        }
    }
}

/// The pixels `a` and `b` share (boxes inclusive, masks row by row).
fn intersection(a: &StoredObject, b: &StoredObject) -> u64 {
    let [ax0, ay0, ax1, ay1] = a.bbox_px;
    let [bx0, by0, bx1, by1] = b.bbox_px;
    let (x0, y0, x1, y1) = (ax0.max(bx0), ay0.max(by0), ax1.min(bx1), ay1.min(by1));
    if x0 > x1 || y0 > y1 {
        return 0;
    }
    let set = |o: &StoredObject, x: u32, y: u32| match &o.mask {
        Some(mask) => {
            let [ox0, oy0, ox1, _] = o.bbox_px;
            let width = (ox1 - ox0 + 1) as usize;
            mask[(y - oy0) as usize * width + (x - ox0) as usize]
        }
        None => true,
    };
    let mut shared = 0;
    for y in y0..=y1 {
        for x in x0..=x1 {
            if set(a, x, y) && set(b, x, y) {
                shared += 1;
            }
        }
    }
    shared
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset::libs::linking::{stacks, track};
    use bitvec::prelude::*;
    use evanalyzer_cfg::core_types::ObjectId;

    /// A filled box, its mask stored.
    fn cell(id: u128, t: i32, bbox: [u32; 4]) -> StoredObject {
        let [x0, y0, x1, y1] = bbox;
        let len = ((x1 - x0 + 1) * (y1 - y0 + 1)) as usize;
        StoredObject {
            id: ObjectId(id),
            image_rel_path: "a.tif".into(),
            t_stack: t,
            z_stack: 0,
            centroid_px: ((x0 + x1) as f64 / 2.0, (y0 + y1) as f64 / 2.0),
            pixel_size_nm: None,
            track_id: None,
            bbox_px: bbox,
            area_px: len as u64,
            object_classes: vec![ObjectClass::Global(1)],
            intensity_mean: Vec::new(),
            mask: Some(BitVec::repeat(true, len)),
        }
    }

    fn tracks(objects: &[StoredObject], splitting: bool) -> Vec<(u64, Vec<u64>)> {
        let options = GapOptions {
            max_gap_frames: 1,
            merging: false,
            splitting,
            amount: MergeSplitAmount::Area.of(ImageChannelIdx(0)),
        };
        let linker = Overlap { min_overlap: 0.3 };
        let mut next_id = 1;
        let mut linked = Vec::new();
        for frames in stacks(objects).values() {
            linked.extend(track(&linker, objects, frames, &options, &mut next_id));
        }
        linked.sort_by_key(|l| l.object);
        linked
            .into_iter()
            .map(|l| (l.track_id, l.parent_track_ids))
            .collect()
    }

    #[test]
    fn the_masks_count_not_the_boxes() {
        // Two L-shaped objects whose boxes overlap but whose pixels don't.
        let mut a = cell(1, 0, [0, 0, 9, 9]);
        let mut b = cell(2, 1, [0, 0, 9, 9]);
        let (ma, mb) = (a.mask.as_mut().unwrap(), b.mask.as_mut().unwrap());
        for i in 0..100 {
            ma.set(i, i < 50);
            mb.set(i, i >= 50);
        }
        assert_eq!(intersection(&a, &b), 0);
        assert_eq!(area(&a), 50.0);
        assert_eq!(
            Overlap { min_overlap: 0.3 }.cost(&(), &a, 1, &b, false),
            None
        );
    }

    #[test]
    fn a_slowly_moving_cell_is_followed_and_a_new_one_starts_a_track() {
        let objects = [
            cell(1, 0, [0, 0, 9, 9]),
            cell(2, 1, [2, 0, 11, 9]), // IoU 80 / 120
            cell(3, 1, [50, 50, 59, 59]),
            cell(4, 2, [4, 0, 13, 9]),
        ];
        let ids: Vec<u64> = tracks(&objects, false).iter().map(|t| t.0).collect();
        assert_eq!(ids, [1, 1, 2, 1]);
    }

    #[test]
    fn a_dividing_cell_gets_two_daughters() {
        let objects = [
            cell(1, 0, [0, 0, 9, 9]),
            cell(2, 1, [0, 0, 9, 9]),
            cell(3, 2, [0, 0, 9, 4]),
            cell(4, 2, [0, 5, 9, 9]),
        ];
        let got = tracks(&objects, true);
        assert_eq!(got.iter().map(|t| t.0).collect::<Vec<_>>(), [1, 1, 2, 3]);
        assert_eq!(got[2].1, [1]);
        assert_eq!(got[3].1, [1]);
    }
}
