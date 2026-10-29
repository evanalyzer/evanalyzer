//! # Track Kalman
//!
//! Links the objects of a time-lapse into tracks, predicting each track's
//! next position from its velocity - for objects moving in a direction
//! (e.g. vesicles transported along microtubules). As Track LAP, but a link
//! costs the squared distance to the predicted position.
//!
//! ## Overview
//! Every track segment carries a Kalman filter (Kalman, 1960) per axis
//! with a constant velocity model: state position and velocity, per
//! frame `position += velocity`, the velocity changing at random by
//! `velocity_noise`, the measured positions off by `position_noise`. A new
//! segment has no velocity yet: its second object is looked for within
//! `initial_max_distance` around the first. Gap closing predicts across
//! the missing frames; merges and splits are scored as in Track LAP (by
//! distance, no prediction). The linking steps are those of
//! [`crate::dataset::libs::linking`].
//!
//! Jaqaman et al. (2008) use such a filter in their frame-to-frame step
//! too (their Supplementary Note 7); this is a filter of our own, not
//! theirs.
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
use crate::dataset::tracking::track_lap::squared_distance;
use crate::dataset::{DatasetAlgorithm, DatasetContext};
use crate::storage::stored_object::StoredObject;
use evanalyzer_cfg::core_types::{
    CitationMetadata, ImageChannelIdx, InternalErrors, ObjectClass, SizeUnits,
};
use macros::CommandsMeta;

const KALMAN_1960: CitationMetadata = CitationMetadata {
    cite_key: "kalman1960new",
    title: "A New Approach to Linear Filtering and Prediction Problems",
    authors: &["Rudolph Emil Kalman"],
    year: 1960,
    container: Some("Journal of Basic Engineering"),
    doi: Some("10.1115/1.3662552"),
    url: Some("https://doi.org/10.1115/1.3662552"),
    pages: Some("35-45"),
};

/// Links objects into tracks by the globally best assignment, predicting
/// each track's next position from its velocity (directed motion).
///
/// Objects of `object_class` in two consecutive frames of the same image and
/// z-plane are linked so that the sum of the squared distances to the
/// predicted positions is smallest, each object once. Each track's velocity
/// is followed by a Kalman filter. Then track ends are linked to starts up
/// to `max_gap_frames` later, and - if switched on - track ends into other
/// tracks (merge) and starts out of them (split), as in Track LAP.
#[derive(CommandsMeta)]
#[cmdsmeta(category = "dataset", display_name = "Track Kalman")]
pub struct TrackKalman {
    /// Object class to track.
    pub object_class: ObjectClass,

    /// Unit of the distances (e.g. pixels, nm).
    pub unit: SizeUnits,

    /// How far a new track's second object may be from its first - the
    /// fastest an object moves per frame.
    #[cmdsmeta(default = 15000.0, min = 0.0, step = 100.0)]
    pub initial_max_distance: f32,

    /// How far an object may be from where its track predicts it - also
    /// the farthest a merge or split partner may be.
    #[cmdsmeta(default = 5000.0, min = 0.0, step = 100.0)]
    pub max_distance: f32,

    /// How many frames in a row an object may be missing and its track
    /// still continue. 0: no gap closing.
    #[cmdsmeta(default = 2, min = 0, max = 100)]
    pub max_gap_frames: usize,

    /// How far an object may be from where its track predicts it after a
    /// gap.
    #[cmdsmeta(default = 10000.0, min = 0.0, step = 100.0)]
    pub max_gap_distance: f32,

    /// How exactly an object's position is known (its standard deviation)
    /// - the larger, the more the predictions follow the velocity rather
    ///   than the last position.
    #[cmdsmeta(default = 200.0, min = 0.0, step = 10.0, visibility = Advanced)]
    pub position_noise: f32,

    /// How much the velocity changes from one frame to the next (standard
    /// deviation) - the larger, the faster a track follows a turn.
    #[cmdsmeta(default = 1000.0, min = 0.0, step = 10.0, visibility = Advanced)]
    pub velocity_noise: f32,

    /// Detect two objects merging into one (e.g. vesicle fusion).
    pub merging: bool,

    /// Detect one object splitting into two (e.g. a vesicle budding off).
    pub splitting: bool,

    /// What a merge adds up and a split divides.
    #[cmdsmeta(default = MergeSplitAmount::Intensity, visibility = Advanced)]
    pub merge_split_amount: MergeSplitAmount,

    /// The channel of the intensity, if `merge_split_amount` is the
    /// intensity.
    #[cmdsmeta(visibility = Advanced)]
    pub channel: ImageChannelIdx,
}

impl DatasetAlgorithm for TrackKalman {
    fn execute(&self, ctx: &mut DatasetContext) -> Result<(), InternalErrors> {
        let distances = [self.initial_max_distance, self.max_distance];
        if distances.iter().any(|d| !(*d > 0.0))
            || (self.max_gap_frames > 0 && !(self.max_gap_distance > 0.0))
        {
            return Err(InternalErrors::InvalidArgument(
                "Track Kalman: the maximum distances must be greater than 0".into(),
            ));
        }
        if !(self.position_noise > 0.0) || !(self.velocity_noise >= 0.0) {
            return Err(InternalErrors::InvalidArgument(
                "Track Kalman: the position noise must be greater than 0, the velocity noise at least 0"
                    .into(),
            ));
        }
        let options = GapOptions {
            max_gap_frames: self.max_gap_frames as i32,
            merging: self.merging,
            splitting: self.splitting,
            amount: self.merge_split_amount.of(self.channel),
        };
        linking::run(ctx, self.object_class, false, &options, |object| {
            let px = |value| linking::to_px(value, self.unit, object, self.name());
            Ok(ConstantVelocity {
                initial_max_distance: px(self.initial_max_distance)?,
                max_distance: px(self.max_distance)?,
                max_gap_distance: px(self.max_gap_distance)?,
                position_variance: px(self.position_noise)?.powi(2),
                velocity_variance: px(self.velocity_noise)?.powi(2),
            })
        })
    }

    fn name(&self) -> &'static str {
        "Track Kalman"
    }

    fn cite(&self) -> Vec<&'static CitationMetadata> {
        vec![&JAQAMAN_2008, &KALMAN_1960, &JONKER_1987]
    }
}

/// Costs for directed motion: the squared distance to the predicted
/// position (in pixels).
struct ConstantVelocity {
    initial_max_distance: f64,
    max_distance: f64,
    max_gap_distance: f64,
    /// Of the measured positions.
    position_variance: f64,
    /// Of the velocity's change per frame.
    velocity_variance: f64,
}

/// A segment's filter, x and y apart (the model treats them alike and
/// independently).
#[derive(Debug, Clone)]
struct Filter {
    x: Axis,
    y: Axis,
    /// Objects seen - the velocity is known from the second on.
    seen: usize,
}

/// The estimate of one axis: position, velocity and their covariance.
#[derive(Debug, Clone, PartialEq)]
struct Axis {
    p: f64,
    v: f64,
    pp: f64,
    pv: f64,
    vv: f64,
}

impl Axis {
    fn new(p: f64, position_variance: f64, velocity_variance: f64) -> Self {
        Self {
            p,
            v: 0.0,
            pp: position_variance,
            pv: 0.0,
            vv: velocity_variance,
        }
    }

    /// One frame on: `p += v`, the velocity changing with variance `q`.
    fn predict(&mut self, q: f64) {
        self.p += self.v;
        self.pp += 2.0 * self.pv + self.vv;
        self.pv += self.vv;
        self.vv += q;
    }

    /// Takes in a measured position `z` of variance `r`.
    fn update(&mut self, z: f64, r: f64) {
        let s = self.pp + r;
        let (kp, kv) = (self.pp / s, self.pv / s);
        let innovation = z - self.p;
        self.p += kp * innovation;
        self.v += kv * innovation;
        self.vv -= kv * self.pv;
        self.pv *= 1.0 - kp;
        self.pp *= 1.0 - kp;
    }
}

impl ConstantVelocity {
    /// Where the segment is expected `frames` after its last object, and
    /// how far from there a partner may be.
    fn predict(
        &self,
        filter: &Filter,
        last: &StoredObject,
        frames: i32,
        closing: bool,
    ) -> ((f64, f64), f64) {
        let radius = if closing {
            self.max_gap_distance
        } else if filter.seen < 2 {
            self.initial_max_distance
        } else {
            self.max_distance
        };
        if filter.seen < 2 {
            return (last.centroid_px, radius);
        }
        let (mut x, mut y) = (filter.x.clone(), filter.y.clone());
        for _ in 0..frames {
            x.predict(self.velocity_variance);
            y.predict(self.velocity_variance);
        }
        ((x.p, y.p), radius)
    }
}

impl Linker for ConstantVelocity {
    type State = Filter;

    fn start(&self, object: &StoredObject) -> Filter {
        // Velocity unknown: as fast as the initial search allows (2 sigma).
        let velocity_variance = (self.initial_max_distance / 2.0).powi(2);
        let (x, y) = object.centroid_px;
        Filter {
            x: Axis::new(x, self.position_variance, velocity_variance),
            y: Axis::new(y, self.position_variance, velocity_variance),
            seen: 1,
        }
    }

    fn extend(&self, filter: &mut Filter, object: &StoredObject) {
        let (x, y) = object.centroid_px;
        for (axis, z) in [(&mut filter.x, x), (&mut filter.y, y)] {
            axis.predict(self.velocity_variance);
            axis.update(z, self.position_variance);
        }
        filter.seen += 1;
    }

    fn search(
        &self,
        filter: &Filter,
        last: &StoredObject,
        frames: i32,
        closing: bool,
    ) -> ((f64, f64), f64) {
        self.predict(filter, last, frames, closing)
    }

    fn cost(
        &self,
        filter: &Filter,
        last: &StoredObject,
        frames: i32,
        object: &StoredObject,
        closing: bool,
    ) -> Option<f64> {
        let (centre, radius) = self.predict(filter, last, frames, closing);
        let d2 = squared_distance(centre, object.centroid_px);
        (d2 <= radius.powi(2)).then_some(d2)
    }

    fn max_cost(&self, closing: bool) -> f64 {
        if closing {
            self.max_gap_distance.powi(2)
        } else {
            self.initial_max_distance.max(self.max_distance).powi(2)
        }
    }

    fn pair_cost(&self, a: &StoredObject, b: &StoredObject) -> Option<f64> {
        let d2 = squared_distance(a.centroid_px, b.centroid_px);
        (d2 <= self.max_distance.powi(2)).then_some(d2)
    }

    fn pair_radius(&self, _: &StoredObject) -> f64 {
        self.max_distance
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset::libs::linking::{stacks, track};
    use evanalyzer_cfg::core_types::ObjectId;

    fn object(id: u128, t: i32, x: f64, y: f64) -> StoredObject {
        StoredObject {
            id: ObjectId(id),
            image_rel_path: "a.tif".into(),
            t_stack: t,
            z_stack: 0,
            centroid_px: (x, y),
            pixel_size_nm: Some(100.0),
            track_id: None,
            bbox_px: [0, 0, 1, 1],
            area_px: 1,
            object_classes: vec![ObjectClass::Global(1)],
            intensity_mean: Vec::new(),
            mask: None,
        }
    }

    fn linker() -> ConstantVelocity {
        ConstantVelocity {
            initial_max_distance: 12.0,
            max_distance: 3.0,
            max_gap_distance: 5.0,
            position_variance: 0.25,
            velocity_variance: 0.25,
        }
    }

    fn ids(objects: &[StoredObject], max_gap_frames: i32) -> Vec<u64> {
        let options = GapOptions {
            max_gap_frames,
            merging: false,
            splitting: false,
            amount: MergeSplitAmount::Area.of(ImageChannelIdx(0)),
        };
        let mut next_id = 1;
        let mut linked = Vec::new();
        for frames in stacks(objects).values() {
            linked.extend(track(&linker(), objects, frames, &options, &mut next_id));
        }
        linked.sort_by_key(|l| l.object);
        linked.into_iter().map(|l| l.track_id).collect()
    }

    #[test]
    fn the_filter_learns_a_constant_velocity() {
        let linker = linker();
        let mut filter = linker.start(&object(1, 0, 0.0, 0.0));
        for t in 1..10 {
            linker.extend(&mut filter, &object(1, t, 10.0 * t as f64, -2.0 * t as f64));
        }
        assert!((filter.x.v - 10.0).abs() < 0.2, "{:?}", filter.x);
        assert!((filter.y.v + 2.0).abs() < 0.1, "{:?}", filter.y);
        let ((x, y), _) = linker.search(&filter, &object(1, 9, 90.0, -18.0), 2, true);
        assert!(
            (x - 110.0).abs() < 1.0 && (y + 22.0).abs() < 1.0,
            "{x}, {y}"
        );
    }

    #[test]
    fn two_fast_objects_passing_each_other_keep_their_tracks() {
        // Two vesicles 8 px apart running towards each other at 4 px per
        // frame: by distance alone they swap after passing, by prediction
        // they don't.
        let mut objects = Vec::new();
        for t in 0..6 {
            objects.push(object(2 * t as u128, t, 4.0 * t as f64, 0.0));
            objects.push(object(2 * t as u128 + 1, t, 20.0 - 4.0 * t as f64, 1.0));
        }
        let tracks = ids(&objects, 0);
        let a: Vec<u64> = tracks.iter().step_by(2).copied().collect();
        let b: Vec<u64> = tracks.iter().skip(1).step_by(2).copied().collect();
        assert!(a.iter().all(|&id| id == a[0]), "{tracks:?}");
        assert!(b.iter().all(|&id| id == b[0]), "{tracks:?}");
        assert_ne!(a[0], b[0]);
    }

    #[test]
    fn a_gap_is_bridged_where_the_track_was_heading() {
        // Moving 4 px per frame, missing in frames 3 and 4: frame 5 is 12
        // px from the last position, but where the track was heading.
        let objects: Vec<_> = [0, 1, 2, 5, 6]
            .iter()
            .map(|&t| object(t as u128, t, 4.0 * t as f64, 0.0))
            .collect();
        assert_eq!(ids(&objects, 2), [1, 1, 1, 1, 1]);
        assert_eq!(ids(&objects, 1), [1, 1, 1, 2, 2]);
    }
}
