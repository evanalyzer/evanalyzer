//! # Track LAP
//!
//! Links the objects of a time-lapse into tracks as Jaqaman et al. (2008)
//! do for random (Brownian) motion: frame to frame by the globally best
//! assignment, then gap closing, merging and splitting between the
//! segments. Every link costs the squared distance (equations 3 to 5).
//!
//! ## Overview
//! See [`crate::dataset::libs::linking`]. Unlike the paper, the cutoffs
//! are fixed distances, not derived from each object's motion.
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
use evanalyzer_cfg::core_types::{
    CitationMetadata, ImageChannelIdx, InternalErrors, ObjectClass, SizeUnits,
};
use macros::CommandsMeta;

/// Links objects into tracks by the globally best assignment, for objects
/// moving at random (Brownian motion).
///
/// Objects of `object_class` in two consecutive frames of the same image and
/// z-plane are linked so that the sum of the squared distances is smallest,
/// each object once (Jaqaman et al., 2008). Then track ends are linked to
/// starts up to `max_gap_frames` later (an object missed by the
/// segmentation), and - if switched on - track ends into other tracks
/// (merge) and starts out of them (split). A merge or split starts new
/// tracks, with the old ones as parents.
#[derive(CommandsMeta)]
#[cmdsmeta(category = "dataset", display_name = "Track LAP")]
pub struct TrackLap {
    /// Object class to track.
    pub object_class: ObjectClass,

    /// Unit of the distances (e.g. pixels, nm).
    pub unit: SizeUnits,

    /// How far an object may move from one frame to the next - also the
    /// farthest a merge or split partner may be.
    #[cmdsmeta(default = 10000.0, min = 0.0, step = 100.0)]
    pub max_distance: f32,

    /// How many frames in a row an object may be missing and its track
    /// still continue. 0: no gap closing.
    #[cmdsmeta(default = 2, min = 0, max = 100)]
    pub max_gap_frames: usize,

    /// How far an object may move while it is missing.
    #[cmdsmeta(default = 15000.0, min = 0.0, step = 100.0)]
    pub max_gap_distance: f32,

    /// Detect two objects merging into one (e.g. vesicle fusion).
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

impl DatasetAlgorithm for TrackLap {
    fn execute(&self, ctx: &mut DatasetContext) -> Result<(), InternalErrors> {
        if !(self.max_distance > 0.0) || (self.max_gap_frames > 0 && !(self.max_gap_distance > 0.0))
        {
            return Err(InternalErrors::InvalidArgument(
                "Track LAP: the maximum distances must be greater than 0".into(),
            ));
        }
        let options = GapOptions {
            max_gap_frames: self.max_gap_frames as i32,
            merging: self.merging,
            splitting: self.splitting,
            amount: self.merge_split_amount.of(self.channel),
        };
        linking::run(ctx, self.object_class, false, &options, |object| {
            Ok(Brownian {
                max_distance: linking::to_px(self.max_distance, self.unit, object, self.name())?,
                max_gap_distance: linking::to_px(
                    self.max_gap_distance,
                    self.unit,
                    object,
                    self.name(),
                )?,
            })
        })
    }

    fn name(&self) -> &'static str {
        "Track LAP"
    }

    fn cite(&self) -> Vec<&'static CitationMetadata> {
        vec![&JAQAMAN_2008, &JONKER_1987]
    }
}

/// Costs for random motion: the squared distance (in pixels).
struct Brownian {
    max_distance: f64,
    max_gap_distance: f64,
}

impl Brownian {
    fn cutoff(&self, closing: bool) -> f64 {
        if closing {
            self.max_gap_distance
        } else {
            self.max_distance
        }
    }
}

impl Linker for Brownian {
    type State = ();

    fn start(&self, _: &StoredObject) {}
    fn extend(&self, _: &mut (), _: &StoredObject) {}

    fn search(&self, _: &(), last: &StoredObject, _: i32, closing: bool) -> ((f64, f64), f64) {
        (last.centroid_px, self.cutoff(closing))
    }

    fn cost(
        &self,
        _: &(),
        last: &StoredObject,
        _: i32,
        object: &StoredObject,
        closing: bool,
    ) -> Option<f64> {
        let d2 = squared_distance(last.centroid_px, object.centroid_px);
        (d2 <= self.cutoff(closing).powi(2)).then_some(d2)
    }

    fn max_cost(&self, closing: bool) -> f64 {
        self.cutoff(closing).powi(2)
    }

    fn pair_cost(&self, a: &StoredObject, b: &StoredObject) -> Option<f64> {
        self.cost(&(), a, 1, b, false)
    }

    fn pair_radius(&self, _: &StoredObject) -> f64 {
        self.max_distance
    }
}

pub(crate) fn squared_distance(a: (f64, f64), b: (f64, f64)) -> f64 {
    (a.0 - b.0).powi(2) + (a.1 - b.1).powi(2)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset::libs::linking::{Linked, stacks, track};
    use evanalyzer_cfg::core_types::ObjectId;

    fn object(id: u128, t: i32, x: f64, y: f64, area: u64) -> StoredObject {
        StoredObject {
            id: ObjectId(id),
            image_rel_path: "a.tif".into(),
            t_stack: t,
            z_stack: 0,
            centroid_px: (x, y),
            pixel_size_nm: Some(100.0),
            track_id: None,
            bbox_px: [0, 0, 1, 1],
            area_px: area,
            object_classes: vec![ObjectClass::Global(1)],
            intensity_mean: Vec::new(),
            mask: None,
        }
    }

    fn options(max_gap_frames: i32, merging: bool, splitting: bool) -> GapOptions {
        GapOptions {
            max_gap_frames,
            merging,
            splitting,
            amount: MergeSplitAmount::Area.of(ImageChannelIdx(0)),
        }
    }

    /// (track, parents) of each object, in input order.
    fn tracks(objects: &[StoredObject], options: &GapOptions) -> Vec<(u64, Vec<u64>)> {
        let linker = Brownian {
            max_distance: 10.0,
            max_gap_distance: 15.0,
        };
        let mut next_id = 1;
        let mut linked: Vec<Linked> = Vec::new();
        for frames in stacks(objects).values() {
            linked.extend(track(&linker, objects, frames, options, &mut next_id));
        }
        linked.sort_by_key(|l| l.object);
        assert_eq!(linked.len(), objects.len(), "every object gets a track");
        linked
            .into_iter()
            .map(|l| (l.track_id, l.parent_track_ids))
            .collect()
    }

    fn ids(tracks: &[(u64, Vec<u64>)]) -> Vec<u64> {
        tracks.iter().map(|t| t.0).collect()
    }

    #[test]
    fn crossing_objects_get_the_globally_best_links() {
        // Closest pair first links 1 to 3 (distance 3), and 2 (12 from 4)
        // can't continue; the best sum links 1 to 4 and 2 to 3.
        let objects = [
            object(1, 0, 0.0, 0.0, 1),
            object(2, 0, 8.0, 0.0, 1),
            object(3, 1, 3.0, 0.0, 1),
            object(4, 1, -4.0, 0.0, 1),
        ];
        assert_eq!(
            ids(&tracks(&objects, &options(0, false, false))),
            [1, 2, 2, 1]
        );
    }

    #[test]
    fn a_missing_frame_is_bridged_up_to_the_maximum_gap() {
        let objects = [
            object(1, 0, 0.0, 0.0, 1),
            object(2, 1, 1.0, 0.0, 1),
            // frames 2 and 3 missing
            object(3, 4, 3.0, 0.0, 1),
            object(4, 5, 4.0, 0.0, 1),
        ];
        assert_eq!(
            ids(&tracks(&objects, &options(2, false, false))),
            [1, 1, 1, 1]
        );
        assert_eq!(
            ids(&tracks(&objects, &options(1, false, false))),
            [1, 1, 2, 2],
            "a gap of 2 > 1"
        );
        assert_eq!(
            ids(&tracks(&objects, &options(0, false, false))),
            [1, 1, 2, 2]
        );
    }

    #[test]
    fn a_division_ends_the_mother_and_starts_two_daughters() {
        // A cell of area 10 splits into two of 5 in frame 2.
        let objects = [
            object(1, 0, 0.0, 0.0, 10),
            object(2, 1, 0.0, 0.0, 10),
            object(3, 2, 0.0, 3.0, 5),
            object(4, 2, 0.0, -3.0, 5),
            object(5, 3, 0.0, 4.0, 5),
            object(6, 3, 0.0, -4.0, 5),
        ];
        let got = tracks(&objects, &options(0, false, true));
        assert_eq!(ids(&got), [1, 1, 2, 3, 2, 3]);
        assert_eq!(got[0].1, Vec::<u64>::new());
        assert_eq!(got[2].1, [1]);
        assert_eq!(got[3].1, [1]);
        assert_eq!(got[5].1, [1], "every object of a daughter has the parent");

        let without = tracks(&objects, &options(0, false, false));
        assert!(without.iter().all(|t| t.1.is_empty()), "splitting is off");
    }

    #[test]
    fn a_fusion_ends_both_tracks_and_starts_one_with_two_parents() {
        let objects = [
            object(1, 0, 0.0, 5.0, 5),
            object(2, 0, 0.0, -5.0, 5),
            object(3, 1, 0.0, 3.0, 5),
            object(4, 1, 0.0, -3.0, 5),
            object(5, 2, 0.0, 0.0, 10),
            object(6, 3, 0.0, 0.0, 10),
        ];
        let got = tracks(&objects, &options(0, true, false));
        assert_eq!(ids(&got), [1, 2, 1, 2, 3, 3]);
        assert_eq!(got[4].1, [1, 2]);
    }

    #[test]
    fn a_merge_whose_amount_does_not_add_up_is_refused() {
        // The merged object is 10 times too small: the penalty (rho^-2 =
        // 100) makes the merge dearer than ending the track.
        let objects = [
            object(1, 0, 0.0, 5.0, 50),
            object(2, 0, 0.0, -5.0, 50),
            object(3, 1, 0.0, 3.0, 50),
            object(4, 1, 0.0, -3.0, 50),
            object(5, 2, 0.0, 0.0, 10),
        ];
        let got = tracks(&objects, &options(0, true, false));
        assert!(got.iter().all(|t| t.1.is_empty()), "{got:?}");
    }

    #[test]
    fn different_z_planes_are_tracked_apart() {
        let mut objects = vec![object(1, 0, 0.0, 0.0, 1), object(2, 1, 0.0, 0.0, 1)];
        objects[1].z_stack = 1;
        assert_eq!(ids(&tracks(&objects, &options(2, false, false))), [1, 2]);
    }
}

#[cfg(test)]
mod store_tests {
    use super::*;
    use crate::object::{Object, ObjectInit, Track};
    use crate::pipeline::pipeline_cache::GlobalPipelineCache;
    use crate::storage::stored_object::ObjectFilter;
    use crate::{DuckDbExporter, PipelineResultExporter};
    use evanalyzer_cfg::core_types::{ImagePlane, ObjectId, TrackId};
    use std::collections::HashSet;
    use std::sync::{Arc, atomic::AtomicBool};

    fn cell(id: u128, class: u32, t: i32, x: u32, track: u64) -> Object {
        let mut object = Object::new(ObjectInit {
            id: ObjectId(id),
            object_class: HashSet::from([ObjectClass::Global(class)]),
            area: 4,
            bbox: [x, 10, x + 1, 11],
            mask_data: bitvec::bitvec![u64, bitvec::order::Lsb0; 1; 4],
            plane: ImagePlane { z: 0, c: 0, t },
            ..Default::default()
        });
        object.track = Track {
            id: TrackId(track),
            ..Default::default()
        };
        object
    }

    #[test]
    fn every_image_is_tracked_on_its_own_after_the_other_classes_tracks() {
        let dir = tempfile::tempdir().unwrap();
        let exporter = DuckDbExporter::new(
            dir.path().join("results.evadb"),
            Default::default(),
            &Default::default(),
        )
        .unwrap();
        for (image, objects) in [
            (
                "a.tif",
                vec![
                    cell(1, 1, 0, 10, 0),
                    cell(2, 1, 1, 12, 0),
                    cell(3, 2, 0, 10, 7),
                ],
            ),
            // Frame 1 missing: bridged.
            ("b.tif", vec![cell(4, 1, 0, 10, 0), cell(5, 1, 2, 13, 0)]),
        ] {
            let mut cache = GlobalPipelineCache::default();
            cache.image_rel_path = image.into();
            cache.image_meta.nr_of_bits = 8;
            for object in objects {
                cache.object_cache.insert(object.id.clone(), object);
            }
            exporter.export(&cache).unwrap();
        }
        let tracker = TrackLap {
            object_class: ObjectClass::Global(1),
            unit: SizeUnits::Pixels,
            max_distance: 5.0,
            max_gap_frames: 1,
            max_gap_distance: 5.0,
            merging: true,
            splitting: true,
            merge_split_amount: MergeSplitAmount::Area,
            channel: ImageChannelIdx(0),
        };
        let store = exporter.dataset_store().unwrap();
        let mut ctx = DatasetContext {
            store,
            progress: None,
            cancel: Arc::new(AtomicBool::new(false)),
        };
        tracker.execute(&mut ctx).unwrap();

        let track = |id: u128| {
            store
                .get_objects(&ObjectFilter::default())
                .unwrap()
                .into_iter()
                .find(|o| o.id.0 == id)
                .unwrap()
                .track_id
        };
        assert_eq!(
            [track(1), track(2)],
            [Some(8), Some(8)],
            "after the other class's 7"
        );
        assert_eq!(
            [track(4), track(5)],
            [Some(9), Some(9)],
            "the next image goes on"
        );
        assert_eq!(track(3), Some(7), "the other class is untouched");
    }

    #[test]
    fn a_zero_distance_is_refused() {
        let exporter = crate::MemoryExporter::default();
        let mut ctx = DatasetContext {
            store: exporter.dataset_store().unwrap(),
            progress: None,
            cancel: Arc::new(AtomicBool::new(false)),
        };
        let tracker = TrackLap {
            object_class: ObjectClass::Global(1),
            unit: SizeUnits::Pixels,
            max_distance: 0.0,
            max_gap_frames: 2,
            max_gap_distance: 10.0,
            merging: false,
            splitting: false,
            merge_split_amount: MergeSplitAmount::Area,
            channel: ImageChannelIdx(0),
        };
        let err = tracker.execute(&mut ctx).unwrap_err().to_string();
        assert!(err.contains("greater than 0"), "{err}");
    }
}
