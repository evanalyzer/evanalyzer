//! # Simple Tracker
//!
//! Links the objects of a time-lapse into tracks: each object in frame
//! `t + 1` continues the track of the nearest object in frame `t`, if that
//! one is close enough and not claimed by an even closer object.
//!
//! ## Overview
//! Per image and z-plane, two consecutive frames are matched by global
//! greedy nearest neighbour: all pairs within `max_distance` (centre of the
//! bounding box), closest first, each object used once. An object without a
//! partner starts a new track; a frame without objects ends every track. No
//! gap closing, no divisions - see TRACKING.md for what may follow.

use crate::dataset::{DatasetAlgorithm, DatasetContext};
use crate::storage::stored_object::{ObjectFilter, StoredObject};
use evanalyzer_cfg::core_types::{CitationMetadata, InternalErrors, ObjectClass, SizeUnits};
use macros::CommandsMeta;
use std::collections::{BTreeMap, HashMap};

/// Links each object to the nearest one in the next frame.
///
/// Objects of `object_class` in two consecutive frames of the same image and
/// z-plane are linked closest pair first, each object once, if they are at
/// most `max_distance` apart. Every object gets a track id; one that can't
/// be linked starts a new track.
#[derive(CommandsMeta)]
#[cmdsmeta(category = "dataset")]
pub struct SimpleTracker {
    /// Object class to track.
    pub object_class: ObjectClass,

    /// Unit in which `max_distance` is expressed (e.g. pixels, nm).
    pub unit: SizeUnits,

    /// How far an object may move from one frame to the next.
    ///
    /// Farther apart, it starts a new track.
    #[cmdsmeta(default = 10000.0, min = 0.0, step = 100.0)]
    pub max_distance: f32,
}

impl DatasetAlgorithm for SimpleTracker {
    fn execute(&self, ctx: &mut DatasetContext) -> Result<(), InternalErrors> {
        if !(self.max_distance > 0.0) {
            return Err(InternalErrors::InvalidArgument(
                "Simple Tracker: the maximum distance must be greater than 0".into(),
            ));
        }
        let objects = ctx.store.get_objects(&ObjectFilter {
            object_class: Some(self.object_class),
            ..Default::default()
        })?;
        // Continue after the tracks other steps (other classes) made, so
        // that track ids stay unique.
        let first_id = ctx.store.max_track_id_except(self.object_class)? + 1;

        let tracks = link_tracks(&objects, first_id, |object| self.max_distance_px(object))?;
        if ctx.is_cancelled() {
            return Err(InternalErrors::Cancelled);
        }
        let ids: Vec<_> = objects
            .iter()
            .zip(tracks)
            .map(|(object, track)| (object.id.clone(), track))
            .collect();
        ctx.store.set_track_ids(&ids)
    }

    fn name(&self) -> &'static str {
        "Simple Tracker"
    }

    fn cite(&self) -> Vec<&'static CitationMetadata> {
        vec![&CitationMetadata::DANMAYR]
    }
}

impl SimpleTracker {
    /// `max_distance` in pixels of `object`'s image.
    fn max_distance_px(&self, object: &StoredObject) -> Result<f64, InternalErrors> {
        match self.unit {
            SizeUnits::Pixels => Ok(self.max_distance as f64),
            SizeUnits::NanoMeter => match object.pixel_size_nm {
                Some(nm_per_px) if nm_per_px > 0.0 => Ok(self.max_distance as f64 / nm_per_px),
                _ => Err(InternalErrors::InvalidArgument(format!(
                    "Simple Tracker: {} has no pixel size - give the maximum distance in pixels",
                    object.image_rel_path
                ))),
            },
        }
    }
}

/// The track id of each of `objects` (same order), numbered from `first_id`
/// in image, z-plane and frame order. `max_distance_px` gives the linking
/// distance for an object's image.
fn link_tracks(
    objects: &[StoredObject],
    first_id: u64,
    max_distance_px: impl Fn(&StoredObject) -> Result<f64, InternalErrors>,
) -> Result<Vec<u64>, InternalErrors> {
    // image, z -> frame -> object indices
    let mut stacks: BTreeMap<(&str, i32), BTreeMap<i32, Vec<usize>>> = BTreeMap::new();
    for (i, object) in objects.iter().enumerate() {
        stacks
            .entry((&object.image_rel_path, object.z_stack))
            .or_default()
            .entry(object.t_stack)
            .or_default()
            .push(i);
    }

    let mut tracks = vec![0u64; objects.len()];
    let mut next_id = first_id;
    for frames in stacks.values() {
        let Some(&first) = frames.values().next().and_then(|f| f.first()) else {
            continue;
        };
        let max_distance = max_distance_px(&objects[first])?;
        let mut previous: Option<(i32, &Vec<usize>)> = None;
        for (&t, current) in frames {
            let links = match previous {
                Some((prev_t, prev)) if prev_t + 1 == t => {
                    nearest_pairs(objects, prev, current, max_distance)
                }
                _ => HashMap::new(),
            };
            for &i in current {
                tracks[i] = match links.get(&i) {
                    Some(&from) => tracks[from],
                    None => {
                        next_id += 1;
                        next_id - 1
                    }
                };
            }
            previous = Some((t, current));
        }
    }
    Ok(tracks)
}

/// Links objects of `current` to objects of `previous` (indices into
/// `objects`): all pairs at most `max_distance` apart, closest first, each
/// object once. Returns current -> previous.
fn nearest_pairs(
    objects: &[StoredObject],
    previous: &[usize],
    current: &[usize],
    max_distance: f64,
) -> HashMap<usize, usize> {
    // A grid with `max_distance` cells: a partner can only be in the same
    // or a neighbouring cell, so each object is compared with few others.
    let cell = |(x, y): (f64, f64)| {
        (
            (x / max_distance).floor() as i64,
            (y / max_distance).floor() as i64,
        )
    };
    let mut grid: HashMap<(i64, i64), Vec<usize>> = HashMap::new();
    for &p in previous {
        grid.entry(cell(objects[p].centroid_px))
            .or_default()
            .push(p);
    }

    let max_sq = max_distance * max_distance;
    let mut pairs: Vec<(f64, usize, usize)> = Vec::new();
    for &c in current {
        let (x, y) = objects[c].centroid_px;
        let (cx, cy) = cell((x, y));
        for dx in -1..=1 {
            for dy in -1..=1 {
                for &p in grid.get(&(cx + dx, cy + dy)).into_iter().flatten() {
                    let (px, py) = objects[p].centroid_px;
                    let d_sq = (x - px).powi(2) + (y - py).powi(2);
                    if d_sq <= max_sq {
                        pairs.push((d_sq, p, c));
                    }
                }
            }
        }
    }
    // Closest first; ties by index, so the result never depends on hashing.
    pairs.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)).then(a.2.cmp(&b.2)));

    let mut taken_previous = std::collections::HashSet::new();
    let mut links = HashMap::new();
    for (_, p, c) in pairs {
        if !links.contains_key(&c) && taken_previous.insert(p) {
            links.insert(c, p);
        }
    }
    links
}

#[cfg(test)]
mod tests {
    use super::*;
    use evanalyzer_cfg::core_types::ObjectId;

    fn object(id: u128, image: &str, t: i32, x: f64, y: f64) -> StoredObject {
        StoredObject {
            id: ObjectId(id),
            image_rel_path: image.into(),
            t_stack: t,
            z_stack: 0,
            centroid_px: (x, y),
            pixel_size_nm: Some(100.0),
            track_id: None,
            bbox_px: [0, 0, 1, 1],
            area_px: 1,
            object_classes: vec![ObjectClass::Valid(1)],
            intensity_mean: Vec::new(),
        }
    }

    fn link(objects: &[StoredObject], max_distance: f64) -> Vec<u64> {
        link_tracks(objects, 1, |_| Ok(max_distance)).unwrap()
    }

    #[test]
    fn two_cells_moving_side_by_side_keep_their_tracks() {
        let objects = [
            object(1, "a.tif", 0, 10.0, 10.0),
            object(2, "a.tif", 0, 50.0, 10.0),
            object(3, "a.tif", 1, 53.0, 12.0),
            object(4, "a.tif", 1, 12.0, 11.0),
            object(5, "a.tif", 2, 15.0, 13.0),
            object(6, "a.tif", 2, 55.0, 14.0),
        ];
        assert_eq!(link(&objects, 10.0), [1, 2, 2, 1, 1, 2]);
    }

    #[test]
    fn the_closest_pair_wins_and_the_other_object_starts_a_new_track() {
        // Both in frame 1 are near the one in frame 0; the closer one
        // continues its track.
        let objects = [
            object(1, "a.tif", 0, 10.0, 10.0),
            object(2, "a.tif", 1, 16.0, 10.0),
            object(3, "a.tif", 1, 12.0, 10.0),
        ];
        assert_eq!(link(&objects, 10.0), [1, 2, 1]);
    }

    #[test]
    fn too_far_a_frame_gap_or_another_image_starts_a_new_track() {
        let objects = [
            object(1, "a.tif", 0, 10.0, 10.0),
            object(2, "a.tif", 1, 30.0, 10.0), // 20 px away
            object(3, "a.tif", 3, 30.0, 10.0), // frame 2 missing
            object(4, "b.tif", 0, 10.0, 10.0),
            object(5, "b.tif", 1, 11.0, 10.0),
        ];
        assert_eq!(link(&objects, 10.0), [1, 2, 3, 4, 4]);
    }

    #[test]
    fn partners_across_a_grid_cell_border_are_found() {
        let objects = [
            object(1, "a.tif", 0, 9.9, 9.9),
            object(2, "a.tif", 1, 10.1, 10.1),
        ];
        assert_eq!(link(&objects, 10.0), [1, 1]);
    }

    #[test]
    fn the_maximum_distance_in_nm_is_converted_with_the_images_pixel_size() {
        let tracker = SimpleTracker {
            object_class: ObjectClass::Valid(1),
            unit: SizeUnits::NanoMeter,
            max_distance: 1000.0,
        };
        let mut row = object(1, "a.tif", 0, 0.0, 0.0);
        assert_eq!(tracker.max_distance_px(&row).unwrap(), 10.0);
        row.pixel_size_nm = None;
        let err = tracker.max_distance_px(&row).unwrap_err().to_string();
        assert!(err.contains("no pixel size"), "{err}");
    }
}

#[cfg(test)]
mod store_tests {
    use super::*;
    use crate::object::{Object, ObjectInit, Track};
    use crate::pipeline::pipeline_cache::GlobalPipelineCache;
    use crate::{DuckDbExporter, PipelineResultExporter};
    use evanalyzer_cfg::core_types::{ImagePlane, ObjectId, TrackId};
    use std::collections::HashSet;
    use std::sync::{Arc, atomic::AtomicBool};

    fn cell(id: u128, class: u32, t: i32, x: u32, track: u64) -> Object {
        let mut object = Object::new(ObjectInit {
            id: ObjectId(id),
            object_class: HashSet::from([ObjectClass::Valid(class)]),
            area: 4,
            bbox: [x, 10, x + 2, 12],
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
    fn tracks_are_written_to_the_results_after_the_other_classes_tracks() {
        let dir = tempfile::tempdir().unwrap();
        let exporter = DuckDbExporter::new(
            dir.path().join("results.evadb"),
            Default::default(),
            &Default::default(),
        )
        .unwrap();
        let mut cache = GlobalPipelineCache::default();
        cache.image_rel_path = "movie.tif".into();
        cache.image_meta.nr_of_bits = 8;
        for object in [
            cell(1, 1, 0, 10, 0),
            cell(2, 1, 1, 13, 0),
            cell(3, 1, 2, 80, 0), // jumped too far
            cell(4, 2, 0, 10, 7), // another class, already tracked
        ] {
            cache.object_cache.insert(object.id.clone(), object);
        }
        exporter.export(&cache).unwrap();

        let tracker = SimpleTracker {
            object_class: ObjectClass::Valid(1),
            unit: SizeUnits::Pixels,
            max_distance: 10.0,
        };
        let store = exporter.dataset_store().unwrap();
        let mut ctx = DatasetContext {
            store,
            progress: None,
            cancel: Arc::new(AtomicBool::new(false)),
        };
        tracker.execute(&mut ctx).unwrap();

        let tracks: Vec<(u128, Option<u64>)> = store
            .get_objects(&ObjectFilter::default())
            .unwrap()
            .into_iter()
            .map(|o| (o.id.0, o.track_id))
            .collect();
        let track = |id| tracks.iter().find(|(o, _)| *o == id).unwrap().1;
        assert_eq!(track(1), Some(8), "after the other class's track 7");
        assert_eq!(track(2), Some(8));
        assert_eq!(track(3), Some(9));
        assert_eq!(track(4), Some(7), "the other class is untouched");

        // Running it again gives the same tracks.
        tracker.execute(&mut ctx).unwrap();
        assert_eq!(
            store
                .get_objects(&ObjectFilter::default())
                .unwrap()
                .iter()
                .find(|o| o.id.0 == 3)
                .unwrap()
                .track_id,
            Some(9)
        );
    }

    #[test]
    fn a_maximum_distance_of_zero_is_refused() {
        let exporter = crate::MemoryExporter {
            out_objects: Default::default(),
        };
        let mut ctx = DatasetContext {
            store: exporter.dataset_store().unwrap(),
            progress: None,
            cancel: Arc::new(AtomicBool::new(false)),
        };
        let tracker = SimpleTracker {
            object_class: ObjectClass::Valid(1),
            unit: SizeUnits::Pixels,
            max_distance: 0.0,
        };
        let err = tracker.execute(&mut ctx).unwrap_err().to_string();
        assert!(err.contains("greater than 0"), "{err}");
    }
}
