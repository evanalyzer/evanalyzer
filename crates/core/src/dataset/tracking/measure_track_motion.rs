//! # Measure Track Motion
//!
//! How each track of a tracking step moved: path, displacement and how far
//! it got from its start, the speeds and ratios of computer-assisted sperm
//! analysis (VCL, VSL, VAP, LIN, STR, WOB, ALH, BCF), its turning (one way
//! when circling) and its mean squared displacement (α, D). Optionally
//! after taking away the drift of the image (stage or flow) found from
//! reference tracks. See [`crate::dataset::libs::track_motion`].
//!
//! The values are stored per track (`measurements.region = 'track'` with
//! its `track_id`), the drift per image and time point - the results only
//! show them.
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-11-09
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.

use crate::dataset::libs::linking;
use crate::dataset::libs::track_motion::{self, MotionOptions, TrackMotion, TrackPoint, drift_at};
use crate::dataset::{DatasetAlgorithm, DatasetContext};
use crate::pipeline::pipeline_measure::Outputs;
use crate::storage::stored_object::{ObjectFilter, StoredObject};
use crate::storage::{DatasetValue, DatasetValueAbout};
use evanalyzer_cfg::core_types::{
    CitationMetadata, InternalErrors, MeasurementInfo, MeasurementUnit, ObjectClass, SizeUnits,
};
use macros::CommandsMeta;
use std::collections::BTreeMap;

/// How each track moved: path, displacement, the CASA speeds and ratios
/// (VCL, VSL, VAP, LIN, STR, WOB, ALH, BCF), turning angle and mean
/// squared displacement (α, D) - per track, in pixels and time points.
///
/// [Track LAP / Kalman / ...] -> [Measure Track Motion]
///
/// With a drift reference (objects that don't swim, e.g. immotile cells or
/// beads, tracked too) the drift of the image is taken away first: the
/// median step of the reference tracks from one time point to the next.
#[derive(CommandsMeta)]
#[cmdsmeta(category = "dataset", display_name = "Measure Track Motion")]
pub struct MeasureTrackMotion {
    /// The step's ID, unique in the project - its values' keys start with
    /// it ("motion_01/...").
    #[cmdsmeta(measurement_id = "motion", display_name = "ID", summary = true)]
    pub id: String,

    /// The tracked objects measured (a tracking step must come before).
    #[cmdsmeta(summary = true)]
    pub objects: ObjectClass,

    /// Steps shorter than this are joined with the next ones before the
    /// turning angle is taken - jitter of the centre isn't turning. The
    /// path and speeds use every step.
    #[cmdsmeta(default = 1.0, min = 0.0, max = 100000.0, step = 0.5)]
    pub min_step: f32,

    /// Unit of the minimum step.
    #[cmdsmeta(default = SizeUnits::Pixels)]
    pub unit: SizeUnits,

    /// The average path (for VAP, STR, WOB, ALH, BCF): each position is
    /// the mean of the positions this many time points before and after it.
    #[cmdsmeta(default = 2, min = 0, max = 100, summary = true)]
    pub average_over: usize,

    /// Objects that don't move by themselves (e.g. immotile cells, beads),
    /// tracked too: their median step is the image's drift and is taken
    /// away first. Unset: no drift correction.
    #[cmdsmeta(default = ObjectClass::Unset, summary = true)]
    pub drift_reference: ObjectClass,
}

/// The per track values: (id, name, unit, range) - in the order of
/// [`values`].
const TRACK_VALUES: [(&str, &str, MeasurementUnit, Option<(f64, f64)>); 16] = [
    ("path_length", "Path length", MeasurementUnit::Pixels, None),
    (
        "displacement",
        "Displacement",
        MeasurementUnit::Pixels,
        None,
    ),
    (
        "max_distance",
        "Max distance from start",
        MeasurementUnit::Pixels,
        None,
    ),
    (
        "vcl",
        "VCL (curvilinear speed)",
        MeasurementUnit::PixelsPerFrame,
        None,
    ),
    (
        "vsl",
        "VSL (straight-line speed)",
        MeasurementUnit::PixelsPerFrame,
        None,
    ),
    (
        "vap",
        "VAP (average path speed)",
        MeasurementUnit::PixelsPerFrame,
        None,
    ),
    (
        "lin",
        "LIN (linearity)",
        MeasurementUnit::Fraction,
        Some((0.0, 1.0)),
    ),
    ("str", "STR (straightness)", MeasurementUnit::Fraction, None),
    ("wob", "WOB (wobble)", MeasurementUnit::Fraction, None),
    (
        "alh",
        "ALH (lateral head amplitude)",
        MeasurementUnit::Pixels,
        None,
    ),
    (
        "bcf",
        "BCF (beat-cross frequency)",
        MeasurementUnit::PerFrame,
        None,
    ),
    (
        "turning",
        "Turning angle (mean)",
        MeasurementUnit::Degrees,
        Some((-180.0, 180.0)),
    ),
    (
        "turning_abs",
        "Turning angle (mean absolute)",
        MeasurementUnit::Degrees,
        Some((0.0, 180.0)),
    ),
    (
        "angular_speed",
        "Angular speed",
        MeasurementUnit::DegreesPerFrame,
        None,
    ),
    ("msd_alpha", "MSD alpha", MeasurementUnit::None, None),
    (
        "msd_d",
        "MSD D",
        MeasurementUnit::SquarePixelsPerFrame,
        None,
    ),
];

/// A track's values in the order of [`TRACK_VALUES`].
fn values(m: &TrackMotion) -> [f64; 16] {
    [
        m.path_length,
        m.displacement,
        m.max_distance,
        m.vcl,
        m.vsl,
        m.vap,
        m.lin,
        m.str_,
        m.wob,
        m.alh,
        m.bcf,
        m.turning_mean,
        m.turning_abs_mean,
        m.angular_speed,
        m.msd_alpha,
        m.msd_d,
    ]
}

/// The drift values, after the track values (with a drift reference).
const DRIFT_VALUES: [(&str, &str); 2] = [("drift_x", "Drift X"), ("drift_y", "Drift Y")];

pub(crate) const WHO_2021: CitationMetadata = CitationMetadata {
    cite_key: "who2021semen",
    title: "WHO laboratory manual for the examination and processing of human semen, sixth edition",
    authors: &["World Health Organization"],
    year: 2021,
    container: Some("World Health Organization, Geneva"),
    doi: None,
    url: None,
    pages: None,
};

impl DatasetAlgorithm for MeasureTrackMotion {
    fn execute(&self, ctx: &mut DatasetContext) -> Result<(), InternalErrors> {
        if self.objects == ObjectClass::Unset {
            return Err(InternalErrors::InvalidArgument(
                "Measure Track Motion: choose the tracked objects".into(),
            ));
        }
        let infos = self.measurement_outputs();
        let mut out = Vec::new();
        for image in ctx.store.image_rel_paths(self.objects)? {
            if ctx.is_cancelled() {
                return Err(InternalErrors::Cancelled);
            }
            let objects = ctx.store.get_objects(&ObjectFilter {
                image_rel_path: Some(image.clone()),
                object_class: Some(self.objects),
                ..Default::default()
            })?;
            let Some(first) = objects.first() else {
                continue;
            };
            let options = MotionOptions {
                min_step_px: linking::to_px(self.min_step, self.unit, first, self.name())?,
                average_half_width: self.average_over,
            };
            let drifts = match self.drift_reference {
                ObjectClass::Unset => BTreeMap::new(),
                reference => {
                    let reference = ctx.store.get_objects(&ObjectFilter {
                        image_rel_path: Some(image.clone()),
                        object_class: Some(reference),
                        ..Default::default()
                    })?;
                    tracks(&reference)
                        .into_iter()
                        .fold(
                            BTreeMap::<i32, Vec<_>>::new(),
                            |mut by_z, ((z, _), points)| {
                                by_z.entry(z).or_default().push(points);
                                by_z
                            },
                        )
                        .into_iter()
                        .map(|(z, tracks)| (z, track_motion::drift(&tracks)))
                        .collect()
                }
            };
            for (z, drift) in &drifts {
                for p in drift {
                    for (k, value) in [p.x, p.y].into_iter().enumerate() {
                        out.push(DatasetValue {
                            image_rel_path: image.clone(),
                            z_stack: *z,
                            t_stack: p.t,
                            about: DatasetValueAbout::Image,
                            info: TRACK_VALUES.len() + k,
                            value,
                        });
                    }
                }
            }
            for ((z, track_id), mut points) in tracks(&objects) {
                if let Some(drift) = drifts.get(&z) {
                    for p in &mut points {
                        let (dx, dy) = drift_at(drift, p.t);
                        p.x -= dx;
                        p.y -= dy;
                    }
                }
                let t_first = points[0].t;
                let motion = track_motion::measure(&points, &options);
                for (k, value) in values(&motion).into_iter().enumerate() {
                    // A value it can't have (one time point, too few
                    // steps) is no row.
                    if value.is_finite() {
                        out.push(DatasetValue {
                            image_rel_path: image.clone(),
                            z_stack: z,
                            t_stack: t_first,
                            about: DatasetValueAbout::Track {
                                track_id,
                                class: self.objects,
                            },
                            info: k,
                            value,
                        });
                    }
                }
            }
        }
        // One write: the step's rows of a run before are replaced.
        ctx.store.write_measurements(&self.id, &infos, &out)
    }

    fn name(&self) -> &'static str {
        "Measure Track Motion"
    }

    fn cite(&self) -> Vec<&'static CitationMetadata> {
        vec![&WHO_2021]
    }

    fn measurement_outputs(&self) -> Vec<MeasurementInfo> {
        let outputs = Outputs::of(&self.id);
        let mut infos: Vec<MeasurementInfo> = TRACK_VALUES
            .iter()
            .map(|(id, name, unit, range)| {
                let info = outputs.track(id, name, *unit);
                match range {
                    Some((min, max)) => info.range(*min, *max),
                    None => info,
                }
            })
            .collect();
        if self.drift_reference != ObjectClass::Unset {
            // Per time point: how far the image moved since the first -
            // no value to average over images.
            infos.extend(DRIFT_VALUES.iter().map(|(id, name)| MeasurementInfo {
                aggregatable: false,
                ..outputs.image(id, name, MeasurementUnit::Pixels)
            }));
        }
        infos
    }
}

/// The tracked objects' tracks by z-plane and track id, each by time - one
/// point per time point (the first object if a track had two).
fn tracks(objects: &[StoredObject]) -> BTreeMap<(i32, u64), Vec<TrackPoint>> {
    let mut tracks: BTreeMap<(i32, u64), Vec<TrackPoint>> = BTreeMap::new();
    for object in objects {
        let Some(track_id) = object.track_id.filter(|id| *id != 0) else {
            continue;
        };
        tracks
            .entry((object.z_stack, track_id))
            .or_default()
            .push(TrackPoint {
                t: object.t_stack,
                x: object.centroid_px.0,
                y: object.centroid_px.1,
            });
    }
    for points in tracks.values_mut() {
        points.sort_by_key(|p| p.t);
        points.dedup_by_key(|p| p.t);
    }
    tracks
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::MemoryExporter;
    use crate::storage::DatasetStore;
    use crate::storage::stored_object::TrackAssignment;
    use evanalyzer_cfg::core_types::{ObjectId, TrackId};
    use evanalyzer_cfg::settings::object_settings::ObjectMetricSettings;
    use std::sync::Arc;
    use std::sync::atomic::AtomicBool;

    const SPOTS: ObjectClass = ObjectClass::Global(1);
    const BEADS: ObjectClass = ObjectClass::Global(2);

    fn step(drift_reference: ObjectClass) -> MeasureTrackMotion {
        MeasureTrackMotion {
            id: "motion_01".into(),
            objects: SPOTS,
            min_step: 0.0,
            unit: SizeUnits::Pixels,
            average_over: 1,
            drift_reference,
        }
    }

    /// An object of `class` on track `track` at (t, x, y).
    fn object(
        id: u128,
        class: ObjectClass,
        track: u64,
        t: i32,
        x: f64,
        y: f64,
    ) -> ObjectMetricSettings {
        let mut o = ObjectMetricSettings::default();
        o.id = ObjectId(id);
        o.object_class = [class].into();
        o.t_stack = t;
        // The store takes the centre of the box: a 1-pixel box at (x, y).
        o.bbox = [x as u32, y as u32, x as u32, y as u32];
        o.track.id = TrackId(track);
        o
    }

    fn run(step: &MeasureTrackMotion, objects: Vec<ObjectMetricSettings>) -> MemoryExporter {
        let exporter = MemoryExporter::default();
        *exporter.out_objects.lock().unwrap() = objects;
        let mut ctx = DatasetContext {
            store: &exporter,
            progress: None,
            cancel: Arc::new(AtomicBool::new(false)),
        };
        step.execute(&mut ctx).unwrap();
        exporter
    }

    fn value(exporter: &MemoryExporter, track: u64, id: &str) -> Option<f64> {
        exporter
            .out_dataset_measurements
            .lock()
            .unwrap()
            .iter()
            .find(|(info, v)| {
                info.key == format!("motion_01/{id}")
                    && v.about
                        == DatasetValueAbout::Track {
                            track_id: track,
                            class: SPOTS,
                        }
            })
            .map(|(_, v)| v.value)
    }

    #[test]
    fn each_track_gets_its_values_and_the_class_is_kept() {
        // Track 1: 2 px per time point to the right. Track 2: one time
        // point. A bead: not measured.
        let mut objects: Vec<_> = (0..10)
            .map(|t| object(t as u128 + 1, SPOTS, 1, t, 2.0 * f64::from(t), 0.0))
            .collect();
        objects.push(object(100, SPOTS, 2, 3, 50.0, 50.0));
        objects.push(object(101, BEADS, 3, 0, 9.0, 9.0));
        let exporter = run(&step(ObjectClass::Unset), objects);
        assert_eq!(value(&exporter, 1, "vcl"), Some(2.0));
        assert_eq!(value(&exporter, 1, "lin"), Some(1.0));
        assert_eq!(value(&exporter, 1, "displacement"), Some(18.0));
        // One time point: a path of 0, but no speed (no row).
        assert_eq!(value(&exporter, 2, "path_length"), Some(0.0));
        assert_eq!(value(&exporter, 2, "vcl"), None);
        assert_eq!(value(&exporter, 3, "path_length"), None);
        let rows = exporter.out_dataset_measurements.lock().unwrap();
        let row = rows.iter().find(|(i, _)| i.key == "motion_01/vcl").unwrap();
        assert_eq!(row.1.t_stack, 0, "a track's first time point");
        assert_eq!(row.0.unit, MeasurementUnit::PixelsPerFrame);
        assert!(
            rows.iter().all(|(i, _)| !i.key.contains("drift")),
            "no drift without a reference"
        );
    }

    #[test]
    fn the_drift_of_the_references_is_taken_away() {
        // Everything flows 3 px per time point to the right: the beads, and
        // the spot, which by itself stays put.
        let mut objects = Vec::new();
        for t in 0..12 {
            objects.push(object(
                t as u128 + 1,
                SPOTS,
                1,
                t,
                10.0 + 3.0 * f64::from(t),
                5.0,
            ));
            for b in 0..3u64 {
                objects.push(object(
                    1000 + u128::from(b) * 100 + t as u128,
                    BEADS,
                    10 + b,
                    t,
                    100.0 * (b as f64 + 1.0) + 3.0 * f64::from(t),
                    40.0,
                ));
            }
        }
        let raw = run(&step(ObjectClass::Unset), objects.clone());
        assert_eq!(value(&raw, 1, "vcl"), Some(3.0));
        let corrected = run(&step(BEADS), objects);
        assert_eq!(
            value(&corrected, 1, "vcl"),
            Some(0.0),
            "by itself it stays put"
        );
        assert_eq!(value(&corrected, 1, "max_distance"), Some(0.0));
    }

    #[test]
    fn a_second_run_replaces_the_first() {
        let objects: Vec<_> = (0..5)
            .map(|t| object(t as u128 + 1, SPOTS, 1, t, f64::from(t), 0.0))
            .collect();
        let exporter = run(&step(ObjectClass::Unset), objects);
        let mut ctx = DatasetContext {
            store: &exporter,
            progress: None,
            cancel: Arc::new(AtomicBool::new(false)),
        };
        exporter
            .set_track_ids(&[TrackAssignment {
                object_id: ObjectId(1),
                track_id: 1,
                parent_track_ids: vec![],
            }])
            .unwrap();
        step(ObjectClass::Unset).execute(&mut ctx).unwrap();
        let rows = exporter.out_dataset_measurements.lock().unwrap();
        assert_eq!(
            rows.iter()
                .filter(|(i, _)| i.key == "motion_01/vcl")
                .count(),
            1
        );
    }

    #[test]
    fn unset_objects_are_refused() {
        let mut s = step(ObjectClass::Unset);
        s.objects = ObjectClass::Unset;
        let exporter = MemoryExporter::default();
        let mut ctx = DatasetContext {
            store: &exporter,
            progress: None,
            cancel: Arc::new(AtomicBool::new(false)),
        };
        assert!(s.execute(&mut ctx).is_err());
    }

    #[test]
    fn the_outputs_are_per_track_and_the_drift_per_image() {
        use evanalyzer_cfg::core_types::MeasurementTarget;
        let plain = step(ObjectClass::Unset).measurement_outputs();
        assert_eq!(plain.len(), TRACK_VALUES.len());
        assert!(plain.iter().all(|i| i.target == MeasurementTarget::Track));
        let with_drift = step(BEADS).measurement_outputs();
        assert_eq!(with_drift.len(), TRACK_VALUES.len() + 2);
        assert_eq!(with_drift[16].key, "motion_01/drift_x");
        assert_eq!(with_drift[16].target, MeasurementTarget::Image);
    }
}
