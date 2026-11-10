//! # Measure Object Sholl
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-11-08
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.
//!
//! ## Overview
//! Sholl analysis (Sholl 1953) of a skeleton (Skeletonize Objects):
//! concentric circles around a centre (the soma) at a fixed step, and per
//! circle the number of times the skeleton crosses it - how much a neuron
//! branches at which distance. The crossings per circle are a profile (a
//! value per radius); the summary values follow Ferreira et al. 2014:
//! the most crossings, the radius they are at (critical radius), the
//! largest radius still crossed (enclosing radius), the mean crossings and
//! the Sholl regression coefficient (semi-log method).
//!
//! The crossings are counted on the skeleton as a line, not by pixels in a
//! ring: a circle of radius r crosses each step between two neighbouring
//! skeleton pixels p, q with d(p) < r ≤ d(q) (d: the distance to the
//! centre) - a 1-pixel ring could be jumped by a diagonal step. Steps are
//! taken as in Measure Object Skeleton (a diagonal bridged by a
//! 4-neighbour isn't one).

use crate::algos::measure::object_skeleton::bridged;
use crate::algos::object::skeletonize_objects::{Grid, one_pixel_wide, thin};
use crate::algos::{ExecutionScope, GlobalPipelineCache, ImageAlgorithm, PipelineContext};
use crate::object::Object;
use crate::pipeline::pipeline_measure::{ObjectValue, Outputs};
use evanalyzer_cfg::core_types::{
    CitationMetadata, InternalErrors, MeasurementInfo, MeasurementUnit, ObjectClass, SizeUnits,
};
use macros::CommandsMeta;
use rayon::prelude::*;

/// Sholl analysis of skeletons: circles around a centre (the soma) at a
/// fixed step, the skeleton's crossings per circle (a profile - the curve
/// over the radius) and its summary: the most crossings, the critical and
/// the enclosing radius, the mean crossings and the Sholl regression
/// coefficient. Radii in pixels.
///
/// [Skeletonize Objects] -> [Relate Objects (skeletons to somata)] -> [Measure Object Sholl]
#[derive(CommandsMeta)]
#[cmdsmeta(category = "measure", display_name = "Measure Object Sholl")]
pub struct MeasureObjectSholl {
    /// The step's ID, unique in the project - its values' keys start with
    /// it ("sholl_01/...").
    #[cmdsmeta(measurement_id = "sholl", display_name = "ID", summary = true)]
    pub id: String,

    /// The skeletons measured (the skeleton class of Skeletonize Objects).
    #[cmdsmeta(summary = true)]
    pub objects: ObjectClass,

    /// The class of the centres (e.g. the somata), linked to the skeletons
    /// as their parents by Relate Objects: the circles are around its
    /// centroid. Unset: around the skeleton's own centroid.
    #[cmdsmeta(default = ObjectClass::Unset, summary = true)]
    pub centre_from: ObjectClass,

    /// The distance between two circles.
    #[cmdsmeta(default = 10.0, min = 0.1, max = 100000.0, step = 1.0, summary = true)]
    pub ring_step: f32,

    /// The radius of the first circle. 0: one step.
    #[cmdsmeta(default = 0.0, min = 0.0, max = 100000.0, step = 1.0)]
    pub first_radius: f32,

    /// The radius of the last circle. 0: up to the farthest skeleton pixel.
    /// Set it to compare neurons at the same radii.
    #[cmdsmeta(default = 0.0, min = 0.0, max = 100000.0, step = 1.0)]
    pub last_radius: f32,

    /// Unit of the radii and the step.
    #[cmdsmeta(default = SizeUnits::Pixels)]
    pub unit: SizeUnits,
}

/// The summary values: (id, name, unit).
const SUMMARY: [(&str, &str, MeasurementUnit); 5] = [
    (
        "max_intersections",
        "Max intersections",
        MeasurementUnit::Count,
    ),
    (
        "critical_radius",
        "Critical radius",
        MeasurementUnit::Pixels,
    ),
    (
        "enclosing_radius",
        "Enclosing radius",
        MeasurementUnit::Pixels,
    ),
    (
        "mean_intersections",
        "Mean intersections",
        MeasurementUnit::Count,
    ),
    (
        "regression_coefficient",
        "Sholl regression coefficient",
        MeasurementUnit::None,
    ),
];

/// The profile's id: the crossings per circle.
const PROFILE: &str = "intersections";

/// The circles in pixels: first radius, step, last radius (`None`: up to
/// the farthest skeleton pixel).
#[derive(Clone, Copy, Debug)]
struct Rings {
    first: f64,
    step: f64,
    last: Option<f64>,
}

impl ImageAlgorithm for MeasureObjectSholl {
    fn execute(
        &self,
        ctx: &mut PipelineContext,
        cache: &mut GlobalPipelineCache,
    ) -> Result<(), InternalErrors> {
        if self.objects == ObjectClass::Unset {
            return Err(InternalErrors::InvalidArgument(
                "Measure object Sholl: choose the skeletons".into(),
            ));
        }
        let rings = self.rings(ctx);
        let objects: Vec<_> = cache
            .object_cache
            .iter()
            .filter(|(_, object)| object.is_of(&self.objects, ctx.scope))
            .collect();
        let centre_from = self.centre_from;
        // Each skeleton on its own: in parallel.
        // Each skeleton on its own: in parallel - with the parent its
        // circles are around (`None`: its own centroid).
        let results: Vec<(Sholl, Option<_>)> = objects
            .par_iter()
            .map(|(_, object)| {
                let parent = (centre_from != ObjectClass::Unset)
                    .then(|| object.parent(&centre_from))
                    .flatten()
                    .and_then(|id| cache.object_cache.get(id).map(|parent| (id, parent)));
                let centre = parent.map_or(*object, |(_, p)| p).get_centroid();
                (
                    sholl(object, (centre.0 as f64, centre.1 as f64), rings),
                    parent.map(|(id, _)| id.clone()),
                )
            })
            .collect();
        let ids: Vec<_> = objects.into_iter().map(|(id, _)| id.clone()).collect();

        // The profile's values name the centre (the results draw the
        // circles around it).
        let mut profile = Vec::new();
        for (id, (result, centre)) in ids.iter().zip(&results) {
            for &(radius, crossings) in &result.profile {
                profile.push(ObjectValue {
                    object: id.clone(),
                    value: crossings as f64,
                    related: centre.clone(),
                    position: Some(radius),
                });
            }
        }
        let results: Vec<Sholl> = results.into_iter().map(|(result, _)| result).collect();
        cache.add_object_measurement(self.measurement_output(PROFILE), profile);
        for (i, (id, ..)) in SUMMARY.iter().enumerate() {
            let values: Vec<ObjectValue> = ids
                .iter()
                .zip(&results)
                .map(|(object, result)| ObjectValue {
                    object: object.clone(),
                    value: result.summary[i],
                    related: None,
                    position: None,
                })
                .collect();
            cache.add_object_measurement(self.measurement_output(id), values);
        }
        Ok(())
    }

    fn name(&self) -> &'static str {
        "Measure Object Sholl"
    }

    fn cite(&self) -> Vec<&'static CitationMetadata> {
        vec![&SHOLL_1953, &FERREIRA_2014]
    }

    fn execution_scope(&self) -> ExecutionScope {
        ExecutionScope::WholeImage
    }

    fn measures(&self) -> bool {
        true
    }

    fn measurement_outputs(&self) -> Vec<MeasurementInfo> {
        let outputs = Outputs::of(&self.id);
        std::iter::once(
            outputs
                .object(PROFILE, "Intersections", MeasurementUnit::Count)
                .profile(),
        )
        .map(|info| match self.centre_from {
            // Its values name the centre object.
            ObjectClass::Unset => info,
            _ => info.naming_objects(),
        })
        .chain(
            SUMMARY
                .iter()
                .map(|(id, name, unit)| outputs.object(id, name, *unit)),
        )
        .collect()
    }
}

static SHOLL_1953: CitationMetadata = CitationMetadata {
    cite_key: "sholl1953dendritic",
    title: "Dendritic Organization in the Neurons of the Visual and Motor Cortices of the Cat",
    authors: &["D. A. Sholl"],
    year: 1953,
    container: Some("Journal of Anatomy"),
    doi: None,
    url: Some("https://www.ncbi.nlm.nih.gov/pmc/articles/PMC1244622/"),
    pages: Some("387-406"),
};

static FERREIRA_2014: CitationMetadata = CitationMetadata {
    cite_key: "ferreira2014neuronal",
    title: "Neuronal Morphometry Directly from Bitmap Images",
    authors: &[
        "Tiago A. Ferreira",
        "Arne V. Blackman",
        "Julia Oyrer",
        "Shiva Jayabal",
        "Andrew J. Chung",
        "Alanna J. Watt",
        "P. Jesper Sjöström",
        "Donald J. van Meyel",
    ],
    year: 2014,
    container: Some("Nature Methods"),
    doi: Some("10.1038/nmeth.3125"),
    url: Some("https://doi.org/10.1038/nmeth.3125"),
    pages: Some("982-984"),
};

impl MeasureObjectSholl {
    /// The circles in pixels; without a pixel size nanometres are taken as
    /// pixels.
    fn rings(&self, ctx: &PipelineContext) -> Rings {
        let sizes = ctx.pixel_sizes();
        let pixel_size_nm = (sizes.px_size_x + sizes.px_size_y) / 2.0;
        let px = |v: f32| match self.unit {
            SizeUnits::NanoMeter if pixel_size_nm > 0.0 => (v / pixel_size_nm) as f64,
            _ => v as f64,
        };
        let step = px(self.ring_step).max(1e-3);
        Rings {
            first: if self.first_radius > 0.0 {
                px(self.first_radius)
            } else {
                step
            },
            step,
            last: (self.last_radius > 0.0).then(|| px(self.last_radius)),
        }
    }
}

/// One skeleton's Sholl analysis: (radius, crossings) per circle and the
/// summary values in [`SUMMARY`] order (NaN where there is none).
struct Sholl {
    profile: Vec<(f64, u32)>,
    summary: [f64; SUMMARY.len()],
}

/// The circles of `rings` around `centre` (image coordinates) and the
/// skeleton's crossings of each.
fn sholl(skeleton: &Object, centre: (f64, f64), rings: Rings) -> Sholl {
    let mut grid = Grid::of(skeleton);
    thin(&mut grid);
    one_pixel_wide(&mut grid);
    let [x0, y0, _, _] = skeleton.bbox;
    let width = grid.width;
    let distance = |p: usize| {
        let x = x0 as f64 + (p % width) as f64 - 1.0;
        let y = y0 as f64 + (p / width) as f64 - 1.0;
        ((x - centre.0).powi(2) + (y - centre.1).powi(2)).sqrt()
    };

    // Every step between two skeleton pixels once: its nearer and farther
    // distance.
    let pixels = grid.ones();
    let mut steps: Vec<(f64, f64)> = Vec::new();
    for &p in &pixels {
        for k in 0..8 {
            let q = grid.neighbour(p, k);
            if q > p && grid.get(q) && !bridged(&grid, p, k) {
                let (a, b) = (distance(p), distance(q));
                steps.push((a.min(b), a.max(b)));
            }
        }
    }
    let farthest = pixels.iter().map(|&p| distance(p)).fold(0.0f64, f64::max);
    let last = rings.last.unwrap_or(farthest);

    // Crossings of the circle r: steps with near < r <= far.
    let mut profile = Vec::new();
    let mut i = 0usize;
    loop {
        let r = rings.first + i as f64 * rings.step;
        // A little tolerance, so a last radius on the grid of the step counts.
        if r > last + 1e-9 {
            break;
        }
        let crossings = steps
            .iter()
            .filter(|&&(near, far)| near < r && r <= far)
            .count() as u32;
        profile.push((r, crossings));
        i += 1;
    }
    Sholl {
        summary: summary(&profile),
        profile,
    }
}

/// The summary values of a profile (Ferreira et al. 2014): the most
/// crossings and the first radius with them, the largest radius still
/// crossed, the mean crossings up to it, and the Sholl regression
/// coefficient k of the semi-log method - log(N(r) / πr²) = -k·r + m,
/// fitted by least squares over the circles with crossings (NaN with fewer
/// than two).
fn summary(profile: &[(f64, u32)]) -> [f64; SUMMARY.len()] {
    let crossed: Vec<(f64, u32)> = profile.iter().copied().filter(|&(_, n)| n > 0).collect();
    let Some(&(enclosing, _)) = crossed.last() else {
        return [0.0, f64::NAN, f64::NAN, 0.0, f64::NAN];
    };
    let (mut critical, mut max) = (f64::NAN, 0u32);
    for &(r, n) in profile {
        if n > max {
            (critical, max) = (r, n);
        }
    }
    let within: Vec<u32> = profile
        .iter()
        .filter(|&&(r, _)| r <= enclosing)
        .map(|&(_, n)| n)
        .collect();
    let mean = within.iter().sum::<u32>() as f64 / within.len() as f64;
    let points: Vec<(f64, f64)> = crossed
        .iter()
        .map(|&(r, n)| (r, (n as f64 / (std::f64::consts::PI * r * r)).ln()))
        .collect();
    let k = slope(&points).map_or(f64::NAN, |s| -s);
    [max as f64, critical, enclosing, mean, k]
}

/// The least-squares slope of `points`; `None` with fewer than two
/// different x.
fn slope(points: &[(f64, f64)]) -> Option<f64> {
    let n = points.len() as f64;
    if points.len() < 2 {
        return None;
    }
    let (mx, my) = points
        .iter()
        .fold((0.0, 0.0), |(a, b), &(x, y)| (a + x / n, b + y / n));
    let sxx: f64 = points.iter().map(|&(x, _)| (x - mx).powi(2)).sum();
    let sxy: f64 = points.iter().map(|&(x, y)| (x - mx) * (y - my)).sum();
    (sxx > 0.0).then(|| sxy / sxx)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::object::ObjectInit;
    use crate::pipeline::pipeline::PipelineImageMeta;
    use crate::pipeline::pipeline_measure::MeasurementValues;
    use crate::pipeline::scope::Scope;
    use bitvec::prelude::*;
    use evanalyzer_cfg::core_types::ObjectId;
    use std::collections::HashSet;

    /// An object of `pixels` (image coordinates).
    fn object(id: u128, pixels: &HashSet<(u32, u32)>) -> Object {
        let x0 = pixels.iter().map(|p| p.0).min().unwrap();
        let y0 = pixels.iter().map(|p| p.1).min().unwrap();
        let x1 = pixels.iter().map(|p| p.0).max().unwrap();
        let y1 = pixels.iter().map(|p| p.1).max().unwrap();
        let width = (x1 - x0 + 1) as usize;
        let mut mask = BitVec::<u64, Lsb0>::repeat(false, width * (y1 - y0 + 1) as usize);
        let (mut sx, mut sy) = (0u64, 0u64);
        for &(x, y) in pixels {
            mask.set((y - y0) as usize * width + (x - x0) as usize, true);
            sx += x as u64;
            sy += y as u64;
        }
        Object::new(ObjectInit {
            id: ObjectId(id),
            bbox: [x0, y0, x1, y1],
            area: pixels.len(),
            mask_data: mask,
            sum_x: sx,
            sum_y: sy,
            ..Default::default()
        })
    }

    /// A star of `arms` straight 1-pixel arms of `length` px from (50, 50):
    /// right, down, left, up, then the diagonals.
    fn star(arms: usize, length: u32) -> HashSet<(u32, u32)> {
        let dirs: [(i32, i32); 8] = [
            (1, 0),
            (0, 1),
            (-1, 0),
            (0, -1),
            (1, 1),
            (-1, 1),
            (-1, -1),
            (1, -1),
        ];
        let mut pixels = HashSet::from([(50, 50)]);
        for &(dx, dy) in &dirs[..arms] {
            for i in 1..=length as i32 {
                pixels.insert(((50 + dx * i) as u32, (50 + dy * i) as u32));
            }
        }
        pixels
    }

    fn rings(step: f64, last: Option<f64>) -> Rings {
        Rings {
            first: step,
            step,
            last,
        }
    }

    #[test]
    fn a_star_crosses_each_circle_once_per_arm() {
        let result = sholl(&object(1, &star(4, 30)), (50.0, 50.0), rings(5.0, None));
        let radii: Vec<f64> = result.profile.iter().map(|p| p.0).collect();
        assert_eq!(radii, [5.0, 10.0, 15.0, 20.0, 25.0, 30.0]);
        assert!(
            result.profile.iter().all(|&(_, n)| n == 4),
            "{:?}",
            result.profile
        );
        assert_eq!(result.summary[0], 4.0); // max
        assert_eq!(result.summary[1], 5.0); // critical: the first with 4
        assert_eq!(result.summary[2], 30.0); // enclosing
        assert_eq!(result.summary[3], 4.0); // mean
    }

    #[test]
    fn diagonal_arms_are_counted_too() {
        // Diagonal steps jump a 1-pixel ring: counting on the line finds them.
        let result = sholl(&object(1, &star(8, 20)), (50.0, 50.0), rings(3.0, None));
        for &(r, n) in &result.profile {
            let expected = if r <= 20.0 { 8 } else { 4 };
            assert_eq!(n, expected, "radius {r}: {:?}", result.profile);
        }
    }

    #[test]
    fn a_last_radius_adds_empty_circles() {
        let result = sholl(
            &object(1, &star(2, 10)),
            (50.0, 50.0),
            rings(5.0, Some(25.0)),
        );
        assert_eq!(
            result.profile,
            [(5.0, 2), (10.0, 2), (15.0, 0), (20.0, 0), (25.0, 0)]
        );
        // The enclosing radius is the last one crossed, the mean up to it.
        assert_eq!(result.summary[2], 10.0);
        assert_eq!(result.summary[3], 2.0);
    }

    /// A tree around (50, 100): a trunk to (70, 100), two diagonal branches
    /// to (90, 80) and (90, 120) (√800 ≈ 28 long), each splitting into one
    /// going right to x 110 and one going on diagonally to x 110. Circles:
    /// r 10 the trunk (1), r 30 the two branches (2), r 50 and 60 the four
    /// arms (4; the right ones end at √4000 ≈ 63), r 70 the two diagonal
    /// ones (their ends at √5200 ≈ 72).
    #[test]
    fn a_branching_tree_has_more_crossings_farther_out() {
        let mut pixels = HashSet::new();
        for x in 50..=70 {
            pixels.insert((x, 100));
        }
        for i in 0..=20u32 {
            pixels.insert((70 + i, 100 - i));
            pixels.insert((70 + i, 100 + i));
            pixels.insert((90 + i, 80));
            pixels.insert((90 + i, 120));
            pixels.insert((90 + i, 80 - i));
            pixels.insert((90 + i, 120 + i));
        }
        let result = sholl(&object(1, &pixels), (50.0, 100.0), rings(10.0, None));
        let at = |r: f64| result.profile.iter().find(|p| p.0 == r).unwrap().1;
        let found: Vec<u32> = [10.0, 30.0, 50.0, 60.0, 70.0].map(at).to_vec();
        assert_eq!(found, [1, 2, 4, 4, 2], "{:?}", result.profile);
        assert_eq!(result.summary[0], 4.0);
        assert_eq!(result.summary[1], 50.0);
        assert_eq!(result.summary[2], 70.0);
    }

    #[test]
    fn the_regression_coefficient_comes_from_the_semi_log_fit() {
        // N(r) = πr²·e^(-0.1·r) exactly gives k = 0.1.
        let profile: Vec<(f64, u32)> = (1..=5).map(|i| (i as f64 * 10.0, 0)).collect();
        let points: Vec<(f64, f64)> = profile.iter().map(|&(r, _)| (r, (-0.1 * r))).collect();
        assert!((slope(&points).unwrap() + 0.1).abs() < 1e-12);
        // An empty profile: nothing crossed.
        let empty = summary(&profile);
        assert_eq!(empty[0], 0.0);
        assert!(empty[2].is_nan());
        let result = sholl(&object(1, &star(4, 30)), (50.0, 50.0), rings(5.0, None));
        // Constant N: ln(N/πr²) falls with r - k > 0.
        assert!(result.summary[4] > 0.0, "{}", result.summary[4]);
    }

    #[test]
    fn the_step_reports_a_profile_and_the_summary_around_the_parent() {
        let mut skeleton = object(1, &star(4, 20));
        skeleton
            .add_object_class(ObjectClass::Global(1), Scope::Global)
            .unwrap();
        // The soma, off the skeleton's own centroid: (50, 50) - the arms'
        // centre - is where the circles must be.
        let soma = object(2, &HashSet::from([(50, 50)]));
        skeleton.set_parent(ObjectClass::Global(2), Some(ObjectId(2)));
        let mut cache = GlobalPipelineCache::default();
        cache.object_cache.insert(ObjectId(1), skeleton);
        cache.object_cache.insert(ObjectId(2), soma);
        let step = MeasureObjectSholl {
            id: "sh".into(),
            objects: ObjectClass::Global(1),
            centre_from: ObjectClass::Global(2),
            ring_step: 10.0,
            first_radius: 0.0,
            last_radius: 0.0,
            unit: SizeUnits::Pixels,
        };
        let mut ctx =
            PipelineContext::without_image(std::path::PathBuf::new(), PipelineImageMeta::default())
                .unwrap();
        step.execute(&mut ctx, &mut cache).unwrap();
        let measurements = &cache.measurements.measurements;
        assert_eq!(measurements.len(), 1 + SUMMARY.len());
        let profile = measurements
            .iter()
            .find(|m| m.info.key == "sh/intersections")
            .unwrap();
        assert!(profile.info.profile);
        let MeasurementValues::Objects(values) = &profile.values else {
            panic!()
        };
        let rows: Vec<(Option<f64>, f64)> = values.iter().map(|v| (v.position, v.value)).collect();
        assert_eq!(rows, [(Some(10.0), 4.0), (Some(20.0), 4.0)]);
        // Each value names the soma the circles are around.
        assert!(profile.info.related_objects);
        assert!(values.iter().all(|v| v.related == Some(ObjectId(2))));
        let max = measurements
            .iter()
            .find(|m| m.info.key == "sh/max_intersections")
            .unwrap();
        assert!(!max.info.profile);
    }

    #[test]
    fn without_a_centre_object_the_values_name_none() {
        let mut skeleton = object(1, &star(4, 20));
        skeleton
            .add_object_class(ObjectClass::Global(1), Scope::Global)
            .unwrap();
        // Centre From set, but this skeleton has no such parent.
        let mut cache = GlobalPipelineCache::default();
        cache.object_cache.insert(ObjectId(1), skeleton);
        let mut step = MeasureObjectSholl {
            id: "sh".into(),
            objects: ObjectClass::Global(1),
            centre_from: ObjectClass::Global(2),
            ring_step: 10.0,
            first_radius: 0.0,
            last_radius: 0.0,
            unit: SizeUnits::Pixels,
        };
        let mut ctx =
            PipelineContext::without_image(std::path::PathBuf::new(), PipelineImageMeta::default())
                .unwrap();
        step.execute(&mut ctx, &mut cache).unwrap();
        let profile = cache
            .measurements
            .measurements
            .iter()
            .find(|m| m.info.key == "sh/intersections")
            .unwrap();
        let MeasurementValues::Objects(values) = &profile.values else {
            panic!()
        };
        assert!(!values.is_empty());
        assert!(values.iter().all(|v| v.related.is_none()));
        // No Centre From: the values name no object at all.
        step.centre_from = ObjectClass::Unset;
        assert!(!step.measurement_outputs()[0].related_objects);
    }

    #[test]
    fn nanometre_rings_use_the_pixel_size() {
        let mut ctx =
            PipelineContext::without_image(std::path::PathBuf::new(), PipelineImageMeta::default())
                .unwrap();
        ctx.image_meta.pixel_sizes.px_size_x = 500.0;
        ctx.image_meta.pixel_sizes.px_size_y = 500.0;
        let step = MeasureObjectSholl {
            id: "sh".into(),
            objects: ObjectClass::Global(1),
            centre_from: ObjectClass::Unset,
            ring_step: 5000.0,
            first_radius: 0.0,
            last_radius: 50000.0,
            unit: SizeUnits::NanoMeter,
        };
        let rings = step.rings(&ctx);
        assert_eq!(
            (rings.first, rings.step, rings.last),
            (10.0, 10.0, Some(100.0))
        );
    }
}
