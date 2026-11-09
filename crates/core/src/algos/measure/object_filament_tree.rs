//! # Measure Filament Tree
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-11-09
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.
//!
//! ## Overview
//! The values of traced trees (Trace Filaments): per segment where it sits
//! in its tree - its branch level (1 = starting at the soma, +1 per branch
//! point), its Strahler order (Strahler 1957: 1 at the tips, one more where
//! two or more of the highest order meet), the path distance from the soma
//! to its far end, its length and straightness - and per neuron, on its
//! soma, the totals: length, primary neurites, branch points, tips, the
//! highest branch level and the longest path.
//!
//! The tree is read from the segments' parents (Trace Filaments links each
//! to the next segment towards the soma, and all to their soma). Lengths
//! as in Measure Object Skeleton, in pixels.

use std::collections::HashMap;

use crate::algos::measure::object_skeleton::line_length_and_span;
use crate::algos::{ExecutionScope, GlobalPipelineCache, ImageAlgorithm, PipelineContext};
use crate::pipeline::pipeline_measure::{ObjectValue, Outputs};
use evanalyzer_cfg::core_types::{
    CitationMetadata, InternalErrors, MeasurementInfo, MeasurementUnit, ObjectClass, ObjectId,
};
use macros::CommandsMeta;
use rayon::prelude::*;

/// Per traced segment (Trace Filaments) its branch level, Strahler order,
/// path distance from the soma to its far end, length and straightness;
/// per neuron (on its soma) the total length, the primary neurites, branch
/// points and tips, the highest branch level and the longest path. Lengths
/// in pixels.
///
/// [Trace Filaments] -> [Measure Filament Tree]
#[derive(CommandsMeta)]
#[cmdsmeta(category = "measure", display_name = "Measure Filament Tree")]
pub struct MeasureFilamentTree {
    /// The step's ID, unique in the project - its values' keys start with
    /// it ("tree_01/...").
    #[cmdsmeta(measurement_id = "tree", display_name = "ID", summary = true)]
    pub id: String,

    /// The segments measured (Trace Filaments' output class).
    #[cmdsmeta(summary = true)]
    pub segments: ObjectClass,

    /// The somata (Trace Filaments' seeds): the per-neuron values go on them.
    #[cmdsmeta(summary = true)]
    pub somata: ObjectClass,
}

/// Per segment: (id, name, unit).
const SEGMENT_VALUES: [(&str, &str, MeasurementUnit); 5] = [
    ("branch_level", "Branch level", MeasurementUnit::Count),
    ("strahler_order", "Strahler order", MeasurementUnit::Count),
    (
        "path_distance",
        "Path distance from the soma",
        MeasurementUnit::Pixels,
    ),
    ("length", "Segment length", MeasurementUnit::Pixels),
    (
        "straightness",
        "Segment straightness",
        MeasurementUnit::Fraction,
    ),
];

/// Per neuron (on its soma): (id, name, unit).
const NEURON_VALUES: [(&str, &str, MeasurementUnit); 6] = [
    (
        "total_length",
        "Neuron total length",
        MeasurementUnit::Pixels,
    ),
    (
        "primary_neurites",
        "Neuron primary neurites",
        MeasurementUnit::Count,
    ),
    (
        "branch_points",
        "Neuron branch points",
        MeasurementUnit::Count,
    ),
    ("tips", "Neuron tips", MeasurementUnit::Count),
    (
        "max_branch_level",
        "Neuron highest branch level",
        MeasurementUnit::Count,
    ),
    (
        "longest_path",
        "Neuron longest path",
        MeasurementUnit::Pixels,
    ),
];

/// One segment's place in its tree.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct SegmentValues {
    branch_level: f64,
    strahler_order: f64,
    path_distance: f64,
    length: f64,
    straightness: f64,
}

/// A segment as read: its length, straightness, parent segment (index)
/// and soma.
struct Node {
    length: f64,
    straightness: f64,
    parent: Option<usize>,
    soma: Option<ObjectId>,
}

impl ImageAlgorithm for MeasureFilamentTree {
    fn execute(
        &self,
        ctx: &mut PipelineContext,
        cache: &mut GlobalPipelineCache,
    ) -> Result<(), InternalErrors> {
        if self.segments == ObjectClass::Unset || self.somata == ObjectClass::Unset {
            return Err(InternalErrors::InvalidArgument(
                "Measure filament tree: choose the segments and the somata".into(),
            ));
        }
        let scope = ctx.scope;
        let mut segments: Vec<(&ObjectId, &crate::object::Object)> = cache
            .object_cache
            .iter()
            .filter(|(_, o)| o.is_of(&self.segments, scope))
            .collect();
        segments.sort_by(|a, b| a.1.position_cmp(b.1));
        let index: HashMap<&ObjectId, usize> = segments
            .iter()
            .enumerate()
            .map(|(i, (id, _))| (*id, i))
            .collect();
        // Lengths in parallel (each segment on its own).
        let shapes: Vec<(f64, f64)> = segments
            .par_iter()
            .map(|(_, o)| {
                let (length, span) = line_length_and_span(o);
                let straightness = match span {
                    Some(span) if length > 0.0 => (span / length).min(1.0),
                    _ => f64::NAN,
                };
                (length, straightness)
            })
            .collect();
        let nodes: Vec<Node> = segments
            .iter()
            .zip(&shapes)
            .map(|((_, o), &(length, straightness))| Node {
                length,
                straightness,
                parent: o.parent(&self.segments).and_then(|p| index.get(p).copied()),
                soma: o.parent(&self.somata).cloned(),
            })
            .collect();
        let (values, neurons) = measure_trees(&nodes);
        let ids: Vec<ObjectId> = segments.iter().map(|(id, _)| (*id).clone()).collect();
        let somata: Vec<ObjectId> = {
            let mut somata: Vec<ObjectId> = neurons.keys().cloned().collect();
            somata.sort();
            somata
        };

        let per_segment = |f: fn(&SegmentValues) -> f64| -> Vec<ObjectValue> {
            ids.iter()
                .zip(&values)
                .map(|(id, v)| ObjectValue {
                    object: id.clone(),
                    value: f(v),
                    related: None,
                    position: None,
                })
                .collect()
        };
        let segment_fns: [fn(&SegmentValues) -> f64; 5] = [
            |v| v.branch_level,
            |v| v.strahler_order,
            |v| v.path_distance,
            |v| v.length,
            |v| v.straightness,
        ];
        let mut out: Vec<(MeasurementInfo, Vec<ObjectValue>)> = Vec::new();
        for ((id, ..), f) in SEGMENT_VALUES.iter().zip(segment_fns) {
            out.push((self.measurement_output(id), per_segment(f)));
        }
        for (i, (id, ..)) in NEURON_VALUES.iter().enumerate() {
            let values = somata
                .iter()
                .map(|soma| ObjectValue {
                    object: soma.clone(),
                    value: neurons[soma][i],
                    related: None,
                    position: None,
                })
                .collect();
            out.push((self.measurement_output(id), values));
        }
        for (info, values) in out {
            cache.add_object_measurement(info, values);
        }
        Ok(())
    }

    fn name(&self) -> &'static str {
        "Measure Filament Tree"
    }

    fn cite(&self) -> Vec<&'static CitationMetadata> {
        vec![&CitationMetadata::DANMAYR, &STRAHLER_1957]
    }

    fn execution_scope(&self) -> ExecutionScope {
        ExecutionScope::WholeImage
    }

    fn measures(&self) -> bool {
        true
    }

    fn measurement_outputs(&self) -> Vec<MeasurementInfo> {
        let outputs = Outputs::of(&self.id);
        SEGMENT_VALUES
            .iter()
            .chain(NEURON_VALUES.iter())
            .map(|(id, name, unit)| {
                let info = outputs.object(id, name, *unit);
                if *id == "straightness" {
                    info.range(0.0, 1.0)
                } else {
                    info
                }
            })
            .collect()
    }
}

static STRAHLER_1957: CitationMetadata = CitationMetadata {
    cite_key: "strahler1957quantitative",
    title: "Quantitative Analysis of Watershed Geomorphology",
    authors: &["Arthur N. Strahler"],
    year: 1957,
    container: Some("Eos, Transactions American Geophysical Union"),
    doi: Some("10.1029/TR038i006p00913"),
    url: Some("https://doi.org/10.1029/TR038i006p00913"),
    pages: Some("913-920"),
};

/// The values of every segment (in `nodes` order) and per soma the
/// neuron's values in [`NEURON_VALUES`] order. A parent chain that loops
/// back (not from Trace Filaments) is cut where it would.
fn measure_trees(nodes: &[Node]) -> (Vec<SegmentValues>, HashMap<ObjectId, [f64; 6]>) {
    let n = nodes.len();
    let mut children: Vec<Vec<usize>> = vec![Vec::new(); n];
    for (i, node) in nodes.iter().enumerate() {
        if let Some(p) = node.parent {
            children[p].push(i);
        }
    }
    // Branch level and path distance: down from the roots.
    let mut values = vec![SegmentValues::default(); n];
    let mut seen = vec![false; n];
    let mut stack: Vec<(usize, f64, f64)> = (0..n)
        .filter(|&i| nodes[i].parent.is_none())
        .map(|i| (i, 1.0, 0.0))
        .collect();
    while let Some((i, level, before)) = stack.pop() {
        if seen[i] {
            continue;
        }
        seen[i] = true;
        let distance = before + nodes[i].length;
        values[i] = SegmentValues {
            branch_level: level,
            strahler_order: 0.0,
            path_distance: distance,
            length: nodes[i].length,
            straightness: nodes[i].straightness,
        };
        for &c in &children[i] {
            stack.push((c, level + 1.0, distance));
        }
    }
    // Strahler order: up from the tips (children before their parent).
    let mut order: Vec<usize> = (0..n).filter(|&i| seen[i]).collect();
    order.sort_by(|&a, &b| values[b].branch_level.total_cmp(&values[a].branch_level));
    for &i in &order {
        let orders: Vec<f64> = children[i]
            .iter()
            .map(|&c| values[c].strahler_order)
            .collect();
        let highest = orders.iter().copied().fold(0.0f64, f64::max);
        let at_highest = orders.iter().filter(|&&o| o == highest).count();
        values[i].strahler_order = match (orders.is_empty(), at_highest) {
            (true, _) => 1.0,
            (false, 1) => highest,
            _ => highest + 1.0,
        };
    }
    // A segment not reached from a root (a loop): no values.
    for i in (0..n).filter(|&i| !seen[i]) {
        values[i] = SegmentValues {
            branch_level: f64::NAN,
            strahler_order: f64::NAN,
            path_distance: f64::NAN,
            length: nodes[i].length,
            straightness: nodes[i].straightness,
        };
    }

    let mut neurons: HashMap<ObjectId, [f64; 6]> = HashMap::new();
    for (i, node) in nodes.iter().enumerate() {
        let Some(soma) = &node.soma else {
            continue;
        };
        let entry = neurons.entry(soma.clone()).or_insert([0.0; 6]);
        entry[0] += node.length;
        if node.parent.is_none() {
            entry[1] += 1.0;
        }
        if children[i].len() >= 2 {
            entry[2] += 1.0;
        }
        if children[i].is_empty() {
            entry[3] += 1.0;
        }
        if values[i].branch_level.is_finite() {
            entry[4] = entry[4].max(values[i].branch_level);
            entry[5] = entry[5].max(values[i].path_distance);
        }
    }
    (values, neurons)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn node(length: f64, parent: Option<usize>, soma: u128) -> Node {
        Node {
            length,
            straightness: 1.0,
            parent,
            soma: Some(ObjectId(soma)),
        }
    }

    /// A trunk (0) splitting into two (1, 2); 1 splitting again (3, 4).
    #[test]
    fn levels_orders_and_distances_of_a_tree() {
        let nodes = [
            node(10.0, None, 9),
            node(5.0, Some(0), 9),
            node(7.0, Some(0), 9),
            node(3.0, Some(1), 9),
            node(4.0, Some(1), 9),
        ];
        let (values, neurons) = measure_trees(&nodes);
        let levels: Vec<f64> = values.iter().map(|v| v.branch_level).collect();
        assert_eq!(levels, [1.0, 2.0, 2.0, 3.0, 3.0]);
        // Tips 1; 1 has two of order 1 -> 2; the trunk: 2 and 1 -> 2.
        let orders: Vec<f64> = values.iter().map(|v| v.strahler_order).collect();
        assert_eq!(orders, [2.0, 2.0, 1.0, 1.0, 1.0]);
        let distances: Vec<f64> = values.iter().map(|v| v.path_distance).collect();
        assert_eq!(distances, [10.0, 15.0, 17.0, 18.0, 19.0]);
        assert_eq!(neurons[&ObjectId(9)], [29.0, 1.0, 2.0, 3.0, 3.0, 19.0]);
    }

    #[test]
    fn two_branches_of_equal_order_raise_it() {
        // A trunk with two children of order 2 each -> 3.
        let nodes = [
            node(1.0, None, 9),
            node(1.0, Some(0), 9),
            node(1.0, Some(0), 9),
            node(1.0, Some(1), 9),
            node(1.0, Some(1), 9),
            node(1.0, Some(2), 9),
            node(1.0, Some(2), 9),
        ];
        let (values, _) = measure_trees(&nodes);
        assert_eq!(values[0].strahler_order, 3.0);
    }

    #[test]
    fn two_neurons_are_counted_apart() {
        let nodes = [node(10.0, None, 1), node(20.0, None, 1), node(5.0, None, 2)];
        let (_, neurons) = measure_trees(&nodes);
        assert_eq!(neurons[&ObjectId(1)][0], 30.0);
        assert_eq!(neurons[&ObjectId(1)][1], 2.0);
        assert_eq!(neurons[&ObjectId(2)][1], 1.0);
    }

    #[test]
    fn a_loop_of_parents_gets_no_values() {
        let nodes = [node(1.0, Some(1), 9), node(1.0, Some(0), 9)];
        let (values, _) = measure_trees(&nodes);
        assert!(values.iter().all(|v| v.branch_level.is_nan()));
    }

    /// Trace Filaments, then this step: the branching neurite of its tests.
    #[test]
    fn traced_tree_end_to_end() {
        use crate::algos::object::trace_filaments::TraceFilaments;
        use crate::image::{ImageContainer, ManagedImage, Point2d};
        use crate::object::{Object, ObjectInit};
        use crate::pipeline::pipeline::PipelineImageMeta;
        use crate::pipeline::pipeline_measure::MeasurementValues;
        use crate::pipeline::scope::Scope;
        use bitvec::prelude::*;
        use evanalyzer_cfg::core_types::{ImageChannelIdx, ImageTile, SizeUnits};
        use kornia_image::{Image, ImageSize};
        use std::sync::Arc;

        const SIDE: usize = 120;
        let (soma_class, tip_class, segment_class) = (
            ObjectClass::Global(1),
            ObjectClass::Global(2),
            ObjectClass::Global(3),
        );
        let line = |x: usize, y: usize| {
            let h = |x0: usize, x1: usize, yy: usize| (x0..=x1).contains(&x) && y.abs_diff(yy) <= 1;
            let v = |xx: usize, y0: usize, y1: usize| (y0..=y1).contains(&y) && x.abs_diff(xx) <= 1;
            if h(20, 60, 60) || v(60, 30, 90) || h(60, 100, 30) || h(60, 100, 90) {
                0.8
            } else {
                0.0
            }
        };
        let size = ImageSize {
            width: SIDE,
            height: SIDE,
        };
        let data: Vec<f32> = (0..SIDE * SIDE).map(|p| line(p % SIDE, p / SIDE)).collect();
        let mut cache = GlobalPipelineCache::default();
        cache.add_to_channel_cache(
            Arc::new(ImageContainer::F32Gray(ManagedImage {
                data: Image::<f32, 1>::new(size, data).unwrap(),
                tile_offset: Point2d { x: 0, y: 0 },
                plane: None,
                dimensionality: Default::default(),
            })),
            0,
            ImageTile {
                offset_x: 0,
                offset_y: 0,
                width: SIDE,
                height: SIDE,
            },
        );
        let point = |id: u128, x: u32, y: u32, class: ObjectClass| {
            let mut o = Object::new(ObjectInit {
                id: ObjectId(id),
                bbox: [x, y, x, y],
                area: 1,
                mask_data: BitVec::repeat(true, 1),
                sum_x: x as u64,
                sum_y: y as u64,
                ..Default::default()
            });
            o.add_object_class(class, Scope::Global).unwrap();
            o
        };
        // A small square soma at (16..=20, 58..=62).
        let mut soma = Object::new(ObjectInit {
            id: ObjectId(1),
            bbox: [16, 58, 20, 62],
            area: 25,
            mask_data: BitVec::repeat(true, 25),
            sum_x: 18 * 25,
            sum_y: 60 * 25,
            ..Default::default()
        });
        soma.add_object_class(soma_class, Scope::Global).unwrap();
        for o in [
            soma,
            point(2, 100, 30, tip_class),
            point(3, 100, 90, tip_class),
        ] {
            cache.object_cache.insert(o.id.clone(), o);
        }
        let meta = PipelineImageMeta {
            full_image_width: size,
            ..Default::default()
        };
        let mut ctx = PipelineContext::without_image(std::path::PathBuf::new(), meta).unwrap();
        TraceFilaments {
            seeds: soma_class,
            targets: tip_class,
            output_class: segment_class,
            channel: ImageChannelIdx(0),
            follow_fibres: true,
            min_width: 2.0,
            max_width: 4.0,
            max_reach: 200.0,
            min_branch_length: 5.0,
            unit: SizeUnits::Pixels,
        }
        .execute(&mut ctx, &mut cache)
        .unwrap();
        MeasureFilamentTree {
            id: "tree".into(),
            segments: segment_class,
            somata: soma_class,
        }
        .execute(&mut ctx, &mut cache)
        .unwrap();
        let values = |key: &str| -> Vec<f64> {
            match &cache
                .measurements
                .measurements
                .iter()
                .find(|m| m.info.key == format!("tree/{key}"))
                .unwrap()
                .values
            {
                MeasurementValues::Objects(v) => {
                    let mut v: Vec<f64> = v.iter().map(|v| v.value).collect();
                    v.sort_by(f64::total_cmp);
                    v
                }
                _ => panic!(),
            }
        };
        assert_eq!(values("branch_level"), [1.0, 2.0, 2.0]);
        assert_eq!(values("strahler_order"), [1.0, 1.0, 2.0]);
        assert_eq!(values("tips"), [2.0]);
        assert_eq!(values("branch_points"), [1.0]);
        assert_eq!(values("primary_neurites"), [1.0]);
        // Soma edge (20) to the branch (60), then 30 up/down and 40 right:
        // about 40 + 70 + 70 px.
        let total = values("total_length")[0];
        assert!((170.0..=200.0).contains(&total), "{total}");
    }
}
