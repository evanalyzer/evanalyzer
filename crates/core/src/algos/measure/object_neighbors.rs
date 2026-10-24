//! # Measure Object Neighbors
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-10-24
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **AGPL-3.0**.
//!
//! How many objects are near each object and how close the nearest are -
//! CellProfiler's MeasureObjectNeighbors
//! (cellprofiler/modules/measureobjectneighbors.py, CellProfiler 4.2.8):
//! the number of neighbours (objects a disk-shaped dilation of the object
//! reaches), the percentage of its outline touching them, the distance to
//! the first and second closest object (centre to centre - the result also
//! names that object) and the angle between those two.
//!
//! Worked out per object on label rasters of its surroundings (like
//! CellProfiler's patches), so whole slides need no raster of their size -
//! except "Expand until adjacent", which first grows every object over the
//! whole image.

use crate::algos::libs::feature_transform::nearest_features;
use crate::algos::{ExecutionScope, GlobalPipelineCache, ImageAlgorithm, PipelineContext};
use crate::object::Object;
use crate::pipeline::pipeline_measure::{ObjectValue, Outputs};
use evanalyzer_cfg::core_types::{
    CitationMetadata, InternalErrors, MeasurementInfo, MeasurementUnit, ObjectClass, ObjectId,
};
use macros::CommandsMeta;
use std::collections::{BTreeSet, HashMap};

/// Which objects count as neighbours (CellProfiler's "Method to determine
/// neighbors").
#[derive(CommandsMeta, Debug, Clone, Copy, PartialEq)]
pub enum NeighborMethod {
    /// Every object first grows over the background until it meets the
    /// others; then the ones it touches.
    #[cmdsmeta(display_name = "Expand until adjacent")]
    ExpandUntilAdjacent,
    /// The objects it touches (1 px apart at most).
    #[cmdsmeta(display_name = "Adjacent")]
    Adjacent,
    /// The objects within the distance.
    #[cmdsmeta(display_name = "Within a distance")]
    WithinDistance,
}

/// For each object: its neighbours, how much of its outline touches them,
/// and its first and second closest object.
///
/// The closest objects are by the distance between the centres; each
/// distance names the object it is to. "Percent touching" is the part of
/// the object's outline pixels within the distance (+ ½ px) of a neighbour.
#[derive(CommandsMeta)]
#[cmdsmeta(category = "measure", display_name = "Measure Object Neighbors")]
pub struct MeasureObjectNeighbors {
    /// The step's ID, unique in the project - its values' keys start with
    /// it ("neighbors_01/...").
    #[cmdsmeta(measurement_id = "neighbors", display_name = "ID", summary = true)]
    pub id: String,

    /// The objects measured.
    #[cmdsmeta(summary = true)]
    pub objects: ObjectClass,

    /// The objects that count as neighbours (unset: the measured objects
    /// themselves).
    #[cmdsmeta(default = ObjectClass::Unset, display_name = "Neighbors")]
    pub neighbors: ObjectClass,

    /// Which objects count as neighbours.
    #[cmdsmeta(default = NeighborMethod::ExpandUntilAdjacent, display_name = "Neighbors are")]
    pub method: NeighborMethod,

    /// With "Within a distance": how far, in pixels.
    #[cmdsmeta(
        default = 5,
        min = 1.0,
        max = 1000.0,
        step = 1.0,
        display_name = "Distance [px]"
    )]
    pub distance: u32,
}

/// The values: (id, name, unit).
const VALUES: [(&str, &str, MeasurementUnit); 5] = [
    ("number_of_neighbors", "Neighbors", MeasurementUnit::Count),
    (
        "percent_touching",
        "Percent touching",
        MeasurementUnit::Percent,
    ),
    (
        "first_closest_distance",
        "First closest distance",
        MeasurementUnit::Pixels,
    ),
    (
        "second_closest_distance",
        "Second closest distance",
        MeasurementUnit::Pixels,
    ),
    (
        "angle_between_neighbors",
        "Angle between neighbors",
        MeasurementUnit::None,
    ),
];

/// "Expand until adjacent" needs the whole image as one raster (labels,
/// the feature transform and the grown labels: 16 bytes per pixel, ~2 GB
/// here) - up to this many pixels.
const EXPAND_MAX_PIXELS: usize = 128 * 1024 * 1024;

/// One object's values: (value, the object it is about) per [`VALUES`].
type Row = [(f64, Option<ObjectId>); VALUES.len()];

impl ImageAlgorithm for MeasureObjectNeighbors {
    fn execute(
        &self,
        ctx: &mut PipelineContext,
        cache: &mut GlobalPipelineCache,
    ) -> Result<(), InternalErrors> {
        if self.objects == ObjectClass::Unset {
            return Err(InternalErrors::InvalidArgument(
                "Measure object neighbors: choose the objects".into(),
            ));
        }
        let size = cache.image_meta.full_image_width;
        let (width, height) = (size.width, size.height);
        // The values per output, worked out while the objects are borrowed.
        let per_output: Vec<Vec<ObjectValue>> = {
            let of_class = |class: ObjectClass| -> Vec<(&ObjectId, &Object)> {
                let mut list: Vec<(&ObjectId, &Object)> = cache
                    .object_cache
                    .iter()
                    .filter(|(_, object)| object.is_of(&class, ctx.scope))
                    .collect();
                list.sort_by(|a, b| a.0.cmp(b.0));
                list
            };
            let same = self.neighbors == ObjectClass::Unset || self.neighbors == self.objects;
            let objects = Indexed::new(of_class(self.objects));
            let others = if same {
                None
            } else {
                Some(Indexed::new(of_class(self.neighbors)))
            };
            let rows = neighbors(
                &objects,
                others.as_ref(),
                self.method,
                self.distance as i64,
                (width, height),
            )?;
            (0..VALUES.len())
                .map(|i| {
                    objects
                        .ids
                        .iter()
                        .zip(&rows)
                        .map(|(object, row)| ObjectValue {
                            object: object.clone(),
                            value: row[i].0,
                            related: row[i].1.clone(),
                        })
                        .collect()
                })
                .collect()
        };
        for ((id, ..), values) in VALUES.iter().zip(per_output) {
            cache.add_object_measurement(self.measurement_output(id), values);
        }
        Ok(())
    }

    fn name(&self) -> &'static str {
        "Measure Object Neighbors"
    }

    fn cite(&self) -> Vec<&'static CitationMetadata> {
        vec![
            &CitationMetadata::CELLPROFILER,
            &CitationMetadata::MAURER_2003,
        ]
    }

    /// Neighbours across tile borders: after the tiles are merged.
    fn execution_scope(&self) -> ExecutionScope {
        ExecutionScope::WholeImage
    }

    fn measures(&self) -> bool {
        true
    }

    fn measurement_outputs(&self) -> Vec<MeasurementInfo> {
        let outputs = Outputs::of(&self.id);
        VALUES
            .iter()
            .map(|(id, name, unit)| {
                let info = outputs.object(id, name, *unit);
                match *id {
                    "percent_touching" => info.range(0.0, 100.0),
                    "angle_between_neighbors" => info.range(0.0, 180.0),
                    "first_closest_distance" | "second_closest_distance" => info.naming_objects(),
                    _ => info,
                }
            })
            .collect()
    }
}

/// Objects by number (their position + 1 is their label), with a grid to
/// find those near a place.
struct Indexed<'a> {
    ids: Vec<ObjectId>,
    objects: Vec<&'a Object>,
    /// The centre of mass of each (x, y), in f64 from its pixels.
    centers: Vec<(f64, f64)>,
    grid: IndexGrid,
}

impl<'a> Indexed<'a> {
    fn new(list: Vec<(&ObjectId, &'a Object)>) -> Self {
        let objects: Vec<&Object> = list.iter().map(|(_, o)| *o).collect();
        let centers = objects.iter().map(|o| center_of_mass(o)).collect();
        Self {
            ids: list.into_iter().map(|(id, _)| id.clone()).collect(),
            grid: IndexGrid::new(objects.iter().map(|o| o.bbox).collect()),
            centers,
            objects,
        }
    }
}

/// The mean pixel position (x, y) of `object`'s mask, in f64.
fn center_of_mass(object: &Object) -> (f64, f64) {
    let [x0, y0, x1, _] = object.bbox;
    let w = (x1 - x0 + 1) as usize;
    let (mut sx, mut sy, mut n) = (0.0, 0.0, 0.0);
    for i in object.mask_data.iter_ones() {
        sx += (x0 as usize + i % w) as f64;
        sy += (y0 as usize + i / w) as f64;
        n += 1.0;
    }
    (sx / n, sy / n)
}

/// A uniform grid over bounding boxes, by index.
struct IndexGrid {
    cell: u32,
    cells: HashMap<(u32, u32), Vec<usize>>,
}

impl IndexGrid {
    fn new(boxes: Vec<[u32; 4]>) -> Self {
        let extent: u64 = boxes
            .iter()
            .map(|b| (b[2] - b[0] + 1 + b[3] - b[1] + 1) as u64)
            .sum();
        let cell = (extent / (boxes.len().max(1) as u64 * 2)).max(16) as u32;
        let mut cells: HashMap<(u32, u32), Vec<usize>> = HashMap::new();
        for (i, b) in boxes.iter().enumerate() {
            for cy in b[1] / cell..=b[3] / cell {
                for cx in b[0] / cell..=b[2] / cell {
                    cells.entry((cx, cy)).or_default().push(i);
                }
            }
        }
        Self { cell, cells }
    }

    /// The indexes whose box may reach into `[x0, y0, x1, y1)`, in order.
    fn near(&self, [x0, y0, x1, y1]: [usize; 4]) -> Vec<usize> {
        let mut found = BTreeSet::new();
        let c = self.cell as usize;
        for cy in y0 / c..=(y1.saturating_sub(1)) / c {
            for cx in x0 / c..=(x1.saturating_sub(1)) / c {
                if let Some(ids) = self.cells.get(&(cx as u32, cy as u32)) {
                    found.extend(ids.iter().copied());
                }
            }
        }
        found.into_iter().collect()
    }
}

/// A window `[x0, y0, x1, y1)` of the image and its labels.
struct Patch {
    window: [usize; 4],
    labels: Vec<u32>,
}

impl Patch {
    fn width(&self) -> usize {
        self.window[2] - self.window[0]
    }

    fn at(&self, x: usize, y: usize) -> u32 {
        self.labels[(y - self.window[1]) * self.width() + (x - self.window[0])]
    }
}

/// Where the labels of the measured objects come from: painted per window
/// (later objects over earlier ones), or the grown whole-image raster.
enum Labels<'a> {
    Painted(&'a Indexed<'a>),
    Raster { labels: Vec<u32>, width: usize },
}

impl Labels<'_> {
    fn patch(&self, window: [usize; 4]) -> Patch {
        let [x0, y0, x1, y1] = window;
        let labels = match self {
            Labels::Painted(indexed) => paint(indexed, window),
            Labels::Raster { labels, width } => (y0..y1)
                .flat_map(|y| labels[y * width + x0..y * width + x1].iter().copied())
                .collect(),
        };
        Patch { window, labels }
    }
}

/// The labels of `indexed`'s objects in `window`, later ones painted over
/// earlier ones (as a label image holds them).
fn paint(indexed: &Indexed, window: [usize; 4]) -> Vec<u32> {
    let [wx0, wy0, wx1, wy1] = window;
    let ww = wx1 - wx0;
    let mut labels = vec![0u32; ww * (wy1 - wy0)];
    for i in indexed.grid.near(window) {
        let object = indexed.objects[i];
        let [ox0, oy0, ox1, oy1] = object.bbox.map(|v| v as usize);
        if ox1 < wx0 || oy1 < wy0 || ox0 >= wx1 || oy0 >= wy1 {
            continue;
        }
        let ow = ox1 - ox0 + 1;
        for y in oy0.max(wy0)..=oy1.min(wy1 - 1) {
            for x in ox0.max(wx0)..=ox1.min(wx1 - 1) {
                if object
                    .mask_data
                    .get((y - oy0) * ow + (x - ox0))
                    .is_some_and(|b| *b)
                {
                    labels[(y - wy0) * ww + (x - wx0)] = i as u32 + 1;
                }
            }
        }
    }
    labels
}

/// CellProfiler's `strel_disk(radius)` as offsets: x² + y² <= radius².
fn disk(radius: f64) -> Vec<(i64, i64)> {
    let r = radius as i64;
    let r2 = radius * radius;
    (-r..=r)
        .flat_map(|y| (-r..=r).map(move |x| (x, y)))
        .filter(|(x, y)| (x * x + y * y) as f64 <= r2)
        .collect()
}

/// The pixels of `patch` a dilation of the pixels `source` picks with the
/// disk `offsets` (of `radius`) reaches - inside the patch (outside counts
/// as empty). For big disks by the nearest source pixel (within the radius:
/// the same pixels, x² + y² <= radius²), in time of the patch's size.
fn dilate(
    patch: &Patch,
    source: impl Fn(u32) -> bool,
    offsets: &[(i64, i64)],
    radius: f64,
) -> Vec<bool> {
    if offsets.len() > 49 {
        let w = patch.width();
        let h = patch.labels.len() / w.max(1);
        let ft = nearest_features(w, h, |x, y| source(patch.labels[y * w + x]));
        let r2 = radius * radius;
        return (0..patch.labels.len())
            .map(|i| {
                ft.get(i).is_some_and(|(r, c)| {
                    let (dy, dx) = (r - (i / w) as i64, c - (i % w) as i64);
                    (dx * dx + dy * dy) as f64 <= r2
                })
            })
            .collect();
    }
    let [x0, y0, x1, y1] = patch.window.map(|v| v as i64);
    let w = (x1 - x0) as usize;
    let mut out = vec![false; patch.labels.len()];
    for (i, label) in patch.labels.iter().enumerate() {
        if !source(*label) {
            continue;
        }
        let (x, y) = (x0 + (i % w) as i64, y0 + (i / w) as i64);
        for (dx, dy) in offsets {
            let (nx, ny) = (x + dx, y + dy);
            if nx >= x0 && ny >= y0 && nx < x1 && ny < y1 {
                out[(ny - y0) as usize * w + (nx - x0) as usize] = true;
            }
        }
    }
    out
}

/// Every object's values (CellProfiler's `MeasureObjectNeighbors.run`);
/// `others`: the neighbours when they aren't the objects themselves.
fn neighbors(
    objects: &Indexed,
    others: Option<&Indexed>,
    method: NeighborMethod,
    distance: i64,
    (width, height): (usize, usize),
) -> Result<Vec<Row>, InternalErrors> {
    let n = objects.objects.len();
    let empty: Row = std::array::from_fn(|_| (f64::NAN, None));
    let mut rows = vec![empty; n];
    if n == 0 {
        return Ok(rows);
    }
    let same = others.is_none();
    let neighbor_list = others.unwrap_or(objects);

    // The labels the dilations and outlines see.
    let (labels, distance) = match method {
        NeighborMethod::ExpandUntilAdjacent => {
            if width * height > EXPAND_MAX_PIXELS {
                return Err(InternalErrors::InvalidArgument(format!(
                    "Measure object neighbors: \"Expand until adjacent\" needs the whole \
                     image at once - {width} x {height} is too big; use \"Within a distance\""
                )));
            }
            (
                Labels::Raster {
                    labels: grown(objects, width, height),
                    width,
                },
                1,
            )
        }
        NeighborMethod::Adjacent => (Labels::Painted(objects), 1),
        NeighborMethod::WithinDistance => (Labels::Painted(objects), distance.max(1)),
    };
    let neighbor_labels = match (&labels, same) {
        (_, true) => None,
        _ => Some(Labels::Painted(neighbor_list)),
    };
    let reach = disk(distance as f64);
    let touch = disk(distance as f64 + 0.5);
    let extents = match &labels {
        Labels::Raster { labels, width } => Some(raster_extents(labels, *width, n)),
        Labels::Painted(_) => None,
    };

    for (index, row) in rows.iter_mut().enumerate() {
        let label = index as u32 + 1;
        // Where the object's pixels are in the labels (others may cover
        // some of its own, as in a label image).
        let extent = match &extents {
            Some(extents) => extents[index],
            None => {
                let [x0, y0, x1, y1] = objects.objects[index].bbox.map(|v| v as usize);
                let own = labels.patch([x0, y0, x1 + 1, y1 + 1]);
                patch_extent(&own, label)
            }
        };
        let Some([ex0, ey0, ex1, ey1]) = extent else {
            row[0].0 = 0.0;
            row[1].0 = 0.0;
            continue;
        };
        let d = distance as usize;
        let window = [
            ex0.saturating_sub(d),
            ey0.saturating_sub(d),
            (ex1 + d + 1).min(width),
            (ey1 + d + 1).min(height),
        ];
        let patch = labels.patch(window);
        let npatch = neighbor_labels.as_ref().map(|l| l.patch(window));
        let npatch = npatch.as_ref().unwrap_or(&patch);

        // The neighbours: the labels a dilation of the object reaches.
        let extended = dilate(&patch, |l| l == label, &reach, distance as f64);
        let found: BTreeSet<u32> = extended
            .iter()
            .zip(&npatch.labels)
            .filter(|(e, l)| **e && **l != 0 && !(same && **l == label))
            .map(|(_, l)| *l)
            .collect();
        row[0].0 = found.len() as f64;

        // Percent touching: outline pixels next to the neighbours.
        let touch_radius = distance as f64 + 0.5;
        let near_others = if same {
            dilate(&patch, |l| l != 0 && l != label, &touch, touch_radius)
        } else {
            dilate(npatch, |l| l != 0, &touch, touch_radius)
        };
        let (mut perimeter, mut touching) = (0usize, 0usize);
        let pw = patch.width();
        for (i, l) in patch.labels.iter().enumerate() {
            let (x, y) = (window[0] + i % pw, window[1] + i / pw);
            if *l != label || !is_outline(&patch, x, y, (width, height)) {
                continue;
            }
            perimeter += 1;
            if near_others[i] {
                touching += 1;
            }
        }
        row[1].0 = touching as f64 * 100.0 / perimeter as f64;
    }

    // The closest objects, centre to centre.
    let first = if same { 1 } else { 0 };
    if neighbor_list.objects.len() > first {
        let points = PointGrid::new(&neighbor_list.centers);
        for (index, row) in rows.iter_mut().enumerate() {
            let exclude = same.then_some(index);
            let (cx, cy) = objects.centers[index];
            let nearest = points.nearest_two((cx, cy), exclude);
            let vector = |j: usize| {
                let (nx, ny) = neighbor_list.centers[j];
                (nx - cx, ny - cy)
            };
            if let Some(j) = nearest[0] {
                let (x, y) = vector(j);
                row[2] = ((x * x + y * y).sqrt(), Some(neighbor_list.ids[j].clone()));
                if let Some(k) = nearest[1] {
                    let (x2, y2) = vector(k);
                    row[3] = (
                        (x2 * x2 + y2 * y2).sqrt(),
                        Some(neighbor_list.ids[k].clone()),
                    );
                    let dot = (x * x2 + y * y2) / ((x * x + y * y) * (x2 * x2 + y2 * y2)).sqrt();
                    row[4].0 = dot.acos() * 180.0 / std::f64::consts::PI;
                }
            }
        }
    }
    Ok(rows)
}

/// The box `[x0, y0, x1, y1]` (inclusive) of `label`'s pixels in `patch`.
fn patch_extent(patch: &Patch, label: u32) -> Option<[usize; 4]> {
    let w = patch.width();
    let mut extent: Option<[usize; 4]> = None;
    for (i, l) in patch.labels.iter().enumerate() {
        if *l != label {
            continue;
        }
        let (x, y) = (patch.window[0] + i % w, patch.window[1] + i / w);
        extent = Some(match extent {
            None => [x, y, x, y],
            Some([x0, y0, x1, y1]) => [x0.min(x), y0.min(y), x1.max(x), y1.max(y)],
        });
    }
    extent
}

/// Each label's box in a whole-image raster.
fn raster_extents(labels: &[u32], width: usize, n: usize) -> Vec<Option<[usize; 4]>> {
    let mut extents: Vec<Option<[usize; 4]>> = vec![None; n];
    for (i, l) in labels.iter().enumerate() {
        if *l == 0 {
            continue;
        }
        let (x, y) = (i % width, i / width);
        let e = &mut extents[*l as usize - 1];
        *e = Some(match *e {
            None => [x, y, x, y],
            Some([x0, y0, x1, y1]) => [x0.min(x), y0.min(y), x1.max(x), y1.max(y)],
        });
    }
    extents
}

/// centrosome's `outline`: a pixel on the image border, or with a
/// different label among its 8 neighbours.
fn is_outline(patch: &Patch, x: usize, y: usize, (width, height): (usize, usize)) -> bool {
    if x == 0 || y == 0 || x + 1 == width || y + 1 == height {
        return true;
    }
    let label = patch.at(x, y);
    let [x0, y0, x1, y1] = patch.window;
    for (dx, dy) in [
        (-1i64, -1i64),
        (0, -1),
        (1, -1),
        (-1, 0),
        (1, 0),
        (-1, 1),
        (0, 1),
        (1, 1),
    ] {
        let (nx, ny) = (x as i64 + dx, y as i64 + dy);
        // Outside the patch: the patch reaches 1 px past the object, so
        // that is only the case where the object isn't.
        if nx < x0 as i64 || ny < y0 as i64 || nx >= x1 as i64 || ny >= y1 as i64 {
            continue;
        }
        if patch.at(nx as usize, ny as usize) != label {
            return true;
        }
    }
    false
}

/// The objects' labels over the whole image, every background pixel given
/// the label of its nearest object pixel (CellProfiler expands with SciPy's
/// `distance_transform_edt` indices).
fn grown(objects: &Indexed, width: usize, height: usize) -> Vec<u32> {
    let labels = paint(objects, [0, 0, width, height]);
    let ft = nearest_features(width, height, |x, y| labels[y * width + x] != 0);
    (0..labels.len())
        .map(|i| match (labels[i], ft.get(i)) {
            (0, Some((r, c))) => labels[r as usize * width + c as usize],
            (l, _) => l,
        })
        .collect()
}

/// The centres in a grid, to find the nearest of a point.
struct PointGrid<'a> {
    points: &'a [(f64, f64)],
    cell: f64,
    cells: HashMap<(i64, i64), Vec<usize>>,
    /// The cells' range, to know when the search has seen everything.
    span: (i64, i64, i64, i64),
}

impl<'a> PointGrid<'a> {
    fn new(points: &'a [(f64, f64)]) -> Self {
        let (mut x0, mut y0, mut x1, mut y1) = (f64::MAX, f64::MAX, f64::MIN, f64::MIN);
        for (x, y) in points {
            (x0, y0, x1, y1) = (x0.min(*x), y0.min(*y), x1.max(*x), y1.max(*y));
        }
        // About one point per cell.
        let area = ((x1 - x0).max(1.0)) * ((y1 - y0).max(1.0));
        let cell = (area / points.len().max(1) as f64).sqrt().max(1.0);
        let key = |x: f64, y: f64| ((x / cell).floor() as i64, (y / cell).floor() as i64);
        let mut cells: HashMap<(i64, i64), Vec<usize>> = HashMap::new();
        for (i, (x, y)) in points.iter().enumerate() {
            cells.entry(key(*x, *y)).or_default().push(i);
        }
        let (kx0, ky0) = key(x0, y0);
        let (kx1, ky1) = key(x1, y1);
        Self {
            points,
            cell,
            cells,
            span: (kx0, ky0, kx1, ky1),
        }
    }

    /// The two nearest points to `p` (not `exclude`), nearest first; equally
    /// near: the lower index first (as CellProfiler's sort).
    fn nearest_two(&self, p: (f64, f64), exclude: Option<usize>) -> [Option<usize>; 2] {
        let (kx, ky) = (
            (p.0 / self.cell).floor() as i64,
            (p.1 / self.cell).floor() as i64,
        );
        let mut best: Vec<(f64, usize)> = Vec::with_capacity(3);
        let consider = |best: &mut Vec<(f64, usize)>, i: usize| {
            if Some(i) == exclude {
                return;
            }
            let (x, y) = self.points[i];
            let d = ((p.0 - x) * (p.0 - x) + (p.1 - y) * (p.1 - y)).sqrt();
            best.push((d, i));
            best.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
            best.truncate(2);
        };
        let (sx0, sy0, sx1, sy1) = self.span;
        let mut ring = 0i64;
        loop {
            for cy in ky - ring..=ky + ring {
                for cx in kx - ring..=kx + ring {
                    if (cy - ky).abs() != ring && (cx - kx).abs() != ring {
                        continue;
                    }
                    if let Some(ids) = self.cells.get(&(cx, cy)) {
                        for &i in ids {
                            consider(&mut best, i);
                        }
                    }
                }
            }
            // Anything in further rings is at least `ring` cells away.
            let all_seen =
                kx - ring <= sx0 && ky - ring <= sy0 && kx + ring >= sx1 && ky + ring >= sy1;
            let bound = ring as f64 * self.cell;
            if all_seen || (best.len() == 2 && best[1].0 < bound) {
                break;
            }
            ring += 1;
        }
        [best.first().map(|b| b.1), best.get(1).map(|b| b.1)]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::object::ObjectInit;
    use crate::pipeline::pipeline::PipelineImageMeta;
    use crate::pipeline::pipeline_measure::MeasurementValues;
    use crate::pipeline::scope::Scope;
    use bitvec::prelude::*;
    use kornia_image::ImageSize;

    const CELL: ObjectClass = ObjectClass::Global(1);
    const SPOT: ObjectClass = ObjectClass::Global(2);
    /// Neighbour objects' ids: this plus their label.
    const SPOT_IDS: u128 = 1000;

    /// One case of `tests/object_neighbors_cellprofiler.json`.
    #[derive(serde::Deserialize)]
    struct Case {
        name: String,
        width: usize,
        height: usize,
        method: String,
        distance: u32,
        labels: Vec<u32>,
        neighbor_labels: Option<Vec<u32>>,
        number_of_neighbors: Vec<f64>,
        percent_touching: Vec<f64>,
        first_closest_object: Vec<u32>,
        first_closest_distance: Vec<f64>,
        second_closest_object: Vec<u32>,
        second_closest_distance: Vec<f64>,
        angle_between_neighbors: Vec<f64>,
    }

    fn add_objects(
        cache: &mut GlobalPipelineCache,
        labels: &[u32],
        width: usize,
        class: ObjectClass,
        ids: u128,
    ) {
        let max = labels.iter().copied().max().unwrap_or(0);
        for label in 1..=max {
            let pixels: Vec<(u32, u32)> = (0..labels.len())
                .filter(|&p| labels[p] == label)
                .map(|p| ((p % width) as u32, (p / width) as u32))
                .collect();
            let x0 = pixels.iter().map(|p| p.0).min().unwrap();
            let y0 = pixels.iter().map(|p| p.1).min().unwrap();
            let x1 = pixels.iter().map(|p| p.0).max().unwrap();
            let y1 = pixels.iter().map(|p| p.1).max().unwrap();
            let w = (x1 - x0 + 1) as usize;
            let mut mask = BitVec::<u64, Lsb0>::repeat(false, w * (y1 - y0 + 1) as usize);
            for (x, y) in &pixels {
                mask.set((y - y0) as usize * w + (x - x0) as usize, true);
            }
            let mut object = Object::new(ObjectInit {
                id: ObjectId(ids + label as u128),
                bbox: [x0, y0, x1, y1],
                area: pixels.len(),
                mask_data: mask,
                ..Default::default()
            });
            object.add_object_class(class, Scope::Global).unwrap();
            cache.object_cache.insert(object.id.clone(), object);
        }
    }

    fn close(got: f64, expected: f64, what: &str) {
        assert!(
            (got - expected).abs() <= 1e-9 * expected.abs().max(1.0),
            "{what}: {got} vs {expected}"
        );
    }

    /// Every value as CellProfiler's MeasureObjectNeighbors gives it
    /// (fixture from `tests/object_neighbors_cellprofiler.py`); the closest
    /// distances name CellProfiler's closest objects.
    #[test]
    fn values_are_cellprofilers() {
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/object_neighbors_cellprofiler.json"
        );
        let cases: Vec<Case> =
            serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
        assert_eq!(cases.len(), 7);
        for case in &cases {
            let mut cache = GlobalPipelineCache::default();
            cache.image_meta.full_image_width = ImageSize {
                width: case.width,
                height: case.height,
            };
            add_objects(&mut cache, &case.labels, case.width, CELL, 0);
            if let Some(spots) = &case.neighbor_labels {
                add_objects(&mut cache, spots, case.width, SPOT, SPOT_IDS);
            }
            let step = MeasureObjectNeighbors {
                id: "n_01".into(),
                objects: CELL,
                neighbors: if case.neighbor_labels.is_some() {
                    SPOT
                } else {
                    ObjectClass::Unset
                },
                method: match case.method.as_str() {
                    "Expand until adjacent" => NeighborMethod::ExpandUntilAdjacent,
                    "Adjacent" => NeighborMethod::Adjacent,
                    _ => NeighborMethod::WithinDistance,
                },
                distance: case.distance,
            };
            let mut ctx = PipelineContext::without_image(
                std::path::PathBuf::new(),
                PipelineImageMeta::default(),
            )
            .unwrap();
            step.execute(&mut ctx, &mut cache).unwrap();
            let values = |id: &str| -> Vec<ObjectValue> {
                let key = format!("n_01/{id}");
                let m = cache
                    .measurements
                    .measurements
                    .iter()
                    .find(|m| m.info.key == key)
                    .unwrap();
                let MeasurementValues::Objects(values) = &m.values else {
                    panic!("per object")
                };
                values.clone()
            };
            let related = |number: u32| {
                let base = if case.neighbor_labels.is_some() {
                    SPOT_IDS
                } else {
                    0
                };
                ObjectId(base + number as u128)
            };
            for v in values("number_of_neighbors") {
                let i = v.object.0 as usize - 1;
                assert_eq!(
                    v.value,
                    case.number_of_neighbors[i],
                    "{} object {}: neighbours",
                    case.name,
                    i + 1
                );
            }
            for v in values("percent_touching") {
                let i = v.object.0 as usize - 1;
                close(
                    v.value,
                    case.percent_touching[i],
                    &format!("{} object {}: touching", case.name, i + 1),
                );
            }
            for (id, objects, distances) in [
                (
                    "first_closest_distance",
                    &case.first_closest_object,
                    &case.first_closest_distance,
                ),
                (
                    "second_closest_distance",
                    &case.second_closest_object,
                    &case.second_closest_distance,
                ),
            ] {
                for v in values(id) {
                    let i = v.object.0 as usize - 1;
                    let what = format!("{} object {}: {id}", case.name, i + 1);
                    close(v.value, distances[i], &what);
                    assert_eq!(v.related, Some(related(objects[i])), "{what}");
                }
            }
            for v in values("angle_between_neighbors") {
                let i = v.object.0 as usize - 1;
                close(
                    v.value,
                    case.angle_between_neighbors[i],
                    &format!("{} object {}: angle", case.name, i + 1),
                );
            }
        }
    }

    /// One object alone: no neighbours, no closest object (no value - not
    /// CellProfiler's 0).
    #[test]
    fn a_lone_object_has_no_closest_one() {
        let mut cache = GlobalPipelineCache::default();
        cache.image_meta.full_image_width = ImageSize {
            width: 10,
            height: 10,
        };
        let mut labels = vec![0u32; 100];
        labels[55] = 1;
        add_objects(&mut cache, &labels, 10, CELL, 0);
        let step = MeasureObjectNeighbors {
            id: "n".into(),
            objects: CELL,
            neighbors: ObjectClass::Unset,
            method: NeighborMethod::Adjacent,
            distance: 5,
        };
        let mut ctx =
            PipelineContext::without_image(std::path::PathBuf::new(), PipelineImageMeta::default())
                .unwrap();
        step.execute(&mut ctx, &mut cache).unwrap();
        let get = |id: &str| match &cache
            .measurements
            .measurements
            .iter()
            .find(|m| m.info.key == format!("n/{id}"))
            .unwrap()
            .values
        {
            MeasurementValues::Objects(v) => v[0].clone(),
            _ => panic!(),
        };
        assert_eq!(get("number_of_neighbors").value, 0.0);
        assert!(get("first_closest_distance").value.is_nan());
        assert_eq!(get("first_closest_distance").related, None);
    }

    /// Speed and peak RAM: a 9 × 9 object every 40 px on `BENCH_SIZE`²
    /// (default 8192: 42,025 objects), `BENCH_METHOD` (expand | adjacent |
    /// within) - release, one per process:
    /// `cargo test --release -p evanalyzer_core --lib bench_object_neighbors -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn bench_object_neighbors() {
        let size: usize = std::env::var("BENCH_SIZE")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(8192);
        let method = match std::env::var("BENCH_METHOD").as_deref() {
            Ok("adjacent") => NeighborMethod::Adjacent,
            Ok("within") => NeighborMethod::WithinDistance,
            _ => NeighborMethod::ExpandUntilAdjacent,
        };
        let mut cache = GlobalPipelineCache::default();
        cache.image_meta.full_image_width = ImageSize {
            width: size,
            height: size,
        };
        let mut id = 1u128;
        for y in (0..size - 9).step_by(40) {
            for x in (0..size - 9).step_by(40) {
                let mut object = Object::new(ObjectInit {
                    id: ObjectId(id),
                    bbox: [x as u32, y as u32, x as u32 + 8, y as u32 + 8],
                    area: 81,
                    mask_data: BitVec::<u64, Lsb0>::repeat(true, 81),
                    ..Default::default()
                });
                object.add_object_class(CELL, Scope::Global).unwrap();
                cache.object_cache.insert(object.id.clone(), object);
                id += 1;
            }
        }
        let step = MeasureObjectNeighbors {
            id: "n".into(),
            objects: CELL,
            neighbors: ObjectClass::Unset,
            method,
            distance: 35,
        };
        let mut ctx =
            PipelineContext::without_image(std::path::PathBuf::new(), PipelineImageMeta::default())
                .unwrap();
        let start = std::time::Instant::now();
        step.execute(&mut ctx, &mut cache).unwrap();
        let elapsed = start.elapsed();
        let hwm = std::fs::read_to_string("/proc/self/status")
            .ok()
            .and_then(|s| {
                s.lines()
                    .find(|l| l.starts_with("VmHWM"))
                    .map(str::to_string)
            })
            .unwrap_or_default();
        eprintln!(
            "BENCH object neighbors {size}², {} objects, {method:?}: {elapsed:?}, {hwm}",
            id - 1
        );
    }
}
