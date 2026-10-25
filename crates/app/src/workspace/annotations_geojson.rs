//! # annotations_geojson
//!
//! Manual annotations as GeoJSON - one image at a time, read and written
//! through the backend's [`FileSystem`] (this computer, or the server the
//! project is on): a `FeatureCollection` of `Polygon`/`MultiPolygon`
//! features in image pixel coordinates (y down) - the layout QuPath reads
//! and writes, no schema of our own.
//!
//! Properties written: `objectType: "annotation"`, `classification`
//! (`{name, color: [r, g, b]}`, the first class - QuPath's), `classes` (all
//! class names, when there are several), `series`, `plane` (`{z, c, t}`),
//! `excludeFromTraining` (only when set). Export: the annotations of the
//! chosen classes. Import: the same properties (without `classes` the
//! `classification` name); each class name found goes where the user maps
//! it - a project class, a new one, no class, or not imported - so files
//! from other tools (QuPath) whose classes are named differently fit.
//!
//! Outlines run along the pixel edges, holes included: exporting and
//! importing again gives back exactly the same pixels. Outlines from other
//! tools are filled by pixel centre (a pixel belongs to the shape when its
//! centre lies inside). Imported annotations are measured on their image
//! (every channel), as drawn ones are.
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-10-19
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.

use crate::api::FileSystem;
use crate::workspace::extensions::annotations_ext::same_shape;
use crate::workspace::extensions::classification_ext::ClassificationExt;
use crate::workspace::extensions::project_ext::ProjectExt;
use crate::workspace::project_owner::ProjectWithRuntime;
use bitvec::prelude::{BitVec, Lsb0};
use evanalyzer_cfg::core_types::{ImagePlane, ObjectClass};
use evanalyzer_cfg::settings::classification_settings::Class;
use evanalyzer_cfg::settings::object_settings::ObjectMetricSettings;
use evanalyzer_core::Object;
use kornia_image::ImageSize;
use serde_json::{Value, json};
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

/// A ring: closed (first point = last), pixel-corner coordinates.
pub type Ring = Vec<[f64; 2]>;
/// A polygon: its outline first, then its holes.
pub type Polygon = Vec<Ring>;

/// What an import read.
#[derive(Debug, Default, Clone, PartialEq)]
pub struct ImportReport {
    pub objects: usize,
    /// Already there (same pixels): not added again.
    pub duplicates: usize,
    /// Classes that did not exist yet and were created.
    pub created_classes: Vec<String>,
    /// Features that are not (multi)polygons (points, lines) or are empty.
    pub skipped_features: usize,
    /// Shapes left out: their classes were set to "don't import".
    pub skipped_by_class: usize,
    /// Images whose file could not be read: their annotations have no
    /// intensities.
    pub unmeasured_images: Vec<PathBuf>,
}

// --- Masks to polygons --------------------------------------------------

/// The outline of the set pixels of `mask` (relative to `bbox` = `[x1, y1,
/// x2, y2]`, inclusive) as polygons with holes, along the pixel edges.
/// Outlines run counter-clockwise (y up, as GeoJSON wants), holes
/// clockwise. Pixels touching only at a corner are separate polygons.
pub fn mask_to_polygons(mask: &BitVec<u64, Lsb0>, bbox: [u32; 4]) -> Vec<Polygon> {
    let width = (bbox[2] - bbox[0] + 1) as i64;
    let height = (bbox[3] - bbox[1] + 1) as i64;
    let set = |x: i64, y: i64| {
        x >= 0 && y >= 0 && x < width && y < height && mask[(y * width + x) as usize]
    };
    // Directed edges with the shape on their right (screen, y down):
    // top edges run +x, right edges +y, bottom -x, left -y.
    let mut edges: Vec<((i64, i64), (i64, i64))> = Vec::new();
    for y in 0..height {
        for x in 0..width {
            if !set(x, y) {
                continue;
            }
            if !set(x, y - 1) {
                edges.push(((x, y), (x + 1, y)));
            }
            if !set(x + 1, y) {
                edges.push(((x + 1, y), (x + 1, y + 1)));
            }
            if !set(x, y + 1) {
                edges.push(((x + 1, y + 1), (x, y + 1)));
            }
            if !set(x - 1, y) {
                edges.push(((x, y + 1), (x, y)));
            }
        }
    }
    let mut from: HashMap<(i64, i64), Vec<usize>> = HashMap::new();
    for (i, (start, _)) in edges.iter().enumerate() {
        from.entry(*start).or_default().push(i);
    }
    let mut used = vec![false; edges.len()];
    let mut rings: Vec<Vec<(i64, i64)>> = Vec::new();
    for first in 0..edges.len() {
        if used[first] {
            continue;
        }
        let mut ring = vec![edges[first].0];
        let mut current = first;
        loop {
            used[current] = true;
            let (start, end) = edges[current];
            ring.push(end);
            let dir = (end.0 - start.0, end.1 - start.1);
            // Where two edges leave a corner (pixels meeting diagonally),
            // turn right first: the ring stays around its own pixels.
            let turns = [(-dir.1, dir.0), dir, (dir.1, -dir.0)];
            let next = from.get(&end).and_then(|candidates| {
                turns.iter().find_map(|turn| {
                    candidates.iter().copied().find(|&e| {
                        !used[e] && (edges[e].1.0 - end.0, edges[e].1.1 - end.1) == *turn
                    })
                })
            });
            match next {
                Some(e) => current = e,
                None => break,
            }
        }
        rings.push(simplify(ring));
    }

    // Outlines (positive area) and holes; each hole goes to the smallest
    // outline around a point just inside the shape next to it.
    let to_image = |ring: &[(i64, i64)]| -> Ring {
        ring.iter()
            .map(|&(x, y)| [(x + bbox[0] as i64) as f64, (y + bbox[1] as i64) as f64])
            .collect()
    };
    let (outlines, holes): (Vec<_>, Vec<_>) = rings.into_iter().partition(|r| signed_area(r) > 0);
    let mut polygons: Vec<(f64, Polygon)> = outlines
        .iter()
        .map(|r| (signed_area(r) as f64, vec![to_image(r)]))
        .collect();
    for hole in &holes {
        let ((x0, y0), (x1, y1)) = (hole[0], hole[1]);
        let (dx, dy) = ((x1 - x0).signum(), (y1 - y0).signum());
        // Midpoint of its first edge, a quarter pixel into the shape (on
        // the edge's right).
        let probe = (
            (x0 + x1) as f64 / 2.0 - dy as f64 * 0.25,
            (y0 + y1) as f64 / 2.0 + dx as f64 * 0.25,
        );
        let owner = outlines
            .iter()
            .enumerate()
            .filter(|(_, outline)| contains(outline, probe))
            .min_by_key(|(_, outline)| signed_area(outline))
            .map(|(i, _)| i);
        if let Some(i) = owner {
            polygons[i].1.push(to_image(hole));
        }
    }
    polygons.into_iter().map(|(_, p)| p).collect()
}

/// Twice the area, positive for counter-clockwise rings (y up).
fn signed_area(ring: &[(i64, i64)]) -> i64 {
    ring.windows(2)
        .map(|w| w[0].0 * w[1].1 - w[1].0 * w[0].1)
        .sum()
}

/// Even-odd point in ring.
fn contains(ring: &[(i64, i64)], (px, py): (f64, f64)) -> bool {
    let mut inside = false;
    for w in ring.windows(2) {
        let ((x0, y0), (x1, y1)) = (
            (w[0].0 as f64, w[0].1 as f64),
            (w[1].0 as f64, w[1].1 as f64),
        );
        if (y0 > py) != (y1 > py) && px < x0 + (py - y0) / (y1 - y0) * (x1 - x0) {
            inside = !inside;
        }
    }
    inside
}

/// Drops the corners where the outline goes straight on; keeps it closed.
fn simplify(ring: Vec<(i64, i64)>) -> Vec<(i64, i64)> {
    let mut points: Vec<(i64, i64)> = ring[..ring.len() - 1].to_vec();
    let straight = |a: (i64, i64), b: (i64, i64), c: (i64, i64)| {
        (b.0 - a.0) * (c.1 - b.1) == (b.1 - a.1) * (c.0 - b.0)
    };
    let mut i = 0;
    while points.len() > 3 && i < points.len() {
        let n = points.len();
        let (a, b, c) = (points[(i + n - 1) % n], points[i], points[(i + 1) % n]);
        if straight(a, b, c) {
            points.remove(i);
            i = i.saturating_sub(1);
        } else {
            i += 1;
        }
    }
    points.push(points[0]);
    points
}

// --- Polygons to masks --------------------------------------------------

/// The pixels whose centre lies inside the rings (even-odd: holes cut
/// out), cut to an image of `width` x `height` (0 = no limit): the mask
/// relative to its bounding box `[x1, y1, x2, y2]` (inclusive). `None` when
/// no pixel is inside.
pub fn rings_to_mask(
    rings: &[Ring],
    width: u32,
    height: u32,
) -> Option<(BitVec<u64, Lsb0>, [u32; 4])> {
    let points = rings.iter().flatten();
    let (mut min_x, mut min_y, mut max_x, mut max_y) = (f64::MAX, f64::MAX, f64::MIN, f64::MIN);
    for p in points {
        if !p[0].is_finite() || !p[1].is_finite() {
            return None;
        }
        min_x = min_x.min(p[0]);
        min_y = min_y.min(p[1]);
        max_x = max_x.max(p[0]);
        max_y = max_y.max(p[1]);
    }
    if min_x > max_x {
        return None;
    }
    let limit = |v: f64, size: u32| if size == 0 { v } else { v.min(size as f64) };
    let x_lo = min_x.floor().max(0.0) as i64;
    let x_hi = limit(max_x.ceil(), width) as i64; // exclusive
    let y_lo = min_y.floor().max(0.0) as i64;
    let y_hi = limit(max_y.ceil(), height) as i64;
    if x_hi <= x_lo || y_hi <= y_lo {
        return None;
    }

    let mut filled: Vec<(i64, i64)> = Vec::new();
    for y in y_lo..y_hi {
        let cy = y as f64 + 0.5;
        let mut crossings: Vec<f64> = Vec::new();
        for ring in rings {
            for w in ring.windows(2) {
                let ([x0, y0], [x1, y1]) = (w[0], w[1]);
                if (y0 <= cy) != (y1 <= cy) {
                    crossings.push(x0 + (cy - y0) / (y1 - y0) * (x1 - x0));
                }
            }
        }
        crossings.sort_by(f64::total_cmp);
        for pair in crossings.chunks_exact(2) {
            // Pixels whose centre x + 0.5 lies in [a, b).
            let from = (pair[0] - 0.5).ceil().max(x_lo as f64) as i64;
            let to = ((pair[1] - 0.5).ceil() as i64).min(x_hi);
            filled.extend((from..to).map(|x| (x, y)));
        }
    }
    if filled.is_empty() {
        return None;
    }
    let bx0 = filled.iter().map(|p| p.0).min()?;
    let bx1 = filled.iter().map(|p| p.0).max()?;
    let by0 = filled.iter().map(|p| p.1).min()?;
    let by1 = filled.iter().map(|p| p.1).max()?;
    let w = (bx1 - bx0 + 1) as usize;
    let mut mask = BitVec::<u64, Lsb0>::repeat(false, w * (by1 - by0 + 1) as usize);
    for (x, y) in filled {
        mask.set((y - by0) as usize * w + (x - bx0) as usize, true);
    }
    Some((mask, [bx0 as u32, by0 as u32, bx1 as u32, by1 as u32]))
}

// --- Export --------------------------------------------------------------

fn rgb(color: u32) -> Value {
    json!([(color >> 16) & 0xff, (color >> 8) & 0xff, color & 0xff])
}

/// One object as a GeoJSON feature: a manual annotation (`objectType`
/// "annotation") or a pipeline object ("detection")
fn feature(
    project: &ProjectWithRuntime,
    series: i32,
    object: &ObjectMetricSettings,
    object_type: &str,
) -> Option<Value> {
    let polygons = mask_to_polygons(&object.mask_data, object.bbox);
    let geometry = match polygons.len() {
        0 => return None,
        1 => json!({ "type": "Polygon", "coordinates": polygons[0] }),
        _ => json!({ "type": "MultiPolygon", "coordinates": polygons }),
    };
    // Class names in a stable order (by id).
    let mut classes: Vec<&Class> = object
        .object_class
        .iter()
        .filter_map(|c| project.get_class_from_id(c))
        .collect();
    classes.sort_by_key(|c| c.id);
    let mut properties = json!({
        "objectType": object_type,
        "series": series,
        "plane": { "z": object.z_stack, "c": object.c_stack, "t": object.t_stack },
    });
    if let Some(first) = classes.first() {
        properties["classification"] = json!({ "name": first.name, "color": rgb(first.color) });
    }
    if classes.len() > 1 {
        properties["classes"] = json!(classes.iter().map(|c| &c.name).collect::<Vec<_>>());
    }
    if object.exclude_from_training {
        properties["excludeFromTraining"] = json!(true);
    }
    Some(json!({
        "type": "Feature",
        "id": object.id.to_string(),
        "geometry": geometry,
        "properties": properties,
    }))
}

/// The classes of image `rel_path`'s objects - its manual annotations
/// (every series) and the `pipeline` objects shown on it - each with how
/// many objects have it; `None`: those without a class. By class id, `None`
/// last: the export dialog's list.
pub fn image_annotation_classes(
    project: &ProjectWithRuntime,
    rel_path: &Path,
    pipeline: &[ObjectMetricSettings],
) -> Vec<(Option<u32>, usize)> {
    let mut counts: HashMap<Option<u32>, usize> = HashMap::new();
    let manual = project
        .settings
        .images
        .list
        .get(rel_path)
        .into_iter()
        .flat_map(|image| image.series.values().flat_map(|s| s.objects.iter()));
    for object in manual.chain(pipeline.iter()) {
        for key in class_keys(object) {
            *counts.entry(key).or_default() += 1;
        }
    }
    let mut list: Vec<(Option<u32>, usize)> = counts.into_iter().collect();
    list.sort_by_key(|(id, _)| (id.is_none(), *id));
    list
}

/// An annotation's classes as the export dialog lists them (`None`: none).
fn class_keys(object: &ObjectMetricSettings) -> Vec<Option<u32>> {
    let ids: Vec<Option<u32>> = object
        .object_class
        .iter()
        .filter_map(|c| match c {
            ObjectClass::Global(id) => Some(Some(*id)),
            _ => None,
        })
        .collect();
    if ids.is_empty() { vec![None] } else { ids }
}

/// The pipeline (preview) objects to export with the annotations, and the
/// series they were found on.
#[derive(Debug, Clone, Copy)]
pub struct PipelineObjects<'a> {
    pub objects: &'a [ObjectMetricSettings],
    pub series: i32,
}

/// Writes the objects of image `rel_path` that have one of `classes`
/// (`None`: those without a class) to `file`, a `FeatureCollection` QuPath
/// reads: its manual annotations (every series) and the `pipeline` objects
/// shown on it. How many were written.
pub fn export_image_geojson(
    project: &ProjectWithRuntime,
    files: &dyn FileSystem,
    rel_path: &Path,
    classes: &HashSet<Option<u32>>,
    file: &Path,
    pipeline: PipelineObjects,
) -> Result<usize, String> {
    let image = project
        .settings
        .images
        .list
        .get(rel_path)
        .ok_or_else(|| format!("{} is not an image of the project", rel_path.display()))?;
    let chosen = |o: &&ObjectMetricSettings| class_keys(o).iter().any(|k| classes.contains(k));
    let features: Vec<Value> = image
        .series
        .iter()
        .flat_map(|(series, s)| {
            s.objects
                .iter()
                .filter(chosen)
                .filter_map(move |o| feature(project, *series, o, "annotation"))
        })
        .chain(
            pipeline
                .objects
                .iter()
                .filter(chosen)
                .filter_map(|o| feature(project, pipeline.series, o, "detection")),
        )
        .collect();
    if let Some(dir) = file.parent().filter(|d| !d.as_os_str().is_empty()) {
        files
            .create_dir_all(dir)
            .map_err(|e| format!("{}: {e}", dir.display()))?;
    }
    let count = features.len();
    let collection = json!({ "type": "FeatureCollection", "features": features });
    let text = serde_json::to_string_pretty(&collection).map_err(|e| e.to_string())?;
    files
        .write_file(file, text.as_bytes())
        .map_err(|e| format!("{}: {e}", file.display()))?;
    Ok(count)
}

// --- Import --------------------------------------------------------------

fn parse_ring(value: &Value) -> Option<Ring> {
    value
        .as_array()?
        .iter()
        .map(|p| {
            let p = p.as_array()?;
            Some([p.first()?.as_f64()?, p.get(1)?.as_f64()?])
        })
        .collect()
}

/// All rings of a (multi)polygon, `None` for other geometries.
fn geometry_rings(geometry: &Value) -> Option<Vec<Ring>> {
    let coordinates = geometry.get("coordinates")?.as_array()?;
    match geometry.get("type")?.as_str()? {
        "Polygon" => coordinates.iter().map(parse_ring).collect(),
        "MultiPolygon" => {
            let mut rings = Vec::new();
            for polygon in coordinates {
                for ring in polygon.as_array()? {
                    rings.push(parse_ring(ring)?);
                }
            }
            Some(rings)
        }
        _ => None,
    }
}

/// Class names of a feature, and the colour of its `classification`.
fn feature_classes(properties: &Value) -> (Vec<String>, Option<u32>) {
    let classification = properties.get("classification");
    let color = classification
        .and_then(|c| c.get("color"))
        .and_then(|c| match c {
            Value::Array(rgb) if rgb.len() >= 3 => {
                let v = |i: usize| rgb[i].as_u64().unwrap_or(0).min(255) as u32;
                Some((v(0) << 16) | (v(1) << 8) | v(2))
            }
            Value::Number(n) => n.as_i64().map(|n| (n as u32) & 0xffffff),
            _ => None,
        });
    let names: Vec<String> = match properties.get("classes").and_then(Value::as_array) {
        Some(names) => names
            .iter()
            .filter_map(|n| n.as_str().map(str::to_string))
            .collect(),
        None => classification
            .and_then(|c| c.get("name"))
            .and_then(Value::as_str)
            .map(|n| vec![n.to_string()])
            .unwrap_or_default(),
    };
    (names, color)
}

/// Where the shapes of a class name found in a file go.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ClassTarget {
    /// Into this project class (its id).
    Class(u32),
    /// Into a new class of that name (its colour from the file).
    New,
    /// Without a class.
    NoClass,
    /// Not imported.
    Skip,
}

/// A class name found in a file ("": shapes without one), how many shapes
/// carry it, its colour there.
#[derive(Debug, Clone, PartialEq)]
pub struct FileClass {
    pub name: String,
    pub count: usize,
    pub color: Option<u32>,
}

/// What a name maps to unless the user picks otherwise: the project class
/// of that name (case does not matter), else a new one; no name: no class.
pub fn default_class_target(project: &ProjectWithRuntime, name: &str) -> ClassTarget {
    if name.is_empty() {
        return ClassTarget::NoClass;
    }
    project
        .settings
        .classification
        .classes()
        .iter()
        .find(|c| c.name.eq_ignore_ascii_case(name))
        .map_or(ClassTarget::New, |c| ClassTarget::Class(c.id))
}

/// One annotation read from a file, not yet in the project.
#[derive(Debug, Clone)]
struct PendingObject {
    series: i32,
    /// Its classes by name (mapped to project classes on apply).
    class_names: Vec<String>,
    /// The colour of its `classification`, for a class created for it.
    color: Option<u32>,
    object: ObjectMetricSettings,
}

/// The annotations of one image, read.
#[derive(Debug, Clone)]
struct PendingImage {
    rel_path: PathBuf,
    /// The image file, to measure on (`None`: the project has no image
    /// folder).
    path: Option<PathBuf>,
    objects: Vec<PendingObject>,
}

/// A `.geojson` file read for one image ([`read_image_geojson`]), then
/// measured on it ([`measure_pending`]) and put into the project
/// ([`apply_import`]) - in steps, so the slow measuring needs no access to
/// the project and the user can map the file's classes in between.
#[derive(Debug, Default, Clone)]
pub struct PendingImport {
    images: Vec<PendingImage>,
    report: ImportReport,
}

impl PendingImport {
    /// The class names in the file, by first appearance; "" for shapes
    /// without one.
    pub fn file_classes(&self) -> Vec<FileClass> {
        let mut list: Vec<FileClass> = Vec::new();
        for object in self.images.iter().flat_map(|i| i.objects.iter()) {
            let names: Vec<&str> = if object.class_names.is_empty() {
                vec![""]
            } else {
                object.class_names.iter().map(String::as_str).collect()
            };
            for name in names {
                match list.iter_mut().find(|c| c.name == name) {
                    Some(class) => class.count += 1,
                    None => list.push(FileClass {
                        name: name.to_string(),
                        count: 1,
                        color: object.color,
                    }),
                }
            }
        }
        list
    }

    /// Shapes read (points, lines and empty shapes not counted).
    pub fn shapes(&self) -> usize {
        self.images.iter().map(|i| i.objects.len()).sum()
    }
}

/// Step 1: reads `file` (from EVAnalyzer, QuPath or another tool) as the
/// annotations of image `rel_path`: shapes, class names, planes. A shape's
/// series is the file's (EVAnalyzer writes it) when the image has it, else
/// `series` - the one shown. Measures nothing yet.
pub fn read_image_geojson(
    project: &ProjectWithRuntime,
    files: &dyn FileSystem,
    file: &Path,
    rel_path: &Path,
    series: i32,
) -> Result<PendingImport, String> {
    let image = project
        .settings
        .images
        .list
        .get(rel_path)
        .ok_or_else(|| format!("{} is not an image of the project", rel_path.display()))?;
    let bytes = files
        .read_file(file)
        .map_err(|e| format!("{}: {e}", file.display()))?;
    let text = String::from_utf8_lossy(&bytes);
    let value: Value =
        serde_json::from_str(&text).map_err(|e| format!("{}: not GeoJSON: {e}", file.display()))?;
    // A FeatureCollection, a list of features (older QuPath), or one.
    let features: Vec<Value> = match &value {
        Value::Array(list) => list.clone(),
        v if v.get("type").and_then(Value::as_str) == Some("FeatureCollection") => v
            .get("features")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default(),
        v => vec![v.clone()],
    };

    let sizes: HashMap<i32, (u32, u32)> = image
        .series
        .iter()
        .map(|(k, s)| (*k, (s.image_width as u32, s.image_height as u32)))
        .collect();
    let mut pending = PendingImport::default();
    let mut objects = Vec::new();
    for feature in &features {
        let Some(rings) = feature.get("geometry").and_then(geometry_rings) else {
            pending.report.skipped_features += 1;
            continue;
        };
        let properties = feature.get("properties").cloned().unwrap_or(Value::Null);
        let series = properties
            .get("series")
            .and_then(Value::as_i64)
            .map(|s| s as i32)
            .filter(|s| sizes.contains_key(s))
            .unwrap_or(series);
        let (width, height) = sizes.get(&series).copied().unwrap_or((0, 0));
        let Some((mask, bbox)) = rings_to_mask(&rings, width, height) else {
            pending.report.skipped_features += 1;
            continue;
        };
        let plane = properties.get("plane");
        let plane_value = |key: &str| {
            plane
                .and_then(|p| p.get(key))
                .and_then(Value::as_i64)
                .unwrap_or(0) as i32
        };
        let (class_names, color) = feature_classes(&properties);
        let size = ImageSize {
            width: if width == 0 {
                usize::MAX
            } else {
                width as usize
            },
            height: if height == 0 {
                usize::MAX
            } else {
                height as usize
            },
        };
        let mut object = Object::from_mask_geometry(
            &size,
            mask,
            bbox,
            ImagePlane {
                z: plane_value("z"),
                c: plane_value("c"),
                t: plane_value("t"),
            },
            HashSet::new(),
        )
        .to_object_settings();
        object.exclude_from_training = properties
            .get("excludeFromTraining")
            .and_then(Value::as_bool)
            .unwrap_or(false);
        objects.push(PendingObject {
            series,
            class_names,
            color,
            object,
        });
    }
    pending.images.push(PendingImage {
        path: project
            .settings
            .images
            .root
            .as_ref()
            .map(|root| root.join(rel_path)),
        rel_path: rel_path.to_path_buf(),
        objects,
    });
    Ok(pending)
}

/// Step 2: measures the read annotations on their images with `measure`
/// (image file, series, the objects: sets their intensities) - every
/// channel, as if drawn on the image. An image that can't be read keeps
/// its annotations unmeasured (counted in the report).
pub fn measure_pending(
    pending: &mut PendingImport,
    measure: impl Fn(&Path, i32, &mut [ObjectMetricSettings]) -> Result<(), String>,
) {
    for image in &mut pending.images {
        let series: HashSet<i32> = image.objects.iter().map(|o| o.series).collect();
        let mut failed = image.path.is_none();
        if let Some(path) = &image.path {
            for s in series {
                let mut objects: Vec<ObjectMetricSettings> = image
                    .objects
                    .iter()
                    .filter(|o| o.series == s)
                    .map(|o| o.object.clone())
                    .collect();
                match measure(path, s, &mut objects) {
                    Ok(()) => {
                        let mut measured = objects.into_iter();
                        for o in image.objects.iter_mut().filter(|o| o.series == s) {
                            if let Some(m) = measured.next() {
                                o.object.intensities = m.intensities;
                            }
                        }
                    }
                    Err(e) => {
                        log::warn!("Annotations of {} not measured: {e}", path.display());
                        failed = true;
                    }
                }
            }
        }
        if failed && !image.objects.is_empty() {
            pending
                .report
                .unmeasured_images
                .push(image.rel_path.clone());
        }
    }
}

/// Step 3: puts the read annotations into the manual annotations of their
/// image, their class names mapped by `mapping` (a name not in it: its
/// [`default_class_target`]). A shape whose names all map to
/// [`ClassTarget::Skip`] is left out. `replace`: the image's manual
/// annotations are deleted first; else added to them (pixels already
/// annotated the same are skipped).
pub fn apply_import(
    project: &mut ProjectWithRuntime,
    pending: PendingImport,
    replace: bool,
    mapping: &HashMap<String, ClassTarget>,
) -> ImportReport {
    let mut report = pending.report;
    // A new class per name, made once.
    let mut created: HashMap<String, ObjectClass> = HashMap::new();
    for image in pending.images {
        let mut objects = Vec::new();
        for p in image.objects {
            let names: Vec<String> = if p.class_names.is_empty() {
                vec![String::new()]
            } else {
                p.class_names.clone()
            };
            let targets: Vec<(String, ClassTarget)> = names
                .into_iter()
                .map(|name| {
                    let target = mapping
                        .get(&name)
                        .cloned()
                        .unwrap_or_else(|| default_class_target(project, &name));
                    (name, target)
                })
                .collect();
            if targets.iter().all(|(_, t)| *t == ClassTarget::Skip) {
                report.skipped_by_class += 1;
                continue;
            }
            let mut object = p.object;
            object.object_class = targets
                .into_iter()
                .filter_map(|(name, target)| match target {
                    ClassTarget::Class(id) => Some(ObjectClass::Global(id)),
                    ClassTarget::New if !name.is_empty() => Some(
                        *created
                            .entry(name.clone())
                            .or_insert_with(|| new_class(project, &name, p.color, &mut report)),
                    ),
                    _ => None,
                })
                .collect();
            objects.push((p.series, object));
        }
        let Some(entry) = project.settings.images.list.get_mut(&image.rel_path) else {
            continue;
        };
        if replace {
            for series in entry.series.values_mut() {
                series.objects.clear();
            }
        }
        let mut added = 0;
        for (series, object) in objects {
            let series = entry.series.entry(series).or_default();
            if series.objects.iter().any(|o| same_shape(o, &object)) {
                report.duplicates += 1;
                continue;
            }
            series.objects.push(object);
            added += 1;
        }
        report.objects += added;
    }
    report
}

/// A new project class `name` (the file's colour, else grey).
fn new_class(
    project: &mut ProjectWithRuntime,
    name: &str,
    color: Option<u32>,
    report: &mut ImportReport,
) -> ObjectClass {
    report.created_classes.push(name.to_string());
    project.settings.classification.add_class(Class {
        id: 0,
        color: color.unwrap_or(0x808080),
        name: name.to_string(),
        notes: String::new(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use evanalyzer_cfg::core_types::ObjectId;
    use evanalyzer_cfg::settings::images_settings::{ImageEntry, SeriesSettings};

    /// A mask from rows of '#' (set) and '.'.
    fn mask(rows: &[&str]) -> (BitVec<u64, Lsb0>, [u32; 4]) {
        let w = rows[0].len();
        let bits: BitVec<u64, Lsb0> = rows
            .iter()
            .flat_map(|r| r.chars().map(|c| c == '#'))
            .collect();
        (bits, [0, 0, w as u32 - 1, rows.len() as u32 - 1])
    }

    /// The mask covering the same pixels, re-expressed in `bbox`.
    fn in_box(mask: &BitVec<u64, Lsb0>, from: [u32; 4], to: [u32; 4]) -> BitVec<u64, Lsb0> {
        let fw = (from[2] - from[0] + 1) as usize;
        let tw = (to[2] - to[0] + 1) as usize;
        let th = (to[3] - to[1] + 1) as usize;
        let mut out = BitVec::<u64, Lsb0>::repeat(false, tw * th);
        for (i, set) in mask.iter().by_vals().enumerate() {
            if set {
                let x = from[0] as usize + i % fw - to[0] as usize;
                let y = from[1] as usize + i / fw - to[1] as usize;
                out.set(y * tw + x, true);
            }
        }
        out
    }

    fn round_trip(rows: &[&str]) -> Vec<Polygon> {
        let (bits, bbox) = mask(rows);
        let polygons = mask_to_polygons(&bits, bbox);
        let rings: Vec<Ring> = polygons.iter().flatten().cloned().collect();
        let (back, back_box) = rings_to_mask(&rings, 0, 0).expect("pixels");
        assert_eq!(
            in_box(&bits, bbox, back_box),
            back,
            "same pixels back for {rows:?}"
        );
        polygons
    }

    #[test]
    fn a_rectangle_is_one_ring_of_four_corners() {
        let polygons = round_trip(&["###", "###"]);
        assert_eq!(polygons.len(), 1);
        assert_eq!(
            polygons[0][0],
            vec![[0.0, 0.0], [3.0, 0.0], [3.0, 2.0], [0.0, 2.0], [0.0, 0.0]]
        );
    }

    #[test]
    fn a_ring_shape_has_its_hole() {
        let polygons = round_trip(&["#####", "#...#", "#.#.#", "#...#", "#####"]);
        // The frame with its hole, and the pixel inside the hole.
        assert_eq!(polygons.len(), 2);
        let frame = polygons
            .iter()
            .find(|p| p.len() == 2)
            .expect("frame with a hole");
        assert_eq!(frame[1].len(), 5, "a square hole");
        assert!(polygons.iter().any(|p| p.len() == 1 && p[0].len() == 5));
    }

    #[test]
    fn pixels_touching_at_a_corner_are_separate_polygons() {
        let polygons = round_trip(&["#.", ".#"]);
        assert_eq!(polygons.len(), 2);
        // Two shapes meeting at two corners around an empty pixel.
        let polygons = round_trip(&["##.", "#.#", ".##"]);
        assert_eq!(polygons.len(), 2);
        assert!(polygons.iter().all(|p| p.len() == 1), "no holes");
    }

    #[test]
    fn odd_shapes_come_back_pixel_exact() {
        round_trip(&["#"]);
        round_trip(&[".##.", "####", "#..#", "####", ".#.."]);
        round_trip(&["#.#.#", ".#.#.", "#.#.#"]);
        // Pseudo-random masks.
        let mut state = 12345u64;
        for _ in 0..200 {
            let rows: Vec<String> = (0..7)
                .map(|_| {
                    (0..9)
                        .map(|_| {
                            state = state
                                .wrapping_mul(6364136223846793005)
                                .wrapping_add(1442695040888963407);
                            if (state >> 60) < 7 { '#' } else { '.' }
                        })
                        .collect()
                })
                .collect();
            if rows.iter().any(|r| r.contains('#')) {
                round_trip(&rows.iter().map(String::as_str).collect::<Vec<_>>());
            }
        }
    }

    #[test]
    fn outlines_are_counter_clockwise_and_holes_clockwise() {
        let polygons = round_trip(&["###", "#.#", "###"]);
        let area = |r: &Ring| {
            r.windows(2)
                .map(|w| w[0][0] * w[1][1] - w[1][0] * w[0][1])
                .sum::<f64>()
        };
        assert!(area(&polygons[0][0]) > 0.0);
        assert!(area(&polygons[0][1]) < 0.0);
    }

    #[test]
    fn foreign_outlines_are_filled_by_pixel_centre() {
        // A triangle with fractional corners, as QuPath draws.
        let ring = vec![[0.2, 0.2], [4.7, 0.2], [0.2, 4.7], [0.2, 0.2]];
        let (mask, bbox) = rings_to_mask(&[ring], 0, 0).unwrap();
        assert_eq!(bbox, [0, 0, 3, 3]);
        // Inside: x + y < 4.9 - row 0 (y 0.5) the centres up to x 3.5.
        let rows: Vec<String> = (0..4)
            .map(|y| {
                (0..4)
                    .map(|x| if mask[y * 4 + x] { '#' } else { '.' })
                    .collect()
            })
            .collect();
        assert_eq!(rows, ["####", "###.", "##..", "#..."]);
        // Cut to the image.
        let (_, bbox) = rings_to_mask(
            &[vec![[-5.0, -5.0], [10.0, -5.0], [10.0, 10.0], [-5.0, -5.0]]],
            3,
            3,
        )
        .unwrap();
        assert_eq!(bbox[2], 2);
        assert!(
            rings_to_mask(
                &[vec![[0.1, 0.1], [0.4, 0.1], [0.4, 0.4], [0.1, 0.1]]],
                0,
                0
            )
            .is_none()
        );
    }

    fn project() -> ProjectWithRuntime {
        let mut project = ProjectWithRuntime::default();
        for name in ["a.tif", "sub/b.tif"] {
            let mut entry = ImageEntry {
                rel_path: PathBuf::from(name),
                ..Default::default()
            };
            entry.series.insert(
                0,
                SeriesSettings {
                    image_width: 20,
                    image_height: 20,
                    ..Default::default()
                },
            );
            project
                .settings
                .images
                .list
                .insert(PathBuf::from(name), entry);
        }
        project.settings.classification.add_class(Class {
            id: 0,
            color: 0x3080f0,
            name: "Nuclei".into(),
            notes: String::new(),
        });
        project
    }

    fn annotation(id: u128, rows: &[&str], x: u32, classes: &[u32]) -> ObjectMetricSettings {
        let (mask_data, bbox) = mask(rows);
        ObjectMetricSettings {
            id: ObjectId(id),
            object_class: classes.iter().map(|c| ObjectClass::Global(*c)).collect(),
            mask_data,
            bbox: [bbox[0] + x, bbox[1], bbox[2] + x, bbox[3]],
            z_stack: 2,
            c_stack: 1,
            t_stack: 3,
            ..Default::default()
        }
    }

    const NO_PIPELINE: PipelineObjects<'static> = PipelineObjects {
        objects: &[],
        series: 0,
    };

    /// The pipeline objects of the ticked classes go out too, as QuPath
    /// "detection"s; the manual annotations as "annotation"s.
    #[test]
    fn pipeline_objects_of_the_chosen_classes_are_exported_too() {
        let mut project = project();
        let nuclei = project
            .settings
            .classification
            .classes()
            .iter()
            .find(|c| c.name == "Nuclei")
            .unwrap()
            .id;
        project
            .settings
            .images
            .list
            .get_mut(Path::new("a.tif"))
            .unwrap()
            .series
            .get_mut(&0)
            .unwrap()
            .objects
            .push(annotation(1, &["##", "##"], 2, &[nuclei]));
        let pipeline = vec![
            annotation(10, &["###"], 8, &[nuclei]),
            annotation(11, &["#"], 12, &[]),
        ];
        assert_eq!(
            image_annotation_classes(&project, Path::new("a.tif"), &pipeline),
            vec![(Some(nuclei), 2), (None, 1)]
        );
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("all.geojson");
        let written = export_image_geojson(
            &project,
            &crate::backend::LocalFileSystem::default(),
            Path::new("a.tif"),
            &HashSet::from([Some(nuclei)]),
            &file,
            PipelineObjects {
                objects: &pipeline,
                series: 0,
            },
        )
        .unwrap();
        assert_eq!(
            written, 2,
            "the manual Nuclei and the pipeline's - not the unclassified one"
        );
        let value: Value = serde_json::from_str(&std::fs::read_to_string(&file).unwrap()).unwrap();
        let types: Vec<&str> = value["features"]
            .as_array()
            .unwrap()
            .iter()
            .map(|f| f["properties"]["objectType"].as_str().unwrap())
            .collect();
        assert_eq!(types, vec!["annotation", "detection"]);
    }

    /// Reads `file` for image `rel` (series 0), measures it on this
    /// computer, puts it in with `mapping`.
    fn import_file(
        project: &mut ProjectWithRuntime,
        file: &Path,
        rel: &str,
        replace: bool,
        mapping: &HashMap<String, ClassTarget>,
    ) -> ImportReport {
        let files = crate::backend::LocalFileSystem::default();
        let mut pending = read_image_geojson(project, &files, file, Path::new(rel), 0).unwrap();
        measure_pending(&mut pending, |path, series, objects| {
            let source = crate::backend::local::image_reader::open_image_source(path)
                .map_err(|e| e.to_string())?;
            crate::workspace::annotation_measure::measure_on_source(&source, series, objects)
                .map_err(|e| e.to_string())
        });
        apply_import(project, pending, replace, mapping)
    }

    #[test]
    fn the_chosen_classes_of_one_image_go_out_and_come_back_into_another_project() {
        let mut source = project();
        let nuclei = source
            .settings
            .classification
            .classes()
            .iter()
            .find(|c| c.name == "Nuclei")
            .unwrap()
            .id;
        let mut excluded = annotation(2, &[".#.", "###", ".#."], 10, &[]);
        excluded.exclude_from_training = true;
        {
            let a = source
                .settings
                .images
                .list
                .get_mut(Path::new("a.tif"))
                .unwrap();
            let series = a.series.get_mut(&0).unwrap();
            series
                .objects
                .push(annotation(1, &["##", "##"], 2, &[nuclei]));
            series.objects.push(excluded);
        }
        assert_eq!(
            image_annotation_classes(&source, Path::new("a.tif"), &[]),
            vec![(Some(nuclei), 1), (None, 1)]
        );
        assert!(image_annotation_classes(&source, Path::new("sub/b.tif"), &[]).is_empty());

        let dir = tempfile::tempdir().unwrap();
        let files = crate::backend::LocalFileSystem::default();
        // Only the Nuclei: one.
        let only = dir.path().join("only nuclei.geojson");
        let written = export_image_geojson(
            &source,
            &files,
            Path::new("a.tif"),
            &HashSet::from([Some(nuclei)]),
            &only,
            NO_PIPELINE,
        )
        .unwrap();
        assert_eq!(written, 1);
        // Both, under a name of the user's choosing.
        let file = dir.path().join("out/my annotations.geojson");
        let written = export_image_geojson(
            &source,
            &files,
            Path::new("a.tif"),
            &HashSet::from([Some(nuclei), None]),
            &file,
            NO_PIPELINE,
        )
        .unwrap();
        assert_eq!(written, 2);
        let value: Value = serde_json::from_str(&std::fs::read_to_string(&file).unwrap()).unwrap();
        assert_eq!(value["type"], "FeatureCollection");
        assert_eq!(
            value["features"][0]["properties"]["classification"]["name"],
            "Nuclei"
        );
        assert_eq!(
            value["features"][0]["properties"]["classification"]["color"],
            json!([0x30, 0x80, 0xf0])
        );

        // Into another image of a project without classes: "Nuclei" is
        // created (the default for an unknown name).
        let mut target = project();
        target.settings.classification.classes_mut().clear();
        let mapping = HashMap::new();
        let report = import_file(&mut target, &file, "sub/b.tif", false, &mapping);
        assert_eq!(report.objects, 2);
        assert_eq!(report.created_classes, vec!["Nuclei".to_string()]);
        let objects = &target.settings.images.list[Path::new("sub/b.tif")].series[&0].objects;
        let source_objects = &source.settings.images.list[Path::new("a.tif")].series[&0].objects;
        for (back, original) in objects.iter().zip(source_objects) {
            assert!(same_shape(back, original), "same pixels and plane");
            assert_eq!(back.object_class.len(), original.object_class.len());
            assert_eq!(back.exclude_from_training, original.exclude_from_training);
        }
        assert_eq!(objects[0].area, 4, "measured from the shape");

        // Again: nothing new; replace: the same two.
        let again = import_file(&mut target, &file, "sub/b.tif", false, &mapping);
        assert_eq!((again.objects, again.duplicates), (0, 2));
        assert!(again.created_classes.is_empty(), "the class exists now");
        let replaced = import_file(&mut target, &file, "sub/b.tif", true, &mapping);
        assert_eq!(replaced.objects, 2);
        assert_eq!(
            target.settings.images.list[Path::new("sub/b.tif")].series[&0]
                .objects
                .len(),
            2
        );
    }

    #[test]
    fn imported_annotations_are_measured_on_their_image() {
        // The core crate's test image, as the project's only image.
        let mut project = ProjectWithRuntime::default();
        let root = PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR"), "/../core/tests"));
        project.settings.images.root = Some(root);
        let mut entry = ImageEntry {
            rel_path: PathBuf::from("slice_Z0_C0_T0.tif"),
            ..Default::default()
        };
        entry.series.insert(0, SeriesSettings::default());
        project
            .settings
            .images
            .list
            .insert(PathBuf::from("slice_Z0_C0_T0.tif"), entry);

        let dir = tempfile::tempdir().unwrap();
        let square = json!({
            "type": "Feature",
            "geometry": { "type": "Polygon", "coordinates": [[[10, 10], [14, 10], [14, 13], [10, 13], [10, 10]]] },
            "properties": { "classification": { "name": "Cells" } }
        });
        let file = dir.path().join("cells.geojson");
        std::fs::write(
            &file,
            json!({ "type": "FeatureCollection", "features": [square] }).to_string(),
        )
        .unwrap();
        let report = import_file(
            &mut project,
            &file,
            "slice_Z0_C0_T0.tif",
            false,
            &HashMap::new(),
        );
        assert_eq!(report.objects, 1);
        assert!(report.unmeasured_images.is_empty(), "{report:?}");

        let object =
            &project.settings.images.list[Path::new("slice_Z0_C0_T0.tif")].series[&0].objects[0];
        assert_eq!(object.area, 12);
        let channel = object.intensities.get(&0).expect("measured on channel 0");
        assert!(channel.sum_intensity > 0.0);
        assert!((channel.avg_intensity - channel.sum_intensity / 12.0).abs() < 1e-9);
        assert!(
            channel.min_intensity <= channel.avg_intensity
                && channel.avg_intensity <= channel.max_intensity
        );

        // Without the image file: imported, not measured - and said so.
        project.settings.images.root = Some(PathBuf::from("/no/such/folder"));
        let report = import_file(
            &mut project,
            &file,
            "slice_Z0_C0_T0.tif",
            true,
            &HashMap::new(),
        );
        assert_eq!(report.objects, 1);
        assert_eq!(
            report.unmeasured_images,
            vec![PathBuf::from("slice_Z0_C0_T0.tif")]
        );
    }

    /// A QuPath file: a JSON list, a point, classes named differently -
    /// mapped onto the project's, one class left out.
    #[test]
    fn qupath_classes_are_mapped_onto_the_projects_and_points_skipped() {
        let mut project = project();
        let nuclei = project
            .settings
            .classification
            .classes()
            .iter()
            .find(|c| c.name == "Nuclei")
            .unwrap()
            .id;
        let dir = tempfile::tempdir().unwrap();
        let shape = |x: f64, class: &str| {
            json!({
                "type": "Feature",
                "geometry": { "type": "Polygon", "coordinates": [[[x, 1.0], [x + 3.0, 1.0], [x + 3.0, 3.0], [x, 3.0], [x, 1.0]]] },
                "properties": { "objectType": "annotation", "classification": { "name": class, "color": [200, 0, 0] } }
            })
        };
        let text = json!([
            shape(1.0, "Nucleus"),
            shape(6.0, "Nucleus"),
            shape(11.0, "Stroma"),
            shape(15.0, "Tumor"),
            { "type": "Feature", "geometry": { "type": "Point", "coordinates": [5.0, 5.0] }, "properties": {} }
        ]);
        let file = dir.path().join("qupath export.geojson");
        std::fs::write(&file, text.to_string()).unwrap();

        // What the mapping dialog lists; the defaults: unknown names new.
        let files = crate::backend::LocalFileSystem::default();
        let pending = read_image_geojson(&project, &files, &file, Path::new("a.tif"), 0).unwrap();
        let names: Vec<(String, usize)> = pending
            .file_classes()
            .into_iter()
            .map(|c| (c.name, c.count))
            .collect();
        assert_eq!(
            names,
            vec![
                ("Nucleus".into(), 2),
                ("Stroma".into(), 1),
                ("Tumor".into(), 1)
            ]
        );
        assert_eq!(pending.shapes(), 4);
        assert_eq!(
            default_class_target(&project, "nuclei"),
            ClassTarget::Class(nuclei)
        );
        assert_eq!(default_class_target(&project, "Stroma"), ClassTarget::New);
        assert_eq!(default_class_target(&project, ""), ClassTarget::NoClass);

        let mapping = HashMap::from([
            ("Nucleus".to_string(), ClassTarget::Class(nuclei)),
            ("Stroma".to_string(), ClassTarget::Skip),
            ("Tumor".to_string(), ClassTarget::New),
        ]);
        let report = import_file(&mut project, &file, "a.tif", false, &mapping);
        assert_eq!(report.objects, 3);
        assert_eq!(report.skipped_features, 1, "the point");
        assert_eq!(report.skipped_by_class, 1, "Stroma");
        assert_eq!(report.created_classes, vec!["Tumor".to_string()]);
        let objects = &project.settings.images.list[Path::new("a.tif")].series[&0].objects;
        assert_eq!(
            objects[0].object_class,
            HashSet::from([ObjectClass::Global(nuclei)])
        );
        assert_eq!(objects[0].bbox, [1, 1, 3, 2]);
        assert_eq!(objects[0].area, 6);
        let tumor = project
            .settings
            .classification
            .classes()
            .iter()
            .find(|c| c.name == "Tumor")
            .unwrap();
        assert_eq!(tumor.color, 0xc80000);
        assert!(
            project.settings.images.list[Path::new("sub/b.tif")].series[&0]
                .objects
                .is_empty(),
            "only the chosen image"
        );
    }
}
