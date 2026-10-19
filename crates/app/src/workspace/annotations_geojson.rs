//! Manual annotations as GeoJSON: one `<image>.geojson` per image (the
//! image's path relative to the project, mirrored in the folder), a
//! `FeatureCollection` of `Polygon`/`MultiPolygon` features in image pixel
//! coordinates (y down) - the layout QuPath reads and writes, no schema of
//! our own.
//!
//! Properties written: `objectType: "annotation"`, `classification`
//! (`{name, color: [r, g, b]}`, the first class - QuPath's), `classes` (all
//! class names, when there are several), `series`, `plane` (`{z, c, t}`),
//! `excludeFromTraining` (only when set). Read: the same; without `classes`
//! the `classification` name; unknown classes are created.
//!
//! Outlines run along the pixel edges, holes included: exporting and
//! importing again gives back exactly the same pixels. Outlines from other
//! tools are filled by pixel centre (a pixel belongs to the shape when its
//! centre lies inside). Imported annotations are measured on their image
//! (every channel), as drawn ones are.

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

/// What an export wrote.
#[derive(Debug, Default, PartialEq)]
pub struct ExportReport {
    pub files: usize,
    pub objects: usize,
}

/// What an import read.
#[derive(Debug, Default, Clone, PartialEq)]
pub struct ImportReport {
    /// Images that got annotations.
    pub images: usize,
    pub objects: usize,
    /// Already there (same pixels): not added again.
    pub duplicates: usize,
    /// Classes that did not exist yet and were created.
    pub created_classes: Vec<String>,
    /// `.geojson` files without a matching image in the project.
    pub unmatched_files: Vec<PathBuf>,
    /// Features that are not (multi)polygons (points, lines) or are empty.
    pub skipped_features: usize,
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

/// One annotation as a GeoJSON feature.
fn feature(
    project: &ProjectWithRuntime,
    series: i32,
    object: &ObjectMetricSettings,
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
        "objectType": "annotation",
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

/// Writes the manual annotations of every image that has any to
/// `<folder>/<image path>.geojson`.
pub fn export_geojson(project: &ProjectWithRuntime, folder: &Path) -> Result<ExportReport, String> {
    let mut report = ExportReport::default();
    for (rel_path, image) in &project.settings.images.list {
        let features: Vec<Value> = image
            .series
            .iter()
            .flat_map(|(series, s)| {
                s.objects
                    .iter()
                    .filter_map(move |o| feature(project, *series, o))
            })
            .collect();
        if features.is_empty() {
            continue;
        }
        let file = folder.join(geojson_name(rel_path));
        if let Some(dir) = file.parent() {
            std::fs::create_dir_all(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
        }
        report.objects += features.len();
        let collection = json!({ "type": "FeatureCollection", "features": features });
        let text = serde_json::to_string_pretty(&collection).map_err(|e| e.to_string())?;
        std::fs::write(&file, text).map_err(|e| format!("{}: {e}", file.display()))?;
        report.files += 1;
    }
    Ok(report)
}

/// `a/b.tif` -> `a/b.tif.geojson`.
fn geojson_name(rel_path: &Path) -> PathBuf {
    let mut name = rel_path.as_os_str().to_owned();
    name.push(".geojson");
    PathBuf::from(name)
}

// --- Import --------------------------------------------------------------

/// Every `.geojson` file below `folder`.
fn geojson_files(folder: &Path, out: &mut Vec<PathBuf>) -> Result<(), String> {
    let entries = std::fs::read_dir(folder).map_err(|e| format!("{}: {e}", folder.display()))?;
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            geojson_files(&path, out)?;
        } else if path
            .extension()
            .is_some_and(|e| e.eq_ignore_ascii_case("geojson"))
        {
            out.push(path);
        }
    }
    Ok(())
}

/// The project image a `.geojson` file belongs to: the same relative path
/// (as exported), else the one image whose file name or stem is the file's
/// stem (as QuPath names them).
fn matching_image(project: &ProjectWithRuntime, folder: &Path, file: &Path) -> Option<PathBuf> {
    let rel = file.strip_prefix(folder).ok()?;
    let without_suffix = rel.with_extension("");
    if project.settings.images.list.contains_key(&without_suffix) {
        return Some(without_suffix);
    }
    let stem = file.file_stem()?.to_string_lossy().to_string();
    let candidates: Vec<&PathBuf> = project
        .settings
        .images
        .list
        .keys()
        .filter(|p| {
            p.file_name().is_some_and(|n| n.to_string_lossy() == stem)
                || p.file_stem().is_some_and(|n| n.to_string_lossy() == stem)
        })
        .collect();
    (candidates.len() == 1).then(|| candidates[0].clone())
}

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

/// The project class named `name`, created when there is none.
fn class_for_name(
    project: &mut ProjectWithRuntime,
    name: &str,
    color: Option<u32>,
    report: &mut ImportReport,
) -> ObjectClass {
    if let Some(class) = project
        .settings
        .classification
        .classes()
        .iter()
        .find(|c| c.name == name)
    {
        return ObjectClass::Global(class.id);
    }
    report.created_classes.push(name.to_string());
    project.settings.classification.add_class(Class {
        id: 0,
        color: color.unwrap_or(0x808080),
        name: name.to_string(),
        notes: String::new(),
    })
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

/// `.geojson` files read and turned into annotations ([`read_geojson`]),
/// then measured on their images ([`measure_pending`]) and put into the
/// project ([`apply_import`]) - in three steps, so the slow middle one
/// needs no access to the project.
#[derive(Debug, Default, Clone)]
pub struct PendingImport {
    images: Vec<PendingImage>,
    report: ImportReport,
}

/// Step 1: reads the `.geojson` files below `folder` for the images of
/// `project` (shapes, classes by name, planes); measures nothing yet.
pub fn read_geojson(project: &ProjectWithRuntime, folder: &Path) -> Result<PendingImport, String> {
    let mut files = Vec::new();
    geojson_files(folder, &mut files)?;
    files.sort();
    let mut pending = PendingImport::default();
    for file in files {
        let Some(rel_path) = matching_image(project, folder, &file) else {
            pending.report.unmatched_files.push(file);
            continue;
        };
        let Some(image) = project.settings.images.list.get(&rel_path) else {
            continue;
        };
        let text =
            std::fs::read_to_string(&file).map_err(|e| format!("{}: {e}", file.display()))?;
        let value: Value = serde_json::from_str(&text)
            .map_err(|e| format!("{}: not GeoJSON: {e}", file.display()))?;
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

        let first_series = image.series.keys().next().copied().unwrap_or(0);
        let sizes: HashMap<i32, (u32, u32)> = image
            .series
            .iter()
            .map(|(k, s)| (*k, (s.image_width as u32, s.image_height as u32)))
            .collect();
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
                .unwrap_or(first_series);
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
                .map(|root| root.join(&rel_path)),
            rel_path,
            objects,
        });
    }
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
/// images. `replace`: an image's manual annotations are deleted first
/// (only images with a file); else added to them (pixels already annotated
/// the same are skipped). Unknown classes are created.
pub fn apply_import(
    project: &mut ProjectWithRuntime,
    pending: PendingImport,
    replace: bool,
) -> ImportReport {
    let mut report = pending.report;
    for image in pending.images {
        let mut objects = Vec::new();
        for p in image.objects {
            let mut object = p.object;
            object.object_class = p
                .class_names
                .iter()
                .map(|name| class_for_name(project, name, p.color, &mut report))
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
        report.images += usize::from(added > 0);
    }
    report
}

/// All three steps at once, measuring on the image files of this computer:
/// reads the `.geojson` files below `folder`, measures them, puts them
/// into the project (see [`apply_import`]).
pub fn import_geojson(
    project: &mut ProjectWithRuntime,
    folder: &Path,
    replace: bool,
) -> Result<ImportReport, String> {
    let mut pending = read_geojson(project, folder)?;
    measure_pending(&mut pending, |path, series, objects| {
        let source = crate::backend::local::image_reader::ReaderPool::open(path)
            .map_err(|e| e.to_string())?;
        crate::workspace::annotation_measure::measure_on_source(&source, series, objects)
            .map_err(|e| e.to_string())
    });
    Ok(apply_import(project, pending, replace))
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

    #[test]
    fn annotations_go_out_and_come_back_into_another_project() {
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
        let dir = tempfile::tempdir().unwrap();
        let report = export_geojson(&source, dir.path()).unwrap();
        assert_eq!(
            report,
            ExportReport {
                files: 1,
                objects: 2
            }
        );
        let file = dir.path().join("a.tif.geojson");
        let text = std::fs::read_to_string(&file).unwrap();
        let value: Value = serde_json::from_str(&text).unwrap();
        assert_eq!(value["type"], "FeatureCollection");
        assert_eq!(
            value["features"][0]["properties"]["classification"]["name"],
            "Nuclei"
        );
        assert_eq!(
            value["features"][0]["properties"]["classification"]["color"],
            json!([0x30, 0x80, 0xf0])
        );

        // Into a project without classes: "Nuclei" is created.
        let mut target = project();
        target.settings.classification.classes_mut().clear();
        let report = import_geojson(&mut target, dir.path(), false).unwrap();
        assert_eq!(report.images, 1);
        assert_eq!(report.objects, 2);
        assert_eq!(report.created_classes, vec!["Nuclei".to_string()]);
        let objects = &target.settings.images.list[Path::new("a.tif")].series[&0].objects;
        let source_objects = &source.settings.images.list[Path::new("a.tif")].series[&0].objects;
        for (back, original) in objects.iter().zip(source_objects) {
            assert!(same_shape(back, original), "same pixels and plane");
            assert_eq!(back.object_class.len(), original.object_class.len());
            assert_eq!(back.exclude_from_training, original.exclude_from_training);
        }
        assert_eq!(objects[0].area, 4, "measured from the shape");

        // Again: nothing new; replace: the same two.
        let again = import_geojson(&mut target, dir.path(), false).unwrap();
        assert_eq!((again.objects, again.duplicates), (0, 2));
        let replaced = import_geojson(&mut target, dir.path(), true).unwrap();
        assert_eq!(replaced.objects, 2);
        assert_eq!(
            target.settings.images.list[Path::new("a.tif")].series[&0]
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
        std::fs::write(
            dir.path().join("slice_Z0_C0_T0.tif.geojson"),
            json!({ "type": "FeatureCollection", "features": [square] }).to_string(),
        )
        .unwrap();
        let report = import_geojson(&mut project, dir.path(), false).unwrap();
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
        let report = import_geojson(&mut project, dir.path(), true).unwrap();
        assert_eq!(report.objects, 1);
        assert_eq!(
            report.unmeasured_images,
            vec![PathBuf::from("slice_Z0_C0_T0.tif")]
        );
    }

    #[test]
    fn qupath_files_are_matched_by_name_and_their_points_skipped() {
        let mut project = project();
        let dir = tempfile::tempdir().unwrap();
        // QuPath: an image name without folder, a JSON list, a point, a
        // classification name only.
        let text = json!([
            {
                "type": "Feature",
                "geometry": { "type": "Polygon", "coordinates": [[[1.0, 1.0], [4.0, 1.0], [4.0, 3.0], [1.0, 3.0], [1.0, 1.0]]] },
                "properties": { "objectType": "annotation", "classification": { "name": "Tumor", "color": [200, 0, 0] } }
            },
            { "type": "Feature", "geometry": { "type": "Point", "coordinates": [5.0, 5.0] }, "properties": {} }
        ]);
        std::fs::write(dir.path().join("b.geojson"), text.to_string()).unwrap();
        std::fs::write(dir.path().join("unknown.geojson"), "[]").unwrap();
        let report = import_geojson(&mut project, dir.path(), false).unwrap();
        assert_eq!(report.objects, 1);
        assert_eq!(report.skipped_features, 1);
        assert_eq!(report.created_classes, vec!["Tumor".to_string()]);
        assert_eq!(
            report.unmatched_files,
            vec![dir.path().join("unknown.geojson")]
        );
        let object = &project.settings.images.list[Path::new("sub/b.tif")].series[&0].objects[0];
        assert_eq!(object.bbox, [1, 1, 3, 2]);
        assert_eq!(object.area, 6);
        let tumor = project
            .settings
            .classification
            .classes()
            .iter()
            .find(|c| c.name == "Tumor")
            .unwrap();
        assert_eq!(tumor.color, 0xc80000);
    }
}
