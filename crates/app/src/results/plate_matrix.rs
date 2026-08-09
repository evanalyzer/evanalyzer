//! Plate/Well matrix computation for the Results window's Matrix view.
//!
//! Reuses [`aggregate_rows`]'s grouping engine unchanged: a "well" is just
//! another `GroupBy::Folder`/`GroupBy::Regex` group, and a "field of view"
//! inside a well is just a `GroupBy::Image` group over that well's objects.
//! The only genuinely new work here is placing those groups into a 2D grid:
//! decoding a well label like `"A14"` into a `(row, col)` plate coordinate,
//! and mapping a captured sub-position number to a slot inside a well via
//! `well_image_order`.

use super::results_loader::{
    AggFunc, ColumnSpec, GroupBy, GroupConfig, aggregate_rows, group_key, group_key_for,
    is_colocalized,
};
use evanalyzer_core::{ImageRow, ObjectRow};
use std::collections::{HashMap, HashSet};

/// One cell of a plate-level matrix: an aggregated value for one well, an
/// occupied-but-empty well (`value: None`, `count: 0`, but a non-empty
/// `label` — a real well whose objects didn't survive the current filter, or
/// had none to begin with), or a genuinely empty placeholder
/// (`label: ""`/`value: None`) if no well or image landed there at all.
#[derive(Debug, Clone, PartialEq)]
pub struct PlateCell {
    pub label: String,
    pub value: Option<f64>,
    pub count: usize,
    /// Objects in this well with at least one colocalization partner.
    pub coloc_count: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PlateMatrixResult {
    pub rows: usize,
    pub cols: usize,
    /// Row-major, length `rows * cols`.
    pub cells: Vec<PlateCell>,
    /// The smallest `(rows, cols)` plate size that would fit every
    /// well-shaped label found in the data (1-indexed span, e.g. a well
    /// decoded at row D / col 19 contributes `(4, 19)`), regardless of
    /// whether it actually fit in the *configured* `plate_rows`/`plate_cols`
    /// passed in. `None` if no label decoded as a well coordinate at all
    /// (pure folder/sequential mode, or a regex whose group 1 isn't
    /// well-shaped). Lets the caller tell "the regex matched nothing" apart
    /// from "the plate size is too small for this data" and offer a fix.
    pub required_span: Option<(usize, usize)>,
    /// How many groups `aggregate_rows` produced, before placement. Together
    /// with `required_span == None`, a nonzero count means the regex *did*
    /// match and group the data - just not into well-shaped keys (e.g. its
    /// first capture group captured more than the well id) - as opposed to
    /// the regex matching nothing at all.
    pub group_count: usize,
    /// One example group label, for diagnostics (e.g. showing the user what
    /// their regex's first capture group actually captured).
    pub sample_label: Option<String>,
}

/// One cell of a well-level matrix: an aggregated value for one field-of-view
/// image inside the well, an occupied-but-empty image (processed but no
/// objects survived the current filter/detection), or a genuinely empty
/// placeholder if no image occupies that acquisition-order slot at all.
#[derive(Debug, Clone, PartialEq)]
pub struct WellCell {
    pub image_name: String,
    pub value: Option<f64>,
    pub count: usize,
    /// Objects in this field of view with at least one colocalization partner.
    pub coloc_count: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct WellMatrixResult {
    pub rows: usize,
    pub cols: usize,
    pub well_label: String,
    /// Row-major, length `rows * cols`.
    pub cells: Vec<WellCell>,
}

/// A best-effort regex suggestion from [`suggest_regex`]: `pattern` matched
/// `matched` of `total` sample filenames.
#[derive(Debug, Clone, PartialEq)]
pub struct RegexSuggestion {
    pub pattern: String,
    pub matched: usize,
    pub total: usize,
}

fn empty_plate_cell() -> PlateCell {
    PlateCell {
        label: String::new(),
        value: None,
        count: 0,
        coloc_count: 0,
    }
}

fn empty_well_cell() -> WellCell {
    WellCell {
        image_name: String::new(),
        value: None,
        count: 0,
        coloc_count: 0,
    }
}

/// Decodes a well label shaped like `<letters><digits>` (e.g. `"A14"`,
/// `"AA7"`) into a zero-based `(row, col)`. Row letters use spreadsheet-style
/// bijective base-26 (`A`=0, `B`=1, ..., `Z`=25, `AA`=26, ...) so plates with
/// more than 26 rows (e.g. a 48x72 preset) still decode correctly. Returns
/// `None` for anything not shaped that way (folder names, non-well-shaped
/// regex groups) — callers fall back to sequential placement in that case.
fn row_col_from_label(label: &str) -> Option<(usize, usize)> {
    let split = label.find(|c: char| c.is_ascii_digit())?;
    let (letters, digits) = label.split_at(split);
    if letters.is_empty()
        || digits.is_empty()
        || !letters.chars().all(|c| c.is_ascii_alphabetic())
        || !digits.chars().all(|c| c.is_ascii_digit())
    {
        return None;
    }
    let mut row: usize = 0;
    for c in letters.chars() {
        let v = (c.to_ascii_uppercase() as u8 - b'A' + 1) as usize;
        row = row * 26 + v;
    }
    let col: usize = digits.parse().ok()?;
    Some((row.checked_sub(1)?, col.checked_sub(1)?))
}

/// Encodes a zero-based plate row index back into a spreadsheet-style
/// bijective base-26 letter label (`0`->`"A"`, `25`->`"Z"`, `26`->`"AA"`,
/// ...) — the inverse of the letter-decoding half of
/// [`row_col_from_label`]. Used for plate row headers (e.g. in the CSV/XLSX
/// matrix export), where a row needs a label independent of any particular
/// well's data.
pub fn row_label(index: usize) -> String {
    let mut n = index + 1;
    let mut letters = Vec::new();
    while n > 0 {
        let rem = (n - 1) % 26;
        letters.push((b'A' + rem as u8) as char);
        n = (n - 1) / 26;
    }
    letters.iter().rev().collect()
}

/// Resolves a color scale's `[min, max]` from either the data itself (`auto
/// = true`, spanning `[0, max(values)]`) or an explicit manual range,
/// guaranteeing `max > min` (a degenerate/inverted manual range collapses to
/// a hairline span above `min` rather than dividing by zero downstream).
/// Shared by the live Matrix view and the CSV/XLSX matrix export so the two
/// can never compute different colors for the same data.
pub fn resolve_range(values: &[f64], auto: bool, min: f64, max: f64) -> (f64, f64) {
    if auto {
        let hi = values.iter().copied().fold(0.0f64, f64::max).max(1e-9);
        (0.0, hi)
    } else if max > min {
        (min, max)
    } else {
        (min, min + 1e-9)
    }
}

/// Forces `spec` visible (matrix cells always need their one metric column
/// populated, regardless of the source column-visibility state) and clears
/// unrelated flags that don't matter to `aggregate_rows`.
fn visible_metric_spec(spec: &ColumnSpec) -> ColumnSpec {
    ColumnSpec {
        id: spec.id.clone(),
        label: spec.label.clone(),
        filterable: spec.filterable,
        visible: true,
    }
}

/// Computes one aggregated value per well/group (via [`aggregate_rows`]) and
/// places it into a `plate_rows x plate_cols` grid.
///
/// Groups whose label decodes as `<letters><digits>` (e.g. `"A14"`) are
/// placed at that exact coordinate; everything else (folder names, or a
/// regex whose capture group 1 isn't well-shaped) fills the remaining empty
/// cells in row-major order. Groups beyond the plate's capacity are dropped.
///
/// `all_images` (every image the file has a record of, regardless of object
/// count — see `evanalyzer_core::ImageRow`) fills in any well/group that
/// none of `objects` landed in, as an occupied-but-valueless cell (`value:
/// None`, `count: 0`), instead of leaving it looking identical to a well
/// that was never imaged at all. Pass `&[]` to skip this (every unoccupied
/// slot then stays a genuinely empty placeholder, `label: ""`).
pub fn compute_plate_matrix(
    objects: &[ObjectRow],
    all_images: &[ImageRow],
    group_by: GroupBy,
    regex: &str,
    agg: AggFunc,
    metric_column: &ColumnSpec,
    plate_rows: usize,
    plate_cols: usize,
) -> PlateMatrixResult {
    let config = GroupConfig {
        group_by,
        regex: regex.to_string(),
        aggs: vec![agg],
        split_colocalized: false,
        group_by_class: false,
    };
    let base_specs = [visible_metric_spec(metric_column)];
    let (specs, rows) = aggregate_rows(objects, &config, &base_specs);
    let value_col = specs
        .iter()
        .position(|s| s.id == format!("{}__{}", metric_column.id, agg.label()));

    let re = (group_by == GroupBy::Regex)
        .then(|| regex::Regex::new(regex).ok())
        .flatten();

    // Per-well count of objects with >=1 colocalization partner, tallied
    // straight from `objects` — `aggregate_rows`'s grouping engine has no
    // notion of "colocalized count" for a group that isn't split in two.
    let mut coloc_counts: HashMap<String, usize> = HashMap::new();
    for object in objects {
        if is_colocalized(&object.coloc_json)
            && let Some(key) = group_key(object, group_by, re.as_ref())
        {
            *coloc_counts.entry(key).or_default() += 1;
        }
    }

    let total = plate_rows * plate_cols;
    let mut cells: Vec<PlateCell> = (0..total).map(|_| empty_plate_cell()).collect();
    let mut occupied = vec![false; total];
    let mut required_span: Option<(usize, usize)> = None;
    let mut present_labels: HashSet<String> = HashSet::new();
    let mut unplaced: Vec<PlateCell> = Vec::new();

    // Shared placement: well-shaped labels go straight to their decoded
    // coordinate (tracking the largest span seen, even out of bounds);
    // everything else queues up for sequential fill-in below.
    let place = |label: &str,
                 cell: PlateCell,
                 required_span: &mut Option<(usize, usize)>,
                 cells: &mut [PlateCell],
                 occupied: &mut [bool],
                 unplaced: &mut Vec<PlateCell>| {
        match row_col_from_label(label) {
            Some((r, c)) => {
                let (need_r, need_c) = required_span.unwrap_or((0, 0));
                *required_span = Some((need_r.max(r + 1), need_c.max(c + 1)));
                if r < plate_rows && c < plate_cols {
                    let idx = r * plate_cols + c;
                    cells[idx] = cell;
                    occupied[idx] = true;
                }
                // else: well-shaped but outside the configured plate bounds -
                // the plate size setting doesn't match this data, so drop it
                // rather than silently misplacing it into an unrelated slot;
                // `required_span` tells the caller how big it needs to be.
            }
            // Not well-shaped at all (folder names, or a regex whose group 1
            // isn't a well id) - fill sequentially instead.
            None => unplaced.push(cell),
        }
    };

    for row in &rows {
        let label = row.values.first().cloned().unwrap_or_default();
        let count: usize = row.values.get(1).and_then(|v| v.parse().ok()).unwrap_or(0);
        let value = value_col
            .and_then(|i| row.values.get(i))
            .and_then(|v| v.parse::<f64>().ok());
        let coloc_count = coloc_counts.get(&label).copied().unwrap_or(0);
        let cell = PlateCell {
            label: label.clone(),
            value,
            count,
            coloc_count,
        };
        present_labels.insert(label.clone());
        place(
            &label,
            cell,
            &mut required_span,
            &mut cells,
            &mut occupied,
            &mut unplaced,
        );
    }

    // Images that produced no objects — or none surviving the current
    // filter — still get a cell for whatever well/group they decode into,
    // as long as no real group already claimed that label.
    for image in all_images {
        let Some(label) = group_key_for(
            &image.image_name,
            &image.image_rel_path,
            group_by,
            re.as_ref(),
        ) else {
            continue;
        };
        if !present_labels.insert(label.clone()) {
            continue;
        }
        let cell = PlateCell {
            label: label.clone(),
            value: None,
            count: 0,
            coloc_count: 0,
        };
        place(
            &label,
            cell,
            &mut required_span,
            &mut cells,
            &mut occupied,
            &mut unplaced,
        );
    }

    let mut next_slot = 0;
    for cell in unplaced {
        while next_slot < total && occupied[next_slot] {
            next_slot += 1;
        }
        if next_slot >= total {
            break; // more groups than the plate has room for; silently truncate
        }
        cells[next_slot] = cell;
        occupied[next_slot] = true;
    }

    PlateMatrixResult {
        rows: plate_rows,
        cols: plate_cols,
        cells,
        required_span,
        group_count: rows.len(),
        sample_label: rows
            .first()
            .map(|r| r.values.first().cloned().unwrap_or_default()),
    }
}

/// Computes one aggregated value per field-of-view image inside a single
/// well (via [`aggregate_rows`], scoped to that well's objects and grouped by
/// `GroupBy::Image`) and places it into a `well_rows x well_cols` grid using
/// `well_image_order` (the acquisition-order preset from `PlateSettings`):
/// the image whose captured sub-position equals `well_image_order[i]` is
/// placed at slot `i` (row-major).
///
/// Requires `regex` to have a 4th capture group (the sub-position — the
/// shape produced by the "Filename well" preset and by [`suggest_regex`]).
/// Returns `None` if the regex has no usable group 4, or no object *and* no
/// image (see `all_images` on [`compute_plate_matrix`]) belongs to
/// `well_label`.
#[allow(clippy::too_many_arguments)]
pub fn compute_well_matrix(
    objects: &[ObjectRow],
    all_images: &[ImageRow],
    regex: &str,
    well_label: &str,
    agg: AggFunc,
    metric_column: &ColumnSpec,
    well_rows: usize,
    well_cols: usize,
    well_image_order: &[i32],
) -> Option<WellMatrixResult> {
    let re = regex::Regex::new(regex).ok()?;
    if re.captures_len() < 5 {
        return None; // no group(4) present - no sub-position data
    }

    let well_objects: Vec<ObjectRow> = objects
        .iter()
        .filter(|o| group_key(o, GroupBy::Regex, Some(&re)).as_deref() == Some(well_label))
        .cloned()
        .collect();
    let well_images: Vec<&ImageRow> = all_images
        .iter()
        .filter(|i| {
            group_key_for(&i.image_name, &i.image_rel_path, GroupBy::Regex, Some(&re)).as_deref()
                == Some(well_label)
        })
        .collect();
    if well_objects.is_empty() && well_images.is_empty() {
        return None;
    }

    let config = GroupConfig {
        group_by: GroupBy::Image,
        regex: String::new(),
        aggs: vec![agg],
        split_colocalized: false,
        group_by_class: false,
    };
    let base_specs = [visible_metric_spec(metric_column)];
    let (specs, rows) = aggregate_rows(&well_objects, &config, &base_specs);
    let value_col = specs
        .iter()
        .position(|s| s.id == format!("{}__{}", metric_column.id, agg.label()));

    // Per-image count of objects with >=1 colocalization partner, scoped to
    // this well (same reasoning as `compute_plate_matrix`'s `coloc_counts`).
    let mut coloc_counts: HashMap<String, usize> = HashMap::new();
    for object in &well_objects {
        if is_colocalized(&object.coloc_json) {
            *coloc_counts.entry(object.image_name.clone()).or_default() += 1;
        }
    }

    let total = well_rows * well_cols;
    let mut cells: Vec<WellCell> = (0..total).map(|_| empty_well_cell()).collect();
    let mut placed_any = false;
    let mut present_images: HashSet<String> = HashSet::new();

    for row in &rows {
        let image_name = row.values.first().cloned().unwrap_or_default();
        let count: usize = row.values.get(1).and_then(|v| v.parse().ok()).unwrap_or(0);
        let value = value_col
            .and_then(|i| row.values.get(i))
            .and_then(|v| v.parse::<f64>().ok());
        let Some(caps) = re.captures(&image_name) else {
            continue;
        };
        let Some(sub) = caps.get(4).and_then(|m| m.as_str().parse::<i32>().ok()) else {
            continue;
        };
        let Some(idx) = well_image_order.iter().position(|v| *v == sub) else {
            continue;
        };
        if idx >= total {
            continue;
        }
        let coloc_count = coloc_counts.get(&image_name).copied().unwrap_or(0);
        present_images.insert(image_name.clone());
        cells[idx] = WellCell {
            image_name,
            value,
            count,
            coloc_count,
        };
        placed_any = true;
    }

    // Field-of-view images in this well that produced no (matching) objects
    // still get their acquisition-order slot, occupied but valueless.
    for image in &well_images {
        if !present_images.insert(image.image_name.clone()) {
            continue;
        }
        let Some(caps) = re.captures(&image.image_name) else {
            continue;
        };
        let Some(sub) = caps.get(4).and_then(|m| m.as_str().parse::<i32>().ok()) else {
            continue;
        };
        let Some(idx) = well_image_order.iter().position(|v| *v == sub) else {
            continue;
        };
        if idx >= total {
            continue;
        }
        cells[idx] = WellCell {
            image_name: image.image_name.clone(),
            value: None,
            count: 0,
            coloc_count: 0,
        };
        placed_any = true;
    }

    if !placed_any {
        return None;
    }

    Some(WellMatrixResult {
        rows: well_rows,
        cols: well_cols,
        well_label: well_label.to_string(),
        cells,
    })
}

/// Suggests a "well" regex (group 1 = well id, group 4 = sub-position) by
/// testing a short list of candidate shapes against sample filenames, in
/// order from most to least specific. Accepts the first candidate matching
/// at least 80% of the (deduplicated) samples with at least 2 distinct well
/// ids — otherwise returns `None`.
///
/// This is explicitly a best-effort heuristic, not a general regex-mining
/// engine: callers must always show the result to the user as an editable
/// suggestion, never apply it silently.
pub fn suggest_regex(image_names: &[String]) -> Option<RegexSuggestion> {
    let mut seen = HashSet::new();
    let samples: Vec<&str> = image_names
        .iter()
        .map(|s| s.as_str())
        .filter(|s| seen.insert(*s))
        .collect();
    if samples.is_empty() {
        return None;
    }

    const CANDIDATES: &[&str] = &[
        r"(([A-Za-z]{1,2})([0-9]{1,3}))_([0-9]+)",
        r"(([A-Za-z]{1,2})([0-9]{1,3}))-([0-9]+)",
        r"(([A-Za-z]{1,2})([0-9]{1,3}))",
        r"(.*)_([0-9]+)",
    ];

    for pattern in CANDIDATES {
        let Ok(re) = regex::Regex::new(pattern) else {
            continue;
        };
        let mut matched = 0usize;
        let mut distinct = HashSet::new();
        for name in &samples {
            if let Some(caps) = re.captures(name)
                && let Some(g1) = caps.get(1)
            {
                matched += 1;
                distinct.insert(g1.as_str().to_string());
            }
        }
        let ratio = matched as f64 / samples.len() as f64;
        if ratio >= 0.8 && distinct.len() >= 2 {
            return Some(RegexSuggestion {
                pattern: pattern.to_string(),
                matched,
                total: samples.len(),
            });
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    fn obj(image_name: &str, image_rel_path: &str, area_px: u64) -> ObjectRow {
        ObjectRow {
            image_name: image_name.into(),
            image_rel_path: image_rel_path.into(),
            c_stack: None,
            z_stack: None,
            t_stack: None,
            object_id: "00000000-0000-0000-0000-000000000001".into(),
            seg_class_name: None,
            seg_class_id: None,
            object_class_name: vec![],
            object_class_id: vec![],
            parent_id: None,
            children: vec![],
            track_id: 0,
            centroid_x_px: 0.0,
            centroid_y_px: 0.0,
            centroid_x_nm: 0.0,
            centroid_y_nm: 0.0,
            area_px,
            area_nm2: area_px as f64,
            perimeter_px: 0.0,
            perimeter_nm: 0.0,
            circularity: 0.0,
            solidity: 0.0,
            aspect_ratio: 0.0,
            roundness: 0.0,
            compactness: 0.0,
            major_axis_px: 0.0,
            minor_axis_px: 0.0,
            touches_edge: false,
            intensities_json: "{}".into(),
            coloc_json: "{}".into(),
            bbox_px: [0, 0, 0, 0],
        }
    }

    fn obj_with_coloc(
        image_name: &str,
        image_rel_path: &str,
        area_px: u64,
        coloc_json: &str,
    ) -> ObjectRow {
        ObjectRow {
            coloc_json: coloc_json.into(),
            ..obj(image_name, image_rel_path, area_px)
        }
    }

    fn image(image_name: &str, image_rel_path: &str) -> ImageRow {
        ImageRow {
            image_name: image_name.into(),
            image_rel_path: image_rel_path.into(),
            status: "ok".into(),
            error_message: None,
        }
    }

    fn area_col() -> ColumnSpec {
        ColumnSpec {
            id: "area_px".into(),
            label: "Area (px)".into(),
            filterable: false,
            visible: true,
        }
    }

    // ---- row_col_from_label ----

    #[test]
    fn row_col_from_label_single_letter() {
        assert_eq!(row_col_from_label("A14"), Some((0, 13)));
        assert_eq!(row_col_from_label("H12"), Some((7, 11)));
    }

    #[test]
    fn row_col_from_label_multi_letter() {
        // Spreadsheet-style bijective base-26: Z=25, AA=26, AB=27.
        assert_eq!(row_col_from_label("AA7"), Some((26, 6)));
        assert_eq!(row_col_from_label("AB1"), Some((27, 0)));
    }

    #[test]
    fn row_col_from_label_not_well_shaped() {
        assert_eq!(row_col_from_label("plateA"), None);
        assert_eq!(row_col_from_label("14"), None);
        assert_eq!(row_col_from_label(""), None);
    }

    // ---- compute_plate_matrix: folder mode (sequential fill) ----

    #[test]
    fn plate_matrix_folder_mode_fills_sequentially() {
        let objects = vec![
            obj("img1.tif", "plateA/img1.tif", 10),
            obj("img2.tif", "plateA/img2.tif", 20),
            obj("img3.tif", "plateB/img3.tif", 100),
        ];
        let result = compute_plate_matrix(
            &objects,
            &[],
            GroupBy::Folder,
            "",
            AggFunc::Avg,
            &area_col(),
            2,
            2,
        );
        assert_eq!(result.rows, 2);
        assert_eq!(result.cols, 2);
        // Two distinct folders -> two occupied cells, filled row-major.
        assert_eq!(result.cells[0].label, "plateA");
        assert_eq!(result.cells[0].value, Some(15.0));
        assert_eq!(result.cells[0].count, 2);
        assert_eq!(result.cells[1].label, "plateB");
        assert_eq!(result.cells[1].value, Some(100.0));
        assert_eq!(result.cells[2], empty_plate_cell());
        assert_eq!(result.cells[3], empty_plate_cell());
    }

    // ---- compute_plate_matrix: regex mode (row/col placement) ----

    #[test]
    fn plate_matrix_regex_mode_places_by_well_coordinate() {
        let objects = vec![
            obj("my_file_01_A14_01.tif", "", 10),
            obj("my_file_01_A14_02.tif", "", 20),
            obj("my_file_01_A15_01.tif", "", 40),
        ];
        let regex = r"((.)([0-9]+))_([0-9]+)";
        let result = compute_plate_matrix(
            &objects,
            &[],
            GroupBy::Regex,
            regex,
            AggFunc::Avg,
            &area_col(),
            8,
            24,
        );
        // A14 -> row 0 ('A'), col 13 (14-1).
        let a14 = &result.cells[13];
        assert_eq!(a14.label, "A14");
        assert_eq!(a14.value, Some(15.0)); // avg(10, 20)
        assert_eq!(a14.count, 2);
        // A15 -> row 0, col 14.
        let a15 = &result.cells[14];
        assert_eq!(a15.label, "A15");
        assert_eq!(a15.value, Some(40.0));
        assert_eq!(a15.count, 1);
    }

    #[test]
    fn plate_matrix_regex_mode_drops_groups_beyond_capacity() {
        let objects = vec![obj("A99_01.tif", "", 1)];
        let regex = r"((.)([0-9]+))_([0-9]+)";
        // Only an 8x12 plate - "A99" decodes to row 0, col 98, out of bounds.
        let result = compute_plate_matrix(
            &objects,
            &[],
            GroupBy::Regex,
            regex,
            AggFunc::Avg,
            &area_col(),
            8,
            12,
        );
        assert!(result.cells.iter().all(|c| c.value.is_none()));
        // But it still reports how big a plate would be needed.
        assert_eq!(result.required_span, Some((1, 99)));
    }

    #[test]
    fn plate_matrix_required_span_reflects_real_sparse_plate_data() {
        // Mirrors a real dataset: only rows D/G populated, columns 2-19,
        // e.g. "D10_02.vsi", "G9_04.vsi" - and the plate size still at the
        // useless 1x1 default, so nothing fits without resizing.
        let objects = vec![obj("D19_02.vsi", "", 10), obj("G9_04.vsi", "", 20)];
        let regex = r"((.)([0-9]+))_([0-9]+)";
        let result = compute_plate_matrix(
            &objects,
            &[],
            GroupBy::Regex,
            regex,
            AggFunc::Avg,
            &area_col(),
            1,
            1,
        );
        assert!(result.cells.iter().all(|c| c.value.is_none()));
        // D -> row 4, col 19; G -> row 7, col 9. Needs at least 7 rows x 19 cols.
        assert_eq!(result.required_span, Some((7, 19)));
    }

    #[test]
    fn plate_matrix_required_span_none_for_folder_mode() {
        let objects = vec![obj("img1.tif", "plateA/img1.tif", 10)];
        let result = compute_plate_matrix(
            &objects,
            &[],
            GroupBy::Folder,
            "",
            AggFunc::Avg,
            &area_col(),
            2,
            2,
        );
        assert_eq!(result.required_span, None);
    }

    #[test]
    fn plate_matrix_reports_group_count_and_sample_when_regex_group1_is_not_well_shaped() {
        // A regex with an extra wrapping parenthesis around the whole
        // pattern - group(1) captures "D2_02" (well + sub-position) instead
        // of just "D2", so nothing decodes as a well coordinate even though
        // the regex clearly matched and grouped the data.
        let objects = vec![
            obj("D2_02.vsi", "", 10),
            obj("D2_03.vsi", "", 20),
            obj("G9_04.vsi", "", 30),
        ];
        let regex = r"((([A-Za-z]{1,2})([0-9]{1,3}))_([0-9]+))";
        let result = compute_plate_matrix(
            &objects,
            &[],
            GroupBy::Regex,
            regex,
            AggFunc::Avg,
            &area_col(),
            1,
            1,
        );
        assert_eq!(result.required_span, None);
        assert_eq!(result.group_count, 3); // one group per image, not per well
        assert_eq!(result.sample_label, Some("D2_02".to_string()));
    }

    // ---- compute_plate_matrix: all_images placeholders ----

    #[test]
    fn plate_matrix_all_images_places_occupied_but_empty_cell_for_a_well_with_no_objects() {
        let objects = vec![obj("A14_01.tif", "", 10)];
        let all_images = vec![
            image("A14_01.tif", "A14_01.tif"), // already has an object - must not duplicate
            image("A15_01.tif", "A15_01.tif"), // zero objects - needs a placeholder
        ];
        let regex = r"((.)([0-9]+))_([0-9]+)";
        let result = compute_plate_matrix(
            &objects,
            &all_images,
            GroupBy::Regex,
            regex,
            AggFunc::Avg,
            &area_col(),
            8,
            24,
        );

        let a14 = &result.cells[13]; // row 0 ('A'), col 13 (14-1)
        assert_eq!(a14.label, "A14");
        assert_eq!(a14.value, Some(10.0));
        assert_eq!(a14.count, 1);

        let a15 = &result.cells[14];
        assert_eq!(a15.label, "A15");
        assert_eq!(
            a15.value, None,
            "a well with zero objects must still be occupied, just valueless"
        );
        assert_eq!(a15.count, 0);
    }

    #[test]
    fn plate_matrix_coloc_count_counts_only_objects_with_a_colocalization_partner() {
        let objects = vec![
            obj_with_coloc(
                "A14_01.tif",
                "",
                10,
                r#"{"ClassB":["00000000-0000-0000-0000-000000000099"]}"#,
            ),
            obj("A14_02.tif", "", 20), // coloc_json defaults to "{}" -> not colocalized
        ];
        let regex = r"((.)([0-9]+))_([0-9]+)";
        let result = compute_plate_matrix(
            &objects,
            &[],
            GroupBy::Regex,
            regex,
            AggFunc::Avg,
            &area_col(),
            8,
            24,
        );
        let a14 = &result.cells[13];
        assert_eq!(a14.count, 2);
        assert_eq!(a14.coloc_count, 1);
    }

    // ---- compute_well_matrix ----

    #[test]
    fn well_matrix_places_fields_by_acquisition_order() {
        let objects = vec![
            obj("my_file_01_A14_01.tif", "", 10),
            obj("my_file_01_A14_02.tif", "", 20),
            obj("my_file_01_A14_03.tif", "", 30),
            obj("my_file_01_A15_01.tif", "", 999), // different well, excluded
        ];
        let regex = r"((.)([0-9]+))_([0-9]+)";
        // well_image_order maps acquisition position 1,2,3,4 to grid slots
        // 0,1,2,3 (row-major over a 2x2 well grid).
        let well_image_order = [1, 2, 3, 4];
        let result = compute_well_matrix(
            &objects,
            &[],
            regex,
            "A14",
            AggFunc::Avg,
            &area_col(),
            2,
            2,
            &well_image_order,
        )
        .expect("well has sub-position data");
        assert_eq!(result.well_label, "A14");
        assert_eq!(result.cells[0].image_name, "my_file_01_A14_01.tif");
        assert_eq!(result.cells[0].value, Some(10.0));
        assert_eq!(result.cells[1].image_name, "my_file_01_A14_02.tif");
        assert_eq!(result.cells[2].image_name, "my_file_01_A14_03.tif");
        assert_eq!(result.cells[3], empty_well_cell());
    }

    #[test]
    fn well_matrix_none_without_group4() {
        let objects = vec![obj("A14.tif", "", 10)];
        // Only 1 capture group - no sub-position.
        let regex = r"([A-Za-z][0-9]+)";
        assert!(
            compute_well_matrix(
                &objects,
                &[],
                regex,
                "A14",
                AggFunc::Avg,
                &area_col(),
                2,
                2,
                &[1, 2, 3, 4]
            )
            .is_none()
        );
    }

    #[test]
    fn well_matrix_none_when_well_not_found() {
        let objects = vec![obj("my_file_01_A14_01.tif", "", 10)];
        let regex = r"((.)([0-9]+))_([0-9]+)";
        assert!(
            compute_well_matrix(
                &objects,
                &[],
                regex,
                "Z99",
                AggFunc::Avg,
                &area_col(),
                2,
                2,
                &[1, 2, 3, 4]
            )
            .is_none()
        );
    }

    #[test]
    fn well_matrix_places_empty_slot_for_a_field_of_view_image_with_no_objects() {
        let objects = vec![obj("my_file_01_A14_01.tif", "", 10)];
        let all_images = vec![
            image("my_file_01_A14_01.tif", "my_file_01_A14_01.tif"),
            image("my_file_01_A14_02.tif", "my_file_01_A14_02.tif"), // zero objects
        ];
        let regex = r"((.)([0-9]+))_([0-9]+)";
        let well_image_order = [1, 2, 3, 4];
        let result = compute_well_matrix(
            &objects,
            &all_images,
            regex,
            "A14",
            AggFunc::Avg,
            &area_col(),
            2,
            2,
            &well_image_order,
        )
        .expect("well has sub-position data");

        assert_eq!(result.cells[0].image_name, "my_file_01_A14_01.tif");
        assert_eq!(result.cells[0].value, Some(10.0));

        assert_eq!(result.cells[1].image_name, "my_file_01_A14_02.tif");
        assert_eq!(
            result.cells[1].value, None,
            "an image with zero objects must still occupy its acquisition-order slot"
        );
        assert_eq!(result.cells[1].count, 0);
    }

    #[test]
    fn well_matrix_returns_a_grid_of_placeholders_when_the_well_has_images_but_zero_objects() {
        let all_images = vec![image("my_file_01_A14_01.tif", "my_file_01_A14_01.tif")];
        let regex = r"((.)([0-9]+))_([0-9]+)";
        let result = compute_well_matrix(
            &[],
            &all_images,
            regex,
            "A14",
            AggFunc::Avg,
            &area_col(),
            2,
            2,
            &[1, 2, 3, 4],
        )
        .expect("well has images even though none produced any objects");

        assert_eq!(result.cells[0].image_name, "my_file_01_A14_01.tif");
        assert_eq!(result.cells[0].value, None);
        assert_eq!(result.cells[0].count, 0);
    }

    // ---- suggest_regex ----

    #[test]
    fn suggest_regex_detects_well_shape() {
        let names: Vec<String> = [
            "my_file_01_A14_01.tif",
            "my_file_01_A14_02.tif",
            "my_file_01_A14_03.tif",
            "my_file_01_A15_01.tif",
            "my_file_01_A15_02.tif",
            "my_file_01_A16_03.tif",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect();

        let suggestion = suggest_regex(&names).expect("should detect a well-shaped pattern");
        assert_eq!(
            suggestion.pattern,
            r"(([A-Za-z]{1,2})([0-9]{1,3}))_([0-9]+)"
        );
        assert_eq!(suggestion.matched, 6);
        assert_eq!(suggestion.total, 6);

        // The suggested regex should actually decode row/col/sub correctly.
        let re = regex::Regex::new(&suggestion.pattern).unwrap();
        let caps = re.captures("my_file_01_A14_02.tif").unwrap();
        assert_eq!(caps.get(1).unwrap().as_str(), "A14");
        assert_eq!(caps.get(4).unwrap().as_str(), "02");
    }

    #[test]
    fn suggest_regex_none_for_unstructured_names() {
        let names: Vec<String> = vec!["foo.tif".into(), "bar.tif".into()];
        assert_eq!(suggest_regex(&names), None);
    }

    #[test]
    fn suggest_regex_empty_input() {
        assert_eq!(suggest_regex(&[]), None);
    }

    // ---- row_label ----

    #[test]
    fn row_label_single_letter() {
        assert_eq!(row_label(0), "A");
        assert_eq!(row_label(25), "Z");
    }

    #[test]
    fn row_label_multi_letter() {
        assert_eq!(row_label(26), "AA");
        assert_eq!(row_label(27), "AB");
    }

    #[test]
    fn row_label_round_trips_with_row_col_from_label() {
        for i in [0usize, 1, 25, 26, 27, 51, 52, 700] {
            let label = format!("{}5", row_label(i));
            assert_eq!(row_col_from_label(&label), Some((i, 4)));
        }
    }

    // ---- resolve_range ----

    #[test]
    fn resolve_range_auto_spans_zero_to_max() {
        assert_eq!(resolve_range(&[1.0, 5.0, 3.0], true, 0.0, 0.0), (0.0, 5.0));
    }

    #[test]
    fn resolve_range_manual_uses_given_bounds() {
        assert_eq!(resolve_range(&[1.0, 5.0], false, 2.0, 10.0), (2.0, 10.0));
    }

    #[test]
    fn resolve_range_manual_degenerate_collapses_to_hairline() {
        let (lo, hi) = resolve_range(&[1.0], false, 5.0, 5.0);
        assert_eq!(lo, 5.0);
        assert!(hi > lo);
    }

    // ---- compute_well_matrix vs. the Table view's grouped-by-Image aggregate ----
    //
    // The Matrix (Well) view's per-image aggregate and the Table view's
    // "group by Image" aggregate are computed by genuinely different code
    // paths for the *same* concept (one aggregated value per image):
    // `compute_well_matrix` here calls `aggregate_rows` (pure Rust) on
    // objects already loaded into memory, while the Table view calls
    // `aggregate_objects_sql` (DuckDB pushdown) directly against the file.
    // A prior parity test already proved `aggregate_rows`/`aggregate_objects_sql`
    // agree in the abstract; this one instead drives the *actual* production
    // entry point (`compute_well_matrix`) side by side with the Table's, so a
    // bug specific to well-filtering/placement in `compute_well_matrix`
    // itself (not just the underlying aggregation) would also be caught.

    use crate::results::results_loader::{
        DatabaseFilter, ResultsLoader, aggregate_objects_sql, build_column_specs,
    };
    use crate::results::test_support::seed_results_db;

    fn circularity_col() -> ColumnSpec {
        ColumnSpec {
            id: "circularity".into(),
            label: "Circularity".into(),
            filterable: false,
            visible: true,
        }
    }

    #[test]
    fn compute_well_matrix_value_matches_the_table_views_grouped_by_image_aggregate() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("results.duckdb");
        seed_results_db(&db_path); // baseline rows the well regex won't match

        // Three objects sharing one field-of-view image in well "A14" - a
        // circularity average with more decimal digits than either display's
        // precision (0.85333...) so any precision-related divergence between
        // the two paths wouldn't hide behind a suspiciously round number.
        let conn = duckdb::Connection::open(&db_path).unwrap();
        for (idx, circularity) in [0.85_f64, 0.86, 0.85].into_iter().enumerate() {
            conn.execute(
                "INSERT INTO objects (
                    image_name, image_rel_path, object_id, seg_class_name, seg_class_id,
                    object_class_name, object_class_id, track_id,
                    centroid_x_px, centroid_y_px, centroid_x_nm, centroid_y_nm,
                    bbox_xmin_px, bbox_ymin_px, bbox_xmax_px, bbox_ymax_px,
                    bbox_xmin_nm, bbox_ymin_nm, bbox_xmax_nm, bbox_ymax_nm,
                    area_px, area_nm2, perimeter_px, perimeter_nm,
                    circularity, solidity, aspect_ratio, roundness, compactness,
                    major_axis_px, minor_axis_px, touches_edge,
                    pixel_size_x_nm, pixel_size_y_nm, pixel_size_z_nm,
                    intensities_json, coloc_json
                ) VALUES (
                    'my_file_01_A14_01.tif', 'my_file_01_A14_01.tif', ?, 'ClassA', 1,
                    '[\"ClassA\"]', '[1]', 0,
                    0, 0, 0, 0, 0, 0, 10, 10, 0, 0, 0, 0,
                    10, 10.0, 40, 40,
                    ?, 1.0, 1.0, 1.0, 1.0, 10, 10, false,
                    1.0, 1.0, 1.0, '{}', '{}'
                )",
                duckdb::params![
                    format!("00000000-0000-0000-0000-0000000000{idx:02}"),
                    circularity
                ],
            )
            .unwrap();
        }
        drop(conn);

        let loader = ResultsLoader::new(&db_path);
        let all_objects = loader.get_objects(DatabaseFilter::default()).unwrap();

        // ---- Matrix (Well) view's path: compute_well_matrix ----
        let regex = r"((.)([0-9]+))_([0-9]+)";
        let well_result = compute_well_matrix(
            &all_objects,
            &[],
            regex,
            "A14",
            AggFunc::Avg,
            &circularity_col(),
            1,
            1,
            &[1],
        )
        .expect("well A14 has sub-position data");
        let matrix_value = well_result.cells[0].value.expect("cell must have a value");
        assert_eq!(well_result.cells[0].count, 3);

        // ---- Table view's path: aggregate_objects_sql, grouped by Image ----
        let base_specs = build_column_specs(&[], &[]);
        let config = GroupConfig {
            group_by: GroupBy::Image,
            aggs: vec![AggFunc::Avg],
            ..Default::default()
        };
        let (specs, rows) =
            aggregate_objects_sql(&loader, DatabaseFilter::default(), &config, &base_specs)
                .unwrap();
        let value_col = specs
            .iter()
            .position(|s| s.id == "circularity__avg")
            .unwrap();
        let table_row = rows
            .iter()
            .find(|r| r.values[0] == "my_file_01_A14_01.tif")
            .expect("the well's image must appear as its own Table group");
        let table_value: f64 = table_row.values[value_col].parse().unwrap();

        // Both paths must agree exactly. Note this is *not* the raw average
        // (0.85333...): both `aggregate_rows` (Matrix) and
        // `aggregate_objects_sql` (Table) round to `metric_precision`
        // (3 decimals for circularity) before the value is ever handed back
        // as a `DisplayRow` string - `compute_well_matrix` re-parses that
        // already-rounded string rather than keeping a full-precision float.
        // So "equal" here means both independently rounded to 0.853, not
        // that neither lost precision relative to the true mean.
        assert_eq!(matrix_value, 0.853);
        assert_eq!(table_value, 0.853);
        assert!(
            (matrix_value - table_value).abs() < 1e-9,
            "Matrix Well value {matrix_value} must equal the Table's grouped-by-Image value {table_value}"
        );
    }
}
