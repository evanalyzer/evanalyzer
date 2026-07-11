use evanalyzer_cfg::core_types::InternalErrors;
pub use evanalyzer_core::RoiRow;
pub use evanalyzer_core::{coloc_filter_label_any, coloc_filter_label_no, coloc_filter_label_with};
use evanalyzer_core::{AggregateSpec, DuckDbReader, GroupKeyMode, RoiFilter};
use std::collections::BTreeMap;
use std::path::PathBuf;

// ---------------------------------------------------------------------------
// Column specification — shared between GUI, CLI, and export
// ---------------------------------------------------------------------------

/// Describes one column in the results table.
#[derive(Debug, Clone)]
pub struct ColumnSpec {
    pub id: String,
    pub label: String,
    pub filterable: bool,
    /// Whether this column's data should be populated (visible in UI / included in export).
    /// When false, `to_display_row` writes an empty string for this column without
    /// doing any parsing work, and `DatabaseFilter::needs_intensities` can be set
    /// to false if no channel column is visible.
    pub visible: bool,
}

/// One display-ready row: `values[i]` corresponds to `ColumnSpec[i]`.
#[derive(Debug, Clone)]
pub struct DisplayRow {
    /// Positional display id (`row_idx + 1`) — only meaningful as a stable
    /// sort/highlight key *within a single reload*, never as a lookup key
    /// across pages: it resets per page/reload and is not a database id. Row
    /// selection uses `object_id` instead (see `ResultsRow`/`RowWindow` in
    /// `crates/gui/src/editor/results_table_controller.rs`).
    pub roi_id: i32,
    /// The source ROI's stable database id (`RoiRow::object_id`), or `""` for
    /// rows with no single source ROI (grouped/aggregated rows) or where
    /// selection is intentionally disabled (the coloc-detail flat table).
    pub object_id: String,
    /// Pre-formatted string values, one per `ColumnSpec` (in the same order).
    /// Hidden columns have an empty string here.
    pub values: Vec<String>,
    /// Alternating zebra-stripe flag: toggles once per "logical row group".
    /// For the per-ROI and aggregated views a logical group is just the row
    /// itself, so this alternates every row. For the coloc-detail flat table,
    /// every flattened row sharing one source ROI gets the same value (it
    /// toggles when moving to the next source ROI instead), so the rows
    /// belonging to one source ROI read as a single visual band regardless of
    /// how many colocalizing partners it has.
    pub stripe: bool,
}

// ---------------------------------------------------------------------------
// Column helpers — channel discovery and spec construction
// ---------------------------------------------------------------------------

/// Discovers which channel indices appear in `intensities_json` across the
/// provided rows. The returned list is sorted numerically.
pub fn discover_channels(rois: &[RoiRow]) -> Vec<i32> {
    let mut channels = std::collections::BTreeSet::new();
    for roi in rois {
        if roi.intensities_json.is_empty() || roi.intensities_json == "{}" {
            continue;
        }
        if let Ok(val) = serde_json::from_str::<serde_json::Value>(&roi.intensities_json) {
            if let Some(obj) = val.as_object() {
                for key in obj.keys() {
                    if let Ok(ch) = key.parse::<i32>() {
                        channels.insert(ch);
                    }
                }
            }
        }
    }
    channels.into_iter().collect()
}

/// Builds the full ordered list of [`ColumnSpec`] for the given channel list
/// and the set of known colocalization-partner classes. All columns start as
/// visible.
///
/// Each partner class contributes two columns: `coloc_partner__{class}__count`
/// (the number of that class's ROIs this ROI colocalizes with — numeric,
/// aggregatable when grouping) and `coloc_partner__{class}__ids` (the
/// colocalizing ROIs' object ids, comma-joined — display-only). The `__`
/// separators are safe even if `class` itself contains `__`: parsing only ever
/// strips the fixed `coloc_partner__` prefix and a fixed `__count`/`__ids`/
/// `__<agg>` suffix, never splits on `__` in the middle.
pub fn build_column_specs(channels: &[i32], coloc_partner_classes: &[String]) -> Vec<ColumnSpec> {
    let mut cols = vec![
        ColumnSpec {
            id: "roi_id".into(),
            label: "ROI ID".into(),
            filterable: false,
            visible: true,
        },
        ColumnSpec {
            id: "image".into(),
            label: "Image".into(),
            filterable: true,
            visible: true,
        },
        ColumnSpec {
            id: "class".into(),
            label: "Class".into(),
            filterable: true,
            visible: true,
        },
        ColumnSpec {
            id: "area_px".into(),
            label: "Area (px\u{00B2})".into(),
            filterable: false,
            visible: true,
        },
        ColumnSpec {
            id: "area_nm2".into(),
            label: "Area (nm\u{00B2})".into(),
            filterable: false,
            visible: true,
        },
        ColumnSpec {
            id: "circularity".into(),
            label: "Circularity".into(),
            filterable: false,
            visible: true,
        },
        ColumnSpec {
            id: "colocalized".into(),
            label: "Colocalized".into(),
            filterable: true,
            visible: true,
        },
    ];
    for &ch in channels {
        cols.push(ColumnSpec {
            id: format!("ch{ch}_min_bit"),
            label: format!("Ch{ch} Min (bit)"),
            filterable: false,
            visible: true,
        });
        cols.push(ColumnSpec {
            id: format!("ch{ch}_max_bit"),
            label: format!("Ch{ch} Max (bit)"),
            filterable: false,
            visible: true,
        });
        cols.push(ColumnSpec {
            id: format!("ch{ch}_avg_bit"),
            label: format!("Ch{ch} Avg (bit)"),
            filterable: false,
            visible: true,
        });
        cols.push(ColumnSpec {
            id: format!("ch{ch}_sum_bit"),
            label: format!("Ch{ch} Sum (bit)"),
            filterable: false,
            visible: true,
        });
    }
    for class in coloc_partner_classes {
        cols.push(ColumnSpec {
            id: format!("coloc_partner__{class}__count"),
            label: format!("Coloc w/ {class} (#)"),
            filterable: false,
            visible: true,
        });
        cols.push(ColumnSpec {
            id: format!("coloc_partner__{class}__ids"),
            label: format!("Coloc w/ {class} (IDs)"),
            filterable: false,
            visible: true,
        });
    }
    cols
}

// ---------------------------------------------------------------------------
// Row conversion
// ---------------------------------------------------------------------------

/// Converts a [`RoiRow`] into a [`DisplayRow`] using the given column specs.
/// Only visible columns have their values computed; hidden columns get `""`.
/// `intensities_json` is parsed at most once per call, and only if at least
/// one channel column is visible.
pub fn to_display_row(row_idx: usize, roi: &RoiRow, columns: &[ColumnSpec]) -> DisplayRow {
    let needs_intensities = columns.iter().any(|c| c.visible && c.id.starts_with("ch"));
    let needs_coloc_partner = columns
        .iter()
        .any(|c| c.visible && c.id.starts_with("coloc_partner__"));

    let intensities: BTreeMap<i32, (f64, f64, f64, f64)> = if needs_intensities {
        parse_intensities(&roi.intensities_json)
    } else {
        BTreeMap::new()
    };
    let coloc_partners: BTreeMap<String, Vec<String>> = if needs_coloc_partner {
        parse_coloc_partners(&roi.coloc_json)
    } else {
        BTreeMap::new()
    };

    let class = compute_class(roi);

    let values = columns
        .iter()
        .map(|col| {
            if !col.visible {
                return String::new();
            }
            match col.id.as_str() {
                "roi_id" => roi.object_id.to_string(),
                "image" => roi.image_name.clone(),
                "class" => class.clone(),
                "area_px" => roi.area_px.to_string(),
                "area_nm2" => format!("{:.2}", roi.area_nm2),
                "circularity" => format!("{:.3}", roi.circularity),
                "colocalized" => coloc_label(is_colocalized(&roi.coloc_json)),
                id if id.starts_with("coloc_partner__") => {
                    coloc_partner_value(id, &coloc_partners)
                }
                id => channel_value(id, &intensities),
            }
        })
        .collect();

    DisplayRow {
        roi_id: (row_idx + 1) as i32,
        object_id: roi.object_id.clone(),
        values,
        stripe: row_idx % 2 == 0,
    }
}

pub(crate) fn compute_class(roi: &RoiRow) -> String {
    if roi.object_class_name.is_empty() {
        roi.seg_class_name.clone().unwrap_or_default()
    } else {
        roi.object_class_name.join(", ")
    }
}

/// True if `coloc_json` (the per-ROI `colocalized_with` map, keyed by partner class label)
/// records at least one colocalization partner for any class.
pub(crate) fn is_colocalized(coloc_json: &str) -> bool {
    if coloc_json.is_empty() || coloc_json == "{}" {
        return false;
    }
    let Ok(val) = serde_json::from_str::<serde_json::Value>(coloc_json) else {
        return false;
    };
    let Some(obj) = val.as_object() else {
        return false;
    };
    obj.values()
        .any(|v| v.as_array().is_some_and(|a| !a.is_empty()))
}

pub(crate) fn coloc_label(colocalized: bool) -> String {
    if colocalized { "Yes" } else { "No" }.to_string()
}

/// Parses `intensities_json` into a map of channel → (min_scaled, max_scaled, mean_scaled, sum_scaled).
pub(crate) fn parse_intensities(json: &str) -> BTreeMap<i32, (f64, f64, f64, f64)> {
    let mut result = BTreeMap::new();
    if json.is_empty() || json == "{}" {
        return result;
    }
    let Ok(val) = serde_json::from_str::<serde_json::Value>(json) else {
        return result;
    };
    let Some(obj) = val.as_object() else {
        return result;
    };
    for (ch_str, stats) in obj {
        let Ok(ch) = ch_str.parse::<i32>() else {
            continue;
        };
        result.insert(
            ch,
            (
                stats["min_scaled"].as_f64().unwrap_or(0.0),
                stats["max_scaled"].as_f64().unwrap_or(0.0),
                stats["mean_scaled"].as_f64().unwrap_or(0.0),
                stats["sum_scaled"].as_f64().unwrap_or(0.0),
            ),
        );
    }
    result
}

/// Extracts a formatted value for channel-intensity columns (e.g. `ch0_min_bit`).
fn channel_value(col_id: &str, intensities: &BTreeMap<i32, (f64, f64, f64, f64)>) -> String {
    let rest = match col_id.strip_prefix("ch") {
        Some(r) => r,
        None => return String::new(),
    };
    let under = match rest.find('_') {
        Some(i) => i,
        None => return String::new(),
    };
    let ch: i32 = match rest[..under].parse() {
        Ok(n) => n,
        Err(_) => return String::new(),
    };
    let stat = &rest[under + 1..];
    let Some(&(min, max, avg, sum)) = intensities.get(&ch) else {
        return String::new();
    };
    match stat {
        "min_bit" => format!("{:.0}", min),
        "max_bit" => format!("{:.0}", max),
        "avg_bit" => format!("{:.1}", avg),
        "sum_bit" => format!("{:.0}", sum),
        _ => String::new(),
    }
}

/// Splits a `coloc_partner__{class}__{suffix}` column id into `(class, suffix)`,
/// where `suffix` is `"count"`, `"ids"`, or (in a grouped column) an aggregate
/// label. Only ever strips the fixed prefix and the *last* `__`-separated
/// segment, so a class label containing `__` round-trips correctly.
fn split_coloc_partner_id(col_id: &str) -> Option<(&str, &str)> {
    let rest = col_id.strip_prefix("coloc_partner__")?;
    rest.rsplit_once("__")
}

/// Parses `coloc_json` into a map of partner class label → colocalizing object ids.
pub(crate) fn parse_coloc_partners(json: &str) -> BTreeMap<String, Vec<String>> {
    let mut result = BTreeMap::new();
    if json.is_empty() || json == "{}" {
        return result;
    }
    let Ok(val) = serde_json::from_str::<serde_json::Value>(json) else {
        return result;
    };
    let Some(obj) = val.as_object() else {
        return result;
    };
    for (class, ids) in obj {
        let Some(ids) = ids.as_array() else { continue };
        // Dedup: older result files may carry a repeated partner id for a class
        // (a multi-class source ROI could be recomputed against the same partner
        // more than once) — collapse those before flattening/counting, so a
        // single overlap never shows up as more than one row/unit.
        let mut ids: Vec<String> = ids
            .iter()
            .filter_map(|v| v.as_str().map(str::to_string))
            .collect();
        ids.sort();
        ids.dedup();
        if !ids.is_empty() {
            result.insert(class.clone(), ids);
        }
    }
    result
}

/// Every partner-ROI object id referenced by any of `source_rois`'
/// `coloc_json`, deduplicated. Used to fetch only the colocalization partner
/// ROIs actually needed for one page of source rows (via
/// `DatabaseFilter::object_id_filter`), instead of loading every ROI in the
/// image.
pub fn coloc_partner_ids(source_rois: &[RoiRow]) -> Vec<String> {
    let mut ids = std::collections::BTreeSet::new();
    for roi in source_rois {
        for partner_ids in parse_coloc_partners(&roi.coloc_json).into_values() {
            ids.extend(partner_ids);
        }
    }
    ids.into_iter().collect()
}

/// Extracts a formatted value for `coloc_partner__{class}__count` /
/// `coloc_partner__{class}__ids` columns.
fn coloc_partner_value(col_id: &str, partners: &BTreeMap<String, Vec<String>>) -> String {
    let Some((class, suffix)) = split_coloc_partner_id(col_id) else {
        return String::new();
    };
    let ids = partners.get(class).map(Vec::as_slice).unwrap_or(&[]);
    match suffix {
        "count" => ids.len().to_string(),
        "ids" => ids.join(", "),
        _ => String::new(),
    }
}

// ---------------------------------------------------------------------------
// Colocalization detail flat table
// ---------------------------------------------------------------------------

/// Builds column specs for the colocalization detail flat table.
///
/// Source ROI columns come first (fixed fields + per-channel intensities).
/// Then, for each partner class, a group of property columns is appended:
/// `coloc_detail__{cls}__roi_id`, `coloc_detail__{cls}__area_px`, …,
/// `coloc_detail__{cls}__ch{N}_avg_bit`, etc.
///
/// The `__` separator uses `rsplit_once`, so class names that themselves
/// contain `__` round-trip correctly (same convention as the normal
/// `coloc_partner__` columns).
pub fn build_coloc_detail_column_specs(
    channels: &[i32],
    coloc_partner_classes: &[String],
) -> Vec<ColumnSpec> {
    let mk = |id: &str, label: &str| ColumnSpec {
        id: id.into(),
        label: label.into(),
        filterable: false,
        visible: true,
    };
    // `image`/`class` are filterable here too, matching `build_column_specs` —
    // without this, the per-column filter icon (and its image/class checkbox
    // popup) never renders on those headers while the coloc-detail view is
    // active, since the header only shows a filter affordance when the
    // column's own spec says it's filterable.
    let mut cols = vec![
        mk("roi_id", "ROI ID"),
        ColumnSpec { id: "image".into(), label: "Image".into(), filterable: true, visible: true },
        ColumnSpec { id: "class".into(), label: "Class".into(), filterable: true, visible: true },
        mk("area_px", "Area (px\u{00B2})"),
        mk("area_nm2", "Area (nm\u{00B2})"),
        mk("circularity", "Circularity"),
    ];
    for &ch in channels {
        cols.push(mk(&format!("ch{ch}_min_bit"), &format!("Ch{ch} Min (bit)")));
        cols.push(mk(&format!("ch{ch}_max_bit"), &format!("Ch{ch} Max (bit)")));
        cols.push(mk(&format!("ch{ch}_avg_bit"), &format!("Ch{ch} Avg (bit)")));
        cols.push(mk(&format!("ch{ch}_sum_bit"), &format!("Ch{ch} Sum (bit)")));
    }
    // One property group per partner class — mirrors the normal table's per-class columns.
    for cls in coloc_partner_classes {
        let p = |field: &str| format!("coloc_detail__{cls}__{field}");
        let l = |label: &str| format!("Coloc {cls} {label}");
        cols.push(mk(&p("roi_id"), &l("ROI ID")));
        cols.push(mk(&p("area_px"), &l("Area (px\u{00B2})")));
        cols.push(mk(&p("area_nm2"), &l("Area (nm\u{00B2})")));
        cols.push(mk(&p("circularity"), &l("Circularity")));
        for &ch in channels {
            cols.push(mk(&p(&format!("ch{ch}_min_bit")), &l(&format!("Ch{ch} Min (bit)"))));
            cols.push(mk(&p(&format!("ch{ch}_max_bit")), &l(&format!("Ch{ch} Max (bit)"))));
            cols.push(mk(&p(&format!("ch{ch}_avg_bit")), &l(&format!("Ch{ch} Avg (bit)"))));
            cols.push(mk(&p(&format!("ch{ch}_sum_bit")), &l(&format!("Ch{ch} Sum (bit)"))));
        }
    }
    cols
}

/// Flattens `source_rois` into one [`DisplayRow`] per (source ROI, colocalized partner)
/// **ROI** — i.e. exactly one row per distinct partner object, no matter how many
/// classes it's recorded under. `coloc_json` is keyed by partner *class label*, so a
/// partner ROI that itself carries more than one coloc-target class can legitimately
/// appear under two different class keys for the same source; naively iterating
/// `(class, id)` pairs would then emit two rows for what is really one colocalization.
/// Source ROIs with no partners produce a single row with all partner columns empty.
/// Each row fills the column group(s) matching every class the partner is recorded
/// under; all other partner class groups are left empty — matching the structure of
/// the normal table's per-class coloc columns.
///
/// `partner_lookup` should be a *superset* of `source_rois` that includes all ROIs that
/// may appear as coloc partners (i.e. loaded without class/coloc filter). When a partner
/// UUID is not found in `partner_lookup` its property columns are left empty.
pub fn flatten_coloc_rows(
    source_rois: &[RoiRow],
    partner_lookup: &[RoiRow],
    specs: &[ColumnSpec],
) -> Vec<DisplayRow> {
    use std::collections::HashMap;
    // Normalize UUIDs to lowercase so case mismatches between coloc_json and
    // object_id storage do not silently drop partners.
    let roi_map: HashMap<String, &RoiRow> =
        partner_lookup.iter().map(|r| (r.object_id.to_lowercase(), r)).collect();

    let mut out = Vec::new();
    let mut idx = 0usize;

    for (src_idx, src) in source_rois.iter().enumerate() {
        // Every row flattened from this source ROI shares one stripe value —
        // it's the *source*'s position that toggles the zebra stripe, not the
        // flattened row's, so a source with many colocalizing partners still
        // reads as a single visual band rather than flickering per partner.
        let stripe = src_idx % 2 == 0;
        let partners = parse_coloc_partners(&src.coloc_json);
        if partners.is_empty() {
            out.push(DisplayRow {
                roi_id: (idx + 1) as i32,
                object_id: String::new(), // no row selection in the coloc-detail view
                values: coloc_detail_values(src, &[], None, specs),
                stripe,
            });
            idx += 1;
        } else {
            // Group by unique partner id first, collecting every class label it's
            // recorded under, so one physical partner never produces more than
            // one row.
            let mut per_partner: BTreeMap<&str, Vec<&str>> = BTreeMap::new();
            for (cls, ids) in &partners {
                for id in ids {
                    per_partner.entry(id.as_str()).or_default().push(cls.as_str());
                }
            }
            for (id, classes) in &per_partner {
                let partner = roi_map.get(&id.to_lowercase()).copied();
                out.push(DisplayRow {
                    roi_id: (idx + 1) as i32,
                    object_id: String::new(), // no row selection in the coloc-detail view
                    values: coloc_detail_values(src, classes, partner, specs),
                    stripe,
                });
                idx += 1;
            }
        }
    }
    out
}

/// Shown in a coloc-detail row's partner "ROI ID" column when the source's
/// `coloc_json` references a partner id that doesn't match any ROI actually
/// present in the file (e.g. a stale colocalization result left over from an
/// object that was later removed) — distinguishes "referenced but missing"
/// from "no partner recorded at all" (which renders as a plain empty cell).
const UNRESOLVED_PARTNER_MARKER: &str = "(missing)";

/// Builds one row's values for the coloc detail flat table.
/// `active_partner_classes` lists every class label the partner in this row is
/// recorded under; only column groups matching one of those classes are filled —
/// all others are empty.
fn coloc_detail_values(
    src: &RoiRow,
    active_partner_classes: &[&str],
    partner: Option<&RoiRow>,
    specs: &[ColumnSpec],
) -> Vec<String> {
    let needs_src_int = specs
        .iter()
        .any(|c| c.visible && c.id.starts_with("ch") && !c.id.starts_with("coloc_detail__"));
    let src_int = if needs_src_int {
        parse_intensities(&src.intensities_json)
    } else {
        BTreeMap::new()
    };
    // Parse partner intensities lazily — only if there is a partner for this row.
    let par_int: BTreeMap<i32, (f64, f64, f64, f64)> = partner
        .map(|p| parse_intensities(&p.intensities_json))
        .unwrap_or_default();

    let src_class = compute_class(src);

    specs
        .iter()
        .map(|col| {
            if !col.visible {
                return String::new();
            }
            match col.id.as_str() {
                // --- source ROI fixed columns ---
                "roi_id" => src.object_id.clone(),
                "image" => src.image_name.clone(),
                "class" => src_class.clone(),
                "area_px" => src.area_px.to_string(),
                "area_nm2" => format!("{:.2}", src.area_nm2),
                "circularity" => format!("{:.3}", src.circularity),
                // --- source ROI channel columns ---
                id if id.starts_with("ch") => channel_value(id, &src_int),
                // --- per-class partner columns: coloc_detail__{cls}__{field} ---
                id if id.starts_with("coloc_detail__") => {
                    let rest = id.strip_prefix("coloc_detail__").unwrap();
                    // rsplit_once so class names containing "__" round-trip correctly.
                    let Some((col_cls, field)) = rest.rsplit_once("__") else {
                        return String::new();
                    };
                    // Only fill column groups for classes the partner is actually
                    // recorded under.
                    if !active_partner_classes.contains(&col_cls) {
                        return String::new();
                    }
                    match field {
                        // A class this row is legitimately active for (checked
                        // above) but with no resolved `partner` means the id
                        // referenced in the source's `coloc_json` doesn't match
                        // any ROI in this file (e.g. a stale reference to an
                        // object that no longer exists) — flag it rather than
                        // rendering a blank cell that looks like "no partner".
                        "roi_id" => partner
                            .map(|p| p.object_id.clone())
                            .unwrap_or_else(|| UNRESOLVED_PARTNER_MARKER.to_string()),
                        "area_px" => {
                            partner.map(|p| p.area_px.to_string()).unwrap_or_default()
                        }
                        "area_nm2" => {
                            partner.map(|p| format!("{:.2}", p.area_nm2)).unwrap_or_default()
                        }
                        "circularity" => {
                            partner.map(|p| format!("{:.3}", p.circularity)).unwrap_or_default()
                        }
                        ch_id => channel_value(ch_id, &par_int),
                    }
                }
                _ => String::new(),
            }
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Database filter and loader
// ---------------------------------------------------------------------------

#[derive(Clone)]
pub struct DatabaseFilter {
    /// `None` = no image filter; `Some([])` = nothing passes; `Some([..])` = restrict to these.
    pub image_filter: Option<Vec<String>>,
    /// `None` = no class filter; `Some([])` = nothing passes; `Some([..])` = restrict to these.
    pub class_filter: Option<Vec<String>>,
    /// `None` = no coloc filter; `Some([])` = nothing passes; `Some([..])` = restrict to
    /// rows whose colocalization status ("Yes"/"No") is in this set.
    pub coloc_filter: Option<Vec<String>>,
    /// Restricts to rows whose `object_id` is in this set. Typically used *alone*
    /// (no image/class/coloc filter set alongside it) to fetch an exact, known
    /// set of ROIs — e.g. the colocalization partners referenced by one page of
    /// source ROIs — without re-deriving or re-checking the source-side filters.
    pub object_id_filter: Option<Vec<String>>,
    /// Restricts to ROIs from this single time-frame index, or `None` for every frame.
    pub t_stack_filter: Option<i32>,
    /// Restricts to ROIs from this single Z-depth index, or `None` for every depth.
    pub z_stack_filter: Option<i32>,
    pub page_size: usize,
    pub page: usize,
    /// When false the database omits `intensities_json` from the SELECT, avoiding
    /// JSON parsing cost when no channel columns are visible.
    pub needs_intensities: bool,
    /// Results-table column id to sort by; `None` keeps the default order.
    pub sort_column: Option<String>,
    pub sort_ascending: bool,
}

/// Converts the app-level filter into the core crate's query filter.
/// Shared by `ResultsLoader::get_rois` and, for callers that hold their own
/// `DuckDbReader` (see [`ResultsLoader::open_reader`]), by `results_exporter.rs`.
pub(crate) fn to_roi_filter(filter: DatabaseFilter) -> RoiFilter {
    RoiFilter {
        image_filter: filter.image_filter,
        class_filter: filter.class_filter,
        coloc_filter: filter.coloc_filter,
        object_id_filter: filter.object_id_filter,
        t_stack_filter: filter.t_stack_filter,
        z_stack_filter: filter.z_stack_filter,
        page_size: filter.page_size,
        page: filter.page,
        fetch_intensities: filter.needs_intensities,
        sort_column: filter.sort_column,
        sort_ascending: filter.sort_ascending,
    }
}

impl Default for DatabaseFilter {
    fn default() -> Self {
        Self {
            image_filter: None,
            class_filter: None,
            coloc_filter: None,
            object_id_filter: None,
            t_stack_filter: None,
            z_stack_filter: None,
            page_size: 500,
            page: 0,
            needs_intensities: true,
            sort_column: None,
            sort_ascending: true,
        }
    }
}

// ---------------------------------------------------------------------------
// Grouping & aggregation
// ---------------------------------------------------------------------------

/// How rows are bucketed into groups before aggregation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GroupBy {
    /// No grouping — caller should display the raw per-ROI rows.
    None,
    /// One group per distinct `image_name`.
    Image,
    /// One group per folder (the directory portion of `image_rel_path`).
    Folder,
    /// One group per key extracted from `image_name` by a regex. The first
    /// capture group is used as the key; if the pattern has no capture group
    /// the whole match is used. Rows whose name does not match are dropped.
    Regex,
}

/// Aggregate applied to every numeric column when grouping is active.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AggFunc {
    Min,
    Max,
    Avg,
    Median,
    Stdev,
    Sum,
}

impl AggFunc {
    /// Short suffix shown in grouped column headers, e.g. `Area (px²) [median]`.
    /// Also used as the id fragment in grouped column ids (`area_px__median`).
    pub fn label(self) -> &'static str {
        match self {
            AggFunc::Min => "min",
            AggFunc::Max => "max",
            AggFunc::Avg => "avg",
            AggFunc::Median => "median",
            AggFunc::Stdev => "stdev",
            AggFunc::Sum => "sum",
        }
    }

    /// Inverse of [`AggFunc::label`].
    fn from_label(s: &str) -> Option<AggFunc> {
        Some(match s {
            "min" => AggFunc::Min,
            "max" => AggFunc::Max,
            "avg" => AggFunc::Avg,
            "median" => AggFunc::Median,
            "stdev" => AggFunc::Stdev,
            "sum" => AggFunc::Sum,
            _ => return None,
        })
    }

    /// The DuckDB aggregate function name computing the same thing as
    /// [`apply_agg`] — `STDDEV_SAMP` for `Stdev` since `apply_agg`'s formula is
    /// the sample (n-1) standard deviation, and DuckDB's `MEDIAN` uses the same
    /// linear interpolation between the two middle values for an even-length
    /// input that `apply_agg`'s `Median` branch does. Used by `aggregate_rois_sql`.
    pub(crate) fn sql_fn(self) -> &'static str {
        match self {
            AggFunc::Min => "MIN",
            AggFunc::Max => "MAX",
            AggFunc::Avg => "AVG",
            AggFunc::Median => "MEDIAN",
            AggFunc::Stdev => "STDDEV_SAMP",
            AggFunc::Sum => "SUM",
        }
    }
}

/// Configuration for a grouped/aggregated results view.
#[derive(Debug, Clone)]
pub struct GroupConfig {
    pub group_by: GroupBy,
    /// Regex pattern, only used when `group_by == GroupBy::Regex`.
    pub regex: String,
    /// Aggregates to compute. Each visible numeric metric gets one grouped
    /// column per entry here (e.g. selecting Min+Max+Avg yields three columns
    /// per metric).
    pub aggs: Vec<AggFunc>,
    /// When true, each group is additionally split into a colocalizing and a
    /// non-colocalizing bucket (based on `RoiRow::coloc_json`), producing two
    /// rows per group instead of one.
    pub split_colocalized: bool,
    /// When true, each group is additionally split by class (e.g. "Image1 /
    /// Class1", "Image1 / Class2", …). A ROI carrying multiple classes
    /// contributes to each of its classes' buckets.
    pub group_by_class: bool,
}

impl Default for GroupConfig {
    fn default() -> Self {
        Self {
            group_by: GroupBy::None,
            regex: String::new(),
            aggs: vec![AggFunc::Avg],
            split_colocalized: false,
            group_by_class: false,
        }
    }
}

/// Returns the directory portion of a relative image path (`""` if none).
fn folder_of(rel_path: &str) -> String {
    std::path::Path::new(rel_path)
        .parent()
        .map(|p| p.to_string_lossy().to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "(root)".to_string())
}

/// Computes the grouping key for a single ROI, or `None` if the row should be
/// excluded from the grouped view (only happens for a non-matching regex).
fn group_key(roi: &RoiRow, group_by: GroupBy, regex: Option<&regex::Regex>) -> Option<String> {
    match group_by {
        GroupBy::None | GroupBy::Image => Some(roi.image_name.clone()),
        GroupBy::Folder => Some(folder_of(&roi.image_rel_path)),
        GroupBy::Regex => {
            let re = regex?;
            let caps = re.captures(&roi.image_name)?;
            // Prefer the first capture group; fall back to the whole match.
            let key = caps
                .get(1)
                .or_else(|| caps.get(0))
                .map(|m| m.as_str().to_string())?;
            Some(key)
        }
    }
}

/// Returns the classes a ROI belongs to, for fan-out grouping by class. Mirrors
/// [`compute_class`]'s fallback to `seg_class_name`, but as a list instead of a
/// joined string, so a multi-class ROI contributes to each of its classes'
/// group buckets separately.
fn roi_classes(roi: &RoiRow) -> Vec<String> {
    if roi.object_class_name.is_empty() {
        vec![roi.seg_class_name.clone().unwrap_or_default()]
    } else {
        roi.object_class_name.clone()
    }
}

/// True for per-ROI columns that carry a numeric value we can aggregate.
/// (Excludes `roi_id`, `image`, `class`.)
pub(crate) fn is_numeric_metric(id: &str) -> bool {
    matches!(id, "area_px" | "area_nm2" | "circularity")
        || id.starts_with("ch")
        || (id.starts_with("coloc_partner__") && id.ends_with("__count"))
}

/// Decimal places used when formatting a base metric's aggregate.
fn metric_precision(id: &str) -> usize {
    if id.starts_with("coloc_partner__") && id.ends_with("__count") {
        return 0; // whole number of colocalizing ROIs
    }
    match id {
        "area_nm2" => 2,
        "circularity" => 3,
        _ => 1, // area_px and channel bit values
    }
}

/// Pulls the numeric value of a base metric column (e.g. `area_px`,
/// `ch0_min_bit`, `coloc_partner__ClassA__count`) from a single ROI, or `None`
/// if absent.
pub(crate) fn metric_value(
    id: &str,
    roi: &RoiRow,
    intensities: &BTreeMap<i32, (f64, f64, f64, f64)>,
    coloc_partners: &BTreeMap<String, Vec<String>>,
) -> Option<f64> {
    match id {
        "area_px" => Some(roi.area_px as f64),
        "area_nm2" => Some(roi.area_nm2),
        "circularity" => Some(roi.circularity),
        id if id.starts_with("coloc_partner__") => {
            let (class, suffix) = split_coloc_partner_id(id)?;
            if suffix != "count" {
                return None;
            }
            Some(coloc_partners.get(class).map_or(0, Vec::len) as f64)
        }
        _ => {
            let rest = id.strip_prefix("ch")?;
            let under = rest.find('_')?;
            let ch: i32 = rest[..under].parse().ok()?;
            let &(min, max, avg, sum) = intensities.get(&ch)?;
            match &rest[under + 1..] {
                "min_bit" => Some(min),
                "max_bit" => Some(max),
                "avg_bit" => Some(avg),
                "sum_bit" => Some(sum),
                _ => None,
            }
        }
    }
}

/// Pulls the numeric value of `column_id` from a single ROI, parsing
/// `intensities_json`/`coloc_json` only if that column actually needs them
/// (same guarded pattern as [`to_display_row`]). Used by the chart module
/// (`results_chart.rs`), which — unlike [`aggregate_rows`] — looks at one
/// column at a time rather than every visible metric at once.
pub(crate) fn numeric_value(roi: &RoiRow, column_id: &str) -> Option<f64> {
    let intensities = if column_id.starts_with("ch") {
        parse_intensities(&roi.intensities_json)
    } else {
        BTreeMap::new()
    };
    let coloc_partners = if column_id.starts_with("coloc_partner__") {
        parse_coloc_partners(&roi.coloc_json)
    } else {
        BTreeMap::new()
    };
    metric_value(column_id, roi, &intensities, &coloc_partners)
}

/// Applies an aggregate to a slice of values. Returns `0.0` for an empty slice.
fn apply_agg(values: &[f64], agg: AggFunc) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    match agg {
        AggFunc::Min => values.iter().copied().fold(f64::INFINITY, f64::min),
        AggFunc::Max => values.iter().copied().fold(f64::NEG_INFINITY, f64::max),
        AggFunc::Sum => values.iter().sum(),
        AggFunc::Avg => values.iter().sum::<f64>() / values.len() as f64,
        AggFunc::Median => {
            let mut sorted = values.to_vec();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let mid = sorted.len() / 2;
            if sorted.len() % 2 == 0 {
                (sorted[mid - 1] + sorted[mid]) / 2.0
            } else {
                sorted[mid]
            }
        }
        AggFunc::Stdev => {
            // Sample standard deviation (n-1), matching spreadsheet STDEV.
            if values.len() < 2 {
                return 0.0;
            }
            let mean = values.iter().sum::<f64>() / values.len() as f64;
            let var = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>()
                / (values.len() - 1) as f64;
            var.sqrt()
        }
    }
}

/// Builds the grouped column specs for `config` over `base_specs`, alongside
/// the visible numeric metrics (in their original column order) that
/// contributed to them. Shared by [`aggregate_rows`] (the in-memory Rust
/// implementation) and `aggregate_rois_sql` (the SQL-pushdown implementation
/// in `evanalyzer_core`'s `DuckDbReader::aggregate_rois`) so the two can never
/// drift on column ids/labels/order — whichever computes the actual values,
/// both describe the same columns.
///
/// Columns: key, (optionally) class, (optionally) colocalized split, count,
/// then each visible numeric metric × each entry in `config.aggs` (id/label
/// carry the aggregate, e.g. `area_px__median` / `Area (px²) [median]`).
pub(crate) fn build_group_specs<'a>(
    config: &GroupConfig,
    base_specs: &'a [ColumnSpec],
) -> (Vec<ColumnSpec>, Vec<&'a ColumnSpec>) {
    let group_label = match config.group_by {
        GroupBy::Folder => "Folder",
        GroupBy::Regex => "Group",
        _ => "Image",
    };

    // Visible numeric metrics, in their original column order.
    let metrics: Vec<&ColumnSpec> = base_specs
        .iter()
        .filter(|c| c.visible && is_numeric_metric(&c.id))
        .collect();

    // Grouped columns: key, (optionally) class, (optionally) colocalized
    // split, count, then each metric × each selected aggregate.
    //
    // `group`/`group_class`/`colocalized` are marked filterable here — even
    // though a grouped row has no single image/class/coloc status of its own
    // — because the filter restricts which *source* ROIs are included before
    // aggregation, not any single grouped row's own data. Grouping replaces
    // the per-ROI table's "Image"/"Class" columns with these, so without this
    // the image/class/coloc filter's popup (only ever shown on a filterable
    // column's header) would have no column left to hang off of, and the
    // filter would become inaccessible the moment grouping is turned on (see
    // `results_table.slint`'s header, which routes `group`/`group_class` to
    // the same image/class filter popups as `image`/`class`).
    let mut specs = vec![ColumnSpec {
        id: "group".into(),
        label: group_label.into(),
        filterable: true,
        visible: true,
    }];
    if config.group_by_class {
        specs.push(ColumnSpec {
            id: "group_class".into(),
            label: "Class".into(),
            filterable: true,
            visible: true,
        });
    }
    if config.split_colocalized {
        specs.push(ColumnSpec {
            id: "colocalized".into(),
            label: "Colocalized".into(),
            filterable: true,
            visible: true,
        });
    }
    specs.push(ColumnSpec {
        id: "count".into(),
        label: "ROIs".into(),
        filterable: false,
        visible: true,
    });
    for m in &metrics {
        for agg in &config.aggs {
            specs.push(ColumnSpec {
                id: format!("{}__{}", m.id, agg.label()),
                label: format!("{} [{}]", m.label, agg.label()),
                filterable: false,
                visible: true,
            });
        }
    }

    (specs, metrics)
}

/// Groups `rois` according to `config` and produces one aggregated
/// [`DisplayRow`] per group, alongside the matching column specs.
///
/// `base_specs` are the per-ROI column specs (as used by [`to_display_row`]);
/// only the **visible** numeric metrics among them become grouped columns, so
/// the column-visibility filter applies in grouped mode too. Each such metric
/// gets one column per entry in `config.aggs` (label/id carry the aggregate,
/// e.g. `area_px__median` / `Area (px²) [median]`). Groups are returned in key
/// order, always preceded by the group-key and ROI-count columns.
pub fn aggregate_rows(
    rois: &[RoiRow],
    config: &GroupConfig,
    base_specs: &[ColumnSpec],
) -> (Vec<ColumnSpec>, Vec<DisplayRow>) {
    let (specs, metrics) = build_group_specs(config, base_specs);

    let regex = if config.group_by == GroupBy::Regex {
        regex::Regex::new(&config.regex).ok()
    } else {
        None
    };
    let needs_intensities = metrics.iter().any(|m| m.id.starts_with("ch"));
    let needs_coloc_partner = metrics.iter().any(|m| m.id.starts_with("coloc_partner__"));

    // (group key, class, colocalized flag) -> (ROI count, base metric id -> collected values).
    // `class` is `None` unless `group_by_class` is set, in which case a ROI fans out into
    // one bucket per class it carries. `coloc flag` is `None` unless `split_colocalized` is
    // set, in which case it is `Some(is_colocalized)`, splitting each group in two.
    let mut groups: BTreeMap<(String, Option<String>, Option<bool>), (usize, BTreeMap<String, Vec<f64>>)> =
        BTreeMap::new();
    for roi in rois {
        let Some(key) = group_key(roi, config.group_by, regex.as_ref()) else {
            continue;
        };
        let classes: Vec<Option<String>> = if config.group_by_class {
            roi_classes(roi).into_iter().map(Some).collect()
        } else {
            vec![None]
        };
        let coloc_flag = config.split_colocalized.then(|| is_colocalized(&roi.coloc_json));

        let intensities = if needs_intensities {
            parse_intensities(&roi.intensities_json)
        } else {
            BTreeMap::new()
        };
        let coloc_partners = if needs_coloc_partner {
            parse_coloc_partners(&roi.coloc_json)
        } else {
            BTreeMap::new()
        };

        for class in classes {
            let entry = groups.entry((key.clone(), class, coloc_flag)).or_default();
            entry.0 += 1;
            for m in &metrics {
                if let Some(v) = metric_value(&m.id, roi, &intensities, &coloc_partners) {
                    entry.1.entry(m.id.clone()).or_default().push(v);
                }
            }
        }
    }

    let rows = groups
        .into_iter()
        .enumerate()
        .map(|(idx, ((key, class, coloc_flag), (count, metric_vals)))| {
            let values = specs
                .iter()
                .map(|col| match col.id.as_str() {
                    "group" => key.clone(),
                    "group_class" => class.clone().unwrap_or_default(),
                    "colocalized" => coloc_label(coloc_flag.unwrap_or(false)),
                    "count" => count.to_string(),
                    id => {
                        // id is "<base_id>__<agg>"
                        let Some((base_id, agg_label)) = id.rsplit_once("__") else {
                            return String::new();
                        };
                        let Some(agg) = AggFunc::from_label(agg_label) else {
                            return String::new();
                        };
                        match metric_vals.get(base_id) {
                            Some(v) if !v.is_empty() => {
                                format!("{:.*}", metric_precision(base_id), apply_agg(v, agg))
                            }
                            _ => String::new(),
                        }
                    }
                })
                .collect();
            DisplayRow {
                roi_id: (idx + 1) as i32,
                object_id: String::new(), // aggregated row has no single source ROI
                values,
                stripe: idx % 2 == 0,
            }
        })
        .collect();

    (specs, rows)
}

/// Computes grouped/aggregated results directly in DuckDB (via
/// `evanalyzer_core::DuckDbReader::aggregate_rois`), instead of pulling every
/// matching row into Rust first — the SQL-pushdown counterpart to
/// [`aggregate_rows`], which stays as the parity ground-truth these two are
/// tested against each other for. Produces the exact same columns
/// ([`build_group_specs`] is the single source of truth both use) and the
/// same row order (group key, then class, then colocalized flag, all
/// ascending — matching `SELECT ... ORDER BY` in `aggregate_rois`, which
/// mirrors the `BTreeMap` iteration order `aggregate_rows` relies on).
///
/// `filter`'s `page_size`/`page`/`sort_column` are ignored — the aggregated
/// result set is always small and unpaginated, same as `aggregate_rows`.
pub fn aggregate_rois_sql(
    loader: &ResultsLoader,
    filter: DatabaseFilter,
    config: &GroupConfig,
    base_specs: &[ColumnSpec],
) -> Result<(Vec<ColumnSpec>, Vec<DisplayRow>), InternalErrors> {
    let (specs, metrics) = build_group_specs(config, base_specs);

    let roi_filter = RoiFilter {
        image_filter: filter.image_filter,
        class_filter: filter.class_filter,
        coloc_filter: filter.coloc_filter,
        object_id_filter: filter.object_id_filter,
        t_stack_filter: filter.t_stack_filter,
        z_stack_filter: filter.z_stack_filter,
        page_size: 0,
        page: 0,
        fetch_intensities: true,
        sort_column: None,
        sort_ascending: true,
    };

    let reader = DuckDbReader::open(&loader.path)?;

    let key_mode = match config.group_by {
        GroupBy::None | GroupBy::Image => GroupKeyMode::Image,
        GroupBy::Folder => {
            // Directory-splitting stays in Rust (`folder_of`, matching
            // `Path::parent()`'s exact semantics incl. the "(root)" fallback);
            // only the resulting small `rel_path -> key` lookup is pushed into
            // the query, via `get_distinct_images` (cheap: proportional to
            // distinct image count, not ROI count).
            let images = reader.get_distinct_images(&roi_filter)?;
            let map: std::collections::HashMap<String, String> = images
                .into_iter()
                .map(|(rel_path, _name)| {
                    let key = folder_of(&rel_path);
                    (rel_path, key)
                })
                .collect();
            GroupKeyMode::ImageRelPathMap(map)
        }
        GroupBy::Regex => {
            let Ok(re) = regex::Regex::new(&config.regex) else {
                // Invalid pattern -> every row's group_key is `None` in the
                // Rust path (`group_key`'s `let re = regex?;`), i.e. zero
                // groups. Match that without touching the DB at all.
                return Ok((specs, vec![]));
            };
            GroupKeyMode::Regex {
                pattern: config.regex.clone(),
                has_capture_group: re.captures_len() > 1,
            }
        }
    };

    let agg_spec = AggregateSpec {
        key_mode,
        group_by_class: config.group_by_class,
        split_colocalized: config.split_colocalized,
        metric_ids: metrics.iter().map(|m| m.id.clone()).collect(),
        agg_fns: config.aggs.iter().map(|a| a.sql_fn()).collect(),
    };

    let agg_rows = reader.aggregate_rois(&roi_filter, &agg_spec)?;

    let rows = agg_rows
        .into_iter()
        .enumerate()
        .map(|(idx, agg_row)| {
            let mut metric_idx = 0usize;
            let values = specs
                .iter()
                .map(|col| match col.id.as_str() {
                    "group" => agg_row.group_key.clone(),
                    "group_class" => agg_row.group_class.clone().unwrap_or_default(),
                    "colocalized" => coloc_label(agg_row.colocalized.unwrap_or(false)),
                    "count" => agg_row.count.to_string(),
                    id => {
                        // id is "<base_id>__<agg>" (base_id itself may contain
                        // "__", see `metric_precision`'s callers) — only the
                        // base id is needed here, `agg_row.metric_values` is
                        // already positional.
                        let Some((base_id, _agg_label)) = id.rsplit_once("__") else {
                            return String::new();
                        };
                        let value = agg_row.metric_values.get(metric_idx).copied().flatten();
                        metric_idx += 1;
                        match value {
                            Some(v) => format!("{:.*}", metric_precision(base_id), v),
                            None => String::new(),
                        }
                    }
                })
                .collect();
            DisplayRow { roi_id: (idx + 1) as i32, object_id: String::new(), values, stripe: idx % 2 == 0 }
        })
        .collect();

    Ok((specs, rows))
}

/// Sorts `rows` in place by the current value of `column_id` (looked up via
/// `specs`), ascending or descending. Values that parse as numbers sort
/// numerically; everything else sorts as text. A `column_id` absent from
/// `specs` leaves `rows` untouched.
///
/// The grouped view is always fully materialized in memory (see
/// [`aggregate_rows`]), so sorting it is a plain in-memory sort; the
/// paginated per-ROI view sorts via SQL `ORDER BY` instead (see
/// `DatabaseFilter::sort_column`), since only a fraction of its rows are
/// ever loaded at once.
pub fn sort_display_rows(rows: &mut [DisplayRow], specs: &[ColumnSpec], column_id: &str, ascending: bool) {
    let Some(idx) = specs.iter().position(|c| c.id == column_id) else {
        return;
    };
    rows.sort_by(|a, b| {
        let av = a.values.get(idx).map(String::as_str).unwrap_or("");
        let bv = b.values.get(idx).map(String::as_str).unwrap_or("");
        let ord = match (av.parse::<f64>(), bv.parse::<f64>()) {
            (Ok(x), Ok(y)) => x.partial_cmp(&y).unwrap_or(std::cmp::Ordering::Equal),
            _ => av.cmp(bv),
        };
        if ascending { ord } else { ord.reverse() }
    });
}

pub struct ResultsLoader {
    path: PathBuf,
}

impl ResultsLoader {
    pub fn new(path: impl Into<PathBuf>) -> Self {
        Self { path: path.into() }
    }

    pub fn get_rois(&self, filter: DatabaseFilter) -> Result<Vec<RoiRow>, InternalErrors> {
        DuckDbReader::open(&self.path)?.get_rois(&to_roi_filter(filter))
    }

    /// Opens a `DuckDbReader` the caller keeps alive across several calls -
    /// e.g. one [`DuckDbReader::stream_rois`] pass interleaved with
    /// per-batch [`DuckDbReader::get_rois`] lookups on the same connection,
    /// as `ResultsExporter`'s colocalization-detail export does - instead of
    /// paying a fresh connection-open on every call the way `get_rois`/
    /// `stream_rois` on `ResultsLoader` itself do.
    pub(crate) fn open_reader(&self) -> Result<DuckDbReader, InternalErrors> {
        DuckDbReader::open(&self.path)
    }

    pub fn get_image_names(&self) -> Result<Vec<String>, InternalErrors> {
        DuckDbReader::open(&self.path)?.get_image_names()
    }

    pub fn get_class_names(&self) -> Result<Vec<String>, InternalErrors> {
        DuckDbReader::open(&self.path)?.get_class_names()
    }

    /// `(min, max)` of `t_stack` present in the file, or `None` when there's
    /// nothing to step through (no time axis, or a single shared value).
    pub fn get_t_stack_range(&self) -> Result<Option<(i32, i32)>, InternalErrors> {
        DuckDbReader::open(&self.path)?.get_t_stack_range()
    }

    /// `(min, max)` of `z_stack` present in the file, or `None` when there's
    /// nothing to step through (no depth axis, or a single shared value).
    pub fn get_z_stack_range(&self) -> Result<Option<(i32, i32)>, InternalErrors> {
        DuckDbReader::open(&self.path)?.get_z_stack_range()
    }

    pub fn get_coloc_partner_class_names(&self) -> Result<Vec<String>, InternalErrors> {
        DuckDbReader::open(&self.path)?.get_coloc_partner_class_names()
    }
}

/// Bounded-sample column discovery for the colocalization detail flat table:
/// the partner classes (a cheap `DISTINCT`-over-JSON-keys query) plus channels
/// discovered from a capped sample of ROIs in scope, instead of scanning every
/// ROI in the image. Called once when coloc-detail mode is entered (or on a
/// filter/sort change that resets to page 0), not per page, so the column set
/// stays fixed while the user scrolls/loads more pages.
const COLOC_DETAIL_COLUMN_DISCOVERY_SAMPLE: usize = 2000;

pub fn discover_coloc_detail_columns(
    loader: &ResultsLoader,
    filter: &DatabaseFilter,
) -> Result<(Vec<i32>, Vec<String>), InternalErrors> {
    let coloc_partner_classes = loader.get_coloc_partner_class_names()?;
    let sample = loader.get_rois(DatabaseFilter {
        image_filter: filter.image_filter.clone(),
        class_filter: None,
        coloc_filter: None,
        object_id_filter: None,
        t_stack_filter: filter.t_stack_filter,
        z_stack_filter: filter.z_stack_filter,
        page_size: COLOC_DETAIL_COLUMN_DISCOVERY_SAMPLE,
        page: 0,
        needs_intensities: true,
        sort_column: None,
        sort_ascending: true,
    })?;
    Ok((discover_channels(&sample), coloc_partner_classes))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::results::test_support::seed_results_db;

    // ---- ResultsLoader against a real DuckDB file (backs the results table) ----

    #[test]
    fn results_loader_get_rois_fetches_real_rows_from_the_database() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("results.duckdb");
        seed_results_db(&db_path);

        let loader = ResultsLoader::new(db_path);
        let mut rois = loader.get_rois(DatabaseFilter::default()).expect("query should succeed");
        rois.sort_by_key(|r| r.image_name.clone());

        assert_eq!(rois.len(), 2);
        assert_eq!(rois[0].image_name, "img1.tif");
        assert_eq!(rois[0].object_class_name, vec!["ClassA".to_string()]);
        assert_eq!(rois[0].area_px, 100);
        assert_eq!(rois[1].image_name, "img2.tif");
        assert_eq!(rois[1].object_class_name, vec!["ClassB".to_string()]);
        assert_eq!(rois[1].area_px, 200);
    }

    #[test]
    fn results_loader_get_rois_respects_the_image_filter() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("results.duckdb");
        seed_results_db(&db_path);

        let loader = ResultsLoader::new(db_path);
        let filter =
            DatabaseFilter { image_filter: Some(vec!["img2.tif".to_string()]), ..Default::default() };
        let rois = loader.get_rois(filter).unwrap();

        assert_eq!(rois.len(), 1);
        assert_eq!(rois[0].image_name, "img2.tif");
    }

    #[test]
    fn results_loader_get_rois_respects_the_class_filter() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("results.duckdb");
        seed_results_db(&db_path);

        let loader = ResultsLoader::new(db_path);
        let filter =
            DatabaseFilter { class_filter: Some(vec!["ClassA".to_string()]), ..Default::default() };
        let rois = loader.get_rois(filter).unwrap();

        assert_eq!(rois.len(), 1);
        assert_eq!(rois[0].object_class_name, vec!["ClassA".to_string()]);
    }

    #[test]
    fn results_loader_get_rois_sorts_by_the_requested_column() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("results.duckdb");
        seed_results_db(&db_path);

        let loader = ResultsLoader::new(db_path);

        let ascending = loader
            .get_rois(DatabaseFilter {
                sort_column: Some("area_px".to_string()),
                sort_ascending: true,
                ..Default::default()
            })
            .unwrap();
        assert_eq!(ascending.iter().map(|r| r.area_px).collect::<Vec<_>>(), vec![100, 200]);

        let descending = loader
            .get_rois(DatabaseFilter {
                sort_column: Some("area_px".to_string()),
                sort_ascending: false,
                ..Default::default()
            })
            .unwrap();
        assert_eq!(descending.iter().map(|r| r.area_px).collect::<Vec<_>>(), vec![200, 100]);
    }

    #[test]
    fn results_loader_get_rois_returns_empty_for_a_filter_matching_nothing() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("results.duckdb");
        seed_results_db(&db_path);

        let loader = ResultsLoader::new(db_path);
        let filter =
            DatabaseFilter { image_filter: Some(vec!["no-such-image.tif".to_string()]), ..Default::default() };
        assert!(loader.get_rois(filter).unwrap().is_empty());
    }

    #[test]
    fn results_loader_get_image_names_and_class_names_reflect_the_seeded_data() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("results.duckdb");
        seed_results_db(&db_path);

        let loader = ResultsLoader::new(db_path);
        let mut images = loader.get_image_names().unwrap();
        images.sort();
        assert_eq!(images, vec!["img1.tif".to_string(), "img2.tif".to_string()]);

        let mut classes = loader.get_class_names().unwrap();
        classes.sort();
        assert_eq!(classes, vec!["ClassA".to_string(), "ClassB".to_string()]);
    }

    // ---- helpers ----

    fn make_roi(
        image_name: &str,
        object_class_name: Vec<String>,
        seg_class_name: Option<&str>,
        area_px: u64,
        area_nm2: f64,
        circularity: f64,
        intensities_json: &str,
    ) -> RoiRow {
        RoiRow {
            image_name: image_name.into(),
            image_rel_path: String::new(),
            c_stack: None,
            z_stack: None,
            t_stack: None,
            object_id: "00000000-0000-0000-0000-000000000001".into(),
            seg_class_name: seg_class_name.map(str::to_owned),
            seg_class_id: None,
            object_class_name,
            object_class_id: vec![],
            parent_id: None,
            children: vec![],
            track_id: 0,
            centroid_x_px: 0.0,
            centroid_y_px: 0.0,
            centroid_x_nm: 0.0,
            centroid_y_nm: 0.0,
            area_px,
            area_nm2,
            perimeter_px: 0.0,
            perimeter_nm: 0.0,
            circularity,
            solidity: 0.0,
            aspect_ratio: 0.0,
            roundness: 0.0,
            compactness: 0.0,
            major_axis_px: 0.0,
            minor_axis_px: 0.0,
            touches_edge: false,
            intensities_json: intensities_json.into(),
            coloc_json: "{}".into(),
            bbox_px: [0, 0, 0, 0],
        }
    }

    // ---- discover_channels ----

    #[test]
    fn discover_channels_empty_input() {
        assert_eq!(discover_channels(&[]), Vec::<i32>::new());
    }

    #[test]
    fn discover_channels_no_intensities() {
        let roi = make_roi("img.tif", vec![], None, 100, 100.0, 0.9, "{}");
        assert_eq!(discover_channels(&[roi]), Vec::<i32>::new());
    }

    #[test]
    fn discover_channels_single_channel() {
        let roi = make_roi(
            "img.tif",
            vec![],
            None,
            100,
            100.0,
            0.9,
            r#"{"0":{"sum_raw":1.0,"sum_scaled":255.0,"mean_raw":0.5,"mean_scaled":127.0,"median_raw":0.5,"median_scaled":127.0,"std_raw":0.1,"std_scaled":25.5,"min_raw":0.0,"min_scaled":0.0,"max_raw":1.0,"max_scaled":255.0}}"#,
        );
        assert_eq!(discover_channels(&[roi]), vec![0]);
    }

    #[test]
    fn discover_channels_multiple_images_merged_and_sorted() {
        let roi1 = make_roi(
            "a.tif",
            vec![],
            None,
            100,
            100.0,
            0.9,
            r#"{"2":{"sum_raw":1.0,"sum_scaled":1.0,"mean_raw":1.0,"mean_scaled":1.0,"median_raw":1.0,"median_scaled":1.0,"std_raw":0.0,"std_scaled":0.0,"min_raw":0.0,"min_scaled":0.0,"max_raw":1.0,"max_scaled":1.0}}"#,
        );
        let roi2 = make_roi(
            "b.tif",
            vec![],
            None,
            100,
            100.0,
            0.9,
            r#"{"0":{"sum_raw":1.0,"sum_scaled":1.0,"mean_raw":1.0,"mean_scaled":1.0,"median_raw":1.0,"median_scaled":1.0,"std_raw":0.0,"std_scaled":0.0,"min_raw":0.0,"min_scaled":0.0,"max_raw":1.0,"max_scaled":1.0},"1":{"sum_raw":1.0,"sum_scaled":1.0,"mean_raw":1.0,"mean_scaled":1.0,"median_raw":1.0,"median_scaled":1.0,"std_raw":0.0,"std_scaled":0.0,"min_raw":0.0,"min_scaled":0.0,"max_raw":1.0,"max_scaled":1.0}}"#,
        );
        assert_eq!(discover_channels(&[roi1, roi2]), vec![0, 1, 2]);
    }

    // ---- build_column_specs ----

    #[test]
    fn build_column_specs_no_channels_has_seven_fixed_cols() {
        let specs = build_column_specs(&[], &[]);
        assert_eq!(specs.len(), 7);
        assert_eq!(specs[0].id, "roi_id");
        assert_eq!(specs[5].id, "circularity");
        assert_eq!(specs[6].id, "colocalized");
        assert!(specs.iter().all(|c| c.visible));
    }

    #[test]
    fn build_coloc_detail_column_specs_image_and_class_are_filterable() {
        // Without this, the Coloc Details view's "Image"/"Class" column
        // headers never show a filter icon at all, since the GUI only draws
        // one when the column's own spec says it's filterable — matching
        // `build_column_specs`, which marks these two filterable too.
        let specs = build_coloc_detail_column_specs(&[], &[]);
        let image = specs.iter().find(|c| c.id == "image").unwrap();
        let class = specs.iter().find(|c| c.id == "class").unwrap();
        assert!(image.filterable, "Image column must be filterable in Coloc Details");
        assert!(class.filterable, "Class column must be filterable in Coloc Details");
    }

    #[test]
    fn build_column_specs_with_channels_adds_four_cols_per_channel() {
        let specs = build_column_specs(&[0, 1], &[]);
        assert_eq!(specs.len(), 7 + 4 * 2);
        let ch_ids: Vec<&str> = specs[7..].iter().map(|c| c.id.as_str()).collect();
        assert_eq!(
            ch_ids,
            [
                "ch0_min_bit",
                "ch0_max_bit",
                "ch0_avg_bit",
                "ch0_sum_bit",
                "ch1_min_bit",
                "ch1_max_bit",
                "ch1_avg_bit",
                "ch1_sum_bit",
            ]
        );
    }

    // ---- to_display_row ----

    #[test]
    fn to_display_row_basic_fixed_columns() {
        let roi = make_roi(
            "sample.tif",
            vec!["Nucleus".into()],
            None,
            1234,
            567.89,
            0.923,
            "{}",
        );
        let specs = build_column_specs(&[], &[]);
        let row = to_display_row(0, &roi, &specs);

        assert_eq!(row.roi_id, 1); // row_idx=0 → roi_id=1
        assert_eq!(row.values[0], "00000000-0000-0000-0000-000000000001"); // roi_id col
        assert_eq!(row.values[1], "sample.tif"); // image
        assert_eq!(row.values[2], "Nucleus"); // class from object_class_name
        assert_eq!(row.values[3], "1234"); // area_px
        assert_eq!(row.values[4], "567.89"); // area_nm2
        assert_eq!(row.values[5], "0.923"); // circularity
        assert_eq!(row.values[6], "No"); // colocalized (coloc_json is "{}")
    }

    #[test]
    fn to_display_row_colocalized_column_reflects_coloc_json() {
        let roi = make_roi("img.tif", vec![], None, 0, 0.0, 0.0, "{}");
        let mut roi = roi;
        roi.coloc_json = r#"{"Target (1)":["00000000-0000-0000-0000-000000000002"]}"#.into();
        let specs = build_column_specs(&[], &[]);
        let row = to_display_row(0, &roi, &specs);
        assert_eq!(row.values[6], "Yes");

        let mut not_colocalized = roi.clone();
        not_colocalized.coloc_json = r#"{"Target (1)":[]}"#.into();
        let row = to_display_row(0, &not_colocalized, &specs);
        assert_eq!(row.values[6], "No");
    }

    #[test]
    fn to_display_row_class_falls_back_to_seg_class_when_object_class_empty() {
        let roi = make_roi("img.tif", vec![], Some("Background"), 0, 0.0, 0.0, "{}");
        let specs = build_column_specs(&[], &[]);
        let row = to_display_row(0, &roi, &specs);
        assert_eq!(row.values[2], "Background");
    }

    #[test]
    fn to_display_row_hidden_column_is_empty_string() {
        let roi = make_roi("img.tif", vec![], None, 100, 100.0, 0.5, "{}");
        let mut specs = build_column_specs(&[], &[]);
        specs[3].visible = false; // hide area_px
        let row = to_display_row(0, &roi, &specs);
        assert_eq!(row.values[3], "");
    }

    #[test]
    fn to_display_row_channel_values_parsed_correctly() {
        let json = r#"{"0":{"sum_raw":0.0,"sum_scaled":0.0,"mean_raw":0.0,"mean_scaled":0.0,"median_raw":0.0,"median_scaled":0.0,"std_raw":0.0,"std_scaled":0.0,"min_raw":0.0,"min_scaled":100.0,"max_raw":0.0,"max_scaled":200.0}}"#;
        let roi = make_roi("img.tif", vec![], None, 0, 0.0, 0.0, json);
        let specs = build_column_specs(&[0], &[]);
        let row = to_display_row(0, &roi, &specs);
        // index 7 = ch0_min_bit, 8 = ch0_max_bit, 9 = ch0_avg_bit
        assert_eq!(row.values[7], "100"); // min_scaled rounded
        assert_eq!(row.values[8], "200"); // max_scaled rounded
    }

    #[test]
    fn to_display_row_row_idx_increments_roi_id() {
        let roi = make_roi("img.tif", vec![], None, 0, 0.0, 0.0, "{}");
        let specs = build_column_specs(&[], &[]);
        assert_eq!(to_display_row(0, &roi, &specs).roi_id, 1);
        assert_eq!(to_display_row(4, &roi, &specs).roi_id, 5);
    }

    // ---- aggregation primitives ----

    #[test]
    fn apply_agg_basic_functions() {
        let v = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        assert_eq!(apply_agg(&v, AggFunc::Min), 2.0);
        assert_eq!(apply_agg(&v, AggFunc::Max), 9.0);
        assert_eq!(apply_agg(&v, AggFunc::Sum), 40.0);
        assert_eq!(apply_agg(&v, AggFunc::Avg), 5.0);
        assert_eq!(apply_agg(&v, AggFunc::Median), 4.5); // even count -> mean of middles
        // Sample stdev (n-1) of this set is sqrt(32/7) ≈ 2.1381
        assert!((apply_agg(&v, AggFunc::Stdev) - (32.0f64 / 7.0).sqrt()).abs() < 1e-9);
    }

    #[test]
    fn apply_agg_edge_cases() {
        assert_eq!(apply_agg(&[], AggFunc::Avg), 0.0);
        assert_eq!(apply_agg(&[42.0], AggFunc::Median), 42.0);
        assert_eq!(apply_agg(&[42.0], AggFunc::Stdev), 0.0); // n<2 -> 0
    }

    // ---- group_key ----

    #[test]
    fn group_key_image_and_folder() {
        let roi = RoiRow {
            image_rel_path: "plate1/wellA/A1_01.tif".into(),
            ..make_roi("A1_01.tif", vec![], None, 0, 0.0, 0.0, "{}")
        };
        assert_eq!(group_key(&roi, GroupBy::Image, None).as_deref(), Some("A1_01.tif"));
        assert_eq!(
            group_key(&roi, GroupBy::Folder, None).as_deref(),
            Some("plate1/wellA")
        );
    }

    #[test]
    fn group_key_regex_capture_group_extracts_well() {
        let re = regex::Regex::new(r"^([A-Z]\d+)_").unwrap();
        let roi = make_roi("A1_02", vec![], None, 0, 0.0, 0.0, "{}");
        assert_eq!(
            group_key(&roi, GroupBy::Regex, Some(&re)).as_deref(),
            Some("A1")
        );
        // Non-matching name is dropped from the grouped view.
        let roi2 = make_roi("control", vec![], None, 0, 0.0, 0.0, "{}");
        assert_eq!(group_key(&roi2, GroupBy::Regex, Some(&re)), None);
    }

    // ---- aggregate_rows ----

    #[test]
    fn aggregate_rows_groups_by_regex_well_and_counts() {
        let rois = vec![
            make_roi("A1_01", vec![], None, 100, 0.0, 0.8, "{}"),
            make_roi("A1_02", vec![], None, 200, 0.0, 0.9, "{}"),
            make_roi("B1_01", vec![], None, 300, 0.0, 0.7, "{}"),
        ];
        let config = GroupConfig {
            group_by: GroupBy::Regex,
            regex: r"^([A-Z]\d+)_".into(),
            aggs: vec![AggFunc::Avg],
            split_colocalized: false,
            group_by_class: false,
        };
        let (specs, rows) = aggregate_rows(&rois, &config, &build_column_specs(&[], &[]));

        // group, count, area_px, area_nm2, circularity
        assert_eq!(specs.len(), 5);
        assert_eq!(specs[0].id, "group");
        assert_eq!(specs[1].id, "count");
        assert_eq!(specs[2].id, "area_px__avg");

        assert_eq!(rows.len(), 2); // A1, B1 (sorted)
        assert_eq!(rows[0].values[0], "A1");
        assert_eq!(rows[0].values[1], "2"); // count
        assert_eq!(rows[0].values[2], "150.0"); // avg area_px (100,200)
        assert_eq!(rows[1].values[0], "B1");
        assert_eq!(rows[1].values[1], "1");
    }

    #[test]
    fn aggregate_rows_channel_columns_use_selected_func() {
        let json = |min: f64, max: f64| {
            format!(
                r#"{{"0":{{"min_scaled":{min},"max_scaled":{max},"mean_scaled":0.0}}}}"#
            )
        };
        let rois = vec![
            make_roi("img", vec![], None, 0, 0.0, 0.0, &json(10.0, 100.0)),
            make_roi("img", vec![], None, 0, 0.0, 0.0, &json(30.0, 200.0)),
        ];
        let config = GroupConfig {
            group_by: GroupBy::Image,
            regex: String::new(),
            aggs: vec![AggFunc::Min],
            split_colocalized: false,
            group_by_class: false,
        };
        let (specs, rows) = aggregate_rows(&rois, &config, &build_column_specs(&[0], &[]));
        // group, count, area_px, area_nm2, circularity, ch0_min, ch0_max, ch0_avg, ch0_sum
        assert_eq!(specs.len(), 9);
        assert_eq!(rows.len(), 1);
        // Min of the per-ROI min_scaled values (10, 30) -> 10
        assert_eq!(rows[0].values[5], "10.0");
        // Min of the per-ROI max_scaled values (100, 200) -> 100
        assert_eq!(rows[0].values[6], "100.0");
    }

    #[test]
    fn aggregate_rows_multiple_aggregates_yield_one_column_each() {
        let rois = vec![
            make_roi("img", vec![], None, 100, 0.0, 0.0, "{}"),
            make_roi("img", vec![], None, 300, 0.0, 0.0, "{}"),
        ];
        let config = GroupConfig {
            group_by: GroupBy::Image,
            regex: String::new(),
            aggs: vec![AggFunc::Min, AggFunc::Max, AggFunc::Avg],
            split_colocalized: false,
            group_by_class: false,
        };
        let (specs, rows) = aggregate_rows(&rois, &config, &build_column_specs(&[], &[]));
        // group, count, then 3 metrics (area_px, area_nm2, circularity) × 3 aggs
        assert_eq!(specs.len(), 2 + 3 * 3);
        assert_eq!(specs[2].id, "area_px__min");
        assert_eq!(specs[2].label, "Area (px²) [min]");
        assert_eq!(specs[3].id, "area_px__max");
        assert_eq!(specs[4].id, "area_px__avg");
        assert_eq!(rows[0].values[2], "100.0"); // min
        assert_eq!(rows[0].values[3], "300.0"); // max
        assert_eq!(rows[0].values[4], "200.0"); // avg
    }

    #[test]
    fn aggregate_rows_respects_column_visibility() {
        let rois = vec![make_roi("img", vec![], None, 100, 50.0, 0.9, "{}")];
        // Only the area_px metric is visible.
        let mut base_specs = build_column_specs(&[], &[]);
        for spec in base_specs.iter_mut() {
            spec.visible = spec.id == "area_px";
        }
        let config = GroupConfig {
            group_by: GroupBy::Image,
            regex: String::new(),
            aggs: vec![AggFunc::Avg],
            split_colocalized: false,
            group_by_class: false,
        };
        let (specs, _rows) = aggregate_rows(&rois, &config, &base_specs);
        // group, count, area_px__avg — area_nm2/circularity hidden.
        assert_eq!(specs.len(), 3);
        assert_eq!(specs[2].id, "area_px__avg");
    }

    #[test]
    fn aggregate_rows_split_colocalized_produces_two_rows_per_group() {
        let mut coloc_roi_1 = make_roi("img", vec![], None, 100, 0.0, 0.0, "{}");
        coloc_roi_1.coloc_json = r#"{"Target (1)":["00000000-0000-0000-0000-000000000002"]}"#.into();
        let mut coloc_roi_2 = make_roi("img", vec![], None, 200, 0.0, 0.0, "{}");
        coloc_roi_2.coloc_json = r#"{"Target (1)":["00000000-0000-0000-0000-000000000003"]}"#.into();
        let not_coloc_roi = make_roi("img", vec![], None, 300, 0.0, 0.0, "{}"); // coloc_json "{}"

        let rois = vec![coloc_roi_1, coloc_roi_2, not_coloc_roi];
        let config = GroupConfig {
            group_by: GroupBy::Image,
            regex: String::new(),
            aggs: vec![AggFunc::Avg],
            split_colocalized: true,
            group_by_class: false,
        };
        let (specs, rows) = aggregate_rows(&rois, &config, &build_column_specs(&[], &[]));

        // group, colocalized, count, area_px__avg, area_nm2__avg, circularity__avg
        assert_eq!(specs[0].id, "group");
        assert_eq!(specs[1].id, "colocalized");
        assert_eq!(specs[2].id, "count");

        assert_eq!(rows.len(), 2, "one colocalizing + one non-colocalizing bucket");
        // BTreeMap orders Option<bool> as None < Some(false) < Some(true), so the
        // non-colocalizing bucket sorts first.
        assert_eq!(rows[0].values[1], "No");
        assert_eq!(rows[0].values[2], "1"); // count
        assert_eq!(rows[0].values[3], "300.0"); // area_px avg
        assert_eq!(rows[1].values[1], "Yes");
        assert_eq!(rows[1].values[2], "2");
        assert_eq!(rows[1].values[3], "150.0"); // avg(100,200)
    }

    // ---- coloc-partner columns ----

    #[test]
    fn build_column_specs_adds_two_cols_per_partner_class() {
        let specs = build_column_specs(&[], &["ClassA".to_string(), "ClassB".to_string()]);
        let ids: Vec<&str> = specs[7..].iter().map(|c| c.id.as_str()).collect();
        assert_eq!(
            ids,
            [
                "coloc_partner__ClassA__count",
                "coloc_partner__ClassA__ids",
                "coloc_partner__ClassB__count",
                "coloc_partner__ClassB__ids",
            ]
        );
    }

    #[test]
    fn to_display_row_coloc_partner_count_and_ids() {
        let mut roi = make_roi("img.tif", vec![], None, 0, 0.0, 0.0, "{}");
        roi.coloc_json = r#"{"ClassA":["00000000-0000-0000-0000-000000000002","00000000-0000-0000-0000-000000000003"]}"#.into();
        let specs = build_column_specs(&[], &["ClassA".to_string(), "ClassB".to_string()]);
        let row = to_display_row(0, &roi, &specs);

        // index 7/8 = ClassA count/ids, 9/10 = ClassB count/ids
        assert_eq!(row.values[7], "2");
        assert_eq!(
            row.values[8],
            "00000000-0000-0000-0000-000000000002, 00000000-0000-0000-0000-000000000003"
        );
        assert_eq!(row.values[9], "0", "no ClassB partners");
        assert_eq!(row.values[10], "");
    }

    #[test]
    fn parse_coloc_partners_dedups_repeated_id_in_same_class() {
        // A multi-class source ROI can be re-anchored against the same partner
        // more than once (see coloc_rois.rs), leaving a repeated id in an
        // older result file's coloc_json. The loader should collapse it so a
        // single overlap is never counted/displayed as two.
        let json = r#"{"ClassA":["00000000-0000-0000-0000-000000000002","00000000-0000-0000-0000-000000000002"]}"#;
        let partners = parse_coloc_partners(json);
        assert_eq!(
            partners.get("ClassA").map(Vec::as_slice),
            Some(["00000000-0000-0000-0000-000000000002".to_string()].as_slice())
        );
    }

    #[test]
    fn coloc_partner_ids_dedups_across_rows_and_classes() {
        let mut a = make_roi("img", vec![], None, 0, 0.0, 0.0, "{}");
        a.coloc_json = r#"{"ClassA":["00000000-0000-0000-0000-000000000002","00000000-0000-0000-0000-000000000003"]}"#.into();
        let mut b = make_roi("img", vec![], None, 0, 0.0, 0.0, "{}");
        // Shares one id with `a` (under a different class) and contributes one new id.
        b.coloc_json = r#"{"ClassB":["00000000-0000-0000-0000-000000000002","00000000-0000-0000-0000-000000000004"]}"#.into();
        let no_partners = make_roi("img", vec![], None, 0, 0.0, 0.0, "{}");

        let ids = coloc_partner_ids(&[a, b, no_partners]);
        assert_eq!(
            ids,
            vec![
                "00000000-0000-0000-0000-000000000002".to_string(),
                "00000000-0000-0000-0000-000000000003".to_string(),
                "00000000-0000-0000-0000-000000000004".to_string(),
            ]
        );
    }

    #[test]
    fn coloc_partner_ids_empty_for_no_partners() {
        let roi = make_roi("img", vec![], None, 0, 0.0, 0.0, "{}");
        assert!(coloc_partner_ids(&[roi]).is_empty());
        assert!(coloc_partner_ids(&[]).is_empty());
    }

    #[test]
    fn flatten_coloc_rows_does_not_duplicate_row_for_repeated_partner_id() {
        let mut src = make_roi("img.tif", vec!["Source".into()], None, 0, 0.0, 0.0, "{}");
        src.object_id = "00000000-0000-0000-0000-000000000001".into();
        src.coloc_json =
            r#"{"ClassA":["00000000-0000-0000-0000-000000000002","00000000-0000-0000-0000-000000000002"]}"#
                .into();

        let mut partner = make_roi("img.tif", vec!["ClassA".into()], None, 0, 0.0, 0.0, "{}");
        partner.object_id = "00000000-0000-0000-0000-000000000002".into();

        let specs = build_coloc_detail_column_specs(&[], &["ClassA".to_string()]);
        let rows = flatten_coloc_rows(&[src], &[partner], &specs);

        assert_eq!(rows.len(), 1, "a repeated partner id must not produce an extra row");
    }

    #[test]
    fn flatten_coloc_rows_one_row_per_distinct_partner_even_if_it_matches_two_classes() {
        // The partner ROI itself carries two coloc-target classes, so the
        // Colocalization algorithm legitimately records it under two class
        // keys in the source's coloc_json. That's still a single physical
        // colocalization — the flat table must show exactly one row for it
        // (with both class column groups filled), not one row per class key.
        let mut src = make_roi("img.tif", vec!["Source".into()], None, 0, 0.0, 0.0, "{}");
        src.object_id = "00000000-0000-0000-0000-000000000001".into();
        src.coloc_json = r#"{"ClassA":["00000000-0000-0000-0000-000000000002"],"ClassB":["00000000-0000-0000-0000-000000000002"]}"#.into();

        let mut partner = make_roi(
            "img.tif",
            vec!["ClassA".into(), "ClassB".into()],
            None,
            0,
            0.0,
            0.0,
            "{}",
        );
        partner.object_id = "00000000-0000-0000-0000-000000000002".into();

        let specs = build_coloc_detail_column_specs(&[], &["ClassA".to_string(), "ClassB".to_string()]);
        let rows = flatten_coloc_rows(&[src], &[partner], &specs);

        assert_eq!(
            rows.len(),
            1,
            "one physical partner ROI must produce exactly one row, even under two class keys"
        );

        let col = |id: &str| specs.iter().position(|c| c.id == id).unwrap();
        assert_eq!(
            rows[0].values[col("coloc_detail__ClassA__roi_id")],
            "00000000-0000-0000-0000-000000000002"
        );
        assert_eq!(
            rows[0].values[col("coloc_detail__ClassB__roi_id")],
            "00000000-0000-0000-0000-000000000002",
            "both matching class groups should be filled in the single row"
        );
    }

    #[test]
    fn flatten_coloc_rows_stripe_groups_by_source_roi_not_flattened_row() {
        // src1 colocalizes with two partners -> two flattened rows, both from
        // the SAME source ROI, so both must share one stripe value. src2 has
        // no partners -> one row. The stripe must toggle between src1's group
        // and src2's row, not on every flattened row.
        let mut src1 = make_roi("img", vec![], None, 0, 0.0, 0.0, "{}");
        src1.object_id = "00000000-0000-0000-0000-000000000001".into();
        src1.coloc_json = r#"{"ClassA":["00000000-0000-0000-0000-000000000010","00000000-0000-0000-0000-000000000011"]}"#.into();
        let src2 = make_roi("img", vec![], None, 0, 0.0, 0.0, "{}");

        let p10 = make_roi("img", vec!["ClassA".into()], None, 0, 0.0, 0.0, "{}");
        let p11 = make_roi("img", vec!["ClassA".into()], None, 0, 0.0, 0.0, "{}");
        let specs = build_coloc_detail_column_specs(&[], &["ClassA".to_string()]);

        let rows = flatten_coloc_rows(&[src1, src2], &[p10, p11], &specs);

        assert_eq!(rows.len(), 3, "2 rows for src1's partners + 1 row for src2");
        assert_eq!(rows[0].stripe, rows[1].stripe, "both of src1's rows share one stripe");
        assert_ne!(rows[1].stripe, rows[2].stripe, "stripe toggles moving to src2");
    }

    #[test]
    fn flatten_coloc_rows_flags_a_partner_reference_that_does_not_resolve() {
        // src's coloc_json references three distinct partner ids, but only
        // two of them (C and D) actually exist in partner_lookup — the third
        // (a stale/orphaned reference) must still produce its own row rather
        // than being silently merged into the "no partner" case, and its ROI
        // ID column must be visibly flagged rather than blank.
        let mut src = make_roi("img", vec![], None, 0, 0.0, 0.0, "{}");
        src.object_id = "00000000-0000-0000-0000-000000000001".into();
        src.coloc_json = r#"{"ClassA":["00000000-0000-0000-0000-000000000002","00000000-0000-0000-0000-000000000003","00000000-0000-0000-0000-000000000099"]}"#.into();

        let mut c = make_roi("img", vec!["ClassA".into()], None, 0, 0.0, 0.0, "{}");
        c.object_id = "00000000-0000-0000-0000-000000000002".into();
        let mut d = make_roi("img", vec!["ClassA".into()], None, 0, 0.0, 0.0, "{}");
        d.object_id = "00000000-0000-0000-0000-000000000003".into();
        // Note: id "...099" is intentionally absent from partner_lookup.

        let specs = build_coloc_detail_column_specs(&[], &["ClassA".to_string()]);
        let rows = flatten_coloc_rows(&[src], &[c, d], &specs);

        assert_eq!(rows.len(), 3, "one row per referenced partner id, including the unresolved one");

        let col = specs.iter().position(|c| c.id == "coloc_detail__ClassA__roi_id").unwrap();
        let roi_ids: Vec<&str> = rows.iter().map(|r| r.values[col].as_str()).collect();
        assert!(roi_ids.contains(&"00000000-0000-0000-0000-000000000002"));
        assert!(roi_ids.contains(&"00000000-0000-0000-0000-000000000003"));
        assert!(
            roi_ids.contains(&"(missing)"),
            "the unresolved reference must be flagged, not rendered as a blank cell: {roi_ids:?}"
        );
    }

    #[test]
    fn flatten_coloc_rows_paginated_matches_single_shot() {
        // Simulates what bg_reload_coloc_detail_page0/bg_load_more_coloc_detail do:
        // flatten one page of source ROIs at a time, resolving only that page's
        // referenced partner ids (as `coloc_partner_ids` + an object_id_filter
        // query would), instead of flattening every source ROI against every
        // partner in one shot. The two must produce the same set of row values.
        let mut src1 = make_roi("img", vec![], None, 0, 0.0, 0.0, "{}");
        src1.object_id = "00000000-0000-0000-0000-000000000001".into();
        src1.coloc_json = r#"{"ClassA":["00000000-0000-0000-0000-000000000010"]}"#.into();
        let mut src2 = make_roi("img", vec![], None, 0, 0.0, 0.0, "{}");
        src2.object_id = "00000000-0000-0000-0000-000000000002".into();
        src2.coloc_json = r#"{"ClassA":["00000000-0000-0000-0000-000000000010","00000000-0000-0000-0000-000000000011"]}"#.into();
        let mut src3 = make_roi("img", vec![], None, 0, 0.0, 0.0, "{}");
        src3.object_id = "00000000-0000-0000-0000-000000000003".into();
        // No partners at all — exercises the "no partners" branch too.

        let mut p10 = make_roi("img", vec!["ClassA".into()], None, 0, 0.0, 0.0, "{}");
        p10.object_id = "00000000-0000-0000-0000-000000000010".into();
        let mut p11 = make_roi("img", vec!["ClassA".into()], None, 0, 0.0, 0.0, "{}");
        p11.object_id = "00000000-0000-0000-0000-000000000011".into();

        let source_rois = vec![src1, src2, src3];
        let all_partners = vec![p10, p11];
        let specs = build_coloc_detail_column_specs(&[], &["ClassA".to_string()]);

        let single_shot = flatten_coloc_rows(&source_rois, &all_partners, &specs);

        // Paginate source_rois into pages of 2, resolving each page's referenced
        // partner ids against `all_partners` (mirroring an object_id_filter query).
        const PAGE: usize = 2;
        let mut paginated = Vec::new();
        for page in source_rois.chunks(PAGE) {
            let ids = coloc_partner_ids(page);
            let page_partners: Vec<RoiRow> = all_partners
                .iter()
                .filter(|p| ids.iter().any(|id| id.eq_ignore_ascii_case(&p.object_id)))
                .cloned()
                .collect();
            paginated.extend(flatten_coloc_rows(page, &page_partners, &specs));
        }

        let single_shot_values: Vec<&Vec<String>> = single_shot.iter().map(|r| &r.values).collect();
        let paginated_values: Vec<&Vec<String>> = paginated.iter().map(|r| &r.values).collect();
        assert_eq!(
            single_shot_values, paginated_values,
            "paging must not change the flattened output, only how it's fetched"
        );
    }

    #[test]
    fn aggregate_rows_coloc_partner_count_is_aggregatable_metric() {
        let mut roi1 = make_roi("img", vec![], None, 0, 0.0, 0.0, "{}");
        roi1.coloc_json = r#"{"ClassA":["00000000-0000-0000-0000-000000000002"]}"#.into();
        let mut roi2 = make_roi("img", vec![], None, 0, 0.0, 0.0, "{}");
        roi2.coloc_json = r#"{"ClassA":["00000000-0000-0000-0000-000000000002","00000000-0000-0000-0000-000000000003"]}"#.into();

        let base_specs = build_column_specs(&[], &["ClassA".to_string()]);
        let config = GroupConfig {
            group_by: GroupBy::Image,
            regex: String::new(),
            aggs: vec![AggFunc::Max],
            split_colocalized: false,
            group_by_class: false,
        };
        let (specs, rows) = aggregate_rows(&[roi1, roi2], &config, &base_specs);

        let max_col = specs
            .iter()
            .position(|c| c.id == "coloc_partner__ClassA__count__max")
            .expect("ClassA coloc-partner count should be an aggregatable metric");
        assert_eq!(rows[0].values[max_col], "2");
    }

    // ---- group_by_class ----

    #[test]
    fn aggregate_rows_group_by_class_splits_into_one_row_per_class() {
        let roi_a = make_roi("img", vec!["ClassA".to_string()], None, 100, 0.0, 0.0, "{}");
        let roi_b = make_roi("img", vec!["ClassB".to_string()], None, 200, 0.0, 0.0, "{}");

        let config = GroupConfig {
            group_by: GroupBy::Image,
            regex: String::new(),
            aggs: vec![AggFunc::Avg],
            split_colocalized: false,
            group_by_class: true,
        };
        let (specs, rows) = aggregate_rows(&[roi_a, roi_b], &config, &build_column_specs(&[], &[]));

        assert_eq!(specs[0].id, "group");
        assert_eq!(specs[1].id, "group_class");
        assert_eq!(specs[2].id, "count");

        assert_eq!(rows.len(), 2, "one row per (image, class) pair");
        assert_eq!(rows[0].values[1], "ClassA");
        assert_eq!(rows[0].values[2], "1");
        assert_eq!(rows[1].values[1], "ClassB");
        assert_eq!(rows[1].values[2], "1");
    }

    #[test]
    fn aggregate_rows_group_by_class_multi_class_roi_fans_out_to_both_buckets() {
        let roi = make_roi(
            "img",
            vec!["ClassA".to_string(), "ClassB".to_string()],
            None,
            100,
            0.0,
            0.0,
            "{}",
        );

        let config = GroupConfig {
            group_by: GroupBy::Image,
            regex: String::new(),
            aggs: vec![AggFunc::Avg],
            split_colocalized: false,
            group_by_class: true,
        };
        let (_specs, rows) = aggregate_rows(&[roi], &config, &build_column_specs(&[], &[]));

        assert_eq!(
            rows.len(),
            2,
            "a ROI carrying two classes contributes to both class buckets"
        );
        assert_eq!(rows[0].values[1], "ClassA");
        assert_eq!(rows[1].values[1], "ClassB");
    }

    // ---- sort_display_rows ----

    #[test]
    fn sort_display_rows_orders_numerically_ascending_and_descending() {
        let specs = vec![ColumnSpec {
            id: "area_px".into(),
            label: "Area".into(),
            filterable: false,
            visible: true,
        }];
        let mut rows = vec![
            DisplayRow { roi_id: 1, object_id: String::new(), values: vec!["300".into()], stripe: true },
            DisplayRow { roi_id: 2, object_id: String::new(), values: vec!["100".into()], stripe: false },
            DisplayRow { roi_id: 3, object_id: String::new(), values: vec!["200".into()], stripe: true },
        ];

        sort_display_rows(&mut rows, &specs, "area_px", true);
        assert_eq!(rows.iter().map(|r| r.roi_id).collect::<Vec<_>>(), vec![2, 3, 1]);

        sort_display_rows(&mut rows, &specs, "area_px", false);
        assert_eq!(rows.iter().map(|r| r.roi_id).collect::<Vec<_>>(), vec![1, 3, 2]);
    }

    #[test]
    fn sort_display_rows_falls_back_to_text_for_non_numeric_values() {
        let specs = vec![ColumnSpec {
            id: "group".into(),
            label: "Group".into(),
            filterable: false,
            visible: true,
        }];
        let mut rows = vec![
            DisplayRow { roi_id: 1, object_id: String::new(), values: vec!["banana".into()], stripe: true },
            DisplayRow { roi_id: 2, object_id: String::new(), values: vec!["apple".into()], stripe: false },
        ];
        sort_display_rows(&mut rows, &specs, "group", true);
        assert_eq!(rows.iter().map(|r| r.roi_id).collect::<Vec<_>>(), vec![2, 1]);
    }

    #[test]
    fn sort_display_rows_unknown_column_leaves_rows_untouched() {
        let specs = vec![ColumnSpec {
            id: "area_px".into(),
            label: "Area".into(),
            filterable: false,
            visible: true,
        }];
        let mut rows = vec![
            DisplayRow { roi_id: 1, object_id: String::new(), values: vec!["300".into()], stripe: true },
            DisplayRow { roi_id: 2, object_id: String::new(), values: vec!["100".into()], stripe: false },
        ];
        sort_display_rows(&mut rows, &specs, "does_not_exist", true);
        assert_eq!(rows.iter().map(|r| r.roi_id).collect::<Vec<_>>(), vec![1, 2]);
    }
}
