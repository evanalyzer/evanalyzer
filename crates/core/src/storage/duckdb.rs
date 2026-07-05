use crate::pipeline::pipeline_cache::PipelineCache;
use crate::roi::Intensity;
use crate::storage::PipelineResultExporter;
use duckdb::types::Value;
use duckdb::{Connection, params};
use evanalyzer_cfg::core_types::{InternalErrors, ObjectClass, ObjectId};
use indexmap::IndexMap;
use std::collections::HashMap;
use std::path::Path;
use std::path::PathBuf;
use std::sync::Mutex;

pub struct DuckDbExporter {
    // Connection is Send but !Sync; the Mutex makes the struct Sync so it can
    // satisfy the `PipelineResultExporter: Send + Sync` bound.
    conn: Mutex<Connection>,
    /// Maps ObjectClass → human-readable name from project classification settings.
    pub class_names: HashMap<ObjectClass, String>,
}

impl DuckDbExporter {
    /// Opens (or creates) the output file, runs DDL once, and returns a ready exporter.
    pub fn new(
        output_path: impl Into<PathBuf>,
        class_names: HashMap<ObjectClass, String>,
    ) -> Result<Self, InternalErrors> {
        let path: PathBuf = output_path.into();
        // These two log lines bracket the DuckDB DDL.  On the Windows (MinGW /
        // x86_64-pc-windows-gnu) build the bundled DuckDB C++ core can crash
        // natively during the first query: if the log shows "opened, running DDL"
        // but never "DDL complete" the crash is inside DuckDB itself - see the
        // build note in the README about using the MSVC toolchain on Windows.
        log::info!("DuckDB: opening {} and running DDL ...", path.display());
        let conn = Connection::open(&path).map_err(|e| InternalErrors::Io(e.to_string()))?;
        conn.execute_batch(CREATE_TABLES)
            .map_err(|e| InternalErrors::Io(e.to_string()))?;

        // Tuning for sustained tile-by-tile appends:
        //
        // * checkpoint_threshold: DuckDB defaults to folding the WAL back into the
        //   main database file every ~16 MB. With thousands of ROIs per image that
        //   fold triggers repeatedly *during* the run, and each one is a blocking,
        //   multi-hundred-ms-to-second stall — exactly the "sometimes the write
        //   takes more than 1 second" symptom. Raising the threshold defers the
        //   fold until the connection closes. Every append is still durably written
        //   to the WAL on disk, so a crash loses nothing and RAM stays flat.
        // * preserve_insertion_order: we never depend on physical row order (every
        //   read query uses ORDER BY), so turning this off lets the appender skip
        //   per-row ordering bookkeeping.
        conn.execute_batch(
            "SET preserve_insertion_order = false;
             SET checkpoint_threshold = '1GB';",
        )
        .map_err(|e| InternalErrors::Io(e.to_string()))?;

        log::info!("DuckDB: DDL complete, exporter ready");
        Ok(Self {
            conn: Mutex::new(conn),
            class_names,
        })
    }

    fn class_label(&self, class: &ObjectClass) -> String {
        match class {
            ObjectClass::Unset => "unset".to_string(),
            ObjectClass::Valid(n) => match self.class_names.get(class) {
                Some(name) => format!("{} ({})", name, n),
                None => format!("class_{}", n),
            },
        }
    }
}

// ---------------------------------------------------------------------------
// DDL
// ---------------------------------------------------------------------------

const CREATE_TABLES: &str = "
CREATE TABLE IF NOT EXISTS rois (
    image_name           VARCHAR NOT NULL,
    image_rel_path       VARCHAR NOT NULL,
    c_stack              INTEGER,
    z_stack              INTEGER,
    t_stack              INTEGER,
    object_id            UUID NOT NULL,
    seg_class_name       VARCHAR,
    seg_class_id         INTEGER,
    object_class_name    VARCHAR,
    object_class_id      VARCHAR,
    parent_id            VARCHAR,
    children             VARCHAR,
    track_id             UBIGINT,
    centroid_x_px        DOUBLE,
    centroid_y_px        DOUBLE,
    centroid_x_nm        DOUBLE,
    centroid_y_nm        DOUBLE,
    bbox_xmin_px         UINTEGER,
    bbox_ymin_px         UINTEGER,
    bbox_xmax_px         UINTEGER,
    bbox_ymax_px         UINTEGER,
    bbox_xmin_nm         DOUBLE,
    bbox_ymin_nm         DOUBLE,
    bbox_xmax_nm         DOUBLE,
    bbox_ymax_nm         DOUBLE,
    area_px              UBIGINT,
    area_nm2             DOUBLE,
    perimeter_px         DOUBLE,
    perimeter_nm         DOUBLE,
    circularity          DOUBLE,
    solidity             DOUBLE,
    aspect_ratio         DOUBLE,
    roundness            DOUBLE,
    compactness          DOUBLE,
    major_axis_px        DOUBLE,
    minor_axis_px        DOUBLE,
    major_axis_nm        DOUBLE,
    minor_axis_nm        DOUBLE,
    major_axis_angle     DOUBLE,
    eccentricity         DOUBLE,
    feret_diameter_px    DOUBLE,
    min_feret_px         DOUBLE,
    feret_diameter_nm    DOUBLE,
    min_feret_nm         DOUBLE,
    touches_edge         BOOLEAN,
    pixel_size_x_nm      DOUBLE,
    pixel_size_y_nm      DOUBLE,
    pixel_size_z_nm      DOUBLE,
    image_bit_depth      UTINYINT,
    intensities_json     JSON,
    coloc_json           JSON
);

CREATE TABLE IF NOT EXISTS coloc_stats (
    image               VARCHAR NOT NULL,
    source_class        VARCHAR NOT NULL,
    target_class        VARCHAR NOT NULL,
    n_colocalized       UBIGINT,
    avg_targets_per_roi DOUBLE,
    total_source_rois   UBIGINT
);
";

// ---------------------------------------------------------------------------
// Helpers for list columns passed as JSON strings to CAST(? AS T[])
// ---------------------------------------------------------------------------

fn json_string_array(values: &[String]) -> String {
    let items: Vec<String> = values
        .iter()
        .map(|s| format!("\"{}\"", s.replace('\\', "\\\\").replace('"', "\\\"")))
        .collect();
    format!("[{}]", items.join(","))
}

fn json_int_array(values: &[i32]) -> String {
    let items: Vec<String> = values.iter().map(|n| n.to_string()).collect();
    format!("[{}]", items.join(","))
}

// ---------------------------------------------------------------------------
// JSON serialisation helpers
// ---------------------------------------------------------------------------

fn coloc_to_json(
    colocalized_with: &IndexMap<ObjectClass, Vec<ObjectId>>,
    label: &dyn Fn(&ObjectClass) -> String,
) -> String {
    let mut entries = Vec::with_capacity(colocalized_with.len());
    for (class, ids) in colocalized_with {
        let ids_str = ids
            .iter()
            .map(|id| format!("\"{}\"", id))
            .collect::<Vec<_>>()
            .join(",");
        entries.push(format!("\"{}\":[{}]", label(class), ids_str));
    }
    format!("{{{}}}", entries.join(","))
}

fn intensities_to_json(intensities: &IndexMap<i32, Intensity>, bit_max: f64) -> String {
    let mut entries = Vec::with_capacity(intensities.len());
    for (ch, v) in intensities {
        // Mean is the precomputed per-channel average (sum / area), so the DB matches
        // what the rest of the app reports rather than re-deriving it here.
        let mean = v.avg_intensity as f64;
        let min = v.min_intensity as f64;
        let max = v.max_intensity as f64;
        entries.push(format!(
            "\"{}\":{{\"sum_raw\":{:.6},\"sum_scaled\":{:.2},\
                               \"mean_raw\":{:.6},\"mean_scaled\":{:.2},\
                               \"min_raw\":{:.6},\"min_scaled\":{:.2},\
                               \"max_raw\":{:.6},\"max_scaled\":{:.2}}}",
            ch,
            v.sum_intensity,
            v.sum_intensity * bit_max,
            mean,
            mean * bit_max,
            min,
            min * bit_max,
            max,
            max * bit_max,
        ));
    }
    format!("{{{}}}", entries.join(","))
}

// ---------------------------------------------------------------------------
// Pre-aggregated colocalization statistics
// ---------------------------------------------------------------------------

struct ColocStat {
    source_class: String,
    target_class: String,
    n_colocalized: u64,
    avg_targets_per_roi: f64,
    total_source_rois: u64,
}

fn compute_coloc_stats(
    cache: &PipelineCache,
    label: &dyn Fn(&ObjectClass) -> String,
) -> Vec<ColocStat> {
    let mut total_per_class: HashMap<String, u64> = HashMap::new();
    for roi in cache.roi_cache.values() {
        for class in &roi.object_class {
            *total_per_class.entry(label(class)).or_default() += 1;
        }
    }

    let mut agg: HashMap<(String, String), (u64, u64)> = HashMap::new();
    for roi in cache.roi_cache.values() {
        for src_class in &roi.object_class {
            let src = label(src_class);
            for (tgt_class, ids) in &roi.colocalized_with {
                if ids.is_empty() {
                    continue;
                }
                let tgt = label(tgt_class);
                let e = agg.entry((src.clone(), tgt)).or_default();
                e.0 += 1;
                e.1 += ids.len() as u64;
            }
        }
    }

    agg.into_iter()
        .map(|((src_class, tgt_class), (n_coloc, sum_targets))| {
            let total = *total_per_class.get(&src_class).unwrap_or(&1);
            ColocStat {
                avg_targets_per_roi: sum_targets as f64 / total.max(1) as f64,
                total_source_rois: total,
                n_colocalized: n_coloc,
                source_class: src_class,
                target_class: tgt_class,
            }
        })
        .collect()
}

// ---------------------------------------------------------------------------
// PipelineResultExporter impl
// ---------------------------------------------------------------------------

impl PipelineResultExporter for DuckDbExporter {
    fn export(&self, cache: &PipelineCache) -> Result<(), InternalErrors> {
        let conn = self.conn.lock().expect("DuckDB connection mutex poisoned");

        let px = &cache.image_cache.image_meta.pixel_sizes;
        let nr_of_bits = cache.image_cache.image_meta.nr_of_bits;
        let bit_max = ((1u64 << nr_of_bits) - 1) as f64;
        let px_len = (px.px_size_x * px.px_size_y).sqrt() as f64;
        let pxx = px.px_size_x as f64;
        let pxy = px.px_size_y as f64;

        let image_rel = cache.image_rel_path.display().to_string();
        let image_name = cache
            .image_rel_path
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_else(|| image_rel.clone());

        let label = |c: &ObjectClass| self.class_label(c);

        // --- ROI rows via Appender ---
        // List columns (object_class_name, object_class_id, children, parent_id)
        // are stored as VARCHAR JSON strings; the read query casts them back to
        // typed arrays so the reader code needs no changes.
        // The Appender flushes its buffer to disk when it is dropped, giving
        // constant memory usage regardless of how many images are processed.
        {
            let mut app = conn
                .appender("rois")
                .map_err(|e| InternalErrors::Io(e.to_string()))?;

            for roi in cache.roi_cache.values() {
                // get_perimeter()/get_ellipse() are precomputed at ROI creation on the
                // parallel workers (see Roi::finalize_geometry), so here on the single
                // writer thread they are just field reads. We pull each into a local and
                // derive the dependent metrics (circularity/roundness from the perimeter;
                // min_feret/aspect_ratio from the ellipse) to build the row from one read.
                let perimeter_f32 = roi.get_perimeter();
                let perimeter = perimeter_f32 as f64;
                let ellipse = roi.get_ellipse();
                let centroid = roi.get_centroid();
                let feret = roi.get_feret_diameter() as f64;
                let min_feret = ellipse.minor as f64;
                let aspect_ratio = if ellipse.minor > 0.0 {
                    (ellipse.major / ellipse.minor) as f64
                } else {
                    1.0
                };

                let parent_id: Option<String> = roi.parent_id.as_ref().map(|id| id.to_string());
                let track_id: u64 = roi.track.id.0;

                let object_class_names: Vec<String> = roi
                    .object_class
                    .iter()
                    .filter(|c| **c != ObjectClass::Unset)
                    .map(|c| label(c))
                    .collect();
                let object_class_ids: Vec<i32> = roi
                    .object_class
                    .iter()
                    .filter_map(|c| match c {
                        ObjectClass::Valid(n) => Some(*n as i32),
                        ObjectClass::Unset => None,
                    })
                    .collect();
                let children_ids: Vec<String> =
                    roi.children.iter().map(|id| id.to_string()).collect();

                let object_class_names_json = json_string_array(&object_class_names);
                let object_class_ids_json = json_int_array(&object_class_ids);
                let children_json = json_string_array(&children_ids);
                let coloc_json = coloc_to_json(&roi.colocalized_with, &label);
                let intensities_json = intensities_to_json(&roi.intensities, bit_max);

                let seg_class_name = roi.segmentation_class.to_string();
                let seg_class_id = roi.segmentation_class.0 as i32;
                let object_id = roi.id.to_string();
                let centroid_x_px = centroid.0 as f64;
                let centroid_y_px = centroid.1 as f64;
                let perimeter_nm = perimeter * px_len;
                let area_px = roi.area as u64;
                let area_nm2 = roi.area as f64 * pxx * pxy;
                // circularity and roundness use the identical 4π·area/perimeter² formula,
                // so compute it once from the perimeter local. (get_roundness also guards
                // perimeter == 0, which roi.circularity() does not.)
                let roundness = roi.get_roundness(perimeter_f32) as f64;
                let circularity = roundness;
                let compactness = roi.get_compactness(perimeter_f32) as f64;
                let feret_nm = feret * px_len;
                let min_feret_nm = min_feret * px_len;
                let px_size_z = px.px_size_z as f64;

                app.append_row(params![
                    &image_name,                   // image_name
                    &image_rel,                    // image_rel_path
                    roi.plane.c,                   // c_stack
                    roi.plane.z,                   // z_stack
                    roi.plane.t,                   // t_stack
                    &object_id,                    // object_id (VARCHAR → UUID column)
                    &seg_class_name,               // seg_class_name
                    seg_class_id,                  // seg_class_id
                    &object_class_names_json,      // object_class_name (VARCHAR JSON)
                    &object_class_ids_json,        // object_class_id   (VARCHAR JSON)
                    &parent_id,                    // parent_id         (VARCHAR)
                    &children_json,                // children          (VARCHAR JSON)
                    track_id,                      // track_id
                    centroid_x_px,                 // centroid_x_px
                    centroid_y_px,                 // centroid_y_px
                    centroid_x_px * pxx,           // centroid_x_nm
                    centroid_y_px * pxy,           // centroid_y_nm
                    roi.bbox[0],                   // bbox_xmin_px
                    roi.bbox[1],                   // bbox_ymin_px
                    roi.bbox[2],                   // bbox_xmax_px
                    roi.bbox[3],                   // bbox_ymax_px
                    roi.bbox[0] as f64 * pxx,      // bbox_xmin_nm
                    roi.bbox[1] as f64 * pxy,      // bbox_ymin_nm
                    roi.bbox[2] as f64 * pxx,      // bbox_xmax_nm
                    roi.bbox[3] as f64 * pxy,      // bbox_ymax_nm
                    area_px,                       // area_px
                    area_nm2,                      // area_nm2
                    perimeter,                     // perimeter_px
                    perimeter_nm,                  // perimeter_nm
                    circularity,                   // circularity
                    roi.get_solidity() as f64,     // solidity
                    aspect_ratio,                  // aspect_ratio
                    roundness,                     // roundness
                    compactness,                   // compactness
                    ellipse.major as f64,          // major_axis_px
                    ellipse.minor as f64,          // minor_axis_px
                    ellipse.major as f64 * px_len, // major_axis_nm
                    ellipse.minor as f64 * px_len, // minor_axis_nm
                    ellipse.angle as f64,          // major_axis_angle
                    ellipse.eccentricity as f64,   // eccentricity
                    feret,                         // feret_diameter_px
                    min_feret,                     // min_feret_px
                    feret_nm,                      // feret_diameter_nm
                    min_feret_nm,                  // min_feret_nm
                    roi.touches_edge,              // touches_edge
                    pxx,                           // pixel_size_x_nm
                    pxy,                           // pixel_size_y_nm
                    px_size_z,                     // pixel_size_z_nm
                    nr_of_bits,                    // image_bit_depth
                    &intensities_json,             // intensities_json
                    &coloc_json,                   // coloc_json
                ])
                .map_err(|e| InternalErrors::Io(e.to_string()))?;
            }
            // Appender flushes to disk on drop
        }

        // --- Colocalization statistics ---
        {
            let stats = compute_coloc_stats(cache, &label);
            let mut app = conn
                .appender("coloc_stats")
                .map_err(|e| InternalErrors::Io(e.to_string()))?;

            for s in stats {
                app.append_row(params![
                    &image_rel,
                    s.source_class,
                    s.target_class,
                    s.n_colocalized,
                    s.avg_targets_per_roi,
                    s.total_source_rois,
                ])
                .map_err(|e| InternalErrors::Io(e.to_string()))?;
            }
        }

        Ok(())
    }
}

// ---------------------------------------------------------------------------
// DuckDbReader
// ---------------------------------------------------------------------------

/// Flat DTO for a row in the `rois` table.
#[derive(Debug, Clone)]
pub struct RoiRow {
    pub image_name: String,
    pub image_rel_path: String,
    pub c_stack: Option<i32>,
    pub z_stack: Option<i32>,
    pub t_stack: Option<i32>,
    pub object_id: String,
    pub seg_class_name: Option<String>,
    pub seg_class_id: Option<i32>,
    pub object_class_name: Vec<String>,
    pub object_class_id: Vec<i32>,
    pub parent_id: Option<String>,
    pub children: Vec<String>,
    pub track_id: u64,
    pub centroid_x_px: f64,
    pub centroid_y_px: f64,
    pub centroid_x_nm: f64,
    pub centroid_y_nm: f64,
    pub area_px: u64,
    pub area_nm2: f64,
    pub perimeter_px: f64,
    pub perimeter_nm: f64,
    pub circularity: f64,
    pub solidity: f64,
    pub aspect_ratio: f64,
    pub roundness: f64,
    pub compactness: f64,
    pub major_axis_px: f64,
    pub minor_axis_px: f64,
    pub touches_edge: bool,
    pub intensities_json: String,
    pub coloc_json: String,
    /// Pixel-space bounding box `[xmin, ymin, xmax, ymax]`, used to highlight the
    /// ROI in the editor viewport when its results row is selected.
    pub bbox_px: [u32; 4],
}

/// Filter criteria for [`DuckDbReader::get_rois`].
///
/// For `image_filter`, `class_filter` and `coloc_filter`:
/// - `None`       → no restriction (return all)
/// - `Some([])`   → active filter with nothing selected → return 0 rows
/// - `Some([..])` → return only rows matching these values
#[derive(Debug, Clone)]
pub struct RoiFilter {
    pub image_filter: Option<Vec<String>>,
    pub class_filter: Option<Vec<String>>,
    /// Restricts to rows whose colocalization status label ("Yes"/"No") is in this set.
    pub coloc_filter: Option<Vec<String>>,
    /// Restricts to rows whose `object_id` is in this set. Unlike the other
    /// filters this is typically used *alone* (no image/class/coloc filter set
    /// alongside it) to fetch an exact, known set of ROIs — e.g. the specific
    /// colocalization partners referenced by one page of source ROIs — without
    /// re-deriving or re-checking the source-side filters.
    pub object_id_filter: Option<Vec<String>>,
    /// Restricts to ROIs from this single time-frame index, or `None` for
    /// every frame (today's default behavior).
    pub t_stack_filter: Option<i32>,
    /// Restricts to ROIs from this single Z-depth index, or `None` for every
    /// depth (today's default behavior).
    pub z_stack_filter: Option<i32>,
    /// Rows per page; 0 means return all.
    pub page_size: usize,
    /// Zero-based page index.
    pub page: usize,
    /// When false, `intensities_json` is replaced with an empty string in the
    /// query result, avoiding JSON parsing cost for hidden channel columns.
    pub fetch_intensities: bool,
    /// Results-table column id to sort by (see [`column_order_expr`] for which
    /// ids are supported). `None` keeps the default `ORDER BY object_id`.
    pub sort_column: Option<String>,
    pub sort_ascending: bool,
}

impl Default for RoiFilter {
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
            fetch_intensities: true,
            sort_column: None,
            sort_ascending: true,
        }
    }
}

// ---------------------------------------------------------------------------
// Grouped/aggregated queries (see `DuckDbReader::aggregate_rois`)
// ---------------------------------------------------------------------------

/// How to compute each ROI's grouping key in [`DuckDbReader::aggregate_rois`].
/// Kept free of `evanalyzer_app` types (`GroupBy`/`GroupConfig`) so the core
/// crate doesn't depend on the app crate — the app-side orchestration
/// (`aggregate_rois_sql` in `results_loader.rs`) translates `GroupBy` into one
/// of these.
pub enum GroupKeyMode {
    /// Group by `image_name` directly (covers both `GroupBy::None` and
    /// `GroupBy::Image` — grouping by image is meaningless when there's no
    /// grouping at all, but `aggregate_rois` is only ever called when
    /// `group_by != GroupBy::None`).
    Image,
    /// Group by a precomputed `image_rel_path -> key` mapping (`GroupBy::Folder`).
    /// The caller builds this via [`DuckDbReader::get_distinct_images`] plus
    /// Rust's own `folder_of()` — `std::path::Path::parent()`'s exact semantics
    /// (including its `"(root)"` fallback) have no faithful single SQL
    /// expression, so the directory-splitting logic itself stays in Rust; only
    /// the resulting small lookup table is pushed into the query as a `CASE`.
    ImageRelPathMap(HashMap<String, String>),
    /// Group by a regex applied to `image_name` (`GroupBy::Regex`). `pattern`
    /// must already be a syntactically valid RE2 pattern (validated by the
    /// caller via the `regex` crate before calling `aggregate_rois` — DuckDB
    /// uses RE2 internally too, so a pattern Rust's `regex` crate accepts
    /// should also be accepted here). `has_capture_group` selects whether the
    /// grouping key is capture group 1 or the whole match (group 0) — mirrors
    /// `results_loader.rs`'s `group_key`, which prefers the first capture
    /// group and falls back to the whole match when the pattern has none.
    /// Rows whose `image_name` doesn't match `pattern` are dropped entirely.
    Regex { pattern: String, has_capture_group: bool },
}

/// Describes one grouped/aggregated query for [`DuckDbReader::aggregate_rois`].
pub struct AggregateSpec {
    pub key_mode: GroupKeyMode,
    /// Fans each ROI out into one bucket per class it carries (mirrors
    /// `results_loader.rs`'s `roi_classes`/`GroupConfig::group_by_class`).
    pub group_by_class: bool,
    /// Additionally splits each group into a colocalizing and a
    /// non-colocalizing bucket (mirrors `GroupConfig::split_colocalized`).
    pub split_colocalized: bool,
    /// Results-table column ids to aggregate — every id here must be one
    /// [`column_order_expr`] maps to a scalar SQL expression (callers filter
    /// via `results_loader.rs`'s `is_numeric_metric` first, which only ever
    /// admits such ids).
    pub metric_ids: Vec<String>,
    /// SQL aggregate function names applied to every metric, e.g. `"MIN"`,
    /// `"AVG"`, `"MEDIAN"`, `"STDDEV_SAMP"`. One output column per
    /// `(metric_id, agg_fn)` pair, in that nested order (metric-major).
    pub agg_fns: Vec<&'static str>,
}

/// One grouped/aggregated result row from [`DuckDbReader::aggregate_rois`].
pub struct AggregatedRow {
    pub group_key: String,
    /// `Some(_)` iff `AggregateSpec::group_by_class` was set.
    pub group_class: Option<String>,
    /// `Some(_)` iff `AggregateSpec::split_colocalized` was set.
    pub colocalized: Option<bool>,
    pub count: i64,
    /// One entry per `(metric_id, agg_fn)` pair in
    /// `AggregateSpec::metric_ids` × `AggregateSpec::agg_fns` order
    /// (metric-major). `None` means every contributing ROI was NULL for that
    /// metric (e.g. a channel column no ROI in the bucket has intensity data
    /// for) — SQL aggregate functions skip NULL inputs the same way
    /// `results_loader.rs`'s `apply_agg` only ever collects `Some` values, so
    /// an all-NULL bucket and an all-absent bucket mean the same thing.
    pub metric_values: Vec<Option<f64>>,
}

/// SQL expression computing the same display class string used throughout the
/// results table/filters/export: the joined `object_class_name` list, falling
/// back to `seg_class_name` when a ROI carries no object classes.
fn class_case_expr() -> &'static str {
    "CASE WHEN json_array_length(object_class_name) = 0 \
     THEN COALESCE(seg_class_name, '') \
     ELSE array_to_string(CAST(object_class_name AS VARCHAR[]), ', ') END"
}

/// SQL expression producing the per-ROI list of classes to fan out over for
/// `AggregateSpec::group_by_class` — mirrors `class_case_expr`'s fallback to
/// `seg_class_name`, but as a one-or-more-element array (via `UNNEST` in the
/// `FROM` clause) instead of a single comma-joined string, so a multi-class
/// ROI contributes to each of its classes' group buckets separately (matching
/// `results_loader.rs`'s `roi_classes`). Always yields at least one element,
/// even when a ROI carries no object classes, so the `UNNEST` cross join
/// never silently drops a row.
fn class_fanout_expr() -> &'static str {
    "CASE WHEN json_array_length(object_class_name) = 0 \
     THEN [COALESCE(seg_class_name, '')] \
     ELSE CAST(object_class_name AS VARCHAR[]) END"
}

/// SQL expression for one of `RoiRow`'s scaled per-channel intensity stats
/// (`min_scaled` / `max_scaled` / `mean_scaled`, see `intensities_to_json`).
///
/// Uses the scalar-path form of `json_extract` (a single string path), not
/// the bracket/list form (`json_extract(col, ['{ch}'])`) — the list form
/// returns a JSON array of results (one per requested path) rather than a
/// scalar, and extracting a field from that array via `->>` throws a
/// "Malformed JSON" error the moment any row's `intensities_json` is missing
/// the requested channel key (e.g. a ROI with no measured intensities at all,
/// whose `intensities_json` is `"{}"`) — a real crash on sort-by-channel-column
/// today for any file where even one ROI lacks that channel's data. The
/// scalar form instead yields SQL NULL for a missing key, matching how
/// `parse_intensities`/`metric_value` already treat "channel absent" in Rust.
fn channel_stat_expr(ch: i32, stat: &str) -> String {
    format!("CAST(json_extract(intensities_json, '{ch}') ->> '{stat}' AS DOUBLE)")
}

/// SQL expression for the number of `class`-colocalizing partners a ROI has —
/// the same value the `coloc_partner__{class}__count` results-table column shows.
fn coloc_partner_count_expr(class: &str) -> String {
    format!(
        "COALESCE(json_array_length(coloc_json -> '{}'), 0)",
        class.replace('\'', "''")
    )
}

/// Maps a results-table column id to the SQL expression `ORDER BY` should sort
/// on, or `None` if that column isn't backed by a sortable SQL value (e.g. the
/// comma-joined `coloc_partner__{class}__ids` text column).
fn column_order_expr(col_id: &str) -> Option<String> {
    match col_id {
        "roi_id" => Some("object_id".to_string()),
        "image" => Some("image_name".to_string()),
        "class" => Some(class_case_expr().to_string()),
        "area_px" => Some("area_px".to_string()),
        "area_nm2" => Some("area_nm2".to_string()),
        "circularity" => Some("circularity".to_string()),
        "colocalized" => {
            Some("(coloc_json IS NOT NULL AND coloc_json != '' AND coloc_json != '{}')".to_string())
        }
        id if id.starts_with("coloc_partner__") => {
            let (class, suffix) = id.strip_prefix("coloc_partner__")?.rsplit_once("__")?;
            (suffix == "count").then(|| coloc_partner_count_expr(class))
        }
        id if id.starts_with("ch") => {
            let rest = id.strip_prefix("ch")?;
            let under = rest.find('_')?;
            let ch: i32 = rest[..under].parse().ok()?;
            let stat = match &rest[under + 1..] {
                "min_bit" => "min_scaled",
                "max_bit" => "max_scaled",
                "avg_bit" => "mean_scaled",
                _ => return None,
            };
            Some(channel_stat_expr(ch, stat))
        }
        _ => None,
    }
}

fn sql_in_list(items: &[String]) -> String {
    items
        .iter()
        .map(|s| format!("'{}'", s.replace('\'', "''")))
        .collect::<Vec<_>>()
        .join(", ")
}

/// `(min, max)` of `column` (one of the fixed `"t_stack"`/`"z_stack"`
/// literals below — never user input), or `None` when there's nothing to
/// step through: the column is all NULL, or every non-NULL value is the same.
fn stack_range(conn: &Connection, column: &str) -> Result<Option<(i32, i32)>, InternalErrors> {
    let err = |e: duckdb::Error| InternalErrors::Io(e.to_string());
    let sql = format!(
        "SELECT MIN({column}), MAX({column}), COUNT(DISTINCT {column}) FROM rois WHERE {column} IS NOT NULL"
    );
    let mut stmt = conn.prepare(&sql).map_err(err)?;
    let (min, max, distinct_count): (Option<i32>, Option<i32>, i64) = stmt
        .query_row([], |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)))
        .map_err(err)?;
    if distinct_count <= 1 {
        return Ok(None);
    }
    match (min, max) {
        (Some(min), Some(max)) => Ok(Some((min, max))),
        _ => Ok(None),
    }
}

// ---------------------------------------------------------------------------
// Coloc filter vocabulary
//
// `RoiFilter::coloc_filter` items are one of three kinds of label, matched
// literally against the value coming back from the GUI's filter popup:
//   - "No"                          -> ROI has no colocalization partners
//   - "Yes (any class)"             -> ROI colocalizes with at least one class
//   - "Colocalizes with <class>"    -> ROI colocalizes with that specific class
// These helpers are the single source of truth for that vocabulary so the SQL
// builder below and the GUI (which populates the filter popup) never drift.
// ---------------------------------------------------------------------------

pub fn coloc_filter_label_no() -> &'static str {
    "No"
}

pub fn coloc_filter_label_any() -> &'static str {
    "Yes (any class)"
}

pub fn coloc_filter_label_with(class: &str) -> String {
    format!("Colocalizes with {class}")
}

fn parse_coloc_filter_with(label: &str) -> Option<&str> {
    label.strip_prefix("Colocalizes with ")
}

/// Builds the SQL boolean expression for a single selected coloc-filter label.
fn coloc_filter_condition(label: &str) -> String {
    if label == coloc_filter_label_no() {
        "(coloc_json IS NULL OR coloc_json = '' OR coloc_json = '{}')".to_string()
    } else if label == coloc_filter_label_any() {
        "(coloc_json IS NOT NULL AND coloc_json != '' AND coloc_json != '{}')".to_string()
    } else if let Some(class) = parse_coloc_filter_with(label) {
        format!(
            "COALESCE(json_array_length(coloc_json -> '{}'), 0) > 0",
            class.replace('\'', "''")
        )
    } else {
        // Unrecognized label (shouldn't happen if the GUI only ever sends labels
        // produced by the helpers above) -> matches nothing.
        "FALSE".to_string()
    }
}

/// True if any of `filter`'s `Option<Vec<_>>` fields is `Some(&[])` — an active
/// filter with nothing selected, which always means zero matching rows. Shared
/// by [`DuckDbReader::get_rois`] and [`DuckDbReader::aggregate_rois`] so both
/// short-circuit identically instead of hitting the database for a query that
/// can never return anything.
fn filter_has_empty_selection(filter: &RoiFilter) -> bool {
    filter.image_filter.as_deref().is_some_and(|v| v.is_empty())
        || filter.class_filter.as_deref().is_some_and(|v| v.is_empty())
        || filter.coloc_filter.as_deref().is_some_and(|v| v.is_empty())
        || filter.object_id_filter.as_deref().is_some_and(|v| v.is_empty())
}

/// Builds the list of SQL boolean conditions `filter` implies (unjoined).
/// Shared by [`DuckDbReader::get_rois`] and [`DuckDbReader::aggregate_rois`]
/// so the two queries can never drift on what a given filter means;
/// `aggregate_rois` also appends its own `regexp_matches` condition on top of
/// this list before joining, for `GroupKeyMode::Regex`.
fn filter_conditions(filter: &RoiFilter) -> Vec<String> {
    let mut conditions: Vec<String> = Vec::new();

    if let Some(images) = &filter.image_filter {
        conditions.push(format!("image_name IN ({})", sql_in_list(images)));
    }
    if let Some(classes) = &filter.class_filter {
        conditions.push(format!(
            "{} IN ({})",
            class_case_expr(),
            sql_in_list(classes)
        ));
    }
    if let Some(labels) = &filter.coloc_filter {
        // Every entry `Colocalization` ever writes into `colocalized_with` has a non-empty
        // ID list (see coloc_rois.rs), so `coloc_json` is exactly "{}" / empty iff the ROI
        // has no colocalization partner — no need to parse the object to check.
        let fragments: Vec<String> = labels.iter().map(|l| coloc_filter_condition(l)).collect();
        conditions.push(format!("({})", fragments.join(" OR ")));
    }
    if let Some(ids) = &filter.object_id_filter {
        // Cast the column rather than the literals, so this reuses `sql_in_list`
        // unchanged and doesn't depend on DuckDB's UUID-literal coercion rules.
        conditions.push(format!(
            "CAST(object_id AS VARCHAR) IN ({})",
            sql_in_list(ids)
        ));
    }
    if let Some(t) = filter.t_stack_filter {
        conditions.push(format!("t_stack = {t}"));
    }
    if let Some(z) = filter.z_stack_filter {
        conditions.push(format!("z_stack = {z}"));
    }

    conditions
}

/// Builds the `WHERE ...` clause (or `""` if `conditions` is empty) by
/// joining an already-built condition list with `AND`.
fn where_clause_from(conditions: &[String]) -> String {
    if conditions.is_empty() {
        String::new()
    } else {
        format!("WHERE {}", conditions.join(" AND "))
    }
}

/// Builds the `WHERE ...` clause (or `""` if `filter` has no active
/// conditions) for `filter`.
fn build_where_clause(filter: &RoiFilter) -> String {
    where_clause_from(&filter_conditions(filter))
}

/// Opens a result file written by [`DuckDbExporter`] for reading.
pub struct DuckDbReader {
    conn: Connection,
}

fn roi_select_sql(fetch_intensities: bool) -> String {
    let intensities_expr = if fetch_intensities {
        "intensities_json"
    } else {
        "'' AS intensities_json"
    };
    format!(
        "SELECT image_name, image_rel_path,\n\
                c_stack, z_stack, t_stack,\n\
                object_id, seg_class_name, seg_class_id,\n\
                CAST(object_class_name AS VARCHAR[]),\n\
                CAST(object_class_id   AS INTEGER[]),\n\
                parent_id,\n\
                CAST(children AS VARCHAR[]),\n\
                track_id,\n\
                centroid_x_px, centroid_y_px, centroid_x_nm, centroid_y_nm,\n\
                area_px, area_nm2, perimeter_px, perimeter_nm,\n\
                circularity, solidity, aspect_ratio, roundness, compactness,\n\
                major_axis_px, minor_axis_px,\n\
                touches_edge, {intensities_expr}, coloc_json,\n\
                bbox_xmin_px, bbox_ymin_px, bbox_xmax_px, bbox_ymax_px\n\
         FROM rois"
    )
}

impl DuckDbReader {
    pub fn open(path: &Path) -> Result<Self, InternalErrors> {
        let conn = Connection::open(path).map_err(|e| InternalErrors::Io(e.to_string()))?;
        Ok(Self { conn })
    }

    /// Returns ROIs matching `filter`, with optional pagination.
    pub fn get_rois(&self, filter: &RoiFilter) -> Result<Vec<RoiRow>, InternalErrors> {
        let err = |e: duckdb::Error| InternalErrors::Io(e.to_string());

        if filter_has_empty_selection(filter) {
            return Ok(vec![]);
        }

        let where_clause = build_where_clause(filter);

        let pagination = if filter.page_size > 0 {
            format!(
                "LIMIT {} OFFSET {}",
                filter.page_size,
                filter.page * filter.page_size
            )
        } else {
            String::new()
        };

        // `object_id` is always the final tiebreaker so pages stay stably ordered
        // even when many rows share the same sort-column value.
        let order_by = match filter
            .sort_column
            .as_deref()
            .and_then(column_order_expr)
        {
            Some(expr) => format!(
                "ORDER BY {expr} {}, object_id",
                if filter.sort_ascending { "ASC" } else { "DESC" }
            ),
            None => "ORDER BY object_id".to_string(),
        };

        let sql = format!(
            "{} {} {order_by} {}",
            roi_select_sql(filter.fetch_intensities),
            where_clause,
            pagination
        );
        let mut stmt = self.conn.prepare(&sql).map_err(err)?;
        stmt.query_map([], map_roi_row)
            .map_err(err)?
            .collect::<Result<Vec<_>, _>>()
            .map_err(err)
    }

    /// Returns distinct `(image_rel_path, image_name)` pairs matching `filter`'s
    /// image/class/coloc/object_id/t/z restrictions (`page_size`/`page`/
    /// `sort_column` are ignored — this is never paginated). Cheap: proportional
    /// to the number of distinct images, not the number of ROIs. Used to
    /// precompute a `GroupKeyMode::ImageRelPathMap` for [`Self::aggregate_rois`]
    /// (`GroupBy::Folder`'s directory-splitting logic stays in Rust — see
    /// `folder_of` in `results_loader.rs` — since `std::path::Path::parent()`'s
    /// semantics have no faithful single SQL expression).
    pub fn get_distinct_images(
        &self,
        filter: &RoiFilter,
    ) -> Result<Vec<(String, String)>, InternalErrors> {
        let err = |e: duckdb::Error| InternalErrors::Io(e.to_string());

        if filter_has_empty_selection(filter) {
            return Ok(vec![]);
        }

        let where_clause = build_where_clause(filter);
        let sql = format!(
            "SELECT DISTINCT image_rel_path, image_name FROM rois {where_clause} \
             ORDER BY image_rel_path"
        );
        let mut stmt = self.conn.prepare(&sql).map_err(err)?;
        stmt.query_map([], |row| Ok((row.get(0)?, row.get(1)?)))
            .map_err(err)?
            .collect::<Result<Vec<_>, _>>()
            .map_err(err)
    }

    /// Computes grouped/aggregated results directly in DuckDB via `GROUP BY`,
    /// instead of pulling every matching row into Rust first (see
    /// `evanalyzer_app::results_loader::aggregate_rows`, the in-memory Rust
    /// implementation this mirrors and is tested against for parity).
    ///
    /// Returns one [`AggregatedRow`] per group, in the same order
    /// `aggregate_rows` would (group key, then class, then colocalized flag,
    /// all ascending) — callers rely on this order matching exactly.
    pub fn aggregate_rois(
        &self,
        filter: &RoiFilter,
        spec: &AggregateSpec,
    ) -> Result<Vec<AggregatedRow>, InternalErrors> {
        let err = |e: duckdb::Error| InternalErrors::Io(e.to_string());

        if filter_has_empty_selection(filter) {
            return Ok(vec![]);
        }

        let esc = |s: &str| s.replace('\'', "''");

        let mut conditions = filter_conditions(filter);
        let group_key_expr = match &spec.key_mode {
            GroupKeyMode::Image => "image_name".to_string(),
            GroupKeyMode::ImageRelPathMap(map) => {
                if map.is_empty() {
                    // No images match `filter` — nothing will be selected anyway,
                    // but keep the expression syntactically valid.
                    "NULL".to_string()
                } else {
                    let cases: Vec<String> = map
                        .iter()
                        .map(|(path, key)| format!("WHEN '{}' THEN '{}'", esc(path), esc(key)))
                        .collect();
                    format!("CASE image_rel_path {} ELSE '(root)' END", cases.join(" "))
                }
            }
            GroupKeyMode::Regex { pattern, has_capture_group } => {
                // Rows that don't match are dropped entirely (not grouped under
                // an empty/NULL key) — `regexp_extract` alone can't distinguish
                // "matched, captured empty string" from "no match", so that's
                // enforced by the separate `regexp_matches` condition below.
                conditions.push(format!("regexp_matches(image_name, '{}')", esc(pattern)));
                let group_idx = if *has_capture_group { 1 } else { 0 };
                format!("regexp_extract(image_name, '{}', {group_idx})", esc(pattern))
            }
        };

        // `FROM rois[, UNNEST(...) AS u(group_class)]` — UNNEST is only valid
        // here as a lateral cross join in the FROM clause, not in the SELECT
        // list combined with GROUP BY (DuckDB rejects that form outright).
        let from_clause = if spec.group_by_class {
            format!(
                "rois, UNNEST({}) AS u(group_class)",
                class_fanout_expr()
            )
        } else {
            "rois".to_string()
        };

        let mut select_cols = vec![format!("{group_key_expr} AS group_key")];
        let mut group_by_cols = vec!["group_key".to_string()];
        if spec.group_by_class {
            select_cols.push("u.group_class AS group_class".to_string());
            group_by_cols.push("group_class".to_string());
        }
        if spec.split_colocalized {
            // Reuses the exact boolean expression `column_order_expr("colocalized")`
            // already uses for sorting, so the two can never define "colocalized"
            // differently.
            let expr = column_order_expr("colocalized").expect("colocalized is always mapped");
            select_cols.push(format!("{expr} AS colocalized"));
            group_by_cols.push("colocalized".to_string());
        }
        select_cols.push("COUNT(*) AS cnt".to_string());

        for metric_id in &spec.metric_ids {
            let Some(metric_expr) = column_order_expr(metric_id) else {
                // Every id here should already be one `column_order_expr` maps
                // (callers only ever pass ids `is_numeric_metric` accepted, a
                // strict subset) — skip defensively rather than panic if not.
                continue;
            };
            for agg_fn in &spec.agg_fns {
                let agg_expr = if *agg_fn == "STDDEV_SAMP" {
                    // `results_loader.rs`'s `apply_agg` is only ever called on a
                    // non-empty value list (`aggregate_rows` renders "" directly
                    // for an empty one without calling it), and its `Stdev`
                    // branch treats a single-value input as `0.0` (sample stdev
                    // is undefined for n=1, but the Rust ground-truth still
                    // reports a number there, not a blank cell) — whereas SQL
                    // `STDDEV_SAMP` returns NULL for both n=0 and n=1. Recover
                    // the n=1 case specifically: if `STDDEV_SAMP` is NULL but at
                    // least one row contributed a non-null value, use `0.0`.
                    format!(
                        "COALESCE(STDDEV_SAMP({metric_expr}), \
                         CASE WHEN COUNT({metric_expr}) >= 1 THEN 0.0 END)"
                    )
                } else {
                    format!("{agg_fn}({metric_expr})")
                };
                select_cols.push(format!("{agg_expr} AS \"{metric_id}__{agg_fn}\""));
            }
        }

        let where_clause = where_clause_from(&conditions);
        let order_by = format!("ORDER BY {}", group_by_cols.join(", "));
        let sql = format!(
            "SELECT {} FROM {from_clause} {where_clause} GROUP BY {} {order_by}",
            select_cols.join(", "),
            group_by_cols.join(", "),
        );

        let mut stmt = self.conn.prepare(&sql).map_err(err)?;
        let metric_count: usize = spec
            .metric_ids
            .iter()
            .filter(|id| column_order_expr(id).is_some())
            .count()
            * spec.agg_fns.len();
        let group_by_class = spec.group_by_class;
        let split_colocalized = spec.split_colocalized;
        stmt.query_map([], move |row| {
            let mut idx = 0usize;
            let group_key: String = row.get(idx)?;
            idx += 1;
            let group_class: Option<String> = if group_by_class {
                let v = row.get(idx)?;
                idx += 1;
                v
            } else {
                None
            };
            let colocalized: Option<bool> = if split_colocalized {
                let v: bool = row.get(idx)?;
                idx += 1;
                Some(v)
            } else {
                None
            };
            let count: i64 = row.get(idx)?;
            idx += 1;
            let mut metric_values = Vec::with_capacity(metric_count);
            for _ in 0..metric_count {
                let v: Option<f64> = row.get(idx)?;
                metric_values.push(v);
                idx += 1;
            }
            Ok(AggregatedRow {
                group_key,
                group_class,
                colocalized,
                count,
                metric_values,
            })
        })
        .map_err(err)?
        .collect::<Result<Vec<_>, _>>()
        .map_err(err)
    }

    /// Returns all distinct computed class strings present in the file, sorted alphabetically.
    pub fn get_class_names(&self) -> Result<Vec<String>, InternalErrors> {
        let err = |e: duckdb::Error| InternalErrors::Io(e.to_string());
        let sql = format!(
            "SELECT DISTINCT {} AS class_str FROM rois ORDER BY class_str",
            class_case_expr()
        );
        let mut stmt = self.conn.prepare(&sql).map_err(err)?;
        stmt.query_map([], |row| row.get(0))
            .map_err(err)?
            .collect::<Result<Vec<_>, _>>()
            .map_err(err)
    }

    /// Returns the `(min, max)` of `t_stack` present in the file, or `None`
    /// when there's nothing to step through — either every ROI's `t_stack` is
    /// NULL (no time axis) or they all share the same single value.
    pub fn get_t_stack_range(&self) -> Result<Option<(i32, i32)>, InternalErrors> {
        stack_range(&self.conn, "t_stack")
    }

    /// Returns the `(min, max)` of `z_stack` present in the file, or `None`
    /// when there's nothing to step through — either every ROI's `z_stack` is
    /// NULL (no depth axis) or they all share the same single value.
    pub fn get_z_stack_range(&self) -> Result<Option<(i32, i32)>, InternalErrors> {
        stack_range(&self.conn, "z_stack")
    }

    /// Returns all distinct `image_name` values present in the file, sorted alphabetically.
    pub fn get_image_names(&self) -> Result<Vec<String>, InternalErrors> {
        let err = |e: duckdb::Error| InternalErrors::Io(e.to_string());
        let mut stmt = self
            .conn
            .prepare("SELECT DISTINCT image_name FROM rois ORDER BY image_name")
            .map_err(err)?;
        stmt.query_map([], |row| row.get(0))
            .map_err(err)?
            .collect::<Result<Vec<_>, _>>()
            .map_err(err)
    }

    /// Returns all distinct partner-class labels appearing as keys in any ROI's
    /// `coloc_json`, sorted alphabetically. Used to populate the "Colocalizes
    /// with <class>" entries of the coloc filter popup.
    pub fn get_coloc_partner_class_names(&self) -> Result<Vec<String>, InternalErrors> {
        let err = |e: duckdb::Error| InternalErrors::Io(e.to_string());
        // Filter in a subquery first: UNNEST in the outer SELECT list is evaluated
        // against every scanned row before the WHERE predicate is applied, so
        // `json_keys('')` would otherwise be hit for rows with no colocalization.
        let sql = "SELECT DISTINCT UNNEST(json_keys(t.cj)) AS k FROM ( \
                       SELECT coloc_json AS cj FROM rois \
                       WHERE coloc_json IS NOT NULL AND coloc_json != '' AND coloc_json != '{}' \
                   ) AS t \
                   ORDER BY k";
        let mut stmt = self.conn.prepare(sql).map_err(err)?;
        stmt.query_map([], |row| row.get(0))
            .map_err(err)?
            .collect::<Result<Vec<_>, _>>()
            .map_err(err)
    }
}

fn map_roi_row(row: &duckdb::Row<'_>) -> duckdb::Result<RoiRow> {
    Ok(RoiRow {
        image_name: row.get(0)?,
        image_rel_path: row.get(1)?,
        c_stack: row.get(2)?,
        z_stack: row.get(3)?,
        t_stack: row.get(4)?,
        object_id: row.get(5)?,
        seg_class_name: row.get(6)?,
        seg_class_id: row.get(7)?,
        object_class_name: extract_string_list(row.get::<_, Value>(8)?),
        object_class_id: extract_int_list(row.get::<_, Value>(9)?),
        parent_id: row.get(10)?,
        children: extract_string_list(row.get::<_, Value>(11)?),
        track_id: row.get(12)?,
        centroid_x_px: row.get(13)?,
        centroid_y_px: row.get(14)?,
        centroid_x_nm: row.get(15)?,
        centroid_y_nm: row.get(16)?,
        area_px: row.get(17)?,
        area_nm2: row.get(18)?,
        perimeter_px: row.get(19)?,
        perimeter_nm: row.get(20)?,
        circularity: row.get(21)?,
        solidity: row.get(22)?,
        aspect_ratio: row.get(23)?,
        roundness: row.get(24)?,
        compactness: row.get(25)?,
        major_axis_px: row.get(26)?,
        minor_axis_px: row.get(27)?,
        touches_edge: row.get(28)?,
        intensities_json: row.get::<_, Option<String>>(29)?.unwrap_or_default(),
        coloc_json: row.get::<_, Option<String>>(30)?.unwrap_or_default(),
        bbox_px: [row.get(31)?, row.get(32)?, row.get(33)?, row.get(34)?],
    })
}

fn extract_string_list(val: Value) -> Vec<String> {
    match val {
        Value::List(items) | Value::Array(items) => items
            .into_iter()
            .filter_map(|v| {
                if let Value::Text(s) = v {
                    Some(s)
                } else {
                    None
                }
            })
            .collect(),
        _ => vec![],
    }
}

fn extract_int_list(val: Value) -> Vec<i32> {
    match val {
        Value::List(items) | Value::Array(items) => items
            .into_iter()
            .filter_map(|v| if let Value::Int(n) = v { Some(n) } else { None })
            .collect(),
        _ => vec![],
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::roi::{Roi, RoiInit};
    use bitvec::prelude::*;
    use evanalyzer_cfg::core_types::{ObjectClass, ObjectId};
    use tempfile::TempDir;

    fn make_filled_roi(id: u128, bbox: [u32; 4]) -> Roi {
        let [x_min, y_min, x_max, y_max] = bbox;
        let w = (x_max - x_min + 1) as usize;
        let h = (y_max - y_min + 1) as usize;
        let area = w * h;
        let mask_data = BitVec::<u64, Lsb0>::repeat(true, area);
        Roi::new(RoiInit {
            id: ObjectId(id),
            bbox,
            mask_data,
            area,
            ..Default::default()
        })
    }

    fn make_filled_roi_at_plane(id: u128, bbox: [u32; 4], t: i32, z: i32) -> Roi {
        let mut roi = make_filled_roi(id, bbox);
        roi.plane.t = t;
        roi.plane.z = z;
        roi
    }

    fn export_and_open(rois: Vec<Roi>) -> (TempDir, DuckDbReader) {
        let dir = TempDir::new().expect("failed to create temp dir");
        let path = dir.path().join("results.duckdb");

        let mut cache = PipelineCache::default();
        for roi in rois {
            cache.roi_cache.insert(roi.id.clone(), roi);
        }

        let exporter = DuckDbExporter::new(&path, HashMap::new()).expect("exporter init failed");
        exporter.export(&cache).expect("export failed");
        drop(exporter); // flushes the appenders / closes the connection

        let reader = DuckDbReader::open(&path).expect("reader open failed");
        (dir, reader)
    }

    /// Like `export_and_open`, but each `(image_rel_path, rois)` pair is
    /// exported as its own image (mirroring real usage — `PipelineCache`
    /// carries one `image_rel_path` per `export()` call), so the fixture can
    /// span more than one distinct image. Needed for `aggregate_rois` tests
    /// (`GroupBy::Image`/`Folder`/`Regex` are meaningless with only one image).
    fn export_multi_image_and_open(images: Vec<(&str, Vec<Roi>)>) -> (TempDir, DuckDbReader) {
        let dir = TempDir::new().expect("failed to create temp dir");
        let path = dir.path().join("results.duckdb");

        let exporter = DuckDbExporter::new(&path, HashMap::new()).expect("exporter init failed");
        for (image_rel_path, rois) in images {
            let mut cache = PipelineCache {
                image_rel_path: image_rel_path.into(),
                ..Default::default()
            };
            for roi in rois {
                cache.roi_cache.insert(roi.id.clone(), roi);
            }
            exporter.export(&cache).expect("export failed");
        }
        drop(exporter);

        let reader = DuckDbReader::open(&path).expect("reader open failed");
        (dir, reader)
    }

    #[test]
    fn aggregate_rois_groups_by_image_and_averages() {
        let a1 = make_filled_roi(1, [0, 0, 9, 9]); // area_px = 100
        let a2 = make_filled_roi(2, [0, 0, 19, 9]); // area_px = 200
        let b1 = make_filled_roi(3, [0, 0, 29, 9]); // area_px = 300

        let (_dir, reader) = export_multi_image_and_open(vec![
            ("img_a.tif", vec![a1, a2]),
            ("img_b.tif", vec![b1]),
        ]);

        let spec = AggregateSpec {
            key_mode: GroupKeyMode::Image,
            group_by_class: false,
            split_colocalized: false,
            metric_ids: vec!["area_px".to_string()],
            agg_fns: vec!["AVG"],
        };
        let rows = reader.aggregate_rois(&RoiFilter::default(), &spec).unwrap();

        assert_eq!(rows.len(), 2, "one row per image");
        assert_eq!(rows[0].group_key, "img_a.tif");
        assert_eq!(rows[0].count, 2);
        assert_eq!(rows[0].metric_values, vec![Some(150.0)]);
        assert_eq!(rows[1].group_key, "img_b.tif");
        assert_eq!(rows[1].count, 1);
        assert_eq!(rows[1].metric_values, vec![Some(300.0)]);
    }

    #[test]
    fn aggregate_rois_group_by_class_fans_out_multi_class_roi() {
        const CLASS_A: ObjectClass = ObjectClass::Valid(1);
        const CLASS_B: ObjectClass = ObjectClass::Valid(2);

        let mut multi = make_filled_roi(1, [0, 0, 9, 9]);
        multi.add_object_class(CLASS_A);
        multi.add_object_class(CLASS_B);

        let (_dir, reader) = export_multi_image_and_open(vec![("img.tif", vec![multi])]);

        let spec = AggregateSpec {
            key_mode: GroupKeyMode::Image,
            group_by_class: true,
            split_colocalized: false,
            metric_ids: vec![],
            agg_fns: vec![],
        };
        let mut rows = reader.aggregate_rois(&RoiFilter::default(), &spec).unwrap();
        rows.sort_by(|a, b| a.group_class.cmp(&b.group_class));

        assert_eq!(rows.len(), 2, "a ROI carrying two classes contributes to both buckets");
        assert_eq!(rows[0].group_class.as_deref(), Some("class_1"));
        assert_eq!(rows[0].count, 1);
        assert_eq!(rows[1].group_class.as_deref(), Some("class_2"));
        assert_eq!(rows[1].count, 1);
    }

    #[test]
    fn aggregate_rois_split_colocalized_produces_two_buckets() {
        let mut coloc = make_filled_roi(1, [0, 0, 9, 9]);
        coloc.add_colocalizing_object(ObjectClass::Valid(9), ObjectId(99));
        let not_coloc = make_filled_roi(2, [0, 0, 9, 9]);

        let (_dir, reader) =
            export_multi_image_and_open(vec![("img.tif", vec![coloc, not_coloc])]);

        let spec = AggregateSpec {
            key_mode: GroupKeyMode::Image,
            group_by_class: false,
            split_colocalized: true,
            metric_ids: vec![],
            agg_fns: vec![],
        };
        let rows = reader.aggregate_rois(&RoiFilter::default(), &spec).unwrap();

        assert_eq!(rows.len(), 2, "one colocalizing + one non-colocalizing bucket");
        // false < true, matching Rust's Option<bool> BTreeMap ordering.
        assert_eq!(rows[0].colocalized, Some(false));
        assert_eq!(rows[0].count, 1);
        assert_eq!(rows[1].colocalized, Some(true));
        assert_eq!(rows[1].count, 1);
    }

    #[test]
    fn aggregate_rois_regex_groups_by_capture_and_drops_non_matching() {
        let a1 = make_filled_roi(1, [0, 0, 9, 9]);
        let a2 = make_filled_roi(2, [0, 0, 9, 9]);
        let control = make_filled_roi(3, [0, 0, 9, 9]);

        let (_dir, reader) = export_multi_image_and_open(vec![
            ("A1_01.tif", vec![a1]),
            ("A1_02.tif", vec![a2]),
            ("control.tif", vec![control]),
        ]);

        let spec = AggregateSpec {
            key_mode: GroupKeyMode::Regex {
                pattern: r"^([A-Z]\d+)_".to_string(),
                has_capture_group: true,
            },
            group_by_class: false,
            split_colocalized: false,
            metric_ids: vec![],
            agg_fns: vec![],
        };
        let rows = reader.aggregate_rois(&RoiFilter::default(), &spec).unwrap();

        assert_eq!(rows.len(), 1, "non-matching 'control.tif' must be dropped, not its own group");
        assert_eq!(rows[0].group_key, "A1");
        assert_eq!(rows[0].count, 2);
    }

    #[test]
    fn aggregate_rois_all_null_metric_bucket_is_none() {
        // Neither ROI has any intensities_json, so any channel metric is NULL
        // for every contributing row in the bucket.
        let a1 = make_filled_roi(1, [0, 0, 9, 9]);
        let a2 = make_filled_roi(2, [0, 0, 9, 9]);
        let (_dir, reader) = export_multi_image_and_open(vec![("img.tif", vec![a1, a2])]);

        let spec = AggregateSpec {
            key_mode: GroupKeyMode::Image,
            group_by_class: false,
            split_colocalized: false,
            metric_ids: vec!["ch0_min_bit".to_string()],
            agg_fns: vec!["AVG"],
        };
        let rows = reader.aggregate_rois(&RoiFilter::default(), &spec).unwrap();

        assert_eq!(rows.len(), 1);
        assert_eq!(
            rows[0].metric_values,
            vec![None],
            "an all-NULL metric bucket must be None, matching apply_agg's empty-vec -> blank-cell behavior"
        );
    }

    #[test]
    fn coloc_filter_restricts_to_matching_status() {
        const ID_COLOC: u128 = 1;
        const ID_NOT_COLOC: u128 = 2;
        const PARTNER_CLASS: ObjectClass = ObjectClass::Valid(1);
        let coloc_id = ObjectId(ID_COLOC).to_string();
        let not_coloc_id = ObjectId(ID_NOT_COLOC).to_string();

        let mut coloc_roi = make_filled_roi(ID_COLOC, [0, 0, 1, 1]);
        coloc_roi.add_colocalizing_object(PARTNER_CLASS, ObjectId(99));
        let not_coloc_roi = make_filled_roi(ID_NOT_COLOC, [5, 5, 6, 6]);

        let (_dir, reader) = export_and_open(vec![coloc_roi, not_coloc_roi]);

        let yes = reader
            .get_rois(&RoiFilter {
                coloc_filter: Some(vec![coloc_filter_label_any().into()]),
                ..Default::default()
            })
            .unwrap();
        assert_eq!(yes.len(), 1);
        assert_eq!(yes[0].object_id, coloc_id);

        let no = reader
            .get_rois(&RoiFilter {
                coloc_filter: Some(vec![coloc_filter_label_no().into()]),
                ..Default::default()
            })
            .unwrap();
        assert_eq!(no.len(), 1);
        assert_eq!(no[0].object_id, not_coloc_id);

        let all = reader.get_rois(&RoiFilter::default()).unwrap();
        assert_eq!(all.len(), 2, "no coloc_filter set → both rows returned");

        let none_selected = reader
            .get_rois(&RoiFilter {
                coloc_filter: Some(vec![]),
                ..Default::default()
            })
            .unwrap();
        assert!(
            none_selected.is_empty(),
            "active filter with nothing selected → 0 rows"
        );
    }

    #[test]
    fn object_id_filter_restricts_to_exact_ids_and_matches_display_format() {
        const ID_A: u128 = 1;
        const ID_B: u128 = 2;
        let id_a = ObjectId(ID_A).to_string();
        let id_b = ObjectId(ID_B).to_string();

        let roi_a = make_filled_roi(ID_A, [0, 0, 1, 1]);
        let roi_b = make_filled_roi(ID_B, [5, 5, 6, 6]);
        let (_dir, reader) = export_and_open(vec![roi_a, roi_b]);

        // Single id: exact match, and the round-tripped `object_id` string is
        // byte-identical to `ObjectId::to_string()` — confirms `CAST(object_id
        // AS VARCHAR)` produces the same lowercase-dashed format written at
        // export time, not just "a filter that happens to work".
        let single = reader
            .get_rois(&RoiFilter {
                object_id_filter: Some(vec![id_a.clone()]),
                ..Default::default()
            })
            .unwrap();
        assert_eq!(single.len(), 1);
        assert_eq!(single[0].object_id, id_a);

        // Two ids: both returned, in the default object_id order.
        let both = reader
            .get_rois(&RoiFilter {
                object_id_filter: Some(vec![id_b.clone(), id_a.clone()]),
                ..Default::default()
            })
            .unwrap();
        assert_eq!(both.len(), 2);
        assert_eq!(both[0].object_id, id_a);
        assert_eq!(both[1].object_id, id_b);

        // Unknown id: no rows.
        let unknown = reader
            .get_rois(&RoiFilter {
                object_id_filter: Some(vec![ObjectId(999).to_string()]),
                ..Default::default()
            })
            .unwrap();
        assert!(unknown.is_empty());

        // Active filter with nothing selected → 0 rows, same convention as the
        // other filters.
        let none_selected = reader
            .get_rois(&RoiFilter {
                object_id_filter: Some(vec![]),
                ..Default::default()
            })
            .unwrap();
        assert!(none_selected.is_empty());
    }

    #[test]
    fn coloc_partner_class_names_and_filter() {
        const ID_WITH_B: u128 = 1;
        const ID_WITH_C: u128 = 2;
        const PARTNER_B: ObjectClass = ObjectClass::Valid(2);
        const PARTNER_C: ObjectClass = ObjectClass::Valid(3);
        let id_with_b = ObjectId(ID_WITH_B).to_string();

        let mut roi_with_b = make_filled_roi(ID_WITH_B, [0, 0, 1, 1]);
        roi_with_b.add_colocalizing_object(PARTNER_B, ObjectId(99));
        let mut roi_with_c = make_filled_roi(ID_WITH_C, [5, 5, 6, 6]);
        roi_with_c.add_colocalizing_object(PARTNER_C, ObjectId(98));

        let (_dir, reader) = export_and_open(vec![roi_with_b, roi_with_c]);

        // class_label() falls back to "class_{n}" when no name map is provided.
        let partner_classes = reader.get_coloc_partner_class_names().unwrap();
        assert_eq!(partner_classes, vec!["class_2".to_string(), "class_3".to_string()]);

        let with_b = reader
            .get_rois(&RoiFilter {
                coloc_filter: Some(vec![coloc_filter_label_with("class_2")]),
                ..Default::default()
            })
            .unwrap();
        assert_eq!(with_b.len(), 1);
        assert_eq!(with_b[0].object_id, id_with_b);
    }

    #[test]
    fn get_rois_sorts_by_requested_column() {
        const ID_SMALL: u128 = 1;
        const ID_LARGE: u128 = 2;
        let small_id = ObjectId(ID_SMALL).to_string();
        let large_id = ObjectId(ID_LARGE).to_string();

        // bbox [0,0,1,1] -> 2x2 = 4px; bbox [0,0,9,9] -> 10x10 = 100px.
        let small = make_filled_roi(ID_SMALL, [0, 0, 1, 1]);
        let large = make_filled_roi(ID_LARGE, [0, 0, 9, 9]);
        let (_dir, reader) = export_and_open(vec![large, small]);

        let asc = reader
            .get_rois(&RoiFilter {
                sort_column: Some("area_px".into()),
                sort_ascending: true,
                ..Default::default()
            })
            .unwrap();
        assert_eq!(
            asc.iter().map(|r| r.object_id.clone()).collect::<Vec<_>>(),
            vec![small_id.clone(), large_id.clone()]
        );

        let desc = reader
            .get_rois(&RoiFilter {
                sort_column: Some("area_px".into()),
                sort_ascending: false,
                ..Default::default()
            })
            .unwrap();
        assert_eq!(
            desc.iter().map(|r| r.object_id.clone()).collect::<Vec<_>>(),
            vec![large_id, small_id]
        );
    }

    #[test]
    fn get_rois_sorts_by_channel_column_when_some_rois_have_no_intensities() {
        // Regression test for a latent bug in `channel_stat_expr`: sorting by a
        // channel column used to throw "Malformed JSON" the moment any row's
        // `intensities_json` was `"{}"` (no measured intensities at all), because
        // the old bracket-form `json_extract(col, ['0'])` returns a JSON array
        // wrapping a NULL rather than a scalar NULL, which `->>` can't handle.
        use crate::roi::Intensity;
        use indexmap::IndexMap;

        const ID_NO_INTENSITY: u128 = 1;
        const ID_HAS_INTENSITY: u128 = 2;
        let no_intensity_id = ObjectId(ID_NO_INTENSITY).to_string();
        let has_intensity_id = ObjectId(ID_HAS_INTENSITY).to_string();

        let no_intensity = make_filled_roi(ID_NO_INTENSITY, [0, 0, 1, 1]);
        let mut has_intensity = make_filled_roi(ID_HAS_INTENSITY, [0, 0, 1, 1]);
        has_intensity.intensities = IndexMap::from([(
            0,
            Intensity {
                sum_intensity: 5.0,
                min_intensity: 5.0,
                max_intensity: 5.0,
                avg_intensity: 5.0,
                pixel_values: vec![],
            },
        )]);

        let (_dir, reader) = export_and_open(vec![no_intensity, has_intensity]);

        let asc = reader
            .get_rois(&RoiFilter {
                sort_column: Some("ch0_min_bit".into()),
                sort_ascending: true,
                ..Default::default()
            })
            .unwrap();
        // The row with no intensities sorts as NULL (DuckDB's default: NULLs
        // last in ASC order), so the row with data comes first.
        assert_eq!(
            asc.iter().map(|r| r.object_id.clone()).collect::<Vec<_>>(),
            vec![has_intensity_id, no_intensity_id]
        );
    }

    #[test]
    fn get_rois_unsortable_column_falls_back_to_object_id_order() {
        let roi_a = make_filled_roi(1, [0, 0, 1, 1]);
        let roi_b = make_filled_roi(2, [0, 0, 1, 1]);
        let (_dir, reader) = export_and_open(vec![roi_b, roi_a]);

        let rows = reader
            .get_rois(&RoiFilter {
                sort_column: Some("coloc_partner__ClassA__ids".into()),
                ..Default::default()
            })
            .unwrap();
        assert_eq!(
            rows.iter().map(|r| r.object_id.clone()).collect::<Vec<_>>(),
            vec![ObjectId(1).to_string(), ObjectId(2).to_string()],
            "unsortable column id falls back to the default ORDER BY object_id"
        );
    }

    #[test]
    fn stack_range_is_none_when_every_roi_shares_one_value() {
        let roi_a = make_filled_roi_at_plane(1, [0, 0, 1, 1], 0, 0);
        let roi_b = make_filled_roi_at_plane(2, [5, 5, 6, 6], 0, 0);
        let (_dir, reader) = export_and_open(vec![roi_a, roi_b]);

        assert_eq!(reader.get_t_stack_range().unwrap(), None);
        assert_eq!(reader.get_z_stack_range().unwrap(), None);
    }

    #[test]
    fn stack_range_reports_min_max_when_values_differ() {
        let roi_t0 = make_filled_roi_at_plane(1, [0, 0, 1, 1], 0, 3);
        let roi_t2 = make_filled_roi_at_plane(2, [5, 5, 6, 6], 2, 3);
        let roi_t5 = make_filled_roi_at_plane(3, [9, 9, 10, 10], 5, 7);
        let (_dir, reader) = export_and_open(vec![roi_t0, roi_t2, roi_t5]);

        assert_eq!(reader.get_t_stack_range().unwrap(), Some((0, 5)));
        assert_eq!(reader.get_z_stack_range().unwrap(), Some((3, 7)));
    }

    #[test]
    fn get_rois_filters_by_t_stack_and_z_stack() {
        const ID_T0: u128 = 1;
        const ID_T2: u128 = 2;
        let id_t0 = ObjectId(ID_T0).to_string();
        let id_t2 = ObjectId(ID_T2).to_string();

        let roi_t0 = make_filled_roi_at_plane(ID_T0, [0, 0, 1, 1], 0, 1);
        let roi_t2 = make_filled_roi_at_plane(ID_T2, [5, 5, 6, 6], 2, 4);
        let (_dir, reader) = export_and_open(vec![roi_t0, roi_t2]);

        let only_t0 = reader
            .get_rois(&RoiFilter {
                t_stack_filter: Some(0),
                ..Default::default()
            })
            .unwrap();
        assert_eq!(only_t0.len(), 1);
        assert_eq!(only_t0[0].object_id, id_t0);

        let only_z4 = reader
            .get_rois(&RoiFilter {
                z_stack_filter: Some(4),
                ..Default::default()
            })
            .unwrap();
        assert_eq!(only_z4.len(), 1);
        assert_eq!(only_z4[0].object_id, id_t2);

        let both_unset = reader.get_rois(&RoiFilter::default()).unwrap();
        assert_eq!(both_unset.len(), 2, "no t/z filter set -> every frame returned");
    }
}

