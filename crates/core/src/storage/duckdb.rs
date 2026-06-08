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
/// For both `image_filter` and `class_filter`:
/// - `None`       → no restriction (return all)
/// - `Some([])`   → active filter with nothing selected → return 0 rows
/// - `Some([..])` → return only rows matching these values
#[derive(Debug, Clone)]
pub struct RoiFilter {
    pub image_filter: Option<Vec<String>>,
    pub class_filter: Option<Vec<String>>,
    /// Rows per page; 0 means return all.
    pub page_size: usize,
    /// Zero-based page index.
    pub page: usize,
    /// When false, `intensities_json` is replaced with an empty string in the
    /// query result, avoiding JSON parsing cost for hidden channel columns.
    pub fetch_intensities: bool,
}

impl Default for RoiFilter {
    fn default() -> Self {
        Self {
            image_filter: None,
            class_filter: None,
            page_size: 500,
            page: 0,
            fetch_intensities: true,
        }
    }
}

fn sql_in_list(items: &[String]) -> String {
    items
        .iter()
        .map(|s| format!("'{}'", s.replace('\'', "''")))
        .collect::<Vec<_>>()
        .join(", ")
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

        // Active filter with empty selection → nothing passes.
        if filter
            .image_filter
            .as_deref()
            .map_or(false, |v| v.is_empty())
        {
            return Ok(vec![]);
        }
        if filter
            .class_filter
            .as_deref()
            .map_or(false, |v| v.is_empty())
        {
            return Ok(vec![]);
        }

        let mut conditions: Vec<String> = Vec::new();

        if let Some(images) = &filter.image_filter {
            conditions.push(format!("image_name IN ({})", sql_in_list(images)));
        }
        if let Some(classes) = &filter.class_filter {
            conditions.push(format!(
                "CASE WHEN json_array_length(object_class_name) = 0 \
                 THEN COALESCE(seg_class_name, '') \
                 ELSE array_to_string(CAST(object_class_name AS VARCHAR[]), ', ') END IN ({})",
                sql_in_list(classes)
            ));
        }

        let where_clause = if conditions.is_empty() {
            String::new()
        } else {
            format!("WHERE {}", conditions.join(" AND "))
        };

        let pagination = if filter.page_size > 0 {
            format!(
                "LIMIT {} OFFSET {}",
                filter.page_size,
                filter.page * filter.page_size
            )
        } else {
            String::new()
        };

        let sql = format!(
            "{} {} ORDER BY object_id {}",
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

    /// Returns all distinct computed class strings present in the file, sorted alphabetically.
    pub fn get_class_names(&self) -> Result<Vec<String>, InternalErrors> {
        let err = |e: duckdb::Error| InternalErrors::Io(e.to_string());
        let sql = "SELECT DISTINCT \
                   CASE WHEN json_array_length(object_class_name) = 0 \
                   THEN COALESCE(seg_class_name, '') \
                   ELSE array_to_string(CAST(object_class_name AS VARCHAR[]), ', ') END AS class_str \
                   FROM rois ORDER BY class_str";
        let mut stmt = self.conn.prepare(sql).map_err(err)?;
        stmt.query_map([], |row| row.get(0))
            .map_err(err)?
            .collect::<Result<Vec<_>, _>>()
            .map_err(err)
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
