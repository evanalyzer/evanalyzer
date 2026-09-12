use crate::object::Intensity;
use crate::pipeline::pipeline_cache::GlobalPipelineCache;
use crate::storage::PipelineResultExporter;
use duckdb::types::Value;
use duckdb::{Connection, params};
use evanalyzer_cfg::core_types::{InternalErrors, ObjectClass, ObjectId};
use indexmap::IndexMap;
use std::collections::HashMap;
use std::path::Path;
use std::path::PathBuf;
use std::sync::{LazyLock, Mutex};
use std::time::Instant;

/// One resident ("anchor") `Connection` per results file path, so every
/// caller in this file that needs a connection to a given `.evadb` shares
/// one already-open DuckDB `Database` instead of each doing its own
/// independent `Connection::open`.
///
/// This matters specifically on Windows: `Connection::open` on a path that
/// some *other* already-open `Connection` in this same process also has
/// open fails with a sharing-violation IO error ("The process cannot access
/// the file because it is being used by another process... File is already
/// open in <this exact exe/PID>") - not a lock held by another program, a
/// self-conflict against this process's own earlier connection. Before this
/// cache existed, `DuckDbReader::open` (called fresh on essentially every
/// results-panel query - see `results_loader.rs`) and `DuckDbExporter::new`
/// each opened the file independently, so any one of them still being open
/// (e.g. a slow aggregate query, or a long-running analyze job) made every
/// other concurrent open attempt on the same file fail outright, surfacing
/// as "failed to load ROIs"/"failed to load image names" warnings in the
/// results panels for as long as the slow one held its connection.
///
/// `Connection::try_clone` (used below) creates a new logical connection to
/// an *already-open* database - no new OS-level file handle, so it can't
/// collide with the anchor or with other clones of it. DuckDB's own
/// concurrency control (MVCC) coordinates reads/writes across clones of one
/// `Database` safely, which is exactly what multiple results panels (or a
/// live-preview writer running alongside a results viewer) need.
///
/// Never evicted: entries stay open for the life of the process. For this
/// app's actual usage (a handful of results files open per session, not
/// thousands) that's a better tradeoff than the complexity of an eviction
/// policy - the cost is a few extra open file handles hanging around, not a
/// resource leak that grows unboundedly during normal use.
static CONNECTION_CACHE: LazyLock<Mutex<HashMap<PathBuf, Connection>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

/// Returns a `Connection` usable for `path`, sharing the resident anchor
/// connection for that path if one is already open (see `CONNECTION_CACHE`'s
/// doc comment), opening and caching a fresh one otherwise.
fn shared_connection(path: &Path) -> Result<Connection, InternalErrors> {
    let to_io_err = |e: duckdb::Error| InternalErrors::Io(e.to_string());
    let mut cache = CONNECTION_CACHE.lock().unwrap_or_else(|e| e.into_inner());

    if let Some(anchor) = cache.get(path) {
        match anchor.try_clone() {
            Ok(handle) => return Ok(handle),
            // The anchor connection has gone bad (e.g. the file was deleted
            // or replaced out from under it) - drop it and fall through to
            // open a fresh one below, rather than keep handing out clones
            // of a connection that can no longer serve queries.
            Err(_) => {
                cache.remove(path);
            }
        }
    }

    let anchor = Connection::open(path).map_err(to_io_err)?;
    let handle = anchor.try_clone().map_err(to_io_err)?;
    cache.insert(path.to_path_buf(), anchor);
    Ok(handle)
}

/// Derives the display `image_name` (bare filename) from an image's relative
/// path, falling back to the full relative path when it has no filename
/// component. Shared by [`DuckDbExporter::export`] (writes `objects` rows)
/// and [`DuckDbExporter::finalize_image`] (writes the `images` row) so both
/// always agree on the same name for the same image.
fn image_display_name(image_rel_path: &Path) -> String {
    let rel = image_rel_path.display().to_string();
    image_rel_path
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or(rel)
}

pub struct DuckDbExporter {
    // Connection is Send but !Sync; the Mutex makes the struct Sync so it can
    // satisfy the `PipelineResultExporter: Send + Sync` bound.
    conn: Mutex<Connection>,
    /// Maps ObjectClass → human-readable name from project classification settings.
    pub class_names: HashMap<ObjectClass, (String, u32)>,
}

impl DuckDbExporter {
    /// Opens (or creates) the output file, runs DDL once, and returns a ready exporter.
    pub fn new(
        output_path: impl Into<PathBuf>,
        class_names: HashMap<ObjectClass, (String, u32)>,
    ) -> Result<Self, InternalErrors> {
        let path: PathBuf = output_path.into();
        // These two log lines bracket the DuckDB DDL.  On the Windows (MinGW /
        // x86_64-pc-windows-gnu) build the bundled DuckDB C++ core can crash
        // natively during the first query: if the log shows "opened, running DDL"
        // but never "DDL complete" the crash is inside DuckDB itself - see the
        // build note in the README about using the MSVC toolchain on Windows.
        log::info!("DuckDB: opening {} and running DDL ...", path.display());
        let conn = shared_connection(&path)?;
        conn.execute_batch(CREATE_TABLES)
            .map_err(|e| InternalErrors::Io(e.to_string()))?;

        // Snapshot the project's classification registry into the file
        // itself, once, up front (like `images`, this assumes a fresh output
        // file per job - see `generate_analyze_job_from_project_settings` -
        // so `class_id`'s PRIMARY KEY never collides with a prior run's
        // rows) — the results view resolves class names from this table
        // rather than from the live project (which may have renamed/added/
        // deleted classes since this file was written) or from whatever
        // ended up baked into each object row's `object_class_name` (which
        // only ever contains the classes actually assigned to some object,
        // missing any class with zero matches).
        {
            let mut app = conn
                .appender("classes")
                .map_err(|e| InternalErrors::Io(e.to_string()))?;
            for (class, (name, color)) in &class_names {
                if let ObjectClass::Valid(n) = class {
                    app.append_row(params![*n, name, color])
                        .map_err(|e| InternalErrors::Io(e.to_string()))?;
                }
            }
        }

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
                Some((name, _color)) => format!("{} ({})", name, n),
                None => format!("class_{}", n),
            },
        }
    }
}

// ---------------------------------------------------------------------------
// DDL
// ---------------------------------------------------------------------------

const CREATE_TABLES: &str = "
CREATE TABLE IF NOT EXISTS objects (
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
    avg_targets_per_object DOUBLE,
    total_source_objects   UBIGINT
);

CREATE TABLE IF NOT EXISTS images (
    image_name      VARCHAR NOT NULL,
    image_rel_path  VARCHAR NOT NULL PRIMARY KEY,
    successful      BOOLEAN NOT NULL DEFAULT true,
    error_message   VARCHAR,
    disabled        BOOLEAN NOT NULL DEFAULT false,
    width           UINTEGER NOT NULL,
    height          UINTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS classes (
    class_id  INTEGER NOT NULL PRIMARY KEY,
    name      VARCHAR NOT NULL,
    color     UINTEGER
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
    avg_targets_per_object: f64,
    total_source_objects: u64,
}

fn compute_coloc_stats(
    cache: &GlobalPipelineCache,
    label: &dyn Fn(&ObjectClass) -> String,
) -> Vec<ColocStat> {
    let mut total_per_class: HashMap<String, u64> = HashMap::new();
    for object in cache.object_cache.values() {
        for class in &object.object_class {
            *total_per_class.entry(label(class)).or_default() += 1;
        }
    }

    let mut agg: HashMap<(String, String), (u64, u64)> = HashMap::new();
    for object in cache.object_cache.values() {
        for src_class in &object.object_class {
            let src = label(src_class);
            for (tgt_class, ids) in &object.colocalized_with {
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
                avg_targets_per_object: sum_targets as f64 / total.max(1) as f64,
                total_source_objects: total,
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
    fn export(&self, cache: &GlobalPipelineCache) -> Result<(), InternalErrors> {
        let start = Instant::now();
        let object_count = cache.object_cache.len();
        let conn = self.conn.lock().expect("DuckDB connection mutex poisoned");
        // Objects and their colocalization stats must land together or not at
        // all - without a transaction, a failure partway through (disk full,
        // a DuckDB error) could leave one written with no matching rows in
        // the other. `unchecked_transaction` (rather than `transaction`,
        // which needs `&mut Connection`) is safe here because `conn` is
        // already the only handle to this connection, serialized by the
        // exporter's own Mutex - nothing else can be mid-transaction on it
        // concurrently. Uncommitted (the `?` early-returns below) rolls back
        // on drop.
        let tx = conn
            .unchecked_transaction()
            .map_err(|e| InternalErrors::Io(e.to_string()))?;

        let px = &cache.image_meta.pixel_sizes;
        let nr_of_bits = cache.image_meta.nr_of_bits;
        // Same implausible-bit-depth guard as image_reader.rs's read path -
        // `nr_of_bits` should already have been rejected there before a
        // cache carrying it could exist, but this shouldn't trust that
        // blindly: unguarded, `1u64 << nr_of_bits` for nr_of_bits > 63 is a
        // shift-by-too-large, silently producing a wrong (not NaN/Inf, since
        // this is only ever used as a multiplier below) scale factor instead
        // of an error.
        if !(1..=32).contains(&nr_of_bits) {
            return Err(InternalErrors::Generic(format!(
                "cannot export {}: implausible bit depth {nr_of_bits} (expected 1-32)",
                cache.image_rel_path.display()
            )));
        }
        let bit_max = ((1u64 << nr_of_bits) - 1) as f64;
        let px_len = (px.px_size_x * px.px_size_y).sqrt() as f64;
        let pxx = px.px_size_x as f64;
        let pxy = px.px_size_y as f64;

        let image_rel = cache.image_rel_path.display().to_string();
        let image_name = image_display_name(&cache.image_rel_path);

        let label = |c: &ObjectClass| self.class_label(c);

        // --- object rows via Appender ---
        // List columns (object_class_name, object_class_id, children, parent_id)
        // are stored as VARCHAR JSON strings; the read query casts them back to
        // typed arrays so the reader code needs no changes.
        // The Appender flushes its buffer to disk when it is dropped, giving
        // constant memory usage regardless of how many images are processed.
        {
            let mut app = tx
                .appender("objects")
                .map_err(|e| InternalErrors::Io(e.to_string()))?;

            for object in cache.object_cache.values() {
                // get_perimeter()/get_ellipse() are precomputed at object creation on the
                // parallel workers (see Object::finalize_geometry), so here on the single
                // writer thread they are just field reads. We pull each into a local and
                // derive the dependent metrics (circularity/roundness from the perimeter;
                // min_feret/aspect_ratio from the ellipse) to build the row from one read.
                let perimeter_f32 = object.get_perimeter();
                let perimeter = perimeter_f32 as f64;
                let ellipse = object.get_ellipse();
                let centroid = object.get_centroid();
                let feret = object.get_feret_diameter() as f64;
                let min_feret = ellipse.minor as f64;
                let aspect_ratio = if ellipse.minor > 0.0 {
                    (ellipse.major / ellipse.minor) as f64
                } else {
                    1.0
                };

                let parent_id: Option<String> = object.parent_id.as_ref().map(|id| id.to_string());
                let track_id: u64 = object.track.id.0;

                let object_class_names: Vec<String> = object
                    .object_class
                    .iter()
                    .filter(|c| **c != ObjectClass::Unset)
                    .map(|c| label(c))
                    .collect();
                let object_class_ids: Vec<i32> = object
                    .object_class
                    .iter()
                    .filter_map(|c| match c {
                        ObjectClass::Valid(n) => Some(*n as i32),
                        ObjectClass::Unset => None,
                    })
                    .collect();
                let children_ids: Vec<String> =
                    object.children.iter().map(|id| id.to_string()).collect();

                let object_class_names_json = json_string_array(&object_class_names);
                let object_class_ids_json = json_int_array(&object_class_ids);
                let children_json = json_string_array(&children_ids);
                let coloc_json = coloc_to_json(&object.colocalized_with, &label);
                let intensities_json = intensities_to_json(&object.intensities, bit_max);

                let seg_class_name = object.segmentation_class.to_string();
                let seg_class_id = object.segmentation_class.0 as i32;
                let object_id = object.id.to_string();
                let centroid_x_px = centroid.0 as f64;
                let centroid_y_px = centroid.1 as f64;
                let perimeter_nm = perimeter * px_len;
                let area_px = object.area as u64;
                let area_nm2 = object.area as f64 * pxx * pxy;
                // circularity and roundness use the identical 4π·area/perimeter² formula,
                // so compute it once from the perimeter local. (get_roundness also guards
                // perimeter == 0, which object.circularity() does not.)
                let roundness = object.get_roundness(perimeter_f32) as f64;
                let circularity = roundness;
                let compactness = object.get_compactness(perimeter_f32) as f64;
                let feret_nm = feret * px_len;
                let min_feret_nm = min_feret * px_len;
                let px_size_z = px.px_size_z as f64;

                app.append_row(params![
                    &image_name,                   // image_name
                    &image_rel,                    // image_rel_path
                    object.plane.c,                // c_stack
                    object.plane.z,                // z_stack
                    object.plane.t,                // t_stack
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
                    object.bbox[0],                // bbox_xmin_px
                    object.bbox[1],                // bbox_ymin_px
                    object.bbox[2],                // bbox_xmax_px
                    object.bbox[3],                // bbox_ymax_px
                    object.bbox[0] as f64 * pxx,   // bbox_xmin_nm
                    object.bbox[1] as f64 * pxy,   // bbox_ymin_nm
                    object.bbox[2] as f64 * pxx,   // bbox_xmax_nm
                    object.bbox[3] as f64 * pxy,   // bbox_ymax_nm
                    area_px,                       // area_px
                    area_nm2,                      // area_nm2
                    perimeter,                     // perimeter_px
                    perimeter_nm,                  // perimeter_nm
                    circularity,                   // circularity
                    object.get_solidity() as f64,  // solidity
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
                    object.touches_edge,           // touches_edge
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
            let mut app = tx
                .appender("coloc_stats")
                .map_err(|e| InternalErrors::Io(e.to_string()))?;

            for s in stats {
                app.append_row(params![
                    &image_rel,
                    s.source_class,
                    s.target_class,
                    s.n_colocalized,
                    s.avg_targets_per_object,
                    s.total_source_objects,
                ])
                .map_err(|e| InternalErrors::Io(e.to_string()))?;
            }
        }

        tx.commit().map_err(|e| InternalErrors::Io(e.to_string()))?;

        log::info!(
            "Database: exported {object_count} object(s) for {} in {:?}",
            cache.image_rel_path.display(),
            start.elapsed()
        );
        Ok(())
    }

    /// Records that this image was processed, regardless of whether it
    /// produced any objects - and regardless of whether `error` is set, so a
    /// partially-failed image still shows up rather than vanishing entirely.
    /// `error` is persisted into `successful`/`error_message` so that
    /// partial failure is still visible instead of looking identical to
    /// success. Called once per image, after every `export()` call for it
    /// has returned.
    fn finalize_image(
        &self,
        image_rel_path: &Path,
        width: u32,
        height: u32,
        error: Option<&str>,
    ) -> Result<(), InternalErrors> {
        let conn = self
            .conn
            .lock()
            .expect("Database connection mutex poisoned");
        let image_rel = image_rel_path.display().to_string();
        let image_name = image_display_name(image_rel_path);
        let successful = error.is_none();

        conn.execute(
            "INSERT INTO images (image_name, image_rel_path, successful, error_message, width, height) VALUES (?, ?, ?, ?, ?, ?)",
            params![image_name, image_rel, successful, error, width, height],
        )
        .map_err(|e| InternalErrors::Io(e.to_string()))?;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// DuckDbReader
// ---------------------------------------------------------------------------

/// Flat DTO for a row in the `images` table: one per processed image,
/// regardless of whether it produced any objects.
#[derive(Debug, Clone)]
pub struct ImageRow {
    pub image_name: String,
    pub image_rel_path: String,
    /// `true` if every tile/plane for this image exported successfully,
    /// `false` if it was finalized despite a failure partway through -
    /// see `error_message` for details, and `DuckDbExporter::finalize_image`
    /// for why a failed image still gets a row here rather than none at all.
    pub successful: bool,
    pub error_message: Option<String>,
    /// User-toggled via `DuckDbReader::set_image_disabled` - excluded from
    /// exports by default and shown crossed out in the Matrix view. Not set
    /// by the analysis pipeline itself (see `successful` for that).
    pub disabled: bool,
}

/// Flat DTO for a row in the `classes` table: one per registered class in
/// the project's classification settings at the moment the file was written.
#[derive(Debug, Clone)]
pub struct ClassRow {
    pub class_id: i32,
    pub name: String,
}

/// Flat DTO for a row in the `objects` table.
#[derive(Debug, Clone)]
pub struct ObjectRow {
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
    pub eccentricity: f64,
    pub touches_edge: bool,
    pub intensities_json: String,
    pub coloc_json: String,
    /// Pixel-space bounding box `[xmin, ymin, xmax, ymax]`, used to highlight the
    /// object in the editor viewport when its results row is selected.
    pub bbox_px: [u32; 4],
}

/// Filter criteria for [`DuckDbReader::get_objects`].
///
/// For `image_filter`, `class_filter` and `coloc_filter`:
/// - `None`       → no restriction (return all)
/// - `Some([])`   → active filter with nothing selected → return 0 rows
/// - `Some([..])` → return only rows matching these values
#[derive(Debug, Clone)]
pub struct ObjectFilter {
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

impl Default for ObjectFilter {
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
// Grouped/aggregated queries (see `DuckDbReader::aggregate_objects`)
// ---------------------------------------------------------------------------

/// How to compute each object's grouping key in [`DuckDbReader::aggregate_objects`].
/// Kept free of `evanalyzer_app` types (`GroupBy`/`GroupConfig`) so the core
/// crate doesn't depend on the app crate — the app-side orchestration
/// (`aggregate_objects_sql` in `results_loader.rs`) translates `GroupBy` into one
/// of these.
pub enum GroupKeyMode {
    /// Group by `image_name` directly (covers both `GroupBy::None` and
    /// `GroupBy::Image` — grouping by image is meaningless when there's no
    /// grouping at all, but `aggregate_objects` is only ever called when
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
    /// caller via the `regex` crate before calling `aggregate_objects` — DuckDB
    /// uses RE2 internally too, so a pattern Rust's `regex` crate accepts
    /// should also be accepted here). `has_capture_group` selects whether the
    /// grouping key is capture group 1 or the whole match (group 0) — mirrors
    /// `results_loader.rs`'s `group_key`, which prefers the first capture
    /// group and falls back to the whole match when the pattern has none.
    /// Rows whose `image_name` doesn't match `pattern` are dropped entirely.
    Regex {
        pattern: String,
        has_capture_group: bool,
    },
}

/// Describes one grouped/aggregated query for [`DuckDbReader::aggregate_objects`].
pub struct AggregateSpec {
    pub key_mode: GroupKeyMode,
    /// Fans each object out into one bucket per class it carries (mirrors
    /// `results_loader.rs`'s `object_classes`/`GroupConfig::group_by_class`).
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

/// One grouped/aggregated result row from [`DuckDbReader::aggregate_objects`].
pub struct AggregatedRow {
    pub group_key: String,
    /// `Some(_)` iff `AggregateSpec::group_by_class` was set.
    pub group_class: Option<String>,
    /// `Some(_)` iff `AggregateSpec::split_colocalized` was set.
    pub colocalized: Option<bool>,
    pub count: i64,
    /// One entry per `(metric_id, agg_fn)` pair in
    /// `AggregateSpec::metric_ids` × `AggregateSpec::agg_fns` order
    /// (metric-major). `None` means every contributing object was NULL for that
    /// metric (e.g. a channel column no object in the bucket has intensity data
    /// for) — SQL aggregate functions skip NULL inputs the same way
    /// `results_loader.rs`'s `apply_agg` only ever collects `Some` values, so
    /// an all-NULL bucket and an all-absent bucket mean the same thing.
    pub metric_values: Vec<Option<f64>>,
}

/// SQL expression computing the same display class string used throughout the
/// results table/filters/export: the joined `object_class_name` list, falling
/// back to `seg_class_name` when a object carries no object classes.
fn class_case_expr() -> &'static str {
    "CASE WHEN json_array_length(object_class_name) = 0 \
     THEN COALESCE(seg_class_name, '') \
     ELSE array_to_string(CAST(object_class_name AS VARCHAR[]), ', ') END"
}

/// SQL expression producing the per-object list of classes to fan out over for
/// `AggregateSpec::group_by_class` — mirrors `class_case_expr`'s fallback to
/// `seg_class_name`, but as a one-or-more-element array (via `UNNEST` in the
/// `FROM` clause) instead of a single comma-joined string, so a multi-class
/// object contributes to each of its classes' group buckets separately (matching
/// `results_loader.rs`'s `object_classes`). Always yields at least one element,
/// even when a object carries no object classes, so the `UNNEST` cross join
/// never silently drops a row.
fn class_fanout_expr() -> &'static str {
    "CASE WHEN json_array_length(object_class_name) = 0 \
     THEN [COALESCE(seg_class_name, '')] \
     ELSE CAST(object_class_name AS VARCHAR[]) END"
}

/// SQL expression for one of `ObjectRow`'s scaled per-channel intensity stats
/// (`min_scaled` / `max_scaled` / `mean_scaled` / `sum_scaled`, see `intensities_to_json`).
///
/// Uses the scalar-path form of `json_extract` (a single string path), not
/// the bracket/list form (`json_extract(col, ['{ch}'])`) — the list form
/// returns a JSON array of results (one per requested path) rather than a
/// scalar, and extracting a field from that array via `->>` throws a
/// "Malformed JSON" error the moment any row's `intensities_json` is missing
/// the requested channel key (e.g. a object with no measured intensities at all,
/// whose `intensities_json` is `"{}"`) — a real crash on sort-by-channel-column
/// today for any file where even one object lacks that channel's data. The
/// scalar form instead yields SQL NULL for a missing key, matching how
/// `parse_intensities`/`metric_value` already treat "channel absent" in Rust.
fn channel_stat_expr(ch: i32, stat: &str) -> String {
    format!("CAST(json_extract(intensities_json, '{ch}') ->> '{stat}' AS DOUBLE)")
}

/// SQL expression for the number of `class`-colocalizing partners a object has —
/// the same value the `coloc_partner__{class}__count` results-table column shows.
///
/// `coloc_json` is stored as a native DuckDB `JSON` column, and the `->`
/// extraction operator always receives a value the column type already
/// guarantees is well-formed JSON — no guard needed here (unlike comparing
/// `coloc_json` to a string literal, which is the actual crash risk; see
/// `coloc_filter_condition`/`get_coloc_partner_class_names`).
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
        "object_id" => Some("object_id".to_string()),
        "image" => Some("image_name".to_string()),
        "class" => Some(class_case_expr().to_string()),
        "area_px" => Some("area_px".to_string()),
        "area_nm2" => Some("area_nm2".to_string()),
        "circularity" => Some("circularity".to_string()),
        "colocalized" => Some(coloc_is_colocalized_expr().to_string()),
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
                "sum_bit" => "sum_scaled",
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
        "SELECT MIN({column}), MAX({column}), COUNT(DISTINCT {column}) FROM objects WHERE {column} IS NOT NULL"
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
// `ObjectFilter::coloc_filter` items are one of three kinds of label, matched
// literally against the value coming back from the GUI's filter popup:
//   - "No"                          -> object has no colocalization partners
//   - "Yes (any class)"             -> object colocalizes with at least one class
//   - "Colocalizes with <class>"    -> object colocalizes with that specific class
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

/// SQL boolean expression for "this object has at least one recorded
/// colocalization partner".
///
/// `coloc_json` is a native DuckDB `JSON` column. Comparing it directly to a
/// string literal (`coloc_json != ''`) makes DuckDB implicitly cast the
/// *literal* to JSON to perform the comparison — and casting `''` to JSON is
/// itself a parse error ("Malformed JSON ... input length is 0"), thrown for
/// *every row scanned* regardless of that row's own value. This crashed
/// loading results on Windows (observed there; apparently tolerated by
/// whatever DuckDB build/version this project used to run on Linux) even
/// though every stored `coloc_json` value is always well-formed JSON, since
/// the column type guarantees that at write time. Casting the column to
/// VARCHAR first forces the safe (JSON → string) cast direction instead.
fn coloc_is_colocalized_expr() -> &'static str {
    "(coloc_json IS NOT NULL AND CAST(coloc_json AS VARCHAR) != '' AND CAST(coloc_json AS VARCHAR) != '{}')"
}

/// Builds the SQL boolean expression for a single selected coloc-filter label.
fn coloc_filter_condition(label: &str) -> String {
    if label == coloc_filter_label_no() {
        "(coloc_json IS NULL OR CAST(coloc_json AS VARCHAR) = '' OR CAST(coloc_json AS VARCHAR) = '{}')".to_string()
    } else if label == coloc_filter_label_any() {
        coloc_is_colocalized_expr().to_string()
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
/// by [`DuckDbReader::get_objects`] and [`DuckDbReader::aggregate_objects`] so both
/// short-circuit identically instead of hitting the database for a query that
/// can never return anything.
fn filter_has_empty_selection(filter: &ObjectFilter) -> bool {
    filter.image_filter.as_deref().is_some_and(|v| v.is_empty())
        || filter.class_filter.as_deref().is_some_and(|v| v.is_empty())
        || filter.coloc_filter.as_deref().is_some_and(|v| v.is_empty())
        || filter
            .object_id_filter
            .as_deref()
            .is_some_and(|v| v.is_empty())
}

/// Builds the list of SQL boolean conditions `filter` implies (unjoined).
/// Shared by [`DuckDbReader::get_objects`] and [`DuckDbReader::aggregate_objects`]
/// so the two queries can never drift on what a given filter means;
/// `aggregate_objects` also appends its own `regexp_matches` condition on top of
/// this list before joining, for `GroupKeyMode::Regex`.
fn filter_conditions(filter: &ObjectFilter) -> Vec<String> {
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
        // ID list (see coloc_objects.rs), so `coloc_json` is exactly "{}" / empty iff the object
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
fn build_where_clause(filter: &ObjectFilter) -> String {
    where_clause_from(&filter_conditions(filter))
}

/// Adds the `images.disabled` column to a `.evadb` written before that
/// column existed. A no-op (DuckDB `ADD COLUMN IF NOT EXISTS` is a cheap
/// metadata-only check, not a data rewrite) once the column is already
/// there, which is always true for a file created from the current
/// `CREATE_TABLES` DDL - so this only ever does real work once per
/// pre-existing file, the first time it's touched by this feature.
///
/// No `NOT NULL` here (unlike `CREATE_TABLES`'s copy of this column) -
/// DuckDB's `ALTER TABLE ADD COLUMN` doesn't support column constraints,
/// only `DEFAULT` (confirmed against the pinned duckdb crate version:
/// `NOT NULL` here fails with "Adding columns with constraints not yet
/// supported"). The `DEFAULT` alone is enough in practice: it backfills
/// every existing row and applies to any future insert that omits the
/// column, so nothing in this codebase ever produces a real `NULL` here -
/// `get_images()` reading it straight into `bool` (not `Option<bool>`)
/// relies on that, not on the database enforcing it.
fn ensure_disabled_column(conn: &Connection) -> Result<(), InternalErrors> {
    conn.execute_batch(
        "ALTER TABLE images ADD COLUMN IF NOT EXISTS disabled BOOLEAN DEFAULT false;",
    )
    .map_err(|e| InternalErrors::Io(e.to_string()))
}

/// Migrates a `.evadb` from the old `status VARCHAR` ("ok"/"error") column
/// to the current `successful BOOLEAN` one: adds `successful` (a no-op if
/// already there, same reasoning as `ensure_disabled_column`), and - only
/// the first time, only if the legacy `status` column is still present -
/// backfills it from `status` and drops `status`. Checking for `status` via
/// `information_schema.columns` rather than just trying the backfill and
/// swallowing a "column does not exist" error keeps this from depending on
/// DuckDB's exact error message wording.
fn ensure_successful_column(conn: &Connection) -> Result<(), InternalErrors> {
    let err = |e: duckdb::Error| InternalErrors::Io(e.to_string());
    conn.execute_batch(
        "ALTER TABLE images ADD COLUMN IF NOT EXISTS successful BOOLEAN DEFAULT true;",
    )
    .map_err(err)?;

    let has_legacy_status: bool = conn
        .query_row(
            "SELECT count(*) > 0 FROM information_schema.columns \
             WHERE table_name = 'images' AND column_name = 'status'",
            [],
            |row| row.get(0),
        )
        .map_err(err)?;
    if has_legacy_status {
        conn.execute_batch(
            "UPDATE images SET successful = (status = 'ok'); \
             ALTER TABLE images DROP COLUMN status;",
        )
        .map_err(err)?;
    }
    Ok(())
}
