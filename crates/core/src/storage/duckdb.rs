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
            for (class, name) in &class_names {
                if let ObjectClass::Valid(n) = class {
                    app.append_row(params![*n, name])
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

-- One row per processed image, written once all of its tiles/planes are
-- done (see DuckDbExporter::finalize_image) — independent of how many
-- objects (if any) it produced, so an image with zero detections still
-- shows up in the file instead of leaving no trace at all. Deliberately
-- just identity, no object/coloc counts: any real count only means
-- something scoped to a class or a coloc partner (\"how many ClassA
-- objects\", \"how many colocalize with ClassB\"), which this table can't
-- express and callers should instead derive live from `objects`.
--
-- `successful`/`error_message`: an image is always finalized here even when
-- a tile or plane failed to export (so it still shows up rather than
-- vanishing), but that means `successful` is the only way to tell a
-- genuinely complete image from one that has silently incomplete/missing
-- object data. Was `status VARCHAR` ('ok'/'error') until this rename - only
-- ever those two values in practice, driven straight off `Option<&str>` in
-- `finalize_image`, so a bool is both smaller and matches how it's actually
-- used; a `.evadb` written before the rename gets migrated on first
-- read/write (see `ensure_successful_column`).
--
-- `disabled`: user-driven, set well after analysis (see
-- `DuckDbReader::set_image_disabled`), not written by the pipeline itself -
-- distinct from `successful`, which the pipeline sets once and never
-- revisits. A `.evadb` written before this column existed gets it added
-- lazily on first read/write (see `ensure_disabled_column`), so this
-- default only matters for files created from this DDL directly.
CREATE TABLE IF NOT EXISTS images (
    image_name      VARCHAR NOT NULL,
    image_rel_path  VARCHAR NOT NULL PRIMARY KEY,
    successful      BOOLEAN NOT NULL DEFAULT true,
    error_message   VARCHAR,
    disabled        BOOLEAN NOT NULL DEFAULT false
);

-- Snapshot of the project's object-classification registry at the moment
-- this file was written (see DuckDbExporter::new) — one row per registered
-- class, regardless of whether any object was actually assigned it. The
-- authoritative source for the results view's class names/filter, instead
-- of the live project (which can change after the fact) or re-deriving
-- names from `objects.object_class_name`.
CREATE TABLE IF NOT EXISTS classes (
    class_id  INTEGER NOT NULL PRIMARY KEY,
    name      VARCHAR NOT NULL
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
            "INSERT INTO images (image_name, image_rel_path, successful, error_message) VALUES (?, ?, ?, ?)",
            params![image_name, image_rel, successful, error],
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

/// Opens a result file written by [`DuckDbExporter`] for reading.
pub struct DuckDbReader {
    conn: Connection,
}

fn object_select_sql(fetch_intensities: bool) -> String {
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
         FROM objects"
    )
}

/// Builds the shared `WHERE`/`ORDER BY` clauses for [`DuckDbReader::get_objects`]
/// and [`DuckDbReader::stream_objects`]. `object_id` is always the final
/// tiebreaker so results stay stably ordered even when many rows share the
/// same sort-column value (load-bearing for `get_objects`'s page stability, and
/// for `stream_objects`'s callers being able to assert on row order).
fn where_and_order(filter: &ObjectFilter) -> (String, String) {
    let where_clause = build_where_clause(filter);
    let order_by = match filter.sort_column.as_deref().and_then(column_order_expr) {
        Some(expr) => format!(
            "ORDER BY {expr} {}, object_id",
            if filter.sort_ascending { "ASC" } else { "DESC" }
        ),
        None => "ORDER BY object_id".to_string(),
    };
    (where_clause, order_by)
}

impl DuckDbReader {
    pub fn open(path: &Path) -> Result<Self, InternalErrors> {
        let conn = shared_connection(path)?;
        Ok(Self { conn })
    }

    /// Returns ROIs matching `filter`, with optional pagination.
    pub fn get_objects(&self, filter: &ObjectFilter) -> Result<Vec<ObjectRow>, InternalErrors> {
        let err = |e: duckdb::Error| InternalErrors::Io(e.to_string());

        if filter_has_empty_selection(filter) {
            return Ok(vec![]);
        }

        let (where_clause, order_by) = where_and_order(filter);

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
            "{} {} {order_by} {}",
            object_select_sql(filter.fetch_intensities),
            where_clause,
            pagination
        );
        let mut stmt = self.conn.prepare(&sql).map_err(err)?;
        stmt.query_map([], map_object_row)
            .map_err(err)?
            .collect::<Result<Vec<_>, _>>()
            .map_err(err)
    }

    /// Streams every object matching `filter` to `on_row`, one at a time, in
    /// `filter`'s sort order, via a single sorted scan over the whole
    /// matching set. `filter.page_size`/`filter.page` are ignored.
    ///
    /// Unlike [`Self::get_objects`], which applies `LIMIT`/`OFFSET` and is the
    /// right tool for the interactive table's random-access paging (jumping
    /// straight to an arbitrary page), this is for callers - like a full
    /// export - that always walk the entire matching set start to finish.
    /// Repeatedly calling `get_objects` with an increasing `OFFSET` to do that
    /// re-sorts and re-scans everything already returned on every single
    /// call, so total cost grows quadratically with the number of matching
    /// rows; this does the sort/scan exactly once.
    ///
    /// Can be called again (e.g. for a colocalization-partner lookup) on the
    /// same `&self` connection from inside `on_row` — `DuckDbReader`'s
    /// methods only ever hold the connection open for the duration of one
    /// call, so nesting a second, fully-executed query inside the first
    /// one's row callback is safe.
    pub fn stream_objects(
        &self,
        filter: &ObjectFilter,
        mut on_row: impl FnMut(ObjectRow) -> Result<(), InternalErrors>,
    ) -> Result<(), InternalErrors> {
        let err = |e: duckdb::Error| InternalErrors::Io(e.to_string());

        if filter_has_empty_selection(filter) {
            return Ok(());
        }

        let (where_clause, order_by) = where_and_order(filter);
        let sql = format!(
            "{} {} {order_by}",
            object_select_sql(filter.fetch_intensities),
            where_clause
        );

        let mut stmt = self.conn.prepare(&sql).map_err(err)?;
        let rows = stmt.query_map([], map_object_row).map_err(err)?;
        for row in rows {
            on_row(row.map_err(err)?)?;
        }
        Ok(())
    }

    /// Returns distinct `(image_rel_path, image_name)` pairs matching `filter`'s
    /// image/class/coloc/object_id/t/z restrictions (`page_size`/`page`/
    /// `sort_column` are ignored — this is never paginated). Cheap: proportional
    /// to the number of distinct images, not the number of ROIs. Used to
    /// precompute a `GroupKeyMode::ImageRelPathMap` for [`Self::aggregate_objects`]
    /// (`GroupBy::Folder`'s directory-splitting logic stays in Rust — see
    /// `folder_of` in `results_loader.rs` — since `std::path::Path::parent()`'s
    /// semantics have no faithful single SQL expression).
    pub fn get_distinct_images(
        &self,
        filter: &ObjectFilter,
    ) -> Result<Vec<(String, String)>, InternalErrors> {
        let err = |e: duckdb::Error| InternalErrors::Io(e.to_string());

        if filter_has_empty_selection(filter) {
            return Ok(vec![]);
        }

        let where_clause = build_where_clause(filter);
        let sql = format!(
            "SELECT DISTINCT image_rel_path, image_name FROM objects {where_clause} \
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
    pub fn aggregate_objects(
        &self,
        filter: &ObjectFilter,
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
            GroupKeyMode::Regex {
                pattern,
                has_capture_group,
            } => {
                // Rows that don't match are dropped entirely (not grouped under
                // an empty/NULL key) — `regexp_extract` alone can't distinguish
                // "matched, captured empty string" from "no match", so that's
                // enforced by the separate `regexp_matches` condition below.
                conditions.push(format!("regexp_matches(image_name, '{}')", esc(pattern)));
                let group_idx = if *has_capture_group { 1 } else { 0 };
                format!(
                    "regexp_extract(image_name, '{}', {group_idx})",
                    esc(pattern)
                )
            }
        };

        // `FROM objects[, UNNEST(...) AS u(group_class)]` — UNNEST is only valid
        // here as a lateral cross join in the FROM clause, not in the SELECT
        // list combined with GROUP BY (DuckDB rejects that form outright).
        let from_clause = if spec.group_by_class {
            format!("objects, UNNEST({}) AS u(group_class)", class_fanout_expr())
        } else {
            "objects".to_string()
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
            "SELECT DISTINCT {} AS class_str FROM objects ORDER BY class_str",
            class_case_expr()
        );
        let mut stmt = self.conn.prepare(&sql).map_err(err)?;
        stmt.query_map([], |row| row.get(0))
            .map_err(err)?
            .collect::<Result<Vec<_>, _>>()
            .map_err(err)
    }

    /// Returns every registered class (from `classes`), sorted by id -
    /// including classes with zero matching objects, and independent of
    /// whatever the live project's classification settings look like now
    /// (see the DDL comment on `classes`). The authoritative source for the
    /// results view's class filter, instead of `get_class_names`'s
    /// DISTINCT-over-`objects` scan (which only reflects classes actually
    /// used, and only the names baked in at export time).
    pub fn get_classes(&self) -> Result<Vec<ClassRow>, InternalErrors> {
        let err = |e: duckdb::Error| InternalErrors::Io(e.to_string());
        let mut stmt = self
            .conn
            .prepare("SELECT class_id, name FROM classes ORDER BY class_id")
            .map_err(err)?;
        stmt.query_map([], |row| {
            Ok(ClassRow {
                class_id: row.get(0)?,
                name: row.get(1)?,
            })
        })
        .map_err(err)?
        .collect::<Result<Vec<_>, _>>()
        .map_err(err)
    }

    /// Returns the `(min, max)` of `t_stack` present in the file, or `None`
    /// when there's nothing to step through — either every object's `t_stack` is
    /// NULL (no time axis) or they all share the same single value.
    pub fn get_t_stack_range(&self) -> Result<Option<(i32, i32)>, InternalErrors> {
        stack_range(&self.conn, "t_stack")
    }

    /// Returns the `(min, max)` of `z_stack` present in the file, or `None`
    /// when there's nothing to step through — either every object's `z_stack` is
    /// NULL (no depth axis) or they all share the same single value.
    pub fn get_z_stack_range(&self) -> Result<Option<(i32, i32)>, InternalErrors> {
        stack_range(&self.conn, "z_stack")
    }

    /// Returns all distinct `image_name` values present in the file, sorted
    /// alphabetically — sourced from `images`, not `objects`, so an image
    /// that produced zero objects still appears (e.g. in the Table view's
    /// image filter popup).
    pub fn get_image_names(&self) -> Result<Vec<String>, InternalErrors> {
        let err = |e: duckdb::Error| InternalErrors::Io(e.to_string());
        let mut stmt = self
            .conn
            .prepare("SELECT DISTINCT image_name FROM images ORDER BY image_name")
            .map_err(err)?;
        stmt.query_map([], |row| row.get(0))
            .map_err(err)?
            .collect::<Result<Vec<_>, _>>()
            .map_err(err)
    }

    /// Returns one row per processed image (from `images`), sorted by
    /// relative path — including images that produced zero objects. Used by
    /// the plate/well Matrix view to render every well/field-of-view slot as
    /// occupied even when it has nothing to aggregate.
    pub fn get_images(&self) -> Result<Vec<ImageRow>, InternalErrors> {
        let err = |e: duckdb::Error| InternalErrors::Io(e.to_string());
        ensure_disabled_column(&self.conn)?;
        ensure_successful_column(&self.conn)?;
        let mut stmt = self
            .conn
            .prepare(
                "SELECT image_name, image_rel_path, successful, error_message, disabled FROM images ORDER BY image_rel_path",
            )
            .map_err(err)?;
        stmt.query_map([], |row| {
            Ok(ImageRow {
                image_name: row.get(0)?,
                image_rel_path: row.get(1)?,
                successful: row.get(2)?,
                error_message: row.get(3)?,
                disabled: row.get(4)?,
            })
        })
        .map_err(err)?
        .collect::<Result<Vec<_>, _>>()
        .map_err(err)
    }

    /// Marks `image_rel_path` disabled/enabled - excluded from exports by
    /// default and shown crossed out in the Matrix view (see
    /// `ImageRow::disabled`).
    ///
    /// This is the one write path against an already-finalized results
    /// file: every other `DuckDbReader` method only reads. `.evadb` files
    /// are otherwise append-only, written once by the analysis pipeline and
    /// never touched again, so this - unlike every read method here - can
    /// race a concurrent reader/writer holding its own connection to the
    /// same file (DuckDB is single-writer per file). Callers should treat a
    /// failure here as "the toggle didn't stick, tell the user and let them
    /// retry" rather than a hard error.
    /// Matches on `image_name` (the bare filename), not `image_rel_path`
    /// (the real primary key) - consistent with every other image-identity
    /// lookup in this codebase (e.g. `filter_conditions`' own `image_name IN
    /// (...)` for `ObjectFilter.image_filter`), which already accepts that
    /// two images sharing a bare filename in different folders can't be
    /// told apart this way. Fixing that would mean plumbing
    /// `image_rel_path` through the image filter checklist too, not just
    /// this one call - out of scope for now.
    pub fn set_image_disabled(
        &self,
        image_name: &str,
        disabled: bool,
    ) -> Result<(), InternalErrors> {
        ensure_disabled_column(&self.conn)?;
        self.conn
            .execute(
                "UPDATE images SET disabled = ? WHERE image_name = ?",
                params![disabled, image_name],
            )
            .map_err(|e| InternalErrors::Io(e.to_string()))?;
        Ok(())
    }

    /// Returns all distinct partner-class labels appearing as keys in any object's
    /// `coloc_json`, sorted alphabetically. Used to populate the "Colocalizes
    /// with <class>" entries of the coloc filter popup.
    pub fn get_coloc_partner_class_names(&self) -> Result<Vec<String>, InternalErrors> {
        let err = |e: duckdb::Error| InternalErrors::Io(e.to_string());
        // `coloc_json` is a native JSON column, always well-formed by
        // construction — `json_keys(coloc_json)` itself needs no guard. The
        // WHERE clause below casts to VARCHAR before comparing against the
        // `''`/`'{}'` literals (see `coloc_is_colocalized_expr`) — comparing
        // the JSON column to those literals directly crashes on Windows
        // ("Malformed JSON ... input length is 0"), since DuckDB casts the
        // *literal* to JSON to perform the comparison rather than the other
        // way around.
        let sql = format!(
            "SELECT DISTINCT UNNEST(json_keys(coloc_json)) AS k FROM objects WHERE {} ORDER BY k",
            coloc_is_colocalized_expr()
        );
        let mut stmt = self.conn.prepare(&sql).map_err(err)?;
        stmt.query_map([], |row| row.get(0))
            .map_err(err)?
            .collect::<Result<Vec<_>, _>>()
            .map_err(err)
    }
}

fn map_object_row(row: &duckdb::Row<'_>) -> duckdb::Result<ObjectRow> {
    Ok(ObjectRow {
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
    use crate::object::{Object, ObjectInit};
    use bitvec::prelude::*;
    use evanalyzer_cfg::core_types::{ObjectClass, ObjectId};
    use std::collections::HashSet;
    use tempfile::TempDir;

    fn make_filled_object(id: u128, bbox: [u32; 4]) -> Object {
        let [x_min, y_min, x_max, y_max] = bbox;
        let w = (x_max - x_min + 1) as usize;
        let h = (y_max - y_min + 1) as usize;
        let area = w * h;
        let mask_data = BitVec::<u64, Lsb0>::repeat(true, area);
        Object::new(ObjectInit {
            id: ObjectId(id),
            bbox,
            mask_data,
            area,
            ..Default::default()
        })
    }

    fn make_filled_object_at_plane(id: u128, bbox: [u32; 4], t: i32, z: i32) -> Object {
        let mut object = make_filled_object(id, bbox);
        object.plane.t = t;
        object.plane.z = z;
        object
    }

    fn export_and_open(objects: Vec<Object>) -> (TempDir, DuckDbReader) {
        let dir = TempDir::new().expect("failed to create temp dir");
        let path = dir.path().join("results.duckdb");

        let mut cache = GlobalPipelineCache::default();
        for object in objects {
            cache.object_cache.insert(object.id.clone(), object);
        }

        let exporter = DuckDbExporter::new(&path, HashMap::new()).expect("exporter init failed");
        exporter.export(&cache).expect("export failed");
        drop(exporter); // flushes the appenders / closes the connection

        let reader = DuckDbReader::open(&path).expect("reader open failed");
        (dir, reader)
    }

    /// Like `export_and_open`, but each `(image_rel_path, objects)` pair is
    /// exported as its own image (mirroring real usage — `PipelineCache`
    /// carries one `image_rel_path` per `export()` call), so the fixture can
    /// span more than one distinct image. Needed for `aggregate_objects` tests
    /// (`GroupBy::Image`/`Folder`/`Regex` are meaningless with only one image).
    fn export_multi_image_and_open(images: Vec<(&str, Vec<Object>)>) -> (TempDir, DuckDbReader) {
        let dir = TempDir::new().expect("failed to create temp dir");
        let path = dir.path().join("results.duckdb");

        let exporter = DuckDbExporter::new(&path, HashMap::new()).expect("exporter init failed");
        for (image_rel_path, objects) in images {
            let mut cache = GlobalPipelineCache {
                image_rel_path: image_rel_path.into(),
                ..Default::default()
            };
            for object in objects {
                cache.object_cache.insert(object.id.clone(), object);
            }
            exporter.export(&cache).expect("export failed");
        }
        drop(exporter);

        let reader = DuckDbReader::open(&path).expect("reader open failed");
        (dir, reader)
    }

    #[test]
    fn finalize_image_records_an_image_that_produced_zero_objects() {
        let dir = TempDir::new().expect("failed to create temp dir");
        let path = dir.path().join("results.duckdb");

        let exporter = DuckDbExporter::new(&path, HashMap::new()).expect("exporter init failed");
        // export() is still called for the image's tile(s), but with no objects.
        let cache = GlobalPipelineCache {
            image_rel_path: "empty.tif".into(),
            ..Default::default()
        };
        exporter.export(&cache).expect("export failed");
        exporter
            .finalize_image(Path::new("empty.tif"), None)
            .expect("finalize_image failed");
        drop(exporter);

        let reader = DuckDbReader::open(&path).expect("reader open failed");
        let images = reader.get_images().expect("get_images failed");
        assert_eq!(images.len(), 1);
        assert_eq!(images[0].image_name, "empty.tif");
        assert_eq!(images[0].image_rel_path, "empty.tif");
        assert!(images[0].successful);
        assert_eq!(images[0].error_message, None);

        // The Table view's image filter must also see it, not just get_images().
        assert_eq!(
            reader.get_image_names().expect("get_image_names failed"),
            vec!["empty.tif".to_string()]
        );

        // No objects were ever written for it.
        assert!(
            reader
                .get_objects(&ObjectFilter::default())
                .expect("get_objects failed")
                .is_empty()
        );
    }

    #[test]
    fn reader_opens_successfully_while_the_writer_connection_is_still_alive() {
        // Regression test for a real production bug: on Windows, opening a
        // *second*, independent `Connection` to a `.evadb` file some other
        // still-open `Connection` in this same process already has open
        // fails with a sharing-violation IO error ("File is already open in
        // <this exact exe/PID>") - not a lock from another program, a
        // self-conflict. Before `shared_connection` existed, every call site
        // (`DuckDbExporter::new`, `DuckDbReader::open`) did its own
        // independent `Connection::open`, so a results panel opening a
        // reader while an analyze job's exporter (or another panel's
        // reader) was still open on the same file would fail outright -
        // this is exactly what happened in production: results panels
        // logged "failed to load ROIs"/"failed to load image names" for
        // hours after a run finished, because *something* still held a
        // connection open.
        //
        // This can't reproduce the Windows-specific IO error on Linux CI
        // (verified empirically: two independent raw `Connection::open`
        // calls to the same path both succeed on Linux) - what this test
        // verifies instead is that the fix's actual mechanism works: a
        // `DuckDbReader` opened while a `DuckDbExporter` is still alive
        // sees the exporter's writes immediately, proving it shares the
        // exporter's live connection (via `Connection::try_clone`, which
        // DuckDB's MVCC coordinates safely) rather than racing it for an
        // independent OS-level file handle.
        let dir = TempDir::new().expect("failed to create temp dir");
        let path = dir.path().join("results.duckdb");

        let exporter = DuckDbExporter::new(&path, HashMap::new()).expect("exporter init failed");
        let cache = GlobalPipelineCache {
            image_rel_path: "still_open.tif".into(),
            ..Default::default()
        };
        exporter.export(&cache).expect("export failed");
        exporter
            .finalize_image(Path::new("still_open.tif"), None)
            .expect("finalize_image failed");

        // `exporter` is deliberately NOT dropped here - this is the crux of
        // the regression: the reader must open successfully, and see the
        // just-written row, with the writer's connection still fully alive.
        let reader = DuckDbReader::open(&path).expect(
            "opening a reader while the writer connection is still open must succeed - it \
             failing here is exactly the production bug this test guards against",
        );
        let images = reader.get_images().expect("get_images failed");
        assert_eq!(images.len(), 1);
        assert_eq!(images[0].image_name, "still_open.tif");

        drop(exporter);
    }

    #[test]
    fn shared_connection_reuses_one_cached_anchor_per_path_instead_of_opening_independently() {
        // The functional test above (`reader_opens_successfully_while_the_
        // writer_connection_is_still_alive`) can't actually fail on Linux
        // even without the fix - two independent `Connection::open` calls
        // to the same file both succeed here, and DuckDB's on-disk state
        // stays consistent across them regardless (verified empirically).
        // The real, Windows-specific regression is unreproducible in this
        // CI environment. This test instead directly verifies the fix's
        // *mechanism* - that repeated `shared_connection` calls for one
        // path reuse a single cached anchor rather than opening
        // independently - so reverting `DuckDbReader::open`/
        // `DuckDbExporter::new` back to a raw `Connection::open` (silently
        // reintroducing the bug) fails this test even on Linux, where the
        // symptom itself wouldn't otherwise show up.
        // `CONNECTION_CACHE` is a process-wide global every test in this
        // module shares (and tests run in parallel by default), so this
        // only ever checks for its own two paths by key - never the
        // cache's overall size or a before/after delta, since other
        // concurrently-running tests are free to insert their own
        // (differently-pathed - `TempDir` is unique per test) entries at
        // the same time, which would make a size-based assertion flaky.
        let contains = |p: &Path| {
            CONNECTION_CACHE
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .contains_key(p)
        };

        let dir = TempDir::new().expect("failed to create temp dir");
        let path_a = dir.path().join("a.duckdb");
        let path_b = dir.path().join("b.duckdb");
        assert!(
            !contains(&path_a) && !contains(&path_b),
            "freshly-generated tempdir paths must not already be cached"
        );

        let _conn1 = shared_connection(&path_a).expect("first open failed");
        assert!(
            contains(&path_a),
            "opening path_a must cache an anchor for it"
        );
        assert!(
            !contains(&path_b),
            "must not cache anything for an unrelated path"
        );

        // A second call for the SAME path must succeed by reusing the
        // cached anchor via `try_clone` - a `HashMap` can only ever hold
        // one entry per key regardless, so this call succeeding without
        // error (rather than failing the way independent
        // `Connection::open` calls to the same file can on Windows) is
        // what actually matters here, not the cache's size.
        let _conn2 = shared_connection(&path_a)
            .expect("second open of the same (still cached) path must succeed");

        // A DIFFERENT path must get its own, separate anchor.
        let _conn3 = shared_connection(&path_b).expect("open of a different path failed");
        assert!(
            contains(&path_b),
            "opening path_b must cache an anchor for it"
        );
    }

    #[test]
    fn finalize_image_with_an_error_is_still_recorded_but_marked_failed() {
        let dir = TempDir::new().expect("failed to create temp dir");
        let path = dir.path().join("results.duckdb");

        let exporter = DuckDbExporter::new(&path, HashMap::new()).expect("exporter init failed");
        // A partially-failed image must still show up (see the finalize_image
        // doc comment) - but as unsuccessful, not indistinguishable from success.
        exporter
            .finalize_image(Path::new("broken.tif"), Some("tile 3: decode failed"))
            .expect("finalize_image failed");
        drop(exporter);

        let reader = DuckDbReader::open(&path).expect("reader open failed");
        let images = reader.get_images().expect("get_images failed");
        assert_eq!(images.len(), 1, "the failed image must still be recorded");
        assert!(!images[0].successful);
        assert_eq!(
            images[0].error_message.as_deref(),
            Some("tile 3: decode failed")
        );
    }

    #[test]
    fn export_rolls_back_object_rows_when_the_coloc_stats_write_fails() {
        // Regression test for wrapping objects + coloc_stats in one
        // transaction: force the second write (coloc_stats) to fail after
        // the first (objects) would already have appended a row, and verify
        // the object row doesn't end up partially committed on its own.
        let dir = TempDir::new().expect("failed to create temp dir");
        let path = dir.path().join("results.duckdb");
        let exporter = DuckDbExporter::new(&path, HashMap::new()).expect("exporter init failed");

        {
            let conn = exporter.conn.lock().expect("connection mutex poisoned");
            conn.execute_batch("DROP TABLE coloc_stats;")
                .expect("failed to drop coloc_stats for the test");
        }

        let mut cache = GlobalPipelineCache {
            image_rel_path: "broken.tif".into(),
            ..Default::default()
        };
        let object = make_filled_object(1, [0, 0, 9, 9]);
        cache.object_cache.insert(object.id.clone(), object);

        let result = exporter.export(&cache);
        assert!(
            result.is_err(),
            "expected export to fail once coloc_stats is missing"
        );
        drop(exporter);

        let reader = DuckDbReader::open(&path).expect("reader open failed");
        let objects = reader
            .get_objects(&ObjectFilter::default())
            .expect("get_objects failed");
        assert!(
            objects.is_empty(),
            "the object row must have been rolled back with the failed coloc_stats write, \
             not partially committed on its own"
        );
    }

    #[test]
    fn get_images_lists_every_finalized_image_regardless_of_object_count() {
        let dir = TempDir::new().expect("failed to create temp dir");
        let path = dir.path().join("results.duckdb");

        let exporter = DuckDbExporter::new(&path, HashMap::new()).expect("exporter init failed");

        let mut with_objects = GlobalPipelineCache {
            image_rel_path: "has_objects.tif".into(),
            ..Default::default()
        };
        let object = make_filled_object(1, [0, 0, 9, 9]);
        with_objects.object_cache.insert(object.id.clone(), object);
        exporter.export(&with_objects).expect("export failed");
        exporter
            .finalize_image(Path::new("has_objects.tif"), None)
            .expect("finalize_image failed");

        let empty = GlobalPipelineCache {
            image_rel_path: "empty.tif".into(),
            ..Default::default()
        };
        exporter.export(&empty).expect("export failed");
        exporter
            .finalize_image(Path::new("empty.tif"), None)
            .expect("finalize_image failed");

        drop(exporter);

        let reader = DuckDbReader::open(&path).expect("reader open failed");
        let mut names: Vec<String> = reader
            .get_images()
            .expect("get_images failed")
            .into_iter()
            .map(|i| i.image_rel_path)
            .collect();
        names.sort();
        assert_eq!(
            names,
            vec!["empty.tif".to_string(), "has_objects.tif".to_string()]
        );
    }

    #[test]
    fn set_image_disabled_persists_and_round_trips_through_get_images() {
        let dir = TempDir::new().expect("failed to create temp dir");
        let path = dir.path().join("results.duckdb");

        let exporter = DuckDbExporter::new(&path, HashMap::new()).expect("exporter init failed");
        exporter
            .finalize_image(Path::new("a.tif"), None)
            .expect("finalize_image failed");
        exporter
            .finalize_image(Path::new("b.tif"), None)
            .expect("finalize_image failed");
        drop(exporter);

        let reader = DuckDbReader::open(&path).expect("reader open failed");
        let images = reader.get_images().expect("get_images failed");
        assert!(
            images.iter().all(|i| !i.disabled),
            "a freshly finalized image must default to not disabled"
        );

        reader
            .set_image_disabled("a.tif", true)
            .expect("set_image_disabled failed");

        // A fresh reader/connection - not the same one that wrote the
        // toggle - to actually prove the write reached disk.
        let reader = DuckDbReader::open(&path).expect("reader re-open failed");
        let images = reader.get_images().expect("get_images failed");
        let a = images.iter().find(|i| i.image_name == "a.tif").unwrap();
        let b = images.iter().find(|i| i.image_name == "b.tif").unwrap();
        assert!(a.disabled, "a.tif must be disabled");
        assert!(!b.disabled, "b.tif must be untouched");

        // Toggling back off must also stick.
        reader
            .set_image_disabled("a.tif", false)
            .expect("set_image_disabled failed");
        let images = DuckDbReader::open(&path)
            .expect("reader re-open failed")
            .get_images()
            .expect("get_images failed");
        assert!(
            images.iter().all(|i| !i.disabled),
            "re-enabling must also persist"
        );
    }

    #[test]
    fn set_image_disabled_migrates_a_file_written_before_the_column_existed() {
        // Hand-builds the *old* `images` schema (no `disabled` column) to
        // stand in for a `.evadb` written before this feature existed -
        // `DuckDbExporter::new` always creates the current schema, so it
        // can't produce this fixture itself.
        let dir = TempDir::new().expect("failed to create temp dir");
        let path = dir.path().join("legacy.duckdb");
        {
            let conn = Connection::open(&path).expect("open failed");
            conn.execute_batch(
                "CREATE TABLE images (
                    image_name VARCHAR NOT NULL, image_rel_path VARCHAR NOT NULL PRIMARY KEY,
                    status VARCHAR NOT NULL DEFAULT 'ok', error_message VARCHAR
                );
                INSERT INTO images (image_name, image_rel_path, status, error_message)
                VALUES ('legacy.tif', 'legacy.tif', 'ok', NULL);",
            )
            .expect("legacy schema setup failed");
        }

        // Reading an un-migrated file must still work, defaulting to false -
        // and this same legacy fixture (`status = 'ok'`) also exercises the
        // `status` -> `successful` migration (see the dedicated test below
        // for the full backfill/column-drop behavior).
        let reader = DuckDbReader::open(&path).expect("reader open failed");
        let images = reader.get_images().expect("get_images failed");
        assert_eq!(images.len(), 1);
        assert!(!images[0].disabled);
        assert!(images[0].successful);

        // And writing to one must add the column on the fly rather than
        // erroring on "no such column".
        reader
            .set_image_disabled("legacy.tif", true)
            .expect("set_image_disabled must migrate the legacy file, not fail");
        let images = DuckDbReader::open(&path)
            .expect("reader re-open failed")
            .get_images()
            .expect("get_images failed");
        assert!(images[0].disabled);
    }

    #[test]
    fn get_images_migrates_the_legacy_status_column_to_successful() {
        // Hand-builds the *old* `images` schema (`status VARCHAR`, no
        // `successful`/`disabled` columns) - one row of each status, to
        // verify the backfill maps both values correctly, not just the
        // default.
        let dir = TempDir::new().expect("failed to create temp dir");
        let path = dir.path().join("legacy.duckdb");
        {
            let conn = Connection::open(&path).expect("open failed");
            conn.execute_batch(
                "CREATE TABLE images (
                    image_name VARCHAR NOT NULL, image_rel_path VARCHAR NOT NULL PRIMARY KEY,
                    status VARCHAR NOT NULL DEFAULT 'ok', error_message VARCHAR
                );
                INSERT INTO images (image_name, image_rel_path, status, error_message)
                VALUES ('ok.tif', 'ok.tif', 'ok', NULL);
                INSERT INTO images (image_name, image_rel_path, status, error_message)
                VALUES ('broken.tif', 'broken.tif', 'error', 'tile 3: decode failed');",
            )
            .expect("legacy schema setup failed");
        }

        let images = DuckDbReader::open(&path)
            .expect("reader open failed")
            .get_images()
            .expect("get_images failed");
        let ok = images.iter().find(|i| i.image_name == "ok.tif").unwrap();
        let broken = images
            .iter()
            .find(|i| i.image_name == "broken.tif")
            .unwrap();
        assert!(ok.successful, "'ok' must backfill to successful = true");
        assert!(
            !broken.successful,
            "'error' must backfill to successful = false"
        );
        assert_eq!(
            broken.error_message.as_deref(),
            Some("tile 3: decode failed")
        );

        // The legacy column must actually be dropped, not just shadowed -
        // otherwise every future open would re-run the (harmless but
        // pointless) backfill forever.
        let conn = Connection::open(&path).expect("re-open failed");
        let has_status: bool = conn
            .query_row(
                "SELECT count(*) > 0 FROM information_schema.columns \
                 WHERE table_name = 'images' AND column_name = 'status'",
                [],
                |row| row.get(0),
            )
            .expect("introspection query failed");
        assert!(!has_status, "the legacy `status` column must be dropped");
    }

    #[test]
    fn get_classes_returns_the_full_registry_snapshot_including_unused_classes() {
        let dir = TempDir::new().expect("failed to create temp dir");
        let path = dir.path().join("results.duckdb");

        let mut class_names = HashMap::new();
        class_names.insert(ObjectClass::Valid(1), "Positive".to_string());
        class_names.insert(ObjectClass::Valid(2), "Negative".to_string());
        // Class 5 is registered but never actually assigned to any object.
        class_names.insert(ObjectClass::Valid(5), "Unused".to_string());

        let exporter = DuckDbExporter::new(&path, class_names).expect("exporter init failed");
        let mut cache = GlobalPipelineCache {
            image_rel_path: "img.tif".into(),
            ..Default::default()
        };
        let mut object = make_filled_object(1, [0, 0, 9, 9]);
        object.object_class = HashSet::from([ObjectClass::Valid(1)]);
        cache.object_cache.insert(object.id.clone(), object);
        exporter.export(&cache).expect("export failed");
        drop(exporter);

        let reader = DuckDbReader::open(&path).expect("reader open failed");
        let mut classes = reader.get_classes().expect("get_classes failed");
        classes.sort_by_key(|c| c.class_id);
        let pairs: Vec<(i32, String)> = classes.into_iter().map(|c| (c.class_id, c.name)).collect();
        assert_eq!(
            pairs,
            vec![
                (1, "Positive".to_string()),
                (2, "Negative".to_string()),
                (5, "Unused".to_string()),
            ],
            "the registry must list every registered class, not just ones with objects"
        );
    }

    /// Guards a real regression: a caller that builds a class filter value
    /// from `get_classes()`'s bare `name` (rather than reproducing
    /// `class_label`'s `"{name} ({id})"` shape) silently matches nothing,
    /// since `class_case_expr()` - what `class_filter` is actually matched
    /// against - reads the full `"{name} ({id})"` string `export()` bakes
    /// into `object_class_name`. This exercises that exact round trip: the
    /// registry's `(id, name)` reformatted the same way `class_label` would,
    /// used as an `ObjectFilter::class_filter` value, must still find the
    /// object it names.
    #[test]
    fn class_filter_matches_when_built_from_get_classes_in_class_labels_own_format() {
        let dir = TempDir::new().expect("failed to create temp dir");
        let path = dir.path().join("results.duckdb");

        let mut class_names = HashMap::new();
        class_names.insert(ObjectClass::Valid(1), "Positive".to_string());

        let exporter = DuckDbExporter::new(&path, class_names).expect("exporter init failed");
        let mut cache = GlobalPipelineCache {
            image_rel_path: "img.tif".into(),
            ..Default::default()
        };
        let mut object = make_filled_object(1, [0, 0, 9, 9]);
        object.object_class = HashSet::from([ObjectClass::Valid(1)]);
        cache.object_cache.insert(object.id.clone(), object);
        exporter.export(&cache).expect("export failed");
        drop(exporter);

        let reader = DuckDbReader::open(&path).expect("reader open failed");
        let classes = reader.get_classes().expect("get_classes failed");
        let class = classes.first().expect("one registered class");
        let filter_value = format!("{} ({})", class.name, class.class_id);
        assert_eq!(filter_value, "Positive (1)");

        let filter = ObjectFilter {
            class_filter: Some(vec![filter_value]),
            ..Default::default()
        };
        let objects = reader.get_objects(&filter).expect("get_objects failed");
        assert_eq!(
            objects.len(),
            1,
            "a class filter built from get_classes() in class_label's own format must still match"
        );
    }

    #[test]
    fn aggregate_objects_groups_by_image_and_averages() {
        let a1 = make_filled_object(1, [0, 0, 9, 9]); // area_px = 100
        let a2 = make_filled_object(2, [0, 0, 19, 9]); // area_px = 200
        let b1 = make_filled_object(3, [0, 0, 29, 9]); // area_px = 300

        let (_dir, reader) =
            export_multi_image_and_open(vec![("img_a.tif", vec![a1, a2]), ("img_b.tif", vec![b1])]);

        let spec = AggregateSpec {
            key_mode: GroupKeyMode::Image,
            group_by_class: false,
            split_colocalized: false,
            metric_ids: vec!["area_px".to_string()],
            agg_fns: vec!["AVG"],
        };
        let rows = reader
            .aggregate_objects(&ObjectFilter::default(), &spec)
            .unwrap();

        assert_eq!(rows.len(), 2, "one row per image");
        assert_eq!(rows[0].group_key, "img_a.tif");
        assert_eq!(rows[0].count, 2);
        assert_eq!(rows[0].metric_values, vec![Some(150.0)]);
        assert_eq!(rows[1].group_key, "img_b.tif");
        assert_eq!(rows[1].count, 1);
        assert_eq!(rows[1].metric_values, vec![Some(300.0)]);
    }

    #[test]
    fn aggregate_objects_group_by_class_fans_out_multi_class_object() {
        const CLASS_A: ObjectClass = ObjectClass::Valid(1);
        const CLASS_B: ObjectClass = ObjectClass::Valid(2);

        let mut multi = make_filled_object(1, [0, 0, 9, 9]);
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
        let mut rows = reader
            .aggregate_objects(&ObjectFilter::default(), &spec)
            .unwrap();
        rows.sort_by(|a, b| a.group_class.cmp(&b.group_class));

        assert_eq!(
            rows.len(),
            2,
            "a object carrying two classes contributes to both buckets"
        );
        assert_eq!(rows[0].group_class.as_deref(), Some("class_1"));
        assert_eq!(rows[0].count, 1);
        assert_eq!(rows[1].group_class.as_deref(), Some("class_2"));
        assert_eq!(rows[1].count, 1);
    }

    #[test]
    fn aggregate_objects_split_colocalized_produces_two_buckets() {
        let mut coloc = make_filled_object(1, [0, 0, 9, 9]);
        coloc.add_colocalizing_object(ObjectClass::Valid(9), ObjectId(99));
        let not_coloc = make_filled_object(2, [0, 0, 9, 9]);

        let (_dir, reader) = export_multi_image_and_open(vec![("img.tif", vec![coloc, not_coloc])]);

        let spec = AggregateSpec {
            key_mode: GroupKeyMode::Image,
            group_by_class: false,
            split_colocalized: true,
            metric_ids: vec![],
            agg_fns: vec![],
        };
        let rows = reader
            .aggregate_objects(&ObjectFilter::default(), &spec)
            .unwrap();

        assert_eq!(
            rows.len(),
            2,
            "one colocalizing + one non-colocalizing bucket"
        );
        // false < true, matching Rust's Option<bool> BTreeMap ordering.
        assert_eq!(rows[0].colocalized, Some(false));
        assert_eq!(rows[0].count, 1);
        assert_eq!(rows[1].colocalized, Some(true));
        assert_eq!(rows[1].count, 1);
    }

    #[test]
    fn aggregate_objects_regex_groups_by_capture_and_drops_non_matching() {
        let a1 = make_filled_object(1, [0, 0, 9, 9]);
        let a2 = make_filled_object(2, [0, 0, 9, 9]);
        let control = make_filled_object(3, [0, 0, 9, 9]);

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
        let rows = reader
            .aggregate_objects(&ObjectFilter::default(), &spec)
            .unwrap();

        assert_eq!(
            rows.len(),
            1,
            "non-matching 'control.tif' must be dropped, not its own group"
        );
        assert_eq!(rows[0].group_key, "A1");
        assert_eq!(rows[0].count, 2);
    }

    #[test]
    fn aggregate_objects_all_null_metric_bucket_is_none() {
        // Neither object has any intensities_json, so any channel metric is NULL
        // for every contributing row in the bucket.
        let a1 = make_filled_object(1, [0, 0, 9, 9]);
        let a2 = make_filled_object(2, [0, 0, 9, 9]);
        let (_dir, reader) = export_multi_image_and_open(vec![("img.tif", vec![a1, a2])]);

        let spec = AggregateSpec {
            key_mode: GroupKeyMode::Image,
            group_by_class: false,
            split_colocalized: false,
            metric_ids: vec!["ch0_min_bit".to_string()],
            agg_fns: vec!["AVG"],
        };
        let rows = reader
            .aggregate_objects(&ObjectFilter::default(), &spec)
            .unwrap();

        assert_eq!(rows.len(), 1);
        assert_eq!(
            rows[0].metric_values,
            vec![None],
            "an all-NULL metric bucket must be None, matching apply_agg's empty-vec -> blank-cell behavior"
        );
    }

    /// `area_px = width` for a `height == 1` bbox - lets these tests pick
    /// exact metric values directly instead of reverse-engineering a bbox
    /// shape for each one.
    fn object_with_area(id: u128, area_px: u32) -> Object {
        make_filled_object(id, [0, 0, area_px - 1, 0])
    }

    /// Every `AggregateSpec::agg_fns` entry actually used in production
    /// (`AggFunc::sql_fn`, `results_loader.rs`) against two groups - one odd
    /// count, one even count (median's two code paths) - with every expected
    /// value hand-calculated independently of the implementation, not just
    /// cross-checked against `apply_agg`. This is the direct correctness
    /// check `apply_agg`'s own tests are for the Rust path; nothing
    /// previously exercised `STDDEV_SAMP`/`MEDIAN`/`SUM`/`MIN`/`MAX` through
    /// the actual SQL aggregation path at all.
    #[test]
    fn aggregate_objects_sum_min_max_median_and_stdev_match_hand_calculated_values() {
        // img_a: odd count (5) -> MEDIAN is a plain middle element.
        let odd = vec![
            object_with_area(1, 10),
            object_with_area(2, 20),
            object_with_area(3, 30),
            object_with_area(4, 40),
            object_with_area(5, 50),
        ];
        // img_b: even count (4) -> MEDIAN averages the two middle elements.
        let even = vec![
            object_with_area(6, 10),
            object_with_area(7, 20),
            object_with_area(8, 30),
            object_with_area(9, 40),
        ];
        let (_dir, reader) =
            export_multi_image_and_open(vec![("img_a.tif", odd), ("img_b.tif", even)]);

        let spec = AggregateSpec {
            key_mode: GroupKeyMode::Image,
            group_by_class: false,
            split_colocalized: false,
            metric_ids: vec!["area_px".to_string()],
            agg_fns: vec!["SUM", "MIN", "MAX", "AVG", "MEDIAN", "STDDEV_SAMP"],
        };
        let rows = reader
            .aggregate_objects(&ObjectFilter::default(), &spec)
            .unwrap();

        assert_eq!(rows.len(), 2);

        let assert_row = |row: &AggregatedRow, expected: [f64; 6], tolerance: f64| {
            let got: Vec<f64> = row.metric_values.iter().map(|v| v.unwrap()).collect();
            for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
                assert!(
                    (g - e).abs() <= tolerance,
                    "{:?} col {i}: got {g}, expected {e}",
                    row.group_key
                );
            }
        };

        // Values: 10,20,30,40,50 -> sum=150, min=10, max=50, avg=30, median=30
        // (the middle element), sample stdev = sqrt(1000/4) = 15.8113883...
        assert_row(
            &rows[0],
            [150.0, 10.0, 50.0, 30.0, 30.0, 1000f64.sqrt() / 2.0],
            1e-9,
        );
        // Values: 10,20,30,40 -> sum=100, min=10, max=40, avg=25,
        // median=(20+30)/2=25, sample stdev = sqrt(500/3) = 12.9099444...
        assert_row(
            &rows[1],
            [100.0, 10.0, 40.0, 25.0, 25.0, (500f64 / 3.0).sqrt()],
            1e-9,
        );
    }

    #[test]
    fn aggregate_objects_stddev_samp_of_a_single_value_group_is_zero_not_null() {
        // Sample stdev is mathematically undefined for n=1; DuckDB's
        // STDDEV_SAMP returns SQL NULL there, but this codebase's Rust
        // ground truth (`apply_agg`'s Stdev branch) reports 0.0 for a
        // single-value input rather than leaving the cell blank - the
        // `COALESCE(STDDEV_SAMP(...), CASE WHEN COUNT(...) >= 1 THEN 0.0 END)`
        // in `aggregate_objects` exists specifically to recover that. This
        // was previously untested end-to-end.
        let solo = vec![object_with_area(1, 42)];
        let (_dir, reader) = export_multi_image_and_open(vec![("img.tif", solo)]);

        let spec = AggregateSpec {
            key_mode: GroupKeyMode::Image,
            group_by_class: false,
            split_colocalized: false,
            metric_ids: vec!["area_px".to_string()],
            agg_fns: vec!["STDDEV_SAMP"],
        };
        let rows = reader
            .aggregate_objects(&ObjectFilter::default(), &spec)
            .unwrap();

        assert_eq!(rows.len(), 1);
        assert_eq!(
            rows[0].metric_values,
            vec![Some(0.0)],
            "a single-value group's sample stdev must be reported as 0.0, not NULL/blank"
        );
    }

    #[test]
    fn aggregate_objects_stddev_samp_of_identical_values_is_zero() {
        let same = vec![
            object_with_area(1, 25),
            object_with_area(2, 25),
            object_with_area(3, 25),
        ];
        let (_dir, reader) = export_multi_image_and_open(vec![("img.tif", same)]);

        let spec = AggregateSpec {
            key_mode: GroupKeyMode::Image,
            group_by_class: false,
            split_colocalized: false,
            metric_ids: vec!["area_px".to_string()],
            agg_fns: vec!["STDDEV_SAMP", "MEDIAN"],
        };
        let rows = reader
            .aggregate_objects(&ObjectFilter::default(), &spec)
            .unwrap();

        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].metric_values, vec![Some(0.0), Some(25.0)]);
    }

    #[test]
    fn coloc_filter_restricts_to_matching_status() {
        const ID_COLOC: u128 = 1;
        const ID_NOT_COLOC: u128 = 2;
        const PARTNER_CLASS: ObjectClass = ObjectClass::Valid(1);
        let coloc_id = ObjectId(ID_COLOC).to_string();
        let not_coloc_id = ObjectId(ID_NOT_COLOC).to_string();

        let mut coloc_object = make_filled_object(ID_COLOC, [0, 0, 1, 1]);
        coloc_object.add_colocalizing_object(PARTNER_CLASS, ObjectId(99));
        let not_coloc_object = make_filled_object(ID_NOT_COLOC, [5, 5, 6, 6]);

        let (_dir, reader) = export_and_open(vec![coloc_object, not_coloc_object]);

        let yes = reader
            .get_objects(&ObjectFilter {
                coloc_filter: Some(vec![coloc_filter_label_any().into()]),
                ..Default::default()
            })
            .unwrap();
        assert_eq!(yes.len(), 1);
        assert_eq!(yes[0].object_id, coloc_id);

        let no = reader
            .get_objects(&ObjectFilter {
                coloc_filter: Some(vec![coloc_filter_label_no().into()]),
                ..Default::default()
            })
            .unwrap();
        assert_eq!(no.len(), 1);
        assert_eq!(no[0].object_id, not_coloc_id);

        let all = reader.get_objects(&ObjectFilter::default()).unwrap();
        assert_eq!(all.len(), 2, "no coloc_filter set → both rows returned");

        let none_selected = reader
            .get_objects(&ObjectFilter {
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

        let object_a = make_filled_object(ID_A, [0, 0, 1, 1]);
        let object_b = make_filled_object(ID_B, [5, 5, 6, 6]);
        let (_dir, reader) = export_and_open(vec![object_a, object_b]);

        // Single id: exact match, and the round-tripped `object_id` string is
        // byte-identical to `ObjectId::to_string()` — confirms `CAST(object_id
        // AS VARCHAR)` produces the same lowercase-dashed format written at
        // export time, not just "a filter that happens to work".
        let single = reader
            .get_objects(&ObjectFilter {
                object_id_filter: Some(vec![id_a.clone()]),
                ..Default::default()
            })
            .unwrap();
        assert_eq!(single.len(), 1);
        assert_eq!(single[0].object_id, id_a);

        // Two ids: both returned, in the default object_id order.
        let both = reader
            .get_objects(&ObjectFilter {
                object_id_filter: Some(vec![id_b.clone(), id_a.clone()]),
                ..Default::default()
            })
            .unwrap();
        assert_eq!(both.len(), 2);
        assert_eq!(both[0].object_id, id_a);
        assert_eq!(both[1].object_id, id_b);

        // Unknown id: no rows.
        let unknown = reader
            .get_objects(&ObjectFilter {
                object_id_filter: Some(vec![ObjectId(999).to_string()]),
                ..Default::default()
            })
            .unwrap();
        assert!(unknown.is_empty());

        // Active filter with nothing selected → 0 rows, same convention as the
        // other filters.
        let none_selected = reader
            .get_objects(&ObjectFilter {
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
        const ID_NO_COLOC: u128 = 3;
        const PARTNER_B: ObjectClass = ObjectClass::Valid(2);
        const PARTNER_C: ObjectClass = ObjectClass::Valid(3);
        let id_with_b = ObjectId(ID_WITH_B).to_string();

        let mut object_with_b = make_filled_object(ID_WITH_B, [0, 0, 1, 1]);
        object_with_b.add_colocalizing_object(PARTNER_B, ObjectId(99));
        let mut object_with_c = make_filled_object(ID_WITH_C, [5, 5, 6, 6]);
        object_with_c.add_colocalizing_object(PARTNER_C, ObjectId(98));
        // A object with no colocalization at all (`coloc_json` = "{}") sitting
        // alongside colocalizing ones — regression coverage for a real bug
        // where `coloc_json` (a native DuckDB JSON column) compared directly
        // to a string literal like `''`/`'{}'` made DuckDB try to parse the
        // *literal* as JSON to perform the comparison, crashing on every row
        // scanned regardless of that row's own value (reproduced on Windows,
        // and reproducible here too once exercised — see
        // `coloc_is_colocalized_expr`'s doc comment for the full story).
        let object_no_coloc = make_filled_object(ID_NO_COLOC, [10, 10, 11, 11]);

        let (_dir, reader) = export_and_open(vec![object_with_b, object_with_c, object_no_coloc]);

        // class_label() falls back to "class_{n}" when no name map is provided.
        let partner_classes = reader.get_coloc_partner_class_names().unwrap();
        assert_eq!(
            partner_classes,
            vec!["class_2".to_string(), "class_3".to_string()]
        );

        let with_b = reader
            .get_objects(&ObjectFilter {
                coloc_filter: Some(vec![coloc_filter_label_with("class_2")]),
                ..Default::default()
            })
            .unwrap();
        assert_eq!(with_b.len(), 1);
        assert_eq!(with_b[0].object_id, id_with_b);
    }

    #[test]
    fn get_objects_sorts_by_requested_column() {
        const ID_SMALL: u128 = 1;
        const ID_LARGE: u128 = 2;
        let small_id = ObjectId(ID_SMALL).to_string();
        let large_id = ObjectId(ID_LARGE).to_string();

        // bbox [0,0,1,1] -> 2x2 = 4px; bbox [0,0,9,9] -> 10x10 = 100px.
        let small = make_filled_object(ID_SMALL, [0, 0, 1, 1]);
        let large = make_filled_object(ID_LARGE, [0, 0, 9, 9]);
        let (_dir, reader) = export_and_open(vec![large, small]);

        let asc = reader
            .get_objects(&ObjectFilter {
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
            .get_objects(&ObjectFilter {
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
    fn get_objects_sorts_by_channel_column_when_some_objects_have_no_intensities() {
        // Regression test for a latent bug in `channel_stat_expr`: sorting by a
        // channel column used to throw "Malformed JSON" the moment any row's
        // `intensities_json` was `"{}"` (no measured intensities at all), because
        // the old bracket-form `json_extract(col, ['0'])` returns a JSON array
        // wrapping a NULL rather than a scalar NULL, which `->>` can't handle.
        use crate::object::Intensity;
        use indexmap::IndexMap;

        const ID_NO_INTENSITY: u128 = 1;
        const ID_HAS_INTENSITY: u128 = 2;
        let no_intensity_id = ObjectId(ID_NO_INTENSITY).to_string();
        let has_intensity_id = ObjectId(ID_HAS_INTENSITY).to_string();

        let no_intensity = make_filled_object(ID_NO_INTENSITY, [0, 0, 1, 1]);
        let mut has_intensity = make_filled_object(ID_HAS_INTENSITY, [0, 0, 1, 1]);
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
            .get_objects(&ObjectFilter {
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
    fn get_objects_unsortable_column_falls_back_to_object_id_order() {
        let object_a = make_filled_object(1, [0, 0, 1, 1]);
        let object_b = make_filled_object(2, [0, 0, 1, 1]);
        let (_dir, reader) = export_and_open(vec![object_b, object_a]);

        let rows = reader
            .get_objects(&ObjectFilter {
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
    fn stack_range_is_none_when_every_object_shares_one_value() {
        let object_a = make_filled_object_at_plane(1, [0, 0, 1, 1], 0, 0);
        let object_b = make_filled_object_at_plane(2, [5, 5, 6, 6], 0, 0);
        let (_dir, reader) = export_and_open(vec![object_a, object_b]);

        assert_eq!(reader.get_t_stack_range().unwrap(), None);
        assert_eq!(reader.get_z_stack_range().unwrap(), None);
    }

    #[test]
    fn stack_range_reports_min_max_when_values_differ() {
        let object_t0 = make_filled_object_at_plane(1, [0, 0, 1, 1], 0, 3);
        let object_t2 = make_filled_object_at_plane(2, [5, 5, 6, 6], 2, 3);
        let object_t5 = make_filled_object_at_plane(3, [9, 9, 10, 10], 5, 7);
        let (_dir, reader) = export_and_open(vec![object_t0, object_t2, object_t5]);

        assert_eq!(reader.get_t_stack_range().unwrap(), Some((0, 5)));
        assert_eq!(reader.get_z_stack_range().unwrap(), Some((3, 7)));
    }

    #[test]
    fn get_objects_filters_by_t_stack_and_z_stack() {
        const ID_T0: u128 = 1;
        const ID_T2: u128 = 2;
        let id_t0 = ObjectId(ID_T0).to_string();
        let id_t2 = ObjectId(ID_T2).to_string();

        let object_t0 = make_filled_object_at_plane(ID_T0, [0, 0, 1, 1], 0, 1);
        let object_t2 = make_filled_object_at_plane(ID_T2, [5, 5, 6, 6], 2, 4);
        let (_dir, reader) = export_and_open(vec![object_t0, object_t2]);

        let only_t0 = reader
            .get_objects(&ObjectFilter {
                t_stack_filter: Some(0),
                ..Default::default()
            })
            .unwrap();
        assert_eq!(only_t0.len(), 1);
        assert_eq!(only_t0[0].object_id, id_t0);

        let only_z4 = reader
            .get_objects(&ObjectFilter {
                z_stack_filter: Some(4),
                ..Default::default()
            })
            .unwrap();
        assert_eq!(only_z4.len(), 1);
        assert_eq!(only_z4[0].object_id, id_t2);

        let both_unset = reader.get_objects(&ObjectFilter::default()).unwrap();
        assert_eq!(
            both_unset.len(),
            2,
            "no t/z filter set -> every frame returned"
        );
    }
}
