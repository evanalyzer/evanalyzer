use crate::object::Intensity;
use crate::pipeline::pipeline_cache::GlobalPipelineCache;
use crate::storage::PipelineResultExporter;
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
///
/// Every connection to a results file in this process must come from here -
/// a plain `Connection::open` next to a cached one fails on Windows.
pub fn shared_connection(path: &Path) -> Result<Connection, InternalErrors> {
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

    // Plain display name for `object_class_name`
    fn class_label(&self, class: &ObjectClass) -> String {
        match class {
            ObjectClass::Unset => "unset".to_string(),
            ObjectClass::Valid(n) => match self.class_names.get(class) {
                Some((name, _color)) => name.clone(),
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

CREATE TABLE IF NOT EXISTS images (
    image_name      VARCHAR NOT NULL,
    image_rel_path  VARCHAR NOT NULL PRIMARY KEY,
    successful      BOOLEAN NOT NULL DEFAULT true,
    error_message   VARCHAR,
    disabled        BOOLEAN NOT NULL DEFAULT false,
    width           UINTEGER NOT NULL,
    height          UINTEGER NOT NULL,
    c_stacks        UINTEGER NOT NULL,
    z_stacks        UINTEGER NOT NULL,
    t_stacks        UINTEGER NOT NULL
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

// Keys by the class's raw numeric id (not its display name) - a class
// rename never invalidates an already-written `coloc_json`, and the id is
// exactly what `Column::ColocCount`/`coloc_count` (results_generator.rs)
// need anyway, since it only sums values regardless of key.
fn coloc_to_json(colocalized_with: &IndexMap<ObjectClass, Vec<ObjectId>>) -> String {
    let mut entries = Vec::with_capacity(colocalized_with.len());
    for (class, ids) in colocalized_with {
        let key = match class {
            ObjectClass::Unset => "unset".to_string(),
            ObjectClass::Valid(n) => n.to_string(),
        };
        let ids_str = ids
            .iter()
            .map(|id| format!("\"{}\"", id))
            .collect::<Vec<_>>()
            .join(",");
        entries.push(format!("\"{}\":[{}]", key, ids_str));
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
                let coloc_json = coloc_to_json(&object.colocalized_with);
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
        nr_c_stacks: u32,
        nr_z_stacks: u32,
        nr_t_stacks: u32,
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
            "INSERT INTO images (image_name, image_rel_path, successful, error_message, width, height, c_stacks, z_stacks, t_stacks) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            params![image_name, image_rel, successful, error, width, height, nr_c_stacks, nr_z_stacks, nr_t_stacks],
        )
        .map_err(|e| InternalErrors::Io(e.to_string()))?;
        Ok(())
    }
}
