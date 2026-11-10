//! # duckdb
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-06-05
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.

use crate::ImageDimensionality;
use crate::object::{Intensity, Object};
use crate::pipeline::pipeline_cache::GlobalPipelineCache;
use crate::pipeline::pipeline_measure::{Measurement, MeasurementRegion, MeasurementValues};
use crate::storage::stored_object::{ObjectFilter, StoredObject, TrackAssignment};
use crate::storage::{
    DatasetStore, DatasetValue, DatasetValueAbout, ImageSummary, PipelineResultExporter,
};
use duckdb::arrow::array::{
    ArrayBuilder, ArrayRef, BinaryBuilder, BooleanBuilder, FixedSizeBinaryBuilder, Float64Builder,
    Int32Builder, ListBuilder, MapBuilder, StringBuilder, UInt8Builder, UInt32Builder,
    UInt64Builder,
};
use duckdb::arrow::datatypes::{DataType, Field, Schema};
use duckdb::arrow::record_batch::RecordBatch;
use duckdb::types::Value;
use duckdb::{Config, Connection, params};
use evanalyzer_cfg::core_types::{
    InternalErrors, MeasurementInfo, ObjectClass, ObjectId, max_gray_value,
};
use evanalyzer_cfg::settings::meta_data::MetaData;
use indexmap::IndexMap;
use std::collections::HashMap;
use std::fmt::Write as _;
use std::path::Path;
use std::path::PathBuf;
use std::sync::{Arc, LazyLock, Mutex};
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
/// Closed with the last [`ResultsConnection`] to the file: an entry stays
/// only while someone uses it (an analysis writing, the results window
/// reading). Before, entries were never closed - every results file opened
/// in a session kept its DuckDB instance, its buffer pool (up to the memory
/// limit) and its file handle until the app ended.
static CONNECTION_CACHE: LazyLock<Mutex<HashMap<PathBuf, SharedDatabase>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

/// One open results file: the anchor connection and how many
/// [`ResultsConnection`]s to it are alive.
struct SharedDatabase {
    anchor: Connection,
    users: usize,
}

/// A connection to a results file from [`shared_connection`] - use it as a
/// [`Connection`]. When the last one to a file is dropped, the file's DuckDB
/// instance closes: its memory is freed, and DuckDB folds its write-ahead
/// log back into the file.
pub struct ResultsConnection {
    /// `Some` until dropped - taken first, so the connection is closed
    /// before the anchor.
    conn: Option<Connection>,
    path: PathBuf,
}

impl ResultsConnection {
    /// Another connection to the same file, counted like this one (not
    /// [`Connection::try_clone`], which the file's closing wouldn't wait
    /// for).
    pub fn try_clone(&self) -> Result<ResultsConnection, InternalErrors> {
        match &self.conn {
            // In memory: not in the cache, a plain clone.
            Some(conn) if self.path.as_os_str().is_empty() => Ok(ResultsConnection {
                conn: Some(
                    conn.try_clone()
                        .map_err(|e| InternalErrors::Io(e.to_string()))?,
                ),
                path: PathBuf::new(),
            }),
            _ => shared_connection(&self.path),
        }
    }

    /// An in-memory database (tests): no file, not shared, no file access
    /// (see [`lock_down`]).
    pub fn open_in_memory() -> Result<ResultsConnection, InternalErrors> {
        let conn = Connection::open_in_memory_with_flags(offline_config()?)
            .map_err(|e| InternalErrors::Io(e.to_string()))?;
        lock_down(&conn, None)?;
        Ok(ResultsConnection {
            conn: Some(conn),
            path: PathBuf::new(),
        })
    }
}

impl ResultsConnection {
    /// The one folder SQL on this file may write to - where a Parquet
    /// export is written first (see [`export_staging_dir`]); `None` in
    /// memory.
    pub fn export_staging_dir(&self) -> Option<PathBuf> {
        (!self.path.as_os_str().is_empty()).then(|| export_staging_dir(&self.path))
    }
}

/// `<results file>.export`, beside it as DuckDB's own `<results file>.tmp`:
/// the one folder SQL on that file may write to (see [`lock_down`]) - a
/// Parquet export goes there first, then to where it was asked for.
pub fn export_staging_dir(results_file: &Path) -> PathBuf {
    let mut dir = results_file.as_os_str().to_owned();
    dir.push(".export");
    PathBuf::from(dir)
}

/// No file access from SQL on `conn`'s database, except below
/// `export_dir`: a results file can come from anyone, and a view or macro
/// in it runs with the query that reads it - `read_text('~/.ssh/id_rsa')`
/// would show a local file in the results or an export, `read_csv('https://…')`
/// would phone home. Set right after opening, before anything reads the
/// file; DuckDB refuses to switch it back on or widen the folders while
/// the database is open, so the file can't undo it (no
/// `lock_configuration` needed - the memory limit is still set later).
/// DuckDB's own files (the WAL, spilling to its temp folder) aren't
/// affected.
fn lock_down(conn: &Connection, export_dir: Option<&Path>) -> Result<(), InternalErrors> {
    let allowed = match export_dir {
        Some(dir) => {
            let dir = std::path::absolute(dir).map_err(|e| InternalErrors::Io(e.to_string()))?;
            format!(
                "SET allowed_directories = ['{}'];",
                dir.to_string_lossy().replace('\'', "''")
            )
        }
        None => String::new(),
    };
    conn.execute_batch(&format!("{allowed} SET enable_external_access = false;"))
        .map_err(|e| InternalErrors::Io(format!("Could not restrict the results file: {e}")))
}

impl std::ops::Deref for ResultsConnection {
    type Target = Connection;
    fn deref(&self) -> &Connection {
        self.conn.as_ref().expect("a live connection")
    }
}

impl std::ops::DerefMut for ResultsConnection {
    fn deref_mut(&mut self) -> &mut Connection {
        self.conn.as_mut().expect("a live connection")
    }
}

impl Drop for ResultsConnection {
    fn drop(&mut self) {
        drop(self.conn.take());
        let mut cache = CONNECTION_CACHE.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(shared) = cache.get_mut(&self.path) {
            shared.users = shared.users.saturating_sub(1);
            if shared.users == 0 {
                // Dropped under the lock: an open of the same file waits
                // until it is closed.
                cache.remove(&self.path);
                share_memory_limit(&cache);
            }
        }
    }
}

/// DuckDB's settings for every connection: no extension is downloaded or
/// loaded on its own (DuckDB would fetch one from extensions.duckdb.org the
/// first time a query needs it) - no network, the same on an offline
/// computer, nothing outside the licence audit. The built-in ones are
/// loaded where used (`LOAD parquet` before a Parquet export), not on every
/// connection (~10 MB). Not locked (`lock_configuration`): the memory limit
/// is set later.
fn offline_config() -> Result<Config, InternalErrors> {
    let err = |e: duckdb::Error| InternalErrors::Io(e.to_string());
    Config::default()
        .enable_autoload_extension(false)
        .and_then(|c| c.with("allow_community_extensions", "false"))
        .map_err(err)
}

/// Returns a connection usable for `path`, sharing the resident anchor
/// connection for that path if one is already open (see `CONNECTION_CACHE`'s
/// doc comment), opening and caching a fresh one otherwise.
///
/// Every connection to a results file in this process must come from here -
/// a plain `Connection::open` next to a cached one fails on Windows.
pub fn shared_connection(path: &Path) -> Result<ResultsConnection, InternalErrors> {
    let to_io_err = |e: duckdb::Error| InternalErrors::Io(e.to_string());
    let mut cache = CONNECTION_CACHE.lock().unwrap_or_else(|e| e.into_inner());
    let handle = |conn: Connection| ResultsConnection {
        conn: Some(conn),
        path: path.to_path_buf(),
    };

    if let Some(shared) = cache.get_mut(path) {
        match shared.anchor.try_clone() {
            Ok(conn) => {
                shared.users += 1;
                return Ok(handle(conn));
            }
            // The anchor connection has gone bad (e.g. the file was deleted
            // or replaced out from under it) - drop it and fall through to
            // open a fresh one below, rather than keep handing out clones
            // of a connection that can no longer serve queries.
            Err(_) => {
                cache.remove(path);
            }
        }
    }

    let anchor = Connection::open_with_flags(path, offline_config()?).map_err(to_io_err)?;
    lock_down(&anchor, Some(&export_staging_dir(path)))?;
    let conn = anchor.try_clone().map_err(to_io_err)?;
    cache.insert(path.to_path_buf(), SharedDatabase { anchor, users: 1 });
    share_memory_limit(&cache);
    Ok(handle(conn))
}

/// Gives every open results file its share of one memory limit (see
/// [`crate::memory::storage::results_memory_limit_per_file_mb`]) - when a
/// file opens or closes. DuckDB keeps up to 80 % of the RAM for itself by
/// default, per open file; capped, an older computer keeps RAM for the
/// rest, and beyond it DuckDB spills to a temporary file.
fn share_memory_limit(cache: &HashMap<PathBuf, SharedDatabase>) {
    if cache.is_empty() {
        return;
    }
    let limit_mb = crate::memory::storage::results_memory_limit_per_file_mb(
        crate::memory::SystemRam::now().total,
        cache.len(),
    );
    // DuckDB's own threads: the cores EVAnalyzer may use (`--max-threads`).
    let threads = crate::memory::cpu_cores();
    for shared in cache.values() {
        if let Err(e) = shared.anchor.execute_batch(&format!(
            "SET memory_limit = '{limit_mb}MB'; SET threads = {threads};"
        )) {
            log::warn!("Could not limit DuckDB to {limit_mb} MB and {threads} threads: {e}");
        }
    }
}

/// The format of the results files this build writes (`run.schema_version`).
/// Raise it with every change a reader of an older build would misread, and
/// teach the readers the older formats they still open.
pub const RESULTS_SCHEMA_VERSION: u32 = 2;

// Format history:
// 1: the first.
// 2: profile measurements - `measurement_info.profile` and
//    `measurements.position` (a value per position, e.g. per Sholl ring);
//    an older build would read a profile's rows as one value each.
//    Also values per track (Measure Track Motion): `measurements.track_id`,
//    `region = 'track'`, `measurement_info.target = 'Track'`.

/// The format `conn`'s results file was written in - 0 for a file from before
/// it was recorded (no `run` table or no `schema_version`).
pub fn results_schema_version(conn: &Connection) -> Result<u32, InternalErrors> {
    let has_column: bool = conn
        .query_row(
            "SELECT count(*) > 0 FROM information_schema.columns
             WHERE table_name = 'run' AND column_name = 'schema_version'",
            [],
            |row| row.get(0),
        )
        .map_err(|e| InternalErrors::Io(e.to_string()))?;
    if !has_column {
        return Ok(0);
    }
    let version: Option<Option<u32>> = conn
        .query_row("SELECT max(schema_version) FROM run", [], |row| row.get(0))
        .map(Some)
        .map_err(|e| InternalErrors::Io(e.to_string()))?;
    Ok(version.flatten().unwrap_or(0))
}

/// Refuses a results file written by a newer build: its format may hold
/// what this build would misread.
pub fn reject_newer_results(conn: &Connection) -> Result<(), InternalErrors> {
    let version = results_schema_version(conn)?;
    if version > RESULTS_SCHEMA_VERSION {
        return Err(InternalErrors::ParseError(format!(
            "these results were saved by a newer version of EVAnalyzer (format version \
             {version}, this build reads up to version {RESULTS_SCHEMA_VERSION}) - update \
             EVAnalyzer to open them"
        )));
    }
    Ok(())
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

/// The per-object measurements' `objects.measurement_values` positions an
/// earlier run gave (`measurement_info.object_index`).
fn existing_object_indices(conn: &Connection) -> Result<HashMap<String, i32>, InternalErrors> {
    let err = |e: duckdb::Error| InternalErrors::Io(e.to_string());
    let mut stmt = conn
        .prepare("SELECT key, object_index FROM measurement_info WHERE object_index IS NOT NULL")
        .map_err(err)?;
    let rows = stmt
        .query_map([], |row| Ok((row.get(0)?, row.get(1)?)))
        .map_err(err)?;
    rows.collect::<Result<_, _>>().map_err(err)
}

pub struct DuckDbExporter {
    // Connection is Send but !Sync; the Mutex makes the struct Sync so it can
    // satisfy the `PipelineResultExporter: Send + Sync` bound.
    conn: Mutex<ResultsConnection>,
    /// Maps ObjectClass → human-readable name from project classification settings.
    pub class_names: HashMap<ObjectClass, (String, u32)>,
    /// Each object's mask is stored (`objects.mask`, see `mask_codec`) - the
    /// project's "Store object masks"; else the column stays NULL.
    store_masks: bool,
    /// Each per-object measurement's position in `objects.measurement_values`
    /// (1-based, by key) - given the first time a key is exported, kept in
    /// `measurement_info.object_index`.
    object_indices: Mutex<HashMap<String, i32>>,
}

impl DuckDbExporter {
    /// Opens (or creates) the output file, runs DDL once, and returns a ready exporter.
    pub fn new(
        output_path: impl Into<PathBuf>,
        class_names: HashMap<ObjectClass, (String, u32)>,
        project_meta: &MetaData,
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
                if let ObjectClass::Global(n) = class {
                    app.append_row(params![*n, name, color])
                        .map_err(|e| InternalErrors::Io(e.to_string()))?;
                }
            }
        }

        let author_names: Value = Value::Text(
            serde_json::to_string(
                &project_meta
                    .authors
                    .iter()
                    .map(|a| &a.full_name)
                    .collect::<Vec<_>>(),
            )
            .map_err(|e| InternalErrors::Io(e.to_string()))?,
        );

        let author_orgs: Value = Value::Text(
            serde_json::to_string(
                &project_meta
                    .authors
                    .iter()
                    .map(|a| &a.organization)
                    .collect::<Vec<_>>(),
            )
            .map_err(|e| InternalErrors::Io(e.to_string()))?,
        );

        conn.execute(
            "INSERT INTO run (
                schema_version, status, title, short_description, description, app_version,
                author_full_name, author_organization
            ) VALUES (
                ?, 'running', ?, ?, ?, ?, ?::VARCHAR[], ?::VARCHAR[]
            )",
            params![
                RESULTS_SCHEMA_VERSION,
                project_meta.name,
                project_meta.short_description,
                project_meta.description,
                project_meta.app_version,
                author_names,
                author_orgs,
            ],
        )
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
            store_masks: true,
            object_indices: Mutex::new(HashMap::new()),
        })
    }

    /// Whether the objects' masks are stored (default: they are).
    pub fn with_masks(mut self, store: bool) -> Self {
        self.store_masks = store;
        self
    }

    // Plain display name for `object_class_name`
    /// The results database of an earlier analysis, to read and add to -
    /// unlike [`Self::new`], it must exist and no new run is recorded.
    pub fn open_existing(path: &Path) -> Result<Self, InternalErrors> {
        if !path.is_file() {
            return Err(InternalErrors::Io(format!(
                "No results database at {}",
                path.display()
            )));
        }
        let conn = shared_connection(path)?;
        // Only into a file of this build's format: another one's rows wouldn't
        // fit. Format 1 becomes 2 here - it only lacks the profile columns
        // (it can't hold a profile, so the defaults are exact).
        let version = results_schema_version(&conn)?;
        if version == 1 {
            conn.execute_batch(
                // DuckDB can't add a NOT NULL column: readers take
                // `coalesce(profile, false)`.
                "ALTER TABLE measurement_info ADD COLUMN IF NOT EXISTS \
                     profile BOOLEAN DEFAULT false;
                 ALTER TABLE measurements ADD COLUMN IF NOT EXISTS position DOUBLE;
                 ALTER TABLE measurements ADD COLUMN IF NOT EXISTS track_id UBIGINT;
                 UPDATE run SET schema_version = 2;",
            )
            .map_err(|e| InternalErrors::Io(e.to_string()))?;
        } else if version != RESULTS_SCHEMA_VERSION {
            return Err(InternalErrors::ParseError(format!(
                "these results are in format version {version}, this build writes version \
                 {RESULTS_SCHEMA_VERSION} - run the analysis again"
            )));
        }
        // Added after version 1 shipped, without a version bump: files
        // written before it get it here, at the end as in `CREATE_TABLES`.
        // `measurements.track_id` came to format 2 before it was released.
        conn.execute_batch(
            "ALTER TABLE objects ADD COLUMN IF NOT EXISTS parent_track_ids UBIGINT[];
             ALTER TABLE measurements ADD COLUMN IF NOT EXISTS track_id UBIGINT;",
        )
        .map_err(|e| InternalErrors::Io(e.to_string()))?;
        let object_indices = existing_object_indices(&conn)?;
        Ok(Self {
            conn: Mutex::new(conn),
            class_names: HashMap::new(),
            store_masks: true,
            object_indices: Mutex::new(object_indices),
        })
    }

    /// [`Self::open_existing`] for a new run on the database's objects (a
    /// copy of an earlier analysis's): its run is recorded as running again,
    /// with this run's project details.
    pub fn continue_run(path: &Path, project_meta: &MetaData) -> Result<Self, InternalErrors> {
        let exporter = Self::open_existing(path)?;
        exporter
            .lock_conn()
            .execute(
                "UPDATE run SET status = 'running', message = NULL, finished_at = NULL,
                     started_at = current_timestamp, title = ?, short_description = ?,
                     description = ?, app_version = ?",
                params![
                    project_meta.name,
                    project_meta.short_description,
                    project_meta.description,
                    project_meta.app_version,
                ],
            )
            .map_err(|e| InternalErrors::Io(e.to_string()))?;
        Ok(exporter)
    }

    /// The measurements of `cache`'s plane: each measurement's description
    /// (once per key, in `measurement_info`) and its values (`measurements`,
    /// a row each with its image, plane, region and classes).
    ///
    /// The values go in through the Arrow appender, like `objects`: a value
    /// per object for every object of a big image is far too many rows to
    /// insert one by one.
    fn write_measurements(
        &self,
        tx: &duckdb::Transaction<'_>,
        cache: &GlobalPipelineCache,
    ) -> Result<(), InternalErrors> {
        let err = |e: duckdb::Error| InternalErrors::Io(e.to_string());
        let measured = &cache.measurements;
        if measured.measurements.is_empty() {
            return Ok(());
        }
        let mut info = tx
            .prepare(
                "INSERT OR IGNORE INTO measurement_info \
                 (key, measurement, step, unit, target, range_min, range_max, aggregatable, \
                 object_index, related_objects, profile) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            )
            .map_err(err)?;
        let indices = self
            .object_indices
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        for Measurement { info: i, .. } in &measured.measurements {
            info.execute(params![
                i.key,
                i.name,
                i.step,
                format!("{:?}", i.unit),
                format!("{:?}", i.target),
                i.range.map(|r| r.0),
                i.range.map(|r| r.1),
                i.aggregatable,
                indices.get(&i.key),
                i.related_objects,
                i.profile
            ])
            .map_err(err)?;
        }
        drop(indices);

        let mut rows = MeasurementRows::new(self, cache);
        let mut app = tx.appender("measurements").map_err(err)?;
        for Measurement { info: i, values } in &measured.measurements {
            let unit = format!("{:?}", i.unit);
            let row =
                |region, object, related| MeasurementRow::of(i, &unit, region, object, related);
            match values {
                MeasurementValues::Image(values) => {
                    for (region, value) in values {
                        match region {
                            MeasurementRegion::Image => {
                                rows.push(row(RowRegion::Image, None, None), *value, None)
                            }
                            MeasurementRegion::Class(class) => {
                                rows.push(row(RowRegion::Class(*class), None, None), *value, None)
                            }
                        }
                        rows.flush_if_full(&mut app)?;
                    }
                }
                MeasurementValues::Objects(values) => {
                    for v in values {
                        rows.push(
                            row(RowRegion::Object, Some(&v.object), v.related.as_ref()),
                            v.value,
                            v.position,
                        );
                        rows.flush_if_full(&mut app)?;
                    }
                }
            }
        }
        rows.flush(&mut app)
    }

    /// `classes`' names as a JSON list of strings (as `json_string_array`
    /// writes them), into `out`.
    fn write_class_names<'c>(
        &self,
        out: &mut String,
        classes: impl Iterator<Item = &'c ObjectClass>,
    ) {
        out.push('[');
        for (i, class) in classes.enumerate() {
            if i > 0 {
                out.push(',');
            }
            match class {
                ObjectClass::Global(_) => match self.class_names.get(class) {
                    Some((name, _color)) => write_json_string(out, name),
                    None => write_json_string(out, &self.class_label(class)),
                },
                _ => write_json_string(out, &self.class_label(class)),
            }
        }
        out.push(']');
    }

    fn class_label(&self, class: &ObjectClass) -> String {
        match class {
            ObjectClass::Unset => "unset".to_string(),
            // Never stored: a module's scope ends before its objects are written.
            ObjectClass::Local(n) => format!("local_{n}"),
            ObjectClass::Global(n) => match self.class_names.get(class) {
                Some((name, _color)) => name.clone(),
                None => format!("class_{}", n),
            },
        }
    }
}

// ---------------------------------------------------------------------------
// Arrow columns for the objects table
// ---------------------------------------------------------------------------

/// Rows per Arrow batch handed to the Appender - bounds how many rows of a
/// huge image are buffered at once.
const ARROW_BATCH_ROWS: usize = 50_000;

/// The `objects` table's DOUBLE columns (placed by
/// `ObjectColumns::finish`).
#[derive(Clone, Copy)]
enum F64 {
    CentroidXPx,
    CentroidYPx,
    CentroidXNm,
    CentroidYNm,
    BboxCenterXPx,
    BboxCenterYPx,
    BboxCenterXNm,
    BboxCenterYNm,
    BboxXminNm,
    BboxYminNm,
    BboxXmaxNm,
    BboxYmaxNm,
    AreaNm2,
    PerimeterPx,
    PerimeterNm,
    Circularity,
    Solidity,
    AspectRatio,
    Roundness,
    Compactness,
    MajorAxisPx,
    MinorAxisPx,
    MajorAxisNm,
    MinorAxisNm,
    MajorAxisAngle,
    Eccentricity,
    FeretDiameterPx,
    MinFeretPx,
    FeretDiameterNm,
    MinFeretNm,
    PixelSizeXNm,
    PixelSizeYNm,
    PixelSizeZNm,
    BboxZminNm,
    BboxZmaxNm,
    CentroidZPx,
    CentroidZNm,
    BboxCenterZPx,
    BboxCenterZNm,
    VolumeNm3,
    SurfaceAreaNm2,
}

const F64_COLUMNS: usize = F64::SurfaceAreaNm2 as usize + 1;

/// One batch of `objects` rows, column by column, for the Appender's Arrow
/// path - the only way to append the `object_class_id` list: the row-wise
/// `append_row` rejects list values ("appending List values is not yet
/// supported", duckdb-rs issue #422). Arrow types map onto the table's:
/// object ids as Arrow UUIDs (16 bytes, see [`uuid_field`]) - as text, DuckDB
/// parsed them back under the connection lock; the JSON columns as text.
struct ObjectColumns {
    image_name: StringBuilder,
    image_rel_path: StringBuilder,
    c_stack: Int32Builder,
    z_stack: Int32Builder,
    t_stack: Int32Builder,
    object_id: FixedSizeBinaryBuilder,
    seg_class_name: StringBuilder,
    seg_class_id: Int32Builder,
    object_class_name: StringBuilder,
    object_class_id: ListBuilder<Int32Builder>,
    parent_ids: MapBuilder<Int32Builder, FixedSizeBinaryBuilder>,
    track_id: UInt64Builder,
    bbox_px: [UInt32Builder; 4],
    area_px: UInt64Builder,
    touches_edge: BooleanBuilder,
    image_bit_depth: UInt8Builder,
    /// `INTENSITY_COLUMNS`, in table order.
    intensities: [ListBuilder<Float64Builder>; INTENSITY_COLUMNS.len()],
    coloc_partner_ids: MapBuilder<Int32Builder, ListBuilder<FixedSizeBinaryBuilder>>,
    /// The per-object measurements' values, by `object_index`.
    measurement_values: ListBuilder<Float64Builder>,
    /// The mask, run-length encoded (`mask_codec`); NULL when not stored.
    mask: BinaryBuilder,
    /// Always NULL on export: tracking steps set it afterwards.
    parent_track_ids: ListBuilder<UInt64Builder>,
    ndim: UInt8Builder,
    bbox_z_px: [UInt32Builder; 2],
    volume_px: UInt64Builder,
    touches_z_edge: BooleanBuilder,
    f64: Vec<Float64Builder>,
    /// A row's text is formatted here, then copied into its column - no
    /// `String` per row.
    text: String,
}

/// The Arrow field of a UUID (canonical extension `arrow.uuid`: 16 bytes,
/// big-endian) - DuckDB takes it into a UUID column as it is.
fn uuid_field(name: &str) -> Field {
    Field::new(name, DataType::FixedSizeBinary(16), true)
        .with_metadata([("ARROW:extension:name".to_string(), "arrow.uuid".to_string())].into())
}

/// `id` as an Arrow UUID's bytes.
fn uuid_bytes(id: &ObjectId) -> [u8; 16] {
    id.0.to_be_bytes()
}

/// Appends `id` (NULL for `None`) to a builder of Arrow UUIDs.
fn append_uuid(builder: &mut FixedSizeBinaryBuilder, id: Option<&ObjectId>) {
    match id {
        // Always 16 bytes: the builder's width.
        Some(id) => builder
            .append_value(uuid_bytes(id))
            .expect("a UUID is 16 bytes"),
        None => builder.append_null(),
    }
}

/// A builder of Arrow UUIDs (see [`uuid_field`]).
fn uuid_builder(rows: usize) -> FixedSizeBinaryBuilder {
    FixedSizeBinaryBuilder::with_capacity(rows, 16)
}

/// `batch` with its `columns` typed as UUIDs (`RecordBatch::try_from_iter`
/// leaves fields without their extension).
fn with_uuid_columns(batch: RecordBatch, columns: &[&str]) -> Result<RecordBatch, InternalErrors> {
    let schema = batch.schema();
    let fields: Vec<Field> = schema
        .fields()
        .iter()
        .map(|f| match columns.contains(&f.name().as_str()) {
            true => uuid_field(f.name()),
            false => f.as_ref().clone(),
        })
        .collect();
    RecordBatch::try_new(Arc::new(Schema::new(fields)), batch.columns().to_vec())
        .map_err(|e| InternalErrors::Io(e.to_string()))
}

/// Writes `value` as a JSON string (quoted, `\` and `"` escaped) to `out`.
fn write_json_string(out: &mut String, value: &str) {
    out.push('"');
    for ch in value.chars() {
        match ch {
            '\\' => out.push_str("\\\\"),
            '"' => out.push_str("\\\""),
            _ => out.push(ch),
        }
    }
    out.push('"');
}

/// The per-channel intensity list columns, in table order, in gray values
/// (as ImageJ/Fiji): sum, mean, min, max, then how the values spread. The
/// spread's lists are NULL for an object without it (a stored annotation).
const INTENSITY_COLUMNS: [&str; 9] = [
    "intensity_sum_gray",
    "intensity_mean_gray",
    "intensity_min_gray",
    "intensity_max_gray",
    "intensity_std_gray",
    "intensity_median_gray",
    "intensity_lower_quartile_gray",
    "intensity_upper_quartile_gray",
    "intensity_mad_gray",
];

/// The first spread column in `INTENSITY_COLUMNS`.
const FIRST_SPREAD_COLUMN: usize = 4;

/// One channel's values in `INTENSITY_COLUMNS`' order (0..1); the spread
/// NaN where it wasn't measured.
fn intensity_stats(v: &Intensity) -> [f64; INTENSITY_COLUMNS.len()] {
    [
        v.sum_intensity,
        v.avg_intensity,
        v.min_intensity,
        v.max_intensity,
        v.std_intensity,
        v.median_intensity,
        v.lower_quartile_intensity,
        v.upper_quartile_intensity,
        v.mad_intensity,
    ]
}

/// `coloc_partner_ids` / `parent_ids` key for a class without a number.
const UNSET_CLASS_KEY: i32 = -1;

/// A class as a `coloc_partner_ids` / `parent_ids` key.
fn class_key(class: &ObjectClass) -> i32 {
    match class {
        ObjectClass::Global(n) => *n as i32,
        ObjectClass::Unset | ObjectClass::Local(_) => UNSET_CLASS_KEY,
    }
}

impl ObjectColumns {
    fn with_capacity(rows: usize) -> Self {
        let strings = || StringBuilder::with_capacity(rows, rows * 16);
        Self {
            image_name: strings(),
            image_rel_path: strings(),
            c_stack: Int32Builder::with_capacity(rows),
            z_stack: Int32Builder::with_capacity(rows),
            t_stack: Int32Builder::with_capacity(rows),
            object_id: uuid_builder(rows),
            seg_class_name: strings(),
            seg_class_id: Int32Builder::with_capacity(rows),
            object_class_name: strings(),
            object_class_id: ListBuilder::with_capacity(Int32Builder::with_capacity(rows), rows),
            parent_ids: MapBuilder::new(None, Int32Builder::new(), uuid_builder(0))
                .with_values_field(uuid_field("values")),
            track_id: UInt64Builder::with_capacity(rows),
            bbox_px: std::array::from_fn(|_| UInt32Builder::with_capacity(rows)),
            area_px: UInt64Builder::with_capacity(rows),
            touches_edge: BooleanBuilder::with_capacity(rows),
            image_bit_depth: UInt8Builder::with_capacity(rows),
            intensities: std::array::from_fn(|_| {
                ListBuilder::with_capacity(Float64Builder::with_capacity(rows * 4), rows)
            }),
            coloc_partner_ids: MapBuilder::new(
                None,
                Int32Builder::new(),
                ListBuilder::new(uuid_builder(0)).with_field(uuid_field("item")),
            )
            .with_values_field(Field::new(
                "values",
                DataType::List(Arc::new(uuid_field("item"))),
                true,
            )),
            measurement_values: ListBuilder::new(Float64Builder::new()),
            mask: BinaryBuilder::with_capacity(rows, rows * 16),
            parent_track_ids: ListBuilder::new(UInt64Builder::new()),
            ndim: UInt8Builder::with_capacity(rows),
            bbox_z_px: std::array::from_fn(|_| UInt32Builder::with_capacity(rows)),
            volume_px: UInt64Builder::with_capacity(rows),
            touches_z_edge: BooleanBuilder::with_capacity(rows),
            f64: (0..F64_COLUMNS)
                .map(|_| Float64Builder::with_capacity(rows))
                .collect(),
            text: String::new(),
        }
    }

    fn f64(&mut self, column: F64, value: f64) {
        self.f64[column as usize].append_value(value);
    }

    /// NULL for `None`.
    fn f64_opt(&mut self, column: F64, value: Option<f64>) {
        self.f64[column as usize].append_option(value);
    }

    /// One row of every `intensity_*` list: position c + 1 holds channel c,
    /// NULL for a channel without a measurement - in gray values (`bit_max`
    /// times the 0..1 values). The spread's lists are NULL for an object
    /// without any (a stored annotation).
    fn append_intensities(&mut self, intensities: &IndexMap<i32, Intensity>, bit_max: f64) {
        let channels = intensities
            .keys()
            .filter(|channel| **channel >= 0)
            .max()
            .map_or(0, |max| *max as usize + 1);
        let with_spread = intensities.values().any(Intensity::has_spread);
        let columns = if with_spread {
            INTENSITY_COLUMNS.len()
        } else {
            FIRST_SPREAD_COLUMN
        };
        for channel in 0..channels {
            let stats = intensities.get(&(channel as i32)).map(intensity_stats);
            for (column, list) in self.intensities[..columns].iter_mut().enumerate() {
                let value = stats.map(|values| values[column]).filter(|v| v.is_finite());
                list.values().append_option(value.map(|v| v * bit_max));
            }
        }
        for (column, list) in self.intensities.iter_mut().enumerate() {
            list.append(column < columns);
        }
    }

    /// One row of `parent_ids`: parent class -> parent object id.
    fn append_parents(
        &mut self,
        parents: &IndexMap<ObjectClass, ObjectId>,
    ) -> Result<(), InternalErrors> {
        for (class, id) in parents {
            self.parent_ids.keys().append_value(class_key(class));
            self.parent_ids
                .values()
                .append_value(uuid_bytes(id))
                .map_err(|e| InternalErrors::Io(e.to_string()))?;
        }
        self.parent_ids
            .append(true)
            .map_err(|e| InternalErrors::Io(e.to_string()))
    }

    /// One row of `coloc_partner_ids`: partner class -> partner object ids.
    fn append_coloc(
        &mut self,
        colocalized_with: &IndexMap<ObjectClass, Vec<ObjectId>>,
    ) -> Result<(), InternalErrors> {
        for (class, ids) in colocalized_with {
            self.coloc_partner_ids.keys().append_value(class_key(class));
            let partners = self.coloc_partner_ids.values();
            for id in ids {
                partners
                    .values()
                    .append_value(uuid_bytes(id))
                    .map_err(|e| InternalErrors::Io(e.to_string()))?;
            }
            partners.append(true);
        }
        self.coloc_partner_ids
            .append(true)
            .map_err(|e| InternalErrors::Io(e.to_string()))
    }

    /// One row of `measurement_values`: position i holds the value of the
    /// measurement with `object_index` i, NULL where the object has none
    /// (or it isn't a number); NULL without any.
    fn append_measurement_values(&mut self, values: Option<&Vec<(i32, f64)>>) {
        let Some(values) = values else {
            self.measurement_values.append(false);
            return;
        };
        let len = values.iter().map(|(i, _)| *i).max().unwrap_or(0);
        let mut row = vec![None; len.max(0) as usize];
        for (index, value) in values {
            row[*index as usize - 1] = value.is_finite().then_some(*value);
        }
        let list = self.measurement_values.values();
        for value in row {
            list.append_option(value);
        }
        self.measurement_values.append(true);
    }

    /// One row of the z columns: where `object` lies in z - for a plane
    /// zmin = zmax = its plane - and a volume's size. NULL where unknown: the
    /// plane (-1), the z spacing (`pxz`); the volume of a plane. `pxx`, `pxy`:
    /// the pixel size in x and y, in nm.
    fn append_z_extent(&mut self, object: &Object, pxx: f64, pxy: f64, pxz: Option<f64>) {
        let dimensionality = object.dimensionality;
        self.ndim.append_value(dimensionality.ndim());
        let z_min = (object.plane.z >= 0).then_some(object.plane.z as u32);
        let z_max = z_min.map(|z| z + dimensionality.depth() - 1);
        let in_nm = |z: Option<f64>| z.zip(pxz).map(|(z, pxz)| z * pxz);
        let centroid_z = z_min.map(|z| match dimensionality {
            ImageDimensionality::Planar => z as f64,
            ImageDimensionality::Volume { .. } => {
                z as f64 + object.sum_z as f64 / object.area.max(1) as f64
            }
        });
        let box_center_z = z_min.zip(z_max).map(|(min, max)| (min + max) as f64 / 2.0);
        let [zmin_px, zmax_px] = &mut self.bbox_z_px;
        zmin_px.append_option(z_min);
        zmax_px.append_option(z_max);
        self.f64_opt(F64::BboxZminNm, in_nm(z_min.map(f64::from)));
        self.f64_opt(F64::BboxZmaxNm, in_nm(z_max.map(f64::from)));
        self.f64_opt(F64::CentroidZPx, centroid_z);
        self.f64_opt(F64::CentroidZNm, in_nm(centroid_z));
        self.f64_opt(F64::BboxCenterZPx, box_center_z);
        self.f64_opt(F64::BboxCenterZNm, in_nm(box_center_z));
        let voxels = match dimensionality {
            ImageDimensionality::Planar => None,
            ImageDimensionality::Volume { .. } => Some(object.area as u64),
        };
        self.volume_px.append_option(voxels);
        self.f64_opt(
            F64::VolumeNm3,
            voxels
                .zip(pxz)
                .map(|(voxels, pxz)| voxels as f64 * pxx * pxy * pxz),
        );
        // Needs the 3D geometry (a later step).
        self.f64_opt(F64::SurfaceAreaNm2, None);
        self.touches_z_edge.append_null();
    }

    fn len(&self) -> usize {
        self.object_id.len()
    }

    /// The collected rows as one batch, in the `objects` table's column
    /// order (the Appender matches columns by position); leaves the
    /// builders empty for the next batch.
    fn finish(&mut self) -> Result<RecordBatch, InternalErrors> {
        fn arr(builder: &mut dyn ArrayBuilder) -> ArrayRef {
            builder.finish()
        }
        // The DOUBLE columns, each finished once, taken in table order.
        let mut doubles: Vec<Option<ArrayRef>> =
            self.f64.iter_mut().map(|b| Some(arr(b))).collect();
        let mut f = |column: F64| {
            doubles[column as usize]
                .take()
                .expect("each DOUBLE column is in the table once")
        };
        let [xmin_px, ymin_px, xmax_px, ymax_px] = &mut self.bbox_px;
        let [zmin_px, zmax_px] = &mut self.bbox_z_px;
        let mut columns: Vec<(&str, ArrayRef)> = vec![
            ("image_name", arr(&mut self.image_name)),
            ("image_rel_path", arr(&mut self.image_rel_path)),
            ("c_stack", arr(&mut self.c_stack)),
            ("z_stack", arr(&mut self.z_stack)),
            ("t_stack", arr(&mut self.t_stack)),
            ("ndim", arr(&mut self.ndim)),
            ("object_id", arr(&mut self.object_id)),
            ("seg_class_name", arr(&mut self.seg_class_name)),
            ("seg_class_id", arr(&mut self.seg_class_id)),
            ("object_class_name", arr(&mut self.object_class_name)),
            ("object_class_id", arr(&mut self.object_class_id)),
            ("parent_ids", arr(&mut self.parent_ids)),
            ("track_id", arr(&mut self.track_id)),
            ("centroid_x_px", f(F64::CentroidXPx)),
            ("centroid_y_px", f(F64::CentroidYPx)),
            ("centroid_z_px", f(F64::CentroidZPx)),
            ("centroid_x_nm", f(F64::CentroidXNm)),
            ("centroid_y_nm", f(F64::CentroidYNm)),
            ("centroid_z_nm", f(F64::CentroidZNm)),
            ("bbox_center_x_px", f(F64::BboxCenterXPx)),
            ("bbox_center_y_px", f(F64::BboxCenterYPx)),
            ("bbox_center_z_px", f(F64::BboxCenterZPx)),
            ("bbox_center_x_nm", f(F64::BboxCenterXNm)),
            ("bbox_center_y_nm", f(F64::BboxCenterYNm)),
            ("bbox_center_z_nm", f(F64::BboxCenterZNm)),
            ("bbox_xmin_px", arr(xmin_px)),
            ("bbox_ymin_px", arr(ymin_px)),
            ("bbox_zmin_px", arr(zmin_px)),
            ("bbox_xmax_px", arr(xmax_px)),
            ("bbox_ymax_px", arr(ymax_px)),
            ("bbox_zmax_px", arr(zmax_px)),
            ("bbox_xmin_nm", f(F64::BboxXminNm)),
            ("bbox_ymin_nm", f(F64::BboxYminNm)),
            ("bbox_zmin_nm", f(F64::BboxZminNm)),
            ("bbox_xmax_nm", f(F64::BboxXmaxNm)),
            ("bbox_ymax_nm", f(F64::BboxYmaxNm)),
            ("bbox_zmax_nm", f(F64::BboxZmaxNm)),
            ("area_px", arr(&mut self.area_px)),
            ("area_nm2", f(F64::AreaNm2)),
            ("volume_px", arr(&mut self.volume_px)),
            ("volume_nm3", f(F64::VolumeNm3)),
            ("perimeter_px", f(F64::PerimeterPx)),
            ("perimeter_nm", f(F64::PerimeterNm)),
            ("surface_area_nm2", f(F64::SurfaceAreaNm2)),
            ("circularity", f(F64::Circularity)),
            ("solidity", f(F64::Solidity)),
            ("aspect_ratio", f(F64::AspectRatio)),
            ("roundness", f(F64::Roundness)),
            ("compactness", f(F64::Compactness)),
            ("major_axis_px", f(F64::MajorAxisPx)),
            ("minor_axis_px", f(F64::MinorAxisPx)),
            ("major_axis_nm", f(F64::MajorAxisNm)),
            ("minor_axis_nm", f(F64::MinorAxisNm)),
            ("major_axis_angle", f(F64::MajorAxisAngle)),
            ("eccentricity", f(F64::Eccentricity)),
            ("feret_diameter_px", f(F64::FeretDiameterPx)),
            ("min_feret_px", f(F64::MinFeretPx)),
            ("feret_diameter_nm", f(F64::FeretDiameterNm)),
            ("min_feret_nm", f(F64::MinFeretNm)),
            ("touches_edge", arr(&mut self.touches_edge)),
            ("touches_z_edge", arr(&mut self.touches_z_edge)),
            ("pixel_size_x_nm", f(F64::PixelSizeXNm)),
            ("pixel_size_y_nm", f(F64::PixelSizeYNm)),
            ("pixel_size_z_nm", f(F64::PixelSizeZNm)),
            ("image_bit_depth", arr(&mut self.image_bit_depth)),
        ];
        for (name, builder) in INTENSITY_COLUMNS.iter().zip(&mut self.intensities) {
            columns.push((name, arr(builder)));
        }
        columns.push(("coloc_partner_ids", arr(&mut self.coloc_partner_ids)));
        columns.push(("measurement_values", arr(&mut self.measurement_values)));
        columns.push(("mask", arr(&mut self.mask)));
        columns.push(("parent_track_ids", arr(&mut self.parent_track_ids)));
        let batch =
            RecordBatch::try_from_iter(columns).map_err(|e| InternalErrors::Io(e.to_string()))?;
        with_uuid_columns(batch, &["object_id"])
    }
}

// ---------------------------------------------------------------------------
// Arrow columns for the measurements table
// ---------------------------------------------------------------------------

/// What a `measurements` row is about.
#[derive(Clone, Copy)]
enum RowRegion {
    Image,
    Class(ObjectClass),
    Object,
}

/// One `measurements` row but its image, plane and value.
struct MeasurementRow<'a> {
    region: RowRegion,
    object: Option<&'a ObjectId>,
    related: Option<&'a ObjectId>,
    step: &'a str,
    measurement: &'a str,
    unit: &'a str,
    key: &'a str,
}

impl<'a> MeasurementRow<'a> {
    fn of(
        info: &'a MeasurementInfo,
        unit: &'a str,
        region: RowRegion,
        object: Option<&'a ObjectId>,
        related: Option<&'a ObjectId>,
    ) -> Self {
        Self {
            region,
            object,
            related,
            step: &info.step,
            measurement: &info.name,
            unit,
            key: &info.key,
        }
    }
}

/// `measurements` rows of one image plane, column by column, handed to the
/// Appender `ARROW_BATCH_ROWS` at a time (in the table's column order).
struct MeasurementRows<'a> {
    exporter: &'a DuckDbExporter,
    cache: &'a GlobalPipelineCache,
    image_name: String,
    image_rel_path: String,
    /// An object's classes as `objects` has them (names as a JSON list,
    /// ids) - looked up once per object, not once per value.
    object_classes: HashMap<ObjectId, (String, Vec<i32>)>,
    columns: MeasurementColumns,
}

struct MeasurementColumns {
    image_name: StringBuilder,
    image_rel_path: StringBuilder,
    z_stack: Int32Builder,
    t_stack: Int32Builder,
    region: StringBuilder,
    object_id: FixedSizeBinaryBuilder,
    related_object_id: FixedSizeBinaryBuilder,
    object_class_name: StringBuilder,
    object_class_id: ListBuilder<Int32Builder>,
    step: StringBuilder,
    measurement: StringBuilder,
    unit: StringBuilder,
    key: StringBuilder,
    value: Float64Builder,
    position: Float64Builder,
    track_id: UInt64Builder,
}

impl MeasurementColumns {
    fn new() -> Self {
        Self {
            image_name: StringBuilder::new(),
            image_rel_path: StringBuilder::new(),
            z_stack: Int32Builder::new(),
            t_stack: Int32Builder::new(),
            region: StringBuilder::new(),
            object_id: uuid_builder(0),
            related_object_id: uuid_builder(0),
            object_class_name: StringBuilder::new(),
            object_class_id: ListBuilder::new(Int32Builder::new()),
            step: StringBuilder::new(),
            measurement: StringBuilder::new(),
            unit: StringBuilder::new(),
            key: StringBuilder::new(),
            value: Float64Builder::new(),
            position: Float64Builder::new(),
            track_id: UInt64Builder::new(),
        }
    }

    /// The collected rows as one batch; leaves the builders empty.
    fn finish(&mut self) -> Result<RecordBatch, InternalErrors> {
        fn arr(builder: &mut dyn ArrayBuilder) -> ArrayRef {
            builder.finish()
        }
        RecordBatch::try_from_iter([
            ("image_name", arr(&mut self.image_name)),
            ("image_rel_path", arr(&mut self.image_rel_path)),
            ("z_stack", arr(&mut self.z_stack)),
            ("t_stack", arr(&mut self.t_stack)),
            ("region", arr(&mut self.region)),
            ("object_id", arr(&mut self.object_id)),
            ("related_object_id", arr(&mut self.related_object_id)),
            ("object_class_name", arr(&mut self.object_class_name)),
            ("object_class_id", arr(&mut self.object_class_id)),
            ("step", arr(&mut self.step)),
            ("measurement", arr(&mut self.measurement)),
            ("unit", arr(&mut self.unit)),
            ("key", arr(&mut self.key)),
            ("value", arr(&mut self.value)),
            ("position", arr(&mut self.position)),
            ("track_id", arr(&mut self.track_id)),
        ])
        .map_err(|e| InternalErrors::Io(e.to_string()))
        .and_then(|batch| with_uuid_columns(batch, &["object_id", "related_object_id"]))
    }
}

impl<'a> MeasurementRows<'a> {
    fn new(exporter: &'a DuckDbExporter, cache: &'a GlobalPipelineCache) -> Self {
        Self {
            exporter,
            cache,
            image_name: image_display_name(&cache.image_rel_path),
            image_rel_path: cache.image_rel_path.display().to_string(),
            object_classes: HashMap::new(),
            columns: MeasurementColumns::new(),
        }
    }

    /// The classes' names (a JSON list, without `Unset`) and ids, sorted.
    fn classes(&self, classes: &[ObjectClass]) -> (String, Vec<i32>) {
        let mut sorted = classes.to_vec();
        sorted.sort();
        let names: Vec<String> = sorted
            .iter()
            .filter(|c| **c != ObjectClass::Unset)
            .map(|c| self.exporter.class_label(c))
            .collect();
        let ids = sorted
            .iter()
            .filter_map(|c| match c {
                ObjectClass::Global(n) => Some(*n as i32),
                _ => None,
            })
            .collect();
        (json_string_array(&names), ids)
    }

    /// Adds a row; a value that isn't a number (NaN, infinite) is NULL.
    /// `position`: where on a profile the value is (else `None`).
    fn push(&mut self, row: MeasurementRow<'_>, value: f64, position: Option<f64>) {
        // An object's classes are cached by id - referenced here, not copied
        // per row.
        if let (RowRegion::Object, Some(id)) = (&row.region, row.object)
            && !self.object_classes.contains_key(id)
            && let Some(object) = self.cache.object_cache.get(id)
        {
            let classes: Vec<ObjectClass> = object.object_classes().iter().copied().collect();
            let entry = self.classes(&classes);
            self.object_classes.insert(id.clone(), entry);
        }
        let class_row;
        let classes: Option<&(String, Vec<i32>)> = match row.region {
            RowRegion::Image => None,
            RowRegion::Class(class) => {
                class_row = self.classes(&[class]);
                Some(&class_row)
            }
            RowRegion::Object => row.object.and_then(|id| self.object_classes.get(id)),
        };
        let c = &mut self.columns;
        c.image_name.append_value(&self.image_name);
        c.image_rel_path.append_value(&self.image_rel_path);
        c.z_stack.append_value(self.cache.measurements.z_stack);
        c.t_stack.append_value(self.cache.measurements.t_stack);
        c.region.append_value(match row.region {
            RowRegion::Image => "image",
            RowRegion::Class(_) => "class",
            RowRegion::Object => "object",
        });
        append_uuid(&mut c.object_id, row.object);
        append_uuid(&mut c.related_object_id, row.related);
        match classes {
            Some((names, ids)) => {
                c.object_class_name.append_value(names);
                for id in ids {
                    c.object_class_id.values().append_value(*id);
                }
                c.object_class_id.append(true);
            }
            None => {
                c.object_class_name.append_null();
                c.object_class_id.append(false);
            }
        }
        c.step.append_value(row.step);
        c.measurement.append_value(row.measurement);
        c.unit.append_value(row.unit);
        c.key.append_value(row.key);
        c.value.append_option(value.is_finite().then_some(value));
        c.position.append_option(position.filter(|p| p.is_finite()));
        c.track_id.append_null();
    }

    fn flush_if_full(&mut self, app: &mut duckdb::Appender<'_>) -> Result<(), InternalErrors> {
        if self.columns.value.len() >= ARROW_BATCH_ROWS {
            self.flush(app)?;
        }
        Ok(())
    }

    fn flush(&mut self, app: &mut duckdb::Appender<'_>) -> Result<(), InternalErrors> {
        if self.columns.value.len() == 0 {
            return Ok(());
        }
        let batch = self.columns.finish()?;
        app.append_record_batch(batch)
            .map_err(|e| InternalErrors::Io(e.to_string()))
    }
}

// ---------------------------------------------------------------------------
// DDL
// ---------------------------------------------------------------------------

const CREATE_TABLES: &str = "
-- An object is a plane's (ndim 2) or a volume's (ndim 3, an object over
-- several z-planes). Each z column follows its x and y ones; a plane's
-- object has zmin = zmax = its plane. The z values are NULL for an unknown
-- plane, their nm values also for an unknown z spacing.
CREATE TABLE IF NOT EXISTS objects (
    image_name           VARCHAR NOT NULL,
    image_rel_path       VARCHAR NOT NULL,
    c_stack              INTEGER,
    -- The object's plane; NULL for a volume's (bbox_zmin_px..bbox_zmax_px).
    z_stack              INTEGER,
    t_stack              INTEGER,
    -- 2: a plane's object, 3: a volume's.
    ndim                 UTINYINT NOT NULL DEFAULT 2,
    object_id            UUID NOT NULL,
    seg_class_name       VARCHAR,
    seg_class_id         INTEGER,
    object_class_name    VARCHAR,
    object_class_id      INTEGER[],
    -- The objects this one belongs to (Relate Objects): parent class id ->
    -- the parent object (key -1: a parent without a class).
    parent_ids           MAP(INTEGER, UUID),
    track_id             UBIGINT,
    centroid_x_px        DOUBLE,
    centroid_y_px        DOUBLE,
    centroid_z_px        DOUBLE,
    centroid_x_nm        DOUBLE,
    centroid_y_nm        DOUBLE,
    centroid_z_nm        DOUBLE,
    -- The centroid is the mean position of the object's pixels; the
    -- bounding box's centre differs from it for asymmetric shapes.
    bbox_center_x_px     DOUBLE,
    bbox_center_y_px     DOUBLE,
    bbox_center_z_px     DOUBLE,
    bbox_center_x_nm     DOUBLE,
    bbox_center_y_nm     DOUBLE,
    bbox_center_z_nm     DOUBLE,
    bbox_xmin_px         UINTEGER,
    bbox_ymin_px         UINTEGER,
    bbox_zmin_px         UINTEGER,
    bbox_xmax_px         UINTEGER,
    bbox_ymax_px         UINTEGER,
    bbox_zmax_px         UINTEGER,
    bbox_xmin_nm         DOUBLE,
    bbox_ymin_nm         DOUBLE,
    bbox_zmin_nm         DOUBLE,
    bbox_xmax_nm         DOUBLE,
    bbox_ymax_nm         DOUBLE,
    bbox_zmax_nm         DOUBLE,
    -- A plane's object's size and shape - area to min_feret are NULL for a
    -- volume's; its volume and surface NULL for a plane's. More 3D shape
    -- measurements come as measurement steps (`measurement_values`).
    area_px              UBIGINT,
    area_nm2             DOUBLE,
    volume_px            UBIGINT,
    volume_nm3           DOUBLE,
    perimeter_px         DOUBLE,
    perimeter_nm         DOUBLE,
    surface_area_nm2     DOUBLE,
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
    -- Touches the image's border in x or y; a volume's also its first or
    -- last z-plane (touches_z_edge, NULL for a plane's).
    touches_edge         BOOLEAN,
    touches_z_edge       BOOLEAN,
    pixel_size_x_nm      DOUBLE,
    pixel_size_y_nm      DOUBLE,
    -- The distance between two z-planes; NULL if unknown.
    pixel_size_z_nm      DOUBLE,
    image_bit_depth      UTINYINT,
    -- Per-channel intensity statistics, one list each, indexed by channel:
    -- channel c is at position c + 1 (DuckDB lists are 1-based), NULL for a
    -- channel that wasn't measured. In gray values, as in ImageJ/Fiji (0..1:
    -- divided by 2^image_bit_depth - 1). Then how the values spread
    -- (dispersion and robust location): standard deviation, median,
    -- quartiles, median absolute deviation - NULL where not measured (a
    -- stored annotation's).
    intensity_sum_gray             DOUBLE[],
    intensity_mean_gray            DOUBLE[],
    intensity_min_gray             DOUBLE[],
    intensity_max_gray             DOUBLE[],
    intensity_std_gray             DOUBLE[],
    intensity_median_gray          DOUBLE[],
    intensity_lower_quartile_gray  DOUBLE[],
    intensity_upper_quartile_gray  DOUBLE[],
    intensity_mad_gray             DOUBLE[],
    -- Colocalization partners: partner object class id -> the ids of the
    -- partner objects of that class (key -1: partners without a class).
    coloc_partner_ids            MAP(INTEGER, UUID[]),
    -- The object's values of the measurement steps measuring per object,
    -- at their `measurement_info.object_index` - the same values as in
    -- `measurements` (one row each), here for the results to aggregate
    -- without a join. NULL where the object has no value.
    measurement_values           DOUBLE[],
    -- The object's mask relative to its bounding box (row by row, of a
    -- volume plane by plane from bbox_zmin_px): the lengths of its
    -- alternating unset / set runs as LEB128 varints (see evanalyzer_core's
    -- mask_codec.rs). NULL when not stored.
    mask                         BLOB,
    -- The tracks this object's track came from (a tracking step): one for
    -- a daughter of a division, two or more for a merge; NULL for most.
    -- The same on every object of the track. Last, as files written
    -- before it get it added at the end (`open_existing`).
    parent_track_ids             UBIGINT[]
);

CREATE TABLE IF NOT EXISTS images (
    image_name      VARCHAR NOT NULL,
    image_rel_path  VARCHAR NOT NULL PRIMARY KEY,
    successful      BOOLEAN NOT NULL DEFAULT true,
    error_message   VARCHAR,
    disabled        BOOLEAN NOT NULL DEFAULT false,
    disabled_reason VARCHAR,
    width           UINTEGER NOT NULL,
    height          UINTEGER NOT NULL,
    c_stacks        UINTEGER NOT NULL,
    z_stacks        UINTEGER NOT NULL,
    t_stacks        UINTEGER NOT NULL,
    t_stacks_analyzed INTEGER[],
    z_stacks_analyzed INTEGER[],
    frame_interval_s  DOUBLE
);

CREATE TABLE IF NOT EXISTS classes (
    class_id  INTEGER NOT NULL PRIMARY KEY,
    name      VARCHAR NOT NULL,
    color     UINTEGER
);

-- What the measurement steps measured: one row per measurement (its key
-- starts with the step's name, e.g. 'pearson_01/pearson').
CREATE TABLE IF NOT EXISTS measurement_info (
    key           VARCHAR NOT NULL PRIMARY KEY,
    measurement   VARCHAR NOT NULL,
    step          VARCHAR NOT NULL,
    unit          VARCHAR NOT NULL,
    target        VARCHAR NOT NULL,   -- 'Object' or 'Image'
    range_min     DOUBLE,
    range_max     DOUBLE,
    aggregatable  BOOLEAN NOT NULL,
    -- A value per object: its position in `objects.measurement_values`
    -- (1-based). NULL for an image measurement.
    object_index  INTEGER,
    -- Each value names another object (`measurements.related_object_id`).
    related_objects BOOLEAN NOT NULL DEFAULT false,
    -- A profile: several values per object, each at a position
    -- (`measurements.position`); no `object_index` (since format 2).
    profile       BOOLEAN NOT NULL DEFAULT false
);

-- The measurements' values, one row each with what it is about - like
-- `objects`, readable without a join: per image plane, of the whole image
-- (region 'image'), of the pixels of a class ('class', the class in
-- `object_class_*`) or of an object ('object': `object_id`, its classes).
-- NULL `value`: no number (e.g. Pearson of a channel that doesn't vary).
-- `measurement_info` lists each measurement once (the results' pickers).
CREATE TABLE IF NOT EXISTS measurements (
    image_name         VARCHAR NOT NULL,
    image_rel_path     VARCHAR NOT NULL,
    -- The plane; NULL for a value of a whole volume (all its z-planes).
    z_stack            INTEGER,
    t_stack            INTEGER NOT NULL,
    region             VARCHAR NOT NULL,
    object_id          UUID,
    -- The other object a value is about (the nearest neighbour a distance
    -- is to, the parent it is from); NULL for most.
    related_object_id  UUID,
    object_class_name  VARCHAR,
    object_class_id    INTEGER[],
    step               VARCHAR NOT NULL,
    measurement        VARCHAR NOT NULL,
    unit               VARCHAR NOT NULL,
    key                VARCHAR NOT NULL,
    value              DOUBLE,
    -- Where on the object's profile the value is (e.g. a Sholl ring's
    -- radius in pixels); NULL but for a profile measurement (since format 2).
    position           DOUBLE,
    -- The track a value is about (region 'track', e.g. Measure Track
    -- Motion's speeds; its objects: `objects.track_id` in the same image);
    -- NULL for the others (since format 2).
    track_id           UBIGINT
);

-- How the analysis run that wrote this file ended: 'running' from its start
-- until 'finished', 'cancelled' or 'failed' (with `message`). Still
-- 'running' afterwards means it was interrupted (crash, killed process).
-- One row. Files from before this table existed have none.
CREATE TABLE IF NOT EXISTS run (
    -- The format of this file (`RESULTS_SCHEMA_VERSION`); NULL in files
    -- from before it was recorded.
    schema_version      INTEGER,
    status              VARCHAR NOT NULL,
    title               VARCHAR,
    short_description   VARCHAR,
    description         VARCHAR,
    app_version         VARCHAR,
    author_full_name    VARCHAR[],
    author_organization VARCHAR[],
    message             VARCHAR,
    started_at          TIMESTAMP NOT NULL DEFAULT current_timestamp,
    finished_at         TIMESTAMP
);
";

// ---------------------------------------------------------------------------
// Helpers for list columns passed as JSON strings to CAST(? AS T[])
// ---------------------------------------------------------------------------

fn json_string_array(values: &[String]) -> String {
    let mut out = String::from("[");
    for (i, value) in values.iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        write_json_string(&mut out, value);
    }
    out.push(']');
    out
}

// ---------------------------------------------------------------------------
// PipelineResultExporter impl
// ---------------------------------------------------------------------------

impl DuckDbExporter {
    /// The connection, also after another image's export panicked while
    /// holding it: every image runs on its own thread, and one panic must
    /// not fail all others. The panicking export's transaction was rolled
    /// back when it was dropped, so the connection is consistent.
    fn lock_conn(&self) -> std::sync::MutexGuard<'_, ResultsConnection> {
        self.conn.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Each object's values of `cache`'s per-object measurements, as
    /// (`object_index`, value) - a measurement exported for the first time
    /// gets the next free index.
    fn object_measurement_values(
        &self,
        cache: &GlobalPipelineCache,
    ) -> HashMap<ObjectId, Vec<(i32, f64)>> {
        let mut indices = self
            .object_indices
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let mut by_object: HashMap<ObjectId, Vec<(i32, f64)>> = HashMap::new();
        for Measurement { info, values } in &cache.measurements.measurements {
            let MeasurementValues::Objects(values) = values else {
                continue;
            };
            // A profile has several values per object: no single one.
            if info.profile {
                continue;
            }
            let next = indices.len() as i32 + 1;
            let index = *indices.entry(info.key.clone()).or_insert(next);
            for v in values {
                by_object
                    .entry(v.object.clone())
                    .or_default()
                    .push((index, v.value));
            }
        }
        by_object
    }

    /// The `objects` rows for the next up to `ARROW_BATCH_ROWS` objects
    /// taken from `objects`, or `None` once it is exhausted. Needs no
    /// database connection - see `export` for why that matters.
    fn next_object_batch<'a>(
        &self,
        cache: &GlobalPipelineCache,
        objects: &mut std::iter::Peekable<impl Iterator<Item = &'a Object>>,
        measurement_values: &HashMap<ObjectId, Vec<(i32, f64)>>,
    ) -> Result<Option<RecordBatch>, InternalErrors> {
        if objects.peek().is_none() {
            return Ok(None);
        }
        let px = &cache.image_meta.pixel_sizes;
        let nr_of_bits = cache.image_meta.nr_of_bits;
        // Same implausible-bit-depth guard as image_reader.rs's read path -
        // `nr_of_bits` should already have been rejected there before a
        // cache carrying it could exist, but this shouldn't trust that
        // blindly (see `max_gray_value`).
        let Some(bit_max) = max_gray_value(nr_of_bits) else {
            return Err(InternalErrors::Generic(format!(
                "cannot export {}: implausible bit depth {nr_of_bits} (expected 1-32)",
                cache.image_rel_path.display()
            )));
        };
        let px_len = (px.px_size_x * px.px_size_y).sqrt() as f64;
        let pxx = px.px_size_x as f64;
        let pxy = px.px_size_y as f64;
        // The reader's 0: the z spacing is unknown - its nm columns NULL.
        let pxz = (px.px_size_z > 0.0).then_some(px.px_size_z as f64);

        let image_rel = cache.image_rel_path.display().to_string();
        let image_name = image_display_name(&cache.image_rel_path);

        let mut columns =
            ObjectColumns::with_capacity(cache.object_cache.len().min(ARROW_BATCH_ROWS));
        for object in objects.by_ref().take(ARROW_BATCH_ROWS) {
            // A plane's shape measurements; NULL for a volume, whose 3D
            // geometry (volume aside) comes with the 3D commands.
            let planar = object.dimensionality == ImageDimensionality::Planar;
            // get_perimeter()/get_ellipse() are precomputed at object creation (see
            // Object::finalize_geometry), so here they are just field reads - which
            // matters for batches built under the connection lock. We pull each into a local and
            // derive the dependent metrics (circularity/roundness from the perimeter;
            // min_feret/aspect_ratio from the ellipse) to build the row from one read.
            let perimeter_f32 = object.get_perimeter();
            let perimeter = perimeter_f32 as f64;
            let ellipse = object.get_ellipse();
            let centroid = object.get_centroid();
            // Computed from the mask here, at export (not kept on the object).
            let (min_feret, feret) = match planar {
                true => object.feret_diameters(),
                false => (f64::NAN, f64::NAN),
            };
            let aspect_ratio = if ellipse.minor > 0.0 {
                (ellipse.major / ellipse.minor) as f64
            } else {
                1.0
            };

            let centroid_x_px = centroid.0 as f64;
            let centroid_y_px = centroid.1 as f64;
            // circularity and roundness use the identical 4π·area/perimeter² formula,
            // so compute it once from the perimeter local. (get_roundness also guards
            // perimeter == 0, which object.circularity() does not.)
            let roundness = object.get_roundness(perimeter_f32) as f64;

            let c = &mut columns;
            c.image_name.append_value(&image_name);
            c.image_rel_path.append_value(&image_rel);
            c.c_stack.append_value(object.plane.c);
            // A volume lies on several planes: its extent is in the z columns.
            c.z_stack.append_option(planar.then_some(object.plane.z));
            c.t_stack.append_value(object.plane.t);
            c.object_id
                .append_value(uuid_bytes(&object.id))
                .map_err(|e| InternalErrors::Io(e.to_string()))?;
            c.text.clear();
            let _ = write!(c.text, "{}", object.segmentation_class);
            c.seg_class_name.append_value(&c.text);
            c.seg_class_id
                .append_value(object.segmentation_class.0 as i32);
            c.text.clear();
            self.write_class_names(
                &mut c.text,
                object
                    .object_classes()
                    .iter()
                    .filter(|c| **c != ObjectClass::Unset),
            );
            c.object_class_name.append_value(&c.text);
            for class in object.object_classes() {
                if let ObjectClass::Global(n) = class {
                    c.object_class_id.values().append_value(*n as i32);
                }
            }
            c.object_class_id.append(true);
            c.append_parents(object.parents())?;
            c.track_id.append_value(object.track.id.0);
            c.f64(F64::CentroidXPx, centroid_x_px);
            c.f64(F64::CentroidYPx, centroid_y_px);
            c.f64(F64::CentroidXNm, centroid_x_px * pxx);
            c.f64(F64::CentroidYNm, centroid_y_px * pxy);
            let (box_x, box_y) = object.get_bbox_center();
            c.f64(F64::BboxCenterXPx, box_x as f64);
            c.f64(F64::BboxCenterYPx, box_y as f64);
            c.f64(F64::BboxCenterXNm, box_x as f64 * pxx);
            c.f64(F64::BboxCenterYNm, box_y as f64 * pxy);
            for (column, value) in c.bbox_px.iter_mut().zip(object.bbox) {
                column.append_value(value);
            }
            c.f64(F64::BboxXminNm, object.bbox[0] as f64 * pxx);
            c.f64(F64::BboxYminNm, object.bbox[1] as f64 * pxy);
            c.f64(F64::BboxXmaxNm, object.bbox[2] as f64 * pxx);
            c.f64(F64::BboxYmaxNm, object.bbox[3] as f64 * pxy);
            let plane_only = |value: f64| planar.then_some(value);
            c.area_px
                .append_option(planar.then_some(object.area as u64));
            c.f64_opt(F64::AreaNm2, plane_only(object.area as f64 * pxx * pxy));
            c.f64_opt(F64::PerimeterPx, plane_only(perimeter));
            c.f64_opt(F64::PerimeterNm, plane_only(perimeter * px_len));
            c.f64_opt(F64::Circularity, plane_only(roundness));
            c.f64_opt(F64::Solidity, planar.then(|| object.get_solidity() as f64));
            c.f64_opt(F64::AspectRatio, plane_only(aspect_ratio));
            c.f64_opt(F64::Roundness, plane_only(roundness));
            c.f64_opt(
                F64::Compactness,
                plane_only(object.get_compactness(perimeter_f32) as f64),
            );
            c.f64_opt(F64::MajorAxisPx, plane_only(ellipse.major as f64));
            c.f64_opt(F64::MinorAxisPx, plane_only(ellipse.minor as f64));
            c.f64_opt(F64::MajorAxisNm, plane_only(ellipse.major as f64 * px_len));
            c.f64_opt(F64::MinorAxisNm, plane_only(ellipse.minor as f64 * px_len));
            c.f64_opt(F64::MajorAxisAngle, plane_only(ellipse.angle as f64));
            c.f64_opt(F64::Eccentricity, plane_only(ellipse.eccentricity as f64));
            c.f64_opt(F64::FeretDiameterPx, plane_only(feret));
            c.f64_opt(F64::MinFeretPx, plane_only(min_feret));
            c.f64_opt(F64::FeretDiameterNm, plane_only(feret * px_len));
            c.f64_opt(F64::MinFeretNm, plane_only(min_feret * px_len));
            c.touches_edge.append_value(object.touches_edge);
            c.f64(F64::PixelSizeXNm, pxx);
            c.f64(F64::PixelSizeYNm, pxy);
            c.f64_opt(F64::PixelSizeZNm, pxz);
            c.image_bit_depth.append_value(nr_of_bits as u8);
            c.append_intensities(&object.intensities, bit_max);
            c.append_coloc(&object.colocalized_with)?;
            c.append_measurement_values(measurement_values.get(&object.id));
            if self.store_masks {
                c.mask
                    .append_value(crate::storage::mask_codec::encode_mask(&object.mask_data));
            } else {
                c.mask.append_null();
            }
            c.parent_track_ids.append_null();
            c.append_z_extent(object, pxx, pxy, pxz);
        }
        columns.finish().map(Some)
    }
}

impl PipelineResultExporter for DuckDbExporter {
    fn dataset_store(&self) -> Option<&dyn DatasetStore> {
        Some(self)
    }

    /// DuckDB's memory limit: a run uses most of it (`tests/haut`: 1.3-1.6
    /// GB of 8 GB).
    fn reserved_memory_bytes(&self) -> u64 {
        crate::memory::storage::results_memory_limit_bytes()
    }

    fn finish_run(&self, outcome: &Result<(), InternalErrors>) -> Result<(), InternalErrors> {
        let (status, message) = match outcome {
            Ok(()) => ("finished", None),
            Err(InternalErrors::Cancelled) => ("cancelled", None),
            Err(e) => ("failed", Some(e.to_string())),
        };
        let conn = self
            .conn
            .lock()
            .map_err(|_| InternalErrors::Internal("results database lock poisoned".into()))?;
        conn.execute(
            "UPDATE run SET status = ?, message = ?, finished_at = current_timestamp",
            params![status, message],
        )
        .map_err(|e| InternalErrors::Io(e.to_string()))?;
        // The file complete on disk now, not only when it is closed: the
        // results window may keep it open, and the user may copy it.
        if let Err(e) = conn.execute_batch("CHECKPOINT;") {
            log::warn!("Could not checkpoint the results file: {e}");
        }
        Ok(())
    }

    fn export(&self, cache: &GlobalPipelineCache) -> Result<(), InternalErrors> {
        let start = Instant::now();
        let object_count = cache.object_cache.len();
        let mut objects = cache.object_cache.values().peekable();

        // The first batch - for typical images every object - is built
        // before taking the connection lock: building the Arrow columns is
        // about half of the export time and needs no database, so images
        // finishing at the same time build theirs in parallel and only take
        // turns for the append and commit. Further batches of a huge image
        // are built under the lock, as before, so no thread ever holds more
        // than one batch in memory.
        let measurement_values = self.object_measurement_values(cache);
        let first_batch = self.next_object_batch(cache, &mut objects, &measurement_values)?;

        let conn = self.lock_conn();
        // Objects and their colocalization stats must land together or not at
        // all - without a transaction, a failure partway through (disk full,
        // a DuckDB error) could leave one written with no matching rows in
        // the other. `unchecked_transaction` (rather than `transaction`,
        // which needs `&mut Connection`) is safe here because `conn` is
        // already the only handle to this connection, serialized by the
        // `conn` Mutex - nothing else can be mid-transaction on it
        // concurrently. Uncommitted (the `?` early-returns below) rolls back
        // on drop.
        let tx = conn
            .unchecked_transaction()
            .map_err(|e| InternalErrors::Io(e.to_string()))?;

        // --- object rows via the Arrow appender ---
        // Rows are collected into Arrow columns and appended a batch at a
        // time: row by row, a 3.8M-object run took ~47 s instead of ~36 s
        // (and `append_row` can't append the `object_class_id` list at all -
        // see `ObjectColumns`). The Appender flushes to the database when
        // dropped.
        if let Some(batch) = first_batch {
            let mut app = tx
                .appender("objects")
                .map_err(|e| InternalErrors::Io(e.to_string()))?;
            app.append_record_batch(batch)
                .map_err(|e| InternalErrors::Io(e.to_string()))?;
            while let Some(batch) =
                self.next_object_batch(cache, &mut objects, &measurement_values)?
            {
                app.append_record_batch(batch)
                    .map_err(|e| InternalErrors::Io(e.to_string()))?;
            }
        }

        self.write_measurements(&tx, cache)?;

        tx.commit().map_err(|e| InternalErrors::Io(e.to_string()))?;

        log::debug!(
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
        summary: &ImageSummary,
        error: Option<&str>,
    ) -> Result<(), InternalErrors> {
        let conn = self.lock_conn();
        let image_rel = image_rel_path.display().to_string();
        let image_name = image_display_name(image_rel_path);
        let successful = error.is_none();
        // "[0, 1, 2]", cast to INTEGER[] by DuckDB.
        let list = |values: &[i32]| format!("{values:?}");

        conn.execute(
            "INSERT INTO images (image_name, image_rel_path, successful, error_message, width, height, c_stacks, z_stacks, t_stacks, t_stacks_analyzed, z_stacks_analyzed, frame_interval_s, disabled, disabled_reason) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, CAST(? AS INTEGER[]), CAST(? AS INTEGER[]), ?, ?, ?)",
            params![
                image_name,
                image_rel,
                successful,
                error,
                summary.width,
                summary.height,
                summary.nr_c_stacks,
                summary.nr_z_stacks,
                summary.nr_t_stacks,
                list(summary.t_stacks_analyzed),
                list(summary.z_stacks_analyzed),
                summary.frame_interval_s,
                summary.excluded.is_some(),
                summary.excluded
            ],
        )
        .map_err(|e| InternalErrors::Io(e.to_string()))?;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// DatasetStore impl
// ---------------------------------------------------------------------------

impl DatasetStore for DuckDbExporter {
    fn get_objects(&self, filter: &ObjectFilter) -> Result<Vec<StoredObject>, InternalErrors> {
        let class_id = match filter.object_class {
            Some(ObjectClass::Global(n)) => Some(n as i32),
            Some(ObjectClass::Unset | ObjectClass::Local(_)) => return Ok(Vec::new()),
            None => None,
        };
        let io = |e: duckdb::Error| InternalErrors::Io(e.to_string());
        let conn = self.lock_conn();
        let mut stmt = conn
            .prepare(
                "SELECT object_id::VARCHAR, image_rel_path, t_stack, z_stack,
                        centroid_x_px, centroid_y_px,
                        bbox_xmin_px, bbox_ymin_px, bbox_xmax_px, bbox_ymax_px,
                        area_px, object_class_id,
                        list_transform(intensity_mean_gray,
                            v -> v / ((1::UBIGINT << image_bit_depth) - 1)),
                        NULLIF(pixel_size_x_nm, 0), NULLIF(track_id, 0),
                        CASE WHEN ?4 THEN mask END
                 FROM objects
                 WHERE (?1 IS NULL OR image_rel_path = ?1)
                   AND (?2 IS NULL OR t_stack = ?2)
                   AND (?3 IS NULL OR list_contains(object_class_id, ?3))
                 ORDER BY image_rel_path, t_stack, object_id",
            )
            .map_err(io)?;
        let rows = stmt
            .query_map(
                params![
                    filter.image_rel_path,
                    filter.t_stack,
                    class_id,
                    filter.with_masks
                ],
                |row| {
                    Ok((
                        row.get::<_, String>(0)?,
                        StoredObject {
                            id: ObjectId(0),
                            image_rel_path: row.get(1)?,
                            t_stack: row.get::<_, Option<i32>>(2)?.unwrap_or(0),
                            z_stack: row.get::<_, Option<i32>>(3)?.unwrap_or(0),
                            centroid_px: (row.get(4)?, row.get(5)?),
                            pixel_size_nm: row.get(13)?,
                            track_id: row.get(14)?,
                            bbox_px: [row.get(6)?, row.get(7)?, row.get(8)?, row.get(9)?],
                            area_px: row.get(10)?,
                            object_classes: list_values(row.get(11)?)
                                .filter_map(|v| match v {
                                    Value::Int(n) if n >= 0 => Some(ObjectClass::Global(n as u32)),
                                    _ => None,
                                })
                                .collect(),
                            intensity_mean: list_values(row.get(12)?)
                                .map(|v| match v {
                                    Value::Double(x) => Some(x),
                                    _ => None,
                                })
                                .collect(),
                            mask: None,
                        },
                        row.get::<_, Option<Vec<u8>>>(15)?,
                    ))
                },
            )
            .map_err(io)?;
        rows.map(|row| {
            let (id, mut row, mask) = row.map_err(io)?;
            row.id = parse_object_id(&id)?;
            if let Some(mask) = mask {
                let [x0, y0, x1, y1] = row.bbox_px;
                let len =
                    (x1.saturating_sub(x0) as usize + 1) * (y1.saturating_sub(y0) as usize + 1);
                // A volume's mask covers all its planes: it doesn't fit, none.
                row.mask = crate::storage::mask_codec::decode_mask(&mask, len);
            }
            Ok(row)
        })
        .collect()
    }

    fn max_track_id_except(&self, class: ObjectClass) -> Result<u64, InternalErrors> {
        let class_id = match class {
            ObjectClass::Global(n) => n as i32,
            ObjectClass::Unset | ObjectClass::Local(_) => -1, // never stored: excludes nothing
        };
        self.lock_conn()
            .query_row(
                "SELECT COALESCE(MAX(track_id), 0) FROM objects
                 WHERE NOT list_contains(object_class_id, ?)",
                params![class_id],
                |row| row.get(0),
            )
            .map_err(|e| InternalErrors::Io(e.to_string()))
    }

    fn image_rel_paths(&self, class: ObjectClass) -> Result<Vec<String>, InternalErrors> {
        let ObjectClass::Global(class_id) = class else {
            return Ok(Vec::new());
        };
        let io = |e: duckdb::Error| InternalErrors::Io(e.to_string());
        let conn = self.lock_conn();
        let mut stmt = conn
            .prepare(
                "SELECT DISTINCT image_rel_path FROM objects
                 WHERE list_contains(object_class_id, ?) ORDER BY image_rel_path",
            )
            .map_err(io)?;
        stmt.query_map(params![class_id as i32], |row| row.get(0))
            .map_err(io)?
            .collect::<Result<_, _>>()
            .map_err(io)
    }

    fn set_track_ids(&self, tracks: &[TrackAssignment]) -> Result<(), InternalErrors> {
        let io = |e: duckdb::Error| InternalErrors::Io(e.to_string());
        let conn = self.lock_conn();
        // All or nothing; `unchecked_transaction` is safe as in `export`.
        let tx = conn.unchecked_transaction().map_err(io)?;
        // One join instead of an UPDATE per object (there can be millions).
        // The parents apart, as text ("3,7" - the appender takes no lists):
        // updating a list column is slow, and most objects have none.
        tx.execute_batch(
            "CREATE OR REPLACE TEMP TABLE new_track_ids (object_id VARCHAR, track_id UBIGINT);
             CREATE OR REPLACE TEMP TABLE new_parents (object_id VARCHAR, parents VARCHAR);",
        )
        .map_err(io)?;
        {
            let mut ids = tx.appender("new_track_ids").map_err(io)?;
            for track in tracks {
                ids.append_row(params![track.object_id.to_string(), track.track_id])
                    .map_err(io)?;
            }
            let mut parents = tx.appender("new_parents").map_err(io)?;
            for track in tracks.iter().filter(|t| !t.parent_track_ids.is_empty()) {
                let list: Vec<String> = track.parent_track_ids.iter().map(u64::to_string).collect();
                parents
                    .append_row(params![track.object_id.to_string(), list.join(",")])
                    .map_err(io)?;
            }
        }
        tx.execute_batch(
            "UPDATE objects SET track_id = n.track_id FROM new_track_ids n
                 WHERE objects.object_id = CAST(n.object_id AS UUID)",
        )
        .map_err(io)?;
        let had_parents: bool = tx
            .query_row(
                "SELECT EXISTS (SELECT 1 FROM objects WHERE parent_track_ids IS NOT NULL)",
                [],
                |row| row.get(0),
            )
            .map_err(io)?;
        if had_parents {
            tx.execute_batch(
                "UPDATE objects SET parent_track_ids = NULL
                     WHERE parent_track_ids IS NOT NULL
                       AND object_id IN (SELECT CAST(object_id AS UUID) FROM new_track_ids)",
            )
            .map_err(io)?;
        }
        if tracks.iter().any(|t| !t.parent_track_ids.is_empty()) {
            tx.execute_batch(
                "UPDATE objects
                     SET parent_track_ids = CAST(string_split(p.parents, ',') AS UBIGINT[])
                     FROM new_parents p
                     WHERE objects.object_id = CAST(p.object_id AS UUID)",
            )
            .map_err(io)?;
        }
        tx.execute_batch("DROP TABLE new_track_ids; DROP TABLE new_parents;")
            .map_err(io)?;
        tx.commit().map_err(io)
    }

    /// The step's earlier rows go first (all or nothing): a run again may
    /// have other tracks. The values go in through the Arrow appender - a
    /// value per track and measurement can be millions of rows.
    fn write_measurements(
        &self,
        step: &str,
        infos: &[MeasurementInfo],
        values: &[DatasetValue],
    ) -> Result<(), InternalErrors> {
        let io = |e: duckdb::Error| InternalErrors::Io(e.to_string());
        let conn = self.lock_conn();
        // `unchecked_transaction` is safe as in `export`.
        let tx = conn.unchecked_transaction().map_err(io)?;
        tx.execute("DELETE FROM measurements WHERE step = ?", [step])
            .map_err(io)?;
        tx.execute("DELETE FROM measurement_info WHERE step = ?", [step])
            .map_err(io)?;
        {
            let mut info = tx
                .prepare(
                    "INSERT INTO measurement_info \
                     (key, measurement, step, unit, target, range_min, range_max, aggregatable, \
                     object_index, related_objects, profile) \
                     VALUES (?, ?, ?, ?, ?, ?, ?, ?, NULL, ?, ?)",
                )
                .map_err(io)?;
            for i in infos {
                info.execute(params![
                    i.key,
                    i.name,
                    i.step,
                    format!("{:?}", i.unit),
                    format!("{:?}", i.target),
                    i.range.map(|r| r.0),
                    i.range.map(|r| r.1),
                    i.aggregatable,
                    i.related_objects,
                    i.profile
                ])
                .map_err(io)?;
            }
        }
        // The classes' names as the run named them (`classes`), else a label.
        let mut class_names: HashMap<u32, String> = HashMap::new();
        {
            let mut stmt = tx
                .prepare("SELECT class_id, name FROM classes")
                .map_err(io)?;
            let rows = stmt
                .query_map([], |row| {
                    Ok((row.get::<_, i64>(0)?, row.get::<_, String>(1)?))
                })
                .map_err(io)?;
            for row in rows {
                let (id, name) = row.map_err(io)?;
                if let Ok(id) = u32::try_from(id) {
                    class_names.insert(id, name);
                }
            }
        }
        let units: Vec<String> = infos.iter().map(|i| format!("{:?}", i.unit)).collect();
        let mut columns = MeasurementColumns::new();
        let mut app = tx.appender("measurements").map_err(io)?;
        let mut image_names: HashMap<&str, String> = HashMap::new();
        for value in values {
            let info = infos.get(value.info).ok_or_else(|| {
                InternalErrors::InvalidArgument(format!("{step}: no measurement {}", value.info))
            })?;
            let image_name = image_names
                .entry(value.image_rel_path.as_str())
                .or_insert_with(|| image_display_name(Path::new(&value.image_rel_path)));
            let c = &mut columns;
            c.image_name.append_value(image_name.as_str());
            c.image_rel_path.append_value(&value.image_rel_path);
            c.z_stack.append_value(value.z_stack);
            c.t_stack.append_value(value.t_stack);
            append_uuid(&mut c.object_id, None);
            append_uuid(&mut c.related_object_id, None);
            match value.about {
                DatasetValueAbout::Track { track_id, class } => {
                    c.region.append_value("track");
                    let name = match class {
                        ObjectClass::Global(n) => class_names
                            .get(&n)
                            .cloned()
                            .unwrap_or_else(|| self.class_label(&class)),
                        _ => self.class_label(&class),
                    };
                    c.object_class_name.append_value(json_string_array(&[name]));
                    if let ObjectClass::Global(n) = class {
                        c.object_class_id.values().append_value(n as i32);
                    }
                    c.object_class_id.append(true);
                    c.track_id.append_value(track_id);
                }
                DatasetValueAbout::Image => {
                    c.region.append_value("image");
                    c.object_class_name.append_null();
                    c.object_class_id.append(false);
                    c.track_id.append_null();
                }
            }
            c.step.append_value(&info.step);
            c.measurement.append_value(&info.name);
            c.unit.append_value(&units[value.info]);
            c.key.append_value(&info.key);
            c.value
                .append_option(value.value.is_finite().then_some(value.value));
            c.position.append_null();
            if c.value.len() >= ARROW_BATCH_ROWS {
                app.append_record_batch(columns.finish()?).map_err(io)?;
            }
        }
        if columns.value.len() > 0 {
            app.append_record_batch(columns.finish()?).map_err(io)?;
        }
        drop(app);
        tx.commit().map_err(io)
    }
}

/// The items of a DuckDB list value (none for NULL).
fn list_values(value: Value) -> impl Iterator<Item = Value> {
    match value {
        Value::List(items) | Value::Array(items) => items,
        _ => Vec::new(),
    }
    .into_iter()
}

/// An [`ObjectId`] from its UUID text, as `object_id` stores it.
fn parse_object_id(uuid: &str) -> Result<ObjectId, InternalErrors> {
    u128::from_str_radix(&uuid.replace('-', ""), 16)
        .map(ObjectId)
        .map_err(|e| InternalErrors::Io(format!("invalid object id {uuid:?}: {e}")))
}

#[cfg(test)]
mod tests {

    /// DuckDB's memory limit of `conn`, in MB (it reports it rounded, in
    /// GiB or MiB).
    fn memory_limit_mb(conn: &Connection) -> f64 {
        let limit: String = conn
            .query_row("SELECT current_setting('memory_limit')", [], |r| r.get(0))
            .unwrap();
        let mib = if let Some(v) = limit.strip_suffix(" GiB") {
            v.trim().parse::<f64>().unwrap() * 1024.0
        } else {
            limit
                .trim_end_matches(" MiB")
                .trim()
                .parse::<f64>()
                .unwrap()
        };
        mib * 1_048_576.0 / 1_000_000.0
    }

    /// Capped - at most the whole limit, at least a file's floor (other
    /// tests may have files open at the same time: they share it).
    #[test]
    fn a_shared_results_connection_has_its_memory_capped() {
        let dir = tempfile::tempdir().unwrap();
        let conn = shared_connection(&dir.path().join("capped.evadb")).unwrap();
        let mb = memory_limit_mb(&conn);
        let whole = (crate::memory::storage::results_memory_limit_bytes() >> 20) as f64;
        let floor = crate::memory::storage::RESULTS_FILE_MIN_MB as f64;
        assert!(
            mb <= whole + 64.0 && mb >= floor - 64.0,
            "{mb} MB of {whole}"
        );
    }

    /// Two open files share one limit; with one closed, the other has all.
    #[test]
    fn open_results_files_share_one_memory_limit() {
        let dir = tempfile::tempdir().unwrap();
        let open = |name: &str| SharedDatabase {
            anchor: Connection::open(dir.path().join(name)).unwrap(),
            users: 1,
        };
        let total = crate::memory::SystemRam::now().total;
        // What DuckDB reports for a share: it truncates to 0.1 GiB (1997 MB
        // reads back as 1.8 GiB), so compare readouts, not MB with a margin.
        let share = |n| {
            let mb = crate::memory::storage::results_memory_limit_per_file_mb(total, n);
            let probe = Connection::open_in_memory().unwrap();
            probe
                .execute_batch(&format!("SET memory_limit = '{mb}MB'"))
                .unwrap();
            memory_limit_mb(&probe)
        };
        let mut files = HashMap::new();
        files.insert(PathBuf::from("a"), open("a.evadb"));
        files.insert(PathBuf::from("b"), open("b.evadb"));

        share_memory_limit(&files);
        for shared in files.values() {
            assert_eq!(memory_limit_mb(&shared.anchor), share(2));
        }

        files.remove(&PathBuf::from("b"));
        share_memory_limit(&files);
        let a = &files[&PathBuf::from("a")].anchor;
        assert_eq!(memory_limit_mb(a), share(1));
    }

    use super::*;
    use crate::object::{Object, ObjectInit};
    use crate::pipeline::pipeline_measure::ObjectValue;
    use bitvec::prelude::*;

    fn object_in_classes(id: u128, classes: &[u32]) -> Object {
        Object::new(ObjectInit {
            id: ObjectId(id),
            object_class: classes.iter().map(|c| ObjectClass::Global(*c)).collect(),
            bbox: [0, 0, 1, 1],
            mask_data: BitVec::<u64, Lsb0>::repeat(true, 4),
            area: 4,
            ..Default::default()
        })
    }

    fn run_row(exporter: &DuckDbExporter) -> (String, Option<String>, bool) {
        exporter
            .conn
            .lock()
            .unwrap()
            .query_row(
                "SELECT status, message, finished_at IS NOT NULL FROM run",
                [],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
            )
            .unwrap()
    }

    #[test]
    fn a_run_is_running_until_its_outcome_is_recorded() {
        let dir = tempfile::tempdir().unwrap();
        for (outcome, status, message) in [
            (Ok(()), "finished", None),
            (Err(InternalErrors::Cancelled), "cancelled", None),
            (
                Err(InternalErrors::Internal("disk full".into())),
                "failed",
                Some("disk full"),
            ),
        ] {
            let path = dir.path().join(format!("{status}.evadb"));
            let exporter = DuckDbExporter::new(
                &path,
                HashMap::new(),
                &MetaData {
                    name: "Experiment name".into(),
                    short_description: "Short description".into(),
                    description: "Long description".into(),
                    authors: vec![],
                    creation_time: chrono::DateTime::from_timestamp_nanos(1791318919124935845),
                    version: None,
                    category: "Test category".into(),
                    tags: vec![],
                    app_version: "v1.0.0".into(),
                },
            )
            .unwrap();
            assert_eq!(run_row(&exporter), ("running".into(), None, false));

            exporter.finish_run(&outcome).unwrap();
            let (got_status, got_message, finished) = run_row(&exporter);
            assert_eq!(got_status, status);
            assert_eq!(
                got_message.is_some_and(|m| m.contains("disk full")),
                message.is_some()
            );
            assert!(finished);
        }
    }

    fn is_open(path: &Path) -> bool {
        CONNECTION_CACHE.lock().unwrap().contains_key(path)
    }

    /// A results file stays open while a connection to it is alive, and is
    /// closed - its memory freed - with the last one.
    #[test]
    fn a_results_file_is_closed_with_its_last_connection() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("results.evadb");
        let first = shared_connection(&path).unwrap();
        let second = first.try_clone().unwrap();
        first.execute_batch("CREATE TABLE t (v INTEGER);").unwrap();

        drop(first);
        assert!(is_open(&path), "the second connection still uses it");
        second.execute_batch("INSERT INTO t VALUES (1);").unwrap();

        drop(second);
        assert!(!is_open(&path));
        // Opened again later, with what was written.
        let again = shared_connection(&path).unwrap();
        let n: i64 = again
            .query_row("SELECT count(*) FROM t", [], |r| r.get(0))
            .unwrap();
        assert_eq!(n, 1);
    }

    /// The end of a run folds the write-ahead log into the file - complete
    /// on disk while the results stay open (the results window).
    #[test]
    fn a_finished_run_leaves_no_write_ahead_log() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("results.evadb");
        let exporter = DuckDbExporter::new(&path, HashMap::new(), &MetaData::default()).unwrap();
        let wal = dir.path().join("results.evadb.wal");
        let wal_len = || std::fs::metadata(&wal).map(|m| m.len()).unwrap_or(0);
        assert!(wal_len() > 0, "the run's rows are in the log first");

        exporter.finish_run(&Ok(())).unwrap();

        assert_eq!(wal_len(), 0);
        assert!(is_open(&path), "still open");
    }

    #[test]
    fn export_writes_object_classes_as_an_integer_list() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("results.evadb");
        let exporter = DuckDbExporter::new(
            &path,
            HashMap::new(),
            &MetaData {
                name: "Experiment name 02".into(),
                short_description: "Short description".into(),
                description: "Long description".into(),
                authors: vec![],
                creation_time: chrono::DateTime::from_timestamp_nanos(1791318919124935845),
                version: None,
                category: "Test category".into(),
                tags: vec![],
                app_version: "v1.0.0".into(),
            },
        )
        .unwrap();
        let mut cache = GlobalPipelineCache::default();
        cache.image_rel_path = PathBuf::from("a.tif");
        cache.image_meta.nr_of_bits = 8;
        cache
            .object_cache
            .insert(ObjectId(1), object_in_classes(1, &[4]));
        cache
            .object_cache
            .insert(ObjectId(2), object_in_classes(2, &[1, 3]));
        cache
            .object_cache
            .insert(ObjectId(3), object_in_classes(3, &[]));

        exporter.export(&cache).unwrap();

        let conn = shared_connection(&path).unwrap();
        let column_type: String = conn
            .query_row(
                "SELECT data_type FROM information_schema.columns \
                 WHERE table_name = 'objects' AND column_name = 'object_class_id'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(column_type, "INTEGER[]");
        let classes: Vec<String> = conn
            .prepare(
                "SELECT CAST(list_sort(object_class_id) AS VARCHAR) FROM objects \
                 ORDER BY len(object_class_id), list_sort(object_class_id)",
            )
            .unwrap()
            .query_map([], |row| row.get(0))
            .unwrap()
            .collect::<Result<_, _>>()
            .unwrap();
        assert_eq!(classes, ["[]", "[4]", "[1, 3]"]);
    }

    fn cache_with_objects(image: &str, count: u128) -> GlobalPipelineCache {
        let mut cache = GlobalPipelineCache::default();
        cache.image_rel_path = PathBuf::from(image);
        cache.image_meta.nr_of_bits = 8;
        for id in 1..=count {
            cache
                .object_cache
                .insert(ObjectId(id), object_in_classes(id, &[1]));
        }
        cache
    }

    fn object_count(path: &Path) -> i64 {
        shared_connection(path)
            .unwrap()
            .query_row("SELECT count(*) FROM objects", [], |row| row.get(0))
            .unwrap()
    }

    /// Class names as JSON: the same text as the `format!`/`replace` it
    /// replaced, also with quotes, backslashes and non-ASCII letters.
    #[test]
    fn class_names_are_written_as_the_same_json() {
        let before = |values: &[String]| {
            let items: Vec<String> = values
                .iter()
                .map(|s| format!("\"{}\"", s.replace('\\', "\\\\").replace('"', "\\\"")))
                .collect();
            format!("[{}]", items.join(","))
        };
        for names in [
            vec![],
            vec!["Nucleus".to_string()],
            vec![
                "Kupffer \"enlarged\"".to_string(),
                "a\\b".to_string(),
                "Zelle ä µ".to_string(),
            ],
            vec!["\\\"".to_string(), String::new()],
        ] {
            assert_eq!(json_string_array(&names), before(&names), "{names:?}");
        }
    }

    #[test]
    fn export_writes_every_object_of_an_image_larger_than_one_batch() {
        // The first batch is built before the connection lock, the rest
        // under it - none may be lost or written twice at the seam.
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("results.evadb");
        let exporter = DuckDbExporter::new(
            &path,
            HashMap::new(),
            &MetaData {
                name: "Experiment name".into(),
                short_description: "Short description".into(),
                description: "Long description".into(),
                authors: vec![],
                creation_time: chrono::DateTime::from_timestamp_nanos(1791318919124935845),
                version: None,
                category: "Test category".into(),
                tags: vec![],
                app_version: "v1.0.0".into(),
            },
        )
        .unwrap();
        let count = ARROW_BATCH_ROWS as u128 * 2 + 1;

        exporter
            .export(&cache_with_objects("a.tif", count))
            .unwrap();

        assert_eq!(object_count(&path), count as i64);
        let distinct: i64 = shared_connection(&path)
            .unwrap()
            .query_row("SELECT count(DISTINCT object_id) FROM objects", [], |row| {
                row.get(0)
            })
            .unwrap();
        assert_eq!(distinct, count as i64);
    }

    #[test]
    fn concurrent_exports_of_several_images_write_every_object() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("results.evadb");
        let exporter = DuckDbExporter::new(
            &path,
            HashMap::new(),
            &MetaData {
                name: "Experiment name".into(),
                short_description: "Short description".into(),
                description: "Long description".into(),
                authors: vec![],
                creation_time: chrono::DateTime::from_timestamp_nanos(1791318919124935845),
                version: None,
                category: "Test category".into(),
                tags: vec![],
                app_version: "v1.0.0".into(),
            },
        )
        .unwrap();
        let caches: Vec<_> = (0..8)
            .map(|i| cache_with_objects(&format!("{i}.tif"), 500))
            .collect();

        std::thread::scope(|s| {
            for cache in &caches {
                let exporter = &exporter;
                s.spawn(move || exporter.export(cache).unwrap());
            }
        });

        assert_eq!(object_count(&path), 8 * 500);
    }

    /// An image analysed on `t_stacks` x `z_stacks`.
    fn summary<'a>(t_stacks: &'a [i32], z_stacks: &'a [i32]) -> ImageSummary<'a> {
        ImageSummary {
            width: 1,
            height: 1,
            nr_c_stacks: 1,
            nr_z_stacks: 30,
            nr_t_stacks: 3,
            t_stacks_analyzed: t_stacks,
            z_stacks_analyzed: z_stacks,
            frame_interval_s: None,
            excluded: None,
        }
    }

    /// An image Exclude Image left out: disabled, with why - the others
    /// not.
    #[test]
    fn an_excluded_image_is_written_disabled_with_its_reason() {
        let dir = tempfile::tempdir().unwrap();
        let exporter = DuckDbExporter::new(
            dir.path().join("r.evadb"),
            HashMap::new(),
            &MetaData::default(),
        )
        .unwrap();
        let reason = "Focus score (channel 0) of quality_01 = 0.09, below 0.15";
        exporter
            .finalize_image(
                Path::new("blurry.tif"),
                &ImageSummary {
                    excluded: Some(reason),
                    ..summary(&[0], &[0])
                },
                None,
            )
            .unwrap();
        exporter
            .finalize_image(Path::new("sharp.tif"), &summary(&[0], &[0]), None)
            .unwrap();
        let conn = exporter.lock_conn();
        let rows: Vec<(String, bool, Option<String>)> = conn
            .prepare("SELECT image_rel_path, disabled, disabled_reason FROM images ORDER BY 1")
            .unwrap()
            .query_map([], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)))
            .unwrap()
            .collect::<Result<_, _>>()
            .unwrap();
        assert_eq!(
            rows,
            [
                ("blurry.tif".into(), true, Some(reason.into())),
                ("sharp.tif".into(), false, None),
            ]
        );
    }

    /// A profile (several values per object, each at a position) is rows
    /// with their position, without an `object_index` - the objects'
    /// `measurement_values` hold only the ordinary measurements.
    #[test]
    fn a_profile_is_rows_with_positions_not_an_object_value() {
        use crate::pipeline::pipeline_measure::PipelineMeasurements;
        use evanalyzer_cfg::core_types::{MeasurementInfo, MeasurementTarget, MeasurementUnit};
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("results.evadb");
        let exporter = DuckDbExporter::new(&path, HashMap::new(), &MetaData::default()).unwrap();
        let info = |key: &str, profile: bool| MeasurementInfo {
            key: key.into(),
            name: key.into(),
            step: "sholl_01".into(),
            unit: MeasurementUnit::Count,
            target: MeasurementTarget::Object,
            range: None,
            aggregatable: true,
            related_objects: false,
            profile,
        };
        let value = |value: f64, position: Option<f64>| ObjectValue {
            object: ObjectId(1),
            value,
            related: None,
            position,
        };
        let mut cache = cache_with_objects("a.tif", 1);
        cache.measurements = PipelineMeasurements {
            measurements: vec![
                Measurement {
                    info: info("sholl_01/intersections", true),
                    values: MeasurementValues::Objects(vec![
                        value(3.0, Some(1000.0)),
                        value(5.0, Some(2000.0)),
                        value(1.0, Some(3000.0)),
                    ]),
                },
                Measurement {
                    info: info("sholl_01/max_intersections", false),
                    values: MeasurementValues::Objects(vec![value(5.0, None)]),
                },
            ],
            z_stack: 0,
            t_stack: 0,
            excluded: None,
        };
        exporter.export(&cache).unwrap();

        let conn = exporter.conn.lock().unwrap();
        let infos: Vec<(String, bool, Option<i32>)> = conn
            .prepare("SELECT key, profile, object_index FROM measurement_info ORDER BY key")
            .unwrap()
            .query_map([], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)))
            .unwrap()
            .collect::<Result<_, _>>()
            .unwrap();
        assert_eq!(
            infos,
            [
                ("sholl_01/intersections".into(), true, None),
                ("sholl_01/max_intersections".into(), false, Some(1)),
            ]
        );
        let rows: Vec<(String, f64, Option<f64>)> = conn
            .prepare("SELECT key, value, position FROM measurements ORDER BY key, position")
            .unwrap()
            .query_map([], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)))
            .unwrap()
            .collect::<Result<_, _>>()
            .unwrap();
        assert_eq!(
            rows,
            [
                ("sholl_01/intersections".into(), 3.0, Some(1000.0)),
                ("sholl_01/intersections".into(), 5.0, Some(2000.0)),
                ("sholl_01/intersections".into(), 1.0, Some(3000.0)),
                ("sholl_01/max_intersections".into(), 5.0, None),
            ]
        );
        let values: String = conn
            .query_row("SELECT measurement_values::VARCHAR FROM objects", [], |r| {
                r.get(0)
            })
            .unwrap();
        assert_eq!(values, "[5.0]");
    }

    /// A results file of format 1 is moved to format 2 when opened to add
    /// to: it gets the profile columns (and can't hold a profile, so
    /// their defaults are exact).
    #[test]
    fn a_format_1_file_is_migrated_when_opened_to_add_to() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("results.evadb");
        drop(test_exporter(&path));
        {
            // What format 1 wrote: the tables without the profile columns.
            let conn = shared_connection(&path).unwrap();
            conn.execute_batch(
                "ALTER TABLE measurement_info DROP COLUMN profile;
                 ALTER TABLE measurements DROP COLUMN position;
                 ALTER TABLE measurements DROP COLUMN track_id;
                 UPDATE run SET schema_version = 1;",
            )
            .unwrap();
        }
        let exporter = DuckDbExporter::open_existing(&path).unwrap();
        let conn = exporter.lock_conn();
        assert_eq!(results_schema_version(&conn).unwrap(), 2);
        let columns: i64 = conn
            .query_row(
                "SELECT count(*) FROM information_schema.columns WHERE \
                 (table_name = 'measurement_info' AND column_name = 'profile') OR \
                 (table_name = 'measurements' AND column_name = 'position') OR \
                 (table_name = 'measurements' AND column_name = 'track_id')",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(columns, 3);
    }

    /// A format 2 file from before `measurements.track_id` (format 2 wasn't
    /// released then) gets it when opened to add to.
    #[test]
    fn a_format_2_file_without_track_ids_gets_the_column() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("results.evadb");
        drop(test_exporter(&path));
        shared_connection(&path)
            .unwrap()
            .execute_batch("ALTER TABLE measurements DROP COLUMN track_id;")
            .unwrap();
        let exporter = DuckDbExporter::open_existing(&path).unwrap();
        let has: bool = exporter
            .lock_conn()
            .query_row(
                "SELECT count(*) > 0 FROM information_schema.columns \
                 WHERE table_name = 'measurements' AND column_name = 'track_id'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert!(has);
    }

    /// A dataset step's values: a row per track (with its track id and
    /// class) or per image time point, its measurements' descriptions once -
    /// and a second run replaces the first's.
    #[test]
    fn dataset_values_are_rows_per_track_and_replace_a_run_before() {
        use crate::storage::{DatasetValue, DatasetValueAbout};
        use evanalyzer_cfg::core_types::{MeasurementTarget, MeasurementUnit};
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("results.evadb");
        let names = HashMap::from([(ObjectClass::Global(1), ("Sperm".to_string(), 0))]);
        let exporter = DuckDbExporter::new(&path, names, &MetaData::default()).unwrap();
        let info = |id: &str, target| MeasurementInfo {
            key: format!("motion_01/{id}"),
            name: id.to_uppercase(),
            step: "motion_01".into(),
            unit: MeasurementUnit::PixelsPerFrame,
            target,
            range: None,
            aggregatable: true,
            related_objects: false,
            profile: false,
        };
        let infos = [
            info("vcl", MeasurementTarget::Track),
            info("drift_x", MeasurementTarget::Image),
        ];
        let track = |track_id, value| DatasetValue {
            image_rel_path: "well_A1/img.tif".into(),
            z_stack: 0,
            t_stack: 3,
            about: DatasetValueAbout::Track {
                track_id,
                class: ObjectClass::Global(1),
            },
            info: 0,
            value,
        };
        let drift = DatasetValue {
            t_stack: 5,
            about: DatasetValueAbout::Image,
            info: 1,
            value: 2.5,
            ..track(0, 0.0)
        };
        let store = exporter.dataset_store().unwrap();
        store
            .write_measurements(
                "motion_01",
                &infos,
                &[track(7, 4.0), track(8, f64::NAN), drift.clone()],
            )
            .unwrap();
        // Again, with other tracks: only these remain.
        store
            .write_measurements("motion_01", &infos, &[track(9, 1.5), drift])
            .unwrap();
        let conn = exporter.lock_conn();
        let mut stmt = conn
            .prepare(
                "SELECT image_name, region, track_id, object_class_name, object_class_id::VARCHAR AS object_class_id, t_stack, \
                 measurement, unit, value FROM measurements ORDER BY region, track_id",
            )
            .unwrap();
        let rows: Vec<String> = stmt
            .query_map([], |r| {
                Ok(format!(
                    "{} {} {:?} {:?} {:?} {} {} {} {:?}",
                    r.get::<_, String>(0)?,
                    r.get::<_, String>(1)?,
                    r.get::<_, Option<u64>>(2)?,
                    r.get::<_, Option<String>>(3)?,
                    r.get::<_, Option<String>>(4)?,
                    r.get::<_, i32>(5)?,
                    r.get::<_, String>(6)?,
                    r.get::<_, String>(7)?,
                    r.get::<_, Option<f64>>(8)?,
                ))
            })
            .unwrap()
            .collect::<Result<_, _>>()
            .unwrap();
        assert_eq!(
            rows,
            [
                "img.tif image None None None 5 DRIFT_X PixelsPerFrame Some(2.5)",
                "img.tif track Some(9) Some(\"[\\\"Sperm\\\"]\") Some(\"[1]\") 3 VCL PixelsPerFrame Some(1.5)",
            ]
        );
        let targets: Vec<String> = conn
            .prepare("SELECT target FROM measurement_info ORDER BY key")
            .unwrap()
            .query_map([], |r| r.get(0))
            .unwrap()
            .collect::<Result<_, _>>()
            .unwrap();
        assert_eq!(targets, ["Image", "Track"]);
    }

    /// Each image's analysed time points and z-planes are kept - also ones
    /// without objects.
    /// Each value is a row with what it is about - readable without a join;
    /// a measurement's description is written once; a value that isn't a
    /// number is NULL.
    #[test]
    fn measurements_are_rows_readable_without_a_join() {
        use crate::pipeline::pipeline_measure::PipelineMeasurements;
        use evanalyzer_cfg::core_types::{MeasurementInfo, MeasurementTarget, MeasurementUnit};
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("results.evadb");
        let names = HashMap::from([(ObjectClass::Global(1), ("Nucleus".to_string(), 0))]);
        let exporter = DuckDbExporter::new(&path, names, &MetaData::default()).unwrap();
        let info = MeasurementInfo {
            key: "pearson_01/pearson/ch0_ch1".into(),
            name: "Pearson (channel 0 × channel 1)".into(),
            step: "pearson_01".into(),
            unit: MeasurementUnit::None,
            target: MeasurementTarget::Image,
            range: Some((-1.0, 1.0)),
            aggregatable: true,
            related_objects: false,
            profile: false,
        };
        for (t, value) in [(0, 0.5), (1, f64::NAN)] {
            // Object 1, in class 1 ("Nucleus").
            let mut cache = cache_with_objects("plate/a.tif", 1);
            cache.measurements = PipelineMeasurements {
                measurements: vec![
                    Measurement {
                        info: info.clone(),
                        values: MeasurementValues::Image(vec![
                            (MeasurementRegion::Image, value),
                            (MeasurementRegion::Class(ObjectClass::Global(1)), 0.75),
                        ]),
                    },
                    Measurement {
                        info: MeasurementInfo {
                            key: "pearson_01/pearson/ch0_ch1/objects".into(),
                            target: MeasurementTarget::Object,
                            ..info.clone()
                        },
                        values: MeasurementValues::Objects(vec![ObjectValue {
                            object: ObjectId(1),
                            value: 0.25,
                            related: Some(ObjectId(7)),
                            position: None,
                        }]),
                    },
                ],
                z_stack: 0,
                t_stack: t,
                excluded: None,
            };
            exporter.export(&cache).unwrap();
        }

        let conn = exporter.conn.lock().unwrap();
        let described: i64 = conn
            .query_row("SELECT count(*) FROM measurement_info", [], |row| {
                row.get(0)
            })
            .unwrap();
        assert_eq!(described, 2, "each measurement described once");
        let related: Vec<Option<String>> = conn
            .prepare("SELECT related_object_id::VARCHAR FROM measurements ORDER BY t_stack, region")
            .unwrap()
            .query_map([], |row| row.get(0))
            .unwrap()
            .map(Result::unwrap)
            .collect();
        let seven = Some(ObjectId(7).to_string());
        assert_eq!(related, [None, None, seven.clone(), None, None, seven]);
        let mut stmt = conn
            .prepare(
                "SELECT image_name, t_stack, region, object_id IS NOT NULL, object_class_name, \
                 object_class_id::VARCHAR, step, measurement, unit, value \
                 FROM measurements ORDER BY t_stack, region",
            )
            .unwrap();
        type Row = (
            String,
            i32,
            String,
            bool,
            Option<String>,
            Option<String>,
            String,
            String,
            String,
            Option<f64>,
        );
        let rows: Vec<Row> = stmt
            .query_map([], |row| {
                Ok((
                    row.get(0)?,
                    row.get(1)?,
                    row.get(2)?,
                    row.get(3)?,
                    row.get(4)?,
                    row.get(5)?,
                    row.get(6)?,
                    row.get(7)?,
                    row.get(8)?,
                    row.get(9)?,
                ))
            })
            .unwrap()
            .map(Result::unwrap)
            .collect();
        let row = |t, region: &str, object, class: bool, value| -> Row {
            (
                "a.tif".into(),
                t,
                region.into(),
                object,
                class.then(|| "[\"Nucleus\"]".into()),
                class.then(|| "[1]".into()),
                "pearson_01".into(),
                "Pearson (channel 0 × channel 1)".into(),
                "None".into(),
                value,
            )
        };
        assert_eq!(
            rows,
            [
                row(0, "class", false, true, Some(0.75)),
                row(0, "image", false, false, Some(0.5)),
                row(0, "object", true, true, Some(0.25)),
                row(1, "class", false, true, Some(0.75)),
                row(1, "image", false, false, None),
                row(1, "object", true, true, Some(0.25)),
            ]
        );
    }

    /// Each per-object measurement gets a position in `measurement_values`
    /// (`measurement_info.object_index`), the same in every image; an
    /// object without a value has NULL there, an image measurement none.
    #[test]
    fn object_values_are_also_kept_on_the_object_by_index() {
        use crate::pipeline::pipeline_measure::PipelineMeasurements;
        use evanalyzer_cfg::core_types::{MeasurementInfo, MeasurementTarget, MeasurementUnit};
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("results.evadb");
        let exporter = DuckDbExporter::new(&path, HashMap::new(), &MetaData::default()).unwrap();
        let info = |key: &str, target| MeasurementInfo {
            key: key.into(),
            name: key.into(),
            step: "s_01".into(),
            unit: MeasurementUnit::None,
            target,
            range: None,
            aggregatable: true,
            related_objects: false,
            profile: false,
        };
        let objects = |key: &str, values: Vec<(u128, f64)>| Measurement {
            info: info(key, MeasurementTarget::Object),
            values: MeasurementValues::Objects(
                values
                    .into_iter()
                    .map(|(id, v)| (ObjectId(id), v).into())
                    .collect(),
            ),
        };
        // Image a: "a" first, then "b"; image b: only "b" - "b" keeps index 2.
        let mut a = cache_with_objects("a.tif", 2);
        a.measurements = PipelineMeasurements {
            measurements: vec![
                Measurement {
                    info: info("s_01/image", MeasurementTarget::Image),
                    values: MeasurementValues::Image(vec![(MeasurementRegion::Image, 1.0)]),
                },
                objects("s_01/a", vec![(1, 0.5)]),
                objects("s_01/b", vec![(1, 1.5), (2, f64::NAN)]),
            ],
            ..Default::default()
        };
        exporter.export(&a).unwrap();
        let mut b = GlobalPipelineCache::default();
        b.image_rel_path = PathBuf::from("b.tif");
        b.image_meta.nr_of_bits = 8;
        b.object_cache
            .insert(ObjectId(3), object_in_classes(3, &[1]));
        b.measurements = PipelineMeasurements {
            measurements: vec![objects("s_01/b", vec![(3, 7.0)])],
            ..Default::default()
        };
        exporter.export(&b).unwrap();

        let conn = exporter.conn.lock().unwrap();
        let strings = |sql: &str| -> Vec<String> {
            conn.prepare(sql)
                .unwrap()
                .query_map([], |row| row.get(0))
                .unwrap()
                .map(Result::unwrap)
                .collect()
        };
        assert_eq!(
            strings(
                "SELECT key || '=' || COALESCE(object_index::VARCHAR, '-') \
                 FROM measurement_info ORDER BY key"
            ),
            ["s_01/a=1", "s_01/b=2", "s_01/image=-"]
        );
        let ids = [ObjectId(1), ObjectId(2), ObjectId(3)].map(|id| id.to_string());
        let values = strings(&format!(
            "SELECT COALESCE(measurement_values::VARCHAR, '-') FROM objects \
             ORDER BY CASE object_id::VARCHAR WHEN '{}' THEN 1 WHEN '{}' THEN 2 ELSE 3 END",
            ids[0], ids[1]
        ));
        assert_eq!(values, ["[0.5, 1.5]", "[NULL, NULL]", "[NULL, 7.0]"]);
    }

    /// More object values than one Arrow batch holds: every one is written.
    #[test]
    fn every_object_value_is_written_also_beyond_one_batch() {
        use crate::pipeline::pipeline_measure::PipelineMeasurements;
        use evanalyzer_cfg::core_types::{MeasurementInfo, MeasurementTarget, MeasurementUnit};
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("results.evadb");
        let exporter = DuckDbExporter::new(&path, HashMap::new(), &MetaData::default()).unwrap();
        let count = ARROW_BATCH_ROWS as u128 + 3;
        let mut cache = cache_with_objects("a.tif", count);
        cache.measurements = PipelineMeasurements {
            measurements: vec![Measurement {
                info: MeasurementInfo {
                    key: "n_01/count".into(),
                    name: "Neighbours".into(),
                    step: "n_01".into(),
                    unit: MeasurementUnit::Count,
                    target: MeasurementTarget::Object,
                    range: None,
                    aggregatable: true,
                    related_objects: false,
                    profile: false,
                },
                values: MeasurementValues::Objects(
                    (1..=count)
                        .map(|id| (ObjectId(id), id as f64).into())
                        .collect(),
                ),
            }],
            ..Default::default()
        };
        exporter.export(&cache).unwrap();
        let conn = exporter.conn.lock().unwrap();
        let (rows, sum): (i64, f64) = conn
            .query_row(
                "SELECT count(*), sum(value) FROM measurements WHERE region = 'object'",
                [],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .unwrap();
        assert_eq!(rows as u128, count);
        assert_eq!(sum, (count * (count + 1) / 2) as f64);
    }

    #[test]
    fn the_analysed_planes_of_each_image_are_kept() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("results.evadb");
        let exporter = DuckDbExporter::new(&path, HashMap::new(), &MetaData::default()).unwrap();
        exporter
            .finalize_image(Path::new("a.tif"), &summary(&[0, 1, 2], &[0]), None)
            .unwrap();
        exporter
            .finalize_image(
                Path::new("b.tif"),
                &ImageSummary {
                    frame_interval_s: Some(30.0),
                    ..summary(&[2], &[4, 5])
                },
                None,
            )
            .unwrap();
        drop(exporter);

        let conn = shared_connection(&path).unwrap();
        let mut statement = conn
            .prepare(
                "SELECT image_rel_path, CAST(t_stacks_analyzed AS VARCHAR), \
                 CAST(z_stacks_analyzed AS VARCHAR), frame_interval_s FROM images ORDER BY 1",
            )
            .unwrap();
        let rows: Vec<(String, String, String, Option<f64>)> = statement
            .query_map([], |row| {
                Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?))
            })
            .unwrap()
            .collect::<Result<_, _>>()
            .unwrap();
        let text = |s: &str| s.to_string();
        assert_eq!(
            rows,
            [
                (text("a.tif"), text("[0, 1, 2]"), text("[0]"), None),
                (text("b.tif"), text("[2]"), text("[4, 5]"), Some(30.0)),
            ]
        );
    }

    #[test]
    fn export_still_works_after_another_export_panicked_holding_the_connection() {
        // Every image exports from its own thread: one panicking while it
        // holds the connection must not fail the export of all others.
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("results.evadb");
        let exporter = DuckDbExporter::new(
            &path,
            HashMap::new(),
            &MetaData {
                name: "Experiment name".into(),
                short_description: "Short description".into(),
                description: "Long description".into(),
                authors: vec![],
                creation_time: chrono::DateTime::from_timestamp_nanos(1791318919124935845),
                version: None,
                category: "Test category".into(),
                tags: vec![],
                app_version: "v1.0.0".into(),
            },
        )
        .unwrap();
        let panicked = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _conn = exporter.conn.lock().unwrap();
            panic!("simulated export panic while holding the connection");
        }));
        assert!(panicked.is_err());
        assert!(exporter.conn.is_poisoned());

        exporter.export(&cache_with_objects("a.tif", 3)).unwrap();
        exporter
            .finalize_image(Path::new("a.tif"), &summary(&[0], &[0]), None)
            .unwrap();

        assert_eq!(object_count(&path), 3);
    }

    #[test]
    fn export_writes_intensities_per_channel_and_coloc_partners_as_a_map() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("results.evadb");
        let exporter = DuckDbExporter::new(
            &path,
            HashMap::new(),
            &MetaData {
                name: "Experiment name".into(),
                short_description: "Short description".into(),
                description: "Long description".into(),
                authors: vec![],
                creation_time: chrono::DateTime::from_timestamp_nanos(1791318919124935845),
                version: None,
                category: "Test category".into(),
                tags: vec![],
                app_version: "v1.0.0".into(),
            },
        )
        .unwrap();
        let mut cache = GlobalPipelineCache::default();
        cache.image_rel_path = PathBuf::from("a.tif");
        cache.image_meta.nr_of_bits = 8; // gray value = 0..1 value * 255
        let mut object = object_in_classes(1, &[4]);
        // Channels 0 and 2 measured, channel 1 not; channel 2 without its
        // spread (as a stored annotation's).
        for (channel, value) in [(0, 0.5), (2, 0.25)] {
            object.intensities.insert(
                channel,
                if channel == 0 {
                    Intensity {
                        sum_intensity: value * 4.0,
                        min_intensity: value / 2.0,
                        max_intensity: value * 2.0,
                        avg_intensity: value,
                        std_intensity: 0.1,
                        median_intensity: 0.4,
                        lower_quartile_intensity: 0.2,
                        upper_quartile_intensity: 0.6,
                        mad_intensity: 0.05,
                    }
                } else {
                    Intensity {
                        sum_intensity: value * 4.0,
                        min_intensity: value / 2.0,
                        max_intensity: value * 2.0,
                        avg_intensity: value,
                        ..Default::default()
                    }
                },
            );
        }
        object
            .colocalized_with
            .insert(ObjectClass::Global(7), vec![ObjectId(2), ObjectId(3)]);
        cache.object_cache.insert(ObjectId(1), object);
        cache
            .object_cache
            .insert(ObjectId(2), object_in_classes(2, &[7]));

        exporter.export(&cache).unwrap();

        let conn = shared_connection(&path).unwrap();
        let row = |sql: &str| -> String {
            conn.query_row(
                &format!("SELECT CAST(({sql}) AS VARCHAR) FROM objects ORDER BY object_id LIMIT 1"),
                [],
                |row| row.get(0),
            )
            .unwrap()
        };
        assert_eq!(row("intensity_mean_gray"), "[127.5, NULL, 63.75]");
        assert_eq!(row("intensity_sum_gray"), "[510.0, NULL, 255.0]");
        assert_eq!(row("intensity_min_gray"), "[63.75, NULL, 31.875]");
        assert_eq!(row("intensity_max_gray"), "[255.0, NULL, 127.5]");
        assert_eq!(row("intensity_std_gray"), "[25.5, NULL, NULL]");
        assert_eq!(row("intensity_median_gray"), "[102.0, NULL, NULL]");
        assert_eq!(row("intensity_lower_quartile_gray"), "[51.0, NULL, NULL]");
        assert_eq!(row("intensity_upper_quartile_gray"), "[153.0, NULL, NULL]");
        assert_eq!(row("intensity_mad_gray"), "[12.75, NULL, NULL]");
        assert_eq!(row("len(coloc_partner_ids[7])"), "2");
        assert_eq!(
            row("coloc_partner_ids[7][1]"),
            ObjectId(2).to_string(),
            "partner ids are stored as UUIDs"
        );
        let without: (String, String) = conn
            .query_row(
                "SELECT CAST(intensity_mean_gray AS VARCHAR), CAST(coloc_partner_ids AS VARCHAR) \
                 FROM objects ORDER BY object_id DESC LIMIT 1",
                [],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .unwrap();
        assert_eq!(without, ("[]".to_string(), "{}".to_string()));
    }

    #[test]
    fn export_writes_the_parents_as_a_map_of_one_id_per_class() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("results.evadb");
        let exporter = test_exporter(&path);
        let mut cache = GlobalPipelineCache::default();
        cache.image_rel_path = PathBuf::from("a.tif");
        let mut spot = object_in_classes(1, &[5]);
        spot.set_parent(ObjectClass::Global(3), Some(ObjectId(2)));
        spot.set_parent(ObjectClass::Global(9), Some(ObjectId(3)));
        cache.object_cache.insert(ObjectId(1), spot);
        cache
            .object_cache
            .insert(ObjectId(2), object_in_classes(2, &[3]));

        exporter.export(&cache).unwrap();

        let conn = shared_connection(&path).unwrap();
        let first = |sql: &str, order: &str| -> String {
            conn.query_row(
                &format!("SELECT CAST(({sql}) AS VARCHAR) FROM objects ORDER BY object_id {order} LIMIT 1"),
                [],
                |row| row.get(0),
            )
            .unwrap()
        };
        assert_eq!(first("parent_ids[3]", "ASC"), ObjectId(2).to_string());
        assert_eq!(first("parent_ids[9]", "ASC"), ObjectId(3).to_string());
        assert_eq!(first("parent_ids", "DESC"), "{}", "the cell has no parent");
        let spots_per_cell: i64 = conn
            .query_row(
                "SELECT count(*) FROM objects WHERE parent_ids[3] = ?",
                [ObjectId(2).to_string()],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(spots_per_cell, 1, "children are a query on the parent ids");
    }

    /// The mask is stored run-length encoded and reads back as it was; with
    /// masks off the column is NULL.
    #[test]
    fn export_stores_the_mask_unless_switched_off() {
        let ring = || {
            // A 5 × 5 ring: the border set, the inside not.
            let mut mask = BitVec::<u64, Lsb0>::repeat(true, 25);
            for (x, y) in [
                (1, 1),
                (2, 1),
                (3, 1),
                (1, 2),
                (2, 2),
                (3, 2),
                (1, 3),
                (2, 3),
                (3, 3),
            ] {
                mask.set(y * 5 + x, false);
            }
            let mut object = Object::new(ObjectInit {
                id: ObjectId(1),
                bbox: [10, 10, 14, 14],
                mask_data: mask,
                area: 16,
                ..Default::default()
            });
            object
                .add_object_class(
                    ObjectClass::Global(1),
                    crate::pipeline::scope::Scope::Global,
                )
                .unwrap();
            object
        };
        for store in [true, false] {
            let dir = tempfile::tempdir().unwrap();
            let path = dir.path().join("results.evadb");
            let exporter = test_exporter(&path).with_masks(store);
            let mut cache = GlobalPipelineCache::default();
            cache.image_rel_path = PathBuf::from("a.tif");
            cache.object_cache.insert(ObjectId(1), ring());
            exporter.export(&cache).unwrap();
            let conn = shared_connection(&path).unwrap();
            let blob: Option<Vec<u8>> = conn
                .query_row("SELECT mask FROM objects", [], |row| row.get(0))
                .unwrap();
            match store {
                true => {
                    let bytes = blob.expect("stored");
                    let mask = crate::storage::mask_codec::decode_mask(&bytes, 25).unwrap();
                    assert_eq!(mask, ring().mask_data);
                }
                false => assert!(blob.is_none(), "switched off"),
            }
        }
    }

    /// The centroid is the mean pixel position; the box's centre is stored
    /// next to it.
    #[test]
    fn export_writes_the_centroid_and_the_bounding_box_centre() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("results.evadb");
        let exporter = test_exporter(&path);
        let mut cache = GlobalPipelineCache::default();
        cache.image_rel_path = PathBuf::from("a.tif");
        // An L: a full first column and a full last row of a 4×4 box.
        let mut mask = bitvec::vec::BitVec::<u64, bitvec::order::Lsb0>::repeat(false, 16);
        for i in 0..4 {
            mask.set(i * 4, true);
            mask.set(12 + i, true);
        }
        let mut object = Object::new(crate::object::ObjectInit {
            id: ObjectId(1),
            bbox: [0, 0, 3, 3],
            mask_data: mask,
            area: 7,
            ..Default::default()
        });
        object
            .add_object_class(
                ObjectClass::Global(1),
                crate::pipeline::scope::Scope::Global,
            )
            .unwrap();
        cache.object_cache.insert(ObjectId(1), object);

        exporter.export(&cache).unwrap();

        let conn = shared_connection(&path).unwrap();
        let values: (f64, f64, f64, f64) = conn
            .query_row(
                "SELECT centroid_x_px, centroid_y_px, bbox_center_x_px, bbox_center_y_px FROM objects",
                [],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?)),
            )
            .unwrap();
        // x: 0,0,0,0 + 1,2,3 → 6/7; y: 0,1,2,3 + 3,3,3 → 15/7.
        assert!((values.0 - 6.0 / 7.0).abs() < 1e-4, "{values:?}");
        assert!((values.1 - 15.0 / 7.0).abs() < 1e-4, "{values:?}");
        assert_eq!((values.2, values.3), (1.5, 1.5));
    }

    /// The Appender matches columns by position: the batch has the
    /// `objects` table's columns in its order.
    #[test]
    fn the_object_batch_has_the_tables_columns_in_order() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("results.evadb");
        drop(test_exporter(&path));
        let conn = shared_connection(&path).unwrap();
        let table: Vec<String> = conn
            .prepare(
                "SELECT column_name FROM information_schema.columns \
                 WHERE table_name = 'objects' ORDER BY ordinal_position",
            )
            .unwrap()
            .query_map([], |r| r.get(0))
            .unwrap()
            .collect::<Result<_, _>>()
            .unwrap();
        let batch = ObjectColumns::with_capacity(0).finish().unwrap();
        let names: Vec<String> = batch
            .schema()
            .fields()
            .iter()
            .map(|f| f.name().clone())
            .collect();
        assert_eq!(names, table);
    }

    /// The z columns of a plane's object (`plane.z` 4) and a volume's (3
    /// planes from z 4, two voxels on its middle plane, one on its last):
    /// a plane's zmin = zmax = its plane, no volume; a volume's extent,
    /// centroid and size, no `z_stack` and no planar shape.
    #[test]
    fn export_writes_where_an_object_lies_in_z() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("results.evadb");
        let exporter = test_exporter(&path);
        let mut cache = GlobalPipelineCache::default();
        cache.image_rel_path = PathBuf::from("a.tif");
        cache.image_meta.nr_of_bits = 8;
        cache.image_meta.pixel_sizes = crate::image::PixelSizes {
            px_size_x: 100.0,
            px_size_y: 100.0,
            px_size_z: 500.0,
        };
        let plane = crate::ImagePlane { z: 4, c: 0, t: 0 };
        let one = |mask: &[bool]| {
            mask.iter()
                .collect::<bitvec::vec::BitVec<u64, bitvec::order::Lsb0>>()
        };
        let flat = Object::new(crate::object::ObjectInit {
            id: ObjectId(1),
            bbox: [0, 0, 1, 0],
            mask_data: one(&[true, true]),
            area: 2,
            plane,
            ..Default::default()
        });
        // A 2 × 1 box over 3 planes: plane 1 (z 5) full, plane 2 (z 6) its first.
        let volume = Object::new(crate::object::ObjectInit {
            id: ObjectId(2),
            bbox: [0, 0, 1, 0],
            mask_data: one(&[false, false, true, true, true, false]),
            area: 3,
            sum_z: 1 + 1 + 2,
            plane,
            dimensionality: ImageDimensionality::Volume { depth: 3 },
            ..Default::default()
        });
        cache.object_cache.insert(ObjectId(1), flat);
        cache.object_cache.insert(ObjectId(2), volume);

        exporter.export(&cache).unwrap();

        let conn = shared_connection(&path).unwrap();
        type Row = (
            u8,
            Option<i32>,
            u32,
            u32,
            f64,
            f64,
            f64,
            Option<u64>,
            Option<f64>,
            Option<u64>,
        );
        let mut rows: Vec<Row> = conn
            .prepare(
                "SELECT ndim, z_stack, bbox_zmin_px, bbox_zmax_px, bbox_zmax_nm, centroid_z_px,
                        bbox_center_z_px, volume_px, volume_nm3, area_px
                 FROM objects ORDER BY ndim",
            )
            .unwrap()
            .query_map([], |r| {
                Ok((
                    r.get(0)?,
                    r.get(1)?,
                    r.get(2)?,
                    r.get(3)?,
                    r.get(4)?,
                    r.get(5)?,
                    r.get(6)?,
                    r.get(7)?,
                    r.get(8)?,
                    r.get(9)?,
                ))
            })
            .unwrap()
            .collect::<Result<_, _>>()
            .unwrap();
        let volume = rows.pop().unwrap();
        let flat = rows.pop().unwrap();
        assert_eq!(
            flat,
            (2, Some(4), 4, 4, 2000.0, 4.0, 4.0, None, None, Some(2))
        );
        // Centroid: 4 + (1 + 1 + 2) / 3; 3 voxels of 100 × 100 × 500 nm.
        assert_eq!(
            volume,
            (
                3,
                None,
                4,
                6,
                3000.0,
                4.0 + 4.0 / 3.0,
                5.0,
                Some(3),
                Some(15_000_000.0),
                None
            )
        );
    }

    /// An unknown z spacing (the reader's 0) is no number: its nm values
    /// are NULL, not 0.
    #[test]
    fn an_unknown_z_spacing_leaves_the_nm_values_in_z_empty() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("results.evadb");
        let exporter = test_exporter(&path);
        let mut cache = GlobalPipelineCache::default();
        cache.image_rel_path = PathBuf::from("a.tif");
        cache.image_meta.nr_of_bits = 8;
        let mut object = frame_object(1, 1, 0);
        object.plane.z = 2;
        cache.object_cache.insert(object.id.clone(), object);

        exporter.export(&cache).unwrap();

        let conn = shared_connection(&path).unwrap();
        let row: (Option<f64>, Option<f64>, f64) = conn
            .query_row(
                "SELECT pixel_size_z_nm, centroid_z_nm, centroid_z_px FROM objects",
                [],
                |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)),
            )
            .unwrap();
        assert_eq!(row, (None, None, 2.0));
    }

    #[test]
    fn a_results_file_records_its_format_and_a_newer_one_is_refused() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("results.evadb");
        let exporter = test_exporter(&path);
        drop(exporter);
        let conn = shared_connection(&path).unwrap();
        assert_eq!(
            results_schema_version(&conn).unwrap(),
            RESULTS_SCHEMA_VERSION
        );
        reject_newer_results(&conn).unwrap();

        conn.execute(
            "UPDATE run SET schema_version = ?",
            params![RESULTS_SCHEMA_VERSION + 1],
        )
        .unwrap();
        let refused = reject_newer_results(&conn).unwrap_err().to_string();
        assert!(refused.contains("newer version"), "{refused}");
        assert!(
            DuckDbExporter::open_existing(&path).is_err(),
            "nor added to"
        );
    }

    /// No extension is downloaded or loaded on its own - on a results file
    /// and in memory.
    #[test]
    fn extensions_are_never_installed_or_loaded_on_their_own() {
        let dir = tempfile::tempdir().unwrap();
        let file = shared_connection(&dir.path().join("results.evadb")).unwrap();
        let memory = ResultsConnection::open_in_memory().unwrap();
        for conn in [&file, &memory] {
            let setting = |name: &str| -> String {
                conn.query_row(
                    "SELECT value FROM duckdb_settings() WHERE name = ?",
                    params![name],
                    |row| row.get(0),
                )
                .unwrap()
            };
            assert_eq!(setting("autoinstall_known_extensions"), "false");
            assert_eq!(setting("autoload_known_extensions"), "false");
            assert_eq!(setting("allow_community_extensions"), "false");
        }
        // A function of an extension that isn't built in fails instead.
        let refused = file.execute_batch("SELECT * FROM read_xlsx('x.xlsx')");
        assert!(refused.is_err());
    }

    /// A results file from someone else can't read this computer's files:
    /// a view in it that would (named like a table, run with every query)
    /// fails, and no query reads or writes outside the file's staging
    /// folder, nor switches that back on.
    #[test]
    fn a_results_file_reads_no_local_file() {
        let dir = tempfile::tempdir().unwrap();
        let secret = dir.path().join("secret.txt");
        std::fs::write(&secret, "TOP SECRET").unwrap();
        let secret = secret.to_string_lossy().replace('\'', "''");

        let shared = dir.path().join("shared.evadb");
        Connection::open(&shared)
            .unwrap()
            .execute_batch(&format!(
                "CREATE VIEW images AS SELECT content AS image_name FROM read_text('{secret}');"
            ))
            .unwrap();
        let conn = shared_connection(&shared).unwrap();
        let read = conn.query_row("SELECT image_name FROM images", [], |row| {
            row.get::<_, String>(0)
        });
        let err = read.expect_err("the view read the file").to_string();
        assert!(
            err.contains("Permission Error") && !err.contains("TOP SECRET"),
            "{err}"
        );
        drop(conn);

        // Any file: SQL on it has no file access.
        let results = dir.path().join("results.evadb");
        let conn = shared_connection(&results).unwrap();
        let memory = ResultsConnection::open_in_memory().unwrap();
        for conn in [&*conn, &*memory] {
            let read = conn.query_row(
                &format!("SELECT content FROM read_text('{secret}')"),
                [],
                |row| row.get::<_, String>(0),
            );
            assert!(read.is_err(), "read a local file: {read:?}");
            assert!(
                conn.execute_batch("SET enable_external_access = true;")
                    .is_err()
            );
            assert!(
                conn.execute_batch("SET allowed_directories = ['/'];")
                    .is_err(),
                "widened"
            );
            let outside = dir.path().join("out.parquet");
            let copy = conn.execute_batch(&format!(
                "LOAD parquet; COPY (SELECT 1 AS x) TO '{}' (FORMAT parquet);",
                outside.to_string_lossy().replace('\'', "''")
            ));
            assert!(copy.is_err() && !outside.exists(), "wrote outside");
        }
        // A clone of the file's connection too; its staging folder works.
        let clone = conn.try_clone().unwrap();
        assert!(
            clone
                .execute_batch(&format!("SELECT * FROM read_text('{secret}')"))
                .is_err()
        );
        let staging = conn.export_staging_dir().unwrap();
        assert_eq!(staging, dir.path().join("results.evadb.export"));
        std::fs::create_dir_all(&staging).unwrap();
        let staged = staging.join("x.parquet");
        conn.execute_batch(&format!(
            "LOAD parquet; COPY (SELECT 1 AS x) TO '{}' (FORMAT parquet);",
            staged.to_string_lossy().replace('\'', "''")
        ))
        .unwrap();
        assert!(staged.is_file());
        assert_eq!(memory.export_staging_dir(), None);
    }

    /// Locked down, DuckDB still spills to its own temp folder: a table
    /// larger than its memory limit is written there and read back. Not
    /// through the shared cache: a file's limit is reset there by every
    /// other file opened (`share_memory_limit`), other tests' too. Its temp
    /// folder can't be moved elsewhere either.
    #[test]
    fn a_locked_down_database_still_spills_to_disk() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("results.evadb");
        let conn = Connection::open_with_flags(&path, offline_config().unwrap()).unwrap();
        lock_down(&conn, Some(&export_staging_dir(&path))).unwrap();
        assert!(
            conn.execute_batch("SET temp_directory = '/somewhere/else';")
                .is_err(),
            "the temp folder moved"
        );
        conn.execute_batch(
            "SET threads = 1; SET memory_limit = '60MB';
             CREATE TEMP TABLE big AS
                SELECT range AS x, md5(range::VARCHAR) AS y FROM range(4000000);",
        )
        .unwrap();
        let spilled: i64 = conn
            .query_row(
                "SELECT coalesce(sum(temporary_storage_bytes), 0) FROM duckdb_memory()",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert!(spilled > 60 << 20, "{spilled} bytes spilled");
        let rows: i64 = conn
            .query_row("SELECT count(*) FROM big", [], |row| row.get(0))
            .unwrap();
        assert_eq!(rows, 4_000_000);
    }

    #[test]
    fn a_results_file_without_a_recorded_format_is_version_0() {
        let dir = tempfile::tempdir().unwrap();
        let conn = Connection::open(dir.path().join("old.evadb")).unwrap();
        assert_eq!(results_schema_version(&conn).unwrap(), 0, "no run table");
        conn.execute_batch(
            "CREATE TABLE run (status VARCHAR); INSERT INTO run VALUES ('finished');",
        )
        .unwrap();
        assert_eq!(
            results_schema_version(&conn).unwrap(),
            0,
            "no version column"
        );
        reject_newer_results(&conn).unwrap();
    }

    // ---- reading objects back for dataset steps ----

    fn test_exporter(path: &Path) -> DuckDbExporter {
        DuckDbExporter::new(
            path,
            HashMap::new(),
            &MetaData {
                name: "Experiment name".into(),
                short_description: "Short description".into(),
                description: "Long description".into(),
                authors: vec![],
                creation_time: chrono::DateTime::from_timestamp_nanos(1791318919124935845),
                version: None,
                category: "Test category".into(),
                tags: vec![],
                app_version: "v1.0.0".into(),
            },
        )
        .unwrap()
    }

    /// An object of class `class` in frame `t`, its id's high bits set so
    /// the UUID round trip is checked over all 128 bits.
    fn frame_object(id: u128, class: u32, t: i32) -> Object {
        // 4 pixels: the first 4 of the box's top row.
        let mut mask = BitVec::<u64, Lsb0>::repeat(false, 5 * 11);
        mask[..4].fill(true);
        let mut object = Object::new(ObjectInit {
            id: ObjectId(u128::MAX - id),
            object_class: [ObjectClass::Global(class)].into(),
            bbox: [10, 20, 14, 30],
            mask_data: mask,
            area: 4,
            ..Default::default()
        });
        object.plane = crate::ImagePlane { z: 2, c: 0, t };
        object.intensities.insert(
            1,
            Intensity {
                sum_intensity: 2.0,
                min_intensity: 0.25,
                max_intensity: 0.75,
                avg_intensity: 0.5,
                ..Default::default()
            },
        );
        object
    }

    #[test]
    fn dataset_steps_read_the_exported_objects_back_filtered() {
        let dir = tempfile::tempdir().unwrap();
        let exporter = test_exporter(&dir.path().join("results.evadb"));
        for (image, objects) in [
            ("b.tif", vec![frame_object(3, 1, 1), frame_object(4, 2, 0)]),
            ("a.tif", vec![frame_object(1, 1, 1), frame_object(2, 1, 0)]),
        ] {
            let mut cache = GlobalPipelineCache::default();
            cache.image_rel_path = PathBuf::from(image);
            cache.image_meta.nr_of_bits = 8;
            for object in objects {
                cache.object_cache.insert(object.id.clone(), object);
            }
            exporter.export(&cache).unwrap();
        }
        let store = exporter
            .dataset_store()
            .expect("DuckDB results can be read back");
        let ids = |filter: ObjectFilter| -> Vec<(String, i32, u128)> {
            store
                .get_objects(&filter)
                .unwrap()
                .into_iter()
                .map(|row| (row.image_rel_path, row.t_stack, u128::MAX - row.id.0))
                .collect()
        };

        assert_eq!(
            ids(ObjectFilter::default()),
            [
                ("a.tif", 0, 2),
                ("a.tif", 1, 1),
                ("b.tif", 0, 4),
                ("b.tif", 1, 3)
            ]
            .map(|(image, t, id)| (image.to_string(), t, id)),
            "ordered by image, then frame"
        );
        let in_b = ObjectFilter {
            image_rel_path: Some("b.tif".into()),
            ..Default::default()
        };
        assert_eq!(ids(in_b).len(), 2);
        let frame_1_class_1 = ObjectFilter {
            t_stack: Some(1),
            object_class: Some(ObjectClass::Global(1)),
            ..Default::default()
        };
        assert_eq!(
            ids(frame_1_class_1),
            [("a.tif".to_string(), 1, 1), ("b.tif".to_string(), 1, 3)]
        );
        let unset = ObjectFilter {
            object_class: Some(ObjectClass::Unset),
            ..Default::default()
        };
        assert!(ids(unset).is_empty());

        let row = &store.get_objects(&ObjectFilter::default()).unwrap()[0];
        assert_eq!(row.z_stack, 2);
        assert_eq!(
            row.centroid_px,
            (11.5, 20.0),
            "the mean position of its pixels"
        );
        assert_eq!(row.bbox_px, [10, 20, 14, 30]);
        assert_eq!(row.area_px, 4);
        assert_eq!(row.object_classes, [ObjectClass::Global(1)]);
        assert_eq!(
            row.intensity_mean,
            [None, Some(0.5)],
            "channel 0 not measured"
        );
    }

    #[test]
    fn dataset_steps_read_masks_and_images_and_write_parent_tracks() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("results.evadb");
        let exporter = test_exporter(&path);
        for (image, objects) in [
            ("b.tif", vec![frame_object(3, 1, 0)]),
            ("a.tif", vec![frame_object(1, 1, 0), frame_object(2, 2, 0)]),
        ] {
            let mut cache = GlobalPipelineCache::default();
            cache.image_rel_path = PathBuf::from(image);
            cache.image_meta.nr_of_bits = 8;
            for object in objects {
                cache.object_cache.insert(object.id.clone(), object);
            }
            exporter.export(&cache).unwrap();
        }
        let store = exporter.dataset_store().unwrap();
        assert_eq!(
            store.image_rel_paths(ObjectClass::Global(1)).unwrap(),
            ["a.tif", "b.tif"]
        );
        assert_eq!(
            store.image_rel_paths(ObjectClass::Global(2)).unwrap(),
            ["a.tif"]
        );
        assert!(
            store
                .image_rel_paths(ObjectClass::Unset)
                .unwrap()
                .is_empty()
        );

        let objects = |with_masks| {
            store
                .get_objects(&ObjectFilter {
                    with_masks,
                    ..Default::default()
                })
                .unwrap()
        };
        assert!(
            objects(false).iter().all(|o| o.mask.is_none()),
            "not asked for"
        );
        let mask = objects(true)[0].mask.clone().expect("asked for");
        assert_eq!(mask.len(), 5 * 11);
        assert_eq!(mask.count_ones(), 4);

        let id = |n: u128| ObjectId(u128::MAX - n);
        store
            .set_track_ids(&[
                TrackAssignment {
                    object_id: id(1),
                    track_id: 5,
                    parent_track_ids: vec![2, 3],
                },
                TrackAssignment {
                    object_id: id(3),
                    track_id: 6,
                    parent_track_ids: vec![],
                },
            ])
            .unwrap();
        let parents = |n: u128| -> (Option<u64>, Option<String>) {
            exporter
                .lock_conn()
                .query_row(
                    "SELECT track_id, parent_track_ids::VARCHAR FROM objects
                     WHERE object_id = CAST(? AS UUID)",
                    params![id(n).to_string()],
                    |row| Ok((row.get(0)?, row.get(1)?)),
                )
                .unwrap()
        };
        assert_eq!(parents(1), (Some(5), Some("[2, 3]".into())));
        assert_eq!(parents(3), (Some(6), None));
        assert_eq!(parents(2), (Some(0), None), "not listed: untouched");
    }

    #[test]
    fn a_results_file_from_before_parent_tracks_gets_the_column() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("results.evadb");
        drop(test_exporter(&path));
        let conn = shared_connection(&path).unwrap();
        conn.execute_batch("ALTER TABLE objects DROP COLUMN parent_track_ids")
            .unwrap();
        drop(conn);

        let exporter = DuckDbExporter::open_existing(&path).unwrap();
        let mut cache = GlobalPipelineCache::default();
        cache.image_rel_path = PathBuf::from("a.tif");
        cache.image_meta.nr_of_bits = 8;
        let object = frame_object(1, 1, 0);
        cache.object_cache.insert(object.id.clone(), object);
        exporter.export(&cache).unwrap();
        exporter
            .dataset_store()
            .unwrap()
            .set_track_ids(&[TrackAssignment {
                object_id: ObjectId(u128::MAX - 1),
                track_id: 1,
                parent_track_ids: vec![4],
            }])
            .unwrap();
    }

    /// A clone of an in-memory connection is a connection to the same
    /// database: a table made through one is seen through the other.
    #[test]
    fn a_clone_of_an_in_memory_connection_sees_the_same_database() {
        let first = ResultsConnection::open_in_memory().unwrap();
        first
            .execute_batch("CREATE TABLE t (v INTEGER); INSERT INTO t VALUES (7);")
            .unwrap();
        let second = first.try_clone().unwrap();
        let v: i32 = second
            .query_row("SELECT v FROM t", [], |r| r.get(0))
            .unwrap();
        assert_eq!(v, 7);
        assert!(!is_open(Path::new("")), "an in-memory one is not cached");
    }

    /// A connection can be used mutably (a transaction needs `&mut`): a
    /// rolled back insert leaves nothing, a committed one stays.
    #[test]
    fn a_results_connection_runs_transactions() {
        let mut conn = ResultsConnection::open_in_memory().unwrap();
        conn.execute_batch("CREATE TABLE t (v INTEGER);").unwrap();
        let tx = conn.transaction().unwrap();
        tx.execute_batch("INSERT INTO t VALUES (1);").unwrap();
        tx.rollback().unwrap();
        let tx = conn.transaction().unwrap();
        tx.execute_batch("INSERT INTO t VALUES (2);").unwrap();
        tx.commit().unwrap();
        let values: Vec<i32> = conn
            .prepare("SELECT v FROM t")
            .unwrap()
            .query_map([], |r| r.get(0))
            .unwrap()
            .collect::<Result<_, _>>()
            .unwrap();
        assert_eq!(values, [2]);
    }

    /// Adding to an earlier analysis needs its results file: a missing one
    /// is an error naming the path, and no file is created.
    #[test]
    fn opening_a_missing_results_file_to_add_to_is_an_error() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("missing.evadb");
        let err = DuckDbExporter::open_existing(&path)
            .err()
            .expect("an error");
        assert!(
            matches!(&err, InternalErrors::Io(msg) if msg.starts_with("No results database at") && msg.contains("missing.evadb")),
            "{err:?}"
        );
        assert!(!path.exists());
    }

    /// An image whose bit depth isn't 1-32 can't be exported: its
    /// intensities couldn't be scaled to gray values.
    #[test]
    fn export_refuses_an_implausible_bit_depth() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("results.evadb");
        let exporter = test_exporter(&path);
        for bits in [0, 33] {
            let mut cache = cache_with_objects("a.tif", 1);
            cache.image_meta.nr_of_bits = bits;
            let err = exporter.export(&cache).unwrap_err();
            assert!(
                matches!(&err, InternalErrors::Generic(msg) if msg.contains("cannot export a.tif") && msg.contains(&format!("implausible bit depth {bits}"))),
                "{err:?}"
            );
        }
        assert_eq!(object_count(&path), 0);
    }

    /// Classes without a global number: a module-local class is written
    /// by its label `local_<n>` without an id, `Unset` is left out of the
    /// names and ids, and a parent of class `Unset` is keyed -1.
    #[test]
    fn classes_without_a_global_number_are_written_by_label_and_key_minus_one() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("results.evadb");
        let exporter = test_exporter(&path);
        let mut cache = cache_with_objects("a.tif", 0);
        let mut object = Object::new(ObjectInit {
            id: ObjectId(1),
            object_class: [
                ObjectClass::Global(4),
                ObjectClass::Local(3),
                ObjectClass::Unset,
            ]
            .into(),
            bbox: [0, 0, 1, 1],
            mask_data: BitVec::<u64, Lsb0>::repeat(true, 4),
            area: 4,
            ..Default::default()
        });
        object.set_parent(ObjectClass::Unset, Some(ObjectId(2)));
        cache.object_cache.insert(ObjectId(1), object);

        exporter.export(&cache).unwrap();

        let conn = shared_connection(&path).unwrap();
        let (names, ids, parent): (String, String, String) = conn
            .query_row(
                "SELECT object_class_name, CAST(list_sort(object_class_id) AS VARCHAR), \
                 CAST(parent_ids[-1] AS VARCHAR) FROM objects",
                [],
                |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)),
            )
            .unwrap();
        let mut names: Vec<String> = serde_json::from_str(&names).unwrap();
        names.sort();
        assert_eq!(names, ["class_4", "local_3"]);
        assert_eq!(ids, "[4]");
        assert_eq!(parent, ObjectId(2).to_string());
    }

    /// A measurement value of an object in no global class has an empty
    /// class list (names and ids) - `Unset` is no class to group by.
    #[test]
    fn a_measurement_of_an_unclassified_object_has_no_classes() {
        use crate::pipeline::pipeline_measure::PipelineMeasurements;
        use evanalyzer_cfg::core_types::{MeasurementInfo, MeasurementTarget, MeasurementUnit};
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("results.evadb");
        let exporter = test_exporter(&path);
        let mut cache = cache_with_objects("a.tif", 0);
        cache.object_cache.insert(
            ObjectId(1),
            Object::new(ObjectInit {
                id: ObjectId(1),
                object_class: [ObjectClass::Unset].into(),
                bbox: [0, 0, 1, 1],
                mask_data: BitVec::<u64, Lsb0>::repeat(true, 4),
                area: 4,
                ..Default::default()
            }),
        );
        let info = MeasurementInfo {
            key: "area_01/area".into(),
            name: "Area".into(),
            step: "area_01".into(),
            unit: MeasurementUnit::None,
            target: MeasurementTarget::Object,
            range: None,
            aggregatable: true,
            related_objects: false,
            profile: false,
        };
        cache.measurements = PipelineMeasurements {
            measurements: vec![
                Measurement {
                    info: info.clone(),
                    values: MeasurementValues::Objects(vec![ObjectValue {
                        object: ObjectId(1),
                        value: 4.0,
                        related: None,
                        position: None,
                    }]),
                },
                // No values at all: described, no rows.
                Measurement {
                    info: MeasurementInfo {
                        key: "area_01/empty".into(),
                        ..info
                    },
                    values: MeasurementValues::Objects(vec![]),
                },
            ],
            z_stack: 0,
            t_stack: 0,
            excluded: None,
        };

        exporter.export(&cache).unwrap();

        let conn = shared_connection(&path).unwrap();
        let rows: Vec<(String, String, String, f64)> = conn
            .prepare(
                "SELECT key, object_class_name, CAST(object_class_id AS VARCHAR), value \
                 FROM measurements",
            )
            .unwrap()
            .query_map([], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?)))
            .unwrap()
            .collect::<Result<_, _>>()
            .unwrap();
        assert_eq!(
            rows,
            [("area_01/area".into(), "[]".into(), "[]".into(), 4.0)]
        );
        let described: i64 = conn
            .query_row("SELECT count(*) FROM measurement_info", [], |r| r.get(0))
            .unwrap();
        assert_eq!(described, 2);
    }

    /// The highest track id outside a class: objects of that class are left
    /// out, a class never stored (`Unset`) leaves none out.
    #[test]
    fn the_highest_track_id_leaves_out_only_the_given_class() {
        let dir = tempfile::tempdir().unwrap();
        let exporter = test_exporter(&dir.path().join("results.evadb"));
        let mut cache = cache_with_objects("a.tif", 0);
        for (n, class) in [(1, 1), (2, 2)] {
            let object = frame_object(n, class, 0);
            cache.object_cache.insert(object.id.clone(), object);
        }
        exporter.export(&cache).unwrap();
        let store = exporter.dataset_store().unwrap();
        let id = |n: u128| ObjectId(u128::MAX - n);
        store
            .set_track_ids(&[
                TrackAssignment {
                    object_id: id(1),
                    track_id: 4,
                    parent_track_ids: vec![],
                },
                TrackAssignment {
                    object_id: id(2),
                    track_id: 9,
                    parent_track_ids: vec![],
                },
            ])
            .unwrap();

        assert_eq!(
            store.max_track_id_except(ObjectClass::Global(2)).unwrap(),
            4
        );
        assert_eq!(
            store.max_track_id_except(ObjectClass::Global(1)).unwrap(),
            9
        );
        assert_eq!(store.max_track_id_except(ObjectClass::Unset).unwrap(), 9);
        assert_eq!(store.max_track_id_except(ObjectClass::Local(1)).unwrap(), 9);
    }

    /// Tracking again replaces the lineage of the objects it tracks: one
    /// now without parents loses the old ones, one not tracked again keeps
    /// its own.
    #[test]
    fn tracking_again_clears_the_old_parents_of_the_retracked_objects() {
        let dir = tempfile::tempdir().unwrap();
        let exporter = test_exporter(&dir.path().join("results.evadb"));
        let mut cache = cache_with_objects("a.tif", 0);
        for (n, class) in [(1, 1), (2, 2)] {
            let object = frame_object(n, class, 0);
            cache.object_cache.insert(object.id.clone(), object);
        }
        exporter.export(&cache).unwrap();
        let store = exporter.dataset_store().unwrap();
        let id = |n: u128| ObjectId(u128::MAX - n);
        let assign = |n, track_id, parent_track_ids: Vec<u64>| TrackAssignment {
            object_id: id(n),
            track_id,
            parent_track_ids,
        };
        store
            .set_track_ids(&[assign(1, 5, vec![2, 3]), assign(2, 6, vec![1])])
            .unwrap();
        store.set_track_ids(&[assign(1, 7, vec![])]).unwrap();

        let parents = |n: u128| -> (Option<u64>, Option<String>) {
            exporter
                .lock_conn()
                .query_row(
                    "SELECT track_id, parent_track_ids::VARCHAR FROM objects
                     WHERE object_id = CAST(? AS UUID)",
                    params![id(n).to_string()],
                    |row| Ok((row.get(0)?, row.get(1)?)),
                )
                .unwrap()
        };
        assert_eq!(parents(1), (Some(7), None), "re-tracked without parents");
        assert_eq!(parents(2), (Some(6), Some("[1]".into())), "not re-tracked");
    }

    /// A dataset value naming a measurement description that doesn't exist
    /// is refused, and nothing of the step is written.
    #[test]
    fn a_dataset_value_of_an_unknown_measurement_is_refused() {
        use crate::storage::{DatasetValue, DatasetValueAbout};
        use evanalyzer_cfg::core_types::{MeasurementTarget, MeasurementUnit};
        let dir = tempfile::tempdir().unwrap();
        let exporter = test_exporter(&dir.path().join("results.evadb"));
        let infos = [MeasurementInfo {
            key: "motion_01/vcl".into(),
            name: "VCL".into(),
            step: "motion_01".into(),
            unit: MeasurementUnit::PixelsPerFrame,
            target: MeasurementTarget::Image,
            range: None,
            aggregatable: true,
            related_objects: false,
            profile: false,
        }];
        let value = DatasetValue {
            image_rel_path: "a.tif".into(),
            z_stack: 0,
            t_stack: 0,
            about: DatasetValueAbout::Image,
            info: 5,
            value: 1.0,
        };
        let store = exporter.dataset_store().unwrap();
        let err = store
            .write_measurements("motion_01", &infos, &[value])
            .unwrap_err();
        assert!(
            matches!(&err, InternalErrors::InvalidArgument(msg) if msg == "motion_01: no measurement 5"),
            "{err:?}"
        );
        let rows: i64 = exporter
            .lock_conn()
            .query_row("SELECT count(*) FROM measurement_info", [], |r| r.get(0))
            .unwrap();
        assert_eq!(rows, 0, "rolled back");
    }

    /// Dataset values of a track whose class has no global number name it
    /// by its label without an id; more values than one batch are all
    /// written.
    #[test]
    fn dataset_values_beyond_one_batch_are_all_written() {
        use crate::storage::{DatasetValue, DatasetValueAbout};
        use evanalyzer_cfg::core_types::{MeasurementTarget, MeasurementUnit};
        let dir = tempfile::tempdir().unwrap();
        let exporter = test_exporter(&dir.path().join("results.evadb"));
        let infos = [MeasurementInfo {
            key: "motion_01/vcl".into(),
            name: "VCL".into(),
            step: "motion_01".into(),
            unit: MeasurementUnit::PixelsPerFrame,
            target: MeasurementTarget::Track,
            range: None,
            aggregatable: true,
            related_objects: false,
            profile: false,
        }];
        let count = ARROW_BATCH_ROWS as u64 + 1;
        let values: Vec<DatasetValue> = (1..=count)
            .map(|track_id| DatasetValue {
                image_rel_path: "a.tif".into(),
                z_stack: 0,
                t_stack: 0,
                about: DatasetValueAbout::Track {
                    track_id,
                    class: ObjectClass::Unset,
                },
                info: 0,
                value: track_id as f64,
            })
            .collect();
        let store = exporter.dataset_store().unwrap();
        store
            .write_measurements("motion_01", &infos, &values)
            .unwrap();
        let (rows, distinct, sum, names, ids): (i64, i64, f64, String, String) = exporter
            .lock_conn()
            .query_row(
                "SELECT count(*), count(DISTINCT track_id), sum(value), \
                 any_value(object_class_name), CAST(any_value(object_class_id) AS VARCHAR) \
                 FROM measurements",
                [],
                |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?, r.get(4)?)),
            )
            .unwrap();
        assert_eq!(rows, count as i64);
        assert_eq!(distinct, count as i64);
        // 1 + 2 + ... + n = n (n + 1) / 2
        assert_eq!(sum, (count * (count + 1) / 2) as f64);
        assert_eq!(names, "[\"unset\"]");
        assert_eq!(ids, "[]");
    }

    /// A NULL list has no items.
    #[test]
    fn a_null_list_has_no_items() {
        assert_eq!(list_values(Value::Null).count(), 0);
        assert_eq!(
            list_values(Value::List(vec![Value::Int(1), Value::Int(2)])).count(),
            2
        );
    }
}
