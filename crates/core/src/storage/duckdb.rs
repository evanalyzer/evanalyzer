use crate::object::Intensity;
use crate::pipeline::pipeline_cache::GlobalPipelineCache;
use crate::storage::PipelineResultExporter;
use duckdb::arrow::array::{
    ArrayBuilder, ArrayRef, BooleanBuilder, Float64Builder, Int32Builder, ListBuilder, MapBuilder,
    StringBuilder, UInt8Builder, UInt32Builder, UInt64Builder,
};
use duckdb::arrow::record_batch::RecordBatch;
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

        conn.execute("INSERT INTO run (status) VALUES ('running')", [])
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
// Arrow columns for the objects table
// ---------------------------------------------------------------------------

/// Rows per Arrow batch handed to the Appender - bounds how many rows of a
/// huge image are buffered at once.
const ARROW_BATCH_ROWS: usize = 50_000;

/// The `objects` table's DOUBLE columns, in table order (see
/// `ObjectColumns::finish`).
#[derive(Clone, Copy)]
enum F64 {
    CentroidXPx,
    CentroidYPx,
    CentroidXNm,
    CentroidYNm,
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
}

const F64_COLUMNS: usize = F64::PixelSizeZNm as usize + 1;

/// One batch of `objects` rows, column by column, for the Appender's Arrow
/// path - the only way to append the `object_class_id` list: the row-wise
/// `append_row` rejects list values ("appending List values is not yet
/// supported", duckdb-rs issue #422). Arrow types map onto the table's:
/// text into the UUID and JSON columns is converted by DuckDB.
struct ObjectColumns {
    image_name: StringBuilder,
    image_rel_path: StringBuilder,
    c_stack: Int32Builder,
    z_stack: Int32Builder,
    t_stack: Int32Builder,
    object_id: StringBuilder,
    seg_class_name: StringBuilder,
    seg_class_id: Int32Builder,
    object_class_name: StringBuilder,
    object_class_id: ListBuilder<Int32Builder>,
    parent_id: StringBuilder,
    children: StringBuilder,
    track_id: UInt64Builder,
    bbox_px: [UInt32Builder; 4],
    area_px: UInt64Builder,
    touches_edge: BooleanBuilder,
    image_bit_depth: UInt8Builder,
    /// `intensity_{sum,mean,min,max}_{raw,scaled}`, in table order.
    intensities: [ListBuilder<Float64Builder>; INTENSITY_COLUMNS.len()],
    coloc_partner_ids: MapBuilder<Int32Builder, ListBuilder<StringBuilder>>,
    f64: Vec<Float64Builder>,
}

/// The per-channel intensity list columns, in table order.
const INTENSITY_COLUMNS: [&str; 8] = [
    "intensity_sum_normalized",
    "intensity_sum_gray",
    "intensity_mean_normalized",
    "intensity_mean_gray",
    "intensity_min_normalized",
    "intensity_min_gray",
    "intensity_max_normalized",
    "intensity_max_gray",
];

/// `coloc_partner_ids` key for partners without an object class.
const COLOC_UNSET_CLASS_KEY: i32 = -1;

impl ObjectColumns {
    fn with_capacity(rows: usize) -> Self {
        let strings = || StringBuilder::with_capacity(rows, rows * 16);
        Self {
            image_name: strings(),
            image_rel_path: strings(),
            c_stack: Int32Builder::with_capacity(rows),
            z_stack: Int32Builder::with_capacity(rows),
            t_stack: Int32Builder::with_capacity(rows),
            object_id: StringBuilder::with_capacity(rows, rows * 36),
            seg_class_name: strings(),
            seg_class_id: Int32Builder::with_capacity(rows),
            object_class_name: strings(),
            object_class_id: ListBuilder::with_capacity(Int32Builder::with_capacity(rows), rows),
            parent_id: strings(),
            children: strings(),
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
                ListBuilder::new(StringBuilder::new()),
            ),
            f64: (0..F64_COLUMNS)
                .map(|_| Float64Builder::with_capacity(rows))
                .collect(),
        }
    }

    fn f64(&mut self, column: F64, value: f64) {
        self.f64[column as usize].append_value(value);
    }

    /// One row of every `intensity_*` list: position c + 1 holds channel c,
    /// NULL for a channel without a measurement. `bit_max` turns the
    /// `_normalized` values into gray values (`_gray`).
    fn append_intensities(&mut self, intensities: &IndexMap<i32, Intensity>, bit_max: f64) {
        let channels = intensities
            .keys()
            .filter(|channel| **channel >= 0)
            .max()
            .map_or(0, |max| *max as usize + 1);
        for channel in 0..channels {
            let stats = intensities.get(&(channel as i32)).map(|v| {
                [
                    v.sum_intensity,
                    v.avg_intensity as f64,
                    v.min_intensity as f64,
                    v.max_intensity as f64,
                ]
            });
            for (stat, pair) in self.intensities.chunks_mut(2).enumerate() {
                let normalized = stats.map(|values| values[stat]);
                pair[0].values().append_option(normalized);
                pair[1]
                    .values()
                    .append_option(normalized.map(|normalized| normalized * bit_max));
            }
        }
        for list in &mut self.intensities {
            list.append(true);
        }
    }

    /// One row of `coloc_partner_ids`: partner class -> partner object ids.
    fn append_coloc(
        &mut self,
        colocalized_with: &IndexMap<ObjectClass, Vec<ObjectId>>,
    ) -> Result<(), InternalErrors> {
        for (class, ids) in colocalized_with {
            self.coloc_partner_ids.keys().append_value(match class {
                ObjectClass::Valid(n) => *n as i32,
                ObjectClass::Unset => COLOC_UNSET_CLASS_KEY,
            });
            let partners = self.coloc_partner_ids.values();
            for id in ids {
                partners.values().append_value(id.to_string());
            }
            partners.append(true);
        }
        self.coloc_partner_ids
            .append(true)
            .map_err(|e| InternalErrors::Io(e.to_string()))
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
        let mut f = |column: F64| arr(&mut self.f64[column as usize]);
        let doubles_1 = [
            f(F64::CentroidXPx),
            f(F64::CentroidYPx),
            f(F64::CentroidXNm),
            f(F64::CentroidYNm),
        ];
        let doubles_2 = [
            f(F64::BboxXminNm),
            f(F64::BboxYminNm),
            f(F64::BboxXmaxNm),
            f(F64::BboxYmaxNm),
        ];
        let doubles_3 = [
            f(F64::AreaNm2),
            f(F64::PerimeterPx),
            f(F64::PerimeterNm),
            f(F64::Circularity),
            f(F64::Solidity),
            f(F64::AspectRatio),
            f(F64::Roundness),
            f(F64::Compactness),
            f(F64::MajorAxisPx),
            f(F64::MinorAxisPx),
            f(F64::MajorAxisNm),
            f(F64::MinorAxisNm),
            f(F64::MajorAxisAngle),
            f(F64::Eccentricity),
            f(F64::FeretDiameterPx),
            f(F64::MinFeretPx),
            f(F64::FeretDiameterNm),
            f(F64::MinFeretNm),
        ];
        let doubles_4 = [
            f(F64::PixelSizeXNm),
            f(F64::PixelSizeYNm),
            f(F64::PixelSizeZNm),
        ];
        let [b0, b1, b2, b3] = &mut self.bbox_px;
        let [n0, n1, n2, n3] = doubles_2;
        let [
            a0,
            a1,
            a2,
            a3,
            a4,
            a5,
            a6,
            a7,
            a8,
            a9,
            a10,
            a11,
            a12,
            a13,
            a14,
            a15,
            a16,
            a17,
        ] = doubles_3;
        let [c0, c1, c2, c3] = doubles_1;
        let [p0, p1, p2] = doubles_4;
        let columns: Vec<(&str, ArrayRef)> = vec![
            ("image_name", arr(&mut self.image_name)),
            ("image_rel_path", arr(&mut self.image_rel_path)),
            ("c_stack", arr(&mut self.c_stack)),
            ("z_stack", arr(&mut self.z_stack)),
            ("t_stack", arr(&mut self.t_stack)),
            ("object_id", arr(&mut self.object_id)),
            ("seg_class_name", arr(&mut self.seg_class_name)),
            ("seg_class_id", arr(&mut self.seg_class_id)),
            ("object_class_name", arr(&mut self.object_class_name)),
            ("object_class_id", arr(&mut self.object_class_id)),
            ("parent_id", arr(&mut self.parent_id)),
            ("children", arr(&mut self.children)),
            ("track_id", arr(&mut self.track_id)),
            ("centroid_x_px", c0),
            ("centroid_y_px", c1),
            ("centroid_x_nm", c2),
            ("centroid_y_nm", c3),
            ("bbox_xmin_px", arr(b0)),
            ("bbox_ymin_px", arr(b1)),
            ("bbox_xmax_px", arr(b2)),
            ("bbox_ymax_px", arr(b3)),
            ("bbox_xmin_nm", n0),
            ("bbox_ymin_nm", n1),
            ("bbox_xmax_nm", n2),
            ("bbox_ymax_nm", n3),
            ("area_px", arr(&mut self.area_px)),
            ("area_nm2", a0),
            ("perimeter_px", a1),
            ("perimeter_nm", a2),
            ("circularity", a3),
            ("solidity", a4),
            ("aspect_ratio", a5),
            ("roundness", a6),
            ("compactness", a7),
            ("major_axis_px", a8),
            ("minor_axis_px", a9),
            ("major_axis_nm", a10),
            ("minor_axis_nm", a11),
            ("major_axis_angle", a12),
            ("eccentricity", a13),
            ("feret_diameter_px", a14),
            ("min_feret_px", a15),
            ("feret_diameter_nm", a16),
            ("min_feret_nm", a17),
            ("touches_edge", arr(&mut self.touches_edge)),
            ("pixel_size_x_nm", p0),
            ("pixel_size_y_nm", p1),
            ("pixel_size_z_nm", p2),
            ("image_bit_depth", arr(&mut self.image_bit_depth)),
        ];
        let mut columns = columns;
        for (name, builder) in INTENSITY_COLUMNS.iter().zip(&mut self.intensities) {
            columns.push((name, arr(builder)));
        }
        columns.push(("coloc_partner_ids", arr(&mut self.coloc_partner_ids)));
        RecordBatch::try_from_iter(columns).map_err(|e| InternalErrors::Io(e.to_string()))
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
    object_class_id      INTEGER[],
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
    -- Per-channel intensity statistics, one list each, indexed by channel:
    -- channel c is at position c + 1 (DuckDB lists are 1-based), NULL for a
    -- channel that wasn't measured. `_normalized` is the value divided by
    -- the image's maximum gray value (0..1); `_gray` is the same value in
    -- gray values, as in ImageJ/Fiji (normalized * (2^image_bit_depth - 1)).
    intensity_sum_normalized     DOUBLE[],
    intensity_sum_gray           DOUBLE[],
    intensity_mean_normalized    DOUBLE[],
    intensity_mean_gray          DOUBLE[],
    intensity_min_normalized     DOUBLE[],
    intensity_min_gray           DOUBLE[],
    intensity_max_normalized     DOUBLE[],
    intensity_max_gray           DOUBLE[],
    -- Colocalization partners: partner object class id -> the ids of the
    -- partner objects of that class (key -1: partners without a class).
    coloc_partner_ids            MAP(INTEGER, UUID[])
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

-- How the analysis run that wrote this file ended: 'running' from its start
-- until 'finished', 'cancelled' or 'failed' (with `message`). Still
-- 'running' afterwards means it was interrupted (crash, killed process).
-- One row. Files from before this table existed have none.
CREATE TABLE IF NOT EXISTS run (
    status       VARCHAR NOT NULL,
    message      VARCHAR,
    started_at   TIMESTAMP NOT NULL DEFAULT current_timestamp,
    finished_at  TIMESTAMP
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

// ---------------------------------------------------------------------------
// PipelineResultExporter impl
// ---------------------------------------------------------------------------

impl PipelineResultExporter for DuckDbExporter {
    fn finish_run(&self, outcome: &Result<(), InternalErrors>) -> Result<(), InternalErrors> {
        let (status, message) = match outcome {
            Ok(()) => ("finished", None),
            Err(InternalErrors::Cancelled) => ("cancelled", None),
            Err(e) => ("failed", Some(e.to_string())),
        };
        self.conn
            .lock()
            .map_err(|_| InternalErrors::Internal("results database lock poisoned".into()))?
            .execute(
                "UPDATE run SET status = ?, message = ?, finished_at = current_timestamp",
                params![status, message],
            )
            .map_err(|e| InternalErrors::Io(e.to_string()))?;
        Ok(())
    }

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

        // --- object rows via the Arrow appender ---
        // Rows are collected into Arrow columns and appended a batch at a
        // time: row by row, a 3.8M-object run took ~47 s instead of ~36 s
        // (and `append_row` can't append the `object_class_id` list at all -
        // see `ObjectColumns`). Batches are capped at `ARROW_BATCH_ROWS`, so
        // a huge image never holds all its rows twice at once. The Appender
        // flushes to the database when dropped.
        {
            let mut app = tx
                .appender("objects")
                .map_err(|e| InternalErrors::Io(e.to_string()))?;
            let mut columns = ObjectColumns::with_capacity(object_count.min(ARROW_BATCH_ROWS));

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

                let object_class_names: Vec<String> = object
                    .object_class
                    .iter()
                    .filter(|c| **c != ObjectClass::Unset)
                    .map(|c| label(c))
                    .collect();
                let children_ids: Vec<String> =
                    object.children.iter().map(|id| id.to_string()).collect();

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
                c.z_stack.append_value(object.plane.z);
                c.t_stack.append_value(object.plane.t);
                c.object_id.append_value(object.id.to_string());
                c.seg_class_name
                    .append_value(object.segmentation_class.to_string());
                c.seg_class_id
                    .append_value(object.segmentation_class.0 as i32);
                c.object_class_name
                    .append_value(json_string_array(&object_class_names));
                for class in &object.object_class {
                    if let ObjectClass::Valid(n) = class {
                        c.object_class_id.values().append_value(*n as i32);
                    }
                }
                c.object_class_id.append(true);
                c.parent_id
                    .append_option(object.parent_id.as_ref().map(|id| id.to_string()));
                c.children.append_value(json_string_array(&children_ids));
                c.track_id.append_value(object.track.id.0);
                c.f64(F64::CentroidXPx, centroid_x_px);
                c.f64(F64::CentroidYPx, centroid_y_px);
                c.f64(F64::CentroidXNm, centroid_x_px * pxx);
                c.f64(F64::CentroidYNm, centroid_y_px * pxy);
                for (column, value) in c.bbox_px.iter_mut().zip(object.bbox) {
                    column.append_value(value);
                }
                c.f64(F64::BboxXminNm, object.bbox[0] as f64 * pxx);
                c.f64(F64::BboxYminNm, object.bbox[1] as f64 * pxy);
                c.f64(F64::BboxXmaxNm, object.bbox[2] as f64 * pxx);
                c.f64(F64::BboxYmaxNm, object.bbox[3] as f64 * pxy);
                c.area_px.append_value(object.area as u64);
                c.f64(F64::AreaNm2, object.area as f64 * pxx * pxy);
                c.f64(F64::PerimeterPx, perimeter);
                c.f64(F64::PerimeterNm, perimeter * px_len);
                c.f64(F64::Circularity, roundness);
                c.f64(F64::Solidity, object.get_solidity() as f64);
                c.f64(F64::AspectRatio, aspect_ratio);
                c.f64(F64::Roundness, roundness);
                c.f64(
                    F64::Compactness,
                    object.get_compactness(perimeter_f32) as f64,
                );
                c.f64(F64::MajorAxisPx, ellipse.major as f64);
                c.f64(F64::MinorAxisPx, ellipse.minor as f64);
                c.f64(F64::MajorAxisNm, ellipse.major as f64 * px_len);
                c.f64(F64::MinorAxisNm, ellipse.minor as f64 * px_len);
                c.f64(F64::MajorAxisAngle, ellipse.angle as f64);
                c.f64(F64::Eccentricity, ellipse.eccentricity as f64);
                c.f64(F64::FeretDiameterPx, feret);
                c.f64(F64::MinFeretPx, min_feret);
                c.f64(F64::FeretDiameterNm, feret * px_len);
                c.f64(F64::MinFeretNm, min_feret * px_len);
                c.touches_edge.append_value(object.touches_edge);
                c.f64(F64::PixelSizeXNm, pxx);
                c.f64(F64::PixelSizeYNm, pxy);
                c.f64(F64::PixelSizeZNm, px.px_size_z as f64);
                c.image_bit_depth.append_value(nr_of_bits as u8);
                c.append_intensities(&object.intensities, bit_max);
                c.append_coloc(&object.colocalized_with)?;

                if columns.len() >= ARROW_BATCH_ROWS {
                    app.append_record_batch(columns.finish()?)
                        .map_err(|e| InternalErrors::Io(e.to_string()))?;
                }
            }
            if columns.len() > 0 {
                app.append_record_batch(columns.finish()?)
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::object::{Object, ObjectInit};
    use bitvec::prelude::*;
    use std::collections::HashSet;

    fn object_in_classes(id: u128, classes: &[u32]) -> Object {
        Object::new(ObjectInit {
            id: ObjectId(id),
            object_class: classes
                .iter()
                .map(|c| ObjectClass::Valid(*c))
                .collect::<HashSet<_>>(),
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
            let exporter = DuckDbExporter::new(&path, HashMap::new()).unwrap();
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

    #[test]
    fn export_writes_object_classes_as_an_integer_list() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("results.evadb");
        let exporter = DuckDbExporter::new(&path, HashMap::new()).unwrap();
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

    #[test]
    fn export_writes_intensities_per_channel_and_coloc_partners_as_a_map() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("results.evadb");
        let exporter = DuckDbExporter::new(&path, HashMap::new()).unwrap();
        let mut cache = GlobalPipelineCache::default();
        cache.image_rel_path = PathBuf::from("a.tif");
        cache.image_meta.nr_of_bits = 8; // gray value = normalized * 255
        let mut object = object_in_classes(1, &[4]);
        // Channels 0 and 2 measured, channel 1 not.
        for (channel, value) in [(0, 0.5), (2, 0.25)] {
            object.intensities.insert(
                channel,
                Intensity {
                    sum_intensity: value * 4.0,
                    min_intensity: value as f32 / 2.0,
                    max_intensity: value as f32 * 2.0,
                    avg_intensity: value as f32,
                    pixel_values: Vec::new(),
                },
            );
        }
        object
            .colocalized_with
            .insert(ObjectClass::Valid(7), vec![ObjectId(2), ObjectId(3)]);
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
        assert_eq!(row("intensity_mean_normalized"), "[0.5, NULL, 0.25]");
        assert_eq!(row("intensity_mean_gray"), "[127.5, NULL, 63.75]");
        assert_eq!(row("intensity_sum_normalized"), "[2.0, NULL, 1.0]");
        assert_eq!(row("intensity_min_normalized"), "[0.25, NULL, 0.125]");
        assert_eq!(row("intensity_max_gray"), "[255.0, NULL, 127.5]");
        assert_eq!(row("len(coloc_partner_ids[7])"), "2");
        assert_eq!(
            row("coloc_partner_ids[7][1]"),
            ObjectId(2).to_string(),
            "partner ids are stored as UUIDs"
        );
        let without: (String, String) = conn
            .query_row(
                "SELECT CAST(intensity_mean_normalized AS VARCHAR), CAST(coloc_partner_ids AS VARCHAR) \
                 FROM objects ORDER BY object_id DESC LIMIT 1",
                [],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .unwrap();
        assert_eq!(without, ("[]".to_string(), "{}".to_string()));
    }
}
