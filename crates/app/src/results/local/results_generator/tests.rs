//! # tests
//!
//! The results generator's tests: every view on small seeded results files.
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-11-22
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.

#[test]
fn coloc_groups_are_the_classes_partnered_together() {
    let c = |n: u32| ObjectClass::Global(n);
    // In-cell spots 4, 5, 6 overlap among themselves, outside spots 7,
    // 8, 9 among themselves; 20 is a marker class the in-cell and the
    // outside spots both carry - not a partner, so it joins nothing.
    let pairs = vec![
        (vec![c(4), c(20)], vec![c(5)]),
        (vec![c(5)], vec![c(4), c(6)]),
        (vec![c(7), c(20)], vec![c(8), c(9)]),
        (vec![c(9)], vec![c(7)]),
    ];
    assert_eq!(
        coloc_groups_of(&pairs),
        vec![vec![c(4), c(5), c(6)], vec![c(7), c(8), c(9)]]
    );
}

use super::columns::coloc_groups_of;
use super::plate::auto_well_size;
use super::*;

/// Performance of the List view, normal vs. transposed, on a large
/// synthetic database. Not a correctness test - run explicitly:
///
/// ```sh
/// BENCH_IMAGES=400 BENCH_CLASSES=4 BENCH_PER_CLASS=300 \
///   cargo test --release -p evanalyzer_app --lib bench_transposed_list \
///   -- --ignored --nocapture
/// ```
///
/// Measures what a user actually waits for: the first page, paging on
/// (GUI scrolling, 500 rows/page like `LIST_PAGE_SIZE`), and walking
/// every page (what an export does). For grouped-by-image it also times
/// transposing in Rust after the normal query, the alternative to doing
/// it in SQL.
#[test]
#[ignore]
fn bench_transposed_list() {
    use super::super::test_support::seed_synthetic_db;
    use std::time::{Duration, Instant};
    let env = |name: &str, default: u32| {
        std::env::var(name)
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(default)
    };
    let (images, classes, per_class) = (
        env("BENCH_IMAGES", 400),
        env("BENCH_CLASSES", 4),
        env("BENCH_PER_CLASS", 300),
    );
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("bench.evadb");
    let start = Instant::now();
    seed_synthetic_db(&path, images, classes, per_class);
    let generator = ResultsGenerator::open_database(path).unwrap();
    println!(
        "\n{} objects ({images} images x {classes} classes x {per_class}), seeded in {:?}",
        images * classes * per_class,
        start.elapsed()
    );
    let columns = vec![
        Column::ImageName,
        Column::ObjectClass,
        Column::AreaSizePx,
        Column::Circularity,
        Column::IntensityAvg(0),
    ];
    const PAGE: i32 = 500;
    let generator = &generator;
    let columns = &columns;

    // Walks up to `max_pages` pages; returns (first page, avg per page,
    // pages, rows).
    let walk = |fetch: &dyn Fn(Option<String>) -> DatabaseResult, max_pages: usize| {
        let mut cursor = None;
        let mut times = Vec::new();
        let mut rows = 0;
        for _ in 0..max_pages {
            let start = Instant::now();
            let page = fetch(cursor.take());
            times.push(start.elapsed());
            rows += page.rows.len();
            cursor = page.row_names.last().cloned();
            if page.rows.len() < PAGE as usize {
                break;
            }
        }
        let total: Duration = times.iter().sum();
        (
            times[0],
            total / times.len() as u32,
            times.len(),
            rows,
            total,
        )
    };
    let report = |name: &str, r: (Duration, Duration, usize, usize, Duration)| {
        println!(
            "{name:<34} first {:>9.2?}  avg/page {:>9.2?}  {:>4} pages {:>8} rows  total {:>9.2?}",
            r.0, r.1, r.2, r.3, r.4
        );
    };
    let list = |transpond_table: bool| {
        move |after: Option<String>| {
            generator
                .get_object_list(&ListFilter {
                    plane: plane(),
                    images: crate::results::ImageSet::All,
                    object_classes: None,
                    columns: columns.clone(),
                    with_coloc_details: false,
                    related_to: None,
                    page: Pagination { limit: PAGE, after },
                    transpond_table,
                    sort: None,
                })
                .unwrap()
        }
    };
    let grouped = |transpond_table: bool| {
        move |after: Option<String>| {
            generator
                .get_grouped_by_image(&GroupedByImageFilter {
                    plane: plane(),
                    images: crate::results::ImageSet::All,
                    object_classes: None,
                    columns: vec![Column::Count, Column::AreaSizePx, Column::Circularity],
                    aggregation: vec![Aggregation::Avg, Aggregation::Max],
                    page: Pagination { limit: PAGE, after },
                    transpond_table,
                    grouping: None,
                    empty_classes: true,
                    sort: None,
                })
                .unwrap()
        }
    };

    for transposed in [false, true] {
        let label = if transposed { "transposed" } else { "normal" };
        report(
            &format!("object list {label} (20 pages)"),
            walk(&list(transposed), 20),
        );
    }
    for transposed in [false, true] {
        let label = if transposed { "transposed" } else { "normal" };
        report(
            &format!("grouped {label} (all pages)"),
            walk(&grouped(transposed), usize::MAX),
        );
    }

    // The alternative: normal grouped query, transposed in Rust.
    let start = Instant::now();
    let mut cursor = None;
    let mut per_image: HashMap<String, HashMap<String, Vec<String>>> = HashMap::new();
    loop {
        let page = grouped(false)(cursor.take());
        for row in &page.rows {
            let image = match &row[0].value {
                CellValue::String(s) => s.clone(),
                _ => String::new(),
            };
            let class = match &row[1].value {
                CellValue::Class((name, _)) => name.clone(),
                _ => String::new(),
            };
            let values = row[2..]
                .iter()
                .map(|cell| match cell.value {
                    CellValue::Float(v) => v.to_string(),
                    _ => String::new(),
                })
                .collect();
            per_image.entry(image).or_default().insert(class, values);
        }
        cursor = page.row_names.last().cloned();
        if page.rows.len() < PAGE as usize {
            break;
        }
    }
    println!(
        "{:<34} total {:>9.2?}  ({} image rows)",
        "grouped normal + pivot in Rust",
        start.elapsed(),
        per_image.len()
    );
}

/// The run's reason for leaving an image out is read; enabling the
/// image by hand takes it away.
#[test]
fn an_excluded_images_reason_is_read_and_goes_when_enabled_by_hand() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("results.evadb");
    {
        let old = duckdb::Connection::open(&path).unwrap();
        old.execute_batch(
            "CREATE TABLE images (image_name VARCHAR, image_rel_path VARCHAR PRIMARY KEY, \
             disabled BOOLEAN NOT NULL DEFAULT false, disabled_reason VARCHAR); \
             INSERT INTO images VALUES ('a', 'a.tif', false, NULL), ('b', 'b.tif', true, NULL);",
        )
        .unwrap();
    }
    let results = ResultsGenerator::open_database(path).unwrap();
    let reasons = |results: &ResultsGenerator| -> Vec<(bool, Option<String>)> {
        results
            .get_images()
            .unwrap()
            .into_iter()
            .map(|image| (image.disabled, image.disabled_reason))
            .collect()
    };
    assert_eq!(reasons(&results), [(false, None), (true, None)]);

    results
        .connection()
        .execute_batch(
            "UPDATE images SET disabled = true, disabled_reason = 'Focus score low' \
             WHERE image_rel_path = 'a.tif'",
        )
        .unwrap();
    assert_eq!(
        reasons(&results)[0],
        (true, Some("Focus score low".to_string()))
    );
    results.enable_image("a.tif", false).unwrap();
    assert_eq!(reasons(&results)[0], (false, None), "the user decided");
}

/// Opening results the analysis in this process just wrote must reuse
/// the analysis' connection. A separate `Connection::open` of the same
/// file is refused on Windows ("file is being used by another process");
/// on Linux it is allowed but becomes a second, stale database instance
/// - which this test detects: it can't see rows written afterwards.
#[test]
fn results_share_the_connection_the_analysis_wrote_with() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("results.evadb");
    let writer = evanalyzer_core::open_results_database(&path).unwrap();
    writer
        .execute_batch("CREATE TABLE t (x INTEGER); INSERT INTO t VALUES (1);")
        .unwrap();

    let results = ResultsGenerator::open_database(path.clone()).unwrap();
    writer.execute("INSERT INTO t VALUES (2)", []).unwrap();

    let count: i64 = results
        .connection()
        .query_row("SELECT count(*) FROM t", [], |row| row.get(0))
        .unwrap();
    assert_eq!(count, 2, "results view must see the analysis' writes");
}

/// Builds a `ResultsGenerator` over an in-memory database with just the
/// columns `get_available_columns`/`get_nr_of_c_stacks` actually touch.
/// `objects.c_stack` is deliberately fixed at `0` for every row (as real
/// pipeline extraction does, see `extract_objects.rs`) to prove channel
/// discovery no longer depends on it — the real source is `images.c_stacks`,
/// populated per image from real metadata (see `finalize_image` in
/// evanalyzer_core's duckdb.rs), so this takes one `c_stacks` value per
/// "image" row directly rather than inferring it from object data.
fn generator_with_c_stacks(c_stacks_per_image: &[u32]) -> ResultsGenerator {
    let database = evanalyzer_core::ResultsConnection::open_in_memory().unwrap();
    database
        .execute_batch(
            "CREATE TABLE classes (class_id INTEGER, name VARCHAR, color UINTEGER);
             CREATE TABLE objects (
                 c_stack INTEGER,
                 coloc_partner_ids MAP(INTEGER, UUID[]),
                 intensity_mean_gray DOUBLE[]
             );
             CREATE TABLE images (
                 image_rel_path VARCHAR,
                 c_stacks UINTEGER
             );",
        )
        .unwrap();
    for (i, c_stacks) in c_stacks_per_image.iter().enumerate() {
        database
            .execute(
                "INSERT INTO images (image_rel_path, c_stacks) VALUES (?, ?)",
                duckdb::params![format!("image_{i}.tif"), c_stacks],
            )
            .unwrap();
    }
    ResultsGenerator {
        database,
        classes_cache: RefCell::new(None),
        coloc_classes_cache: RefCell::new(None),
        classes_with_objects_cache: RefCell::new(None),
        auto_grouping_cache: RefCell::new(None),
        object_columns_cache: RefCell::new(None),
        volumes_cache: RefCell::new(None),
        profiles_cache: RefCell::new(None),
        relations_cache: RefCell::new(None),
        track_motion_cache: RefCell::new(None),
    }
}

#[test]
fn nr_of_c_stacks_reads_the_real_per_image_channel_count() {
    let generator = generator_with_c_stacks(&[3]);
    assert_eq!(generator.get_nr_of_c_stacks(), 3);
}

#[test]
fn nr_of_c_stacks_is_the_max_across_every_image() {
    let generator = generator_with_c_stacks(&[1, 3, 2]);
    assert_eq!(generator.get_nr_of_c_stacks(), 3);
}

#[test]
fn available_columns_include_one_intensity_group_per_measured_channel() {
    let generator = generator_with_c_stacks(&[3]);
    let columns = generator.get_available_columns().unwrap();
    let intensity_avg_columns = columns
        .iter()
        .filter(|c| matches!(c.key, Column::IntensityAvg(_)))
        .count();
    assert_eq!(intensity_avg_columns, 3);
}

#[test]
fn nr_of_c_stacks_falls_back_to_one_when_images_table_is_empty() {
    let generator = generator_with_c_stacks(&[]);
    assert_eq!(generator.get_nr_of_c_stacks(), 1);
}

use super::super::test_support::{ObjectSpec, TestResults, seed_db};

/// Opens a fresh `ResultsGenerator` over a temp `.evadb` seeded with
/// `objects` (see `test_support::seed_db`) - its folder goes with it.
fn open(objects: &[ObjectSpec]) -> TestResults {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("results.evadb");
    seed_db(&path, objects);
    TestResults::open(dir, &path)
}

fn plane() -> PlaneFilter {
    PlaneFilter {
        z_stack: 0,
        t_stack: 0,
    }
}

#[test]
fn the_run_status_comes_from_the_run_table_and_is_unknown_without_it() {
    use crate::results::RunStatus;
    let old = open(&[]);
    assert_eq!(old.run_status().unwrap(), RunStatus::Unknown);

    for (sql, expected) in [
        ("('running', NULL)", RunStatus::NotFinished),
        ("('finished', NULL)", RunStatus::Finished),
        ("('cancelled', NULL)", RunStatus::Cancelled),
        (
            "('failed', 'disk full')",
            RunStatus::Failed("disk full".into()),
        ),
    ] {
        let db = open(&[]);
        db.database
            .execute_batch(&format!(
                "CREATE TABLE run (status VARCHAR, message VARCHAR); \
                 INSERT INTO run VALUES {sql};"
            ))
            .unwrap();
        assert_eq!(db.run_status().unwrap(), expected);
    }
    assert!(RunStatus::Finished.warning().is_none());
    assert!(
        RunStatus::Cancelled
            .warning()
            .unwrap()
            .contains("incomplete")
    );
}

/// A limit generous enough to fetch every row a test seeds in one page
/// — unlike `fetch_all_list_rows`/`fetch_all_grouped_by_image_rows`
/// (results_exporter.rs), which page through `get_object_list`/
/// `get_grouped_by_image` themselves and only ever pass `limit: 0` as a
/// template immediately overwritten before use, a direct call here with
/// `limit: 0` would produce a literal `LIMIT 0` — zero rows.
fn no_page() -> Pagination {
    Pagination {
        limit: 1000,
        after: None,
    }
}

// -- relations (Relate Objects) ------------------------------------------

const CELL: ObjectClass = ObjectClass::Global(1);
const SPOT: ObjectClass = ObjectClass::Global(2);

/// The id `seed_db` gives the `idx`-th object.
fn seeded_id(idx: usize) -> String {
    format!("00000000-0000-0000-0000-{idx:012}")
}

/// Two cells (100 px, 200 px) and four spots (10, 20, 30, 40 px): the
/// first two spots in cell 0, the third in cell 1, the last in none.
fn cells_with_spots() -> TestResults {
    let generator = open(&[
        ObjectSpec::new("img1.tif", "Cell", 1, 100),
        ObjectSpec::new("img1.tif", "Cell", 1, 200),
        ObjectSpec::new("img1.tif", "Spot", 2, 10),
        ObjectSpec::new("img1.tif", "Spot", 2, 20),
        ObjectSpec::new("img1.tif", "Spot", 2, 30),
        ObjectSpec::new("img1.tif", "Spot", 2, 40),
    ]);
    for (spot, cell) in [(2, 0), (3, 0), (4, 1)] {
        generator
            .database
            .execute(
                &format!(
                    "UPDATE objects SET parent_ids = MAP {{1: '{}'::UUID}} \
                     WHERE object_id = '{}'",
                    seeded_id(cell),
                    seeded_id(spot)
                ),
                [],
            )
            .unwrap();
    }
    generator
}

fn children(value: ChildValue) -> Column {
    Column::Children(ChildColumn {
        parent_class: CELL,
        child_class: SPOT,
        value,
    })
}

fn parent(value: ParentValue) -> Column {
    Column::Parent(ParentColumn {
        parent_class: CELL,
        value,
    })
}

fn list_of(
    generator: &ResultsGenerator,
    class: ObjectClass,
    columns: Vec<Column>,
) -> DatabaseResult {
    generator
        .get_object_list(&ListFilter {
            plane: plane(),
            images: crate::results::ImageSet::All,
            object_classes: Some(vec![class]),
            columns,
            with_coloc_details: false,
            related_to: None,
            page: no_page(),
            transpond_table: false,
            sort: None,
        })
        .unwrap()
}

fn number(cell: &Cell) -> Option<f64> {
    match &cell.value {
        CellValue::Integer(v) => Some(*v as f64),
        CellValue::Float(v) => Some(*v as f64),
        CellValue::Empty => None,
        other => panic!("expected a number, got {:?}", std::mem::discriminant(other)),
    }
}

#[test]
fn the_links_are_offered_as_children_and_parent_columns() {
    let columns = cells_with_spots().get_available_columns().unwrap();
    let keys: Vec<&Column> = columns.iter().map(|c| &c.key).collect();
    assert!(keys.contains(&&children(ChildValue::Count)));
    assert!(keys.contains(&&parent(ParentValue::Id)));
    let unlinked = open(&[ObjectSpec::new("img1.tif", "Cell", 1, 100)]);
    assert!(
        !unlinked
            .get_available_columns()
            .unwrap()
            .iter()
            .any(|c| matches!(c.key, Column::Children(_) | Column::Parent(_)))
    );
}

#[test]
fn a_cell_row_shows_its_spots_count_and_mean() {
    let generator = cells_with_spots();
    let mean_area = children(ChildValue::Of(
        Aggregation::Avg,
        Box::new(Column::AreaSizePx),
    ));
    let result = list_of(
        &generator,
        CELL,
        vec![Column::AreaSizePx, children(ChildValue::Count), mean_area],
    );
    assert_eq!(
        result.column_names,
        ["Area [px]", "Spot · Count", "Spot · Mean · Area [px]"]
    );
    let rows: Vec<Vec<Option<f64>>> = result
        .rows
        .iter()
        .map(|row| row.iter().map(number).collect())
        .collect();
    assert_eq!(
        rows,
        [
            vec![Some(100.0), Some(2.0), Some(15.0)],
            vec![Some(200.0), Some(1.0), Some(30.0)]
        ]
    );
}

#[test]
fn a_spot_row_shows_its_cell_and_the_cells_values() {
    let generator = cells_with_spots();
    let result = list_of(
        &generator,
        SPOT,
        vec![
            Column::AreaSizePx,
            parent(ParentValue::Id),
            parent(ParentValue::Of(Box::new(Column::AreaSizePx))),
        ],
    );
    assert_eq!(result.column_names[1], "Parent Cell");
    assert_eq!(result.column_names[2], "Parent Cell · Area [px]");
    let rows: Vec<(f64, String, Option<f64>)> = result
        .rows
        .iter()
        .map(|row| {
            let id = match &row[1].value {
                CellValue::String(id) => id.clone(),
                CellValue::Empty => "-".into(),
                _ => panic!("expected the parent's id"),
            };
            (number(&row[0]).unwrap(), id, number(&row[2]))
        })
        .collect();
    assert_eq!(
        rows,
        [
            (10.0, seeded_id(0), Some(100.0)),
            (20.0, seeded_id(0), Some(100.0)),
            (30.0, seeded_id(1), Some(200.0)),
            (40.0, "-".into(), None),
        ]
    );
}

/// Per image the children's count is aggregated over the cells like
/// any column: the mean and the sum of spots per cell.
#[test]
fn spots_per_cell_are_aggregated_per_image() {
    let generator = cells_with_spots();
    let result = generator
        .get_grouped_by_image(&GroupedByImageFilter {
            plane: plane(),
            images: crate::results::ImageSet::All,
            object_classes: Some(vec![CELL]),
            columns: vec![children(ChildValue::Count)],
            aggregation: vec![Aggregation::Avg, Aggregation::Sum],
            page: no_page(),
            transpond_table: false,
            grouping: None,
            empty_classes: true,
            sort: None,
        })
        .unwrap();
    let values: Vec<Option<f64>> = result.rows[0][2..].iter().map(number).collect();
    assert_eq!(values, [Some(1.5), Some(3.0)]);
}

#[test]
fn relation_columns_round_trip_through_their_keys() {
    for column in [
        children(ChildValue::Count),
        children(ChildValue::Of(
            Aggregation::Median,
            Box::new(Column::IntensityAvg(2)),
        )),
        parent(ParentValue::Id),
        parent(ParentValue::Of(Box::new(Column::Circularity))),
    ] {
        let key = column.as_key(&[]);
        assert_eq!(Column::from_key(&key, &[]), Some(column), "{key}");
    }
}

// -- get_object_list --------------------------------------------------

#[test]
fn get_object_list_returns_one_row_per_object_with_requested_columns() {
    let generator = open(&[
        ObjectSpec::new("img1.tif", "ClassA", 1, 100),
        ObjectSpec::new("img2.tif", "ClassB", 2, 200),
    ]);
    let result = generator
        .get_object_list(&ListFilter {
            plane: plane(),
            images: crate::results::ImageSet::All,
            object_classes: None,
            columns: vec![Column::ObjectClass, Column::AreaSizePx],
            with_coloc_details: false,
            related_to: None,
            page: no_page(),
            transpond_table: false,
            sort: None,
        })
        .unwrap();

    assert_eq!(
        result.column_names,
        vec!["Class".to_string(), "Area [px]".to_string()]
    );
    assert_eq!(result.rows.len(), 2);
    assert_eq!(result.source_object_count, 2);
    assert_eq!(result.row_names.len(), 2);
    assert_eq!(result.row_locations.len(), 2);
}

#[test]
fn get_object_list_shows_each_objects_track_and_nothing_when_untracked() {
    let generator = open(&[
        ObjectSpec::new("img1.tif", "ClassA", 1, 100).in_track(7),
        ObjectSpec::new("img1.tif", "ClassA", 1, 200),
    ]);
    let result = generator
        .get_object_list(&ListFilter {
            plane: plane(),
            images: crate::results::ImageSet::All,
            object_classes: None,
            columns: vec![Column::TrackId, Column::AreaSizePx],
            with_coloc_details: false,
            related_to: None,
            page: no_page(),
            transpond_table: false,
            sort: None,
        })
        .unwrap();

    assert_eq!(result.column_names[0], "Track ID");
    let mut tracks: Vec<String> = result
        .rows
        .iter()
        .map(|row| match &row[0].value {
            CellValue::String(id) => id.clone(),
            CellValue::Empty => "-".into(),
            _ => panic!("expected a text or empty cell"),
        })
        .collect();
    tracks.sort();
    assert_eq!(tracks, ["-", "7"]);
}

#[test]
fn the_track_id_column_round_trips_and_is_never_aggregated() {
    let generator = open(&[]);
    let columns = generator.get_available_columns().unwrap();
    assert!(columns.iter().any(|c| c.key == Column::TrackId));
    let key = Column::TrackId.as_key(&[]);
    assert_eq!(Column::from_key(&key, &[]), Some(Column::TrackId));
    assert!(!Column::TrackId.is_aggregable());
}

/// Every cell in a row belongs to the same source object, so every one
/// of them - not just an image-name/path column - must carry that
/// object's own image's disabled flag, letting the GUI strike the whole
/// row through regardless of which columns are actually shown.
#[test]
fn get_object_list_flags_every_cell_of_a_disabled_images_row() {
    let generator = open(&[
        ObjectSpec::new("img1.tif", "ClassA", 1, 100),
        ObjectSpec::new("img2.tif", "ClassB", 2, 200),
    ]);
    generator.enable_image("img2.tif", true).unwrap();

    let result = generator
        .get_object_list(&ListFilter {
            plane: plane(),
            images: crate::results::ImageSet::All,
            object_classes: None,
            columns: vec![Column::ImageName, Column::ObjectClass, Column::AreaSizePx],
            with_coloc_details: false,
            related_to: None,
            page: no_page(),
            transpond_table: false,
            sort: None,
        })
        .unwrap();

    let img1_row = result
        .rows
        .iter()
        .position(|row| matches!(&row[0].value, CellValue::String(s) if s == "img1.tif"))
        .expect("img1's row");
    let img2_row = result
        .rows
        .iter()
        .position(|row| matches!(&row[0].value, CellValue::String(s) if s == "img2.tif"))
        .expect("img2's row");

    assert!(
        result.rows[img1_row].iter().all(|cell| !cell.disabled),
        "img1 is enabled, so none of its cells should be struck through"
    );
    assert!(
        result.rows[img2_row].iter().all(|cell| cell.disabled),
        "img2 is disabled, so every cell of its row - not just the image name - must be flagged"
    );
}

#[test]
fn get_object_list_image_filter_restricts_to_the_selected_image() {
    let generator = open(&[
        ObjectSpec::new("img1.tif", "ClassA", 1, 100),
        ObjectSpec::new("img2.tif", "ClassB", 2, 200),
    ]);
    let result = generator
        .get_object_list(&ListFilter {
            plane: plane(),
            images: crate::results::ImageSet::Listed(vec!["img1.tif".to_string()]),
            object_classes: None,
            columns: vec![Column::ImageName],
            with_coloc_details: false,
            related_to: None,
            page: no_page(),
            transpond_table: false,
            sort: None,
        })
        .unwrap();
    assert_eq!(result.source_object_count, 1);
}

/// Images in subfolders: the selection carries their relative paths
/// (the objects' `image_rel_path`), not only the file name.
#[test]
fn get_object_list_image_filter_finds_images_in_subfolders() {
    let generator = open(&[
        ObjectSpec::new("plate1/img1.tif", "ClassA", 1, 100),
        ObjectSpec::new("plate1/img1.tif", "ClassA", 1, 110),
        ObjectSpec::new("plate2/img1.tif", "ClassB", 2, 200),
    ]);
    // As a run writes them: the name is the file name only.
    generator
        .database
        .execute_batch(
            "UPDATE images SET image_name = regexp_extract(image_rel_path, '[^/]+$'); \
             UPDATE objects SET image_name = regexp_extract(image_rel_path, '[^/]+$');",
        )
        .unwrap();
    let list = |images: Vec<&str>| {
        generator
            .get_object_list(&ListFilter {
                plane: plane(),
                images: crate::results::ImageSet::Listed(
                    images.into_iter().map(String::from).collect(),
                ),
                object_classes: None,
                columns: vec![Column::ImageName],
                with_coloc_details: false,
                related_to: None,
                page: no_page(),
                transpond_table: false,
                sort: None,
            })
            .unwrap()
            .source_object_count
    };
    assert_eq!(list(vec!["plate1/img1.tif"]), 2);
    assert_eq!(
        list(vec!["plate2/img1.tif"]),
        1,
        "same file name, other folder"
    );
    assert_eq!(list(vec!["plate1/img1.tif", "plate2/img1.tif"]), 3);
    assert_eq!(list(vec!["plate3/missing.tif"]), 0);
}

#[test]
fn get_object_list_class_filter_restricts_to_the_selected_class() {
    let generator = open(&[
        ObjectSpec::new("img1.tif", "ClassA", 1, 100),
        ObjectSpec::new("img1.tif", "ClassB", 2, 200),
    ]);
    let result = generator
        .get_object_list(&ListFilter {
            plane: plane(),
            images: crate::results::ImageSet::All,
            object_classes: Some(vec![ObjectClass::Global(2)]),
            columns: vec![Column::ObjectClass],
            with_coloc_details: false,
            related_to: None,
            page: no_page(),
            transpond_table: false,
            sort: None,
        })
        .unwrap();
    assert_eq!(result.source_object_count, 1);
}

#[test]
fn get_object_list_with_an_explicitly_empty_image_selection_is_empty() {
    let generator = open(&[ObjectSpec::new("img1.tif", "ClassA", 1, 100)]);
    let result = generator
        .get_object_list(&ListFilter {
            plane: plane(),
            images: crate::results::ImageSet::Listed(vec![]),
            object_classes: None,
            columns: vec![Column::ObjectClass],
            with_coloc_details: false,
            related_to: None,
            page: no_page(),
            transpond_table: false,
            sort: None,
        })
        .unwrap();
    assert_eq!(result.source_object_count, 0);
    assert!(result.rows.is_empty());
}

#[test]
fn get_object_list_plane_filter_excludes_objects_on_other_planes() {
    let generator = open(&[
        ObjectSpec::new("img1.tif", "ClassA", 1, 100),
        ObjectSpec::new("img1.tif", "ClassA", 1, 200).at_plane(1, 0),
    ]);
    let result = generator
        .get_object_list(&ListFilter {
            plane: PlaneFilter {
                z_stack: 1,
                t_stack: 0,
            },
            images: crate::results::ImageSet::All,
            object_classes: None,
            columns: vec![Column::AreaSizePx],
            with_coloc_details: false,
            related_to: None,
            page: no_page(),
            transpond_table: false,
            sort: None,
        })
        .unwrap();
    assert_eq!(result.source_object_count, 1);
}

/// A volume's object (z-planes 1 to 3) is on each of its planes, a
/// plane's object (z 2) only on its own.
#[test]
fn a_volumes_object_is_on_every_plane_it_covers() {
    let generator = open(&[
        ObjectSpec::new("img1.tif", "ClassA", 1, 100).at_plane(2, 0),
        ObjectSpec::new("img1.tif", "ClassA", 1, 200).in_volume(1, 3),
    ]);
    for (z, objects) in [(0, 0), (1, 1), (2, 2), (3, 1), (4, 0)] {
        let result = generator
            .get_object_list(&ListFilter {
                plane: PlaneFilter {
                    z_stack: z,
                    t_stack: 0,
                },
                images: crate::results::ImageSet::All,
                object_classes: None,
                columns: vec![Column::AreaSizePx],
                with_coloc_details: false,
                related_to: None,
                page: no_page(),
                transpond_table: false,
                sort: None,
            })
            .unwrap();
        assert_eq!(result.source_object_count, objects, "objects at Z {z}");
    }
}

/// The List shows one time point: objects of the others are not on
/// it - by object and per image.
#[test]
fn the_list_shows_only_the_time_point_asked_for() {
    let generator = open(&[
        ObjectSpec::new("img1.tif", "ClassA", 1, 100),
        ObjectSpec::new("img1.tif", "ClassA", 1, 200).at_plane(0, 1),
        ObjectSpec::new("img1.tif", "ClassA", 1, 300).at_plane(0, 2),
        ObjectSpec::new("img1.tif", "ClassA", 1, 400).at_plane(0, 2),
    ]);
    for (t, objects) in [(0, 1), (1, 1), (2, 2)] {
        let plane = PlaneFilter {
            z_stack: 0,
            t_stack: t,
        };
        let result = generator
            .get_object_list(&ListFilter {
                plane,
                images: crate::results::ImageSet::All,
                object_classes: None,
                columns: vec![Column::AreaSizePx],
                with_coloc_details: false,
                related_to: None,
                page: no_page(),
                transpond_table: false,
                sort: None,
            })
            .unwrap();
        assert_eq!(result.rows.len(), objects, "objects at T {t}");
        let grouped = generator
            .get_grouped_by_image(&GroupedByImageFilter {
                plane,
                images: crate::results::ImageSet::All,
                object_classes: None,
                columns: vec![Column::Count],
                aggregation: vec![Aggregation::Sum],
                page: no_page(),
                transpond_table: false,
                grouping: None,
                empty_classes: true,
                sort: None,
            })
            .unwrap();
        let count = grouped.rows[0]
            .iter()
            .find_map(|cell| match cell.value {
                CellValue::Integer(n) => Some(n as usize),
                CellValue::Float(n) => Some(n as usize),
                _ => None,
            })
            .unwrap();
        assert_eq!(count, objects, "count per image at T {t}");
    }
}

/// Every page of `filter`'s list as the List walks it (each page from
/// the last row of the one before): `(page start, row ids)` per page.
fn walk_pages(
    generator: &ResultsGenerator,
    filter: &ListFilter,
) -> Vec<(Option<String>, Vec<String>)> {
    let mut pages = Vec::new();
    let mut after = None;
    loop {
        let page = generator
            .get_object_list(&ListFilter {
                page: Pagination {
                    limit: filter.page.limit,
                    after: after.clone(),
                },
                ..filter.clone()
            })
            .unwrap();
        if page.row_names.is_empty() {
            return pages;
        }
        let next = page.row_names.last().cloned();
        pages.push((after, page.row_names));
        after = next;
    }
}

/// An object's page and the starts of the pages before it are exactly
/// what paging through the list from the first page finds - for every
/// object, with and without a class or image filter; an object the
/// list doesn't hold has none.
#[test]
fn an_objects_page_is_where_paging_through_the_list_finds_it() {
    let images = ["a.tif", "b.tif", "c.tif"];
    let objects: Vec<ObjectSpec> = (0..23u64)
        .map(|i| {
            let (name, id) = if i % 3 == 0 {
                ("ClassB", 2)
            } else {
                ("ClassA", 1)
            };
            ObjectSpec::new(images[(i % 3) as usize], name, id, i)
        })
        .collect();
    let generator = open(&objects);
    let all = ListFilter {
        plane: plane(),
        images: crate::results::ImageSet::All,
        object_classes: None,
        columns: vec![Column::AreaSizePx],
        with_coloc_details: false,
        related_to: None,
        page: Pagination {
            limit: 5,
            after: None,
        },
        transpond_table: false,
        sort: None,
    };
    let filters = [
        all.clone(),
        ListFilter {
            object_classes: Some(vec![ObjectClass::Global(1)]),
            ..all.clone()
        },
        ListFilter {
            images: crate::results::ImageSet::Listed(vec!["b.tif".into(), "c.tif".into()]),
            ..all.clone()
        },
        ListFilter {
            page: Pagination {
                limit: 1,
                after: None,
            },
            ..all.clone()
        },
        ListFilter {
            with_coloc_details: true,
            ..all.clone()
        },
    ];
    let every_id = walk_pages(&generator, &all)
        .into_iter()
        .flat_map(|(_, ids)| ids)
        .collect::<Vec<_>>();
    assert_eq!(every_id.len(), 23);
    for filter in &filters {
        let pages = walk_pages(&generator, filter);
        let starts: Vec<String> = pages
            .iter()
            .skip(1)
            .filter_map(|(s, _)| s.clone())
            .collect();
        for id in &every_id {
            let found = generator.object_list_page_starts(filter, id).unwrap();
            match pages.iter().position(|(_, ids)| ids.contains(id)) {
                Some(page) => {
                    assert_eq!(
                        found.as_deref(),
                        Some(&starts[..page]),
                        "object {id}, filter {:?} {:?}",
                        filter.object_classes,
                        filter.images
                    );
                }
                None => assert_eq!(found, None, "object {id} isn't listed"),
            }
        }
    }
    // A cursor left in the filter changes nothing.
    let with_cursor = ListFilter {
        page: Pagination {
            limit: 5,
            after: every_id.get(7).cloned(),
        },
        ..all.clone()
    };
    assert_eq!(
        generator
            .object_list_page_starts(&with_cursor, &every_id[3])
            .unwrap(),
        Some(vec![])
    );
    // An unknown object, a transposed table: none.
    let unknown = "ffffffff-ffff-ffff-ffff-fffffffffffe";
    assert!(!every_id.iter().any(|id| id == unknown));
    assert_eq!(
        generator.object_list_page_starts(&all, unknown).unwrap(),
        None
    );
    let transposed = ListFilter {
        transpond_table: true,
        ..all.clone()
    };
    assert_eq!(
        generator
            .object_list_page_starts(&transposed, &every_id[0])
            .unwrap(),
        None
    );
}

#[test]
fn get_object_list_pagination_cursor_returns_the_next_page() {
    let objects: Vec<ObjectSpec> = (0..5)
        .map(|i| ObjectSpec::new("img1.tif", "ClassA", 1, i))
        .collect();
    let generator = open(&objects);

    let first_page = generator
        .get_object_list(&ListFilter {
            plane: plane(),
            images: crate::results::ImageSet::All,
            object_classes: None,
            columns: vec![Column::AreaSizePx],
            with_coloc_details: false,
            related_to: None,
            page: Pagination {
                limit: 2,
                after: None,
            },
            transpond_table: false,
            sort: None,
        })
        .unwrap();
    assert_eq!(first_page.rows.len(), 2);
    let cursor = first_page.row_names.last().cloned();

    let second_page = generator
        .get_object_list(&ListFilter {
            plane: plane(),
            images: crate::results::ImageSet::All,
            object_classes: None,
            columns: vec![Column::AreaSizePx],
            with_coloc_details: false,
            related_to: None,
            page: Pagination {
                limit: 2,
                after: cursor.clone(),
            },
            transpond_table: false,
            sort: None,
        })
        .unwrap();
    assert_eq!(second_page.rows.len(), 2);
    assert_ne!(
        first_page.row_names, second_page.row_names,
        "second page must not repeat the first page's rows"
    );
    assert!(
        !second_page.row_names.contains(cursor.as_ref().unwrap()),
        "the cursor row itself must not repeat on the next page"
    );
}

#[test]
fn get_object_list_intensity_column_reads_the_seeded_channel_value() {
    use super::super::test_support::CH0_INTENSITIES_JSON;
    let generator = open(&[
        ObjectSpec::new("img1.tif", "ClassA", 1, 100).with_intensities(CH0_INTENSITIES_JSON)
    ]);
    let result = generator
        .get_object_list(&ListFilter {
            plane: plane(),
            images: crate::results::ImageSet::All,
            object_classes: None,
            columns: vec![Column::IntensityAvg(0)],
            with_coloc_details: false,
            related_to: None,
            page: no_page(),
            transpond_table: false,
            sort: None,
        })
        .unwrap();
    assert_eq!(result.rows.len(), 1);
    match &result.rows[0][0].value {
        CellValue::Float(v) => assert_eq!(*v, 127.0),
        _ => panic!("expected a float cell"),
    }
}

// -- get_grouped_by_image ---------------------------------------------

#[test]
fn get_grouped_by_image_averages_the_selected_column_per_image_and_class() {
    let generator = open(&[
        ObjectSpec::new("img1.tif", "ClassA", 1, 10),
        ObjectSpec::new("img1.tif", "ClassA", 1, 20),
        ObjectSpec::new("img2.tif", "ClassA", 1, 100),
    ]);
    let result = generator
        .get_grouped_by_image(&GroupedByImageFilter {
            plane: plane(),
            images: crate::results::ImageSet::All,
            object_classes: None,
            columns: vec![Column::AreaSizePx],
            aggregation: vec![Aggregation::Avg],
            page: no_page(),
            transpond_table: false,
            grouping: None,
            empty_classes: true,
            sort: None,
        })
        .unwrap();

    assert_eq!(result.rows.len(), 2, "one row per (image, class) group");
    assert_eq!(result.column_names[0], "image");
    assert_eq!(result.column_names[1], "class");
    assert_eq!(result.column_names[2], "Area [px] (AVG)");
}

/// A row per image, also with the classes side by side: every cell of a
/// disabled image's row is flagged, like a disabled image's objects.
#[test]
fn get_grouped_by_image_flags_a_disabled_images_rows() {
    let generator = open(&[
        ObjectSpec::new("img1.tif", "ClassA", 1, 10),
        ObjectSpec::new("img2.tif", "ClassA", 1, 100),
    ]);
    generator.enable_image("img2.tif", true).unwrap();
    for transpond_table in [false, true] {
        let result = generator
            .get_grouped_by_image(&GroupedByImageFilter {
                plane: plane(),
                images: crate::results::ImageSet::All,
                object_classes: None,
                columns: vec![Column::AreaSizePx],
                aggregation: vec![Aggregation::Avg],
                page: no_page(),
                transpond_table,
                grouping: None,
                empty_classes: true,
                sort: None,
            })
            .unwrap();
        let flags: Vec<(String, bool)> = result
            .rows
            .iter()
            .zip(&result.row_locations)
            .map(|(row, (path, _))| (path.clone(), row.iter().all(|cell| cell.disabled)))
            .collect();
        assert_eq!(
            flags,
            [
                ("img1.tif".to_string(), false),
                ("img2.tif".to_string(), true)
            ],
            "transposed: {transpond_table}"
        );
        assert!(
            result.rows[0].iter().all(|cell| !cell.disabled),
            "img1's cells aren't flagged"
        );
    }
}

#[test]
fn get_grouped_by_image_aggregates_intensity_columns() {
    use super::super::test_support::CH0_INTENSITIES_JSON;
    let generator = open(&[
        ObjectSpec::new("img1.tif", "ClassA", 1, 10).with_intensities(CH0_INTENSITIES_JSON),
        ObjectSpec::new("img1.tif", "ClassA", 1, 20).with_intensities(CH0_INTENSITIES_JSON),
    ]);
    let result = generator
        .get_grouped_by_image(&GroupedByImageFilter {
            plane: plane(),
            images: crate::results::ImageSet::All,
            object_classes: None,
            columns: vec![
                Column::IntensityAvg(0),
                Column::IntensitySum(0),
                Column::IntensityMin(0),
                Column::IntensityMax(0),
            ],
            aggregation: vec![Aggregation::Avg, Aggregation::Sum],
            page: no_page(),
            transpond_table: false,
            grouping: None,
            empty_classes: true,
            sort: None,
        })
        .unwrap();

    assert_eq!(result.rows.len(), 1);
    // Columns are sorted (Avg, Sum, Min, Max), each x (Avg, Sum).
    assert_eq!(result.column_names[2], "Avg Intensity (Ch 0) (AVG)");
    let values: Vec<Option<f64>> = result.rows[0][2..].iter().map(cell_value).collect();
    assert_eq!(
        values,
        vec![
            Some(127.0),
            Some(254.0),
            Some(255.0),
            Some(510.0),
            Some(0.0),
            Some(0.0),
            Some(255.0),
            Some(510.0),
        ]
    );
}

/// `Column::is_aggregable` (what the GUI's Images mode and the export
/// filter the selected columns with) must accept exactly the columns
/// the aggregating queries can handle - intensities were once dropped
/// there although `column_aggregate_expr` already supported them.
#[test]
fn column_is_aggregable_matches_what_the_aggregating_queries_accept() {
    let every_column = [
        Column::ObjectId,
        Column::TrackId,
        Column::ImageName,
        Column::ObjectClass,
        Column::Count,
        Column::AreaSizePx,
        Column::AreaSizeNm,
        Column::PerimeterPx,
        Column::PerimeterNm,
        Column::Circularity,
        Column::Solidity,
        Column::Eccentricity,
        Column::ColocCount(ObjectClass::Global(1)),
        Column::IntensityAvg(0),
        Column::IntensitySum(1),
        Column::IntensityMin(2),
        Column::IntensityMax(3),
    ];
    for column in &every_column {
        let accepted = aggregate_sql(column, &Aggregation::Avg).is_ok();
        assert_eq!(column.is_aggregable(), accepted, "{column:?}");
    }
}

// -- transposed (classes side by side) -------------------------------

fn grouped_transposed(
    generator: &ResultsGenerator,
    object_classes: Option<Vec<ObjectClass>>,
    page: Pagination,
) -> DatabaseResult {
    generator
        .get_grouped_by_image(&GroupedByImageFilter {
            plane: plane(),
            images: crate::results::ImageSet::All,
            object_classes,
            columns: vec![Column::Count, Column::AreaSizePx],
            aggregation: vec![Aggregation::Avg],
            page,
            transpond_table: true,
            grouping: None,
            empty_classes: true,
            sort: None,
        })
        .unwrap()
}

fn cell_value(cell: &Cell) -> Option<f64> {
    match &cell.value {
        CellValue::Float(v) => Some(*v as f64),
        CellValue::Empty => None,
        _ => panic!("expected a float or empty cell"),
    }
}

#[test]
fn transposed_grouped_puts_each_class_of_an_image_in_one_row() {
    let generator = open(&[
        ObjectSpec::new("img1.tif", "ClassA", 1, 10),
        ObjectSpec::new("img1.tif", "ClassA", 1, 20),
        ObjectSpec::new("img1.tif", "ClassB", 2, 100),
        ObjectSpec::new("img2.tif", "ClassA", 1, 50),
    ]);
    let result = grouped_transposed(&generator, None, no_page());

    assert_eq!(
        result.column_names,
        [
            "image",
            "Count (COUNT) (ClassA)",
            "Area [px] (AVG) (ClassA)",
            "Count (COUNT) (ClassB)",
            "Area [px] (AVG) (ClassB)",
        ]
    );
    assert_eq!(result.rows.len(), 2, "one row per image");
    let values = |row: &[Cell]| row[1..].iter().map(cell_value).collect::<Vec<_>>();
    assert_eq!(
        values(&result.rows[0]),
        [Some(2.0), Some(15.0), Some(1.0), Some(100.0)]
    );
    // img2 has no ClassB objects: no count of 0 pretending to be data
    // for the average, just empty.
    assert_eq!(values(&result.rows[1])[..2], [Some(1.0), Some(50.0)]);
    assert_eq!(values(&result.rows[1])[3], None);
}

#[test]
fn transposed_grouped_matches_the_normal_view_value_for_value() {
    let generator = open(&[
        ObjectSpec::new("img1.tif", "ClassA", 1, 10),
        ObjectSpec::new("img1.tif", "ClassB", 2, 30),
        ObjectSpec::new("img1.tif", "ClassB", 2, 50),
        ObjectSpec::new("img2.tif", "ClassB", 2, 70),
    ]);
    let normal = generator
        .get_grouped_by_image(&GroupedByImageFilter {
            plane: plane(),
            images: crate::results::ImageSet::All,
            object_classes: None,
            columns: vec![Column::Count, Column::AreaSizePx],
            aggregation: vec![Aggregation::Avg],
            page: no_page(),
            transpond_table: false,
            grouping: None,
            empty_classes: true,
            sort: None,
        })
        .unwrap();
    let transposed = grouped_transposed(&generator, None, no_page());
    // Every (image, class) row of the normal view appears in the
    // transposed row of its image, in its class' block.
    for row in &normal.rows {
        let image = match &row[0].value {
            CellValue::String(s) => s.clone(),
            _ => unreachable!(),
        };
        let block = match &row[1].value {
            CellValue::Class((name, _)) if name == "ClassA" => 0,
            _ => 1,
        };
        let wide = transposed
            .rows
            .iter()
            .find(|r| matches!(&r[0].value, CellValue::String(s) if *s == image))
            .unwrap();
        for (i, cell) in row[2..].iter().enumerate() {
            assert_eq!(cell_value(cell), cell_value(&wide[1 + block * 2 + i]));
        }
    }
}

#[test]
fn transposed_grouped_pages_by_image_and_respects_the_class_filter() {
    let generator = open(&[
        ObjectSpec::new("img1.tif", "ClassA", 1, 10),
        ObjectSpec::new("img2.tif", "ClassB", 2, 20),
        ObjectSpec::new("img3.tif", "ClassA", 1, 30),
    ]);
    let first = grouped_transposed(
        &generator,
        None,
        Pagination {
            limit: 2,
            after: None,
        },
    );
    assert_eq!(first.row_names, ["img1.tif", "img2.tif"]);
    let next = grouped_transposed(
        &generator,
        None,
        Pagination {
            limit: 2,
            after: first.row_names.last().cloned(),
        },
    );
    assert_eq!(next.row_names, ["img3.tif"]);

    let only_a = grouped_transposed(&generator, Some(vec![ObjectClass::Global(1)]), no_page());
    assert_eq!(only_a.column_names.len(), 1 + 2, "image + one class block");
    // img2 has no ClassA objects, but was analysed: it keeps its row,
    // with a ClassA count of 0 and no average area.
    assert_eq!(only_a.row_names, ["img1.tif", "img2.tif", "img3.tif"]);
    assert_eq!(cell_value(&only_a.rows[1][1]), Some(0.0));
    assert!(matches!(only_a.rows[1][2].value, CellValue::Empty));
}

fn list_transposed(
    generator: &ResultsGenerator,
    object_classes: Option<Vec<ObjectClass>>,
    page: Pagination,
) -> DatabaseResult {
    generator
        .get_object_list(&ListFilter {
            plane: plane(),
            images: crate::results::ImageSet::All,
            object_classes,
            columns: vec![Column::ImageName, Column::ObjectClass, Column::AreaSizePx],
            with_coloc_details: false,
            related_to: None,
            page,
            transpond_table: true,
            sort: None,
        })
        .unwrap()
}

fn area(cell: &Cell) -> Option<u64> {
    match &cell.value {
        CellValue::Integer(v) => Some(*v as u64),
        CellValue::Float(v) => Some(*v as u64),
        CellValue::Empty => None,
        other => panic!(
            "unexpected area cell {:?}",
            matches!(other, CellValue::String(_))
        ),
    }
}

#[test]
fn transposed_list_puts_the_classes_of_an_image_side_by_side() {
    // ObjectSpec ids are assigned in seeding order, so within an image
    // and class the n-th seeded object is the n-th row.
    let generator = open(&[
        ObjectSpec::new("img1.tif", "ClassA", 1, 11),
        ObjectSpec::new("img1.tif", "ClassA", 1, 12),
        ObjectSpec::new("img1.tif", "ClassA", 1, 13),
        ObjectSpec::new("img1.tif", "ClassB", 2, 21),
        ObjectSpec::new("img2.tif", "ClassB", 2, 22),
    ]);
    let result = list_transposed(&generator, None, no_page());

    assert_eq!(
        result.column_names,
        ["image", "Area [px] (ClassA)", "Area [px] (ClassB)"]
    );
    let rows: Vec<(String, Option<u64>, Option<u64>)> = result
        .rows
        .iter()
        .map(|row| {
            let image = match &row[0].value {
                CellValue::String(s) => s.clone(),
                _ => unreachable!(),
            };
            (image, area(&row[1]), area(&row[2]))
        })
        .collect();
    assert_eq!(
        rows,
        [
            ("img1.tif".into(), Some(11), Some(21)),
            ("img1.tif".into(), Some(12), None),
            ("img1.tif".into(), Some(13), None),
            ("img2.tif".into(), None, Some(22)),
        ]
    );
    assert_eq!(result.row_locations.len(), 4, "every row can be clicked");
    assert_eq!(result.source_object_count, 5);
}

#[test]
fn transposed_list_pages_without_gaps_or_repeats_across_images() {
    let mut specs = Vec::new();
    for (image, n) in [("img1.tif", 3), ("img2.tif", 2), ("img3.tif", 4)] {
        for i in 0..n {
            specs.push(ObjectSpec::new(image, "ClassA", 1, 100 + i));
            specs.push(ObjectSpec::new(image, "ClassB", 2, 200 + i));
        }
    }
    let generator = open(&specs);
    let all = list_transposed(&generator, None, no_page());
    assert_eq!(all.rows.len(), 9);

    let mut paged = Vec::new();
    let mut after = None;
    loop {
        let page = list_transposed(&generator, None, Pagination { limit: 2, after });
        if page.rows.is_empty() {
            break;
        }
        after = page.row_names.last().cloned();
        paged.extend(page.row_names);
    }
    assert_eq!(paged, all.row_names);
}

#[test]
fn transposed_list_with_a_class_filter_shows_only_that_block() {
    let generator = open(&[
        ObjectSpec::new("img1.tif", "ClassA", 1, 11),
        ObjectSpec::new("img1.tif", "ClassB", 2, 21),
        ObjectSpec::new("img1.tif", "ClassB", 2, 22),
    ]);
    let result = list_transposed(&generator, Some(vec![ObjectClass::Global(2)]), no_page());
    assert_eq!(result.column_names, ["image", "Area [px] (ClassB)"]);
    assert_eq!(result.rows.len(), 2);
}

#[test]
fn get_grouped_by_image_with_no_columns_selected_is_empty() {
    let generator = open(&[ObjectSpec::new("img1.tif", "ClassA", 1, 10)]);
    let result = generator
        .get_grouped_by_image(&GroupedByImageFilter {
            plane: plane(),
            images: crate::results::ImageSet::All,
            object_classes: None,
            columns: vec![],
            aggregation: vec![Aggregation::Avg],
            page: no_page(),
            transpond_table: false,
            grouping: None,
            empty_classes: true,
            sort: None,
        })
        .unwrap();
    assert!(result.rows.is_empty());
}

#[test]
fn get_grouped_by_image_with_an_explicitly_empty_image_selection_is_empty() {
    let generator = open(&[ObjectSpec::new("img1.tif", "ClassA", 1, 10)]);
    let result = generator
        .get_grouped_by_image(&GroupedByImageFilter {
            plane: plane(),
            images: crate::results::ImageSet::Listed(vec![]),
            object_classes: None,
            columns: vec![Column::AreaSizePx],
            aggregation: vec![Aggregation::Avg],
            page: no_page(),
            transpond_table: false,
            grouping: None,
            empty_classes: true,
            sort: None,
        })
        .unwrap();
    assert!(result.rows.is_empty());
}

// -- get_group_by_plate -------------------------------------------------

#[test]
fn get_group_by_plate_groups_by_the_default_regex_and_averages_per_well() {
    // Default regex expects `<well>_<field>.<ext>`, e.g. "A1_01.tif".
    let generator = open(&[
        ObjectSpec::new("A1_01.tif", "ClassA", 1, 10),
        ObjectSpec::new("A1_02.tif", "ClassA", 1, 20),
        ObjectSpec::new("B2_01.tif", "ClassA", 1, 100),
    ]);
    let result = generator
        .get_group_by_plate(
            &PlateFilter {
                plane: plane(),
                grouping: Grouping::Auto,
                aggregation: Aggregation::Avg,
                object_class: ObjectClass::Unset,
                column: Column::AreaSizePx,
                color_schema: ColorSchema::default(),
                color_scale: ColorScale::default(),
                plate_size: PlateSize::Auto,
            },
            &View::List,
        )
        .unwrap();

    assert_eq!(result.row_names.len(), 2);
    let a1_idx = result
        .row_names
        .iter()
        .position(|n| n == "A1")
        .expect("A1 group");
    let b2_idx = result
        .row_names
        .iter()
        .position(|n| n == "B2")
        .expect("B2 group");
    let cell_f64 = |cell: &Cell| match &cell.value {
        CellValue::Float(v) => *v as f64,
        _ => panic!("expected a float cell"),
    };
    assert_eq!(cell_f64(&result.rows[a1_idx][1]), 15.0, "avg(10, 20)");
    assert_eq!(cell_f64(&result.rows[b2_idx][1]), 100.0);
}

#[test]
fn get_group_by_plate_groups_by_the_default_regex_and_averages_per_well_count() {
    // Default regex expects `<well>_<field>.<ext>`, e.g. "A1_01.tif".
    let generator = open(&[
        ObjectSpec::new("A1_01.tif", "ClassA", 1, 10),
        ObjectSpec::new("A1_01.tif", "ClassA", 1, 8),
        ObjectSpec::new("A1_01.tif", "ClassA", 1, 6),
        ObjectSpec::new("A1_02.tif", "ClassA", 1, 20),
        ObjectSpec::new("B2_01.tif", "ClassA", 1, 100),
        ObjectSpec::new("B2_02.tif", "ClassA", 1, 50),
        ObjectSpec::new("B2_02.tif", "ClassA", 1, 50),
    ]);
    let result = generator
        .get_group_by_plate(
            &PlateFilter {
                plane: plane(),
                grouping: Grouping::Auto,
                aggregation: Aggregation::Avg,
                object_class: ObjectClass::Unset,
                column: Column::Count,
                color_schema: ColorSchema::default(),
                color_scale: ColorScale::default(),
                plate_size: PlateSize::Auto,
            },
            &View::List,
        )
        .unwrap();

    assert_eq!(result.row_names.len(), 2);
    let a1_idx = result
        .row_names
        .iter()
        .position(|n| n == "A1")
        .expect("A1 group");
    let b2_idx = result
        .row_names
        .iter()
        .position(|n| n == "B2")
        .expect("B2 group");
    let cell_f64 = |cell: &Cell| match &cell.value {
        CellValue::Float(v) => *v as f64,
        _ => panic!("expected a float cell"),
    };
    assert_eq!(cell_f64(&result.rows[a1_idx][1]), 2.0, "average per image");
    assert_eq!(cell_f64(&result.rows[b2_idx][1]), 1.5);
}

#[test]
fn get_group_by_plate_groups_by_the_default_regex_and_sums_per_well_count() {
    // Default regex expects `<well>_<field>.<ext>`, e.g. "A1_01.tif".
    let generator = open(&[
        ObjectSpec::new("A1_01.tif", "ClassA", 1, 10),
        ObjectSpec::new("A1_01.tif", "ClassA", 1, 8),
        ObjectSpec::new("A1_01.tif", "ClassA", 1, 6),
        ObjectSpec::new("A1_02.tif", "ClassA", 1, 20),
        ObjectSpec::new("B2_01.tif", "ClassA", 1, 100),
        ObjectSpec::new("B2_02.tif", "ClassA", 1, 50),
        ObjectSpec::new("B2_02.tif", "ClassA", 1, 50),
    ]);
    let result = generator
        .get_group_by_plate(
            &PlateFilter {
                plane: plane(),
                grouping: Grouping::Auto,
                aggregation: Aggregation::Sum,
                object_class: ObjectClass::Unset,
                column: Column::Count,
                color_schema: ColorSchema::default(),
                color_scale: ColorScale::default(),
                plate_size: PlateSize::Auto,
            },
            &View::List,
        )
        .unwrap();

    assert_eq!(result.row_names.len(), 2);
    let a1_idx = result
        .row_names
        .iter()
        .position(|n| n == "A1")
        .expect("A1 group");
    let b2_idx = result
        .row_names
        .iter()
        .position(|n| n == "B2")
        .expect("B2 group");
    let cell_f64 = |cell: &Cell| match &cell.value {
        CellValue::Float(v) => *v as f64,
        _ => panic!("expected a float cell"),
    };
    assert_eq!(cell_f64(&result.rows[a1_idx][1]), 4.0);
    assert_eq!(cell_f64(&result.rows[b2_idx][1]), 3.0);
}

/// An image that was analyzed but produced zero objects (e.g. an empty
/// well) must still show up as its own group rather than being silently
/// absent from the matrix, since "absent" and "present but empty" mean
/// different things in a plate view. The row survives *and* its value
/// cell is `CellValue::Empty` (not a misleading `0.0`), same convention
/// as `Heatmap` and as the export's own "None"-for-empty-cell text.
#[test]
fn get_group_by_plate_includes_a_well_with_no_objects_as_an_empty_group() {
    let generator = open(&[ObjectSpec::new("A1_01.tif", "ClassA", 1, 10)]);
    generator
        .database
        .execute(
            "INSERT INTO images (image_name, image_rel_path, width, height, c_stacks, z_stacks, t_stacks) \
             VALUES ('B2_01.tif', 'B2_01.tif', 100, 100, 1, 1, 1)",
            [],
        )
        .unwrap();

    let result = generator
        .get_group_by_plate(
            &PlateFilter {
                plane: plane(),
                grouping: Grouping::Auto,
                aggregation: Aggregation::Avg,
                object_class: ObjectClass::Unset,
                column: Column::AreaSizePx,
                color_schema: ColorSchema::default(),
                color_scale: ColorScale::default(),
                plate_size: PlateSize::Auto,
            },
            &View::List,
        )
        .unwrap();

    assert_eq!(result.row_names.len(), 2, "B2 must still appear");
    let b2_idx = result
        .row_names
        .iter()
        .position(|n| n == "B2")
        .expect("B2 group");
    assert!(
        matches!(result.rows[b2_idx][1].value, CellValue::Empty),
        "a well with no objects has no average to show"
    );

    // The Heatmap view must place a real, selectable (empty) tile at
    // B2's grid position - not skip it as if no well were there at all.
    let heatmap = generator
        .get_group_by_plate(
            &PlateFilter {
                plane: plane(),
                grouping: Grouping::Auto,
                aggregation: Aggregation::Avg,
                object_class: ObjectClass::Unset,
                column: Column::AreaSizePx,
                color_schema: ColorSchema::default(),
                color_scale: ColorScale::default(),
                plate_size: PlateSize::Auto,
            },
            &View::Heatmap,
        )
        .unwrap();
    // A1 -> (row 0, col 0), B2 -> (row 1, col 1).
    assert!(
        matches!(heatmap.rows[1][1].value, CellValue::Empty),
        "B2 has no objects, so no number to show"
    );
    assert!(
        heatmap.rows[1][1].search_key.is_some(),
        "B2 must still be a real, clickable well"
    );
    assert!(
        heatmap.rows[1][2].search_key.is_none(),
        "position (1, 2) has no well at all and must stay non-clickable"
    );
}

/// A disabled image's objects must not contribute to its well's
/// aggregate, but the well itself is never dropped - even one made up
/// only of disabled images still appears (as an empty cell, same as any
/// other well with no contributing objects). The well itself is never
/// flagged `Cell::disabled` though - only individual images are - so a
/// well made up only of disabled images renders the same as any other
/// empty well.
#[test]
fn get_group_by_plate_ignores_objects_from_a_disabled_image() {
    let generator = open(&[
        ObjectSpec::new("A1_01.tif", "ClassA", 1, 10),
        ObjectSpec::new("B2_01.tif", "ClassA", 1, 100),
    ]);
    generator.enable_image("B2_01.tif", true).unwrap();

    let result = generator
        .get_group_by_plate(
            &PlateFilter {
                plane: plane(),
                grouping: Grouping::Auto,
                aggregation: Aggregation::Avg,
                object_class: ObjectClass::Unset,
                column: Column::AreaSizePx,
                color_schema: ColorSchema::default(),
                color_scale: ColorScale::default(),
                plate_size: PlateSize::Auto,
            },
            &View::List,
        )
        .unwrap();

    assert_eq!(
        result.row_names,
        vec!["A1".to_string(), "B2".to_string()],
        "B2 must still appear even though every one of its images is disabled"
    );
    let a1_idx = result.row_names.iter().position(|n| n == "A1").unwrap();
    let b2_idx = result.row_names.iter().position(|n| n == "B2").unwrap();
    assert!(
        !result.rows[a1_idx][1].disabled,
        "A1 has no disabled images"
    );
    assert!(
        !result.rows[b2_idx][1].disabled,
        "the well itself is never flagged disabled, only individual images are"
    );
    assert!(
        matches!(result.rows[b2_idx][1].value, CellValue::Empty),
        "B2's disabled image must not contribute to the average, leaving no value to show"
    );
}

/// The mixed case: a well with both an enabled and a disabled image
/// must still average only the enabled one's objects, but the well
/// itself must not be flagged `disabled` - only individual images are,
/// so the well keeps its normal heatmap color.
#[test]
fn get_group_by_plate_flags_a_well_with_a_mix_of_enabled_and_disabled_images() {
    let generator = open(&[
        ObjectSpec::new("A1_01.tif", "ClassA", 1, 10),
        ObjectSpec::new("A1_02.tif", "ClassA", 1, 1000),
    ]);
    generator.enable_image("A1_02.tif", true).unwrap();

    let result = generator
        .get_group_by_plate(
            &PlateFilter {
                plane: plane(),
                grouping: Grouping::Auto,
                aggregation: Aggregation::Avg,
                object_class: ObjectClass::Unset,
                column: Column::AreaSizePx,
                color_schema: ColorSchema::default(),
                color_scale: ColorScale::default(),
                plate_size: PlateSize::Auto,
            },
            &View::List,
        )
        .unwrap();

    assert_eq!(result.row_names, vec!["A1".to_string()]);
    assert!(
        !result.rows[0][1].disabled,
        "the well itself must not be flagged disabled just because one of its images is"
    );
    assert!(
        result.rows[0][1].any_disabled,
        "the well must still be flagged any_disabled so the GUI can badge it"
    );
    assert!(
        matches!(result.rows[0][1].value, CellValue::Float(v) if v == 10.0),
        "only the enabled image's objects (10) must be averaged, not the disabled one's 1000"
    );

    let heatmap = generator
        .get_group_by_plate(
            &PlateFilter {
                plane: plane(),
                grouping: Grouping::Auto,
                aggregation: Aggregation::Avg,
                object_class: ObjectClass::Unset,
                column: Column::AreaSizePx,
                color_schema: ColorSchema::default(),
                color_scale: ColorScale::default(),
                plate_size: PlateSize::Auto,
            },
            &View::Heatmap,
        )
        .unwrap();
    let a1_cell = &heatmap.rows[0][0];
    assert!(
        !a1_cell.disabled,
        "the heatmap's well cell must not be flagged disabled either"
    );
    assert!(
        a1_cell.any_disabled,
        "the heatmap's well cell must carry any_disabled too, for the same corner badge"
    );
}

// -- get_images ---------------------------------------------------------

#[test]
fn get_images_returns_every_seeded_image_sorted_by_name() {
    let generator = open(&[
        ObjectSpec::new("b.tif", "ClassA", 1, 10),
        ObjectSpec::new("a.tif", "ClassA", 1, 10),
    ]);
    let images = generator.get_images().unwrap();
    let names: Vec<String> = images.iter().map(|i| i.name.clone()).collect();
    assert_eq!(names, vec!["a.tif".to_string(), "b.tif".to_string()]);
    assert!(images.iter().all(|i| !i.disabled));
}

#[test]
fn enable_image_toggles_the_disabled_flag() {
    let generator = open(&[ObjectSpec::new("a.tif", "ClassA", 1, 10)]);
    assert!(!generator.get_images().unwrap()[0].disabled);

    generator.enable_image("a.tif", true).unwrap();
    assert!(generator.get_images().unwrap()[0].disabled);

    generator.enable_image("a.tif", false).unwrap();
    assert!(!generator.get_images().unwrap()[0].disabled);
}

/// Left out in the results it is the user's (`disabled_by` results),
/// with the reason given then; taken back in, both go. An image the run
/// left out keeps who and why when it is left out again.
#[test]
fn leaving_out_in_the_results_records_who_and_keeps_the_reason() {
    let generator = open(&[ObjectSpec::new("a.tif", "ClassA", 1, 10)]);
    generator.enable_image("a.tif", true).unwrap();
    generator
        .change_image_review(
            "a.tif",
            &ImageReviewChange::LeftOutReason(Some("Empty".into())),
        )
        .unwrap();
    let image = &generator.get_images().unwrap()[0];
    assert_eq!(image.disabled_by.as_deref(), Some("results"));
    assert_eq!(image.disabled_reason.as_deref(), Some("Empty"));

    generator.enable_image("a.tif", true).unwrap();
    let image = &generator.get_images().unwrap()[0];
    assert_eq!(
        image.disabled_reason.as_deref(),
        Some("Empty"),
        "already out"
    );

    generator.enable_image("a.tif", false).unwrap();
    let image = &generator.get_images().unwrap()[0];
    assert_eq!(
        (image.disabled_by.clone(), image.disabled_reason.clone()),
        (None, None)
    );
    generator
        .change_image_review(
            "a.tif",
            &ImageReviewChange::LeftOutReason(Some("Empty".into())),
        )
        .unwrap();
    assert_eq!(
        generator.get_images().unwrap()[0].disabled_reason,
        None,
        "included"
    );
}

/// Note and marker change in the results only, and clear again.
#[test]
fn a_results_note_and_marker_can_be_changed_and_cleared() {
    let generator = open(&[ObjectSpec::new("a.tif", "ClassA", 1, 10)]);
    let image = &generator.get_images().unwrap()[0];
    assert_eq!((image.note.as_str(), image.marker), ("", None));
    generator
        .change_image_review("a.tif", &ImageReviewChange::Note("check focus".into()))
        .unwrap();
    generator
        .change_image_review("a.tif", &ImageReviewChange::Marker(Some(ImageMarker::Blue)))
        .unwrap();
    let image = &generator.get_images().unwrap()[0];
    assert_eq!(
        (image.note.as_str(), image.marker),
        ("check focus", Some(ImageMarker::Blue))
    );
    generator
        .change_image_review("a.tif", &ImageReviewChange::Note(String::new()))
        .unwrap();
    generator
        .change_image_review("a.tif", &ImageReviewChange::Marker(None))
        .unwrap();
    let image = &generator.get_images().unwrap()[0];
    assert_eq!((image.note.as_str(), image.marker), ("", None));
}

#[test]
fn enable_image_on_an_unknown_path_is_a_no_op() {
    let generator = open(&[ObjectSpec::new("a.tif", "ClassA", 1, 10)]);
    generator.enable_image("does-not-exist.tif", true).unwrap();
    assert!(!generator.get_images().unwrap()[0].disabled);
}

/// `try_clone` is the fix for a real Windows bug: exporting used to
/// reopen the `.evadb` file by path on a background thread while the
/// original `ResultsGenerator` connection was still open, which Windows
/// (unlike Linux/macOS) refuses - "used by another process", reporting
/// the app's own PID. `try_clone` gets a second, independent connection
/// to the *same already-open* database instead of reopening the file,
/// so this pins down that both connections stay live and see the same
/// data at the same time - the exact scenario a background export needs.
#[test]
fn try_clone_gives_an_independent_connection_to_the_same_live_database() {
    let generator = open(&[ObjectSpec::new("a.tif", "ClassA", 1, 10)]);
    let cloned = generator.try_clone().expect("clone connection");

    // Both connections are usable at the same time...
    assert_eq!(generator.get_images().unwrap().len(), 1);
    assert_eq!(cloned.get_images().unwrap().len(), 1);

    // ...and see the same underlying data, not two separate files: a
    // write through one is visible through the other.
    cloned.enable_image("a.tif", true).unwrap();
    assert!(
        generator.get_images().unwrap()[0].disabled,
        "the clone must share the original's already-open database, not a second file"
    );
}

// -- Column key/label round trips --------------------------------------

#[test]
fn column_as_key_and_from_key_round_trip_for_plain_columns() {
    let classes: Vec<Class> = vec![];
    for column in [
        Column::ObjectId,
        Column::ImageName,
        Column::ObjectClass,
        Column::Count,
        Column::AreaSizePx,
        Column::AreaSizeNm,
        Column::VolumePx,
        Column::VolumeNm,
        Column::SurfaceAreaNm,
        Column::PerimeterPx,
        Column::PerimeterNm,
        Column::Circularity,
        Column::Solidity,
        Column::Eccentricity,
    ] {
        let key = column.as_key(&classes);
        assert_eq!(Column::from_key(&key, &classes), Some(column));
    }
}

#[test]
fn column_as_key_and_from_key_round_trip_for_intensity_channels() {
    let classes: Vec<Class> = vec![];
    for column in [
        Column::IntensityAvg(0),
        Column::IntensitySum(1),
        Column::IntensityMin(2),
        Column::IntensityMax(3),
    ] {
        let key = column.as_key(&classes);
        assert_eq!(Column::from_key(&key, &classes), Some(column));
    }
}

#[test]
fn column_as_key_and_from_key_round_trip_for_coloc_count() {
    let classes = vec![Class {
        id: 1,
        name: "ClassA".to_string(),
        color: 0,
        notes: String::new(),
        results: Default::default(),
    }];
    let column = Column::ColocCount(ObjectClass::Global(1));
    let key = column.as_key(&classes);
    assert_eq!(key, "n_colocalized_class_ClassA");
    assert_eq!(Column::from_key(&key, &classes), Some(column));

    let unset = Column::ColocCount(ObjectClass::Unset);
    assert_eq!(unset.as_key(&classes), "n_colocalized_unset");
    assert_eq!(
        Column::from_key("n_colocalized_unset", &classes),
        Some(unset)
    );
}

#[test]
fn column_from_key_returns_none_for_an_unrecognized_key() {
    assert_eq!(Column::from_key("not_a_real_column", &[]), None);
}

#[test]
fn column_display_label_is_stable_and_distinct_from_its_key() {
    let classes: Vec<Class> = vec![];
    assert_eq!(Column::AreaSizePx.display_label(&classes), "Area [px]");
    assert_eq!(Column::AreaSizePx.as_key(&classes), "area_px");
    assert_eq!(
        Column::IntensityAvg(2).display_label(&classes),
        "Avg Intensity (Ch 2)"
    );
}

// -- class_display_label -------------------------------------------------

#[test]
fn class_display_label_falls_back_to_a_generic_name_for_an_unknown_class_id() {
    let classes: Vec<Class> = vec![];
    assert_eq!(
        class_display_label(ObjectClass::Global(7), &classes),
        "class 7"
    );
    assert_eq!(class_display_label(ObjectClass::Unset, &classes), "unset");
}

#[test]
fn class_display_label_uses_the_registered_class_name_when_known() {
    let classes = vec![Class {
        id: 1,
        name: "Nuclei".to_string(),
        color: 0,
        notes: String::new(),
        results: Default::default(),
    }];
    assert_eq!(
        class_display_label(ObjectClass::Global(1), &classes),
        "Nuclei"
    );
}

// -- get_available_columns -------------------------------------------------

/// Volume and surface are offered for results with a volume's objects,
/// not for planes only.
#[test]
fn the_volume_columns_are_offered_only_for_results_with_volumes() {
    let volume_columns = |objects: &[ObjectSpec]| -> Vec<Column> {
        open(objects)
            .get_available_columns()
            .unwrap()
            .into_iter()
            .map(|c| c.key)
            .filter(|c| {
                matches!(
                    c,
                    Column::VolumePx | Column::VolumeNm | Column::SurfaceAreaNm
                )
            })
            .collect()
    };
    assert_eq!(
        volume_columns(&[ObjectSpec::new("img1.tif", "ClassA", 1, 10)]),
        []
    );
    assert_eq!(
        volume_columns(&[
            ObjectSpec::new("img1.tif", "ClassA", 1, 10),
            ObjectSpec::new("img1.tif", "ClassA", 1, 10).in_volume(0, 2),
        ]),
        [Column::VolumePx, Column::VolumeNm, Column::SurfaceAreaNm]
    );
}

#[test]
fn get_available_columns_only_lists_coloc_count_for_classes_that_actually_colocalize() {
    let objects = vec![
        ObjectSpec::new("img1.tif", "ClassA", 1, 10)
            .with_coloc(r#"{"2":["00000000-0000-0000-0000-000000000001"]}"#),
        ObjectSpec::new("img1.tif", "ClassB", 2, 20),
    ];
    let generator = open(&objects);
    let columns = generator.get_available_columns().unwrap();
    let coloc_columns: Vec<&ColumnEntry> = columns
        .iter()
        .filter(|c| matches!(c.key, Column::ColocCount(_)))
        .collect();
    assert_eq!(coloc_columns.len(), 1);
}

// -- coloc combinations -------------------------------------------------

/// Class 1 objects (area 10..40) with partners of class 2 and 3: both,
/// only 2, only 3 (2 of them), none.
fn coloc_combinations() -> TestResults {
    let id = |n: u32| format!("00000000-0000-0000-0000-00000000000{n}");
    open(&[
        ObjectSpec::new("img1.tif", "ClassA", 1, 10).with_coloc(&format!(
            r#"{{"2":["{}"],"3":["{}","{}"]}}"#,
            id(5),
            id(6),
            id(7)
        )),
        ObjectSpec::new("img1.tif", "ClassA", 1, 20)
            .with_coloc(&format!(r#"{{"2":["{}"]}}"#, id(5))),
        ObjectSpec::new("img1.tif", "ClassA", 1, 30).with_coloc(&format!(
            r#"{{"3":["{}","{}"]}}"#,
            id(6),
            id(7)
        )),
        ObjectSpec::new("img1.tif", "ClassA", 1, 40),
        ObjectSpec::new("img1.tif", "ClassB", 2, 50),
        ObjectSpec::new("img1.tif", "ClassC", 3, 60),
        ObjectSpec::new("img1.tif", "ClassC", 3, 70),
    ])
}

/// The combinations of the fixture: (column, value per object by area,
/// their sum).
fn coloc_combination_cases() -> Vec<(Column, [i32; 4], f64)> {
    let b = ObjectClass::Global(2);
    let c = ObjectClass::Global(3);
    let at_least = |partners| Column::coloc(ColocMatch::AtLeast, vec![b, c], vec![], partners);
    let exactly = |with: Vec<ObjectClass>, partners| {
        Column::coloc(ColocMatch::Exactly, with, vec![b, c], partners)
    };
    vec![
        (
            Column::coloc(ColocMatch::AtLeast, vec![b], vec![], false),
            [1, 1, 0, 0],
            2.0,
        ),
        (Column::ColocCount(c), [2, 0, 2, 0], 4.0),
        (at_least(false), [1, 0, 0, 0], 1.0),
        (at_least(true), [3, 0, 0, 0], 3.0),
        (exactly(vec![b], false), [0, 1, 0, 0], 1.0),
        (exactly(vec![c], true), [0, 0, 2, 0], 2.0),
        (exactly(vec![b, c], false), [1, 0, 0, 0], 1.0),
        (exactly(vec![], false), [0, 0, 0, 1], 1.0),
    ]
}

#[test]
fn coloc_combinations_per_object() {
    let generator = coloc_combinations();
    for (column, expected, _) in coloc_combination_cases() {
        let result = generator
            .get_object_list(&ListFilter {
                plane: plane(),
                images: crate::results::ImageSet::All,
                object_classes: Some(vec![ObjectClass::Global(1)]),
                columns: vec![Column::AreaSizePx, column.clone()],
                with_coloc_details: false,
                related_to: None,
                page: no_page(),
                transpond_table: false,
                sort: None,
            })
            .unwrap();
        let mut values: Vec<(i32, i32)> = result
            .rows
            .iter()
            .map(|row| match (&row[0].value, &row[1].value) {
                (CellValue::Integer(area), CellValue::Integer(v)) => (*area, *v),
                _ => panic!("expected integers"),
            })
            .collect();
        values.sort();
        let got: Vec<i32> = values.iter().map(|(_, v)| *v).collect();
        assert_eq!(got, expected, "{column:?}");
    }
}

/// The SQL of a group (per image) counts as the list does per object.
#[test]
fn coloc_combinations_summed_per_image() {
    let generator = coloc_combinations();
    for (column, _, sum) in coloc_combination_cases() {
        let result = generator
            .get_grouped_by_image(&GroupedByImageFilter {
                plane: plane(),
                images: crate::results::ImageSet::All,
                object_classes: Some(vec![ObjectClass::Global(1)]),
                columns: vec![column.clone()],
                aggregation: vec![Aggregation::Sum],
                page: no_page(),
                transpond_table: false,
                grouping: None,
                empty_classes: true,
                sort: None,
            })
            .unwrap();
        assert_eq!(number(&result.rows[0][2]), Some(sum), "{column:?}");
    }
}

#[test]
fn the_combinations_that_occur_are_offered() {
    let generator = coloc_combinations();
    let b = ObjectClass::Global(2);
    let c = ObjectClass::Global(3);
    let sets: Vec<Vec<ObjectClass>> = generator
        .get_available_columns()
        .unwrap()
        .into_iter()
        .filter_map(|entry| match entry.key {
            Column::Coloc(ColocColumn {
                how: ColocMatch::Exactly,
                with,
                among,
                ..
            }) => {
                assert_eq!(among, vec![b, c]);
                Some(with)
            }
            _ => None,
        })
        .collect();
    assert_eq!(sets, vec![vec![b], vec![b, c], vec![c]]);
}

// -- with_coloc_details fan-out ----------------------------------------

#[test]
fn a_coloc_with_the_objects_own_class_has_no_value() {
    // A (class 1) overlaps B (class 2): B's "coloc with class 2" was
    // never measured - empty, not 0; A's is 1.
    let objects = vec![
        ObjectSpec::new("img1.tif", "ClassA", 1, 10)
            .with_coloc(r#"{"2":["00000000-0000-0000-0000-000000000001"]}"#),
        ObjectSpec::new("img1.tif", "ClassB", 2, 99),
    ];
    let generator = open(&objects);
    let result = generator
        .get_object_list(&ListFilter {
            plane: plane(),
            images: crate::results::ImageSet::All,
            object_classes: None,
            columns: vec![Column::ColocCount(ObjectClass::Global(2))],
            with_coloc_details: false,
            related_to: None,
            page: no_page(),
            transpond_table: false,
            sort: None,
        })
        .unwrap();
    let values: Vec<&CellValue> = result.rows.iter().map(|row| &row[0].value).collect();
    assert!(matches!(values[0], CellValue::Integer(1)));
    assert!(matches!(values[1], CellValue::Empty), "own class: no value");
}

#[test]
fn get_object_list_with_coloc_details_resolves_the_partners_own_metric() {
    // Object A colocalizes with class 2, partnered with object B (the
    // second seeded object, id index 1). Both a coloc-class column
    // (`ColocCount(2)`) and a resolvable metric (`AreaSizePx`) must be
    // selected together for `with_coloc_details` fan-out to activate
    // (see `is_resolvable_metric`/`details_active`).
    let objects = vec![
        ObjectSpec::new("img1.tif", "ClassA", 1, 10)
            .with_coloc(r#"{"2":["00000000-0000-0000-0000-000000000001"]}"#),
        ObjectSpec::new("img1.tif", "ClassB", 2, 99),
    ];
    let generator = open(&objects);

    let result = generator
        .get_object_list(&ListFilter {
            plane: plane(),
            images: crate::results::ImageSet::All,
            object_classes: None,
            columns: vec![
                Column::ColocCount(ObjectClass::Global(2)),
                Column::AreaSizePx,
            ],
            with_coloc_details: true,
            related_to: None,
            page: no_page(),
            transpond_table: false,
            sort: None,
        })
        .unwrap();

    assert_eq!(result.column_names.len(), 3);
    assert!(result.column_names[2].contains("coloc"));
    assert_eq!(result.rows.len(), 2, "one fanned-out row per source object");

    // Column order is `Column`'s declared `Ord` (`AreaSizePx` before
    // `ColocCount`), so column 0 is the object's own area, column 1 its
    // coloc count, column 2 the cross-resolved partner metric.
    let int_cell = |cell: &Cell| match &cell.value {
        CellValue::Integer(v) => *v,
        _ => panic!("expected an integer cell"),
    };
    let row_a = &result.rows[0];
    assert_eq!(int_cell(&row_a[0]), 10, "object A's own area");
    assert_eq!(
        int_cell(&row_a[1]),
        1,
        "object A colocalizes with exactly 1 class-2 object"
    );
    assert_eq!(
        int_cell(&row_a[2]),
        99,
        "resolved onto B, the colocalizing partner"
    );

    let row_b = &result.rows[1];
    assert_eq!(int_cell(&row_b[0]), 99, "object B's own area");
    assert!(
        matches!(row_b[1].value, CellValue::Empty),
        "object B is of class 2 itself: its coloc with class 2 has no value"
    );
    assert!(
        matches!(row_b[2].value, CellValue::String(ref s) if s == "-"),
        "no partner to resolve onto for B"
    );
}

// -- plate/well/heatmap grouping correctness & cross-view plausibility --
//
// These check that the same underlying data reported through different
// "views" of the same aggregate (List vs Heatmap; a single well's own
// query vs the batched every-well query used by exports; a single
// aggregation vs the multi-aggregation batch) always agree — exactly the
// kind of drift a hand-rolled SQL string per view/batch variant could
// silently introduce.

fn plate_filter(column: Column) -> PlateFilter {
    PlateFilter {
        plane: plane(),
        grouping: Grouping::Auto,
        aggregation: Aggregation::Avg,
        object_class: ObjectClass::Unset,
        column,
        color_schema: ColorSchema::default(),
        color_scale: ColorScale::default(),
        plate_size: PlateSize::Auto,
    }
}

fn well_filter(group_name: &str, column: Column) -> WellFilter {
    WellFilter {
        plane: plane(),
        group_name: group_name.to_string(),
        grouping: Grouping::Auto,
        aggregation: Aggregation::Avg,
        object_class: ObjectClass::Unset,
        column,
        color_schema: ColorSchema::default(),
        color_scale: ColorScale::default(),
        well_size: None,
        well_order: None,
    }
}

fn wells_batch_filter(column: Column) -> WellsBatchFilter {
    WellsBatchFilter {
        plane: plane(),
        grouping: Grouping::Auto,
        aggregation: Aggregation::Avg,
        object_class: ObjectClass::Unset,
        column,
        color_schema: ColorSchema::default(),
        color_scale: ColorScale::default(),
        well_size: None,
        well_order: None,
    }
}

fn float_cell(cell: &Cell) -> f64 {
    match &cell.value {
        CellValue::Float(v) => *v as f64,
        _ => panic!("expected a float cell"),
    }
}

#[test]
fn plate_list_and_plate_heatmap_report_the_same_values_at_matching_positions() {
    // Well A1 (row 0, col 0): avg(10, 20) = 15. Well B2 (row 1, col 1): 100.
    let generator = open(&[
        ObjectSpec::new("A1_01.tif", "ClassA", 1, 10),
        ObjectSpec::new("A1_02.tif", "ClassA", 1, 20),
        ObjectSpec::new("B2_01.tif", "ClassA", 1, 100),
    ]);
    let filter = plate_filter(Column::AreaSizePx);

    let list = generator.get_group_by_plate(&filter, &View::List).unwrap();
    let heatmap = generator
        .get_group_by_plate(&filter, &View::Heatmap)
        .unwrap();

    let list_value = |well: &str| {
        let idx = list.row_names.iter().position(|n| n == well).unwrap();
        float_cell(&list.rows[idx][1])
    };
    assert_eq!(list_value("A1"), 15.0);
    assert_eq!(list_value("B2"), 100.0);

    // Heatmap row_names are letters ("A", "B", ...), column_names are
    // 1-based numbers ("1", "2", ...) - A1 is heatmap[0][0], B2 is
    // heatmap[1][1].
    assert_eq!(heatmap.row_names[0], "A");
    assert_eq!(heatmap.column_names[0], "1");
    assert_eq!(float_cell(&heatmap.rows[0][0]), list_value("A1"));
    assert_eq!(float_cell(&heatmap.rows[1][1]), list_value("B2"));
    // Every other cell in range has no matching well - must stay empty,
    // not a stray 0.
    assert!(matches!(heatmap.rows[0][1].value, CellValue::Empty));
}

#[test]
fn plate_object_class_filter_restricts_the_grouped_objects() {
    let generator = open(&[
        ObjectSpec::new("A1_01.tif", "ClassA", 1, 10),
        ObjectSpec::new("A1_02.tif", "ClassB", 2, 1000),
    ]);
    let mut filter = plate_filter(Column::AreaSizePx);
    filter.object_class = ObjectClass::Global(1);
    let list = generator.get_group_by_plate(&filter, &View::List).unwrap();
    assert_eq!(list.row_names, vec!["A1".to_string()]);
    assert_eq!(
        float_cell(&list.rows[0][1]),
        10.0,
        "ClassB's 1000 must be excluded"
    );
}

#[test]
fn well_list_and_well_heatmap_report_the_same_values_at_matching_positions() {
    // Field "01" -> position 0 -> heatmap (row 0, col 0); field "02" ->
    // position 1 -> heatmap (row 0, col 1) (a 4x4 well, no
    // well_order: idx-1 read row-major).
    let generator = open(&[
        ObjectSpec::new("A1_01.tif", "ClassA", 1, 10),
        ObjectSpec::new("A1_02.tif", "ClassA", 1, 20),
    ]);
    let filter = WellFilter {
        well_size: Some(WellSize { rows: 4, cols: 4 }),
        ..well_filter("A1", Column::AreaSizePx)
    };

    let list = generator.get_group_by_well(&filter, &View::List).unwrap();
    let heatmap = generator
        .get_group_by_well(&filter, &View::Heatmap)
        .unwrap();

    assert_eq!(list.row_names, vec!["01".to_string(), "02".to_string()]);
    assert_eq!(float_cell(&list.rows[0][1]), 10.0);
    assert_eq!(float_cell(&list.rows[1][1]), 20.0);

    assert_eq!(float_cell(&heatmap.rows[0][0]), 10.0);
    assert_eq!(float_cell(&heatmap.rows[0][1]), 20.0);
    assert!(matches!(heatmap.rows[1][0].value, CellValue::Empty));
}

// -- grouping / plate size / well layout -------------------------------

/// The heatmap value at each grid cell (`None` = empty or no field).
fn heatmap_values(result: &DatabaseResult) -> Vec<Vec<Option<f64>>> {
    result
        .rows
        .iter()
        .map(|row| {
            row.iter()
                .map(|cell| match cell.value {
                    CellValue::Float(v) => Some(v as f64),
                    _ => None,
                })
                .collect()
        })
        .collect()
}

#[test]
fn auto_grouping_detects_a_prefixed_well_pattern_with_field_letters() {
    let generator = open(&[
        ObjectSpec::new("Exp1_B03_s1.tif", "ClassA", 1, 10),
        ObjectSpec::new("Exp1_B03_s2.tif", "ClassA", 1, 20),
        ObjectSpec::new("Exp1_C11_s1.tif", "ClassA", 1, 30),
    ]);
    assert_eq!(
        generator.grouping_regex(&Grouping::Auto).unwrap(),
        AUTO_GROUPING_PATTERNS[1]
    );

    let plate = generator
        .get_group_by_plate(
            &PlateFilter {
                plate_size: PlateSize::Plate8x12,
                ..plate_filter(Column::AreaSizePx)
            },
            &View::Heatmap,
        )
        .unwrap();
    let values = heatmap_values(&plate);
    assert_eq!(values.len(), 8);
    assert_eq!(values[1][2], Some(15.0), "B03: average of both fields");
    assert_eq!(values[2][10], Some(30.0), "C11");

    let well = generator
        .get_group_by_well(&well_filter("B03", Column::AreaSizePx), &View::Heatmap)
        .unwrap();
    assert_eq!(heatmap_values(&well), vec![vec![Some(10.0), Some(20.0)]]);
}

#[test]
fn auto_grouping_falls_back_to_the_first_pattern_when_nothing_matches() {
    let generator = open(&[ObjectSpec::new("image.tif", "ClassA", 1, 10)]);
    assert_eq!(
        generator.grouping_regex(&Grouping::Auto).unwrap(),
        AUTO_GROUPING_PATTERNS[0]
    );
    // An empty custom regex means Auto too.
    assert_eq!(
        generator
            .grouping_regex(&Grouping::Regex(" ".into()))
            .unwrap(),
        AUTO_GROUPING_PATTERNS[0]
    );
}

#[test]
fn one_image_per_well_is_image_one_of_its_well() {
    let generator = open(&[
        ObjectSpec::new("A01.tif", "ClassA", 1, 10),
        ObjectSpec::new("B02.tif", "ClassA", 1, 20),
    ]);
    assert_eq!(
        generator.grouping_regex(&Grouping::Auto).unwrap(),
        AUTO_GROUPING_PATTERNS[2]
    );
    let well = generator
        .get_group_by_well(&well_filter("B02", Column::AreaSizePx), &View::Heatmap)
        .unwrap();
    assert_eq!(heatmap_values(&well), vec![vec![Some(20.0)]]);
}

#[test]
fn folder_grouping_takes_the_well_from_the_folder_with_either_separator() {
    let generator = open(&[
        ObjectSpec::new("run/B03/img_1.tif", "ClassA", 1, 10),
        ObjectSpec::new("run/B03/img_2.ome.tif", "ClassA", 1, 20),
        ObjectSpec::new("run\\C04\\field 1.tif", "ClassA", 1, 30),
        ObjectSpec::new("control/x_1.tif", "ClassA", 1, 40),
    ]);
    let filter = PlateFilter {
        grouping: Grouping::Folder,
        ..plate_filter(Column::AreaSizePx)
    };
    let list = generator.get_group_by_plate(&filter, &View::List).unwrap();
    let mut wells = list.row_names.clone();
    wells.sort();
    assert_eq!(wells, vec!["B03", "C04", "control"]);

    let heatmap = generator
        .get_group_by_plate(&filter, &View::Heatmap)
        .unwrap();
    let values = heatmap_values(&heatmap);
    assert_eq!(values[1][2], Some(15.0), "B03 on the plate");
    assert_eq!(values[2][3], Some(30.0), "C04 on the plate");

    let well = generator
        .get_group_by_well(
            &WellFilter {
                grouping: Grouping::Folder,
                ..well_filter("B03", Column::AreaSizePx)
            },
            &View::Heatmap,
        )
        .unwrap();
    assert_eq!(heatmap_values(&well), vec![vec![Some(10.0), Some(20.0)]]);
}

#[test]
fn a_custom_regex_is_used_as_given() {
    let generator = open(&[ObjectSpec::new("p1-w-A-2-f-3.tif", "ClassA", 1, 10)]);
    let regex = r"^p1-w-(([A-Z])-([0-9]+))-f-([0-9]+)\.tif$";
    let filter = PlateFilter {
        grouping: Grouping::Regex(regex.into()),
        ..plate_filter(Column::AreaSizePx)
    };
    let list = generator.get_group_by_plate(&filter, &View::List).unwrap();
    assert_eq!(list.row_names, vec!["A-2"]);
    assert_eq!(generator.grouping_regex(&filter.grouping).unwrap(), regex);
}

#[test]
fn auto_well_layout_fits_the_highest_image_number_and_keeps_gaps() {
    let generator = open(&[
        ObjectSpec::new("A1_01.tif", "ClassA", 1, 10),
        ObjectSpec::new("A1_05.tif", "ClassA", 1, 50),
    ]);
    let well = generator
        .get_group_by_well(&well_filter("A1", Column::AreaSizePx), &View::Heatmap)
        .unwrap();
    // Highest image 5 -> a 2 x 3 grid; 2 to 4 are missing.
    assert_eq!(
        heatmap_values(&well),
        vec![vec![Some(10.0), None, None], vec![None, Some(50.0), None],]
    );
}

#[test]
fn a_fixed_well_layout_places_the_images_in_the_given_order() {
    let generator = open(&[
        ObjectSpec::new("A1_01.tif", "ClassA", 1, 10),
        ObjectSpec::new("A1_02.tif", "ClassA", 1, 20),
        ObjectSpec::new("A1_03.tif", "ClassA", 1, 30),
        ObjectSpec::new("A1_04.tif", "ClassA", 1, 40),
    ]);
    // Snake order: 1 2 / 4 3.
    let well = generator
        .get_group_by_well(
            &WellFilter {
                well_size: Some(WellSize { rows: 2, cols: 2 }),
                well_order: Some(vec![1, 2, 4, 3]),
                ..well_filter("A1", Column::AreaSizePx)
            },
            &View::Heatmap,
        )
        .unwrap();
    assert_eq!(
        heatmap_values(&well),
        vec![vec![Some(10.0), Some(20.0)], vec![Some(40.0), Some(30.0)]]
    );
}

#[test]
fn a_fixed_plate_size_is_used_even_when_smaller_wells_would_fit() {
    let generator = open(&[ObjectSpec::new("A1_01.tif", "ClassA", 1, 10)]);
    for (size, rows, cols) in [
        (PlateSize::Auto, 2, 3),
        (PlateSize::Plate1x1, 1, 1),
        (PlateSize::Plate48x72, 48, 72),
    ] {
        let plate = generator
            .get_group_by_plate(
                &PlateFilter {
                    plate_size: size,
                    ..plate_filter(Column::AreaSizePx)
                },
                &View::Heatmap,
            )
            .unwrap();
        assert_eq!(plate.rows.len(), rows, "{size:?}");
        assert_eq!(plate.rows[0].len(), cols, "{size:?}");
    }
}

#[test]
fn wells_for_plate_batched_matches_group_by_well_single_call() {
    let generator = open(&[
        ObjectSpec::new("A1_01.tif", "ClassA", 1, 10),
        ObjectSpec::new("A1_02.tif", "ClassA", 1, 20),
        ObjectSpec::new("B2_01.tif", "ClassA", 1, 100),
    ]);

    let batched = generator
        .get_wells_for_plate(&wells_batch_filter(Column::AreaSizePx), &View::List)
        .unwrap();
    assert_eq!(batched.len(), 2, "one entry per distinct well");

    for well in ["A1", "B2"] {
        let single = generator
            .get_group_by_well(&well_filter(well, Column::AreaSizePx), &View::List)
            .unwrap();
        let batch_result = &batched[well];
        assert_eq!(batch_result.row_names, single.row_names, "well {well}");
        for (b_row, s_row) in batch_result.rows.iter().zip(&single.rows) {
            assert_eq!(float_cell(&b_row[1]), float_cell(&s_row[1]), "well {well}");
        }
    }
}

/// Mirrors `get_group_by_plate_includes_a_well_with_no_objects_as_an_empty_group`
/// one level down: a field (image) with zero objects must still get its
/// own tile in the well view instead of silently vanishing.
#[test]
fn get_group_by_well_includes_a_field_with_no_objects_as_an_empty_tile() {
    let generator = open(&[ObjectSpec::new("A1_01.tif", "ClassA", 1, 10)]);
    generator
        .database
        .execute(
            "INSERT INTO images (image_name, image_rel_path, width, height, c_stacks, z_stacks, t_stacks) \
             VALUES ('A1_02.tif', 'A1_02.tif', 100, 100, 1, 1, 1)",
            [],
        )
        .unwrap();

    let list = generator
        .get_group_by_well(&well_filter("A1", Column::AreaSizePx), &View::List)
        .unwrap();

    assert_eq!(list.row_names, vec!["01".to_string(), "02".to_string()]);
    assert_eq!(float_cell(&list.rows[0][1]), 10.0);
    assert!(
        matches!(list.rows[1][1].value, CellValue::Empty),
        "field with no objects has no average to show"
    );

    // The Heatmap view must place a real, selectable (empty) tile at
    // field 02's grid position - not skip it as if no field were
    // there at all (that's reserved for a position no field occupies).
    let heatmap = generator
        .get_group_by_well(
            &WellFilter {
                well_size: Some(WellSize { rows: 4, cols: 4 }),
                ..well_filter("A1", Column::AreaSizePx)
            },
            &View::Heatmap,
        )
        .unwrap();
    // A 4x4 well, no `well_order`: idx 1 -> position 0 (row 0, col 0),
    // idx 2 -> position 1 (row 0, col 1).
    assert!(
        matches!(heatmap.rows[0][1].value, CellValue::Empty),
        "field 02 has no objects, so no number to show"
    );
    assert!(
        heatmap.rows[0][1].search_key.is_some(),
        "field 02 must still be a real, clickable tile"
    );
    assert!(
        heatmap.rows[0][2].search_key.is_none(),
        "position (0, 2) has no field at all and must stay non-clickable"
    );
}

/// A disabled image's *own* field must still show its real value (a
/// single image's own aggregate is never affected by its own disabled
/// flag - only a value that combines several images, like the plate
/// view's well average, excludes it), just flagged via `Cell::disabled`.
/// Mirrors `get_group_by_plate_flags_a_well_with_a_mix_of_enabled_and_disabled_images`
/// one level down.
#[test]
fn get_wells_for_plate_flags_but_keeps_a_disabled_images_field() {
    let generator = open(&[
        ObjectSpec::new("A1_01.tif", "ClassA", 1, 10),
        ObjectSpec::new("A1_02.tif", "ClassA", 1, 20),
    ]);
    generator.enable_image("A1_02.tif", true).unwrap();

    let batched = generator
        .get_wells_for_plate(&wells_batch_filter(Column::AreaSizePx), &View::List)
        .unwrap();
    assert_eq!(
        batched["A1"].row_names,
        vec!["01".to_string(), "02".to_string()],
        "the disabled field must still appear"
    );
    assert!(!batched["A1"].rows[0][1].disabled, "field 01 is enabled");
    assert!(batched["A1"].rows[1][1].disabled, "field 02 is disabled");
    assert!(
        matches!(batched["A1"].rows[1][1].value, CellValue::Float(v) if v == 20.0),
        "a disabled image's own field still shows its own real value"
    );

    let single = generator
        .get_group_by_well(&well_filter("A1", Column::AreaSizePx), &View::List)
        .unwrap();
    assert_eq!(single.row_names, vec!["01".to_string(), "02".to_string()]);
    assert!(single.rows[1][1].disabled);
}

/// A disabled field's own value is still shown (see the test above),
/// but must not skew the Auto color range every *other* field's tile is
/// colored against - disabling an outlier should change how the
/// remaining fields compare to each other, not leave them exactly where
/// they were as if nothing happened.
#[test]
fn get_group_by_well_heatmap_excludes_a_disabled_fields_value_from_the_auto_color_range() {
    let generator = open(&[
        ObjectSpec::new("A1_01.tif", "ClassA", 1, 10),
        ObjectSpec::new("A1_02.tif", "ClassA", 1, 20),
        ObjectSpec::new("A1_03.tif", "ClassA", 1, 1000),
    ]);
    generator.enable_image("A1_03.tif", true).unwrap();

    let heatmap = generator
        .get_group_by_well(&well_filter("A1", Column::AreaSizePx), &View::Heatmap)
        .unwrap();

    // Default 4x4 well grid, no `well_order`: field "01" -> (0, 0),
    // "02" -> (0, 1), "03" -> (0, 2).
    let schema = ColorSchema::default();
    assert_eq!(
        heatmap.rows[0][0].bg_color,
        value_to_color(10.0, 10.0, 20.0, &schema),
        "10 must be colored as the range's own min, ignoring the disabled 1000"
    );
    assert_eq!(
        heatmap.rows[0][1].bg_color,
        value_to_color(20.0, 10.0, 20.0, &schema),
        "20 must be colored as the range's own max, ignoring the disabled 1000"
    );
}

#[test]
fn group_by_plate_multi_agg_matches_group_by_plate_per_aggregation() {
    let generator = open(&[
        ObjectSpec::new("A1_01.tif", "ClassA", 1, 10),
        ObjectSpec::new("A1_01.tif", "ClassA", 1, 11),
        ObjectSpec::new("A1_01.tif", "ClassA", 1, 12),
        ObjectSpec::new("A1_02.tif", "ClassA", 1, 21),
        ObjectSpec::new("A1_02.tif", "ClassA", 1, 22),
        ObjectSpec::new("A1_03.tif", "ClassA", 1, 31),
        ObjectSpec::new("A1_03.tif", "ClassA", 1, 32),
        //
        ObjectSpec::new("B1_01.tif", "ClassA", 1, 40),
        ObjectSpec::new("B1_01.tif", "ClassA", 1, 41),
        ObjectSpec::new("B1_02.tif", "ClassA", 1, 42),
        ObjectSpec::new("B1_02.tif", "ClassA", 1, 43),
    ]);
    generator.enable_image("A1_03.tif", true).unwrap();
    let aggregations = vec![
        Aggregation::Avg,
        Aggregation::Min,
        Aggregation::Max,
        Aggregation::Sum,
    ];
    let multi_filter = PlateFilterMulti {
        plane: plane(),
        grouping: Grouping::Auto,
        aggregation: aggregations.clone(),
        object_class: vec![ObjectClass::Unset],
        column: vec![Column::AreaSizePx],
        color_schema: ColorSchema::default(),
        color_scale: ColorScale::default(),
        plate_size: PlateSize::Auto,
    };
    let multi = generator
        .get_group_by_plate_multi(&multi_filter, &View::List)
        .unwrap();
    assert_eq!(multi.len(), 4);

    for (aggregation, batched_result) in aggregations.iter().zip(&multi) {
        let mut filter = plate_filter(Column::AreaSizePx);
        filter.aggregation = aggregation.clone();
        let single = generator.get_group_by_plate(&filter, &View::List).unwrap();
        assert_eq!(
            float_cell(&batched_result.rows[0][1]),
            float_cell(&single.rows[0][1]),
            "aggregation {aggregation:?} disagrees between multi_agg batch and single call",
        );
        assert_eq!(
            batched_result.rows[0][1].disabled, single.rows[0][1].disabled,
            "aggregation {aggregation:?}'s disabled flag disagrees between multi_agg batch and single call",
        );
        assert!(
            !batched_result.rows[0][1].disabled,
            "the well itself must not be flagged disabled just because A1_03 is"
        );
    }
    // Sanity on the actual numbers (A1_03's 30 excluded from every
    // aggregate since it's disabled), not just internal agreement.
    assert_eq!(
        (float_cell(&multi[0].rows[0][1]) * 100.0).round() / 100.0,
        15.2
    ); //A1 avg(10, 20)
    assert_eq!(float_cell(&multi[1].rows[0][1]), 10.0); //A1 min(10, 20)
    assert_eq!(float_cell(&multi[2].rows[0][1]), 22.0); //A1 max(10, 20)
    assert_eq!(float_cell(&multi[3].rows[0][1]), 76.0); //A1 sum(10, 20)

    assert_eq!(float_cell(&multi[0].rows[1][1]), 41.5); //B1 avg(10, 20)
    assert_eq!(float_cell(&multi[1].rows[1][1]), 40.0); //B1 min(10, 20)
    assert_eq!(float_cell(&multi[2].rows[1][1]), 43.0); //B1 max(10, 20)
    assert_eq!(float_cell(&multi[3].rows[1][1]), 166.0); //B1 sum(10, 20)
}

#[test]
fn group_by_plate_multi_agg_matches_group_by_plate_per_aggregation_count() {
    let generator = open(&[
        ObjectSpec::new("A1_01.tif", "ClassA", 1, 10),
        ObjectSpec::new("A1_01.tif", "ClassA", 1, 11),
        ObjectSpec::new("A1_01.tif", "ClassA", 1, 12),
        ObjectSpec::new("A1_02.tif", "ClassA", 1, 21),
        ObjectSpec::new("A1_02.tif", "ClassA", 1, 22),
        ObjectSpec::new("A1_03.tif", "ClassA", 1, 31),
        ObjectSpec::new("A1_03.tif", "ClassA", 1, 32),
        //
        ObjectSpec::new("B1_01.tif", "ClassA", 1, 40),
        ObjectSpec::new("B1_01.tif", "ClassA", 1, 41),
        ObjectSpec::new("B1_02.tif", "ClassA", 1, 42),
        ObjectSpec::new("B1_02.tif", "ClassA", 1, 43),
    ]);
    generator.enable_image("A1_03.tif", true).unwrap();
    let aggregations = vec![
        Aggregation::Avg,
        Aggregation::Min,
        Aggregation::Max,
        Aggregation::Sum,
    ];
    let multi_filter = PlateFilterMulti {
        plane: plane(),
        grouping: Grouping::Auto,
        aggregation: aggregations.clone(),
        object_class: vec![ObjectClass::Unset],
        column: vec![Column::Count, Column::AreaSizePx],
        color_schema: ColorSchema::default(),
        color_scale: ColorScale::default(),
        plate_size: PlateSize::Auto,
    };
    let multi = generator
        .get_group_by_plate_multi(&multi_filter, &View::List)
        .unwrap();
    assert_eq!(multi.len(), 8);

    for (aggregation, batched_result) in aggregations.iter().zip(&multi) {
        let mut filter = plate_filter(Column::Count);
        filter.aggregation = aggregation.clone();
        let single = generator.get_group_by_plate(&filter, &View::List).unwrap();
        assert_eq!(
            float_cell(&batched_result.rows[0][1]),
            float_cell(&single.rows[0][1]),
            "aggregation {aggregation:?} disagrees between multi_agg batch and single call",
        );
        assert_eq!(
            batched_result.rows[0][1].disabled, single.rows[0][1].disabled,
            "aggregation {aggregation:?}'s disabled flag disagrees between multi_agg batch and single call",
        );
        assert!(
            !batched_result.rows[0][1].disabled,
            "the well itself must not be flagged disabled just because A1_03 is"
        );
    }

    // Count
    assert_eq!(float_cell(&multi[0].rows[0][1]), 2.5); //A1 
    assert_eq!(float_cell(&multi[1].rows[0][1]), 2.0); //A1 
    assert_eq!(float_cell(&multi[2].rows[0][1]), 3.0); //A1 
    assert_eq!(float_cell(&multi[3].rows[0][1]), 5.0); //A1 

    // Area size
    assert_eq!(
        (float_cell(&multi[4].rows[0][1]) * 100.0).round() / 100.0,
        15.2
    ); //A1 avg(10, 20)
    assert_eq!(float_cell(&multi[5].rows[0][1]), 10.0); //A1 min(10, 20)
    assert_eq!(float_cell(&multi[6].rows[0][1]), 22.0); //A1 max(10, 20)
    assert_eq!(float_cell(&multi[7].rows[0][1]), 76.0); //A1 sum(10, 20)

    // Count
    assert_eq!(float_cell(&multi[0].rows[1][1]), 2.0); //B1 
    assert_eq!(float_cell(&multi[1].rows[1][1]), 2.0); //B1 
    assert_eq!(float_cell(&multi[2].rows[1][1]), 2.0); //B1 
    assert_eq!(float_cell(&multi[3].rows[1][1]), 4.0); //B1 

    // Area size
    assert_eq!(float_cell(&multi[4].rows[1][1]), 41.5); //B1 avg(10, 20)
    assert_eq!(float_cell(&multi[5].rows[1][1]), 40.0); //B1 min(10, 20)
    assert_eq!(float_cell(&multi[6].rows[1][1]), 43.0); //B1 max(10, 20)
    assert_eq!(float_cell(&multi[7].rows[1][1]), 166.0); //B1 sum(10, 20)
}

#[test]
fn wells_for_plate_multi_agg_matches_wells_for_plate_per_aggregation() {
    let generator = open(&[
        ObjectSpec::new("A1_01.tif", "ClassA", 1, 10),
        ObjectSpec::new("A1_02.tif", "ClassA", 1, 20),
    ]);
    generator.enable_image("A1_02.tif", true).unwrap();
    let aggregations = vec![Aggregation::Avg, Aggregation::Sum];
    let multi_filter = WellsBatchFilterMulti {
        plane: plane(),
        grouping: Grouping::Auto,
        aggregation: aggregations.clone(),
        object_class: vec![ObjectClass::Unset],
        column: vec![Column::AreaSizePx],
        color_schema: ColorSchema::default(),
        color_scale: ColorScale::default(),
        well_size: None,
        well_order: None,
    };
    let multi = generator
        .get_wells_for_plate_multi(&multi_filter, &View::List)
        .unwrap();
    assert_eq!(multi.len(), 2);

    for (aggregation, batched_by_well) in aggregations.iter().zip(&multi) {
        let mut filter = wells_batch_filter(Column::AreaSizePx);
        filter.aggregation = aggregation.clone();
        let single_by_well = generator.get_wells_for_plate(&filter, &View::List).unwrap();
        for (well_id, single_result) in &single_by_well {
            let batch_result = &batched_by_well[well_id];
            for (b_row, s_row) in batch_result.rows.iter().zip(&single_result.rows) {
                assert_eq!(
                    float_cell(&b_row[1]),
                    float_cell(&s_row[1]),
                    "well {well_id}, aggregation {aggregation:?}",
                );
                assert_eq!(
                    b_row[1].disabled, s_row[1].disabled,
                    "well {well_id}, aggregation {aggregation:?}",
                );
            }
        }
    }
    // A1_02 (field "02") is disabled - its own field still shows its
    // own value (20), and only field "01" (value 10) is unflagged.
    assert!(!multi[0]["A1"].rows[0][1].disabled);
    assert!(multi[0]["A1"].rows[1][1].disabled);
    assert_eq!(float_cell(&multi[0]["A1"].rows[1][1]), 20.0);
}

#[test]
fn image_heatmap_list_and_heatmap_report_the_same_values_at_matching_positions() {
    // A 100x100 image, 50px tiles -> a 2x2 grid. One object per tile.
    let generator = open(&[
        ObjectSpec::new("img1.tif", "ClassA", 1, 10)
            .at_centroid(10.0, 10.0)
            .with_image_size(100, 100),
        ObjectSpec::new("img1.tif", "ClassA", 1, 20)
            .at_centroid(60.0, 10.0)
            .with_image_size(100, 100),
    ]);
    let filter = ImageHeatmapFilter {
        plane: plane(),
        image_rel_path: "img1.tif".to_string(),
        aggregation: Aggregation::Avg,
        object_class: ObjectClass::Unset,
        column: Column::AreaSizePx,
        color_schema: ColorSchema::default(),
        color_scale: ColorScale::default(),
        square_size: Some(50),
    };

    let list = generator.get_image_heatmap(&filter, &View::List).unwrap();
    let heatmap = generator
        .get_image_heatmap(&filter, &View::Heatmap)
        .unwrap();

    assert_eq!(heatmap.rows.len(), 2, "2x2 grid");
    assert_eq!(heatmap.rows[0].len(), 2);
    assert_eq!(float_cell(&heatmap.rows[0][0]), 10.0, "R0C0");
    assert_eq!(float_cell(&heatmap.rows[0][1]), 20.0, "R0C1");
    assert!(matches!(heatmap.rows[1][0].value, CellValue::Empty));

    assert_eq!(list.row_names, vec!["R0C0".to_string(), "R0C1".to_string()]);
    assert_eq!(
        float_cell(&list.rows[0][1]),
        float_cell(&heatmap.rows[0][0])
    );
    assert_eq!(
        float_cell(&list.rows[1][1]),
        float_cell(&heatmap.rows[0][1])
    );
}

#[test]
fn image_heatmap_clamps_a_centroid_on_the_far_edge_into_the_last_tile() {
    // A 100x100 image, 50px tiles -> valid tile indices 0..=1. A centroid
    // sitting exactly on the image's far edge (100.0) floor-divides to
    // tile index 2, which the 2x2 grid has no slot for - must clamp into
    // the last tile (index 1) rather than being dropped or panicking.
    let generator = open(&[ObjectSpec::new("img1.tif", "ClassA", 1, 10)
        .at_centroid(100.0, 100.0)
        .with_image_size(100, 100)]);
    let filter = ImageHeatmapFilter {
        plane: plane(),
        image_rel_path: "img1.tif".to_string(),
        aggregation: Aggregation::Avg,
        object_class: ObjectClass::Unset,
        column: Column::AreaSizePx,
        color_schema: ColorSchema::default(),
        color_scale: ColorScale::default(),
        square_size: Some(50),
    };
    let heatmap = generator
        .get_image_heatmap(&filter, &View::Heatmap)
        .unwrap();
    assert_eq!(float_cell(&heatmap.rows[1][1]), 10.0);
}

// -- small pure helpers ---------------------------------------------------

#[test]
fn row_letter_index_round_trips_including_double_letters() {
    for (letters, index) in [("A", 0), ("B", 1), ("Z", 25), ("AA", 26), ("AB", 27)] {
        assert_eq!(row_letter_to_index(letters), Some(index));
        assert_eq!(row_index_to_letter(index), letters);
    }
}

#[test]
fn row_letter_to_index_rejects_non_alphabetic_input() {
    assert_eq!(row_letter_to_index(""), None);
    assert_eq!(row_letter_to_index("A1"), None);
}

#[test]
fn col_number_to_index_is_one_based() {
    assert_eq!(col_number_to_index("1"), Some(0));
    assert_eq!(col_number_to_index("12"), Some(11));
    assert_eq!(
        col_number_to_index("0"),
        None,
        "0 has no 0-based predecessor"
    );
    assert_eq!(col_number_to_index("abc"), None);
}

#[test]
fn best_matching_dimensions_picks_the_smallest_plate_that_fits() {
    // Needs 8 rows (max_row index 7) x 10 cols (max_col index 9) - the
    // smallest standard size with at least 8 rows and 10 cols is 8x12
    // (6x8 falls short on rows: 6 < 8).
    assert_eq!(best_matching_dimensions(Some(7), Some(9)), (8, 12));
    assert_eq!(best_matching_dimensions(None, None), (2, 3));
    // Bigger than every standard size: the largest.
    assert_eq!(best_matching_dimensions(Some(60), Some(80)), (48, 72));
}

#[test]
fn auto_well_size_is_the_smallest_square_ish_grid_for_the_highest_image() {
    let size = |n| {
        let s = auto_well_size(n);
        (s.rows, s.cols)
    };
    assert_eq!(size(0), (1, 1));
    assert_eq!(size(1), (1, 1));
    assert_eq!(size(2), (1, 2));
    assert_eq!(size(4), (2, 2));
    assert_eq!(size(6), (2, 3));
    assert_eq!(size(9), (3, 3));
    assert_eq!(size(10), (3, 4));
    assert_eq!(size(16), (4, 4));
}

#[test]
fn color_scale_gradient_and_value_to_color_cover_every_schema() {
    for schema in [
        ColorSchema::Excel,
        ColorSchema::Viridis,
        ColorSchema::Plasma,
        ColorSchema::Inferno,
        ColorSchema::Cividis,
        ColorSchema::Coolwarm,
        ColorSchema::RedBlue,
        ColorSchema::YlGnBu,
        ColorSchema::Haline,
        ColorSchema::Algae,
        ColorSchema::Thermal,
    ] {
        let stops = color_scale_gradient(&schema);
        assert_eq!(stops.len(), COLOR_SCALE_GRADIENT_STOPS);
        assert_ne!(
            stops[0],
            stops[COLOR_SCALE_GRADIENT_STOPS - 1],
            "gradient should not be flat"
        );
    }
}

#[test]
fn column_display_label_coloc_count_falls_back_to_a_numeric_class_label() {
    let classes: Vec<Class> = vec![];
    assert_eq!(
        Column::ColocCount(ObjectClass::Global(7)).display_label(&classes),
        "Coloc with class 7"
    );
    assert_eq!(
        Column::ColocCount(ObjectClass::Unset).display_label(&classes),
        "Coloc with unset"
    );
}

#[test]
fn get_object_list_with_an_explicitly_empty_class_selection_is_empty() {
    let generator = open(&[ObjectSpec::new("img1.tif", "ClassA", 1, 10)]);
    let result = generator
        .get_object_list(&ListFilter {
            plane: plane(),
            images: crate::results::ImageSet::All,
            object_classes: Some(vec![]),
            columns: vec![Column::ObjectClass],
            with_coloc_details: false,
            related_to: None,
            page: no_page(),
            transpond_table: false,
            sort: None,
        })
        .unwrap();
    assert_eq!(result.source_object_count, 0);
    assert!(result.rows.is_empty());
}

#[test]
fn get_object_list_on_a_plane_with_no_objects_at_all_is_empty() {
    let generator = open(&[ObjectSpec::new("img1.tif", "ClassA", 1, 10)]);
    let result = generator
        .get_object_list(&ListFilter {
            plane: PlaneFilter {
                z_stack: 99,
                t_stack: 99,
            },
            images: crate::results::ImageSet::All,
            object_classes: None,
            columns: vec![Column::AreaSizePx],
            with_coloc_details: false,
            related_to: None,
            page: no_page(),
            transpond_table: false,
            sort: None,
        })
        .unwrap();
    assert_eq!(result.source_object_count, 0);
    assert!(result.rows.is_empty());
}

#[test]
fn get_grouped_by_image_object_classes_filter_restricts_to_the_selected_class() {
    let generator = open(&[
        ObjectSpec::new("img1.tif", "ClassA", 1, 10),
        ObjectSpec::new("img1.tif", "ClassB", 2, 999),
    ]);
    let result = generator
        .get_grouped_by_image(&GroupedByImageFilter {
            plane: plane(),
            images: crate::results::ImageSet::All,
            object_classes: Some(vec![ObjectClass::Global(1)]),
            columns: vec![Column::AreaSizePx],
            aggregation: vec![Aggregation::Avg],
            page: no_page(),
            transpond_table: false,
            grouping: None,
            empty_classes: true,
            sort: None,
        })
        .unwrap();
    assert_eq!(
        result.rows.len(),
        1,
        "only ClassA's group should be reported"
    );
}

#[test]
fn get_grouped_by_image_with_an_explicitly_empty_class_selection_is_empty() {
    let generator = open(&[ObjectSpec::new("img1.tif", "ClassA", 1, 10)]);
    let result = generator
        .get_grouped_by_image(&GroupedByImageFilter {
            plane: plane(),
            images: crate::results::ImageSet::All,
            object_classes: Some(vec![]),
            columns: vec![Column::AreaSizePx],
            aggregation: vec![Aggregation::Avg],
            page: no_page(),
            transpond_table: false,
            grouping: None,
            empty_classes: true,
            sort: None,
        })
        .unwrap();
    assert!(result.rows.is_empty());
}

#[test]
fn get_grouped_by_image_on_a_plane_with_no_objects_reports_a_zeroed_range() {
    let generator = open(&[ObjectSpec::new("img1.tif", "ClassA", 1, 10)]);
    let result = generator
        .get_grouped_by_image(&GroupedByImageFilter {
            plane: PlaneFilter {
                z_stack: 99,
                t_stack: 99,
            },
            images: crate::results::ImageSet::All,
            object_classes: None,
            columns: vec![Column::AreaSizePx],
            aggregation: vec![Aggregation::Avg],
            page: no_page(),
            transpond_table: false,
            grouping: None,
            empty_classes: true,
            sort: None,
        })
        .unwrap();
    assert!(result.rows.is_empty());
    assert_eq!(result.min, 0.0);
    assert_eq!(result.max, 0.0);
}

#[test]
fn get_grouped_by_image_pagination_cursor_returns_the_next_page() {
    let generator = open(&[
        ObjectSpec::new("img1.tif", "ClassA", 1, 10),
        ObjectSpec::new("img2.tif", "ClassA", 1, 20),
        ObjectSpec::new("img3.tif", "ClassA", 1, 30),
    ]);
    let base = GroupedByImageFilter {
        plane: plane(),
        images: crate::results::ImageSet::All,
        object_classes: None,
        columns: vec![Column::AreaSizePx],
        aggregation: vec![Aggregation::Avg],
        page: Pagination {
            limit: 1,
            after: None,
        },
        transpond_table: false,
        grouping: None,
        empty_classes: true,
        sort: None,
    };
    let first = generator.get_grouped_by_image(&base).unwrap();
    assert_eq!(first.rows.len(), 1);
    let cursor = first.row_names.last().cloned();

    let second = generator
        .get_grouped_by_image(&GroupedByImageFilter {
            page: Pagination {
                limit: 1,
                after: cursor,
            },
            ..base
        })
        .unwrap();
    assert_eq!(second.rows.len(), 1);
    assert_ne!(first.row_names, second.row_names);
}

#[test]
fn get_object_list_with_coloc_details_and_no_colocalizing_partners_still_fills_every_row() {
    // Both a coloc-class column and a resolvable metric are selected
    // (activating `with_coloc_details`'s fan-out), but neither object
    // colocalizes with anything at all - `partners_by_object`/
    // `partner_rows` must handle "not a single partner anywhere"
    // gracefully rather than only "some partners, some not".
    let generator = open(&[
        ObjectSpec::new("img1.tif", "ClassA", 1, 10),
        ObjectSpec::new("img1.tif", "ClassB", 2, 20),
    ]);
    let result = generator
        .get_object_list(&ListFilter {
            plane: plane(),
            images: crate::results::ImageSet::All,
            object_classes: None,
            columns: vec![
                Column::ColocCount(ObjectClass::Global(2)),
                Column::AreaSizePx,
            ],
            with_coloc_details: true,
            related_to: None,
            page: no_page(),
            transpond_table: false,
            sort: None,
        })
        .unwrap();
    assert_eq!(
        result.rows.len(),
        2,
        "one row per source object, no fan-out"
    );
    for row in &result.rows {
        assert!(matches!(row.last().unwrap().value, CellValue::String(ref s) if s == "-"));
    }
}

#[test]
fn get_object_list_with_coloc_details_dashes_the_inactive_coloc_class_column() {
    // Two coloc-class columns selected at once: object A colocalizes
    // with class 2 only, so its fanned-out row(s) must resolve the
    // class-2 cross-cell but dash the class-3 one, never the reverse.
    let objects = vec![
        ObjectSpec::new("img1.tif", "ClassA", 1, 10)
            .with_coloc(r#"{"2":["00000000-0000-0000-0000-000000000001"]}"#),
        ObjectSpec::new("img1.tif", "ClassB", 2, 99),
    ];
    let generator = open(&objects);
    let result = generator
        .get_object_list(&ListFilter {
            plane: plane(),
            images: crate::results::ImageSet::All,
            object_classes: None,
            columns: vec![
                Column::AreaSizePx,
                Column::ColocCount(ObjectClass::Global(2)),
                Column::ColocCount(ObjectClass::Global(3)),
            ],
            with_coloc_details: true,
            related_to: None,
            page: no_page(),
            transpond_table: false,
            sort: None,
        })
        .unwrap();

    // Column order (sorted): AreaSizePx, ColocCount(2), ColocCount(3),
    // then the coloc-detail cross-columns in the same
    // coloc-class-then-metric order: (2, AreaSizePx), (3, AreaSizePx).
    let object_a_row = &result.rows[0];
    let dash = |cell: &Cell| matches!(cell.value, CellValue::String(ref s) if s == "-");
    assert!(
        !dash(&object_a_row[3]),
        "class 2's cross-cell should resolve, not dash"
    );
    assert!(
        dash(&object_a_row[4]),
        "class 3's cross-cell must dash - A doesn't colocalize with it"
    );
}

#[test]
fn get_group_by_plate_honors_a_custom_grouping_regex() {
    // A single well under this custom `(plate)-(well)(idx)` scheme -
    // proves the non-default `grouping_regex` branch is actually used
    // (rather than falling back to `DEFAULT_GROUPING_REGEX`, which
    // wouldn't match this image name at all and would report nothing).
    let generator = open(&[ObjectSpec::new("plateX-well1.tif", "ClassA", 1, 42)]);
    let result = generator
        .get_group_by_plate(
            &PlateFilter {
                plane: plane(),
                grouping: Grouping::Regex(r"^(plateX)-(well)(\d+)\.".to_string()),
                aggregation: Aggregation::Avg,
                object_class: ObjectClass::Unset,
                column: Column::AreaSizePx,
                color_schema: ColorSchema::default(),
                color_scale: ColorScale::default(),
                plate_size: PlateSize::Auto,
            },
            &View::List,
        )
        .unwrap();
    assert_eq!(result.row_names, vec!["plateX".to_string()]);
    let cell_f64 = |cell: &Cell| match &cell.value {
        CellValue::Float(v) => *v as f64,
        _ => panic!("expected a float cell"),
    };
    assert_eq!(cell_f64(&result.rows[0][1]), 42.0);
}

// -- groups without objects: 0 vs. empty ---------------------------------
//
// A Count or Sum of a group without objects is 0 when its images were
// analysed on the selected plane; every other statistic stays empty, and
// so does everything of a failed, disabled-only or not-analysed group.

/// How an image row without objects of its own is seeded.
struct ImageRow {
    name: &'static str,
    successful: bool,
    disabled: bool,
    z_stacks: u32,
}

impl ImageRow {
    fn analysed(name: &'static str) -> Self {
        Self {
            name,
            successful: true,
            disabled: false,
            z_stacks: 1,
        }
    }
}

fn add_image(generator: &ResultsGenerator, image: ImageRow) {
    generator
        .database
        .execute(
            "INSERT INTO images (image_name, image_rel_path, successful, disabled, width, \
             height, c_stacks, z_stacks, t_stacks) VALUES (?, ?, ?, ?, 512, 512, 1, ?, 1)",
            duckdb::params![
                image.name,
                image.name,
                image.successful,
                image.disabled,
                image.z_stacks
            ],
        )
        .unwrap();
}

fn plate_value(generator: &ResultsGenerator, filter: &PlateFilter, well: &str) -> Option<f64> {
    let result = generator.get_group_by_plate(filter, &View::List).unwrap();
    let idx = result
        .row_names
        .iter()
        .position(|n| n == well)
        .unwrap_or_else(|| panic!("well {well} missing: {:?}", result.row_names));
    as_float(&result.rows[idx][1].value)
}

fn as_float(value: &CellValue) -> Option<f64> {
    match value {
        CellValue::Float(v) => Some(*v as f64),
        CellValue::Empty => None,
        _ => panic!("expected a float or empty cell"),
    }
}

#[test]
fn plate_count_and_sum_are_zero_for_a_well_whose_images_found_nothing() {
    let generator = open(&[ObjectSpec::new("A1_01.tif", "ClassA", 1, 10)]);
    add_image(&generator, ImageRow::analysed("B2_01.tif"));

    let count = plate_value(&generator, &plate_filter(Column::Count), "B2");
    let sum = plate_value(
        &generator,
        &PlateFilter {
            aggregation: Aggregation::Sum,
            ..plate_filter(Column::AreaSizePx)
        },
        "B2",
    );
    let avg = plate_value(&generator, &plate_filter(Column::AreaSizePx), "B2");

    assert_eq!(count, Some(0.0));
    assert_eq!(sum, Some(0.0));
    assert_eq!(avg, None, "no average without objects");
    assert_eq!(
        plate_value(&generator, &plate_filter(Column::Count), "A1"),
        Some(1.0)
    );
}

#[test]
fn plate_count_stays_empty_for_a_well_of_only_failed_or_only_disabled_images() {
    let generator = open(&[ObjectSpec::new("A1_01.tif", "ClassA", 1, 10)]);
    add_image(
        &generator,
        ImageRow {
            successful: false,
            ..ImageRow::analysed("B2_01.tif")
        },
    );
    add_image(
        &generator,
        ImageRow {
            disabled: true,
            ..ImageRow::analysed("C3_01.tif")
        },
    );

    let filter = plate_filter(Column::Count);
    assert_eq!(plate_value(&generator, &filter, "B2"), None);
    assert_eq!(plate_value(&generator, &filter, "C3"), None);
}

#[test]
fn plate_count_is_zero_when_at_least_one_enabled_image_of_the_well_was_analysed() {
    let generator = open(&[ObjectSpec::new("A1_01.tif", "ClassA", 1, 10)]);
    add_image(
        &generator,
        ImageRow {
            successful: false,
            ..ImageRow::analysed("B2_01.tif")
        },
    );
    add_image(&generator, ImageRow::analysed("B2_02.tif"));

    let value = plate_value(&generator, &plate_filter(Column::Count), "B2");

    assert_eq!(value, Some(0.0));
}

fn count_on_plane(z_stack: u32) -> PlateFilter {
    PlateFilter {
        plane: PlaneFilter {
            z_stack,
            t_stack: 0,
        },
        ..plate_filter(Column::Count)
    }
}

#[test]
fn plate_count_stays_empty_on_a_plane_the_run_did_not_analyse() {
    // A Z-projection run: 3 planes in the file, objects only on plane 0.
    let generator = open(&[ObjectSpec::new("A1_01.tif", "ClassA", 1, 10)]);
    add_image(
        &generator,
        ImageRow {
            z_stacks: 3,
            ..ImageRow::analysed("B2_01.tif")
        },
    );

    assert_eq!(plate_value(&generator, &count_on_plane(0), "B2"), Some(0.0));
    assert_eq!(plate_value(&generator, &count_on_plane(1), "B2"), None);
}

#[test]
fn planes_between_the_lowest_and_highest_object_plane_count_as_analysed() {
    let generator = open(&[
        ObjectSpec::new("A1_01.tif", "ClassA", 1, 10).at_plane(0, 0),
        ObjectSpec::new("A1_01.tif", "ClassA", 1, 10).at_plane(2, 0),
    ]);
    add_image(
        &generator,
        ImageRow {
            z_stacks: 3,
            ..ImageRow::analysed("B2_01.tif")
        },
    );
    // Has only one plane itself - plane 1 doesn't exist for it.
    add_image(&generator, ImageRow::analysed("C3_01.tif"));

    // Plane 1: nothing found anywhere, but inside the analysed range.
    assert_eq!(plate_value(&generator, &count_on_plane(1), "A1"), Some(0.0));
    assert_eq!(plate_value(&generator, &count_on_plane(1), "B2"), Some(0.0));
    assert_eq!(plate_value(&generator, &count_on_plane(1), "C3"), None);
}

#[test]
fn a_run_without_any_object_has_no_values_at_all() {
    let generator = open(&[]);
    add_image(&generator, ImageRow::analysed("B2_01.tif"));

    assert_eq!(plate_value(&generator, &count_on_plane(0), "B2"), None);
}

#[test]
fn plate_sum_of_objects_without_a_value_stays_empty_rather_than_zero() {
    // Objects exist, but none has an nm area (no pixel size): the sum
    // is unknown, not 0.
    let generator = open(&[ObjectSpec::new("A1_01.tif", "ClassA", 1, 10)]);
    generator
        .database
        .execute("UPDATE objects SET area_nm2 = NULL", [])
        .unwrap();
    let filter = PlateFilter {
        aggregation: Aggregation::Sum,
        ..plate_filter(Column::AreaSizeNm)
    };

    assert_eq!(plate_value(&generator, &filter, "A1"), None);
}

#[test]
fn plate_multi_fills_zero_only_for_count_and_sum() {
    let generator = open(&[ObjectSpec::new("A1_01.tif", "ClassA", 1, 10)]);
    add_image(&generator, ImageRow::analysed("B2_01.tif"));
    let results = generator
        .get_group_by_plate_multi(
            &PlateFilterMulti {
                plane: plane(),
                grouping: Grouping::Auto,
                aggregation: vec![Aggregation::Avg, Aggregation::Sum],
                object_class: vec![ObjectClass::Unset],
                column: vec![Column::Count, Column::AreaSizePx],
                color_schema: ColorSchema::default(),
                color_scale: ColorScale::default(),
                plate_size: PlateSize::Auto,
            },
            &View::List,
        )
        .unwrap();
    let b2 = |result: &DatabaseResult| {
        let idx = result.row_names.iter().position(|n| n == "B2").unwrap();
        as_float(&result.rows[idx][1].value)
    };

    // (Count, Avg), (Count, Sum), (area, Avg), (area, Sum)
    let values: Vec<Option<f64>> = results.iter().map(b2).collect();
    assert_eq!(values, [Some(0.0), Some(0.0), None, Some(0.0)]);
}

#[test]
fn well_fields_without_objects_count_zero_unless_their_analysis_failed() {
    let generator = open(&[ObjectSpec::new("A1_01.tif", "ClassA", 1, 10)]);
    add_image(&generator, ImageRow::analysed("A1_02.tif"));
    add_image(
        &generator,
        ImageRow {
            successful: false,
            ..ImageRow::analysed("A1_03.tif")
        },
    );
    // A disabled field's own value is still its own real result.
    add_image(
        &generator,
        ImageRow {
            disabled: true,
            ..ImageRow::analysed("A1_04.tif")
        },
    );
    let filter = well_filter("A1", Column::Count);

    let single = generator.get_group_by_well(&filter, &View::List).unwrap();
    let values: Vec<Option<f64>> = single
        .rows
        .iter()
        .map(|row| as_float(&row[1].value))
        .collect();
    assert_eq!(values, [Some(1.0), Some(0.0), None, Some(0.0)]);

    let batched = generator
        .get_wells_for_plate(&wells_batch_filter(Column::Count), &View::List)
        .unwrap();
    let batched_values: Vec<Option<f64>> = batched["A1"]
        .rows
        .iter()
        .map(|row| as_float(&row[1].value))
        .collect();
    assert_eq!(batched_values, values);

    let multi = generator
        .get_wells_for_plate_multi(
            &WellsBatchFilterMulti {
                plane: plane(),
                grouping: Grouping::Auto,
                aggregation: vec![Aggregation::Avg],
                object_class: vec![ObjectClass::Unset],
                column: vec![Column::Count, Column::AreaSizePx],
                color_schema: ColorSchema::default(),
                color_scale: ColorScale::default(),
                well_size: None,
                well_order: None,
            },
            &View::List,
        )
        .unwrap();
    let multi_values = |combo: usize| -> Vec<Option<f64>> {
        multi[combo]["A1"]
            .rows
            .iter()
            .map(|row| as_float(&row[1].value))
            .collect()
    };
    assert_eq!(multi_values(0), values);
    assert_eq!(multi_values(1), [Some(10.0), None, None, None]);
}

fn grouped(generator: &ResultsGenerator, aggregation: Aggregation) -> DatabaseResult {
    generator
        .get_grouped_by_image(&GroupedByImageFilter {
            plane: plane(),
            images: crate::results::ImageSet::All,
            object_classes: None,
            columns: vec![Column::Count, Column::AreaSizePx],
            aggregation: vec![aggregation],
            page: no_page(),
            transpond_table: false,
            grouping: None,
            empty_classes: true,
            sort: None,
        })
        .unwrap()
}

/// `(image, class label, values...)` per row.
fn grouped_rows(result: &DatabaseResult) -> Vec<(String, String, Vec<Option<f64>>)> {
    result
        .rows
        .iter()
        .map(|row| {
            let text = |cell: &Cell| match &cell.value {
                CellValue::String(s) => s.clone(),
                CellValue::Class((label, _)) => label.clone(),
                _ => panic!("expected text"),
            };
            (
                text(&row[0]),
                text(&row[1]),
                row[2..].iter().map(|c| as_float(&c.value)).collect(),
            )
        })
        .collect()
}

fn grouped_by(
    generator: &ResultsGenerator,
    grouping: Grouping,
    aggregation: Aggregation,
    empty_classes: bool,
) -> Vec<(String, String, Vec<Option<f64>>)> {
    grouped_rows(
        &generator
            .get_grouped_by_image(&GroupedByImageFilter {
                plane: plane(),
                images: crate::results::ImageSet::All,
                object_classes: None,
                columns: vec![Column::Count, Column::AreaSizePx],
                aggregation: vec![aggregation],
                page: no_page(),
                transpond_table: false,
                grouping: Some(grouping),
                empty_classes,
                sort: None,
            })
            .unwrap(),
    )
}

/// A row per well and class: the area over the well's objects, Count
/// over its images' counts (the mean: objects per image); images the
/// pattern doesn't fit in "No group".
#[test]
fn grouped_by_well_aggregates_the_objects_and_counts_of_its_images() {
    let generator = open(&[
        ObjectSpec::new("A1_01.tif", "ClassA", 1, 10),
        ObjectSpec::new("A1_01.tif", "ClassA", 1, 30),
        ObjectSpec::new("A1_02.tif", "ClassA", 1, 20),
        ObjectSpec::new("B2_01.tif", "ClassB", 2, 40),
        ObjectSpec::new("notes.tif", "ClassA", 1, 50),
    ]);
    let row = |group: &str, class: &str, count: f64, area: f64| {
        (
            group.to_string(),
            class.to_string(),
            vec![Some(count), Some(area)],
        )
    };
    assert_eq!(
        grouped_by(&generator, Grouping::Auto, Aggregation::Avg, false),
        [
            row("No group", "ClassA", 1.0, 50.0),
            row("A1", "ClassA", 1.5, 20.0),
            row("B2", "ClassB", 1.0, 40.0),
        ]
    );
    assert_eq!(
        grouped_by(&generator, Grouping::Auto, Aggregation::Sum, false)[1],
        row("A1", "ClassA", 3.0, 60.0)
    );
    // Classes without objects too: B2 has no ClassA - a real 0 count.
    let all = grouped_by(&generator, Grouping::Auto, Aggregation::Sum, true);
    assert!(
        all.contains(&(
            "B2".to_string(),
            "ClassA".to_string(),
            vec![Some(0.0), Some(0.0)]
        )),
        "{all:?}"
    );
}

/// Wells with the classes side by side: a row per well, per class its
/// values in its block (empty where a well has none of a class, Count 0
/// with empty classes) - and page by page every well once, whatever the
/// page size, a well never cut between pages.
#[test]
fn transposed_wells_put_each_class_of_a_well_in_one_row_across_pages() {
    let generator = open(&[
        ObjectSpec::new("A1_01.tif", "ClassA", 1, 10),
        ObjectSpec::new("A1_02.tif", "ClassB", 2, 30),
        ObjectSpec::new("A2_01.tif", "ClassA", 1, 20),
        ObjectSpec::new("B1_01.tif", "ClassB", 2, 40),
        ObjectSpec::new("B2_01.tif", "ClassA", 1, 50),
        ObjectSpec::new("B2_01.tif", "ClassB", 2, 60),
    ]);
    let query = |limit: i32, after: Option<String>, empty_classes: bool| {
        generator
            .get_grouped_by_image(&GroupedByImageFilter {
                plane: plane(),
                images: crate::results::ImageSet::All,
                object_classes: Some(vec![ObjectClass::Global(1), ObjectClass::Global(2)]),
                columns: vec![Column::Count, Column::AreaSizePx],
                aggregation: vec![Aggregation::Sum],
                page: Pagination { limit, after },
                transpond_table: true,
                grouping: Some(Grouping::Auto),
                empty_classes,
                sort: None,
            })
            .unwrap()
    };
    let text = |result: &DatabaseResult| -> Vec<String> {
        result
            .rows
            .iter()
            .map(|row| {
                row.iter()
                    .map(|cell| match &cell.value {
                        CellValue::String(s) => s.clone(),
                        CellValue::Float(v) => v.to_string(),
                        _ => String::new(),
                    })
                    .collect::<Vec<_>>()
                    .join(",")
            })
            .collect()
    };
    for empty_classes in [false, true] {
        let all = query(1000, None, empty_classes);
        assert_eq!(all.column_names.len(), 1 + 2 * 2, "{:?}", all.column_names);
        assert_eq!(all.column_names[0], "well");
        let rows = text(&all);
        // Without empty classes a class a well lacks has no row to place:
        // empty, as in the normal view.
        let expected = if empty_classes {
            ["A1,1,10,1,30", "A2,1,20,0,0", "B1,0,0,1,40", "B2,1,50,1,60"]
        } else {
            ["A1,1,10,1,30", "A2,1,20,,", "B1,,,1,40", "B2,1,50,1,60"]
        };
        assert_eq!(rows, expected, "empty classes {empty_classes}");
        for limit in [1, 2, 3] {
            let mut paged = Vec::new();
            let mut after = None;
            loop {
                let page = query(limit, after.take(), empty_classes);
                paged.extend(text(&page));
                if page.source_object_count < limit as usize {
                    break;
                }
                after = page.row_names.last().cloned();
                assert!(paged.len() <= 4, "no repeats: {paged:?}");
            }
            assert_eq!(paged, rows, "limit {limit}, empty classes {empty_classes}");
        }
    }
}

/// Wells in plate order (A2 before A10, B1 after them), page by page
/// without a gap.
#[test]
fn grouped_by_well_is_in_plate_order_across_pages() {
    let generator = open(&[
        ObjectSpec::new("A10_01.tif", "ClassA", 1, 10),
        ObjectSpec::new("B1_01.tif", "ClassA", 1, 10),
        ObjectSpec::new("A2_01.tif", "ClassA", 1, 10),
        ObjectSpec::new("A1_01.tif", "ClassA", 1, 10),
    ]);
    let page = |after: Option<String>| {
        generator
            .get_grouped_by_image(&GroupedByImageFilter {
                plane: plane(),
                images: crate::results::ImageSet::All,
                object_classes: None,
                columns: vec![Column::Count],
                aggregation: vec![Aggregation::Sum],
                page: Pagination { limit: 1, after },
                transpond_table: false,
                grouping: Some(Grouping::Auto),
                empty_classes: false,
                sort: None,
            })
            .unwrap()
    };
    let mut wells = Vec::new();
    let mut after = None;
    for _ in 0..10 {
        let result = page(after.clone());
        let Some(last) = result.row_names.last().cloned() else {
            break;
        };
        wells.extend(grouped_rows(&result).into_iter().map(|(well, ..)| well));
        after = Some(last);
    }
    assert_eq!(wells, ["A1", "A2", "A10", "B1"]);
}

/// Sorted by a column: over all the objects, page by page by offset.
#[test]
fn the_object_list_sorts_by_a_column_across_pages() {
    let generator = open(&[
        ObjectSpec::new("a.tif", "ClassA", 1, 30),
        ObjectSpec::new("a.tif", "ClassA", 1, 10),
        ObjectSpec::new("b.tif", "ClassA", 1, 50),
        ObjectSpec::new("b.tif", "ClassA", 1, 20),
        ObjectSpec::new("c.tif", "ClassA", 1, 40),
    ]);
    let page = |after: Option<String>, descending: bool| {
        let result = generator
            .get_object_list(&ListFilter {
                plane: plane(),
                images: crate::results::ImageSet::All,
                object_classes: None,
                columns: vec![Column::AreaSizePx],
                with_coloc_details: false,
                related_to: None,
                page: Pagination { limit: 2, after },
                transpond_table: false,
                sort: Some(SortBy {
                    column: Column::AreaSizePx,
                    aggregation: None,
                    descending,
                }),
            })
            .unwrap();
        result
            .rows
            .iter()
            .map(|row| match &row[0].value {
                CellValue::Integer(v) => f64::from(*v),
                other => as_float(other).unwrap(),
            })
            .collect::<Vec<_>>()
    };
    let all = |descending| {
        [None, Some("2".to_string()), Some("4".to_string())]
            .into_iter()
            .flat_map(|after| page(after, descending))
            .collect::<Vec<_>>()
    };
    assert_eq!(all(false), [10.0, 20.0, 30.0, 40.0, 50.0]);
    assert_eq!(all(true), [50.0, 40.0, 30.0, 20.0, 10.0]);

    // Finding an object (a click on the image): the pages before it as
    // row offsets, under the same order - area 40 is row 3 ascending
    // (page 2 of 2 rows), row 1 descending (page 1).
    let starts = |descending| {
        generator
            .object_list_page_starts(
                &ListFilter {
                    plane: plane(),
                    images: crate::results::ImageSet::All,
                    object_classes: None,
                    columns: vec![Column::AreaSizePx],
                    with_coloc_details: false,
                    related_to: None,
                    page: Pagination {
                        limit: 2,
                        after: None,
                    },
                    transpond_table: false,
                    sort: Some(SortBy {
                        column: Column::AreaSizePx,
                        aggregation: None,
                        descending,
                    }),
                },
                "00000000-0000-0000-0000-000000000004",
            )
            .unwrap()
    };
    assert_eq!(starts(false), Some(vec!["2".to_string()]));
    assert_eq!(starts(true), Some(Vec::new()));
    assert!(page(Some("2".to_string()), false).contains(&40.0));
}

/// Grouped rows sorted by a statistic - per image and per well.
#[test]
fn grouped_rows_sort_by_a_statistic() {
    let generator = open(&[
        ObjectSpec::new("A1_01.tif", "ClassA", 1, 10),
        ObjectSpec::new("A2_01.tif", "ClassA", 1, 30),
        ObjectSpec::new("A2_01.tif", "ClassA", 1, 30),
        ObjectSpec::new("A3_01.tif", "ClassA", 1, 20),
    ]);
    let order = |grouping: Option<Grouping>| {
        let result = generator
            .get_grouped_by_image(&GroupedByImageFilter {
                plane: plane(),
                images: crate::results::ImageSet::All,
                object_classes: None,
                columns: vec![Column::Count, Column::AreaSizePx],
                aggregation: vec![Aggregation::Avg],
                page: no_page(),
                transpond_table: false,
                grouping,
                empty_classes: false,
                sort: Some(SortBy {
                    column: Column::AreaSizePx,
                    aggregation: Some(Aggregation::Avg),
                    descending: true,
                }),
            })
            .unwrap();
        grouped_rows(&result)
            .into_iter()
            .map(|(name, ..)| name)
            .collect::<Vec<_>>()
    };
    assert_eq!(order(None), ["A2_01.tif", "A3_01.tif", "A1_01.tif"]);
    assert_eq!(order(Some(Grouping::Auto)), ["A2", "A3", "A1"]);
}

#[test]
fn grouped_by_folder_a_row_per_folder() {
    let generator = open(&[
        ObjectSpec::new("x/a.tif", "ClassA", 1, 10),
        ObjectSpec::new("x/b.tif", "ClassA", 1, 30),
        ObjectSpec::new("y/c.tif", "ClassA", 1, 20),
        ObjectSpec::new("d.tif", "ClassA", 1, 40),
    ]);
    let rows = grouped_by(&generator, Grouping::Folder, Aggregation::Sum, false);
    let groups: Vec<(&str, Option<f64>)> = rows
        .iter()
        .map(|(group, _, values)| (group.as_str(), values[0]))
        .collect();
    // An image directly in the image root: the root folder.
    assert_eq!(
        groups,
        [("(root)", Some(1.0)), ("x", Some(2.0)), ("y", Some(1.0))]
    );
}

/// A row per image without the empty classes: only the classes with
/// objects in the image.
#[test]
fn grouped_by_image_can_leave_out_classes_without_objects() {
    let generator = open(&[
        ObjectSpec::new("img1.tif", "ClassA", 1, 10),
        ObjectSpec::new("img2.tif", "ClassB", 2, 20),
    ]);
    let result = generator
        .get_grouped_by_image(&GroupedByImageFilter {
            plane: plane(),
            images: crate::results::ImageSet::All,
            object_classes: None,
            columns: vec![Column::Count],
            aggregation: vec![Aggregation::Sum],
            page: no_page(),
            transpond_table: false,
            grouping: None,
            empty_classes: false,
            sort: None,
        })
        .unwrap();
    let rows: Vec<(String, String)> = grouped_rows(&result)
        .into_iter()
        .map(|(image, class, _)| (image, class))
        .collect();
    assert_eq!(
        rows,
        [
            ("img1.tif".to_string(), "ClassA".to_string()),
            ("img2.tif".to_string(), "ClassB".to_string())
        ]
    );
}

#[test]
fn grouped_by_image_lists_every_analysed_image_and_class_with_count_zero() {
    let generator = open(&[
        ObjectSpec::new("img1.tif", "ClassA", 1, 10),
        ObjectSpec::new("img2.tif", "ClassB", 2, 20),
    ]);
    add_image(&generator, ImageRow::analysed("img3.tif"));
    add_image(
        &generator,
        ImageRow {
            successful: false,
            ..ImageRow::analysed("img4.tif")
        },
    );

    let rows = grouped_rows(&grouped(&generator, Aggregation::Avg));

    let row = |image: &str, class: &str, count, area| {
        (image.to_string(), class.to_string(), vec![count, area])
    };
    assert_eq!(
        rows,
        [
            row("img1.tif", "ClassA", Some(1.0), Some(10.0)),
            row("img1.tif", "ClassB", Some(0.0), None),
            row("img2.tif", "ClassA", Some(0.0), None),
            row("img2.tif", "ClassB", Some(1.0), Some(20.0)),
            row("img3.tif", "ClassA", Some(0.0), None),
            row("img3.tif", "ClassB", Some(0.0), None),
            // img4.tif failed and found nothing: no row at all.
        ]
    );
}

#[test]
fn grouped_by_image_shows_a_missing_spread_as_empty_not_zero() {
    // A sample standard deviation needs two values; one object has none.
    let generator = open(&[ObjectSpec::new("img1.tif", "ClassA", 1, 10)]);

    let rows = grouped_rows(&grouped(&generator, Aggregation::Stddev));

    assert_eq!(rows[0].2, [Some(1.0), None]);
}

#[test]
fn grouped_by_image_pages_through_images_without_objects_without_gaps() {
    let generator = open(&[ObjectSpec::new("img1.tif", "ClassA", 1, 10)]);
    add_image(&generator, ImageRow::analysed("img2.tif"));
    add_image(&generator, ImageRow::analysed("img3.tif"));
    let page = |after: Option<String>| {
        generator
            .get_grouped_by_image(&GroupedByImageFilter {
                plane: plane(),
                images: crate::results::ImageSet::All,
                object_classes: None,
                columns: vec![Column::Count],
                aggregation: vec![Aggregation::Avg],
                page: Pagination { limit: 2, after },
                transpond_table: false,
                grouping: None,
                empty_classes: true,
                sort: None,
            })
            .unwrap()
    };

    let mut seen = Vec::new();
    let mut after = None;
    loop {
        let result = page(after);
        seen.extend(grouped_rows(&result).into_iter().map(|(image, ..)| image));
        match result.row_names.last() {
            Some(last) if result.rows.len() == 2 => after = Some(last.clone()),
            _ => break,
        }
    }

    assert_eq!(seen, ["img1.tif", "img2.tif", "img3.tif"]);
}

#[test]
fn grouped_by_image_has_no_rows_on_a_plane_the_run_did_not_analyse() {
    let generator = open(&[ObjectSpec::new("img1.tif", "ClassA", 1, 10)]);
    add_image(
        &generator,
        ImageRow {
            z_stacks: 3,
            ..ImageRow::analysed("img2.tif")
        },
    );

    let result = generator
        .get_grouped_by_image(&GroupedByImageFilter {
            plane: PlaneFilter {
                z_stack: 1,
                t_stack: 0,
            },
            images: crate::results::ImageSet::All,
            object_classes: None,
            columns: vec![Column::Count],
            aggregation: vec![Aggregation::Avg],
            page: no_page(),
            transpond_table: false,
            grouping: None,
            empty_classes: true,
            sort: None,
        })
        .unwrap();

    assert!(result.rows.is_empty(), "{:?}", result.row_names);
}

#[test]
fn grouped_by_image_leaves_out_background_unless_it_is_selected() {
    let generator = open(&[ObjectSpec::new("img1.tif", "ClassA", 1, 10)]);
    generator
        .database
        .execute(
            "INSERT INTO classes (class_id, name, color) VALUES (0, 'Background', 0)",
            [],
        )
        .unwrap();
    generator.classes_cache.replace(None);

    let all = grouped_rows(&grouped(&generator, Aggregation::Avg));
    let selected = grouped_rows(
        &generator
            .get_grouped_by_image(&GroupedByImageFilter {
                plane: plane(),
                images: crate::results::ImageSet::All,
                object_classes: Some(vec![ObjectClass::BACKGROUND]),
                columns: vec![Column::Count],
                aggregation: vec![Aggregation::Avg],
                page: no_page(),
                transpond_table: false,
                grouping: None,
                empty_classes: true,
                sort: None,
            })
            .unwrap(),
    );

    let classes: Vec<&str> = all.iter().map(|(_, class, _)| class.as_str()).collect();
    assert_eq!(classes, ["ClassA"]);
    assert_eq!(selected.len(), 1);
    assert_eq!(selected[0].1, "Background");
    assert_eq!(selected[0].2, [Some(0.0)]);
}

#[test]
fn transposed_grouped_shows_an_analysed_image_without_objects_with_count_zero() {
    let generator = open(&[ObjectSpec::new("img1.tif", "ClassA", 1, 10)]);
    add_image(&generator, ImageRow::analysed("img2.tif"));
    add_image(
        &generator,
        ImageRow {
            successful: false,
            ..ImageRow::analysed("img3.tif")
        },
    );

    let result = grouped_transposed(&generator, None, no_page());

    assert_eq!(result.row_names, ["img1.tif", "img2.tif"]);
    assert_eq!(cell_value(&result.rows[1][1]), Some(0.0), "Count");
    assert!(
        matches!(result.rows[1][2].value, CellValue::Empty),
        "Avg area"
    );
}

fn heatmap(
    generator: &ResultsGenerator,
    image: &str,
    column: Column,
    aggregation: Aggregation,
) -> DatabaseResult {
    generator
        .get_image_heatmap(
            &ImageHeatmapFilter {
                plane: plane(),
                image_rel_path: image.to_string(),
                aggregation,
                object_class: ObjectClass::Unset,
                column,
                color_schema: ColorSchema::default(),
                color_scale: ColorScale::default(),
                square_size: Some(256),
            },
            &View::Heatmap,
        )
        .unwrap()
}

#[test]
fn image_heatmap_counts_zero_in_squares_without_objects() {
    let generator = open(&[ObjectSpec::new("img1.tif", "ClassA", 1, 10)
        .with_image_size(512, 256)
        .at_centroid(10.0, 10.0)]);

    let count = heatmap(&generator, "img1.tif", Column::Count, Aggregation::Avg);
    let avg = heatmap(&generator, "img1.tif", Column::AreaSizePx, Aggregation::Avg);

    let values = |result: &DatabaseResult| -> Vec<Option<f64>> {
        result.rows[0].iter().map(|c| as_float(&c.value)).collect()
    };
    assert_eq!(values(&count), [Some(1.0), Some(0.0)]);
    assert_eq!(values(&avg), [Some(10.0), None]);
}

#[test]
fn image_heatmap_of_a_failed_image_stays_empty() {
    let generator = open(&[ObjectSpec::new("img1.tif", "ClassA", 1, 10)]);
    add_image(
        &generator,
        ImageRow {
            successful: false,
            ..ImageRow::analysed("img2.tif")
        },
    );

    let count = heatmap(&generator, "img2.tif", Column::Count, Aggregation::Avg);

    assert!(
        count
            .rows
            .iter()
            .flatten()
            .all(|c| matches!(c.value, CellValue::Empty))
    );
}

// -- failed images ------------------------------------------------------
//
// An image whose analysis stopped with an error only has the objects
// found before the error: excluded from a well's value like a disabled
// image, and marked wherever it's shown on its own.

fn mark_failed(generator: &ResultsGenerator, image: &str) {
    generator
        .database
        .execute(
            "UPDATE images SET successful = false WHERE image_rel_path = ?",
            [image],
        )
        .unwrap();
}

#[test]
fn plate_leaves_out_a_failed_images_objects_and_marks_the_well() {
    let generator = open(&[
        ObjectSpec::new("A1_01.tif", "ClassA", 1, 10),
        ObjectSpec::new("A1_02.tif", "ClassA", 1, 100),
        ObjectSpec::new("B2_01.tif", "ClassA", 1, 20),
    ]);
    mark_failed(&generator, "A1_02.tif");

    let result = generator
        .get_group_by_plate(&plate_filter(Column::AreaSizePx), &View::List)
        .unwrap();
    let a1 = result.row_names.iter().position(|n| n == "A1").unwrap();
    let b2 = result.row_names.iter().position(|n| n == "B2").unwrap();

    assert_eq!(as_float(&result.rows[a1][1].value), Some(10.0));
    assert!(result.rows[a1][1].any_failed);
    assert!(!result.rows[a1][1].failed, "a well itself never failed");
    assert!(!result.rows[b2][1].any_failed);
    assert_eq!(
        plate_value(&generator, &plate_filter(Column::Count), "A1"),
        Some(1.0)
    );

    let multi = generator
        .get_group_by_plate_multi(
            &PlateFilterMulti {
                plane: plane(),
                grouping: Grouping::Auto,
                aggregation: vec![Aggregation::Avg],
                object_class: vec![ObjectClass::Unset],
                column: vec![Column::AreaSizePx],
                color_schema: ColorSchema::default(),
                color_scale: ColorScale::default(),
                plate_size: PlateSize::Auto,
            },
            &View::Heatmap,
        )
        .unwrap();
    // A1 -> (row 0, col 0)
    assert_eq!(as_float(&multi[0].rows[0][0].value), Some(10.0));
    assert!(multi[0].rows[0][0].any_failed);
}

#[test]
fn well_shows_a_failed_fields_own_value_but_marks_it() {
    let generator = open(&[
        ObjectSpec::new("A1_01.tif", "ClassA", 1, 10),
        ObjectSpec::new("A1_02.tif", "ClassA", 1, 100),
    ]);
    mark_failed(&generator, "A1_02.tif");

    let single = generator
        .get_group_by_well(&well_filter("A1", Column::AreaSizePx), &View::List)
        .unwrap();
    let batched = generator
        .get_wells_for_plate(&wells_batch_filter(Column::AreaSizePx), &View::Heatmap)
        .unwrap();

    assert_eq!(as_float(&single.rows[1][1].value), Some(100.0));
    assert!(single.rows[1][1].failed);
    assert!(!single.rows[0][1].failed);
    // Field 02 -> heatmap position 1 (row 0, col 1).
    assert!(batched["A1"].rows[0][1].failed);
    assert!(!batched["A1"].rows[0][0].failed);
}

#[test]
fn grouped_by_image_marks_every_cell_of_a_failed_images_rows() {
    let generator = open(&[
        ObjectSpec::new("img1.tif", "ClassA", 1, 10),
        ObjectSpec::new("img2.tif", "ClassA", 1, 20),
    ]);
    mark_failed(&generator, "img2.tif");

    let normal = grouped(&generator, Aggregation::Avg);
    let transposed = grouped_transposed(&generator, None, no_page());

    for result in [&normal, &transposed] {
        assert!(result.rows[0].iter().all(|cell| !cell.failed));
        assert!(result.rows[1].iter().all(|cell| cell.failed));
    }
    // Its objects (found before the error) are still shown.
    assert_eq!(cell_value(&transposed.rows[1][2]), Some(20.0));
}

#[test]
fn transposed_grouped_places_each_classs_values_in_its_own_block() {
    let generator = open(&[
        ObjectSpec::new("img1.tif", "ClassA", 1, 10),
        ObjectSpec::new("img1.tif", "ClassA", 1, 30),
        ObjectSpec::new("img1.tif", "ClassB", 2, 5),
        ObjectSpec::new("img2.tif", "ClassB", 2, 7),
    ]);

    let result = grouped_transposed(&generator, None, no_page());

    let values = |row: usize| -> Vec<Option<f64>> {
        result.rows[row][1..]
            .iter()
            .map(|c| as_float(&c.value))
            .collect()
    };
    // (Count, Avg area) for ClassA, then for ClassB.
    assert_eq!(values(0), [Some(2.0), Some(20.0), Some(1.0), Some(5.0)]);
    assert_eq!(values(1), [Some(0.0), None, Some(1.0), Some(7.0)]);
}

#[test]
fn class_filter_matches_objects_with_several_classes() {
    let generator = open(&[
        ObjectSpec::new("A1_01.tif", "ClassA", 1, 10),
        ObjectSpec::new("A1_01.tif", "ClassB", 2, 20),
    ]);
    // The first object is in both classes.
    generator
        .database
        .execute(
            "UPDATE objects SET object_class_id = '[1, 2]' WHERE area_px = 10",
            [],
        )
        .unwrap();
    let count_of = |id| PlateFilter {
        object_class: ObjectClass::Global(id),
        ..plate_filter(Column::Count)
    };

    assert_eq!(plate_value(&generator, &count_of(1), "A1"), Some(1.0));
    assert_eq!(plate_value(&generator, &count_of(2), "A1"), Some(2.0));
    assert_eq!(plate_value(&generator, &count_of(3), "A1"), Some(0.0));
}

#[test]
fn grouped_by_image_counts_a_multi_class_object_in_each_of_its_classes() {
    // A multi-class object lands in the group of each of its classes.
    let generator = open(&[
        ObjectSpec::new("img1.tif", "ClassA", 1, 10),
        ObjectSpec::new("img1.tif", "ClassB", 2, 20),
        ObjectSpec::new("img1.tif", "ClassB", 2, 40),
    ]);
    generator
        .database
        .execute_batch(
            "UPDATE objects SET object_class_id = '[1, 2]' WHERE area_px = 10;
             UPDATE objects SET object_class_id = '[ 2 ]' WHERE area_px = 40;",
        )
        .unwrap();

    let rows = grouped_rows(&grouped(&generator, Aggregation::Sum));

    let row =
        |class: &str, count, sum| ("img1.tif".to_string(), class.to_string(), vec![count, sum]);
    assert_eq!(
        rows,
        [
            row("ClassA", Some(1.0), Some(10.0)),
            row("ClassB", Some(3.0), Some(70.0)),
        ]
    );
}

// -- intensity lists and colocalization map ---------------------------------

const PARTNER: &str = "00000000-0000-0000-0000-000000000001";

#[test]
fn plate_aggregates_a_colocalization_count() {
    // Used to fail with an SQL error: the whole expression was prefixed
    // with `o.` (`o.COALESCE(...)`).
    let generator = open(&[
        ObjectSpec::new("A1_01.tif", "ClassA", 1, 10)
            .with_coloc(&format!(r#"{{"2":["{PARTNER}","{PARTNER}"]}}"#)),
        ObjectSpec::new("A1_01.tif", "ClassA", 1, 20),
    ]);

    let value = plate_value(
        &generator,
        &PlateFilter {
            aggregation: Aggregation::Sum,
            ..plate_filter(Column::ColocCount(ObjectClass::Global(2)))
        },
        "A1",
    );

    assert_eq!(value, Some(2.0));
}

#[test]
fn plate_aggregates_an_intensity_and_leaves_out_unmeasured_channels() {
    let channel_0 = |mean: f64| format!(r#"{{"0":{{"mean_normalized":0.1,"mean_gray":{mean}}}}}"#);
    let generator = open(&[
        ObjectSpec::new("A1_01.tif", "ClassA", 1, 10).with_intensities(&channel_0(100.0)),
        ObjectSpec::new("A1_01.tif", "ClassA", 1, 10).with_intensities(&channel_0(300.0)),
        // Measured in channel 1 only - no channel-0 value to average.
        ObjectSpec::new("A1_01.tif", "ClassA", 1, 10)
            .with_intensities(r#"{"1":{"mean_normalized":0.9,"mean_gray":900.0}}"#),
    ]);

    let mean_of = |channel| {
        plate_value(
            &generator,
            &plate_filter(Column::IntensityAvg(channel)),
            "A1",
        )
    };

    assert_eq!(mean_of(0), Some(200.0));
    assert_eq!(mean_of(1), Some(900.0));
}

#[test]
fn object_list_shows_intensities_in_gray_values_and_zero_for_an_unmeasured_channel() {
    let generator = open(&[
        ObjectSpec::new("img1.tif", "ClassA", 1, 10).with_intensities(
            r#"{"1":{"sum_normalized":0.5,"sum_gray":500.0,"mean_normalized":0.2,"mean_gray":200.0,
                 "min_normalized":0.1,"min_gray":100.0,"max_normalized":0.3,"max_gray":300.0}}"#,
        ),
    ]);

    let result = generator
        .get_object_list(&ListFilter {
            plane: plane(),
            images: crate::results::ImageSet::All,
            object_classes: None,
            columns: vec![
                Column::IntensityAvg(1),
                Column::IntensitySum(1),
                Column::IntensityMin(1),
                Column::IntensityMax(1),
                Column::IntensityAvg(0),
            ],
            with_coloc_details: false,
            related_to: None,
            page: no_page(),
            transpond_table: false,
            sort: None,
        })
        .unwrap();

    // The list orders its columns: channel 0 first.
    let values: Vec<Option<f64>> = result.rows[0].iter().map(cell_value).collect();
    assert_eq!(
        values,
        [
            Some(0.0),
            Some(200.0),
            Some(500.0),
            Some(100.0),
            Some(300.0)
        ]
    );
}

// -- measurements -----------------------------------------------------

/// A plate of 3 images in A1 (A1_03 without objects) and 1 in B2, with
/// an image measurement `p` and an object measurement `q`:
///
/// | image  | objects | p   | q per object |
/// |--------|---------|-----|--------------|
/// | A1_01  | 2       | 0.2 | 0.1, 0.3     |
/// | A1_02  | 1       | 0.4 | 0.6          |
/// | A1_03  | 0       | 0.9 | -            |
/// | B2_01  | 1       | 0.5 | 0.8          |
fn with_measurements() -> (TestResults, Column, Column) {
    let generator = open(&[
        ObjectSpec::new("A1_01.tif", "ClassA", 1, 10),
        ObjectSpec::new("A1_01.tif", "ClassA", 1, 20),
        ObjectSpec::new("A1_02.tif", "ClassA", 1, 30),
        ObjectSpec::new("B2_01.tif", "ClassA", 1, 40),
    ]);
    add_image(&generator, ImageRow::analysed("A1_03.tif"));
    generator
        .database
        .execute_batch(
            "CREATE TABLE measurement_info (key VARCHAR PRIMARY KEY, measurement VARCHAR, \
             step VARCHAR, unit VARCHAR, target VARCHAR, range_min DOUBLE, \
             range_max DOUBLE, aggregatable BOOLEAN, object_index INTEGER);
             CREATE TABLE measurements (image_name VARCHAR, image_rel_path VARCHAR, \
             z_stack INTEGER, t_stack INTEGER, region VARCHAR, object_id UUID, \
             related_object_id UUID, object_class_name VARCHAR, object_class_id INTEGER[], \
             step VARCHAR, measurement VARCHAR, unit VARCHAR, key VARCHAR, value DOUBLE);
             INSERT INTO measurement_info VALUES
                 ('p_01/p', 'P', 'p_01', 'None', 'Image', -1, 1, true, NULL),
                 ('q_01/q', 'Q', 'q_01', 'None', 'Object', -1, 1, true, 1);
             INSERT INTO measurements (image_rel_path, z_stack, t_stack, region, key, value)
             VALUES ('A1_01.tif', 0, 0, 'image', 'p_01/p', 0.2),
                    ('A1_02.tif', 0, 0, 'image', 'p_01/p', 0.4),
                    ('A1_03.tif', 0, 0, 'image', 'p_01/p', 0.9),
                    ('B2_01.tif', 0, 0, 'image', 'p_01/p', 0.5);",
        )
        .unwrap();
    // q per object, in the order the objects were seeded.
    let ids: Vec<String> = generator
        .database
        .prepare("SELECT object_id::VARCHAR FROM objects ORDER BY area_px")
        .unwrap()
        .query_map([], |row| row.get(0))
        .unwrap()
        .map(Result::unwrap)
        .collect();
    for (id, q) in ids.iter().zip([0.1, 0.3, 0.6, 0.8]) {
        generator
            .database
            .execute(
                "INSERT INTO measurements (image_rel_path, z_stack, t_stack, region, \
                 object_id, key, value) VALUES ('', 0, 0, 'object', ?::UUID, 'q_01/q', ?)",
                duckdb::params![id, q],
            )
            .unwrap();
        // ... and on the object, as the analysis writes it.
        generator
            .database
            .execute(
                "UPDATE objects SET measurement_values = [?] WHERE object_id = ?::UUID",
                duckdb::params![q, id],
            )
            .unwrap();
    }
    let column = |key: &str, step: &str, name: &str, per_object| {
        Column::Measurement(MeasurementColumn {
            key: key.into(),
            step: step.into(),
            name: name.into(),
            per_object,
        })
    };
    (
        generator,
        column("p_01/p", "p_01", "P", false),
        column("q_01/q", "q_01", "Q", true),
    )
}

fn rounded(value: Option<f64>) -> Option<f64> {
    value.map(|v| (v * 1000.0).round() / 1000.0)
}

#[test]
fn measurements_are_columns_of_their_results() {
    let (generator, p, q) = with_measurements();
    let measurements: Vec<Column> = generator
        .get_available_columns()
        .unwrap()
        .into_iter()
        .filter(|c| c.group == "Measurements")
        .map(|c| c.key)
        .collect();
    assert_eq!(measurements, [p.clone(), q.clone()]);
    assert_eq!(p.display_label(&[]), "p_01 · P");
    assert!(
        open(&[]).get_measurement_columns().unwrap().is_empty(),
        "older results"
    );
}

/// Each object row: its image's value and its own.
#[test]
fn the_object_list_shows_the_images_and_the_objects_own_value() {
    let (generator, p, q) = with_measurements();
    let result = generator
        .get_object_list(&ListFilter {
            plane: plane(),
            images: crate::results::ImageSet::All,
            object_classes: None,
            columns: vec![Column::AreaSizePx, p, q],
            with_coloc_details: false,
            related_to: None,
            page: no_page(),
            transpond_table: false,
            sort: None,
        })
        .unwrap();
    let mut rows: Vec<(i32, Option<f64>, Option<f64>)> = result
        .rows
        .iter()
        .map(|row| {
            let CellValue::Integer(area) = row[0].value else {
                panic!("the area");
            };
            (
                area,
                rounded(as_float(&row[1].value)),
                rounded(as_float(&row[2].value)),
            )
        })
        .collect();
    rows.sort_by_key(|row| row.0);
    assert_eq!(
        rows,
        [
            (10, Some(0.2), Some(0.1)),
            (20, Some(0.2), Some(0.3)),
            (30, Some(0.4), Some(0.6)),
            (40, Some(0.5), Some(0.8)),
        ]
    );
}

/// Per image: the image's own value - also without objects - and the
/// objects' average.
#[test]
fn per_image_an_image_measurement_is_the_images_value() {
    let (generator, p, q) = with_measurements();
    let result = generator
        .get_grouped_by_image(&GroupedByImageFilter {
            plane: plane(),
            images: crate::results::ImageSet::All,
            object_classes: None,
            columns: vec![p, q],
            aggregation: vec![Aggregation::Avg],
            page: no_page(),
            transpond_table: false,
            grouping: None,
            empty_classes: true,
            sort: None,
        })
        .unwrap();
    let rows: Vec<(String, Option<f64>, Option<f64>)> = result
        .rows
        .iter()
        .map(|row| {
            (
                match &row[0].value {
                    CellValue::String(s) => s.clone(),
                    _ => panic!("the image's name"),
                },
                rounded(as_float(&row[2].value)),
                rounded(as_float(&row[3].value)),
            )
        })
        .collect();
    assert_eq!(
        rows,
        [
            ("A1_01.tif".to_string(), Some(0.2), Some(0.2)),
            ("A1_02.tif".to_string(), Some(0.4), Some(0.6)),
            ("A1_03.tif".to_string(), Some(0.9), None),
            ("B2_01.tif".to_string(), Some(0.5), Some(0.8)),
        ]
    );
}

/// A well's image measurement is over its images (each once, also one
/// without objects); its object measurement over its objects.
#[test]
fn per_well_an_image_measurement_is_over_the_images() {
    let (generator, p, q) = with_measurements();
    let a1 =
        |column: &Column| rounded(plate_value(&generator, &plate_filter(column.clone()), "A1"));
    // (0.2 + 0.4 + 0.9) / 3 - not (2·0.2 + 0.4) / 3 over the objects.
    assert_eq!(a1(&p), Some(0.5));
    assert_eq!(a1(&q), Some(0.333));

    let multi = generator
        .get_group_by_plate_multi(
            &PlateFilterMulti {
                plane: plane(),
                grouping: Grouping::Auto,
                aggregation: vec![Aggregation::Avg, Aggregation::Max],
                object_class: vec![ObjectClass::Unset],
                column: vec![p.clone(), q.clone()],
                color_schema: ColorSchema::default(),
                color_scale: ColorScale::default(),
                plate_size: PlateSize::Auto,
            },
            &View::List,
        )
        .unwrap();
    let values: Vec<Option<f64>> = multi
        .iter()
        .map(|result| {
            let idx = result.row_names.iter().position(|n| n == "A1").unwrap();
            rounded(as_float(&result.rows[idx][1].value))
        })
        .collect();
    // (p, Avg), (p, Max), (q, Avg), (q, Max)
    assert_eq!(values, [Some(0.5), Some(0.9), Some(0.333), Some(0.6)]);

    // A well's fields: each image's own value.
    let fields = |result: &DatabaseResult| -> Vec<Option<f64>> {
        result
            .rows
            .iter()
            .map(|row| rounded(as_float(&row[1].value)))
            .collect()
    };
    let well = generator
        .get_group_by_well(&well_filter("A1", p.clone()), &View::List)
        .unwrap();
    assert_eq!(fields(&well), [Some(0.2), Some(0.4), Some(0.9)]);
    let batched = generator
        .get_wells_for_plate(&wells_batch_filter(p.clone()), &View::List)
        .unwrap();
    assert_eq!(fields(&batched["A1"]), fields(&well));
    let multi = generator
        .get_wells_for_plate_multi(
            &WellsBatchFilterMulti {
                plane: plane(),
                grouping: Grouping::Auto,
                aggregation: vec![Aggregation::Avg],
                object_class: vec![ObjectClass::Unset],
                column: vec![q, p],
                color_schema: ColorSchema::default(),
                color_scale: ColorScale::default(),
                well_size: None,
                well_order: None,
            },
            &View::List,
        )
        .unwrap();
    assert_eq!(
        fields(&multi[0]["A1"]),
        [Some(0.2), Some(0.6), None],
        "q: objects' average"
    );
    assert_eq!(fields(&multi[1]["A1"]), fields(&well), "p: the images' own");
}

/// An object measurement is charted like any per-object value.
#[test]
fn an_object_measurement_is_charted_over_the_objects() {
    let (generator, _, q) = with_measurements();
    let histogram = super::super::results_charts::ResultCharts {}
        .paint_histogram(
            &generator,
            &HistogramFilter {
                plane: plane(),
                images: None,
                object_classes: None,
                column: q,
                bins: 2,
                full_range: false,
            },
        )
        .unwrap();
    assert_eq!(
        (rounded(Some(histogram.min)), rounded(Some(histogram.max))),
        (Some(0.1), Some(0.8))
    );
    assert_eq!(histogram.counts.iter().sum::<u64>(), 4, "one per object");
}

/// A spread column (e.g. the median intensity of a channel) in the
/// list, per image and in a chart - the same values everywhere; an
/// object without it (a stored annotation) has none.
#[test]
fn a_spread_column_is_read_in_every_view() {
    let generator = open(&[
        ObjectSpec::new("A1_01.tif", "ClassA", 1, 10),
        ObjectSpec::new("A1_01.tif", "ClassA", 1, 20),
        ObjectSpec::new("A1_02.tif", "ClassA", 1, 30),
    ]);
    generator
        .database
        .execute_batch(
            "UPDATE objects SET intensity_median_gray = [CAST(area_px AS DOUBLE) / 10, 99.0] \
             WHERE area_px < 30",
        )
        .unwrap();
    let median = Column::IntensityMedian(0);
    assert!(
        generator
            .get_available_columns()
            .unwrap()
            .iter()
            .any(|c| c.key == median),
        "offered per channel"
    );
    assert_eq!(
        Column::from_key(&median.as_key(&[]), &[]),
        Some(median.clone())
    );

    let list = generator
        .get_object_list(&ListFilter {
            plane: plane(),
            images: crate::results::ImageSet::All,
            object_classes: None,
            columns: vec![Column::AreaSizePx, median.clone()],
            with_coloc_details: false,
            related_to: None,
            page: no_page(),
            transpond_table: false,
            sort: None,
        })
        .unwrap();
    let mut rows: Vec<(i32, Option<f64>)> = list
        .rows
        .iter()
        .map(|row| {
            let CellValue::Integer(area) = row[0].value else {
                panic!("the area");
            };
            (area, rounded(as_float(&row[1].value)))
        })
        .collect();
    rows.sort_by_key(|row| row.0);
    assert_eq!(rows, [(10, Some(1.0)), (20, Some(2.0)), (30, None)]);

    let per_image = generator
        .get_grouped_by_image(&GroupedByImageFilter {
            plane: plane(),
            images: crate::results::ImageSet::All,
            object_classes: None,
            columns: vec![median.clone()],
            aggregation: vec![Aggregation::Avg],
            page: no_page(),
            transpond_table: false,
            grouping: None,
            empty_classes: true,
            sort: None,
        })
        .unwrap();
    let averages: Vec<Option<f64>> = per_image
        .rows
        .iter()
        .map(|row| rounded(as_float(&row[2].value)))
        .collect();
    assert_eq!(averages, [Some(1.5), None]);

    let histogram = super::super::results_charts::ResultCharts {}
        .paint_histogram(
            &generator,
            &HistogramFilter {
                plane: plane(),
                images: None,
                object_classes: None,
                column: median,
                bins: 2,
                full_range: false,
            },
        )
        .unwrap();
    assert_eq!(
        histogram.counts.iter().sum::<u64>(),
        2,
        "objects with a value"
    );
}

/// Shape columns: stored ones (max Feret) and computed ones (extent:
/// area over the bounding box's) - in the list and per image.
#[test]
fn shape_columns_are_read_and_computed() {
    let generator = open(&[
        ObjectSpec::new("A1_01.tif", "ClassA", 1, 60),
        ObjectSpec::new("A1_01.tif", "ClassA", 1, 30),
    ]);
    // Every seeded object's box is 0..=10 (121 pixels).
    generator
        .database
        .execute_batch("UPDATE objects SET feret_diameter_px = area_px / 10.0")
        .unwrap();
    let (feret, extent) = (Column::MaxFeret, Column::Extent);
    for column in [&feret, &extent] {
        assert_eq!(
            Column::from_key(&column.as_key(&[]), &[]),
            Some(column.clone())
        );
    }
    let list = generator
        .get_object_list(&ListFilter {
            plane: plane(),
            images: crate::results::ImageSet::All,
            object_classes: None,
            columns: vec![Column::AreaSizePx, feret.clone(), extent.clone()],
            with_coloc_details: false,
            related_to: None,
            page: no_page(),
            transpond_table: false,
            sort: None,
        })
        .unwrap();
    let mut rows: Vec<(i32, Option<f64>, Option<f64>)> = list
        .rows
        .iter()
        .map(|row| {
            let CellValue::Integer(area) = row[0].value else {
                panic!("area")
            };
            (
                area,
                rounded(as_float(&row[1].value)),
                rounded(as_float(&row[2].value)),
            )
        })
        .collect();
    rows.sort_by_key(|r| r.0);
    assert_eq!(
        rows,
        [
            (30, Some(3.0), rounded(Some(30.0 / 121.0))),
            (60, Some(6.0), rounded(Some(60.0 / 121.0)))
        ]
    );
    let per_image = generator
        .get_grouped_by_image(&GroupedByImageFilter {
            plane: plane(),
            images: crate::results::ImageSet::All,
            object_classes: None,
            columns: vec![extent],
            aggregation: vec![Aggregation::Avg],
            page: no_page(),
            transpond_table: false,
            grouping: None,
            empty_classes: true,
            sort: None,
        })
        .unwrap();
    assert_eq!(
        rounded(as_float(&per_image.rows[0][2].value)),
        rounded(Some(45.0 / 121.0))
    );
}

/// A measurement whose values name another object (the nearest
/// neighbour) offers that object as a column; a click on it - or on a
/// parent's id - lists that one object (`RelatedTo::RelatedOf` /
/// `ParentOf`).
#[test]
fn the_object_a_value_names_is_a_column_and_can_be_listed() {
    let generator = open(&[
        ObjectSpec::new("A1_01.tif", "ClassA", 1, 10),
        ObjectSpec::new("A1_01.tif", "ClassA", 1, 20),
        ObjectSpec::new("A1_01.tif", "ClassA", 1, 30),
    ]);
    let ids: Vec<String> = generator
        .database
        .prepare("SELECT object_id::VARCHAR FROM objects ORDER BY area_px")
        .unwrap()
        .query_map([], |row| row.get(0))
        .unwrap()
        .map(Result::unwrap)
        .collect();
    let (a, b, c) = (&ids[0], &ids[1], &ids[2]);
    generator
        .database
        .execute_batch(&format!(
            "CREATE TABLE measurement_info (key VARCHAR PRIMARY KEY, measurement VARCHAR, \
             step VARCHAR, unit VARCHAR, target VARCHAR, range_min DOUBLE, \
             range_max DOUBLE, aggregatable BOOLEAN, object_index INTEGER, \
             related_objects BOOLEAN);
             CREATE TABLE measurements (image_name VARCHAR, image_rel_path VARCHAR, \
             z_stack INTEGER, t_stack INTEGER, region VARCHAR, object_id UUID, \
             related_object_id UUID, object_class_name VARCHAR, object_class_id INTEGER[], \
             step VARCHAR, measurement VARCHAR, unit VARCHAR, key VARCHAR, value DOUBLE);
             INSERT INTO measurement_info VALUES
                 ('n_01/first', 'First closest', 'n_01', 'Pixels', 'Object', NULL, NULL,
                  true, 1, true);
             INSERT INTO measurements (image_rel_path, z_stack, t_stack, region, object_id,
                 related_object_id, key, value) VALUES
                 ('A1_01.tif', 0, 0, 'object', '{a}', '{b}', 'n_01/first', 3.0),
                 ('A1_01.tif', 0, 0, 'object', '{b}', '{a}', 'n_01/first', 3.0),
                 ('A1_01.tif', 0, 0, 'object', '{c}', '{b}', 'n_01/first', 5.0);
             UPDATE objects SET parent_ids = MAP {{1: '{c}'::UUID}} WHERE object_id = '{a}';"
        ))
        .unwrap();
    let first = generator
        .get_available_columns()
        .unwrap()
        .into_iter()
        .find_map(|entry| match entry.key {
            Column::MeasurementObject(m) => Some(m),
            _ => None,
        })
        .expect("the named object is offered");
    assert_eq!(first.key, "n_01/first");
    let list = |columns: Vec<Column>, related_to: Option<RelatedTo>| {
        generator
            .get_object_list(&ListFilter {
                plane: plane(),
                images: crate::results::ImageSet::All,
                object_classes: None,
                columns,
                with_coloc_details: false,
                related_to,
                page: no_page(),
                transpond_table: false,
                sort: None,
            })
            .unwrap()
    };
    let rows = list(
        vec![Column::AreaSizePx, Column::MeasurementObject(first.clone())],
        None,
    );
    let named: Vec<(i32, String)> = rows
        .rows
        .iter()
        .map(|row| {
            let CellValue::Integer(area) = row[0].value else {
                panic!("area")
            };
            let CellValue::String(id) = &row[1].value else {
                panic!("an id")
            };
            (area, id.clone())
        })
        .collect();
    let mut named = named;
    named.sort();
    assert_eq!(named, [(10, b.clone()), (20, a.clone()), (30, b.clone())]);
    // Clicked: the object A's value names - B.
    let opened = list(
        vec![Column::AreaSizePx],
        Some(RelatedTo::RelatedOf(first.key.clone(), a.clone())),
    );
    assert_eq!(opened.row_names, [b.clone()]);
    // A's parent of class 1 - C.
    let parent = list(
        vec![Column::AreaSizePx],
        Some(RelatedTo::ParentOf(ObjectClass::Global(1), a.clone())),
    );
    assert_eq!(parent.row_names, [c.clone()]);
}

// -- more list, grouped and status cases --------------------------------

/// A spot's children of every class (no child class): a cell's spots
/// all the same - cell 0 has two, cell 1 one.
#[test]
fn children_of_any_class_are_every_object_linked_to_the_parent() {
    let generator = cells_with_spots();
    let any_child = Column::Children(ChildColumn {
        parent_class: CELL,
        child_class: ObjectClass::Unset,
        value: ChildValue::Count,
    });
    let result = list_of(&generator, CELL, vec![Column::AreaSizePx, any_child]);
    let rows: Vec<Vec<Option<f64>>> = result
        .rows
        .iter()
        .map(|row| row.iter().map(number).collect())
        .collect();
    assert_eq!(
        rows,
        [vec![Some(100.0), Some(2.0)], vec![Some(200.0), Some(1.0)]]
    );
}

/// A parent's id is a text, not a number: aggregating it is refused.
#[test]
fn a_parents_id_cannot_be_aggregated() {
    let error = column_aggregate_expr(&parent(ParentValue::Id)).expect_err("an id");
    assert!(
        matches!(&error, InternalErrors::InvalidArgument(m) if m.contains("parent's id")),
        "{error}"
    );
}

/// A page without objects - no image of the selection has any - is
/// empty, the relation and measurement columns too, not an error.
#[test]
fn a_page_without_objects_with_relation_and_measurement_columns_is_empty() {
    let (generator, p, q) = with_measurements();
    let named = Column::MeasurementObject(MeasurementColumn {
        key: "q_01/q".into(),
        step: "q_01".into(),
        name: "Q".into(),
        per_object: true,
    });
    let result = generator
        .get_object_list(&ListFilter {
            plane: plane(),
            images: crate::results::ImageSet::Listed(vec!["missing.tif".into()]),
            object_classes: None,
            columns: vec![Column::AreaSizePx, children(ChildValue::Count), p, q, named],
            with_coloc_details: false,
            related_to: None,
            page: no_page(),
            transpond_table: false,
            sort: None,
        })
        .unwrap();
    assert!(result.rows.is_empty());
    assert_eq!(result.source_object_count, 0);
}

/// A class selection without a single class id: no rows per image.
#[test]
fn grouped_by_image_of_a_selection_without_a_class_id_is_empty() {
    let generator = open(&[ObjectSpec::new("img1.tif", "ClassA", 1, 10)]);
    let result = generator
        .get_grouped_by_image(&GroupedByImageFilter {
            plane: plane(),
            images: crate::results::ImageSet::All,
            object_classes: Some(vec![ObjectClass::Unset]),
            columns: vec![Column::AreaSizePx],
            aggregation: vec![Aggregation::Avg],
            page: no_page(),
            transpond_table: false,
            grouping: None,
            empty_classes: true,
            sort: None,
        })
        .unwrap();
    assert!(result.rows.is_empty());
}

/// Transposed, an image measurement is the image's own value beside
/// the objects' average (the values of
/// `per_image_an_image_measurement_is_the_images_value`) - the images
/// with objects; A1_03 without any is left out here.
#[test]
fn transposed_grouped_shows_an_image_measurement_beside_the_objects_values() {
    let (generator, p, q) = with_measurements();
    let result = generator
        .get_grouped_by_image(&GroupedByImageFilter {
            plane: plane(),
            images: crate::results::ImageSet::All,
            object_classes: None,
            columns: vec![p, q],
            aggregation: vec![Aggregation::Avg],
            page: no_page(),
            transpond_table: true,
            grouping: None,
            empty_classes: true,
            sort: None,
        })
        .unwrap();
    assert_eq!(
        result.column_names.len(),
        1 + 2,
        "{:?}",
        result.column_names
    );
    let rows: Vec<(String, Option<f64>, Option<f64>)> = result
        .rows
        .iter()
        .map(|row| {
            (
                match &row[0].value {
                    CellValue::String(s) => s.clone(),
                    _ => panic!("the image's name"),
                },
                rounded(as_float(&row[1].value)),
                rounded(as_float(&row[2].value)),
            )
        })
        .collect();
    // A1_03 has no objects: its image measurement still shows, as in
    // the normal view (`per_image_an_image_measurement_is_the_images_value`).
    assert_eq!(
        rows,
        [
            ("A1_01.tif".to_string(), Some(0.2), Some(0.2)),
            ("A1_02.tif".to_string(), Some(0.4), Some(0.6)),
            ("A1_03.tif".to_string(), Some(0.9), None),
            ("B2_01.tif".to_string(), Some(0.5), Some(0.8)),
        ]
    );
}

/// A transposed list page of 10 rows over five images of one object
/// each: the first four images don't fill it, so more are numbered -
/// all five, in order.
#[test]
fn a_transposed_list_page_reaches_past_the_first_images() {
    let generator = open(&[
        ObjectSpec::new("img1.tif", "ClassA", 1, 1),
        ObjectSpec::new("img2.tif", "ClassA", 1, 2),
        ObjectSpec::new("img3.tif", "ClassA", 1, 3),
        ObjectSpec::new("img4.tif", "ClassA", 1, 4),
        ObjectSpec::new("img5.tif", "ClassA", 1, 5),
    ]);
    let page = list_transposed(
        &generator,
        None,
        Pagination {
            limit: 10,
            after: None,
        },
    );
    let areas: Vec<Option<u64>> = page.rows.iter().map(|row| area(&row[1])).collect();
    assert_eq!(areas, [Some(1), Some(2), Some(3), Some(4), Some(5)]);
    // After the last image's last row: nothing more.
    let next = list_transposed(
        &generator,
        None,
        Pagination {
            limit: 10,
            after: page.row_names.last().cloned(),
        },
    );
    assert!(next.rows.is_empty());
    assert_eq!(next.column_names, ["image", "Area [px] (ClassA)"]);
}

/// A run table without a row: unknown, like results without one; one
/// that can't be read is an error, not a status.
#[test]
fn an_empty_run_table_is_unknown_and_an_unreadable_one_an_error() {
    use crate::results::RunStatus;
    let db = open(&[]);
    db.database
        .execute_batch("CREATE TABLE run (status VARCHAR, message VARCHAR);")
        .unwrap();
    assert_eq!(db.run_status().unwrap(), RunStatus::Unknown);

    let broken = open(&[]);
    broken
        .database
        .execute_batch("CREATE TABLE run (status VARCHAR);")
        .unwrap();
    assert!(matches!(broken.run_status(), Err(InternalErrors::Io(_))));
}

/// Results from before the intensity spread (SD, median, quartiles, MAD)
/// don't offer those columns - a query of them would fail on the file.
#[test]
fn an_older_file_offers_only_the_intensity_columns_it_stores() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("results.evadb");
    seed_db(
        &path,
        &[ObjectSpec::new("img1.tif", "A", 1, 10).with_intensities(r#"{"0":{"mean":5.0}}"#)],
    );
    let offered = |path: &std::path::Path| -> Vec<Column> {
        ResultsGenerator::open_database(path.into())
            .unwrap()
            .get_available_columns()
            .unwrap()
            .into_iter()
            .map(|entry| entry.key)
            .collect()
    };
    assert!(
        offered(&path).contains(&Column::IntensityStd(0)),
        "a file that has it"
    );
    {
        let conn = duckdb::Connection::open(&path).unwrap();
        conn.execute_batch(
            "ALTER TABLE objects DROP COLUMN intensity_std_gray;
             ALTER TABLE objects DROP COLUMN intensity_mad_gray;",
        )
        .unwrap();
    }
    let columns = offered(&path);
    assert!(!columns.contains(&Column::IntensityStd(0)));
    assert!(!columns.contains(&Column::IntensityMad(0)));
    assert!(columns.contains(&Column::IntensityAvg(0)));
    assert!(columns.contains(&Column::IntensityMedian(0)));
}
