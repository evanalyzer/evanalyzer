//! # tests
//!
//! The plate layouts and image heatmaps (moved with their writer from the
//! old results export on 2026-11-25): grids, lists, tabs per plane, the
//! same values as the Results' plate queries.
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-11-25
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.

use super::*;
use crate::results::local::LocalResults;
use crate::results::local::test_support::{ObjectSpec, TestResults, seed_db};
use crate::results::{PlateFilter, WellsBatchFilter};
use calamine::Reader as _;
use std::path::PathBuf;
use std::range::Range;

fn data_f64(data: Option<&calamine::Data>) -> f64 {
    match data {
        Some(calamine::Data::Float(v)) => *v,
        Some(calamine::Data::Int(v)) => *v as f64,
        other => panic!("expected a numeric cell, got {other:?}"),
    }
}
/// `database` as the results an export runs on (its own connection).
fn local(database: &TestResults) -> LocalResults {
    LocalResults::from_generator(database.try_clone().expect("a second connection"))
}

/// Opens a fresh `ResultsGenerator` over a temp `.evadb` seeded with
/// `objects` (see `test_support::seed_db`), and a folder to export into
/// in the same temp folder - which goes with the results.
fn open(objects: &[ObjectSpec]) -> (TestResults, PathBuf) {
    let dir = tempfile::tempdir().expect("tempdir");
    let db_path = dir.path().join("results.evadb");
    seed_db(&db_path, objects);
    let out_dir = dir.path().join("out");
    (TestResults::open(dir, &db_path), out_dir)
}
/// Same as `open`, plus one extra `images` row with no objects at all
/// (a well/field that has an image but zero detected objects) -
/// `ResultsGenerator`'s own `database` connection is private even to
/// this sibling test module, and `seed_db`/`ObjectSpec` only ever
/// create an image alongside a seeded object, so the raw insert has to
/// happen through its own connection, before `ResultsGenerator` ever
/// opens the file (matching how `seed_db` itself closes its connection
/// before returning).
fn open_with_empty_image(objects: &[ObjectSpec], empty_image_name: &str) -> (TestResults, PathBuf) {
    let dir = tempfile::tempdir().expect("tempdir");
    let db_path = dir.path().join("results.evadb");
    seed_db(&db_path, objects);
    {
        let conn = duckdb::Connection::open(&db_path).expect("open db for extra image");
        conn.execute(
            "INSERT INTO images (image_name, image_rel_path, width, height, c_stacks, z_stacks, t_stacks) \
             VALUES (?, ?, 100, 100, 1, 1, 1)",
            duckdb::params![empty_image_name, empty_image_name],
        )
        .expect("insert empty image");
    }
    let out_dir = dir.path().join("out");
    (TestResults::open(dir, &db_path), out_dir)
}
/// `open`, with the planes a run records analysing
/// (`images.t_stacks_analyzed`/`z_stacks_analyzed`): every image at every time point
/// `0..times` (one z-plane) - also those where nothing was found.
fn open_analysed(objects: &[ObjectSpec], times: i32) -> (TestResults, PathBuf) {
    let dir = tempfile::tempdir().expect("tempdir");
    let db_path = dir.path().join("results.evadb");
    seed_db(&db_path, objects);
    {
        let conn = duckdb::Connection::open(&db_path).expect("open db");
        conn.execute_batch(
            "ALTER TABLE images ADD COLUMN t_stacks_analyzed INTEGER[];
             ALTER TABLE images ADD COLUMN z_stacks_analyzed INTEGER[];",
        )
        .unwrap();
        conn.execute(
            "UPDATE images SET t_stacks = ?, \
             t_stacks_analyzed = CAST(range(0, ?) AS INTEGER[]), z_stacks_analyzed = [0]",
            duckdb::params![times, times],
        )
        .unwrap();
    }
    let out_dir = dir.path().join("out");
    (TestResults::open(dir, &db_path), out_dir)
}
fn no_cancel() -> AtomicBool {
    AtomicBool::new(false)
}
/// Asserts that `range`'s grid starting at `start_row` (as written by
/// `write_grid_block`: a caption row, then a header row of
/// `expected.column_names`, then one row per `expected.row_names`)
/// matches `expected` cell-for-cell.
fn assert_grid_matches_at(
    range: &calamine::Range<calamine::Data>,
    start_row: u32,
    expected: &DatabaseResult,
) {
    let header_row = start_row + 1;
    for (col_idx, name) in expected.column_names.iter().enumerate() {
        let col = (col_idx + 1) as u32;
        assert_eq!(
            range.get_value((header_row, col)),
            Some(&calamine::Data::String(name.clone())),
            "column header at ({header_row}, {col})"
        );
    }
    for (row_idx, row_name) in expected.row_names.iter().enumerate() {
        let row = header_row + 1 + row_idx as u32;
        assert_eq!(
            range.get_value((row, 0)),
            Some(&calamine::Data::String(row_name.clone())),
            "row label at ({row}, 0)"
        );
        for (col_idx, cell) in expected.rows[row_idx].iter().enumerate() {
            let col = (col_idx + 1) as u32;
            match &cell.value {
                CellValue::Float(v) => {
                    let got = data_f64(range.get_value((row, col)));
                    assert!(
                        (got - *v as f64).abs() < 1e-6,
                        "value at ({row}, {col}): expected {v}, got {got}"
                    );
                }
                CellValue::Empty => {
                    assert!(
                        matches!(
                            range.get_value((row, col)),
                            None | Some(calamine::Data::Empty)
                        ),
                        "expected an empty cell at ({row}, {col})"
                    );
                }
                _ => {}
            }
        }
    }
}
/// Every z-plane and time point (`None`: all).
fn full_range() -> (Option<Range<u32>>, Option<Range<u32>>) {
    (None, None)
}
/// A plate grid for every analysed time point - the one where nothing
/// was found too.
#[test]
fn plate_grids_get_a_tab_for_an_empty_time_point_too() {
    let (database, out_dir) = open_analysed(&[ObjectSpec::new("A1_01.tif", "ClassA", 1, 10)], 2);
    PlateExport {
        output_dir: out_dir.clone(),
        columns: vec![Column::AreaSizePx],
        aggregations: vec![Aggregation::Avg],
        with_plate_view_heatmap: true,
        ..Default::default()
    }
    .write(&local(&database), &no_cancel())
    .expect("plate export");
    let workbook: calamine::Xlsx<_> =
        calamine::open_workbook(out_dir.join("plate.xlsx")).expect("open plate.xlsx");
    assert_eq!(workbook.sheet_names(), ["T0 ClassA", "T1 ClassA"]);
}
/// A plate grid per time point: a tab each, the time point first.
#[test]
fn plate_grids_get_a_tab_per_time_point() {
    let (database, out_dir) = open(&[
        ObjectSpec::new("A1_01.tif", "ClassA", 1, 10),
        ObjectSpec::new("A1_01.tif", "ClassA", 1, 30).at_plane(0, 1),
    ]);
    PlateExport {
        output_dir: out_dir.clone(),
        columns: vec![Column::AreaSizePx],
        aggregations: vec![Aggregation::Avg],
        with_plate_view_heatmap: true,
        ..Default::default()
    }
    .write(&local(&database), &no_cancel())
    .expect("plate export");
    let mut workbook: calamine::Xlsx<_> =
        calamine::open_workbook(out_dir.join("plate.xlsx")).expect("open plate.xlsx");
    assert_eq!(workbook.sheet_names(), ["T0 ClassA", "T1 ClassA"]);
    let t1 = workbook.worksheet_range("T1 ClassA").unwrap();
    assert_eq!(data_f64(t1.get_value((2, 1))), 30.0, "A1 at time point 1");
}
#[test]
fn plate_grid_matches_get_group_by_plate_heatmap() {
    let (database, out_dir) = open(&[
        ObjectSpec::new("A1_01.tif", "ClassA", 1, 10),
        ObjectSpec::new("A1_02.tif", "ClassA", 1, 20),
        ObjectSpec::new("B2_01.tif", "ClassA", 1, 100),
    ]);
    let export = PlateExport {
        output_dir: out_dir.clone(),
        columns: vec![Column::AreaSizePx],
        aggregations: vec![Aggregation::Avg],
        with_plate_view_heatmap: true,
        ..Default::default()
    };
    export
        .write(&local(&database), &no_cancel())
        .expect("plate export");

    // Same filter `export_plate_and_well` builds internally for this
    // (single class, single column, single aggregation) configuration.
    let expected = database
        .get_group_by_plate(
            &PlateFilter {
                plane: PlaneFilter {
                    z_stack: 0,
                    t_stack: 0,
                },
                grouping: Grouping::Auto,
                aggregation: Aggregation::Avg,
                object_class: ObjectClass::Global(1),
                column: Column::AreaSizePx,
                color_schema: ColorSchema::default(),
                color_scale: ColorScale::default(),
                plate_size: PlateSize::Auto,
            },
            &View::Heatmap,
        )
        .expect("expected plate grid");

    let mut workbook: calamine::Xlsx<_> =
        calamine::open_workbook(out_dir.join("plate.xlsx")).expect("open plate.xlsx");
    let range = workbook.worksheet_range("ClassA").expect("ClassA sheet");
    assert_grid_matches_at(&range, 0, &expected);

    // Belt-and-suspenders on the actual numbers, independent of the
    // `DatabaseResult` plumbing above.
    assert_eq!(data_f64(range.get_value((2, 1))), 15.0, "A1 = avg(10, 20)");
    assert_eq!(data_f64(range.get_value((3, 2))), 100.0, "B2");
}
#[test]
fn well_grid_matches_get_wells_for_plate_heatmap() {
    let (database, out_dir) = open(&[
        ObjectSpec::new("A1_01.tif", "ClassA", 1, 10),
        ObjectSpec::new("A1_02.tif", "ClassA", 1, 20),
    ]);
    let export = PlateExport {
        output_dir: out_dir.clone(),
        columns: vec![Column::AreaSizePx],
        aggregations: vec![Aggregation::Avg],
        with_well_view_heatmap: true,
        ..Default::default()
    };
    export
        .write(&local(&database), &no_cancel())
        .expect("well export");

    let expected_by_well = database
        .get_wells_for_plate(
            &WellsBatchFilter {
                plane: PlaneFilter {
                    z_stack: 0,
                    t_stack: 0,
                },
                grouping: Grouping::Auto,
                aggregation: Aggregation::Avg,
                object_class: ObjectClass::Global(1),
                column: Column::AreaSizePx,
                color_schema: ColorSchema::default(),
                color_scale: ColorScale::default(),
                well_size: None,
                well_order: None,
            },
            &View::Heatmap,
        )
        .expect("expected well grid");
    assert_eq!(expected_by_well.len(), 1, "only well A1 was seeded");
    let expected = &expected_by_well["A1"];

    let mut workbook: calamine::Xlsx<_> =
        calamine::open_workbook(out_dir.join("well.xlsx")).expect("open well.xlsx");
    let range = workbook.worksheet_range("ClassA").expect("ClassA sheet");
    assert_grid_matches_at(&range, 0, expected);
    assert_eq!(data_f64(range.get_value((2, 1))), 10.0, "field 01");
    assert_eq!(data_f64(range.get_value((2, 2))), 20.0, "field 02");
}
#[test]
fn heatmap_grid_matches_get_image_heatmap() {
    let (database, out_dir) = open(&[
        ObjectSpec::new("img1.tif", "ClassA", 1, 10)
            .at_centroid(10.0, 10.0)
            .with_image_size(100, 100),
        ObjectSpec::new("img1.tif", "ClassA", 1, 20)
            .at_centroid(60.0, 10.0)
            .with_image_size(100, 100),
    ]);
    let export = PlateExport {
        output_dir: out_dir.clone(),
        columns: vec![Column::AreaSizePx],
        aggregations: vec![Aggregation::Avg],
        square_size: Some(50),
        with_heatmap: true,
        ..Default::default()
    };
    export
        .write(&local(&database), &no_cancel())
        .expect("heatmap export");

    let expected = database
        .get_image_heatmap(
            &ImageHeatmapFilter {
                plane: PlaneFilter {
                    z_stack: 0,
                    t_stack: 0,
                },
                image_rel_path: "img1.tif".to_string(),
                aggregation: Aggregation::Avg,
                object_class: ObjectClass::Global(1),
                column: Column::AreaSizePx,
                color_schema: ColorSchema::default(),
                color_scale: ColorScale::default(),
                square_size: Some(50),
            },
            &View::Heatmap,
        )
        .expect("expected image heatmap grid");

    let mut workbook: calamine::Xlsx<_> =
        calamine::open_workbook(out_dir.join("heatmap_img1.xlsx")).expect("open heatmap_img1.xlsx");
    let range = workbook.worksheet_range("ClassA").expect("ClassA sheet");
    assert_grid_matches_at(&range, 0, &expected);
    assert_eq!(data_f64(range.get_value((2, 1))), 10.0, "R0C0");
    assert_eq!(data_f64(range.get_value((2, 2))), 20.0, "R0C1");
}
#[test]
fn plate_list_and_well_list_report_the_seeded_average() {
    let (database, out_dir) = open(&[
        ObjectSpec::new("A1_01.tif", "ClassA", 1, 10),
        ObjectSpec::new("A1_02.tif", "ClassA", 1, 20),
    ]);
    let export = PlateExport {
        output_dir: out_dir.clone(),
        columns: vec![Column::AreaSizePx],
        aggregations: vec![Aggregation::Avg],
        with_plate_view_list: true,
        with_well_view_list: true,
        ..Default::default()
    };
    export
        .write(&local(&database), &no_cancel())
        .expect("flat-pivot export");

    let mut plate_wb: calamine::Xlsx<_> =
        calamine::open_workbook(out_dir.join("plate_list.xlsx")).expect("open plate_list.xlsx");
    let plate_range = plate_wb.worksheet_range("Plate").expect("Plate sheet");
    assert_eq!(
        plate_range.get_value((0, 0)),
        Some(&calamine::Data::String("Well".to_string()))
    );
    assert_eq!(
        plate_range.get_value((1, 0)),
        Some(&calamine::Data::String("A1".to_string()))
    );
    assert_eq!(data_f64(plate_range.get_value((1, 1))), 15.0);

    let mut well_wb: calamine::Xlsx<_> =
        calamine::open_workbook(out_dir.join("well_list.xlsx")).expect("open well_list.xlsx");
    let well_range = well_wb.worksheet_range("Well").expect("Well sheet");
    assert_eq!(
        well_range.get_value((0, 0)),
        Some(&calamine::Data::String("Well".to_string()))
    );
    // Two data rows: field "01" (value 10) and field "02" (value 20),
    // sorted by (well, field-as-number) - see `export_plate_and_well_as_flat_list`.
    assert_eq!(data_f64(well_range.get_value((1, 3))), 10.0);
    assert_eq!(data_f64(well_range.get_value((2, 3))), 20.0);
}
/// Intensity columns are aggregable like every other measurement - they
/// were once silently dropped from every grouped/grid document.
#[test]
fn intensity_columns_are_in_the_plate_documents() {
    use crate::results::local::test_support::CH0_INTENSITIES_JSON;
    let (database, out_dir) = open(&[
        ObjectSpec::new("A1_01.tif", "ClassA", 1, 10).with_intensities(CH0_INTENSITIES_JSON),
        ObjectSpec::new("A1_02.tif", "ClassA", 1, 20).with_intensities(CH0_INTENSITIES_JSON),
    ]);
    let export = PlateExport {
        output_dir: out_dir.clone(),
        columns: vec![Column::AreaSizePx, Column::IntensityAvg(0)],
        aggregations: vec![Aggregation::Sum],
        with_plate_view_list: true,
        with_plate_view_heatmap: true,
        ..Default::default()
    };
    export
        .write(&local(&database), &no_cancel())
        .expect("export");

    // Plate list: one value column per column x aggregation.
    let mut plate_wb: calamine::Xlsx<_> =
        calamine::open_workbook(out_dir.join("plate_list.xlsx")).expect("open plate_list.xlsx");
    let plate_range = plate_wb.worksheet_range("Plate").expect("Plate sheet");
    let header: Vec<String> = (0..plate_range.width() as u32)
        .filter_map(|c| match plate_range.get_value((0, c)) {
            Some(calamine::Data::String(s)) => Some(s.clone()),
            _ => None,
        })
        .collect();
    let intensity_col = header
        .iter()
        .position(|h| h.contains("Avg Intensity (Ch 0)"))
        .unwrap_or_else(|| panic!("no intensity column in the plate list: {header:?}"));
    assert_eq!(
        data_f64(plate_range.get_value((1, intensity_col as u32))),
        254.0,
        "A1: sum of both objects' mean intensity"
    );

    // Plate heatmap: a grid per column, the intensity one included.
    let mut heatmap_wb: calamine::Xlsx<_> =
        calamine::open_workbook(out_dir.join("plate.xlsx")).expect("open plate.xlsx");
    let sheet = heatmap_wb.sheet_names()[0].clone();
    let heatmap_range = heatmap_wb.worksheet_range(&sheet).expect("heatmap sheet");
    let has_intensity_caption = heatmap_range.cells().any(|(_, _, value)| {
        matches!(value, calamine::Data::String(s) if s.contains("Avg Intensity (Ch 0)"))
    });
    assert!(
        has_intensity_caption,
        "no intensity grid in the plate heatmap"
    );
}
/// A well with an image but zero matching objects must still get a row
/// in the flat-pivot list export (not vanish for having nothing to
/// aggregate), with `"-"` in its value cell rather than a misleading `0`.
#[test]
fn plate_list_writes_a_dash_for_a_well_with_no_objects() {
    let (database, out_dir) = open_with_empty_image(
        &[ObjectSpec::new("A1_01.tif", "ClassA", 1, 10)],
        "B2_01.tif",
    );
    let export = PlateExport {
        output_dir: out_dir.clone(),
        columns: vec![Column::AreaSizePx],
        aggregations: vec![Aggregation::Avg],
        with_plate_view_list: true,
        ..Default::default()
    };
    export
        .write(&local(&database), &no_cancel())
        .expect("flat-pivot export");

    let mut plate_wb: calamine::Xlsx<_> =
        calamine::open_workbook(out_dir.join("plate_list.xlsx")).expect("open plate_list.xlsx");
    let plate_range = plate_wb.worksheet_range("Plate").expect("Plate sheet");
    assert_eq!(
        plate_range.get_value((1, 0)),
        Some(&calamine::Data::String("A1".to_string()))
    );
    assert_eq!(data_f64(plate_range.get_value((1, 1))), 10.0);
    assert_eq!(
        plate_range.get_value((2, 0)),
        Some(&calamine::Data::String("B2".to_string())),
        "B2 has an image but no objects and must still get a row"
    );
    assert_eq!(
        plate_range.get_value((2, 1)),
        Some(&calamine::Data::String("-".to_string())),
        "B2 has no matching objects, so its cell must read \"-\", not 0"
    );
}
/// A well made up only of disabled images must still get a row (its
/// images are real, just excluded from the statistics) - written as a
/// dash like any other well with no matching data, not struck through:
/// a well itself is never flagged disabled, only its individual images
/// are (see `plate_groups_to_result`).
#[test]
fn plate_list_still_includes_a_fully_disabled_well() {
    let (database, out_dir) = open(&[
        ObjectSpec::new("A1_01.tif", "ClassA", 1, 10),
        ObjectSpec::new("B2_01.tif", "ClassA", 1, 100),
    ]);
    database
        .enable_image("B2_01.tif", true)
        .expect("disable B2");
    let export = PlateExport {
        output_dir: out_dir.clone(),
        columns: vec![Column::AreaSizePx],
        aggregations: vec![Aggregation::Avg],
        with_plate_view_list: true,
        ..Default::default()
    };
    export
        .write(&local(&database), &no_cancel())
        .expect("flat-pivot export");

    let mut plate_wb: calamine::Xlsx<_> =
        calamine::open_workbook(out_dir.join("plate_list.xlsx")).expect("open plate_list.xlsx");
    let plate_range = plate_wb.worksheet_range("Plate").expect("Plate sheet");
    assert_eq!(
        plate_range.get_value((2, 0)),
        Some(&calamine::Data::String("B2".to_string())),
        "B2 must still appear even though its only image is disabled"
    );
    assert_eq!(
        plate_range.get_value((2, 1)),
        Some(&calamine::Data::String("-".to_string())),
        "B2's disabled image must not contribute to the average"
    );
}
#[test]
fn the_classes_filter_restricts_plate_well_flat_list_and_heatmap() {
    let (database, out_dir) = open(&[
        ObjectSpec::new("A1_01.tif", "ClassA", 1, 10)
            .at_centroid(10.0, 10.0)
            .with_image_size(100, 100),
        ObjectSpec::new("A1_01.tif", "ClassB", 2, 999)
            .at_centroid(10.0, 10.0)
            .with_image_size(100, 100),
    ]);
    let export = PlateExport {
        output_dir: out_dir.clone(),
        columns: vec![Column::AreaSizePx],
        aggregations: vec![Aggregation::Avg],
        object_classes: vec![ObjectClass::Global(1)],
        square_size: Some(50),
        with_plate_view_heatmap: true,
        with_well_view_heatmap: true,
        with_plate_view_list: true,
        with_well_view_list: true,
        with_heatmap: true,
        ..Default::default()
    };
    export
        .write(&local(&database), &no_cancel())
        .expect("filtered export");

    for file in ["plate.xlsx", "well.xlsx", "heatmap_A1_01.xlsx"] {
        let workbook: calamine::Xlsx<_> = calamine::open_workbook(out_dir.join(file))
            .unwrap_or_else(|e| panic!("open {file}: {e}"));
        let sheets = workbook.sheet_names();
        assert_eq!(
            sheets,
            vec!["ClassA".to_string()],
            "{file} must only have a sheet for the selected class, not ClassB: {sheets:?}"
        );
    }

    let mut plate_list_wb: calamine::Xlsx<_> =
        calamine::open_workbook(out_dir.join("plate_list.xlsx")).expect("open plate_list.xlsx");
    let plate_list_range = plate_list_wb.worksheet_range("Plate").expect("Plate sheet");
    let header = plate_list_range.get_value((0, 1)).expect("combo header");
    assert!(
        matches!(header, calamine::Data::String(s) if s.contains("ClassA") && !s.contains("ClassB")),
        "plate_list.xlsx combo header must only reference the selected class: {header:?}"
    );
}
#[test]
fn plate_and_well_documents_are_written_even_without_any_class_column_combination() {
    let (database, out_dir) = open(&[ObjectSpec::new("img1.tif", "ClassA", 1, 100)]);
    let (z_stacks, t_stacks) = full_range();
    let export = PlateExport {
        output_dir: out_dir.clone(),
        z_stacks,
        t_stacks,
        columns: vec![Column::AreaSizePx],
        // No aggregation: nothing to compute per class and column.
        aggregations: Vec::new(),
        with_plate_view_heatmap: true,
        with_well_view_heatmap: true,
        with_plate_view_list: true,
        with_well_view_list: true,
        ..Default::default()
    };

    export
        .write(&local(&database), &no_cancel())
        .expect("xlsx export without combinations");

    for name in [
        "plate.xlsx",
        "well.xlsx",
        "plate_list.xlsx",
        "well_list.xlsx",
    ] {
        assert!(out_dir.join(name).is_file(), "{name} missing");
    }
}
#[test]
fn plate_and_well_exports_do_nothing_when_none_of_their_views_is_wanted() {
    let (database, out_dir) = open(&[ObjectSpec::new("img1.tif", "ClassA", 1, 100)]);
    let export = PlateExport {
        output_dir: out_dir.clone(),
        ..Default::default()
    };

    export.write(&local(&database), &no_cancel()).unwrap();

    assert!(!out_dir.exists(), "no document written");
}
/// Plate grids of results with several z-planes: a tab per z-plane,
/// named with it.
#[test]
fn plate_grids_get_a_tab_per_z_plane() {
    let (database, out_dir) = open(&[
        ObjectSpec::new("A1_01.tif", "ClassA", 1, 10),
        ObjectSpec::new("A1_01.tif", "ClassA", 1, 30).at_plane(1, 0),
    ]);
    PlateExport {
        output_dir: out_dir.clone(),
        columns: vec![Column::AreaSizePx],
        aggregations: vec![Aggregation::Avg],
        with_plate_view_heatmap: true,
        ..Default::default()
    }
    .write(&local(&database), &no_cancel())
    .expect("plate export");
    let mut workbook: calamine::Xlsx<_> =
        calamine::open_workbook(out_dir.join("plate.xlsx")).expect("open plate.xlsx");
    assert_eq!(workbook.sheet_names(), ["Z0 ClassA", "Z1 ClassA"]);
    let z1 = workbook.worksheet_range("Z1 ClassA").unwrap();
    assert_eq!(data_f64(z1.get_value((2, 1))), 30.0, "A1 on z-plane 1");
}
