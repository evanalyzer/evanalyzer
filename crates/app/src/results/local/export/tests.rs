//! # tests
//!
//! The exports' tests: every format and table on small seeded results
//! files.
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-11-22
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.

use super::super::test_support::{ObjectSpec, TestResults, seed_db};
use super::grouped::stream_grouped_by_image_pages;
use super::*;
use crate::results::{ColorScale, ColorSchema};
use crate::results::{Grouping, PlateSize, TimeCourseExport, TimeCourseStat};
use crate::results::{PlateFilter, WellsBatchFilter};
use calamine::Reader as _;
use std::path::PathBuf;
use std::range::Range;

/// `write_f64` writes text that reads back as the same value, without
/// an exponent, no longer than what `f64` prints - mostly the very same
/// text. Small, large, negative, whole, odd values and many random ones.
#[test]
fn numbers_read_back_as_written_without_an_exponent() {
    let mut values = vec![
        0.0,
        -0.0,
        1.0,
        -2.0,
        0.1,
        1.5e-7,
        1e-5,
        9.99e-6,
        123456.789,
        1e15,
        1e16,
        1.7e308,
        f64::MIN_POSITIVE,
        5e-324,
        f64::NAN,
        f64::INFINITY,
        f64::NEG_INFINITY,
        2.0 / 3.0,
    ];
    let mut seed = 0x9E37_79B9_7F4A_7C15u64;
    for _ in 0..200_000 {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        values.push(f64::from_bits(seed));
        values.push((seed % 1_000_000) as f64 / 1000.0);
        values.push(f64::from_bits(seed).abs() % 2e14);
        values.push(f64::from_bits(seed).abs() % 1e3);
    }
    let mut same = 0;
    for value in &values {
        let mut out = Vec::new();
        write_f64(&mut out, *value);
        let text = String::from_utf8(out).unwrap();
        let printed = value.to_string();
        let back: f64 = text.parse().unwrap();
        assert!(
            back.to_bits() == value.to_bits() || value.is_nan(),
            "{value:e}: {text}"
        );
        assert!(
            !text.contains('e') && text.len() <= printed.len(),
            "{value:e}: {text}"
        );
        same += usize::from(text == printed);
    }
    assert!(
        same * 100 >= values.len() * 99,
        "{same} of {}",
        values.len()
    );
}

/// List cells: `f32` values read back exactly, like `f32` prints them
/// (but for the last of 9 digits), text quoted where needed.
#[test]
fn csv_cells_read_back_as_their_values() {
    let text = |value| {
        let mut out = Vec::new();
        write_csv_cell(&mut out, &plain_cell(value));
        String::from_utf8(out).unwrap()
    };
    let mut seed = 0x2545_F491u32;
    for _ in 0..200_000 {
        seed ^= seed << 13;
        seed ^= seed >> 17;
        seed ^= seed << 5;
        let value = f32::from_bits(seed);
        let written = text(CellValue::Float(value));
        let back: f32 = written.parse().unwrap();
        assert!(
            back.to_bits() == value.to_bits() || value.is_nan(),
            "{value:e}: {written}"
        );
        assert!(!written.contains('e') && written.len() <= value.to_string().len());
    }
    assert_eq!(text(CellValue::Float(150.0)), "150");
    assert_eq!(text(CellValue::Integer(-3)), "-3");
    assert_eq!(text(CellValue::String("a,b".into())), "\"a,b\"");
    assert_eq!(text(CellValue::Empty), "");
}

/// One field as `write_csv_field` writes it.
fn csv_escape(field: &str) -> String {
    let mut out = Vec::new();
    write_csv_field(&mut out, field).unwrap();
    String::from_utf8(out).unwrap()
}

/// Text a spreadsheet would run as a formula gets a `'`; numbers,
/// negative ones too, and text with the characters elsewhere don't.
#[test]
fn text_that_would_run_as_a_formula_is_not_a_formula() {
    assert_eq!(
        csv_escape("=HYPERLINK(\"http://x\")"),
        "\"'=HYPERLINK(\"\"http://x\"\")\""
    );
    assert_eq!(csv_escape("=1+2"), "'=1+2");
    assert_eq!(csv_escape("+cmd|' /C calc'!A0"), "'+cmd|' /C calc'!A0");
    assert_eq!(csv_escape("-2+3"), "'-2+3");
    assert_eq!(csv_escape("@SUM(A1)"), "'@SUM(A1)");
    assert_eq!(csv_escape("\t=1"), "'\t=1");
    assert_eq!(csv_escape("\r=1"), "\"'\r=1\"");
    for number in ["-0.5", "-3", "+7", "-1e-3", "-inf"] {
        assert_eq!(csv_escape(number), number);
    }
    for text in ["Nucleus", "a=b", "x-1", "", "e-mail"] {
        assert_eq!(csv_escape(text), text);
    }
}

/// Fields go out as before: empty ones (the first too) between commas,
/// whole numbers without a fraction, the others as `f64` prints them.
#[test]
fn csv_fields_are_written_like_their_text() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("fields.csv");
    write_fields_csv(
        &vec![
            vec![Field::Empty, Field::Text("a,b".into()), Field::Number(3.0)],
            vec![Field::Number(0.1), Field::Empty, Field::Number(-2.0)],
            vec![Field::Number(-0.0), Field::Number(1e300), Field::Empty],
        ],
        &path,
    )
    .unwrap();
    assert_eq!(
        std::fs::read_to_string(&path).unwrap(),
        format!(",\"a,b\",3\n0.1,,-2\n-0,{},\n", 1e300)
    );
}

#[test]
fn csv_escape_quotes_fields_containing_commas_or_quotes() {
    assert_eq!(csv_escape("plain"), "plain");
    assert_eq!(csv_escape("a,b"), "\"a,b\"");
    assert_eq!(csv_escape("a\"b"), "\"a\"\"b\"");
}

#[test]
fn csv_escape_leaves_a_plain_field_untouched() {
    assert_eq!(csv_escape("plain text"), "plain text");
    assert_eq!(csv_escape(""), "");
}

#[test]
fn csv_escape_quotes_a_field_containing_a_newline() {
    assert_eq!(csv_escape("a\nb"), "\"a\nb\"");
    assert_eq!(csv_escape("a\rb"), "\"a\rb\"");
}

fn cell_str(value: &str) -> Cell {
    Cell {
        value: CellValue::String(value.to_string()),
        bg_color: 0,
        alternating_color: false,
        search_key: None,
        disabled: false,
        any_disabled: false,
        failed: false,
        any_failed: false,
    }
}

#[test]
fn cell_format_is_just_the_font_size_for_a_plain_enabled_cell() {
    assert_eq!(
        cell_format(&cell_str("x"), None),
        Format::new().set_font_size(EXPORT_FONT_SIZE)
    );
}

#[test]
fn cell_format_grays_and_strikes_through_a_disabled_cell_with_no_other_formatting() {
    assert_eq!(
        cell_format(
            &Cell {
                disabled: true,
                ..cell_str("x")
            },
            None
        ),
        Format::new()
            .set_font_size(EXPORT_FONT_SIZE)
            .set_background_color(Color::RGB(EMPTY_CELL_BG))
            .set_font_strikethrough()
    );
}

#[test]
fn cell_format_grays_out_a_disabled_cell_instead_of_its_own_background_color() {
    // A disabled cell must not keep its heatmap/class color - otherwise
    // it still reads as "just another colored cell" at a glance, same
    // as the GUI grid's flat gray fill for a disabled tile.
    assert_eq!(
        cell_format(
            &Cell {
                bg_color: 0x112233,
                disabled: true,
                ..cell_str("x")
            },
            None
        ),
        Format::new()
            .set_font_size(EXPORT_FONT_SIZE)
            .set_background_color(Color::RGB(EMPTY_CELL_BG))
            .set_font_strikethrough()
    );
}

#[test]
fn cell_format_grays_out_a_disabled_cell_instead_of_the_alternating_row_background() {
    assert_eq!(
        cell_format(
            &Cell {
                alternating_color: true,
                disabled: true,
                ..cell_str("x")
            },
            None
        ),
        Format::new()
            .set_font_size(EXPORT_FONT_SIZE)
            .set_background_color(Color::RGB(EMPTY_CELL_BG))
            .set_font_strikethrough()
    );
}

#[test]
fn cell_format_fills_an_existing_but_empty_grid_cell() {
    assert_eq!(
        cell_format(
            &Cell {
                value: CellValue::Empty,
                search_key: Some(("A1".to_string(), "A1".to_string())),
                ..cell_str("x")
            },
            None
        ),
        Format::new()
            .set_font_size(EXPORT_FONT_SIZE)
            .set_background_color(Color::RGB(EMPTY_CELL_BG))
    );
}

#[test]
fn cell_format_leaves_a_truly_absent_grid_position_with_just_the_font_size() {
    assert_eq!(
        cell_format(
            &Cell {
                value: CellValue::Empty,
                search_key: None,
                ..cell_str("x")
            },
            None
        ),
        Format::new().set_font_size(EXPORT_FONT_SIZE),
        "no well/field sits here at all, so it must stay blank like write_blank's no-op"
    );
}

#[test]
fn cell_format_applies_a_num_format_and_centers_a_grid_cell() {
    assert_eq!(
        cell_format(&cell_str("x"), Some("0.00")),
        Format::new()
            .set_font_size(EXPORT_FONT_SIZE)
            .set_num_format("0.00")
            .set_align(FormatAlign::Center)
            .set_align(FormatAlign::VerticalCenter)
    );
}

/// `write_cell` itself (not just `cell_format`) - an existing-but-empty
/// grid cell (a real well/field with no value) must read back as the
/// text "-", not a truly blank cell (that's reserved for a grid
/// position with no well/field at all).
#[test]
fn write_cell_labels_an_existing_but_empty_grid_cell_as_a_dash() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("cell.xlsx");
    let mut workbook = Workbook::new();
    let sheet = workbook.add_worksheet();
    write_cell(
        sheet,
        0,
        0,
        &Cell {
            value: CellValue::Empty,
            search_key: Some(("A1".to_string(), "A1".to_string())),
            ..cell_str("unused")
        },
        None,
    )
    .expect("write existing-but-empty cell");
    write_cell(
        sheet,
        0,
        1,
        &Cell {
            value: CellValue::Empty,
            search_key: None,
            ..cell_str("unused")
        },
        None,
    )
    .expect("write truly absent cell");
    workbook.save(&path).expect("save workbook");

    let mut wb: calamine::Xlsx<_> = calamine::open_workbook(&path).expect("open cell.xlsx");
    let range = wb.worksheet_range_at(0).expect("sheet").expect("range");
    assert_eq!(
        range.get_value((0, 0)),
        Some(&calamine::Data::String("-".to_string())),
        "a real well/field with no value reads as \"-\""
    );
    assert!(
        matches!(range.get_value((0, 1)), None | Some(calamine::Data::Empty)),
        "a grid position with no well/field at all stays truly blank"
    );
}

#[test]
fn cell_text_reads_back_every_cell_value_variant_as_plain_text() {
    assert_eq!(cell_text(&cell_str("hi")), "hi");
    assert_eq!(
        cell_text(&Cell {
            value: CellValue::Empty,
            bg_color: 0,
            alternating_color: false,
            search_key: None,
            disabled: false,
            any_disabled: false,
            failed: false,
            any_failed: false,
        }),
        ""
    );
    assert_eq!(
        cell_text(&Cell {
            value: CellValue::Float(1.5),
            bg_color: 0,
            alternating_color: false,
            search_key: None,
            disabled: false,
            any_disabled: false,
            failed: false,
            any_failed: false,
        }),
        "1.5"
    );
    assert_eq!(
        cell_text(&Cell {
            value: CellValue::Integer(7),
            bg_color: 0,
            alternating_color: false,
            search_key: None,
            disabled: false,
            any_disabled: false,
            failed: false,
            any_failed: false,
        }),
        "7"
    );
    assert_eq!(
        cell_text(&Cell {
            value: CellValue::Class(("ClassA".to_string(), 0)),
            bg_color: 0,
            alternating_color: false,
            search_key: None,
            disabled: false,
            any_disabled: false,
            failed: false,
            any_failed: false,
        }),
        "ClassA"
    );
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

fn no_progress() -> impl FnMut(&str, usize, usize) {
    |_message: &str, _current: usize, _total: usize| {}
}

fn no_cancel() -> AtomicBool {
    AtomicBool::new(false)
}

/// Every z-plane and time point (`None`: all).
fn full_range() -> (Option<Range<u32>>, Option<Range<u32>>) {
    (None, None)
}

/// End-to-end proof that `start_export`'s CSV path (shared by the CLI
/// and the GUI export dialog) actually writes a real
/// `list.csv`/`grouped_by_image.csv` with the expected content — not
/// just that a caller's own call site happens to work.
#[test]
fn transposed_export_writes_the_classes_side_by_side() {
    let (database, out_dir) = open(&[
        ObjectSpec::new("img1.tif", "ClassA", 1, 100),
        ObjectSpec::new("img1.tif", "ClassB", 2, 200),
        ObjectSpec::new("img2.tif", "ClassA", 1, 300),
    ]);
    let (z_stacks, t_stacks) = full_range();
    ResultExport {
        output_dir: out_dir.clone(),
        format: ExportFormat::CSV,
        z_stacks,
        t_stacks,
        columns: vec![Column::AreaSizePx],
        aggregations: vec![Aggregation::Avg],
        with_list_view: true,
        with_grouped_by_image_list: true,
        transpond_table: true,
        ..Default::default()
    }
    .start_export(&database, &no_cancel(), &mut no_progress())
    .expect("transposed export");

    let list = std::fs::read_to_string(out_dir.join("list.csv")).unwrap();
    let mut lines = list.lines();
    assert_eq!(
        lines.next().unwrap(),
        "image,Area [px] (ClassA),Area [px] (ClassB)"
    );
    assert_eq!(lines.count(), 2, "one row per image here: {list}");

    let grouped = std::fs::read_to_string(out_dir.join("grouped_by_image.csv")).unwrap();
    let mut lines = grouped.lines();
    assert_eq!(
        lines.next().unwrap(),
        "image,Area [px] (AVG) (ClassA),Area [px] (AVG) (ClassB)"
    );
    // img2 has no ClassB objects: an empty field, not 0.
    assert!(
        lines.any(|line| line.starts_with("img2.tif,300") && line.ends_with(',')),
        "{grouped}"
    );
}

/// With a marker or a note on an image, the grouped table ends with
/// them (the results' own copy); the other images' fields stay empty.
#[test]
fn the_grouped_export_ends_with_the_images_marker_and_note() {
    let (database, out_dir) = open(&[
        ObjectSpec::new("img1.tif", "ClassA", 1, 100),
        ObjectSpec::new("img2.tif", "ClassA", 2, 200),
    ]);
    database
        .change_image_review(
            "img1.tif",
            &crate::results::ImageReviewChange::Note("bubble, left".into()),
        )
        .unwrap();
    database
        .change_image_review(
            "img1.tif",
            &crate::results::ImageReviewChange::Marker(Some(ImageMarker::Red)),
        )
        .unwrap();
    let (z_stacks, t_stacks) = full_range();
    ResultExport {
        output_dir: out_dir.clone(),
        format: ExportFormat::CSV,
        z_stacks,
        t_stacks,
        columns: vec![Column::AreaSizePx],
        aggregations: vec![Aggregation::Avg],
        with_grouped_by_image_list: true,
        ..Default::default()
    }
    .start_export(&database, &no_cancel(), &mut no_progress())
    .expect("export");
    let grouped = std::fs::read_to_string(out_dir.join("grouped_by_image.csv")).unwrap();
    let lines: Vec<&str> = grouped.lines().collect();
    assert!(lines[0].ends_with(",image_marker,image_note"), "{grouped}");
    for line in &lines[1..] {
        if line.starts_with("img1.tif") {
            assert!(line.ends_with(",red,\"bubble, left\""), "{grouped}");
        } else {
            assert!(
                line.starts_with("img2.tif") && line.ends_with(",,"),
                "{grouped}"
            );
        }
    }
}

#[test]
fn output_file_prefix_is_prepended_to_every_document() {
    let (database, out_dir) = open(&[
        ObjectSpec::new("img1.tif", "ClassA", 1, 100),
        ObjectSpec::new("img2.tif", "ClassB", 2, 200),
    ]);
    let columns: Vec<Column> = database
        .get_available_columns()
        .expect("available columns")
        .into_iter()
        .map(|entry| entry.key)
        .collect();
    let (z_stacks, t_stacks) = full_range();
    ResultExport {
        output_dir: out_dir.clone(),
        outputfile_prefix: Some("exp1".into()),
        format: ExportFormat::XLSX,
        z_stacks,
        t_stacks,
        columns,
        aggregations: vec![Aggregation::Avg],
        with_list_view: true,
        with_grouped_by_image_list: true,
        with_heatmap: true,
        ..Default::default()
    }
    .start_export(&database, &no_cancel(), &mut no_progress())
    .expect("prefixed export");

    let mut written: Vec<String> = std::fs::read_dir(&out_dir)
        .unwrap()
        .map(|entry| entry.unwrap().file_name().to_string_lossy().into_owned())
        .collect();
    written.sort();
    assert!(
        written.contains(&"exp1_list.xlsx".to_string()),
        "{written:?}"
    );
    assert!(
        written.contains(&"exp1_grouped_by_image.xlsx".to_string()),
        "{written:?}"
    );
    assert!(
        written.iter().any(|f| f.starts_with("exp1_heatmap_")),
        "{written:?}"
    );
    assert!(
        written.iter().all(|f| f.starts_with("exp1_")),
        "{written:?}"
    );
}

#[test]
fn output_path_sanitizes_the_prefix_and_ignores_a_blank_one() {
    let export = |prefix: Option<&str>| ResultExport {
        output_dir: PathBuf::from("/out"),
        outputfile_prefix: prefix.map(str::to_string),
        ..Default::default()
    };
    assert_eq!(
        export(None).output_path("list.csv"),
        PathBuf::from("/out/list.csv")
    );
    assert_eq!(
        export(Some("  ")).output_path("list.csv"),
        PathBuf::from("/out/list.csv")
    );
    assert_eq!(
        export(Some(" run 2 ")).output_path("list.csv"),
        PathBuf::from("/out/run_2_list.csv")
    );
    // Can't escape the output folder.
    assert_eq!(
        export(Some("../../etc/x")).output_path("list.csv"),
        PathBuf::from("/out/______etc_x_list.csv")
    );
}

#[test]
fn start_export_writes_csv_for_list_and_grouped_by_image() {
    let (database, out_dir) = open(&[
        ObjectSpec::new("img1.tif", "ClassA", 1, 100),
        ObjectSpec::new("img2.tif", "ClassB", 2, 200),
    ]);
    let columns: Vec<Column> = database
        .get_available_columns()
        .expect("available columns")
        .into_iter()
        .map(|entry| entry.key)
        .collect();
    let (z_stacks, t_stacks) = full_range();

    let list_export = ResultExport {
        output_dir: out_dir.clone(),
        format: ExportFormat::CSV,
        z_stacks,
        t_stacks,
        columns: columns.clone(),
        with_list_view: true,
        ..Default::default()
    };
    list_export
        .start_export(&database, &no_cancel(), &mut no_progress())
        .expect("csv list export");
    let list_csv = std::fs::read_to_string(out_dir.join("list.csv")).expect("read list.csv");
    let mut lines = list_csv.lines();
    assert!(lines.next().unwrap().contains("Class"));
    assert_eq!(lines.count(), 2, "expected 2 data rows");
    assert!(list_csv.contains("ClassA"));
    assert!(list_csv.contains("ClassB"));

    let grouped_export = ResultExport {
        output_dir: out_dir.clone(),
        format: ExportFormat::CSV,
        z_stacks,
        t_stacks,
        columns,
        aggregations: vec![Aggregation::Avg],
        with_grouped_by_image_list: true,
        ..Default::default()
    };
    grouped_export
        .start_export(&database, &no_cancel(), &mut no_progress())
        .expect("csv grouped export");
    let grouped_csv = std::fs::read_to_string(out_dir.join("grouped_by_image.csv"))
        .expect("read grouped_by_image.csv");
    // One row per (image, class) - including the class an image has no
    // objects of: a count of 0, and no averages.
    assert_eq!(
        grouped_csv.lines().count(),
        5,
        "expected a header plus one row per image and class: {grouped_csv}"
    );
    assert!(grouped_csv.contains("img1.tif,ClassB,0,,"), "{grouped_csv}");
}

#[test]
fn start_export_writes_a_valid_xlsx_for_list_and_grouped_by_image() {
    let (database, out_dir) = open(&[
        ObjectSpec::new("img1.tif", "ClassA", 1, 100),
        ObjectSpec::new("img2.tif", "ClassB", 2, 200),
    ]);
    let columns: Vec<Column> = database
        .get_available_columns()
        .expect("available columns")
        .into_iter()
        .map(|entry| entry.key)
        .collect();
    let (z_stacks, t_stacks) = full_range();

    let export = ResultExport {
        output_dir: out_dir.clone(),
        format: ExportFormat::XLSX,
        z_stacks,
        t_stacks,
        columns,
        aggregations: vec![Aggregation::Avg],
        with_list_view: true,
        with_grouped_by_image_list: true,
        ..Default::default()
    };
    export
        .start_export(&database, &no_cancel(), &mut no_progress())
        .expect("xlsx export");

    for name in ["list.xlsx", "grouped_by_image.xlsx"] {
        let bytes =
            std::fs::read(out_dir.join(name)).unwrap_or_else(|e| panic!("read {name}: {e}"));
        assert!(
            bytes.len() > 4,
            "{name} is too small: {} bytes",
            bytes.len()
        );
        assert_eq!(&bytes[..4], b"PK\x03\x04", "{name} is not a zip/xlsx file");
    }
}

#[test]
fn start_export_with_list_coloc_details_writes_a_second_csv_document() {
    let (database, out_dir) = open(&[ObjectSpec::new("img1.tif", "ClassA", 1, 100)]);
    let columns: Vec<Column> = database
        .get_available_columns()
        .expect("available columns")
        .into_iter()
        .map(|entry| entry.key)
        .collect();
    let (z_stacks, t_stacks) = full_range();

    let export = ResultExport {
        output_dir: out_dir.clone(),
        format: ExportFormat::CSV,
        z_stacks,
        t_stacks,
        columns,
        with_list_view: true,
        with_list_coloc_details: true,
        ..Default::default()
    };
    export
        .start_export(&database, &no_cancel(), &mut no_progress())
        .expect("csv export with coloc details");

    assert!(out_dir.join("list.csv").exists());
    assert!(out_dir.join("list_coloc_details.csv").exists());
}

#[test]
fn start_export_with_one_file_per_image_writes_a_document_per_image() {
    let (database, out_dir) = open(&[
        ObjectSpec::new("img1.tif", "ClassA", 1, 100),
        ObjectSpec::new("img2.tif", "ClassB", 2, 200),
    ]);
    let columns: Vec<Column> = database
        .get_available_columns()
        .expect("available columns")
        .into_iter()
        .map(|entry| entry.key)
        .collect();
    let (z_stacks, t_stacks) = full_range();

    let export = ResultExport {
        output_dir: out_dir.clone(),
        format: ExportFormat::CSV,
        z_stacks,
        t_stacks,
        columns,
        with_list_view: true,
        with_list_one_file_per_image: true,
        ..Default::default()
    };
    export
        .start_export(&database, &no_cancel(), &mut no_progress())
        .expect("csv per-image export");

    let img1 = std::fs::read_to_string(out_dir.join("list_img1.csv")).expect("list_img1.csv");
    let img2 = std::fs::read_to_string(out_dir.join("list_img2.csv")).expect("list_img2.csv");
    assert!(img1.contains("ClassA") && !img1.contains("ClassB"));
    assert!(img2.contains("ClassB") && !img2.contains("ClassA"));
}

#[test]
fn start_export_rejects_csv_combined_with_plate_view_heatmap() {
    let (database, out_dir) = open(&[ObjectSpec::new("img1.tif", "ClassA", 1, 100)]);
    let (z_stacks, t_stacks) = full_range();
    let export = ResultExport {
        output_dir: out_dir,
        format: ExportFormat::CSV,
        z_stacks,
        t_stacks,
        with_plate_view_heatmap: true,
        ..Default::default()
    };
    let result = export.start_export(&database, &no_cancel(), &mut no_progress());
    assert!(result.is_err());
}

#[test]
fn start_export_rejects_csv_combined_with_heatmap() {
    let (database, out_dir) = open(&[ObjectSpec::new("img1.tif", "ClassA", 1, 100)]);
    let (z_stacks, t_stacks) = full_range();
    let export = ResultExport {
        output_dir: out_dir,
        format: ExportFormat::CSV,
        z_stacks,
        t_stacks,
        with_heatmap: true,
        ..Default::default()
    };
    let result = export.start_export(&database, &no_cancel(), &mut no_progress());
    assert!(result.is_err());
}

/// The Parquet path (`export_as_parquet`) executes DuckDB's own
/// `COPY objects TO '<path>' (FORMAT parquet)` directly against the raw
/// `objects` table — this proves it actually produces a file DuckDB
/// itself can read back, with every seeded row intact, and (unlike
/// CSV/XLSX) completely ignoring `columns`/`with_list_view`/filters.
#[test]
fn start_export_parquet_dumps_the_whole_objects_table_ignoring_other_settings() {
    let (database, out_dir) = open(&[
        ObjectSpec::new("img1.tif", "ClassA", 1, 100),
        ObjectSpec::new("img2.tif", "ClassB", 2, 200),
        ObjectSpec::new("img2.tif", "ClassB", 2, 300),
    ]);
    let export = ResultExport {
        output_dir: out_dir.clone(),
        format: ExportFormat::Parquet,
        // Deliberately left at defaults / unset: `columns` is empty,
        // every `with_*` flag is false - Parquet must ignore all of it.
        ..Default::default()
    };
    export
        .start_export(&database, &no_cancel(), &mut no_progress())
        .expect("parquet export");

    let parquet_path = out_dir.join("objects.parquet");
    let bytes = std::fs::read(&parquet_path).expect("read objects.parquet");
    assert!(bytes.len() > 8);
    assert_eq!(&bytes[..4], b"PAR1", "missing leading PAR1 magic");
    assert_eq!(
        &bytes[bytes.len() - 4..],
        b"PAR1",
        "missing trailing PAR1 magic"
    );

    // Read it back through a *separate* DuckDB connection - proof the
    // file is genuinely valid Parquet, not just magic-byte-shaped.
    let verify = duckdb::Connection::open_in_memory().expect("open verify connection");
    let count: i64 = verify
        .query_row(
            &format!(
                "SELECT COUNT(*) FROM read_parquet('{}')",
                parquet_path.to_string_lossy().replace('\'', "''")
            ),
            [],
            |row| row.get(0),
        )
        .expect("query parquet row count");
    assert_eq!(count, 3, "expected every seeded object, unfiltered");
    // Written below the results file's staging folder first (SQL may
    // write nowhere else), which is gone again.
    let staging = out_dir.parent().unwrap().join("results.evadb.export");
    assert!(!staging.exists(), "{} left behind", staging.display());
}

/// `write_list_document`'s `Parquet` arm is unreachable through
/// `start_export` (which returns early for Parquet before ever calling
/// `export_list`) - this proves it's still a real, safe error rather
/// than a panic if something calls it directly.
#[test]
fn write_list_document_rejects_parquet_directly() {
    let (database, out_dir) = open(&[ObjectSpec::new("img1.tif", "ClassA", 1, 100)]);
    let export = ResultExport {
        output_dir: out_dir,
        format: ExportFormat::Parquet,
        ..Default::default()
    };
    let result = export.write_list_document(
        &database,
        &no_cancel(),
        ImageSelection::Listed(&[]),
        &None,
        "list",
    );
    assert!(result.is_err());
}

/// Same defensive-arm proof as `write_list_document_rejects_parquet_directly`,
/// for `export_grouped_by_image`'s equivalent early return.
#[test]
fn export_grouped_by_image_rejects_parquet_directly() {
    let (database, out_dir) = open(&[ObjectSpec::new("img1.tif", "ClassA", 1, 100)]);
    let export = ResultExport {
        output_dir: out_dir,
        format: ExportFormat::Parquet,
        ..Default::default()
    };
    let result = export.export_grouped_by_image(&database, &no_cancel(), &mut no_progress());
    assert!(result.is_err());
}

#[test]
fn resolve_images_defaults_to_every_non_disabled_image_sorted_by_name() {
    let (database, out_dir) = open(&[
        ObjectSpec::new("img2.tif", "ClassA", 1, 100),
        ObjectSpec::new("img1.tif", "ClassA", 1, 100),
    ]);
    let export = ResultExport {
        output_dir: out_dir,
        ..Default::default()
    };
    let images = resolve_images(&database, &export).expect("resolve images");
    assert_eq!(images, vec!["img1.tif".to_string(), "img2.tif".to_string()]);
}

#[test]
fn resolve_images_with_an_explicit_list_returns_only_those_images() {
    let (database, out_dir) = open(&[
        ObjectSpec::new("img1.tif", "ClassA", 1, 100),
        ObjectSpec::new("img2.tif", "ClassA", 1, 100),
    ]);
    let export = ResultExport {
        output_dir: out_dir,
        image_rel_paths: vec!["img2.tif".to_string()],
        ..Default::default()
    };
    let images = resolve_images(&database, &export).expect("resolve images");
    assert_eq!(images, vec!["img2.tif".to_string()]);
}

#[test]
fn image_stub_strips_directory_and_extension() {
    assert_eq!(image_stub("folder/img1.tif"), "img1");
    assert_eq!(image_stub("img2.ome.tif"), "img2.ome");
    assert_eq!(image_stub("no_extension"), "no_extension");
}

#[test]
fn sanitize_filename_component_replaces_unsafe_characters() {
    assert_eq!(sanitize_filename_component("a/b:c"), "a_b_c");
    assert_eq!(sanitize_filename_component("plain-name_1"), "plain-name_1");
}

#[test]
fn sheet_namer_deduplicates_and_truncates_long_names() {
    let mut namer = SheetNamer::new();
    assert_eq!(namer.unique("Class A"), "Class A");
    assert_eq!(namer.unique("Class A"), "Class A (2)");
    assert_eq!(namer.unique("Class A"), "Class A (3)");

    let long_name = "x".repeat(40);
    let sanitized = namer.unique(&long_name);
    assert!(sanitized.chars().count() <= 31);
}

#[test]
fn sheet_namer_rejects_excel_forbidden_characters() {
    let mut namer = SheetNamer::new();
    let name = namer.unique("a[b]:c*d?e/f\\g");
    assert!(!name.contains(['[', ']', ':', '*', '?', '/', '\\']));
}

// -- calamine smoke test: confirms the reader API used below -------------

/// "Images with objects drawn": a PNG per image, of the real file.
#[test]
fn images_with_objects_drawn_are_written_as_png() {
    let name = "multi-channel-4D-series.ome.tif";
    let (database, out_dir) = open(&[ObjectSpec::new(name, "ClassA", 1, 100)]);
    let (z_stacks, t_stacks) = full_range();
    let export = ResultExport {
        output_dir: out_dir.clone(),
        format: ExportFormat::XLSX,
        z_stacks,
        t_stacks,
        drawn_images: Some(crate::results::DrawnImages {
            image_root: PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../core/tests"),
            series: 0,
            max_side: 4096,
            colors: [(0, [1.0, 0.0, 0.0])].into(),
            ..Default::default()
        }),
        ..Default::default()
    };
    export
        .start_export(&database, &no_cancel(), &mut no_progress())
        .expect("export");
    let png = out_dir
        .join("images")
        .join("multi-channel-4D-series.ome.png");
    let picture = image::open(&png).expect("a png").to_rgb8();
    assert!(picture.width() > 0 && picture.height() > 0);
}

/// The image folder changed since the run: the images are read where
/// the project has them now - still never outside the images folder.
#[test]
fn images_with_objects_drawn_are_read_where_the_project_has_them_now() {
    let run_path = "old-run/multi-channel-4D-series.ome.tif";
    let export_with = |image: &str| {
        let (database, out_dir) = open(&[ObjectSpec::new(run_path, "ClassA", 1, 100)]);
        let (z_stacks, t_stacks) = full_range();
        let export = ResultExport {
            output_dir: out_dir.clone(),
            format: ExportFormat::XLSX,
            z_stacks,
            t_stacks,
            drawn_images: Some(crate::results::DrawnImages {
                image_root: PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../core/tests"),
                series: 0,
                max_side: 4096,
                colors: [(0, [1.0, 0.0, 0.0])].into(),
                image_paths: [(run_path.to_string(), image.to_string())].into(),
                ..Default::default()
            }),
            ..Default::default()
        };
        let result = export.start_export(&database, &no_cancel(), &mut no_progress());
        let drawn = std::fs::read_dir(out_dir.join("images")).map_or(0, |d| d.count());
        (result, drawn)
    };
    let (result, drawn) = export_with("multi-channel-4D-series.ome.tif");
    result.expect("export");
    assert_eq!(drawn, 1);
    let (result, drawn) = export_with("../tests/multi-channel-4D-series.ome.tif");
    let err = result.expect_err("refused").to_string();
    assert!(err.contains("leaves the images folder"), "{err}");
    assert_eq!(drawn, 0);
}

/// Results with several time points: a PNG per plane, named by it; the
/// chosen time range limits them.
#[test]
fn images_with_objects_drawn_get_a_png_per_time_point() {
    let name = "multi-channel-4D-series.ome.tif";
    let (database, out_dir) = open(&[
        ObjectSpec::new(name, "ClassA", 1, 100).at_plane(0, 0),
        ObjectSpec::new(name, "ClassB", 2, 100).at_plane(0, 1),
        ObjectSpec::new(name, "ClassB", 2, 100).at_plane(0, 2),
    ]);
    let (z_stacks, _) = full_range();
    let export = ResultExport {
        output_dir: out_dir.clone(),
        format: ExportFormat::XLSX,
        z_stacks,
        t_stacks: Some(std::range::Range { start: 0, end: 2 }),
        object_classes: vec![ObjectClass::Global(2)],
        drawn_images: Some(crate::results::DrawnImages {
            image_root: PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../core/tests"),
            series: 0,
            max_side: 4096,
            colors: [(0, [1.0, 0.0, 0.0])].into(),
            ..Default::default()
        }),
        ..Default::default()
    };
    export
        .start_export(&database, &no_cancel(), &mut no_progress())
        .expect("export");
    let mut written: Vec<String> = std::fs::read_dir(out_dir.join("images"))
        .expect("images")
        .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
        .collect();
    written.sort();
    assert_eq!(
        written,
        [
            "multi-channel-4D-series.ome_t0000.png",
            "multi-channel-4D-series.ome_t0001.png"
        ]
    );
}

/// The image paths come from the results file - anyone's: one that
/// leaves the images folder (`..`, absolute) isn't drawn, though the
/// file is there.
#[test]
fn images_outside_the_images_folder_are_not_drawn() {
    let core = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../core");
    for rel_path in [
        "../tests/multi-channel-4D-series.ome.tif",
        concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../core/tests/multi-channel-4D-series.ome.tif"
        ),
    ] {
        assert!(core.join("src").join(rel_path).is_file(), "{rel_path}");
        let (database, out_dir) = open(&[ObjectSpec::new(rel_path, "ClassA", 1, 100)]);
        let (z_stacks, t_stacks) = full_range();
        let export = ResultExport {
            output_dir: out_dir.clone(),
            format: ExportFormat::XLSX,
            z_stacks,
            t_stacks,
            drawn_images: Some(crate::results::DrawnImages {
                image_root: core.join("src"),
                series: 0,
                max_side: 4096,
                colors: [(0, [1.0, 0.0, 0.0])].into(),
                ..Default::default()
            }),
            ..Default::default()
        };
        let err = export
            .start_export(&database, &no_cancel(), &mut no_progress())
            .expect_err("refused")
            .to_string();
        assert!(err.contains("leaves the images folder"), "{err}");
        let drawn = std::fs::read_dir(out_dir.join("images")).map_or(0, |d| d.count());
        assert_eq!(drawn, 0, "{rel_path}");
    }
}

#[test]
fn calamine_can_read_back_a_real_exported_xlsx() {
    use calamine::{Data, Reader, Xlsx, open_workbook};

    let (database, out_dir) = open(&[ObjectSpec::new("img1.tif", "ClassA", 1, 100)]);
    let columns: Vec<Column> = database
        .get_available_columns()
        .expect("available columns")
        .into_iter()
        .map(|entry| entry.key)
        .collect();
    let (z_stacks, t_stacks) = full_range();
    let export = ResultExport {
        output_dir: out_dir.clone(),
        format: ExportFormat::XLSX,
        z_stacks,
        t_stacks,
        columns,
        with_list_view: true,
        ..Default::default()
    };
    export
        .start_export(&database, &no_cancel(), &mut no_progress())
        .expect("xlsx export");

    let mut workbook: Xlsx<_> =
        open_workbook(out_dir.join("list.xlsx")).expect("open exported xlsx");
    let range = workbook.worksheet_range("List").expect("List sheet");
    // The image first, as in the List.
    assert_eq!(
        range.get_value((0, 0)),
        Some(&Data::String("Image".to_string()))
    );
}

// -- plate/well/heatmap exports: plausibility against the same values --
// `ResultsGenerator` reports through its own query methods.
//
// Each of these re-derives the *expected* grid by calling the exact same
// `ResultsGenerator` method `export_plate_and_well`/`export_heatmap`
// calls internally, then reads the real produced XLSX file back with
// `calamine` and asserts every header/row-label/value cell matches - this
// is what actually exercises `write_grid_block`'s XLSX-writing logic
// (previously untested), and proves the exported file faithfully
// reflects the grid `ResultsGenerator` computed rather than the two
// silently drifting apart.

fn data_f64(data: Option<&calamine::Data>) -> f64 {
    match data {
        Some(calamine::Data::Float(v)) => *v,
        Some(calamine::Data::Int(v)) => *v as f64,
        other => panic!("expected a numeric cell, got {other:?}"),
    }
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

/// A time series: img1 and img2 at time point 0, img1 again at 1.
fn time_series() -> (TestResults, PathBuf) {
    open(&[
        ObjectSpec::new("img1.tif", "ClassA", 1, 100),
        ObjectSpec::new("img2.tif", "ClassA", 1, 200),
        ObjectSpec::new("img1.tif", "ClassA", 1, 300).at_plane(0, 1),
    ])
}

/// The CSV `name` of an export of `export`'s settings (into a folder of
/// its own), as rows of cells.
fn exported_csv(mut export: ResultExport, name: &str) -> Vec<Vec<String>> {
    let (database, out_dir) = time_series();
    export.output_dir = out_dir.clone();
    export
        .start_export(&database, &no_cancel(), &mut no_progress())
        .expect("export");
    std::fs::read_to_string(out_dir.join(name))
        .expect(name)
        .lines()
        .map(|line| line.split(',').map(str::to_string).collect())
        .collect()
}

/// Every time point by default, time point by time point, each row with
/// its "T" - no "Z": the results have one z-plane.
#[test]
fn a_time_series_list_has_every_time_point_with_its_t() {
    let rows = exported_csv(
        ResultExport {
            format: ExportFormat::CSV,
            columns: vec![Column::AreaSizePx],
            with_list_view: true,
            ..Default::default()
        },
        "list.csv",
    );
    assert_eq!(rows[0][0], "T");
    assert_ne!(rows[0][1], "Z");
    let t_and_area: Vec<(&str, &str)> = rows[1..]
        .iter()
        .map(|row| (row[0].as_str(), row.last().unwrap().as_str()))
        .collect();
    assert_eq!(t_and_area, [("0", "100"), ("0", "200"), ("1", "300")]);
}

/// The per-image export: a row per image at each of its time points
/// (img2 has only time point 0), time point by time point.
#[test]
fn the_grouped_export_has_a_row_per_time_point_and_image() {
    let rows = exported_csv(
        ResultExport {
            format: ExportFormat::CSV,
            columns: vec![Column::AreaSizePx],
            aggregations: vec![Aggregation::Avg],
            with_grouped_by_image_list: true,
            ..Default::default()
        },
        "grouped_by_image.csv",
    );
    assert_eq!(rows[0], ["T", "image", "class", "Area [px] (AVG)"]);
    let t_image_area: Vec<(&str, &str, &str)> = rows[1..]
        .iter()
        .map(|row| (row[0].as_str(), row[1].as_str(), row[3].as_str()))
        .collect();
    assert_eq!(
        t_image_area,
        [
            ("0", "img1.tif", "100"),
            ("0", "img2.tif", "200"),
            ("1", "img1.tif", "300")
        ]
    );
}

/// A chosen range: only its time points - still with their "T".
#[test]
fn a_time_point_range_exports_only_those() {
    let rows = exported_csv(
        ResultExport {
            format: ExportFormat::CSV,
            t_stacks: Some(Range { start: 1, end: 2 }),
            columns: vec![Column::AreaSizePx],
            with_list_view: true,
            ..Default::default()
        },
        "list.csv",
    );
    assert_eq!(rows[0][0], "T");
    assert_eq!(rows.len(), 2, "the header and time point 1's object");
    assert_eq!(
        (rows[1][0].as_str(), rows[1].last().unwrap().as_str()),
        ("1", "300")
    );
}

/// Results of one plane: no "T"/"Z" columns - exports as before.
#[test]
fn results_of_one_plane_have_no_plane_columns() {
    let (database, out_dir) = open(&[ObjectSpec::new("img1.tif", "ClassA", 1, 100)]);
    ResultExport {
        output_dir: out_dir.clone(),
        format: ExportFormat::CSV,
        columns: vec![Column::AreaSizePx],
        with_list_view: true,
        ..Default::default()
    }
    .start_export(&database, &no_cancel(), &mut no_progress())
    .unwrap();
    let csv = std::fs::read_to_string(out_dir.join("list.csv")).unwrap();
    let header = csv.lines().next().unwrap();
    assert!(
        !header.starts_with("T,") && !header.starts_with("Z,"),
        "{header}"
    );
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

/// Every analysed time point - also one where no image found anything -
/// and every image at each: Count 0 where it found nothing.
#[test]
fn the_grouped_export_has_every_analysed_time_point_and_image() {
    let (database, out_dir) = open_analysed(
        &[
            ObjectSpec::new("img1.tif", "ClassA", 1, 100),
            ObjectSpec::new("img2.tif", "ClassA", 1, 200),
            ObjectSpec::new("img1.tif", "ClassA", 1, 300).at_plane(0, 1),
        ],
        3,
    );
    ResultExport {
        output_dir: out_dir.clone(),
        format: ExportFormat::CSV,
        columns: vec![Column::Count],
        aggregations: vec![Aggregation::Sum],
        with_grouped_by_image_list: true,
        ..Default::default()
    }
    .start_export(&database, &no_cancel(), &mut no_progress())
    .expect("export");
    let csv = std::fs::read_to_string(out_dir.join("grouped_by_image.csv")).unwrap();
    let rows: Vec<Vec<&str>> = csv.lines().map(|line| line.split(',').collect()).collect();
    assert_eq!(rows[0][..3], ["T", "image", "class"]);
    let t_image_count: Vec<(&str, &str, &str)> = rows[1..]
        .iter()
        .map(|row| (row[0], row[1], row[3]))
        .collect();
    assert_eq!(
        t_image_count,
        [
            ("0", "img1.tif", "1"),
            ("0", "img2.tif", "1"),
            ("1", "img1.tif", "1"),
            ("1", "img2.tif", "0"),
            ("2", "img1.tif", "0"),
            ("2", "img2.tif", "0"),
        ]
    );
}

/// The time course exported; the results (whose temp folder holds the
/// files - kept while the test reads them) and where the files are.
fn time_course_export(format: ExportFormat, layout: TimeCourseLayout) -> (TestResults, PathBuf) {
    let (database, out_dir) = open_analysed(
        &[
            ObjectSpec::new("img1.tif", "ClassA", 1, 100),
            ObjectSpec::new("img2.tif", "ClassA", 1, 200),
            ObjectSpec::new("img1.tif", "ClassA", 1, 300).at_plane(0, 1),
        ],
        3,
    );
    ResultExport {
        output_dir: out_dir.clone(),
        format,
        columns: vec![Column::AreaSizePx],
        with_time_course: true,
        time_course: TimeCourseExport {
            layout,
            ..Default::default()
        },
        ..Default::default()
    }
    .start_export(&database, &no_cancel(), &mut no_progress())
    .expect("time course export");
    (database, out_dir)
}

/// Long: a row per time point and class line - the empty T2 too; wide:
/// per value and statistic, a row per time point, a column per line.
#[test]
fn the_time_course_exports_long_and_wide_csv() {
    let (_results, out_dir) = time_course_export(ExportFormat::CSV, TimeCourseLayout::Both);
    let long = std::fs::read_to_string(out_dir.join("time_course.csv")).unwrap();
    assert_eq!(
        long.lines().collect::<Vec<_>>(),
        [
            "T,Class,N images,N objects,Area [px] mean,Area [px] SD",
            "0,ClassA,2,2,150,70.71067811865476",
            "1,ClassA,2,1,300,",
            "2,ClassA,2,0,,",
        ]
    );
    let wide = std::fs::read_to_string(out_dir.join("time_course_Area__px__mean.csv")).unwrap();
    assert_eq!(
        wide.lines().collect::<Vec<_>>(),
        ["T,ClassA", "0,150", "1,300", "2,"]
    );
    assert!(out_dir.join("time_course_Area__px__SD.csv").exists());
}

/// XLSX: the long sheet and a sheet per value and statistic - or only
/// what the layout asks for.
#[test]
fn the_time_course_xlsx_has_a_sheet_per_table() {
    let sheets = |layout| {
        let (_results, out_dir) = time_course_export(ExportFormat::XLSX, layout);
        let workbook: calamine::Xlsx<_> =
            calamine::open_workbook(out_dir.join("time_course.xlsx")).expect("open");
        workbook.sheet_names()
    };
    assert_eq!(
        sheets(TimeCourseLayout::Both),
        ["Long", "Area (px) mean", "Area (px) SD"]
    );
    assert_eq!(sheets(TimeCourseLayout::Long), ["Long"]);
    assert_eq!(
        sheets(TimeCourseLayout::Wide),
        ["Area (px) mean", "Area (px) SD"]
    );
}

/// An XLSX sheet holds 1,048,576 rows: one more is refused with a hint
/// to export CSV - not cut off.
#[test]
fn an_xlsx_sheet_refuses_rows_beyond_its_limit() {
    let mut workbook = Workbook::new();
    let mut sheet = FieldsSheet::new(
        workbook
            .add_worksheet_with_constant_memory()
            .expect("a constant-memory sheet"),
    );
    sheet.next_row = XLSX_MAX_ROWS - 1;
    sheet.row(&[Field::Number(1.0)]).expect("the last row");
    let error = sheet.row(&[Field::Number(2.0)]).expect_err("beyond");
    assert!(error.to_string().contains("export it as CSV"), "{error}");
}

/// Tracks: a row per track - who, when, its values. How it moved only
/// with Measure Track Motion (see the next test).
#[test]
fn the_tracks_export_has_a_row_per_track() {
    let (database, out_dir) = open_analysed(&track_seven(), 2);
    ResultExport {
        output_dir: out_dir.clone(),
        format: ExportFormat::CSV,
        columns: vec![Column::AreaSizePx],
        with_track_summary: true,
        ..Default::default()
    }
    .start_export(&database, &no_cancel(), &mut no_progress())
    .expect("tracks export");
    let csv = std::fs::read_to_string(out_dir.join("tracks.csv")).unwrap();
    let lines: Vec<&str> = csv.lines().collect();
    assert_eq!(
        lines[0],
        "Image,Well,Class,Track,First T,Last T,Time points,Duration [time points],\
         Touches edge,From first T,Area [px] mean"
    );
    let row: Vec<&str> = lines[1].split(',').collect();
    assert_eq!(row[0], "img1.tif");
    assert_eq!(
        row[2..],
        ["ClassA", "7", "0", "1", "2", "1", "FALSE", "TRUE", "20"]
    );
    assert_eq!(lines.len(), 2, "the untracked object isn't a track");
}

fn track_seven() -> Vec<ObjectSpec> {
    vec![
        ObjectSpec::new("img1.tif", "ClassA", 1, 10)
            .in_track(7)
            .at_centroid(0.0, 0.0),
        ObjectSpec::new("img1.tif", "ClassA", 1, 30)
            .in_track(7)
            .at_plane(0, 1)
            .at_centroid(3.0, 4.0),
        ObjectSpec::new("img1.tif", "ClassA", 1, 99),
    ]
}

/// With Measure Track Motion: its values after the duration - lengths
/// in µm with the fixture's pixel size (1 nm), the rest per time point
/// without a frame interval, whatever unit was asked for; a value it
/// couldn't have is empty.
#[test]
fn the_tracks_export_has_the_stored_motion() {
    let (database, out_dir) = open_analysed(&track_seven(), 2);
    database
        .connection()
        .execute_batch(
            "CREATE TABLE measurements (image_name VARCHAR, image_rel_path VARCHAR, \
             z_stack INTEGER, t_stack INTEGER, region VARCHAR, object_id UUID, \
             related_object_id UUID, object_class_name VARCHAR, object_class_id INTEGER[], \
             step VARCHAR, measurement VARCHAR, unit VARCHAR, key VARCHAR, value DOUBLE, \
             position DOUBLE, track_id UBIGINT);
             INSERT INTO measurements VALUES
                 ('img1.tif', 'img1.tif', 0, 0, 'track', NULL, NULL, NULL, [1], 'motion_01', \
                  'Path length', 'Pixels', 'motion_01/path_length', 5.0, NULL, 7),
                 ('img1.tif', 'img1.tif', 0, 0, 'track', NULL, NULL, NULL, [1], 'motion_01', \
                  'VCL', 'PixelsPerFrame', 'motion_01/vcl', 5.0, NULL, 7),
                 ('img1.tif', 'img1.tif', 0, 0, 'track', NULL, NULL, NULL, [1], 'motion_01', \
                  'LIN', 'Fraction', 'motion_01/lin', 1.0, NULL, 7);",
        )
        .unwrap();
    ResultExport {
        output_dir: out_dir.clone(),
        format: ExportFormat::CSV,
        with_track_summary: true,
        ..Default::default()
    }
    .start_export(&database, &no_cancel(), &mut no_progress())
    .expect("tracks export");
    let csv = std::fs::read_to_string(out_dir.join("tracks.csv")).unwrap();
    let lines: Vec<&str> = csv.lines().collect();
    assert_eq!(
        lines[0],
        "Image,Well,Class,Track,First T,Last T,Time points,Duration [time points],\
         Path length [µm],Distance moved [µm],Max distance from start [µm],\
         VCL (mean speed) [px/time point],VSL [px/time point],VAP [px/time point],\
         LIN (straightness),STR,WOB,ALH [µm],BCF [1/time point],Turning angle mean [°],\
         Turning angle mean absolute [°],Angular speed [°/time point],MSD alpha,\
         MSD D [px²/time point],Touches edge,From first T"
    );
    let row: Vec<&str> = lines[1].split(',').collect();
    assert_eq!(row[8..12], ["0.005", "", "", "5"]);
    assert_eq!(row[14], "1");
}

/// With divisions: after "Track" its family - parents, children (ids
/// as "1;2"), the tree's root, the generation, how it starts and
/// ends. The parents are the tree's edges: a script can draw it.
#[test]
fn the_tracks_export_has_the_family_tree() {
    let at = |track: u64, t: i32| {
        ObjectSpec::new("img1.tif", "ClassA", 1, 10)
            .in_track(track)
            .at_plane(0, t)
    };
    let (database, out_dir) = open_analysed(&[at(1, 0), at(2, 1), at(3, 1)], 2);
    database
        .connection()
        .execute_batch(
            "ALTER TABLE objects ADD COLUMN parent_track_ids UBIGINT[];
             UPDATE objects SET parent_track_ids = [1] WHERE track_id IN (2, 3);",
        )
        .unwrap();
    ResultExport {
        output_dir: out_dir.clone(),
        format: ExportFormat::CSV,
        with_track_summary: true,
        ..Default::default()
    }
    .start_export(&database, &no_cancel(), &mut no_progress())
    .expect("tracks export");
    let csv = std::fs::read_to_string(out_dir.join("tracks.csv")).unwrap();
    let lines: Vec<&str> = csv.lines().collect();
    assert!(
        lines[0].starts_with(
            "Image,Well,Class,Track,Parent tracks,Child tracks,Lineage,Generation,\
             Starts with,Ends with,First T,"
        ),
        "{}",
        lines[0]
    );
    let family = |line: &str| {
        line.split(',')
            .skip(3)
            .take(7)
            .collect::<Vec<_>>()
            .join(",")
    };
    assert_eq!(family(lines[1]), "1,,2;3,1,0,,Division");
    assert_eq!(family(lines[2]), "2,1,,1,1,Division,");
    assert_eq!(family(lines[3]), "3,1,,1,1,Division,");
}

/// A wide table per track would have a column per track: refused with
/// a hint to export it long.
#[test]
fn a_wide_time_course_per_track_is_refused() {
    let (database, out_dir) = open_analysed(&[ObjectSpec::new("img1.tif", "ClassA", 1, 100)], 1);
    let export = |layout| {
        ResultExport {
            output_dir: out_dir.clone(),
            format: ExportFormat::CSV,
            columns: vec![Column::AreaSizePx],
            with_time_course: true,
            time_course: TimeCourseExport {
                lines: TimeCourseLines::Track,
                layout,
                ..Default::default()
            },
            ..Default::default()
        }
        .start_export(&database, &no_cancel(), &mut no_progress())
    };
    let error = export(TimeCourseLayout::Both).expect_err("wide per track");
    assert!(error.to_string().contains("export it long"), "{error}");
    export(TimeCourseLayout::Long).expect("long per track");
    assert!(out_dir.join("time_course.csv").exists());
}

/// A plate grid for every analysed time point - the one where nothing
/// was found too.
#[test]
fn plate_grids_get_a_tab_for_an_empty_time_point_too() {
    let (database, out_dir) = open_analysed(&[ObjectSpec::new("A1_01.tif", "ClassA", 1, 10)], 2);
    ResultExport {
        output_dir: out_dir.clone(),
        format: ExportFormat::XLSX,
        columns: vec![Column::AreaSizePx],
        aggregations: vec![Aggregation::Avg],
        with_plate_view_heatmap: true,
        ..Default::default()
    }
    .start_export(&database, &no_cancel(), &mut no_progress())
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
    ResultExport {
        output_dir: out_dir.clone(),
        format: ExportFormat::XLSX,
        columns: vec![Column::AreaSizePx],
        aggregations: vec![Aggregation::Avg],
        with_plate_view_heatmap: true,
        ..Default::default()
    }
    .start_export(&database, &no_cancel(), &mut no_progress())
    .expect("plate export");
    let mut workbook: calamine::Xlsx<_> =
        calamine::open_workbook(out_dir.join("plate.xlsx")).expect("open plate.xlsx");
    assert_eq!(workbook.sheet_names(), ["T0 ClassA", "T1 ClassA"]);
    let t1 = workbook.worksheet_range("T1 ClassA").unwrap();
    assert_eq!(data_f64(t1.get_value((2, 1))), 30.0, "A1 at time point 1");
}

#[test]
fn start_export_xlsx_plate_grid_matches_get_group_by_plate_heatmap() {
    let (database, out_dir) = open(&[
        ObjectSpec::new("A1_01.tif", "ClassA", 1, 10),
        ObjectSpec::new("A1_02.tif", "ClassA", 1, 20),
        ObjectSpec::new("B2_01.tif", "ClassA", 1, 100),
    ]);
    let export = ResultExport {
        output_dir: out_dir.clone(),
        format: ExportFormat::XLSX,
        columns: vec![Column::AreaSizePx],
        aggregations: vec![Aggregation::Avg],
        with_plate_view_heatmap: true,
        ..Default::default()
    };
    export
        .start_export(&database, &no_cancel(), &mut no_progress())
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
fn start_export_xlsx_well_grid_matches_get_wells_for_plate_heatmap() {
    let (database, out_dir) = open(&[
        ObjectSpec::new("A1_01.tif", "ClassA", 1, 10),
        ObjectSpec::new("A1_02.tif", "ClassA", 1, 20),
    ]);
    let export = ResultExport {
        output_dir: out_dir.clone(),
        format: ExportFormat::XLSX,
        columns: vec![Column::AreaSizePx],
        aggregations: vec![Aggregation::Avg],
        with_well_view_heatmap: true,
        ..Default::default()
    };
    export
        .start_export(&database, &no_cancel(), &mut no_progress())
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
fn start_export_xlsx_heatmap_grid_matches_get_image_heatmap() {
    let (database, out_dir) = open(&[
        ObjectSpec::new("img1.tif", "ClassA", 1, 10)
            .at_centroid(10.0, 10.0)
            .with_image_size(100, 100),
        ObjectSpec::new("img1.tif", "ClassA", 1, 20)
            .at_centroid(60.0, 10.0)
            .with_image_size(100, 100),
    ]);
    let export = ResultExport {
        output_dir: out_dir.clone(),
        format: ExportFormat::XLSX,
        columns: vec![Column::AreaSizePx],
        aggregations: vec![Aggregation::Avg],
        square_size: Some(50),
        with_heatmap: true,
        ..Default::default()
    };
    export
        .start_export(&database, &no_cancel(), &mut no_progress())
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
fn start_export_xlsx_plate_list_and_well_list_report_the_seeded_average() {
    let (database, out_dir) = open(&[
        ObjectSpec::new("A1_01.tif", "ClassA", 1, 10),
        ObjectSpec::new("A1_02.tif", "ClassA", 1, 20),
    ]);
    let export = ResultExport {
        output_dir: out_dir.clone(),
        format: ExportFormat::XLSX,
        columns: vec![Column::AreaSizePx],
        aggregations: vec![Aggregation::Avg],
        with_plate_view_list: true,
        with_well_view_list: true,
        ..Default::default()
    };
    export
        .start_export(&database, &no_cancel(), &mut no_progress())
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
fn start_export_includes_intensity_columns_in_grouped_and_plate_documents() {
    use super::super::test_support::CH0_INTENSITIES_JSON;
    let (database, out_dir) = open(&[
        ObjectSpec::new("A1_01.tif", "ClassA", 1, 10).with_intensities(CH0_INTENSITIES_JSON),
        ObjectSpec::new("A1_02.tif", "ClassA", 1, 20).with_intensities(CH0_INTENSITIES_JSON),
    ]);
    let export = ResultExport {
        output_dir: out_dir.clone(),
        format: ExportFormat::XLSX,
        columns: vec![Column::AreaSizePx, Column::IntensityAvg(0)],
        aggregations: vec![Aggregation::Sum],
        with_grouped_by_image_list: true,
        with_plate_view_list: true,
        with_plate_view_heatmap: true,
        ..Default::default()
    };
    export
        .start_export(&database, &no_cancel(), &mut no_progress())
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

    // Grouped by image: the intensity column per image.
    let mut grouped_wb: calamine::Xlsx<_> =
        calamine::open_workbook(out_dir.join("grouped_by_image.xlsx"))
            .expect("open grouped_by_image.xlsx");
    let sheet = grouped_wb.sheet_names()[0].clone();
    let grouped_range = grouped_wb.worksheet_range(&sheet).expect("grouped sheet");
    let header: Vec<String> = (0..grouped_range.width() as u32)
        .filter_map(|c| match grouped_range.get_value((0, c)) {
            Some(calamine::Data::String(s)) => Some(s.clone()),
            _ => None,
        })
        .collect();
    let intensity_col = header
        .iter()
        .position(|h| h.contains("Avg Intensity (Ch 0)"))
        .unwrap_or_else(|| panic!("no intensity column grouped by image: {header:?}"));
    assert_eq!(
        data_f64(grouped_range.get_value((1, intensity_col as u32))),
        127.0
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
fn start_export_xlsx_plate_list_writes_a_dash_for_a_well_with_no_objects() {
    let (database, out_dir) = open_with_empty_image(
        &[ObjectSpec::new("A1_01.tif", "ClassA", 1, 10)],
        "B2_01.tif",
    );
    let export = ResultExport {
        output_dir: out_dir.clone(),
        format: ExportFormat::XLSX,
        columns: vec![Column::AreaSizePx],
        aggregations: vec![Aggregation::Avg],
        with_plate_view_list: true,
        ..Default::default()
    };
    export
        .start_export(&database, &no_cancel(), &mut no_progress())
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
fn start_export_xlsx_plate_list_still_includes_a_fully_disabled_well() {
    let (database, out_dir) = open(&[
        ObjectSpec::new("A1_01.tif", "ClassA", 1, 10),
        ObjectSpec::new("B2_01.tif", "ClassA", 1, 100),
    ]);
    database
        .enable_image("B2_01.tif", true)
        .expect("disable B2");
    let export = ResultExport {
        output_dir: out_dir.clone(),
        format: ExportFormat::XLSX,
        columns: vec![Column::AreaSizePx],
        aggregations: vec![Aggregation::Avg],
        with_plate_view_list: true,
        ..Default::default()
    };
    export
        .start_export(&database, &no_cancel(), &mut no_progress())
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
fn start_export_runs_every_xlsx_document_type_together() {
    let (database, out_dir) = open(&[
        ObjectSpec::new("A1_01.tif", "ClassA", 1, 10)
            .at_centroid(10.0, 10.0)
            .with_image_size(100, 100),
        ObjectSpec::new("B2_01.tif", "ClassA", 1, 20)
            .at_centroid(10.0, 10.0)
            .with_image_size(100, 100),
    ]);
    let columns: Vec<Column> = database
        .get_available_columns()
        .expect("available columns")
        .into_iter()
        .map(|entry| entry.key)
        .collect();
    let (z_stacks, t_stacks) = full_range();
    let export = ResultExport {
        output_dir: out_dir.clone(),
        format: ExportFormat::XLSX,
        z_stacks,
        t_stacks,
        columns,
        aggregations: vec![Aggregation::Avg],
        with_list_view: true,
        with_grouped_by_image_list: true,
        with_plate_view_heatmap: true,
        with_well_view_heatmap: true,
        with_plate_view_list: true,
        with_well_view_list: true,
        with_heatmap: true,
        ..Default::default()
    };
    export
        .start_export(&database, &no_cancel(), &mut no_progress())
        .expect("full xlsx export");

    for name in [
        "list.xlsx",
        "grouped_by_image.xlsx",
        "plate.xlsx",
        "well.xlsx",
        "plate_list.xlsx",
        "well_list.xlsx",
        "heatmap_A1_01.xlsx",
        "heatmap_B2_01.xlsx",
    ] {
        let bytes =
            std::fs::read(out_dir.join(name)).unwrap_or_else(|e| panic!("read {name}: {e}"));
        assert_eq!(&bytes[..4], b"PK\x03\x04", "{name} is not a zip/xlsx file");
    }
}

#[test]
fn start_export_object_classes_filter_restricts_plate_well_flat_list_and_heatmap() {
    let (database, out_dir) = open(&[
        ObjectSpec::new("A1_01.tif", "ClassA", 1, 10)
            .at_centroid(10.0, 10.0)
            .with_image_size(100, 100),
        ObjectSpec::new("A1_01.tif", "ClassB", 2, 999)
            .at_centroid(10.0, 10.0)
            .with_image_size(100, 100),
    ]);
    let export = ResultExport {
        output_dir: out_dir.clone(),
        format: ExportFormat::XLSX,
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
        .start_export(&database, &no_cancel(), &mut no_progress())
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
fn start_export_with_list_coloc_details_xlsx_writes_a_second_sheet_with_shaded_rows() {
    let objects = vec![
        ObjectSpec::new("img1.tif", "ClassA", 1, 10)
            .with_coloc(r#"{"2":["00000000-0000-0000-0000-000000000001"]}"#),
        ObjectSpec::new("img1.tif", "ClassB", 2, 99),
    ];
    let (database, out_dir) = open(&objects);
    let (z_stacks, t_stacks) = full_range();
    let export = ResultExport {
        output_dir: out_dir.clone(),
        format: ExportFormat::XLSX,
        z_stacks,
        t_stacks,
        columns: vec![
            Column::ObjectClass,
            Column::AreaSizePx,
            Column::ColocCount(ObjectClass::Global(2)),
        ],
        with_list_view: true,
        with_list_coloc_details: true,
        ..Default::default()
    };
    export
        .start_export(&database, &no_cancel(), &mut no_progress())
        .expect("xlsx export with coloc details");

    let mut workbook: calamine::Xlsx<_> =
        calamine::open_workbook(out_dir.join("list.xlsx")).expect("open list.xlsx");
    let sheets = workbook.sheet_names();
    assert!(sheets.contains(&"List".to_string()));
    assert!(sheets.contains(&"List (Coloc Details)".to_string()));
    let coloc_range = workbook
        .worksheet_range("List (Coloc Details)")
        .expect("coloc details sheet");
    // 2 fanned-out rows (one per source object) plus the header.
    assert_eq!(coloc_range.rows().count(), 3);
}

#[test]
fn start_export_reports_an_error_when_the_output_directory_cannot_be_created() {
    let (database, _out_dir) = open(&[ObjectSpec::new("img1.tif", "ClassA", 1, 100)]);
    let dir = tempfile::tempdir().expect("tempdir");
    let blocking_file = dir.path().join("blocker");
    std::fs::write(&blocking_file, b"not a directory").expect("write blocking file");
    let export = ResultExport {
        // A regular file can't have a subdirectory created under it.
        output_dir: blocking_file.join("sub"),
        format: ExportFormat::CSV,
        with_list_view: true,
        ..Default::default()
    };
    let result = export.start_export(&database, &no_cancel(), &mut no_progress());
    assert!(result.is_err());
}

#[test]
fn start_export_xlsx_save_failure_surfaces_as_an_internal_error() {
    let (database, out_dir) = open(&[ObjectSpec::new("img1.tif", "ClassA", 1, 100)]);
    // Pre-create a *directory* at the exact path `list.xlsx` would be
    // saved to, so `Workbook::save` fails instead of succeeding.
    std::fs::create_dir_all(out_dir.join("list.xlsx")).expect("pre-create blocking directory");
    let columns: Vec<Column> = database
        .get_available_columns()
        .expect("available columns")
        .into_iter()
        .map(|entry| entry.key)
        .collect();
    let export = ResultExport {
        output_dir: out_dir,
        format: ExportFormat::XLSX,
        columns,
        with_list_view: true,
        ..Default::default()
    };
    let result = export.start_export(&database, &no_cancel(), &mut no_progress());
    assert!(result.is_err());
}

#[test]
fn start_export_grouped_by_image_with_a_class_without_objects_gives_each_image_an_empty_row() {
    let (database, out_dir) = open(&[ObjectSpec::new("img1.tif", "ClassA", 1, 100)]);
    let export = ResultExport {
        output_dir: out_dir.clone(),
        format: ExportFormat::CSV,
        columns: vec![Column::AreaSizePx],
        aggregations: vec![Aggregation::Avg],
        // No class 99 object was ever seeded - a selection matching
        // zero objects (as opposed to an explicitly-empty `Some(vec![])`
        // selection, which short-circuits earlier). The analysed image
        // still gets its row: "none found" is a result, and there's no
        // average to show.
        object_classes: vec![ObjectClass::Global(99)],
        with_grouped_by_image_list: true,
        ..Default::default()
    };
    export
        .start_export(&database, &no_cancel(), &mut no_progress())
        .expect("export with a non-matching filter should still succeed");
    let content = std::fs::read_to_string(out_dir.join("grouped_by_image.csv")).expect("read csv");
    let lines: Vec<&str> = content.lines().collect();
    assert_eq!(lines.len(), 2, "header plus the image's row: {content}");
    assert!(lines[1].starts_with("img1.tif,"), "{content}");
    assert!(
        lines[1].ends_with(','),
        "no average without objects: {content}"
    );
}

/// The image files `export` writes into `file` (CSV) name.
fn exported_images(export: &ResultExport, database: &ResultsGenerator, file: &str) -> String {
    export
        .start_export(database, &no_cancel(), &mut no_progress())
        .expect("export");
    std::fs::read_to_string(export.output_dir.join(file)).expect("read csv")
}

#[test]
fn an_export_of_all_images_leaves_out_a_disabled_image() {
    let (database, out_dir) = open(&[
        ObjectSpec::new("img1.tif", "ClassA", 1, 100),
        ObjectSpec::new("img2.tif", "ClassA", 1, 100),
    ]);
    database.enable_image("img2.tif", true).unwrap();
    for (list, file) in [(true, "list.csv"), (false, "grouped_by_image.csv")] {
        let export = ResultExport {
            output_dir: out_dir.clone(),
            format: ExportFormat::CSV,
            columns: vec![Column::ImageName, Column::AreaSizePx],
            aggregations: vec![Aggregation::Avg],
            with_list_view: list,
            with_grouped_by_image_list: !list,
            ..Default::default()
        };

        let content = exported_images(&export, &database, file);

        assert!(content.contains("img1.tif"), "{file}: {content}");
        assert!(!content.contains("img2.tif"), "{file}: {content}");
    }
}

#[test]
fn a_grouped_export_in_blocks_of_one_image_has_the_rows_of_one_block() {
    let (database, _) = open(&[
        ObjectSpec::new("img1.tif", "ClassA", 1, 100),
        ObjectSpec::new("img1.tif", "ClassA", 1, 40),
        ObjectSpec::new("img2.tif", "ClassA", 1, 50),
        ObjectSpec::new("img2.tif", "ClassB", 2, 70),
        ObjectSpec::new("img3.tif", "ClassB", 2, 20),
    ]);
    database.enable_image("img2.tif", true).unwrap();
    let filter = GroupedByImageFilter {
        plane: PlaneFilter {
            z_stack: 0,
            t_stack: 0,
        },
        images: crate::results::ImageSet::All,
        object_classes: None,
        columns: vec![Column::Count, Column::AreaSizePx],
        aggregation: vec![Aggregation::Avg],
        page: Pagination {
            limit: 0,
            after: None,
        },
        transpond_table: false,
        grouping: None,
        empty_classes: true,
        sort: None,
    };
    let rows = |rows_per_page, transpond_table| {
        let filter = GroupedByImageFilter {
            transpond_table,
            ..filter.clone()
        };
        let mut texts = Vec::new();
        let mut pages = 0;
        stream_grouped_by_image_pages(&database, &no_cancel(), &filter, rows_per_page, |page| {
            pages += 1;
            texts.extend(
                page.rows
                    .iter()
                    .map(|row| row.iter().map(cell_text).collect::<Vec<_>>()),
            );
            Ok(())
        })
        .unwrap();
        (pages, texts)
    };

    for transposed in [false, true] {
        let (one_page, all) = rows(usize::MAX, transposed);
        let (pages, blocks) = rows(1, transposed);

        assert_eq!(one_page, 1);
        assert_eq!(pages, 2, "img1 and img3 - img2 is disabled");
        assert_eq!(blocks, all, "transposed: {transposed}");
        assert!(all.iter().any(|row| row[0] == "img3.tif"), "{all:?}");
        assert!(!all.iter().any(|row| row[0] == "img2.tif"), "{all:?}");
    }
}

#[test]
fn a_disabled_image_picked_for_the_list_export_is_exported() {
    let (database, out_dir) = open(&[
        ObjectSpec::new("img1.tif", "ClassA", 1, 100),
        ObjectSpec::new("img2.tif", "ClassA", 1, 100),
    ]);
    database.enable_image("img2.tif", true).unwrap();
    let export = ResultExport {
        output_dir: out_dir,
        format: ExportFormat::CSV,
        columns: vec![Column::ImageName, Column::AreaSizePx],
        image_rel_paths: vec!["img2.tif".to_string()],
        with_list_view: true,
        ..Default::default()
    };

    let content = exported_images(&export, &database, "list.csv");

    assert!(content.contains("img2.tif"), "{content}");
    assert!(!content.contains("img1.tif"), "{content}");
}

#[test]
fn aggregation_label_covers_every_variant() {
    assert_eq!(aggregation_label(&Aggregation::Avg), "Average");
    assert_eq!(aggregation_label(&Aggregation::Min), "Minimum");
    assert_eq!(aggregation_label(&Aggregation::Max), "Maximum");
    assert_eq!(aggregation_label(&Aggregation::Stddev), "Std Dev");
    assert_eq!(aggregation_label(&Aggregation::Sum), "Sum");
    assert_eq!(aggregation_label(&Aggregation::Median), "Median");
    assert_eq!(aggregation_label(&Aggregation::Skewness), "Skewness");
}

#[test]
fn cell_to_f64_reads_both_numeric_cell_variants_and_rejects_others() {
    let float_cell = Cell {
        value: CellValue::Float(1.5),
        bg_color: 0,
        alternating_color: false,
        search_key: None,
        disabled: false,
        any_disabled: false,
        failed: false,
        any_failed: false,
    };
    let int_cell = Cell {
        value: CellValue::Integer(7),
        bg_color: 0,
        alternating_color: false,
        search_key: None,
        disabled: false,
        any_disabled: false,
        failed: false,
        any_failed: false,
    };
    let string_cell = Cell {
        value: CellValue::String("x".to_string()),
        bg_color: 0,
        alternating_color: false,
        search_key: None,
        disabled: false,
        any_disabled: false,
        failed: false,
        any_failed: false,
    };
    assert_eq!(cell_to_f64(&float_cell), Some(1.5));
    assert_eq!(cell_to_f64(&int_cell), Some(7.0));
    assert_eq!(cell_to_f64(&string_cell), None);
}

#[test]
fn sheet_namer_falls_back_to_a_generic_name_for_a_blank_class_name() {
    let mut namer = SheetNamer::new();
    assert_eq!(namer.unique("   "), "Sheet");
}

// -- cancellation -----------------------------------------------------

#[test]
fn start_export_returns_cancelled_immediately_when_already_cancelled() {
    let (database, out_dir) = open(&[ObjectSpec::new("img1.tif", "ClassA", 1, 100)]);
    let (z_stacks, t_stacks) = full_range();
    let export = ResultExport {
        output_dir: out_dir.clone(),
        format: ExportFormat::CSV,
        z_stacks,
        t_stacks,
        with_list_view: true,
        ..Default::default()
    };
    let cancel = AtomicBool::new(true);
    let result = export.start_export(&database, &cancel, &mut no_progress());
    assert!(matches!(result, Err(InternalErrors::Cancelled)));
    // Nothing should have been written - cancellation is checked before
    // even creating the output directory.
    assert!(!out_dir.join("list.csv").exists());
}

#[test]
fn start_export_stops_partway_through_a_one_file_per_image_export_once_cancelled() {
    let (database, out_dir) = open(&[
        ObjectSpec::new("img1.tif", "ClassA", 1, 10),
        ObjectSpec::new("img2.tif", "ClassA", 1, 20),
        ObjectSpec::new("img3.tif", "ClassA", 1, 30),
    ]);
    let (z_stacks, t_stacks) = full_range();
    let export = ResultExport {
        output_dir: out_dir.clone(),
        format: ExportFormat::CSV,
        z_stacks,
        t_stacks,
        columns: vec![Column::AreaSizePx],
        with_list_view: true,
        with_list_one_file_per_image: true,
        ..Default::default()
    };
    let cancel = AtomicBool::new(false);
    // Cancel as soon as the very first image's document has been
    // written, from inside the progress callback - proves cancellation
    // is actually observed *during* the export, not just checked once
    // up front.
    let mut seen_images = 0usize;
    let result = export.start_export(&database, &cancel, &mut |_message, _current, _total| {
        seen_images += 1;
        cancel.store(true, Ordering::Relaxed);
    });
    assert!(matches!(result, Err(InternalErrors::Cancelled)));
    assert_eq!(
        seen_images, 1,
        "should stop right after the first image's progress callback"
    );
    assert!(
        out_dir.join("list_img1.csv").exists(),
        "the first image's file was already written"
    );
    assert!(
        !out_dir.join("list_img2.csv").exists(),
        "must not start the second image"
    );
}

#[test]
fn cell_format_marks_a_failed_images_cell() {
    let cell = |failed, any_failed| Cell {
        value: CellValue::Float(1.0),
        bg_color: 0,
        alternating_color: false,
        search_key: None,
        disabled: false,
        any_disabled: false,
        failed,
        any_failed,
    };
    let plain = cell_format(&cell(false, false), None);

    assert_ne!(cell_format(&cell(true, false), None), plain);
    assert_ne!(cell_format(&cell(false, true), None), plain);
    assert_eq!(cell_format(&cell(false, false), None), plain);
}

#[test]
fn plate_and_well_documents_are_written_even_without_any_class_column_combination() {
    let (database, out_dir) = open(&[ObjectSpec::new("img1.tif", "ClassA", 1, 100)]);
    let (z_stacks, t_stacks) = full_range();
    let export = ResultExport {
        output_dir: out_dir.clone(),
        format: ExportFormat::XLSX,
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
        .start_export(&database, &no_cancel(), &mut no_progress())
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
    let export = ResultExport {
        output_dir: out_dir.clone(),
        format: ExportFormat::XLSX,
        ..Default::default()
    };

    export
        .export_plate_and_well(&database, &no_cancel(), &mut no_progress())
        .unwrap();
    export
        .export_plate_and_well_as_flat_list(&database, &no_cancel(), &mut no_progress())
        .unwrap();

    assert!(!out_dir.exists(), "no document written");
}

#[test]
fn listing_no_images_never_queries_a_page() {
    let (database, _) = open(&[ObjectSpec::new("img1.tif", "ClassA", 1, 100)]);
    let (z_stacks, t_stacks) = full_range();
    let export = ResultExport {
        z_stacks,
        t_stacks,
        columns: vec![Column::AreaSizePx],
        ..Default::default()
    };
    let mut pages = 0;

    stream_list_pages(
        &database,
        &no_cancel(),
        &export,
        ImageSelection::Listed(&[]),
        &None,
        false,
        &ExportPlanes::of(&database, &export).unwrap(),
        |_, _, _| {
            pages += 1;
            Ok(())
        },
    )
    .unwrap();

    assert_eq!(pages, 0);
}

#[test]
fn a_csv_document_that_cannot_be_created_is_an_error_naming_it() {
    let dir = tempfile::tempdir().unwrap();
    let file = dir.path().join("a-file");
    std::fs::write(&file, b"").unwrap();

    // Its folder would have to be inside a file.
    let Err(e) = create_csv_writer(&file.join("list.csv")) else {
        panic!("expected an error");
    };
    assert!(e.to_string().contains("could not create"), "{e}");

    // The document's own path is a folder.
    let Err(e) = create_csv_writer(dir.path()) else {
        panic!("expected an error");
    };
    assert!(e.to_string().contains("could not create"), "{e}");
}

// -- measurements -----------------------------------------------------

/// `img1.tif`'s two time points with an image measurement each, and one
/// object's value; `img2.tif` with a value too.
fn with_measurements() -> (TestResults, PathBuf, String) {
    let (generator, out_dir) = open(&[
        ObjectSpec::new("img1.tif", "ClassA", 1, 10),
        ObjectSpec::new("img2.tif", "ClassA", 1, 20),
    ]);
    let object: String = generator
        .connection()
        .query_row(
            "SELECT object_id::VARCHAR FROM objects WHERE area_px = 10",
            [],
            |row| row.get(0),
        )
        .unwrap();
    generator
        .connection()
        .execute_batch(&format!(
            "CREATE TABLE measurement_info (key VARCHAR PRIMARY KEY, measurement VARCHAR, \
             step VARCHAR, unit VARCHAR, target VARCHAR, range_min DOUBLE, \
             range_max DOUBLE, aggregatable BOOLEAN, object_index INTEGER);
             CREATE TABLE measurements (image_name VARCHAR, image_rel_path VARCHAR, \
             z_stack INTEGER, t_stack INTEGER, region VARCHAR, object_id UUID, \
             related_object_id UUID, object_class_name VARCHAR, object_class_id INTEGER[], \
             step VARCHAR, measurement VARCHAR, unit VARCHAR, key VARCHAR, value DOUBLE);
             INSERT INTO measurement_info VALUES
                 ('p_01/p', 'Pearson', 'p_01', 'None', 'Image', -1, 1, true, NULL),
                 ('q_01/q', 'Pearson', 'q_01', 'None', 'Object', -1, 1, true, 1);
             INSERT INTO measurements VALUES
                 ('img1.tif', 'img1.tif', 0, 1, 'image', NULL, NULL, NULL, NULL, 'p_01', 'Pearson', 'None', 'p_01/p', NULL),
                 ('img1.tif', 'img1.tif', 0, 0, 'image', NULL, NULL, NULL, NULL, 'p_01', 'Pearson', 'None', 'p_01/p', 0.5),
                 ('img1.tif', 'img1.tif', 0, 0, 'object', '{object}', '{object}', '[\"ClassA\"]', [1], 'q_01', 'Pearson', 'None', 'q_01/q', 0.25),
                 ('img2.tif', 'img2.tif', 0, 0, 'image', NULL, NULL, NULL, NULL, 'p_01', 'Pearson', 'None', 'p_01/p', 0.75);
             UPDATE objects SET measurement_values = [0.25] WHERE object_id = '{object}';"
        ))
        .unwrap();
    (generator, out_dir, object)
}

/// One row per value, the columns named like the database's; a value
/// that isn't there stays empty.
#[test]
fn the_measurements_export_has_a_row_per_value() {
    let (generator, out_dir, object) = with_measurements();
    ResultExport {
        output_dir: out_dir.clone(),
        format: ExportFormat::CSV,
        with_measurements: true,
        ..Default::default()
    }
    .start_export(&generator, &no_cancel(), &mut no_progress())
    .expect("export");
    let csv = std::fs::read_to_string(out_dir.join("measurements.csv")).unwrap();
    let lines: Vec<&str> = csv.lines().collect();
    assert_eq!(
        lines,
        [
            "image_name,image_rel_path,z_stack,t_stack,region,object_id,related_object_id,\
             object_class_name,step,measurement,unit,key,value",
            "img1.tif,img1.tif,0,0,image,,,,p_01,Pearson,None,p_01/p,0.5",
            &format!(
                "img1.tif,img1.tif,0,0,object,{object},{object},\"[\"\"ClassA\"\"]\",q_01,Pearson,None,q_01/q,0.25"
            ),
            "img1.tif,img1.tif,0,1,image,,,,p_01,Pearson,None,p_01/p,",
            "img2.tif,img2.tif,0,0,image,,,,p_01,Pearson,None,p_01/p,0.75",
        ]
    );
}

/// Format 2: a profile (rows with a position) isn't offered as a value,
/// and the export gets a `position` column - empty for ordinary values.
#[test]
fn a_profile_is_exported_with_its_positions_and_not_offered_as_a_value() {
    let (generator, out_dir, object) = with_measurements();
    generator
        .connection()
        .execute_batch(&format!(
            "ALTER TABLE measurement_info ADD COLUMN profile BOOLEAN DEFAULT false;
             ALTER TABLE measurements ADD COLUMN position DOUBLE;
             INSERT INTO measurement_info VALUES
                 ('s_01/i', 'Intersections', 's_01', 'Count', 'Object', NULL, NULL, true, NULL, true);
             INSERT INTO measurements VALUES
                 ('img1.tif', 'img1.tif', 0, 0, 'object', '{object}', NULL, NULL, NULL, 's_01', 'Intersections', 'Count', 's_01/i', 4, 2000),
                 ('img1.tif', 'img1.tif', 0, 0, 'object', '{object}', NULL, NULL, NULL, 's_01', 'Intersections', 'Count', 's_01/i', 2, 1000);"
        ))
        .unwrap();
    let offered: Vec<String> = generator
        .get_measurement_columns()
        .unwrap()
        .into_iter()
        .map(|c| c.key)
        .collect();
    assert_eq!(offered, ["p_01/p", "q_01/q"]);
    ResultExport {
        output_dir: out_dir.clone(),
        format: ExportFormat::CSV,
        with_measurements: true,
        ..Default::default()
    }
    .start_export(&generator, &no_cancel(), &mut no_progress())
    .expect("export");
    let csv = std::fs::read_to_string(out_dir.join("measurements.csv")).unwrap();
    let lines: Vec<&str> = csv.lines().collect();
    assert!(lines[0].ends_with(",key,value,position"), "{}", lines[0]);
    assert!(lines.contains(&"img1.tif,img1.tif,0,0,image,,,,p_01,Pearson,None,p_01/p,0.5,"));
    let profile: Vec<&str> = lines
        .iter()
        .copied()
        .filter(|l| l.contains("s_01/i"))
        .collect();
    assert_eq!(
        profile,
        [
            format!(
                "img1.tif,img1.tif,0,0,object,{object},,,s_01,Intersections,Count,s_01/i,2,1000"
            ),
            format!(
                "img1.tif,img1.tif,0,0,object,{object},,,s_01,Intersections,Count,s_01/i,4,2000"
            ),
        ]
    );
}

/// A temp folder to export into, used as its path; dropped, it goes.
struct OutDir(tempfile::TempDir);

impl std::ops::Deref for OutDir {
    type Target = std::path::Path;
    fn deref(&self) -> &std::path::Path {
        self.0.path()
    }
}

fn profile_export(
    content: crate::results::ProfileExportContent,
    layout: TimeCourseLayout,
) -> (OutDir, ResultExport) {
    let dir = OutDir(tempfile::tempdir().unwrap());
    let export = ResultExport {
        output_dir: dir.to_path_buf(),
        format: ExportFormat::CSV,
        with_profile: true,
        profile: crate::results::ProfileExport {
            key: "sholl_01/intersections".into(),
            lines: crate::results::ProfileLines::Class,
            content,
            layout,
        },
        ..Default::default()
    };
    (dir, export)
}

/// The chart's lines: long a row per line × radius (a missing radius
/// counted as 0), wide a file per statistic.
#[test]
fn the_profile_lines_export_long_and_wide() {
    let db = super::super::test_support::profile_database();
    let (dir, export) = profile_export(
        crate::results::ProfileExportContent::Lines,
        TimeCourseLayout::Both,
    );
    export
        .start_export(&db, &no_cancel(), &mut no_progress())
        .expect("export");
    let long = std::fs::read_to_string(dir.join("profile_lines.csv")).unwrap();
    let lines: Vec<&str> = long.lines().collect();
    assert_eq!(lines[0], "line,position,n_objects,mean,sd,median,q1,q3");
    assert!(lines.iter().any(|l| l.starts_with("A,30,2,1,")), "{long}");
    assert_eq!(lines.len(), 1 + 3 + 1, "{long}");
    let mean = std::fs::read_to_string(dir.join("profile_lines_mean.csv")).unwrap();
    assert_eq!(
        mean.lines().collect::<Vec<_>>(),
        ["position,A,B", "10,3,1", "20,5,", "30,1,"]
    );
}

/// Each neuron's profile: long its rows as stored, wide a row per
/// neuron with 0 where it doesn't reach.
#[test]
fn the_profile_objects_export_long_and_wide() {
    let db = super::super::test_support::profile_database();
    let (dir, export) = profile_export(
        crate::results::ProfileExportContent::Objects,
        TimeCourseLayout::Both,
    );
    export
        .start_export(&db, &no_cancel(), &mut no_progress())
        .expect("export");
    let long = std::fs::read_to_string(dir.join("profile_objects.csv")).unwrap();
    assert_eq!(long.lines().count(), 1 + 6, "{long}");
    let wide = std::fs::read_to_string(dir.join("profile_objects_wide.csv")).unwrap();
    let rows: Vec<&str> = wide.lines().collect();
    assert_eq!(
        rows[0],
        "image_name,image_rel_path,object_id,object_class_name,r_10,r_20,r_30"
    );
    // The first neuron doesn't reach r 30: 0 there.
    assert!(rows[1].ends_with(",2,4,0"), "{wide}");
    assert!(rows[3].ends_with(",1,0,0"), "{wide}");
}

#[test]
fn a_profile_export_needs_its_measurement() {
    let db = super::super::test_support::profile_database();
    let (_dir, mut export) = profile_export(
        crate::results::ProfileExportContent::Lines,
        TimeCourseLayout::Long,
    );
    export.profile.key.clear();
    assert!(
        export
            .start_export(&db, &no_cancel(), &mut no_progress())
            .is_err()
    );
}

/// A value of a whole volume has no plane (`z_stack` NULL): an empty
/// cell, not a failed export.
#[test]
fn a_whole_volume_value_exports_with_an_empty_z() {
    let (generator, out_dir, _) = with_measurements();
    generator
        .connection()
        .execute_batch(
            "INSERT INTO measurements VALUES ('img2.tif', 'img2.tif', NULL, 0, 'image', \
             NULL, NULL, NULL, NULL, 'p_01', 'Pearson', 'None', 'p_01/p', 0.5);",
        )
        .unwrap();
    ResultExport {
        output_dir: out_dir.clone(),
        format: ExportFormat::CSV,
        with_measurements: true,
        ..Default::default()
    }
    .start_export(&generator, &no_cancel(), &mut no_progress())
    .expect("export");
    let csv = std::fs::read_to_string(out_dir.join("measurements.csv")).unwrap();
    assert!(
        csv.lines()
            .any(|l| l == "img2.tif,img2.tif,,0,image,,,,p_01,Pearson,None,p_01/p,0.5"),
        "{csv}"
    );
}

/// The measurements chosen among the columns, of the chosen images and
/// time points.
#[test]
fn the_measurements_export_keeps_to_the_chosen_ones() {
    let (generator, out_dir, _) = with_measurements();
    let p = generator
        .get_measurement_columns()
        .unwrap()
        .into_iter()
        .find(|m| m.key == "p_01/p")
        .unwrap();
    ResultExport {
        output_dir: out_dir.clone(),
        format: ExportFormat::CSV,
        with_measurements: true,
        columns: vec![Column::AreaSizePx, Column::Measurement(p)],
        image_rel_paths: vec!["img1.tif".into()],
        t_stacks: Some(Range { start: 0, end: 1 }),
        ..Default::default()
    }
    .start_export(&generator, &no_cancel(), &mut no_progress())
    .expect("export");
    let csv = std::fs::read_to_string(out_dir.join("measurements.csv")).unwrap();
    assert_eq!(
        csv.lines().skip(1).collect::<Vec<_>>(),
        ["img1.tif,img1.tif,0,0,image,,,,p_01,Pearson,None,p_01/p,0.5"]
    );
}

#[test]
fn results_without_measurements_say_so_and_parquet_writes_them_too() {
    let (generator, out_dir) = open(&[ObjectSpec::new("img1.tif", "ClassA", 1, 10)]);
    let err = ResultExport {
        output_dir: out_dir.clone(),
        format: ExportFormat::CSV,
        with_measurements: true,
        ..Default::default()
    }
    .start_export(&generator, &no_cancel(), &mut no_progress())
    .err()
    .expect("refused")
    .to_string();
    assert!(err.contains("no measurements"), "{err}");

    let (generator, out_dir, _) = with_measurements();
    ResultExport {
        output_dir: out_dir.clone(),
        format: ExportFormat::Parquet,
        ..Default::default()
    }
    .start_export(&generator, &no_cancel(), &mut no_progress())
    .expect("export");
    // Read through a connection of its own: the results' connection
    // may read no file outside its staging folder.
    let rows: i64 = duckdb::Connection::open_in_memory()
        .unwrap()
        .query_row(
            &format!(
                "SELECT count(*) FROM '{}'",
                out_dir.join("measurements.parquet").display()
            ),
            [],
            |row| row.get(0),
        )
        .unwrap();
    assert_eq!(rows, 4);
}

// -- more documents and their content ---------------------------------

/// A sheet's cells as text: numbers as Rust prints them ("0.5", "10"),
/// empty cells "".
fn sheet_rows(path: &Path, sheet: &str) -> Vec<Vec<String>> {
    let mut workbook: calamine::Xlsx<_> = calamine::open_workbook(path).expect("open xlsx");
    let range = workbook.worksheet_range(sheet).expect("the sheet");
    range
        .rows()
        .map(|row| {
            row.iter()
                .map(|cell| match cell {
                    calamine::Data::Float(v) => v.to_string(),
                    calamine::Data::Int(v) => v.to_string(),
                    calamine::Data::String(s) => s.clone(),
                    calamine::Data::Empty => String::new(),
                    other => panic!("unexpected cell {other:?}"),
                })
                .collect()
        })
        .collect()
}

/// A CSV document's lines.
fn csv_lines(path: &Path) -> Vec<String> {
    std::fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("{}: {e}", path.display()))
        .lines()
        .map(str::to_string)
        .collect()
}

/// XLSX: the measurements in a "Measurements" sheet - the same rows as
/// the CSV (see `the_measurements_export_has_a_row_per_value`), the
/// numbers as numbers.
#[test]
fn the_measurements_xlsx_has_a_row_per_value() {
    let (generator, out_dir, object) = with_measurements();
    ResultExport {
        output_dir: out_dir.clone(),
        format: ExportFormat::XLSX,
        with_measurements: true,
        ..Default::default()
    }
    .start_export(&generator, &no_cancel(), &mut no_progress())
    .expect("export");
    let rows = sheet_rows(&out_dir.join("measurements.xlsx"), "Measurements");
    assert_eq!(rows.len(), 5, "the header and four values");
    assert_eq!(rows[0][0], "image_name");
    assert_eq!(rows[0][12], "value");
    assert_eq!(
        rows[1],
        [
            "img1.tif", "img1.tif", "0", "0", "image", "", "", "", "p_01", "Pearson", "None",
            "p_01/p", "0.5"
        ]
    );
    assert_eq!(
        rows[2],
        [
            "img1.tif",
            "img1.tif",
            "0",
            "0",
            "object",
            &object,
            &object,
            "[\"ClassA\"]",
            "q_01",
            "Pearson",
            "None",
            "q_01/q",
            "0.25"
        ]
    );
    // T1's value isn't there: empty.
    assert_eq!(rows[3][3], "1");
    assert_eq!(rows[3][12], "");
    assert_eq!(rows[4][0], "img2.tif");
    assert_eq!(rows[4][12], "0.75");
}

/// Chosen images none of which the results hold: no value - the header
/// alone, not every value.
#[test]
fn the_measurements_of_unknown_images_are_only_the_header() {
    let (generator, out_dir, _) = with_measurements();
    ResultExport {
        output_dir: out_dir.clone(),
        format: ExportFormat::CSV,
        with_measurements: true,
        image_rel_paths: vec!["missing.tif".into()],
        ..Default::default()
    }
    .start_export(&generator, &no_cancel(), &mut no_progress())
    .expect("export");
    let lines = csv_lines(&out_dir.join("measurements.csv"));
    assert_eq!(lines.len(), 1, "{lines:?}");
    assert!(lines[0].starts_with("image_name,"), "{}", lines[0]);
}

/// `track_seven` with a pixel size of 500 nm (0.5 µm) and 30 s between
/// time points, and its motion: path length 5 px, VCL 5 px per time
/// point, BCF 1 per time point, MSD D 4 px² per time point.
fn timed_track_seven() -> (TestResults, PathBuf) {
    let (database, out_dir) = open_analysed(&track_seven(), 2);
    database
        .connection()
        .execute_batch(
            "ALTER TABLE images ADD COLUMN frame_interval_s DOUBLE;
             UPDATE images SET frame_interval_s = 30;
             UPDATE objects SET pixel_size_x_nm = 500;
             CREATE TABLE measurements (image_name VARCHAR, image_rel_path VARCHAR, \
             z_stack INTEGER, t_stack INTEGER, region VARCHAR, object_id UUID, \
             related_object_id UUID, object_class_name VARCHAR, object_class_id INTEGER[], \
             step VARCHAR, measurement VARCHAR, unit VARCHAR, key VARCHAR, value DOUBLE, \
             position DOUBLE, track_id UBIGINT);
             INSERT INTO measurements VALUES
                 ('img1.tif', 'img1.tif', 0, 0, 'track', NULL, NULL, NULL, [1], 'motion_01', \
                  'Path length', 'Pixels', 'motion_01/path_length', 5.0, NULL, 7),
                 ('img1.tif', 'img1.tif', 0, 0, 'track', NULL, NULL, NULL, [1], 'motion_01', \
                  'VCL', 'PixelsPerFrame', 'motion_01/vcl', 5.0, NULL, 7),
                 ('img1.tif', 'img1.tif', 0, 0, 'track', NULL, NULL, NULL, [1], 'motion_01', \
                  'BCF', 'PerFrame', 'motion_01/bcf', 1.0, NULL, 7),
                 ('img1.tif', 'img1.tif', 0, 0, 'track', NULL, NULL, NULL, [1], 'motion_01', \
                  'MSD D', 'PixelsSquaredPerFrame', 'motion_01/msd_d', 4.0, NULL, 7);",
        )
        .unwrap();
    (database, out_dir)
}

/// With a pixel size and a frame interval: the duration in minutes
/// (one step of 30 s: 0.5 min) and the motion in µm per minute - VCL
/// 5 px x 0.5 µm x 2 steps per min = 5, BCF 1 x 2 = 2, MSD D
/// 4 x 0.25 µm² x 2 = 2; the median and Q1 of the track's areas 10 and
/// 30: 20 and 15. XLSX: a "Tracks" sheet.
#[test]
fn the_tracks_xlsx_has_the_motion_in_micrometres_per_minute() {
    let (database, out_dir) = timed_track_seven();
    ResultExport {
        output_dir: out_dir.clone(),
        format: ExportFormat::XLSX,
        columns: vec![Column::AreaSizePx],
        with_track_summary: true,
        track_summary: crate::results::TrackSummaryExport {
            statistics: vec![TimeCourseStat::Median, TimeCourseStat::Q1],
            speed_unit: SpeedUnit::UmPerMin,
            ..Default::default()
        },
        ..Default::default()
    }
    .start_export(&database, &no_cancel(), &mut no_progress())
    .expect("tracks export");
    let rows = sheet_rows(&out_dir.join("tracks.xlsx"), "Tracks");
    assert_eq!(rows.len(), 2, "the header and track 7");
    let column = |name: &str| {
        rows[0]
            .iter()
            .position(|h| h == name)
            .unwrap_or_else(|| panic!("no {name} in {:?}", rows[0]))
    };
    let value = |name: &str| rows[1][column(name)].clone();
    assert_eq!(value("Track"), "7");
    assert_eq!(value("Duration [min]"), "0.5");
    assert_eq!(value("Path length [µm]"), "2.5");
    assert_eq!(value("VCL (mean speed) [µm/min]"), "5");
    assert_eq!(value("BCF [1/min]"), "2");
    assert_eq!(value("MSD D [µm²/min]"), "2");
    assert_eq!(value("Angular speed [°/min]"), "");
    assert_eq!(value("Area [px] median"), "20");
    assert_eq!(value("Area [px] Q1"), "15");
}

/// In µm per second: VCL 5 x 0.5 µm / 30 s, BCF 1 / 30 s, MSD D
/// 4 x 0.25 µm² / 30 s; the Q3 of the areas 10 and 30: 25.
#[test]
fn the_tracks_export_has_the_motion_in_micrometres_per_second() {
    let (database, out_dir) = timed_track_seven();
    ResultExport {
        output_dir: out_dir.clone(),
        format: ExportFormat::CSV,
        columns: vec![Column::AreaSizePx],
        with_track_summary: true,
        track_summary: crate::results::TrackSummaryExport {
            statistics: vec![TimeCourseStat::Q3],
            speed_unit: SpeedUnit::UmPerS,
            ..Default::default()
        },
        ..Default::default()
    }
    .start_export(&database, &no_cancel(), &mut no_progress())
    .expect("tracks export");
    let lines = csv_lines(&out_dir.join("tracks.csv"));
    let header: Vec<&str> = lines[0].split(',').collect();
    let row: Vec<f64> = lines[1]
        .split(',')
        .map(|v| v.parse().unwrap_or(f64::NAN))
        .collect();
    let value = |name: &str| {
        row[header
            .iter()
            .position(|h| *h == name)
            .unwrap_or_else(|| panic!("no {name} in {header:?}"))]
    };
    let close = |a: f64, b: f64| (a - b).abs() < 1e-12;
    assert!(close(value("VCL (mean speed) [µm/s]"), 2.5 / 30.0));
    assert!(close(value("BCF [1/s]"), 1.0 / 30.0));
    assert!(close(value("MSD D [µm²/s]"), 1.0 / 30.0));
    assert!(header.contains(&"Angular speed [°/s]"), "{header:?}");
    assert_eq!(value("Area [px] Q3"), 25.0);
}

/// Parquet holds no tracks or time course: asked for directly, each is
/// refused rather than writing nothing.
#[test]
fn tracks_and_time_course_are_refused_as_parquet() {
    let (database, out_dir) = open_analysed(&track_seven(), 2);
    let export = ResultExport {
        output_dir: out_dir,
        format: ExportFormat::Parquet,
        columns: vec![Column::AreaSizePx],
        ..Default::default()
    };
    let tracks = export
        .export_track_summary(&database, &no_cancel(), &mut no_progress())
        .expect_err("tracks as parquet");
    assert!(
        matches!(&tracks, InternalErrors::Internal(m) if m.contains("doesn't support the tracks")),
        "{tracks}"
    );
    let course = export
        .export_time_course(&database, &no_cancel(), &mut no_progress())
        .expect_err("time course as parquet");
    assert!(
        matches!(&course, InternalErrors::Internal(m) if m.contains("doesn't support the time course")),
        "{course}"
    );
}

/// Well lines: a row per time point, well and class - every class of
/// every well, also where none was found; 120 s between time points
/// (the series runs 120 s): the time in minutes. Wide: a column per
/// well and class.
#[test]
fn the_time_course_per_well_has_its_wells_and_the_time_in_minutes() {
    let (database, out_dir) = open_analysed(
        &[
            ObjectSpec::new("A1_01.tif", "ClassA", 1, 10),
            ObjectSpec::new("A1_01.tif", "ClassB", 2, 40),
            ObjectSpec::new("B2_01.tif", "ClassA", 1, 100),
            ObjectSpec::new("A1_01.tif", "ClassA", 1, 30).at_plane(0, 1),
        ],
        2,
    );
    database
        .connection()
        .execute_batch(
            "ALTER TABLE images ADD COLUMN frame_interval_s DOUBLE;
             UPDATE images SET frame_interval_s = 120;",
        )
        .unwrap();
    ResultExport {
        output_dir: out_dir.clone(),
        format: ExportFormat::CSV,
        columns: vec![Column::AreaSizePx],
        with_time_course: true,
        time_course: TimeCourseExport {
            lines: TimeCourseLines::Well,
            statistics: vec![TimeCourseStat::Mean],
            layout: TimeCourseLayout::Both,
            ..Default::default()
        },
        ..Default::default()
    }
    .start_export(&database, &no_cancel(), &mut no_progress())
    .expect("time course export");
    assert_eq!(
        csv_lines(&out_dir.join("time_course.csv")),
        [
            "T,Time [min],Well,Class,N images,N objects,Area [px] mean",
            "0,0,A1,ClassA,1,1,10",
            "0,0,A1,ClassB,1,1,40",
            "0,0,B2,ClassA,1,1,100",
            "0,0,B2,ClassB,1,0,",
            "1,2,A1,ClassA,1,1,30",
            "1,2,A1,ClassB,1,0,",
            "1,2,B2,ClassA,1,0,",
            "1,2,B2,ClassB,1,0,",
        ]
    );
    assert_eq!(
        csv_lines(&out_dir.join("time_course_Area__px__mean.csv")),
        [
            "T,Time [min],A1 · ClassA,A1 · ClassB,B2 · ClassA,B2 · ClassB",
            "0,0,10,40,100,",
            "1,2,30,,,",
        ]
    );
}

/// Wide only, image lines of one class: no long file; a column per
/// image, named by the image alone.
#[test]
fn a_wide_only_time_course_per_image_names_the_columns_by_image() {
    let (database, out_dir) = open_analysed(
        &[
            ObjectSpec::new("img1.tif", "ClassA", 1, 100),
            ObjectSpec::new("img2.tif", "ClassA", 1, 200),
            ObjectSpec::new("img1.tif", "ClassA", 1, 300).at_plane(0, 1),
        ],
        2,
    );
    ResultExport {
        output_dir: out_dir.clone(),
        format: ExportFormat::CSV,
        columns: vec![Column::AreaSizePx],
        with_time_course: true,
        time_course: TimeCourseExport {
            lines: TimeCourseLines::Image,
            statistics: vec![TimeCourseStat::Mean],
            layout: TimeCourseLayout::Wide,
            ..Default::default()
        },
        ..Default::default()
    }
    .start_export(&database, &no_cancel(), &mut no_progress())
    .expect("time course export");
    assert!(!out_dir.join("time_course.csv").exists());
    assert_eq!(
        csv_lines(&out_dir.join("time_course_Area__px__mean.csv")),
        ["T,img1.tif,img2.tif", "0,100,200", "1,300,"]
    );
}

/// Track lines of tracked results: the track and its image, and the
/// number of tracks - the untracked object is in no line; SD of one
/// object is none.
#[test]
fn the_time_course_per_track_counts_the_tracks() {
    let (database, out_dir) = open_analysed(&track_seven(), 2);
    ResultExport {
        output_dir: out_dir.clone(),
        format: ExportFormat::CSV,
        columns: vec![Column::AreaSizePx],
        with_time_course: true,
        time_course: TimeCourseExport {
            lines: TimeCourseLines::Track,
            layout: TimeCourseLayout::Long,
            ..Default::default()
        },
        ..Default::default()
    }
    .start_export(&database, &no_cancel(), &mut no_progress())
    .expect("time course export");
    assert_eq!(
        csv_lines(&out_dir.join("time_course.csv")),
        [
            "T,Track,Image,Class,N images,N objects,N tracks,Area [px] mean,Area [px] SD",
            "0,#7,img1.tif,ClassA,1,1,1,10,",
            "1,#7,img1.tif,ClassA,1,1,1,30,",
        ]
    );
}

/// Several z-planes: a "Z" after "T"; a series of 7200 s (an hour
/// between time points): the time in hours.
#[test]
fn the_time_course_of_z_planes_has_z_and_the_time_in_hours() {
    let (database, out_dir) = open_analysed(
        &[
            ObjectSpec::new("img1.tif", "ClassA", 1, 10),
            ObjectSpec::new("img1.tif", "ClassA", 1, 20).at_plane(1, 0),
            ObjectSpec::new("img1.tif", "ClassA", 1, 30).at_plane(0, 2),
        ],
        3,
    );
    database
        .connection()
        .execute_batch(
            "ALTER TABLE images ADD COLUMN frame_interval_s DOUBLE;
             UPDATE images SET frame_interval_s = 3600, z_stacks = 2, \
                 z_stacks_analyzed = [0, 1];",
        )
        .unwrap();
    ResultExport {
        output_dir: out_dir.clone(),
        format: ExportFormat::CSV,
        columns: vec![Column::AreaSizePx],
        with_time_course: true,
        time_course: TimeCourseExport {
            statistics: vec![TimeCourseStat::Mean],
            layout: TimeCourseLayout::Long,
            ..Default::default()
        },
        ..Default::default()
    }
    .start_export(&database, &no_cancel(), &mut no_progress())
    .expect("time course export");
    assert_eq!(
        csv_lines(&out_dir.join("time_course.csv")),
        [
            "T,Z,Time [h],Class,N images,N objects,Area [px] mean",
            "0,0,0,ClassA,1,1,10",
            "0,1,0,ClassA,1,1,20",
            "1,0,1,ClassA,1,0,",
            "1,1,1,ClassA,1,0,",
            "2,0,2,ClassA,1,1,30",
            "2,1,2,ClassA,1,0,",
        ]
    );
}

/// Plate grids of results with several z-planes: a tab per z-plane,
/// named with it.
#[test]
fn plate_grids_get_a_tab_per_z_plane() {
    let (database, out_dir) = open(&[
        ObjectSpec::new("A1_01.tif", "ClassA", 1, 10),
        ObjectSpec::new("A1_01.tif", "ClassA", 1, 30).at_plane(1, 0),
    ]);
    ResultExport {
        output_dir: out_dir.clone(),
        format: ExportFormat::XLSX,
        columns: vec![Column::AreaSizePx],
        aggregations: vec![Aggregation::Avg],
        with_plate_view_heatmap: true,
        ..Default::default()
    }
    .start_export(&database, &no_cancel(), &mut no_progress())
    .expect("plate export");
    let mut workbook: calamine::Xlsx<_> =
        calamine::open_workbook(out_dir.join("plate.xlsx")).expect("open plate.xlsx");
    assert_eq!(workbook.sheet_names(), ["Z0 ClassA", "Z1 ClassA"]);
    let z1 = workbook.worksheet_range("Z1 ClassA").unwrap();
    assert_eq!(data_f64(z1.get_value((2, 1))), 30.0, "A1 on z-plane 1");
}

/// The profile's lines over all objects: one line "all" - at r 10 all
/// three neurons, (2 + 4 + 1) / 3; at r 20 (4 + 6 + 0) / 3 and at r 30
/// (0 + 2 + 0) / 3, a neuron not reaching a radius counting 0.
#[test]
fn the_profile_line_of_all_objects_is_named_all() {
    let db = super::super::test_support::profile_database();
    let (dir, mut export) = profile_export(
        crate::results::ProfileExportContent::Lines,
        TimeCourseLayout::Long,
    );
    export.profile.lines = ProfileLines::All;
    export
        .start_export(&db, &no_cancel(), &mut no_progress())
        .expect("export");
    let lines = csv_lines(&dir.join("profile_lines.csv"));
    assert_eq!(lines.len(), 1 + 3, "{lines:?}");
    let mean_at = |i: usize| -> (String, f64, f64, f64) {
        let fields: Vec<&str> = lines[i].split(',').collect();
        (
            fields[0].to_string(),
            fields[1].parse().unwrap(),
            fields[2].parse().unwrap(),
            fields[3].parse().unwrap(),
        )
    };
    for (i, (position, mean)) in [(10.0, 7.0 / 3.0), (20.0, 10.0 / 3.0), (30.0, 2.0 / 3.0)]
        .into_iter()
        .enumerate()
    {
        let (line, at, n, value) = mean_at(i + 1);
        assert_eq!((line.as_str(), at, n), ("all", position, 3.0), "{lines:?}");
        assert!((value - mean).abs() < 1e-12, "{lines:?}");
    }
    assert!(!dir.join("profile_lines_mean.csv").exists(), "long only");
}

/// Image lines, wide: a column per image - A01's two neurons' mean
/// per radius, B01's neuron only at r 10, where it is.
#[test]
fn the_profile_lines_per_image_have_a_column_per_image() {
    let db = super::super::test_support::profile_database();
    let (dir, mut export) = profile_export(
        crate::results::ProfileExportContent::Lines,
        TimeCourseLayout::Wide,
    );
    export.profile.lines = ProfileLines::Image;
    export
        .start_export(&db, &no_cancel(), &mut no_progress())
        .expect("export");
    assert!(!dir.join("profile_lines.csv").exists(), "wide only");
    assert_eq!(
        csv_lines(&dir.join("profile_lines_mean.csv")),
        ["position,A01_1.tif,B01_1.tif", "10,3,1", "20,5,", "30,1,"]
    );
    assert_eq!(
        csv_lines(&dir.join("profile_lines_n_objects.csv")),
        ["position,A01_1.tif,B01_1.tif", "10,2,1", "20,2,", "30,2,"]
    );
}

/// XLSX: the profile's lines in one workbook - the long sheet and one
/// per statistic, the same values as the CSV.
#[test]
fn the_profile_lines_xlsx_has_a_sheet_per_table() {
    let db = super::super::test_support::profile_database();
    let (dir, mut export) = profile_export(
        crate::results::ProfileExportContent::Lines,
        TimeCourseLayout::Both,
    );
    export.format = ExportFormat::XLSX;
    export
        .start_export(&db, &no_cancel(), &mut no_progress())
        .expect("export");
    let path = dir.join("profile_lines.xlsx");
    let workbook: calamine::Xlsx<_> = calamine::open_workbook(&path).expect("open");
    assert_eq!(
        workbook.sheet_names(),
        ["Long", "mean", "sd", "median", "q1", "q3", "n_objects"]
    );
    assert_eq!(
        sheet_rows(&path, "mean"),
        [
            vec!["position", "A", "B"],
            vec!["10", "3", "1"],
            vec!["20", "5", ""],
            vec!["30", "1", ""],
        ]
    );
    let long = sheet_rows(&path, "Long");
    assert_eq!(long[0][..4], ["line", "position", "n_objects", "mean"]);
    assert_eq!(long.len(), 1 + 3 + 1, "A at three radii, B at one");
}

/// XLSX: each object's profile - long its stored rows, wide a row per
/// object with 0 where it doesn't reach.
#[test]
fn the_profile_objects_xlsx_has_a_long_and_a_wide_sheet() {
    let db = super::super::test_support::profile_database();
    let (dir, mut export) = profile_export(
        crate::results::ProfileExportContent::Objects,
        TimeCourseLayout::Both,
    );
    export.format = ExportFormat::XLSX;
    export
        .start_export(&db, &no_cancel(), &mut no_progress())
        .expect("export");
    let path = dir.join("profile_objects.xlsx");
    let workbook: calamine::Xlsx<_> = calamine::open_workbook(&path).expect("open");
    assert_eq!(workbook.sheet_names(), ["Long", "Wide"]);
    let long = sheet_rows(&path, "Long");
    assert_eq!(
        long[0],
        [
            "image_name",
            "image_rel_path",
            "object_id",
            "object_class_name",
            "position",
            "value"
        ]
    );
    assert_eq!(long.len(), 1 + 6);
    assert_eq!(long[1][2], "00000000-0000-0000-0000-000000000000");
    assert_eq!(long[1][4..], ["10", "2"]);
    let wide = sheet_rows(&path, "Wide");
    assert_eq!(wide[0][4..], ["r_10", "r_20", "r_30"]);
    assert_eq!(wide[1][4..], ["2", "4", "0"]);
    assert_eq!(wide[2][4..], ["4", "6", "2"]);
    assert_eq!(wide[3][4..], ["1", "0", "0"]);
}

/// Each object's profile, wide only, of the chosen image and class:
/// only A01's two class-A neurons, no long file.
#[test]
fn the_profile_objects_keep_to_the_chosen_image_and_class() {
    let db = super::super::test_support::profile_database();
    let (dir, mut export) = profile_export(
        crate::results::ProfileExportContent::Objects,
        TimeCourseLayout::Wide,
    );
    export.image_rel_paths = vec!["A01_1.tif".into(), "B01_1.tif".into()];
    export.object_classes = vec![ObjectClass::Global(1)];
    export
        .start_export(&db, &no_cancel(), &mut no_progress())
        .expect("export");
    assert!(!dir.join("profile_objects.csv").exists(), "wide only");
    let wide = csv_lines(&dir.join("profile_objects_wide.csv"));
    assert_eq!(wide.len(), 1 + 2, "{wide:?}");
    assert!(
        wide[1].contains("A01_1.tif") && wide[1].ends_with(",2,4,0"),
        "{wide:?}"
    );
    assert!(
        wide[2].contains("A01_1.tif") && wide[2].ends_with(",4,6,2"),
        "{wide:?}"
    );
}

/// Results without profiles (no `position`, before format 2): each
/// object's profile is refused, saying so.
#[test]
fn profile_objects_of_results_without_profiles_are_refused() {
    let (generator, _, _) = with_measurements();
    let (_dir, export) = profile_export(
        crate::results::ProfileExportContent::Objects,
        TimeCourseLayout::Long,
    );
    let error = export
        .start_export(&generator, &no_cancel(), &mut no_progress())
        .expect_err("no profiles");
    assert!(
        matches!(&error, InternalErrors::InvalidArgument(m) if m.contains("hold no profiles")),
        "{error}"
    );
}

/// A file that can be neither renamed nor copied: an error naming the
/// destination.
#[test]
fn a_file_that_cannot_be_moved_is_an_error_naming_where_it_should_go() {
    let dir = tempfile::tempdir().unwrap();
    let to = dir.path().join("objects.parquet");
    let error = move_file(&dir.path().join("missing.parquet"), &to).expect_err("no source");
    assert!(
        matches!(&error, InternalErrors::Io(m) if m.contains("could not write") && m.contains("objects.parquet")),
        "{error}"
    );
}
