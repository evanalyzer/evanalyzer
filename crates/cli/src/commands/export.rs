use crate::args::{ExportArgs, ExportCommand, TableExportArgs};
use crate::commands::common::{cell_text, resolve_grouping, resolve_image_rel_paths, resolve_object_classes};
use evanalyzer_app::result::{
    Column, DatabaseResult, ExportFormat, GroupedByImageFilter, ListFilter, Pagination,
    PlaneFilter, ResultExport, ResultsGenerator,
};
use evanalyzer_cfg::core_types::InternalErrors;
use std::path::Path;

pub fn run(args: ExportArgs) -> Result<(), InternalErrors> {
    match args.command {
        ExportCommand::Csv(table) => export_table(table, true),
        ExportCommand::Xlsx(table) => export_table(table, false),
    }
}

fn export_table(args: TableExportArgs, csv: bool) -> Result<(), InternalErrors> {
    let db = ResultsGenerator::open_database(args.db.clone())?;
    if args.filter.colocalized.is_some() {
        return Err(InternalErrors::InvalidArgument(
            "--colocalized isn't supported by the current results backend".to_string(),
        ));
    }
    let grouping = resolve_grouping(&args.group)?;
    let image_rel_paths = resolve_image_rel_paths(&db, &args.filter.images)?;
    let object_classes = resolve_object_classes(&db, &args.filter.classes)?;
    let columns: Vec<Column> = db
        .get_available_columns()?
        .into_iter()
        .map(|entry| entry.key)
        .collect();

    if csv {
        let result = if grouping.group_by_image {
            let groupable_columns = columns.into_iter().filter(is_groupable_column).collect();
            fetch_all_grouped_by_image(
                &db,
                &GroupedByImageFilter {
                    plane: PlaneFilter {
                        z_stack: 0,
                        t_stack: 0,
                    },
                    images: (!image_rel_paths.is_empty()).then_some(image_rel_paths),
                    object_classes: (!object_classes.is_empty()).then_some(object_classes),
                    columns: groupable_columns,
                    aggregation: grouping.aggregations,
                    page: Pagination {
                        limit: 0,
                        after: None,
                    },
                },
            )?
        } else {
            // Every z/t plane, like the XLSX path's own `list.xlsx`
            // (`write_list_sheet`'s `for z ... for t ...` loop) — a flat,
            // ungrouped export means "every object," not just whichever
            // happens to sit at the first plane.
            let images_for_flat = (!image_rel_paths.is_empty()).then_some(image_rel_paths);
            let object_classes_for_flat = (!object_classes.is_empty()).then_some(object_classes);
            let mut merged = empty_database_result();
            let mut first_plane = true;
            // `get_nr_of_z_stacks`/`get_nr_of_t_stacks` return the *max*
            // stack index (not a count), so the exclusive upper bound of a
            // range covering every plane is that value plus one.
            for z_stack in 0..(db.get_nr_of_z_stacks() + 1) {
                for t_stack in 0..(db.get_nr_of_t_stacks() + 1) {
                    let mut plane_result = fetch_all_objects(
                        &db,
                        &ListFilter {
                            plane: PlaneFilter { z_stack, t_stack },
                            images: images_for_flat.clone(),
                            object_classes: object_classes_for_flat.clone(),
                            columns: columns.clone(),
                            with_coloc_details: false,
                            page: Pagination {
                                limit: 0,
                                after: None,
                            },
                        },
                    )?;
                    if first_plane {
                        merged.column_names = std::mem::take(&mut plane_result.column_names);
                        first_plane = false;
                    }
                    merged.row_names.extend(plane_result.row_names);
                    merged.rows.extend(plane_result.rows);
                    merged.source_object_count += plane_result.source_object_count;
                }
            }
            merged
        };
        write_csv(&result, &args.out)?;
        println!("Exported to {}", args.out.display());
        return Ok(());
    }

    // XLSX: reuse `ResultExport` directly rather than re-deriving its
    // List/Group-by-image writing logic — this is exactly the "wire the CLI
    // back up with the new ResultsExporter" the CLI lost. It writes into a
    // directory under its own fixed filenames (`list.xlsx`/
    // `grouped_by_image.xlsx`), which would collide with anything already
    // sitting next to `--out`, so it writes into a private scratch
    // directory first and moves the one file it produces to `--out`.
    let scratch_dir = std::env::temp_dir().join(format!(
        "evanalyzer_cli_export_{}_{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos(),
    ));
    let export = ResultExport {
        output_dir: scratch_dir.clone(),
        format: ExportFormat::XLSX,
        // Same "max stack index, not a count" caveat as the CSV path above.
        z_stacks: std::range::Range {
            start: 0,
            end: db.get_nr_of_z_stacks() + 1,
        },
        t_stacks: std::range::Range {
            start: 0,
            end: db.get_nr_of_t_stacks() + 1,
        },
        image_rel_paths,
        columns,
        object_classes,
        with_list_view: !grouping.group_by_image,
        with_grouped_by_image_list: grouping.group_by_image,
        aggregations: grouping.aggregations,
        ..Default::default()
    };
    let mut no_progress = |_message: &str, _current: usize, _total: usize| {};
    let outcome = export
        .start_export(&db, &mut no_progress)
        .and_then(|_| {
            if let Some(parent) = args.out.parent()
                && !parent.as_os_str().is_empty()
            {
                std::fs::create_dir_all(parent).map_err(|e| {
                    InternalErrors::Internal(format!(
                        "could not create {}: {e}",
                        parent.display()
                    ))
                })?;
            }
            let produced = scratch_dir.join(if grouping.group_by_image {
                "grouped_by_image.xlsx"
            } else {
                "list.xlsx"
            });
            std::fs::rename(&produced, &args.out).map_err(|e| {
                InternalErrors::Internal(format!("could not move export output into place: {e}"))
            })
        });
    let _ = std::fs::remove_dir_all(&scratch_dir);
    outcome?;
    println!("Exported to {}", args.out.display());
    Ok(())
}

/// Whether `column` is something `get_grouped_by_image` can actually
/// resolve — mirrors the GUI's own `is_aggregable_column`
/// (results_state_controller.rs)/`is_aggregable`
/// (results_exporter.rs): identity columns and per-channel intensity aren't
/// resolvable to the single per-object SQL expression that function needs.
fn is_groupable_column(column: &Column) -> bool {
    !matches!(
        column,
        Column::ObjectId
            | Column::ImageName
            | Column::ObjectClass
            | Column::IntensityAvg(_)
            | Column::IntensitySum(_)
            | Column::IntensityMin(_)
            | Column::IntensityMax(_)
    )
}

fn empty_database_result() -> DatabaseResult {
    DatabaseResult {
        column_names: Vec::new(),
        row_names: Vec::new(),
        rows: Vec::new(),
        min: 0.0,
        max: 0.0,
        source_object_count: 0,
        row_locations: Vec::new(),
    }
}

/// Walks every page of `get_object_list` for `base` (its own `page` is
/// overwritten each iteration), concatenating them — mirrors
/// `results_exporter.rs`'s own `fetch_all_list_rows`, which is private to
/// `evanalyzer_app`, so duplicated here against the same public API.
fn fetch_all_objects(
    db: &ResultsGenerator,
    base: &ListFilter,
) -> Result<DatabaseResult, InternalErrors> {
    const PAGE_SIZE: i32 = 20_000;
    let mut merged = empty_database_result();
    let mut cursor: Option<String> = None;
    let mut first_page = true;

    loop {
        let filter = ListFilter {
            page: Pagination {
                limit: PAGE_SIZE,
                after: cursor.take(),
            },
            ..base.clone()
        };
        let mut page = db.get_object_list(&filter)?;
        let is_last_page = page.source_object_count < PAGE_SIZE as usize;
        cursor = page.row_names.last().cloned();

        if first_page {
            merged.column_names = std::mem::take(&mut page.column_names);
            first_page = false;
        }
        merged.row_names.extend(page.row_names);
        merged.rows.extend(page.rows);
        merged.source_object_count += page.source_object_count;

        if is_last_page {
            break;
        }
    }
    Ok(merged)
}

/// Same walk as `fetch_all_objects`, over `get_grouped_by_image` instead —
/// mirrors `results_exporter.rs`'s private `fetch_all_grouped_by_image_rows`.
fn fetch_all_grouped_by_image(
    db: &ResultsGenerator,
    base: &GroupedByImageFilter,
) -> Result<DatabaseResult, InternalErrors> {
    const PAGE_SIZE: i32 = 20_000;
    let mut merged = empty_database_result();
    let mut cursor: Option<String> = None;
    let mut first_page = true;

    loop {
        let filter = GroupedByImageFilter {
            page: Pagination {
                limit: PAGE_SIZE,
                after: cursor.take(),
            },
            ..base.clone()
        };
        let mut page = db.get_grouped_by_image(&filter)?;
        let is_last_page = page.source_object_count < PAGE_SIZE as usize;
        cursor = page.row_names.last().cloned();

        if first_page {
            merged.column_names = std::mem::take(&mut page.column_names);
            first_page = false;
        }
        merged.row_names.extend(page.row_names);
        merged.rows.extend(page.rows);
        merged.source_object_count += page.source_object_count;

        if is_last_page {
            break;
        }
    }
    Ok(merged)
}

fn write_csv(result: &DatabaseResult, path: &Path) -> Result<(), InternalErrors> {
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        std::fs::create_dir_all(parent).map_err(|e| {
            InternalErrors::Internal(format!("could not create {}: {e}", parent.display()))
        })?;
    }
    let file = std::fs::File::create(path)
        .map_err(|e| InternalErrors::Internal(format!("could not create {}: {e}", path.display())))?;
    let mut out = std::io::BufWriter::new(file);
    let write_err = |e: std::io::Error| InternalErrors::Internal(format!("could not write {}: {e}", path.display()));

    write_csv_row(&mut out, &result.column_names).map_err(write_err)?;
    for row in &result.rows {
        let cells: Vec<String> = row.iter().map(cell_text).collect();
        write_csv_row(&mut out, &cells).map_err(write_err)?;
    }
    Ok(())
}

fn write_csv_row(out: &mut impl std::io::Write, fields: &[String]) -> std::io::Result<()> {
    let line: Vec<String> = fields.iter().map(|f| csv_escape(f)).collect();
    writeln!(out, "{}", line.join(","))
}

fn csv_escape(field: &str) -> String {
    if field.contains(',') || field.contains('"') || field.contains('\n') || field.contains('\r') {
        format!("\"{}\"", field.replace('"', "\"\""))
    } else {
        field.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::args::{FilterArgs, GroupArgs};
    use crate::commands::test_support::TempResultsDb;

    #[test]
    fn export_table_writes_a_csv_file_with_the_expected_rows() {
        let db = TempResultsDb::seeded();
        let out_dir = tempfile::tempdir().expect("tempdir");
        let out = out_dir.path().join("out.csv");

        export_table(
            TableExportArgs {
                db: db.path.clone(),
                out: out.clone(),
                filter: FilterArgs::default(),
                group: GroupArgs::default(),
            },
            true,
        )
        .expect("csv export should succeed");

        let content = std::fs::read_to_string(&out).expect("read csv back");
        let mut lines = content.lines();
        let header = lines.next().expect("header row present");
        assert!(header.contains("Class"), "header: {header}");
        let body: Vec<&str> = lines.collect();
        assert_eq!(body.len(), 2, "expected 2 data rows, got: {body:?}");
        assert!(content.contains("ClassA"));
        assert!(content.contains("ClassB"));
    }

    #[test]
    fn export_table_writes_an_xlsx_file_with_the_expected_rows() {
        let db = TempResultsDb::seeded();
        let out_dir = tempfile::tempdir().expect("tempdir");
        let out = out_dir.path().join("out.xlsx");

        export_table(
            TableExportArgs {
                db: db.path.clone(),
                out: out.clone(),
                filter: FilterArgs::default(),
                group: GroupArgs::default(),
            },
            false,
        )
        .expect("xlsx export should succeed");

        let bytes = std::fs::read(&out).expect("read xlsx back");
        // XLSX files are zip archives - "PK\x03\x04" is the local-file-header magic.
        assert!(bytes.len() > 4, "xlsx file is too small: {} bytes", bytes.len());
        assert_eq!(&bytes[..4], b"PK\x03\x04", "not a zip/xlsx file");
    }

    #[test]
    fn export_table_grouped_by_image_writes_a_csv_file() {
        let db = TempResultsDb::seeded();
        let out_dir = tempfile::tempdir().expect("tempdir");
        let out = out_dir.path().join("grouped.csv");

        export_table(
            TableExportArgs {
                db: db.path.clone(),
                out: out.clone(),
                filter: FilterArgs::default(),
                group: GroupArgs {
                    group_by: Some(crate::args::GroupByKind::Image),
                    ..Default::default()
                },
            },
            true,
        )
        .expect("grouped csv export should succeed");

        let content = std::fs::read_to_string(&out).expect("read csv back");
        assert!(content.lines().count() >= 2, "expected a header and at least one data row");
    }

    #[test]
    fn export_table_rejects_unsupported_group_by_folder() {
        let db = TempResultsDb::seeded();
        let out_dir = tempfile::tempdir().expect("tempdir");
        let out = out_dir.path().join("out.csv");

        let result = export_table(
            TableExportArgs {
                db: db.path.clone(),
                out,
                filter: FilterArgs::default(),
                group: GroupArgs {
                    group_by: Some(crate::args::GroupByKind::Folder),
                    ..Default::default()
                },
            },
            true,
        );

        assert!(result.is_err());
    }

    #[test]
    fn csv_escape_quotes_fields_containing_commas_or_quotes() {
        assert_eq!(csv_escape("plain"), "plain");
        assert_eq!(csv_escape("a,b"), "\"a,b\"");
        assert_eq!(csv_escape("a\"b"), "\"a\"\"b\"");
    }
}
