use crate::args::{ExportArgs, ExportCommand, ParquetExportArgs, TableExportArgs};
use crate::commands::common::{resolve_grouping, resolve_image_rel_paths, resolve_object_classes};
use evanalyzer_app::result::{Column, ExportFormat, ResultExport, ResultsGenerator};
use evanalyzer_cfg::core_types::InternalErrors;
use std::sync::atomic::AtomicBool;

pub fn run(args: ExportArgs) -> Result<(), InternalErrors> {
    match args.command {
        ExportCommand::Csv(table) => export_table(table, ExportFormat::CSV),
        ExportCommand::Xlsx(table) => export_table(table, ExportFormat::XLSX),
        ExportCommand::Parquet(args) => export_parquet(args),
    }
}

/// `objects.parquet`: the raw `objects` table, every column, unfiltered
/// (see `ResultExport::export_as_parquet`) — unlike `export_table`, there's
/// no column/filter/grouping resolution to do first, so this is a much
/// thinner wrapper around `start_export`.
fn export_parquet(args: ParquetExportArgs) -> Result<(), InternalErrors> {
    let db = ResultsGenerator::open_database(args.db.clone())?;

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
        format: ExportFormat::Parquet,
        ..Default::default()
    };

    let mut no_progress = |_message: &str, _current: usize, _total: usize| {};
    let cancel = AtomicBool::new(false);
    let outcome = export.start_export(&db, &cancel, &mut no_progress).and_then(|_| {
        if let Some(parent) = args.out.parent()
            && !parent.as_os_str().is_empty()
        {
            std::fs::create_dir_all(parent).map_err(|e| {
                InternalErrors::Internal(format!("could not create {}: {e}", parent.display()))
            })?;
        }
        std::fs::rename(scratch_dir.join("objects.parquet"), &args.out).map_err(|e| {
            InternalErrors::Internal(format!("could not move export output into place: {e}"))
        })
    });
    let _ = std::fs::remove_dir_all(&scratch_dir);
    outcome?;
    println!("Exported to {}", args.out.display());
    Ok(())
}

/// Builds a `ResultExport` from `args` and runs it — all the actual
/// List/Grouped-by-Image fetching and CSV/XLSX writing logic lives in
/// `evanalyzer_app`'s `ResultExport::start_export`, shared with the GUI's
/// export dialog, so the CLI and the GUI can never produce different output
/// for the same settings. This function's only job is translating CLI args
/// into that shared config and moving the one file it produces into place.
fn export_table(args: TableExportArgs, format: ExportFormat) -> Result<(), InternalErrors> {
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

    // `ResultExport` always writes into a directory it owns, under its own
    // fixed filenames (`list.{ext}`/`grouped_by_image.{ext}`) — this writes
    // into a private scratch directory first (which would otherwise collide
    // with anything already sitting next to `--out`) and moves the one file
    // it produces to `--out`.
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
        format,
        // `get_nr_of_z_stacks`/`get_nr_of_t_stacks` return the *max* stack
        // index (not a count), so the exclusive upper bound of a range
        // covering every plane is that value plus one.
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

    // `export_table` is only ever called with CSV/XLSX from `run()` above -
    // Parquet goes through `export_parquet` instead, since it needs none of
    // the column/filter/grouping resolution this function does.
    let extension = match format {
        ExportFormat::CSV => "csv",
        ExportFormat::XLSX => "xlsx",
        ExportFormat::Parquet => {
            return Err(InternalErrors::Internal(
                "export_table doesn't support Parquet — use export_parquet".to_string(),
            ));
        }
    };
    let mut no_progress = |_message: &str, _current: usize, _total: usize| {};
    let cancel = AtomicBool::new(false);
    let outcome = export.start_export(&db, &cancel, &mut no_progress).and_then(|_| {
        if let Some(parent) = args.out.parent()
            && !parent.as_os_str().is_empty()
        {
            std::fs::create_dir_all(parent).map_err(|e| {
                InternalErrors::Internal(format!("could not create {}: {e}", parent.display()))
            })?;
        }
        let produced = scratch_dir.join(if grouping.group_by_image {
            format!("grouped_by_image.{extension}")
        } else {
            format!("list.{extension}")
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::args::{FilterArgs, GroupArgs};
    use crate::commands::test_support::TempResultsDb;

    #[test]
    fn export_parquet_writes_a_valid_parquet_file() {
        let db = TempResultsDb::seeded();
        let out_dir = tempfile::tempdir().expect("tempdir");
        let out = out_dir.path().join("out.parquet");

        export_parquet(ParquetExportArgs {
            db: db.path.clone(),
            out: out.clone(),
        })
        .expect("parquet export should succeed");

        let bytes = std::fs::read(&out).expect("read parquet back");
        // Every Parquet file starts and ends with the 4-byte "PAR1" magic.
        assert!(
            bytes.len() > 8,
            "parquet file is too small: {} bytes",
            bytes.len()
        );
        assert_eq!(&bytes[..4], b"PAR1", "missing leading PAR1 magic");
        assert_eq!(
            &bytes[bytes.len() - 4..],
            b"PAR1",
            "missing trailing PAR1 magic"
        );
    }

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
            ExportFormat::CSV,
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
            ExportFormat::XLSX,
        )
        .expect("xlsx export should succeed");

        let bytes = std::fs::read(&out).expect("read xlsx back");
        // XLSX files are zip archives - "PK\x03\x04" is the local-file-header magic.
        assert!(
            bytes.len() > 4,
            "xlsx file is too small: {} bytes",
            bytes.len()
        );
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
            ExportFormat::CSV,
        )
        .expect("grouped csv export should succeed");

        let content = std::fs::read_to_string(&out).expect("read csv back");
        assert!(
            content.lines().count() >= 2,
            "expected a header and at least one data row"
        );
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
            ExportFormat::CSV,
        );

        assert!(result.is_err());
    }

    #[test]
    fn export_table_writes_one_intensity_column_group_per_real_image_channel() {
        let db = TempResultsDb::with_channels(3);
        let out_dir = tempfile::tempdir().expect("tempdir");
        let out = out_dir.path().join("out.csv");

        export_table(
            TableExportArgs {
                db: db.path.clone(),
                out: out.clone(),
                filter: FilterArgs::default(),
                group: GroupArgs::default(),
            },
            ExportFormat::CSV,
        )
        .expect("csv export should succeed");

        let content = std::fs::read_to_string(&out).expect("read csv back");
        let header = content.lines().next().expect("header row present");
        for ch in 0..3 {
            assert!(
                header.contains(&format!("Avg Intensity (Ch {ch})")),
                "header missing channel {ch}'s intensity column: {header}"
            );
        }
        assert!(
            !header.contains("Ch 3)"),
            "header should not report a 4th channel that was never measured: {header}"
        );
    }
}
