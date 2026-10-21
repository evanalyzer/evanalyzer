use crate::args::{ExportArgs, ExportCommand, ParquetExportArgs, TableExportArgs};
use crate::commands::common::{
    ResolvedGrouping, resolve_grouping, resolve_image_rel_paths, resolve_object_classes,
};
use evanalyzer_app::backends::Backend;
use evanalyzer_app::results::Column;
use evanalyzer_app::results::ExportFormat;
use evanalyzer_app::results::ResultExport;
use evanalyzer_cfg::core_types::InternalErrors;
use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicBool;

/// The export runs where the backend runs (the server, remotely), so `db`
/// and `--out` are paths on that machine.
pub fn run(args: ExportArgs, backend: &dyn Backend) -> Result<(), InternalErrors> {
    match args.command {
        ExportCommand::Csv(table) => export_table(backend, table, ExportFormat::CSV),
        ExportCommand::Xlsx(table) => export_table(backend, table, ExportFormat::XLSX),
        ExportCommand::Parquet(args) => export_parquet(backend, args),
    }
}

/// A private folder next to `out` for `ResultExport` to write into (it
/// always writes its own fixed filenames into a directory it owns, which
/// would otherwise collide with anything already next to `--out`). Next to
/// `out` rather than in the temp folder so the final move never crosses
/// file systems.
fn scratch_dir_next_to(out: &Path) -> PathBuf {
    let parent = out
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    parent.join(format!(
        ".evanalyzer_cli_export_{}_{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos(),
    ))
}

/// Moves the one file an export produced to `out`, then removes the scratch
/// folder - both on the backend's machine.
fn move_into_place(
    backend: &dyn Backend,
    produced: &Path,
    out: &Path,
    scratch_dir: &Path,
    outcome: Result<(), InternalErrors>,
) -> Result<(), InternalErrors> {
    let files = backend.files();
    let outcome = outcome.and_then(|_| {
        files.rename(produced, out).map_err(|e| {
            InternalErrors::Internal(format!("could not move export output into place: {e}"))
        })
    });
    let _ = files.remove_all(scratch_dir);
    outcome
}

/// `objects.parquet`: the raw `objects` table, every column, unfiltered
/// (see `ResultExport::export_as_parquet`) — unlike `export_table`, there's
/// no column/filter/grouping resolution to do first, so this is a much
/// thinner wrapper around `start_export`.
fn export_parquet(backend: &dyn Backend, args: ParquetExportArgs) -> Result<(), InternalErrors> {
    let db = backend.open_results(&args.db)?;

    let scratch_dir = scratch_dir_next_to(&args.out);
    let export = ResultExport {
        output_dir: scratch_dir.clone(),
        format: ExportFormat::Parquet,
        ..Default::default()
    };

    let mut no_progress = |_message: &str, _current: usize, _total: usize| {};
    let cancel = AtomicBool::new(false);
    let outcome = db.export(&export, &cancel, &mut no_progress);
    move_into_place(
        backend,
        &scratch_dir.join("objects.parquet"),
        &args.out,
        &scratch_dir,
        outcome,
    )?;
    println!("Exported to {}", args.out.display());
    Ok(())
}

/// Builds a `ResultExport` from `args` and runs it — all the actual
/// List/Grouped-by-Image fetching and CSV/XLSX writing logic lives in
/// `evanalyzer_app`'s `ResultExport::start_export`, shared with the GUI's
/// export dialog, so the CLI and the GUI can never produce different output
/// for the same settings. This function's only job is translating CLI args
/// into that shared config and moving the one file it produces into place.
fn export_table(
    backend: &dyn Backend,
    args: TableExportArgs,
    format: ExportFormat,
) -> Result<(), InternalErrors> {
    let db = backend.open_results(&args.db)?;
    let db = db.as_ref();
    if args.filter.colocalized.is_some() {
        return Err(InternalErrors::InvalidArgument(
            "--colocalized isn't supported by the current results backend".to_string(),
        ));
    }
    let grouping = resolve_grouping(&args.group)?;
    let image_rel_paths = resolve_image_rel_paths(db, &args.filter.images)?;
    let object_classes = resolve_object_classes(db, &args.filter.classes)?;
    let columns: Vec<Column> = db
        .get_available_columns()?
        .into_iter()
        .map(|entry| entry.key)
        .collect();

    // `ResultExport` writes `list.{ext}`/`grouped_by_image.{ext}`/
    // `time_course*.{ext}` into a directory it owns - see
    // `scratch_dir_next_to`.
    let scratch_dir = scratch_dir_next_to(&args.out);
    let mut export = ResultExport {
        output_dir: scratch_dir.clone(),
        format,
        // `None`: every plane - the export adds "T"/"Z" columns when the
        // results have several.
        z_stacks: args.filter.z_stacks,
        t_stacks: args.filter.t_stacks,
        image_rel_paths,
        columns,
        object_classes,
        ..Default::default()
    };
    let document = match grouping {
        ResolvedGrouping::List => {
            export.with_list_view = true;
            "list"
        }
        ResolvedGrouping::PerImage(aggregations) => {
            export.with_grouped_by_image_list = true;
            export.aggregations = aggregations;
            "grouped_by_image"
        }
        ResolvedGrouping::TimeCourse { grouping, settings } => {
            export.with_time_course = true;
            export.grouping = grouping;
            export.time_course = settings;
            "time_course"
        }
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
    let outcome = db.export(&export, &cancel, &mut no_progress);
    // A wide time course in CSV: a file per value next to `--out`
    // (`<name>_<value>.csv`), the long table (if any) at `--out`.
    if document == "time_course" && matches!(format, ExportFormat::CSV) {
        return move_time_course_csv_into_place(backend, &args.out, &scratch_dir, outcome);
    }
    let produced = scratch_dir.join(format!("{document}.{extension}"));
    move_into_place(backend, &produced, &args.out, &scratch_dir, outcome)?;
    println!("Exported to {}", args.out.display());
    Ok(())
}

/// The time course's CSV files to their places: `time_course.csv` to `out`,
/// each `time_course_<value>.csv` next to it as `<out's name>_<value>.csv` -
/// then the scratch folder goes.
fn move_time_course_csv_into_place(
    backend: &dyn Backend,
    out: &Path,
    scratch_dir: &Path,
    outcome: Result<(), InternalErrors>,
) -> Result<(), InternalErrors> {
    let files = backend.files();
    let stem = out
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| "time_course".to_string());
    let moved = outcome.and_then(|_| {
        let mut moved = Vec::new();
        for entry in files.list_dir(scratch_dir)? {
            let to = match entry.name.as_str() {
                "time_course.csv" => out.to_path_buf(),
                name => match name.strip_prefix("time_course_") {
                    Some(value) => out.with_file_name(format!("{stem}_{value}")),
                    None => continue,
                },
            };
            files.rename(&entry.path, &to).map_err(|e| {
                InternalErrors::Internal(format!("could not move export output into place: {e}"))
            })?;
            moved.push(to);
        }
        Ok(moved)
    });
    let _ = files.remove_all(scratch_dir);
    let mut moved = moved?;
    moved.sort();
    for path in moved {
        println!("Exported to {}", path.display());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use evanalyzer_app::backends::local::LocalBackend;

    use super::*;

    fn export_parquet(args: ParquetExportArgs) -> Result<(), InternalErrors> {
        super::export_parquet(&LocalBackend::default(), args)
    }

    fn export_table(args: TableExportArgs, format: ExportFormat) -> Result<(), InternalErrors> {
        super::export_table(&LocalBackend::default(), args, format)
    }
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
    fn export_table_rejects_group_by_group_without_its_regex() {
        let db = TempResultsDb::seeded();
        let out_dir = tempfile::tempdir().expect("tempdir");
        let out = out_dir.path().join("out.csv");

        let result = export_table(
            TableExportArgs {
                db: db.path.clone(),
                out,
                filter: FilterArgs::default(),
                group: GroupArgs {
                    group_by: Some(crate::args::GroupByKind::Group),
                    ..Default::default()
                },
            },
            ExportFormat::CSV,
        );

        assert!(result.is_err());
    }

    fn time_course(group: GroupArgs, out: &std::path::Path) -> String {
        let db = TempResultsDb::seeded();
        export_table(
            TableExportArgs {
                db: db.path.clone(),
                out: out.to_path_buf(),
                filter: FilterArgs::default(),
                group,
            },
            ExportFormat::CSV,
        )
        .expect("time course export");
        std::fs::read_to_string(out).unwrap_or_default()
    }

    /// `--group-by class`: the time course's long table at `--out` - a row
    /// per plane and class, mean and SD.
    #[test]
    fn export_table_group_by_class_writes_the_time_course() {
        let out_dir = tempfile::tempdir().expect("tempdir");
        let out = out_dir.path().join("course.csv");
        let csv = time_course(
            GroupArgs {
                group_by: Some(crate::args::GroupByKind::Class),
                ..Default::default()
            },
            &out,
        );
        let header = csv.lines().next().unwrap();
        assert!(header.starts_with("T,Z,Class,N images,N objects,"), "{header}");
        assert!(header.contains("Area [px] mean,Area [px] SD"), "{header}");
        assert_eq!(csv.lines().count(), 1 + 2 * 2, "2 planes x 2 classes");
        let names: Vec<_> = std::fs::read_dir(out_dir.path())
            .unwrap()
            .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
            .collect();
        assert_eq!(names, ["course.csv"], "only --out, the scratch folder is gone");
    }

    /// `--layout both` in CSV: the long table at `--out`, a file per value
    /// next to it.
    #[test]
    fn export_table_time_course_wide_csv_goes_next_to_out() {
        let out_dir = tempfile::tempdir().expect("tempdir");
        let out = out_dir.path().join("course.csv");
        time_course(
            GroupArgs {
                group_by: Some(crate::args::GroupByKind::Well),
                agg: vec![crate::args::AggKind::Avg],
                layout: crate::args::LayoutKind::Both,
                ..Default::default()
            },
            &out,
        );
        assert!(out.exists());
        assert!(out_dir.path().join("course_Area__px__mean.csv").exists());
        let leftovers: Vec<_> = std::fs::read_dir(out_dir.path())
            .unwrap()
            .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
            .filter(|name| name.starts_with(".evanalyzer_cli_export"))
            .collect();
        assert!(leftovers.is_empty(), "{leftovers:?}");
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
