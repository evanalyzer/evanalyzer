//! # export
//!
//! `cli results export`: the flags as one data export, run by the same
//! runner as the project's exports (`ResultsSource::run_export`) - the
//! command line, the app and a project export can never write a table
//! differently. `--out` is one file; the runner's file goes there.
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-06-26
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.

use crate::args::{
    AggKind, ExportArgs, ExportCommand, GroupByKind, LayoutKind, LengthUnitKind, NormalizeKind,
    ParquetExportArgs, SpeedUnitKind, TableExportArgs,
};
use crate::commands::common::{resolve_image_rel_paths, resolve_object_classes};
use evanalyzer_app::backend::Backend;
use evanalyzer_app::exports::{EntryStatus, ExportRun, ScriptPermission, entry_from_view};
use evanalyzer_app::project::load_project;
use evanalyzer_app::results::ResultsSource;
use evanalyzer_cfg::core_types::{InternalErrors, ObjectClass};
use evanalyzer_cfg::settings::export_settings::{
    DataExport, DataOutput, ExportAggregation, ExportEntry, ExportSettings, ImageScope, LengthUnit,
    PlaneRange, SpeedUnit, TableFileFormat, TimeCourseLineKind, TimeCourseOptions,
    TimeCourseTableLayout, TrackOptions,
};
use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicBool;

/// The export runs where the backend runs (the server, remotely), so `db`
/// and `--out` are paths on that machine.
pub fn run(args: ExportArgs, backend: &dyn Backend) -> Result<(), InternalErrors> {
    match args.command {
        ExportCommand::Csv(table) => export_table(backend, table, TableFileFormat::Csv),
        ExportCommand::Xlsx(table) => export_table(backend, table, TableFileFormat::Xlsx),
        ExportCommand::Parquet(args) => export_parquet(backend, args),
    }
}

/// A private folder next to `out` for the export to write into (its own
/// file names, which would otherwise collide with anything next to
/// `--out`). Next to `out` rather than in the temp folder so the final move
/// never crosses file systems.
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

/// Moves the files an export wrote into `scratch_dir` to their places -
/// `main` to `out`, each other `<main's stem>_<rest>` next to it as
/// `<out's stem>_<rest>` - then removes the scratch folder, all on the
/// backend's machine. The files where they went, sorted.
fn move_into_place(
    backend: &dyn Backend,
    scratch_dir: &Path,
    main: &str,
    out: &Path,
    outcome: Result<(), InternalErrors>,
) -> Result<Vec<PathBuf>, InternalErrors> {
    let files = backend.files();
    let stem = out
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| main.to_string());
    let moved = outcome.and_then(|_| {
        let mut moved = Vec::new();
        for entry in files.list_dir(scratch_dir)? {
            let name = entry.name.as_str();
            let file_stem = Path::new(name)
                .file_stem()
                .map(|s| s.to_string_lossy().into_owned())
                .unwrap_or_default();
            let to = if file_stem == main {
                out.to_path_buf()
            } else if let Some(rest) = name.strip_prefix(&format!("{main}_")) {
                out.with_file_name(format!("{stem}_{rest}"))
            } else {
                continue;
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
    Ok(moved)
}

/// The raw `objects` table (and `measurements`, where there are any) as
/// Parquet: `objects` to `--out`, `measurements` next to it.
fn export_parquet(backend: &dyn Backend, args: ParquetExportArgs) -> Result<(), InternalErrors> {
    let db = backend.open_results(&args.db)?;
    let scratch_dir = scratch_dir_next_to(&args.out);
    let mut no_progress = |_message: &str, _current: usize, _total: usize| {};
    let cancel = AtomicBool::new(false);
    let outcome = backend
        .files()
        .create_dir_all(&scratch_dir)
        .and_then(|_| db.export_parquet(&scratch_dir, &cancel, &mut no_progress));
    for path in move_into_place(backend, &scratch_dir, "objects", &args.out, outcome)? {
        println!("Exported to {}", path.display());
    }
    Ok(())
}

/// Builds the data export `args` ask for and runs it, then moves its file
/// to `--out` (a wide time course's files per value next to it).
fn export_table(
    backend: &dyn Backend,
    args: TableExportArgs,
    format: TableFileFormat,
) -> Result<(), InternalErrors> {
    if args.tracks.tracks && args.group.group_by.is_some() {
        return Err(InternalErrors::InvalidArgument(
            "--tracks is a row per track: it can't be grouped (--group-by) too".to_string(),
        ));
    }
    let db = backend.open_results(&args.db)?;
    let db = db.as_ref();
    let data = data_export(backend, db, &args, format)?;
    let output = data.outputs[0];
    let export = ExportSettings {
        id: 0,
        name: "cli".into(),
        prefix: String::new(),
        length_unit: match args.length_unit {
            LengthUnitKind::Um => LengthUnit::Um,
            LengthUnitKind::Nm => LengthUnit::Nm,
        },
        output: Default::default(),
        run_after_analysis: false,
        table_format: evanalyzer_cfg::settings::export_settings::CURRENT_TABLE_FORMAT,
        entries: vec![ExportEntry::Data(data)],
    };
    let scratch_dir = scratch_dir_next_to(&args.out);
    let run = ExportRun {
        export,
        project_dir: scratch_dir.clone(),
        results_dir: args.db.parent().map(Path::to_path_buf).unwrap_or_default(),
        output: Some(scratch_dir.clone()),
        scripts: ScriptPermission::Skip,
        interpreters: Default::default(),
        drawing: None,
    };
    let mut no_progress = |_message: &str, _current: usize, _total: usize| {};
    let cancel = AtomicBool::new(false);
    let outcome = db
        .run_export(&run, &cancel, &mut no_progress)
        .and_then(|report| {
            let Some(entry) = report.entries.first() else {
                return Err(InternalErrors::Internal("the export wrote nothing".into()));
            };
            for note in &entry.notes {
                eprintln!("Note: {note}");
            }
            match &entry.status {
                EntryStatus::Written => Ok(()),
                EntryStatus::Skipped(why) | EntryStatus::Failed(why) => {
                    Err(InternalErrors::Internal(why.clone()))
                }
            }
        });
    let main = evanalyzer_app::exports::naming::rows_name(table_rows(output));
    for path in move_into_place(backend, &scratch_dir, main, &args.out, outcome)? {
        println!("Exported to {}", path.display());
    }
    Ok(())
}

/// What `args` ask for as one data export of one output: `--view`'s, if
/// given, with the flags winning; else the flags' - every column these
/// results have.
fn data_export(
    backend: &dyn Backend,
    db: &dyn ResultsSource,
    args: &TableExportArgs,
    format: TableFileFormat,
) -> Result<DataExport, InternalErrors> {
    let mut data = match (&args.project, &args.view) {
        (Some(project), Some(name)) => view_export(backend, project, name)?,
        _ => {
            let classes = db.get_object_classes()?;
            let mut data: DataExport = serde_json::from_value(serde_json::json!({}))
                .map_err(|e| InternalErrors::Internal(e.to_string()))?;
            data.outputs = vec![DataOutput::Objects];
            data.columns = Some(
                db.get_available_columns()?
                    .iter()
                    .filter(|entry| !evanalyzer_app::exports::is_fixed_column(&entry.key))
                    .map(|entry| entry.key.as_key(&classes))
                    .collect(),
            );
            data
        }
    };
    data.format = format;
    data.name = String::new();
    data.split = Default::default();
    if !args.filter.images.is_empty() {
        data.images = ImageScope::All;
        data.only_images = Some(resolve_image_rel_paths(db, &args.filter.images)?);
    }
    if !args.filter.classes.is_empty() {
        let ids = resolve_object_classes(db, &args.filter.classes)?
            .into_iter()
            .filter_map(|class| match class {
                ObjectClass::Global(id) => Some(id),
                _ => None,
            })
            .collect();
        data.classes = Some(ids);
    }
    let range = |r: std::range::Range<u32>| PlaneRange {
        from: r.start,
        to: r.end.saturating_sub(1),
    };
    if let Some(t) = args.filter.t_stacks {
        data.planes.t = Some(range(t));
    }
    if let Some(z) = args.filter.z_stacks {
        data.planes.z = Some(range(z));
    }
    let group = &args.group;
    let statistics = |default: &[ExportAggregation]| match group.agg.is_empty() {
        true => default.to_vec(),
        false => group.agg.iter().copied().map(aggregation).collect(),
    };
    if args.tracks.tracks {
        data.outputs = vec![DataOutput::Tracks];
        data.aggregations = statistics(&[ExportAggregation::Avg]);
        data.group_regex = group.group_regex.clone().filter(|r| !r.is_empty());
        data.tracks = Some(TrackOptions {
            min_points: args.tracks.min_points,
            from_start: args.tracks.from_start,
            no_edge: args.tracks.no_edge,
            speed_unit: match args.tracks.speed_unit {
                SpeedUnitKind::PxFrame => SpeedUnit::PxFrame,
                SpeedUnitKind::UmMin => SpeedUnit::UmMin,
                SpeedUnitKind::UmS => SpeedUnit::UmS,
            },
        });
    } else if args.measurements {
        data.outputs = vec![DataOutput::Measurements];
        data.columns = None;
    } else if let Some(kind) = group.group_by {
        let lines = match kind {
            GroupByKind::Image => {
                if group
                    .agg
                    .iter()
                    .any(|a| matches!(a, AggKind::Q1 | AggKind::Q3))
                {
                    return Err(InternalErrors::InvalidArgument(
                        "q1/q3 are only for a time course (--group-by class, well, group, \
                         folder, regex or track)"
                            .to_string(),
                    ));
                }
                data.outputs = vec![DataOutput::Images];
                data.aggregations = statistics(&[ExportAggregation::Avg]);
                return Ok(data);
            }
            GroupByKind::Class => TimeCourseLineKind::Class,
            GroupByKind::Well => TimeCourseLineKind::Well,
            GroupByKind::Folder => TimeCourseLineKind::Folder,
            GroupByKind::Track => TimeCourseLineKind::Track,
            GroupByKind::Group | GroupByKind::Regex => {
                let regex = group.group_regex.clone().filter(|r| !r.is_empty());
                if regex.is_none() {
                    return Err(InternalErrors::InvalidArgument(
                        "--group-by group / regex needs --group-regex (its first capture group \
                         is the group), or use --group-by folder"
                            .to_string(),
                    ));
                }
                data.group_regex = regex;
                TimeCourseLineKind::Well
            }
        };
        data.outputs = vec![DataOutput::TimeCourse];
        data.aggregations = statistics(&[ExportAggregation::Avg, ExportAggregation::Stddev]);
        data.time_course = Some(TimeCourseOptions {
            lines,
            layout: match group.layout {
                LayoutKind::Long => TimeCourseTableLayout::Long,
                LayoutKind::Wide => TimeCourseTableLayout::Wide,
                LayoutKind::Both => TimeCourseTableLayout::Both,
            },
            relative: matches!(group.normalize, Some(NormalizeKind::T0)),
            with_parents: group.with_parents,
        });
    }
    Ok(data)
}

/// The data export view `name` (case aside) of the project at `project`
/// writes - an error for no such view, or one that isn't one table.
fn view_export(
    backend: &dyn Backend,
    project: &Path,
    name: &str,
) -> Result<DataExport, InternalErrors> {
    let project = load_project(backend.files(), &project.to_path_buf())?;
    let views = &project.results_views.views;
    let Some(view) = views
        .iter()
        .find(|view| view.name.eq_ignore_ascii_case(name))
    else {
        let names: Vec<&str> = views.iter().map(|view| view.name.as_str()).collect();
        return Err(InternalErrors::InvalidArgument(format!(
            "no view \"{name}\" in the project - its views: {}",
            names.join(", ")
        )));
    };
    match entry_from_view(view) {
        Some(ExportEntry::Data(data)) if data.outputs.len() == 1 && is_table(data.outputs[0]) => {
            Ok(data)
        }
        _ => Err(InternalErrors::InvalidArgument(format!(
            "view \"{}\" is a {:?} view - --view exports views that are a table",
            view.name, view.kind
        ))),
    }
}

/// Whether `output` is a table (not a heatmap).
fn is_table(output: DataOutput) -> bool {
    !matches!(
        output,
        DataOutput::Plate | DataOutput::WellGrid | DataOutput::WellList | DataOutput::ImageHeatmap
    )
}

/// The rows of a table output.
fn table_rows(output: DataOutput) -> evanalyzer_app::exports::jobs::TableRows {
    use evanalyzer_app::exports::jobs::TableRows;
    match output {
        DataOutput::Images => TableRows::Image,
        DataOutput::Wells => TableRows::Well,
        DataOutput::Folders => TableRows::Folder,
        DataOutput::ColocPairs => TableRows::ColocPair,
        DataOutput::Measurements => TableRows::Measurement,
        DataOutput::Tracks => TableRows::Track,
        DataOutput::TimeCourse => TableRows::TimePoint,
        DataOutput::Profile => TableRows::ProfilePosition,
        _ => TableRows::Object,
    }
}

fn aggregation(kind: AggKind) -> ExportAggregation {
    match kind {
        AggKind::Min => ExportAggregation::Min,
        AggKind::Max => ExportAggregation::Max,
        AggKind::Avg => ExportAggregation::Avg,
        AggKind::Median => ExportAggregation::Median,
        AggKind::Stdev => ExportAggregation::Stddev,
        AggKind::Sum => ExportAggregation::Sum,
        AggKind::Q1 => ExportAggregation::Q1,
        AggKind::Q3 => ExportAggregation::Q3,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::args::{FilterArgs, GroupArgs, TrackArgs};
    use crate::commands::test_support::TempResultsDb;
    use evanalyzer_app::backend::local::LocalBackend;
    use evanalyzer_cfg::settings::project_settings::ResultsViewKind;

    fn export_parquet(args: ParquetExportArgs) -> Result<(), InternalErrors> {
        super::export_parquet(&LocalBackend::default(), args)
    }

    fn export_table(args: TableExportArgs, format: TableFileFormat) -> Result<(), InternalErrors> {
        super::export_table(&LocalBackend::default(), args, format)
    }

    fn table_args(db: &Path, out: &Path) -> TableExportArgs {
        TableExportArgs {
            db: db.to_path_buf(),
            out: out.to_path_buf(),
            filter: FilterArgs::default(),
            group: GroupArgs::default(),
            tracks: Default::default(),
            length_unit: Default::default(),
            measurements: false,
            project: None,
            view: None,
        }
    }

    /// The names in `dir`, sorted.
    fn names_in(dir: &Path) -> Vec<String> {
        let mut names: Vec<String> = std::fs::read_dir(dir)
            .unwrap()
            .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
            .collect();
        names.sort();
        names
    }

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
        assert!(bytes.len() > 8, "too small: {} bytes", bytes.len());
        assert_eq!(&bytes[..4], b"PAR1", "missing leading PAR1 magic");
        assert_eq!(
            &bytes[bytes.len() - 4..],
            b"PAR1",
            "missing trailing PAR1 magic"
        );
        assert_eq!(
            names_in(out_dir.path()),
            ["out.parquet"],
            "no scratch folder left"
        );
    }

    /// A row per object, every column these results have, in format 1.
    #[test]
    fn export_table_writes_a_csv_file_with_the_expected_rows() {
        let db = TempResultsDb::seeded();
        let out_dir = tempfile::tempdir().expect("tempdir");
        let out = out_dir.path().join("out.csv");

        export_table(table_args(&db.path, &out), TableFileFormat::Csv)
            .expect("csv export should succeed");

        let content = std::fs::read_to_string(&out).expect("read csv back");
        let mut lines = content.lines();
        let header = lines.next().expect("header row present");
        assert!(header.starts_with("image,"), "header: {header}");
        assert!(header.contains(",class,"), "header: {header}");
        assert!(header.contains("area_px"), "header: {header}");
        let body: Vec<&str> = lines.collect();
        assert_eq!(body.len(), 2, "expected 2 data rows, got: {body:?}");
        assert!(content.contains("ClassA"));
        assert!(content.contains("ClassB"));
        assert_eq!(
            names_in(out_dir.path()),
            ["out.csv"],
            "no scratch folder left"
        );
    }

    /// `--image` and `--class`: only those; an unknown name is an error.
    #[test]
    fn export_table_keeps_to_the_images_and_classes_asked_for() {
        let db = TempResultsDb::seeded();
        let out_dir = tempfile::tempdir().expect("tempdir");
        let out = out_dir.path().join("out.csv");
        let mut args = table_args(&db.path, &out);
        args.filter.classes = vec!["ClassB".into()];
        export_table(args, TableFileFormat::Csv).expect("export");
        let content = std::fs::read_to_string(&out).unwrap();
        assert_eq!(content.lines().count(), 2, "{content}");
        assert!(!content.contains("ClassA"), "{content}");

        let mut args = table_args(&db.path, &out);
        args.filter.images = vec!["nope.tif".into()];
        assert!(export_table(args, TableFileFormat::Csv).is_err());
    }

    /// `--length-unit nm`: lengths in nm, named so.
    #[test]
    fn export_table_writes_lengths_in_the_unit_asked_for() {
        let db = TempResultsDb::seeded();
        let out_dir = tempfile::tempdir().expect("tempdir");
        let header = |unit: LengthUnitKind| {
            let out = out_dir.path().join("out.csv");
            let mut args = table_args(&db.path, &out);
            args.length_unit = unit;
            export_table(args, TableFileFormat::Csv).expect("export");
            std::fs::read_to_string(&out)
                .unwrap()
                .lines()
                .next()
                .unwrap()
                .to_string()
        };
        assert!(header(LengthUnitKind::Um).contains("area_um2"));
        assert!(header(LengthUnitKind::Nm).contains("area_nm2"));
    }

    /// A project with a List view "QC" (two columns, a row per image) and
    /// a Plate view.
    fn project_with_views() -> crate::commands::test_support::TempProjectFile {
        use evanalyzer_cfg::settings::project_settings::{ProjectSettings, ResultsView};
        let mut settings = ProjectSettings::default();
        settings.results_views.views.push(ResultsView {
            id: 9,
            name: "QC".into(),
            kind: ResultsViewKind::List,
            split: false,
            settings: serde_json::json!({"list": {
                "columns": ["object_class_name", "area_px", "measurement:gone_01/noise"],
                "rowsPerImage": true,
                "aggregations": ["Max"]
            }}),
        });
        settings.results_views.views.push(ResultsView {
            id: 10,
            name: "Plate".into(),
            kind: ResultsViewKind::Plate,
            split: false,
            settings: serde_json::json!({}),
        });
        crate::commands::test_support::TempProjectFile::new(&settings)
    }

    fn view_export(view: &str, project: &Path, out: &Path) -> Result<(), InternalErrors> {
        let db = TempResultsDb::seeded();
        let mut args = table_args(&db.path, out);
        args.project = Some(project.to_path_buf());
        args.view = Some(view.into());
        export_table(args, TableFileFormat::Csv)
    }

    /// `--view`: as the List view shows it - its columns (one these results
    /// lack left out), a row per image with its aggregation.
    #[test]
    fn export_as_a_list_view_of_the_project() {
        let project = project_with_views();
        let out_dir = tempfile::tempdir().expect("tempdir");
        let out = out_dir.path().join("qc.csv");
        view_export("qc", &project.path, &out).expect("export as the view");
        let content = std::fs::read_to_string(&out).expect("read csv back");
        let header = content.lines().next().unwrap();
        assert!(header.contains("area_px_max"), "{header}");
        assert!(
            !header.contains("circularity"),
            "only the view's columns: {header}"
        );

        let err = view_export("Nope", &project.path, &out)
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("no view \"Nope\"") && err.contains("QC"),
            "{err}"
        );
        let err = view_export("Plate", &project.path, &out)
            .unwrap_err()
            .to_string();
        assert!(err.contains("views that are a table"), "{err}");
    }

    /// `--tracks`: a row per track (here each object its own, of one time
    /// point); `--min-points 2` leaves them out.
    #[test]
    fn export_tracks_writes_a_row_per_track() {
        let db = TempResultsDb::seeded();
        duckdb::Connection::open(&db.path)
            .unwrap()
            .execute_batch("UPDATE objects SET track_id = object_class_id[1];")
            .unwrap();
        let out_dir = tempfile::tempdir().expect("tempdir");
        let export = |min_points: u32| {
            let out = out_dir.path().join(format!("tracks_{min_points}.csv"));
            let mut args = table_args(&db.path, &out);
            args.tracks = TrackArgs {
                tracks: true,
                min_points,
                ..Default::default()
            };
            export_table(args, TableFileFormat::Csv).expect("tracks export");
            std::fs::read_to_string(&out).expect("read csv back")
        };
        let all = export(1);
        let lines: Vec<&str> = all.lines().collect();
        assert!(lines[0].contains("track_id"), "{}", lines[0]);
        assert_eq!(lines.len(), 3, "{all}");
        assert_eq!(export(2).lines().count(), 1, "the header only");
    }

    #[test]
    fn export_table_writes_an_xlsx_file_with_the_expected_rows() {
        let db = TempResultsDb::seeded();
        let out_dir = tempfile::tempdir().expect("tempdir");
        let out = out_dir.path().join("out.xlsx");

        export_table(table_args(&db.path, &out), TableFileFormat::Xlsx)
            .expect("xlsx export should succeed");

        use calamine::Reader;
        let mut workbook: calamine::Xlsx<_> = calamine::open_workbook(&out).expect("an xlsx");
        let range = workbook.worksheet_range("Table").expect("Table sheet");
        assert_eq!(range.rows().count(), 1 + 2, "a header, a row per object");
    }

    #[test]
    fn export_table_grouped_by_image_writes_a_csv_file() {
        let db = TempResultsDb::seeded();
        let out_dir = tempfile::tempdir().expect("tempdir");
        let out = out_dir.path().join("grouped.csv");
        let mut args = table_args(&db.path, &out);
        args.group.group_by = Some(GroupByKind::Image);
        export_table(args, TableFileFormat::Csv).expect("grouped csv export should succeed");

        let content = std::fs::read_to_string(&out).expect("read csv back");
        let header = content.lines().next().unwrap();
        assert!(header.contains("area_px_avg"), "{header}");
        assert!(header.ends_with("marker,note"), "{header}");
        assert!(content.lines().count() >= 2, "a header and a row at least");
    }

    /// What the grouping flags ask for, as the data export.
    fn data_of(group: GroupArgs) -> Result<DataExport, InternalErrors> {
        let db = TempResultsDb::seeded();
        let backend = LocalBackend::default();
        let results = backend.open_results(&db.path)?;
        let mut args = table_args(&db.path, Path::new("out.csv"));
        args.group = group;
        data_export(&backend, results.as_ref(), &args, TableFileFormat::Csv)
    }

    #[test]
    fn group_by_image_is_a_row_per_image_with_its_statistics() {
        let data = data_of(GroupArgs {
            group_by: Some(GroupByKind::Image),
            agg: vec![AggKind::Min, AggKind::Max, AggKind::Sum],
            ..Default::default()
        })
        .unwrap();
        assert_eq!(data.outputs, [DataOutput::Images]);
        assert_eq!(
            data.aggregations,
            [
                ExportAggregation::Min,
                ExportAggregation::Max,
                ExportAggregation::Sum
            ]
        );
        let data = data_of(GroupArgs {
            group_by: Some(GroupByKind::Image),
            ..Default::default()
        })
        .unwrap();
        assert_eq!(data.aggregations, [ExportAggregation::Avg]);
        assert!(
            data_of(GroupArgs {
                group_by: Some(GroupByKind::Image),
                agg: vec![AggKind::Q1],
                ..Default::default()
            })
            .is_err(),
            "q1 only for a time course"
        );
    }

    /// class / well / group / folder / regex / track: the time course -
    /// mean and SD unless asked otherwise.
    #[test]
    fn the_time_course_groupings() {
        let resolve = |group_by, group_regex: Option<&str>| {
            data_of(GroupArgs {
                group_by: Some(group_by),
                group_regex: group_regex.map(str::to_string),
                ..Default::default()
            })
        };
        let lines = |data: DataExport| (data.time_course.unwrap().lines, data.group_regex);
        assert_eq!(
            lines(resolve(GroupByKind::Class, None).unwrap()),
            (TimeCourseLineKind::Class, None)
        );
        assert_eq!(
            lines(resolve(GroupByKind::Well, None).unwrap()),
            (TimeCourseLineKind::Well, None)
        );
        assert_eq!(
            lines(resolve(GroupByKind::Folder, None).unwrap()),
            (TimeCourseLineKind::Folder, None)
        );
        assert_eq!(
            lines(resolve(GroupByKind::Group, Some("^(.*)_")).unwrap()),
            (TimeCourseLineKind::Well, Some("^(.*)_".into()))
        );
        assert_eq!(
            lines(resolve(GroupByKind::Track, None).unwrap()),
            (TimeCourseLineKind::Track, None)
        );
        assert!(
            resolve(GroupByKind::Group, None).is_err(),
            "needs --group-regex"
        );
        assert!(resolve(GroupByKind::Regex, Some("")).is_err());

        let data = resolve(GroupByKind::Class, None).unwrap();
        assert_eq!(
            data.aggregations,
            [ExportAggregation::Avg, ExportAggregation::Stddev]
        );
        let course = data.time_course.unwrap();
        assert!(!course.relative);
        assert_eq!(course.layout, TimeCourseTableLayout::Long);

        let data = data_of(GroupArgs {
            group_by: Some(GroupByKind::Class),
            agg: vec![AggKind::Median, AggKind::Q1, AggKind::Q3],
            normalize: Some(NormalizeKind::T0),
            layout: LayoutKind::Both,
            ..Default::default()
        })
        .unwrap();
        assert_eq!(
            data.aggregations,
            [
                ExportAggregation::Median,
                ExportAggregation::Q1,
                ExportAggregation::Q3
            ]
        );
        let course = data.time_course.unwrap();
        assert!(course.relative);
        assert_eq!(course.layout, TimeCourseTableLayout::Both);
    }

    fn time_course(group: GroupArgs, out: &Path) -> String {
        let db = TempResultsDb::seeded();
        let mut args = table_args(&db.path, out);
        args.group = group;
        export_table(args, TableFileFormat::Csv).expect("time course export");
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
                group_by: Some(GroupByKind::Class),
                ..Default::default()
            },
            &out,
        );
        let header = csv.lines().next().unwrap();
        assert!(
            header.starts_with("class,t,z,time_s,n_images,n_objects,"),
            "{header}"
        );
        assert!(header.contains("area_px_avg,area_px_stddev"), "{header}");
        assert_eq!(csv.lines().count(), 1 + 2 * 2, "2 planes x 2 classes");
        assert_eq!(
            names_in(out_dir.path()),
            ["course.csv"],
            "the scratch folder is gone"
        );
    }

    /// q1/q3 and folder lines: the runner has them for the command line.
    #[test]
    fn export_table_time_course_per_folder_with_quartiles() {
        let out_dir = tempfile::tempdir().expect("tempdir");
        let out = out_dir.path().join("course.csv");
        let csv = time_course(
            GroupArgs {
                group_by: Some(GroupByKind::Folder),
                agg: vec![AggKind::Q1, AggKind::Q3],
                ..Default::default()
            },
            &out,
        );
        let header = csv.lines().next().unwrap();
        assert!(header.starts_with("folder,class,"), "{header}");
        assert!(header.contains("area_px_q1,area_px_q3"), "{header}");
    }

    /// `--layout both` in CSV: the long table at `--out`, a file per value
    /// next to it.
    #[test]
    fn export_table_time_course_wide_csv_goes_next_to_out() {
        let out_dir = tempfile::tempdir().expect("tempdir");
        let out = out_dir.path().join("course.csv");
        time_course(
            GroupArgs {
                group_by: Some(GroupByKind::Well),
                agg: vec![AggKind::Avg],
                layout: LayoutKind::Both,
                ..Default::default()
            },
            &out,
        );
        let names = names_in(out_dir.path());
        assert!(names.contains(&"course.csv".to_string()), "{names:?}");
        assert!(
            names
                .iter()
                .any(|n| n.starts_with("course_wide_area_px_avg")),
            "{names:?}"
        );
        assert!(
            names.iter().all(|n| n.starts_with("course")),
            "no scratch folder left: {names:?}"
        );
    }

    #[test]
    fn export_table_writes_one_intensity_column_group_per_real_image_channel() {
        let db = TempResultsDb::with_channels(3);
        let out_dir = tempfile::tempdir().expect("tempdir");
        let out = out_dir.path().join("out.csv");

        export_table(table_args(&db.path, &out), TableFileFormat::Csv)
            .expect("csv export should succeed");

        let content = std::fs::read_to_string(&out).expect("read csv back");
        let header = content.lines().next().expect("header row present");
        for ch in 0..3 {
            assert!(
                header.contains(&format!("intensity_mean_gray_ch{ch}")),
                "header missing channel {ch}'s intensity column: {header}"
            );
        }
        assert!(
            !header.contains("_ch3"),
            "no 4th channel that was never measured: {header}"
        );
    }
}
