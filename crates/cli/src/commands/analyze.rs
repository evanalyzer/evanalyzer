//! # analyze
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-06-26
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.

use crate::args::AnalyzeArgs;
use evanalyzer_app::backend::Backend;
use evanalyzer_app::jobs::AnalysisRequest;
use evanalyzer_app::jobs::ProgressEvent;
use evanalyzer_app::jobs::RunningJob;
use evanalyzer_app::project::ImageListExt;
use evanalyzer_app::project::load_project;
use evanalyzer_cfg::core_types::InternalErrors;
use std::io::Write;
use std::path::PathBuf;
use std::time::SystemTime;

pub fn run(args: AnalyzeArgs, backend: &dyn Backend) -> Result<(), InternalErrors> {
    let mut project = load_project(backend.files(), &args.project)?;

    if let Some(images_dir) = &args.images {
        project.images.root = Some(images_dir.clone());
        project.scan_image_folder_and_add(backend);
    }

    for line in crate::overrides::apply_overrides(&mut project, &args.set)? {
        println!("Set:       {line}");
    }

    let image_count = project.images.list.len();
    if image_count == 0 {
        return Err(InternalErrors::InvalidArgument(
            "Project has no images - pass --images <dir> or add images to the project first".into(),
        ));
    }

    let project_dir = args
        .project
        .parent()
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("."));

    let enabled_pipelines = project.pipelines.iter().filter(|p| p.enabled).count();
    println!("Project:   {}", args.project.display());
    println!("Images:    {image_count}");
    println!("Pipelines: {enabled_pipelines} enabled");

    let start = SystemTime::now();
    let job = backend.start_analysis(AnalysisRequest {
        settings: project.settings.clone(),
        project_path: project_dir,
        job_name: args.job_name.clone(),
        threads: args.threads,
        base_results: None,
    })?;
    let output_path = job.output_path().clone();
    println!("Output:    {}", output_path.display());
    println!(
        "Running with {} parallel thread(s) (Ctrl+C to cancel)...\n",
        job.parallelism()
    );

    follow(job, backend, image_count, start)?;
    run_exports_after(&args, backend, &project, &output_path)
}

/// The exports after a run: those set to run after analysis, or the
/// `--export`s, none with `--no-exports`. Fails if one of them does.
fn run_exports_after(
    args: &AnalyzeArgs,
    backend: &dyn Backend,
    project: &evanalyzer_app::project::ProjectWithRuntime,
    run_folder: &std::path::Path,
) -> Result<(), InternalErrors> {
    use crate::commands::exports::{Scripts, no_such_export, run_export_on};
    let exports = &project.settings.exports;
    let chosen: Vec<_> = if args.no_exports {
        Vec::new()
    } else if args.exports.is_empty() {
        exports
            .iter()
            .filter(|e| e.run_after_analysis)
            .cloned()
            .collect()
    } else {
        args.exports
            .iter()
            .map(|wanted| {
                evanalyzer_app::exports::find_export(exports, wanted)
                    .cloned()
                    .ok_or_else(|| no_such_export(exports, wanted))
            })
            .collect::<Result<_, _>>()?
    };
    if chosen.is_empty() {
        return Ok(());
    }
    let project_dir = args
        .project
        .parent()
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("."));
    let database =
        evanalyzer_app::exports::results_database(backend.files(), &project_dir, Some(run_folder))?;
    let scripts = if args.allow_scripts {
        Scripts::All
    } else {
        Scripts::Allowed
    };
    let mut failed = Vec::new();
    for export in chosen {
        println!();
        let name = export.name.clone();
        match run_export_on(
            backend,
            project,
            &project_dir,
            export,
            &database,
            None,
            scripts,
        ) {
            Ok(0) => {}
            Ok(n) => failed.push(format!("“{name}”: {n} entries failed")),
            Err(e) => failed.push(format!("“{name}”: {e}")),
        }
    }
    if failed.is_empty() {
        Ok(())
    } else {
        Err(InternalErrors::Internal(format!(
            "the analysis is done, but exports failed - {}",
            failed.join("; ")
        )))
    }
}

/// Prints `job`'s progress until it ends (Ctrl+C cancels it), then its
/// summary - with the time since `started`. Shared with `attach`.
pub(crate) fn follow(
    job: RunningJob,
    backend: &dyn Backend,
    image_count: usize,
    started: SystemTime,
) -> Result<(), InternalErrors> {
    let cancel = job.cancel_handle();
    if let Err(e) = ctrlc::set_handler(move || {
        eprintln!("\nCancelling... (waiting for in-flight images to finish)");
        cancel.cancel();
    }) {
        eprintln!("Warning: could not install Ctrl+C handler: {e}");
    }

    let output_path = job.output_path().clone();
    let id = job.id().map(str::to_string);
    let mut failed = 0usize;
    let mut total = image_count;
    for event in job.events() {
        apply_progress_event(event, &mut total, &mut failed);
    }
    match job.wait() {
        Ok(_) => {}
        // A server's analysis goes on without us.
        Err(e) if !backend.is_connected() => {
            return Err(match id {
                Some(id) => InternalErrors::Io(format!(
                    "Lost the connection to the server - the analysis keeps running there. \
                     Follow it again with `cli jobs attach --job {id}` (same --remote and --user)."
                )),
                None => e,
            });
        }
        Err(e) => return Err(e),
    }

    println!(
        "Done: {total} image(s) analyzed in {:.1?} ({failed} failed)",
        started.elapsed().unwrap_or_default()
    );
    println!("Results database written under: {}", output_path.display());
    Ok(())
}

/// Applies one [`ProgressEvent`] to the running `total`/`failed` counters and
/// prints the corresponding progress line to stdout, mirroring the CLI's
/// console output.
///
/// Factored out of `run`'s event loop so the state-transition logic (what
/// each event does to `total`/`failed`) can be unit-tested without needing a
/// real pipeline run to produce these events.
fn apply_progress_event(event: ProgressEvent, total: &mut usize, failed: &mut usize) {
    match event {
        ProgressEvent::Started { total: t } => *total = t,
        ProgressEvent::ImageCompleted {
            index,
            total: t,
            path,
        } => {
            *total = t;
            print!("\r[{index}/{t}] {}          ", path.display());
            std::io::stdout().flush().ok();
        }
        ProgressEvent::ImageFailed { path } => {
            *failed += 1;
            println!("\nFAILED: {}", path.display());
        }
        ProgressEvent::Finished => println!(),
        ProgressEvent::TilesScheduled { .. }
        | ProgressEvent::ImageStarted { .. }
        | ProgressEvent::TileCompleted { .. }
        | ProgressEvent::WholeImagePhaseCompleted { .. }
        | ProgressEvent::BreakpointReached { .. }
        | ProgressEvent::PredictionReached { .. }
        | ProgressEvent::ObjectPredictionReached { .. } => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::commands::test_support::TempProjectFile;
    use evanalyzer_app::backend::local::LocalBackend;
    use evanalyzer_app::results::LocalResultsGenerator;
    use evanalyzer_app::results::{Cell, CellValue, Column, ListFilter, Pagination, PlaneFilter};
    use evanalyzer_cfg::settings::project_settings::ProjectSettings;

    fn run(args: AnalyzeArgs) -> Result<(), InternalErrors> {
        super::run(args, &LocalBackend::default())
    }

    #[test]
    fn run_rejects_a_project_with_no_images_and_no_images_dir_override() {
        let file = TempProjectFile::new(&ProjectSettings::default());

        let result = run(AnalyzeArgs {
            project: file.path.clone(),
            images: None,
            threads: None,
            job_name: None,
            set: Vec::new(),
            exports: Vec::new(),
            no_exports: false,
            allow_scripts: false,
        });

        let err = result.expect_err("expected an empty project to be rejected before any job runs");
        let InternalErrors::InvalidArgument(msg) = err else {
            panic!("expected InvalidArgument, got {err:?}");
        };
        assert!(msg.contains("no images"));
    }

    #[test]
    fn run_errors_when_the_project_file_does_not_exist() {
        let result = run(AnalyzeArgs {
            project: std::path::PathBuf::from("/nonexistent/does_not_exist.evaproj"),
            images: None,
            threads: None,
            job_name: None,
            set: Vec::new(),
            exports: Vec::new(),
            no_exports: false,
            allow_scripts: false,
        });

        assert!(result.is_err());
    }

    #[test]
    fn run_applies_the_images_override_and_still_rejects_an_empty_scan_result() {
        let file = TempProjectFile::new(&ProjectSettings::default());
        // A real (empty) directory - the `--images` override path must set
        // `project.images.root` to this and call `scan_image_folder_and_add()`
        // before the "no images" check runs, not merely reuse whatever (if
        // anything) was already in the project's saved image list.
        let images_dir = file.path.parent().unwrap().join("images");
        std::fs::create_dir_all(&images_dir).unwrap();

        let result = run(AnalyzeArgs {
            project: file.path.clone(),
            images: Some(images_dir),
            threads: None,
            job_name: None,
            set: Vec::new(),
            exports: Vec::new(),
            no_exports: false,
            allow_scripts: false,
        });

        let err = result
            .expect_err("an images dir override that scans to zero images must still be rejected");
        let InternalErrors::InvalidArgument(msg) = err else {
            panic!("expected InvalidArgument, got {err:?}");
        };
        assert!(msg.contains("no images"));
    }

    #[test]
    fn run_errors_when_the_images_override_directory_does_not_exist() {
        let file = TempProjectFile::new(&ProjectSettings::default());
        let missing_dir = file.path.parent().unwrap().join("does_not_exist");

        let result = run(AnalyzeArgs {
            project: file.path.clone(),
            images: Some(missing_dir),
            threads: None,
            job_name: None,
            set: Vec::new(),
            exports: Vec::new(),
            no_exports: false,
            allow_scripts: false,
        });

        let err = result.expect_err("scanning a nonexistent images dir must still yield 0 images");
        let InternalErrors::InvalidArgument(msg) = err else {
            panic!("expected InvalidArgument, got {err:?}");
        };
        assert!(msg.contains("no images"));
    }

    /// The only test that drives `run`'s full happy path (job creation,
    /// `run_async`, the progress loop, thread join, final println) - every
    /// other test above only reaches the early "no images" return. Reuses
    /// the same real fixture image `core`'s own tests read from
    /// (`multi-channel-4D-series.ome.tif`), copied into an isolated
    /// directory so `--images` scans exactly one image, not `core/tests`'
    /// whole mixed fixture directory.
    #[test]
    fn run_succeeds_end_to_end_against_a_real_image_with_no_pipelines() {
        let file = TempProjectFile::new(&ProjectSettings::default());
        let images_dir = file.path.parent().unwrap().join("images");
        std::fs::create_dir_all(&images_dir).unwrap();
        let fixture = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../core/tests/multi-channel-4D-series.ome.tif");
        std::fs::copy(&fixture, images_dir.join("fixture.ome.tif"))
            .expect("copy the real fixture image into an isolated scan directory");

        let result = run(AnalyzeArgs {
            project: file.path.clone(),
            images: Some(images_dir),
            threads: Some(1),
            job_name: Some("cli_test".into()),
            set: Vec::new(),
            exports: Vec::new(),
            no_exports: false,
            allow_scripts: false,
        });

        result.expect("a real image with zero enabled pipelines should still run to completion");
    }

    // -- apply_progress_event -------------------------------------------------

    fn sample_path() -> PathBuf {
        PathBuf::from("some/image.tif")
    }

    #[test]
    fn apply_progress_event_started_sets_total() {
        let mut total = 0;
        let mut failed = 0;
        apply_progress_event(ProgressEvent::Started { total: 7 }, &mut total, &mut failed);
        assert_eq!(total, 7);
        assert_eq!(failed, 0);
    }

    #[test]
    fn apply_progress_event_image_completed_updates_total_and_leaves_failed_unchanged() {
        let mut total = 1;
        let mut failed = 2;
        apply_progress_event(
            ProgressEvent::ImageCompleted {
                index: 3,
                total: 5,
                path: sample_path(),
            },
            &mut total,
            &mut failed,
        );
        assert_eq!(total, 5);
        assert_eq!(failed, 2);
    }

    #[test]
    fn apply_progress_event_image_failed_increments_failed_and_leaves_total_unchanged() {
        let mut total = 5;
        let mut failed = 0;
        apply_progress_event(
            ProgressEvent::ImageFailed {
                path: sample_path(),
            },
            &mut total,
            &mut failed,
        );
        assert_eq!(total, 5);
        assert_eq!(failed, 1);

        apply_progress_event(
            ProgressEvent::ImageFailed {
                path: sample_path(),
            },
            &mut total,
            &mut failed,
        );
        assert_eq!(failed, 2, "a second failure must accumulate, not overwrite");
    }

    #[test]
    fn apply_progress_event_finished_leaves_counters_unchanged() {
        let mut total = 4;
        let mut failed = 1;
        apply_progress_event(ProgressEvent::Finished, &mut total, &mut failed);
        assert_eq!(total, 4);
        assert_eq!(failed, 1);
    }

    #[test]
    fn apply_progress_event_ignores_tiles_scheduled_and_tile_completed() {
        let mut total = 4;
        let mut failed = 1;

        apply_progress_event(
            ProgressEvent::TilesScheduled { total_tiles: 99 },
            &mut total,
            &mut failed,
        );
        apply_progress_event(
            ProgressEvent::TileCompleted {
                tile_index: 1,
                total_tiles: 99,
                objects: Vec::new(),
            },
            &mut total,
            &mut failed,
        );

        assert_eq!(total, 4);
        assert_eq!(failed, 1);
    }

    /// Full end-to-end proof the CLI actually works, not just that its
    /// pieces do in isolation: `analyze` runs a real (manual, full-range
    /// threshold -> connected components -> extract objects, mirroring
    /// evanalyzer_core's own `threshold_connected_components_extract_pipeline`
    /// integration test) pipeline against two real fixture images, then the
    /// resulting `.evadb` is exported via the CLI's own `export csv`/
    /// `export xlsx` commands (`crate::commands::export::run`, the exact
    /// function `evanalyzer cli results export csv/xlsx` invokes) and the produced
    /// files are checked for real content, not just "didn't error".
    #[test]
    fn run_then_export_csv_and_xlsx_via_the_cli_end_to_end() {
        use crate::args::{ExportArgs, ExportCommand, FilterArgs, GroupArgs, TableExportArgs};
        use evanalyzer_cfg::core_types::{ImageAddress, PipelineId, SegmentationClass};
        use evanalyzer_cfg::settings::pipeline_command::PipelineCommand;
        use evanalyzer_cfg::settings::pipeline_command_settings::{
            ConnectedComponentsSettings, ExtractObjectsSettings, ThresholdEntrySettings,
            ThresholdSettings,
        };
        use evanalyzer_cfg::settings::pipeline_settings::{PipelineSettings, PipelineStepSettings};

        // A manual threshold defaults to `min_threshold: 0.0, max_threshold:
        // 65535.0` (`ThresholdEntrySettings::default()`) - every pixel of
        // any real image falls in range, so this reliably produces at least
        // one connected component regardless of the fixture's actual
        // intensity distribution, same reasoning as the core-level test this
        // mirrors.
        let mut settings = ProjectSettings::default();
        settings.pipelines = vec![PipelineSettings {
            id: PipelineId(1),
            name: "detect".to_string(),
            description: None,
            image_source: ImageAddress::Channel(0),
            enabled: true,
            steps: vec![
                PipelineStepSettings {
                    id: evanalyzer_cfg::core_types::StepId::UNASSIGNED,
                    enabled: true,
                    content: evanalyzer_cfg::settings::pipeline_settings::StepContent::Command(
                        PipelineCommand::Threshold(ThresholdSettings {
                            id: String::new(),
                            thresholds: vec![ThresholdEntrySettings {
                                segmentation_class: SegmentationClass(1),
                                ..Default::default()
                            }],
                        }),
                    ),
                },
                PipelineStepSettings {
                    id: evanalyzer_cfg::core_types::StepId::UNASSIGNED,
                    enabled: true,
                    content: evanalyzer_cfg::settings::pipeline_settings::StepContent::Command(
                        PipelineCommand::ConnectedComponents(ConnectedComponentsSettings::default()),
                    ),
                },
                PipelineStepSettings {
                    id: evanalyzer_cfg::core_types::StepId::UNASSIGNED,
                    enabled: true,
                    content: evanalyzer_cfg::settings::pipeline_settings::StepContent::Command(
                        PipelineCommand::ExtractObjects(ExtractObjectsSettings::default()),
                    ),
                },
            ],
        }];
        // Runs after the analysis, into its results folder.
        settings.exports = serde_json::from_value(serde_json::json!([
            { "id": 1, "name": "After", "runAfterAnalysis": true, "entries": [
                { "kind": "DATA", "outputs": ["IMAGES"] },
                { "kind": "DRAWN_IMAGES", "maxSide": 128 }
            ] },
            { "id": 2, "name": "Not after", "entries": [{ "kind": "DATA", "outputs": ["OBJECTS"] }] }
        ]))
        .unwrap();

        let file = TempProjectFile::new(&settings);
        let images_dir = file.path.parent().unwrap().join("images");
        std::fs::create_dir_all(&images_dir).unwrap();
        // Two real fixture images (already used elsewhere in this crate's
        // and evanalyzer_core's own tests), so the export step has more than
        // one image's worth of objects to report on.
        let fixture_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../core/tests");
        std::fs::copy(
            fixture_dir.join("multi-channel-4D-series.ome.tif"),
            images_dir.join("a.ome.tif"),
        )
        .expect("copy fixture a");
        std::fs::copy(
            fixture_dir.join("slice_Z0_C0_T0.tif"),
            images_dir.join("b.tif"),
        )
        .expect("copy fixture b");

        run(AnalyzeArgs {
            project: file.path.clone(),
            images: Some(images_dir),
            threads: Some(1),
            job_name: Some("e2e".into()),
            set: Vec::new(),
            exports: Vec::new(),
            no_exports: false,
            allow_scripts: false,
        })
        .expect("analyze should succeed against real fixture images with a working pipeline");

        let results_root = file.path.parent().unwrap().join("results");
        let evadb = find_evadb(&results_root).expect("analyze should have produced a .evadb file");
        // The export set to run after analysis wrote next to the results,
        // the other one didn't.
        let run_folder = evadb.parent().unwrap();
        let images_csv = std::fs::read_to_string(run_folder.join("images.csv")).unwrap();
        assert!(images_csv.starts_with("image,class,"), "{images_csv}");
        assert!(run_folder.join("drawn_images_a_ome.png").exists());
        assert!(run_folder.join("drawn_images_b.png").exists());
        assert!(!run_folder.join("objects.csv").exists());

        let out_dir = tempfile::tempdir().expect("tempdir");

        // -- Ground truth: every detected object's full row, straight from
        // `ResultsGenerator` itself (the same source both `export csv` and
        // `export xlsx` read from) - swept across every z/t plane the same
        // way `export_table` does (`0..=get_nr_of_{z,t}_stacks()`), and
        // keyed by object id so row *order* differences between the direct
        // query and the exported files can't cause a false mismatch.
        let database = LocalResultsGenerator::open_database(evadb.clone()).expect("open evadb");
        // The list's cells come in `Column`'s order.
        let mut expected_columns: Vec<Column> = database
            .get_available_columns()
            .expect("available columns")
            .into_iter()
            .map(|entry| entry.key)
            .collect();
        expected_columns.sort();
        expected_columns.dedup();
        let mut expected_by_id: std::collections::HashMap<String, Vec<String>> =
            std::collections::HashMap::new();
        for z in 0..=database.get_nr_of_z_stacks() {
            for t in 0..=database.get_nr_of_t_stacks() {
                let page = database
                    .get_object_list(&ListFilter {
                        plane: PlaneFilter {
                            z_stack: z,
                            t_stack: t,
                        },
                        images: evanalyzer_app::results::ImageSet::All,
                        object_classes: None,
                        columns: expected_columns.clone(),
                        with_coloc_details: false,
                        related_to: None,
                        page: Pagination {
                            limit: 1_000_000,
                            after: None,
                        },
                        transpond_table: false,
                        sort: None,
                    })
                    .expect("ground-truth object list");
                for (id, row) in page.row_names.iter().zip(page.rows) {
                    expected_by_id.insert(id.clone(), row.iter().map(cell_to_comparable).collect());
                }
            }
        }
        assert!(
            !expected_by_id.is_empty(),
            "thresholding the whole pixel range should have detected at least one object"
        );

        // Each exported row against the ground truth, column by column by
        // its format 1 name (lengths in µm: the stored nm times the factor).
        let classes = database.get_object_classes().expect("classes");
        let check = |header: &[String], rows: &[Vec<String>], format: &str| {
            assert_eq!(header[0], "image", "{format} header: {header:?}");
            let at = |name: &str| {
                header
                    .iter()
                    .position(|h| h == name)
                    .unwrap_or_else(|| panic!("{format}: no column {name}: {header:?}"))
            };
            let object_id_col = at("object_id");
            for row in rows {
                assert_eq!(row.len(), header.len(), "{format} row: {row:?}");
                let id = row[object_id_col]
                    .parse::<f64>()
                    .map(|id| (id as i64).to_string())
                    .unwrap_or_else(|_| row[object_id_col].clone());
                let expected_row = expected_by_id
                    .get(&id)
                    .unwrap_or_else(|| panic!("{format} row for object {id} has no ground truth"));
                for (column, expected) in expected_columns.iter().zip(expected_row) {
                    // The count is always 1 for an object: not in its rows.
                    if evanalyzer_app::exports::is_fixed_column(column) || *column == Column::Count
                    {
                        continue;
                    }
                    let shown = evanalyzer_app::results::present::present(
                        column,
                        evanalyzer_app::results::present::LengthUnit::Um,
                        &classes,
                    );
                    let expected = match expected.parse::<f64>() {
                        Ok(v) if shown.factor != 1.0 => (v * shown.factor).to_string(),
                        _ => expected.clone(),
                    };
                    assert_values_match(&row[at(&shown.name)], &expected, &shown.name, format);
                }
            }
            assert_eq!(
                rows.len(),
                expected_by_id.len(),
                "{format}: exactly one row per detected object"
            );
        };

        let csv_out = out_dir.path().join("out.csv");
        crate::commands::export::run(
            ExportArgs {
                command: ExportCommand::Csv(TableExportArgs {
                    db: evadb.clone(),
                    out: csv_out.clone(),
                    filter: FilterArgs::default(),
                    group: GroupArgs::default(),
                    tracks: Default::default(),
                    length_unit: Default::default(),
                    measurements: false,
                    project: None,
                    view: None,
                }),
            },
            &LocalBackend::default(),
        )
        .expect("csv export should succeed");
        let csv_content = std::fs::read_to_string(&csv_out).expect("read exported csv");
        let mut csv_lines = csv_content.lines();
        let csv_header: Vec<String> = csv_lines
            .next()
            .expect("csv header row")
            .split(',')
            .map(str::to_string)
            .collect();
        let csv_rows: Vec<Vec<String>> = csv_lines
            .map(|line| line.split(',').map(str::to_string).collect())
            .collect();
        check(&csv_header, &csv_rows, "csv");

        let xlsx_out = out_dir.path().join("out.xlsx");
        crate::commands::export::run(
            ExportArgs {
                command: ExportCommand::Xlsx(TableExportArgs {
                    db: evadb,
                    out: xlsx_out.clone(),
                    filter: FilterArgs::default(),
                    group: GroupArgs::default(),
                    tracks: Default::default(),
                    length_unit: Default::default(),
                    measurements: false,
                    project: None,
                    view: None,
                }),
            },
            &LocalBackend::default(),
        )
        .expect("xlsx export should succeed");

        use calamine::Reader;
        let mut workbook: calamine::Xlsx<_> =
            calamine::open_workbook(&xlsx_out).expect("open exported xlsx");
        let range = workbook.worksheet_range("Table").expect("Table sheet");
        let text = |cell: &calamine::Data| match cell {
            calamine::Data::String(s) => s.clone(),
            calamine::Data::Float(f) => f.to_string(),
            calamine::Data::Int(i) => i.to_string(),
            calamine::Data::Empty => String::new(),
            other => panic!("unexpected xlsx cell type: {other:?}"),
        };
        let mut rows = range.rows();
        let xlsx_header: Vec<String> = rows
            .next()
            .expect("xlsx header row")
            .iter()
            .map(text)
            .collect();
        assert_eq!(xlsx_header, csv_header, "the same columns as the CSV");
        let xlsx_rows: Vec<Vec<String>> = rows.map(|row| row.iter().map(text).collect()).collect();
        check(&xlsx_header, &xlsx_rows, "xlsx");
    }

    /// Canonical text form of a ground-truth `Cell`, matching what CSV/XLSX
    /// actually write for the same value (`results_exporter.rs`'s own
    /// `cell_text`/`write_cell`) closely enough for [`assert_values_match`]
    /// to compare against - only the numeric-vs-text distinction and the
    /// literal digits matter, not exact formatting.
    fn cell_to_comparable(cell: &Cell) -> String {
        match &cell.value {
            CellValue::Empty => String::new(),
            CellValue::String(s) => s.clone(),
            CellValue::Class((s, _)) => s.clone(),
            CellValue::Float(v) => v.to_string(),
            CellValue::Integer(v) => v.to_string(),
        }
    }

    /// Asserts `got` (a field read back from an exported CSV/XLSX file) and
    /// `expected` (the same column's `ResultsGenerator`-derived ground
    /// truth, via `cell_to_comparable`) agree - numerically (with a small
    /// tolerance, since XLSX round-trips every number through `f64`
    /// regardless of the original `Cell`'s `Integer`/`Float` type) when both
    /// parse as numbers, textually otherwise.
    fn assert_values_match(got: &str, expected: &str, column: &str, format: &str) {
        match (got.parse::<f64>(), expected.parse::<f64>()) {
            (Ok(g), Ok(e)) => assert!(
                (g - e).abs() < 1e-3,
                "{format} column {column:?}: expected {expected}, got {got}"
            ),
            _ => assert_eq!(got, expected, "{format} column {column:?}"),
        }
    }

    /// Recursively finds the first `*.evadb` file under `dir` — `analyze`
    /// writes its results database under a timestamped subdirectory of
    /// `<project_dir>/results/`, so the exact path isn't predictable ahead
    /// of time.
    /// `--set` reaches the run: the run's copy of the project next to its
    /// results holds the overridden value, the project file the old one;
    /// an override that doesn't apply stops before anything runs.
    #[test]
    fn run_with_set_runs_with_the_overridden_value_and_keeps_the_project_file() {
        use evanalyzer_cfg::core_types::{ImageAddress, PipelineId, SegmentationClass, StepId};
        use evanalyzer_cfg::settings::pipeline_command::PipelineCommand;
        use evanalyzer_cfg::settings::pipeline_command_settings::{
            ConnectedComponentsSettings, ExtractObjectsSettings, ThresholdEntrySettings,
            ThresholdSettings,
        };
        use evanalyzer_cfg::settings::pipeline_settings::{
            PipelineSettings, PipelineStepSettings, StepContent,
        };

        let step = |id, command| PipelineStepSettings {
            id: StepId(id),
            enabled: true,
            content: StepContent::Command(command),
        };
        let mut settings = ProjectSettings::default();
        settings.pipelines = vec![PipelineSettings {
            id: PipelineId(1),
            name: "Varroa detection".to_string(),
            description: None,
            image_source: ImageAddress::Channel(0),
            enabled: true,
            steps: vec![
                step(
                    1,
                    PipelineCommand::Threshold(ThresholdSettings {
                        id: String::new(),
                        thresholds: vec![ThresholdEntrySettings {
                            segmentation_class: SegmentationClass(1),
                            ..Default::default()
                        }],
                    }),
                ),
                step(
                    2,
                    PipelineCommand::ConnectedComponents(ConnectedComponentsSettings::default()),
                ),
                step(
                    3,
                    PipelineCommand::ExtractObjects(ExtractObjectsSettings::default()),
                ),
            ],
        }];
        let file = TempProjectFile::new(&settings);
        let images_dir = file.path.parent().unwrap().join("images");
        std::fs::create_dir_all(&images_dir).unwrap();
        std::fs::copy(
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../core/tests/slice_Z0_C0_T0.tif"),
            images_dir.join("a.tif"),
        )
        .expect("copy fixture");
        let min_threshold = |project: &serde_json::Value| {
            project["pipelines"][0]["steps"][0]["command"]["thresholds"][0]["minThreshold"].as_f64()
        };
        let read = |path: &std::path::Path| -> serde_json::Value {
            serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
        };

        let err = run(AnalyzeArgs {
            project: file.path.clone(),
            images: Some(images_dir.clone()),
            threads: Some(1),
            job_name: Some("wrong".into()),
            set: vec!["Varroa detection.1.thresholds.0.minThreshold=low".into()],
            exports: Vec::new(),
            no_exports: false,
            allow_scripts: false,
        })
        .expect_err("a value the setting doesn't take must stop the run");
        assert!(err.to_string().contains("it takes a number"), "{err}");
        let results_root = file.path.parent().unwrap().join("results");
        assert!(find_evadb(&results_root).is_none(), "nothing may have run");

        run(AnalyzeArgs {
            project: file.path.clone(),
            images: Some(images_dir),
            threads: Some(1),
            job_name: Some("set".into()),
            set: vec!["Varroa detection.1.thresholds.0.minThreshold=123".into()],
            exports: Vec::new(),
            no_exports: false,
            allow_scripts: false,
        })
        .expect("analyze with --set should succeed");

        let evadb = find_evadb(&results_root).expect("analyze should have produced a .evadb file");
        let snapshot = evadb.with_extension("evaproj");
        assert_eq!(
            min_threshold(&read(&snapshot)),
            Some(123.0),
            "the run's copy"
        );
        assert_eq!(
            min_threshold(&read(&file.path)),
            Some(0.0),
            "the project file"
        );
    }

    fn find_evadb(dir: &std::path::Path) -> Option<PathBuf> {
        for entry in std::fs::read_dir(dir).ok()? {
            let entry = entry.ok()?;
            let path = entry.path();
            if path.is_dir() {
                if let Some(found) = find_evadb(&path) {
                    return Some(found);
                }
            } else if path.extension().and_then(|e| e.to_str()) == Some("evadb") {
                return Some(path);
            }
        }
        None
    }
}
