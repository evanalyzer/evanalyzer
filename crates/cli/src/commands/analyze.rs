use crate::args::AnalyzeArgs;
use evanalyzer_app::extensions::project_ext::{ProjectExt, load_project};
use evanalyzer_cfg::core_types::InternalErrors;
use evanalyzer_core::ProgressEvent;
use std::io::Write;
use std::path::PathBuf;
use std::sync::atomic::Ordering;
use std::time::Instant;

pub fn run(args: AnalyzeArgs) -> Result<(), InternalErrors> {
    let mut project = load_project(&args.project)?;

    if let Some(images_dir) = &args.images {
        project.images.root = Some(images_dir.clone());
        project.scan_image_folder_and_add();
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

    let job = evanalyzer_core::generate_analyze_job_from_project_settings(
        project.settings.clone(),
        project_dir,
        args.job_name.clone(),
    )?;
    let output_path = job.output_path.clone();

    // Caps parallelism to available RAM as well as CPU cores, so a low-memory
    // machine doesn't try to run as many concurrent workers as it has cores.
    // The per-worker estimate is sized to the images actually being
    // analyzed, not a flat guess - see `estimate_ram_per_worker_bytes`.
    let threads = args.threads.unwrap_or_else(|| {
        evanalyzer_core::recommended_parallelism(job.estimate_ram_per_worker_bytes())
    });
    println!("Output:    {}", output_path.display());
    println!("Running with {threads} parallel thread(s) (Ctrl+C to cancel)...\n");

    let start = Instant::now();
    let (handle, rx, cancel) = job.run_async(threads);

    if let Err(e) = ctrlc::set_handler(move || {
        eprintln!("\nCancelling... (waiting for in-flight images to finish)");
        cancel.store(true, Ordering::SeqCst);
    }) {
        eprintln!("Warning: could not install Ctrl+C handler: {e}");
    }

    let mut failed = 0usize;
    let mut total = image_count;
    for event in rx {
        apply_progress_event(event, &mut total, &mut failed);
    }

    let result = handle
        .join()
        .map_err(|_| InternalErrors::Internal("Pipeline worker thread panicked".into()))?;
    result?;

    println!(
        "Done: {total} image(s) analyzed in {:.1?} ({failed} failed)",
        start.elapsed()
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
        | ProgressEvent::TileCompleted { .. }
        | ProgressEvent::WholeImagePhaseCompleted { .. }
        | ProgressEvent::BreakpointReached { .. } => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use calamine::DataType as _;
    use crate::commands::test_support::TempProjectFile;
    use evanalyzer_app::result::{Cell, CellValue, Column, ListFilter, Pagination, PlaneFilter, ResultsGenerator};
    use evanalyzer_cfg::settings::project_settings::ProjectSettings;

    #[test]
    fn run_rejects_a_project_with_no_images_and_no_images_dir_override() {
        let file = TempProjectFile::new(&ProjectSettings::default());

        let result = run(AnalyzeArgs {
            project: file.path.clone(),
            images: None,
            threads: None,
            job_name: None,
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
    /// function `evanalyzer_cli export csv/xlsx` invokes) and the produced
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
                    enabled: true,
                    command: PipelineCommand::Threshold(ThresholdSettings {
                        thresholds: vec![ThresholdEntrySettings {
                            object_class_id: SegmentationClass(1),
                            ..Default::default()
                        }],
                    }),
                },
                PipelineStepSettings {
                    enabled: true,
                    command: PipelineCommand::ConnectedComponents(
                        ConnectedComponentsSettings::default(),
                    ),
                },
                PipelineStepSettings {
                    enabled: true,
                    command: PipelineCommand::ExtractObjects(ExtractObjectsSettings::default()),
                },
            ],
        }];

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
        std::fs::copy(fixture_dir.join("slice_Z0_C0_T0.tif"), images_dir.join("b.tif"))
            .expect("copy fixture b");

        run(AnalyzeArgs {
            project: file.path.clone(),
            images: Some(images_dir),
            threads: Some(1),
            job_name: Some("e2e".into()),
        })
        .expect("analyze should succeed against real fixture images with a working pipeline");

        let results_root = file.path.parent().unwrap().join("results");
        let evadb =
            find_evadb(&results_root).expect("analyze should have produced a .evadb file");

        let out_dir = tempfile::tempdir().expect("tempdir");

        // -- Ground truth: every detected object's full row, straight from
        // `ResultsGenerator` itself (the same source both `export csv` and
        // `export xlsx` read from) - swept across every z/t plane the same
        // way `export_table` does (`0..=get_nr_of_{z,t}_stacks()`), and
        // keyed by object id so row *order* differences between the direct
        // query and the exported files can't cause a false mismatch.
        let database = ResultsGenerator::open_database(evadb.clone()).expect("open evadb");
        let expected_columns: Vec<Column> = database
            .get_available_columns()
            .expect("available columns")
            .into_iter()
            .map(|entry| entry.key)
            .collect();
        let mut expected_by_id: std::collections::HashMap<String, Vec<String>> =
            std::collections::HashMap::new();
        for z in 0..=database.get_nr_of_z_stacks() {
            for t in 0..=database.get_nr_of_t_stacks() {
                let page = database
                    .get_object_list(&ListFilter {
                        plane: PlaneFilter { z_stack: z, t_stack: t },
                        images: None,
                        object_classes: None,
                        columns: expected_columns.clone(),
                        with_coloc_details: false,
                        page: Pagination { limit: 1_000_000, after: None },
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

        let csv_out = out_dir.path().join("out.csv");
        crate::commands::export::run(ExportArgs {
            command: ExportCommand::Csv(TableExportArgs {
                db: evadb.clone(),
                out: csv_out.clone(),
                filter: FilterArgs::default(),
                group: GroupArgs::default(),
            }),
        })
        .expect("csv export should succeed");
        let csv_content = std::fs::read_to_string(&csv_out).expect("read exported csv");
        let mut csv_lines = csv_content.lines();
        let csv_header: Vec<&str> = csv_lines.next().expect("csv header row").split(',').collect();
        assert_eq!(csv_header[0], "Object ID", "header: {csv_header:?}");
        let object_id_col = 0;

        let mut csv_row_count = 0;
        for line in csv_lines {
            csv_row_count += 1;
            let fields: Vec<&str> = line.split(',').collect();
            assert_eq!(
                fields.len(),
                expected_columns.len(),
                "csv row has a different column count than expected: {line}"
            );
            let id = fields[object_id_col];
            let expected_row = expected_by_id
                .get(id)
                .unwrap_or_else(|| panic!("csv row for object {id} has no ground-truth match"));
            for (col_idx, (field, expected)) in fields.iter().zip(expected_row).enumerate() {
                assert_values_match(field, expected, &csv_header[col_idx], "csv");
            }
        }
        assert_eq!(
            csv_row_count,
            expected_by_id.len(),
            "csv must contain exactly one row per detected object"
        );

        let xlsx_out = out_dir.path().join("out.xlsx");
        crate::commands::export::run(ExportArgs {
            command: ExportCommand::Xlsx(TableExportArgs {
                db: evadb,
                out: xlsx_out.clone(),
                filter: FilterArgs::default(),
                group: GroupArgs::default(),
            }),
        })
        .expect("xlsx export should succeed");

        use calamine::Reader;
        let mut workbook: calamine::Xlsx<_> =
            calamine::open_workbook(&xlsx_out).expect("open exported xlsx");
        let range = workbook.worksheet_range("List").expect("List sheet");
        let mut rows = range.rows();
        let xlsx_header = rows.next().expect("xlsx header row");
        assert_eq!(
            xlsx_header.first().and_then(|c| c.get_string()),
            Some("Object ID")
        );

        let mut xlsx_row_count = 0;
        for row in rows {
            xlsx_row_count += 1;
            let id = row[object_id_col]
                .get_string()
                .expect("object id cell should be a string");
            let expected_row = expected_by_id
                .get(id)
                .unwrap_or_else(|| panic!("xlsx row for object {id} has no ground-truth match"));
            for (col_idx, (cell, expected)) in row.iter().zip(expected_row).enumerate() {
                let got = match cell {
                    calamine::Data::String(s) => s.clone(),
                    calamine::Data::Float(f) => f.to_string(),
                    calamine::Data::Int(i) => i.to_string(),
                    calamine::Data::Empty => String::new(),
                    other => panic!("unexpected xlsx cell type: {other:?}"),
                };
                assert_values_match(&got, expected, &xlsx_header[col_idx].to_string(), "xlsx");
            }
        }
        assert_eq!(
            xlsx_row_count,
            expected_by_id.len(),
            "xlsx must contain exactly one row per detected object"
        );
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
