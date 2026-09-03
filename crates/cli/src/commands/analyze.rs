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
    use crate::commands::test_support::TempProjectFile;
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
}
