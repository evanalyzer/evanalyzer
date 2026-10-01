//! Starting and driving analysis/preview jobs - the one place that knows how
//! to turn `ProjectSettings` into a running `evanalyzer_core::JobExecutor`.
//!
//! Both the GUI and the CLI go through here instead of building jobs from
//! `evanalyzer_core` themselves, so job setup (parallelism, preview tile
//! limits, breakpoints, panic handling) can't drift between the two. Front
//! ends only consume the resulting [`ProgressEvent`](evanalyzer_core::ProgressEvent)
//! stream and present it.
mod analysis;
mod preview;
mod running_job;

pub use analysis::start_analysis;
pub use preview::{
    MAX_PREVIEW_VISIBLE_TILES, PreviewRequest, PreviewViewport, StartPreviewError, start_preview,
};
pub use running_job::{CancelHandle, JobOutput, RunningJob};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ProjectWithRuntime;
    use crate::extensions::project_ext::ProjectExt;
    use evanalyzer_cfg::core_types::InternalErrors;
    use evanalyzer_cfg::settings::project_settings::ProjectSettings;
    use evanalyzer_core::ProgressEvent;

    /// A project in its own temp dir whose image root holds exactly one copy
    /// of the real `core` fixture image, so jobs run end to end.
    fn project_with_fixture_image() -> (tempfile::TempDir, ProjectSettings) {
        let dir = tempfile::tempdir().unwrap();
        let images_dir = dir.path().join("images");
        std::fs::create_dir_all(&images_dir).unwrap();
        let fixture = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../core/tests/multi-channel-4D-series.ome.tif");
        std::fs::copy(&fixture, images_dir.join("fixture.ome.tif")).unwrap();

        let mut project = ProjectWithRuntime::default();
        project.images.root = Some(images_dir);
        project.scan_image_folder_and_add();
        assert_eq!(project.images.list.len(), 1);
        (dir, project.settings)
    }

    fn drain(job: &RunningJob) -> Vec<ProgressEvent> {
        job.events().iter().collect()
    }

    #[test]
    fn start_analysis_runs_to_completion_and_writes_under_results() {
        let (dir, settings) = project_with_fixture_image();

        let job = start_analysis(
            settings,
            dir.path().to_path_buf(),
            Some("app_job_test".into()),
            Some(1),
        )
        .unwrap();
        assert_eq!(job.parallelism(), 1);
        assert!(job.output_path().starts_with(dir.path().join("results")));

        let events = drain(&job);
        assert!(matches!(
            events.first(),
            Some(ProgressEvent::Started { total: 1 })
        ));
        assert!(matches!(events.last(), Some(ProgressEvent::Finished)));
        let output = job.wait().unwrap();
        assert!(output.preview_objects.is_none());
    }

    #[test]
    fn start_analysis_picks_a_parallelism_when_none_is_given() {
        let (dir, settings) = project_with_fixture_image();
        let job = start_analysis(settings, dir.path().to_path_buf(), None, None).unwrap();
        assert!(job.parallelism() >= 1);
        drain(&job);
        job.wait().unwrap();
    }

    #[test]
    fn analysis_cancelled_before_it_starts_returns_cancelled() {
        let (dir, settings) = project_with_fixture_image();
        let job = start_analysis(settings, dir.path().to_path_buf(), None, Some(1)).unwrap();
        job.cancel_handle().cancel();
        drain(&job);
        // The job may already have finished its single image before seeing
        // the flag - either outcome is fine, but a cancel must never surface
        // as a different error (the GUI shows that as a failure).
        match job.wait() {
            Ok(_) | Err(InternalErrors::Cancelled) => {}
            Err(e) => panic!("unexpected error: {e:?}"),
        }
    }

    #[test]
    fn start_preview_returns_final_objects_from_wait() {
        let (dir, settings) = project_with_fixture_image();

        let job = start_preview(PreviewRequest {
            settings,
            project_path: dir.path().to_path_buf(),
            viewport: PreviewViewport {
                offset_x: 0.0,
                offset_y: 0.0,
                viewport_width: 256.0,
                viewport_height: 256.0,
                zoom: 1.0,
            },
            breakpoint: None,
        })
        .unwrap();
        drain(&job);
        let output = job.wait().unwrap();
        // No pipelines configured, so no objects - but the preview store is there.
        assert!(matches!(output.preview_objects, Some(objects) if objects.is_empty()));
    }
}
