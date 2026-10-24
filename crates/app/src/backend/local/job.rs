//! Starting and driving analysis/preview jobs - the one place that knows how
//! to turn `ProjectSettings` into a running `evanalyzer_core::JobExecutor`.
//!
//! Front ends reach this through [`crate::api::Backend`] (the local
//! backend calls straight into here) instead of building jobs from
//! `evanalyzer_core` themselves, so job setup (parallelism, preview tile
//! limits, breakpoints, panic handling) can't drift between the two. Front
//! ends only consume the resulting [`ProgressEvent`] stream and present it.
//!
//! [`CancelHandle`] and `join_job` are shared with classifier training
//! ([`crate::backend::local::training::start_training`]).
mod analysis;
mod preview;
mod running_job;

pub(crate) use analysis::start_analysis;
pub(crate) use preview::start_preview;
pub(crate) use running_job::{join_job, spawn_job};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::{PreviewRequest, PreviewViewport, RunningJob};
    use crate::backend::LocalBackend;
    use crate::workspace::ProjectWithRuntime;
    use crate::workspace::extensions::project_ext::ProjectExt;
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
        project.scan_image_folder_and_add(&LocalBackend::default());
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
            None,
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
        let job = start_analysis(settings, dir.path().to_path_buf(), None, None, None).unwrap();
        assert!(job.parallelism() >= 1);
        drain(&job);
        job.wait().unwrap();
    }

    #[test]
    fn analysis_cancelled_before_it_starts_returns_cancelled() {
        let (dir, settings) = project_with_fixture_image();
        let job = start_analysis(settings, dir.path().to_path_buf(), None, Some(1), None).unwrap();
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
            max_visible_tiles: None,
            breakpoint_tiles: 1,
            whole_image: false,
        })
        .unwrap();
        drain(&job);
        let output = job.wait().unwrap();
        // No pipelines configured, so no objects - but the preview store is there.
        assert!(matches!(output.preview_objects, Some(objects) if objects.is_empty()));
    }

    /// A preview brings its measurements back with its objects.
    #[test]
    fn start_preview_returns_the_measurements_too() {
        use evanalyzer_cfg::core_types::{
            ImageAddress, MeasurementRegion, MeasurementValues, PipelineId, StepId,
        };
        use evanalyzer_cfg::settings::pipeline_command::PipelineCommand;
        use evanalyzer_cfg::settings::pipeline_settings::{
            PipelineSettings, PipelineStepSettings, StepContent,
        };
        let (dir, mut settings) = project_with_fixture_image();
        let mut pearson = PipelineCommand::default_for_key("measure_colocalization").unwrap();
        *pearson.measurement_id_mut().unwrap() = "pearson_01".into();
        settings.pipelines.push(PipelineSettings {
            id: PipelineId(1),
            name: "Coloc".into(),
            description: None,
            image_source: ImageAddress::Channel(0),
            enabled: true,
            steps: vec![PipelineStepSettings {
                id: StepId(1),
                enabled: true,
                content: StepContent::Command(pearson),
            }],
        });

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
            max_visible_tiles: None,
            breakpoint_tiles: 1,
            whole_image: false,
        })
        .unwrap();
        drain(&job);
        let output = job.wait().unwrap();
        let planes = output
            .preview_measurements
            .expect("a preview's measurements");
        let [plane] = &planes[..] else {
            panic!("one plane, got {}", planes.len());
        };
        // Measure Colocalization: Pearson among its values.
        let measurement = plane
            .measurements
            .iter()
            .find(|m| m.info.key == "pearson_01/pearson")
            .expect("Pearson measured");
        let MeasurementValues::Image(values) = &measurement.values else {
            panic!("an image measurement");
        };
        assert!(
            matches!(values[..], [(MeasurementRegion::Image, r)] if (-1.0..=1.0).contains(&r)),
            "{values:?}"
        );
    }

    /// The tiles a preview of the fixture image runs.
    fn tiles_run(whole_image: bool, max_visible_tiles: Option<u32>) -> usize {
        let (dir, settings) = project_with_fixture_image();
        let job = start_preview(PreviewRequest {
            settings,
            project_path: dir.path().to_path_buf(),
            // Beside the image: none of it in view.
            viewport: PreviewViewport {
                offset_x: 100_000.0,
                offset_y: 100_000.0,
                viewport_width: 256.0,
                viewport_height: 256.0,
                zoom: 8.0,
            },
            breakpoint: None,
            max_visible_tiles,
            breakpoint_tiles: 1,
            whole_image,
        })
        .unwrap();
        let events = drain(&job);
        job.wait().unwrap();
        events
            .iter()
            .filter(|e| matches!(e, ProgressEvent::TileCompleted { .. }))
            .count()
    }

    #[test]
    fn a_whole_image_preview_runs_every_tile_also_out_of_view() {
        assert_eq!(tiles_run(false, None), 0, "nothing in view");
        assert_eq!(tiles_run(true, None), 1, "the image's one tile");
        // The viewport's limit isn't the whole image's.
        assert_eq!(tiles_run(true, Some(1)), 1);
    }
}
