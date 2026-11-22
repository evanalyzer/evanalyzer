//! # job
//!
//! Starting and driving analysis/preview jobs - the one place that knows how
//! to turn `ProjectSettings` into a running `evanalyzer_core::JobExecutor`.
//!
//! Front ends reach this through [`crate::backend::Backend`] (the local
//! backend calls straight into here) instead of building jobs from
//! `evanalyzer_core` themselves, so job setup (parallelism, preview tile
//! limits, breakpoints, panic handling) can't drift between the two. Front
//! ends only consume the resulting [`ProgressEvent`] stream and present it.
//!
//! [`CancelHandle`] and `join_job` are shared with classifier training
//! ([`crate::training::local::start_training`]).
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-10-10
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.

mod analysis;
mod preview;
mod running_job;

pub(crate) use analysis::start_analysis;
pub(crate) use preview::start_preview;
pub(crate) use running_job::{join_job, spawn_job};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::LocalBackend;
    use crate::jobs::{PreviewRequest, PreviewViewport, RunningJob};
    use crate::project::ProjectWithRuntime;
    use crate::project::images::ImageListExt;
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
            prediction: None,
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
        let mut pearson = PipelineCommand::default_for_key("measure_image_colocalization").unwrap();
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
            prediction: None,
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

    /// A preview of the fixture image's first plane: its measurements and
    /// why it was left out (if it was).
    fn preview_plane(
        settings: ProjectSettings,
        dir: &tempfile::TempDir,
    ) -> evanalyzer_cfg::core_types::PipelineMeasurements {
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
            prediction: None,
            max_visible_tiles: None,
            breakpoint_tiles: 1,
            whole_image: false,
        })
        .unwrap();
        drain(&job);
        let output = job.wait().unwrap();
        let mut planes = output
            .preview_measurements
            .expect("a preview's measurements");
        assert_eq!(planes.len(), 1);
        planes.remove(0)
    }

    /// A detection module with a Threshold - its ID empty, as in a library
    /// file - used in a pipeline, twice inside another module, and again
    /// with the same class: each use measures under a name of its own, an
    /// Exclude rule in the module picks its own use's threshold, one in the
    /// pipeline the threshold of a use inside the other module.
    #[test]
    fn a_module_used_several_times_and_nested_measures_once_per_use() {
        use evanalyzer_cfg::settings::modules::ModuleDefinition;
        use evanalyzer_cfg::settings::pipeline_settings::PipelineSettings;
        let (dir, mut settings) = project_with_fixture_image();
        let meta =
            serde_json::to_value(evanalyzer_cfg::settings::meta_data::MetaData::default()).unwrap();
        let ev: ModuleDefinition = serde_json::from_value(serde_json::json!({
            "key": "EV",
            "meta": meta.clone(),
            "steps": [
                {"id": 4, "enabled": true, "command": {"type": "THRESHOLD", "thresholds": [
                    {"method": {"type": "MANUAL"}, "minThreshold": 100.0,
                     "maxThreshold": 65535.0, "unit": "BIT", "segmentationClass": 1,
                     "valueSource": "ACTUAL_IMAGE"}]}},
                {"id": 5, "enabled": true, "command": {"type": "CONNECTED_COMPONENTS", "minSize": 1}},
                {"id": 6, "enabled": true, "command": {"type": "EXTRACT_OBJECTS",
                    "maxObjectsBeforeFail": 1000000,
                    "classMapping": [{"segmentationClass": 1, "objectClass": {"GLOBAL": 1}}]}},
                {"id": 8, "enabled": true, "command": {"type": "EXCLUDE_IMAGE",
                    "excludeWhen": "ANY_RULE",
                    "rules": [{"measurement": "threshold_01/threshold_1",
                               "comparison": "ABOVE", "value": 1.0e12}]}}
            ],
            "parameters": [{"key": "evs", "label": "EVs", "default": "1",
                "bindings": [{"step": 6, "param": "class_mapping.0.object_class"}]}]
        }))
        .unwrap();
        let coloc: ModuleDefinition = serde_json::from_value(serde_json::json!({
            "key": "COLOC",
            "meta": meta,
            "steps": [
                {"id": 1, "enabled": true, "module": {"key": "EV", "values": {"evs": "15"}}},
                {"id": 2, "enabled": true, "module": {"key": "EV", "values": {"evs": "16"}}}
            ],
            "parameters": []
        }))
        .unwrap();
        let pipeline: PipelineSettings = serde_json::from_value(serde_json::json!({
            "id": 1, "name": "EVs", "imageSource": {"CHANNEL": 0}, "enabled": true,
            "steps": [
                {"id": 1, "enabled": true, "module": {"key": "EV", "values": {}}},
                {"id": 2, "enabled": true, "module": {"key": "COLOC", "values": {}}},
                {"id": 3, "enabled": true, "module": {"key": "EV", "values": {}}},
                {"id": 4, "enabled": true, "command": {"type": "EXCLUDE_IMAGE",
                    "excludeWhen": "ANY_RULE",
                    "rules": [{"measurement": "@1.2.2.4:threshold_01/threshold_1",
                               "comparison": "BELOW", "value": 1.0e12}]}}
            ]
        }))
        .unwrap();
        settings.modules = vec![ev, coloc];
        settings.pipelines = vec![pipeline];
        for (id, name) in [(1, "EVs"), (15, "EVs CY3"), (16, "EVs CY5")] {
            if let Some(class) = settings
                .classification
                .classes_mut()
                .iter_mut()
                .find(|c| c.id == id)
            {
                class.name = name.into();
            } else {
                settings.classification.classes_mut().push(
                    evanalyzer_cfg::settings::classification_settings::Class {
                        id,
                        name: name.into(),
                        ..Default::default()
                    },
                );
            }
        }
        // As loading a project does: the module's Threshold gets its ID.
        crate::project::utils::assign_measurement_ids(&mut settings);
        let problems = crate::modules::checks::check_project(&settings);
        assert!(
            problems
                .iter()
                .all(|p| p.severity != crate::modules::checks::Severity::Error),
            "{problems:?}"
        );

        let plane = preview_plane(settings, &dir);
        let mut steps: Vec<&str> = plane
            .measurements
            .iter()
            .map(|m| m.info.step.as_str())
            .collect();
        steps.dedup();
        assert_eq!(
            steps,
            [
                "threshold_evs",
                "threshold_evs_cy3",
                "threshold_evs_cy5",
                "threshold_evs_2"
            ]
        );
        // The pipeline's rule picked the second use inside Coloc.
        let reason = plane.excluded.expect("left out by the pipeline's rule");
        assert!(reason.contains("threshold_evs_cy5"), "{reason}");
        assert!(!reason.contains("threshold_evs_cy3"), "{reason}");
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
            prediction: None,
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
