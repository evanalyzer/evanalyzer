//! End-to-end: a real server on a local port running `LocalBackend`, and a
//! `RemoteBackend` client talking to it over WebSocket. Remote results must
//! match what the local backend produces for the same request.

use evanalyzer_app::ProjectWithRuntime;
use evanalyzer_app::ai_learning::{
    PixelTrainingParams, StartTrainingError, TrainingItems, save_trained_model,
};
use evanalyzer_app::backend::{
    AnalysisRequest, Backend, ImageSource, LocalBackend, TileRequest, TrainingRequest,
};
use evanalyzer_app::extensions::project_ext::ProjectExt;
use evanalyzer_app::job::{PreviewRequest, PreviewViewport, ProgressEvent, RunningJob};
use evanalyzer_cfg::core_types::{
    ImageTile, InternalErrors, ObjectClass, TrainingProgressEvent, ZProjection,
};
use evanalyzer_cfg::settings::ai_learning_object_settings::{
    AiLearningObjectFeatureSettings, ObjectMetric,
};
use evanalyzer_cfg::settings::ai_learning_settings::{
    AiLearningBackendSettings, AiLearningClassifierSettings, AiLearningSettings, ObjectClassLabel,
};
use evanalyzer_cfg::settings::images_settings::{ImageEntry, SeriesSettings};
use evanalyzer_cfg::settings::object_settings::ObjectMetricSettings;
use evanalyzer_cfg::settings::project_settings::ProjectSettings;
use evanalyzer_net::{RemoteBackend, Server};
use std::path::PathBuf;
use std::sync::Arc;

const TOKEN: &str = "test-token";

fn start_server() -> String {
    let server = Server::bind("127.0.0.1:0", TOKEN.into()).unwrap();
    let addr = server.local_addr().unwrap();
    std::thread::spawn(move || server.run(Arc::new(LocalBackend)));
    format!("ws://{addr}")
}

fn connect() -> RemoteBackend {
    RemoteBackend::connect(&start_server(), TOKEN).unwrap()
}

fn fixture() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../core/tests/multi-channel-4D-series.ome.tif")
}

/// A project in its own temp dir whose image root holds one copy of the
/// fixture image, so jobs run end to end.
fn project_with_fixture_image() -> (tempfile::TempDir, ProjectSettings) {
    let dir = tempfile::tempdir().unwrap();
    let images_dir = dir.path().join("images");
    std::fs::create_dir_all(&images_dir).unwrap();
    std::fs::copy(fixture(), images_dir.join("fixture.ome.tif")).unwrap();
    let mut project = ProjectWithRuntime::default();
    project.images.root = Some(images_dir);
    project.scan_image_folder_and_add();
    assert_eq!(project.images.list.len(), 1);
    (dir, project.settings)
}

fn drain(job: &RunningJob) -> Vec<ProgressEvent> {
    job.events().iter().collect()
}

fn tile_request(source: &dyn ImageSource, offset: usize) -> TileRequest {
    let level = &source.meta().series[&0].resolutions[&0];
    let size = 32usize;
    TileRequest {
        series: 0,
        resolution_idx: 0,
        z_projection: ZProjection::None,
        z_range: None,
        t_stack: 0,
        tile: ImageTile {
            offset_x: offset.min(level.width as usize - size),
            offset_y: 0,
            width: size,
            height: size,
        },
    }
}

#[test]
fn a_wrong_token_is_rejected_with_a_readable_message() {
    let url = start_server();
    let error = match RemoteBackend::connect(&url, "wrong") {
        Ok(_) => panic!("connection with a wrong token must fail"),
        Err(e) => e.to_string(),
    };
    assert!(error.contains("invalid token"), "{error}");
    // The server survives a rejected client.
    RemoteBackend::connect(&url, TOKEN).unwrap();
}

#[test]
fn connecting_to_a_closed_port_fails_instead_of_hanging() {
    let port = std::net::TcpListener::bind("127.0.0.1:0")
        .unwrap()
        .local_addr()
        .unwrap()
        .port();
    assert!(RemoteBackend::connect(&format!("ws://127.0.0.1:{port}"), TOKEN).is_err());
}

#[test]
fn non_ws_urls_are_rejected_up_front() {
    for url in ["wss://host:7400", "http://host:7400", "not a url"] {
        assert!(RemoteBackend::connect(url, TOKEN).is_err(), "{url}");
    }
}

#[test]
fn remote_image_metadata_and_pixels_match_local_reads() {
    let remote = connect();
    let remote_image = remote.open_image(&fixture()).unwrap();
    let local_image = LocalBackend.open_image(&fixture()).unwrap();

    let (remote_meta, local_meta) = (remote_image.meta(), local_image.meta());
    assert_eq!(remote_meta.name, local_meta.name);
    assert_eq!(remote_meta.series.len(), local_meta.series.len());
    let (r0, l0) = (&remote_meta.series[&0], &local_meta.series[&0]);
    assert_eq!(r0.nr_c_stacks, l0.nr_c_stacks);
    assert_eq!(r0.resolutions[&0].width, l0.resolutions[&0].width);

    let req = tile_request(local_image.as_ref(), 0);
    let remote_tile = remote_image.read_tile(&req).unwrap();
    let local_tile = local_image.read_tile(&req).unwrap();
    assert_eq!(remote_tile.len(), local_tile.len());
    for (r, l) in remote_tile.iter().zip(&local_tile) {
        assert_eq!(r.name, l.name);
        assert_eq!(r.c_stack, l.c_stack);
        assert_eq!(r.color, l.color);
        assert_eq!(r.image.tile_offset(), l.image.tile_offset());
        assert_eq!(r.image.as_f32_slice(), l.image.as_f32_slice());
    }
}

#[test]
fn concurrent_tile_reads_share_one_connection() {
    let remote = Arc::new(connect());
    let image = remote.open_image(&fixture()).unwrap();
    let local = LocalBackend.open_image(&fixture()).unwrap();
    let threads: Vec<_> = (0..8)
        .map(|i| {
            let image = Arc::clone(&image);
            let local = Arc::clone(&local);
            std::thread::spawn(move || {
                let req = tile_request(local.as_ref(), i * 8);
                let remote_tile = image.read_tile(&req).unwrap();
                let local_tile = local.read_tile(&req).unwrap();
                assert_eq!(
                    remote_tile[0].image.as_f32_slice(),
                    local_tile[0].image.as_f32_slice()
                );
            })
        })
        .collect();
    for t in threads {
        t.join().unwrap();
    }
}

#[test]
fn opening_a_missing_image_reports_the_servers_error() {
    let remote = connect();
    let error = match remote.open_image(&PathBuf::from("/no/such/image.tif")) {
        Ok(_) => panic!("opening a missing file must fail"),
        Err(e) => e.to_string(),
    };
    assert!(error.contains("Server"), "{error}");
}

#[test]
fn remote_analysis_runs_to_completion_like_a_local_one() {
    let remote = connect();
    let (dir, settings) = project_with_fixture_image();
    let job = remote
        .start_analysis(AnalysisRequest {
            settings,
            project_path: dir.path().to_path_buf(),
            job_name: Some("remote_test".into()),
            threads: Some(1),
        })
        .unwrap();
    assert_eq!(job.parallelism(), 1);
    assert!(job.output_path().starts_with(dir.path().join("results")));

    let events = drain(&job);
    assert!(matches!(
        events.first(),
        Some(ProgressEvent::Started { total: 1 })
    ));
    assert!(matches!(events.last(), Some(ProgressEvent::Finished)));
    assert!(job.wait().unwrap().preview_objects.is_none());
    assert!(dir.path().join("results").exists());
}

#[test]
fn remote_preview_returns_its_final_objects_from_wait() {
    let remote = connect();
    let (dir, settings) = project_with_fixture_image();
    let job = remote
        .start_preview(PreviewRequest {
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
    assert!(matches!(output.preview_objects, Some(objects) if objects.is_empty()));
}

#[test]
fn a_remote_cancel_never_surfaces_as_a_different_error() {
    let remote = connect();
    let (dir, settings) = project_with_fixture_image();
    let job = remote
        .start_analysis(AnalysisRequest {
            settings,
            project_path: dir.path().to_path_buf(),
            job_name: None,
            threads: Some(1),
        })
        .unwrap();
    job.cancel_handle().cancel();
    assert!(job.cancel_handle().is_cancelled());
    drain(&job);
    match job.wait() {
        Ok(_) | Err(InternalErrors::Cancelled) => {}
        Err(e) => panic!("unexpected error: {e:?}"),
    }
}

fn two_class_object_settings() -> AiLearningSettings {
    AiLearningSettings {
        schema_version: evanalyzer_cfg::CURRENT_AI_LEARNING_SETTINGS_SCHEMA_VERSION,
        meta: Default::default(),
        backend: AiLearningBackendSettings::RandomForest(Default::default()),
        classifier: AiLearningClassifierSettings::Object {
            feature_spec: AiLearningObjectFeatureSettings {
                metrics: vec![ObjectMetric::Area],
            },
            class_labels: vec![
                ObjectClassLabel {
                    class: ObjectClass::Valid(1),
                    name: "A".into(),
                },
                ObjectClassLabel {
                    class: ObjectClass::Valid(2),
                    name: "B".into(),
                },
            ],
        },
    }
}

/// Object training reads only already-computed metrics, so no image files
/// are needed.
fn project_with_labeled_objects(classes: &[u32]) -> ProjectSettings {
    let mut series = SeriesSettings::default();
    for (i, class) in classes.iter().enumerate() {
        series.objects.push(ObjectMetricSettings {
            area: 10 + 1000 * i,
            object_class: [ObjectClass::Valid(*class)].into(),
            ..Default::default()
        });
    }
    let mut entry = ImageEntry::default();
    entry.series.insert(0, series);
    let mut project = ProjectSettings::default();
    project.images.list.insert(PathBuf::from("img.tif"), entry);
    project
}

#[test]
fn training_without_labeled_data_maps_to_no_training_data() {
    let remote = connect();
    let result = remote.start_training(TrainingRequest {
        project: ProjectSettings::default(),
        settings: two_class_object_settings(),
        pixel_params: PixelTrainingParams::default(),
    });
    assert!(matches!(result, Err(StartTrainingError::NoTrainingData)));
}

#[test]
fn a_remotely_trained_model_comes_back_and_saves_like_a_local_one() {
    let remote = connect();
    let training = remote
        .start_training(TrainingRequest {
            project: project_with_labeled_objects(&[1, 2]),
            settings: two_class_object_settings(),
            pixel_params: PixelTrainingParams::default(),
        })
        .unwrap();
    assert_eq!(training.items(), TrainingItems::Objects(2));
    let events: Vec<_> = training.events().iter().collect();
    assert!(
        events
            .iter()
            .any(|e| matches!(e, TrainingProgressEvent::Finished { .. }))
    );
    let model = training.wait().unwrap();

    let dir = tempfile::tempdir().unwrap();
    let path = save_trained_model(&model, dir.path(), "remote").unwrap();
    let settings = evanalyzer_app::ai_learning::load_classifier_settings(&path).unwrap();
    assert_eq!(
        serde_json::to_value(&settings).unwrap(),
        serde_json::to_value(two_class_object_settings()).unwrap()
    );
}

#[test]
fn one_connection_serves_several_jobs_in_a_row() {
    let remote = connect();
    for _ in 0..3 {
        let (dir, settings) = project_with_fixture_image();
        let job = remote
            .start_analysis(AnalysisRequest {
                settings,
                project_path: dir.path().to_path_buf(),
                job_name: None,
                threads: Some(1),
            })
            .unwrap();
        drain(&job);
        job.wait().unwrap();
    }
}

#[test]
fn dropping_a_client_does_not_affect_the_next_one() {
    let url = start_server();
    {
        let first = RemoteBackend::connect(&url, TOKEN).unwrap();
        let _image = first.open_image(&fixture()).unwrap();
    }
    let second = RemoteBackend::connect(&url, TOKEN).unwrap();
    let image = second.open_image(&fixture()).unwrap();
    image.read_tile(&tile_request(image.as_ref(), 0)).unwrap();
}
