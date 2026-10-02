//! End-to-end: a real server on a local port running `LocalBackend`, and a
//! `RemoteBackend` client talking to it over WebSocket. Remote results must
//! match what the local backend produces for the same request.

use evanalyzer_app::ai_learning::PixelTrainingParams;
use evanalyzer_app::ai_learning::StartTrainingError;
use evanalyzer_app::ai_learning::TrainingItems;
use evanalyzer_app::ai_learning::TrainingRequest;
use evanalyzer_app::ai_learning::save_trained_model;
use evanalyzer_app::analysis::AnalysisRequest;
use evanalyzer_app::analysis::ProgressEvent;
use evanalyzer_app::analysis::RunningJob;
use evanalyzer_app::backends::Backend;
use evanalyzer_app::backends::local::LocalBackend;
use evanalyzer_app::backends::remote::RemoteBackend;
use evanalyzer_app::backends::remote::Server;
use evanalyzer_app::images::ImageSource;
use evanalyzer_app::images::TileRequest;
use evanalyzer_app::preview::PreviewRequest;
use evanalyzer_app::preview::PreviewViewport;
use evanalyzer_app::project::ProjectExt;
use evanalyzer_app::project::ProjectWithRuntime;
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
use std::path::PathBuf;
use std::sync::Arc;

const TOKEN: &str = "test-token";

fn start_server() -> String {
    let server = Server::bind("127.0.0.1:0", TOKEN.into()).unwrap();
    let addr = server.local_addr().unwrap();
    std::thread::spawn(move || server.run(Arc::new(LocalBackend::default())));
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
    project.scan_image_folder_and_add(&LocalBackend::default());
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
    let local_image = LocalBackend::default().open_image(&fixture()).unwrap();

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
    let local = LocalBackend::default().open_image(&fixture()).unwrap();
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
    let path = save_trained_model(remote.files(), &model, dir.path(), "remote").unwrap();
    let settings =
        evanalyzer_app::ai_learning::load_classifier_settings(remote.files(), &path).unwrap();
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

// -- File access ----------------------------------------------------------

fn start_restricted_server(root: &std::path::Path) -> String {
    let server = Server::bind("127.0.0.1:0", TOKEN.into()).unwrap();
    let addr = server.local_addr().unwrap();
    let backend = LocalBackend::restricted_to(&[root.to_path_buf()]).unwrap();
    std::thread::spawn(move || server.run(Arc::new(backend)));
    format!("ws://{addr}")
}

#[test]
fn remote_files_list_read_and_write_like_local_ones() {
    let remote = connect();
    let dir = tempfile::tempdir().unwrap();
    std::fs::create_dir(dir.path().join("sub")).unwrap();
    std::fs::write(dir.path().join("a.txt"), b"hello").unwrap();

    let files = remote.files();
    assert_eq!(
        files.list_dir(dir.path()).unwrap(),
        LocalBackend::default()
            .files()
            .list_dir(dir.path())
            .unwrap()
    );
    assert_eq!(
        files.read_file(&dir.path().join("a.txt")).unwrap(),
        b"hello"
    );

    let target = dir.path().join("sub/deeper/project.evaproj");
    let big = vec![42u8; 3 * 1024 * 1024];
    files.write_file(&target, &big).unwrap();
    assert_eq!(std::fs::read(&target).unwrap(), big);
    assert!(files.stat(&target).unwrap().is_some());
    assert!(files.stat(&dir.path().join("missing")).unwrap().is_none());

    files.create_dir_all(&dir.path().join("x/y")).unwrap();
    assert!(dir.path().join("x/y").is_dir());
    assert!(!files.places().unwrap().is_empty());
}

#[test]
fn a_restricted_server_refuses_paths_outside_its_roots() {
    let root = tempfile::tempdir().unwrap();
    let outside = tempfile::tempdir().unwrap();
    std::fs::write(outside.path().join("secret.txt"), b"x").unwrap();
    let remote = RemoteBackend::connect(&start_restricted_server(root.path()), TOKEN).unwrap();
    let files = remote.files();

    let places = files.places().unwrap();
    assert_eq!(places.len(), 1);
    assert_eq!(places[0].path, root.path().canonicalize().unwrap());

    let error = files
        .read_file(&outside.path().join("secret.txt"))
        .unwrap_err()
        .to_string();
    assert!(error.contains("not allowed"), "{error}");
    assert!(files.list_dir(outside.path()).is_err());
    assert!(
        files
            .write_file(&outside.path().join("x.txt"), b"no")
            .is_err()
    );
    assert!(remote.open_image(&fixture()).is_err());
    files
        .write_file(&root.path().join("ok.txt"), b"yes")
        .unwrap();

    // Jobs whose project folder lies outside are refused too.
    let (dir, settings) = project_with_fixture_image();
    let result = remote.start_analysis(AnalysisRequest {
        settings,
        project_path: dir.path().to_path_buf(),
        job_name: None,
        threads: Some(1),
    });
    assert!(result.is_err());
}

// -- Project files, folder scans, templates ---------------------------------

#[test]
fn projects_save_and_load_through_the_server() {
    let remote = connect();
    let (dir, settings) = project_with_fixture_image();
    let path = dir.path().join("remote.evaproj");

    let mut project = ProjectWithRuntime::default();
    project.settings = settings;
    project.save_project_as(remote.files(), &path).unwrap();

    let loaded = evanalyzer_app::project::load_project(remote.files(), &path).unwrap();
    assert_eq!(loaded.images.list.len(), 1);
    assert_eq!(loaded.tmp_settings.current_project, Some(path));
}

#[test]
fn a_remote_folder_scan_finds_the_same_images_as_a_local_one() {
    let remote = connect();
    let (dir, _) = project_with_fixture_image();
    let scan = |backend: &dyn Backend| {
        let mut found: Vec<(PathBuf, String)> =
            evanalyzer_app::project::collect_images_at_root(backend, dir.path())
                .into_iter()
                .map(|(path, meta)| (path, meta.name))
                .collect();
        found.sort();
        found
    };
    let remote_found = scan(&remote);
    assert_eq!(remote_found.len(), 1);
    assert_eq!(remote_found, scan(&LocalBackend::default()));
}

#[test]
fn template_folders_are_the_servers() {
    let remote = connect();
    assert_eq!(
        remote.template_folders().unwrap(),
        LocalBackend::default().template_folders().unwrap()
    );
}

// -- Results databases -------------------------------------------------------

/// Runs an analysis of the fixture image through `backend` and returns the
/// `.evadb` it wrote (plus the temp dir keeping it alive).
fn analyzed_database(backend: &dyn Backend) -> (tempfile::TempDir, PathBuf) {
    let (dir, settings) = project_with_fixture_image();
    let job = backend
        .start_analysis(AnalysisRequest {
            settings,
            project_path: dir.path().to_path_buf(),
            job_name: Some("results_test".into()),
            threads: Some(1),
        })
        .unwrap();
    drain(&job);
    let output = job.output_path().clone();
    job.wait().unwrap();
    let db = std::fs::read_dir(&output)
        .unwrap()
        .map(|e| e.unwrap().path())
        .find(|p| p.extension().is_some_and(|e| e == "evadb"))
        .expect("analysis writes an .evadb");
    (dir, db)
}

#[test]
fn remote_results_queries_match_local_ones() {
    let remote = connect();
    let (_dir, db_path) = analyzed_database(&remote);
    let remote_db = remote.open_results(&db_path).unwrap();
    let local_db = LocalBackend::default().open_results(&db_path).unwrap();

    let names = |db: &dyn evanalyzer_app::results::ResultsSource| -> Vec<String> {
        db.get_images()
            .unwrap()
            .into_iter()
            .map(|i| i.name)
            .collect()
    };
    assert_eq!(names(remote_db.as_ref()), names(local_db.as_ref()));
    assert!(!names(remote_db.as_ref()).is_empty());
    assert_eq!(
        remote_db.get_nr_of_z_stacks(),
        local_db.get_nr_of_z_stacks()
    );
    assert_eq!(
        remote_db.get_nr_of_t_stacks(),
        local_db.get_nr_of_t_stacks()
    );
    let columns = |db: &dyn evanalyzer_app::results::ResultsSource| -> Vec<String> {
        db.get_available_columns()
            .unwrap()
            .into_iter()
            .map(|c| c.display_name)
            .collect()
    };
    assert_eq!(columns(remote_db.as_ref()), columns(local_db.as_ref()));

    let filter = evanalyzer_app::results::ListFilter {
        plane: evanalyzer_app::results::PlaneFilter {
            z_stack: 0,
            t_stack: 0,
        },
        images: None,
        object_classes: None,
        columns: vec![evanalyzer_app::results::Column::AreaSizePx],
        with_coloc_details: false,
        page: evanalyzer_app::results::Pagination {
            limit: 100,
            after: None,
        },
        transpond_table: false,
    };
    let (remote_list, local_list) = (
        remote_db.get_object_list(&filter).unwrap(),
        local_db.get_object_list(&filter).unwrap(),
    );
    assert_eq!(remote_list.column_names, local_list.column_names);
    assert_eq!(remote_list.row_names, local_list.row_names);
    assert_eq!(
        remote_list.source_object_count,
        local_list.source_object_count
    );
}

#[test]
fn non_finite_chart_values_survive_the_trip() {
    let remote = connect();
    let (_dir, db_path) = analyzed_database(&remote);
    let remote_db = remote.open_results(&db_path).unwrap();
    let local_db = LocalBackend::default().open_results(&db_path).unwrap();
    // No objects at all: the scatter range keeps its +/- infinity start
    // values, which JSON couldn't carry.
    let filter = evanalyzer_app::results::ScatterFilter {
        plane: evanalyzer_app::results::PlaneFilter {
            z_stack: 0,
            t_stack: 0,
        },
        images: None,
        object_classes: None,
        x_column: evanalyzer_app::results::Column::AreaSizePx,
        y_column: evanalyzer_app::results::Column::PerimeterPx,
        max_points: Some(10),
    };
    let (remote_scatter, local_scatter) = (
        remote_db.scatter(&filter).unwrap(),
        local_db.scatter(&filter).unwrap(),
    );
    assert_eq!(remote_scatter.points.len(), local_scatter.points.len());
    for (r, l) in [
        (remote_scatter.x_min, local_scatter.x_min),
        (remote_scatter.x_max, local_scatter.x_max),
        (remote_scatter.y_min, local_scatter.y_min),
        (remote_scatter.y_max, local_scatter.y_max),
    ] {
        assert!(r == l || (r.is_nan() && l.is_nan()), "{r} vs {l}");
    }
}

#[test]
fn a_remote_export_writes_its_files_on_the_server_and_reports_progress() {
    let remote = connect();
    let (dir, db_path) = analyzed_database(&remote);
    let remote_db = remote.open_results(&db_path).unwrap();
    let out = dir.path().join("export");
    let export = evanalyzer_app::results::ResultExport {
        output_dir: out.clone(),
        format: evanalyzer_app::results::ExportFormat::CSV,
        z_stacks: std::range::Range { start: 0, end: 1 },
        t_stacks: std::range::Range { start: 0, end: 1 },
        columns: vec![evanalyzer_app::results::Column::AreaSizePx],
        with_list_view: true,
        ..Default::default()
    };
    let mut progress_calls = 0;
    remote_db
        .export(
            &export,
            &std::sync::atomic::AtomicBool::new(false),
            &mut |_, _, _| progress_calls += 1,
        )
        .unwrap();
    assert!(out.join("list.csv").exists());
    assert!(progress_calls > 0);
}

#[test]
fn a_cancelled_remote_export_reports_cancelled() {
    let remote = connect();
    let (dir, db_path) = analyzed_database(&remote);
    let remote_db = remote.open_results(&db_path).unwrap();
    let export = evanalyzer_app::results::ResultExport {
        output_dir: dir.path().join("export"),
        format: evanalyzer_app::results::ExportFormat::CSV,
        with_list_view: true,
        ..Default::default()
    };
    let result = remote_db.export(
        &export,
        &std::sync::atomic::AtomicBool::new(true),
        &mut |_, _, _| {},
    );
    assert!(
        matches!(result, Err(InternalErrors::Cancelled) | Ok(())),
        "{result:?}"
    );
}

#[test]
fn a_dropped_connection_shows_as_disconnected() {
    // Relay between client and server that the test can cut.
    let server_url = start_server();
    let server_addr = server_url.trim_start_matches("ws://").to_string();
    let relay = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let relay_url = format!("ws://{}", relay.local_addr().unwrap());
    let (cut_tx, cut_rx) = std::sync::mpsc::channel::<std::net::TcpStream>();
    std::thread::spawn(move || {
        let (client, _) = relay.accept().unwrap();
        let upstream = std::net::TcpStream::connect(server_addr).unwrap();
        cut_tx.send(client.try_clone().unwrap()).unwrap();
        let (mut client_r, mut upstream_w) =
            (client.try_clone().unwrap(), upstream.try_clone().unwrap());
        std::thread::spawn(move || std::io::copy(&mut client_r, &mut upstream_w));
        let (mut upstream_r, mut client_w) = (upstream, client);
        let _ = std::io::copy(&mut upstream_r, &mut client_w);
    });

    let remote = RemoteBackend::connect(&relay_url, TOKEN).unwrap();
    assert!(remote.is_remote());
    assert!(remote.is_connected());
    assert_eq!(remote.user(), None, "token connections have no login user");

    cut_rx
        .recv()
        .unwrap()
        .shutdown(std::net::Shutdown::Both)
        .unwrap();
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
    while remote.is_connected() && std::time::Instant::now() < deadline {
        std::thread::sleep(std::time::Duration::from_millis(20));
    }
    assert!(!remote.is_connected());
}
