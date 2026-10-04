//! Analyses on a worker outlive the client connection that started them:
//! a dropped connection doesn't cancel them, and another connection can list
//! them and attach again. The worker runs a backend whose analysis waits at
//! a gate the test opens, so "still running while the client is gone" is
//! deterministic; the network drop is a proxy between client and worker
//! that the test cuts.

use evanalyzer_app::analysis::{AnalysisRequest, JobState, ProgressEvent, RunningJob};
use evanalyzer_app::backends::Backend;
use evanalyzer_app::backends::local::LocalBackend;
use evanalyzer_app::backends::remote::{RemoteBackend, Worker};
use evanalyzer_app::project::{ProjectExt, ProjectWithRuntime};
use evanalyzer_cfg::core_types::InternalErrors;
use std::io;
use std::net::{Shutdown, TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex, mpsc};
use std::time::Duration;

const TOKEN: &str = "jobs-test-token";

/// Holds the gated analysis between its first and second image.
#[derive(Default)]
struct Gate {
    open: Mutex<bool>,
    opened: Condvar,
}

impl Gate {
    fn open(&self) {
        *self.open.lock().unwrap() = true;
        self.opened.notify_all();
    }

    /// Waits until opened (`true`) or `cancelled` (`false`).
    fn wait(&self, cancelled: &AtomicBool) -> bool {
        let mut open = self.open.lock().unwrap();
        while !*open {
            if cancelled.load(Ordering::SeqCst) {
                return false;
            }
            open = self
                .opened
                .wait_timeout(open, Duration::from_millis(10))
                .unwrap()
                .0;
        }
        true
    }
}

/// `LocalBackend`, except that an analysis is two fake images with the
/// [`Gate`] in between.
struct GatedBackend {
    local: LocalBackend,
    gate: Arc<Gate>,
}

impl Backend for GatedBackend {
    fn start_analysis(&self, req: AnalysisRequest) -> Result<RunningJob, InternalErrors> {
        let output = req
            .project_path
            .join("results")
            .join(req.job_name.unwrap_or_else(|| "gated".into()));
        let (events, events_rx) = mpsc::channel();
        let (done, done_rx) = mpsc::channel();
        let cancelled = Arc::new(AtomicBool::new(false));
        let (gate, flag) = (Arc::clone(&self.gate), Arc::clone(&cancelled));
        std::thread::spawn(move || {
            let image = |index| ProgressEvent::ImageCompleted {
                index,
                total: 2,
                path: PathBuf::from(format!("/images/{index}.tif")),
            };
            let _ = events.send(ProgressEvent::Started { total: 2 });
            let _ = events.send(image(1));
            let result = if gate.wait(&flag) {
                let _ = events.send(image(2));
                let _ = events.send(ProgressEvent::Finished);
                Ok(Default::default())
            } else {
                Err(InternalErrors::Cancelled)
            };
            drop(events);
            let _ = done.send(result);
        });
        Ok(RunningJob::from_parts(
            events_rx,
            evanalyzer_app::analysis::CancelHandle::with_callback(move || {
                cancelled.store(true, Ordering::SeqCst)
            }),
            output,
            1,
            Box::new(move || done_rx.recv().unwrap_or(Err(InternalErrors::Cancelled))),
        ))
    }

    fn start_preview(
        &self,
        req: evanalyzer_app::preview::PreviewRequest,
    ) -> Result<RunningJob, evanalyzer_app::preview::StartPreviewError> {
        self.local.start_preview(req)
    }

    fn start_training(
        &self,
        req: evanalyzer_app::ai_learning::TrainingRequest,
    ) -> Result<
        evanalyzer_app::ai_learning::RunningTraining,
        evanalyzer_app::ai_learning::StartTrainingError,
    > {
        self.local.start_training(req)
    }

    fn open_image(
        &self,
        path: &Path,
    ) -> Result<Arc<dyn evanalyzer_app::images::ImageSource>, InternalErrors> {
        self.local.open_image(path)
    }

    fn open_results(
        &self,
        path: &Path,
    ) -> Result<Arc<dyn evanalyzer_app::results::ResultsSource>, InternalErrors> {
        self.local.open_results(path)
    }

    fn read_image_meta(
        &self,
        path: &Path,
    ) -> Result<evanalyzer_app::images::ImageMeta, InternalErrors> {
        self.local.read_image_meta(path)
    }

    fn template_folders(
        &self,
    ) -> Result<evanalyzer_app::templates::TemplateFolders, InternalErrors> {
        self.local.template_folders()
    }

    fn files(&self) -> &dyn evanalyzer_app::fs::FileSystem {
        self.local.files()
    }

    fn description(&self) -> String {
        "gated".into()
    }

    fn system_info(&self) -> Result<evanalyzer_app::backends::SystemInfo, InternalErrors> {
        self.local.system_info()
    }

    fn image_formats(&self) -> Vec<String> {
        self.local.image_formats()
    }

    fn load_app_settings(&self) -> Result<evanalyzer_app::global::AppSettings, InternalErrors> {
        self.local.load_app_settings()
    }

    fn save_app_settings(
        &self,
        settings: &evanalyzer_app::global::AppSettings,
    ) -> Result<(), InternalErrors> {
        self.local.save_app_settings(settings)
    }
}

/// A worker serving [`GatedBackend`]; returns its address and the gate.
fn gated_worker() -> (String, Arc<Gate>) {
    let gate = Arc::new(Gate::default());
    let worker = Worker::bind("127.0.0.1:0", TOKEN.into()).unwrap();
    let addr = worker.local_addr().unwrap().to_string();
    let backend = GatedBackend {
        local: LocalBackend::default(),
        gate: Arc::clone(&gate),
    };
    std::thread::spawn(move || worker.run(Arc::new(backend)));
    (addr, gate)
}

/// Forwards one connection to `target` until [`Proxy::cut`] - a network
/// drop, as far as both ends can tell.
struct Proxy {
    url: String,
    sockets: Arc<Mutex<Vec<TcpStream>>>,
}

impl Proxy {
    fn to(target: String) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let url = format!("ws://{}", listener.local_addr().unwrap());
        let sockets = Arc::new(Mutex::new(Vec::new()));
        let kept = Arc::clone(&sockets);
        std::thread::spawn(move || {
            for client in listener.incoming().flatten() {
                let worker = TcpStream::connect(&target).unwrap();
                kept.lock()
                    .unwrap()
                    .extend([client.try_clone().unwrap(), worker.try_clone().unwrap()]);
                let pipe = |mut from: TcpStream, mut to: TcpStream| {
                    std::thread::spawn(move || {
                        let _ = io::copy(&mut from, &mut to);
                        let _ = to.shutdown(Shutdown::Both);
                    });
                };
                pipe(client.try_clone().unwrap(), worker.try_clone().unwrap());
                pipe(worker, client);
            }
        });
        Self { url, sockets }
    }

    fn cut(&self) {
        for socket in self.sockets.lock().unwrap().drain(..) {
            let _ = socket.shutdown(Shutdown::Both);
        }
    }
}

fn request(dir: &Path) -> AnalysisRequest {
    AnalysisRequest {
        settings: Default::default(),
        project_path: dir.to_path_buf(),
        job_name: Some("plate-3".into()),
        threads: Some(1),
    }
}

/// Waits until `remote` lists exactly one job in a state `matches` accepts.
fn wait_for_state(remote: &RemoteBackend, matches: impl Fn(&JobState) -> bool) -> JobState {
    let mut last = None;
    for _ in 0..500 {
        let jobs = remote.list_jobs().unwrap();
        if let [job] = jobs.as_slice() {
            if matches(&job.state) {
                return job.state.clone();
            }
            last = Some(job.state.clone());
        }
        std::thread::sleep(Duration::from_millis(10));
    }
    panic!("job never reached the expected state, last: {last:?}");
}

#[test]
fn an_analysis_survives_a_dropped_connection_and_can_be_attached_again() {
    let (worker, gate) = gated_worker();
    let dir = tempfile::tempdir().unwrap();

    // Client A starts the analysis and sees its first image, then the
    // network goes.
    let proxy = Proxy::to(worker.clone());
    let first = RemoteBackend::connect(&proxy.url, TOKEN).unwrap();
    let job = first.start_analysis(request(dir.path())).unwrap();
    let id = job
        .id()
        .expect("analyses on a worker have an id")
        .to_string();
    assert!(matches!(
        job.events().recv().unwrap(),
        ProgressEvent::Started { total: 2 }
    ));
    proxy.cut();
    assert!(job.wait().is_err(), "client A lost the connection");

    // Client B finds it still running, where A left it.
    let second = RemoteBackend::connect(&format!("ws://{worker}"), TOKEN).unwrap();
    assert_eq!(
        wait_for_state(&second, |state| matches!(state, JobState::Running { .. })),
        JobState::Running { done: 1, total: 2 }
    );
    let jobs = second.list_jobs().unwrap();
    assert_eq!(jobs[0].id, id);
    assert_eq!(jobs[0].name(), "plate-3");

    // Attaching replays the progress so far, then follows it live.
    let attached = second.attach_job(&id).unwrap();
    assert_eq!(attached.id(), Some(id.as_str()));
    assert!(attached.output_path().ends_with("results/plate-3"));
    assert!(matches!(
        attached.events().recv().unwrap(),
        ProgressEvent::Started { total: 2 }
    ));
    assert!(matches!(
        attached.events().recv().unwrap(),
        ProgressEvent::ImageCompleted { index: 1, .. }
    ));
    gate.open();
    let rest: Vec<_> = attached.events().iter().collect();
    assert!(matches!(
        rest.as_slice(),
        [
            ProgressEvent::ImageCompleted { index: 2, .. },
            ProgressEvent::Finished
        ]
    ));
    attached.wait().unwrap();
    assert_eq!(second.list_jobs().unwrap()[0].state, JobState::Succeeded);
}

#[test]
fn a_second_analysis_is_refused_while_one_runs() {
    let (worker, gate) = gated_worker();
    let url = format!("ws://{worker}");
    let dir = tempfile::tempdir().unwrap();
    let window = RemoteBackend::connect(&url, TOKEN).unwrap();
    let running = window.start_analysis(request(dir.path())).unwrap();

    // E.g. the CLI of the same user, while the GUI's analysis runs.
    let cli = RemoteBackend::connect(&url, TOKEN).unwrap();
    let error = cli.start_analysis(request(dir.path())).err().unwrap();
    assert!(error.to_string().contains("already running"), "{error}");

    gate.open();
    running.events().iter().for_each(drop);
    running.wait().unwrap();
    let again = cli.start_analysis(request(dir.path())).unwrap();
    again.events().iter().for_each(drop);
    again.wait().unwrap();
}

#[test]
fn an_attached_client_can_cancel_the_analysis() {
    let (worker, _gate) = gated_worker();
    let url = format!("ws://{worker}");
    let dir = tempfile::tempdir().unwrap();
    let starter = RemoteBackend::connect(&url, TOKEN).unwrap();
    let job = starter.start_analysis(request(dir.path())).unwrap();
    let id = job.id().unwrap().to_string();

    let other = RemoteBackend::connect(&url, TOKEN).unwrap();
    let attached = other.attach_job(&id).unwrap();
    attached.cancel_handle().cancel();

    attached.events().iter().for_each(drop);
    assert!(matches!(attached.wait(), Err(InternalErrors::Cancelled)));
    job.events().iter().for_each(drop);
    assert!(matches!(job.wait(), Err(InternalErrors::Cancelled)));
    assert_eq!(
        wait_for_state(&other, |state| !matches!(state, JobState::Running { .. })),
        JobState::Cancelled
    );
}

#[test]
fn finished_analyses_are_listed_until_forgotten_and_unknown_ids_are_reported() {
    let (worker, gate) = gated_worker();
    gate.open();
    let remote = RemoteBackend::connect(&format!("ws://{worker}"), TOKEN).unwrap();
    let dir = tempfile::tempdir().unwrap();
    let job = remote.start_analysis(request(dir.path())).unwrap();
    let id = job.id().unwrap().to_string();
    job.events().iter().for_each(drop);
    job.wait().unwrap();

    wait_for_state(&remote, |state| *state == JobState::Succeeded);
    remote.forget_job(&id).unwrap();
    assert!(remote.list_jobs().unwrap().is_empty());

    let error = remote.attach_job(&id).err().unwrap();
    assert!(error.to_string().contains("No analysis"), "{error}");
}

#[test]
fn a_real_analysis_is_tracked_too() {
    let worker = Worker::bind("127.0.0.1:0", TOKEN.into()).unwrap();
    let url = format!("ws://{}", worker.local_addr().unwrap());
    std::thread::spawn(move || worker.run(Arc::new(LocalBackend::default())));
    let remote = RemoteBackend::connect(&url, TOKEN).unwrap();

    let dir = tempfile::tempdir().unwrap();
    let images = dir.path().join("images");
    std::fs::create_dir_all(&images).unwrap();
    std::fs::copy(
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../core/tests/multi-channel-4D-series.ome.tif"),
        images.join("fixture.ome.tif"),
    )
    .unwrap();
    let mut project = ProjectWithRuntime::default();
    project.images.root = Some(images);
    project.scan_image_folder_and_add(&LocalBackend::default());

    let job = remote
        .start_analysis(AnalysisRequest {
            settings: project.settings,
            project_path: dir.path().to_path_buf(),
            job_name: Some("real".into()),
            threads: Some(1),
        })
        .unwrap();
    let id = job.id().unwrap().to_string();
    job.events().iter().for_each(drop);
    job.wait().unwrap();

    wait_for_state(&remote, |state| *state == JobState::Succeeded);
    assert_eq!(remote.list_jobs().unwrap()[0].id, id);
    assert!(LocalBackend::default().list_jobs().unwrap().is_empty());
}
