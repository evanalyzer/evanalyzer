//! `evanalyzer worker`: accepts clients and executes their requests on a
//! [`Backend`] (normally `LocalBackend`) in this process - the server side of
//! a [`RemoteBackend`](super::RemoteBackend) connection. Started per
//! logged-in user by `evanalyzer server` (the multi-user gateway in
//! `crates/server`), or by hand for a direct `--remote-token` connection.
//!
//! Security model: every client must present the shared token in its first
//! message, and the worker binds to localhost unless told otherwise. The
//! connection itself is plain `ws://` - not encrypted - so across machines it
//! belongs behind an SSH tunnel or VPN. An authenticated client can make the
//! worker read any image and write results anywhere this process may, so the
//! token must be treated like a password.

use super::job_registry::{JobEntry, JobRegistry};
use super::wire::conn::{self, HANDSHAKE_MESSAGE_SIZE, MAX_MESSAGE_SIZE};
use super::wire::frame::{self, Frame};
use super::wire::protocol::{
    APP_VERSION, ClientMsg, PROTOCOL_VERSION, Reply, Request, ResultsAnswer, ResultsQuery,
    ServerMsg, WireError, channels_to_wire, event_to_wire, from_postcard, to_postcard,
};
use crate::api::Backend;
use crate::api::CancelHandle;
use crate::api::ImageSource;
use crate::api::ResultsSource;
use crate::api::StartPreviewError;
use crate::api::StartTrainingError;
use evanalyzer_cfg::core_types::InternalErrors;
use evanalyzer_cfg::settings::project_settings::ProjectSettings;
use std::collections::HashMap;
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{self, Sender};
use std::sync::{Arc, Mutex};
use std::time::Duration;

/// How long a new connection may take to authenticate.
const HANDSHAKE_TIMEOUT: Duration = Duration::from_secs(10);

pub struct Worker {
    listener: TcpListener,
    token: String,
}

impl Worker {
    /// Binds `listen` (e.g. `127.0.0.1:7400`; port 0 picks a free one).
    pub fn bind(listen: &str, token: String) -> Result<Self, InternalErrors> {
        if token.is_empty() {
            return Err(InternalErrors::InvalidArgument(
                "the server token must not be empty".into(),
            ));
        }
        let listener = TcpListener::bind(listen)
            .map_err(|e| InternalErrors::Io(format!("Could not listen on {listen}: {e}")))?;
        Ok(Self { listener, token })
    }

    pub fn local_addr(&self) -> Result<SocketAddr, InternalErrors> {
        Ok(self.listener.local_addr()?)
    }

    /// Serves clients until the process exits, one thread per connection.
    pub fn run(self, backend: Arc<dyn Backend>) {
        let token = Arc::new(self.token);
        // Shared by all connections: analyses outlive the one that started them.
        let jobs = Arc::new(JobRegistry::default());
        for stream in self.listener.incoming() {
            let stream = match stream {
                Ok(stream) => stream,
                Err(e) => {
                    log::warn!("Failed to accept connection: {e}");
                    continue;
                }
            };
            let backend = Arc::clone(&backend);
            let token = Arc::clone(&token);
            let jobs = Arc::clone(&jobs);
            std::thread::spawn(move || {
                let peer = stream
                    .peer_addr()
                    .map(|a| a.to_string())
                    .unwrap_or_else(|_| "?".into());
                // A connection closed without sending a single byte is a port
                // check (e.g. `evanalyzer server` waiting for this worker to
                // come up), not a client - don't warn about it.
                let mut first = [0u8; 1];
                let _ = stream.set_read_timeout(Some(HANDSHAKE_TIMEOUT));
                if matches!(stream.peek(&mut first), Ok(0)) {
                    log::debug!("{peer} closed without sending anything (port check)");
                    return;
                }
                match serve_connection(stream, &token, backend, jobs) {
                    Ok(()) => log::info!("Client {peer} disconnected"),
                    Err(reason) => log::warn!("Client {peer} rejected: {reason}"),
                }
            });
        }
    }
}

/// `path` as this machine spells paths. A client on another OS joins the
/// paths it got from here with its own separator: a Windows client turns
/// `/home/me/images` + `a.vsi` into `/home/me/images\a.vsi`, which on Linux
/// names a file called `images\a.vsi`. So on Linux/macOS a `\` from a
/// client is taken as the separator it was meant to be (a file name that
/// really contains `\` can't be reached remotely then). Windows itself
/// accepts `/` as well, so paths stay as they are there.
fn native_path(path: PathBuf) -> PathBuf {
    if cfg!(windows) {
        return path;
    }
    match path.to_str() {
        Some(text) if text.contains('\\') => PathBuf::from(text.replace('\\', "/")),
        _ => path,
    }
}

/// The project's image paths in this machine's spelling - see
/// [`native_path`]; relative image paths of a project saved on Windows
/// contain `\` too.
fn native_project_paths(project: &mut ProjectSettings) {
    let images = &mut project.images;
    images.root = images.root.take().map(native_path);
    images.list = std::mem::take(&mut images.list)
        .into_iter()
        .map(|(path, entry)| (native_path(path), entry))
        .collect();
}

/// `request` with every path from the client in this machine's spelling -
/// see [`native_path`].
fn native_paths(request: Request) -> Request {
    match request {
        Request::StartAnalysis(mut req) => {
            req.project_path = native_path(req.project_path);
            native_project_paths(&mut req.settings);
            Request::StartAnalysis(req)
        }
        Request::StartPreview(mut req) => {
            req.project_path = native_path(req.project_path);
            native_project_paths(&mut req.settings);
            Request::StartPreview(req)
        }
        Request::StartTraining(mut req) => {
            native_project_paths(&mut req.project);
            Request::StartTraining(req)
        }
        Request::OpenImage { path } => Request::OpenImage {
            path: native_path(path),
        },
        Request::ReadImageMeta { path } => Request::ReadImageMeta {
            path: native_path(path),
        },
        Request::OpenResults { path } => Request::OpenResults {
            path: native_path(path),
        },
        Request::ListDir { path } => Request::ListDir {
            path: native_path(path),
        },
        Request::Stat { path } => Request::Stat {
            path: native_path(path),
        },
        Request::ReadFile { path } => Request::ReadFile {
            path: native_path(path),
        },
        Request::WriteFile { path } => Request::WriteFile {
            path: native_path(path),
        },
        Request::CreateDir { path } => Request::CreateDir {
            path: native_path(path),
        },
        Request::Rename { from, to } => Request::Rename {
            from: native_path(from),
            to: native_path(to),
        },
        Request::RemoveAll { path } => Request::RemoveAll {
            path: native_path(path),
        },
        other @ (Request::ReadTile { .. }
        | Request::QueryResults { .. }
        | Request::ExportResults { .. }
        | Request::ListJobs
        | Request::AttachJob { .. }
        | Request::ForgetJob { .. }
        | Request::TemplateFolders
        | Request::Places
        | Request::SystemInfo
        | Request::LoadAppSettings
        | Request::SaveAppSettings(_)) => other,
    }
}

/// Generates a random token for a worker started without one.
pub fn generate_token() -> Result<String, InternalErrors> {
    let mut bytes = [0u8; 24];
    getrandom::fill(&mut bytes)
        .map_err(|e| InternalErrors::Internal(format!("could not generate token: {e}")))?;
    Ok(bytes.iter().map(|b| format!("{b:02x}")).collect())
}

fn serve_connection(
    stream: TcpStream,
    token: &str,
    backend: Arc<dyn Backend>,
    jobs: Arc<JobRegistry>,
) -> Result<(), String> {
    stream.set_nodelay(true).ok();
    stream
        .set_read_timeout(Some(HANDSHAKE_TIMEOUT))
        .map_err(|e| e.to_string())?;
    let config = tungstenite::protocol::WebSocketConfig::default()
        .max_message_size(Some(HANDSHAKE_MESSAGE_SIZE))
        .max_frame_size(Some(HANDSHAKE_MESSAGE_SIZE));
    let mut ws = tungstenite::accept_with_config(stream, Some(config))
        .map_err(|e| format!("WebSocket handshake failed: {e}"))?;

    let hello = conn::read_binary(&mut ws)?;
    let rejection = match frame::decode::<ClientMsg>(&hello).map(|f| f.msg) {
        Ok(ClientMsg::Hello {
            protocol_version,
            app_version,
            token: client_token,
        }) => {
            if !constant_time_eq(client_token.as_bytes(), token.as_bytes()) {
                Some("invalid token".to_string())
            } else if protocol_version != PROTOCOL_VERSION || app_version != APP_VERSION {
                Some(format!(
                    "version mismatch: server is EVAnalyzer {APP_VERSION} (protocol {PROTOCOL_VERSION}), \
                     client is {app_version} (protocol {protocol_version}) - both sides must run the same version"
                ))
            } else {
                None
            }
        }
        Ok(_) => Some("expected a Hello message".into()),
        Err(e) => Some(e.to_string()),
    };
    if let Some(reason) = rejection {
        if let Ok(bytes) = frame::encode(
            &ServerMsg::Rejected {
                reason: reason.clone(),
            },
            &[],
        ) {
            let _ = ws.send(tungstenite::Message::Binary(bytes.into()));
        }
        let _ = ws.close(None);
        let _ = ws.flush();
        return Err(reason);
    }
    let welcome = frame::encode(
        &ServerMsg::Welcome {
            app_version: APP_VERSION.into(),
            image_formats: backend.image_formats(),
        },
        &[],
    )
    .map_err(|e| e.to_string())?;
    ws.send(tungstenite::Message::Binary(welcome.into()))
        .map_err(|e| e.to_string())?;
    conn::set_message_limit(&mut ws, MAX_MESSAGE_SIZE);
    log::info!("Client authenticated");

    let (outgoing, outgoing_rx) = mpsc::channel();
    let session = Arc::new(Session {
        backend,
        jobs,
        outgoing,
        running: Mutex::new(HashMap::new()),
        analyses: Mutex::new(HashMap::new()),
        images: Mutex::new(HashMap::new()),
        results: Mutex::new(HashMap::new()),
        next_handle: AtomicU64::new(1),
    });
    let handler = Arc::clone(&session);
    conn::run_io(ws, outgoing_rx, move |bytes| handler.on_frame(&bytes));

    // The client is gone: nobody is left to receive previews, trainings or
    // exports, so stop them instead of letting them run unseen. Analyses go
    // on - their results land on disk, and a client can attach again.
    for cancel in session.running.lock().unwrap().values() {
        cancel.cancel();
    }
    session.images.lock().unwrap().clear();
    session.results.lock().unwrap().clear();
    Ok(())
}

fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    a.len() == b.len() && a.iter().zip(b).fold(0u8, |acc, (x, y)| acc | (x ^ y)) == 0
}

/// One authenticated connection's state.
struct Session {
    backend: Arc<dyn Backend>,
    /// The worker's analyses, shared with all connections.
    jobs: Arc<JobRegistry>,
    outgoing: Sender<Vec<u8>>,
    /// Cancel handles of running previews, trainings and exports, by
    /// request id - cancelled when the connection drops.
    running: Mutex<HashMap<u64, CancelHandle>>,
    /// Analyses this connection follows, by request id - for `Cancel`.
    /// Not cancelled when the connection drops.
    analyses: Mutex<HashMap<u64, Arc<JobEntry>>>,
    /// Images the client has opened, by handle.
    images: Mutex<HashMap<u64, Arc<dyn ImageSource>>>,
    /// Results databases the client has opened, by handle.
    results: Mutex<HashMap<u64, Arc<dyn ResultsSource>>>,
    next_handle: AtomicU64,
}

impl Session {
    /// Returns `false` to drop the connection.
    fn on_frame(self: &Arc<Self>, bytes: &[u8]) -> bool {
        let (msg, blobs) = match frame::decode::<ClientMsg>(bytes) {
            Ok(Frame { msg, blobs }) => (msg, blobs),
            Err(e) => {
                log::warn!("Dropping client after malformed message: {e}");
                return false;
            }
        };
        match msg {
            ClientMsg::Request { id, request } => {
                let session = Arc::clone(self);
                std::thread::spawn(move || session.handle(id, request, blobs));
            }
            ClientMsg::Cancel { id } => {
                if let Some(cancel) = self.running.lock().unwrap().get(&id) {
                    cancel.cancel();
                }
                if let Some(analysis) = self.analyses.lock().unwrap().get(&id) {
                    analysis.cancel_handle().cancel();
                }
            }
            ClientMsg::CloseResults { handle } => {
                self.results.lock().unwrap().remove(&handle);
            }
            ClientMsg::CloseImage { handle } => {
                self.images.lock().unwrap().remove(&handle);
            }
            ClientMsg::Hello { .. } => {}
        }
        true
    }

    fn reply(&self, id: u64, reply: Reply, blobs: Vec<Vec<u8>>) {
        match frame::encode(&ServerMsg::Reply { id, reply }, &blobs) {
            // A send error means the connection is gone; the cancel-all on
            // disconnect stops the work.
            Ok(bytes) => {
                let _ = self.outgoing.send(bytes);
            }
            Err(e) => log::error!("Could not encode reply: {e}"),
        }
    }

    fn fail(&self, id: u64, e: &InternalErrors) {
        self.reply(id, Reply::Failed(e.into()), Vec::new());
    }

    fn handle(&self, id: u64, request: Request, blobs: Vec<Vec<u8>>) {
        let files = self.backend.files();
        let none = Vec::new;
        match native_paths(request) {
            Request::StartAnalysis(req) => {
                match self.jobs.start(|| self.backend.start_analysis(req)) {
                    Ok(analysis) => self.follow(id, analysis),
                    Err(e) => self.fail(id, &e),
                }
            }
            Request::AttachJob { job_id } => match self.jobs.get(&job_id) {
                Some(analysis) => self.follow(id, analysis),
                None => self.fail(
                    id,
                    &InternalErrors::InvalidArgument(format!(
                        "No analysis '{job_id}' on the server (finished long ago, \
                         or the worker restarted)"
                    )),
                ),
            },
            Request::ListJobs => self.reply(id, Reply::Jobs(self.jobs.list()), none()),
            Request::ForgetJob { job_id } => match self.jobs.forget(&job_id) {
                Ok(()) => self.reply(id, Reply::Done, none()),
                Err(e) => self.fail(id, &e),
            },
            Request::StartPreview(req) => match self.backend.start_preview(req) {
                Ok(job) => self.stream_job(id, job),
                Err(StartPreviewError::TooManyTiles { tiles }) => {
                    self.reply(id, Reply::PreviewTooManyTiles { tiles }, Vec::new())
                }
                Err(StartPreviewError::Failed(e)) => self.fail(id, &e),
            },
            Request::StartTraining(req) => match self.backend.start_training(req) {
                Ok(training) => {
                    // Registered before replying, so a cancel sent right
                    // after "started" can't arrive before it is known.
                    self.running
                        .lock()
                        .unwrap()
                        .insert(id, training.cancel_handle());
                    self.reply(
                        id,
                        Reply::TrainingStarted {
                            items: training.items(),
                        },
                        Vec::new(),
                    );
                    for event in training.events() {
                        self.reply(id, Reply::TrainingEvent(event), Vec::new());
                    }
                    let result = training.wait().and_then(|model| model.to_bytes());
                    self.running.lock().unwrap().remove(&id);
                    match result {
                        Ok(bytes) => self.reply(id, Reply::TrainingDone(Ok(())), vec![bytes]),
                        Err(e) => self.reply(id, Reply::TrainingDone(Err((&e).into())), Vec::new()),
                    }
                }
                Err(StartTrainingError::NoTrainingData) => {
                    self.reply(id, Reply::NoTrainingData, Vec::new())
                }
                Err(StartTrainingError::Failed(e)) => self.fail(id, &e),
            },
            Request::OpenImage { path } => match self.backend.open_image(&path) {
                Ok(source) => {
                    let handle = self.next_handle.fetch_add(1, Ordering::Relaxed);
                    let meta = source.meta().clone();
                    self.images.lock().unwrap().insert(handle, source);
                    self.reply(id, Reply::ImageOpened { handle, meta }, Vec::new());
                }
                Err(e) => self.fail(id, &e),
            },
            Request::ReadTile { handle, tile } => {
                let source = self.images.lock().unwrap().get(&handle).cloned();
                let Some(source) = source else {
                    self.reply(
                        id,
                        Reply::Failed(WireError::message("image is not open on the server")),
                        Vec::new(),
                    );
                    return;
                };
                match source.read_tile(&tile) {
                    Ok(channels) => {
                        let (wire, blobs) = channels_to_wire(&channels);
                        self.reply(id, Reply::Tile(wire), blobs);
                    }
                    Err(e) => self.fail(id, &e),
                }
            }
            Request::ReadImageMeta { path } => match self.backend.read_image_meta(&path) {
                Ok(meta) => self.reply(id, Reply::ImageMeta(meta), none()),
                Err(e) => self.fail(id, &e),
            },
            Request::OpenResults { path } => match self.backend.open_results(&path) {
                Ok(source) => {
                    let handle = self.next_handle.fetch_add(1, Ordering::Relaxed);
                    self.results.lock().unwrap().insert(handle, source);
                    self.reply(id, Reply::ResultsOpened { handle }, none());
                }
                Err(e) => self.fail(id, &e),
            },
            Request::QueryResults { handle } => {
                let answer = self
                    .results_for(handle)
                    .and_then(|source| answer_query(source.as_ref(), from_postcard(blobs.first())?))
                    .and_then(|answer| to_postcard(&answer));
                match answer {
                    Ok(bytes) => self.reply(id, Reply::ResultsAnswer, vec![bytes]),
                    Err(e) => self.fail(id, &e),
                }
            }
            Request::ExportResults { handle } => {
                let prepared = self.results_for(handle).and_then(|source| {
                    Ok((
                        source,
                        from_postcard::<crate::api::ResultExport>(blobs.first())?,
                    ))
                });
                let (source, export) = match prepared {
                    Ok(prepared) => prepared,
                    Err(e) => return self.fail(id, &e),
                };
                let flag = Arc::new(std::sync::atomic::AtomicBool::new(false));
                self.running
                    .lock()
                    .unwrap()
                    .insert(id, CancelHandle::new(Arc::clone(&flag)));
                let result = source.export(&export, &flag, &mut |message, current, total| {
                    let progress = Reply::ExportProgress {
                        message: message.to_string(),
                        current,
                        total,
                    };
                    self.reply(id, progress, Vec::new());
                });
                self.running.lock().unwrap().remove(&id);
                self.reply(
                    id,
                    Reply::ExportDone(result.map_err(|e| WireError::from(&e))),
                    none(),
                );
            }
            Request::TemplateFolders => match self.backend.template_folders() {
                Ok(folders) => self.reply(id, Reply::TemplateFolders(folders), none()),
                Err(e) => self.fail(id, &e),
            },
            Request::Places => match files.places() {
                Ok(places) => self.reply(id, Reply::Places(places), none()),
                Err(e) => self.fail(id, &e),
            },
            Request::SystemInfo => match self.backend.system_info() {
                Ok(info) => self.reply(id, Reply::SystemInfo(info), none()),
                Err(e) => self.fail(id, &e),
            },
            Request::LoadAppSettings => match self.backend.load_app_settings() {
                Ok(settings) => self.reply(id, Reply::AppSettings(settings), none()),
                Err(e) => self.fail(id, &e),
            },
            Request::SaveAppSettings(settings) => match self.backend.save_app_settings(&settings) {
                Ok(()) => self.reply(id, Reply::Done, none()),
                Err(e) => self.fail(id, &e),
            },
            Request::ListDir { path } => match files.list_dir(&path) {
                Ok(entries) => self.reply(id, Reply::DirEntries(entries), none()),
                Err(e) => self.fail(id, &e),
            },
            Request::Stat { path } => match files.stat(&path) {
                Ok(entry) => self.reply(id, Reply::Stat(entry), none()),
                Err(e) => self.fail(id, &e),
            },
            Request::ReadFile { path } => match files.read_file(&path) {
                Ok(data) => self.reply(id, Reply::FileData, vec![data]),
                Err(e) => self.fail(id, &e),
            },
            Request::WriteFile { path } => {
                let Some(data) = blobs.first() else {
                    self.reply(
                        id,
                        Reply::Failed(WireError::message("no file contents sent")),
                        none(),
                    );
                    return;
                };
                match files.write_file(&path, data) {
                    Ok(()) => self.reply(id, Reply::Done, none()),
                    Err(e) => self.fail(id, &e),
                }
            }
            Request::CreateDir { path } => match files.create_dir_all(&path) {
                Ok(()) => self.reply(id, Reply::Done, none()),
                Err(e) => self.fail(id, &e),
            },
            Request::Rename { from, to } => match files.rename(&from, &to) {
                Ok(()) => self.reply(id, Reply::Done, none()),
                Err(e) => self.fail(id, &e),
            },
            Request::RemoveAll { path } => match files.remove_all(&path) {
                Ok(()) => self.reply(id, Reply::Done, none()),
                Err(e) => self.fail(id, &e),
            },
        }
    }

    fn results_for(&self, handle: u64) -> Result<Arc<dyn ResultsSource>, InternalErrors> {
        self.results
            .lock()
            .unwrap()
            .get(&handle)
            .cloned()
            .ok_or_else(|| {
                InternalErrors::Internal("results database is not open on the server".into())
            })
    }

    /// Lets this connection follow `analysis` under request `id`.
    fn follow(&self, id: u64, analysis: Arc<JobEntry>) {
        // Known before the first reply, so a cancel sent right after
        // "started" finds it.
        self.analyses
            .lock()
            .unwrap()
            .insert(id, Arc::clone(&analysis));
        analysis.subscribe(self.outgoing.clone(), id);
    }

    fn stream_job(&self, id: u64, job: crate::api::RunningJob) {
        // Registered before replying, so a cancel sent right after
        // "started" can't arrive before it is known.
        self.running.lock().unwrap().insert(id, job.cancel_handle());
        self.reply(
            id,
            Reply::JobStarted {
                output_path: job.output_path().clone(),
                parallelism: job.parallelism(),
                job_id: None,
            },
            Vec::new(),
        );
        for event in job.events() {
            let (wire, blobs) = event_to_wire(event);
            self.reply(id, Reply::JobEvent(wire), blobs);
        }
        let result = job.wait();
        self.running.lock().unwrap().remove(&id);
        self.reply(
            id,
            Reply::JobDone(result.map_err(|e| WireError::from(&e))),
            Vec::new(),
        );
    }
}

fn answer_query(
    source: &dyn ResultsSource,
    query: ResultsQuery,
) -> Result<ResultsAnswer, InternalErrors> {
    Ok(match query {
        ResultsQuery::ObjectList(filter) => ResultsAnswer::Table(source.get_object_list(&filter)?),
        ResultsQuery::GroupedByImage(filter) => {
            ResultsAnswer::Table(source.get_grouped_by_image(&filter)?)
        }
        ResultsQuery::GroupByPlate(filter, view) => {
            ResultsAnswer::Table(source.get_group_by_plate(&filter, &view)?)
        }
        ResultsQuery::GroupByWell(filter, view) => {
            ResultsAnswer::Table(source.get_group_by_well(&filter, &view)?)
        }
        ResultsQuery::ImageHeatmap(filter, view) => {
            ResultsAnswer::Table(source.get_image_heatmap(&filter, &view)?)
        }
        ResultsQuery::Images => ResultsAnswer::Images(source.get_images()?),
        ResultsQuery::EnableImage {
            image_rel_path,
            disable,
        } => {
            source.enable_image(&image_rel_path, disable)?;
            ResultsAnswer::Done
        }
        ResultsQuery::ObjectClasses => ResultsAnswer::Classes(source.get_object_classes()?),
        ResultsQuery::AvailableColumns => ResultsAnswer::Columns(source.get_available_columns()?),
        ResultsQuery::ZStacks => ResultsAnswer::Count(source.get_nr_of_z_stacks()),
        ResultsQuery::TStacks => ResultsAnswer::Count(source.get_nr_of_t_stacks()),
        ResultsQuery::Boxplot(filter) => ResultsAnswer::Boxplot(source.boxplot(&filter)?),
        ResultsQuery::Histogram(filter) => ResultsAnswer::Histogram(source.histogram(&filter)?),
        ResultsQuery::Scatter(filter) => ResultsAnswer::Scatter(source.scatter(&filter)?),
    })
}

#[cfg(all(test, not(windows)))]
mod tests {
    use super::*;
    use crate::api::AnalysisRequest;
    use evanalyzer_cfg::settings::images_settings::ImageEntry;

    #[test]
    fn a_windows_client_s_separators_become_slashes() {
        // `/home/me/images` from this worker, joined on a Windows client.
        let path = native_path(PathBuf::from(r"/home/me/images\sub\a.vsi"));

        assert_eq!(path, PathBuf::from("/home/me/images/sub/a.vsi"));
    }

    #[test]
    fn a_path_without_backslashes_is_left_alone() {
        let path = PathBuf::from("/home/me/images/a.vsi");

        assert_eq!(native_path(path.clone()), path);
    }

    #[test]
    fn every_path_of_a_request_is_converted() {
        let Request::Rename { from, to } = native_paths(Request::Rename {
            from: PathBuf::from(r"/data\old.tif"),
            to: PathBuf::from(r"/data\new.tif"),
        }) else {
            panic!("expected a rename");
        };
        assert_eq!(from, PathBuf::from("/data/old.tif"));
        assert_eq!(to, PathBuf::from("/data/new.tif"));

        let Request::OpenImage { path } = native_paths(Request::OpenImage {
            path: PathBuf::from(r"/home/me/images\image.vsi"),
        }) else {
            panic!("expected an image open");
        };
        assert_eq!(path, PathBuf::from("/home/me/images/image.vsi"));
    }

    #[test]
    fn an_analysis_gets_its_project_and_image_paths_converted() {
        // A project saved on Windows: relative image paths with `\`.
        let mut settings = ProjectSettings::default();
        settings.images.root = Some(PathBuf::from(r"/home/me\images"));
        settings
            .images
            .list
            .insert(PathBuf::from(r"well_A1\a.vsi"), ImageEntry::default());
        settings
            .images
            .list
            .insert(PathBuf::from("b.vsi"), ImageEntry::default());

        let Request::StartAnalysis(req) = native_paths(Request::StartAnalysis(AnalysisRequest {
            settings,
            project_path: PathBuf::from(r"/home/me\project"),
            job_name: None,
            threads: None,
        })) else {
            panic!("expected an analysis");
        };

        assert_eq!(req.project_path, PathBuf::from("/home/me/project"));
        assert_eq!(
            req.settings.images.root,
            Some(PathBuf::from("/home/me/images"))
        );
        let images: Vec<&PathBuf> = req.settings.images.list.keys().collect();
        assert_eq!(
            images,
            [&PathBuf::from("well_A1/a.vsi"), &PathBuf::from("b.vsi")]
        );
    }
}
