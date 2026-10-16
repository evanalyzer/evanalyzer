//! `evanalyzer worker`: accepts clients and executes their requests on a
//! [`Backend`] (normally `LocalBackend`) in this process - the server side of
//! a [`RemoteBackend`](super::RemoteBackend) connection. Started per
//! logged-in user by `evanalyzer server` (the multi-user gateway in
//! `crates/server`), or by hand for a direct `--remote-token` connection.
//!
//! Security model: every client must present the shared token in its first
//! message, and the worker binds to localhost unless told otherwise. The
//! worker itself speaks plain `ws://`: started by `evanalyzer server`, it
//! listens on 127.0.0.1 only and clients reach it through the server, whose
//! client connections are encrypted (`wss://`, see `crates/server`). A
//! worker started by hand on another address is unencrypted - put it behind
//! an SSH tunnel or VPN. An authenticated client can make the worker read
//! any image and write results anywhere this process may (within its
//! `--root` folders), so the token must be treated like a password.

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
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::mpsc::{self, Sender};
use std::sync::{Arc, Mutex};
use std::time::Duration;

/// How long a new connection may take to authenticate.
const HANDSHAKE_TIMEOUT: Duration = Duration::from_secs(10);

pub struct Worker {
    listener: TcpListener,
    token: String,
    idle_timeout: Option<Duration>,
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
        Ok(Self {
            listener,
            token,
            idle_timeout: None,
        })
    }

    /// Ends the process once no client has been connected and no analysis
    /// has run for `timeout` - what `evanalyzer server` starts its workers
    /// with, so a user's worker doesn't outlive its use (the next login
    /// starts a new one).
    pub fn with_idle_timeout(mut self, timeout: Duration) -> Self {
        self.idle_timeout = Some(timeout);
        self
    }

    pub fn local_addr(&self) -> Result<SocketAddr, InternalErrors> {
        Ok(self.listener.local_addr()?)
    }

    /// Serves clients until the process exits, one thread per connection.
    pub fn run(self, backend: Arc<dyn Backend>) {
        let token = Arc::new(self.token);
        // Shared by all connections: analyses outlive the one that started them.
        let jobs = Arc::new(JobRegistry::default());
        let connections = Arc::new(AtomicUsize::new(0));
        if let Some(timeout) = self.idle_timeout {
            let (jobs, connections) = (Arc::clone(&jobs), Arc::clone(&connections));
            std::thread::spawn(move || {
                watch_idle(timeout, || {
                    connections.load(Ordering::SeqCst) == 0 && !jobs.any_running()
                });
                log::info!(
                    "No client and no analysis for {} min - stopping",
                    timeout.as_secs() / 60
                );
                std::process::exit(0);
            });
        }
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
            let connections = Arc::clone(&connections);
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
                connections.fetch_add(1, Ordering::SeqCst);
                match serve_connection(stream, &token, backend, jobs) {
                    Ok(()) => log::info!("Client {peer} disconnected"),
                    Err(reason) => log::warn!("Client {peer} rejected: {reason}"),
                }
                connections.fetch_sub(1, Ordering::SeqCst);
            });
        }
    }
}

/// Returns once `is_idle` has held for `timeout` without interruption,
/// checking a few times per timeout (at least every 30 s).
fn watch_idle(timeout: Duration, is_idle: impl Fn() -> bool) {
    let interval = (timeout / 4).clamp(Duration::from_millis(10), Duration::from_secs(30));
    let mut idle_since: Option<std::time::Instant> = None;
    loop {
        std::thread::sleep(interval);
        if !is_idle() {
            idle_since = None;
            continue;
        }
        if idle_since
            .get_or_insert_with(std::time::Instant::now)
            .elapsed()
            >= timeout
        {
            return;
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
            req.base_results = req.base_results.map(native_path);
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
            if let Some(save_to) = &mut req.save_to {
                save_to.project_dir = native_path(std::mem::take(&mut save_to.project_dir));
            }
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
            } else if protocol_version != PROTOCOL_VERSION {
                // Only the protocol has to match - see `PROTOCOL_VERSION`.
                let update = if protocol_version < PROTOCOL_VERSION {
                    "update EVAnalyzer on this computer"
                } else {
                    "the server needs a newer EVAnalyzer"
                };
                Some(format!(
                    "incompatible versions: server is EVAnalyzer {APP_VERSION} (protocol \
                     {PROTOCOL_VERSION}), client is {app_version} (protocol {protocol_version}) \
                     - {update}"
                ))
            } else {
                if app_version != APP_VERSION {
                    log::info!(
                        "Client runs EVAnalyzer {app_version}, this worker {APP_VERSION} \
                         (same protocol {PROTOCOL_VERSION})"
                    );
                }
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
        background: Mutex::new(HashMap::new()),
        disconnected: std::sync::atomic::AtomicBool::new(false),
        images: Mutex::new(HashMap::new()),
        results: Mutex::new(HashMap::new()),
        next_handle: AtomicU64::new(1),
    });
    let handler = Arc::clone(&session);
    conn::run_io(ws, outgoing_rx, move |bytes| handler.on_frame(&bytes));

    // The client is gone: nobody is left to receive previews, trainings or
    // exports, so stop them instead of letting them run unseen. Analyses go
    // on - their results land on disk, and a client can attach again - and
    // so do trainings that know where to save their model.
    session.disconnected.store(true, Ordering::SeqCst);
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
    /// Trainings that save their model themselves if the client is gone
    /// (`TrainingRequest::save_to`), by request id - for `Cancel`. Not
    /// cancelled when the connection drops.
    background: Mutex<HashMap<u64, CancelHandle>>,
    /// Set once the connection has dropped.
    disconnected: std::sync::atomic::AtomicBool,
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
                if let Some(cancel) = self.background.lock().unwrap().get(&id) {
                    cancel.cancel();
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
            Request::StartTraining(req) => {
                let save_to = req.save_to.clone();
                self.train(id, req, save_to)
            }
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

    /// Runs a training and streams its progress and model back. With
    /// `save_to` it survives the client: if the connection dropped by the
    /// time it's done, the model is saved there on this machine.
    fn train(
        &self,
        id: u64,
        req: crate::api::TrainingRequest,
        save_to: Option<crate::api::ModelDestination>,
    ) {
        let training = match self.backend.start_training(req) {
            Ok(training) => training,
            Err(StartTrainingError::NoTrainingData) => {
                return self.reply(id, Reply::NoTrainingData, Vec::new());
            }
            Err(StartTrainingError::Failed(e)) => return self.fail(id, &e),
        };
        // Registered before replying, so a cancel sent right after
        // "started" can't arrive before it is known.
        let cancellable = if save_to.is_some() {
            &self.background
        } else {
            &self.running
        };
        cancellable
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
        let result = training.wait();
        cancellable.lock().unwrap().remove(&id);
        if let (Some(save_to), Ok(model), true) =
            (&save_to, &result, self.disconnected.load(Ordering::SeqCst))
        {
            match crate::workspace::ai_learning::save_trained_model(
                self.backend.files(),
                model,
                &save_to.project_dir,
                &save_to.model_name,
            ) {
                Ok(path) => log::info!(
                    "Training finished without its client - model saved to {}",
                    path.display()
                ),
                Err(e) => log::error!("Training finished without its client, saving failed: {e}"),
            }
            return;
        }
        match result.and_then(|model| model.to_bytes()) {
            Ok(bytes) => self.reply(id, Reply::TrainingDone(Ok(())), vec![bytes]),
            Err(e) => self.reply(id, Reply::TrainingDone(Err((&e).into())), Vec::new()),
        }
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
        ResultsQuery::RunStatus => ResultsAnswer::RunStatus(source.run_status()?),
        ResultsQuery::Boxplot(filter) => ResultsAnswer::Boxplot(source.boxplot(&filter)?),
        ResultsQuery::Histogram(filter) => ResultsAnswer::Histogram(source.histogram(&filter)?),
        ResultsQuery::Scatter(filter) => ResultsAnswer::Scatter(source.scatter(&filter)?),
        ResultsQuery::GroupingRegex(grouping) => {
            ResultsAnswer::Text(source.grouping_regex(&grouping)?)
        }
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
            base_results: None,
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

#[cfg(test)]
mod idle_tests {
    use super::*;
    use std::sync::atomic::AtomicBool;
    use std::time::Instant;

    #[test]
    fn the_watch_returns_only_after_an_uninterrupted_idle_stretch() {
        let busy_until = Instant::now() + Duration::from_millis(150);
        let start = Instant::now();
        watch_idle(Duration::from_millis(100), || Instant::now() >= busy_until);
        // Busy for 150 ms, then idle for the full 100 ms.
        assert!(
            start.elapsed() >= Duration::from_millis(250),
            "{:?}",
            start.elapsed()
        );
    }

    #[test]
    fn activity_restarts_the_idle_stretch() {
        let poke = Arc::new(AtomicBool::new(false));
        let poker = Arc::clone(&poke);
        std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(80));
            poker.store(true, Ordering::SeqCst);
        });
        let start = Instant::now();
        watch_idle(Duration::from_millis(100), || {
            !poke.swap(false, Ordering::SeqCst)
        });
        // Interrupted at ~80 ms, so idle for 100 ms only after that.
        assert!(
            start.elapsed() >= Duration::from_millis(180),
            "{:?}",
            start.elapsed()
        );
    }
}

#[cfg(test)]
mod version_tests {
    use super::*;
    use crate::backend::local::LocalBackend;

    /// What a worker answers a `Hello` with these versions.
    fn hello(protocol_version: u32, app_version: &str) -> ServerMsg {
        let worker = Worker::bind("127.0.0.1:0", "t".into()).unwrap();
        let url = format!("ws://{}/", worker.local_addr().unwrap());
        std::thread::spawn(move || worker.run(Arc::new(LocalBackend::default())));
        let (mut ws, _) = tungstenite::connect(url).unwrap();
        let hello = frame::encode(
            &ClientMsg::Hello {
                protocol_version,
                app_version: app_version.into(),
                token: "t".into(),
            },
            &[],
        )
        .unwrap();
        ws.send(tungstenite::Message::Binary(hello.into())).unwrap();
        let answer = loop {
            match ws.read().unwrap() {
                tungstenite::Message::Binary(bytes) => break bytes,
                _ => continue,
            }
        };
        frame::decode::<ServerMsg>(&answer).unwrap().msg
    }

    #[test]
    fn another_app_version_with_the_same_protocol_is_welcome() {
        assert!(matches!(
            hello(PROTOCOL_VERSION, "0.0.1-older"),
            ServerMsg::Welcome { .. }
        ));
    }

    #[test]
    fn another_protocol_is_refused_saying_which_side_to_update() {
        let ServerMsg::Rejected { reason } = hello(PROTOCOL_VERSION - 1, APP_VERSION) else {
            panic!("expected a rejection");
        };
        assert!(
            reason.contains("update EVAnalyzer on this computer"),
            "{reason}"
        );
        let ServerMsg::Rejected { reason } = hello(PROTOCOL_VERSION + 1, APP_VERSION) else {
            panic!("expected a rejection");
        };
        assert!(
            reason.contains("the server needs a newer EVAnalyzer"),
            "{reason}"
        );
    }
}

#[cfg(test)]
mod session_tests {
    use super::*;
    use crate::api::{
        AnalysisRequest, DirEntry, FileSystem, ImageChannel, ImageMeta, PreviewRequest, RunningJob,
        RunningTraining, SystemInfo, TemplateFolders, TileRequest, TrainingRequest,
    };
    use crate::workspace::settings::AppSettings;
    use evanalyzer_cfg::settings::ai_learning_settings::{
        AiLearningBackendSettings, AiLearningClassifierSettings, AiLearningSettings,
    };
    use std::path::Path;
    use std::sync::atomic::AtomicBool;
    use std::sync::mpsc::Receiver;

    fn broken(what: &str) -> InternalErrors {
        InternalErrors::Internal(format!("{what} is broken"))
    }

    /// A backend whose every operation fails - except opening an image,
    /// which gives one whose tiles can't be read.
    struct FailingBackend {
        preview_tiles: Option<usize>,
    }

    struct UnreadableImage(ImageMeta);

    impl ImageSource for UnreadableImage {
        fn meta(&self) -> &ImageMeta {
            &self.0
        }
        fn read_tile(&self, _: &TileRequest) -> Result<Vec<ImageChannel>, InternalErrors> {
            Err(broken("the tile"))
        }
    }

    impl FileSystem for FailingBackend {
        fn places(&self) -> Result<Vec<crate::api::Place>, InternalErrors> {
            Err(broken("places"))
        }
        fn list_dir(&self, _: &Path) -> Result<Vec<DirEntry>, InternalErrors> {
            Err(broken("list_dir"))
        }
        fn stat(&self, _: &Path) -> Result<Option<DirEntry>, InternalErrors> {
            Err(broken("stat"))
        }
        fn read_file(&self, _: &Path) -> Result<Vec<u8>, InternalErrors> {
            Err(broken("read_file"))
        }
        fn write_file(&self, _: &Path, _: &[u8]) -> Result<(), InternalErrors> {
            Err(broken("write_file"))
        }
        fn create_dir_all(&self, _: &Path) -> Result<(), InternalErrors> {
            Err(broken("create_dir_all"))
        }
        fn rename(&self, _: &Path, _: &Path) -> Result<(), InternalErrors> {
            Err(broken("rename"))
        }
        fn remove_all(&self, _: &Path) -> Result<(), InternalErrors> {
            Err(broken("remove_all"))
        }
    }

    impl Backend for FailingBackend {
        fn start_analysis(&self, _: AnalysisRequest) -> Result<RunningJob, InternalErrors> {
            Err(broken("analysis"))
        }
        fn start_preview(&self, _: PreviewRequest) -> Result<RunningJob, StartPreviewError> {
            Err(match self.preview_tiles {
                Some(tiles) => StartPreviewError::TooManyTiles { tiles },
                None => StartPreviewError::Failed(broken("preview")),
            })
        }
        fn start_training(
            &self,
            _: TrainingRequest,
        ) -> Result<RunningTraining, StartTrainingError> {
            Err(StartTrainingError::Failed(broken("training")))
        }
        fn open_image(&self, _: &Path) -> Result<Arc<dyn ImageSource>, InternalErrors> {
            Ok(Arc::new(UnreadableImage(ImageMeta::default())))
        }
        fn open_results(&self, _: &Path) -> Result<Arc<dyn ResultsSource>, InternalErrors> {
            Err(broken("open_results"))
        }
        fn read_image_meta(&self, _: &Path) -> Result<ImageMeta, InternalErrors> {
            Err(broken("read_image_meta"))
        }
        fn template_folders(&self) -> Result<TemplateFolders, InternalErrors> {
            Err(broken("template_folders"))
        }
        fn files(&self) -> &dyn FileSystem {
            self
        }
        fn description(&self) -> String {
            "failing".into()
        }
        fn system_info(&self) -> Result<SystemInfo, InternalErrors> {
            Err(broken("system_info"))
        }
        fn image_formats(&self) -> Vec<String> {
            Vec::new()
        }
        fn load_app_settings(&self) -> Result<AppSettings, InternalErrors> {
            Err(broken("load_app_settings"))
        }
        fn save_app_settings(&self, _: &AppSettings) -> Result<(), InternalErrors> {
            Err(broken("save_app_settings"))
        }
    }

    fn session_with(backend: FailingBackend) -> (Arc<Session>, Receiver<Vec<u8>>) {
        let (outgoing, outgoing_rx) = mpsc::channel();
        let session = Arc::new(Session {
            backend: Arc::new(backend),
            jobs: Arc::new(JobRegistry::default()),
            outgoing,
            running: Mutex::new(HashMap::new()),
            analyses: Mutex::new(HashMap::new()),
            background: Mutex::new(HashMap::new()),
            disconnected: AtomicBool::new(false),
            images: Mutex::new(HashMap::new()),
            results: Mutex::new(HashMap::new()),
            next_handle: AtomicU64::new(1),
        });
        (session, outgoing_rx)
    }

    fn failing_session() -> (Arc<Session>, Receiver<Vec<u8>>) {
        session_with(FailingBackend {
            preview_tiles: None,
        })
    }

    /// The one reply `request` gets, handled synchronously.
    fn reply_to(session: &Session, rx: &Receiver<Vec<u8>>, request: Request) -> (u64, Reply) {
        session.handle(7, request, Vec::new());
        let ServerMsg::Reply { id, reply } = frame::decode::<ServerMsg>(&rx.try_recv().unwrap())
            .unwrap()
            .msg
        else {
            panic!("expected a reply");
        };
        assert!(rx.try_recv().is_err(), "exactly one reply");
        (id, reply)
    }

    fn failure_text(reply: Reply) -> String {
        match reply {
            Reply::Failed(e) => e.text().to_string(),
            _ => panic!("expected a failure"),
        }
    }

    #[test]
    fn every_failing_backend_call_is_answered_with_its_error() {
        let (session, rx) = failing_session();
        let path = || PathBuf::from("/data/x");
        for (request, expected) in [
            (Request::ReadImageMeta { path: path() }, "read_image_meta"),
            (Request::OpenResults { path: path() }, "open_results"),
            (Request::TemplateFolders, "template_folders"),
            (Request::Places, "places"),
            (Request::SystemInfo, "system_info"),
            (Request::LoadAppSettings, "load_app_settings"),
            (
                Request::SaveAppSettings(AppSettings::default()),
                "save_app_settings",
            ),
            (Request::ListDir { path: path() }, "list_dir"),
            (Request::Stat { path: path() }, "stat"),
            (Request::ReadFile { path: path() }, "read_file"),
            (Request::CreateDir { path: path() }, "create_dir_all"),
            (
                Request::Rename {
                    from: path(),
                    to: path(),
                },
                "rename",
            ),
            (Request::RemoveAll { path: path() }, "remove_all"),
        ] {
            let (id, reply) = reply_to(&session, &rx, request);
            assert_eq!(id, 7);
            let text = failure_text(reply);
            assert!(text.contains(&format!("{expected} is broken")), "{text}");
        }
    }

    #[test]
    fn a_failing_write_and_a_write_without_contents_are_errors() {
        let (session, rx) = failing_session();

        let (_, reply) = reply_to(
            &session,
            &rx,
            Request::WriteFile {
                path: PathBuf::from("/data/x"),
            },
        );
        assert_eq!(failure_text(reply), "no file contents sent");

        session.handle(
            8,
            Request::WriteFile {
                path: PathBuf::from("/data/x"),
            },
            vec![b"contents".to_vec()],
        );
        let ServerMsg::Reply { id, reply } = frame::decode::<ServerMsg>(&rx.try_recv().unwrap())
            .unwrap()
            .msg
        else {
            panic!("expected a reply");
        };
        assert_eq!(id, 8);
        assert!(failure_text(reply).contains("write_file is broken"));
    }

    #[test]
    fn a_remove_all_path_from_a_windows_client_is_converted() {
        let Request::RemoveAll { path } = native_paths(Request::RemoveAll {
            path: PathBuf::from(r"/data\old"),
        }) else {
            panic!("expected a remove");
        };
        assert_eq!(path, PathBuf::from("/data/old"));
    }

    #[test]
    fn queries_and_exports_on_a_results_handle_that_is_not_open_fail() {
        let (session, rx) = failing_session();
        for request in [
            Request::QueryResults { handle: 42 },
            Request::ExportResults { handle: 42 },
        ] {
            let (_, reply) = reply_to(&session, &rx, request);
            assert!(
                failure_text(reply).contains("results database is not open on the server"),
                "a not opened handle"
            );
        }
    }

    #[test]
    fn reading_a_tile_needs_an_open_image_and_reports_read_errors() {
        let (session, rx) = failing_session();
        let tile = TileRequest {
            series: 0,
            resolution_idx: 0,
            z_projection: Default::default(),
            z_range: None,
            t_stack: 0,
            tile: Default::default(),
        };

        let (_, reply) = reply_to(
            &session,
            &rx,
            Request::ReadTile {
                handle: 99,
                tile: tile.clone(),
            },
        );
        assert_eq!(failure_text(reply), "image is not open on the server");

        let (_, reply) = reply_to(
            &session,
            &rx,
            Request::OpenImage {
                path: PathBuf::from("/data/a.tif"),
            },
        );
        let Reply::ImageOpened { handle, .. } = reply else {
            panic!("expected the image to open");
        };
        let (_, reply) = reply_to(&session, &rx, Request::ReadTile { handle, tile });
        assert!(failure_text(reply).contains("the tile is broken"));
    }

    #[test]
    fn a_closed_image_can_no_longer_be_read() {
        let (session, rx) = failing_session();
        let (_, reply) = reply_to(
            &session,
            &rx,
            Request::OpenImage {
                path: PathBuf::from("/data/a.tif"),
            },
        );
        let Reply::ImageOpened { handle, .. } = reply else {
            panic!("expected the image to open");
        };
        let close = frame::encode(&ClientMsg::CloseImage { handle }, &[]).unwrap();

        assert!(session.on_frame(&close));

        assert!(session.images.lock().unwrap().is_empty());
    }

    #[test]
    fn a_preview_with_too_many_tiles_or_a_failed_start_is_reported() {
        let preview = || {
            Request::StartPreview(PreviewRequest {
                settings: ProjectSettings::default(),
                project_path: PathBuf::from("/data/project"),
                viewport: Default::default(),
                breakpoint: None,
            })
        };
        let (session, rx) = session_with(FailingBackend {
            preview_tiles: Some(1234),
        });
        let (_, reply) = reply_to(&session, &rx, preview());
        assert!(matches!(reply, Reply::PreviewTooManyTiles { tiles: 1234 }));

        let (session, rx) = failing_session();
        let (_, reply) = reply_to(&session, &rx, preview());
        assert!(failure_text(reply).contains("preview is broken"));
    }

    #[test]
    fn a_training_that_cannot_start_is_reported() {
        let (session, rx) = failing_session();
        let (_, reply) = reply_to(
            &session,
            &rx,
            Request::StartTraining(TrainingRequest {
                project: ProjectSettings::default(),
                settings: AiLearningSettings {
                    schema_version: evanalyzer_cfg::CURRENT_AI_LEARNING_SETTINGS_SCHEMA_VERSION,
                    meta: Default::default(),
                    backend: AiLearningBackendSettings::RandomForest(Default::default()),
                    classifier: AiLearningClassifierSettings::Object {
                        feature_spec: evanalyzer_cfg::settings::ai_learning_object_settings::AiLearningObjectFeatureSettings { metrics: Vec::new() },
                        class_labels: Vec::new(),
                    },
                },
                pixel_params: Default::default(),
                save_to: None,
            }),
        );
        assert!(failure_text(reply).contains("training is broken"));
    }

    #[test]
    fn an_analysis_that_cannot_start_is_reported() {
        let (session, rx) = failing_session();
        let (_, reply) = reply_to(
            &session,
            &rx,
            Request::StartAnalysis(AnalysisRequest {
                settings: ProjectSettings::default(),
                project_path: PathBuf::from("/data/project"),
                job_name: None,
                threads: None,
                base_results: None,
            }),
        );
        assert!(failure_text(reply).contains("analysis is broken"));
    }

    #[test]
    fn a_cancel_reaches_running_and_background_work() {
        let (session, _rx) = failing_session();
        let running = Arc::new(AtomicBool::new(false));
        let background = Arc::new(AtomicBool::new(false));
        session
            .running
            .lock()
            .unwrap()
            .insert(1, CancelHandle::new(Arc::clone(&running)));
        session
            .background
            .lock()
            .unwrap()
            .insert(2, CancelHandle::new(Arc::clone(&background)));

        for id in [1, 2] {
            let cancel = frame::encode(&ClientMsg::Cancel { id }, &[]).unwrap();
            assert!(session.on_frame(&cancel));
        }

        assert!(running.load(Ordering::SeqCst));
        assert!(background.load(Ordering::SeqCst));
    }

    #[test]
    fn a_cancel_for_unknown_work_and_a_repeated_hello_are_ignored() {
        let (session, rx) = failing_session();
        let cancel = frame::encode(&ClientMsg::Cancel { id: 5 }, &[]).unwrap();
        let hello = frame::encode(
            &ClientMsg::Hello {
                protocol_version: PROTOCOL_VERSION,
                app_version: APP_VERSION.into(),
                token: "t".into(),
            },
            &[],
        )
        .unwrap();

        assert!(session.on_frame(&cancel));
        assert!(session.on_frame(&hello));

        assert!(rx.try_recv().is_err(), "nothing to answer");
    }

    #[test]
    fn a_malformed_message_drops_the_client() {
        let (session, _rx) = failing_session();

        assert!(!session.on_frame(b"not a frame"));
    }

    #[test]
    fn an_empty_token_is_refused() {
        let Err(InternalErrors::InvalidArgument(message)) = Worker::bind("127.0.0.1:0", "".into())
        else {
            panic!("expected an invalid argument");
        };
        assert!(message.contains("token must not be empty"), "{message}");
    }

    #[test]
    fn an_address_that_cannot_be_bound_is_an_io_error() {
        assert!(matches!(
            Worker::bind("not an address", "t".into()),
            Err(InternalErrors::Io(_))
        ));
    }

    #[test]
    fn generated_tokens_are_long_random_hex() {
        let first = generate_token().unwrap();
        let second = generate_token().unwrap();

        assert_eq!(first.len(), 48);
        assert!(first.chars().all(|c| c.is_ascii_hexdigit()));
        assert_ne!(first, second);
    }

    #[test]
    fn the_idle_timeout_is_kept_until_the_worker_runs() {
        let worker = Worker::bind("127.0.0.1:0", "t".into())
            .unwrap()
            .with_idle_timeout(Duration::from_secs(60));

        assert_eq!(worker.idle_timeout, Some(Duration::from_secs(60)));
    }

    #[test]
    fn tokens_are_compared_by_content_and_length() {
        assert!(constant_time_eq(b"secret", b"secret"));
        assert!(!constant_time_eq(b"secret", b"secreT"));
        assert!(!constant_time_eq(b"secret", b"secret2"));
    }
}

#[cfg(test)]
mod handshake_tests {
    use super::*;
    use crate::backend::local::LocalBackend;

    fn start_worker() -> String {
        let worker = Worker::bind("127.0.0.1:0", "t".into()).unwrap();
        let addr = worker.local_addr().unwrap();
        std::thread::spawn(move || worker.run(Arc::new(LocalBackend::default())));
        addr.to_string()
    }

    /// What the worker answers `first`, sent as the first message.
    fn answer_to(addr: &str, first: Vec<u8>) -> ServerMsg {
        let (mut ws, _) = tungstenite::connect(format!("ws://{addr}/")).unwrap();
        ws.send(tungstenite::Message::Binary(first.into())).unwrap();
        let answer = loop {
            match ws.read().unwrap() {
                tungstenite::Message::Binary(bytes) => break bytes,
                _ => continue,
            }
        };
        frame::decode::<ServerMsg>(&answer).unwrap().msg
    }

    #[test]
    fn a_first_message_other_than_hello_is_rejected() {
        let addr = start_worker();
        let cancel = frame::encode(&ClientMsg::Cancel { id: 1 }, &[]).unwrap();

        let ServerMsg::Rejected { reason } = answer_to(&addr, cancel) else {
            panic!("expected a rejection");
        };
        assert_eq!(reason, "expected a Hello message");
    }

    #[test]
    fn an_undecodable_first_message_is_rejected() {
        let addr = start_worker();

        let answer = answer_to(&addr, b"garbage".to_vec());

        assert!(matches!(answer, ServerMsg::Rejected { .. }));
    }

    #[test]
    fn a_port_check_does_not_stop_the_worker_from_serving_clients() {
        let addr = start_worker();
        drop(TcpStream::connect(&addr).unwrap());
        let hello = frame::encode(
            &ClientMsg::Hello {
                protocol_version: PROTOCOL_VERSION,
                app_version: APP_VERSION.into(),
                token: "t".into(),
            },
            &[],
        )
        .unwrap();

        assert!(matches!(answer_to(&addr, hello), ServerMsg::Welcome { .. }));
    }
}
