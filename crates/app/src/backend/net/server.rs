//! `evanalyzer serve`: accepts clients and executes their requests on a
//! [`Backend`] (normally `LocalBackend`) in this process.
//!
//! Security model: every client must present the shared token in its first
//! message, and the server binds to localhost unless told otherwise. The
//! connection itself is plain `ws://` - not encrypted - so across machines it
//! belongs behind an SSH tunnel or VPN. An authenticated client can make the
//! server read any image and write results anywhere this process may, so the
//! token must be treated like a password.

use super::conn::{self, HANDSHAKE_MESSAGE_SIZE, MAX_MESSAGE_SIZE};
use super::frame::{self, Frame};
use super::protocol::{
    APP_VERSION, ClientMsg, PROTOCOL_VERSION, Reply, Request, ServerMsg, WireError,
    channels_to_wire, event_to_wire,
};
use crate::ai_learning::StartTrainingError;
use crate::backend::{Backend, ImageSource};
use crate::job::{CancelHandle, StartPreviewError};
use evanalyzer_cfg::core_types::InternalErrors;
use std::collections::HashMap;
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{self, Sender};
use std::sync::{Arc, Mutex};
use std::time::Duration;

/// How long a new connection may take to authenticate.
const HANDSHAKE_TIMEOUT: Duration = Duration::from_secs(10);

pub struct Server {
    listener: TcpListener,
    token: String,
}

impl Server {
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
            std::thread::spawn(move || {
                let peer = stream
                    .peer_addr()
                    .map(|a| a.to_string())
                    .unwrap_or_else(|_| "?".into());
                match serve_connection(stream, &token, backend) {
                    Ok(()) => log::info!("Client {peer} disconnected"),
                    Err(reason) => log::warn!("Client {peer} rejected: {reason}"),
                }
            });
        }
    }
}

/// Generates a random token for a server started without one.
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
        outgoing,
        running: Mutex::new(HashMap::new()),
        images: Mutex::new(HashMap::new()),
        next_handle: AtomicU64::new(1),
    });
    let handler = Arc::clone(&session);
    conn::run_io(ws, outgoing_rx, move |bytes| handler.on_frame(&bytes));

    // The client is gone: nobody is left to receive results, so stop
    // whatever it started instead of letting it run to completion unseen.
    for cancel in session.running.lock().unwrap().values() {
        cancel.cancel();
    }
    session.images.lock().unwrap().clear();
    Ok(())
}

fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    a.len() == b.len() && a.iter().zip(b).fold(0u8, |acc, (x, y)| acc | (x ^ y)) == 0
}

/// One authenticated connection's state.
struct Session {
    backend: Arc<dyn Backend>,
    outgoing: Sender<Vec<u8>>,
    /// Cancel handles of running jobs/trainings, by request id.
    running: Mutex<HashMap<u64, CancelHandle>>,
    /// Images the client has opened, by handle.
    images: Mutex<HashMap<u64, Arc<dyn ImageSource>>>,
    next_handle: AtomicU64,
}

impl Session {
    /// Returns `false` to drop the connection.
    fn on_frame(self: &Arc<Self>, bytes: &[u8]) -> bool {
        let msg = match frame::decode::<ClientMsg>(bytes) {
            Ok(Frame { msg, .. }) => msg,
            Err(e) => {
                log::warn!("Dropping client after malformed message: {e}");
                return false;
            }
        };
        match msg {
            ClientMsg::Request { id, request } => {
                let session = Arc::clone(self);
                std::thread::spawn(move || session.handle(id, request));
            }
            ClientMsg::Cancel { id } => {
                if let Some(cancel) = self.running.lock().unwrap().get(&id) {
                    cancel.cancel();
                }
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

    fn handle(&self, id: u64, request: Request) {
        match request {
            Request::StartAnalysis(req) => match self.backend.start_analysis(req) {
                Ok(job) => self.stream_job(id, job),
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
        }
    }

    fn stream_job(&self, id: u64, job: crate::job::RunningJob) {
        // Registered before replying, so a cancel sent right after
        // "started" can't arrive before it is known.
        self.running.lock().unwrap().insert(id, job.cancel_handle());
        self.reply(
            id,
            Reply::JobStarted {
                output_path: job.output_path().clone(),
                parallelism: job.parallelism(),
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
