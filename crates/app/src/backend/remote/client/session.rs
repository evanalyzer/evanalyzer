//! The connection to a worker: opening it (directly with a token, or through
//! an `evanalyzer server` login), the protocol handshake, and matching each
//! request with its replies. Everything else in the client is a thin adapter
//! that turns API calls into requests on a [`Session`].

use super::tls::{self, NetStream, TlsTrust};
use crate::api::CancelHandle;
use crate::api::ConnectionSecurity;
use crate::api::RunningJob;
use crate::backend::remote::wire::conn::{self, MAX_MESSAGE_SIZE, Socket};
use crate::backend::remote::wire::frame::{self, Frame};
use crate::backend::remote::wire::protocol::{
    APP_VERSION, ClientMsg, PROTOCOL_VERSION, Reply, Request, ServerMsg, event_from_wire,
};
use evanalyzer_cfg::core_types::InternalErrors;
use std::collections::HashMap;
use std::net::{TcpStream, ToSocketAddrs};
use std::path::PathBuf;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tungstenite::{Message, WebSocket};

/// Port used when the URL doesn't name one.
const DEFAULT_PORT: u16 = 7400;

const CONNECT_TIMEOUT: Duration = Duration::from_secs(10);
/// Logging in to an `evanalyzer server` may include starting the user's
/// worker, which takes longer than a plain connect.
const LOGIN_TIMEOUT: Duration = Duration::from_secs(60);

pub(super) struct Session {
    url: String,
    /// The worker's image formats, from its `Welcome`.
    image_formats: Vec<String>,
    security: ConnectionSecurity,
    outgoing: Sender<Vec<u8>>,
    /// Requests waiting for replies, by id. Emptied when the connection
    /// drops, which turns every pending `recv` into a "connection lost".
    pending: Mutex<HashMap<u64, Sender<Frame<Reply>>>>,
    next_id: AtomicU64,
    /// Cleared by the I/O thread when the connection ends.
    connected: AtomicBool,
}

impl Session {
    /// Remote protocol handshake (`Hello`) on an open WebSocket, then starts
    /// the I/O thread.
    pub(super) fn open(
        mut ws: WebSocket<NetStream>,
        url: &str,
        token: &str,
    ) -> Result<Arc<Self>, InternalErrors> {
        let hello = frame::encode(
            &ClientMsg::Hello {
                protocol_version: PROTOCOL_VERSION,
                app_version: APP_VERSION.into(),
                token: token.into(),
            },
            &[],
        )?;
        ws.send(tungstenite::Message::Binary(hello.into()))
            .map_err(|e| InternalErrors::Io(format!("Could not reach {url}: {e}")))?;
        let answer = conn::read_binary(&mut ws)
            .map_err(|e| InternalErrors::Io(format!("No answer from {url}: {e}")))?;
        let image_formats = match frame::decode::<ServerMsg>(&answer)?.msg {
            ServerMsg::Welcome {
                app_version,
                image_formats,
            } => {
                log::info!("Connected to EVAnalyzer {app_version} server at {url}");
                image_formats
            }
            ServerMsg::Rejected { reason } => {
                return Err(InternalErrors::InvalidArgument(format!(
                    "Server {url} refused the connection: {reason}"
                )));
            }
            ServerMsg::Reply { .. } => {
                return Err(InternalErrors::Internal(
                    "unexpected reply before handshake".into(),
                ));
            }
        };

        let security = ws.get_ref().security();
        match security {
            ConnectionSecurity::Encrypted => {
                log::info!("Connection to {url} is encrypted (TLS)")
            }
            ConnectionSecurity::EncryptedUnverified => log::warn!(
                "Connection to {url} is encrypted, but the server's identity was not verified"
            ),
            ConnectionSecurity::Unencrypted | ConnectionSecurity::Local => log::warn!(
                "Connection to {url} is NOT encrypted: passwords and data cross the network readable"
            ),
        }

        let (outgoing, outgoing_rx) = mpsc::channel();
        let session = Arc::new(Self {
            url: url.into(),
            image_formats,
            security,
            outgoing,
            pending: Mutex::new(HashMap::new()),
            next_id: AtomicU64::new(1),
            connected: AtomicBool::new(true),
        });
        // The I/O thread holds only a weak reference: dropping the last
        // `RemoteBackend` (and with it the outgoing sender) ends the thread.
        let weak = Arc::downgrade(&session);
        std::thread::Builder::new()
            .name("evanalyzer-net-client".into())
            .spawn(move || {
                conn::run_io(ws, outgoing_rx, |bytes| {
                    let Some(session) = weak.upgrade() else {
                        return false;
                    };
                    session.dispatch(&bytes)
                });
                if let Some(session) = weak.upgrade() {
                    log::warn!("Lost connection to server {}", session.url);
                    session.connected.store(false, Ordering::Relaxed);
                    session.pending.lock().unwrap().clear();
                }
            })?;
        Ok(session)
    }

    pub(super) fn url(&self) -> &str {
        &self.url
    }

    pub(super) fn image_formats(&self) -> &[String] {
        &self.image_formats
    }

    pub(super) fn security(&self) -> ConnectionSecurity {
        self.security
    }

    /// Closes the connection: everything waiting for a reply gets "lost
    /// connection". What runs on the worker goes on (analyses, trainings
    /// with a destination).
    pub(super) fn close(&self) {
        let _ = self.outgoing.send(Vec::new());
    }

    pub(super) fn is_connected(&self) -> bool {
        self.connected.load(Ordering::Relaxed)
    }

    fn dispatch(&self, bytes: &[u8]) -> bool {
        let frame = match frame::decode::<ServerMsg>(bytes) {
            Ok(frame) => frame,
            Err(e) => {
                log::warn!("Dropping connection after malformed server message: {e}");
                return false;
            }
        };
        if let ServerMsg::Reply { id, reply } = frame.msg {
            let target = self.pending.lock().unwrap().get(&id).cloned();
            if let Some(target) = target {
                let _ = target.send(Frame {
                    msg: reply,
                    blobs: frame.blobs,
                });
            }
        }
        true
    }

    pub(super) fn disconnected(&self) -> InternalErrors {
        InternalErrors::Io(format!("Lost connection to server {}", self.url))
    }

    pub(super) fn send(&self, msg: &ClientMsg) -> Result<(), InternalErrors> {
        self.send_with_blobs(msg, &[])
    }

    fn send_with_blobs(&self, msg: &ClientMsg, blobs: &[Vec<u8>]) -> Result<(), InternalErrors> {
        let bytes = frame::encode(msg, blobs)?;
        self.outgoing.send(bytes).map_err(|_| self.disconnected())
    }

    /// Sends `request` and returns its id plus the channel its replies
    /// arrive on.
    pub(super) fn request(
        &self,
        request: Request,
    ) -> Result<(u64, Receiver<Frame<Reply>>), InternalErrors> {
        self.request_with_blobs(request, Vec::new())
    }

    pub(super) fn request_with_blobs(
        &self,
        request: Request,
        blobs: Vec<Vec<u8>>,
    ) -> Result<(u64, Receiver<Frame<Reply>>), InternalErrors> {
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let (tx, rx) = mpsc::channel();
        self.pending.lock().unwrap().insert(id, tx);
        if let Err(e) = self.send_with_blobs(&ClientMsg::Request { id, request }, &blobs) {
            self.finish(id);
            return Err(e);
        }
        Ok((id, rx))
    }

    /// Forgets request `id`; later replies to it are dropped.
    pub(super) fn finish(&self, id: u64) {
        self.pending.lock().unwrap().remove(&id);
    }

    pub(super) fn recv(&self, rx: &Receiver<Frame<Reply>>) -> Result<Frame<Reply>, InternalErrors> {
        rx.recv().map_err(|_| self.disconnected())
    }

    pub(super) fn cancel_handle(self: &Arc<Self>, id: u64) -> CancelHandle {
        let session = Arc::downgrade(self);
        CancelHandle::with_callback(move || {
            if let Some(session) = session.upgrade() {
                let _ = session.send(&ClientMsg::Cancel { id });
            }
        })
    }

    /// The first reply to `StartAnalysis` or `AttachJob`: on `JobStarted`
    /// the analysis as a `RunningJob`, carrying the worker's job id.
    pub(super) fn job_started(
        self: &Arc<Self>,
        id: u64,
        rx: Receiver<Frame<Reply>>,
    ) -> Result<RunningJob, InternalErrors> {
        match self.recv(&rx)?.msg {
            Reply::JobStarted {
                output_path,
                parallelism,
                job_id,
            } => {
                let job = self.running_job(id, rx, output_path, parallelism);
                Ok(match job_id {
                    Some(job_id) => job.with_id(job_id),
                    None => job,
                })
            }
            Reply::Failed(e) => {
                self.finish(id);
                Err(e.into_internal())
            }
            _ => {
                self.finish(id);
                Err(unexpected_reply())
            }
        }
    }

    /// Turns the replies of a started job into a `RunningJob`.
    pub(super) fn running_job(
        self: &Arc<Self>,
        id: u64,
        rx: Receiver<Frame<Reply>>,
        output_path: PathBuf,
        parallelism: usize,
    ) -> RunningJob {
        let (events_tx, events_rx) = mpsc::channel();
        let (done_tx, done_rx) = mpsc::channel();
        let session = Arc::clone(self);
        std::thread::spawn(move || {
            let result = loop {
                match session.recv(&rx) {
                    Ok(Frame {
                        msg: Reply::JobEvent(event),
                        blobs,
                    }) => match event_from_wire(event, blobs) {
                        Ok(event) => {
                            let _ = events_tx.send(event);
                        }
                        Err(e) => log::warn!("Dropping undecodable job event: {e}"),
                    },
                    Ok(Frame {
                        msg: Reply::JobDone(result),
                        ..
                    }) => break result.map_err(|e| e.into_internal()),
                    Ok(Frame {
                        msg: Reply::Failed(e),
                        ..
                    }) => break Err(e.into_internal()),
                    Ok(_) => break Err(unexpected_reply()),
                    Err(e) => break Err(e),
                }
            };
            session.finish(id);
            // Close the event stream before the result is available - front
            // ends drain events first, then wait.
            drop(events_tx);
            let _ = done_tx.send(result);
        });
        let disconnected = self.disconnected();
        RunningJob::from_parts(
            events_rx,
            self.cancel_handle(id),
            output_path,
            parallelism,
            Box::new(move || done_rx.recv().unwrap_or(Err(disconnected))),
        )
    }
}

pub(super) fn unexpected_reply() -> InternalErrors {
    InternalErrors::Internal("unexpected reply from server".into())
}

/// The TLS certificate of the server at `url`, without trusting it - `None`
/// for `ws://`, which has none.
pub(crate) fn server_certificate(
    url: &str,
) -> Result<Option<tls::ServerCertificate>, InternalErrors> {
    let uri: tungstenite::http::Uri = url
        .parse()
        .map_err(|e| InternalErrors::InvalidArgument(format!("Invalid server URL '{url}': {e}")))?;
    match uri.scheme_str() {
        Some("ws") => return Ok(None),
        Some("wss") => {}
        _ => {
            return Err(InternalErrors::InvalidArgument(format!(
                "Server URL must start with wss:// or ws:// (got '{url}')"
            )));
        }
    }
    let host = uri
        .host()
        .ok_or_else(|| InternalErrors::InvalidArgument(format!("No host in '{url}'")))?;
    let stream = connect_tcp(host, uri.port_u16().unwrap_or(DEFAULT_PORT))?;
    stream.set_read_timeout(Some(CONNECT_TIMEOUT))?;
    tls::inspect(stream, host, url).map(Some)
}

/// Opens the WebSocket to `url`: `ws://host[:port]`, or encrypted
/// `wss://host[:port]`, accepting the server's certificate as `trust` says
/// (see [`tls`]).
pub(super) fn open_websocket(
    url: &str,
    trust: &TlsTrust,
) -> Result<WebSocket<NetStream>, InternalErrors> {
    let uri: tungstenite::http::Uri = url
        .parse()
        .map_err(|e| InternalErrors::InvalidArgument(format!("Invalid server URL '{url}': {e}")))?;
    let encrypted = match uri.scheme_str() {
        Some("ws") => false,
        Some("wss") => true,
        _ => {
            return Err(InternalErrors::InvalidArgument(format!(
                "Server URL must start with wss:// or ws:// (got '{url}')"
            )));
        }
    };
    if *trust != TlsTrust::PublicAuthorities && !encrypted {
        return Err(InternalErrors::InvalidArgument(format!(
            "Certificate options (fingerprint, no verification) need a wss:// URL (got '{url}')"
        )));
    }
    let host = uri
        .host()
        .ok_or_else(|| InternalErrors::InvalidArgument(format!("No host in '{url}'")))?;
    let port = uri.port_u16().unwrap_or(DEFAULT_PORT);
    let request_url = format!("ws://{host}:{port}/");

    let stream = connect_tcp(host, port)?;
    stream.set_nodelay(true).ok();
    stream.set_read_timeout(Some(CONNECT_TIMEOUT))?;
    let stream = if encrypted {
        tls::connect(stream, host, trust, url)?
    } else {
        NetStream::Plain(stream)
    };
    let config = tungstenite::protocol::WebSocketConfig::default()
        .max_message_size(Some(MAX_MESSAGE_SIZE))
        .max_frame_size(Some(MAX_MESSAGE_SIZE));
    let (ws, _) =
        tungstenite::client::client_with_config(request_url.as_str(), stream, Some(config))
            .map_err(|e| {
                let hint = if encrypted {
                    ""
                } else {
                    " (if the server uses TLS, connect with wss://)"
                };
                InternalErrors::Io(format!("WebSocket handshake with {url} failed: {e}{hint}"))
            })?;
    Ok(ws)
}

#[derive(serde::Deserialize)]
enum LoginState {
    #[serde(alias = "accepted")]
    Accepted,
    #[serde(alias = "error")]
    Error,
}

/// Reply of an `evanalyzer server` to a text command.
#[derive(serde::Deserialize)]
struct LoginReply {
    response: LoginState,
    #[serde(default)]
    msg: String,
    session_token: Option<String>,
}

/// Sends the server's JSON login command and returns the session token.
pub(super) fn login(
    ws: &mut WebSocket<NetStream>,
    url: &str,
    username: &str,
    password: &str,
) -> Result<String, InternalErrors> {
    let request = serde_json::json!({
        "cmd": "login",
        "username": username,
        "password": password,
    });
    server_command(ws, url, request, "Login")
}

/// Attaches to the still-running worker of an earlier login with its
/// session token - no password needed. Fails once that worker has ended.
pub(super) fn resume(
    ws: &mut WebSocket<NetStream>,
    url: &str,
    session_token: &str,
) -> Result<(), InternalErrors> {
    let request = serde_json::json!({
        "cmd": "resume",
        "session_token": session_token,
    });
    server_command(ws, url, request, "Reconnecting").map(|_| ())
}

/// Sends a JSON command to an `evanalyzer server` and returns the session
/// token its acceptance carries. Text messages that aren't a reply (the
/// server's greeting) are skipped. `what` names the command in errors.
fn server_command(
    ws: &mut WebSocket<NetStream>,
    url: &str,
    request: serde_json::Value,
    what: &str,
) -> Result<String, InternalErrors> {
    ws.send(Message::text(request.to_string()))
        .map_err(|e| InternalErrors::Io(format!("Could not reach {url}: {e}")))?;
    ws.get_ref().tcp().set_read_timeout(Some(LOGIN_TIMEOUT))?;
    let reply = loop {
        let text = match ws.read() {
            Ok(Message::Text(text)) => text,
            Ok(Message::Binary(_)) => {
                return Err(InternalErrors::InvalidArgument(format!(
                    "{url} is not an EVAnalyzer server (connect with a token instead)"
                )));
            }
            Ok(Message::Close(_)) => {
                return Err(InternalErrors::Io(format!("{url} closed the connection")));
            }
            Ok(_) => continue,
            Err(e) => return Err(InternalErrors::Io(format!("No answer from {url}: {e}"))),
        };
        if let Ok(reply) = serde_json::from_str::<LoginReply>(&text) {
            break reply;
        }
    };
    ws.get_ref().tcp().set_read_timeout(Some(CONNECT_TIMEOUT))?;
    match (reply.response, reply.session_token) {
        (LoginState::Accepted, Some(token)) => Ok(token),
        (LoginState::Accepted, None) => Err(InternalErrors::Internal(format!(
            "{url} accepted the login but sent no session"
        ))),
        (LoginState::Error, _) => Err(InternalErrors::InvalidArgument(format!(
            "{what} at {url} failed: {}",
            reply.msg
        ))),
    }
}

fn connect_tcp(host: &str, port: u16) -> Result<TcpStream, InternalErrors> {
    let addrs = (host, port)
        .to_socket_addrs()
        .map_err(|e| InternalErrors::Io(format!("Could not resolve {host}: {e}")))?;
    let mut last_error = None;
    for addr in addrs {
        match TcpStream::connect_timeout(&addr, CONNECT_TIMEOUT) {
            Ok(stream) => return Ok(stream),
            Err(e) => last_error = Some(e),
        }
    }
    Err(InternalErrors::Io(match last_error {
        Some(e) => format!("Could not connect to {host}:{port}: {e}"),
        None => format!("{host} did not resolve to any address"),
    }))
}
