use crate::{
    api::{Request, Response},
    session_management::SessionManagement,
    user_management::{
        AuthenticationStatus, UserManagement, linux_users::LinuxUsers, single_user::SingleUser,
    },
};
use log::{error, info, warn};
use std::{
    collections::HashMap,
    net::{SocketAddr, TcpListener, TcpStream},
    sync::{
        Arc, Mutex,
        atomic::{AtomicU64, Ordering},
    },
    thread,
    time::SystemTime,
};
use tungstenite::{Message, WebSocket};

enum State {
    WaitingForLogin,
    WaitingForCommands,
}

pub type SessionId = u64;

/// What the server knows about one open WebSocket connection.
pub struct SessionInfo {
    pub peer: SocketAddr,
    pub connected_at: SystemTime,
    /// Set once the client logged in.
    pub username: Option<String>,
}

/// All open connections, shared between the connection threads.
pub type Sessions = Arc<Mutex<HashMap<SessionId, SessionInfo>>>;

struct Server {
    user_management: Arc<dyn UserManagement>,
    session_management: Arc<SessionManagement>,
    sessions: Sessions,
    next_session_id: AtomicU64,
}

/// One per client: owned by the connection's thread.
struct Connection {
    session_id: SessionId,
    sessions: Sessions,
    user_management: Arc<dyn UserManagement>,
    session_management: Arc<SessionManagement>,
    state: State,
    username: Option<String>,
    /// Session this connection logged into.
    session_token: Option<String>,
}

pub fn serve(listen: String) -> std::io::Result<()> {
    Server::new()?.serve(&listen)
}

impl Server {
    pub fn new() -> std::io::Result<Self> {
        Ok(Self {
            user_management: Arc::new(SingleUser::default()),
            session_management: Arc::new(SessionManagement::new()?),
            sessions: Arc::default(),
            next_session_id: AtomicU64::new(1),
        })
    }

    pub fn serve(&self, listen: &str) -> std::io::Result<()> {
        let listener = TcpListener::bind(listen)?;
        info!("Starting server on ws://{}", listener.local_addr()?);
        for stream in listener.incoming().flatten() {
            let Ok(peer) = stream.peer_addr() else {
                continue;
            };
            let session_id = self.next_session_id.fetch_add(1, Ordering::Relaxed);
            self.sessions.lock().unwrap().insert(
                session_id,
                SessionInfo {
                    peer,
                    connected_at: SystemTime::now(),
                    username: None,
                },
            );
            info!("Session {session_id}: {peer} connected");
            // Dropping `connection` removes the session again (see `Drop`).
            let mut connection = Connection {
                session_id,
                sessions: Arc::clone(&self.sessions),
                user_management: Arc::clone(&self.user_management),
                session_management: Arc::clone(&self.session_management),
                state: State::WaitingForLogin,
                username: None,
                session_token: None,
            };
            thread::spawn(move || {
                let mut socket = match tungstenite::accept(stream) {
                    Ok(socket) => socket,
                    Err(err) => return warn!("WebSocket handshake failed: {err}"),
                };
                // Greet the client right after it connected.
                if socket
                    .send(Message::text("Welcome to EVAnalyzer server."))
                    .is_err()
                {
                    return;
                }
                while let Some(data) = Self::listen_for_data(&mut socket) {
                    let command = String::from_utf8_lossy(&data)
                        .to_string()
                        .trim()
                        .to_string();
                    // Never log the content: login messages carry passwords.
                    log::debug!(
                        "Session {}: received {} bytes",
                        connection.session_id,
                        command.len()
                    );
                    let answer = connection.state_machine(command);
                    let json = serde_json::to_string(&answer).expect("Response serializes");
                    if socket.send(Message::text(json)).is_err() {
                        break;
                    }
                }
            });
        }
        Ok(())
    }

    /// Waits for the next text or binary message and returns its bytes.
    /// `None` when the client disconnected.
    pub fn listen_for_data(socket: &mut WebSocket<TcpStream>) -> Option<Vec<u8>> {
        loop {
            match socket.read().ok()? {
                Message::Text(text) => return Some(text.as_bytes().to_vec()),
                Message::Binary(bytes) => return Some(bytes.to_vec()),
                Message::Close(_) => return None,
                _ => {} // ping/pong are handled by tungstenite
            }
        }
    }
}

impl Connection {
    /// Handles one command from the client and returns the answer to send.
    pub fn state_machine(&mut self, command: String) -> Response {
        let request = match serde_json::from_str::<Request>(&command) {
            Ok(request) => request,
            Err(err) => return Response::error(format!("Invalid request: {err}")),
        };
        match (&self.state, request) {
            (State::WaitingForLogin, Request::Login { username, password }) => {
                match self.user_management.login(username.clone(), password) {
                    AuthenticationStatus::Authenticated(user) => {
                        info!(
                            "Session {}: logged in as {}",
                            self.session_id, user.username
                        );
                        if let Some(info) = self.sessions.lock().unwrap().get_mut(&self.session_id)
                        {
                            info.username = Some(user.username.clone());
                        }
                        let session = match self.session_management.open_or_create_session(&user) {
                            Ok(session) => session,
                            Err(err) => {
                                error!("Cannot start EVAnalyzer for {}: {err}", user.username);
                                return Response::error("Could not start EVAnalyzer for this user");
                            }
                        };
                        // TODO: forward this connection to `session.worker_url()`,
                        // authenticating with `session.worker_token`.
                        self.username = Some(user.username);
                        self.session_token = Some(session.session_token.clone());
                        self.state = State::WaitingForCommands;
                        Response {
                            session_token: Some(session.session_token),
                            ..Response::accepted("Logged in")
                        }
                    }
                    // Same answer for both, so clients can't probe which
                    // usernames exist.
                    AuthenticationStatus::UserNotFound | AuthenticationStatus::PasswordWrong => {
                        warn!("Session {}: failed login for '{username}'", self.session_id);
                        Response::error("Invalid username or password")
                    }
                }
            }
            (State::WaitingForCommands, Request::Login { .. }) => {
                Response::error("Already logged in")
            }
            (State::WaitingForCommands, Request::Exit) => self.exit(),
            (State::WaitingForLogin, Request::Exit) => Response::error("Not logged in"),
        }
    }
}

impl Connection {
    /// Stops the user's EVAnalyzer worker, closes the session and logs this
    /// connection out. The WebSocket stays open for a new login.
    fn exit(&mut self) -> Response {
        if let Some(token) = self.session_token.take()
            && let Err(err) = self.session_management.close_session(&token)
        {
            error!("Session {}: closing failed: {err}", self.session_id);
            return Response::error("Could not close the session");
        }
        info!(
            "Session {}: {} exited",
            self.session_id,
            self.username.as_deref().unwrap_or_default()
        );
        if let Some(info) = self.sessions.lock().unwrap().get_mut(&self.session_id) {
            info.username = None;
        }
        self.username = None;
        self.state = State::WaitingForLogin;
        Response::accepted("Session closed")
    }
}

impl Drop for Connection {
    /// Runs when the connection's thread ends, also after a panic.
    fn drop(&mut self) {
        // `lock()` fails only if another thread panicked while holding it.
        if let Ok(mut sessions) = self.sessions.lock() {
            sessions.remove(&self.session_id);
        }
        info!("Session {} closed", self.session_id);
    }
}
