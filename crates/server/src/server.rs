use crate::{
    api::{Request, Response},
    session_management::SessionManagement,
    user_management::{
        AuthenticationStatus, UserManagement, linux_users::LinuxUsers, single_user::SingleUser,
    },
};
use log::{info, warn};
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
    session_management: SessionManagement,
    sessions: Sessions,
    next_session_id: AtomicU64,
}

/// One per client: owned by the connection's thread.
struct Connection {
    session_id: SessionId,
    sessions: Sessions,
    user_management: Arc<dyn UserManagement>,
    state: State,
    username: Option<String>,
}

pub fn serve(listen: String) -> std::io::Result<()> {
    Server::new().serve(&listen)
}

impl Server {
    pub fn new() -> Self {
        Self {
            user_management: Arc::new(SingleUser::default()),
            session_management: SessionManagement {},
            sessions: Arc::default(),
            next_session_id: AtomicU64::new(1),
        }
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
                state: State::WaitingForLogin,
                username: None,
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
                    info!("Session {}: received {command}", connection.session_id);
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
                        self.username = Some(user.username);
                        self.state = State::WaitingForCommands;
                        Response::accepted("Logged in")
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
        }
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
