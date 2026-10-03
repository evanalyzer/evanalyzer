use crate::{
    api::{Request, Response},
    session_management::SessionManagement,
    user_management::{AuthenticationStatus, UserManagement, single_user::SingleUser},
};
use log::{error, info, warn};
use std::{
    collections::HashMap,
    io,
    net::{Ipv4Addr, Shutdown, SocketAddr, TcpListener, TcpStream},
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
    pub _peer: SocketAddr,
    pub _connected_at: SystemTime,
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
    /// Port of the session's worker on 127.0.0.1.
    worker_port: Option<u16>,
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
                    _peer: peer,
                    _connected_at: SystemTime::now(),
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
                worker_port: None,
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
                while let Some(message) = Self::listen_for_data(&mut socket) {
                    // Text messages are commands for this server; the first
                    // binary message (the remote protocol's `Hello`) hands
                    // the connection over to the user's worker.
                    let Message::Text(text) = message else {
                        match connection.worker_port {
                            Some(port) => {
                                let id = connection.session_id;
                                info!("Session {id}: forwarding to worker on port {port}");
                                if let Err(err) = forward_to_worker(socket, port, message) {
                                    warn!("Session {id}: forwarding to worker failed: {err}");
                                }
                                return;
                            }
                            None => {
                                let answer = Response::error("Log in first");
                                let json =
                                    serde_json::to_string(&answer).expect("Response serializes");
                                if socket.send(Message::text(json)).is_err() {
                                    break;
                                }
                                continue;
                            }
                        }
                    };
                    let command = text.trim().to_string();
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

    /// Waits for the next text or binary message. `None` when the client
    /// disconnected.
    pub fn listen_for_data(socket: &mut WebSocket<TcpStream>) -> Option<Message> {
        loop {
            match socket.read().ok()? {
                message @ (Message::Text(_) | Message::Binary(_)) => return Some(message),
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
                        self.username = Some(user.username);
                        self.worker_port = Some(session.port);
                        self.session_token = Some(session.session_token.clone());
                        self.state = State::WaitingForCommands;
                        Response {
                            session_token: Some(session.session_token),
                            ..Response::accepted("Logged in")
                        }
                    }
                    // Same answer for both, so clients can't probe which
                    // usernames exist.
                    AuthenticationStatus::PasswordWrong => {
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
        self.worker_port = None;
        self.state = State::WaitingForLogin;
        Response::accepted("Session closed")
    }
}

/// Hands the client's connection over to its worker: opens a WebSocket to
/// the worker, passes on the client's first binary message (the remote
/// protocol's `Hello`, whose token the worker checks) and from then on copies
/// raw bytes both ways until one side closes.
///
/// Copying bytes works because the frames already have the form the other
/// side expects: the client's frames are masked, as a server (the worker)
/// requires from its client, and the worker's frames are unmasked, as the
/// client requires from its server. The client sends nothing after `Hello`
/// until the worker answered, so tungstenite holds no unread data when the
/// raw copying starts.
fn forward_to_worker(client: WebSocket<TcpStream>, port: u16, hello: Message) -> io::Result<()> {
    let stream = TcpStream::connect((Ipv4Addr::LOCALHOST, port))?;
    stream.set_nodelay(true).ok();
    let (mut worker, _) = tungstenite::client(format!("ws://127.0.0.1:{port}/"), stream)
        .map_err(|e| io::Error::other(format!("handshake with worker failed: {e}")))?;
    worker.send(hello).map_err(io::Error::other)?;
    pipe(client.get_ref().try_clone()?, worker.get_ref().try_clone()?)
}

/// Copies bytes between both sockets in both directions until either side
/// closes, then closes the other one too.
fn pipe(client: TcpStream, worker: TcpStream) -> io::Result<()> {
    client.set_read_timeout(None)?;
    worker.set_read_timeout(None)?;
    let (mut client_read, mut worker_write) = (client.try_clone()?, worker.try_clone()?);
    let upstream = thread::spawn(move || {
        let _ = io::copy(&mut client_read, &mut worker_write);
        let _ = worker_write.shutdown(Shutdown::Both);
    });
    let (mut worker_read, mut client_write) = (worker, client);
    let _ = io::copy(&mut worker_read, &mut client_write);
    // Also ends the upstream copy if the worker went away first.
    let _ = client_write.shutdown(Shutdown::Both);
    let _ = upstream.join();
    Ok(())
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

#[cfg(test)]
mod tests {
    use super::*;

    /// A worker that answers every binary message with its bytes reversed.
    fn reversing_worker() -> u16 {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        thread::spawn(move || {
            let (stream, _) = listener.accept().unwrap();
            let mut ws = tungstenite::accept(stream).unwrap();
            while let Ok(Message::Binary(bytes)) = ws.read() {
                let reversed: Vec<u8> = bytes.iter().rev().copied().collect();
                ws.send(Message::binary(reversed)).unwrap();
            }
        });
        port
    }

    #[test]
    fn forwarded_connection_reaches_the_worker_both_ways() {
        let worker_port = reversing_worker();
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let url = format!("ws://{}", listener.local_addr().unwrap());
        let server = thread::spawn(move || {
            let (stream, _) = listener.accept().unwrap();
            let mut socket = tungstenite::accept(stream).unwrap();
            let hello = socket.read().unwrap();
            forward_to_worker(socket, worker_port, hello).unwrap();
        });

        let (mut client, _) = tungstenite::connect(url).unwrap();
        client.send(Message::binary(vec![1, 2, 3])).unwrap();
        assert_eq!(client.read().unwrap().into_data().as_ref(), &[3, 2, 1]);
        // Large messages and many frames pass the raw copy unchanged.
        let big: Vec<u8> = (0..1_000_000u32).map(|i| i as u8).collect();
        for _ in 0..3 {
            client.send(Message::binary(big.clone())).unwrap();
            let back = client.read().unwrap().into_data();
            assert!(back.iter().eq(big.iter().rev()));
        }

        client.close(None).unwrap();
        let _ = client.read();
        server.join().unwrap();
    }

    #[test]
    fn forwarding_to_a_missing_worker_fails() {
        let port = TcpListener::bind("127.0.0.1:0")
            .unwrap()
            .local_addr()
            .unwrap()
            .port();
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let url = format!("ws://{}", listener.local_addr().unwrap());
        let server = thread::spawn(move || {
            let (stream, _) = listener.accept().unwrap();
            let socket = tungstenite::accept(stream).unwrap();
            forward_to_worker(socket, port, Message::binary(vec![0]))
        });
        let (_client, _) = tungstenite::connect(url).unwrap();
        assert!(server.join().unwrap().is_err());
    }

    // -- the whole conversation, through the real accept loop ---------------

    #[cfg(unix)]
    mod conversation {
        use super::super::*;
        use crate::session_management::tests::fake_worker;
        use std::net::TcpStream;
        use tungstenite::stream::MaybeTlsStream;

        type Client = WebSocket<MaybeTlsStream<TcpStream>>;

        /// A server on a free port with the single "admin/1234" user and
        /// workers replaced by a stand-in that just listens.
        fn start_server(dir: &std::path::Path) -> String {
            let port = TcpListener::bind("127.0.0.1:0")
                .unwrap()
                .local_addr()
                .unwrap()
                .port();
            let server = Server {
                user_management: Arc::new(SingleUser::default()),
                session_management: Arc::new(
                    SessionManagement::with_store(dir.join("sessions.json"), fake_worker(dir))
                        .unwrap(),
                ),
                sessions: Arc::default(),
                next_session_id: AtomicU64::new(1),
            };
            let listen = format!("127.0.0.1:{port}");
            thread::spawn(move || server.serve(&listen));
            let url = format!("ws://127.0.0.1:{port}");
            for _ in 0..200 {
                if TcpStream::connect(("127.0.0.1", port)).is_ok() {
                    return url;
                }
                thread::sleep(std::time::Duration::from_millis(10));
            }
            panic!("server did not start");
        }

        fn connect(url: &str) -> Client {
            let (mut client, _) = tungstenite::connect(url).unwrap();
            let greeting = client.read().unwrap().into_text().unwrap();
            assert!(greeting.starts_with("Welcome"), "{greeting}");
            client
        }

        fn ask(client: &mut Client, message: Message) -> serde_json::Value {
            client.send(message).unwrap();
            let reply = client.read().unwrap().into_text().unwrap();
            serde_json::from_str(&reply).unwrap()
        }

        fn text(json: &str) -> Message {
            Message::text(json.to_string())
        }

        #[test]
        fn login_exit_and_the_errors_in_between() {
            let dir = tempfile::tempdir().unwrap();
            let url = start_server(dir.path());
            let mut client = connect(&url);

            let reply = ask(&mut client, Message::binary(vec![1, 2, 3]));
            assert_eq!(reply["msg"], "Log in first");
            let reply = ask(&mut client, text("not json"));
            assert_eq!(reply["response"], "Error");
            assert!(
                reply["msg"]
                    .as_str()
                    .unwrap()
                    .starts_with("Invalid request")
            );
            let reply = ask(&mut client, text(r#"{"cmd":"exit"}"#));
            assert_eq!(reply["msg"], "Not logged in");
            let reply = ask(
                &mut client,
                text(r#"{"cmd":"login","username":"admin","password":"nope"}"#),
            );
            assert_eq!(reply["msg"], "Invalid username or password");

            let reply = ask(
                &mut client,
                text(r#"{"cmd":"login","username":"admin","password":"1234"}"#),
            );
            assert_eq!(reply["response"], "Accepted");
            let token = reply["session_token"].as_str().unwrap().to_string();
            assert_eq!(token.len(), 48);

            let reply = ask(
                &mut client,
                text(r#"{"cmd":"login","username":"admin","password":"1234"}"#),
            );
            assert_eq!(reply["msg"], "Already logged in");

            // A second connection of the same user gets the same session.
            let mut second = connect(&url);
            let reply = ask(
                &mut second,
                text(r#"{"cmd":"login","username":"admin","password":"1234"}"#),
            );
            assert_eq!(reply["session_token"], token.as_str());

            let reply = ask(&mut client, text(r#"{"cmd":"exit"}"#));
            assert_eq!(reply["msg"], "Session closed");
            // Logged out again: commands need a new login.
            let reply = ask(&mut client, text(r#"{"cmd":"exit"}"#));
            assert_eq!(reply["msg"], "Not logged in");
            client.close(None).unwrap();
        }
    }
}
