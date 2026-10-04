use crate::{
    api::{Request, Response},
    config::ServerConfig,
    session_management::{SessionManagement, default_store_path},
    tls,
    user_management::{AuthenticationStatus, UserManagement},
};
use log::{error, info, warn};
use rustls::{ServerConnection, StreamOwned};
use std::{
    collections::HashMap,
    io::{self, Read, Write},
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
    /// ID of the logged-in user ([`crate::user_management::User::user_id`]),
    /// set once the client logged in.
    pub user_id: Option<String>,
}

/// All open connections, shared between the connection threads.
pub type Sessions = Arc<Mutex<HashMap<SessionId, SessionInfo>>>;

/// A client's connection: TLS-encrypted (`wss://`) or plain (`ws://`).
enum ClientStream {
    Plain(TcpStream),
    Tls(Box<StreamOwned<ServerConnection, TcpStream>>),
}

impl Read for ClientStream {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        match self {
            Self::Plain(stream) => stream.read(buf),
            Self::Tls(stream) => stream.read(buf),
        }
    }
}

impl Write for ClientStream {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        match self {
            Self::Plain(stream) => stream.write(buf),
            Self::Tls(stream) => stream.write(buf),
        }
    }

    fn flush(&mut self) -> io::Result<()> {
        match self {
            Self::Plain(stream) => stream.flush(),
            Self::Tls(stream) => stream.flush(),
        }
    }
}

struct Server {
    /// `None`: clients connect without encryption (`tls.enabled = false`).
    tls: Option<Arc<rustls::ServerConfig>>,
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
    /// ID of the logged-in user.
    user_id: Option<String>,
    /// Session this connection logged into.
    session_token: Option<String>,
    /// Port of the session's worker on 127.0.0.1.
    worker_port: Option<u16>,
}

/// `evanalyzer server`: listens on `config.listen`, logs users in through
/// the configured user source, keeps its session file at
/// `config.session_store` and starts workers logging at `config.log_level`.
pub fn serve(config: ServerConfig) -> std::io::Result<()> {
    let session_store = config
        .session_store
        .clone()
        .unwrap_or_else(default_store_path);
    let tls = tls::load(&config.tls)?;
    match &tls {
        Some(tls) => info!(
            "Clients connect with wss:// - certificate fingerprint {}",
            tls.fingerprint
        ),
        None if is_loopback(&config.listen) => {
            info!("TLS is off: clients connect with ws://")
        }
        None => warn!(
            "TLS is off and {} is reachable from the network: passwords and data \
             travel unencrypted. Set tls.enabled = true unless something else encrypts.",
            config.listen
        ),
    }
    Server {
        tls: tls.map(|tls| tls.config),
        user_management: config.user_management()?,
        session_management: Arc::new(SessionManagement::new(session_store, config.log_level)?),
        sessions: Arc::default(),
        next_session_id: AtomicU64::new(1),
    }
    .serve(&config.listen)
}

impl Server {
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
                    user_id: None,
                },
            );
            info!(
                "Session {session_id}: {peer} connected ({})",
                if self.tls.is_some() {
                    "encrypted, TLS"
                } else {
                    "NOT encrypted"
                }
            );
            // Dropping `connection` removes the session again (see `Drop`).
            let mut connection = Connection {
                session_id,
                sessions: Arc::clone(&self.sessions),
                user_management: Arc::clone(&self.user_management),
                session_management: Arc::clone(&self.session_management),
                state: State::WaitingForLogin,
                user_id: None,
                session_token: None,
                worker_port: None,
            };
            let stream = match &self.tls {
                None => ClientStream::Plain(stream),
                Some(config) => match ServerConnection::new(Arc::clone(config)) {
                    Ok(tls) => ClientStream::Tls(Box::new(StreamOwned::new(tls, stream))),
                    Err(err) => {
                        warn!("Session {session_id}: cannot start TLS: {err}");
                        continue;
                    }
                },
            };
            thread::spawn(move || {
                // The TLS handshake runs inside the WebSocket handshake.
                let mut socket = match tungstenite::accept(stream) {
                    Ok(socket) => socket,
                    Err(err) => return warn!("Session {session_id}: handshake failed: {err}"),
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
    fn listen_for_data(socket: &mut WebSocket<ClientStream>) -> Option<Message> {
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
                            "Session {}: logged in as {} (id {})",
                            self.session_id, user.username, user.user_id
                        );
                        if let Some(info) = self.sessions.lock().unwrap().get_mut(&self.session_id)
                        {
                            info.user_id = Some(user.user_id.clone());
                        }
                        let session = match self.session_management.open_or_create_session(&user) {
                            Ok(session) => session,
                            Err(err) => {
                                error!("Cannot start EVAnalyzer for {}: {err}", user.username);
                                return Response::error("Could not start EVAnalyzer for this user");
                            }
                        };
                        self.user_id = Some(user.user_id);
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
            "Session {}: user {} exited",
            self.session_id,
            self.user_id.as_deref().unwrap_or_default()
        );
        if let Some(info) = self.sessions.lock().unwrap().get_mut(&self.session_id) {
            info.user_id = None;
        }
        self.user_id = None;
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
/// raw copying starts (nor does rustls, for an encrypted client).
fn forward_to_worker(client: WebSocket<ClientStream>, port: u16, hello: Message) -> io::Result<()> {
    let stream = TcpStream::connect((Ipv4Addr::LOCALHOST, port))?;
    stream.set_nodelay(true).ok();
    let (mut worker, _) = tungstenite::client(format!("ws://127.0.0.1:{port}/"), stream)
        .map_err(|e| io::Error::other(format!("handshake with worker failed: {e}")))?;
    worker.send(hello).map_err(io::Error::other)?;
    let worker = worker.get_ref().try_clone()?;
    match client.into_inner() {
        ClientStream::Plain(client) => pipe(client, worker),
        ClientStream::Tls(client) => pipe_tls(*client, worker),
    }
}

/// Is `listen` an address only this machine can reach?
fn is_loopback(listen: &str) -> bool {
    listen
        .parse::<SocketAddr>()
        .is_ok_and(|addr| addr.ip().is_loopback())
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

/// [`pipe`] for an encrypted client: decrypts what the client sends for the
/// worker and encrypts the worker's answers.
///
/// Like `pipe`, one thread per direction, so neither waits for the other: a
/// client and worker that both send large messages at once can't block each
/// other. The two threads share the TLS state (`tls`) but never hold it
/// while waiting on a socket. `client_writer` serializes writes to the
/// client socket: encrypted records must go out in the order they were
/// encrypted, so it is taken before `tls` and held until they're sent.
fn pipe_tls(client: StreamOwned<ServerConnection, TcpStream>, worker: TcpStream) -> io::Result<()> {
    let StreamOwned {
        conn: tls,
        sock: client,
    } = client;
    client.set_read_timeout(None)?;
    worker.set_read_timeout(None)?;
    let tls = Arc::new(Mutex::new(tls));
    let client_writer = Arc::new(Mutex::new(client.try_clone()?));

    let upstream = {
        let (tls, client_writer) = (Arc::clone(&tls), Arc::clone(&client_writer));
        let (mut client_read, mut worker_write) = (client.try_clone()?, worker.try_clone()?);
        thread::spawn(move || {
            let _ = decrypt_to_worker(&mut client_read, &tls, &client_writer, &mut worker_write);
            let _ = worker_write.shutdown(Shutdown::Both);
        })
    };

    let mut worker_read = worker;
    let _ = encrypt_to_client(&mut worker_read, &tls, &client_writer);
    // Tell the client we're done, then end the upstream copy as well.
    if let Ok(mut client) = client_writer.lock() {
        let mut records = Vec::new();
        if let Ok(mut tls) = tls.lock() {
            tls.send_close_notify();
            let _ = tls.write_tls(&mut records);
        }
        let _ = client.write_all(&records);
        let _ = client.shutdown(Shutdown::Both);
    }
    let _ = upstream.join();
    Ok(())
}

/// Client to worker: reads TLS records from the client, decrypts them and
/// writes the plaintext to the worker, until the client closes.
fn decrypt_to_worker(
    client: &mut TcpStream,
    tls: &Mutex<ServerConnection>,
    client_writer: &Mutex<TcpStream>,
    worker: &mut TcpStream,
) -> io::Result<()> {
    // What rustls already decrypted before the handover. (Nothing, as the
    // client waits for the worker's answer - but cheap to be sure.)
    let mut plaintext = Vec::new();
    let _ = tls.lock().unwrap().reader().read_to_end(&mut plaintext);
    worker.write_all(&plaintext)?;

    let mut buf = vec![0u8; 64 * 1024];
    loop {
        let n = client.read(&mut buf)?;
        if n == 0 {
            return Ok(());
        }
        plaintext.clear();
        let (closed, has_records) = {
            let mut tls = tls.lock().unwrap();
            let mut received = &buf[..n];
            let mut closed = false;
            while !received.is_empty() {
                tls.read_tls(&mut received)?;
                // Decrypt everything buffered; rustls stops at its
                // plaintext limit until that has been read.
                loop {
                    let state = tls.process_new_packets().map_err(io::Error::other)?;
                    let available = state.plaintext_bytes_to_read();
                    if available == 0 {
                        closed = state.peer_has_closed();
                        break;
                    }
                    let start = plaintext.len();
                    plaintext.resize(start + available, 0);
                    tls.reader().read_exact(&mut plaintext[start..])?;
                }
            }
            (closed, tls.wants_write())
        };
        // TLS messages of its own (e.g. a key update reply) for the client.
        if has_records {
            let mut client = client_writer.lock().unwrap();
            let mut records = Vec::new();
            {
                let mut tls = tls.lock().unwrap();
                while tls.wants_write() {
                    tls.write_tls(&mut records)?;
                }
            }
            client.write_all(&records)?;
        }
        worker.write_all(&plaintext)?;
        if closed {
            return Ok(());
        }
    }
}

/// Worker to client: reads plaintext from the worker, encrypts it and sends
/// it to the client, until the worker closes.
fn encrypt_to_client(
    worker: &mut TcpStream,
    tls: &Mutex<ServerConnection>,
    client_writer: &Mutex<TcpStream>,
) -> io::Result<()> {
    let mut buf = vec![0u8; 64 * 1024];
    loop {
        let n = worker.read(&mut buf)?;
        if n == 0 {
            return Ok(());
        }
        let mut client = client_writer.lock().unwrap();
        let mut records = Vec::new();
        {
            let mut tls = tls.lock().unwrap();
            // rustls buffers a limited amount of encrypted output: encrypt
            // piecewise, draining it in between.
            let mut rest = &buf[..n];
            while !rest.is_empty() {
                let taken = tls.writer().write(rest)?;
                if taken == 0 && !tls.wants_write() {
                    return Err(io::ErrorKind::WriteZero.into());
                }
                rest = &rest[taken..];
                while tls.wants_write() {
                    tls.write_tls(&mut records)?;
                }
            }
        }
        // Sent without holding `tls`, so decrypting the client's data goes
        // on even while a slow client takes its time to receive this.
        client.write_all(&records)?;
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

#[cfg(test)]
mod tests {
    use super::*;
    use rustls::ClientConnection;
    use rustls::client::danger::{HandshakeSignatureValid, ServerCertVerified, ServerCertVerifier};
    use rustls::pki_types::{CertificateDer, ServerName, UnixTime};

    /// A self-signed server certificate in `dir`, as a server without a
    /// configured certificate makes it.
    pub(super) fn test_tls(dir: &std::path::Path) -> tls::Tls {
        tls::load(&crate::config::TlsConfig {
            self_signed_dir: Some(dir.join("tls")),
            ..Default::default()
        })
        .unwrap()
        .unwrap()
    }

    /// Accepts exactly the certificate with this fingerprint - what the real
    /// client does with `--remote-fingerprint`.
    #[derive(Debug)]
    struct Pinned(String, Arc<rustls::crypto::CryptoProvider>);

    impl ServerCertVerifier for Pinned {
        fn verify_server_cert(
            &self,
            end_entity: &CertificateDer<'_>,
            _: &[CertificateDer<'_>],
            _: &ServerName<'_>,
            _: &[u8],
            _: UnixTime,
        ) -> Result<ServerCertVerified, rustls::Error> {
            if tls::fingerprint(end_entity) == self.0 {
                Ok(ServerCertVerified::assertion())
            } else {
                Err(rustls::Error::General("wrong certificate".into()))
            }
        }

        fn verify_tls12_signature(
            &self,
            message: &[u8],
            cert: &CertificateDer<'_>,
            dss: &rustls::DigitallySignedStruct,
        ) -> Result<HandshakeSignatureValid, rustls::Error> {
            rustls::crypto::verify_tls12_signature(
                message,
                cert,
                dss,
                &self.1.signature_verification_algorithms,
            )
        }

        fn verify_tls13_signature(
            &self,
            message: &[u8],
            cert: &CertificateDer<'_>,
            dss: &rustls::DigitallySignedStruct,
        ) -> Result<HandshakeSignatureValid, rustls::Error> {
            rustls::crypto::verify_tls13_signature(
                message,
                cert,
                dss,
                &self.1.signature_verification_algorithms,
            )
        }

        fn supported_verify_schemes(&self) -> Vec<rustls::SignatureScheme> {
            self.1.signature_verification_algorithms.supported_schemes()
        }
    }

    pub(super) type TlsClient = WebSocket<StreamOwned<ClientConnection, TcpStream>>;

    /// A `wss://` connection to `addr` that trusts the certificate with
    /// `fingerprint`.
    pub(super) fn connect_tls(addr: SocketAddr, fingerprint: &str) -> TlsClient {
        let provider = Arc::new(rustls::crypto::ring::default_provider());
        let config = rustls::ClientConfig::builder_with_provider(Arc::clone(&provider))
            .with_safe_default_protocol_versions()
            .unwrap()
            .dangerous()
            .with_custom_certificate_verifier(Arc::new(Pinned(fingerprint.into(), provider)))
            .with_no_client_auth();
        let tls = ClientConnection::new(
            Arc::new(config),
            ServerName::try_from("evanalyzer-server").unwrap(),
        )
        .unwrap();
        let stream = StreamOwned::new(tls, TcpStream::connect(addr).unwrap());
        tungstenite::client(format!("wss://{addr}/"), stream)
            .unwrap()
            .0
    }

    /// The server side of [`connect_tls`]'s connection.
    fn accept_tls(stream: TcpStream, tls: &tls::Tls) -> WebSocket<ClientStream> {
        let conn = ServerConnection::new(Arc::clone(&tls.config)).unwrap();
        tungstenite::accept(ClientStream::Tls(Box::new(StreamOwned::new(conn, stream)))).unwrap()
    }

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
            let mut socket = tungstenite::accept(ClientStream::Plain(stream)).unwrap();
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
    fn an_encrypted_connection_is_forwarded_to_the_worker_both_ways() {
        let dir = tempfile::tempdir().unwrap();
        let tls = test_tls(dir.path());
        let fingerprint = tls.fingerprint.clone();
        let worker_port = reversing_worker();
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let server = thread::spawn(move || {
            let (stream, _) = listener.accept().unwrap();
            let mut socket = accept_tls(stream, &tls);
            let hello = socket.read().unwrap();
            forward_to_worker(socket, worker_port, hello).unwrap();
        });

        let mut client = connect_tls(addr, &fingerprint);
        client.send(Message::binary(vec![1, 2, 3])).unwrap();
        assert_eq!(client.read().unwrap().into_data().as_ref(), &[3, 2, 1]);
        // Much larger than a TLS record and rustls' buffers.
        let big: Vec<u8> = (0..3_000_000u32).map(|i| (i * 7) as u8).collect();
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
            let socket = tungstenite::accept(ClientStream::Plain(stream)).unwrap();
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
        use crate::user_management::single_user::SingleUser;
        use std::net::TcpStream;
        use tungstenite::stream::MaybeTlsStream;

        type Client = WebSocket<MaybeTlsStream<TcpStream>>;

        /// A server on a free port with the single "admin/1234" user and
        /// workers replaced by a stand-in that just listens.
        fn start_server(dir: &std::path::Path) -> String {
            start_server_with(dir, None).0
        }

        /// [`start_server`], encrypting with `tls` if given. Also returns
        /// the address.
        fn start_server_with(
            dir: &std::path::Path,
            tls: Option<&tls::Tls>,
        ) -> (String, SocketAddr) {
            let port = TcpListener::bind("127.0.0.1:0")
                .unwrap()
                .local_addr()
                .unwrap()
                .port();
            let server = Server {
                tls: tls.map(|tls| Arc::clone(&tls.config)),
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
            let addr = SocketAddr::from((Ipv4Addr::LOCALHOST, port));
            for _ in 0..200 {
                if TcpStream::connect(addr).is_ok() {
                    return (url, addr);
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

        fn ask<S: Read + Write>(client: &mut WebSocket<S>, message: Message) -> serde_json::Value {
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

        #[test]
        fn with_tls_the_login_is_encrypted_and_plain_clients_are_turned_away() {
            let dir = tempfile::tempdir().unwrap();
            let tls = super::test_tls(dir.path());
            let (url, addr) = start_server_with(dir.path(), Some(&tls));

            let mut client = super::connect_tls(addr, &tls.fingerprint);
            let greeting = client.read().unwrap().into_text().unwrap();
            assert!(greeting.starts_with("Welcome"), "{greeting}");
            let reply = ask(
                &mut client,
                text(r#"{"cmd":"login","username":"admin","password":"1234"}"#),
            );
            assert_eq!(reply["response"], "Accepted");
            let reply = ask(&mut client, text(r#"{"cmd":"exit"}"#));
            assert_eq!(reply["msg"], "Session closed");

            // ws:// to a TLS server fails instead of hanging.
            let plain = TcpStream::connect(addr).unwrap();
            plain
                .set_read_timeout(Some(std::time::Duration::from_secs(5)))
                .unwrap();
            assert!(tungstenite::client(url.as_str(), plain).is_err());
        }
    }
}
