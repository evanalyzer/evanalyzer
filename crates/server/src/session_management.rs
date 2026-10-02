//! Keeps one EVAnalyzer worker process (`evanalyzer serve`) per logged-in
//! user and records them in a session file, so a restarted server finds the
//! still running workers again.
//!
//! The file lives in the runtime directory (`/run/evanalyzer/` when running
//! as root, `$XDG_RUNTIME_DIR/evanalyzer/` otherwise): runtime state that must
//! not survive a reboot, just like the processes it describes. It is written
//! atomically (temp file + rename) and readable by the server's user only,
//! because it holds the worker tokens.

use crate::user_management::User;
use log::{info, warn};
use serde::{Deserialize, Serialize};
use std::{
    collections::HashMap,
    fs, io,
    net::{Ipv4Addr, SocketAddr, TcpListener, TcpStream},
    path::{Path, PathBuf},
    process::{Child, Command, Stdio},
    sync::Mutex,
    thread,
    time::{Duration, Instant, SystemTime},
};

/// How long a freshly started worker may take until it accepts connections.
const WORKER_STARTUP_TIMEOUT: Duration = Duration::from_secs(30);

/// Environment variables passed on to workers; everything else is dropped.
const WORKER_ENV: &[&str] = &["PATH", "LANG", "LC_ALL", "RUST_LOG", "LD_LIBRARY_PATH"];

#[derive(Clone, Serialize, Deserialize)]
pub struct SessionEntry {
    /// Process ID of the started evanalyzer instance for this session
    pub pid: u32,
    /// Token the client presents to get back into this session
    pub session_token: String,
    /// User id of the user which started the session
    pub user_id: String,
    pub username: String,
    /// The worker listens on `127.0.0.1:<port>`
    pub port: u16,
    /// Token the worker expects (`EVANALYZER_SERVER_TOKEN`); only the server knows it
    pub worker_token: String,
    /// Date time when the session has been started
    pub start_date: SystemTime,
}

impl SessionEntry {
    pub fn worker_url(&self) -> String {
        format!("ws://127.0.0.1:{}", self.port)
    }
}

pub struct SessionManagement {
    pub path_to_session_store: PathBuf,
    /// Executable started as worker, with `serve --listen 127.0.0.1:<port>`.
    pub worker_command: PathBuf,
    state: Mutex<State>,
}

#[derive(Default)]
struct State {
    sessions: Vec<SessionEntry>,
    /// Workers started by this server process. Workers taken over from the
    /// session file after a restart are not in here.
    children: HashMap<u32, Child>,
}

impl SessionManagement {
    /// Session file in the default runtime directory, the running
    /// executable as worker.
    pub fn new() -> io::Result<Self> {
        Self::with_store(default_store_path(), std::env::current_exe()?)
    }

    /// Loads the session file and keeps the sessions whose worker still runs.
    pub fn with_store(path_to_session_store: PathBuf, worker_command: PathBuf) -> io::Result<Self> {
        let mut sessions = load(&path_to_session_store);
        let before = sessions.len();
        sessions.retain(|s| worker_responds(s.port));
        info!(
            "Session store {}: {} running session(s), {} stale dropped",
            path_to_session_store.display(),
            sessions.len(),
            before - sessions.len()
        );
        let manager = Self {
            path_to_session_store,
            worker_command,
            state: Mutex::new(State {
                sessions,
                children: HashMap::new(),
            }),
        };
        manager.persist(&manager.state.lock().unwrap().sessions)?;
        Ok(manager)
    }

    /// Returns the user's running session, or starts a worker for a new one.
    ///
    /// Holds the lock while a worker starts, so the same user logging in
    /// twice at once can't end up with two workers.
    pub fn open_or_create_session(&self, user: &User) -> io::Result<SessionEntry> {
        let mut state = self.state.lock().unwrap();
        if let Some(index) = state
            .sessions
            .iter()
            .position(|s| s.username == user.username)
        {
            if state.is_alive(index) {
                info!("Restoring session of {}", user.username);
                return Ok(state.sessions[index].clone());
            }
            warn!("Worker of {} is gone, starting a new one", user.username);
            state.remove(index);
        }
        let entry = self.create_session(&mut state, user)?;
        self.persist(&state.sessions)?;
        Ok(entry)
    }

    /// Looks up an open session by the token the client got at login.
    pub fn find_session(&self, session_token: &str) -> Option<SessionEntry> {
        let state = self.state.lock().unwrap();
        state
            .sessions
            .iter()
            .find(|s| constant_time_eq(s.session_token.as_bytes(), session_token.as_bytes()))
            .cloned()
    }

    /// Stops the session's worker and forgets the session.
    pub fn close_session(&self, session_token: &str) -> io::Result<()> {
        let mut state = self.state.lock().unwrap();
        if let Some(index) = state
            .sessions
            .iter()
            .position(|s| s.session_token == session_token)
        {
            let entry = state.remove(index);
            info!("Closed session of {}", entry.username);
            self.persist(&state.sessions)?;
        }
        Ok(())
    }

    fn create_session(&self, state: &mut State, user: &User) -> io::Result<SessionEntry> {
        let port = free_port()?;
        let worker_token = generate_token()?;
        let mut command = Command::new(&self.worker_command);
        command
            .arg("serve")
            .arg("--listen")
            .arg(format!("127.0.0.1:{port}"))
            .stdin(Stdio::null())
            .env_clear()
            .envs(
                WORKER_ENV
                    .iter()
                    .filter_map(|k| Some((k, std::env::var_os(k)?))),
            )
            // Not on the command line: that is visible to every local user.
            .env("EVANALYZER_SERVER_TOKEN", &worker_token);
        run_as(&mut command, user);

        let mut child = command.spawn()?;
        if let Err(err) = wait_until_listening(&mut child, port) {
            let _ = child.kill();
            let _ = child.wait();
            return Err(err);
        }
        let entry = SessionEntry {
            pid: child.id(),
            session_token: generate_token()?,
            user_id: user.user_id.clone(),
            username: user.username.clone(),
            port,
            worker_token,
            start_date: SystemTime::now(),
        };
        info!(
            "Started worker for {} (pid {}, port {port})",
            user.username, entry.pid
        );
        state.children.insert(entry.pid, child);
        state.sessions.push(entry.clone());
        Ok(entry)
    }

    /// Writes the sessions to a temp file and renames it over the store, so
    /// readers never see a half-written file.
    fn persist(&self, sessions: &[SessionEntry]) -> io::Result<()> {
        let path = &self.path_to_session_store;
        if let Some(dir) = path.parent().filter(|d| !d.as_os_str().is_empty()) {
            create_private_dir(dir)?;
        }
        let tmp = path.with_extension("json.tmp");
        let _ = fs::remove_file(&tmp);
        let mut options = fs::OpenOptions::new();
        options.write(true).create_new(true);
        #[cfg(unix)]
        std::os::unix::fs::OpenOptionsExt::mode(&mut options, 0o600);
        let mut file = options.open(&tmp)?;
        serde_json::to_writer_pretty(&mut file, sessions)?;
        file.sync_all()?;
        fs::rename(&tmp, path)
    }
}

impl State {
    fn is_alive(&mut self, index: usize) -> bool {
        let entry = &self.sessions[index];
        match self.children.get_mut(&entry.pid) {
            Some(child) => matches!(child.try_wait(), Ok(None)),
            // Taken over from the session file: all we can do is knock.
            None => worker_responds(entry.port),
        }
    }

    /// Forgets the session and stops its worker if this process started it.
    fn remove(&mut self, index: usize) -> SessionEntry {
        let entry = self.sessions.remove(index);
        if let Some(mut child) = self.children.remove(&entry.pid) {
            let _ = child.kill();
            let _ = child.wait();
        } else if worker_responds(entry.port) {
            // Started by an earlier server process: no `Child` handle, so
            // stop it by PID. Only while its port still answers, so a PID
            // reused by an unrelated process after the worker died is left
            // alone.
            terminate(entry.pid);
        }
        entry
    }
}

/// `/run/evanalyzer/sessions.json` for a system service, otherwise the
/// user's runtime directory.
pub fn default_store_path() -> PathBuf {
    let system = Path::new("/run/evanalyzer");
    if create_private_dir(system).is_ok() {
        return system.join("sessions.json");
    }
    let base = std::env::var_os("XDG_RUNTIME_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(std::env::temp_dir);
    base.join("evanalyzer").join("sessions.json")
}

fn load(path: &Path) -> Vec<SessionEntry> {
    match fs::read(path) {
        Ok(bytes) => serde_json::from_slice(&bytes).unwrap_or_else(|err| {
            warn!("Ignoring unreadable session file {}: {err}", path.display());
            Vec::new()
        }),
        Err(err) if err.kind() == io::ErrorKind::NotFound => Vec::new(),
        Err(err) => {
            warn!("Cannot read session file {}: {err}", path.display());
            Vec::new()
        }
    }
}

fn create_private_dir(dir: &Path) -> io::Result<()> {
    let mut builder = fs::DirBuilder::new();
    builder.recursive(true);
    #[cfg(unix)]
    std::os::unix::fs::DirBuilderExt::mode(&mut builder, 0o700);
    builder.create(dir)
}

/// Asks the OS for an unused port. Another process could grab it before
/// the worker binds it; the worker then fails to start and login reports it.
fn free_port() -> io::Result<u16> {
    Ok(TcpListener::bind((Ipv4Addr::LOCALHOST, 0))?
        .local_addr()?
        .port())
}

fn worker_responds(port: u16) -> bool {
    let addr = SocketAddr::from((Ipv4Addr::LOCALHOST, port));
    TcpStream::connect_timeout(&addr, Duration::from_millis(500)).is_ok()
}

fn wait_until_listening(child: &mut Child, port: u16) -> io::Result<()> {
    let deadline = Instant::now() + WORKER_STARTUP_TIMEOUT;
    while Instant::now() < deadline {
        if let Some(status) = child.try_wait()? {
            return Err(io::Error::other(format!(
                "worker exited during startup: {status}"
            )));
        }
        if worker_responds(port) {
            return Ok(());
        }
        thread::sleep(Duration::from_millis(100));
    }
    Err(io::Error::new(
        io::ErrorKind::TimedOut,
        "worker did not start listening in time",
    ))
}

/// When the server runs as root, the worker runs as the logged-in user, in
/// their home folder - so it can only touch that user's files. Otherwise all
/// workers run as the server's user.
#[cfg(unix)]
fn run_as(command: &mut Command, user: &User) {
    use std::os::unix::{fs::MetadataExt, process::CommandExt};
    let Some(account) = &user.unix_account else {
        return;
    };
    let is_root = fs::metadata("/proc/self").is_ok_and(|m| m.uid() == 0);
    if !is_root {
        warn!(
            "Server is not root: worker of {} runs as the server's user",
            user.username
        );
        return;
    }
    // std drops the supplementary groups when switching the uid as root.
    command
        .uid(account.uid)
        .gid(account.gid)
        .current_dir(&account.home)
        .env("HOME", &account.home)
        .env("USER", &user.username)
        .env("LOGNAME", &user.username);
}

#[cfg(not(unix))]
fn run_as(_command: &mut Command, _user: &User) {}

fn terminate(pid: u32) {
    #[cfg(unix)]
    let status = Command::new("kill").arg(pid.to_string()).status();
    #[cfg(windows)]
    let status = Command::new("taskkill")
        .args(["/PID", &pid.to_string(), "/F"])
        .status();
    if !status.is_ok_and(|s| s.success()) {
        warn!("Could not stop worker process {pid}");
    }
}

fn generate_token() -> io::Result<String> {
    let mut bytes = [0u8; 24];
    getrandom::fill(&mut bytes).map_err(|e| io::Error::other(e.to_string()))?;
    Ok(bytes.iter().map(|b| format!("{b:02x}")).collect())
}

fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    a.len() == b.len() && a.iter().zip(b).fold(0u8, |acc, (x, y)| acc | (x ^ y)) == 0
}

#[cfg(all(test, unix))]
mod tests {
    use super::*;
    use std::os::unix::fs::PermissionsExt;

    /// Stand-in for `evanalyzer serve --listen 127.0.0.1:<port>`: just
    /// listens on the port.
    fn fake_worker(dir: &Path) -> PathBuf {
        let path = dir.join("fake-worker");
        fs::write(
            &path,
            "#!/bin/sh\nexec python3 -m http.server \"${3##*:}\" --bind 127.0.0.1 >/dev/null 2>&1\n",
        )
        .unwrap();
        fs::set_permissions(&path, fs::Permissions::from_mode(0o755)).unwrap();
        path
    }

    fn user(name: &str) -> User {
        User {
            user_id: name.into(),
            username: name.into(),
            unix_account: None,
        }
    }

    fn manager(dir: &Path) -> SessionManagement {
        SessionManagement::with_store(dir.join("run/sessions.json"), fake_worker(dir)).unwrap()
    }

    #[test]
    fn login_starts_a_worker_and_reuses_it() {
        let dir = tempfile::tempdir().unwrap();
        let sessions = manager(dir.path());

        let first = sessions.open_or_create_session(&user("alice")).unwrap();
        assert!(worker_responds(first.port));
        let again = sessions.open_or_create_session(&user("alice")).unwrap();
        assert_eq!(again.pid, first.pid);
        assert_eq!(again.session_token, first.session_token);

        let bob = sessions.open_or_create_session(&user("bob")).unwrap();
        assert_ne!(bob.port, first.port);
        assert_ne!(bob.worker_token, first.worker_token);

        sessions.close_session(&first.session_token).unwrap();
        sessions.close_session(&bob.session_token).unwrap();
    }

    #[test]
    fn store_is_private_and_survives_a_server_restart() {
        let dir = tempfile::tempdir().unwrap();
        let sessions = manager(dir.path());
        let entry = sessions.open_or_create_session(&user("alice")).unwrap();

        let store = &sessions.path_to_session_store;
        assert_eq!(
            fs::metadata(store).unwrap().permissions().mode() & 0o777,
            0o600
        );
        let dir_mode = fs::metadata(store.parent().unwrap())
            .unwrap()
            .permissions()
            .mode();
        assert_eq!(dir_mode & 0o777, 0o700);

        // A new server process finds the running worker again.
        let restarted = manager(dir.path());
        let restored = restarted.find_session(&entry.session_token).unwrap();
        assert_eq!(restored.pid, entry.pid);
        assert_eq!(
            restarted
                .open_or_create_session(&user("alice"))
                .unwrap()
                .pid,
            entry.pid
        );

        sessions.close_session(&entry.session_token).unwrap();
    }

    #[test]
    fn closing_stops_the_worker_and_forgets_the_session() {
        let dir = tempfile::tempdir().unwrap();
        let sessions = manager(dir.path());
        let entry = sessions.open_or_create_session(&user("alice")).unwrap();

        sessions.close_session(&entry.session_token).unwrap();
        assert!(sessions.find_session(&entry.session_token).is_none());
        assert!(!worker_responds(entry.port));
        assert!(
            manager(dir.path())
                .find_session(&entry.session_token)
                .is_none()
        );
    }

    #[test]
    fn closing_after_a_restart_stops_the_taken_over_worker() {
        let dir = tempfile::tempdir().unwrap();
        let entry = manager(dir.path())
            .open_or_create_session(&user("alice"))
            .unwrap();

        let restarted = manager(dir.path());
        restarted.close_session(&entry.session_token).unwrap();
        let deadline = Instant::now() + Duration::from_secs(5);
        while worker_responds(entry.port) && Instant::now() < deadline {
            thread::sleep(Duration::from_millis(50));
        }
        assert!(!worker_responds(entry.port));
    }

    #[test]
    fn crashed_worker_is_replaced_on_next_login() {
        let dir = tempfile::tempdir().unwrap();
        let sessions = manager(dir.path());
        let entry = sessions.open_or_create_session(&user("alice")).unwrap();
        {
            let mut state = sessions.state.lock().unwrap();
            let child = state.children.get_mut(&entry.pid).unwrap();
            child.kill().unwrap();
            child.wait().unwrap();
        }

        let fresh = sessions.open_or_create_session(&user("alice")).unwrap();
        assert_ne!(fresh.pid, entry.pid);
        assert_ne!(fresh.session_token, entry.session_token);
        sessions.close_session(&fresh.session_token).unwrap();
    }

    #[test]
    fn worker_that_fails_to_start_is_reported() {
        let dir = tempfile::tempdir().unwrap();
        let sessions = SessionManagement::with_store(
            dir.path().join("sessions.json"),
            PathBuf::from("/bin/false"),
        )
        .unwrap();
        assert!(sessions.open_or_create_session(&user("alice")).is_err());
        assert!(sessions.state.lock().unwrap().sessions.is_empty());
    }
}
