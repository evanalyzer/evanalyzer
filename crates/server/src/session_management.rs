//! Keeps one EVAnalyzer worker process (`evanalyzer worker`) per logged-in
//! user and records them in a session file, so a restarted server finds the
//! still running workers again.
//!
//! The file lives wherever `evanalyzer server --session-store` says - by
//! default in the runtime directory (`/run/evanalyzer/` when running as root,
//! the temp folder otherwise, see [`default_store_path`]): runtime state that
//! must not survive a reboot, just like the processes it describes. It is
//! written atomically (temp file + rename) and readable by the server's user
//! only, because it holds the worker tokens.

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

/// The only environment variables a worker gets: what the operating system
/// needs to run a process at all. Everything EVAnalyzer itself is configured
/// with (home, allowed folders, log level) travels as arguments instead.
const WORKER_OS_ENV: &[&str] = &["PATH"];

/// On Windows also these: without `SystemRoot` in particular, Winsock can't
/// load its provider DLLs and the worker's `bind` fails with
/// WSAEPROVIDERFAILEDINIT (os error 10106); `TEMP`/`TMP` are where the OS
/// puts temporary files. (Variable names are case-insensitive on Windows, so
/// `std::env::var_os` finds them however they are spelled.)
#[cfg(windows)]
const WINDOWS_WORKER_OS_ENV: &[&str] = &["SystemRoot", "windir", "SystemDrive", "TEMP", "TMP"];

/// The variables of the server's environment a worker gets - see
/// [`WORKER_OS_ENV`] and, on Windows, [`WINDOWS_WORKER_OS_ENV`].
fn worker_env() -> Vec<(&'static str, std::ffi::OsString)> {
    #[cfg(windows)]
    let keys = WORKER_OS_ENV.iter().chain(WINDOWS_WORKER_OS_ENV);
    #[cfg(not(windows))]
    let keys = WORKER_OS_ENV.iter();
    keys.filter_map(|k| Some((*k, std::env::var_os(k)?)))
        .collect()
}

#[derive(Clone, Serialize, Deserialize)]
pub struct SessionEntry {
    /// Process ID of the started evanalyzer instance for this session
    pub pid: u32,
    /// Token of this session. The worker expects it in the remote protocol's
    pub session_token: String,
    /// ID of the user which started the session ([`User::user_id`]) - never
    /// the username, which may change.
    pub user_id: String,
    /// The worker listens on `127.0.0.1:<port>`
    pub port: u16,
    /// Date time when the session has been started
    pub start_date: SystemTime,
}

pub struct SessionManagement {
    pub path_to_session_store: PathBuf,
    /// Executable started as worker, with `worker --listen 127.0.0.1:<port>`.
    pub worker_command: PathBuf,
    /// Passed to every worker as `--log-level`, so it logs like the server.
    pub worker_log_level: Option<String>,
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
    /// Session file at `path_to_session_store`, the running executable as
    /// worker, logging with `worker_log_level`.
    pub fn new(path_to_session_store: PathBuf, worker_log_level: String) -> io::Result<Self> {
        let mut sessions = Self::with_store(path_to_session_store, std::env::current_exe()?)?;
        sessions.worker_log_level = Some(worker_log_level);
        Ok(sessions)
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
            worker_log_level: None,
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
            .position(|s| s.user_id == user.user_id)
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

    /// Stops the session's worker and forgets the session.
    pub fn close_session(&self, session_token: &str) -> io::Result<()> {
        let mut state = self.state.lock().unwrap();
        if let Some(index) = state
            .sessions
            .iter()
            .position(|s| s.session_token == session_token)
        {
            let entry = state.remove(index);
            info!("Closed session of user {}", entry.user_id);
            self.persist(&state.sessions)?;
        }
        Ok(())
    }

    fn create_session(&self, state: &mut State, user: &User) -> io::Result<SessionEntry> {
        let port = free_port()?;
        let session_token = generate_token()?;
        let mut command = Command::new(&self.worker_command);
        command
            .arg("worker")
            .arg("--listen")
            .arg(format!("127.0.0.1:{port}"))
            .arg("--token")
            .arg(&session_token)
            .arg("--home")
            .arg(&user.home);
        for dir in &user.allowed_dirs {
            command.arg("--root").arg(dir);
        }
        if let Some(level) = &self.worker_log_level {
            command.arg("--log-level").arg(level);
        }
        command
            .current_dir(&user.home)
            .stdin(Stdio::null())
            .env_clear()
            .envs(worker_env());
        run_as(&mut command, user);

        let mut child = command.spawn()?;
        if let Err(err) = wait_until_listening(&mut child, port) {
            let _ = child.kill();
            let _ = child.wait();
            return Err(err);
        }
        let entry = SessionEntry {
            pid: child.id(),
            session_token,
            user_id: user.user_id.clone(),
            port,
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

/// Where the session file goes unless `--session-store` says otherwise:
/// `/run/evanalyzer/sessions.json` for a system service, otherwise the
/// OS's temp folder.
pub fn default_store_path() -> PathBuf {
    let system = Path::new("/run/evanalyzer");
    if create_private_dir(system).is_ok() {
        return system.join("sessions.json");
    }
    std::env::temp_dir()
        .join("evanalyzer")
        .join("sessions.json")
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

/// When the server runs as root, the worker runs as the logged-in user - so
/// it can only touch that user's files. Otherwise all workers run as the
/// server's user. (Home and allowed folders are arguments, see
/// `create_session`.)
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
    command.uid(account.uid).gid(account.gid);
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

#[cfg(all(test, unix))]
pub(crate) mod tests {
    use super::*;
    use std::os::unix::fs::PermissionsExt;

    /// Stand-in for `evanalyzer worker --listen 127.0.0.1:<port>`: just
    /// listens on the port.
    pub(crate) fn fake_worker(dir: &Path) -> PathBuf {
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
            user_id: format!("id-{name}"),
            username: name.into(),
            home: std::env::temp_dir(),
            allowed_dirs: vec![std::env::temp_dir()],
            unix_account: None,
        }
    }

    /// Like [`fake_worker`], but first writes its arguments and its whole
    /// environment to `<dir>/args.txt` and `<dir>/env.txt`.
    fn recording_worker(dir: &Path) -> PathBuf {
        let path = dir.join("recording-worker");
        let record = dir.display();
        fs::write(
            &path,
            format!(
                "#!/bin/sh\nprintf '%s\\n' \"$@\" > {record}/args.txt\n/usr/bin/env > {record}/env.txt\nexec python3 -m http.server \"${{3##*:}}\" --bind 127.0.0.1 >/dev/null 2>&1\n"
            ),
        )
        .unwrap();
        fs::set_permissions(&path, fs::Permissions::from_mode(0o755)).unwrap();
        path
    }

    #[test]
    fn the_worker_gets_home_folders_and_log_level_as_arguments_not_environment() {
        let dir = tempfile::tempdir().unwrap();
        let mut sessions = SessionManagement::with_store(
            dir.path().join("run/sessions.json"),
            recording_worker(dir.path()),
        )
        .unwrap();
        sessions.worker_log_level = Some("debug".into());
        let home = dir.path().join("home");
        let data = dir.path().join("data");
        fs::create_dir_all(&home).unwrap();
        let alice = User {
            user_id: "alice".into(),
            username: "alice".into(),
            home: home.clone(),
            allowed_dirs: vec![home.clone(), data.clone()],
            unix_account: None,
        };

        let entry = sessions.open_or_create_session(&alice).unwrap();

        let args = fs::read_to_string(dir.path().join("args.txt")).unwrap();
        let args: Vec<&str> = args.lines().collect();
        let after = |flag: &str| -> Vec<&str> {
            args.windows(2)
                .filter(|w| w[0] == flag)
                .map(|w| w[1])
                .collect()
        };
        assert_eq!(after("--home"), [home.to_str().unwrap()]);
        assert_eq!(
            after("--root"),
            [home.to_str().unwrap(), data.to_str().unwrap()]
        );
        assert_eq!(after("--log-level"), ["debug"]);
        let env = fs::read_to_string(dir.path().join("env.txt")).unwrap();
        for variable in ["HOME=", "USER=", "LOGNAME=", "RUST_LOG=", "LANG="] {
            assert!(
                !env.lines().any(|line| line.starts_with(variable)),
                "{variable} in\n{env}"
            );
        }

        sessions.close_session(&entry.session_token).unwrap();
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
        assert_ne!(bob.session_token, first.session_token);

        sessions.close_session(&first.session_token).unwrap();
        sessions.close_session(&bob.session_token).unwrap();
    }

    #[test]
    fn a_renamed_user_gets_their_running_session_back() {
        let dir = tempfile::tempdir().unwrap();
        let sessions = manager(dir.path());
        let first = sessions.open_or_create_session(&user("bob")).unwrap();

        let renamed = User {
            username: "robert".into(),
            ..user("bob")
        };
        let again = sessions.open_or_create_session(&renamed).unwrap();
        assert_eq!(again.pid, first.pid);

        // A new user taking over the old name is someone else.
        let new_bob = User {
            user_id: "id-new-bob".into(),
            ..user("bob")
        };
        let other = sessions.open_or_create_session(&new_bob).unwrap();
        assert_ne!(other.pid, first.pid);

        sessions.close_session(&first.session_token).unwrap();
        sessions.close_session(&other.session_token).unwrap();
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

        // The store records the session...
        let stored: Vec<SessionEntry> = serde_json::from_slice(&fs::read(store).unwrap()).unwrap();
        assert_eq!(stored.len(), 1);
        assert_eq!(stored[0].pid, entry.pid);
        // ...so a new server process finds the running worker again.
        let restarted = manager(dir.path());
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
        assert!(sessions.state.lock().unwrap().sessions.is_empty());
        assert!(!worker_responds(entry.port));
        // Gone from the store too: a restarted server doesn't bring it back.
        assert!(
            manager(dir.path())
                .state
                .lock()
                .unwrap()
                .sessions
                .is_empty()
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

#[cfg(test)]
mod worker_env_tests {
    use super::*;

    #[test]
    fn workers_get_only_the_variables_the_os_needs() {
        #[cfg(windows)]
        let allowed: Vec<&str> = WORKER_OS_ENV
            .iter()
            .chain(WINDOWS_WORKER_OS_ENV)
            .copied()
            .collect();
        #[cfg(not(windows))]
        let allowed: Vec<&str> = WORKER_OS_ENV.to_vec();

        let env = worker_env();

        assert!(env.iter().all(|(key, _)| allowed.contains(key)));
        assert!(
            env.iter()
                .all(|(key, _)| !matches!(*key, "HOME" | "USER" | "RUST_LOG"))
        );
    }

    /// Without it the worker can't open a socket (os error 10106).
    #[cfg(windows)]
    #[test]
    fn windows_workers_get_system_root() {
        assert!(worker_env().iter().any(|(key, _)| *key == "SystemRoot"));
    }
}
