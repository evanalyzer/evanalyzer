//! # config
//!
//! `evanalyzer server --config <file>`: the server's configuration as a TOML
//! file. Every setting has a default (see the `Default` impls), so a config
//! file only needs what differs; command-line arguments override the file
//! (see [`ServerConfig::with_overrides`]). Unknown keys are an error, so a
//! misspelled setting can't silently fall back to its default.
//!
//! These doc comments are the reference for every key; `docs/server.toml`
//! and `docs/server-users.toml` are complete, commented examples (parsed by
//! this module's tests, so they can't drift from the code), and
//! `docs/server.md` explains how the pieces fit together.
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-10-10
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.

use crate::user_management::{
    UserManagement,
    file_users::FileUsers,
    linux_users::LinuxUsers,
    password::{
        SUPPORTED_FORMATS, hash_password, is_plain_text, is_supported_format, random_password,
    },
    single_user::{SingleUser, server_account_home},
};
use serde::Deserialize;
use std::{
    collections::BTreeMap,
    io,
    path::{Path, PathBuf},
    sync::Arc,
};

/// Name of the EVAnalyzer folder in a home folder - the same as the app's
/// `USER_FOLDER_NAME` (this crate doesn't depend on the app): hidden on
/// Linux, visible elsewhere, lower case everywhere.
const USER_FOLDER_NAME: &str = if cfg!(target_os = "linux") {
    ".evanalyzer"
} else {
    "evanalyzer"
};

/// Where the server keeps its state unless `data_dir` says otherwise:
/// `/var/lib/evanalyzer` for a system service (running as root on Linux),
/// otherwise `server` in the server account's EVAnalyzer folder
/// (`~/.evanalyzer/server` on Linux, `~/evanalyzer/server` elsewhere).
pub fn default_data_dir() -> PathBuf {
    #[cfg(target_os = "linux")]
    {
        use std::os::unix::fs::MetadataExt;
        if std::fs::metadata("/proc/self").is_ok_and(|me| me.uid() == 0) {
            return PathBuf::from("/var/lib/evanalyzer");
        }
    }
    server_account_home().join(USER_FOLDER_NAME).join("server")
}

/// The whole server configuration.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct ServerConfig {
    /// Address and port to accept clients on. Default `127.0.0.1:7400`
    /// (this machine only); `0.0.0.0:7400` accepts every network.
    pub listen: String,
    /// Folder the server keeps everything it creates in: `tls/` (the
    /// self-signed certificate), `sessions.json` (the running workers) and
    /// `single-user/` (the single user's default home). Absolute path.
    /// Default: `/var/lib/evanalyzer` for a system service (root on Linux),
    /// otherwise `server` in the server account's EVAnalyzer folder
    /// (`~/.evanalyzer/server` on Linux, `~/evanalyzer/server` elsewhere).
    pub data_dir: Option<PathBuf>,
    /// What the server and its workers log, in env_logger filter syntax: a
    /// level (`error`, `warn`, `info`, `debug`, `trace`, `off`) or per
    /// module, e.g. `info,evanalyzer_core=debug`. Default `info`.
    pub log_level: String,
    /// Who may log in, and where their workers may read and write.
    pub users: UsersConfig,
    /// Encryption of client connections.
    pub tls: TlsConfig,
    /// The per-user worker processes.
    pub workers: WorkersConfig,
    /// How many users and connections the server takes at once.
    pub limits: LimitsConfig,
}

/// `[workers]`: the `evanalyzer worker` process each logged-in user gets.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct WorkersConfig {
    /// A worker stops once no client has been connected to it and no
    /// analysis has run for this many minutes; the user's next login starts
    /// a new one. A running analysis keeps it alive however long it takes.
    /// Default 120.
    pub idle_timeout_minutes: u64,
    /// Where the workers run AI models: `auto` (every GPU, else the CPU),
    /// `cpu`, or GPU numbers like `0,2`. Default `auto`.
    pub ai_devices: Option<String>,
    /// Parallel AI inferences per GPU, at least 1. Default 1.
    pub gpu_slots: Option<usize>,
}

impl Default for WorkersConfig {
    fn default() -> Self {
        Self {
            idle_timeout_minutes: 120,
            ai_devices: None,
            gpu_slots: None,
        }
    }
}

/// `[limits]`: protects the machine from more work than it can do. Not set:
/// no limit.
#[derive(Debug, Clone, Default, PartialEq, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct LimitsConfig {
    /// At most this many workers (= users working at the same time). A
    /// login that would start one more is refused with "try again later";
    /// users whose worker already runs can always log in.
    pub max_workers: Option<usize>,
    /// At most this many client connections at once, all users together
    /// (one GUI or CLI is one connection). Further ones are refused.
    pub max_connections: Option<usize>,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            listen: "127.0.0.1:7400".into(),
            data_dir: None,
            log_level: "info".into(),
            users: UsersConfig::default(),
            tls: TlsConfig::default(),
            workers: WorkersConfig::default(),
            limits: LimitsConfig::default(),
        }
    }
}

/// `[tls]`: encryption of client connections. On by default, with a
/// self-signed certificate the server creates itself - see
/// [`crate::tls`].
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct TlsConfig {
    /// Encrypt client connections: clients then connect with `wss://`
    /// instead of `ws://`. Default `true`. Turn it off only when something
    /// else encrypts (a reverse proxy, VPN, SSH tunnel) or for tests on this
    /// machine - otherwise passwords and data cross the network readable.
    pub enabled: bool,
    /// Certificate chain (PEM), e.g. from Let's Encrypt or the
    /// organisation's CA. Needs `key`. Default: a self-signed certificate the
    /// server creates in `<data_dir>/tls` and keeps there.
    pub cert: Option<PathBuf>,
    /// Private key of `cert` (PEM). Needs `cert`.
    pub key: Option<PathBuf>,
}

impl Default for TlsConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            cert: None,
            key: None,
        }
    }
}

/// `[users]`: where accounts come from.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct UsersConfig {
    /// Which accounts may log in: `single` (one account configured in
    /// `[users.single]`), `linux` (this machine's system users, see
    /// `[users.linux]`) or `file` (the accounts of a users file, see
    /// `[users.file]`). Default `single`.
    pub source: UserSource,
    /// Folders every user's worker may read and write, unless their account
    /// sets its own. `{home}` stands for the user's home folder. Default
    /// `["{home}"]`.
    pub default_allowed_dirs: Vec<String>,
    /// `[users.single]`, for `source = "single"`.
    pub single: SingleUserConfig,
    /// `[users.linux]`, for `source = "linux"`.
    pub linux: LinuxUsersConfig,
    /// `[users.file]`, for `source = "file"`.
    pub file: UserFileConfig,
}

impl Default for UsersConfig {
    fn default() -> Self {
        Self {
            source: UserSource::default(),
            default_allowed_dirs: vec!["{home}".into()],
            single: SingleUserConfig::default(),
            linux: LinuxUsersConfig::default(),
            file: UserFileConfig::default(),
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum UserSource {
    #[default]
    Single,
    Linux,
    File,
}

/// `[users.single]`: one account, configured right here.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct SingleUserConfig {
    /// Login name. Default `admin`.
    pub username: String,
    /// The password: an Argon2 (`$argon2id$...`), bcrypt (`$2b$...`),
    /// yescrypt (`$y$...`) or sha-crypt (`$6$...`, `$5$...`) hash, or
    /// `plain:<password>` for testing (the server warns about it). Create
    /// one with `evanalyzer hash-password`. Default: none - the server
    /// makes up a random password at every start and prints it to the
    /// console.
    pub password: Option<String>,
    /// Home folder of the account: the worker's working directory and where
    /// it keeps the user's EVAnalyzer folder (settings, templates). Default:
    /// a folder of its own, created for it: `<data_dir>/single-user` - by
    /// default all the worker may access.
    pub home: Option<PathBuf>,
    /// Folders the worker may read and write (`{home}` = `home`). Default:
    /// `users.default_allowed_dirs`.
    pub allowed_dirs: Option<Vec<String>>,
}

impl Default for SingleUserConfig {
    fn default() -> Self {
        Self {
            username: "admin".into(),
            password: None,
            home: None,
            allowed_dirs: None,
        }
    }
}

/// `[users.linux]`: this machine's system users. Passwords are checked
/// against the shadow file (needs root or the `shadow` group); home folder,
/// uid and gid come from the passwd file. When the server runs as root, each
/// worker runs as its user's system account.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct LinuxUsersConfig {
    /// Default `/etc/shadow`.
    pub shadow_file: PathBuf,
    /// Default `/etc/passwd`.
    pub passwd_file: PathBuf,
    /// Per-user exceptions, `[users.linux.overrides.<username>]`.
    pub overrides: BTreeMap<String, UserOverride>,
}

impl Default for LinuxUsersConfig {
    fn default() -> Self {
        Self {
            shadow_file: "/etc/shadow".into(),
            passwd_file: "/etc/passwd".into(),
            overrides: BTreeMap::new(),
        }
    }
}

/// `[users.linux.overrides.<username>]`: what differs for one system user.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct UserOverride {
    /// Folders this user's worker may read and write instead of
    /// `users.default_allowed_dirs` (`{home}` = their home folder).
    pub allowed_dirs: Vec<String>,
}

/// `[users.file]`: accounts listed in a separate users file - see
/// [`FileUsers`] for its format.
#[derive(Debug, Clone, Default, PartialEq, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct UserFileConfig {
    /// The users file. Required for `source = "file"`.
    pub path: Option<PathBuf>,
}

impl ServerConfig {
    /// Reads and checks the TOML file at `path` - unknown keys, wrong types
    /// and impossible values are an error naming the problem.
    pub fn load(path: &Path) -> io::Result<Self> {
        let text = std::fs::read_to_string(path).map_err(|e| {
            io::Error::new(e.kind(), format!("Cannot read {}: {e}", path.display()))
        })?;
        let config: Self = toml::from_str(&text).map_err(|e| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("Invalid server config {}: {e}", path.display()),
            )
        })?;
        config.check()?;
        Ok(config)
    }

    /// The configuration with the settings given on the command line
    /// replacing the file's (or the default's).
    pub fn with_overrides(
        mut self,
        listen: Option<String>,
        data_dir: Option<PathBuf>,
        log_level: Option<String>,
    ) -> Self {
        if let Some(listen) = listen {
            self.listen = listen;
        }
        if let Some(data_dir) = data_dir {
            self.data_dir = Some(data_dir);
        }
        if let Some(log_level) = log_level {
            self.log_level = log_level;
        }
        self
    }

    /// The folder the server keeps its state in: `data_dir`, else
    /// [`default_data_dir`].
    pub fn data_dir(&self) -> PathBuf {
        self.data_dir.clone().unwrap_or_else(default_data_dir)
    }

    /// The file the running workers are recorded in, so a restarted server
    /// finds them again.
    pub fn session_store(&self) -> PathBuf {
        self.data_dir().join("sessions.json")
    }

    /// Where the self-signed certificate is created and kept when no
    /// `tls.cert` is configured. Must survive restarts and reboots, or every
    /// client has to pin a new fingerprint.
    pub fn self_signed_dir(&self) -> PathBuf {
        self.data_dir().join("tls")
    }

    /// The workers' AI device settings given on the command line replacing
    /// the file's.
    pub fn with_worker_ai_overrides(
        mut self,
        ai_devices: Option<String>,
        gpu_slots: Option<usize>,
    ) -> Self {
        if let Some(ai_devices) = ai_devices {
            self.workers.ai_devices = Some(ai_devices);
        }
        if let Some(gpu_slots) = gpu_slots {
            self.workers.gpu_slots = Some(gpu_slots);
        }
        self
    }

    /// What can be checked without touching the system (folders, users
    /// file): the rest is checked when the user management is built.
    fn check(&self) -> io::Result<()> {
        let invalid = |message: String| Err(io::Error::new(io::ErrorKind::InvalidInput, message));
        if self.listen.parse::<std::net::SocketAddr>().is_err() {
            return invalid(format!(
                "listen = \"{}\" is not an address with port, e.g. 127.0.0.1:7400",
                self.listen
            ));
        }
        let single = &self.users.single;
        if single
            .password
            .as_ref()
            .is_some_and(|password| !is_supported_format(password))
        {
            return invalid(format!("users.single.password must be {SUPPORTED_FORMATS}"));
        }
        if let Some(home) = &single.home {
            if !home.is_absolute() {
                return invalid(format!(
                    "users.single.home must be an absolute path, got {}",
                    home.display()
                ));
            }
        }
        if let Some(data_dir) = &self.data_dir
            && !data_dir.is_absolute()
        {
            return invalid(format!(
                "data_dir must be an absolute path, got {}",
                data_dir.display()
            ));
        }
        if self.users.source == UserSource::File && self.users.file.path.is_none() {
            return invalid("users.source = \"file\" needs users.file.path".into());
        }
        if self.workers.idle_timeout_minutes == 0 {
            return invalid(
                "workers.idle_timeout_minutes must be at least 1 - idle workers would \
                 otherwise pile up"
                    .into(),
            );
        }
        if self.limits.max_workers == Some(0) || self.limits.max_connections == Some(0) {
            return invalid(
                "limits.max_workers and limits.max_connections must be at least 1 \
                 (leave them out for no limit)"
                    .into(),
            );
        }
        if self.tls.cert.is_some() != self.tls.key.is_some() {
            return invalid("tls.cert and tls.key go together - set both or neither".into());
        }
        Ok(())
    }

    /// The configured accounts, ready for logins. Reads the users file for
    /// `source = "file"`, and warns about settings that work but are risky.
    pub fn user_management(&self) -> io::Result<Arc<dyn UserManagement>> {
        let users = &self.users;
        Ok(match users.source {
            UserSource::Single => {
                let single = &users.single;
                let password = match &single.password {
                    Some(password) => {
                        if is_plain_text(password) {
                            log::warn!(
                                "Single user '{}' has a plain-text password - store a hash instead",
                                single.username
                            );
                        }
                        password.clone()
                    }
                    None => {
                        let password = random_password()?;
                        announce_generated_password(&single.username, &password);
                        hash_password(&password)?
                    }
                };
                let home = match &single.home {
                    Some(home) => home.clone(),
                    None => {
                        let home = self.data_dir().join("single-user");
                        crate::session_management::create_private_dir(&home).map_err(|err| {
                            io::Error::new(
                                err.kind(),
                                format!(
                                    "Cannot create the single user's home {}: {err} - set \
                                     users.single.home",
                                    home.display()
                                ),
                            )
                        })?;
                        home
                    }
                };
                Arc::new(SingleUser::new(
                    single.username.clone(),
                    password,
                    home,
                    single
                        .allowed_dirs
                        .clone()
                        .unwrap_or_else(|| users.default_allowed_dirs.clone()),
                ))
            }
            UserSource::Linux => Arc::new(LinuxUsers {
                shadow_file: users.linux.shadow_file.clone(),
                passwd_file: users.linux.passwd_file.clone(),
                default_allowed_dirs: users.default_allowed_dirs.clone(),
                overrides: users
                    .linux
                    .overrides
                    .iter()
                    .map(|(name, user)| (name.clone(), user.allowed_dirs.clone()))
                    .collect(),
            }),
            UserSource::File => {
                let path = users.file.path.as_deref().ok_or_else(|| {
                    io::Error::new(
                        io::ErrorKind::InvalidInput,
                        "users.source = \"file\" needs users.file.path",
                    )
                })?;
                warn_if_others_can_read(path);
                Arc::new(FileUsers::load(path, &users.default_allowed_dirs)?)
            }
        })
    }
}

/// Shows the password made up for the single user on the console - the
/// only place it is ever shown. The log only says that one was made up.
fn announce_generated_password(username: &str, password: &str) {
    log::info!(
        "No users.single.password configured: made up a random password for '{username}', \
         valid until the server stops (printed to the console)"
    );
    println!(
        "\n  Log in as '{username}' with password: {password}\n  \
         (made up for this run - set users.single.password in a --config file to keep one; \
         create it with `evanalyzer hash-password`)\n"
    );
}

/// Password hashes shouldn't be readable by other accounts - they can be
/// attacked offline.
pub(crate) fn warn_if_others_can_read(path: &Path) {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        if let Ok(meta) = std::fs::metadata(path) {
            if meta.permissions().mode() & 0o044 != 0 {
                log::warn!(
                    "{} holds password hashes but other accounts can read it - \
                     `chmod 600` it",
                    path.display()
                );
            }
        }
    }
    #[cfg(not(unix))]
    let _ = path;
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(text: &str) -> Result<ServerConfig, String> {
        let config: ServerConfig = toml::from_str(text).map_err(|e| e.to_string())?;
        config.check().map_err(|e| e.to_string())?;
        Ok(config)
    }

    #[test]
    fn an_empty_file_is_the_default_configuration() {
        assert_eq!(parse("").unwrap(), ServerConfig::default());
    }

    #[test]
    fn the_documented_example_files_are_valid() {
        // docs/server.toml documents every key - if it stops parsing, the
        // docs and the code disagree.
        let config = parse(include_str!("../../../docs/server.toml")).unwrap();
        assert_eq!(config.users.source, UserSource::Single);

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("users.toml");
        std::fs::write(&path, include_str!("../../../docs/server-users.toml")).unwrap();
        let users = FileUsers::load(&path, &["{home}".into()]).unwrap();
        for name in ["alice", "bob", "carol"] {
            assert!(
                matches!(
                    users.login(name.into(), "correct horse".into()),
                    crate::user_management::AuthenticationStatus::Authenticated(_)
                ),
                "the documented password of {name}"
            );
        }
    }

    #[test]
    fn without_a_configured_password_the_single_user_gets_a_random_one() {
        use crate::user_management::AuthenticationStatus;
        let home = tempfile::tempdir().unwrap();
        let config = parse(&format!(
            "[users.single]\nhome = '{}'",
            home.path().display()
        ))
        .unwrap();
        assert_eq!(config.users.single.password, None);

        let users = config.user_management().unwrap();
        assert!(
            matches!(
                users.login("admin".into(), "1234".into()),
                AuthenticationStatus::PasswordWrong
            ),
            "no built-in password any more"
        );
    }

    #[test]
    fn a_configured_password_is_used() {
        use crate::user_management::AuthenticationStatus;
        let home = tempfile::tempdir().unwrap();
        let config = parse(&format!(
            "[users.single]\npassword = 'plain:secret'\nhome = '{}'",
            home.path().display()
        ))
        .unwrap();
        let users = config.user_management().unwrap();
        let AuthenticationStatus::Authenticated(user) =
            users.login("admin".into(), "secret".into())
        else {
            panic!("expected a login");
        };
        assert_eq!(user.home, home.path());
        assert_eq!(user.allowed_dirs, [home.path().to_path_buf()]);
    }

    #[test]
    fn a_misspelled_key_is_an_error_not_a_silent_default() {
        let error = parse("listn = \"0.0.0.0:7400\"").unwrap_err();
        assert!(error.contains("listn"), "{error}");
        let error = parse("[users.single]\npasswd_hash = \"$6$x\"").unwrap_err();
        assert!(error.contains("passwd_hash"), "{error}");
    }

    #[test]
    fn impossible_values_are_rejected_with_the_key_named() {
        for (text, key) in [
            ("listen = \"localhost\"", "listen"),
            ("[users.single]\npassword = \"1234\"", "password"),
            (
                "[users.single]\nhome = \"relative/home\"",
                "users.single.home",
            ),
            ("[users]\nsource = \"file\"", "users.file.path"),
            ("[users]\nsource = \"ldap\"", "ldap"),
            ("[tls]\ncert = \"/etc/evanalyzer/cert.pem\"", "tls.key"),
            (
                "[workers]\nidle_timeout_minutes = 0",
                "idle_timeout_minutes",
            ),
            ("[limits]\nmax_workers = 0", "max_workers"),
        ] {
            let error = parse(text).unwrap_err();
            assert!(error.contains(key), "{text}: {error}");
        }
    }

    #[test]
    fn workers_stop_after_two_idle_hours_and_nothing_is_limited_by_default() {
        let config = parse("").unwrap();
        assert_eq!(config.workers.idle_timeout_minutes, 120);
        assert_eq!(config.limits, LimitsConfig::default());
        let limited =
            parse("[limits]\nmax_workers = 10\nmax_connections = 40\n[workers]\nidle_timeout_minutes = 30")
                .unwrap();
        assert_eq!(limited.limits.max_workers, Some(10));
        assert_eq!(limited.limits.max_connections, Some(40));
        assert_eq!(limited.workers.idle_timeout_minutes, 30);
    }

    #[test]
    fn tls_is_on_by_default_and_can_be_turned_off() {
        assert!(parse("").unwrap().tls.enabled);
        assert!(!parse("[tls]\nenabled = false").unwrap().tls.enabled);
        let own = parse("[tls]\ncert = \"/c.pem\"\nkey = \"/k.pem\"").unwrap();
        assert_eq!(own.tls.cert, Some(PathBuf::from("/c.pem")));
    }

    #[test]
    fn worker_ai_settings_come_from_the_file_or_the_command_line() {
        let file = parse("[workers]\nai_devices = \"0,1\"\ngpu_slots = 2").unwrap();
        assert_eq!(file.workers.ai_devices.as_deref(), Some("0,1"));

        let config = file.with_worker_ai_overrides(None, Some(3));
        assert_eq!(
            config.workers.ai_devices.as_deref(),
            Some("0,1"),
            "not overridden"
        );
        assert_eq!(config.workers.gpu_slots, Some(3));
        assert_eq!(ServerConfig::default().workers.ai_devices, None);
    }

    #[test]
    fn everything_the_server_creates_is_kept_in_its_data_dir() {
        let config = parse("data_dir = \"/srv/eva\"").unwrap();

        assert_eq!(config.data_dir(), Path::new("/srv/eva"));
        assert_eq!(config.session_store(), Path::new("/srv/eva/sessions.json"));
        assert_eq!(config.self_signed_dir(), Path::new("/srv/eva/tls"));
    }

    #[test]
    fn the_default_data_dir_is_the_system_folder_or_in_the_evanalyzer_folder() {
        let dir = default_data_dir();

        assert_eq!(ServerConfig::default().data_dir(), dir);
        assert!(dir.is_absolute(), "{}", dir.display());
        let in_user_folder = dir.ends_with(Path::new(USER_FOLDER_NAME).join("server"));
        assert!(
            in_user_folder || dir == Path::new("/var/lib/evanalyzer"),
            "{}",
            dir.display()
        );
        assert_ne!(dir, server_account_home(), "not the account's home itself");
    }

    #[test]
    fn a_relative_data_dir_is_rejected() {
        let error = parse("data_dir = \"eva\"").unwrap_err();
        assert!(error.contains("data_dir must be an absolute path"), "{error}");
    }

    #[test]
    fn the_single_user_s_default_home_is_a_folder_of_its_own_and_all_they_may_access() {
        use crate::user_management::AuthenticationStatus;
        let dir = tempfile::tempdir().unwrap();
        let mut config = ServerConfig::default();
        config.data_dir = Some(dir.path().to_path_buf());
        config.users.single.password = Some("plain:pw".into());

        let users = config.user_management().unwrap();

        let AuthenticationStatus::Authenticated(user) = users.login("admin".into(), "pw".into())
        else {
            panic!("expected a login");
        };
        let home = dir.path().join("single-user");
        assert!(home.is_dir(), "created on start");
        assert_eq!(user.home, home);
        assert_eq!(user.allowed_dirs, [home]);
    }

    #[test]
    fn command_line_arguments_override_the_file() {
        let file = parse("listen = \"0.0.0.0:7400\"\nlog_level = \"info\"").unwrap();

        let config = file
            .clone()
            .with_overrides(Some("127.0.0.1:9000".into()), None, None);

        assert_eq!(config.listen, "127.0.0.1:9000");
        assert_eq!(config.log_level, "info", "not given on the command line");
        assert_eq!(config.data_dir, None);
    }

    #[test]
    fn each_user_source_can_be_configured() {
        let linux = parse(
            "[users]\nsource = \"linux\"\n\
             [users.linux.overrides.alice]\nallowed_dirs = [\"{home}\", \"/data\"]",
        )
        .unwrap();
        assert_eq!(linux.users.source, UserSource::Linux);
        assert_eq!(
            linux.users.linux.overrides["alice"].allowed_dirs,
            ["{home}", "/data"]
        );

        let file = parse(
            "[users]\nsource = \"file\"\n[users.file]\npath = \"/etc/evanalyzer/users.toml\"",
        )
        .unwrap();
        assert_eq!(file.users.source, UserSource::File);
    }

    #[test]
    fn a_config_file_is_loaded_and_checked() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("server.toml");
        std::fs::write(&path, "listen = \"127.0.0.1:7500\"\n").unwrap();

        assert_eq!(ServerConfig::load(&path).unwrap().listen, "127.0.0.1:7500");

        std::fs::write(&path, "[limits]\nmax_connections = 0\n").unwrap();
        let error = ServerConfig::load(&path).unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::InvalidInput);
        assert!(error.to_string().contains("max_connections"), "{error}");
    }

    #[test]
    fn an_unreadable_or_malformed_config_file_names_the_file() {
        let dir = tempfile::tempdir().unwrap();
        let missing = dir.path().join("missing.toml");
        let error = ServerConfig::load(&missing).unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::NotFound);
        assert!(error.to_string().contains("missing.toml"), "{error}");

        let broken = dir.path().join("broken.toml");
        std::fs::write(&broken, "listen = ").unwrap();
        let error = ServerConfig::load(&broken).unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::InvalidData);
        assert!(
            error.to_string().starts_with("Invalid server config"),
            "{error}"
        );
    }

    #[test]
    fn every_command_line_override_replaces_the_file_s_value() {
        let config = ServerConfig::default()
            .with_overrides(
                None,
                Some("/srv/eva".into()),
                Some("debug".into()),
            )
            .with_worker_ai_overrides(Some("cpu".into()), None);

        assert_eq!(config.listen, "127.0.0.1:7400", "not given");
        assert_eq!(config.data_dir, Some(PathBuf::from("/srv/eva")));
        assert_eq!(config.log_level, "debug");
        assert_eq!(config.workers.ai_devices.as_deref(), Some("cpu"));
        assert_eq!(config.workers.gpu_slots, None);
    }

    #[test]
    fn linux_users_are_looked_up_in_the_configured_files() {
        use crate::user_management::AuthenticationStatus;
        let dir = tempfile::tempdir().unwrap();
        let (shadow, passwd) = (dir.path().join("shadow"), dir.path().join("passwd"));
        std::fs::write(&shadow, "").unwrap();
        std::fs::write(&passwd, "").unwrap();
        let mut config = ServerConfig::default();
        config.users.source = UserSource::Linux;
        config.users.linux.shadow_file = shadow;
        config.users.linux.passwd_file = passwd;

        let users = config.user_management().unwrap();

        assert!(matches!(
            users.login("alice".into(), "secret".into()),
            AuthenticationStatus::PasswordWrong
        ));
    }

    #[test]
    fn file_users_come_from_the_users_file() {
        use crate::user_management::AuthenticationStatus;
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("users.toml");
        std::fs::write(&path, include_str!("../../../docs/server-users.toml")).unwrap();
        let mut config = ServerConfig::default();
        config.users.source = UserSource::File;
        config.users.file.path = Some(path);

        let users = config.user_management().unwrap();

        assert!(matches!(
            users.login("alice".into(), "correct horse".into()),
            AuthenticationStatus::Authenticated(_)
        ));
    }

    #[test]
    fn file_users_without_a_users_file_are_an_error() {
        let mut config = ServerConfig::default();
        config.users.source = UserSource::File;

        let Err(error) = config.user_management() else {
            panic!("expected an error");
        };

        assert_eq!(error.kind(), io::ErrorKind::InvalidInput);
        assert!(error.to_string().contains("users.file.path"), "{error}");
    }
}
