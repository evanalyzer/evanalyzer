pub mod linux_users;
pub mod single_user;

pub struct User {
    pub user_id: String,
    pub username: String,
    /// The user's home folder: the worker's working directory and where it
    /// keeps the user's EVAnalyzer folder (templates). Passed to the worker
    /// as `--home`.
    pub home: std::path::PathBuf,
    /// The only folders the user's worker may read and write (each passed
    /// as `--root`); its own EVAnalyzer folder below `home` stays reachable
    /// as well.
    pub allowed_dirs: Vec<std::path::PathBuf>,
    /// Local system account to run the user's worker as, if there is one.
    pub unix_account: Option<UnixAccount>,
}

pub struct UnixAccount {
    pub uid: u32,
    pub gid: u32,
}

pub enum AuthenticationStatus {
    Authenticated(User),
    PasswordWrong,
}

pub trait UserManagement: Send + Sync {
    fn login(&self, username: String, password: String) -> AuthenticationStatus;
}
