pub mod linux_users;
pub mod single_user;

pub struct User {
    pub user_id: String,
    pub username: String,
    /// Local system account to run the user's worker as, if there is one.
    pub unix_account: Option<UnixAccount>,
}

pub struct UnixAccount {
    pub uid: u32,
    pub gid: u32,
    pub home: std::path::PathBuf,
}

pub enum AuthenticationStatus {
    Authenticated(User),
    PasswordWrong,
}

pub trait UserManagement: Send + Sync {
    fn login(&self, username: String, password: String) -> AuthenticationStatus;
}
