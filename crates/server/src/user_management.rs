pub mod file_users;
pub mod linux_users;
pub(crate) mod password;
pub mod single_user;

pub struct User {
    /// Identifies the user everywhere inside the server (sessions, workers).
    /// Stable: it stays the same when the username changes - the uid for
    /// system users, the `id` of a users file entry.
    pub user_id: String,
    /// The name the user logged in with. Only for log messages - never
    /// match on it, it may change.
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

/// `templates` with every `{home}` replaced by `home` - the configured
/// allowed folders of one user.
pub(crate) fn expand_dirs(templates: &[String], home: &std::path::Path) -> Vec<std::path::PathBuf> {
    templates
        .iter()
        .map(|template| template.replace("{home}", &home.to_string_lossy()).into())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::{Path, PathBuf};

    #[test]
    fn home_placeholders_are_replaced_by_the_user_s_home() {
        let dirs = expand_dirs(
            &[
                "{home}".into(),
                "{home}/projects".into(),
                "/data/shared".into(),
            ],
            Path::new("/home/alice"),
        );
        assert_eq!(
            dirs,
            [
                PathBuf::from("/home/alice"),
                PathBuf::from("/home/alice/projects"),
                PathBuf::from("/data/shared"),
            ]
        );
    }
}
