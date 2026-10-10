use std::path::PathBuf;

use crate::user_management::UserManagement;
use crate::user_management::password::verify_password;

/// The ID of the single account. Fixed, so changing its username changes
/// nothing but the login.
pub const SINGLE_USER_ID: &str = "single";

/// One account, configured in `[users.single]` (see
/// `config::SingleUserConfig`).
pub struct SingleUser {
    pub username: String,
    /// The stored password, see [`crate::user_management::password`].
    pub stored_password: String,
    pub user_home: PathBuf,
    /// Folders the user's worker may access.
    pub allowed_dirs: Vec<PathBuf>,
}

impl SingleUser {
    /// `username` with `stored_password`, at home in `home`, allowed to
    /// access `allowed_dirs` (`{home}` = the home folder).
    pub fn new(
        username: String,
        stored_password: String,
        home: PathBuf,
        allowed_dirs: Vec<String>,
    ) -> Self {
        Self {
            username,
            stored_password,
            allowed_dirs: super::expand_dirs(&allowed_dirs, &home),
            user_home: home,
        }
    }

    /// `admin` with password `1234`, at home in a fresh temporary folder -
    /// for tests.
    #[cfg(test)]
    pub fn for_tests() -> Self {
        let home = tempfile::tempdir().unwrap().keep();
        Self::new(
            "admin".into(),
            "plain:1234".into(),
            home,
            vec!["{home}".into()],
        )
    }
}

impl UserManagement for SingleUser {
    fn login(&self, username: String, password: String) -> super::AuthenticationStatus {
        if username == self.username && verify_password(&password, &self.stored_password) {
            return super::AuthenticationStatus::Authenticated(super::User {
                user_id: SINGLE_USER_ID.into(),
                username,
                home: self.user_home.clone(),
                allowed_dirs: self.allowed_dirs.clone(),
                unix_account: None,
            });
        }

        return super::AuthenticationStatus::PasswordWrong;
    }
}

/// The single user's home when none is configured: a folder of its own,
/// created for it, which is all its worker may access by default - nothing
/// else of the machine is reachable. `/var/lib/evanalyzer/single-user` for a
/// system service (root on Linux), otherwise below the server account's
/// data folder: `~/.local/share/evanalyzer/single-user` on Linux,
/// `%LOCALAPPDATA%\evanalyzer\single-user` on Windows,
/// `~/Library/Application Support/evanalyzer/single-user` on macOS.
pub(crate) fn default_home() -> PathBuf {
    #[cfg(target_os = "linux")]
    {
        use std::os::unix::fs::MetadataExt;
        if std::fs::metadata("/proc/self").is_ok_and(|me| me.uid() == 0) {
            return PathBuf::from("/var/lib/evanalyzer/single-user");
        }
        server_account_home()
            .join(".local")
            .join("share")
            .join("evanalyzer")
            .join("single-user")
    }
    #[cfg(not(target_os = "linux"))]
    {
        dirs::data_local_dir()
            .unwrap_or_else(server_account_home)
            .join("evanalyzer")
            .join("single-user")
    }
}

/// The home folder of the account this server runs as. Not read from `HOME`
/// (configuration comes from arguments, never the environment): on Linux
/// it is the account's `/etc/passwd` entry, elsewhere the OS's profile
/// folder.
pub(crate) fn server_account_home() -> PathBuf {
    #[cfg(target_os = "linux")]
    {
        use std::os::unix::fs::MetadataExt;
        let home = std::fs::metadata("/proc/self").ok().and_then(|me| {
            super::linux_users::passwd_entry_by_uid(std::path::Path::new("/etc/passwd"), me.uid())
                .ok()
                .flatten()
                .map(|entry| entry.home)
        });
        if let Some(home) = home {
            return home;
        }
    }
    #[cfg(not(target_os = "linux"))]
    if let Some(home) = dirs::home_dir() {
        return home;
    }
    log::warn!("Cannot determine the server account's home folder - using /");
    PathBuf::from("/")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::user_management::AuthenticationStatus;

    #[test]
    fn only_the_configured_name_and_password_log_in() {
        let users = SingleUser::for_tests();
        match users.login("admin".into(), "1234".into()) {
            AuthenticationStatus::Authenticated(user) => {
                assert_eq!(user.username, "admin");
                assert_eq!(user.user_id, SINGLE_USER_ID);
                assert!(user.unix_account.is_none());
            }
            _ => panic!("expected a login"),
        }
        for (name, password) in [("admin", "12345"), ("root", "1234"), ("", "")] {
            assert!(matches!(
                users.login(name.into(), password.into()),
                AuthenticationStatus::PasswordWrong
            ));
        }
    }

    #[test]
    fn renaming_the_user_keeps_their_id() {
        let users = SingleUser::new("joachim".into(), "plain:pw".into(), "/srv/x".into(), vec![]);
        let AuthenticationStatus::Authenticated(user) = users.login("joachim".into(), "pw".into())
        else {
            panic!("expected a login");
        };
        assert_eq!(user.username, "joachim");
        assert_eq!(user.user_id, SINGLE_USER_ID);
    }

    #[test]
    fn the_default_home_is_a_folder_of_its_own_and_the_user_may_only_access_it() {
        let home = default_home();
        assert!(home.is_absolute(), "{}", home.display());
        assert!(home.ends_with("single-user"), "{}", home.display());
        assert_ne!(home, server_account_home(), "not the server account's home");

        let users = SingleUser::new(
            "admin".into(),
            "plain:pw".into(),
            home.clone(),
            vec!["{home}".into()],
        );
        let AuthenticationStatus::Authenticated(user) = users.login("admin".into(), "pw".into())
        else {
            panic!("expected a login");
        };
        assert_eq!(user.home, home);
        assert_eq!(user.allowed_dirs, [home]);
    }
}
