use std::path::PathBuf;

use crate::user_management::UserManagement;

pub struct SingleUser {
    pub username: String,
    pub password: String,
    pub userid: String,
    pub user_home: PathBuf,
    /// Folders the user's worker may access; just `user_home` by default.
    pub allowed_dirs: Vec<PathBuf>,
}

impl SingleUser {
    /// The built-in account, at home in the server account's own home folder
    /// (until a configuration file sets it).
    pub fn default() -> Self {
        let home = server_account_home();
        Self {
            username: "admin".into(),
            password: "1234".into(),
            userid: "user-id".into(),
            allowed_dirs: vec![home.clone()],
            user_home: home,
        }
    }
}

impl UserManagement for SingleUser {
    fn login(&self, username: String, password: String) -> super::AuthenticationStatus {
        if username == self.username && password == self.password {
            return super::AuthenticationStatus::Authenticated(super::User {
                user_id: self.userid.clone(),
                username,
                home: self.user_home.clone(),
                allowed_dirs: self.allowed_dirs.clone(),
                unix_account: None,
            });
        }

        return super::AuthenticationStatus::PasswordWrong;
    }
}

/// The home folder of the account this server runs as. Not read from `HOME`
/// (configuration comes from arguments, never the environment): on Linux
/// it is the account's `/etc/passwd` entry, elsewhere the OS's profile
/// folder.
fn server_account_home() -> PathBuf {
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
        let users = SingleUser::default();
        match users.login("admin".into(), "1234".into()) {
            AuthenticationStatus::Authenticated(user) => {
                assert_eq!(user.username, "admin");
                assert_eq!(user.user_id, "user-id");
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
    fn the_default_user_lives_in_the_server_account_s_home_and_may_only_access_it() {
        let users = SingleUser::default();
        let AuthenticationStatus::Authenticated(user) = users.login("admin".into(), "1234".into())
        else {
            panic!("expected a login");
        };

        assert!(user.home.is_absolute(), "{}", user.home.display());
        assert_eq!(user.allowed_dirs, [user.home.clone()]);
    }
}
