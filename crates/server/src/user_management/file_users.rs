//! Accounts listed in a users file (`users.source = "file"`), independent of
//! the machine's system users. A TOML file with one `[[user]]` table per
//! account - `docs/server-users.toml` is a complete example:
//!
//! ```toml
//! [[user]]
//! id = "1"                         # stable, unique - never change it
//! name = "alice"
//! password = "$argon2id$..."       # or $2b$, $y$, $6$, plain:...
//! home = "/srv/evanalyzer/alice"
//! allowed_dirs = ["{home}", "/data/shared"]   # optional
//! ```
//!
//! The `id` identifies the account inside EVAnalyzer (sessions, its
//! worker); `name` is only what the user types to log in, so it can be
//! changed without affecting anything else.
//!
//! Read and checked once when the server starts; restart it after editing.
//! Workers run as the server's own account (there's no system account to
//! switch to), confined to their user's allowed folders.

use crate::user_management::{
    AuthenticationStatus, User, UserManagement, expand_dirs,
    password::{SUPPORTED_FORMATS, is_plain_text, is_supported_format, verify_password},
};
use serde::Deserialize;
use std::{
    collections::HashSet,
    io,
    path::{Path, PathBuf},
};

/// The users file as written.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct UsersFile {
    #[serde(default)]
    user: Vec<UserEntry>,
}

/// One `[[user]]` table.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct UserEntry {
    /// Identifies the account internally, unique in the file. Must stay the
    /// same when `name` changes.
    id: String,
    /// Login name, unique in the file. Used for nothing but logging in.
    name: String,
    /// The password, hashed or `plain:<password>` - formats in
    /// [`crate::user_management::password`].
    password: String,
    /// Absolute home folder: the worker's working directory and where it
    /// keeps the user's EVAnalyzer folder (settings, templates).
    home: PathBuf,
    /// Folders the worker may read and write (`{home}` = `home`). Default:
    /// the config's `users.default_allowed_dirs`.
    allowed_dirs: Option<Vec<String>>,
}

/// The accounts of a users file - see the module docs.
pub struct FileUsers {
    users: Vec<FileUser>,
}

struct FileUser {
    id: String,
    name: String,
    stored_password: String,
    home: PathBuf,
    allowed_dirs: Vec<PathBuf>,
}

impl FileUsers {
    /// Reads and checks the users file at `path`; users without their own
    /// `allowed_dirs` get `default_allowed_dirs`.
    pub fn load(path: &Path, default_allowed_dirs: &[String]) -> io::Result<Self> {
        let text = std::fs::read_to_string(path).map_err(|e| {
            io::Error::new(
                e.kind(),
                format!("Cannot read users file {}: {e}", path.display()),
            )
        })?;
        Self::parse(&text, default_allowed_dirs).map_err(|message| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("Invalid users file {}: {message}", path.display()),
            )
        })
    }

    fn parse(text: &str, default_allowed_dirs: &[String]) -> Result<Self, String> {
        let file: UsersFile = toml::from_str(text).map_err(|e| e.to_string())?;
        let mut ids = HashSet::new();
        let mut names = HashSet::new();
        let mut users = Vec::with_capacity(file.user.len());
        for entry in file.user {
            let name = &entry.name;
            if name.is_empty() {
                return Err("a user has an empty name".into());
            }
            if !names.insert(name.clone()) {
                return Err(format!("user '{name}' is listed twice"));
            }
            if entry.id.is_empty() {
                return Err(format!("user '{name}' has an empty id"));
            }
            if !ids.insert(entry.id.clone()) {
                return Err(format!("user '{name}': id '{}' is used twice", entry.id));
            }
            if !is_supported_format(&entry.password) {
                return Err(format!(
                    "user '{name}': password must be {SUPPORTED_FORMATS}"
                ));
            }
            if is_plain_text(&entry.password) {
                log::warn!("User '{name}' has a plain-text password - store a hash instead");
            }
            if !entry.home.is_absolute() {
                return Err(format!(
                    "user '{name}': home must be an absolute path, got {}",
                    entry.home.display()
                ));
            }
            let allowed_dirs = expand_dirs(
                entry
                    .allowed_dirs
                    .as_deref()
                    .unwrap_or(default_allowed_dirs),
                &entry.home,
            );
            if let Some(dir) = allowed_dirs.iter().find(|dir| !dir.is_absolute()) {
                return Err(format!(
                    "user '{name}': allowed folder {} is not an absolute path",
                    dir.display()
                ));
            }
            users.push(FileUser {
                id: entry.id,
                name: entry.name,
                stored_password: entry.password,
                home: entry.home,
                allowed_dirs,
            });
        }
        if users.is_empty() {
            log::warn!("The users file lists no users - nobody can log in");
        }
        Ok(Self { users })
    }
}

impl UserManagement for FileUsers {
    fn login(&self, username: String, password: String) -> AuthenticationStatus {
        let Some(user) = self.users.iter().find(|user| user.name == username) else {
            return AuthenticationStatus::PasswordWrong;
        };
        if !verify_password(&password, &user.stored_password) {
            return AuthenticationStatus::PasswordWrong;
        }
        AuthenticationStatus::Authenticated(User {
            user_id: user.id.clone(),
            username,
            home: user.home.clone(),
            allowed_dirs: user.allowed_dirs.clone(),
            unix_account: None,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `openssl passwd -6 -salt saltsalt 'correct horse'`
    const HASH: &str = "$6$saltsalt$hRM5XZ86KXEw9UOmjigeVqFgULtFB2sgpC9lXQDfMib3Zgw7mEiUvBJI2EplzfAqxL5Vvwp2scFtv/uamSo5z0";

    fn users(text: &str) -> Result<FileUsers, String> {
        FileUsers::parse(text, &["{home}".into()])
    }

    fn two_users() -> FileUsers {
        users(&format!(
            "[[user]]\nid = \"1\"\nname = \"alice\"\npassword = \"{HASH}\"\nhome = \"/srv/alice\"\n\
             [[user]]\nid = \"2\"\nname = \"bob\"\npassword = \"{HASH}\"\nhome = \"/srv/bob\"\n\
             allowed_dirs = [\"{{home}}\", \"/data/shared\"]\n"
        ))
        .unwrap()
    }

    #[test]
    fn a_listed_user_logs_in_with_their_home_and_allowed_folders() {
        let users = two_users();

        let AuthenticationStatus::Authenticated(bob) =
            users.login("bob".into(), "correct horse".into())
        else {
            panic!("expected a login");
        };
        assert_eq!(bob.user_id, "2");
        assert_eq!(bob.username, "bob");
        assert_eq!(bob.home, PathBuf::from("/srv/bob"));
        assert_eq!(
            bob.allowed_dirs,
            [PathBuf::from("/srv/bob"), PathBuf::from("/data/shared")]
        );
        assert!(bob.unix_account.is_none());

        let AuthenticationStatus::Authenticated(alice) =
            users.login("alice".into(), "correct horse".into())
        else {
            panic!("expected a login");
        };
        assert_eq!(
            alice.allowed_dirs,
            [PathBuf::from("/srv/alice")],
            "the default"
        );
    }

    #[test]
    fn a_wrong_password_or_unknown_user_is_rejected() {
        let users = two_users();
        assert!(matches!(
            users.login("alice".into(), "wrong".into()),
            AuthenticationStatus::PasswordWrong
        ));
        assert!(matches!(
            users.login("carol".into(), "correct horse".into()),
            AuthenticationStatus::PasswordWrong
        ));
    }

    #[test]
    fn mistakes_in_the_file_are_reported_at_startup() {
        let entry_with_id = |id: &str, extra: &str| {
            format!(
                "[[user]]\nid = \"{id}\"\nname = \"alice\"\npassword = \"{HASH}\"\nhome = \"/srv/alice\"\n{extra}"
            )
        };
        let entry = |extra: &str| entry_with_id("1", extra);
        for (text, expected) in [
            (format!("{}{}", entry(""), entry("")), "listed twice"),
            (
                format!(
                    "{}{}",
                    entry(""),
                    entry_with_id("1", "").replace("alice", "bob")
                ),
                "used twice",
            ),
            (entry_with_id("", ""), "empty id"),
            (entry("").replace("id = \"1\"\n", ""), "missing field `id`"),
            (entry("").replace(HASH, "secret"), "password must be"),
            (entry("").replace("/srv/alice", "srv/alice"), "absolute"),
            (entry("allowed_dirs = [\"data\"]"), "allowed folder"),
            (entry("shell = \"/bin/sh\""), "shell"),
        ] {
            let error = users(&text).err().unwrap();
            assert!(error.contains(expected), "{expected}: {error}");
        }
    }

    #[test]
    fn renaming_a_user_keeps_their_id() {
        let renamed = two_users_renamed();
        let AuthenticationStatus::Authenticated(user) =
            renamed.login("robert".into(), "correct horse".into())
        else {
            panic!("expected a login");
        };
        assert_eq!(user.user_id, "2");
        assert_eq!(user.username, "robert");
        assert!(matches!(
            renamed.login("bob".into(), "correct horse".into()),
            AuthenticationStatus::PasswordWrong
        ));
    }

    fn two_users_renamed() -> FileUsers {
        users(&format!(
            "[[user]]\nid = \"1\"\nname = \"alice\"\npassword = \"{HASH}\"\nhome = \"/srv/alice\"\n\
             [[user]]\nid = \"2\"\nname = \"robert\"\npassword = \"{HASH}\"\nhome = \"/srv/bob\"\n"
        ))
        .unwrap()
    }
}
