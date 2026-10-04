//! Authenticates against the local Linux user database: the password hash
//! comes from `/etc/shadow`, uid, gid and home folder from `/etc/passwd`.
//!
//! Supported hash formats: yescrypt `$y$` (the default on current
//! Debian/Ubuntu/Fedora), sha512-crypt `$6$` and sha256-crypt `$5$`.
//!
//! Reading `/etc/shadow` requires root or membership in the `shadow` group.

use crate::user_management::{AuthenticationStatus, UnixAccount, User, UserManagement};
use sha_crypt::{PasswordVerifier, ShaCrypt};
use std::{
    fs::File,
    io::{BufRead, BufReader},
    path::{Path, PathBuf},
};
use yescrypt::Yescrypt;

pub struct LinuxUsers {
    pub shadow_file: PathBuf,
    pub passwd_file: PathBuf,
}

impl Default for LinuxUsers {
    fn default() -> Self {
        Self {
            shadow_file: "/etc/shadow".into(),
            passwd_file: "/etc/passwd".into(),
        }
    }
}

impl UserManagement for LinuxUsers {
    fn login(&self, username: String, password: String) -> AuthenticationStatus {
        // A `:` or newline in the name could never match a real entry, and
        // must not be able to confuse the field splitting.
        if username.is_empty() || username.contains([':', '\n', '\0']) {
            return AuthenticationStatus::PasswordWrong;
        }
        let hash = match find_field(&self.shadow_file, &username, 1) {
            Ok(Some(hash)) => hash,
            Ok(None) => return AuthenticationStatus::PasswordWrong,
            Err(err) => {
                log::error!(
                    "Cannot read {}: {err} (root or the shadow group is required)",
                    self.shadow_file.display()
                );
                return AuthenticationStatus::PasswordWrong;
            }
        };
        if !verify_password(&password, &hash) {
            return AuthenticationStatus::PasswordWrong;
        }
        // The worker runs in the user's home and as their account, so a user
        // without a passwd entry can't get one.
        let entry = match passwd_entry_by_name(&self.passwd_file, &username) {
            Ok(Some(entry)) => entry,
            Ok(None) => {
                log::error!(
                    "{username} has a password in {} but no entry in {} - cannot start a worker without a home folder",
                    self.shadow_file.display(),
                    self.passwd_file.display()
                );
                return AuthenticationStatus::PasswordWrong;
            }
            Err(err) => {
                log::error!("Cannot read {}: {err}", self.passwd_file.display());
                return AuthenticationStatus::PasswordWrong;
            }
        };
        AuthenticationStatus::Authenticated(User {
            user_id: entry.uid.to_string(),
            username,
            allowed_dirs: vec![entry.home.clone()],
            home: entry.home,
            unix_account: Some(UnixAccount {
                uid: entry.uid,
                gid: entry.gid,
            }),
        })
    }
}

/// Returns field `index` of the `:`-separated line whose first field is
/// `username` (the format shared by `/etc/passwd` and `/etc/shadow`).
fn find_field(path: &Path, username: &str, index: usize) -> std::io::Result<Option<String>> {
    let reader = BufReader::new(File::open(path)?);
    for line in reader.lines() {
        let line = line?;
        if line.starts_with('#') {
            continue;
        }
        let mut fields = line.split(':');
        if fields.next() == Some(username) {
            return Ok(fields.nth(index - 1).map(str::to_owned));
        }
    }
    Ok(None)
}

/// One `/etc/passwd` line (`name:x:uid:gid:gecos:home:shell`), the parts
/// the server needs.
pub(crate) struct PasswdEntry {
    pub uid: u32,
    pub gid: u32,
    pub home: PathBuf,
}

/// The `/etc/passwd` entry of `username`.
fn passwd_entry_by_name(path: &Path, username: &str) -> std::io::Result<Option<PasswdEntry>> {
    passwd_entry(path, |fields| fields[0] == username)
}

/// The `/etc/passwd` entry of the account with `uid`.
pub(crate) fn passwd_entry_by_uid(path: &Path, uid: u32) -> std::io::Result<Option<PasswdEntry>> {
    let uid = uid.to_string();
    passwd_entry(path, |fields| fields[2] == uid)
}

/// The first well-formed `/etc/passwd` entry whose fields match `select`.
fn passwd_entry(
    path: &Path,
    select: impl Fn(&[&str]) -> bool,
) -> std::io::Result<Option<PasswdEntry>> {
    let reader = BufReader::new(File::open(path)?);
    for line in reader.lines() {
        let line = line?;
        let fields: Vec<&str> = line.split(':').collect();
        if fields.len() < 6 || !select(&fields) {
            continue;
        }
        let (Ok(uid), Ok(gid)) = (fields[2].parse(), fields[3].parse()) else {
            return Ok(None);
        };
        return Ok(Some(PasswdEntry {
            uid,
            gid,
            home: fields[5].into(),
        }));
    }
    Ok(None)
}

/// Checks `password` against a crypt(3) hash from the shadow file.
///
/// Locked (`!...`), disabled (`*`) and empty hashes never match, so an
/// account without a password cannot log in remotely. Hash formats other
/// than yescrypt and sha256/sha512-crypt are rejected.
fn verify_password(password: &str, hash: &str) -> bool {
    let password = password.as_bytes();
    if hash.starts_with("$y$") {
        Yescrypt::default().verify_password(password, hash).is_ok()
    } else if hash.starts_with("$5$") || hash.starts_with("$6$") {
        ShaCrypt::default().verify_password(password, hash).is_ok()
    } else {
        if !hash.is_empty() && !hash.starts_with(['!', '*']) {
            log::warn!("Unsupported password hash format, login rejected");
        }
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    /// `openssl passwd -6 -salt saltsalt 'correct horse'`
    const SHA512_HASH: &str = "$6$saltsalt$hRM5XZ86KXEw9UOmjigeVqFgULtFB2sgpC9lXQDfMib3Zgw7mEiUvBJI2EplzfAqxL5Vvwp2scFtv/uamSo5z0";

    /// yescrypt (the Debian/Ubuntu default) of `hunter2`.
    const YESCRYPT_HASH: &str =
        "$y$j9T$XVkLXR1yKbZGmWnO4Pq0L/$TLTKX2zOmyJD0n2OXwEiH71iPNpxklwvG4TNfrcqziA";

    /// `openssl passwd -5 -salt saltsalt 'correct horse'`
    const SHA256_HASH: &str = "$5$saltsalt$myjXcpMpE2Ofk7fj9hqyNYSn6lmWG4Mqnjx.KIRRr4/";

    struct Fixture {
        _dir: tempfile::TempDir,
        users: LinuxUsers,
    }

    fn fixture(shadow: &str, passwd: &str) -> Fixture {
        let dir = tempfile::tempdir().unwrap();
        let shadow_file = dir.path().join("shadow");
        let passwd_file = dir.path().join("passwd");
        File::create(&shadow_file)
            .unwrap()
            .write_all(shadow.as_bytes())
            .unwrap();
        File::create(&passwd_file)
            .unwrap()
            .write_all(passwd.as_bytes())
            .unwrap();
        Fixture {
            _dir: dir,
            users: LinuxUsers {
                shadow_file,
                passwd_file,
            },
        }
    }

    fn standard() -> Fixture {
        fixture(
            &format!(
                "root:*:19000:0:99999:7:::\n\
                 alice:{SHA512_HASH}:19000:0:99999:7:::\n\
                 bob:{YESCRYPT_HASH}:19000:0:99999:7:::\n\
                 carol:{SHA256_HASH}:19000:0:99999:7:::\n\
                 legacy:$1$saltsalt$abcdefghijklmnopqrstuv:19000:0:99999:7:::\n\
                 locked:!{SHA512_HASH}:19000:0:99999:7:::\n\
                 nopass::19000:0:99999:7:::\n\
                 nohome:{SHA512_HASH}:19000:0:99999:7:::\n"
            ),
            "root:x:0:0:root:/root:/bin/bash\n\
             alice:x:1000:1000:Alice:/home/alice:/bin/bash\n\
             bob:x:1001:1002:Bob:/srv/bob:/bin/bash\n\
             carol:x:1003:1003:Carol:/home/carol:/bin/bash\n",
        )
    }

    fn login(f: &Fixture, user: &str, password: &str) -> AuthenticationStatus {
        f.users.login(user.into(), password.into())
    }

    #[test]
    fn correct_sha512_password_authenticates_with_uid() {
        let f = standard();
        match login(&f, "alice", "correct horse") {
            AuthenticationStatus::Authenticated(user) => {
                assert_eq!(user.username, "alice");
                assert_eq!(user.user_id, "1000");
            }
            _ => panic!("expected authentication"),
        }
    }

    #[test]
    fn the_user_s_home_comes_from_passwd_and_is_the_only_allowed_folder() {
        let f = standard();
        let AuthenticationStatus::Authenticated(user) = login(&f, "bob", "hunter2") else {
            panic!("expected authentication");
        };

        assert_eq!(user.home, PathBuf::from("/srv/bob"));
        assert_eq!(user.allowed_dirs, [PathBuf::from("/srv/bob")]);
        let account = user.unix_account.expect("a unix account");
        assert_eq!((account.uid, account.gid), (1001, 1002));
    }

    #[test]
    fn a_user_without_a_passwd_entry_gets_no_session() {
        // Correct password, but no home folder to run a worker in.
        let f = standard();
        assert!(matches!(
            login(&f, "nohome", "correct horse"),
            AuthenticationStatus::PasswordWrong
        ));
    }

    #[test]
    fn a_passwd_entry_can_be_found_by_uid() {
        let f = standard();
        let entry = passwd_entry_by_uid(&f.users.passwd_file, 1003)
            .unwrap()
            .expect("carol's entry");
        assert_eq!(entry.home, PathBuf::from("/home/carol"));
        assert!(
            passwd_entry_by_uid(&f.users.passwd_file, 4242)
                .unwrap()
                .is_none()
        );
    }

    #[test]
    fn yescrypt_hash_is_supported() {
        let f = standard();
        match login(&f, "bob", "hunter2") {
            AuthenticationStatus::Authenticated(user) => assert_eq!(user.user_id, "1001"),
            _ => panic!("expected authentication"),
        }
        assert!(matches!(
            login(&f, "bob", "hunter3"),
            AuthenticationStatus::PasswordWrong
        ));
    }

    #[test]
    fn sha256_hash_is_supported() {
        let f = standard();
        assert!(matches!(
            login(&f, "carol", "correct horse"),
            AuthenticationStatus::Authenticated(_)
        ));
        assert!(matches!(
            login(&f, "carol", "wrong"),
            AuthenticationStatus::PasswordWrong
        ));
    }

    #[test]
    fn unsupported_hash_format_is_rejected() {
        let f = standard();
        assert!(matches!(
            login(&f, "legacy", "anything"),
            AuthenticationStatus::PasswordWrong
        ));
    }

    #[test]
    fn wrong_password_is_rejected() {
        let f = standard();
        assert!(matches!(
            login(&f, "alice", "correct horse "),
            AuthenticationStatus::PasswordWrong
        ));
        assert!(matches!(
            login(&f, "alice", ""),
            AuthenticationStatus::PasswordWrong
        ));
    }

    #[test]
    fn unknown_user_is_reported() {
        let f = standard();
        assert!(matches!(
            login(&f, "mallory", "x"),
            AuthenticationStatus::PasswordWrong
        ));
        assert!(matches!(
            login(&f, "", ""),
            AuthenticationStatus::PasswordWrong
        ));
        assert!(matches!(
            login(&f, "alice:x", "correct horse"),
            AuthenticationStatus::PasswordWrong
        ));
    }

    #[test]
    fn locked_disabled_and_empty_accounts_never_authenticate() {
        let f = standard();
        assert!(matches!(
            login(&f, "locked", "correct horse"),
            AuthenticationStatus::PasswordWrong
        ));
        assert!(matches!(
            login(&f, "root", "*"),
            AuthenticationStatus::PasswordWrong
        ));
        assert!(matches!(
            login(&f, "nopass", ""),
            AuthenticationStatus::PasswordWrong
        ));
    }

    #[test]
    fn password_with_nul_byte_is_rejected() {
        let f = standard();
        assert!(matches!(
            login(&f, "alice", "correct\0horse"),
            AuthenticationStatus::PasswordWrong
        ));
    }

    #[test]
    fn unreadable_shadow_file_rejects_login() {
        let users = LinuxUsers {
            shadow_file: "/nonexistent/shadow".into(),
            passwd_file: "/nonexistent/passwd".into(),
        };
        assert!(matches!(
            users.login("alice".into(), "correct horse".into()),
            AuthenticationStatus::PasswordWrong
        ));
    }
}
