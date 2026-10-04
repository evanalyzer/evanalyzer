//! Stored passwords - what `/etc/shadow`, the server config and the users
//! file hold. Supported formats:
//!
//! | Format                     | Prefix                           | Generate with                         |
//! |----------------------------|----------------------------------|---------------------------------------|
//! | Argon2 (PHC string)        | `$argon2id$`, `$argon2i$`, `$argon2d$` | `evanalyzer hash-password` ([`hash_password`]) |
//! | bcrypt                     | `$2b$`, `$2a$`, `$2y$`           | `htpasswd -nbBC 12 "" <password>`     |
//! | yescrypt                   | `$y$`                            | `mkpasswd -m yescrypt`                |
//! | sha512-crypt / sha256-crypt| `$6$` / `$5$`                    | `openssl passwd -6`                   |
//! | plain text                 | `plain:`                         | `plain:<password>` - testing only     |
//!
//! Plain text needs its explicit prefix so that a mistyped or truncated hash
//! is rejected instead of silently becoming the password; it can't occur in
//! `/etc/shadow`, whose fields never contain `:`.

use sha_crypt::{PasswordVerifier, ShaCrypt};
use yescrypt::Yescrypt;

/// Marks a stored password as plain text.
pub(crate) const PLAIN_PREFIX: &str = "plain:";

const ARGON2_PREFIXES: [&str; 3] = ["$argon2id$", "$argon2i$", "$argon2d$"];
const BCRYPT_PREFIXES: [&str; 3] = ["$2b$", "$2a$", "$2y$"];
const CRYPT_PREFIXES: [&str; 3] = ["$y$", "$5$", "$6$"];

/// A new Argon2id hash of `password` with a random salt and the OWASP
/// recommended cost (19 MiB, 2 passes, 1 lane) - what `evanalyzer
/// hash-password` prints for the config and users file.
pub fn hash_password(password: &str) -> std::io::Result<String> {
    use argon2::password_hash::PasswordHasher;
    let mut salt = [0u8; 16];
    getrandom::fill(&mut salt).map_err(|e| std::io::Error::other(e.to_string()))?;
    argon2::Argon2::default()
        .hash_password_with_salt(password.as_bytes(), &salt)
        .map(|hash| hash.to_string())
        .map_err(|e| std::io::Error::other(format!("Could not hash the password: {e}")))
}

/// Checks `password` against a stored password in one of the formats in the
/// module docs.
///
/// Locked (`!...`), disabled (`*`) and empty entries never match, so an
/// account without a password cannot log in remotely. Other formats are
/// rejected.
pub(crate) fn verify_password(password: &str, stored: &str) -> bool {
    let bytes = password.as_bytes();
    if let Some(plain) = stored.strip_prefix(PLAIN_PREFIX) {
        !plain.is_empty() && constant_time_eq(bytes, plain.as_bytes())
    } else if starts_with_any(stored, &ARGON2_PREFIXES) {
        argon2::PasswordVerifier::verify_password(&argon2::Argon2::default(), bytes, stored).is_ok()
    } else if starts_with_any(stored, &BCRYPT_PREFIXES) {
        bcrypt::verify(bytes, stored).unwrap_or(false)
    } else if stored.starts_with("$y$") {
        Yescrypt::default().verify_password(bytes, stored).is_ok()
    } else if stored.starts_with("$5$") || stored.starts_with("$6$") {
        ShaCrypt::default().verify_password(bytes, stored).is_ok()
    } else {
        if !stored.is_empty() && !stored.starts_with(['!', '*']) {
            log::warn!("Unsupported password hash format, login rejected");
        }
        false
    }
}

/// Whether `stored` is in a format [`verify_password`] can check - for
/// rejecting a configured password at startup instead of at every login.
pub(crate) fn is_supported_format(stored: &str) -> bool {
    stored
        .strip_prefix(PLAIN_PREFIX)
        .map_or(false, |plain| !plain.is_empty())
        || [&ARGON2_PREFIXES[..], &BCRYPT_PREFIXES, &CRYPT_PREFIXES]
            .concat()
            .iter()
            .any(|prefix| stored.starts_with(prefix))
}

/// Whether `stored` is a plain-text password (worth a warning at startup).
pub(crate) fn is_plain_text(stored: &str) -> bool {
    stored.starts_with(PLAIN_PREFIX)
}

/// The supported formats, for error messages.
pub(crate) const SUPPORTED_FORMATS: &str = "an Argon2 ($argon2id$), bcrypt ($2b$), yescrypt ($y$) or sha-crypt ($6$, $5$) hash, \
     or plain:<password>";

fn starts_with_any(stored: &str, prefixes: &[&str]) -> bool {
    prefixes.iter().any(|prefix| stored.starts_with(prefix))
}

/// Compares without returning early, so response times don't reveal how
/// much of a guess was right.
fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    a.len() == b.len() && a.iter().zip(b).fold(0, |diff, (x, y)| diff | (x ^ y)) == 0
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every supported format, each storing "correct horse".
    const STORED: [&str; 5] = [
        // `openssl passwd -6 -salt saltsalt 'correct horse'`
        "$6$saltsalt$hRM5XZ86KXEw9UOmjigeVqFgULtFB2sgpC9lXQDfMib3Zgw7mEiUvBJI2EplzfAqxL5Vvwp2scFtv/uamSo5z0",
        "plain:correct horse",
        ARGON2,
        BCRYPT,
        // `openssl passwd -5 -salt saltsalt 'correct horse'`
        SHA256,
    ];
    /// Argon2id, m=1024 t=1 p=1 (cheap, for the test only).
    const ARGON2: &str = "$argon2id$v=19$m=1024,t=1,p=1$c2FsdHNhbHRzYWx0$otnZMF7gn3T/mBjwQNsF2/BkISQse8/VHNX8NziXT1c";
    /// bcrypt, cost 4 (cheap, for the test only).
    const BCRYPT: &str = "$2b$04$7QrsodIEEP88l8WlLMWYCuxf7XYDXVE64/XaujWs3S9ebo9k5u8fe";
    const SHA256: &str = "$5$saltsalt$myjXcpMpE2Ofk7fj9hqyNYSn6lmWG4Mqnjx.KIRRr4/";

    #[test]
    fn every_supported_format_accepts_the_right_password_only() {
        for stored in STORED {
            assert!(is_supported_format(stored), "{stored}");
            assert!(verify_password("correct horse", stored), "{stored}");
            assert!(!verify_password("correct horsE", stored), "{stored}");
            assert!(!verify_password("", stored), "{stored}");
        }
    }

    #[test]
    fn a_new_hash_verifies_and_is_salted() {
        let hash = hash_password("correct horse").unwrap();
        assert!(
            hash.starts_with("$argon2id$v=19$m=19456,t=2,p=1$"),
            "{hash}"
        );
        assert!(is_supported_format(&hash));
        assert!(verify_password("correct horse", &hash));
        assert!(!verify_password("correct horsE", &hash));
        assert_ne!(hash, hash_password("correct horse").unwrap(), "random salt");
    }

    #[test]
    fn locked_empty_and_unknown_entries_never_match() {
        for stored in [
            "",
            "!",
            "*",
            "!$6$x$y",
            "plain:",
            "correct horse",
            "$1$md5$abc",
        ] {
            assert!(!is_supported_format(stored), "{stored}");
            assert!(!verify_password("", stored), "{stored}");
            assert!(!verify_password("correct horse", stored), "{stored}");
        }
    }
}
