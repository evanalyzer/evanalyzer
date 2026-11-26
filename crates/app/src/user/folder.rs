//! # folder
//!
//! The user's EVAnalyzer folder: its name in the home folder and where the
//! settings file lies in it.
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-11-22
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.

use std::path::{Path, PathBuf};

/// Name of the EVAnalyzer folder in a home folder: hidden on Linux, where
/// that is the convention for per-user state, visible elsewhere so users
/// find it in their file manager. Lower case everywhere, so it is the same
/// folder on case-insensitive and case-sensitive file systems.
pub const USER_FOLDER_NAME: &str = if cfg!(target_os = "linux") {
    ".evanalyzer"
} else {
    "evanalyzer"
};

/// Returns this account's EVAnalyzer folder (`~/.evanalyzer` on Linux,
/// `~/evanalyzer` on macOS and Windows), where user settings, templates, and
/// other per-user state are stored - see [`user_folder_in`].
///
/// The folder (and its parents) is created if it does not exist yet.
pub fn get_user_folder() -> PathBuf {
    let home = dirs::home_dir().unwrap_or_else(std::env::temp_dir);
    let folder = user_folder_in(&home);
    let _ = std::fs::create_dir_all(&folder);
    folder
}

/// The EVAnalyzer folder in `home` - this account's home or a worker's
/// `--home`, so a user finds the same folder whether they work on the
/// machine directly or through a worker. Its layout:
///
/// ```text
/// <home>/.evanalyzer/        (evanalyzer/ on macOS and Windows)
/// ├── settings.json          app settings, incl. pinned server fingerprints
/// ├── trust.json             the exports' scripts this user allows to run
/// ├── templates/             the user's templates and modules
/// ├── logs/crash.log
/// └── cache/gpu/             the GPU's autotune results and compiled kernels
///                            (deletable: costs one slow first model run)
/// ```
pub fn user_folder_in(home: &Path) -> PathBuf {
    home.join(USER_FOLDER_NAME)
}

/// The settings file inside a user folder (`<user folder>/settings.json`).
pub fn settings_file_in(user_folder: &Path) -> PathBuf {
    user_folder.join("settings.json")
}

/// The trusted scripts file inside a user folder (`<user folder>/trust.json`).
pub fn trust_file_in(user_folder: &Path) -> PathBuf {
    user_folder.join("trust.json")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_user_folder_is_hidden_on_linux_and_visible_elsewhere() {
        let folder = user_folder_in(Path::new("/home/alice"));

        let expected = if cfg!(target_os = "linux") {
            Path::new("/home/alice/.evanalyzer")
        } else {
            Path::new("/home/alice/evanalyzer")
        };
        assert_eq!(folder, expected);
    }

    #[test]
    fn get_user_folder_creates_dir_and_ends_with_evanalyzer() {
        let folder = get_user_folder();

        assert!(folder.is_dir());
        assert_eq!(folder.file_name().unwrap(), USER_FOLDER_NAME);
    }
}
