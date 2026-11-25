//! # trust
//!
//! The exports' scripts this user allows to run: whole folders (every
//! script in them, whatever its content) and single scripts (by path, while
//! their content - its hash - is the one allowed). Kept in a file of its
//! own in the user folder (`trust.json`), never in the settings, which get
//! copied to other computers, nor in a project, so a project can't allow
//! itself.
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-11-25
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.

use serde::{Deserialize, Serialize};
use std::path::{Component, Path, PathBuf};

/// The format of `trust.json`; a change that older versions would misread
/// raises it.
pub const TRUST_VERSION: u32 = 1;

/// The folders and scripts this user trusts.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ScriptTrust {
    #[serde(default)]
    pub version: u32,
    /// Every script below one of these runs, also after it changed.
    #[serde(default)]
    pub folders: Vec<TrustedFolder>,
    /// These run while their content is the one allowed - one entry per
    /// path; allowing a changed script again replaces its hash.
    #[serde(default)]
    pub scripts: Vec<TrustedScript>,
}

/// A trusted folder: absolute, as [`normalize`] writes it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TrustedFolder {
    pub path: String,
    /// The day it was trusted (`2026-11-25`; empty: not known).
    #[serde(default)]
    pub since: String,
}

/// A trusted script: absolute path (as [`normalize`] writes it) and its
/// content's hash (`exports::script_hash`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TrustedScript {
    pub path: String,
    pub hash: String,
    /// The day it was allowed (`2026-11-25`; empty: not known).
    #[serde(default)]
    pub since: String,
}

/// Why a script may run or not - [`ScriptTrust::check`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TrustCheck {
    /// It lies in this trusted folder.
    Folder(String),
    /// This very content of it was allowed.
    Script,
    /// It was allowed, but its content changed since (the day it was).
    Changed(String),
    /// Never allowed.
    Unknown,
}

impl TrustCheck {
    pub fn allowed(&self) -> bool {
        matches!(self, Self::Folder(_) | Self::Script)
    }
}

impl ScriptTrust {
    /// Whether the script at `path` with content hash `hash` may run, and
    /// why.
    pub fn check(&self, path: &Path, hash: &str) -> TrustCheck {
        let script = normalize(path);
        if let Some(folder) = self
            .folders
            .iter()
            .find(|folder| is_within(&script, &folder.path))
        {
            return TrustCheck::Folder(folder.path.clone());
        }
        match self
            .scripts
            .iter()
            .find(|entry| same_path(&entry.path, &script))
        {
            Some(entry) if entry.hash == hash => TrustCheck::Script,
            Some(entry) => TrustCheck::Changed(entry.since.clone()),
            None => TrustCheck::Unknown,
        }
    }

    /// Whether the script at `path` with content hash `hash` may run.
    pub fn allows(&self, path: &Path, hash: &str) -> bool {
        self.check(path, hash).allowed()
    }

    /// Allows the script at `path` while its content hash is `hash` -
    /// replacing the hash allowed for it before.
    pub fn allow_script(&mut self, path: &Path, hash: &str, since: &str) {
        let path = normalize(path);
        self.scripts.retain(|entry| !same_path(&entry.path, &path));
        self.scripts.push(TrustedScript {
            path,
            hash: hash.to_string(),
            since: since.to_string(),
        });
    }

    /// Trusts every script below `folder`.
    pub fn trust_folder(&mut self, folder: &Path, since: &str) {
        let path = normalize(folder);
        if self
            .folders
            .iter()
            .any(|entry| same_path(&entry.path, &path))
        {
            return;
        }
        self.folders.push(TrustedFolder {
            path,
            since: since.to_string(),
        });
    }

    /// Takes the trust of the folder `path` back (as listed).
    pub fn revoke_folder(&mut self, path: &str) {
        self.folders.retain(|entry| entry.path != path);
    }

    /// Takes the trust of the script `path` back (as listed).
    pub fn revoke_script(&mut self, path: &str) {
        self.scripts.retain(|entry| entry.path != path);
    }

    pub fn is_empty(&self) -> bool {
        self.folders.is_empty() && self.scripts.is_empty()
    }
}

/// Today, as the trust file keeps it (`2026-11-25`).
pub fn today() -> String {
    chrono::Local::now().format("%Y-%m-%d").to_string()
}

/// `path` without `.` and `..` (worked out on the path, not the disk - the
/// files may be on a server), so `scripts/../../x.R` isn't taken for a
/// script inside the project folder.
pub fn normalize(path: &Path) -> String {
    let mut parts: Vec<Component> = Vec::new();
    for part in path.components() {
        match part {
            Component::CurDir => {}
            Component::ParentDir => {
                if matches!(parts.last(), Some(Component::Normal(_))) {
                    parts.pop();
                } else if parts.is_empty() {
                    parts.push(part);
                }
            }
            _ => parts.push(part),
        }
    }
    parts
        .iter()
        .collect::<PathBuf>()
        .to_string_lossy()
        .into_owned()
}

/// Whether two normalized paths are the same - ignoring case on Windows.
fn same_path(a: &str, b: &str) -> bool {
    if cfg!(windows) {
        a.to_lowercase() == b.to_lowercase()
    } else {
        a == b
    }
}

/// Whether the normalized `path` lies below the normalized `folder` (whole
/// names: `/data/a` isn't below `/data/ab`).
fn is_within(path: &str, folder: &str) -> bool {
    if cfg!(windows) {
        Path::new(&path.to_lowercase()).starts_with(folder.to_lowercase())
    } else {
        Path::new(path).starts_with(folder)
    }
}

/// Loads the trust from `path` - none if the file doesn't exist yet or
/// can't be read (nothing runs without asking then).
pub fn load_trust_from(path: &Path) -> ScriptTrust {
    std::fs::read_to_string(path)
        .ok()
        .and_then(|data| serde_json::from_str::<ScriptTrust>(&data).ok())
        .unwrap_or_default()
}

/// Writes the trust to `path`, creating its folder if needed.
pub fn save_trust_to(path: &Path, trust: &ScriptTrust) -> std::io::Result<()> {
    if let Some(folder) = path.parent() {
        std::fs::create_dir_all(folder)?;
    }
    let trust = ScriptTrust {
        version: TRUST_VERSION,
        ..trust.clone()
    };
    let json = serde_json::to_string_pretty(&trust).map_err(std::io::Error::other)?;
    std::fs::write(path, json)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn abs(path: &str) -> PathBuf {
        if cfg!(windows) {
            PathBuf::from(format!("C:{}", path.replace('/', "\\")))
        } else {
            PathBuf::from(path)
        }
    }

    #[test]
    fn a_script_in_a_trusted_folder_runs_whatever_its_content() {
        let mut trust = ScriptTrust::default();
        trust.trust_folder(&abs("/projects/liver"), "2026-11-25");
        let script = abs("/projects/liver/scripts/plot.R");
        assert!(trust.allows(&script, "aaa"));
        assert!(trust.allows(&script, "bbb"));
        assert_eq!(
            trust.check(&script, "aaa"),
            TrustCheck::Folder(normalize(&abs("/projects/liver")))
        );
    }

    #[test]
    fn a_folder_covers_only_what_lies_below_it() {
        let mut trust = ScriptTrust::default();
        trust.trust_folder(&abs("/projects/liver"), "");
        // A longer name beside it.
        assert!(!trust.allows(&abs("/projects/liver2/scripts/plot.R"), "a"));
        // Out of it by `..`.
        assert!(!trust.allows(&abs("/projects/liver/scripts/../../evil.R"), "a"));
        assert!(trust.allows(&abs("/projects/liver/scripts/../plot.R"), "a"));
    }

    #[test]
    fn a_script_runs_while_unchanged_and_says_when_it_changed() {
        let mut trust = ScriptTrust::default();
        let script = abs("/projects/kidney/scripts/summary.R");
        trust.allow_script(&script, "v1", "2026-11-20");
        assert_eq!(trust.check(&script, "v1"), TrustCheck::Script);
        assert_eq!(
            trust.check(&script, "v2"),
            TrustCheck::Changed("2026-11-20".into())
        );
        assert_eq!(
            trust.check(&abs("/projects/kidney/scripts/other.R"), "v1"),
            TrustCheck::Unknown,
            "a hash alone allows no other script"
        );
    }

    #[test]
    fn allowing_a_changed_script_again_replaces_its_entry() {
        let mut trust = ScriptTrust::default();
        let script = abs("/projects/kidney/scripts/summary.R");
        trust.allow_script(&script, "v1", "2026-11-20");
        trust.allow_script(
            &abs("/projects/kidney/scripts/./summary.R"),
            "v2",
            "2026-11-25",
        );
        assert_eq!(trust.scripts.len(), 1);
        assert_eq!(trust.scripts[0].hash, "v2");
        assert_eq!(trust.scripts[0].since, "2026-11-25");
        assert!(!trust.allows(&script, "v1"));
    }

    #[test]
    fn a_folder_is_trusted_once_and_revoking_removes_entries() {
        let mut trust = ScriptTrust::default();
        trust.trust_folder(&abs("/projects/liver"), "");
        trust.trust_folder(&abs("/projects/liver/"), "");
        assert_eq!(trust.folders.len(), 1);
        let script = abs("/projects/kidney/a.R");
        trust.allow_script(&script, "h", "");
        let (folder, script_path) = (trust.folders[0].path.clone(), trust.scripts[0].path.clone());
        trust.revoke_folder(&folder);
        trust.revoke_script(&script_path);
        assert!(trust.is_empty());
        assert!(!trust.allows(&script, "h"));
    }

    #[test]
    fn the_trust_file_round_trips_and_a_missing_one_trusts_nothing() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("trust.json");
        assert!(load_trust_from(&path).is_empty());
        let mut trust = ScriptTrust::default();
        trust.trust_folder(&abs("/projects/liver"), "2026-11-25");
        trust.allow_script(&abs("/projects/kidney/a.R"), "h", "2026-11-24");
        save_trust_to(&path, &trust).unwrap();
        let loaded = load_trust_from(&path);
        assert_eq!(loaded.version, TRUST_VERSION);
        assert_eq!(loaded.folders, trust.folders);
        assert_eq!(loaded.scripts, trust.scripts);
        // A broken file trusts nothing.
        std::fs::write(&path, "{ not json").unwrap();
        assert!(load_trust_from(&path).is_empty());
    }
}
