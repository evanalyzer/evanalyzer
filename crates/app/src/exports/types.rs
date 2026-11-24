//! # types
//!
//! What travels for an export run: what to run ([`ExportRun`]) and what
//! became of it ([`ExportReport`]). Plain data - the same on this machine
//! and over the wire to a server.
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-11-24
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.

use crate::results::DrawnImages;
use evanalyzer_cfg::settings::export_settings::ExportSettings;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// One export to run on one run's results.
#[derive(Clone, Serialize, Deserialize)]
pub struct ExportRun {
    pub export: ExportSettings,
    /// The project's folder: a fixed output folder and the scripts are
    /// relative to it.
    pub project_dir: PathBuf,
    /// The folder of the results database the export runs on
    /// (`OutputLocation::ResultsFolder`).
    pub results_dir: PathBuf,
    /// Write into this folder instead of the export's own output, this once
    /// (`export run --output`).
    pub output: Option<PathBuf>,
    /// Whether the export's scripts run.
    pub scripts: ScriptPermission,
    /// Where the interpreters are on the machine the export runs on;
    /// unset: `Rscript` / `python3` from the `PATH`.
    #[serde(default)]
    pub interpreters: Interpreters,
    /// How the project draws its images (root, channel colours, display) -
    /// for drawn images entries; `None`: such entries are skipped.
    pub drawing: Option<DrawnImages>,
}

/// Whether an export's scripts run.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum ScriptPermission {
    /// Not at all (skipped, the tables are written).
    #[default]
    Skip,
    /// Those whose content (its hash)
    /// is among these - the ones the user allowed.
    Trusted(Vec<String>),
    /// All of them (`--allow-scripts`).
    All,
    /// None, and why (a server that doesn't allow scripts).
    Refused(String),
}

/// The interpreters scripts run with - a setting of the computer (or
/// server), never of the project.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct Interpreters {
    /// `Rscript`; unset: `Rscript` from the `PATH`.
    pub r: Option<PathBuf>,
    /// `python`; unset: `python3` (`python` on Windows) from the `PATH`.
    pub python: Option<PathBuf>,
}

/// What an export run wrote.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ExportReport {
    /// The folder the files went into.
    pub folder: PathBuf,
    /// Per entry, in the export's order.
    pub entries: Vec<EntryReport>,
}

impl ExportReport {
    /// Whether every entry was written (none failed or was skipped).
    pub fn all_written(&self) -> bool {
        self.entries
            .iter()
            .all(|entry| matches!(entry.status, EntryStatus::Written))
    }
}

/// What became of one entry.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EntryReport {
    /// Its place in the export (0 the first).
    pub entry: usize,
    pub status: EntryStatus,
    /// The files written, relative to [`ExportReport::folder`].
    pub files: Vec<PathBuf>,
    /// What the user should know, e.g. columns these results don't have.
    pub notes: Vec<String>,
    /// A script's output (stdout and stderr), empty for other entries.
    pub log: String,
}

/// See [`EntryReport::status`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum EntryStatus {
    Written,
    /// Not run, and why (a script not allowed, no image settings).
    Skipped(String),
    /// Failed, and why - the other entries still ran.
    Failed(String),
}
