//! # backend
//!
//! What a front end can ask for: the [`Backend`] trait. [`LocalBackend`]
//! runs everything in this process; `crate::remote::RemoteBackend` sends it
//! to a server, which runs a `LocalBackend` itself.
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-10-10
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.

pub mod local;

pub use local::LocalBackend;

use crate::files::FileSystem;
use crate::images::{ImageMeta, ImageSource};
use crate::jobs::{AnalysisRequest, JobInfo, PreviewRequest, RunningJob, StartPreviewError};
use crate::results::ResultsSource;
use crate::training::{
    NetworkTrainingRequest, RunningTraining, StartTrainingError, TrainedNetworkModel,
    TrainingRequest,
};
use crate::user::settings::AppSettings;
use crate::user::trust::ScriptTrust;
use evanalyzer_cfg::core_types::InternalErrors;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::sync::Arc;

pub type ExportProgressFn<'a> = &'a mut dyn FnMut(&str, usize, usize);

pub trait Backend: Send + Sync {
    /// Starts a full analysis run over every image in the project, writing a
    /// results database under `<project_path>/results`.
    fn start_analysis(&self, req: AnalysisRequest) -> Result<RunningJob, InternalErrors>;

    /// The analyses this backend keeps track of - on a server the running
    /// one (it continues when the client disconnects) and recently finished
    /// ones, oldest first. Nothing for a local backend: its analyses end
    /// with the process.
    fn list_jobs(&self) -> Result<Vec<JobInfo>, InternalErrors> {
        Ok(Vec::new())
    }

    /// Follows the analysis `id` from [`list_jobs`](Self::list_jobs), as if
    /// this client had started it: its progress so far, then live events and
    /// its result (at once, if it has finished).
    fn attach_job(&self, id: &str) -> Result<RunningJob, InternalErrors> {
        Err(InternalErrors::InvalidArgument(format!(
            "No analysis '{id}': only a server keeps track of analyses"
        )))
    }

    /// Drops the finished analysis `id` from [`list_jobs`](Self::list_jobs)
    /// (its results stay on disk).
    fn forget_job(&self, _id: &str) -> Result<(), InternalErrors> {
        Ok(())
    }

    /// Starts a preview run restricted to the tiles visible in the viewport.
    fn start_preview(&self, req: PreviewRequest) -> Result<RunningJob, StartPreviewError>;

    /// Starts training a classifier on the project's labeled data.
    fn start_training(&self, req: TrainingRequest) -> Result<RunningTraining, StartTrainingError>;

    /// Starts training a deep-learning network on the project's annotations
    /// (the Deep-Learning tab). A backend that can't: an error.
    fn start_network_training(
        &self,
        _req: NetworkTrainingRequest,
    ) -> Result<RunningTraining<TrainedNetworkModel>, StartTrainingError> {
        Err(StartTrainingError::Failed(InternalErrors::InvalidArgument(
            "This backend can't train deep-learning networks".into(),
        )))
    }

    /// Opens an image for reading its metadata and tiles. Opening is
    /// expensive (a full format parse) - callers cache the result.
    fn open_image(&self, path: &Path) -> Result<Arc<dyn ImageSource>, InternalErrors>;

    /// Opens a results database (`.evadb`) for querying - the results
    /// window and CLI `view`/`export` only go through this.
    fn open_results(&self, path: &Path) -> Result<Arc<dyn ResultsSource>, InternalErrors>;

    /// Reads just an image's metadata - much cheaper than [`Self::open_image`]
    /// (one reader instead of a pool), for scanning folders of images.
    fn read_image_meta(&self, path: &Path) -> Result<ImageMeta, InternalErrors>;

    /// The file's own metadata of one series that [`ImageMeta`] doesn't
    /// map (Image info › Format metadata), as key/value pairs - read on
    /// request. A backend that can't: an error the dialog shows.
    fn read_format_metadata(
        &self,
        _path: &Path,
        _series: i32,
    ) -> Result<Vec<(String, String)>, InternalErrors> {
        Err(InternalErrors::Io(
            "This backend can't read the file's format metadata".into(),
        ))
    }

    /// Where pipeline/project templates live on the backend's machine:
    /// the user's own templates (new ones are saved there) and the ones
    /// shipped with the application.
    fn template_folders(&self) -> Result<TemplateFolders, InternalErrors>;

    /// Files on the backend's machine - what the file browser shows, and
    /// where projects, templates and models are read from and saved to.
    fn files(&self) -> &dyn FileSystem;

    /// Whether work and files are on another machine - the UI says so,
    /// e.g. in the file browser.
    fn is_remote(&self) -> bool {
        false
    }

    /// Human-readable location of this backend, for logs and status text
    /// ("local", "ws://host:7400").
    fn description(&self) -> String;

    /// Whether the backend can still be reached. A remote backend turns
    /// `false` for good once its connection drops; the UI then warns.
    fn is_connected(&self) -> bool {
        true
    }

    /// How well the connection to the backend is protected - shown in the
    /// status bar.
    fn connection_security(&self) -> ConnectionSecurity {
        ConnectionSecurity::Local
    }

    /// Replaces a dropped connection with a new one to the same worker - no
    /// password needed while the user's worker runs. Open images and results
    /// databases carry over. Nothing to do locally or while connected.
    fn reconnect(&self) -> Result<(), InternalErrors> {
        Ok(())
    }

    /// Closes the connection to a server for good - the window moves on to
    /// another backend. What runs on the server goes on. Nothing locally.
    fn disconnect(&self) {}

    /// Who is logged in on a remote server (`--user`), if anyone.
    fn user(&self) -> Option<String> {
        None
    }

    /// The machine the work runs on - the worker's in remote mode - for the
    /// About dialog. Slow on first call (probing the GPUs opens them), so
    /// call it off the UI thread.
    fn system_info(&self) -> Result<SystemInfo, InternalErrors>;

    /// Image file extensions (lowercase, no dot) the backend's readers
    /// accept - which files the file browser and folder scans treat as
    /// images. Depends on how the backend's build was configured, so a
    /// worker can accept other formats than the client.
    fn image_formats(&self) -> Vec<String>;

    /// The user's app preferences (dark mode, focus mode, ...), kept in the
    /// user folder on the backend's machine - in remote mode in the
    /// logged-in user's home on the server, so they follow the user to
    /// every client. Defaults if none were saved yet.
    fn load_app_settings(&self) -> Result<AppSettings, InternalErrors>;

    /// Saves the user's app preferences - see [`Self::load_app_settings`].
    fn save_app_settings(&self, settings: &AppSettings) -> Result<(), InternalErrors>;

    /// The folders and scripts the user allows exports to run, kept in the
    /// user folder on the backend's machine (where the scripts run) - in a
    /// file of its own, not the settings. None if nothing was trusted yet.
    fn load_script_trust(&self) -> Result<ScriptTrust, InternalErrors>;

    /// Saves the trusted folders and scripts - see
    /// [`Self::load_script_trust`].
    fn save_script_trust(&self, trust: &ScriptTrust) -> Result<(), InternalErrors>;
}

/// What the machine a backend computes on offers.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SystemInfo {
    /// EVAnalyzer version running there.
    pub app_version: String,
    /// Operating system (`linux`, `windows`, `macos`).
    pub os: String,
    /// Logical CPU cores.
    pub cpu_cores: usize,
    /// Total RAM in bytes.
    pub ram_total_bytes: u64,
    /// The GPUs there, discrete ones first; AI models run on the `used`
    /// ones, else on the CPU.
    pub gpus: Vec<Gpu>,
}

/// A GPU of the machine a backend computes on.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Gpu {
    /// As its driver names it (e.g. "NVIDIA GeForce RTX 3060"), or
    /// "Integrated GPU".
    pub name: String,
    /// Has its own memory - AI models run only on discrete GPUs.
    pub discrete: bool,
    /// AI models run on it (`--ai-devices`).
    pub used: bool,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TemplateFolders {
    /// User-created templates; saving a template defaults to this folder.
    pub user: PathBuf,
    /// Templates shipped next to the application binary (read-only).
    pub bundled: PathBuf,
}

/// How the connection to a backend is protected.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConnectionSecurity {
    /// No connection: everything runs on this computer.
    Local,
    /// `ws://`: passwords and data cross the network readable.
    Unencrypted,
    /// `wss://` with the server's certificate checked - against the
    /// fingerprint the user gave, or signed by a public authority.
    Encrypted,
    /// `wss://` without checking the certificate (`--no-tls-verification`):
    /// encrypted, but anyone in between could pose as the server.
    EncryptedUnverified,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_local_backend_keeps_no_analyses_to_list_attach_or_forget() {
        let backend = LocalBackend::default();

        assert!(backend.list_jobs().unwrap().is_empty());
        let Err(InternalErrors::InvalidArgument(text)) = backend.attach_job("job-7") else {
            panic!("attaching locally must fail");
        };
        assert!(text.contains("'job-7'"), "{text}");
        assert!(text.contains("only a server"), "{text}");
        backend.forget_job("job-7").unwrap();
    }

    #[test]
    fn a_local_backend_is_always_connected_and_needs_no_login() {
        let backend = LocalBackend::default();

        assert!(backend.is_connected());
        assert_eq!(backend.connection_security(), ConnectionSecurity::Local);
        assert_eq!(backend.user(), None);
        backend.reconnect().unwrap();
        backend.disconnect();
        // Disconnecting has nothing to close locally.
        assert!(backend.is_connected());
    }
}
