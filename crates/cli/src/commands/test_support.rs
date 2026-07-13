//! Test-only scratch-directory helper shared by command test modules.
//!
//! Hand-rolled instead of pulling in the `tempfile` crate (not currently a
//! dependency of this crate) - all a command test needs is "a project file
//! on disk nobody else is using", which `std::env::temp_dir()` plus a unique
//! subdirectory name already gives us.
#![cfg(test)]

use evanalyzer_cfg::settings::project_settings::ProjectSettings;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

static COUNTER: AtomicU64 = AtomicU64::new(0);

pub(crate) struct TempProjectFile {
    dir: PathBuf,
    pub(crate) path: PathBuf,
}

impl TempProjectFile {
    pub(crate) fn new(settings: &ProjectSettings) -> Self {
        let n = COUNTER.fetch_add(1, Ordering::Relaxed);
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir = std::env::temp_dir().join(format!(
            "evanalyzer_cli_test_{}_{n}_{nanos}",
            std::process::id()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("project.evaproj");
        std::fs::write(&path, serde_json::to_string(settings).unwrap()).unwrap();
        Self { dir, path }
    }
}

impl Drop for TempProjectFile {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.dir);
    }
}
