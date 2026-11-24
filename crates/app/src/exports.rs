//! # exports
//!
//! The project's exports (`ProjectSettings::exports`, see `docs/exports.md`):
//! named recipes of the tables, plate layouts, drawn images and scripts to
//! write from a run's results.
//!
//! - [`check`]: what is wrong with a project's exports - for
//!   `project validate`, the editor and before every run.
//! - [`naming`]: the file names an export writes (automatic names, the
//!   prefix, unique names per image).
//! - [`ExportRun`] / [`ExportReport`]: what a run gets and what it wrote.
//!   It runs where the results are (`ResultsSource::run_export`): here,
//!   or on the server.
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-11-24
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.

pub mod check;
pub(crate) mod local;
pub mod naming;
mod types;

pub use types::{EntryReport, EntryStatus, ExportReport, ExportRun, ScriptPermission};

use evanalyzer_cfg::settings::export_settings::ExportSettings;

/// The export `id_or_name` names among `exports`: by id if it is a number
/// one of them has, else by name ignoring case.
pub fn find_export<'a>(
    exports: &'a [ExportSettings],
    id_or_name: &str,
) -> Option<&'a ExportSettings> {
    let wanted = id_or_name.trim();
    wanted
        .parse::<u32>()
        .ok()
        .and_then(|id| exports.iter().find(|export| export.id == id))
        .or_else(|| {
            exports
                .iter()
                .find(|export| export.name.trim().eq_ignore_ascii_case(wanted))
        })
}

/// The results database an export runs on: `given` (a `.evadb` file, or
/// the folder of one), else the project's newest analysis (the folders of
/// `<project_dir>/results/` are named date first; the preview's isn't one).
pub fn results_database(
    files: &dyn crate::files::FileSystem,
    project_dir: &std::path::Path,
    given: Option<&std::path::Path>,
) -> Result<std::path::PathBuf, evanalyzer_cfg::core_types::InternalErrors> {
    use evanalyzer_cfg::RESULTS_FILE_EXTENSION;
    use evanalyzer_cfg::core_types::InternalErrors;
    let database_in =
        |dir: &std::path::Path| -> Result<Option<std::path::PathBuf>, InternalErrors> {
            Ok(files
                .list_dir(dir)?
                .into_iter()
                .filter(|entry| !entry.is_dir)
                .map(|entry| entry.path)
                .find(|path| {
                    path.extension().and_then(|e| e.to_str()) == Some(RESULTS_FILE_EXTENSION)
                }))
        };
    if let Some(given) = given {
        let is_dir = files.stat(given)?.is_some_and(|entry| entry.is_dir);
        if !is_dir {
            return Ok(given.to_path_buf());
        }
        return database_in(given)?.ok_or_else(|| {
            InternalErrors::InvalidArgument(format!("no results database in {}", given.display()))
        });
    }
    let results = project_dir.join("results");
    let mut runs: Vec<_> = files
        .list_dir(&results)
        .unwrap_or_default()
        .into_iter()
        .filter(|entry| entry.is_dir && entry.name != "preview")
        .collect();
    runs.sort_by(|a, b| b.name.cmp(&a.name));
    for run in runs {
        if let Some(database) = database_in(&run.path)? {
            return Ok(database);
        }
    }
    Err(InternalErrors::InvalidArgument(format!(
        "the project has no analysis yet (nothing in {}) - run one first, or give --results",
        results.display()
    )))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::files::LocalFileSystem;

    /// The newest analysis, not the preview; a given folder or file.
    #[test]
    fn the_results_database_is_the_given_one_or_the_newest() {
        let dir = tempfile::tempdir().unwrap();
        let project = dir.path();
        for run in ["2026-11-01__a", "2026-11-20__b", "preview"] {
            std::fs::create_dir_all(project.join("results").join(run)).unwrap();
            std::fs::write(project.join("results").join(run).join("r.evadb"), b"").unwrap();
        }
        // A newer folder without a database is passed over.
        std::fs::create_dir_all(project.join("results/2026-11-30__empty")).unwrap();
        let files = LocalFileSystem::default();
        assert_eq!(
            results_database(&files, project, None).unwrap(),
            project.join("results/2026-11-20__b/r.evadb")
        );
        let given = project.join("results/2026-11-01__a");
        assert_eq!(
            results_database(&files, project, Some(&given)).unwrap(),
            given.join("r.evadb")
        );
        let file = given.join("r.evadb");
        assert_eq!(
            results_database(&files, project, Some(&file)).unwrap(),
            file
        );
        let empty = tempfile::tempdir().unwrap();
        assert!(results_database(&files, empty.path(), None).is_err());
    }

    fn export(id: u32, name: &str) -> ExportSettings {
        serde_json::from_value(serde_json::json!({ "id": id, "name": name })).unwrap()
    }

    /// By id, else by name ignoring case - a name that is a number still
    /// finds its export when no id matches.
    #[test]
    fn an_export_is_found_by_id_or_name() {
        let exports = [export(3, "Per image"), export(7, "2024")];
        assert_eq!(find_export(&exports, "3").unwrap().id, 3);
        assert_eq!(find_export(&exports, "per IMAGE ").unwrap().id, 3);
        assert_eq!(find_export(&exports, "2024").unwrap().id, 7);
        assert!(find_export(&exports, "9").is_none());
        assert!(find_export(&exports, "nothing").is_none());
    }
}
