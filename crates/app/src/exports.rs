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
mod from_view;
pub mod jobs;
pub(crate) mod local;
pub mod naming;
mod types;

pub use from_view::entry_from_view;

/// Why a script didn't run: the user hasn't allowed it (its content) on
/// this computer - `EntryStatus::Skipped` says so in these words.
pub const NOT_ALLOWED: &str = "not allowed yet on this computer";
pub use types::{
    EntryReport, EntryStatus, ExportReport, ExportRun, Interpreters, ScriptPermission,
};

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

/// What a script's OK is kept by (`ScriptPermission::Trusted`, beside its
/// path): its content's SHA-256, as hex - a changed script asks again.
pub fn script_hash(content: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    Sha256::digest(content)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

/// Whether a trusted script is still the one allowed - for the list of
/// trusted scripts (app and command line).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrustedScriptStatus {
    /// Its content is the one allowed: it runs.
    Matches,
    /// It changed since: the next run asks again.
    Changed,
    /// No file there (any more), or it can't be read.
    Missing,
}

/// Reads the trusted script `script` (through `files`: on the server in
/// remote mode) and tells whether it is still the one allowed.
pub fn trusted_script_status(
    files: &dyn crate::files::FileSystem,
    script: &crate::user::TrustedScript,
) -> TrustedScriptStatus {
    match files.read_file(std::path::Path::new(&script.path)) {
        Ok(bytes) if script_hash(&bytes) == script.hash => TrustedScriptStatus::Matches,
        Ok(_) => TrustedScriptStatus::Changed,
        Err(_) => TrustedScriptStatus::Missing,
    }
}

/// How the project draws the images of the results `results_db`
/// (`ExportRun::drawing`): where its images are, the series, the channels'
/// colours and display - a drawn images entry sets its own style, brightness
/// and size. `results_images` the results' images (relative paths), for
/// those the project has below another path (its image folder changed
/// since the run).
pub fn drawing(
    project: &crate::project::ProjectWithRuntime,
    files: &dyn crate::files::FileSystem,
    results_db: &std::path::Path,
    results_images: &[String],
) -> crate::results::DrawnImages {
    let run_root = crate::project::run_image_root(files, results_db);
    let (display, image_display) = project.drawn_image_display();
    let finder = crate::project::ResultsImageFinder::new(
        &project.images.list,
        project.images.root.as_deref(),
        run_root.as_deref(),
    );
    let image_paths = results_images
        .iter()
        .filter_map(|rel| {
            let image = finder.find(std::path::Path::new(rel))?;
            let image = image.to_string_lossy().into_owned();
            (image != *rel).then(|| (rel.clone(), image))
        })
        .collect();
    crate::results::DrawnImages {
        image_root: project.images.root.clone().unwrap_or_default(),
        series: project.images.settings.selected_series.unwrap_or(0),
        max_side: 2048,
        colors: project.get_channel_display_colors().into_iter().collect(),
        style: Default::default(),
        brightness: Default::default(),
        display,
        image_display,
        image_paths,
    }
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

/// The value columns a table of `rows` has without `columns` - none for
/// tracks and measurements (they have their own: all of theirs).
pub fn default_columns(rows: crate::exports::jobs::TableRows) -> Vec<crate::results::Column> {
    use crate::exports::jobs::TableRows;
    match rows {
        TableRows::Object | TableRows::ColocPair => local::table::DEFAULT_OBJECT_COLUMNS.to_vec(),
        TableRows::Image | TableRows::Well | TableRows::Folder => {
            local::table::DEFAULT_GROUPED_COLUMNS.to_vec()
        }
        // A time course sums up values (not the count) per time point.
        TableRows::TimePoint => local::table::DEFAULT_GROUPED_COLUMNS
            .iter()
            .filter(|column| **column != crate::results::Column::Count)
            .cloned()
            .collect(),
        TableRows::Track | TableRows::Measurement | TableRows::ProfilePosition => Vec::new(),
    }
}

/// Columns every table writes first, picked or not (image, class, object
/// id - table format 1): shown ticked and greyed in the picker. The track
/// id is a leading column too, but only when picked.
pub fn is_fixed_column(column: &crate::results::Column) -> bool {
    use crate::results::Column;
    matches!(
        column,
        Column::ImageName | Column::ObjectClass | Column::ObjectId
    )
}

/// Opens the results `database` (where they are: here or on the server)
/// and puts together how `export` runs on them - for the CLI and the
/// editor alike: the user's interpreters, how the project draws its images.
pub fn prepare_run(
    backend: &dyn crate::backend::Backend,
    project: &crate::project::ProjectWithRuntime,
    project_dir: &std::path::Path,
    export: ExportSettings,
    database: &std::path::Path,
    output: Option<std::path::PathBuf>,
    scripts: ScriptPermission,
) -> Result<
    (std::sync::Arc<dyn crate::results::ResultsSource>, ExportRun),
    evanalyzer_cfg::core_types::InternalErrors,
> {
    // The user's settings on the machine the export runs on.
    let settings = backend.load_app_settings().unwrap_or_default();
    let results = backend.open_results(database)?;
    let results_images: Vec<String> = results
        .get_images()?
        .into_iter()
        .map(|image| image.rel_path.to_string_lossy().replace('\\', "/"))
        .collect();
    let run = ExportRun {
        export,
        project_dir: project_dir.to_path_buf(),
        results_dir: database
            .parent()
            .map(std::path::Path::to_path_buf)
            .unwrap_or_default(),
        output,
        scripts,
        interpreters: settings.script_interpreters.clone(),
        drawing: Some(drawing(project, backend.files(), database, &results_images)),
    };
    Ok((results, run))
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
