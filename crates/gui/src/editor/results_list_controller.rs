//! # results_list_controller
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-06-05
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.

use crate::AppWindow;
use crate::editor::results_state_controller::ResultsStateController;
use crate::{PipelineRunningState, ResultItemData, ResultsListState, UiState};
use evanalyzer_app::fs::FileSystem;
use evanalyzer_app::results::{AnalysisFolder, analysis_folder, delete_analysis};
use evanalyzer_cfg::RESULTS_FILE_EXTENSION;
use log::warn;
use slint::ComponentHandle;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

pub struct ResultsListController {
    pub(crate) ui: slint::Weak<AppWindow>,
    pub(crate) app_state: Arc<UiState>,
    pub(crate) results_state_controller: Arc<ResultsStateController>,
    /// The results database of the analysis that finished last - what its
    /// note's "Open results" opens.
    last_analysis_results: Mutex<Option<PathBuf>>,
    /// The folder the running analysis writes into: not to be deleted.
    running_analysis: Mutex<Option<PathBuf>>,
    /// The analysis the delete question is about, and whether it is the
    /// one open in Results.
    pending_delete: Mutex<Option<(AnalysisFolder, bool)>>,
}

impl ResultsListController {
    pub fn new(
        ui: slint::Weak<AppWindow>,
        app_state: Arc<UiState>,
        results_state_controller: Arc<ResultsStateController>,
    ) -> Self {
        Self {
            ui,
            app_state: app_state.clone(),
            results_state_controller,
            last_analysis_results: Mutex::new(None),
            running_analysis: Mutex::new(None),
            pending_delete: Mutex::new(None),
        }
    }

    /// The folder an analysis now writes into (`None`: none runs).
    pub fn set_running_analysis(&self, dir: Option<PathBuf>) {
        *self.running_analysis.lock().expect("Poisoned") = dir;
    }

    /// An analysis finished, writing into `output_dir`: its results database
    /// is the one "Open results" opens, and the list shows it.
    pub fn analysis_finished(&self, output_dir: &Path) {
        let backend = self.app_state.backend();
        let found = results_database_in(backend.files(), output_dir);
        if found.is_none() {
            warn!("No results database in {output_dir:?}");
        }
        *self.last_analysis_results.lock().expect("Poisoned") = found;
        self.sync_results_files_to_slint();
    }

    pub fn attach_callbacks(self: &Arc<Self>) {
        let ui_handle = self.ui.clone();
        if let Some(ui) = ui_handle.upgrade() {
            let manager = self.clone();
            ui.global::<ResultsListState>().on_refresh_clicked(move || {
                manager.sync_results_files_to_slint();
            });

            let table = self.results_state_controller.clone();
            ui.global::<ResultsListState>()
                .on_result_selected(move |path| {
                    table.open_database(PathBuf::from(path.as_str()));
                });

            let manager = self.clone();
            ui.global::<ResultsListState>()
                .on_open_folder_clicked(move || {
                    manager.open_results_folder();
                });

            let manager = self.clone();
            ui.global::<ResultsListState>()
                .on_show_in_folder(move |path| {
                    if let Some(dir) = Path::new(path.as_str()).parent() {
                        manager.open_folder(dir.to_path_buf());
                    }
                });

            let manager = self.clone();
            let weak = ui.as_weak();
            ui.global::<ResultsListState>()
                .on_delete_requested(move |path| {
                    let Some(ui) = weak.upgrade() else {
                        return;
                    };
                    let state = ui.global::<ResultsListState>();
                    let open_now = open_results_path(&ui).as_deref() == Some(path.as_str());
                    state.set_delete_error("".into());
                    state.set_delete_busy(false);
                    manager.ask_to_delete(PathBuf::from(path.as_str()), open_now);
                });

            let manager = self.clone();
            let weak = ui.as_weak();
            ui.global::<ResultsListState>()
                .on_delete_confirmed(move || {
                    // The analysis open now - selected again in the new list.
                    let open = weak.upgrade().and_then(|ui| open_results_path(&ui));
                    manager.delete_pending(open);
                });

            // The finished analysis' note: its results, not whichever
            // database the Results view had open before.
            let manager = self.clone();
            ui.global::<PipelineRunningState>()
                .on_open_results(move || {
                    let last = manager
                        .last_analysis_results
                        .lock()
                        .expect("Poisoned")
                        .clone();
                    match last {
                        Some(path) => manager.results_state_controller.open_database(path),
                        None => warn!("No finished analysis' results to open"),
                    }
                });
        }
    }

    /// Another project: its results list, nothing of the last one's open
    /// or selected results.
    pub fn project_switched(&self) {
        *self.last_analysis_results.lock().expect("Poisoned") = None;
        self.results_state_controller.close_database();
        let ui_handle = self.ui.clone();
        let _ = crate::helper::ui_thread::invoke_from_event_loop(move || {
            if let Some(ui) = ui_handle.upgrade() {
                ui.global::<ResultsListState>()
                    .set_selected_result_index(-1);
            }
        });
        self.sync_results_files_to_slint();
    }

    /// Recursively scans `<project_dir>/results/` for `*.RESULTS_FILE_EXTENSION`
    /// files and pushes them to the Slint results list, sorted by modification
    /// time (newest first).
    pub fn sync_results_files_to_slint(&self) {
        let project = self.app_state.get_project();

        // A project not saved yet has no results folder: an empty list (not
        // the last project's).
        let results_dir = project
            .tmp_settings
            .current_project
            .as_ref()
            .and_then(|current_project| current_project.parent())
            .map(|project_dir| project_dir.join("results"));

        // Drop the read lock before doing file I/O
        drop(project);

        let mut items: Vec<(i64, ResultItemData)> = Vec::new();
        if let Some(results_dir) = results_dir {
            let backend = self.app_state.backend();
            let files = backend.files();
            if matches!(files.stat(&results_dir), Ok(Some(entry)) if entry.is_dir) {
                collect_results_files(files, &results_dir, &mut items);
            }
        }

        // Newest first
        items.sort_by(|a, b| b.0.cmp(&a.0));
        let slint_items: Vec<ResultItemData> = items.into_iter().map(|(_, d)| d).collect();

        let ui_handle = self.ui.clone();
        let _ = crate::helper::ui_thread::invoke_from_event_loop(move || {
            if let Some(ui) = ui_handle.upgrade() {
                ui.global::<ResultsListState>()
                    .set_results_list(slint::ModelRc::new(slint::VecModel::from(slint_items)));
            }
        });
    }

    /// The project's folder (`None`: not saved yet).
    fn project_dir(&self) -> Option<PathBuf> {
        let project = self.app_state.get_project();
        project
            .tmp_settings
            .current_project
            .as_ref()
            .and_then(|p| p.parent())
            .map(Path::to_path_buf)
    }

    /// Opens the question for deleting the analysis of `results_file`:
    /// what its folder holds (read off the UI thread - a run's control
    /// images can be many), or that it is still running.
    fn ask_to_delete(self: &Arc<Self>, results_file: PathBuf, open_now: bool) {
        let Some(project_dir) = self.project_dir() else {
            return;
        };
        let manager = self.clone();
        crate::helper::ui_thread::spawn(move || {
            let backend = manager.app_state.backend();
            let folder = match analysis_folder(backend.files(), &project_dir, &results_file) {
                Ok(folder) => folder,
                Err(e) => {
                    warn!("Can't delete {results_file:?}: {e}");
                    manager.sync_results_files_to_slint();
                    return;
                }
            };
            let running = manager
                .running_analysis
                .lock()
                .expect("Poisoned")
                .as_deref()
                == Some(folder.dir.as_path());
            let shown = |bytes: u64| format_file_size(bytes);
            let images = match folder.images {
                0 => String::new(),
                1 => "1 file".to_string(),
                n => format!("{n} files"),
            };
            let other = match folder.other_bytes {
                0 => String::new(),
                b => shown(b),
            };
            let (name, dir) = (
                folder.name.clone(),
                relative_folder(&project_dir, &folder.dir),
            );
            let (results, images_size) = (shown(folder.results_bytes), shown(folder.images_bytes));
            let (settings, total) = (shown(folder.settings_bytes), shown(folder.total_bytes()));
            *manager.pending_delete.lock().expect("Poisoned") = Some((folder, open_now));
            let ui_handle = manager.ui.clone();
            let _ = crate::helper::ui_thread::invoke_from_event_loop(move || {
                let Some(ui) = ui_handle.upgrade() else {
                    return;
                };
                let state = ui.global::<ResultsListState>();
                state.set_delete_name(name.into());
                state.set_delete_folder(dir.into());
                state.set_delete_results(results.into());
                state.set_delete_images(images.into());
                state.set_delete_images_size(images_size.into());
                state.set_delete_settings(settings.into());
                state.set_delete_other(other.into());
                state.set_delete_total(total.into());
                state.set_delete_open_now(open_now);
                state.set_delete_running(running);
                ui.global::<crate::GlobalAppState>()
                    .set_active_dialog(crate::DialogType::DeleteAnalysis);
            });
        });
    }

    /// Deletes the analysis the question is about - its results closed
    /// first when they are open. The question closes when it is gone, or
    /// says what was left.
    fn delete_pending(self: &Arc<Self>, open: Option<String>) {
        let Some((folder, open_now)) = self.pending_delete.lock().expect("Poisoned").clone() else {
            return;
        };
        if self.running_analysis.lock().expect("Poisoned").as_deref() == Some(folder.dir.as_path())
        {
            return;
        }
        self.set_delete_busy(true);
        let manager = self.clone();
        let delete = move || {
            let backend = manager.app_state.backend();
            let result = delete_analysis(backend.files(), &folder);
            {
                let mut last = manager.last_analysis_results.lock().expect("Poisoned");
                if last.as_deref().and_then(Path::parent) == Some(folder.dir.as_path()) {
                    *last = None;
                }
            }
            match &result {
                Ok(()) => log::info!("Deleted the analysis {:?}", folder.dir),
                Err(e) => warn!("Deleting the analysis {:?}: {e}", folder.dir),
            }
            // The deleted one isn't selected any more.
            let open = open.filter(|_| !open_now);
            manager.after_delete(result, open);
        };
        match open_now {
            true => self.results_state_controller.close_database_then(delete),
            false => {
                crate::helper::ui_thread::spawn(delete);
            }
        }
    }

    /// The list after deleting: the question closed (or saying what was
    /// left), `open` (the analysis open in Results, if another one) still
    /// selected.
    fn after_delete(&self, result: Result<(), String>, open: Option<String>) {
        let ui_handle = self.ui.clone();
        self.sync_results_files_to_slint();
        let _ = crate::helper::ui_thread::invoke_from_event_loop(move || {
            let Some(ui) = ui_handle.upgrade() else {
                return;
            };
            let state = ui.global::<ResultsListState>();
            let list = state.get_results_list();
            let index = open
                .and_then(|open| {
                    (0..slint::Model::row_count(&list))
                        .find(|&i| slint::Model::row_data(&list, i).is_some_and(|r| r.path == open))
                })
                .map_or(-1, |i| i as i32);
            state.set_selected_result_index(index);
            state.set_delete_busy(false);
            match result {
                Ok(()) => {
                    state.set_delete_error("".into());
                    ui.global::<crate::GlobalAppState>()
                        .set_active_dialog(crate::DialogType::None);
                }
                Err(e) => state.set_delete_error(e.into()),
            }
        });
    }

    fn set_delete_busy(&self, busy: bool) {
        let ui_handle = self.ui.clone();
        let _ = crate::helper::ui_thread::invoke_from_event_loop(move || {
            if let Some(ui) = ui_handle.upgrade() {
                ui.global::<ResultsListState>().set_delete_busy(busy);
            }
        });
    }

    fn open_results_folder(&self) {
        let project = self.app_state.get_project();
        let Some(current_project) = &project.tmp_settings.current_project else {
            return;
        };
        let Some(project_dir) = current_project.parent() else {
            return;
        };
        let results_dir = project_dir.join("results");
        drop(project);

        let path = if results_dir.exists() || self.app_state.backend().is_remote() {
            results_dir
        } else {
            // Fall back to the project directory itself
            results_dir
                .parent()
                .map(|p| p.to_path_buf())
                .unwrap_or(results_dir)
        };
        self.open_folder(path);
    }

    /// Shows `dir` - in the file manager, or for results on a server in the
    /// app's file browser (picking a results file opens it).
    fn open_folder(&self, dir: PathBuf) {
        if self.app_state.backend().is_remote() {
            let request = crate::FileRequest::open_file("Results")
                .filter("Results", &[RESULTS_FILE_EXTENSION])
                .start_in(dir);
            let table = self.results_state_controller.clone();
            self.app_state.file_browser.open(request, move |path| {
                if let Some(path) = path {
                    table.open_database(path);
                }
            });
            return;
        }

        #[cfg(target_os = "linux")]
        let _ = std::process::Command::new("xdg-open").arg(&dir).spawn();
        #[cfg(target_os = "macos")]
        let _ = std::process::Command::new("open").arg(&dir).spawn();
        #[cfg(target_os = "windows")]
        let _ = std::process::Command::new("explorer").arg(&dir).spawn();
    }
}

/// The results file of the analysis open in Results (`None`: none).
fn open_results_path(ui: &AppWindow) -> Option<String> {
    let state = ui.global::<ResultsListState>();
    let index = usize::try_from(state.get_selected_result_index()).ok()?;
    slint::Model::row_data(&state.get_results_list(), index).map(|r| r.path.to_string())
}

/// `dir` below the project's folder, as the question shows it
/// ("results/20261023T091000Z__serum/").
fn relative_folder(project_dir: &Path, dir: &Path) -> String {
    let shown = dir.strip_prefix(project_dir).unwrap_or(dir);
    format!("{}/", shown.to_string_lossy().replace('\\', "/"))
}

/// The results database an analysis wrote into `dir` (its own results
/// folder: `<job name>.evadb`), through the backend's file system.
fn results_database_in(files: &dyn FileSystem, dir: &Path) -> Option<PathBuf> {
    let entries = match files.list_dir(dir) {
        Ok(entries) => entries,
        Err(e) => {
            warn!("Could not read results directory {dir:?}: {e}");
            return None;
        }
    };
    entries
        .into_iter()
        .filter(|entry| !entry.is_dir)
        .map(|entry| entry.path)
        .find(|path| path.extension().and_then(|e| e.to_str()) == Some(RESULTS_FILE_EXTENSION))
}

/// Recursively walks `dir` through the backend's file system, appending
/// every file whose extension matches [`RESULTS_FILE_EXTENSION`] to `items`
/// together with its modification time (seconds since the Unix epoch).
fn collect_results_files(
    files: &dyn FileSystem,
    dir: &Path,
    items: &mut Vec<(i64, ResultItemData)>,
) {
    let entries = match files.list_dir(dir) {
        Ok(entries) => entries,
        Err(e) => {
            warn!("Could not read results directory {:?}: {}", dir, e);
            return;
        }
    };

    for entry in entries {
        let path = entry.path;

        if entry.is_dir {
            collect_results_files(files, &path, items);
            continue;
        }

        if path.extension().and_then(|e| e.to_str()) != Some(RESULTS_FILE_EXTENSION) {
            continue;
        }

        let mtime = entry.modified.unwrap_or(0);
        let modified = std::time::UNIX_EPOCH + std::time::Duration::from_secs(mtime.max(0) as u64);
        let (file_size, modified) = (format_file_size(entry.size), format_modified_time(modified));

        let name = extract_name_from_path(&path).unwrap_or("-");
        items.push((
            mtime,
            ResultItemData {
                name: name.into(),
                path: path.to_string_lossy().to_string().into(),
                file_size: file_size.into(),
                modified: modified.into(),
            },
        ));
    }
}

fn format_file_size(bytes: u64) -> String {
    if bytes < 1_024 {
        format!("{} B", bytes)
    } else if bytes < 1_024 * 1_024 {
        format!("{} KB", bytes / 1_024)
    } else {
        format!("{:.1} MB", bytes as f64 / (1_024.0 * 1_024.0))
    }
}

fn format_modified_time(time: std::time::SystemTime) -> String {
    let Ok(duration) = time.elapsed() else {
        return String::new();
    };
    let secs = duration.as_secs();
    if secs < 60 {
        "just now".to_string()
    } else if secs < 3_600 {
        format!("{} min ago", secs / 60)
    } else if secs < 86_400 {
        format!("{} hr ago", secs / 3_600)
    } else {
        format!("{} days ago", secs / 86_400)
    }
}

fn extract_name_from_path(path: &PathBuf) -> Option<&str> {
    // Get just the file name element without the directories
    // (e.g., "/path/to/file.csv" -> "file.csv")
    let file_name = path.file_name()?.to_str()?;

    // Strip the extension (e.g., "file.csv" -> "file")
    let file_stem = Path::new(file_name).file_stem()?.to_str()?;

    // Split at the "__" and return everything to the right
    match file_stem.split_once("__") {
        Some((_, name_part)) if !name_part.is_empty() => Some(name_part),
        _ => Some(file_stem),
    }
}

#[cfg(test)]
mod tests {
    use evanalyzer_app::{fs::LocalFileSystem, project::ProjectWithRuntime};

    use crate::editor::{
        histogram_controller::HistogramController, image_meta_controller::ImageMetaController,
        images_list_controller::ImagesListController, object_list_controller::ObjectListController,
        test_support::test_ui_state_with_project, viewport_controller::ViewportController,
    };

    use super::*;

    // -- format_file_size ----------------------------------------------------

    #[test]
    fn format_file_size_below_1024_bytes_uses_bytes() {
        assert_eq!(format_file_size(0), "0 B");
        assert_eq!(format_file_size(1023), "1023 B");
    }

    #[test]
    fn format_file_size_at_1024_bytes_switches_to_kilobytes() {
        assert_eq!(format_file_size(1_024), "1 KB");
        assert_eq!(format_file_size(2_048), "2 KB");
    }

    #[test]
    fn format_file_size_just_below_1_mb_stays_in_kilobytes() {
        assert_eq!(format_file_size(1_024 * 1_024 - 1), "1023 KB");
    }

    #[test]
    fn format_file_size_at_1_mb_switches_to_megabytes_with_one_decimal() {
        assert_eq!(format_file_size(1_024 * 1_024), "1.0 MB");
        assert_eq!(format_file_size(1_024 * 1_024 * 3 / 2), "1.5 MB");
    }

    // -- format_modified_time -------------------------------------------------

    #[test]
    fn format_modified_time_just_now_for_a_few_seconds_ago() {
        let t = std::time::SystemTime::now() - std::time::Duration::from_secs(10);
        assert_eq!(format_modified_time(t), "just now");
    }

    #[test]
    fn format_modified_time_minutes_ago_boundary() {
        let t = std::time::SystemTime::now() - std::time::Duration::from_secs(60);
        assert_eq!(format_modified_time(t), "1 min ago");
        let t = std::time::SystemTime::now() - std::time::Duration::from_secs(59);
        assert_eq!(format_modified_time(t), "just now");
    }

    #[test]
    fn format_modified_time_hours_ago_boundary() {
        let t = std::time::SystemTime::now() - std::time::Duration::from_secs(3_600);
        assert_eq!(format_modified_time(t), "1 hr ago");
        let t = std::time::SystemTime::now() - std::time::Duration::from_secs(3_599);
        assert_eq!(format_modified_time(t), "59 min ago");
    }

    #[test]
    fn format_modified_time_days_ago_boundary() {
        let t = std::time::SystemTime::now() - std::time::Duration::from_secs(86_400);
        assert_eq!(format_modified_time(t), "1 days ago");
        let t = std::time::SystemTime::now() - std::time::Duration::from_secs(86_399);
        assert_eq!(format_modified_time(t), "23 hr ago");
    }

    #[test]
    fn format_modified_time_in_the_future_is_empty() {
        // `SystemTime::elapsed()` errors when `time` is later than "now" -
        // must not panic, just report nothing.
        let t = std::time::SystemTime::now() + std::time::Duration::from_secs(3_600);
        assert_eq!(format_modified_time(t), "");
    }

    // -- extract_name_from_path ------------------------------------------------

    #[test]
    fn extract_name_from_path_strips_directories_and_extension() {
        let path = PathBuf::from("/a/b/file.evadb");
        assert_eq!(extract_name_from_path(&path), Some("file"));
    }

    #[test]
    fn extract_name_from_path_returns_the_part_after_the_last_double_underscore() {
        let path = PathBuf::from("/a/2026-01-01__My Project.evadb");
        assert_eq!(extract_name_from_path(&path), Some("My Project"));
    }

    #[test]
    fn extract_name_from_path_falls_back_to_the_full_stem_when_the_suffix_after_double_underscore_is_empty()
     {
        let path = PathBuf::from("/a/2026-01-01__.evadb");
        assert_eq!(extract_name_from_path(&path), Some("2026-01-01__"));
    }

    #[test]
    fn extract_name_from_path_without_a_file_name_returns_none() {
        // `Path::file_name()` returns `None` for a path ending in `..` (or
        // the root path) - there's no filename component to extract.
        let path = PathBuf::from("/a/..");
        assert_eq!(extract_name_from_path(&path), None);
    }

    // -- collect_results_files -------------------------------------------------

    fn temp_subdir(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "evanalyzer_results_list_controller_test_{name}_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn collect_results_files_finds_only_matching_extension_recursively() {
        let dir = temp_subdir("matching_ext");
        std::fs::write(dir.join("keep.evadb"), b"x").unwrap();
        std::fs::write(dir.join("skip.txt"), b"x").unwrap();
        let nested = dir.join("nested");
        std::fs::create_dir_all(&nested).unwrap();
        std::fs::write(nested.join("also_keep.evadb"), b"x").unwrap();

        let mut items = Vec::new();
        collect_results_files(&LocalFileSystem::default(), &dir, &mut items);

        let names: std::collections::HashSet<String> =
            items.iter().map(|(_, d)| d.name.to_string()).collect();
        assert_eq!(items.len(), 2);
        assert!(names.contains("keep"));
        assert!(names.contains("also_keep"));

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn collect_results_files_on_a_nonexistent_directory_leaves_items_empty() {
        let dir = std::env::temp_dir().join("evanalyzer_results_list_controller_does_not_exist");
        let mut items = Vec::new();
        collect_results_files(&LocalFileSystem::default(), &dir, &mut items);
        assert!(items.is_empty());
    }

    #[test]
    fn open_results_folder_test() {
        let ui = slint::Weak::default();
        let ui_state: Arc<UiState> = test_ui_state_with_project(ProjectWithRuntime::default());

        let viewport_controller = Arc::new(ViewportController::new(
            ui.clone(),
            ui_state.clone(),
            Arc::new(crate::editor::viewport_cache::ViewportCache::new(
                ui_state.clone(),
            )),
        ));
        let object_list_controller = Arc::new(ObjectListController::new(
            ui.clone(),
            ui_state.clone(),
            viewport_controller.clone(),
        ));

        let image_list_controller = Arc::new(ImagesListController::new(
            ui.clone(),
            ui_state.clone(),
            viewport_controller.clone(),
            Arc::new(HistogramController::new(
                ui.clone(),
                ui_state.clone(),
                viewport_controller.clone(),
            )),
            Arc::new(ImageMetaController::new(
                ui.clone(),
                ui_state.clone(),
                viewport_controller.clone(),
            )),
            object_list_controller.clone(),
            Arc::new(
                crate::editor::classification_controller::ClassificationController::new(
                    ui.clone(),
                    ui_state.clone(),
                    object_list_controller.clone(),
                    viewport_controller.clone(),
                ),
            ),
        ));

        let results_table_controller = Arc::new(ResultsStateController::new(
            ui.clone(),
            ui_state.clone(),
            image_list_controller.clone(),
        ));

        let results_list_controller = Arc::new(ResultsListController::new(
            ui.clone(),
            ui_state.clone(),
            results_table_controller,
        ));

        results_list_controller.open_results_folder();
    }
}
