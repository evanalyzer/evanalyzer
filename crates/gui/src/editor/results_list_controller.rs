use crate::AppWindow;
use crate::editor::results_table_controller::ResultsTableController;
use crate::{ResultItemData, ResultsListState, UiState};
use evanalyzer_cfg::RESULTS_FILE_EXTENSION;
use log::warn;
use slint::ComponentHandle;
use std::path::{Path, PathBuf};
use std::sync::Arc;

pub struct ResultsListController {
    pub(crate) ui: slint::Weak<AppWindow>,
    pub(crate) app_state: Arc<UiState>,
    pub(crate) results_table_controller: Arc<ResultsTableController>,
}

impl ResultsListController {
    pub fn new(
        ui: slint::Weak<AppWindow>,
        app_state: Arc<UiState>,
        results_table_controller: Arc<ResultsTableController>,
    ) -> Self {
        Self {
            ui,
            app_state: app_state.clone(),
            results_table_controller,
        }
    }

    pub fn attach_callbacks(self: &Arc<Self>) {
        let ui_handle = self.ui.clone();
        if let Some(ui) = ui_handle.upgrade() {
            let manager = self.clone();
            ui.global::<ResultsListState>().on_refresh_clicked(move || {
                manager.sync_results_files_to_slint();
            });

            let table = self.results_table_controller.clone();
            ui.global::<ResultsListState>()
                .on_result_selected(move |path| {
                    table.load_from_file(PathBuf::from(path.as_str()));
                });

            let manager = self.clone();
            ui.global::<ResultsListState>()
                .on_open_folder_clicked(move || {
                    manager.open_results_folder();
                });
        }
    }

    /// Recursively scans `<project_dir>/results/` for `*.RESULTS_FILE_EXTENSION`
    /// files and pushes them to the Slint results list, sorted by modification
    /// time (newest first).
    pub fn sync_results_files_to_slint(&self) {
        let project = self.app_state.get_project();

        let Some(current_project) = &project.tmp_settings.current_project else {
            warn!("No project open cannot scan results folder");
            return;
        };
        let Some(project_dir) = current_project.parent() else {
            return;
        };
        let results_dir = project_dir.join("results");

        // Drop the read lock before doing file I/O
        drop(project);

        let mut items: Vec<(std::time::SystemTime, ResultItemData)> = Vec::new();

        if results_dir.exists() {
            collect_results_files(&results_dir, &mut items);
        }

        // Newest first
        items.sort_by(|a, b| b.0.cmp(&a.0));
        let slint_items: Vec<ResultItemData> = items.into_iter().map(|(_, d)| d).collect();

        let ui_handle = self.ui.clone();
        let _ = slint::invoke_from_event_loop(move || {
            if let Some(ui) = ui_handle.upgrade() {
                ui.global::<ResultsListState>()
                    .set_results_list(slint::ModelRc::new(slint::VecModel::from(slint_items)));
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

        let path = if results_dir.exists() {
            results_dir
        } else {
            // Fall back to the project directory itself
            match std::fs::metadata(&results_dir.parent().unwrap_or(&results_dir)) {
                _ => results_dir
                    .parent()
                    .map(|p| p.to_path_buf())
                    .unwrap_or(results_dir),
            }
        };

        #[cfg(target_os = "linux")]
        let _ = std::process::Command::new("xdg-open").arg(&path).spawn();
        #[cfg(target_os = "macos")]
        let _ = std::process::Command::new("open").arg(&path).spawn();
        #[cfg(target_os = "windows")]
        let _ = std::process::Command::new("explorer").arg(&path).spawn();
    }
}

/// Recursively walks `dir`, appending every file whose extension matches
/// [`RESULTS_FILE_EXTENSION`] to `items` together with its modification time.
fn collect_results_files(dir: &Path, items: &mut Vec<(std::time::SystemTime, ResultItemData)>) {
    let entries = match std::fs::read_dir(dir) {
        Ok(entries) => entries,
        Err(e) => {
            warn!("Could not read results directory {:?}: {}", dir, e);
            return;
        }
    };

    for entry in entries.flatten() {
        let path = entry.path();

        if path.is_dir() {
            collect_results_files(&path, items);
            continue;
        }

        if path.extension().and_then(|e| e.to_str()) != Some(RESULTS_FILE_EXTENSION) {
            continue;
        }

        let (file_size, modified, mtime) = match std::fs::metadata(&path) {
            Ok(meta) => {
                let mtime = meta.modified().unwrap_or(std::time::UNIX_EPOCH);
                (
                    format_file_size(meta.len()),
                    format_modified_time(mtime),
                    mtime,
                )
            }
            Err(_) => (String::new(), String::new(), std::time::UNIX_EPOCH),
        };

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
        collect_results_files(&dir, &mut items);

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
        collect_results_files(&dir, &mut items);
        assert!(items.is_empty());
    }
}
