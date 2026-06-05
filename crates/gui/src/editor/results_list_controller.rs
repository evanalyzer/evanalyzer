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

    /// Scans `<project_dir>/results/` for `*.RESULTS_FILE_EXTENSION` files and
    /// pushes them to the Slint results list, sorted by modification time (newest first).
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
            match std::fs::read_dir(&results_dir) {
                Ok(entries) => {
                    for entry in entries.flatten() {
                        let path = entry.path();
                        if path.extension().and_then(|e| e.to_str()) != Some(RESULTS_FILE_EXTENSION)
                        {
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

                        // let name = path
                        //     .file_name()
                        //     .and_then(|n| n.to_str())
                        //     .unwrap_or("")
                        //     .to_string();
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
                Err(e) => {
                    warn!("Could not read results directory {:?}: {}", results_dir, e);
                }
            }
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
    let (_, name_part) = file_stem.split_once("__")?;

    Some(name_part)
}
