//! # start_page_controller
//!
//! The start page (board AA): shown while nothing is open - at startup
//! (unless the last project opens right away) and after File › New. Its
//! ways in are the existing ones (File › Open, the images folder, a
//! template, connect). The controller answers the page; the functions
//! below keep the recent projects and say when the page shows - the
//! project and image controllers call them as projects open and save.
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-10-17
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.

use crate::editor::project_controller::ProjectController;
use crate::{AppWindow, LayoutState, RecentProjectRow, StartState, UiState, WorkspaceMode};
use evanalyzer_app::project::ImageListExt;
use evanalyzer_app::user::RecentProject;
use slint::{ComponentHandle, ModelRc, VecModel};
use std::path::PathBuf;
use std::rc::Rc;
use std::sync::Arc;

pub struct StartPageController {
    ui: slint::Weak<AppWindow>,
    app_state: Arc<UiState>,
    project_controller: Arc<ProjectController>,
}

impl StartPageController {
    pub fn new(
        ui: slint::Weak<AppWindow>,
        app_state: Arc<UiState>,
        project_controller: Arc<ProjectController>,
    ) -> Self {
        Self {
            ui,
            app_state,
            project_controller,
        }
    }

    /// At startup without a project to open: the last project (if so set
    /// and it is still there), else the start page.
    pub fn show_start(&self) {
        let backend = self.app_state.backend();
        self.show_start_with(
            move |path| matches!(backend.files().stat(path), Ok(Some(_))),
            LAST_PROJECT_WAIT,
        );
    }

    /// [`Self::show_start`] with `exists` telling whether the last project
    /// is still there. It is asked on a thread of its own and waited for at
    /// most `wait` - the window shows only after this, and a network share
    /// that is gone can take many seconds to answer. Answered later: the
    /// start page meanwhile, and the project opened when the answer comes
    /// if the start page is still untouched (as if its entry was clicked).
    pub(crate) fn show_start_with(
        &self,
        exists: impl FnOnce(&std::path::Path) -> bool + Send + 'static,
        wait: std::time::Duration,
    ) {
        let settings = self.app_state.load_app_settings();
        let last = settings
            .recent_projects
            .first()
            .map(|p| p.path.clone())
            .filter(|_| settings.open_last_project);
        // The EVA menu's RECENT either way.
        push_recent(&self.ui, &self.app_state);
        let Some(path) = last else {
            set_active(&self.ui, true, None);
            return;
        };
        let (answer, answered) = std::sync::mpsc::channel();
        let late = Arc::new(std::sync::Mutex::new(None::<LateOpen>));
        let late_in = Arc::clone(&late);
        let asked = path.clone();
        crate::helper::ui_thread::spawn(move || {
            let there = exists(&asked);
            // Waited for: the answer goes back. Too late: opened from here.
            // Under the lock, so the wait can't end between the two.
            let mut waiting = late_in.lock().expect("Poisoned");
            match waiting.take() {
                None => {
                    let _ = answer.send(there);
                }
                Some(late) if there => late.open(asked),
                Some(_) => {}
            }
        });
        match answered.recv_timeout(wait) {
            Ok(true) => Arc::clone(&self.project_controller).open_new_project(&path),
            Ok(false) => set_active(&self.ui, true, None),
            Err(_) => {
                let mut waiting = late.lock().expect("Poisoned");
                match answered.try_recv() {
                    // Answered just now.
                    Ok(true) => {
                        drop(waiting);
                        Arc::clone(&self.project_controller).open_new_project(&path);
                    }
                    Ok(false) => set_active(&self.ui, true, None),
                    Err(_) => {
                        log::info!(
                            "The last project {path:?} doesn't answer yet - the start page meanwhile"
                        );
                        *waiting = Some(LateOpen {
                            ui: self.ui.clone(),
                            app_state: Arc::clone(&self.app_state),
                            project_controller: Arc::clone(&self.project_controller),
                        });
                        set_active(&self.ui, true, None);
                    }
                }
            }
        }
    }

    /// The start page's ways in - the existing ones.
    pub fn attach_callbacks(self: &Arc<Self>) {
        let Some(ui) = self.ui.upgrade() else {
            return;
        };
        let start = ui.global::<StartState>();
        let ui_weak = self.ui.clone();
        start.on_open_images_folder(move || {
            if let Some(ui) = ui_weak.upgrade() {
                ui.global::<crate::ImagesListState>()
                    .invoke_open_images_folder_clicked();
            }
        });
        let ui_weak = self.ui.clone();
        start.on_open_image(move || {
            if let Some(ui) = ui_weak.upgrade() {
                ui.global::<crate::ToolbarState>()
                    .invoke_open_image_clicked();
            }
        });
        let ui_weak = self.ui.clone();
        start.on_open_project(move || {
            if let Some(ui) = ui_weak.upgrade() {
                ui.global::<crate::ToolbarState>()
                    .invoke_open_file_clicked();
            }
        });
        let ui_weak = self.ui.clone();
        // Images first, as for any project: where its images come in.
        start.on_empty_project(move || {
            set_active(&ui_weak, false, Some(crate::WorkspaceMode::Images));
        });
        let ui_weak = self.ui.clone();
        start.on_from_template(move || {
            if let Some(ui) = ui_weak.upgrade() {
                ui.global::<crate::ToolbarState>()
                    .invoke_new_from_project_template_clicked();
            }
        });
        let ui_weak = self.ui.clone();
        start.on_connect(move || {
            if let Some(ui) = ui_weak.upgrade() {
                ui.global::<crate::ToolbarState>().invoke_connect_clicked();
            }
        });
        let manager = Arc::clone(self);
        start.on_open_recent(move |index| {
            if let Some(path) = recent_path(&manager.app_state, index) {
                manager.project_controller.open_project_guarded(path);
            }
        });
        // Moved: picked again, the old entry goes.
        let manager = Arc::clone(self);
        start.on_locate_recent(move |index| {
            let Some(old) = recent_path(&manager.app_state, index) else {
                return;
            };
            let request = crate::FileRequest::open_file("Locate the project")
                .filter("Project files", &[evanalyzer_cfg::PROJECT_FILE_EXTENSIONS]);
            let in_thread = Arc::clone(&manager);
            manager.app_state.file_browser.open(request, move |path| {
                let Some(path) = path else {
                    return;
                };
                in_thread
                    .app_state
                    .update_app_settings(|settings| settings.forget_project(&old));
                in_thread.project_controller.open_project_guarded(path);
            });
        });
        let manager = Arc::clone(self);
        start.on_forget_recent(move |index| {
            if let Some(path) = recent_path(&manager.app_state, index) {
                manager
                    .app_state
                    .update_app_settings(|settings| settings.forget_project(&path));
                push_recent(&manager.ui, &manager.app_state);
            }
        });
        let manager = Arc::clone(self);
        start.on_clear_recent(move || {
            manager
                .app_state
                .update_app_settings(|settings| settings.recent_projects.clear());
            push_recent(&manager.ui, &manager.app_state);
        });
        let manager = Arc::clone(self);
        start.on_open_last_changed(move |on| {
            manager
                .app_state
                .update_app_settings(|settings| settings.open_last_project = on);
        });
    }
}

/// How long startup waits for the last project to answer whether it is
/// still there - a local disk answers at once, a share that is gone may
/// take many seconds.
const LAST_PROJECT_WAIT: std::time::Duration = std::time::Duration::from_millis(1500);

/// The last project, opened once its late answer comes - if nothing
/// happened on the start page meanwhile.
struct LateOpen {
    ui: slint::Weak<AppWindow>,
    app_state: Arc<UiState>,
    project_controller: Arc<ProjectController>,
}

impl LateOpen {
    /// Opens `path` on the UI thread - only while the start page shows, no
    /// dialog is open and the project is the untouched blank one: anything
    /// the user did meanwhile wins.
    fn open(self, path: PathBuf) {
        crate::helper::ui_thread::invoke_from_event_loop(move || {
            let Some(ui) = self.ui.upgrade() else {
                return;
            };
            let untouched = ui.global::<StartState>().get_active()
                && ui.global::<crate::GlobalAppState>().get_active_dialog()
                    == crate::DialogType::None
                && !ui.global::<crate::FileBrowserState>().get_visible()
                && !self.app_state.is_dirty()
                && self
                    .app_state
                    .get_project()
                    .tmp_settings
                    .current_project
                    .is_none();
            if untouched {
                self.project_controller.open_new_project(&path);
            } else {
                log::info!(
                    "The last project {path:?} answered, but the start page was used - not opened"
                );
            }
        })
        .ok();
    }
}

/// Shows the start page (`true`) or leaves it, then shows `mode` (the next
/// step: Images after opening something). Any thread.
pub(crate) fn set_active(ui: &slint::Weak<AppWindow>, active: bool, mode: Option<WorkspaceMode>) {
    let ui = ui.clone();
    crate::helper::ui_thread::invoke_from_event_loop(move || {
        if let Some(ui) = ui.upgrade() {
            ui.global::<StartState>().set_active(active);
            if let Some(mode) = mode {
                ui.global::<LayoutState>().set_mode(mode);
            }
        }
    })
    .ok();
}

/// The project open now goes first in the recent projects (once it has a
/// file: after opening or saving it).
pub(crate) fn remember_current_project(app_state: &Arc<UiState>) {
    let entry = {
        let project = app_state.get_project();
        let Some(path) = project.tmp_settings.current_project.clone() else {
            return;
        };
        let name = if project.meta.name.trim().is_empty() {
            path.file_stem()
                .map(|s| s.to_string_lossy().into_owned())
                .unwrap_or_else(|| path.to_string_lossy().into_owned())
        } else {
            project.meta.name.clone()
        };
        RecentProject {
            path,
            name,
            images: project.images.list.len() as u32,
            pipelines: project.pipelines.len() as u32,
            opened_at: chrono::Utc::now().timestamp(),
        }
    };
    app_state.update_app_settings(|settings| settings.remember_project(entry));
    // The EVA menu's RECENT and the page show it.
    push_recent(&app_state.ui_handle.clone(), app_state);
}

/// The recent projects and the "open the last one" switch on the page;
/// whether each file is still there is asked off the UI thread (a server
/// in remote mode).
pub(crate) fn push_recent(ui: &slint::Weak<AppWindow>, app_state: &Arc<UiState>) {
    let ui = ui.clone();
    let app_state = app_state.clone();
    crate::helper::ui_thread::spawn(move || {
        let settings = app_state.load_app_settings();
        let backend = app_state.backend();
        let files = backend.files();
        let now = chrono::Local::now();
        let rows: Vec<(RecentProject, bool)> = settings
            .recent_projects
            .iter()
            .map(|p| (p.clone(), !matches!(files.stat(&p.path), Ok(Some(_)))))
            .collect();
        let open_last = settings.open_last_project;
        crate::helper::ui_thread::invoke_from_event_loop(move || {
            if let Some(ui) = ui.upgrade() {
                let rows: Vec<RecentProjectRow> = rows
                    .iter()
                    .map(|(p, missing)| row(p, *missing, &now))
                    .collect();
                let state = ui.global::<StartState>();
                state.set_recent(ModelRc::from(Rc::new(VecModel::from(rows))));
                state.set_open_last(open_last);
            }
        })
        .ok();
    });
}

/// The recent project at `index` (as listed).
pub(crate) fn recent_path(app_state: &UiState, index: i32) -> Option<PathBuf> {
    app_state
        .load_app_settings()
        .recent_projects
        .get(usize::try_from(index).ok()?)
        .map(|p| p.path.clone())
}

fn row(
    project: &RecentProject,
    missing: bool,
    now: &chrono::DateTime<chrono::Local>,
) -> RecentProjectRow {
    let plural = |n: u32, one: &str, many: &str| format!("{n} {}", if n == 1 { one } else { many });
    RecentProjectRow {
        name: project.name.clone().into(),
        path: project.path.to_string_lossy().into_owned().into(),
        meta: if missing {
            String::new()
        } else {
            format!(
                "{} · {}",
                plural(project.images, "image", "images"),
                plural(project.pipelines, "pipeline", "pipelines")
            )
        }
        .into(),
        when: when_text(project.opened_at, now).into(),
        missing,
    }
}

/// "Today 09:12", "Yesterday", "14 Oct", "14 Oct 2025" (another year).
fn when_text(opened_at: i64, now: &chrono::DateTime<chrono::Local>) -> String {
    use chrono::{Datelike, TimeZone};
    let Some(then) = chrono::Local.timestamp_opt(opened_at, 0).single() else {
        return String::new();
    };
    let days = (now.date_naive() - then.date_naive()).num_days();
    match days {
        0 => then.format("Today %H:%M").to_string(),
        1 => "Yesterday".to_string(),
        _ if then.year() == now.year() => then.format("%-d %b").to_string(),
        _ => then.format("%-d %b %Y").to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::TimeZone;

    #[test]
    fn when_says_today_yesterday_or_the_date() {
        let now = chrono::Local
            .with_ymd_and_hms(2026, 10, 17, 15, 0, 0)
            .unwrap();
        let at = |y, m, d, h, min| {
            chrono::Local
                .with_ymd_and_hms(y, m, d, h, min, 0)
                .unwrap()
                .timestamp()
        };
        assert_eq!(when_text(at(2026, 10, 17, 9, 12), &now), "Today 09:12");
        assert_eq!(when_text(at(2026, 10, 16, 23, 0), &now), "Yesterday");
        assert_eq!(when_text(at(2026, 10, 2, 8, 0), &now), "2 Oct");
        assert_eq!(when_text(at(2025, 10, 2, 8, 0), &now), "2 Oct 2025");
    }

    #[test]
    fn a_row_counts_images_and_pipelines_and_a_missing_one_says_nothing_of_them() {
        let now = chrono::Local::now();
        let project = RecentProject {
            path: PathBuf::from("/data/ev/ev.evaproj"),
            name: "EV plate".into(),
            images: 1,
            pipelines: 4,
            opened_at: now.timestamp(),
        };
        assert_eq!(row(&project, false, &now).meta, "1 image · 4 pipelines");
        let missing = row(&project, true, &now);
        assert!(missing.missing);
        assert_eq!(missing.meta, "");
    }
}
