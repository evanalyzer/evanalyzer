//! # trusted_scripts_controller
//!
//! Settings › Trusted scripts (design canvas "Settings dialog - project vs.
//! this computer, trusted scripts", board Scripts): the folders and scripts
//! the user trusts (`trust.json` in the user folder - the server's in
//! remote mode, where the scripts run). Revokes are kept until Apply;
//! Cancel forgets them. Whether each script still matches is read in the
//! background, so a slow share doesn't hold up the dialog.
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-11-25
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.

use crate::helper::ui_thread;
use crate::{AppWindow, TrustedFolderRow, TrustedScriptRow, TrustedScriptsState, UiState};
use evanalyzer_app::exports::{TrustedScriptStatus, trusted_script_status};
use evanalyzer_app::user::ScriptTrust;
use slint::{ComponentHandle, ModelRc, VecModel};
use std::collections::HashMap;
use std::path::Path;
use std::sync::{Arc, Mutex};

pub struct TrustedScriptsController {
    ui: slint::Weak<AppWindow>,
    app_state: Arc<UiState>,
    shown: Mutex<Shown>,
}

/// The list as shown: as saved, with the revokes not applied yet, and each
/// script's status once read (by its path).
#[derive(Default)]
struct Shown {
    saved: ScriptTrust,
    draft: ScriptTrust,
    status: HashMap<String, TrustedScriptStatus>,
    /// Counts the loads - a status read for an older one is dropped.
    load: u64,
}

impl TrustedScriptsController {
    pub fn new(ui: slint::Weak<AppWindow>, app_state: Arc<UiState>) -> Self {
        Self {
            ui,
            app_state,
            shown: Mutex::new(Shown::default()),
        }
    }

    pub fn attach_callbacks(self: &Arc<Self>) {
        let Some(ui) = self.ui.upgrade() else {
            return;
        };
        let state = ui.global::<TrustedScriptsState>();
        let this = self.clone();
        state.on_load(move || this.load());
        let this = self.clone();
        state.on_revoke_folder(move |index| {
            this.change(|shown| {
                if let Some(path) = shown
                    .draft
                    .folders
                    .get(index as usize)
                    .map(|f| f.path.clone())
                {
                    shown.draft.revoke_folder(&path);
                }
            })
        });
        let this = self.clone();
        state.on_revoke_script(move |index| {
            this.change(|shown| {
                if let Some(path) = shown
                    .draft
                    .scripts
                    .get(index as usize)
                    .map(|s| s.path.clone())
                {
                    shown.draft.revoke_script(&path);
                }
            })
        });
        let this = self.clone();
        state.on_remove_missing(move || {
            this.change(|shown| {
                let status = &shown.status;
                shown.draft.scripts.retain(|script| {
                    status.get(&script.path) != Some(&TrustedScriptStatus::Missing)
                });
            })
        });
        let this = self.clone();
        state.on_revoke_all(move || this.change(|shown| shown.draft = ScriptTrust::default()));
        let this = self.clone();
        state.on_apply(move || this.apply());
    }

    /// Reads the trust anew (the dialog opened): shown at once, the
    /// scripts' status follows.
    fn load(self: &Arc<Self>) {
        let backend = self.app_state.backend();
        let trust = backend.load_script_trust().unwrap_or_else(|e| {
            log::warn!("Could not read the trusted scripts: {e}");
            ScriptTrust::default()
        });
        let load = {
            let mut shown = self.shown.lock().expect("Poisoned");
            shown.load += 1;
            shown.saved = trust.clone();
            shown.draft = trust.clone();
            shown.status.clear();
            shown.load
        };
        if let Some(ui) = self.ui.upgrade() {
            ui.global::<TrustedScriptsState>().set_place(
                if backend.is_remote() {
                    "the server"
                } else {
                    "this computer"
                }
                .into(),
            );
        }
        self.push();
        if trust.scripts.is_empty() {
            return;
        }
        let this = self.clone();
        ui_thread::spawn(move || {
            let status: HashMap<String, TrustedScriptStatus> = trust
                .scripts
                .iter()
                .map(|script| {
                    (
                        script.path.clone(),
                        trusted_script_status(backend.files(), script),
                    )
                })
                .collect();
            ui_thread::invoke_from_event_loop(move || {
                {
                    let mut shown = this.shown.lock().expect("Poisoned");
                    if shown.load != load {
                        return;
                    }
                    shown.status = status;
                }
                this.push();
            })
            .ok();
        });
    }

    /// Changes the draft and shows it.
    fn change(&self, change: impl FnOnce(&mut Shown)) {
        change(&mut self.shown.lock().expect("Poisoned"));
        self.push();
    }

    /// Saves the revokes (Apply) - nothing when there are none.
    fn apply(&self) {
        let draft = {
            let shown = self.shown.lock().expect("Poisoned");
            if shown.draft == shown.saved {
                return;
            }
            shown.draft.clone()
        };
        match self.app_state.backend().save_script_trust(&draft) {
            Ok(()) => self.shown.lock().expect("Poisoned").saved = draft,
            Err(e) => log::warn!("Could not save the trusted scripts: {e}"),
        }
    }

    fn push(&self) {
        let Some(ui) = self.ui.upgrade() else {
            return;
        };
        let state = ui.global::<TrustedScriptsState>();
        let shown = self.shown.lock().expect("Poisoned");
        let folders: Vec<TrustedFolderRow> = shown
            .draft
            .folders
            .iter()
            .map(|folder| TrustedFolderRow {
                name: file_name(&folder.path).into(),
                path: folder.path.as_str().into(),
                detail: match since(&folder.since) {
                    day if day.is_empty() => folder.path.clone(),
                    day => format!("{day} · {}", folder.path),
                }
                .into(),
                network: is_network_path(&folder.path),
            })
            .collect();
        let scripts: Vec<TrustedScriptRow> = shown
            .draft
            .scripts
            .iter()
            .map(|script| TrustedScriptRow {
                name: file_name(&script.path).into(),
                path: script.path.as_str().into(),
                status: match shown.status.get(&script.path) {
                    Some(TrustedScriptStatus::Matches) => 0,
                    Some(TrustedScriptStatus::Changed) => 1,
                    Some(TrustedScriptStatus::Missing) => 2,
                    None => -1,
                },
            })
            .collect();
        state.set_folders(ModelRc::new(VecModel::from(folders)));
        state.set_scripts(ModelRc::new(VecModel::from(scripts)));
    }
}

/// The last part of `path` (the whole path if it has none).
fn file_name(path: &str) -> String {
    Path::new(path)
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_else(|| path.to_string())
}

/// "since 2026-11-25" ("" not known).
fn since(day: &str) -> String {
    if day.is_empty() {
        String::new()
    } else {
        format!("since {day}")
    }
}

/// A folder on a network share (`\\server\share`, `//server/share`) - what
/// others change there runs too. Mounted shares (`/mnt/...`) look local.
fn is_network_path(path: &str) -> bool {
    path.starts_with("\\\\") || path.starts_with("//")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::editor::test_support::*;
    use crate::helper::ui_thread::drain_ui_queue;
    use evanalyzer_app::exports::script_hash;
    use slint::Model;

    fn rows(ui: &AppWindow) -> (Vec<TrustedFolderRow>, Vec<TrustedScriptRow>) {
        let state = ui.global::<TrustedScriptsState>();
        (
            state.get_folders().iter().collect(),
            state.get_scripts().iter().collect(),
        )
    }

    /// The list shows what is trusted and each script's state; revokes wait
    /// for Apply - a reload without it (Cancel) shows them again.
    #[test]
    fn the_list_shows_the_trust_and_revokes_wait_for_apply() {
        let dir = tempfile::tempdir().unwrap();
        let ui = test_ui_window();
        let ui_state = ui_state_with_window(&ui, Default::default());
        let backend = ui_state.backend();
        let work = dir.path().join("work");
        std::fs::create_dir_all(&work).unwrap();
        let matching = work.join("a.R");
        let changed = work.join("b.R");
        std::fs::write(&matching, "a").unwrap();
        std::fs::write(&changed, "b2").unwrap();
        let mut trust = ScriptTrust::default();
        trust.trust_folder(&dir.path().join("project"), "2026-11-25");
        trust.allow_script(&matching, &script_hash(b"a"), "2026-11-25");
        trust.allow_script(&changed, &script_hash(b"b1"), "2026-11-25");
        trust.allow_script(&work.join("gone.R"), "x", "");
        backend.save_script_trust(&trust).unwrap();
        let trust = backend.load_script_trust().unwrap();

        let controller = Arc::new(TrustedScriptsController::new(
            ui.as_weak(),
            ui_state.clone(),
        ));
        controller.attach_callbacks();
        let state = ui.global::<TrustedScriptsState>();
        state.invoke_load();
        drain_ui_queue();
        let (folders, scripts) = rows(&ui);
        assert_eq!(folders.len(), 1);
        assert_eq!(folders[0].name, "project");
        assert!(folders[0].detail.starts_with("since 2026-11-25 · "));
        let status: Vec<(String, i32)> = scripts
            .iter()
            .map(|s| (s.name.to_string(), s.status))
            .collect();
        assert_eq!(
            status,
            [("a.R".into(), 0), ("b.R".into(), 1), ("gone.R".into(), 2)]
        );

        state.invoke_remove_missing();
        state.invoke_revoke_folder(0);
        let (folders, scripts) = rows(&ui);
        assert!(folders.is_empty());
        assert_eq!(scripts.len(), 2);
        assert_eq!(backend.load_script_trust().unwrap(), trust, "not saved yet");

        // Cancel: the dialog opens again with the saved list.
        state.invoke_load();
        drain_ui_queue();
        assert_eq!(rows(&ui).1.len(), 3);

        state.invoke_revoke_script(0);
        state.invoke_apply();
        let saved = backend.load_script_trust().unwrap();
        assert_eq!(saved.folders.len(), 1);
        assert_eq!(saved.scripts.len(), 2);
        assert!(!saved.allows(&matching, &script_hash(b"a")));

        state.invoke_revoke_all();
        state.invoke_apply();
        assert!(backend.load_script_trust().unwrap().is_empty());
    }

    #[test]
    fn only_shares_count_as_network_folders() {
        assert!(is_network_path("\\\\labserver\\screens"));
        assert!(is_network_path("//labserver/screens"));
        assert!(!is_network_path("/home/anna/projects"));
        assert!(!is_network_path("C:\\Users\\anna"));
    }
}
