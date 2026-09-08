use crate::AppWindow;
use crate::{ResultItemData, ResultsListState, UiState};
use evanalyzer_cfg::RESULTS_FILE_EXTENSION;
use evanalyzer_gui_slint::ResultsWindow;
use log::warn;
use slint::ComponentHandle;
use std::path::{Path, PathBuf};
use std::sync::Arc;

pub struct ResultsStateController {
    pub(crate) ui: slint::Weak<ResultsWindow>,
    pub(crate) app_state: Arc<UiState>,
}

impl ResultsStateController {
    pub fn new(ui: slint::Weak<ResultsWindow>, app_state: Arc<UiState>) -> Self {
        Self {
            ui,
            app_state: app_state.clone(),
        }
    }

    pub fn attach_callbacks(self: &Arc<Self>) {
        let ui_handle = self.ui.clone();
        if let Some(ui) = ui_handle.upgrade() {
            let manager = self.clone();
            ui.global::<ResultsListState>()
                .on_refresh_clicked(move || {});

            let manager = self.clone();
            ui.global::<ResultsListState>()
                .on_open_folder_clicked(move || {});
        }
    }

    pub fn load_from_file(&self, path: PathBuf) {}
}
