use crate::extensions::utils::is_in_root;
use log::{info, trace, warn};
use pathdiff::diff_paths;
use std::collections::BTreeMap;
use std::fs;
use std::path::Path;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::Mutex;
use std::time::Instant;

pub trait ProjectExt {
    fn get_z_stack(&self, project: &Project) -> ZStackSettings;
    fn get_t_stack(&self, project: &Project) -> TStackSettings;
}

impl ProjectExt for ImageEntry {
    /// Retrieves the active Z-stack settings (Local -> Global -> Default).
    fn get_z_stack(&self, project: &Project) -> ZStackSettings {
        if let Some(series_settings) = self.series.get(&self.selected_series) {
            if let Some(z_stack_settings) = &series_settings.z_stack {
                return z_stack_settings.clone();
            }
        }

        // use global settings
        return project
            .images
            .settings
            .read()
            .expect("Poised")
            .z_stack
            .clone()
            .unwrap_or_default();
    }

    /// Retrieves the active Z-stack settings (Local -> Global -> Default).
    fn get_t_stack(&self, project: &Project) -> Arc<TStackSettings> {
        if let Some(series_settings) = self.series.get(&self.selected_series) {
            if let Some(t_stack_settings) = &series_settings.t_stack {
                return t_stack_settings.clone();
            }
        }

        // use global settings
        return project
            .images
            .settings
            .read()
            .expect("Poised")
            .t_stack
            .clone()
            .unwrap_or_default();
    }
}
