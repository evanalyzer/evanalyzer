use evanalyzer_cfg::settings::images_settings::{GlobalImageSettings, ImageEntry};

/// Extension methods for [`ImageEntry`]: which of the image's series counts.
pub trait ImageEntryExt {
    /// The series that counts for this image: the project-wide choice if
    /// there is one, else the image's own, else its first series - see
    /// `evanalyzer_core::active_series`, which decides it.
    fn active_series(&self, global: &GlobalImageSettings) -> i32;

    /// [`Self::active_series`] for the project-wide choice `project_series`
    /// (`GlobalImageSettings::selected_series`) - for callers that can't
    /// borrow the global settings alongside the image.
    fn series_for(&self, project_series: Option<i32>) -> i32;
}

impl ImageEntryExt for ImageEntry {
    fn active_series(&self, global: &GlobalImageSettings) -> i32 {
        self.series_for(global.selected_series)
    }

    fn series_for(&self, project_series: Option<i32>) -> i32 {
        evanalyzer_core::active_series(self, project_series)
    }
}
