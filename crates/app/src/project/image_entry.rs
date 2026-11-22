//! # image_entry_ext
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-10-10
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.

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
        let series = evanalyzer_core::active_series(self, project_series);
        // The project's series that this image lacks (e.g. series 4 of an
        // image with 2): its first one is shown, previewed and trained on.
        // An analysis still refuses that (`evanalyzer_core::check_series`).
        match self.series.keys().next() {
            Some(first) if !self.series.contains_key(&series) => *first,
            _ => series,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use evanalyzer_cfg::settings::images_settings::ImageEntry;

    #[test]
    fn a_project_series_the_image_lacks_shows_its_first() {
        let image = ImageEntry {
            series: [(0, Default::default()), (1, Default::default())].into(),
            ..Default::default()
        };
        assert_eq!(image.series_for(Some(1)), 1);
        assert_eq!(image.series_for(Some(3)), 0, "only 2 series");
        let unread = ImageEntry::default();
        assert_eq!(unread.series_for(Some(3)), 3, "series not read yet");
    }
}
