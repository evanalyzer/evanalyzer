//! # undo_history
//!
//! The undo (and redo) steps: snapshots of the project's settings, oldest
//! first, kept within a RAM budget (`ViewportMemorySettings`'s undo history
//! setting) - but never fewer than `MIN_UNDO_STEPS`, and never more than
//! `MAX_STEPS`.
//!
//! A step's RAM is what it holds that the step after it doesn't share: the
//! image list is shared between snapshots until an edit changes it
//! (`ImageListExt`), the rest of the settings is copied per step.
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-10-27
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.

use evanalyzer_app::global::ViewportMemorySettings;
use evanalyzer_app::project::ImageListExt;
use evanalyzer_cfg::settings::project_settings::ProjectSettings;
use std::collections::VecDeque;

/// Steps kept at most, whatever the budget.
pub(crate) const MAX_STEPS: usize = 100;

/// RAM of a step's settings besides the image list (pipelines, classes,
/// views) - copied with every step.
const SETTINGS_BYTES: u64 = 64 * 1024;

struct Step {
    settings: ProjectSettings,
    /// What it holds that the step after it doesn't share - 0 while it is
    /// the newest (it shares everything with the project then).
    bytes: u64,
}

#[derive(Default)]
pub(crate) struct UndoHistory {
    steps: VecDeque<Step>,
}

impl UndoHistory {
    /// Adds `settings` as the newest step, then drops the oldest ones over
    /// [`MAX_STEPS`] or over `budget_bytes` (keeping the minimum).
    pub(crate) fn push(&mut self, settings: ProjectSettings, budget_bytes: u64) {
        if let Some(newest) = self.steps.back_mut() {
            newest.bytes = SETTINGS_BYTES
                + newest
                    .settings
                    .images
                    .list
                    .unshared_bytes(&settings.images.list);
        }
        self.steps.push_back(Step { settings, bytes: 0 });
        while self.steps.len() > MAX_STEPS
            || (self.steps.len() > ViewportMemorySettings::MIN_UNDO_STEPS
                && self.bytes() > budget_bytes)
        {
            self.steps.pop_front();
        }
    }

    /// Takes the newest step off.
    pub(crate) fn pop(&mut self) -> Option<ProjectSettings> {
        let step = self.steps.pop_back()?;
        if let Some(newest) = self.steps.back_mut() {
            // Newest again: what it shares is now the project's.
            newest.bytes = 0;
        }
        Some(step.settings)
    }

    pub(crate) fn clear(&mut self) {
        self.steps.clear();
    }

    #[cfg(test)]
    pub(crate) fn len(&self) -> usize {
        self.steps.len()
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.steps.is_empty()
    }

    #[cfg(test)]
    pub(crate) fn oldest(&self) -> Option<&ProjectSettings> {
        self.steps.front().map(|step| &step.settings)
    }

    #[cfg(test)]
    pub(crate) fn newest(&self) -> Option<&ProjectSettings> {
        self.steps.back().map(|step| &step.settings)
    }

    /// Roughly the RAM the steps hold.
    pub(crate) fn bytes(&self) -> u64 {
        self.steps.iter().map(|step| step.bytes).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use evanalyzer_cfg::settings::images_settings::{ImageEntry, ImageList};
    use std::path::PathBuf;

    fn project(images: usize) -> ProjectSettings {
        let mut settings = ProjectSettings::default();
        settings.images.list = ImageList::from_entries(
            (0..images).map(|i| (PathBuf::from(format!("{i}.tif")), ImageEntry::default())),
        );
        settings
    }

    /// An edit of one image: the step before holds the map's pointers and
    /// that entry on its own.
    fn edited(settings: &ProjectSettings, i: u64) -> ProjectSettings {
        let mut next = settings.clone();
        next.images
            .list
            .get_mut(&PathBuf::from("0.tif"))
            .unwrap()
            .file_size = i;
        next
    }

    /// Steps sharing everything cost only their settings; the newest
    /// nothing.
    #[test]
    fn a_step_costs_what_the_next_one_does_not_share() {
        let mut history = UndoHistory::default();
        let first = project(1000);
        history.push(first.clone(), u64::MAX);
        assert_eq!(history.bytes(), 0);
        history.push(first.clone(), u64::MAX);
        assert_eq!(history.bytes(), SETTINGS_BYTES);
        let second = edited(&first, 1);
        history.push(second, u64::MAX);
        assert!(history.bytes() > SETTINGS_BYTES + 1000 * 90);
        history.pop();
        assert_eq!(history.bytes(), SETTINGS_BYTES, "the newest costs nothing");
    }

    /// Over the budget the oldest steps go - but at least the minimum stays;
    /// never more than the maximum.
    #[test]
    fn the_budget_drops_the_oldest_steps_but_keeps_the_minimum() {
        let mut history = UndoHistory::default();
        let mut settings = project(1000);
        for i in 0..50 {
            settings = edited(&settings, i);
            history.push(settings.clone(), 1);
        }
        assert_eq!(history.len(), ViewportMemorySettings::MIN_UNDO_STEPS);

        let mut history = UndoHistory::default();
        for i in 0..150 {
            settings = edited(&settings, i);
            history.push(settings.clone(), u64::MAX);
        }
        assert_eq!(history.len(), MAX_STEPS);

        // A budget for about 20 such steps.
        let mut history = UndoHistory::default();
        let per_step = SETTINGS_BYTES + 1000 * 96 + 2048;
        for i in 0..50 {
            settings = edited(&settings, i);
            history.push(settings.clone(), 20 * per_step);
        }
        assert!((20..=21).contains(&history.len()), "{}", history.len());
        // The newest is the last pushed.
        assert_eq!(
            history
                .pop()
                .unwrap()
                .images
                .list
                .get(&PathBuf::from("0.tif"))
                .unwrap()
                .file_size,
            49
        );
    }
}
