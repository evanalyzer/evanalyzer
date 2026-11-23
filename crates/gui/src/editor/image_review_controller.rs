// # image_review_controller
//
// **Author:** Joachim Danmayr
// **Date:** 2026-11-20
//
// ## License
// Copyright 2026 Joachim Danmayr.
// Licensed under the **PolyForm Noncommercial License 1.0.0**.

//! An image's review in the editor (design N6): its note, marker and left
//! out, set in Image info's "Note & marker", in the popover of a list row or
//! the image chip, and with keys 1-6 / 0. Stored in the project
//! (`ImageEntry::review`), copied into the results at a run.

use crate::editor::images_list_controller::ImagesListController;
use crate::{
    AppWindow, ImageItemData, ImageReviewData, ImageReviewState, ImagesListState, UiState,
};
use evanalyzer_app::project::ImageListExt;
use evanalyzer_cfg::settings::images_settings::{
    ImageMarker, ImageReview, LeaveOut, LeaveOutReason,
};
use slint::{ComponentHandle, Model};
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// One change of an image's review.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum ReviewChange {
    Marker(Option<ImageMarker>),
    Note(String),
    LeftOut(bool),
    /// Only for a left-out image.
    Reason(Option<LeaveOutReason>),
}

impl ReviewChange {
    /// Typing a note is one undo step per pause (coalesced); a click is one
    /// of its own.
    fn own_undo_step(&self) -> bool {
        !matches!(self, Self::Note(_))
    }

    /// Applies the change; whether it changed anything.
    pub(crate) fn apply(self, review: &mut ImageReview) -> bool {
        let before = review.clone();
        match self {
            Self::Marker(marker) => review.marker = marker,
            Self::Note(note) => review.note = note,
            Self::LeftOut(true) => {
                review.leave_out.get_or_insert_with(LeaveOut::default);
            }
            Self::LeftOut(false) => review.leave_out = None,
            Self::Reason(reason) => {
                if let Some(leave_out) = review.leave_out.as_mut() {
                    leave_out.reason = reason;
                }
            }
        }
        *review != before
    }
}

/// The review as the UI shows it.
pub(crate) fn review_data(review: &ImageReview) -> ImageReviewData {
    let reason = review
        .leave_out
        .as_ref()
        .and_then(|leave_out| leave_out.reason)
        .and_then(|reason| LeaveOutReason::ALL.iter().position(|r| *r == reason));
    ImageReviewData {
        note: review.note.as_str().into(),
        marker: review.marker.map_or(0, |marker| marker.number() as i32),
        left_out: review.leave_out.is_some(),
        reason: reason.map_or(-1, |i| i as i32),
    }
}

/// Puts the open image's review into Image info and the image chip -
/// after another image was opened, an undo, another project.
pub(crate) fn sync_current_review_to_slint(ui: &slint::Weak<AppWindow>, app_state: &UiState) {
    let current = {
        let project = app_state.get_project();
        project.get_current_rel_image_path_cloned().and_then(|rel| {
            project
                .images
                .list
                .get(&rel)
                .map(|image| image.review.clone())
        })
    };
    let ui = ui.clone();
    crate::helper::ui_thread::invoke_from_event_loop(move || {
        if let Some(ui) = ui.upgrade() {
            let state = ui.global::<ImageReviewState>();
            state.set_has_image(current.is_some());
            state.set_current(review_data(&current.unwrap_or_default()));
        }
    })
    .ok();
}

impl ImagesListController {
    pub(crate) fn init_review_callbacks(self: &Arc<Self>) {
        let Some(ui) = self.ui.upgrade() else {
            return;
        };
        let state = ui.global::<ImageReviewState>();

        let manager = Arc::clone(self);
        state.on_open_popup(move |path| manager.open_review_popup(&path));

        let manager = Arc::clone(self);
        state.on_set_marker(move |path, marker| {
            let marker = ImageMarker::from_number(i64::from(marker));
            manager.change_review(&path, ReviewChange::Marker(marker));
        });

        let manager = Arc::clone(self);
        state.on_set_note(move |path, note| {
            manager.change_review(&path, ReviewChange::Note(note.to_string()));
        });

        let manager = Arc::clone(self);
        state.on_set_left_out(move |path, left_out| {
            manager.change_review(&path, ReviewChange::LeftOut(left_out));
        });

        let manager = Arc::clone(self);
        state.on_set_reason(move |path, reason| {
            let reason = usize::try_from(reason)
                .ok()
                .and_then(|i| LeaveOutReason::ALL.get(i).copied());
            manager.change_review(&path, ReviewChange::Reason(reason));
        });
    }

    /// The image a review callback is about: `path` relative to the image
    /// folder, "" the open image.
    fn review_target(&self, path: &str) -> Option<PathBuf> {
        if path.is_empty() {
            self.app_state
                .get_project()
                .get_current_rel_image_path_cloned()
        } else {
            Some(PathBuf::from(path))
        }
    }

    /// Fills the popover for the image at `path` ("" the open image).
    fn open_review_popup(&self, path: &str) {
        let Some(rel) = self.review_target(path) else {
            return;
        };
        let review = {
            let project = self.app_state.get_project();
            project
                .images
                .list
                .get(&rel)
                .map(|image| image.review.clone())
        };
        let Some(review) = review else {
            return;
        };
        let Some(ui) = self.ui.upgrade() else {
            return;
        };
        let state = ui.global::<ImageReviewState>();
        let name = rel
            .file_name()
            .map(|name| name.to_string_lossy().into_owned())
            .unwrap_or_default();
        state.set_popup_path(rel.to_string_lossy().as_ref().into());
        state.set_popup_name(name.into());
        state.set_popup(review_data(&review));
    }

    /// Changes the review of the image at `path` ("" the open image) in the
    /// project and shows it where it is shown. Runs on the UI thread.
    pub(crate) fn change_review(&self, path: &str, change: ReviewChange) {
        let Some(rel) = self.review_target(path) else {
            return;
        };
        let changes = {
            let project = self.app_state.get_project();
            project
                .images
                .list
                .get(&rel)
                .is_some_and(|image| change.clone().apply(&mut image.review.clone()))
        };
        if !changes {
            return;
        }
        let marker_changed = matches!(change, ReviewChange::Marker(_));
        if change.own_undo_step() {
            self.app_state.begin_undo_step();
        }
        let review = {
            let mut project = self.app_state.get_project_write();
            let Some(image) = project.images.list.get_mut(&rel) else {
                return;
            };
            change.apply(&mut image.review);
            image.review.clone()
        };
        self.app_state.mark_dirty();
        // Narrowed to markers: a new marker may take the image out of the
        // list or bring it in.
        if marker_changed {
            if self.marker_filter() != 0 {
                self.refilter();
            } else {
                self.show_marker_filter_row(review.marker.is_some());
            }
        }
        self.show_review(&rel, &review);
    }

    /// Shows the marker filter row while some image has a marker: at once
    /// when one was just set, else (one removed) only if another is left.
    fn show_marker_filter_row(&self, marker_set: bool) {
        let any_marker = marker_set
            || self
                .app_state
                .get_project()
                .images
                .list
                .values()
                .any(|image| image.review.marker.is_some());
        if let Some(ui) = self.ui.upgrade() {
            ui.global::<ImagesListState>().set_any_marker(any_marker);
        }
    }

    /// Shows `review` of the image at `rel`: Image info and the chip (the
    /// open image), the popover (its image) and its rows in the list and
    /// the chip's picker - only those rows, not the whole list.
    fn show_review(&self, rel: &Path, review: &ImageReview) {
        let Some(ui) = self.ui.upgrade() else {
            return;
        };
        let data = review_data(review);
        let state = ui.global::<ImageReviewState>();
        let rel_text = rel.to_string_lossy();
        if self
            .app_state
            .get_project()
            .get_current_rel_image_path_cloned()
            .as_deref()
            == Some(rel)
        {
            state.set_current(data.clone());
        }
        if state.get_popup_path().as_str() == rel_text {
            state.set_popup(data.clone());
        }
        let list = ui.global::<ImagesListState>();
        for model in [list.get_images_list(), list.get_quick_images()] {
            let found = (0..model.row_count()).find_map(|i| {
                model
                    .row_data(i)
                    .filter(|row| !row.folder && row.path.as_str() == rel_text)
                    .map(|row| (i, row))
            });
            if let Some((index, row)) = found {
                model.set_row_data(index, with_review(row, &data));
            }
        }
    }
}

/// A list row with `data`'s marks.
fn with_review(row: ImageItemData, data: &ImageReviewData) -> ImageItemData {
    ImageItemData {
        marker: data.marker,
        has_note: !data.note.is_empty(),
        left_out: data.left_out,
        ..row
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_reason_only_counts_for_a_left_out_image_and_goes_with_it() {
        let mut review = ImageReview::default();
        assert!(!ReviewChange::Reason(Some(LeaveOutReason::Empty)).apply(&mut review));
        assert!(ReviewChange::LeftOut(true).apply(&mut review));
        assert!(ReviewChange::Reason(Some(LeaveOutReason::Empty)).apply(&mut review));
        assert_eq!(review_data(&review).reason, 2);
        assert!(
            !ReviewChange::LeftOut(true).apply(&mut review),
            "already out"
        );
        assert_eq!(review_data(&review).reason, 2, "the reason stays");
        assert!(ReviewChange::LeftOut(false).apply(&mut review));
        assert!(review.is_empty());
    }

    #[test]
    fn the_same_marker_or_note_again_changes_nothing() {
        let mut review = ImageReview::default();
        assert!(ReviewChange::Marker(Some(ImageMarker::Green)).apply(&mut review));
        assert!(!ReviewChange::Marker(Some(ImageMarker::Green)).apply(&mut review));
        assert!(ReviewChange::Note("x".into()).apply(&mut review));
        assert!(!ReviewChange::Note("x".into()).apply(&mut review));
        assert_eq!(review_data(&review).marker, 4);
    }
}
