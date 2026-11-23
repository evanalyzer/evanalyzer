//! # classification_ext
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-06-05
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.

use crate::images::image_entry::ImageEntryExt;
use crate::images::image_list::ImageListExt;
use crate::project::ProjectWithRuntime;
use crate::project::utils::{channel_colors, class_name_of_channel, rgb_to_u32};

use evanalyzer_cfg::{
    core_types::{InternalErrors, ObjectClass},
    settings::classification_settings::{Class, ClassificationSettings},
};

/// Extension methods for [`ClassificationSettings`] providing
/// mutation and lookup operations on the class list.
pub trait ClassificationExt {
    /// Adds a new class to the collection with an auto-assigned unique ID.
    ///
    /// The ID is calculated as `max(existing_ids) + 1`, ensuring IDs are
    /// never reused even after deletions.
    ///
    /// # Returns
    /// The [`ObjectClass`] ID assigned to the new class.
    fn add_class(&mut self, new_class: Class) -> ObjectClass;

    /// Adds `new_class` with its own id - one of
    /// [`free_class_ids`](crate::project::class_ids::free_class_ids).
    ///
    /// # Errors
    /// If the id is 0 (the background's) or taken.
    fn add_class_with_id(&mut self, new_class: Class) -> Result<ObjectClass, InternalErrors>;

    /// Replaces an existing class identified by its ID.
    ///
    /// # Errors
    /// Returns [`InternalErrors`] if no class with the given ID exists.
    fn update_class(&mut self, new_class: Class) -> Result<(), InternalErrors>;

    /// Returns a reference to the class with the given ID.
    ///
    /// Returns `None` if no class with that ID exists.
    fn get_class(&self, class_id: ObjectClass) -> Option<&Class>;

    /// Moves the class one position earlier in the list.
    ///
    /// No-op if the class is already first or the ID is not found.
    fn move_up(&mut self, class_id: ObjectClass);

    /// Moves the class one position later in the list.
    ///
    /// No-op if the class is already last or the ID is not found.
    fn move_down(&mut self, class_id: ObjectClass);

    /// Removes the class with the given ID from the collection.
    ///
    /// No-op if no class with that ID exists.
    fn delete_class(&mut self, class_id: ObjectClass);

    /// A copy of the class right below it: a new id (as [`Self::add_class`]),
    /// its name with " copy", the same color, notes and results setting.
    ///
    /// # Returns
    /// The copy's id; `None` for the background or an unknown id.
    fn duplicate_class(&mut self, class_id: ObjectClass) -> Option<ObjectClass>;

    /// The class with the global id `id`.
    fn get_class_from_id(&self, id: &ObjectClass) -> Option<&Class>;

    /// Removes every class.
    fn delete_all_classes(&mut self);
}

impl ClassificationExt for ClassificationSettings {
    fn add_class(&mut self, mut new_class: Class) -> ObjectClass {
        new_class.id = crate::project::class_ids::next_class_id(self);
        let id = new_class.global();
        self.classes_mut().push(new_class);
        id
    }

    fn add_class_with_id(&mut self, new_class: Class) -> Result<ObjectClass, InternalErrors> {
        if new_class.id == 0 || self.classes().iter().any(|c| c.id == new_class.id) {
            return Err(InternalErrors::InvalidArgument(format!(
                "class id {} is taken",
                new_class.id
            )));
        }
        let id = new_class.global();
        self.classes_mut().push(new_class);
        Ok(id)
    }

    fn update_class(&mut self, new_class: Class) -> Result<(), InternalErrors> {
        let index = self
            .classes()
            .iter()
            .position(|c| c.id == new_class.id)
            .ok_or_else(|| InternalErrors::Generic("Class does not exist!".into()))?;
        self.classes_mut()[index] = new_class;
        Ok(())
    }

    fn get_class(&self, class_id: ObjectClass) -> Option<&Class> {
        self.classes().iter().find(|c| c.global() == class_id)
    }

    fn move_up(&mut self, class_id: ObjectClass) {
        if let Some(i) = self.classes().iter().position(|c| c.global() == class_id) {
            if i > 1 {
                // Background class must always be on top, therefore 1 and not 0
                self.classes_mut().swap(i, i - 1);
            }
        }
    }

    fn move_down(&mut self, class_id: ObjectClass) {
        if let Some(i) = self.classes().iter().position(|c| c.global() == class_id) {
            if i < self.classes().len() - 1 {
                self.classes_mut().swap(i, i + 1);
            }
        }
    }

    fn duplicate_class(&mut self, class_id: ObjectClass) -> Option<ObjectClass> {
        if class_id == ObjectClass::BACKGROUND {
            return None;
        }
        let index = self.classes().iter().position(|c| c.global() == class_id)?;
        let mut copy = self.classes()[index].clone();
        copy.id = crate::project::class_ids::next_class_id(self);
        copy.name = format!("{} copy", copy.name);
        let id = copy.global();
        self.classes_mut().insert(index + 1, copy);
        Some(id)
    }

    fn delete_class(&mut self, class_id: ObjectClass) {
        if class_id == ObjectClass::BACKGROUND {
            // Background class is not allowed to be deleted
            return;
        }
        self.classes_mut().retain(|c| c.global() != class_id);
    }

    fn get_class_from_id(&self, id: &ObjectClass) -> Option<&Class> {
        self.classes().iter().find(|c| c.global() == *id)
    }

    fn delete_all_classes(&mut self) {
        *self = ClassificationSettings::new();
    }
}

impl ProjectWithRuntime {
    pub fn auto_add_classes_based_on_image_meta(&mut self) {
        // Collect channel data first to avoid holding Deref borrows while mutating.
        let classes: Vec<Class> = {
            // The selected image, or the first one of the project if none is selected.
            let image = self
                .get_current_image_settings()
                .or_else(|| self.settings.images.list.values().next());
            image
                .and_then(|image| {
                    image
                        .series
                        .get(&image.active_series(&self.images.settings))
                })
                .map(|series_data| {
                    // A global wavelength set by the user wins over the image's own,
                    // so the class gets the colour the viewer shows for the channel -
                    // and channels without one get distinct default colours.
                    let wave_lengths: Vec<(i32, f32)> = series_data
                        .channels
                        .iter()
                        .map(|(channel_id, ch)| {
                            let wave_length = self
                                .images
                                .settings
                                .channels
                                .get(channel_id)
                                .and_then(|channel| channel.emission_wave_length)
                                .or(ch.emission_wave_length)
                                .unwrap_or(0.0);
                            (*channel_id, wave_length)
                        })
                        .collect();
                    let colors = channel_colors(&wave_lengths);
                    series_data
                        .channels
                        .iter()
                        .map(|(channel_id, ch)| Class {
                            id: 0, // `add_class` gives it the next free one
                            color: rgb_to_u32(colors[channel_id]),
                            name: class_name_of_channel(&ch.name, *channel_id),
                            notes: "".into(),
                            results: Default::default(),
                        })
                        .collect()
                })
                .unwrap_or_default()
        };
        for class in classes {
            self.classification.add_class(class);
        }
    }

    pub fn set_selected_object_class(&mut self, object_class: ObjectClass) {
        self.tmp_settings.selected_object_class = object_class;
    }

    pub fn get_selected_object_class(&self) -> ObjectClass {
        self.tmp_settings.selected_object_class.clone()
    }

    /// Gives class `from` the id `to` with every reference to it (see
    /// [`change_class_id`](crate::project::class_ids::change_class_id)),
    /// and the editor's state too: the selected class, hidden classes.
    pub fn change_class_id(
        &mut self,
        from: u32,
        to: u32,
    ) -> Result<crate::project::class_ids::ClassIdChange, InternalErrors> {
        let change = crate::project::class_ids::change_class_id(&mut self.settings, from, to)?;
        let (old, new) = (ObjectClass::Global(from), ObjectClass::Global(to));
        let tmp = &mut self.tmp_settings;
        if tmp.selected_object_class == old {
            tmp.selected_object_class = new;
        }
        if tmp.hidden_classes.remove(&old) {
            tmp.hidden_classes.insert(new);
        }
        if let Some(focus) = &mut tmp.pipeline_focus
            && focus.visible_classes.remove(&old)
        {
            focus.visible_classes.insert(new);
        }
        Ok(change)
    }

    pub fn toggle_class_visibility(&mut self, class_id: ObjectClass) {
        if let Some(focus) = &mut self.tmp_settings.pipeline_focus {
            if !focus.visible_classes.remove(&class_id) {
                focus.visible_classes.insert(class_id);
            }
            return;
        }
        if self.tmp_settings.hidden_classes.contains(&class_id) {
            self.tmp_settings.hidden_classes.remove(&class_id);
        } else {
            self.tmp_settings.hidden_classes.insert(class_id);
        }
    }

    pub fn is_class_visible(&self, class_id: &ObjectClass) -> bool {
        match &self.tmp_settings.pipeline_focus {
            Some(focus) => focus.visible_classes.contains(class_id),
            None => !self.tmp_settings.hidden_classes.contains(class_id),
        }
    }

    pub fn toggle_hide_unclassified_objects(&mut self) {
        self.tmp_settings.hide_unclassified_objects = !self.tmp_settings.hide_unclassified_objects;
    }

    pub fn hide_unclassified_objects(&self) -> bool {
        self.tmp_settings.hide_unclassified_objects
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn class(id: u32, name: &str) -> Class {
        Class {
            id,
            name: name.into(),
            ..Default::default()
        }
    }

    fn settings(classes: Vec<Class>) -> ClassificationSettings {
        ClassificationSettings::new_from_existing(classes)
    }

    // ---- duplicate_class ----

    /// The copy sits right below the original, with a new id, " copy" after
    /// the name and the rest the same.
    #[test]
    fn duplicate_class_puts_a_renamed_copy_with_a_new_id_below_it() {
        let mut s = settings(vec![class(1, "A"), class(5, "B")]);
        let mut a = s.get_class(ObjectClass::Global(1)).unwrap().clone();
        a.color = 0x00FF_0000;
        a.notes = "kept".into();
        s.update_class(a).unwrap();

        let id = s.duplicate_class(ObjectClass::Global(1));

        assert_eq!(id, Some(ObjectClass::Global(6)));
        let names: Vec<&str> = s.classes().iter().map(|c| c.name.as_str()).collect();
        assert_eq!(names[1..], ["A", "A copy", "B"]);
        let copy = &s.classes()[2];
        assert_eq!(
            (copy.id, copy.color, copy.notes.as_str()),
            (6, 0x00FF_0000, "kept")
        );
    }

    #[test]
    fn duplicate_class_leaves_the_background_and_unknown_ids_alone() {
        let mut s = settings(vec![class(1, "A")]);
        assert_eq!(s.duplicate_class(ObjectClass::BACKGROUND), None);
        assert_eq!(s.duplicate_class(ObjectClass::Global(42)), None);
        assert_eq!(s.classes().len(), 2);
    }

    // ---- add_class / next_class_id ----

    #[test]
    fn add_class_assigns_id_one_to_the_first_class() {
        let mut s = settings(vec![]);
        let id = s.add_class(class(0, "Nuclei"));
        assert_eq!(id, ObjectClass::Global(1));
        // classes()[0] is always the auto-prepended Background class.
        assert_eq!(s.classes().len(), 2);
        assert_eq!(s.classes()[1].id, 1);
    }

    #[test]
    fn add_class_assigns_max_plus_one_regardless_of_the_classs_own_id() {
        // The caller's `new_class.id` is always overwritten by `next_class_id`.
        let mut s = settings(vec![class(1, "A"), class(5, "B")]);
        let id = s.add_class(class(999, "C"));
        assert_eq!(id, ObjectClass::Global(6));
    }

    #[test]
    fn add_class_does_not_reuse_a_deleted_non_max_id() {
        let mut s = settings(vec![]);
        let a_id = s.add_class(class(0, "A")); // -> id 1
        s.add_class(class(0, "B")); // -> id 2
        s.delete_class(a_id); // delete the non-max id
        let c_id = s.add_class(class(0, "C"));
        assert_eq!(
            c_id,
            ObjectClass::Global(3),
            "max() is still 2 (B), so the next id must be 3, not the just-freed 1"
        );
    }

    #[test]
    fn add_class_with_id_keeps_a_free_id_and_refuses_a_taken_one_or_zero() {
        let mut s = settings(vec![class(1, "A"), class(5, "B")]);
        assert_eq!(
            s.add_class_with_id(class(3, "C")).unwrap(),
            ObjectClass::Global(3)
        );
        assert!(s.add_class_with_id(class(5, "D")).is_err());
        assert!(s.add_class_with_id(class(0, "E")).is_err());
        let ids: Vec<u32> = s.classes().iter().map(|c| c.id).collect();
        assert_eq!(ids, [0, 1, 5, 3]);
    }

    #[test]
    fn add_class_can_reuse_an_id_after_the_highest_numbered_class_is_deleted() {
        // `next_class_id` is `max(ids currently present) + 1`, not "highest id
        // ever assigned" - deleting the class with the highest id frees that
        // id again (the exception its doc comment names).
        let mut s = settings(vec![]);
        s.add_class(class(0, "A")); // -> id 1
        let b_id = s.add_class(class(0, "B")); // -> id 2 (currently the max)
        s.delete_class(b_id);
        let c_id = s.add_class(class(0, "C"));
        assert_eq!(
            c_id,
            ObjectClass::Global(2),
            "deleting the current max id makes it immediately reusable"
        );
    }

    // ---- update_class ----

    #[test]
    fn update_class_replaces_the_class_with_a_matching_id() {
        let mut s = settings(vec![class(1, "Old")]);
        let updated = Class {
            id: 1,
            name: "New".into(),
            ..Default::default()
        };
        s.update_class(updated).unwrap();
        // classes()[0] is the auto-prepended Background class.
        assert_eq!(s.classes()[1].name, "New");
    }

    #[test]
    fn update_class_errors_for_an_unknown_id() {
        let mut s = settings(vec![class(1, "A")]);
        let err = s.update_class(class(2, "B")).unwrap_err();
        assert!(matches!(err, InternalErrors::Generic(_)));
        // The existing class must be untouched. classes()[0] is Background.
        assert_eq!(s.classes().len(), 2);
        assert_eq!(s.classes()[1].name, "A");
    }

    // ---- get_class ----

    #[test]
    fn get_class_finds_a_class_by_id() {
        let s = settings(vec![class(1, "A"), class(2, "B")]);
        assert_eq!(
            s.get_class(ObjectClass::Global(2)).map(|c| c.name.as_str()),
            Some("B")
        );
    }

    #[test]
    fn get_class_returns_none_for_an_unknown_id() {
        let s = settings(vec![class(1, "A")]);
        assert!(s.get_class(ObjectClass::Global(99)).is_none());
    }

    // ---- move_up / move_down ----

    // classes()[0] is always the auto-prepended Background class - `move_up`'s
    // `i > 1` guard (not `i > 0`) keeps it pinned there, so index 1 is the
    // effective "first" position a real class can reach.

    #[test]
    fn move_up_swaps_with_the_previous_class() {
        let mut s = settings(vec![class(1, "A"), class(2, "B"), class(3, "C")]);
        s.move_up(ObjectClass::Global(2));
        assert_eq!(
            s.classes()
                .iter()
                .map(|c| c.name.as_str())
                .collect::<Vec<_>>(),
            vec!["Background", "B", "A", "C"]
        );
    }

    #[test]
    fn move_up_is_a_noop_when_already_first() {
        let mut s = settings(vec![class(1, "A"), class(2, "B")]);
        s.move_up(ObjectClass::Global(1));
        assert_eq!(
            s.classes()
                .iter()
                .map(|c| c.name.as_str())
                .collect::<Vec<_>>(),
            vec!["Background", "A", "B"]
        );
    }

    #[test]
    fn move_up_is_a_noop_for_an_unknown_id() {
        let mut s = settings(vec![class(1, "A"), class(2, "B")]);
        s.move_up(ObjectClass::Global(99));
        assert_eq!(
            s.classes()
                .iter()
                .map(|c| c.name.as_str())
                .collect::<Vec<_>>(),
            vec!["Background", "A", "B"]
        );
    }

    #[test]
    fn move_down_swaps_with_the_next_class() {
        let mut s = settings(vec![class(1, "A"), class(2, "B"), class(3, "C")]);
        s.move_down(ObjectClass::Global(2));
        assert_eq!(
            s.classes()
                .iter()
                .map(|c| c.name.as_str())
                .collect::<Vec<_>>(),
            vec!["Background", "A", "C", "B"]
        );
    }

    #[test]
    fn move_down_is_a_noop_when_already_last() {
        let mut s = settings(vec![class(1, "A"), class(2, "B")]);
        s.move_down(ObjectClass::Global(2));
        assert_eq!(
            s.classes()
                .iter()
                .map(|c| c.name.as_str())
                .collect::<Vec<_>>(),
            vec!["Background", "A", "B"]
        );
    }

    #[test]
    fn move_down_is_a_noop_for_an_unknown_id() {
        let mut s = settings(vec![class(1, "A"), class(2, "B")]);
        s.move_down(ObjectClass::Global(99));
        assert_eq!(
            s.classes()
                .iter()
                .map(|c| c.name.as_str())
                .collect::<Vec<_>>(),
            vec!["Background", "A", "B"]
        );
    }

    #[test]
    fn move_down_on_a_single_class_list_does_not_underflow() {
        // Regression guard for `i < self.classes().len() - 1`: with Background
        // + one class, `i` is 1 (A) and `len() - 1` is also 1, so the swap
        // must not run (and must not panic via `usize` underflow either).
        let mut s = settings(vec![class(1, "A")]);
        s.move_down(ObjectClass::Global(1));
        assert_eq!(s.classes().len(), 2);
    }

    // ---- delete_class ----

    #[test]
    fn delete_class_removes_the_matching_class() {
        let mut s = settings(vec![class(1, "A"), class(2, "B")]);
        s.delete_class(ObjectClass::Global(1));
        assert_eq!(
            s.classes()
                .iter()
                .map(|c| c.name.as_str())
                .collect::<Vec<_>>(),
            vec!["Background", "B"]
        );
    }

    #[test]
    fn delete_class_is_a_noop_for_an_unknown_id() {
        let mut s = settings(vec![class(1, "A")]);
        s.delete_class(ObjectClass::Global(99));
        assert_eq!(s.classes().len(), 2);
    }
}
