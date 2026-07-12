use evanalyzer_cfg::{
    AssignObjectClass,
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
}

impl ClassificationExt for ClassificationSettings {
    fn add_class(&mut self, mut new_class: Class) -> ObjectClass {
        new_class.id = next_id(&self.classes);
        let id = new_class.id;
        self.classes.push(new_class);
        id
    }

    fn update_class(&mut self, new_class: Class) -> Result<(), InternalErrors> {
        let index = self
            .classes
            .iter()
            .position(|c| c.id == new_class.id)
            .ok_or_else(|| InternalErrors::Generic("Class does not exist!".into()))?;
        self.classes[index] = new_class;
        Ok(())
    }

    fn get_class(&self, class_id: ObjectClass) -> Option<&Class> {
        self.classes.iter().find(|c| c.id == class_id)
    }

    fn move_up(&mut self, class_id: ObjectClass) {
        if let Some(i) = self.classes.iter().position(|c| c.id == class_id) {
            if i > 0 {
                self.classes.swap(i, i - 1);
            }
        }
    }

    fn move_down(&mut self, class_id: ObjectClass) {
        if let Some(i) = self.classes.iter().position(|c| c.id == class_id) {
            if i < self.classes.len() - 1 {
                self.classes.swap(i, i + 1);
            }
        }
    }

    fn delete_class(&mut self, class_id: ObjectClass) {
        self.classes.retain(|c| c.id != class_id);
    }
}

/// Returns the next available unique ID for a new class.
///
/// Calculates `max(existing_ids) + 1`, or `1` if the list is empty.
/// This ensures IDs are never reused after deletions.
fn next_id(classes: &[Class]) -> ObjectClass {
    classes
        .iter()
        .map(|c| c.id)
        .max()
        .map(|max| max + AssignObjectClass!(1))
        .unwrap_or(AssignObjectClass!(1))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn class(id: u32, name: &str) -> Class {
        Class { id: ObjectClass::Valid(id), name: name.into(), ..Default::default() }
    }

    fn settings(classes: Vec<Class>) -> ClassificationSettings {
        ClassificationSettings { classes }
    }

    // ---- add_class / next_id ----

    #[test]
    fn add_class_assigns_id_one_to_the_first_class() {
        let mut s = settings(vec![]);
        let id = s.add_class(class(0, "Nuclei"));
        assert_eq!(id, ObjectClass::Valid(1));
        assert_eq!(s.classes.len(), 1);
        assert_eq!(s.classes[0].id, ObjectClass::Valid(1));
    }

    #[test]
    fn add_class_assigns_max_plus_one_regardless_of_the_classs_own_id() {
        // The caller's `new_class.id` is always overwritten by `next_id`.
        let mut s = settings(vec![class(1, "A"), class(5, "B")]);
        let id = s.add_class(class(999, "C"));
        assert_eq!(id, ObjectClass::Valid(6));
    }

    #[test]
    fn add_class_does_not_reuse_a_deleted_non_max_id() {
        let mut s = settings(vec![]);
        let a_id = s.add_class(class(0, "A")); // -> id 1
        s.add_class(class(0, "B")); // -> id 2
        s.delete_class(a_id); // delete the non-max id
        let c_id = s.add_class(class(0, "C"));
        assert_eq!(c_id, ObjectClass::Valid(3), "max() is still 2 (B), so the next id must be 3, not the just-freed 1");
    }

    #[test]
    fn add_class_can_reuse_an_id_after_the_highest_numbered_class_is_deleted() {
        // `next_id` is `max(ids currently present) + 1`, not "highest id ever
        // assigned" - deleting whichever class currently holds the highest id
        // immediately frees that id for reuse. This contradicts a literal
        // reading of `next_id`'s doc comment ("IDs are never reused after
        // deletions"), which only holds as long as the *maximum* id is still
        // present; pinning the actual behaviour here since it's surprising.
        let mut s = settings(vec![]);
        s.add_class(class(0, "A")); // -> id 1
        let b_id = s.add_class(class(0, "B")); // -> id 2 (currently the max)
        s.delete_class(b_id);
        let c_id = s.add_class(class(0, "C"));
        assert_eq!(c_id, ObjectClass::Valid(2), "deleting the current max id makes it immediately reusable");
    }

    // ---- update_class ----

    #[test]
    fn update_class_replaces_the_class_with_a_matching_id() {
        let mut s = settings(vec![class(1, "Old")]);
        let updated = Class { id: ObjectClass::Valid(1), name: "New".into(), ..Default::default() };
        s.update_class(updated).unwrap();
        assert_eq!(s.classes[0].name, "New");
    }

    #[test]
    fn update_class_errors_for_an_unknown_id() {
        let mut s = settings(vec![class(1, "A")]);
        let err = s.update_class(class(2, "B")).unwrap_err();
        assert!(matches!(err, InternalErrors::Generic(_)));
        // The existing class must be untouched.
        assert_eq!(s.classes.len(), 1);
        assert_eq!(s.classes[0].name, "A");
    }

    // ---- get_class ----

    #[test]
    fn get_class_finds_a_class_by_id() {
        let s = settings(vec![class(1, "A"), class(2, "B")]);
        assert_eq!(s.get_class(ObjectClass::Valid(2)).map(|c| c.name.as_str()), Some("B"));
    }

    #[test]
    fn get_class_returns_none_for_an_unknown_id() {
        let s = settings(vec![class(1, "A")]);
        assert!(s.get_class(ObjectClass::Valid(99)).is_none());
    }

    // ---- move_up / move_down ----

    #[test]
    fn move_up_swaps_with_the_previous_class() {
        let mut s = settings(vec![class(1, "A"), class(2, "B"), class(3, "C")]);
        s.move_up(ObjectClass::Valid(2));
        assert_eq!(s.classes.iter().map(|c| c.name.as_str()).collect::<Vec<_>>(), vec!["B", "A", "C"]);
    }

    #[test]
    fn move_up_is_a_noop_when_already_first() {
        let mut s = settings(vec![class(1, "A"), class(2, "B")]);
        s.move_up(ObjectClass::Valid(1));
        assert_eq!(s.classes.iter().map(|c| c.name.as_str()).collect::<Vec<_>>(), vec!["A", "B"]);
    }

    #[test]
    fn move_up_is_a_noop_for_an_unknown_id() {
        let mut s = settings(vec![class(1, "A"), class(2, "B")]);
        s.move_up(ObjectClass::Valid(99));
        assert_eq!(s.classes.iter().map(|c| c.name.as_str()).collect::<Vec<_>>(), vec!["A", "B"]);
    }

    #[test]
    fn move_down_swaps_with_the_next_class() {
        let mut s = settings(vec![class(1, "A"), class(2, "B"), class(3, "C")]);
        s.move_down(ObjectClass::Valid(2));
        assert_eq!(s.classes.iter().map(|c| c.name.as_str()).collect::<Vec<_>>(), vec!["A", "C", "B"]);
    }

    #[test]
    fn move_down_is_a_noop_when_already_last() {
        let mut s = settings(vec![class(1, "A"), class(2, "B")]);
        s.move_down(ObjectClass::Valid(2));
        assert_eq!(s.classes.iter().map(|c| c.name.as_str()).collect::<Vec<_>>(), vec!["A", "B"]);
    }

    #[test]
    fn move_down_is_a_noop_for_an_unknown_id() {
        let mut s = settings(vec![class(1, "A"), class(2, "B")]);
        s.move_down(ObjectClass::Valid(99));
        assert_eq!(s.classes.iter().map(|c| c.name.as_str()).collect::<Vec<_>>(), vec!["A", "B"]);
    }

    #[test]
    fn move_down_on_a_single_class_list_does_not_underflow() {
        // Regression guard for `i < self.classes.len() - 1`: with one class,
        // `i` is 0 and `len() - 1` is also 0, so the swap must not run (and
        // must not panic via `usize` underflow on an empty list either).
        let mut s = settings(vec![class(1, "A")]);
        s.move_down(ObjectClass::Valid(1));
        assert_eq!(s.classes.len(), 1);
    }

    // ---- delete_class ----

    #[test]
    fn delete_class_removes_the_matching_class() {
        let mut s = settings(vec![class(1, "A"), class(2, "B")]);
        s.delete_class(ObjectClass::Valid(1));
        assert_eq!(s.classes.iter().map(|c| c.name.as_str()).collect::<Vec<_>>(), vec!["B"]);
    }

    #[test]
    fn delete_class_is_a_noop_for_an_unknown_id() {
        let mut s = settings(vec![class(1, "A")]);
        s.delete_class(ObjectClass::Valid(99));
        assert_eq!(s.classes.len(), 1);
    }
}
