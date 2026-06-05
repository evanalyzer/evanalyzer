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
