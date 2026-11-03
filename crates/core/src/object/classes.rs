//! # classes
//!
//! An object's classes as a sorted list: the same order in every run (the
//! export, scripts and saved annotations list them in it - a hash set gave a
//! random order per run), and smaller than a hash set for the one to few
//! classes an object has (no table, no control bytes; `contains` compares a
//! handful of values instead of hashing).
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-11-03
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.

use evanalyzer_cfg::core_types::ObjectClass;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeSet, HashSet};

/// An object's classes, sorted (`ObjectClass`'s order: unset, the project's
/// by number, a module's by number), each once. Serialized as a list (as the
/// hash set before it); sorted again when read.
#[derive(Debug, Default, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(from = "Vec<ObjectClass>", into = "Vec<ObjectClass>")]
pub struct ObjectClasses(Vec<ObjectClass>);

impl ObjectClasses {
    pub fn new() -> Self {
        Self(Vec::new())
    }

    /// Adds `class`; `false` if it was there.
    pub fn insert(&mut self, class: ObjectClass) -> bool {
        match self.0.binary_search(&class) {
            Ok(_) => false,
            Err(at) => {
                self.0.insert(at, class);
                true
            }
        }
    }

    /// Removes `class`; `false` if it wasn't there.
    pub fn remove(&mut self, class: &ObjectClass) -> bool {
        match self.0.binary_search(class) {
            Ok(at) => {
                self.0.remove(at);
                true
            }
            Err(_) => false,
        }
    }

    pub fn contains(&self, class: &ObjectClass) -> bool {
        self.0.contains(class)
    }

    pub fn clear(&mut self) {
        self.0.clear();
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// In order.
    pub fn iter(&self) -> std::slice::Iter<'_, ObjectClass> {
        self.0.iter()
    }

    pub fn as_slice(&self) -> &[ObjectClass] {
        &self.0
    }

    /// Keeps the classes `keep` picks.
    pub fn retain(&mut self, keep: impl FnMut(&ObjectClass) -> bool) {
        self.0.retain(keep);
    }

    /// As a set, as the settings keep them (sorted too).
    pub fn to_set(&self) -> BTreeSet<ObjectClass> {
        self.0.iter().copied().collect()
    }
}

impl FromIterator<ObjectClass> for ObjectClasses {
    fn from_iter<I: IntoIterator<Item = ObjectClass>>(classes: I) -> Self {
        let mut classes: Vec<ObjectClass> = classes.into_iter().collect();
        classes.sort_unstable();
        classes.dedup();
        classes.shrink_to_fit();
        Self(classes)
    }
}

impl Extend<ObjectClass> for ObjectClasses {
    fn extend<I: IntoIterator<Item = ObjectClass>>(&mut self, classes: I) {
        for class in classes {
            self.insert(class);
        }
    }
}

impl<const N: usize> From<[ObjectClass; N]> for ObjectClasses {
    fn from(classes: [ObjectClass; N]) -> Self {
        classes.into_iter().collect()
    }
}

impl From<Vec<ObjectClass>> for ObjectClasses {
    fn from(classes: Vec<ObjectClass>) -> Self {
        classes.into_iter().collect()
    }
}

impl From<ObjectClasses> for Vec<ObjectClass> {
    fn from(classes: ObjectClasses) -> Self {
        classes.0
    }
}

impl From<HashSet<ObjectClass>> for ObjectClasses {
    fn from(classes: HashSet<ObjectClass>) -> Self {
        classes.into_iter().collect()
    }
}

impl From<BTreeSet<ObjectClass>> for ObjectClasses {
    fn from(classes: BTreeSet<ObjectClass>) -> Self {
        // In order already.
        Self(classes.into_iter().collect())
    }
}

impl<'a> IntoIterator for &'a ObjectClasses {
    type Item = &'a ObjectClass;
    type IntoIter = std::slice::Iter<'a, ObjectClass>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.iter()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// However they're added, the classes come out sorted, each once.
    #[test]
    fn classes_stay_sorted_and_unique() {
        let mut classes = ObjectClasses::from([ObjectClass::Global(5), ObjectClass::Global(2)]);
        assert!(classes.insert(ObjectClass::Local(1)));
        assert!(classes.insert(ObjectClass::Global(3)));
        assert!(!classes.insert(ObjectClass::Global(5)));
        classes.extend([ObjectClass::Unset, ObjectClass::Global(2)]);
        assert_eq!(
            classes.as_slice(),
            [
                ObjectClass::Unset,
                ObjectClass::Global(2),
                ObjectClass::Global(3),
                ObjectClass::Global(5),
                ObjectClass::Local(1),
            ]
        );
        assert!(classes.remove(&ObjectClass::Global(3)));
        assert!(!classes.remove(&ObjectClass::Global(3)));
        assert!(classes.contains(&ObjectClass::Global(5)));
        assert_eq!(classes.len(), 4);
        let from_set: ObjectClasses =
            HashSet::from([ObjectClass::Global(9), ObjectClass::Global(1)]).into();
        assert_eq!(
            from_set.as_slice(),
            [ObjectClass::Global(1), ObjectClass::Global(9)]
        );
    }
}
