//! # images_ext
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-10-27
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.

use evanalyzer_cfg::settings::images_settings::{ImageEntry, ImageList};
use indexmap::{Equivalent, IndexMap};
use std::hash::Hash;
use std::path::PathBuf;
use std::sync::Arc;

/// Extension methods for [`ImageList`]: an `IndexMap` of the images to the
/// outside. Reading shares; changing copies first what another copy (an
/// undo snapshot) still shares - the map's pointers once, then only the
/// entries changed.
pub trait ImageListExt {
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool;
    fn contains_key<Q: ?Sized + Hash + Equivalent<PathBuf>>(&self, key: &Q) -> bool;
    fn get<Q: ?Sized + Hash + Equivalent<PathBuf>>(&self, key: &Q) -> Option<&ImageEntry>;
    /// The entry at position `index` (insertion order).
    fn get_index(&self, index: usize) -> Option<(&PathBuf, &ImageEntry)>;
    /// The entry to change - copied first if another copy shares it.
    fn get_mut<Q: ?Sized + Hash + Equivalent<PathBuf>>(
        &mut self,
        key: &Q,
    ) -> Option<&mut ImageEntry>;
    /// Adds or replaces the entry at `key`; the one replaced.
    fn insert(&mut self, key: PathBuf, entry: ImageEntry) -> Option<ImageEntry>;
    /// Removes the entry at `key` (the later ones move up).
    fn shift_remove<Q: ?Sized + Hash + Equivalent<PathBuf>>(
        &mut self,
        key: &Q,
    ) -> Option<ImageEntry>;
    fn clear(&mut self);
    fn keys(&self) -> impl ExactSizeIterator<Item = &PathBuf>;
    fn values(&self) -> impl ExactSizeIterator<Item = &ImageEntry>;
    fn iter(&self) -> impl ExactSizeIterator<Item = (&PathBuf, &ImageEntry)>;
    /// Every entry to change - each copied first if shared, so only for
    /// edits of all images.
    fn values_mut(&mut self) -> impl ExactSizeIterator<Item = &mut ImageEntry>;
    fn iter_mut(&mut self) -> impl ExactSizeIterator<Item = (&PathBuf, &mut ImageEntry)>;
    /// The entries as a plain map (copied where another copy shares them).
    fn into_index_map(self) -> IndexMap<PathBuf, ImageEntry>;
    fn from_entries(entries: impl IntoIterator<Item = (PathBuf, ImageEntry)>) -> Self;
    /// Roughly the RAM `self` holds that `newer` doesn't share: the map's
    /// pointers unless both share the map, and every entry `newer` doesn't
    /// share - for the undo history's budget.
    fn unshared_bytes(&self, newer: &ImageList) -> u64;
}

/// RAM of one image in the map itself (key path and pointer) - measured:
/// ~9.6 MB for 100,000 images.
const MAP_SLOT_BYTES: u64 = 96;
/// RAM of an image entry without its objects (series, channels) - measured:
/// ~2.4 KB with one series of 4 channels.
const ENTRY_BYTES: u64 = 2048;
/// RAM of a hand-drawn object besides its mask (classes, links, track).
const OBJECT_BYTES: u64 = 256;

/// Roughly the RAM `entry` holds.
fn entry_bytes(entry: &ImageEntry) -> u64 {
    let objects: u64 = entry
        .series
        .values()
        .flat_map(|series| &series.objects)
        .map(|object| OBJECT_BYTES + object.mask_data.len() as u64 / 8)
        .sum();
    ENTRY_BYTES + objects
}

impl ImageListExt for ImageList {
    fn len(&self) -> usize {
        self.0.len()
    }

    fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    fn contains_key<Q: ?Sized + Hash + Equivalent<PathBuf>>(&self, key: &Q) -> bool {
        self.0.contains_key(key)
    }

    fn get<Q: ?Sized + Hash + Equivalent<PathBuf>>(&self, key: &Q) -> Option<&ImageEntry> {
        self.0.get(key).map(|entry| &**entry)
    }

    fn get_index(&self, index: usize) -> Option<(&PathBuf, &ImageEntry)> {
        self.0
            .get_index(index)
            .map(|(path, entry)| (path, &**entry))
    }

    fn get_mut<Q: ?Sized + Hash + Equivalent<PathBuf>>(
        &mut self,
        key: &Q,
    ) -> Option<&mut ImageEntry> {
        // A missing key copies nothing.
        if !self.0.contains_key(key) {
            return None;
        }
        Arc::make_mut(&mut self.0).get_mut(key).map(Arc::make_mut)
    }

    fn insert(&mut self, key: PathBuf, entry: ImageEntry) -> Option<ImageEntry> {
        Arc::make_mut(&mut self.0)
            .insert(key, Arc::new(entry))
            .map(Arc::unwrap_or_clone)
    }

    fn shift_remove<Q: ?Sized + Hash + Equivalent<PathBuf>>(
        &mut self,
        key: &Q,
    ) -> Option<ImageEntry> {
        if !self.0.contains_key(key) {
            return None;
        }
        Arc::make_mut(&mut self.0)
            .shift_remove(key)
            .map(Arc::unwrap_or_clone)
    }

    fn clear(&mut self) {
        self.0 = Arc::default();
    }

    fn keys(&self) -> impl ExactSizeIterator<Item = &PathBuf> {
        self.0.keys()
    }

    fn values(&self) -> impl ExactSizeIterator<Item = &ImageEntry> {
        self.0.values().map(|entry| &**entry)
    }

    fn iter(&self) -> impl ExactSizeIterator<Item = (&PathBuf, &ImageEntry)> {
        self.0.iter().map(|(path, entry)| (path, &**entry))
    }

    fn values_mut(&mut self) -> impl ExactSizeIterator<Item = &mut ImageEntry> {
        Arc::make_mut(&mut self.0).values_mut().map(Arc::make_mut)
    }

    fn iter_mut(&mut self) -> impl ExactSizeIterator<Item = (&PathBuf, &mut ImageEntry)> {
        Arc::make_mut(&mut self.0)
            .iter_mut()
            .map(|(path, entry)| (&*path, Arc::make_mut(entry)))
    }

    fn into_index_map(self) -> IndexMap<PathBuf, ImageEntry> {
        Arc::unwrap_or_clone(self.0)
            .into_iter()
            .map(|(path, entry)| (path, Arc::unwrap_or_clone(entry)))
            .collect()
    }

    fn unshared_bytes(&self, newer: &ImageList) -> u64 {
        if Arc::ptr_eq(&self.0, &newer.0) {
            return 0;
        }
        // Usually the same images in the same order: compared position by
        // position, without hashing; by path where they differ.
        let entries: u64 = self
            .0
            .iter()
            .enumerate()
            .filter(|(i, (path, entry))| {
                let other = match newer.0.get_index(*i) {
                    Some((other_path, other)) if other_path == *path => Some(other),
                    _ => newer.0.get(*path),
                };
                other.is_none_or(|other| !Arc::ptr_eq(entry, other))
            })
            .map(|(_, (_, entry))| entry_bytes(entry))
            .sum();
        self.0.len() as u64 * MAP_SLOT_BYTES + entries
    }

    fn from_entries(entries: impl IntoIterator<Item = (PathBuf, ImageEntry)>) -> Self {
        ImageList(Arc::new(
            entries
                .into_iter()
                .map(|(path, entry)| (path, Arc::new(entry)))
                .collect(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entry(name: &str) -> ImageEntry {
        ImageEntry {
            rel_path: PathBuf::from(name),
            ..Default::default()
        }
    }

    /// A copy shares everything; changing one entry in it copies that one
    /// only - the original and the other entries stay shared.
    #[test]
    fn a_copy_shares_its_entries_until_one_is_changed() {
        let original = ImageList::from_entries([
            (PathBuf::from("a.tif"), entry("a.tif")),
            (PathBuf::from("b.tif"), entry("b.tif")),
        ]);
        let mut copy = original.clone();
        assert!(Arc::ptr_eq(&original.0, &copy.0));

        copy.get_mut(&PathBuf::from("a.tif")).unwrap().file_size = 7;

        assert_eq!(original.get(&PathBuf::from("a.tif")).unwrap().file_size, 0);
        assert_eq!(copy.get(&PathBuf::from("a.tif")).unwrap().file_size, 7);
        assert!(!Arc::ptr_eq(&original.0, &copy.0), "the map's pointers");
        assert!(Arc::ptr_eq(
            &original.0[&PathBuf::from("b.tif")],
            &copy.0[&PathBuf::from("b.tif")]
        ));
        assert!(copy.get_mut(&PathBuf::from("missing.tif")).is_none());
    }

    /// What a copy holds on its own: nothing while it shares the map; the
    /// map's pointers and the changed entry after an edit.
    #[test]
    fn the_unshared_bytes_are_what_an_edit_copied() {
        let original = ImageList::from_entries(
            (0..10).map(|i| (PathBuf::from(format!("{i}.tif")), entry("x"))),
        );
        let mut copy = original.clone();
        assert_eq!(original.unshared_bytes(&copy), 0);
        copy.get_mut(&PathBuf::from("3.tif")).unwrap().file_size = 1;
        assert_eq!(
            original.unshared_bytes(&copy),
            10 * MAP_SLOT_BYTES + ENTRY_BYTES
        );
        // Another order: found by path.
        let reordered = ImageList(Arc::new(
            original
                .0
                .iter()
                .rev()
                .map(|(p, e)| (p.clone(), e.clone()))
                .collect(),
        ));
        assert_eq!(original.unshared_bytes(&reordered), 10 * MAP_SLOT_BYTES);
    }

    /// Saved exactly as the plain map it replaced.
    #[test]
    fn it_is_saved_as_a_plain_map() {
        let mut map = IndexMap::new();
        map.insert(PathBuf::from("b.tif"), entry("b.tif"));
        map.insert(PathBuf::from("a.tif"), entry("a.tif"));
        let list = ImageList::from_entries(map.clone());
        let json = serde_json::to_string(&list).unwrap();
        assert_eq!(json, serde_json::to_string(&map).unwrap());
        let back: ImageList = serde_json::from_str(&json).unwrap();
        assert_eq!(
            back.keys().collect::<Vec<_>>(),
            map.keys().collect::<Vec<_>>()
        );
        assert_eq!(back.into_index_map().len(), 2);
    }
}
