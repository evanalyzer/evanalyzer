//! # object_cache
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-09-03
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.

use crate::object::Object;
use evanalyzer_cfg::core_types::ObjectId;
use std::collections::BTreeMap;

/// Drop-in replacement for `BTreeMap<ObjectId, Object>` as `PipelineCache`'s
/// object store. Every method here matches `BTreeMap`'s own signature
/// exactly, so every existing call site (~250 of them, across every object
/// algorithm) keeps compiling unchanged - the only thing that changed is
/// this field's declared type.
///
/// Named "cache" rather than "store" because the real store, in the end, is
/// the DuckDB results database - this type is the in-memory (and, for the
/// whole-image phase, disk-scratch-backed) working view over it that a
/// pipeline run actually manipulates.
///
/// Today this is a thin pass-through, identical in behavior and cost to the
/// `BTreeMap` it replaces. It exists as the seam for the whole-image
/// object-store work: a `PipelineCache` built for a single tile's
/// `ExecutionScope::Tile` run never needs to hold more than that tile's own
/// (already small) object set, so it can stay exactly this simple; the
/// whole-image `PipelineCache` built for `ExecutionScope::WholeImage` will
/// back this same type with the per-tile scratch files from
/// `job::object_scratch`, paging tiles in/out of memory as `insert`/`get`/
/// `values`/etc. are called - transparently to every algorithm that already
/// calls these methods, since none of their call sites need to change.
///
/// The objects are boxed: a B-tree keeps its values in its nodes, which are
/// partly empty - with the 528-byte `Object` inline that was ~450 bytes more
/// per object (measured 2026-11-03: 1.6 KB per small object instead of
/// ~1.1 KB), a pointer per slot now.
#[derive(Default, Clone)]
pub struct ObjectCache {
    objects: BTreeMap<ObjectId, Box<Object>>,
}

impl ObjectCache {
    pub fn insert(&mut self, id: ObjectId, object: Object) -> Option<Object> {
        self.objects.insert(id, Box::new(object)).map(|o| *o)
    }

    pub fn get(&self, id: &ObjectId) -> Option<&Object> {
        self.objects.get(id).map(|o| &**o)
    }

    pub fn get_mut(&mut self, id: &ObjectId) -> Option<&mut Object> {
        self.objects.get_mut(id).map(|o| &mut **o)
    }

    pub fn remove(&mut self, id: &ObjectId) -> Option<Object> {
        self.objects.remove(id).map(|o| *o)
    }

    pub fn values(&self) -> impl Iterator<Item = &Object> {
        self.objects.values().map(|o| &**o)
    }

    pub fn iter(&self) -> impl Iterator<Item = (&ObjectId, &Object)> {
        self.objects.iter().map(|(id, o)| (id, &**o))
    }

    /// Moves every object of `other` here (boxes moved, not copied).
    pub fn append(&mut self, other: &mut ObjectCache) {
        self.objects.append(&mut other.objects);
    }

    /// About how much RAM the objects take, the cache's own share included
    /// (see [`Object::ram_bytes`]).
    pub fn ram_bytes(&self) -> u64 {
        self.objects.values().map(|o| Self::entry_bytes(o)).sum()
    }

    /// About how much RAM `object` takes in a cache: itself, its box's
    /// allocation, its key and pointer in a node (nodes are partly empty:
    /// counted at half full).
    pub fn entry_bytes(object: &Object) -> u64 {
        let slot = std::mem::size_of::<ObjectId>() + std::mem::size_of::<Box<Object>>();
        object.ram_bytes() + 16 + 2 * slot as u64
    }

    pub fn len(&self) -> usize {
        self.objects.len()
    }

    pub fn is_empty(&self) -> bool {
        self.objects.is_empty()
    }

    pub fn retain(&mut self, mut keep: impl FnMut(&ObjectId, &mut Object) -> bool) {
        self.objects.retain(|id, o| keep(id, o));
    }

    pub fn values_mut(&mut self) -> impl Iterator<Item = &mut Object> {
        self.objects.values_mut().map(|o| &mut **o)
    }
}

impl Extend<(ObjectId, Object)> for ObjectCache {
    fn extend<T: IntoIterator<Item = (ObjectId, Object)>>(&mut self, iter: T) {
        self.objects
            .extend(iter.into_iter().map(|(id, o)| (id, Box::new(o))));
    }
}

impl FromIterator<(ObjectId, Object)> for ObjectCache {
    fn from_iter<T: IntoIterator<Item = (ObjectId, Object)>>(iter: T) -> Self {
        Self {
            objects: iter.into_iter().map(|(id, o)| (id, Box::new(o))).collect(),
        }
    }
}

impl IntoIterator for ObjectCache {
    type Item = (ObjectId, Object);
    type IntoIter = std::iter::Map<
        std::collections::btree_map::IntoIter<ObjectId, Box<Object>>,
        fn((ObjectId, Box<Object>)) -> (ObjectId, Object),
    >;

    fn into_iter(self) -> Self::IntoIter {
        self.objects.into_iter().map(|(id, o)| (id, *o))
    }
}
