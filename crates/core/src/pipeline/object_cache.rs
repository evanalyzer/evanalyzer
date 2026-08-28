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
#[derive(Default)]
pub struct ObjectCache {
    objects: BTreeMap<ObjectId, Object>,
}

impl ObjectCache {
    pub fn insert(&mut self, id: ObjectId, object: Object) -> Option<Object> {
        self.objects.insert(id, object)
    }

    pub fn get(&self, id: &ObjectId) -> Option<&Object> {
        self.objects.get(id)
    }

    pub fn get_mut(&mut self, id: &ObjectId) -> Option<&mut Object> {
        self.objects.get_mut(id)
    }

    pub fn remove(&mut self, id: &ObjectId) -> Option<Object> {
        self.objects.remove(id)
    }

    pub fn values(&self) -> impl Iterator<Item = &Object> {
        self.objects.values()
    }

    pub fn iter(&self) -> impl Iterator<Item = (&ObjectId, &Object)> {
        self.objects.iter()
    }

    pub fn len(&self) -> usize {
        self.objects.len()
    }

    pub fn is_empty(&self) -> bool {
        self.objects.is_empty()
    }
}

impl Extend<(ObjectId, Object)> for ObjectCache {
    fn extend<T: IntoIterator<Item = (ObjectId, Object)>>(&mut self, iter: T) {
        self.objects.extend(iter);
    }
}

impl FromIterator<(ObjectId, Object)> for ObjectCache {
    fn from_iter<T: IntoIterator<Item = (ObjectId, Object)>>(iter: T) -> Self {
        Self {
            objects: BTreeMap::from_iter(iter),
        }
    }
}

impl IntoIterator for ObjectCache {
    type Item = (ObjectId, Object);
    type IntoIter = std::collections::btree_map::IntoIter<ObjectId, Object>;

    fn into_iter(self) -> Self::IntoIter {
        self.objects.into_iter()
    }
}
