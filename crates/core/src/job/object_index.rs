//! Lightweight, always-resident metadata for every object across a whole
//! image, built once from the per-tile scratch files ([`TileObjectStore`])
//! written during the tile loop.
//!
//! The point of this index is to let the whole-image phase (tile-merge,
//! then any `ExecutionScope::WholeImage` command) answer "who's spatially
//! near whom" and "did these two come from the same tile" across every
//! object the image produced, *without* requiring every object's full data
//! (mask, per-pixel intensities - the expensive, large part) to stay
//! resident in RAM at once. Building it costs one full read-and-deserialize
//! pass over every tile's scratch file - unavoidable with the flat per-tile
//! bincode format, there's no way to read just the metadata - but the
//! result discards each object's mask/intensities immediately after
//! extracting what's needed here, so the index itself stays cheap in
//! memory afterward. Retrieving an object's *full* data again later is a
//! separate, later concern (the paged/LRU cache behind `ObjectCache`).

use super::object_scratch::TileObjectStore;
use crate::ImagePlane;
use evanalyzer_cfg::core_types::{InternalErrors, ObjectClass, ObjectId};
use std::collections::{HashMap, HashSet};

/// Everything a spatial query (`spatial_grid::BboxGrid::build_from_bboxes`)
/// or the "never merge two objects from the same tile" rule needs about one
/// object, without its mask or intensities.
pub(crate) struct ObjectMetadata {
    pub(crate) tile_index: usize,
    pub(crate) bbox: [u32; 4],
    pub(crate) object_class: HashSet<ObjectClass>,
    pub(crate) plane: ImagePlane,
}

pub(crate) struct WholeImageIndex {
    entries: HashMap<ObjectId, ObjectMetadata>,
}

impl WholeImageIndex {
    /// Reads every tile from `0..tile_file_count` out of `store` once,
    /// keeping only each object's lightweight metadata. `tile_file_count`
    /// is the number of scratch files actually written - the caller's own
    /// running count from `object_scratch`, not necessarily the image's
    /// spatial tile count (an image with multiple z/t-stacks writes more
    /// scratch files than it has spatial tiles).
    pub(crate) fn build(
        store: &TileObjectStore,
        tile_file_count: usize,
    ) -> Result<Self, InternalErrors> {
        let mut entries = HashMap::new();
        for tile_index in 0..tile_file_count {
            let (_, objects) = store.read_tile(tile_index)?;
            for object in objects {
                entries.insert(
                    object.id.clone(),
                    ObjectMetadata {
                        tile_index,
                        bbox: object.bbox,
                        object_class: object.object_class.clone(),
                        plane: object.plane,
                    },
                );
            }
        }
        Ok(Self { entries })
    }

    pub(crate) fn get(&self, id: &ObjectId) -> Option<&ObjectMetadata> {
        self.entries.get(id)
    }

    pub(crate) fn len(&self) -> usize {
        self.entries.len()
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub(crate) fn iter(&self) -> impl Iterator<Item = (&ObjectId, &ObjectMetadata)> {
        self.entries.iter()
    }

    /// Every entry's `(id, bbox)`, the shape `spatial_grid::BboxGrid::build_from_bboxes`
    /// needs to index the whole image without touching any object's full data.
    pub(crate) fn all_bboxes(&self) -> Vec<(ObjectId, [u32; 4])> {
        self.entries
            .iter()
            .map(|(id, meta)| (id.clone(), meta.bbox))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ImageTile;
    use crate::object::{Object, ObjectInit};
    use bitvec::prelude::*;
    use evanalyzer_cfg::core_types::ObjectId;

    fn tile(offset_x: usize, offset_y: usize, width: usize, height: usize) -> ImageTile {
        ImageTile {
            offset_x,
            offset_y,
            width,
            height,
        }
    }

    fn rect_object(id: u128, bbox: [u32; 4], class: ObjectClass) -> Object {
        let [x0, y0, x1, y1] = bbox;
        let w = (x1 - x0 + 1) as usize;
        let h = (y1 - y0 + 1) as usize;
        let mut object = Object::new(ObjectInit {
            id: ObjectId(id),
            bbox,
            mask_data: BitVec::<u64, Lsb0>::repeat(true, w * h),
            area: w * h,
            ..Default::default()
        });
        object.add_object_class(class);
        object
    }

    #[test]
    fn build_indexes_every_object_across_every_written_tile() {
        let store = TileObjectStore::new().expect("scratch dir must be creatable");
        store
            .write_tile(
                0,
                tile(0, 0, 10, 10),
                vec![rect_object(1, [1, 1, 4, 4], ObjectClass::Valid(1))],
            )
            .expect("write tile 0");
        store
            .write_tile(
                1,
                tile(10, 0, 10, 10),
                vec![rect_object(2, [11, 1, 14, 4], ObjectClass::Valid(2))],
            )
            .expect("write tile 1");

        let index = WholeImageIndex::build(&store, 2).expect("index must build");

        assert_eq!(index.len(), 2);
        let meta1 = index.get(&ObjectId(1)).expect("object 1 must be indexed");
        assert_eq!(meta1.tile_index, 0);
        assert_eq!(meta1.bbox, [1, 1, 4, 4]);
        assert_eq!(meta1.object_class, [ObjectClass::Valid(1)].into());

        let meta2 = index.get(&ObjectId(2)).expect("object 2 must be indexed");
        assert_eq!(meta2.tile_index, 1);
        assert_eq!(meta2.bbox, [11, 1, 14, 4]);
    }

    #[test]
    fn build_over_zero_tiles_is_empty() {
        let store = TileObjectStore::new().expect("scratch dir must be creatable");
        let index = WholeImageIndex::build(&store, 0).expect("index must build");
        assert!(index.is_empty());
    }

    #[test]
    fn all_bboxes_matches_every_indexed_entry() {
        let store = TileObjectStore::new().expect("scratch dir must be creatable");
        store
            .write_tile(
                0,
                tile(0, 0, 10, 10),
                vec![
                    rect_object(1, [1, 1, 4, 4], ObjectClass::Valid(1)),
                    rect_object(2, [5, 5, 8, 8], ObjectClass::Valid(1)),
                ],
            )
            .expect("write tile 0");

        let index = WholeImageIndex::build(&store, 1).expect("index must build");
        let mut bboxes = index.all_bboxes();
        bboxes.sort_by_key(|(id, _)| id.0);

        assert_eq!(
            bboxes,
            vec![(ObjectId(1), [1, 1, 4, 4]), (ObjectId(2), [5, 5, 8, 8])]
        );
    }
}
