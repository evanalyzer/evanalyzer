//! Temporary per-tile object storage for the whole-image command phase.
//!
//! `job_executor` runs each tile's `ExecutionScope::Tile` commands and, for
//! any image split into more than one tile, hands that tile's resulting
//! objects to a [`TileObjectStore`] instead of only keeping them in the
//! per-tile in-memory cache. This is what lets the later whole-image phase
//! (tile-merge, then any `ExecutionScope::WholeImage` command) see every
//! tile's objects without requiring the whole image's object population to
//! stay resident in RAM for the entire tile loop - see the discussion in
//! `docs/tile_merge_plan.md`.
//!
//! The on-disk format has no stability requirements: files are written and
//! read back within the same run and deleted once that image's whole-image
//! phase finishes (`TileObjectStore` owns a `tempfile::TempDir` and cleans
//! up on drop), so nothing here is ever read by a different version of the
//! app the way the DuckDB results schema must be.

use crate::ImageTile;
use crate::object::Object;
use evanalyzer_cfg::core_types::InternalErrors;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// One tile's worth of extracted objects, persisted together with the tile
/// they came from - needed later to determine cross-tile fragment adjacency
/// the same way `tile_merge::PendingFragment` does today.
#[derive(Serialize, Deserialize)]
struct TileObjectRecord {
    tile: ImageTile,
    objects: Vec<Object>,
}

/// Scratch directory holding one image's per-tile object files. Created
/// lazily by [`TileObjectStore::new`] and removed wholesale once the store
/// is dropped - callers should keep it alive for exactly as long as an
/// image's whole-image phase needs to read tiles back, and no longer.
pub(crate) struct TileObjectStore {
    dir: tempfile::TempDir,
}

impl TileObjectStore {
    /// Creates the backing scratch directory. Returns `Err` only if the OS
    /// can't create a temp directory (disk full, permissions) - callers
    /// should treat that as fatal for the multi-tile whole-image path
    /// rather than silently falling back to "no objects found".
    pub(crate) fn new() -> Result<Self, InternalErrors> {
        let dir = tempfile::Builder::new()
            .prefix("evanalyzer-tile-objects-")
            .tempdir()
            .map_err(|e| {
                InternalErrors::Io(format!("Failed to create tile-object scratch dir: {e}"))
            })?;
        Ok(Self { dir })
    }

    fn path_for(&self, tile_index: usize) -> PathBuf {
        self.dir.path().join(format!("tile_{tile_index:06}.bin"))
    }

    /// Serializes `objects` (with their originating `tile`) to this tile's
    /// scratch file. `tile_index` only needs to be unique within this
    /// store - the real tile geometry travels inside the record itself.
    pub(crate) fn write_tile(
        &self,
        tile_index: usize,
        tile: ImageTile,
        objects: Vec<Object>,
    ) -> Result<(), InternalErrors> {
        let record = TileObjectRecord { tile, objects };
        let bytes = bincode::serialize(&record)
            .map_err(|e| InternalErrors::Io(format!("Failed to serialize tile objects: {e}")))?;
        std::fs::write(self.path_for(tile_index), bytes).map_err(|e| {
            InternalErrors::Io(format!("Failed to write tile-object scratch file: {e}"))
        })
    }

    /// Reads back one tile's objects and the tile they came from.
    #[allow(dead_code)] // consumed by the whole-image phase, not wired up yet
    pub(crate) fn read_tile(&self, tile_index: usize) -> Result<(ImageTile, Vec<Object>), InternalErrors> {
        let bytes = std::fs::read(self.path_for(tile_index)).map_err(|e| {
            InternalErrors::Io(format!("Failed to read tile-object scratch file: {e}"))
        })?;
        let record: TileObjectRecord = bincode::deserialize(&bytes).map_err(|e| {
            InternalErrors::Io(format!("Failed to deserialize tile objects: {e}"))
        })?;
        Ok((record.tile, record.objects))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::object::{Object, ObjectInit};
    use bitvec::prelude::*;
    use evanalyzer_cfg::core_types::{ObjectClass, ObjectId};

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
    fn write_then_read_round_trips_the_tile_and_every_object_field() {
        let store = TileObjectStore::new().expect("scratch dir must be creatable");
        let original_tile = tile(20, 0, 20, 20);
        let object = rect_object(1, [21, 1, 24, 4], ObjectClass::Valid(7));

        store
            .write_tile(0, original_tile.clone(), vec![object.clone()])
            .expect("write must succeed");
        let (read_tile, read_objects) = store.read_tile(0).expect("read must succeed");

        assert_eq!(read_tile.offset_x, original_tile.offset_x);
        assert_eq!(read_tile.offset_y, original_tile.offset_y);
        assert_eq!(read_tile.width, original_tile.width);
        assert_eq!(read_tile.height, original_tile.height);

        assert_eq!(read_objects.len(), 1);
        let round_tripped = &read_objects[0];
        assert_eq!(round_tripped.id, object.id);
        assert_eq!(round_tripped.bbox, object.bbox);
        assert_eq!(round_tripped.area, object.area);
        assert_eq!(round_tripped.mask_data, object.mask_data);
        assert_eq!(round_tripped.object_class, object.object_class);
    }

    #[test]
    fn different_tiles_in_the_same_store_do_not_collide() {
        let store = TileObjectStore::new().expect("scratch dir must be creatable");
        let a = rect_object(1, [0, 0, 4, 4], ObjectClass::Valid(1));
        let b = rect_object(2, [0, 0, 4, 4], ObjectClass::Valid(2));

        store
            .write_tile(0, tile(0, 0, 5, 5), vec![a])
            .expect("write tile 0 must succeed");
        store
            .write_tile(1, tile(5, 0, 5, 5), vec![b])
            .expect("write tile 1 must succeed");

        let (tile0, objects0) = store.read_tile(0).expect("read tile 0 must succeed");
        let (tile1, objects1) = store.read_tile(1).expect("read tile 1 must succeed");

        assert_eq!(tile0.offset_x, 0);
        assert_eq!(objects0[0].object_class, [ObjectClass::Valid(1)].into());
        assert_eq!(tile1.offset_x, 5);
        assert_eq!(objects1[0].object_class, [ObjectClass::Valid(2)].into());
    }

    #[test]
    fn reading_a_tile_that_was_never_written_fails_instead_of_panicking() {
        let store = TileObjectStore::new().expect("scratch dir must be creatable");
        assert!(store.read_tile(0).is_err());
    }
}
