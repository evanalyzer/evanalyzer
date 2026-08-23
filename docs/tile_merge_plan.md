# Cross-Tile Object Merging (Whole-Organ / Tile-Spanning Objects)

## Context

Whole-slide images are analyzed in tiles (up to 4096×4096px, `crates/core/src/job/job_executor.rs`). Each tile gets its own `PipelineCache`/`Object` set, processed and exported to the database independently. This is correct for small objects (cells, nuclei, spots) but silently **splits** any object that spans an internal tile boundary into multiple disconnected objects, each with wrong area/intensity/shape, and no flag exists to catch it (`Object::touches_edge` is checked only against the *full image* border, not tile borders — `object.rs`, via `rasterize_geometry`).

The scientist use case that makes this a real requirement, not a hypothetical: measuring a whole organ/tissue region's true area and intensity across an entire whole-slide image, and colocalizing/counting smaller objects (spots) within it. A downsampled/pyramid-level approximation (discussed and rejected earlier) loses exactly the precision needed here. This plan adds an opt-in, project-level feature that detects tile-boundary fragments of the same real-world object and merges them into one correct object after all tiles finish.

## Assessment of the proposed architecture

The user's proposed design — buffer edge-touching object fragments during the per-tile DB-write loop, and resolve them in one batch pass after all tiles finish — is sound and is a legitimate, simpler variant of the standard technique, not something invented from scratch:

- It correctly narrows the "at risk" set to only objects that touch a tile boundary (not the full image edge), instead of trying to synchronize adjacent tiles live during the parallel loop, which the current rayon `par_iter`-based tile loop offers no ordering/adjacency guarantees for anyway.
- Deferring resolution to a single end-of-image batch pass, rather than incremental online merging, avoids concurrency/synchronization complexity entirely — a fragment can only be safely finalized once every tile that could touch it has run, which in practice means waiting for the whole image regardless of approach, so batching costs little extra and is much simpler to implement correctly.

**Industry standard**: this is a direct generalization of the classical **two-pass connected-component labeling algorithm** (Rosenfeld & Pfaltz, 1966), which resolves "these two provisional labels are actually the same component" via a union-find/equivalence table — normally applied at raster-scan-line granularity. The tiled/blocked variant (process each block independently, then reconcile object identity across block boundaries via the same union-find idea) is a well-documented technique in distributed/out-of-core image processing — used in large-scale microscopy pipelines (BigStitcher/N5/Zarr-based tooling), distributed connected-components in geospatial raster processing, and GPU/parallel CCL implementations. So: **not novel** — "buffer boundary candidates → build an adjacency graph → union-find into merge-groups → recompute merged stats" is exactly the standard pattern, just implemented as an end-of-batch pass instead of incremental online resolution.

## Key facts from this codebase (grounding the plan)

- `Object` (`crates/core/src/object.rs:35-95`): `id: ObjectId`, `bbox: [u32;4]` (inclusive), `mask_data: BitVec<u64,Lsb0>` (row-major, relative to bbox), `area: usize`, `plane: ImagePlane`, `object_class: HashSet<ObjectClass>`, `intensities: IndexMap<i32, Intensity>` (sum/min/max/avg per channel), moment accumulators (`sum_x/sum_y/sum_x2/sum_y2/sum_xy`), `touches_edge: bool`. Geometry (`perimeter`, `ellipse`) is **eagerly computed once** by `finalize_geometry()` inside `Object::new` (`object.rs:285-288`) — constructing a new `Object` from a merged mask via the existing `Object::new(ObjectInit{..})` constructor automatically produces correct perimeter/circularity/solidity/eccentricity/Feret for the merged shape. No manual geometry recomputation needed.
- `Object::overlaps()` (`object.rs:710`): bbox-intersection fast path + windowed mask scan. The new fragment-adjacency check (touching, not overlapping) is a small variant of this same pattern.
- `PipelineResultExporter::export(&self, cache: &PipelineCache)` (`storage.rs:7-31`) only accepts a whole `PipelineCache`, never individual objects. `finalize_image(&self, image_rel_path, error)` is called once per image after all of that image's tiles are exported — this is the natural hook point to run the merge pass and export merged objects via a small synthetic `PipelineCache` (image_cache/meta + just the merged objects).
- `PipelineImageMeta` (carried on `PipelineContext`) has both `image_tile_info: ImageTile` (`offset_x, offset_y, width, height`) and `full_image_width: ImageSize` already available at the point each tile's `PipelineCache` is finalized in `job_executor.rs` — enough to compute "touches this tile's own edge" without new plumbing.
- `BboxGrid` (`crates/core/src/spatial_grid.rs`, `pub(crate)`, `build(ids, cache)` / `candidates(bbox)`) — reusable for the fragment-adjacency prefilter by building a small throwaway `PipelineCache` wrapping just the buffered fragments.
- Both `analyze_image` (multi-image batch path) and `analyze_image_tiles_parallel` (single large-image path) do their own internal tiling — the buffering/merge logic needs to be shared by both, not just one.
- Settings pattern: per-class `Vec<ObjectClass>` allow-lists are the established shape (`ColocalizationSettings.classes_to_coloc`, etc. in `crates/cfg/src/modules/pipeline_command_settings.rs`). `ProjectSettings` (`crates/cfg/src/modules/project_settings.rs:13-36`) is the right home for a *project-level* (not per-pipeline-step) toggle, as a new peer field alongside `classification`/`plate`/`pipelines`.

## Implementation Plan

### 1. New project-level settings
Add to `crates/cfg/src/modules/project_settings.rs`: new field `pub tile_merge: TileMergeSettings` on `ProjectSettings`, plus a new struct (new file or inline, following `ColocalizationSettings`'s shape):
```rust
pub struct TileMergeSettings {
    pub enabled: bool,
    pub classes_to_merge: Vec<ObjectClass>,   // opt-in allow-list, default empty/disabled
    pub connectivity: TileMergeConnectivity,  // 4- or 8-connected boundary adjacency
}
```
`#[serde(default)]` so old project files load with the feature off. Surface in the GUI project-settings screen alongside existing class/pipeline settings (out of scope to design the UI here beyond noting where it plugs in).

### 2. Distinguish tile-edge from image-edge touching
In `crates/core/src/object.rs` (or at the call site that already knows tile bounds), add a way to test an object's bbox against a *tile's* bounds (`ImageTile`) separately from the existing full-image `touches_edge`. This does not need to be a new field on `Object` — it can be a free function/method taking the object's bbox and the tile's `ImageTile`, used only at the job_executor merge-decision point.

### 3. Thread a shared fragment buffer through the tile loop
In `job_executor.rs`, both `analyze_image` and `analyze_image_tiles_parallel`:
- Before the tile `par_iter`/`try_for_each` loop, if `tile_merge.enabled`, construct `let pending: Arc<Mutex<Vec<PendingFragment>>> = Arc::new(Mutex::new(Vec::new()));` (one per image, not global — merging is per-image).
- At the point each tile's `PipelineCache.object_cache` is finalized (right before `cache_tx.try_send(cache)`), split it: for each object whose class is in `tile_merge.classes_to_merge` AND touches this tile's own edge (step 2) AND does *not* touch the full image edge, remove it from `cache.object_cache` and push a `PendingFragment { object, tile: ImageTile, plane: ImagePlane }` onto `pending`. Everything else keeps flowing through the existing per-tile export path unchanged.
- `PendingFragment` needs enough to place the fragment in absolute image space and know which tile it came from (for the adjacency check in step 4); `Object.bbox` is already absolute, so mostly this is `(Object, ImageTile)`.

### 4. Match + merge pass, run per image at `finalize_image` time
New function, e.g. `crate::job::tile_merge::merge_pending_fragments(pending: Vec<PendingFragment>, settings: &TileMergeSettings) -> Vec<Object>`:
- Group fragments by `(ObjectClass, plane)`.
- Build a throwaway `PipelineCache` wrapping the group's fragments and a `BboxGrid` over it (reuse `spatial_grid::BboxGrid`) to prefilter candidate pairs whose bboxes are near each other.
- For each candidate pair from *different* tiles whose bboxes are adjacent (not necessarily overlapping — touching across the shared tile seam), do an exact adjacency check: walk the shared border strip and test for a foreground pixel in fragment A immediately next to (4- or 8-connected, per `settings.connectivity`) a foreground pixel in fragment B — a small variant of `Object::overlaps()`'s windowed mask scan.
- Union matched pairs via a union-find (plain `Vec<usize>` parent array, path compression) to get merge-groups.
- For each group of size 1: leave the fragment alone (return it unchanged — it touched a tile edge but had no real neighbor, e.g. hit the image edge diagonally).
- For each group of size > 1: compute the union bbox, allocate one `BitVec` mask sized to it, OR each fragment's mask in at its offset, sum `area`/`intensities` across fragments (same additive accumulators the codebase already uses for per-object stats), and construct the merged object via **`Object::new(ObjectInit { .. })`** so `finalize_geometry()` recomputes perimeter/circularity/solidity/eccentricity/Feret correctly from the real merged shape — no separate geometry-merging math needed.
- Assign fresh `ObjectId`s to merged objects (they're new logical objects, not any one fragment's id).

### 5. Export merged objects
In `job_executor.rs`, right before calling `exporter.finalize_image(image_rel_path, ..)` for an image, if `pending` is non-empty: run step 4, wrap the resulting `Vec<Object>` in a synthetic `PipelineCache` (reuse the image's own `image_cache`/`image_meta` for pixel-size scaling, empty otherwise), and call `exporter.export(&synthetic_cache)` before `finalize_image`.

### 6. Downstream colocalization — no change needed for pixel-level containment
Spot-vs-organ colocalization via `ClassifyObjects.overlapping_with` / `Colocalization` against the organ's **segmentation class** (pixel labels, not the `Object`) already works correctly per-tile today, since that classification is a local per-pixel operation — it does not need to wait for the merge pass. Only a downstream need to group counts by *which specific organ instance* (if a slide has multiple distinct organ regions) would need to join against the merged objects' IDs after step 5 — call this out as a known follow-up, not part of this plan's scope.

### 7. Guardrails
- A safety cap on fragment count per merge-group (mirror `ExtractObjects.max_objects_before_fail`'s pattern) so a misconfigured `classes_to_merge` (e.g. accidentally including a class with thousands of small tile-edge-touching objects) fails loudly instead of building a pathological merge.
- `tile_merge.enabled = false` (default) must be a complete no-op: zero behavior change to the existing per-tile export path when the feature is off.

## Files touched
- `crates/cfg/src/modules/project_settings.rs` — new `TileMergeSettings` + field on `ProjectSettings`.
- `crates/core/src/object.rs` — tile-edge-touch helper; reuse existing `Object::new`/`finalize_geometry` for merged objects (no structural change needed there).
- `crates/core/src/job/job_executor.rs` — fragment buffering in `analyze_image` and `analyze_image_tiles_parallel`; call the merge pass and export before `finalize_image`.
- New `crates/core/src/job/tile_merge.rs` (or similar) — the match/merge algorithm itself (adjacency check, union-find, merged-object construction), reusing `spatial_grid::BboxGrid`.
- `crates/core/src/storage.rs`/`duckdb.rs` — none expected; merged objects flow through the existing `export`/`PipelineCache` path.

## Verification
- Unit tests for the adjacency/union-find logic in `tile_merge.rs` directly (construct synthetic fragments with known adjacency, assert correct grouping) — same style as this session's `coloc_objects.rs`/`spatial_grid.rs` tests (deterministic RNG fuzz + handcrafted edge cases: fragments touching at exactly one pixel, fragments from >2 tiles meeting at a 4-tile corner, fragments that touch the image edge and must NOT be buffered).
- An end-to-end `JobExecutor` test with a synthetic multi-tile image containing one object deliberately spanning 2-4 tiles, asserting the exported object count/area/intensity matches what a single-tile (untiled) run of the same image produces — the strongest correctness check, mirroring this session's cross-check pattern (compare tiled-merged result against an untiled reference).
- Confirm `tile_merge.enabled = false` reproduces byte-identical results to today's behavior (regression guard against accidentally changing the default path).
