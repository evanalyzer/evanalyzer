# Performance & RAM Review

Findings from a review of `crates/core` and `crates/gui` against execution-time and
RAM bottlenecks, with a focus on staying smooth on large microscopy images.

This is a findings list, nothing has been changed yet. Check the box next to any
item you want implemented, and optionally add a note, then hand the file back.

---

## Top priority (highest impact)

- [x] **Pipeline deep-clones the whole image on every run** — [`crates/core/src/pipeline/pipeline.rs:106`](crates/core/src/pipeline/pipeline.rs#L106)
  `initial_image.as_ref().clone()` clones the pixel buffer inside the `Arc`, not just the `Arc` handle. For a 4096×4096 f32 tile that's 64MB+ copied per pipeline run, multiplied by the number of chained pipelines per tile ([`job_executor.rs:530-543`](crates/core/src/job/job_executor.rs#L530)).
  *Fix:* share via `Arc`/`Cow`, only deep-copy on first mutation.
  *Done:* investigated first — confirmed the cache genuinely retains its own `Arc` per channel (so isolation between pipelines sharing an input is real and had to be preserved), but found only 1 of 15 image-transform commands (`rolling_ball.rs`) mutates `ctx.image` in place; 3 more turned up during implementation (`hessian.rs`, `intensity_transform.rs`, `enhance_contrast.rs`) that weren't caught by the initial grep. Everything else uses the existing scratch+swap pattern, which becomes a free `Arc` handle exchange. `PipelineContext.image`/`scratch_pad` are now `Arc<ImageContainer>`; the one true mutation path goes through `Arc::make_mut` (clones only when the cache/another pipeline still holds a reference). Read-only pipelines (segmentation + measurement, e.g. Cellpose → ExtractRois) now pay **zero** clones — measured 24.6ms → 20ns per run on a 4096×4096 tile. Two new tests directly prove the mechanism: one confirms a read-only pipeline shares the cache's `Arc` (`Arc::ptr_eq`), the other confirms a mutating pipeline clones before writing and leaves the cached copy untouched. Touched ~20 files; full suite green (277 core / 280 with `--features ai` / 78 gui).

- [x] **AI models reload from disk on every call** — [`cellpose.rs:86`](crates/core/src/algos/ai_segmentation/cellpose.rs#L86), [`stardist.rs:81`](crates/core/src/algos/ai_segmentation/stardist.rs#L81), [`unet.rs:109`](crates/core/src/algos/ai_segmentation/unet.rs#L109)
  Every tile/image re-reads and re-parses the TorchScript file and re-uploads weights. Dominates wall-clock time for any batch/tiled AI job.
  *Fix:* cache the loaded `CModule` (e.g. in `PipelineCache`, keyed by model path), reload only when the path changes.
  *Done:* thread-local cache (`ai_segmentation/model_cache.rs`) keyed by model path, wired into all three algos. 3 unit tests on the cache logic; all 275 tests pass with `--features ai`.

- [x] **Image decode allocates 2-3x the needed buffers** — [`image_reader.rs:786-855`](crates/core/src/image/image_reader.rs#L786)
  Planar RGB/RGBA tiles: full `raw_f32` buffer, then a *second* full-size buffer built via a single-threaded per-pixel `push` loop, on top of the original byte buffer.
  *Fix:* parallelize the planar→interleaved step with rayon, or write directly into the final buffer instead of two intermediate `Vec`s.
  *Done:* planar RGB/RGBA now decodes straight into the final interleaved buffer (alpha plane bytes never read for RGBA). Measured on a 4096×4096 RGB16 tile: peak float memory 384→192 MiB (50% less), wall time 113ms→17ms (6.6x faster, after restoring the parallel decode on the new path). New tests pin exact output; existing JNI-backed tests still pass.

- [x] **Z-projection reallocates a buffer per Z-slice** — [`image_reader.rs:677-727`](crates/core/src/image/image_reader.rs#L677)
  Each Z slice in sum/max/min/avg projection gets a fresh decode buffer that's discarded right after use.
  *Fix:* reuse one scratch buffer across the Z loop, resize once.
  *Done:* one scratch buffer per channel, reused across all Z-slices (resized once). Measured allocator-churn saving: ~30x (8.7ms→0.29ms for 20 reads of an 8MB tile), mostly from skipping the repeated memset. All Z-projection tests (max/min/avg/sum) still pass against real reference data.

- [x] **Morphology kernels are O(k²) when they could be O(k)** — [`morph_ops.rs:74-196`](crates/core/src/extlibs/libmorphology/morph_ops.rs#L74)
  Box/cross structuring elements are separable but scanned as full 2D kernels with bounds-checked `get_pixel` in the innermost loop. Likely the biggest CPU cost in the morphology path (Open/Close runs dilate+erode back to back, kernel sizes up to 27).
  *Fix:* add a separable fast path (two 1D passes) for Box/Cross; keep full scan only for Ellipse; use raw slice indexing instead of `get_pixel`.
  *Done:* Box uses the Minkowski-sum decomposition (two chained 1D passes); Cross uses the union decomposition (two independent 1D passes + elementwise combine) — different math, same O(k) result. Ellipse still uses the full 2D scan. Cross-validated against the original O(k²) scan across 5 image sizes × 4 kernel sizes × both shapes × all 5 padding modes × dilate/erode (400+ exact-equality comparisons) plus a multichannel RGB case, all passing, before wiring it in. Measured on a 2048×2048 u8 image, Open with k=27: Box 931ms→33.7ms (27.6x), Cross 311ms→33.6ms (9.2x, lower because the naive cross mask was already sparse).

- [x] **Repeated full-image clones in math ops** — [`image_cache.rs:91`](crates/core/src/algos/math/image_cache.rs#L91), [`image_math.rs:120`](crates/core/src/algos/math/image_math.rs#L120), [`median_subtract.rs:57`](crates/core/src/algos/math/median_subtract.rs#L57)
  Same pattern as the pipeline clone, but in the algorithm layer — full pixel-buffer clones on every call where a borrow or buffer-swap would do (pattern already used correctly in `blur.rs` / `structure_tensor.rs`).
  *Done:* fixed as a byproduct of the pipeline `Arc` refactor above — all three `.clone()` calls now clone an `Arc<ImageContainer>` (cheap) instead of an `ImageContainer` (deep pixel copy). Traced `median_subtract.rs`'s snapshot-then-subtract composition (RankFilter → ImageMath) by hand to confirm `Arc::make_mut`'s copy-on-write still produces the correct result when the scratchpad aliases the main image; confirmed by the passing test suite.

- [x] **Viewport cache does a linear scan on every tile miss** — [`viewport_cache.rs:341-507`](crates/gui/src/editor/viewport_cache.rs#L341)
  `find_in_cache` walks the entire cache (up to 1GB/256MB of tiles) for spatial matches. Runs on the **undebounced** low-res pan/zoom path — i.e. every mouse-drag tick.
  *Fix:* index cached tiles by `(series, level, t, z)` so the scan is restricted to a small candidate set.
  *Done:* added `TileGroupIndex`, a `HashMap<(series, level, t, z_projection, z_range), Vec<TileKey>>` kept alongside each `CLruCache` in a new `IndexedTileCache` wrapper. `clru` has no eviction callback, so the index resyncs fully whenever an insert doesn't grow the cache by exactly one (covers both eviction and key-replace); a `cache.get()` re-check before trusting any candidate guards against any residual staleness regardless. 6 new unit tests cover exact-match, spatial-superset-within-group, cross-group isolation, candidate-set scoping with many groups cached, eviction safety, and clear. Measured: 60 T/Z groups × 10 tiles/group, 20k miss lookups — 56ms → 0.74ms (75.6x).

- [ ] **GUI preview clones the entire project on every parameter tweak** — [`pipelines_controller.rs:1091`](crates/gui/src/editor/pipelines_controller.rs#L1091)
  `project.clone()` copies all pipelines/images/ROIs/settings just to preview one image, partially defeating the existing 400ms debounce.
  *Fix:* build a minimal single-image settings struct directly, or share the non-image parts via `Arc`.

- [ ] **Canny is unparallelized and over-allocates** — [`edge_detection_canny.rs:106-156`](crates/core/src/algos/filters/edge_detection_canny.rs#L106)
  4 fresh full-frame buffers allocated per call; every stage (except hysteresis DFS) is per-pixel parallel but runs single-threaded.
  *Fix:* reusable scratch buffers, `par_iter_mut` for magnitude/direction/threshold, row-wise parallel NMS.

- [ ] **Hessian wastes half its compute and is unparallelized** — [`hessian.rs:113-164`](crates/core/src/algos/filters/hessian.rs#L113)
  Calls the gradient function 4 times; 3 outputs are discarded immediately (~half the convolution work wasted). Final eigenvalue loop is sequential, unlike the near-identical code in `structure_tensor.rs` which already uses `par_iter_mut`.

---

## Medium priority

- [x] **Correctness bug**: `ImageContainer::clone_empty` builds the wrong variant for `U32` — [`image_reader.rs:117-125`](crates/core/src/image/image_reader.rs#L117). Builds an `F32Rgb`-shaped buffer instead of matching the source's channel count — likely an oversized allocation downstream, not just a perf issue.
  *Done:* now returns a matching `U32` buffer. Regression test written first (confirmed it failed against the old code — was returning `F32Rgb{width,height,channels:3}`), then fixed; all tests pass.
- [ ] RGBA→RGB strip allocates a second full buffer via `flat_map`/`collect` instead of in-place compaction — [`image_reader.rs:823-828`](crates/core/src/image/image_reader.rs#L823).
- [ ] `sync_channel::<PipelineCache>(4)` in job executor is a flat buffer count, not scaled to tile size — could hold several hundred MB extra in flight for large tiles. [`job_executor.rs:502,757`](crates/core/src/job/job_executor.rs#L502)
- [ ] `pipeline_cache.rs` `Scratchpad` allocates a fresh zero-filled buffer on every request instead of reusing one — [`pipeline_cache.rs:76-119`](crates/core/src/pipeline/pipeline_cache.rs#L76)
- [ ] Sequential per-pixel passes not using rayon, unlike sibling filters that already do:
  - [`color_filter.rs:126-157`](crates/core/src/algos/filters/color_filter.rs#L126) (RGB→HSV via bounds-checked `get_pixel`/`set_pixel`)
  - [`enhance_contrast.rs:206-286`](crates/core/src/algos/filters/enhance_contrast.rs#L206) and [`intensity_transform.rs:130-152`](crates/core/src/algos/filters/intensity_transform.rs#L130) (histogram + LUT passes)
  - [`rolling_ball.rs:99-190`](crates/core/src/algos/filters/rolling_ball.rs#L99) (erosion/dilation passes, row-independent)
- [ ] `cellpose.rs` `follow_flows` per-pixel Euler integration (up to 1000 iterations) is unparallelized despite independent pixel trajectories — [`cellpose.rs:252-269`](crates/core/src/algos/ai_segmentation/cellpose.rs#L252)
- [ ] `stardist.rs` non-max-suppression is O(n²) candidate comparisons — could reuse the spatial-grid approach already implemented in `voronoi.rs` — [`stardist.rs:368-411`](crates/core/src/algos/ai_segmentation/stardist.rs#L368)
- [ ] GUI: `visible_channels.contains()` checked inside the innermost per-pixel loop, turning O(pixels×channels) into O(pixels×channels²) — [`viewport_worker.rs:557-577`](crates/gui/src/editor/viewport_worker.rs#L557)
- [ ] GUI: screen-space compositing loop is a serial ~8M-iteration loop per 4K frame while neighboring code uses rayon — [`viewport_worker.rs:387-406`](crates/gui/src/editor/viewport_worker.rs#L387)
- [ ] GUI: `arc-swap` is a declared dependency but never used — all shared state goes through one coarse `Arc<RwLock<ProjectWithRuntime>>`, so a long write (folder scan, pipeline execution) can stall UI reads. Either wire it up for read-mostly snapshots or drop the dependency.
- [ ] GUI: `composite_roi_instances` allocates a full viewport-sized buffer and always runs a full border scan even when no ROIs are visible — [`viewport_controller.rs:435-507,824-844`](crates/gui/src/editor/viewport_controller.rs#L435)
- [ ] CSV/DuckDB exporters re-scan the full ROI set to discover channel/class schema on every `export()` call instead of caching it once per job — [`storage/file.rs:48-61`](crates/core/src/storage/file.rs#L48), [`storage/duckdb.rs:223-244`](crates/core/src/storage/duckdb.rs#L223)

---

## Low priority / polish

- [ ] `MemoryExporter` clones each ROI twice on export (temp `Vec`, then again inside `to_roi_settings()`) — [`storage/memory.rs:24-37`](crates/core/src/storage/memory.rs#L24)
- [ ] `rank_filter.rs` median/outlier filter uses naive per-pixel sorting instead of a sliding histogram; has an unimplemented `// For parallel loop` hook — [`rank_filter.rs:149-218`](crates/core/src/algos/filters/rank_filter.rs#L149)
- [ ] `Roi::measure_intensities`/`overlaps` are scalar nested loops, not parallelized (low severity — ROI bboxes are usually small) — [`roi.rs:634-755`](crates/core/src/roi.rs#L634)
- [ ] GUI: mouse-move handler isn't throttled, unlike other UI update paths — [`viewport_image_controller.rs:317-392`](crates/gui/src/editor/viewport_image_controller.rs#L317)
- [ ] GUI: command-picker filter rebuilds 5 `Vec`s on every keystroke with no debounce — low impact while the command list stays small — [`pipelines_controller.rs:1465-1502`](crates/gui/src/editor/pipelines_controller.rs#L1465)
- [ ] `save_image.rs` full-buffer `map().collect()` per save call is sequential — low severity, debug/checkpoint path only — [`save_image.rs:78-158`](crates/core/src/algos/math/save_image.rs#L78)

---

## Already good — no action needed

Connected components (proper O(N) union-find), watershed/maximum-finder (faithful O(N) ImageJ port), EDM (O(N) two-pass), `structure_tensor.rs`/`weighted_deviation.rs` (rayon-parallel, buffer-swap not clone), Voronoi/coloc/ROI extraction (sparse/bbox-local with a real spatial grid), JNI tile reads (zero-copy `DirectByteBuffer`), DuckDB export (streaming `Appender`), OME-XML parsing (streaming `quick_xml`, not DOM), the `clru` viewport caches (correctly weight-bounded), and the GUI worker architecture overall (disk I/O and pixel conversion off the UI thread, locks dropped before slow work, 150ms/immediate debounce split for high/low-res rendering).

---

## Suggested order of attack

1. Pipeline image clone + AI model reload — both simple, surgical, outsized impact on every run.
2. Image decode buffer reduction + morphology separable kernels — RAM and CPU on the hottest paths.
3. Viewport cache indexing + project-clone-on-preview — GUI smoothness.
4. The rest, as time allows.
