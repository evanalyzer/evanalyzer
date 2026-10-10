The instrumentation is removed: the three files match the committed version again and `evanalyzer_core` builds. The main bottleneck is that saving results to DuckDB is serialized: only one image at a time can save. That limits you to about 7 images per second, however many cores you have.

## Measurements
Synthetic workload: 22 images of 4096×4096 pixels, about 11,000 objects each, pipeline Gaussian → Threshold → ConnectedComponents → ExtractObjects, on 22 cores.

| Threads | Wall time | Speed-up |
|---|---|---|
| 1 | 11.1 s | 1× |
| 4 | 4.6 s | 2.4× |
| 11 | 3.0 s | 3.7× |
| 22 | 3.0–4.9 s | no further gain, only about 7 cores busy |

## Bottlenecks, ordered by impact

**1. The DuckDB export runs one image at a time.** This is measured and it is the main limit.

In `JobExecutor::analyze_image`, `exporter.lock().export(...)` and then `finalize_image` both run while one global mutex is held. Each image's export takes about 100–140 ms:

| Part of the export | Time per image | Needs the lock? |
|---|---|---|
| Building the Arrow columns | about 55 ms (45%) | no |
| `append_record_batch` | 8 ms | yes |
| Appender flush | 22 ms | yes |
| Commit (one transaction per image) | about 33 ms | yes |

At 22 threads, an image waits 1.46 s on average for the export lock, plus another 1.15 s before `finalize_image`. The export times add up to about the whole wall time.

Fixes:
- Split the exporter into two steps. A `prepare(&cache) -> RecordBatch` step runs in parallel without the lock; only the append runs under it. This alone cuts the locked time by about 45%.
- Add a dedicated writer thread fed by a bounded channel. It would commit several images per transaction instead of one each, which removes most of the 33 ms commit and 22 ms flush per image. `finalize_image` would move into that writer too.
- Together these should raise the limit from about 7 images/s to roughly 30 or more, after which the compute steps become the limit.

**2. Large buffers are allocated and zeroed per tile and per pipeline.** I checked this in the code; it is not timed separately.

`PipelineContext::new_from_image` allocates three full-size buffers every time a pipeline starts on a tile:
- a zeroed scratch pad (`clone_empty`)
- a zeroed `segmentation_map`
- a zeroed `instance_map`

At 4096², that is about 200 MB written to memory before any real work starts. This fits what we saw: every step runs 2–3× slower in parallel (Gaussian 153 → 357 ms, ExtractObjects 73 → 243 ms), which points to memory bandwidth.

Fix: keep a buffer pool per worker thread and reuse the buffers across tiles, or allocate them only when a step first needs them. `segmentation_map` already has that lazy path in `get_f32_gray_and_segmentation_mask_mut`. The TODO about the scratch pad allocating per request covers this.

**3. Two filters in a row copy a full image.** I confirmed this from reference counts and the code.

After the first filter's `swap()`, `scratch_pad` holds the cached input image, which the image cache also references (2 references). The next filter that writes into it (any of the roughly 15 filters in `algos/filters/*`) then makes a full copy through `Arc::make_mut`: 67 MB per step at 4096², only to overwrite it. Your benchmark pipeline doesn't hit this, because Threshold, ConnectedComponents and ExtractObjects never touch the scratch pad. Typical pipelines like Gaussian → Rolling Ball → Threshold do.

Fix: in `swap()`, if the swapped-out image is still shared (`Arc::strong_count > 1`), replace `scratch_pad` with a fresh or pooled buffer instead of keeping the shared one.

**4. Reading tiles slows down under load.** Measured.

Reading a tile takes 50 ms with 1 thread and 133 ms with 22. Each tile opens its own `ImageReader` and loads all channels, even when the pipelines only use some of them. Worth checking whether only the channels the pipelines and measurements use can be loaded.

**5. Some single steps are slow.**
- Threshold takes 38 ms for 16.7 megapixels, which is slow for a simple comparison per pixel. The loop should be easy to vectorise and parallelise.
- The kornia Gaussian takes 153 ms with a 5×5 kernel. A separable blur working on rows in parallel would likely be 3–5× faster.
- The open performance items in TODO still apply: Canny, Hessian, rank filter, cellpose `follow_flows`, and the O(n²) NMS in stardist.

## Suggested order
Start with #1: it is the biggest win and is limited to `duckdb.rs` plus the exporter trait. Then #2 and #3 together, since both are about buffer handling in `PipelineContext`.

I left the benchmark at `crates/core/examples/bench_pipeline.rs` (untracked) so we can measure each fix before and after. Delete it if you don't want it in the repo.

Shall I start with the export rework (#1)?