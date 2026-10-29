# Benchmarks - GUI speed and RAM

Measured 2026-10-17 on the dev container: 22 cores, 64 GB RAM, release
build. The analysis itself (running pipelines, writing results) has its
own benchmarks: core's `bench_pipeline` and `bench_export` examples (see
"Not covered").

## How to run them

All in release mode. Only tests that run AI models need libtorch on the
library path:

```sh
export LD_LIBRARY_PATH=$PWD/libs/libtorch/lib
```

**Image viewer** - opens a slide in a 1920 × 1080 viewport, zooms to 5 %,
25 % and 100 %, pans 8 steps right and back. Each step is one settled
re-render; time, cache size and the process's RAM per step.

```sh
# A whole slide image: 60000 x 40000 px, 7 pyramid levels, 745 MB
cargo run --release -p evanalyzer_gui --example make_test_slide -- /tmp/slide.tif

EVANALYZER_BENCH_SLIDE=/tmp/slide.tif BENCH_TRIM=1 \
  cargo test --release -p evanalyzer_gui --lib bench_viewport_whole_slide -- --ignored --nocapture

# A weak computer: 2 threads, small caches
RAYON_NUM_THREADS=2 BENCH_TILE_CACHE_MB=256 BENCH_OVERVIEW_CACHE_MB=64 \
EVANALYZER_BENCH_SLIDE=/tmp/slide.tif BENCH_TRIM=1 \
  cargo test --release -p evanalyzer_gui --lib bench_viewport_whole_slide -- --ignored --nocapture
```

Other settings: `BENCH_COMPARE=1` (a 4096 × 4096 breakpoint tile at the
centre, compared), `BENCH_VIEWPORT=WxH`, `BENCH_PAN_STEPS`, `BENCH_SERIES`.

**Results views and split view** - opens a results file, shows List,
Plate and Charts, then the split view's work: one image's objects read
and drawn as ellipses (1920 × 1080 plus the overlay's margin). Time and
RAM (now / peak) per step.

```sh
EVADB=tests/20260831_Uptake1_Exp0706_PH1.evadb \
  cargo test --release -p evanalyzer_gui --lib bench_results_views -- --ignored --nocapture
```

(`bench_real_database` toggles an image in the file - give it a copy.)

**Screen repaints** - a full repaint of a screen with Slint's software
renderer: the scratchpad screenshot harness (`shot`, not in the repo) with
`bench=30`. The software renderer is the slowest one; it is what runs
without a usable GPU.

## Results

### Image viewer - whole slide (60000 × 40000, 7 levels)

| Step | 22 cores, default caches | 2 threads, 256 / 64 MB caches |
|---|---|---|
| Open (fit to screen) | 125 ms | 212 ms |
| Zoom 5 % / 25 % / 100 % | 125 / 97 / 104 ms | 217 / 166 / 164 ms |
| Pan to a new area | 24-46 ms | 47-72 ms |
| Pan back over seen tiles | 8-10 ms | 22-26 ms |
| RAM (peak, above start) | ~280 MB | ~190 MB |

Between re-renders the image follows the mouse right away (the last render
is moved and scaled); the times above are when it is sharp again.

### Breakpoint compare (2026-10-18)

`BENCH_COMPARE=1`: a 4096 × 4096 breakpoint tile at the centre, compared.
Before, the whole tile was rendered again after every pan or zoom; now
only when it changes.

| Step | Before | After | Compare off |
|---|---|---|---|
| Zoom 100 % / 25 % | 222 / 244 ms | 104 / 105 ms | 106 / 104 ms |
| Pan to a new area | 65-114 ms | 28-52 ms | 25-45 ms |
| Pan back | 32-38 ms | 9-12 ms | 8-10 ms |
| Whole tour | 1944 ms | 918 ms | 858 ms |
| RAM (peak) | ~460 MB | ~400 MB | ~265 MB |

### Results - real analysis (1.7 GB, 1776 images)

| Step | Time | RAM now / peak |
|---|---|---|
| Test process at start | - | 130 MB |
| Open, List first page (500 rows) | 72-80 ms | 276 MB |
| Plate | 9 ms | 292 MB |
| Charts (histogram) | 20 ms | 304 MB |
| Split: one image's objects (4500) | 6-9 ms | 308 MB |
| Split: draw the ellipses, fitted | 17 ms | 323 MB |
| Split: draw the ellipses, zoom 1 | 37 ms | 323 MB |

The ellipses were 36 / 73 ms before a fast path for pixels with nothing
under them yet (same pixels, checked by a test). Since 2026-10-23 the
split view draws masks with the editor's code, no ellipses - see "One
drawing code for the editor and the split view". They are drawn on the
viewer's worker thread, not the UI thread.

### Screen repaints, 1920 × 1080, software renderer

| Screen | Full repaint |
|---|---|
| Start page | 2.7 ms |
| Pipelines (steps open) | 3.0 ms |
| Images (image, objects) | 2.6 ms |
| Results › List / Plate | 1.8 / 2.1 ms |
| Results split - List / Plate | 8.9 / 7.7 ms |
| Settings dialog / Add step | 2.3 / 3.2 ms |

All well inside a 60 fps frame (16.7 ms). The start page's drifting cells
repaint 10 times a second: about 2 % of one core while it shows, none with
Settings › Viewport › Reduce motion.

## The real app: startup, idle RAM and CPU (2026-10-18)

Measured on the release binary under a virtual X server (Xvfb, no
graphics card - frames are drawn by Mesa's software OpenGL/Vulkan, as on
a computer without a working graphics driver). Script: start, poll for
the window every 10 ms (`xdotool search`), read `/proc/<pid>/status`
4 s later. (`xdotool search --sync` polls only every 0.5 s - the
"window ~1.0 s" first noted here was that, not the app.)

### libtorch as a plugin (2026-10-19)

libtorch is now loaded on the first AI step (`ai/cpu`, `ai/cuda` next
to the program) instead of being linked in. Same machine, same day, 5
warm and 3 cold runs each (cold: page cache dropped before every run):

| | Before (libtorch linked) | After (plugin) |
|---|---|---|
| Window on screen, warm | 347-369 ms | 154-174 ms |
| Window on screen, cold | 1064-1094 ms | 483-491 ms |
| RAM after startup, idle | 517 MB | 207 MB |

The first AI step pays the loading instead: a whole CLI run of a
one-image StarDist pipeline (start, load the plugin, set it up, run) is
~390 ms warm.

Before the split (2026-10-18 breakdown, release binary):

| | Warm | Cold |
|---|---|---|
| Before `main` (loading libtorch, its start-up code) | 530 ms | 1600 ms |
| `main` → controllers ready (logged, `--log-level debug`) | 25 ms | - |

An empty program starts in 3 ms; the dynamic loader itself needs ~7 ms.
The half second before `main` was libtorch registering its operators and
CUDA kernels - in every session, AI used or not.

RAM after startup, idle, before the split: ~510 MB.

| Part | RSS |
|---|---|
| libtorch / CUDA libraries (cublasLt 60, torch_cpu 52, torch_cuda 23, …) | ~137 MB mapped - now only once AI runs |
| libtorch's registrations on the heap | ~50 MB - now only once AI runs |
| The program's code | 53 MB |
| Software OpenGL/Vulkan of the test machine (LLVM, lavapipe) | ~150 MB (a real GPU driver instead) |
| The app's own heap | a few MB |

Idle CPU: 0 % with a project open. The start page's drifting cells cost
~370 % CPU with software rendering (2.7 ms a frame with Slint's own
software renderer, next to nothing with a graphics card) - Settings ›
Viewport › Reduce motion stops them.

Renderer (Settings › Viewport › Draw in software, read at startup): with
Slint's own software renderer the idle app takes ~135 MB less (500 → 365
MB release, 582 → 444 MB debug, measured before the libtorch split) - no OpenGL/Vulkan stack is loaded.

### Leak checks (soak tests)

- Viewer, 400 renders (zoom, pan, compare on/off with a breakpoint tile):
  RAM rises while the tile cache fills (+296 → +346 MB in the first 150),
  then stays flat. No leak.
  `EVANALYZER_BENCH_SLIDE=… cargo test --release -p evanalyzer_gui --lib bench_soak_viewport -- --ignored --nocapture`
- Results, the 1.7 GB analysis opened again and again with List, Plate,
  Charts and 20 split selections each round: +0.2-0.4 MB a round, slowly
  flattening (`BENCH_ROUNDS`, `BENCH_PART=open|views`). heaptrack (60 vs
  10 rounds) puts the difference in DuckDB's buffer pool, not in our code.
  DuckDB may keep 80 % of the RAM by default, per open results file (they
  stay open for the session) - now capped at a quarter of the RAM
  (1-8 GB); beyond that it spills to a temporary file. Same speed with
  and without the cap.
  `EVADB=… cargo test --release -p evanalyzer_gui --lib bench_soak_results -- --ignored --nocapture`

## AI models: Cellpose through the plugin (2026-10-19)

Is Cellpose slower now that libtorch runs in a plugin? No. Stand-in
models (`cargo run --release -p evanalyzer_ai --features test-support
--example make_bench_models -- <dir>`: "light" = next to no compute, the
cost around the model; "heavy" = 6 convolution layers) and the real
Cellpose-SAM (`tests/cpsam_v2.pt`, 1.2 GB), CPU only, 22 shared cores
(other load on the host: expect ±10 %).

**The plugin boundary** - one Cellpose request on libtorch linked in vs.
through the plugin, same process (~16 s):

```sh
cargo test --release -p evanalyzer_ai --features torch-plugin,test-support \
  --lib bench_plugin_against_in_process -- --ignored --nocapture
# + BENCH_CELLPOSE_MODEL=$PWD/tests/cpsam_v2.pt for the real model (~1 min more)
```

| Model, image | Linked in | Plugin | Difference |
|---|---|---|---|
| light, 512² / 1024² / 2048² | 4 / 14 / 77 ms | 5 / 18 / 106-121 ms | +1 / +4 / +30-44 ms |
| heavy, 512² / 1024² / 2048² | 0.19 / 0.6 / 2.3 s | same within ±6 % | noise |
| real Cellpose-SAM, 512² / 1024² | 25 / 63 s | 23 / 67 s | noise |

The only cost is handing the three result planes back (48 MB at 2048²).
They cross as raw bytes (`api::floats`); with postcard's float-by-float
encoding it was +65 ms at 2048². Next to a real model's seconds per tile
it does not show.

**End to end** - a Cellpose pipeline (`bench_cellpose` example) with the
real model, one 1024² image, old build vs. new, alternated:

```sh
cargo run --release -p evanalyzer_core --features libtorch --example bench_cellpose -- \
  <workdir> --model tests/cpsam_v2.pt --channels 3 --images 1 --size 1024 --repeat 2
# with --plugin target/release/libevanalyzer_ai.so (then EVANALYZER_AI_PLUGIN) and libtorch on LD_LIBRARY_PATH
```

| Build | Time per image | Peak RAM |
|---|---|---|
| Before (HEAD, libtorch linked into the program) | 123-137 s | ~1.7 GB |
| New code, libtorch linked in (`libtorch-in-process`) | 137-152 s | ~1.7 GB |
| New code, plugin (what ships) | 65-67 s | ~1.9 GB |

The same Cellpose code runs **twice as fast** with libtorch as a plugin
than linked into the program - in both old and new code, so it is how
libtorch is loaded, not the code. Not explained yet (likely threads or
libraries libtorch shares with the program when linked in, e.g. its
OpenMP runtime). The stand-in models show no such gap, so it may be
specific to the ViT model. The plugin's ~0.2 GB more peak RAM is not
explained yet either.

## Time course: "Over time" chart and its export (2026-10-21)

A synthetic time series: 48 wells × 2 fields, 48 time points, 1500
tracked objects per image and time point - 6.6M objects, 144,000 tracks,
1.3 GB (`tests/timeseries_96img_48t_tracked.evadb`, not in git). Each case
runs in its own process, so the peak RAM is that case's alone (the process
starts at 27 MB with the file open); chart times are the median of 5 runs.

```sh
# Make the dataset (~70 s, ~2.4 GB RAM)
cargo run --release -p evanalyzer_core --example make_time_series -- tests/timeseries_96img_48t_tracked.evadb

# One case (0-6 chart queries, 7-10 exports); without --case all in one process
cargo build --release -p evanalyzer_app --example bench_time_course
./target/release/examples/bench_time_course tests/timeseries_96img_48t_tracked.evadb --iters 5 --case 10
```

Exports write 3 values (area, two intensities) as mean + SD.

| Case | Before | After RAM work | After speed work |
|---|---|---|---|
| Chart: a line per class, mean ± SD | 186 ms / 684 MB | 92 ms / 187 MB | 91 ms / 183 MB |
| Chart: a line per class, median + IQR | 254 ms / 1153 MB | 126 ms / 462 MB | 117 ms / 467 MB |
| Chart: a line per class, relative to start | 191 ms / 685 MB | 90 ms / 182 MB | 88 ms / 181 MB |
| Chart: a line per image (13,824 points) | 221 ms / 709 MB | 102 ms / 194 MB | 97 ms / 195 MB |
| Chart: a line per well (6,912 points) | 207 ms / 696 MB | 101 ms / 194 MB | 93 ms / 186 MB |
| Chart: faint tracks (400 sampled) | 53 ms / 232 MB | 47 ms / 183 MB | 47 ms / 188 MB |
| Every track held in memory (6.8M points) | 5274 ms / 5055 MB | 2736 ms / 2440 MB | 2058 ms / 2137 MB |
| Export CSV long + wide, per class | 245 ms / 700 MB | 137 ms / 245 MB | 137 ms / 247 MB |
| Export CSV long + wide, per well | 274 ms / 717 MB | 180 ms / 253 MB | 160 ms / 252 MB |
| Export XLSX long + wide, per image | 502 ms / 757 MB | 387 ms / 263 MB | 384 ms / 262 MB |
| Export CSV long, per track (6.8M rows, 610 MB) | 15.7 s / 6205 MB | 11.7 s / 428 MB | 3.6 s / 411 MB |

Before these the per-track export did not finish at all (stopped after
12 minutes at 4.9 GB): the time unit was looked up over every point for
every row.

What brought the RAM down:

- Tracks counted without `COUNT(DISTINCT track_id)`: a track has at most
  one object per plane and class (the tracker links each object once), so
  its objects there are its tracks. The distinct count held every
  (plane, track) pair: ~470 MB of the chart's 684 MB.
- Rows streamed from DuckDB (Arrow chunks) and each point built once.
- Every track: one query per time point - grouping and sorting 6.8M rows
  in one go took ~1.6 GB inside DuckDB. Four time points per query would
  be ~10 % faster for ~100 MB more.
- Median, Q1 and Q3 in one aggregate (each kept its own copy of the
  values: 730 → 390 MB for the query alone).
- The export writes the long table as the rows come; points are kept only
  for the wide tables (never per track - refused).

What made the export faster: fields written straight into one reused line
(no vectors and strings per row), one point reused for every row, whole
numbers written as integers and fractions through `zmij`. `zmij` writes
the shortest text that reads back as the same number, like before; where
17 digits are needed it may pick a different last digit (same value).
`zmij` instead of `ryu`: 3.70-3.82 vs 3.84-4.19 s for the per-track
export, alternated (~5 %; both crates were already dependencies). Rust's
own `{}` formatting instead: 4.45-4.75 s (`zmij` saves ~0.75 s, ~17 %).

Every track held in memory (2.1 GB) is only the benchmark's: the chart
draws a sample of 400 tracks, the export streams.

### List export - a row per object (2026-10-21)

Same dataset, the List (area and two intensities per object, T column),
cases 11 and 12 of `bench_time_course`. It already wrote page by page
(20,000 rows), so its RAM was fine; the time was not.

| Case | Before | After |
|---|---|---|
| List CSV, 6.6M rows (163 MB) | 39.1 s / 827 MB | 13.4-13.9 s / 772-789 MB |
| List XLSX (over the sheet's row limit) | fails after 7.7 s: "Row or column exceeds Excel's allowed limits" | fails after 4.2 s: "more than an XLSX sheet holds: export it as CSV, or one file per image" |

- Each page fetched its rows with `object_id IN (...)` - 20,000 ids
  DuckDB parsed and planned for every page (~330 pages). Now the page's
  first and last id: the same rows (the page is the first 20,000 matching
  ids in order).
- CSV rows written straight into one line (no string per cell), floats
  through `zmij` as in the time course.
- Pages of 50,000 / 100,000 / 200,000 rows: 13.6 / 12.8 / 12.7 s for
  +130-190 MB - kept at 20,000. The object ids are random, so each page's
  query still scans the time point; what's left is spread over DuckDB,
  reading rows through `duckdb`'s row API and building cells - a
  dedicated Arrow-based List export would be the next step.

The GUI's List uses the same query: open + first page of the 1.7 GB
analysis 65-75 ms (was 75-79 ms, alternated), RAM unchanged (~200 MB).

### Results › Tracks (2026-10-21)

One image's tracks for the table and the trails: one query per image,
read once (stepping through time only reads that time point's objects).
Same dataset (1500 tracks per image, 48 time points).

```sh
./target/release/examples/bench_time_course tests/timeseries_96img_48t_tracked.evadb --iters 5 --case 13
cargo test --release -p evanalyzer_gui --lib bench_trails -- --ignored --nocapture
```

| Step | Time | RAM |
|---|---|---|
| An image's tracks (1500 tracks, 68,511 points) | 15-17 ms | 57-63 MB peak (query process); ~1 MB kept |
| Trails of 10 time points, 1500 tracks, 1920 × 1080 | 1.4 ms | - |
| Whole tracks up to the time point | 5.3 ms | - |
| The selected track only | 0.01 ms | - |
| Export: a row per track, 144,000 tracks, 3 values (CSV, 25.6 MB) | 1.27-1.50 s | 406-410 MB peak (query process) |
| Tracks, Full: every track's row kept, 144,000 tracks | 1.05 s | 291-300 MB peak (query process) |

The track export (case 14 of `bench_time_course`) reads one image at a
time - a track never leaves its image - and writes each track's row as it
comes, so its RAM doesn't grow with the number of tracks.

At most 1,000,000 points of an image are read (~16 MB): beyond, a
repeatable sample of whole tracks, and the table says "N of M tracks (a
sample)". The table's rows are made only when shown (`ListView`), so
100,000 tracks cost no more there than 100. The trails are drawn with the
objects on the viewer's worker thread (the ellipses take 17-37 ms there).

The window's generated UI code grew 52.49 → 53.33 MB (+1.6 %) with the
Tracks view.

### Help texts as markdown (2026-10-22)

The step and setting help is drawn as markdown (`InkMarkdownText`, Slint
`StyledText`; `crates/gui/src/helper/markdown.rs`). Measured over all 235
help texts (46 KB, the longest 1.2 KB) in release, outside the app with a
counting allocator; the code is not in the repo:

| | Time | RAM |
|---|---|---|
| Markdown -> styled text, one help text | ~1 µs on average, 4.3 µs the longest | ~350 bytes (1.8× the text), 1.6 KB the longest |
| Plain text without the marks (picker rows) | ~1.5 µs | a copy of the text |
| All 235 at once (never happens: one is shown at a time, a picker row each) | ~0.25 ms | ~82 KB |
| Generated UI code | | +21.5 KB |

A text is converted once when its element is made (a card opened, a step's
dialog shown, a module or command picked in Add step), not per frame. Drawing
was not measured separately - styled and plain text both go through the same
text layout.

### Propagation - cells grown from nuclei (2026-10-22)

`BENCH_SIZE=<n> cargo test --release -p evanalyzer_core --lib bench_propagation -- --ignored --nocapture`
(one size per process for its own peak): a synthetic n² cell stain with a
nucleus (7 × 7 px) every 40 px, λ 0.05, no mask, 22 cores. CellProfiler's
code (centrosome 1.3.4, `propagate()` on the same image and seeds, script
in the session scratchpad) for comparison; its regions are identical
(`regions_match_reference`).

| Image | Nuclei | EVA | Peak RAM (before) | CellProfiler | Its peak RAM |
|---|---|---|---|---|---|
| 2048² (4.2 MP) | 2,601 | 1.9 s | 120 MB (32) | 8.9 s | 297 MB |
| 4096² (16.8 MP) | 10,404 | 11.3 s | 438 MB (87) | 55.0 s | 1,117 MB |
| 8192² (67 MP) | 42,025 | 64 s | 1,714 MB (310) | - | - |

About 21 bytes per pixel on top of the image (values, read and mask flags,
seed labels, labels, costs) plus the queue; the whole image is held at once,
like Voronoi - not for whole-slide images. Single-threaded; time grows a
little faster than the pixels (the queue's log n).

After the speed/RAM step (same day; flags as bits, seeds as a list instead of
a label per pixel, 16-byte queue entries, no border clamping away from the
border) - old and new alternated with 10 s pauses, results still identical
to CellProfiler's:

| Image | Before | After |
|---|---|---|
| 4096² | 13.3 / 14.3 s, 438 MB | 10.1 / 10.7 s, 397 MB |
| 8192² | 65.8 / 70.7 s, 1,714 MB | 58.4 / 57.5 s, 1,551 MB |

The seed labels hardly counted before (an all-zero buffer's pages are never
touched), so the RAM saved is the two flag arrays and the smaller queue:
now ~18.5 bytes per pixel on top of the image at 8192²; the costs (`f64`,
8 bytes) stay - `f32` would change sums and ties against CellProfiler.

### Relate Objects - spots linked to cells (2026-10-23)

`BENCH_SIZE=8192 BENCH_RULE=centre|overlap cargo test --release -p evanalyzer_core --lib bench_relate_objects -- --ignored --nocapture`
(one rule per process): a cell (40 × 40 px) every 40 px on an 8192² image,
10 spots (3 × 3 px) per cell along its diagonal, the last one across the
border. "It overlaps the parent most" gives the same parents as
CellProfiler's RelateObjects (`largest_overlap_matches_reference`).

| Rule | Cells | Spots | Time | Peak RAM (objects before) |
|---|---|---|---|---|
| Its centre is in the parent | 41,616 | 416,160 | 0.42 s | 577 MB (475) |
| It overlaps the parent most | 41,616 | 416,160 | 0.31 s | 577 MB (475) |

Single-threaded; a grid over the parents' boxes, so each spot is tested
against a handful of cells. The ~100 MB on top of the objects are the
grid, the id lists and the links before they are written.

### Centroid as centre of mass, bbox centre and parent ids stored (2026-10-23)

`get_centroid()` is now the mean pixel position (was the bounding box's
centre, now `get_bbox_center()` and its own columns); `parent_ids` replaces
`parent_id`/`children`. Last commit vs. this change, alternated with 10 s
pauses, 22 cores:

| Benchmark | Before | After |
|---|---|---|
| `bench_pipeline` 200 × 2048², 8 threads (4 runs each) | 6.05 / 5.53 / 5.35 / 5.44 s (mean 5.59) | 5.97 / 5.60 / 5.46 / 5.72 s (mean 5.69) |
| `bench_export` 3.8M objects (2 runs each) | 33.0 / 32.7 s, 2.28-2.39 GB | 34.9 / 34.2 s, 2.40-2.43 GB |

A first version walked every mask pixel for the centroid: +0.25 s in
`bench_pipeline`. Now it is computed a 64-bit word at a time (six popcounts
per word for the bit positions, a count per row for y) - the same result,
no measurable cost (a build without any centroid: mean 5.60 s). The writer's
~5 % are the four `bbox_center_*` columns and the `parent_ids` map; the file
is 1,317 MB instead of 1,309 MB.

## Object masks in the results (2026-10-23)

Each object's mask as run lengths (`objects.mask BLOB`, LEB128 varints of
alternating unset/set runs, row by row in its box -
`storage/mask_codec.rs`), switch `resultsStorage.storeObjectMasks`
(Settings › Objects). Three builds alternated, 10 s pauses, 22 cores:
**base** = the last build without the column, **off** / **on** = this one
with the switch off / on.

| Benchmark | base | off | on |
|---|---|---|---|
| `bench_pipeline` 200 × 2048², 8 threads, 208,002 objects (2 runs each) | 5.63 / 5.31 s, 982-1040 MB | 5.20 / 5.33 s, 993-1009 MB | 5.34 / 5.43 s, 1028-1053 MB |
| Results file of that run | 54.0 MB | 54.0 MB | 112-114 MB (+~290 bytes per object) |
| `bench_export --object-size 24`, 3.8M objects, 4 masks repeated (1 run each) | 47.1 s, 2509 MB, 1473 MB | 46.3 s, 2505 MB, 1474 MB | 47.3 s, 2511 MB, 1476 MB |
| `bench_export --object-size 24`, 3.8M objects, every mask its own (2 runs each) | - | 51.4 / 50.4 s, 2509-2561 MB, 1489 MB | 53.2 / 53.3 s, 2527-2557 MB, 1576-1580 MB |

- **Switched off, the same as before**: the column holds NULLs - same
  speed, RAM and file size as the build without it (both benchmarks).
- **On**: no measurable cost in a pipeline run (the image work dominates);
  writing alone ~5 % slower (+2.4 s for 3.8M objects); RAM unchanged. The
  file grows by the masks: ~290 bytes per object for the pipeline's
  segmented spots (the file doubles - those results are small otherwise),
  ~24 bytes for the 24 px objects of `bench_export` (+6 %). Big objects
  cost more: about two varints per row of the object.
- The first `bench_export` generator made only 4 different masks - DuckDB
  stores repeated BLOBs once, so that row understates the size. Its
  objects now each get their own outline (`bench_export`'s area/centroid
  columns compress a little worse with it: 1489 instead of 1473 MB, and
  writing is slower - base isn't comparable on that row).
- Split view: the masks are decoded per shown image (one image's objects);
  drawing them see "One drawing code for the editor and the split view".

## One drawing code for the editor and the split view, old vs new (2026-10-23)

Results' split view draws and picks its objects with the editor's code
(`editor/object_render.rs`), from the stored masks (switched off: their
boxes, filled); no ellipses. The compositor fills each mask row's runs of
set pixels at once instead of pixel by pixel (both views gain). **Old** =
95d79ded (before masks), **new** = this change; release builds, alternated,
10 s pauses, 22 cores. Time per step; RAM peak of the test process.

Results views (`bench_results_views`), 2 runs each:

| File | Step | Old | New |
|---|---|---|---|
| Real analysis, 1.7 GB, 4500 objects an image (no masks: boxes) | Open, List first page | 71-74 ms | 73-76 ms |
| | Plate / Charts | 8-9 / 18-19 ms | 9-10 / 17 ms |
| | Split: one image's objects | 7-9 ms | 6-9 ms |
| | Split: draw, fitted | 16-17 ms | 19-20 ms |
| | Split: draw, zoom 1 | 32 ms | 38-39 ms |
| | RAM peak | 221-224 MB | 227-231 MB |
| Real segmentation with masks (`bench_pipeline`), 1045 objects an image | Split: one image's objects | 3.3-3.5 ms | 2.8-2.9 ms |
| | Split: draw, fitted | 9.7-9.8 ms | 8.1-8.7 ms |
| | Split: draw, zoom 1 | 17.3-17.4 ms | 10.4-10.8 ms |
| | RAM peak | 112-115 MB | 116-120 MB |
| Synthetic, 3.8M objects with masks, 1.58 GB | Open, List first page | 169-173 ms | 166-180 ms |
| | Plate / Charts | 7-10 / 14-17 ms | 6-7 / 12-13 ms |
| | Split: one image's objects (2150) | 6.5-7.9 ms | 6.4-6.9 ms |
| | Split: draw (24 px objects, mostly beside the view) | 0-0.4 ms | 0.6-2.0 ms |
| | RAM peak | 169-172 MB | 171-175 MB |

- With masks the split view draws faster than the old ellipses. Without
  them (old results, or the switch off) its filled boxes take ~20 % longer
  than the ellipses did: a box has 27 % more pixels.
- Reading one image's objects: the check whether the file keeps masks is
  made once per open file (an `information_schema` query each time cost
  3 ms).
- Leak check (`bench_soak_results`, the 3.8M file opened 15 times): old
  and new alike, +88 / +91 / +93 MB after 5 / 10 / 15 rounds - levelling
  off.

The compositor alone (`object_render::bench_draw_objects`, 1045 round
objects on a 2048² image, best of 20) - the editor's drawing too:

| Objects | Before (column runs zoomed out) | After (runs of set pixels) |
|---|---|---|
| 20 px, fitted (zoom 0.47) / zoom 1 | 2.2 / 2.5 ms | 1.8 / 2.7 ms |
| 60 px, fitted / zoom 1 | 12.7 / 11.7 ms | 6.8 / 12.3 ms |
| 120 px, fitted / zoom 1 | 50.7 / 57.2 ms | 24.5 / 58.8 ms |

At zoom 1 the time is the blending (every covered screen pixel once per
object); the first new version without the runs was 87-90 ms at zoom 1 on
the real analysis.

Writing and running, same session (2 runs each):

| Benchmark | Old | New, masks off | New, masks on |
|---|---|---|---|
| `bench_export --object-size 24`, 3.8M objects (old built with the same generator) | 52.7 / 51.3 s, 2.54-2.55 GB, 1489 MB | 51.4 / 51.3 s, 2.49-2.60 GB, 1489 MB | 53.0 / 52.9 s, 2.55-2.58 GB, 1577-1578 MB |
| `bench_pipeline` 200 × 2048², 8 threads, 208,002 objects | 5.45 / 5.54 s, 1.00-1.11 GB, 54.0 MB | 5.24 / 5.36 s, 1.00-1.02 GB, 54.3 MB | 5.34 / 5.65 s, 1.04-1.13 GB, 111.4-111.7 MB |

## Measurement storage and object intensity spread (2026-10-24)

22 cores, release. "base" = the commit before (4f167d47, a worktree with
its own target dir), alternated with "new", 10 s pauses.

### Writing per-object measurement values

`measurements` rows went in row by row (prepared `INSERT`); now through
the Arrow appender like `objects`. Temporary test calling
`write_measurements` on one image's values (peak RSS of the test process):

| Values per object measurement | row by row | Arrow appender |
|---|---|---|
| 20,000 | 10.1-11.6 s, 3.6 GB | 52-56 ms, 90 MB |
| 200,000 | out of memory (DuckDB: 7.6 GB of 7.6 GB) | 0.53-0.75 s, 400 MB |

### Reading them: `objects.measurement_values` (+ `measurement_info.object_index`)

`bench_measurements` (app example) on a copy of the 1.7 GB reference
analysis (5.6M objects) with 10 per-object measurements added (56M
`measurements` rows); one query per process, median of 3, peak RSS.
Before: each object's value by a subquery on `measurements` (DuckDB turns
it into a join over every value of the key); now a list on the object,
the position looked up once.

| Query | Area (`objects` column) | measurement, before | measurement, now |
|---|---|---|---|
| Plate, mean | 29 ms, 109 MB | 359 ms, 1225 MB | 41 ms, 205 MB |
| Per image, first page (500) | 29 ms, 132 MB | 205 ms, 1054 MB | 41 ms, 228 MB |
| Object list, first page (500) | 15 ms, 96 MB | 22 ms, 110 MB | 23 ms, 113 MB |
| Histogram | 17 ms, 69 MB | 460 ms, 1047 MB | 44 ms, 174 MB |

Several measurement columns at once (per image, the query alone in
DuckDB 1.5.5): 3 columns 520 ms / 1.85 GB before, 10 columns 2.1 s /
4.2 GB; with the list 10 columns 56-77 ms / 275-285 MB. Also tried:
aggregating straight from `measurements` (10 columns 494 ms / 611 MB) and
a pivot joined to `objects` (10 columns 2.7 s / 8.4 GB). Size: the list
of 10 values adds 109 MB to `objects` (1719 → 1828 MB, written fresh);
`measurements` itself is 946 MB.

### Intensity spread (std, median, quartiles, MAD per channel)

CellProfiler's MeasureObjectIntensity numbers (fixture
`crates/core/tests/object_intensity.json`), computed for every
object; stored as `intensity_{std,median,lower_quartile,upper_quartile,mad}_gray`.

| Benchmark | base | new |
|---|---|---|
| `bench_pipeline` 200 × 2048², 8 threads, 208,002 objects (507M object pixels), 5 runs | 5.42-5.66 s (median 5.57), 1065-1104 MB | 6.23-7.32 s (median 6.62), 1107-1157 MB |
| `bench_export --object-size 24`, 3.8M objects × 6 channels, with spread (2 runs) | 53.8 / 54.3 s, 2.53-2.59 GB, file 1578 MB | 61.3 / 61.8 s, 3.12-3.27 GB, file 2375 MB |
| same, objects without spread (as stored annotations) | | 55.6 / 56.3 s, 2.50-2.54 GB, file 1576 MB |

- The spread is 30 more values per object here (6 channels × 5): +13 %
  writing time, +50 % file size (~210 bytes per object), +0.6 GB peak
  RAM while writing. Normalized copies as well (first version) made
  writing +30 %; only gray values are stored now.
- `bench_pipeline` is bound by the database writer (its objects are big;
  the per-image writes add up to ~6 s): ~0.5 s of the +1 s is writing the
  spread, ~0.5 s collecting and ordering 507M pixel values.
- Ordering the values: sorting each object's values made the pipeline
  ~+23 % (median 6.9 s); selecting the few values needed was no better on
  these big objects; a histogram over 65535 levels (exact for 8/16-bit,
  falls back to selecting otherwise): 9 → 3.7 ns per pixel after taking
  the variance from the levels too and rounding without `round()`.
- Per pixel (2M values, `bench_spread` in core, ignored): 3.6-4.2 ns.

### Transform Objects › Expand Without Overlap (2026-10-24)

`bench_expand_without_overlap` (core, ignored test, release): a 9 × 9
object every 40 px on 8192² (42,025 objects), worked out in blocks of
2048² plus the margin; single-threaded.

| Margin | Time | Peak RSS (test process) |
|---|---|---|
| 8 px (2 runs) | 2.37 / 2.66 s | 376 MB |
| 100 px (every pixel of the image ends up in an object) | 2.94 s | 861 MB |

Most of the RAM at 100 px is the list of added pixels (8 bytes each)
before the masks are built (67M pixels); the masks themselves are 1 bit
per pixel.

### Measure Colocalization (2026-10-24)

`bench_measure_colocalization` (core, ignored test, release): 2048²
16-bit image in 4 tiles, 2,601 objects (9 × 9), every value per image
and per object.

| Costes thresholds | Time | Peak RSS (test process, incl. the test images) |
|---|---|---|
| Off | 35 ms | 90 MB |
| Faster (default, 2 runs) | 211 / 212 ms | 92 MB |
| Fast | 749 ms | 92 MB |

Costes "Faster" is a few passes over the image per candidate (~15
candidates for 16 bit); "Fast"/"Accurate" walk down from the maximum one
gray level (or a few) at a time - on 16-bit images with little
correlation that can be thousands of passes (CellProfiler warns the same).

### Measure Object Neighbors (2026-10-24)

`bench_object_neighbors` (core, ignored test, release): a 9 × 9 object
every 40 px on 8192² (42,025 objects), same class as neighbours.

| Method | Time | Peak RSS (test process) |
|---|---|---|
| Adjacent | 221-267 ms | 93 MB |
| Within 35 px | 40.1 s → 5.5 s | 93 MB |
| Expand until adjacent (whole-image raster) | 4.0 s / 2.76 GB → 3.1 s / 925 MB | |

- Within a distance: dilating with each of the ~4,000 offsets of a 35 px
  disk was 40 s; dilations with more than 49 offsets now use the nearest
  source pixel (feature transform of the patch) - the same pixels.
- Expand: the feature transform stored (row, column) as two `i64` in an
  `Option` (24 bytes per pixel); now two `i32` (8 bytes). Expand
  Without Overlap gained the same: 2.4 s / 376 MB → 1.3 s / 255 MB
  (margin 8, same test as above). The whole-image raster is limited to
  128 MP (~2 GB); bigger images: "Within a distance".

### Shape: Feret, convex area, Euler number, radii (2026-10-24)

Per object (core, ignored tests `bench_small_shape` / `bench_shape`,
release): a disk of r = 11 (379 px, like `bench_export --object-size 24`)
and of r = 700 (1.5M px).

| Value | r = 11, first version | r = 11, now | r = 700, first → now |
|---|---|---|---|
| Feret (hull + calipers) | 2.8 µs | ~1.8 µs (O(h) calipers, no sort) | 16.5 → 0.2-0.5 ms |
| Convex area | 7.4 µs | 4.5-8 µs | 111 → 0.6-1.1 ms |
| Euler number | 1.5 µs | 0.5 µs | 14 → 3.5 ms |
| Radii (distance transform) | 9.2 µs | ~7 µs | 102 → 33 ms |

In every export they made writing 3.8M objects 180 s instead of 54 s,
and `bench_pipeline` 11-12 s instead of 5.7 s (its objects' boxes are
large). Now only Feret is computed at export (it was the bounding box's
diagonal and the ellipse's minor axis - wrong); convex area, Euler number
and radii are the step "Measure Object Shape":

| Benchmark | base | new (spread + Feret) |
|---|---|---|
| `bench_pipeline` 200 × 2048², 8 threads (3 runs) | 5.58 / 5.80 / 6.96 s, 1.02-1.15 GB | 6.46 / 6.60 / 6.95 s, 1.09-1.19 GB |
| `bench_export --object-size 24`, 3.8M objects (2 runs) | 53.8 / 54.3 s, 2.53-2.57 GB | 67.7 / 67.9 s, 3.22-3.24 GB |

### Measure Texture (2026-10-24)

`bench_measure_texture` (core, ignored test, release): a textured 16-bit
image in 4 tiles, a 21 × 21 object every 30 px, scale 3, the 4 directions
(the machine was busy: load ~14).

| Image | Gray levels | Image only | Image + objects | Peak RSS (test process, incl. the test image) |
|---|---|---|---|---|
| 2048², 4,624 objects | 256 | 25 ms | 55-77 ms | 74 MB |
| 8192², 74,529 objects | 256 | 366 ms | 823 ms | 603 MB |
| 8192², 74,529 objects | 8 | 382 ms | 506 ms | 603 MB |

~6 µs per 441 px object at 256 gray levels (~2 µs at 8). The counts are
kept per entry counted, not as dense 256 × 256 matrices summed in full;
the image's tiles are counted in parallel (1 byte per pixel per tile in
flight). Most of the 603 MB is the test's own image (256 MB of f32 plus
its copy while the tiles are built).

### Measure Image / Object split: texture and colocalization (2026-10-25)

Measure Texture and Measure Colocalization became Measure Image Texture
+ Measure Object Texture and Measure Image Colocalization + Measure
Object Colocalization (one target per command). `bench_measure_texture`
/ `bench_measure_colocalization` (release, 2048², 4,624 / 2,601
objects), the step before vs. the two new ones, alternated, 2 runs each
(the machine was busy: load ~14-19):

| Benchmark | before (one step, image + objects) | after: image | after: objects | Peak RSS before → after |
|---|---|---|---|---|
| Texture, 256 gray levels | 48.8 / 54.9 ms | 24.9 / 25.4 ms | 27.5 / 28.2 ms | 69-72 → 62-63 MB |
| Colocalization, Costes off | 31.7 / 38.2 ms | 17.1 / 20.4 ms | 13.2 / 15.1 ms | 88-89 MB |
| Colocalization, Costes Faster | 204 / 237 ms | 186 / 205 ms | 185 / 191 ms | 88-89 MB |

The same work, split - except Costes: the objects' Costes values use the
whole image's Costes thresholds, so Measure Object Colocalization
searches for them itself. Both commands with Costes on take ~1.8× the
step before; either alone about as long as it did.

### Intensities in gray values only - no `_normalized` lists (2026-10-24)

`intensity_{sum,mean,min,max}_normalized` dropped (nothing read them; 0..1
= gray / (2^image_bit_depth - 1)); the spread's lists are in the same
column group. `bench_export --object-size 24` (3.8M objects × 6
channels, with spread), alternated, 2 runs each (the machine was busy:
load ~11, so both are slower than in the section above):

| | before | after |
|---|---|---|
| Time | 72.8 / 75.6 s | 66.8 / 67.9 s (-8 %) |
| Peak RSS | 3.16-3.18 GB | 2.57-2.64 GB |
| File after checkpoint | 2425 MB | 1802-1805 MB (-26 %) |

## Reproducible runs and the whole slide's whole-image phase (2026-10-24)

The performance-check project (`tests/performance-check`, its own repo:
81 × 2048² `.vsi`, and one 5-channel whole slide in 21 tiles), run through
`evanalyzer cli analyze` + CSV export, objects compared row by row without
their ids (`cmp.py`-style: rows only in one run).

**Same input, same objects.** Ties were broken by `ObjectId` - UUID v7,
random below the millisecond - so each run picked differently:

| Run vs run | before | after |
|---|---|---|
| 81 images (Coloc "only the best partner") | 9-15 of 298,262 objects differ | 0 |
| Whole slide (Voronoi cells) | 1,657-1,711 of 99,473 differ | 0 |

Coloc, Voronoi and Relate now break ties by position
(`Object::position_cmp`: the first pixel in reading order - how
scipy/scikit-image number labels, so CellProfiler's "lower label" ties
fall the same way).

**Whole slide 131 s → 76 s.** With `--log-level
info,evanalyzer_core::algos=debug,evanalyzer_core::job::job_executor=debug`
(new: a line per step and per phase): 21 tiles 58-60 s (parallel), then
single-threaded Tile Merge 50 s and Voronoi 19 s. Tile Merge measured its
258 merged objects (188M px) one by one, each fetching its tiles from the
image cache - 1,821 loads of 64 MB from the spill files (Voronoi 568).
Now `Object::measure_intensities_of`: tile by tile, each channel tile
fetched once, objects in parallel (an object over several tiles keeps one
accumulator per channel until its last tile; from 65,536 values on gray
values are counted per level, 256 KB per channel however big). Also fixed:
the histogram path for median/quartiles expected values 0..1 - with raw
gray values it never ran (selection instead; same numbers, slower).

| Whole slide (old / new build alternated) | before | after |
|---|---|---|
| Wall | 139 / 154 / 118 / 135 s | 78 / 75 s |
| Peak RSS | 9.1-9.2 GB | 7.8-8.3 GB |
| Cores used (avg) | 6.3-7.5 | 11.5-12.0 |
| Tile Merge / Voronoi | 50 / 19 s | 7.0 / 6.5 s |
| Disk loads after the tiles | 2,389 | 210 |
| 81 images | 13.0-14.7 s, 3.0-3.1 GB | 13.6-13.7 s, 3.0-3.1 GB |

The exported values are the same as before (0 rows differ at the CSV's
precision). Left: the tiles' 58 s - the steps add up to ~15 s per tile
(Median Subtract 4.8 s per call), the rest is reading the slide (not
measured yet).

### Whole slide's tiles: reading and Median Subtract (2026-10-24)

Per tile (new debug line "tile (x, y): read … s, all … s"): a full 4096²
tile takes ~55 s with 21 tiles at once, 35-41 s of it reading. The 5-channel
series is 5 z-planes × 5 channels of JPEG 2000 tiles (512², 25,650 at full
resolution, max projection over z): ~1,600 decoded per 4096² tile, ~8 ms
each on one core (the image reader's JPEG 2000 decoder) - 13-14 s per tile
alone, 35-41 s when 21 tiles share this laptop CPU (Core Ultra 9 185H: 6 P +
10 E cores). Reading is the image reader's part (not changed here).

Median Subtract was the biggest step: 174-183 s CPU per run (42 calls,
4.8 s per 4096² tile, radius 3: 37 values ranked per pixel). Now a sliding
histogram (Huang) for gray values - raw or scaled to 0..1 (the pipeline's
images are scaled; 8-bit too) - when each level holds one value; bitmaps of
the used levels let the median jump; the same values bit for bit (checked
against ranking every window, NaN/ties/RGB/-0.0 fallback). Rows in parallel.

| | before | after |
|---|---|---|
| Median Subtract, whole slide (CPU, 42 calls) | 174-183 s | 33 s |
| 4096² random 200..3200, one P-core | 3.9-4.4 s | 2.8-3.0 s |
| Whole slide: 21 tiles / wall | 58-63 s / 76-78 s | 50 s / 67-69 s |
| 81 images | 13.6-13.7 s | 14.4 s (no median; machine busier) |

Exported objects: 0 rows differ from before, whole slide and 81 images.
(The machine had load 4-9 from outside during these runs.)

## One shared image reader (2026-10-25)

The image reader (scopeio) is thread-safe: one reader per image now serves
every tile of a pipeline run (before: a fresh reader per tile) and the viewer
(before: a pool of up to 8 readers, one per channel). Opening takes
0.2-1.2 ms (`.vsi`, `.nd2`, OME-TIFF in `tests/`), so the per-tile reopen
cost ~1 ms - the change is mostly simpler code. Old / new builds alternated,
10 s apart:

| | before | after |
|---|---|---|
| Pipeline, 6 images 12288² (9 tiles each), 8 threads | 3.87 / 3.94 / 4.23 s | 4.03 / 4.05 / 4.12 s |
| Its peak RSS | 3.9-4.6 GB | 4.5-4.6 GB |
| Viewer read, 3-channel ND2 2424², max of 23 z (3 readers / 1) | 610-760 ms | 660-940 ms |
| Its peak RSS | 285-398 MB | 375-386 MB |
| Opening for the viewer | 2.2-2.7 ms | 0.8-1.0 ms |

Same objects (112,941, same area sum). Within noise, no speed or RAM change.

## scopeio 0.0.0-alpha.2 and where the time goes (2026-10-25)

scopeio now normalizes to f32 in the same pass as the read
(`F32SignificantBits`, signed data `F32NormalizedShifted` like ImageJ), so the
native buffer and EVAnalyzer's own conversion pass are gone; the extension list
comes from scopeio's `FormatInfo`. Performance-check project (81 × 2048² `.vsi`;
the whole slide with its 5-channel series, now series 0), old / new build
alternated, 10 s apart:

| | before | after |
|---|---|---|
| 81 images: wall | 13.0 / 14.9 / 15.0 s | 13.1 / 13.9 / 14.1 s |
| 81 images: CPU / peak RSS | 230-234 s / 3.18-3.28 GB | 229-239 s / 3.15-3.36 GB |
| Whole slide: wall | 54.6 / 54.8 s | 54.1 / 58.9 s |
| Whole slide: CPU / peak RSS | 693-712 s / 8.25-8.29 GB | 699-715 s / 7.41-7.91 GB |

Same objects before and after (298,262 and 99,441, compared row by row without
ids). The whole slide had 99,473 objects with bioformats-rs (2026-10-24): 32
differ since the reader switch, not this update.

Against bioformats-rs (commit 2cf743ca, the last build before the switch),
same project, old / new alternated:

| | bioformats-rs | scopeio alpha.2 |
|---|---|---|
| 81 images: wall | 12.7 / 12.8 / 13.7 s | 12.4 / 12.5 / 12.5 s |
| 81 images: CPU / peak RSS | 217-225 s / 3.01-3.26 GB | 223-227 s / 3.30-3.33 GB |
| Whole slide: wall | 62.0 / 62.9 s | 55.2 / 55.7 s |
| Whole slide: CPU / peak RSS | 802-817 s / 7.50-7.77 GB | 693-697 s / 7.56-7.96 GB |

81 images: the same objects and intensities; physical sizes (nm) differ in the
7th digit (pixel size now in nm from the reader, before µm × 1000 in f32).
Whole slide: 99,473 → 99,441 objects.

Reading split RGB with one decode per tile (`ChannelSelect::Samples`) was tried
and dropped: on the uncompressed RGB fixture (4096 × 3000) the same time
(127-188 / 146-152 ms) but 157 → 286 MB peak, because the planar buffer has to
be copied into one image per channel.

Where the time goes (debug log, CPU seconds summed over tiles):

| | 81 images | whole slide |
|---|---|---|
| Reading (decoding) | 5.5 s | 380 s of 533 s in the tiles (JPEG 2000) |
| Rolling Ball (sliding paraboloid) | 210 s of 242 s | 68 s |
| Median Subtract | - | 29 s |
| Single-threaded after the tiles | - | Tile Merge 5.6 s, Voronoi 6.3 s, export 1.7 s |

Sliding paraboloid, 2048², radius 4, pre-smooth, one idle core: 570 ms - 3×3
max 69 ms and 3×3 mean 61 ms before the slide, x passes 49 ms each, y 78 ms,
diagonal half-passes 30-36 ms.

## Rolling Ball pre-filters (2026-10-25)

The sliding paraboloid's two 3×3 pre-filters (maximum and mean) worked
column by column; now row by row, the maximum's shift summed in the original
order, so the output is the same to the last bit (tested against the old code,
end to end). 2048² image, one core: 3×3 mean 38-58 → 7 ms, 3×3 max 38-61 →
18-25 ms, whole background 385-399 → 308-342 ms. Copying the columns into
lines for the parabola pass brought nothing and was dropped. Performance-check
project, old / new alternated:

| | before | after |
|---|---|---|
| 81 images: wall | 12.9 / 12.9 / 13.9 s | 10.7 / 11.1 / 11.9 s |
| 81 images: CPU / peak RSS | 227-235 s / 3.21-3.42 GB | 190-195 s / 3.17-3.40 GB |
| Whole slide: wall / CPU | 52.5 / 53.5 s, 687-706 s | 53.0 / 55.3 s, 694-702 s |

Identical objects (298,262 and 99,441), every column except the ids.

## 3D-ready results: z columns and the schema version (2026-10-25)

Phase 1 of 3D support: the `objects` table got its z columns (`ndim`,
`bbox_z*`, `centroid_z_*`, `bbox_center_z_*`, `volume_*`,
`surface_area_nm2`, `touches_z_edge` - for a plane's objects zmin = zmax =
its plane, the volume columns NULL), the readers filter planes by the z
extent, `run.schema_version` records the format. `bench_pipeline` 50 ×
2048², 8 threads, `--classify 2`, 52,103 objects; base = HEAD before,
alternated, 10 s pauses, 22 cores:

| | base | 3D-ready |
|---|---|---|
| Pipeline time (2 runs each) | 2.08 / 2.05 s | 1.95 / 2.01 s |
| Peak RAM | 885 / 935 MB | 860 / 963 MB |
| Results file | 26.2 MB | 26.8 MB (+~10 bytes per object) |

- No measurable cost: time and RAM within the run-to-run spread.
- **Same results**: all 60 earlier `objects` columns compared row by row
  (without the random ids; class lists sorted - their order varies between
  two runs of the same build too): no difference. `measurements` and
  `images` the same. Only `pixel_size_z_nm` changed on purpose: an unknown
  z spacing is NULL now (was a number).

## 3D viewer: volume objects on their plane, orthogonal views (2026-10-25)

**2D unchanged.** `bench_viewport_whole_slide` (generated 60000 × 40000
slide, 1920 × 1080, 22 cores, default caches), base = HEAD without these
changes, 20 runs alternated (ABBA, BAAB, ABBA ×3), 6-10 s pauses:

| | base | new |
|---|---|---|
| Total of the 20 steps (open, zoom, pan away and back) | mean 435 ms, median 432 ms (422-455) | mean 438 ms, median 438 ms (421-455) |
| Process peak | 261-275 MB | 260-270 MB |

The difference (0.6 %) is within the spread between runs (one round each
way had the other build ahead). On the 2D path only a flag check per sharp
render was added (the orthogonal views off); a plane's objects are drawn as
before (`mask_on_plane` hands back the whole mask).

**Orthogonal views** (real app, Fly Brain: 256 × 256 × 57 planes, RGB as
3 channels, zoom 308 %): moving the crosshair reads both views' lines on all
57 planes in parallel in **7-8 ms**; a new display range or colour only
renders again: **0.2-1 ms**. Kept: **66 KB** (the two strips, u8). Nothing
runs or is kept while the views are off.

## 3D view: maximum intensity projection, turned freely (2026-10-25)

`editor/viewport_volume.rs`: the stack read once, reduced to at most
256 × 256 × 256 (pyramid level + every n-th pixel / plane), kept u8/u16;
one ray per pixel (nearest voxel, a step per voxel), smaller renders while
turning. Nothing runs or is kept while the view is off.

| | 22 threads | 2 threads |
|---|---|---|
| 256³ volume (worst case), turning: 384² | 11 ms | 36 ms |
| 256³ volume, at rest: 1024² | 53 ms | 244 ms |
| Fly Brain 256 × 256 × 57, 3 channels, turning: 384 × 212 (real app) | 3.5-4.5 ms | - |
| Fly Brain, at rest: 1024 × 566 (real app) | 24 ms | - |

(`cargo test --release -p evanalyzer_gui bench_volume -- --ignored
--nocapture`, one channel; the Fly Brain from the app's debug log.) Kept:
the volume, 3.7 MB per 8-bit channel for the Fly Brain (11.2 MB), at most
16.8 MB per 8-bit / 33.6 MB per 16-bit channel; the renders are a few MB.

## Pixel classifier: class and certainty in one pass (2026-10-27)

The random forest's certainty ("Minimum certainty", later the prediction
overlay) ran smartcore's `predict` and then every tree again on the whole
tile, holding every tree's votes for every pixel. Now `predict_with_share`
runs each tree once on chunks of 4096 rows, counts the votes (the class is
exactly smartcore's - tested, ties included) and the chunks run in parallel;
plain predictions take the same path. `bench_predict_with_certainty`
(core, release): one 512² tile, 20 features, 50 trees, 3 classes, forest
trained on 5000 samples; median of 3, peak RSS of the test process above
its start (the pixel rows themselves ~27 MB).

| | before | 1 thread | 4 threads | 22 threads |
|---|---|---|---|---|
| Class + certainty | 1.58-1.60 s, +141 MB | 0.54 s, +62 MB | 0.16 s, +68 MB | 0.05-0.08 s, +102-104 MB |
| Class only (smartcore `predict`, before) | 1.05 s | | | |

Same classes everywhere; one thread is 3x faster than before with
certainty and 2x faster than smartcore's class-only `predict` (the chunks
fit the CPU cache). In an analysis the tiles already run in parallel, so
the per-tile thread gain there is smaller.

## Prediction overlay: drawing it (2026-10-27)

`bench_draw_prediction` (gui, release, one thread): the object layer of a
full-HD viewport with its 64 px margin (2048 × 1208) covered by 512²
prediction tiles of 3 classes, blended in the class colours; median of 5.

| Zoom | Time |
|---|---|
| 1 | 21.7 ms (20.1-24.3) |
| 4 | 19.7 ms |
| 0.25 (the tiles cover a quarter of the screen) | 4.0 ms |

Drawn with the objects (after a change, not per frame - a pan moves the
layer). Memory: 2 bytes per pixel of every preview tile (class +
probability), 512 KB per 512² tile, kept until the next preview.

## Undo history on a large project (2026-10-27)

Every undo step stored a full copy of the project's settings, including
every image entry and every hand-drawn object. Now the image list
(`ImageList`) shares the map and its entries between copies until an edit
changes them. `bench_undo_snapshots` (gui, release): 100,000 images (one
series, 4 named channels), 2,000 hand-drawn objects (40×40 masks); 3 runs
each.

| | before | after |
|---|---|---|
| One snapshot | 32-34 ms, ~240 MB | 0.0 ms, ~0 |
| 100 edits (each an undo step) | 10.9-11.4 s, +23,897 MB | <0.01 s, +0-1 MB |
| 100 edits of one image's objects (as drawing) | as above | 0.62-0.81 s, +960 MB |

An edit of an image still copies the map's pointers (~9.6 MB at 100,000
images) per undo step. That is bounded by the undo history's RAM budget
(Settings › Viewport › Undo history: Auto or MB; at least 10 steps, at most
100). Auto: 10 % of the RAM left after a full analysis on every core but
one, the viewport caches and 512 MB - 8 GB / 4 cores: 510 MB, 16 GB / 22
cores: 375 MB. Same benchmark, 100 edits of one image's objects:

| Budget | Steps kept | Time | RSS |
|---|---|---|---|
| 256 MB | 28 (~250 MB estimated) | 0.71-0.79 s | +288 MB |
| 2 GB | 100 (~917 MB estimated) | 0.79-0.80 s | +960 MB |

## Results views: queries off the UI thread (2026-10-27)

The list, charts, plate, well and image heatmap queried the database on the
UI thread - the window froze for every filter tick, page or chart change.
Now they run on the database thread; a refresh replaced by a newer one
before it starts is skipped (quick clicks query once; page clicks all run,
in order); "Loading" shows only after 400 ms and then for at least 400 ms.
The image selection uses sets, and all images ticked is no image filter.

`bench_results_actions` (gui, release): each action through its Slint
callback as a click, on a copy of the 1.7 GB results file (1,776 images);
UI blocked = the callback, shown after = the database thread idle and the UI
queue drained. Median of 3 per run; three runs each, alternating (one per
build, 10 s apart). RSS after the action.

| Action | UI blocked before | after | Shown after before | after | RSS before | after |
|---|---|---|---|---|---|---|
| t changed | 25-31 ms | 0.0 ms | 26-31 ms | 28-30 ms | 215-219 MB | 213-221 MB |
| class off + on | 50-56 ms | 0.0-0.1 ms | 51-57 ms | 25-31 ms | 228-236 MB | 226-228 MB |
| group by images + back | 98-101 ms | 0.1 ms | 99-101 ms | 26-30 ms | 350-353 MB | 232-234 MB |
| next page | 25-29 ms | 0.0-0.1 ms | 25-29 ms | 23-34 ms | | |
| previous page | 27-32 ms | 0.0-0.1 ms | 27-32 ms | 29-32 ms | | |
| all images + none | 51-63 ms | 0.0-0.1 ms | 52-64 ms | 24-27 ms | | |
| chart: scatter + histogram | 32-35 ms | 0.1 ms | 32-35 ms | 21-23 ms | 381-384 MB | 246-248 MB |
| plate: class off + on | 8 ms | 0.0-0.1 ms | 8 ms | 9-10 ms | | |
| 10 × next page (quick) | 270-315 ms | 0.0 ms | 273-316 ms | 269-299 ms | 392-394 MB | 266-270 MB |
| 100k images: select all | 15-17 ms | 14-21 ms | 19-21 ms | 18-23 ms | | |
| 100k images: a keystroke in the search | 7-8 ms | 7 ms | | | | |
| 20k images all ticked: clear the search | 8,261-8,597 ms | 2.9-3.5 ms | | 4.0-4.7 ms | | |

The UI thread is free during every query now. Two-step actions (off + on,
there + back) query once instead of twice, so they show sooner and the
process stays ~100 MB lower (the skipped "group by images" query is the
biggest). Quick page clicks still read every page (each starts where the
one before ended). "Select all" of 100,000 images still builds the
selector's rows on the UI thread (14-21 ms).

## Prediction overlay: its cost in the step (2026-10-27)

`bench_prediction_overlay_in_the_step` (core, release): the AI Pixel
Classifier step on one 512² tile, the panel's "Recommended" features (10),
a forest of 50 trees (depth 20, 3 classes); 5 runs per case, the cases
alternating, peak RSS above the start.

| Threads | Overlay off | Overlay on | Extra RAM |
|---|---|---|---|
| 1 | 0.243-0.279 s | 0.243-0.245 s | ~0-2 MB |
| 4 | 0.181-0.184 s | 0.180-0.217 s | ~1-3 MB |

No measurable time: a forest's plain prediction already counts the votes
the certainty comes from. The overlay adds a copy of 2 bytes per pixel
(512 KB per 512² tile), kept by the editor until the next preview, and
drawing it (~20 ms full HD, "Prediction overlay: drawing it").

## Trained models in a pipeline run (2026-10-27)

A random forest or MLP from the AI training panel made a run's RAM jump by
GBs. Four causes, fixed one by one:

1. The forest kept smartcore's out-of-bag samples (per tree, which samples
   it used - on by default, read by nothing): 57 MB of JSON for 200,000
   samples. Now never kept; old models load without them, the panel's
   checkbox is gone.
2. The certainty read the trees by turning the whole forest into a
   `serde_json::Value` (every stored bool a ~32-byte value): a ~600 MB peak
   per model copy. Now the forest keeps the trees themselves.
3. Every worker thread loaded and kept its own copy of the model. Now one
   copy per run, shared; dropped when the last running job ends.
4. The MLP predicted with the training backend (`Autodiff`, which keeps
   every layer's output) on the whole tile at once: ~230 MB per 512² tile
   and thread. Now the plain backend in blocks of 4096 rows, in parallel
   (identical results - tested).

`bench_trained_model_in_a_run` (core, release): the model saved to disk,
then 8 worker threads load it through the cache and predict 16 tiles of
512² with certainty (20 features, 3 classes). Forest: 50 trees, depth 20,
200,000 samples, out-of-bag on (the old default); MLP: 64, 32. Three runs
each, alternating; peak RSS of the test process above its start.

| | Model file | 16 tiles, 8 threads | Peak RAM |
|---|---|---|---|
| Forest, before | 57.2 MB | 1.87-1.89 s | +5,881 MB |
| Forest, after 1 + 2 | 6.0 MB | 1.33-1.35 s | +78-79 MB |
| Forest, after 3 | 6.0 MB | 1.28-1.31 s | +42-56 MB |
| MLP, before | 14 KB | 0.72-0.74 s | +3,838-3,930 MB |
| MLP, after 3 | 14 KB | 0.73-0.76 s | +3,805-4,009 MB |
| MLP, after 4 | 14 KB | 0.22-0.25 s | +41-52 MB |

With 22 threads (a typical analysis) the "before" numbers grow with the
threads; the "after" ones hardly. Still open: freed memory isn't handed
back to the system after a run (`malloc_trim`), and the AI plugin keeps
every deep-learning model loaded until the app ends (TODO).

## Whole slide: the cache budget in the tile merge - baseline (2026-10-28)

`bench_whole_slide` (app example, release): `tests/haut` (9343 × 27143,
5 channels, 5 z, series 0) through the same path as `cli analyze` - load
channel 0 with a max projection, Gaussian blur, Otsu, objects, Voronoi cells
(max radius 50 px). Peak RSS of the process, the spill folder's peak size
(TMPDIR), 10 s apart. 212,413 objects every run.

| Threads | Wall | Peak RSS | Spill peak |
|---|---|---|---|
| 21 | 26.3 / 27.2 / 33.9 s | 7.38-7.56 GB | 6.09 GB |
| 4 | 50.1 / 54.9 s | 7.42-7.44 GB | 0 |
| 4, only 6.0 GB available | 54.8 s | 6.07 GB, then the kernel's OOM killer | 6.09 GB |

Over time (`BENCH_TRACE=1`, 21 threads): in the tile phase RSS climbs to
~7.1 GB with nothing spilled - every merge accumulator stays under its own
cap (available RAM / workers), so all channel tiles stay in RAM (5 channels
× 253 MPx × f32 = 5.1 GB). Only the final merge puts them into one
accumulator: 6.1 GB spilled in ~3 s, RSS down to 3.2 GB. The peak grows
with the slide, not with the budget.

Low RAM: another process held memory until 6.0 GB were available. The run
(4 threads) went to 6.07 GB in the tile phase and the kernel killed the
largest other process (the memory holder) - on an 8 GB PC it would have
been EVAnalyzer. Spilling started only at the final merge (t ≈ 31 s).

## Whole slide: one cache budget for the merged tiles - (a) vs (b) (2026-10-28)

Both variants give the merged channel tiles of a tiled image one cap:
available RAM minus every worker's full tile estimate
(`recommended_merged_cache_bytes`).

- (a) Tiles run in parallel, each finished tile is merged into one cache,
  one at a time (instead of `try_reduce`, whose partial results each had
  the full per-worker cap).
- (b) The channel tiles go to one store shared by all tile caches
  (`ImageCache::share_channels`, each copy sees only its own keys); the
  merge stays a `try_reduce`.

`bench_whole_slide` on `tests/haut` as above, today / (a) / (b) alternated,
10 s apart. Same 212,413 objects and area in every run.

| | today | (a) | (b) |
|---|---|---|---|
| 21 threads: wall | 22.9 / 24.0 s | 18.6 / 20.1 s | 18.3 / 19.9 s |
| 21 threads: CPU (user + sys) | 225-238 s | 218-222 s | 220 s |
| 21 threads: peak RSS / spill | 7.50-7.66 GB / 6.09 GB | 7.56-7.57 GB / 0 | 7.62-7.87 GB / 0 |
| 4 threads: wall | 42.1 / 42.4 s | 40.3 / 46.4 s | 43.3 / 44.3 s |
| 4 threads: peak RSS | 7.35-7.49 GB | 7.26-7.31 GB | 7.36-7.40 GB |
| 4 threads, ~6 GB available: peak RSS | 5.86 / 6.07 / 6.28 GB | 5.27 / 5.42 / 5.43 GB | 4.24 / 5.51 / 5.55 GB |
| 4 threads, ~6 GB available: wall / CPU | 49-51 s / 156-160 s | 50-54 s / 156-162 s | 52-56 s / 156-158 s |
| 4 threads, ~4 GB available: peak RSS | 5.44 / 5.49 GB | 3.47 / 3.57 GB | 3.17 / 3.26 GB |
| 4 threads, ~4 GB available: wall / CPU | 47.8 / 48.9 s, 156 s | 50.3 / 51.7 s, 157-159 s | 51.9 / 57.5 s, 156 s |

With plenty of RAM today's run spilled 6 GB at 21 threads (one accumulator
over its per-worker cap at the end); both variants keep it in RAM - 15-20 %
faster. With little RAM today's peak doesn't follow the RAM at all (5.4 GB
with 4 GB available - into swap); both variants stay under it, spilling
while the tiles run. (b) is ~0.3 GB lower (the running tiles' channels count
toward the one cap) but a little slower under pressure (every channel read
and every spill write goes through the one shared lock). Same CPU.
"Available" is MemAvailable after another process held the rest; the
container has swap, so the numbers above it are noisy.

## Whole slide: one unnamed spill file (2026-10-28)

The image cache spilled every evicted image into its own named file in a
temp folder, deleted only when the cache was dropped normally - a killed
process (OOM killer, End task, power cut) left them. Now one unnamed file
per cache (`tempfile::tempfile`; Linux: unlinked, Windows: delete on close),
images at offsets with positional reads/writes, freed ranges reused - the OS
deletes it however the process ends, and it is one handle per cache.

`bench_whole_slide` on `tests/haut`, 4 threads, ~4 GB available (another
process holds the rest), named / unnamed alternated, 10-15 s apart. Spill
6.09 GB in every run (the benchmark counts unnamed files through
`/proc/self/fd`), same objects.

| | named files (before) | one unnamed file |
|---|---|---|
| Wall | 58.1 / 53.8 / 54.2 / 53.8 / 54.3 / 56.2 s (mean 55.1) | 51.7 / 58.8 / 58.1 / 54.8 / 55.0 / 50.8 s (mean 54.9) |
| CPU (user + sys) | 168-171 s | 166-172 s |
| Peak RSS | 3.38-3.77 GB | 3.36-3.62 GB |
| Left after `kill -9` mid-spill | 1.2 GB in 27 files | nothing |

A first round (load average 11-12 from other work) was left out: 81.9 /
69.1 s.

Parallel writes into one file: the OS serialises them (one inode lock).
Micro-benchmark, 64 MB writes into the page cache (overlay fs), 8 threads:
one file per write 16-20 GB/s, one shared file 4.8-5.1 GB/s, 2 / 4 / 8
shared files 9.2 / 13.2 / 17.7 GB/s; reads the same (~25 GB/s). It doesn't
show in the run: a cache's spills already happen one at a time, under its
lock (TODO: once they leave the lock, a few files per cache).

Temp disk space - for the system requirements: the spill is only used when
RAM is short; worst case per z/t plane width × height × channels × 4 bytes
(+ 4 bytes per pixel for each channel variant a Load Image Channel step
reads), independent of the file size - `tests/haut` 9343 × 27143 × 5
channels: 5.1 GB (6.1 GB measured), an RGB slide 60000 × 40000: ~29 GB.

## Exports of a 4,000-image batch (2026-10-28)

Synthetic results like today's largest batch: core `bench_export
--images 4000 --objects-per-image 1000 --object-size 12` (4M objects, 1.8 GB
file). `bench_results_export` (app example): every image, columns Count,
Area, Intensity avg (channel 0), Circularity, aggregations Avg + Median; one
export per process.

| Export | Wall | Peak RSS | Output |
|---|---|---|---|
| Grouped by image, CSV | 1.0 s | 654 MB | 6 MB |
| Grouped by image, XLSX | 1.8 s | 726 MB | 5 MB |
| List of all objects, CSV | 17.5 s | 1,124 MB | 100 MB |
| List of all objects, XLSX | refused after 7.1 s ("more than an XLSX sheet holds: export it as CSV, or one file per image") | 617 MB | - |

The audit's export items (2026-10-27: all-images list "minutes per page",
grouped export "4+ GB") were estimated for 100k images - at 4k images
neither shows.

## Measure Object Colocalization: one object at a time (2026-10-28)

The step collected every object's (A, B) pixel pairs before computing
anything (8 bytes per object pixel). Now two passes, one object's pixels
at a time per thread: pass 1 computes each object's values and collects the
different values for the RWC ranks (which rank over all objects' pixels),
pass 2 reads the pixels of the objects with pixels above both thresholds
again for RWC. Objects are processed tile by tile (in parallel within a
tile): a first version spread the threads over the slide, and with the
channel tiles spilled the cache reloaded tiles again and again (low RAM:
72-75 → 89-106 s, system CPU 17 → 73-102 s).

`bench_whole_slide --coloc 1` on `tests/haut` (Measure Object
Colocalization of the Voronoi cells, channels 0/1, 15 %, Costes "Faster"),
before / after alternated, 10-15 s apart. The hash over every measurement
value (`bit_xor(hash(key, value))`) is the same in every run - the values
are bit-identical.

| | before | after |
|---|---|---|
| Cells within 50 px, 21 threads: wall | 28.7 / 30.0 s | 27.7 / 28.6 s |
| CPU (user + sys) | 290-291 s | 299 s |
| Peak RSS (tile phase) | 7.76-7.84 GB | 7.74-7.81 GB |
| Cells covering the slide (no radius limit, 276M object pixels), 21 threads: wall | 62.0 / 64.0 s | 56.3 / 57.9 s |
| CPU | 612-613 s | 625-626 s |
| Peak RSS | 11.03-11.04 GB | **8.04-8.22 GB** |
| Cells within 50 px, 4 threads, ~4 GB available: wall | 73.6 / 70.0 s | 72.2 / 70.0 s |
| CPU | 211-213 s | 225-226 s |
| Peak RSS | 3.58-3.77 GB | 3.52-3.60 GB |

CPU +2-6 %: the pixels are read twice (pass 2). Wall is shorter with RAM to
spare (the pixels are now read in parallel).

## Results files closed when unused, DuckDB's memory in the budget (2026-10-28)

Every results file opened in a session kept its DuckDB instance, buffer
pool and file handle until the app ended. Now the connection cache counts
each file's users and closes it with the last one; `finish_run`
checkpoints.

`bench_results_sessions` (app example): four copies of the 4M-object file
(`bench_export --images 4000 --objects-per-image 1000`) opened one after
another, each read (grouped-by-image export) and closed; old / new
alternated.

| | before | after |
|---|---|---|
| RSS after closing file 1 / 2 / 3 / 4 | 276 / 520 / 763 / 1,009 MB | 47-55 MB, flat |
| Open file handles | 5 → 8 | 4 |
| Peak RSS | 1,286-1,294 MB | 572-578 MB |
| Reopening the same file (open + read) | 0.7 s | 0.7 s |

DuckDB itself during a whole-slide run (`bench_whole_slide --coloc 1`,
`duckdb_memory()` sampled): 1.3-1.6 GB, limit 8 GB here; the WAL is 0 MB at
the end (large appends go straight into the file). The merged cache now
leaves it its memory - at most a quarter of the free RAM, never below the
tiles in work. Reserving the whole limit (from the total RAM: 8 GB with 4 GB
free) starved the cache: 73 → 307-326 s, system CPU 32 → 800-839 s.

| `tests/haut` + coloc | before | after |
|---|---|---|
| 21 threads: wall | 28.2 / 29.3 s | 26.9 / 27.6 s |
| 21 threads: peak RSS | 7.61-7.63 GB | 7.57-7.59 GB |
| 4 threads, ~4 GB free: wall | 72.8 / 72.9 / 74.7 / 99.0 s (once over the RAM) | 79.0 / 79.0 s |
| 4 threads, ~3 GB free: wall | 130.2 / 198.9 s | 129.4 / 334.6 s |

At ~3 GB free both are erratic: four threads leave the merged cache ~0.6 GB
either way and it reloads tiles from disk (the thread count, not the
reservation - TODO, RAM rework step 2).

## RAM rework step 2: threads that keep the merged cache, RAM per plane, containers (2026-10-28)

The automatic thread count only asked whether each worker's tile fits the
free RAM; on one tiled image every worker more also took a tile's worth from
the merged cache. With little RAM free the cache got too small and reloaded
tiles from disk again and again (a cliff, not a slope). Now, for one image of
several tiles, the most threads at which the merged cache still holds
`MERGED_CACHE_MIN_TILES` = 2 tiles of every channel
(`memory::analysis::tiled_image_parallelism`, through
`JobExecutor::recommended_parallelism`; a thread count set by hand is kept).
Also: the merged cache's cap measured again at every plane (after
`malloc_trim`, so what the planes before freed counts as free), and a
container's memory limit (cgroup) or the user's RAM limit (`--max-ram`; then `EVANALYZER_RAM_LIMIT_MB`) caps the RAM
EVAnalyzer plans with.

Where the threshold comes from - `tests/haut` + coloc, ~2.9 GB free, threads
set by hand (the merged cache with them):

| Threads | Merged cache | Wall | CPU (user + sys) |
|---|---|---|---|
| 4 | ~0.4 GB (~1 tile) | 307.2 s (the kernel killed the memory holder) | 206 + 763 s |
| 3 | ~1.0 GB (~2.7 tiles) | **88.6 s** | 196 + 32 s |
| 2 | | 120.3 s | 186 + 28 s |
| 1 | | 190.4 s | 171 + 20 s |

`bench_whole_slide --coloc 1` with the thread count chosen automatically
(as the app does), before / after alternated:

| | before | after |
|---|---|---|
| ~3 GB free: threads / wall | 4 / 225.5, 278.7 s | 3 / **95.1, 85.4 s** |
| ~3 GB free: CPU | 199 + 511, 193 + 732 s | 195 + 32, 185 + 31 s |
| ~3 GB free: peak RSS | 2.91-3.03 GB | 2.74-2.76 GB |
| ~4 GB free: threads / wall | 6 / 270.8, 250.6 s | 5 / **69.1, 70.2 s** |
| ~4 GB free: CPU | 199 + 992, 202 + 897 s | 192 + 59, 193 + 60 s |
| ~4 GB free: peak RSS | 4.31-4.41 GB | 2.97-3.22 GB |
| plenty of RAM: threads / wall | 21 / 27.2, 27.8 s | 21 / 26.9, 27.9 s |
| plenty of RAM: peak RSS | 7.62-7.64 GB | 7.57-7.68 GB |

With ~4 GB free the app picked 6 threads before - 3.6-3.9 × slower than
with the 5 it picks now; nothing changes with RAM to spare.

## RAM rework step 3: one DuckDB limit for all open results files (2026-10-28)

Each open results file got DuckDB's whole limit (a quarter of the RAM,
1-8 GB) - the results window and a running analysis twice. Now one limit for
all of them together, shared out when a file opens or closes
(`results_memory_limit_per_file_mb`, at least 512 MB each).

`bench_results_sessions --together` (app example): copies of the 4M-object
file open at once, each read in full (the list as CSV), on a simulated 8 GB
computer (`--max-ram 8192`, then `EVANALYZER_RAM_LIMIT_MB=8192`: DuckDB 2 GB in all); before = the
same build with every file getting the whole limit, alternated.

| | before | after |
|---|---|---|
| 2 files: DuckDB per file / peak RSS | 743 MB / 1.91-1.94 GB | 743 MB / 1.93-1.96 GB |
| 3 files: DuckDB per file / peak RSS | 743 MB / 2.68-2.71 GB | 614-621 MB / 2.43-2.45 GB |
| 3 files: wall / CPU | 45.8-46.4 s / 120-121 s | 46.2-47.0 s / 120-121 s |

Two files stay under their share (1 GB each) with this reading; from three on
the share caps DuckDB at no measurable cost in time. The total is now at most
the one limit (2 GB here), not 2 GB per open file.

## RAM rework step 5a: AI models released after an analysis (2026-10-28)

Deep-learning models stayed loaded on every device until the app ended.
Now released after every analysis (not after a preview, which runs again and
again): `evanalyzer_ai::release_models` → the plugin's
`eva_ai_release_models` (interface version 2 → 3). No real Cellpose model on
this machine - a synthetic TorchScript model with ~400 MB of weights (a
constant traced into it), loaded on two devices (`bench_released_models_free_their_ram`,
evanalyzer_ai with `test-support`):

| | RSS |
|---|---|
| before loading | 115 MB |
| loaded on two devices | 879 MB (+764) |
| released (+ `malloc_trim`) | 116 MB (+1) |

The next analysis loads the model again - how long that takes for a real
Cellpose-SAM model is not measured here (no model file on this machine).

## RAM rework step 4: viewer caches held small during an analysis (2026-10-28)

When a local analysis started, the viewer's tile and overview caches were
emptied - and grew back to their full size (1 GB + 256 MB automatic) while
the user looked at images during the analysis. Now they are held at their
smallest size (64 MB each, the most recently used tiles) until it ends.

`bench_viewport_whole_slide` (gui, release, `BENCH_TRIM=1`) on `tests/haut`:
open, zoom, 8 pans right, 8 back; with `BENCH_DURING_ANALYSIS=1` the
analysis starts after the open. Without it the caches behave as they did
during an analysis before (after the start's emptying). Two rounds each:

| | before | after |
|---|---|---|
| Cache (tiles + overviews) after the walk | 184 + 29 MB, still growing | 62 + 29 MB, flat |
| Peak RSS (process) | 552-571 MB | 484-492 MB |
| Pan right (new tiles) | 66-112 ms | 68-85 ms |
| Pan back (seen before) | 9-17 ms | 9-91 ms |

The cost: panning back during an analysis reads tiles again that no longer
fit. This slide fills the tile cache to only ~184 MB; a bigger one or a
longer look grew it to the full 1 GB + 256 MB during an analysis before.

## Most CPU cores and most RAM: --max-threads, --max-ram (2026-10-29)

New global options of every mode (and Settings › Performance, read at
startup; the command line wins): `--max-threads N` - the analysis, its
thread pool, the general pool (viewer, exports), the AI engine and DuckDB;
`--max-ram MB` - EVAnalyzer plans as if the computer had no more (a
planning limit, not enforced on every allocation). A server passes
`[workers] max_threads / max_ram_mb` to each worker. Replaces the
environment variables `EVANALYZER_RAM_LIMIT_MB` and `EVANALYZER_AI_PLUGIN`
(now `--ai-plugin`).

The real program, `evanalyzer [options] cli analyze` on `tests/haut` + coloc
(project saved by `bench_whole_slide`), 22 cores, 62 GB:

| Options | Threads | Image cache per worker / merged | Wall | CPU (user + sys) | Cores busy (CPU / wall) | Peak RSS |
|---|---|---|---|---|---|---|
| none | 21 | 2.00 / 26.0 GiB | 27.3-28.4 s | 285 + 9 s | 10.7 | 7.67 GB |
| `--max-threads 4` | 4 | 11.6 / 36.7 GiB | 53.9 s | 177 + 6 s | 3.4 | |
| `--max-ram 8192` | 11 | 0.47 / 1.08 GiB | 44.5-47.0 s | 220 + 70 s | 6.5 | 5.30 GB |
| `--max-ram 4096` | | | 59.4 s | | | 3.47 GB |
| both | 4 | 1.75 / 3.48 GiB | 64.5 s | 186 + 23 s | 3.2 | |

## Everyday images on a small computer (2026-10-29)

For the system requirements' smallest case (students' laptops): core
`bench_pipeline --images 50 --size 2048 --threads N` (Gaussian blur →
threshold → objects, 52,103 objects, results written), peak RSS sampled
from `/proc`. This container's desktop CPU - a laptop core is slower.

| Threads | Peak RSS | 50 images |
|---|---|---|
| 1 | 226 MB | 10.2 s (4.9 images/s) |
| 2 | 326 MB | 5.4 s (9.2 images/s) |
| 4 | 499 MB | 2.9 s (17.5 images/s) |

## Object layer: a selected object with 100,000 relatives (2026-10-29)

Selecting a region all cells are related to (Relate Objects) froze the
viewer: the family was a `Vec` looked up for every object drawn
(O(objects × children)), and a line was drawn to every child. Now the
family as `HashSet`s, objects culled to the view and its margin before
colouring and sorting (same pixels), and lines only to the family members
in view - the 1,000 closest to the selected object (the rest show by colour).

`bench_object_layer` (gui, release, ignored): `tests/haut` (9343 × 27143),
100,000 objects of 12 px over the slide, all related to one 64 px parent;
1920 × 1080 view; median of 5 redraws of the object layer.

| | before | `HashSet` + culling | + lines in view, at most 1,000 |
|---|---|---|---|
| zoomed in (zoom 1), nothing selected | 9.0 ms | 4.7 ms | 5.3 ms |
| zoomed in, the parent selected | 13,181 ms | 10,411 ms | **82 ms** |
| fitted, nothing selected | 7.5 ms | 7.8 ms | 8.4 ms |
| fitted, the parent selected | 6,430 ms | 3,892 ms | **19 ms** |

Without any lines (measured once): 12.9 / 17.3 ms with the parent
selected - a line costs 0.04 ms (fitted) to 0.1 ms (zoomed in, longer on
screen).

## Tracking: Track LAP, Track Kalman, Track Overlap (2026-10-29)

Core `bench_tracking <copy.evadb> --class 1 --distance 10 --tracker ...` on
a `make_time_series --wells 8 --fields 2` file (16 images × 48 time
points, 1,500 tracks per image; class 1: 432,575 objects). Links found: of
the 423,071 links of the file's own tracks. Peak RSS with `--no-truth 1`
(the tracker alone).

| Tracker | Time | Peak RSS | Links found (wrong) |
|---|---|---|---|
| Nearest Neighbor (before / after the store change) | 0.82-0.98 / 0.85-0.95 s | 384 MB | 423,006 (65) |
| LAP | 1.00-1.18 s | 203 MB | 423,038 (33) |
| Kalman | 1.00-1.17 s | 219 MB | 423,032 (39) |
| Overlap (masks read) | 1.28-1.29 s | 219 MB | 423,031 (40) |
| LAP / Overlap with merges and splits | 1.30-1.46 / 1.40-1.53 s | 356 / 385 MB | 423,013 (32) / 423,028 (38) |

The LAP trackers read one image at a time; the nearest neighbour tracker
still reads all objects of the class at once. Their first version wrote
the tracks per image: 7.4-7.8 s and 1.3-1.7 GB (with the benchmark's
links check), as every `UPDATE` rewrites the table - now one write at the
end. Writing `parent_track_ids` on every row made Nearest Neighbor 2.5 ×
slower and +480 MB; it is updated only where parents are or were.

## Peak RAM overview - for the system requirements

The highest peaks measured so far, per activity. "Test process" numbers
are without the window; add the idle app (~210 MB, plus ~150 MB without a
graphics driver).

| Activity | Peak RAM | Measured | Where |
|---|---|---|---|
| App idle after start (AI not loaded) | 207 MB | 2026-10-19 | The real app above |
| Analysis of 50 images 2048², 1 / 2 / 4 threads | 226 / 326 / 499 MB | 2026-10-29 | Everyday images on a small computer |
| Image viewer, whole slide, default caches | ~280 MB above start | 2026-10-17 | Image viewer |
| Image viewer, 2 threads, small caches | ~190 MB above start | 2026-10-17 | Image viewer |
| Viewer with a breakpoint tile compared | ~400 MB | 2026-10-18 | Breakpoint compare |
| Results views, 1.7 GB analysis (List, Plate, Charts, split) | 323 MB (test process) | 2026-10-17 | Results |
| Results views, 3.8M objects with masks (List, Plate, Charts, split) | 175 MB (test process) | 2026-10-23 | One drawing code, old vs new |
| Over time chart, 6.6M objects | 180-470 MB (query process) | 2026-10-21 | Time course |
| Time course export, 6.6M objects | 250-410 MB (query process) | 2026-10-21 | Time course |
| List export (a row per object), 6.6M objects, CSV | ~790 MB (query process) | 2026-10-21 | List export |
| Tracks: one image's tracks (1500 tracks) | ~60 MB (query process), ~1 MB kept | 2026-10-21 | Results › Tracks |
| Track export, a row per track (144,000 tracks) | ~410 MB (query process) | 2026-10-21 | Results › Tracks |
| Tracks, Full: every track's row (144,000 tracks) | ~300 MB (query process) | 2026-10-21 | Results › Tracks |
| Cellpose-SAM, one 1024² image, CPU | ~1.9 GB | 2026-10-19 | AI models |
| Running a pipeline: 200 images 2048², 8 threads | ~1 GB (1.05 GB with masks) | 2026-10-23 | Object masks in the results |
| Running a pipeline: 50 images 2048², 8 threads, with the z columns | 860-963 MB | 2026-10-25 | 3D-ready results |
| Image viewer, whole slide, after the 3D viewer changes (2D) | 260-275 MB (test process) | 2026-10-25 | 3D viewer |
| Orthogonal views of a z-stack (57 planes, 256²) | +66 KB kept | 2026-10-25 | 3D viewer |
| 3D view of a z-stack | +16.8 MB per 8-bit / 33.6 MB per 16-bit channel at most (Fly Brain: 11.2 MB) | 2026-10-25 | 3D view |
| Propagation, one 67 MP image (8192²) | 1.55 GB (test process) | 2026-10-22 | Propagation |
| Relate Objects, 416,000 spots in 42,000 cells | 577 MB (test process, 475 MB of it the objects) | 2026-10-23 | Relate Objects |
| Writing results: 1776 images × 2150 objects | ~2.4 GB (2.56 GB with masks) | 2026-10-23 | Object masks in the results |
| Writing results: 1776 images × 2150 objects, 6 channels with intensity spread | 2.57-2.64 GB (gray values only; 3.12-3.27 GB with normalized copies) | 2026-10-24 | Intensities in gray values only |
| Running a pipeline with intensity spread: 200 images 2048², 8 threads | 1.1-1.16 GB | 2026-10-24 | Measurement storage and object intensity spread |
| Expand Without Overlap, 42,000 objects on 8192², margin 8 / 100 px | 255 / 861 MB (test process; 100 px measured before the smaller feature transform) | 2026-10-24 | Expand Without Overlap |
| Measure Object Neighbors, 42,000 objects on 8192², expand until adjacent | 925 MB (test process) | 2026-10-24 | Measure Object Neighbors |
| Whole slide (5 channels, 21 tiles, 99,473 objects), performance-check project | 7.8-8.3 GB (was 9.1-9.2 GB) | 2026-10-24 | Reproducible runs and the whole slide |
| Measure Texture, 8192² image + 74,529 objects | 603 MB (test process, ~520 MB of it the test image) | 2026-10-24 | Measure Texture |
| Results views, 5.6M objects, measurement columns (plate, per image, list, histogram) | 113-228 MB (query process) | 2026-10-24 | Measurement storage and object intensity spread |
| Making the time series dataset (6.6M objects) | 2.4 GB | 2026-10-21 | `make_time_series` |
| Running a pipeline: 6 images 12288² (54 tiles), 8 threads | 4.5-4.6 GB | 2026-10-25 | One shared image reader |
| Reading a 3-channel ND2 2424² with a max projection of 23 z | ~390 MB (test process) | 2026-10-25 | One shared image reader |
| Whole slide (5 channels, 21 tiles, 99,441 objects), performance-check project | 7.41-7.91 GB (was 8.25-8.29 GB) | 2026-10-25 | scopeio 0.0.0-alpha.2 |
| Pixel classifier with certainty, one 512² tile (20 features, 50 trees) | +62 MB (1 thread) / +104 MB (22 threads) above the test process (was +141 MB) | 2026-10-27 | Pixel classifier: class and certainty in one pass |
| Prediction overlay tiles in the editor | 512 KB per 512² preview tile | 2026-10-27 | Prediction overlay: drawing it |
| Trained forest / MLP in a run, 16 tiles 512² on 8 threads | +42-56 MB / +41-52 MB above the test process (was +5.9 GB / +3.9 GB) | 2026-10-27 | Trained models in a pipeline run |
| Undo history, 100,000 images, 100 steps | +1 MB (other edits) / +960 MB (edits of an image) (was +23.9 GB) | 2026-10-27 | Undo history on a large project |
| Undo history with the budget, 100,000 images, drawing | budget (Auto: 32 MB-1 GB, e.g. 510 MB on 8 GB / 4 cores); 256 MB: +288 MB | 2026-10-27 | Undo history on a large project |
| Results actions, 1.7 GB results file | 213-270 MB (test process; was 215-394 MB) | 2026-10-27 | Results views: queries off the UI thread |
| Whole slide 9343 × 27143, 5 channels (`tests/haut`, 212,413 objects), 4 or 21 threads | 7.4-7.6 GB; with 6 GB available it overruns them (OOM) | 2026-10-28 | Whole slide: the cache budget in the tile merge |
| Same slide, one cache budget for the merged tiles (a), 4 threads with ~4 / ~6 GB available | 3.5-3.6 / 5.3-5.4 GB (was 5.4-5.5 / 5.9-6.3 GB) | 2026-10-28 | Whole slide: one cache budget for the merged tiles |
| Exports of 4,000 images / 4M objects: grouped by image, list CSV | 0.65-0.73 GB / 1.1 GB | 2026-10-28 | Exports of a 4,000-image batch |
| Whole slide `tests/haut` with Measure Object Colocalization, cells covering the slide, 21 threads | 8.0-8.2 GB (was 11.0 GB) | 2026-10-28 | Measure Object Colocalization: one object at a time |
| Results window, switching between 4M-object results files | ~50 MB after closing each (was +250 MB per file opened) | 2026-10-28 | Results files closed when unused |
| Whole slide `tests/haut` + coloc, automatic threads, ~4 GB free | 2.97-3.22 GB (was 4.31-4.41 GB, and 3.6-3.9 × slower) | 2026-10-28 | RAM rework step 2 |
| Three 4M-object results files open at once, 8 GB computer (simulated) | 2.43-2.45 GB (was 2.68-2.71 GB); DuckDB at most 2 GB in all | 2026-10-28 | RAM rework step 3 |
| Whole slide `tests/haut` + coloc with `--max-ram 8192` / `--max-ram 4096` (real CLI) | 5.30 GB / 3.47 GB | 2026-10-29 | --max-threads, --max-ram |
| A deep-learning model after an analysis | released: +1 MB (was +764 MB for ~400 MB of weights on two devices, until the app ended) | 2026-10-28 | RAM rework step 5a |
| Track LAP / Kalman / Overlap, 432,575 objects (16 images × 48 frames) | 203 / 219 / 219 MB; with merges and splits 356-385 MB | 2026-10-29 | Tracking |
| Track Nearest Neighbor, same objects (all read at once) | 384 MB | 2026-10-29 | Tracking |

Temp disk space for the image cache's spill: see "Whole slide: one unnamed spill
file" - up to width × height × channels × 4 bytes per plane when RAM is short.

DuckDB may use up to a quarter of the RAM per open results file (1-8 GB,
then it spills to a temporary file) - the queries above stay far below.

Not measured yet: the real app's total with a large results file open,
a run on a computer with little RAM.

## Not covered

- The RAM of the real app (window, fonts, GPU renderer): there is no
  display in the dev container. The numbers are the test process's;
  expect roughly 150-300 MB with an image open, plus what the caches may
  use (Settings › Viewport).
- Real weak hardware: "2 threads, small caches" runs on this machine -
  its disk and memory speed, no slow GPU.
- Running pipelines and analyses: not written up here yet - core's
  `bench_pipeline <workdir> --images 200 --size 2048 --threads 8`
  (~7 s, ~1 GB RSS) and `bench_export <out.evadb>` (~33 s, ~2.4 GB),
  measured 2026-10-21 (see the Peak RAM overview).
