# Benchmarks - GUI speed and RAM

Measured 2026-10-17 on the dev container: 22 cores, 64 GB RAM, release
build. The analysis itself (running pipelines, writing results) is in
[performave.md](performave.md).

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
centre, compared), `BENCH_VIEWPORT=WxH`, `BENCH_PAN_STEPS`, `BENCH_SERIES`
(has no effect on `tests/Slide1_distal_zoom_40x.vsi` yet - it stays on the
512 × 375 series).

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

The real VSI (`tests/Slide1_distal_zoom_40x.vsi`, label series 512 × 375):
open 22 ms, every other step under 6 ms, about 50 MB.

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
under them yet (same pixels, checked by a test). They are drawn on the
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
models (`cargo run --release -p evanalyzer_ai_engine --features test-support
--example make_bench_models -- <dir>`: "light" = next to no compute, the
cost around the model; "heavy" = 6 convolution layers) and the real
Cellpose-SAM (`tests/cpsam_v2.pt`, 1.2 GB), CPU only, 22 shared cores
(other load on the host: expect ±10 %).

**The plugin boundary** - one Cellpose request on libtorch linked in vs.
through the plugin, same process (~16 s):

```sh
cargo test --release -p evanalyzer_ai_engine --features torch-plugin,test-support \
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
# with EVANALYZER_AI_PLUGIN=target/release/libevanalyzer_ai_engine.so and libtorch on LD_LIBRARY_PATH
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

## Peak RAM overview - for the system requirements

The highest peaks measured so far, per activity. "Test process" numbers
are without the window; add the idle app (~210 MB, plus ~150 MB without a
graphics driver).

| Activity | Peak RAM | Measured | Where |
|---|---|---|---|
| App idle after start (AI not loaded) | 207 MB | 2026-10-19 | The real app above |
| Image viewer, whole slide, default caches | ~280 MB above start | 2026-10-17 | Image viewer |
| Image viewer, 2 threads, small caches | ~190 MB above start | 2026-10-17 | Image viewer |
| Viewer with a breakpoint tile compared | ~400 MB | 2026-10-18 | Breakpoint compare |
| Results views, 1.7 GB analysis (List, Plate, Charts, split) | 323 MB (test process) | 2026-10-17 | Results |
| Over time chart, 6.6M objects | 180-470 MB (query process) | 2026-10-21 | Time course |
| Time course export, 6.6M objects | 250-410 MB (query process) | 2026-10-21 | Time course |
| Cellpose-SAM, one 1024² image, CPU | ~1.9 GB | 2026-10-19 | AI models |
| Running a pipeline: 200 images 2048², 8 threads | ~1 GB | 2026-10-21 | not written up yet (`bench_pipeline`) |
| Writing results: 1776 images × 2150 objects | ~2.4 GB | 2026-10-21 | not written up yet (`bench_export`) |
| Making the time series dataset (6.6M objects) | 2.4 GB | 2026-10-21 | `make_time_series` |

DuckDB may use up to a quarter of the RAM per open results file (1-8 GB,
then it spills to a temporary file) - the queries above stay far below.

Not measured yet: the per-object List export of a large file, the real
app's total with a large results file open, a run on a computer with
little RAM.

## Not covered

- The RAM of the real app (window, fonts, GPU renderer): there is no
  display in the dev container. The numbers are the test process's;
  expect roughly 150-300 MB with an image open, plus what the caches may
  use (Settings › Viewport).
- Real weak hardware: "2 threads, small caches" runs on this machine -
  its disk and memory speed, no slow GPU.
- Running pipelines and analyses: see [performave.md](performave.md).
