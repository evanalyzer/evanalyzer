# Benchmarks - GUI speed and RAM

Measured 2026-10-17 on the dev container: 22 cores, 64 GB RAM, release
build. The analysis itself (running pipelines, writing results) is in
[performave.md](performave.md).

## How to run them

All in release mode, with libtorch on the library path:

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

## Not covered

- The RAM of the real app (window, fonts, GPU renderer): there is no
  display in the dev container. The numbers are the test process's;
  expect roughly 150-300 MB with an image open, plus what the caches may
  use (Settings › Viewport).
- Real weak hardware: "2 threads, small caches" runs on this machine -
  its disk and memory speed, no slow GPU.
- Running pipelines and analyses: see [performave.md](performave.md).
