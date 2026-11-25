# Plan: libtorch → Burn, crate `evanalyzer_nn`

Status: proposal (2026-11-25), branch `remove-libtorch`. Nothing built yet except the
empty `crates/nn` skeleton (80974378).

## Why now

- PyTorch deprecated TorchScript - the only format our libtorch engine can run.
- libtorch is the largest part of every package (CPU plugin, CUDA plugin < 2 GiB only
  since cu130), needs the plugin loader, the CUDA probe, the NVIDIA-wheel download,
  `LIBTORCH_BYPASS_VERSION_CHECK`, the glibc 2.28 floor and the "2x faster as a plugin"
  mystery.
- burn is held at 0.21 only because 0.22's `std` pulls `burn-tch`, whose `torch-sys`
  collides with `tch` 0.26. Without `tch` that block is gone.
- TorchScript files are code-like (TODO "Model files are code-like"). Safetensors are
  data only.

## Target architecture

```
gui ─┐
cli ─┼─ app ── core ── nn ── burn (the only crate depending on burn)
srv ─┘          │
                └─ smartcore (KNN, random forest - not neural networks, stay in core)
```

**`evanalyzer_nn` (neural networks)** owns everything that touches burn:

| Module | Content | Comes from |
|---|---|---|
| `api` | `Request`, `Task`, `Response`, `Image`, `Detection`, `EngineConfig`, `DeviceSelection`, `Pieces`, `PIECE_SIDES`, `piece_side_for`, `largest_model_input`, `box_iou` | `crates/ai/src/api.rs` minus `ffi`, `API_VERSION`, the `floats` byte serde |
| `runtime` | backend + device choice (once per process, lazily on the first model run), `EngineInfo`, thread budget | `crates/ai/src/lib.rs` `infer`/`start`/`release_models` |
| `scheduler` | model cache per device + path (+ mtime), slots per GPU, blocking when busy, `release_models` | `crates/ai/src/torch/scheduler.rs` (framework-free parts port 1:1) |
| `pieces` | receptive-field probe, halo, InstanceNorm detection, piece loop | `crates/ai/src/torch/pieces.rs` |
| `models::{unet, stardist, cellpose_sam, …}` | architectures (burn `Module`s), pre/post-processing that ran on tensors | `crates/ai/src/torch/*.rs` + new architecture code |
| `weights` | model file format, importers (PyTorch state dict, bioimage.io) | new |
| `mlp` | pixel/object classifier MLP + its training loop | `crates/core/src/ai_learning/model/mlp.rs` |
| `train` | later: StarDist/Cellpose training (TODO "StarDist … Cellpose … ~2-3 weeks each") | new, not in this migration |

Public surface stays plain data (slices in, `Vec<f32>`/`Detection`s out, no burn types),
so core's call sites (`algos/ai_segmentation/*`, `job_executor`, `ai_devices`, `memory`,
`tests/tile_ram`) change from `evanalyzer_ai::` to `evanalyzer_nn::` and little else.

**Removed**: `crates/ai` (plugin, `ffi`, `runtime/plugin.rs`, `runtime/cuda_probe.rs`,
`torch/*`, `test_support`, examples), `tch`, `libloading` (if nothing else uses it),
features `libtorch` / `libtorch-in-process` / `torch` / `torch-plugin`, alias
`build-ai-plugin`, `LIBTORCH*` env in `.cargo/config.toml`, `--ai-plugin` CLI flag,
libtorch + NVIDIA parts of `libs/download.sh`, `ai/cpu` + `ai/cuda` packaging,
`docs/convert_cellpose.py` (replaced, see D2).

## Decisions needed before starting

**D1 - Licence of YOLOv5 (blocker).** YOLOv5 has no paper; its only description is
Ultralytics' code, which is **AGPL-3.0**. Today we only decode the output of a
user-supplied TorchScript graph; in burn we must write the architecture ourselves,
and doing that from AGPL code breaks the clean-room rules. Options:
- a) drop YOLOv5 (recommended unless users depend on it);
- b) replace it by a detector with a permissive reference (e.g. YOLOX, Apache-2.0, has
  a paper), with a schema migration that turns YOLOv5 steps into the new step;
- c) keep YOLOv5 on a generic ONNX runtime (see D3) - no architecture code from us.

Cellpose (BSD-3), SAM (Apache-2.0), StarDist (BSD-3) and torch-em U-Net (MIT) are fine
to port.

**D2 - Model file format.** Recommended: one file `*.safetensors` per model, with
`__metadata__` = `{ "evanalyzer_arch": "unet2d", "arch_version": "1", "config": {…} }`
(channels, depth, filters, norm, n_rays, grid …). Self-describing, no pickle, mmap-able
(no copy of 1.2 GB Cellpose-SAM weights). Importers in nn turn foreign files into it:
- PyTorch `state_dict` `.pth`/`.pt` (burn-store `PytorchStore`, reads the pickle without
  executing it) - covers user-trained Cellpose models directly;
- bioimage.io packages (`rdf.yaml` + `pytorch_state_dict` weights + known architecture);
- StarDist Keras `.h5`: no pure-Rust HDF5 reader we trust → a small Python converter
  script (as `convert_cellpose.py` today) that writes the safetensors file.

Import happens when the user picks a foreign file (model path setting): converted once
next to it or into the project, the step then points to the `.safetensors`.

**D3 - Arbitrary architectures.** Today any TorchScript U-Net runs (we measure its
receptive field). With burn only architectures we wrote run. Options:
- a) a fixed set: torch-em `UNet2d` (bioimage.io's U-Nets, e.g. `joyful-pig`), the
  StarDist U-Net, Cellpose-SAM; anything else is refused with a clear message
  (recommended to start);
- b) additionally `tract` (MIT/Apache, pure Rust, CPU only) as a generic ONNX fallback -
  a second engine, more code, more binary. Only if users ask.

**D4 - Existing MLP classifier files.** `.evamodel` MLP weights are burn 0.21
`BinBytesRecorder` bytes; burn 0.22 can't read them (migration in 0.22 = "load with old
burn, export via burn-store"). Alpha tags 1.0.0-alpha.5…8 shipped, so files exist.
Recommended: store MLP weights as plain named f32 tensors (our own, framework-free:
`Vec<f32>` + shape per layer in the existing file), bump
`CURRENT_SAVED_CLASSIFIER_VERSION`, and a migration step that decodes the old
`BinBytesRecorder` layout itself (four `Linear`s max, a simple layout) - verified against
fixture files written *now* with burn 0.21. Alternative: say old MLP files must be
retrained (alpha).

**D5 - Backends.**
- CPU: burn-flex (pure Rust, replaces ndarray) or CubeCL CPU - picked by the spike (P0).
- GPU: wgpu first (Vulkan / Metal / DX12, every vendor, no extra download). CUDA backend
  (`cubecl-cuda`, needs NVRTC at runtime → a download again) only if measured
  clearly faster on NVIDIA. Check that burn's wgpu version equals Slint 1.19's
  (FemtoVG-wgpu renderer) - two wgpu versions = compiled twice + bigger binary.
- Old-GPU rule (`MIN_COMPUTE_CAPABILITY 7.5`) goes away; wgpu adapters that fail
  set-up fall back to the CPU.

## Rules that apply throughout

- Results: same as libtorch up to float rounding (parity tests below); on GPU vs CPU
  rounding only (feedback "results independent of hardware").
- Speed/RAM: every model at least as fast as libtorch on small images (512²-2048²),
  same or less peak RAM; idle RAM and startup unchanged or better (no backend, device
  or model set up before the first model run). Measured with `cargo build-linux`,
  old build in `/workspaces/target-baseline`, alternated with pauses.
- `release_models` after every analysis keeps working (models + GPU buffers freed).
- RAM plan: `shared_ram_bytes` / per-tile numbers re-measured for burn (they were
  libtorch's: U-Net 1.6 KB, StarDist 0.4 KB per px …), `tests::tile_ram` runs with
  burn stand-ins - no feature flag needed any more, so those tests always run.
- Thread budget: burn's CPU backend must use the job's `cores`, not its own pool size;
  check how it interacts with our rayon pool (oversubscription = slower).
- Licences: `cargo about` / `cargo deny` over the new tree (burn, cubecl, wgpu, naga …)
  before merging.

## Phases

### P0 - Spike: is burn fast enough on the CPU? (2-3 days, go/no-go)

On a scratch branch, nothing deleted:
1. burn 0.22 in a throw-away crate (outside the workspace, so `tch` doesn't collide).
2. The stand-in layouts from `crates/ai/src/torch/test_support.rs` (StarDist
   `2D_versatile_fluo`-like, U-Net 32/64 filters depth 4) rebuilt in burn, random
   weights.
3. Same sizes as BENCHMARK.md "AI models: RAM per inference" (512², 1024², 2048²,
   4096² in pieces), CPU, 4 and 22 threads, wall + CPU time + peak RAM; burn-flex vs
   CubeCL CPU; wgpu on the GPU here if there is one.
4. A ViT-L block (Cellpose-SAM's encoder) at 256² - its matmul speed decides P6.

**Gate:** burn CPU ≥ libtorch speed on 512²-2048² (or within noise) and ≤ RAM. If it
fails: stop, or the hybrid fallback (tract for CPU inference, burn only for GPU +
training) - a new decision, not part of this plan.

### P1 - Freeze references from libtorch (1-2 days, on the current code)

Before anything is deleted (afterwards libtorch can't be built next to burn 0.22):
1. Golden outputs: per model (real files where we have them: `cpsam_v2.pt`,
   `joyful-pig`, a StarDist `2D_versatile_fluo` export, a YOLO export if D1 keeps it;
   plus the stand-ins) a few fixed inputs (synthetic + 1-2 real tiles) → raw outputs
   (`[C,H,W]` f32) and the final objects (counts, areas) of the core step. Stored under
   `tests/nn_reference/` (large ones outside git, as `cpsam_v2.pt`).
2. MLP fixtures for D4: `.evamodel` files written by today's code + their predictions.
3. Baselines: `bench_ai_ram` per model, `bench_cellpose` (1024²), startup + idle RAM
   (BENCHMARK.md "The real app"), package sizes per platform, clean + incremental build
   time of the workspace. Old build kept in `/workspaces/target-baseline`.

### P2 - Swap the crate, upgrade burn (3-4 days)

1. `crates/nn`: `api` moved from `crates/ai` (minus `ffi`), `infer` / `release_models`
   / `set_plugin_path` removed or replaced; core depends on `evanalyzer_nn`.
2. MLP moved from core into `nn::mlp`, D4 format change + migration on burn **0.21**
   first (fixtures from P1 pass), then
3. delete `crates/ai`, `tch`, features, alias, env, `--ai-plugin`; upgrade burn to
   0.22 (backend generic gone: `Module`s without `<B: Backend>`, records → burnpack /
   our own format, `burn-dispatch` for runtime backend choice).
4. The model steps compile and fail with "not available yet" until their phase - the
   branch is not merged before P6.
5. `cargo test --workspace` green (MLP training + prediction tests, D4 migration).

### P3 - Runtime, scheduler, pieces (3-5 days)

1. Backend/device choice per `DeviceSelection` + `EngineConfig` (GPU slots, cores),
   lazily on the first run; `EngineInfo` ("burn 0.22 wgpu: <adapter>" / "CPU") for
   the log and the settings dialog (`ai_devices`).
2. Scheduler ported (framework-free: cache keyed by path + mtime, slots, blocking) with
   its tests (`scheduler.rs` tests port as they are).
3. Pieces: receptive-field probe + InstanceNorm detection on burn, same constants
   (`PLANNED_HALO`, `ALIGN`, `WHOLE_IMAGE_HALO`); tests from `pieces.rs`.
4. Model file loading (D2): safetensors + metadata → architecture config → module;
   mmap; size checks on every dimension from the file (untrusted input, TODO
   `.evamodel sizes untrusted` - same for nn files).
5. GPU fallback: a device that fails set-up or runs out of memory → CPU, logged once.

### P4 - U-Net (3-5 days)

1. torch-em `UNet2d` (MIT; depth, initial filters, gain, norm: None/BatchNorm/
   InstanceNorm/GroupNorm, final activation) in burn.
2. Importer: bioimage.io package with `pytorch_state_dict` → safetensors (key mapping
   tested on `joyful-pig`).
3. Pre/post as today: pad to a multiple of 16 with context, softmax / independent maps,
   foreground + boundary channel, crop.
4. Parity vs P1 golden outputs (max abs diff ≤ 1e-4 of the range, objects identical or
   documented), stand-ins too.
5. Benchmarks vs P1 baselines (speed, RAM per pixel → RAM plan numbers updated).

### P5 - StarDist (3-4 days)

1. StarDist 2D network (BSD-3: U-Net backbone from `config.json`: `n_rays`, `grid`,
   `unet_n_depth`, `unet_n_filter_base`, `net_conv_after_unet` …).
2. `docs/convert_stardist.py`: Keras `.h5` / bioimage.io TF model → our safetensors.
3. Padding to 16 (mirrored), grid cells, ray decoding as today; core post-processing
   unchanged.
4. Parity + benchmarks as P4.

### P6 - Cellpose-SAM (1-2 weeks)

1. Cellpose-SAM network (Cellpose BSD-3 + SAM ViT-L encoder Apache-2.0): patch embed
   (1-3 channels), 24 blocks with windowed + global attention, fixed 256² positional
   embedding, neck, flow head.
2. Importer: Cellpose's own `.pth` state dict (`cpsam`, `cpsam_v2`, user-trained) read
   directly - no Python converter any more; bf16 → f32 on load (f16/bf16 on GPU later,
   measured).
3. 256² tiles + blending as today.
4. Parity vs `cpsam_v2.pt` golden outputs; benchmark vs the 65-67 s / 1.9 GB per 1024²
   of the libtorch plugin (`bench_cellpose`).

### P7 - YOLO (per D1: 0 days or ~1 week)

Only if D1 b) or c). b): new step + schema migration (protocol bump), parity against the
reference implementation's own exported outputs instead of libtorch.

### P8 - Packaging, CI, docs (2-3 days)

1. `release.yml`, `ci.yml`, `docker.yml`, `check-glibc.sh` (2.28 floor reason gone),
   `release-notes.sh`, `libs/download.sh` (libtorch + NVIDIA wheels removed, DuckDB
   stays), Windows/macOS packaging (no `ai/` folders), the CPU/CUDA package split →
   one package per platform. ARM: Linux ARM64 gets AI for the first time (pure Rust).
2. README ("run on libtorch" section), LIMITATIONS, `docs/server.md`,
   `docker/README.md`, step help texts (model file formats), THIRD-PARTY-NOTICES +
   `about.toml`, `.cargo/audit.toml`.
3. Templates repo: Cellpose-SAM template points to the new model file.
4. TODO entries closed / rewritten: "AI plugin", "2x faster as a plugin",
   "Model files are code-like", libtorch thread determinism note, glibc floor, the
   "Another engine (e.g. Burn)" note; memories `burn-migration-deferred`,
   `cuda13-switch`, `release-tests-libtorch` updated.

### P9 - Final measurement and merge (1-2 days)

Whole run of P1's baselines against the new build: every model, small images + a whole
slide (wall, CPU, peak RAM), startup + idle RAM, package sizes, build time - into
BENCHMARK.md (incl. the RAM overview for the minimum-RAM estimate). Then
`cargo test --workspace --all-targets`, clippy, commit.

## Effort

| Phase | Days |
|---|---|
| P0 spike | 2-3 |
| P1 references | 1-2 |
| P2 crate swap + burn 0.22 + MLP | 3-4 |
| P3 runtime | 3-5 |
| P4 U-Net | 3-5 |
| P5 StarDist | 3-4 |
| P6 Cellpose-SAM | 5-10 |
| P7 YOLO | 0-5 |
| P8 packaging | 2-3 |
| P9 measure + merge | 1-2 |
| **Total** | **~5-8 weeks** |

## Risks

| Risk | Effect | Mitigation |
|---|---|---|
| burn CPU slower than libtorch/oneDNN | violates "at least as fast" | P0 gate before anything is deleted |
| ViT-L attention slow / RAM-heavy in burn | Cellpose-SAM regresses | P0 item 4; f16 on GPU; flash-attention kernels in CubeCL if available |
| Users' arbitrary TorchScript models stop working | broken pipelines | clear error naming the supported architectures + importers; D3 b) if needed |
| wgpu version clash with Slint | double compile, bigger binary | check in P0; align versions or pin |
| burn 0.x API churn (0.22 just removed backend generics) | rework per upgrade | keep burn types inside nn; plain-data API |
| Bigger binary / build time (cubecl, wgpu, naga) | slower CI, larger download | measured in P0 + P9; still far below libtorch |
| GPU drivers without Vulkan/DX12 compute | GPU unused | CPU fallback, logged |
