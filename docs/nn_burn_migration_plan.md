# Plan: libtorch → Burn, crate `evanalyzer_nn`

Status: planned (2026-11-25; D1-D5 decided), branch `remove-libtorch`. The `crates/nn` skeleton (traits, types, `todo!()` bodies) is built (2026-11-25).

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
cli ─┼─ app ── core ── nn ─┬─ burn       (the only crate depending on burn)
srv ─┘                     └─ smartcore  (the only crate depending on smartcore)
```

**`evanalyzer_nn` (neural networks and the other trained models)** owns every
machine-learning algorithm - all of them train and infer, behind one interface:

| Module | Content | Comes from |
|---|---|---|
| `model` | the `Model` trait (below), `Image`, `NnError` | new (replaces `api.rs`'s `Request`/`Task`/`Response` enums) |
| `model::{unet, stardist, cellpose_sam}` | architectures (burn `Module`s), their `Settings` + `Output` types, pre/post-processing that ran on tensors | `crates/ai/src/torch/*.rs` + new architecture code |
| `classifier::{mlp, knn, rtree}` | trained classifiers: MLP (burn), KNN + random forest (smartcore), their training (see "Training interface") | `crates/core/src/ai_learning/model/*.rs`, the `Classifier` enum of `ai_learning/model.rs`, the smartcore matrix conversion of `ai_learning/utils.rs` |
| `runtime` | backend + device choice (once per process, lazily on the first model run), `EngineConfig`, `DeviceSelection`, `EngineInfo`, thread budget | `crates/ai/src/lib.rs` `infer`/`start`, `api.rs` |
| `cache` | loaded models per (file, device) (+ mtime), slots per GPU, blocking when busy, `release_models` | `crates/ai/src/torch/scheduler.rs` (framework-free parts port 1:1) |
| `pieces` | receptive-field probe, halo, InstanceNorm detection, `run_in_pieces` helper, `Pieces`, `PIECE_SIDES`, `piece_side_for`, `largest_model_input` | `crates/ai/src/torch/pieces.rs`, `api.rs` |
| `weights` | safetensors + metadata format, importers (PyTorch state dict, bioimage.io) | new |
| `train` | `TrainControl`, `TrainProgress`, `TrainStats` shared by classifier and deep-model training; deep-model training itself (`Trainable`, targets, augmentation) is a later track | new |

### API: a trait per model instead of one big enum (decided 2026-11-25)

The `Request`/`Task`/`Response` enums only existed for the libtorch plugin: they crossed
a C interface as postcard bytes, and a Rust trait object can't. Without the plugin each
model is its own type:

```rust
pub trait Model: Sized + Send + Sync + 'static {
    /// What a step chooses per run (channels, softmax, pieces, ...).
    type Settings;
    /// What the model gives back - plain data, no burn types.
    type Output;
    /// Builds the architecture from the file's metadata, loads the weights.
    fn load(file: &Path, device: &Device) -> Result<Self, NnError>;
    fn infer(&self, image: Image<'_>, settings: &Self::Settings) -> Result<Self::Output, NnError>;
}

pub fn infer<M: Model>(file: &Path, image: Image<'_>, settings: &M::Settings)
    -> Result<M::Output, NnError>;
```

- Architecture config (depth, filters, rays …) is stored in the model - it belongs to
  the weights. Run settings are passed per call: the loaded model is cached per file +
  device and shared by every step using that file (Cellpose-SAM: 1.2 GB), so two steps
  with other settings must not need two copies.
- Static dispatch: each core step knows its model type (`nn::infer::<UNet>(…)` →
  `UNetOutput`). The only `dyn` is the cache (`Arc<dyn Any + Send + Sync>`, downcast).
- The engine stays global (`OnceLock`), set up lazily: the model steps sit deep in core
  without a handle to pass.
- `run_in_pieces` is a helper U-Net and StarDist call in their own `infer`.
- Classifiers get the same shape with `train` added (trait or inherent fn, decided when
  moving them); core hands rows as plain `&[f32]` + feature count, never a smartcore
  `DenseMatrix` or burn tensor.
- No `serde` on inputs/outputs any more (the `floats` byte encoding, `API_VERSION` go).

### Module layout

```
crates/nn/src/
├── lib.rs            re-exports, `infer::<M>()`, `release_models()`
├── errors.rs         NnError
├── train.rs          TrainControl, TrainProgress, TrainStats - shared by all training
├── classifier.rs     Samples, Predictions, trait Classifier, enum TrainedClassifier
├── classifier/
│   ├── mlp.rs        burn
│   ├── knn.rs        smartcore
│   └── rtree.rs      smartcore (random forest)
├── model.rs          Image, trait Model, trait Trainable, trait Dataset, InstanceLabels, ClassMask
├── model/
│   ├── unet.rs       UNet, UNetSettings, UNetOutput (+ later its training targets)
│   ├── stardist.rs
│   └── cellpose_sam.rs
├── runtime.rs        EngineConfig, DeviceSelection, EngineInfo, backend/device choice
├── cache.rs          loaded models per (file, device), slots, release
├── pieces.rs         Pieces, PIECE_SIDES, piece_side_for, run_in_pieces, receptive-field probe
└── weights.rs        safetensors + metadata, importers (PyTorch state dict, bioimage.io)
```

`mlp`, `knn`, `rtree` are under `classifier/`, not `model/`: they implement `Classifier`
(feature rows in), not `Model` (an image in, a cached model per device). nn does not
depend on `evanalyzer_cfg`: core maps cfg settings and `TrainingProgressEvent` to nn's
types, so nn stays at the bottom of the graph.

### Training interface (decided 2026-11-25)

**Classifiers** (`classifier.rs`). Object and pixel classifiers are the same to nn: feature
rows + labels. Where the rows come from (object measurements, filter responses at
annotated pixels) needs images, objects and filters - that stays in core.

```rust
pub struct Samples<'a> {
    pub rows: &'a [f32],      // n_samples * n_features, row-major (no Vec<Vec<f32>>)
    pub n_features: usize,
    pub labels: &'a [u32],    // dense class indices 0..n_classes
    pub n_classes: usize,
}

pub trait Classifier: Sized + Send + Sync {
    /// Hyperparameters: trees / k / hidden layers, epochs, ...
    type Params;
    fn train(samples: &Samples<'_>, params: &Self::Params, ctl: &TrainControl)
        -> Result<(Self, TrainStats), NnError>;
    /// Class + certainty per row.
    fn predict(&self, rows: &[f32], n_features: usize) -> Result<Predictions, NnError>;
}

/// Stored in the .evamodel - the type comes from the file, so here an enum is right.
pub enum TrainedClassifier { RandomForest(Forest), Knn(KnnModel), Mlp(Mlp) }
```

The `.evamodel` container with `AiLearningSettings` (cfg) stays in core; nn gives the
`TrainedClassifier` (serde for forest/KNN with today's names and layout, so their files
keep loading; safetensors bytes for the MLP). Classifiers run on the CPU, outside the
model cache (core's `ClassifierCacheScope` as today).

**Deep models: training and fine-tuning** (`model.rs`, later track). Images + label masks;
the targets are model-specific and live in the model's module (Cellpose: flows by
diffusion from the centre; StarDist: probability + ray distances; U-Net: class/boundary
maps).

```rust
/// Core loads the samples lazily - never all images in RAM at once.
pub trait Dataset<T>: Sync {
    fn len(&self) -> usize;
    fn get(&self, i: usize) -> Result<(OwnedImage, T), NnError>;
}
pub struct InstanceLabels { pub width: usize, pub height: usize, pub ids: Vec<u32> }   // Cellpose, StarDist
pub struct ClassMask      { pub width: usize, pub height: usize, pub classes: Vec<u8> } // U-Net

pub trait Trainable: Model {
    type Target;        // InstanceLabels / ClassMask
    type TrainParams;   // epochs, lr, batch, crop size, freeze encoder, augmentations
    /// `start`: None = new weights, Some(file) = fine-tune those.
    fn train(start: Option<&Path>, data: &dyn Dataset<Self::Target>,
             params: &Self::TrainParams, ctl: &TrainControl) -> Result<(Self, TrainStats), NnError>;
    /// Safetensors + metadata (architecture, base model, training params, seed).
    fn save(&self, file: &Path) -> Result<(), NnError>;
}
```

Augmentation (random crops, flips, rotations) and the validation split are nn's. The
metadata (base model, params, seed) keeps a training reproducible. Each trainable model
tells its RAM per batch sample, so core plans the batch size from the RAM budget (like
the tile RAM plan).

**Shared** (`train.rs`):

```rust
pub struct TrainControl<'a> {
    pub progress: &'a (dyn Fn(TrainProgress) + Sync), // core maps to cfg's TrainingProgressEvent
    pub cancel: &'a AtomicBool,
    pub seed: u64,
    pub device: DeviceChoice,   // CPU or GPU for training
    pub cores: usize,
}
```

Replaces `fit_mlp`'s `Sender<TrainingProgressEvent>` + `Arc<AtomicBool>` (cfg types).
Training needs burn's `std` + `optim` - possible only after `tch` is gone (P2).

### Review of the skeleton (2026-11-25)

To fix in `crates/nn` before building on it:
1. `Model::infer` without `&self` can't reach the weights → `fn infer(&self, image, settings: &Self::Settings)`.
2. Run settings are struct fields (`UNet { foreground_channel, softmax, … }`) and the
   struct has no weights → the struct holds the network + architecture config; run
   settings are `type Settings`, passed per call (the cache shares one model).
3. `load(file, &EngineConfig)` gets the process-wide config → `load(file, &Device)`; the
   cache picks the device.
4. Cellpose: `input_channels` is architecture (file metadata); `net_width`/`net_height`
   and `replicate_gray_channel` are run settings.
5. `i32` with `-1` = none → `Option<usize>` / `usize`.
6. Output fields private → `pub`; names `…Output` (not `…Response`), `StarDist`.
7. `NnError`: drop `Xml`/`JsonParsingError`; add `UnsupportedArchitecture`,
   `InvalidModelFile`, `Cancelled`, `Device` (lost / out of memory). Messages are shown
   to the user.
8. License header missing in `lib.rs`, `model.rs`, the model files; `errors.rs` dated
   2026-06-05.
9. `Cargo.toml`: `postcard` not needed; no `evanalyzer_cfg` dependency.
10. `mlp`/`knn`/`rtree` from `model/` to `classifier/`; `Pieces` to `pieces.rs`; unused
    import in `cellpose_sam.rs`.

Public surface stays plain data, so core's call sites (`algos/ai_segmentation/*`,
`job_executor`, `ai_devices`, `memory`, `tests/tile_ram`, `ai_learning/*`) change from
`evanalyzer_ai::` / smartcore / burn to `evanalyzer_nn::`. After P2 core depends on
neither burn nor smartcore.

**Removed**: `crates/ai` (plugin, `ffi`, `runtime/plugin.rs`, `runtime/cuda_probe.rs`,
`torch/*`, `test_support`, examples), `tch`, `libloading` (if nothing else uses it),
features `libtorch` / `libtorch-in-process` / `torch` / `torch-plugin`, alias
`build-ai-plugin`, `LIBTORCH*` env in `.cargo/config.toml`, `--ai-plugin` CLI flag,
libtorch + NVIDIA parts of `libs/download.sh`, `ai/cpu` + `ai/cuda` packaging,
`docs/convert_cellpose.py` (replaced, see D2).

## Decisions

**D1 - YOLOv5: dropped (decided 2026-11-25).** Its only description is Ultralytics'
AGPL-3.0 code, so writing the architecture ourselves would break the clean-room rules.
The YOLOv5 step, its settings (cfg), `algos/ai_segmentation/yolov5.rs` and the
`Task::Yolo` / `Detection` / `box_iou` API parts are removed. No backwards
compatibility: a pipeline with the step fails to load like any unknown command.
Cellpose (BSD-3), SAM (Apache-2.0), StarDist (BSD-3) and torch-em U-Net (MIT) are fine
to port.

**D2 - Model file format: safetensors (decided 2026-11-25).** Each architecture is Rust code in nn; the
weights come from a file. The file must also say *which variant* of the architecture
it is for (a U-Net's depth, filters, channels, norm; StarDist's rays and grid), so the
right module is built before the weights go in. Recommended: one `*.safetensors` file
per model (named f32 tensors + a string map `__metadata__` with
`{ "evanalyzer_arch": "unet2d", "arch_version": "1", "config": {…} }`). Data only (no
pickle), mmap-able (no copy of 1.2 GB Cellpose-SAM weights), independent of burn's own
formats (see D4). Foreign files are converted once when the user picks them:
- PyTorch `state_dict` `.pth`/`.pt` (burn-store `PytorchStore`, reads the pickle without
  executing it) - covers user-trained Cellpose models; the variant is read from the
  tensor shapes / the Cellpose defaults;
- bioimage.io packages (`rdf.yaml` gives the variant, `pytorch_state_dict` the weights);
- StarDist Keras `.h5`: a small Python converter script writes the safetensors file.
The step then points to the `.safetensors` file.

**D3 - Supported models only (decided 2026-11-25).** TorchScript is dropped; only the
architectures we implement run: torch-em `UNet2d` (bioimage.io's U-Nets, e.g.
`joyful-pig`), StarDist 2D, Cellpose-SAM. Anything else is refused with a clear message
naming the supported ones. No generic ONNX runtime.

**D4 - Weights never in burn's own format (decided 2026-11-25).** The MLP stores burn
0.21 `BinBytesRecorder` bytes - burn's serialization of its module records, which 0.22
replaced (burnpack). burn is 0.x, so this can happen on any upgrade. Rule: every file we
write (MLP classifiers, later trained StarDist/Cellpose models) stores weights in the D2
format - safetensors with our own tensor names - and nn maps those names onto the burn
modules itself. A burn upgrade can then only break nn's loading code, never files; a
test loads checked-in fixture files of every format so such a break shows in CI.
The MLP switches to safetensors too: `Classifier::Mlp.weights` holds safetensors bytes
(tensor names `layer{i}.weight` / `layer{i}.bias`, written and read with the
`safetensors` crate in nn, not burn's recorders). Existing MLP `.evamodel` files break
now (no backwards compatibility needed): bump `CURRENT_SAVED_CLASSIFIER_VERSION`, old
files are refused with "retrain the classifier".

**D5 - Backends (decided 2026-11-25).**
- GPU: wgpu (Vulkan / Metal / DX12, every vendor, no extra download). Check that burn's
  wgpu version equals Slint 1.19's (FemtoVG-wgpu renderer) - two wgpu versions = compiled
  twice + bigger binary.
- CPU: burn-flex (pure Rust, replaces ndarray). Not CubeCL CPU (burn feature `cpu`): it
  compiles kernels at runtime through LLVM (`cubecl-llvm` → `llvm-sys` 231 = the C++
  LLVM 23 library linked in and shipped) - against "almost pure Rust" (checked
  2026-11-25; dropped from nn's features).
- Fallback, in our code (burn doesn't switch backends by itself): on set-up nn asks wgpu
  for adapters; none, or only a software adapter (`DeviceType::Cpu`: llvmpipe/lavapipe,
  Windows WARP - very slow), or set-up fails → the CPU backend. A device lost / out of
  memory during a run → that inference again on the CPU, logged once. Same idea as the
  renderer's CPU-adapter detection (slow drawing notice).
- No CUDA backend for now (needs NVRTC at runtime = a download again); only if measured
  clearly faster on NVIDIA. The old-GPU rule (`MIN_COMPUTE_CAPABILITY 7.5`) goes away.

What wgpu changes against CUDA:
- Same: the model runs on the NVIDIA card; results equal up to float rounding (as CPU vs
  GPU today); no usable driver → CPU.
- Another API layer: wgpu reaches the card through Vulkan (Linux, Windows), DirectX 12
  (Windows) or Metal (macOS) instead of CUDA. The user needs only the normal graphics
  driver (NVIDIA's includes Vulkan) - no CUDA toolkit, no cuDNN, no 2 GB download.
- More hardware: AMD, Intel and Apple Silicon GPUs too; older NVIDIA cards (GTX 9xx/10xx)
  work again if their driver has Vulkan.
- The one real difference is speed: CUDA + cuDNN have kernels NVIDIA tuned for its own
  cards (convolutions, tensor cores); burn's kernels are generic (written once in CubeCL,
  translated to Vulkan / DX12 / Metal) and may be slower on NVIDIA - most likely for
  Cellpose-SAM's ViT-L. Measured in P0 (if a GPU is there) and P6.
- If wgpu is too slow on NVIDIA: burn's CUDA backend (CubeCL CUDA) as an extra option
  next to wgpu - needs the NVIDIA driver + NVRTC (~100 MB runtime library), far less
  than libtorch's CUDA package. Only if the measurements justify it.

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

In `crates/nn`, nothing deleted:
1. burn 0.22 in nn next to `tch` - resolves only without burn's `std`, `store`
   (`safetensors`/`pytorch`) and `optim` (checked 2026-11-25: they pull `burn-tch` via
   `burn-dispatch`, which collides with `tch`'s `links`):
   `burn = { version = "0.22.1", default-features = false, features = ["flex", "wgpu"] }`,
   `burn-flex = "0.22.1"` (its defaults std/simd/rayon: multi-threaded CPU),
   `burn-dispatch = { version = "0.22.1", default-features = false, features = ["rayon", "simd"] }`.
   Not the workspace burn (0.21, core's MLP) - two versions only during P0. nn compiles
   with these (checked 2026-11-25, skeleton); unverified: running at full speed without
   `std`; if not, exclude nn from the
   workspace for the spike or do P0 right after removing `tch`.
   Needs `cargo update -p combine --precise 4.6.9` first: `pliron` 0.19 (via
   `cubecl-core`) derives `Hash` on `combine`'s `SourcePosition`, which has it only from
   4.6.9; the lockfile had 4.6.7 → nn never type-checked and rust-analyzer showed no
   errors for it.
2. The stand-in layouts from `crates/ai/src/torch/test_support.rs` (StarDist
   `2D_versatile_fluo`-like, U-Net 32/64 filters depth 4) rebuilt in burn, random
   weights.
3. Same sizes as BENCHMARK.md "AI models: RAM per inference" (512², 1024², 2048²,
   4096² in pieces), CPU, 4 and 22 threads, wall + CPU time + peak RAM; burn-flex on
   the CPU, wgpu on the GPU here if there is one.
4. A ViT-L block (Cellpose-SAM's encoder) at 256² - its matmul speed decides P6.
5. On an NVIDIA GPU (if one is available): wgpu against libtorch CUDA, same cases -
   decides whether the CUDA backend (D5) is ever needed.
6. libtorch side measured in the same build with `crates/ai`'s `bench_ai_ram`, burn with a
   matching nn example (`cargo build-linux -p evanalyzer_nn --example <name>`),
   alternated.

**Gate:** burn CPU ≥ libtorch speed on 512²-2048² (or within noise) and ≤ RAM. If it
fails: stop, or the hybrid fallback (tract for CPU inference, burn only for GPU +
training) - a new decision, not part of this plan.

**First P0 run (2026-11-25)** - romantic-shark's layout (torch-em `UNet2d`, InstanceNorm,
depth 4, 64 features, 28.9 M parameters). libtorch: `bench_ai_ram unet-file=` the real
TorchScript (includes padding/crop around the network); burn: `bench_unet 64` (burn-flex,
random weights, network only). 22-core host, CPU, 2 runs each alternated with pauses, means:

| Case | libtorch wall / CPU / peak | burn-flex wall / CPU / peak | burn wall |
|---|---|---|---|
| 512², 4 threads | 1.76 s / 6.8 s / 463 MB | 2.36 s / 6.6 s / 434 MB | 1.34x |
| 512², 22 threads | 1.84 s / 35.5 s / 456 MB | 1.86 s / 18.7 s / 768 MB | 1.01x |
| 1024², 4 threads | 6.72 s / 24.7 s / 1,668 MB | 9.48 s / 26.7 s / 1,613 MB | 1.41x |
| 1024², 22 threads | 4.87 s / 85.5 s / 1,656 MB | 7.53 s / 79.5 s / 2,120 MB | 1.55x |
| 2048², 4 threads | 25.5 s / 90.4 s / 6,412 MB | 40.5 s / 112 s / 6,210 MB | 1.59x |
| 2048², 22 threads | 16.5 s / 263 s / 6,407 MB | 31.5 s / 332 s / 6,885 MB | 1.91x |

Gate not met as is: burn-flex 1.3-1.9x slower (only 512² on 22 threads equal), RAM equal on
4 threads, +5-70 % on 22. burn leaves cores idle (4 threads, 2048²: 112 s CPU in 40.5 s
= 2.8 cores busy) - serial parts. Next: profile (`perf`) which ops cost (3x3 convolutions,
InstanceNorm in several passes, bilinear interpolation) before deciding. wgpu not measurable
here (no NVIDIA driver in the container; the run hung > 5 min, likely on the Intel iGPU or
lavapipe).

**GPU run (2026-11-25)** - laptop RTX 3000 Ada (8 GB), driver 595, same models as above,
`bench_unet 64 <side> 4 dgpu|vulkan f16` against `bench_ai_ram unet-file=… <side> 4 2048 gpu`
(libtorch CUDA 13, `LD_PRELOAD=libs/libtorch/lib/libtorch_cuda.so` - the example doesn't
link it), 2 runs alternated, round 2:

| Side | burn wgpu f16 | burn Vulkan (SPIR-V) f16 | libtorch CUDA (f32, TF32 convs) |
|---|---|---|---|
| 512² | 0.07 s | 0.07 s | 0.07 s |
| 1024² | 0.38 s | 0.38 s | 0.26 s |
| 2048² | 1.80 s | 1.80 s | 1.06 s |

Peak host RAM: burn +5-150 MB, libtorch +14-76 MB over its 905 MB idle (CUDA context).

Findings on the way (burn 0.22.1):
- Without `default-features` burn also loses `autotune` and `fusion` - turned on again
  (`features = ["flex", "wgpu", "vulkan", "autotune", "fusion"]`, still without `std`).
- CubeCL's fast convolutions (implicit GEMM on tensor cores) take only f16/bf16 here; f32
  runs `conv_direct`: 4.1 s for 512², 60x libtorch. And **with a bias they decline as
  well**, even in f16 (2.2 s): `conv_forward` in `model/unet/network.rs` runs the f16
  convolution without bias and adds the bias after it.
- f16 against f32 on one conv 64->64: max error 7.5e-4 on values up to 1.4 (mean 9e-5) -
  in the range of libtorch's default TF32. Open decision: f16 on the GPU (it is more than
  float rounding, see "results independent of hardware"); `UNet::set_half_convs`, off by
  default.
- burn-flex keeps its speed with the new features and the f16 switch (old and new build
  alternated: 512² 2.52 / 2.54 s, 1024² 10.63 / 10.75 s; a separate bias add on the CPU
  was 5 % slower - only the f16 path adds it separately).
- Vulkan (SPIR-V) brings nothing over wgpu's default here.
- `TensorData` from `into_data()` is lazy on wgpu: the bytes arrive on first access - time
  after `as_slice`/`to_vec`, not after `into_data`.
- The container needs `libvulkan1` + `libegl1` and NVIDIA's EGL vendor file
  `/usr/share/glvnd/egl_vendor.d/10_nvidia.json` (not mounted by the toolkit here),
  else the NVIDIA Vulkan driver fails (`vk_icdGetInstanceProcAddr`).

**Results vs libtorch (2026-11-25)** - romantic-shark's real weights (`crates/ai` example
`unet_reference` exports them to safetensors + libtorch's outputs; `crates/nn` example
`compare_unet`, loader `UNet::with_torch_em_weights`). Inputs normalized
zero-mean-unit-variance; flips = pixels whose foreground (> 0.5) changes:

| Pair | 256² max diff | 1024² max diff | 1024² flips (of 1,048,576) |
|---|---|---|---|
| burn-flex f32 vs package's expected output | 3e-6 | - | - |
| burn GPU f32 vs libtorch CPU | 2e-6 | 6e-5 | 0 |
| burn-flex f32 vs libtorch CPU | 4e-6 | 4e-4 | 3 |
| libtorch GPU (TF32 convs) vs libtorch CPU | 8e-4 | 9e-3 | 166 |
| burn GPU f16 vs libtorch CPU | 1.2e-3 | 6e-3 | 143 |

burn in f32 is as exact as libtorch; burn's f16 GPU path deviates as much as libtorch's
own GPU path does today (libtorch runs f32, but cuDNN convolutions in TF32 by default).

**GPU 2048² profile** (per-layer sync, f16): before, InstanceNorm took 55 % (1.38 s) and
grew 5-9x per 4x pixels - burn's InstanceNorm (GroupNorm reshaped to `[B, C, H*W]`) on a
CubeCL convolution's output (stored channels-last) is 4x slower than on a plain tensor
(37 vs 9 ms, 64x1024²). `instance_norm_channels_last` (reduce over `[B, H*W, C]`, free
views) on the GPU path: 11 ms. Now 512² 0.07 s, 1024² 0.34 s, 2048² 1.53 s (libtorch
0.07 / 0.26 / 1.06), results as before. What remains at 2048²: 3x3 convs 693 ms (45 %),
norm 487 ms (31 %), 1x1 convs 133 ms, skip `cat` 103 ms, bilinear 66 ms.
First GPU run per new shape: autotune takes 1-4 minutes (1024²: 61 s, 2048²: 219 s) -
must be cached (on disk) or limited before it can ship.

**CPU 2048² profile** (burn-flex, 4 threads, 46 s, 128 s CPU = 2.8 of 4 cores busy):
3x3 convs 71 % (33 s), InstanceNorm 12 % (5.5 s), bilinear 6 % (3.0 s), 1x1 convs 4 %,
cat 4 %, pool 2.5 %. `perf`: 57 % in gemm's f32 microkernel (useful), 15 % in burn-flex's
`conv3d_impl` tile closure (im2col + strided output scatter), 5 % gemm packing. The idle
cores: `conv3d_impl` transposes the whole input NCHW → NHWC and reorders the weights
single-threaded on every convolution before the parallel part.

Open: GPU - autotune cache, norm and `cat` further (a conv over the skip and the
upsampled half separately instead of `cat`); CPU - burn-flex's serial transpose (upstream
fix or own conv), bilinear and norm; bf16 instead of f16 (range).

### P1 - Freeze references from libtorch (1-2 days, on the current code)

Before anything is deleted (afterwards libtorch can't be built next to burn 0.22):
1. Golden outputs: per model (real files where we have them: `cpsam_v2.pt`,
   `joyful-pig`, a StarDist `2D_versatile_fluo` export;
   plus the stand-ins) a few fixed inputs (synthetic + 1-2 real tiles) → raw outputs
   (`[C,H,W]` f32) and the final objects (counts, areas) of the core step. Stored under
   `tests/nn_reference/` (large ones outside git, as `cpsam_v2.pt`).
2. Baselines: `bench_ai_ram` per model, `bench_cellpose` (1024²), startup + idle RAM
   (BENCHMARK.md "The real app"), package sizes per platform, clean + incremental build
   time of the workspace. Old build kept in `/workspaces/target-baseline`.

### P2 - Swap the crate, upgrade burn (3-4 days)

1. `crates/nn`: the `Model` trait + `Image`, `runtime` / `pieces` types moved from
   `crates/ai`'s `api.rs` (no `ffi`, no enums, no serde); `set_plugin_path` removed;
   core depends on `evanalyzer_nn`.
2. MLP, KNN, random forest moved from core into `nn::classifier::{mlp, knn, rtree}`
   behind the `Classifier` trait + `TrainedClassifier` enum + `TrainControl`, with the
   smartcore matrix conversion; core hands `Samples` (flat rows) instead of
   `Vec<Vec<f32>>`; burn + smartcore removed
   from core's `Cargo.toml`; MLP weights in the D2/D4 format, old files refused;
   YOLOv5 step removed (D1); then
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

### P7 - Packaging, CI, docs (2-3 days)

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

### P8 - Final measurement and merge (1-2 days)

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
| P7 packaging | 2-3 |
| P8 measure + merge | 1-2 |
| **Total** | **~5-7 weeks** |

## Risks

| Risk | Effect | Mitigation |
|---|---|---|
| burn CPU slower than libtorch/oneDNN | violates "at least as fast" | P0 gate before anything is deleted |
| ViT-L attention slow / RAM-heavy in burn | Cellpose-SAM regresses | P0 item 4; f16 on GPU; flash-attention kernels in CubeCL if available |
| Users' arbitrary TorchScript models stop working | broken pipelines | accepted (D3); clear error naming the supported architectures + importers |
| wgpu version clash with Slint | double compile, bigger binary | check in P0; align versions or pin |
| burn 0.x API churn (0.22 just removed backend generics) | rework per upgrade | keep burn types inside nn; plain-data API |
| Bigger binary / build time (cubecl, wgpu, naga) | slower CI, larger download | measured in P0 + P8; still far below libtorch |
| GPU drivers without Vulkan/DX12 compute | GPU unused | CPU fallback, logged |
