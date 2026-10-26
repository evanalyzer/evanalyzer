# EVAnalyzer
***Enhanced Visual Analyzer***

<div align="center">

[![Documentation](https://img.shields.io/badge/docs-evanalyzer.org-009933)](https://evanalyzer.org)
[![Image.sc forum](https://img.shields.io/badge/dynamic/json.svg?label=forum&url=https%3A%2F%2Fforum.image.sc%2Ftags%2Fevanalyzer.json&query=%24.topic_list.tags.0.topic_count&colorB=brightgreen&suffix=%20topics&logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAA4AAAAOCAYAAAAfSC3RAAABPklEQVR42m3SyyqFURTA8Y2BER0TDyExZ+aSPIKUlPIITFzKeQWXwhBlQrmFgUzMMFLKZeguBu5y+//17dP3nc5vuPdee6299gohUYYaDGOyyACq4JmQVoFujOMR77hNfOAGM+hBOQqB9TjHD36xhAa04RCuuXeKOvwHVWIKL9jCK2bRiV284QgL8MwEjAneeo9VNOEaBhzALGtoRy02cIcWhE34jj5YxgW+E5Z4iTPkMYpPLCNY3hdOYEfNbKYdmNngZ1jyEzw7h7AIb3fRTQ95OAZ6yQpGYHMMtOTgouktYwxuXsHgWLLl+4x++Kx1FJrjLTagA77bTPvYgw1rRqY56e+w7GNYsqX6JfPwi7aR+Y5SA+BXtKIRfkfJAYgj14tpOF6+I46c4/cAM3UhM3JxyKsxiOIhH0IO6SH/A1Kb1WBeUjbkAAAAAElFTkSuQmCC&style=flat&color=009933)](https://forum.image.sc/tag/evanalyzer)
[![GitHub Release](https://img.shields.io/github/v/release/evanalyzer/evanalyzer?include_prereleases)](https://github.com/evanalyzer/evanalyzer/releases/latest)
[![Downloads](https://img.shields.io/github/downloads/evanalyzer/evanalyzer/total)](https://github.com/evanalyzer/evanalyzer/releases)
[![CI](https://github.com/evanalyzer/evanalyzer/actions/workflows/ci.yml/badge.svg)](https://github.com/evanalyzer/evanalyzer/actions/workflows/ci.yml)
[![Coverage](https://img.shields.io/endpoint?url=https%3A%2F%2Fevanalyzer.github.io%2Fevanalyzer%2Fcoverage-badge.json)](https://evanalyzer.github.io/evanalyzer/)
[![License: PolyForm Noncommercial 1.0.0](https://img.shields.io/badge/license-PolyForm_Noncommercial-blue)](#license)
[![Platform](https://img.shields.io/badge/platform-Linux%20%7C%20Windows%20%7C%20macOS-lightgrey)](https://github.com/evanalyzer/evanalyzer/releases/latest)

<a href="https://slint.dev"><img src="docs/MadeWithSlint-logo-light.svg#gh-light-mode-only" height="40" alt="Made with Slint"></a><a href="https://slint.dev"><img src="docs/MadeWithSlint-logo-dark.svg#gh-dark-mode-only" height="40" alt="Made with Slint"></a>

**Find, classify and measure objects in microscopy images - reproducibly, on ordinary computers.**

![EVAnalyzer](docs/screenshot.png)

</div>

EVAnalyzer is a desktop application for bioimage analysis. You build a pipeline
of steps, tune it live on the image, and run it over thousands of images or
whole slides. Results go into one database you can explore, chart and export.

- **Reads microscopy and whole-slide formats directly** - CZI, ND2, LIF, VSI,
  OME-TIFF, SVS, NDPI, MRXS and more, with pyramids, z-stacks, time series and
  multi-well plates.
- **Pipelines you tune on the image** - filters, thresholds, watershed, AI
  segmentation (Cellpose, StarDist, U-Net, YOLOv5), trained pixel and object
  classifiers; every change is previewed right away.
- **Measurements that match the reference tools** - intensity, shape, texture,
  colocalization, neighbours and parent/child relations, checked against
  CellProfiler and ImageJ.
- **Results in one place** - tables, plate heat maps and charts; export to CSV
  and Excel.
- **Runs where your data is** - on your computer, from the command line, or on
  a shared workstation or server that others log in to.

## Download

Get the latest version for Linux, Windows or macOS from the
[releases page](https://github.com/evanalyzer/evanalyzer/releases/latest).
Unpack the archive and start `evanalyzer` - everything it needs is included.
GPU (CUDA) builds for AI models are available for Linux and Windows.

## Where EVAnalyzer stores its data

**Your settings and templates** are in one folder in your home folder:

| System | Folder |
|---|---|
| Linux | `~/.evanalyzer/` |
| macOS | `~/evanalyzer/` |
| Windows | `C:\Users\<name>\evanalyzer\` |

```
evanalyzer/
├── settings.json      app settings, incl. the servers you trust
├── templates/         your templates and modules
└── logs/crash.log     details of crashes, for bug reports
```

Copy this folder to move your settings to another computer. When you work
on a server, the same folder is in your home folder on the server.

**A server** keeps everything it creates in its data folder - change it with
`--data-dir` or `data_dir` in the config file:

| Server runs | Data folder |
|---|---|
| as a system service (root, Linux) | `/var/lib/evanalyzer/` |
| as a normal user, Linux | `~/.evanalyzer/server/` |
| as a normal user, macOS / Windows | `~/evanalyzer/server/` |

```
<data folder>/
├── tls/               self-signed certificate and key - back it up, or
│                      clients have to trust a new fingerprint
├── sessions.json      the running workers
└── single-user/       home of the single user (default user setup)
```

Files you write yourself as the server admin are not in the data folder. The
server reads them from the path you give it, so you can keep them anywhere
(`/etc/evanalyzer/` is the usual place):

| File | Path given by |
|---|---|
| server config | `evanalyzer server --config /etc/evanalyzer/server.toml` |
| users file (optional) | `[users.file] path = "..."` in the server config |
| your own TLS certificate and key (optional) | `[tls] cert = "..."` and `key = "..."` in the server config |

See [docs/server.md](docs/server.md) for all settings.

## Documentation and help

- **User guide:** [evanalyzer.org](https://evanalyzer.org) - getting started,
  every pipeline step, the command line and running a server.
- **Questions:** ask on the [image.sc forum](https://forum.image.sc/tag/evanalyzer)
  with the tag `evanalyzer`.
- **Bugs and feature requests:** open an
  [issue](https://github.com/evanalyzer/evanalyzer/issues).

---

## Development Setup

### Prerequisites

- **Rust** 1.85 or later (2024 edition): `rustup update`
- **Native libraries** (libtorch for the AI plugin, DuckDB on Windows/macOS),
  downloaded once into `libs/`:

  ```sh
  libs/download.sh linux-cpu      # or linux-cuda | win-cpu | win-cuda | win-arm64-cpu | mac-cpu
  ```

  Then unpack libtorch: `unzip libs/libtorch-*.zip -d libs` (gives
  `libs/libtorch`). `linux-cuda` also puts NVIDIA's CUDA libraries into
  `libs/nvidia-cu13/lib` (the Linux libtorch zip has none) - see
  [AI models: the libtorch plugin](#ai-models-the-libtorch-plugin).

- **Linux system libraries** for the GUI:

  ```sh
  apt-get install libinput10 libxkbcommon0 libxkbcommon-x11-0 libfontconfig1 libgbm1
  ```

### Workspace

| Crate | What it does |
|---|---|
| `evanalyzer_core` | Image reading ([scopeio](https://github.com/evanalyzer/scopeio)), algorithms, objects, pipeline execution |
| `evanalyzer_cfg` | Project files and settings, generated pipeline command types |
| `evanalyzer_app` | Shared project state, local and remote backends |
| `evanalyzer_gui` | Slint desktop GUI |
| `evanalyzer_cli` | Headless command line (`evanalyzer cli ...`) |
| `evanalyzer_server` | Multi-user server (`evanalyzer server`, see [docs/server.md](docs/server.md)) |
| `evanalyzer_ai` | AI model interface and the libtorch plugin |
| `evanalyzer_bin` | The `evanalyzer` binary: GUI, CLI or server |

### Building

The app and the AI plugin are built separately - only the plugin links
libtorch:

```sh
cargo build                          # the app
cargo build-ai-plugin                # the libtorch plugin (alias in .cargo/config.toml)
cargo build --no-default-features    # the app without AI model support
cargo run -p evanalyzer_bin -- --project my.evaproj   # start the GUI with a project
```

Release builds per platform (aliases in `.cargo/config.toml`):

| Target | Command | Notes |
|---|---|---|
| Linux x86-64 | `cargo build-linux` | plugin: `cargo build-ai-plugin --release --target x86_64-unknown-linux-gnu` |
| Linux ARM64 | `cargo build-linux-arm` | no AI plugin (no libtorch for Linux ARM64); cross-compiling needs an arm64 sysroot with fontconfig - release builds run on an ARM runner |
| Windows x86-64 / ARM64 | `cargo build-win` / `cargo build-win-arm` | cross-compiled with [`cargo-xwin`](https://github.com/rust-cross/cargo-xwin) |
| macOS Apple Silicon | `cargo build-mac` | native; set `DUCKDB_LIB_DIR`/`DUCKDB_INCLUDE_DIR` (see `libs/download.sh mac-cpu`) |

### AI models: the libtorch plugin

The AI model steps (Cellpose, StarDist, U-Net, YOLOv5) run on libtorch, but
the app never links it: libtorch lives in a separate library, the **AI
plugin** - the `evanalyzer_ai` crate built as a cdylib with its `torch`
feature (`cargo build-ai-plugin`). The app loads it on the first model run;
without it everything else works and model steps report the plugin as not
installed.

| | Version | Where it is set |
|---|---|---|
| libtorch | 2.13.0 | `LIBTORCH_VERSION` in `libs/download.sh` |
| tch (Rust bindings) | 0.26 - built for exactly libtorch 2.13 | `tch` in `Cargo.toml` |
| CUDA | 13.0 (`cu130`): NVIDIA GPUs with compute capability 7.5+ (RTX 20xx / GTX 16xx and newer), driver 580+ | `CUDA_VARIANT` in `libs/download.sh`, `MIN_COMPUTE_CAPABILITY` in `crates/ai/src/runtime/cuda_probe.rs` |
| NVIDIA libraries (Linux) | the versions torch 2.13 pins on PyPI | `NVIDIA_WHEELS` in `libs/download.sh` |

The plugin is compiled against `libs/libtorch` (`LIBTORCH` in
`.cargo/config.toml`), and must run with exactly that libtorch.

**Where the app looks for it** (first found wins):

1. `EVANALYZER_AI_PLUGIN=/path/to/libevanalyzer_ai.so`
2. next to the `evanalyzer` binary
3. `ai/cuda/` next to the binary - first when the NVIDIA driver sees a GPU new
   enough for this build
4. `ai/cpu/` next to the binary
5. `ai/cuda/` otherwise (CUDA libtorch runs on the CPU too)

The plugin's libtorch must be in the plugin's own folder (its RUNPATH is
`$ORIGIN`; on Windows the app searches the plugin's folder); the CUDA part
(`libtorch_cuda.so` / `torch_cuda.dll`) is loaded from there by the app. GPUs
older than the build supports are skipped - models run on the CPU.

**Development (CPU):**

```sh
libs/download.sh linux-cpu && unzip -q libs/libtorch-*.zip -d libs
cargo build-ai-plugin                # target/debug/libevanalyzer_ai.so, next to the binary
LD_LIBRARY_PATH=$PWD/libs/libtorch/lib cargo run -p evanalyzer_bin
```

**A runnable build with GPU support** - the package layout, in the target
folder:

```sh
libs/download.sh linux-cuda && unzip -q libs/libtorch-*.zip -d libs
cargo build-linux
cargo build-ai-plugin --release --target x86_64-unknown-linux-gnu

R=target/x86_64-unknown-linux-gnu/release
mkdir -p $R/ai/cuda
mv $R/libevanalyzer_ai.so $R/ai/cuda/          # not next to the binary: that copy would win
cp libs/libtorch/lib/*.so* libs/nvidia-cu13/lib/*.so* $R/ai/cuda/
ldd $R/ai/cuda/libtorch_cuda.so | grep "not found"   # only libcuda/libnvidia-ml may show: the driver's
$R/evanalyzer
```

After rebuilding the plugin, move the new `libevanalyzer_ai.so` into
`ai/cuda` again.

**Shipping.** The release pipeline (`.github/workflows/release.yml`)
builds and packages every platform; a tag (`v0.2.0` or `0.2.0`) publishes
a GitHub release, each platform's files added as soon as its build is done.
Pushing to the `pipeline_start` branch runs the same builds without a
release. Package layout:

```text
evanalyzer(.exe)            the app
duckdb.dll / libduckdb.dylib  Windows / macOS only (Linux has DuckDB built in)
templates/                  project and module templates
ai/cpu/  or  ai/cuda/       the plugin + its libtorch (+ NVIDIA's libraries in ai/cuda)
```

| Package | AI | Built from |
|---|---|---|
| Linux / Windows x86-64 | CPU (`ai/cpu`) | `download.sh linux-cpu` / `win-cpu` |
| Linux / Windows x86-64 CUDA | GPU (`ai/cuda`), one `.7z` < 2 GiB | `download.sh linux-cuda` / `win-cuda` |
| Windows ARM64 | CPU | `download.sh win-arm64-cpu` |
| macOS Apple Silicon | CPU, dylibs relinked to `@loader_path` and ad-hoc signed | `download.sh mac-cpu` |
| Linux ARM64 | none - PyTorch publishes no libtorch for it | - |

Files in `.github/NOT_SHIP_LIST.txt` (Python bindings, test fixtures,
profiler and multi-GPU libraries) are left out of the CUDA packages, and the
Linux CUDA job fails if `ldd` finds a library libtorch needs missing, or if
the archive reaches GitHub's 2 GiB limit.

**Updating libtorch:**

1. Pick a version a `tch` release supports (its CHANGELOG: "PyTorch v2.x
   support") and for which pytorch.org has every zip we use (CPU, `cu130`
   Linux/Windows, Windows ARM64, macOS).
2. Set `tch` in `Cargo.toml` and `LIBTORCH_VERSION` in `libs/download.sh`.
3. Update `NVIDIA_WHEELS` in `libs/download.sh` to the versions that torch
   version pins on PyPI (`requires_dist` of `torch` and of its
   `cuda-toolkit`), and check with `readelf -d libs/libtorch/lib/*.so` that
   every NVIDIA library libtorch links is in the list.
4. Delete `libs/libtorch` and any old libtorch copies (e.g. in
   `target/debug/deps` - `cargo test` puts that folder first on the library
   path), download again, run the AI tests (below) and one model on a GPU.

### Testing

```sh
cargo test                           # everything except the AI model steps
LD_LIBRARY_PATH=$PWD/libs/libtorch/lib \
  cargo test --features evanalyzer_core/libtorch,evanalyzer_core/libtorch-in-process
```

With the CUDA libtorch add `libs/nvidia-cu13/lib` to `LD_LIBRARY_PATH`.

`libtorch-in-process` links libtorch into the test process for the AI model
steps' tests; `libtorch` adds the plugin loader, whose test builds and loads
the real plugin. Benchmarks and how to compare runs are in
[BENCHMARK.md](BENCHMARK.md).

### Adding a pipeline command

Pipeline commands (`Blur`, `Threshold`, `Cellpose`, ...) are declared once, in
`evanalyzer_core`, and generated everywhere else:

1. Write a plain struct in `crates/core/src/algos/<category>/<name>.rs`:

   ```rust
   #[derive(CommandsMeta)]
   #[cmdsmeta(category = "segment", display_name = "My Command")]
   pub struct MyCommand {
       #[cmdsmeta(min = 1, max = 100, step = 1, default = 10)]
       pub some_number: usize,

       /// Path to a trained model.
       #[cmdsmeta(file_extensions = "pt,pth")]
       pub model_path: PathBuf,
   }

   impl ImageAlgorithm for MyCommand {
       fn execute(&self, ctx: &mut PipelineContext, cache: &mut PipelineCache) -> Result<(), InternalErrors> {
           /* ... */
       }
       fn name(&self) -> &'static str { "MyCommand" }
   }
   ```

2. Build. `crates/cfg/build/pipeline_commands_generator.rs` finds every
   `#[derive(CommandsMeta)]` struct and regenerates
   `crates/cfg/src/settings/pipeline_command.rs`,
   `crates/cfg/src/settings/pipeline_command_settings.rs` and
   `crates/core/src/job/algos_from_config.rs` (all `// @generated`): the
   settings type, the `PipelineCommand` variant, the editor parameters and the
   conversion to `MyCommand`. Nothing to wire up by hand.

Field types bring their own UI and behaviour:

- `PathBuf` - a file picker, stored relative to the project so projects can be
  moved. `file_extensions` only narrows the picker's filter.
- `ObjectClass` / `SegmentationClass` (also in `Vec`/`Option`) - a class
  picker, and listed in `PipelineCommand::object_classes()`. A trained model's
  own class ids use `ModelClass` / `ModelSegmentationClass` instead.
- `#[cmdsmeta(section = "Detection")]` groups settings under a header on the
  step card and on a page of their own in the step's settings dialog.

All field keys (`min`, `max`, `step`, `default`, `unit`, `regex`, `optional`,
`visibility`, `file_extensions`, `section`) are documented on
`commands_meta_derive` in `crates/core/macros/src/lib.rs`.

`PipelineCommand` variants are numbered alphabetically, so a new command
shifts the numbers of the ones after it. Run
`cargo test -p evanalyzer_cfg --test pipeline_command_coverage` and update
any number the test reports as shifted.

### File headers

Every source file starts with a license header. For Rust, the VS Code snippet
`comm-file-header-open` (`.vscode/comments.code-snippets`) writes it:

```rust
//! # my_file
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-10-25
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.
```

Files that `crates/cfg/build.rs` pulls in with `include!()` use `//` instead
of `//!`. Slint and Python files use a two-line comment:
`Copyright <year> Joachim Danmayr.` / `Licensed under the PolyForm
Noncommercial License 1.0.0.`

### Tools

```sh
rustup component add rustfmt
cargo install slint-lsp       # language server for .slint files
cargo install slint-viewer    # live preview of .slint files
```

To show the GUI from inside a dev container, allow X11 forwarding on the host
first: `xhost +local:docker`.

### Coverage, audit, SBOM and licenses

| What | Locally | In CI |
|---|---|---|
| Coverage | `cargo llvm-cov --html` (`cargo install cargo-llvm-cov`, `rustup component add llvm-tools-preview`) | `ci.yml` on `main` publishes the HTML report and badge to [GitHub Pages](https://evanalyzer.github.io/evanalyzer/); `release.yml` attaches `evanalyzer-<version>-coverage.lcov`, `-coverage-summary.txt` and `-coverage-html.zip` to every release |
| Dependency audit | `cargo audit` (`cargo install cargo-audit`) | `ci.yml` shows the result in the job summary; `release.yml` attaches `evanalyzer-<version>-audit.txt` and `.json` to every release; informational, not a gate |
| SBOM | `cargo cyclonedx --format json` (`cargo install cargo-cyclonedx`) | `release.yml` attaches `evanalyzer-<version>-sbom.cdx.json` to every release |
| Third-party licenses | see below | `ci.yml` (`licenses`) regenerates the list and fails if it differs from the committed one or a license isn't accepted; the test run fails if `THIRD-PARTY-NOTICES.txt` is out of date; `release.yml` puts `LICENSE` and `THIRD-PARTY-NOTICES.txt` into every archive |

After changing dependencies (adding or updating a crate, `cargo update`),
regenerate the license list and commit both files:

```sh
# 1. The About dialog's license list (all crates, incl. the AI plugin's)
cargo about generate --locked --workspace --features evanalyzer_ai/torch \
  docs/about.hbs -o crates/gui/src/generated/third_party_licenses.json

# 2. THIRD-PARTY-NOTICES.txt, built from that list and
#    crates/gui/src/third_party_notices.json
UPDATE_NOTICES=1 cargo test -p evanalyzer_gui notices_file
```

cargo-about must be exactly 0.9.3 (`about.toml` works around a bug of that
version, and another version may format the output differently):
`cargo install cargo-about --locked --version 0.9.3 --features cli` - the
devcontainer image has it. What isn't a crate (code ported from other
projects, fonts, icons, native libraries such as libtorch, DuckDB and NVIDIA's
CUDA libraries) is kept by hand in `crates/gui/src/third_party_notices.json`;
after editing it, only step 2 is needed.

`about.toml` lists the accepted licenses. Copyleft licenses are rejected unless
reviewed and listed per crate there.

### Licensing rules for contributors

EVAnalyzer is a clean-room implementation: no GPL/LGPL/AGPL code, and new
dependencies must be permissive (the only MPL-2.0 crates, `smartstring` via
`rhai` and `option-ext` via `dirs`, are reviewed per-crate exceptions).

- **Algorithms come from papers.** Implement a method from its publication and
  cite it (`CitationMetadata`); other tools may only serve as black boxes for
  reference values or as documentation of a file/model format. Never read or
  translate the code of GPL/AGPL projects (ImageJ's `AutoThresholder` is
  Landini's GPL plugin, not public-domain ImageJ 1 code; Ultralytics YOLO is
  AGPL; ilastik, QuPath, TrackMate and Fiji plugins are GPL).
- **Ports from permissive code** (BSD, MIT, Apache, public domain) are allowed:
  name the source in the file header and add its full license text to
  `crates/gui/src/third_party_notices.json` (then regenerate
  `THIRD-PARTY-NOTICES.txt`, see above).
- **Test data** needs a license that allows redistribution and commercial use,
  with its attribution in `crates/core/tests/README.md`. Not usable: BBBC035
  and other Cell Tracking Challenge data (copying forbidden, commercial use
  needs permission).
- **On a libtorch/CUDA update**, check the native libraries again:
  - libgomp is GPL-3.0 with the GCC Runtime Library Exception; its notice holds
    a written source offer. Look up the GCC version it was built with
    (`strings libgomp.so.1 | grep annobin` - 2.13.0: GCC 8.5.0, Red Hat
    8.5.0-28), update the notice and keep a copy of that source package.
  - Every NVIDIA library shipped must be redistributable: listed in
    Attachment A of the [CUDA EULA](https://docs.nvidia.com/cuda/eula/index.html)
    (e.g. `libcufile` as "NVIDIA CUDA File IO Libraries"), or allowed by its own
    license (NVSHMEM's SLA: "any portion of the SDK"). Update their license
    texts in `third_party_notices.json` from the wheels'
    `*.dist-info/licenses/License.txt`.

---

## Contributing

Contributions are welcome. Please open an issue first for larger changes, so
we can agree on the direction. Then fork the repository, work on a branch and
open a pull request.

Before your first pull request can be merged, you need to accept the
[Contributor License Agreement](CLA.md). It transfers the rights in your
contribution to the maintainer, so EVAnalyzer can stay available under both
non-commercial and commercial licenses.

## License

EVAnalyzer is licensed under the
[PolyForm Noncommercial License 1.0.0](LICENSE): free for personal, academic,
research and other non-commercial use. Commercial use - in a company, in a
commercial product or service - needs a separate license; please
[open an issue](https://github.com/evanalyzer/evanalyzer/issues) or contact the
maintainer.

Third-party components keep their own licenses (listed in the About dialog
and in [THIRD-PARTY-NOTICES.txt](THIRD-PARTY-NOTICES.txt), which every release
archive carries next to `LICENSE`).
The image reader [scopeio](https://github.com/evanalyzer/scopeio) is AGPL-3.0
with an exception that lets it be distributed with EVAnalyzer under
EVAnalyzer's license.

## Acknowledgements

- [Slint](https://slint.dev/) - cross-platform UI toolkit
- [kornia-rs](https://github.com/kornia/kornia-rs) - computer vision primitives in Rust
- [DuckDB](https://duckdb.org/) - in-process analytical database
- [ImageJ](https://imagej.net/) and [CellProfiler](https://cellprofiler.org/) - algorithms ported to EVAnalyzer and reference values for its tests
