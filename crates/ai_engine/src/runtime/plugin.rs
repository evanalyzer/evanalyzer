//! The libtorch plugin library (this crate built as a cdylib with its
//! `torch` engine - `cargo build-ai-plugin`), loaded on the first
//! model run.
//!
//! Installed next to the program in two builds, each with its own libtorch:
//!
//! ```text
//! evanalyzer(.exe)
//! ai/cpu/   evanalyzer_ai_engine (the plugin) + libtorch (CPU)
//! ai/cuda/  evanalyzer_ai_engine (the plugin) + libtorch (CUDA)  - CUDA packages only
//! ```
//!
//! The CUDA one is tried first where the NVIDIA driver sees a GPU (and the
//! CPU is not asked for), else the CPU one; a package with only the CUDA
//! one uses it on the CPU too (CUDA libtorch runs without a GPU, just
//! larger). A plugin
//! next to the program itself (a `cargo build` of the workspace) or at
//! `EVANALYZER_AI_PLUGIN` (a file path, for development and tests) is
//! taken first. A loaded plugin stays loaded: libtorch cannot be unloaded.

use super::cuda_probe;
use crate::api::{
    API_VERSION, DeviceSelection, EngineConfig, EngineInfo, Image, InferenceEngine, Request,
    Response,
    ffi::{
        ApiVersionFn, Buf, FreeFn, InferFn, InitFn, SYMBOL_API_VERSION, SYMBOL_FREE, SYMBOL_INFER,
        SYMBOL_INIT, decode, encode,
    },
};
use libloading::Library;
use std::path::{Path, PathBuf};

/// Overrides where the plugin is looked for: the path of the library file.
const PLUGIN_ENV: &str = "EVANALYZER_AI_PLUGIN";

/// A loaded plugin, set up.
struct PluginEngine {
    info: EngineInfo,
    infer: InferFn,
    free: FreeFn,
    /// Keeps the functions above valid; never unloaded (see the module).
    _library: Library,
}

/// Loads the plugin that fits this computer and sets it up with `config`.
pub(crate) fn load(config: &EngineConfig) -> Result<Box<dyn InferenceEngine>, String> {
    let exe_dir = std::env::current_exe()
        .ok()
        .and_then(|exe| exe.parent().map(Path::to_path_buf));
    let wants_gpu = config.devices != DeviceSelection::Cpu && cuda_probe::gpu_count() > 0;
    let candidates = candidates(
        std::env::var_os(PLUGIN_ENV).map(PathBuf::from),
        exe_dir.as_deref(),
        wants_gpu,
    );
    let mut errors = Vec::new();
    for path in candidates.iter().filter(|p| p.is_file()) {
        log::info!("Loading the AI plugin {}", path.display());
        match PluginEngine::load(path, config) {
            Ok(engine) => return Ok(Box::new(engine)),
            Err(e) => {
                log::warn!("AI plugin {}: {e}", path.display());
                errors.push(format!("{}: {e}", path.display()));
            }
        }
    }
    if errors.is_empty() {
        Err(format!(
            "AI models need the AI plugin (`{}`), which is not installed. Looked in: {}",
            library_name().display(),
            candidates
                .iter()
                .map(|p| p.display().to_string())
                .collect::<Vec<_>>()
                .join(", ")
        ))
    } else {
        Err(format!(
            "The AI plugin could not be started: {}",
            errors.join("; ")
        ))
    }
}

/// Whether the CUDA build of the plugin is installed.
pub(crate) fn cuda_plugin_installed() -> bool {
    let exe_dir = std::env::current_exe()
        .ok()
        .and_then(|exe| exe.parent().map(Path::to_path_buf));
    exe_dir.is_some_and(|dir| dir.join("ai").join("cuda").join(library_name()).is_file())
}

/// `libevanalyzer_ai_engine.so`, `evanalyzer_ai_engine.dll`, `….dylib`.
fn library_name() -> PathBuf {
    PathBuf::from(libloading::library_filename("evanalyzer_ai_engine"))
}

/// Where the plugin is looked for, in this order (files that don't exist
/// are skipped by the caller).
fn candidates(env: Option<PathBuf>, exe_dir: Option<&Path>, wants_gpu: bool) -> Vec<PathBuf> {
    let mut paths = Vec::new();
    if let Some(path) = env {
        paths.push(path);
    }
    if let Some(dir) = exe_dir {
        let name = library_name();
        paths.push(dir.join(&name));
        let cuda = dir.join("ai").join("cuda").join(&name);
        if wants_gpu {
            paths.push(cuda.clone());
        }
        paths.push(dir.join("ai").join("cpu").join(&name));
        if !wants_gpu {
            paths.push(cuda);
        }
    }
    paths
}

impl PluginEngine {
    fn load(path: &Path, config: &EngineConfig) -> Result<Self, String> {
        let path = std::path::absolute(path).map_err(|e| e.to_string())?;
        // SAFETY: loading runs the library's initializers - libtorch's,
        // which are what this is for. Its functions are taken with the
        // types `crate::api::ffi` defines, after the API version
        // check below.
        unsafe {
            let library = open(&path)?;
            let symbol =
                |name: &[u8]| String::from_utf8_lossy(&name[..name.len() - 1]).into_owned();
            let api_version = *library
                .get::<ApiVersionFn>(SYMBOL_API_VERSION)
                .map_err(|e| format!("not an AI plugin ({}): {e}", symbol(SYMBOL_API_VERSION)))?;
            let version = api_version();
            if version != API_VERSION {
                return Err(format!(
                    "built for interface version {version}, this program needs {API_VERSION} - \
                     install the plugin that came with this version"
                ));
            }
            let init = *library
                .get::<InitFn>(SYMBOL_INIT)
                .map_err(|e| format!("{}: {e}", symbol(SYMBOL_INIT)))?;
            let infer = *library
                .get::<InferFn>(SYMBOL_INFER)
                .map_err(|e| format!("{}: {e}", symbol(SYMBOL_INFER)))?;
            let free = *library
                .get::<FreeFn>(SYMBOL_FREE)
                .map_err(|e| format!("{}: {e}", symbol(SYMBOL_FREE)))?;

            let config = encode(config)?;
            let (mut out, mut err) = (Buf::EMPTY, Buf::EMPTY);
            let code = init(
                config.as_ptr(),
                config.len(),
                forward_log,
                log::max_level() as u8,
                &mut out,
                &mut err,
            );
            let info = answer(free, code, out, err)?;
            Ok(Self {
                info,
                infer,
                free,
                _library: library,
            })
        }
    }
}

impl InferenceEngine for PluginEngine {
    fn info(&self) -> EngineInfo {
        self.info.clone()
    }

    fn infer(&self, request: &Request, image: Image<'_>) -> Result<Response, String> {
        let request = encode(request)?;
        let (mut out, mut err) = (Buf::EMPTY, Buf::EMPTY);
        // SAFETY: the pointers and lengths describe live buffers; the
        // answer is taken back through the plugin's own `free`.
        unsafe {
            let code = (self.infer)(
                request.as_ptr(),
                request.len(),
                image.width,
                image.height,
                image.channels,
                image.data.as_ptr(),
                image.data.len(),
                &mut out,
                &mut err,
            );
            answer(self.free, code, out, err)
        }
    }
}

/// Reads the plugin's answer (decoded straight from its buffer, no copy
/// first) and gives its buffers back: the value on success (code 0), else
/// the error message.
///
/// # Safety
/// `out` and `err` must come from the plugin whose `free` this is.
unsafe fn answer<T: serde::de::DeserializeOwned>(
    free: FreeFn,
    code: i32,
    out: Buf,
    err: Buf,
) -> Result<T, String> {
    let result = unsafe {
        if code == 0 {
            decode(out.as_slice())
        } else {
            Err(String::from_utf8_lossy(err.as_slice()).into_owned())
        }
    };
    unsafe {
        free(out);
        free(err);
    }
    result
}

/// Opens the library so that the libraries next to it (its libtorch) are
/// found before any others.
unsafe fn open(path: &Path) -> Result<Library, String> {
    #[cfg(windows)]
    let library = unsafe {
        use libloading::os::windows::{
            LOAD_LIBRARY_SEARCH_DEFAULT_DIRS, LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR,
        };
        libloading::os::windows::Library::load_with_flags(
            path,
            LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS,
        )
        .map(Library::from)
    };
    // Linux and macOS: the plugin's rpath ($ORIGIN, @loader_path) points
    // at its own folder.
    #[cfg(not(windows))]
    let library = unsafe { Library::new(path) };
    library.map_err(|e| format!("could not be loaded: {e}"))
}

/// The plugin's log lines, into this program's log.
extern "C" fn forward_log(
    level: u8,
    target: *const u8,
    target_len: usize,
    message: *const u8,
    message_len: usize,
) {
    let text = |ptr: *const u8, len: usize| {
        if ptr.is_null() {
            String::new()
        } else {
            // SAFETY: the plugin passes `len` bytes at `ptr`, valid for the call.
            String::from_utf8_lossy(unsafe { std::slice::from_raw_parts(ptr, len) }).into_owned()
        }
    };
    let level = match level {
        1 => log::Level::Error,
        2 => log::Level::Warn,
        3 => log::Level::Info,
        4 => log::Level::Debug,
        _ => log::Level::Trace,
    };
    let target = text(target, target_len);
    log::log!(target: &target, level, "{}", text(message, message_len));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_plugin_is_looked_for_in_this_order() {
        let exe = Path::new("/opt/eva");
        let name = library_name();
        assert_eq!(
            candidates(Some("/dev/plugin.so".into()), Some(exe), true),
            vec![
                PathBuf::from("/dev/plugin.so"),
                exe.join(&name),
                exe.join("ai/cuda").join(&name),
                exe.join("ai/cpu").join(&name),
            ]
        );
        assert_eq!(
            candidates(None, Some(exe), false),
            vec![
                exe.join(&name),
                exe.join("ai/cpu").join(&name),
                exe.join("ai/cuda").join(&name),
            ],
            "without a GPU (or with the CPU asked for) the CUDA plugin only when \
             there is no CPU one"
        );
    }

    #[test]
    fn a_missing_plugin_is_explained() {
        let dir = tempfile::tempdir().unwrap();
        let err = PluginEngine::load(&dir.path().join("nope.so"), &config())
            .err()
            .unwrap();
        assert!(err.contains("could not be loaded"), "{err}");
    }

    fn config() -> EngineConfig {
        EngineConfig {
            devices: DeviceSelection::Cpu,
            gpu_slots: 1,
            cores: 2,
        }
    }

    /// The real plugin, built by cargo: loaded, set up, a model run through
    /// the C interface.
    #[test]
    fn the_built_plugin_loads_and_runs_a_model() {
        let plugin = build_plugin();
        let engine = PluginEngine::load(&plugin, &config()).unwrap();
        assert!(
            engine.info().name.starts_with("libtorch"),
            "{:?}",
            engine.info()
        );
        assert_eq!(engine.info().devices[0].name, "CPU");

        // The plugin's own error for a model that isn't there comes back
        // as a message.
        let request = Request {
            model: "/no/such/model.pt".into(),
            task: crate::api::Task::StarDist,
        };
        let pixels = [0.0f32; 4];
        let image = Image {
            width: 2,
            height: 2,
            channels: 1,
            data: &pixels,
        };
        let err = engine.infer(&request, image).unwrap_err();
        assert!(err.contains("Failed to load StarDist model"), "{err}");
    }

    /// `cargo build-ai-plugin` (a no-op when it is current), in the
    /// profile of this test; the library's path.
    fn build_plugin() -> PathBuf {
        let release = !cfg!(debug_assertions);
        let workspace = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let cargo = std::env::var_os("CARGO").unwrap_or("cargo".into());
        let status = std::process::Command::new(cargo)
            .current_dir(&workspace)
            .args([
                "rustc",
                "-p",
                "evanalyzer_ai_engine",
                "--features",
                "torch",
                "--crate-type",
                "cdylib",
            ])
            .args(release.then_some("--release"))
            .status()
            .expect("cargo runs");
        assert!(status.success(), "building the plugin failed");
        let target = std::env::var_os("CARGO_TARGET_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|| workspace.join("target"));
        target
            .join(if release { "release" } else { "debug" })
            .join(library_name())
    }

    /// The plugin boundary's cost: one Cellpose request on libtorch linked
    /// in and through the plugin, same process, same models. Prints the
    /// median of 5 runs per image size; ~20 s. A real model at
    /// `BENCH_CELLPOSE_MODEL` (e.g. `tests/cpsam_v2.pt`, 3 input channels)
    /// runs once at 512 x 512 - about a minute on 22 CPU cores.
    /// Run with:
    ///   cargo test --release -p evanalyzer_ai_engine --features torch-plugin,test-support \
    ///     --lib bench_plugin_against_in_process -- --ignored --nocapture
    #[cfg(feature = "test-support")]
    #[test]
    #[ignore]
    fn bench_plugin_against_in_process() {
        use crate::torch::{TorchEngine, test_support::bench_cellpose_model};
        use std::time::Instant;

        let config = EngineConfig {
            devices: DeviceSelection::Cpu,
            gpu_slots: 1,
            cores: std::thread::available_parallelism().map_or(1, |n| n.get()),
        };
        let in_process = TorchEngine::new(&config);
        let plugin = PluginEngine::load(&build_plugin(), &config).unwrap();
        let median = |engine: &dyn InferenceEngine, request: &Request, image: Image<'_>| {
            engine.infer(request, image).unwrap(); // model loaded, warmed up
            let runs = if request.model.ends_with("model.pt") {
                5
            } else {
                1
            };
            let mut times: Vec<f64> = (0..runs)
                .map(|_| {
                    let start = Instant::now();
                    engine.infer(request, image).unwrap();
                    start.elapsed().as_secs_f64() * 1000.0
                })
                .collect();
            times.sort_by(f64::total_cmp);
            times[times.len() / 2]
        };
        println!("model  size       in-process   plugin   difference");
        let (_light_dir, light) = bench_cellpose_model(0);
        let (_heavy_dir, heavy) = bench_cellpose_model(6);
        let mut models = vec![("light", light, 2, vec![512, 1024, 2048])];
        models.push(("heavy", heavy, 2, vec![512, 1024]));
        if let Some(real) = std::env::var_os("BENCH_CELLPOSE_MODEL") {
            models.push(("real", real.into(), 3, vec![512]));
        }
        for (name, model, input_channels, sizes) in models {
            for size in sizes {
                let pixels: Vec<f32> = (0..size * size).map(|i| (i % 97) as f32 / 97.0).collect();
                let image = Image {
                    width: size,
                    height: size,
                    channels: 1,
                    data: &pixels,
                };
                let request = Request {
                    model: model.clone(),
                    task: crate::api::Task::Cellpose {
                        net_width: size,
                        net_height: size,
                        input_channels,
                        replicate_gray_channel: true,
                    },
                };
                let a = median(&in_process, &request, image);
                let b = median(&plugin, &request, image);
                println!(
                    "{name:<6} {size:>4}x{size:<4}  {a:>8.1} ms {b:>8.1} ms  {:>+7.1} ms ({:+.1} %)",
                    b - a,
                    (b - a) / a * 100.0
                );
            }
        }
    }
}
