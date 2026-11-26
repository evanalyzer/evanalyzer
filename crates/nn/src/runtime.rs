//! # runtime
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-11-25
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.
//!
//! Backend and devices, chosen once per process on the first model run -
//! nothing is set up before (idle RAM, startup). Ported from the libtorch
//! engine's scheduler (removed `crates/ai/src/torch/scheduler.rs`), the same rules:
//!
//! - every discrete GPU wgpu finds (or the ones asked for), else the CPU
//!   (burn-flex); integrated GPUs and software adapters are not used;
//! - one copy of each model file per device, shared by that device's slots,
//!   kept until [`Engine::release_models`] or until the file changes (its
//!   mtime);
//! - a device runs at most its slots of inferences at once (the CPU 1 - it
//!   uses every core already -, a GPU [`EngineConfig::gpu_slots`]); a caller
//!   takes the least busy device or waits;
//! - a GPU run that fails (burn reports a lost device or exhausted memory as
//!   a panic) costs that GPU a slot and is retried; down to one slot, the GPU
//!   is given up and the CPU takes over (logged once);
//! - each inference runs on a thread of its own inside the engine's thread
//!   pool, the caller only waits: run on a pipeline worker of rayon, the
//!   model's own rayon work could pick up another tile's task while waiting,
//!   whose model run would then wait for the slot this thread holds.

use crate::errors::NnError;
use crate::model::{Image, Model, RawModel, Written};
use burn::tensor::{Device, DeviceType};
use log::{info, warn};
use std::any::{Any, TypeId};
use std::collections::HashMap;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Condvar, Mutex, MutexGuard, PoisonError};
use std::time::SystemTime;

/// Which devices model runs may use.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum DeviceSelection {
    /// Every discrete GPU there is, else the CPU.
    #[default]
    Auto,
    Cpu,
    /// These discrete GPUs (indexes); ones that don't exist are ignored.
    Gpus(Vec<usize>),
}

/// How the engine sets itself up, once per process.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EngineConfig {
    pub devices: DeviceSelection,
    /// Model runs at once per GPU (at least 1).
    pub gpu_slots: usize,
    /// CPU cores the engine may use.
    pub cores: usize,
}

/// A device model runs use, and how many at once.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeviceInfo {
    /// E.g. "CPU", "NVIDIA GeForce RTX 4070".
    pub name: String,
    pub slots: usize,
}

/// What the engine is, after it set itself up - for the log and the
/// settings dialog.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EngineInfo {
    /// E.g. "burn 0.22".
    pub name: String,
    pub devices: Vec<DeviceInfo>,
}

/// How a model is loaded onto a device: `Model::load` or `RawModel::load`.
type Loader<M> = fn(&Path, &Device, bool) -> Result<M, NnError>;

type ModelKey = (TypeId, PathBuf);
type ModelCache = HashMap<ModelKey, (Option<SystemTime>, Arc<dyn Any + Send + Sync>)>;

/// One device with the models loaded onto it.
struct EngineDevice {
    device: Device,
    name: String,
    gpu: bool,
    models: Mutex<ModelCache>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct SlotState {
    /// Inferences this device may run at once (0: not used).
    capacity: usize,
    in_use: usize,
}

/// The devices and their slots, see the module docs.
pub struct Engine {
    /// The GPUs first, the CPU last (its capacity 0 while a GPU runs).
    devices: Vec<EngineDevice>,
    slots: Mutex<Vec<SlotState>>,
    freed: Condvar,
    pool: rayon::ThreadPool,
    /// Held by a run the GPU had no memory for while it runs on the CPU:
    /// one at a time, as the RAM plan has one CPU inference.
    cpu_fallback: Mutex<()>,
}

/// A panic's message.
fn panic_text(panic: &Box<dyn Any + Send>) -> String {
    panic
        .downcast_ref::<String>()
        .map(String::as_str)
        .or_else(|| panic.downcast_ref::<&str>().copied())
        .unwrap_or("a panic")
        .to_string()
}

/// The memory a GPU states it can hold (Vulkan: its largest device-local
/// heap, Metal: the recommended working set), `None` for the CPU or when the
/// driver gives no figure. Opens the device - call it for one models run on.
pub(crate) fn gpu_memory(device: &Device) -> Option<u64> {
    use burn_dispatch::DispatchDevice;
    use burn_dispatch::devices::CubeDevice;
    match device.as_dispatch() {
        DispatchDevice::Cube(cube @ CubeDevice::Wgpu(_)) => {
            catch_unwind(|| cube.client().properties().memory.max_memory())
                .ok()
                .flatten()
        }
        _ => None,
    }
}

/// What a device wgpu lists is, as far as choosing one goes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum GpuKind {
    Discrete,
    Integrated,
    /// Anything else: software adapters (llvmpipe, lavapipe, WARP report
    /// themselves as CPUs), virtual GPUs, other backends.
    Unused,
}

/// The kind of `device`, read from wgpu's adapter type (without opening it).
fn gpu_kind(device: &Device) -> GpuKind {
    use burn_dispatch::DispatchDevice;
    use burn_dispatch::devices::{CubeDevice, WgpuDevice, WgpuDeviceKind};
    match device.as_dispatch() {
        DispatchDevice::Cube(CubeDevice::Wgpu(WgpuDevice { kind, .. })) => match kind {
            WgpuDeviceKind::DiscreteGpu(_) => GpuKind::Discrete,
            WgpuDeviceKind::IntegratedGpu(_) => GpuKind::Integrated,
            _ => GpuKind::Unused,
        },
        _ => GpuKind::Unused,
    }
}

/// Sets CubeCL's (the GPU backend's) configuration once per process, before
/// any GPU is opened: with `cache`, the autotune results (which kernel is
/// fastest for each operation and tensor shape on this GPU) and the compiled
/// kernels are kept in that folder, so only the first model run on a GPU
/// tunes and compiles - measured 2026-11-26: Cellpose-SAM's first image
/// ~28 s slower than the next. Without one, nothing is written. Either way
/// CubeCL reads no `cubecl.toml` / `burn.toml` from the current folder and
/// keeps no cache in a `target/` folder above it (its defaults). The cache
/// is keyed by the GPU, the shapes and CubeCL's version - not the weights;
/// a folder that can't be written falls back to memory. Returns `false` when
/// the configuration was already set (or read) - the first call wins.
pub fn configure_gpu(cache: Option<&Path>) -> bool {
    use burn::cubecl::config::cache::CacheConfig;
    use burn::cubecl::config::{CubeClRuntimeConfig, RuntimeConfig};
    let mut config = CubeClRuntimeConfig::default();
    match cache {
        Some(dir) => {
            config.environment.path = CacheConfig::Directory(dir.to_path_buf());
            config.compilation.cache = true;
        }
        None => config.autotune.disable_cache = true,
    }
    CubeClRuntimeConfig::try_set(config)
}

/// The GPUs wgpu finds, discrete ones with their device and name (opened to
/// read it); integrated ones are only counted - never opened, models don't
/// run on them.
fn wgpu_gpus() -> (Vec<(Device, String)>, usize) {
    // Nobody configured CubeCL: no disk cache, no config files read.
    configure_gpu(None);
    let found = catch_unwind(|| {
        let mut discrete = Vec::new();
        let mut integrated = 0;
        for d in Device::enumerate(DeviceType::Wgpu).into_vec() {
            match gpu_kind(&d) {
                GpuKind::Discrete => {
                    let name = d.identity().map_or_else(|| "GPU".to_string(), |id| id.name);
                    discrete.push((d, name));
                }
                GpuKind::Integrated => integrated += 1,
                GpuKind::Unused => {}
            }
        }
        (discrete, integrated)
    });
    found.unwrap_or_else(|_| {
        warn!("Looking for GPUs failed - models run on the CPU");
        (Vec::new(), 0)
    })
}

/// The discrete GPUs wgpu finds: their devices and names.
fn discrete_gpus() -> Vec<(Device, String)> {
    wgpu_gpus().0
}

/// The GPU indexes to use out of `count`; asked-for ones that don't exist
/// are ignored (with a warning).
fn selected_gpus(selection: &DeviceSelection, count: usize) -> Vec<usize> {
    match selection {
        DeviceSelection::Auto => (0..count).collect(),
        DeviceSelection::Cpu => Vec::new(),
        DeviceSelection::Gpus(gpus) => {
            let (exist, missing): (Vec<usize>, Vec<usize>) = gpus.iter().partition(|&&g| g < count);
            if !missing.is_empty() {
                warn!("GPUs {missing:?} don't exist ({count} found); ignoring them");
            }
            exist
        }
    }
}

/// The device with the most free slots (the first on a tie); `None` if all
/// are busy.
fn pick_slot(slots: &[SlotState]) -> Option<usize> {
    let mut best: Option<(usize, usize)> = None;
    for (index, slot) in slots.iter().enumerate() {
        let free = slot.capacity.saturating_sub(slot.in_use);
        if free > 0 && best.is_none_or(|(_, best_free)| free > best_free) {
            best = Some((index, free));
        }
    }
    best.map(|(index, _)| index)
}

/// A taken slot, given back when dropped (also on errors and panics).
struct SlotLease<'a> {
    engine: &'a Engine,
    index: usize,
}

impl Drop for SlotLease<'_> {
    fn drop(&mut self) {
        let mut slots = self.engine.lock_slots();
        slots[self.index].in_use = slots[self.index].in_use.saturating_sub(1);
        self.engine.freed.notify_all();
    }
}

impl Engine {
    /// The devices `config` selects (see the module docs).
    pub fn new(config: &EngineConfig) -> Self {
        let gpus = if config.devices == DeviceSelection::Cpu {
            Vec::new()
        } else {
            discrete_gpus()
        };
        let chosen = selected_gpus(&config.devices, gpus.len());
        Self::with_devices(
            gpus.into_iter()
                .enumerate()
                .filter(|(i, _)| chosen.contains(i))
                .map(|(_, gpu)| gpu)
                .collect(),
            config.gpu_slots,
            config.cores,
        )
    }

    /// `gpus` with `gpu_slots` each, the CPU as the fallback, `cores` threads.
    fn with_devices(gpus: Vec<(Device, String)>, gpu_slots: usize, cores: usize) -> Self {
        let any_gpu = !gpus.is_empty();
        let mut devices: Vec<EngineDevice> = gpus
            .into_iter()
            .map(|(device, name)| EngineDevice {
                device,
                name,
                gpu: true,
                models: Mutex::new(HashMap::new()),
            })
            .collect();
        devices.push(EngineDevice {
            device: Device::flex(),
            name: "CPU".into(),
            gpu: false,
            models: Mutex::new(HashMap::new()),
        });
        let slots = devices
            .iter()
            .map(|d| SlotState {
                capacity: match (d.gpu, any_gpu) {
                    (true, _) => gpu_slots.max(1),
                    (false, true) => 0,
                    (false, false) => 1,
                },
                in_use: 0,
            })
            .collect();
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(cores.max(1))
            .thread_name(|i| format!("nn-{i}"))
            .build()
            .expect("the model threads start");
        let engine = Self {
            devices,
            slots: Mutex::new(slots),
            freed: Condvar::new(),
            pool,
            cpu_fallback: Mutex::new(()),
        };
        info!(
            "AI models: {:?} ({} threads)",
            engine.info().devices,
            cores.max(1)
        );
        engine
    }

    pub fn info(&self) -> EngineInfo {
        let slots = self.lock_slots();
        EngineInfo {
            name: "burn 0.22".into(),
            devices: self
                .devices
                .iter()
                .zip(slots.iter())
                .filter(|(_, s)| s.capacity > 0)
                .map(|(d, s)| DeviceInfo {
                    name: d.name.clone(),
                    slots: s.capacity,
                })
                .collect(),
        }
    }

    fn lock_slots(&self) -> MutexGuard<'_, Vec<SlotState>> {
        self.slots.lock().unwrap_or_else(PoisonError::into_inner)
    }

    fn acquire(&self) -> SlotLease<'_> {
        let mut slots = self.lock_slots();
        loop {
            if let Some(index) = pick_slot(&slots) {
                slots[index].in_use += 1;
                return SlotLease {
                    engine: self,
                    index,
                };
            }
            slots = self
                .freed
                .wait(slots)
                .unwrap_or_else(PoisonError::into_inner);
        }
    }

    /// Device `index` failed: one slot fewer, or - down to one - given up
    /// with the CPU taking over when no GPU is left. Logged.
    fn device_failed(&self, index: usize, why: &str) {
        let mut slots = self.lock_slots();
        let name = &self.devices[index].name;
        if slots[index].capacity > 1 {
            slots[index].capacity -= 1;
            warn!("{name} failed ({why}); running fewer models on it at once");
        } else {
            slots[index].capacity = 0;
            self.devices[index]
                .models
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .clear();
            let gpus_left = self
                .devices
                .iter()
                .zip(slots.iter())
                .any(|(d, s)| d.gpu && s.capacity > 0);
            if !gpus_left && let Some(cpu) = slots.last_mut() {
                cpu.capacity = 1;
            }
            warn!("{name} failed ({why}); models run on the other devices from now on");
        }
        self.freed.notify_all();
    }

    /// The model `M` from `path` on device `index`, loaded once per device
    /// (again when the file changed).
    fn model<M: Send + Sync + 'static>(
        &self,
        index: usize,
        path: &Path,
        load: Loader<M>,
    ) -> Result<Arc<M>, NnError> {
        let device = &self.devices[index];
        let mtime = std::fs::metadata(path).and_then(|m| m.modified()).ok();
        let key = (TypeId::of::<M>(), path.to_path_buf());
        let mut models = device.models.lock().unwrap_or_else(PoisonError::into_inner);
        if let Some((cached_mtime, model)) = models.get(&key)
            && *cached_mtime == mtime
            && let Ok(model) = Arc::clone(model).downcast::<M>()
        {
            return Ok(model);
        }
        let model = Arc::new(load(path, &device.device, device.gpu)?);
        models.insert(
            key,
            (mtime, Arc::clone(&model) as Arc<dyn Any + Send + Sync>),
        );
        Ok(model)
    }

    /// Runs model `M` from `file` on `image`: [`Model::pre`], then
    /// [`Model::infer`] in a slot of a free device (see the module docs),
    /// then [`Model::post`] into `out` - pre and post outside the slot.
    /// Blocks while every device is busy.
    pub fn run<M: Model>(
        &self,
        file: &Path,
        image: Image<'_>,
        params: &M::Params,
        out: &mut [u32],
    ) -> Result<Written, NnError> {
        let header = M::header(file)?;
        let (input, plan) = M::pre(&header, image, params)?;
        let raw = self.in_slot(file, M::load, |model: &M| model.infer(&input))?;
        // A resized or padded copy of the image is gone before post.
        drop(input);
        M::post(raw, plan, params, out)
    }

    /// Runs `M` (the interface before pre- and post-processing moved into
    /// nn) from `file` on `image` in a slot of a free device.
    pub fn infer_raw<M: RawModel>(
        &self,
        file: &Path,
        image: Image<'_>,
        settings: &M::Settings,
    ) -> Result<M::Output, NnError>
    where
        M::Settings: Sync,
        M::Output: Send,
    {
        self.in_slot(file, M::load, |model: &M| model.infer(image, settings))
    }

    /// Runs `work` with the model from `path` (loaded with `load`) on a free
    /// device, holding its slot (see the module docs). `work` must not have
    /// side effects before it succeeds: a failed GPU run is repeated.
    fn in_slot<M: Send + Sync + 'static, T: Send>(
        &self,
        path: &Path,
        load: Loader<M>,
        work: impl Fn(&M) -> Result<T, NnError> + Sync,
    ) -> Result<T, NnError> {
        loop {
            let lease = self.acquire();
            let index = lease.index;
            let model = self.model::<M>(index, path, load)?;
            let result = std::thread::scope(|s| {
                s.spawn(|| {
                    self.pool
                        .install(|| catch_unwind(AssertUnwindSafe(|| work(&model))))
                })
                .join()
            });
            let gpu = self.devices[index].gpu;
            let failure = match result {
                Ok(Ok(Err(NnError::GpuOutOfMemory(why)))) if gpu => {
                    drop(model);
                    drop(lease);
                    return self.run_on_cpu_once(index, path, load, &work, &why);
                }
                Ok(Ok(result)) => return result,
                Ok(Err(panic)) | Err(panic) => panic,
            };
            let why = panic_text(&failure);
            drop(model);
            drop(lease);
            if !gpu {
                return Err(NnError::Internal(format!("the model run failed: {why}")));
            }
            if crate::errors::is_out_of_memory(&why) {
                return self.run_on_cpu_once(index, path, load, &work, &why);
            }
            self.device_failed(index, &why);
        }
    }

    /// A run GPU `gpu` had no memory for (even in its smallest pieces, or a
    /// model whose pieces are part of its result): this one on the CPU -
    /// the same pieces, so the same result up to float rounding - and the
    /// GPU stays in use for the next runs (an image too large for it says
    /// nothing about the others).
    fn run_on_cpu_once<M: Send + Sync + 'static, T: Send>(
        &self,
        gpu: usize,
        path: &Path,
        load: Loader<M>,
        work: &(impl Fn(&M) -> Result<T, NnError> + Sync),
        why: &str,
    ) -> Result<T, NnError> {
        warn!(
            "{} ran out of memory ({}); this run goes to the CPU, the GPU stays in use",
            self.devices[gpu].name,
            why.lines().last().unwrap_or("")
        );
        let _one_at_a_time = self
            .cpu_fallback
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        let cpu = self.devices.len() - 1;
        let model = self.model::<M>(cpu, path, load)?;
        let result = std::thread::scope(|s| {
            s.spawn(|| {
                self.pool
                    .install(|| catch_unwind(AssertUnwindSafe(|| work(&model))))
            })
            .join()
        });
        match result {
            Ok(Ok(result)) => result,
            Ok(Err(panic)) | Err(panic) => Err(NnError::Internal(format!(
                "the model run failed: {}",
                panic_text(&panic)
            ))),
        }
    }

    /// Inferences running now, on all devices.
    #[cfg(test)]
    fn slots_in_use(&self) -> usize {
        self.lock_slots().iter().map(|s| s.in_use).sum()
    }

    /// Lets go of every loaded model (a run using one keeps it until done).
    pub fn release_models(&self) {
        for device in &self.devices {
            device
                .models
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .clear();
        }
    }
}

/// A GPU of the computer, as [`gpus`] reports it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GpuInfo {
    /// As its driver names it (e.g. "NVIDIA GeForce RTX 3060"); an
    /// integrated GPU is not opened, so "Integrated GPU".
    pub name: String,
    /// Has its own memory. Models run only on discrete GPUs; integrated
    /// ones (sharing the RAM, e.g. Intel's in most laptops) are not used.
    pub discrete: bool,
    /// Models run on it: discrete and chosen by the [`DeviceSelection`].
    pub used: bool,
}

/// Discrete GPUs (number order) then integrated ones, found once per
/// process - slow the first time (opens the discrete GPUs).
fn found_gpus() -> &'static [GpuInfo] {
    use std::sync::OnceLock;
    static FOUND: OnceLock<Vec<GpuInfo>> = OnceLock::new();
    FOUND.get_or_init(|| {
        let (discrete, integrated) = wgpu_gpus();
        let discrete = discrete.into_iter().map(|(_, name)| GpuInfo {
            name,
            discrete: true,
            used: false,
        });
        let integrated = (0..integrated).map(|_| GpuInfo {
            name: "Integrated GPU".into(),
            discrete: false,
            used: false,
        });
        discrete.chain(integrated).collect()
    })
}

/// The computer's GPUs and which ones models run on with `selection` (for
/// display, e.g. the About dialog). Slow on the first call: opens the
/// discrete GPUs (once per process).
pub fn gpus(selection: &DeviceSelection) -> Vec<GpuInfo> {
    let found = found_gpus();
    let chosen = selected_gpus(selection, found.iter().filter(|g| g.discrete).count());
    found
        .iter()
        .enumerate()
        .map(|(index, gpu)| GpuInfo {
            used: gpu.discrete && chosen.contains(&index),
            ..gpu.clone()
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A "model" counting its loads per file - per file, so tests running
    /// at the same time don't count each other's.
    struct Counting;
    static LOADS: Mutex<Vec<(PathBuf, usize)>> = Mutex::new(Vec::new());

    fn loads(path: &Path) -> usize {
        let loads = LOADS.lock().unwrap();
        loads.iter().find(|(p, _)| p == path).map_or(0, |(_, n)| *n)
    }

    impl Counting {
        fn load(path: &Path, _: &Device, _: bool) -> Result<Self, NnError> {
            let mut loads = LOADS.lock().unwrap();
            match loads.iter_mut().find(|(p, _)| p == path) {
                Some((_, n)) => *n += 1,
                None => loads.push((path.to_path_buf(), 1)),
            }
            Ok(Self)
        }
    }

    #[test]
    fn a_model_loads_once_per_device_and_again_after_release() {
        let engine = Engine::with_devices(Vec::new(), 1, 2);
        let dir = tempfile::TempDir::new().unwrap();
        let path = dir.path().join("m");
        std::fs::write(&path, b"x").unwrap();
        for i in 0..3 {
            let out = engine
                .in_slot(&path, Counting::load, |_| Ok(2 * i))
                .unwrap();
            assert_eq!(out, 2 * i);
        }
        assert_eq!(loads(&path), 1);
        engine.release_models();
        engine.in_slot(&path, Counting::load, |_| Ok(())).unwrap();
        assert_eq!(loads(&path), 2);
        assert_eq!(
            engine.info().devices,
            [DeviceInfo {
                name: "CPU".into(),
                slots: 1
            }]
        );
    }

    /// A "model" the GPU never has memory for: out of memory on a GPU,
    /// fine on the CPU.
    struct TooBigForTheGpu {
        gpu: bool,
    }

    impl TooBigForTheGpu {
        fn load(_: &Path, _: &Device, gpu: bool) -> Result<Self, NnError> {
            Ok(Self { gpu })
        }
        fn infer(&self) -> Result<&'static str, NnError> {
            if self.gpu {
                return Err(NnError::GpuOutOfMemory(
                    "out of device memory allocating 2216689664 bytes".into(),
                ));
            }
            Ok("cpu")
        }
    }

    /// Out of GPU memory: that run on the CPU, and the GPU stays in use (a
    /// device failure would give it up).
    #[test]
    fn out_of_gpu_memory_runs_once_on_the_cpu_and_keeps_the_gpu() {
        // burn-flex standing in for a GPU (the engine only reads `gpu`).
        let engine = Engine::with_devices(vec![(Device::flex(), "Test GPU".into())], 1, 2);
        let path = PathBuf::from("too-big");
        for _ in 0..2 {
            let out = engine
                .in_slot(&path, TooBigForTheGpu::load, TooBigForTheGpu::infer)
                .unwrap();
            assert_eq!(out, "cpu");
        }
        assert_eq!(
            engine.info().devices,
            [DeviceInfo {
                name: "Test GPU".into(),
                slots: 1
            }]
        );
    }

    #[test]
    fn a_panic_on_the_cpu_is_an_error_and_the_slot_comes_back() {
        let engine = Engine::with_devices(Vec::new(), 1, 1);
        let path = PathBuf::from("does-not-matter");
        let err = engine
            .in_slot::<_, usize>(&path, Counting::load, |_| panic!("boom"))
            .unwrap_err();
        assert!(err.to_string().contains("boom"), "{err}");
        assert_eq!(engine.in_slot(&path, Counting::load, |_| Ok(6)).unwrap(), 6);
        assert_eq!(engine.slots_in_use(), 0);
    }

    /// The engine every `Stages` run goes through, and what each stage saw.
    static STAGES_ENGINE: std::sync::OnceLock<Engine> = std::sync::OnceLock::new();
    static SEEN: Mutex<Vec<(&'static str, usize)>> = Mutex::new(Vec::new());

    fn seen(stage: &'static str) {
        let in_use = STAGES_ENGINE.get().expect("set up").slots_in_use();
        SEEN.lock().unwrap().push((stage, in_use));
    }

    /// A model noting the slots in use in each of its stages.
    struct Stages;

    impl Model for Stages {
        type Header = ();
        type Params = u32;
        type Input<'a> = &'a [f32];
        type Raw = u32;
        type Plan = usize;
        const NET_BYTES_PER_PIXEL: f64 = 0.0;
        const POST_BYTES_PER_PIXEL: f64 = 0.0;

        fn header(_: &Path) -> Result<(), NnError> {
            seen("header");
            Ok(())
        }
        fn load(_: &Path, _: &Device, _: bool) -> Result<Self, NnError> {
            Ok(Self)
        }
        fn pre<'a>(_: &(), image: Image<'a>, _: &u32) -> Result<(&'a [f32], usize), NnError> {
            seen("pre");
            Ok((image.data, image.width * image.height))
        }
        fn infer(&self, input: &&[f32]) -> Result<u32, NnError> {
            seen("infer");
            Ok(input.len() as u32)
        }
        fn post(raw: u32, plan: usize, params: &u32, out: &mut [u32]) -> Result<Written, NnError> {
            seen("post");
            out[..plan].fill(raw + params);
            Ok(Written::Mask)
        }
    }

    /// Only `infer` runs inside a slot: header, pre and post see none in
    /// use, infer exactly one - and the answer of post lands in `out`.
    #[test]
    fn only_infer_holds_a_slot() {
        let engine = STAGES_ENGINE.get_or_init(|| Engine::with_devices(Vec::new(), 1, 1));
        let image = Image {
            width: 2,
            height: 2,
            channels: 1,
            data: &[0.0; 4],
        };
        let mut out = [0u32; 4];
        let written = engine
            .run::<Stages>(Path::new("stages"), image, &10, &mut out)
            .unwrap();
        assert_eq!(written, Written::Mask);
        assert_eq!(out, [14; 4]);
        assert_eq!(
            *SEEN.lock().unwrap(),
            [("header", 0), ("pre", 0), ("infer", 1), ("post", 0)]
        );
    }

    #[test]
    fn slots_go_to_the_least_busy_device() {
        let free = |capacity, in_use| SlotState { capacity, in_use };
        assert_eq!(pick_slot(&[free(2, 1), free(2, 0), free(0, 0)]), Some(1));
        assert_eq!(pick_slot(&[free(1, 1), free(0, 0)]), None);
        assert_eq!(selected_gpus(&DeviceSelection::Gpus(vec![0, 3]), 2), [0]);
        assert_eq!(selected_gpus(&DeviceSelection::Auto, 2), [0, 1]);
        assert!(selected_gpus(&DeviceSelection::Cpu, 2).is_empty());
    }

    /// Devices are told apart by their wgpu kind: only discrete GPUs run
    /// models, software adapters (wgpu kind `Cpu`) and the CPU backend never
    /// count as GPUs. Lazy devices - nothing is opened, no GPU needed.
    #[test]
    fn devices_are_classified_by_their_wgpu_kind() {
        use burn::tensor::DeviceKind;
        assert_eq!(
            gpu_kind(&Device::wgpu(DeviceKind::DiscreteGpu(1))),
            GpuKind::Discrete
        );
        assert_eq!(
            gpu_kind(&Device::wgpu(DeviceKind::IntegratedGpu(0))),
            GpuKind::Integrated
        );
        for unused in [
            Device::wgpu(DeviceKind::Cpu),
            Device::wgpu(DeviceKind::VirtualGpu(0)),
            Device::flex(),
        ] {
            assert_eq!(gpu_kind(&unused), GpuKind::Unused, "{unused:?}");
        }
    }

    /// Many callers on a pipeline's rayon pool, each running a model whose
    /// work uses rayon itself: no deadlock (each run on its own thread).
    #[test]
    fn rayon_callers_running_rayon_models_finish() {
        use rayon::prelude::*;
        let engine = Engine::with_devices(Vec::new(), 1, 2);
        let path = PathBuf::from("rayon-model");
        let pipeline = rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .unwrap();
        let total: usize = pipeline.install(|| {
            (0..32)
                .into_par_iter()
                .map(|i| {
                    engine
                        .in_slot(&path, Counting::load, |_| {
                            Ok((0..100).into_par_iter().map(|j| j * 0 + i).sum::<usize>())
                        })
                        .unwrap()
                })
                .sum()
        });
        assert_eq!(total, (0..32).map(|i| i * 100).sum::<usize>());
    }
}
