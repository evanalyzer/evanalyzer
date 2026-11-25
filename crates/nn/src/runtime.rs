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
//! engine's scheduler (`crates/ai/src/torch/scheduler.rs`), the same rules:
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
use crate::model::Model;
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
}

/// The discrete GPUs wgpu finds: their devices and names.
fn discrete_gpus() -> Vec<(Device, String)> {
    let found = catch_unwind(|| {
        Device::enumerate(DeviceType::Wgpu)
            .into_vec()
            .into_iter()
            .filter(|d| format!("{d:?}").contains("DiscreteGpu"))
            .map(|d| {
                let name = d.identity().map_or_else(|| "GPU".to_string(), |id| id.name);
                (d, name)
            })
            .collect::<Vec<_>>()
    });
    found.unwrap_or_else(|_| {
        warn!("Looking for GPUs failed - models run on the CPU");
        Vec::new()
    })
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
    fn model<M: Model>(&self, index: usize, path: &Path) -> Result<Arc<M>, NnError> {
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
        let model = Arc::new(M::load(path, &device.device, device.gpu)?);
        models.insert(
            key,
            (mtime, Arc::clone(&model) as Arc<dyn Any + Send + Sync>),
        );
        Ok(model)
    }

    /// Runs `work` with the model `M` from `path` on a free device (see the
    /// module docs). `work` must not have side effects before it succeeds:
    /// a failed GPU run is repeated.
    pub fn run<M: Model, T: Send>(
        &self,
        path: &Path,
        work: impl Fn(&M) -> Result<T, NnError> + Sync,
    ) -> Result<T, NnError> {
        loop {
            let lease = self.acquire();
            let index = lease.index;
            let model = self.model::<M>(index, path)?;
            let result = std::thread::scope(|s| {
                s.spawn(|| {
                    self.pool
                        .install(|| catch_unwind(AssertUnwindSafe(|| work(&model))))
                })
                .join()
            });
            let failure = match result {
                Ok(Ok(result)) => return result,
                Ok(Err(panic)) | Err(panic) => panic,
            };
            let why = failure
                .downcast_ref::<String>()
                .map(String::as_str)
                .or_else(|| failure.downcast_ref::<&str>().copied())
                .unwrap_or("a panic")
                .to_string();
            drop(model);
            drop(lease);
            if !self.devices[index].gpu {
                return Err(NnError::Internal(format!("the model run failed: {why}")));
            }
            self.device_failed(index, &why);
        }
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

/// Whether wgpu sees a discrete GPU (opens the adapters once).
pub fn gpu_available() -> bool {
    use std::sync::OnceLock;
    static FOUND: OnceLock<bool> = OnceLock::new();
    *FOUND.get_or_init(|| !discrete_gpus().is_empty())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::Image;
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// A "model" counting its loads, failing (panicking) on the first runs
    /// it is told to.
    struct Counting;
    static LOADS: AtomicUsize = AtomicUsize::new(0);

    impl Model for Counting {
        type Settings = usize;
        type Output = usize;
        fn load(_: &Path, _: &Device, _: bool) -> Result<Self, NnError> {
            LOADS.fetch_add(1, Ordering::SeqCst);
            Ok(Self)
        }
        fn infer(&self, _: Image<'_>, settings: &usize) -> Result<usize, NnError> {
            Ok(*settings * 2)
        }
    }

    fn image() -> Image<'static> {
        Image {
            width: 1,
            height: 1,
            channels: 1,
            data: &[0.0],
        }
    }

    #[test]
    fn a_model_loads_once_per_device_and_again_after_release() {
        let engine = Engine::with_devices(Vec::new(), 1, 2);
        let dir = tempfile::TempDir::new().unwrap();
        let path = dir.path().join("m");
        std::fs::write(&path, b"x").unwrap();
        let before = LOADS.load(Ordering::SeqCst);
        for i in 0..3 {
            let out = engine
                .run::<Counting, _>(&path, |m| m.infer(image(), &i))
                .unwrap();
            assert_eq!(out, 2 * i);
        }
        assert_eq!(LOADS.load(Ordering::SeqCst) - before, 1);
        engine.release_models();
        engine
            .run::<Counting, _>(&path, |m| m.infer(image(), &1))
            .unwrap();
        assert_eq!(LOADS.load(Ordering::SeqCst) - before, 2);
        assert_eq!(
            engine.info().devices,
            [DeviceInfo {
                name: "CPU".into(),
                slots: 1
            }]
        );
    }

    #[test]
    fn a_panic_on_the_cpu_is_an_error_and_the_slot_comes_back() {
        let engine = Engine::with_devices(Vec::new(), 1, 1);
        let path = PathBuf::from("does-not-matter");
        let err = engine
            .run::<Counting, usize>(&path, |_| panic!("boom"))
            .unwrap_err();
        assert!(err.to_string().contains("boom"), "{err}");
        assert_eq!(
            engine
                .run::<Counting, _>(&path, |m| m.infer(image(), &3))
                .unwrap(),
            6
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
                        .run::<Counting, _>(&path, |_| {
                            Ok((0..100).into_par_iter().map(|j| j * 0 + i).sum::<usize>())
                        })
                        .unwrap()
                })
                .sum()
        });
        assert_eq!(total, (0..32).map(|i| i * 100).sum::<usize>());
    }
}
