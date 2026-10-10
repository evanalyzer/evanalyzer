//! # model_cache
//!
//! **Author:** Joachim Danmayr
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **AGPL-3.0**.

use crate::ai_devices::{AiDeviceSelection, ai_device_options};
use evanalyzer_cfg::core_types::InternalErrors;
use log::{info, warn};
use std::{
    cell::RefCell,
    collections::HashMap,
    path::Path,
    path::PathBuf,
    sync::{Arc, Condvar, LazyLock, Mutex, PoisonError},
    time::SystemTime,
};
use tch::{CModule, Device, TchError};

use crate::ai_learning::model::SavedClassifier;

/// Looks up `path` in `cache`, inserting the result of `load` on a miss.
/// A hit additionally requires `path`'s current mtime to match the mtime
/// recorded at load time - otherwise the file has been overwritten since
/// (e.g. a model retrained and re-exported under the same path) and the
/// cached value is stale, so this falls through to `load` exactly as on a
/// plain miss. A path whose mtime can't be read (missing file, unsupported
/// filesystem, ...) reads as `None` on both sides and still compares equal,
/// so it degrades to the old always-reuse behavior rather than refusing to
/// cache at all.
///
/// Kept generic (and free of any `tch`/thread-local dependency) so the cache
/// semantics can be unit tested directly.
fn get_or_insert<V, E>(
    cache: &mut HashMap<PathBuf, (Option<SystemTime>, Arc<V>)>,
    path: &Path,
    load: impl FnOnce() -> Result<V, E>,
) -> Result<Arc<V>, E> {
    let mtime = std::fs::metadata(path).and_then(|m| m.modified()).ok();
    if let Some((cached_mtime, cached)) = cache.get(path)
        && *cached_mtime == mtime
    {
        return Ok(Arc::clone(cached));
    }
    let value = Arc::new(load()?);
    cache.insert(path.to_path_buf(), (mtime, Arc::clone(&value)));
    Ok(value)
}

type ModelMap = HashMap<PathBuf, (Option<SystemTime>, Arc<CModule>)>;

/// One device AI models run on, with the models loaded onto it.
struct TorchDevice {
    device: Device,
    /// One copy per model file, shared by all of this device's slots.
    models: Mutex<ModelMap>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct SlotState {
    /// Inferences this device may run at once.
    capacity: usize,
    in_use: usize,
}

/// Spreads AI inference over every available device - all CUDA GPUs, or
/// the CPU when there is none.
///
/// Why not just let every pipeline worker run its model (as before):
/// - Each worker loaded its own copy of the model. With ~20 workers that is
///   ~20 copies on one GPU (a Cellpose-SAM model is ~1.2 GB), which
///   overflows its memory - on Windows the driver then silently pages into
///   system RAM and inference all but stops.
/// - Torch runs CPU ops on its own thread pool sized to all cores; ~20
///   workers doing that at once start hundreds of busy threads fighting
///   over the cores (100% CPU, little progress).
///
/// Instead each device holds one copy of each model and runs at most its
/// slot count of inferences at once; a worker takes the least busy device or
/// waits. The CPU gets one slot (Torch already uses every core there), a GPU
/// [`AiDeviceOptions::gpu_slots`](crate::ai_devices::AiDeviceOptions). Torch's
/// CPU threads are split between all slots. Configured once per process with
/// [`configure_ai_devices`](crate::ai_devices::configure_ai_devices).
pub(crate) struct TorchScheduler {
    devices: Vec<TorchDevice>,
    slots: Mutex<Vec<SlotState>>,
    freed: Condvar,
    cores: usize,
}

static SCHEDULER: LazyLock<TorchScheduler> = LazyLock::new(TorchScheduler::from_system);

/// Runs `work` with the model at `path` on a free device of the process-wide
/// [`TorchScheduler`]; see [`TorchScheduler::run`].
pub(crate) fn run_with_model<T>(
    path: &Path,
    load_error: impl Fn(TchError) -> InternalErrors,
    work: impl Fn(&CModule, Device) -> Result<T, InternalErrors>,
) -> Result<T, InternalErrors> {
    SCHEDULER.run(path, load_error, work)
}

impl TorchScheduler {
    /// `devices` with their slot counts, sharing `cores` CPU threads.
    fn new(devices: Vec<(Device, usize)>, cores: usize) -> Self {
        let slots = devices
            .iter()
            .map(|(_, capacity)| SlotState {
                capacity: (*capacity).max(1),
                in_use: 0,
            })
            .collect();
        Self {
            devices: devices
                .into_iter()
                .map(|(device, _)| TorchDevice {
                    device,
                    models: Mutex::new(HashMap::new()),
                })
                .collect(),
            slots: Mutex::new(slots),
            freed: Condvar::new(),
            cores: cores.max(1),
        }
    }

    fn from_system() -> Self {
        let options = ai_device_options();
        let cores = std::thread::available_parallelism().map_or(1, |n| n.get());
        let gpu_count = if tch::Cuda::is_available() {
            tch::Cuda::device_count().max(0) as usize
        } else {
            0
        };
        let gpus = selected_gpus(&options.devices, gpu_count);
        let devices: Vec<(Device, usize)> = if gpus.is_empty() {
            vec![(Device::Cpu, 1)]
        } else {
            gpus.into_iter()
                .map(|g| (Device::Cuda(g), options.gpu_slots))
                .collect()
        };
        info!(
            "AI inference devices (device, parallel inferences): {devices:?} \
             ({gpu_count} available GPUs, {cores} CPU cores)"
        );
        Self::new(devices, cores)
    }

    /// Runs `work` with the model at `path` loaded on a free device.
    ///
    /// Waits while every device is busy. If `work` runs out of memory and
    /// its device allows more than one inference at once, that device gets
    /// one slot fewer from then on and `work` is retried - `work` must
    /// therefore not have side effects before it succeeds.
    ///
    /// `work` must not use rayon: a thread waiting inside rayon can pick up
    /// another tile's task, whose AI step would then wait for the slot this
    /// thread already holds.
    pub(crate) fn run<T>(
        &self,
        path: &Path,
        load_error: impl Fn(TchError) -> InternalErrors,
        work: impl Fn(&CModule, Device) -> Result<T, InternalErrors>,
    ) -> Result<T, InternalErrors> {
        loop {
            let lease = self.acquire();
            let index = lease.index;
            let device = self.devices[index].device;
            // Torch's thread count is per calling thread, so it is set for
            // every inference (cheap).
            tch::set_num_threads(self.threads_per_slot());
            disable_jit_fusers();
            let model = self.model(index, path).map_err(&load_error)?;
            match work(&model, device) {
                Err(e) if is_out_of_memory(&e) => {
                    drop(model);
                    drop(lease);
                    if self.reduce_capacity(index) {
                        warn!(
                            "{device:?} ran out of memory; running fewer inferences on it at once"
                        );
                        continue;
                    }
                    return Err(InternalErrors::Generic(format!(
                        "{device:?} ran out of memory even running one inference at a time - \
                         use smaller tiles or a smaller model: {e}"
                    )));
                }
                result => return result,
            }
        }
    }

    fn threads_per_slot(&self) -> i32 {
        let total: usize = self.lock_slots().iter().map(|s| s.capacity).sum();
        (self.cores / total.max(1)).max(1) as i32
    }

    fn lock_slots(&self) -> std::sync::MutexGuard<'_, Vec<SlotState>> {
        self.slots.lock().unwrap_or_else(PoisonError::into_inner)
    }

    fn acquire(&self) -> SlotLease<'_> {
        let mut slots = self.lock_slots();
        loop {
            if let Some(index) = pick_slot(&slots) {
                slots[index].in_use += 1;
                return SlotLease {
                    scheduler: self,
                    index,
                };
            }
            slots = self
                .freed
                .wait(slots)
                .unwrap_or_else(PoisonError::into_inner);
        }
    }

    fn release(&self, index: usize) {
        let mut slots = self.lock_slots();
        slots[index].in_use = slots[index].in_use.saturating_sub(1);
        self.freed.notify_all();
    }

    /// One slot fewer on device `index`; `false` if it is already down to one.
    fn reduce_capacity(&self, index: usize) -> bool {
        let mut slots = self.lock_slots();
        if slots[index].capacity > 1 {
            slots[index].capacity -= 1;
            true
        } else {
            false
        }
    }

    /// The model at `path` on device `index`, loaded once per device.
    fn model(&self, index: usize, path: &Path) -> Result<Arc<CModule>, TchError> {
        let TorchDevice { device, models } = &self.devices[index];
        let mut models = models.lock().unwrap_or_else(PoisonError::into_inner);
        get_or_insert(&mut models, path, || CModule::load_on_device(path, *device))
    }
}

/// A taken inference slot; given back when dropped (also on errors/panics).
struct SlotLease<'a> {
    scheduler: &'a TorchScheduler,
    index: usize,
}

impl Drop for SlotLease<'_> {
    fn drop(&mut self) {
        self.scheduler.release(self.index);
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

/// The GPU indexes to use out of `gpu_count`; selected ones that don't
/// exist are ignored (with a warning).
fn selected_gpus(selection: &AiDeviceSelection, gpu_count: usize) -> Vec<usize> {
    match selection {
        AiDeviceSelection::Auto => (0..gpu_count).collect(),
        AiDeviceSelection::Cpu => Vec::new(),
        AiDeviceSelection::Gpus(gpus) => {
            let (exist, missing): (Vec<usize>, Vec<usize>) =
                gpus.iter().partition(|&&g| g < gpu_count);
            if !missing.is_empty() {
                warn!("GPUs {missing:?} don't exist ({gpu_count} found); ignoring them");
            }
            exist
        }
    }
}

/// CUDA ("CUDA out of memory") and CPU allocator out-of-memory errors.
fn is_out_of_memory(err: &InternalErrors) -> bool {
    let message = err.to_string().to_ascii_lowercase();
    message.contains("out of memory") || message.contains("can't allocate memory")
}

/// Some traced graphs (e.g. Cellpose-SAM's ViT encoder, whose relative-
/// position-embedding math is a long chain of elementwise adds/unsqueezes)
/// contain op sequences PyTorch's JIT fuser tries to compile into a single
/// CUDA kernel via NVRTC on first run. That requires the CUDA *toolkit*'s
/// `libnvrtc-builtins` to be installed system-wide - most end-user machines
/// that only have a GPU driver installed don't have it, and the failure only
/// surfaces the first time that particular fused op pattern runs. Disabling
/// both JIT fusers trades that (unfused, marginally slower) elementwise math
/// for never depending on NVRTC being present. Cheap enough to set before
/// every inference - whether this flag is process-global or per-thread
/// isn't documented.
fn disable_jit_fusers() {
    tch::jit::set_tensor_expr_fuser_enabled(false);
    tch::jit::fuser_cuda_set_enabled(false);
}

thread_local! {
    // Rayon reuses a fixed pool of worker threads, so this persists per
    // worker thread for the process lifetime, not just one pipeline run, so a whole-slide
    // image's many tiles don't each pay the JSON-deserialize + smartcore/burn
    // reconstruction cost of loading the same `.evamodel` file again.
    static CLASSIFIER_CACHE: RefCell<HashMap<PathBuf, (Option<SystemTime>, Arc<SavedClassifier>)>> =
        RefCell::new(HashMap::new());
}

/// Returns the [`SavedClassifier`] loaded from `path`, reusing a previously
/// loaded instance on this thread instead of re-reading and re-deserializing
/// the model file. `load` is only invoked on a cache miss.
pub fn load_cached_classifier<E>(
    path: &Path,
    load: impl FnOnce() -> Result<SavedClassifier, E>,
) -> Result<Arc<SavedClassifier>, E> {
    CLASSIFIER_CACHE.with(|cache| get_or_insert(&mut cache.borrow_mut(), path, load))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;
    use std::time::Duration;

    // -- TorchScheduler ------------------------------------------------------

    use crate::algos::ai_segmentation::test_support::trace_and_save_model;
    use std::sync::atomic::{AtomicUsize, Ordering};

    fn test_model() -> (tempfile::TempDir, PathBuf) {
        trace_and_save_model(1, 1, 2, |x| x.shallow_clone())
    }

    fn load_error(e: TchError) -> InternalErrors {
        InternalErrors::Generic(format!("load failed: {e}"))
    }

    /// CPU devices (stand-ins for GPUs) with these slot counts.
    fn devices(slots: &[usize], cores: usize) -> TorchScheduler {
        TorchScheduler::new(slots.iter().map(|&n| (Device::Cpu, n)).collect(), cores)
    }

    fn capacity(scheduler: &TorchScheduler, index: usize) -> usize {
        scheduler.lock_slots()[index].capacity
    }

    /// Runs `threads` inferences at once, each holding its slot ~30 ms, and
    /// returns how many ran at the same time at most.
    fn max_concurrency(scheduler: &TorchScheduler, path: &Path, threads: usize) -> usize {
        let running = AtomicUsize::new(0);
        let peak = AtomicUsize::new(0);
        std::thread::scope(|scope| {
            for _ in 0..threads {
                scope.spawn(|| {
                    scheduler
                        .run(path, load_error, |_, _| {
                            let now = running.fetch_add(1, Ordering::SeqCst) + 1;
                            peak.fetch_max(now, Ordering::SeqCst);
                            std::thread::sleep(std::time::Duration::from_millis(30));
                            running.fetch_sub(1, Ordering::SeqCst);
                            Ok(())
                        })
                        .unwrap();
                });
            }
        });
        peak.load(Ordering::SeqCst)
    }

    #[test]
    fn inferences_are_limited_to_the_free_slots() {
        let (_dir, path) = test_model();
        assert_eq!(max_concurrency(&devices(&[1], 8), &path, 4), 1);
    }

    #[test]
    fn inferences_spread_over_all_devices() {
        // Two devices with two slots each: all four run at once.
        let (_dir, path) = test_model();
        assert_eq!(max_concurrency(&devices(&[2, 2], 8), &path, 4), 4);
    }

    #[test]
    fn each_device_loads_a_model_once() {
        let (_dir, path) = test_model();
        let scheduler = devices(&[2, 2], 8);
        max_concurrency(&scheduler, &path, 8);
        for device in &scheduler.devices {
            assert_eq!(device.models.lock().unwrap().len(), 1);
        }
        let first = scheduler.model(0, &path).unwrap();
        assert!(Arc::ptr_eq(&first, &scheduler.model(0, &path).unwrap()));
        assert!(!Arc::ptr_eq(&first, &scheduler.model(1, &path).unwrap()));
    }

    #[test]
    fn out_of_memory_lowers_the_slots_and_retries() {
        let (_dir, path) = test_model();
        let scheduler = devices(&[2], 8);
        let calls = AtomicUsize::new(0);
        let result = scheduler.run(&path, load_error, |_, _| {
            if calls.fetch_add(1, Ordering::SeqCst) == 0 {
                Err(InternalErrors::Generic(
                    "CUDA out of memory. Tried to allocate 2 GiB".into(),
                ))
            } else {
                Ok(7)
            }
        });
        assert_eq!(result.unwrap(), 7);
        assert_eq!(calls.load(Ordering::SeqCst), 2);
        assert_eq!(capacity(&scheduler, 0), 1);
        assert_eq!(scheduler.lock_slots()[0].in_use, 0);
    }

    #[test]
    fn out_of_memory_with_one_slot_left_is_an_error() {
        let (_dir, path) = test_model();
        let err = devices(&[1], 8)
            .run(&path, load_error, |_, _| -> Result<(), _> {
                Err(InternalErrors::Generic("CUDA out of memory".into()))
            })
            .unwrap_err()
            .to_string();
        assert!(err.contains("one inference at a time"), "{err}");
    }

    #[test]
    fn other_errors_are_not_retried() {
        let (_dir, path) = test_model();
        let scheduler = devices(&[2], 8);
        let calls = AtomicUsize::new(0);
        let result = scheduler.run(&path, load_error, |_, _| -> Result<(), _> {
            calls.fetch_add(1, Ordering::SeqCst);
            Err(InternalErrors::Generic("bad output shape".into()))
        });
        assert!(result.is_err());
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        assert_eq!(capacity(&scheduler, 0), 2);
    }

    #[test]
    fn load_errors_use_the_callers_message_and_free_the_slot() {
        let scheduler = devices(&[1], 8);
        let err = scheduler
            .run(Path::new("/no/such/model.pt"), load_error, |_, _| Ok(()))
            .unwrap_err()
            .to_string();
        assert!(err.contains("load failed"), "{err}");
        assert_eq!(scheduler.lock_slots()[0].in_use, 0);
    }

    #[test]
    fn a_panic_during_inference_frees_the_slot() {
        let (_dir, path) = test_model();
        let scheduler = devices(&[1], 8);
        let panicked = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _ = scheduler.run(&path, load_error, |_, _| -> Result<(), _> {
                panic!("model failed")
            });
        }));
        assert!(panicked.is_err());
        assert_eq!(scheduler.lock_slots()[0].in_use, 0);
        scheduler.run(&path, load_error, |_, _| Ok(())).unwrap();
    }

    #[test]
    fn cpu_threads_follow_the_current_slots() {
        assert_eq!(devices(&[1], 16).threads_per_slot(), 16);
        assert_eq!(devices(&[2, 2], 16).threads_per_slot(), 4);
        assert_eq!(devices(&[8, 8], 4).threads_per_slot(), 1);
    }

    #[test]
    fn pick_slot_prefers_the_least_busy_device() {
        let s = |capacity, in_use| SlotState { capacity, in_use };
        assert_eq!(pick_slot(&[s(2, 1), s(2, 0)]), Some(1));
        assert_eq!(pick_slot(&[s(2, 0), s(2, 0)]), Some(0));
        assert_eq!(pick_slot(&[s(1, 1), s(2, 2)]), None);
        // A device whose capacity was lowered below its running inferences.
        assert_eq!(pick_slot(&[s(1, 2), s(1, 0)]), Some(1));
    }

    #[test]
    fn gpu_selection() {
        assert_eq!(selected_gpus(&AiDeviceSelection::Auto, 3), vec![0, 1, 2]);
        assert_eq!(
            selected_gpus(&AiDeviceSelection::Cpu, 2),
            Vec::<usize>::new()
        );
        assert_eq!(
            selected_gpus(&AiDeviceSelection::Gpus(vec![2, 0, 9]), 3),
            vec![2, 0]
        );
        assert_eq!(
            selected_gpus(&AiDeviceSelection::Auto, 0),
            Vec::<usize>::new()
        );
    }

    #[test]
    fn out_of_memory_detection() {
        let e = |m: &str| InternalErrors::Generic(m.into());
        assert!(is_out_of_memory(&e(
            "Cellpose: CUDA out of memory. Tried to allocate"
        )));
        assert!(is_out_of_memory(&e(
            "DefaultCPUAllocator: can't allocate memory"
        )));
        assert!(!is_out_of_memory(&e("shape mismatch")));
    }

    #[test]
    fn cache_hit_reuses_value_without_calling_loader_again() {
        let mut cache: HashMap<PathBuf, (Option<SystemTime>, Arc<u32>)> = HashMap::new();
        let path = PathBuf::from("/models/one.pt");
        let load_calls = Cell::new(0);

        let first = get_or_insert::<u32, ()>(&mut cache, &path, || {
            load_calls.set(load_calls.get() + 1);
            Ok(42)
        })
        .unwrap();
        let second = get_or_insert::<u32, ()>(&mut cache, &path, || {
            load_calls.set(load_calls.get() + 1);
            Ok(0) // Would prove staleness if this ever won.
        })
        .unwrap();

        assert_eq!(
            load_calls.get(),
            1,
            "loader must run exactly once for a repeated path"
        );
        assert_eq!(*first, 42);
        assert_eq!(*second, 42);
        assert!(
            Arc::ptr_eq(&first, &second),
            "second call must return the same cached Arc"
        );
    }

    #[test]
    fn different_paths_are_cached_independently() {
        let mut cache: HashMap<PathBuf, (Option<SystemTime>, Arc<u32>)> = HashMap::new();
        let load_calls = Cell::new(0);

        let a = get_or_insert::<u32, ()>(&mut cache, Path::new("/models/a.pt"), || {
            load_calls.set(load_calls.get() + 1);
            Ok(1)
        })
        .unwrap();
        let b = get_or_insert::<u32, ()>(&mut cache, Path::new("/models/b.pt"), || {
            load_calls.set(load_calls.get() + 1);
            Ok(2)
        })
        .unwrap();

        assert_eq!(
            load_calls.get(),
            2,
            "a different path must trigger its own load"
        );
        assert_eq!(*a, 1);
        assert_eq!(*b, 2);
    }

    #[test]
    fn loader_error_is_not_cached() {
        let mut cache: HashMap<PathBuf, (Option<SystemTime>, Arc<u32>)> = HashMap::new();
        let path = PathBuf::from("/models/broken.pt");
        let load_calls = Cell::new(0);

        let first = get_or_insert::<u32, &'static str>(&mut cache, &path, || {
            load_calls.set(load_calls.get() + 1);
            Err("boom")
        });
        assert!(first.is_err());

        let second = get_or_insert::<u32, &'static str>(&mut cache, &path, || {
            load_calls.set(load_calls.get() + 1);
            Ok(7)
        })
        .unwrap();

        assert_eq!(
            load_calls.get(),
            2,
            "a failed load must not poison the cache entry"
        );
        assert_eq!(*second, 7);
    }

    #[test]
    fn a_changed_mtime_invalidates_the_cache_entry() {
        // Simulates retraining and re-saving a model under the same path
        // while the process is still running - the point of keying on mtime
        // at all, since these caches persist for the whole process lifetime,
        // not just one pipeline run.
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("model.evamodel");
        std::fs::write(&path, b"v1").unwrap();

        let mut cache: HashMap<PathBuf, (Option<SystemTime>, Arc<u32>)> = HashMap::new();
        let load_calls = Cell::new(0);

        let first = get_or_insert::<u32, ()>(&mut cache, &path, || {
            load_calls.set(load_calls.get() + 1);
            Ok(1)
        })
        .unwrap();

        // Same path, unchanged mtime: still a hit.
        let still_cached = get_or_insert::<u32, ()>(&mut cache, &path, || {
            load_calls.set(load_calls.get() + 1);
            Ok(99)
        })
        .unwrap();
        assert_eq!(
            load_calls.get(),
            1,
            "unchanged mtime must still hit the cache"
        );
        assert!(Arc::ptr_eq(&first, &still_cached));

        // Overwrite the file with a distinctly later mtime (filesystem mtime
        // resolution can be coarse, so bump it explicitly rather than
        // relying on wall-clock delay between writes).
        std::fs::write(&path, b"v2").unwrap();
        let new_mtime =
            std::fs::metadata(&path).unwrap().modified().unwrap() + Duration::from_secs(5);
        let file = std::fs::File::open(&path).unwrap();
        file.set_modified(new_mtime).unwrap();

        let after_retrain = get_or_insert::<u32, ()>(&mut cache, &path, || {
            load_calls.set(load_calls.get() + 1);
            Ok(2)
        })
        .unwrap();

        assert_eq!(
            load_calls.get(),
            2,
            "a changed mtime must be treated as a cache miss"
        );
        assert_eq!(*after_retrain, 2);
        assert!(!Arc::ptr_eq(&first, &after_retrain));
    }

    // -- load_cached_classifier (real end-to-end, unlike `TorchSession::model` -
    // a `SavedClassifier` is a plain in-memory value, no TorchScript fixture
    // file needed to exercise the real cache) ------------------------------

    fn sample_classifier() -> SavedClassifier {
        use crate::ai_learning::model::CURRENT_SAVED_CLASSIFIER_VERSION;
        use evanalyzer_cfg::settings::ai_learning_object_settings::AiLearningObjectFeatureSettings;
        use evanalyzer_cfg::settings::ai_learning_settings::{
            AiLearningBackendSettings, AiLearningClassifierSettings, AiLearningSettings,
            RandomForestSettings,
        };
        use evanalyzer_cfg::settings::meta_data::MetaData;

        let rows = vec![vec![0.0], vec![1.0]];
        let labels = [0usize, 1];
        let classifier = crate::ai_learning::model::random_forest::fit_random_forest(
            &rows,
            &labels,
            &RandomForestSettings::default(),
        )
        .unwrap();
        SavedClassifier {
            version: CURRENT_SAVED_CLASSIFIER_VERSION,
            classifier,
            settings: AiLearningSettings {
                schema_version: evanalyzer_cfg::CURRENT_AI_LEARNING_SETTINGS_SCHEMA_VERSION,
                meta: MetaData::default(),
                backend: AiLearningBackendSettings::RandomForest(RandomForestSettings::default()),
                classifier: AiLearningClassifierSettings::Object {
                    feature_spec: AiLearningObjectFeatureSettings { metrics: vec![] },
                    class_labels: vec![],
                },
            },
        }
    }

    #[test]
    fn load_cached_classifier_reuses_the_same_arc_on_a_repeated_path() {
        let path = PathBuf::from("/models/does-not-need-to-exist.evamodel");
        let load_calls = Cell::new(0);

        let first = load_cached_classifier(&path, || {
            load_calls.set(load_calls.get() + 1);
            Ok::<_, InternalErrors>(sample_classifier())
        })
        .unwrap();
        let second = load_cached_classifier(&path, || {
            load_calls.set(load_calls.get() + 1);
            Ok::<_, InternalErrors>(sample_classifier())
        })
        .unwrap();

        assert_eq!(load_calls.get(), 1, "second call must hit the cache");
        assert!(Arc::ptr_eq(&first, &second));
    }
}
