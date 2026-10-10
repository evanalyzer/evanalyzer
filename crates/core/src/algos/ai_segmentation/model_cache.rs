//! # model_cache
//!
//! **Author:** Joachim Danmayr
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **AGPL-3.0**.

use std::{
    cell::RefCell,
    collections::HashMap,
    path::Path,
    path::PathBuf,
    sync::{Arc, LazyLock, Mutex, MutexGuard, PoisonError},
    time::SystemTime,
};
use tch::{CModule, Device};

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

/// TorchScript models loaded in this process, per device and file.
type ModelCache = HashMap<Device, HashMap<PathBuf, (Option<SystemTime>, Arc<CModule>)>>;

/// All Torch work in the process goes through this lock (see
/// [`torch_session`]); it also owns the loaded models.
///
/// Why one lock instead of letting every pipeline worker run its model:
/// - Each worker used to load its own copy of the model. With ~20 workers
///   that is ~20 copies on the GPU (a Cellpose-SAM model is ~1.2 GB),
///   which overflows GPU memory - on Windows the driver then silently pages
///   into system RAM and inference all but stops.
/// - Torch runs every CPU op on its own thread pool sized to all cores. ~20
///   workers doing that at once start hundreds of busy threads fighting
///   over the cores (100% CPU, little progress).
///
/// A GPU runs one model at a time anyway, so serializing inference costs
/// little; the lock-free parts of the AI steps (mask building, rasterizing)
/// and all other steps still run in parallel.
static TORCH: LazyLock<Mutex<ModelCache>> = LazyLock::new(|| Mutex::new(HashMap::new()));

/// Exclusive access to Torch for one AI step. Hold it for all tensor work -
/// loading, inference and copying results back to plain Rust data - and drop
/// it (with the tensors) before the remaining CPU-only work.
pub(crate) struct TorchSession {
    models: MutexGuard<'static, ModelCache>,
}

/// Waits for and returns exclusive access to Torch, see [`TORCH`].
pub(crate) fn torch_session() -> TorchSession {
    // A panic inside an earlier session (e.g. a failing model) must not
    // block every later AI step; the model map stays consistent either way.
    let models = TORCH.lock().unwrap_or_else(PoisonError::into_inner);
    TorchSession { models }
}

impl TorchSession {
    /// The model loaded from `path` onto `device`, shared by every worker:
    /// loaded once per process instead of re-read and re-parsed, and only
    /// one copy kept on the device. `load` only runs on a cache miss (or
    /// when the file changed, see [`get_or_insert`]).
    pub fn model<E>(
        &mut self,
        path: &Path,
        device: Device,
        load: impl FnOnce() -> Result<CModule, E>,
    ) -> Result<Arc<CModule>, E> {
        // Some traced graphs (e.g. Cellpose-SAM's ViT encoder, whose relative-
        // position-embedding math is a long chain of elementwise adds/unsqueezes)
        // contain op sequences PyTorch's JIT fuser tries to compile into a single
        // CUDA kernel via NVRTC on first run. That requires the CUDA *toolkit*'s
        // `libnvrtc-builtins` to be installed system-wide - most end-user
        // machines that only have a GPU driver installed don't have it, and the
        // failure only surfaces the first time that particular fused op pattern
        // runs. Disabling both JIT fusers trades that (unfused, marginally
        // slower) elementwise math for never depending on NVRTC being present.
        // Cheap enough to set unconditionally on every call - whether this flag
        // is process-global or per-thread isn't documented, and redoing it on a
        // cache hit costs nothing.
        tch::jit::set_tensor_expr_fuser_enabled(false);
        tch::jit::fuser_cuda_set_enabled(false);

        get_or_insert(self.models.entry(device).or_default(), path, load)
    }
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
    use evanalyzer_cfg::core_types::InternalErrors;
    use std::cell::Cell;
    use std::time::Duration;

    #[test]
    fn workers_share_one_loaded_model() {
        // The point of the shared session: many pipeline workers asking for
        // the same model get one copy, loaded once - not one per thread.
        use crate::algos::ai_segmentation::test_support::trace_and_save_model;
        use std::sync::atomic::{AtomicUsize, Ordering};
        let (_dir, path) = trace_and_save_model(1, 1, 2, |x| x.shallow_clone());
        let loads = AtomicUsize::new(0);
        let models: Vec<Arc<CModule>> = std::thread::scope(|scope| {
            let handles: Vec<_> = (0..4)
                .map(|_| {
                    scope.spawn(|| {
                        torch_session()
                            .model(&path, Device::Cpu, || {
                                loads.fetch_add(1, Ordering::SeqCst);
                                CModule::load_on_device(&path, Device::Cpu)
                            })
                            .unwrap()
                    })
                })
                .collect();
            handles.into_iter().map(|h| h.join().unwrap()).collect()
        });
        assert_eq!(loads.load(Ordering::SeqCst), 1);
        assert!(models.iter().all(|m| Arc::ptr_eq(m, &models[0])));
    }

    #[test]
    fn a_panic_inside_a_session_does_not_block_later_sessions() {
        let _ = std::thread::spawn(|| {
            let _session = torch_session();
            panic!("model failed");
        })
        .join();
        drop(torch_session());
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
