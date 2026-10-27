//! # classifier_cache
//!
//! **Author:** Joachim Danmayr
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.
//!
//! Trained classifiers (`.evamodel`) loaded once and shared by all worker
//! threads, while the file is unchanged - and only while a pipeline runs:
//! when the last running job ends ([`ClassifierCacheScope`]), they are
//! dropped.
//!
//! Was one copy per worker thread: a run on 8 threads read the model 8
//! times, all at its start, and kept 8 copies.

use std::{
    collections::HashMap,
    path::{Path, PathBuf},
    sync::{Arc, LazyLock, Mutex, PoisonError},
    time::SystemTime,
};

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
/// Kept generic (and free of the thread-local) so the cache semantics can
/// be unit tested directly.
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

/// The loaded models and how many jobs run.
#[derive(Default)]
struct Cache {
    models: HashMap<PathBuf, (Option<SystemTime>, Arc<SavedClassifier>)>,
    runs: usize,
}

static CLASSIFIER_CACHE: LazyLock<Mutex<Cache>> = LazyLock::new(Mutex::default);

fn cache() -> std::sync::MutexGuard<'static, Cache> {
    CLASSIFIER_CACHE
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
}

/// Returns the [`SavedClassifier`] loaded from `path`, reusing the instance
/// any thread loaded before instead of re-reading and re-deserializing the
/// model file. `load` is only invoked on a cache miss - while it runs, the
/// other threads wait for its result instead of loading it too.
pub fn load_cached_classifier<E>(
    path: &Path,
    load: impl FnOnce() -> Result<SavedClassifier, E>,
) -> Result<Arc<SavedClassifier>, E> {
    get_or_insert(&mut cache().models, path, load)
}

/// Keeps the loaded models while a job runs (`JobExecutor::run`); when the
/// last running job's scope ends, they are dropped - a model is loaded once
/// per run, not kept in RAM between runs.
pub struct ClassifierCacheScope(());

impl ClassifierCacheScope {
    pub fn enter() -> Self {
        cache().runs += 1;
        Self(())
    }
}

impl Drop for ClassifierCacheScope {
    fn drop(&mut self) {
        let mut cache = cache();
        cache.runs = cache.runs.saturating_sub(1);
        if cache.runs == 0 {
            cache.models.clear();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use evanalyzer_cfg::core_types::InternalErrors;
    use std::cell::Cell;
    use std::time::Duration;

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

    // -- load_cached_classifier (real end-to-end -
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
                    feature_spec: AiLearningObjectFeatureSettings::default(),
                    class_labels: vec![],
                },
            },
        }
    }

    /// RAM and time of a trained model in a pipeline run: saved to disk,
    /// then 8 worker threads load it through this cache and predict 16
    /// tiles with certainty (512² pixels, 20 features). `MODEL=forest`
    /// (50 trees, depth 20, 200 000 samples, out-of-bag on - the panel's
    /// old default) or `MODEL=mlp` (64, 32). Run one model per process:
    /// `MODEL=forest cargo test --release -p evanalyzer_core --lib
    /// bench_trained_model_in_a_run -- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn bench_trained_model_in_a_run() {
        use crate::ai_learning::model::{load_from_file, save_to_file};
        use evanalyzer_cfg::settings::ai_learning_settings::{MlpSettings, RandomForestSettings};
        use rayon::prelude::*;
        let mb = |key: &str| {
            std::fs::read_to_string("/proc/self/status")
                .ok()
                .and_then(|s| {
                    s.lines()
                        .find(|l| l.starts_with(key))
                        .and_then(|l| l.split_whitespace().nth(1)?.parse::<u64>().ok())
                })
                .unwrap_or(0)
                / 1024
        };
        let mut seed = 0x9E37_79B9_7F4A_7C15u64;
        let mut next = move || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            (seed >> 40) as f32 / (1u64 << 24) as f32
        };
        let mut sample = |class: usize| -> Vec<f32> {
            (0..20)
                .map(|f| next() + if f % 4 == 0 { class as f32 * 0.5 } else { 0.0 })
                .collect()
        };
        let forest = std::env::var("MODEL").as_deref() != Ok("mlp");
        let n = if forest { 200_000 } else { 20_000 };
        let labels: Vec<usize> = (0..n).map(|i| i % 3).collect();
        let rows: Vec<Vec<f32>> = labels.iter().map(|&c| sample(c)).collect();
        let tile: Vec<Vec<f32>> = (0..512 * 512).map(|i| sample(i % 3)).collect();
        let classifier = if forest {
            crate::ai_learning::model::random_forest::fit_random_forest(
                &rows,
                &labels,
                &RandomForestSettings {
                    n_trees: 50,
                    max_depth: Some(20),
                    min_samples_split: 2,
                    min_samples_leaf: 1,
                    keep_samples: true,
                    seed: 1,
                    ..Default::default()
                },
            )
            .unwrap()
        } else {
            let (tx, _rx) = std::sync::mpsc::channel();
            crate::ai_learning::model::mlp::fit_mlp(
                &rows,
                &labels,
                3,
                &MlpSettings {
                    hidden_layers: vec![64, 32],
                    epochs: 5,
                    learning_rate: 0.001,
                    batch_size: 32,
                    seed: 1,
                    ..Default::default()
                },
                &tx,
                &Arc::new(std::sync::atomic::AtomicBool::new(false)),
            )
            .unwrap()
            .0
        };
        let mut saved = sample_classifier();
        saved.classifier = classifier;
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("bench.evamodel");
        save_to_file(&saved, &path).unwrap();
        drop((saved, rows, labels));
        let file_mb = std::fs::metadata(&path).unwrap().len() as f64 / 1_048_576.0;

        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(8)
            .build()
            .unwrap();
        let (rss, peak) = (mb("VmRSS:"), mb("VmHWM:"));
        let start = std::time::Instant::now();
        pool.install(|| {
            (0..16).into_par_iter().for_each(|_| {
                let model = load_cached_classifier(&path, || load_from_file(&path)).unwrap();
                let (classes, _) = model.classifier.predict_with_certainty(&tile).unwrap();
                assert_eq!(classes.len(), tile.len());
            })
        });
        println!(
            "{}: file {file_mb:.1} MB; 16 tiles on 8 threads {:.2} s; RSS {rss} -> {} MB, peak {peak} -> {} MB (+{} MB)",
            if forest { "forest" } else { "mlp" },
            start.elapsed().as_secs_f64(),
            mb("VmRSS:"),
            mb("VmHWM:"),
            mb("VmHWM:").saturating_sub(peak)
        );
    }

    #[test]
    fn load_cached_classifier_reuses_the_same_arc_on_a_repeated_path() {
        // A job ending in a parallel test must not empty the cache between
        // the two calls.
        let _scope = ClassifierCacheScope::enter();
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

    /// All threads share one load; the models go when the last run ends,
    /// not while another one still runs.
    #[test]
    fn threads_share_one_load_and_the_last_run_drops_it() {
        use rayon::prelude::*;
        let path = PathBuf::from("/models/shared-by-threads.evamodel");
        let loads = std::sync::atomic::AtomicUsize::new(0);
        let load = || {
            load_cached_classifier(&path, || {
                loads.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                Ok::<_, InternalErrors>(sample_classifier())
            })
            .unwrap()
        };
        let first_run = ClassifierCacheScope::enter();
        let second_run = ClassifierCacheScope::enter();
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .unwrap();
        let models: Vec<_> = pool.install(|| (0..16).into_par_iter().map(|_| load()).collect());
        assert_eq!(loads.load(std::sync::atomic::Ordering::SeqCst), 1);
        assert!(models.iter().all(|m| Arc::ptr_eq(m, &models[0])));

        let held = Arc::downgrade(&models[0]);
        drop(models);
        drop(first_run);
        assert!(held.upgrade().is_some(), "another run still runs");
        drop(second_run);
        // Unless a parallel test's run still runs: then it goes with that.
        if cache().runs == 0 {
            assert!(held.upgrade().is_none(), "the last run dropped it");
        }
    }
}
