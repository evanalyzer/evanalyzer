//! # model_cache
//!
//! **Author:** Joachim Danmayr
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **AGPL-3.0**.

use std::{cell::RefCell, collections::HashMap, path::Path, path::PathBuf, sync::Arc};
use tch::CModule;

/// Looks up `path` in `cache`, inserting the result of `load` on a miss.
///
/// Kept generic (and free of any `tch`/thread-local dependency) so the cache
/// semantics — hit reuses without calling `load`, miss calls `load` exactly
/// once and stores it — can be unit tested directly.
fn get_or_insert<V, E>(
    cache: &mut HashMap<PathBuf, Arc<V>>,
    path: &Path,
    load: impl FnOnce() -> Result<V, E>,
) -> Result<Arc<V>, E> {
    if let Some(cached) = cache.get(path) {
        return Ok(Arc::clone(cached));
    }
    let value = Arc::new(load()?);
    cache.insert(path.to_path_buf(), Arc::clone(&value));
    Ok(value)
}

thread_local! {
    // Rayon reuses a fixed pool of worker threads across tasks, so a
    // thread-local cache persists across every tile/image a given worker
    // processes for the lifetime of the process — not just for one
    // PipelineCache/pipeline run. CModule is loaded and only ever touched by
    // the thread that loaded it, so no Send/Sync bound on CModule is needed.
    static MODEL_CACHE: RefCell<HashMap<PathBuf, Arc<CModule>>> = RefCell::new(HashMap::new());
}

/// Returns the `CModule` loaded from `path`, reusing a previously loaded
/// instance on this thread instead of re-reading and re-parsing the
/// TorchScript file. `load` is only invoked on a cache miss.
///
/// `load_cached_model` is a thin delegation to [`get_or_insert`], whose
/// hit/miss/error semantics are covered directly below; no TorchScript
/// fixture is available in this environment to exercise a real `CModule`
/// load in a unit test.
pub fn load_cached_model<E>(
    path: &Path,
    load: impl FnOnce() -> Result<CModule, E>,
) -> Result<Arc<CModule>, E> {
    MODEL_CACHE.with(|cache| get_or_insert(&mut cache.borrow_mut(), path, load))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;

    #[test]
    fn cache_hit_reuses_value_without_calling_loader_again() {
        let mut cache: HashMap<PathBuf, Arc<u32>> = HashMap::new();
        let path = PathBuf::from("/models/one.pt");
        let load_calls = Cell::new(0);

        let first =
            get_or_insert::<u32, ()>(&mut cache, &path, || {
                load_calls.set(load_calls.get() + 1);
                Ok(42)
            })
            .unwrap();
        let second =
            get_or_insert::<u32, ()>(&mut cache, &path, || {
                load_calls.set(load_calls.get() + 1);
                Ok(0) // Would prove staleness if this ever won.
            })
            .unwrap();

        assert_eq!(load_calls.get(), 1, "loader must run exactly once for a repeated path");
        assert_eq!(*first, 42);
        assert_eq!(*second, 42);
        assert!(Arc::ptr_eq(&first, &second), "second call must return the same cached Arc");
    }

    #[test]
    fn different_paths_are_cached_independently() {
        let mut cache: HashMap<PathBuf, Arc<u32>> = HashMap::new();
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

        assert_eq!(load_calls.get(), 2, "a different path must trigger its own load");
        assert_eq!(*a, 1);
        assert_eq!(*b, 2);
    }

    #[test]
    fn loader_error_is_not_cached() {
        let mut cache: HashMap<PathBuf, Arc<u32>> = HashMap::new();
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

        assert_eq!(load_calls.get(), 2, "a failed load must not poison the cache entry");
        assert_eq!(*second, 7);
    }
}
