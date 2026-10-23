//! Ends a module use: its scope, or - nested in another module - its
//! own classes and memory slots in that one's scope.

use crate::algos::{ExecutionScope, ImageAlgorithm};
use crate::job::modules::UseEnd;
use crate::object::Object;
use crate::pipeline::{
    object_cache::ObjectCache,
    pipeline_cache::{CacheAddress, GlobalPipelineCache},
    pipeline_context::PipelineContext,
    scope::Scope,
};
use evanalyzer_cfg::core_types::{CitationMetadata, InternalErrors, ObjectClass, ObjectId};

/// Ends a module use - added by the job generator right after the use's
/// last command, in the phase that command runs in (its tile commands run
/// before every whole-image command). Frees what [`UseEnd`] says - objects
/// and the references to them, memory slots (`MemoryId::ModuleContext`) -
/// so nothing internal reaches the results and the memory is free for the
/// rest of the run.
///
/// `pub(crate)` - internal for now: a fully `pub` command here would become a
/// pipeline command the user can add.
pub(crate) struct EndScope {
    pub end: UseEnd,
    /// The phase of the use's last command.
    pub phase: ExecutionScope,
}

impl ImageAlgorithm for EndScope {
    fn execute(
        &self,
        _ctx: &mut PipelineContext,
        cache: &mut GlobalPipelineCache,
    ) -> Result<(), InternalErrors> {
        match &self.end {
            UseEnd::Scope(scope) => {
                let removed = remove_scope(&mut cache.object_cache, *scope);
                if let Scope::Scoped(scope_id) = scope {
                    cache.image_cache.retain(|address| {
                        !matches!(address, CacheAddress::ModuleMemory(id, _) if id == scope_id)
                    });
                }
                log::debug!("End of {scope:?}: {removed} objects removed");
            }
            UseEnd::Nested {
                scope,
                classes,
                slots,
            } => {
                let removed = remove_classes(&mut cache.object_cache, *scope, classes);
                if let Scope::Scoped(scope_id) = scope {
                    cache.image_cache.retain(|address| {
                        !matches!(address, CacheAddress::ModuleMemory(id, slot)
                            if id == scope_id && slots.contains(slot))
                    });
                }
                log::debug!("End of a nested use of {scope:?}: {removed} objects removed");
            }
        }
        Ok(())
    }

    fn name(&self) -> &'static str {
        "End Scope"
    }

    fn cite(&self) -> Vec<&'static CitationMetadata> {
        vec![]
    }

    fn execution_scope(&self) -> ExecutionScope {
        self.phase
    }

    /// Only objects are touched - no scratch image needed.
    fn scratch_is_workspace(&self) -> bool {
        false
    }
}

/// Removes `scope`'s objects from `cache` and every reference the remaining
/// objects hold to them - parents, colocalization partners, track
/// members. Returns how many objects were removed.
///
/// Two passes over the cache, no index: one removes, one cleans up. Most
/// objects hold no references, so the second pass is a few cheap checks per
/// object. The removed ids come out of the (ordered) cache sorted, so
/// looking one up is a binary search - no hashing.
fn remove_scope(cache: &mut ObjectCache, scope: Scope) -> usize {
    remove_where(cache, |object| object.scope() == scope)
}

/// Ends a nested module use's classes `classes` in `scope`: objects with
/// only those classes are removed (see [`remove_scope`]), the others - the
/// module around it's, given one of them too - lose them. Returns how many
/// objects were removed.
fn remove_classes(cache: &mut ObjectCache, scope: Scope, classes: &[ObjectClass]) -> usize {
    let own = |class: &ObjectClass| classes.contains(class);
    for object in cache.values_mut() {
        if object.scope() == scope
            && object.object_classes().iter().any(own)
            && !object.object_classes().iter().all(own)
        {
            for class in classes {
                object.remove_object_class(class, scope);
            }
        }
    }
    remove_where(cache, |object| {
        object.scope() == scope
            && !object.object_classes().is_empty()
            && object.object_classes().iter().all(own)
    })
}

/// Removes the objects `remove` picks from `cache` and every reference the
/// remaining objects hold to them. Returns how many objects were removed.
pub(crate) fn remove_where(cache: &mut ObjectCache, remove: impl Fn(&Object) -> bool) -> usize {
    let to_remove = cache.values().filter(|object| remove(object)).count();
    if to_remove == 0 {
        return 0;
    }
    let mut removed_ids: Vec<ObjectId> = Vec::with_capacity(to_remove);
    let mut keep = |id: &ObjectId, object: &Object| {
        let keep = !remove(object);
        if !keep {
            removed_ids.push(id.clone());
        }
        keep
    };
    if to_remove * REBUILD_FRACTION >= cache.len() {
        // Rebuilt rather than `retain`ed: the cache holds the objects inline
        // in a `BTreeMap`, and removing many leaves half-empty nodes that new
        // objects (whose ids sort after every existing one) never fill again.
        // Built from a sorted iterator, the new map is compact.
        *cache = std::mem::take(cache)
            .into_iter()
            .filter(|(id, object)| keep(id, object))
            .collect();
    } else {
        cache.retain(|id, object| keep(id, object));
    }
    debug_assert!(removed_ids.is_sorted());
    let removed = SortedIds(&removed_ids);

    for object in cache.values_mut() {
        if !object.parents().is_empty() {
            object.retain_parents(|_, parent| !removed.contains(parent));
        }
        if !object.colocalized_with.is_empty() {
            // An entry emptied here pointed only at removed objects; one that
            // was empty before stays (a recorded "no partner").
            object.colocalized_with.retain(|_, partners| {
                let before = partners.len();
                partners.retain(|partner| !removed.contains(partner));
                before == 0 || !partners.is_empty()
            });
        }
        if !object.track.object_ids.is_empty() {
            object
                .track
                .object_ids
                .retain(|member| !removed.contains(member));
        }
    }
    removed_ids.len()
}

/// [`remove_scope`] rebuilds the cache when at least 1 / this of its objects
/// go - measured: rebuilding a million objects with half of them removed
/// takes ~0.2 s instead of ~0.1 s, and frees ~240 MB of the map's nodes
/// instead of none.
const REBUILD_FRACTION: usize = 4;

/// Ids in ascending order, looked up by binary search.
struct SortedIds<'a>(&'a [ObjectId]);

impl SortedIds<'_> {
    fn contains(&self, id: &ObjectId) -> bool {
        self.0.binary_search(id).is_ok()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::object::ObjectInit;
    use crate::pipeline::scope::ScopeId;
    use evanalyzer_cfg::core_types::{ObjectClass, ObjectId};

    const MODULE: Scope = Scope::Scoped(ScopeId(1));
    const OTHER_MODULE: Scope = Scope::Scoped(ScopeId(2));

    fn object(id: u128, class: ObjectClass, scope: Scope) -> Object {
        let mut object = Object::new(ObjectInit {
            id: ObjectId(id),
            ..Default::default()
        });
        object.add_object_class(class, scope).unwrap();
        object
    }

    fn cache(objects: Vec<Object>) -> ObjectCache {
        objects.into_iter().map(|o| (o.id.clone(), o)).collect()
    }

    #[test]
    fn ending_a_scope_removes_only_its_objects() {
        let mut cache = cache(vec![
            object(1, ObjectClass::Local(1), MODULE),
            object(2, ObjectClass::Local(1), OTHER_MODULE),
            object(3, ObjectClass::Global(1), MODULE),
        ]);
        assert_eq!(remove_scope(&mut cache, MODULE), 1);
        assert!(cache.get(&ObjectId(1)).is_none());
        assert!(cache.get(&ObjectId(2)).is_some(), "another instance's");
        assert!(cache.get(&ObjectId(3)).is_some(), "the project's");
        assert_eq!(remove_scope(&mut cache, MODULE), 0, "nothing left");
    }

    #[test]
    fn references_to_removed_objects_are_dropped() {
        let seed = object(1, ObjectClass::Local(1), MODULE);
        let mut cell = object(2, ObjectClass::Global(1), MODULE);
        let kept = object(3, ObjectClass::Global(2), Scope::Global);
        cell.set_parent(ObjectClass::Local(1), Some(ObjectId(1)));
        cell.set_parent(ObjectClass::Global(2), Some(ObjectId(3)));
        cell.colocalized_with
            .insert(ObjectClass::Local(1), vec![ObjectId(1)]);
        cell.colocalized_with
            .insert(ObjectClass::Global(2), vec![ObjectId(1), ObjectId(3)]);
        // A recorded "no partner" stays.
        cell.colocalized_with.insert(ObjectClass::Global(3), vec![]);
        cell.track.object_ids = vec![ObjectId(1), ObjectId(2)];
        let mut cache = cache(vec![seed, cell, kept]);

        remove_scope(&mut cache, MODULE);

        let cell = cache.get(&ObjectId(2)).unwrap();
        let parents: Vec<_> = cell.parents().iter().collect();
        assert_eq!(parents, [(&ObjectClass::Global(2), &ObjectId(3))]);
        assert!(!cell.colocalized_with.contains_key(&ObjectClass::Local(1)));
        assert_eq!(
            cell.colocalized_with[&ObjectClass::Global(2)],
            [ObjectId(3)]
        );
        assert!(cell.colocalized_with[&ObjectClass::Global(3)].is_empty());
        assert_eq!(cell.track.object_ids, [ObjectId(2)]);
    }

    fn rss_mb() -> f64 {
        let statm = std::fs::read_to_string("/proc/self/statm").unwrap_or_default();
        let pages: f64 = statm
            .split_whitespace()
            .nth(1)
            .and_then(|v| v.parse().ok())
            .unwrap_or(0.0);
        pages * 4096.0 / 1e6
    }

    /// Speed and RAM of `remove_scope` with millions of objects: half of them
    /// in the ending scope, every tenth remaining one pointing at one of
    /// them (parent + coloc partner). Run with
    /// `cargo test --release -p evanalyzer_core --lib -- --ignored bench_end_scope --nocapture`.
    #[test]
    #[ignore]
    fn bench_end_scope() {
        for n in [1_000_000u128, 5_000_000] {
            let rss_start = rss_mb();
            let build = || -> ObjectCache {
                (0..n)
                    .map(|i| {
                        let in_module = i % 2 == 0;
                        let mut o = object(
                            i + 1,
                            if in_module {
                                ObjectClass::Local(1)
                            } else {
                                ObjectClass::Global(1)
                            },
                            MODULE,
                        );
                        if !in_module && i % 10 == 1 {
                            o.set_parent(ObjectClass::Local(1), Some(ObjectId(i)));
                            o.colocalized_with
                                .insert(ObjectClass::Local(1), vec![ObjectId(i)]);
                        }
                        (o.id.clone(), o)
                    })
                    .collect()
            };

            // Reference: only dropping the same objects.
            let mut plain = build();
            let start = std::time::Instant::now();
            plain.retain(|_, o| o.scope() != MODULE);
            let drop_only = start.elapsed();
            drop(plain);

            let mut cache = build();
            let rss_full = rss_mb();
            let start = std::time::Instant::now();
            let removed = remove_scope(&mut cache, MODULE);
            let elapsed = start.elapsed();
            let rss_after = rss_mb();
            println!(
                "{n} objects: end_scope removed {removed} in {:.0} ms | RSS {:.0} MB -> {:.0} MB \
                 (cache {:.0} MB before, {:.0} MB freed)",
                elapsed.as_secs_f64() * 1000.0,
                rss_full,
                rss_after,
                rss_full - rss_start,
                rss_full - rss_after
            );
            assert_eq!(removed as u128, n / 2);
            println!(
                "    of which dropping the objects alone: {:.0} ms",
                drop_only.as_secs_f64() * 1000.0
            );
            println!(
                "    size_of::<Object>() = {} bytes",
                std::mem::size_of::<Object>()
            );
            // The freed memory is reused: as many new objects as were removed,
            // added to the cache like a later command would.
            for i in 0..n / 2 {
                let o = object(n + i + 1, ObjectClass::Global(1), Scope::Global);
                cache.insert(o.id.clone(), o);
            }
            println!(
                "    RSS after adding {} new objects to the cache: {:.0} MB",
                n / 2,
                rss_mb()
            );
            drop(cache);
        }
    }

    #[test]
    fn ending_a_scope_frees_its_memory_slots_only() {
        use crate::image::ImageContainer;
        use evanalyzer_cfg::core_types::MemoryId;
        let image = || {
            std::sync::Arc::new(ImageContainer::new_f32_gray_from_image_test(
                kornia_image::Image::new(
                    kornia_image::ImageSize {
                        width: 1,
                        height: 1,
                    },
                    vec![0.0],
                )
                .unwrap(),
            ))
        };
        let mut cache = GlobalPipelineCache::default();
        let own = CacheAddress::ModuleMemory(ScopeId(1), 1);
        let other = CacheAddress::ModuleMemory(ScopeId(2), 1);
        let pipeline_slot = CacheAddress::Memory(MemoryId::PipelineContext(1));
        for address in [own, other, pipeline_slot] {
            cache.image_cache.insert(address, image());
        }
        let mut ctx = crate::pipeline::pipeline_context::PipelineContext::new_from_image_test(
            kornia_image::Image::new(
                kornia_image::ImageSize {
                    width: 1,
                    height: 1,
                },
                vec![0.0],
            )
            .unwrap(),
        )
        .unwrap();

        EndScope {
            end: UseEnd::Scope(MODULE),
            phase: ExecutionScope::Tile,
        }
        .execute(&mut ctx, &mut cache)
        .unwrap();

        assert!(!cache.image_cache.contains_key(&own));
        assert!(cache.image_cache.contains_key(&other), "another instance's");
        assert!(cache.image_cache.contains_key(&pipeline_slot));

        // A nested use: only its own slots of the scope.
        let nested_own = CacheAddress::ModuleMemory(ScopeId(2), 5);
        cache.image_cache.insert(nested_own, image());
        EndScope {
            end: UseEnd::Nested {
                scope: OTHER_MODULE,
                classes: vec![],
                slots: vec![5],
            },
            phase: ExecutionScope::Tile,
        }
        .execute(&mut ctx, &mut cache)
        .unwrap();
        assert!(!cache.image_cache.contains_key(&nested_own));
        assert!(
            cache.image_cache.contains_key(&other),
            "the module around it's slot"
        );
    }

    #[test]
    fn ending_a_nested_use_removes_only_its_own_classes() {
        let inner = ObjectClass::Local(5);
        let mut both = object(3, ObjectClass::Local(1), MODULE);
        both.add_object_class(inner, MODULE).unwrap();
        let seed = object(1, inner, MODULE);
        let mut outer = object(2, ObjectClass::Local(1), MODULE);
        outer.set_parent(inner, Some(ObjectId(1)));
        let mut cache = cache(vec![
            seed,
            outer,
            both,
            object(4, inner, OTHER_MODULE),
            object(5, ObjectClass::Global(1), Scope::Global),
        ]);

        assert_eq!(remove_classes(&mut cache, MODULE, &[inner]), 1);
        assert!(
            cache.get(&ObjectId(1)).is_none(),
            "only the nested use's class"
        );
        assert!(
            cache.get(&ObjectId(2)).unwrap().parents().is_empty(),
            "references to it go too"
        );
        let both = cache.get(&ObjectId(3)).unwrap();
        assert_eq!(
            both.object_classes().iter().copied().collect::<Vec<_>>(),
            [ObjectClass::Local(1)],
            "the outer module's object keeps its own class"
        );
        assert!(cache.get(&ObjectId(4)).is_some(), "another use's scope");
        assert!(cache.get(&ObjectId(5)).is_some(), "the project's");
    }

    #[test]
    fn end_scope_runs_in_its_use_phase_and_needs_no_scratch_or_citation() {
        for phase in [ExecutionScope::Tile, ExecutionScope::WholeImage] {
            let end = EndScope {
                end: UseEnd::Scope(MODULE),
                phase,
            };
            assert_eq!(end.execution_scope(), phase);
            assert_eq!(end.name(), "End Scope");
            assert!(end.cite().is_empty(), "an internal step cites nothing");
            assert!(!end.scratch_is_workspace());
        }
    }

    /// Run through `execute`, a nested use's end removes its own class's
    /// objects from the object cache, not just its memory slots.
    #[test]
    fn executing_a_nested_use_end_removes_its_classes_objects() {
        let inner = ObjectClass::Local(5);
        let objects = cache(vec![
            object(1, inner, MODULE),
            object(2, ObjectClass::Local(1), MODULE),
        ]);
        let mut cache = GlobalPipelineCache::default();
        cache.object_cache = objects;
        let mut ctx = crate::pipeline::pipeline_context::PipelineContext::new_from_image_test(
            kornia_image::Image::new(
                kornia_image::ImageSize {
                    width: 1,
                    height: 1,
                },
                vec![0.0],
            )
            .unwrap(),
        )
        .unwrap();

        EndScope {
            end: UseEnd::Nested {
                scope: MODULE,
                classes: vec![inner],
                slots: vec![],
            },
            phase: ExecutionScope::WholeImage,
        }
        .execute(&mut ctx, &mut cache)
        .unwrap();

        assert!(cache.object_cache.get(&ObjectId(1)).is_none());
        assert!(cache.object_cache.get(&ObjectId(2)).is_some());
    }

    /// Few removed objects take the `retain` path instead of the rebuild;
    /// references to them are dropped all the same.
    #[test]
    fn removing_a_small_share_keeps_the_cache_and_still_drops_references() {
        let mut objects: Vec<Object> = (10..20)
            .map(|id| object(id, ObjectClass::Global(1), Scope::Global))
            .collect();
        objects[0].set_parent(ObjectClass::Local(1), Some(ObjectId(1)));
        objects[1].set_parent(ObjectClass::Global(1), Some(ObjectId(11)));
        objects.push(object(1, ObjectClass::Local(1), MODULE));
        let mut cache = cache(objects);
        assert!(
            REBUILD_FRACTION < cache.len(),
            "one removal must not rebuild"
        );

        assert_eq!(remove_scope(&mut cache, MODULE), 1);
        assert_eq!(cache.len(), 10);
        assert!(cache.get(&ObjectId(10)).unwrap().parents().is_empty());
        assert_eq!(
            cache
                .get(&ObjectId(11))
                .unwrap()
                .parent(&ObjectClass::Global(1)),
            Some(&ObjectId(11))
        );
    }
}
