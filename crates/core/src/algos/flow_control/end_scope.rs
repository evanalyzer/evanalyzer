//! Ends a component instance's scope.

use crate::algos::{ExecutionScope, ImageAlgorithm};
use crate::object::Object;
use crate::pipeline::{
    object_cache::ObjectCache,
    pipeline_cache::{CacheAddress, GlobalPipelineCache},
    pipeline_context::PipelineContext,
    scope::Scope,
};
use evanalyzer_cfg::core_types::{CitationMetadata, InternalErrors, ObjectId};

/// Ends a component instance's scope - added by the job generator right
/// after the instance's last command, in the phase that command runs in (its
/// tile commands run before every whole-image command). Frees the scope's
/// objects and the references to them, and its memory slots
/// (`MemoryId::ComponentContext`), so nothing internal reaches the results and the
/// memory is free for the rest of the run.
///
/// `pub(crate)` - internal for now: a fully `pub` command here would become a
/// pipeline command the user can add.
pub(crate) struct EndScope {
    pub scope: Scope,
    /// The phase of the instance's last command.
    pub phase: ExecutionScope,
}

impl ImageAlgorithm for EndScope {
    fn execute(
        &self,
        _ctx: &mut PipelineContext,
        cache: &mut GlobalPipelineCache,
    ) -> Result<(), InternalErrors> {
        let removed = remove_scope(&mut cache.object_cache, self.scope);
        if let Scope::Scoped(scope_id) = self.scope {
            cache.image_cache.retain(
                |address| !matches!(address, CacheAddress::ComponentMemory(id, _) if *id == scope_id),
            );
        }
        log::debug!("End of {:?}: {removed} objects removed", self.scope);
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
/// objects hold to them - parent, children, colocalization partners, track
/// members. Returns how many objects were removed.
///
/// Two passes over the cache, no index: one removes, one cleans up. Most
/// objects hold no references, so the second pass is a few cheap checks per
/// object. The removed ids come out of the (ordered) cache sorted, so
/// looking one up is a binary search - no hashing.
fn remove_scope(cache: &mut ObjectCache, scope: Scope) -> usize {
    let to_remove = cache
        .values()
        .filter(|object| object.scope() == scope)
        .count();
    if to_remove == 0 {
        return 0;
    }
    let mut removed_ids: Vec<ObjectId> = Vec::with_capacity(to_remove);
    let mut keep = |id: &ObjectId, object: &Object| {
        let keep = object.scope() != scope;
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
        if object
            .parent_id
            .as_ref()
            .is_some_and(|parent| removed.contains(parent))
        {
            object.parent_id = None;
        }
        if !object.children.is_empty() {
            object.children.retain(|child| !removed.contains(child));
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

    const COMPONENT: Scope = Scope::Scoped(ScopeId(1));
    const OTHER_COMPONENT: Scope = Scope::Scoped(ScopeId(2));

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
            object(1, ObjectClass::Local(1), COMPONENT),
            object(2, ObjectClass::Local(1), OTHER_COMPONENT),
            object(3, ObjectClass::Global(1), COMPONENT),
        ]);
        assert_eq!(remove_scope(&mut cache, COMPONENT), 1);
        assert!(cache.get(&ObjectId(1)).is_none());
        assert!(cache.get(&ObjectId(2)).is_some(), "another instance's");
        assert!(cache.get(&ObjectId(3)).is_some(), "the project's");
        assert_eq!(remove_scope(&mut cache, COMPONENT), 0, "nothing left");
    }

    #[test]
    fn references_to_removed_objects_are_dropped() {
        let seed = object(1, ObjectClass::Local(1), COMPONENT);
        let mut cell = object(2, ObjectClass::Global(1), COMPONENT);
        let kept = object(3, ObjectClass::Global(2), Scope::Global);
        cell.parent_id = Some(ObjectId(1));
        cell.children = vec![ObjectId(1), ObjectId(3)];
        cell.colocalized_with
            .insert(ObjectClass::Local(1), vec![ObjectId(1)]);
        cell.colocalized_with
            .insert(ObjectClass::Global(2), vec![ObjectId(1), ObjectId(3)]);
        // A recorded "no partner" stays.
        cell.colocalized_with.insert(ObjectClass::Global(3), vec![]);
        cell.track.object_ids = vec![ObjectId(1), ObjectId(2)];
        let mut cache = cache(vec![seed, cell, kept]);

        remove_scope(&mut cache, COMPONENT);

        let cell = cache.get(&ObjectId(2)).unwrap();
        assert_eq!(cell.parent_id, None);
        assert_eq!(cell.children, [ObjectId(3)]);
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
    /// `cargo test --release -p evanalyzer_core --features ai --lib -- --ignored bench_end_scope --nocapture`.
    #[test]
    #[ignore]
    fn bench_end_scope() {
        for n in [1_000_000u128, 5_000_000] {
            let rss_start = rss_mb();
            let build = || -> ObjectCache {
                (0..n)
                    .map(|i| {
                        let in_component = i % 2 == 0;
                        let mut o = object(
                            i + 1,
                            if in_component {
                                ObjectClass::Local(1)
                            } else {
                                ObjectClass::Global(1)
                            },
                            COMPONENT,
                        );
                        if !in_component && i % 10 == 1 {
                            o.parent_id = Some(ObjectId(i));
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
            plain.retain(|_, o| o.scope() != COMPONENT);
            let drop_only = start.elapsed();
            drop(plain);

            let mut cache = build();
            let rss_full = rss_mb();
            let start = std::time::Instant::now();
            let removed = remove_scope(&mut cache, COMPONENT);
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
        let own = CacheAddress::ComponentMemory(ScopeId(1), 1);
        let other = CacheAddress::ComponentMemory(ScopeId(2), 1);
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
            scope: COMPONENT,
            phase: ExecutionScope::Tile,
        }
        .execute(&mut ctx, &mut cache)
        .unwrap();

        assert!(!cache.image_cache.contains_key(&own));
        assert!(cache.image_cache.contains_key(&other), "another instance's");
        assert!(cache.image_cache.contains_key(&pipeline_slot));
    }
}
