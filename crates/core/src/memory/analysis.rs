//! # analysis
//!
//! The analysis' RAM: how many images/tiles run at once, and the image
//! caches' caps - from the RAM that is free when the run starts.
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-06-28
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.

use super::available_memory_bytes;

/// Bytes per pixel for the f32/u32 buffers used throughout a pipeline run
/// (the working image, scratch pad, segmentation map, instance map, and
/// cached channel planes are all one 4-byte value per pixel).
const BYTES_PER_PIXEL: u64 = 4;

/// Tile-sized working buffers one in-flight [`PipelineContext`](crate::pipeline::pipeline_context::PipelineContext)
/// holds at once: the working image, scratch pad, segmentation map and
/// instance map.
const WORKING_SET_BUFFERS: u64 = 4;

/// Object masks/metadata margin, as a fraction of one tile-sized plane -
/// covers a densely segmented tile's per-object bitmasks without needing to
/// actually run segmentation to know how many objects there will be. Purely
/// a heuristic guardrail, not a measured bound (same caveat as
/// [`RamBudget::total_bytes`] as a whole).
const OBJECT_CACHE_MARGIN_DIVISOR: u64 = 16;

/// If an image bigger than this size should be loaded it is splitted into tiles of this size
pub const MAX_TILE_SIZE: usize = 4096;

/// Peak per-worker RAM broken into the pieces that scale differently, so a
/// caller can size an `ImageCache` budget on top of what's already
/// accounted for (see [`recommended_image_cache_bytes`]) instead of only
/// getting one opaque total. Returned by [`estimate_ram_budget`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RamBudget {
    /// One in-flight [`PipelineContext`](crate::pipeline::pipeline_context::PipelineContext)'s
    /// own tile-sized buffers (working image, scratch pad, segmentation map,
    /// instance map) - the "double buffer" a worker is actively computing
    /// into. Independent of channel count: a pipeline step only ever has
    /// one *current* working image, regardless of how many channels the
    /// image has.
    pub working_set_bytes: u64,
    /// The smallest `ImageCache` hot-cache size that avoids the worst-case
    /// thrashing pattern for one tile: holding every channel plane
    /// (`channel_count` of them) for the tile currently being processed, so
    /// a pipeline step that reads back an earlier channel (e.g.
    /// colocalization) never forces a disk round trip for data from the
    /// *same* tile still in flight. [`recommended_image_cache_bytes`] never
    /// goes below this, even under RAM pressure - see its own doc comment
    /// for why.
    pub min_image_cache_bytes: u64,
    /// Heuristic margin for one tile's worth of object masks/metadata.
    pub object_cache_margin_bytes: u64,
}

impl RamBudget {
    /// The full per-worker estimate - what earlier versions of this module
    /// returned as a single `u64`.
    pub fn total_bytes(&self) -> u64 {
        self.working_set_bytes + self.min_image_cache_bytes + self.object_cache_margin_bytes
    }
}

/// Estimates the peak RAM one parallel worker can need to process one tile
/// of the image actually being analyzed, instead of guessing with a flat
/// constant - the two components that actually scale are the number of
/// pixels in a tile and how many channel planes a pipeline can have cached
/// at once (e.g. a colocalization step reading back an earlier channel's
/// classification). See [`RamBudget`]'s field docs for what each component
/// covers.
///
/// `tile_width`/`tile_height` should be the actual per-tile dimensions a
/// worker will process (the smaller of the image's own size and the
/// pipeline's max tile size), not necessarily the whole image - a worker
/// never holds more than one tile's worth of pixels per buffer regardless of
/// how large the source image is. Heuristic guardrail against over-committing
/// on low-RAM machines, not a measured per-pipeline bound.
pub fn estimate_ram_budget(
    tile_width: usize,
    tile_height: usize,
    channel_count: usize,
) -> RamBudget {
    let tile_pixels = tile_width as u64 * tile_height as u64;
    let plane_bytes = tile_pixels * BYTES_PER_PIXEL;

    RamBudget {
        working_set_bytes: plane_bytes * WORKING_SET_BUFFERS,
        min_image_cache_bytes: plane_bytes * channel_count.max(1) as u64,
        object_cache_margin_bytes: plane_bytes / OBJECT_CACHE_MARGIN_DIVISOR,
    }
}

/// Recommended `ImageCache` hot-cache capacity for each of `parallelism`
/// concurrently active workers (tiles are always processed through one
/// shared rayon pool of exactly this size - see `JobExecutor::run`'s own
/// doc comment - so this is genuinely how many `ImageCache` clones can be
/// resident at once, not just an upper bound).
///
/// Divides available RAM evenly across `parallelism` workers, gives each
/// share back what it needs for `non_cache_bytes_per_worker` (typically
/// [`RamBudget::working_set_bytes`] + [`RamBudget::object_cache_margin_bytes`]),
/// and hands the remainder to the cache - so `parallelism * (non_cache_bytes_per_worker
/// + image_cache_bytes) <= available RAM` holds, the same constraint
/// [`recommended_parallelism`] already enforces for `non_cache_bytes_per_worker`
/// alone, extended to also cover the cache. On a machine where cores (not
/// RAM) were the binding constraint on `parallelism`, this naturally grants
/// each worker a bigger cache than the bare minimum, for free.
///
/// Never goes below `min_image_cache_bytes` even if that means the bound
/// above doesn't hold on an extremely RAM-constrained machine: an
/// under-sized `ImageCache` doesn't just use less memory, it thrashes -
/// every pipeline step that reads back an earlier channel re-reads it from
/// disk - which measured at 15x the wall-clock time of a properly-sized
/// cache. A modest RAM overcommit is a far smaller risk than that.
pub fn recommended_image_cache_bytes(
    parallelism: usize,
    non_cache_bytes_per_worker: u64,
    min_image_cache_bytes: u64,
) -> u64 {
    let per_worker_share = available_memory_bytes() / (parallelism.max(1) as u64);
    per_worker_share
        .saturating_sub(non_cache_bytes_per_worker)
        .max(min_image_cache_bytes)
}

/// Cap for the one cache a tiled image's tiles are merged into: the RAM
/// left once every one of `parallelism` workers has its own tile's full
/// estimate ([`RamBudget::total_bytes`]: working set, its channel tiles,
/// object margin), less the results storage's own memory (`reserved_bytes`,
/// DuckDB's limit). Whatever doesn't fit goes to disk while the tiles still
/// run - not only at the end - so the merged channel tiles of a whole slide
/// can't grow past the RAM. See [`merged_cache_bytes`].
pub fn recommended_merged_cache_bytes(
    parallelism: usize,
    budget: &RamBudget,
    reserved_bytes: u64,
) -> u64 {
    merged_cache_bytes(
        available_memory_bytes(),
        parallelism,
        budget,
        reserved_bytes,
    )
}

/// [`recommended_merged_cache_bytes`] with `available` RAM.
///
/// The storage's share is at most a quarter of `available` (DuckDB's limit
/// comes from the total RAM, not what is free), and it never takes the
/// channel tiles of the tiles in work (`parallelism` × one tile's
/// channels): a cache smaller than that reloads tiles from disk again and
/// again - `tests/haut` with 4 GB available and the whole 8 GB limit
/// reserved: 73 → 326 s, 32 → 839 s system CPU. Never below
/// `min_image_cache_bytes`, as [`recommended_image_cache_bytes`].
pub fn merged_cache_bytes(
    available: u64,
    parallelism: usize,
    budget: &RamBudget,
    reserved_bytes: u64,
) -> u64 {
    let workers = parallelism.max(1) as u64;
    let free = available.saturating_sub(workers * budget.total_bytes());
    let tiles_in_work = (workers * budget.min_image_cache_bytes).min(free);
    free.saturating_sub(reserved_bytes.min(available / 4))
        .max(tiles_in_work)
        .max(budget.min_image_cache_bytes)
}

/// Recommended number of images/tiles to analyze in parallel.
///
/// Starts from the number of CPU cores (minus one, to leave a core free for
/// the UI/OS), then caps that down if available RAM can't comfortably support
/// that many concurrent workers - better to run fewer workers than to hit a
/// system OOM partway through a batch. `ram_per_worker_bytes` should come
/// from [`RamBudget::total_bytes`] (via [`estimate_ram_budget`]), sized to
/// the job actually being run - a flat estimate independent of the image
/// being analyzed either
/// over-commits on a large/many-channel image or, just as bad, needlessly
/// caps a small/modest image down to a single thread on a low-RAM machine.
pub fn recommended_parallelism(ram_per_worker_bytes: u64) -> usize {
    let cores = std::thread::available_parallelism()
        .map(|n| n.get().saturating_sub(1).max(1))
        .unwrap_or(1);
    let ram_capped = (available_memory_bytes() / ram_per_worker_bytes.max(1)).max(1) as usize;
    cores.min(ram_capped).max(1)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A realistic per-worker estimate for a modest single-channel 2048x2048
    /// tile - small enough that every test below is exercising the "cores
    /// are the binding constraint" path on any machine with a few GB of RAM,
    /// not accidentally the RAM-capping path.
    fn modest_ram_per_worker_bytes() -> u64 {
        estimate_ram_budget(2048, 2048, 1).total_bytes()
    }

    #[test]
    fn recommended_parallelism_is_never_zero() {
        assert!(recommended_parallelism(modest_ram_per_worker_bytes()) >= 1);
    }

    #[test]
    fn recommended_parallelism_never_exceeds_available_cores() {
        let cores = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1);
        assert!(recommended_parallelism(modest_ram_per_worker_bytes()) <= cores);
    }

    #[test]
    fn recommended_parallelism_is_capped_down_to_one_by_an_enormous_per_worker_estimate() {
        // A per-worker estimate far larger than any real machine's RAM must
        // still cap down to 1 thread, not divide-by/overflow into 0.
        assert_eq!(recommended_parallelism(u64::MAX / 2), 1);
    }

    #[test]
    fn recommended_parallelism_a_smaller_per_worker_estimate_never_allows_fewer_threads() {
        // Halving the per-worker estimate can only ever raise (or leave
        // unchanged) how many workers fit in the same available RAM - it
        // must never come out lower than the bigger estimate's result.
        let big = modest_ram_per_worker_bytes();
        let small = big / 2;
        assert!(recommended_parallelism(small) >= recommended_parallelism(big));
    }

    #[test]
    fn estimate_ram_budget_scales_with_tile_pixel_count() {
        let small = estimate_ram_budget(512, 512, 1).total_bytes();
        let large = estimate_ram_budget(4096, 4096, 1).total_bytes();
        // 4096x4096 has 64x the pixels of 512x512.
        assert_eq!(large, small * 64);
    }

    #[test]
    fn estimate_ram_budget_scales_with_channel_count() {
        let one_channel = estimate_ram_budget(1024, 1024, 1);
        let four_channels = estimate_ram_budget(1024, 1024, 4);
        assert!(
            four_channels.min_image_cache_bytes > one_channel.min_image_cache_bytes,
            "more cached channel planes must cost more, not be ignored"
        );
        // Only the cache component depends on channel count.
        assert_eq!(
            four_channels.working_set_bytes,
            one_channel.working_set_bytes
        );
        assert_eq!(
            four_channels.object_cache_margin_bytes,
            one_channel.object_cache_margin_bytes
        );
    }

    #[test]
    fn estimate_ram_budget_treats_zero_channels_as_one() {
        // An image metadata gap (e.g. an unpopulated channel list) must not
        // make the estimate collapse to just the working set - that would
        // silently under-budget the cache.
        assert_eq!(
            estimate_ram_budget(1024, 1024, 0).total_bytes(),
            estimate_ram_budget(1024, 1024, 1).total_bytes()
        );
    }

    #[test]
    fn estimate_ram_budget_is_realistic_for_a_modest_multi_channel_image() {
        // A 2048x2048/4-channel image - well under the old flat 1.5 GB
        // estimate, and comfortably in the low hundreds of MB rather than a
        // handful of MB (which would under-budget and risk OOM).
        let estimate = estimate_ram_budget(2048, 2048, 4).total_bytes();
        assert!(
            estimate > 50_000_000 && estimate < 300_000_000,
            "expected a low-hundreds-of-MB estimate, got {estimate} bytes"
        );
    }

    #[test]
    fn estimate_ram_budget_stays_sane_at_the_max_analysis_tile_size() {
        // 4096x4096 (the pipeline's actual max tile size) with several
        // channels cached is a genuinely large working set - this only
        // checks the estimate stays in a sane range (comfortably below the
        // old flat 1.5 GB guess, comfortably above a handful of MB), not a
        // tight bound, since holding multiple full-resolution channel planes
        // at once really is that much memory.
        let estimate = estimate_ram_budget(4096, 4096, 4).total_bytes();
        assert!(
            estimate > 100_000_000 && estimate < 1_500_000_000,
            "expected a sub-1.5GB estimate, got {estimate} bytes"
        );
    }

    #[test]
    fn recommended_image_cache_bytes_is_never_below_the_minimum() {
        // An absurdly high parallelism (far more workers than any real
        // machine's RAM could give a meaningful share to) must still fall
        // back to the minimum, not starve the cache to near-zero.
        let min = 64_000_000;
        assert_eq!(recommended_image_cache_bytes(1_000_000, 10_000, min), min);
    }

    #[test]
    fn recommended_image_cache_bytes_grows_when_fewer_workers_share_the_same_ram() {
        let min = 1_000_000;
        let many_workers = recommended_image_cache_bytes(64, 10_000_000, min);
        let few_workers = recommended_image_cache_bytes(2, 10_000_000, min);
        assert!(
            few_workers >= many_workers,
            "fewer concurrent workers must never result in a smaller per-worker cache"
        );
    }

    #[test]
    fn recommended_merged_cache_bytes_is_never_below_the_minimum() {
        // More workers than any machine has RAM for: the minimum still.
        let budget = estimate_ram_budget(4096, 4096, 5);
        assert_eq!(
            recommended_merged_cache_bytes(1_000_000, &budget, 0),
            budget.min_image_cache_bytes
        );
    }

    #[test]
    fn recommended_merged_cache_bytes_leaves_every_worker_its_tile() {
        let budget = estimate_ram_budget(4096, 4096, 5);
        let few_workers = recommended_merged_cache_bytes(2, &budget, 0);
        let many_workers = recommended_merged_cache_bytes(16, &budget, 0);
        assert!(
            many_workers <= few_workers,
            "each more worker's tile leaves the merged cache less"
        );
    }

    const GB: u64 = 1 << 30;

    #[test]
    fn the_merged_cache_leaves_the_storage_its_memory() {
        let budget = estimate_ram_budget(4096, 4096, 5);
        let free = 16 * GB - 4 * budget.total_bytes();
        assert_eq!(merged_cache_bytes(16 * GB, 4, &budget, 0), free);
        assert_eq!(
            merged_cache_bytes(16 * GB, 4, &budget, 2 * GB),
            free - 2 * GB
        );
    }

    /// DuckDB's limit comes from the total RAM: with little of it free, the
    /// storage gets a quarter of what is free, not its whole limit.
    #[test]
    fn the_storage_takes_at_most_a_quarter_of_the_free_ram() {
        let budget = estimate_ram_budget(512, 512, 1);
        let free = 4 * GB - budget.total_bytes();
        assert_eq!(merged_cache_bytes(4 * GB, 1, &budget, 8 * GB), free - GB);
    }

    /// The storage's share never takes the channel tiles of the tiles in
    /// work - smaller, the cache reloads them from disk again and again.
    #[test]
    fn the_storage_never_takes_the_tiles_in_work() {
        let budget = estimate_ram_budget(4096, 4096, 5);
        let available = 4 * budget.total_bytes() + 4 * budget.min_image_cache_bytes + GB / 2;
        assert_eq!(
            merged_cache_bytes(available, 4, &budget, 8 * GB),
            4 * budget.min_image_cache_bytes
        );
    }
}
