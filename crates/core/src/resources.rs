//! # resources
//!
//! Caps pipeline parallelism and reader-pool size based on how much system
//! RAM is actually free, instead of fixed constants that can over-commit on
//! small machines (OOM) or under-use big ones.
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-06-28
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **AGPL-3.0**.

use sysinfo::System;

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
/// [`estimate_ram_per_worker_bytes`] as a whole).
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
pub fn estimate_ram_budget(tile_width: usize, tile_height: usize, channel_count: usize) -> RamBudget {
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

/// Formats a byte count in binary (1024-based) units - GiB/MiB - not the
/// decimal (1000-based) GB/MB a raw byte count is easy to misread as. Only
/// two tiers: everything this module sizes (a per-worker RAM/cache budget)
/// falls in the tens-of-MB-to-low-single-digit-GB range, never small enough
/// to need KiB or large enough to need TiB.
pub fn format_binary_bytes(bytes: u64) -> String {
    const MIB: f64 = 1024.0 * 1024.0;
    const GIB: f64 = MIB * 1024.0;
    let bytes = bytes as f64;
    if bytes >= GIB {
        format!("{:.2} GiB", bytes / GIB)
    } else {
        format!("{:.2} MiB", bytes / MIB)
    }
}

/// Rough estimate of the peak RAM one pooled reader can hold: parsed
/// metadata/tile index and headroom for one in-flight tile buffer. Much
/// lighter than a pipeline worker (see [`estimate_ram_per_worker_bytes`]) -
/// a reader has no pipeline scratch buffers or object masks - so
/// reader-pool sizing is a separate, smaller budget from
/// [`recommended_parallelism`]. Heuristic guardrail, not a measured bound.
const ESTIMATED_RAM_PER_READER_BYTES: u64 = 150_000_000;

/// Ceiling on reader-pool size regardless of cores/RAM available. Some
/// multiplexed formats have dozens of channels; there's no benefit to a
/// pool bigger than a handful of readers even on a large, idle machine, and
/// every extra reader is another format reader's worth of held metadata plus
/// another potential in-flight tile buffer.
const MAX_READER_POOL_SIZE: usize = 8;

/// Currently available system RAM, in bytes (free memory plus easily
/// reclaimable caches/buffers - what the OS would actually hand out to a new
/// allocation right now).
fn available_memory_bytes() -> u64 {
    let mut sys = System::new();
    sys.refresh_memory();
    sys.available_memory()
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

/// Recommended number of independent format readers to keep pooled for one
/// open image, so multiple channels/Z-slices can be read in parallel instead
/// of serializing through a single reader's `Mutex` (see `ReaderPool` in
/// `evanalyzer_app`).
///
/// Same shape as [`recommended_parallelism`] - cores (minus one, for the
/// UI thread, since this pool backs interactive viewport rendering) capped
/// by however many readers available RAM can comfortably hold - but against
/// [`ESTIMATED_RAM_PER_READER_BYTES`] instead, since a reader is far
/// lighter than a full pipeline worker, and additionally bounded by
/// [`MAX_READER_POOL_SIZE`].
pub fn recommended_reader_pool_size() -> usize {
    let cores = std::thread::available_parallelism()
        .map(|n| n.get().saturating_sub(1).max(1))
        .unwrap_or(1);
    let ram_capped = (available_memory_bytes() / ESTIMATED_RAM_PER_READER_BYTES).max(1) as usize;
    cores.min(ram_capped).min(MAX_READER_POOL_SIZE).max(1)
}

/// Snapshot of host machine capabilities, surfaced read-only (e.g. in the
/// About dialog's System tab). Not used for any sizing decision - see
/// [`recommended_parallelism`]/[`recommended_jvm_heap_bytes`] for that.
pub struct SystemDiagnostics {
    pub cpu_cores: usize,
    pub total_ram_bytes: u64,
    pub cuda_available: bool,
}

/// Reads current host CPU core count and total RAM. Cheap and fast (no
/// driver/device probing) - safe to call on a startup path.
pub fn cpu_ram_diagnostics() -> (usize, u64) {
    let cpu_cores = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);

    let mut sys = System::new();
    sys.refresh_memory();

    (cpu_cores, sys.total_memory())
}

/// Reads current host CPU/RAM/CUDA availability once, for display purposes.
///
/// Includes the CUDA probe (see [`cuda_is_available`]), which is slow - don't
/// call this on a UI startup path; use [`cpu_ram_diagnostics`] plus a
/// backgrounded [`cuda_is_available`] instead (see `evanalyzer_gui`).
pub fn system_diagnostics() -> SystemDiagnostics {
    let (cpu_cores, total_ram_bytes) = cpu_ram_diagnostics();

    SystemDiagnostics {
        cpu_cores,
        total_ram_bytes,
        cuda_available: cuda_is_available(),
    }
}

/// Probes CUDA driver/device availability. Loading the CUDA driver and
/// creating a context on first use is slow (commonly hundreds of ms), so
/// callers on a UI startup path should run this on a background thread
/// rather than blocking on it - see its use in `evanalyzer_gui`.
#[cfg(feature = "ai")]
pub fn cuda_is_available() -> bool {
    tch::Device::cuda_if_available().is_cuda()
}

#[cfg(not(feature = "ai"))]
pub fn cuda_is_available() -> bool {
    false
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
        assert_eq!(
            recommended_image_cache_bytes(1_000_000, 10_000, min),
            min
        );
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
    fn recommended_reader_pool_size_is_never_zero() {
        assert!(recommended_reader_pool_size() >= 1);
    }

    #[test]
    fn recommended_reader_pool_size_never_exceeds_available_cores() {
        let cores = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1);
        assert!(recommended_reader_pool_size() <= cores);
    }

    #[test]
    fn recommended_reader_pool_size_never_exceeds_the_hard_cap() {
        assert!(recommended_reader_pool_size() <= MAX_READER_POOL_SIZE);
    }

    #[test]
    fn system_diagnostics_reports_at_least_one_core_and_nonzero_ram() {
        let diag = system_diagnostics();
        assert!(diag.cpu_cores >= 1);
        assert!(diag.total_ram_bytes > 0);
    }

    #[test]
    fn format_binary_bytes_uses_mib_below_one_gib() {
        assert_eq!(format_binary_bytes(0), "0.00 MiB");
        assert_eq!(format_binary_bytes(1024 * 1024), "1.00 MiB");
        assert_eq!(format_binary_bytes(789_000_000), "752.45 MiB");
    }

    #[test]
    fn format_binary_bytes_switches_to_gib_at_the_boundary() {
        assert_eq!(format_binary_bytes(1024 * 1024 * 1024), "1.00 GiB");
        assert_eq!(format_binary_bytes(2_270_377_496), "2.11 GiB");
    }
}
