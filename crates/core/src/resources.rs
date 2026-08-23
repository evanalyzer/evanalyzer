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

/// Bound on the `sync_channel::<PipelineCache>` handoff to the DB writer
/// thread (see both `analyze_image` and `analyze_image_tiles_parallel`
/// below). This is the number that actually decides a worker's worst-case
/// memory footprint: if the writer falls behind, this many completed
/// `PipelineCache`s (cached channel planes plus every object mask/metadata
/// for that tile) can queue up in memory before a worker blocks on send,
/// rather than just the one currently being written. Kept at 1 - enough to
/// keep a worker computing the next tile while the writer drains the
/// previous one (no double-buffering was needed beyond that in practice) -
/// instead of the 4 it used to be, which let up to 4x a tile's worth of
/// cached data pile up per worker in the worst case. `estimate_ram_per_worker_bytes`
/// budgets against exactly this constant, so the two must be changed together.
pub const CACHE_QUEUE_DEPTH: usize = 1;

/// Estimates the peak RAM one parallel worker can need to process one tile
/// of the image actually being analyzed, instead of guessing with a flat
/// constant - the two components that actually scale are the number of
/// pixels in a tile and how many channel planes a pipeline can have cached
/// at once (e.g. a colocalization step reading back an earlier channel's
/// classification):
///
/// - **Working set**: one in-flight [`PipelineContext`](crate::pipeline::pipeline_context::PipelineContext)'s
///   own tile-sized buffers ([`WORKING_SET_BUFFERS`] of them).
/// - **Per queued result**: a completed tile's cached channel planes
///   (`channel_count` of them) plus an object-mask margin, times
///   `queue_depth` - the DB-writer backpressure channel a worker's
///   completed [`PipelineCache`](crate::pipeline::pipeline_cache::PipelineCache)
///   is handed off to lets up to `queue_depth` of them queue up if the
///   writer falls behind, so that's how many can be held in memory at once
///   in the worst case, not just one.
///
/// `tile_width`/`tile_height` should be the actual per-tile dimensions a
/// worker will process (the smaller of the image's own size and the
/// pipeline's max tile size), not necessarily the whole image - a worker
/// never holds more than one tile's worth of pixels per buffer regardless of
/// how large the source image is. Heuristic guardrail against over-committing
/// on low-RAM machines, not a measured per-pipeline bound.
pub fn estimate_ram_per_worker_bytes(
    tile_width: usize,
    tile_height: usize,
    channel_count: usize,
    queue_depth: usize,
) -> u64 {
    let tile_pixels = tile_width as u64 * tile_height as u64;
    let plane_bytes = tile_pixels * BYTES_PER_PIXEL;

    let working_set = plane_bytes * WORKING_SET_BUFFERS;
    let image_cache = plane_bytes * channel_count.max(1) as u64;
    let object_cache_margin = plane_bytes / OBJECT_CACHE_MARGIN_DIVISOR;
    let per_queued_result = image_cache + object_cache_margin;

    working_set + per_queued_result * queue_depth.max(1) as u64
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
/// from [`estimate_ram_per_worker_bytes`], sized to the job actually being
/// run - a flat estimate independent of the image being analyzed either
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
        estimate_ram_per_worker_bytes(2048, 2048, 1, 1)
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
    fn estimate_ram_per_worker_bytes_scales_with_tile_pixel_count() {
        let small = estimate_ram_per_worker_bytes(512, 512, 1, 1);
        let large = estimate_ram_per_worker_bytes(4096, 4096, 1, 1);
        // 4096x4096 has 64x the pixels of 512x512.
        assert_eq!(large, small * 64);
    }

    #[test]
    fn estimate_ram_per_worker_bytes_scales_with_channel_count() {
        let one_channel = estimate_ram_per_worker_bytes(1024, 1024, 1, 1);
        let four_channels = estimate_ram_per_worker_bytes(1024, 1024, 4, 1);
        assert!(
            four_channels > one_channel,
            "more cached channel planes must cost more, not be ignored"
        );
    }

    #[test]
    fn estimate_ram_per_worker_bytes_scales_with_queue_depth() {
        let depth_1 = estimate_ram_per_worker_bytes(1024, 1024, 2, 1);
        let depth_4 = estimate_ram_per_worker_bytes(1024, 1024, 2, 4);
        assert!(
            depth_4 > depth_1,
            "a deeper backpressure queue must cost more, not be ignored"
        );
    }

    #[test]
    fn estimate_ram_per_worker_bytes_treats_zero_channels_and_zero_queue_depth_as_one() {
        // An image metadata gap (e.g. an unpopulated channel list) must not
        // make the estimate collapse to just the working set - that would
        // silently under-budget every worker's queued-result memory.
        assert_eq!(
            estimate_ram_per_worker_bytes(1024, 1024, 0, 0),
            estimate_ram_per_worker_bytes(1024, 1024, 1, 1)
        );
    }

    #[test]
    fn estimate_ram_per_worker_bytes_is_realistic_for_a_modest_multi_channel_image() {
        // A 2048x2048/4-channel image at queue depth 1 - well under the old
        // flat 1.5 GB estimate, and comfortably in the low hundreds of MB
        // rather than a handful of MB (which would under-budget and risk OOM).
        let estimate = estimate_ram_per_worker_bytes(2048, 2048, 4, 1);
        assert!(
            estimate > 50_000_000 && estimate < 300_000_000,
            "expected a low-hundreds-of-MB estimate, got {estimate} bytes"
        );
    }

    #[test]
    fn estimate_ram_per_worker_bytes_stays_sane_at_the_max_analysis_tile_size() {
        // 4096x4096 (the pipeline's actual max tile size) with several
        // channels cached is a genuinely large working set - this only
        // checks the estimate stays in a sane range (comfortably below the
        // old flat 1.5 GB guess, comfortably above a handful of MB), not a
        // tight bound, since holding multiple full-resolution channel planes
        // at once really is that much memory.
        let estimate = estimate_ram_per_worker_bytes(4096, 4096, 4, 1);
        assert!(
            estimate > 100_000_000 && estimate < 1_500_000_000,
            "expected a sub-1.5GB estimate, got {estimate} bytes"
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
}
