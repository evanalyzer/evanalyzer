//! # resources
//!
//! Sizes the embedded JVM heap and caps pipeline parallelism based on how much
//! system RAM is actually free, instead of fixed constants that can over-commit
//! on small machines (JVM heap OOM) or under-use big ones.
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-06-28
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **AGPL-3.0**.

use sysinfo::System;

/// Floor for the JVM heap: below this, Bio-Formats itself struggles to load
/// even moderately sized images.
const MIN_JVM_HEAP_BYTES: u64 = 512 * 1024 * 1024;
/// Ceiling for the JVM heap: the JVM is only used as a thin bridge to
/// Bio-Formats for reading image tiles, so it never needs a large heap even
/// on machines with a lot of RAM - leave that RAM for the Rust-side pipeline.
const MAX_JVM_HEAP_BYTES: u64 = 4 * 1024 * 1024 * 1024;
/// Share of currently available RAM the JVM heap is allowed to claim.
const JVM_HEAP_RAM_FRACTION: f64 = 0.125;

/// Rough estimate of the peak RAM one parallel worker (one whole image, or
/// one tile when previewing a single whole-slide image) can need: loaded
/// channel planes, pipeline intermediate/scratch buffers and ROI masks for a
/// 4096px tile across several channels. This is a heuristic guardrail against
/// over-committing on low-RAM machines, not a measured per-pipeline bound.
const ESTIMATED_RAM_PER_WORKER_BYTES: u64 = 1_500_000_000;

/// Currently available system RAM, in bytes (free memory plus easily
/// reclaimable caches/buffers - what the OS would actually hand out to a new
/// allocation right now).
fn available_memory_bytes() -> u64 {
    let mut sys = System::new();
    sys.refresh_memory();
    sys.available_memory()
}

/// Recommended JVM heap size, in bytes, based on currently available RAM.
///
/// Scales with [`JVM_HEAP_RAM_FRACTION`] of available memory, clamped to
/// [`MIN_JVM_HEAP_BYTES`]..=[`MAX_JVM_HEAP_BYTES`] so it neither starves on a
/// constrained machine nor reserves more than the JVM bridge ever needs.
pub fn recommended_jvm_heap_bytes() -> u64 {
    let target = (available_memory_bytes() as f64 * JVM_HEAP_RAM_FRACTION) as u64;
    target.clamp(MIN_JVM_HEAP_BYTES, MAX_JVM_HEAP_BYTES)
}

/// Recommended number of images/tiles to analyze in parallel.
///
/// Starts from the number of CPU cores (minus one, to leave a core free for
/// the UI/OS), then caps that down if available RAM can't comfortably support
/// that many concurrent workers - better to run fewer workers than to hit a
/// JVM heap or system OOM partway through a batch.
pub fn recommended_parallelism() -> usize {
    let cores = std::thread::available_parallelism()
        .map(|n| n.get().saturating_sub(1).max(1))
        .unwrap_or(1);
    let ram_capped = (available_memory_bytes() / ESTIMATED_RAM_PER_WORKER_BYTES).max(1) as usize;
    cores.min(ram_capped).max(1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recommended_jvm_heap_stays_within_bounds() {
        let heap = recommended_jvm_heap_bytes();
        assert!(heap >= MIN_JVM_HEAP_BYTES);
        assert!(heap <= MAX_JVM_HEAP_BYTES);
    }

    #[test]
    fn recommended_parallelism_is_never_zero() {
        assert!(recommended_parallelism() >= 1);
    }

    #[test]
    fn recommended_parallelism_never_exceeds_available_cores() {
        let cores = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
        assert!(recommended_parallelism() <= cores);
    }
}
