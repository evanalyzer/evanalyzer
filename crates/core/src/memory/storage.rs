//! # storage
//!
//! The results database's RAM: one DuckDB memory limit for all open results
//! files together, shared out among them.
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-10-28
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.

/// The memory DuckDB may use for a results file on a computer with
/// `total_ram_bytes`: a quarter of it, between 1 GB and 8 GB.
pub fn results_memory_limit_mb(total_ram_bytes: u64) -> u64 {
    (total_ram_bytes / 4 / (1 << 20)).clamp(1024, 8192)
}

/// A results file's share in MB of [`results_memory_limit_mb`] when
/// `open_files` are open at once: one limit for all of them (the results
/// window and a running analysis each had the whole of it), at least
/// [`RESULTS_FILE_MIN_MB`] each.
pub fn results_memory_limit_per_file_mb(total_ram_bytes: u64, open_files: usize) -> u64 {
    (results_memory_limit_mb(total_ram_bytes) / open_files.max(1) as u64).max(RESULTS_FILE_MIN_MB)
}

/// A results file's smallest share: below it DuckDB can't run some queries
/// at all (not every operator spills to disk).
pub const RESULTS_FILE_MIN_MB: u64 = 512;

/// [`results_memory_limit_mb`] on this computer, in bytes - also for the
/// analysis' RAM budget: a run writes into one, and DuckDB uses most of it
/// (`tests/haut`: 1.3-1.6 GB).
pub fn results_memory_limit_bytes() -> u64 {
    results_memory_limit_mb(super::SystemRam::now().total) << 20
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn open_results_files_share_one_limit() {
        let gb = |n: u64| n << 30;
        assert_eq!(results_memory_limit_per_file_mb(gb(16), 1), 4096);
        assert_eq!(results_memory_limit_per_file_mb(gb(16), 2), 2048);
        assert_eq!(results_memory_limit_per_file_mb(gb(16), 0), 4096);
        // Never below the floor: 8 GB, 4 files - 512 MB each, not 256.
        assert_eq!(
            results_memory_limit_per_file_mb(gb(8), 4),
            RESULTS_FILE_MIN_MB
        );
    }

    #[test]
    fn duckdb_takes_a_quarter_of_the_ram_between_1_and_8_gb() {
        let gb = |n: u64| n << 30;
        assert_eq!(results_memory_limit_mb(gb(16)), 4096);
        assert_eq!(results_memory_limit_mb(gb(2)), 1024);
        assert_eq!(results_memory_limit_mb(gb(256)), 8192);
    }
}
