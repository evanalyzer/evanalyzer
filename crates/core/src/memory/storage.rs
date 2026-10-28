//! # storage
//!
//! The results database's RAM: DuckDB's memory limit per open results file.
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
    fn duckdb_takes_a_quarter_of_the_ram_between_1_and_8_gb() {
        let gb = |n: u64| n << 30;
        assert_eq!(results_memory_limit_mb(gb(16)), 4096);
        assert_eq!(results_memory_limit_mb(gb(2)), 1024);
        assert_eq!(results_memory_limit_mb(gb(256)), 8192);
    }
}
