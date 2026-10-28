//! # memory
//!
//! Every rule for how much RAM the parts of EVAnalyzer may take, in one
//! place - so they can be added up against what the computer has.
//!
//! - [`analysis`]: the analysis' threads and image caches (each worker's,
//!   the merged cache of a whole slide) - from the RAM free when a run
//!   starts.
//! - [`storage`]: the results database (DuckDB's memory limit per open
//!   results file) - from the total RAM.
//! - [`viewer`]: the editor's automatic sizes (tile and overview caches,
//!   preview and breakpoint tiles, undo history) - from the total RAM; the
//!   user can set each by hand (Settings › Viewport).
//!
//! Also here: what the computer has ([`SystemRam`], [`SystemDiagnostics`]).
//!
//! Not bounded yet: a whole slide's objects until export, deep-learning
//! models (loaded until the app ends), the editor's preview objects.
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-10-28
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.

pub mod analysis;
pub mod storage;
pub mod viewer;

use sysinfo::System;

/// The computer's RAM now.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SystemRam {
    /// All of it.
    pub total: u64,
    /// Free memory plus easily reclaimable caches/buffers - what the OS
    /// would hand out to a new allocation right now.
    pub available: u64,
}

impl SystemRam {
    pub fn now() -> Self {
        let mut sys = System::new();
        sys.refresh_memory();
        Self {
            total: sys.total_memory(),
            available: sys.available_memory(),
        }
    }
}

/// Currently available system RAM, in bytes - [`SystemRam::available`].
pub(crate) fn available_memory_bytes() -> u64 {
    SystemRam::now().available
}

/// Snapshot of host machine capabilities, surfaced read-only (e.g. in the
/// About dialog's System tab). Not used for any sizing decision - see
/// [`SystemRam`] and the submodules for that.
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

    (cpu_cores, SystemRam::now().total)
}

/// Reads current host CPU/RAM/CUDA availability once, for display purposes.
///
/// Includes the CUDA probe (see [`cuda_is_available`]), which is slow - don't
/// call this on a UI startup path; use [`cpu_ram_diagnostics`] plus a
/// backgrounded [`cuda_is_available`] instead (see `evanalyzer_gui`).
///
/// [`cuda_is_available`]: crate::cuda_is_available
pub fn system_diagnostics() -> SystemDiagnostics {
    let (cpu_cores, total_ram_bytes) = cpu_ram_diagnostics();

    SystemDiagnostics {
        cpu_cores,
        total_ram_bytes,
        cuda_available: evanalyzer_ai::cuda_is_available(),
    }
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

#[cfg(test)]
mod tests {
    use super::*;

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

    #[test]
    fn system_diagnostics_reports_at_least_one_core_and_nonzero_ram() {
        let diag = system_diagnostics();
        assert!(diag.cpu_cores >= 1);
        assert!(diag.total_ram_bytes > 0);
    }
}
