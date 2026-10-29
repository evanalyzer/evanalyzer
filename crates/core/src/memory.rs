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
//! Also here: what the computer has ([`SystemRam`], [`SystemDiagnostics`]) -
//! within a container's memory limit and the user's [`ResourceLimits`]
//! (`--max-threads`, `--max-ram`, Settings › Performance).
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

use std::sync::OnceLock;
use sysinfo::System;

/// The computer's RAM now - in a container with a memory limit (a cgroup:
/// Docker, Kubernetes, a server's systemd slice), the container's.
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
        let ram = Self {
            total: sys.total_memory(),
            available: sys.available_memory(),
        };
        // The host's RAM otherwise: a container limited to 4 GB on a 64 GB
        // host was sized for 64 GB - and killed by its limit.
        let ram = match sys.cgroup_limits() {
            Some(limits) => ram.within(limits.total_memory, limits.free_memory),
            None => ram,
        };
        match resource_limits().max_ram_mb {
            Some(mb) => ram.within(mb << 20, mb << 20),
            None => ram,
        }
    }

    /// This RAM within a limit of `total` with `free` of it unused.
    fn within(self, total: u64, free: u64) -> Self {
        Self {
            total: self.total.min(total),
            available: self.available.min(free),
        }
    }
}

/// How much of the computer EVAnalyzer may use, set by the user: the
/// command line (`--max-threads`, `--max-ram`) or Settings › Performance.
/// `None`: no limit of the user's (the computer's and a container's apply).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ResourceLimits {
    /// Most CPU cores - threads for the analysis, the AI engine, DuckDB and
    /// the general thread pool.
    pub max_threads: Option<usize>,
    /// Most RAM in MB EVAnalyzer plans with - as if the computer had no
    /// more (see [`SystemRam`]). A planning limit: the caches, threads and
    /// DuckDB are sized within it; it isn't enforced on every allocation
    /// (the operating system does that, e.g. a container's limit).
    pub max_ram_mb: Option<u64>,
}

static LIMITS: OnceLock<ResourceLimits> = OnceLock::new();

/// Sets this process' [`ResourceLimits`]. Once, at startup, before anything
/// sizes itself; `false` (nothing changed) if they were set or read before.
/// With a thread limit, the general thread pool (viewer, exports, the
/// steps' own parallel loops) gets that many threads too - it is built here,
/// before its first use.
pub fn set_resource_limits(limits: ResourceLimits) -> bool {
    let set = LIMITS
        .set(ResourceLimits {
            max_threads: limits.max_threads.map(|n| n.max(1)),
            max_ram_mb: limits.max_ram_mb.map(|mb| mb.max(1)),
        })
        .is_ok();
    if set && limits.max_threads.is_some() {
        let threads = cpu_cores();
        if let Err(e) = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build_global()
        {
            log::warn!("Could not limit the thread pool to {threads} threads: {e}");
        }
    }
    set
}

/// This process' [`ResourceLimits`] - none until set.
pub fn resource_limits() -> ResourceLimits {
    *LIMITS.get_or_init(ResourceLimits::default)
}

/// CPU cores EVAnalyzer may use: the computer's (within its affinity and a
/// container's quota) within [`ResourceLimits::max_threads`].
pub fn cpu_cores() -> usize {
    cores_within(
        std::thread::available_parallelism().map_or(1, |n| n.get()),
        resource_limits().max_threads,
    )
}

fn cores_within(cores: usize, max_threads: Option<usize>) -> usize {
    max_threads.map_or(cores, |max| cores.min(max)).max(1)
}

/// Currently available system RAM, in bytes - [`SystemRam::available`].
pub(crate) fn available_memory_bytes() -> u64 {
    SystemRam::now().available
}

/// Hands freed heap memory back to the operating system. glibc keeps freed
/// chunks of its heap for reuse instead - the RAM measured free afterwards
/// would leave out what this process already freed. Elsewhere large blocks
/// go back on their own.
pub fn return_freed_memory_to_os() {
    #[cfg(all(target_os = "linux", target_env = "gnu"))]
    // SAFETY: malloc_trim only releases free memory of the allocator.
    unsafe {
        libc::malloc_trim(0);
    }
}

/// Snapshot of host machine capabilities, surfaced read-only (e.g. in the
/// About dialog's System tab). Not used for any sizing decision - see
/// [`SystemRam`] and the submodules for that.
pub struct SystemDiagnostics {
    pub cpu_cores: usize,
    pub total_ram_bytes: u64,
    pub cuda_available: bool,
}

/// The CPU cores and total RAM EVAnalyzer may use ([`cpu_cores`],
/// [`SystemRam::total`]) - within the user's [`ResourceLimits`]. Cheap and
/// fast (no driver/device probing) - safe to call on a startup path.
pub fn cpu_ram_diagnostics() -> (usize, u64) {
    (cpu_cores(), SystemRam::now().total)
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

    const GB: u64 = 1 << 30;

    #[test]
    fn the_cores_stay_within_the_users_limit() {
        assert_eq!(cores_within(22, None), 22);
        assert_eq!(cores_within(22, Some(4)), 4);
        // A limit above the computer's cores changes nothing.
        assert_eq!(cores_within(4, Some(64)), 4);
        assert_eq!(cores_within(4, Some(0)), 1);
    }

    /// In a container the smaller of its limit and the host's RAM counts.
    #[test]
    fn a_container_limit_caps_the_ram() {
        let host = SystemRam {
            total: 64 * GB,
            available: 50 * GB,
        };
        assert_eq!(
            host.within(4 * GB, 3 * GB),
            SystemRam {
                total: 4 * GB,
                available: 3 * GB
            }
        );
        // A limit above the host's RAM changes nothing.
        assert_eq!(host.within(128 * GB, 100 * GB), host);
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

    #[test]
    fn system_diagnostics_reports_at_least_one_core_and_nonzero_ram() {
        let diag = system_diagnostics();
        assert!(diag.cpu_cores >= 1);
        assert!(diag.total_ram_bytes > 0);
    }
}
