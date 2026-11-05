//! # accounting
//!
//! What the process really has allocated in large blocks - the image tiles,
//! windows and buffers where the RAM goes - next to the budget planned in
//! [`analysis`](super::analysis). The system allocator, counting only
//! allocations of [`LARGE_BYTES`] and more: a small allocation (the vast
//! majority) passes straight through after one comparison, so counting costs
//! the common path nothing measurable. Measured 2026-11-02 (BENCHMARK.md
//! "What the memory counter costs"): a small allocation +0-0.5 ns, a large
//! one +7-13 ns (up to +1.9 µs with 22 threads allocating nothing else); a
//! pipeline run at most 0.02-0.06 % CPU, below the run-to-run noise;
//! < 1 KB of RAM. Counting every allocation cost 3-5 % CPU.
//!
//! The application installs it as its global allocator (see `evanalyzer_bin`).
//! With it:
//! - each whole-image step logs how much RAM it took in large blocks when
//!   that is a lot ([`ImageAlgorithm::run`](crate::algos::ImageAlgorithm)),
//! - a failed allocation (out of memory: the process ends) writes a line to
//!   the crash log first, with the size and the command running
//!   ([`log_out_of_memory_to`]). Not where the operating system ends the
//!   process itself (Linux' OOM killer): nothing can log then.
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-10-30
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::fs::File;
use std::io::Write;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicUsize, Ordering};

/// From this size on an allocation is counted (1 MiB).
pub const LARGE_BYTES: usize = 1 << 20;

/// A counter on a cache line of its own: threads updating one don't slow
/// down the others (no false sharing).
#[repr(align(128))]
struct Counter(AtomicUsize);

static LARGE_IN_USE: Counter = Counter(AtomicUsize::new(0));
static LARGE_PEAK: Counter = Counter(AtomicUsize::new(0));
/// The peak since the last [`start_phase`].
static PHASE_PEAK: Counter = Counter(AtomicUsize::new(0));
/// Where a failed allocation is noted (the crash log, opened at start).
static OUT_OF_MEMORY_LOG: OnceLock<File> = OnceLock::new();

thread_local! {
    /// The command this thread runs (`""`: none) - for the crash log.
    static COMMAND: Cell<&'static str> = const { Cell::new("") };
}

/// The system allocator, counting the large allocations.
pub struct AccountingAllocator;

unsafe impl GlobalAlloc for AccountingAllocator {
    #[inline]
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        // SAFETY: the caller's contract, passed on.
        let ptr = unsafe { System.alloc(layout) };
        if layout.size() >= LARGE_BYTES && !ptr.is_null() {
            added(layout.size());
        } else if ptr.is_null() {
            out_of_memory(layout.size());
        }
        ptr
    }

    #[inline]
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        // SAFETY: the caller's contract, passed on.
        let ptr = unsafe { System.alloc_zeroed(layout) };
        if layout.size() >= LARGE_BYTES && !ptr.is_null() {
            added(layout.size());
        } else if ptr.is_null() {
            out_of_memory(layout.size());
        }
        ptr
    }

    #[inline]
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: the caller's contract, passed on.
        unsafe { System.dealloc(ptr, layout) };
        if layout.size() >= LARGE_BYTES {
            LARGE_IN_USE.0.fetch_sub(layout.size(), Ordering::Relaxed);
        }
    }

    #[inline]
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        // SAFETY: the caller's contract, passed on.
        let new = unsafe { System.realloc(ptr, layout, new_size) };
        if !new.is_null() {
            if layout.size() >= LARGE_BYTES {
                LARGE_IN_USE.0.fetch_sub(layout.size(), Ordering::Relaxed);
            }
            if new_size >= LARGE_BYTES {
                added(new_size);
            }
        } else {
            out_of_memory(new_size);
        }
        new
    }
}

#[inline]
fn added(size: usize) {
    let now = LARGE_IN_USE.0.fetch_add(size, Ordering::Relaxed) + size;
    // A peak rarely rises: read first, write only then - a `fetch_max` on
    // every allocation is a contended write loop with many threads
    // (22 threads: 3.4 µs per large allocation, measured 2026-11-02).
    raise(&PHASE_PEAK, now);
    raise(&LARGE_PEAK, now);
}

/// Raises `peak` to `now` if it's lower.
#[inline]
fn raise(peak: &Counter, now: usize) {
    if now > peak.0.load(Ordering::Relaxed) {
        peak.0.fetch_max(now, Ordering::Relaxed);
    }
}

/// Notes a failed allocation in the crash log - without allocating: the
/// line is formatted on the stack and written to the file opened at start.
#[cold]
fn out_of_memory(size: usize) {
    let Some(mut file) = OUT_OF_MEMORY_LOG.get() else {
        return;
    };
    let command = COMMAND.try_with(Cell::get).unwrap_or("");
    let seconds = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_secs());
    let mut line = StackLine::default();
    let _ = std::fmt::Write::write_fmt(
        &mut line,
        format_args!(
            "\n=== EVAnalyzer out of memory (unix time {seconds}) ===\n\
             An allocation of {} MB failed, {} MB in large blocks in use (peak {} MB), \
             command: {}\n",
            size >> 20,
            large_in_use() >> 20,
            large_peak() >> 20,
            if command.is_empty() { "-" } else { command },
        ),
    );
    let _ = file.write_all(line.as_bytes());
}

/// A line formatted without allocating (cut at 512 bytes).
struct StackLine {
    buffer: [u8; 512],
    len: usize,
}

impl Default for StackLine {
    fn default() -> Self {
        Self {
            buffer: [0; 512],
            len: 0,
        }
    }
}

impl StackLine {
    fn as_bytes(&self) -> &[u8] {
        &self.buffer[..self.len]
    }
}

impl std::fmt::Write for StackLine {
    fn write_str(&mut self, s: &str) -> std::fmt::Result {
        let n = s.len().min(self.buffer.len() - self.len);
        self.buffer[self.len..self.len + n].copy_from_slice(&s.as_bytes()[..n]);
        self.len += n;
        Ok(())
    }
}

/// Where a failed allocation is noted from now on (once; the crash log).
pub fn log_out_of_memory_to(file: File) {
    let _ = OUT_OF_MEMORY_LOG.set(file);
}

/// Marks `command` as running on this thread until the guard drops.
pub fn running(command: &'static str) -> Running {
    Running(COMMAND.replace(command))
}

/// See [`running`]: restores the command before.
pub struct Running(&'static str);

impl Drop for Running {
    fn drop(&mut self) {
        COMMAND.set(self.0);
    }
}

/// Starts a new phase peak from the bytes in use now, and returns them. A
/// phase is a whole-image step; while other images' tiles run beside it,
/// their blocks count too - exact for a whole slide (one image).
pub fn start_phase() -> usize {
    let now = large_in_use();
    PHASE_PEAK.0.store(now, Ordering::Relaxed);
    now
}

/// The most bytes in large blocks since [`start_phase`].
pub fn phase_peak() -> usize {
    PHASE_PEAK.0.load(Ordering::Relaxed)
}

/// The bytes in large allocations now.
pub fn large_in_use() -> usize {
    LARGE_IN_USE.0.load(Ordering::Relaxed)
}

/// The most bytes in large allocations so far.
pub fn large_peak() -> usize {
    LARGE_PEAK.0.load(Ordering::Relaxed)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fmt::Write as _;
    use std::sync::Mutex;

    /// Held by the tests calling the allocator: one at a time changes the
    /// counters.
    static ALLOCATING: Mutex<()> = Mutex::new(());

    #[test]
    fn a_line_on_the_stack_is_cut_not_overflowed() {
        let mut line = StackLine::default();
        for _ in 0..100 {
            let _ = write!(line, "0123456789");
        }
        assert_eq!(line.as_bytes().len(), 512);
        assert!(line.as_bytes().iter().all(u8::is_ascii_digit));
    }

    #[test]
    fn the_command_running_is_restored_after_the_guard() {
        let outer = running("Voronoi");
        {
            let _inner = running("Propagation");
            assert_eq!(COMMAND.with(Cell::get), "Propagation");
        }
        assert_eq!(COMMAND.with(Cell::get), "Voronoi");
        drop(outer);
        assert_eq!(COMMAND.with(Cell::get), "");
    }

    /// Only these tests call the allocator itself: the counters change by
    /// exactly their blocks, though other tests run beside them.
    #[test]
    fn large_blocks_are_counted_and_small_ones_pass_through() {
        let _one_at_a_time = ALLOCATING.lock().unwrap_or_else(|e| e.into_inner());
        let allocator = AccountingAllocator;
        let mib = LARGE_BYTES;
        let base = large_in_use();
        let small = Layout::from_size_align(LARGE_BYTES - 1, 8).unwrap();
        let one = Layout::from_size_align(mib, 8).unwrap();
        let two = Layout::from_size_align(2 * mib, 8).unwrap();
        unsafe {
            let p = allocator.alloc(small);
            assert!(!p.is_null());
            assert_eq!(large_in_use(), base, "below 1 MiB: not counted");
            allocator.dealloc(p, small);
            assert_eq!(large_in_use(), base);

            let a = allocator.alloc(one);
            assert!(!a.is_null());
            assert_eq!(large_in_use(), base + mib, "1 MiB on: counted");
            let z = allocator.alloc_zeroed(two);
            assert!(!z.is_null());
            assert!(
                std::slice::from_raw_parts(z, 2 * mib)
                    .iter()
                    .all(|&b| b == 0)
            );
            assert_eq!(large_in_use(), base + 3 * mib);
            assert!(large_peak() >= base + 3 * mib);
            assert_eq!(
                start_phase(),
                base + 3 * mib,
                "a phase starts from the bytes in use"
            );

            // Large to larger: the old size goes, the new comes.
            a.write(7);
            let a = allocator.realloc(a, one, 3 * mib);
            assert!(!a.is_null());
            assert_eq!(*a, 7, "realloc keeps the content");
            assert_eq!(large_in_use(), base + 5 * mib);
            let peak = large_peak();
            assert!(peak >= base + 5 * mib);
            // Large to small: no longer counted.
            let three = Layout::from_size_align(3 * mib, 8).unwrap();
            let a = allocator.realloc(a, three, 16);
            assert!(!a.is_null());
            assert_eq!(large_in_use(), base + 2 * mib);
            // Small to large: counted from then on.
            let tiny = Layout::from_size_align(16, 8).unwrap();
            let a = allocator.realloc(a, tiny, mib);
            assert!(!a.is_null());
            assert_eq!(large_in_use(), base + 3 * mib);

            allocator.dealloc(a, one);
            allocator.dealloc(z, two);
            assert_eq!(large_in_use(), base, "every block given back");
            assert_eq!(large_peak(), peak, "freeing never lowers the peak");
        }
    }

    /// A failed allocation counts nothing and - once a crash log is set -
    /// notes its size, the large blocks in use and the command running.
    #[test]
    fn a_failed_allocation_is_noted_in_the_crash_log() {
        use std::io::{Read as _, Seek as _};
        let _one_at_a_time = ALLOCATING.lock().unwrap_or_else(|e| e.into_inner());
        let allocator = AccountingAllocator;
        // Larger than any address space: the system allocator fails.
        let huge = Layout::from_size_align(1 << 62, 1).unwrap();
        let base = large_in_use();
        unsafe {
            // No crash log yet: nothing to write to, nothing breaks.
            assert!(allocator.alloc(huge).is_null());
            assert_eq!(large_in_use(), base);

            let file = tempfile::tempfile().unwrap();
            log_out_of_memory_to(file.try_clone().unwrap());
            assert!(allocator.alloc(huge).is_null());
            {
                let _running = running("Voronoi");
                assert!(allocator.alloc_zeroed(huge).is_null());
                let small = Layout::from_size_align(64, 8).unwrap();
                let p = allocator.alloc(small);
                assert!(allocator.realloc(p, small, 1 << 61).is_null());
                assert_eq!(large_in_use(), base, "nothing counted on failure");
                // A failed realloc leaves the old block as it was.
                allocator.dealloc(p, small);
            }

            let mut log = String::new();
            (&file).rewind().unwrap();
            (&file).read_to_string(&mut log).unwrap();
            let lines: Vec<&str> = log
                .lines()
                .filter(|l| l.starts_with("An allocation"))
                .collect();
            assert_eq!(lines.len(), 3, "{log}");
            assert_eq!(
                log.matches("=== EVAnalyzer out of memory (unix time ")
                    .count(),
                3
            );
            let in_use = format!("{} MB in large blocks in use", base >> 20);
            assert!(lines[0].starts_with(&format!(
                "An allocation of {} MB failed, {in_use}",
                1u64 << 42
            )));
            assert!(lines[0].ends_with("command: -"), "no command running");
            assert!(lines[1].starts_with(&format!("An allocation of {} MB failed", 1u64 << 42)));
            assert!(lines[1].ends_with("command: Voronoi"));
            assert!(lines[2].starts_with(&format!("An allocation of {} MB failed", 1u64 << 41)));
            assert!(lines[2].ends_with("command: Voronoi"));
        }
    }

    #[test]
    fn a_stack_line_keeps_what_fits() {
        let mut line = StackLine::default();
        assert!(line.as_bytes().is_empty());
        write!(line, "{} MB", 42).unwrap();
        assert_eq!(line.as_bytes(), b"42 MB");
    }
}
