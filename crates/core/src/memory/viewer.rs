//! # viewer
//!
//! The editor's automatic RAM sizes (Settings › Viewport, "Auto"): the
//! viewer's tile and overview caches, the preview and breakpoint tiles, the
//! undo history - all from the computer's total RAM. Set by hand, the
//! user's value is used instead.
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-10-28
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.

use super::analysis::{MAX_TILE_SIZE, estimate_ram_budget};

const MB: u64 = 1024 * 1024;

/// The automatic tile cache's largest size - also its default.
pub const TILE_CACHE_MAX_MB: u32 = 1024;
/// The automatic overview cache's largest size - also its default.
pub const OVERVIEW_CACHE_MAX_MB: u32 = 256;
/// Below this a cache can't hold even one typical tile/overview.
pub const CACHE_MIN_MB: u32 = 64;
/// Share of the RAM the automatic tile cache takes.
const TILE_CACHE_RAM_PERCENT: u64 = 10;
const TILE_CACHE_AUTO_MIN_MB: u32 = 128;
const OVERVIEW_CACHE_DIVISOR: u32 = 4;
const UNDO_MIN_MB: u64 = 32;
const UNDO_MAX_MB: u64 = 1024;
/// Share of the RAM left over that the automatic undo history takes.
const UNDO_PERCENT_OF_SPARE: u64 = 10;
/// RAM kept for the app itself and the operating system.
const APP_AND_SYSTEM_RESERVE_MB: u64 = 512;

/// Cache sizes (tile, overview) in MB for a computer with
/// `total_ram_bytes`: 10 % of the RAM for tiles, between 128 MB and 1 GB,
/// and a quarter of that for overviews, between 64 MB and 256 MB - so from
/// 16 GB RAM up it's the defaults, and a small machine keeps most of its
/// RAM.
pub fn cache_mb(total_ram_bytes: u64) -> (u32, u32) {
    let total_mb = total_ram_bytes / MB;
    let tile = (total_mb * TILE_CACHE_RAM_PERCENT / 100)
        .clamp(TILE_CACHE_AUTO_MIN_MB as u64, TILE_CACHE_MAX_MB as u64) as u32;
    let overview = (tile / OVERVIEW_CACHE_DIVISOR).clamp(CACHE_MIN_MB, OVERVIEW_CACHE_MAX_MB);
    (tile, overview)
}

/// Preview tiles a computer with `total_ram_bytes` takes: one per 4 GB
/// (16 GB: 4), between 1 and 32.
pub fn preview_tiles(total_ram_bytes: u64) -> u32 {
    ((total_ram_bytes / (4 << 30)) as u32).clamp(1, 32)
}

/// Breakpoint tiles a computer with `total_ram_bytes` keeps: about 5 % of
/// its RAM at ~250 MB a tile (the images and their render) - 16 GB: 3,
/// 8 GB: 1, between 1 and 16.
pub fn breakpoint_tiles(total_ram_bytes: u64) -> u32 {
    ((total_ram_bytes / 20 / (250 << 20)) as u32).clamp(1, 16)
}

/// The undo history's RAM in MB on a computer with `total_ram_bytes` and
/// `cores`, with the viewer's caches (`tile_cache_mb`, `overview_cache_mb`)
/// in effect: 10 % of what is left after a full analysis on every core but
/// one (each worker a 4096² tile of 4 channels, ~0.5 GB -
/// [`estimate_ram_budget`]), the caches and 512 MB for the app and the
/// system; between 32 MB and 1 GB. 8 GB / 4 cores: ~510 MB; 16 GB / 22
/// cores: ~375 MB.
pub fn undo_history_mb(
    total_ram_bytes: u64,
    cores: usize,
    tile_cache_mb: u32,
    overview_cache_mb: u32,
) -> u32 {
    let worker = estimate_ram_budget(MAX_TILE_SIZE, MAX_TILE_SIZE, 4).total_bytes();
    let analysis = worker * cores.saturating_sub(1).max(1) as u64;
    let reserved = analysis
        + (tile_cache_mb as u64 + overview_cache_mb as u64 + APP_AND_SYSTEM_RESERVE_MB) * MB;
    let spare_mb = total_ram_bytes.saturating_sub(reserved) / MB;
    (spare_mb * UNDO_PERCENT_OF_SPARE / 100).clamp(UNDO_MIN_MB, UNDO_MAX_MB) as u32
}
