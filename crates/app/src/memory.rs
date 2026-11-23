//! # memory
//!
//! RAM: what a command takes (the step picker shows it), formatted, and the
//! allocator that counts it - from core.
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-11-23
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.

pub use evanalyzer_core::memory::accounting::AccountingAllocator;
pub use evanalyzer_core::memory::analysis::MAX_TILE_SIZE;
pub use evanalyzer_core::memory::command_ram::{CommandRam, command_ram};
pub use evanalyzer_core::memory::format_binary_bytes;
