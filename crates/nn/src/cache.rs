//! # cache
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-11-25
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.
//!
//! P3: the loaded models - per (file, device), dropped when the file changes
//! (modification time) and by `release_models` after every analysis. One
//! loaded model is shared by every step using that file, whatever their
//! settings. Slots per GPU; a run blocks while every device is busy.
//! Entries are `Arc<dyn Any + Send + Sync>`, cast back to the `Model` type -
//! the only dynamic part of the model interface.
//! Ported from `crates/ai/src/torch/scheduler.rs` (its framework-free parts
//! and tests).
