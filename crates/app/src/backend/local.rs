//! # local
//!
//! Runs everything in this process - the only part of the app that calls
//! `evanalyzer_core` (and DuckDB) to do the actual work.
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-10-10
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.

mod local_backend;
pub(crate) mod system;
pub(crate) mod templates;

pub use local_backend::LocalBackend;
