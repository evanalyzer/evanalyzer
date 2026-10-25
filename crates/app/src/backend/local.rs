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

mod filesystem;
pub(crate) mod image_reader;
pub(crate) mod job;
mod local_backend;
pub mod results;
pub(crate) mod system;
pub(crate) mod templates;
pub(crate) mod training;

pub use filesystem::LocalFileSystem;
pub use local_backend::LocalBackend;
pub use results::ResultsGenerator;
