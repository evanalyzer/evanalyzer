//! Runs everything in this process - the only part of the app that calls
//! `evanalyzer_core` (and DuckDB) to do the actual work.

mod filesystem;
mod image_reader;
pub(crate) mod job;
mod local_backend;
pub mod results;
pub(crate) mod system;
pub(crate) mod templates;
pub(crate) mod training;

pub use filesystem::LocalFileSystem;
pub use local_backend::LocalBackend;
pub use results::ResultsGenerator;
