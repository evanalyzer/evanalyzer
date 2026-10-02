//! The contract between front ends (GUI, CLI) and backends: the
//! [`Backend`] traits and every type a request or result carries. Front ends
//! only use this - whether the backend runs in this process or on a server
//! is decided once at startup.
//!
//! Meant to become its own crate for a WebAssembly client; the few `core`
//! re-exports left here (images, `ProgressEvent`, `TrainedClassifier`'s
//! inner model) are what still stands in the way.

mod backend;
mod files;
mod images;
mod jobs;
mod results;
mod training;

pub use backend::*;
pub use files::*;
pub use images::*;
pub use jobs::*;
pub use results::*;
pub use training::*;
