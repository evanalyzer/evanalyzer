//! # files
//!
//! Files and folders where the backend computes: [`FileSystem`] - this
//! computer's ([`LocalFileSystem`]) or a server's (through `crate::remote`).
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-11-22
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.

mod local;
mod source;

pub use local::LocalFileSystem;
pub use source::*;
