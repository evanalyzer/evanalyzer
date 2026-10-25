//! # backend
//!
//! The two implementations of [`crate::api::Backend`]: [`LocalBackend`]
//! runs everything in this process (calling `evanalyzer_core`), and
//! [`RemoteBackend`] sends it to an `evanalyzer worker` instance - which
//! itself runs a `LocalBackend` (see [`Server`]).
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-10-10
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.

pub mod local;
pub mod remote;

pub use local::{LocalBackend, LocalFileSystem};
pub use remote::{
    RemoteBackend, ServerCertificate, TlsTrust, Worker, generate_token, server_certificate,
};
