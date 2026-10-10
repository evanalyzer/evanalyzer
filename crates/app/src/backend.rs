//! The two implementations of [`crate::api::Backend`]: [`LocalBackend`]
//! runs everything in this process (calling `evanalyzer_core`), and
//! [`RemoteBackend`] sends it to an `evanalyzer worker` instance - which
//! itself runs a `LocalBackend` (see [`Server`]).

pub mod local;
pub mod remote;

pub use local::{LocalBackend, LocalFileSystem};
pub use remote::{
    RemoteBackend, ServerCertificate, TlsTrust, Worker, generate_token, server_certificate,
};
