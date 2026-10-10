//! [`RemoteBackend`]: a [`Backend`](crate::api::Backend) that runs everything
//! on an `evanalyzer worker` instance. Front ends use it exactly like the
//! local backend - jobs still arrive as `RunningJob`s with an event channel.
//!
//! Images and results are referenced by path, so client and server must see
//! the same files under the same paths (shared storage).
//!
//! Laid out like `backend::local`: [`session`] holds the connection and
//! matches requests with replies; the other modules adapt one API trait each
//! onto it.

mod filesystem;
mod image_reader;
mod link;
mod remote_backend;
mod results;
mod session;
mod tls;

pub use remote_backend::RemoteBackend;
pub use tls::{ServerCertificate, TlsTrust};

/// The TLS certificate of the server at `url` (`wss://`), without trusting
/// it - for asking the user. `None` for `ws://`.
pub fn server_certificate(
    url: &str,
) -> Result<Option<ServerCertificate>, evanalyzer_cfg::core_types::InternalErrors> {
    session::server_certificate(url)
}
