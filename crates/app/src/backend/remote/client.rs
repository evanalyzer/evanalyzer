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
mod remote_backend;
mod results;
mod session;
mod tls;

pub use remote_backend::RemoteBackend;
pub use tls::TlsTrust;
