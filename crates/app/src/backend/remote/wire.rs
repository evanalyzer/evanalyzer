//! The transport [`RemoteBackend`](super::RemoteBackend) (client) and
//! [`Worker`](super::Worker) (server side) share: the WebSocket connection,
//! the framing of requests and answers on it, the messages themselves, and
//! the binary encoding of images inside them.

pub(super) mod conn;
pub(super) mod frame;
pub(crate) mod pixels;
pub(super) mod protocol;
