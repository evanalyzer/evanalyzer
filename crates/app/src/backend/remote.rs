//! Remote compute over WebSocket.
//!
//! - [`RemoteBackend`] (client): implements `crate::api::Backend`
//!   by sending each request to a server - front ends use it exactly like
//!   the local backend.
//! - [`Server`]: `evanalyzer serve`, executes requests on a local backend.
//!
//! The transport is synchronous (`tungstenite` over a plain `TcpStream`), so
//! nothing async leaks into the GUI or CLI: results still arrive on the same
//! `mpsc` channels a local job uses. One connection carries any number of
//! concurrent requests (e.g. a preview job while the viewport reads tiles),
//! matched up by request id.
//!
//! Assumptions of this first version: client and server see the same files
//! under the same paths (shared storage), results are written by the server
//! into the project's results folder, and a dropped connection cancels the
//! client's running jobs (no reconnect).

mod client;
mod conn;
mod frame;
pub(crate) mod pixels;
mod protocol;
mod server;

pub use client::{DEFAULT_PORT, RemoteBackend};
pub use protocol::PROTOCOL_VERSION;
pub use server::{Server, generate_token};
