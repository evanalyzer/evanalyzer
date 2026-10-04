//! Remote compute over WebSocket.
//!
//! - [`RemoteBackend`] (client): implements `crate::api::Backend`
//!   by sending each request to a server - front ends use it exactly like
//!   the local backend.
//! - [`Worker`]: `evanalyzer worker`, executes requests on a local backend
//!   (started per logged-in user by `evanalyzer server`, the separate
//!   multi-user gateway in `crates/server`, or by hand).
//! - [`wire`]: the transport both sides share - connection, framing,
//!   messages, image encoding.
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
pub(crate) mod wire;
mod worker;

pub use client::{RemoteBackend, TlsTrust};
pub use worker::{Worker, generate_token};
