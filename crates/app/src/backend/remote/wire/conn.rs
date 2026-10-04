//! The socket loop shared by client and server.
//!
//! A `tungstenite` WebSocket can't be split into independent read and write
//! halves, so one thread owns it: it alternates between flushing queued
//! outgoing frames and reading with a short timeout. Every other thread only
//! talks to it through a channel.

use std::io::{ErrorKind, Read, Write};
use std::net::TcpStream;
use std::sync::mpsc::{Receiver, TryRecvError};
use std::time::Duration;
use tungstenite::{Message, WebSocket};

/// Upper bound on how long a queued outgoing frame waits while the loop is
/// blocked reading an idle socket.
const POLL_INTERVAL: Duration = Duration::from_millis(5);

/// Largest message accepted from an authenticated peer - a full-resolution
/// multi-channel tile stays well below this.
pub(crate) const MAX_MESSAGE_SIZE: usize = 1 << 30;

/// Limit before the handshake, so an unauthenticated peer can't make us allocate large buffers.
pub(crate) const HANDSHAKE_MESSAGE_SIZE: usize = 64 * 1024;

/// A stream a WebSocket runs over - plain TCP, or TLS on top of it - whose
/// TCP socket can be reached for timeouts.
pub(crate) trait Socket: Read + Write {
    fn tcp(&self) -> &TcpStream;
}

impl Socket for TcpStream {
    fn tcp(&self) -> &TcpStream {
        self
    }
}

pub(crate) fn set_message_limit<S: Socket>(ws: &mut WebSocket<S>, limit: usize) {
    ws.set_config(|c| {
        c.max_message_size = Some(limit);
        c.max_frame_size = Some(limit);
    });
}

/// Runs until the peer closes, the connection fails, every sender of
/// `outgoing` is dropped, or `on_frame` returns `false`.
pub(crate) fn run_io<S: Socket>(
    mut ws: WebSocket<S>,
    outgoing: Receiver<Vec<u8>>,
    mut on_frame: impl FnMut(Vec<u8>) -> bool,
) {
    if let Err(e) = ws.get_ref().tcp().set_read_timeout(Some(POLL_INTERVAL)) {
        log::warn!("Could not set socket read timeout: {e}");
        return;
    }
    loop {
        loop {
            match outgoing.try_recv() {
                Ok(frame) => {
                    if let Err(e) = ws.write(Message::Binary(frame.into())) {
                        log::warn!("Connection write failed: {e}");
                        return;
                    }
                }
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => {
                    let _ = ws.close(None);
                    let _ = ws.flush();
                    return;
                }
            }
        }
        if let Err(e) = ws.flush() {
            if !is_timeout(&e) {
                log::warn!("Connection flush failed: {e}");
                return;
            }
        }
        match ws.read() {
            Ok(Message::Binary(bytes)) => {
                if !on_frame(bytes.to_vec()) {
                    return;
                }
            }
            Ok(Message::Close(_)) => return,
            Ok(_) => {}
            Err(e) if is_timeout(&e) => {}
            Err(tungstenite::Error::ConnectionClosed) => return,
            Err(e) => {
                log::warn!("Connection read failed: {e}");
                return;
            }
        }
    }
}

fn is_timeout(e: &tungstenite::Error) -> bool {
    matches!(e, tungstenite::Error::Io(io) if matches!(io.kind(), ErrorKind::WouldBlock | ErrorKind::TimedOut))
}

/// Blocking read of one binary message, for the handshake before
/// [`run_io`] takes over.
pub(crate) fn read_binary<S: Socket>(ws: &mut WebSocket<S>) -> Result<Vec<u8>, String> {
    loop {
        match ws.read() {
            Ok(Message::Binary(bytes)) => return Ok(bytes.to_vec()),
            Ok(Message::Close(_)) => return Err("connection closed".into()),
            Ok(_) => continue,
            Err(e) if is_timeout(&e) => return Err("timed out".into()),
            Err(e) => return Err(e.to_string()),
        }
    }
}
