//! # conn
//!
//! The socket loop shared by client and server.
//!
//! A `tungstenite` WebSocket can't be split into independent read and write
//! halves, so one thread owns it: it alternates between flushing queued
//! outgoing frames and reading with a short timeout. Every other thread only
//! talks to it through a channel.
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-10-10
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.

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
                // Asked to close (see `Session::close`): real frames are
                // never empty.
                Ok(frame) if frame.is_empty() => {
                    let _ = ws.close(None);
                    let _ = ws.flush();
                    return;
                }
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::TcpListener;
    use std::sync::mpsc;
    use std::thread::JoinHandle;

    /// Both ends of a WebSocket over loopback: (server, client).
    fn pair() -> (WebSocket<TcpStream>, WebSocket<TcpStream>) {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let server = std::thread::spawn(move || {
            let (stream, _) = listener.accept().unwrap();
            tungstenite::accept(stream).unwrap()
        });
        let stream = TcpStream::connect(addr).unwrap();
        let (client, _) = tungstenite::client(format!("ws://{addr}/"), stream).unwrap();
        (server.join().unwrap(), client)
    }

    /// `run_io` on the server end in its own thread; every frame it is
    /// handed goes to the returned receiver, and it stops after `stop_after`.
    fn serve(
        server: WebSocket<TcpStream>,
        outgoing: Receiver<Vec<u8>>,
        stop_after: usize,
    ) -> (JoinHandle<()>, Receiver<Vec<u8>>) {
        let (frames, frames_rx) = mpsc::channel();
        let handle = std::thread::spawn(move || {
            let mut seen = 0;
            run_io(server, outgoing, |frame| {
                frames.send(frame).unwrap();
                seen += 1;
                seen < stop_after
            });
        });
        (handle, frames_rx)
    }

    /// The next message the client gets that is not a ping or pong.
    fn next(client: &mut WebSocket<TcpStream>) -> Result<Message, tungstenite::Error> {
        loop {
            match client.read() {
                Ok(Message::Ping(_) | Message::Pong(_)) => continue,
                other => return other,
            }
        }
    }

    #[test]
    fn queued_frames_go_out_and_binary_frames_come_in_until_told_to_stop() {
        let (server, mut client) = pair();
        let (tx, rx) = mpsc::channel();
        tx.send(vec![9, 8]).unwrap();
        let (io, frames) = serve(server, rx, 1);

        assert_eq!(next(&mut client).unwrap(), Message::binary(vec![9, 8]));
        // Text is not part of the protocol: skipped.
        client.send(Message::text("ignored")).unwrap();
        client.send(Message::binary(vec![1, 2, 3])).unwrap();

        assert_eq!(frames.recv().unwrap(), vec![1, 2, 3]);
        io.join().unwrap();
        assert!(frames.try_recv().is_err(), "only the binary frame");
        drop(tx);
    }

    #[test]
    fn an_empty_frame_closes_the_connection() {
        let (server, mut client) = pair();
        let (tx, rx) = mpsc::channel();
        tx.send(Vec::new()).unwrap();
        let (io, _) = serve(server, rx, usize::MAX);

        io.join().unwrap();
        assert!(matches!(next(&mut client), Ok(Message::Close(_))));
        drop(tx);
    }

    #[test]
    fn dropping_every_sender_closes_the_connection() {
        let (server, mut client) = pair();
        let (tx, rx) = mpsc::channel::<Vec<u8>>();
        let (io, _) = serve(server, rx, usize::MAX);

        drop(tx);

        io.join().unwrap();
        assert!(matches!(next(&mut client), Ok(Message::Close(_))));
    }

    #[test]
    fn the_loop_ends_when_the_peer_closes() {
        let (server, mut client) = pair();
        let (tx, rx) = mpsc::channel();
        let (io, frames) = serve(server, rx, usize::MAX);

        client.close(None).unwrap();
        client.flush().unwrap();

        io.join().unwrap();
        assert!(frames.try_recv().is_err());
        drop(tx);
    }

    #[test]
    fn a_peer_vanishing_without_a_close_ends_the_loop() {
        let (server, client) = pair();
        let (tx, rx) = mpsc::channel();
        let (io, _) = serve(server, rx, usize::MAX);

        client.get_ref().shutdown(std::net::Shutdown::Both).unwrap();
        drop(client);

        io.join().unwrap();
        drop(tx);
    }

    #[test]
    fn writing_to_a_vanished_peer_ends_the_loop() {
        let (server, client) = pair();
        client.get_ref().shutdown(std::net::Shutdown::Both).unwrap();
        drop(client);
        let (tx, rx) = mpsc::channel();
        // Large enough that writing can't just disappear into socket buffers.
        for _ in 0..8 {
            tx.send(vec![7; 4 << 20]).unwrap();
        }
        let (io, frames) = serve(server, rx, usize::MAX);

        io.join().unwrap();
        assert!(frames.try_recv().is_err());
    }

    #[test]
    fn the_handshake_read_skips_text_and_returns_the_first_binary_message() {
        let (mut server, mut client) = pair();
        client.send(Message::text("hello")).unwrap();
        client.send(Message::binary(vec![4, 2])).unwrap();

        assert_eq!(read_binary(&mut server).unwrap(), vec![4, 2]);
    }

    #[test]
    fn the_handshake_read_reports_a_close_a_timeout_and_a_broken_connection() {
        let (mut server, mut client) = pair();
        server
            .get_ref()
            .set_read_timeout(Some(Duration::from_millis(20)))
            .unwrap();
        assert_eq!(read_binary(&mut server).unwrap_err(), "timed out");

        client.close(None).unwrap();
        client.flush().unwrap();
        assert_eq!(read_binary(&mut server).unwrap_err(), "connection closed");

        let (mut server, client) = pair();
        client.get_ref().shutdown(std::net::Shutdown::Both).unwrap();
        drop(client);
        let error = read_binary(&mut server).unwrap_err();
        assert!(
            error != "timed out" && error != "connection closed" && !error.is_empty(),
            "{error}"
        );
    }

    /// What the server end, limited to [`HANDSHAKE_MESSAGE_SIZE`], makes of
    /// a binary message of `len` bytes.
    fn read_limited(len: usize) -> Result<Vec<u8>, String> {
        let (mut server, mut client) = pair();
        set_message_limit(&mut server, HANDSHAKE_MESSAGE_SIZE);
        // Sent from its own thread: it may not fit into the socket buffers.
        let sender = std::thread::spawn(move || {
            let _ = client.send(Message::binary(vec![1; len]));
            client
        });
        let result = read_binary(&mut server);
        drop(server);
        let _ = sender.join();
        result
    }

    #[test]
    fn a_message_over_the_limit_is_refused() {
        let error = read_limited(HANDSHAKE_MESSAGE_SIZE + 1).unwrap_err();
        assert!(error.to_lowercase().contains("too"), "{error}");

        assert_eq!(
            read_limited(HANDSHAKE_MESSAGE_SIZE).unwrap().len(),
            HANDSHAKE_MESSAGE_SIZE
        );
    }
}
