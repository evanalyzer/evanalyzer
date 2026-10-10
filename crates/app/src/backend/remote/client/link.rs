//! [`Link`]: the client's current connection to the worker, replaceable
//! by a new one when it drops ([`Link::reconnect`]).
//!
//! Everything that talks to the worker goes through the link instead of
//! holding a [`Session`] itself, so after a reconnect it uses the new
//! connection. Images and results databases opened on the worker are tied
//! to the connection that opened them; [`Reopen`] opens them again on the
//! new one, by path, the first time they're used after a reconnect.

use super::session::Session;
use evanalyzer_cfg::core_types::InternalErrors;
use std::sync::{Arc, Mutex, RwLock};

/// Opens a new connection to the same worker, authenticated the same way.
pub(super) type Connect = Box<dyn Fn() -> Result<Arc<Session>, InternalErrors> + Send + Sync>;

pub(super) struct Link {
    /// The connection, and how many reconnects ago it was opened.
    current: RwLock<(Arc<Session>, u64)>,
    connect: Connect,
    /// One reconnect at a time.
    reconnecting: Mutex<()>,
}

impl Link {
    pub(super) fn new(session: Arc<Session>, connect: Connect) -> Arc<Self> {
        Arc::new(Self {
            current: RwLock::new((session, 0)),
            connect,
            reconnecting: Mutex::new(()),
        })
    }

    pub(super) fn session(&self) -> Arc<Session> {
        Arc::clone(&self.current.read().unwrap().0)
    }

    /// The connection with its generation, which counts the reconnects.
    pub(super) fn current(&self) -> (Arc<Session>, u64) {
        let current = self.current.read().unwrap();
        (Arc::clone(&current.0), current.1)
    }

    /// Replaces a dropped connection with a new one. Nothing to do while the
    /// connection is up (e.g. another thread reconnected meanwhile).
    pub(super) fn reconnect(&self) -> Result<(), InternalErrors> {
        let _one_at_a_time = self.reconnecting.lock().unwrap();
        if self.session().is_connected() {
            return Ok(());
        }
        let session = (self.connect)()?;
        let mut current = self.current.write().unwrap();
        let generation = current.1 + 1;
        *current = (session, generation);
        log::info!("Reconnected to {}", current.0.url());
        Ok(())
    }
}

/// A handle to something opened on the worker (an image, a results
/// database), valid on the connection of one generation only; opened
/// again with `open` when the link has reconnected since.
pub(super) struct Reopen {
    link: Arc<Link>,
    handle: Mutex<(u64, u64)>,
}

impl Reopen {
    /// `handle`, opened on the link's current connection.
    pub(super) fn new(link: Arc<Link>, generation: u64, handle: u64) -> Self {
        Self {
            link,
            handle: Mutex::new((generation, handle)),
        }
    }

    /// The connection to use and the handle valid on it, opening it again
    /// with `open` first if the link has reconnected since.
    pub(super) fn get(
        &self,
        open: impl FnOnce(&Session) -> Result<u64, InternalErrors>,
    ) -> Result<(Arc<Session>, u64), InternalErrors> {
        let (session, generation) = self.link.current();
        let mut handle = self.handle.lock().unwrap();
        if handle.0 != generation {
            *handle = (generation, open(&session)?);
        }
        Ok((session, handle.1))
    }

    /// The connection and handle to close, if the handle is still valid.
    pub(super) fn to_close(&self) -> Option<(Arc<Session>, u64)> {
        let (session, generation) = self.link.current();
        let handle = self.handle.lock().unwrap();
        (handle.0 == generation).then_some((session, handle.1))
    }
}
