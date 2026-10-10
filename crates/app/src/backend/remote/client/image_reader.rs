//! [`RemoteImageSource`]: an image opened on the worker; tiles are fetched
//! on demand. Survives a reconnect: opened again on the new connection.

use super::link::{Link, Reopen};
use super::session::{Session, unexpected_reply};
use crate::api::ImageChannel;
use crate::api::ImageMeta;
use crate::api::ImageSource;
use crate::api::TileRequest;
use crate::backend::remote::wire::frame::Frame;
use crate::backend::remote::wire::protocol::{ClientMsg, Reply, Request, channels_from_wire};
use evanalyzer_cfg::core_types::InternalErrors;
use std::path::{Path, PathBuf};
use std::sync::Arc;

pub(super) struct RemoteImageSource {
    path: PathBuf,
    handle: Reopen,
    meta: ImageMeta,
}

/// Opens `path` on `session`: its handle and metadata.
pub(super) fn open(session: &Session, path: &Path) -> Result<(u64, ImageMeta), InternalErrors> {
    let (id, rx) = session.request(Request::OpenImage {
        path: path.to_path_buf(),
    })?;
    let reply = session.recv(&rx);
    session.finish(id);
    match reply?.msg {
        Reply::ImageOpened { handle, meta } => Ok((handle, meta)),
        Reply::Failed(e) => Err(e.into_internal()),
        _ => Err(unexpected_reply()),
    }
}

impl RemoteImageSource {
    /// Opens `path` on the link's connection.
    pub(super) fn open(link: &Arc<Link>, path: &Path) -> Result<Self, InternalErrors> {
        let (session, generation) = link.current();
        let (handle, meta) = open(&session, path)?;
        Ok(Self {
            path: path.to_path_buf(),
            handle: Reopen::new(Arc::clone(link), generation, handle),
            meta,
        })
    }
}

impl ImageSource for RemoteImageSource {
    fn meta(&self) -> &ImageMeta {
        &self.meta
    }

    fn read_tile(&self, req: &TileRequest) -> Result<Vec<ImageChannel>, InternalErrors> {
        let (session, handle) = self
            .handle
            .get(|session| open(session, &self.path).map(|(handle, _)| handle))?;
        let (id, rx) = session.request(Request::ReadTile {
            handle,
            tile: req.clone(),
        })?;
        let reply = session.recv(&rx);
        session.finish(id);
        let Frame { msg, blobs } = reply?;
        match msg {
            Reply::Tile(channels) => channels_from_wire(channels, blobs),
            Reply::Failed(e) => Err(e.into_internal()),
            _ => Err(unexpected_reply()),
        }
    }
}

impl Drop for RemoteImageSource {
    fn drop(&mut self) {
        if let Some((session, handle)) = self.handle.to_close() {
            let _ = session.send(&ClientMsg::CloseImage { handle });
        }
    }
}
