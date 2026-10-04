//! [`RemoteImageSource`]: an image opened on the worker; tiles are fetched
//! on demand.

use super::session::{Session, unexpected_reply};
use crate::api::ImageChannel;
use crate::api::ImageMeta;
use crate::api::ImageSource;
use crate::api::TileRequest;
use crate::backend::remote::wire::frame::Frame;
use crate::backend::remote::wire::protocol::{ClientMsg, Reply, Request, channels_from_wire};
use evanalyzer_cfg::core_types::InternalErrors;
use std::sync::Arc;

pub(super) struct RemoteImageSource {
    session: Arc<Session>,
    handle: u64,
    meta: ImageMeta,
}

impl RemoteImageSource {
    pub(super) fn new(session: Arc<Session>, handle: u64, meta: ImageMeta) -> Self {
        Self {
            session,
            handle,
            meta,
        }
    }
}

impl ImageSource for RemoteImageSource {
    fn meta(&self) -> &ImageMeta {
        &self.meta
    }

    fn read_tile(&self, req: &TileRequest) -> Result<Vec<ImageChannel>, InternalErrors> {
        let session = &self.session;
        let (id, rx) = session.request(Request::ReadTile {
            handle: self.handle,
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
        let _ = self.session.send(&ClientMsg::CloseImage {
            handle: self.handle,
        });
    }
}
