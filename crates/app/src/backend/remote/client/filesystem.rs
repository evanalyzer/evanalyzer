//! [`RemoteFiles`]: the worker's file system, one request per call.

use super::session::{Session, unexpected_reply};
use crate::api::DirEntry;
use crate::api::FileSystem;
use crate::api::Place;
use crate::backend::remote::wire::frame::Frame;
use crate::backend::remote::wire::protocol::{Reply, Request};
use evanalyzer_cfg::core_types::InternalErrors;
use std::path::Path;
use std::sync::Arc;

pub(super) struct RemoteFiles {
    session: Arc<Session>,
}

impl RemoteFiles {
    pub(super) fn new(session: Arc<Session>) -> Self {
        Self { session }
    }

    /// Sends `request` and waits for its single reply; a `Failed` reply
    /// becomes the error.
    pub(super) fn call(
        &self,
        request: Request,
        blobs: Vec<Vec<u8>>,
    ) -> Result<Frame<Reply>, InternalErrors> {
        let (id, rx) = self.session.request_with_blobs(request, blobs)?;
        let reply = self.session.recv(&rx);
        self.session.finish(id);
        match reply? {
            Frame {
                msg: Reply::Failed(e),
                ..
            } => Err(e.into_internal()),
            frame => Ok(frame),
        }
    }
}

impl FileSystem for RemoteFiles {
    fn places(&self) -> Result<Vec<Place>, InternalErrors> {
        match self.call(Request::Places, Vec::new())?.msg {
            Reply::Places(places) => Ok(places),
            _ => Err(unexpected_reply()),
        }
    }

    fn list_dir(&self, dir: &Path) -> Result<Vec<DirEntry>, InternalErrors> {
        let request = Request::ListDir {
            path: dir.to_path_buf(),
        };
        match self.call(request, Vec::new())?.msg {
            Reply::DirEntries(entries) => Ok(entries),
            _ => Err(unexpected_reply()),
        }
    }

    fn stat(&self, path: &Path) -> Result<Option<DirEntry>, InternalErrors> {
        let request = Request::Stat {
            path: path.to_path_buf(),
        };
        match self.call(request, Vec::new())?.msg {
            Reply::Stat(entry) => Ok(entry),
            _ => Err(unexpected_reply()),
        }
    }

    fn read_file(&self, path: &Path) -> Result<Vec<u8>, InternalErrors> {
        let request = Request::ReadFile {
            path: path.to_path_buf(),
        };
        match self.call(request, Vec::new())? {
            Frame {
                msg: Reply::FileData,
                mut blobs,
            } if blobs.len() == 1 => Ok(blobs.remove(0)),
            _ => Err(unexpected_reply()),
        }
    }

    fn write_file(&self, path: &Path, data: &[u8]) -> Result<(), InternalErrors> {
        let request = Request::WriteFile {
            path: path.to_path_buf(),
        };
        match self.call(request, vec![data.to_vec()])?.msg {
            Reply::Done => Ok(()),
            _ => Err(unexpected_reply()),
        }
    }

    fn create_dir_all(&self, path: &Path) -> Result<(), InternalErrors> {
        let request = Request::CreateDir {
            path: path.to_path_buf(),
        };
        match self.call(request, Vec::new())?.msg {
            Reply::Done => Ok(()),
            _ => Err(unexpected_reply()),
        }
    }

    fn rename(&self, from: &Path, to: &Path) -> Result<(), InternalErrors> {
        let request = Request::Rename {
            from: from.to_path_buf(),
            to: to.to_path_buf(),
        };
        match self.call(request, Vec::new())?.msg {
            Reply::Done => Ok(()),
            _ => Err(unexpected_reply()),
        }
    }

    fn remove_all(&self, path: &Path) -> Result<(), InternalErrors> {
        let request = Request::RemoveAll {
            path: path.to_path_buf(),
        };
        match self.call(request, Vec::new())?.msg {
            Reply::Done => Ok(()),
            _ => Err(unexpected_reply()),
        }
    }
}
