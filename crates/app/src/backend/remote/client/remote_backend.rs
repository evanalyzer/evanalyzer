//! [`RemoteBackend`] itself: connecting, and the `Backend` calls that start
//! work on the worker (analysis, preview, training) or open something there.

use super::filesystem::RemoteFiles;
use super::image_reader::RemoteImageSource;
use super::results::RemoteResults;
use super::session::{Session, login, open_websocket, unexpected_reply};
use crate::api::AnalysisRequest;
use crate::api::Backend;
use crate::api::FileSystem;
use crate::api::ImageMeta;
use crate::api::ImageSource;
use crate::api::PreviewRequest;
use crate::api::ResultsSource;
use crate::api::RunningJob;
use crate::api::RunningTraining;
use crate::api::StartPreviewError;
use crate::api::StartTrainingError;
use crate::api::TemplateFolders;
use crate::api::TrainedClassifier;
use crate::api::TrainingRequest;
use crate::backend::remote::wire::frame::Frame;
use crate::backend::remote::wire::protocol::{Reply, Request};
use evanalyzer_cfg::core_types::InternalErrors;
use std::path::Path;
use std::sync::Arc;
use std::sync::mpsc;

pub struct RemoteBackend {
    session: Arc<Session>,
    files: RemoteFiles,
    user: Option<String>,
}

impl RemoteBackend {
    /// Connects to `url` (`ws://host[:port]`) and authenticates with
    /// `token`. Fails with a readable message if the server is unreachable,
    /// rejects the token, or runs a different version.
    pub fn connect(url: &str, token: &str) -> Result<Self, InternalErrors> {
        let ws = open_websocket(url)?;
        Ok(Self::new(Session::open(ws, url, token)?, None))
    }

    /// Connects to an `evanalyzer server` (`ws://host[:port]`), logs in as
    /// `username` and attaches to the user's worker, which the server starts
    /// if it isn't running yet.
    pub fn connect_with_login(
        url: &str,
        username: &str,
        password: &str,
    ) -> Result<Self, InternalErrors> {
        let mut ws = open_websocket(url)?;
        let session_token = login(&mut ws, url, username, password)?;
        // From here on the server forwards everything to the worker, which
        // accepts the session token in `Hello`.
        let session = Session::open(ws, url, &session_token)?;
        Ok(Self::new(session, Some(username.to_string())))
    }

    fn new(session: Arc<Session>, user: Option<String>) -> Self {
        Self {
            files: RemoteFiles::new(Arc::clone(&session)),
            session,
            user,
        }
    }
}

impl Backend for RemoteBackend {
    fn start_analysis(&self, req: AnalysisRequest) -> Result<RunningJob, InternalErrors> {
        let session = &self.session;
        let (id, rx) = session.request(Request::StartAnalysis(req))?;
        match session.recv(&rx)?.msg {
            Reply::JobStarted {
                output_path,
                parallelism,
            } => Ok(session.running_job(id, rx, output_path, parallelism)),
            Reply::Failed(e) => {
                session.finish(id);
                Err(e.into_internal())
            }
            _ => {
                session.finish(id);
                Err(unexpected_reply())
            }
        }
    }

    fn start_preview(&self, req: PreviewRequest) -> Result<RunningJob, StartPreviewError> {
        let session = &self.session;
        let (id, rx) = session.request(Request::StartPreview(req))?;
        let reply = session.recv(&rx)?.msg;
        if !matches!(reply, Reply::JobStarted { .. }) {
            session.finish(id);
        }
        match reply {
            Reply::JobStarted {
                output_path,
                parallelism,
            } => Ok(session.running_job(id, rx, output_path, parallelism)),
            Reply::PreviewTooManyTiles { tiles } => Err(StartPreviewError::TooManyTiles { tiles }),
            Reply::Failed(e) => Err(StartPreviewError::Failed(e.into_internal())),
            _ => Err(StartPreviewError::Failed(unexpected_reply())),
        }
    }

    fn start_training(&self, req: TrainingRequest) -> Result<RunningTraining, StartTrainingError> {
        let session = &self.session;
        let (id, rx) = session
            .request(Request::StartTraining(req))
            .map_err(StartTrainingError::Failed)?;
        let reply = session.recv(&rx).map_err(StartTrainingError::Failed)?.msg;
        let items = match reply {
            Reply::TrainingStarted { items } => items,
            other => {
                session.finish(id);
                return Err(match other {
                    Reply::NoTrainingData => StartTrainingError::NoTrainingData,
                    Reply::Failed(e) => StartTrainingError::Failed(e.into_internal()),
                    _ => StartTrainingError::Failed(unexpected_reply()),
                });
            }
        };

        let (events_tx, events_rx) = mpsc::channel();
        let (done_tx, done_rx) = mpsc::channel();
        let pump = Arc::clone(session);
        std::thread::spawn(move || {
            let result = loop {
                match pump.recv(&rx) {
                    Ok(Frame {
                        msg: Reply::TrainingEvent(event),
                        ..
                    }) => {
                        let _ = events_tx.send(event);
                    }
                    Ok(Frame {
                        msg: Reply::TrainingDone(Ok(())),
                        blobs,
                    }) => {
                        break match blobs.first() {
                            Some(model) => TrainedClassifier::from_bytes(model),
                            None => Err(InternalErrors::Internal(
                                "server sent no trained model".into(),
                            )),
                        };
                    }
                    Ok(Frame {
                        msg: Reply::TrainingDone(Err(e)) | Reply::Failed(e),
                        ..
                    }) => break Err(e.into_internal()),
                    Ok(_) => break Err(unexpected_reply()),
                    Err(e) => break Err(e),
                }
            };
            pump.finish(id);
            drop(events_tx);
            let _ = done_tx.send(result);
        });
        let disconnected = session.disconnected();
        Ok(RunningTraining::from_parts(
            events_rx,
            session.cancel_handle(id),
            items,
            Box::new(move || done_rx.recv().unwrap_or(Err(disconnected))),
        ))
    }

    fn open_image(&self, path: &Path) -> Result<Arc<dyn ImageSource>, InternalErrors> {
        let session = &self.session;
        let (id, rx) = session.request(Request::OpenImage {
            path: path.to_path_buf(),
        })?;
        let reply = session.recv(&rx);
        session.finish(id);
        match reply?.msg {
            Reply::ImageOpened { handle, meta } => Ok(Arc::new(RemoteImageSource::new(
                Arc::clone(session),
                handle,
                meta,
            ))),
            Reply::Failed(e) => Err(e.into_internal()),
            _ => Err(unexpected_reply()),
        }
    }

    fn open_results(&self, path: &Path) -> Result<Arc<dyn ResultsSource>, InternalErrors> {
        let request = Request::OpenResults {
            path: path.to_path_buf(),
        };
        match self.files.call(request, Vec::new())?.msg {
            Reply::ResultsOpened { handle } => Ok(Arc::new(RemoteResults::new(
                Arc::clone(&self.session),
                handle,
            ))),
            _ => Err(unexpected_reply()),
        }
    }

    fn read_image_meta(&self, path: &Path) -> Result<ImageMeta, InternalErrors> {
        let request = Request::ReadImageMeta {
            path: path.to_path_buf(),
        };
        match self.files.call(request, Vec::new())?.msg {
            Reply::ImageMeta(meta) => Ok(meta),
            _ => Err(unexpected_reply()),
        }
    }

    fn template_folders(&self) -> Result<TemplateFolders, InternalErrors> {
        match self.files.call(Request::TemplateFolders, Vec::new())?.msg {
            Reply::TemplateFolders(folders) => Ok(folders),
            _ => Err(unexpected_reply()),
        }
    }

    fn files(&self) -> &dyn FileSystem {
        &self.files
    }

    fn is_remote(&self) -> bool {
        true
    }

    fn description(&self) -> String {
        self.session.url().to_string()
    }

    fn is_connected(&self) -> bool {
        self.session.is_connected()
    }

    fn user(&self) -> Option<String> {
        self.user.clone()
    }
}
