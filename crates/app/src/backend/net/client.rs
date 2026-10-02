//! [`RemoteBackend`]: a [`Backend`] that runs everything on an
//! `evanalyzer serve` instance. Front ends use it exactly like the local
//! backend - jobs still arrive as `RunningJob`s with an event channel.
//!
//! Images and results are referenced by path, so client and server must see
//! the same files under the same paths (shared storage).

use crate::ai_learning::{RunningTraining, StartTrainingError, TrainedClassifier};
use crate::backend::net::conn::{self, MAX_MESSAGE_SIZE};
use crate::backend::net::frame::{self, Frame};
use crate::backend::net::protocol::{
    APP_VERSION, ClientMsg, PROTOCOL_VERSION, Reply, Request, ServerMsg, channels_from_wire,
    event_from_wire,
};
use crate::backend::net::protocol::{ResultsAnswer, ResultsQuery, from_postcard, to_postcard};
use crate::backend::{
    AnalysisRequest, Backend, DirEntry, ExportProgressFn, FileSystem, ImageSource, Place,
    ResultsSource, TemplateFolders, TileRequest, TrainingRequest,
};
use crate::images::{ImageChannel, ImageMeta};
use crate::job::{CancelHandle, PreviewRequest, RunningJob, StartPreviewError};
use crate::result::{
    BoxplotFilter, BoxplotResult, ColumnEntry, DatabaseResult, GroupedByImageFilter,
    HistogramFilter, HistogramResult, ImageEntry, ImageHeatmapFilter, ListFilter, PlateFilter,
    ResultExport, ScatterFilter, ScatterResult, View, WellFilter,
};
use evanalyzer_cfg::core_types::InternalErrors;
use evanalyzer_cfg::settings::classification_settings::Class;
use std::collections::HashMap;
use std::net::{TcpStream, ToSocketAddrs};
use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicBool;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::{Arc, Mutex};
use std::time::Duration;

/// Port used when the URL doesn't name one.
pub const DEFAULT_PORT: u16 = 7400;

const CONNECT_TIMEOUT: Duration = Duration::from_secs(10);

pub struct RemoteBackend {
    inner: Arc<Inner>,
    files: RemoteFiles,
}

impl Backend for RemoteBackend {
    fn start_analysis(&self, req: AnalysisRequest) -> Result<RunningJob, InternalErrors> {
        let inner = &self.inner;
        let (id, rx) = inner.request(Request::StartAnalysis(req))?;
        match inner.recv(&rx)?.msg {
            Reply::JobStarted {
                output_path,
                parallelism,
            } => Ok(inner.running_job(id, rx, output_path, parallelism)),
            Reply::Failed(e) => {
                inner.finish(id);
                Err(e.into_internal())
            }
            _ => {
                inner.finish(id);
                Err(unexpected_reply())
            }
        }
    }

    fn start_preview(&self, req: PreviewRequest) -> Result<RunningJob, StartPreviewError> {
        let inner = &self.inner;
        let (id, rx) = inner.request(Request::StartPreview(req))?;
        let reply = inner.recv(&rx)?.msg;
        if !matches!(reply, Reply::JobStarted { .. }) {
            inner.finish(id);
        }
        match reply {
            Reply::JobStarted {
                output_path,
                parallelism,
            } => Ok(inner.running_job(id, rx, output_path, parallelism)),
            Reply::PreviewTooManyTiles { tiles } => Err(StartPreviewError::TooManyTiles { tiles }),
            Reply::Failed(e) => Err(StartPreviewError::Failed(e.into_internal())),
            _ => Err(StartPreviewError::Failed(unexpected_reply())),
        }
    }

    fn start_training(&self, req: TrainingRequest) -> Result<RunningTraining, StartTrainingError> {
        let inner = &self.inner;
        let (id, rx) = inner
            .request(Request::StartTraining(req))
            .map_err(StartTrainingError::Failed)?;
        let reply = inner.recv(&rx).map_err(StartTrainingError::Failed)?.msg;
        let items = match reply {
            Reply::TrainingStarted { items } => items,
            other => {
                inner.finish(id);
                return Err(match other {
                    Reply::NoTrainingData => StartTrainingError::NoTrainingData,
                    Reply::Failed(e) => StartTrainingError::Failed(e.into_internal()),
                    _ => StartTrainingError::Failed(unexpected_reply()),
                });
            }
        };

        let (events_tx, events_rx) = mpsc::channel();
        let (done_tx, done_rx) = mpsc::channel();
        let pump = Arc::clone(inner);
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
        let disconnected = inner.disconnected();
        Ok(RunningTraining::from_parts(
            events_rx,
            inner.cancel_handle(id),
            items,
            Box::new(move || done_rx.recv().unwrap_or(Err(disconnected))),
        ))
    }

    fn open_image(&self, path: &Path) -> Result<Arc<dyn ImageSource>, InternalErrors> {
        let inner = &self.inner;
        let (id, rx) = inner.request(Request::OpenImage {
            path: path.to_path_buf(),
        })?;
        let reply = inner.recv(&rx);
        inner.finish(id);
        match reply?.msg {
            Reply::ImageOpened { handle, meta } => Ok(Arc::new(RemoteImageSource {
                inner: Arc::clone(inner),
                handle,
                meta,
            })),
            Reply::Failed(e) => Err(e.into_internal()),
            _ => Err(unexpected_reply()),
        }
    }

    fn open_results(&self, path: &Path) -> Result<Arc<dyn ResultsSource>, InternalErrors> {
        let request = Request::OpenResults {
            path: path.to_path_buf(),
        };
        match self.files.call(request, Vec::new())?.msg {
            Reply::ResultsOpened { handle } => Ok(Arc::new(RemoteResults {
                inner: Arc::clone(&self.inner),
                handle,
            })),
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
        self.inner.url.clone()
    }
}

/// The server's file system, one request per call.
struct RemoteFiles {
    inner: Arc<Inner>,
}

impl RemoteFiles {
    fn call(&self, request: Request, blobs: Vec<Vec<u8>>) -> Result<Frame<Reply>, InternalErrors> {
        let (id, rx) = self.inner.request_with_blobs(request, blobs)?;
        let reply = self.inner.recv(&rx);
        self.inner.finish(id);
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

struct Inner {
    url: String,
    outgoing: Sender<Vec<u8>>,
    /// Requests waiting for replies, by id. Emptied when the connection
    /// drops, which turns every pending `recv` into a "connection lost".
    pending: Mutex<HashMap<u64, Sender<Frame<Reply>>>>,
    next_id: AtomicU64,
}

impl RemoteBackend {
    /// Connects to `url` (`ws://host[:port]`) and authenticates with
    /// `token`. Fails with a readable message if the server is unreachable,
    /// rejects the token, or runs a different version.
    pub fn connect(url: &str, token: &str) -> Result<Self, InternalErrors> {
        let uri: tungstenite::http::Uri = url.parse().map_err(|e| {
            InternalErrors::InvalidArgument(format!("Invalid server URL '{url}': {e}"))
        })?;
        match uri.scheme_str() {
            Some("ws") => {}
            Some("wss") => {
                return Err(InternalErrors::InvalidArgument(
                    "wss:// is not supported - use ws:// through an SSH tunnel or VPN".into(),
                ));
            }
            _ => {
                return Err(InternalErrors::InvalidArgument(format!(
                    "Server URL must start with ws:// (got '{url}')"
                )));
            }
        }
        let host = uri
            .host()
            .ok_or_else(|| InternalErrors::InvalidArgument(format!("No host in '{url}'")))?;
        let port = uri.port_u16().unwrap_or(DEFAULT_PORT);
        let request_url = format!("ws://{host}:{port}/");

        let stream = connect_tcp(host, port)?;
        stream.set_nodelay(true).ok();
        stream.set_read_timeout(Some(CONNECT_TIMEOUT))?;
        let config = tungstenite::protocol::WebSocketConfig::default()
            .max_message_size(Some(MAX_MESSAGE_SIZE))
            .max_frame_size(Some(MAX_MESSAGE_SIZE));
        let (mut ws, _) =
            tungstenite::client::client_with_config(request_url.as_str(), stream, Some(config))
                .map_err(|e| {
                    InternalErrors::Io(format!("WebSocket handshake with {url} failed: {e}"))
                })?;

        let hello = frame::encode(
            &ClientMsg::Hello {
                protocol_version: PROTOCOL_VERSION,
                app_version: APP_VERSION.into(),
                token: token.into(),
            },
            &[],
        )?;
        ws.send(tungstenite::Message::Binary(hello.into()))
            .map_err(|e| InternalErrors::Io(format!("Could not reach {url}: {e}")))?;
        let answer = conn::read_binary(&mut ws)
            .map_err(|e| InternalErrors::Io(format!("No answer from {url}: {e}")))?;
        match frame::decode::<ServerMsg>(&answer)?.msg {
            ServerMsg::Welcome { app_version } => {
                log::info!("Connected to EVAnalyzer {app_version} server at {url}");
            }
            ServerMsg::Rejected { reason } => {
                return Err(InternalErrors::InvalidArgument(format!(
                    "Server {url} refused the connection: {reason}"
                )));
            }
            ServerMsg::Reply { .. } => {
                return Err(InternalErrors::Internal(
                    "unexpected reply before handshake".into(),
                ));
            }
        }

        let (outgoing, outgoing_rx) = mpsc::channel();
        let inner = Arc::new(Inner {
            url: url.into(),
            outgoing,
            pending: Mutex::new(HashMap::new()),
            next_id: AtomicU64::new(1),
        });
        // The I/O thread holds only a weak reference: dropping the last
        // `RemoteBackend` (and with it the outgoing sender) ends the thread.
        let weak = Arc::downgrade(&inner);
        std::thread::Builder::new()
            .name("evanalyzer-net-client".into())
            .spawn(move || {
                conn::run_io(ws, outgoing_rx, |bytes| {
                    let Some(inner) = weak.upgrade() else {
                        return false;
                    };
                    inner.dispatch(&bytes)
                });
                if let Some(inner) = weak.upgrade() {
                    log::warn!("Lost connection to server {}", inner.url);
                    inner.pending.lock().unwrap().clear();
                }
            })?;
        Ok(Self {
            files: RemoteFiles {
                inner: Arc::clone(&inner),
            },
            inner,
        })
    }
}

fn connect_tcp(host: &str, port: u16) -> Result<TcpStream, InternalErrors> {
    let addrs = (host, port)
        .to_socket_addrs()
        .map_err(|e| InternalErrors::Io(format!("Could not resolve {host}: {e}")))?;
    let mut last_error = None;
    for addr in addrs {
        match TcpStream::connect_timeout(&addr, CONNECT_TIMEOUT) {
            Ok(stream) => return Ok(stream),
            Err(e) => last_error = Some(e),
        }
    }
    Err(InternalErrors::Io(match last_error {
        Some(e) => format!("Could not connect to {host}:{port}: {e}"),
        None => format!("{host} did not resolve to any address"),
    }))
}

impl Inner {
    fn dispatch(&self, bytes: &[u8]) -> bool {
        let frame = match frame::decode::<ServerMsg>(bytes) {
            Ok(frame) => frame,
            Err(e) => {
                log::warn!("Dropping connection after malformed server message: {e}");
                return false;
            }
        };
        if let ServerMsg::Reply { id, reply } = frame.msg {
            let target = self.pending.lock().unwrap().get(&id).cloned();
            if let Some(target) = target {
                let _ = target.send(Frame {
                    msg: reply,
                    blobs: frame.blobs,
                });
            }
        }
        true
    }

    fn disconnected(&self) -> InternalErrors {
        InternalErrors::Io(format!("Lost connection to server {}", self.url))
    }

    fn send(&self, msg: &ClientMsg) -> Result<(), InternalErrors> {
        self.send_with_blobs(msg, &[])
    }

    fn send_with_blobs(&self, msg: &ClientMsg, blobs: &[Vec<u8>]) -> Result<(), InternalErrors> {
        let bytes = frame::encode(msg, blobs)?;
        self.outgoing.send(bytes).map_err(|_| self.disconnected())
    }

    /// Sends `request` and returns its id plus the channel its replies
    /// arrive on.
    fn request(&self, request: Request) -> Result<(u64, Receiver<Frame<Reply>>), InternalErrors> {
        self.request_with_blobs(request, Vec::new())
    }

    fn request_with_blobs(
        &self,
        request: Request,
        blobs: Vec<Vec<u8>>,
    ) -> Result<(u64, Receiver<Frame<Reply>>), InternalErrors> {
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let (tx, rx) = mpsc::channel();
        self.pending.lock().unwrap().insert(id, tx);
        if let Err(e) = self.send_with_blobs(&ClientMsg::Request { id, request }, &blobs) {
            self.finish(id);
            return Err(e);
        }
        Ok((id, rx))
    }

    fn finish(&self, id: u64) {
        self.pending.lock().unwrap().remove(&id);
    }

    fn recv(&self, rx: &Receiver<Frame<Reply>>) -> Result<Frame<Reply>, InternalErrors> {
        rx.recv().map_err(|_| self.disconnected())
    }

    fn cancel_handle(self: &Arc<Self>, id: u64) -> CancelHandle {
        let inner = Arc::downgrade(self);
        CancelHandle::with_callback(move || {
            if let Some(inner) = inner.upgrade() {
                let _ = inner.send(&ClientMsg::Cancel { id });
            }
        })
    }

    /// Turns the replies of a started job into a `RunningJob`.
    fn running_job(
        self: &Arc<Self>,
        id: u64,
        rx: Receiver<Frame<Reply>>,
        output_path: PathBuf,
        parallelism: usize,
    ) -> RunningJob {
        let (events_tx, events_rx) = mpsc::channel();
        let (done_tx, done_rx) = mpsc::channel();
        let inner = Arc::clone(self);
        std::thread::spawn(move || {
            let result = loop {
                match inner.recv(&rx) {
                    Ok(Frame {
                        msg: Reply::JobEvent(event),
                        blobs,
                    }) => match event_from_wire(event, blobs) {
                        Ok(event) => {
                            let _ = events_tx.send(event);
                        }
                        Err(e) => log::warn!("Dropping undecodable job event: {e}"),
                    },
                    Ok(Frame {
                        msg: Reply::JobDone(result),
                        ..
                    }) => break result.map_err(|e| e.into_internal()),
                    Ok(Frame {
                        msg: Reply::Failed(e),
                        ..
                    }) => break Err(e.into_internal()),
                    Ok(_) => break Err(unexpected_reply()),
                    Err(e) => break Err(e),
                }
            };
            inner.finish(id);
            // Close the event stream before the result is available - front
            // ends drain events first, then wait.
            drop(events_tx);
            let _ = done_tx.send(result);
        });
        let disconnected = self.disconnected();
        RunningJob::from_parts(
            events_rx,
            self.cancel_handle(id),
            output_path,
            parallelism,
            Box::new(move || done_rx.recv().unwrap_or(Err(disconnected))),
        )
    }
}

fn unexpected_reply() -> InternalErrors {
    InternalErrors::Internal("unexpected reply from server".into())
}

/// An image opened on the server; tiles are fetched on demand.
struct RemoteImageSource {
    inner: Arc<Inner>,
    handle: u64,
    meta: ImageMeta,
}

impl ImageSource for RemoteImageSource {
    fn meta(&self) -> &ImageMeta {
        &self.meta
    }

    fn read_tile(&self, req: &TileRequest) -> Result<Vec<ImageChannel>, InternalErrors> {
        let inner = &self.inner;
        let (id, rx) = inner.request(Request::ReadTile {
            handle: self.handle,
            tile: req.clone(),
        })?;
        let reply = inner.recv(&rx);
        inner.finish(id);
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
        let _ = self.inner.send(&ClientMsg::CloseImage {
            handle: self.handle,
        });
    }
}

/// A results database opened on the server; every query is one request.
struct RemoteResults {
    inner: Arc<Inner>,
    handle: u64,
}

impl RemoteResults {
    fn query(&self, query: ResultsQuery) -> Result<ResultsAnswer, InternalErrors> {
        let request = Request::QueryResults {
            handle: self.handle,
        };
        let (id, rx) = self
            .inner
            .request_with_blobs(request, vec![to_postcard(&query)?])?;
        let reply = self.inner.recv(&rx);
        self.inner.finish(id);
        match reply? {
            Frame {
                msg: Reply::ResultsAnswer,
                blobs,
            } => from_postcard(blobs.first()),
            Frame {
                msg: Reply::Failed(e),
                ..
            } => Err(e.into_internal()),
            _ => Err(unexpected_reply()),
        }
    }

    fn table(&self, query: ResultsQuery) -> Result<DatabaseResult, InternalErrors> {
        match self.query(query)? {
            ResultsAnswer::Table(table) => Ok(table),
            _ => Err(unexpected_reply()),
        }
    }

    fn count(&self, query: ResultsQuery) -> u32 {
        match self.query(query) {
            Ok(ResultsAnswer::Count(n)) => n,
            Ok(_) => {
                log::warn!("Unexpected answer to a stack-count query");
                1
            }
            Err(e) => {
                log::warn!("Could not read stack count from the server: {e}");
                1
            }
        }
    }
}

impl ResultsSource for RemoteResults {
    fn get_object_list(&self, filter: &ListFilter) -> Result<DatabaseResult, InternalErrors> {
        self.table(ResultsQuery::ObjectList(filter.clone()))
    }

    fn get_grouped_by_image(
        &self,
        filter: &GroupedByImageFilter,
    ) -> Result<DatabaseResult, InternalErrors> {
        self.table(ResultsQuery::GroupedByImage(filter.clone()))
    }

    fn get_group_by_plate(
        &self,
        filter: &PlateFilter,
        view: &View,
    ) -> Result<DatabaseResult, InternalErrors> {
        self.table(ResultsQuery::GroupByPlate(filter.clone(), view.clone()))
    }

    fn get_group_by_well(
        &self,
        filter: &WellFilter,
        view: &View,
    ) -> Result<DatabaseResult, InternalErrors> {
        self.table(ResultsQuery::GroupByWell(filter.clone(), view.clone()))
    }

    fn get_image_heatmap(
        &self,
        filter: &ImageHeatmapFilter,
        view: &View,
    ) -> Result<DatabaseResult, InternalErrors> {
        self.table(ResultsQuery::ImageHeatmap(filter.clone(), view.clone()))
    }

    fn get_images(&self) -> Result<Vec<ImageEntry>, InternalErrors> {
        match self.query(ResultsQuery::Images)? {
            ResultsAnswer::Images(images) => Ok(images),
            _ => Err(unexpected_reply()),
        }
    }

    fn enable_image(&self, image_rel_path: &str, disable: bool) -> Result<(), InternalErrors> {
        let query = ResultsQuery::EnableImage {
            image_rel_path: image_rel_path.into(),
            disable,
        };
        match self.query(query)? {
            ResultsAnswer::Done => Ok(()),
            _ => Err(unexpected_reply()),
        }
    }

    fn get_object_classes(&self) -> Result<Vec<Class>, InternalErrors> {
        match self.query(ResultsQuery::ObjectClasses)? {
            ResultsAnswer::Classes(classes) => Ok(classes),
            _ => Err(unexpected_reply()),
        }
    }

    fn get_available_columns(&self) -> Result<Vec<ColumnEntry>, InternalErrors> {
        match self.query(ResultsQuery::AvailableColumns)? {
            ResultsAnswer::Columns(columns) => Ok(columns),
            _ => Err(unexpected_reply()),
        }
    }

    fn get_nr_of_z_stacks(&self) -> u32 {
        self.count(ResultsQuery::ZStacks)
    }

    fn get_nr_of_t_stacks(&self) -> u32 {
        self.count(ResultsQuery::TStacks)
    }

    fn boxplot(&self, filter: &BoxplotFilter) -> Result<BoxplotResult, InternalErrors> {
        match self.query(ResultsQuery::Boxplot(filter.clone()))? {
            ResultsAnswer::Boxplot(result) => Ok(result),
            _ => Err(unexpected_reply()),
        }
    }

    fn histogram(&self, filter: &HistogramFilter) -> Result<HistogramResult, InternalErrors> {
        match self.query(ResultsQuery::Histogram(filter.clone()))? {
            ResultsAnswer::Histogram(result) => Ok(result),
            _ => Err(unexpected_reply()),
        }
    }

    fn scatter(&self, filter: &ScatterFilter) -> Result<ScatterResult, InternalErrors> {
        match self.query(ResultsQuery::Scatter(filter.clone()))? {
            ResultsAnswer::Scatter(result) => Ok(result),
            _ => Err(unexpected_reply()),
        }
    }

    fn export(
        &self,
        export: &ResultExport,
        cancel: &AtomicBool,
        on_progress: ExportProgressFn,
    ) -> Result<(), InternalErrors> {
        let request = Request::ExportResults {
            handle: self.handle,
        };
        let (id, rx) = self
            .inner
            .request_with_blobs(request, vec![to_postcard(export)?])?;
        let mut cancel_sent = false;
        let result = loop {
            if !cancel_sent && cancel.load(std::sync::atomic::Ordering::SeqCst) {
                cancel_sent = true;
                let _ = self.inner.send(&ClientMsg::Cancel { id });
            }
            match rx.recv_timeout(Duration::from_millis(100)) {
                Ok(Frame {
                    msg:
                        Reply::ExportProgress {
                            message,
                            current,
                            total,
                        },
                    ..
                }) => on_progress(&message, current, total),
                Ok(Frame {
                    msg: Reply::ExportDone(result),
                    ..
                }) => break result.map_err(|e| e.into_internal()),
                Ok(Frame {
                    msg: Reply::Failed(e),
                    ..
                }) => break Err(e.into_internal()),
                Ok(_) => break Err(unexpected_reply()),
                Err(mpsc::RecvTimeoutError::Timeout) => continue,
                Err(mpsc::RecvTimeoutError::Disconnected) => {
                    break Err(self.inner.disconnected());
                }
            }
        };
        self.inner.finish(id);
        result
    }
}

impl Drop for RemoteResults {
    fn drop(&mut self) {
        let _ = self.inner.send(&ClientMsg::CloseResults {
            handle: self.handle,
        });
    }
}
