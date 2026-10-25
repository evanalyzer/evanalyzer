//! # jobs
//!
//! Analysis and preview runs as front ends see them: what to ask for, and
//! the handle to a running job (events, cancel, result) - the same whether
//! it runs in this process or on a server.
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-10-10
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.

use evanalyzer_cfg::{
    core_types::{BreakpointSettings, InternalErrors},
    settings::{object_settings::ObjectMetricSettings, project_settings::ProjectSettings},
};
pub use evanalyzer_core::ProgressEvent;
use serde::{Deserialize, Serialize};
use std::{
    path::PathBuf,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
        mpsc::Receiver,
    },
};

/// The visible area a preview is restricted to, in screen pixels.
#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize)]
pub struct PreviewViewport {
    /// Current pan offset (screen pixels from the image's top-left corner).
    pub offset_x: f32,
    pub offset_y: f32,
    pub viewport_width: f32,
    pub viewport_height: f32,
    /// Current zoom level (1.0 = 100 %).
    pub zoom: f32,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct PreviewRequest {
    pub settings: ProjectSettings,
    pub project_path: PathBuf,
    pub viewport: PreviewViewport,
    /// Stop at (or snapshot) a pipeline step and stream its image back via
    /// `ProgressEvent::BreakpointReached`.
    pub breakpoint: Option<BreakpointSettings>,
    /// Previews covering more tiles are rejected (at low zoom a whole slide
    /// spans hundreds). `None`: what the computer running it can take
    /// ([`ViewportMemorySettings::auto_preview_tiles`]); a server also caps
    /// a number asked for at that.
    ///
    /// [`ViewportMemorySettings::auto_preview_tiles`]: crate::workspace::settings::ViewportMemorySettings::auto_preview_tiles
    pub max_visible_tiles: Option<u32>,
    /// How many tiles send their breakpoint image (the visible ones nearest
    /// the viewport centre) - they are kept where the window is, so the
    /// window's computer decides.
    pub breakpoint_tiles: u32,
    /// The whole image instead of the viewport: the visible tiles first, then
    /// the others - the counts of the whole image. Refused for an image of
    /// more than [`WHOLE_IMAGE_PREVIEW_MAX_TILES`] full-size tiles (a whole
    /// slide: the analysis is the way there).
    pub whole_image: bool,
}

/// A whole-image preview's limit: full-size (4096 px) tiles, 16 = a
/// 16384 x 16384 image.
pub const WHOLE_IMAGE_PREVIEW_MAX_TILES: usize = 16;

#[derive(Debug)]
pub enum StartPreviewError {
    /// The viewport covers more than `max` tiles (see
    /// [`PreviewRequest::max_visible_tiles`]) - the user has to zoom in
    /// first. A whole-image preview: the image has more than `max`
    /// ([`WHOLE_IMAGE_PREVIEW_MAX_TILES`]).
    TooManyTiles {
        tiles: usize,
        max: usize,
    },
    Failed(InternalErrors),
}

impl From<InternalErrors> for StartPreviewError {
    fn from(e: InternalErrors) -> Self {
        StartPreviewError::Failed(e)
    }
}
/// Cloneable handle to request cancellation of a [`RunningJob`] or
/// [`RunningTraining`](crate::api::RunningTraining) from another thread (a Cancel
/// button, a Ctrl+C handler). The job stops after in-flight work finishes and
/// its `wait` returns `InternalErrors::Cancelled`.
///
/// Locally the job polls the shared flag; a remote backend additionally
/// registers `on_cancel` to forward the request to the server.
#[derive(Clone)]
pub struct CancelHandle {
    flag: Arc<AtomicBool>,
    on_cancel: Option<Arc<dyn Fn() + Send + Sync>>,
}

impl std::fmt::Debug for CancelHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CancelHandle")
            .field("cancelled", &self.is_cancelled())
            .finish()
    }
}

impl CancelHandle {
    pub(crate) fn new(flag: Arc<AtomicBool>) -> Self {
        Self {
            flag,
            on_cancel: None,
        }
    }

    /// For backends whose job doesn't poll a local flag: `on_cancel` runs
    /// (once per `cancel` call) in addition to setting the flag.
    pub fn with_callback(on_cancel: impl Fn() + Send + Sync + 'static) -> Self {
        Self {
            flag: Arc::new(AtomicBool::new(false)),
            on_cancel: Some(Arc::new(on_cancel)),
        }
    }

    pub fn cancel(&self) {
        self.flag.store(true, Ordering::SeqCst);
        if let Some(on_cancel) = &self.on_cancel {
            on_cancel();
        }
    }

    pub fn is_cancelled(&self) -> bool {
        self.flag.load(Ordering::SeqCst)
    }
}

/// What a job produced once it has finished successfully.
#[derive(Debug, Default, Serialize, Deserialize)]
pub struct JobOutput {
    /// Preview runs only: the final, tile-merged object set. Replaces the
    /// per-tile objects streamed earlier via `ProgressEvent::TileCompleted`,
    /// which are sent *before* the whole-image phase's `TileMerge` ran and so
    /// still contain cross-tile fragments. `None` for analysis runs.
    pub preview_objects: Option<Vec<ObjectMetricSettings>>,
    /// Preview runs only: the measurements, per plane - like the objects.
    #[serde(default)]
    pub preview_measurements: Option<Vec<evanalyzer_cfg::core_types::PipelineMeasurements>>,
}

/// An analysis a backend keeps track of: running, or finished not long ago.
/// A server's analyses outlive the client connection that started them.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct JobInfo {
    pub id: String,
    /// Directory the analysis writes its results into; its last component
    /// is the job name.
    pub output_path: PathBuf,
    pub started_at: std::time::SystemTime,
    pub state: JobState,
}

impl JobInfo {
    /// The job's name: its results folder's.
    pub fn name(&self) -> String {
        self.output_path
            .file_name()
            .map(|name| name.to_string_lossy().into_owned())
            .unwrap_or_default()
    }

    pub fn is_running(&self) -> bool {
        matches!(self.state, JobState::Running { .. })
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum JobState {
    /// `done` of `total` images processed (`total` is 0 until known).
    Running {
        done: usize,
        total: usize,
    },
    Succeeded,
    Cancelled,
    Failed(String),
}

/// Blocks until a job has finished and yields its result - a local job
/// joins its thread, a remote one waits for the server's final message.
pub type JobCompletion = Box<dyn FnOnce() -> Result<JobOutput, InternalErrors> + Send>;

/// A running job, wherever it runs. Drain [`events`](Self::events) until it
/// closes, then call [`wait`](Self::wait) for the result.
pub struct RunningJob {
    events: Receiver<ProgressEvent>,
    cancel: CancelHandle,
    output_path: PathBuf,
    parallelism: usize,
    completion: JobCompletion,
    id: Option<String>,
}
impl RunningJob {
    /// Assembles a job run by some other backend (e.g. on a server). The
    /// `events` channel must close once the job is over, like a local job's.
    pub fn from_parts(
        events: Receiver<ProgressEvent>,
        cancel: CancelHandle,
        output_path: PathBuf,
        parallelism: usize,
        completion: JobCompletion,
    ) -> Self {
        Self {
            events,
            cancel,
            output_path,
            parallelism,
            completion,
            id: None,
        }
    }

    /// The job as one the backend keeps track of under `id` - see
    /// [`Backend::list_jobs`](crate::api::Backend::list_jobs).
    pub fn with_id(mut self, id: String) -> Self {
        self.id = Some(id);
        self
    }

    /// Under which id the backend tracks the job, if it does: a server's
    /// analyses keep running when the connection drops, and can be attached
    /// to again with this id.
    pub fn id(&self) -> Option<&str> {
        self.id.as_deref()
    }

    /// Progress events, in order. The channel closes once the job is over,
    /// so `for event in job.events()` ends by itself.
    pub fn events(&self) -> &Receiver<ProgressEvent> {
        &self.events
    }

    pub fn cancel_handle(&self) -> CancelHandle {
        self.cancel.clone()
    }

    /// Directory the job writes its results into.
    pub fn output_path(&self) -> &PathBuf {
        &self.output_path
    }

    /// Number of parallel workers the job was started with.
    pub fn parallelism(&self) -> usize {
        self.parallelism
    }

    /// Blocks until the job is over and returns its result.
    ///
    /// A panic inside a local job thread (e.g. a malformed tile at the image
    /// edge) is returned as `InternalErrors::Internal` instead of being
    /// re-raised, so the calling worker survives to run the next job.
    pub fn wait(self) -> Result<JobOutput, InternalErrors> {
        (self.completion)()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cancel_handle_clones_share_one_flag() {
        let handle = CancelHandle::new(Arc::new(AtomicBool::new(false)));
        let clone = handle.clone();
        assert!(!handle.is_cancelled());
        clone.cancel();
        assert!(handle.is_cancelled());
    }

    #[test]
    fn a_callback_cancel_handle_runs_its_callback_and_reports_cancelled() {
        let calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let counter = Arc::clone(&calls);
        let handle = CancelHandle::with_callback(move || {
            counter.fetch_add(1, Ordering::SeqCst);
        });
        assert!(!handle.is_cancelled());
        handle.clone().cancel();
        assert!(handle.is_cancelled());
        assert_eq!(calls.load(Ordering::SeqCst), 1);
    }
}
