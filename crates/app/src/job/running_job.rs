use evanalyzer_cfg::{core_types::InternalErrors, settings::object_settings::ObjectMetricSettings};
use evanalyzer_core::{JobExecutor, ProgressEvent};
use serde::{Deserialize, Serialize};
use std::{
    any::Any,
    path::PathBuf,
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
        mpsc::Receiver,
    },
    thread::JoinHandle,
};

/// Cloneable handle to request cancellation of a [`RunningJob`] or
/// [`RunningTraining`](crate::ai_learning::RunningTraining) from another thread (a Cancel
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
}

impl RunningJob {
    /// Runs `job` on its own thread in this process.
    pub(crate) fn spawn(
        job: JobExecutor,
        parallelism: usize,
        preview_objects: Option<Arc<Mutex<Vec<ObjectMetricSettings>>>>,
    ) -> Self {
        let output_path = job.output_path.clone();
        let (handle, events, cancel) = job.run_async(parallelism);
        let completion: JobCompletion = Box::new(move || {
            join_job(handle, "Pipeline worker")?;
            let preview_objects = preview_objects
                .map(|objects| objects.lock().unwrap_or_else(|e| e.into_inner()).clone());
            Ok(JobOutput { preview_objects })
        });
        Self {
            events,
            cancel: CancelHandle::new(cancel),
            output_path,
            parallelism,
            completion,
        }
    }

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
        }
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

/// Joins a job thread, returning a panic inside it as
/// `InternalErrors::Internal("<what> crashed: <panic message>")` instead of
/// re-raising it in the caller.
pub(crate) fn join_job<T>(
    handle: JoinHandle<Result<T, InternalErrors>>,
    what: &str,
) -> Result<T, InternalErrors> {
    match handle.join() {
        Ok(result) => result,
        Err(payload) => Err(InternalErrors::Internal(format!(
            "{what} crashed: {}",
            panic_message(&payload)
        ))),
    }
}

fn panic_message(payload: &Box<dyn Any + Send>) -> String {
    if let Some(s) = payload.downcast_ref::<&str>() {
        s.to_string()
    } else if let Some(s) = payload.downcast_ref::<String>() {
        s.clone()
    } else {
        "non-string panic payload".to_string()
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

    #[test]
    fn join_job_returns_a_panic_as_an_internal_error_naming_the_job() {
        let handle = std::thread::spawn(|| -> Result<(), InternalErrors> { panic!("boom") });
        match join_job(handle, "Test worker") {
            Err(InternalErrors::Internal(msg)) => assert_eq!(msg, "Test worker crashed: boom"),
            other => panic!("expected an Internal error, got {other:?}"),
        }
    }

    #[test]
    fn join_job_passes_through_the_threads_own_result() {
        let ok = std::thread::spawn(|| Ok::<_, InternalErrors>(7));
        assert_eq!(join_job(ok, "Test worker").unwrap(), 7);
        let cancelled = std::thread::spawn(|| Err::<(), _>(InternalErrors::Cancelled));
        assert!(matches!(
            join_job(cancelled, "Test worker"),
            Err(InternalErrors::Cancelled)
        ));
    }

    #[test]
    fn panic_message_extracts_str_and_string_payloads() {
        let str_payload: Box<dyn Any + Send> = Box::new("boom");
        assert_eq!(panic_message(&str_payload), "boom");
        let string_payload: Box<dyn Any + Send> = Box::new(String::from("owned boom"));
        assert_eq!(panic_message(&string_payload), "owned boom");
        let other_payload: Box<dyn Any + Send> = Box::new(42i32);
        assert_eq!(panic_message(&other_payload), "non-string panic payload");
    }
}
