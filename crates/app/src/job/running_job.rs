use evanalyzer_cfg::{core_types::InternalErrors, settings::object_settings::ObjectMetricSettings};
use evanalyzer_core::{JobExecutor, ProgressEvent};
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

/// Cloneable handle to request cancellation of a [`RunningJob`] from another
/// thread (a Cancel button, a Ctrl+C handler). The job stops after in-flight
/// work finishes and [`RunningJob::wait`] returns `InternalErrors::Cancelled`.
#[derive(Clone, Debug)]
pub struct CancelHandle(Arc<AtomicBool>);

impl CancelHandle {
    pub fn cancel(&self) {
        self.0.store(true, Ordering::SeqCst);
    }

    pub fn is_cancelled(&self) -> bool {
        self.0.load(Ordering::SeqCst)
    }
}

/// What a job produced once it has finished successfully.
#[derive(Debug, Default)]
pub struct JobOutput {
    /// Preview runs only: the final, tile-merged object set. Replaces the
    /// per-tile objects streamed earlier via `ProgressEvent::TileCompleted`,
    /// which are sent *before* the whole-image phase's `TileMerge` ran and so
    /// still contain cross-tile fragments. `None` for analysis runs.
    pub preview_objects: Option<Vec<ObjectMetricSettings>>,
}

/// A job running on its own thread. Drain [`events`](Self::events) until it
/// closes, then call [`wait`](Self::wait) for the result.
pub struct RunningJob {
    handle: JoinHandle<Result<(), InternalErrors>>,
    events: Receiver<ProgressEvent>,
    cancel: CancelHandle,
    output_path: PathBuf,
    parallelism: usize,
    preview_objects: Option<Arc<Mutex<Vec<ObjectMetricSettings>>>>,
}

impl RunningJob {
    pub(super) fn spawn(
        job: JobExecutor,
        parallelism: usize,
        preview_objects: Option<Arc<Mutex<Vec<ObjectMetricSettings>>>>,
    ) -> Self {
        let output_path = job.output_path.clone();
        let (handle, events, cancel) = job.run_async(parallelism);
        Self {
            handle,
            events,
            cancel: CancelHandle(cancel),
            output_path,
            parallelism,
            preview_objects,
        }
    }

    /// Progress events, in order. The channel closes once the job thread
    /// exits, so `for event in job.events()` ends by itself.
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

    /// Blocks until the job thread exits and returns its result.
    ///
    /// A panic inside the job thread (e.g. a malformed tile at the image
    /// edge) is returned as `InternalErrors::Internal` instead of being
    /// re-raised, so the calling worker survives to run the next job.
    pub fn wait(self) -> Result<JobOutput, InternalErrors> {
        match self.handle.join() {
            Ok(result) => result?,
            Err(payload) => {
                return Err(InternalErrors::Internal(format!(
                    "Pipeline worker crashed: {}",
                    panic_message(&payload)
                )));
            }
        }
        let preview_objects = self
            .preview_objects
            .map(|objects| objects.lock().unwrap_or_else(|e| e.into_inner()).clone());
        Ok(JobOutput { preview_objects })
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
        let handle = CancelHandle(Arc::new(AtomicBool::new(false)));
        let clone = handle.clone();
        assert!(!handle.is_cancelled());
        clone.cancel();
        assert!(handle.is_cancelled());
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
