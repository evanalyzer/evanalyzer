use crate::api::{CancelHandle, JobCompletion, JobOutput, RunningJob};
use evanalyzer_cfg::{core_types::InternalErrors, settings::object_settings::ObjectMetricSettings};
use evanalyzer_core::JobExecutor;
use std::{
    any::Any,
    sync::{Arc, Mutex},
    thread::JoinHandle,
};

/// Runs `job` on its own thread in this process.
pub(crate) fn spawn_job(
    job: JobExecutor,
    parallelism: usize,
    preview_objects: Option<Arc<Mutex<Vec<ObjectMetricSettings>>>>,
) -> RunningJob {
    let output_path = job.output_path.clone();
    let (handle, events, cancel) = job.run_async(parallelism);
    let completion: JobCompletion = Box::new(move || {
        join_job(handle, "Pipeline worker")?;
        let preview_objects = preview_objects
            .map(|objects| objects.lock().unwrap_or_else(|e| e.into_inner()).clone());
        Ok(JobOutput { preview_objects })
    });
    RunningJob::from_parts(
        events,
        CancelHandle::new(cancel),
        output_path,
        parallelism,
        completion,
    )
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
