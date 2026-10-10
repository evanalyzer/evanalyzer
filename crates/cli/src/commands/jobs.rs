//! `jobs` and `attach`: the analyses on a `--remote` server, which keep
//! running when the client disconnects.

use crate::args::AttachArgs;
use evanalyzer_app::analysis::{JobInfo, JobState};
use evanalyzer_app::backends::Backend;
use evanalyzer_cfg::core_types::InternalErrors;
use std::time::SystemTime;

pub fn run_list(backend: &dyn Backend) -> Result<(), InternalErrors> {
    let jobs = backend.list_jobs()?;
    if jobs.is_empty() {
        println!("{}", no_jobs(backend));
        return Ok(());
    }
    for job in &jobs {
        println!(
            "{}  {:<24} {:<20} started {}",
            job.id,
            describe(&job.state),
            job.name(),
            ago(job.started_at)
        );
    }
    Ok(())
}

pub fn run_attach(args: AttachArgs, backend: &dyn Backend) -> Result<(), InternalErrors> {
    let jobs = backend.list_jobs()?;
    let info = match &args.job {
        Some(id) => jobs.iter().find(|job| job.id == *id),
        None => Some(
            running(&jobs).ok_or_else(|| InternalErrors::InvalidArgument(no_running(backend)))?,
        ),
    };
    let id = info.map_or_else(
        || args.job.clone().unwrap_or_default(),
        |job| job.id.clone(),
    );
    let job = backend.attach_job(&id)?;
    println!("Analysis:  {id}");
    println!("Output:    {}", job.output_path().display());
    println!("(Ctrl+C cancels the analysis)\n");
    let started = info.map_or_else(SystemTime::now, |job| job.started_at);
    super::analyze::follow(job, backend, 0, started)
}

fn running(jobs: &[JobInfo]) -> Option<&JobInfo> {
    jobs.iter().find(|job| job.is_running())
}

fn no_jobs(backend: &dyn Backend) -> String {
    if backend.is_remote() {
        "No analyses on the server.".into()
    } else {
        "No analyses: only a server (--remote) keeps track of them.".into()
    }
}

fn no_running(backend: &dyn Backend) -> String {
    if backend.is_remote() {
        "No analysis is running on the server (`cli jobs` lists the finished ones).".into()
    } else {
        "Nothing to attach to: only a server (--remote) runs analyses in the background.".into()
    }
}

fn describe(state: &JobState) -> String {
    match state {
        JobState::Running { done, total } => format!("running {done}/{total}"),
        JobState::Succeeded => "finished".into(),
        JobState::Cancelled => "cancelled".into(),
        JobState::Failed(_) => "failed".into(),
    }
}

/// "5 min ago".
fn ago(at: SystemTime) -> String {
    let minutes = SystemTime::now()
        .duration_since(at)
        .unwrap_or_default()
        .as_secs()
        / 60;
    match minutes {
        0 => "just now".into(),
        1..=119 => format!("{minutes} min ago"),
        _ => format!("{} h ago", minutes / 60),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use evanalyzer_app::backends::local::LocalBackend;
    use std::time::Duration;

    #[test]
    fn states_and_ages_read_naturally() {
        assert_eq!(
            describe(&JobState::Running { done: 3, total: 10 }),
            "running 3/10"
        );
        assert_eq!(describe(&JobState::Failed("x".into())), "failed");
        let now = SystemTime::now();
        assert_eq!(ago(now), "just now");
        assert_eq!(ago(now - Duration::from_secs(5 * 60)), "5 min ago");
        assert_eq!(ago(now - Duration::from_secs(3 * 3600)), "3 h ago");
    }

    #[test]
    fn locally_there_is_nothing_to_list_or_attach_to() {
        let local = LocalBackend::default();
        run_list(&local).unwrap();
        let error = run_attach(AttachArgs { job: None }, &local)
            .err()
            .unwrap()
            .to_string();
        assert!(error.contains("--remote"), "{error}");
    }

    #[test]
    fn the_running_job_is_the_default_to_attach_to() {
        let job = |id: &str, state| JobInfo {
            id: id.into(),
            output_path: "/p/results/x".into(),
            started_at: SystemTime::now(),
            state,
        };
        let jobs = [
            job("old", JobState::Succeeded),
            job("now", JobState::Running { done: 0, total: 1 }),
        ];
        assert_eq!(running(&jobs).unwrap().id, "now");
        assert!(running(&jobs[..1]).is_none());
    }
}
