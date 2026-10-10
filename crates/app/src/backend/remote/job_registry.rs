//! The worker's analyses, independent of client connections.
//!
//! An analysis belongs to the worker, not to the connection that started it:
//! when the client disconnects (network gone, laptop asleep, window closed)
//! it keeps running, and any connection can follow it again with
//! `AttachJob` - the GUI does so silently after reconnecting. Clients are
//! only subscribers; one pump thread per analysis reads its events, keeps a
//! progress [`Snapshot`] and passes each event on to every subscriber.
//!
//! A client attaching late doesn't need every event that happened, only
//! where the analysis stands: the snapshot is replayed as a few ordinary
//! events (`Started`, `ImageFailed`s, the latest `ImageCompleted`, the tile
//! progress), so a front end follows an attached analysis with the same code
//! as one it started itself, and memory stays constant however long it runs.
//!
//! Only one analysis runs at a time per worker (= per user): a second would
//! claim the same cores and RAM, and reattaching would need the user to
//! pick one.

use super::wire::frame;
use super::wire::protocol::{Reply, ServerMsg, WireError, event_to_wire};
use crate::api::{CancelHandle, JobInfo, JobOutput, JobState, ProgressEvent, RunningJob};
use evanalyzer_cfg::core_types::InternalErrors;
use std::path::PathBuf;
use std::sync::mpsc::Sender;
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime};

/// Finished analyses are listed this long (or until a client forgets them);
/// their results are on disk anyway.
const KEEP_FINISHED: Duration = Duration::from_secs(24 * 60 * 60);
/// ... and at most this many of them.
const MAX_FINISHED: usize = 50;

#[derive(Default)]
pub(super) struct JobRegistry {
    /// Oldest first.
    jobs: Mutex<Vec<Arc<JobEntry>>>,
}

/// One analysis: running, or finished.
pub(super) struct JobEntry {
    id: String,
    output_path: PathBuf,
    parallelism: usize,
    started_at: SystemTime,
    cancel: CancelHandle,
    state: Mutex<EntryState>,
}

#[derive(Default)]
struct EntryState {
    progress: Snapshot,
    /// Set once the analysis is over.
    outcome: Option<(Result<(), WireError>, SystemTime)>,
    subscribers: Vec<Subscriber>,
}

/// A connection following the analysis: its outgoing frame queue and the
/// request id its replies go to.
struct Subscriber {
    outgoing: Sender<Vec<u8>>,
    request_id: u64,
}

/// Where an analysis stands - enough to show its progress to a client that
/// attaches late.
#[derive(Default)]
struct Snapshot {
    total: Option<usize>,
    /// Images done, and the latest of them.
    done: usize,
    last_image: Option<PathBuf>,
    failed: Vec<PathBuf>,
    /// Tile progress of the image in work: (completed, total).
    tiles: Option<(usize, usize)>,
    finished: bool,
}

impl Snapshot {
    fn apply(&mut self, event: &ProgressEvent) {
        match event {
            ProgressEvent::Started { total } => self.total = Some(*total),
            ProgressEvent::TilesScheduled { total_tiles } => self.tiles = Some((0, *total_tiles)),
            ProgressEvent::TileCompleted {
                tile_index,
                total_tiles,
                ..
            } => self.tiles = Some((*tile_index, *total_tiles)),
            ProgressEvent::WholeImagePhaseCompleted {
                completed,
                total_tiles,
            } => self.tiles = Some((*completed, *total_tiles)),
            ProgressEvent::ImageCompleted { index, total, path } => {
                self.done = *index;
                self.total = Some(*total);
                self.last_image = Some(path.clone());
                self.tiles = None;
            }
            ProgressEvent::ImageFailed { path } => {
                self.failed.push(path.clone());
                self.tiles = None;
            }
            ProgressEvent::Finished => self.finished = true,
            ProgressEvent::BreakpointReached { .. } => {}
        }
    }

    /// The snapshot as the events a client would have needed to get here.
    fn replay(&self) -> Vec<ProgressEvent> {
        let mut events = Vec::new();
        let Some(total) = self.total else {
            return events;
        };
        events.push(ProgressEvent::Started { total });
        events.extend(
            self.failed
                .iter()
                .map(|path| ProgressEvent::ImageFailed { path: path.clone() }),
        );
        if let Some(path) = &self.last_image {
            events.push(ProgressEvent::ImageCompleted {
                index: self.done,
                total,
                path: path.clone(),
            });
        }
        if let Some((completed, total_tiles)) = self.tiles {
            events.push(ProgressEvent::TilesScheduled { total_tiles });
            if completed > 0 {
                events.push(ProgressEvent::TileCompleted {
                    tile_index: completed,
                    total_tiles,
                    // Already part of the results database; a preview's
                    // objects never get here (previews aren't registered).
                    objects: Vec::new(),
                });
            }
        }
        if self.finished {
            events.push(ProgressEvent::Finished);
        }
        events
    }
}

impl JobRegistry {
    /// Starts an analysis with `start` - unless one is running already - and
    /// keeps it running regardless of who follows it.
    pub(super) fn start(
        &self,
        start: impl FnOnce() -> Result<RunningJob, InternalErrors>,
    ) -> Result<Arc<JobEntry>, InternalErrors> {
        // Held while starting, so two clients can't both pass the check.
        let mut jobs = self.jobs.lock().unwrap();
        if let Some(running) = jobs.iter().find(|job| job.is_running()) {
            return Err(InternalErrors::InvalidArgument(format!(
                "An analysis is already running on the server ('{}', started {}). \
                 Wait for it to finish or cancel it.",
                running.info().name(),
                ago(running.started_at)
            )));
        }
        let job = start()?;
        let entry = Arc::new(JobEntry {
            id: new_id()?,
            output_path: job.output_path().clone(),
            parallelism: job.parallelism(),
            started_at: SystemTime::now(),
            cancel: job.cancel_handle(),
            state: Mutex::new(EntryState::default()),
        });
        prune(&mut jobs);
        jobs.push(Arc::clone(&entry));
        drop(jobs);

        let pump = Arc::clone(&entry);
        std::thread::Builder::new()
            .name("evanalyzer-job".into())
            .spawn(move || pump.run(job))
            .map_err(|e| InternalErrors::Internal(format!("could not start job thread: {e}")))?;
        log::info!(
            "Analysis {} started ({})",
            entry.id,
            entry.output_path.display()
        );
        Ok(entry)
    }

    pub(super) fn get(&self, id: &str) -> Option<Arc<JobEntry>> {
        self.jobs
            .lock()
            .unwrap()
            .iter()
            .find(|job| job.id == id)
            .cloned()
    }

    pub(super) fn list(&self) -> Vec<JobInfo> {
        let mut jobs = self.jobs.lock().unwrap();
        prune(&mut jobs);
        jobs.iter().map(|job| job.info()).collect()
    }

    /// Removes finished analysis `id` from the list.
    pub(super) fn forget(&self, id: &str) -> Result<(), InternalErrors> {
        let mut jobs = self.jobs.lock().unwrap();
        match jobs.iter().position(|job| job.id == id) {
            Some(index) if jobs[index].is_running() => Err(InternalErrors::InvalidArgument(
                "The analysis is still running - cancel it first".into(),
            )),
            Some(index) => {
                jobs.remove(index);
                Ok(())
            }
            None => Ok(()),
        }
    }

    /// Whether an analysis is running - the worker isn't idle then.
    #[allow(dead_code)] // for the worker's idle timeout (next step)
    pub(super) fn any_running(&self) -> bool {
        self.jobs.lock().unwrap().iter().any(|job| job.is_running())
    }
}

/// Drops finished analyses older than [`KEEP_FINISHED`], and the oldest
/// beyond [`MAX_FINISHED`].
fn prune(jobs: &mut Vec<Arc<JobEntry>>) {
    let now = SystemTime::now();
    jobs.retain(|job| match job.finished_at() {
        Some(at) => now.duration_since(at).unwrap_or_default() < KEEP_FINISHED,
        None => true,
    });
    let mut finished = jobs.iter().filter(|job| !job.is_running()).count();
    jobs.retain(|job| {
        if finished > MAX_FINISHED && !job.is_running() {
            finished -= 1;
            false
        } else {
            true
        }
    });
}

impl JobEntry {
    pub(super) fn cancel_handle(&self) -> CancelHandle {
        self.cancel.clone()
    }

    fn is_running(&self) -> bool {
        self.state.lock().unwrap().outcome.is_none()
    }

    fn finished_at(&self) -> Option<SystemTime> {
        self.state
            .lock()
            .unwrap()
            .outcome
            .as_ref()
            .map(|(_, at)| *at)
    }

    fn info(&self) -> JobInfo {
        let state = self.state.lock().unwrap();
        JobInfo {
            id: self.id.clone(),
            output_path: self.output_path.clone(),
            started_at: self.started_at,
            state: match &state.outcome {
                None => JobState::Running {
                    done: state.progress.done + state.progress.failed.len(),
                    total: state.progress.total.unwrap_or(0),
                },
                Some((Ok(()), _)) => JobState::Succeeded,
                Some((Err(e), _)) if e.is_cancelled() => JobState::Cancelled,
                Some((Err(e), _)) => JobState::Failed(e.text().to_string()),
            },
        }
    }

    /// Lets the connection with `outgoing` follow the analysis under
    /// `request_id`: `JobStarted`, the progress so far, then live events and
    /// `JobDone` - at once if the analysis is over. Atomic with respect to
    /// the pump, so no event is missed or sent twice.
    pub(super) fn subscribe(&self, outgoing: Sender<Vec<u8>>, request_id: u64) {
        let mut state = self.state.lock().unwrap();
        let subscriber = Subscriber {
            outgoing,
            request_id,
        };
        subscriber.send(
            Reply::JobStarted {
                output_path: self.output_path.clone(),
                parallelism: self.parallelism,
                job_id: Some(self.id.clone()),
            },
            Vec::new(),
        );
        for event in state.progress.replay() {
            let (wire, blobs) = event_to_wire(event);
            subscriber.send(Reply::JobEvent(wire), blobs);
        }
        match &state.outcome {
            Some((result, _)) => {
                subscriber.send(Reply::JobDone(done_reply(result)), Vec::new());
            }
            None => state.subscribers.push(subscriber),
        }
    }

    /// The pump: passes the job's events on until it ends, then records and
    /// announces the result.
    fn run(&self, job: RunningJob) {
        for event in job.events() {
            let mut state = self.state.lock().unwrap();
            state.progress.apply(&event);
            if state.subscribers.is_empty() {
                continue;
            }
            let (wire, blobs) = event_to_wire(event);
            state
                .subscribers
                .retain(|subscriber| subscriber.send(Reply::JobEvent(wire.clone()), blobs.clone()));
        }
        let result = job.wait().map(|_| ()).map_err(|e| WireError::from(&e));
        match &result {
            Ok(()) => log::info!("Analysis {} finished", self.id),
            Err(e) => log::info!("Analysis {} ended: {}", self.id, e.text()),
        }
        let mut state = self.state.lock().unwrap();
        for subscriber in std::mem::take(&mut state.subscribers) {
            subscriber.send(Reply::JobDone(done_reply(&result)), Vec::new());
        }
        state.outcome = Some((result, SystemTime::now()));
    }
}

impl Subscriber {
    /// A failed send means the connection is gone - the analysis goes on.
    fn send(&self, reply: Reply, blobs: Vec<Vec<u8>>) -> bool {
        encode_reply(self.request_id, reply, &blobs)
            .is_some_and(|frame| self.outgoing.send(frame).is_ok())
    }
}

fn done_reply(result: &Result<(), WireError>) -> Result<JobOutput, WireError> {
    result.clone().map(|()| JobOutput::default())
}

fn encode_reply(id: u64, reply: Reply, blobs: &[Vec<u8>]) -> Option<Vec<u8>> {
    frame::encode(&ServerMsg::Reply { id, reply }, blobs)
        .inspect_err(|e| log::error!("Could not encode job event: {e}"))
        .ok()
}

/// A random id, short enough to type (`evanalyzer cli attach <id>`).
fn new_id() -> Result<String, InternalErrors> {
    let mut bytes = [0u8; 8];
    getrandom::fill(&mut bytes)
        .map_err(|e| InternalErrors::Internal(format!("could not generate job id: {e}")))?;
    Ok(bytes.iter().map(|b| format!("{b:02x}")).collect())
}

/// "5 min ago", for messages.
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
    use crate::backend::remote::wire::frame;
    use crate::backend::remote::wire::protocol::WireProgressEvent;
    use std::sync::atomic::AtomicBool;
    use std::sync::mpsc::{self, Receiver};

    /// A job the test drives: send events, then the result.
    struct Controlled {
        events: Sender<ProgressEvent>,
        done: Sender<Result<JobOutput, InternalErrors>>,
    }

    fn controlled_job() -> (RunningJob, Controlled) {
        let (events_tx, events_rx) = mpsc::channel();
        let (done_tx, done_rx) = mpsc::channel::<Result<JobOutput, InternalErrors>>();
        let job = RunningJob::from_parts(
            events_rx,
            CancelHandle::new(Arc::new(AtomicBool::new(false))),
            PathBuf::from("/project/results/plate-3"),
            4,
            Box::new(move || done_rx.recv().unwrap_or(Err(InternalErrors::Cancelled))),
        );
        (
            job,
            Controlled {
                events: events_tx,
                done: done_tx,
            },
        )
    }

    impl Controlled {
        fn event(&self, event: ProgressEvent) {
            self.events.send(event).unwrap();
        }

        /// Ends the job with `result` (closing its event stream first, as a
        /// real job does).
        fn finish(self, result: Result<JobOutput, InternalErrors>) {
            drop(self.events);
            self.done.send(result).unwrap();
        }
    }

    fn start(registry: &JobRegistry) -> (Arc<JobEntry>, Controlled) {
        let (job, control) = controlled_job();
        (registry.start(|| Ok(job)).unwrap(), control)
    }

    /// What a subscriber received, in a form tests can compare.
    #[derive(Debug, PartialEq)]
    enum Got {
        Started(Option<String>),
        Event(String),
        Done(Result<(), String>),
    }

    fn next(rx: &Receiver<Vec<u8>>, request_id: u64) -> Got {
        let bytes = rx
            .recv_timeout(Duration::from_secs(5))
            .expect("a message for the subscriber");
        let frame::Frame { msg, .. } = frame::decode::<ServerMsg>(&bytes).unwrap();
        let ServerMsg::Reply { id, reply } = msg else {
            panic!("not a reply");
        };
        assert_eq!(id, request_id);
        match reply {
            Reply::JobStarted { job_id, .. } => Got::Started(job_id),
            Reply::JobEvent(event) => Got::Event(describe(&event)),
            Reply::JobDone(result) => Got::Done(result.map(|_| ()).map_err(|e| e.text().into())),
            _ => panic!("unexpected reply"),
        }
    }

    fn describe(event: &WireProgressEvent) -> String {
        match event {
            WireProgressEvent::Started { total } => format!("started {total}"),
            WireProgressEvent::TilesScheduled { total_tiles } => format!("tiles 0/{total_tiles}"),
            WireProgressEvent::TileCompleted {
                tile_index,
                total_tiles,
                ..
            } => format!("tiles {tile_index}/{total_tiles}"),
            WireProgressEvent::ImageCompleted { index, total, .. } => {
                format!("image {index}/{total}")
            }
            WireProgressEvent::ImageFailed { path } => format!("failed {}", path.display()),
            WireProgressEvent::Finished => "finished".into(),
            _ => "other".into(),
        }
    }

    fn image_done(index: usize) -> ProgressEvent {
        ProgressEvent::ImageCompleted {
            index,
            total: 3,
            path: PathBuf::from(format!("/images/{index}.tif")),
        }
    }

    /// Waits until the pump has recorded the end of the job.
    fn wait_until_finished(registry: &JobRegistry) {
        for _ in 0..500 {
            if !registry.any_running() {
                return;
            }
            std::thread::sleep(Duration::from_millis(10));
        }
        panic!("job did not finish");
    }

    #[test]
    fn a_subscriber_gets_the_start_every_event_and_the_result() {
        let registry = JobRegistry::default();
        let (entry, control) = start(&registry);
        let (tx, rx) = mpsc::channel();
        entry.subscribe(tx, 7);
        assert_eq!(next(&rx, 7), Got::Started(Some(entry.id.clone())));

        control.event(ProgressEvent::Started { total: 3 });
        control.event(image_done(1));
        assert_eq!(next(&rx, 7), Got::Event("started 3".into()));
        assert_eq!(next(&rx, 7), Got::Event("image 1/3".into()));
        assert_eq!(
            registry.list()[0].state,
            JobState::Running { done: 1, total: 3 }
        );

        control.finish(Ok(JobOutput::default()));
        assert_eq!(next(&rx, 7), Got::Done(Ok(())));
        wait_until_finished(&registry);
        assert_eq!(registry.list()[0].state, JobState::Succeeded);
    }

    #[test]
    fn the_analysis_goes_on_when_its_client_disconnects() {
        let registry = JobRegistry::default();
        let (entry, control) = start(&registry);
        let (tx, rx) = mpsc::channel();
        entry.subscribe(tx, 1);
        drop(rx); // connection gone

        control.event(ProgressEvent::Started { total: 3 });
        control.event(image_done(1));
        control.event(image_done(2));
        assert!(registry.any_running());
        control.event(image_done(3));
        control.finish(Ok(JobOutput::default()));

        wait_until_finished(&registry);
        assert_eq!(registry.list()[0].state, JobState::Succeeded);
    }

    #[test]
    fn a_late_subscriber_gets_the_progress_so_far_then_live_events() {
        let registry = JobRegistry::default();
        let (entry, control) = start(&registry);
        control.event(ProgressEvent::Started { total: 3 });
        control.event(ProgressEvent::ImageFailed {
            path: PathBuf::from("/images/broken.tif"),
        });
        control.event(image_done(1));
        control.event(ProgressEvent::TilesScheduled { total_tiles: 8 });
        control.event(ProgressEvent::TileCompleted {
            tile_index: 5,
            total_tiles: 8,
            objects: Vec::new(),
        });
        // Let the pump take them in.
        for _ in 0..500 {
            if matches!(
                registry.list()[0].state,
                JobState::Running { done: 2, total: 3 }
            ) {
                break;
            }
            std::thread::sleep(Duration::from_millis(10));
        }

        let (tx, rx) = mpsc::channel();
        entry.subscribe(tx, 3);
        assert_eq!(next(&rx, 3), Got::Started(Some(entry.id.clone())));
        for expected in [
            "started 3",
            "failed /images/broken.tif",
            "image 1/3",
            "tiles 0/8",
            "tiles 5/8",
        ] {
            assert_eq!(next(&rx, 3), Got::Event(expected.into()));
        }

        control.event(image_done(2));
        assert_eq!(next(&rx, 3), Got::Event("image 2/3".into()));
        control.finish(Err(InternalErrors::Cancelled));
        assert!(matches!(next(&rx, 3), Got::Done(Err(_))));
        wait_until_finished(&registry);
        assert_eq!(registry.list()[0].state, JobState::Cancelled);
    }

    #[test]
    fn attaching_to_a_finished_analysis_answers_at_once() {
        let registry = JobRegistry::default();
        let (entry, control) = start(&registry);
        control.event(ProgressEvent::Started { total: 3 });
        control.event(image_done(1));
        control.event(ProgressEvent::Finished);
        control.finish(Err(InternalErrors::Internal("disk full".into())));
        wait_until_finished(&registry);

        let (tx, rx) = mpsc::channel();
        entry.subscribe(tx, 9);
        assert_eq!(next(&rx, 9), Got::Started(Some(entry.id.clone())));
        assert_eq!(next(&rx, 9), Got::Event("started 3".into()));
        assert_eq!(next(&rx, 9), Got::Event("image 1/3".into()));
        assert_eq!(next(&rx, 9), Got::Event("finished".into()));
        let Got::Done(Err(message)) = next(&rx, 9) else {
            panic!("expected the failure");
        };
        assert!(message.contains("disk full"), "{message}");
        assert!(matches!(
            &registry.list()[0].state,
            JobState::Failed(message) if message.contains("disk full")
        ));
    }

    #[test]
    fn only_one_analysis_runs_at_a_time() {
        let registry = JobRegistry::default();
        let (_, control) = start(&registry);

        let (second, _) = controlled_job();
        let error = registry.start(|| Ok(second)).err().unwrap().to_string();
        assert!(error.contains("already running"), "{error}");
        assert!(error.contains("plate-3"), "names the running one: {error}");

        control.finish(Ok(JobOutput::default()));
        wait_until_finished(&registry);
        let (third, _) = controlled_job();
        assert!(registry.start(|| Ok(third)).is_ok(), "free again");
    }

    #[test]
    fn a_failed_start_registers_nothing() {
        let registry = JobRegistry::default();
        let error = registry.start(|| Err(InternalErrors::InvalidArgument("no images".into())));
        assert!(error.is_err());
        assert!(registry.list().is_empty());
    }

    #[test]
    fn only_finished_analyses_can_be_forgotten() {
        let registry = JobRegistry::default();
        let (entry, control) = start(&registry);
        assert!(registry.forget(&entry.id).is_err(), "still running");

        control.finish(Ok(JobOutput::default()));
        wait_until_finished(&registry);
        registry.forget(&entry.id).unwrap();
        assert!(registry.list().is_empty());
        assert!(registry.get(&entry.id).is_none());
    }
}
