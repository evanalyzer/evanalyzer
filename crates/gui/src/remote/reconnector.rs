//! # reconnector
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-10-10
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.

use std::{
    sync::{Arc, Mutex},
    time::Duration,
};

use evanalyzer_app::backends::Backend;
use evanalyzer_gui_slint::{AppWindow, ConnectionState};
use slint::ComponentHandle;

/// Gets a dropped server connection back: tries after 2, 5, 10, then every
/// 30 seconds, or at once on "Reconnect now". Gives up when the server says
/// the session has ended (the user's worker stopped) or the certificate
/// changed - those need the user.
pub struct Reconnector {
    pub ui: slint::Weak<AppWindow>,
    pub backend: Arc<dyn Backend>,
    pub on_reconnected: Box<dyn Fn() + Send + Sync>,
    /// Wakes the waiting reconnect thread; `Some` while it runs.
    pub try_now: Mutex<Option<std::sync::mpsc::Sender<()>>>,
    /// Set when the window moved on to another backend ("Disconnect", or a
    /// connection to another server): no more attempts.
    pub stopped: std::sync::atomic::AtomicBool,
}

impl Reconnector {
    /// Starts reconnecting, unless already at it.
    pub fn start(self: &Arc<Self>) {
        if self.stopped.load(std::sync::atomic::Ordering::SeqCst) {
            return;
        }
        let mut try_now = self.try_now.lock().unwrap();
        if try_now.is_some() {
            return;
        }
        let (wake, woken) = std::sync::mpsc::channel();
        *try_now = Some(wake);
        let this = Arc::clone(self);
        std::thread::spawn(move || {
            this.run(woken);
            *this.try_now.lock().unwrap() = None;
        });
    }

    /// Ends the attempts - the window doesn't use this backend any more.
    pub fn stop(&self) {
        self.stopped
            .store(true, std::sync::atomic::Ordering::SeqCst);
        // Dropping the sender wakes the waiting thread, which then ends.
        *self.try_now.lock().unwrap() = None;
    }

    pub fn try_now(&self) {
        if let Some(wake) = self.try_now.lock().unwrap().as_ref() {
            let _ = wake.send(());
        }
    }

    fn run(&self, woken: std::sync::mpsc::Receiver<()>) {
        const DELAYS: [u64; 4] = [2, 5, 10, 30];
        for attempt in 0.. {
            let delay = DELAYS[attempt.min(DELAYS.len() - 1)];
            self.show(&format!("reconnecting in {delay} s."), true);
            if let Err(std::sync::mpsc::RecvTimeoutError::Disconnected) =
                woken.recv_timeout(Duration::from_secs(delay))
            {
                return;
            }
            if self.stopped.load(std::sync::atomic::Ordering::SeqCst) {
                return;
            }
            self.show("reconnecting ...", true);
            match self.backend.reconnect() {
                Ok(()) if self.stopped.load(std::sync::atomic::Ordering::SeqCst) => return,
                Ok(()) => {
                    log::info!("Connection to the server is back");
                    let ui = self.ui.clone();
                    let _ = crate::helper::ui_thread::invoke_from_event_loop(move || {
                        if let Some(ui) = ui.upgrade() {
                            ui.global::<ConnectionState>().set_connected(true);
                        }
                    });
                    (self.on_reconnected)();
                    return;
                }
                // The server answered, but won't take us back.
                Err(e @ evanalyzer_cfg::core_types::InternalErrors::InvalidArgument(_)) => {
                    log::error!("Cannot reconnect: {e}");
                    self.show(&format!("{e}. Restart EVAnalyzer to log in again."), false);
                    return;
                }
                Err(e) => log::warn!("Reconnecting failed: {e}"),
            }
        }
    }

    fn show(&self, message: &str, can_reconnect: bool) {
        let ui = self.ui.clone();
        let message = message.to_string();
        let _ = crate::helper::ui_thread::invoke_from_event_loop(move || {
            if let Some(ui) = ui.upgrade() {
                let state = ui.global::<ConnectionState>();
                state.set_lost_message(message.into());
                state.set_can_reconnect(can_reconnect);
            }
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::editor::test_support::test_ui_window;
    use crate::helper::ui_thread::drain_ui_queue;
    use evanalyzer_app::ai_learning::{RunningTraining, StartTrainingError, TrainingRequest};
    use evanalyzer_app::analysis::{AnalysisRequest, RunningJob};
    use evanalyzer_app::backends::SystemInfo;
    use evanalyzer_app::fs::FileSystem;
    use evanalyzer_app::global::AppSettings;
    use evanalyzer_app::images::{ImageMeta, ImageSource};
    use evanalyzer_app::preview::{PreviewRequest, StartPreviewError};
    use evanalyzer_app::results::ResultsSource;
    use evanalyzer_app::templates::TemplateFolders;
    use evanalyzer_cfg::core_types::InternalErrors;
    use std::collections::VecDeque;
    use std::path::Path;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{OnceLock, Weak};

    /// A server connection that answers `reconnect` from a script, "down"
    /// once the script is done. Nothing else is used.
    #[derive(Default)]
    struct Server {
        answers: Mutex<VecDeque<Result<(), InternalErrors>>>,
        attempts: AtomicUsize,
        /// Stopped while the attempt is under way.
        stop_during: OnceLock<Weak<Reconnector>>,
        attempted: Mutex<Option<std::sync::mpsc::Sender<()>>>,
    }

    impl Backend for Server {
        fn reconnect(&self) -> Result<(), InternalErrors> {
            self.attempts.fetch_add(1, Ordering::SeqCst);
            if let Some(reconnector) = self.stop_during.get().and_then(Weak::upgrade) {
                reconnector.stop();
            }
            if let Some(attempted) = self.attempted.lock().unwrap().as_ref() {
                let _ = attempted.send(());
            }
            self.answers
                .lock()
                .unwrap()
                .pop_front()
                .unwrap_or_else(|| Err(InternalErrors::Internal("down".into())))
        }
        fn start_analysis(&self, _: AnalysisRequest) -> Result<RunningJob, InternalErrors> {
            unreachable!()
        }
        fn start_preview(&self, _: PreviewRequest) -> Result<RunningJob, StartPreviewError> {
            unreachable!()
        }
        fn start_training(
            &self,
            _: TrainingRequest,
        ) -> Result<RunningTraining, StartTrainingError> {
            unreachable!()
        }
        fn open_image(&self, _: &Path) -> Result<Arc<dyn ImageSource>, InternalErrors> {
            unreachable!()
        }
        fn open_results(&self, _: &Path) -> Result<Arc<dyn ResultsSource>, InternalErrors> {
            unreachable!()
        }
        fn read_image_meta(&self, _: &Path) -> Result<ImageMeta, InternalErrors> {
            unreachable!()
        }
        fn template_folders(&self) -> Result<TemplateFolders, InternalErrors> {
            unreachable!()
        }
        fn files(&self) -> &dyn FileSystem {
            unreachable!()
        }
        fn description(&self) -> String {
            "test server".into()
        }
        fn system_info(&self) -> Result<SystemInfo, InternalErrors> {
            unreachable!()
        }
        fn image_formats(&self) -> Vec<String> {
            Vec::new()
        }
        fn load_app_settings(&self) -> Result<AppSettings, InternalErrors> {
            unreachable!()
        }
        fn save_app_settings(&self, _: &AppSettings) -> Result<(), InternalErrors> {
            unreachable!()
        }
    }

    struct Fixture {
        ui: AppWindow,
        server: Arc<Server>,
        reconnected: Arc<AtomicUsize>,
        reconnector: Arc<Reconnector>,
    }

    fn fixture(answers: Vec<Result<(), InternalErrors>>) -> Fixture {
        let ui = test_ui_window();
        ui.global::<ConnectionState>().set_connected(false);
        let server = Arc::new(Server {
            answers: Mutex::new(answers.into()),
            ..Default::default()
        });
        let reconnected = Arc::new(AtomicUsize::new(0));
        let counter = Arc::clone(&reconnected);
        let reconnector = Arc::new(Reconnector {
            ui: ui.as_weak(),
            backend: server.clone(),
            on_reconnected: Box::new(move || {
                counter.fetch_add(1, Ordering::SeqCst);
            }),
            try_now: Mutex::new(None),
            stopped: Default::default(),
        });
        Fixture {
            ui,
            server,
            reconnected,
            reconnector,
        }
    }

    impl Fixture {
        /// Runs the attempts on this thread: `wakes` "Reconnect now"s
        /// waiting, then the window gone from this backend (`hang_up`) or
        /// not. Applies the UI updates.
        fn run(&self, wakes: usize, hang_up: bool) {
            let (wake, woken) = std::sync::mpsc::channel();
            for _ in 0..wakes {
                wake.send(()).unwrap();
            }
            if hang_up {
                drop(wake);
                self.reconnector.run(woken);
            } else {
                self.reconnector.run(woken);
                drop(wake);
            }
            drain_ui_queue();
        }

        fn attempts(&self) -> usize {
            self.server.attempts.load(Ordering::SeqCst)
        }

        fn state(&self) -> (bool, String, bool) {
            let state = self.ui.global::<ConnectionState>();
            (
                state.get_connected(),
                state.get_lost_message().to_string(),
                state.get_can_reconnect(),
            )
        }
    }

    /// Back at the first attempt: connected again, told once.
    #[test]
    fn a_successful_attempt_marks_the_window_connected_and_reports_it_once() {
        let f = fixture(vec![Ok(())]);
        f.run(1, false);
        assert_eq!(f.attempts(), 1);
        assert_eq!(f.reconnected.load(Ordering::SeqCst), 1);
        assert_eq!(f.state(), (true, "reconnecting ...".into(), true));
    }

    /// A failed attempt is retried; the next one gets it back.
    #[test]
    fn a_failed_attempt_is_retried_until_the_server_takes_us_back() {
        let f = fixture(vec![
            Err(InternalErrors::Internal("refused".into())),
            Ok(()),
        ]);
        f.run(2, false);
        assert_eq!(f.attempts(), 2);
        assert_eq!(f.reconnected.load(Ordering::SeqCst), 1);
        assert!(f.state().0);
    }

    /// The server says no (session ended): no more attempts, the user is
    /// told to log in again and can't reconnect.
    #[test]
    fn a_refusal_ends_the_attempts_and_asks_for_a_new_login() {
        let refusal = InternalErrors::InvalidArgument("Session ended".into());
        let message = format!("{refusal}. Restart EVAnalyzer to log in again.");
        let f = fixture(vec![Err(refusal)]);
        // More wakes than attempts: none after the refusal.
        f.run(3, false);
        assert_eq!(f.attempts(), 1);
        assert_eq!(f.reconnected.load(Ordering::SeqCst), 0);
        assert_eq!(f.state(), (false, message, false));
    }

    /// The waits grow 2, 5, 10, then stay at 30 seconds.
    #[test]
    fn the_wait_before_each_attempt_grows_to_thirty_seconds() {
        for (failed, delay) in [(0, 2), (1, 5), (2, 10), (3, 30), (4, 30), (6, 30)] {
            let f = fixture(Vec::new());
            f.run(failed, true);
            assert_eq!(f.attempts(), failed, "{failed} attempts made");
            assert_eq!(
                f.state(),
                (false, format!("reconnecting in {delay} s."), true),
                "after {failed} failed attempts"
            );
        }
    }

    /// Stopped (the window moved on): a wake tries nothing.
    #[test]
    fn a_stopped_reconnector_makes_no_attempt() {
        let f = fixture(vec![Ok(())]);
        f.reconnector.stop();
        f.run(1, false);
        assert_eq!(f.attempts(), 0);
        assert_eq!(f.reconnected.load(Ordering::SeqCst), 0);
        assert!(!f.state().0);
    }

    /// Stopped while the attempt ran: its success is ignored - the window
    /// uses another backend by now.
    #[test]
    fn a_success_after_stopping_is_not_reported() {
        let f = fixture(vec![Ok(())]);
        f.server
            .stop_during
            .set(Arc::downgrade(&f.reconnector))
            .unwrap();
        f.run(1, false);
        assert_eq!(f.attempts(), 1);
        assert_eq!(f.reconnected.load(Ordering::SeqCst), 0);
        assert!(!f.state().0, "not marked connected");
    }

    /// Waits for the background thread's next attempt, 10 s at most.
    fn next_attempt(attempted: &std::sync::mpsc::Receiver<()>) {
        attempted
            .recv_timeout(Duration::from_secs(10))
            .expect("an attempt was made");
    }

    /// Waits until only the test holds the reconnector: its thread ended.
    fn thread_ended(reconnector: &Arc<Reconnector>) {
        let deadline = std::time::Instant::now() + Duration::from_secs(10);
        while Arc::strong_count(reconnector) > 1 {
            assert!(std::time::Instant::now() < deadline, "thread still runs");
            std::thread::yield_now();
        }
    }

    /// "Reconnect now" wakes the waiting thread at once; a second start
    /// while it runs starts no other; stopping ends it.
    #[test]
    fn reconnect_now_wakes_the_one_running_thread_and_stop_ends_it() {
        let f = fixture(Vec::new());
        let (attempted, attempts) = std::sync::mpsc::channel();
        *f.server.attempted.lock().unwrap() = Some(attempted);

        f.reconnector.start();
        f.reconnector.start();
        assert_eq!(Arc::strong_count(&f.reconnector), 2, "one thread");
        f.reconnector.try_now();
        next_attempt(&attempts);
        f.reconnector.try_now();
        next_attempt(&attempts);
        assert_eq!(f.attempts(), 2);

        f.reconnector.stop();
        thread_ended(&f.reconnector);
        assert!(f.reconnector.try_now.lock().unwrap().is_none());
        assert_eq!(f.attempts(), 2, "none after stopping");
        // Stopped for good: neither starts again.
        f.reconnector.start();
        assert!(f.reconnector.try_now.lock().unwrap().is_none());
        f.reconnector.try_now();
    }

    /// Back again: the thread ends by itself, so a later drop can start
    /// a new one.
    #[test]
    fn after_reconnecting_a_new_drop_starts_a_new_thread() {
        let f = fixture(vec![Ok(()), Ok(())]);
        let (attempted, attempts) = std::sync::mpsc::channel();
        *f.server.attempted.lock().unwrap() = Some(attempted);

        f.reconnector.start();
        f.reconnector.try_now();
        next_attempt(&attempts);
        thread_ended(&f.reconnector);
        assert!(f.reconnector.try_now.lock().unwrap().is_none());
        assert_eq!(f.reconnected.load(Ordering::SeqCst), 1);

        f.reconnector.start();
        f.reconnector.try_now();
        next_attempt(&attempts);
        thread_ended(&f.reconnector);
        assert_eq!(f.reconnected.load(Ordering::SeqCst), 2);
    }
}
