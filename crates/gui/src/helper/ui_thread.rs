//! The one place the GUI hands work to the Slint event loop, and starts the
//! short-lived background tasks whose results come back through it.
//!
//! In the app these are exactly `slint::invoke_from_event_loop` and
//! `std::thread::spawn`. Unit tests run on Slint's headless test platform,
//! which has no event loop, so there the UI update inside the closure would
//! never run - a controller's whole "push this to the window" half was
//! untestable. Under `cfg(test)`:
//!
//! - [`invoke_from_event_loop`] queues the closure instead,
//! - a thread started with [`spawn`] shares the queue of the thread that
//!   started it, so results of background work (a folder listing, a loaded
//!   project) end up in the test's queue too,
//! - [`drain_ui_queue`] runs the queued updates on the test thread - in the
//!   order the real event loop would, and only after the caller returned -
//!   until nothing is queued and no task started with [`spawn`] is running.
//!
//! Every test thread has its own queue, so parallel tests never see each
//! other's updates.

/// Runs `func` on the UI thread, like [`slint::invoke_from_event_loop`].
pub(crate) fn invoke_from_event_loop(
    func: impl FnOnce() + Send + 'static,
) -> Result<(), slint::EventLoopError> {
    #[cfg(not(test))]
    {
        slint::invoke_from_event_loop(func)
    }
    #[cfg(test)]
    {
        test_queue::current()
            .queue
            .lock()
            .unwrap()
            .push_back(Box::new(func));
        Ok(())
    }
}

/// Starts a short-lived background task, like [`std::thread::spawn`].
/// Long-running workers that live as long as the app use `std::thread`
/// directly - a test would otherwise wait for them forever.
pub(crate) fn spawn<F, T>(f: F) -> std::thread::JoinHandle<T>
where
    F: FnOnce() -> T + Send + 'static,
    T: Send + 'static,
{
    #[cfg(not(test))]
    {
        std::thread::spawn(f)
    }
    #[cfg(test)]
    {
        test_queue::spawn(f)
    }
}

/// Runs every UI update queued by this test thread and its [`spawn`]ed
/// tasks, waiting for those tasks to finish. Panics after 10 s of waiting,
/// so a hung task fails the test instead of hanging it.
#[cfg(test)]
pub(crate) fn drain_ui_queue() {
    test_queue::drain();
}

/// Gives this test thread an empty queue of its own. The test harness reuses
/// threads, so without this a test could run updates a previous test on the
/// same thread queued (or wait for its tasks). Tasks still running from
/// before keep reporting to the old queue.
#[cfg(test)]
pub(crate) fn fresh_test_queue() {
    test_queue::reset();
}

#[cfg(test)]
mod test_queue {
    use std::cell::RefCell;
    use std::collections::VecDeque;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Arc, Mutex};
    use std::time::{Duration, Instant};

    type Update = Box<dyn FnOnce() + Send>;

    #[derive(Clone, Default)]
    pub(super) struct Context {
        pub(super) queue: Arc<Mutex<VecDeque<Update>>>,
        running: Arc<AtomicUsize>,
    }

    thread_local! {
        static CONTEXT: RefCell<Option<Context>> = const { RefCell::new(None) };
    }

    pub(super) fn reset() {
        CONTEXT.with(|c| *c.borrow_mut() = Some(Context::default()));
    }

    pub(super) fn current() -> Context {
        CONTEXT.with(|c| c.borrow_mut().get_or_insert_with(Context::default).clone())
    }

    struct Running(Arc<AtomicUsize>);
    impl Drop for Running {
        fn drop(&mut self) {
            self.0.fetch_sub(1, Ordering::SeqCst);
        }
    }

    pub(super) fn spawn<F, T>(f: F) -> std::thread::JoinHandle<T>
    where
        F: FnOnce() -> T + Send + 'static,
        T: Send + 'static,
    {
        let context = current();
        context.running.fetch_add(1, Ordering::SeqCst);
        std::thread::spawn(move || {
            // Decrements even if `f` panics.
            let _running = Running(Arc::clone(&context.running));
            CONTEXT.with(|c| *c.borrow_mut() = Some(context));
            f()
        })
    }

    pub(super) fn drain() {
        let context = current();
        let deadline = Instant::now() + Duration::from_secs(10);
        loop {
            let next = context.queue.lock().unwrap().pop_front();
            match next {
                Some(update) => update(),
                None if context.running.load(Ordering::SeqCst) == 0 => {
                    // A task may have queued its result right before it ended.
                    if context.queue.lock().unwrap().is_empty() {
                        return;
                    }
                }
                None => {
                    assert!(
                        Instant::now() < deadline,
                        "drain_ui_queue: background tasks still running after 10 s"
                    );
                    std::thread::sleep(Duration::from_millis(1));
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Mutex};

    #[test]
    fn updates_run_in_order_only_when_drained() {
        let log = Arc::new(Mutex::new(Vec::new()));
        for i in 0..3 {
            let log = Arc::clone(&log);
            invoke_from_event_loop(move || log.lock().unwrap().push(i)).unwrap();
        }
        assert!(
            log.lock().unwrap().is_empty(),
            "nothing runs before draining"
        );
        drain_ui_queue();
        assert_eq!(*log.lock().unwrap(), [0, 1, 2]);
    }

    #[test]
    fn updates_queued_by_an_update_run_in_the_same_drain() {
        let log = Arc::new(Mutex::new(Vec::new()));
        let outer = Arc::clone(&log);
        invoke_from_event_loop(move || {
            outer.lock().unwrap().push("outer");
            let inner = Arc::clone(&outer);
            invoke_from_event_loop(move || inner.lock().unwrap().push("inner")).unwrap();
        })
        .unwrap();
        drain_ui_queue();
        assert_eq!(*log.lock().unwrap(), ["outer", "inner"]);
    }

    #[test]
    fn spawned_tasks_report_back_to_the_starting_thread_and_are_waited_for() {
        let log = Arc::new(Mutex::new(Vec::new()));
        let from_task = Arc::clone(&log);
        spawn(move || {
            std::thread::sleep(std::time::Duration::from_millis(30));
            let on_ui = Arc::clone(&from_task);
            invoke_from_event_loop(move || {
                on_ui.lock().unwrap().push(std::thread::current().id());
            })
            .unwrap();
        });
        drain_ui_queue();
        assert_eq!(
            *log.lock().unwrap(),
            [std::thread::current().id()],
            "the update ran here, after waiting for the task"
        );
    }

    #[test]
    fn tasks_spawned_by_tasks_share_the_queue_too() {
        let log = Arc::new(Mutex::new(0));
        let outer = Arc::clone(&log);
        spawn(move || {
            spawn(move || {
                invoke_from_event_loop(move || *outer.lock().unwrap() += 1).unwrap();
            });
        });
        drain_ui_queue();
        assert_eq!(*log.lock().unwrap(), 1);
    }

    #[test]
    fn a_fresh_queue_drops_what_was_queued_before() {
        let log = Arc::new(Mutex::new(0));
        let stale = Arc::clone(&log);
        invoke_from_event_loop(move || *stale.lock().unwrap() += 1).unwrap();
        fresh_test_queue();
        drain_ui_queue();
        assert_eq!(*log.lock().unwrap(), 0);
    }

    #[test]
    fn plain_std_threads_keep_their_own_queue() {
        let log = Arc::new(Mutex::new(Vec::new()));
        let from_thread = Arc::clone(&log);
        std::thread::spawn(move || {
            invoke_from_event_loop(move || from_thread.lock().unwrap().push(1)).unwrap();
        })
        .join()
        .unwrap();
        drain_ui_queue();
        assert!(log.lock().unwrap().is_empty());
    }
}
