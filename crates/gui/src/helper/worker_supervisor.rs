use std::any::Any;
use std::panic::{self, AssertUnwindSafe};
use std::time::Duration;

/// Runs `body` in a loop, catching panics instead of letting them kill the
/// thread for good. Worker threads (viewport rendering, pipeline execution)
/// are long-lived `-> !` loops with no supervisor above them - previously a
/// single panic (e.g. a malformed tile at the image edge) permanently killed
/// that subsystem for the rest of the session. `body` is expected to loop
/// forever on its own; a normal return is treated the same as a panic (it
/// means the loop exited some other way it shouldn't have) and is also
/// logged and restarted.
///
/// A short sleep between restarts avoids a tight spin loop burning a core if
/// the same panic fires immediately on every retry (e.g. a poisoned lock).
pub(crate) fn run_supervised(thread_name: &str, mut body: impl FnMut()) -> ! {
    loop {
        run_one_attempt(thread_name, &mut body);
        std::thread::sleep(Duration::from_millis(200));
    }
}

/// One catch-unwind-wrapped call to `body`, logging either a panic or an
/// unexpected normal return. Split out from [`run_supervised`] so the
/// catch-and-log behavior is unit-testable without looping forever.
fn run_one_attempt(thread_name: &str, body: &mut impl FnMut()) {
    match panic::catch_unwind(AssertUnwindSafe(body)) {
        Ok(()) => {
            log::error!("{thread_name} worker loop returned unexpectedly, restarting");
        }
        Err(payload) => {
            log::error!(
                "{thread_name} worker thread panicked, restarting: {}",
                panic_message(&payload)
            );
        }
    }
}

/// Best-effort extraction of a human-readable message from a caught panic
/// payload - covers the two payload shapes `panic!`/`.expect()`/`.unwrap()`
/// actually produce (`&str` and `String`); anything else is a custom
/// `panic_any` payload we can't stringify meaningfully.
pub(crate) fn panic_message(payload: &Box<dyn Any + Send>) -> String {
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
    fn a_panicking_attempt_does_not_propagate_and_a_later_attempt_still_runs() {
        let hook = panic::take_hook();
        panic::set_hook(Box::new(|_| {})); // silence the panic backtrace in test output

        let mut ran = false;
        run_one_attempt("test-worker", &mut || panic!("boom"));
        run_one_attempt("test-worker", &mut || ran = true);

        panic::set_hook(hook);

        assert!(ran, "supervisor stopped retrying after a single panic");
    }

    #[test]
    fn a_normal_return_is_treated_as_a_recoverable_failure_too() {
        // `body` is documented to never return normally (real worker loops are
        // `-> !`); this just confirms that if one somehow did, it's logged and
        // handled the same as a panic instead of the supervisor breaking.
        run_one_attempt("test-worker", &mut || {});
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
