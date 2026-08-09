//! Crash visibility for the packaged app.
//!
//! Logging today (see the binary's `main.rs`) goes to stderr via
//! `env_logger`, which is fine for a terminal but invisible once the app is
//! double-clicked as a packaged binary with no attached console - a panic
//! just makes the window disappear, with nothing a user could attach to a
//! bug report. [`install_panic_hook`] appends every panic's message,
//! location, and backtrace to a log file in the user's per-app data
//! directory instead.

use crate::settings::get_user_folder;
use std::fs::OpenOptions;
use std::io::Write;
use std::panic::PanicHookInfo;
use std::path::{Path, PathBuf};

fn crash_log_path() -> PathBuf {
    get_user_folder().join("crash.log")
}

/// Installs a panic hook that appends a timestamped entry (message,
/// source location, backtrace) to `<user folder>/crash.log` for every panic,
/// anywhere in the process, in addition to running whatever hook was
/// previously installed - by default, Rust's own stderr printer, so a
/// developer running from a terminal still sees the usual output.
///
/// Must be called once, as early as possible in `main()` (before spawning
/// any other threads, so it's in place for panics on all of them too).
pub fn install_panic_hook() {
    let previous_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        write_crash_log_entry_to(&crash_log_path(), info);
        previous_hook(info);
    }));
}

/// Formats and appends one crash log entry to `path`, creating the file if
/// it doesn't exist yet. Failures are swallowed - this already runs inside a
/// panic hook, where there's no reasonable way to surface a second error,
/// and the chained default hook's stderr output remains the fallback.
fn write_crash_log_entry_to(path: &Path, info: &PanicHookInfo<'_>) {
    let Ok(mut file) = OpenOptions::new().create(true).append(true).open(path) else {
        return;
    };
    let _ = file.write_all(format_crash_log_entry(info).as_bytes());
}

/// Renders one crash log entry: timestamp, source location, panic message,
/// and a full backtrace (captured unconditionally - unlike
/// `Backtrace::capture()`, `force_capture()` doesn't depend on the user
/// having set `RUST_BACKTRACE`, which nobody running a packaged app will
/// have done).
fn format_crash_log_entry(info: &PanicHookInfo<'_>) -> String {
    let timestamp = chrono::Local::now().format("%Y-%m-%d %H:%M:%S%.3f %z");
    let location = info
        .location()
        .map(|l| format!("{}:{}:{}", l.file(), l.line(), l.column()))
        .unwrap_or_else(|| "<unknown location>".to_string());
    let message = panic_message(info);
    let backtrace = std::backtrace::Backtrace::force_capture();

    format!(
        "\n=== EVAnalyzer crash: {timestamp} ===\n\
         Location: {location}\n\
         Message: {message}\n\
         Backtrace:\n{backtrace}\n"
    )
}

/// Extracts a panic's message the same way Rust's own default hook does:
/// almost every panic (`.unwrap()`, `panic!()`, `assert!()`, indexing) has a
/// `&str` or `String` payload.
fn panic_message(info: &PanicHookInfo<'_>) -> String {
    if let Some(s) = info.payload().downcast_ref::<&str>() {
        (*s).to_string()
    } else if let Some(s) = info.payload().downcast_ref::<String>() {
        s.clone()
    } else {
        "<non-string panic payload>".to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Mutex, OnceLock};

    // `std::panic::set_hook`/`take_hook` mutate process-global state, so
    // tests that install a hook must not run concurrently with each other
    // (or with any other test that happens to panic while one is
    // installed) - hold this for the duration of any such test.
    fn hook_test_lock() -> &'static Mutex<()> {
        static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        LOCK.get_or_init(|| Mutex::new(()))
    }

    /// Runs `f` inside `catch_unwind` with a temporary hook that captures
    /// the formatted crash log entry instead of printing to stderr, then
    /// restores whatever hook was installed before this test ran.
    ///
    /// `set_hook`/`take_hook` are process-global, so this hook can also fire
    /// for panics on *other* threads running unrelated tests in parallel
    /// (e.g. `project_owner`'s poisoned-lock test, which panics on purpose)
    /// - `hook_test_lock` only serializes this module's own tests against
    /// each other, not against those. Filtering by thread id here keeps
    /// such cross-thread panics from clobbering `captured` with the wrong
    /// message.
    fn capture_crash_log_entry(f: impl FnOnce() + std::panic::UnwindSafe) -> String {
        let _guard = hook_test_lock().lock().unwrap();

        let this_thread = std::thread::current().id();
        let captured: Arc<Mutex<Option<String>>> = Arc::new(Mutex::new(None));
        let captured_for_hook = Arc::clone(&captured);
        let previous = std::panic::take_hook();
        std::panic::set_hook(Box::new(move |info| {
            if std::thread::current().id() == this_thread {
                *captured_for_hook.lock().unwrap() = Some(format_crash_log_entry(info));
            }
        }));

        let result = std::panic::catch_unwind(f);
        std::panic::set_hook(previous);

        assert!(result.is_err(), "test closure was expected to panic");
        captured
            .lock()
            .unwrap()
            .take()
            .expect("hook should have run")
    }

    #[test]
    fn format_crash_log_entry_includes_message_and_location() {
        let entry = capture_crash_log_entry(|| panic!("boom"));

        assert!(entry.contains("boom"), "entry was: {entry}");
        assert!(entry.contains("crash_log.rs"), "entry was: {entry}");
        assert!(
            entry.contains("=== EVAnalyzer crash:"),
            "entry was: {entry}"
        );
        assert!(entry.contains("Backtrace:"), "entry was: {entry}");
    }

    #[test]
    fn format_crash_log_entry_handles_a_string_payload() {
        // `panic!("{msg}")` with a runtime-built format string produces an
        // owned `String` payload, unlike a `&'static str` literal.
        let msg = String::from("dynamic message");
        let entry = capture_crash_log_entry(move || panic!("{msg}"));

        assert!(entry.contains("dynamic message"), "entry was: {entry}");
    }

    #[test]
    fn write_crash_log_entry_to_appends_across_multiple_panics() {
        let _guard = hook_test_lock().lock().unwrap();
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("crash.log");

        let previous = std::panic::take_hook();
        std::panic::set_hook(Box::new({
            let path = path.clone();
            move |info| write_crash_log_entry_to(&path, info)
        }));

        let _ = std::panic::catch_unwind(|| panic!("first"));
        let _ = std::panic::catch_unwind(|| panic!("second"));

        std::panic::set_hook(previous);

        let contents = std::fs::read_to_string(&path).unwrap();
        assert!(contents.contains("first"), "contents were: {contents}");
        assert!(contents.contains("second"), "contents were: {contents}");
    }
}
