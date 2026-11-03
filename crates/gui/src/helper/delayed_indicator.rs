//! # delayed_indicator
//!
//! A busy indicator only for work that takes a while: shown after
//! [`SHOW_AFTER`] (fast work never flashes it), then kept at least
//! [`SHOW_FOR`] (no flash when the work ends right after). Lives on the UI
//! thread, typically in a `thread_local!`.
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-10-29
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.

use std::cell::Cell;
use std::rc::{Rc, Weak};
use std::time::Duration;

use crate::AppWindow;

pub(crate) const SHOW_AFTER: Duration = Duration::from_millis(400);
pub(crate) const SHOW_FOR: Duration = Duration::from_millis(400);

/// Shows or hides the indicator in the window.
type Show = fn(&AppWindow, bool);

/// Both delays on Slint's timers - one clock (a test's mocked one too; the
/// time shown was measured on the system's clock before, which a slow test
/// run let pass between two lines).
struct Inner {
    busy: Cell<bool>,
    shown: Cell<bool>,
    /// Shown for [`SHOW_FOR`] already: may go.
    shown_long_enough: Cell<bool>,
    /// Shows it after [`SHOW_AFTER`].
    show_timer: slint::Timer,
    /// Ends [`SHOW_FOR`] after it was shown.
    shown_timer: slint::Timer,
    show: Show,
}

pub(crate) struct DelayedIndicator(Rc<Inner>);

impl DelayedIndicator {
    pub(crate) fn new(show: Show) -> Self {
        Self(Rc::new(Inner {
            busy: Cell::new(false),
            shown: Cell::new(false),
            shown_long_enough: Cell::new(false),
            show_timer: slint::Timer::default(),
            shown_timer: slint::Timer::default(),
            show,
        }))
    }

    /// The work started (`true`) or ended (`false`).
    pub(crate) fn set_busy(&self, ui: &slint::Weak<AppWindow>, busy: bool) {
        let inner = &self.0;
        inner.busy.set(busy);
        match (busy, inner.shown.get()) {
            // Shown already: stays.
            (true, true) => {}
            (true, false) => {
                let (ui, weak) = (ui.clone(), Rc::downgrade(inner));
                inner
                    .show_timer
                    .start(slint::TimerMode::SingleShot, SHOW_AFTER, move || {
                        let Some(inner) = weak.upgrade() else {
                            return;
                        };
                        if !inner.busy.get() || inner.shown.get() {
                            return;
                        }
                        inner.shown.set(true);
                        inner.shown_long_enough.set(false);
                        show(&inner, &ui, true);
                        let (ui, weak) = (ui.clone(), Rc::downgrade(&inner));
                        inner.shown_timer.start(
                            slint::TimerMode::SingleShot,
                            SHOW_FOR,
                            move || {
                                if let Some(inner) = Weak::upgrade(&weak) {
                                    inner.shown_long_enough.set(true);
                                    hide(&inner, &ui);
                                }
                            },
                        );
                    });
            }
            // Never shown: nothing to hide.
            (false, false) => inner.show_timer.stop(),
            // Shown long enough: goes now - else when it has been.
            (false, true) => {
                if inner.shown_long_enough.get() {
                    hide(inner, ui);
                }
            }
        }
    }
}

fn show(inner: &Inner, ui: &slint::Weak<AppWindow>, on: bool) {
    if let Some(ui) = ui.upgrade() {
        (inner.show)(&ui, on);
    }
}

/// Hides it unless the work goes on.
fn hide(inner: &Inner, ui: &slint::Weak<AppWindow>) {
    if !inner.busy.get() {
        inner.shown.set(false);
        show(inner, ui, false);
    }
}
