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
use std::time::{Duration, Instant};

use crate::AppWindow;

pub(crate) const SHOW_AFTER: Duration = Duration::from_millis(400);
pub(crate) const SHOW_FOR: Duration = Duration::from_millis(400);

/// Shows or hides the indicator in the window.
type Show = fn(&AppWindow, bool);

struct Inner {
    busy: Cell<bool>,
    shown_since: Cell<Option<Instant>>,
    timer: slint::Timer,
    show: Show,
}

pub(crate) struct DelayedIndicator(Rc<Inner>);

impl DelayedIndicator {
    pub(crate) fn new(show: Show) -> Self {
        Self(Rc::new(Inner {
            busy: Cell::new(false),
            shown_since: Cell::new(None),
            timer: slint::Timer::default(),
            show,
        }))
    }

    /// The work started (`true`) or ended (`false`).
    pub(crate) fn set_busy(&self, ui: &slint::Weak<AppWindow>, busy: bool) {
        let inner = &self.0;
        inner.busy.set(busy);
        match (busy, inner.shown_since.get()) {
            // Shown already: stays (a pending hide is off).
            (true, Some(_)) => inner.timer.stop(),
            (true, None) => {
                let (ui, weak) = (ui.clone(), Rc::downgrade(inner));
                inner
                    .timer
                    .start(slint::TimerMode::SingleShot, SHOW_AFTER, move || {
                        if let Some(inner) = weak.upgrade() {
                            if inner.busy.get() && inner.shown_since.get().is_none() {
                                inner.shown_since.set(Some(Instant::now()));
                                show(&inner, &ui, true);
                            }
                        }
                    });
            }
            // Never shown: nothing to hide.
            (false, None) => inner.timer.stop(),
            (false, Some(since)) => match SHOW_FOR.checked_sub(since.elapsed()) {
                None => hide(inner, ui),
                Some(left) => {
                    let (ui, weak) = (ui.clone(), Rc::downgrade(inner));
                    inner
                        .timer
                        .start(slint::TimerMode::SingleShot, left, move || {
                            if let Some(inner) = Weak::upgrade(&weak) {
                                hide(&inner, &ui);
                            }
                        });
                }
            },
        }
    }
}

fn show(inner: &Inner, ui: &slint::Weak<AppWindow>, on: bool) {
    if let Some(ui) = ui.upgrade() {
        (inner.show)(&ui, on);
    }
}

fn hide(inner: &Inner, ui: &slint::Weak<AppWindow>) {
    if !inner.busy.get() {
        inner.shown_since.set(None);
        show(inner, ui, false);
    }
}
