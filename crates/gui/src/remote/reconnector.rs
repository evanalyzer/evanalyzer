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
}

impl Reconnector {
    /// Starts reconnecting, unless already at it.
    pub fn start(self: &Arc<Self>) {
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
            self.show("reconnecting ...", true);
            match self.backend.reconnect() {
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
