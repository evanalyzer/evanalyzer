//! "File > Connect to server": the dialog that logs in to an `evanalyzer
//! server` and moves the window there (see
//! [`ProjectController::switch_backend`]).
//!
//! Before logging in, the dialog looks at the server's TLS certificate:
//! one a public authority signed is trusted as is; the server's own
//! self-signed one is shown for the user to confirm on the first
//! connection, then remembered (on this computer, with the recent servers)
//! - and a different one later gets a warning, like SSH does.

use crate::editor::project_controller::ProjectController;
use crate::{AppWindow, ConnectState, DialogType, GlobalAppState, RecentServerRow, UiState};
use evanalyzer_app::backends::remote::{RemoteBackend, TlsTrust, server_certificate};
use evanalyzer_app::global::RecentServer;
use log::info;
use slint::ComponentHandle;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

/// Dialog steps, as `ConnectState.step` numbers them.
const STEP_FORM: i32 = 0;
const STEP_CONFIRM: i32 = 1;
const STEP_CHANGED: i32 = 2;

pub struct ConnectionController {
    ui: slint::Weak<AppWindow>,
    app_state: Arc<UiState>,
    project_controller: Arc<ProjectController>,
    /// Counts connection attempts; "Cancel" counts too, so an attempt that
    /// finishes after it is dropped instead of switching the window.
    attempt: AtomicU64,
    /// The login waiting for the user to confirm the certificate.
    pending: Mutex<Option<Login>>,
}

/// What the user entered, and the certificate the server showed.
#[derive(Clone)]
struct Login {
    url: String,
    user: String,
    password: String,
    fingerprint: String,
}

impl ConnectionController {
    pub fn new(
        ui: slint::Weak<AppWindow>,
        app_state: Arc<UiState>,
        project_controller: Arc<ProjectController>,
    ) -> Self {
        Self {
            ui,
            app_state,
            project_controller,
            attempt: AtomicU64::new(0),
            pending: Mutex::new(None),
        }
    }

    pub fn attach_callbacks(self: &Arc<Self>) {
        let Some(ui) = self.ui.upgrade() else {
            return;
        };
        let state = ui.global::<ConnectState>();
        let this = Arc::clone(self);
        state.on_opened(move || this.opened());
        let this = Arc::clone(self);
        state.on_pick_recent(move |index| this.pick_recent(index as usize));
        let this = Arc::clone(self);
        state.on_connect(move || this.connect());
        let this = Arc::clone(self);
        state.on_trust(move || this.trust());
        let this = Arc::clone(self);
        state.on_back(move || {
            *this.pending.lock().unwrap() = None;
            this.show(|state| {
                state.set_step(STEP_FORM);
                state.set_error("".into());
            });
        });
        let this = Arc::clone(self);
        state.on_cancel(move || {
            this.attempt.fetch_add(1, Ordering::SeqCst);
            *this.pending.lock().unwrap() = None;
            this.show(|state| {
                state.set_busy(false);
                state.set_password("".into());
            });
        });
    }

    /// Fresh dialog: the recent servers, the latest one filled in.
    fn opened(&self) {
        let recent = self.app_state.recent_servers();
        let rows: Vec<RecentServerRow> = recent
            .iter()
            .map(|server| RecentServerRow {
                url: server.url.clone().into(),
                user: server.user.clone().into(),
            })
            .collect();
        let latest = recent.first().cloned();
        self.show(move |state| {
            state.set_recent(slint::ModelRc::new(slint::VecModel::from(rows)));
            if let Some(latest) = latest {
                state.set_url(latest.url.into());
                state.set_user(latest.user.into());
            }
            state.set_password("".into());
            state.set_error("".into());
            state.set_busy(false);
            state.set_step(STEP_FORM);
        });
    }

    fn pick_recent(&self, index: usize) {
        if let Some(server) = self.app_state.recent_servers().into_iter().nth(index) {
            self.show(move |state| {
                state.set_url(server.url.into());
                state.set_user(server.user.into());
            });
        }
    }

    /// "Connect": looks at the server's certificate, then logs in - or asks
    /// the user about the certificate first.
    fn connect(self: &Arc<Self>) {
        let Some(ui) = self.ui.upgrade() else {
            return;
        };
        let state = ui.global::<ConnectState>();
        let url = normalize_url(&state.get_url());
        let user = state.get_user().trim().to_string();
        let password = state.get_password().to_string();
        if url.is_empty() || user.is_empty() {
            return;
        }
        state.set_url(url.clone().into());
        state.set_busy(true);
        state.set_error("".into());
        let attempt = self.attempt.fetch_add(1, Ordering::SeqCst) + 1;
        let this = Arc::clone(self);
        crate::helper::ui_thread::spawn(move || {
            let certificate = match server_certificate(&url) {
                Ok(certificate) => certificate,
                Err(e) => return this.failed(attempt, &e.to_string()),
            };
            let Some(certificate) = certificate.filter(|c| !c.publicly_trusted) else {
                // ws://, or a certificate a public authority vouches for.
                return this.log_in(attempt, url, user, password, TlsTrust::default(), None);
            };
            let known = this
                .app_state
                .local_backend()
                .load_app_settings()
                .ok()
                .and_then(|settings| settings.known_fingerprint(&url).map(str::to_string));
            if known.as_deref() == Some(certificate.fingerprint.as_str()) {
                let trust = TlsTrust::Fingerprint(certificate.fingerprint.clone());
                return this.log_in(
                    attempt,
                    url,
                    user,
                    password,
                    trust,
                    Some(certificate.fingerprint),
                );
            }
            if this.attempt.load(Ordering::SeqCst) != attempt {
                return;
            }
            let fingerprint = certificate.fingerprint.clone();
            *this.pending.lock().unwrap() = Some(Login {
                url,
                user,
                password,
                fingerprint: certificate.fingerprint,
            });
            this.show(move |state| {
                state.set_busy(false);
                state.set_fingerprint(fingerprint.into());
                match known {
                    Some(known) => {
                        state.set_known_fingerprint(known.into());
                        state.set_step(STEP_CHANGED);
                    }
                    None => state.set_step(STEP_CONFIRM),
                }
            });
        });
    }

    /// The user vouched for the certificate shown: log in trusting it.
    fn trust(self: &Arc<Self>) {
        let Some(login) = self.pending.lock().unwrap().clone() else {
            return;
        };
        let attempt = self.attempt.fetch_add(1, Ordering::SeqCst) + 1;
        self.show(|state| {
            state.set_busy(true);
            state.set_error("".into());
        });
        let this = Arc::clone(self);
        crate::helper::ui_thread::spawn(move || {
            let trust = TlsTrust::Fingerprint(login.fingerprint.clone());
            this.log_in(
                attempt,
                login.url,
                login.user,
                login.password,
                trust,
                Some(login.fingerprint),
            );
        });
    }

    /// Logs in and, if that works (and wasn't cancelled meanwhile), moves
    /// the window to the server and remembers it.
    fn log_in(
        &self,
        attempt: u64,
        url: String,
        user: String,
        password: String,
        trust: TlsTrust,
        fingerprint: Option<String>,
    ) {
        let remote = match RemoteBackend::connect_with_login(&url, &user, &password, &trust) {
            Ok(remote) => remote,
            Err(e) => return self.failed(attempt, &e.to_string()),
        };
        if self.attempt.load(Ordering::SeqCst) != attempt {
            use evanalyzer_app::backends::Backend;
            remote.disconnect();
            return;
        }
        info!("Connected to {url} as {user}");
        *self.pending.lock().unwrap() = None;
        self.app_state.remember_server(RecentServer {
            url,
            user,
            fingerprint,
        });
        self.show(|state| {
            state.set_busy(false);
            state.set_password("".into());
            state.set_step(STEP_FORM);
        });
        let ui = self.ui.clone();
        crate::helper::ui_thread::invoke_from_event_loop(move || {
            if let Some(ui) = ui.upgrade() {
                ui.global::<GlobalAppState>()
                    .set_active_dialog(DialogType::None);
            }
        })
        .ok();
        self.project_controller.switch_backend(Arc::new(remote));
    }

    fn failed(&self, attempt: u64, message: &str) {
        if self.attempt.load(Ordering::SeqCst) != attempt {
            return;
        }
        log::warn!("Connecting failed: {message}");
        let message = message.to_string();
        self.show(move |state| {
            state.set_busy(false);
            state.set_error(message.into());
        });
    }

    /// Changes `ConnectState` on the UI thread.
    fn show(&self, change: impl FnOnce(ConnectState<'_>) + Send + 'static) {
        let ui = self.ui.clone();
        crate::helper::ui_thread::invoke_from_event_loop(move || {
            if let Some(ui) = ui.upgrade() {
                change(ui.global::<ConnectState>());
            }
        })
        .ok();
    }
}

/// `host[:port]` as typed means an encrypted connection; an explicit
/// `ws://`/`wss://` stays as it is.
fn normalize_url(typed: &str) -> String {
    let typed = typed.trim().trim_end_matches('/');
    if typed.is_empty() || typed.contains("://") {
        typed.to_string()
    } else {
        format!("wss://{typed}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::editor::test_support::{test_ui_windows, ui_state_with_windows};
    use crate::helper::ui_thread::drain_ui_queue;
    use evanalyzer_app::backends::remote::Worker;
    use std::io;
    use std::net::{Shutdown, TcpListener, TcpStream};
    use std::time::Duration;
    use tungstenite::Message;

    const TOKEN: &str = "session-token";

    /// Stands in for `evanalyzer server`: answers the login (password
    /// "pw"), then hands the connection to an in-process worker.
    fn login_gateway() -> String {
        let worker = Worker::bind("127.0.0.1:0", TOKEN.into()).unwrap();
        let worker_addr = worker.local_addr().unwrap();
        std::thread::spawn(move || {
            worker.run(Arc::new(
                evanalyzer_app::backends::local::LocalBackend::default(),
            ))
        });
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let url = format!("ws://{}", listener.local_addr().unwrap());
        std::thread::spawn(move || {
            for stream in listener.incoming().flatten() {
                std::thread::spawn(move || {
                    let mut client = tungstenite::accept(stream).unwrap();
                    client.send(Message::text("Welcome")).unwrap();
                    let login = client.read().unwrap().into_text().unwrap();
                    if !login.contains(r#""password":"pw""#) {
                        let refusal =
                            r#"{"response":"Error","msg":"Invalid username or password"}"#;
                        let _ = client.send(Message::text(refusal));
                        return;
                    }
                    let accepted = format!(
                        r#"{{"response":"Accepted","msg":"Logged in","session_token":"{TOKEN}"}}"#
                    );
                    client.send(Message::text(accepted)).unwrap();
                    let hello = client.read().unwrap();
                    let stream = TcpStream::connect(worker_addr).unwrap();
                    let (mut worker, _) =
                        tungstenite::client(format!("ws://{worker_addr}/"), stream).unwrap();
                    worker.send(hello).unwrap();
                    let (c, w) = (
                        client.get_ref().try_clone().unwrap(),
                        worker.get_ref().try_clone().unwrap(),
                    );
                    let (mut c_read, mut w_write) =
                        (c.try_clone().unwrap(), w.try_clone().unwrap());
                    std::thread::spawn(move || {
                        let _ = io::copy(&mut c_read, &mut w_write);
                        let _ = w_write.shutdown(Shutdown::Both);
                    });
                    let (mut w_read, mut c_write) = (w, c);
                    let _ = io::copy(&mut w_read, &mut c_write);
                    let _ = c_write.shutdown(Shutdown::Both);
                });
            }
        });
        url
    }

    struct Fixture {
        ui: AppWindow,
        _results_ui: crate::ResultsWindow,
        ui_state: Arc<UiState>,
        connection: Arc<ConnectionController>,
    }

    fn fixture() -> Fixture {
        let (ui, results_ui) = test_ui_windows();
        let ui_state = ui_state_with_windows(&ui, &results_ui, Default::default());
        let (_, project) = crate::editor::project_controller::tests::make_controller_on(
            ui.as_weak(),
            results_ui.as_weak(),
            ui_state.clone(),
        );
        let connection = Arc::new(ConnectionController::new(
            ui.as_weak(),
            ui_state.clone(),
            project,
        ));
        connection.attach_callbacks();
        Fixture {
            ui,
            _results_ui: results_ui,
            ui_state,
            connection,
        }
    }

    impl Fixture {
        fn state(&self) -> ConnectState<'_> {
            self.ui.global::<ConnectState>()
        }

        /// Clicks "Connect" with these entries and waits for the outcome.
        fn connect(&self, url: &str, user: &str, password: &str) {
            let state = self.state();
            state.set_url(url.into());
            state.set_user(user.into());
            state.set_password(password.into());
            state.invoke_connect();
            for _ in 0..500 {
                drain_ui_queue();
                if !self.state().get_busy() {
                    return;
                }
                std::thread::sleep(Duration::from_millis(10));
            }
            panic!("the connection attempt did not finish");
        }
    }

    #[test]
    fn a_failed_login_is_shown_in_the_dialog_and_nothing_changes() {
        let f = fixture();
        let url = login_gateway();
        f.ui.global::<GlobalAppState>()
            .set_active_dialog(DialogType::ConnectServer);

        f.connect(&url, "alice", "wrong");

        assert!(
            f.state()
                .get_error()
                .contains("Invalid username or password"),
            "{}",
            f.state().get_error()
        );
        assert!(!f.ui_state.backend().is_remote());
        assert_eq!(
            f.ui.global::<GlobalAppState>().get_active_dialog(),
            DialogType::ConnectServer,
            "stays open to try again"
        );
    }

    #[test]
    fn a_login_moves_the_window_to_the_server_and_remembers_it() {
        let f = fixture();
        let url = login_gateway();
        f.ui.global::<GlobalAppState>()
            .set_active_dialog(DialogType::ConnectServer);

        f.connect(&url, "alice", "pw");
        drain_ui_queue();

        assert!(
            f.ui_state.backend().is_remote(),
            "{}",
            f.state().get_error()
        );
        assert_eq!(f.ui_state.backend().user().as_deref(), Some("alice"));
        assert_eq!(
            f.ui.global::<GlobalAppState>().get_active_dialog(),
            DialogType::None
        );
        assert_eq!(f.state().get_password(), "", "not kept");
        let recent = f.ui_state.recent_servers();
        assert_eq!(recent[0].url, url);
        assert_eq!(recent[0].user, "alice");
        assert_eq!(recent[0].fingerprint, None, "ws:// has no certificate");

        // Next time the dialog offers it.
        f.state().set_url("".into());
        f.state().invoke_opened();
        drain_ui_queue();
        assert_eq!(f.state().get_url(), url.as_str());
        assert_eq!(f.state().get_user(), "alice");
        use slint::Model;
        assert_eq!(f.state().get_recent().row_count(), 1);
    }

    /// A TLS server with a fresh self-signed certificate that only shakes
    /// hands; returns its `wss://` URL and the certificate's fingerprint.
    fn tls_server() -> (String, String) {
        use std::io::Read;
        let generated =
            rcgen::generate_simple_self_signed(vec!["evanalyzer-server".to_string()]).unwrap();
        let fingerprint = ring::digest::digest(&ring::digest::SHA256, generated.cert.der())
            .as_ref()
            .iter()
            .map(|b| format!("{b:02X}"))
            .collect::<Vec<_>>()
            .join(":");
        let config = rustls::ServerConfig::builder_with_provider(Arc::new(
            rustls::crypto::ring::default_provider(),
        ))
        .with_safe_default_protocol_versions()
        .unwrap()
        .with_no_client_auth()
        .with_single_cert(
            vec![generated.cert.der().clone()],
            rustls::pki_types::PrivateKeyDer::try_from(generated.signing_key.serialize_der())
                .unwrap(),
        )
        .unwrap();
        let config = Arc::new(config);
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let url = format!("wss://127.0.0.1:{}", listener.local_addr().unwrap().port());
        std::thread::spawn(move || {
            for stream in listener.incoming().flatten() {
                let config = Arc::clone(&config);
                std::thread::spawn(move || {
                    let conn = rustls::ServerConnection::new(config).unwrap();
                    let mut tls = rustls::StreamOwned::new(conn, stream);
                    let _ = tls.read(&mut [0u8; 1]);
                });
            }
        });
        (url, fingerprint)
    }

    #[test]
    fn a_new_self_signed_certificate_is_shown_for_confirming() {
        let f = fixture();
        let (url, fingerprint) = tls_server();

        f.connect(&url, "alice", "pw");

        assert_eq!(
            f.state().get_step(),
            STEP_CONFIRM,
            "{}",
            f.state().get_error()
        );
        assert_eq!(f.state().get_fingerprint(), fingerprint.as_str());
        assert!(!f.ui_state.backend().is_remote(), "nothing trusted yet");
    }

    #[test]
    fn a_changed_certificate_is_shown_with_the_one_confirmed_before() {
        let f = fixture();
        let (url, fingerprint) = tls_server();
        let before = "AB:".repeat(31) + "CD";
        f.ui_state.remember_server(RecentServer {
            url: url.clone(),
            user: "alice".into(),
            fingerprint: Some(before.clone()),
        });

        f.connect(&url, "alice", "pw");

        assert_eq!(f.state().get_step(), STEP_CHANGED);
        assert_eq!(f.state().get_known_fingerprint(), before.as_str());
        assert_eq!(f.state().get_fingerprint(), fingerprint.as_str());

        // "Back" returns to the form without trusting anything.
        f.state().invoke_back();
        drain_ui_queue();
        assert_eq!(f.state().get_step(), STEP_FORM);
        assert!(f.connection.pending.lock().unwrap().is_none());
    }

    #[test]
    fn cancelling_drops_an_attempt_still_under_way() {
        let f = fixture();
        f.state().set_url("ws://127.0.0.1:9".into());
        f.state().set_user("alice".into());
        f.state().invoke_connect();
        f.state().invoke_cancel();
        for _ in 0..100 {
            drain_ui_queue();
            std::thread::sleep(Duration::from_millis(10));
        }
        assert!(!f.state().get_busy());
        assert_eq!(f.state().get_error(), "", "the late failure isn't shown");
        assert!(!f.ui_state.backend().is_remote());
        let _ = &f.connection;
    }

    #[test]
    fn a_bare_host_means_an_encrypted_connection() {
        assert_eq!(normalize_url(" lab-server:7400 "), "wss://lab-server:7400");
        assert_eq!(normalize_url("ws://lab-server/"), "ws://lab-server");
        assert_eq!(normalize_url("wss://lab-server"), "wss://lab-server");
        assert_eq!(normalize_url(""), "");
    }
}
