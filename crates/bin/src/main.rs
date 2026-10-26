//! # evanalyzer_bin
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-06-05
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.

mod args;

use args::{TopCommand, parse_args};
use env_logger::Builder;
use evanalyzer_app::ai::{
    AiDeviceOptions, AiDeviceSelection, DEFAULT_GPU_SLOTS, configure_ai_devices,
};
use evanalyzer_app::backends::Backend;
use evanalyzer_app::backends::local::LocalBackend;
use evanalyzer_app::backends::remote::TlsTrust;
use evanalyzer_app::global::Frontend;
use evanalyzer_app::project::ProjectOwner;
use evanalyzer_cfg::core_types::InternalErrors;
use evanalyzer_cli::CliCommand;
use evanalyzer_server::{ServerConfig, serve};
use log::{LevelFilter, info, warn};
use std::{path::PathBuf, sync::Arc};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Startup times (log level debug) are measured from here.
    evanalyzer_gui::mark_process_start();
    let args = parse_args();
    // The server's log level may come from its config file, so the logger
    // waits for that.
    let server_config = match &args.command {
        Some(TopCommand::Server {
            config,
            listen,
            data_dir,
        }) => Some(
            server_config(
                config.as_deref(),
                listen.clone(),
                data_dir.clone(),
                args.log_level.clone(),
                &args.ai_devices,
                args.gpu_slots,
            )
            .unwrap_or_else(|e| {
                // `main`'s error return would print it Debug-formatted, which
                // garbles the multi-line TOML error.
                eprintln!("Error: {e}");
                std::process::exit(2)
            }),
        ),
        _ => None,
    };
    init_logger(match &server_config {
        Some(config) => &config.log_level,
        None => args.log_level.as_deref().unwrap_or("info"),
    });
    // A server only forwards these to its workers (see `server_config`);
    // everywhere else the analysis may run in this process.
    if server_config.is_none() {
        configure_ai_devices(AiDeviceOptions {
            devices: args.ai_devices.clone().unwrap_or_default(),
            gpu_slots: args.gpu_slots.unwrap_or(DEFAULT_GPU_SLOTS),
        });
    }

    // The one place that decides where compute runs - front ends only ever
    // see the `Backend` trait.
    let backend = generate_backend(
        args.remote,
        args.remote_token,
        args.user,
        args.password,
        match (args.remote_fingerprint, args.no_tls_verification) {
            (Some(fingerprint), _) => TlsTrust::Fingerprint(fingerprint),
            (None, true) => TlsTrust::NoVerification,
            (None, false) => TlsTrust::PublicAuthorities,
        },
    )?;
    log::info!("Compute backend: {}", backend.description());

    let ret = match args.command {
        Some(cmd) => match cmd {
            TopCommand::Cli { command } => start_cli(&backend, command),
            TopCommand::Worker {
                listen,
                token,
                roots,
                home,
                idle_timeout,
            } => start_worker(listen, token, roots, home, idle_timeout),
            TopCommand::HashPassword => print_password_hash(),
            TopCommand::Server { .. } => {
                start_server(server_config.expect("loaded above for the server command"))
            }
        },
        None => start_gui(backend, args.project),
    };

    ret
}

/// Generate a backend
fn generate_backend(
    remote: Option<String>,
    token: Option<String>,
    user: Option<String>,
    password: Option<String>,
    trust: TlsTrust,
) -> Result<Arc<dyn Backend>, Box<dyn std::error::Error>> {
    use evanalyzer_app::backends::remote::RemoteBackend;
    match &remote {
        None => Ok(Arc::new(LocalBackend::default())),
        Some(url) => {
            let connected = match (user, token) {
                // `evanalyzer server`: log in, the server attaches us to our worker.
                (Some(user), _) => {
                    let password = match password {
                        Some(password) => password,
                        None => rpassword::prompt_password(format!("Password for {user}: "))?,
                    };
                    RemoteBackend::connect_with_login(url, &user, &password, &trust)
                }
                // `evanalyzer worker` directly.
                (None, Some(token)) => RemoteBackend::connect(url, &token),
                (None, None) => {
                    return Err("--remote needs --user (evanalyzer server) \
                                or --remote-token (evanalyzer worker)"
                        .into());
                }
            };
            match connected {
                Ok(remote) => Ok(Arc::new(remote)),
                Err(e) => {
                    log::error!("Cannot connect to {url}: {e}");
                    std::process::exit(1);
                }
            }
        }
    }
}

fn start_gui(
    backend: Arc<dyn Backend>,
    project: Option<std::path::PathBuf>,
) -> Result<(), Box<dyn std::error::Error>> {
    // GUI mode (default)
    let owner = ProjectOwner::with_backend(backend);
    let frontend: Box<dyn Frontend> = Box::new(evanalyzer_gui::create().with_project(project));
    frontend.start(owner);
    Ok(())
}

/// Start CLI
fn start_cli(
    backend: &Arc<dyn Backend>,
    command: CliCommand,
) -> Result<(), Box<dyn std::error::Error>> {
    if let Err(e) = evanalyzer_cli::run(command, backend.as_ref()) {
        match e {
            InternalErrors::Cancelled => {
                eprintln!("Cancelled.");
                std::process::exit(130);
            }
            other => {
                eprintln!("Error: {other}");
                std::process::exit(1);
            }
        }
    }
    return Ok(());
}

/// `evanalyzer worker`: serve a local backend to remote clients over WebSocket.
fn start_worker(
    listen: String,
    token: Option<String>,
    roots: Vec<PathBuf>,
    home: Option<PathBuf>,
    idle_timeout_minutes: Option<u64>,
) -> Result<(), Box<dyn std::error::Error>> {
    let token = match token {
        Some(token) => token,
        None => {
            let token = evanalyzer_app::backends::remote::generate_token()?;
            info!("No token given - generated one for this session:\n\n  {token}\n");
            eprintln!("Clients connect with --remote-token set to this value.");
            token
        }
    };
    let mut worker = evanalyzer_app::backends::remote::Worker::bind(&listen, token)?;
    if let Some(minutes) = idle_timeout_minutes {
        info!("Stopping after {minutes} min without client and analysis");
        worker = worker.with_idle_timeout(std::time::Duration::from_secs(minutes * 60));
    }
    info!(
        "EVAnalyzer worker listening on ws://{}",
        worker.local_addr()?
    );
    let backend = if roots.is_empty() {
        warn!("Warning: no --root given - clients can reach every file this process can.");
        LocalBackend::default()
    } else {
        for root in &roots {
            info!("Serving folder {}", root.display());
        }
        LocalBackend::restricted_to(&roots)?
    };
    let backend = match home {
        Some(home) => {
            info!("Serving user at home in {}", home.display());
            backend.with_home(&home)?
        }
        None => backend,
    };
    worker.run(Arc::new(backend));
    return Ok(());
}

/// The server's settings: defaults, then the `--config` file, then the
/// command-line arguments.
fn server_config(
    file: Option<&std::path::Path>,
    listen: Option<String>,
    data_dir: Option<PathBuf>,
    log_level: Option<String>,
    ai_devices: &Option<AiDeviceSelection>,
    gpu_slots: Option<usize>,
) -> std::io::Result<ServerConfig> {
    let config = match file {
        Some(file) => ServerConfig::load(file)?,
        None => ServerConfig::default(),
    };
    let config = config
        .with_overrides(listen, data_dir, log_level)
        .with_worker_ai_overrides(ai_devices.as_ref().map(ToString::to_string), gpu_slots);
    // Checked here, so a typo in the file fails at startup instead of in
    // every worker.
    let invalid = |e: String| std::io::Error::new(std::io::ErrorKind::InvalidInput, e);
    if let Some(devices) = &config.workers.ai_devices {
        devices
            .parse::<AiDeviceSelection>()
            .map_err(|e| invalid(format!("workers.ai_devices: {e}")))?;
    }
    if config.workers.gpu_slots == Some(0) {
        return Err(invalid("workers.gpu_slots must be at least 1".into()));
    }
    Ok(config)
}

/// `evanalyzer hash-password`: the hash goes to stdout alone (prompts go to
/// the terminal), so it works in `$(...)` and pipes.
fn print_password_hash() -> Result<(), Box<dyn std::error::Error>> {
    use std::io::{BufRead, IsTerminal};
    let password = if std::io::stdin().is_terminal() {
        let password = rpassword::prompt_password("Password: ")?;
        if rpassword::prompt_password("Repeat password: ")? != password {
            return Err("The passwords differ".into());
        }
        password
    } else {
        let mut line = String::new();
        std::io::stdin().lock().read_line(&mut line)?;
        line.trim_end_matches(['\r', '\n']).to_string()
    };
    if password.is_empty() {
        return Err("The password is empty".into());
    }
    println!("{}", evanalyzer_server::hash_password(&password)?);
    Ok(())
}

/// Start evanalyzer server; its workers log with the same `log_level`.
fn start_server(config: ServerConfig) -> Result<(), Box<dyn std::error::Error>> {
    serve(config)?;
    Ok(())
}

/// Init the logger with `log_level` (env_logger filter syntax, from
/// `--log-level`) on top of the noisy-crate defaults.
fn init_logger(log_level: &str) {
    evanalyzer_app::global::crash_log::install_panic_hook();
    let mut builder = Builder::new();
    builder.filter_level(LevelFilter::Debug);
    builder
        .filter_module("slint", LevelFilter::Off)
        .filter_module("rustls", LevelFilter::Off)
        .filter_module("winit", LevelFilter::Off)
        .filter_module("sctk", LevelFilter::Off)
        .filter_module("glow", LevelFilter::Off)
        .filter_module("zbus", LevelFilter::Off)
        .filter_module("naga", LevelFilter::Off)
        .filter_module("tungstenite", LevelFilter::Off)
        .filter_module("wgpu_core", LevelFilter::Off)
        .filter_module("wgpu_hal", LevelFilter::Off)
        .filter_module("arboard", LevelFilter::Off)
        .filter_module("sctk_adwaita", LevelFilter::Off)
        .filter_module("tracing::span", LevelFilter::Off);

    builder.parse_filters(log_level);
    builder.init();
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A file of its own in the temp folder, removed again on drop.
    struct TempFile(PathBuf);

    impl TempFile {
        fn with(name: &str, contents: &str) -> Self {
            let path = std::env::temp_dir()
                .join(format!("evanalyzer-bin-test-{}-{name}", std::process::id()));
            std::fs::write(&path, contents).unwrap();
            Self(path)
        }
    }

    impl Drop for TempFile {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.0);
        }
    }

    #[test]
    fn without_remote_the_analysis_runs_on_this_machine() {
        let backend =
            generate_backend(None, None, None, None, TlsTrust::PublicAuthorities).unwrap();

        assert!(!backend.is_remote());
    }

    #[test]
    fn remote_without_user_or_token_is_an_error_naming_both() {
        let Err(e) = generate_backend(
            Some("ws://127.0.0.1:1".into()),
            None,
            None,
            None,
            TlsTrust::PublicAuthorities,
        ) else {
            panic!("expected an error");
        };

        let message = e.to_string();
        assert!(message.contains("--user") && message.contains("--remote-token"));
    }

    #[test]
    fn remote_with_a_token_connects_to_the_worker_directly() {
        let worker =
            evanalyzer_app::backends::remote::Worker::bind("127.0.0.1:0", "token".into()).unwrap();
        let url = format!("ws://{}", worker.local_addr().unwrap());
        std::thread::spawn(move || worker.run(Arc::new(LocalBackend::default())));

        let backend = generate_backend(
            Some(url.clone()),
            Some("token".into()),
            None,
            None,
            TlsTrust::PublicAuthorities,
        )
        .unwrap();

        assert!(backend.is_remote());
        assert!(
            backend.description().contains(&url),
            "{}",
            backend.description()
        );
    }

    #[test]
    fn a_cli_command_that_succeeds_returns_ok() {
        let backend: Arc<dyn Backend> = Arc::new(LocalBackend::default());

        assert!(start_cli(&backend, CliCommand::Jobs).is_ok());
    }

    #[test]
    fn a_worker_that_cannot_listen_fails_with_or_without_a_token() {
        for token in [Some("token".to_string()), None] {
            let result = start_worker("not an address".into(), token, Vec::new(), None, Some(1));

            let message = result.unwrap_err().to_string();
            assert!(message.contains("Could not listen on"), "{message}");
        }
    }

    #[test]
    fn without_a_file_the_server_uses_defaults_and_the_command_line() {
        let config = server_config(
            None,
            Some("0.0.0.0:9000".into()),
            Some("/srv/eva".into()),
            Some("debug".into()),
            &Some(AiDeviceSelection::Cpu),
            Some(2),
        )
        .unwrap();

        assert_eq!(config.listen, "0.0.0.0:9000");
        assert_eq!(config.data_dir, Some("/srv/eva".into()));
        assert_eq!(config.log_level, "debug");
        assert_eq!(config.workers.ai_devices.as_deref(), Some("cpu"));
        assert_eq!(config.workers.gpu_slots, Some(2));
    }

    #[test]
    fn no_settings_at_all_give_the_default_server_config() {
        let config = server_config(None, None, None, None, &None, None).unwrap();

        assert_eq!(config, ServerConfig::default());
    }

    #[test]
    fn the_command_line_wins_over_the_config_file() {
        let file = TempFile::with(
            "override.toml",
            "listen = \"127.0.0.1:7500\"\nlog_level = \"warn\"\n",
        );

        let from_file = server_config(Some(&file.0), None, None, None, &None, None).unwrap();
        let overridden = server_config(
            Some(&file.0),
            Some("127.0.0.1:7600".into()),
            None,
            None,
            &None,
            None,
        )
        .unwrap();

        assert_eq!(
            (from_file.listen.as_str(), from_file.log_level.as_str()),
            ("127.0.0.1:7500", "warn")
        );
        assert_eq!(overridden.listen, "127.0.0.1:7600");
        assert_eq!(overridden.log_level, "warn", "kept from the file");
    }

    #[test]
    fn a_missing_config_file_is_an_error() {
        let missing = std::env::temp_dir().join("evanalyzer-bin-test-does-not-exist.toml");

        assert!(server_config(Some(&missing), None, None, None, &None, None).is_err());
    }

    #[test]
    fn a_misspelled_ai_device_list_in_the_file_fails_at_startup() {
        let file = TempFile::with("devices.toml", "[workers]\nai_devices = \"gpu\"\n");

        let e = server_config(Some(&file.0), None, None, None, &None, None).unwrap_err();

        assert_eq!(e.kind(), std::io::ErrorKind::InvalidInput);
        assert!(e.to_string().starts_with("workers.ai_devices:"), "{e}");
    }

    #[test]
    fn zero_gpu_slots_are_refused() {
        let file = TempFile::with("slots.toml", "[workers]\ngpu_slots = 0\n");

        let from_file = server_config(Some(&file.0), None, None, None, &None, None);
        let from_command_line = server_config(None, None, None, None, &None, Some(0));

        for result in [from_file, from_command_line] {
            let e = result.unwrap_err();
            assert_eq!(e.kind(), std::io::ErrorKind::InvalidInput);
            assert!(
                e.to_string().contains("gpu_slots must be at least 1"),
                "{e}"
            );
        }
    }
}
