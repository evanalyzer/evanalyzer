mod args;

use args::{TopCommand, parse_args};
use env_logger::Builder;
use evanalyzer_app::backends::Backend;
use evanalyzer_app::backends::local::LocalBackend;
use evanalyzer_app::global::Frontend;
use evanalyzer_app::project::ProjectOwner;
use evanalyzer_cfg::core_types::InternalErrors;
use evanalyzer_cli::CliCommand;
use evanalyzer_server::serve;
use log::{LevelFilter, info, warn};
use std::{path::PathBuf, sync::Arc};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = parse_args();
    init_logger(&args.log_level);

    // The one place that decides where compute runs - front ends only ever
    // see the `Backend` trait.
    let backend = generate_backend(args.remote, args.remote_token, args.user, args.password)?;
    log::info!("Compute backend: {}", backend.description());

    let ret = match args.command {
        Some(cmd) => match cmd {
            TopCommand::Cli { command } => start_cli(&backend, command),
            TopCommand::Worker {
                listen,
                token,
                roots,
                home,
            } => start_worker(listen, token, roots, home),
            TopCommand::Server {
                listen,
                session_store,
            } => start_server(listen, session_store, args.log_level.clone()),
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
                    RemoteBackend::connect_with_login(url, &user, &password)
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
                    eprintln!("Error: {e}");
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
    if let Some(path) = &project {
        owner.load_project(path)?;
    }
    let frontend: Box<dyn Frontend> = Box::new(evanalyzer_gui::create());
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
    let worker = evanalyzer_app::backends::remote::Worker::bind(&listen, token)?;
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

/// Start evanalyzer server; its workers log with the same `log_level`.
fn start_server(
    listen: String,
    session_store: Option<PathBuf>,
    log_level: String,
) -> Result<(), Box<dyn std::error::Error>> {
    let session_store = session_store.unwrap_or_else(evanalyzer_server::default_store_path);
    serve(listen, session_store, log_level)?;
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
        .filter_module("winit", LevelFilter::Off)
        .filter_module("glow", LevelFilter::Off)
        .filter_module("zbus", LevelFilter::Off)
        .filter_module("naga", LevelFilter::Off)
        .filter_module("tungstenite", LevelFilter::Off)
        .filter_module("wgpu_core", LevelFilter::Off)
        .filter_module("wgpu_hal", LevelFilter::Off)
        .filter_module("tracing::span", LevelFilter::Off);

    builder.parse_filters(log_level);
    builder.init();
}
