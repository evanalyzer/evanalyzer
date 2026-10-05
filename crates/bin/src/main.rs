mod args;

use args::{TopCommand, parse_args};
use env_logger::Builder;
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
    let args = parse_args();
    // The server's log level may come from its config file, so the logger
    // waits for that.
    let server_config = match &args.command {
        Some(TopCommand::Server {
            config,
            listen,
            session_store,
        }) => Some(
            server_config(
                config.as_deref(),
                listen.clone(),
                session_store.clone(),
                args.log_level.clone(),
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
        None => args.log_level.as_deref().unwrap_or("debug"),
    });

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
    session_store: Option<PathBuf>,
    log_level: Option<String>,
) -> std::io::Result<ServerConfig> {
    let config = match file {
        Some(file) => ServerConfig::load(file)?,
        None => ServerConfig::default(),
    };
    Ok(config.with_overrides(listen, session_store, log_level))
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
