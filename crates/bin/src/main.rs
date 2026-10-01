mod args;

use args::{TopCommand, parse_args};
use env_logger::Builder;
use evanalyzer_app::backend::{Backend, LocalBackend};
use evanalyzer_app::{Frontend, ProjectOwner};
use evanalyzer_cfg::core_types::InternalErrors;
use log::LevelFilter;
use std::sync::Arc;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // As early as possible, before any other setup: a packaged build has no
    // attached console, so without this a panic anywhere in the process
    // just makes the window disappear with nothing to diagnose it from.
    evanalyzer_app::crash_log::install_panic_hook();

    let mut builder = Builder::new();
    builder.filter_level(LevelFilter::Debug);
    builder
        .filter_module("slint", LevelFilter::Off)
        .filter_module("winit", LevelFilter::Off)
        .filter_module("glow", LevelFilter::Off)
        .filter_module("zbus", LevelFilter::Off)
        .filter_module("tracing::span", LevelFilter::Off);

    if let Ok(rust_log) = std::env::var("RUST_LOG") {
        builder.parse_filters(&rust_log);
    }
    builder.init();

    let args = parse_args();

    // Server mode: execute remote clients' requests on this machine.
    if let Some(TopCommand::Serve { listen, token }) = args.command {
        let token = match token {
            Some(token) => token,
            None => {
                let token = evanalyzer_app::net::generate_token()?;
                eprintln!("No token given - generated one for this session:\n\n  {token}\n");
                eprintln!("Clients connect with EVANALYZER_REMOTE_TOKEN set to this value.");
                token
            }
        };
        let server = evanalyzer_app::net::Server::bind(&listen, token)?;
        eprintln!(
            "EVAnalyzer server listening on ws://{}",
            server.local_addr()?
        );
        server.run(Arc::new(LocalBackend));
        return Ok(());
    }

    // The one place that decides where compute runs - front ends only ever
    // see the `Backend` trait.
    let backend: Arc<dyn Backend> = match &args.remote {
        None => Arc::new(LocalBackend),
        Some(url) => {
            let token = args
                .remote_token
                .as_deref()
                .ok_or("--remote needs a token: set EVANALYZER_REMOTE_TOKEN (or --remote-token)")?;
            match evanalyzer_app::net::RemoteBackend::connect(url, token) {
                Ok(remote) => Arc::new(remote),
                Err(e) => {
                    eprintln!("Error: {e}");
                    std::process::exit(1);
                }
            }
        }
    };
    log::info!("Compute backend: {}", backend.description());

    // CLI mode: run the requested batch command and exit, no GUI event loop involved.
    if let Some(TopCommand::Cli { command }) = args.command {
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

    // GUI mode (default)
    let owner = ProjectOwner::with_backend(backend);
    if let Some(path) = &args.project {
        owner.load_project(path)?;
    }

    let frontend: Box<dyn Frontend> = Box::new(evanalyzer_gui::create());
    frontend.start(owner);
    Ok(())
}
