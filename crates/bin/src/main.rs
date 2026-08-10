mod args;

use args::{TopCommand, parse_args};
use env_logger::Builder;
use evanalyzer_app::{Frontend, ProjectOwner};
use evanalyzer_cfg::core_types::InternalErrors;
use log::LevelFilter;

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
        .filter_module("tracing::span", LevelFilter::Off)
        .filter_module("jni::wrapper::java_vm::vm", LevelFilter::Off);

    if let Ok(rust_log) = std::env::var("RUST_LOG") {
        builder.parse_filters(&rust_log);
    }
    builder.init();

    // No eager JVM init here: it's started lazily on first actual image read
    // (see `ensure_java_wrapper` in evanalyzer_core), so `--help` and other
    // JVM-free invocations aren't stuck waiting on it, and the GUI's window
    // shows up without blocking on it either - see `evanalyzer_gui::run`,
    // which warms it up on a background thread right after creating the
    // window instead.

    let args = parse_args();

    // CLI mode: run the requested batch command and exit, no GUI event loop involved.
    if let Some(TopCommand::Cli { command }) = args.command {
        if let Err(e) = evanalyzer_cli::run(command) {
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
    let owner = ProjectOwner::new();
    if let Some(path) = &args.project {
        owner.load_project(path)?;
    }

    let frontend: Box<dyn Frontend> = Box::new(evanalyzer_gui::create());
    frontend.start(owner);
    Ok(())
}
