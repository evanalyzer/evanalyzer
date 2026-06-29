mod args;

use args::{TopCommand, parse_args};
use env_logger::Builder;
use evanalyzer_app::{Frontend, ProjectOwner};
use evanalyzer_cfg::core_types::InternalErrors;
use evanalyzer_core::CoreConfig;
use log::LevelFilter;

fn main() -> Result<(), Box<dyn std::error::Error>> {
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

    // Core init - JVM heap is sized from available system RAM, see CoreConfig::default()
    evanalyzer_core::init(CoreConfig::default())?;

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
