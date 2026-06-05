mod args;

use args::{Mode, parse_args};
use env_logger::Builder;
use evanalyzer_app::{Frontend, ProjectOwner};
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

    // Core init
    evanalyzer_core::init(CoreConfig {
        jvm_heap_size: 1_000_000_000,
    })?;

    // Project init
    let args = parse_args();
    let owner = ProjectOwner::new();

    if let Some(path) = &args.project {
        owner.load_project(path)?;
    }

    // Frontend init
    let frontend: Box<dyn Frontend> = match args.mode {
        Mode::Gui => Box::new(evanalyzer_gui::create()),
        Mode::Cli => Box::new(evanalyzer_cli::create()),
    };

    frontend.start(owner);
    Ok(())
}
