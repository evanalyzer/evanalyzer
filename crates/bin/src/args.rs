// main.rs or app/src/args.rs

use clap::{Parser, ValueEnum};

#[derive(Parser)]
#[command(name = "EVAnalyzer", version, about = "Image processing tool")]
pub struct Args {
    /// Run in GUI mode (default) or CLI mode
    #[arg(long, default_value = "gui")]
    pub mode: Mode,

    /// Optional project file to open on startup
    #[arg(long)]
    pub project: Option<std::path::PathBuf>,
}

#[derive(ValueEnum, Clone, Default, PartialEq)]
pub enum Mode {
    #[default]
    Gui,
    Cli,
}

pub fn parse_args() -> Args {
    Args::parse()
}
