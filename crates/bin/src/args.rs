// main.rs or app/src/args.rs

use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(name = "EVAnalyzer", version, about = "Image processing tool")]
pub struct Args {
    /// Optional project file to open on startup (GUI mode only)
    #[arg(long)]
    pub project: Option<std::path::PathBuf>,

    #[command(subcommand)]
    pub command: Option<TopCommand>,
}

#[derive(Subcommand)]
pub enum TopCommand {
    /// Run a one-shot CLI command (analyze / export / view / ...) instead of launching the GUI
    Cli {
        #[command(subcommand)]
        command: evanalyzer_cli::CliCommand,
    },
}

pub fn parse_args() -> Args {
    Args::parse()
}
