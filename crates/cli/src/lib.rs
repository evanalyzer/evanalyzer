//! Batch CLI commands: analyze a project, export results, and peek at a
//! results database. Unlike the GUI, the CLI doesn't run an event loop - each
//! command loads what it needs, does the work, prints a result and returns,
//! so it has no `Frontend` impl (that trait models a long-running UI loop).
mod args;
mod commands;
mod table;

pub use args::CliCommand;

use evanalyzer_cfg::core_types::InternalErrors;

pub fn run(command: CliCommand) -> Result<(), InternalErrors> {
    match command {
        CliCommand::Analyze(args) => commands::analyze::run(args),
        CliCommand::ProjectInfo(args) => commands::project::run(args),
        CliCommand::Export(args) => commands::export::run(args),
        CliCommand::View(args) => commands::view::run(args),
        CliCommand::Columns(args) => commands::view::run_columns(args),
    }
}
