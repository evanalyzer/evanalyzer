//! Batch CLI commands: analyze a project, export results, and peek at a
//! results database. Unlike the GUI, the CLI doesn't run an event loop - each
//! command loads what it needs, does the work, prints a result and returns,
//! so it has no `Frontend` impl (that trait models a long-running UI loop).
mod args;
mod commands;
// mod table; // commented out for now, will fix later

pub use args::CliCommand;

use evanalyzer_cfg::core_types::InternalErrors;

pub fn run(command: CliCommand) -> Result<(), InternalErrors> {
    match command {
        CliCommand::Analyze(args) => commands::analyze::run(args),
        CliCommand::ProjectInfo(args) => commands::project::run(args),
        CliCommand::Validate(args) => commands::project::run_validate(args),
        // CliCommand::Export(args) => commands::export::run(args), // commented out for now, will fix later
        // CliCommand::View(args) => commands::view::run(args), // commented out for now, will fix later
        // CliCommand::Columns(args) => commands::view::run_columns(args), // commented out for now, will fix later
        CliCommand::TrainClassifier(args) => commands::train_classifier::run(args),
    }
}
