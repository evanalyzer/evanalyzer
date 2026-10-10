//! Batch CLI commands: analyze a project, export results, and peek at a
//! results database. Unlike the GUI, the CLI doesn't run an event loop - each
//! command loads what it needs, does the work, prints a result and returns,
//! so it has no `Frontend` impl (that trait models a long-running UI loop).
mod args;
mod commands;
mod table;

pub use args::CliCommand;

use evanalyzer_app::backends::Backend;
use evanalyzer_cfg::core_types::InternalErrors;

/// Runs `command`. Analysis and training run on `backend` (local or a
/// server); the other commands only read local files.
pub fn run(command: CliCommand, backend: &dyn Backend) -> Result<(), InternalErrors> {
    match command {
        CliCommand::Analyze(args) => commands::analyze::run(args, backend),
        CliCommand::ProjectInfo(args) => commands::project::run(args, backend),
        CliCommand::Validate(args) => commands::project::run_validate(args, backend),
        CliCommand::Export(args) => commands::export::run(args, backend),
        CliCommand::View(args) => commands::view::run(args, backend),
        CliCommand::Columns(args) => commands::view::run_columns(args, backend),
        CliCommand::TrainClassifier(args) => commands::train_classifier::run(args, backend),
        CliCommand::Jobs => commands::jobs::run_list(backend),
        CliCommand::Attach(args) => commands::jobs::run_attach(args, backend),
    }
}
