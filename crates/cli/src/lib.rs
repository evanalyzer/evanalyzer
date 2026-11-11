//! # evanalyzer_cli
//!
//! Batch CLI commands: analyze a project, export results, and peek at a
//! results database. Unlike the GUI, the CLI doesn't run an event loop - each
//! command loads what it needs, does the work, prints a result and returns,
//! so it has no `Frontend` impl (that trait models a long-running UI loop).
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-06-05
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.
mod args;
mod commands;
mod overrides;
mod table;

pub use args::{CliCommand, JobsSubCommand, ProjectSubCommand, ResultsSubCommand};

use evanalyzer_app::backends::Backend;
use evanalyzer_cfg::core_types::InternalErrors;

/// Runs `command`. Analysis and training run on `backend` (local or a
/// server); the other commands only read local files.
pub fn run(command: CliCommand, backend: &dyn Backend) -> Result<(), InternalErrors> {
    match command {
        CliCommand::Analyze(analyze_args) => commands::analyze::run(analyze_args, backend),
        CliCommand::Project(project_sub_command) => match project_sub_command {
            args::ProjectSubCommand::New(project_new_args) => commands::project::new_project(
                project_new_args,
                backend,
                &evanalyzer_app::backends::local::LocalBackend::default(),
            ),
            args::ProjectSubCommand::Info(project_info_args) => {
                commands::project::run(project_info_args, backend)
            }
            args::ProjectSubCommand::Validate(validate_args) => {
                commands::project::run_validate(validate_args, backend)
            }
            args::ProjectSubCommand::Cite(cite_args) => {
                commands::project::run_cite(cite_args, backend)
            }
        },
        CliCommand::Results(results_sub_command) => match results_sub_command {
            args::ResultsSubCommand::Export(export_args) => {
                commands::export::run(export_args, backend)
            }
            args::ResultsSubCommand::View(view_args) => commands::view::run(view_args, backend),
            args::ResultsSubCommand::Columns(columns_args) => {
                commands::view::run_columns(columns_args, backend)
            }
        },
        CliCommand::Jobs(jobs_sub_command) => match jobs_sub_command {
            args::JobsSubCommand::List => commands::jobs::run_list(backend),
            args::JobsSubCommand::Attach(attach_args) => {
                commands::jobs::run_attach(attach_args, backend)
            }
        },
        CliCommand::TrainClassifier(train_classifier_args) => {
            commands::train_classifier::run(train_classifier_args, backend)
        }
    }
}
