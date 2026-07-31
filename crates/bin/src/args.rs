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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_arguments_launches_gui_mode_with_no_project() {
        let args = Args::try_parse_from(["evanalyzer"]).unwrap();
        assert!(args.project.is_none());
        assert!(args.command.is_none());
    }

    #[test]
    fn project_flag_is_parsed_for_gui_mode() {
        let args = Args::try_parse_from(["evanalyzer", "--project", "my.evaproj"]).unwrap();
        assert_eq!(args.project, Some(std::path::PathBuf::from("my.evaproj")));
        assert!(args.command.is_none());
    }

    #[test]
    fn cli_subcommand_is_routed_to_the_evanalyzer_cli_command_tree() {
        let args = Args::try_parse_from([
            "evanalyzer",
            "cli",
            "project-info",
            "--project",
            "p.evaproj",
        ])
        .unwrap();
        match args.command {
            Some(TopCommand::Cli {
                command: evanalyzer_cli::CliCommand::ProjectInfo(a),
            }) => {
                assert_eq!(a.project, std::path::PathBuf::from("p.evaproj"));
            }
            other => panic!(
                "expected Cli(ProjectInfo), got a different command tree: {}",
                other.is_some()
            ),
        }
    }

    #[test]
    fn an_unknown_flag_is_a_parse_error_not_a_panic() {
        assert!(Args::try_parse_from(["evanalyzer", "--not-a-real-flag"]).is_err());
    }
}
