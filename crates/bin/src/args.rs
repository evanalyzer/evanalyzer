// main.rs or app/src/args.rs

use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(name = "EVAnalyzer", version, about = "Image processing tool")]
pub struct Args {
    /// Optional project file to open on startup (GUI mode only)
    #[arg(long)]
    pub project: Option<std::path::PathBuf>,

    /// Run analysis, preview, training and image reads on an
    /// `evanalyzer serve` instance instead of this machine, e.g.
    /// `ws://workstation:7400`. Both machines must see the images and the
    /// project folder under the same paths.
    #[arg(long, global = true, value_name = "URL")]
    pub remote: Option<String>,

    /// Token printed by (or given to) the server. Prefer the environment
    /// variable - command-line arguments are visible to other local users.
    #[arg(
        long,
        global = true,
        value_name = "TOKEN",
        env = "EVANALYZER_REMOTE_TOKEN",
        hide_env_values = true
    )]
    pub remote_token: Option<String>,

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
    /// Run as a compute server for remote GUIs/CLIs (`--remote ws://...`).
    ///
    /// The connection is not encrypted: across machines, reach the server
    /// through an SSH tunnel or VPN. Anyone with the token can make this
    /// machine read images and write results wherever this process may.
    Serve {
        /// Address to listen on. Defaults to this machine only; use e.g.
        /// `0.0.0.0:7400` to accept other machines.
        #[arg(long, default_value = "127.0.0.1:7400")]
        listen: String,

        /// Token clients must present. Generated and printed if not set.
        #[arg(long, env = "EVANALYZER_SERVER_TOKEN", hide_env_values = true)]
        token: Option<String>,
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
    fn remote_flags_apply_to_cli_commands_too() {
        let args = Args::try_parse_from([
            "evanalyzer",
            "cli",
            "--remote",
            "ws://workstation:7400",
            "--remote-token",
            "secret",
            "project-info",
            "--project",
            "p.evaproj",
        ])
        .unwrap();
        assert_eq!(args.remote.as_deref(), Some("ws://workstation:7400"));
        assert_eq!(args.remote_token.as_deref(), Some("secret"));
        assert!(matches!(args.command, Some(TopCommand::Cli { .. })));
    }

    #[test]
    fn serve_defaults_to_localhost_only() {
        let args = Args::try_parse_from(["evanalyzer", "serve"]).unwrap();
        match args.command {
            Some(TopCommand::Serve { listen, .. }) => assert_eq!(listen, "127.0.0.1:7400"),
            _ => panic!("expected the serve command"),
        }
    }

    #[test]
    fn an_unknown_flag_is_a_parse_error_not_a_panic() {
        assert!(Args::try_parse_from(["evanalyzer", "--not-a-real-flag"]).is_err());
    }
}
