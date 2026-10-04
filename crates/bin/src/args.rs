// main.rs or app/src/args.rs
use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(name = "EVAnalyzer", version, about = "Image processing tool")]
pub struct Args {
    /// Optional project file to open on startup (GUI mode only)
    #[arg(long)]
    pub project: Option<std::path::PathBuf>,

    /// What to log, in env_logger filter syntax: a level (`error`, `warn`,
    /// `info`, `debug`, `trace`, `off`) or per module, e.g.
    /// `info,evanalyzer_core=debug`. Default `debug`, or for `server` the
    /// config file's `log_level`.
    #[arg(long, global = true, value_name = "FILTER")]
    pub log_level: Option<String>,

    /// EVAnalyzer server to work on instead of this machine, e.g.
    /// `ws://workstation:7400`. Projects, images and results are then read
    /// and written there; all paths refer to that machine.
    #[arg(long, global = true, value_name = "URL", help_heading = "Remote")]
    pub remote: Option<String>,

    /// User to log in as on the `--remote` server.
    #[arg(
        long,
        global = true,
        value_name = "USER",
        requires = "remote",
        help_heading = "Remote"
    )]
    pub user: Option<String>,

    /// Password for `--user`. Asked for (hidden) if not given. Note that
    /// command-line arguments are visible to other local users.
    #[arg(
        long,
        global = true,
        value_name = "PASSWORD",
        requires = "user",
        help_heading = "Remote"
    )]
    pub password: Option<String>,

    /// Connect to an `evanalyzer worker` directly, without server login,
    /// using the token it printed (instead of `--user`).
    #[arg(
        long,
        global = true,
        value_name = "TOKEN",
        requires = "remote",
        conflicts_with = "user",
        help_heading = "Remote"
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
    /// Start an EVAnalyzer server which manages more evanalyzer worker sessions
    ///
    /// Settings come from the built-in defaults, then the `--config` file,
    /// then these arguments. `docs/server.toml` documents every setting.
    Server {
        /// Configuration file (TOML): users, allowed folders, and the
        /// settings below. See `docs/server.md`.
        #[arg(long, value_name = "FILE")]
        config: Option<std::path::PathBuf>,

        /// Address and port to accept clients on. Default `127.0.0.1:7400`.
        #[arg(long, value_name = "ADDRESS")]
        listen: Option<String>,

        /// File the running workers are recorded in, so a restarted server
        /// finds them again. Default: `/run/evanalyzer/sessions.json` for a
        /// system service, otherwise in the temp folder.
        #[arg(long, value_name = "FILE")]
        session_store: Option<std::path::PathBuf>,
    },
    /// Print an Argon2id hash of a password, for `password = "..."` in the
    /// server's config or users file. Asks for the password twice (hidden),
    /// or reads one line from stdin when it is not a terminal:
    /// `echo 'secret' | evanalyzer hash-password`.
    HashPassword,
    /// Run one compute instance. Started by `evanalyzer server` for each
    /// logged-in user, or by hand for a direct `--remote-token` connection.
    Worker {
        /// Address to listen on.
        #[arg(long, default_value = "127.0.0.1:7400")]
        listen: String,

        /// Token clients must present. Generated and printed if not set.
        #[arg(long)]
        token: Option<String>,

        /// Folder clients may browse and use (repeatable).
        #[arg(long = "root", value_name = "FOLDER")]
        roots: Vec<std::path::PathBuf>,

        /// Home folder of the user this worker serves: its user folder
        /// (templates) lives below it. Default: this account's.
        #[arg(long, value_name = "FOLDER")]
        home: Option<std::path::PathBuf>,
    },
}

pub fn parse_args() -> Args {
    Args::parse()
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::CommandFactory;

    #[test]
    fn help_lists_all_options_without_environment_variables() {
        let help = Args::command().render_long_help().to_string();
        for shown in [
            "Remote:",
            "--remote",
            "--user",
            "--password",
            "--remote-token",
            "\n  server ",
            "\n  worker ",
            "\n  hash-password ",
        ] {
            assert!(help.contains(shown), "{shown} missing in:\n{help}");
        }
        assert!(!help.contains("[env:"), "no environment variables:\n{help}");
    }

    #[test]
    fn server_settings_not_given_are_left_to_the_config_file() {
        let args =
            Args::try_parse_from(["evanalyzer", "server", "--config", "/etc/eva.toml"]).unwrap();
        assert!(
            args.log_level.is_none(),
            "not defaulted - the file may set it"
        );
        let Some(TopCommand::Server {
            config,
            listen,
            session_store,
        }) = args.command
        else {
            panic!("expected the server command");
        };
        assert_eq!(config, Some("/etc/eva.toml".into()));
        assert!(listen.is_none() && session_store.is_none());
    }

    #[test]
    fn user_without_remote_and_user_with_token_are_rejected() {
        assert!(Args::try_parse_from(["evanalyzer", "--user", "alice"]).is_err());
        assert!(
            Args::try_parse_from(["evanalyzer", "--remote", "ws://h", "--password", "x"]).is_err()
        );
        assert!(
            Args::try_parse_from([
                "evanalyzer",
                "--remote",
                "ws://h",
                "--user",
                "a",
                "--remote-token",
                "t"
            ])
            .is_err()
        );
    }

    #[test]
    fn user_flag_selects_server_login() {
        let args = Args::try_parse_from([
            "evanalyzer",
            "cli",
            "--remote",
            "ws://server:7400",
            "--user",
            "alice",
            "--password",
            "secret",
            "project-info",
            "--project",
            "p.evaproj",
        ])
        .unwrap();
        assert_eq!(args.remote.as_deref(), Some("ws://server:7400"));
        assert_eq!(args.user.as_deref(), Some("alice"));
        assert_eq!(args.password.as_deref(), Some("secret"));
    }

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
    fn worker_defaults_to_localhost_only() {
        let args = Args::try_parse_from(["evanalyzer", "worker"]).unwrap();
        match args.command {
            Some(TopCommand::Worker { listen, roots, .. }) => {
                assert_eq!(listen, "127.0.0.1:7400");
                assert!(roots.is_empty());
            }
            _ => panic!("expected the worker command"),
        }
    }

    #[test]
    fn worker_accepts_several_roots() {
        let args = Args::try_parse_from([
            "evanalyzer",
            "worker",
            "--root",
            "/data",
            "--root",
            "/scratch",
        ])
        .unwrap();
        match args.command {
            Some(TopCommand::Worker { roots, .. }) => {
                assert_eq!(
                    roots,
                    [std::path::PathBuf::from("/data"), "/scratch".into()]
                );
            }
            _ => panic!("expected the worker command"),
        }
    }

    #[test]
    fn an_unknown_flag_is_a_parse_error_not_a_panic() {
        assert!(Args::try_parse_from(["evanalyzer", "--not-a-real-flag"]).is_err());
    }
}
