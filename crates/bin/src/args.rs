//! # args
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-06-05
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.

// main.rs or app/src/args.rs
use clap::{Parser, Subcommand};
use evanalyzer_app::ai::{AiDeviceSelection, parse_gpu_slots};

#[derive(Parser)]
#[command(name = "EVAnalyzer", version, about = "Image processing tool")]
pub struct Args {
    /// Optional project file to open on startup (GUI mode only)
    #[arg(long)]
    pub project: Option<std::path::PathBuf>,

    /// What to log, in env_logger filter syntax: a level (`error`, `warn`,
    /// `info`, `debug`, `trace`, `off`) or per module, e.g.
    /// `info,evanalyzer_core=debug`. Default `info`, or for `server` the
    /// config file's `log_level`.
    #[arg(long, global = true, value_name = "FILTER")]
    pub log_level: Option<String>,

    /// EVAnalyzer server to work on instead of this machine, e.g.
    /// `wss://workstation:7400` (`ws://` if the server runs without TLS).
    /// Projects, images and results are then read and written there; all
    /// paths refer to that machine.
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

    /// SHA-256 fingerprint of the server's TLS certificate, as `evanalyzer
    /// server` logs it at startup. Needed for the server's own self-signed
    /// certificate (the default); not for one signed by a public authority.
    #[arg(
        long,
        global = true,
        value_name = "SHA256",
        requires = "user",
        help_heading = "Remote"
    )]
    pub remote_fingerprint: Option<String>,

    /// Accept any TLS certificate from the server, without checking it.
    /// The connection is still encrypted, but anyone in between could pose
    /// as the server and read your password - only for tests and networks
    /// you trust. Prefer `--remote-fingerprint`.
    #[arg(
        long,
        global = true,
        requires = "user",
        conflicts_with = "remote_fingerprint",
        help_heading = "Remote"
    )]
    pub no_tls_verification: bool,

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

    /// Where AI models (Cellpose, StarDist, U-Net, YOLOv5) run: `auto`
    /// (every GPU, else the CPU), `cpu`, or GPU numbers like `0,2`. Applies
    /// where the analysis runs - this machine, or the workers of a `server`
    /// started with it; ignored with `--remote`. Default `auto`.
    #[arg(long, global = true, value_name = "DEVICES", help_heading = "AI")]
    pub ai_devices: Option<AiDeviceSelection>,

    /// Parallel AI inferences per GPU. Raise it for GPUs with memory to
    /// spare, so one tile's transfers overlap another's computation; a GPU
    /// that runs out of memory goes back down by itself. Default 1.
    #[arg(long, global = true, value_name = "N", value_parser = parse_gpu_slots, help_heading = "AI")]
    pub gpu_slots: Option<usize>,

    /// The AI plugin library to try first (`libevanalyzer_ai.so`,
    /// `evanalyzer_ai.dll`, `libevanalyzer_ai.dylib`) - for development and
    /// tests; a package finds its own next to the program.
    #[arg(long, global = true, value_name = "FILE", help_heading = "AI")]
    pub ai_plugin: Option<std::path::PathBuf>,

    /// Most CPU cores EVAnalyzer uses - the analysis, AI models, the results
    /// database and everything else - e.g. to keep a computer usable during
    /// a long analysis. Default: Settings › Performance, else all of them.
    #[arg(long, global = true, value_name = "N", value_parser = parse_at_least_one::<usize>, help_heading = "Resources")]
    pub max_threads: Option<usize>,

    /// Most RAM in MB EVAnalyzer plans with - caches, threads and the
    /// results database are sized as if the computer had no more. Not a
    /// hard limit on every allocation (a container's limit is). Default:
    /// Settings › Performance, else the computer's RAM.
    #[arg(long, global = true, value_name = "MB", value_parser = parse_at_least_one::<u64>, help_heading = "Resources")]
    pub max_ram: Option<u64>,

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

        /// Folder the server keeps its state in (certificate, sessions, the
        /// single user's home). Default: `/var/lib/evanalyzer` for a system
        /// service, otherwise `~/.evanalyzer/server` (Linux) or
        /// `~/evanalyzer/server` (macOS, Windows).
        #[arg(long, value_name = "DIR")]
        data_dir: Option<std::path::PathBuf>,
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

        /// Stop once no client has been connected and no analysis has run
        /// for this many minutes. Default: run until stopped.
        #[arg(long, value_name = "MINUTES")]
        idle_timeout: Option<u64>,
    },
}

pub fn parse_args() -> Args {
    Args::parse()
}

/// A number of at least 1, for `--max-threads` and `--max-ram`.
fn parse_at_least_one<T: std::str::FromStr + PartialOrd + From<u8>>(s: &str) -> Result<T, String> {
    match s.trim().parse::<T>() {
        Ok(n) if n >= T::from(1) => Ok(n),
        _ => Err(format!("`{s}` is not a number of at least 1")),
    }
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
            "--remote-fingerprint",
            "--no-tls-verification",
            "\n  server ",
            "\n  worker ",
            "\n  hash-password ",
            "AI:",
            "--ai-devices",
            "--gpu-slots",
            "--ai-plugin",
            "Resources:",
            "--max-threads",
            "--max-ram",
        ] {
            assert!(help.contains(shown), "{shown} missing in:\n{help}");
        }
        assert!(!help.contains("[env:"), "no environment variables:\n{help}");
    }

    #[test]
    fn ai_device_options_parse_in_every_mode() {
        let args = Args::try_parse_from(["evanalyzer", "--ai-devices", "0,2", "--gpu-slots", "3"])
            .unwrap();
        assert_eq!(args.ai_devices, Some(AiDeviceSelection::Gpus(vec![0, 2])));
        assert_eq!(args.gpu_slots, Some(3));

        // Global: also after a subcommand.
        let args = Args::try_parse_from([
            "evanalyzer",
            "worker",
            "--ai-devices",
            "cpu",
            "--gpu-slots",
            "2",
        ])
        .unwrap();
        assert_eq!(args.ai_devices, Some(AiDeviceSelection::Cpu));
        assert_eq!(args.gpu_slots, Some(2));

        let defaults = Args::try_parse_from(["evanalyzer"]).unwrap();
        assert_eq!((defaults.ai_devices, defaults.gpu_slots), (None, None));

        assert!(Args::try_parse_from(["evanalyzer", "--ai-devices", "gpu"]).is_err());
        assert!(Args::try_parse_from(["evanalyzer", "--gpu-slots", "0"]).is_err());
    }

    #[test]
    fn resource_limits_parse_in_every_mode() {
        let args = Args::try_parse_from(["evanalyzer", "--max-threads", "4", "--max-ram", "8192"])
            .unwrap();
        assert_eq!((args.max_threads, args.max_ram), (Some(4), Some(8192)));

        // Global: also after a subcommand (a server passes them to workers).
        let args = Args::try_parse_from([
            "evanalyzer",
            "worker",
            "--max-threads",
            "2",
            "--ai-plugin",
            "/opt/a.so",
        ])
        .unwrap();
        assert_eq!(args.max_threads, Some(2));
        assert_eq!(args.ai_plugin, Some(std::path::PathBuf::from("/opt/a.so")));

        let defaults = Args::try_parse_from(["evanalyzer"]).unwrap();
        assert_eq!((defaults.max_threads, defaults.max_ram), (None, None));

        for wrong in [
            ["--max-threads", "0"],
            ["--max-ram", "0"],
            ["--max-ram", "8G"],
        ] {
            let mut args = vec!["evanalyzer"];
            args.extend(wrong);
            assert!(Args::try_parse_from(args).is_err(), "{wrong:?}");
        }
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
            data_dir,
        }) = args.command
        else {
            panic!("expected the server command");
        };
        assert_eq!(config, Some("/etc/eva.toml".into()));
        assert!(listen.is_none() && data_dir.is_none());
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
    fn tls_verification_is_skipped_only_on_request_and_not_with_a_fingerprint() {
        let login = ["evanalyzer", "--remote", "wss://h", "--user", "a"];
        assert!(!Args::try_parse_from(login).unwrap().no_tls_verification);
        let args = Args::try_parse_from([&login[..], &["--no-tls-verification"]].concat()).unwrap();
        assert!(args.no_tls_verification);
        assert!(
            Args::try_parse_from(
                [
                    &login[..],
                    &["--no-tls-verification", "--remote-fingerprint", "AB"]
                ]
                .concat()
            )
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
