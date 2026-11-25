//! # trust
//!
//! `evanalyzer cli export trust ...`: the folders and scripts exports may
//! run without asking (`trust.json` in the user folder - the server's with
//! `--remote`, where the scripts run) - list them, trust a folder, allow a
//! script, take it back.
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-11-25
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.

use crate::args::TrustSubCommand;
use evanalyzer_app::backend::Backend;
use evanalyzer_app::exports::{TrustedScriptStatus, script_hash, trusted_script_status};
use evanalyzer_app::user::ScriptTrust;
use evanalyzer_app::user::trust::{normalize, today};
use evanalyzer_cfg::core_types::InternalErrors;
use std::path::{Path, PathBuf};

pub fn run(command: TrustSubCommand, backend: &dyn Backend) -> Result<(), InternalErrors> {
    let mut trust = backend.load_script_trust()?;
    match command {
        TrustSubCommand::List => {
            print!("{}", list_text(backend, &trust));
            return Ok(());
        }
        TrustSubCommand::Folder(args) => {
            let folder = absolute(backend, &args.path)?;
            if !backend.files().stat(&folder)?.is_some_and(|e| e.is_dir) {
                return Err(InternalErrors::InvalidArgument(format!(
                    "{} is not a folder",
                    folder.display()
                )));
            }
            trust.trust_folder(&folder, &today());
            println!("Trusted: every script in {}", normalize(&folder));
        }
        TrustSubCommand::Allow(args) => {
            let script = absolute(backend, &args.path)?;
            let bytes = backend.files().read_file(&script)?;
            trust.allow_script(&script, &script_hash(&bytes), &today());
            println!(
                "Allowed: {} (asks again once it changes)",
                normalize(&script)
            );
        }
        TrustSubCommand::Revoke(args) => {
            if args.all {
                let n = trust.folders.len() + trust.scripts.len();
                trust = ScriptTrust::default();
                println!("Revoked: {n} folder(s) and script(s)");
            } else if let Some(path) = args.path {
                let path = normalize(&absolute(backend, &path)?);
                let before = trust.folders.len() + trust.scripts.len();
                trust.revoke_folder(&path);
                trust.revoke_script(&path);
                if trust.folders.len() + trust.scripts.len() == before {
                    return Err(InternalErrors::InvalidArgument(format!(
                        "{path} is not trusted - `export trust list` shows what is"
                    )));
                }
                println!("Revoked: {path}");
            }
        }
    }
    backend.save_script_trust(&trust)
}

/// `path` absolute: as given on a server (there is no folder to start
/// from there), else from the current folder.
fn absolute(backend: &dyn Backend, path: &Path) -> Result<PathBuf, InternalErrors> {
    if path.is_absolute() {
        return Ok(path.to_path_buf());
    }
    if backend.is_remote() {
        return Err(InternalErrors::InvalidArgument(format!(
            "{}: give the path on the server in full",
            path.display()
        )));
    }
    std::path::absolute(path).map_err(|e| InternalErrors::Io(e.to_string()))
}

fn list_text(backend: &dyn Backend, trust: &ScriptTrust) -> String {
    if trust.is_empty() {
        return "Nothing trusted - exports ask before they run a script.\n".into();
    }
    let mut text = String::new();
    if !trust.folders.is_empty() {
        text.push_str("Folders (every script in them runs):\n");
        for folder in &trust.folders {
            text.push_str(&format!("  {}{}\n", folder.path, since(&folder.since)));
        }
    }
    if !trust.scripts.is_empty() {
        text.push_str("Scripts (run while unchanged):\n");
        for script in &trust.scripts {
            let status = match trusted_script_status(backend.files(), script) {
                TrustedScriptStatus::Matches => "matches",
                TrustedScriptStatus::Changed => "changed - asks again",
                TrustedScriptStatus::Missing => "file missing",
            };
            text.push_str(&format!(
                "  {}{} · {status}\n",
                script.path,
                since(&script.since)
            ));
        }
    }
    text
}

fn since(day: &str) -> String {
    if day.is_empty() {
        String::new()
    } else {
        format!(" · since {day}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::args::{TrustPathArgs, TrustRevokeArgs};
    use evanalyzer_app::backend::local::LocalBackend;

    fn backend(home: &Path) -> LocalBackend {
        LocalBackend::default().with_home(home).unwrap()
    }

    #[test]
    fn folders_and_scripts_are_trusted_listed_and_revoked() {
        let home = tempfile::tempdir().unwrap();
        let work = tempfile::tempdir().unwrap();
        let backend = backend(home.path());
        let project = work.path().join("project");
        std::fs::create_dir_all(project.join("scripts")).unwrap();
        let script = work.path().join("other/plot.R");
        std::fs::create_dir_all(script.parent().unwrap()).unwrap();
        std::fs::write(&script, "plot(1)").unwrap();

        run(
            TrustSubCommand::Folder(TrustPathArgs {
                path: project.clone(),
            }),
            &backend,
        )
        .unwrap();
        run(
            TrustSubCommand::Allow(TrustPathArgs {
                path: script.clone(),
            }),
            &backend,
        )
        .unwrap();
        let trust = backend.load_script_trust().unwrap();
        assert_eq!(trust.folders.len(), 1);
        assert!(trust.allows(&project.join("scripts/any.py"), "x"));
        assert!(trust.allows(&script, &script_hash(b"plot(1)")));
        assert!(list_text(&backend, &trust).contains("matches"));

        std::fs::write(&script, "plot(2)").unwrap();
        assert!(list_text(&backend, &trust).contains("changed - asks again"));
        std::fs::remove_file(&script).unwrap();
        assert!(list_text(&backend, &trust).contains("file missing"));

        run(
            TrustSubCommand::Revoke(TrustRevokeArgs {
                path: Some(project.clone()),
                all: false,
            }),
            &backend,
        )
        .unwrap();
        let trust = backend.load_script_trust().unwrap();
        assert!(trust.folders.is_empty());
        assert_eq!(trust.scripts.len(), 1);

        // Not in the list: says so, changes nothing.
        assert!(
            run(
                TrustSubCommand::Revoke(TrustRevokeArgs {
                    path: Some(project),
                    all: false,
                }),
                &backend,
            )
            .is_err()
        );
        run(
            TrustSubCommand::Revoke(TrustRevokeArgs {
                path: None,
                all: true,
            }),
            &backend,
        )
        .unwrap();
        assert!(backend.load_script_trust().unwrap().is_empty());
    }

    #[test]
    fn a_file_is_not_trusted_as_a_folder() {
        let home = tempfile::tempdir().unwrap();
        let work = tempfile::tempdir().unwrap();
        let file = work.path().join("a.R");
        std::fs::write(&file, "1").unwrap();
        let backend = backend(home.path());
        assert!(
            run(
                TrustSubCommand::Folder(TrustPathArgs { path: file }),
                &backend
            )
            .is_err()
        );
        assert!(backend.load_script_trust().unwrap().is_empty());
    }
}
