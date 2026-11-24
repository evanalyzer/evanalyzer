//! # script
//!
//! A `SCRIPT` entry: the project's R or Python script run on the files the
//! entries before it wrote - `<interpreter> <file> --input <output folder>
//! --output <output folder>/<output>`, in the project's folder, nothing
//! else passed (no environment variables). Only with the user's OK for this
//! script's content, or for all (`--allow-scripts`).
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-11-24
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.

use super::Output;
use crate::exports::naming;
use crate::exports::script_hash;
use crate::exports::types::{EntryReport, EntryStatus, ExportRun, ScriptPermission};
use evanalyzer_cfg::core_types::InternalErrors;
use evanalyzer_cfg::settings::export_settings::{ScriptExport, ScriptLanguage};
use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

/// At most this much of a script's output is kept (its end).
const LOG_LIMIT: usize = 64 * 1024;

pub(super) fn run(
    run: &ExportRun,
    script: &ScriptExport,
    output: &Output,
    report: &mut EntryReport,
    cancel: &AtomicBool,
) -> Result<(), InternalErrors> {
    let file = run.project_dir.join(script.file.trim().replace('\\', "/"));
    let content = std::fs::read(&file).map_err(|e| {
        InternalErrors::Io(format!("Could not read the script {}: {e}", file.display()))
    })?;
    let hash = script_hash(&content);
    let allowed = match &run.scripts {
        ScriptPermission::All => true,
        ScriptPermission::Trusted(hashes) => hashes.contains(&hash),
        ScriptPermission::Skip => {
            report.status = EntryStatus::Skipped("scripts are off for this run".into());
            return Ok(());
        }
        ScriptPermission::Refused(why) => {
            report.status = EntryStatus::Skipped(why.clone());
            return Ok(());
        }
    };
    if !allowed {
        report.status = EntryStatus::Skipped(format!(
            "{} hasn't been allowed to run (yet) - allow it, or run with --allow-scripts",
            script.file
        ));
        report.notes.push(format!("script hash {hash}"));
        return Ok(());
    }

    let folder = naming::with_prefix(&run.export, &naming::script_output(script));
    let out_dir = output.path(Path::new(&folder));
    std::fs::create_dir_all(&out_dir)
        .map_err(|e| InternalErrors::Io(format!("Could not create {}: {e}", out_dir.display())))?;
    let interpreter = interpreter(run, script.language);
    let mut child = Command::new(&interpreter)
        .arg(&file)
        .arg("--input")
        .arg(output.path(Path::new("")))
        .arg("--output")
        .arg(&out_dir)
        .current_dir(&run.project_dir)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| {
            InternalErrors::Io(format!(
                "Could not start {} - is it installed? Its path can be set in the settings ({e})",
                interpreter.display()
            ))
        })?;

    // Both streams read meanwhile, so a chatty script never blocks on a
    // full pipe.
    let read = |stream: Option<Box<dyn Read + Send>>| {
        std::thread::spawn(move || {
            let mut text = Vec::new();
            if let Some(mut stream) = stream {
                let _ = stream.read_to_end(&mut text);
            }
            text
        })
    };
    let stdout = read(
        child
            .stdout
            .take()
            .map(|s| Box::new(s) as Box<dyn Read + Send>),
    );
    let stderr = read(
        child
            .stderr
            .take()
            .map(|s| Box::new(s) as Box<dyn Read + Send>),
    );
    let status = loop {
        if cancel.load(Ordering::Relaxed) {
            let _ = child.kill();
            let _ = child.wait();
            return Err(InternalErrors::Cancelled);
        }
        match child.try_wait() {
            Ok(Some(status)) => break status,
            Ok(None) => std::thread::sleep(Duration::from_millis(50)),
            Err(e) => {
                return Err(InternalErrors::Io(format!(
                    "Could not wait for {}: {e}",
                    interpreter.display()
                )));
            }
        }
    };
    let mut log = stdout.join().unwrap_or_default();
    log.extend(stderr.join().unwrap_or_default());
    report.log = tail(&String::from_utf8_lossy(&log), LOG_LIMIT);

    report.files = files_below(&out_dir)
        .into_iter()
        .filter_map(|path| {
            path.strip_prefix(output.path(Path::new("")))
                .ok()
                .map(Path::to_path_buf)
        })
        .collect();
    if !status.success() {
        report.status = EntryStatus::Failed(match status.code() {
            Some(code) => format!("{} stopped with exit code {code}", script.file),
            None => format!("{} was stopped", script.file),
        });
    }
    Ok(())
}

/// The interpreter of `language`: the set one, else its usual name on the
/// `PATH`.
fn interpreter(run: &ExportRun, language: ScriptLanguage) -> PathBuf {
    let (set, name) = match language {
        ScriptLanguage::R => (&run.interpreters.r, "Rscript"),
        ScriptLanguage::Python => (
            &run.interpreters.python,
            if cfg!(windows) { "python" } else { "python3" },
        ),
    };
    set.clone().unwrap_or_else(|| PathBuf::from(name))
}

/// The last `limit` bytes of `text` (at a character boundary).
fn tail(text: &str, limit: usize) -> String {
    if text.len() <= limit {
        return text.to_string();
    }
    let mut start = text.len() - limit;
    while !text.is_char_boundary(start) {
        start += 1;
    }
    format!("…{}", &text[start..])
}

fn files_below(dir: &Path) -> Vec<PathBuf> {
    let mut found = Vec::new();
    let mut todo = vec![dir.to_path_buf()];
    while let Some(dir) = todo.pop() {
        let Ok(entries) = std::fs::read_dir(&dir) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                todo.push(path);
            } else {
                found.push(path);
            }
        }
    }
    found.sort();
    found
}
