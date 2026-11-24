//! # local
//!
//! Runs an export on this machine, on results it can read: the output
//! folder, then the entries in their order - each one's files, or why it
//! failed or was skipped. Where the results are on a server, the server's
//! worker runs this.
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-11-24
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.

mod measurements;
mod reused;
mod script;
mod series;
pub(crate) mod table;
mod tracks;

use super::check::check_export;
use super::naming;
use super::types::{EntryReport, EntryStatus, ExportReport, ExportRun};
use crate::backend::ExportProgressFn;
use crate::modules::Severity;
use crate::results::ResultsSource;
use evanalyzer_cfg::core_types::InternalErrors;
use evanalyzer_cfg::settings::export_settings::{ExportEntry, IfExists, OutputLocation};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};

/// Runs `run` on `results`. `may_write` says whether the folder may be
/// written (a server: within its roots). Refused as a whole for an export
/// with errors or a folder it may not write; otherwise every entry runs -
/// one failing doesn't stop the others.
pub(crate) fn run_export(
    results: &dyn ResultsSource,
    run: &ExportRun,
    may_write: &dyn Fn(&Path) -> Result<(), InternalErrors>,
    cancel: &AtomicBool,
    on_progress: ExportProgressFn,
) -> Result<ExportReport, InternalErrors> {
    let export = &run.export;
    let errors: Vec<String> = check_export(export)
        .into_iter()
        .filter(|problem| problem.severity == Severity::Error)
        .map(|problem| match problem.entry {
            Some(i) => format!("entry {}: {}", i + 1, problem.message),
            None => problem.message,
        })
        .collect();
    if !errors.is_empty() {
        return Err(InternalErrors::InvalidArgument(format!(
            "export “{}” can't run: {}",
            export.name,
            errors.join("; ")
        )));
    }

    let folder = output_folder(run);
    may_write(&folder)?;
    std::fs::create_dir_all(&folder)
        .map_err(|e| InternalErrors::Io(format!("Could not create {}: {e}", folder.display())))?;
    let output = Output {
        dir: folder.clone(),
        if_exists: export.output.if_exists,
    };

    let total = export.entries.len();
    let mut entries = Vec::with_capacity(total);
    for (i, entry) in export.entries.iter().enumerate() {
        if cancel.load(Ordering::Relaxed) {
            return Err(InternalErrors::Cancelled);
        }
        on_progress(&entry_label(export, entry), i, total);
        let mut report = EntryReport {
            entry: i,
            status: EntryStatus::Written,
            files: Vec::new(),
            notes: Vec::new(),
            log: String::new(),
        };
        let result = match entry {
            ExportEntry::Data(data) => {
                data_export(results, export, data, &output, &mut report, cancel)
            }
            ExportEntry::DrawnImages(drawn) => match &run.drawing {
                Some(drawing) => reused::drawn_images(
                    results,
                    export,
                    drawn,
                    drawing,
                    &output,
                    &mut report,
                    cancel,
                ),
                None => {
                    report.status =
                        EntryStatus::Skipped("drawing needs the project's image settings".into());
                    Ok(())
                }
            },
            ExportEntry::Script(script) => script::run(run, script, &output, &mut report, cancel),
        };
        match result {
            Ok(()) => {}
            Err(InternalErrors::Cancelled) => return Err(InternalErrors::Cancelled),
            Err(e) => report.status = EntryStatus::Failed(e.to_string()),
        }
        entries.push(report);
    }
    on_progress("Export done", total, total);
    Ok(ExportReport { folder, entries })
}

/// A data export: each output with the shared filter - an output that fails
/// fails alone (said in the report), the others are written.
fn data_export(
    results: &dyn ResultsSource,
    export: &evanalyzer_cfg::settings::export_settings::ExportSettings,
    data: &evanalyzer_cfg::settings::export_settings::DataExport,
    output: &Output,
    report: &mut EntryReport,
    cancel: &AtomicBool,
) -> Result<(), InternalErrors> {
    let mut failed: Vec<String> = Vec::new();
    for (_, job) in crate::exports::jobs::jobs_of(data) {
        if cancel.load(Ordering::Relaxed) {
            return Err(InternalErrors::Cancelled);
        }
        let name = naming::job_name(&job);
        // Notes of this output, named by it.
        let notes_before = report.notes.len();
        let result = match &job {
            crate::exports::jobs::Job::Table(table) => {
                table::write(results, export, table, output, report, cancel)
            }
            crate::exports::jobs::Job::PlateLayout(plate) => {
                reused::plate_layout(results, export, plate, output, report, cancel)
            }
        };
        for note in &mut report.notes[notes_before..] {
            *note = format!("{name}: {note}");
        }
        match result {
            Ok(()) => {}
            Err(InternalErrors::Cancelled) => return Err(InternalErrors::Cancelled),
            Err(e) => failed.push(format!("{name}: {e}")),
        }
    }
    if !failed.is_empty() {
        report.status = EntryStatus::Failed(failed.join("; "));
    }
    Ok(())
}

/// Where `run` writes: `--output` if given, else the results folder (and
/// its subfolder) or the fixed folder, relative to the project.
pub(crate) fn output_folder(run: &ExportRun) -> PathBuf {
    if let Some(output) = &run.output {
        return output.clone();
    }
    let folder = run.export.output.folder.trim().replace('\\', "/");
    match run.export.output.location {
        OutputLocation::ResultsFolder if folder.is_empty() => run.results_dir.clone(),
        OutputLocation::ResultsFolder => run.results_dir.join(&folder),
        OutputLocation::FixedFolder => {
            let path = Path::new(&folder);
            if path.is_absolute() {
                path.to_path_buf()
            } else {
                run.project_dir.join(path)
            }
        }
    }
}

/// "objects", "images", "script stats" - for the progress.
fn entry_label(
    export: &evanalyzer_cfg::settings::export_settings::ExportSettings,
    entry: &ExportEntry,
) -> String {
    match entry {
        ExportEntry::Script(script) => format!("Script {}", script.file),
        other => naming::entry_names(other)
            .iter()
            .map(|name| naming::with_prefix(export, name))
            .collect::<Vec<_>>()
            .join(", "),
    }
}

/// The folder an export writes into, and what happens to a file already
/// there.
pub(super) struct Output {
    dir: PathBuf,
    if_exists: IfExists,
}

impl Output {
    /// Where the file `name` goes (`RENAME`: the next free name), relative
    /// to the folder.
    pub(super) fn file(&self, name: &str) -> PathBuf {
        match self.if_exists {
            IfExists::Overwrite => PathBuf::from(name),
            IfExists::Rename => PathBuf::from(naming::free_name(name, |candidate| {
                self.dir.join(candidate).exists()
            })),
        }
    }

    pub(super) fn path(&self, relative: &Path) -> PathBuf {
        self.dir.join(relative)
    }
}

#[cfg(test)]
mod tests;

/// Whether the plane `(z, t)` is one of `planes` (`Planes`: ranges, both
/// ends included; `None`: all).
pub(crate) fn in_planes(
    planes: &evanalyzer_cfg::settings::export_settings::Planes,
    z: u32,
    t: u32,
) -> bool {
    let within = |range: Option<[u32; 2]>, value: u32| {
        range.is_none_or(|[from, to]| from <= value && value <= to)
    };
    within(planes.z, z) && within(planes.t, t)
}

/// `planes`' range of `axis` as the results' export takes it (end
/// excluded).
pub(crate) fn plane_range(range: Option<[u32; 2]>) -> Option<std::range::Range<u32>> {
    range.map(|[from, to]| std::range::Range {
        start: from,
        end: to.saturating_add(1),
    })
}
