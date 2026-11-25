//! # exports
//!
//! `evanalyzer cli export ...`: a project's exports (`docs/exports.md`) -
//! list them, show one.
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-11-24
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.

use crate::args::{ExportListArgs, ExportRunArgs, ExportViewArgs};
use evanalyzer_app::backend::Backend;
use evanalyzer_app::exports::find_export;
use evanalyzer_app::exports::naming;
use evanalyzer_app::project::load_project;
use evanalyzer_cfg::core_types::InternalErrors;
use evanalyzer_cfg::settings::export_settings::{
    ExportEntry, ExportSettings, IfExists, LengthUnit, OutputLocation,
};

pub fn run_list(args: ExportListArgs, backend: &dyn Backend) -> Result<(), InternalErrors> {
    let project = load_project(backend.files(), &args.project)?;
    let exports = &project.settings.exports;
    if args.json {
        println!(
            "{}",
            serde_json::to_string_pretty(exports).map_err(json_error)?
        );
    } else {
        print!("{}", list_text(exports));
    }
    Ok(())
}

pub fn run_view(args: ExportViewArgs, backend: &dyn Backend) -> Result<(), InternalErrors> {
    let project = load_project(backend.files(), &args.project)?;
    let exports = &project.settings.exports;
    let export =
        find_export(exports, &args.export).ok_or_else(|| no_such_export(exports, &args.export))?;
    if args.json {
        println!(
            "{}",
            serde_json::to_string_pretty(export).map_err(json_error)?
        );
    } else {
        print!("{}", view_text(export));
    }
    Ok(())
}

/// Runs an export on a run's results (where they are: here or on the
/// server) and prints what it wrote. Fails if an entry failed.
pub fn run_run(args: ExportRunArgs, backend: &dyn Backend) -> Result<(), InternalErrors> {
    let project = load_project(backend.files(), &args.project)?;
    let exports = &project.settings.exports;
    let export = find_export(exports, &args.export)
        .ok_or_else(|| no_such_export(exports, &args.export))?
        .clone();
    let project_dir = args
        .project
        .parent()
        .map(std::path::Path::to_path_buf)
        .unwrap_or_default();
    let database = evanalyzer_app::exports::results_database(
        backend.files(),
        &project_dir,
        args.results.as_deref(),
    )?;
    let scripts = if args.allow_scripts {
        Scripts::All
    } else if args.no_scripts {
        Scripts::None
    } else {
        Scripts::Allowed
    };
    let failed = run_export_on(
        backend,
        &project,
        &project_dir,
        export,
        &database,
        args.output,
        scripts,
    )?;
    if failed > 0 {
        return Err(InternalErrors::Internal(format!("{failed} entries failed")));
    }
    Ok(())
}

/// Which of an export's scripts run.
#[derive(Clone, Copy)]
pub(crate) enum Scripts {
    /// All (`--allow-scripts`).
    All,
    /// None (`--no-scripts`).
    None,
    /// Those the user trusts (`trust.json`: folders and scripts).
    Allowed,
}

/// Runs `export` on the results `database` (where they are: here or on
/// the server) and prints what it wrote - the number of entries that
/// failed.
pub(crate) fn run_export_on(
    backend: &dyn Backend,
    project: &evanalyzer_app::project::ProjectWithRuntime,
    project_dir: &std::path::Path,
    export: ExportSettings,
    database: &std::path::Path,
    output: Option<std::path::PathBuf>,
    scripts: Scripts,
) -> Result<usize, InternalErrors> {
    use evanalyzer_app::exports::{EntryStatus, ScriptPermission};
    println!("Export:  {} “{}”", export.id, export.name);
    println!("Results: {}", database.display());
    let scripts = match scripts {
        Scripts::All => ScriptPermission::All,
        Scripts::None => ScriptPermission::Skip,
        Scripts::Allowed => ScriptPermission::Trusted(backend.load_script_trust()?),
    };
    let (results, run) = evanalyzer_app::exports::prepare_run(
        backend,
        project,
        project_dir,
        export,
        database,
        output,
        scripts,
    )?;
    let cancel = std::sync::atomic::AtomicBool::new(false);
    let report = results.run_export(&run, &cancel, &mut |message, current, total| {
        if current < total {
            println!("  [{}/{total}] {message}", current + 1);
        }
    })?;
    println!("Folder:  {}", report.folder.display());
    let mut failed = 0;
    for entry in &report.entries {
        let label = run
            .export
            .entries
            .get(entry.entry)
            .map(|e| entry_text(&run.export, e))
            .unwrap_or_default();
        let n = entry.entry + 1;
        match &entry.status {
            EntryStatus::Written => println!("  {n}. {label}: {} file(s)", entry.files.len()),
            EntryStatus::Skipped(why) if why == evanalyzer_app::exports::NOT_ALLOWED => println!(
                "  {n}. {label}: skipped - {why}: allow it in the app, or run with --allow-scripts"
            ),
            EntryStatus::Skipped(why) => println!("  {n}. {label}: skipped - {why}"),
            EntryStatus::Failed(why) => {
                failed += 1;
                println!("  {n}. {label}: failed - {why}");
            }
        }
        for file in &entry.files {
            println!("       {}", file.display());
        }
        for note in &entry.notes {
            println!("       note: {note}");
        }
        if !matches!(entry.status, EntryStatus::Written) && !entry.log.is_empty() {
            for line in entry
                .log
                .lines()
                .rev()
                .take(20)
                .collect::<Vec<_>>()
                .into_iter()
                .rev()
            {
                println!("       | {line}");
            }
        }
    }
    Ok(failed)
}

fn json_error(e: serde_json::Error) -> InternalErrors {
    InternalErrors::Internal(e.to_string())
}

/// No export `wanted` - with the ones there are.
pub(crate) fn no_such_export(exports: &[ExportSettings], wanted: &str) -> InternalErrors {
    let there: Vec<String> = exports
        .iter()
        .map(|export| format!("{} “{}”", export.id, export.name))
        .collect();
    InternalErrors::InvalidArgument(if there.is_empty() {
        format!("no export “{wanted}” - the project has no exports")
    } else {
        format!(
            "no export “{wanted}” - the project's exports: {}",
            there.join(", ")
        )
    })
}

/// A line per export: id, name, what it writes, where to, after analysis.
fn list_text(exports: &[ExportSettings]) -> String {
    if exports.is_empty() {
        return "The project has no exports.\n".to_string();
    }
    let mut out = String::new();
    let id_width = exports
        .iter()
        .map(|e| e.id.to_string().len())
        .max()
        .unwrap_or(2)
        .max(2);
    let name_width = exports
        .iter()
        .map(|e| e.name.chars().count())
        .max()
        .unwrap_or(4)
        .max(4);
    out.push_str(&format!(
        "{:>id_width$}  {:<name_width$}  {:<5}  {}\n",
        "ID", "Name", "After", "Writes"
    ));
    for export in exports {
        out.push_str(&format!(
            "{:>id_width$}  {:<name_width$}  {:<5}  {} → {}\n",
            export.id,
            export.name,
            if export.run_after_analysis {
                "yes"
            } else {
                "no"
            },
            entries_summary(export),
            output_text(export)
        ));
    }
    out
}

/// "objects (CSV), images (CSV), drawn images, script stats".
fn entries_summary(export: &ExportSettings) -> String {
    if export.entries.is_empty() {
        return "nothing yet".to_string();
    }
    export
        .entries
        .iter()
        .map(|entry| entry_text(export, entry))
        .collect::<Vec<_>>()
        .join(", ")
}

fn entry_text(export: &ExportSettings, entry: &ExportEntry) -> String {
    let names: Vec<String> = naming::entry_names(entry)
        .iter()
        .map(|name| naming::with_prefix(export, name))
        .collect();
    match entry {
        ExportEntry::Data(data) => {
            let split = match data.split {
                evanalyzer_cfg::settings::export_settings::TableSplit::None => String::new(),
                split => match data.split_into {
                    evanalyzer_cfg::settings::export_settings::SplitInto::Sheets => {
                        format!(", a sheet per {split:?}").to_lowercase()
                    }
                    evanalyzer_cfg::settings::export_settings::SplitInto::Files => {
                        format!(", a file per {split:?}").to_lowercase()
                    }
                },
            };
            format!("data: {} ({:?}{split})", names.join(", "), data.format)
        }
        ExportEntry::DrawnImages(_) => format!("{} (PNG per image)", names.join(", ")),
        ExportEntry::Script(script) => format!(
            "script {} → {}",
            script.file,
            naming::with_prefix(export, &naming::script_output(script))
        ),
    }
}

fn output_text(export: &ExportSettings) -> String {
    let folder = export.output.folder.trim();
    match export.output.location {
        OutputLocation::ResultsFolder if folder.is_empty() => "the results folder".to_string(),
        OutputLocation::ResultsFolder => format!("the results folder/{folder}"),
        OutputLocation::FixedFolder => folder.to_string(),
    }
}

/// One export's settings, readable.
fn view_text(export: &ExportSettings) -> String {
    let mut out = String::new();
    out.push_str(&format!("Export {}: {}\n", export.id, export.name));
    out.push_str(&format!("  Output:      {}\n", output_text(export)));
    out.push_str(&format!(
        "  If a file exists: {}\n",
        match export.output.if_exists {
            IfExists::Overwrite => "overwrite it",
            IfExists::Rename => "keep it, write the new one with _2, _3, ...",
        }
    ));
    out.push_str(&format!(
        "  Prefix:      {}\n",
        if export.prefix.trim().is_empty() {
            "none"
        } else {
            export.prefix.trim()
        }
    ));
    out.push_str(&format!(
        "  Lengths in:  {}\n",
        match export.length_unit {
            LengthUnit::Um => "µm",
            LengthUnit::Nm => "nm",
        }
    ));
    out.push_str(&format!(
        "  After every run: {}\n",
        if export.run_after_analysis {
            "yes"
        } else {
            "no"
        }
    ));
    out.push_str(&format!("  Table format: {}\n", export.table_format));
    if export.entries.is_empty() {
        out.push_str("  Writes nothing yet.\n");
    } else {
        out.push_str("  Writes, in this order:\n");
        for (i, entry) in export.entries.iter().enumerate() {
            out.push_str(&format!("    {}. {}\n", i + 1, entry_text(export, entry)));
            out.push_str(&entry_details(entry));
        }
    }
    out
}

fn entry_details(entry: &ExportEntry) -> String {
    let classes = |classes: &Option<Vec<u32>>| match classes {
        None => "all".to_string(),
        Some(ids) => ids
            .iter()
            .map(u32::to_string)
            .collect::<Vec<_>>()
            .join(", "),
    };
    let columns = |columns: &Option<Vec<String>>, default: &str| match columns {
        None => default.to_string(),
        Some(keys) => keys.join(", "),
    };
    let lines: Vec<String> = match entry {
        ExportEntry::Data(data) => {
            let mut lines = vec![
                format!("outputs: {:?}", data.outputs),
                format!("classes: {}", classes(&data.classes)),
                format!(
                    "columns: {}",
                    columns(&data.columns, "each output's default ones")
                ),
                format!("aggregations: {:?}", data.aggregations),
                format!("images: {:?}", data.images),
            ];
            if data.planes.z.is_some() || data.planes.t.is_some() {
                lines.push(format!(
                    "planes: z {:?}, t {:?}",
                    data.planes.z, data.planes.t
                ));
            }
            if data.transposed {
                lines.push("transposed: the classes side by side".into());
            }
            if !data.children.is_empty() || !data.parents.is_empty() {
                lines.push(format!(
                    "children: {:?} ({:?}), linked objects: {:?}",
                    data.children, data.child_aggregations, data.parents
                ));
            }
            lines
        }
        ExportEntry::DrawnImages(drawn) => vec![
            format!("classes: {}", classes(&drawn.classes)),
            format!(
                "style: {:?}, brightness: {:?}",
                drawn.style, drawn.brightness
            ),
            format!("max side: {} px", drawn.max_side),
        ],
        ExportEntry::Script(script) => vec![format!("language: {:?}", script.language)],
    };
    lines
        .iter()
        .map(|line| format!("       {line}\n"))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::commands::test_support::TempProjectFile;
    use evanalyzer_app::backend::local::LocalBackend;
    use evanalyzer_cfg::settings::project_settings::ProjectSettings;
    use serde_json::json;

    fn exports() -> Vec<ExportSettings> {
        serde_json::from_value(json!([
            { "id": 3, "name": "Per image for R", "prefix": "exp1", "runAfterAnalysis": true,
              "entries": [
                { "kind": "DATA", "outputs": ["OBJECTS"], "split": "IMAGE" },
                { "kind": "DRAWN_IMAGES" },
                { "kind": "SCRIPT", "language": "R", "file": "scripts/stats.R" }
              ] },
            { "id": 4, "name": "Plate", "output": { "location": "FIXED_FOLDER", "folder": "exports" },
              "entries": [{ "kind": "DATA", "outputs": ["PLATE"] }] }
        ]))
        .unwrap()
    }

    #[test]
    fn the_list_has_a_line_per_export() {
        let text = list_text(&exports());
        let lines: Vec<&str> = text.lines().collect();
        assert_eq!(lines.len(), 3);
        assert!(lines[1].contains("Per image for R") && lines[1].contains("yes"));
        assert!(lines[1].contains("exp1_objects (Csv, a file per image)"));
        assert!(lines[1].contains("script scripts/stats.R → exp1_stats"));
        assert!(lines[1].ends_with("the results folder"));
        assert!(lines[2].contains("data: plate (Csv)") && lines[2].ends_with("exports"));
        assert_eq!(list_text(&[]), "The project has no exports.\n");
    }

    #[test]
    fn the_view_shows_every_setting_and_entry() {
        let exports = exports();
        let text = view_text(&exports[0]);
        assert!(text.starts_with("Export 3: Per image for R\n"));
        assert!(text.contains("Prefix:      exp1"));
        assert!(text.contains("Lengths in:  µm"));
        assert!(text.contains("1. data: exp1_objects"));
        assert!(text.contains("outputs: [Objects]"));
        assert!(text.contains("3. script scripts/stats.R"));
    }

    /// `export run` on given results, into a folder of its own: the tables
    /// are there; an unknown export names the ones there are.
    #[test]
    fn run_writes_the_export_into_the_output_folder() {
        let mut settings = ProjectSettings::default();
        settings.exports = serde_json::from_value(json!([
            { "id": 1, "name": "Tables", "entries": [
                { "kind": "DATA", "outputs": ["OBJECTS"] },
                { "kind": "DATA", "outputs": ["IMAGES"] }
            ] }
        ]))
        .unwrap();
        let file = TempProjectFile::new(&settings);
        let db = crate::commands::test_support::TempResultsDb::seeded();
        let out = tempfile::tempdir().unwrap();
        let backend = LocalBackend::default();
        let run = |export: &str| {
            run_run(
                ExportRunArgs {
                    export: export.into(),
                    project: file.path.clone(),
                    results: Some(db.path.clone()),
                    output: Some(out.path().to_path_buf()),
                    allow_scripts: false,
                    no_scripts: false,
                },
                &backend,
            )
        };
        run("tables").expect("the export runs");
        let objects = std::fs::read_to_string(out.path().join("objects.csv")).unwrap();
        assert!(objects.starts_with("image,class,object_id,"), "{objects}");
        assert!(out.path().join("images.csv").exists());
        assert!(run("nothing").is_err());
    }

    #[test]
    fn view_finds_an_export_by_id_or_name_and_names_the_others_if_not() {
        let mut settings = ProjectSettings::default();
        settings.exports = exports();
        let file = TempProjectFile::new(&settings);
        let backend = LocalBackend::default();
        let view = |export: &str| {
            run_view(
                ExportViewArgs {
                    export: export.into(),
                    project: file.path.clone(),
                    json: true,
                },
                &backend,
            )
        };
        assert!(view("3").is_ok());
        assert!(view("plate").is_ok());
        let Err(InternalErrors::InvalidArgument(message)) = view("nothing") else {
            panic!("an error");
        };
        assert!(message.contains("3 “Per image for R”") && message.contains("4 “Plate”"));
    }
}
