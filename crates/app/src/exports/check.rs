//! # check
//!
//! What is wrong with a project's exports (`docs/exports.md` §1.4):
//! errors a run refuses, warnings about what probably isn't meant. Needs
//! only the project - what a run's results hold is checked when it runs.
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-11-24
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.

use super::naming;
use crate::modules::Severity;
use crate::results::Column;
use evanalyzer_cfg::core_types::{is_contained_rel_path, is_file_name_part};
use evanalyzer_cfg::settings::export_settings::{
    CURRENT_TABLE_FORMAT, ExportEntry, ExportSettings, OutputLocation, SplitInto, TableExport,
    TableFileFormat, TableRows, TableSplit,
};
use evanalyzer_cfg::settings::project_settings::ProjectSettings;
use std::collections::{BTreeMap, BTreeSet};

/// One problem of a project's exports.
#[derive(Debug, Clone, PartialEq)]
pub struct ExportProblem {
    pub severity: Severity,
    /// The export's id (`None`: about the exports as a whole).
    pub export: Option<u32>,
    /// The entry's place in the export (0 the first), if it is about one.
    pub entry: Option<usize>,
    pub message: String,
}

/// The problems of `project`'s exports, export by export, entry by entry.
pub fn check_exports(project: &ProjectSettings) -> Vec<ExportProblem> {
    let mut problems = Vec::new();
    let exports = &project.exports;

    let mut ids = BTreeSet::new();
    let mut names = BTreeSet::new();
    for export in exports {
        let mut at = |severity, entry, message: String| {
            problems.push(ExportProblem {
                severity,
                export: Some(export.id),
                entry,
                message,
            })
        };
        if !ids.insert(export.id) {
            at(
                Severity::Error,
                None,
                format!("another export has the id {}", export.id),
            );
        }
        let name = export.name.trim();
        if name.is_empty() {
            at(Severity::Error, None, "has no name".into());
        } else if !names.insert(name.to_lowercase()) {
            at(
                Severity::Error,
                None,
                format!("another export is called “{name}”"),
            );
        }
        if !is_file_name_part(export.prefix.trim()) {
            at(
                Severity::Error,
                None,
                format!("the prefix “{}” isn't usable in a file name", export.prefix),
            );
        }
        check_output(export, &mut at);
        if export.table_format != CURRENT_TABLE_FORMAT {
            at(
                Severity::Error,
                None,
                format!(
                    "table format {} is unknown - this app writes table format {CURRENT_TABLE_FORMAT}",
                    export.table_format
                ),
            );
        }
        for (i, entry) in export.entries.iter().enumerate() {
            check_entry(project, entry, &mut |severity, message| {
                at(severity, Some(i), message)
            });
        }
        check_names_within(export, &mut at);
    }
    check_names_across(exports, &mut problems);
    problems
}

fn check_output(export: &ExportSettings, at: &mut impl FnMut(Severity, Option<usize>, String)) {
    let folder = export.output.folder.trim();
    let has_up = folder.split(['/', '\\']).any(|part| part == "..");
    match export.output.location {
        OutputLocation::ResultsFolder => {
            if !folder.is_empty() && !is_contained_rel_path(folder) {
                at(
                    Severity::Error,
                    None,
                    format!("the folder “{folder}” isn't a folder inside the results folder"),
                );
            }
        }
        OutputLocation::FixedFolder => {
            if folder.is_empty() {
                at(
                    Severity::Error,
                    None,
                    "a fixed folder needs its path".into(),
                );
            } else if has_up || folder.contains('\0') {
                at(
                    Severity::Error,
                    None,
                    format!("the folder “{folder}” can't contain “..”"),
                );
            } else if !is_contained_rel_path(folder) {
                at(
                    Severity::Warning,
                    None,
                    format!(
                        "the folder “{folder}” is an absolute path - it won't be found when the \
                         project is used on another computer"
                    ),
                );
            }
        }
    }
}

fn check_entry(
    project: &ProjectSettings,
    entry: &ExportEntry,
    at: &mut impl FnMut(Severity, String),
) {
    let own_name = match entry {
        ExportEntry::Table(table) => {
            check_table(table, at);
            check_classes(project, table.classes.as_deref(), at);
            check_columns(project, table.columns.as_deref(), at);
            Some(&table.name)
        }
        ExportEntry::PlateLayout(plate) => {
            check_classes(project, plate.classes.as_deref(), at);
            check_columns(project, plate.columns.as_deref(), at);
            Some(&plate.name)
        }
        ExportEntry::DrawnImages(drawn) => {
            check_classes(project, drawn.classes.as_deref(), at);
            if drawn.max_side < 16 {
                at(
                    Severity::Error,
                    format!(
                        "the images can't be smaller than 16 px (max side {})",
                        drawn.max_side
                    ),
                );
            }
            Some(&drawn.name)
        }
        ExportEntry::Script(script) => {
            let file = script.file.trim().replace('\\', "/");
            if !is_contained_rel_path(&file) || !file.starts_with("scripts/") {
                at(
                    Severity::Error,
                    format!(
                        "the script “{}” has to be in the project's scripts folder",
                        script.file
                    ),
                );
            }
            if !is_file_name_part(script.output.trim()) {
                at(
                    Severity::Error,
                    format!(
                        "the output folder “{}” has to be a single folder name",
                        script.output
                    ),
                );
            }
            None
        }
    };
    if let Some(name) = own_name
        && !is_file_name_part(name.trim())
    {
        at(
            Severity::Error,
            format!("the name “{name}” isn't usable in a file name"),
        );
    }
}

fn check_table(table: &TableExport, at: &mut impl FnMut(Severity, String)) {
    let grouped = matches!(
        table.rows,
        TableRows::Image | TableRows::Well | TableRows::Folder | TableRows::Track
    );
    if grouped && table.aggregations.is_empty() {
        at(
            Severity::Error,
            "grouped rows need at least one aggregation".into(),
        );
    }
    if table.split_into == SplitInto::Sheets && table.format == TableFileFormat::Csv {
        at(
            Severity::Error,
            "sheets need XLSX - a CSV file has none".into(),
        );
    }
    if table.tracks.is_some() && table.rows != TableRows::Track {
        at(
            Severity::Error,
            "track options are for a table of tracks only".into(),
        );
    }
    let fits = match table.split {
        TableSplit::None => true,
        TableSplit::Image => !matches!(table.rows, TableRows::Well | TableRows::Folder),
        TableSplit::Well => table.rows != TableRows::Folder,
        TableSplit::Folder => table.rows != TableRows::Well,
        TableSplit::Class => table.rows != TableRows::Measurement,
    };
    if !fits {
        at(
            Severity::Error,
            format!(
                "a table of {:?} rows can't be split by {:?}",
                table.rows, table.split
            )
            .to_lowercase(),
        );
    }
}

/// Class ids the project doesn't have - a class can be added later, so a
/// warning.
fn check_classes(
    project: &ProjectSettings,
    classes: Option<&[u32]>,
    at: &mut impl FnMut(Severity, String),
) {
    let Some(classes) = classes else { return };
    let known: BTreeSet<u32> = project
        .classification
        .classes()
        .iter()
        .map(|c| c.id)
        .collect();
    let unknown: Vec<String> = classes
        .iter()
        .filter(|id| !known.contains(id))
        .map(|id| id.to_string())
        .collect();
    if !unknown.is_empty() {
        at(
            Severity::Warning,
            format!("the project has no class {}", unknown.join(", ")),
        );
    }
}

/// Column keys that aren't any - measurements depend on the run, only their
/// key's form is checked here.
fn check_columns(
    project: &ProjectSettings,
    columns: Option<&[String]>,
    at: &mut impl FnMut(Severity, String),
) {
    let Some(columns) = columns else { return };
    let classes = project.classification.classes();
    let unknown: Vec<&str> = columns
        .iter()
        .filter(|key| {
            Column::from_key(key, classes).is_none() && Column::measurement_key(key).is_none()
        })
        .map(String::as_str)
        .collect();
    if !unknown.is_empty() {
        at(
            Severity::Warning,
            format!("unknown columns: {}", unknown.join(", ")),
        );
    }
}

/// Two entries of one export writing the same name.
fn check_names_within(
    export: &ExportSettings,
    at: &mut impl FnMut(Severity, Option<usize>, String),
) {
    let mut seen: BTreeMap<String, usize> = BTreeMap::new();
    for (i, entry) in export.entries.iter().enumerate() {
        let name = match entry {
            ExportEntry::Script(script) => naming::script_output(script),
            other => naming::entry_name(other).unwrap_or_default(),
        };
        if let Some(first) = seen.insert(name.clone(), i) {
            at(
                Severity::Error,
                Some(i),
                format!(
                    "writes “{name}” like entry {} - give one of them its own name",
                    first + 1
                ),
            );
        }
    }
}

/// Two exports writing the same name into the same folder.
fn check_names_across(exports: &[ExportSettings], problems: &mut Vec<ExportProblem>) {
    let mut seen: BTreeMap<(String, String), u32> = BTreeMap::new();
    for export in exports {
        let folder = (
            matches!(export.output.location, OutputLocation::FixedFolder),
            export
                .output
                .folder
                .trim()
                .trim_end_matches(['/', '\\'])
                .replace('\\', "/"),
        );
        let folder = format!("{}:{}", folder.0, folder.1);
        let mut own = BTreeSet::new();
        for entry in &export.entries {
            let name = match entry {
                ExportEntry::Script(script) => naming::script_output(script),
                other => naming::entry_name(other).unwrap_or_default(),
            };
            let name = naming::with_prefix(export, &name);
            if !own.insert(name.clone()) {
                continue;
            }
            match seen.get(&(folder.clone(), name.clone())) {
                Some(&other) if other != export.id => problems.push(ExportProblem {
                    severity: Severity::Error,
                    export: Some(export.id),
                    entry: None,
                    message: format!(
                        "writes “{name}” into the same folder as export {other} - give one of \
                         them a prefix or another folder"
                    ),
                }),
                _ => {
                    seen.insert((folder.clone(), name), export.id);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn project(exports: serde_json::Value) -> ProjectSettings {
        let mut project = ProjectSettings::default();
        project.exports = serde_json::from_value(exports).unwrap();
        project
    }

    fn messages(project: &ProjectSettings) -> Vec<(Severity, String)> {
        check_exports(project)
            .into_iter()
            .map(|p| (p.severity, p.message))
            .collect()
    }

    fn errors(project: &ProjectSettings) -> Vec<String> {
        messages(project)
            .into_iter()
            .filter(|(s, _)| *s == Severity::Error)
            .map(|(_, m)| m)
            .collect()
    }

    #[test]
    fn a_sound_export_has_no_problems() {
        let project = project(json!([{
            "id": 1, "name": "Per image", "prefix": "exp1",
            "entries": [
                { "kind": "TABLE", "rows": "OBJECT", "split": "IMAGE" },
                { "kind": "TABLE", "rows": "IMAGE", "name": "per_image" },
                { "kind": "DRAWN_IMAGES" },
                { "kind": "SCRIPT", "language": "R", "file": "scripts/a.R" }
            ]
        }]));
        assert_eq!(messages(&project), vec![]);
    }

    #[test]
    fn names_and_ids_are_unique_and_names_not_empty() {
        let project = project(json!([
            { "id": 1, "name": "Tables" },
            { "id": 1, "name": "tables " },
            { "id": 2, "name": " " }
        ]));
        let errors = errors(&project);
        assert!(errors.contains(&"another export has the id 1".to_string()));
        assert!(errors.contains(&"another export is called “tables”".to_string()));
        assert!(errors.contains(&"has no name".to_string()));
    }

    #[test]
    fn folders_stay_where_they_belong() {
        let up = project(json!([{ "id": 1, "name": "a",
            "output": { "location": "FIXED_FOLDER", "folder": "../outside" } }]));
        assert_eq!(errors(&up), ["the folder “../outside” can't contain “..”"]);

        let absolute = project(json!([{ "id": 1, "name": "a",
            "output": { "location": "FIXED_FOLDER", "folder": "/data/exports" } }]));
        assert_eq!(messages(&absolute)[0].0, Severity::Warning);

        let none = project(json!([{ "id": 1, "name": "a",
            "output": { "location": "FIXED_FOLDER", "folder": "" } }]));
        assert_eq!(errors(&none), ["a fixed folder needs its path"]);

        let results_up = project(json!([{ "id": 1, "name": "a",
            "output": { "location": "RESULTS_FOLDER", "folder": "../x" } }]));
        assert_eq!(errors(&results_up).len(), 1);
    }

    #[test]
    fn a_prefix_or_name_with_a_separator_is_refused() {
        let project = project(json!([{ "id": 1, "name": "a", "prefix": "a/b",
            "entries": [{ "kind": "TABLE", "rows": "OBJECT", "name": "x/y" }] }]));
        assert_eq!(errors(&project).len(), 2);
    }

    #[test]
    fn tables_are_checked_for_what_fits_together() {
        let project = project(json!([{ "id": 1, "name": "a", "entries": [
            { "kind": "TABLE", "rows": "IMAGE", "aggregations": [], "name": "a" },
            { "kind": "TABLE", "rows": "OBJECT", "splitInto": "SHEETS", "name": "b" },
            { "kind": "TABLE", "rows": "OBJECT", "tracks": {}, "name": "c" },
            { "kind": "TABLE", "rows": "WELL", "split": "IMAGE", "name": "d" },
            { "kind": "TABLE", "rows": "MEASUREMENT", "split": "CLASS", "name": "e" }
        ]}]));
        let problems = check_exports(&project);
        let entries: Vec<Option<usize>> = problems.iter().map(|p| p.entry).collect();
        assert_eq!(entries, [Some(0), Some(1), Some(2), Some(3), Some(4)]);
    }

    /// Object rows ignore the (defaulted) aggregations - no error.
    #[test]
    fn object_rows_keep_their_default_aggregation_without_error() {
        let project = project(json!([{ "id": 1, "name": "a",
            "entries": [{ "kind": "TABLE", "rows": "OBJECT" }] }]));
        assert_eq!(messages(&project), vec![]);
    }

    #[test]
    fn a_script_has_to_be_in_the_scripts_folder() {
        let project = project(json!([{ "id": 1, "name": "a", "entries": [
            { "kind": "SCRIPT", "language": "R", "file": "a.R" },
            { "kind": "SCRIPT", "language": "PYTHON", "file": "scripts/../../x.py", "output": "o" },
            { "kind": "SCRIPT", "language": "R", "file": "scripts/b.R", "output": "x/y" }
        ]}]));
        assert_eq!(errors(&project).len(), 3);
    }

    #[test]
    fn two_entries_writing_the_same_name_are_refused() {
        let project = project(json!([{ "id": 1, "name": "a", "entries": [
            { "kind": "TABLE", "rows": "OBJECT" },
            { "kind": "TABLE", "rows": "OBJECT", "classes": [2] }
        ]}]));
        assert_eq!(
            errors(&project),
            ["writes “objects” like entry 1 - give one of them its own name"]
        );
    }

    #[test]
    fn two_exports_writing_the_same_name_into_one_folder_are_refused() {
        let clash = project(json!([
            { "id": 1, "name": "a", "entries": [{ "kind": "TABLE", "rows": "OBJECT" }] },
            { "id": 2, "name": "b", "entries": [{ "kind": "TABLE", "rows": "OBJECT" }] }
        ]));
        assert_eq!(errors(&clash).len(), 1);

        let prefixed = project(json!([
            { "id": 1, "name": "a", "entries": [{ "kind": "TABLE", "rows": "OBJECT" }] },
            { "id": 2, "name": "b", "prefix": "b", "entries": [{ "kind": "TABLE", "rows": "OBJECT" }] }
        ]));
        assert_eq!(errors(&prefixed), Vec::<String>::new());
    }

    #[test]
    fn unknown_classes_and_columns_are_warnings() {
        let project = project(json!([{ "id": 1, "name": "a", "entries": [
            { "kind": "TABLE", "rows": "OBJECT", "classes": [42],
              "columns": ["count", "area_nm2", "measurement:threshold_x", "no_such_column"] }
        ]}]));
        let messages = messages(&project);
        assert_eq!(messages.len(), 2);
        assert!(messages.iter().all(|(s, _)| *s == Severity::Warning));
        assert!(messages[1].1.contains("no_such_column"));
        assert!(!messages[1].1.contains("count"));
    }

    #[test]
    fn an_unknown_table_format_is_refused() {
        let project = project(json!([{ "id": 1, "name": "a", "tableFormat": 2 }]));
        assert_eq!(errors(&project).len(), 1);
    }
}
