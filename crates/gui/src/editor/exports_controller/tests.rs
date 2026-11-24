// Copyright 2026 Joachim Danmayr.
// Licensed under the PolyForm Noncommercial License 1.0.0.

//! The Exports tab: edits land in the project (undoable), the tab shows
//! them, a run writes the files and shows its report.

use super::*;
use crate::editor::results_state_controller::tests::seed_test_db;
use crate::editor::test_support::{test_ui_window, ui_state_with_window};
use crate::helper::ui_thread::drain_ui_queue;
use evanalyzer_app::project::ProjectWithRuntime;
use slint::Model;

fn setup(project: ProjectWithRuntime) -> (AppWindow, Arc<UiState>, Arc<ExportsController>) {
    let ui = test_ui_window();
    let ui_state = ui_state_with_window(&ui, project);
    let controller = Arc::new(ExportsController::new(ui.as_weak(), ui_state.clone()));
    controller.attach_callbacks();
    (ui, ui_state, controller)
}

fn entries(ui: &AppWindow) -> Vec<ExportEntryRow> {
    ui.global::<ExportsState>().get_entries().iter().collect()
}

fn exports(ui_state: &UiState) -> Vec<ExportSettings> {
    ui_state.get_project().settings.exports.clone()
}

/// New: an export with a table, picked; its name stays unique.
#[test]
fn a_new_export_has_a_table_and_is_picked() {
    let (ui, ui_state, _controller) = setup(ProjectWithRuntime::default());
    let state = ui.global::<ExportsState>();
    state.invoke_new_export();
    state.invoke_new_export();
    let all = exports(&ui_state);
    assert_eq!(all.len(), 2);
    assert_eq!(all[0].name, "Export");
    assert_eq!(all[1].name, "Export 2");
    assert_ne!(all[0].id, all[1].id);
    assert!(matches!(
        all[1].entries.as_slice(),
        [ExportEntry::Table(TableExport {
            rows: TableRows::Image,
            ..
        })]
    ));
    assert_eq!(state.get_exports().row_count(), 2);
    assert_eq!(state.get_selected_id(), all[1].id as i32);
    assert_eq!(state.get_entry(), 0);
    assert_eq!(state.get_entry_kind(), "TABLE");
    assert!(entries(&ui)[0].title.contains("images.csv"));
}

/// Settings of the export and its entry go into the project; an undo
/// brings the old ones back, and the tab shows them.
#[test]
fn edits_land_in_the_project_and_undo_shows_the_old_ones() {
    let (ui, ui_state, _controller) = setup(ProjectWithRuntime::default());
    let state = ui.global::<ExportsState>();
    state.invoke_new_export();
    state.invoke_set_export_field("name".into(), "Per image".into());
    state.invoke_set_export_field("location".into(), "1".into());
    state.invoke_set_export_field("folder".into(), " exports ".into());
    state.invoke_set_export_field("length-unit".into(), "1".into());
    state.invoke_set_entry_field("rows".into(), "0".into());
    state.invoke_set_entry_field("format".into(), "1".into());
    state.invoke_set_entry_field("aggregation:median".into(), "1".into());
    state.invoke_set_entry_field("aggregation:avg".into(), "0".into());

    let export = &exports(&ui_state)[0];
    assert_eq!(export.name, "Per image");
    assert_eq!(export.output.location, OutputLocation::FixedFolder);
    assert_eq!(export.output.folder, "exports");
    assert_eq!(export.length_unit, LengthUnit::Nm);
    let ExportEntry::Table(table) = &export.entries[0] else {
        panic!("a table");
    };
    assert_eq!(table.rows, TableRows::Object);
    assert_eq!(table.format, TableFileFormat::Xlsx);
    assert_eq!(table.aggregations, vec![ExportAggregation::Median]);
    assert_eq!(state.get_name(), "Per image");
    assert_eq!(state.get_format(), 1);
    // An empty name keeps the old one.
    state.invoke_set_export_field("name".into(), "  ".into());
    assert_eq!(exports(&ui_state)[0].name, "Per image");

    // An undo takes back the last edit (here: avg unticked) - also on the
    // tab.
    assert!(ui_state.undo());
    drain_ui_queue();
    let ExportEntry::Table(table) = &exports(&ui_state)[0].entries[0] else {
        panic!("a table");
    };
    assert_eq!(
        table.aggregations,
        vec![ExportAggregation::Avg, ExportAggregation::Median]
    );
    let ticked: Vec<SharedString> = state
        .get_aggregations()
        .iter()
        .filter(|tick| tick.selected)
        .map(|tick| tick.key)
        .collect();
    assert_eq!(ticked, ["avg", "median"]);
}

fn project_with_classes(names: &[&str]) -> ProjectWithRuntime {
    use evanalyzer_cfg::settings::classification_settings::Class;
    let mut project = ProjectWithRuntime::default();
    for (i, name) in names.iter().enumerate() {
        project.settings.classification.classes_mut().push(Class {
            id: i as u32 + 1,
            name: name.to_string(),
            ..Default::default()
        });
    }
    project
}

/// Classes as the Results List ticks them: all ticked is every class
/// (`None`, also ones new in a run); None is none.
#[test]
fn all_classes_ticked_is_every_class() {
    let (ui, ui_state, _controller) = setup(project_with_classes(&["Nucleus", "Cell", "Spot"]));
    let state = ui.global::<ExportsState>();
    state.invoke_new_export();
    let classes = |ui_state: &UiState| match &exports(ui_state)[0].entries[0] {
        ExportEntry::Table(table) => table.classes.clone(),
        _ => panic!("a table"),
    };
    let ticked = |ui: &AppWindow| -> Vec<String> {
        ui.global::<ExportsState>()
            .get_classes()
            .iter()
            .filter(|item| item.selected)
            .map(|item| item.key.to_string())
            .collect()
    };
    assert_eq!(ticked(&ui), ["1", "2", "3"]);
    assert_eq!(state.get_classes_summary(), "all");
    state.invoke_set_entry_field("class:2".into(), "0".into());
    assert_eq!(classes(&ui_state), Some(vec![1, 3]));
    assert_eq!(state.get_classes_summary(), "2 of 3");
    state.invoke_set_entry_field("class:2".into(), "1".into());
    assert_eq!(classes(&ui_state), None);
    state.invoke_set_entry_field("no-classes".into(), "".into());
    assert_eq!(classes(&ui_state), Some(vec![]));
    assert!(ticked(&ui).is_empty());
    // It would write nothing: refused, and said.
    assert!(
        state.get_problems().contains("no class"),
        "{}",
        state.get_problems()
    );
    state.invoke_set_entry_field("all-classes".into(), "".into());
    assert_eq!(classes(&ui_state), None);
}

/// Entries move and go; the one picked stays picked.
#[test]
fn moving_and_removing_entries_keeps_the_pick() {
    let (ui, ui_state, _controller) = setup(ProjectWithRuntime::default());
    let state = ui.global::<ExportsState>();
    state.invoke_new_export();
    state.invoke_add_entry("DRAWN_IMAGES".into());
    state.invoke_add_entry("SCRIPT".into());
    assert_eq!(state.get_entry(), 2);
    assert_eq!(state.get_entry_kind(), "SCRIPT");
    state.invoke_move_entry(2, -1);
    assert_eq!(state.get_entry(), 1);
    assert_eq!(entries(&ui)[1].kind, "SCRIPT");
    // Past the ends: nothing.
    state.invoke_move_entry(0, -1);
    assert_eq!(entries(&ui)[0].kind, "TABLE");
    state.invoke_remove_entry(0);
    assert_eq!(state.get_entry(), 0);
    assert_eq!(state.get_entry_kind(), "SCRIPT");
    let kinds: Vec<&str> = exports(&ui_state)[0]
        .entries
        .iter()
        .map(kind_name)
        .collect();
    assert_eq!(kinds, ["SCRIPT", "DRAWN_IMAGES"]);
    // A script without a file can't run: said, and Run is off.
    assert!(state.get_problems().len() > 0);
}

/// A script picked by name: its language follows.
#[test]
fn a_python_script_is_python() {
    let (ui, ui_state, _controller) = setup(ProjectWithRuntime::default());
    let state = ui.global::<ExportsState>();
    state.invoke_new_export();
    state.invoke_add_entry("SCRIPT".into());
    state.invoke_set_entry_field("script-file".into(), "scripts/stats.py".into());
    let ExportEntry::Script(script) = &exports(&ui_state)[0].entries[1] else {
        panic!("a script");
    };
    assert_eq!(script.language, ScriptLanguage::Python);
    assert_eq!(state.get_language(), 1);
}

/// Duplicate: a copy that doesn't run after every run; delete takes it
/// away and the pick with it.
#[test]
fn duplicate_and_delete() {
    let (ui, ui_state, _controller) = setup(ProjectWithRuntime::default());
    let state = ui.global::<ExportsState>();
    state.invoke_new_export();
    let id = state.get_selected_id();
    state.invoke_set_after(id, true);
    state.invoke_duplicate(id);
    let all = exports(&ui_state);
    assert_eq!(all.len(), 2);
    assert_eq!(all[1].name, "Export copy");
    assert!(all[0].run_after_every_run);
    assert!(!all[1].run_after_every_run);
    assert_eq!(all[1].entries, all[0].entries);
    assert_eq!(state.get_selected_id(), all[1].id as i32);
    state.invoke_delete(all[1].id as i32);
    assert_eq!(exports(&ui_state).len(), 1);
    assert_eq!(state.get_selected_id(), -1);
}

/// Run on an unsaved project: said why it can't.
#[test]
fn a_run_needs_a_saved_project() {
    let (ui, _ui_state, _controller) = setup(ProjectWithRuntime::default());
    let state = ui.global::<ExportsState>();
    state.invoke_new_export();
    state.invoke_run();
    drain_ui_queue();
    assert!(!state.get_running());
    let report: Vec<SharedString> = state.get_report().iter().collect();
    assert!(report[0].contains("Save the project first"), "{report:?}");
    assert_eq!(state.get_exports().row_data(0).unwrap().status_tone, 2);
}

/// Run: on the newest analysis, into its folder; the report says what it
/// wrote and the list shows the run.
#[test]
fn a_run_writes_into_the_newest_analysis() {
    let dir = tempfile::tempdir().unwrap();
    let run = dir.path().join("results").join("20261124T090000Z__run");
    std::fs::create_dir_all(&run).unwrap();
    seed_test_db(&run.join("run.evadb"));
    let mut project = ProjectWithRuntime::default();
    project.tmp_settings.current_project = Some(dir.path().join("project.evaproj"));
    let (ui, _ui_state, _controller) = setup(project);
    let state = ui.global::<ExportsState>();
    state.invoke_new_export();
    state.invoke_set_entry_field("images".into(), "0".into());
    state.invoke_run();
    drain_ui_queue();

    assert!(!state.get_running());
    let report: Vec<String> = state.get_report().iter().map(|l| l.to_string()).collect();
    assert!(run.join("images.csv").is_file(), "{report:?}");
    assert!(report.iter().any(|l| l.contains("1 file")), "{report:?}");
    let row = state.get_exports().row_data(0).unwrap();
    assert_eq!(row.status_tone, 1, "{report:?}");
    assert!(row.status.starts_with('✓'));
}

/// A saved project with one analysis (seeded) in its `results/`.
fn saved_project_with_an_analysis(dir: &std::path::Path) -> ProjectWithRuntime {
    let run = dir.join("results").join("20261124T090000Z__run");
    std::fs::create_dir_all(&run).unwrap();
    seed_test_db(&run.join("run.evadb"));
    let mut project = ProjectWithRuntime::default();
    project.tmp_settings.current_project = Some(dir.join("project.evaproj"));
    project
}

fn table_columns(ui_state: &UiState) -> Option<Vec<String>> {
    match &exports(ui_state)[0].entries[0] {
        ExportEntry::Table(table) => table.columns.clone(),
        _ => panic!("a table"),
    }
}

/// Every column the picker shows ticked, of every section.
fn ticked_in_picker(ui: &AppWindow) -> Vec<String> {
    ui.global::<ExportsState>()
        .get_picker_sections()
        .iter()
        .flat_map(|section| section.items.iter().collect::<Vec<_>>())
        .filter(|item| item.selected)
        .map(|item| item.key.to_string())
        .collect()
}

/// The Results List's picker over the newest analysis: its sections,
/// starting sets and intensity grid; the default columns ticked until one
/// is picked - then they are the table's, plus the new one.
#[test]
fn columns_are_picked_with_the_results_column_picker() {
    let dir = tempfile::tempdir().unwrap();
    let (ui, ui_state, _controller) = setup(saved_project_with_an_analysis(dir.path()));
    let state = ui.global::<ExportsState>();
    state.invoke_new_export();
    state.invoke_refresh();
    drain_ui_queue();

    let offered: Vec<String> = state
        .get_picker_sections()
        .iter()
        .flat_map(|section| section.items.iter().collect::<Vec<_>>())
        .map(|item| item.key.to_string())
        .collect();
    assert!(offered.contains(&"area_px".to_string()), "{offered:?}");
    assert!(
        !offered.contains(&"object_id".to_string()),
        "no leading columns"
    );
    assert!(
        state.get_picker_presets().row_count() > 0,
        "the starting sets"
    );
    assert!(
        state.get_picker_intensity_cells().row_count() > 0,
        "the grid"
    );
    assert!(state.get_picker_grouped(), "image rows");
    assert!(state.get_columns_default());
    let defaults = ticked_in_picker(&ui);
    assert!(defaults.contains(&"count".to_string()), "{defaults:?}");

    state.invoke_picker("picked".into(), "area_px".into(), true);
    assert!(!state.get_columns_default());
    let columns = table_columns(&ui_state).unwrap();
    assert!(columns.contains(&"area_px".to_string()), "{columns:?}");
    for default in &defaults {
        assert!(columns.contains(default), "{default} stays: {columns:?}");
    }
    assert!(ticked_in_picker(&ui).contains(&"area_px".to_string()));

    // A whole channel of the intensity grid: on, then off again.
    let before = table_columns(&ui_state).unwrap().len();
    state.invoke_picker("intensity-channel".into(), "0".into(), true);
    let with_channel = table_columns(&ui_state).unwrap();
    assert!(with_channel.len() > before, "{with_channel:?}");
    state.invoke_picker("intensity-channel".into(), "0".into(), true);
    assert_eq!(table_columns(&ui_state).unwrap().len(), before);

    // A starting set replaces the columns.
    let preset = state.get_picker_presets().row_data(0).unwrap();
    state.invoke_picker("preset".into(), preset.key.clone(), true);
    assert!(
        state
            .get_picker_presets()
            .iter()
            .any(|p| p.key == preset.key && p.active),
        "the set is the one shown"
    );

    // The search narrows the sections, not the columns.
    let columns = table_columns(&ui_state);
    state.invoke_picker("filter".into(), "area".into(), true);
    assert!(
        state
            .get_picker_sections()
            .iter()
            .flat_map(|section| section.items.iter().collect::<Vec<_>>())
            .all(|item| item.value.to_lowercase().contains("area"))
    );
    assert_eq!(table_columns(&ui_state), columns);
    state.invoke_set_entry_field("clear-columns".into(), "".into());
    assert!(state.get_columns_default());
}

/// A key the newest analysis lacks stays in the table, and is said.
#[test]
fn keys_the_newest_analysis_lacks_are_kept() {
    let dir = tempfile::tempdir().unwrap();
    let (ui, ui_state, _controller) = setup(saved_project_with_an_analysis(dir.path()));
    let state = ui.global::<ExportsState>();
    state.invoke_new_export();
    {
        let mut project = ui_state.get_project_write();
        if let ExportEntry::Table(table) = &mut project.settings.exports[0].entries[0] {
            table.columns = Some(vec!["count".into(), "measurement:gone".into()]);
        }
    }
    state.invoke_refresh();
    drain_ui_queue();
    assert!(state.get_columns_note().contains("measurement:gone"));
    state.invoke_picker("picked".into(), "area_px".into(), true);
    let columns = table_columns(&ui_state).unwrap();
    assert!(
        columns.contains(&"measurement:gone".to_string()),
        "{columns:?}"
    );
    assert!(columns.contains(&"area_px".to_string()), "{columns:?}");
}

/// Empty classes and all / no aggregations, as the List's Values.
#[test]
fn empty_classes_and_values_like_the_list() {
    let (ui, ui_state, _controller) = setup(ProjectWithRuntime::default());
    let state = ui.global::<ExportsState>();
    state.invoke_new_export();
    state.invoke_set_entry_field("empty-classes".into(), "1".into());
    state.invoke_set_entry_field("all-aggregations".into(), "".into());
    let ExportEntry::Table(table) = &exports(&ui_state)[0].entries[0] else {
        panic!("a table");
    };
    assert!(table.empty_classes);
    assert_eq!(table.aggregations.len(), 7);
    assert!(state.get_empty_classes());
    state.invoke_set_entry_field("no-aggregations".into(), "".into());
    // Grouped rows without an aggregation can't run: said.
    assert!(state.get_problems().contains("aggregation"));
}

/// Side by side: on for image rows, gone when the rows can't have it.
#[test]
fn transposed_is_for_object_and_image_rows() {
    let (ui, ui_state, _controller) = setup(ProjectWithRuntime::default());
    let state = ui.global::<ExportsState>();
    state.invoke_new_export();
    state.invoke_set_entry_field("transposed".into(), "1".into());
    assert!(state.get_transposed());
    assert!(entries(&ui)[0].subtitle.contains("side by side"));
    let transposed = |ui_state: &UiState| match &exports(ui_state)[0].entries[0] {
        ExportEntry::Table(table) => table.transposed,
        _ => panic!("a table"),
    };
    assert!(transposed(&ui_state));
    // Well rows.
    state.invoke_set_entry_field("rows".into(), "2".into());
    assert!(!transposed(&ui_state));
    assert!(!state.get_transposed());
}

/// The tab on screen at the smallest window: a table picked shows its
/// settings to the right, the List's panel at least as wide as in Results.
#[test]
fn the_entry_settings_show_beside_the_export() {
    use crate::{LayoutState, StartState, WorkspaceMode};
    use i_slint_backend_testing::ElementHandle;

    let (ui, _ui_state, _controller) = setup(ProjectWithRuntime::default());
    ui.window().set_size(slint::LogicalSize::new(1280.0, 680.0));
    ui.global::<StartState>().set_active(false);
    ui.global::<LayoutState>().set_mode(WorkspaceMode::Exports);
    let state = ui.global::<ExportsState>();
    state.invoke_new_export();

    // The panel's part switch ("Show": Table, Columns, ...).
    let switch = ElementHandle::find_by_accessible_label(&ui, "Show")
        .find(|e| e.size().width > 0.0)
        .expect("the entry's panel is on screen");
    assert!(switch.size().width > 300.0, "{:?}", switch.size());
    let panel_x = switch.absolute_position().x;
    assert!(panel_x > 300.0 + 520.0, "right of the export: {panel_x}");
    assert!(panel_x + switch.size().width <= 1280.0, "inside the window");
}

/// Table | Columns | Values & classes - no Images part (its line is in
/// Table); the values, then the classes below them; a row per object has
/// "Classes" alone. The panel keeps its width in a wide window.
#[test]
fn the_table_panel_has_values_then_classes() {
    use crate::{LayoutState, StartState, WorkspaceMode};
    use i_slint_backend_testing::ElementHandle;

    let (ui, _ui_state, _controller) = setup(project_with_classes(&["Nucleus", "Cell"]));
    ui.window().set_size(slint::LogicalSize::new(2400.0, 900.0));
    ui.global::<StartState>().set_active(false);
    ui.global::<LayoutState>().set_mode(WorkspaceMode::Exports);
    let state = ui.global::<ExportsState>();
    state.invoke_new_export();
    let shown = |label: &str| {
        ElementHandle::find_by_accessible_label(&ui, label).find(|e| e.size().width > 0.0)
    };
    // The images line in Table.
    assert!(shown("Images of the run").is_some());
    // A readable width, not the window's.
    let switch = shown("Show").expect("the panel's parts");
    assert!(switch.size().width < 460.0, "{:?}", switch.size());

    state.set_panel_part("Values & classes".into());
    let classes = shown("Nucleus").expect("the classes");
    let values = shown("Mean").expect("the values");
    assert!(
        classes.absolute_position().y > values.absolute_position().y + 100.0,
        "values, then classes: {:?} {:?}",
        values.absolute_position(),
        classes.absolute_position()
    );

    // A row per object: Classes alone, still open.
    state.invoke_set_entry_field("rows".into(), "0".into());
    assert!(shown("Nucleus").is_some(), "still on the classes");
    assert!(shown("Mean").is_none(), "no values for objects");
}
