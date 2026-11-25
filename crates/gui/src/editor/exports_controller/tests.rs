// Copyright 2026 Joachim Danmayr.
// Licensed under the PolyForm Noncommercial License 1.0.0.

//! The Exports tab: edits land in the project (undoable), the tab shows
//! them, a run writes the files and shows its report.

use super::*;
use crate::editor::results_state_controller::tests::seed_test_db;
use crate::editor::test_support::{test_ui_window, ui_state_with_window};
use crate::helper::ui_thread::drain_ui_queue;
use evanalyzer_app::project::ProjectWithRuntime;
use evanalyzer_cfg::settings::export_settings::DataExport;
use slint::Model;

fn setup(project: ProjectWithRuntime) -> (AppWindow, Arc<UiState>, Arc<ExportsController>) {
    let ui = test_ui_window();
    let ui_state = ui_state_with_window(&ui, project);
    let controller = Arc::new(ExportsController::new(ui.as_weak(), ui_state.clone()));
    controller.attach_callbacks();
    (ui, ui_state, controller)
}

/// Ticks `key` alone among a data export's outputs.
fn only(state: &ExportsState, key: &str) {
    for row in state.get_outputs().iter() {
        if row.key.as_str() != key && row.on {
            state.invoke_set_entry_field(format!("output:{}", row.key).into(), "0".into());
        }
    }
    state.invoke_set_entry_field(format!("output:{key}").into(), "1".into());
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
        [ExportEntry::Data(DataExport { outputs, .. })] if outputs == &[DataOutput::Images]
    ));
    assert_eq!(state.get_exports().row_count(), 2);
    assert_eq!(state.get_selected_id(), all[1].id as i32);
    assert_eq!(state.get_entry(), 0);
    assert_eq!(state.get_entry_kind(), "DATA");
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
    only(&state, "OBJECTS");
    state.invoke_set_entry_field("format".into(), "1".into());
    state.invoke_set_entry_field("aggregation:median".into(), "1".into());
    state.invoke_set_entry_field("aggregation:avg".into(), "0".into());

    let export = &exports(&ui_state)[0];
    assert_eq!(export.name, "Per image");
    assert_eq!(export.output.location, OutputLocation::FixedFolder);
    assert_eq!(export.output.folder, "exports");
    assert_eq!(export.length_unit, LengthUnit::Nm);
    let ExportEntry::Data(table) = &export.entries[0] else {
        panic!("a table");
    };
    assert_eq!(table.outputs, vec![DataOutput::Objects]);
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
    let ExportEntry::Data(table) = &exports(&ui_state)[0].entries[0] else {
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
        ExportEntry::Data(table) => table.classes.clone(),
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
    assert_eq!(entries(&ui)[0].kind, "DATA");
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

/// Duplicate: a copy that doesn't run after analysis; delete takes it
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
    assert!(all[0].run_after_analysis);
    assert!(!all[1].run_after_analysis);
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
    let report: Vec<String> = report_of(&state);
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
    let report: Vec<String> = report_of(&state);
    assert!(run.join("images.csv").is_file(), "{report:?}");
    assert!(report.iter().any(|l| l.contains("1 file")), "{report:?}");
    let row = state.get_exports().row_data(0).unwrap();
    assert_eq!(row.status_tone, 1, "{report:?}");
    assert!(row.status.starts_with('✓'));
}

/// The last run's report lines, of every analysis.
fn report_of(state: &ExportsState) -> Vec<String> {
    state
        .get_run_parts()
        .iter()
        .flat_map(|part| part.lines.iter().map(|l| l.to_string()).collect::<Vec<_>>())
        .collect()
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
        ExportEntry::Data(table) => table.columns.clone(),
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
    // Image, class and object id: always written - ticked and greyed;
    // the track id to pick.
    let locked: Vec<String> = state
        .get_picker_sections()
        .iter()
        .flat_map(|section| {
            let locked: Vec<bool> = section.locked.iter().collect();
            section
                .items
                .iter()
                .enumerate()
                .filter(|(i, _)| locked.get(*i).copied().unwrap_or(false))
                .map(|(_, item)| item.key.to_string())
                .collect::<Vec<_>>()
        })
        .collect();
    for key in ["image_name", "object_class", "object_id"] {
        let key = evanalyzer_app::results::Column::from_key(key, &[])
            .map(|column| column.as_key(&[]))
            .unwrap_or_else(|| key.to_string());
        if offered.contains(&key) {
            assert!(locked.contains(&key), "{key} greyed: {locked:?}");
            assert!(ticked_in_picker(&ui).contains(&key), "{key} ticked");
        }
    }
    assert!(offered.contains(&"object_id".to_string()), "{offered:?}");
    assert!(offered.contains(&"track_id".to_string()), "{offered:?}");
    assert!(
        !locked.contains(&"track_id".to_string()),
        "track id to pick"
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
    for default in defaults.iter().filter(|key| !locked.contains(key)) {
        assert!(columns.contains(default), "{default} stays: {columns:?}");
    }
    assert!(
        !columns.iter().any(|key| locked.contains(key)),
        "the greyed ones aren't stored: {columns:?}"
    );
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
        if let ExportEntry::Data(table) = &mut project.settings.exports[0].entries[0] {
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
    let ExportEntry::Data(table) = &exports(&ui_state)[0].entries[0] else {
        panic!("a table");
    };
    assert!(table.empty_classes);
    assert_eq!(table.aggregations.len(), 7);
    assert!(state.get_empty_classes());
    state.invoke_set_entry_field("no-aggregations".into(), "".into());
    // Grouped rows without an aggregation can't run: said.
    assert!(state.get_problems().contains("aggregation"));
}

/// Side by side: for objects, images, wells and folders.
#[test]
fn transposed_is_for_the_row_per_tables() {
    let (ui, ui_state, _controller) = setup(ProjectWithRuntime::default());
    let state = ui.global::<ExportsState>();
    state.invoke_new_export();
    state.invoke_set_entry_field("transposed".into(), "1".into());
    assert!(state.get_transposed());
    assert!(entries(&ui)[0].subtitle.contains("side by side"));
    let transposed = |ui_state: &UiState| match &exports(ui_state)[0].entries[0] {
        ExportEntry::Data(table) => table.transposed,
        _ => panic!("a table"),
    };
    assert!(transposed(&ui_state));
    only(&state, "WELLS");
    assert!(state.get_can_transpose());
    assert!(entries(&ui)[0].subtitle.contains("side by side"));
    // Only coloc pairs: nothing to transpose - the setting stays, not
    // offered, not on the card.
    only(&state, "COLOC_PAIRS");
    assert!(transposed(&ui_state));
    assert!(!state.get_can_transpose());
    assert!(!entries(&ui)[0].subtitle.contains("side by side"));
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
    only(&state, "OBJECTS");
    assert!(shown("Nucleus").is_some(), "still on the classes");
    assert!(shown("Mean").is_none(), "no values for objects");
}

/// After an analysis: the exports left on in the run dialog run on its
/// results - into its folder - as its last step: its progress says
/// "Exporting" meanwhile, its note says what they wrote; no note of their
/// own.
#[test]
fn the_exports_run_after_an_analysis() {
    let dir = tempfile::tempdir().unwrap();
    let project = saved_project_with_an_analysis(dir.path());
    let (ui, ui_state, _controller) = setup(project);
    let state = ui.global::<ExportsState>();
    let running = ui.global::<crate::PipelineRunningState>();
    state.invoke_new_export(); // "Export": images.csv, not after analysis
    state.invoke_set_entry_field("images".into(), "0".into());
    let run = dir.path().join("results").join("20261124T090000Z__run");

    // Not set to run after analysis, nothing left on: nothing runs.
    ui_state.analysis_finished(&run);
    drain_ui_queue();
    assert!(!running.get_exporting());
    assert_eq!(running.get_export_note(), "");
    assert!(!run.join("images.csv").exists());

    // Left on in the run dialog: the analysis' last step.
    let id = state.get_selected_id() as u32;
    ui_state.set_exports_for_run(Some(vec![id]));
    ui_state.analysis_finished(&run);
    assert!(!running.get_exporting(), "set on the UI thread");
    drain_ui_queue();
    assert!(run.join("images.csv").is_file());
    assert!(!running.get_exporting(), "done");
    assert!(!state.get_running(), "the tab unlocked");
    assert!(
        running.get_export_note().contains("1 file"),
        "{}",
        running.get_export_note()
    );
    assert!(!running.get_export_problem());
    assert!(!state.get_after_shown(), "no note of their own");
    // Its report in the tab, its status in the list.
    assert!(report_of(&state).iter().any(|l| l.contains("1 file")));
    assert_eq!(state.get_exports().row_data(0).unwrap().status_tone, 1);

    // Set to run after analysis (no run dialog): runs too.
    std::fs::remove_file(run.join("images.csv")).unwrap();
    state.invoke_set_after(id as i32, true);
    ui_state.analysis_finished(&run);
    drain_ui_queue();
    assert!(run.join("images.csv").is_file());
}

/// A problem: the analysis' note says so (and stays), "Open in Exports"
/// picks that export.
#[test]
fn a_problem_after_an_analysis_opens_that_export() {
    let dir = tempfile::tempdir().unwrap();
    let project = saved_project_with_an_analysis(dir.path());
    let (ui, ui_state, _controller) = setup(project);
    let state = ui.global::<ExportsState>();
    state.invoke_new_export();
    let id = state.get_selected_id();
    state.invoke_set_after(id, true);
    let run = dir.path().join("results").join("20261124T090000Z__run");
    // Where the table should go, a folder: it can't be written.
    std::fs::create_dir_all(run.join("images.csv")).unwrap();
    state.invoke_select(-1);

    ui_state.analysis_finished(&run);
    drain_ui_queue();
    let running = ui.global::<crate::PipelineRunningState>();
    assert!(running.get_export_problem());
    assert!(
        running.get_export_note().contains("problem"),
        "{}",
        running.get_export_note()
    );
    state.invoke_after_open();
    assert_eq!(state.get_selected_id(), id);
    assert!(report_of(&state).iter().any(|l| l.contains("failed")));
}

/// A saved project with an analysis and an export running the script
/// `scripts/stats.py` - picked in the tab.
fn project_with_a_script(
    dir: &std::path::Path,
) -> (AppWindow, Arc<UiState>, Arc<ExportsController>) {
    std::fs::create_dir_all(dir.join("scripts")).unwrap();
    std::fs::write(dir.join("scripts/stats.py"), "print('stats')\n").unwrap();
    let (ui, ui_state, controller) = setup(saved_project_with_an_analysis(dir));
    let state = ui.global::<ExportsState>();
    state.invoke_new_export();
    state.invoke_remove_entry(0);
    state.invoke_add_entry("SCRIPT".into());
    state.invoke_set_entry_field("script-file".into(), "scripts/stats.py".into());
    (ui, ui_state, controller)
}

fn trusted(ui_state: &UiState) -> Vec<String> {
    ui_state
        .app_settings
        .lock()
        .unwrap()
        .trusted_scripts
        .clone()
}

/// A script not allowed yet: asked before the run. Don't run skips it (and
/// asks again next time), always allow keeps its content's hash - not asked
/// again; run once allows it for that run only.
#[test]
fn a_script_not_allowed_yet_is_asked_about_before_the_run() {
    use crate::ScriptTrustState;
    let dir = tempfile::tempdir().unwrap();
    let (ui, ui_state, _controller) = project_with_a_script(dir.path());
    let state = ui.global::<ExportsState>();
    let trust = ui.global::<ScriptTrustState>();

    state.invoke_run();
    drain_ui_queue();
    assert!(trust.get_shown(), "asked first");
    assert_eq!(trust.get_name(), "stats.py");
    assert!(trust.get_preview().contains("print('stats')"));
    assert!(trust.get_about().contains("1 line"));
    assert!(!state.get_running(), "nothing runs while asked");

    trust.invoke_answer(0);
    drain_ui_queue();
    assert!(!trust.get_shown());
    let report: Vec<String> = report_of(&state);
    assert!(
        report
            .iter()
            .any(|l| l
                .contains("not run - you chose \"Don't run\". Run the export again to be asked.")),
        "{report:?}"
    );
    assert!(trusted(&ui_state).is_empty());

    // Asked again; run once: allowed for this run, not kept.
    state.invoke_run();
    drain_ui_queue();
    assert!(trust.get_shown());
    trust.invoke_answer(1);
    drain_ui_queue();
    let report: Vec<String> = report_of(&state);
    assert!(!report.iter().any(|l| l.contains("not run")), "{report:?}");
    assert!(trusted(&ui_state).is_empty());

    // Always allow this version: kept, not asked again.
    state.invoke_run();
    drain_ui_queue();
    trust.invoke_answer(2);
    drain_ui_queue();
    assert_eq!(trusted(&ui_state).len(), 1);
    state.invoke_run();
    drain_ui_queue();
    assert!(!trust.get_shown(), "this version is allowed");

    // A changed script is another version: asked again.
    std::fs::write(dir.path().join("scripts/stats.py"), "print('changed')\n").unwrap();
    state.invoke_run();
    drain_ui_queue();
    assert!(trust.get_shown());
    trust.invoke_answer(0);
    drain_ui_queue();
}

/// After an analysis: the exports wait for the answer, then run.
#[test]
fn the_exports_after_an_analysis_wait_for_the_scripts_answer() {
    use crate::ScriptTrustState;
    let dir = tempfile::tempdir().unwrap();
    let (ui, ui_state, _controller) = project_with_a_script(dir.path());
    let state = ui.global::<ExportsState>();
    let trust = ui.global::<ScriptTrustState>();
    let id = state.get_selected_id();
    state.invoke_set_after(id, true);
    let run = dir.path().join("results").join("20261124T090000Z__run");

    let running = ui.global::<crate::PipelineRunningState>();
    ui_state.analysis_finished(&run);
    drain_ui_queue();
    assert!(trust.get_shown());
    assert!(running.get_exporting(), "the analysis' last step waits");
    assert_eq!(running.get_export_note(), "", "nothing ran yet");
    trust.invoke_answer(0);
    drain_ui_queue();
    assert!(!running.get_exporting());
    assert_ne!(running.get_export_note(), "");
}

/// The analysis of `saved_project_with_an_analysis` open in Results.
fn open_the_analysis(ui: &AppWindow, dir: &std::path::Path) -> std::path::PathBuf {
    let db = dir.join("results/20261124T090000Z__run/run.evadb");
    let list = ui.global::<crate::ResultsListState>();
    list.set_results_list(model(vec![crate::ResultItemData {
        name: "run".into(),
        path: db.to_string_lossy().as_ref().into(),
        ..Default::default()
    }]));
    list.set_selected_result_index(0);
    db
}

/// Results › Export… on the List: written once into the folder chosen, as
/// the view shows it; the done step says what was written; nothing kept.
#[test]
fn export_this_view_writes_the_list_once() {
    use crate::QuickExportState;
    let dir = tempfile::tempdir().unwrap();
    let (ui, ui_state, _controller) = setup(saved_project_with_an_analysis(dir.path()));
    open_the_analysis(&ui, dir.path());
    let quick = ui.global::<QuickExportState>();
    quick.invoke_open(-1);
    assert!(quick.get_shown());
    assert!(!quick.get_done());
    assert_eq!(quick.get_view_name(), "List");
    assert!(quick.get_table());
    assert!(quick.get_what().contains("columns"), "{}", quick.get_what());
    assert!(
        quick.get_folder().ends_with("20261124T090000Z__run"),
        "beside the analysis"
    );
    assert!(quick.get_can_open_folder(), "on this computer");
    assert_eq!(quick.get_error(), "");

    let out = dir.path().join("out");
    std::fs::create_dir_all(&out).unwrap();
    quick.set_folder(out.to_string_lossy().as_ref().into());
    quick.set_file("for_r".into());
    quick.set_format(1);
    quick.invoke_export();
    drain_ui_queue();
    assert_eq!(quick.get_error(), "");
    assert!(out.join("for_r_objects.xlsx").is_file());
    // The done step: what, where; Done closes.
    assert!(quick.get_shown());
    assert!(quick.get_done());
    assert_eq!(quick.get_done_text(), "Written: for_r_objects.xlsx");
    assert!(
        quick.get_done_path().ends_with("for_r_objects.xlsx"),
        "{}",
        quick.get_done_path()
    );
    quick.invoke_cancel();
    assert!(!quick.get_shown());
    assert!(exports(&ui_state).is_empty(), "nothing kept");

    // Again: the file there stays, the new one gets "_2".
    quick.invoke_open(-1);
    assert!(!quick.get_done(), "the form again");
    quick.set_file("for_r".into());
    quick.set_format(1);
    quick.invoke_export();
    drain_ui_queue();
    assert!(out.join("for_r_objects_2.xlsx").is_file());
}

/// Add to project exports: the view as a new export (named like the view,
/// its name and format as set) - nothing written; the note opens it. From
/// the done step too: exported and kept.
#[test]
fn export_this_view_adds_it_to_the_project() {
    use crate::QuickExportState;
    let dir = tempfile::tempdir().unwrap();
    let (ui, ui_state, _controller) = setup(saved_project_with_an_analysis(dir.path()));
    open_the_analysis(&ui, dir.path());
    let quick = ui.global::<QuickExportState>();
    let state = ui.global::<ExportsState>();
    let out = dir.path().join("out");
    std::fs::create_dir_all(&out).unwrap();

    // The Plate view: added, nothing written.
    quick.invoke_open(2);
    assert!(!quick.get_table(), "a plate layout");
    quick.set_folder(out.to_string_lossy().as_ref().into());
    quick.set_file("ctrl".into());
    quick.invoke_add_to_project();
    drain_ui_queue();
    assert!(!quick.get_shown());
    assert_eq!(
        std::fs::read_dir(&out).unwrap().count(),
        0,
        "nothing written"
    );
    let all = exports(&ui_state);
    assert_eq!(all.len(), 1);
    assert_eq!(all[0].name, "Plate");
    assert!(!all[0].run_after_analysis);
    match &all[0].entries[0] {
        ExportEntry::Data(data) => assert_eq!(data.name, "ctrl"),
        _ => panic!("a data export"),
    }
    assert!(state.get_after_shown());
    assert!(
        state
            .get_after_text()
            .contains("Added to Exports as “Plate”"),
        "{}",
        state.get_after_text()
    );
    state.invoke_after_open();
    assert_eq!(state.get_selected_id(), all[0].id as i32);

    // The List: exported, then added from the done step.
    quick.invoke_open(1);
    quick.set_folder(out.to_string_lossy().as_ref().into());
    quick.set_format(1);
    quick.invoke_export();
    drain_ui_queue();
    assert!(quick.get_done());
    quick.invoke_add_to_project();
    drain_ui_queue();
    assert!(!quick.get_shown());
    let all = exports(&ui_state);
    assert_eq!(all.len(), 2);
    assert_eq!(all[1].name, "List");
    match &all[1].entries[0] {
        ExportEntry::Data(data) => assert_eq!(data.format, TableFileFormat::Xlsx),
        _ => panic!("a data export"),
    }
}

/// Run ▾: the Results list's analyses, the newest ticked; run on the ones
/// ticked, each into its own folder - a part of the report and a folder to
/// open per analysis. Run alone: the newest, its folder beside "Last run".
#[test]
fn an_export_runs_on_the_analyses_picked() {
    let dir = tempfile::tempdir().unwrap();
    let (ui, _ui_state, _controller) = setup(saved_project_with_an_analysis(dir.path()));
    let older = dir.path().join("results").join("20261120T080000Z__run");
    std::fs::create_dir_all(&older).unwrap();
    seed_test_db(&older.join("run.evadb"));
    let newer = dir.path().join("results/20261124T090000Z__run");
    let list = ui.global::<crate::ResultsListState>();
    list.set_results_list(model(vec![
        crate::ResultItemData {
            name: "run".into(),
            path: newer.join("run.evadb").to_string_lossy().as_ref().into(),
            modified: "24 Nov 09:00".into(),
            ..Default::default()
        },
        crate::ResultItemData {
            name: "run".into(),
            path: older.join("run.evadb").to_string_lossy().as_ref().into(),
            modified: "20 Nov 08:00".into(),
            ..Default::default()
        },
    ]));
    let state = ui.global::<ExportsState>();
    state.invoke_new_export();
    state.invoke_set_entry_field("images".into(), "0".into());

    // Run alone: the newest, named in the title; its folder beside it,
    // no "Into …" line.
    state.invoke_run();
    drain_ui_queue();
    assert!(newer.join("images.csv").is_file());
    assert!(!older.join("images.csv").exists());
    assert!(
        state.get_last_run_title().ends_with(" · run"),
        "{}",
        state.get_last_run_title()
    );
    assert_eq!(state.get_last_run_note(), "");
    let parts: Vec<_> = state.get_run_parts().iter().collect();
    assert_eq!(parts.len(), 1);
    assert!(parts[0].ok && parts[0].open);
    assert_eq!(std::path::PathBuf::from(parts[0].folder.as_str()), newer);
    assert!(!report_of(&state).iter().any(|l| l.starts_with("Into ")));
    std::fs::remove_file(newer.join("images.csv")).unwrap();

    // The ▾: both listed, the newest ticked.
    state.invoke_run_on_open();
    let rows: Vec<_> = state.get_run_on().iter().collect();
    assert_eq!(rows.len(), 2);
    assert!(rows[0].on && !rows[1].on);
    assert_eq!(state.get_run_on_ticked(), 1);
    state.invoke_run_on_pick(-1, true);
    assert_eq!(state.get_run_on_ticked(), 2, "All");
    state.invoke_run_on_pick(-1, false);
    assert_eq!(state.get_run_on_ticked(), 1, "Newest only");
    state.invoke_run_on_pick(0, false);
    state.invoke_run_on_pick(1, true);
    state.invoke_run_on_picked();
    drain_ui_queue();
    assert!(older.join("images.csv").is_file(), "the older alone");
    assert!(!newer.join("images.csv").exists());

    // Both: a row per analysis - name, age, what it wrote, its folder;
    // folded while all went well, a click unfolds it.
    state.invoke_run_on_open();
    state.invoke_run_on_pick(-1, true);
    state.invoke_run_on_picked();
    drain_ui_queue();
    assert!(newer.join("images.csv").is_file());
    assert!(
        state.get_last_run_title().ends_with(" · 2 analyses"),
        "{}",
        state.get_last_run_title()
    );
    assert_eq!(state.get_last_run_note(), "all written");
    let parts: Vec<_> = state.get_run_parts().iter().collect();
    assert_eq!(parts.len(), 2);
    assert_eq!(parts[0].name, "run");
    assert_eq!(parts[0].when, "24 Nov 09:00");
    assert_eq!(parts[0].summary, "1 file");
    assert!(parts[0].ok && !parts[0].open, "folded");
    assert_eq!(std::path::PathBuf::from(parts[1].folder.as_str()), older);
    state.invoke_run_part_toggle(1);
    assert!(state.get_run_parts().row_data(1).unwrap().open);
    let row = state.get_exports().row_data(0).unwrap();
    assert_eq!(row.status_tone, 1);
}

/// A view's ⋯ › Save as export…: a new export of it, opened in the tab -
/// nothing written; a chart's is the table behind it.
#[test]
fn save_a_view_as_export() {
    use crate::{LayoutState, QuickExportState, WorkspaceMode};
    let dir = tempfile::tempdir().unwrap();
    let (ui, ui_state, _controller) = setup(saved_project_with_an_analysis(dir.path()));
    open_the_analysis(&ui, dir.path());
    let quick = ui.global::<QuickExportState>();
    quick.invoke_save_view(4);
    drain_ui_queue();
    let all = exports(&ui_state);
    assert_eq!(all.len(), 1);
    assert_eq!(all[0].name, "Tracks");
    match &all[0].entries[0] {
        ExportEntry::Data(table) => assert_eq!(table.outputs, vec![DataOutput::Tracks]),
        _ => panic!("a table of tracks"),
    }
    assert_eq!(
        ui.global::<LayoutState>().get_mode(),
        WorkspaceMode::Exports
    );

    // The Charts view: the table behind its chart.
    quick.invoke_save_view(3);
    drain_ui_queue();
    let all = exports(&ui_state);
    assert_eq!(all.len(), 2);
    assert!(matches!(all[1].entries[0], ExportEntry::Data(_)));
    quick.invoke_open(3);
    assert!(quick.get_shown());
}

/// At the smallest window the new dialogs show whole: their buttons inside.
#[test]
fn the_export_dialogs_fit_the_smallest_window() {
    use crate::{LayoutState, QuickExportState, ScriptTrustState, StartState, WorkspaceMode};
    use i_slint_backend_testing::ElementHandle;
    let dir = tempfile::tempdir().unwrap();
    let (ui, _ui_state, _controller) = project_with_a_script(dir.path());
    ui.window().set_size(slint::LogicalSize::new(1280.0, 680.0));
    ui.global::<StartState>().set_active(false);
    ui.global::<LayoutState>().set_mode(WorkspaceMode::Results);
    open_the_analysis(&ui, dir.path());
    let inside = |label: &str| {
        let e = ElementHandle::find_by_accessible_label(&ui, label)
            .find(|e| e.size().width > 0.0)
            .unwrap_or_else(|| panic!("{label} shown"));
        let (p, s) = (e.absolute_position(), e.size());
        assert!(
            p.x >= 0.0 && p.y >= 0.0 && p.x + s.width <= 1280.0 && p.y + s.height <= 680.0,
            "{label} inside: {p:?} {s:?}"
        );
    };

    let quick = ui.global::<QuickExportState>();
    quick.invoke_open(1);
    inside("Export");
    inside("Add to project exports");
    quick.invoke_cancel();

    ui.global::<ExportsState>().invoke_run();
    drain_ui_queue();
    assert!(ui.global::<ScriptTrustState>().get_shown());
    inside("Always allow this version");
    inside("Don't run");
    inside("Run once");
    ui.global::<ScriptTrustState>().invoke_answer(0);
    drain_ui_queue();
}

/// Children and linked objects: the Results List's lists, from the newest
/// analysis' links; ticked, kept as links on the table.
#[test]
fn children_and_linked_objects_are_picked_like_the_list() {
    let dir = tempfile::tempdir().unwrap();
    let project = saved_project_with_an_analysis(dir.path());
    // ClassB objects inside the first ClassA object.
    {
        let db = dir.path().join("results/20261124T090000Z__run/run.evadb");
        let conn = duckdb::Connection::open(db).unwrap();
        conn.execute_batch(
            "UPDATE objects SET parent_ids = MAP {1: (SELECT object_id FROM objects \
             WHERE list_contains(object_class_id, 1) ORDER BY object_id LIMIT 1)} \
             WHERE list_contains(object_class_id, 2);",
        )
        .unwrap();
    }
    let (ui, ui_state, _controller) = setup(project);
    let state = ui.global::<ExportsState>();
    state.invoke_new_export();
    only(&state, "OBJECTS");
    state.invoke_refresh();
    drain_ui_queue();

    let children: Vec<String> = state
        .get_related_children()
        .iter()
        .map(|i| i.key.to_string())
        .collect();
    assert_eq!(children, ["1:2"], "the link of the analysis");
    assert_eq!(state.get_related_parents().row_count(), 1);
    state.invoke_set_entry_field("child:1:2".into(), "1".into());
    state.invoke_set_entry_field("child-aggregation:Maximum".into(), "1".into());
    state.invoke_set_entry_field("parents-all".into(), "".into());
    let ExportEntry::Data(table) = &exports(&ui_state)[0].entries[0] else {
        panic!("a table");
    };
    assert_eq!(table.children, vec![(1, 2)]);
    assert_eq!(table.child_aggregations, vec![ExportAggregation::Max]);
    assert_eq!(table.parents, vec![1]);
    assert!(state.get_related_children().row_data(0).unwrap().selected);
    assert!(state.get_related_children_summary().starts_with("1 of 1"));

    // Written with their columns.
    state.invoke_set_entry_field("images".into(), "0".into());
    state.invoke_run();
    drain_ui_queue();
    let csv = std::fs::read_to_string(dir.path().join("results/20261124T090000Z__run/objects.csv"))
        .unwrap();
    let header = csv.lines().next().unwrap();
    assert!(header.contains("children_1_2_count"), "{header}");
    assert!(header.contains("parent_1_id"), "{header}");
}

/// A time course: its options with the rows; a profile: its options; the
/// options go when the rows change.
#[test]
fn time_course_and_profile_options() {
    let (ui, ui_state, _controller) = setup(ProjectWithRuntime::default());
    let state = ui.global::<ExportsState>();
    state.invoke_new_export();
    let table = |ui_state: &UiState| match &exports(ui_state)[0].entries[0] {
        ExportEntry::Data(table) => table.clone(),
        _ => panic!("a table"),
    };
    only(&state, "TIME_COURSE");
    state.invoke_set_entry_field("time-lines".into(), "3".into());
    state.invoke_set_entry_field("time-layout".into(), "0".into());
    state.invoke_set_entry_field("time-with-parents".into(), "1".into());
    let course = table(&ui_state).time_course.unwrap();
    assert_eq!(course.lines, TimeCourseLineKind::Track);
    assert!(course.with_parents);
    assert_eq!(state.get_time_lines(), 3);
    assert!(
        entries(&ui)[0].title.contains("time_course.csv"),
        "{}",
        entries(&ui)[0].title
    );

    only(&state, "PROFILE");
    // An output's options stay while it is unticked (ticked again: as set).
    assert_eq!(
        table(&ui_state).time_course.unwrap().lines,
        TimeCourseLineKind::Track
    );
    state.invoke_set_entry_field("profile-objects".into(), "1".into());
    assert_eq!(
        table(&ui_state).profile.unwrap().content,
        ProfileContent::Objects
    );
    // No profile picked yet: it can't run.
    assert!(state.get_problems().contains("profile measurement"));
}

/// Z-planes and time points: all, or a range.
#[test]
fn plane_ranges() {
    let (ui, ui_state, _controller) = setup(ProjectWithRuntime::default());
    let state = ui.global::<ExportsState>();
    state.invoke_new_export();
    state.invoke_set_entry_field("t-planes".into(), "4-2".into());
    let planes = |ui_state: &UiState| match &exports(ui_state)[0].entries[0] {
        ExportEntry::Data(table) => table.planes,
        _ => panic!("a table"),
    };
    assert_eq!(planes(&ui_state).t, Some([2, 4]), "from the smaller");
    assert!(state.get_t_range());
    assert_eq!(state.get_t_from(), "2");
    state.invoke_set_entry_field("t-planes".into(), "all".into());
    assert_eq!(planes(&ui_state).t, None);
    assert!(!state.get_t_range());
}

/// A plate layout: Layout, Columns (the picker), Values & classes - its
/// columns, several values, colours, the image grid.
#[test]
fn a_plate_layout_picks_columns_and_values() {
    let dir = tempfile::tempdir().unwrap();
    let (ui, ui_state, _controller) = setup(saved_project_with_an_analysis(dir.path()));
    let state = ui.global::<ExportsState>();
    state.invoke_new_export();
    state.invoke_remove_entry(0);
    state.invoke_add_entry("DATA".into());
    only(&state, "PLATE");
    state.invoke_refresh();
    drain_ui_queue();
    let plate = |ui_state: &UiState| match &exports(ui_state)[0].entries[0] {
        ExportEntry::Data(plate) => plate.clone(),
        _ => panic!("a plate layout"),
    };
    assert!(state.get_columns_default(), "the count");
    assert!(ticked_in_picker(&ui).contains(&"count".to_string()));
    state.invoke_picker("picked".into(), "area_px".into(), true);
    assert_eq!(
        plate(&ui_state).columns,
        Some(vec!["count".to_string(), "area_px".to_string()])
    );
    state.invoke_set_entry_field("aggregation:max".into(), "1".into());
    assert_eq!(
        plate(&ui_state).aggregations,
        vec![ExportAggregation::Avg, ExportAggregation::Max]
    );
    state.invoke_set_entry_field("colors".into(), "1".into());
    only(&state, "IMAGE_HEATMAP");
    state.invoke_set_entry_field("square".into(), "3".into());
    let p = plate(&ui_state);
    assert_eq!(
        p.heatmap.colors,
        evanalyzer_cfg::settings::export_settings::HeatmapColors::Viridis
    );
    assert_eq!(p.outputs, vec![DataOutput::ImageHeatmap]);
    assert_eq!(p.heatmap.square_size, Some(256));
    state.invoke_set_entry_field("clear-columns".into(), "".into());
    assert_eq!(plate(&ui_state).columns, None);
}

/// A data export's outputs: ticked in any order, kept in the list's; the
/// panel follows them (values, classes without objects, transpose,
/// heatmap colours); the tracks' settings under their tick.
#[test]
fn a_data_export_ticks_its_outputs() {
    let (ui, ui_state, _controller) = setup(ProjectWithRuntime::default());
    let state = ui.global::<ExportsState>();
    state.invoke_new_export();
    let data = |ui_state: &UiState| match &exports(ui_state)[0].entries[0] {
        ExportEntry::Data(data) => data.clone(),
        _ => panic!("a data export"),
    };
    // A new one: per image.
    assert_eq!(data(&ui_state).outputs, vec![DataOutput::Images]);
    assert!(state.get_grouped() && state.get_show_empty() && state.get_can_transpose());
    assert!(!state.get_any_heatmap());

    state.invoke_set_entry_field("output:PLATE".into(), "1".into());
    state.invoke_set_entry_field("output:OBJECTS".into(), "1".into());
    state.invoke_set_entry_field("output:TRACKS".into(), "1".into());
    assert_eq!(
        data(&ui_state).outputs,
        vec![
            DataOutput::Objects,
            DataOutput::Images,
            DataOutput::Tracks,
            DataOutput::Plate
        ]
    );
    assert!(state.get_any_heatmap());
    let ticked: Vec<String> = state
        .get_outputs()
        .iter()
        .filter(|row| row.on)
        .map(|row| row.key.to_string())
        .collect();
    assert_eq!(ticked, ["OBJECTS", "IMAGES", "TRACKS", "PLATE"]);
    // The first of each group carries its caption.
    let captions: Vec<String> = state
        .get_outputs()
        .iter()
        .filter(|row| !row.group.is_empty())
        .map(|row| row.group.to_string())
        .collect();
    assert_eq!(
        captions,
        [
            "A ROW PER",
            "OVER TIME AND RADIUS",
            "PLATE AND WELLS (XLSX)"
        ]
    );
    assert!(
        entries(&ui)[0].title.contains("4 outputs"),
        "{}",
        entries(&ui)[0].title
    );

    state.invoke_set_entry_field("tracks-min-points".into(), "5".into());
    state.invoke_set_entry_field("tracks-no-edge".into(), "1".into());
    state.invoke_set_entry_field("tracks-speed".into(), "2".into());
    let tracks = data(&ui_state).tracks.unwrap();
    assert_eq!(tracks.min_points, 5);
    assert!(tracks.no_edge && !tracks.from_start);
    assert_eq!(
        tracks.speed_unit,
        evanalyzer_cfg::settings::export_settings::SpeedUnit::UmS
    );
    assert_eq!(state.get_tracks_min_points(), "5");

    // Only objects: no values, no classes without objects, no heatmap.
    only(&state, "OBJECTS");
    assert!(!state.get_grouped() && !state.get_show_empty() && !state.get_any_heatmap());
    // None ticked: it can't run.
    state.invoke_set_entry_field("output:OBJECTS".into(), "0".into());
    assert!(
        state.get_problems().contains("output"),
        "{}",
        state.get_problems()
    );
}

/// A run of a data export writes every ticked output, one filter.
#[test]
fn a_data_export_run_writes_each_output() {
    let dir = tempfile::tempdir().unwrap();
    let (ui, _ui_state, _controller) = setup(saved_project_with_an_analysis(dir.path()));
    let state = ui.global::<ExportsState>();
    state.invoke_new_export();
    state.invoke_set_entry_field("output:OBJECTS".into(), "1".into());
    state.invoke_set_entry_field("images".into(), "0".into());
    state.invoke_run();
    drain_ui_queue();
    let run = dir.path().join("results/20261124T090000Z__run");
    assert!(run.join("objects.csv").is_file());
    assert!(run.join("images.csv").is_file());
    assert!(
        report_of(&state).iter().any(|l| l.contains("2 files")),
        "{:?}",
        report_of(&state)
    );
}

/// Classes side by side from the tab: per image and objects written
/// transposed.
#[test]
fn classes_side_by_side_from_the_tab() {
    let dir = tempfile::tempdir().unwrap();
    let (ui, _ui_state, _controller) = setup(saved_project_with_an_analysis(dir.path()));
    let state = ui.global::<ExportsState>();
    state.invoke_new_export();
    state.invoke_set_entry_field("output:OBJECTS".into(), "1".into());
    state.invoke_set_entry_field("images".into(), "0".into());
    state.invoke_set_entry_field("transposed".into(), "1".into());
    state.invoke_run();
    drain_ui_queue();
    let run = dir.path().join("results/20261124T090000Z__run");
    let images = std::fs::read_to_string(run.join("images.csv")).unwrap();
    let header = images.lines().next().unwrap();
    assert!(
        !header.contains(",class,"),
        "per image transposed: {header}"
    );
    assert!(header.contains("_classa"), "per image transposed: {header}");
    let objects = std::fs::read_to_string(run.join("objects.csv")).unwrap();
    let header = objects.lines().next().unwrap();
    assert!(
        header.contains("object_id_classa"),
        "objects transposed: {header}"
    );
}

/// The list like every list: the eye runs it after analysis (the head
/// counts them), ⋯ › Move up / down reorders - at an end it stays.
#[test]
fn the_list_eye_and_order() {
    let (ui, ui_state, _controller) = setup(ProjectWithRuntime::default());
    let state = ui.global::<ExportsState>();
    state.invoke_new_export();
    state.invoke_new_export();
    let ids: Vec<i32> = state.get_exports().iter().map(|row| row.id).collect();
    assert_eq!(ids.len(), 2);
    let after = state.get_after_count();
    let first = state.get_exports().row_data(0).unwrap();
    state.invoke_set_after(first.id, !first.after);
    let toggled = state.get_exports().row_data(0).unwrap();
    assert_eq!(toggled.after, !first.after);
    assert_eq!(
        state.get_after_count(),
        if toggled.after { after + 1 } else { after - 1 }
    );

    state.invoke_move_export(ids[0], 1);
    let order: Vec<u32> = exports(&ui_state).iter().map(|e| e.id).collect();
    assert_eq!(order, [ids[1] as u32, ids[0] as u32]);
    state.invoke_move_export(ids[0], 1);
    let order: Vec<u32> = exports(&ui_state).iter().map(|e| e.id).collect();
    assert_eq!(order, [ids[1] as u32, ids[0] as u32], "at the end it stays");
    let shown: Vec<i32> = state.get_exports().iter().map(|row| row.id).collect();
    assert_eq!(shown, [ids[1], ids[0]], "the list too");
}

/// One at a time: no export while an analysis runs (also not while its
/// own exports, its last step, run). A run by hand ends with its note,
/// like an analysis': how it went, where; the tab unlocked again.
#[test]
fn an_export_waits_for_the_analysis_and_ends_with_its_note() {
    let dir = tempfile::tempdir().unwrap();
    let (ui, _ui_state, _controller) = setup(saved_project_with_an_analysis(dir.path()));
    let state = ui.global::<ExportsState>();
    let running = ui.global::<crate::PipelineRunningState>();
    state.invoke_new_export();
    state.invoke_set_entry_field("images".into(), "0".into());
    let run = dir.path().join("results/20261124T090000Z__run");

    running.set_active(true);
    state.invoke_run();
    drain_ui_queue();
    assert!(!run.join("images.csv").exists(), "the analysis runs");
    running.set_done(true);
    running.set_exporting(true);
    state.invoke_run();
    drain_ui_queue();
    assert!(!run.join("images.csv").exists(), "its exports run");
    running.set_exporting(false);

    state.invoke_run();
    drain_ui_queue();
    assert!(run.join("images.csv").is_file());
    assert!(!state.get_running());
    assert_eq!(state.get_run_name(), "", "no pill after it");
    assert!(state.get_after_shown());
    assert_eq!(state.get_after_tone(), 1);
    assert!(
        state
            .get_after_text()
            .starts_with("Export finished · “Export”"),
        "{}",
        state.get_after_text()
    );
    assert!(state.get_after_folder(), "Open folder");
}

/// The run pills are in the top bar on every tab: an analysis' and an
/// export's (run by hand), also in Results and Exports.
#[test]
fn the_run_pills_show_on_every_tab() {
    use crate::{LayoutState, PipelineRunningState, StartState, WorkspaceMode};
    use i_slint_backend_testing::ElementHandle;
    let (ui, _ui_state, _controller) = setup(ProjectWithRuntime::default());
    ui.window().set_size(slint::LogicalSize::new(1280.0, 680.0));
    ui.global::<StartState>().set_active(false);
    let shown = |label: &str| {
        ElementHandle::find_by_accessible_label(&ui, label).any(|e| e.size().width > 0.0)
    };
    let state = ui.global::<ExportsState>();
    for mode in [
        WorkspaceMode::Images,
        WorkspaceMode::Results,
        WorkspaceMode::Exports,
    ] {
        ui.global::<LayoutState>().set_mode(mode);
        // Set after the tab is made (on the first look): the Exports tab
        // refreshes its state when it opens.
        let _ = shown("");
        drain_ui_queue();
        state.set_running(true);
        state.set_run_name("“Export”".into());
        state.set_progress("2/2 · drawn_images".into());
        assert!(
            shown("Exporting, “Export” · 2/2 · drawn_images"),
            "the export's pill in {mode:?} (running {})",
            state.get_running()
        );
    }
    // Its pill takes the place of Run (and Auto preview).
    ui.global::<LayoutState>().set_mode(WorkspaceMode::Images);
    state.set_running(true);
    assert!(!shown("Run"), "no Run beside the pill");
    state.set_running(false);
    assert!(shown("Run"), "Run back");
    let running = ui.global::<PipelineRunningState>();
    running.set_active(true);
    for mode in [WorkspaceMode::Results, WorkspaceMode::Exports] {
        ui.global::<LayoutState>().set_mode(mode);
        assert!(
            shown("Analyzing, starting"),
            "the analysis' pill in {mode:?}"
        );
    }
}
