//! # exports_controller
//!
//! The Exports tab (`ui/exports/exports_view.slint`): the project's exports
//! (`ProjectSettings::exports`, `docs/exports.md`), the one being edited,
//! its entries, and running it on the project's newest analysis. Every edit
//! goes into the project (undoable, marks it unsaved) and pushes the state
//! back; an undo or redo of the exports pushes it again.
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-11-24
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.

use crate::UiState;
use crate::editor::column_picker::{
    self as picker, COLOC_BOX, ColocGroupUi, ColocSettings, PickerSource, SectionUi,
};
use crate::helper::ui_thread;
use crate::{AppWindow, ExportEntryRow, ExportRow, ExportsState, MultiSelectItem};
use evanalyzer_app::exports::{
    EntryStatus, ExportReport, ScriptPermission, check::check_export, default_columns,
    is_leading_column, naming, prepare_run, results_database,
};
use evanalyzer_app::modules::Severity;
use evanalyzer_app::results::view_settings::ViewSettings;
use evanalyzer_app::results::{Column, ColumnEntry};
use evanalyzer_cfg::settings::classification_settings::Class;
use evanalyzer_cfg::settings::export_settings::{
    DrawnBrightness, DrawnStyle, ExportAggregation, ExportEntry, ExportSettings, IfExists,
    ImageScope, LengthUnit, OutputLocation, PlateGrid, ScriptLanguage, SplitInto, TableExport,
    TableFileFormat, TableRows, TableSplit,
};
use slint::{ComponentHandle, ModelRc, SharedString, VecModel};
use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

/// The models' orders - the indexes the view's segments and selects use.
const ROWS: [TableRows; 7] = [
    TableRows::Object,
    TableRows::Image,
    TableRows::Well,
    TableRows::Folder,
    TableRows::ColocPair,
    TableRows::Track,
    TableRows::Measurement,
];
const SPLITS: [TableSplit; 5] = [
    TableSplit::None,
    TableSplit::Image,
    TableSplit::Well,
    TableSplit::Folder,
    TableSplit::Class,
];
const SCOPES: [ImageScope; 3] = [ImageScope::All, ImageScope::Included, ImageScope::LeftOut];
const STYLES: [DrawnStyle; 3] = [DrawnStyle::Outline, DrawnStyle::Filled, DrawnStyle::Box];
const MAX_SIDES: [u32; 4] = [1024, 2048, 4096, 8192];
const AGGREGATIONS: [(ExportAggregation, &str, &str); 7] = [
    (ExportAggregation::Avg, "avg", "Mean"),
    (ExportAggregation::Median, "median", "Median"),
    (ExportAggregation::Stddev, "stddev", "Standard deviation"),
    (ExportAggregation::Sum, "sum", "Sum"),
    (ExportAggregation::Min, "min", "Minimum"),
    (ExportAggregation::Max, "max", "Maximum"),
    (ExportAggregation::Skewness, "skewness", "Skewness"),
];

/// What the tab shows besides the project: what is picked, the last runs.
#[derive(Default)]
struct Shown {
    /// The export being edited (its id).
    selected: Option<u32>,
    /// Its entry picked.
    entry: Option<usize>,
    /// The last run here, per export: (text, tone).
    status: BTreeMap<u32, (String, i32)>,
    /// The last run's report and a failed script's log.
    report: Vec<String>,
    log: String,
    /// The run going on: its cancel flag.
    running: Option<Arc<AtomicBool>>,
    progress: String,
    /// The newest analysis' columns and classes - what the column picker
    /// offers; read when the tab is shown.
    picked_from: Option<PickedFrom>,
    /// Why there is nothing to pick from ("" there is).
    columns_note: String,
    /// The picker's coloc settings, its sections folded by hand, its
    /// search - for the session, like the Results List's.
    coloc: ColocSettings,
    sections_open: HashMap<String, bool>,
    column_filter: String,
}

/// What the column picker is made of: the results' own columns and classes.
#[derive(Clone)]
struct PickedFrom {
    raw: Vec<ColumnEntry>,
    classes: Vec<Class>,
}

impl PickedFrom {
    /// The columns `table` has, as the picker knows them (its default ones
    /// without `columns`) - and its keys these results lack.
    fn selection(&self, table: &TableExport) -> (Vec<Column>, Vec<String>) {
        let Some(keys) = &table.columns else {
            return (default_columns(table.rows), Vec::new());
        };
        let mut selected = Vec::new();
        let mut missing = Vec::new();
        for key in keys {
            let found = Column::from_key(key, &self.classes).or_else(|| {
                Column::measurement_key(key)?;
                self.raw
                    .iter()
                    .find(|entry| entry.key.as_key(&self.classes) == *key)
                    .map(|entry| entry.key.clone())
            });
            match found {
                Some(column) if !selected.contains(&column) => selected.push(column),
                Some(_) => {}
                None => missing.push(key.clone()),
            }
        }
        (selected, missing)
    }

    fn source<'a>(
        &'a self,
        settings: &'a ColocSettings,
        open: &'a HashMap<String, bool>,
        filter: &'a str,
    ) -> PickerSource<'a> {
        PickerSource {
            raw: &self.raw,
            classes: &self.classes,
            settings,
            open,
            filter,
        }
    }
}

/// The columns offered for a table of `rows`: never the leading ones; a
/// table of measurements has measurements only.
fn offered_for(rows: TableRows, mut offered: Vec<ColumnEntry>) -> Vec<ColumnEntry> {
    offered.retain(|entry| !is_leading_column(&entry.key));
    if rows == TableRows::Measurement {
        offered.retain(|entry| matches!(entry.key, Column::Measurement(_)));
    }
    offered
}

/// What an entry's setting may need besides the entry: the views (their
/// columns).
struct Pickable {
    views: evanalyzer_cfg::settings::project_settings::ResultsViews,
    /// The project's classes (ids) - all ticked is every class.
    classes: Vec<u32>,
}

pub struct ExportsController {
    ui: slint::Weak<AppWindow>,
    app_state: Arc<UiState>,
    shown: Mutex<Shown>,
}

impl ExportsController {
    pub fn new(ui: slint::Weak<AppWindow>, app_state: Arc<UiState>) -> Self {
        Self {
            ui,
            app_state,
            shown: Mutex::new(Shown::default()),
        }
    }

    pub fn attach_callbacks(self: &Arc<Self>) {
        let Some(ui) = self.ui.upgrade() else {
            return;
        };
        let state = ui.global::<ExportsState>();

        let this = self.clone();
        state.on_refresh(move || {
            this.push();
            this.load_columns();
        });

        let this = self.clone();
        state.on_picker(move |what, key, on| this.picker_action(&what, &key, on));

        let this = self.clone();
        state.on_select(move |id| {
            {
                let mut shown = this.shown.lock().expect("Poisoned");
                shown.selected = u32::try_from(id).ok();
                shown.entry = None;
            }
            this.push();
        });

        let this = self.clone();
        state.on_new_export(move || this.new_export());

        let this = self.clone();
        state.on_duplicate(move |id| this.duplicate(id));

        let this = self.clone();
        state.on_delete(move |id| this.delete(id));

        let this = self.clone();
        state.on_set_after(move |id, on| {
            let Ok(id) = u32::try_from(id) else {
                return;
            };
            this.edit(id, |export| export.run_after_every_run = on);
        });

        let this = self.clone();
        state.on_set_export_field(move |key, value| {
            let Some(id) = this.selected() else {
                return;
            };
            this.edit(id, |export| set_export_field(export, &key, &value));
        });

        let this = self.clone();
        state.on_add_entry(move |kind| this.add_entry(&kind));

        let this = self.clone();
        state.on_remove_entry(move |index| {
            let (Some(id), Ok(index)) = (this.selected(), usize::try_from(index)) else {
                return;
            };
            this.edit(id, |export| {
                if index < export.entries.len() {
                    export.entries.remove(index);
                }
            });
            let mut shown = this.shown.lock().expect("Poisoned");
            shown.entry = match shown.entry {
                Some(picked) if picked == index => None,
                Some(picked) if picked > index => Some(picked - 1),
                other => other,
            };
            drop(shown);
            this.push();
        });

        let this = self.clone();
        state.on_move_entry(move |index, by| {
            let (Some(id), Ok(index)) = (this.selected(), usize::try_from(index)) else {
                return;
            };
            let Some(to) = index.checked_add_signed(by as isize) else {
                return;
            };
            let mut moved = false;
            this.edit(id, |export| {
                if index < export.entries.len() && to < export.entries.len() {
                    export.entries.swap(index, to);
                    moved = true;
                }
            });
            if moved {
                let mut shown = this.shown.lock().expect("Poisoned");
                if shown.entry == Some(index) {
                    shown.entry = Some(to);
                } else if shown.entry == Some(to) {
                    shown.entry = Some(index);
                }
                drop(shown);
                this.push();
            }
        });

        let this = self.clone();
        state.on_select_entry(move |index| {
            this.shown.lock().expect("Poisoned").entry = usize::try_from(index).ok();
            this.push();
        });

        let this = self.clone();
        state.on_set_entry_field(move |key, value| {
            let (Some(id), Some(index)) = (this.selected(), this.entry()) else {
                return;
            };
            let pickable = {
                let project = this.app_state.get_project();
                Pickable {
                    views: project.settings.results_views.clone(),
                    classes: project
                        .settings
                        .classification
                        .classes()
                        .iter()
                        .map(|class| class.id)
                        .filter(|id| *id != 0)
                        .collect(),
                }
            };
            this.edit(id, |export| {
                if let Some(entry) = export.entries.get_mut(index) {
                    set_entry_field(entry, &key, &value, &pickable);
                }
            });
        });

        let this = self.clone();
        state.on_run(move || this.run());

        let this = self.clone();
        state.on_cancel(move || {
            if let Some(cancel) = &this.shown.lock().expect("Poisoned").running {
                cancel.store(true, Ordering::Relaxed);
            }
        });

        // An undo or redo of the exports: shown anew (on the UI thread).
        let this = self.clone();
        self.app_state.on_exports_restored(move || {
            let this = this.clone();
            ui_thread::invoke_from_event_loop(move || this.push()).ok();
        });
    }

    fn selected(&self) -> Option<u32> {
        self.shown.lock().expect("Poisoned").selected
    }

    fn entry(&self) -> Option<usize> {
        self.shown.lock().expect("Poisoned").entry
    }

    /// Changes the export `id` in the project, then shows it. A change is
    /// an undo step of its own (every edit here is a click or an accepted
    /// field); one that changes nothing is none.
    fn edit(&self, id: u32, change: impl FnOnce(&mut ExportSettings)) {
        let changed = {
            let project = self.app_state.get_project();
            project
                .settings
                .exports
                .iter()
                .find(|e| e.id == id)
                .and_then(|export| {
                    let mut edited = export.clone();
                    change(&mut edited);
                    (edited != *export).then_some(edited)
                })
        };
        if let Some(edited) = changed {
            self.change_exports(|exports| {
                if let Some(export) = exports.iter_mut().find(|e| e.id == id) {
                    *export = edited;
                }
            });
        }
        self.push();
    }

    /// Changes the project's exports as one undo step, marks it unsaved.
    fn change_exports<T>(&self, change: impl FnOnce(&mut Vec<ExportSettings>) -> T) -> T {
        self.app_state.begin_undo_step();
        let result = change(&mut self.app_state.get_project_write().settings.exports);
        self.app_state.mark_dirty();
        result
    }

    fn new_export(&self) {
        let id = self.change_exports(|exports| {
            let id = exports.iter().map(|e| e.id).max().map_or(1, |max| max + 1);
            let name = free_export_name(exports, "Export");
            let mut export = new_export(id, name);
            export.entries.push(new_entry("TABLE"));
            exports.push(export);
            id
        });
        {
            let mut shown = self.shown.lock().expect("Poisoned");
            shown.selected = Some(id);
            shown.entry = Some(0);
        }
        self.push();
    }

    fn duplicate(&self, id: i32) {
        let Ok(id) = u32::try_from(id) else {
            return;
        };
        let Some(original) = self
            .app_state
            .get_project()
            .settings
            .exports
            .iter()
            .find(|e| e.id == id)
            .cloned()
        else {
            return;
        };
        let copy = self.change_exports(|exports| {
            let copy = exports.iter().map(|e| e.id).max().unwrap_or(0) + 1;
            let name = free_export_name(exports, &format!("{} copy", original.name));
            exports.push(ExportSettings {
                id: copy,
                name,
                // Two of the same after every run write into each other.
                run_after_every_run: false,
                ..original
            });
            copy
        });
        {
            let mut shown = self.shown.lock().expect("Poisoned");
            shown.selected = Some(copy);
            shown.entry = None;
        }
        self.push();
    }

    fn delete(&self, id: i32) {
        let Ok(id) = u32::try_from(id) else {
            return;
        };
        let exists = self
            .app_state
            .get_project()
            .settings
            .exports
            .iter()
            .any(|e| e.id == id);
        if exists {
            self.change_exports(|exports| exports.retain(|e| e.id != id));
            let mut shown = self.shown.lock().expect("Poisoned");
            if shown.selected == Some(id) {
                shown.selected = None;
                shown.entry = None;
            }
            shown.status.remove(&id);
        }
        self.push();
    }

    fn add_entry(&self, kind: &str) {
        let Some(id) = self.selected() else {
            return;
        };
        let mut added = None;
        self.edit(id, |export| {
            export.entries.push(new_entry(kind));
            added = Some(export.entries.len() - 1);
        });
        self.shown.lock().expect("Poisoned").entry = added;
        self.push();
    }

    /// The project's folder (`None`: not saved yet).
    fn project_dir(&self) -> Option<PathBuf> {
        self.app_state
            .get_project()
            .tmp_settings
            .current_project
            .as_ref()
            .and_then(|p| p.parent())
            .map(Path::to_path_buf)
    }

    /// Runs the export being edited on the project's newest analysis, in
    /// the background; the report shows when done.
    fn run(self: &Arc<Self>) {
        let Some(id) = self.selected() else {
            return;
        };
        if self.shown.lock().expect("Poisoned").running.is_some() {
            return;
        }
        let Some(export) = self
            .app_state
            .get_project()
            .settings
            .exports
            .iter()
            .find(|e| e.id == id)
            .cloned()
        else {
            return;
        };
        let Some(project_dir) = self.project_dir() else {
            self.finish(
                id,
                Err("Save the project first - an export runs on its analyses.".into()),
            );
            return;
        };
        let cancel = Arc::new(AtomicBool::new(false));
        {
            let mut shown = self.shown.lock().expect("Poisoned");
            shown.running = Some(cancel.clone());
            shown.progress = "Starting…".into();
            shown.report.clear();
            shown.log.clear();
        }
        self.push();

        let this = self.clone();
        ui_thread::spawn(move || {
            let backend = this.app_state.backend();
            let result = (|| {
                let database = results_database(backend.files(), &project_dir, None)?;
                let trusted = backend
                    .load_app_settings()
                    .unwrap_or_default()
                    .trusted_scripts;
                let (results, run) = {
                    let project = this.app_state.get_project();
                    prepare_run(
                        backend.as_ref(),
                        &project,
                        &project_dir,
                        export.clone(),
                        &database,
                        None,
                        ScriptPermission::Trusted(trusted),
                    )?
                };
                let progress_to = this.clone();
                results.run_export(&run, &cancel, &mut |message, current, total| {
                    let text = if current < total {
                        format!("{}/{total} · {message}", current + 1)
                    } else {
                        message.to_string()
                    };
                    let this = progress_to.clone();
                    ui_thread::invoke_from_event_loop(move || {
                        this.shown.lock().expect("Poisoned").progress = text;
                        this.push();
                    })
                    .ok();
                })
            })();
            let result = result.map_err(|e| e.to_string());
            ui_thread::invoke_from_event_loop(move || this.finish(id, result.map(|r| (export, r))))
                .ok();
        });
    }

    /// A run of the export `id` ended: its report and status.
    fn finish(&self, id: u32, result: Result<(ExportSettings, ExportReport), String>) {
        {
            let mut shown = self.shown.lock().expect("Poisoned");
            shown.running = None;
            shown.progress.clear();
            let time = chrono::Local::now().format("%H:%M").to_string();
            match result {
                Err(why) => {
                    shown.report = vec![format!("Did not run: {why}")];
                    shown.log.clear();
                    shown.status.insert(id, ("⚠ did not run".into(), 2));
                }
                Ok((export, report)) => {
                    let (lines, log, failed) = report_lines(&export, &report);
                    shown.report = lines;
                    shown.log = log;
                    let status = if failed > 0 {
                        (format!("⚠ {failed} failed · {time}"), 2)
                    } else {
                        (format!("✓ {time}"), 1)
                    };
                    shown.status.insert(id, status);
                }
            }
        }
        self.push();
    }

    /// Shows the project's exports and the one being edited.
    fn push(&self) {
        let Some(ui) = self.ui.upgrade() else {
            return;
        };
        let state = ui.global::<ExportsState>();
        let project = self.app_state.get_project();
        let settings = &project.settings;
        let mut shown = self.shown.lock().expect("Poisoned");
        if shown
            .selected
            .is_some_and(|id| !settings.exports.iter().any(|e| e.id == id))
        {
            shown.selected = None;
            shown.entry = None;
        }

        let rows: Vec<ExportRow> = settings
            .exports
            .iter()
            .map(|export| {
                let (status, tone) = shown.status.get(&export.id).cloned().unwrap_or_default();
                ExportRow {
                    id: export.id as i32,
                    name: export.name.as_str().into(),
                    summary: summary(export).into(),
                    after: export.run_after_every_run,
                    status: status.into(),
                    status_tone: tone,
                }
            })
            .collect();
        state.set_exports(model(rows));
        state.set_selected_id(shown.selected.map_or(-1, |id| id as i32));
        state.set_running(shown.running.is_some());
        state.set_progress(shown.progress.as_str().into());
        state.set_report(model(shown.report.iter().map(SharedString::from).collect()));
        state.set_log(shown.log.as_str().into());

        let Some(export) = shown
            .selected
            .and_then(|id| settings.exports.iter().find(|e| e.id == id))
        else {
            state.set_entries(model(Vec::new()));
            state.set_entry(-1);
            state.set_entry_kind("".into());
            return;
        };
        if shown.entry.is_some_and(|i| i >= export.entries.len()) {
            shown.entry = None;
        }
        state.set_name(export.name.as_str().into());
        state.set_prefix(export.prefix.as_str().into());
        state.set_location(match export.output.location {
            OutputLocation::ResultsFolder => 0,
            OutputLocation::FixedFolder => 1,
        });
        state.set_folder(export.output.folder.as_str().into());
        state.set_if_exists(match export.output.if_exists {
            IfExists::Overwrite => 0,
            IfExists::Rename => 1,
        });
        state.set_length_unit(match export.length_unit {
            LengthUnit::Um => 0,
            LengthUnit::Nm => 1,
        });
        state.set_after(export.run_after_every_run);
        state.set_problems(problems_text(export).into());
        state.set_entries(model(
            export
                .entries
                .iter()
                .map(|entry| entry_row(export, entry))
                .collect(),
        ));
        state.set_entry(shown.entry.map_or(-1, |i| i as i32));

        let Some(entry) = shown.entry.and_then(|i| export.entries.get(i)) else {
            state.set_entry_kind("".into());
            return;
        };
        let project_classes: Vec<&Class> = settings
            .classification
            .classes()
            .iter()
            .filter(|class| class.id != 0)
            .collect();
        let set_classes = |picked: &Option<Vec<u32>>| {
            let items: Vec<MultiSelectItem> = project_classes
                .iter()
                .map(|class| MultiSelectItem {
                    key: class.id.to_string().into(),
                    value: class.name.as_str().into(),
                    color: crate::editor::results_state_controller::class_color_to_slint(
                        class.color,
                    ),
                    group: "".into(),
                    selected: picked.as_ref().is_none_or(|ids| ids.contains(&class.id)),
                })
                .collect();
            let ticked = items.iter().filter(|item| item.selected).count();
            state.set_classes_summary(
                if picked.is_none() {
                    "all".to_string()
                } else {
                    format!("{ticked} of {}", items.len())
                }
                .into(),
            );
            state.set_classes(model(items));
        };
        state.set_entry_kind(kind_name(entry).into());
        match entry {
            ExportEntry::Table(table) => {
                state.set_rows(index_of(&ROWS, &table.rows));
                state.set_format(match table.format {
                    TableFileFormat::Csv => 0,
                    TableFileFormat::Xlsx => 1,
                });
                state.set_split(index_of(&SPLITS, &table.split));
                state.set_split_into(match table.split_into {
                    SplitInto::Files => 0,
                    SplitInto::Sheets => 1,
                });
                state.set_images(index_of(&SCOPES, &table.images));
                set_classes(&table.classes);
                state.set_transposed(table.transposed);
                state.set_empty_classes(table.empty_classes);
                push_picker(&state, table, &shown);
                state.set_aggregations(model(
                    AGGREGATIONS
                        .iter()
                        .map(|(aggregation, key, text)| MultiSelectItem {
                            key: (*key).into(),
                            value: (*text).into(),
                            color: Default::default(),
                            group: "".into(),
                            selected: table.aggregations.contains(aggregation),
                        })
                        .collect(),
                ));
                state.set_views(model(
                    settings
                        .results_views
                        .views
                        .iter()
                        .filter(|view| view_columns(view).is_some())
                        .map(|view| SharedString::from(view.name.as_str()))
                        .collect(),
                ));
            }
            ExportEntry::PlateLayout(plate) => {
                state.set_grid(match plate.grid {
                    PlateGrid::Plate => 0,
                    PlateGrid::Well => 1,
                });
                set_classes(&plate.classes);
            }
            ExportEntry::DrawnImages(drawn) => {
                state.set_style(index_of(&STYLES, &drawn.style));
                state.set_brightness(match drawn.brightness {
                    DrawnBrightness::Viewer => 0,
                    DrawnBrightness::Auto => 1,
                });
                state.set_max_side(
                    MAX_SIDES
                        .iter()
                        .position(|side| *side >= drawn.max_side)
                        .unwrap_or(MAX_SIDES.len() - 1) as i32,
                );
                set_classes(&drawn.classes);
            }
            ExportEntry::Script(script) => {
                state.set_script_file(script.file.as_str().into());
                state.set_language(match script.language {
                    ScriptLanguage::R => 0,
                    ScriptLanguage::Python => 1,
                });
                state.set_script_output(script.output.as_str().into());
                drop(shown);
                let files = self.project_dir().map(|dir| self.script_files(&dir));
                state.set_script_files(model(
                    files
                        .unwrap_or_default()
                        .iter()
                        .map(SharedString::from)
                        .collect(),
                ));
            }
        }
    }

    /// Reads the columns of the project's newest analysis in the
    /// background - what a table's columns are picked from.
    fn load_columns(self: &Arc<Self>) {
        let Some(project_dir) = self.project_dir() else {
            let mut shown = self.shown.lock().expect("Poisoned");
            shown.picked_from = None;
            shown.columns_note = "Save the project and run an analysis to pick columns.".into();
            drop(shown);
            self.push();
            return;
        };
        let this = self.clone();
        ui_thread::spawn(move || {
            let backend = this.app_state.backend();
            let found = (|| {
                let database = results_database(backend.files(), &project_dir, None)?;
                let results = backend.open_results(&database)?;
                Ok::<_, evanalyzer_cfg::core_types::InternalErrors>(PickedFrom {
                    classes: results.get_object_classes()?,
                    raw: results.get_available_columns()?,
                })
            })();
            ui_thread::invoke_from_event_loop(move || {
                {
                    let mut shown = this.shown.lock().expect("Poisoned");
                    match found {
                        Ok(from) => {
                            // Every partner class, as a fresh analysis in
                            // Results starts.
                            shown.coloc = picker::default_settings(&from.raw, &[]);
                            shown.picked_from = Some(from);
                            shown.columns_note.clear();
                        }
                        Err(e) => {
                            shown.picked_from = None;
                            shown.columns_note = format!("No columns to pick from: {e}");
                        }
                    }
                }
                this.push();
            })
            .ok();
        });
    }

    /// The column picker's `what` (see `ExportsState.picker`): its search,
    /// sections and coloc settings for the session, the columns picked in
    /// the table picked.
    fn picker_action(&self, what: &str, key: &str, on: bool) {
        {
            let mut shown = self.shown.lock().expect("Poisoned");
            match what {
                "filter" => shown.column_filter = key.to_string(),
                "section" => {
                    shown.sections_open.insert(key.to_string(), on);
                }
                "coloc-box" => {
                    shown.sections_open.insert(COLOC_BOX.to_string(), on);
                }
                _ => {
                    drop(shown);
                    self.change_columns(what, key, on);
                    return;
                }
            }
        }
        self.push();
    }

    /// A pick that changes the table's columns (or the coloc settings and
    /// with them its coloc columns) - as the Results List does it.
    fn change_columns(&self, what: &str, key: &str, on: bool) {
        let (Some(id), Some(index)) = (self.selected(), self.entry()) else {
            return;
        };
        let Some(ExportEntry::Table(table)) = self
            .app_state
            .get_project()
            .settings
            .exports
            .iter()
            .find(|e| e.id == id)
            .and_then(|e| e.entries.get(index))
            .cloned()
        else {
            return;
        };
        let keys = {
            let mut shown = self.shown.lock().expect("Poisoned");
            let Some(from) = shown.picked_from.clone() else {
                return;
            };
            let (mut selected, missing) = from.selection(&table);
            // The coloc settings first - what is offered follows them.
            match what {
                "coloc-partner" => picker::toggle_partner(&mut shown.coloc, key),
                "coloc-partners-all" => {
                    let all = picker::coloc_classes(&from.raw);
                    picker::change_settings(&mut shown.coloc, |s| {
                        s.partners = if on { all } else { Vec::new() };
                    });
                }
                "coloc-count" => {
                    picker::change_settings(&mut shown.coloc, |s| s.count_partners = key == "1");
                }
                "coloc-only-occurring" => {
                    picker::change_settings(&mut shown.coloc, |s| s.only_occurring = on);
                }
                // Coloc first makes every partner class chosen if none is.
                "preset" if key == "coloc" && shown.coloc.partners.is_empty() => {
                    let all = picker::coloc_classes(&from.raw);
                    picker::change_settings(&mut shown.coloc, |s| s.partners = all);
                }
                _ => {}
            }
            let coloc_changed = what.starts_with("coloc-");
            if coloc_changed {
                selected = picker::adapt_columns(&selected, &shown.coloc);
            }
            let filter = shown.column_filter.to_lowercase();
            let source = from.source(&shown.coloc, &shown.sections_open, &filter);
            let offered = offered_for(table.rows, source.offered(&selected));
            match what {
                "picked" => {
                    if let Some(entry) = offered
                        .iter()
                        .find(|entry| entry.key.as_key(&from.classes) == key)
                    {
                        picker::show(&mut selected, std::slice::from_ref(&entry.key), on);
                    }
                }
                "section-select" => {
                    let columns = source.section_columns(&offered, key);
                    picker::show(&mut selected, &columns, on);
                }
                "intensity-channel" | "intensity-value" => {
                    let index = key.parse::<usize>().unwrap_or(usize::MAX);
                    let line = picker::intensity_line(&offered, index, what == "intensity-channel");
                    let all_shown = !line.is_empty() && line.iter().all(|c| selected.contains(c));
                    picker::show(&mut selected, &line, !all_shown);
                }
                "preset" => {
                    if let Some(columns) = source.preset_columns(&offered, key) {
                        selected = columns;
                    }
                }
                _ => {}
            }
            // The default columns stay the default ones while a coloc
            // setting changes nothing of them.
            if coloc_changed && table.columns.is_none() && selected == default_columns(table.rows) {
                None
            } else {
                Some(
                    selected
                        .iter()
                        .filter(|column| !is_leading_column(column))
                        .map(|column| column.as_key(&from.classes))
                        .chain(missing)
                        .collect::<Vec<String>>(),
                )
            }
        };
        // The coloc settings show anew even when the columns stay.
        self.push();
        let Some(keys) = keys else {
            return;
        };
        self.edit(id, |export| {
            if let Some(ExportEntry::Table(table)) = export.entries.get_mut(index) {
                table.columns = Some(keys);
            }
        });
    }

    /// The scripts in the project's `scripts/` folder (`scripts/<name>`).
    fn script_files(&self, project_dir: &Path) -> Vec<String> {
        let backend = self.app_state.backend();
        let mut files: Vec<String> = backend
            .files()
            .list_dir(&project_dir.join("scripts"))
            .unwrap_or_default()
            .into_iter()
            .filter(|entry| !entry.is_dir)
            .filter(|entry| {
                Path::new(&entry.name)
                    .extension()
                    .and_then(|e| e.to_str())
                    .is_some_and(|e| matches!(e.to_ascii_lowercase().as_str(), "r" | "py"))
            })
            .map(|entry| format!("scripts/{}", entry.name))
            .collect();
        files.sort();
        files
    }
}

/// The column picker of `table` (see `ColumnPicker`): over the newest
/// analysis' columns, its columns picked.
fn push_picker(state: &ExportsState, table: &TableExport, shown: &Shown) {
    state.set_picker_grouped(matches!(
        table.rows,
        TableRows::Image | TableRows::Well | TableRows::Folder
    ));
    state.set_picker_filter(shown.column_filter.as_str().into());
    state.set_columns_default(table.columns.is_none());
    let Some(from) = &shown.picked_from else {
        state.set_columns_note(shown.columns_note.as_str().into());
        state.set_picker_presets(model(Vec::new()));
        state.set_picker_sections(model(Vec::new()));
        state.set_picker_intensity_cells(model(Vec::new()));
        state.set_picker_intensity_channels(model(Vec::new()));
        state.set_picker_has_coloc(false);
        state.set_picker_coloc_groups(model(Vec::new()));
        return;
    };
    let (selected, missing) = from.selection(table);
    let filter = shown.column_filter.to_lowercase();
    let source = from.source(&shown.coloc, &shown.sections_open, &filter);
    let offered = offered_for(table.rows, source.offered(&selected));
    let parts = source.parts(&offered, &selected);
    state.set_columns_note(
        if missing.is_empty() {
            String::new()
        } else {
            format!("Not in the newest analysis (kept): {}", missing.join(", "))
        }
        .into(),
    );
    state.set_picker_presets(model(parts.presets));
    state.set_picker_sections(model(
        parts
            .sections
            .into_iter()
            .map(SectionUi::into_slint)
            .collect(),
    ));
    state.set_picker_intensity_title(parts.intensity_title.into());
    state.set_picker_intensity_open(parts.intensity_open);
    state.set_picker_intensity_channels(model(
        parts
            .intensity_channels
            .into_iter()
            .map(SharedString::from)
            .collect(),
    ));
    state.set_picker_intensity_stats(model(parts.intensity_stats));
    state.set_picker_intensity_cells(model(parts.intensity_cells));
    state.set_picker_has_coloc(!parts.partner_items.is_empty());
    state.set_picker_coloc_groups(model(
        parts
            .coloc_groups
            .into_iter()
            .map(ColocGroupUi::into_slint)
            .collect(),
    ));
    state.set_coloc_title(parts.coloc_title.into());
    state.set_coloc_open(parts.coloc_open);
    state.set_coloc_partner_items(model(parts.partner_items));
    state.set_coloc_partner_summary(parts.partner_summary.into());
    state.set_coloc_partner_order(parts.partner_order.into());
    state.set_coloc_count(parts.count);
    state.set_coloc_only_occurring(parts.only_occurring);
}

fn model<T: Clone + 'static>(items: Vec<T>) -> ModelRc<T> {
    ModelRc::new(VecModel::from(items))
}

fn index_of<T: PartialEq>(all: &[T], value: &T) -> i32 {
    all.iter().position(|v| v == value).unwrap_or(0) as i32
}

/// The setting `value` at `index` of `all` - `None` for anything else.
fn pick<T: Copy>(all: &[T], value: &str) -> Option<T> {
    value
        .parse::<usize>()
        .ok()
        .and_then(|index| all.get(index).copied())
}

fn new_export(id: u32, name: String) -> ExportSettings {
    let mut export: ExportSettings =
        serde_json::from_value(serde_json::json!({ "id": id, "name": "" }))
            .expect("an export with defaults");
    export.name = name;
    export
}

/// A new entry of `kind` with its defaults.
fn new_entry(kind: &str) -> ExportEntry {
    let value = match kind {
        "PLATE_LAYOUT" => serde_json::json!({ "kind": "PLATE_LAYOUT" }),
        "DRAWN_IMAGES" => serde_json::json!({ "kind": "DRAWN_IMAGES" }),
        "SCRIPT" => serde_json::json!({ "kind": "SCRIPT", "language": "R", "file": "" }),
        _ => serde_json::json!({ "kind": "TABLE", "rows": "IMAGE" }),
    };
    serde_json::from_value(value).expect("an entry with defaults")
}

/// `wanted`, or `wanted 2`, `wanted 3`… - the first no export has.
fn free_export_name(exports: &[ExportSettings], wanted: &str) -> String {
    let taken = |name: &str| exports.iter().any(|e| e.name.eq_ignore_ascii_case(name));
    if !taken(wanted) {
        return wanted.to_string();
    }
    (2..)
        .map(|n| format!("{wanted} {n}"))
        .find(|name| !taken(name))
        .expect("a free name")
}

fn set_export_field(export: &mut ExportSettings, key: &str, value: &str) {
    match key {
        "name" => {
            let name = value.trim();
            if !name.is_empty() {
                export.name = name.to_string();
            }
        }
        "prefix" => export.prefix = value.trim().to_string(),
        "folder" => export.output.folder = value.trim().to_string(),
        "location" => {
            if let Some(location) = pick(
                &[OutputLocation::ResultsFolder, OutputLocation::FixedFolder],
                value,
            ) {
                export.output.location = location;
            }
        }
        "if-exists" => {
            if let Some(if_exists) = pick(&[IfExists::Overwrite, IfExists::Rename], value) {
                export.output.if_exists = if_exists;
            }
        }
        "length-unit" => {
            if let Some(unit) = pick(&[LengthUnit::Um, LengthUnit::Nm], value) {
                export.length_unit = unit;
            }
        }
        "after" => export.run_after_every_run = value == "1",
        _ => log::warn!("Unknown export setting {key}"),
    }
}

/// Ticks `id` in or out of `classes` (`None`: all; none ticked again: all).
fn tick_class(classes: &mut Option<Vec<u32>>, id: &str, on: bool, all: &[u32]) {
    let Ok(id) = id.parse::<u32>() else {
        return;
    };
    // All ticked: every class (`None`) - also ones new in a run.
    let ids = classes.get_or_insert_with(|| all.to_vec());
    ids.retain(|c| *c != id);
    if on {
        ids.push(id);
        ids.sort_unstable();
    }
    if all.iter().all(|c| ids.contains(c)) {
        *classes = None;
    }
}

/// A view's column keys, if it has them set.
fn view_columns(
    view: &evanalyzer_cfg::settings::project_settings::ResultsView,
) -> Option<Vec<String>> {
    use evanalyzer_cfg::settings::project_settings::ResultsViewKind;
    let settings = ViewSettings::read(&view.settings);
    match view.kind {
        ResultsViewKind::Tracks => settings.tracks.columns,
        ResultsViewKind::List => settings.list.columns,
        _ => None,
    }
}

fn set_entry_field(entry: &mut ExportEntry, key: &str, value: &str, pickable: &Pickable) {
    let on = value == "1";
    let classes = match entry {
        ExportEntry::Table(t) => Some(&mut t.classes),
        ExportEntry::PlateLayout(p) => Some(&mut p.classes),
        ExportEntry::DrawnImages(d) => Some(&mut d.classes),
        ExportEntry::Script(_) => None,
    };
    if let Some(classes) = classes {
        if let Some(id) = key.strip_prefix("class:") {
            tick_class(classes, id, on, &pickable.classes);
            return;
        }
        match key {
            "all-classes" => {
                *classes = None;
                return;
            }
            "no-classes" => {
                *classes = Some(Vec::new());
                return;
            }
            _ => {}
        }
    }
    if key == "name" {
        let name = value.trim().to_string();
        match entry {
            ExportEntry::Table(t) => t.name = name,
            ExportEntry::PlateLayout(p) => p.name = name,
            ExportEntry::DrawnImages(d) => d.name = name,
            ExportEntry::Script(_) => {}
        }
        return;
    }
    match entry {
        ExportEntry::Table(table) => set_table_field(table, key, value, pickable),
        ExportEntry::PlateLayout(plate) => match key {
            "grid" => {
                if let Some(grid) = pick(&[PlateGrid::Plate, PlateGrid::Well], value) {
                    plate.grid = grid;
                }
            }
            "images" => {
                if let Some(scope) = pick(&SCOPES, value) {
                    plate.images = scope;
                }
            }
            _ => {}
        },
        ExportEntry::DrawnImages(drawn) => match key {
            "style" => {
                if let Some(style) = pick(&STYLES, value) {
                    drawn.style = style;
                }
            }
            "brightness" => {
                if let Some(brightness) =
                    pick(&[DrawnBrightness::Viewer, DrawnBrightness::Auto], value)
                {
                    drawn.brightness = brightness;
                }
            }
            "max-side" => {
                if let Some(side) = pick(&MAX_SIDES, value) {
                    drawn.max_side = side;
                }
            }
            "images" => {
                if let Some(scope) = pick(&SCOPES, value) {
                    drawn.images = scope;
                }
            }
            _ => {}
        },
        ExportEntry::Script(script) => match key {
            "script-file" => {
                script.file = value.to_string();
                // A Python script is Python.
                script.language = if value.to_ascii_lowercase().ends_with(".py") {
                    ScriptLanguage::Python
                } else {
                    ScriptLanguage::R
                };
            }
            "language" => {
                if let Some(language) = pick(&[ScriptLanguage::R, ScriptLanguage::Python], value) {
                    script.language = language;
                }
            }
            "script-output" => script.output = value.trim().to_string(),
            _ => {}
        },
    }
}

fn set_table_field(table: &mut TableExport, key: &str, value: &str, pickable: &Pickable) {
    let views = &pickable.views;
    match key {
        "all-aggregations" => {
            table.aggregations = AGGREGATIONS.iter().map(|(a, ..)| *a).collect();
            return;
        }
        "no-aggregations" => {
            table.aggregations.clear();
            return;
        }
        "empty-classes" => {
            table.empty_classes = value == "1";
            return;
        }
        _ => {}
    }
    if key == "transposed" {
        table.transposed = value == "1";
        return;
    }
    if let Some(aggregation) = key.strip_prefix("aggregation:") {
        let Some((aggregation, ..)) = AGGREGATIONS.iter().find(|(_, k, _)| *k == aggregation)
        else {
            return;
        };
        table.aggregations.retain(|a| a != aggregation);
        if value == "1" {
            table.aggregations.push(*aggregation);
            // In the order of the list, not of the clicks.
            table.aggregations.sort_by_key(|a| {
                AGGREGATIONS
                    .iter()
                    .position(|(b, ..)| b == a)
                    .unwrap_or(usize::MAX)
            });
        }
        return;
    }
    if let Some(name) = key.strip_prefix("columns-from-view:") {
        if let Some(view) = views.views.iter().find(|view| view.name == name) {
            table.columns = view_columns(view);
        }
        return;
    }
    if let Some(column) = key.strip_prefix("remove-column:") {
        if let Some(columns) = &mut table.columns {
            columns.retain(|c| c != column);
            if columns.is_empty() {
                table.columns = None;
            }
        }
        return;
    }
    match key {
        "clear-columns" => table.columns = None,
        "rows" => {
            if let Some(rows) = pick(&ROWS, value) {
                table.rows = rows;
                // Side by side is for objects and images only.
                if !matches!(rows, TableRows::Object | TableRows::Image) {
                    table.transposed = false;
                }
            }
        }
        "format" => {
            if let Some(format) = pick(&[TableFileFormat::Csv, TableFileFormat::Xlsx], value) {
                table.format = format;
            }
        }
        "split" => {
            if let Some(split) = pick(&SPLITS, value) {
                table.split = split;
            }
        }
        "split-into" => {
            if let Some(into) = pick(&[SplitInto::Files, SplitInto::Sheets], value) {
                table.split_into = into;
            }
        }
        "images" => {
            if let Some(scope) = pick(&SCOPES, value) {
                table.images = scope;
            }
        }
        _ => {}
    }
}

fn kind_name(entry: &ExportEntry) -> &'static str {
    match entry {
        ExportEntry::Table(_) => "TABLE",
        ExportEntry::PlateLayout(_) => "PLATE_LAYOUT",
        ExportEntry::DrawnImages(_) => "DRAWN_IMAGES",
        ExportEntry::Script(_) => "SCRIPT",
    }
}

/// "2 tables · drawn images · script".
fn summary(export: &ExportSettings) -> String {
    let count = |kind: &str| {
        export
            .entries
            .iter()
            .filter(|e| kind_name(e) == kind)
            .count()
    };
    let part = |n: usize, one: &str, many: &str| match n {
        0 => None,
        1 => Some(one.to_string()),
        n => Some(format!("{n} {many}")),
    };
    let parts: Vec<String> = [
        part(count("TABLE"), "a table", "tables"),
        part(count("PLATE_LAYOUT"), "plate layout", "plate layouts"),
        part(count("DRAWN_IMAGES"), "drawn images", "drawn images"),
        part(count("SCRIPT"), "script", "scripts"),
    ]
    .into_iter()
    .flatten()
    .collect();
    if parts.is_empty() {
        "nothing yet".into()
    } else {
        parts.join(" · ")
    }
}

fn rows_text(rows: TableRows) -> &'static str {
    match rows {
        TableRows::Object => "object",
        TableRows::Image => "image",
        TableRows::Well => "well",
        TableRows::Folder => "folder",
        TableRows::ColocPair => "coloc pair",
        TableRows::Track => "track",
        TableRows::Measurement => "measurement",
    }
}

fn entry_row(export: &ExportSettings, entry: &ExportEntry) -> ExportEntryRow {
    let file = naming::entry_name(entry)
        .map(|name| naming::with_prefix(export, &name))
        .unwrap_or_default();
    let (title, subtitle, format) = match entry {
        ExportEntry::Table(table) => {
            let ext = match table.format {
                TableFileFormat::Csv => "csv",
                TableFileFormat::Xlsx => "xlsx",
            };
            let mut subtitle = vec![format!("A row per {}", rows_text(table.rows))];
            subtitle.push(match &table.columns {
                None => "default columns".into(),
                Some(c) if c.len() == 1 => "1 column".into(),
                Some(c) => format!("{} columns", c.len()),
            });
            if matches!(
                table.rows,
                TableRows::Image | TableRows::Well | TableRows::Folder
            ) {
                let aggregations: Vec<&str> = table
                    .aggregations
                    .iter()
                    .filter_map(|a| AGGREGATIONS.iter().find(|(b, ..)| b == a))
                    .map(|(_, key, _)| *key)
                    .collect();
                subtitle.push(aggregations.join(", "));
            }
            if table.transposed {
                subtitle.push("classes side by side".into());
            }
            if table.split != TableSplit::None {
                subtitle.push(format!(
                    "per {}",
                    match table.split {
                        TableSplit::Image => "image",
                        TableSplit::Well => "well",
                        TableSplit::Folder => "folder",
                        _ => "class",
                    }
                ));
            }
            (
                format!("Table · {file}.{ext}"),
                subtitle.join(" · "),
                ext.to_uppercase(),
            )
        }
        ExportEntry::PlateLayout(plate) => (
            format!("Plate layout · {file}.xlsx"),
            match plate.grid {
                PlateGrid::Plate => "The plate's grid".to_string(),
                PlateGrid::Well => "The well's grid".to_string(),
            },
            "XLSX".to_string(),
        ),
        ExportEntry::DrawnImages(drawn) => (
            format!("Drawn images · {file}_<image>.png"),
            format!(
                "{} · up to {} px",
                match drawn.style {
                    DrawnStyle::Outline => "Outlines",
                    DrawnStyle::Filled => "Filled",
                    DrawnStyle::Box => "Boxes",
                },
                drawn.max_side
            ),
            "PNG".to_string(),
        ),
        ExportEntry::Script(script) => (
            format!(
                "Script · {}",
                if script.file.is_empty() {
                    "none picked"
                } else {
                    script.file.as_str()
                }
            ),
            format!("Writes into {}/", naming::script_output(script)),
            match script.language {
                ScriptLanguage::R => "R".to_string(),
                ScriptLanguage::Python => "PY".to_string(),
            },
        ),
    };
    ExportEntryRow {
        kind: kind_name(entry).into(),
        title: title.into(),
        subtitle: subtitle.into(),
        format: format.into(),
    }
}

/// What keeps it from running, line by line ("" nothing) - its warnings
/// too, marked as such.
fn problems_text(export: &ExportSettings) -> String {
    check_export(export)
        .iter()
        .map(|problem| {
            let at = problem
                .entry
                .map(|i| format!("{}. ", i + 1))
                .unwrap_or_default();
            match problem.severity {
                Severity::Error => format!("{at}{}", problem.message),
                Severity::Warning => format!("{at}{} (a warning)", problem.message),
            }
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// A run's report: its lines, the log of a script that didn't write, the
/// number of entries that failed.
fn report_lines(export: &ExportSettings, report: &ExportReport) -> (Vec<String>, String, usize) {
    let mut lines = vec![format!("Into {}", report.folder.display())];
    let mut log = String::new();
    let mut failed = 0;
    for entry in &report.entries {
        let label = export
            .entries
            .get(entry.entry)
            .map(|e| entry_row(export, e).title.to_string())
            .unwrap_or_default();
        let n = entry.entry + 1;
        match &entry.status {
            EntryStatus::Written => lines.push(format!(
                "{n}. {label}: {} file{}",
                entry.files.len(),
                if entry.files.len() == 1 { "" } else { "s" }
            )),
            EntryStatus::Skipped(why) => lines.push(format!("{n}. {label}: skipped - {why}")),
            EntryStatus::Failed(why) => {
                failed += 1;
                lines.push(format!("{n}. {label}: failed - {why}"));
            }
        }
        for note in &entry.notes {
            lines.push(format!("    {note}"));
        }
        if !matches!(entry.status, EntryStatus::Written) && !entry.log.is_empty() {
            let tail: Vec<&str> = entry.log.lines().rev().take(20).collect();
            log = tail.into_iter().rev().collect::<Vec<_>>().join("\n");
        }
    }
    (lines, log, failed)
}

#[cfg(test)]
mod tests;
