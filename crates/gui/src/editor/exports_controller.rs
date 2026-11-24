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
use crate::{
    AppWindow, ExportEntryRow, ExportRow, ExportsState, MultiSelectItem, QuickExportState,
    ScriptTrustState,
};
use evanalyzer_app::exports::jobs::{Job, TableExport, TableRows, jobs_of};
use evanalyzer_app::exports::{
    EntryStatus, ExportReport, ScriptPermission, check::check_export, default_columns,
    is_fixed_column, naming, prepare_run, results_database,
};
use evanalyzer_app::modules::Severity;
use evanalyzer_app::results::view_settings::ViewSettings;
use evanalyzer_app::results::{Column, ColumnEntry};
use evanalyzer_cfg::core_types::ObjectClass;
use evanalyzer_cfg::settings::classification_settings::Class;
use evanalyzer_cfg::settings::export_settings::TimeCourseOptions;
use evanalyzer_cfg::settings::export_settings::{
    DataExport, DataOutput, DrawnBrightness, DrawnStyle, ExportAggregation, ExportEntry,
    ExportSettings, HeatmapColors, IfExists, ImageScope, LengthUnit, OutputLocation, Planes,
    ProfileContent, ProfileLineKind, ScriptLanguage, SpeedUnit, SplitInto, TableFileFormat,
    TableSplit, TimeCourseLineKind, TimeCourseTableLayout,
};
use evanalyzer_cfg::settings::project_settings::ResultsView;
use slint::{ComponentHandle, ModelRc, SharedString, VecModel};
use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

/// The models' orders - the indexes the view's segments and selects use.
/// A data export's outputs as the tab lists them: (output, key, text,
/// group) - the old export dialog's ticks.
/// What a table output writes, as the Results List's "A row per" says it.
fn output_tip(output: DataOutput) -> &'static str {
    match output {
        DataOutput::Objects => "A row per object",
        DataOutput::Images => "A row per image and class",
        DataOutput::Wells => {
            "A row per well and class - the wells from the image names (the project's plate grouping)"
        }
        DataOutput::Folders => "A row per folder of images and class",
        DataOutput::ColocPairs => "A row per pair of overlapping objects",
        DataOutput::Measurements => "A row per value of the measurements table, with its unit",
        DataOutput::Tracks => "A row per track",
        _ => "",
    }
}

const OUTPUTS: [(DataOutput, &str, &str, &str); 13] = [
    (DataOutput::Objects, "OBJECTS", "Objects", "A ROW PER"),
    (DataOutput::Images, "IMAGES", "Images", "A ROW PER"),
    (DataOutput::Wells, "WELLS", "Wells", "A ROW PER"),
    (DataOutput::Folders, "FOLDERS", "Folders", "A ROW PER"),
    (
        DataOutput::ColocPairs,
        "COLOC_PAIRS",
        "Coloc pairs",
        "A ROW PER",
    ),
    (
        DataOutput::Measurements,
        "MEASUREMENTS",
        "Measurements",
        "A ROW PER",
    ),
    (DataOutput::Tracks, "TRACKS", "Tracks", "A ROW PER"),
    (
        DataOutput::TimeCourse,
        "TIME_COURSE",
        "Time course - per time point",
        "OVER TIME AND RADIUS",
    ),
    (
        DataOutput::Profile,
        "PROFILE",
        "Profile - per radius (e.g. Sholl)",
        "OVER TIME AND RADIUS",
    ),
    (
        DataOutput::Plate,
        "PLATE",
        "Plate heatmap",
        "PLATE AND WELLS (XLSX)",
    ),
    (
        DataOutput::WellGrid,
        "WELL_GRID",
        "Wells as heatmap",
        "PLATE AND WELLS (XLSX)",
    ),
    (
        DataOutput::WellList,
        "WELL_LIST",
        "Wells as list (a row per image position)",
        "PLATE AND WELLS (XLSX)",
    ),
    (
        DataOutput::ImageHeatmap,
        "IMAGE_HEATMAP",
        "Image heatmap",
        "PLATE AND WELLS (XLSX)",
    ),
];
const SPEEDS: [SpeedUnit; 3] = [SpeedUnit::PxFrame, SpeedUnit::UmMin, SpeedUnit::UmS];
const SQUARES: [u32; 4] = [32, 64, 128, 256];
const COLORS: [HeatmapColors; 11] = [
    HeatmapColors::Excel,
    HeatmapColors::Viridis,
    HeatmapColors::Plasma,
    HeatmapColors::Inferno,
    HeatmapColors::Cividis,
    HeatmapColors::Coolwarm,
    HeatmapColors::RedBlue,
    HeatmapColors::YlGnBu,
    HeatmapColors::Haline,
    HeatmapColors::Algae,
    HeatmapColors::Thermal,
];
const TIME_LINES: [TimeCourseLineKind; 4] = [
    TimeCourseLineKind::Class,
    TimeCourseLineKind::Image,
    TimeCourseLineKind::Well,
    TimeCourseLineKind::Track,
];
const TIME_LAYOUTS: [TimeCourseTableLayout; 3] = [
    TimeCourseTableLayout::Long,
    TimeCourseTableLayout::Wide,
    TimeCourseTableLayout::Both,
];
const PROFILE_LINES: [ProfileLineKind; 4] = [
    ProfileLineKind::Class,
    ProfileLineKind::Image,
    ProfileLineKind::Well,
    ProfileLineKind::All,
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
    /// Per export its last run: a part per analysis it ran on.
    last_runs: BTreeMap<u32, LastRun>,
    /// The analyses the Run ▾ lists: (results file, name, when, ticked).
    run_on: Vec<(PathBuf, String, String, bool)>,
    /// The exports running after an analysis (the note over the window).
    after: AfterRun,
    /// Scripts asked about before a run (the trust prompt).
    trust: Option<TrustAsk>,
    /// The scripts answered "Don't run" (as the exports name them) - the
    /// report says so.
    refused: std::collections::HashSet<String>,
    /// The view "Export this view" is open for (its id).
    quick_view: Option<u32>,
    /// The folder its Export wrote into (the done step).
    quick_written: Option<PathBuf>,
    /// The run going on: its cancel flag, how far (text, 0..1), the
    /// export (by hand: its id and name).
    running: Option<Arc<AtomicBool>>,
    running_id: Option<u32>,
    running_name: String,
    progress: String,
    fraction: f32,
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

/// An export's last run (design canvas "Quick export, open folder, run on",
/// board 4 L1): when, a part per analysis, a failed script's output.
#[derive(Clone, Default)]
struct LastRun {
    time: String,
    parts: Vec<RunPart>,
    log: String,
}

/// What a run did on one analysis: its name and age, all written or not,
/// a few words, where it wrote (on this computer), the report's lines,
/// unfolded.
#[derive(Clone, Default)]
struct RunPart {
    name: String,
    when: String,
    ok: bool,
    summary: String,
    folder: Option<PathBuf>,
    lines: Vec<String>,
    open: bool,
}

/// The analysis a run is on: its name and age as the Results list says
/// them ("" when not known yet - the newest), its results file (`None`:
/// the newest).
#[derive(Clone)]
struct Target {
    name: String,
    when: String,
    database: Option<PathBuf>,
}

/// A script an export runs that the user hasn't allowed yet.
struct Untrusted {
    export: String,
    /// As the export names it: `scripts/stats.R`.
    file: String,
    path: PathBuf,
    hash: String,
    content: String,
    language: ScriptLanguage,
}

/// The scripts still to ask about before a run, those allowed for it once,
/// and the run to start then.
struct TrustAsk {
    queue: Vec<Untrusted>,
    once: Vec<String>,
    then: Box<dyn FnOnce(Vec<String>) + Send>,
}

/// The exports running after an analysis: the note's text, its tone (0
/// running, 1 done, 2 a problem), the export to open, the cancel flag.
#[derive(Default)]
struct AfterRun {
    shown: bool,
    text: String,
    tone: i32,
    open: Option<u32>,
    cancel: Option<Arc<AtomicBool>>,
    /// The folder the exports wrote into (Open folder) - on this computer.
    folder: Option<PathBuf>,
}

/// What the column picker is made of: the results' own columns and classes.
#[derive(Clone)]
struct PickedFrom {
    raw: Vec<ColumnEntry>,
    classes: Vec<Class>,
    /// The profile measurements (key, name) - what a profile table picks.
    profiles: Vec<(String, String)>,
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

    /// The links Relate Objects made (parent, child class ids) and the
    /// parent classes, from the results' columns.
    fn links(&self) -> (Vec<(u32, u32)>, Vec<u32>) {
        let mut links = Vec::new();
        let mut parents = Vec::new();
        for entry in &self.raw {
            match &entry.key {
                Column::Children(c) if c.value == evanalyzer_app::results::ChildValue::Count => {
                    if let (Some(p), Some(c)) = (c.parent_class.to_u32(), c.child_class.to_u32()) {
                        links.push((p, c));
                    }
                }
                Column::Parent(p) => {
                    if let Some(id) = p.parent_class.to_u32()
                        && !parents.contains(&id)
                    {
                        parents.push(id);
                    }
                }
                _ => {}
            }
        }
        (links, parents)
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

/// A data export as one table, for the settings the tab shares with a
/// table and for the column picker: its first table output's rows (the
/// plate's grouped rows without one - its default column the count), the
/// data export's shared settings.
fn stand_in(data: &DataExport) -> TableExport {
    let first = jobs_of(data).into_iter().find_map(|(_, job)| match job {
        Job::Table(table) => Some(table),
        Job::PlateLayout(_) => None,
    });
    let mut table = first.unwrap_or_else(|| {
        let mut table: TableExport =
            serde_json::from_value(serde_json::json!({ "rows": "WELL" })).expect("a table");
        table.columns = Some(
            data.columns
                .clone()
                .unwrap_or_else(|| vec!["count".to_string()]),
        );
        table
    });
    table.name = String::new();
    table.classes = data.classes.clone();
    table.aggregations = data.aggregations.clone();
    table.empty_classes = data.empty_classes;
    table.transposed = data.transposed;
    table.tracks = data.tracks.clone();
    table.time_course = data.time_course.clone();
    table.profile = data.profile.clone();
    table.planes = data.planes;
    table
}

/// What an edit of `stand_in(data)` changed, into `data`.
fn write_back(data: &mut DataExport, before: &TableExport, after: TableExport) {
    macro_rules! changed {
        ($($field:ident),*) => {
            $(if after.$field != before.$field {
                data.$field = after.$field;
            })*
        };
    }
    changed!(
        format,
        classes,
        columns,
        aggregations,
        images,
        empty_classes,
        transposed,
        children,
        child_aggregations,
        parents,
        split,
        split_into,
        tracks,
        time_course,
        profile,
        planes
    );
}

/// The outputs that sum values up (the Values tab), and those with a row per
/// image, well or folder (classes without objects).
fn grouping(data: &DataExport) -> (bool, bool) {
    let values = data.outputs.iter().any(|o| {
        !matches!(
            o,
            DataOutput::Objects
                | DataOutput::ColocPairs
                | DataOutput::Measurements
                | DataOutput::Profile
        )
    });
    let empty = data.outputs.iter().any(|o| {
        matches!(
            o,
            DataOutput::Images | DataOutput::Wells | DataOutput::Folders
        )
    });
    (values, empty)
}

/// The columns offered for a table of `rows`: never the leading ones; a
/// table of measurements has measurements only.
fn offered_for(rows: TableRows, mut offered: Vec<ColumnEntry>) -> Vec<ColumnEntry> {
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
    /// The newest analysis' links (parent, child class ids) and parent
    /// classes - for All of Children and Linked objects.
    links: Vec<(u32, u32)>,
    parent_classes: Vec<u32>,
    /// The newest analysis' profile measurements (key, name).
    profiles: Vec<(String, String)>,
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
        state.on_move_export(move |id, by| this.move_export(id, by));

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
                    links: Vec::new(),
                    parent_classes: Vec::new(),
                    profiles: Vec::new(),
                }
            };
            let pickable = {
                let mut pickable = pickable;
                if let Some(from) = &this.shown.lock().expect("Poisoned").picked_from {
                    (pickable.links, pickable.parent_classes) = from.links();
                    pickable.profiles = from.profiles.clone();
                }
                pickable
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
        state.on_run_part_toggle(move |index| this.toggle_run_part(index));
        let this = self.clone();
        state.on_run_on_open(move || this.run_on_open());
        let this = self.clone();
        state.on_run_on_pick(move |index, on| this.run_on_pick(index, on));
        let this = self.clone();
        state.on_run_on_picked(move || this.run_on_picked());
        let this = self.clone();
        state.on_open_folder(move |path| {
            let local = !this.app_state.backend().is_remote();
            if local && !path.is_empty() {
                open_in_system(Path::new(path.as_str()));
            }
        });

        let this = self.clone();
        state.on_cancel(move || {
            if let Some(cancel) = &this.shown.lock().expect("Poisoned").running {
                cancel.store(true, Ordering::Relaxed);
            }
        });

        // An analysis finished: the exports set to run afterwards.
        let this = self.clone();
        self.app_state
            .on_analysis_finished(move |output_dir| this.run_after(output_dir));

        let this = self.clone();
        state.on_after_close(move || {
            if let Some(cancel) = &this.shown.lock().expect("Poisoned").after.cancel {
                cancel.store(true, Ordering::Relaxed);
            }
            this.shown.lock().expect("Poisoned").after.shown = false;
            this.push_after();
        });

        // "Open folder": where the exports wrote.
        let this = self.clone();
        state.on_after_open_folder(move || {
            let folder = this.shown.lock().expect("Poisoned").after.folder.clone();
            if let Some(folder) = folder {
                open_in_system(&folder);
            }
        });

        // "Open in Exports": the export with the problem (else the one
        // picked) and its report.
        let this = self.clone();
        state.on_after_open(move || {
            {
                let mut shown = this.shown.lock().expect("Poisoned");
                if let Some(id) = shown.after.open {
                    shown.selected = Some(id);
                    shown.entry = None;
                }
                shown.after.shown = false;
            }
            this.push();
            this.push_after();
        });

        // Results › Export…, a view's ⋯ › Save as export.
        let quick = ui.global::<QuickExportState>();
        let this = self.clone();
        quick.on_open(move |view| this.quick_open(view));
        let this = self.clone();
        quick.on_save_view(move |view| this.quick_save_view(view));
        let this = self.clone();
        quick.on_export(move || this.quick_export());
        let this = self.clone();
        quick.on_add_to_project(move || this.quick_add());
        let this = self.clone();
        quick.on_open_folder(move || {
            let written = this.shown.lock().expect("Poisoned").quick_written.clone();
            if let Some(folder) = written {
                open_in_system(&folder);
            }
        });
        let this = self.clone();
        quick.on_cancel(move || {
            let mut shown = this.shown.lock().expect("Poisoned");
            shown.quick_view = None;
            shown.quick_written = None;
            drop(shown);
            if let Some(ui) = this.ui.upgrade() {
                ui.global::<QuickExportState>().set_shown(false);
            }
        });
        let weak = ui.as_weak();
        let files = self.app_state.file_browser.clone();
        quick.on_choose_folder(move || {
            let Some(ui) = weak.upgrade() else {
                return;
            };
            let start = ui.global::<QuickExportState>().get_folder().to_string();
            let weak = weak.clone();
            let request =
                crate::FileRequest::open_folder("Export into").start_in(PathBuf::from(start));
            files.open(request, move |folder| {
                if let (Some(folder), Some(ui)) = (folder, weak.upgrade()) {
                    ui.global::<QuickExportState>()
                        .set_folder(folder.to_string_lossy().as_ref().into());
                }
            });
        });

        // The trust prompt.
        let trust = ui.global::<ScriptTrustState>();
        let this = self.clone();
        trust.on_answer(move |answer| this.trust_answer(answer));
        let this = self.clone();
        trust.on_open_in_editor(move || {
            let path = this
                .shown
                .lock()
                .expect("Poisoned")
                .trust
                .as_ref()
                .and_then(|ask| ask.queue.first())
                .map(|script| script.path.clone());
            if let Some(path) = path {
                open_in_system(&path);
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
            export.entries.push(new_entry("DATA"));
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

    /// Moves the export `id` by `by` places (the list's and the runs'
    /// order); at an end it stays.
    fn move_export(&self, id: i32, by: i32) {
        let Ok(id) = u32::try_from(id) else {
            return;
        };
        let (from, count) = {
            let project = self.app_state.get_project();
            let exports = &project.settings.exports;
            let Some(from) = exports.iter().position(|e| e.id == id) else {
                return;
            };
            (from, exports.len())
        };
        let to = from as i64 + i64::from(by);
        if to < 0 || to >= count as i64 {
            return;
        }
        self.change_exports(|exports| {
            let export = exports.remove(from);
            exports.insert(to as usize, export);
        });
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
        self.run_on(vec![Target {
            name: String::new(),
            when: String::new(),
            database: None,
        }]);
    }

    /// The Run ▾ opened: the analyses of the Results list (newest first),
    /// the newest ticked - nothing remembered from before.
    fn run_on_open(&self) {
        let Some(ui) = self.ui.upgrade() else {
            return;
        };
        let list = ui.global::<crate::ResultsListState>().get_results_list();
        let rows: Vec<(PathBuf, String, String, bool)> = slint::Model::iter(&list)
            .enumerate()
            .map(|(i, row)| {
                (
                    PathBuf::from(row.path.as_str()),
                    row.name.to_string(),
                    row.modified.to_string(),
                    i == 0,
                )
            })
            .collect();
        self.shown.lock().expect("Poisoned").run_on = rows;
        self.push();
    }

    /// A tick in the Run ▾ (`index`), or All / Newest only (`index` < 0:
    /// `on` all, else the newest alone).
    fn run_on_pick(&self, index: i32, on: bool) {
        {
            let mut shown = self.shown.lock().expect("Poisoned");
            match usize::try_from(index) {
                Ok(i) => {
                    if let Some(row) = shown.run_on.get_mut(i) {
                        row.3 = on;
                    }
                }
                Err(_) => {
                    for (i, row) in shown.run_on.iter_mut().enumerate() {
                        row.3 = on || i == 0;
                    }
                }
            }
        }
        self.push();
    }

    /// "Run on N analyses": the export on each ticked one, newest first.
    fn run_on_picked(self: &Arc<Self>) {
        let picked: Vec<Target> = self
            .shown
            .lock()
            .expect("Poisoned")
            .run_on
            .iter()
            .filter(|row| row.3)
            .map(|(path, name, when, _)| Target {
                name: name.clone(),
                when: when.clone(),
                database: Some(path.clone()),
            })
            .collect();
        if picked.is_empty() {
            return;
        }
        self.run_on(picked);
    }

    /// Runs the export being edited on `analyses`, one after the other, in
    /// the background.
    fn run_on(self: &Arc<Self>, analyses: Vec<Target>) {
        let Some(id) = self.selected() else {
            return;
        };
        if self.shown.lock().expect("Poisoned").running.is_some() {
            return;
        }
        // One at a time: no export while an analysis runs (its own
        // exports are its last step).
        let analysing = self.ui.upgrade().is_some_and(|ui| {
            let running = ui.global::<crate::PipelineRunningState>();
            running.get_active() && (!running.get_done() || running.get_exporting())
        });
        if analysing {
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
        // A script not allowed yet: asked first.
        let this = self.clone();
        let exports = vec![export.clone()];
        self.ask_scripts(&project_dir.clone(), &exports, move |once| {
            this.start_run(id, export, project_dir, analyses, once);
        });
    }

    /// Runs `export` by hand on `analyses`, the scripts allowed `once`
    /// with them.
    fn start_run(
        self: &Arc<Self>,
        id: u32,
        export: ExportSettings,
        project_dir: PathBuf,
        analyses: Vec<Target>,
        once: Vec<String>,
    ) {
        let cancel = Arc::new(AtomicBool::new(false));
        {
            let mut shown = self.shown.lock().expect("Poisoned");
            shown.running = Some(cancel.clone());
            shown.running_id = Some(id);
            shown.running_name = format!("“{}”", export.name);
            shown.progress = "Starting…".into();
            shown.fraction = 0.0;
            shown.last_runs.remove(&id);
        }
        self.push();

        let this = self.clone();
        ui_thread::spawn(move || {
            let backend = this.app_state.backend();
            let total = analyses.len();
            let mut results = Vec::with_capacity(total);
            for (i, mut target) in analyses.into_iter().enumerate() {
                if cancel.load(Ordering::Relaxed) {
                    break;
                }
                let at = if total > 1 {
                    format!("{}/{total} · {} · ", i + 1, target.name)
                } else {
                    String::new()
                };
                let progress_to = this.clone();
                let result =
                    results_database(backend.files(), &project_dir, target.database.as_deref())
                        .map_err(|e| e.to_string())
                        .and_then(|database| {
                            if target.name.is_empty() {
                                target.name = analysis_name(&database);
                            }
                            this.run_one(
                                &project_dir,
                                &export,
                                &database,
                                &once,
                                &cancel,
                                move |text, done| {
                                    let this = progress_to.clone();
                                    let text = format!("{at}{text}");
                                    let fraction = (i as f32 + done) / total as f32;
                                    ui_thread::invoke_from_event_loop(move || {
                                        let mut shown = this.shown.lock().expect("Poisoned");
                                        shown.progress = text;
                                        shown.fraction = fraction;
                                        drop(shown);
                                        this.push();
                                    })
                                    .ok();
                                },
                            )
                        });
                results.push((target, result.map(|r| (export.clone(), r))));
            }
            ui_thread::invoke_from_event_loop(move || this.finish_many(id, results)).ok();
        });
    }

    /// Runs `export` on the results `database` (here or on the server) -
    /// its report; `progress` gets what it is at. The scripts the user
    /// allowed run, the others are skipped (said in the report).
    fn run_one(
        &self,
        project_dir: &Path,
        export: &ExportSettings,
        database: &Path,
        once: &[String],
        cancel: &AtomicBool,
        progress: impl Fn(String, f32),
    ) -> Result<ExportReport, String> {
        self.run_one_into(project_dir, export, database, None, once, cancel, progress)
    }

    /// [`Self::run_one`] into `output` (`None`: the export's own folder).
    #[allow(clippy::too_many_arguments)]
    fn run_one_into(
        &self,
        project_dir: &Path,
        export: &ExportSettings,
        database: &Path,
        output: Option<PathBuf>,
        once: &[String],
        cancel: &AtomicBool,
        progress: impl Fn(String, f32),
    ) -> Result<ExportReport, String> {
        let backend = self.app_state.backend();
        let mut trusted = backend
            .load_app_settings()
            .unwrap_or_default()
            .trusted_scripts;
        trusted.extend(once.iter().cloned());
        let (results, run) = {
            let project = self.app_state.get_project();
            prepare_run(
                backend.as_ref(),
                &project,
                project_dir,
                export.clone(),
                database,
                output,
                ScriptPermission::Trusted(trusted),
            )
            .map_err(|e| e.to_string())?
        };
        results
            .run_export(&run, cancel, &mut |message, current, total| {
                let text = if current < total {
                    format!("{}/{total} · {message}", current + 1)
                } else {
                    message.to_string()
                };
                let done = if total == 0 {
                    0.0
                } else {
                    current.min(total) as f32 / total as f32
                };
                progress(text, done);
            })
            .map_err(|e| e.to_string())
    }

    /// An analysis finished into `output_dir`: the exports left on in the
    /// run dialog (else those set to run after analysis) run on its
    /// results, one after the other, in the background - the analysis'
    /// last step: its progress says "Exporting", its note waits for them
    /// (design canvas "Quick export, open folder, run on", board 6).
    fn run_after(self: &Arc<Self>, output_dir: &Path) {
        let wanted = self.app_state.take_exports_for_run();
        let exports: Vec<ExportSettings> = self
            .app_state
            .get_project()
            .settings
            .exports
            .iter()
            .filter(|export| match &wanted {
                Some(ids) => ids.contains(&export.id),
                None => export.run_after_every_run,
            })
            .cloned()
            .collect();
        if exports.is_empty() {
            return;
        }
        let Some(project_dir) = self.project_dir() else {
            return;
        };
        let output_dir = output_dir.to_path_buf();
        let this = self.clone();
        // On the UI thread, before the analysis says it is done (queued
        // first): its last step starts; a script not allowed yet is asked.
        ui_thread::invoke_from_event_loop(move || {
            if let Some(ui) = this.ui.upgrade() {
                let running = ui.global::<crate::PipelineRunningState>();
                running.set_exporting(true);
                running.set_export_text("starting".into());
                running.set_export_fraction(0.0);
                running.set_export_note("".into());
                running.set_export_problem(false);
            }
            let run = this.clone();
            let asked = exports.clone();
            this.ask_scripts(&project_dir.clone(), &asked, move |once| {
                run.start_after(exports, project_dir, output_dir, once);
            });
        })
        .ok();
    }

    /// The analysis' export step as far as it is (on the UI thread).
    fn set_phase(&self, text: String, fraction: f32) {
        let weak = self.ui.clone();
        ui_thread::invoke_from_event_loop(move || {
            if let Some(ui) = weak.upgrade() {
                let running = ui.global::<crate::PipelineRunningState>();
                running.set_export_text(text.into());
                running.set_export_fraction(fraction);
            }
        })
        .ok();
    }

    /// Runs `exports` after an analysis into `output_dir`, the scripts
    /// allowed `once` with them; the Exports tab is locked meanwhile (its
    /// Stop stops them, as the analysis' does).
    fn start_after(
        self: &Arc<Self>,
        exports: Vec<ExportSettings>,
        project_dir: PathBuf,
        output_dir: PathBuf,
        once: Vec<String>,
    ) {
        let cancel = Arc::new(AtomicBool::new(false));
        {
            let mut shown = self.shown.lock().expect("Poisoned");
            shown.running = Some(cancel.clone());
            shown.progress = "After the analysis · starting".into();
            shown.fraction = 0.0;
        }
        self.push();
        let this = self.clone();
        ui_thread::spawn(move || {
            let backend = this.app_state.backend();
            let total = exports.len();
            let mut failed = 0;
            let mut files = 0;
            let mut first_problem = None;
            // Where they wrote (the first that did): for Open folder - on
            // this computer only.
            let mut folder: Option<PathBuf> = None;
            let local = !backend.is_remote();
            let mut cant_run = None;
            match results_database(backend.files(), &project_dir, Some(&output_dir)) {
                Err(e) => cant_run = Some(e.to_string()),
                Ok(database) => {
                    let analysis = analysis_name(&database);
                    for (i, export) in exports.into_iter().enumerate() {
                        if cancel.load(Ordering::Relaxed) {
                            break;
                        }
                        let name = export.name.clone();
                        let counter = if total > 1 {
                            format!(" · {}/{total}", i + 1)
                        } else {
                            String::new()
                        };
                        let at = format!("“{name}”{counter}");
                        this.set_phase(at.clone(), i as f32 / total as f32);
                        let phase_to = this.clone();
                        let result = this.run_one(
                            &project_dir,
                            &export,
                            &database,
                            &once,
                            &cancel,
                            move |text, done| {
                                phase_to.set_phase(
                                    format!("{at} · {text}"),
                                    (i as f32 + done) / total as f32,
                                );
                            },
                        );
                        let id = export.id;
                        let problem = match &result {
                            Err(_) => true,
                            Ok(report) => {
                                let written =
                                    report.entries.iter().map(|e| e.files.len()).sum::<usize>();
                                if written > 0 && local {
                                    folder.get_or_insert_with(|| report.folder.clone());
                                }
                                files += written;
                                !report.all_written()
                            }
                        };
                        if problem {
                            failed += 1;
                            first_problem.get_or_insert(id);
                        }
                        let this2 = this.clone();
                        let analysis = analysis.clone();
                        ui_thread::invoke_from_event_loop(move || {
                            this2.record(id, analysis, result.map(|r| (export, r)));
                        })
                        .ok();
                    }
                }
            }
            let cancelled = cancel.load(Ordering::Relaxed);
            let (note, problem) = if let Some(why) = cant_run {
                (format!("⚠ exports didn't run: {why}"), true)
            } else if cancelled {
                ("⚠ exports stopped".to_string(), true)
            } else if failed > 0 {
                (
                    format!("⚠ {failed} of {total} exports with a problem"),
                    true,
                )
            } else {
                (
                    format!(
                        "{total} {} written · {files} {}",
                        if total == 1 { "export" } else { "exports" },
                        if files == 1 { "file" } else { "files" }
                    ),
                    false,
                )
            };
            let this2 = this.clone();
            ui_thread::invoke_from_event_loop(move || {
                {
                    let mut shown = this2.shown.lock().expect("Poisoned");
                    shown.running = None;
                    shown.progress.clear();
                    // "Open in Exports" of the note: the export with the
                    // problem; "Open folder": where they wrote.
                    shown.after = AfterRun {
                        open: first_problem,
                        folder,
                        ..AfterRun::default()
                    };
                }
                this2.push();
                if let Some(ui) = this2.ui.upgrade() {
                    let running = ui.global::<crate::PipelineRunningState>();
                    running.set_export_note(note.into());
                    running.set_export_problem(problem);
                    running.set_exporting(false);
                }
            })
            .ok();
        });
    }

    /// The view `view_id` (`< 0`: the one shown) and the analysis open in
    /// Results (its database), if any.
    fn view_and_results(&self, view_id: i32) -> Option<(ResultsView, Option<PathBuf>)> {
        let project = self.app_state.get_project();
        let views = &project.settings.results_views;
        let wanted = u32::try_from(view_id).unwrap_or(views.active);
        let view = views.views.iter().find(|v| v.id == wanted)?.clone();
        drop(project);
        let database = self.ui.upgrade().and_then(|ui| {
            let list = ui.global::<crate::ResultsListState>();
            let index = usize::try_from(list.get_selected_result_index()).ok()?;
            slint::Model::row_data(&list.get_results_list(), index)
                .map(|row| PathBuf::from(row.path.as_str()))
        });
        Some((view, database))
    }

    /// Opens "Export this view" for `view_id` (`< 0`: the one shown).
    fn quick_open(self: &Arc<Self>, view_id: i32) {
        let Some(ui) = self.ui.upgrade() else {
            return;
        };
        let Some((view, database)) = self.view_and_results(view_id) else {
            return;
        };
        // What it shows, from the view alone (its columns named once it
        // runs on the analysis).
        let Some(entry) = evanalyzer_app::exports::entry_from_view(&view) else {
            return;
        };
        let export = new_export(0, view.name.clone());
        let state = ui.global::<QuickExportState>();
        state.set_view_name(view.name.as_str().into());
        state.set_what(entry_row(&export, &entry).subtitle);
        state.set_table(matches!(&entry, ExportEntry::Data(data)
            if data.outputs.iter().any(|o| !matches!(o,
                DataOutput::Plate | DataOutput::WellGrid | DataOutput::WellList | DataOutput::ImageHeatmap))));
        state.set_format(0);
        // Beside the analysis, unless chosen before.
        if state.get_folder().is_empty() {
            let folder = database
                .as_ref()
                .and_then(|db| db.parent())
                .map(|dir| dir.to_string_lossy().into_owned())
                .unwrap_or_default();
            state.set_folder(folder.into());
        }
        state.set_file("".into());
        // A name in front of the automatic one(s).
        state.set_file_placeholder(
            naming::entry_names(&entry)
                .iter()
                .map(|name| naming::with_prefix(&export, name))
                .collect::<Vec<_>>()
                .join(", ")
                .into(),
        );
        state.set_done(false);
        state.set_can_open_folder(!self.app_state.backend().is_remote());
        state.set_busy(false);
        state.set_error(
            if database.is_none() {
                "Open an analysis first - this view is written from it."
            } else {
                ""
            }
            .into(),
        );
        {
            let mut shown = self.shown.lock().expect("Poisoned");
            shown.quick_view = Some(view.id);
            shown.quick_written = None;
        }
        state.set_shown(true);
    }

    /// The view's entry as the dialog sets it: the name in front, the
    /// format; a profile chart without one chosen shows the first profile
    /// of `database`.
    fn quick_entry(
        &self,
        view: &ResultsView,
        database: Option<&Path>,
        file: &str,
        xlsx: bool,
    ) -> Option<ExportEntry> {
        let mut entry = evanalyzer_app::exports::entry_from_view(view)?;
        if let ExportEntry::Data(data) = &mut entry {
            if let Some(profile) = &mut data.profile
                && profile.key.is_empty()
                && let Some(database) = database
            {
                profile.key = self
                    .app_state
                    .backend()
                    .open_results(database)
                    .and_then(|results| results.profile_columns())
                    .ok()
                    .and_then(|columns| columns.into_iter().next())
                    .map(|column| column.key)
                    .unwrap_or_default();
            }
            data.name = file.to_string();
            if xlsx {
                data.format = TableFileFormat::Xlsx;
            }
        }
        Some(entry)
    }

    /// "Add to project exports": the view as a new export (named like the
    /// view) - nothing written; closes with a note to open it.
    fn quick_add(self: &Arc<Self>) {
        let Some(ui) = self.ui.upgrade() else {
            return;
        };
        let state = ui.global::<QuickExportState>();
        let Some(view_id) = self.shown.lock().expect("Poisoned").quick_view else {
            return;
        };
        let Some((view, database)) = self.view_and_results(view_id as i32) else {
            return;
        };
        let file = state.get_file().trim().to_string();
        let xlsx = state.get_format() == 1;
        let Some(entry) = self.quick_entry(&view, database.as_deref(), &file, xlsx) else {
            state.set_error("This view can't be exported.".into());
            return;
        };
        let id = self.keep_entry(entry, 0, "", &view.name, false);
        let name = self
            .app_state
            .get_project()
            .settings
            .exports
            .iter()
            .find(|e| e.id == id)
            .map(|e| e.name.clone())
            .unwrap_or_default();
        {
            let mut shown = self.shown.lock().expect("Poisoned");
            shown.quick_view = None;
            shown.quick_written = None;
        }
        state.set_shown(false);
        self.set_after(move |note| {
            *note = AfterRun {
                shown: true,
                text: format!("Added to Exports as “{name}”"),
                tone: 1,
                open: Some(id),
                cancel: None,
                folder: None,
            };
        });
    }

    /// Exports the view now; the done step shows what was written.
    fn quick_export(self: &Arc<Self>) {
        let Some(ui) = self.ui.upgrade() else {
            return;
        };
        let state = ui.global::<QuickExportState>();
        let Some(view_id) = self.shown.lock().expect("Poisoned").quick_view else {
            return;
        };
        let Some((view, Some(database))) = self.view_and_results(view_id as i32) else {
            state.set_error("Open an analysis first - this view is written from it.".into());
            return;
        };
        let folder = state.get_folder().trim().to_string();
        if folder.is_empty() {
            state.set_error("Choose the folder to write into.".into());
            return;
        }
        let Some(project_dir) = self.project_dir() else {
            state.set_error("Save the project first.".into());
            return;
        };
        let file = state.get_file().trim().to_string();
        let xlsx = state.get_format() == 1;
        state.set_busy(true);
        state.set_error("".into());

        let this = self.clone();
        ui_thread::spawn(move || {
            let result = (|| {
                let entry = this
                    .quick_entry(&view, Some(&database), &file, xlsx)
                    .ok_or("this view can't be exported")?;
                let mut export = new_export(0, view.name.clone());
                // A file of that name already there stays: the new one
                // gets "_2".
                export.output.if_exists = IfExists::Rename;
                export.entries.push(entry);
                if let Some(problem) = check_export(&export)
                    .into_iter()
                    .find(|p| p.severity == Severity::Error)
                {
                    return Err(problem.message);
                }
                let cancel = AtomicBool::new(false);
                let report = this.run_one_into(
                    &project_dir,
                    &export,
                    &database,
                    Some(PathBuf::from(&folder)),
                    &[],
                    &cancel,
                    |_, _| {},
                )?;
                if !report.all_written() {
                    let (lines, ..) = report_lines(&export, &report, &Default::default());
                    return Err(lines.join(" · "));
                }
                Ok(report)
            })();
            ui_thread::invoke_from_event_loop(move || {
                this.quick_done(result.map_err(|e| e.to_string()));
            })
            .ok();
        });
    }

    /// "Export this view" wrote its files (or didn't): the done step (or
    /// the problem in the form).
    fn quick_done(self: &Arc<Self>, result: Result<ExportReport, String>) {
        let Some(ui) = self.ui.upgrade() else {
            return;
        };
        let state = ui.global::<QuickExportState>();
        state.set_busy(false);
        let report = match result {
            Ok(report) => report,
            Err(why) => {
                state.set_error(why.into());
                return;
            }
        };
        let files: Vec<&PathBuf> = report.entries.iter().flat_map(|e| &e.files).collect();
        let (text, path) = match files.as_slice() {
            [one] => (
                format!(
                    "Written: {}",
                    one.file_name().unwrap_or(one.as_os_str()).to_string_lossy()
                ),
                report.folder.join(one),
            ),
            many => (
                format!("Written: {} files", many.len()),
                report.folder.clone(),
            ),
        };
        state.set_done_text(text.into());
        state.set_done_path(path.to_string_lossy().as_ref().into());
        state.set_done(true);
        self.shown.lock().expect("Poisoned").quick_written = Some(report.folder);
    }

    /// Keeps `entry` in the project: as a new export (`target` 0, named
    /// `name`, else the view's) or added to the `target`-th export.
    /// Its id.
    fn keep_entry(
        &self,
        entry: ExportEntry,
        target: usize,
        name: &str,
        view_name: &str,
        after: bool,
    ) -> u32 {
        self.change_exports(|exports| {
            if let Some(export) = target.checked_sub(1).and_then(|i| exports.get_mut(i)) {
                export.entries.push(entry);
                return export.id;
            }
            let id = exports.iter().map(|e| e.id).max().map_or(1, |max| max + 1);
            let wanted = if name.is_empty() { view_name } else { name };
            let mut export = new_export(id, free_export_name(exports, wanted));
            export.run_after_every_run = after;
            export.entries.push(entry);
            exports.push(export);
            id
        })
    }

    /// Shows the export `id` in the Exports tab.
    fn open_export(&self, id: u32) {
        {
            let mut shown = self.shown.lock().expect("Poisoned");
            shown.selected = Some(id);
            shown.entry = Some(0);
        }
        if let Some(ui) = self.ui.upgrade() {
            ui.global::<crate::LayoutState>()
                .set_mode(crate::WorkspaceMode::Exports);
        }
        self.push();
    }

    /// A view's ⋯ › Save as export…: a new export of it (named like the
    /// view), opened in the Exports tab - nothing written now.
    fn quick_save_view(self: &Arc<Self>, view_id: i32) {
        let Some((view, _)) = self.view_and_results(view_id) else {
            return;
        };
        if let Some(entry) = evanalyzer_app::exports::entry_from_view(&view) {
            let id = self.keep_entry(entry, 0, "", &view.name, false);
            self.open_export(id);
        }
    }

    /// The scripts `exports` run that the user hasn't allowed yet (by their
    /// content), each once.
    fn untrusted_scripts(&self, project_dir: &Path, exports: &[ExportSettings]) -> Vec<Untrusted> {
        let backend = self.app_state.backend();
        let trusted = backend
            .load_app_settings()
            .unwrap_or_default()
            .trusted_scripts;
        let mut found: Vec<Untrusted> = Vec::new();
        for export in exports {
            for entry in &export.entries {
                let ExportEntry::Script(script) = entry else {
                    continue;
                };
                if script.file.is_empty() {
                    continue;
                }
                let path = project_dir.join(&script.file);
                // Missing or unreadable: the run says so.
                let Ok(bytes) = backend.files().read_file(&path) else {
                    continue;
                };
                let hash = evanalyzer_app::exports::script_hash(&bytes);
                if trusted.contains(&hash) || found.iter().any(|u| u.hash == hash) {
                    continue;
                }
                found.push(Untrusted {
                    export: export.name.clone(),
                    file: script.file.clone(),
                    path,
                    hash,
                    content: String::from_utf8_lossy(&bytes).into_owned(),
                    language: script.language,
                });
            }
        }
        found
    }

    /// Asks about each script of `exports` not allowed yet, then calls
    /// `then` with those allowed for this run only (right away when there
    /// is none to ask about). On the UI thread.
    fn ask_scripts(
        self: &Arc<Self>,
        project_dir: &Path,
        exports: &[ExportSettings],
        then: impl FnOnce(Vec<String>) + Send + 'static,
    ) {
        let queue = self.untrusted_scripts(project_dir, exports);
        if queue.is_empty() {
            then(Vec::new());
            return;
        }
        {
            let mut shown = self.shown.lock().expect("Poisoned");
            // Asked already (another run waits): this one follows it.
            if let Some(ask) = shown.trust.take() {
                let earlier = ask.then;
                let mut queue = ask.queue;
                for script in self.untrusted_scripts(project_dir, exports) {
                    if !queue.iter().any(|u| u.hash == script.hash) {
                        queue.push(script);
                    }
                }
                shown.trust = Some(TrustAsk {
                    queue,
                    once: ask.once,
                    then: Box::new(move |once: Vec<String>| {
                        earlier(once.clone());
                        then(once);
                    }),
                });
            } else {
                shown.trust = Some(TrustAsk {
                    queue,
                    once: Vec::new(),
                    then: Box::new(then),
                });
            }
        }
        self.push_trust();
    }

    /// The trust prompt: the first script still to ask about, or none.
    fn push_trust(&self) {
        let Some(ui) = self.ui.upgrade() else {
            return;
        };
        let state = ui.global::<ScriptTrustState>();
        let shown = self.shown.lock().expect("Poisoned");
        let Some(script) = shown.trust.as_ref().and_then(|ask| ask.queue.first()) else {
            state.set_shown(false);
            return;
        };
        let lines = script.content.lines().count();
        let name = Path::new(&script.file)
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_else(|| script.file.clone());
        state.set_name(name.into());
        state.set_export_name(script.export.as_str().into());
        state.set_about(
            format!(
                "{} · {lines} {} · {}",
                script.file,
                if lines == 1 { "line" } else { "lines" },
                match script.language {
                    ScriptLanguage::R => "R",
                    ScriptLanguage::Python => "Python",
                }
            )
            .into(),
        );
        let preview: Vec<&str> = script.content.lines().take(40).collect();
        state.set_preview(preview.join("\n").into());
        state.set_can_open(!self.app_state.backend().is_remote());
        state.set_more(
            shown
                .trust
                .as_ref()
                .map_or(0, |ask| ask.queue.len() as i32 - 1),
        );
        state.set_shown(true);
    }

    /// The answer about the first script: 0 don't run, 1 run once, 2
    /// always allow this version - then the next, or the run.
    fn trust_answer(&self, answer: i32) {
        let done = {
            let mut shown = self.shown.lock().expect("Poisoned");
            let Some(ask) = shown.trust.as_mut() else {
                return;
            };
            if ask.queue.is_empty() {
                None
            } else {
                let script = ask.queue.remove(0);
                if answer == 0 {
                    shown.refused.insert(script.file.clone());
                } else {
                    shown.refused.remove(&script.file);
                }
                let Some(ask) = shown.trust.as_mut() else {
                    return;
                };
                match answer {
                    1 => ask.once.push(script.hash),
                    2 => {
                        let hash = script.hash;
                        self.app_state.update_app_settings(|settings| {
                            if !settings.trusted_scripts.contains(&hash) {
                                settings.trusted_scripts.push(hash.clone());
                            }
                        });
                        ask.once.push(hash);
                    }
                    _ => {}
                }
                if ask.queue.is_empty() {
                    shown.trust.take()
                } else {
                    None
                }
            }
        };
        self.push_trust();
        if let Some(ask) = done {
            (ask.then)(ask.once);
        }
    }

    /// Changes the note of the exports after a run, then shows it (from
    /// any thread).
    fn set_after(self: &Arc<Self>, change: impl FnOnce(&mut AfterRun) + Send + 'static) {
        let this = self.clone();
        ui_thread::invoke_from_event_loop(move || {
            change(&mut this.shown.lock().expect("Poisoned").after);
            this.push_after();
        })
        .ok();
    }

    /// The note of the exports after a run.
    fn push_after(&self) {
        let Some(ui) = self.ui.upgrade() else {
            return;
        };
        let state = ui.global::<ExportsState>();
        let shown = self.shown.lock().expect("Poisoned");
        state.set_after_shown(shown.after.shown);
        state.set_after_text(shown.after.text.as_str().into());
        state.set_after_tone(shown.after.tone);
        state.set_after_busy(shown.after.cancel.is_some());
        state.set_after_folder(shown.after.folder.is_some());
    }

    /// A run of the export `id` ended: its report and status.
    fn finish(&self, id: u32, result: Result<(ExportSettings, ExportReport), String>) {
        self.finish_many(
            id,
            vec![(
                Target {
                    name: String::new(),
                    when: String::new(),
                    database: None,
                },
                result,
            )],
        );
    }

    /// [`Self::finish`] of a run on several analyses.
    fn finish_many(
        &self,
        id: u32,
        results: Vec<(Target, Result<(ExportSettings, ExportReport), String>)>,
    ) {
        let (cancelled, name) = {
            let mut shown = self.shown.lock().expect("Poisoned");
            let cancelled = shown
                .running
                .take()
                .is_some_and(|cancel| cancel.load(Ordering::Relaxed));
            shown.progress.clear();
            shown.running_id = None;
            (cancelled, std::mem::take(&mut shown.running_name))
        };
        // The note, like an analysis' (board 6): how it went, where.
        let total = results.len();
        let files: usize = results
            .iter()
            .filter_map(|(_, r)| r.as_ref().ok())
            .flat_map(|(_, report)| &report.entries)
            .map(|e| e.files.len())
            .sum();
        let problems = results
            .iter()
            .filter(|(_, r)| r.as_ref().map_or(true, |(_, report)| !report.all_written()))
            .count();
        let folder = (!self.app_state.backend().is_remote())
            .then(|| {
                results
                    .iter()
                    .filter_map(|(_, r)| r.as_ref().ok())
                    .find(|(_, report)| report.entries.iter().any(|e| !e.files.is_empty()))
                    .map(|(_, report)| report.folder.clone())
            })
            .flatten();
        let analyses = if total == 1 {
            String::new()
        } else {
            format!(" · {total} analyses")
        };
        let (text, tone) = if cancelled {
            (format!("Export stopped · {name}"), 2)
        } else if problems > 0 {
            (
                format!("Export {name} · {problems} of {total} with a problem"),
                2,
            )
        } else {
            (
                format!(
                    "Export finished · {name}{analyses} · {files} {}",
                    if files == 1 { "file" } else { "files" }
                ),
                1,
            )
        };
        self.shown.lock().expect("Poisoned").after = AfterRun {
            shown: true,
            text,
            tone,
            open: Some(id),
            cancel: None,
            folder,
        };
        self.push_after();
        self.record_many(id, results);
    }

    /// The report and status of a run of the export `id` on the analysis
    /// `name` (after an analysis).
    fn record(
        &self,
        id: u32,
        name: String,
        result: Result<(ExportSettings, ExportReport), String>,
    ) {
        self.record_many(
            id,
            vec![(
                Target {
                    name,
                    when: String::new(),
                    database: None,
                },
                result,
            )],
        );
    }

    /// The last run of the export `id`: a part per analysis (board 4 L1) -
    /// a part that went well folded, one with a problem open.
    fn record_many(
        &self,
        id: u32,
        results: Vec<(Target, Result<(ExportSettings, ExportReport), String>)>,
    ) {
        let local = !self.app_state.backend().is_remote();
        {
            let mut shown = self.shown.lock().expect("Poisoned");
            let time = chrono::Local::now().format("%H:%M").to_string();
            let mut run = LastRun {
                time: time.clone(),
                ..LastRun::default()
            };
            let mut logs = Vec::new();
            for (target, result) in results {
                let mut part = RunPart {
                    name: target.name,
                    when: target.when,
                    ..RunPart::default()
                };
                match result {
                    Err(why) => {
                        part.summary = "did not run".into();
                        part.lines.push(format!("Did not run: {why}"));
                    }
                    Ok((export, report)) => {
                        let (mut lines, log, failed) =
                            report_lines(&export, &report, &shown.refused);
                        let files: usize = report.entries.iter().map(|e| e.files.len()).sum();
                        if local && files > 0 {
                            // The folder button says where: no "Into …" line.
                            lines.remove(0);
                            part.folder = Some(report.folder.clone());
                        }
                        part.ok = failed == 0 && report.all_written();
                        part.summary = if failed > 0 {
                            format!("{failed} failed")
                        } else if !report.all_written() {
                            "not all written".into()
                        } else {
                            format!("{files} {}", if files == 1 { "file" } else { "files" })
                        };
                        part.lines = lines;
                        if !log.is_empty() {
                            logs.push(log);
                        }
                    }
                }
                part.open = !part.ok;
                run.parts.push(part);
            }
            run.log = logs.join("\n");
            let problems = run.parts.iter().filter(|p| !p.ok).count();
            let status = match (run.parts.as_slice(), problems) {
                ([one], 1) if one.summary == "did not run" => ("⚠ did not run".to_string(), 2),
                (_, 0) => (format!("✓ {time}"), 1),
                (_, n) => (format!("⚠ {n} failed · {time}"), 2),
            };
            shown.last_runs.insert(id, run);
            shown.status.insert(id, status);
        }
        self.push();
    }

    /// A part of the last run (its index) folded or unfolded.
    fn toggle_run_part(&self, index: i32) {
        {
            let mut shown = self.shown.lock().expect("Poisoned");
            let Some(id) = shown.selected else {
                return;
            };
            if let Some(part) = shown.last_runs.get_mut(&id).and_then(|run| {
                run.parts
                    .get_mut(usize::try_from(index).unwrap_or(usize::MAX))
            }) {
                part.open = !part.open;
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
                let (status, tone) = match shown.running_id == Some(export.id) {
                    true => (format!("● {}%", (shown.fraction * 100.0).round() as i32), 0),
                    false => shown.status.get(&export.id).cloned().unwrap_or_default(),
                };
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
        state.set_after_count(
            settings
                .exports
                .iter()
                .filter(|e| e.run_after_every_run)
                .count() as i32,
        );
        state.set_exports(model(rows));
        state.set_selected_id(shown.selected.map_or(-1, |id| id as i32));
        state.set_running(shown.running.is_some());
        state.set_progress(shown.progress.as_str().into());
        state.set_progress_fraction(shown.fraction);
        state.set_run_name(shown.running_name.as_str().into());
        let run = shown
            .selected
            .and_then(|id| shown.last_runs.get(&id))
            .cloned()
            .unwrap_or_default();
        let parts = run.parts.len();
        state.set_last_run_title(
            match run.parts.as_slice() {
                [] => String::new(),
                [one] if one.name.is_empty() => format!("Last run · {}", run.time),
                [one] => format!("Last run · {} · {}", run.time, one.name),
                _ => format!("Last run · {} · {parts} analyses", run.time),
            }
            .into(),
        );
        let problems = run.parts.iter().filter(|p| !p.ok).count();
        state.set_last_run_note(
            if parts < 2 {
                String::new()
            } else if problems == 0 {
                "all written".into()
            } else {
                format!("{} written · {problems} with a problem", parts - problems)
            }
            .into(),
        );
        state.set_run_parts(model(
            run.parts
                .iter()
                .map(|part| crate::RunPart {
                    name: part.name.as_str().into(),
                    when: part.when.as_str().into(),
                    ok: part.ok,
                    summary: part.summary.as_str().into(),
                    folder: part
                        .folder
                        .as_ref()
                        .map(|f| f.to_string_lossy().into_owned())
                        .unwrap_or_default()
                        .into(),
                    lines: model(part.lines.iter().map(SharedString::from).collect()),
                    open: part.open || parts == 1,
                })
                .collect(),
        ));
        state.set_log(run.log.as_str().into());
        state.set_run_on(model(
            shown
                .run_on
                .iter()
                .map(|(path, name, when, on)| crate::AnalysisRow {
                    path: path.to_string_lossy().as_ref().into(),
                    name: name.as_str().into(),
                    when: when.as_str().into(),
                    on: *on,
                })
                .collect(),
        ));
        state.set_run_on_ticked(shown.run_on.iter().filter(|row| row.3).count() as i32);

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
            ExportEntry::Data(data) => {
                let table = &stand_in(data);
                state.set_outputs(model(
                    OUTPUTS
                        .iter()
                        .enumerate()
                        .map(|(i, (output, key, text, group))| crate::DataOutputRow {
                            key: (*key).into(),
                            text: (*text).into(),
                            tip: output_tip(*output).into(),
                            // The group's caption above its first output.
                            group: if i == 0 || OUTPUTS[i - 1].3 != *group {
                                (*group).into()
                            } else {
                                "".into()
                            },
                            on: data.outputs.contains(output),
                        })
                        .collect(),
                ));
                state.set_can_transpose(data.outputs.iter().any(|o| {
                    matches!(
                        o,
                        DataOutput::Objects
                            | DataOutput::Images
                            | DataOutput::Wells
                            | DataOutput::Folders
                    )
                }));
                state.set_any_heatmap(data.outputs.iter().any(|o| {
                    matches!(
                        o,
                        DataOutput::Plate | DataOutput::WellGrid | DataOutput::ImageHeatmap
                    )
                }));
                let (values, empty) = grouping(data);
                state.set_grouped(values);
                state.set_show_empty(empty);
                state.set_square(index_of(&SQUARES, &data.heatmap.square_size.unwrap_or(64)));
                state.set_colors(index_of(&COLORS, &data.heatmap.colors));
                let tracks = data.tracks.clone().unwrap_or_default();
                state.set_tracks_min_points(tracks.min_points.to_string().into());
                state.set_tracks_from_start(tracks.from_start);
                state.set_tracks_no_edge(tracks.no_edge);
                state.set_tracks_speed(index_of(&SPEEDS, &tracks.speed_unit));
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
                push_planes(&state, &table.planes);
                let course = table.time_course.clone().unwrap_or_default();
                state.set_time_lines(index_of(&TIME_LINES, &course.lines));
                state.set_time_layout(index_of(&TIME_LAYOUTS, &course.layout));
                state.set_time_relative(course.relative);
                state.set_time_with_parents(course.with_parents);
                let profile = table.profile.clone().unwrap_or_default();
                state.set_profile_lines(index_of(&PROFILE_LINES, &profile.lines));
                state.set_profile_objects(profile.content == ProfileContent::Objects);
                let profiles = shown
                    .picked_from
                    .as_ref()
                    .map(|from| from.profiles.clone())
                    .unwrap_or_default();
                state.set_profile_names(model(
                    profiles
                        .iter()
                        .map(|(_, name)| SharedString::from(name.as_str()))
                        .collect(),
                ));
                state.set_profile_name(
                    profiles
                        .iter()
                        .find(|(key, _)| *key == profile.key)
                        .map_or(profile.key.as_str(), |(_, name)| name.as_str())
                        .into(),
                );
                push_picker(&state, table, &shown);
                state.set_columns_default(data.columns.is_none());
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
            ExportEntry::DrawnImages(drawn) => {
                push_planes(&state, &drawn.planes);
                state.set_images(index_of(&SCOPES, &drawn.images));
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
                    profiles: results
                        .profile_columns()
                        .unwrap_or_default()
                        .into_iter()
                        .map(|column| (column.key.clone(), column.name.clone()))
                        .collect(),
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
        let entry = self
            .app_state
            .get_project()
            .settings
            .exports
            .iter()
            .find(|e| e.id == id)
            .and_then(|e| e.entries.get(index))
            .cloned();
        let table = match entry {
            Some(ExportEntry::Data(data)) => stand_in(&data),
            _ => return,
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
                        .filter(|column| !is_fixed_column(column))
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
        self.edit(id, |export| match export.entries.get_mut(index) {
            Some(ExportEntry::Data(data)) => data.columns = Some(keys),
            _ => {}
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

/// An entry's z-planes and time points: "" all, else "from" and "to".
fn push_planes(state: &ExportsState, planes: &Planes) {
    let text = |range: Option<[u32; 2]>, i: usize| {
        range
            .map_or(String::new(), |range| range[i].to_string())
            .into()
    };
    state.set_z_range(planes.z.is_some());
    state.set_z_from(text(planes.z, 0));
    state.set_z_to(text(planes.z, 1));
    state.set_t_range(planes.t.is_some());
    state.set_t_from(text(planes.t, 0));
    state.set_t_to(text(planes.t, 1));
}

/// A plane range setting: "all", or "<from>-<to>" (both from 0).
fn set_planes(planes: &mut Planes, key: &str, value: &str) {
    let range = if value == "all" {
        None
    } else {
        let Some((from, to)) = value.split_once('-') else {
            return;
        };
        let (Ok(from), Ok(to)) = (from.trim().parse::<u32>(), to.trim().parse::<u32>()) else {
            return;
        };
        Some([from.min(to), from.max(to)])
    };
    match key {
        "z-planes" => planes.z = range,
        "t-planes" => planes.t = range,
        _ => {}
    }
}

/// `section` with the columns every table writes anyway
/// ([`is_fixed_column`]) ticked and greyed: not to untick.
fn lock_fixed(mut section: SectionUi, offered: &[ColumnEntry], classes: &[Class]) -> SectionUi {
    let keys: Vec<String> = offered
        .iter()
        .filter(|entry| is_fixed_column(&entry.key))
        .map(|entry| entry.key.as_key(classes))
        .collect();
    section.locked = section
        .items
        .iter()
        .map(|item| keys.iter().any(|key| item.key.as_str() == key))
        .collect();
    if !section.locked.iter().any(|locked| *locked) {
        section.locked.clear();
    }
    section
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
        state.set_related_children(model(Vec::new()));
        state.set_related_parents(model(Vec::new()));
        return;
    };
    let (selected, missing) = from.selection(table);
    let filter = shown.column_filter.to_lowercase();
    let source = from.source(&shown.coloc, &shown.sections_open, &filter);
    let offered = offered_for(table.rows, source.offered(&selected));
    // The fixed columns count as picked in the sections' titles; the
    // starting sets compare without them (they're never stored).
    let mut shown_selected = selected.clone();
    shown_selected.extend(
        offered
            .iter()
            .map(|entry| &entry.key)
            .filter(|column| is_fixed_column(column) && !selected.contains(column))
            .cloned(),
    );
    let mut parts = source.parts(&offered, &shown_selected);
    let unfixed: Vec<ColumnEntry> = offered
        .iter()
        .filter(|entry| !is_fixed_column(&entry.key))
        .cloned()
        .collect();
    parts.presets = source.presets(&unfixed, &selected);
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
            .map(|section| lock_fixed(section, &offered, &from.classes))
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
    // Children and linked objects.
    let related = picker::related_items(
        &from.raw,
        &from.classes,
        &table
            .children
            .iter()
            .map(|(p, c)| (ObjectClass::Global(*p), ObjectClass::Global(*c)))
            .collect::<Vec<_>>(),
        &table
            .child_aggregations
            .iter()
            .map(|a| results_aggregation(*a))
            .collect::<Vec<_>>(),
        &table
            .parents
            .iter()
            .map(|p| ObjectClass::Global(*p))
            .collect::<Vec<_>>(),
    );
    state.set_related_children(model(related.children));
    state.set_related_children_summary(related.children_summary.into());
    state.set_related_aggregations(model(related.aggregations));
    state.set_related_aggregate_summary(related.aggregate_summary);
    state.set_related_parents(model(related.parents));
    state.set_related_parent_summary(related.parent_summary.into());
}

/// An export's aggregation as the results name it, and back.
fn results_aggregation(aggregation: ExportAggregation) -> evanalyzer_app::results::Aggregation {
    use evanalyzer_app::results::Aggregation;
    match aggregation {
        ExportAggregation::Avg => Aggregation::Avg,
        ExportAggregation::Median => Aggregation::Median,
        ExportAggregation::Stddev => Aggregation::Stddev,
        ExportAggregation::Sum => Aggregation::Sum,
        ExportAggregation::Min => Aggregation::Min,
        ExportAggregation::Max => Aggregation::Max,
        ExportAggregation::Skewness => Aggregation::Skewness,
    }
}

fn export_aggregation_of(aggregation: &evanalyzer_app::results::Aggregation) -> ExportAggregation {
    use evanalyzer_app::results::Aggregation;
    match aggregation {
        Aggregation::Avg => ExportAggregation::Avg,
        Aggregation::Median => ExportAggregation::Median,
        Aggregation::Stddev => ExportAggregation::Stddev,
        Aggregation::Sum => ExportAggregation::Sum,
        Aggregation::Min => ExportAggregation::Min,
        Aggregation::Max => ExportAggregation::Max,
        Aggregation::Skewness => ExportAggregation::Skewness,
    }
}

/// Opens `path` with the system's program for it (a script: the editor).
fn open_in_system(path: &Path) {
    #[cfg(target_os = "linux")]
    let _ = std::process::Command::new("xdg-open").arg(path).spawn();
    #[cfg(target_os = "macos")]
    let _ = std::process::Command::new("open").arg(path).spawn();
    #[cfg(target_os = "windows")]
    let _ = std::process::Command::new("cmd")
        .args(["/C", "start", ""])
        .arg(path)
        .spawn();
}

/// An analysis' name as the Results list shows it, from its results file.
fn analysis_name(database: &Path) -> String {
    crate::editor::results_list_controller::extract_name_from_path(&database.to_path_buf())
        .unwrap_or_default()
        .to_string()
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
        "DRAWN_IMAGES" => serde_json::json!({ "kind": "DRAWN_IMAGES" }),
        "SCRIPT" => serde_json::json!({ "kind": "SCRIPT", "language": "R", "file": "" }),
        _ => serde_json::json!({ "kind": "DATA", "outputs": ["IMAGES"] }),
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
        ExportEntry::Data(d) => Some(&mut d.classes),
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
    if key == "z-planes" || key == "t-planes" {
        match entry {
            ExportEntry::Data(d) => set_planes(&mut d.planes, key, value),
            ExportEntry::DrawnImages(d) => set_planes(&mut d.planes, key, value),
            ExportEntry::Script(_) => {}
        }
        return;
    }
    if key == "name" {
        let name = value.trim().to_string();
        match entry {
            ExportEntry::Data(d) => d.name = name,
            ExportEntry::DrawnImages(d) => d.name = name,
            ExportEntry::Script(_) => {}
        }
        return;
    }
    match entry {
        ExportEntry::Data(data) => set_data_field(data, key, value, pickable),
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

/// A setting of a data export: its outputs, heatmaps and tracks here - the
/// rest as a table's, on its shared settings.
fn set_data_field(data: &mut DataExport, key: &str, value: &str, pickable: &Pickable) {
    if let Some(name) = key.strip_prefix("output:") {
        let Some((output, ..)) = OUTPUTS.iter().find(|(_, k, ..)| *k == name) else {
            return;
        };
        data.outputs.retain(|o| o != output);
        if value == "1" {
            data.outputs.push(*output);
            // In the list's order, not the clicks'.
            data.outputs
                .sort_by_key(|o| OUTPUTS.iter().position(|(p, ..)| p == o));
        }
        return;
    }
    let tracks =
        |data: &mut DataExport| -> evanalyzer_cfg::settings::export_settings::TrackOptions {
            data.tracks.clone().unwrap_or_default()
        };
    match key {
        "square" => {
            if let Some(side) = pick(&SQUARES, value) {
                data.heatmap.square_size = Some(side);
            }
            return;
        }
        "colors" => {
            if let Some(colors) = pick(&COLORS, value) {
                data.heatmap.colors = colors;
            }
            return;
        }
        "tracks-min-points" => {
            if let Ok(n) = value.trim().parse::<u32>() {
                let mut t = tracks(data);
                t.min_points = n.max(1);
                data.tracks = Some(t);
            }
            return;
        }
        "tracks-from-start" | "tracks-no-edge" => {
            let mut t = tracks(data);
            if key == "tracks-from-start" {
                t.from_start = value == "1";
            } else {
                t.no_edge = value == "1";
            }
            data.tracks = Some(t);
            return;
        }
        "tracks-speed" => {
            if let Some(unit) = pick(&SPEEDS, value) {
                let mut t = tracks(data);
                t.speed_unit = unit;
                data.tracks = Some(t);
            }
            return;
        }
        _ => {}
    }
    let before = stand_in(data);
    let mut after = before.clone();
    set_table_field(&mut after, key, value, pickable);
    write_back(data, &before, after);
}

fn set_table_field(table: &mut TableExport, key: &str, value: &str, pickable: &Pickable) {
    let views = &pickable.views;
    if let Some(link) = key.strip_prefix("child:") {
        let Some((p, c)) = link
            .split_once(':')
            .and_then(|(p, c)| Some((p.parse::<u32>().ok()?, c.parse::<u32>().ok()?)))
        else {
            return;
        };
        table.children.retain(|l| *l != (p, c));
        if value == "1" {
            table.children.push((p, c));
            table.children.sort_unstable();
        }
        return;
    }
    if let Some(name) = key.strip_prefix("child-aggregation:") {
        let Some(aggregation) = crate::editor::results_state_controller::aggregation_from_key(name)
        else {
            return;
        };
        let aggregation = export_aggregation_of(&aggregation);
        table.child_aggregations.retain(|a| *a != aggregation);
        if value == "1" {
            table.child_aggregations.push(aggregation);
        }
        return;
    }
    if let Some(id) = key.strip_prefix("parent:") {
        let Ok(id) = id.parse::<u32>() else {
            return;
        };
        table.parents.retain(|p| *p != id);
        if value == "1" {
            table.parents.push(id);
            table.parents.sort_unstable();
        }
        return;
    }
    match key {
        "children-all" => {
            table.children = pickable.links.clone();
            return;
        }
        "children-none" => {
            table.children.clear();
            return;
        }
        "parents-all" => {
            table.parents = pickable.parent_classes.clone();
            return;
        }
        "parents-none" => {
            table.parents.clear();
            return;
        }
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
    fn course(table: &mut TableExport) -> &mut TimeCourseOptions {
        table.time_course.get_or_insert_with(Default::default)
    }
    fn profile(
        table: &mut TableExport,
    ) -> &mut evanalyzer_cfg::settings::export_settings::ProfileOptions {
        table.profile.get_or_insert_with(Default::default)
    }
    match key {
        "time-lines" => {
            if let Some(lines) = pick(&TIME_LINES, value) {
                course(table).lines = lines;
            }
            return;
        }
        "time-layout" => {
            if let Some(layout) = pick(&TIME_LAYOUTS, value) {
                course(table).layout = layout;
            }
            return;
        }
        "time-relative" => {
            course(table).relative = value == "1";
            return;
        }
        "time-with-parents" => {
            course(table).with_parents = value == "1";
            return;
        }
        "profile-lines" => {
            if let Some(lines) = pick(&PROFILE_LINES, value) {
                profile(table).lines = lines;
            }
            return;
        }
        "profile-objects" => {
            profile(table).content = if value == "1" {
                ProfileContent::Objects
            } else {
                ProfileContent::Lines
            };
            return;
        }
        "profile-pick" => {
            if let Some((key, _)) = value
                .parse::<usize>()
                .ok()
                .and_then(|i| pickable.profiles.get(i))
            {
                profile(table).key = key.clone();
            }
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
        ExportEntry::Data(_) => "DATA",
        ExportEntry::DrawnImages(_) => "DRAWN_IMAGES",
        ExportEntry::Script(_) => "SCRIPT",
    }
}

/// "2 tables · drawn images · script".
pub(crate) fn summary(export: &ExportSettings) -> String {
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
        part(count("DATA"), "a data export", "data exports"),
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

fn entry_row(export: &ExportSettings, entry: &ExportEntry) -> ExportEntryRow {
    let files: Vec<String> = naming::entry_names(entry)
        .iter()
        .map(|name| naming::with_prefix(export, name))
        .collect();
    let file = files.first().cloned().unwrap_or_default();
    let (title, subtitle, format) = match entry {
        ExportEntry::Data(data) => {
            let ext = match data.format {
                TableFileFormat::Csv => "csv",
                TableFileFormat::Xlsx => "xlsx",
            };
            let named: Vec<String> = jobs_of(data)
                .iter()
                .map(|(_, job)| match job {
                    Job::Table(_) => format!(
                        "{}.{ext}",
                        naming::with_prefix(export, &naming::job_name(job))
                    ),
                    Job::PlateLayout(_) => format!(
                        "{}.xlsx",
                        naming::with_prefix(export, &naming::job_name(job))
                    ),
                })
                .collect();
            let title = match named.len() {
                0 => "Data · no output ticked".to_string(),
                1 => format!("Data · {}", named[0]),
                n => format!("Data · {n} outputs"),
            };
            let mut subtitle = Vec::new();
            if named.len() > 1 {
                subtitle.push(named.join(" · "));
            }
            subtitle.push(match &data.columns {
                None => "default columns".into(),
                Some(c) if c.len() == 1 => "1 column".into(),
                Some(c) => format!("{} columns", c.len()),
            });
            if grouping(data).0 {
                let aggregations: Vec<&str> = data
                    .aggregations
                    .iter()
                    .filter_map(|a| AGGREGATIONS.iter().find(|(b, ..)| b == a))
                    .map(|(_, key, _)| *key)
                    .collect();
                subtitle.push(aggregations.join(", "));
            }
            if data.classes.is_none() {
                subtitle.push("all classes".into());
            }
            if data.transposed
                && data.outputs.iter().any(|o| {
                    matches!(
                        o,
                        DataOutput::Objects
                            | DataOutput::Images
                            | DataOutput::Wells
                            | DataOutput::Folders
                    )
                })
            {
                subtitle.push("classes side by side".into());
            }
            if let Some([from, to]) = data.planes.t {
                subtitle.push(format!("time points {from}-{to}"));
            }
            (title, subtitle.join(" · "), ext.to_uppercase())
        }
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
fn report_lines(
    export: &ExportSettings,
    report: &ExportReport,
    refused: &std::collections::HashSet<String>,
) -> (Vec<String>, String, usize) {
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
            // A script not allowed: what to do about it, here in the app.
            EntryStatus::Skipped(why) if why == evanalyzer_app::exports::NOT_ALLOWED => {
                let chosen = match export.entries.get(entry.entry) {
                    Some(ExportEntry::Script(script)) => refused.contains(&script.file),
                    _ => false,
                };
                lines.push(if chosen {
                    format!(
                        "{n}. {label}: not run - you chose \"Don't run\". Run the export again to be asked."
                    )
                } else {
                    format!("{n}. {label}: not run - not allowed yet. Run the export to be asked.")
                });
            }
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
