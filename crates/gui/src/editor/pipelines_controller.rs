use crate::AppWindow;
use crate::DialogType;
use crate::FileRequest;
use crate::editor::focus_controller::FocusController;
use crate::editor::object_list_controller::ObjectListController;
use crate::editor::pipeline_task::PipelineTask;
use crate::editor::template_controller::TemplateController;
use crate::editor::viewport_controller::ViewportController;
use crate::{
    CardTable, CardTableRow, ClassItemData, CommandDef, CommandParameter, CommandPickerState,
    ComponentBindingRow, ComponentClassRow, ComponentClassesState, ComponentEditChoiceState,
    ComponentParamRow, ComponentSettingsState, ComponentUpdateState, CreateComponentState,
    DefaultParamPreview, ExposeChoiceState, GlobalAppState, GroupItem, LeafParam, ParamType,
    Pipeline, PipelineCommand as SlintPipelineCommand, PipelineStatus, PipelinesPanelState,
    ProblemItem, RunAnalysisState, SectionView, StepCategory, UiState, WarningState,
};
use crate::{PipelineDeleteConfirmState, PipelineEditState, PipelineRunningState};
use crate::{ResultItemData, ResultsListState};
use evanalyzer_app::ai_learning::load_classifier_settings;
use evanalyzer_app::components::ClassRole;
use evanalyzer_app::fs::FileSystem;
use evanalyzer_app::project::ImageEntryExt;
use evanalyzer_app::project::ProjectExt;
use evanalyzer_app::templates::load_component_templates;
use evanalyzer_cfg::core_types::ComponentParamKey;
use evanalyzer_cfg::core_types::MemorySlot;
use evanalyzer_cfg::core_types::ObjectClass;
use evanalyzer_cfg::core_types::PipelineId;
use evanalyzer_cfg::core_types::SegmentationClass;
use evanalyzer_cfg::core_types::StepId;
use evanalyzer_cfg::core_types::{ImageAddress, MemoryId};
use evanalyzer_cfg::settings::ai_learning_settings::{
    AiLearningClassifierSettings, AiLearningSettings, ObjectClassLabel, PixelClassLabel,
    PixelInputColor,
};
use evanalyzer_cfg::settings::components::{ComponentDefinition, ParamValue};
use evanalyzer_cfg::settings::images_settings::{
    ImageEntry, ImageSettings, TStackHandling, TStackSettings,
};
use evanalyzer_cfg::settings::meta_data::AuthorInformation;
use evanalyzer_cfg::settings::parameter_def::{ParamType as CfgParamType, ParameterDef};
use evanalyzer_cfg::settings::pipeline_command::CommandMeta;
use evanalyzer_cfg::settings::pipeline_command::PipelineCommand;
use evanalyzer_cfg::settings::pipeline_command::{
    CommandCategory, all_command_meta, default_command,
};
use evanalyzer_cfg::settings::pipeline_command_settings::{
    AiObjectClassifierSettings, ClassificationMappingSettings, PixelClassifierSettings,
    SegmentationMappingSettings,
};
use evanalyzer_cfg::settings::pipeline_settings::{
    PipelineSettings, PipelineStepSettings, StepContent,
};
use evanalyzer_cfg::settings::project_settings::ProjectSettings;
use evanalyzer_cfg::settings::templates::ComponentTemplate;
use evanalyzer_gui_slint::CommandAuthor;
use log::debug;
use log::info;
use log::warn;
use slint::{ComponentHandle, Model, ModelRc, SharedString, VecModel};
use std::path::Path;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::{Condvar, Mutex};

/// Quiet period (in ms) the user must pause editing before an auto preview
/// runs. Resets on every parameter change. See `pipeline_settings_changed`.
const PREVIEW_DEBOUNCE_MS: u64 = 400;

thread_local! {
    /// Single-shot debounce timer for auto preview execution.
    ///
    /// Kept in a thread-local rather than on `PipelinesController` because
    /// `slint::Timer` is `!Send`/`!Sync`, while the controller is shared with
    /// the pipeline worker thread. `pipeline_settings_changed` only runs on the
    /// Slint event-loop thread, so the thread-local is always valid there.
    static PREVIEW_DEBOUNCE: slint::Timer = slint::Timer::default();
}

pub struct PipelinesController {
    pub(crate) ui: slint::Weak<AppWindow>,
    pub(crate) app_state: Arc<UiState>,
    pub(crate) _object_list_controller: Arc<ObjectListController>,
    pub(crate) viewport_controller: Arc<ViewportController>,
    pub(crate) template_controller: Arc<TemplateController>,
    pub(crate) task_request: Arc<(Mutex<Option<PipelineTask>>, Condvar)>,
    pub(crate) pipeline_cancel_flag: Arc<Mutex<Option<evanalyzer_app::analysis::CancelHandle>>>,
    /// An analysis (not a preview) is running - closing the window asks
    /// what to do with it.
    pub(crate) analysis_running: Arc<std::sync::atomic::AtomicBool>,
    /// Currently active breakpoint: (pipeline_id, step_id, mode).  `None` = no breakpoint.
    pub(crate) breakpoint: Arc<
        Mutex<
            Option<(
                u32,
                evanalyzer_cfg::core_types::StepId,
                evanalyzer_cfg::core_types::BreakpointMode,
            )>,
        >,
    >,

    /// If true the trigger_pipeline_preview_execution is called on parameter change
    auto_preview_enabled: Mutex<bool>,

    /// Pipeline focus, told about every pipeline change (see
    /// `pipeline_settings_changed`).
    pub(crate) focus_controller: Arc<FocusController>,

    /// The components opened in the panel ("Edit"/"View" on a component card):
    /// (pipeline id, the path from the component opened from the pipeline to
    /// the one shown - a component opened inside an opened one goes on top).
    /// The panel shows and edits the last one's steps - its definition, so
    /// every use of the component - instead of the pipeline's; "‹ Back" goes
    /// one level up.
    open_component: Mutex<Option<(u32, Vec<OpenComponent>)>>,

    /// While "Edit for all / Only this step" asks: the opened components the
    /// asked-about card's step index refers to (see [`Self::edit_component`]).
    pending_edit_base: Mutex<Option<Vec<OpenComponent>>>,

    /// A setting waiting for "Expose setting": (pipeline id, step index,
    /// setting path, the parameters it could be added to).
    pending_expose: Mutex<Option<(u32, usize, String, Vec<ComponentParamKey>)>>,

    /// The component library: `.evacomp` files of the templates folders,
    /// reloaded whenever the picker is opened.
    component_library: Mutex<Vec<(PathBuf, ComponentTemplate)>>,

    /// A component waiting for "Update": (its key, index into `component_library`).
    pending_update: Mutex<Option<(String, usize)>>,

    /// The opened component as it was when a parameter was opened to edit in
    /// its settings dialog (✎) - what Cancel goes back to.
    parameter_edit_snapshot: Mutex<Option<ComponentDefinition>>,

    /// What the checks found when the panel was last filled - the problems
    /// `show-problem`/`fix-problem` refer to by their index.
    last_problems: Mutex<Vec<evanalyzer_app::components::ComponentProblem>>,

    /// Debug mode (toolbar): settings show what they're wired to.
    debug_mode: std::sync::atomic::AtomicBool,

    /// The open sections of component cards: (pipeline, step id, section).
    /// The others show their values on one line.
    open_sections: Mutex<std::collections::HashSet<(u32, String, String)>>,
}

/// One level of [`PipelinesController::open_component`].
#[derive(Debug, Clone)]
struct OpenComponent {
    key: String,
    /// Only looked at ("View" on a component card): the index of the card it was
    /// opened from in the level below, for "Edit" in the bar. Nothing can be
    /// changed meanwhile - also not in components opened inside it.
    viewing_from: Option<usize>,
}

impl PipelinesController {
    pub fn new(
        ui: slint::Weak<AppWindow>,
        app_state: Arc<UiState>,
        object_list_controller: Arc<ObjectListController>,
        viewport_controller: Arc<ViewportController>,
        template_controller: Arc<TemplateController>,
        focus_controller: Arc<FocusController>,
    ) -> Self {
        Self {
            ui,
            app_state: app_state.clone(),
            _object_list_controller: object_list_controller,
            viewport_controller,
            template_controller,
            task_request: Arc::new((Mutex::new(None), Condvar::new())),
            pipeline_cancel_flag: Arc::new(Mutex::new(None)),
            analysis_running: Arc::default(),
            breakpoint: Arc::new(Mutex::new(None)),
            auto_preview_enabled: Mutex::new(false),
            focus_controller,
            open_component: Mutex::new(None),
            pending_edit_base: Mutex::new(None),
            pending_expose: Mutex::new(None),
            component_library: Mutex::new(Vec::new()),
            pending_update: Mutex::new(None),
            parameter_edit_snapshot: Mutex::new(None),
            last_problems: Mutex::new(Vec::new()),
            debug_mode: std::sync::atomic::AtomicBool::new(false),
            open_sections: Mutex::new(std::collections::HashSet::new()),
        }
    }

    pub fn attach_callbacks(self: &Arc<Self>) {
        let ui_handle = self.ui.clone();
        if let Some(ui) = ui_handle.upgrade() {
            // The remembered "Always show advanced settings"
            ui.global::<PipelinesPanelState>().set_always_show_advanced(
                self.app_state
                    .load_app_settings()
                    .always_show_advanced_settings,
            );

            // Save pipeline as component
            let manager = self.clone();
            ui.global::<PipelinesPanelState>()
                .on_save_as_component(move || {
                    manager.save_pipeline_as_component();
                });

            // Dry run pipeline
            let manager = self.clone();
            ui.global::<PipelinesPanelState>().on_dry_run(move || {
                manager.trigger_pipeline_preview_execution();
            });

            // Auto preview toggled
            let manager = self.clone();
            ui.global::<PipelinesPanelState>()
                .on_auto_preview(move |auto_preview| {
                    *manager.auto_preview_enabled.lock().expect("Poisned") = auto_preview;
                });

            // Set breakpoint (mode: 1=Stop, 2=Snapshot)
            let manager = self.clone();
            ui.global::<PipelinesPanelState>().on_set_breakpoint(
                move |pipeline_id, step_id, mode| {
                    let bp_mode = if mode == 2 {
                        evanalyzer_cfg::core_types::BreakpointMode::Snapshot
                    } else {
                        evanalyzer_cfg::core_types::BreakpointMode::Stop
                    };
                    *manager.breakpoint.lock().unwrap() = parse_step_id(&step_id)
                        .map(|step_id| (pipeline_id as u32, step_id, bp_mode));
                },
            );

            // Clear breakpoint
            let manager = self.clone();
            ui.global::<PipelinesPanelState>()
                .on_clear_breakpoint(move || {
                    *manager.breakpoint.lock().unwrap() = None;
                });

            // Toggle breakpoint image view
            let manager = self.clone();
            ui.global::<PipelinesPanelState>()
                .on_show_breakpoint_image_changed(move |show| {
                    manager.viewport_controller.set_show_breakpoint(show);
                });

            // Breakpoint buffer picker: 0 = processed image, 1 = segmentation,
            // 2 = instances.
            let manager = self.clone();
            ui.global::<PipelinesPanelState>()
                .on_breakpoint_view_mode_changed(move |mode| {
                    manager.viewport_controller.set_breakpoint_view_mode(mode);
                });

            // Full run pipeline - opens the job-name confirm dialog first
            // rather than dispatching immediately (see `open_run_analysis_dialog`).
            let manager = self.clone();
            ui.global::<PipelinesPanelState>().on_run_all(move || {
                manager.open_run_analysis_dialog();
            });

            // Run-analysis dialog confirmed: read back the (optional) job
            // name and actually dispatch the run.
            let manager = self.clone();
            ui.global::<RunAnalysisState>().on_confirm(move || {
                manager.on_run_analysis_confirmed();
            });

            // Run-analysis dialog cancelled: just close it, nothing to run.
            let manager = self.clone();
            ui.global::<RunAnalysisState>().on_cancel(move || {
                if let Some(ui) = manager.ui.upgrade() {
                    ui.global::<GlobalAppState>()
                        .set_active_dialog(DialogType::None);
                }
            });

            // Selected pipeline
            // Expand/collapse a step's advanced settings
            let manager = self.clone();
            ui.global::<PipelinesPanelState>()
                .on_toggle_advanced(move |step_id| {
                    let Some(ui) = manager.ui.upgrade() else {
                        return;
                    };
                    let state = ui.global::<PipelinesPanelState>();
                    let commands = state.get_active_commands();
                    for i in 0..commands.row_count() {
                        if let Some(mut cmd) = commands.row_data(i) {
                            if cmd.id == step_id {
                                cmd.show_advanced = !cmd.show_advanced;
                                commands.set_row_data(i, cmd);
                            }
                        }
                    }
                    let pipeline_id = PipelineId(state.get_active_pipeline_id() as u32);
                    manager.sync_steps_of_selected_pipeline_to_slint(pipeline_id, false);
                });

            // ⤢ on a step card: its settings in the larger dialog.
            let manager = self.clone();
            ui.global::<PipelinesPanelState>()
                .on_open_step_settings(move |step_id| {
                    let Some(ui) = manager.ui.upgrade() else {
                        return;
                    };
                    let state = ui.global::<PipelinesPanelState>();
                    // Its advanced settings are only sent while shown - the
                    // dialog has them on a page of their own and searches them.
                    let commands = state.get_active_commands();
                    let mut opened_advanced = false;
                    for i in 0..commands.row_count() {
                        if let Some(mut cmd) = commands.row_data(i) {
                            if cmd.step_id == step_id && !cmd.show_advanced {
                                cmd.show_advanced = true;
                                commands.set_row_data(i, cmd);
                                opened_advanced = true;
                            }
                        }
                    }
                    if opened_advanced && !state.get_always_show_advanced() {
                        let pipeline_id = PipelineId(state.get_active_pipeline_id() as u32);
                        manager.sync_steps_of_selected_pipeline_to_slint(pipeline_id, false);
                    }
                    state.set_settings_step_id(step_id);
                    ui.global::<GlobalAppState>()
                        .set_active_dialog(DialogType::StepSettings);
                });

            // Debug mode (user preference): settings show their wiring.
            let debug_mode = self.app_state.load_app_settings().debug_mode;
            self.debug_mode
                .store(debug_mode, std::sync::atomic::Ordering::Relaxed);
            ui.global::<PipelinesPanelState>()
                .set_debug_mode(debug_mode);
            let manager = self.clone();
            ui.global::<PipelinesPanelState>()
                .on_debug_mode_toggled(move |on| {
                    let Some(ui) = manager.ui.upgrade() else {
                        return;
                    };
                    let state = ui.global::<PipelinesPanelState>();
                    state.set_debug_mode(on);
                    manager
                        .debug_mode
                        .store(on, std::sync::atomic::Ordering::Relaxed);
                    manager
                        .app_state
                        .update_app_settings(|settings| settings.debug_mode = on);
                    let pipeline_id = PipelineId(state.get_active_pipeline_id() as u32);
                    manager.sync_steps_of_selected_pipeline_to_slint(pipeline_id, false);
                });

            // "Always show advanced settings" (user preference)
            let manager = self.clone();
            ui.global::<PipelinesPanelState>()
                .on_always_show_advanced_toggled(move |on| {
                    let Some(ui) = manager.ui.upgrade() else {
                        return;
                    };
                    let state = ui.global::<PipelinesPanelState>();
                    state.set_always_show_advanced(on);
                    manager.app_state.update_app_settings(|settings| {
                        settings.always_show_advanced_settings = on
                    });
                    let pipeline_id = PipelineId(state.get_active_pipeline_id() as u32);
                    manager.sync_steps_of_selected_pipeline_to_slint(pipeline_id, false);
                });

            let manager = self.clone();
            ui.global::<PipelinesPanelState>()
                .on_select_pipeline(move |pipeline_id| {
                    // Another pipeline: an opened component is closed.
                    if manager.open_component_key(pipeline_id as u32).is_none() {
                        manager.close_open_component();
                    }
                    manager.sync_steps_of_selected_pipeline_to_slint(
                        PipelineId(pipeline_id as u32),
                        true,
                    );
                });

            // Toggle pipeline
            let manager = self.clone();
            ui.global::<PipelinesPanelState>()
                .on_toggle_pipeline(move |pipeline_id| {
                    let enabled = {
                        let project = manager.app_state.get_project();
                        project
                            .pipelines
                            .iter()
                            .find(|p| p.id.0 == pipeline_id as u32)
                            .map(|p| !p.enabled)
                            .unwrap_or(false)
                    };
                    {
                        let mut project = manager.app_state.get_project_write();
                        project.enable_pipeline(enabled, PipelineId(pipeline_id as u32));
                    }
                    manager.pipeline_settings_changed();
                    manager.sync_pipelines_to_slint();
                });

            // Move pipeline up
            let manager = self.clone();
            ui.global::<PipelinesPanelState>()
                .on_move_pipeline_up(move |pipeline_id| {
                    {
                        let mut project = manager.app_state.get_project_write();
                        project.move_pipeline_up(PipelineId(pipeline_id as u32));
                    }
                    manager.pipeline_settings_changed();
                    manager.sync_pipelines_to_slint();
                });

            // Move pipeline down
            let manager = self.clone();
            ui.global::<PipelinesPanelState>()
                .on_move_pipeline_down(move |pipeline_id| {
                    {
                        let mut project = manager.app_state.get_project_write();
                        project.move_pipeline_down(PipelineId(pipeline_id as u32));
                    }
                    manager.pipeline_settings_changed();
                    manager.sync_pipelines_to_slint();
                });

            // New pipeline
            let manager = self.clone();
            ui.global::<PipelinesPanelState>().on_new_pipeline(move || {
                let (new_id, name) = {
                    let mut project = manager.app_state.get_project_write();
                    let next_id = project.pipelines.iter().map(|p| p.id.0).max().unwrap_or(0) + 1;
                    let name = format!("Pipeline {}", next_id);
                    project.add_pipeline(PipelineSettings {
                        id: PipelineId(next_id),
                        name: name.clone(),
                        image_source: ImageAddress::Channel(0),
                        enabled: true,
                        steps: vec![],
                        description: None,
                    });
                    (next_id, name)
                };
                manager.pipeline_settings_changed();
                manager.sync_pipelines_to_slint();
                manager.sync_steps_of_selected_pipeline_to_slint(PipelineId(new_id), true);
                let ui_weak = manager.ui.clone();
                crate::helper::ui_thread::invoke_from_event_loop(move || {
                    if let Some(ui) = ui_weak.upgrade() {
                        ui.global::<PipelinesPanelState>()
                            .set_active_pipeline_id(new_id as i32);
                        let edit = ui.global::<PipelineEditState>();
                        edit.set_pipeline_id(new_id as i32);
                        edit.set_pipeline_name(name.into());
                        edit.set_source_type(2); // Channel
                        edit.set_source_slot(1);
                        edit.set_source_channel(0);
                        ui.global::<GlobalAppState>()
                            .set_active_dialog(DialogType::PipelineEdit);
                    }
                })
                .ok();
            });

            // Edit pipeline - open edit dialog
            let manager = self.clone();
            ui.global::<PipelinesPanelState>()
                .on_pipeline_more(move |pipeline_id| {
                    let state = {
                        let project = manager.app_state.get_project();
                        project
                            .pipelines
                            .iter()
                            .find(|p| p.id.0 == pipeline_id as u32)
                            .map(|p| {
                                let (stype, slot, ch) = match p.image_source {
                                    ImageAddress::Scratchpad => (0i32, 1i32, 0i32),
                                    ImageAddress::Memory(MemoryId::PipelineContext(s)) => {
                                        (1, s as i32, 0)
                                    }
                                    ImageAddress::Memory(MemoryId::ProjectCache(s))
                                    | ImageAddress::Memory(MemoryId::ComponentContext(s)) => {
                                        (1, s as i32, 0)
                                    }
                                    ImageAddress::Channel(c) => (2, 1, c),
                                    ImageAddress::None => (3, 1, 0),
                                };
                                (p.name.clone(), stype, slot, ch)
                            })
                    };
                    let Some((name, stype, slot, ch)) = state else {
                        return;
                    };
                    let Some(ui) = manager.ui.upgrade() else {
                        return;
                    };
                    let edit = ui.global::<PipelineEditState>();
                    edit.set_pipeline_id(pipeline_id);
                    edit.set_pipeline_name(name.into());
                    edit.set_source_type(stype);
                    edit.set_source_slot(slot);
                    edit.set_source_channel(ch);
                    ui.global::<GlobalAppState>()
                        .set_active_dialog(DialogType::PipelineEdit);
                });

            // Duplicate pipeline
            let manager = self.clone();
            ui.global::<PipelinesPanelState>()
                .on_duplicate_pipeline(move |pipeline_id| {
                    let new_pipeline = {
                        let project = manager.app_state.get_project();
                        project
                            .pipelines
                            .iter()
                            .find(|p| p.id.0 == pipeline_id as u32)
                            .map(|p| {
                                let next_id =
                                    project.pipelines.iter().map(|p| p.id.0).max().unwrap_or(0) + 1;
                                let mut clone = p.clone();
                                clone.id = PipelineId(next_id);
                                // Keeps its step ids: they're unique per pipeline.
                                clone.name = format!("{} (Copy)", p.name);
                                clone
                            })
                    };
                    let Some(new_p) = new_pipeline else { return };
                    let new_id = new_p.id.0;
                    {
                        let mut project = manager.app_state.get_project_write();
                        project.pipelines.push(new_p);
                    }
                    manager.pipeline_settings_changed();
                    manager.sync_pipelines_to_slint();
                    manager.sync_steps_of_selected_pipeline_to_slint(PipelineId(new_id), true);
                    let ui_weak = manager.ui.clone();
                    crate::helper::ui_thread::invoke_from_event_loop(move || {
                        if let Some(ui) = ui_weak.upgrade() {
                            ui.global::<PipelinesPanelState>()
                                .set_active_pipeline_id(new_id as i32);
                        }
                    })
                    .ok();
                });

            // Delete pipeline - open confirm dialog
            let manager = self.clone();
            ui.global::<PipelinesPanelState>()
                .on_delete_pipeline(move |pipeline_id| {
                    let pipeline_name: String = {
                        let project = manager.app_state.get_project();
                        project
                            .pipelines
                            .iter()
                            .find(|p| p.id.0 == pipeline_id as u32)
                            .and_then(|p| Some(p.name.clone()))
                            .unwrap_or_else(|| format!("Pipeline {}", pipeline_id))
                    };
                    let Some(ui) = manager.ui.upgrade() else {
                        return;
                    };
                    let confirm = ui.global::<PipelineDeleteConfirmState>();
                    confirm.set_pipeline_id(pipeline_id);
                    confirm.set_pipeline_name(pipeline_name.into());
                    ui.global::<GlobalAppState>()
                        .set_active_dialog(DialogType::PipelineDeleteConfirm);
                });

            // Delete pipeline confirm
            let manager = self.clone();
            ui.global::<PipelineDeleteConfirmState>()
                .on_confirm(move || {
                    let Some(ui) = manager.ui.upgrade() else {
                        return;
                    };
                    let pipeline_id =
                        ui.global::<PipelineDeleteConfirmState>().get_pipeline_id() as u32;
                    let next_active = {
                        let mut project = manager.app_state.get_project_write();
                        if let Some(idx) =
                            project.pipelines.iter().position(|p| p.id.0 == pipeline_id)
                        {
                            project.pipelines.remove(idx);
                        }
                        project.pipelines.first().map(|p| p.id.0)
                    };
                    ui.global::<GlobalAppState>()
                        .set_active_dialog(DialogType::None);
                    manager.pipeline_settings_changed();
                    manager.sync_pipelines_to_slint();
                    match next_active {
                        Some(nid) => {
                            manager.sync_steps_of_selected_pipeline_to_slint(PipelineId(nid), true);
                            let ui_weak = manager.ui.clone();
                            crate::helper::ui_thread::invoke_from_event_loop(move || {
                                if let Some(ui) = ui_weak.upgrade() {
                                    ui.global::<PipelinesPanelState>()
                                        .set_active_pipeline_id(nid as i32);
                                }
                            })
                            .ok();
                        }
                        None => {
                            let ui_weak = manager.ui.clone();
                            crate::helper::ui_thread::invoke_from_event_loop(move || {
                                if let Some(ui) = ui_weak.upgrade() {
                                    let ps = ui.global::<PipelinesPanelState>();
                                    ps.set_active_pipeline_id(0);
                                    ps.set_active_pipeline_name("".into());
                                    ps.set_active_pipeline_image_source("".into());
                                    ps.set_active_commands(ModelRc::default());
                                }
                            })
                            .ok();
                        }
                    }
                });

            // Delete pipeline cancel
            let manager = self.clone();
            ui.global::<PipelineDeleteConfirmState>()
                .on_cancel(move || {
                    if let Some(ui) = manager.ui.upgrade() {
                        ui.global::<GlobalAppState>()
                            .set_active_dialog(DialogType::None);
                    }
                });

            // Edit dialog: save changes
            let manager = self.clone();
            ui.global::<PipelineEditState>().on_confirm(move || {
                let Some(ui) = manager.ui.upgrade() else {
                    return;
                };
                let edit = ui.global::<PipelineEditState>();
                let pipeline_id = edit.get_pipeline_id() as u32;
                let name = edit.get_pipeline_name().to_string();
                let stype = edit.get_source_type();
                let slot = edit.get_source_slot() as MemorySlot;
                let ch = edit.get_source_channel();
                let display_name = if name.is_empty() {
                    format!("Pipeline {}", pipeline_id)
                } else {
                    name.clone()
                };
                let image_source = match stype {
                    0 => ImageAddress::Scratchpad,
                    // A start from a memory slot reads the shared one.
                    1 => ImageAddress::Memory(MemoryId::ProjectCache(slot.max(1))),
                    2 => ImageAddress::Channel(ch),
                    3 => ImageAddress::None,
                    _ => ImageAddress::default(),
                };
                {
                    let mut project = manager.app_state.get_project_write();
                    if let Some(p) = project.pipelines.iter_mut().find(|p| p.id.0 == pipeline_id) {
                        p.name = name;
                        p.image_source = image_source;
                    }
                }
                ui.global::<GlobalAppState>()
                    .set_active_dialog(DialogType::None);
                // Update the "EDITING" bar immediately if this is the active pipeline.
                let ps = ui.global::<PipelinesPanelState>();
                if ps.get_active_pipeline_id() as u32 == pipeline_id {
                    ps.set_active_pipeline_name(display_name.into());
                    let image_source_str: slint::SharedString = match image_source {
                        ImageAddress::Scratchpad => "Scratchpad".into(),
                        ImageAddress::Memory(
                            MemoryId::PipelineContext(s) | MemoryId::ProjectCache(s),
                        ) => format!("Shared[{s}]").into(),
                        ImageAddress::Memory(MemoryId::ComponentContext(s)) => {
                            format!("Component[{s}]").into()
                        }
                        ImageAddress::None => "None".into(),
                        ImageAddress::Channel(c) => format!("Channel {c}").into(),
                    };
                    ps.set_active_pipeline_image_source(image_source_str);
                }
                manager.pipeline_settings_changed();
                manager.sync_pipelines_to_slint();
            });

            // Edit dialog: cancel
            let manager = self.clone();
            ui.global::<PipelineEditState>().on_cancel(move || {
                if let Some(ui) = manager.ui.upgrade() {
                    ui.global::<GlobalAppState>()
                        .set_active_dialog(DialogType::None);
                }
            });

            // Running dialog: cancel analysis
            let manager = self.clone();
            ui.global::<PipelineRunningState>().on_cancel(move || {
                if let Some(cancel) = manager.pipeline_cancel_flag.lock().unwrap().as_ref() {
                    cancel.cancel();
                }
            });

            // Running dialog: close after done
            let manager = self.clone();
            ui.global::<PipelineRunningState>().on_close(move || {
                if let Some(ui) = manager.ui.upgrade() {
                    ui.global::<PipelineRunningState>().set_done(false);
                    ui.global::<PipelineRunningState>()
                        .set_status_message("".into());
                    ui.global::<GlobalAppState>()
                        .set_active_dialog(DialogType::None);
                }
            });

            // Toggle step
            let manager = self.clone();
            ui.global::<PipelinesPanelState>()
                .on_toggle_step(move |step_idx| {
                    let Some(ui) = manager.ui.upgrade() else {
                        return;
                    };
                    let pipeline_id =
                        ui.global::<PipelinesPanelState>().get_active_pipeline_id() as u32;
                    if manager.viewing(pipeline_id) {
                        manager.sync_steps_of_selected_pipeline_to_slint(
                            PipelineId(pipeline_id),
                            false,
                        );
                        return;
                    }
                    if manager.open_component_key(pipeline_id).is_some() {
                        let mut project = manager.app_state.get_project_write();
                        if let Some(step) = manager
                            .shown_steps_mut(&mut project, pipeline_id)
                            .and_then(|steps| steps.get_mut(step_idx as usize))
                        {
                            step.enabled = !step.enabled;
                        }
                    } else {
                        let enabled = {
                            let project = manager.app_state.get_project();
                            project
                                .pipelines
                                .iter()
                                .find(|p| p.id.0 == pipeline_id)
                                .and_then(|p| p.steps.get(step_idx as usize))
                                .map(|s| !s.enabled)
                                .unwrap_or(false)
                        };
                        let mut project = manager.app_state.get_project_write();
                        project.enable_pipeline_step(
                            enabled,
                            PipelineId(pipeline_id),
                            step_idx as usize,
                        );
                    }
                    manager.pipeline_settings_changed();
                    manager
                        .sync_steps_of_selected_pipeline_to_slint(PipelineId(pipeline_id), false);
                });

            // Expand step
            let _manager = self.clone();
            ui.global::<PipelinesPanelState>()
                .on_expand_step(move |_step_idx| {});

            // Remove step
            let manager = self.clone();
            ui.global::<PipelinesPanelState>()
                .on_remove_step(move |step_idx| {
                    let Some(ui) = manager.ui.upgrade() else {
                        return;
                    };
                    let pipeline_id =
                        ui.global::<PipelinesPanelState>().get_active_pipeline_id() as u32;
                    if manager.viewing(pipeline_id) {
                        manager.sync_steps_of_selected_pipeline_to_slint(
                            PipelineId(pipeline_id),
                            false,
                        );
                        return;
                    }
                    if let Some(key) = manager.open_component_key(pipeline_id) {
                        // A step of the opened component: its bindings go with it.
                        let mut project = manager.app_state.get_project_write();
                        if let Some(definition) =
                            project.components.iter_mut().find(|d| d.key == key)
                        {
                            evanalyzer_app::components::remove_component_step(
                                definition,
                                step_idx as usize,
                            );
                        }
                    } else {
                        let mut project = manager.app_state.get_project_write();
                        if let Some(pipeline) =
                            project.pipelines.iter_mut().find(|p| p.id.0 == pipeline_id)
                        {
                            let idx = step_idx as usize;
                            if idx < pipeline.steps.len() {
                                let removed = pipeline.steps.remove(idx);
                                // Its number may be given to the next new step
                                // (the highest + 1) - the breakpoint must not move
                                // there.
                                let mut breakpoint = manager.breakpoint.lock().unwrap();
                                if breakpoint.is_some_and(|(p, step, _)| {
                                    p == pipeline_id && step == removed.id
                                }) {
                                    *breakpoint = None;
                                    drop(breakpoint);
                                    ui.global::<PipelinesPanelState>()
                                        .set_breakpoint_pipeline_id(-1);
                                    ui.global::<PipelinesPanelState>()
                                        .set_breakpoint_step_id("".into());
                                    ui.global::<PipelinesPanelState>().set_breakpoint_mode(0);
                                }
                            }
                        }
                    }
                    manager.pipeline_settings_changed();
                    // The other cards keep their state: it's keyed by step id.
                    manager
                        .sync_steps_of_selected_pipeline_to_slint(PipelineId(pipeline_id), false);
                });

            // Duplicate step
            let manager = self.clone();
            ui.global::<PipelinesPanelState>()
                .on_duplicate_step(move |step_idx| {
                    let Some(ui) = manager.ui.upgrade() else {
                        return;
                    };
                    let pipeline_id =
                        ui.global::<PipelinesPanelState>().get_active_pipeline_id() as u32;
                    if manager.viewing(pipeline_id) {
                        manager.sync_steps_of_selected_pipeline_to_slint(
                            PipelineId(pipeline_id),
                            false,
                        );
                        return;
                    }
                    {
                        let mut project = manager.app_state.get_project_write();
                        if let Some(steps) = manager.shown_steps_mut(&mut project, pipeline_id) {
                            let idx = step_idx as usize;
                            if idx < steps.len() {
                                let mut cloned = steps[idx].clone();
                                cloned.id = evanalyzer_cfg::core_types::StepId::UNASSIGNED;
                                steps.insert(idx + 1, cloned);
                                evanalyzer_app::utils::assign_step_ids(steps);
                            }
                        }
                    }
                    manager.pipeline_settings_changed();
                    manager
                        .sync_steps_of_selected_pipeline_to_slint(PipelineId(pipeline_id), false);
                });

            // Insert step - open command picker
            let manager = self.clone();
            ui.global::<PipelinesPanelState>().on_insert_step(
                move |pipeline_id, step_after_idx| {
                    let (pipeline_name, total_steps, context_cat, suggested_filter) = {
                        let project = manager.app_state.get_project();
                        if let Some(p) = project
                            .pipelines
                            .iter()
                            .find(|p| p.id.0 == pipeline_id as u32)
                        {
                            let name = p.name.clone();
                            // Find the category of the last step at or before the insertion point.
                            // step_after_idx is the 0-based index of the step we insert AFTER.
                            // A value >= steps.len() means "append at end".
                            let context_idx = if step_after_idx >= 0 {
                                (step_after_idx as usize).min(p.steps.len().saturating_sub(1))
                            } else {
                                usize::MAX
                            };
                            // Pre-select the chips for *all* categories that may
                            // follow the previous step (its `allowed_next`, which is
                            // decoupled from the display `category` — e.g.
                            // ConnectedComponents suggests Object + Measure so both
                            // Watershed and ExtractObjects are in view). These are only
                            // suggestions: the user can toggle any chip, and clearing
                            // them all shows every command.
                            // A component suggests nothing (yet).
                            let prev = p.steps.get(context_idx).and_then(|s| s.command());
                            let (ctx_cat, suggested) = match prev {
                                Some(prev) => {
                                    let ctx_order = prev.category().display_order() as i32;
                                    let mut flags = [false; 6];
                                    for c in prev.allowed_next() {
                                        flags[c.display_order() as usize] = true;
                                    }
                                    (ctx_order, flags)
                                }
                                None => (-1i32, [false; 6]),
                            };
                            (name, p.steps.len() as i32, ctx_cat, suggested)
                        } else {
                            (String::new(), 0, -1i32, [false; 6])
                        }
                    };
                    manager.reload_component_library_async();
                    if let Some(ui) = manager.ui.upgrade() {
                        let picker = ui.global::<CommandPickerState>();
                        picker.set_pipeline_id(pipeline_id);
                        picker.set_insert_after_idx(step_after_idx);
                        picker.set_target_pipeline(pipeline_name.into());
                        picker.set_total_steps(total_steps);
                        picker.set_context_category(context_cat);
                        picker.set_query("".into());
                        picker.set_filter_components(false);
                        // Pre-select the suggested next categories (all of them); an
                        // empty set leaves the "All" view.
                        picker.set_fcat_pre(suggested_filter[0]);
                        picker.set_fcat_seg(suggested_filter[1]);
                        picker.set_fcat_obj(suggested_filter[2]);
                        picker.set_fcat_mea(suggested_filter[3]);
                        picker.set_fcat_cls(suggested_filter[4]);
                        picker.set_fcat_dat(suggested_filter[5]);
                        picker.set_selected_id(-1);
                        // Apply the filter immediately so the list matches the chips.
                        manager.apply_picker_filter(&ui, "");
                        ui.global::<GlobalAppState>()
                            .set_active_dialog(DialogType::CommandSelectionDialog);
                    }
                },
            );

            // Picker: query changed - re-filter (also called by category chips)
            let manager = self.clone();
            ui.global::<CommandPickerState>()
                .on_query_changed(move |query| {
                    let Some(ui) = manager.ui.upgrade() else {
                        return;
                    };
                    manager.apply_picker_filter(&ui, query.as_str());
                });

            // Picker: select - update detail pane
            let manager = self.clone();
            ui.global::<CommandPickerState>()
                .on_select(move |command_id| {
                    let Some(ui) = manager.ui.upgrade() else {
                        return;
                    };
                    ui.global::<CommandPickerState>()
                        .set_detail_removable(false);
                    if command_id >= LIBRARY_COMPONENT_PICKER_ID_BASE {
                        let idx = (command_id - LIBRARY_COMPONENT_PICKER_ID_BASE) as usize;
                        let library = manager.component_library.lock().expect("Poisoned");
                        if let Some((path, template)) = library.get(idx) {
                            let picker = ui.global::<CommandPickerState>();
                            picker
                                .set_detail(library_component_to_command_def(idx, path, template));
                            picker.set_has_detail(true);
                        }
                        return;
                    }
                    if command_id >= COMPONENT_PICKER_ID_BASE {
                        let project = manager.app_state.get_project();
                        let idx = (command_id - COMPONENT_PICKER_ID_BASE) as usize;
                        if let Some(definition) = project.components.get(idx) {
                            let detail = component_to_command_def(idx, definition, &project);
                            let picker = ui.global::<CommandPickerState>();
                            picker.set_detail(detail);
                            picker.set_has_detail(true);
                            picker.set_detail_removable(
                                evanalyzer_app::components::unused_components(&project)
                                    .contains(&definition.key),
                            );
                        }
                        return;
                    }
                    let metas = all_command_meta();
                    if let Some(m) = metas.iter().find(|m| m.id == command_id) {
                        let detail = to_command_def(m);
                        let picker = ui.global::<CommandPickerState>();
                        picker.set_detail(detail);
                        picker.set_has_detail(true);
                    }
                });

            // Picker: confirm - insert command and close
            let manager = self.clone();
            ui.global::<CommandPickerState>()
                .on_confirm(move |command_id| {
                    let Some(ui) = manager.ui.upgrade() else {
                        return;
                    };
                    let picker = ui.global::<CommandPickerState>();
                    let pipeline_id = picker.get_pipeline_id() as u32;
                    let after_idx = picker.get_insert_after_idx();
                    if manager.viewing(pipeline_id) {
                        return;
                    }

                    let new_steps: Vec<PipelineStepSettings> =
                        if command_id >= LIBRARY_COMPONENT_PICKER_ID_BASE {
                            // A library component: copied into the project, then used.
                            let idx = (command_id - LIBRARY_COMPONENT_PICKER_ID_BASE) as usize;
                            let Some((path, template)) = manager
                                .component_library
                                .lock()
                                .expect("Poisoned")
                                .get(idx)
                                .cloned()
                            else {
                                warn!(
                                    "picker confirm: unknown library component id {}",
                                    command_id
                                );
                                return;
                            };
                            let inserted = {
                                let mut project = manager.app_state.get_project_write();
                                evanalyzer_app::components::insert_component_template(
                                    &mut project,
                                    &template,
                                    &evanalyzer_app::components::library_name(&path),
                                )
                            };
                            let key = match inserted {
                                Ok(key) => key,
                                Err(e) => {
                                    warn!("Could not insert the library component: {e}");
                                    return;
                                }
                            };
                            vec![PipelineStepSettings {
                                id: evanalyzer_cfg::core_types::StepId::UNASSIGNED,
                                enabled: true,
                                content: StepContent::Component(
                                    evanalyzer_cfg::settings::components::ComponentInstance {
                                        key,
                                        values: Default::default(),
                                    },
                                ),
                            }]
                        } else if command_id >= COMPONENT_PICKER_ID_BASE {
                            // A component: one step using it, at its defaults.
                            let project = manager.app_state.get_project();
                            let idx = (command_id - COMPONENT_PICKER_ID_BASE) as usize;
                            let Some(definition) = project.components.get(idx) else {
                                warn!("picker confirm: unknown component id {}", command_id);
                                return;
                            };
                            vec![PipelineStepSettings {
                                id: evanalyzer_cfg::core_types::StepId::UNASSIGNED,
                                enabled: true,
                                content: StepContent::Component(
                                    evanalyzer_cfg::settings::components::ComponentInstance {
                                        key: definition.key.clone(),
                                        values: Default::default(),
                                    },
                                ),
                            }]
                        } else {
                            let Some(cmd) = default_command(command_id) else {
                                warn!("picker confirm: unknown command id {}", command_id);
                                return;
                            };
                            vec![PipelineStepSettings {
                            id: evanalyzer_cfg::core_types::StepId::UNASSIGNED,
                            enabled: true,
                            content:
                                evanalyzer_cfg::settings::pipeline_settings::StepContent::Command(
                                    cmd,
                                ),
                        }]
                        };

                    // A component can't contain itself - also not through the
                    // components it contains (e.g. a library component's).
                    if let Some(open_key) = manager.open_component_key(pipeline_id) {
                        let project = manager.app_state.get_project();
                        let cycle = new_steps.iter().find_map(|s| match &s.content {
                            StepContent::Component(instance)
                                if evanalyzer_app::components::would_contain_itself(
                                    &project,
                                    &open_key,
                                    &instance.key,
                                ) =>
                            {
                                Some(step_label(&project, s))
                            }
                            _ => None,
                        });
                        drop(project);
                        if let Some(name) = cycle {
                            let warning = ui.global::<WarningState>();
                            warning.set_info(false);
                            warning.set_title("Cannot insert the component".into());
                            warning.set_message(
                                format!(
                                    "\"{name}\" contains the component it would be inserted \
                                     into - a component can't contain itself."
                                )
                                .into(),
                            );
                            ui.global::<GlobalAppState>()
                                .set_active_dialog(DialogType::Warning);
                            return;
                        }
                    }
                    {
                        let mut project = manager.app_state.get_project_write();
                        if let Some(steps) = manager.shown_steps_mut(&mut project, pipeline_id) {
                            let insert_at = if after_idx < 0 {
                                0
                            } else {
                                ((after_idx as usize) + 1).min(steps.len())
                            };
                            steps.splice(insert_at..insert_at, new_steps);
                            evanalyzer_app::utils::assign_step_ids(steps);
                        }
                    }
                    ui.global::<GlobalAppState>()
                        .set_active_dialog(DialogType::None);
                    manager.pipeline_settings_changed();
                    manager
                        .sync_steps_of_selected_pipeline_to_slint(PipelineId(pipeline_id), false);
                });

            // "Open" on a component card: its steps in the panel - asking first
            // if other steps use the component too.
            let manager = self.clone();
            ui.global::<PipelinesPanelState>()
                .on_open_component(move |step_idx| {
                    let Some(ui) = manager.ui.upgrade() else {
                        return;
                    };
                    let pipeline_id =
                        ui.global::<PipelinesPanelState>().get_active_pipeline_id() as u32;
                    if manager.viewing(pipeline_id) {
                        return;
                    }
                    let shown = manager.open_component_path(pipeline_id);
                    manager.edit_component(&ui, pipeline_id, shown, step_idx as usize);
                });

            // "Edit for all": the component itself.
            let manager = self.clone();
            ui.global::<ComponentEditChoiceState>()
                .on_edit_all(move || {
                    let Some(ui) = manager.ui.upgrade() else {
                        return;
                    };
                    let choice = ui.global::<ComponentEditChoiceState>();
                    let pipeline_id = choice.get_pipeline_id() as u32;
                    ui.global::<GlobalAppState>()
                        .set_active_dialog(DialogType::None);
                    manager.take_pending_edit_base(pipeline_id);
                    manager.open_component_in_panel(
                        pipeline_id,
                        choice.get_step_idx() as usize,
                        false,
                    );
                });

            // "Only this step": a copy of the component for it.
            let manager = self.clone();
            ui.global::<ComponentEditChoiceState>()
                .on_only_this(move || {
                    let Some(ui) = manager.ui.upgrade() else {
                        return;
                    };
                    let choice = ui.global::<ComponentEditChoiceState>();
                    let pipeline_id = choice.get_pipeline_id() as u32;
                    let step_idx = choice.get_step_idx() as usize;
                    ui.global::<GlobalAppState>()
                        .set_active_dialog(DialogType::None);
                    manager.take_pending_edit_base(pipeline_id);
                    let copied = {
                        let mut project = manager.app_state.get_project_write();
                        evanalyzer_app::components::clone_component_for_step(
                            &mut project,
                            &manager.shown_list(pipeline_id),
                            step_idx,
                        )
                    };
                    match copied {
                        Ok(_) => {
                            manager.pipeline_settings_changed();
                            manager.open_component_in_panel(pipeline_id, step_idx, false);
                        }
                        Err(e) => warn!("Could not copy the component: {e}"),
                    }
                });

            let manager = self.clone();
            ui.global::<ComponentEditChoiceState>().on_cancel(move || {
                if let Some(ui) = manager.ui.upgrade() {
                    *manager.pending_edit_base.lock().expect("Poisoned") = None;
                    ui.global::<GlobalAppState>()
                        .set_active_dialog(DialogType::None);
                }
            });

            // "‹ Back" from an opened component: the steps it was opened from
            // again - the component's it's nested in, or the pipeline's.
            let manager = self.clone();
            ui.global::<PipelinesPanelState>()
                .on_close_component(move || {
                    let Some(ui) = manager.ui.upgrade() else {
                        return;
                    };
                    manager.close_component_level();
                    let pipeline_id =
                        ui.global::<PipelinesPanelState>().get_active_pipeline_id() as u32;
                    manager.sync_steps_of_selected_pipeline_to_slint(PipelineId(pipeline_id), true);
                });

            // "View" on a component card: its steps in the panel, read only.
            let manager = self.clone();
            ui.global::<PipelinesPanelState>()
                .on_view_component(move |step_idx| {
                    let Some(ui) = manager.ui.upgrade() else {
                        return;
                    };
                    let pipeline_id =
                        ui.global::<PipelinesPanelState>().get_active_pipeline_id() as u32;
                    manager.open_component_in_panel(pipeline_id, step_idx as usize, true);
                });

            // A problem in the panel's list: its pipeline, with the card of its
            // step expanded - for a component's own, the first step using it,
            // opened.
            let manager = self.clone();
            ui.global::<PipelinesPanelState>()
                .on_show_problem(move |index| manager.show_problem(index as usize));

            // ▸ / ▾ of a component card's section: open or close it - in
            // place, the card isn't rebuilt.
            let manager = self.clone();
            ui.global::<PipelinesPanelState>()
                .on_toggle_section(move |step_idx, section| {
                    let Some(ui) = manager.ui.upgrade() else {
                        return;
                    };
                    let state = ui.global::<PipelinesPanelState>();
                    let pipeline_id = state.get_active_pipeline_id() as u32;
                    let Some(card) = state.get_active_commands().row_data(step_idx as usize) else {
                        return;
                    };
                    let key = (pipeline_id, card.step_id.to_string(), section.to_string());
                    let open = {
                        let mut open_sections = manager.open_sections.lock().expect("Poisoned");
                        if open_sections.remove(&key) {
                            false
                        } else {
                            open_sections.insert(key);
                            true
                        }
                    };
                    let views = card.section_views.clone();
                    for i in 0..views.row_count() {
                        if let Some(mut view) = views.row_data(i)
                            && view.name == section
                        {
                            view.open = open;
                            views.set_row_data(i, view);
                        }
                    }
                });

            // A problem's fix ("Add class 1").
            let manager = self.clone();
            ui.global::<PipelinesPanelState>()
                .on_fix_problem(move |index| manager.fix_problem(index as usize));

            // "Edit" in the bar of a viewed component: as "Edit" on its card. In
            // a component opened inside a viewed one, the first viewed one is
            // opened to edit - the ones inside it can then be opened to edit.
            let manager = self.clone();
            ui.global::<PipelinesPanelState>()
                .on_edit_open_component(move || {
                    let Some(ui) = manager.ui.upgrade() else {
                        return;
                    };
                    let pipeline_id =
                        ui.global::<PipelinesPanelState>().get_active_pipeline_id() as u32;
                    let mut path = manager.open_component_path(pipeline_id);
                    let Some(first_viewed) =
                        path.iter().position(|level| level.viewing_from.is_some())
                    else {
                        return;
                    };
                    let Some(step_idx) = path[first_viewed].viewing_from else {
                        return;
                    };
                    path.truncate(first_viewed);
                    manager.edit_component(&ui, pipeline_id, path, step_idx);
                });

            // "Settings…" in the opened component's bar: the component settings dialog.
            let manager = self.clone();
            ui.global::<PipelinesPanelState>()
                .on_open_component_settings(move || {
                    manager.show_component_settings(None);
                });

            // ✎ next to an exposed setting: the dialog at its parameter.
            let manager = self.clone();
            ui.global::<PipelinesPanelState>()
                .on_edit_exposed(move |step_idx, path| {
                    let Some(ui) = manager.ui.upgrade() else {
                        return;
                    };
                    let pipeline_id =
                        ui.global::<PipelinesPanelState>().get_active_pipeline_id() as u32;
                    let Some(key) = manager.open_component_key(pipeline_id) else {
                        return;
                    };
                    let id =
                        {
                            let project = manager.app_state.get_project();
                            project.components.iter().find(|d| d.key == key).and_then(
                                |definition| {
                                    let step = definition.steps.get(step_idx as usize)?;
                                    evanalyzer_app::components::parameter_for_setting(
                                        definition, step.id, &path,
                                    )
                                },
                            )
                        };
                    if id.is_some() {
                        manager.show_component_settings(id);
                    }
                });

            // The dialog's name and description. No refill: the line being
            // typed in keeps its focus.
            let manager = self.clone();
            ui.global::<ComponentSettingsState>()
                .on_name_edited(move |name| {
                    manager.edit_open_component_definition(|definition| {
                        let name = name.trim();
                        if name.is_empty() {
                            return false;
                        }
                        definition.meta.name = name.to_string();
                        true
                    });
                });
            let manager = self.clone();
            ui.global::<ComponentSettingsState>()
                .on_description_edited(move |description| {
                    manager.edit_open_component_definition(|definition| {
                        definition.meta.short_description = description.to_string();
                        true
                    });
                });

            let manager = self.clone();
            ui.global::<ComponentSettingsState>()
                .on_long_description_edited(move |description| {
                    manager.edit_open_component_definition(|definition| {
                        definition.meta.description = description.to_string();
                        true
                    });
                });

            // The first author: both empty removes it.
            let manager = self.clone();
            ui.global::<ComponentSettingsState>()
                .on_author_edited(move |name, organization| {
                    manager.edit_open_component_definition(|definition| {
                        set_first_author(&mut definition.meta.authors, &name, &organization);
                        true
                    });
                });

            // "Reset help text": the bound setting's own help again.
            let manager = self.clone();
            ui.global::<ComponentSettingsState>()
                .on_parameter_help_reset(move |id| {
                    let id = ComponentParamKey::from(id.as_str());
                    if manager.edit_open_component_definition(|definition| {
                        let Some(label) = definition
                            .parameters
                            .iter()
                            .find(|p| p.key == id)
                            .map(|p| p.label.clone())
                        else {
                            return false;
                        };
                        evanalyzer_app::components::edit_parameter(definition, &id, &label, "")
                    }) {
                        manager.fill_component_settings(-1);
                    }
                });

            // A parameter renamed or its help text edited - no refill either.
            let manager = self.clone();
            ui.global::<ComponentSettingsState>()
                .on_parameter_name_edited(move |id, label| {
                    manager.edit_open_component_definition(|definition| {
                        let id = ComponentParamKey::from(id.as_str());
                        let Some(description) = definition
                            .parameters
                            .iter()
                            .find(|p| p.key == id)
                            .map(|p| p.description.clone())
                        else {
                            return false;
                        };
                        evanalyzer_app::components::edit_parameter(
                            definition,
                            &id,
                            &label,
                            &description,
                        )
                    });
                });
            let manager = self.clone();
            ui.global::<ComponentSettingsState>()
                .on_parameter_help_edited(move |id, description| {
                    manager.edit_open_component_definition(|definition| {
                        let id = ComponentParamKey::from(id.as_str());
                        let Some(label) = definition
                            .parameters
                            .iter()
                            .find(|p| p.key == id)
                            .map(|p| p.label.clone())
                        else {
                            return false;
                        };
                        evanalyzer_app::components::edit_parameter(
                            definition,
                            &id,
                            &label,
                            &description,
                        )
                    });
                });

            // The parameter's section on the component card - no refill while
            // typing either; the card shows it on close.
            let manager = self.clone();
            ui.global::<ComponentSettingsState>()
                .on_parameter_section_edited(move |id, section| {
                    manager.edit_open_component_definition(|definition| {
                        evanalyzer_app::components::set_parameter_section(
                            definition,
                            &ComponentParamKey::from(id.as_str()),
                            &section,
                        )
                    });
                });

            // ↑/↓: the parameter's place on the component card.
            let manager = self.clone();
            ui.global::<ComponentSettingsState>()
                .on_parameter_moved(move |id, delta| {
                    if manager.edit_open_component_definition(|definition| {
                        evanalyzer_app::components::move_parameter(
                            definition,
                            &ComponentParamKey::from(id.as_str()),
                            delta,
                        )
                    }) {
                        manager.fill_component_settings(-1);
                    }
                });

            // ✕: the parameter is no longer exposed.
            let manager = self.clone();
            ui.global::<ComponentSettingsState>()
                .on_parameter_removed(move |id| {
                    if manager.edit_open_component_definition(|definition| {
                        evanalyzer_app::components::remove_parameter(
                            definition,
                            &ComponentParamKey::from(id.as_str()),
                        )
                    }) {
                        manager.pipeline_settings_changed();
                        manager.fill_component_settings(-1);
                    }
                });

            // ✕ next to what a parameter sets: it no longer sets it.
            let manager = self.clone();
            ui.global::<ComponentSettingsState>()
                .on_binding_removed(move |id, step_id, path| {
                    if manager.edit_open_component_definition(|definition| {
                        evanalyzer_app::components::remove_binding(
                            definition,
                            &ComponentParamKey::from(id.as_str()),
                            evanalyzer_cfg::core_types::StepId(step_id as u32),
                            &path,
                        )
                    }) {
                        manager.pipeline_settings_changed();
                        manager.fill_component_settings(manager.editing_parameter(&id));
                    }
                });

            // "Also set…": the parameter sets that setting too.
            let manager = self.clone();
            ui.global::<ComponentSettingsState>()
                .on_binding_added(move |id, index| {
                    let Some(ui) = manager.ui.upgrade() else {
                        return;
                    };
                    let pipeline_id =
                        ui.global::<PipelinesPanelState>().get_active_pipeline_id() as u32;
                    let Some(key) = manager.open_component_key(pipeline_id) else {
                        return;
                    };
                    let parameter = ComponentParamKey::from(id.as_str());
                    let target = evanalyzer_app::components::bindable_settings(
                        &manager.app_state.get_project().components,
                        &key,
                        &parameter,
                    )
                    .into_iter()
                    .nth(index as usize);
                    let Some((step_id, path, _)) = target else {
                        return;
                    };
                    if manager.edit_open_component_definition(|definition| {
                        evanalyzer_app::components::add_binding(
                            definition, &parameter, step_id, &path,
                        )
                        .is_ok()
                    }) {
                        manager.pipeline_settings_changed();
                        manager.fill_component_settings(manager.editing_parameter(&id));
                    }
                });

            // ✎: what changes from now on can be undone with Cancel.
            let manager = self.clone();
            ui.global::<ComponentSettingsState>()
                .on_parameter_edit_started(move |_key| {
                    let Some(ui) = manager.ui.upgrade() else {
                        return;
                    };
                    let pipeline_id =
                        ui.global::<PipelinesPanelState>().get_active_pipeline_id() as u32;
                    let Some(key) = manager.open_component_key(pipeline_id) else {
                        return;
                    };
                    *manager.parameter_edit_snapshot.lock().expect("Poisoned") = manager
                        .app_state
                        .get_project()
                        .components
                        .iter()
                        .find(|d| d.key == key)
                        .cloned();
                });

            // Apply: the changes stay.
            let manager = self.clone();
            ui.global::<ComponentSettingsState>()
                .on_parameter_edit_applied(move || {
                    *manager.parameter_edit_snapshot.lock().expect("Poisoned") = None;
                    manager.rekey_open_component();
                    manager.fill_component_settings(-1);
                });

            // Cancel: the component as it was at ✎.
            let manager = self.clone();
            ui.global::<ComponentSettingsState>()
                .on_parameter_edit_cancelled(move || {
                    let Some(snapshot) = manager
                        .parameter_edit_snapshot
                        .lock()
                        .expect("Poisoned")
                        .take()
                    else {
                        return;
                    };
                    if manager.edit_open_component_definition(|definition| {
                        if definition.key != snapshot.key {
                            return false;
                        }
                        *definition = snapshot;
                        true
                    }) {
                        manager.pipeline_settings_changed();
                    }
                    manager.fill_component_settings(-1);
                });

            // Close: the panel shows the new name and ◆/◇. A parameter still
            // opened keeps its changes.
            let manager = self.clone();
            ui.global::<ComponentSettingsState>().on_close(move || {
                *manager.parameter_edit_snapshot.lock().expect("Poisoned") = None;
                manager.rekey_open_component();
                let Some(ui) = manager.ui.upgrade() else {
                    return;
                };
                ui.global::<GlobalAppState>()
                    .set_active_dialog(DialogType::None);
                let pipeline_id =
                    ui.global::<PipelinesPanelState>().get_active_pipeline_id() as u32;
                manager.sync_steps_of_selected_pipeline_to_slint(PipelineId(pipeline_id), false);
            });

            // "Expand" on a component card: its steps in its place.
            let manager = self.clone();
            ui.global::<PipelinesPanelState>()
                .on_expand_component(move |step_idx| {
                    let Some(ui) = manager.ui.upgrade() else {
                        return;
                    };
                    let pipeline_id =
                        ui.global::<PipelinesPanelState>().get_active_pipeline_id() as u32;
                    if manager.viewing(pipeline_id) {
                        manager.sync_steps_of_selected_pipeline_to_slint(
                            PipelineId(pipeline_id),
                            false,
                        );
                        return;
                    }
                    let result = {
                        let mut project = manager.app_state.get_project_write();
                        evanalyzer_app::components::expand_component(
                            &mut project,
                            manager.shown_list(pipeline_id),
                            step_idx as usize,
                        )
                    };
                    if let Err(e) = result {
                        let warning = ui.global::<WarningState>();
                        warning.set_info(false);
                        warning.set_title("Cannot expand the component".into());
                        warning.set_message(e.to_string().into());
                        ui.global::<GlobalAppState>()
                            .set_active_dialog(DialogType::Warning);
                        return;
                    }
                    manager.pipeline_settings_changed();
                    manager
                        .sync_steps_of_selected_pipeline_to_slint(PipelineId(pipeline_id), false);
                });

            // ◇/◆ on a setting inside an opened component: hide it if exposed; else
            // expose it - asking first if existing parameters could set it.
            let manager = self.clone();
            ui.global::<PipelinesPanelState>()
                .on_toggle_exposed(move |step_idx, path| {
                    let Some(ui) = manager.ui.upgrade() else {
                        return;
                    };
                    let pipeline_id =
                        ui.global::<PipelinesPanelState>().get_active_pipeline_id() as u32;
                    if manager.viewing(pipeline_id) {
                        manager.sync_steps_of_selected_pipeline_to_slint(
                            PipelineId(pipeline_id),
                            false,
                        );
                        return;
                    }
                    let Some(key) = manager.open_component_key(pipeline_id) else {
                        return;
                    };
                    let step_idx = step_idx as usize;
                    let path = path.to_string();
                    let ask = {
                        let mut project = manager.app_state.get_project_write();
                        let definitions = project.components.clone();
                        let Some(definition) = project.components.iter_mut().find(|d| d.key == key)
                        else {
                            return;
                        };
                        let Some((step_id, (setting, label))) =
                            definition.steps.get(step_idx).and_then(|step| {
                                Some((
                                    step.id,
                                    evanalyzer_app::components::step_setting(
                                        &definitions,
                                        step,
                                        &path,
                                    )?,
                                ))
                            })
                        else {
                            return;
                        };
                        let exposed =
                            evanalyzer_app::components::exposed_settings(definition, step_id)
                                .contains(&path);
                        let candidates = if exposed {
                            Vec::new()
                        } else {
                            evanalyzer_app::components::compatible_parameters(
                                &definitions,
                                &key,
                                &setting,
                            )
                        };
                        if candidates.is_empty() {
                            evanalyzer_app::components::toggle_exposed(
                                definition,
                                step_id,
                                &path,
                                &label,
                                &setting.value,
                            );
                            None
                        } else {
                            Some((label, candidates))
                        }
                    };
                    match ask {
                        None => {
                            manager.pipeline_settings_changed();
                            manager.sync_steps_of_selected_pipeline_to_slint(
                                PipelineId(pipeline_id),
                                false,
                            );
                        }
                        Some((label, candidates)) => {
                            // Each with what it sets already - parameters of equal
                            // steps are often named alike.
                            let (choices, details): (
                                Vec<slint::SharedString>,
                                Vec<slint::SharedString>,
                            ) = {
                                let project = manager.app_state.get_project();
                                std::iter::once(("New parameter".to_string(), String::new()))
                                    .chain(candidates.iter().map(|(id, l)| {
                                        (
                                            format!("Add to “{l}”"),
                                            parameter_target(&project, &key, id)
                                                .map(|target| format!("sets {target}"))
                                                .unwrap_or_default(),
                                        )
                                    }))
                                    .map(|(choice, detail)| (choice.into(), detail.into()))
                                    .unzip()
                            };
                            *manager.pending_expose.lock().expect("Poisoned") = Some((
                                pipeline_id,
                                step_idx,
                                path,
                                candidates.into_iter().map(|(id, _)| id).collect(),
                            ));
                            let choice = ui.global::<ExposeChoiceState>();
                            choice.set_setting_name(label.into());
                            choice.set_choices(ModelRc::new(VecModel::from(choices)));
                            choice.set_details(ModelRc::new(VecModel::from(details)));
                            ui.global::<GlobalAppState>()
                                .set_active_dialog(DialogType::ExposeChoice);
                        }
                    }
                });

            // "Expose setting": 0 = a new parameter, else the n-th candidate.
            let manager = self.clone();
            ui.global::<ExposeChoiceState>().on_chosen(move |choice| {
                let Some(ui) = manager.ui.upgrade() else {
                    return;
                };
                ui.global::<GlobalAppState>()
                    .set_active_dialog(DialogType::None);
                let Some((pipeline_id, step_idx, path, candidates)) =
                    manager.pending_expose.lock().expect("Poisoned").take()
                else {
                    return;
                };
                let Some(key) = manager.open_component_key(pipeline_id) else {
                    return;
                };
                {
                    let mut project = manager.app_state.get_project_write();
                    let definitions = project.components.clone();
                    let Some(definition) = project.components.iter_mut().find(|d| d.key == key)
                    else {
                        return;
                    };
                    let Some((step_id, (setting, label))) =
                        definition.steps.get(step_idx).and_then(|step| {
                            Some((
                                step.id,
                                evanalyzer_app::components::step_setting(
                                    &definitions,
                                    step,
                                    &path,
                                )?,
                            ))
                        })
                    else {
                        return;
                    };
                    match candidates.get((choice as usize).wrapping_sub(1)) {
                        Some(key) if choice > 0 => {
                            if let Err(e) = evanalyzer_app::components::add_binding(
                                definition, key, step_id, &path,
                            ) {
                                warn!("Could not add the setting to the parameter: {e}");
                            }
                        }
                        _ => {
                            evanalyzer_app::components::toggle_exposed(
                                definition,
                                step_id,
                                &path,
                                &label,
                                &setting.value,
                            );
                        }
                    }
                }
                manager.pipeline_settings_changed();
                manager.sync_steps_of_selected_pipeline_to_slint(PipelineId(pipeline_id), false);
            });

            let manager = self.clone();
            ui.global::<ExposeChoiceState>().on_cancel(move || {
                if let Some(ui) = manager.ui.upgrade() {
                    *manager.pending_expose.lock().expect("Poisoned") = None;
                    ui.global::<GlobalAppState>()
                        .set_active_dialog(DialogType::None);
                }
            });

            // "Save to library…" on a component card: the template dialog.
            let manager = self.clone();
            ui.global::<PipelinesPanelState>()
                .on_save_component_to_library(move |step_idx| {
                    let Some(ui) = manager.ui.upgrade() else {
                        return;
                    };
                    let pipeline_id =
                        ui.global::<PipelinesPanelState>().get_active_pipeline_id() as u32;
                    if manager.viewing(pipeline_id) {
                        manager.sync_steps_of_selected_pipeline_to_slint(
                            PipelineId(pipeline_id),
                            false,
                        );
                        return;
                    }
                    let found = {
                        let project = manager.app_state.get_project();
                        manager
                            .shown_steps(&project, pipeline_id)
                            .and_then(|steps| steps.get(step_idx as usize))
                            .and_then(|step| match &step.content {
                                StepContent::Component(instance) => project
                                    .components
                                    .iter()
                                    .find(|d| d.key == instance.key)
                                    .map(|d| (d.key.clone(), d.meta.clone())),
                                StepContent::Command(_) => None,
                            })
                    };
                    if let Some((key, meta)) = found {
                        manager
                            .template_controller
                            .start_component_template_save(key, meta);
                    }
                });

            // "Update to vN" on a component card: asks, then takes the library's
            // version.
            let manager = self.clone();
            ui.global::<PipelinesPanelState>()
                .on_update_component(move |step_idx| {
                    let Some(ui) = manager.ui.upgrade() else {
                        return;
                    };
                    let pipeline_id =
                        ui.global::<PipelinesPanelState>().get_active_pipeline_id() as u32;
                    if manager.viewing(pipeline_id) {
                        manager.sync_steps_of_selected_pipeline_to_slint(
                            PipelineId(pipeline_id),
                            false,
                        );
                        return;
                    }
                    let found = {
                        let project = manager.app_state.get_project();
                        let library = manager.component_library.lock().expect("Poisoned");
                        manager
                            .shown_steps(&project, pipeline_id)
                            .and_then(|steps| steps.get(step_idx as usize))
                            .and_then(|step| match &step.content {
                                StepContent::Component(instance) => {
                                    project.components.iter().find(|d| d.key == instance.key)
                                }
                                StepContent::Command(_) => None,
                            })
                            .and_then(|definition| {
                                let (index, version) =
                                    evanalyzer_app::components::update_available(
                                        definition, &library,
                                    )?;
                                Some((
                                    definition.key.clone(),
                                    definition.meta.name.clone(),
                                    definition.origin.as_ref().map_or(0, |o| o.version),
                                    version,
                                    evanalyzer_app::components::modified_since_inserted(
                                        definition,
                                        &project.components,
                                    ),
                                    index,
                                ))
                            })
                    };
                    let Some((key, name, from, to, changed_here, index)) = found else {
                        return;
                    };
                    *manager.pending_update.lock().expect("Poisoned") = Some((key, index));
                    let state = ui.global::<ComponentUpdateState>();
                    state.set_component_name(name.into());
                    state.set_versions(format!("v{from} → v{to}").into());
                    state.set_changed_here(changed_here);
                    ui.global::<GlobalAppState>()
                        .set_active_dialog(DialogType::ComponentUpdate);
                });

            let manager = self.clone();
            ui.global::<ComponentUpdateState>().on_confirm(move || {
                let Some(ui) = manager.ui.upgrade() else {
                    return;
                };
                ui.global::<GlobalAppState>()
                    .set_active_dialog(DialogType::None);
                let Some((key, index)) = manager.pending_update.lock().expect("Poisoned").take()
                else {
                    return;
                };
                let Some((path, template)) = manager
                    .component_library
                    .lock()
                    .expect("Poisoned")
                    .get(index)
                    .cloned()
                else {
                    return;
                };
                let result = {
                    let mut project = manager.app_state.get_project_write();
                    evanalyzer_app::components::update_from_library(
                        &mut project,
                        &key,
                        &template,
                        &evanalyzer_app::components::library_name(&path),
                    )
                };
                if let Err(e) = result {
                    warn!("Could not update the component: {e}");
                    return;
                }
                manager.pipeline_settings_changed();
                let pipeline_id =
                    ui.global::<PipelinesPanelState>().get_active_pipeline_id() as u32;
                manager.sync_steps_of_selected_pipeline_to_slint(PipelineId(pipeline_id), false);
            });

            let manager = self.clone();
            ui.global::<ComponentUpdateState>().on_cancel(move || {
                if let Some(ui) = manager.ui.upgrade() {
                    *manager.pending_update.lock().expect("Poisoned") = None;
                    ui.global::<GlobalAppState>()
                        .set_active_dialog(DialogType::None);
                }
            });

            // "Create component…" on a step card: the dialog, starting with that step.
            let manager = self.clone();
            ui.global::<PipelinesPanelState>()
                .on_create_component(move |step_idx| {
                    let Some(ui) = manager.ui.upgrade() else {
                        return;
                    };
                    let pipeline_id =
                        ui.global::<PipelinesPanelState>().get_active_pipeline_id() as u32;
                    if manager.viewing(pipeline_id) {
                        manager.sync_steps_of_selected_pipeline_to_slint(
                            PipelineId(pipeline_id),
                            false,
                        );
                        return;
                    }
                    // The steps shown: the pipeline's or an opened component's.
                    let names: Vec<slint::SharedString> = {
                        let project = manager.app_state.get_project();
                        let Some(steps) = manager.shown_steps(&project, pipeline_id) else {
                            return;
                        };
                        // As the cards show them: by their ids.
                        steps
                            .iter()
                            .map(|step| step_ref(&project, step).into())
                            .collect()
                    };
                    let state = ui.global::<CreateComponentState>();
                    state.set_pipeline_id(pipeline_id as i32);
                    state.set_step_names(ModelRc::new(VecModel::from(names)));
                    state.set_name("".into());
                    manager.refresh_component_plan(&ui, step_idx, step_idx);
                    ui.global::<GlobalAppState>()
                        .set_active_dialog(DialogType::CreateComponent);
                });

            let manager = self.clone();
            ui.global::<CreateComponentState>()
                .on_range_changed(move |from, to| {
                    if let Some(ui) = manager.ui.upgrade() {
                        manager.refresh_component_plan(&ui, from.min(to), from.max(to));
                    }
                });

            let manager = self.clone();
            ui.global::<CreateComponentState>()
                .on_role_changed(move |row, role| {
                    let Some(ui) = manager.ui.upgrade() else {
                        return;
                    };
                    let rows = ui.global::<CreateComponentState>().get_classes();
                    if let Some(mut data) = rows.row_data(row as usize) {
                        data.role = role;
                        rows.set_row_data(row as usize, data);
                    }
                });

            let manager = self.clone();
            ui.global::<CreateComponentState>().on_confirm(move || {
                let Some(ui) = manager.ui.upgrade() else {
                    return;
                };
                let state = ui.global::<CreateComponentState>();
                let pipeline_id = PipelineId(state.get_pipeline_id() as u32);
                let range = state.get_from_step() as usize..=state.get_to_step() as usize;
                let roles: Vec<(ObjectClass, ClassRole)> = state
                    .get_classes()
                    .iter()
                    .filter_map(|row| {
                        let class = ObjectClass::from_text(row.class_text.as_str())?;
                        let role = match row.role {
                            0 => ClassRole::Local,
                            1 => ClassRole::Parameter,
                            _ => ClassRole::Fixed,
                        };
                        Some((class, role))
                    })
                    .collect();
                let result = {
                    let mut project = manager.app_state.get_project_write();
                    evanalyzer_app::components::create_component(
                        &mut project,
                        manager.shown_list(pipeline_id.0),
                        range,
                        state.get_name().as_str(),
                        &roles,
                    )
                };
                match result {
                    Ok(key) => {
                        // The user (app settings) is the new component's author.
                        let author = manager
                            .app_state
                            .app_settings
                            .lock()
                            .expect("Poisoned")
                            .author
                            .clone();
                        if let Some(author) = author
                            && let Some(definition) = manager
                                .app_state
                                .get_project_write()
                                .components
                                .iter_mut()
                                .find(|d| d.key == key)
                        {
                            set_first_author(
                                &mut definition.meta.authors,
                                &author.full_name,
                                &author.organization,
                            );
                        }
                        ui.global::<GlobalAppState>()
                            .set_active_dialog(DialogType::None);
                        manager.pipeline_settings_changed();
                        manager.sync_steps_of_selected_pipeline_to_slint(pipeline_id, false);
                    }
                    Err(e) => state.set_error(e.to_string().into()),
                }
            });

            let manager = self.clone();
            ui.global::<CreateComponentState>().on_cancel(move || {
                if let Some(ui) = manager.ui.upgrade() {
                    ui.global::<GlobalAppState>()
                        .set_active_dialog(DialogType::None);
                }
            });

            // Picker: "Remove from project" on an unused component.
            let manager = self.clone();
            ui.global::<CommandPickerState>()
                .on_remove_component(move |command_id| {
                    let Some(ui) = manager.ui.upgrade() else {
                        return;
                    };
                    if !(COMPONENT_PICKER_ID_BASE..LIBRARY_COMPONENT_PICKER_ID_BASE)
                        .contains(&command_id)
                    {
                        return;
                    }
                    let idx = (command_id - COMPONENT_PICKER_ID_BASE) as usize;
                    let result = {
                        let mut project = manager.app_state.get_project_write();
                        match project.components.get(idx).map(|d| d.key.clone()) {
                            Some(key) => {
                                evanalyzer_app::components::remove_component(&mut project, &key)
                            }
                            None => return,
                        }
                    };
                    if let Err(e) = result {
                        warn!("Could not remove the component: {e}");
                        return;
                    }
                    manager.after_components_removed(&ui);
                });

            // Picker: "Remove" next to "N components no pipeline uses".
            let manager = self.clone();
            ui.global::<CommandPickerState>()
                .on_remove_unused_components(move || {
                    let Some(ui) = manager.ui.upgrade() else {
                        return;
                    };
                    let removed = evanalyzer_app::components::remove_unused_components(
                        &mut manager.app_state.get_project_write(),
                    );
                    if !removed.is_empty() {
                        manager.after_components_removed(&ui);
                    }
                });

            // Picker: cancel - close dialog
            let manager = self.clone();
            ui.global::<CommandPickerState>().on_cancel(move || {
                if let Some(ui) = manager.ui.upgrade() {
                    ui.global::<GlobalAppState>()
                        .set_active_dialog(DialogType::None);
                }
            });

            // Picker: import a bioimage.io model - pick its rdf.yaml, configure an
            // AI segmentation command from it, and insert it like a normal step.
            let manager = self.clone();
            ui.global::<CommandPickerState>()
                .on_import_bioimageio(move || {
                    let Some(ui) = manager.ui.upgrade() else {
                        return;
                    };
                    let picker = ui.global::<CommandPickerState>();
                    let pipeline_id = picker.get_pipeline_id() as u32;
                    let after_idx = picker.get_insert_after_idx();
                    if manager.viewing(pipeline_id) {
                        return;
                    }

                    let request = FileRequest::open_file("Import bioimage.io model")
                        .filter("bioimage.io RDF", &["yaml", "yml"]);
                    let manager = manager.clone();
                    manager.app_state.clone().file_browser.open(request, move |path| {
                        // Cancelled: leave the command picker open.
                        let Some(path) = path else {
                            return;
                        };
                        // The RDF is read through the backend, so a model on
                        // the server resolves to server paths.
                        crate::helper::ui_thread::spawn(move || {
                            let result = evanalyzer_app::bioimageio::configure_from(
                                manager.app_state.backend().files(),
                                &path,
                            );
                            let _ = crate::helper::ui_thread::invoke_from_event_loop(move || {
                                let Some(ui) = manager.ui.upgrade() else {
                                    return;
                                };
                                match result {
                                    Ok(configured) => {
                                        let step = PipelineStepSettings {
                                            id: evanalyzer_cfg::core_types::StepId::UNASSIGNED,
                                            enabled: true,
                                            content: evanalyzer_cfg::settings::pipeline_settings::StepContent::Command(configured.command),
                                        };
                                        {
                                            let mut project = manager.app_state.get_project_write();
                                            if let Some(pipeline) =
                                                project.pipelines.iter_mut().find(|p| p.id.0 == pipeline_id)
                                            {
                                                let insert_at = if after_idx < 0 {
                                                    0
                                                } else {
                                                    ((after_idx as usize) + 1).min(pipeline.steps.len())
                                                };
                                                pipeline.steps.insert(insert_at, step);
                                                evanalyzer_app::utils::assign_step_ids(
                                                    &mut pipeline.steps,
                                                );
                                            }
                                        }
                                        ui.global::<GlobalAppState>()
                                            .set_active_dialog(DialogType::None);
                                        manager.pipeline_settings_changed();
                                        manager.sync_steps_of_selected_pipeline_to_slint(
                                            PipelineId(pipeline_id),
                                            false,
                                        );

                                        // Surface any caveats (assumed defaults, required
                                        // normalization, remote weights) so the user can verify.
                                        if !configured.notes.is_empty() {
                                            let body = configured
                                                .notes
                                                .iter()
                                                .map(|n| format!("• {n}"))
                                                .collect::<Vec<_>>()
                                                .join("\n\n");
                                            let msg = format!(
                                                "Imported a model from {}.\n\nPlease review:\n\n{body}",
                                                path.display()
                                            );
                                            let warning = ui.global::<WarningState>();
                                            warning.set_info(true);
                                            warning.set_title("bioimage.io model imported".into());
                                            warning.set_message(msg.into());
                                            ui.global::<GlobalAppState>()
                                                .set_active_dialog(DialogType::Warning);
                                        }
                                    }
                                    Err(e) => {
                                        let warning = ui.global::<WarningState>();
                                        warning.set_info(false);
                                        warning.set_title("bioimage.io import failed".into());
                                        warning.set_message(
                                            format!("Could not import the bioimage.io model:\n\n{e}").into(),
                                        );
                                        ui.global::<GlobalAppState>()
                                            .set_active_dialog(DialogType::Warning);
                                    }
                                }
                            });
                        });
                    });
                });

            // Step parameter changed
            let manager = self.clone();
            ui.global::<PipelinesPanelState>().on_param_changed(
                move |step_idx, parameter_name, value| {
                    let Some(ui) = manager.ui.upgrade() else {
                        return;
                    };
                    let pipeline_id =
                        ui.global::<PipelinesPanelState>().get_active_pipeline_id() as u32;
                    if manager.viewing(pipeline_id) {
                        manager.sync_steps_of_selected_pipeline_to_slint(
                            PipelineId(pipeline_id),
                            false,
                        );
                        return;
                    }
                    let param_name = parameter_name.as_str().to_owned();
                    let value_str = value.as_str().to_owned();

                    // Apply the change and capture the new summary + the updated value
                    // (so we can write it back into the UI model - otherwise the next
                    // re-bind from set_row_data would revert the displayed value to the
                    // pre-change one and dropdown/spinner selections appear "lost").
                    //
                    // Parameter names are either flat ("kernel_size", "criteria.min_area")
                    // or group-item paths ("thresholds.0.method") where the middle segment
                    // is a numeric index into the component. We resolve both shapes from the
                    // ParameterDef tree returned by to_parameters().
                    let is_toggle = value_str.starts_with("toggle:");
                    let nested_path: Option<(String, usize, String)> = {
                        let mut parts = param_name.splitn(3, '.');
                        match (parts.next(), parts.next(), parts.next()) {
                            (Some(g), Some(i), Some(f)) => i
                                .parse::<usize>()
                                .ok()
                                .map(|idx| (g.to_string(), idx, f.to_string())),
                            _ => None,
                        }
                    };

                    // A Script step's `param` lines that don't parse.
                    let mut script_problem: Option<String> = None;
                    let (new_summary, params_now, needs_full_resync) = {
                        let mut project = manager.app_state.get_project_write();
                        let definitions = project.components.clone();
                        let Some(step) = manager
                            .shown_steps_mut(&mut project, pipeline_id)
                            .and_then(|steps| steps.get_mut(step_idx as usize))
                        else {
                            return;
                        };
                        // A component: its parameter is named by its key.
                        let step_id = step.id;
                        if let StepContent::Component(instance) = &mut step.content {
                            if let Err(e) = evanalyzer_app::components::set_component_parameter(
                                instance,
                                &definitions,
                                &ComponentParamKey::from(param_name.as_str()),
                                &value_str,
                            ) {
                                warn!("Component parameter {param_name}: {e}");
                            }
                            // Inside an opened component, a parameter of it may
                            // set this one: the value becomes its default.
                            if let Some(key) = manager.open_component_key(pipeline_id)
                                && let Some(definition) =
                                    project.components.iter_mut().find(|d| d.key == key)
                            {
                                evanalyzer_app::components::bound_setting_edited(
                                    definition,
                                    &definitions,
                                    step_id,
                                    &param_name,
                                );
                            }
                            drop(project);
                            manager.sync_steps_of_selected_pipeline_to_slint(
                                PipelineId(pipeline_id),
                                false,
                            );
                            manager.pipeline_settings_changed();
                            return;
                        }
                        let Some(command) = step.command_mut() else {
                            return;
                        };
                        evanalyzer_app::settings_of::apply_command_param(
                            command,
                            &param_name,
                            &value_str,
                        );

                        // A PixelClassifier/AiObjectClassifier's `segmentation_mapping`
                        // row count is driven by the loaded model's own class count,
                        // not something the user adds/removes rows for - resize it to
                        // match whenever the model changes. This changes the component's
                        // row count (not just one value), so the single-value patch
                        // below can't reflect it; fall back to a full resync instead,
                        // same as a changed field set does.
                        let mut needs_full_resync = false;
                        if param_name == "model_path" {
                            match command {
                                PipelineCommand::PixelClassifier(settings) => {
                                    reconcile_pixel_classifier_mapping(
                                        manager.app_state.backend().files(),
                                        settings,
                                    );
                                    needs_full_resync = true;
                                }
                                PipelineCommand::AiObjectClassifier(settings) => {
                                    reconcile_ai_object_classifier_mapping(
                                        manager.app_state.backend().files(),
                                        settings,
                                    );
                                    needs_full_resync = true;
                                }
                                _ => {}
                            }
                        }

                        // A new script: its `param`s are the step's settings now.
                        if param_name == "source"
                            && let PipelineCommand::Script(settings) = command
                        {
                            needs_full_resync = true;
                            if let Err(e) =
                                evanalyzer_app::settings_of::declared_script_params(&settings.source)
                            {
                                script_problem = Some(e);
                            }
                        }
                        let mut result = (
                            command.to_summary(),
                            evanalyzer_app::settings_of::command_parameters(command),
                            needs_full_resync,
                        );
                        // A setting of the opened component a parameter sets: the
                        // new value becomes the parameter's default (else it
                        // would never run), and the other settings it sets
                        // show it too.
                        if let Some(key) = manager.open_component_key(pipeline_id)
                            && let Some(definition) =
                                project.components.iter_mut().find(|d| d.key == key)
                            && evanalyzer_app::components::bound_setting_edited(
                                definition,
                                &definitions,
                                step_id,
                                &param_name,
                            )
                        {
                            result.2 = true;
                        }
                        result
                    }; // write lock dropped here

                    if let Some(problem) = script_problem {
                        manager.show_warning_titled(
                            "Script parameters",
                            &format!(
                                "A param line of the script has a problem, so the step shows no settings for its params:\n{problem}"
                            ),
                        );
                    }

                    if needs_full_resync {
                        manager.sync_steps_of_selected_pipeline_to_slint(
                            PipelineId(pipeline_id),
                            false,
                        );
                        manager.pipeline_settings_changed();
                        return;
                    }

                    // Update the affected step in the Slint model: summary + the
                    // changed param's value (and, for multi-select toggles, its flags).
                    let model = ui.global::<PipelinesPanelState>().get_active_commands();
                    let Some(mut cmd) = model.row_data(step_idx as usize) else {
                        return;
                    };
                    cmd.summary = new_summary.into();
                    let params = cmd.parameters.clone();
                    let advanced_ui = cmd.advanced_parameters.clone();
                    // Compare/patch against the settings exactly as displayed: advanced
                    // ones split off (and left out while collapsed).
                    let show_advanced = ui
                        .global::<PipelinesPanelState>()
                        .get_always_show_advanced()
                        || cmd.show_advanced;
                    cmd.advanced_changed = advanced_stats(&params_now).1 as i32;
                    let params_now = filter_for_display(params_now, show_advanced);

                    if let Some((group_name, idx, field_name)) = nested_path {
                        // Nested group field: find the group CommandParameter, then either
                        // patch fields[k].value inside group_items[idx], or - if changing this
                        // field altered *which* fields that group item has (e.g. a
                        // "thresholds.0.method" switch to Otsu adding a "method.classes"
                        // dropdown, or a "thresholds.0.method.classes" switch to Three adding
                        // "method.classes.middleClass") - fall back to a full resync, the same
                        // "field set changed" guard the flat (non-grouped) case below already
                        // has. Without this, a nested rich-enum variant switch inside a group
                        // item only took effect after some *other*, unrelated change happened
                        // to trigger a full resync.
                        let new_group_fields = params_now
                            .iter()
                            .find(|p| p.name == group_name)
                            .and_then(|p| p.groups.get(idx));
                        let new_field_names: Vec<String> = new_group_fields
                            .map(|fields| fields.iter().map(|fd| fd.name.clone()).collect())
                            .unwrap_or_default();

                        let mut old_field_names = Vec::new();
                        let mut old_fields_model = None;
                        for i in 0..params.row_count() {
                            if let Some(p) = params.row_data(i) {
                                if p.name.as_str() == group_name {
                                    if let Some(item) = p.group_items.row_data(idx) {
                                        let fields = item.fields.clone();
                                        old_field_names = (0..fields.row_count())
                                            .filter_map(|k| {
                                                fields.row_data(k).map(|fd| fd.name.to_string())
                                            })
                                            .collect();
                                        old_fields_model = Some(fields);
                                    }
                                    break;
                                }
                            }
                        }

                        if old_field_names != new_field_names {
                            manager.sync_steps_of_selected_pipeline_to_slint(
                                PipelineId(pipeline_id),
                                false,
                            );
                            manager.pipeline_settings_changed();
                            return;
                        }

                        let new_param_value = new_group_fields
                            .and_then(|item| item.iter().find(|fd| fd.name == field_name))
                            .map(|fd| fd.value.clone())
                            .unwrap_or_default();
                        if let Some(fields) = old_fields_model {
                            for k in 0..fields.row_count() {
                                if let Some(mut lp) = fields.row_data(k) {
                                    if lp.name.as_str() == field_name {
                                        lp.value = new_param_value.clone().into();
                                        fields.set_row_data(k, lp);
                                        break;
                                    }
                                }
                            }
                        }
                        model.set_row_data(step_idx as usize, cmd);
                    } else {
                        // Some fields (e.g. a "function" dropdown backed by a Rust enum whose
                        // variants each carry their own sub-fields) change *which* parameters
                        // exist, not just one value, when switched. Patching a single row in
                        // that case would leave stale fields on screen or miss new ones, so
                        // detect a changed field set and fall back to a full resync — the same
                        // thing add_group_item/remove_group_item already do for the other case
                        // where the parameter list's shape can change.
                        let old_names: Vec<String> = (0..params.row_count())
                            .filter_map(|i| params.row_data(i).map(|p| p.name.to_string()))
                            .chain((0..advanced_ui.row_count()).filter_map(|i| {
                                advanced_ui.row_data(i).map(|p| p.name.to_string())
                            }))
                            .collect();
                        let new_names: Vec<String> = params_now
                            .iter()
                            .filter(|p| !p.advanced)
                            .chain(params_now.iter().filter(|p| p.advanced && show_advanced))
                            .map(|p| p.name.clone())
                            .collect();
                        if old_names != new_names {
                            manager.sync_steps_of_selected_pipeline_to_slint(
                                PipelineId(pipeline_id),
                                false,
                            );
                            manager.pipeline_settings_changed();
                            return;
                        }

                        // Flat parameter, field set unchanged: patch the single value (and
                        // flags for multi-select) in place.
                        let new_param_value = params_now
                            .into_iter()
                            .find(|p| p.name == param_name)
                            .map(|p| p.value)
                            .unwrap_or_default();
                        'patch: for list in [&params, &advanced_ui] {
                            for i in 0..list.row_count() {
                                if let Some(mut p) = list.row_data(i) {
                                    if p.name.as_str() == param_name {
                                        p.value = new_param_value.clone().into();
                                        if is_toggle {
                                            let selected: std::collections::HashSet<u32> =
                                                new_param_value
                                                    .split(',')
                                                    .filter_map(|s| s.trim().parse::<u32>().ok())
                                                    .collect();
                                            let selected_local: Vec<&str> =
                                                new_param_value.split(',').map(str::trim).collect();
                                            let new_flags: Vec<SharedString> = (0u32..33u32)
                                                .map(|idx| selected.contains(&idx))
                                                // Then the opened component's classes.
                                                .chain(
                                                    manager
                                                        .open_component_class_ids(pipeline_id)
                                                        .into_iter()
                                                        .map(|id| {
                                                            selected_local.contains(
                                                                &ObjectClass::Local(id)
                                                                    .to_text()
                                                                    .as_str(),
                                                            )
                                                        }),
                                                )
                                                .map(|on| if on { "1" } else { "0" }.into())
                                                .collect();
                                            p.options = ModelRc::new(VecModel::from(new_flags));
                                        }
                                        // The rows show copies: they take it too.
                                        patch_card_copies(&cmd, &p);
                                        list.set_row_data(i, p);
                                        break 'patch;
                                    }
                                }
                            }
                        }
                        model.set_row_data(step_idx as usize, cmd);
                    }
                    manager.pipeline_settings_changed();
                },
            );

            // Add group item
            let manager = self.clone();
            ui.global::<PipelinesPanelState>()
                .on_add_group_item(move |step_id, param_name| {
                    if let Some(ui) = manager.ui.upgrade() {
                        let pipeline_id =
                            ui.global::<PipelinesPanelState>().get_active_pipeline_id() as u32;
                        manager.modify_group_item(
                            pipeline_id,
                            step_id as usize,
                            param_name.as_str(),
                            true,
                            None,
                        );
                        manager.pipeline_settings_changed();
                    }
                });

            // Remove group item
            let manager = self.clone();
            ui.global::<PipelinesPanelState>().on_remove_group_item(
                move |step_id, param_name, item_idx| {
                    if let Some(ui) = manager.ui.upgrade() {
                        let pipeline_id =
                            ui.global::<PipelinesPanelState>().get_active_pipeline_id() as u32;
                        manager.modify_group_item(
                            pipeline_id,
                            step_id as usize,
                            param_name.as_str(),
                            false,
                            Some(item_idx as usize),
                        );
                        manager.pipeline_settings_changed();
                    }
                },
            );

            // Show a PixelClassifier/AiObjectClassifier step's loaded model info
            // (metadata + classes)
            let manager = self.clone();
            ui.global::<PipelinesPanelState>()
                .on_show_model_info(move |step_id| {
                    manager.show_classifier_model_info(step_id as usize);
                });

            // Browse for a file (e.g. a TorchScript model path) - opens the
            // file browser filtered by the given comma-separated extensions,
            // starting next to `current_path`. The result goes back to the
            // row that asked via its `token` (see `browse-file` in Slint).
            let manager = self.clone();
            ui.global::<PipelinesPanelState>().on_browse_file(
                move |extensions_csv, current_path, token| {
                    let extensions: Vec<&str> = extensions_csv
                        .split(',')
                        .map(str::trim)
                        .filter(|e| !e.is_empty())
                        .collect();
                    let mut request =
                        FileRequest::open_file("Choose file").start_in(current_path.as_str());
                    if !extensions.is_empty() {
                        request = request.filter("Allowed files", &extensions);
                    }
                    let ui_weak = manager.ui.clone();
                    manager.app_state.file_browser.open(request, move |path| {
                        let (Some(path), Some(ui)) = (path, ui_weak.upgrade()) else {
                            return;
                        };
                        let state = ui.global::<PipelinesPanelState>();
                        state.set_browse_result_path(path.to_string_lossy().as_ref().into());
                        state.set_browse_result_token(token);
                    });
                },
            );
        }

        // Must be called onece at startup
        self.sync_commands_to_selection_dialog_slint();
        // The component library, for "Update" on component cards.
        self.reload_component_library_async();
    }

    /// Builds the `ProjectSettings` for a single-image preview run.
    ///
    /// Clones every project-level field except `images.list`: that map holds
    /// one `ImageEntry` per project image, each carrying its own
    /// `SeriesSettings::objects` (full `ObjectMetricSettings`, mask data included) - so a
    /// naive `ProjectSettings::clone()` followed by `list.clear()` pays to
    /// clone every other image's object results just to immediately discard
    /// them. Building `list` fresh with only the preview image skips that
    /// entirely.
    fn build_preview_project_settings(
        project: &ProjectSettings,
        image_path: PathBuf,
        mut image_settings: ImageEntry,
    ) -> ProjectSettings {
        // Preview must only ever analyze the single time frame currently
        // shown in the viewer - never every t-stack. The viewer forces
        // `TStackHandling::AllStacks` into the project's global settings on
        // every channel/z/t interaction (see `update_channel_options_in_project`
        // in viewport_image_controller.rs) so the *real* run always
        // processes every frame regardless of whatever the viewer last
        // happened to show (see `trigger_pipeline_full_run`) - but that
        // same global setting is what Preview would otherwise inherit
        // unchanged, silently previewing every frame instead of just the
        // one being looked at. `job_executor::prepare_t_stack_iterator`
        // checks the image's own series override before falling back to
        // the project-wide setting, so both are forced to `SingleStack`
        // here to guarantee one frame either way, keeping whichever t index
        // was actually active.
        let active = image_settings.active_series(&project.images.settings);
        let effective_t = image_settings
            .series
            .get(&active)
            .and_then(|series| series.t_stack.clone())
            .or_else(|| project.images.settings.t_stack.clone())
            .unwrap_or_default();
        let single_stack_t = TStackSettings {
            stack_handling: TStackHandling::SingleStack,
            ..effective_t
        };
        if let Some(series) = image_settings.series.get_mut(&active) {
            series.t_stack = Some(single_stack_t.clone());
        }
        let mut settings = project.images.settings.clone();
        settings.t_stack = Some(single_stack_t);

        let mut list = indexmap::IndexMap::with_capacity(1);
        list.insert(image_path, image_settings);
        ProjectSettings {
            schema_version: project.schema_version,
            meta: project.meta.clone(),
            classification: project.classification.clone(),
            plate: project.plate.clone(),
            images: ImageSettings {
                root: project.images.root.clone(),
                list,
                settings,
            },
            pipelines: project.pipelines.clone(),
            tile_merge: project.tile_merge.clone(),
            components: project.components.clone(),
        }
    }

    /// Forces a real (non-preview) analysis run to always process every
    /// t-stack, regardless of whatever `TStackHandling` the viewer last left
    /// in the project's settings. The viewer writes
    /// `TStackHandling::AllStacks` into the project's global settings on
    /// every channel/z/t interaction (see `update_channel_options_in_project`
    /// in viewport_image_controller.rs) - which happened to make a full run
    /// already process every frame, but only as an accidental side effect of
    /// that same write, not because a full run actually asked for it. Forcing
    /// it explicitly here means a full run's time coverage no longer depends
    /// on what the viewer happened to be showing (or a future fix to that
    /// viewer behavior - see `build_preview_project_settings`, which forces
    /// the opposite, a single frame, the same explicit way). Both the
    /// project-wide default and any per-image series override are forced,
    /// since `job_executor::prepare_t_stack_iterator` checks the series
    /// override first.
    fn force_all_t_stacks(settings: &mut ProjectSettings) {
        let (playback_speed, frame_interval) = settings
            .images
            .settings
            .t_stack
            .as_ref()
            .map(|t| (t.playback_speed, t.frame_interval))
            .unwrap_or_default();
        settings.images.settings.t_stack = Some(TStackSettings {
            stack_handling: TStackHandling::AllStacks,
            playback_speed,
            t_stack: 0,
            // The project's frame interval goes into the analysis as set.
            frame_interval,
        });
        for image in settings.images.list.values_mut() {
            for series in image.series.values_mut() {
                if let Some(t_stack) = series.t_stack.as_mut() {
                    t_stack.stack_handling = TStackHandling::AllStacks;
                }
            }
        }
    }

    /// Dispatches a lightweight, single-image pipeline execution task for real-time UI preview.
    ///
    /// This function acts as a safety-gated entry point for the preview system. It validates
    /// that a valid project layout and an active target image are loaded, isolates the selected
    /// image into a temporary project scope to avoid processing the entire dataset, updates the global
    /// UI progress state, and offloads the workflow to the background worker thread.
    ///
    /// # Behavior & Side Effects
    /// 1. **Early Return Guardrails**: Logs a `warn!` message and aborts immediately if the project path,
    ///    target image path, or structural image metadata cannot be resolved.
    /// 2. **Isolates Preview Scope**: Clones project settings but wipes the multi-image queue, inserting
    ///    *only* the currently active image to minimize processing times.
    /// 3. **UI State Transition**: Mutates global Slint/UI properties to reset progress tracking metrics
    ///    to zero and forcefully triggers the `PipelineRunning` overlay dialog screen.
    /// 4. **Asynchronous Dispatch**: Offloads the generated `PipelineTask` directly to the pipeline worker.
    pub fn trigger_pipeline_preview_execution(&self) {
        // Auto preview fires this from a debounced timer (see
        // `pipeline_settings_changed`), completely independent of whatever
        // dialog the user currently has open - e.g. opening the "New
        // pipeline" dialog changes pipeline settings, which arms this same
        // timer. Below, a successful run force-closes the dialog
        // (`DialogType::None`) once it completes, and starting one
        // force-opens `PreviewRendering` - either would silently steal focus
        // from/dismiss an unrelated modal like the pipeline-edit dialog if
        // this fired while one was open. Skip this run entirely rather than
        // fight over `active_dialog`; the next real settings change re-arms
        // the debounce and tries again once the user isn't mid-dialog.
        if let Some(ui) = self.app_state.ui_handle.upgrade() {
            let active_dialog = ui.global::<GlobalAppState>().get_active_dialog();
            if active_dialog != DialogType::None && active_dialog != DialogType::PreviewRendering {
                debug!("Skipping auto preview - {:?} dialog is open", active_dialog);
                return;
            }
        }

        let project = self.app_state.get_project();

        let Some(current_project) = &project.tmp_settings.current_project else {
            warn!("No project path set, please save project first!");
            self.show_warning(
                "No project is open. Please save the project first before running a preview.",
            );
            return;
        };

        let Some(current_image_path) = project.get_current_rel_image_path_cloned() else {
            warn!("Selected image not found in project!");
            self.show_warning(
                "No image is selected. Please select an image before running a preview.",
            );
            return;
        };

        let Some(current_image_settings) = project.get_current_image_settings() else {
            warn!("Selected image not found in project!");
            self.show_warning("The selected image could not be found in the project. Please select a valid image before running a preview.");
            return;
        };

        info!("Started preview for {:?}", current_image_path);

        // Only the selected image is needed for a preview - build the settings
        // fresh instead of cloning every other image's settings (and their
        // stored object results) just to discard them.
        let project_tmp = Self::build_preview_project_settings(
            &project.settings,
            current_image_path,
            current_image_settings.clone(),
        );

        let breakpoint = self
            .breakpoint
            .lock()
            .unwrap()
            .map(|(pid, sid, mode)| (evanalyzer_cfg::core_types::PipelineId(pid), sid, mode));

        let task: PipelineTask = PipelineTask {
            project_settings: project_tmp,
            project_path: current_project
                .parent()
                .unwrap_or(current_project)
                .to_path_buf(),
            preview: true,
            breakpoint,
            job_name: None,
            attach: None,
            base_results: None,
        };
        drop(project);

        if let Some(ui) = self.app_state.ui_handle.upgrade() {
            ui.global::<PipelineRunningState>().set_processed(0);
            ui.global::<PipelineRunningState>().set_total(0);
            ui.global::<GlobalAppState>()
                .set_active_dialog(DialogType::PreviewRendering);
        }

        self.dispatch_worker_task(task);
    }

    /// Opens the "Run Analysis" confirm dialog (job name, optional) before
    /// actually starting a full run. Bails out immediately - same as the old
    /// pre-dialog behavior - if there's no saved project to run against, so
    /// the user isn't asked for a job name only to hit that error afterward.
    pub fn open_run_analysis_dialog(self: &Arc<Self>) {
        let has_project_path = self
            .app_state
            .get_project()
            .tmp_settings
            .current_project
            .is_some();
        if !has_project_path {
            warn!("No project path set, please save project first!");
            self.show_warning(
                "No project is open. Please save the project first before starting an analysis.",
            );
            return;
        }

        let Some(ui) = self.ui.upgrade() else {
            return;
        };
        let has_dataset_steps = self
            .app_state
            .get_project()
            .settings
            .pipelines
            .iter()
            .filter(|p| p.enabled)
            .flat_map(|p| &p.steps)
            .any(|s| {
                s.enabled
                    && s.command()
                        .is_some_and(|c| matches!(c.category(), CommandCategory::Dataset))
            });
        // The earlier analyses, as the results list shows them (newest
        // first); the one open there is the default basis.
        let results = ui.global::<ResultsListState>();
        let earlier: Vec<ResultItemData> = results.get_results_list().iter().collect();
        let open = usize::try_from(results.get_selected_result_index())
            .ok()
            .filter(|i| *i < earlier.len())
            .unwrap_or(0);
        let names: Vec<SharedString> = earlier
            .iter()
            .map(|r| format!("{} ({})", r.name, r.modified).into())
            .collect();
        let paths: Vec<SharedString> = earlier.iter().map(|r| r.path.clone()).collect();

        let state = ui.global::<RunAnalysisState>();
        state.set_job_name("".into());
        state.set_dataset_only(false);
        state.set_has_dataset_steps(has_dataset_steps);
        state.set_base_names(ModelRc::new(VecModel::from(names)));
        state.set_base_paths(ModelRc::new(VecModel::from(paths)));
        state.set_base_index(open as i32);
        ui.global::<GlobalAppState>()
            .set_active_dialog(DialogType::RunAnalysisConfirm);
    }

    /// The "Run Analysis" dialog was confirmed: read back the (optional) job
    /// name, close the dialog, and start the run.
    fn on_run_analysis_confirmed(self: &Arc<Self>) {
        let Some(ui) = self.ui.upgrade() else {
            return;
        };
        let state = ui.global::<RunAnalysisState>();
        let job_name = state.get_job_name().to_string();
        let base_results = state
            .get_dataset_only()
            .then(|| {
                state
                    .get_base_paths()
                    .row_data(state.get_base_index() as usize)
            })
            .flatten()
            .map(|path| PathBuf::from(path.as_str()));
        ui.global::<GlobalAppState>()
            .set_active_dialog(DialogType::None);

        let job_name = (!job_name.trim().is_empty()).then_some(job_name);
        self.trigger_pipeline_full_run(job_name, base_results);
    }

    /// Starts a full analysis - or, with `base_results`, a run of only the
    /// dataset steps on a copy of that earlier analysis's results.
    fn trigger_pipeline_full_run(&self, job_name: Option<String>, base_results: Option<PathBuf>) {
        let project = self.app_state.get_project();

        let Some(current_project) = &project.tmp_settings.current_project else {
            warn!("No project path set, please save project first!");
            self.show_warning(
                "No project is open. Please save the project first before starting an analysis.",
            );
            return;
        };

        let mut project_settings = project.settings.clone();
        Self::force_all_t_stacks(&mut project_settings);

        // User settings from app settings if available
        if let Some(user_info) = &self.app_state.app_settings.lock().expect("Poisened").author {
            project_settings.meta.authors.push(AuthorInformation {
                full_name: user_info.full_name.clone(),
                organization: user_info.organization.clone(),
            });
        }

        let task: PipelineTask = PipelineTask {
            project_settings,
            project_path: current_project
                .parent()
                .unwrap_or(current_project)
                .to_path_buf(),
            preview: false,
            breakpoint: None,
            job_name,
            attach: None,
            base_results,
        };
        drop(project);

        if let Some(ui) = self.app_state.ui_handle.upgrade() {
            ui.global::<PipelineRunningState>().set_processed(0);
            ui.global::<PipelineRunningState>().set_total(0);
            ui.global::<GlobalAppState>()
                .set_active_dialog(DialogType::PipelineRunning);
        }

        // A local analysis gets the RAM of the viewport caches; a server's
        // doesn't need this computer's.
        if !self.app_state.backend().is_remote() {
            self.viewport_controller
                .release_preview_memory_for_analysis();
        }

        self.dispatch_worker_task(task);
    }

    /// "Save pipeline as component" for the active pipeline: the metadata
    /// dialog, then the file. A pipeline without steps has nothing to save.
    fn save_pipeline_as_component(self: &Arc<Self>) {
        let Some(ui) = self.ui.upgrade() else {
            return;
        };
        let panel = ui.global::<PipelinesPanelState>();
        let pipeline_id = PipelineId(panel.get_active_pipeline_id() as u32);
        let has_steps = self
            .app_state
            .get_project()
            .pipelines
            .iter()
            .any(|p| p.id == pipeline_id && !p.steps.is_empty());
        if !has_steps {
            let warning = ui.global::<WarningState>();
            warning.set_info(true);
            warning.set_title("Nothing to save".into());
            warning.set_message("The pipeline has no steps yet.".into());
            ui.global::<GlobalAppState>()
                .set_active_dialog(DialogType::Warning);
            return;
        }
        let name = panel.get_active_pipeline_name().to_string();
        self.template_controller
            .start_pipeline_component_save(pipeline_id, name);
    }

    /// A pipeline setting has been changed
    ///
    /// Marks the settings as dirty and triggers a preview update if
    /// auto preview is enabled.
    ///
    /// The preview execution is debounced: each change cancels any pending
    /// preview and restarts a single-shot timer, so the (expensive) preview
    /// only runs once the user has stopped editing for `PREVIEW_DEBOUNCE_MS`.
    /// This avoids a flood of preview refreshes while the user is still typing.
    /// Connects the pipeline focus (see `focus_controller`).
    fn pipeline_settings_changed(self: &Arc<Self>) {
        // Values for parameters a component no longer has go.
        evanalyzer_app::components::prune_component_values(&mut self.app_state.get_project_write());
        self.app_state.mark_dirty();
        self.focus_controller.refresh();
        self.refresh_problems();

        // Trigger preview if auto preview is enabled
        let auto_preview = *self.auto_preview_enabled.lock().expect("Poisned");
        if auto_preview {
            let this = self.clone();
            PREVIEW_DEBOUNCE.with(|timer| {
                // Cancel any fire still pending from a previous change
                timer.stop();
                timer.start(
                    slint::TimerMode::SingleShot,
                    std::time::Duration::from_millis(PREVIEW_DEBOUNCE_MS),
                    move || {
                        debug!("Auto preview triggered (debounced)!");
                        this.trigger_pipeline_preview_execution();
                    },
                );
            });
        }
    }

    /// Dispatches a drawing task to the background worker threads based on the specified scope.
    ///
    /// This method manages the distribution of rendering work to either the low-resolution
    /// preview pipeline, the high-resolution production pipeline, or both. It uses a
    /// condition variable pattern to wake up waiting worker threads after updating
    /// the atomic task counters.
    ///
    /// ### Arguments
    /// * `task` - The `DrawingTask` containing the parameters and data required for the render.
    ///   should receive the task.
    ///
    /// ### Implementation Details
    /// The function uses an internal helper closure `notify` to:
    /// 1. Acquire the mutex lock on a task slot.
    /// 2. Inject the new task into the slot.
    /// 3. Signal the `Condvar` to wake up a blocked worker thread.
    fn dispatch_worker_task(&self, task: PipelineTask) {
        let notify = |pair: &Arc<(Mutex<Option<PipelineTask>>, Condvar)>, t: PipelineTask| {
            let (lock, cvar) = &**pair;
            let mut slot = lock.lock().unwrap();
            *slot = Some(t);
            cvar.notify_one();
        };

        notify(&self.task_request, task);
    }

    /// On a server: follows the analysis running there, as if this window had
    /// started it - one started before the window opened, or before the
    /// connection dropped. Silently: the running dialog just appears. And
    /// tells about analyses that ended while no window was following them.
    /// Runs the server queries off the UI thread.
    pub(crate) fn follow_server_analyses(self: &Arc<Self>) {
        let backend = self.app_state.backend();
        if !backend.is_remote() {
            return;
        }
        let manager = Arc::clone(self);
        crate::helper::ui_thread::spawn(move || {
            let jobs = match backend.list_jobs() {
                Ok(jobs) => jobs,
                Err(e) => return warn!("Could not list the server's analyses: {e}"),
            };
            let ended: Vec<_> = jobs.iter().filter(|job| !job.is_running()).collect();
            if !ended.is_empty() {
                let lines: Vec<String> = ended
                    .iter()
                    .map(|job| format!("'{}': {}", job.name(), describe_end(&job.state)))
                    .collect();
                manager.show_info(
                    "Analyses on the server",
                    &format!(
                        "While no window was following them, these analyses ended:\n\n{}\n\n\
                         Their results are in the project's results folder.",
                        lines.join("\n")
                    ),
                );
                for job in &ended {
                    let _ = backend.forget_job(&job.id);
                }
            }
            if let Some(running) = jobs.iter().find(|job| job.is_running()) {
                info!("Following analysis {} running on the server", running.id);
                let id = running.id.clone();
                let manager = Arc::clone(&manager);
                let _ = crate::helper::ui_thread::invoke_from_event_loop(move || {
                    if let Some(ui) = manager.app_state.ui_handle.upgrade() {
                        ui.global::<PipelineRunningState>().set_processed(0);
                        ui.global::<PipelineRunningState>().set_total(0);
                        ui.global::<GlobalAppState>()
                            .set_active_dialog(DialogType::PipelineRunning);
                    }
                    manager.dispatch_worker_task(PipelineTask {
                        attach: Some(id),
                        ..PipelineTask::default()
                    });
                });
            }
        });
    }

    /// Shows the generic warning dialog as an information.
    fn show_info(&self, title: &str, message: &str) {
        let (title, message) = (title.to_owned(), message.to_owned());
        let ui_weak = self.ui.clone();
        let _ = crate::helper::ui_thread::invoke_from_event_loop(move || {
            if let Some(ui) = ui_weak.upgrade() {
                let warning = ui.global::<WarningState>();
                warning.set_info(true);
                warning.set_title(title.into());
                warning.set_message(message.into());
                ui.global::<GlobalAppState>()
                    .set_active_dialog(DialogType::Warning);
            }
        });
    }

    /// Shows the generic warning dialog with `message`.
    ///
    /// Used at the early-return guard points in the pipeline trigger functions so
    /// the user gets visible feedback instead of a silent log-only `warn!`.
    fn show_warning(&self, message: &str) {
        self.show_warning_titled("Cannot start analysis", message);
    }

    fn show_warning_titled(&self, title: &str, message: &str) {
        let title = title.to_owned();
        let message = message.to_owned();
        let ui_weak = self.ui.clone();
        if let Err(e) = crate::helper::ui_thread::invoke_from_event_loop(move || {
            if let Some(ui) = ui_weak.upgrade() {
                let warning = ui.global::<WarningState>();
                warning.set_info(false);
                warning.set_title(title.into());
                warning.set_message(message.into());
                ui.global::<GlobalAppState>()
                    .set_active_dialog(DialogType::Warning);
            }
        }) {
            warn!("Failed to show warning dialog: {e}");
        }
    }

    /// Turns auto-preview off, e.g. when a preview run is rejected for covering
    /// too many tiles - otherwise the next debounced settings change would
    /// immediately retrigger and get rejected again while the user is still
    /// zoomed out.
    pub(crate) fn disable_auto_preview(&self) {
        *self.auto_preview_enabled.lock().expect("Poisned") = false;
        let ui_weak = self.ui.clone();
        if let Err(e) = crate::helper::ui_thread::invoke_from_event_loop(move || {
            if let Some(ui) = ui_weak.upgrade() {
                ui.global::<PipelinesPanelState>()
                    .set_auto_preview_enabled(false);
            }
        }) {
            warn!("Failed to disable auto-preview toggle in Slint: {e}");
        }
    }

    /// Loads the model currently set on a `PixelClassifier`/`AiObjectClassifier`
    /// step's `model_path` and shows its metadata via the shared warning/info
    /// dialog - see `pipelines_controller`'s module doc comment on why this
    /// reuses that dialog instead of a bespoke one.
    /// The components opened in pipeline `pipeline_id`'s panel, outermost
    /// first; empty if none.
    fn open_component_path(&self, pipeline_id: u32) -> Vec<OpenComponent> {
        self.open_component
            .lock()
            .expect("Poisoned")
            .as_ref()
            .filter(|(pipeline, _)| *pipeline == pipeline_id)
            .map(|(_, path)| path.clone())
            .unwrap_or_default()
    }

    /// The key of the component shown in pipeline `pipeline_id`'s panel, if any.
    fn open_component_key(&self, pipeline_id: u32) -> Option<String> {
        self.open_component_path(pipeline_id)
            .pop()
            .map(|level| level.key)
    }

    /// Whether pipeline `pipeline_id`'s panel shows a component only to look at.
    fn viewing(&self, pipeline_id: u32) -> bool {
        self.open_component_path(pipeline_id)
            .iter()
            .any(|level| level.viewing_from.is_some())
    }

    /// The steps shown in pipeline `pipeline_id`'s panel, as a
    /// [`StepList`](evanalyzer_app::components::StepList).
    fn shown_list(&self, pipeline_id: u32) -> evanalyzer_app::components::StepList {
        match self.open_component_key(pipeline_id) {
            Some(key) => evanalyzer_app::components::StepList::Component(key),
            None => evanalyzer_app::components::StepList::Pipeline(PipelineId(pipeline_id)),
        }
    }

    /// The classes (their numbers) of the component shown in pipeline
    /// `pipeline_id`'s panel - none in a pipeline.
    fn open_component_class_ids(&self, pipeline_id: u32) -> Vec<u32> {
        let Some(key) = self.open_component_key(pipeline_id) else {
            return Vec::new();
        };
        self.app_state
            .get_project()
            .components
            .iter()
            .find(|d| d.key == key)
            .map(|d| d.classes.iter().map(|c| c.id).collect())
            .unwrap_or_default()
    }

    /// Closes every opened component, if any.
    fn close_open_component(&self) {
        *self.open_component.lock().expect("Poisoned") = None;
    }

    /// Opens exactly the components `path` (outermost first; empty: none).
    fn set_open_path(&self, pipeline_id: u32, path: Vec<OpenComponent>) {
        *self.open_component.lock().expect("Poisoned") =
            (!path.is_empty()).then_some((pipeline_id, path));
    }

    /// "Edit" on component card `step_idx` of the steps shown with the
    /// components `base` opened (empty: the pipeline's): opens it to edit -
    /// asking first if other steps use the component too. Until that's
    /// answered the panel stays as it is.
    fn edit_component(
        self: &Arc<Self>,
        ui: &AppWindow,
        pipeline_id: u32,
        base: Vec<OpenComponent>,
        step_idx: usize,
    ) {
        let shown = self.open_component_path(pipeline_id);
        self.set_open_path(pipeline_id, base.clone());
        let found = {
            let project = self.app_state.get_project();
            self.shown_steps(&project, pipeline_id)
                .and_then(|steps| steps.get(step_idx))
                .and_then(|step| match &step.content {
                    StepContent::Component(instance) => Some((
                        step_label(&project, step),
                        evanalyzer_app::components::component_uses(&project, &instance.key),
                    )),
                    StepContent::Command(_) => None,
                })
        };
        match found {
            Some((name, uses)) if uses > 1 => {
                self.set_open_path(pipeline_id, shown);
                *self.pending_edit_base.lock().expect("Poisoned") = Some(base);
                let choice = ui.global::<ComponentEditChoiceState>();
                choice.set_pipeline_id(pipeline_id as i32);
                choice.set_step_idx(step_idx as i32);
                choice.set_component_name(name.into());
                choice.set_uses(uses as i32);
                ui.global::<GlobalAppState>()
                    .set_active_dialog(DialogType::ComponentEditChoice);
            }
            Some(_) => self.open_component_in_panel(pipeline_id, step_idx, false),
            None => self.set_open_path(pipeline_id, shown),
        }
    }

    /// "Edit for all"/"Only this step" answered: the opened components the
    /// card was shown with (see [`Self::edit_component`]).
    fn take_pending_edit_base(&self, pipeline_id: u32) {
        if let Some(base) = self.pending_edit_base.lock().expect("Poisoned").take() {
            self.set_open_path(pipeline_id, base);
        }
    }

    /// "‹ Back": closes the shown component - the one it was opened in, or the
    /// pipeline, is shown again.
    fn close_component_level(&self) {
        let mut open = self.open_component.lock().expect("Poisoned");
        if let Some((_, path)) = open.as_mut() {
            path.pop();
            if path.is_empty() {
                *open = None;
            }
        }
    }

    /// Opens the component settings dialog for the opened component - read only
    /// while it's only viewed - at parameter `focus`, if given.
    fn show_component_settings(&self, focus: Option<ComponentParamKey>) {
        let Some(ui) = self.ui.upgrade() else {
            return;
        };
        let pipeline_id = ui.global::<PipelinesPanelState>().get_active_pipeline_id() as u32;
        // Only for a component opened to edit.
        if self.viewing(pipeline_id) {
            return;
        }
        let Some(key) = self.open_component_key(pipeline_id) else {
            return;
        };
        let focus_index = {
            let project = self.app_state.get_project();
            let Some(definition) = project.components.iter().find(|d| d.key == key) else {
                return;
            };
            let state = ui.global::<ComponentSettingsState>();
            state.set_name(definition.meta.name.clone().into());
            state.set_description(definition.meta.short_description.clone().into());
            state.set_long_description(definition.meta.description.clone().into());
            let author = definition.meta.authors.first();
            state.set_author_name(
                author
                    .map(|a| a.full_name.clone())
                    .unwrap_or_default()
                    .into(),
            );
            state.set_author_organization(
                author
                    .map(|a| a.organization.clone())
                    .unwrap_or_default()
                    .into(),
            );
            focus
                .and_then(|key| definition.parameters.iter().position(|p| p.key == key))
                .map_or(-1, |index| index as i32)
        };
        // Opened at a parameter (✎ next to an exposed setting): it's being
        // edited from the start.
        *self.parameter_edit_snapshot.lock().expect("Poisoned") = if focus_index >= 0 {
            self.app_state
                .get_project()
                .components
                .iter()
                .find(|d| d.key == key)
                .cloned()
        } else {
            None
        };
        self.fill_component_settings(focus_index);
        ui.global::<GlobalAppState>()
            .set_active_dialog(DialogType::ComponentSettings);
    }

    /// The component settings dialog's parameter rows, from the opened component.
    /// Gives the opened component's parameters the keys their sections and
    /// labels make - its uses follow (see `rekey_parameters`). Done when a
    /// parameter's edit is done, not while: the settings dialog's rows name
    /// the parameters by their keys.
    fn rekey_open_component(&self) {
        let Some(ui) = self.ui.upgrade() else {
            return;
        };
        let pipeline_id = ui.global::<PipelinesPanelState>().get_active_pipeline_id() as u32;
        if self.viewing(pipeline_id) {
            return;
        }
        let Some(key) = self.open_component_key(pipeline_id) else {
            return;
        };
        let renamed = evanalyzer_app::components::rekey_parameters(
            &mut self.app_state.get_project_write(),
            &key,
        );
        if renamed > 0 {
            self.app_state.mark_dirty();
        }
    }

    fn fill_component_settings(&self, focus_index: i32) {
        let Some(ui) = self.ui.upgrade() else {
            return;
        };
        let pipeline_id = ui.global::<PipelinesPanelState>().get_active_pipeline_id() as u32;
        let Some(key) = self.open_component_key(pipeline_id) else {
            return;
        };
        let project = self.app_state.get_project();
        let Some(definition) = project.components.iter().find(|d| d.key == key) else {
            return;
        };
        let rows: Vec<ComponentParamRow> = definition
            .parameters
            .iter()
            .map(|parameter| ComponentParamRow {
                key: parameter.key.to_string().into(),
                label: parameter.label.clone().into(),
                section: parameter.section.clone().into(),
                description: parameter.description.clone().into(),
                default_value: param_value_display(&parameter.default).into(),
                default_description: parameter
                    .bindings
                    .first()
                    .and_then(|binding| {
                        let step = definition.steps.iter().find(|s| s.id == binding.step)?;
                        let (setting, _) = evanalyzer_app::components::step_setting(
                            &project.components,
                            step,
                            &binding.param,
                        )?;
                        // One line, shortened in the dialog.
                        Some(
                            setting
                                .description
                                .split_whitespace()
                                .collect::<Vec<_>>()
                                .join(" "),
                        )
                    })
                    .unwrap_or_default()
                    .into(),
                sets: parameter
                    .bindings
                    .iter()
                    .filter_map(|binding| {
                        let step = definition.steps.iter().find(|s| s.id == binding.step)?;
                        let (_, label) = evanalyzer_app::components::step_setting(
                            &project.components,
                            step,
                            &binding.param,
                        )?;
                        Some(format!("{} › {label}", step_ref(&project, step)))
                    })
                    .collect::<Vec<_>>()
                    .join(", ")
                    .into(),
                bindings: ModelRc::new(VecModel::from(
                    parameter
                        .bindings
                        .iter()
                        .filter_map(|binding| {
                            let step = definition.steps.iter().find(|s| s.id == binding.step)?;
                            let (_, label) = evanalyzer_app::components::step_setting(
                                &project.components,
                                step,
                                &binding.param,
                            )?;
                            Some(ComponentBindingRow {
                                text: format!("{} › {label}", step_ref(&project, step)).into(),
                                step_id: binding.step.0 as i32,
                                path: binding.param.clone().into(),
                            })
                        })
                        .collect::<Vec<_>>(),
                )),
                addable: {
                    let addable = evanalyzer_app::components::bindable_settings(
                        &project.components,
                        &key,
                        &parameter.key,
                    );
                    let labels: Vec<SharedString> = addable
                        .iter()
                        .filter_map(|(step_id, _, label)| {
                            let step = definition.steps.iter().find(|s| s.id == *step_id)?;
                            Some(format!("{} › {label}", step_ref(&project, step)).into())
                        })
                        .collect();
                    // "Also set…" first, as the choice's prompt.
                    let all: Vec<SharedString> = if labels.is_empty() {
                        Vec::new()
                    } else {
                        std::iter::once("Also set…".into()).chain(labels).collect()
                    };
                    ModelRc::new(VecModel::from(all))
                },
            })
            .collect();
        let state = ui.global::<ComponentSettingsState>();
        state.set_focus_index(focus_index);
        state.set_parameters(ModelRc::new(VecModel::from(rows)));
    }

    /// The row of the opened component's parameter `key` in the settings
    /// dialog, or -1 - to keep it open after a change refills the rows.
    fn editing_parameter(&self, key: &str) -> i32 {
        let Some(ui) = self.ui.upgrade() else {
            return -1;
        };
        let pipeline_id = ui.global::<PipelinesPanelState>().get_active_pipeline_id() as u32;
        let Some(open) = self.open_component_key(pipeline_id) else {
            return -1;
        };
        self.app_state
            .get_project()
            .components
            .iter()
            .find(|d| d.key == open)
            .and_then(|d| d.parameters.iter().position(|p| p.key.to_string() == key))
            .map_or(-1, |i| i as i32)
    }

    /// Applies `edit` to the opened component's definition - not while it's only
    /// viewed. Returns whether it changed something (the project is then
    /// marked as changed).
    fn edit_open_component_definition(
        &self,
        edit: impl FnOnce(&mut ComponentDefinition) -> bool,
    ) -> bool {
        let Some(ui) = self.ui.upgrade() else {
            return false;
        };
        let pipeline_id = ui.global::<PipelinesPanelState>().get_active_pipeline_id() as u32;
        if self.viewing(pipeline_id) {
            return false;
        }
        let Some(key) = self.open_component_key(pipeline_id) else {
            return false;
        };
        let edited = {
            let mut project = self.app_state.get_project_write();
            project
                .components
                .iter_mut()
                .find(|d| d.key == key)
                .is_some_and(edit)
        };
        if edited {
            self.app_state.mark_dirty();
        }
        edited
    }

    /// The steps the panel shows for pipeline `pipeline_id`: an opened
    /// component's, else the pipeline's.
    fn shown_steps<'a>(
        &self,
        project: &'a ProjectSettings,
        pipeline_id: u32,
    ) -> Option<&'a Vec<PipelineStepSettings>> {
        match self.open_component_key(pipeline_id) {
            Some(key) => project
                .components
                .iter()
                .find(|d| d.key == key)
                .map(|d| &d.steps),
            None => project
                .pipelines
                .iter()
                .find(|p| p.id.0 == pipeline_id)
                .map(|p| &p.steps),
        }
    }

    /// Checks the project again and patches what the checks found into the
    /// panel in place - its problem list, the opened component's, each
    /// card's and its settings' - without rebuilding the cards, so a field
    /// being edited keeps its focus. Edits that only patch their card (a
    /// changed setting) would leave the old problems up otherwise.
    fn refresh_problems(self: &Arc<Self>) {
        let Some(ui) = self.ui.upgrade() else {
            return;
        };
        let state = ui.global::<PipelinesPanelState>();
        let pipeline_id = PipelineId(state.get_active_pipeline_id() as u32);
        let debug_mode = self.debug_mode.load(std::sync::atomic::Ordering::Relaxed);
        let open_component = self.open_component_key(pipeline_id.0);
        let viewed_from = self.viewed_from(pipeline_id);
        let (data, open_problem, marks) = {
            let project = self.app_state.get_project();
            let problems = evanalyzer_app::components::check_project(&project);
            let data = problem_data(&project, &problems, debug_mode);
            let open_definition = open_component
                .as_ref()
                .and_then(|key| project.components.iter().find(|d| &d.key == key));
            let open_problem = open_definition
                .map(|d| {
                    problems
                        .iter()
                        .filter(|p| p.step.is_none() && p.component.as_deref() == Some(&d.key))
                        .map(|p| p.message.clone())
                        .collect::<Vec<_>>()
                        .join("\n⚠ ")
                })
                .unwrap_or_default();
            let shown: Vec<PipelineStepSettings> = self
                .viewed_steps(&project, pipeline_id.0)
                .or_else(|| open_definition.map(evanalyzer_app::components::apply_defaults))
                .or_else(|| {
                    project
                        .pipelines
                        .iter()
                        .find(|p| p.id == pipeline_id)
                        .map(|p| p.steps.clone())
                })
                .unwrap_or_default();
            let context = Marks {
                project: &project,
                problems: &problems,
                data: &data,
                pipeline_id,
                in_pipeline: open_component.is_none(),
                open_definition,
                viewed_from,
                debug_mode,
            };
            let marks: Vec<(
                SharedString,
                Vec<ProblemData>,
                std::collections::HashMap<String, ParamNote>,
            )> = shown
                .iter()
                .map(|step| {
                    let (problems, notes) = step_marks(&context, step);
                    (step_id_text(step.id), problems, notes)
                })
                .collect();
            *self.last_problems.lock().expect("Poisoned") = problems;
            (data, open_problem, marks)
        };

        state.set_open_component_problem(open_problem.into());
        state.set_project_problem_errors(data.iter().filter(|p| p.error).count() as i32);
        state.set_project_problems(ModelRc::new(VecModel::from(
            data.into_iter().map(problem_item).collect::<Vec<_>>(),
        )));
        let cards = state.get_active_commands();
        for (i, (step_id, problems, notes)) in marks.into_iter().enumerate() {
            let Some(mut card) = cards.row_data(i) else {
                continue;
            };
            if card.step_id != step_id {
                continue;
            }
            let mut changed = false;
            // The settings, and the copies a component card's sections and
            // tables show (see `card_layout`).
            let mut lists = vec![card.parameters.clone(), card.advanced_parameters.clone()];
            lists.extend(card.section_views.iter().map(|v| v.params));
            for table in card.card_tables.iter() {
                lists.extend(table.rows.iter().map(|r| r.cells));
            }
            for list in lists {
                for k in 0..list.row_count() {
                    let Some(mut parameter) = list.row_data(k) else {
                        continue;
                    };
                    let note = notes
                        .get(parameter.name.as_str())
                        .cloned()
                        .unwrap_or_default();
                    if parameter.wiring != note.wiring.as_str()
                        || parameter.problem != note.problem.as_str()
                        || parameter.problem_error != note.error
                    {
                        parameter.wiring = note.wiring.into();
                        parameter.problem = note.problem.into();
                        parameter.problem_error = note.error;
                        list.set_row_data(k, parameter);
                    }
                }
            }
            refresh_section_problems(&card);
            let shown: Vec<(SharedString, bool, SharedString)> = card
                .problems
                .iter()
                .map(|p| (p.text, p.error, p.fix_label))
                .collect();
            let now: Vec<(SharedString, bool, SharedString)> = problems
                .iter()
                .map(|p| (p.text.as_str().into(), p.error, p.fix_label.as_str().into()))
                .collect();
            if shown != now || card.problems.row_count() != problems.len() {
                card.problem_errors = problems.iter().filter(|p| p.error).count() as i32;
                card.problems = ModelRc::new(VecModel::from(
                    problems.into_iter().map(problem_item).collect::<Vec<_>>(),
                ));
                changed = true;
            } else {
                // Same problems, maybe at other places in the list.
                for (k, p) in problems.into_iter().enumerate() {
                    if let Some(mut item) = card.problems.row_data(k)
                        && item.index != p.index
                    {
                        item.index = p.index;
                        card.problems.set_row_data(k, item);
                    }
                }
            }
            if changed {
                cards.set_row_data(i, card);
            }
        }
    }

    /// Shows problem `index` of the panel's list (see `last_problems`): its
    /// pipeline, the card of its step expanded. A component's own problem:
    /// the first pipeline step using the component, opened to edit.
    fn show_problem(self: &Arc<Self>, index: usize) {
        let Some(problem) = self
            .last_problems
            .lock()
            .expect("Poisoned")
            .get(index)
            .cloned()
        else {
            return;
        };
        let Some(ui) = self.ui.upgrade() else {
            return;
        };
        let target: Option<(PipelineId, Option<StepId>, bool)> =
            match (problem.step, &problem.component) {
                (Some((pipeline, step)), _) => Some((pipeline, Some(step), false)),
                (None, Some(key)) => self.app_state.get_project().pipelines.iter().find_map(|p| {
                    p.steps
                        .iter()
                        .find(|s| matches!(&s.content, StepContent::Component(i) if &i.key == key))
                        .map(|s| (p.id, Some(s.id), true))
                }),
                (None, None) => None,
            };
        let Some((pipeline_id, step_id, open)) = target else {
            return;
        };
        self.close_open_component();
        ui.global::<PipelinesPanelState>()
            .set_active_pipeline_id(pipeline_id.0 as i32);
        self.sync_steps_of_selected_pipeline_to_slint(pipeline_id, true);
        let Some(step_id) = step_id.filter(|s| *s != StepId::UNASSIGNED) else {
            return;
        };
        if open {
            let index = self
                .app_state
                .get_project()
                .pipelines
                .iter()
                .find(|p| p.id == pipeline_id)
                .and_then(|p| p.steps.iter().position(|s| s.id == step_id));
            if let Some(index) = index {
                self.open_component_in_panel(pipeline_id.0, index, false);
            }
            return;
        }
        // After the sync above (the event loop runs them in order).
        let ui_weak = self.ui.clone();
        let wanted = step_id_text(step_id);
        if let Err(e) = crate::helper::ui_thread::invoke_from_event_loop(move || {
            let Some(ui) = ui_weak.upgrade() else {
                return;
            };
            let commands = ui.global::<PipelinesPanelState>().get_active_commands();
            for i in 0..commands.row_count() {
                if let Some(mut command) = commands.row_data(i)
                    && command.step_id == wanted
                {
                    command.expanded = true;
                    commands.set_row_data(i, command);
                }
            }
        }) {
            warn!("Failed to show a problem's step: {e}");
        }
    }

    /// Makes the fix of problem `index` of the panel's list (see `last_problems`).
    fn fix_problem(self: &Arc<Self>, index: usize) {
        let Some(fix) = self
            .last_problems
            .lock()
            .expect("Poisoned")
            .get(index)
            .and_then(|p| p.fix.clone())
        else {
            return;
        };
        let fixed =
            evanalyzer_app::components::apply_fix(&mut self.app_state.get_project_write(), &fix);
        if !fixed {
            return;
        }
        self.pipeline_settings_changed();
        if matches!(fix, evanalyzer_app::components::ProblemFix::AddClass(_)) {
            self.focus_controller.classes_changed();
        }
        if let Some(ui) = self.ui.upgrade() {
            let pipeline_id = ui.global::<PipelinesPanelState>().get_active_pipeline_id() as u32;
            self.sync_steps_of_selected_pipeline_to_slint(PipelineId(pipeline_id), false);
        }
    }

    /// The pipeline step the component shown in pipeline `pipeline_id`'s
    /// panel is only viewed from - when it's the first level opened.
    fn viewed_from(&self, pipeline_id: PipelineId) -> Option<(PipelineId, StepId)> {
        match self.open_component_path(pipeline_id.0).as_slice() {
            [level] => level.viewing_from.and_then(|idx| {
                self.app_state
                    .get_project()
                    .pipelines
                    .iter()
                    .find(|p| p.id == pipeline_id)
                    .and_then(|p| p.steps.get(idx))
                    .map(|step| (pipeline_id, step.id))
            }),
            _ => None,
        }
    }

    /// The steps of the component shown in pipeline `pipeline_id`'s panel
    /// with the values of the step it's viewed from - when it and each
    /// component around it are only viewed, from a pipeline step on. `None`
    /// otherwise: then it shows its defaults.
    fn viewed_steps(
        &self,
        project: &ProjectSettings,
        pipeline_id: u32,
    ) -> Option<Vec<PipelineStepSettings>> {
        let path = self.open_component_path(pipeline_id);
        if path.is_empty() {
            return None;
        }
        let mut steps = project
            .pipelines
            .iter()
            .find(|p| p.id.0 == pipeline_id)?
            .steps
            .clone();
        for level in &path {
            let StepContent::Component(instance) = &steps.get(level.viewing_from?)?.content else {
                return None;
            };
            if instance.key != level.key {
                return None;
            }
            steps = evanalyzer_app::components::component_steps_with_values(
                instance,
                &project.components,
            )
            .ok()?;
        }
        Some(steps)
    }

    /// [`Self::shown_steps`], to edit - `None` while a component is only viewed.
    fn shown_steps_mut<'a>(
        &self,
        project: &'a mut ProjectSettings,
        pipeline_id: u32,
    ) -> Option<&'a mut Vec<PipelineStepSettings>> {
        if self.viewing(pipeline_id) {
            return None;
        }
        match self.open_component_key(pipeline_id) {
            Some(key) => project
                .components
                .iter_mut()
                .find(|d| d.key == key)
                .map(|d| &mut d.steps),
            None => project
                .pipelines
                .iter_mut()
                .find(|p| p.id.0 == pipeline_id)
                .map(|p| &mut p.steps),
        }
    }

    /// Opens component step `step_idx` of the steps pipeline `pipeline_id`'s
    /// panel shows (the pipeline's or an opened component's) in the panel - to
    /// edit, or with `read_only` only to look at. Inside a component opened
    /// only to look at, the nested one is only looked at too.
    fn open_component_in_panel(
        self: &Arc<Self>,
        pipeline_id: u32,
        step_idx: usize,
        read_only: bool,
    ) {
        let key = {
            let project = self.app_state.get_project();
            let Some(StepContent::Component(instance)) = self
                .shown_steps(&project, pipeline_id)
                .and_then(|steps| steps.get(step_idx))
                .map(|s| &s.content)
            else {
                return;
            };
            instance.key.clone()
        };
        let read_only = read_only || self.viewing(pipeline_id);
        let mut path = self.open_component_path(pipeline_id);
        path.push(OpenComponent {
            key,
            viewing_from: read_only.then_some(step_idx),
        });
        *self.open_component.lock().expect("Poisoned") = Some((pipeline_id, path));
        self.sync_steps_of_selected_pipeline_to_slint(PipelineId(pipeline_id), true);
    }

    fn show_classifier_model_info(self: &Arc<Self>, step_idx: usize) {
        let Some(ui) = self.ui.upgrade() else {
            return;
        };
        let pipeline_id = ui.global::<PipelinesPanelState>().get_active_pipeline_id() as u32;

        let model_path = {
            let project = self.app_state.get_project();
            let Some(step) = self
                .shown_steps(&project, pipeline_id)
                .and_then(|steps| steps.get(step_idx))
            else {
                return;
            };
            match step.command() {
                Some(PipelineCommand::PixelClassifier(settings)) => settings.model_path.clone(),
                Some(PipelineCommand::AiObjectClassifier(settings)) => settings.model_path.clone(),
                _ => return,
            }
        };

        let (title, message) =
            match load_classifier_settings(self.app_state.backend().files(), &model_path) {
                Ok(settings) => (
                    "AI Classifier Model".to_string(),
                    format_classifier_model_info(&settings),
                ),
                Err(e) => (
                    "Could not load model".to_string(),
                    format!("Could not load '{}':\n\n{e}", model_path.display()),
                ),
            };

        let warning = ui.global::<WarningState>();
        warning.set_info(true);
        warning.set_title(title.into());
        warning.set_message(message.into());
        ui.global::<GlobalAppState>()
            .set_active_dialog(DialogType::Warning);
    }

    fn modify_group_item(
        self: &Arc<Self>,
        pipeline_id: u32,
        step_idx: usize,
        param_name: &str,
        add: bool,
        remove_idx: Option<usize>,
    ) {
        let param_name = param_name.to_owned();
        {
            let mut project = self.app_state.get_project_write();
            let mut removed_from: Option<evanalyzer_cfg::core_types::StepId> = None;
            if let Some(steps) = self.shown_steps_mut(&mut project, pipeline_id) {
                if let Some(step) = steps.get_mut(step_idx) {
                    let step_id = step.id;
                    if let Some(command) = step.command_mut() {
                        if add {
                            command.add_group_item(&param_name);
                        } else if let Some(idx) = remove_idx {
                            command.remove_group_item(&param_name, idx);
                            removed_from = Some(step_id);
                        }
                    }
                }
            }
            // Inside an opened component: bindings to later entries move up.
            if let (Some(step_id), Some(idx), Some(key)) = (
                removed_from,
                remove_idx,
                self.open_component_key(pipeline_id),
            ) && let Some(definition) = project.components.iter_mut().find(|d| d.key == key)
            {
                evanalyzer_app::components::list_entry_removed(
                    definition,
                    step_id,
                    &param_name,
                    idx,
                );
            }
        }
        self.sync_steps_of_selected_pipeline_to_slint(PipelineId(pipeline_id), false);
    }

    /// Synchronizes the pipeline list from project settings into the Slint UI.
    ///
    /// Reads `project.pipelines`, maps each `PipelineSettings` to a Slint `Pipeline`
    /// struct, and pushes the result to `PipelinesPanelState` via the event loop.
    /// Also updates `enabled_pipeline_count`.
    ///
    /// The project lock is released before `invoke_from_event_loop` is called, so
    /// this method is safe to call from any thread.
    ///
    /// Logs a `warn!` if the Slint event loop is unreachable.
    pub fn sync_pipelines_to_slint(self: &Arc<Self>) {
        // Also runs when a project is opened or created: the focus banner
        // must follow the (focus-less) new project.
        self.focus_controller.refresh();
        let ui_weak = self.ui.clone();

        let slint_pipelines: Vec<Pipeline> = {
            let project = self.app_state.get_project();
            project
                .pipelines
                .iter()
                .map(|p| {
                    let total = p.steps.len() as i32;
                    let enabled_steps = p.steps.iter().filter(|s| s.enabled).count() as i32;
                    Pipeline {
                        id: p.id.0 as i32,
                        name: p.name.clone().into(),
                        image_source: match p.image_source {
                            ImageAddress::Scratchpad => "Scratchpad".into(),
                            ImageAddress::Memory(
                                MemoryId::PipelineContext(s) | MemoryId::ProjectCache(s),
                            ) => format!("Shared[{s}]").into(),
                            ImageAddress::Memory(MemoryId::ComponentContext(s)) => {
                                format!("Component[{s}]").into()
                            }
                            ImageAddress::None => "None".into(),
                            ImageAddress::Channel(c) => format!("Channel {c}").into(),
                        },
                        enabled: p.enabled,
                        dirty: false,
                        status: PipelineStatus::Idle,
                        total_step_count: total,
                        enabled_step_count: enabled_steps,
                    }
                })
                .collect()
        };

        let enabled_count = slint_pipelines.iter().filter(|p| p.enabled).count() as i32;

        if let Err(e) = crate::helper::ui_thread::invoke_from_event_loop(move || {
            if let Some(ui) = ui_weak.upgrade() {
                let model = ModelRc::new(VecModel::from(slint_pipelines));
                let state = ui.global::<PipelinesPanelState>();
                state.set_pipelines(model);
                state.set_enabled_pipeline_count(enabled_count);
            }
        }) {
            warn!("Failed to sync pipelines to Slint: {}", e);
        }
    }

    /// After components were removed: the picker's list and detail anew (ids
    /// moved), the project changed.
    fn after_components_removed(self: &Arc<Self>, ui: &AppWindow) {
        let picker = ui.global::<CommandPickerState>();
        picker.set_has_detail(false);
        picker.set_detail_removable(false);
        picker.set_selected_id(-1);
        let query = picker.get_query().to_string();
        self.apply_picker_filter(ui, &query);
        self.pipeline_settings_changed();
    }

    /// Fills "Create component" for steps `from..=to` of its pipeline: the range,
    /// the classes with their suggested roles, the channels.
    fn refresh_component_plan(self: &Arc<Self>, ui: &AppWindow, from: i32, to: i32) {
        let state = ui.global::<CreateComponentState>();
        state.set_from_step(from);
        state.set_to_step(to);
        state.set_error("".into());
        let pipeline_id = PipelineId(state.get_pipeline_id() as u32);
        let plan = {
            let project = self.app_state.get_project();
            evanalyzer_app::components::plan_component(
                &project,
                self.shown_list(pipeline_id.0),
                from.max(0) as usize..=to.max(0) as usize,
            )
        };
        match plan {
            Ok(plan) => {
                let rows: Vec<ComponentClassRow> = plan
                    .classes
                    .iter()
                    .map(|c| ComponentClassRow {
                        class_text: c.class.to_text().into(),
                        name: c.name.clone().into(),
                        detail: if c.used_elsewhere {
                            "Also used elsewhere".into()
                        } else {
                            "Only used by these steps".into()
                        },
                        role: match c.suggested {
                            ClassRole::Local => 0,
                            ClassRole::Parameter => 1,
                            ClassRole::Fixed => 2,
                        },
                    })
                    .collect();
                state.set_classes(ModelRc::new(VecModel::from(rows)));
                let channels: Vec<String> = plan.channels.iter().map(|c| c.to_string()).collect();
                let slots: Vec<String> =
                    plan.component_slots.iter().map(|s| s.to_string()).collect();
                let handed: Vec<String> = plan.handed_slots.iter().map(|s| s.to_string()).collect();
                let lines: Vec<String> = [
                    match channels.len() {
                        0 => String::new(),
                        1 => format!("Channel {} becomes a component parameter.", channels[0]),
                        _ => format!("Channels {} become component parameters.", channels.join(", ")),
                    },
                    match slots.len() {
                        0 => String::new(),
                        1 => format!(
                            "Memory slot {} is only used by these steps: it becomes component memory, each use of the component has its own.",
                            slots[0]
                        ),
                        _ => format!(
                            "Memory slots {} are only used by these steps: they become component memory, each use of the component has its own.",
                            slots.join(", ")
                        ),
                    },
                    match handed.len() {
                        0 => String::new(),
                        1 => format!(
                            "Component memory slot {} is also used by other steps: the new component gets it as a parameter.",
                            handed[0]
                        ),
                        _ => format!(
                            "Component memory slots {} are also used by other steps: the new component gets them as parameters.",
                            handed.join(", ")
                        ),
                    },
                ]
                .into_iter()
                .filter(|line| !line.is_empty())
                .collect();
                state.set_channels_text(lines.join("\n").into());
            }
            Err(e) => {
                state.set_classes(ModelRc::default());
                state.set_channels_text("".into());
                state.set_error(e.to_string().into());
            }
        }
    }

    fn apply_picker_filter(self: &Arc<Self>, ui: &AppWindow, query: &str) {
        let q = query.to_ascii_lowercase();
        let metas = all_command_meta();

        let text_matches = |m: &&evanalyzer_cfg::settings::pipeline_command::CommandMeta| {
            q.is_empty()
                || m.name.to_ascii_lowercase().contains(&q)
                || m.summary.to_ascii_lowercase().contains(&q)
        };
        // Per-category chip state (multi-select). When no chip is active the
        // picker shows every category ("All").
        let picker_state = ui.global::<CommandPickerState>();
        let (fc_pre, fc_seg, fc_obj, fc_mea, fc_cls, fc_dat) = (
            picker_state.get_fcat_pre(),
            picker_state.get_fcat_seg(),
            picker_state.get_fcat_obj(),
            picker_state.get_fcat_mea(),
            picker_state.get_fcat_cls(),
            picker_state.get_fcat_dat(),
        );
        let any_active = fc_pre || fc_seg || fc_obj || fc_mea || fc_cls || fc_dat;
        let cat_enabled = move |target: CommandCategory| -> bool {
            if !any_active {
                return true;
            }
            match target {
                CommandCategory::Preprocess => fc_pre,
                CommandCategory::Segment => fc_seg,
                CommandCategory::InstanceSegmentation => fc_obj,
                CommandCategory::Measure => fc_mea,
                CommandCategory::Object => fc_cls,
                CommandCategory::Dataset => fc_dat,
            }
        };
        let make = |m: &evanalyzer_cfg::settings::pipeline_command::CommandMeta,
                    _cat: StepCategory|
         -> CommandDef { to_command_def(m) };
        let pre: Vec<CommandDef> = metas
            .iter()
            .filter(|m| {
                matches!(m.category, CommandCategory::Preprocess)
                    && cat_enabled(m.category)
                    && text_matches(m)
            })
            .map(|m| make(m, StepCategory::Preprocess))
            .collect();
        let seg: Vec<CommandDef> = metas
            .iter()
            .filter(|m| {
                matches!(m.category, CommandCategory::Segment)
                    && cat_enabled(m.category)
                    && text_matches(m)
            })
            .map(|m| make(m, StepCategory::Segment))
            .collect();
        let obj: Vec<CommandDef> = metas
            .iter()
            .filter(|m| {
                matches!(m.category, CommandCategory::InstanceSegmentation)
                    && cat_enabled(m.category)
                    && text_matches(m)
            })
            .map(|m| make(m, StepCategory::InstanceSegmentation))
            .collect();
        let mea: Vec<CommandDef> = metas
            .iter()
            .filter(|m| {
                matches!(m.category, CommandCategory::Measure)
                    && cat_enabled(m.category)
                    && text_matches(m)
            })
            .map(|m| make(m, StepCategory::Measure))
            .collect();
        let cls: Vec<CommandDef> = metas
            .iter()
            .filter(|m| {
                matches!(m.category, CommandCategory::Object)
                    && cat_enabled(m.category)
                    && text_matches(m)
            })
            .map(|m| make(m, StepCategory::Object))
            .collect();
        let dat: Vec<CommandDef> = metas
            .iter()
            .filter(|m| {
                matches!(m.category, CommandCategory::Dataset)
                    && cat_enabled(m.category)
                    && text_matches(m)
            })
            .map(|m| make(m, StepCategory::Dataset))
            .collect();
        // Components: all of the project's with the Components chip, else only when
        // searching (they don't crowd the category lists).
        let filter_components = picker_state.get_filter_components();
        // Inside an opened component: not the ones that would contain it.
        let open_key = self.open_component_key(picker_state.get_pipeline_id() as u32);
        let matching_components: Vec<CommandDef> = {
            let project = self.app_state.get_project();
            project
                .components
                .iter()
                .enumerate()
                .filter(|(_, d)| {
                    open_key.as_ref().is_none_or(|open| {
                        !evanalyzer_app::components::would_contain_itself(&project, open, &d.key)
                    })
                })
                .filter(|(_, d)| {
                    q.is_empty()
                        || d.meta.name.to_ascii_lowercase().contains(&q)
                        || d.meta.short_description.to_ascii_lowercase().contains(&q)
                        || d.key.to_ascii_lowercase().contains(&q)
                })
                .map(|(idx, d)| component_to_command_def(idx, d, &project))
                .collect()
        };
        let library_components: Vec<CommandDef> = {
            let project = self.app_state.get_project();
            let library = self.component_library.lock().expect("Poisoned");
            library
                .iter()
                .enumerate()
                .filter(|(_, (path, template))| {
                    evanalyzer_app::components::project_copy(
                        &project,
                        template,
                        &evanalyzer_app::components::library_name(path),
                    )
                    .is_none()
                })
                .filter(|(_, (_, t))| {
                    q.is_empty()
                        || t.component.meta.name.to_ascii_lowercase().contains(&q)
                        || t.component
                            .meta
                            .short_description
                            .to_ascii_lowercase()
                            .contains(&q)
                })
                .map(|(idx, (path, template))| {
                    library_component_to_command_def(idx, path, template)
                })
                .collect()
        };
        let cg = (matching_components.len() + library_components.len()) as i32;
        let (components, library_components) = if filter_components || !q.is_empty() {
            (matching_components, library_components)
        } else {
            (Vec::new(), Vec::new())
        };

        let total = if filter_components {
            cg
        } else {
            (pre.len()
                + seg.len()
                + obj.len()
                + mea.len()
                + cls.len()
                + dat.len()
                + components.len()
                + library_components.len()) as i32
        };
        let (cp, cs, co, cm, cc, cd) = (
            pre.len() as i32,
            seg.len() as i32,
            obj.len() as i32,
            mea.len() as i32,
            cls.len() as i32,
            dat.len() as i32,
        );
        let picker = ui.global::<CommandPickerState>();
        picker.set_shown_preprocess(ModelRc::new(VecModel::from(pre)));
        picker.set_shown_segment(ModelRc::new(VecModel::from(seg)));
        picker.set_shown_object(ModelRc::new(VecModel::from(obj)));
        picker.set_shown_measure(ModelRc::new(VecModel::from(mea)));
        picker.set_shown_classify(ModelRc::new(VecModel::from(cls)));
        picker.set_shown_dataset(ModelRc::new(VecModel::from(dat)));
        picker.set_shown_components(ModelRc::new(VecModel::from(components)));
        picker.set_unused_component_count(
            evanalyzer_app::components::unused_components(&self.app_state.get_project()).len()
                as i32,
        );
        picker.set_shown_library_components(ModelRc::new(VecModel::from(library_components)));
        picker.set_cat_count_grp(cg);
        picker.set_total_shown(total);
        picker.set_cat_count_pre(cp);
        picker.set_cat_count_seg(cs);
        picker.set_cat_count_obj(co);
        picker.set_cat_count_mea(cm);
        picker.set_cat_count_cls(cc);
        picker.set_cat_count_dat(cd);
    }

    /// Reloads the component library from the user and app templates folders in the
    /// background, then refreshes the command picker if it is still open.
    ///
    /// Scanning the templates folders involves blocking filesystem I/O which can
    /// be slow (e.g. on Windows with antivirus scanning or network drives), so it
    /// must not run on the Slint event-loop thread - doing so would freeze the UI
    /// while the "Add Step" dialog opens. The dialog opens immediately with the
    /// previously cached library and is updated in place once the reload
    /// completes.
    fn reload_component_library_async(self: &Arc<Self>) {
        let manager = self.clone();
        crate::helper::ui_thread::spawn(move || {
            *manager.component_library.lock().expect("Poisoned") =
                load_component_templates(manager.app_state.backend().as_ref());

            let manager = manager.clone();
            if let Err(e) = crate::helper::ui_thread::invoke_from_event_loop(move || {
                let Some(ui) = manager.ui.upgrade() else {
                    return;
                };
                if ui.global::<GlobalAppState>().get_active_dialog()
                    != DialogType::CommandSelectionDialog
                {
                    return;
                }
                let picker = ui.global::<CommandPickerState>();
                let query = picker.get_query().to_string();
                manager.apply_picker_filter(&ui, &query);
            }) {
                warn!("Failed to refresh the component library in picker: {}", e);
            }
        });
    }

    fn sync_commands_to_selection_dialog_slint(self: &Arc<Self>) {
        let raw = all_command_meta();

        let ui_weak = self.ui.clone();
        if let Err(e) = crate::helper::ui_thread::invoke_from_event_loop(move || {
            if let Some(ui) = ui_weak.upgrade() {
                let all: Vec<CommandDef> = raw.iter().map(to_command_def).collect();
                let shown_pre: Vec<CommandDef> = raw
                    .iter()
                    .filter(|m| matches!(m.category, CommandCategory::Preprocess))
                    .map(to_command_def)
                    .collect();
                let shown_seg: Vec<CommandDef> = raw
                    .iter()
                    .filter(|m| matches!(m.category, CommandCategory::Segment))
                    .map(to_command_def)
                    .collect();
                let shown_obj: Vec<CommandDef> = raw
                    .iter()
                    .filter(|m| matches!(m.category, CommandCategory::InstanceSegmentation))
                    .map(to_command_def)
                    .collect();
                let shown_mea: Vec<CommandDef> = raw
                    .iter()
                    .filter(|m| matches!(m.category, CommandCategory::Measure))
                    .map(to_command_def)
                    .collect();
                let shown_cls: Vec<CommandDef> = raw
                    .iter()
                    .filter(|m| matches!(m.category, CommandCategory::Object))
                    .map(to_command_def)
                    .collect();
                let shown_dat: Vec<CommandDef> = raw
                    .iter()
                    .filter(|m| matches!(m.category, CommandCategory::Dataset))
                    .map(to_command_def)
                    .collect();
                let total = all.len() as i32;
                let (cp, cs, co, cm, cc, cd) = (
                    shown_pre.len() as i32,
                    shown_seg.len() as i32,
                    shown_obj.len() as i32,
                    shown_mea.len() as i32,
                    shown_cls.len() as i32,
                    shown_dat.len() as i32,
                );
                let picker = ui.global::<CommandPickerState>();
                picker.set_all_commands(ModelRc::new(VecModel::from(all)));
                picker.set_shown_preprocess(ModelRc::new(VecModel::from(shown_pre)));
                picker.set_shown_segment(ModelRc::new(VecModel::from(shown_seg)));
                picker.set_shown_object(ModelRc::new(VecModel::from(shown_obj)));
                picker.set_shown_measure(ModelRc::new(VecModel::from(shown_mea)));
                picker.set_shown_classify(ModelRc::new(VecModel::from(shown_cls)));
                picker.set_shown_dataset(ModelRc::new(VecModel::from(shown_dat)));
                picker.set_total_shown(total);
                picker.set_cat_count_pre(cp);
                picker.set_cat_count_seg(cs);
                picker.set_cat_count_obj(co);
                picker.set_cat_count_mea(cm);
                picker.set_cat_count_cls(cc);
                picker.set_cat_count_dat(cd);
            }
        }) {
            warn!("Failed to sync commands to picker: {}", e);
        }
    }

    /// Synchronizes the steps of the selected pipeline into the Slint UI.
    ///
    /// Finds the pipeline by `pipeline_id`, maps each `PipelineStepSettings` to a
    /// Slint `PipelineCommand` struct, and pushes the result to `PipelinesPanelState`
    /// via the event loop. Also updates `active_pipeline_name`.
    ///
    /// The project lock is released before `invoke_from_event_loop` is called.
    pub fn sync_steps_of_selected_pipeline_to_slint(
        self: &Arc<Self>,
        pipeline_id: PipelineId,
        reset_expanded: bool,
    ) {
        let ui_weak = self.ui.clone();

        // Collect only Send-safe data before crossing the thread boundary.
        // ModelRc (Rc-backed) must be created inside invoke_from_event_loop.
        struct StepData {
            idx: i32,
            id: evanalyzer_cfg::core_types::StepId,
            name: String,
            summary: String,
            category: StepCategory,
            enabled: bool,
            parameters: Vec<ParameterDef>,
            /// Settings that are parameters of the opened component.
            exposed: Vec<String>,
            /// A component step: the library's newer version, or empty.
            update_version: String,
            /// What the command or component does (the card's "?").
            help: String,
            /// What the checks found about the step.
            problems: Vec<ProblemData>,
            /// Per setting (its name): what it's wired to, what a check found.
            notes: std::collections::HashMap<String, ParamNote>,
        }
        // The opened component's name, or empty. A component that's gone (e.g. undone)
        // is closed - and the ones opened inside it.
        let mut path = self.open_component_path(pipeline_id.0);
        let still_there = {
            let project = self.app_state.get_project();
            path.iter()
                .take_while(|level| project.components.iter().any(|d| d.key == level.key))
                .count()
        };
        if still_there < path.len() {
            path.truncate(still_there);
            self.set_open_path(pipeline_id.0, path.clone());
        }
        let open_component = path.last().map(|level| level.key.clone());
        // Each problem of the opened component, on a line of its own.
        let open_component_problem: String = open_component
            .as_ref()
            .map(|key| {
                evanalyzer_app::components::check_component(&self.app_state.get_project(), key)
                    .into_iter()
                    .map(|problem| problem.message)
                    .collect::<Vec<_>>()
                    .join("\n⚠ ")
            })
            .unwrap_or_default();
        // What the checks find in the whole project: the panel's list, and
        // the marks on the cards and settings.
        let problems = evanalyzer_app::components::check_project(&self.app_state.get_project());
        let debug_mode = self.debug_mode.load(std::sync::atomic::Ordering::Relaxed);
        let problem_data = problem_data(&self.app_state.get_project(), &problems, debug_mode);
        *self.last_problems.lock().expect("Poisoned") = problems.clone();
        // A component only looked at from a pipeline step (and the ones it
        // contains): shown with that step's values, not the defaults.
        let viewed_steps = self.viewed_steps(&self.app_state.get_project(), pipeline_id.0);
        // The pipeline step it's looked at from - whose problems mark its settings.
        let viewed_from = self.viewed_from(pipeline_id);
        let open_component_values: String = match (&viewed_steps, path.first()) {
            (Some(_), Some(level)) => level
                .viewing_from
                .map(|idx| format!("Shown with the values of step {} in this pipeline", idx + 1))
                .unwrap_or_default(),
            _ => String::new(),
        };
        let read_only = open_component.is_some() && self.viewing(pipeline_id.0);
        // "Outer › ◆ Inner" for a component opened inside another.
        let open_component_name: String = {
            let project = self.app_state.get_project();
            path.iter()
                .map(|level| {
                    project
                        .components
                        .iter()
                        .find(|d| d.key == level.key)
                        .map_or_else(|| level.key.clone(), |d| d.meta.name.clone())
                })
                .collect::<Vec<_>>()
                .join(" › ◆ ")
        };
        // The opened component's own classes - its class settings may take them.
        let component_classes: Vec<evanalyzer_cfg::settings::classification_settings::Class> =
            open_component
                .as_ref()
                .and_then(|key| {
                    self.app_state
                        .get_project()
                        .components
                        .iter()
                        .find(|d| &d.key == key)
                        .map(|d| d.classes.clone())
                })
                .unwrap_or_default();
        // What a collapsed section's values are called: the images' channel
        // names, the project's and the opened component's class names.
        let value_names = {
            let project = self.app_state.get_project();
            ValueNames {
                channels: evanalyzer_app::components::image_channels(&project)
                    .map(|c| c.names)
                    .unwrap_or_default(),
                classes: project
                    .classification
                    .classes()
                    .iter()
                    .map(|c| (c.id.to_string(), c.name.clone()))
                    .chain(
                        component_classes
                            .iter()
                            .map(|c| (ObjectClass::Local(c.id).to_text(), c.name.clone())),
                    )
                    .collect(),
            }
        };
        let open_sections = self.open_sections.lock().expect("Poisoned").clone();
        let component_class_items = component_classes.clone();
        // A class list's flags: the project's 33 (see `MultiClassDropdown`),
        // then one per class of the opened component.
        let with_component_flags = move |param_type: &CfgParamType,
                                         value: &str,
                                         mut options: Vec<String>|
              -> Vec<String> {
            if *param_type != CfgParamType::MultiObjClass || component_classes.is_empty() {
                return options;
            }
            options.resize(33, "0".into());
            let selected: Vec<&str> = value.split(',').map(str::trim).collect();
            options.extend(component_classes.iter().map(|class| {
                let text = ObjectClass::Local(class.id).to_text();
                if selected.contains(&text.as_str()) {
                    "1"
                } else {
                    "0"
                }
                .to_string()
            }));
            options
        };

        let (
            pipeline_name,
            pipeline_image_source,
            step_data,
            total_steps_count,
            enabled_steps_count,
            total_enabled_across_all,
        ): (String, String, Vec<StepData>, i32, i32, i32) = {
            let project = self.app_state.get_project();
            let total_enabled_across_all = project
                .pipelines
                .iter()
                .filter(|p| p.enabled)
                .flat_map(|p| p.steps.iter())
                .filter(|s| s.enabled)
                .count() as i32;
            if let Some(pipeline) = project.pipelines.iter().find(|p| p.id == pipeline_id) {
                let name = pipeline.name.clone();
                let image_src = match pipeline.image_source {
                    ImageAddress::Scratchpad => "Scratchpad".to_string(),
                    ImageAddress::Memory(
                        MemoryId::PipelineContext(s) | MemoryId::ProjectCache(s),
                    ) => {
                        format!("Shared[{s}]")
                    }
                    ImageAddress::Memory(MemoryId::ComponentContext(s)) => {
                        format!("Component[{s}]")
                    }
                    ImageAddress::None => "None".to_string(),
                    ImageAddress::Channel(c) => format!("Channel {c}"),
                };
                let open_definition = open_component
                    .as_ref()
                    .and_then(|key| project.components.iter().find(|d| &d.key == key));
                // An opened component's settings as a use without values of its
                // own runs them: a setting a parameter sets shows the
                // parameter's default.
                let with_defaults = viewed_steps
                    .clone()
                    .or_else(|| open_definition.map(evanalyzer_app::components::apply_defaults));
                let shown = match &with_defaults {
                    Some(steps) => steps,
                    None => self
                        .shown_steps(&project, pipeline_id.0)
                        .unwrap_or(&pipeline.steps),
                };
                let steps: Vec<StepData> = shown
                    .iter()
                    .enumerate()
                    .map(|(idx, step)| {
                        let (step_problems, notes) = step_marks(
                            &Marks {
                                project: &project,
                                problems: &problems,
                                data: &problem_data,
                                pipeline_id,
                                in_pipeline: open_component.is_none(),
                                open_definition,
                                viewed_from,
                                debug_mode,
                            },
                            step,
                        );
                        match &step.content {
                            StepContent::Command(command) => StepData {
                                idx: idx as i32,
                                id: step.id,
                                name: command.name().to_owned(),
                                // What a run would refuse (e.g. a component's class
                                // outside a component), else the settings' summary.
                                summary: match open_component {
                                    Some(_) => command.to_summary(),
                                    None => {
                                        evanalyzer_app::components::step_problem(&project, step)
                                            .map(|problem| format!("⚠ {problem}"))
                                            .unwrap_or_else(|| command.to_summary())
                                    }
                                },
                                category: step_category(*command.category()),
                                enabled: step.enabled,
                                parameters: evanalyzer_app::settings_of::command_parameters(
                                    command,
                                ),
                                exposed: open_definition
                                    .map(|d| {
                                        evanalyzer_app::components::exposed_settings(d, step.id)
                                    })
                                    .unwrap_or_default(),
                                update_version: String::new(),
                                help: command_help(command.name()),
                                problems: step_problems,
                                notes,
                            },
                            // The component's name and parameters - each shown with the
                            // control of the inner setting it's bound to.
                            StepContent::Component(component) => {
                                let (summary, parameters) =
                                    match evanalyzer_app::components::component_parameters(
                                        component,
                                        &project.components,
                                    ) {
                                        Ok(parameters) => (
                                            evanalyzer_app::components::step_problem(
                                                &project, step,
                                            )
                                            .map(|problem| format!("⚠ {problem}"))
                                            .unwrap_or_else(|| "Component".to_string()),
                                            parameters,
                                        ),
                                        Err(e) => (format!("⚠ {e}"), Vec::new()),
                                    };
                                StepData {
                                    idx: idx as i32,
                                    id: step.id,
                                    name: project
                                        .components
                                        .iter()
                                        .find(|d| d.key == component.key)
                                        .map_or_else(
                                            || component.key.clone(),
                                            |d| d.meta.name.clone(),
                                        ),
                                    summary,
                                    category: StepCategory::Component,
                                    enabled: step.enabled,
                                    parameters,
                                    // Inside an opened component: its parameters
                                    // that are the opened one's.
                                    exposed: open_definition
                                        .map(|d| {
                                            evanalyzer_app::components::exposed_settings(d, step.id)
                                        })
                                        .unwrap_or_default(),
                                    update_version: project
                                        .components
                                        .iter()
                                        .find(|d| d.key == component.key)
                                        .and_then(|d| {
                                            evanalyzer_app::components::update_available(
                                                d,
                                                &self.component_library.lock().expect("Poisoned"),
                                            )
                                        })
                                        .map(|(_, version)| version.to_string())
                                        .unwrap_or_default(),
                                    help: project
                                        .components
                                        .iter()
                                        .find(|d| d.key == component.key)
                                        .map(|d| {
                                            join_help(
                                                &d.meta.short_description,
                                                &d.meta.description,
                                            )
                                        })
                                        .unwrap_or_default(),
                                    problems: step_problems,
                                    notes,
                                }
                            }
                        }
                    })
                    .collect();
                let total = steps.len() as i32;
                let enabled = steps.iter().filter(|s| s.enabled).count() as i32;
                (
                    name,
                    image_src,
                    steps,
                    total,
                    enabled,
                    total_enabled_across_all,
                )
            } else {
                warn!(
                    "sync_steps_of_selected_pipeline_to_slint: pipeline {:?} not found",
                    pipeline_id
                );
                (
                    String::new(),
                    String::new(),
                    vec![],
                    0,
                    0,
                    total_enabled_across_all,
                )
            }
        };
        let pid = pipeline_id.0 as i32;
        // Model files are read through the backend (the server's, remotely).
        let backend = self.app_state.backend();

        if let Err(e) = crate::helper::ui_thread::invoke_from_event_loop(move || {
            if let Some(ui) = ui_weak.upgrade() {
                let state = ui.global::<PipelinesPanelState>();

                // Keyed by the step's id, not its position: a step inserted
                // or removed above must not move the state to another card.
                let expanded_by_id: std::collections::HashMap<SharedString, bool> =
                    if !reset_expanded {
                        let current = state.get_active_commands();
                        (0..current.row_count())
                            .filter_map(|i| {
                                current.row_data(i).map(|cmd| (cmd.step_id, cmd.expanded))
                            })
                            .collect()
                    } else {
                        std::collections::HashMap::new()
                    };
                // Same for each step's expanded advanced section.
                let show_advanced_by_id: std::collections::HashMap<SharedString, bool> = {
                    let current = state.get_active_commands();
                    (0..current.row_count())
                        .filter_map(|i| {
                            current
                                .row_data(i)
                                .map(|cmd| (cmd.step_id, cmd.show_advanced))
                        })
                        .collect()
                };
                let always_show_advanced = state.get_always_show_advanced();

                let commands: Vec<SlintPipelineCommand> = step_data
                    .into_iter()
                    .map(|d| {
                        let is_pixel_classifier = d.name.as_str() == PIXEL_CLASSIFIER_COMMAND_NAME;
                        let is_ai_object_classifier =
                            d.name.as_str() == AI_OBJECT_CLASSIFIER_COMMAND_NAME;
                        // Model class id -> the model's own display name (in the
                        // model's declared order), for relabeling
                        // `segmentation_mapping`'s `model_class`/`object_class`
                        // leaves below (best-effort: empty if the model hasn't been
                        // set or fails to load - those rows just keep their raw
                        // numeric label in that case). Order matters here: it's what
                        // the relabeled dropdown's full option list is built from
                        // below, and a `HashMap`-only version would shuffle it.
                        let model_class_name_list: Vec<(u32, String)> = if is_pixel_classifier {
                            d.parameters
                                .iter()
                                .find(|p| p.name == "model_path")
                                .and_then(|p| {
                                    load_pixel_classifier_class_labels(
                                        backend.files(),
                                        Path::new(&p.value),
                                    )
                                })
                                .map(|labels| {
                                    labels
                                        .into_iter()
                                        .map(|l| (l.class.as_u32(), l.name))
                                        .collect()
                                })
                                .unwrap_or_default()
                        } else if is_ai_object_classifier {
                            d.parameters
                                .iter()
                                .find(|p| p.name == "model_path")
                                .and_then(|p| {
                                    load_ai_object_classifier_class_labels(
                                        backend.files(),
                                        Path::new(&p.value),
                                    )
                                })
                                .map(|labels| {
                                    labels
                                        .into_iter()
                                        .filter_map(|l| l.class.to_u32().map(|id| (id, l.name)))
                                        .collect()
                                })
                                .unwrap_or_default()
                        } else {
                            Vec::new()
                        };
                        let model_class_names: std::collections::HashMap<u32, String> =
                            model_class_name_list.iter().cloned().collect();
                        let model_class_name_options: Vec<String> = model_class_name_list
                            .iter()
                            .map(|(_, name)| name.clone())
                            .collect();

                        let step_id = step_id_text(d.id);
                        let step_show_advanced =
                            show_advanced_by_id.get(&step_id).copied().unwrap_or(false);
                        let show_advanced = always_show_advanced || step_show_advanced;
                        let (advanced_count, advanced_changed) = advanced_stats(&d.parameters);
                        let exposed = d.exposed.clone();
                        let notes = d.notes.clone();
                        let (advanced_defs, basic_defs): (Vec<ParameterDef>, Vec<ParameterDef>) =
                            filter_for_display(d.parameters, show_advanced)
                                .into_iter()
                                .partition(|p| p.advanced);

                        let convert = |p: ParameterDef| -> CommandParameter {
                            let p_name = p.name.clone();
                            let group_items: Vec<GroupItem> = p
                                .groups
                                .into_iter()
                                .enumerate()
                                .map(|(entry, group)| GroupItem {
                                    fields: ModelRc::new(VecModel::from(
                                        group
                                            .into_iter()
                                            .map(|lp| {
                                                let leaf_exposed = exposed.contains(&format!(
                                                    "{p_name}.{entry}.{}",
                                                    lp.name
                                                ));
                                                let relabeled = p_name == "segmentation_mapping"
                                                    && ((is_pixel_classifier
                                                        && lp.name == "model_class")
                                                        || (is_ai_object_classifier
                                                            && lp.name == "object_class"));
                                                let model_name = relabeled
                                                    .then(|| lp.value.parse::<u32>().ok())
                                                    .flatten()
                                                    .and_then(|id| model_class_names.get(&id));
                                                // The model's predicted class is fixed by the
                                                // model file (one row per model class, see
                                                // `reconcile_*_mapping`), not something to
                                                // reassign by hand - `ParamType.obj-class` would
                                                // let the user pick, but its widget always
                                                // renders the *project's* class list regardless
                                                // of what label/value we set here.
                                                // `ParamType.dropdown` is the one fully generic
                                                // combo box (its options/value are just plain
                                                // strings we control), so route through that
                                                // instead: show every class the model declares
                                                // (for context - `apply_param_change` only
                                                // accepts a numeric id back, so picking a
                                                // different one is a harmless no-op), with this
                                                // row's own class preselected.
                                                let (param_type, value, options) = match model_name
                                                {
                                                    Some(name) => (
                                                        ParamType::Dropdown,
                                                        name.clone(),
                                                        model_class_name_options.clone(),
                                                    ),
                                                    None => (
                                                        map_cfg_param_type(lp.param_type.clone()),
                                                        lp.value.clone(),
                                                        with_component_flags(
                                                            &lp.param_type,
                                                            &lp.value,
                                                            lp.options.clone(),
                                                        ),
                                                    ),
                                                };
                                                LeafParam {
                                                    exposed: leaf_exposed,
                                                    name: lp.name.into(),
                                                    display_name: lp.display_name.into(),
                                                    description: lp.description.into(),
                                                    value: value.into(),
                                                    param_type,
                                                    options: ModelRc::new(VecModel::from(
                                                        options
                                                            .into_iter()
                                                            .map(SharedString::from)
                                                            .collect::<Vec<_>>(),
                                                    )),
                                                    min: lp.min,
                                                    max: lp.max,
                                                    step: lp.step,
                                                }
                                            })
                                            .collect::<Vec<_>>(),
                                    )),
                                })
                                .collect();
                            let has_model_info = (is_pixel_classifier || is_ai_object_classifier)
                                && p_name == "model_path"
                                && !p.value.is_empty();
                            let note = notes.get(&p_name).cloned().unwrap_or_default();
                            CommandParameter {
                                wiring: note.wiring.into(),
                                problem: note.problem.into(),
                                problem_error: note.error,
                                name: p.name.into(),
                                display_name: p.display_name.into(),
                                description: p.description.into(),
                                value: p.value.clone().into(),
                                param_type: map_cfg_param_type(p.param_type.clone()),
                                options: ModelRc::new(VecModel::from(
                                    with_component_flags(&p.param_type, &p.value, p.options)
                                        .into_iter()
                                        .map(SharedString::from)
                                        .collect::<Vec<_>>(),
                                )),
                                min: p.min,
                                max: p.max,
                                step: p.step,
                                exposed: exposed.contains(&p_name),
                                group_items: ModelRc::new(VecModel::from(group_items)),
                                has_model_info,
                                section_header: SharedString::new(),
                                section: SharedString::new(),
                                short_description: SharedString::new(),
                                more_help: false,
                            }
                        };
                        let in_sections = |defs: Vec<ParameterDef>| -> Vec<CommandParameter> {
                            let defs = order_by_section(defs);
                            let sections = section_headers(&defs);
                            defs.into_iter()
                                .zip(sections)
                                .map(|(p, section)| CommandParameter {
                                    section_header: section.into(),
                                    section: p.section.clone().into(),
                                    short_description: short_description(&p.description).into(),
                                    more_help: p
                                        .description
                                        .split("\n\n")
                                        .filter(|paragraph| !paragraph.trim().is_empty())
                                        .count()
                                        > 1,
                                    ..convert(p)
                                })
                                .collect()
                        };
                        let sections: Vec<SharedString> = order_by_section(basic_defs.clone())
                            .iter()
                            .map(|p| SharedString::from(p.section.as_str()))
                            .fold(Vec::new(), |mut sections, section| {
                                if sections.last() != Some(&section) {
                                    sections.push(section);
                                }
                                sections
                            });
                        let params = in_sections(basic_defs);
                        let advanced_params = if show_advanced {
                            in_sections(advanced_defs)
                        } else {
                            Vec::new()
                        };
                        let (section_views, card_tables) = {
                            card_layout(
                                &params,
                                |section| {
                                    open_sections.contains(&(
                                        pid as u32,
                                        step_id.to_string(),
                                        section.to_string(),
                                    ))
                                },
                                &value_names,
                            )
                        };
                        SlintPipelineCommand {
                            section_views: ModelRc::new(VecModel::from(section_views)),
                            card_tables: ModelRc::new(VecModel::from(card_tables)),
                            id: d.idx,
                            step_id: step_id.clone(),
                            name: d.name.into(),
                            category: d.category,
                            summary: d.summary.into(),
                            enabled: d.enabled,
                            expanded: expanded_by_id.get(&step_id).copied().unwrap_or(false),
                            parameters: ModelRc::new(VecModel::from(params)),
                            advanced_parameters: ModelRc::new(VecModel::from(advanced_params)),
                            show_advanced: step_show_advanced,
                            advanced_count: advanced_count as i32,
                            advanced_changed: advanced_changed as i32,
                            update_version: d.update_version.into(),
                            help: d.help.into(),
                            sections: ModelRc::new(VecModel::from(sections)),
                            problem_errors: d.problems.iter().filter(|p| p.error).count() as i32,
                            problems: ModelRc::new(VecModel::from(
                                d.problems.into_iter().map(problem_item).collect::<Vec<_>>(),
                            )),
                        }
                    })
                    .collect();
                state.set_active_pipeline_name(pipeline_name.into());
                state.set_open_component_name(open_component_name.into());
                ui.global::<ComponentClassesState>()
                    .set_classes(ModelRc::new(VecModel::from(
                        component_class_items
                            .iter()
                            .map(|class| ClassItemData {
                                id: class.id as i32,
                                name: class.name.clone().into(),
                                color: slint::Color::from_rgb_u8(
                                    ((class.color >> 16) & 0xff) as u8,
                                    ((class.color >> 8) & 0xff) as u8,
                                    (class.color & 0xff) as u8,
                                ),
                                visible: true,
                                count: 0,
                            })
                            .collect::<Vec<_>>(),
                    )));
                state.set_open_component_problem(open_component_problem.into());
                state.set_open_component_values(open_component_values.into());
                state.set_project_problem_errors(
                    problem_data.iter().filter(|p| p.error).count() as i32
                );
                state.set_project_problems(ModelRc::new(VecModel::from(
                    problem_data
                        .into_iter()
                        .map(problem_item)
                        .collect::<Vec<_>>(),
                )));
                state.set_open_component_read_only(read_only);
                state.set_active_pipeline_image_source(pipeline_image_source.into());
                state.set_active_commands(ModelRc::new(VecModel::from(commands)));

                // Keep the pipeline tab's step counts in sync
                let pipelines = state.get_pipelines();
                for i in 0..pipelines.row_count() {
                    if let Some(mut p) = pipelines.row_data(i) {
                        if p.id == pid {
                            p.total_step_count = total_steps_count;
                            p.enabled_step_count = enabled_steps_count;
                            pipelines.set_row_data(i, p);
                            break;
                        }
                    }
                }
                state.set_total_enabled_steps(total_enabled_across_all);
            }
        }) {
            warn!("Failed to sync steps to Slint: {}", e);
        }
    }
}

/// What values are called in a collapsed section's summary: the images'
/// channel names (by channel), the classes' names (by `ObjectClass::to_text`).
struct ValueNames {
    channels: Vec<String>,
    classes: std::collections::HashMap<String, String>,
}

/// Setting `p`'s value as a collapsed section's summary shows it: a
/// channel or class by its name, a number as it is.
fn value_summary(p: &CommandParameter, names: &ValueNames) -> String {
    let value = p.value.as_str();
    let class = |v: &str| {
        names
            .classes
            .get(v)
            .filter(|n| !n.is_empty())
            .cloned()
            .unwrap_or_else(|| match v {
                "-1" => "no class".into(),
                "0" => "background".into(),
                v => format!("class {v}"),
            })
    };
    match p.param_type {
        ParamType::ImageChannel => value
            .parse::<usize>()
            .ok()
            .and_then(|i| names.channels.get(i))
            .filter(|n| !n.is_empty())
            .cloned()
            .unwrap_or_else(|| format!("channel {value}")),
        ParamType::ObjClass => class(value),
        ParamType::MultiObjClass => value
            .split(',')
            .map(str::trim)
            .filter(|v| !v.is_empty())
            .map(class)
            .collect::<Vec<_>>()
            .join(", "),
        ParamType::SegClass => format!("seg. {value}"),
        ParamType::Toggle => format!(
            "{} {}",
            p.display_name,
            if value == "true" { "on" } else { "off" }
        ),
        ParamType::Group => format!("{} entries", p.group_items.row_count()),
        ParamType::Script | ParamType::Label => String::new(),
        _ => value.to_string(),
    }
}

/// Writes setting `changed`'s value (and options, a class list's flags)
/// into the copies `card`'s sections and tables show (see `card_layout`).
fn patch_card_copies(card: &SlintPipelineCommand, changed: &CommandParameter) {
    let patch = |list: ModelRc<CommandParameter>| {
        for k in 0..list.row_count() {
            if let Some(mut copy) = list.row_data(k)
                && copy.name == changed.name
            {
                copy.value = changed.value.clone();
                copy.options = changed.options.clone();
                list.set_row_data(k, copy);
            }
        }
    };
    for view in card.section_views.iter() {
        patch(view.params);
    }
    for table in card.card_tables.iter() {
        for row in table.rows.iter() {
            patch(row.cells);
        }
    }
}

/// What a check found among `list`: 0 nothing, 1 a warning, 2 an error.
fn problem_level<'a>(list: impl IntoIterator<Item = &'a CommandParameter>) -> i32 {
    list.into_iter()
        .map(|p| match (p.problem.is_empty(), p.problem_error) {
            (true, _) => 0,
            (false, false) => 1,
            (false, true) => 2,
        })
        .max()
        .unwrap_or(0)
}

/// Brings the problem marks of `card`'s sections and tables in line with
/// their settings' (after these were patched).
fn refresh_section_problems(card: &SlintPipelineCommand) {
    for i in 0..card.section_views.row_count() {
        if let Some(mut view) = card.section_views.row_data(i) {
            let level = problem_level(&view.params.iter().collect::<Vec<_>>());
            if view.problem != level {
                view.problem = level;
                card.section_views.set_row_data(i, view);
            }
        }
    }
    for t in 0..card.card_tables.row_count() {
        let Some(mut table) = card.card_tables.row_data(t) else {
            continue;
        };
        let mut problems: Vec<SharedString> = Vec::new();
        let mut error = false;
        for r in 0..table.rows.row_count() {
            let Some(mut row) = table.rows.row_data(r) else {
                continue;
            };
            let cells: Vec<CommandParameter> = row.cells.iter().collect();
            for cell in &cells {
                if !cell.problem.is_empty() && !problems.contains(&cell.problem) {
                    problems.push(cell.problem.clone());
                }
                error |= cell.problem_error;
            }
            let level = problem_level(&cells);
            if row.problem != level {
                row.problem = level;
                table.rows.set_row_data(r, row);
            }
        }
        let now: Vec<SharedString> = table.problems.iter().collect();
        if now != problems || table.problem_error != error {
            table.problems = ModelRc::new(VecModel::from(problems));
            table.problem_error = error;
            card.card_tables.set_row_data(t, table);
        }
    }
}

/// How a card shows `params` (in section order): those without a section
/// as rows, always; a section collapses to its values on one line,
/// open (`is_open`) it shows its settings; sections with the same settings
/// (at least two, each up to 4 simple ones) are a table instead: a row per
/// section, a column per setting.
fn card_layout(
    params: &[CommandParameter],
    is_open: impl Fn(&str) -> bool,
    names: &ValueNames,
) -> (Vec<SectionView>, Vec<CardTable>) {
    let mut groups: Vec<(SharedString, Vec<CommandParameter>)> = Vec::new();
    for p in params {
        match groups.last_mut() {
            Some((name, list)) if *name == p.section => list.push(p.clone()),
            _ => groups.push((p.section.clone(), vec![p.clone()])),
        }
    }
    if groups.is_empty() {
        return (Vec::new(), Vec::new());
    }
    let in_a_cell = |p: &CommandParameter| {
        matches!(
            p.param_type,
            ParamType::Number
                | ParamType::Text
                | ParamType::Spinner
                | ParamType::Dropdown
                | ParamType::Toggle
                | ParamType::ObjClass
                | ParamType::SegClass
                | ParamType::ImageChannel
                | ParamType::PixelUnits
                | ParamType::SizeUnits
        )
    };
    // A section's settings, if a table can show them: (label, type) each.
    let signature = |list: &[CommandParameter]| -> Option<Vec<(SharedString, ParamType)>> {
        (list.len() <= 4 && list.iter().all(in_a_cell)).then(|| {
            list.iter()
                .map(|p| (p.display_name.clone(), p.param_type))
                .collect()
        })
    };
    let signatures: Vec<Option<Vec<(SharedString, ParamType)>>> = groups
        .iter()
        .map(|(name, list)| {
            if name.is_empty() {
                None
            } else {
                signature(list)
            }
        })
        .collect();
    let level = |list: &[CommandParameter]| problem_level(list);

    // Which table each section is a row of: (its signature, its sections).
    let mut tables: Vec<(Vec<(SharedString, ParamType)>, Vec<usize>)> = Vec::new();
    let mut table_of: Vec<Option<usize>> = vec![None; groups.len()];
    for (i, signature) in signatures.iter().enumerate() {
        let Some(signature) = signature else {
            continue;
        };
        if signatures
            .iter()
            .filter(|s| s.as_ref() == Some(signature))
            .count()
            < 2
        {
            continue;
        }
        let t = match tables.iter().position(|(s, _)| s == signature) {
            Some(t) => t,
            None => {
                tables.push((signature.clone(), Vec::new()));
                tables.len() - 1
            }
        };
        tables[t].1.push(i);
        table_of[i] = Some(t);
    }

    let views = groups
        .iter()
        .enumerate()
        .map(|(i, (name, list))| SectionView {
            name: name.clone(),
            summary: list
                .iter()
                .map(|p| value_summary(p, names))
                .filter(|v| !v.is_empty())
                .collect::<Vec<_>>()
                .join(" · ")
                .into(),
            open: is_open(name),
            kind: match table_of[i] {
                _ if name.is_empty() => 3,
                Some(t) if tables[t].1[0] == i => 1,
                Some(_) => 2,
                None => 0,
            },
            table: table_of[i].unwrap_or(0) as i32,
            problem: level(list),
            params: ModelRc::new(VecModel::from(list.clone())),
        })
        .collect();
    let tables = tables
        .into_iter()
        .map(|(signature, members)| {
            let cells: Vec<&CommandParameter> =
                members.iter().flat_map(|&i| groups[i].1.iter()).collect();
            let mut problems: Vec<SharedString> = Vec::new();
            for cell in &cells {
                if !cell.problem.is_empty() && !problems.contains(&cell.problem) {
                    problems.push(cell.problem.clone());
                }
            }
            CardTable {
                columns: ModelRc::new(VecModel::from(
                    signature
                        .into_iter()
                        .map(|(label, _)| label)
                        .collect::<Vec<_>>(),
                )),
                rows: ModelRc::new(VecModel::from(
                    members
                        .iter()
                        .map(|&i| CardTableRow {
                            name: groups[i].0.clone(),
                            problem: level(&groups[i].1),
                            cells: ModelRc::new(VecModel::from(groups[i].1.clone())),
                        })
                        .collect::<Vec<_>>(),
                )),
                problem_error: cells.iter().any(|c| c.problem_error),
                problems: ModelRc::new(VecModel::from(problems)),
            }
        })
        .collect();
    (views, tables)
}

/// A problem as the panel shows it - `Send`, so it can cross to the UI thread.
#[derive(Clone, PartialEq)]
struct ProblemData {
    text: String,
    error: bool,
    place: String,
    fix_label: String,
    /// Its number in `PipelinesController::last_problems`.
    index: i32,
}

/// What's shown below a setting: what it's wired to (only in debug mode),
/// and the first problem about it.
#[derive(Clone, Default, PartialEq)]
struct ParamNote {
    wiring: String,
    problem: String,
    error: bool,
}

/// `problems` as the panel shows them.
fn problem_data(
    project: &ProjectSettings,
    problems: &[evanalyzer_app::components::ComponentProblem],
    debug_mode: bool,
) -> Vec<ProblemData> {
    problems
        .iter()
        .enumerate()
        .filter(|(_, problem)| shown_in_pipelines(problem, debug_mode))
        .map(|(index, problem)| ProblemData {
            text: if debug_mode && !problem.detail.is_empty() {
                format!("{} · in {}", problem.message, problem.detail)
            } else {
                problem.message.clone()
            },
            error: problem.severity == evanalyzer_app::components::Severity::Error,
            place: problem_place(project, problem),
            fix_label: problem.fix.as_ref().map(|f| f.label()).unwrap_or_default(),
            index: index as i32,
        })
        .collect()
}

/// Whether `problem` shows outside its component - in the panel's list and
/// on the cards: a warning about a component's own definition only in debug
/// mode (who doesn't look into components can't act on it; a run works).
/// Opened, a component shows all of its own.
fn shown_in_pipelines(
    problem: &evanalyzer_app::components::ComponentProblem,
    debug_mode: bool,
) -> bool {
    debug_mode
        || problem.step.is_some()
        || problem.severity == evanalyzer_app::components::Severity::Error
}

fn problem_item(p: ProblemData) -> ProblemItem {
    ProblemItem {
        text: p.text.into(),
        error: p.error,
        r#where: p.place.into(),
        fix_label: p.fix_label.into(),
        index: p.index,
    }
}

/// Marks setting `name` with `problem` - the first one about it.
fn mark_note(
    notes: &mut std::collections::HashMap<String, ParamNote>,
    name: &str,
    problem: &evanalyzer_app::components::ComponentProblem,
) {
    let note = notes.entry(name.to_string()).or_default();
    if note.problem.is_empty() {
        note.problem = problem.message.clone();
        note.error = problem.severity == evanalyzer_app::components::Severity::Error;
    }
}

/// What [`step_marks`] marks the shown steps from.
struct Marks<'a> {
    project: &'a ProjectSettings,
    problems: &'a [evanalyzer_app::components::ComponentProblem],
    /// `problems` as the panel shows them.
    data: &'a [ProblemData],
    pipeline_id: PipelineId,
    /// The panel shows the pipeline's steps (no component opened).
    in_pipeline: bool,
    open_definition: Option<&'a ComponentDefinition>,
    /// The pipeline step an opened component is only viewed from.
    viewed_from: Option<(PipelineId, StepId)>,
    /// Show what parameters and settings are wired to.
    debug_mode: bool,
}

/// The problems of shown step `step`, and the notes below its settings:
/// on a component card, the problems about each parameter (at this step,
/// else the component's own) and what it sets; in an opened component,
/// which parameter sets each setting, and the problems about that
/// parameter (a list entry's on its list).
fn step_marks(
    marks: &Marks,
    step: &PipelineStepSettings,
) -> (
    Vec<ProblemData>,
    std::collections::HashMap<String, ParamNote>,
) {
    let at_step = |p: &evanalyzer_app::components::ComponentProblem| {
        marks.in_pipeline && p.step == Some((marks.pipeline_id, step.id))
    };
    let step_problems: Vec<ProblemData> = marks
        .problems
        .iter()
        .enumerate()
        .filter(|(_, p)| at_step(p))
        .filter_map(|(index, _)| marks.data.iter().find(|d| d.index == index as i32))
        .cloned()
        .collect();
    let mut notes: std::collections::HashMap<String, ParamNote> = std::collections::HashMap::new();
    if let StepContent::Component(instance) = &step.content {
        for problem in marks.problems.iter().filter(|p| {
            at_step(p)
                || (p.step.is_none()
                    && p.component.as_deref() == Some(instance.key.as_str())
                    && (!marks.in_pipeline || shown_in_pipelines(p, marks.debug_mode)))
        }) {
            for key in &problem.parameters {
                mark_note(&mut notes, key.as_str(), problem);
            }
        }
        if marks.debug_mode {
            for (key, text) in evanalyzer_app::components::parameter_wiring(
                &marks.project.components,
                &instance.key,
            ) {
                if !text.is_empty() {
                    notes.entry(key.0).or_default().wiring = format!("Sets {text}");
                }
            }
        }
    }
    if let Some(definition) = marks.open_definition {
        let about_it: Vec<&evanalyzer_app::components::ComponentProblem> = marks
            .problems
            .iter()
            .filter(|p| {
                (p.step.is_none() && p.component.as_deref() == Some(definition.key.as_str()))
                    || (marks.viewed_from.is_some() && p.step == marks.viewed_from)
            })
            .collect();
        for (path, _, label) in
            evanalyzer_app::components::step_settings(&marks.project.components, step)
        {
            let top = path.split('.').next().unwrap_or(&path).to_string();
            if marks.debug_mode
                && let Some(wiring) =
                    evanalyzer_app::components::setting_wiring(definition, step.id, &path)
            {
                let text = if top == path {
                    wiring
                } else {
                    format!("{label}: {wiring}")
                };
                let note = notes.entry(top.clone()).or_default();
                if note.wiring.is_empty() {
                    note.wiring = text;
                } else {
                    note.wiring = format!("{}\n{text}", note.wiring);
                }
            }
            for problem in &about_it {
                let sets_it = definition.parameters.iter().any(|p| {
                    problem.parameters.contains(&p.key)
                        && p.bindings
                            .iter()
                            .any(|b| b.step == step.id && b.param == path)
                });
                if sets_it {
                    mark_note(&mut notes, &top, problem);
                }
            }
        }
    }
    (step_problems, notes)
}

/// Where `problem` is, for the panel's problem list: "Channel1 · #3" (the
/// step's id, as its card shows it), "Channel1" for the pipeline itself,
/// "◆ Coloc 3CH" for a component.
fn problem_place(
    project: &ProjectSettings,
    problem: &evanalyzer_app::components::ComponentProblem,
) -> String {
    if let Some((pipeline_id, step_id)) = problem.step {
        let name = project
            .pipelines
            .iter()
            .find(|p| p.id == pipeline_id)
            .map_or_else(|| pipeline_id.0.to_string(), |p| p.name.clone());
        return if step_id == StepId::UNASSIGNED {
            format!("{name} · image source")
        } else {
            format!("{name} · #{}", step_id_text(step_id))
        };
    }
    problem
        .component
        .as_ref()
        .map(|key| {
            let name = project
                .components
                .iter()
                .find(|d| &d.key == key)
                .map_or_else(|| key.clone(), |d| d.meta.name.clone());
            format!("◆ {name}")
        })
        .unwrap_or_default()
}

/// The step-card / picker category of a command category.
/// Sets the first of `authors` - adds it if there's none; both texts empty
/// removes it.
fn set_first_author(authors: &mut Vec<AuthorInformation>, name: &str, organization: &str) {
    let (name, organization) = (name.trim(), organization.trim());
    if name.is_empty() && organization.is_empty() {
        if !authors.is_empty() {
            authors.remove(0);
        }
        return;
    }
    let author = AuthorInformation {
        full_name: name.to_string(),
        organization: organization.to_string(),
    };
    match authors.first_mut() {
        Some(first) => *first = author,
        None => authors.push(author),
    }
}

/// A short and a long description as one help text: either alone, or both
/// with a blank line between.
fn join_help(short: &str, long: &str) -> String {
    [short.trim(), long.trim()]
        .into_iter()
        .filter(|text| !text.is_empty())
        .collect::<Vec<_>>()
        .join("\n\n")
}

/// The help text of the command named `name` (as `PipelineCommand::name`):
/// its summary and description from the command list.
fn command_help(name: &str) -> String {
    all_command_meta()
        .iter()
        .find(|meta| meta.name == name)
        .map(|meta| join_help(meta.summary, meta.description))
        .unwrap_or_default()
}

fn step_category(category: CommandCategory) -> StepCategory {
    match category {
        CommandCategory::Preprocess => StepCategory::Preprocess,
        CommandCategory::Segment => StepCategory::Segment,
        CommandCategory::InstanceSegmentation => StepCategory::InstanceSegmentation,
        CommandCategory::Measure => StepCategory::Measure,
        CommandCategory::Object => StepCategory::Object,
        CommandCategory::Dataset => StepCategory::Dataset,
    }
}

fn to_command_def(m: &CommandMeta) -> CommandDef {
    let cat = step_category(m.category);
    let detail = CommandDef {
        id: m.id,
        name: m.name.into(),
        summary: m.summary.into(),
        description: m.description.into(),
        category: cat,
        icon_glyph: "ƒ".into(),
        keywords: m.name.to_ascii_lowercase().into(),
        source: "built-in".into(),
        favorite: false,
        recent: false,
        default_params: ModelRc::default(),
        is_component: false,
        authors: ModelRc::default(),
        creation_time: "".into(),
    };
    detail
}

/// A step's name as lists show it: the command's, or the component's.
/// A step as the card shows it: its id (the card's number) and name -
/// "#2 EV Detection". Tells equal steps apart.
fn step_ref(project: &ProjectSettings, step: &PipelineStepSettings) -> String {
    format!("#{} {}", step.id, step_label(project, step))
}

/// What parameter `key` of component `component` sets first - "#2 EV
/// Detection › Output class".
fn parameter_target(
    project: &ProjectSettings,
    component: &str,
    key: &ComponentParamKey,
) -> Option<String> {
    let definition = project.components.iter().find(|d| d.key == component)?;
    let binding = definition
        .parameters
        .iter()
        .find(|p| &p.key == key)?
        .bindings
        .first()?;
    let step = definition.steps.iter().find(|s| s.id == binding.step)?;
    let (_, label) =
        evanalyzer_app::components::step_setting(&project.components, step, &binding.param)?;
    Some(format!("{} › {label}", step_ref(project, step)))
}

fn step_label(project: &ProjectSettings, step: &PipelineStepSettings) -> String {
    match &step.content {
        StepContent::Command(command) => command.name().to_string(),
        StepContent::Component(instance) => project
            .components
            .iter()
            .find(|d| d.key == instance.key)
            .map_or_else(|| instance.key.clone(), |d| d.meta.name.clone()),
    }
}

/// Picker ids of components start here - commands have small positive
/// ids.
const COMPONENT_PICKER_ID_BASE: i32 = 1_000_000;

/// Picker ids of the component library's components (not yet in the project).
const LIBRARY_COMPONENT_PICKER_ID_BASE: i32 = 2_000_000;

/// The picker entry of library component number `idx` (file `path`).
fn library_component_to_command_def(
    idx: usize,
    path: &Path,
    template: &ComponentTemplate,
) -> CommandDef {
    let definition = &template.component;
    let version = definition
        .meta
        .version
        .map(|v| format!(" · v{v}"))
        .unwrap_or_default();
    let defaults: Vec<DefaultParamPreview> = definition
        .parameters
        .iter()
        .map(|p| DefaultParamPreview {
            name: p.label.clone().into(),
            value: param_value_display(&p.default).into(),
        })
        .collect();
    let authors: Vec<CommandAuthor> = definition
        .meta
        .authors
        .iter()
        .map(|a| CommandAuthor {
            full_name: a.full_name.clone().into(),
            organization: a.organization.clone().into(),
        })
        .collect();
    CommandDef {
        id: LIBRARY_COMPONENT_PICKER_ID_BASE + idx as i32,
        name: definition.meta.name.clone().into(),
        summary: if definition.meta.short_description.is_empty() {
            "Component from the library".into()
        } else {
            definition.meta.short_description.clone().into()
        },
        description: definition.meta.description.clone().into(),
        category: StepCategory::Component,
        icon_glyph: "◆".into(),
        keywords: definition.meta.name.to_ascii_lowercase().into(),
        source: format!(
            "library · {}{version}",
            evanalyzer_app::components::library_name(path)
        )
        .into(),
        favorite: false,
        recent: false,
        default_params: ModelRc::new(VecModel::from(defaults)),
        is_component: true,
        authors: ModelRc::new(VecModel::from(authors)),
        creation_time: definition
            .meta
            .creation_time
            .format("%Y-%m-%d")
            .to_string()
            .into(),
    }
}

/// A component parameter's value as the picker's detail pane shows it.
fn param_value_display(value: &ParamValue) -> String {
    match value {
        ParamValue::Bool(b) => if *b { "on" } else { "off" }.to_string(),
        ParamValue::Int(i) => i.to_string(),
        ParamValue::Float(f) => f.to_string(),
        ParamValue::Text(t) => t.clone(),
        ParamValue::List(items) => items
            .iter()
            .map(param_value_display)
            .collect::<Vec<_>>()
            .join(", "),
    }
}

/// The picker entry of the project's component number `idx`: shown with the
/// component colour; its source line names where it lives and the categories of
/// its first and last command (e.g. "this project · Segment → Object").
fn component_to_command_def(
    idx: usize,
    definition: &ComponentDefinition,
    project: &ProjectSettings,
) -> CommandDef {
    let commands: Vec<&PipelineCommand> = definition
        .steps
        .iter()
        .filter_map(|s| s.command())
        .collect();
    let category_name = |c: &&PipelineCommand| category_name(*c.category());
    let flow = match (commands.first(), commands.last()) {
        (Some(first), Some(last)) if commands.len() > 1 => {
            format!(" · {} → {}", category_name(first), category_name(last))
        }
        (Some(only), _) => format!(" · {}", category_name(only)),
        _ => String::new(),
    };
    let uses = evanalyzer_app::components::component_uses(project, &definition.key);
    let summary = if definition.meta.short_description.is_empty() {
        format!("Component · used {uses}×")
    } else {
        format!("{} · used {uses}×", definition.meta.short_description)
    };
    let defaults: Vec<DefaultParamPreview> = definition
        .parameters
        .iter()
        .map(|p| DefaultParamPreview {
            name: p.label.clone().into(),
            value: param_value_display(&p.default).into(),
        })
        .collect();
    CommandDef {
        id: COMPONENT_PICKER_ID_BASE + idx as i32,
        name: definition.meta.name.clone().into(),
        summary: summary.into(),
        description: definition.meta.description.clone().into(),
        category: StepCategory::Component,
        icon_glyph: "◆".into(),
        keywords: format!(
            "{} {}",
            definition.meta.name.to_ascii_lowercase(),
            definition.key.to_ascii_lowercase()
        )
        .into(),
        source: format!(
            "this project{}{flow}",
            definition
                .meta
                .version
                .map(|v| format!(" · v{v}"))
                .unwrap_or_default()
        )
        .into(),
        favorite: false,
        recent: false,
        default_params: ModelRc::new(VecModel::from(defaults)),
        is_component: true,
        authors: ModelRc::new(VecModel::from(
            definition
                .meta
                .authors
                .iter()
                .map(|a| CommandAuthor {
                    full_name: a.full_name.clone().into(),
                    organization: a.organization.clone().into(),
                })
                .collect::<Vec<_>>(),
        )),
        creation_time: definition
            .meta
            .creation_time
            .format("%Y-%m-%d")
            .to_string()
            .into(),
    }
}

/// A command category's name, as the picker's chips show it.
fn category_name(category: CommandCategory) -> &'static str {
    match category {
        CommandCategory::Preprocess => "Preprocess",
        CommandCategory::Segment => "Segment",
        CommandCategory::InstanceSegmentation => "Instance Seg.",
        CommandCategory::Measure => "Measure",
        CommandCategory::Object => "Object",
        CommandCategory::Dataset => "Dataset",
    }
}

fn map_cfg_param_type(t: CfgParamType) -> ParamType {
    match t {
        CfgParamType::Number => ParamType::Number,
        CfgParamType::Text => ParamType::Text,
        CfgParamType::Dropdown => ParamType::Dropdown,
        CfgParamType::Toggle => ParamType::Toggle,
        CfgParamType::Slider => ParamType::Slider,
        CfgParamType::Spinner => ParamType::Spinner,
        CfgParamType::Group => ParamType::Group,
        CfgParamType::ObjClass => ParamType::ObjClass,
        CfgParamType::SegClass => ParamType::SegClass,
        CfgParamType::MultiObjClass => ParamType::MultiObjClass,
        CfgParamType::MultiSegClass => ParamType::MultiSegClass,
        CfgParamType::PixelUnits => ParamType::PixelUnits,
        CfgParamType::SizeUnits => ParamType::SizeUnits,
        CfgParamType::Label => ParamType::Label,
        CfgParamType::FilePath => ParamType::FilePath,
        CfgParamType::ImageAddress => ParamType::ImageAddress,
        CfgParamType::ImageChannel => ParamType::ImageChannel,
        CfgParamType::Script => ParamType::Script,
    }
}

/// Display name `PixelClassifier` commands report via `CommandsMeta`'s
/// `display_name` - used to special-case the `model_path` field's info
/// button and the `segmentation_mapping` row labels, since neither fits the
/// otherwise fully generic, codegen-driven parameter rendering.
const PIXEL_CLASSIFIER_COMMAND_NAME: &str = "AI Pixel Classifier";

/// Display name `AiObjectClassifier` commands report via `CommandsMeta`'s
/// `display_name` - the `AiObjectClassifier` analog of `PIXEL_CLASSIFIER_COMMAND_NAME`.
const AI_OBJECT_CLASSIFIER_COMMAND_NAME: &str = "AI Object Classifier";

/// Loads `model_path` and returns the classes it declares, or `None` if the
/// path is empty, unreadable, or not a pixel classifier model. Swallowing
/// the error here is deliberate: callers use this for best-effort UI
/// affordances (row labels, info button availability), not validation - the
/// pipeline step's own `execute()` is what surfaces a real error.
fn load_pixel_classifier_class_labels(
    files: &dyn FileSystem,
    model_path: &Path,
) -> Option<Vec<PixelClassLabel>> {
    if model_path.as_os_str().is_empty() {
        return None;
    }
    let settings = load_classifier_settings(files, model_path).ok()?;
    let AiLearningClassifierSettings::Pixel { class_labels, .. } = settings.classifier else {
        return None;
    };
    Some(class_labels)
}

/// After `model_path` changes on a `PixelClassifier` step, resizes
/// `segmentation_mapping` to exactly one entry per class the newly loaded
/// model declares - the model's classes are a closed set fixed by the file,
/// not something to freely add/remove rows for by hand. Existing
/// `segmentation_class` choices are preserved wherever the same model class is
/// still present; new entries (or a load failure, which clears the list -
/// there's nothing to map without a readable model) default to
/// `SegmentationClass::BACKGROUND`.
fn reconcile_pixel_classifier_mapping(
    files: &dyn FileSystem,
    settings: &mut PixelClassifierSettings,
) {
    let Some(class_labels) = load_pixel_classifier_class_labels(files, &settings.model_path) else {
        settings.segmentation_mapping.clear();
        return;
    };
    let old = std::mem::take(&mut settings.segmentation_mapping);
    settings.segmentation_mapping = class_labels
        .iter()
        .map(|label| {
            let segmentation_class = old
                .iter()
                .find(|m| m.model_class == label.class)
                .map(|m| m.segmentation_class)
                .unwrap_or(SegmentationClass::BACKGROUND);
            SegmentationMappingSettings {
                model_class: label.class,
                segmentation_class,
            }
        })
        .collect();
}

/// Loads `model_path` and returns the classes it declares, or `None` if the
/// path is empty, unreadable, or not an object classifier model - the
/// `AiObjectClassifier` analog of `load_pixel_classifier_class_labels`.
fn load_ai_object_classifier_class_labels(
    files: &dyn FileSystem,
    model_path: &Path,
) -> Option<Vec<ObjectClassLabel>> {
    if model_path.as_os_str().is_empty() {
        return None;
    }
    let settings = load_classifier_settings(files, model_path).ok()?;
    let AiLearningClassifierSettings::Object { class_labels, .. } = settings.classifier else {
        return None;
    };
    Some(class_labels)
}

/// After `model_path` changes on an `AiObjectClassifier` step, resizes
/// `segmentation_mapping` to exactly one entry per class the newly loaded
/// model declares - the `AiObjectClassifier` analog of
/// `reconcile_pixel_classifier_mapping`. Existing `output_class` choices are
/// preserved wherever the same model class is still present; new entries (or
/// a load failure) default to `ObjectClass::Unset` - unlike the pixel
/// classifier's `SegmentationClass::BACKGROUND` default, `Unset` here is a
/// deliberate "not mapped yet" that `AiObjectClassifier::execute` treats the
/// same as no entry at all, rather than a value that gets written out.
fn reconcile_ai_object_classifier_mapping(
    files: &dyn FileSystem,
    settings: &mut AiObjectClassifierSettings,
) {
    let Some(class_labels) = load_ai_object_classifier_class_labels(files, &settings.model_path)
    else {
        settings.segmentation_mapping.clear();
        return;
    };
    let old = std::mem::take(&mut settings.segmentation_mapping);
    settings.segmentation_mapping = class_labels
        .iter()
        .map(|label| {
            let output_class = old
                .iter()
                .find(|m| m.object_class == label.class)
                .map(|m| m.output_class)
                .unwrap_or(ObjectClass::Unset);
            ClassificationMappingSettings {
                object_class: label.class,
                output_class,
            }
        })
        .collect();
}

/// Formats a saved model's metadata + declared classes for the
/// PixelClassifier/AiObjectClassifier step's info dialog.
fn format_classifier_model_info(settings: &AiLearningSettings) -> String {
    let meta = &settings.meta;
    let mut out = format!("{}\n", meta.name);
    if !meta.short_description.is_empty() {
        out.push_str(&format!("{}\n", meta.short_description));
    }
    if !meta.description.is_empty() {
        out.push_str(&format!("\n{}\n", meta.description));
    }
    if !meta.authors.is_empty() {
        out.push_str(&format!(
            "\nAuthor: {}\n",
            meta.authors
                .clone()
                .into_iter()
                .map(|x| x.full_name)
                .collect::<Vec<String>>()
                .join(", ")
        ));
    }
    match &settings.classifier {
        AiLearningClassifierSettings::Pixel {
            input_color,
            class_labels,
            ..
        } => {
            let trained_on = match input_color {
                PixelInputColor::Gray => "greyscale images",
                PixelInputColor::Rgb => "colour images (RGB)",
            };
            out.push_str(&format!("\nTrained on: {trained_on}\n"));
            out.push_str("\nClasses:\n");
            for label in class_labels {
                out.push_str(&format!(
                    "  - {} (id {})\n",
                    label.name,
                    label.class.as_u32()
                ));
            }
        }
        AiLearningClassifierSettings::Object { class_labels, .. } => {
            out.push_str("\nClasses:\n");
            for label in class_labels {
                let id = label
                    .class
                    .to_u32()
                    .map(|v| v.to_string())
                    .unwrap_or_else(|| "unset".to_string());
                out.push_str(&format!("  - {} (id {id})\n", label.name));
            }
        }
    }
    out
}

/// What a step's "Advanced settings" row stands for: how many advanced items
/// there are - advanced settings (also inside list entries) and advanced
/// dropdown options, so the row appears even when a step's only advanced item
/// is an option - and how many advanced settings differ from their default.
/// (A selected advanced option isn't "changed": it stays visible anyway.)
fn advanced_stats(params: &[ParameterDef]) -> (usize, usize) {
    let mut count = 0;
    let mut changed = 0;
    for p in params {
        count += p
            .option_advanced
            .iter()
            .filter(|&&advanced| advanced)
            .count();
        if p.advanced {
            count += 1;
            if !p.default_value.is_empty() && p.value != p.default_value {
                changed += 1;
            }
        }
        for item in &p.groups {
            let (c, ch) = advanced_stats(item);
            count += c;
            changed += ch;
        }
    }
    (count, changed)
}

/// Prepares a step's settings for display. While its advanced section is
/// collapsed, advanced fields inside list entries are left out, and so are
/// advanced dropdown options - except the selected one, so a field never
/// shows a value missing from its list. (Top-level advanced settings are
/// split off by the caller.)
/// A step's settings ordered by section: the ones without a section first,
/// then each section in the order it first appears. Within a section the
/// settings keep their order.
fn order_by_section(mut params: Vec<ParameterDef>) -> Vec<ParameterDef> {
    let mut sections_seen: Vec<String> = vec![String::new()];
    for p in &params {
        if !sections_seen.contains(&p.section) {
            sections_seen.push(p.section.clone());
        }
    }
    params.sort_by_key(|p| sections_seen.iter().position(|c| *c == p.section));
    params
}

/// The first paragraph of a setting's help text, on one line - what the
/// settings dialog shows under its name ("?" shows all of it).
fn short_description(description: &str) -> String {
    description
        .split("\n\n")
        .find(|paragraph| !paragraph.trim().is_empty())
        .unwrap_or("")
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

/// Per setting of `params` (ordered by [`order_by_section`]): the section
/// header shown above it - its section where one starts, else empty.
fn section_headers(params: &[ParameterDef]) -> Vec<String> {
    let mut previous = "";
    params
        .iter()
        .map(|p| {
            let starts = p.section != previous;
            previous = &p.section;
            if starts {
                p.section.clone()
            } else {
                String::new()
            }
        })
        .collect()
}

/// A step id as Slint carries it.
fn step_id_text(id: evanalyzer_cfg::core_types::StepId) -> SharedString {
    id.0.to_string().into()
}

/// The step id `text` (see [`step_id_text`]) names - `None` for none.
fn parse_step_id(text: &str) -> Option<evanalyzer_cfg::core_types::StepId> {
    text.parse().ok().map(evanalyzer_cfg::core_types::StepId)
}

fn filter_for_display(params: Vec<ParameterDef>, show_advanced: bool) -> Vec<ParameterDef> {
    params
        .into_iter()
        .map(|mut p| {
            if !show_advanced && p.option_advanced.len() == p.options.len() {
                let (options, flags): (Vec<String>, Vec<bool>) = p
                    .options
                    .into_iter()
                    .zip(p.option_advanced)
                    .filter(|(option, advanced)| !advanced || *option == p.value)
                    .unzip();
                p.options = options;
                p.option_advanced = flags;
            }
            p.groups = p
                .groups
                .into_iter()
                .map(|item| {
                    filter_for_display(item, show_advanced)
                        .into_iter()
                        .filter(|leaf| show_advanced || !leaf.advanced)
                        .collect()
                })
                .collect();
            p
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use evanalyzer_app::fs::LocalFileSystem;
    use evanalyzer_cfg::settings::images_settings::SeriesSettings;
    use evanalyzer_cfg::settings::meta_data::MetaData;
    use evanalyzer_cfg::settings::object_settings::ObjectMetricSettings;
    use std::collections::BTreeMap;

    fn image_entry_with_objects(rel_path: &str, object_count: usize) -> ImageEntry {
        ImageEntry {
            rel_path: PathBuf::from(rel_path),
            file_size: 42,
            selected_series: 0,
            series: BTreeMap::from([(
                0,
                SeriesSettings {
                    objects: (0..object_count)
                        .map(|_| ObjectMetricSettings::default())
                        .collect(),
                    ..Default::default()
                },
            )]),
        }
    }

    fn project_with_images(entries: &[(&str, usize)]) -> ProjectSettings {
        let mut list = indexmap::IndexMap::new();
        for (rel_path, object_count) in entries {
            list.insert(
                PathBuf::from(rel_path),
                image_entry_with_objects(rel_path, *object_count),
            );
        }
        ProjectSettings {
            meta: MetaData {
                name: "test-project".into(),
                ..Default::default()
            },
            images: ImageSettings {
                root: Some(PathBuf::from("/root")),
                list,
                settings: Default::default(),
            },
            ..Default::default()
        }
    }

    #[test]
    fn build_preview_project_settings_keeps_only_the_selected_image() {
        let project = project_with_images(&[("a.tif", 5), ("b.tif", 3), ("c.tif", 7)]);
        let selected = project
            .images
            .list
            .get(&PathBuf::from("b.tif"))
            .unwrap()
            .clone();

        let preview = PipelinesController::build_preview_project_settings(
            &project,
            PathBuf::from("b.tif"),
            selected,
        );

        assert_eq!(preview.images.list.len(), 1);
        let (path, entry) = preview.images.list.iter().next().unwrap();
        assert_eq!(path, &PathBuf::from("b.tif"));
        assert_eq!(
            entry.series[&0].objects.len(),
            3,
            "preview must keep the selected image's own ROIs"
        );
    }

    /// Regression test: the viewer writes `TStackHandling::AllStacks` into
    /// the project's global settings on every channel/z/t interaction (see
    /// `update_channel_options_in_project` in viewport_image_controller.rs),
    /// so without this forcing, Preview would silently process every t-stack
    /// of the selected image instead of just the one shown - see the
    /// `force_all_t_stacks` tests below for the real run's opposite
    /// requirement (always every frame).
    #[test]
    fn build_preview_project_settings_forces_a_single_t_stack_from_the_global_setting() {
        let mut project = project_with_images(&[("a.tif", 1)]);
        project.images.settings.t_stack = Some(TStackSettings {
            stack_handling: TStackHandling::AllStacks,
            playback_speed: 2.0,
            t_stack: 4,
            frame_interval: None,
        });
        let selected = project
            .images
            .list
            .get(&PathBuf::from("a.tif"))
            .unwrap()
            .clone();
        assert!(
            selected.series[&0].t_stack.is_none(),
            "no per-image override seeded - the global setting above is what's in play"
        );

        let preview = PipelinesController::build_preview_project_settings(
            &project,
            PathBuf::from("a.tif"),
            selected,
        );

        let global = preview.images.settings.t_stack.as_ref().unwrap();
        assert_eq!(global.stack_handling, TStackHandling::SingleStack);
        assert_eq!(global.t_stack, 4, "must keep the frame that was active");

        let (_, entry) = preview.images.list.iter().next().unwrap();
        let local = entry.series[&0].t_stack.as_ref().unwrap();
        assert_eq!(
            local.stack_handling,
            TStackHandling::SingleStack,
            "the per-image override job_executor checks first must also be forced, \
             not just the global fallback"
        );
        assert_eq!(local.t_stack, 4);
    }

    /// A per-image series override (job_executor checks this before the
    /// global setting) must win over the global setting when picking which
    /// frame to keep, same precedence `prepare_t_stack_iterator` uses.
    #[test]
    fn build_preview_project_settings_prefers_the_images_own_t_stack_override() {
        let mut project = project_with_images(&[("a.tif", 1)]);
        project.images.settings.t_stack = Some(TStackSettings {
            stack_handling: TStackHandling::AllStacks,
            playback_speed: 1.0,
            t_stack: 2,
            frame_interval: None,
        });
        let mut selected = project
            .images
            .list
            .get(&PathBuf::from("a.tif"))
            .unwrap()
            .clone();
        selected.series.get_mut(&0).unwrap().t_stack = Some(TStackSettings {
            stack_handling: TStackHandling::AllStacks,
            playback_speed: 1.0,
            t_stack: 7,
            frame_interval: None,
        });

        let preview = PipelinesController::build_preview_project_settings(
            &project,
            PathBuf::from("a.tif"),
            selected,
        );

        let (_, entry) = preview.images.list.iter().next().unwrap();
        let local = entry.series[&0].t_stack.as_ref().unwrap();
        assert_eq!(local.stack_handling, TStackHandling::SingleStack);
        assert_eq!(
            local.t_stack, 7,
            "the image's own override (7) must win over the global setting (2)"
        );
    }

    /// Regression test for the flip side of the same bug: a real run must
    /// always process every t-stack, regardless of whatever the viewer last
    /// left in the project's settings - previously this only happened
    /// because the viewer's `AllStacks` write was never undone, an
    /// accidental side effect rather than something the real run actually
    /// asked for.
    #[test]
    fn force_all_t_stacks_overrides_a_single_stack_global_setting() {
        let mut project = project_with_images(&[("a.tif", 1)]);
        project.images.settings.t_stack = Some(TStackSettings {
            stack_handling: TStackHandling::SingleStack,
            playback_speed: 1.5,
            t_stack: 3,
            frame_interval: Some(30.0),
        });

        PipelinesController::force_all_t_stacks(&mut project);

        let global = project.images.settings.t_stack.as_ref().unwrap();
        assert_eq!(global.stack_handling, TStackHandling::AllStacks);
        assert_eq!(
            global.playback_speed, 1.5,
            "unrelated fields like playback speed must survive the force"
        );
        assert_eq!(global.frame_interval, Some(30.0), "the analysis gets it");
    }

    #[test]
    fn force_all_t_stacks_overrides_every_images_own_series_override_too() {
        let mut project = project_with_images(&[("a.tif", 1), ("b.tif", 1)]);
        for image in project.images.list.values_mut() {
            image.series.get_mut(&0).unwrap().t_stack = Some(TStackSettings {
                stack_handling: TStackHandling::SingleStack,
                playback_speed: 1.0,
                t_stack: 5,
                frame_interval: None,
            });
        }

        PipelinesController::force_all_t_stacks(&mut project);

        for image in project.images.list.values() {
            let local = image.series[&0].t_stack.as_ref().unwrap();
            assert_eq!(
                local.stack_handling,
                TStackHandling::AllStacks,
                "job_executor checks the per-image override first, so it must be forced too, \
                 not just the global setting"
            );
        }
    }

    #[test]
    fn build_preview_project_settings_preserves_every_other_project_field() {
        let mut project = project_with_images(&[("a.tif", 1)]);
        project
            .components
            .push(evanalyzer_cfg::settings::components::ComponentDefinition {
                key: "CELL_RECONSTRUCTION".into(),
                meta: Default::default(),
                classes: vec![],
                steps: Vec::new(),
                parameters: Vec::new(),
                origin: None,
            });
        let selected = project
            .images
            .list
            .get(&PathBuf::from("a.tif"))
            .unwrap()
            .clone();

        let preview = PipelinesController::build_preview_project_settings(
            &project,
            PathBuf::from("a.tif"),
            selected,
        );

        assert_eq!(preview.meta.name, "test-project");
        assert_eq!(preview.images.root, project.images.root);
        assert_eq!(
            preview.components.len(),
            1,
            "the preview runs component instances too"
        );
    }

    #[test]
    fn build_preview_project_settings_does_not_mutate_the_source_project() {
        let project = project_with_images(&[("a.tif", 1), ("b.tif", 2)]);
        let selected = project
            .images
            .list
            .get(&PathBuf::from("a.tif"))
            .unwrap()
            .clone();

        let _preview = PipelinesController::build_preview_project_settings(
            &project,
            PathBuf::from("a.tif"),
            selected,
        );

        assert_eq!(
            project.images.list.len(),
            2,
            "the source project's image list must be untouched"
        );
    }

    #[test]
    fn map_cfg_param_type_maps_every_variant_to_its_slint_counterpart() {
        let cases = [
            (CfgParamType::Number, ParamType::Number),
            (CfgParamType::Text, ParamType::Text),
            (CfgParamType::Dropdown, ParamType::Dropdown),
            (CfgParamType::Toggle, ParamType::Toggle),
            (CfgParamType::Slider, ParamType::Slider),
            (CfgParamType::Spinner, ParamType::Spinner),
            (CfgParamType::Group, ParamType::Group),
            (CfgParamType::ObjClass, ParamType::ObjClass),
            (CfgParamType::SegClass, ParamType::SegClass),
            (CfgParamType::MultiObjClass, ParamType::MultiObjClass),
            (CfgParamType::MultiSegClass, ParamType::MultiSegClass),
            (CfgParamType::PixelUnits, ParamType::PixelUnits),
            (CfgParamType::SizeUnits, ParamType::SizeUnits),
            (CfgParamType::Label, ParamType::Label),
            (CfgParamType::FilePath, ParamType::FilePath),
            (CfgParamType::ImageAddress, ParamType::ImageAddress),
            (CfgParamType::ImageChannel, ParamType::ImageChannel),
            (CfgParamType::Script, ParamType::Script),
        ];
        for (input, expected) in cases {
            let label = format!("{input:?}");
            assert_eq!(map_cfg_param_type(input), expected, "{label}");
        }
    }

    #[test]
    fn to_command_def_maps_every_built_in_category() {
        for meta in all_command_meta() {
            let def = to_command_def(&meta);
            assert_eq!(def.id, meta.id);
            assert_eq!(def.name, SharedString::from(meta.name));
            assert_eq!(def.source, SharedString::from("built-in"));
            let expected_category = match meta.category {
                CommandCategory::Preprocess => StepCategory::Preprocess,
                CommandCategory::Segment => StepCategory::Segment,
                CommandCategory::InstanceSegmentation => StepCategory::InstanceSegmentation,
                CommandCategory::Measure => StepCategory::Measure,
                CommandCategory::Object => StepCategory::Object,
                CommandCategory::Dataset => StepCategory::Dataset,
            };
            assert_eq!(
                def.category, expected_category,
                "id {}: {}",
                meta.id, meta.name
            );
        }
    }

    // -- reconcile_*_classifier_mapping ----------------------------------------------

    #[test]
    fn reconcile_pixel_classifier_mapping_clears_the_mapping_when_model_path_is_empty() {
        let mut settings = PixelClassifierSettings {
            model_path: PathBuf::new(),
            segmentation_mapping: vec![SegmentationMappingSettings {
                model_class: SegmentationClass(1),
                segmentation_class: SegmentationClass(2),
            }],
        };

        reconcile_pixel_classifier_mapping(&LocalFileSystem::default(), &mut settings);

        assert!(
            settings.segmentation_mapping.is_empty(),
            "an empty model path has no classes to map to"
        );
    }

    #[test]
    fn reconcile_pixel_classifier_mapping_clears_the_mapping_when_the_model_file_does_not_exist() {
        let mut settings = PixelClassifierSettings {
            model_path: PathBuf::from("/nonexistent/model.evamodel"),
            segmentation_mapping: vec![SegmentationMappingSettings {
                model_class: SegmentationClass(1),
                segmentation_class: SegmentationClass(2),
            }],
        };

        reconcile_pixel_classifier_mapping(&LocalFileSystem::default(), &mut settings);

        assert!(settings.segmentation_mapping.is_empty());
    }

    #[test]
    fn reconcile_ai_object_classifier_mapping_clears_the_mapping_when_model_path_is_empty() {
        let mut settings = AiObjectClassifierSettings {
            segmentation_mapping: vec![ClassificationMappingSettings {
                object_class: ObjectClass::Global(1),
                output_class: ObjectClass::Global(2),
            }],
            ..Default::default()
        };

        reconcile_ai_object_classifier_mapping(&LocalFileSystem::default(), &mut settings);

        assert!(settings.segmentation_mapping.is_empty());
    }

    // -- format_classifier_model_info --------------------------------------------

    fn saved_classifier_with(
        classifier: AiLearningClassifierSettings,
        meta: evanalyzer_cfg::settings::meta_data::MetaData,
    ) -> AiLearningSettings {
        use evanalyzer_cfg::settings::ai_learning_settings::{
            AiLearningBackendSettings, RandomForestSettings,
        };
        AiLearningSettings {
            schema_version: evanalyzer_cfg::CURRENT_AI_LEARNING_SETTINGS_SCHEMA_VERSION,
            meta,
            backend: AiLearningBackendSettings::RandomForest(RandomForestSettings::default()),
            classifier,
        }
    }

    #[test]
    fn format_classifier_model_info_lists_pixel_classes_with_their_ids() {
        use evanalyzer_cfg::core_types::SegmentationClass;
        let saved = saved_classifier_with(
            AiLearningClassifierSettings::Pixel {
                feature_spec: evanalyzer_cfg::settings::ai_learning_pixel_settings::AiLearningPixelFeatureSettings {
                    channels: vec![],
                },
                input_color: Default::default(),
                class_labels: vec![
                    PixelClassLabel {
                        class: SegmentationClass(1),
                        name: "Nucleus".into(),
                    },
                    PixelClassLabel {
                        class: SegmentationClass(2),
                        name: "Background".into(),
                    },
                ],
            },
            evanalyzer_cfg::settings::meta_data::MetaData {
                name: "My Pixel Model".into(),
                ..Default::default()
            },
        );

        let info = format_classifier_model_info(&saved);

        assert!(info.starts_with("My Pixel Model\n"));
        assert!(info.contains("Trained on: greyscale images"));
        assert!(info.contains("Classes:"));
        assert!(info.contains(&format!("- Nucleus (id {})", SegmentationClass(1).as_u32())));
        assert!(info.contains(&format!(
            "- Background (id {})",
            SegmentationClass(2).as_u32()
        )));
    }

    #[test]
    fn format_classifier_model_info_says_when_a_pixel_model_was_trained_on_colour_images() {
        let saved = saved_classifier_with(
            AiLearningClassifierSettings::Pixel {
                feature_spec: evanalyzer_cfg::settings::ai_learning_pixel_settings::AiLearningPixelFeatureSettings {
                    channels: vec![],
                },
                input_color: PixelInputColor::Rgb,
                class_labels: vec![],
            },
            evanalyzer_cfg::settings::meta_data::MetaData::default(),
        );

        let info = format_classifier_model_info(&saved);

        assert!(info.contains("Trained on: colour images (RGB)"), "{info}");
    }

    #[test]
    fn format_classifier_model_info_lists_object_classes_and_falls_back_to_unset() {
        let saved = saved_classifier_with(
            AiLearningClassifierSettings::Object {
                feature_spec: evanalyzer_cfg::settings::ai_learning_object_settings::AiLearningObjectFeatureSettings {
                    metrics: vec![],
                },
                class_labels: vec![ObjectClassLabel {
                    class: ObjectClass::Unset,
                    name: "Unassigned".into(),
                }],
            },
            evanalyzer_cfg::settings::meta_data::MetaData {
                name: "My Object Model".into(),
                ..Default::default()
            },
        );

        let info = format_classifier_model_info(&saved);

        assert!(info.contains("- Unassigned (id unset)"));
    }

    #[test]
    fn format_classifier_model_info_includes_description_and_author_when_present() {
        let saved = saved_classifier_with(
            AiLearningClassifierSettings::Pixel {
                feature_spec: evanalyzer_cfg::settings::ai_learning_pixel_settings::AiLearningPixelFeatureSettings {
                    channels: vec![],
                },
                input_color: Default::default(),
                class_labels: vec![],
            },
            evanalyzer_cfg::settings::meta_data::MetaData {
                name: "Model".into(),
                short_description: "A short summary".into(),
                description: "A longer description.".into(),
                authors: vec![AuthorInformation{full_name: "Ada Lovelace".into(), organization: "University".into()}],
                ..Default::default()
            },
        );

        let info = format_classifier_model_info(&saved);

        assert!(info.contains("A short summary"));
        assert!(info.contains("A longer description."));
        assert!(info.contains("Author: Ada Lovelace"));
    }

    #[test]
    fn format_classifier_model_info_omits_the_author_line_when_both_names_are_empty() {
        let saved = saved_classifier_with(
            AiLearningClassifierSettings::Pixel {
                feature_spec: evanalyzer_cfg::settings::ai_learning_pixel_settings::AiLearningPixelFeatureSettings {
                    channels: vec![],
                },
                input_color: Default::default(),
                class_labels: vec![],
            },
            evanalyzer_cfg::settings::meta_data::MetaData {
                name: "Model".into(),
                ..Default::default()
            },
        );

        let info = format_classifier_model_info(&saved);

        assert!(!info.contains("Author:"));
    }

    // -- modify_group_item --------------------------------------------------------

    #[test]
    fn modify_group_item_add_appends_a_clone_of_the_last_entry() {
        use evanalyzer_cfg::settings::pipeline_command_settings::{
            ThresholdEntrySettings, ThresholdSettings,
        };
        let (ui, _results_ui) = test_ui_windows();
        let (ui_state, controller) = make_controller(ui.as_weak());
        {
            let mut project = ui_state.get_project_write();
            project.add_pipeline(PipelineSettings {
                id: PipelineId(1),
                name: "Pipeline 1".into(),
                description: None,
                image_source: ImageAddress::Channel(0),
                enabled: true,
                steps: vec![PipelineStepSettings {
                    id: evanalyzer_cfg::core_types::StepId::UNASSIGNED,
                    enabled: true,
                    content: evanalyzer_cfg::settings::pipeline_settings::StepContent::Command(
                        PipelineCommand::Threshold(ThresholdSettings {
                            thresholds: vec![ThresholdEntrySettings {
                                min_threshold: 0.5,
                                ..Default::default()
                            }],
                            ..Default::default()
                        }),
                    ),
                }],
            });
        }

        controller.modify_group_item(1, 0, "thresholds", true, None);

        let project = ui_state.get_project();
        let PipelineCommand::Threshold(settings) = project.pipelines[0].steps[0].command().unwrap()
        else {
            panic!("expected a Threshold command");
        };
        assert_eq!(settings.thresholds.len(), 2);
        assert_eq!(
            settings.thresholds[1].min_threshold, 0.5,
            "the new entry must clone the last existing one, not reset to default"
        );
    }

    #[test]
    fn modify_group_item_remove_drops_the_entry_at_the_given_index() {
        use evanalyzer_cfg::settings::pipeline_command_settings::{
            ThresholdEntrySettings, ThresholdSettings,
        };
        let (ui, _results_ui) = test_ui_windows();
        let (ui_state, controller) = make_controller(ui.as_weak());
        {
            let mut project = ui_state.get_project_write();
            project.add_pipeline(PipelineSettings {
                id: PipelineId(1),
                name: "Pipeline 1".into(),
                description: None,
                image_source: ImageAddress::Channel(0),
                enabled: true,
                steps: vec![PipelineStepSettings {
                    id: evanalyzer_cfg::core_types::StepId::UNASSIGNED,
                    enabled: true,
                    content: evanalyzer_cfg::settings::pipeline_settings::StepContent::Command(
                        PipelineCommand::Threshold(ThresholdSettings {
                            thresholds: vec![
                                ThresholdEntrySettings {
                                    min_threshold: 0.1,
                                    ..Default::default()
                                },
                                ThresholdEntrySettings {
                                    min_threshold: 0.9,
                                    ..Default::default()
                                },
                            ],
                            ..Default::default()
                        }),
                    ),
                }],
            });
        }

        controller.modify_group_item(1, 0, "thresholds", false, Some(0));

        let project = ui_state.get_project();
        let PipelineCommand::Threshold(settings) = project.pipelines[0].steps[0].command().unwrap()
        else {
            panic!("expected a Threshold command");
        };
        assert_eq!(settings.thresholds.len(), 1);
        assert_eq!(settings.thresholds[0].min_threshold, 0.9);
    }

    // -- apply_picker_filter --------------------------------------------------------

    #[test]
    fn apply_picker_filter_with_an_empty_query_shows_every_built_in_command_by_category() {
        let (ui, _results_ui) = test_ui_windows();
        let (_ui_state, controller) = make_controller(ui.as_weak());

        controller.apply_picker_filter(&ui, "");

        let picker = ui.global::<CommandPickerState>();
        let expected_pre = all_command_meta()
            .iter()
            .filter(|m| matches!(m.category, CommandCategory::Preprocess))
            .count() as i32;
        assert_eq!(picker.get_cat_count_pre(), expected_pre);
        assert_eq!(
            picker.get_total_shown(),
            picker.get_cat_count_pre()
                + picker.get_cat_count_seg()
                + picker.get_cat_count_obj()
                + picker.get_cat_count_mea()
                + picker.get_cat_count_cls()
                + picker.get_cat_count_dat(),
        );
    }

    #[test]
    fn apply_picker_filter_by_text_query_matches_name_or_summary_case_insensitively() {
        let (ui, _results_ui) = test_ui_windows();
        let (_ui_state, controller) = make_controller(ui.as_weak());
        let meta = all_command_meta()
            .into_iter()
            .find(|m| matches!(m.category, CommandCategory::Segment))
            .expect("at least one built-in Segment command exists");

        controller.apply_picker_filter(&ui, &meta.name.to_uppercase());

        let picker = ui.global::<CommandPickerState>();
        assert!(
            picker
                .get_shown_segment()
                .iter()
                .any(|c| c.name.as_str() == meta.name),
            "an upper-cased query must still match the command's (lower-cased) name"
        );
    }

    #[test]
    fn apply_picker_filter_with_no_query_match_shows_nothing() {
        let (ui, _results_ui) = test_ui_windows();
        let (_ui_state, controller) = make_controller(ui.as_weak());

        controller.apply_picker_filter(&ui, "no-such-command-exists-xyz");

        let picker = ui.global::<CommandPickerState>();
        assert_eq!(picker.get_total_shown(), 0);
        assert_eq!(picker.get_cat_count_pre(), 0);
    }

    #[test]
    fn apply_picker_filter_restricts_to_the_active_category_chips() {
        let (ui, _results_ui) = test_ui_windows();
        let (_ui_state, controller) = make_controller(ui.as_weak());
        ui.global::<CommandPickerState>().set_fcat_seg(true);

        controller.apply_picker_filter(&ui, "");

        let picker = ui.global::<CommandPickerState>();
        assert_eq!(picker.get_cat_count_pre(), 0, "Preprocess chip is inactive");
        assert!(
            picker.get_cat_count_seg() > 0,
            "Segment chip is active and there are built-in Segment commands"
        );
        assert_eq!(
            picker.get_total_shown(),
            picker.get_cat_count_seg(),
            "only the active category contributes to the total while a chip is set"
        );
    }

    fn component_definition(key: &str, name: &str) -> ComponentDefinition {
        ComponentDefinition {
            key: key.into(),
            meta: evanalyzer_cfg::settings::meta_data::MetaData {
                name: name.into(),
                ..Default::default()
            },
            classes: vec![],
            steps: vec![],
            parameters: vec![],
            origin: None,
        }
    }

    #[test]
    fn the_projects_components_show_with_their_chip_or_when_searching() {
        let (ui, _results_ui) = test_ui_windows();
        let (ui_state, controller) = make_controller(ui.as_weak());
        ui_state.get_project_write().components = vec![
            component_definition("CELLS", "Cell reconstruction"),
            component_definition("SMOOTH", "Smoothing"),
        ];
        let picker = ui.global::<CommandPickerState>();

        controller.apply_picker_filter(&ui, "");
        assert_eq!(
            picker.get_shown_components().row_count(),
            0,
            "not among the categories"
        );
        assert_eq!(picker.get_cat_count_grp(), 2, "but counted on the chip");

        picker.set_filter_components(true);
        controller.apply_picker_filter(&ui, "");
        assert_eq!(picker.get_shown_components().row_count(), 2);
        assert_eq!(
            picker.get_total_shown(),
            2,
            "only the components while the chip is on"
        );
        let first = picker.get_shown_components().row_data(0).unwrap();
        assert!(first.is_component && first.category == StepCategory::Component);
        assert_eq!(first.summary, "Component · used 0×");

        picker.set_filter_components(false);
        controller.apply_picker_filter(&ui, "cell");
        let found = picker.get_shown_components();
        assert_eq!(found.row_count(), 1, "a search finds components too");
        assert_eq!(found.row_data(0).unwrap().name, "Cell reconstruction");
    }

    #[test]
    fn a_project_components_picker_entry_has_its_authors_and_creation_date() {
        let mut definition = component_definition("CELLS", "Cell reconstruction");
        definition.meta.authors = vec![AuthorInformation {
            full_name: "Ada".into(),
            organization: "Lab".into(),
        }];
        definition.meta.version = Some(2);
        let project = ProjectSettings::default();
        let def = component_to_command_def(0, &definition, &project);
        let authors: Vec<_> = def.authors.iter().collect();
        assert_eq!(authors.len(), 1);
        assert_eq!(
            (
                authors[0].full_name.as_str(),
                authors[0].organization.as_str()
            ),
            ("Ada", "Lab")
        );
        assert_eq!(
            def.creation_time,
            definition.meta.creation_time.format("%Y-%m-%d").to_string()
        );
        assert!(def.source.contains("v2"), "{}", def.source);
    }

    #[test]
    fn set_first_author_replaces_adds_and_removes_only_the_first() {
        let author = |n: &str| AuthorInformation {
            full_name: n.into(),
            organization: String::new(),
        };
        let mut authors = Vec::new();
        set_first_author(&mut authors, " Ada ", "");
        assert_eq!(authors.len(), 1);
        assert_eq!(authors[0].full_name, "Ada");
        authors.push(author("Bob"));
        set_first_author(&mut authors, "Ada", "Lab");
        assert_eq!(
            (authors.len(), authors[0].organization.as_str()),
            (2, "Lab")
        );
        set_first_author(&mut authors, "", " ");
        assert_eq!(authors.len(), 1);
        assert_eq!(authors[0].full_name, "Bob", "the co-author stays");
    }

    /// The detail pane shows a component's author and creation date, the rows
    /// together.
    #[test]
    fn the_picker_details_author_and_creation_date_stay_together() {
        use i_slint_backend_testing::ElementHandle;
        let (ui, _results_ui) = test_ui_windows();
        let (ui_state, _controller) = make_controller(ui.as_weak());
        ui.window().set_size(slint::LogicalSize::new(1280.0, 800.0));
        let mut definition = component_definition("CELLS", "Cell reconstruction");
        definition.meta.authors = vec![AuthorInformation {
            full_name: "Ada".into(),
            organization: "Lab".into(),
        }];
        let component = component_to_command_def(0, &definition, &ui_state.get_project());
        ui.global::<GlobalAppState>()
            .set_active_dialog(DialogType::CommandSelectionDialog);

        for (detail, author_text) in [(component, "Ada (Lab)")] {
            let picker = ui.global::<CommandPickerState>();
            picker.set_detail(detail);
            picker.set_has_detail(true);
            let y = |label: &str| {
                ElementHandle::find_by_accessible_label(&ui, label)
                    .next()
                    .unwrap_or_else(|| panic!("{label:?} not found"))
                    .absolute_position()
                    .y
            };
            let (author, created) = (y("Author"), y("Created"));
            assert!(
                created > author && created - author < 40.0,
                "{author_text}: Author at {author}, Created at {created}"
            );
            y(author_text);
        }
    }

    // -- attach_callbacks (live AppWindow) -----------------------------------------

    use crate::editor::test_support::{test_ui_state, test_ui_windows};

    fn make_controller(ui: slint::Weak<AppWindow>) -> (Arc<UiState>, Arc<PipelinesController>) {
        let ui_state = test_ui_state();
        let viewport_controller = Arc::new(ViewportController::new(
            ui.clone(),
            ui_state.clone(),
            Arc::new(crate::editor::viewport_cache::ViewportCache::new(
                ui_state.clone(),
            )),
        ));
        let object_list_controller = Arc::new(ObjectListController::new(
            ui.clone(),
            ui_state.clone(),
            viewport_controller.clone(),
        ));
        let template_controller = Arc::new(TemplateController::new(ui.clone(), ui_state.clone()));
        let controller = Arc::new(PipelinesController::new(
            ui.clone(),
            ui_state.clone(),
            object_list_controller.clone(),
            viewport_controller.clone(),
            template_controller,
            crate::editor::test_support::test_focus_controller(
                ui.clone(),
                &ui_state,
                &object_list_controller,
                &viewport_controller,
            ),
        ));
        (ui_state, controller)
    }

    fn add_pipeline(ui_state: &UiState, id: u32) {
        let mut project = ui_state.get_project_write();
        project.add_pipeline(PipelineSettings {
            id: PipelineId(id),
            name: format!("Pipeline {id}"),
            description: None,
            image_source: ImageAddress::Channel(0),
            enabled: true,
            steps: vec![],
        });
    }

    #[test]
    fn attach_callbacks_auto_preview_toggle_stores_the_flag() {
        let (ui, _results_ui) = test_ui_windows();
        let (_ui_state, controller) = make_controller(ui.as_weak());
        controller.attach_callbacks();

        ui.global::<PipelinesPanelState>().invoke_auto_preview(true);
        assert!(*controller.auto_preview_enabled.lock().unwrap());

        ui.global::<PipelinesPanelState>()
            .invoke_auto_preview(false);
        assert!(!*controller.auto_preview_enabled.lock().unwrap());
    }

    #[test]
    fn attach_callbacks_set_and_clear_breakpoint() {
        let (ui, _results_ui) = test_ui_windows();
        let (_ui_state, controller) = make_controller(ui.as_weak());
        controller.attach_callbacks();

        let step = evanalyzer_cfg::core_types::StepId(0x2a);
        ui.global::<PipelinesPanelState>()
            .invoke_set_breakpoint(7, step_id_text(step), 1);
        assert_eq!(
            *controller.breakpoint.lock().unwrap(),
            Some((7, step, evanalyzer_cfg::core_types::BreakpointMode::Stop))
        );

        ui.global::<PipelinesPanelState>()
            .invoke_set_breakpoint(7, step_id_text(step), 2);
        assert_eq!(
            *controller.breakpoint.lock().unwrap(),
            Some((
                7,
                step,
                evanalyzer_cfg::core_types::BreakpointMode::Snapshot
            )),
            "mode 2 must map to Snapshot"
        );

        ui.global::<PipelinesPanelState>().invoke_clear_breakpoint();
        assert!(controller.breakpoint.lock().unwrap().is_none());
    }

    #[test]
    fn attach_callbacks_breakpoint_image_and_view_mode_forward_to_the_viewport_controller() {
        let (ui, _results_ui) = test_ui_windows();
        let (_ui_state, controller) = make_controller(ui.as_weak());
        controller.attach_callbacks();

        ui.global::<PipelinesPanelState>()
            .invoke_show_breakpoint_image_changed(true);
        assert!(
            controller
                .viewport_controller
                .show_breakpoint
                .load(std::sync::atomic::Ordering::Relaxed)
        );

        ui.global::<PipelinesPanelState>()
            .invoke_breakpoint_view_mode_changed(1);
        assert_eq!(
            controller.viewport_controller.breakpoint_view_mode(),
            crate::editor::viewport_controller::BreakpointViewMode::Segmentation
        );
    }

    #[test]
    fn attach_callbacks_toggle_pipeline_flips_enabled_and_resyncs() {
        let (ui, _results_ui) = test_ui_windows();
        let (ui_state, controller) = make_controller(ui.as_weak());
        add_pipeline(&ui_state, 1);
        controller.attach_callbacks();

        ui.global::<PipelinesPanelState>().invoke_toggle_pipeline(1);

        let project = ui_state.get_project();
        assert!(
            !project
                .pipelines
                .iter()
                .find(|p| p.id.0 == 1)
                .unwrap()
                .enabled
        );
    }

    #[test]
    fn attach_callbacks_move_pipeline_up_and_down_reorder_the_list() {
        let (ui, _results_ui) = test_ui_windows();
        let (ui_state, controller) = make_controller(ui.as_weak());
        add_pipeline(&ui_state, 1);
        add_pipeline(&ui_state, 2);
        controller.attach_callbacks();

        ui.global::<PipelinesPanelState>()
            .invoke_move_pipeline_up(2);
        {
            let project = ui_state.get_project();
            assert_eq!(project.pipelines[0].id, PipelineId(2));
        }

        ui.global::<PipelinesPanelState>()
            .invoke_move_pipeline_down(2);
        let project = ui_state.get_project();
        assert_eq!(project.pipelines[1].id, PipelineId(2));
    }

    #[test]
    fn attach_callbacks_new_pipeline_adds_a_pipeline_with_the_next_free_id() {
        let (ui, _results_ui) = test_ui_windows();
        let (ui_state, controller) = make_controller(ui.as_weak());
        add_pipeline(&ui_state, 5);
        controller.attach_callbacks();

        ui.global::<PipelinesPanelState>().invoke_new_pipeline();

        let project = ui_state.get_project();
        assert_eq!(project.pipelines.len(), 2);
        assert!(project.pipelines.iter().any(|p| p.id == PipelineId(6)));
    }

    // -- callbacks + UI sync (via the test UI queue) ---------------------------

    use crate::helper::ui_thread::drain_ui_queue;
    use evanalyzer_cfg::settings::pipeline_command::{all_command_meta, default_command};

    fn step(command_name: &str) -> PipelineStepSettings {
        let meta = all_command_meta()
            .into_iter()
            .find(|m| m.name == command_name)
            .unwrap_or_else(|| panic!("no command {command_name}"));
        PipelineStepSettings {
            id: evanalyzer_cfg::core_types::StepId::UNASSIGNED,
            enabled: true,
            content: evanalyzer_cfg::settings::pipeline_settings::StepContent::Command(
                default_command(meta.id).unwrap(),
            ),
        }
    }

    fn add_pipeline_with(
        ui_state: &UiState,
        id: u32,
        source: ImageAddress,
        steps: Vec<PipelineStepSettings>,
    ) {
        ui_state.get_project_write().add_pipeline(PipelineSettings {
            id: PipelineId(id),
            name: format!("Pipeline {id}"),
            description: None,
            image_source: source,
            enabled: true,
            steps,
        });
    }

    /// A window with the controller attached and pipeline `id` selected
    /// (steps synced into `active_commands`), the queue drained.
    fn selected(
        steps: Vec<PipelineStepSettings>,
    ) -> (AppWindow, Arc<UiState>, Arc<PipelinesController>) {
        let (ui, _results_ui) = test_ui_windows();
        let (ui_state, controller) = make_controller(ui.as_weak());
        add_pipeline_with(&ui_state, 1, ImageAddress::Channel(0), steps);
        controller.attach_callbacks();
        controller.sync_pipelines_to_slint();
        let panel = ui.global::<PipelinesPanelState>();
        panel.set_active_pipeline_id(1);
        panel.invoke_select_pipeline(1);
        drain_ui_queue();
        (ui, ui_state, controller)
    }

    fn step_names(ui_state: &UiState, id: u32) -> Vec<String> {
        let project = ui_state.get_project();
        project
            .pipelines
            .iter()
            .find(|p| p.id.0 == id)
            .unwrap()
            .steps
            .iter()
            .map(|s| s.command().unwrap().name().to_string())
            .collect()
    }

    fn active_dialog(ui: &AppWindow) -> DialogType {
        ui.global::<GlobalAppState>().get_active_dialog()
    }

    // ---- advanced settings ----

    fn def(name: &str, value: &str, default: &str, advanced: bool) -> ParameterDef {
        ParameterDef {
            name: name.into(),
            display_name: name.into(),
            description: String::new(),
            value: value.into(),
            param_type: CfgParamType::Number,
            options: vec![],
            min: 0.0,
            max: 0.0,
            step: 1.0,
            groups: vec![],
            default_value: default.into(),
            advanced,
            option_advanced: vec![],
            section: String::new(),
        }
    }

    #[test]
    fn advanced_stats_count_advanced_settings_and_changed_ones_also_in_list_entries() {
        let mut group = def("thresholds", "", "", false);
        group.groups = vec![vec![
            def("unit", "%", "bit", true),
            def("method", "Li", "Manual", false),
        ]];
        let params = vec![
            def("min_area", "5", "0", false), // basic: not counted
            def("max_feret", "9", "9", true), // advanced, at default
            def("min_feret", "3", "0", true), // advanced, changed
            def("unknown", "3", "", true),    // no default known: not "changed"
            group,
        ];
        assert_eq!(advanced_stats(&params), (4, 2));
    }

    #[test]
    fn the_short_description_is_the_first_paragraph_on_one_line() {
        assert_eq!(
            short_description("The radius\nin pixels.\n\nLarger values are slower."),
            "The radius in pixels."
        );
        assert_eq!(short_description("\n\nOnly this."), "Only this.");
        assert_eq!(short_description(""), "");
    }

    #[test]
    fn settings_are_ordered_by_section_with_a_header_where_each_starts() {
        let in_section = |name: &str, section: &str| ParameterDef {
            section: section.into(),
            ..def(name, "", "", false)
        };
        let ordered = order_by_section(vec![
            in_section("radius", "Detection"),
            in_section("channel", ""),
            in_section("min_area", "Filtering"),
            in_section("sigma", "Detection"),
            in_section("output", ""),
        ]);
        let names: Vec<&str> = ordered.iter().map(|p| p.name.as_str()).collect();
        // No section first, then each in the order it first appears; the
        // settings of a section keep their order.
        assert_eq!(names, ["channel", "output", "radius", "sigma", "min_area"]);
        assert_eq!(
            section_headers(&ordered),
            ["", "", "Detection", "", "Filtering"]
        );
        // Without sections: no headers, order unchanged.
        let plain = order_by_section(vec![def("a", "", "", false), def("b", "", "", false)]);
        assert_eq!(section_headers(&plain), ["", ""]);
        assert_eq!(plain[0].name, "a");
    }

    #[test]
    fn advanced_dropdown_options_count_toward_the_advanced_row() {
        // A step whose only advanced items are two dropdown options still
        // gets the row - otherwise those options could never be shown.
        let mut dropdown = def("method", "Common", "Common", false);
        dropdown.options = vec!["Common".into(), "Rare".into(), "Exotic".into()];
        dropdown.option_advanced = vec![false, true, true];
        assert_eq!(advanced_stats(&[dropdown.clone()]), (2, 0));

        // Selecting one doesn't make it "changed": it stays visible.
        dropdown.value = "Rare".into();
        assert_eq!(advanced_stats(&[dropdown]), (2, 0));
    }

    #[test]
    fn collapsed_display_drops_advanced_entry_fields_and_options_but_keeps_the_selected_one() {
        let mut dropdown = def("method", "Rare", "Common", false);
        dropdown.options = vec!["Common".into(), "Rare".into(), "Exotic".into()];
        dropdown.option_advanced = vec![false, true, true];
        let mut group = def("thresholds", "", "", false);
        group.groups = vec![vec![
            def("unit", "bit", "bit", true),
            def("min", "0", "0", false),
        ]];

        let collapsed = filter_for_display(vec![dropdown.clone(), group.clone()], false);
        assert_eq!(
            collapsed[0].options,
            ["Common", "Rare"],
            "selected advanced option stays"
        );
        let fields: Vec<&str> = collapsed[1].groups[0]
            .iter()
            .map(|f| f.name.as_str())
            .collect();
        assert_eq!(fields, ["min"]);

        let expanded = filter_for_display(vec![dropdown, group], true);
        assert_eq!(expanded[0].options.len(), 3);
        assert_eq!(expanded[1].groups[0].len(), 2);
    }

    fn ui_names(list: &ModelRc<CommandParameter>) -> Vec<String> {
        (0..list.row_count())
            .filter_map(|i| list.row_data(i))
            .map(|p| p.name.to_string())
            .collect()
    }

    /// The (basic, advanced) top-level setting names a command declares - so
    /// these tests follow whatever split the command currently uses.
    fn declared_split(command_name: &str) -> (Vec<String>, Vec<String>) {
        let params = step(command_name).command().unwrap().to_parameters();
        let names = |advanced: bool| {
            params
                .iter()
                .filter(|p| p.advanced == advanced)
                .map(|p| p.name.clone())
                .collect::<Vec<_>>()
        };
        (names(false), names(true))
    }

    #[test]
    fn a_step_shows_its_basic_settings_and_counts_the_advanced_ones() {
        let (ui, _ui_state, _controller) = selected(vec![step("ClassifyObjects")]);
        let cmd = ui
            .global::<PipelinesPanelState>()
            .get_active_commands()
            .row_data(0)
            .unwrap();
        let (basic, advanced) = declared_split("ClassifyObjects");
        assert!(!advanced.is_empty());
        assert_eq!(ui_names(&cmd.parameters), basic);
        assert_eq!(cmd.advanced_parameters.row_count(), 0, "collapsed");
        let expected = advanced_stats(&step("ClassifyObjects").command().unwrap().to_parameters());
        assert_eq!(
            (cmd.advanced_count as usize, cmd.advanced_changed as usize),
            expected
        );
        assert_eq!(expected.1, 0, "a fresh step has nothing changed");
    }

    #[test]
    fn changing_a_hidden_advanced_setting_shows_up_as_changed() {
        let (ui, _ui_state, _controller) = selected(vec![step("ClassifyObjects")]);
        let panel = ui.global::<PipelinesPanelState>();
        // An advanced setting (solidity, kept advanced) set off its default.
        let (_, advanced) = declared_split("ClassifyObjects");
        assert!(advanced.contains(&"min_solidity".to_string()));
        panel.invoke_param_changed(0, "min_solidity".into(), "0.5".into());
        assert_eq!(
            panel
                .get_active_commands()
                .row_data(0)
                .unwrap()
                .advanced_changed,
            1
        );
    }

    #[test]
    fn expanding_shows_the_advanced_settings_below_and_always_show_applies_to_every_step() {
        let (ui, _ui_state, _controller) =
            selected(vec![step("ClassifyObjects"), step("Colocalization")]);
        let panel = ui.global::<PipelinesPanelState>();
        let commands = || panel.get_active_commands();

        panel.invoke_toggle_advanced(commands().row_data(0).unwrap().id);
        drain_ui_queue();
        let classify = commands().row_data(0).unwrap();
        assert!(classify.show_advanced);
        let (basic, advanced) = declared_split("ClassifyObjects");
        assert_eq!(ui_names(&classify.advanced_parameters), advanced);
        assert_eq!(
            ui_names(&classify.parameters),
            basic,
            "basic settings stay on top"
        );
        assert_eq!(
            commands()
                .row_data(1)
                .unwrap()
                .advanced_parameters
                .row_count(),
            0
        );

        panel.invoke_always_show_advanced_toggled(true);
        drain_ui_queue();
        assert!(panel.get_always_show_advanced());
        assert!(
            _ui_state.load_app_settings().always_show_advanced_settings,
            "remembered in the user settings"
        );
        let coloc = commands().row_data(1).unwrap();
        assert_eq!(
            ui_names(&coloc.advanced_parameters),
            ["multiplicity", "exclude_classes"]
        );
    }

    #[test]
    fn advanced_fields_inside_threshold_entries_follow_the_section() {
        let mut threshold = step("Threshold");
        if let PipelineCommand::Threshold(s) = threshold.command_mut().unwrap() {
            s.thresholds.push(Default::default());
        }
        let (ui, _ui_state, _controller) = selected(vec![threshold]);
        let panel = ui.global::<PipelinesPanelState>();
        let entry_fields = || {
            let p = ui_param(&ui, 0, "thresholds");
            let fields = p.group_items.row_data(0).unwrap().fields;
            (0..fields.row_count())
                .filter_map(|i| fields.row_data(i))
                .map(|f| f.name.to_string())
                .collect::<Vec<_>>()
        };
        assert!(!entry_fields().contains(&"unit".to_string()));
        assert!(!entry_fields().contains(&"value_source".to_string()));
        assert_eq!(
            panel
                .get_active_commands()
                .row_data(0)
                .unwrap()
                .advanced_count,
            2
        );

        panel.invoke_toggle_advanced(panel.get_active_commands().row_data(0).unwrap().id);
        drain_ui_queue();
        assert!(entry_fields().contains(&"unit".to_string()));
        assert!(entry_fields().contains(&"value_source".to_string()));
    }

    fn ui_param(ui: &AppWindow, step_idx: usize, name: &str) -> CommandParameter {
        let commands = ui.global::<PipelinesPanelState>().get_active_commands();
        let cmd = commands.row_data(step_idx).expect("step in UI model");
        let params = cmd.parameters.clone();
        (0..params.row_count())
            .filter_map(|i| params.row_data(i))
            .find(|p| p.name.as_str() == name)
            .unwrap_or_else(|| panic!("no UI parameter {name}"))
    }

    #[test]
    fn selecting_a_pipeline_pushes_its_steps_name_and_counts_to_the_ui() {
        let (ui, _ui_state, _controller) =
            selected(vec![step("Threshold"), step("Colocalization")]);
        let panel = ui.global::<PipelinesPanelState>();
        assert_eq!(panel.get_active_pipeline_name(), "Pipeline 1");
        assert_eq!(panel.get_active_pipeline_image_source(), "Channel 0");
        let commands = panel.get_active_commands();
        assert_eq!(commands.row_count(), 2);
        assert_eq!(commands.row_data(1).unwrap().name, "Colocalization");
        assert_eq!(panel.get_total_enabled_steps(), 2);
        let pipelines = panel.get_pipelines();
        assert_eq!(pipelines.row_count(), 1);
        assert_eq!(pipelines.row_data(0).unwrap().total_step_count, 2);
        assert_eq!(panel.get_enabled_pipeline_count(), 1);
    }

    #[test]
    fn selecting_an_unknown_pipeline_shows_an_empty_step_list() {
        let (ui, _ui_state, _controller) = selected(vec![step("Threshold")]);
        ui.global::<PipelinesPanelState>()
            .invoke_select_pipeline(99);
        drain_ui_queue();
        let panel = ui.global::<PipelinesPanelState>();
        assert_eq!(panel.get_active_commands().row_count(), 0);
        assert_eq!(panel.get_active_pipeline_name(), "");
    }

    #[test]
    fn pipeline_list_shows_each_image_source_kind() {
        let (ui, _results_ui) = test_ui_windows();
        let (ui_state, controller) = make_controller(ui.as_weak());
        add_pipeline_with(&ui_state, 1, ImageAddress::Scratchpad, vec![]);
        add_pipeline_with(
            &ui_state,
            2,
            ImageAddress::Memory(MemoryId::PipelineContext(3)),
            vec![],
        );
        add_pipeline_with(
            &ui_state,
            3,
            ImageAddress::Memory(MemoryId::ProjectCache(4)),
            vec![],
        );
        add_pipeline_with(&ui_state, 4, ImageAddress::Channel(2), vec![]);
        controller.sync_pipelines_to_slint();
        drain_ui_queue();
        let pipelines = ui.global::<PipelinesPanelState>().get_pipelines();
        let sources: Vec<String> = (0..pipelines.row_count())
            .map(|i| pipelines.row_data(i).unwrap().image_source.to_string())
            .collect();
        assert_eq!(
            sources,
            // A start from a memory slot reads the shared one, either kind.
            ["Scratchpad", "Shared[3]", "Shared[4]", "Channel 2"]
        );
    }

    #[test]
    fn new_pipeline_opens_the_edit_dialog_prefilled_for_it() {
        let (ui, _ui_state, _controller) = selected(vec![]);
        ui.global::<PipelinesPanelState>().invoke_new_pipeline();
        drain_ui_queue();
        let edit = ui.global::<PipelineEditState>();
        assert_eq!(edit.get_pipeline_id(), 2);
        assert_eq!(edit.get_pipeline_name(), "Pipeline 2");
        assert_eq!(edit.get_source_type(), 2, "channel source");
        assert_eq!(
            ui.global::<PipelinesPanelState>().get_active_pipeline_id(),
            2
        );
        assert_eq!(active_dialog(&ui), DialogType::PipelineEdit);
    }

    #[test]
    fn pipeline_more_fills_the_edit_dialog_for_every_source_kind() {
        let (ui, _results_ui) = test_ui_windows();
        let (ui_state, controller) = make_controller(ui.as_weak());
        add_pipeline_with(&ui_state, 1, ImageAddress::Scratchpad, vec![]);
        add_pipeline_with(
            &ui_state,
            2,
            ImageAddress::Memory(MemoryId::PipelineContext(3)),
            vec![],
        );
        add_pipeline_with(
            &ui_state,
            3,
            ImageAddress::Memory(MemoryId::ProjectCache(4)),
            vec![],
        );
        add_pipeline_with(&ui_state, 4, ImageAddress::Channel(2), vec![]);
        controller.attach_callbacks();
        let edit = ui.global::<PipelineEditState>();
        let panel = ui.global::<PipelinesPanelState>();
        for (id, expected) in [
            (1, (0, 1, 0)),
            (2, (1, 3, 0)),
            (3, (1, 4, 0)),
            (4, (2, 1, 2)),
        ] {
            panel.invoke_pipeline_more(id);
            assert_eq!(
                (
                    edit.get_source_type(),
                    edit.get_source_slot(),
                    edit.get_source_channel()
                ),
                expected,
                "pipeline {id}"
            );
            assert_eq!(edit.get_pipeline_name(), format!("Pipeline {id}").as_str());
        }
        assert_eq!(active_dialog(&ui), DialogType::PipelineEdit);

        // Unknown pipeline: dialog state untouched.
        edit.set_pipeline_id(-5);
        panel.invoke_pipeline_more(42);
        assert_eq!(edit.get_pipeline_id(), -5);
    }

    #[test]
    fn edit_dialog_confirm_renames_and_changes_the_source() {
        let (ui, ui_state, _controller) = selected(vec![]);
        let edit = ui.global::<PipelineEditState>();
        for (stype, slot, channel, expected, shown) in [
            (0, 1, 0, ImageAddress::Scratchpad, "Scratchpad"),
            (
                1,
                0,
                0,
                ImageAddress::Memory(MemoryId::ProjectCache(1)),
                "Shared[1]",
            ),
            (2, 1, 3, ImageAddress::Channel(3), "Channel 3"),
            (3, 1, 0, ImageAddress::None, "None"),
        ] {
            ui.global::<GlobalAppState>()
                .set_active_dialog(DialogType::PipelineEdit);
            edit.set_pipeline_id(1);
            edit.set_pipeline_name("Nuclei".into());
            edit.set_source_type(stype);
            edit.set_source_slot(slot);
            edit.set_source_channel(channel);
            edit.invoke_confirm();
            let project = ui_state.get_project();
            assert_eq!(project.pipelines[0].name, "Nuclei");
            assert_eq!(project.pipelines[0].image_source, expected);
            drop(project);
            // It's the active pipeline: the "EDITING" bar follows at once.
            let panel = ui.global::<PipelinesPanelState>();
            assert_eq!(panel.get_active_pipeline_name(), "Nuclei");
            assert_eq!(panel.get_active_pipeline_image_source(), shown);
            assert_eq!(active_dialog(&ui), DialogType::None);
        }
        assert!(ui_state.is_dirty());
    }

    #[test]
    fn edit_dialog_confirm_of_another_pipeline_leaves_the_editing_bar_alone() {
        let (ui, ui_state, _controller) = selected(vec![]);
        add_pipeline_with(&ui_state, 2, ImageAddress::Channel(0), vec![]);
        let edit = ui.global::<PipelineEditState>();
        edit.set_pipeline_id(2);
        edit.set_pipeline_name("".into());
        edit.set_source_type(7); // unknown kind: default source
        edit.invoke_confirm();
        assert_eq!(
            ui_state.get_project().pipelines[1].image_source,
            ImageAddress::default()
        );
        assert_eq!(
            ui.global::<PipelinesPanelState>()
                .get_active_pipeline_name(),
            "Pipeline 1"
        );
    }

    #[test]
    fn dialog_cancel_buttons_just_close_their_dialog() {
        let (ui, ui_state, _controller) = selected(vec![step("Threshold")]);
        let close = |open: DialogType, cancel: &dyn Fn()| {
            ui.global::<GlobalAppState>().set_active_dialog(open);
            cancel();
            assert_eq!(active_dialog(&ui), DialogType::None, "{open:?}");
        };
        close(DialogType::PipelineEdit, &|| {
            ui.global::<PipelineEditState>().invoke_cancel()
        });
        close(DialogType::PipelineDeleteConfirm, &|| {
            ui.global::<PipelineDeleteConfirmState>().invoke_cancel()
        });
        close(DialogType::CommandSelectionDialog, &|| {
            ui.global::<CommandPickerState>().invoke_cancel()
        });
        close(DialogType::RunAnalysisConfirm, &|| {
            ui.global::<RunAnalysisState>().invoke_cancel()
        });
        assert_eq!(step_names(&ui_state, 1), ["Threshold"], "nothing changed");
    }

    #[test]
    fn a_steps_card_state_and_breakpoint_stay_with_it_when_a_step_above_is_removed() {
        let (ui, ui_state, _controller) = selected(vec![step("Threshold"), step("Colocalization")]);
        let panel = ui.global::<PipelinesPanelState>();
        let second = ui_state.get_project().pipelines[0].steps[1].id;
        // Expand the second step and put the breakpoint on it.
        let commands = panel.get_active_commands();
        let mut cmd = commands.row_data(1).unwrap();
        assert_eq!(cmd.step_id, step_id_text(second));
        cmd.expanded = true;
        commands.set_row_data(1, cmd);
        panel.set_breakpoint_step_id(step_id_text(second));

        panel.invoke_remove_step(0);
        drain_ui_queue();

        let first = panel.get_active_commands().row_data(0).unwrap();
        assert_eq!(first.step_id, step_id_text(second), "it moved up");
        assert!(first.expanded, "and is still expanded");
        assert_eq!(panel.get_breakpoint_step_id(), first.step_id);
    }

    #[test]
    fn removing_the_breakpoints_step_clears_the_breakpoint() {
        let (ui, ui_state, controller) = selected(vec![step("Threshold"), step("Colocalization")]);
        let panel = ui.global::<PipelinesPanelState>();
        let last = ui_state.get_project().pipelines[0].steps[1].id;
        panel.invoke_set_breakpoint(1, step_id_text(last), 1);
        panel.set_breakpoint_step_id(step_id_text(last));

        // The highest number is free again - a new step may get it.
        panel.invoke_remove_step(1);
        drain_ui_queue();

        assert!(controller.breakpoint.lock().unwrap().is_none());
        assert_eq!(panel.get_breakpoint_step_id(), "");
    }

    #[test]
    fn step_ids_go_to_slint_and_back_unchanged() {
        let id = evanalyzer_cfg::core_types::StepId::UNASSIGNED;
        assert_eq!(parse_step_id(&step_id_text(id)), Some(id));
        assert_eq!(parse_step_id(""), None);
    }

    #[test]
    fn a_component_step_shows_as_a_card_with_the_components_name() {
        let component = PipelineStepSettings {
            id: evanalyzer_cfg::core_types::StepId::UNASSIGNED,
            enabled: true,
            content: StepContent::Component(
                evanalyzer_cfg::settings::components::ComponentInstance {
                    key: "CELL_RECONSTRUCTION".into(),
                    values: Default::default(),
                },
            ),
        };
        let (ui, ui_state, controller) = selected(vec![step("Threshold"), component]);
        ui_state.get_project_write().components.push(
            evanalyzer_cfg::settings::components::ComponentDefinition {
                key: "CELL_RECONSTRUCTION".into(),
                meta: evanalyzer_cfg::settings::meta_data::MetaData {
                    name: "Cell reconstruction".into(),
                    ..Default::default()
                },
                classes: vec![],
                steps: Vec::new(),
                parameters: Vec::new(),
                origin: None,
            },
        );
        controller.sync_steps_of_selected_pipeline_to_slint(PipelineId(1), true);
        drain_ui_queue();

        let card = ui
            .global::<PipelinesPanelState>()
            .get_active_commands()
            .row_data(1)
            .unwrap();
        assert_eq!(card.name, "Cell reconstruction");
        assert_eq!(card.summary, "Component");
        assert_eq!(card.parameters.row_count(), 0);
    }

    #[test]
    fn a_component_card_shows_the_components_parameters_and_edits_the_instance() {
        use evanalyzer_cfg::settings::components::{
            ComponentDefinition, ComponentInstance, ComponentParameterDef, ParamBinding, ParamValue,
        };
        use evanalyzer_cfg::settings::pipeline_command_settings::BlurSettings;
        let component = PipelineStepSettings {
            id: evanalyzer_cfg::core_types::StepId(2),
            enabled: true,
            content: StepContent::Component(ComponentInstance {
                key: "SMOOTH".into(),
                values: Default::default(),
            }),
        };
        let (ui, ui_state, controller) = selected(vec![component]);
        ui_state
            .get_project_write()
            .components
            .push(ComponentDefinition {
                key: "SMOOTH".into(),
                meta: Default::default(),
                classes: vec![],
                steps: vec![PipelineStepSettings {
                    id: evanalyzer_cfg::core_types::StepId(1),
                    enabled: true,
                    content: StepContent::Command(PipelineCommand::Blur(BlurSettings::default())),
                }],
                parameters: vec![ComponentParameterDef {
                    key: ComponentParamKey::from("1"),
                    label: "Smoothing".into(),
                    description: String::new(),
                    section: String::new(),
                    default: ParamValue::Int(3),
                    bindings: vec![ParamBinding {
                        step: evanalyzer_cfg::core_types::StepId(1),
                        param: "kernel_size".into(),
                    }],
                }],
                origin: None,
            });
        controller.sync_steps_of_selected_pipeline_to_slint(PipelineId(1), true);
        drain_ui_queue();

        let card = |ui: &AppWindow| {
            ui.global::<PipelinesPanelState>()
                .get_active_commands()
                .row_data(0)
                .unwrap()
        };
        let parameter = card(&ui).parameters.row_data(0).unwrap();
        assert_eq!(parameter.display_name, "Smoothing");
        assert_eq!(parameter.value, "3");

        ui.global::<PipelinesPanelState>()
            .invoke_param_changed(0, "1".into(), "5".into());
        drain_ui_queue();
        let StepContent::Component(instance) =
            &ui_state.get_project().pipelines[0].steps[0].content
        else {
            panic!("still a component");
        };
        assert_eq!(
            instance.values[&ComponentParamKey::from("1")],
            ParamValue::Int(5)
        );
        assert_eq!(card(&ui).parameters.row_data(0).unwrap().value, "5");
    }

    #[test]
    fn a_component_parameter_in_a_section_shows_its_header_on_the_card() {
        use evanalyzer_cfg::settings::components::{
            ComponentDefinition, ComponentInstance, ComponentParameterDef, ParamBinding, ParamValue,
        };
        use evanalyzer_cfg::settings::pipeline_command_settings::BlurSettings;
        let component = PipelineStepSettings {
            id: evanalyzer_cfg::core_types::StepId(2),
            enabled: true,
            content: StepContent::Component(ComponentInstance {
                key: "SMOOTH".into(),
                values: Default::default(),
            }),
        };
        let (ui, ui_state, controller) = selected(vec![component]);
        let parameter = |key: &str, label: &str, section: &str| ComponentParameterDef {
            key: ComponentParamKey::from(key),
            label: label.into(),
            description: String::new(),
            section: section.into(),
            default: ParamValue::Int(3),
            bindings: vec![ParamBinding {
                step: evanalyzer_cfg::core_types::StepId(1),
                param: "kernel_size".into(),
            }],
        };
        ui_state
            .get_project_write()
            .components
            .push(ComponentDefinition {
                key: "SMOOTH".into(),
                meta: Default::default(),
                classes: vec![],
                steps: vec![PipelineStepSettings {
                    id: evanalyzer_cfg::core_types::StepId(1),
                    enabled: true,
                    content: StepContent::Command(PipelineCommand::Blur(BlurSettings::default())),
                }],
                parameters: vec![
                    parameter("1", "Smoothing", "Filtering"),
                    parameter("2", "Size", ""),
                ],
                origin: None,
            });
        controller.sync_steps_of_selected_pipeline_to_slint(PipelineId(1), true);
        drain_ui_queue();

        let card = ui
            .global::<PipelinesPanelState>()
            .get_active_commands()
            .row_data(0)
            .unwrap();
        let shown: Vec<(String, String, String)> = card
            .parameters
            .iter()
            .map(|p| {
                (
                    p.display_name.to_string(),
                    p.section_header.to_string(),
                    p.section.to_string(),
                )
            })
            .collect();
        assert_eq!(
            shown,
            [
                ("Size".into(), "".into(), "".into()),
                ("Smoothing".into(), "Filtering".into(), "Filtering".into()),
            ]
        );
        let sections: Vec<String> = card.sections.iter().map(|s| s.to_string()).collect();
        assert_eq!(sections, ["", "Filtering"]);
    }

    #[test]
    fn the_expand_button_opens_the_steps_settings_dialog() {
        let (ui, _ui_state, _controller) = selected(vec![step("Threshold")]);
        let panel = ui.global::<PipelinesPanelState>();
        let step_id = panel.get_active_commands().row_data(0).unwrap().step_id;
        panel.invoke_open_step_settings(step_id.clone());
        assert_eq!(panel.get_settings_step_id(), step_id);
        assert_eq!(
            ui.global::<GlobalAppState>().get_active_dialog(),
            DialogType::StepSettings
        );
        assert!(
            panel
                .get_active_commands()
                .row_data(0)
                .unwrap()
                .show_advanced,
            "its advanced settings are there to pick and search"
        );
    }

    #[test]
    fn picking_a_component_inserts_a_step_using_it() {
        let (ui, ui_state, _controller) = selected(vec![step("Threshold")]);
        ui_state.get_project_write().components =
            vec![component_definition("CELLS", "Cell reconstruction")];
        let picker = ui.global::<CommandPickerState>();
        picker.set_pipeline_id(1);
        picker.set_insert_after_idx(0);

        picker.invoke_select(COMPONENT_PICKER_ID_BASE);
        assert_eq!(picker.get_detail().name, "Cell reconstruction");
        picker.invoke_confirm(COMPONENT_PICKER_ID_BASE);
        drain_ui_queue();

        let project = ui_state.get_project();
        let StepContent::Component(instance) = &project.pipelines[0].steps[1].content else {
            panic!("a component step");
        };
        assert_eq!(instance.key, "CELLS");
        assert!(instance.values.is_empty(), "at the component's defaults");
        drop(project);
        let card = ui
            .global::<PipelinesPanelState>()
            .get_active_commands()
            .row_data(1)
            .unwrap();
        assert_eq!(
            card.category,
            StepCategory::Component,
            "the component colour"
        );
    }

    #[test]
    fn create_component_turns_the_chosen_steps_into_a_component() {
        let (ui, ui_state, _controller) = selected(vec![
            step("Threshold"),
            step("Blur"),
            step("Blur"),
            step("Threshold"),
        ]);
        ui.global::<PipelinesPanelState>()
            .invoke_create_component(1);
        drain_ui_queue();
        assert_eq!(active_dialog(&ui), DialogType::CreateComponent);
        let state = ui.global::<CreateComponentState>();
        assert_eq!((state.get_from_step(), state.get_to_step()), (1, 1));
        assert_eq!(state.get_step_names().row_count(), 4);
        assert_eq!(state.get_step_names().row_data(1).unwrap(), "#2 Blur");

        state.invoke_range_changed(2, 1); // either order
        assert_eq!((state.get_from_step(), state.get_to_step()), (1, 2));

        // A name is needed - the dialog stays open and says so.
        state.invoke_confirm();
        assert!(
            state.get_error().contains("needs a name"),
            "{}",
            state.get_error()
        );
        assert_eq!(active_dialog(&ui), DialogType::CreateComponent);

        state.set_name("Smooth twice".into());
        state.invoke_confirm();
        drain_ui_queue();
        assert_eq!(active_dialog(&ui), DialogType::None);
        let project = ui_state.get_project();
        let steps = &project.pipelines[0].steps;
        assert_eq!(steps.len(), 3);
        assert!(matches!(&steps[1].content, StepContent::Component(g) if g.key == "SMOOTH_TWICE"));
        assert_eq!(project.components[0].steps.len(), 2);
        drop(project);
        let card = ui
            .global::<PipelinesPanelState>()
            .get_active_commands()
            .row_data(1)
            .unwrap();
        assert_eq!(card.name, "Smooth twice");
    }

    /// A pipeline whose steps 2-3 (two blurs) are made the component "Smooth".
    fn with_a_component() -> (AppWindow, Arc<UiState>, Arc<PipelinesController>) {
        let (ui, ui_state, controller) =
            selected(vec![step("Threshold"), step("Blur"), step("Blur")]);
        {
            let mut project = ui_state.get_project_write();
            evanalyzer_app::components::create_component(
                &mut project,
                PipelineId(1),
                1..=2,
                "Smooth",
                &[],
            )
            .unwrap();
        }
        controller.sync_steps_of_selected_pipeline_to_slint(PipelineId(1), true);
        drain_ui_queue();
        (ui, ui_state, controller)
    }

    fn card_names(ui: &AppWindow) -> Vec<String> {
        let model = ui.global::<PipelinesPanelState>().get_active_commands();
        (0..model.row_count())
            .map(|i| model.row_data(i).unwrap().name.to_string())
            .collect()
    }

    /// A number with − / + (a spinner setting): + steps it up by its step,
    /// − back down, and the step's setting takes the value.
    #[test]
    fn a_spinners_buttons_step_the_setting() {
        use i_slint_backend_testing::ElementHandle;
        let (ui, ui_state, _controller) = selected(vec![step("Blur")]);
        ui.window().set_size(slint::LogicalSize::new(1280.0, 900.0));
        ui.global::<GlobalAppState>().set_sidebar_left_active_tab(2);
        let commands = ui.global::<PipelinesPanelState>().get_active_commands();
        let mut card = commands.row_data(0).unwrap();
        card.expanded = true;
        commands.set_row_data(0, card);
        let size = || match ui_state.get_project().pipelines[0].steps[0].command() {
            Some(PipelineCommand::Blur(blur)) => blur.kernel_size,
            other => panic!("{other:?}"),
        };
        let before = size();
        let button = |label: &str| {
            ElementHandle::find_by_accessible_label(&ui, label)
                .next()
                .unwrap_or_else(|| panic!("{label}"))
        };
        button("Increase").invoke_accessible_default_action();
        drain_ui_queue();
        assert!(size() > before, "{} > {before}", size());
        button("Decrease").invoke_accessible_default_action();
        drain_ui_queue();
        assert_eq!(size(), before);
    }

    /// A channel and a single class are picked with the same field (label, swatch
    /// and name, "↓") - only a class list has its own (chips).
    #[test]
    fn channel_and_class_pickers_share_one_design() {
        use i_slint_backend_testing::ElementHandle;
        let (ui, _ui_state, _controller) =
            selected(vec![step("LoadChannel"), step("ClassifyObjects")]);
        ui.window()
            .set_size(slint::LogicalSize::new(1280.0, 1400.0));
        ui.global::<GlobalAppState>().set_sidebar_left_active_tab(2);
        let commands = ui.global::<PipelinesPanelState>().get_active_commands();
        for index in 0..2 {
            let mut card = commands.row_data(index).unwrap();
            card.expanded = true;
            commands.set_row_data(index, card);
        }
        let count = |label: &str| ElementHandle::find_by_accessible_label(&ui, label).count();
        assert_eq!(count("↓"), 2, "the channel and the output class field");
        assert_eq!(count("▾"), 1, "only the input class list");
    }

    /// A class list shows its selected classes as chips while they fit - not
    /// only a "+N" badge.
    #[test]
    fn a_class_lists_selected_classes_show_as_chips() {
        use i_slint_backend_testing::ElementHandle;
        let mut classify = step("ClassifyObjects");
        if let StepContent::Command(PipelineCommand::ClassifyObjects(settings)) =
            &mut classify.content
        {
            settings.input_classes = vec![
                ObjectClass::Global(1),
                ObjectClass::Global(2),
                ObjectClass::Global(3),
            ];
        }
        let (ui, _ui_state, _controller) = selected(vec![classify]);
        ui.window()
            .set_size(slint::LogicalSize::new(1280.0, 1400.0));
        ui.global::<GlobalAppState>().set_sidebar_left_active_tab(2);
        let commands = ui.global::<PipelinesPanelState>().get_active_commands();
        let mut card = commands.row_data(0).unwrap();
        card.expanded = true;
        commands.set_row_data(0, card);

        // The badge only exists while chips are left out.
        let badge = ElementHandle::find_by_accessible_label(&ui, "+3").count();
        assert_eq!(badge, 0, "the three chips fit - no \"+3\" instead");
    }

    /// A short pipeline doesn't stretch an expanded card to fill the panel -
    /// that used to put a big gap under a section header and at the card's
    /// bottom.
    #[test]
    fn an_expanded_card_in_a_short_pipeline_keeps_its_own_height() {
        use i_slint_backend_testing::ElementHandle;
        let (ui, ui_state, controller) = with_a_component();
        ui.window()
            .set_size(slint::LogicalSize::new(1280.0, 1400.0));
        ui.global::<GlobalAppState>().set_sidebar_left_active_tab(2);
        {
            let mut project = ui_state.get_project_write();
            let definition = &mut project.components[0];
            let step = definition.steps[0].id;
            evanalyzer_app::components::toggle_exposed(
                definition,
                step,
                "kernel_size",
                "Kernel",
                "3",
            );
            let key = definition.parameters[0].key.clone();
            evanalyzer_app::components::set_parameter_section(definition, &key, "Staining 1");
        }
        controller.sync_steps_of_selected_pipeline_to_slint(PipelineId(1), true);
        drain_ui_queue();
        let commands = ui.global::<PipelinesPanelState>().get_active_commands();
        let mut card = commands.row_data(1).unwrap();
        card.expanded = true;
        commands.set_row_data(1, card);
        ui.global::<PipelinesPanelState>()
            .invoke_toggle_section(1, "Staining 1".into());

        let text = |label: &str| {
            let e = ElementHandle::find_by_accessible_label(&ui, label)
                .next()
                .unwrap_or_else(|| panic!("{label:?} not found"));
            (e.absolute_position().y, e.size().height)
        };
        let (header_y, header_height) = text("Staining 1");
        let (setting_y, _) = text("Kernel");
        assert!(header_height <= 30.0, "header {header_height}px high");
        assert!(
            setting_y - header_y < 40.0,
            "the setting is {}px below its section header",
            setting_y - header_y
        );
    }

    /// "Outer" (a threshold and "Smooth", two blurs) as the pipeline's only
    /// step - "Smooth" nested in "Outer".
    fn with_a_nested_component() -> (AppWindow, Arc<UiState>, Arc<PipelinesController>) {
        let (ui, ui_state, controller) = with_a_component();
        {
            let mut project = ui_state.get_project_write();
            evanalyzer_app::components::create_component(
                &mut project,
                PipelineId(1),
                0..=1,
                "Outer",
                &[],
            )
            .unwrap();
        }
        controller.sync_steps_of_selected_pipeline_to_slint(PipelineId(1), true);
        drain_ui_queue();
        (ui, ui_state, controller)
    }

    fn smooth_kernel_size(ui_state: &UiState, key: &str) -> i32 {
        let project = ui_state.get_project();
        let definition = project.components.iter().find(|d| d.key == key).unwrap();
        let PipelineCommand::Blur(blur) = definition.steps[0].command().unwrap() else {
            panic!("blur")
        };
        blur.kernel_size as i32
    }

    #[test]
    fn a_component_opens_inside_an_opened_component_and_back_goes_one_level_up() {
        let (ui, ui_state, _controller) = with_a_nested_component();
        let panel = ui.global::<PipelinesPanelState>();
        assert_eq!(card_names(&ui), ["Outer"]);

        panel.invoke_open_component(0);
        drain_ui_queue();
        assert_eq!(panel.get_open_component_name(), "Outer");
        assert_eq!(card_names(&ui), ["Threshold", "Smooth"]);

        panel.invoke_open_component(1);
        drain_ui_queue();
        assert_eq!(panel.get_open_component_name(), "Outer › ◆ Smooth");
        assert_eq!(card_names(&ui), ["Blur", "Blur"]);

        // Edits go to the nested component's definition.
        panel.invoke_param_changed(0, "kernel_size".into(), "7".into());
        drain_ui_queue();
        assert_eq!(smooth_kernel_size(&ui_state, "SMOOTH"), 7);

        panel.invoke_close_component();
        drain_ui_queue();
        assert_eq!(panel.get_open_component_name(), "Outer");
        assert_eq!(card_names(&ui), ["Threshold", "Smooth"]);

        panel.invoke_close_component();
        drain_ui_queue();
        assert_eq!(panel.get_open_component_name(), "");
        assert_eq!(card_names(&ui), ["Outer"]);
    }

    #[test]
    fn inside_a_viewed_component_a_nested_one_is_only_viewed_and_edit_opens_the_viewed_one() {
        let (ui, _ui_state, _controller) = with_a_nested_component();
        let panel = ui.global::<PipelinesPanelState>();
        panel.invoke_view_component(0);
        drain_ui_queue();
        assert!(panel.get_open_component_read_only());

        // "Edit" on a card isn't offered while viewing - and does nothing.
        panel.invoke_open_component(1);
        drain_ui_queue();
        assert_eq!(panel.get_open_component_name(), "Outer");

        panel.invoke_view_component(1);
        drain_ui_queue();
        assert_eq!(panel.get_open_component_name(), "Outer › ◆ Smooth");
        assert!(panel.get_open_component_read_only());

        // "Edit" in the bar: the first viewed component, to edit.
        panel.invoke_edit_open_component();
        drain_ui_queue();
        assert_eq!(panel.get_open_component_name(), "Outer");
        assert!(!panel.get_open_component_read_only());
        assert_eq!(card_names(&ui), ["Threshold", "Smooth"]);
    }

    #[test]
    fn only_this_step_inside_an_opened_component_gives_its_step_a_copy() {
        let (ui, ui_state, controller) = with_a_nested_component();
        // "Smooth" is used in the pipeline too.
        {
            let mut project = ui_state.get_project_write();
            let mut smooth = project
                .components
                .iter()
                .find(|d| d.key == "OUTER")
                .unwrap()
                .steps[1]
                .clone();
            smooth.id = evanalyzer_cfg::core_types::StepId::UNASSIGNED;
            project.pipelines[0].steps.push(smooth);
            evanalyzer_app::utils::assign_step_ids(&mut project.pipelines[0].steps);
        }
        controller.sync_steps_of_selected_pipeline_to_slint(PipelineId(1), true);
        drain_ui_queue();
        let panel = ui.global::<PipelinesPanelState>();
        panel.invoke_open_component(0);
        drain_ui_queue();

        // Asked - cancelling leaves the panel as it was.
        panel.invoke_open_component(1);
        drain_ui_queue();
        assert_eq!(active_dialog(&ui), DialogType::ComponentEditChoice);
        ui.global::<ComponentEditChoiceState>().invoke_cancel();
        drain_ui_queue();
        assert_eq!(panel.get_open_component_name(), "Outer");

        panel.invoke_open_component(1);
        drain_ui_queue();
        ui.global::<ComponentEditChoiceState>().invoke_only_this();
        drain_ui_queue();
        assert_eq!(panel.get_open_component_name(), "Outer › ◆ Smooth (2)");
        panel.invoke_param_changed(0, "kernel_size".into(), "9".into());
        drain_ui_queue();
        assert_eq!(smooth_kernel_size(&ui_state, "SMOOTH_2"), 9);
        assert_ne!(
            smooth_kernel_size(&ui_state, "SMOOTH"),
            9,
            "the pipeline's use keeps the original"
        );
        let project = ui_state.get_project();
        let outer = project
            .components
            .iter()
            .find(|d| d.key == "OUTER")
            .unwrap();
        assert!(
            matches!(&outer.steps[1].content, StepContent::Component(i) if i.key == "SMOOTH_2")
        );
    }

    #[test]
    fn edit_from_the_bar_asks_and_cancelling_keeps_viewing() {
        let (ui, ui_state, controller) = with_a_nested_component();
        // "Outer" used twice: "Edit" asks.
        {
            let mut project = ui_state.get_project_write();
            let mut outer = project.pipelines[0].steps[0].clone();
            outer.id = evanalyzer_cfg::core_types::StepId::UNASSIGNED;
            project.pipelines[0].steps.push(outer);
            evanalyzer_app::utils::assign_step_ids(&mut project.pipelines[0].steps);
        }
        controller.sync_steps_of_selected_pipeline_to_slint(PipelineId(1), true);
        drain_ui_queue();
        let panel = ui.global::<PipelinesPanelState>();
        panel.invoke_view_component(0);
        panel.invoke_view_component(1);
        drain_ui_queue();

        panel.invoke_edit_open_component();
        drain_ui_queue();
        assert_eq!(active_dialog(&ui), DialogType::ComponentEditChoice);
        ui.global::<ComponentEditChoiceState>().invoke_cancel();
        drain_ui_queue();
        assert_eq!(panel.get_open_component_name(), "Outer › ◆ Smooth");
        assert!(panel.get_open_component_read_only());

        panel.invoke_edit_open_component();
        drain_ui_queue();
        ui.global::<ComponentEditChoiceState>().invoke_edit_all();
        drain_ui_queue();
        assert_eq!(panel.get_open_component_name(), "Outer");
        assert!(!panel.get_open_component_read_only());
    }

    #[test]
    fn inside_an_opened_component_the_picker_leaves_out_the_components_containing_it() {
        let (ui, ui_state, controller) = with_a_nested_component();
        let panel = ui.global::<PipelinesPanelState>();
        let picker = ui.global::<CommandPickerState>();
        let shown = |controller: &Arc<PipelinesController>| {
            picker.set_pipeline_id(1);
            picker.set_filter_components(true);
            controller.apply_picker_filter(&ui, "");
            let model = picker.get_shown_components();
            (0..model.row_count())
                .map(|i| model.row_data(i).unwrap().name.to_string())
                .collect::<Vec<_>>()
        };
        assert_eq!(shown(&controller), ["Smooth", "Outer"]);

        panel.invoke_open_component(0);
        drain_ui_queue();
        assert_eq!(shown(&controller), ["Smooth"], "Outer would contain itself");

        panel.invoke_open_component(1);
        drain_ui_queue();
        assert!(
            shown(&controller).is_empty(),
            "Outer contains Smooth through itself"
        );

        // Picked anyway (e.g. a stale list): refused, with a message.
        let outer = ui_state
            .get_project()
            .components
            .iter()
            .position(|d| d.key == "OUTER")
            .unwrap() as i32;
        picker.set_insert_after_idx(0);
        picker.invoke_confirm(COMPONENT_PICKER_ID_BASE + outer);
        drain_ui_queue();
        assert_eq!(active_dialog(&ui), DialogType::Warning);
        assert!(
            ui.global::<WarningState>()
                .get_message()
                .contains("can't contain itself")
        );
        let project = ui_state.get_project();
        let smooth = project
            .components
            .iter()
            .find(|d| d.key == "SMOOTH")
            .unwrap();
        assert_eq!(smooth.steps.len(), 2, "nothing inserted");
    }

    #[test]
    fn create_component_inside_an_opened_component_nests_the_new_one() {
        let (ui, ui_state, _controller) = with_a_component();
        let panel = ui.global::<PipelinesPanelState>();
        panel.invoke_open_component(1);
        drain_ui_queue();
        // The first blur's kernel size is a parameter of "Smooth".
        panel.invoke_toggle_exposed(0, "kernel_size".into());
        drain_ui_queue();

        panel.invoke_create_component(0);
        drain_ui_queue();
        assert_eq!(active_dialog(&ui), DialogType::CreateComponent);
        let state = ui.global::<CreateComponentState>();
        assert_eq!(
            state.get_step_names().row_count(),
            2,
            "the component's steps"
        );
        assert_eq!(
            state.get_step_names().row_data(0).unwrap(),
            "#2 Blur",
            "by its id, as its card shows it"
        );
        state.set_name("Soften".into());
        state.invoke_confirm();
        drain_ui_queue();
        assert_eq!(
            active_dialog(&ui),
            DialogType::None,
            "{}",
            state.get_error()
        );

        assert_eq!(panel.get_open_component_name(), "Smooth");
        assert_eq!(card_names(&ui), ["Soften", "Blur"]);
        {
            let project = ui_state.get_project();
            let soften = project
                .components
                .iter()
                .find(|d| d.key == "SOFTEN")
                .unwrap();
            assert_eq!(soften.steps.len(), 1);
            assert_eq!(soften.parameters.len(), 1, "the exposed setting");
        }

        // The pipeline's card still shows - and sets - the kernel size.
        panel.invoke_close_component();
        drain_ui_queue();
        let card = panel.get_active_commands().row_data(1).unwrap();
        assert_eq!(card.parameters.row_count(), 1);
        let name = card.parameters.row_data(0).unwrap().name.to_string();
        panel.invoke_param_changed(1, name.into(), "9".into());
        drain_ui_queue();
        let project = ui_state.get_project();
        let with_values = |step: &PipelineStepSettings| {
            let StepContent::Component(instance) = &step.content else {
                panic!("a component step")
            };
            evanalyzer_app::components::component_steps_with_values(instance, &project.components)
                .unwrap()
        };
        let smooth = with_values(&project.pipelines[0].steps[1]);
        let soften = with_values(&smooth[0]);
        let Some(PipelineCommand::Blur(blur)) = soften[0].command() else {
            panic!("blur")
        };
        assert_eq!(blur.kernel_size, 9);
    }

    #[test]
    fn inside_an_opened_component_class_settings_offer_its_own_classes() {
        let (ui, ui_state, controller) = with_a_nested_component();
        {
            let mut project = ui_state.get_project_write();
            let outer = project
                .components
                .iter_mut()
                .find(|d| d.key == "OUTER")
                .unwrap();
            outer
                .classes
                .push(evanalyzer_cfg::settings::classification_settings::Class {
                    id: 1,
                    name: "Vesicles".into(),
                    color: 0x00FF00,
                    notes: String::new(),
                });
            let mut classify = step("ClassifyObjects");
            classify.id = evanalyzer_cfg::core_types::StepId(9);
            outer.steps.push(classify);
        }
        controller.sync_steps_of_selected_pipeline_to_slint(PipelineId(1), true);
        drain_ui_queue();
        let panel = ui.global::<PipelinesPanelState>();
        let offered = || ui.global::<ComponentClassesState>().get_classes();
        assert_eq!(offered().row_count(), 0, "in the pipeline: none");

        panel.invoke_open_component(0);
        drain_ui_queue();
        assert_eq!(offered().row_count(), 1);
        assert_eq!(offered().row_data(0).unwrap().name, "Vesicles");

        panel.invoke_param_changed(2, "input_classes".into(), "toggle:L1".into());
        drain_ui_queue();
        {
            let project = ui_state.get_project();
            let outer = project
                .components
                .iter()
                .find(|d| d.key == "OUTER")
                .unwrap();
            let Some(PipelineCommand::ClassifyObjects(classify)) = outer.steps[2].command() else {
                panic!("classify")
            };
            assert_eq!(classify.input_classes, [ObjectClass::Local(1)]);
        }
        let flags = ui_param(&ui, 2, "input_classes").options;
        assert_eq!(flags.row_count(), 34, "the project's 33, then Vesicles");
        assert_eq!(flags.row_data(33).unwrap(), "1", "Vesicles is selected");

        // Back in the pipeline: none offered again.
        panel.invoke_close_component();
        drain_ui_queue();
        assert_eq!(offered().row_count(), 0);
    }

    #[test]
    fn expand_inside_an_opened_component_puts_the_nested_steps_there() {
        let (ui, ui_state, _controller) = with_a_nested_component();
        let panel = ui.global::<PipelinesPanelState>();
        panel.invoke_open_component(0);
        drain_ui_queue();
        assert_eq!(card_names(&ui), ["Threshold", "Smooth"]);

        panel.invoke_expand_component(1);
        drain_ui_queue();
        assert_eq!(active_dialog(&ui), DialogType::None);
        assert_eq!(panel.get_open_component_name(), "Outer");
        assert_eq!(card_names(&ui), ["Threshold", "Blur", "Blur"]);
        let project = ui_state.get_project();
        let outer = project
            .components
            .iter()
            .find(|d| d.key == "OUTER")
            .unwrap();
        assert_eq!(outer.steps.len(), 3);
        assert_eq!(card_names(&ui).len(), 3);
        assert_eq!(
            project.pipelines[0].steps.len(),
            1,
            "the pipeline unchanged"
        );
    }

    #[test]
    fn a_nested_components_parameter_is_exposed_on_the_component_around_it() {
        let (ui, ui_state, controller) = with_a_nested_component();
        // "Smooth" shows its first blur's kernel size as "Kernel".
        {
            let mut project = ui_state.get_project_write();
            let smooth = project
                .components
                .iter_mut()
                .find(|d| d.key == "SMOOTH")
                .unwrap();
            let blur = smooth.steps[0].id;
            evanalyzer_app::components::toggle_exposed(smooth, blur, "kernel_size", "Kernel", "3");
        }
        controller.sync_steps_of_selected_pipeline_to_slint(PipelineId(1), true);
        drain_ui_queue();
        let panel = ui.global::<PipelinesPanelState>();
        panel.invoke_open_component(0);
        drain_ui_queue();
        let kernel = || ui_param(&ui, 1, "kernel");
        assert!(!kernel().exposed);

        panel.invoke_toggle_exposed(1, "kernel".into());
        drain_ui_queue();
        assert!(kernel().exposed, "the card marks it");
        {
            let project = ui_state.get_project();
            let outer = project
                .components
                .iter()
                .find(|d| d.key == "OUTER")
                .unwrap();
            assert_eq!(outer.parameters.len(), 1);
            assert_eq!(outer.parameters[0].label, "Kernel");
            assert_eq!(outer.parameters[0].bindings[0].param, "kernel");
        }

        // A new value on the card becomes the parameter's default - and runs.
        panel.invoke_param_changed(1, "kernel".into(), "9".into());
        drain_ui_queue();
        let project = ui_state.get_project();
        let outer = project
            .components
            .iter()
            .find(|d| d.key == "OUTER")
            .unwrap();
        assert_eq!(outer.parameters[0].default, ParamValue::Int(9));
        let with_values = |step: &PipelineStepSettings| {
            let StepContent::Component(instance) = &step.content else {
                panic!("a component step")
            };
            evanalyzer_app::components::component_steps_with_values(instance, &project.components)
                .unwrap()
        };
        let outer_steps = with_values(&project.pipelines[0].steps[0]);
        let smooth = with_values(&outer_steps[1]);
        let Some(PipelineCommand::Blur(blur)) = smooth[0].command() else {
            panic!("blur")
        };
        assert_eq!(blur.kernel_size, 9);
    }

    /// Equal steps are told apart by their id - the number on their card - where
    /// a parameter names what it sets.
    #[test]
    fn what_a_parameter_sets_names_the_step_by_its_id() {
        let (ui, _ui_state, _controller) = with_a_component();
        let panel = ui.global::<PipelinesPanelState>();
        panel.invoke_open_component(1);
        drain_ui_queue();
        let ids: Vec<String> = (0..2)
            .map(|i| {
                panel
                    .get_active_commands()
                    .row_data(i)
                    .unwrap()
                    .step_id
                    .to_string()
            })
            .collect();
        // The first blur's kernel size is a parameter.
        panel.invoke_toggle_exposed(0, "kernel_size".into());
        drain_ui_queue();

        // The second blur's: "add to" says which blur the parameter sets.
        panel.invoke_toggle_exposed(1, "kernel_size".into());
        drain_ui_queue();
        assert_eq!(active_dialog(&ui), DialogType::ExposeChoice);
        let detail = ui
            .global::<ExposeChoiceState>()
            .get_details()
            .row_data(1)
            .unwrap();
        assert!(
            detail.starts_with(&format!("sets #{} Blur ›", ids[0])),
            "{detail}"
        );
        ui.global::<ExposeChoiceState>().invoke_chosen(1);
        drain_ui_queue();

        // The settings dialog lists both blurs, apart.
        panel.invoke_open_component_settings();
        drain_ui_queue();
        let sets = ui
            .global::<ComponentSettingsState>()
            .get_parameters()
            .row_data(0)
            .unwrap()
            .sets
            .to_string();
        assert!(
            sets.contains(&format!("#{} Blur", ids[0]))
                && sets.contains(&format!("#{} Blur", ids[1])),
            "{sets}"
        );
    }

    #[test]
    fn the_component_dialogs_cancel_buttons_just_close_their_dialog() {
        let (ui, ui_state, _controller) = with_a_component();
        let before = serde_json::to_value(&ui_state.get_project().components).unwrap();
        let cancels: [(DialogType, &dyn Fn(&AppWindow)); 4] = [
            (DialogType::ComponentEditChoice, &|ui| {
                ui.global::<ComponentEditChoiceState>().invoke_cancel()
            }),
            (DialogType::ExposeChoice, &|ui| {
                ui.global::<ExposeChoiceState>().invoke_cancel()
            }),
            (DialogType::ComponentUpdate, &|ui| {
                ui.global::<ComponentUpdateState>().invoke_cancel()
            }),
            (DialogType::CreateComponent, &|ui| {
                ui.global::<CreateComponentState>().invoke_cancel()
            }),
        ];
        for (open, cancel) in cancels {
            ui.global::<GlobalAppState>().set_active_dialog(open);
            cancel(&ui);
            drain_ui_queue();
            assert_eq!(active_dialog(&ui), DialogType::None, "{open:?}");
        }
        assert_eq!(
            serde_json::to_value(&ui_state.get_project().components).unwrap(),
            before,
            "nothing changed"
        );
    }

    /// A component opened only to look at can't be changed - whatever is
    /// clicked on its cards.
    #[test]
    fn a_component_opened_to_view_refuses_every_change() {
        let (ui, ui_state, _controller) = with_a_nested_component();
        let snapshot = || {
            let project = ui_state.get_project();
            serde_json::to_value((&project.components, &project.pipelines)).unwrap()
        };
        let panel = ui.global::<PipelinesPanelState>();
        panel.invoke_view_component(0);
        drain_ui_queue();
        assert!(panel.get_open_component_read_only());
        let before = snapshot();

        panel.invoke_expand_component(1);
        panel.invoke_toggle_exposed(0, "segmentation_class".into());
        panel.invoke_save_component_to_library(1);
        panel.invoke_update_component(1);
        panel.invoke_create_component(0);
        panel.invoke_open_component_settings();
        panel.invoke_param_changed(0, "segmentation_class".into(), "3".into());
        drain_ui_queue();

        assert_eq!(snapshot(), before);
        assert_eq!(active_dialog(&ui), DialogType::None);
        assert!(panel.get_open_component_read_only(), "still only viewed");
    }

    #[test]
    fn save_to_library_on_a_component_card_asks_for_its_metadata() {
        let (ui, _ui_state, _controller) = with_a_nested_component();
        let panel = ui.global::<PipelinesPanelState>();
        let meta = || ui.global::<crate::TemplateMetaState>();

        panel.invoke_save_component_to_library(0);
        drain_ui_queue();
        assert_eq!(meta().get_dialog_title(), "Save Component to Library");
        assert_eq!(meta().get_meta().name, "Outer");

        // A component nested in the opened one: its own name.
        panel.invoke_open_component(0);
        drain_ui_queue();
        panel.invoke_save_component_to_library(1);
        drain_ui_queue();
        assert_eq!(meta().get_meta().name, "Smooth");
    }

    #[test]
    fn a_created_component_is_authored_by_the_user() {
        let (ui, ui_state, _controller) =
            selected(vec![step("Threshold"), step("Blur"), step("Blur")]);
        ui_state.update_app_settings(|settings| {
            settings.author = Some(evanalyzer_app::global::UserInformation {
                full_name: "Ada Lovelace".into(),
                organization: "Analytical Engines".into(),
            });
        });
        ui.global::<PipelinesPanelState>()
            .invoke_create_component(1);
        drain_ui_queue();
        let state = ui.global::<CreateComponentState>();
        state.set_name("Smooth".into());
        state.invoke_confirm();
        drain_ui_queue();

        let project = ui_state.get_project();
        let author = &project.components[0].meta.authors[0];
        assert_eq!(
            (author.full_name.as_str(), author.organization.as_str()),
            ("Ada Lovelace", "Analytical Engines")
        );
    }

    /// The opened component's bar stays on top - it isn't part of the
    /// scrolling step list.
    #[test]
    fn the_opened_components_bar_stays_above_the_scrolling_steps() {
        use i_slint_backend_testing::ElementHandle;
        let (ui, _ui_state, _controller) = with_a_component();
        ui.window().set_size(slint::LogicalSize::new(1280.0, 800.0));
        ui.global::<GlobalAppState>().set_sidebar_left_active_tab(2);
        ui.global::<PipelinesPanelState>().invoke_open_component(1);
        drain_ui_queue();
        let back = ElementHandle::find_by_accessible_label(&ui, "‹ Back")
            .next()
            .expect("the bar's Back");
        let y = back.absolute_position().y;
        let inside_a_scroll_area = ElementHandle::find_by_element_type_name(&ui, "ScrollView")
            .filter(|e| e.absolute_position().x < 400.0)
            .any(|e| {
                let top = e.absolute_position().y;
                top <= y && y < top + e.size().height
            });
        assert!(
            !inside_a_scroll_area,
            "the bar doesn't scroll with the steps"
        );
    }

    #[test]
    fn edit_or_view_on_a_command_card_opens_nothing() {
        let (ui, _ui_state, _controller) = with_a_component();
        let panel = ui.global::<PipelinesPanelState>();
        panel.invoke_open_component(0);
        panel.invoke_view_component(0);
        drain_ui_queue();
        assert_eq!(panel.get_open_component_name(), "", "step 1 is a threshold");
        assert_eq!(active_dialog(&ui), DialogType::None);
    }

    #[test]
    fn removing_a_component_by_an_id_the_picker_doesnt_have_does_nothing() {
        let (ui, ui_state, _controller) = with_a_component();
        let picker = ui.global::<CommandPickerState>();
        // A command's id, a library component's, a component that isn't there.
        for id in [
            3,
            LIBRARY_COMPONENT_PICKER_ID_BASE,
            COMPONENT_PICKER_ID_BASE + 9,
        ] {
            picker.invoke_remove_component(id);
        }
        // The one there is used - it stays.
        picker.invoke_remove_component(COMPONENT_PICKER_ID_BASE);
        drain_ui_queue();
        assert_eq!(ui_state.get_project().components.len(), 1);
    }

    #[test]
    fn a_nested_component_that_is_gone_is_closed_and_its_outer_one_shown() {
        let (ui, ui_state, controller) = with_a_nested_component();
        let panel = ui.global::<PipelinesPanelState>();
        panel.invoke_open_component(0);
        panel.invoke_open_component(1);
        drain_ui_queue();
        // E.g. undone.
        ui_state
            .get_project_write()
            .components
            .retain(|d| d.key != "SMOOTH");
        controller.sync_steps_of_selected_pipeline_to_slint(PipelineId(1), false);
        drain_ui_queue();
        assert_eq!(panel.get_open_component_name(), "Outer");
    }

    #[test]
    fn an_opened_component_shows_its_steps_and_edits_its_definition() {
        let (ui, ui_state, _controller) = with_a_component();
        let panel = ui.global::<PipelinesPanelState>();
        assert_eq!(card_names(&ui), ["Threshold", "Smooth"]);

        panel.invoke_open_component(1);
        drain_ui_queue();
        assert_eq!(panel.get_open_component_name(), "Smooth");
        assert_eq!(card_names(&ui), ["Blur", "Blur"]);

        // A change edits the definition.
        panel.invoke_param_changed(0, "kernel_size".into(), "7".into());
        drain_ui_queue();
        {
            let project = ui_state.get_project();
            let PipelineCommand::Blur(blur) = project.components[0].steps[0].command().unwrap()
            else {
                panic!("blur")
            };
            assert_eq!(blur.kernel_size, 7);
        }

        // ◇ exposes the setting: a component parameter, shown filled.
        panel.invoke_toggle_exposed(0, "kernel_size".into());
        drain_ui_queue();
        assert_eq!(ui_state.get_project().components[0].parameters.len(), 1);
        assert!(ui_param(&ui, 0, "kernel_size").exposed);

        panel.invoke_close_component();
        drain_ui_queue();
        assert_eq!(panel.get_open_component_name(), "");
        assert_eq!(card_names(&ui), ["Threshold", "Smooth"]);
        let card = panel.get_active_commands().row_data(1).unwrap();
        assert_eq!(
            card.parameters.row_count(),
            1,
            "the exposed setting on the card"
        );
    }

    /// A ◆ setting's own value never runs - the parameter's replaces it - so
    /// editing it in the opened component sets the parameter's default, and the
    /// setting always shows that default.
    #[test]
    fn editing_an_exposed_setting_sets_the_parameters_default() {
        let (ui, ui_state, _controller) = with_a_component();
        let panel = ui.global::<PipelinesPanelState>();
        panel.invoke_open_component(1);
        drain_ui_queue();
        panel.invoke_toggle_exposed(0, "kernel_size".into());
        drain_ui_queue();

        panel.invoke_param_changed(0, "kernel_size".into(), "7".into());
        drain_ui_queue();
        assert_eq!(
            ui_state.get_project().components[0].parameters[0].default,
            ParamValue::Int(7)
        );
        assert_eq!(ui_param(&ui, 0, "kernel_size").value, "7");

        // A component whose step still holds an old value shows the default.
        {
            let mut project = ui_state.get_project_write();
            let blur = project.components[0].steps[0].command_mut().unwrap();
            blur.apply_param_change("kernel_size", "9");
        }
        panel.invoke_close_component();
        panel.invoke_open_component(1);
        drain_ui_queue();
        assert_eq!(ui_param(&ui, 0, "kernel_size").value, "7");

        // The card shows it as the default too.
        panel.invoke_close_component();
        drain_ui_queue();
        let card = panel.get_active_commands().row_data(1).unwrap();
        assert_eq!(card.parameters.row_data(0).unwrap().value, "7");
    }

    #[test]
    fn a_step_in_an_opened_component_can_use_component_memory() {
        use evanalyzer_cfg::core_types::{ImageAddress, MemoryId};
        let (ui, ui_state, controller) = selected(vec![step("ImageCache"), step("Blur")]);
        {
            let mut project = ui_state.get_project_write();
            evanalyzer_app::components::create_component(
                &mut project,
                PipelineId(1),
                0..=0,
                "Keep",
                &[],
            )
            .unwrap();
        }
        controller.sync_steps_of_selected_pipeline_to_slint(PipelineId(1), true);
        drain_ui_queue();
        let panel = ui.global::<PipelinesPanelState>();
        panel.invoke_open_component(0);
        drain_ui_queue();

        // "Component memory", slot 2 - as the picker reports it.
        panel.invoke_param_changed(0, "address".into(), "component:2".into());
        drain_ui_queue();
        let project = ui_state.get_project();
        let Some(PipelineCommand::ImageCache(cache)) = project.components[0].steps[0].command()
        else {
            panic!("image cache")
        };
        assert_eq!(
            cache.address,
            ImageAddress::Memory(MemoryId::ComponentContext(2))
        );
        drop(project);
        let shown = ui_param(&ui, 0, "address");
        assert_eq!(shown.options.row_data(0).unwrap(), "component");
    }

    /// A Script step in a component: its script's `param`s are its settings;
    /// one exposed on the component card runs with the card's value - and stays
    /// bound when `param` lines move, since bindings use the name.
    #[test]
    fn a_scripts_param_is_a_setting_and_can_go_on_the_component_card() {
        let (ui, ui_state, controller) = selected(vec![step("Script")]);
        {
            let mut project = ui_state.get_project_write();
            evanalyzer_app::components::create_component(
                &mut project,
                PipelineId(1),
                0..=0,
                "Run",
                &[],
            )
            .unwrap();
        }
        controller.sync_steps_of_selected_pipeline_to_slint(PipelineId(1), true);
        drain_ui_queue();
        let panel = ui.global::<PipelinesPanelState>();
        panel.invoke_open_component(0);
        drain_ui_queue();

        panel.invoke_param_changed(
            0,
            "source".into(),
            "param rounds = 10;\nparam start = 800.0;".into(),
        );
        drain_ui_queue();
        assert_eq!(ui_param(&ui, 0, "values.rounds").value, "10");

        panel.invoke_toggle_exposed(0, "values.rounds".into());
        drain_ui_queue();
        // A new param in front: nothing to move, the binding is by name.
        panel.invoke_param_changed(
            0,
            "source".into(),
            "param verbose = true;\nparam rounds = 10;\nparam start = 800.0;".into(),
        );
        drain_ui_queue();
        assert_eq!(
            ui_state.get_project().components[0].parameters[0].bindings[0].param,
            "values.rounds"
        );

        // On the card: 42 - what the script gets.
        panel.invoke_close_component();
        drain_ui_queue();
        let key = ui_state.get_project().components[0].parameters[0]
            .key
            .clone();
        panel.invoke_param_changed(0, key.to_string().into(), "42".into());
        drain_ui_queue();
        let project = ui_state.get_project();
        let StepContent::Component(instance) = &project.pipelines[0].steps[0].content else {
            panic!("component step")
        };
        let steps =
            evanalyzer_app::components::component_steps_with_values(instance, &project.components)
                .unwrap();
        let Some(PipelineCommand::Script(script)) = steps[0].command() else {
            panic!("script")
        };
        assert_eq!(script.values["rounds"], ParamValue::Int(42));
    }

    #[test]
    fn a_script_with_a_wrong_param_line_says_so() {
        let (ui, _ui_state, _controller) = selected(vec![step("Script")]);
        let panel = ui.global::<PipelinesPanelState>();
        panel.invoke_param_changed(0, "source".into(), "param n = 1 + 1;".into());
        drain_ui_queue();
        assert_eq!(active_dialog(&ui), DialogType::Warning);
    }

    #[test]
    fn a_second_blur_can_join_the_first_ones_parameter() {
        let (ui, ui_state, _controller) = with_a_component();
        let panel = ui.global::<PipelinesPanelState>();
        panel.invoke_open_component(1);
        drain_ui_queue();

        panel.invoke_toggle_exposed(0, "kernel_size".into());
        drain_ui_queue();
        assert_eq!(active_dialog(&ui), DialogType::None, "nothing to join yet");

        panel.invoke_toggle_exposed(1, "kernel_size".into());
        drain_ui_queue();
        assert_eq!(active_dialog(&ui), DialogType::ExposeChoice);
        let choice = ui.global::<ExposeChoiceState>();
        let choices: Vec<String> = choice.get_choices().iter().map(|c| c.to_string()).collect();
        assert_eq!(choices.len(), 2);
        assert_eq!(choices[0], "New parameter");
        assert!(choices[1].starts_with("Add to “"), "{}", choices[1]);

        choice.invoke_chosen(1);
        drain_ui_queue();
        assert_eq!(active_dialog(&ui), DialogType::None);
        let project = ui_state.get_project();
        let parameters = &project.components[0].parameters;
        assert_eq!(parameters.len(), 1, "one parameter ...");
        assert_eq!(parameters[0].bindings.len(), 2, "... setting both blurs");
        drop(project);
        assert!(ui_param(&ui, 1, "kernel_size").exposed);
    }

    #[test]
    fn a_setting_of_a_list_entry_is_exposed_from_the_card() {
        let (ui, ui_state, controller) = selected(vec![step("Threshold")]);
        {
            let mut project = ui_state.get_project_write();
            evanalyzer_app::components::create_component(
                &mut project,
                PipelineId(1),
                0..=0,
                "Detect",
                &[],
            )
            .unwrap();
        }
        controller.sync_steps_of_selected_pipeline_to_slint(PipelineId(1), true);
        drain_ui_queue();
        let panel = ui.global::<PipelinesPanelState>();
        panel.invoke_open_component(0);
        drain_ui_queue();
        // A threshold entry, added inside the opened component.
        panel.invoke_add_group_item(0, "thresholds".into());
        drain_ui_queue();

        panel.invoke_toggle_exposed(0, "thresholds.0.min_threshold".into());
        drain_ui_queue();
        let project = ui_state.get_project();
        let parameter = &project.components[0].parameters[0];
        assert_eq!(parameter.bindings[0].param, "thresholds.0.min_threshold");
        assert!(
            parameter.label.starts_with("Thresholds 1 · "),
            "{}",
            parameter.label
        );
        drop(project);
        let list = ui_param(&ui, 0, "thresholds");
        let entry = list.group_items.row_data(0).unwrap();
        let min = entry
            .fields
            .iter()
            .find(|f| f.name == "min_threshold")
            .unwrap();
        assert!(min.exposed, "shown filled");
    }

    #[test]
    fn removing_a_list_entry_in_a_component_moves_the_bindings_along() {
        let (ui, ui_state, controller) = selected(vec![step("Threshold")]);
        {
            let mut project = ui_state.get_project_write();
            evanalyzer_app::components::create_component(
                &mut project,
                PipelineId(1),
                0..=0,
                "Detect",
                &[],
            )
            .unwrap();
        }
        controller.sync_steps_of_selected_pipeline_to_slint(PipelineId(1), true);
        drain_ui_queue();
        let panel = ui.global::<PipelinesPanelState>();
        panel.invoke_open_component(0);
        drain_ui_queue();
        for _ in 0..3 {
            panel.invoke_add_group_item(0, "thresholds".into());
        }
        drain_ui_queue();
        panel.invoke_toggle_exposed(0, "thresholds.2.min_threshold".into());
        drain_ui_queue();

        panel.invoke_remove_group_item(0, "thresholds".into(), 0);
        drain_ui_queue();
        let project = ui_state.get_project();
        assert_eq!(
            project.components[0].parameters[0].bindings[0].param, "thresholds.1.min_threshold",
            "still the same entry, now the second"
        );
    }

    /// A library file with component SMOOTH (one blur), version `version`.
    fn library_file(version: u32) -> (PathBuf, ComponentTemplate) {
        let mut definition = component_definition("SMOOTH", "Smooth");
        definition.meta.version = Some(version);
        definition.steps = vec![step("Blur")];
        definition.steps[0].id = evanalyzer_cfg::core_types::StepId(1);
        (
            PathBuf::from("/templates/smooth.evacomp"),
            ComponentTemplate {
                schema_version: 1,
                component: definition,
                uses: vec![],
            },
        )
    }

    #[test]
    fn a_library_component_is_listed_until_the_project_has_its_copy() {
        let (ui, ui_state, controller) = selected(vec![step("Threshold")]);
        *controller.component_library.lock().unwrap() = vec![library_file(1)];
        let picker = ui.global::<CommandPickerState>();
        picker.set_filter_components(true);
        controller.apply_picker_filter(&ui, "");
        assert_eq!(picker.get_shown_library_components().row_count(), 1);
        assert!(
            picker
                .get_shown_library_components()
                .row_data(0)
                .unwrap()
                .source
                .contains("smooth.evacomp")
        );

        picker.set_pipeline_id(1);
        picker.set_insert_after_idx(0);
        picker.invoke_confirm(LIBRARY_COMPONENT_PICKER_ID_BASE);
        drain_ui_queue();
        {
            let project = ui_state.get_project();
            assert_eq!(project.components.len(), 1, "a copy in the project");
            assert!(matches!(
                &project.pipelines[0].steps[1].content,
                StepContent::Component(g) if g.key == "SMOOTH"
            ));
        }
        controller.apply_picker_filter(&ui, "");
        assert_eq!(
            picker.get_shown_library_components().row_count(),
            0,
            "now it's the project's"
        );
        assert_eq!(picker.get_shown_components().row_count(), 1);
    }

    #[test]
    fn a_newer_library_version_is_offered_on_the_component_card() {
        let (ui, ui_state, controller) = selected(vec![step("Threshold")]);
        let (path, v1) = library_file(1);
        {
            let mut project = ui_state.get_project_write();
            let key = evanalyzer_app::components::insert_component_template(
                &mut project,
                &v1,
                "smooth.evacomp",
            )
            .unwrap();
            project.pipelines[0].steps.push(PipelineStepSettings {
                id: evanalyzer_cfg::core_types::StepId(9),
                enabled: true,
                content: StepContent::Component(
                    evanalyzer_cfg::settings::components::ComponentInstance {
                        key,
                        values: Default::default(),
                    },
                ),
            });
        }
        let (_, mut v2) = library_file(2);
        v2.component.steps.push(step("Blur"));
        v2.component.steps[1].id = evanalyzer_cfg::core_types::StepId(2);
        *controller.component_library.lock().unwrap() = vec![(path, v2)];
        controller.sync_steps_of_selected_pipeline_to_slint(PipelineId(1), true);
        drain_ui_queue();
        let card = |ui: &AppWindow| {
            ui.global::<PipelinesPanelState>()
                .get_active_commands()
                .row_data(1)
                .unwrap()
        };
        assert_eq!(card(&ui).update_version, "2");

        ui.global::<PipelinesPanelState>()
            .invoke_update_component(1);
        drain_ui_queue();
        assert_eq!(active_dialog(&ui), DialogType::ComponentUpdate);
        let state = ui.global::<ComponentUpdateState>();
        assert_eq!(state.get_versions(), "v1 → v2");
        assert!(!state.get_changed_here());

        state.invoke_confirm();
        drain_ui_queue();
        let project = ui_state.get_project();
        assert_eq!(project.components[0].steps.len(), 2, "the new version");
        assert_eq!(project.components[0].origin.as_ref().unwrap().version, 2);
        drop(project);
        assert_eq!(card(&ui).update_version, "", "up to date");
    }

    #[test]
    fn unused_components_can_be_removed_in_the_picker() {
        let (ui, ui_state, controller) = with_a_component(); // SMOOTH, used
        ui_state.get_project_write().components.extend([
            component_definition("OLD", "Old"),
            component_definition("OLDER", "Older"),
        ]);
        let picker = ui.global::<CommandPickerState>();
        picker.set_filter_components(true);
        controller.apply_picker_filter(&ui, "");
        assert_eq!(picker.get_unused_component_count(), 2);

        picker.invoke_select(COMPONENT_PICKER_ID_BASE);
        assert!(!picker.get_detail_removable(), "SMOOTH is used");
        picker.invoke_select(COMPONENT_PICKER_ID_BASE + 1);
        assert!(picker.get_detail_removable());
        picker.invoke_remove_component(COMPONENT_PICKER_ID_BASE + 1);
        assert_eq!(picker.get_unused_component_count(), 1);
        assert!(!picker.get_has_detail());

        picker.invoke_remove_unused_components();
        assert_eq!(picker.get_unused_component_count(), 0);
        let keys: Vec<String> = ui_state
            .get_project()
            .components
            .iter()
            .map(|d| d.key.clone())
            .collect();
        assert_eq!(keys, ["SMOOTH"]);
    }

    #[test]
    fn a_component_step_whose_values_dont_work_says_so_on_its_card() {
        let (ui, ui_state, controller) = with_a_component();
        {
            let mut project = ui_state.get_project_write();
            let definition = &mut project.components[0];
            evanalyzer_app::components::toggle_exposed(
                definition,
                definition.steps[0].id,
                "kernel_size",
                "Size",
                "3",
            );
            let id = definition.parameters[0].key.clone();
            let StepContent::Component(instance) = &mut project.pipelines[0].steps[1].content
            else {
                panic!("component step")
            };
            instance.values.insert(
                id,
                evanalyzer_cfg::settings::components::ParamValue::Text("big".into()),
            );
        }
        controller.sync_steps_of_selected_pipeline_to_slint(PipelineId(1), true);
        drain_ui_queue();
        let card = ui
            .global::<PipelinesPanelState>()
            .get_active_commands()
            .row_data(1)
            .unwrap();
        assert!(card.summary.starts_with("⚠ "), "{}", card.summary);
        assert!(card.summary.contains("big"), "{}", card.summary);
    }

    /// The component card's buttons and the opened component's bar fit the panel -
    /// they used to run past its right edge and get cut off.
    #[test]
    fn component_buttons_and_the_opened_components_bar_fit_the_panel() {
        use i_slint_backend_testing::ElementHandle;
        let (ui, _ui_state, _controller) = with_a_component();
        // Tall enough for the whole card - only the width matters here.
        ui.window()
            .set_size(slint::LogicalSize::new(1280.0, 1400.0));
        ui.global::<GlobalAppState>().set_sidebar_left_active_tab(2);
        let model = ui.global::<PipelinesPanelState>().get_active_commands();
        let mut card = model.row_data(1).unwrap();
        card.expanded = true;
        model.set_row_data(1, card);

        let right_edge_of_steps = || {
            ElementHandle::find_by_element_type_name(&ui, "ScrollView")
                .filter(|e| e.absolute_position().x < 400.0)
                .map(|e| e.absolute_position().x + e.size().width)
                .fold(f32::MAX, f32::min)
        };
        let fits = |label: &str| {
            let element = ElementHandle::find_by_accessible_label(&ui, label)
                .next()
                .unwrap_or_else(|| panic!("{label:?} not found"));
            let right = element.absolute_position().x + element.size().width;
            let edge = right_edge_of_steps();
            assert!(
                right <= edge,
                "{label:?} ends at {right}, the step list at {edge}"
            );
        };
        for label in [
            "View",
            "Edit",
            "Expand",
            "Save as",
            "Duplicate",
            "Bundle",
            "Remove",
            "About Threshold",
        ] {
            fits(label);
        }

        ui.global::<PipelinesPanelState>().invoke_open_component(1);
        drain_ui_queue();
        fits("Changes apply to every use of this component · ◇ exposes a setting");
        fits("Settings…");

        ui.global::<PipelinesPanelState>().invoke_close_component();
        ui.global::<PipelinesPanelState>().invoke_view_component(1);
        drain_ui_queue();
        fits("Viewing · read only");
        fits("Edit");
        assert!(
            ElementHandle::find_by_accessible_label(&ui, "Settings…")
                .next()
                .is_none(),
            "no component settings while only viewing"
        );
    }

    #[test]
    fn a_viewed_component_shows_its_steps_but_changes_nothing_until_edit() {
        let (ui, ui_state, _controller) = with_a_component();
        let panel = ui.global::<PipelinesPanelState>();
        let before = {
            let p = ui_state.get_project();
            serde_json::to_value((&p.pipelines, &p.components)).unwrap()
        };

        panel.invoke_view_component(1);
        drain_ui_queue();
        assert_eq!(panel.get_open_component_name(), "Smooth");
        assert!(panel.get_open_component_read_only());
        assert_eq!(card_names(&ui), ["Blur", "Blur"]);

        panel.invoke_param_changed(0, "kernel_size".into(), "7".into());
        panel.invoke_toggle_step(0);
        panel.invoke_duplicate_step(0);
        panel.invoke_remove_step(1);
        panel.invoke_toggle_exposed(0, "kernel_size".into());
        panel.invoke_add_group_item(0, "thresholds".into());
        drain_ui_queue();
        assert_eq!(
            {
                let p = ui_state.get_project();
                serde_json::to_value((&p.pipelines, &p.components)).unwrap()
            },
            before,
            "nothing changed"
        );
        assert_eq!(card_names(&ui), ["Blur", "Blur"]);
        assert_eq!(active_dialog(&ui), DialogType::None);

        // "Edit" in the bar: the same component, now editable.
        panel.invoke_edit_open_component();
        drain_ui_queue();
        assert!(!panel.get_open_component_read_only());
        assert_eq!(panel.get_open_component_name(), "Smooth");
        panel.invoke_param_changed(0, "kernel_size".into(), "7".into());
        drain_ui_queue();
        let project = ui_state.get_project();
        let PipelineCommand::Blur(blur) = project.components[0].steps[0].command().unwrap() else {
            panic!("blur")
        };
        assert_eq!(blur.kernel_size, 7);
    }

    #[test]
    fn closing_a_viewed_component_and_opening_it_again_makes_it_editable() {
        let (ui, _ui_state, _controller) = with_a_component();
        let panel = ui.global::<PipelinesPanelState>();
        panel.invoke_view_component(1);
        drain_ui_queue();
        panel.invoke_close_component();
        drain_ui_queue();
        assert!(!panel.get_open_component_read_only());
        panel.invoke_open_component(1);
        drain_ui_queue();
        assert!(!panel.get_open_component_read_only());
    }

    /// The opened component with both blurs' kernel sizes exposed as two
    /// parameters.
    fn with_two_parameters() -> (AppWindow, Arc<UiState>, Arc<PipelinesController>) {
        let (ui, ui_state, controller) = with_a_component();
        let panel = ui.global::<PipelinesPanelState>();
        panel.invoke_open_component(1);
        drain_ui_queue();
        panel.invoke_toggle_exposed(0, "kernel_size".into());
        drain_ui_queue();
        panel.invoke_toggle_exposed(1, "kernel_size".into());
        drain_ui_queue();
        // A second, own parameter.
        ui.global::<ExposeChoiceState>().invoke_chosen(0);
        drain_ui_queue();
        (ui, ui_state, controller)
    }

    fn settings_rows(ui: &AppWindow) -> Vec<ComponentParamRow> {
        ui.global::<ComponentSettingsState>()
            .get_parameters()
            .iter()
            .collect()
    }

    #[test]
    fn the_component_settings_dialog_renames_describes_and_reorders() {
        let (ui, ui_state, _controller) = with_two_parameters();
        ui.global::<PipelinesPanelState>()
            .invoke_open_component_settings();
        drain_ui_queue();
        assert_eq!(active_dialog(&ui), DialogType::ComponentSettings);
        let dialog = ui.global::<ComponentSettingsState>();
        assert_eq!(dialog.get_name(), "Smooth");
        assert_eq!(dialog.get_focus_index(), -1);
        assert_eq!(settings_rows(&ui).len(), 2);
        assert!(
            settings_rows(&ui)[0].sets.contains("Blur"),
            "names what it sets: {}",
            settings_rows(&ui)[0].sets
        );
        let (first, second) = (
            settings_rows(&ui)[0].key.clone(),
            settings_rows(&ui)[1].key.clone(),
        );

        dialog.invoke_name_edited("Smoothing".into());
        dialog.invoke_name_edited("  ".into());
        dialog.invoke_description_edited("Blurs twice".into());
        dialog.invoke_parameter_name_edited(first.clone(), "Radius".into());
        dialog.invoke_parameter_help_edited(first.clone(), "How strong".into());
        drain_ui_queue();
        {
            let project = ui_state.get_project();
            let definition = &project.components[0];
            assert_eq!(
                definition.meta.name, "Smoothing",
                "an empty name is ignored"
            );
            assert_eq!(definition.meta.short_description, "Blurs twice");
            let parameter = &definition.parameters[0];
            assert_eq!(
                (parameter.label.as_str(), parameter.description.as_str()),
                ("Radius", "How strong")
            );
        }

        dialog.invoke_parameter_moved(second.clone(), -1);
        drain_ui_queue();
        let ids: Vec<SharedString> = settings_rows(&ui).iter().map(|r| r.key.clone()).collect();
        assert_eq!(ids, [second, first]);
        assert_eq!(settings_rows(&ui)[1].label, "Radius");

        dialog.invoke_close();
        drain_ui_queue();
        assert_eq!(active_dialog(&ui), DialogType::None);
        assert_eq!(
            ui.global::<PipelinesPanelState>().get_open_component_name(),
            "Smoothing"
        );
    }

    /// What a parameter sets is edited in the settings dialog: "Also set…"
    /// offers the settings with its control, ✕ takes one away again.
    #[test]
    fn the_settings_dialog_edits_what_a_parameter_sets() {
        let (ui, ui_state, _controller) = with_a_component();
        let panel = ui.global::<PipelinesPanelState>();
        panel.invoke_open_component(1);
        drain_ui_queue();
        panel.invoke_toggle_exposed(0, "kernel_size".into());
        drain_ui_queue();
        let second_blur = panel
            .get_active_commands()
            .row_data(1)
            .unwrap()
            .step_id
            .to_string();
        panel.invoke_open_component_settings();
        drain_ui_queue();
        let dialog = ui.global::<ComponentSettingsState>();
        let row = |ui: &AppWindow| {
            ui.global::<ComponentSettingsState>()
                .get_parameters()
                .row_data(0)
                .unwrap()
        };
        assert_eq!(row(&ui).bindings.row_count(), 1);
        let addable: Vec<String> = (0..row(&ui).addable.row_count())
            .map(|i| row(&ui).addable.row_data(i).unwrap().to_string())
            .collect();
        assert_eq!(addable[0], "Also set…");
        let index = addable
            .iter()
            .position(|a| a.starts_with(&format!("#{second_blur} Blur › Kernel")))
            .unwrap_or_else(|| panic!("the other blur's kernel size: {addable:?}"));

        dialog.invoke_binding_added(row(&ui).key, index as i32 - 1);
        drain_ui_queue();
        assert_eq!(row(&ui).bindings.row_count(), 2);
        assert_eq!(
            ui_state.get_project().components[0].parameters[0]
                .bindings
                .len(),
            2
        );

        let added = row(&ui).bindings.row_data(1).unwrap();
        assert!(
            added.text.starts_with(&format!("#{second_blur} Blur")),
            "{}",
            added.text
        );
        dialog.invoke_binding_removed(row(&ui).key, added.step_id, added.path);
        drain_ui_queue();
        assert_eq!(row(&ui).bindings.row_count(), 1);
        assert_eq!(
            ui_state.get_project().components[0].parameters[0]
                .bindings
                .len(),
            1
        );
    }

    /// A parameter opened with ✎ is edited until Apply or Cancel: Cancel
    /// undoes everything since ✎ - its name and what it sets too.
    #[test]
    fn cancel_undoes_a_parameters_edit_and_apply_keeps_it() {
        let (ui, ui_state, _controller) = with_a_component();
        let panel = ui.global::<PipelinesPanelState>();
        panel.invoke_open_component(1);
        drain_ui_queue();
        panel.invoke_toggle_exposed(0, "kernel_size".into());
        drain_ui_queue();
        panel.invoke_open_component_settings();
        drain_ui_queue();
        let dialog = ui.global::<ComponentSettingsState>();
        let key = dialog.get_parameters().row_data(0).unwrap().key;
        let parameter =
            |ui_state: &UiState| ui_state.get_project().components[0].parameters[0].clone();
        let before = parameter(&ui_state);

        let edit = |dialog: &ComponentSettingsState| {
            dialog.invoke_parameter_edit_started(key.clone());
            dialog.invoke_parameter_name_edited(key.clone(), "Radius".into());
            let row = dialog.get_parameters().row_data(0).unwrap();
            let other_blur = (1..row.addable.row_count())
                .position(|i| row.addable.row_data(i).unwrap().contains("Blur"))
                .expect("the other blur's kernel size");
            dialog.invoke_binding_added(key.clone(), other_blur as i32);
            drain_ui_queue();
        };

        edit(&dialog);
        assert_eq!(parameter(&ui_state).label, "Radius");
        assert_eq!(parameter(&ui_state).bindings.len(), 2);
        dialog.invoke_parameter_edit_cancelled();
        drain_ui_queue();
        let cancelled = parameter(&ui_state);
        assert_eq!(
            (cancelled.label, cancelled.bindings),
            (before.label.clone(), before.bindings.clone()),
            "as it was at ✎"
        );
        assert_eq!(
            dialog.get_parameters().row_data(0).unwrap().label,
            before.label,
            "the row shows it too"
        );

        edit(&dialog);
        dialog.invoke_parameter_edit_applied();
        drain_ui_queue();
        assert_eq!(parameter(&ui_state).label, "Radius");
        assert_eq!(parameter(&ui_state).bindings.len(), 2);
        // Nothing left to cancel.
        dialog.invoke_parameter_edit_cancelled();
        drain_ui_queue();
        assert_eq!(parameter(&ui_state).label, "Radius");
    }

    /// Opened from ✎ next to an exposed setting, the parameter is edited from
    /// the start - Cancel works without pressing ✎ first.
    #[test]
    fn a_parameter_opened_from_its_setting_can_be_cancelled_right_away() {
        let (ui, ui_state, _controller) = with_two_parameters();
        ui.global::<PipelinesPanelState>()
            .invoke_edit_exposed(0, "kernel_size".into());
        drain_ui_queue();
        let dialog = ui.global::<ComponentSettingsState>();
        let key = dialog.get_parameters().row_data(0).unwrap().key;
        let label = ui_state.get_project().components[0].parameters[0]
            .label
            .clone();
        dialog.invoke_parameter_name_edited(key, "Something else".into());
        dialog.invoke_parameter_edit_cancelled();
        drain_ui_queue();
        assert_eq!(
            ui_state.get_project().components[0].parameters[0].label,
            label
        );
    }

    /// Typing a name and then a help text keeps both - the help text's
    /// field used to send the name as the dialog opened, undoing the edit.
    #[test]
    fn a_name_and_a_help_text_typed_one_after_the_other_both_stay() {
        use i_slint_backend_testing::ElementHandle;
        let (ui, ui_state, _controller) = with_two_parameters();
        ui.window().set_size(slint::LogicalSize::new(1280.0, 800.0));
        // ✎ next to the first exposed setting: the dialog with its parameter
        // opened to edit.
        ui.global::<PipelinesPanelState>()
            .invoke_edit_exposed(0, "kernel_size".into());
        drain_ui_queue();
        assert_eq!(ui.global::<ComponentSettingsState>().get_focus_index(), 0);
        let field = |label: &str| {
            ElementHandle::find_by_accessible_label(&ui, label)
                .next()
                .unwrap_or_else(|| panic!("{label:?} not found"))
        };

        field("Parameter name").set_accessible_value("Radius");
        drain_ui_queue();
        field("Parameter help text").set_accessible_value("How strong");
        drain_ui_queue();
        field("Parameter name").set_accessible_value("Blur radius");
        drain_ui_queue();

        let project = ui_state.get_project();
        let parameter = &project.components[0].parameters[0];
        assert_eq!(
            (parameter.label.as_str(), parameter.description.as_str()),
            ("Blur radius", "How strong")
        );
    }

    #[test]
    fn the_pen_next_to_an_exposed_setting_opens_the_dialog_at_its_parameter() {
        let (ui, ui_state, _controller) = with_two_parameters();
        let panel = ui.global::<PipelinesPanelState>();
        panel.invoke_edit_exposed(1, "kernel_size".into());
        drain_ui_queue();
        assert_eq!(active_dialog(&ui), DialogType::ComponentSettings);
        assert_eq!(ui.global::<ComponentSettingsState>().get_focus_index(), 1);

        // ✕ stops exposing it: the setting shows ◇ again.
        let second = settings_rows(&ui)[1].key.clone();
        ui.global::<ComponentSettingsState>()
            .invoke_parameter_removed(second);
        drain_ui_queue();
        assert_eq!(settings_rows(&ui).len(), 1);
        assert_eq!(ui_state.get_project().components[0].parameters.len(), 1);
        ui.global::<ComponentSettingsState>().invoke_close();
        drain_ui_queue();
        assert!(!ui_param(&ui, 1, "kernel_size").exposed);
        assert!(ui_param(&ui, 0, "kernel_size").exposed);

        // A setting that isn't exposed has no parameter to show.
        panel.invoke_edit_exposed(1, "kernel_size".into());
        drain_ui_queue();
        assert_eq!(active_dialog(&ui), DialogType::None);
    }

    /// ✎ on a parameter far down the list scrolls the dialog to it, and
    /// the dialog fits the window however many parameters there are.
    #[test]
    fn the_dialog_scrolls_to_the_pens_parameter_and_fits_the_window() {
        use evanalyzer_cfg::settings::components::{ComponentParameterDef, ParamBinding};
        use i_slint_backend_testing::ElementHandle;
        let (ui, ui_state, _controller) = with_two_parameters();
        ui.window().set_size(slint::LogicalSize::new(1280.0, 800.0));
        {
            let mut project = ui_state.get_project_write();
            let definition = &mut project.components[0];
            let fillers = (0..10).map(|i| ComponentParameterDef {
                key: ComponentParamKey(format!("filler_{i}")),
                label: format!("Filler {i}"),
                description: String::new(),
                section: String::new(),
                default: ParamValue::Int(0),
                bindings: vec![ParamBinding {
                    step: evanalyzer_cfg::core_types::StepId(999),
                    param: "kernel_size".into(),
                }],
            });
            definition.parameters.splice(0..0, fillers);
        }
        let last = ui_state.get_project().components[0].parameters.len() - 1;
        ui.global::<PipelinesPanelState>()
            .invoke_edit_exposed(1, "kernel_size".into());
        drain_ui_queue();
        assert_eq!(
            ui.global::<ComponentSettingsState>().get_focus_index(),
            last as i32
        );

        let label = settings_rows(&ui)[last].label.to_string();
        let edits: Vec<_> = ElementHandle::find_by_accessible_label(&ui, "Parameter name")
            .filter(|e| e.accessible_value().is_some_and(|v| v == label))
            .collect();
        let row = edits.last().expect("the parameter's name field");
        let y = row.absolute_position().y;
        assert!(
            (0.0..800.0 - row.size().height).contains(&y),
            "the focused parameter is scrolled into view: y = {y}"
        );
        let close = ElementHandle::find_by_accessible_label(&ui, "Close")
            .last()
            .expect("Close");
        assert!(
            close.absolute_position().y + close.size().height <= 800.0,
            "the dialog fits the window"
        );
    }

    #[test]
    fn a_viewed_components_settings_do_not_open() {
        let (ui, _ui_state, _controller) = with_two_parameters();
        let panel = ui.global::<PipelinesPanelState>();
        panel.invoke_close_component();
        panel.invoke_view_component(1);
        drain_ui_queue();

        panel.invoke_open_component_settings();
        panel.invoke_edit_exposed(0, "kernel_size".into());
        drain_ui_queue();
        assert_eq!(active_dialog(&ui), DialogType::None);
    }

    #[test]
    fn the_long_description_is_edited_and_a_help_text_reset_to_the_settings_own() {
        let (ui, ui_state, _controller) = with_two_parameters();
        ui.global::<PipelinesPanelState>()
            .invoke_open_component_settings();
        drain_ui_queue();
        let dialog = ui.global::<ComponentSettingsState>();
        let row = settings_rows(&ui)[0].clone();
        assert!(
            !row.default_description.is_empty(),
            "the blur's own help text shows while there's none"
        );

        dialog.invoke_long_description_edited("Two blurs.\nUse for noisy images.".into());
        dialog.invoke_parameter_help_edited(row.key.clone(), "Own help".into());
        drain_ui_queue();
        assert_eq!(
            ui_state.get_project().components[0].meta.description,
            "Two blurs.\nUse for noisy images."
        );
        assert_eq!(
            ui_state.get_project().components[0].parameters[0].description,
            "Own help"
        );

        dialog.invoke_parameter_help_reset(row.key.clone());
        drain_ui_queue();
        let project = ui_state.get_project();
        let parameter = &project.components[0].parameters[0];
        assert_eq!(parameter.description, "");
        assert_eq!(parameter.label, row.label.as_str(), "the name stays");
        drop(project);
        assert_eq!(settings_rows(&ui)[0].description, "");
    }

    #[test]
    fn step_cards_carry_the_commands_and_the_components_descriptions() {
        let (ui, ui_state, controller) = with_a_component();
        let card = |i: usize| {
            ui.global::<PipelinesPanelState>()
                .get_active_commands()
                .row_data(i)
                .unwrap()
        };
        let threshold = all_command_meta()
            .into_iter()
            .find(|m| m.name == "Threshold")
            .unwrap();
        assert!(card(0).help.starts_with(threshold.summary));
        assert_eq!(
            card(1).help,
            "",
            "a component without descriptions has no ?"
        );

        {
            let mut project = ui_state.get_project_write();
            project.components[0].meta.short_description = "Smooths.".into();
            project.components[0].meta.description = "Twice, with blurs.".into();
        }
        controller.sync_steps_of_selected_pipeline_to_slint(PipelineId(1), false);
        drain_ui_queue();
        assert_eq!(card(1).help, "Smooths.\n\nTwice, with blurs.");
    }

    #[test]
    fn every_command_has_a_help_text_for_its_card() {
        for meta in all_command_meta() {
            let command = evanalyzer_cfg::settings::pipeline_command::default_command(meta.id)
                .unwrap_or_else(|| panic!("{} has no default", meta.name));
            assert!(
                !command_help(command.name()).is_empty(),
                "{} has no help text",
                command.name()
            );
        }
    }

    #[test]
    fn opening_a_component_used_twice_asks_and_can_copy_it_for_this_step() {
        let (ui, ui_state, controller) = with_a_component();
        {
            let mut project = ui_state.get_project_write();
            let mut second = project.pipelines[0].steps[1].clone();
            second.id = evanalyzer_cfg::core_types::StepId(99);
            project.pipelines[0].steps.push(second);
        }
        controller.sync_steps_of_selected_pipeline_to_slint(PipelineId(1), true);
        drain_ui_queue();

        ui.global::<PipelinesPanelState>().invoke_open_component(1);
        drain_ui_queue();
        assert_eq!(active_dialog(&ui), DialogType::ComponentEditChoice);
        let choice = ui.global::<ComponentEditChoiceState>();
        assert_eq!(choice.get_uses(), 2);

        choice.invoke_only_this();
        drain_ui_queue();
        assert_eq!(active_dialog(&ui), DialogType::None);
        assert_eq!(
            ui.global::<PipelinesPanelState>().get_open_component_name(),
            "Smooth (2)"
        );
        let project = ui_state.get_project();
        assert_eq!(project.components.len(), 2);
        let keys: Vec<String> = project.pipelines[0]
            .steps
            .iter()
            .filter_map(|s| match &s.content {
                StepContent::Component(g) => Some(g.key.clone()),
                _ => None,
            })
            .collect();
        assert_eq!(keys, ["SMOOTH_2", "SMOOTH"], "only this step uses the copy");
    }

    #[test]
    fn expanding_a_component_card_puts_the_steps_back() {
        let (ui, ui_state, _controller) = with_a_component();
        ui.global::<PipelinesPanelState>()
            .invoke_expand_component(1);
        drain_ui_queue();
        assert_eq!(card_names(&ui), ["Threshold", "Blur", "Blur"]);
        assert_eq!(ui_state.get_project().pipelines[0].steps.len(), 3);
    }

    #[test]
    fn a_duplicated_step_gets_an_id_of_its_own() {
        let (ui, ui_state, _controller) = selected(vec![step("Threshold")]);
        ui.global::<PipelinesPanelState>().invoke_duplicate_step(0);
        drain_ui_queue();
        let project = ui_state.get_project();
        let steps = &project.pipelines[0].steps;
        assert_eq!(steps.len(), 2);
        assert_ne!(steps[0].id, steps[1].id);
    }

    #[test]
    fn duplicate_pipeline_appends_a_copy_and_selects_it() {
        let (ui, ui_state, _controller) = selected(vec![step("Threshold")]);
        let panel = ui.global::<PipelinesPanelState>();
        panel.invoke_duplicate_pipeline(1);
        drain_ui_queue();
        {
            let project = ui_state.get_project();
            assert_eq!(project.pipelines.len(), 2);
            assert_eq!(project.pipelines[1].id, PipelineId(2));
            assert_eq!(project.pipelines[1].name, "Pipeline 1 (Copy)");
            assert_eq!(project.pipelines[1].steps.len(), 1);
            assert_eq!(
                project.pipelines[1].steps[0].id, project.pipelines[0].steps[0].id,
                "step ids are unique per pipeline - the copy keeps them"
            );
        }
        assert_eq!(panel.get_active_pipeline_id(), 2);
        assert_eq!(panel.get_pipelines().row_count(), 2);

        panel.invoke_duplicate_pipeline(99);
        assert_eq!(
            ui_state.get_project().pipelines.len(),
            2,
            "unknown id: no-op"
        );
    }

    #[test]
    fn deleting_a_pipeline_asks_first_then_selects_the_next_one() {
        let (ui, ui_state, _controller) = selected(vec![]);
        add_pipeline_with(
            &ui_state,
            2,
            ImageAddress::Channel(0),
            vec![step("Threshold")],
        );
        let panel = ui.global::<PipelinesPanelState>();

        panel.invoke_delete_pipeline(1);
        let confirm = ui.global::<PipelineDeleteConfirmState>();
        assert_eq!(confirm.get_pipeline_name(), "Pipeline 1");
        assert_eq!(active_dialog(&ui), DialogType::PipelineDeleteConfirm);
        assert_eq!(ui_state.get_project().pipelines.len(), 2, "not deleted yet");

        confirm.invoke_confirm();
        drain_ui_queue();
        assert_eq!(ui_state.get_project().pipelines.len(), 1);
        assert_eq!(active_dialog(&ui), DialogType::None);
        assert_eq!(panel.get_active_pipeline_id(), 2);
        assert_eq!(panel.get_active_commands().row_count(), 1);

        // Deleting the last one clears the editor.
        panel.invoke_delete_pipeline(2);
        confirm.invoke_confirm();
        drain_ui_queue();
        assert!(ui_state.get_project().pipelines.is_empty());
        assert_eq!(panel.get_active_pipeline_id(), 0);
        assert_eq!(panel.get_active_pipeline_name(), "");
        assert_eq!(panel.get_active_commands().row_count(), 0);
    }

    #[test]
    fn deleting_an_unknown_pipeline_names_it_by_id() {
        let (ui, _ui_state, _controller) = selected(vec![]);
        ui.global::<PipelinesPanelState>().invoke_delete_pipeline(7);
        assert_eq!(
            ui.global::<PipelineDeleteConfirmState>()
                .get_pipeline_name(),
            "Pipeline 7"
        );
    }

    #[test]
    fn toggle_remove_and_duplicate_step_change_the_active_pipeline() {
        let (ui, ui_state, _controller) = selected(vec![step("Threshold"), step("Colocalization")]);
        let panel = ui.global::<PipelinesPanelState>();

        panel.invoke_toggle_step(0);
        drain_ui_queue();
        assert!(!ui_state.get_project().pipelines[0].steps[0].enabled);
        assert!(!panel.get_active_commands().row_data(0).unwrap().enabled);
        assert_eq!(panel.get_total_enabled_steps(), 1);

        panel.invoke_duplicate_step(1);
        drain_ui_queue();
        assert_eq!(
            step_names(&ui_state, 1),
            ["Threshold", "Colocalization", "Colocalization"]
        );

        panel.invoke_remove_step(0);
        drain_ui_queue();
        assert_eq!(
            step_names(&ui_state, 1),
            ["Colocalization", "Colocalization"]
        );
        assert_eq!(panel.get_active_commands().row_count(), 2);

        // Out of range: ignored.
        panel.invoke_remove_step(9);
        panel.invoke_duplicate_step(9);
        panel.invoke_expand_step(0);
        assert_eq!(step_names(&ui_state, 1).len(), 2);
    }

    #[test]
    fn after_an_object_step_the_picker_also_suggests_dataset_steps() {
        // Colocalization allows Object and Dataset steps after it - the
        // last chip, which a too-short suggestion list would overrun.
        let (ui, _ui_state, _controller) = selected(vec![step("Colocalization")]);
        ui.global::<PipelinesPanelState>().invoke_insert_step(1, 0);
        let picker = ui.global::<CommandPickerState>();
        assert_eq!(active_dialog(&ui), DialogType::CommandSelectionDialog);
        assert!(picker.get_fcat_cls());
        assert!(picker.get_fcat_dat());
        assert!(!picker.get_fcat_pre());
    }

    #[test]
    fn insert_step_opens_the_picker_with_the_next_categories_suggested() {
        let (ui, _ui_state, _controller) = selected(vec![step("Threshold")]);
        ui.global::<PipelinesPanelState>().invoke_insert_step(1, 0);
        let picker = ui.global::<CommandPickerState>();
        assert_eq!(active_dialog(&ui), DialogType::CommandSelectionDialog);
        assert_eq!(picker.get_target_pipeline(), "Pipeline 1");
        assert_eq!(picker.get_total_steps(), 1);
        assert_eq!(picker.get_insert_after_idx(), 0);
        // Threshold is a segmentation step: something follows it.
        assert!(picker.get_context_category() >= 0);
        let any_chip = picker.get_fcat_pre()
            || picker.get_fcat_seg()
            || picker.get_fcat_obj()
            || picker.get_fcat_mea()
            || picker.get_fcat_cls();
        assert!(any_chip, "suggested follow-up categories are pre-selected");
        assert!(picker.get_total_shown() > 0);
    }

    #[test]
    fn insert_step_into_an_empty_or_unknown_pipeline_shows_all_categories() {
        let (ui, _ui_state, _controller) = selected(vec![]);
        let picker = ui.global::<CommandPickerState>();
        for pipeline_id in [1, 42] {
            ui.global::<PipelinesPanelState>()
                .invoke_insert_step(pipeline_id, -1);
            assert_eq!(picker.get_context_category(), -1);
            assert!(!picker.get_fcat_seg() && !picker.get_fcat_pre());
        }
        assert_eq!(picker.get_target_pipeline(), "");
    }

    #[test]
    fn picker_select_shows_a_command_and_confirm_inserts_it_after_the_given_step() {
        let (ui, ui_state, _controller) = selected(vec![step("Threshold"), step("Threshold")]);
        let coloc_id = all_command_meta()
            .into_iter()
            .find(|m| m.name == "Colocalization")
            .unwrap()
            .id;
        let picker = ui.global::<CommandPickerState>();

        picker.invoke_select(coloc_id);
        assert!(picker.get_has_detail());
        assert_eq!(picker.get_detail().name, "Colocalization");

        // After step 0.
        picker.set_pipeline_id(1);
        picker.set_insert_after_idx(0);
        ui.global::<GlobalAppState>()
            .set_active_dialog(DialogType::CommandSelectionDialog);
        picker.invoke_confirm(coloc_id);
        drain_ui_queue();
        assert_eq!(
            step_names(&ui_state, 1),
            ["Threshold", "Colocalization", "Threshold"]
        );
        assert_eq!(active_dialog(&ui), DialogType::None);

        // Before everything, and past the end.
        picker.set_insert_after_idx(-1);
        picker.invoke_confirm(coloc_id);
        picker.set_insert_after_idx(99);
        picker.invoke_confirm(coloc_id);
        let names = step_names(&ui_state, 1);
        assert_eq!(names.first().unwrap(), "Colocalization");
        assert_eq!(names.last().unwrap(), "Colocalization");
        assert_eq!(names.len(), 5);

        // Unknown and negative ids are ignored.
        picker.invoke_confirm(99_999);
        picker.invoke_confirm(-7);
        picker.invoke_select(99_999);
        picker.invoke_select(-7);
        assert_eq!(step_names(&ui_state, 1).len(), 5);
    }

    #[test]
    fn saving_an_empty_pipeline_as_component_says_there_is_nothing_to_save() {
        let (ui, _ui_state, _controller) = selected(vec![]);
        ui.global::<PipelinesPanelState>()
            .invoke_save_as_component();
        assert_eq!(
            ui.global::<GlobalAppState>().get_active_dialog(),
            DialogType::Warning
        );
        assert_eq!(ui.global::<WarningState>().get_title(), "Nothing to save");
    }

    #[test]
    fn saving_a_pipeline_as_component_opens_the_metadata_dialog() {
        let (ui, _ui_state, _controller) = selected(vec![step("Threshold")]);
        ui.global::<PipelinesPanelState>()
            .invoke_save_as_component();
        assert_eq!(
            ui.global::<GlobalAppState>().get_active_dialog(),
            DialogType::TemplateMeta
        );
        assert_eq!(
            ui.global::<crate::TemplateMetaState>().get_dialog_title(),
            "Save Pipeline as Component"
        );
    }

    #[test]
    fn picker_query_filters_the_list() {
        let (ui, _ui_state, _controller) = selected(vec![]);
        let picker = ui.global::<CommandPickerState>();
        picker.invoke_query_changed("".into());
        let all = picker.get_total_shown();
        picker.invoke_query_changed("coloc".into());
        let filtered = picker.get_total_shown();
        assert!(filtered >= 1 && filtered < all, "{filtered} of {all}");
    }

    #[test]
    fn attach_fills_the_picker_with_every_command_once() {
        let (ui, _ui_state, _controller) = selected(vec![]);
        let picker = ui.global::<CommandPickerState>();
        let all = picker.get_all_commands().row_count();
        assert_eq!(all, all_command_meta().len());
        let per_category = picker.get_cat_count_pre()
            + picker.get_cat_count_seg()
            + picker.get_cat_count_obj()
            + picker.get_cat_count_mea()
            + picker.get_cat_count_cls()
            + picker.get_cat_count_dat();
        assert_eq!(per_category as usize, all);
    }

    #[test]
    fn changing_a_flat_parameter_updates_the_project_and_the_shown_value() {
        let (ui, ui_state, _controller) = selected(vec![step("Colocalization")]);
        let panel = ui.global::<PipelinesPanelState>();
        let summary_before = panel.get_active_commands().row_data(0).unwrap().summary;

        panel.invoke_param_changed(0, "min_coloc_area".into(), "5".into());
        assert_eq!(ui_param(&ui, 0, "min_coloc_area").value, "5");
        let project = ui_state.get_project();
        let PipelineCommand::Colocalization(settings) =
            project.pipelines[0].steps[0].command().unwrap()
        else {
            panic!("expected coloc");
        };
        assert_eq!(settings.min_coloc_area, 5.0);
        drop(project);
        assert!(ui_state.is_dirty());
        let _ = summary_before;
    }

    #[test]
    fn toggling_a_multi_select_parameter_updates_its_flags() {
        let (ui, _ui_state, _controller) = selected(vec![step("Colocalization")]);
        ui.global::<PipelinesPanelState>().invoke_param_changed(
            0,
            "classes_to_coloc".into(),
            "toggle:3".into(),
        );
        let param = ui_param(&ui, 0, "classes_to_coloc");
        assert_eq!(param.value, "3");
        let flags: Vec<String> = (0..param.options.row_count())
            .map(|i| param.options.row_data(i).unwrap().to_string())
            .collect();
        assert_eq!(flags[3], "1");
        assert_eq!(flags.iter().filter(|f| *f == "1").count(), 1);
    }

    #[test]
    fn switching_a_variant_that_changes_the_fields_resyncs_the_step() {
        let (ui, _ui_state, _controller) = selected(vec![step("TransformObjects")]);
        let names = |ui: &AppWindow| -> Vec<String> {
            let params = ui
                .global::<PipelinesPanelState>()
                .get_active_commands()
                .row_data(0)
                .unwrap()
                .parameters;
            (0..params.row_count())
                .map(|i| params.row_data(i).unwrap().name.to_string())
                .collect()
        };
        let before = names(&ui);
        assert!(before.contains(&"function.factor".to_string()));
        ui.global::<PipelinesPanelState>().invoke_param_changed(
            0,
            "function".into(),
            "Expand".into(),
        );
        drain_ui_queue();
        let after = names(&ui);
        assert_ne!(before, after, "the Expand variant has other fields");
        assert_eq!(ui_param(&ui, 0, "function").value, "Expand");
    }

    #[test]
    fn group_items_can_be_added_changed_and_removed() {
        let (ui, ui_state, _controller) = selected(vec![step("Threshold")]);
        let panel = ui.global::<PipelinesPanelState>();
        panel.invoke_add_group_item(0, "thresholds".into());
        panel.invoke_add_group_item(0, "thresholds".into());
        drain_ui_queue();
        let thresholds = ui_param(&ui, 0, "thresholds");
        assert_eq!(thresholds.group_items.row_count(), 2);

        // Change a nested field that keeps the field set: patched in place.
        let fields = thresholds.group_items.row_data(0).unwrap().fields;
        let (name, numeric) = (0..fields.row_count())
            .filter_map(|k| fields.row_data(k))
            .find(|f| f.value.parse::<f64>().is_ok())
            .map(|f| (f.name.to_string(), f.value.to_string()))
            .expect("a numeric field");
        let new_value = if numeric == "7" { "8" } else { "7" };
        panel.invoke_param_changed(0, format!("thresholds.0.{name}").into(), new_value.into());
        drain_ui_queue();
        let fields = ui_param(&ui, 0, "thresholds")
            .group_items
            .row_data(0)
            .unwrap()
            .fields;
        let shown = (0..fields.row_count())
            .filter_map(|k| fields.row_data(k))
            .find(|f| f.name.as_str() == name)
            .unwrap()
            .value;
        assert_eq!(
            shown.parse::<f64>().unwrap(),
            new_value.parse::<f64>().unwrap()
        );

        panel.invoke_remove_group_item(0, "thresholds".into(), 0);
        drain_ui_queue();
        assert_eq!(ui_param(&ui, 0, "thresholds").group_items.row_count(), 1);
        let project = ui_state.get_project();
        let params = project.pipelines[0].steps[0]
            .command()
            .unwrap()
            .to_parameters();
        assert_eq!(
            params
                .iter()
                .find(|p| p.name == "thresholds")
                .unwrap()
                .groups
                .len(),
            1
        );
    }

    #[test]
    fn param_change_for_an_unknown_step_or_pipeline_is_ignored() {
        let (ui, ui_state, _controller) = selected(vec![step("Colocalization")]);
        let panel = ui.global::<PipelinesPanelState>();
        panel.invoke_param_changed(5, "min_coloc_area".into(), "9".into());
        panel.set_active_pipeline_id(42);
        panel.invoke_param_changed(0, "min_coloc_area".into(), "9".into());
        let project = ui_state.get_project();
        let PipelineCommand::Colocalization(settings) =
            project.pipelines[0].steps[0].command().unwrap()
        else {
            panic!("expected coloc");
        };
        assert_eq!(settings.min_coloc_area, 0.0);
    }

    #[test]
    fn show_model_info_reports_a_model_that_cannot_be_loaded() {
        let mut classifier = step(PIXEL_CLASSIFIER_COMMAND_NAME);
        if let PipelineCommand::PixelClassifier(settings) = classifier.command_mut().unwrap() {
            settings.model_path = PathBuf::from("/nonexistent/model.evamodel");
        }
        let (ui, _ui_state, _controller) = selected(vec![classifier, step("Threshold")]);
        let panel = ui.global::<PipelinesPanelState>();
        panel.invoke_show_model_info(0);
        let warning = ui.global::<WarningState>();
        assert_eq!(warning.get_title(), "Could not load model");
        assert!(warning.get_message().contains("model.evamodel"));
        assert_eq!(active_dialog(&ui), DialogType::Warning);

        // Not a classifier step / unknown step: nothing happens.
        ui.global::<GlobalAppState>()
            .set_active_dialog(DialogType::None);
        panel.invoke_show_model_info(1);
        panel.invoke_show_model_info(9);
        assert_eq!(active_dialog(&ui), DialogType::None);
    }

    #[test]
    fn run_all_without_a_saved_project_warns_instead_of_asking_for_a_job_name() {
        let (ui, _ui_state, _controller) = selected(vec![]);
        ui.global::<PipelinesPanelState>().invoke_run_all();
        drain_ui_queue();
        assert_eq!(active_dialog(&ui), DialogType::Warning);
        assert!(
            ui.global::<WarningState>()
                .get_message()
                .contains("save the project")
        );
    }

    #[test]
    fn run_all_asks_for_a_job_name_and_dispatches_a_full_run() {
        let (ui, ui_state, controller) = selected(vec![]);
        ui_state.get_project_write().tmp_settings.current_project =
            Some(PathBuf::from("/data/exp/project.evaproj"));
        ui.global::<PipelinesPanelState>().invoke_run_all();
        assert_eq!(active_dialog(&ui), DialogType::RunAnalysisConfirm);

        ui.global::<RunAnalysisState>()
            .set_job_name("  run 1  ".into());
        ui.global::<RunAnalysisState>().invoke_confirm();
        let task = controller
            .task_request
            .0
            .lock()
            .unwrap()
            .take()
            .expect("dispatched");
        assert!(!task.preview);
        assert_eq!(task.job_name.as_deref(), Some("  run 1  "));
        assert_eq!(task.project_path, PathBuf::from("/data/exp"));

        // A blank job name means none.
        ui.global::<RunAnalysisState>().set_job_name("   ".into());
        ui.global::<RunAnalysisState>().invoke_confirm();
        let task = controller.task_request.0.lock().unwrap().take().unwrap();
        assert_eq!(task.job_name, None);
    }

    #[test]
    fn run_all_can_run_only_the_dataset_steps_on_earlier_results() {
        let (ui, ui_state, controller) = selected(vec![step("SimpleTracker")]);
        ui_state.get_project_write().tmp_settings.current_project =
            Some(PathBuf::from("/data/exp/project.evaproj"));
        let result = |name: &str| ResultItemData {
            name: name.into(),
            path: format!("/data/exp/results/x__{name}/{name}.evadb").into(),
            file_size: "1 MB".into(),
            modified: "today".into(),
        };
        let results = ui.global::<ResultsListState>();
        results.set_results_list(ModelRc::new(VecModel::from(vec![
            result("newest"),
            result("older"),
        ])));
        results.set_selected_result_index(1);

        ui.global::<PipelinesPanelState>().invoke_run_all();
        let state = ui.global::<RunAnalysisState>();
        assert!(state.get_has_dataset_steps());
        assert!(!state.get_dataset_only(), "a full analysis by default");
        assert_eq!(state.get_base_names().row_count(), 2);
        assert_eq!(state.get_base_index(), 1, "the results open in the list");

        state.set_dataset_only(true);
        state.invoke_confirm();
        let task = controller.task_request.0.lock().unwrap().take().unwrap();
        assert_eq!(
            task.base_results,
            Some(PathBuf::from("/data/exp/results/x__older/older.evadb"))
        );

        // Reopened: a full analysis again; without dataset steps, no choice.
        ui_state.get_project_write().pipelines[0].steps[0].enabled = false;
        ui.global::<PipelinesPanelState>().invoke_run_all();
        assert!(!state.get_dataset_only());
        assert!(!state.get_has_dataset_steps());
        state.invoke_confirm();
        let task = controller.task_request.0.lock().unwrap().take().unwrap();
        assert_eq!(task.base_results, None);
    }

    #[test]
    fn dry_run_needs_a_saved_project_and_a_selected_image() {
        let (ui, ui_state, controller) = selected(vec![]);
        let dry_run = || {
            ui.global::<GlobalAppState>()
                .set_active_dialog(DialogType::None);
            ui.global::<PipelinesPanelState>().invoke_dry_run();
            drain_ui_queue();
        };
        dry_run();
        assert!(
            ui.global::<WarningState>()
                .get_message()
                .contains("No project is open")
        );

        ui_state.get_project_write().tmp_settings.current_project =
            Some(PathBuf::from("/data/project.evaproj"));
        dry_run();
        assert!(
            ui.global::<WarningState>()
                .get_message()
                .contains("No image is selected")
        );
        assert!(controller.task_request.0.lock().unwrap().is_none());

        // With an image selected a preview task is dispatched.
        {
            let mut project = ui_state.get_project_write();
            let mut with_image = crate::editor::test_support::project_with_one_image();
            with_image.tmp_settings.current_project = project.tmp_settings.current_project.clone();
            with_image.settings.pipelines = project.settings.pipelines.clone();
            *project = with_image;
        }
        *controller.breakpoint.lock().unwrap() = Some((
            1,
            evanalyzer_cfg::core_types::StepId(0),
            evanalyzer_cfg::core_types::BreakpointMode::Stop,
        ));
        dry_run();
        let task = controller
            .task_request
            .0
            .lock()
            .unwrap()
            .take()
            .expect("preview dispatched");
        assert!(task.preview);
        assert_eq!(task.project_settings.images.list.len(), 1);
        assert!(task.breakpoint.is_some());
    }

    #[test]
    fn running_dialog_cancel_and_close() {
        let (ui, _ui_state, controller) = selected(vec![]);
        let cancel = evanalyzer_app::analysis::CancelHandle::with_callback(|| {});
        *controller.pipeline_cancel_flag.lock().unwrap() = Some(cancel.clone());
        let running = ui.global::<PipelineRunningState>();
        running.invoke_cancel();
        assert!(cancel.is_cancelled());

        running.set_done(true);
        running.set_status_message("done".into());
        ui.global::<GlobalAppState>()
            .set_active_dialog(DialogType::PipelineRunning);
        running.invoke_close();
        assert!(!running.get_done());
        assert_eq!(running.get_status_message(), "");
        assert_eq!(active_dialog(&ui), DialogType::None);
    }

    #[test]
    fn disable_auto_preview_turns_the_toggle_off_in_the_ui() {
        let (ui, _ui_state, controller) = selected(vec![]);
        ui.global::<PipelinesPanelState>().invoke_auto_preview(true);
        ui.global::<PipelinesPanelState>()
            .set_auto_preview_enabled(true);
        controller.disable_auto_preview();
        drain_ui_queue();
        assert!(!*controller.auto_preview_enabled.lock().unwrap());
        assert!(
            !ui.global::<PipelinesPanelState>()
                .get_auto_preview_enabled()
        );
    }

    #[test]
    fn settings_changes_with_auto_preview_arm_the_debounce_instead_of_running_at_once() {
        let (ui, ui_state, controller) = selected(vec![step("Threshold")]);
        ui.global::<PipelinesPanelState>().invoke_auto_preview(true);
        ui.global::<PipelinesPanelState>().invoke_toggle_step(0);
        assert!(ui_state.is_dirty());
        assert!(
            controller.task_request.0.lock().unwrap().is_none(),
            "preview waits for the debounce timer"
        );
    }

    #[test]
    fn browse_file_and_bioimageio_import_do_nothing_when_cancelled() {
        // The test UiState's file browser has no window: it reports
        // "cancelled" right away.
        let (ui, ui_state, _controller) = selected(vec![]);
        let token_before = ui.global::<PipelinesPanelState>().get_browse_result_token();
        ui.global::<PipelinesPanelState>()
            .invoke_browse_file("pt, onnx".into(), "/tmp".into(), 3);
        assert_eq!(
            ui.global::<PipelinesPanelState>().get_browse_result_token(),
            token_before
        );
        ui.global::<CommandPickerState>().set_pipeline_id(1);
        ui.global::<CommandPickerState>().invoke_import_bioimageio();
        drain_ui_queue();
        assert!(step_names(&ui_state, 1).is_empty());
    }

    // ---- problems, wiring, viewing with values, keys ----

    /// Pipeline 1 uses SPOTS once: it extracts spots (its "Spot › Output
    /// class") and beads ("Bead › Output class"); the step gives the beads
    /// `bead_class`. The project has class 1 "Spot" and 2 "Bead".
    fn with_spots(bead_class: i64) -> (AppWindow, Arc<UiState>, Arc<PipelinesController>) {
        use evanalyzer_cfg::settings::classification_settings::Class;
        use evanalyzer_cfg::settings::components::{
            ComponentInstance, ComponentParameterDef, ParamBinding,
        };
        use evanalyzer_cfg::settings::pipeline_command_settings::{
            ExtractClassMappingSettings, ExtractObjectsSettings,
        };
        let (ui, ui_state, controller) = selected(vec![]);
        {
            let mut project = ui_state.get_project_write();
            for (id, name) in [(1, "Spot"), (2, "Bead")] {
                project.classification.classes_mut().push(Class {
                    id,
                    name: name.into(),
                    ..Default::default()
                });
            }
            let extract = |id: u32, class: u32| PipelineStepSettings {
                id: StepId(id),
                enabled: true,
                content: StepContent::Command(PipelineCommand::ExtractObjects(
                    ExtractObjectsSettings {
                        class_mapping: vec![ExtractClassMappingSettings {
                            segmentation_class: SegmentationClass(1),
                            object_class: ObjectClass::Global(class),
                        }],
                        ..Default::default()
                    },
                )),
            };
            let parameter =
                |key: &str, section: &str, default: i64, step: u32| ComponentParameterDef {
                    key: ComponentParamKey::from(key),
                    label: "Output class".into(),
                    description: String::new(),
                    section: section.into(),
                    default: ParamValue::Int(default),
                    bindings: vec![ParamBinding {
                        step: StepId(step),
                        param: "class_mapping.0.object_class".into(),
                    }],
                };
            project.components.push(ComponentDefinition {
                key: "SPOTS".into(),
                meta: MetaData {
                    name: "Spots".into(),
                    ..Default::default()
                },
                classes: vec![],
                steps: vec![extract(1, 1), extract(2, 2)],
                parameters: vec![
                    parameter("spot_output_class", "Spot", 1, 1),
                    parameter("bead_output_class", "Bead", 2, 2),
                ],
                origin: None,
            });
            project.pipelines[0].steps = vec![PipelineStepSettings {
                id: StepId(1),
                enabled: true,
                content: StepContent::Component(ComponentInstance {
                    key: "SPOTS".into(),
                    values: [(
                        ComponentParamKey::from("bead_output_class"),
                        ParamValue::Int(bead_class),
                    )]
                    .into(),
                }),
            }];
        }
        controller.sync_steps_of_selected_pipeline_to_slint(PipelineId(1), true);
        drain_ui_queue();
        (ui, ui_state, controller)
    }

    fn panel_problems(ui: &AppWindow) -> Vec<ProblemItem> {
        ui.global::<PipelinesPanelState>()
            .get_project_problems()
            .iter()
            .collect()
    }

    /// Beads given the spots' class: the card says so, at both parameters,
    /// and the panel lists it with where it is.
    #[test]
    fn a_problem_shows_on_the_card_at_its_parameters_and_in_the_panels_list() {
        let (ui, _ui_state, _controller) = with_spots(1);
        let card = ui
            .global::<PipelinesPanelState>()
            .get_active_commands()
            .row_data(0)
            .unwrap();
        let texts: Vec<String> = card.problems.iter().map(|p| p.text.to_string()).collect();
        assert_eq!(texts.len(), 1, "{texts:?}");
        assert!(
            texts[0].contains("both create objects of class 1 “Spot”"),
            "{}",
            texts[0]
        );
        assert_eq!(card.problem_errors, 0);
        for key in ["spot_output_class", "bead_output_class"] {
            let parameter = ui_param(&ui, 0, key);
            assert!(
                parameter.problem.contains("both create"),
                "{key}: {}",
                parameter.problem
            );
            assert!(!parameter.problem_error);
        }
        assert_eq!(
            ui_param(&ui, 0, "spot_output_class").wiring,
            "",
            "the wiring only in debug mode"
        );
        ui.global::<PipelinesPanelState>()
            .invoke_debug_mode_toggled(true);
        drain_ui_queue();
        assert_eq!(
            ui_param(&ui, 0, "spot_output_class").wiring,
            "Sets ExtractObjects › Class Mapping 1 · Object Class (step 1)"
        );
        let listed = panel_problems(&ui);
        assert_eq!(listed.len(), 1);
        assert_eq!(listed[0].r#where, "Pipeline 1 · #1");
        assert_eq!(listed[0].text, card.problems.row_data(0).unwrap().text);

        // Each its own class: nothing to show.
        let (ui, _ui_state, _controller) = with_spots(2);
        assert!(panel_problems(&ui).is_empty());
        assert_eq!(ui_param(&ui, 0, "bead_output_class").problem, "");
    }

    #[test]
    fn a_problems_fix_is_made_from_the_panel() {
        let (ui, ui_state, _controller) = with_spots(7);
        let listed = panel_problems(&ui);
        let missing = listed
            .iter()
            .find(|p| p.fix_label == "Add class 7")
            .unwrap_or_else(|| panic!("{listed:?}"));
        ui.global::<PipelinesPanelState>()
            .invoke_fix_problem(missing.index);
        drain_ui_queue();
        assert!(
            ui_state
                .get_project()
                .classification
                .classes()
                .iter()
                .any(|c| c.id == 7)
        );
        assert!(
            panel_problems(&ui)
                .iter()
                .all(|p| p.fix_label != "Add class 7"),
            "gone once fixed"
        );
    }

    #[test]
    fn showing_a_problem_expands_the_card_of_its_step() {
        let (ui, _ui_state, _controller) = with_spots(1);
        let panel = ui.global::<PipelinesPanelState>();
        assert!(!panel.get_active_commands().row_data(0).unwrap().expanded);
        panel.invoke_show_problem(panel_problems(&ui)[0].index);
        drain_ui_queue();
        assert_eq!(panel.get_active_pipeline_id(), 1);
        assert!(panel.get_active_commands().row_data(0).unwrap().expanded);
    }

    /// "View" on a card shows the component with that step's values - and
    /// the step's problems at the settings they're about.
    #[test]
    fn viewing_a_component_shows_the_values_of_the_step_its_viewed_from() {
        let (ui, _ui_state, _controller) = with_spots(1);
        let panel = ui.global::<PipelinesPanelState>();
        panel.invoke_view_component(0);
        drain_ui_queue();
        assert!(panel.get_open_component_read_only());
        assert_eq!(
            panel.get_open_component_values(),
            "Shown with the values of step 1 in this pipeline"
        );
        let class_of = |step: usize| -> String {
            let mapping = ui_param(&ui, step, "class_mapping");
            mapping
                .group_items
                .row_data(0)
                .unwrap()
                .fields
                .iter()
                .find(|f| f.name == "object_class")
                .map(|f| f.value.to_string())
                .unwrap()
        };
        assert_eq!(class_of(1), "1", "the step's value, not the default 2");
        let mapping = ui_param(&ui, 1, "class_mapping");
        assert!(
            mapping.problem.contains("both create"),
            "{}",
            mapping.problem
        );
        assert_eq!(mapping.wiring, "", "only in debug mode");
        panel.invoke_debug_mode_toggled(true);
        drain_ui_queue();
        let mapping = ui_param(&ui, 1, "class_mapping");
        assert!(
            mapping.wiring.contains("Set by “Output class” (Bead)"),
            "{}",
            mapping.wiring
        );

        // Opened to edit: the defaults.
        panel.invoke_close_component();
        drain_ui_queue();
        panel.invoke_open_component(0);
        drain_ui_queue();
        assert_eq!(panel.get_open_component_values(), "");
        assert_eq!(class_of(1), "2");
    }

    fn section_views(ui: &AppWindow, step: usize) -> Vec<SectionView> {
        ui.global::<PipelinesPanelState>()
            .get_active_commands()
            .row_data(step)
            .unwrap()
            .section_views
            .iter()
            .collect()
    }

    /// A component card's section shows its values on one line until it's
    /// opened - and stays open when the card is filled again.
    #[test]
    fn a_section_shows_its_values_until_its_opened() {
        let (ui, ui_state, controller) = with_a_component();
        {
            let mut project = ui_state.get_project_write();
            let definition = &mut project.components[0];
            let step = definition.steps[0].id;
            evanalyzer_app::components::toggle_exposed(
                definition,
                step,
                "kernel_size",
                "Kernel",
                "3",
            );
            let key = definition.parameters[0].key.clone();
            evanalyzer_app::components::set_parameter_section(definition, &key, "Staining 1");
        }
        controller.sync_steps_of_selected_pipeline_to_slint(PipelineId(1), true);
        drain_ui_queue();
        let views = section_views(&ui, 1);
        assert_eq!(views.len(), 1);
        assert_eq!(
            (views[0].name.as_str(), views[0].kind, views[0].open),
            ("Staining 1", 0, false)
        );
        assert_eq!(views[0].summary, "3");
        let command = section_views(&ui, 0);
        assert_eq!(
            command
                .iter()
                .map(|v| (v.name.as_str(), v.kind))
                .collect::<Vec<_>>(),
            [("", 3)],
            "a command card: its settings as rows, always shown"
        );

        ui.global::<PipelinesPanelState>()
            .invoke_toggle_section(1, "Staining 1".into());
        assert!(section_views(&ui, 1)[0].open);
        controller.sync_steps_of_selected_pipeline_to_slint(PipelineId(1), false);
        drain_ui_queue();
        assert!(section_views(&ui, 1)[0].open, "still open");
        ui.global::<PipelinesPanelState>()
            .invoke_toggle_section(1, "Staining 1".into());
        assert!(!section_views(&ui, 1)[0].open);
    }

    /// Sections with the same settings are one table - a row per section,
    /// its problems at the row and below the table.
    #[test]
    fn sections_with_the_same_settings_are_a_table() {
        let (ui, _ui_state, _controller) = with_spots(1);
        let views = section_views(&ui, 0);
        let kinds: Vec<(String, i32)> =
            views.iter().map(|v| (v.name.to_string(), v.kind)).collect();
        assert_eq!(kinds, [("Spot".to_string(), 1), ("Bead".to_string(), 2)]);
        let table = ui
            .global::<PipelinesPanelState>()
            .get_active_commands()
            .row_data(0)
            .unwrap()
            .card_tables
            .row_data(0)
            .unwrap();
        assert_eq!(
            table.columns.iter().collect::<Vec<_>>(),
            [SharedString::from("Output class")]
        );
        let rows: Vec<CardTableRow> = table.rows.iter().collect();
        assert_eq!(
            rows.iter().map(|r| r.name.to_string()).collect::<Vec<_>>(),
            ["Spot", "Bead"]
        );
        assert_eq!(rows[1].cells.row_data(0).unwrap().value, "1");
        assert!(rows.iter().all(|r| r.problem == 1), "both rows collide");
        assert_eq!(table.problems.row_count(), 1);
        assert!(table.problems.row_data(0).unwrap().contains("both create"));

        // Each its own class: no marks.
        let (ui, _ui_state, _controller) = with_spots(2);
        let table = ui
            .global::<PipelinesPanelState>()
            .get_active_commands()
            .row_data(0)
            .unwrap()
            .card_tables
            .row_data(0)
            .unwrap();
        assert_eq!(table.problems.row_count(), 0);
        assert!(table.rows.iter().all(|r| r.problem == 0));
    }

    /// An expanded card draws its table: a row per section, side by side
    /// below the column's name.
    #[test]
    fn an_expanded_card_draws_its_table() {
        use i_slint_backend_testing::ElementHandle;
        let (ui, _ui_state, _controller) = with_spots(2);
        ui.window().set_size(slint::LogicalSize::new(1280.0, 900.0));
        ui.global::<GlobalAppState>().set_sidebar_left_active_tab(2);
        let commands = ui.global::<PipelinesPanelState>().get_active_commands();
        let mut card = commands.row_data(0).unwrap();
        card.expanded = true;
        commands.set_row_data(0, card);
        let at = |label: &str| {
            let e = ElementHandle::find_by_accessible_label(&ui, label)
                .next()
                .unwrap_or_else(|| panic!("{label:?} not found"));
            e.absolute_position()
        };
        let (column, spot, bead) = (at("Output class"), at("Spot"), at("Bead"));
        assert!(
            column.y < spot.y && spot.y < bead.y,
            "{column:?} {spot:?} {bead:?}"
        );
        assert_eq!(spot.x, bead.x, "rows one below the other");
    }

    /// The table's cells share its width - each wide enough to use (a cell
    /// used to be 1px wide, its column's name elided to nothing).
    #[test]
    fn a_tables_cells_share_its_width() {
        use i_slint_backend_testing::ElementHandle;
        let (ui, _ui_state, _controller) = with_spots(2);
        ui.window().set_size(slint::LogicalSize::new(1280.0, 900.0));
        ui.global::<GlobalAppState>().set_sidebar_left_active_tab(2);
        let commands = ui.global::<PipelinesPanelState>().get_active_commands();
        let mut card = commands.row_data(0).unwrap();
        card.expanded = true;
        commands.set_row_data(0, card);
        let column = ElementHandle::find_by_accessible_label(&ui, "Output class")
            .next()
            .expect("the column's name");
        assert!(
            column.size().width > 60.0,
            "column {}px",
            column.size().width
        );
        let fields: Vec<f32> = ElementHandle::find_by_accessible_label(&ui, "Spot")
            .chain(ElementHandle::find_by_accessible_label(&ui, "Bead"))
            .map(|e| e.size().width)
            .collect();
        assert!(!fields.is_empty());
    }

    /// A command card shows its settings as rows: the name left, the
    /// control right, on one line.
    #[test]
    fn a_command_cards_settings_are_rows() {
        use i_slint_backend_testing::ElementHandle;
        let (ui, _ui_state, _controller) = selected(vec![step("Blur")]);
        ui.window().set_size(slint::LogicalSize::new(1280.0, 900.0));
        ui.global::<GlobalAppState>().set_sidebar_left_active_tab(2);
        let commands = ui.global::<PipelinesPanelState>().get_active_commands();
        let mut card = commands.row_data(0).unwrap();
        card.expanded = true;
        commands.set_row_data(0, card);
        let name = ui_param(&ui, 0, "kernel_size").display_name.to_string();
        let label = ElementHandle::find_by_accessible_label(&ui, &name)
            .next()
            .unwrap_or_else(|| panic!("{name}"));
        let field = ElementHandle::find_by_accessible_label(&ui, "Increase")
            .next()
            .expect("the spinner's +");
        let middle = |e: &ElementHandle| e.absolute_position().y + e.size().height / 2.0;
        assert!(
            (middle(&label) - middle(&field)).abs() < 8.0,
            "label at {}, field at {}",
            middle(&label),
            middle(&field)
        );
        assert!(label.absolute_position().x < field.absolute_position().x);
    }

    /// A changed setting is patched into its card in place - the rows show
    /// copies of the settings, and they take the new value too.
    #[test]
    fn a_changed_setting_reaches_the_row_showing_it() {
        let (ui, _ui_state, _controller) = selected(vec![step("Blur")]);
        let panel = ui.global::<PipelinesPanelState>();
        panel.invoke_param_changed(0, "kernel_size".into(), "7".into());
        drain_ui_queue();
        let card = panel.get_active_commands().row_data(0).unwrap();
        let shown = card
            .section_views
            .row_data(0)
            .unwrap()
            .params
            .iter()
            .find(|p| p.name == "kernel_size")
            .expect("in the row group");
        assert_eq!(shown.value, ui_param(&ui, 0, "kernel_size").value);
    }

    /// A changed setting only patches its card - the problems are checked
    /// again all the same: fixed, they're gone from the card, its settings
    /// and the panel's list.
    #[test]
    fn changing_a_setting_checks_the_problems_again() {
        let (ui, _ui_state, _controller) = with_spots(1);
        assert_eq!(panel_problems(&ui).len(), 1);
        ui.global::<PipelinesPanelState>().invoke_param_changed(
            0,
            "bead_output_class".into(),
            "2".into(),
        );
        drain_ui_queue();
        assert!(panel_problems(&ui).is_empty(), "{:?}", panel_problems(&ui));
        let card = ui
            .global::<PipelinesPanelState>()
            .get_active_commands()
            .row_data(0)
            .unwrap();
        assert_eq!(card.problems.row_count(), 0);
        assert_eq!(ui_param(&ui, 0, "spot_output_class").problem, "");
        assert_eq!(ui_param(&ui, 0, "bead_output_class").value, "2");

        // And back.
        ui.global::<PipelinesPanelState>().invoke_param_changed(
            0,
            "bead_output_class".into(),
            "1".into(),
        );
        drain_ui_queue();
        assert_eq!(panel_problems(&ui).len(), 1);
        assert!(
            ui_param(&ui, 0, "bead_output_class")
                .problem
                .contains("both create")
        );
    }

    /// Debug mode shows what only who looks into components can act on: a
    /// component's own warnings, and where inside it a problem is.
    #[test]
    fn debug_mode_shows_a_components_own_warnings_and_where_inside_problems_are() {
        let (ui, ui_state, controller) = with_spots(7);
        ui_state.get_project_write().components[0].parameters.push(
            evanalyzer_cfg::settings::components::ComponentParameterDef {
                key: ComponentParamKey::from("unused"),
                label: "Unused".into(),
                description: String::new(),
                section: String::new(),
                default: ParamValue::Int(0),
                bindings: vec![],
            },
        );
        controller.sync_steps_of_selected_pipeline_to_slint(PipelineId(1), false);
        drain_ui_queue();
        let texts = |ui: &AppWindow| -> Vec<String> {
            panel_problems(ui)
                .iter()
                .map(|p| p.text.to_string())
                .collect()
        };
        assert_eq!(
            texts(&ui),
            ["“Output class” (Bead) is class 7, which the project doesn't have"]
        );
        ui.global::<PipelinesPanelState>()
            .invoke_debug_mode_toggled(true);
        drain_ui_queue();
        let all = texts(&ui);
        assert!(
            all.contains(
                &"“Output class” (Bead) is class 7, which the project doesn't have · in \
                  ExtractObjects (its step 2)"
                    .to_string()
            ),
            "{all:?}"
        );
        assert!(
            all.iter().any(|t| t.contains("\"Unused\" sets nothing")),
            "{all:?}"
        );
    }

    #[test]
    fn debug_mode_is_remembered() {
        let (ui, ui_state, _controller) = with_spots(1);
        let panel = ui.global::<PipelinesPanelState>();
        assert!(!panel.get_debug_mode());
        panel.invoke_debug_mode_toggled(true);
        drain_ui_queue();
        assert!(panel.get_debug_mode());
        assert!(ui_state.load_app_settings().debug_mode);
        panel.invoke_debug_mode_toggled(false);
        drain_ui_queue();
        assert!(!ui_state.load_app_settings().debug_mode);
        assert_eq!(ui_param(&ui, 0, "spot_output_class").wiring, "");
    }

    /// The problem strip is as high as its content: it doesn't stretch and
    /// take height from the step list - an open list used to push the
    /// cards down with a gap, more the taller the window.
    #[test]
    fn the_problem_strip_doesnt_stretch() {
        use i_slint_backend_testing::ElementHandle;
        let (ui, _ui_state, _controller) = with_spots(7);
        ui.global::<GlobalAppState>().set_sidebar_left_active_tab(2);
        let commands = ui.global::<PipelinesPanelState>().get_active_commands();
        let mut card = commands.row_data(0).unwrap();
        card.expanded = true;
        commands.set_row_data(0, card);
        let y_of = |label: &str| {
            ElementHandle::find_by_accessible_label(&ui, label)
                .next()
                .unwrap_or_else(|| panic!("{label}"))
                .absolute_position()
                .y
        };
        ui.global::<PipelinesPanelState>()
            .set_problems_expanded(true);
        let at = |height: f32| {
            ui.window()
                .set_size(slint::LogicalSize::new(1280.0, height));
            (y_of("Problems"), y_of("1 problems"))
        };
        let short = at(500.0);
        assert_eq!(at(700.0), short, "the strip's head and the card below it");
        assert_eq!(at(900.0), short);
    }

    /// The panel shows the problems: a strip with their count (its list
    /// opens on a click, a fix as a button), a badge on the card.
    #[test]
    fn the_panel_shows_a_problem_strip_and_a_badge_on_the_card() {
        use i_slint_backend_testing::ElementHandle;
        let (ui, _ui_state, _controller) = with_spots(7);
        ui.window().set_size(slint::LogicalSize::new(1280.0, 900.0));
        ui.global::<GlobalAppState>().set_sidebar_left_active_tab(2);
        let count = |label: &str| ElementHandle::find_by_accessible_label(&ui, label).count();
        assert_eq!(count("Problems"), 1, "the strip");
        assert_eq!(count("1 problems"), 1, "the card's badge");
        assert_eq!(count("Add class 7"), 0, "the list is closed");
        ui.global::<PipelinesPanelState>()
            .set_problems_expanded(true);
        // The button, and its text.
        assert!(count("Add class 7") > 0, "the fix in the open list");
    }

    /// A parameter's key follows its section and label once the settings
    /// dialog is closed - and the pipeline's values follow it.
    #[test]
    fn closing_the_settings_dialog_gives_parameters_keys_from_section_and_label() {
        let (ui, ui_state, _controller) = with_spots(5);
        let panel = ui.global::<PipelinesPanelState>();
        panel.invoke_open_component(0);
        drain_ui_queue();
        panel.invoke_open_component_settings();
        drain_ui_queue();
        let dialog = ui.global::<ComponentSettingsState>();
        dialog.invoke_parameter_name_edited("bead_output_class".into(), "Class".into());
        drain_ui_queue();
        assert_eq!(
            ui_state.get_project().components[0].parameters[1]
                .key
                .as_str(),
            "bead_output_class",
            "not while the dialog is open"
        );
        dialog.invoke_close();
        drain_ui_queue();
        let project = ui_state.get_project();
        assert_eq!(
            project.components[0].parameters[1].key.as_str(),
            "bead_class"
        );
        let StepContent::Component(instance) = &project.pipelines[0].steps[0].content else {
            panic!("a component step");
        };
        assert_eq!(
            instance.values.get(&ComponentParamKey::from("bead_class")),
            Some(&ParamValue::Int(5))
        );
    }
}

/// How an analysis ended, for the "ended while away" notice.
fn describe_end(state: &evanalyzer_app::analysis::JobState) -> String {
    use evanalyzer_app::analysis::JobState;
    match state {
        JobState::Succeeded => "finished".into(),
        JobState::Cancelled => "cancelled".into(),
        JobState::Failed(message) => format!("failed ({message})"),
        JobState::Running { .. } => "running".into(),
    }
}
