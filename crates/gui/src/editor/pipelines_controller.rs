use crate::AppWindow;
use crate::DialogType;
use crate::FileRequest;
use crate::editor::focus_controller::FocusController;
use crate::editor::object_list_controller::ObjectListController;
use crate::editor::pipeline_task::PipelineTask;
use crate::editor::template_controller::TemplateController;
use crate::editor::viewport_controller::ViewportController;
use crate::{
    CommandDef, CommandParameter, CommandPickerState, GlobalAppState, GroupItem, LeafParam,
    ParamType, Pipeline, PipelineCommand as SlintPipelineCommand, PipelineStatus,
    PipelinesPanelState, RunAnalysisState, StepCategory, UiState, WarningState,
};
use crate::{PipelineDeleteConfirmState, PipelineEditState, PipelineRunningState};
use evanalyzer_app::ai_learning::load_classifier_settings;
use evanalyzer_app::fs::FileSystem;
use evanalyzer_app::project::ProjectExt;
use evanalyzer_app::templates::load_pipeline_templates;
use evanalyzer_cfg::core_types::MemorySlot;
use evanalyzer_cfg::core_types::ObjectClass;
use evanalyzer_cfg::core_types::PipelineId;
use evanalyzer_cfg::core_types::SegmentationClass;
use evanalyzer_cfg::core_types::{ImageAddress, MemoryId};
use evanalyzer_cfg::settings::ai_learning_settings::{
    AiLearningClassifierSettings, AiLearningSettings, ObjectClassLabel, PixelClassLabel,
};
use evanalyzer_cfg::settings::images_settings::{
    ImageEntry, ImageSettings, TStackHandling, TStackSettings,
};
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
use evanalyzer_cfg::settings::pipeline_settings::{PipelineSettings, PipelineStepSettings};
use evanalyzer_cfg::settings::project_settings::ProjectSettings;
use evanalyzer_cfg::settings::templates::PipelineTemplate;
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
    /// Currently active breakpoint: (pipeline_id, step_id, mode).  `None` = no breakpoint.
    pub(crate) breakpoint:
        Arc<Mutex<Option<(u32, i32, evanalyzer_cfg::core_types::BreakpointMode)>>>,

    /// If true the trigger_pipeline_preview_execution is called on parameter change
    auto_preview_enabled: Mutex<bool>,

    /// Pipeline templates currently shown in the command picker's "Templates"
    /// section. Reloaded from disk whenever the picker is opened.
    pipeline_templates: Mutex<Vec<PipelineTemplate>>,

    /// Pipeline focus, told about every pipeline change (see
    /// `pipeline_settings_changed`).
    pub(crate) focus_controller: Arc<FocusController>,
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
            breakpoint: Arc::new(Mutex::new(None)),
            auto_preview_enabled: Mutex::new(false),
            pipeline_templates: Mutex::new(Vec::new()),
            focus_controller,
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

            // Save as template
            let manager = self.clone();
            ui.global::<PipelinesPanelState>()
                .on_save_as_template(move || {
                    manager.save_pipeline_as_template();
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
                    *manager.breakpoint.lock().unwrap() =
                        Some((pipeline_id as u32, step_id, bp_mode));
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
                                    ImageAddress::Memory(MemoryId::ProjectCache(s)) => {
                                        (1, s as i32, 0)
                                    }
                                    ImageAddress::Channel(c) => (2, 1, c),
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
                    1 => ImageAddress::Memory(MemoryId::PipelineContext(slot.max(1))),
                    2 => ImageAddress::Channel(ch),
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
                        ImageAddress::Memory(MemoryId::PipelineContext(s)) => {
                            format!("Memory[{s}]").into()
                        }
                        ImageAddress::Memory(MemoryId::ProjectCache(s)) => {
                            format!("Cache[{s}]").into()
                        }
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
                    {
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
                    {
                        let mut project = manager.app_state.get_project_write();
                        if let Some(pipeline) =
                            project.pipelines.iter_mut().find(|p| p.id.0 == pipeline_id)
                        {
                            let idx = step_idx as usize;
                            if idx < pipeline.steps.len() {
                                pipeline.steps.remove(idx);
                            }
                        }
                    }
                    manager.pipeline_settings_changed();
                    manager.sync_steps_of_selected_pipeline_to_slint(PipelineId(pipeline_id), true);
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
                    {
                        let mut project = manager.app_state.get_project_write();
                        if let Some(pipeline) =
                            project.pipelines.iter_mut().find(|p| p.id.0 == pipeline_id)
                        {
                            let idx = step_idx as usize;
                            if idx < pipeline.steps.len() {
                                let cloned = pipeline.steps[idx].clone();
                                pipeline.steps.insert(idx + 1, cloned);
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
                            let (ctx_cat, suggested) =
                                if p.steps.is_empty() || context_idx == usize::MAX {
                                    (-1i32, [false; 5])
                                } else {
                                    let prev = &p.steps[context_idx].command;
                                    let ctx_order = prev.category().display_order() as i32;
                                    let mut flags = [false; 5];
                                    for c in prev.allowed_next() {
                                        flags[c.display_order() as usize] = true;
                                    }
                                    (ctx_order, flags)
                                };
                            (name, p.steps.len() as i32, ctx_cat, suggested)
                        } else {
                            (String::new(), 0, -1i32, [false; 5])
                        }
                    };
                    manager.reload_pipeline_templates_async();
                    if let Some(ui) = manager.ui.upgrade() {
                        let picker = ui.global::<CommandPickerState>();
                        picker.set_pipeline_id(pipeline_id);
                        picker.set_insert_after_idx(step_after_idx);
                        picker.set_target_pipeline(pipeline_name.into());
                        picker.set_total_steps(total_steps);
                        picker.set_context_category(context_cat);
                        picker.set_query("".into());
                        picker.set_filter_favorites(false);
                        // Pre-select the suggested next categories (all of them); an
                        // empty set leaves the "All" view.
                        picker.set_fcat_pre(suggested_filter[0]);
                        picker.set_fcat_seg(suggested_filter[1]);
                        picker.set_fcat_obj(suggested_filter[2]);
                        picker.set_fcat_mea(suggested_filter[3]);
                        picker.set_fcat_cls(suggested_filter[4]);
                        picker.set_selected_id(-1);
                        // Apply the filter immediately so the list matches the chips.
                        manager.apply_picker_filter(&ui, "", false);
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
                    let picker = ui.global::<CommandPickerState>();
                    let filter_favorites = picker.get_filter_favorites();
                    manager.apply_picker_filter(&ui, query.as_str(), filter_favorites);
                });

            // Picker: select - update detail pane
            let manager = self.clone();
            ui.global::<CommandPickerState>()
                .on_select(move |command_id| {
                    let Some(ui) = manager.ui.upgrade() else {
                        return;
                    };
                    if command_id < 0 {
                        // Pipeline template entry.
                        let templates = manager.pipeline_templates.lock().expect("Poisoned");
                        let idx = (-command_id - 1) as usize;
                        if let Some(t) = templates.get(idx) {
                            let detail = template_to_command_def(idx, t);
                            let picker = ui.global::<CommandPickerState>();
                            picker.set_detail(detail);
                            picker.set_has_detail(true);
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

                    let new_steps: Vec<PipelineStepSettings> = if command_id < 0 {
                        // Pipeline template entry: insert all of its steps.
                        let templates = manager.pipeline_templates.lock().expect("Poisoned");
                        let idx = (-command_id - 1) as usize;
                        let Some(template) = templates.get(idx) else {
                            warn!("picker confirm: unknown template id {}", command_id);
                            return;
                        };
                        template.steps.clone()
                    } else {
                        let Some(cmd) = default_command(command_id) else {
                            warn!("picker confirm: unknown command id {}", command_id);
                            return;
                        };
                        vec![PipelineStepSettings {
                            enabled: true,
                            command: cmd,
                        }]
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
                            pipeline.steps.splice(insert_at..insert_at, new_steps);
                        }
                    }
                    ui.global::<GlobalAppState>()
                        .set_active_dialog(DialogType::None);
                    manager.pipeline_settings_changed();
                    manager
                        .sync_steps_of_selected_pipeline_to_slint(PipelineId(pipeline_id), false);
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
                                            enabled: true,
                                            command: configured.command,
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
                    let param_name = parameter_name.as_str().to_owned();
                    let value_str = value.as_str().to_owned();

                    // Apply the change and capture the new summary + the updated value
                    // (so we can write it back into the UI model - otherwise the next
                    // re-bind from set_row_data would revert the displayed value to the
                    // pre-change one and dropdown/spinner selections appear "lost").
                    //
                    // Parameter names are either flat ("kernel_size", "criteria.min_area")
                    // or group-item paths ("thresholds.0.method") where the middle segment
                    // is a numeric index into the group. We resolve both shapes from the
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

                    let (new_summary, params_now, needs_full_resync) = {
                        let mut project = manager.app_state.get_project_write();
                        let Some(pipeline) =
                            project.pipelines.iter_mut().find(|p| p.id.0 == pipeline_id)
                        else {
                            return;
                        };
                        let Some(step) = pipeline.steps.get_mut(step_idx as usize) else {
                            return;
                        };
                        step.command.apply_param_change(&param_name, &value_str);

                        // A PixelClassifier/AiObjectClassifier's `segmentation_mapping`
                        // row count is driven by the loaded model's own class count,
                        // not something the user adds/removes rows for - resize it to
                        // match whenever the model changes. This changes the group's
                        // row count (not just one value), so the single-value patch
                        // below can't reflect it; fall back to a full resync instead,
                        // same as a changed field set does.
                        let mut needs_full_resync = false;
                        if param_name == "model_path" {
                            match &mut step.command {
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

                        (
                            step.command.to_summary(),
                            step.command.to_parameters(),
                            needs_full_resync,
                        )
                    }; // write lock dropped here

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
                                            let new_flags: Vec<SharedString> = (0u32..33u32)
                                                .map(|idx| {
                                                    if selected.contains(&idx) {
                                                        "1".into()
                                                    } else {
                                                        "0".into()
                                                    }
                                                })
                                                .collect();
                                            p.options = ModelRc::new(VecModel::from(new_flags));
                                        }
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
        let effective_t = image_settings
            .series
            .get(&image_settings.selected_series)
            .and_then(|series| series.t_stack.clone())
            .or_else(|| project.images.settings.t_stack.clone())
            .unwrap_or_default();
        let single_stack_t = TStackSettings {
            stack_handling: TStackHandling::SingleStack,
            ..effective_t
        };
        if let Some(series) = image_settings
            .series
            .get_mut(&image_settings.selected_series)
        {
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
        let playback_speed = settings
            .images
            .settings
            .t_stack
            .as_ref()
            .map(|t| t.playback_speed)
            .unwrap_or_default();
        settings.images.settings.t_stack = Some(TStackSettings {
            stack_handling: TStackHandling::AllStacks,
            playback_speed,
            t_stack: 0,
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
        ui.global::<RunAnalysisState>().set_job_name("".into());
        ui.global::<GlobalAppState>()
            .set_active_dialog(DialogType::RunAnalysisConfirm);
    }

    /// The "Run Analysis" dialog was confirmed: read back the (optional) job
    /// name, close the dialog, and start the run.
    fn on_run_analysis_confirmed(self: &Arc<Self>) {
        let Some(ui) = self.ui.upgrade() else {
            return;
        };
        let job_name = ui.global::<RunAnalysisState>().get_job_name().to_string();
        ui.global::<GlobalAppState>()
            .set_active_dialog(DialogType::None);

        let job_name = (!job_name.trim().is_empty()).then_some(job_name);
        self.trigger_pipeline_full_run(job_name);
    }

    fn trigger_pipeline_full_run(&self, job_name: Option<String>) {
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

        let task: PipelineTask = PipelineTask {
            project_settings,
            project_path: current_project
                .parent()
                .unwrap_or(current_project)
                .to_path_buf(),
            preview: false,
            breakpoint: None,
            job_name,
        };
        drop(project);

        if let Some(ui) = self.app_state.ui_handle.upgrade() {
            ui.global::<PipelineRunningState>().set_processed(0);
            ui.global::<PipelineRunningState>().set_total(0);
            ui.global::<GlobalAppState>()
                .set_active_dialog(DialogType::PipelineRunning);
        }

        self.dispatch_worker_task(task);
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
        self.app_state.mark_dirty();
        self.focus_controller.refresh();

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

    /// Shows the generic warning dialog with `message`.
    ///
    /// Used at the early-return guard points in the pipeline trigger functions so
    /// the user gets visible feedback instead of a silent log-only `warn!`.
    fn show_warning(&self, message: &str) {
        let message = message.to_owned();
        let ui_weak = self.ui.clone();
        if let Err(e) = crate::helper::ui_thread::invoke_from_event_loop(move || {
            if let Some(ui) = ui_weak.upgrade() {
                let warning = ui.global::<WarningState>();
                warning.set_info(false);
                warning.set_title("Cannot start analysis".into());
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
    fn show_classifier_model_info(self: &Arc<Self>, step_idx: usize) {
        let Some(ui) = self.ui.upgrade() else {
            return;
        };
        let pipeline_id = ui.global::<PipelinesPanelState>().get_active_pipeline_id() as u32;

        let model_path = {
            let project = self.app_state.get_project();
            let Some(pipeline) = project.pipelines.iter().find(|p| p.id.0 == pipeline_id) else {
                return;
            };
            let Some(step) = pipeline.steps.get(step_idx) else {
                return;
            };
            match &step.command {
                PipelineCommand::PixelClassifier(settings) => settings.model_path.clone(),
                PipelineCommand::AiObjectClassifier(settings) => settings.model_path.clone(),
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
            if let Some(pipeline) = project.pipelines.iter_mut().find(|p| p.id.0 == pipeline_id) {
                if let Some(step) = pipeline.steps.get_mut(step_idx) {
                    if add {
                        step.command.add_group_item(&param_name);
                    } else if let Some(idx) = remove_idx {
                        step.command.remove_group_item(&param_name, idx);
                    }
                }
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
                            ImageAddress::Memory(MemoryId::PipelineContext(s)) => {
                                format!("Memory[{s}]").into()
                            }
                            ImageAddress::Memory(MemoryId::ProjectCache(s)) => {
                                format!("Cache[{s}]").into()
                            }
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

    fn apply_picker_filter(self: &Arc<Self>, ui: &AppWindow, query: &str, filter_favorites: bool) {
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
        let (fc_pre, fc_seg, fc_obj, fc_mea, fc_cls) = (
            picker_state.get_fcat_pre(),
            picker_state.get_fcat_seg(),
            picker_state.get_fcat_obj(),
            picker_state.get_fcat_mea(),
            picker_state.get_fcat_cls(),
        );
        let any_active = fc_pre || fc_seg || fc_obj || fc_mea || fc_cls;
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
        let templates_lock = self.pipeline_templates.lock().expect("Poisoned");
        let templates: Vec<CommandDef> = templates_lock
            .iter()
            .enumerate()
            .filter(|(_, t)| {
                q.is_empty()
                    || t.meta.name.to_ascii_lowercase().contains(&q)
                    || t.meta.short_description.to_ascii_lowercase().contains(&q)
            })
            .map(|(idx, t)| template_to_command_def(idx, t))
            .collect();
        drop(templates_lock);
        let cf = templates.len() as i32;

        let total = if filter_favorites {
            cf
        } else {
            (pre.len() + seg.len() + obj.len() + mea.len() + cls.len()) as i32
        };
        let (cp, cs, co, cm, cc) = (
            pre.len() as i32,
            seg.len() as i32,
            obj.len() as i32,
            mea.len() as i32,
            cls.len() as i32,
        );
        let picker = ui.global::<CommandPickerState>();
        picker.set_shown_preprocess(ModelRc::new(VecModel::from(pre)));
        picker.set_shown_segment(ModelRc::new(VecModel::from(seg)));
        picker.set_shown_object(ModelRc::new(VecModel::from(obj)));
        picker.set_shown_measure(ModelRc::new(VecModel::from(mea)));
        picker.set_shown_classify(ModelRc::new(VecModel::from(cls)));
        picker.set_shown_templates(ModelRc::new(VecModel::from(templates)));
        picker.set_total_shown(total);
        picker.set_cat_count_pre(cp);
        picker.set_cat_count_seg(cs);
        picker.set_cat_count_obj(co);
        picker.set_cat_count_mea(cm);
        picker.set_cat_count_cls(cc);
        picker.set_cat_count_fav(cf);
    }

    /// Reloads pipeline templates from the user and app templates folders in the
    /// background, then refreshes the command picker if it is still open.
    ///
    /// Scanning the templates folders involves blocking filesystem I/O which can
    /// be slow (e.g. on Windows with antivirus scanning or network drives), so it
    /// must not run on the Slint event-loop thread - doing so would freeze the UI
    /// while the "Add Step" dialog opens. The dialog opens immediately with the
    /// previously cached templates and is updated in place once the reload
    /// completes.
    fn reload_pipeline_templates_async(self: &Arc<Self>) {
        let manager = self.clone();
        crate::helper::ui_thread::spawn(move || {
            let templates: Vec<PipelineTemplate> =
                load_pipeline_templates(manager.app_state.backend().as_ref())
                    .into_iter()
                    .map(|(_path, template)| template)
                    .collect();
            *manager.pipeline_templates.lock().expect("Poisoned") = templates;

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
                let filter_favorites = picker.get_filter_favorites();
                manager.apply_picker_filter(&ui, &query, filter_favorites);
            }) {
                warn!("Failed to refresh pipeline templates in picker: {}", e);
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
                let total = all.len() as i32;
                let (cp, cs, co, cm, cc) = (
                    shown_pre.len() as i32,
                    shown_seg.len() as i32,
                    shown_obj.len() as i32,
                    shown_mea.len() as i32,
                    shown_cls.len() as i32,
                );
                let picker = ui.global::<CommandPickerState>();
                picker.set_all_commands(ModelRc::new(VecModel::from(all)));
                picker.set_shown_preprocess(ModelRc::new(VecModel::from(shown_pre)));
                picker.set_shown_segment(ModelRc::new(VecModel::from(shown_seg)));
                picker.set_shown_object(ModelRc::new(VecModel::from(shown_obj)));
                picker.set_shown_measure(ModelRc::new(VecModel::from(shown_mea)));
                picker.set_shown_classify(ModelRc::new(VecModel::from(shown_cls)));
                picker.set_total_shown(total);
                picker.set_cat_count_pre(cp);
                picker.set_cat_count_seg(cs);
                picker.set_cat_count_obj(co);
                picker.set_cat_count_mea(cm);
                picker.set_cat_count_cls(cc);
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
            name: String,
            summary: String,
            category: StepCategory,
            enabled: bool,
            parameters: Vec<ParameterDef>,
        }

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
                    ImageAddress::Memory(MemoryId::PipelineContext(s)) => format!("Memory[{s}]"),
                    ImageAddress::Memory(MemoryId::ProjectCache(s)) => format!("Cache[{s}]"),
                    ImageAddress::Channel(c) => format!("Channel {c}"),
                };
                let steps: Vec<StepData> = pipeline
                    .steps
                    .iter()
                    .enumerate()
                    .map(|(idx, step)| StepData {
                        idx: idx as i32,
                        name: step.command.name().to_owned(),
                        summary: step.command.to_summary(),
                        category: match step.command.category() {
                            CommandCategory::Preprocess => StepCategory::Preprocess,
                            CommandCategory::Segment => StepCategory::Segment,
                            CommandCategory::InstanceSegmentation => {
                                StepCategory::InstanceSegmentation
                            }
                            CommandCategory::Measure => StepCategory::Measure,
                            CommandCategory::Object => StepCategory::Object,
                        },
                        enabled: step.enabled,
                        parameters: step.command.to_parameters(),
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
        let backend = Arc::clone(self.app_state.backend());

        if let Err(e) = crate::helper::ui_thread::invoke_from_event_loop(move || {
            if let Some(ui) = ui_weak.upgrade() {
                let state = ui.global::<PipelinesPanelState>();

                let expanded_by_id: std::collections::HashMap<i32, bool> = if !reset_expanded {
                    let current = state.get_active_commands();
                    (0..current.row_count())
                        .filter_map(|i| current.row_data(i).map(|cmd| (cmd.id, cmd.expanded)))
                        .collect()
                } else {
                    std::collections::HashMap::new()
                };
                // Same for each step's expanded advanced section.
                let show_advanced_by_id: std::collections::HashMap<i32, bool> = {
                    let current = state.get_active_commands();
                    (0..current.row_count())
                        .filter_map(|i| current.row_data(i).map(|cmd| (cmd.id, cmd.show_advanced)))
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
                        // `segmentation_mapping`'s `segmentation_class`/`object_class`
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

                        let step_show_advanced =
                            show_advanced_by_id.get(&d.idx).copied().unwrap_or(false);
                        let show_advanced = always_show_advanced || step_show_advanced;
                        let (advanced_count, advanced_changed) = advanced_stats(&d.parameters);
                        let (advanced_defs, basic_defs): (Vec<ParameterDef>, Vec<ParameterDef>) =
                            filter_for_display(d.parameters, show_advanced)
                                .into_iter()
                                .partition(|p| p.advanced);

                        let convert = |p: ParameterDef| -> CommandParameter {
                            let p_name = p.name.clone();
                            let group_items: Vec<GroupItem> = p
                                .groups
                                .into_iter()
                                .map(|group| GroupItem {
                                    fields: ModelRc::new(VecModel::from(
                                        group
                                            .into_iter()
                                            .map(|lp| {
                                                let relabeled = p_name == "segmentation_mapping"
                                                    && ((is_pixel_classifier
                                                        && lp.name == "segmentation_class")
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
                                                        map_cfg_param_type(lp.param_type),
                                                        lp.value.clone(),
                                                        lp.options.clone(),
                                                    ),
                                                };
                                                LeafParam {
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
                            CommandParameter {
                                name: p.name.into(),
                                display_name: p.display_name.into(),
                                description: p.description.into(),
                                value: p.value.into(),
                                param_type: map_cfg_param_type(p.param_type),
                                options: ModelRc::new(VecModel::from(
                                    p.options
                                        .into_iter()
                                        .map(SharedString::from)
                                        .collect::<Vec<_>>(),
                                )),
                                min: p.min,
                                max: p.max,
                                step: p.step,
                                group_items: ModelRc::new(VecModel::from(group_items)),
                                has_model_info,
                            }
                        };
                        let params: Vec<CommandParameter> =
                            basic_defs.into_iter().map(&convert).collect();
                        let advanced_params: Vec<CommandParameter> = if show_advanced {
                            advanced_defs.into_iter().map(&convert).collect()
                        } else {
                            Vec::new()
                        };
                        SlintPipelineCommand {
                            id: d.idx,
                            name: d.name.into(),
                            category: d.category,
                            summary: d.summary.into(),
                            enabled: d.enabled,
                            expanded: expanded_by_id.get(&d.idx).copied().unwrap_or(false),
                            parameters: ModelRc::new(VecModel::from(params)),
                            advanced_parameters: ModelRc::new(VecModel::from(advanced_params)),
                            show_advanced: step_show_advanced,
                            advanced_count: advanced_count as i32,
                            advanced_changed: advanced_changed as i32,
                        }
                    })
                    .collect();
                state.set_active_pipeline_name(pipeline_name.into());
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

    /// Opens the "Save as Template" flow for the currently active pipeline.
    fn save_pipeline_as_template(self: &Arc<Self>) {
        let Some(ui) = self.ui.upgrade() else {
            return;
        };
        let panel = ui.global::<PipelinesPanelState>();
        let pipeline_id = PipelineId(panel.get_active_pipeline_id() as u32);
        let name = panel.get_active_pipeline_name().to_string();

        self.template_controller
            .start_pipeline_template_save(pipeline_id, name);
    }
}

/// Builds the `CommandDef` shown in the command picker's "Templates" section
/// for a loaded `PipelineTemplate`.
///
/// Picker ids for templates are encoded as negative numbers (`-(idx + 1)`)
/// so they can't collide with the non-negative built-in command ids returned
/// by [`all_command_meta`].
fn template_to_command_def(idx: usize, template: &PipelineTemplate) -> CommandDef {
    let category = template
        .steps
        .first()
        .map(|s| match s.command.category() {
            CommandCategory::Preprocess => StepCategory::Preprocess,
            CommandCategory::Segment => StepCategory::Segment,
            CommandCategory::InstanceSegmentation => StepCategory::InstanceSegmentation,
            CommandCategory::Measure => StepCategory::Measure,
            CommandCategory::Object => StepCategory::Object,
        })
        .unwrap_or(StepCategory::Preprocess);

    let author = template.meta.authors.first().cloned().unwrap_or_default();
    let co_authors = template.meta.authors.get(1..).unwrap_or(&[]).join(", ");

    CommandDef {
        id: -(idx as i32) - 1,
        name: template.meta.name.clone().into(),
        summary: template.meta.short_description.clone().into(),
        description: template.meta.description.clone().into(),
        category,
        icon_glyph: "★".into(),
        keywords: template.meta.name.to_ascii_lowercase().into(),
        source: "template".into(),
        favorite: false,
        recent: false,
        default_params: ModelRc::default(),
        is_template: true,
        author: author.into(),
        co_authors: co_authors.into(),
        organization: template.meta.author_organization.clone().into(),
        creation_time: template
            .meta
            .creation_time
            .format("%Y-%m-%d")
            .to_string()
            .into(),
    }
}

fn to_command_def(m: &CommandMeta) -> CommandDef {
    let cat = match m.category {
        CommandCategory::Preprocess => StepCategory::Preprocess,
        CommandCategory::Segment => StepCategory::Segment,
        CommandCategory::InstanceSegmentation => StepCategory::InstanceSegmentation,
        CommandCategory::Measure => StepCategory::Measure,
        CommandCategory::Object => StepCategory::Object,
    };
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
        is_template: false,
        author: "".into(),
        co_authors: "".into(),
        organization: "".into(),
        creation_time: "".into(),
    };
    detail
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
/// `object_class_id` choices are preserved wherever the same model class is
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
            let object_class_id = old
                .iter()
                .find(|m| m.segmentation_class == label.class)
                .map(|m| m.object_class_id)
                .unwrap_or(SegmentationClass::BACKGROUND);
            SegmentationMappingSettings {
                segmentation_class: label.class,
                object_class_id,
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
        out.push_str(&format!("\nAuthor: {}\n", meta.authors.join(", ")));
    }
    match &settings.classifier {
        AiLearningClassifierSettings::Pixel { class_labels, .. } => {
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
        });

        PipelinesController::force_all_t_stacks(&mut project);

        let global = project.images.settings.t_stack.as_ref().unwrap();
        assert_eq!(global.stack_handling, TStackHandling::AllStacks);
        assert_eq!(
            global.playback_speed, 1.5,
            "unrelated fields like playback speed must survive the force"
        );
    }

    #[test]
    fn force_all_t_stacks_overrides_every_images_own_series_override_too() {
        let mut project = project_with_images(&[("a.tif", 1), ("b.tif", 1)]);
        for image in project.images.list.values_mut() {
            image.series.get_mut(&0).unwrap().t_stack = Some(TStackSettings {
                stack_handling: TStackHandling::SingleStack,
                playback_speed: 1.0,
                t_stack: 5,
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
        let project = project_with_images(&[("a.tif", 1)]);
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

    fn template(idx_seed: &str, steps: Vec<PipelineStepSettings>) -> PipelineTemplate {
        PipelineTemplate {
            meta: MetaData {
                name: format!("Template {idx_seed}"),
                short_description: "short".into(),
                description: "long description".into(),
                authors: vec!["Ada Lovelace".into()],
                ..Default::default()
            },
            steps: steps,
            ..Default::default()
        }
    }

    #[test]
    fn template_to_command_def_encodes_the_index_as_a_negative_id() {
        let t = template("A", vec![]);
        assert_eq!(template_to_command_def(0, &t).id, -1);
        assert_eq!(template_to_command_def(4, &t).id, -5);
    }

    #[test]
    fn template_to_command_def_defaults_to_preprocess_with_no_steps() {
        let t = template("Empty", vec![]);
        let def = template_to_command_def(0, &t);
        assert_eq!(def.category, StepCategory::Preprocess);
        assert!(def.is_template);
        assert_eq!(def.source, SharedString::from("template"));
    }

    #[test]
    fn template_to_command_def_takes_its_category_from_the_first_step() {
        // id 6 is "ConnectedComponents" -> CommandCategory::InstanceSegmentation.
        let step = PipelineStepSettings {
            enabled: true,
            command: default_command(6).unwrap(),
        };
        let t = template("Multi", vec![step]);
        let def = template_to_command_def(0, &t);
        assert_eq!(def.category, StepCategory::InstanceSegmentation);
    }

    #[test]
    fn template_to_command_def_uses_the_first_author_and_joins_the_rest_as_co_authors() {
        let mut t = template("X", vec![]);
        let def = template_to_command_def(0, &t);
        assert_eq!(def.author, SharedString::from("Ada Lovelace"));
        assert_eq!(def.co_authors, SharedString::from(""));

        t.meta.authors.push("Alan Turing".into());
        let def_with_co_author = template_to_command_def(0, &t);
        assert_eq!(
            def_with_co_author.co_authors,
            SharedString::from("Alan Turing")
        );

        t.meta.authors.clear();
        let def_empty = template_to_command_def(0, &t);
        assert_eq!(
            def_empty.author,
            SharedString::from(""),
            "no author when the authors list is empty"
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
            assert!(!def.is_template);
            assert_eq!(def.source, SharedString::from("built-in"));
            let expected_category = match meta.category {
                CommandCategory::Preprocess => StepCategory::Preprocess,
                CommandCategory::Segment => StepCategory::Segment,
                CommandCategory::InstanceSegmentation => StepCategory::InstanceSegmentation,
                CommandCategory::Measure => StepCategory::Measure,
                CommandCategory::Object => StepCategory::Object,
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
                segmentation_class: SegmentationClass(1),
                object_class_id: SegmentationClass(2),
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
                segmentation_class: SegmentationClass(1),
                object_class_id: SegmentationClass(2),
            }],
        };

        reconcile_pixel_classifier_mapping(&LocalFileSystem::default(), &mut settings);

        assert!(settings.segmentation_mapping.is_empty());
    }

    #[test]
    fn reconcile_ai_object_classifier_mapping_clears_the_mapping_when_model_path_is_empty() {
        let mut settings = AiObjectClassifierSettings {
            segmentation_mapping: vec![ClassificationMappingSettings {
                object_class: ObjectClass::Valid(1),
                output_class: ObjectClass::Valid(2),
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
        assert!(info.contains("Classes:"));
        assert!(info.contains(&format!("- Nucleus (id {})", SegmentationClass(1).as_u32())));
        assert!(info.contains(&format!(
            "- Background (id {})",
            SegmentationClass(2).as_u32()
        )));
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
                class_labels: vec![],
            },
            evanalyzer_cfg::settings::meta_data::MetaData {
                name: "Model".into(),
                short_description: "A short summary".into(),
                description: "A longer description.".into(),
                authors: vec!["Ada Lovelace".into()],
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
                    enabled: true,
                    command: PipelineCommand::Threshold(ThresholdSettings {
                        thresholds: vec![ThresholdEntrySettings {
                            min_threshold: 0.5,
                            ..Default::default()
                        }],
                        ..Default::default()
                    }),
                }],
            });
        }

        controller.modify_group_item(1, 0, "thresholds", true, None);

        let project = ui_state.get_project();
        let PipelineCommand::Threshold(settings) = &project.pipelines[0].steps[0].command else {
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
                    enabled: true,
                    command: PipelineCommand::Threshold(ThresholdSettings {
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
                }],
            });
        }

        controller.modify_group_item(1, 0, "thresholds", false, Some(0));

        let project = ui_state.get_project();
        let PipelineCommand::Threshold(settings) = &project.pipelines[0].steps[0].command else {
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

        controller.apply_picker_filter(&ui, "", false);

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
                + picker.get_cat_count_cls(),
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

        controller.apply_picker_filter(&ui, &meta.name.to_uppercase(), false);

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

        controller.apply_picker_filter(&ui, "no-such-command-exists-xyz", false);

        let picker = ui.global::<CommandPickerState>();
        assert_eq!(picker.get_total_shown(), 0);
        assert_eq!(picker.get_cat_count_pre(), 0);
    }

    #[test]
    fn apply_picker_filter_restricts_to_the_active_category_chips() {
        let (ui, _results_ui) = test_ui_windows();
        let (_ui_state, controller) = make_controller(ui.as_weak());
        ui.global::<CommandPickerState>().set_fcat_seg(true);

        controller.apply_picker_filter(&ui, "", false);

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

    #[test]
    fn apply_picker_filter_favorites_mode_shows_only_matching_templates() {
        let (ui, _results_ui) = test_ui_windows();
        let (_ui_state, controller) = make_controller(ui.as_weak());
        *controller.pipeline_templates.lock().unwrap() =
            vec![template("Only", vec![]), template("Other", vec![])];

        controller.apply_picker_filter(&ui, "Only", true);

        let picker = ui.global::<CommandPickerState>();
        assert_eq!(
            picker.get_total_shown(),
            1,
            "favorites mode counts only matching templates, not built-in commands"
        );
        assert_eq!(picker.get_shown_templates().row_count(), 1);
    }

    // -- attach_callbacks (live AppWindow) -----------------------------------------

    use crate::editor::test_support::{test_ui_state, test_ui_windows};

    fn make_controller(ui: slint::Weak<AppWindow>) -> (Arc<UiState>, Arc<PipelinesController>) {
        let ui_state = test_ui_state();
        let viewport_controller = Arc::new(ViewportController::new(ui.clone(), ui_state.clone()));
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

        ui.global::<PipelinesPanelState>()
            .invoke_set_breakpoint(7, 2, 1);
        assert_eq!(
            *controller.breakpoint.lock().unwrap(),
            Some((7, 2, evanalyzer_cfg::core_types::BreakpointMode::Stop))
        );

        ui.global::<PipelinesPanelState>()
            .invoke_set_breakpoint(7, 2, 2);
        assert_eq!(
            *controller.breakpoint.lock().unwrap(),
            Some((7, 2, evanalyzer_cfg::core_types::BreakpointMode::Snapshot)),
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
            enabled: true,
            command: default_command(meta.id).unwrap(),
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
            .map(|s| s.command.name().to_string())
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
        let params = step(command_name).command.to_parameters();
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
        let expected = advanced_stats(&step("ClassifyObjects").command.to_parameters());
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
        if let PipelineCommand::Threshold(s) = &mut threshold.command {
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
            ["Scratchpad", "Memory[3]", "Cache[4]", "Channel 2"]
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
                ImageAddress::Memory(MemoryId::PipelineContext(1)),
                "Memory[1]",
            ),
            (2, 1, 3, ImageAddress::Channel(3), "Channel 3"),
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

        // Unknown command / template ids are ignored.
        picker.invoke_confirm(99_999);
        picker.invoke_confirm(-7);
        picker.invoke_select(99_999);
        picker.invoke_select(-7);
        assert_eq!(step_names(&ui_state, 1).len(), 5);
    }

    #[test]
    fn picker_template_entries_insert_all_their_steps() {
        let (ui, ui_state, controller) = selected(vec![]);
        *controller.pipeline_templates.lock().unwrap() = vec![template(
            "Nuclei",
            vec![step("Threshold"), step("Colocalization")],
        )];
        let picker = ui.global::<CommandPickerState>();
        picker.invoke_select(-1);
        assert!(picker.get_has_detail());
        picker.set_pipeline_id(1);
        picker.set_insert_after_idx(-1);
        picker.invoke_confirm(-1);
        assert_eq!(step_names(&ui_state, 1), ["Threshold", "Colocalization"]);
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
            + picker.get_cat_count_cls();
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
        let PipelineCommand::Colocalization(settings) = &project.pipelines[0].steps[0].command
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
        let params = project.pipelines[0].steps[0].command.to_parameters();
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
        let PipelineCommand::Colocalization(settings) = &project.pipelines[0].steps[0].command
        else {
            panic!("expected coloc");
        };
        assert_eq!(settings.min_coloc_area, 0.0);
    }

    #[test]
    fn show_model_info_reports_a_model_that_cannot_be_loaded() {
        let mut classifier = step(PIXEL_CLASSIFIER_COMMAND_NAME);
        if let PipelineCommand::PixelClassifier(settings) = &mut classifier.command {
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
        *controller.breakpoint.lock().unwrap() =
            Some((1, 0, evanalyzer_cfg::core_types::BreakpointMode::Stop));
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
}
