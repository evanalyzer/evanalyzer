use crate::AppWindow;
use crate::DialogType;
use crate::editor::pipeline_task::PipelineTask;
use crate::editor::roi_list_controller::RoiListController;
use crate::editor::template_controller::TemplateController;
use crate::editor::viewport_controller::ViewportController;
use crate::{
    CommandDef, CommandParameter, CommandPickerState, GlobalAppState, GroupItem, LeafParam,
    ParamType, Pipeline, PipelineCommand as SlintPipelineCommand, PipelineStatus,
    PipelinesPanelState, StepCategory, UiState, WarningState,
};
use crate::{PipelineDeleteConfirmState, PipelineEditState, PipelineRunningState};
use evanalyzer_app::extensions::project_ext::ProjectExt;
use evanalyzer_app::templates::load_pipeline_templates;
use evanalyzer_cfg::core_types::MemorySlot;
use evanalyzer_cfg::core_types::PipelineId;
use evanalyzer_cfg::core_types::{ImageAddress, MemoryId};
use evanalyzer_cfg::settings::parameter_def::{ParamType as CfgParamType, ParameterDef};
use evanalyzer_cfg::settings::pipeline_command::{
    CommandCategory, all_command_meta, default_command,
};
use evanalyzer_cfg::settings::pipeline_settings::{PipelineSettings, PipelineStepSettings};
use evanalyzer_cfg::settings::templates::PipelineTemplate;
use log::debug;
use log::info;
use log::warn;
use slint::{ComponentHandle, Model, ModelRc, SharedString, VecModel};
use std::sync::Arc;
use std::sync::{Condvar, Mutex, atomic::AtomicBool};

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
    pub(crate) _roi_list_controller: Arc<RoiListController>,
    pub(crate) viewport_controller: Arc<ViewportController>,
    pub(crate) template_controller: Arc<TemplateController>,
    pub(crate) task_request: Arc<(Mutex<Option<PipelineTask>>, Condvar)>,
    pub(crate) pipeline_cancel_flag: Arc<Mutex<Option<Arc<AtomicBool>>>>,
    /// Currently active breakpoint: (pipeline_id, step_id, mode).  `None` = no breakpoint.
    pub(crate) breakpoint: Arc<Mutex<Option<(u32, i32, evanalyzer_core::BreakpointMode)>>>,

    /// If true the trigger_pipeline_preview_execution is called on parameter change
    auto_preview_enabled: Mutex<bool>,

    /// Pipeline templates currently shown in the command picker's "Templates"
    /// section. Reloaded from disk whenever the picker is opened.
    pipeline_templates: Mutex<Vec<PipelineTemplate>>,
}

impl PipelinesController {
    pub fn new(
        ui: slint::Weak<AppWindow>,
        app_state: Arc<UiState>,
        roi_list_controller: Arc<RoiListController>,
        viewport_controller: Arc<ViewportController>,
        template_controller: Arc<TemplateController>,
    ) -> Self {
        Self {
            ui,
            app_state: app_state.clone(),
            _roi_list_controller: roi_list_controller,
            viewport_controller,
            template_controller,
            task_request: Arc::new((Mutex::new(None), Condvar::new())),
            pipeline_cancel_flag: Arc::new(Mutex::new(None)),
            breakpoint: Arc::new(Mutex::new(None)),
            auto_preview_enabled: Mutex::new(false),
            pipeline_templates: Mutex::new(Vec::new()),
        }
    }

    pub fn attach_callbacks(self: &Arc<Self>) {
        let ui_handle = self.ui.clone();
        if let Some(ui) = ui_handle.upgrade() {
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
                        evanalyzer_core::BreakpointMode::Snapshot
                    } else {
                        evanalyzer_core::BreakpointMode::Stop
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

            // Full run pipeline
            let manager = self.clone();
            ui.global::<PipelinesPanelState>().on_run_all(move || {
                manager.trigger_pipeline_full_run();
            });

            // Selected pipeline
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
                        name: Some(name.clone()),
                        image_source: ImageAddress::Channel(0),
                        enabled: true,
                        steps: vec![],
                    });
                    (next_id, name)
                };
                manager.pipeline_settings_changed();
                manager.sync_pipelines_to_slint();
                manager.sync_steps_of_selected_pipeline_to_slint(PipelineId(new_id), true);
                let ui_weak = manager.ui.clone();
                slint::invoke_from_event_loop(move || {
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
                                (p.name.clone().unwrap_or_default(), stype, slot, ch)
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
                                clone.name = Some(format!(
                                    "{} (Copy)",
                                    p.name.as_deref().unwrap_or(&format!("Pipeline {}", p.id.0))
                                ));
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
                    slint::invoke_from_event_loop(move || {
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
                    let pipeline_name = {
                        let project = manager.app_state.get_project();
                        project
                            .pipelines
                            .iter()
                            .find(|p| p.id.0 == pipeline_id as u32)
                            .and_then(|p| p.name.clone())
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
                            slint::invoke_from_event_loop(move || {
                                if let Some(ui) = ui_weak.upgrade() {
                                    ui.global::<PipelinesPanelState>()
                                        .set_active_pipeline_id(nid as i32);
                                }
                            })
                            .ok();
                        }
                        None => {
                            let ui_weak = manager.ui.clone();
                            slint::invoke_from_event_loop(move || {
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
                        p.name = if name.is_empty() { None } else { Some(name) };
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
                if let Some(flag) = manager.pipeline_cancel_flag.lock().unwrap().as_ref() {
                    flag.store(true, std::sync::atomic::Ordering::Relaxed);
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
                            let name = p
                                .name
                                .clone()
                                .unwrap_or_else(|| format!("Pipeline {}", p.id.0));
                            // Find the category of the last step at or before the insertion point.
                            // step_after_idx is the 0-based index of the step we insert AFTER.
                            // A value >= steps.len() means "append at end".
                            let context_idx = if step_after_idx >= 0 {
                                (step_after_idx as usize).min(p.steps.len().saturating_sub(1))
                            } else {
                                usize::MAX
                            };
                            let (ctx_cat, suggested) = if p.steps.is_empty() {
                                (-1i32, -1i32)
                            } else if context_idx == usize::MAX {
                                (-1i32, -1i32)
                            } else {
                                let last_cat = p.steps[context_idx].command.category();
                                let ctx_order = last_cat.display_order() as i32;
                                // Preprocess can be followed by another Preprocess or Segment,
                                // so show All (-1) rather than locking to Segment.
                                // All other stages have a single clear next stage.
                                let next_order = if matches!(last_cat, CommandCategory::Preprocess)
                                {
                                    -1i32
                                } else {
                                    last_cat.suggested_next().display_order() as i32
                                };
                                (ctx_order, next_order)
                            };
                            (name, p.steps.len() as i32, ctx_cat, suggested)
                        } else {
                            (String::new(), 0, -1i32, -1i32)
                        }
                    };
                    manager.reload_pipeline_templates();
                    if let Some(ui) = manager.ui.upgrade() {
                        let picker = ui.global::<CommandPickerState>();
                        picker.set_pipeline_id(pipeline_id);
                        picker.set_insert_after_idx(step_after_idx);
                        picker.set_target_pipeline(pipeline_name.into());
                        picker.set_total_steps(total_steps);
                        picker.set_context_category(context_cat);
                        picker.set_query("".into());
                        picker.set_filter_favorites(false);
                        // Auto-select the filter for the suggested next category.
                        // -1 keeps the "All" view (empty pipeline or unknown context).
                        picker.set_filter_category(suggested_filter);
                        picker.set_selected_id(-1);
                        // Apply the filter immediately so the list matches the pre-selected chip.
                        manager.apply_picker_filter(&ui, "", suggested_filter, false);
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
                    let filter_cat = picker.get_filter_category();
                    let filter_favorites = picker.get_filter_favorites();
                    manager.apply_picker_filter(&ui, query.as_str(), filter_cat, filter_favorites);
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
                        let cat = match m.category {
                            CommandCategory::Preprocess => StepCategory::Preprocess,
                            CommandCategory::Segment => StepCategory::Segment,
                            CommandCategory::Object => StepCategory::Object,
                            CommandCategory::Measure => StepCategory::Measure,
                            CommandCategory::Classify => StepCategory::Classify,
                        };
                        let detail = CommandDef {
                            id: m.id,
                            name: m.name.into(),
                            summary: m.summary.into(),
                            description: m.description.into(),
                            category: cat,
                            icon_glyph: "▭".into(),
                            keywords: m.name.to_ascii_lowercase().into(),
                            source: "built-in".into(),
                            favorite: false,
                            recent: false,
                            default_params: ModelRc::default(),
                        };
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
                        template.pipeline_steps.clone()
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

                    let (new_summary, new_param_value) = {
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
                        let summary = step.command.to_summary();
                        let params_now = step.command.to_parameters();
                        let updated_value = if let Some((g, idx, f)) = &nested_path {
                            params_now
                                .iter()
                                .find(|p| p.name == *g)
                                .and_then(|p| p.groups.get(*idx))
                                .and_then(|item| item.iter().find(|fd| fd.name == *f))
                                .map(|fd| fd.value.clone())
                                .unwrap_or_default()
                        } else {
                            params_now
                                .into_iter()
                                .find(|p| p.name == param_name)
                                .map(|p| p.value)
                                .unwrap_or_default()
                        };
                        (summary, updated_value)
                    }; // write lock dropped here

                    // Update the affected step in the Slint model: summary + the
                    // changed param's value (and, for multi-select toggles, its flags).
                    let model = ui.global::<PipelinesPanelState>().get_active_commands();
                    if let Some(mut cmd) = model.row_data(step_idx as usize) {
                        cmd.summary = new_summary.into();
                        let params = cmd.parameters.clone();

                        if let Some((group_name, idx, field_name)) = nested_path {
                            // Nested group field: find the group CommandParameter, then
                            // update fields[k].value inside group_items[idx].
                            for i in 0..params.row_count() {
                                if let Some(p) = params.row_data(i) {
                                    if p.name.as_str() == group_name {
                                        let items = p.group_items.clone();
                                        if let Some(item) = items.row_data(idx) {
                                            let fields = item.fields.clone();
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
                                        break;
                                    }
                                }
                            }
                        } else {
                            // Flat parameter: update value (and flags for multi-select).
                            for i in 0..params.row_count() {
                                if let Some(mut p) = params.row_data(i) {
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
                                        params.set_row_data(i, p);
                                        break;
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
        }

        // Must be called onece at startup
        self.sync_commands_to_selection_dialog_slint();
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

        // We remove all images and add only the actual selected, because this is just the preview
        let mut project_tmp = project.clone();
        project_tmp.images.list.clear();
        project_tmp
            .images
            .list
            .insert(current_image_path, current_image_settings.clone());

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

    pub fn trigger_pipeline_full_run(&self) {
        let project = self.app_state.get_project();

        let Some(current_project) = &project.tmp_settings.current_project else {
            warn!("No project path set, please save project first!");
            self.show_warning(
                "No project is open. Please save the project first before starting an analysis.",
            );
            return;
        };

        let task: PipelineTask = PipelineTask {
            project_settings: project.settings.clone(),
            project_path: current_project
                .parent()
                .unwrap_or(current_project)
                .to_path_buf(),
            preview: false,
            breakpoint: None,
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
    fn pipeline_settings_changed(self: &Arc<Self>) {
        self.app_state.mark_dirty();

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
        if let Err(e) = slint::invoke_from_event_loop(move || {
            if let Some(ui) = ui_weak.upgrade() {
                ui.global::<WarningState>().set_message(message.into());
                ui.global::<GlobalAppState>()
                    .set_active_dialog(DialogType::Warning);
            }
        }) {
            warn!("Failed to show warning dialog: {e}");
        }
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
                        name: p
                            .name
                            .clone()
                            .unwrap_or_else(|| format!("Pipeline {}", p.id.0))
                            .into(),
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

        if let Err(e) = slint::invoke_from_event_loop(move || {
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

    fn apply_picker_filter(
        self: &Arc<Self>,
        ui: &AppWindow,
        query: &str,
        filter_cat: i32,
        filter_favorites: bool,
    ) {
        let q = query.to_ascii_lowercase();
        let metas = all_command_meta();

        let text_matches = |m: &&evanalyzer_cfg::settings::pipeline_command::CommandMeta| {
            q.is_empty()
                || m.name.to_ascii_lowercase().contains(&q)
                || m.summary.to_ascii_lowercase().contains(&q)
        };
        let cat_enabled = |target: CommandCategory| -> bool {
            match filter_cat {
                -1 => true,
                0 => matches!(target, CommandCategory::Preprocess),
                1 => matches!(target, CommandCategory::Segment),
                2 => matches!(target, CommandCategory::Object),
                3 => matches!(target, CommandCategory::Measure),
                4 => matches!(target, CommandCategory::Classify),
                _ => false,
            }
        };
        let make = |m: &evanalyzer_cfg::settings::pipeline_command::CommandMeta,
                    cat: StepCategory|
         -> CommandDef {
            CommandDef {
                id: m.id,
                name: m.name.into(),
                summary: m.summary.into(),
                description: m.description.into(),
                category: cat,
                icon_glyph: "▭".into(),
                keywords: m.name.to_ascii_lowercase().into(),
                source: "built-in".into(),
                favorite: false,
                recent: false,
                default_params: ModelRc::default(),
            }
        };
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
                matches!(m.category, CommandCategory::Object)
                    && cat_enabled(m.category)
                    && text_matches(m)
            })
            .map(|m| make(m, StepCategory::Object))
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
                matches!(m.category, CommandCategory::Classify)
                    && cat_enabled(m.category)
                    && text_matches(m)
            })
            .map(|m| make(m, StepCategory::Classify))
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

    /// Reloads pipeline templates from the user and app templates folders.
    fn reload_pipeline_templates(self: &Arc<Self>) {
        let templates: Vec<PipelineTemplate> = load_pipeline_templates()
            .into_iter()
            .map(|(_path, template)| template)
            .collect();
        *self.pipeline_templates.lock().expect("Poisoned") = templates;
    }

    fn sync_commands_to_selection_dialog_slint(self: &Arc<Self>) {
        // Collect only Send-safe primitives outside the event loop closure.
        struct RawCmd {
            id: i32,
            name: &'static str,
            summary: &'static str,
            category: StepCategory,
        }
        let raw: Vec<RawCmd> = all_command_meta()
            .into_iter()
            .map(|m| RawCmd {
                id: m.id,
                name: m.name,
                summary: m.summary,
                category: match m.category {
                    CommandCategory::Preprocess => StepCategory::Preprocess,
                    CommandCategory::Segment => StepCategory::Segment,
                    CommandCategory::Object => StepCategory::Object,
                    CommandCategory::Measure => StepCategory::Measure,
                    CommandCategory::Classify => StepCategory::Classify,
                },
            })
            .collect();

        let ui_weak = self.ui.clone();
        if let Err(e) = slint::invoke_from_event_loop(move || {
            if let Some(ui) = ui_weak.upgrade() {
                let make_def = |r: &RawCmd| CommandDef {
                    id: r.id,
                    name: r.name.into(),
                    summary: r.summary.into(),
                    description: "".into(),
                    category: r.category,
                    icon_glyph: "▭".into(),
                    keywords: r.name.to_ascii_lowercase().into(),
                    source: "built-in".into(),
                    favorite: false,
                    recent: false,
                    default_params: ModelRc::default(),
                };
                let all: Vec<CommandDef> = raw.iter().map(make_def).collect();
                let shown_pre: Vec<CommandDef> = raw
                    .iter()
                    .filter(|r| r.category == StepCategory::Preprocess)
                    .map(make_def)
                    .collect();
                let shown_seg: Vec<CommandDef> = raw
                    .iter()
                    .filter(|r| r.category == StepCategory::Segment)
                    .map(make_def)
                    .collect();
                let shown_obj: Vec<CommandDef> = raw
                    .iter()
                    .filter(|r| r.category == StepCategory::Object)
                    .map(make_def)
                    .collect();
                let shown_mea: Vec<CommandDef> = raw
                    .iter()
                    .filter(|r| r.category == StepCategory::Measure)
                    .map(make_def)
                    .collect();
                let shown_cls: Vec<CommandDef> = raw
                    .iter()
                    .filter(|r| r.category == StepCategory::Classify)
                    .map(make_def)
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
                let name = pipeline
                    .name
                    .clone()
                    .unwrap_or_else(|| format!("Pipeline {}", pipeline.id.0));
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
                            CommandCategory::Object => StepCategory::Object,
                            CommandCategory::Measure => StepCategory::Measure,
                            CommandCategory::Classify => StepCategory::Classify,
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

        if let Err(e) = slint::invoke_from_event_loop(move || {
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

                let commands: Vec<SlintPipelineCommand> = step_data
                    .into_iter()
                    .map(|d| {
                        let params: Vec<CommandParameter> = d
                            .parameters
                            .into_iter()
                            .map(|p| {
                                let group_items: Vec<GroupItem> = p
                                    .groups
                                    .into_iter()
                                    .map(|group| GroupItem {
                                        fields: ModelRc::new(VecModel::from(
                                            group
                                                .into_iter()
                                                .map(|lp| LeafParam {
                                                    name: lp.name.into(),
                                                    display_name: lp.display_name.into(),
                                                    description: lp.description.into(),
                                                    value: lp.value.into(),
                                                    param_type: match lp.param_type {
                                                        CfgParamType::Number => ParamType::Number,
                                                        CfgParamType::Text => ParamType::Text,
                                                        CfgParamType::Dropdown => {
                                                            ParamType::Dropdown
                                                        }
                                                        CfgParamType::Toggle => ParamType::Toggle,
                                                        CfgParamType::Slider => ParamType::Slider,
                                                        CfgParamType::Spinner => ParamType::Spinner,
                                                        CfgParamType::Group => ParamType::Group,
                                                        CfgParamType::ObjClass => {
                                                            ParamType::ObjClass
                                                        }
                                                        CfgParamType::SegClass => {
                                                            ParamType::SegClass
                                                        }
                                                        CfgParamType::MultiObjClass => {
                                                            ParamType::MultiObjClass
                                                        }
                                                        CfgParamType::MultiSegClass => {
                                                            ParamType::MultiSegClass
                                                        }
                                                        CfgParamType::PixelUnits => {
                                                            ParamType::PixelUnits
                                                        }
                                                        CfgParamType::SizeUnits => {
                                                            ParamType::SizeUnits
                                                        }
                                                        CfgParamType::Label => ParamType::Label,
                                                    },
                                                    options: ModelRc::new(VecModel::from(
                                                        lp.options
                                                            .into_iter()
                                                            .map(SharedString::from)
                                                            .collect::<Vec<_>>(),
                                                    )),
                                                    min: lp.min,
                                                    max: lp.max,
                                                    step: lp.step,
                                                })
                                                .collect::<Vec<_>>(),
                                        )),
                                    })
                                    .collect();
                                CommandParameter {
                                    name: p.name.into(),
                                    display_name: p.display_name.into(),
                                    description: p.description.into(),
                                    value: p.value.into(),
                                    param_type: match p.param_type {
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
                                    },
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
                                }
                            })
                            .collect();
                        SlintPipelineCommand {
                            id: d.idx,
                            name: d.name.into(),
                            category: d.category,
                            summary: d.summary.into(),
                            enabled: d.enabled,
                            expanded: expanded_by_id.get(&d.idx).copied().unwrap_or(false),
                            parameters: ModelRc::new(VecModel::from(params)),
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
        .pipeline_steps
        .first()
        .map(|s| match s.command.category() {
            CommandCategory::Preprocess => StepCategory::Preprocess,
            CommandCategory::Segment => StepCategory::Segment,
            CommandCategory::Object => StepCategory::Object,
            CommandCategory::Measure => StepCategory::Measure,
            CommandCategory::Classify => StepCategory::Classify,
        })
        .unwrap_or(StepCategory::Preprocess);

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
    }
}
