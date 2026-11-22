//! # modules
//!
//! Modules in the editor: module cards, creating and expanding modules, the
//! library, checks, and the measurement settings of a step.
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-11-22
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.

pub(crate) mod checks;
pub(crate) mod editing;
pub(crate) mod insights;
pub(crate) mod library;
pub(crate) mod measurement_choices;

pub use checks::{
    ModuleProblem, ProblemFix, Severity, apply_fix, check_module, check_modules, check_project,
    problems_at, step_problem,
};
pub use editing::{
    ClassRole, InPlaceCopy, ModulePlan, ModulePlanClass, StepList, add_binding, apply_defaults,
    bindable_settings, bound_setting_edited, clone_module_for_step, compatible_parameters,
    copy_name_problem, create_module, edit_parameter, expand_module, exposed_settings,
    in_place_copy, list_entry_removed, module_in_place_for_step, module_uses, move_parameter,
    new_parameter_key, parameter_for_setting, plan_module, prune_module_values, remove_binding,
    remove_module, remove_module_step, remove_parameter, remove_unused_modules,
    set_parameter_section, setting_at, step_setting, step_settings, toggle_exposed, unused_modules,
    would_contain_itself,
};
pub use insights::{
    ImageChannels, image_channels, parameter_key, parameter_wiring, rekey_parameters,
    resolved_commands, setting_wiring, size_settings_set,
};
pub use library::{
    content_hash, insert_module_template, library_name, merge_module_definitions,
    modified_since_inserted, modules_used_by, project_copy, save_module_template,
    save_pipeline_as_module, update_available, update_from_library,
};

pub use crate::templates::checks::{
    TemplateCheck, check_module_template, check_project_template, check_template_file,
    is_template_path, validate_templates,
};
pub use evanalyzer_core::module_parameters;
pub use evanalyzer_core::module_steps_with_values;
pub use evanalyzer_core::set_module_parameter;
/// What a step does at its place in the pipeline, and what may follow.
pub use evanalyzer_core::{RoleTracker, step_roles, tracker_after};
