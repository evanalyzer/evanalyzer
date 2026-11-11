//! # step_roles
//!
//! The role a step runs in: the category it works as at its place in the
//! pipeline. A command of one category always runs as that category; a
//! command of several (Morphological: Preprocess and Instance Seg.) runs as
//! the first of them the step before it lets follow - after a Threshold on
//! the segmentation mask, after a blur on the image. A utility step (Script,
//! Save Image, Image Cache) has no role of its own: the step after it looks
//! past it.
//!
//! The runner (what a command works on), the step's colour and the picker's
//! suggestion all use this one rule, so the colour always tells what runs.
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-11-11
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.

use crate::settings::{
    modules::ModuleDefinition,
    pipeline_command::{CommandCategory, PipelineCommand},
    pipeline_settings::{PipelineStepSettings, StepContent},
};

/// Walks the commands of a pipeline in the order they run and knows what may
/// follow the last one (utility steps skipped).
#[derive(Clone, Copy, Debug, Default)]
pub struct RoleTracker {
    /// The categories that may follow the last command run, the first
    /// suggested - `None` before the first one.
    next: Option<&'static [CommandCategory]>,
}

impl RoleTracker {
    pub fn new() -> Self {
        Self::default()
    }

    /// The role `command` would run in here.
    pub fn role_of(&self, command: &PipelineCommand) -> CommandCategory {
        let categories = command.categories();
        let home = *command.category();
        if categories.len() < 2 {
            return home;
        }
        self.next
            .and_then(|next| categories.iter().copied().find(|c| next.contains(c)))
            .unwrap_or(home)
    }

    /// Runs `command` here: its role, and what may follow it from now on.
    pub fn run(&mut self, command: &PipelineCommand) -> CommandCategory {
        let role = self.role_of(command);
        if role != CommandCategory::Utility {
            self.next = Some(if command.categories().len() > 1 {
                role.default_next()
            } else {
                command.allowed_next()
            });
        }
        role
    }

    /// The categories that may follow the commands run so far, the first
    /// suggested - `None` before the first one.
    pub fn suggested_next(&self) -> Option<&'static [CommandCategory]> {
        self.next
    }
}

/// The commands `steps` run, in order: disabled steps left out, a module
/// opened up (one nested in it too) - the order a run executes them in
/// (core's `expand_step`). Only the commands' kinds count here, so a
/// module's values aren't applied; an unknown or self-containing module adds
/// nothing (a run refuses it anyway).
pub fn run_commands<'a>(
    steps: &'a [PipelineStepSettings],
    modules: &'a [ModuleDefinition],
) -> Vec<&'a PipelineCommand> {
    let mut commands = Vec::new();
    collect_commands(steps, modules, &mut Vec::new(), &mut commands);
    commands
}

fn collect_commands<'a>(
    steps: &'a [PipelineStepSettings],
    modules: &'a [ModuleDefinition],
    open: &mut Vec<&'a str>,
    out: &mut Vec<&'a PipelineCommand>,
) {
    for step in steps.iter().filter(|s| s.enabled) {
        match &step.content {
            StepContent::Command(command) => out.push(command),
            StepContent::Module(instance) => {
                if open.contains(&instance.key.as_str()) {
                    continue;
                }
                if let Some(definition) = modules.iter().find(|d| d.key == instance.key) {
                    open.push(&definition.key);
                    collect_commands(&definition.steps, modules, open, out);
                    open.pop();
                }
            }
        }
    }
}

/// The role of every command step of `steps`, at its place - `None` for a
/// module step. A disabled step gets the role it would run in; it doesn't
/// change the steps after it (a run skips it).
pub fn step_roles(
    steps: &[PipelineStepSettings],
    modules: &[ModuleDefinition],
) -> Vec<Option<CommandCategory>> {
    let mut tracker = RoleTracker::new();
    steps
        .iter()
        .map(|step| match &step.content {
            StepContent::Command(command) if step.enabled => Some(tracker.run(command)),
            StepContent::Command(command) => Some(tracker.role_of(command)),
            StepContent::Module(_) => {
                for command in run_commands(std::slice::from_ref(step), modules) {
                    tracker.run(command);
                }
                None
            }
        })
        .collect()
}

/// Where the commands of the first `count` steps of `steps` leave a
/// pipeline: what a step inserted after them may be.
pub fn tracker_after(
    steps: &[PipelineStepSettings],
    modules: &[ModuleDefinition],
    count: usize,
) -> RoleTracker {
    let mut tracker = RoleTracker::new();
    let count = count.min(steps.len());
    for command in run_commands(&steps[..count], modules) {
        tracker.run(command);
    }
    tracker
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::settings::modules::ModuleInstance;
    use crate::types::ids::StepId;
    use CommandCategory::*;

    fn command(key: &str) -> PipelineCommand {
        PipelineCommand::default_for_key(key).unwrap_or_else(|| panic!("no command `{key}`"))
    }

    fn step(key: &str) -> PipelineStepSettings {
        PipelineStepSettings {
            id: StepId::default(),
            enabled: true,
            content: StepContent::Command(command(key)),
        }
    }

    fn disabled(key: &str) -> PipelineStepSettings {
        PipelineStepSettings {
            enabled: false,
            ..step(key)
        }
    }

    fn module_step(key: &str) -> PipelineStepSettings {
        PipelineStepSettings {
            id: StepId::default(),
            enabled: true,
            content: StepContent::Module(ModuleInstance {
                key: key.into(),
                values: Default::default(),
            }),
        }
    }

    fn module(key: &str, steps: Vec<PipelineStepSettings>) -> ModuleDefinition {
        ModuleDefinition {
            key: key.into(),
            meta: Default::default(),
            classes: Vec::new(),
            steps,
            parameters: Vec::new(),
            origin: None,
        }
    }

    /// The role of the last step of `keys`.
    fn last_role(keys: &[&str]) -> CommandCategory {
        let steps: Vec<_> = keys.iter().map(|k| step(k)).collect();
        step_roles(&steps, &[]).last().copied().flatten().unwrap()
    }

    const MORPH: &str = "morphological_command";

    #[test]
    fn morphological_has_preprocess_as_home_and_instance_segmentation_too() {
        let morph = command(MORPH);
        assert_eq!(*morph.category(), Preprocess);
        assert_eq!(morph.categories(), &[Preprocess, InstanceSegmentation]);
    }

    #[test]
    fn script_save_image_and_image_cache_are_utilities() {
        for key in ["script", "save_image", "image_cache"] {
            assert_eq!(command(key).categories(), &[Utility], "{key}");
        }
    }

    #[test]
    fn a_first_morphological_works_as_preprocessing() {
        assert_eq!(last_role(&[MORPH]), Preprocess);
    }

    #[test]
    fn morphological_after_image_filters_works_as_preprocessing() {
        assert_eq!(last_role(&["gaussian_blur", MORPH]), Preprocess);
        assert_eq!(last_role(&["load_channel", MORPH]), Preprocess);
    }

    #[test]
    fn morphological_after_a_segmentation_works_on_the_mask() {
        assert_eq!(last_role(&["threshold", MORPH]), InstanceSegmentation);
        assert_eq!(
            last_role(&["threshold", "fill_holes", MORPH]),
            InstanceSegmentation
        );
        assert_eq!(
            last_role(&["threshold", "connected_components", MORPH]),
            InstanceSegmentation
        );
    }

    #[test]
    fn a_second_morphological_on_the_mask_stays_on_the_mask() {
        assert_eq!(
            last_role(&["threshold", MORPH, MORPH]),
            InstanceSegmentation
        );
        assert_eq!(last_role(&["gaussian_blur", MORPH, MORPH]), Preprocess);
    }

    #[test]
    fn utility_steps_are_looked_past() {
        assert_eq!(
            last_role(&["threshold", "save_image", "image_cache", MORPH]),
            InstanceSegmentation
        );
        assert_eq!(last_role(&["threshold", "save_image"]), Utility);
    }

    #[test]
    fn loading_a_channel_after_a_threshold_works_on_the_image_again() {
        assert_eq!(last_role(&["threshold", "load_channel", MORPH]), Preprocess);
    }

    #[test]
    fn without_a_fitting_category_the_home_is_taken() {
        // After extracting objects neither of its categories may follow.
        assert_eq!(
            last_role(&[
                "threshold",
                "connected_components",
                "extract_objects",
                MORPH
            ]),
            Preprocess
        );
    }

    #[test]
    fn a_disabled_step_doesnt_count_but_gets_its_role() {
        let steps = vec![step("gaussian_blur"), disabled("threshold"), step(MORPH)];
        assert_eq!(step_roles(&steps, &[])[2], Some(Preprocess));
        // A disabled morphological after a threshold: shown on the mask.
        let steps = vec![step("threshold"), disabled(MORPH), step("fill_holes")];
        assert_eq!(step_roles(&steps, &[])[1], Some(InstanceSegmentation));
    }

    #[test]
    fn a_module_counts_as_its_commands() {
        let modules = vec![module(
            "SEGMENT",
            vec![
                step("gaussian_blur"),
                step("threshold"),
                disabled("load_channel"),
            ],
        )];
        let steps = vec![module_step("SEGMENT"), step(MORPH)];
        assert_eq!(
            step_roles(&steps, &modules),
            [None, Some(InstanceSegmentation)]
        );
    }

    #[test]
    fn run_commands_open_nested_modules_and_skip_disabled_steps() {
        let modules = vec![
            module("INNER", vec![step("threshold"), disabled("fill_holes")]),
            module("OUTER", vec![step("gaussian_blur"), module_step("INNER")]),
            module("SELF", vec![module_step("SELF"), step("voronoi")]),
        ];
        let steps = vec![
            module_step("OUTER"),
            disabled("rank_filter"),
            step(MORPH),
            module_step("SELF"),
            module_step("MISSING"),
        ];
        let keys: Vec<&str> = run_commands(&steps, &modules)
            .iter()
            .map(|c| c.key())
            .collect();
        assert_eq!(keys, ["gaussian_blur", "threshold", MORPH, "voronoi"]);
    }

    #[test]
    fn the_suggestion_after_a_step_follows_its_role() {
        let steps = vec![step("threshold"), step(MORPH)];
        assert_eq!(
            tracker_after(&steps, &[], 2).suggested_next(),
            Some(InstanceSegmentation.default_next())
        );
        let steps = vec![step("gaussian_blur"), step(MORPH)];
        assert_eq!(
            tracker_after(&steps, &[], 2).suggested_next(),
            Some(Preprocess.default_next())
        );
        // After a utility step: what the step before it suggests.
        let steps = vec![step("threshold"), step("save_image")];
        assert_eq!(
            tracker_after(&steps, &[], 2).suggested_next(),
            Some(command("threshold").allowed_next())
        );
        assert_eq!(tracker_after(&steps, &[], 0).suggested_next(), None);
    }
}
