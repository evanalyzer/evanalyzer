//! # measurement_names
//!
//! The names a run gives the measurements made inside modules.
//!
//! A measurement step's ID only has to be unique where it is written: among
//! the pipelines' own steps, or in one module definition. A module can be
//! used several times - and inside other modules - so in a run each use
//! names its measurements itself: the ID, then the object class the use
//! makes (`threshold_01_evs_cy5`), or the module's key if it makes none.
//! The only step of its kind in the module goes without the number its ID
//! got by itself (`threshold_evs_cy5`). A name already given in the run
//! gets `_2`, `_3`, ...
//!
//! The settings that pick a measurement (an Exclude Image rule's) follow:
//! - inside a module, a key of the module's own step (`threshold_01/...`)
//!   means that step of the same use;
//! - a key starting with `@` names a step inside a module use directly -
//!   `@<pipeline>.<step>.<inner step>...:<key>`, how a pipeline's step picks
//!   a measurement made inside a module (see [`use_reference`]).
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-11-21
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.

use crate::job::modules::{ExpandedCommand, ScopeCounter, UseId, expand_step};
use evanalyzer_cfg::{
    core_types::{InternalErrors, ObjectClass, PipelineId, StepId},
    settings::{
        classification_settings::Class,
        parameter_def::{ParamType, ParameterDef},
        project_settings::ProjectSettings,
    },
};
use std::collections::{HashMap, HashSet};

/// One measurement step inside a module use and the name it has in a run.
#[derive(Clone, Debug, PartialEq)]
pub struct UseMeasurement {
    pub pipeline: PipelineId,
    /// The module step in the pipeline, then the steps leading to the
    /// measurement step inside it.
    pub path: Vec<StepId>,
    /// Its ID in its module.
    pub local: String,
    /// Its name in the run.
    pub name: String,
}

/// How a pipeline's step refers to measurement `key` (`threshold_01/...`)
/// of the step at `path` inside a module use of pipeline `pipeline`.
pub fn use_reference(pipeline: PipelineId, path: &[StepId], key: &str) -> String {
    let steps: Vec<String> = path.iter().map(|s| s.0.to_string()).collect();
    format!("@{}.{}:{key}", pipeline.0, steps.join("."))
}

/// A [`use_reference`] taken apart: pipeline, path, key.
fn parse_use_reference(text: &str) -> Option<(PipelineId, Vec<StepId>, &str)> {
    let (place, key) = text.strip_prefix('@')?.split_once(':')?;
    let mut numbers = place.split('.').map(|n| n.parse::<u32>().ok());
    let pipeline = PipelineId(numbers.next()??);
    let path: Option<Vec<StepId>> = numbers.map(|n| n.map(StepId)).collect();
    Some((pipeline, path?, key))
}

/// Names the measurements of a run's module uses, step by step in the order
/// they run, and points the settings picking them at those names.
pub struct MeasurementNamer {
    /// The project's class names, by number.
    class_names: HashMap<u32, String>,
    /// Every name given so far (lowercase).
    taken: HashSet<String>,
    /// The names of the module uses' measurement steps, by (pipeline, path).
    by_path: HashMap<(PipelineId, Vec<StepId>), String>,
    /// The names in each use, by (use, local ID lowercase).
    by_use: HashMap<(UseId, String), String>,
    named: Vec<UseMeasurement>,
}

impl MeasurementNamer {
    /// For a project with `classes`.
    pub fn new(classes: &[Class]) -> Self {
        Self {
            class_names: classes.iter().map(|c| (c.id, c.name.clone())).collect(),
            taken: HashSet::new(),
            by_path: HashMap::new(),
            by_use: HashMap::new(),
            named: Vec::new(),
        }
    }

    /// The measurements named so far inside module uses.
    pub fn named(&self) -> &[UseMeasurement] {
        &self.named
    }

    /// Names the measurement steps of `commands` - pipeline step `step` of
    /// `pipeline`, expanded with `scopes` - and rewrites the measurements
    /// their settings pick. A pipeline's own measurement step keeps its ID.
    pub fn name_step(
        &mut self,
        pipeline: PipelineId,
        step: StepId,
        commands: &mut [ExpandedCommand],
        scopes: &ScopeCounter,
    ) -> Result<(), InternalErrors> {
        // How many measurement steps of each kind each use has: one alone
        // goes without its number.
        let mut kinds: HashMap<(UseId, &'static str), usize> = HashMap::new();
        for expanded in commands.iter() {
            if let (Some(&use_id), Some(prefix), Some(id)) = (
                expanded.uses.last(),
                expanded.command.measurement_id_prefix(),
                expanded.command.measurement_id(),
            ) && !id.is_empty()
            {
                *kinds.entry((use_id, prefix)).or_default() += 1;
            }
        }
        for expanded in commands.iter_mut() {
            self.rewrite_references(expanded)?;
            let prefix = expanded.command.measurement_id_prefix();
            let innermost = expanded.uses.last().copied();
            let Some(id) = expanded.command.measurement_id_mut() else {
                continue;
            };
            // A Threshold without an ID measures nothing.
            if id.is_empty() {
                continue;
            }
            let Some(use_id) = innermost else {
                self.taken.insert(id.to_lowercase());
                continue;
            };
            let local = id.clone();
            let label = scopes
                .naming_of(use_id)
                .map(|n| self.use_label(n.class, &n.module))
                .unwrap_or_else(|| "module".to_string());
            let alone = prefix.is_some_and(|p| kinds.get(&(use_id, p)) == Some(&1));
            let base = match prefix {
                Some(prefix) if alone && is_numbered(&local, prefix) => prefix,
                _ => local.as_str(),
            };
            let name = self.free_name(&format!("{base}_{label}"));
            *id = name.clone();
            let mut path = Vec::with_capacity(expanded.path.len() + 1);
            path.push(step);
            path.extend_from_slice(&expanded.path);
            self.by_use
                .insert((use_id, local.to_lowercase()), name.clone());
            self.by_path.insert((pipeline, path.clone()), name.clone());
            self.named.push(UseMeasurement {
                pipeline,
                path,
                local,
                name,
            });
        }
        Ok(())
    }

    /// `base`, or with `_2`, `_3`, ... if a name already has it.
    fn free_name(&mut self, base: &str) -> String {
        let mut name = base.to_string();
        let mut n = 2;
        while self.taken.contains(&name.to_lowercase()) {
            name = format!("{base}_{n}");
            n += 1;
        }
        self.taken.insert(name.to_lowercase());
        name
    }

    /// The class's name as it goes into a measurement's name
    /// (`evs_cy5`), else the module's key's.
    fn use_label(&self, class: Option<ObjectClass>, module: &str) -> String {
        let class_name = class.and_then(|c| match c {
            ObjectClass::Global(n) => self.class_names.get(&n).cloned(),
            _ => None,
        });
        let label = slug(class_name.as_deref().unwrap_or(module));
        if label.is_empty() {
            slug(module)
        } else {
            label
        }
    }

    /// Points `expanded`'s measurement settings at the names in the run.
    fn rewrite_references(&self, expanded: &mut ExpandedCommand) -> Result<(), InternalErrors> {
        let mut settings = Vec::new();
        measurement_settings(
            &crate::algos::command_parameters(&expanded.command),
            "",
            &mut settings,
        );
        for (path, key) in settings {
            let Some(renamed) = self.renamed(expanded, &key)? else {
                continue;
            };
            crate::algos::apply_command_param(&mut expanded.command, &path, &renamed);
        }
        Ok(())
    }

    /// What measurement `key` of `expanded` is called in the run - `None`:
    /// as it is.
    fn renamed(
        &self,
        expanded: &ExpandedCommand,
        key: &str,
    ) -> Result<Option<String>, InternalErrors> {
        if let Some((pipeline, path, inner_key)) = parse_use_reference(key) {
            let (_, value) = inner_key.split_once('/').unwrap_or((inner_key, ""));
            return match self.by_path.get(&(pipeline, path)) {
                Some(name) => Ok(Some(format!("{name}/{value}"))),
                None => Err(InternalErrors::InvalidArgument(format!(
                    "\"{}\" picks a measurement no earlier step makes (\"{inner_key}\" in a \
                     module) - pick it again",
                    expanded.command.name()
                ))),
            };
        }
        let Some(&use_id) = expanded.uses.last() else {
            return Ok(None);
        };
        let (local, value) = key.split_once('/').unwrap_or((key, ""));
        Ok(self
            .by_use
            .get(&(use_id, local.to_lowercase()))
            .map(|name| format!("{name}/{value}")))
    }
}

/// The measurement settings among `params` (path, value), e.g.
/// ("rules.0.measurement", "quality_01/focus").
fn measurement_settings(params: &[ParameterDef], prefix: &str, out: &mut Vec<(String, String)>) {
    for param in params {
        let path = format!("{prefix}{}", param.name);
        if param.param_type == ParamType::Measurement && !param.value.is_empty() {
            out.push((path.clone(), param.value.clone()));
        }
        for (i, entry) in param.groups.iter().enumerate() {
            measurement_settings(entry, &format!("{path}.{i}."), out);
        }
    }
}

/// Whether `id` is a name as given by itself: `prefix`, `_`, a number
/// (`threshold_01`).
fn is_numbered(id: &str, prefix: &str) -> bool {
    id.len() > prefix.len() + 1
        && id[..prefix.len()].eq_ignore_ascii_case(prefix)
        && id[prefix.len()..].starts_with('_')
        && id[prefix.len() + 1..].bytes().all(|b| b.is_ascii_digit())
}

/// `text` as part of a measurement's name: lowercase letters and digits,
/// the rest one `_`.
fn slug(text: &str) -> String {
    let mut out = String::new();
    for c in text.chars().flat_map(char::to_lowercase) {
        if c.is_ascii_alphanumeric() {
            out.push(c);
        } else if !out.is_empty() && !out.ends_with('_') {
            out.push('_');
        }
    }
    out.trim_end_matches('_').to_string()
}

/// The names a run gives the measurements made inside `project`'s module
/// uses - every enabled pipeline and step, as the job generator names them.
/// Steps that can't be expanded are left out.
pub fn measurement_names(project: &ProjectSettings) -> Vec<UseMeasurement> {
    let mut namer = MeasurementNamer::new(project.classification.classes());
    let mut scopes = ScopeCounter::default();
    for pipeline in project.pipelines.iter().filter(|p| p.enabled) {
        for step in pipeline.steps.iter().filter(|s| s.enabled) {
            let Ok(mut commands) = expand_step(step, &project.modules, &mut scopes) else {
                continue;
            };
            let _ = namer.name_step(pipeline.id, step.id, &mut commands, &scopes);
        }
    }
    namer.named
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Two Thresholds in a pipeline keep their IDs; a module with two,
    /// used twice making the same class: each use names both, the second
    /// with `_2`.
    #[test]
    fn thresholds_in_a_pipeline_and_twice_in_a_module_used_twice_get_names_of_their_own() {
        let threshold = |id: &str| {
            serde_json::json!({"type": "THRESHOLD", "id": id, "thresholds": [
                {"method": {"type": "MANUAL"}, "minThreshold": 100.0, "maxThreshold": 65535.0,
                 "unit": "BIT", "segmentationClass": 1, "valueSource": "ACTUAL_IMAGE"}]})
        };
        let meta =
            serde_json::to_value(evanalyzer_cfg::settings::meta_data::MetaData::default()).unwrap();
        let mut project = ProjectSettings::default();
        project.modules.push(
            serde_json::from_value(serde_json::json!({
                "key": "EV", "meta": meta,
                "steps": [
                    {"id": 1, "enabled": true, "command": threshold("threshold_01")},
                    {"id": 2, "enabled": true, "command": threshold("threshold_02")},
                    {"id": 3, "enabled": true, "command": {"type": "EXTRACT_OBJECTS",
                        "maxObjectsBeforeFail": 1000,
                        "classMapping": [{"segmentationClass": 1, "objectClass": {"GLOBAL": 1}}]}}
                ],
                "parameters": [{"key": "evs", "label": "EVs", "default": "1",
                    "bindings": [{"step": 3, "param": "class_mapping.0.object_class"}]}]
            }))
            .unwrap(),
        );
        project.pipelines.push(
            serde_json::from_value(serde_json::json!({
                "id": 1, "name": "P", "imageSource": {"CHANNEL": 0}, "enabled": true,
                "steps": [
                    {"id": 1, "enabled": true, "command": threshold("threshold_01")},
                    {"id": 2, "enabled": true, "command": threshold("threshold_02")},
                    {"id": 3, "enabled": true, "module": {"key": "EV", "values": {}}},
                    {"id": 4, "enabled": true, "module": {"key": "EV", "values": {}}}
                ]
            }))
            .unwrap(),
        );
        project.classification.classes_mut().push(Class {
            id: 1,
            name: "EVs".into(),
            ..Default::default()
        });

        let names: Vec<(Vec<u32>, String)> = measurement_names(&project)
            .into_iter()
            .map(|n| (n.path.iter().map(|s| s.0).collect(), n.name))
            .collect();
        assert_eq!(
            names,
            [
                (vec![3, 1], "threshold_01_evs".to_string()),
                (vec![3, 2], "threshold_02_evs".to_string()),
                (vec![4, 1], "threshold_01_evs_2".to_string()),
                (vec![4, 2], "threshold_02_evs_2".to_string()),
            ]
        );
    }

    #[test]
    fn a_use_reference_goes_there_and_back() {
        let text = use_reference(PipelineId(3), &[StepId(4), StepId(2)], "threshold_01/t");
        assert_eq!(text, "@3.4.2:threshold_01/t");
        assert_eq!(
            parse_use_reference(&text),
            Some((PipelineId(3), vec![StepId(4), StepId(2)], "threshold_01/t"))
        );
        assert_eq!(parse_use_reference("threshold_01/t"), None);
        assert_eq!(parse_use_reference("@x:threshold_01/t"), None);
    }

    #[test]
    fn a_numbered_id_is_the_prefix_and_a_number() {
        assert!(is_numbered("threshold_01", "threshold"));
        assert!(is_numbered("Threshold_12", "threshold"));
        assert!(!is_numbered("threshold_", "threshold"));
        assert!(!is_numbered("threshold_dapi", "threshold"));
        assert!(!is_numbered("thresholds_01", "threshold"));
        assert!(!is_numbered("thr", "threshold"));
    }

    #[test]
    fn a_slug_has_lowercase_letters_digits_and_single_underscores() {
        assert_eq!(slug("EVs CY5"), "evs_cy5");
        assert_eq!(slug("Cy5-Cy7-Fret"), "cy5_cy7_fret");
        assert_eq!(slug("  Nuclei (DAPI) "), "nuclei_dapi");
        assert_eq!(slug("µ"), "");
    }
}
