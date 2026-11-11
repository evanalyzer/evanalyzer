//! # load_problems
//!
//! Says where a project file that does not load is broken. serde stops at
//! its first error and, in a step's command (an internally tagged enum
//! behind a flattened field), loses the place: "invalid type: string,
//! expected f32" names neither the pipeline, the step nor the field. This
//! walks the raw JSON instead and loads every part on its own: each
//! pipeline and module without its steps, each step, each field of a
//! step's command against the command's default, and the rest of the
//! project - so one message lists every problem at its place.
//!
//! Only runs after a load failed: the normal load is untouched.
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-11-11
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.

use crate::settings::modules::{ModuleDefinition, ModuleInstance};
use crate::settings::pipeline_command::{PipelineCommand, all_command_meta, default_command};
use crate::settings::pipeline_settings::{PipelineSettings, PipelineStepSettings};
use crate::settings::project_settings::ProjectSettings;
use serde::Deserialize;
use serde::de::DeserializeOwned;
use serde_json::{Map, Value};

/// Up to this many variants an "unknown variant" message lists the
/// expected ones; a longer list (e.g. every command in every spelling)
/// is left out.
const LISTED_VARIANTS: usize = 12;

/// Every problem of a (migrated) project file, each starting with its
/// place ("pipeline “Nuclei”, step 2 (WATERSHED): smoothingSigma: ...").
/// Empty if each part loads on its own.
pub(crate) fn project_problems(raw: &Value) -> Vec<String> {
    let Some(project) = raw.as_object() else {
        return error_of::<ProjectSettings>(raw).into_iter().collect();
    };
    let mut problems = Vec::new();
    for (i, pipeline) in array(project, "pipelines").iter().enumerate() {
        let place = match pipeline.get("name").and_then(Value::as_str) {
            Some(name) => format!("pipeline “{name}”"),
            None => format!("pipeline {}", i + 1),
        };
        problems.extend(list_problems::<PipelineSettings>(&place, pipeline));
    }
    for (i, module) in array(project, "modules").iter().enumerate() {
        let place = match module.get("key").and_then(Value::as_str) {
            Some(key) => format!("module “{key}”"),
            None => format!("module {}", i + 1),
        };
        problems.extend(list_problems::<ModuleDefinition>(&place, module));
    }
    // The rest, without the pipelines and modules checked above.
    let mut rest = project.clone();
    for key in ["pipelines", "modules"] {
        if let Some(list) = rest.get_mut(key).filter(|v| v.is_array()) {
            *list = Value::Array(Vec::new());
        }
    }
    problems.extend(error_of::<ProjectSettings>(&Value::Object(rest)));
    problems
}

/// A pipeline's or module's (`T`) problems: its own fields (loaded with
/// no steps), then each step's.
fn list_problems<T: DeserializeOwned>(place: &str, list: &Value) -> Vec<String> {
    let Some(obj) = list.as_object() else {
        return error_of::<T>(list)
            .map(|e| format!("{place}: {e}"))
            .into_iter()
            .collect();
    };
    let mut problems = Vec::new();
    let mut shell = obj.clone();
    if let Some(steps) = shell.get_mut("steps").filter(|v| v.is_array()) {
        *steps = Value::Array(Vec::new());
    }
    if let Some(e) = error_of::<T>(&Value::Object(shell)) {
        problems.push(format!("{place}: {e}"));
    }
    for (j, step) in array(obj, "steps").iter().enumerate() {
        let command = step
            .get("command")
            .and_then(|c| c.get("type"))
            .and_then(Value::as_str);
        let step_place = match command {
            Some(tag) => format!("{place}, step {} ({tag})", j + 1),
            None => format!("{place}, step {}", j + 1),
        };
        for problem in step_problems(step) {
            problems.push(format!("{step_place}: {problem}"));
        }
    }
    problems
}

/// A step's problems: its command's (or module use's), then its own
/// fields (`id`, `enabled`) with a valid command in place of its own.
fn step_problems(step: &Value) -> Vec<String> {
    if PipelineStepSettings::deserialize(step).is_ok() {
        return Vec::new();
    }
    let Some(obj) = step.as_object() else {
        return error_of::<PipelineStepSettings>(step).into_iter().collect();
    };
    let mut problems = Vec::new();
    match (obj.get("command"), obj.get("module")) {
        (Some(command), None) => problems.extend(command_problems(command)),
        (None, Some(module)) => problems.extend(error_of::<ModuleInstance>(module)),
        (Some(_), Some(_)) => problems.push("has both a command and a module".into()),
        (None, None) => problems.push("has neither a command nor a module".into()),
    }
    let mut shell = obj.clone();
    shell.remove("module");
    shell.insert("command".into(), any_valid_command());
    if let Some(e) = error_of::<PipelineStepSettings>(&Value::Object(shell)) {
        problems.push(e);
    }
    if problems.is_empty() {
        // Not found part by part - the step's own error, as it is.
        problems.extend(error_of::<PipelineStepSettings>(step));
    }
    problems
}

/// A command's problems, field by field: each field the file has, put
/// into the command's default, must load; each field it lacks must
/// have a default.
fn command_problems(command: &Value) -> Vec<String> {
    if PipelineCommand::deserialize(command).is_ok() {
        return Vec::new();
    }
    let Some(obj) = command.as_object() else {
        return error_of::<PipelineCommand>(command).into_iter().collect();
    };
    let Some(tag) = obj.get("type").and_then(Value::as_str) else {
        return error_of::<PipelineCommand>(command).into_iter().collect();
    };
    let Some(default) = default_of(tag) else {
        let tags = command_tags();
        return vec![match closest(tag, tags.iter().map(String::as_str)) {
            Some(near) => format!("unknown command `{tag}` - did you mean `{near}`?"),
            None => format!("unknown command `{tag}`"),
        }];
    };
    let mut problems = Vec::new();
    for (field, value) in obj {
        if field == "type" {
            continue;
        }
        let mut probe = default.clone();
        probe.insert(field.clone(), value.clone());
        if let Err(e) = PipelineCommand::deserialize(&Value::Object(probe)) {
            problems.push(format!("{field}: {}", shorten(&e.to_string())));
        }
    }
    for field in default.keys().filter(|f| !obj.contains_key(*f)) {
        let mut probe = default.clone();
        probe.remove(field);
        if PipelineCommand::deserialize(&Value::Object(probe)).is_err() {
            problems.push(format!("missing field `{field}`"));
        }
    }
    if problems.is_empty() {
        problems.extend(error_of::<PipelineCommand>(command));
    }
    problems
}

/// The default command, as written on save, whose `type` is `tag` (in any
/// spelling the loader accepts) - with that spelling, so a probe keeps it.
fn default_of(tag: &str) -> Option<Map<String, Value>> {
    all_command_meta().into_iter().find_map(|meta| {
        let command = default_command(meta.id)?;
        let Value::Object(mut obj) = serde_json::to_value(&command).ok()? else {
            return None;
        };
        let canonical = obj.get("type")?.as_str()?.to_string();
        obj.insert("type".into(), Value::String(tag.to_string()));
        let loaded = PipelineCommand::deserialize(&Value::Object(obj.clone())).ok()?;
        let same = serde_json::to_value(&loaded).ok()?.get("type")?.as_str()? == canonical;
        same.then_some(obj)
    })
}

/// Every command's `type`, as written on save.
fn command_tags() -> Vec<String> {
    all_command_meta()
        .into_iter()
        .filter_map(|meta| {
            let value = serde_json::to_value(default_command(meta.id)?).ok()?;
            Some(value.get("type")?.as_str()?.to_string())
        })
        .collect()
}

/// A command that loads, standing in for a step's own while its other
/// fields are checked.
fn any_valid_command() -> Value {
    let command = default_command(all_command_meta()[0].id).expect("a default command");
    serde_json::to_value(command).expect("a default command serializes")
}

/// `T`'s load error of `value`, with the path to the broken element when
/// serde keeps it ("images.root: invalid type ..."); `None` if it loads.
fn error_of<T: DeserializeOwned>(value: &Value) -> Option<String> {
    let error = serde_path_to_error::deserialize::<_, T>(value).err()?;
    let message = shorten(&error.inner().to_string());
    let path = error.path().to_string();
    Some(if path.is_empty() || path == "." {
        message
    } else {
        format!("{path}: {message}")
    })
}

/// The list's elements, or none if `key` is missing or no list.
fn array<'a>(obj: &'a Map<String, Value>, key: &str) -> &'a [Value] {
    obj.get(key)
        .and_then(Value::as_array)
        .map(Vec::as_slice)
        .unwrap_or_default()
}

/// serde's "unknown variant `x`, expected one of ..." without a long list
/// of expected variants, but with the closest one if it looks like a typo.
fn shorten(message: &str) -> String {
    const EXPECTED: &str = ", expected one of ";
    let Some(rest) = message.strip_prefix("unknown variant `") else {
        return message.to_string();
    };
    let Some((variant, expected)) = rest.split_once('`') else {
        return message.to_string();
    };
    let Some(list) = expected.strip_prefix(EXPECTED) else {
        return message.to_string();
    };
    let names: Vec<&str> = list
        .split(", ")
        .map(|n| n.trim_matches('`'))
        .filter(|n| !n.is_empty())
        .collect();
    let suggestion = closest(variant, names.iter().copied())
        .map(|near| format!(" - did you mean `{near}`?"))
        .unwrap_or_default();
    if names.len() <= LISTED_VARIANTS {
        format!("{message}{suggestion}")
    } else {
        format!("unknown variant `{variant}`{suggestion}")
    }
}

/// The name closest to `typed` (case aside) if it is near enough to be a
/// typo of it: at most a third of its letters off.
fn closest<'a>(typed: &str, names: impl Iterator<Item = &'a str>) -> Option<&'a str> {
    let typed = typed.to_lowercase();
    let limit = (typed.chars().count() / 3).max(1);
    names
        .map(|name| (edit_distance(&typed, &name.to_lowercase()), name))
        .filter(|(distance, _)| *distance <= limit)
        .min_by_key(|(distance, _)| *distance)
        .map(|(_, name)| name)
}

/// Levenshtein distance: the fewest single-letter inserts, deletes and
/// substitutions that turn `a` into `b`.
fn edit_distance(a: &str, b: &str) -> usize {
    let b: Vec<char> = b.chars().collect();
    let mut row: Vec<usize> = (0..=b.len()).collect();
    for (i, ca) in a.chars().enumerate() {
        let mut diagonal = row[0];
        row[0] = i + 1;
        for (j, cb) in b.iter().enumerate() {
            let above = row[j + 1];
            row[j + 1] = (above + 1)
                .min(row[j] + 1)
                .min(diagonal + usize::from(ca != *cb));
            diagonal = above;
        }
    }
    row[b.len()]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn edit_distance_counts_single_letter_changes() {
        assert_eq!(edit_distance("watershed", "watershed"), 0);
        assert_eq!(edit_distance("watershd", "watershed"), 1);
        assert_eq!(edit_distance("", "abc"), 3);
        assert_eq!(edit_distance("kitten", "sitting"), 3);
    }

    #[test]
    fn closest_suggests_a_typo_but_not_something_else() {
        let names = ["WATERSHED", "THRESHOLD", "BLUR"];
        assert_eq!(closest("watershd", names.into_iter()), Some("WATERSHED"));
        assert_eq!(closest("NOT_A_COMMAND", names.into_iter()), None);
    }

    #[test]
    fn a_long_variant_list_is_left_out_a_short_one_kept() {
        let many: Vec<String> = (0..20).map(|i| format!("`V{i}`")).collect();
        let long = format!("unknown variant `X`, expected one of {}", many.join(", "));
        assert_eq!(shorten(&long), "unknown variant `X`");

        let short = "unknown variant `OTSUU`, expected one of `OTSU`, `MANUAL`";
        assert_eq!(shorten(short), format!("{short} - did you mean `OTSU`?"));
    }
}
