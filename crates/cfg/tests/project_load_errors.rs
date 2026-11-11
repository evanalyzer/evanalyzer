//! # project_load_errors
//!
//! Tests that a project file which does not load says where it is broken:
//! every broken step at once, each with its pipeline, its step and the
//! field - not only the first serde error, without a place.
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-11-11
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.

use evanalyzer_cfg::CURRENT_PROJECT_SCHEMA_VERSION;
use evanalyzer_cfg::core_types::{ImageAddress, InternalErrors, PipelineId, StepId};
use evanalyzer_cfg::load_project_settings_from_str;
use evanalyzer_cfg::settings::pipeline_command::PipelineCommand;
use evanalyzer_cfg::settings::pipeline_settings::{
    PipelineSettings, PipelineStepSettings, StepContent,
};
use evanalyzer_cfg::settings::project_settings::ProjectSettings;
use serde_json::{Value, json};

fn step(id: u32, key: &str) -> PipelineStepSettings {
    PipelineStepSettings {
        id: StepId(id),
        enabled: true,
        content: StepContent::Command(PipelineCommand::default_for_key(key).unwrap()),
    }
}

fn pipeline(id: u32, name: &str, steps: Vec<PipelineStepSettings>) -> PipelineSettings {
    PipelineSettings {
        id: PipelineId(id),
        name: name.into(),
        description: None,
        image_source: ImageAddress::Channel(0),
        enabled: true,
        steps,
    }
}

/// A valid project with two pipelines, as written on save.
fn valid_project() -> Value {
    let mut project = ProjectSettings::default();
    project.schema_version = CURRENT_PROJECT_SCHEMA_VERSION;
    project.pipelines.push(pipeline(
        1,
        "Nuclei",
        vec![step(1, "blur"), step(2, "watershed")],
    ));
    project
        .pipelines
        .push(pipeline(2, "Cells", vec![step(1, "watershed")]));
    serde_json::to_value(&project).unwrap()
}

/// The load error's message (the test fails if the project loads).
fn load_error(raw: Value) -> String {
    match load_project_settings_from_str(&raw.to_string()) {
        Ok(_) => panic!("the broken project loaded"),
        Err(InternalErrors::ParseError(msg)) => msg,
        Err(other) => panic!("expected a ParseError, got {other:?}"),
    }
}

/// The unbroken project loads - so the failures below come from the
/// broken fields only.
#[test]
fn the_valid_project_loads() {
    load_project_settings_from_str(&valid_project().to_string()).expect("the valid project loads");
}

/// One broken field: the message names the pipeline, the step and the field.
#[test]
fn a_broken_field_names_its_pipeline_step_and_field() {
    let mut raw = valid_project();
    raw["pipelines"][0]["steps"][1]["command"]["smoothingSigma"] = json!("soft");

    let msg = load_error(raw);
    println!("{msg}");
    assert!(msg.contains("smoothingSigma"), "no field in: {msg}");
    assert!(msg.contains("Nuclei"), "no pipeline name in: {msg}");
    assert!(msg.contains("step 2"), "no step in: {msg}");
}

/// A required field left out (as in the report: "missing field ..."): the
/// message names the pipeline and the step, not only the field.
#[test]
fn a_missing_field_names_its_pipeline_and_step() {
    let mut raw = valid_project();
    raw["pipelines"][1]["steps"][0]
        .as_object_mut()
        .unwrap()
        .remove("enabled");

    let msg = load_error(raw);
    println!("{msg}");
    assert!(msg.contains("enabled"), "no field in: {msg}");
    assert!(msg.contains("Cells"), "no pipeline name in: {msg}");
    assert!(msg.contains("step 1"), "no step in: {msg}");
}

/// Broken fields in two pipelines and an unknown command: all three are
/// reported in one go, each at its place.
#[test]
fn every_broken_step_is_reported_at_once() {
    let mut raw = valid_project();
    raw["pipelines"][0]["steps"][1]["command"]["smoothingSigma"] = json!("soft");
    raw["pipelines"][1]["steps"][0]["command"]["smoothingSigma"] = json!([1, 2]);
    raw["pipelines"][0]["steps"][0]["command"]["type"] = json!("NOT_A_COMMAND");

    let msg = load_error(raw);
    println!("{msg}");
    assert!(
        msg.contains("NOT_A_COMMAND"),
        "unknown command missing: {msg}"
    );
    assert!(msg.contains("Cells"), "second pipeline missing: {msg}");
    assert_eq!(
        msg.matches("smoothingSigma").count(),
        2,
        "both broken smoothingSigma fields expected: {msg}"
    );
}

/// A misspelt command gets the closest one suggested, not the list of
/// every command in every spelling.
#[test]
fn a_misspelt_command_gets_a_suggestion() {
    let mut raw = valid_project();
    raw["pipelines"][1]["steps"][0]["command"]["type"] = json!("WATERSHD");

    let msg = load_error(raw);
    assert!(
        msg.contains("unknown command `WATERSHD` - did you mean `WATERSHED`?"),
        "{msg}"
    );
    assert!(!msg.contains("expected one of"), "{msg}");
}

/// A broken field outside the pipelines keeps its path.
#[test]
fn a_broken_field_outside_the_pipelines_names_its_path() {
    let mut raw = valid_project();
    raw["images"]["root"] = json!(5);

    let msg = load_error(raw);
    assert!(msg.contains("images.root: invalid type"), "{msg}");
}

/// A module's step is named by the module's key.
#[test]
fn a_broken_module_step_names_its_module() {
    let mut raw = valid_project();
    let mut module = serde_json::to_value(evanalyzer_cfg::settings::modules::ModuleDefinition {
        key: "SEGMENT".into(),
        meta: Default::default(),
        classes: vec![],
        steps: vec![step(1, "watershed")],
        parameters: vec![],
        origin: None,
    })
    .unwrap();
    module["steps"][0]["command"]["smoothingSigma"] = json!("soft");
    raw["modules"] = json!([module]);

    let msg = load_error(raw);
    assert!(
        msg.contains("module “SEGMENT”, step 1 (WATERSHED): smoothingSigma"),
        "{msg}"
    );
}

/// Many broken steps: the first 20 are listed, the rest counted.
#[test]
fn many_problems_are_capped() {
    let mut raw = valid_project();
    let watershed = raw["pipelines"][1]["steps"][0].clone();
    let steps: Vec<Value> = (0..25)
        .map(|i| {
            let mut step = watershed.clone();
            step["id"] = json!(i + 1);
            step["command"]["smoothingSigma"] = json!("soft");
            step
        })
        .collect();
    raw["pipelines"][1]["steps"] = json!(steps);

    let msg = load_error(raw);
    assert!(msg.contains("25 problems"), "{msg}");
    assert!(msg.contains("step 20 (WATERSHED)"), "{msg}");
    assert!(!msg.contains("step 21 (WATERSHED)"), "{msg}");
    assert!(msg.ends_with("and 5 more"), "{msg}");
}
