//! Shared machinery for loading versioned, on-disk JSON documents (project
//! files, AI training settings, ...) and migrating older schema versions
//! forward before the final typed deserialize.
//!
//! Migrating the raw [`serde_json::Value`] first - rather than deserializing
//! straight into the target struct and patching it up afterwards - is what
//! lets a migration step handle a real breaking change (a renamed field, a
//! restructured shape), not just additive `#[serde(default)]` fields: the old
//! JSON never has to successfully deserialize into the *current* struct shape
//! on its own.
//!
//! Deliberately not `include!()`-d by `build.rs` (unlike `modules::*`) - it
//! only needs to exist in the real library build, not the build-script's
//! schema-generation copy.

use crate::core_types::InternalErrors;
use serde::de::DeserializeOwned;
use serde_json::Value;

/// A single migration step: mutates a document currently at version `N`
/// (its position in the step list) into version `N + 1`'s shape.
pub type MigrationStep = fn(&mut Value);

/// Reads `field` off `raw` as the document's on-disk schema version,
/// defaulting to 0 for documents written before the field existed.
pub fn schema_version_of(raw: &Value, field: &str) -> u32 {
    raw.get(field).and_then(Value::as_u64).unwrap_or(0) as u32
}

/// Runs every migration step from `from_version` onward over `raw`, then
/// deserializes the result into `T`.
///
/// Callers must reject `from_version > steps.len()` (a file written by a
/// newer build) before calling this - there's no step to run for a version
/// this build has never heard of, so `steps[from_version..]` would panic.
pub fn migrate_and_deserialize<T: DeserializeOwned>(
    mut raw: Value,
    from_version: u32,
    steps: &[MigrationStep],
) -> Result<T, InternalErrors> {
    for step in &steps[from_version as usize..] {
        step(&mut raw);
    }
    serde_json::from_value(raw).map_err(|e| InternalErrors::ParseError(e.to_string()))
}

/// Rejects a document whose `schema_version` is newer than this build
/// supports - proceeding would silently drop fields/variants this build
/// doesn't know about on the next save, so refuse instead of guessing.
pub fn reject_if_too_new(version: u32, current: u32, kind: &str) -> Result<(), InternalErrors> {
    if version > current {
        return Err(InternalErrors::ParseError(format!(
            "this {kind} was saved by a newer version of EVAnalyzer (format version {version}, this build supports up to version {current}) - update EVAnalyzer to open it"
        )));
    }
    Ok(())
}

use crate::settings::ai_learning_settings::AiLearningSettings;
use crate::settings::project_settings::ProjectSettings;
use crate::{CURRENT_AI_LEARNING_SETTINGS_SCHEMA_VERSION, CURRENT_PROJECT_SCHEMA_VERSION};

/// Migration steps for [`ProjectSettings`], keyed on the version they migrate
/// *from*: index 0 migrates version 0 (pre-versioning legacy files) to 1, etc.
/// Every field added to `ProjectSettings` so far has been an additive
/// `#[serde(default)]` field, so there is nothing to migrate yet - this is
/// the seam future breaking changes hook into, e.g.:
///
/// ```ignore
/// const PROJECT_MIGRATIONS: &[MigrationStep] = &[
///     |_raw| {},               // 0 -> 1: nothing to do yet
///     |raw| {                  // 1 -> 2: e.g. a field rename
///         if let Some(obj) = raw.as_object_mut() {
///             if let Some(v) = obj.remove("oldName") {
///                 obj.insert("newName".into(), v);
///             }
///         }
///     },
/// ];
/// ```
const PROJECT_MIGRATIONS: &[MigrationStep] = &[|_raw| {}];

/// Loads a [`ProjectSettings`] from its raw on-disk JSON, migrating it
/// forward from whatever `schemaVersion` it was written with.
pub fn load_project_settings(raw: Value) -> Result<ProjectSettings, InternalErrors> {
    let version = schema_version_of(&raw, "schemaVersion");
    reject_if_too_new(version, CURRENT_PROJECT_SCHEMA_VERSION, "project")?;
    migrate_and_deserialize(raw, version, PROJECT_MIGRATIONS)
}

/// Migration steps for [`AiLearningSettings`] - see [`PROJECT_MIGRATIONS`].
/// Nothing to migrate yet.
const AI_LEARNING_SETTINGS_MIGRATIONS: &[MigrationStep] = &[|_raw| {}];

/// Loads an [`AiLearningSettings`] from its raw on-disk JSON (a standalone
/// `--settings` file, or embedded in a saved classifier), migrating it
/// forward from whatever `schemaVersion` it was written with.
pub fn load_ai_learning_settings(raw: Value) -> Result<AiLearningSettings, InternalErrors> {
    let version = schema_version_of(&raw, "schemaVersion");
    reject_if_too_new(
        version,
        CURRENT_AI_LEARNING_SETTINGS_SCHEMA_VERSION,
        "AI training settings",
    )?;
    migrate_and_deserialize(raw, version, AI_LEARNING_SETTINGS_MIGRATIONS)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::settings::ai_learning_object_settings::AiLearningObjectFeatureSettings;
    use crate::settings::ai_learning_settings::{
        AiLearningBackendSettings, AiLearningClassifierSettings, RandomForestSettings,
    };
    use crate::settings::meta_data::MetaData;

    #[test]
    fn schema_version_of_defaults_to_zero_when_absent() {
        let raw = serde_json::json!({});
        assert_eq!(schema_version_of(&raw, "schemaVersion"), 0);
    }

    #[test]
    fn load_project_settings_rejects_a_newer_schema_version() {
        let mut raw = serde_json::to_value(ProjectSettings::default()).unwrap();
        raw["schemaVersion"] = serde_json::json!(CURRENT_PROJECT_SCHEMA_VERSION + 1);

        let err = load_project_settings(raw).expect_err("newer version must be rejected");
        assert!(err.to_string().contains("newer version"));
    }

    #[test]
    fn load_project_settings_accepts_the_current_version() {
        let mut raw = serde_json::to_value(ProjectSettings::default()).unwrap();
        raw["schemaVersion"] = serde_json::json!(CURRENT_PROJECT_SCHEMA_VERSION);

        let settings = load_project_settings(raw).expect("current version must load");
        assert_eq!(settings.schema_version, CURRENT_PROJECT_SCHEMA_VERSION);
    }

    fn sample_ai_learning_settings() -> AiLearningSettings {
        AiLearningSettings {
            schema_version: CURRENT_AI_LEARNING_SETTINGS_SCHEMA_VERSION,
            metadata: MetaData::default(),
            backend: AiLearningBackendSettings::RandomForest(RandomForestSettings::default()),
            classifier: AiLearningClassifierSettings::Object {
                feature_spec: AiLearningObjectFeatureSettings { metrics: vec![] },
                class_labels: vec![],
            },
        }
    }

    #[test]
    fn load_ai_learning_settings_rejects_a_newer_schema_version() {
        let mut raw = serde_json::to_value(sample_ai_learning_settings()).unwrap();
        raw["schemaVersion"] = serde_json::json!(CURRENT_AI_LEARNING_SETTINGS_SCHEMA_VERSION + 1);

        let err = load_ai_learning_settings(raw).expect_err("newer version must be rejected");
        assert!(err.to_string().contains("newer version"));
    }

    #[test]
    fn load_ai_learning_settings_defaults_missing_version_to_zero_and_still_loads() {
        let mut raw = serde_json::to_value(sample_ai_learning_settings()).unwrap();
        raw.as_object_mut().unwrap().remove("schemaVersion");

        let settings = load_ai_learning_settings(raw)
            .expect("a pre-versioning file (implicit version 0) must still load");
        assert_eq!(settings.schema_version, 0);
    }
}
