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
use crate::migration::migration_v1_to_v2::migrate_from_v1_to_v2;
use crate::settings::ai_learning_settings::AiLearningSettings;
use crate::settings::project_settings::ProjectSettings;
use crate::{CURRENT_AI_LEARNING_SETTINGS_SCHEMA_VERSION, CURRENT_PROJECT_SCHEMA_VERSION};
use serde::de::DeserializeOwned;
use serde_json::Value;

/// Migration steps for [`ProjectSettings`], keyed on the version they migrate
/// *from*: index 0 migrates version 0 (pre-versioning legacy files) to 1, etc.
/// Each step's actual logic lives in its own `crate::migration::migration_vN_to_vM`
/// module - this list only wires them in, so a legacy step can be dropped later
/// by deleting its module and this one line, without touching anything else here.
const PROJECT_MIGRATIONS: &[MigrationStep] = &[
    |_raw| {}, // 0 -> 1: nothing to do yet
    migrate_from_v1_to_v2,
];

/// Loads a [`ProjectSettings`] from its raw on-disk JSON, migrating it
/// forward from whatever `schemaVersion` it was written with.
pub fn load_project_settings(raw: Value) -> Result<ProjectSettings, InternalErrors> {
    let version = schema_version_of(&raw, "schemaVersion");
    reject_if_too_new(version, CURRENT_PROJECT_SCHEMA_VERSION, "project")?;
    migrate_and_deserialize(raw, version, PROJECT_MIGRATIONS)
}

/// Migration steps for [`AiLearningSettings`] - see [`PROJECT_MIGRATIONS`].
/// Nothing to migrate yet; both steps are no-ops, but the list's length must
/// still track `CURRENT_AI_LEARNING_SETTINGS_SCHEMA_VERSION` (currently 2) -
/// `migrate_and_deserialize` slices `steps[from_version..]`, which panics for
/// a file already at the current version if this list is shorter than that.
const AI_LEARNING_SETTINGS_MIGRATIONS: &[MigrationStep] = &[
    |_raw| {}, // 0 -> 1: nothing to do yet
    |_raw| {}, // 1 -> 2: nothing to do yet
];

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

    // Version-1-to-2 migration logic (camelCase "type" tags, bare-string
    // ThresholdMethod) and its own tests live in
    // `crate::migration::migration_v1_to_v2` - see that module's doc comment.

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

    /// Regression test: `AI_LEARNING_SETTINGS_MIGRATIONS` must have one entry
    /// per version bump (matching `CURRENT_AI_LEARNING_SETTINGS_SCHEMA_VERSION`),
    /// same as `PROJECT_MIGRATIONS` - a file already at the current version
    /// (the common case: anything freshly saved) must load without running
    /// any migration step, not panic on an out-of-range slice.
    #[test]
    fn load_ai_learning_settings_accepts_the_current_version() {
        let mut raw = serde_json::to_value(sample_ai_learning_settings()).unwrap();
        raw["schemaVersion"] = serde_json::json!(CURRENT_AI_LEARNING_SETTINGS_SCHEMA_VERSION);

        let settings = load_ai_learning_settings(raw).expect("current version must load");
        assert_eq!(
            settings.schema_version,
            CURRENT_AI_LEARNING_SETTINGS_SCHEMA_VERSION
        );
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
