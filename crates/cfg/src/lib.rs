mod legacy_import;
mod legacy_schema;
mod load_config;
mod migration;
mod modules;
mod types;
mod utils;

pub use legacy_import::{LegacyImportError, LegacyImportOutcome, import_legacy_project};
pub use load_config::{load_ai_learning_settings, load_project_settings};

// Constants
pub const PROJECT_FILE_EXTENSIONS: &str = &"evaproj";
pub const PROJECT_FILE_TEMPLATE_EXTENSIONS: &str = &"evapt";
pub const PIPELINE_EXTENSIONS: &str = &"evapipe";
pub const RESULTS_FILE_EXTENSION: &str = &"evadb";
pub const EVANALYZER_TRAINED_AI_MODELS: &str = &"evamodel";
/// Project file extension used by the old (pre-rewrite) application.
pub const LEGACY_PROJECT_FILE_EXTENSION: &str = &"icproj";

/// Current on-disk format version for [`settings::project_settings::ProjectSettings`].
/// Bump this, add a new `crate::migration::migration_vN_to_vM` module, and
/// wire it into `load_config::PROJECT_MIGRATIONS` whenever a change to
/// `ProjectSettings` (or something it contains) would break deserialization
/// of `.evaproj` files written by an older version of the app - e.g. a
/// renamed field/enum variant that isn't just an additive
/// `#[serde(default)]` field.
pub const CURRENT_PROJECT_SCHEMA_VERSION: u32 = 2;

/// Current on-disk format version for
/// [`settings::ai_learning_settings::AiLearningSettings`] - both the
/// standalone `--settings` file `train-classifier` reads and the copy
/// embedded in every saved `.evamodel` classifier. Bump alongside a new
/// migration step in `load_config::AI_LEARNING_SETTINGS_MIGRATIONS` whenever a
/// change to `AiLearningSettings` would break deserialization of a file
/// written by an older version of the app.
pub const CURRENT_AI_LEARNING_SETTINGS_SCHEMA_VERSION: u32 = 2;

/// Current on-disk format version for
/// [`settings::templates::ProjectTemplate`] (`.evapt` files). Unlike
/// `ProjectSettings`/`AiLearningSettings`, nothing currently loads templates
/// through a version-checked/migrated path (see that struct's
/// `schema_version` doc comment) - this only reserves the version number for
/// when that's added.
pub const CURRENT_PROJECT_TEMPLATE_SCHEMA_VERSION: u32 = 2;

/// Current on-disk format version for
/// [`settings::templates::PipelineTemplate`] (`.evapipe` files, and each
/// entry of a `ProjectTemplate`'s `pipelines`). See
/// `CURRENT_PROJECT_TEMPLATE_SCHEMA_VERSION`.
pub const CURRENT_PIPELINE_TEMPLATE_SCHEMA_VERSION: u32 = 2;

// Project Settings structs
pub mod settings {
    pub use super::modules::*;
}

// Shared types
pub mod core_types {
    pub use crate::types::cite::CitationMetadata;
    pub use crate::types::classes::ObjectClass;
    pub use crate::types::classes::SegmentationClass;
    pub use crate::types::errors::*;
    pub use crate::types::ids::ImageAddress;
    pub use crate::types::ids::MemoryId;
    pub use crate::types::ids::MemorySlot;
    pub use crate::types::ids::ObjectId;
    pub use crate::types::ids::PipelineId;
    pub use crate::types::ids::TrackId;
    pub use crate::types::units::PixelUnits;
    pub use crate::types::units::SizeUnits;
}

#[cfg(test)]
mod tests {
    use crate::load_project_settings;
    use std::path::PathBuf;

    /// `example.improj` is a real version-1 fixture (no `schemaVersion`
    /// field, camelCase command tags, bare-string `ThresholdMethod`
    /// including a plain `"OTSU"`) - loading it through
    /// [`load_project_settings`] rather than a raw `serde_json::from_str`
    /// exercises the actual migration path real saved projects go through,
    /// not just the current struct shape.
    #[test]
    fn example_project_deserializes() {
        let path =
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/example.improj");
        let json = std::fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()));
        let raw: serde_json::Value = serde_json::from_str(&json)
            .unwrap_or_else(|e| panic!("{} is not valid JSON: {e}", path.display()));
        let project =
            load_project_settings(raw).expect("example.improj failed to deserialize/migrate");
        assert_eq!(project.pipelines.len(), 2);
        assert_eq!(project.pipelines[0].steps.len(), 7);
        assert_eq!(project.pipelines[1].steps.len(), 5);
        assert_eq!(project.classification.classes().len(), 3);
    }

    /// `PipelineSettings.name` went from `Option<String>` to `String` with no
    /// migration step added. This is an accepted, deliberate break: real
    /// project files (schema versions 1 and 2) that have a pipeline with
    /// `"name": null` - as `example.improj` itself did, before this test's
    /// sibling fixed that up to keep exercising the rest of the v1-to-v2
    /// migration - no longer load. If this starts passing, either a migration
    /// was added (this test can be deleted) or the null case was
    /// reintroduced silently (worth double-checking that was intentional).
    #[test]
    fn null_pipeline_name_is_rejected_not_migrated() {
        let raw = serde_json::json!({
            "pipelines": [{
                "id": 1,
                "name": null,
                "imageSource": "SCRATCHPAD",
                "enabled": true,
                "steps": []
            }]
        });
        let err = load_project_settings(raw).expect_err("null pipeline name should fail to load");
        assert!(
            err.to_string()
                .contains("invalid type: null, expected a string"),
            "unexpected error: {err}"
        );
    }

    /// All shipped `.evapt`/`.evapipe` template files must deserialize into
    /// `ProjectTemplate`/`PipelineTemplate` - the same types and the same
    /// `serde_json::from_str` call the GUI/CLI actually use to load them. Unlike
    /// validating against `docs/project.schema.json` (which describes the full
    /// `ProjectSettings` project-file format, not these template formats), this
    /// exercises the real deserialization path, so it would have caught templates
    /// genuinely failing to load.
    #[test]
    fn shipped_templates_deserialize() {
        use crate::settings::templates::{PipelineTemplate, ProjectTemplate};

        let templates_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../templates");
        let mut checked_evapt = 0;
        let mut checked_evapipe = 0;

        for entry in std::fs::read_dir(&templates_dir)
            .unwrap_or_else(|e| panic!("cannot read {}: {e}", templates_dir.display()))
        {
            let path = entry.unwrap().path();
            let Some(ext) = path.extension().and_then(|e| e.to_str()) else {
                continue;
            };
            let json = std::fs::read_to_string(&path)
                .unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()));

            match ext {
                "evapt" => {
                    serde_json::from_str::<ProjectTemplate>(&json).unwrap_or_else(|e| {
                        panic!(
                            "{} failed to deserialize as ProjectTemplate: {e}",
                            path.display()
                        )
                    });
                    checked_evapt += 1;
                }
                "evapipe" => {
                    serde_json::from_str::<PipelineTemplate>(&json).unwrap_or_else(|e| {
                        panic!(
                            "{} failed to deserialize as PipelineTemplate: {e}",
                            path.display()
                        )
                    });
                    checked_evapipe += 1;
                }
                _ => {}
            }
        }

        assert!(
            checked_evapt > 0,
            "no .evapt files found under {}",
            templates_dir.display()
        );
        assert!(
            checked_evapipe > 0,
            "no .evapipe files found under {}",
            templates_dir.display()
        );
    }

    /// Regression test for the generator's "rich enum" support: a `TransformFunction`-like
    /// field whose variants each carry their own sub-fields must change *which* parameters
    /// `to_parameters()` returns when the discriminant is switched via `apply_param_change`,
    /// not just relabel a fixed slot. This is what `pipelines_controller.rs` relies on to
    /// detect a structural change and trigger a full resync instead of patching one row.
    mod rich_enum_settings {
        use crate::settings::pipeline_command::PipelineCommand;
        use crate::settings::pipeline_command_settings::TransformObjectsSettings;

        fn names(cmd: &PipelineCommand) -> Vec<String> {
            cmd.to_parameters().into_iter().map(|p| p.name).collect()
        }

        #[test]
        fn switching_variant_changes_the_field_set() {
            let mut cmd = PipelineCommand::TransformObjects(TransformObjectsSettings::default());
            assert_eq!(
                names(&cmd),
                vec!["function", "function.factor", "input_class", "output_class"],
                "Scale (the default variant) only exposes its own factor field"
            );

            cmd.apply_param_change("function", "Snap Area");
            assert_eq!(
                names(&cmd),
                vec![
                    "function",
                    "function.extra_size",
                    "function.unit",
                    "input_class",
                    "output_class"
                ],
                "switching to SnapArea must swap in its own fields, not keep Scale's factor"
            );
        }

        #[test]
        fn setting_a_variant_field_does_not_change_the_active_variant() {
            let mut cmd = PipelineCommand::TransformObjects(TransformObjectsSettings::default());
            cmd.apply_param_change("function", "Draw Circle");
            cmd.apply_param_change("function.diameter", "12.5");

            let params = cmd.to_parameters();
            let diameter = params
                .iter()
                .find(|p| p.name == "function.diameter")
                .expect("diameter field must still be present after editing it");
            assert_eq!(diameter.value, "12.5");
            assert!(
                params.iter().any(|p| p.name == "function.unit"),
                "DrawCircle's other field must survive editing a sibling field"
            );
        }

        #[test]
        fn to_summary_reflects_the_active_variant() {
            let mut cmd = PipelineCommand::TransformObjects(TransformObjectsSettings::default());
            assert_eq!(cmd.to_summary(), "Function: Scale");
            cmd.apply_param_change("function", "Fitting Ellipse");
            assert_eq!(cmd.to_summary(), "Function: Fitting Ellipse");
        }

        /// The on-disk project-file format: serde's internal tagging (`#[serde(tag = "type")]`)
        /// must round-trip through JSON exactly as schemars advertises it (a `type` discriminant
        /// alongside the variant's own fields), not the externally-tagged shape plain enums use.
        #[test]
        fn json_round_trips_through_the_internally_tagged_shape() {
            let mut cmd = PipelineCommand::TransformObjects(TransformObjectsSettings::default());
            cmd.apply_param_change("function", "Snap Area");
            cmd.apply_param_change("function.extra_size", "7.5");

            let json = serde_json::to_value(&cmd).unwrap();
            assert_eq!(
                json["function"],
                serde_json::json!({"type": "SNAP_AREA", "extraSize": 7.5, "unit": "NANO_METER"})
            );

            let round_tripped: PipelineCommand = serde_json::from_value(json).unwrap();
            assert_eq!(round_tripped.to_summary(), cmd.to_summary());
        }
    }
}
