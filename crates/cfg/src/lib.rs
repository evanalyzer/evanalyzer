mod legacy_import;
mod legacy_schema;
mod modules;
mod types;
mod utils;

pub use legacy_import::{LegacyImportError, LegacyImportOutcome, import_legacy_project};

// Constants
pub const PROJECT_FILE_EXTENSIONS: &str = &"evaproj";
pub const PROJECT_FILE_TEMPLATE_EXTENSIONS: &str = &"evapt";
pub const PIPELINE_EXTENSIONS: &str = &"evapipe";
pub const RESULTS_FILE_EXTENSION: &str = &"evadb";
pub const EVANALYZER_TRAINED_AI_MODELS: &str = &"evamodel";
/// Project file extension used by the old (pre-rewrite) application.
pub const LEGACY_PROJECT_FILE_EXTENSION: &str = &"icproj";

/// Current on-disk format version for [`settings::project_settings::ProjectSettings`].
/// Bump this and add a case to `ProjectSettings::migrate` whenever a change
/// to `ProjectSettings` (or something it contains) would break
/// deserialization of `.evaproj` files written by an older version of the
/// app - e.g. a renamed field/enum variant that isn't just an additive
/// `#[serde(default)]` field.
pub const CURRENT_PROJECT_SCHEMA_VERSION: u32 = 1;

// Project Settings structs
pub mod settings {
    pub use super::modules::*;
}

// Shared types
pub mod core_types {
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
    use crate::settings::project_settings::ProjectSettings;
    use std::path::PathBuf;

    #[test]
    fn example_project_deserializes() {
        let path =
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/example.improj");
        let json = std::fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()));
        let project: ProjectSettings =
            serde_json::from_str(&json).expect("example.improj failed to deserialize");
        assert_eq!(project.pipelines.len(), 2);
        assert_eq!(project.pipelines[0].steps.len(), 7);
        assert_eq!(project.pipelines[1].steps.len(), 5);
        assert_eq!(project.classification.classes.len(), 3);
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
                serde_json::json!({"type": "snapArea", "extraSize": 7.5, "unit": "nm"})
            );

            let round_tripped: PipelineCommand = serde_json::from_value(json).unwrap();
            assert_eq!(round_tripped.to_summary(), cmd.to_summary());
        }
    }
}
