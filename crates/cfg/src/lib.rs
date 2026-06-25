mod modules;
mod types;
mod utils;

// Constants
pub const PROJECT_FILE_EXTENSIONS: &str = &"evaproj";
pub const PROJECT_FILE_TEMPLATE_EXTENSIONS: &str = &"evapt";
pub const PIPELINE_EXTENSIONS: &str = &"evapipe";
pub const RESULTS_FILE_EXTENSION: &str = &"evadb";

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

    /// Regression test for the generator's "rich enum" support: a `TransformFunction`-like
    /// field whose variants each carry their own sub-fields must change *which* parameters
    /// `to_parameters()` returns when the discriminant is switched via `apply_param_change`,
    /// not just relabel a fixed slot. This is what `pipelines_controller.rs` relies on to
    /// detect a structural change and trigger a full resync instead of patching one row.
    mod rich_enum_settings {
        use crate::settings::pipeline_command::PipelineCommand;
        use crate::settings::pipeline_command_settings::TransformRoisSettings;

        fn names(cmd: &PipelineCommand) -> Vec<String> {
            cmd.to_parameters().into_iter().map(|p| p.name).collect()
        }

        #[test]
        fn switching_variant_changes_the_field_set() {
            let mut cmd = PipelineCommand::TransformRois(TransformRoisSettings::default());
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
            let mut cmd = PipelineCommand::TransformRois(TransformRoisSettings::default());
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
            let mut cmd = PipelineCommand::TransformRois(TransformRoisSettings::default());
            assert_eq!(cmd.to_summary(), "Function: Scale");
            cmd.apply_param_change("function", "Fitting Ellipse");
            assert_eq!(cmd.to_summary(), "Function: Fitting Ellipse");
        }

        /// The on-disk project-file format: serde's internal tagging (`#[serde(tag = "type")]`)
        /// must round-trip through JSON exactly as schemars advertises it (a `type` discriminant
        /// alongside the variant's own fields), not the externally-tagged shape plain enums use.
        #[test]
        fn json_round_trips_through_the_internally_tagged_shape() {
            let mut cmd = PipelineCommand::TransformRois(TransformRoisSettings::default());
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
