mod modules;
mod types;
mod utils;

// Constants
pub const PROJECT_FILE_EXTENSIONS: &str = &"improj";
pub const PROJECT_FILE_TEMPLATE_EXTENSIONS: &str = &"impt";

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
}
