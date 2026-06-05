// Module tree mirroring src/ so `crate::` references inside include!()-d
// source files resolve correctly against the build-script crate root.
// We use include!(concat!(env!("CARGO_MANIFEST_DIR"), ...)) to produce
// absolute paths, avoiding #[path = "../"] traversal through non-existent dirs.
mod utils {
    pub mod hex_colors {
        include!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/src/utils/hex_colors.rs"
        ));
    }
}
mod types {
    pub mod classes {
        include!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/types/classes.rs"));
    }
    pub mod ids {
        include!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/types/ids.rs"));
    }
    pub mod units {
        include!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/types/units.rs"));
    }
}
mod core_types {
    pub use super::types::ids::ImageAddress;
    pub use super::types::units::{PixelUnits, SizeUnits};
}
mod modules {
    pub mod experimant_meta_settings {
        include!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/src/modules/experimant_meta_settings.rs"
        ));
    }
    pub mod classification_settings {
        include!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/src/modules/classification_settings.rs"
        ));
    }
    pub mod roi_settings {
        include!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/src/modules/roi_settings.rs"
        ));
    }
    pub mod images_settings {
        include!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/src/modules/images_settings.rs"
        ));
    }
    pub mod parameter_def {
        include!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/src/modules/parameter_def.rs"
        ));
    }
    pub mod pipeline_command_settings {
        include!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/src/modules/pipeline_command_settings.rs"
        ));
    }
    pub mod pipeline_command {
        include!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/src/modules/pipeline_command.rs"
        ));
    }
    pub mod pipeline_settings {
        include!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/src/modules/pipeline_settings.rs"
        ));
    }
    pub mod plate_settings {
        include!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/src/modules/plate_settings.rs"
        ));
    }
    pub mod project_settings {
        include!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/src/modules/project_settings.rs"
        ));
    }
}
mod settings {
    pub use crate::modules::classification_settings;
    pub use crate::modules::experimant_meta_settings;
    pub use crate::modules::images_settings;
    pub use crate::modules::pipeline_command;
    pub use crate::modules::pipeline_settings;
    pub use crate::modules::plate_settings;
    pub use crate::modules::roi_settings;
}

#[path = "build/pipeline_commands_generator.rs"]
mod pipeline_commands_generator_old;

#[path = "build/json_schema_builder.rs"]
mod schema_generator;

fn main() {
    // Build script sources
    println!("cargo:rerun-if-changed=build/pipeline_commands_generator.rs");
    println!("cargo:rerun-if-changed=build/json_schema_builder.rs");
    // Input: type definitions included by the build script
    println!("cargo:rerun-if-changed=src/utils/hex_colors.rs");
    println!("cargo:rerun-if-changed=src/types/classes.rs");
    println!("cargo:rerun-if-changed=src/types/ids.rs");
    println!("cargo:rerun-if-changed=src/types/units.rs");
    println!("cargo:rerun-if-changed=src/modules/experimant_meta_settings.rs");
    println!("cargo:rerun-if-changed=src/modules/classification_settings.rs");
    println!("cargo:rerun-if-changed=src/modules/roi_settings.rs");
    println!("cargo:rerun-if-changed=src/modules/images_settings.rs");
    println!("cargo:rerun-if-changed=src/modules/parameter_def.rs");
    println!("cargo:rerun-if-changed=src/modules/pipeline_settings.rs");
    println!("cargo:rerun-if-changed=src/modules/plate_settings.rs");
    println!("cargo:rerun-if-changed=src/modules/project_settings.rs");
    // Input: algo structs the generator reads to produce pipeline_command*.rs
    println!("cargo:rerun-if-changed=../core/src/algos");

    if let Err(e) = pipeline_commands_generator_old::generate_mappings() {
        eprintln!("Command Mapping Error: {}", e);
        std::process::exit(1);
    }

    if let Err(e) = schema_generator::generate_project_struct_schema() {
        eprintln!("Schema generator Error: {}", e);
        std::process::exit(1);
    }
}
