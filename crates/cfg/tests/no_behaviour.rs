//! # no_behaviour
//!
//! `cfg` is the project file format and nothing else: its structs and enums
//! and what reading and writing them needs. Behaviour lives where it is
//! used - in core (what an analysis needs) or in the app (what the editor
//! needs), as an extension trait on the `cfg` type there.
//!
//! This test reads the sources and fails on an inherent method or a free
//! function that is not on the list below. Trait impls (`Default`,
//! `Display`, serde, ...) are allowed, and so are whole files that belong to
//! the format: loading, migrations, load problems and the generated
//! commands. Something new for the list must be a conversion or accessor of
//! a value, a constructor, or what serde needs - not a decision or a
//! calculation.
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-11-22
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.

use std::path::{Path, PathBuf};

/// Files that are the format as a whole: allowed to have functions.
const FORMAT_FILES: &[&str] = &[
    "src/load_config.rs",
    "src/load_problems.rs",
    "src/settings/pipeline_command.rs",
    "src/settings/pipeline_command_settings.rs",
];
/// Folders that are the format as a whole.
const FORMAT_FOLDERS: &[&str] = &["src/migration/"];

/// The allowed inherent methods (`Type::method`) and free functions
/// (`fn name`), by kind.
const ALLOWED: &[&str] = &[
    // Conversions of a value to and from its own forms.
    "SegmentationClass::as_u32",
    "SegmentationClass::as_usize",
    "ObjectClass::from_segmentation_class",
    "ObjectClass::to_u32",
    "ObjectClass::to_i32",
    "ObjectClass::to_text",
    "ObjectClass::from_text",
    "ImageMarker::number",
    "ImageMarker::from_number",
    "ImageMarker::name",
    "ImageMarker::from_name",
    "LeaveOut::reason_text",
    "LeaveOutReason::text",
    "ZStackHandling::z_projection",
    "PlateSize::dimensions",
    "PlateSize::from_dimensions",
    "PlateSize::label",
    "ImageAddress::param_parts",
    "ImageAddress::to_param_value",
    "ImageAddress::from_param_value",
    "ModuleParamKey::as_str",
    "ImageDimensionality::depth",
    "ImageDimensionality::ndim",
    "PixelUnits::to_relative",
    "SizeUnits::to_pixel",
    "SizeUnitsRel::to_pixel",
    "fn max_gray_value",
    "BreakpointSettings::run_step",
    // Accessors.
    "Class::global",
    "Class::local",
    "ClassificationSettings::classes",
    "ClassificationSettings::classes_mut",
    "PipelineStepSettings::command",
    "PipelineStepSettings::command_mut",
    "ObjectMetric::channel",
    "ObjectMetric::uses_ring",
    "ImageReview::is_empty",
    // Constructors and builders.
    "ClassificationSettings::new",
    "ClassificationSettings::new_from_existing",
    "ObjectId::next",
    "MeasurementInfo::range",
    "MeasurementInfo::naming_objects",
    "MeasurementInfo::profile",
    "ParameterDef::in_section",
    // An error with its context.
    "InternalErrors::from_kornia",
    "InternalErrors::in_step",
    "InternalErrors::in_image",
    "InternalErrors::step",
    "InternalErrors::message",
    // Rules of the file format: a measurement id's syntax, paths that stay
    // below their root.
    "fn is_valid_measurement_id",
    "fn is_contained_rel_path",
    "fn is_file_name_part",
    "fn has_drive_prefix",
    // What serde needs (`default = "..."`, `skip_serializing_if = "..."`).
    "fn default_ring_width",
    "fn is_zero",
    "fn is_one_plane",
];

fn rust_files(dir: &Path, out: &mut Vec<PathBuf>) {
    for entry in std::fs::read_dir(dir).expect("cfg sources") {
        let path = entry.expect("a directory entry").path();
        if path.is_dir() {
            rust_files(&path, out);
        } else if path.extension().is_some_and(|ext| ext == "rs") {
            out.push(path);
        }
    }
}

/// `Type::method` of the inherent `impl` blocks and `fn name` of the free
/// functions in `source`, before its tests.
fn functions(source: &str) -> Vec<String> {
    let mut found = Vec::new();
    let mut inherent: Option<String> = None;
    for line in source.lines() {
        if line.starts_with("#[cfg(test)]") {
            break;
        }
        if let Some(rest) = line.strip_prefix("impl") {
            let header = rest.trim_start_matches(|c: char| c != ' ').trim();
            inherent = (!header.contains(" for ")).then(|| {
                header
                    .split(|c: char| !(c.is_alphanumeric() || c == '_'))
                    .next()
                    .unwrap_or("")
                    .to_string()
            });
            continue;
        }
        if line == "}" {
            inherent = None;
            continue;
        }
        let name_after = |text: &str| -> Option<String> {
            let text = text
                .trim_start_matches("pub(crate) ")
                .trim_start_matches("pub(super) ")
                .trim_start_matches("pub ")
                .trim_start_matches("const ");
            text.strip_prefix("fn ").map(|rest| {
                rest.split(|c: char| !(c.is_alphanumeric() || c == '_'))
                    .next()
                    .unwrap_or("")
                    .to_string()
            })
        };
        if let Some(method) = line.strip_prefix("    ").and_then(name_after)
            && let Some(ty) = &inherent
        {
            found.push(format!("{ty}::{method}"));
        } else if !line.starts_with(' ')
            && let Some(name) = name_after(line)
        {
            found.push(format!("fn {name}"));
        }
    }
    found
}

#[test]
fn cfg_has_no_behaviour_beyond_its_format() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let mut files = Vec::new();
    rust_files(&root.join("src"), &mut files);
    let mut unexpected = Vec::new();
    let mut seen = Vec::new();
    for file in files {
        let rel = file
            .strip_prefix(root)
            .expect("below the crate")
            .to_string_lossy()
            .replace('\\', "/");
        if FORMAT_FILES.contains(&rel.as_str())
            || FORMAT_FOLDERS.iter().any(|folder| rel.starts_with(folder))
        {
            continue;
        }
        let source = std::fs::read_to_string(&file).expect("a source file");
        for function in functions(&source) {
            if ALLOWED.contains(&function.as_str()) {
                seen.push(function);
            } else {
                unexpected.push(format!("{rel}: {function}"));
            }
        }
    }
    assert!(
        unexpected.is_empty(),
        "behaviour in cfg - move it to core or the app (an extension trait \
         on the cfg type), or, if it is a conversion, accessor, constructor \
         or serde helper, add it to ALLOWED:\n{}",
        unexpected.join("\n")
    );
    let stale: Vec<&&str> = ALLOWED
        .iter()
        .filter(|allowed| !seen.iter().any(|s| s == **allowed))
        .collect();
    assert!(
        stale.is_empty(),
        "on ALLOWED but no longer in cfg - take them off the list: {stale:?}"
    );
}
