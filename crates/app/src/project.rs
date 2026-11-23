//! # project
//!
//! The open project: who owns it ([`ProjectOwner`]), loading and saving,
//! its images, classes, objects and annotations, plate layout and citation.
//! Reaches files and compute only through `crate::backend::Backend`, so it
//! works the same against a server.
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-11-22
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.

pub mod annotations;
pub(crate) mod cite;
pub(crate) mod class_ids;
pub mod classes;
pub(crate) mod current_image;
pub(crate) mod display;
pub(crate) mod frontend;
pub(crate) mod image_folder;
pub(crate) mod load_save;
pub mod objects;
pub(crate) mod owner;
pub(crate) mod pipelines;
pub(crate) mod plate_layout;
pub(crate) mod results_images;
#[cfg(test)]
mod tests;
pub mod utils;

pub use cite::{cite_project, command_citations};
pub use frontend::Frontend;
pub use owner::{
    AppHandle, FocusChannels, PipelineFocus, ProjectOwner, ProjectTmpSettings, ProjectWithRuntime,
};

pub use crate::annotations::geojson::{
    ClassTarget, FileClass, ImportReport, PendingImport, PipelineObjects, apply_import,
    default_class_target, export_image_geojson, image_annotation_classes, measure_pending,
    read_image_geojson,
};
pub use crate::annotations::measure::measure_on_source;
pub use crate::images::ImageEntryExt;
pub use crate::images::{GlobalImageSettingsExt, ImageListExt};
pub use annotations::AnnotationScope;
pub use class_ids::{ClassIdChange, change_class_id, free_class_ids, next_class_id};
pub use classes::ClassificationExt;
pub use current_image::{ConfigValueSource, DEFAULT_FRAME_INTERVAL_S};
pub use image_folder::{
    ImageSettingsExt, ProjectAction, SelectNewProjectRootAction, collect_images_at_root,
};
pub use load_save::{SaveProjectActions, load_project};
pub use objects::{ObjectExt, ShownPlane};
pub use plate_layout::{PlateLayout, PlateSettingsExt, WellImages, plate_layout, texts_in_group};
pub use results_images::{ResultsImageFinder, find_results_image, run_image_root};
