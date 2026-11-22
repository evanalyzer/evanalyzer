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
pub(crate) mod frontend;
pub mod image_entry;
pub mod images;
pub mod objects;
pub(crate) mod owner;
pub(crate) mod plate_layout;
pub mod project_ext;
pub mod utils;

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
pub use annotations::{AnnotationScope, AnnotationsExt};
pub use class_ids::{ClassIdChange, change_class_id, free_class_ids, next_class_id};
pub use image_entry::ImageEntryExt;
pub use images::ImageListExt;
pub use objects::{ObjectExt, ShownPlane};
pub use project_ext::{
    ConfigValueSource, ProjectExt, ResultsImageFinder, SaveProjectActions,
    SelectNewProjectRootAction, collect_images_at_root, load_project, run_image_root,
};
