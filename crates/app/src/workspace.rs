//! The UI side: the open project and everything the user does with it -
//! editing, undo, templates, settings. Reaches files and compute only
//! through `crate::api::Backend`, so it works the same against a server.

pub mod ai_learning;
pub mod bioimageio;
pub mod class_ids;
pub mod component_checks;
pub mod component_editing;
pub mod component_insights;
pub mod component_library;
pub mod crash_log;
pub mod export;
pub mod extensions;
pub mod frontend;
pub mod images;
pub mod plate_layout;
pub(crate) mod project_owner;
pub mod settings;
pub mod templates;

pub use frontend::Frontend;
pub use project_owner::{
    AppHandle, FocusChannels, PipelineFocus, ProjectOwner, ProjectTmpSettings, ProjectWithRuntime,
};
