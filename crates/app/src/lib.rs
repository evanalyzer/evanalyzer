pub mod ai_learning;
pub mod bioimageio;
pub mod crash_log;
pub mod export;
pub mod extensions;
pub mod frontend;
mod project_owner;
mod results;
pub mod settings;
pub mod templates;

pub use frontend::Frontend;
pub use project_owner::{
    AppHandle, ProjectOwner, ProjectTmpSettings, ProjectWithRuntime, ReaderPool,
};

pub mod prelude {
    pub use super::Frontend;
    pub use super::extensions::*;
}

pub mod result {
    pub use crate::results::*;
}

pub mod exporter {
    pub use crate::export::cite_project::cite_project;
}
