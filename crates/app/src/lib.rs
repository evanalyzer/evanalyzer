//! - [`api`]: the contract between front ends and backends.
//! - [`backend`]: where the work runs - [`backend::local`] (this process,
//!   the only module calling `evanalyzer_core`) or [`backend::remote`].
//! - [`workspace`]: the UI side - the open project, editing, undo.

pub mod api;
pub mod backend;
pub mod workspace;

pub use workspace::{AppHandle, Frontend, ProjectOwner, ProjectTmpSettings, ProjectWithRuntime};

pub mod prelude {
    pub use super::Frontend;
    pub use super::workspace::extensions::*;
}
