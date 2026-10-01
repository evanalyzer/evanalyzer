//! Image data front ends work with directly.

/// In-memory pixel data the viewer renders. Re-exported rather than wrapped:
/// it carries no I/O, and a remote transport converts to/from its own wire
/// type (in `evanalyzer_cfg`) inside this crate, so front ends never see the
/// difference.
pub use evanalyzer_core::{ImageContainer, ManagedImage};
