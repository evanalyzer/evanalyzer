//! # Script functions
//!
//! The functions a script calls that are not pipeline commands (those run
//! through `run(...)`, see `script_bridge`): one file each, every file
//! registers its own on the engine.
//!
//! - [`image_stats`]: `mean()`, `min()`, `max()` of an image
//! - [`dot`]: `dot(a, b [, #{ margin, mask }])`, the mean of the pixel-wise
//!   product of two images
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-11-17
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.

pub(crate) mod dot;
pub(crate) mod image_stats;

use rhai::Engine;

/// Registers every script function on `engine`.
pub(crate) fn register(engine: &mut Engine) {
    image_stats::register(engine);
    dot::register(engine);
}
