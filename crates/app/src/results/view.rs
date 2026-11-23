//! # view
//!
//! How the results are shown: the views' settings (saved in the project)
//! and the colour scales.
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-11-22
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.

mod colors;
pub mod view_settings;

pub(crate) use colors::value_to_color;
pub use colors::{COLOR_SCALE_GRADIENT_STOPS, color_scale_gradient};
