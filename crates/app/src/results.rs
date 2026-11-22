//! # results
//!
//! An analysis' results: what is asked for and answered per view
//! ([`types`]), the [`ResultsSource`] a backend offers, reading them in this
//! process ([`local`]: the queries, charts and exports on the results
//! file) and how they are shown ([`view`]).
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-11-22
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.

mod column;
pub(crate) mod local;
mod source;
mod types;
pub mod view;

pub use column::{
    ChildColumn, ChildValue, ColocColumn, ColocMatch, Column, MeasurementColumn, ParentColumn,
    ParentValue,
};
pub use source::ResultsSource;
pub use types::*;
pub(crate) use view::value_to_color;
pub use view::{COLOR_SCALE_GRADIENT_STOPS, color_scale_gradient, view_settings};

pub use local::ResultsGenerator as LocalResultsGenerator;
pub use local::{colorize, colorize_thumbnail};

// The public paths these had before (`evanalyzer_app::results::...`); they
// move into `results/` with the project features.
pub use crate::workspace::analysis_folders::{AnalysisFolder, analysis_folder, delete_analysis};
/// A track's MSD over lag as Measure Track Motion fits it (the details
/// panel's plot: the same points the step fitted).
pub use evanalyzer_core::{MotionPoint, msd_curve};
