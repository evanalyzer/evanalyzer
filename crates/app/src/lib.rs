//! # evanalyzer_app
//!
//! One folder per feature.
//!
//! - Features that do the work, each with what travels (types), the trait
//!   a backend offers and the code that runs here (`local`): [`results`],
//!   [`images`], [`files`], [`jobs`], [`training`], [`exports`].
//! - Features of the open project, which reach the work only through the
//!   `Backend` trait: [`project`], [`modules`], [`templates`],
//!   [`annotations`], [`ai`], [`user`].
//! - [`backend`]: the `Backend` trait and the local backend.
//! - [`remote`]: everything on the network - the remote backend, the
//!   worker and the wire between them. No feature knows it exists.
//! - [`charts`]: drawing only, no IO. [`memory`], [`number_format`]: small
//!   helpers.
//!
//! The public paths are these modules: `evanalyzer_app::results::...`,
//! `evanalyzer_app::backend::local::LocalBackend`, ...
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-06-05
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.

pub mod ai;
pub mod annotations;
pub mod backend;
pub mod charts;
pub mod exports;
pub mod files;
pub mod images;
pub mod jobs;
pub mod memory;
pub mod modules;
pub mod number_format;
pub mod project;
pub mod remote;
pub mod results;
pub mod templates;
pub mod training;
pub mod user;
