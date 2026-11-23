//! # user
//!
//! The user on this computer: their app settings (preferences, recent
//! projects and servers, the user folder) and the crash log.
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-11-22
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.

pub mod crash_log;
pub mod folder;
pub mod recent;
pub mod settings;

pub use folder::USER_FOLDER_NAME;
pub use recent::{RecentProject, RecentServer};
pub use settings::{
    AppSettings, ColorScheme, InterfaceSize, NavigatorMode, PanelSettings, PerformanceSettings,
    PredictionOverlayPrefs, ResultsSplitSettings, TrailMode, UserInformation,
    ViewportMemorySettings,
};
