//! # evanalyzer_server
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-10-10
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.

mod api;
mod config;
mod server;
mod session_management;
mod tls;
mod user_management;

pub use config::{ServerConfig, UserSource, default_data_dir};
pub use server::serve;

pub use user_management::file_users::FileUsers;
pub use user_management::linux_users::LinuxUsers;
pub use user_management::password::hash_password;
pub use user_management::single_user::SingleUser;
