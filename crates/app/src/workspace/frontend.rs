//! # frontend
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-06-05
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.

// app/src/frontend.rs - app defines the trait, knows nothing about gui

use crate::workspace::ProjectOwner;

pub trait Frontend: Send + Sync {
    fn start(self: Box<Self>, owner: ProjectOwner);
}
