//! # migration
//!
//! Version-to-version migrations for on-disk documents (see the
//! `versioning` module for the overall load/migrate/deserialize mechanism).
//!
//! Each migration step lives in its own `migration_vN_to_vM` file, entirely
//! self-contained: its private helper functions, its single public
//! `migrate_from_vN_to_vM` entry point, and its own tests. `versioning`'s
//! `PROJECT_MIGRATIONS`/`AI_LEARNING_SETTINGS_MIGRATIONS` step lists just
//! reference that one function per step - nothing about a migration's
//! internals leaks into `versioning.rs`. This keeps `versioning.rs` itself
//! free of migration-specific logic, and means an old migration can be
//! dropped later by deleting its file plus the one line wiring it into the
//! step list, without touching anything else.
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-08-23
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.

pub mod migration_v1_to_v2;
pub mod migration_v2_to_v3;
pub mod migration_v3_to_v4;
