//! # Tests across the library
//!
//! Test-only code that isn't one module's own tests: the allocator that
//! counts RAM, and the tests that use it. Compiled only for the library's
//! tests, never into the app.
//!
//! A `mod.rs` on purpose (not `tests.rs` next to the folder, as elsewhere):
//! test-only folders keep their module file inside, out of the source tree's
//! top level.
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-10-30
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.

pub(crate) mod counting_allocator;
mod whole_image_ram;
