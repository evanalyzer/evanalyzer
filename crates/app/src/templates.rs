//! # templates
//!
//! Project and module templates: loading them (from the app's and the
//! user's folders) and checking them.
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-11-22
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.

pub(crate) mod checks;
pub(crate) mod load;

pub use crate::backend::TemplateFolders;
pub use load::{
    load_module_template_from_file, load_module_templates, load_project_template_from_file,
    load_project_templates,
};
