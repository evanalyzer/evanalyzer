//! # config
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-11-25
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.
//!
//! **The `.evamodel`'s classifier**: which classifier a file holds (the
//! variant names). Each classifier's own part is in its `config.rs`
//! (`rtree`, `knn`, `mlp`).
//!
//! Any change here (a field, a variant, a name, a serde attribute) changes
//! what saved `.evamodel` files contain: core's
//! `CURRENT_SAVED_CLASSIFIER_VERSION` goes up with it (older files are
//! refused - no backwards compatibility). The test
//! `classifier::tests::the_file_layout_is_pinned` guards it.

use super::{Knn, Mlp, RandomForest};
use serde::{Deserialize, Serialize};

/// A trained classifier as stored in a `.evamodel` file (JSON). A smartcore
/// upgrade changing its serde form changes the files too.
#[derive(Debug, Serialize, Deserialize)]
pub enum TrainedClassifier {
    RandomForest(RandomForest),
    Knn(Knn),
    Mlp(Mlp),
}
