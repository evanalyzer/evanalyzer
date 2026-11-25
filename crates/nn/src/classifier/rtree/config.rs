//! # config
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-11-25
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.
//!
//! **The random forest's part of the `.evamodel` format**: smartcore's trees
//! in smartcore's own serde form - a smartcore upgrade that changes its form
//! changes the files too.
//!
//! Any change here (a field, a variant, a name, a serde attribute) changes
//! what saved `.evamodel` files contain: core's
//! `CURRENT_SAVED_CLASSIFIER_VERSION` goes up with it (older files are
//! refused - no backwards compatibility). The test
//! `classifier::tests::the_file_layout_is_pinned` guards it.

use serde::{Deserialize, Serialize};
use smartcore::linalg::basic::matrix::DenseMatrix;
use smartcore::tree::decision_tree_classifier::DecisionTreeClassifier;

pub(super) type TreeModel = DecisionTreeClassifier<f32, usize, DenseMatrix<f32>, Vec<usize>>;

/// A fitted random forest: smartcore's trees, which it keeps private -
/// read out once after fitting (`trees`, `classes`, smartcore's field
/// names). The samples smartcore can keep per tree for out-of-bag estimates
/// are never kept: nothing reads them, and with them a model trained on
/// 200 000 pixels was 57 MB on disk and hundreds of MB in RAM.
#[derive(Debug, Serialize, Deserialize)]
pub struct RandomForest {
    pub(super) trees: Vec<TreeModel>,
    pub(super) classes: Option<Vec<usize>>,
}
