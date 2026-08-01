pub mod object;
pub mod pixel;

use crate::ai_learning::model::mlp::predict_mlp;
use crate::ai_learning::{model::Classifier, utils::to_dense_matrix};
use evanalyzer_cfg::core_types::InternalErrors;
