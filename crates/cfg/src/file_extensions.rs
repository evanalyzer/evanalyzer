// # file_extensions
//
// **Author:** Joachim Danmayr
// **Date:** 2026-11-26
//
// ## License
// Copyright 2026 Joachim Danmayr.
// Licensed under the **PolyForm Noncommercial License 1.0.0**.
//
// The extensions (no leading dot) of EVAnalyzer's own files. Also
// include!()-d into the build script, so a command's
// `#[cmdsmeta(file_extensions = NN_MODEL_EXTENSION)]` names one of these
// instead of repeating the string.

/// A project.
pub const PROJECT_FILE_EXTENSIONS: &str = "evaproj";
/// A project template.
pub const PROJECT_FILE_TEMPLATE_EXTENSIONS: &str = "evatemplate";
/// A module (a group of steps).
pub const MODULE_TEMPLATE_EXTENSION: &str = "evamodule";
/// A run's results database.
pub const RESULTS_FILE_EXTENSION: &str = "evadb";
/// A classifier trained in the AI training dialog (KNN, random forest, MLP).
pub const EVANALYZER_TRAINED_AI_MODELS: &str = "evamodel";
/// A neural network model (StarDist, U-Net, Cellpose-SAM): a safetensors
/// file with EVAnalyzer's metadata in `__metadata__`.
pub const NN_MODEL_EXTENSION: &str = "evanet";
