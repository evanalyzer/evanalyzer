//! # ai_segmentation
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-06-26
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.

pub mod cellpose;
pub(crate) mod classifier_cache;
pub mod input_normalization;
pub mod model_file;
pub mod piece_size;
pub mod pixel_classifier;
pub mod spotnet;
pub mod stardist;
pub mod unet;

pub use input_normalization::InputNormalization;
pub use piece_size::PieceSize;

/// The segmentation map of an instance map: `class` where an object is,
/// background elsewhere.
pub(crate) fn fill_class(
    instances: &[u32],
    segmentation: &mut [u32],
    class: evanalyzer_cfg::core_types::SegmentationClass,
) {
    let (class, background) = (
        class.as_u32(),
        evanalyzer_cfg::core_types::SegmentationClass::BACKGROUND.as_u32(),
    );
    for (seg, &instance) in segmentation.iter_mut().zip(instances) {
        *seg = if instance == 0 { background } else { class };
    }
}

/// The RAM a model's weights take once loaded: about its file's size
/// (0 when it can't be read - no model chosen yet).
pub(crate) fn model_bytes(path: &std::path::Path) -> u64 {
    std::fs::metadata(path).map_or(0, |m| m.len())
}
