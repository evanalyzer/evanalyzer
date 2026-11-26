//! # post
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-11-26
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.
//!
//! The U-Net's post-processing: its probabilities thresholded into a mask.

use crate::errors::NnError;

/// `foreground` where the probability reaches `probability_threshold` and -
/// for boundary-aware models - the boundary probability stays below
/// `boundary_threshold`, else 0. Carving the boundaries out as thin gaps is
/// what lets a boundary-aware model separate touching nuclei, which a plain
/// mask merges.
pub(super) fn threshold(
    probabilities: &[f32],
    boundary: Option<&[f32]>,
    probability_threshold: f32,
    boundary_threshold: f32,
    foreground: u32,
    out: &mut [u32],
) -> Result<(), NnError> {
    if out.len() != probabilities.len() || boundary.is_some_and(|b| b.len() != out.len()) {
        return Err(NnError::InvalidInput(format!(
            "U-Net: a mask of {} pixels for {} probabilities",
            out.len(),
            probabilities.len()
        )));
    }
    for (i, (value, &p)) in out.iter_mut().zip(probabilities).enumerate() {
        let open = boundary.is_none_or(|b| b[i] < boundary_threshold);
        *value = if p >= probability_threshold && open {
            foreground
        } else {
            0
        };
    }
    Ok(())
}
