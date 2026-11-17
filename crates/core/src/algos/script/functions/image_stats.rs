//! # Image statistics
//!
//! `mean()`, `min()` and `max()` on a script's `Image`, over all pixels and
//! colour values:
//!
//! ```rhai
//! let raw = channel(0);
//! print(`mean ${raw.mean()}, range ${raw.min()} - ${raw.max()}`);
//! ```
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-10-10
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.

use crate::algos::script::script_bridge::ScriptImage;
use crate::image::ImageContainer;
use rhai::Engine;

/// Registers `mean`, `min` and `max` on `Image`.
pub(crate) fn register(engine: &mut Engine) {
    engine
        .register_fn("mean", ScriptImage::mean)
        .register_fn("min", ScriptImage::min)
        .register_fn("max", ScriptImage::max);
}

impl ScriptImage {
    /// Mean over all pixels (and color channels).
    pub(crate) fn mean(&mut self) -> f64 {
        let (sum, count) = self.fold(0.0, |acc, v| acc + v);
        if count == 0 { 0.0 } else { sum / count as f64 }
    }

    pub(crate) fn min(&mut self) -> f64 {
        self.fold(f64::INFINITY, f64::min).0
    }

    pub(crate) fn max(&mut self) -> f64 {
        self.fold(f64::NEG_INFINITY, f64::max).0
    }

    fn fold(&self, init: f64, f: impl Fn(f64, f64) -> f64) -> (f64, usize) {
        match self.0.as_ref() {
            ImageContainer::F32Gray(img) => fold_slice(img.data.as_slice(), init, f),
            ImageContainer::F32Rgb(img) => fold_slice(img.data.as_slice(), init, f),
            ImageContainer::U32(img) => fold_slice(img.data.as_slice(), init, f),
        }
    }
}

fn fold_slice<T: Copy + Into<f64>>(
    data: &[T],
    init: f64,
    f: impl Fn(f64, f64) -> f64,
) -> (f64, usize) {
    (
        data.iter().fold(init, |acc, v| f(acc, (*v).into())),
        data.len(),
    )
}
