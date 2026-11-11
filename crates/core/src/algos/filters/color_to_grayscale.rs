//! # Color to Grayscale
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-11-11
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.
//!
//! Turns a colour (RGB) image into one gray image, so the commands that
//! need one value per pixel (Threshold, Hessian, ...) can follow. A gray
//! image passes through unchanged, so one pipeline works for both.
use crate::algos::{ExecutionScope, GlobalPipelineCache, ImageAlgorithm, PipelineContext};
use crate::image::ImageContainer;
use crate::pipeline::pipeline_cache::sample_channel_pixel;
use evanalyzer_cfg::core_types::{CitationMetadata, InternalErrors};
use macros::CommandsMeta;
use rayon::iter::{IndexedParallelIterator, IntoParallelRefMutIterator, ParallelIterator};
use rayon::slice::ParallelSlice;

/// How the red, green and blue of a pixel become one gray value.
#[derive(CommandsMeta, Debug, Clone, Copy, PartialEq)]
pub enum GrayscaleMethod {
    /// The brightness as the eye sees it: 0.2126 R + 0.7152 G + 0.0722 B
    /// (ITU-R BT.709) - the same value intensities of a colour image are
    /// measured with.
    #[cmdsmeta(display_name = "Brightness")]
    Brightness,
    /// The red channel alone.
    #[cmdsmeta(display_name = "Red")]
    Red,
    /// The green channel alone - often the best contrast in H&E.
    #[cmdsmeta(display_name = "Green")]
    Green,
    /// The blue channel alone.
    #[cmdsmeta(display_name = "Blue")]
    Blue,
    /// (R + G + B) / 3.
    #[cmdsmeta(display_name = "Average")]
    Average,
    /// The brightest of R, G and B per pixel (HSV value, as Color Filter
    /// uses).
    #[cmdsmeta(display_name = "Maximum")]
    Maximum,
    /// 1 − brightness: stained tissue bright on a dark background, as
    /// Threshold expects objects - for brightfield slides.
    #[cmdsmeta(display_name = "Darkness")]
    Darkness,
}

impl GrayscaleMethod {
    /// The gray value of one pixel's red, green and blue (each 0-1).
    #[inline]
    fn gray(self, rgb: &[f32]) -> f32 {
        let (r, g, b) = (rgb[0], rgb[1], rgb[2]);
        match self {
            Self::Brightness => sample_channel_pixel(true, rgb, 0),
            Self::Red => r,
            Self::Green => g,
            Self::Blue => b,
            Self::Average => (r + g + b) / 3.0,
            Self::Maximum => r.max(g).max(b),
            Self::Darkness => 1.0 - sample_channel_pixel(true, rgb, 0),
        }
    }
}

/// Turns a colour (RGB) image into a gray image - e.g. before Threshold,
/// which needs one value per pixel. A gray image passes through unchanged.
///
/// For stains (Hematoxylin, DAB, Eosin) use Color Deconvolution instead; for
/// one colour range, Color Filter.
#[derive(CommandsMeta)]
#[cmdsmeta(category = "Preprocessing", display_name = "Color to Grayscale")]
pub struct ColorToGrayscale {
    /// How the red, green and blue of a pixel become one gray value.
    #[cmdsmeta(default = GrayscaleMethod::Brightness, display_name = "Method", summary = true)]
    pub method: GrayscaleMethod,
}

impl ImageAlgorithm for ColorToGrayscale {
    /// ### Supported Formats
    /// * **Input:** `F32Rgb`, or `F32Gray` (passed through)
    /// * **Output:** `F32Gray`, 0-1
    fn execute(
        &self,
        ctx: &mut PipelineContext,
        _cache: &mut GlobalPipelineCache,
    ) -> Result<(), InternalErrors> {
        match ctx.image.as_ref() {
            ImageContainer::F32Rgb(_) => {}
            ImageContainer::F32Gray(_) => return Ok(()),
            other => {
                return Err(InternalErrors::FormatMismatch {
                    expected: "F32Rgb or F32Gray".into(),
                    found: format!("{other:?}"),
                });
            }
        }
        ctx.prepare_f32_gray_scratch()?;
        let (input, output) = match (
            ctx.image.as_ref(),
            std::sync::Arc::make_mut(&mut ctx.scratch_pad),
        ) {
            (ImageContainer::F32Rgb(input), ImageContainer::F32Gray(output)) => (input, output),
            _ => unreachable!("checked above; the scratch pad was just made gray"),
        };
        let method = self.method;
        output
            .as_slice_mut()
            .par_iter_mut()
            .zip(input.as_slice().par_chunks_exact(3))
            .for_each(|(out, rgb)| *out = method.gray(rgb));

        ctx.swap()
    }

    fn name(&self) -> &'static str {
        "Color to Grayscale"
    }

    /// Measured by `tests::tile_ram` (see the trait method).
    fn ram_per_tile_pixel(&self) -> f64 {
        4.0
    }

    fn cite(&self) -> Vec<&'static CitationMetadata> {
        vec![&CitationMetadata::DANMAYR]
    }

    fn execution_scope(&self) -> ExecutionScope {
        ExecutionScope::Tile
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kornia_image::{Image, ImageSize};

    fn run(method: GrayscaleMethod, pixels: &[[f32; 3]]) -> Vec<f32> {
        let size = ImageSize {
            width: pixels.len(),
            height: 1,
        };
        let data = pixels.iter().flatten().copied().collect();
        let img = Image::<f32, 3>::new(size, data).unwrap();
        let mut ctx = PipelineContext::new_from_image_test_rgb(img).unwrap();
        ColorToGrayscale { method }
            .execute(&mut ctx, &mut GlobalPipelineCache::default())
            .unwrap();
        match ctx.image.as_ref() {
            ImageContainer::F32Gray(out) => out.as_slice().to_vec(),
            other => panic!("expected F32Gray, got {other:?}"),
        }
    }

    const PIXELS: [[f32; 3]; 2] = [[0.8, 0.4, 0.2], [0.1, 0.6, 0.9]];

    fn close(a: &[f32], b: &[f32]) {
        assert_eq!(a.len(), b.len());
        for (x, y) in a.iter().zip(b) {
            assert!((x - y).abs() < 1e-6, "{a:?} != {b:?}");
        }
    }

    #[test]
    fn each_method_gives_its_gray_value_per_pixel() {
        let luma = |p: [f32; 3]| 0.2126 * p[0] + 0.7152 * p[1] + 0.0722 * p[2];
        let [a, b] = PIXELS;
        close(
            &run(GrayscaleMethod::Brightness, &PIXELS),
            &[luma(a), luma(b)],
        );
        close(&run(GrayscaleMethod::Red, &PIXELS), &[0.8, 0.1]);
        close(&run(GrayscaleMethod::Green, &PIXELS), &[0.4, 0.6]);
        close(&run(GrayscaleMethod::Blue, &PIXELS), &[0.2, 0.9]);
        close(
            &run(GrayscaleMethod::Average, &PIXELS),
            &[1.4 / 3.0, 1.6 / 3.0],
        );
        close(&run(GrayscaleMethod::Maximum, &PIXELS), &[0.8, 0.9]);
        close(
            &run(GrayscaleMethod::Darkness, &PIXELS),
            &[1.0 - luma(a), 1.0 - luma(b)],
        );
    }

    /// Brightness is the value intensities of a colour image are measured
    /// with, so a threshold on it and a measurement agree.
    #[test]
    fn brightness_matches_how_a_colour_image_is_measured() {
        let flat: Vec<f32> = PIXELS.iter().flatten().copied().collect();
        let measured: Vec<f32> = (0..2)
            .map(|i| sample_channel_pixel(true, &flat, i))
            .collect();
        close(&run(GrayscaleMethod::Brightness, &PIXELS), &measured);
    }

    #[test]
    fn a_gray_image_passes_through_unchanged() {
        let size = ImageSize {
            width: 3,
            height: 1,
        };
        let img = Image::<f32, 1>::new(size, vec![0.1, 0.5, 0.9]).unwrap();
        let mut ctx = PipelineContext::new_from_image_test(img).unwrap();
        ColorToGrayscale {
            method: GrayscaleMethod::Darkness,
        }
        .execute(&mut ctx, &mut GlobalPipelineCache::default())
        .unwrap();
        match ctx.image.as_ref() {
            ImageContainer::F32Gray(out) => close(out.as_slice(), &[0.1, 0.5, 0.9]),
            other => panic!("expected F32Gray, got {other:?}"),
        }
    }
}
