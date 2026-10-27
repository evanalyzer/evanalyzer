//! # Color Deconvolution
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-10-26
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.
//!
//! Separates a brightfield RGB image into the amounts of its stains
//! (Hematoxylin, DAB, Eosin) - Ruifrok & Johnston's color deconvolution
//! (Anal Quant Cytol Histol 23(4), 2001), written from the paper.
//!
//! Light passing a stain is absorbed by Beer-Lambert, so in optical density
//! (OD = -log10(I / white)) the stains add up linearly: each pixel's OD in
//! R, G, B is the sum of every stain's amount times its OD color (its stain
//! vector). With three stain vectors as the rows of a matrix M, the amounts
//! are OD · M⁻¹. Two-stain presets get a third vector perpendicular to both
//! (the residual: what neither stain explains).
//!
//! The step keeps one stain, as a gray image: 0 = no stain, 1 = OD 2.55 or
//! more ([`MAX_OPTICAL_DENSITY`]) - so on an 8-bit image's 0-255 scale,
//! 100 means OD 1.0.

use crate::algos::{ExecutionScope, GlobalPipelineCache, ImageAlgorithm, PipelineContext};
use crate::image::ImageContainer;
use evanalyzer_cfg::core_types::{CitationMetadata, InternalErrors};
use macros::CommandsMeta;
use rayon::iter::{IndexedParallelIterator, IntoParallelRefMutIterator, ParallelIterator};
use rayon::slice::ParallelSlice;

/// The OD shown as 1.0 (255 in 8 bit): OD = value × 2.55, or the 8-bit
/// value / 100. Real stains rarely go past OD 2.5 (nearly black).
pub const MAX_OPTICAL_DENSITY: f32 = 2.55;

/// The stains' colors: Ruifrok & Johnston's measured stain vectors (OD of
/// red, green, blue), or your own.
#[derive(CommandsMeta, Debug, Clone, Copy, PartialEq)]
pub enum StainVectors {
    /// Hematoxylin and DAB (immunohistochemistry).
    #[cmdsmeta(display_name = "H-DAB")]
    HDab,
    /// Hematoxylin and Eosin.
    #[cmdsmeta(display_name = "H&E")]
    HE,
    /// Hematoxylin, Eosin and DAB.
    #[cmdsmeta(display_name = "H-E-DAB")]
    HEDab,
    /// Your own stain vectors, e.g. measured on a single-stained slide: the
    /// optical density of each stain in red, green and blue. Stain 3 all 0:
    /// the residual (perpendicular to stains 1 and 2).
    #[cmdsmeta(display_name = "Custom")]
    Custom {
        #[cmdsmeta(default = 0.650, min = 0.0, max = 1.0, step = 0.001)]
        stain_1_red: f32,
        #[cmdsmeta(default = 0.704, min = 0.0, max = 1.0, step = 0.001)]
        stain_1_green: f32,
        #[cmdsmeta(default = 0.286, min = 0.0, max = 1.0, step = 0.001)]
        stain_1_blue: f32,
        #[cmdsmeta(default = 0.268, min = 0.0, max = 1.0, step = 0.001)]
        stain_2_red: f32,
        #[cmdsmeta(default = 0.570, min = 0.0, max = 1.0, step = 0.001)]
        stain_2_green: f32,
        #[cmdsmeta(default = 0.776, min = 0.0, max = 1.0, step = 0.001)]
        stain_2_blue: f32,
        #[cmdsmeta(default = 0.0, min = 0.0, max = 1.0, step = 0.001)]
        stain_3_red: f32,
        #[cmdsmeta(default = 0.0, min = 0.0, max = 1.0, step = 0.001)]
        stain_3_green: f32,
        #[cmdsmeta(default = 0.0, min = 0.0, max = 1.0, step = 0.001)]
        stain_3_blue: f32,
    },
}

/// Which stain the step keeps.
#[derive(CommandsMeta, Debug, Clone, Copy, PartialEq)]
pub enum StainOutput {
    /// Stain 1: Hematoxylin in every preset (nuclei).
    #[cmdsmeta(display_name = "Stain 1 (Hematoxylin)")]
    Stain1,
    /// Stain 2: DAB in H-DAB, Eosin in H&E and H-E-DAB.
    #[cmdsmeta(display_name = "Stain 2 (DAB / Eosin)")]
    Stain2,
    /// Stain 3: DAB in H-E-DAB, else the residual - what neither stain
    /// explains (high where the stain vectors don't fit the slide).
    #[cmdsmeta(display_name = "Stain 3 (DAB in H-E-DAB / residual)")]
    Stain3,
}

/// Separates a brightfield RGB image into its stains and keeps one as a
/// gray image: how much of the stain each pixel has (0 = none, 255 in 8 bit
/// = OD 2.55; the 8-bit value / 100 is the optical density).
///
/// For IHC: Stain 2 of H-DAB is the DAB (brown) signal - threshold it, or
/// measure the DAB per nucleus found on Stain 1.
#[derive(CommandsMeta)]
#[cmdsmeta(category = "Preprocessing", display_name = "Color Deconvolution")]
pub struct ColorDeconvolution {
    /// The stains on the slide.
    #[cmdsmeta(default = StainVectors::HDab, display_name = "Stains", summary = true)]
    pub stains: StainVectors,

    /// The stain the step keeps.
    #[cmdsmeta(default = StainOutput::Stain1, display_name = "Keep", summary = true)]
    pub output: StainOutput,

    /// The brightness of empty glass (no tissue), 0-255 whatever the image's
    /// bit depth - lower it when the scanner's background isn't pure white.
    #[cmdsmeta(
        default = 255.0,
        min = 1.0,
        max = 255.0,
        step = 1.0,
        display_name = "Background (white)",
        visibility = Advanced
    )]
    pub background: f32,
}

/// Ruifrok & Johnston (2001), table 1. Unit-length rows are made in
/// [`unmixing_matrix`].
const HEMATOXYLIN: [f32; 3] = [0.650, 0.704, 0.286];
const HEMATOXYLIN_HE: [f32; 3] = [0.644211, 0.716556, 0.266844];
const EOSIN_HE: [f32; 3] = [0.092789, 0.954111, 0.283111];
const EOSIN: [f32; 3] = [0.072, 0.990, 0.105];
const DAB: [f32; 3] = [0.268, 0.570, 0.776];

impl StainVectors {
    /// Stain 1, 2 and 3 (`None`: the residual).
    fn vectors(&self) -> ([f32; 3], [f32; 3], Option<[f32; 3]>) {
        match *self {
            StainVectors::HDab => (HEMATOXYLIN, DAB, None),
            StainVectors::HE => (HEMATOXYLIN_HE, EOSIN_HE, None),
            StainVectors::HEDab => (HEMATOXYLIN, EOSIN, Some(DAB)),
            StainVectors::Custom {
                stain_1_red,
                stain_1_green,
                stain_1_blue,
                stain_2_red,
                stain_2_green,
                stain_2_blue,
                stain_3_red,
                stain_3_green,
                stain_3_blue,
            } => {
                let third = [stain_3_red, stain_3_green, stain_3_blue];
                (
                    [stain_1_red, stain_1_green, stain_1_blue],
                    [stain_2_red, stain_2_green, stain_2_blue],
                    (third != [0.0; 3]).then_some(third),
                )
            }
        }
    }
}

fn normalized(v: [f32; 3]) -> Option<[f32; 3]> {
    let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    (len > 1e-6).then(|| [v[0] / len, v[1] / len, v[2] / len])
}

fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

/// M⁻¹ for the stain vectors as M's rows: amounts = OD · M⁻¹. Errors when a
/// vector is all 0 or the stains can't be told apart (same color).
fn unmixing_matrix(stains: &StainVectors) -> Result<[[f32; 3]; 3], InternalErrors> {
    let invalid = |why: &str| InternalErrors::Generic(format!("Color deconvolution: {why}"));
    let (first, second, third) = stains.vectors();
    let first = normalized(first).ok_or_else(|| invalid("stain 1 is all 0"))?;
    let second = normalized(second).ok_or_else(|| invalid("stain 2 is all 0"))?;
    let third = match third {
        Some(v) => normalized(v).ok_or_else(|| invalid("stain 3 is all 0"))?,
        None => normalized(cross(first, second))
            .ok_or_else(|| invalid("stains 1 and 2 have the same color"))?,
    };
    let m = [first, second, third];

    // Inverse by the adjugate (cofactors, transposed) / determinant.
    let det = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
    if det.abs() < 1e-4 {
        return Err(invalid("the stains can't be told apart (same color)"));
    }
    let mut inv = [[0.0f32; 3]; 3];
    for (i, row) in inv.iter_mut().enumerate() {
        for (j, value) in row.iter_mut().enumerate() {
            let (r0, r1) = ((j + 1) % 3, (j + 2) % 3);
            let (c0, c1) = ((i + 1) % 3, (i + 2) % 3);
            *value = (m[r0][c0] * m[r1][c1] - m[r0][c1] * m[r1][c0]) / det;
        }
    }
    Ok(inv)
}

impl ImageAlgorithm for ColorDeconvolution {
    /// ### Supported Formats
    /// * **Input:** `F32Rgb`
    /// * **Output:** `F32Gray` - the kept stain's amount / [`MAX_OPTICAL_DENSITY`], 0-1.
    fn execute(
        &self,
        ctx: &mut PipelineContext,
        _cache: &mut GlobalPipelineCache,
    ) -> Result<(), InternalErrors> {
        if !matches!(ctx.image.as_ref(), ImageContainer::F32Rgb(_)) {
            return Err(InternalErrors::FormatMismatch {
                expected: "F32Rgb (a brightfield RGB image)".into(),
                found: format!("{:?}", ctx.image),
            });
        }
        let inv = unmixing_matrix(&self.stains)?;
        let column = match self.output {
            StainOutput::Stain1 => 0,
            StainOutput::Stain2 => 1,
            StainOutput::Stain3 => 2,
        };
        let weights = [inv[0][column], inv[1][column], inv[2][column]];
        let white = (self.background / 255.0).clamp(1.0 / 255.0, 1.0);
        // A pixel darker than 1/255 of the background counts as 1/255 (OD
        // 2.41) - 0 would be infinite.
        let darkest = white / 255.0;
        let scale = 1.0 / MAX_OPTICAL_DENSITY;

        ctx.prepare_f32_gray_scratch()?;
        let (input, output) = match (
            ctx.image.as_ref(),
            std::sync::Arc::make_mut(&mut ctx.scratch_pad),
        ) {
            (ImageContainer::F32Rgb(input), ImageContainer::F32Gray(output)) => (input, output),
            _ => unreachable!("checked above; the scratch pad was just made gray"),
        };
        output
            .as_slice_mut()
            .par_iter_mut()
            .zip(input.as_slice().par_chunks_exact(3))
            .for_each(|(out, rgb)| {
                let mut amount = 0.0;
                for (k, &value) in rgb.iter().enumerate() {
                    let od = -(value.clamp(darkest, white) / white).log10();
                    amount += od * weights[k];
                }
                *out = (amount * scale).clamp(0.0, 1.0);
            });

        ctx.swap()
    }

    fn name(&self) -> &'static str {
        "Color Deconvolution"
    }

    fn cite(&self) -> Vec<&'static CitationMetadata> {
        vec![&CitationMetadata {
            cite_key: "ruifrok2001quantification",
            title: "Quantification of histochemical staining by color deconvolution",
            authors: &["Arnout C. Ruifrok", "Dennis A. Johnston"],
            year: 2001,
            container: Some("Analytical and Quantitative Cytology and Histology"),
            doi: None,
            url: Some("https://pubmed.ncbi.nlm.nih.gov/11531144/"),
            pages: Some("291-299"),
        }]
    }

    fn execution_scope(&self) -> ExecutionScope {
        ExecutionScope::Tile
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kornia_image::{Image, ImageSize};

    /// The RGB a pixel gets with `amounts` of the stains on white glass.
    fn stained(amounts: [f32; 3], vectors: [[f32; 3]; 3]) -> [f32; 3] {
        let mut rgb = [0.0; 3];
        for (k, value) in rgb.iter_mut().enumerate() {
            let od: f32 = (0..3).map(|s| amounts[s] * vectors[s][k]).sum();
            *value = 10f32.powf(-od);
        }
        rgb
    }

    fn run(cmd: &ColorDeconvolution, pixels: &[[f32; 3]]) -> Vec<f32> {
        let size = ImageSize {
            width: pixels.len(),
            height: 1,
        };
        let data = pixels.iter().flatten().copied().collect();
        let img = Image::<f32, 3>::new(size, data).unwrap();
        let mut ctx = PipelineContext::new_from_image_test_rgb(img).unwrap();
        cmd.execute(&mut ctx, &mut GlobalPipelineCache::default())
            .unwrap();
        match ctx.image.as_ref() {
            ImageContainer::F32Gray(out) => out.as_slice().to_vec(),
            other => panic!("expected F32Gray, got {other:?}"),
        }
    }

    fn cmd(stains: StainVectors, output: StainOutput) -> ColorDeconvolution {
        ColorDeconvolution {
            stains,
            output,
            background: 255.0,
        }
    }

    fn unit(v: [f32; 3]) -> [f32; 3] {
        normalized(v).unwrap()
    }

    #[test]
    fn white_glass_has_no_stain() {
        for output in [
            StainOutput::Stain1,
            StainOutput::Stain2,
            StainOutput::Stain3,
        ] {
            let out = run(&cmd(StainVectors::HDab, output), &[[1.0, 1.0, 1.0]]);
            assert!(out[0].abs() < 1e-6, "{output:?}: {}", out[0]);
        }
    }

    #[test]
    fn a_mixed_pixel_separates_into_its_stain_amounts() {
        // OD 0.5 Hematoxylin + OD 0.8 DAB, mixed by Beer-Lambert.
        let h = unit(HEMATOXYLIN);
        let d = unit(DAB);
        let residual = unit(cross(h, d));
        let pixel = stained([0.5, 0.8, 0.0], [h, d, residual]);

        let hema = run(&cmd(StainVectors::HDab, StainOutput::Stain1), &[pixel])[0];
        let dab = run(&cmd(StainVectors::HDab, StainOutput::Stain2), &[pixel])[0];
        let rest = run(&cmd(StainVectors::HDab, StainOutput::Stain3), &[pixel])[0];
        assert!((hema * MAX_OPTICAL_DENSITY - 0.5).abs() < 1e-3, "H {hema}");
        assert!((dab * MAX_OPTICAL_DENSITY - 0.8).abs() < 1e-3, "DAB {dab}");
        assert!(rest.abs() < 1e-3, "residual {rest}");
    }

    #[test]
    fn three_stains_separate_with_h_e_dab() {
        let vectors = [unit(HEMATOXYLIN), unit(EOSIN), unit(DAB)];
        let pixel = stained([0.3, 0.2, 0.6], vectors);
        let outputs = [
            StainOutput::Stain1,
            StainOutput::Stain2,
            StainOutput::Stain3,
        ];
        for (output, expected) in outputs.into_iter().zip([0.3, 0.2, 0.6]) {
            let got = run(&cmd(StainVectors::HEDab, output), &[pixel])[0];
            assert!(
                (got * MAX_OPTICAL_DENSITY - expected).abs() < 1e-3,
                "{output:?}: {got}"
            );
        }
    }

    #[test]
    fn the_8_bit_value_over_100_is_the_optical_density() {
        let pixel = stained([1.0, 0.0, 0.0], [unit(DAB), unit(HEMATOXYLIN), [0.0; 3]]);
        let dab = run(&cmd(StainVectors::HDab, StainOutput::Stain2), &[pixel])[0];
        assert!((dab * 255.0 / 100.0 - 1.0).abs() < 1e-3, "{dab}");
    }

    #[test]
    fn a_grey_background_counts_as_no_stain() {
        let mut c = cmd(StainVectors::HDab, StainOutput::Stain1);
        c.background = 230.0;
        let glass = 230.0 / 255.0;
        let out = run(&c, &[[glass, glass, glass], [1.0, 1.0, 1.0]]);
        assert!(out[0].abs() < 1e-6 && out[1].abs() < 1e-6, "{out:?}");
    }

    #[test]
    fn black_pixels_stay_finite_and_in_range() {
        let out = run(
            &cmd(StainVectors::HE, StainOutput::Stain1),
            &[[0.0, 0.0, 0.0]],
        );
        assert!(
            out[0].is_finite() && (0.0..=1.0).contains(&out[0]),
            "{out:?}"
        );
    }

    #[test]
    fn custom_vectors_equal_to_the_preset_give_the_same_result() {
        let custom = StainVectors::Custom {
            stain_1_red: HEMATOXYLIN[0],
            stain_1_green: HEMATOXYLIN[1],
            stain_1_blue: HEMATOXYLIN[2],
            stain_2_red: DAB[0],
            stain_2_green: DAB[1],
            stain_2_blue: DAB[2],
            stain_3_red: 0.0,
            stain_3_green: 0.0,
            stain_3_blue: 0.0,
        };
        let pixels = [[0.4, 0.5, 0.7], [0.9, 0.3, 0.2]];
        assert_eq!(
            run(&cmd(custom, StainOutput::Stain2), &pixels),
            run(&cmd(StainVectors::HDab, StainOutput::Stain2), &pixels)
        );
    }

    #[test]
    fn stains_of_the_same_color_are_refused() {
        let same = StainVectors::Custom {
            stain_1_red: 0.5,
            stain_1_green: 0.5,
            stain_1_blue: 0.5,
            stain_2_red: 0.2,
            stain_2_green: 0.2,
            stain_2_blue: 0.2,
            stain_3_red: 0.0,
            stain_3_green: 0.0,
            stain_3_blue: 0.0,
        };
        assert!(unmixing_matrix(&same).is_err());
    }

    #[test]
    fn a_gray_image_is_refused() {
        let img = Image::<f32, 1>::from_size_val(
            ImageSize {
                width: 1,
                height: 1,
            },
            0.5,
        )
        .unwrap();
        let mut ctx = PipelineContext::new_from_image_test(img).unwrap();
        let result = cmd(StainVectors::HDab, StainOutput::Stain1)
            .execute(&mut ctx, &mut GlobalPipelineCache::default());
        assert!(matches!(result, Err(InternalErrors::FormatMismatch { .. })));
    }
}
