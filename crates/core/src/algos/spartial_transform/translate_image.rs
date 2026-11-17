//! # Translate Image
//!
//! Shifts an image by a fraction of a pixel (`output(y, x) = input(y − dy,
//! x − dx)`, as `scipy.ndimage.shift`), the same size as before - e.g. a
//! fixed shift between two channels of a microscope, or a structure
//! channel aligned to a spot channel before it is subtracted (a script
//! runs it as `run("translate_image", image, #{ dy: 0.5, dx: -1.2 })`).
//!
//! The shift is the same for every pixel, so are the interpolation
//! weights: each axis has one kernel (one tap for a whole-pixel shift or
//! Nearest, two for Linear, four for Cubic) and a table of the source
//! indices per output row and column with the padding already applied.
//! The rows are interpolated first into one row buffer, then along it - no
//! temporary image.
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-11-17
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.

use crate::algos::{ExecutionScope, GlobalPipelineCache, ImageAlgorithm, PipelineContext};
use crate::image::{ImageContainer, ManagedImage};
use evanalyzer_cfg::core_types::{CitationMetadata, InternalErrors};
use kornia_image::Image;
use macros::CommandsMeta;
use std::sync::Arc;

/// Shifts the image by a fraction of a pixel; it keeps its size.
///
/// Positive `dy` moves the content down, positive `dx` to the right:
/// `output(y, x) = input(y − dy, x − dx)` - as `scipy.ndimage.shift`, so
/// shifts measured in Python can be used as they are. The strip the shift
/// empties at the edge is filled as set under Padding. A volume is shifted
/// plane by plane; a colour image the same in every colour.
///
/// Cubic is Keys' cubic convolution with a = −0.5 (Catmull-Rom). Whole-pixel
/// shifts copy the pixels, whatever the interpolation. Label images can
/// only be shifted with Nearest.
///
/// Runs per tile: on an image split into tiles (larger than 4096 px) the
/// padding also fills the strip at the inner tile borders - as wide as the
/// shift.
///
/// A fixed channel shift as a step of a module:
///
/// ```json
/// { "id": 2, "enabled": true,
///   "command": { "type": "TRANSLATE_IMAGE", "dy": 0.5, "dx": -1.25,
///                "interpolation": "CUBIC", "padding": "REPLICATE", "padValue": 0.0 } }
/// ```
#[derive(CommandsMeta)]
#[cmdsmeta(category = "Preprocessing", display_name = "Translate Image")]
pub struct TranslateImage {
    /// Shift down (+) or up (−) in pixels; fractions allowed.
    #[cmdsmeta(default = 0.0, min = -1000.0, max = 1000.0, step = 0.1, summary = true)]
    pub dy: f64,

    /// Shift right (+) or left (−) in pixels; fractions allowed.
    #[cmdsmeta(default = 0.0, min = -1000.0, max = 1000.0, step = 0.1, summary = true)]
    pub dx: f64,

    /// How pixel values between grid positions are computed.
    #[cmdsmeta(default = Interpolation::Linear)]
    pub interpolation: Interpolation,

    /// What fills the strip that the shift empties at the image edge.
    #[cmdsmeta(default = Padding::Replicate)]
    pub padding: Padding,

    /// Fill value for padding = Constant.
    #[cmdsmeta(default = 0.0, min = -65535.0, max = 65535.0, step = 1.0, visibility = Advanced)]
    pub pad_value: f32,
}

/// How [`TranslateImage`] computes the values between pixels.
#[derive(Clone, Copy, PartialEq, Debug, CommandsMeta)]
pub enum Interpolation {
    /// The nearest pixel (the only choice for label images).
    Nearest,
    /// Bilinear, from the 4 nearest pixels.
    Linear,
    /// Bicubic, from the 16 nearest pixels (Keys, a = −0.5): sharper than
    /// Linear, may over- and undershoot at edges.
    Cubic,
}

/// What fills the strip [`TranslateImage`] empties at the image edge.
#[derive(Clone, Copy, PartialEq, Debug, CommandsMeta)]
pub enum Padding {
    /// The edge pixel repeated.
    Replicate,
    /// Mirrored at the edge, without repeating the edge pixel
    /// (…, 2, 1, 0, 1, 2, …).
    Reflect,
    /// A fixed value (`pad_value`).
    Constant,
}

impl ImageAlgorithm for TranslateImage {
    /// Shifts `ctx.image` into the scratch pad and swaps (a label image:
    /// into a new image).
    ///
    /// # Errors
    /// A shift that isn't finite or isn't smaller than the plane, and a
    /// label image with Linear or Cubic.
    fn execute(
        &self,
        ctx: &mut PipelineContext,
        _cache: &mut GlobalPipelineCache,
    ) -> Result<(), InternalErrors> {
        let size = ctx.image.size();
        let depth = ctx.image.dimensionality().depth().max(1) as usize;
        let (width, height) = (size.width, size.height / depth);
        self.check(width, height)?;
        let rows = AxisKernel::new(self.dy, self.interpolation).table(height, self.padding);
        let cols = AxisKernel::new(self.dx, self.interpolation).table(width, self.padding);
        let plane = Plane {
            width,
            height,
            depth,
        };

        match (ctx.image.as_ref(), Arc::make_mut(&mut ctx.scratch_pad)) {
            (ImageContainer::F32Gray(input), ImageContainer::F32Gray(output)) => {
                shift_f32::<1>(input, output, &plane, &rows, &cols, self.pad_value);
                ctx.swap()
            }
            (ImageContainer::F32Rgb(input), ImageContainer::F32Rgb(output)) => {
                shift_f32::<3>(input, output, &plane, &rows, &cols, self.pad_value);
                ctx.swap()
            }
            (ImageContainer::U32(input), _) => {
                if self.interpolation != Interpolation::Nearest {
                    return Err(InternalErrors::InvalidArgument(
                        "Translate Image: label images can only be shifted with Nearest - \
                         Linear and Cubic would mix the labels"
                            .into(),
                    ));
                }
                // A label image never becomes the scratch pad (see `swap`).
                let pad = self.pad_value.max(0.0).round() as u32;
                let mut data = vec![0u32; input.data.as_slice().len()];
                copy_shifted::<u32, 1>(input.data.as_slice(), &mut data, &plane, &rows, &cols, pad);
                ctx.image = Arc::new(ImageContainer::U32(ManagedImage {
                    data: Image::new(size, data).map_err(InternalErrors::from_kornia)?,
                    tile_offset: input.tile_offset,
                    plane: input.plane,
                    dimensionality: input.dimensionality,
                }));
                Ok(())
            }
            _ => Err(InternalErrors::FormatMismatch {
                expected: "F32Gray, F32Rgb or U32".into(),
                found: format!("{:?}", ctx.image),
            }),
        }
    }

    fn name(&self) -> &'static str {
        "Translate Image"
    }

    /// Measured by `tests::tile_ram` (see the trait method): one row buffer
    /// and the index tables, per row and column - nothing per pixel.
    fn ram_per_tile_pixel(&self) -> f64 {
        0.0
    }

    fn cite(&self) -> Vec<&'static CitationMetadata> {
        if self.interpolation == Interpolation::Cubic {
            vec![&CitationMetadata::DANMAYR, &KEYS_1981]
        } else {
            vec![&CitationMetadata::DANMAYR]
        }
    }

    fn execution_scope(&self) -> ExecutionScope {
        ExecutionScope::Tile
    }
}

impl TranslateImage {
    fn check(&self, width: usize, height: usize) -> Result<(), InternalErrors> {
        for (name, shift, len, side) in [
            ("dy", self.dy, height, "height"),
            ("dx", self.dx, width, "width"),
        ] {
            if !shift.is_finite() {
                return Err(InternalErrors::InvalidArgument(format!(
                    "Translate Image: {name} must be a finite number, got {shift}"
                )));
            }
            if shift.abs() >= len as f64 {
                return Err(InternalErrors::InvalidArgument(format!(
                    "Translate Image: |{name}| = {} must be smaller than the image's {side} \
                     ({len} px)",
                    shift.abs()
                )));
            }
        }
        Ok(())
    }
}

/// The size of one plane, and how many there are (stacked in `data`).
struct Plane {
    width: usize,
    height: usize,
    depth: usize,
}

/// Index-table entry for a source position outside the image with
/// Constant padding.
const PAD: u32 = u32::MAX;

/// One axis' interpolation kernel: output index `i` reads the source at
/// `i + first + k` with weight `weights[k]`, `k < taps`.
struct AxisKernel {
    first: isize,
    weights: [f32; 4],
    taps: usize,
}

impl AxisKernel {
    /// The kernel for a shift of `shift` px (`output(i) = input(i − shift)`).
    fn new(shift: f64, interpolation: Interpolation) -> Self {
        let source = -shift;
        let floor = source.floor();
        let t = source - floor;
        let one = |first: f64| AxisKernel {
            first: first as isize,
            weights: [1.0, 0.0, 0.0, 0.0],
            taps: 1,
        };
        if t == 0.0 {
            return one(floor);
        }
        match interpolation {
            // floor(i - shift + 0.5): ties round up for every pixel alike.
            Interpolation::Nearest => one((source + 0.5).floor()),
            Interpolation::Linear => AxisKernel {
                first: floor as isize,
                weights: [(1.0 - t) as f32, t as f32, 0.0, 0.0],
                taps: 2,
            },
            Interpolation::Cubic => AxisKernel {
                first: floor as isize - 1,
                weights: [
                    keys(1.0 + t) as f32,
                    keys(t) as f32,
                    keys(1.0 - t) as f32,
                    keys(2.0 - t) as f32,
                ],
                taps: 4,
            },
        }
    }

    /// The source indices of every output index `0..len` (`taps` each),
    /// padded - [`PAD`] where Constant padding fills.
    fn table(&self, len: usize, padding: Padding) -> AxisTable {
        let mut indices = Vec::with_capacity(len * self.taps);
        for i in 0..len {
            for k in 0..self.taps {
                let source = i as isize + self.first + k as isize;
                indices.push(padded_index(source, len, padding).map_or(PAD, |s| s as u32));
            }
        }
        AxisTable {
            indices,
            weights: self.weights,
            taps: self.taps,
            offset: self.first,
        }
    }
}

/// Keys' cubic convolution kernel with a = −0.5 (Catmull-Rom).
fn keys(d: f64) -> f64 {
    const A: f64 = -0.5;
    let d = d.abs();
    if d <= 1.0 {
        (A + 2.0) * d * d * d - (A + 3.0) * d * d + 1.0
    } else if d < 2.0 {
        A * d * d * d - 5.0 * A * d * d + 8.0 * A * d - 4.0 * A
    } else {
        0.0
    }
}

/// `source` inside `0..len` as `padding` maps it; `None`: Constant padding.
fn padded_index(source: isize, len: usize, padding: Padding) -> Option<usize> {
    let n = len as isize;
    if (0..n).contains(&source) {
        return Some(source as usize);
    }
    match padding {
        Padding::Replicate => Some(source.clamp(0, n - 1) as usize),
        Padding::Reflect => {
            if n == 1 {
                return Some(0);
            }
            // Period 2(n - 1): 0, 1, …, n - 1, n - 2, …, 1, 0, 1, …
            let period = 2 * (n - 1);
            let m = source.rem_euclid(period);
            Some(if m < n { m } else { period - m } as usize)
        }
        Padding::Constant => None,
    }
}

/// An axis' kernel with its padded source indices per output index.
struct AxisTable {
    indices: Vec<u32>,
    weights: [f32; 4],
    taps: usize,
    /// The whole-pixel offset (`first`) - with one tap, output `i` reads
    /// `i + offset` inside the image.
    offset: isize,
}

impl AxisTable {
    #[inline]
    fn taps_of(&self, i: usize) -> &[u32] {
        &self.indices[i * self.taps..(i + 1) * self.taps]
    }
}

/// Shifts a grey (`C` = 1) or colour (3) image into `output`.
fn shift_f32<const C: usize>(
    input: &ManagedImage<f32, C>,
    output: &mut ManagedImage<f32, C>,
    plane: &Plane,
    rows: &AxisTable,
    cols: &AxisTable,
    pad: f32,
) {
    output.tile_offset = input.tile_offset;
    output.plane = input.plane;
    output.dimensionality = input.dimensionality;
    let src = input.data.as_slice();
    let dst = output.data.as_slice_mut();
    if rows.taps == 1 && cols.taps == 1 {
        copy_shifted::<f32, C>(src, dst, plane, rows, cols, pad);
    } else {
        interpolate::<C>(src, dst, plane, rows, cols, pad);
    }
}

/// A whole-pixel shift: copies rows, the part inside the image in one go.
fn copy_shifted<T: Copy, const C: usize>(
    src: &[T],
    dst: &mut [T],
    plane: &Plane,
    rows: &AxisTable,
    cols: &AxisTable,
    pad: T,
) {
    let (w, h) = (plane.width, plane.height);
    let row_len = w * C;
    // Output columns whose source lies inside the image: x + offset in 0..w.
    let inside_start = (-cols.offset).clamp(0, w as isize) as usize;
    let inside_end = (w as isize - cols.offset).clamp(0, w as isize) as usize;
    for z in 0..plane.depth {
        for y in 0..h {
            let out_row = &mut dst[(z * h + y) * row_len..][..row_len];
            let source_row = rows.taps_of(y)[0];
            if source_row == PAD {
                out_row.fill(pad);
                continue;
            }
            let in_row = &src[(z * h + source_row as usize) * row_len..][..row_len];
            if inside_start < inside_end {
                let from = (inside_start as isize + cols.offset) as usize;
                out_row[inside_start * C..inside_end * C]
                    .copy_from_slice(&in_row[from * C..(from + inside_end - inside_start) * C]);
            }
            for x in (0..inside_start.min(w)).chain(inside_end.max(inside_start)..w) {
                let source = cols.taps_of(x)[0];
                for c in 0..C {
                    out_row[x * C + c] = if source == PAD {
                        pad
                    } else {
                        in_row[source as usize * C + c]
                    };
                }
            }
        }
    }
}

/// A fractional shift: each output row from the source rows (into one row
/// buffer), then along the row.
fn interpolate<const C: usize>(
    src: &[f32],
    dst: &mut [f32],
    plane: &Plane,
    rows: &AxisTable,
    cols: &AxisTable,
    pad: f32,
) {
    let (w, h) = (plane.width, plane.height);
    let row_len = w * C;
    let mut buffer = vec![0f32; row_len];
    for z in 0..plane.depth {
        let plane_src = &src[z * h * row_len..(z + 1) * h * row_len];
        for y in 0..h {
            let taps = rows.taps_of(y);
            // Along y - a single tap inside the image is the source row itself.
            let row: &[f32] = if rows.taps == 1 && taps[0] != PAD {
                &plane_src[taps[0] as usize * row_len..][..row_len]
            } else {
                buffer.fill(0.0);
                for (k, &source) in taps.iter().enumerate() {
                    let weight = rows.weights[k];
                    if source == PAD {
                        let add = weight * pad;
                        buffer.iter_mut().for_each(|v| *v += add);
                    } else {
                        let in_row = &plane_src[source as usize * row_len..][..row_len];
                        for (v, s) in buffer.iter_mut().zip(in_row) {
                            *v += weight * s;
                        }
                    }
                }
                &buffer
            };
            // Along x.
            let out_row = &mut dst[(z * h + y) * row_len..][..row_len];
            for x in 0..w {
                let taps = cols.taps_of(x);
                for c in 0..C {
                    let mut value = 0.0f32;
                    for (k, &source) in taps.iter().enumerate() {
                        let sample = if source == PAD {
                            pad
                        } else {
                            row[source as usize * C + c]
                        };
                        value += cols.weights[k] * sample;
                    }
                    out_row[x * C + c] = value;
                }
            }
        }
    }
}

static KEYS_1981: CitationMetadata = CitationMetadata {
    cite_key: "keys1981cubic",
    title: "Cubic convolution interpolation for digital image processing",
    authors: &["Robert G. Keys"],
    year: 1981,
    container: Some("IEEE Transactions on Acoustics, Speech, and Signal Processing"),
    doi: Some("10.1109/TASSP.1981.1163711"),
    url: None,
    pages: Some("1153-1160"),
};

// --- Test ------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ImageDimensionality;
    use crate::algos::script::script_executor::Script;
    use kornia_image::ImageSize;

    fn shift(dy: f64, dx: f64, interpolation: Interpolation, padding: Padding) -> TranslateImage {
        TranslateImage {
            dy,
            dx,
            interpolation,
            padding,
            pad_value: -1.0,
        }
    }

    fn context(image: ImageContainer) -> PipelineContext {
        let mut ctx = PipelineContext::new_from_image_test(
            Image::<f32, 1>::from_size_slice(
                ImageSize {
                    width: 1,
                    height: 1,
                },
                &[0.0],
            )
            .unwrap(),
        )
        .unwrap();
        ctx.image = Arc::new(image);
        ctx
    }

    fn gray(width: usize, height: usize, depth: u32, values: Vec<f32>) -> ImageContainer {
        ImageContainer::F32Gray(ManagedImage {
            data: Image::new(ImageSize { width, height }, values).unwrap(),
            tile_offset: Default::default(),
            plane: None,
            dimensionality: if depth > 1 {
                ImageDimensionality::Volume { depth }
            } else {
                ImageDimensionality::Planar
            },
        })
    }

    /// Runs `command` on `image`; the result's values.
    fn run(command: &TranslateImage, image: ImageContainer) -> Vec<f32> {
        let mut ctx = context(image);
        command
            .run(&mut ctx, &mut GlobalPipelineCache::default())
            .unwrap();
        values(&ctx)
    }

    fn values(ctx: &PipelineContext) -> Vec<f32> {
        match ctx.image.as_ref() {
            ImageContainer::F32Gray(img) => img.data.as_slice().to_vec(),
            ImageContainer::F32Rgb(img) => img.data.as_slice().to_vec(),
            ImageContainer::U32(img) => img.data.as_slice().iter().map(|v| *v as f32).collect(),
        }
    }

    /// 4 x 3, value = 10 · y + x.
    fn grid() -> ImageContainer {
        gray(
            4,
            3,
            1,
            (0..3)
                .flat_map(|y| (0..4).map(move |x| (10 * y + x) as f32))
                .collect(),
        )
    }

    #[test]
    fn a_whole_pixel_shift_copies_and_pads() {
        // dy = 1, dx = -2: output(y, x) = input(y - 1, x + 2).
        let expect = |padding, edge: [f32; 12]| {
            for interpolation in [
                Interpolation::Nearest,
                Interpolation::Linear,
                Interpolation::Cubic,
            ] {
                assert_eq!(
                    run(&shift(1.0, -2.0, interpolation, padding), grid()),
                    edge,
                    "{padding:?} {interpolation:?}"
                );
            }
        };
        #[rustfmt::skip]
        expect(Padding::Constant, [
            -1.0, -1.0, -1.0, -1.0,
             2.0,  3.0, -1.0, -1.0,
            12.0, 13.0, -1.0, -1.0,
        ]);
        #[rustfmt::skip]
        expect(Padding::Replicate, [
             2.0,  3.0,  3.0,  3.0,
             2.0,  3.0,  3.0,  3.0,
            12.0, 13.0, 13.0, 13.0,
        ]);
        // Rows: -1 -> 1. Columns: 4 -> 2, 5 -> 1.
        #[rustfmt::skip]
        expect(Padding::Reflect, [
            12.0, 13.0, 12.0, 11.0,
             2.0,  3.0,  2.0,  1.0,
            12.0, 13.0, 12.0, 11.0,
        ]);
    }

    #[test]
    fn half_a_pixel_linear_is_the_mean_of_the_neighbours() {
        // A ramp along x: 0, 1, 4, 9, … (not linear, so the mean is visible).
        let width = 8;
        let ramp: Vec<f32> = (0..width).map(|x| (x * x) as f32).collect();
        let out = run(
            &shift(0.0, 0.5, Interpolation::Linear, Padding::Replicate),
            gray(width, 1, 1, ramp.clone()),
        );
        for x in 1..width {
            assert_eq!(out[x], (ramp[x - 1] + ramp[x]) / 2.0, "x = {x}");
        }
    }

    #[test]
    fn positive_shifts_move_down_and_right() {
        let mut values = vec![0.0f32; 12 * 12];
        values[5 * 12 + 5] = 1.0;
        for interpolation in [
            Interpolation::Nearest,
            Interpolation::Linear,
            Interpolation::Cubic,
        ] {
            let out = run(
                &shift(2.0, 3.0, interpolation, Padding::Constant),
                gray(12, 12, 1, values.clone()),
            );
            let brightest = (0..out.len())
                .max_by(|a, b| out[*a].total_cmp(&out[*b]))
                .unwrap();
            assert_eq!(
                (brightest / 12, brightest % 12),
                (7, 8),
                "{interpolation:?}"
            );
        }
    }

    #[test]
    fn cubic_there_and_back_restores_a_smooth_image() {
        let (w, h) = (32, 24);
        let smooth: Vec<f32> = (0..h)
            .flat_map(|y| {
                (0..w).map(move |x| {
                    100.0 + 20.0 * (x as f32 / 5.0).sin() + 15.0 * (y as f32 / 7.0).cos()
                })
            })
            .collect();
        let there = shift(0.7, -1.3, Interpolation::Cubic, Padding::Replicate);
        let back = shift(-0.7, 1.3, Interpolation::Cubic, Padding::Replicate);
        let mut ctx = context(gray(w, h, 1, smooth.clone()));
        let mut cache = GlobalPipelineCache::default();
        there.run(&mut ctx, &mut cache).unwrap();
        back.run(&mut ctx, &mut cache).unwrap();
        let out = values(&ctx);
        for y in 3..h - 3 {
            for x in 3..w - 3 {
                let (got, want) = (out[y * w + x], smooth[y * w + x]);
                assert!((got - want).abs() < 0.5, "({y}, {x}): {got} vs {want}");
            }
        }
    }

    #[test]
    fn padding_fills_the_emptied_strip() {
        let row = || gray(5, 1, 1, vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let whole = |padding| run(&shift(0.0, 2.0, Interpolation::Linear, padding), row());
        assert_eq!(whole(Padding::Replicate), [1.0, 1.0, 1.0, 2.0, 3.0]);
        assert_eq!(whole(Padding::Reflect), [3.0, 2.0, 1.0, 2.0, 3.0]);
        assert_eq!(whole(Padding::Constant), [-1.0, -1.0, 1.0, 2.0, 3.0]);

        // Linear by half a pixel with Constant: the first pixel mixes the
        // pad value and 1.
        let half = run(
            &shift(0.0, 0.5, Interpolation::Linear, Padding::Constant),
            row(),
        );
        assert_eq!(half[0], 0.0);
        assert_eq!(half[1], 1.5);
    }

    /// A shift is smaller than the image, but the cubic kernel reaches
    /// further: on a 2 x 2 image by 1.5 px, output column 0 reads sources
    /// -3..=0, which Reflect folds more than once (period 2: 1, 0, 1, 0).
    #[test]
    fn reflect_folds_as_often_as_needed() {
        #[rustfmt::skip]
        let small = gray(2, 2, 1, vec![
            1.0, 2.0,
            3.0, 4.0,
        ]);
        // Weights keys(1.5), keys(0.5), keys(0.5), keys(1.5) = -1/16, 9/16,
        // 9/16, -1/16 on 2, 1, 2, 1 (column 0) or 1, 2, 1, 2 (column 1).
        assert_eq!(
            run(
                &shift(0.0, 1.5, Interpolation::Cubic, Padding::Reflect),
                small
            ),
            [1.5, 1.5, 3.5, 3.5]
        );
        assert_eq!(padded_index(-3, 2, Padding::Reflect), Some(1));
        assert_eq!(padded_index(-4, 2, Padding::Reflect), Some(0));
        assert_eq!(padded_index(5, 2, Padding::Reflect), Some(1));
        assert_eq!(padded_index(-7, 3, Padding::Reflect), Some(1));
        assert_eq!(padded_index(-5, 1, Padding::Reflect), Some(0));
    }

    #[test]
    fn a_volume_shifts_plane_by_plane() {
        // Two 3 x 2 planes: 1s, then 2s.
        let volume = gray(3, 4, 2, [vec![1.0; 6], vec![2.0; 6]].concat());
        for interpolation in [
            Interpolation::Nearest,
            Interpolation::Linear,
            Interpolation::Cubic,
        ] {
            for dy in [1.0, 0.5] {
                let out = run(
                    &shift(dy, 0.0, interpolation, Padding::Replicate),
                    volume.clone(),
                );
                assert!(
                    out[..6].iter().all(|v| *v == 1.0),
                    "{interpolation:?} {dy}: {out:?}"
                );
                assert!(
                    out[6..].iter().all(|v| *v == 2.0),
                    "{interpolation:?} {dy}: {out:?}"
                );
            }
        }
    }

    #[test]
    fn a_colour_image_shifts_every_colour_alike() {
        // 4 x 1 RGB: R = x, G = 10 x, B = 100 x.
        let values: Vec<f32> = (0..4)
            .flat_map(|x| [x as f32, 10.0 * x as f32, 100.0 * x as f32])
            .collect();
        let image = ImageContainer::F32Rgb(ManagedImage {
            data: Image::new(
                ImageSize {
                    width: 4,
                    height: 1,
                },
                values,
            )
            .unwrap(),
            tile_offset: Default::default(),
            plane: None,
            dimensionality: ImageDimensionality::Planar,
        });
        let out = run(
            &shift(0.0, 0.5, Interpolation::Linear, Padding::Replicate),
            image,
        );
        for x in 0..4 {
            let r = out[x * 3];
            assert_eq!(
                (out[x * 3 + 1], out[x * 3 + 2]),
                (10.0 * r, 100.0 * r),
                "x = {x}"
            );
        }
        assert_eq!(out[3], 0.5);
    }

    #[test]
    fn labels_shift_only_with_nearest() {
        let labels = || {
            ImageContainer::U32(ManagedImage {
                data: Image::new(
                    ImageSize {
                        width: 3,
                        height: 1,
                    },
                    vec![7u32, 8, 9],
                )
                .unwrap(),
                tile_offset: Default::default(),
                plane: None,
                dimensionality: ImageDimensionality::Planar,
            })
        };
        let mut nearest = shift(0.0, 1.2, Interpolation::Nearest, Padding::Replicate);
        assert_eq!(run(&nearest, labels()), [7.0, 7.0, 8.0]);
        nearest.padding = Padding::Constant;
        // A negative pad value is no label: 0.
        assert_eq!(run(&nearest, labels()), [0.0, 7.0, 8.0]);

        let mut ctx = context(labels());
        let err = shift(0.0, 1.0, Interpolation::Linear, Padding::Replicate)
            .run(&mut ctx, &mut GlobalPipelineCache::default())
            .unwrap_err();
        assert!(
            err.to_string().contains("only be shifted with Nearest"),
            "{err}"
        );
    }

    #[test]
    fn a_shift_must_be_finite_and_smaller_than_the_image() {
        for (dy, dx) in [
            (f64::NAN, 0.0),
            (0.0, f64::INFINITY),
            (3.0, 0.0),
            (0.0, -4.0),
        ] {
            let mut ctx = context(grid());
            let result = shift(dy, dx, Interpolation::Linear, Padding::Replicate)
                .run(&mut ctx, &mut GlobalPipelineCache::default());
            assert!(result.is_err(), "({dy}, {dx})");
        }
        assert!(
            run(
                &shift(2.9, -3.9, Interpolation::Linear, Padding::Replicate),
                grid()
            )
            .len()
                == 12
        );
    }

    /// The step of the doc comment's example loads as written.
    #[test]
    fn the_documented_module_step_loads() {
        use evanalyzer_cfg::settings::pipeline_command::PipelineCommand;
        use evanalyzer_cfg::settings::pipeline_command_settings::{
            SpartialTransformTranslateImageInterpolationSettings as I,
            SpartialTransformTranslateImagePaddingSettings as P,
        };
        use evanalyzer_cfg::settings::pipeline_settings::PipelineStepSettings;
        let step: PipelineStepSettings = serde_json::from_str(
            r#"{ "id": 2, "enabled": true,
                 "command": { "type": "TRANSLATE_IMAGE", "dy": 0.5, "dx": -1.25,
                              "interpolation": "CUBIC", "padding": "REPLICATE", "padValue": 0.0 } }"#,
        )
        .unwrap();
        let Some(PipelineCommand::TranslateImage(s)) = step.command() else {
            panic!("a Translate Image step")
        };
        assert_eq!((s.dy, s.dx, s.pad_value), (0.5, -1.25, 0.0));
        assert_eq!(
            (s.interpolation.clone(), s.padding.clone()),
            (I::Cubic, P::Replicate)
        );
    }

    #[test]
    fn a_script_runs_it_like_the_command() {
        let direct = run(
            &shift(1.0, 0.0, Interpolation::Linear, Padding::Replicate),
            grid(),
        );

        let mut ctx = context(grid());
        Script {
            classes: vec![],
            source: r#"
                let raw = image();
                // The options as the structure subtraction will pass them.
                let cubic = run("translate_image", raw, #{
                    dy: 0.5, dx: -1.2, interpolation: "CUBIC", padding: "REPLICATE" });
                let shifted = run("translate_image", raw, #{ dy: 1.0, dx: 0.0 });
                set_image(shifted);
            "#
            .into(),
            values: Default::default(),
        }
        .execute(&mut ctx, &mut GlobalPipelineCache::default())
        .unwrap();
        assert_eq!(values(&ctx), direct);
    }
}
