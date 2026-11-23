//! # Translate
//!
//! Shifts an image by a fraction of a pixel: `output(y, x) = input(y − dy,
//! x − dx)` (as `scipy.ndimage.shift`), the same size, padded where the
//! shift empties the edge - for Translate Image and for aligning a channel
//! (Subtract Reference Channel).
//!
//! The shift is the same for every pixel, so are the interpolation
//! weights: each axis has one kernel (one tap for a whole-pixel shift or
//! Nearest, two for Linear, four for Cubic - Keys 1981, a = −0.5) and a
//! table of the source indices per output row and column with the padding
//! already applied. The rows are interpolated first into one row buffer,
//! then along it - no temporary image. Volumes (planes stacked in `data`)
//! are shifted plane by plane.
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-11-17
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.

use crate::algos::{Interpolation, Padding};
use evanalyzer_cfg::core_types::CitationMetadata;

/// A shift by `(dy, dx)` of `width × height` planes, its kernels and
/// index tables computed once.
pub(crate) struct Translation {
    rows: AxisTable,
    cols: AxisTable,
    width: usize,
    height: usize,
}

impl Translation {
    pub(crate) fn new(
        dy: f64,
        dx: f64,
        interpolation: Interpolation,
        padding: Padding,
        width: usize,
        height: usize,
    ) -> Self {
        Translation {
            rows: AxisKernel::new(dy, interpolation).table(height, padding),
            cols: AxisKernel::new(dx, interpolation).table(width, padding),
            width,
            height,
        }
    }

    /// Shifts `depth` planes of `src` (`C` values per pixel) into `dst`;
    /// `pad` fills for Constant padding.
    pub(crate) fn apply<const C: usize>(
        &self,
        src: &[f32],
        dst: &mut [f32],
        depth: usize,
        pad: f32,
    ) {
        if self.rows.taps == 1 && self.cols.taps == 1 {
            self.copy::<f32, C>(src, dst, depth, pad);
        } else {
            self.interpolate::<C>(src, dst, depth, pad);
        }
    }

    /// A whole-pixel shift (or Nearest) of any pixel type: copies rows,
    /// the part inside the image in one go. Only for a translation without
    /// interpolation ([`Self::apply`] picks it for those).
    pub(crate) fn copy<T: Copy, const C: usize>(
        &self,
        src: &[T],
        dst: &mut [T],
        depth: usize,
        pad: T,
    ) {
        let plane = Plane {
            width: self.width,
            height: self.height,
            depth,
        };
        copy_shifted::<T, C>(src, dst, &plane, &self.rows, &self.cols, pad);
    }

    fn interpolate<const C: usize>(&self, src: &[f32], dst: &mut [f32], depth: usize, pad: f32) {
        let plane = Plane {
            width: self.width,
            height: self.height,
            depth,
        };
        interpolate::<C>(src, dst, &plane, &self.rows, &self.cols, pad);
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

/// The frequency response of a shift of `shift` px along one axis as
/// [`Translation`] interpolates it, at `f` cycles per pixel: `Σ w_k ·
/// e^(2πi f (first + k))` (`output(i) = Σ w_k · input(i + first + k)`) -
/// the shift's phase and the interpolation's damping of high frequencies.
pub(crate) fn axis_response(shift: f64, interpolation: Interpolation, f: f64) -> (f64, f64) {
    let kernel = AxisKernel::new(shift, interpolation);
    let (mut re, mut im) = (0.0f64, 0.0f64);
    for k in 0..kernel.taps {
        let phase = 2.0 * std::f64::consts::PI * f * (kernel.first + k as isize) as f64;
        re += kernel.weights[k] as f64 * phase.cos();
        im += kernel.weights[k] as f64 * phase.sin();
    }
    (re, im)
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
pub(crate) fn padded_index(source: isize, len: usize, padding: Padding) -> Option<usize> {
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

/// Keys 1981, the cubic kernel.
pub(crate) static KEYS_1981: CitationMetadata = CitationMetadata {
    cite_key: "keys1981cubic",
    title: "Cubic convolution interpolation for digital image processing",
    authors: &["Robert G. Keys"],
    year: 1981,
    container: Some("IEEE Transactions on Acoustics, Speech, and Signal Processing"),
    doi: Some("10.1109/TASSP.1981.1163711"),
    url: None,
    pages: Some("1153-1160"),
};
