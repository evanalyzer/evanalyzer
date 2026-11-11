//! # ops
//!
//! Dilation, erosion, opening and closing of Morphological Transformation.
//!
//! Derived from kornia-rs (`crates/kornia-imgproc/src/morphology/ops.rs`,
//! <https://github.com/kornia/kornia-rs>), Copyright kornia.org, licensed
//! under the Apache License 2.0 (<https://www.apache.org/licenses/LICENSE-2.0>).
//! Modified by Joachim Danmayr, 2026 (moved into this command and adapted).
//! The modifications are licensed under the **PolyForm Noncommercial
//! License 1.0.0** like the rest of this crate.
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-06-05
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.

use super::kernels::{Kernel, Separable};
use kornia_image::{Image, ImageError, ImageSize};
use kornia_imgproc::padding::{Padding2D, PaddingMode, spatial_padding};
use rayon::prelude::*;

/// How the values inside a structuring element's window are combined into
/// one output value.
///
/// `combine` must be associative and commutative (the fast paths fold rows
/// first and columns second, or the two lines of a cross separately), so
/// the result doesn't depend on the order the window is visited in.
pub(super) trait Fold<T>: Copy + Send + Sync {
    /// Folds `candidate` into the window's value so far.
    fn combine(self, candidate: T, acc: T) -> T;

    /// Whether [`finish`](Self::finish) reads the window's centre pixel - a
    /// constant, so the folds that don't need it never load it.
    const USES_CENTER: bool = false;

    /// Turns the folded window value into the output pixel.
    fn finish(self, _center: T, acc: T) -> T {
        acc
    }
}

/// Grayscale dilation: the window's maximum.
#[derive(Clone, Copy)]
pub(super) struct Max;
impl<T: PartialOrd + Copy> Fold<T> for Max {
    #[inline(always)]
    fn combine(self, candidate: T, acc: T) -> T {
        if candidate > acc { candidate } else { acc }
    }
}

/// Grayscale erosion: the window's minimum (NaN never replaces a value
/// already found, as before).
#[derive(Clone, Copy)]
pub(super) struct Min;
impl<T: PartialOrd + Copy> Fold<T> for Min {
    #[inline(always)]
    fn combine(self, candidate: T, acc: T) -> T {
        if candidate < acc { candidate } else { acc }
    }
}

/// Label erosion: a pixel keeps its class only if every pixel of the window
/// has that same class, otherwise it becomes background (`T::default()`).
///
/// That is each class's mask eroded on its own: where two classes touch,
/// both shrink away from each other by the same amount, independent of the
/// class numbers (the window minimum let the lower id eat into the higher).
/// With only one class it equals the minimum exactly. Background absorbs:
/// `agree(0, x) == 0`, so the fold is associative and commutative.
#[derive(Clone, Copy)]
pub(super) struct Agree;
impl<T: PartialEq + Default + Copy> Fold<T> for Agree {
    #[inline(always)]
    fn combine(self, candidate: T, acc: T) -> T {
        if candidate == acc { acc } else { T::default() }
    }
}

/// Label dilation: classes grow into background only - a labelled pixel
/// keeps its class, a background pixel takes the highest class in its
/// window.
///
/// That is each class's mask dilated on its own, merged so that a pixel
/// that already has a class keeps it (the window maximum let the higher id
/// overwrite the lower). Where several classes reach the same background
/// pixel the highest id wins - arbitrary but deterministic. With only one
/// class it equals the maximum exactly.
#[derive(Clone, Copy)]
pub(super) struct GrowIntoBackground;
impl<T: PartialOrd + Default + Copy> Fold<T> for GrowIntoBackground {
    #[inline(always)]
    fn combine(self, candidate: T, acc: T) -> T {
        if candidate > acc { candidate } else { acc }
    }

    const USES_CENTER: bool = true;

    #[inline(always)]
    fn finish(self, center: T, acc: T) -> T {
        if center != T::default() { center } else { acc }
    }
}

/// Dilate an image using a [`Kernel`].
///
/// Dilation expands white regions in the image. Each pixel is replaced
/// by the maximum value in the neighborhood defined by the kernel.
///
/// # Arguments
///
/// * `src` - The source image.
/// * `dst` - The destination image (will be overwritten).
/// * `kernel` - The morphological structuring element ([`Kernel`]).
/// * `padding_mode` - The border handling mode ([`PaddingMode`]).
/// * `constant_value` - The fill value for constant padding.
///
/// # Returns
///
/// Ok(()) on success, or [`ImageError`] if shapes don't match.
pub fn dilate<T: Copy + Default + Send + Sync + PartialOrd, const C: usize>(
    src: &Image<T, C>,
    dst: &mut Image<T, C>,
    kernel: &Kernel,
    padding_mode: PaddingMode,
    constant_value: [T; C],
) -> Result<(), ImageError> {
    morph(src, dst, kernel, padding_mode, constant_value, Max)
}

/// Erode an image using a [`Kernel`].
///
/// Erosion shrinks white regions in the image. Each pixel is replaced
/// by the minimum value in the neighborhood defined by the kernel.
///
/// # Arguments
///
/// * `src` - The source image.
/// * `dst` - The destination image (will be overwritten).
/// * `kernel` - The morphological structuring element ([`Kernel`]).
/// * `padding_mode` - The border handling mode ([`PaddingMode`]).
/// * `constant_value` - The fill value for constant padding.
///
/// # Returns
///
/// Ok(()) on success, or [`ImageError`] if shapes don't match.
pub fn erode<T: Copy + Default + Send + Sync + PartialOrd, const C: usize>(
    src: &Image<T, C>,
    dst: &mut Image<T, C>,
    kernel: &Kernel,
    padding_mode: PaddingMode,
    constant_value: [T; C],
) -> Result<(), ImageError> {
    morph(src, dst, kernel, padding_mode, constant_value, Min)
}

/// Opening: erosion followed by dilation.
///
/// Removes small objects and smooths object boundaries.
///
/// # Arguments
///
/// * `src` - The source image.
/// * `dst` - The destination image (will be overwritten).
/// * `kernel` - The morphological structuring element ([`Kernel`]).
/// * `padding_mode` - The border handling mode ([`PaddingMode`]).
/// * `constant_value` - The fill value for constant padding.
///
/// # Returns
///
/// Ok(()) on success, or [`ImageError`] if shapes don't match.
pub fn open<T: Copy + Default + Send + Sync + PartialOrd, const C: usize>(
    src: &Image<T, C>,
    dst: &mut Image<T, C>,
    kernel: &Kernel,
    padding_mode: PaddingMode,
    constant_value: [T; C],
) -> Result<(), ImageError> {
    let mut temp_img = src.clone();
    erode(src, &mut temp_img, kernel, padding_mode, constant_value)?;
    dilate(&temp_img, dst, kernel, padding_mode, constant_value)?;
    Ok(())
}

/// Closing: dilation followed by erosion.
///
/// Fills small holes and smooths object boundaries.
///
/// # Arguments
///
/// * `src` - The source image.
/// * `dst` - The destination image (will be overwritten).
/// * `kernel` - The morphological structuring element ([`Kernel`]).
/// * `padding_mode` - The border handling mode ([`PaddingMode`]).
/// * `constant_value` - The fill value for constant padding.
///
/// # Returns
///
/// Ok(()) on success, or [`ImageError`] if shapes don't match.
pub fn close<T: Copy + Default + Send + Sync + PartialOrd, const C: usize>(
    src: &Image<T, C>,
    dst: &mut Image<T, C>,
    kernel: &Kernel,
    padding_mode: PaddingMode,
    constant_value: [T; C],
) -> Result<(), ImageError> {
    let mut temp_img = src.clone();
    dilate(src, &mut temp_img, kernel, padding_mode, constant_value)?;
    erode(&temp_img, dst, kernel, padding_mode, constant_value)?;
    Ok(())
}

// --- Label maps (class ids, 0 = background) -----------------------------
//
// Every class is treated as its own binary mask, so the result never depends
// on how the classes are numbered. With one class (plus background) each of
// these equals the grayscale operation above exactly. The kernels must be
// symmetric (odd sizes - the command rejects even ones).

/// Label dilation: every class grows into the background, a labelled pixel
/// keeps its class. Where several classes reach the same background pixel,
/// the highest class id wins.
pub fn dilate_labels(
    src: &Image<u32, 1>,
    dst: &mut Image<u32, 1>,
    kernel: &Kernel,
    padding_mode: PaddingMode,
    constant_value: [u32; 1],
) -> Result<(), ImageError> {
    morph(
        src,
        dst,
        kernel,
        padding_mode,
        constant_value,
        GrowIntoBackground,
    )
}

/// Label erosion: a pixel keeps its class only where the whole window has
/// that class; touching classes shrink away from each other equally.
pub fn erode_labels(
    src: &Image<u32, 1>,
    dst: &mut Image<u32, 1>,
    kernel: &Kernel,
    padding_mode: PaddingMode,
    constant_value: [u32; 1],
) -> Result<(), ImageError> {
    morph(src, dst, kernel, padding_mode, constant_value, Agree)
}

/// Label opening: each class opened on its own.
///
/// The label erosion followed by a plain maximum dilation is exact: a pixel
/// `q` survives the erosion with class `d` only if `q`'s whole window is `d`,
/// so (the kernel being symmetric) every pixel whose window contains `q` is
/// itself `d` in the input. The dilation's window of any pixel therefore
/// holds only that pixel's own class or background - no class can spread
/// into another, and the maximum picks exactly the per-class opening.
pub fn open_labels(
    src: &Image<u32, 1>,
    dst: &mut Image<u32, 1>,
    kernel: &Kernel,
    padding_mode: PaddingMode,
    constant_value: [u32; 1],
) -> Result<(), ImageError> {
    let mut temp_img = src.clone();
    erode_labels(src, &mut temp_img, kernel, padding_mode, constant_value)?;
    dilate(&temp_img, dst, kernel, padding_mode, constant_value)?;
    Ok(())
}

/// Label closing: each class closed on its own; a pixel keeps its class
/// where that class's closing still covers it, any other pixel takes the
/// highest class whose closing covers it.
///
/// Like the plain closing (and with one class exactly it), objects touching
/// the image border lose their border pixels: the border is padded with
/// background before the erosion.
///
/// Unlike opening, the per-class closings can overlap (a class may bridge
/// over a thin line of another), so with two or more classes each class is
/// closed as its own `u8` mask (less RAM than one `u32` closing). With one
/// class it is the plain closing (the command checks that once with
/// [`has_several_classes`] and calls [`close`] directly then).
pub fn close_labels(
    src: &Image<u32, 1>,
    dst: &mut Image<u32, 1>,
    kernel: &Kernel,
    padding_mode: PaddingMode,
    constant_value: [u32; 1],
) -> Result<(), ImageError> {
    let classes = distinct_classes(src.as_slice());
    if classes.len() < 2 {
        return close(src, dst, kernel, padding_mode, constant_value);
    }
    if src.size() != dst.size() {
        return Err(ImageError::InvalidImageSize(
            dst.width(),
            dst.height(),
            src.width(),
            src.height(),
        ));
    }

    let size = src.size();
    let src_slice = src.as_slice();
    dst.as_slice_mut().fill(0);
    let mut mask = Image::<u8, 1>::from_size_val(size, 0)?;
    let mut closed = Image::<u8, 1>::from_size_val(size, 0)?;
    // Ascending, so a later (higher) class wins a pixel - unless the pixel's
    // own class already covers it.
    for &class in &classes {
        mask.as_slice_mut()
            .par_iter_mut()
            .zip(src_slice.par_iter())
            .for_each(|(m, &s)| *m = (s == class) as u8);
        let pad = [(constant_value[0] == class) as u8];
        close(&mask, &mut closed, kernel, padding_mode, pad)?;
        dst.as_slice_mut()
            .par_iter_mut()
            .zip(src_slice.par_iter())
            .zip(closed.as_slice().par_iter())
            .for_each(|((d, &s), &c)| {
                if c != 0 && (s == 0 || *d != s) {
                    *d = class;
                }
            });
    }
    Ok(())
}

/// Whether a label map holds two or more classes (besides background) - a
/// separate pass; [`transform_labels`] gets the same answer from its padded
/// copy instead (constant padding).
pub fn has_several_classes(labels: &[u32]) -> bool {
    let Some(&first) = labels.iter().find(|&&v| v != 0) else {
        return false;
    };
    labels.par_chunks(1 << 16).any(|chunk| {
        !chunk
            .iter()
            .fold(true, |ok, &v| ok & ((v == 0) | (v == first)))
    })
}

/// What one or more rows of a label map hold: the first class seen (0: none)
/// and whether a second one turned up.
#[derive(Default, Clone, Copy)]
struct ClassScan {
    first: u32,
    several: bool,
}

impl ClassScan {
    /// A branch-free, vectorizable compare per pixel.
    fn of_row(row: &[u32]) -> Self {
        let first = row.iter().copied().find(|&v| v != 0).unwrap_or(0);
        let several = first != 0
            && !row
                .iter()
                .fold(true, |ok, &v| ok & ((v == 0) | (v == first)));
        Self { first, several }
    }

    fn combine(a: Self, b: Self) -> Self {
        Self {
            first: if a.first != 0 { a.first } else { b.first },
            several: a.several || b.several || (a.first != 0 && b.first != 0 && a.first != b.first),
        }
    }
}

/// The label map transformation the command runs: with two or more classes
/// the label operations above, with one class (the usual case after a
/// Threshold) the plain minimum/maximum, which equal them then.
///
/// Which case it is comes from the padded copy the first pass needs anyway
/// (with constant padding): each row is checked right after it is copied,
/// so the one-class path costs no extra pass over the map.
pub fn transform_labels(
    op: super::MorphOps,
    src: &Image<u32, 1>,
    dst: &mut Image<u32, 1>,
    kernel: &Kernel,
    padding_mode: PaddingMode,
    constant_value: [u32; 1],
) -> Result<(), ImageError> {
    use super::MorphOps;
    check_sizes(src, dst)?;
    let (padded, several) = if let PaddingMode::Constant = padding_mode {
        let (padded, scan) = padded_constant(
            src,
            kernel,
            constant_value,
            ClassScan::of_row,
            ClassScan::combine,
        )?;
        (padded, scan.several)
    } else {
        let padded = padded_copy(src, kernel, padding_mode, constant_value)?;
        (padded, has_several_classes(src.as_slice()))
    };

    if several {
        // Rare: the label operations pad again themselves.
        drop(padded);
        return match op {
            MorphOps::Dilate => dilate_labels(src, dst, kernel, padding_mode, constant_value),
            MorphOps::Erode => erode_labels(src, dst, kernel, padding_mode, constant_value),
            MorphOps::Open => open_labels(src, dst, kernel, padding_mode, constant_value),
            MorphOps::Close => close_labels(src, dst, kernel, padding_mode, constant_value),
        };
    }

    match op {
        MorphOps::Dilate => morph_padded(src, &padded, dst, kernel, Max),
        MorphOps::Erode => morph_padded(src, &padded, dst, kernel, Min),
        MorphOps::Open => {
            let mut temp_img = src.clone();
            morph_padded(src, &padded, &mut temp_img, kernel, Min)?;
            drop(padded);
            dilate(&temp_img, dst, kernel, padding_mode, constant_value)
        }
        MorphOps::Close => {
            let mut temp_img = src.clone();
            morph_padded(src, &padded, &mut temp_img, kernel, Max)?;
            drop(padded);
            erode(&temp_img, dst, kernel, padding_mode, constant_value)
        }
    }
}

/// The distinct non-background values of a label map, ascending. The last
/// value seen is cached, so a map with one class costs one compare per
/// pixel.
fn distinct_classes(labels: &[u32]) -> Vec<u32> {
    let mut classes: Vec<u32> = Vec::new();
    let mut last = 0u32;
    for &v in labels {
        if v != last {
            last = v;
            if v != 0 && !classes.contains(&v) {
                classes.push(v);
            }
        }
    }
    classes.sort_unstable();
    classes
}

// --- Shared implementation ----------------------------------------------

/// Runs `fold` over every pixel's window: the O(k) fast paths for box and
/// cross kernels, a scan of the kernel's set pixels otherwise.
fn morph<T: Copy + Default + Send + Sync + PartialEq, F: Fold<T>, const C: usize>(
    src: &Image<T, C>,
    dst: &mut Image<T, C>,
    kernel: &Kernel,
    padding_mode: PaddingMode,
    constant_value: [T; C],
    fold: F,
) -> Result<(), ImageError> {
    check_sizes(src, dst)?;
    let padded = padded_copy(src, kernel, padding_mode, constant_value)?;
    morph_padded(src, &padded, dst, kernel, fold)
}

/// [`morph`] on an already padded copy of `src` (see [`padded_copy`]).
fn morph_padded<T: Copy + Default + Send + Sync, F: Fold<T>, const C: usize>(
    src: &Image<T, C>,
    padded: &Image<T, C>,
    dst: &mut Image<T, C>,
    kernel: &Kernel,
    fold: F,
) -> Result<(), ImageError> {
    check_sizes(src, dst)?;
    match kernel.separable() {
        Separable::Box => box_separable(src, padded, dst, kernel, fold),
        Separable::Cross => cross_separable(src, padded, dst, kernel, fold),
        Separable::None => naive(src, padded, dst, kernel, fold),
    }
}

fn check_sizes<T, const C: usize>(src: &Image<T, C>, dst: &Image<T, C>) -> Result<(), ImageError> {
    if src.size() != dst.size() {
        return Err(ImageError::InvalidImageSize(
            dst.width(),
            dst.height(),
            src.width(),
            src.height(),
        ));
    }
    Ok(())
}

/// Copies `src` into a buffer padded by the kernel's radius on every side.
fn padded_copy<T: Copy + Default + Send + Sync + PartialEq, const C: usize>(
    src: &Image<T, C>,
    kernel: &Kernel,
    padding_mode: PaddingMode,
    constant_value: [T; C],
) -> Result<Image<T, C>, ImageError> {
    if let PaddingMode::Constant = padding_mode {
        return Ok(padded_constant(src, kernel, constant_value, |_| (), |_, _| ())?.0);
    }
    let (pad_h, pad_w) = kernel.pad();
    let padded_size = ImageSize {
        width: src.width() + 2 * pad_w,
        height: src.height() + 2 * pad_h,
    };
    let padded_buffer = vec![T::default(); padded_size.width * padded_size.height * C];
    let mut padded = Image::new(padded_size, padded_buffer)?;
    let padding = Padding2D {
        top: pad_h,
        bottom: pad_h,
        left: pad_w,
        right: pad_w,
    };
    spatial_padding(src, &mut padded, padding, padding_mode, constant_value)?;
    Ok(padded)
}

/// Constant padding: the same buffer as kornia's `spatial_padding`, built
/// with less work - a constant equal to `T::default()` comes with the zeroed
/// allocation (kornia writes it into every pixel first). The rows are copied
/// on one thread like kornia's: in parallel the first touches of the fresh
/// pages contend in the kernel - faster wall time, but measurably more CPU
/// time, which batch runs (every core busy with tiles) pay for. `on_row` sees every source row right after its copy
/// (while it is in the cache), its results combined with `combine` - how the
/// label path checks the classes without an extra pass over the map.
fn padded_constant<T, S, const C: usize>(
    src: &Image<T, C>,
    kernel: &Kernel,
    constant_value: [T; C],
    on_row: impl Fn(&[T]) -> S,
    combine: impl Fn(S, S) -> S,
) -> Result<(Image<T, C>, S), ImageError>
where
    T: Copy + Default + Send + Sync + PartialEq,
    S: Send + Default,
{
    let (pad_h, pad_w) = kernel.pad();
    let (width, height) = (src.width(), src.height());
    let padded_size = ImageSize {
        width: width + 2 * pad_w,
        height: height + 2 * pad_h,
    };
    let pixels = padded_size.width * padded_size.height;
    let buffer = if constant_value.iter().all(|&v| v == T::default()) {
        vec![T::default(); pixels * C]
    } else {
        let mut buffer = Vec::with_capacity(pixels * C);
        for _ in 0..pixels {
            buffer.extend_from_slice(&constant_value);
        }
        buffer
    };
    let mut padded = Image::new(padded_size, buffer)?;

    let (src_stride, padded_stride) = (width * C, padded_size.width * C);
    let summary = if src_stride == 0 {
        S::default()
    } else {
        padded
            .as_slice_mut()
            .chunks_exact_mut(padded_stride)
            .skip(pad_h)
            .take(height)
            .zip(src.as_slice().chunks_exact(src_stride))
            .map(|(dst_row, src_row)| {
                dst_row[pad_w * C..pad_w * C + src_stride].copy_from_slice(src_row);
                on_row(src_row)
            })
            .fold(S::default(), &combine)
    };
    Ok((padded, summary))
}

/// Scan of every set pixel of an arbitrary kernel (ellipse) - O(k^2) per
/// pixel. The set pixels' offsets are listed once, so the inner loop is a
/// plain slice read per offset.
fn naive<T: Copy + Default + Send + Sync, F: Fold<T>, const C: usize>(
    src: &Image<T, C>,
    padded: &Image<T, C>,
    dst: &mut Image<T, C>,
    kernel: &Kernel,
    fold: F,
) -> Result<(), ImageError> {
    let width = src.width();
    let k_width = kernel.width();
    let padded_slice = padded.as_slice();
    let padded_stride = padded.width() * C;
    let src_slice = src.as_slice();

    // Offset of every set kernel pixel from the window's top-left corner.
    let offsets: Vec<usize> = kernel
        .data()
        .iter()
        .enumerate()
        .filter(|&(_, &v)| v == 1)
        .map(|(i, _)| (i / k_width) * padded_stride + (i % k_width) * C)
        .collect();
    let Some((&first, rest)) = offsets.split_first() else {
        // An empty kernel has no window to fold: the image stays as it is.
        dst.as_slice_mut().copy_from_slice(src_slice);
        return Ok(());
    };

    dst.as_slice_mut()
        .par_chunks_mut(width * C)
        .enumerate()
        .for_each(|(h, row_out)| {
            let row_base = h * padded_stride;
            for w in 0..width {
                for c in 0..C {
                    let corner = row_base + w * C + c;
                    let mut val = padded_slice[corner + first];
                    for &offset in rest {
                        val = fold.combine(padded_slice[corner + offset], val);
                    }
                    let idx = w * C + c;
                    if F::USES_CENTER {
                        val = fold.finish(src_slice[h * width * C + idx], val);
                    }
                    row_out[idx] = val;
                }
            }
        });

    Ok(())
}

/// O(k) fast path for a full k*k box structuring element (`Separable::Box`).
///
/// A box is the Minkowski sum of a 1*k row segment and a k*1 column segment,
/// and dilation/erosion distribute over Minkowski sums:
/// `op(f, A ⊕ B) == op(op(f, A), B)`. So instead of scanning a k*k window per
/// pixel (O(k^2) per pixel), this runs a k-wide horizontal pass into a scratch
/// buffer, then a k-tall vertical pass over that buffer (O(k) per pixel,
/// O(2k) total). This holds for every [`Fold`], as each is an associative,
/// commutative combine over the window (the label erosion: "every value of
/// every row segment agrees" == "every value of the box agrees").
fn box_separable<T: Copy + Default + Send + Sync, F: Fold<T>, const C: usize>(
    src: &Image<T, C>,
    padded: &Image<T, C>,
    dst: &mut Image<T, C>,
    kernel: &Kernel,
    fold: F,
) -> Result<(), ImageError> {
    let width = src.width();
    let k_height = kernel.height();
    let k_width = kernel.width();

    let padded_height = padded.height();
    let padded_slice = padded.as_slice();
    let padded_stride = padded.width() * C;
    let src_slice = src.as_slice();

    // Pass 1: horizontal window over every padded row (still full padded
    // height - pass 2 needs to slide a vertical window over these rows).
    let mut horiz = vec![T::default(); width * padded_height * C];
    horiz
        .par_chunks_mut(width * C)
        .enumerate()
        .for_each(|(py, row_out)| {
            let row_in = &padded_slice[py * padded_stride..(py + 1) * padded_stride];
            for c in 0..C {
                for w in 0..width {
                    let mut val = row_in[w * C + c];
                    for kw in 1..k_width {
                        val = fold.combine(row_in[(w + kw) * C + c], val);
                    }
                    row_out[w * C + c] = val;
                }
            }
        });

    // Pass 2: vertical window over pass 1's output.
    dst.as_slice_mut()
        .par_chunks_mut(width * C)
        .enumerate()
        .for_each(|(h, row_out)| {
            for c in 0..C {
                for w in 0..width {
                    let mut val = horiz[h * width * C + w * C + c];
                    for kh in 1..k_height {
                        val = fold.combine(horiz[(h + kh) * width * C + w * C + c], val);
                    }
                    let idx = w * C + c;
                    if F::USES_CENTER {
                        val = fold.finish(src_slice[h * width * C + idx], val);
                    }
                    row_out[idx] = val;
                }
            }
        });

    Ok(())
}

/// O(k) fast path for a plus/cross structuring element (`Separable::Cross`).
///
/// A cross is the *union* of a 1*k row segment and a k*1 column segment
/// through its center, and dilation/erosion distribute over unions:
/// `op(f, A ∪ B) == combine(op(f, A), op(f, B))` - unlike a box, the two 1D
/// passes are independent, not chained, and their results are combined
/// with the same [`Fold`].
fn cross_separable<T: Copy + Default + Send + Sync, F: Fold<T>, const C: usize>(
    src: &Image<T, C>,
    padded: &Image<T, C>,
    dst: &mut Image<T, C>,
    kernel: &Kernel,
    fold: F,
) -> Result<(), ImageError> {
    let width = src.width();
    let height = src.height();
    let (pad_h, pad_w) = kernel.pad();
    let k_height = kernel.height();
    let k_width = kernel.width();

    let padded_slice = padded.as_slice();
    let padded_stride = padded.width() * C;

    // Horizontal-line pass: k_width window centered on each output row.
    let mut horiz = vec![T::default(); width * height * C];
    horiz
        .par_chunks_mut(width * C)
        .enumerate()
        .for_each(|(h, row_out)| {
            let row_in =
                &padded_slice[(h + pad_h) * padded_stride..(h + pad_h + 1) * padded_stride];
            for c in 0..C {
                for w in 0..width {
                    let mut val = row_in[w * C + c];
                    for kw in 1..k_width {
                        val = fold.combine(row_in[(w + kw) * C + c], val);
                    }
                    row_out[w * C + c] = val;
                }
            }
        });

    // Vertical-line pass: k_height window centered on each output column.
    let mut vert = vec![T::default(); width * height * C];
    vert.par_chunks_mut(width * C)
        .enumerate()
        .for_each(|(h, row_out)| {
            for c in 0..C {
                for w in 0..width {
                    let col = w + pad_w;
                    let mut val = padded_slice[h * padded_stride + col * C + c];
                    for kh in 1..k_height {
                        val =
                            fold.combine(padded_slice[(h + kh) * padded_stride + col * C + c], val);
                    }
                    row_out[w * C + c] = val;
                }
            }
        });

    // Combine: cross = union of the two lines.
    let combined = dst
        .as_slice_mut()
        .par_iter_mut()
        .zip(horiz.par_iter())
        .zip(vert.par_iter());
    if F::USES_CENTER {
        let src_slice = src.as_slice();
        combined
            .enumerate()
            .for_each(|(i, ((d, &h), &v))| *d = fold.finish(src_slice[i], fold.combine(h, v)));
    } else {
        combined.for_each(|((d, &h), &v)| *d = fold.combine(h, v));
    }

    Ok(())
}

#[cfg(test)]
mod tests {

    use super::super::kernels::KernelShape;

    use super::*;

    #[test]
    fn test_box_kernel() {
        let kernel = Kernel::new(KernelShape::Box { size: 3 });
        assert_eq!(kernel.width(), 3);
        assert_eq!(kernel.height(), 3);
        assert!(kernel.data().iter().all(|&x| x == 1));
    }

    #[test]
    fn test_cross_kernel() {
        let kernel = Kernel::new(KernelShape::Cross { size: 3 });
        let data = kernel.data();
        // center row
        assert_eq!(data[3], 1);
        assert_eq!(data[4], 1);
        assert_eq!(data[5], 1);
        // center column
        assert_eq!(data[1], 1);
        assert_eq!(data[7], 1);
        // corners
        assert_eq!(data[0], 0);
    }

    #[test]
    fn test_ellipse_kernel() {
        let kernel = Kernel::new(KernelShape::Ellipse {
            width: 5,
            height: 5,
        });
        assert_eq!(kernel.width(), 5);
        assert_eq!(kernel.height(), 5);
        // center
        assert_eq!(kernel.data()[12], 1);
    }

    #[test]
    fn test_kernel_padding() {
        let kernel = Kernel::new(KernelShape::Box { size: 5 });
        let (pad_h, pad_w) = kernel.pad();
        assert_eq!(pad_h, 2);
        assert_eq!(pad_w, 2);
    }

    #[test]
    fn test_dilate_3x3() -> Result<(), ImageError> {
        let size = ImageSize {
            width: 3,
            height: 3,
        };
        let data = vec![0u8, 0, 0, 0, 255, 0, 0, 0, 0];
        let src = Image::new(size, data)?;
        let mut dst = Image::new(size, vec![0u8; 9])?;

        let kernel = Kernel::new(KernelShape::Box { size: 3 });
        dilate(&src, &mut dst, &kernel, PaddingMode::Constant, [0])?;
        let result = dst.as_slice();

        assert!(
            result.iter().all(|&x| x == 255),
            "All pixels should be 255 after dilation"
        );
        Ok(())
    }

    #[test]
    fn test_erode_3x3() -> Result<(), ImageError> {
        let size = ImageSize {
            width: 3,
            height: 3,
        };
        let src = Image::new(size, vec![255u8; 9])?;
        let mut dst = Image::new(size, vec![0u8; 9])?;

        let kernel = Kernel::new(KernelShape::Box { size: 3 });
        erode(&src, &mut dst, &kernel, PaddingMode::Constant, [0])?;
        let result = dst.as_slice();

        assert_eq!(result[4], 255, "Center pixel should survive erosion");
        assert_eq!(
            result[0], 0,
            "Corner pixel should be eroded due to zero padding"
        );
        Ok(())
    }

    #[test]
    fn test_open_remove_noise() -> Result<(), ImageError> {
        let size = ImageSize {
            width: 5,
            height: 5,
        };
        let mut data = vec![0u8; 25];
        data[6] = 255;

        let src = Image::new(size, data)?;
        let mut dst = Image::new(size, vec![0u8; 25])?;

        let kernel = Kernel::new(KernelShape::Box { size: 3 });
        open(&src, &mut dst, &kernel, PaddingMode::Constant, [0])?;
        let result = dst.as_slice();

        assert!(
            result.iter().all(|&x| x == 0),
            "Opening should remove the isolated noise pixel"
        );
        Ok(())
    }

    #[test]
    fn test_close_fill_hole() -> Result<(), ImageError> {
        let size = ImageSize {
            width: 5,
            height: 5,
        };
        let mut data = vec![0u8; 25];
        for y in 1..=3 {
            for x in 1..=3 {
                data[y * 5 + x] = 255;
            }
        }
        data[12] = 0;

        let src = Image::new(size, data)?;
        let mut dst = Image::new(size, vec![0u8; 25])?;

        let kernel = Kernel::new(KernelShape::Box { size: 3 });
        close(&src, &mut dst, &kernel, PaddingMode::Constant, [0])?;
        let result = dst.as_slice();

        assert_eq!(
            result[12], 255,
            "Closing should fill the hole in the center"
        );
        assert_eq!(result[6], 255);
        assert_eq!(result[18], 255);
        Ok(())
    }

    /// Same mask as `kernel`, but tagged `Separable::None` so `dilate`/`erode`
    /// fall through to the O(k^2) naive scan instead of the fast path -
    /// giving a reference to cross-validate the fast path against.
    fn force_naive(kernel: &Kernel) -> Kernel {
        Kernel::new_test_with_separable(
            kernel.data().to_vec(),
            kernel.width(),
            kernel.height(),
            Separable::None,
        )
    }

    /// Deterministic (not random, for reproducibility) pseudo-noisy u8 pattern.
    fn pattern_u8(width: usize, height: usize) -> Vec<u8> {
        (0..width * height)
            .map(|i| ((i * 37 + 11) % 256) as u8)
            .collect()
    }

    #[test]
    fn box_and_cross_fast_paths_match_the_naive_scan_across_sizes_and_padding_modes() {
        let sizes = [(7usize, 7usize), (10, 6), (13, 13), (1, 1), (4, 9)];
        let kernel_sizes = [1usize, 3, 5, 7];
        let padding_modes = [
            PaddingMode::Constant,
            PaddingMode::Replicate,
            PaddingMode::Reflect101,
            PaddingMode::Reflect,
            PaddingMode::Wrap,
        ];

        for &(width, height) in &sizes {
            let size = ImageSize { width, height };
            let data = pattern_u8(width, height);
            let src = Image::<u8, 1>::new(size, data).unwrap();

            for &k in &kernel_sizes {
                // Kernel must fit inside the (padded) image; skip combinations
                // spatial_padding can't service.
                if k > width.max(height) * 2 + 1 {
                    continue;
                }
                for shape in [KernelShape::Box { size: k }, KernelShape::Cross { size: k }] {
                    let kernel = Kernel::new(shape.clone());
                    let naive_kernel = force_naive(&kernel);
                    let shape_name = format!("{shape:?}");

                    for &padding_mode in &padding_modes {
                        let blank =
                            || Image::<u8, 1>::new(size, vec![0u8; width * height]).unwrap();
                        let ctx = format!(
                            "{shape_name} k={k} size={width}x{height} padding={padding_mode:?}"
                        );

                        let mut fast = blank();
                        let mut naive = blank();
                        dilate(&src, &mut fast, &kernel, padding_mode, [0]).unwrap();
                        dilate(&src, &mut naive, &naive_kernel, padding_mode, [0]).unwrap();
                        assert_eq!(
                            fast.as_slice(),
                            naive.as_slice(),
                            "dilate mismatch for {ctx}"
                        );

                        let mut fast = blank();
                        let mut naive = blank();
                        erode(&src, &mut fast, &kernel, padding_mode, [0]).unwrap();
                        erode(&src, &mut naive, &naive_kernel, padding_mode, [0]).unwrap();
                        assert_eq!(
                            fast.as_slice(),
                            naive.as_slice(),
                            "erode mismatch for {ctx}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn box_and_cross_fast_paths_match_the_naive_scan_for_multichannel_rgb() {
        let size = ImageSize {
            width: 9,
            height: 6,
        };
        let data: Vec<f32> = (0..size.width * size.height * 3)
            .map(|i| ((i * 53 + 7) % 101) as f32 / 100.0)
            .collect();
        let src = Image::<f32, 3>::new(size, data).unwrap();

        for shape in [KernelShape::Box { size: 5 }, KernelShape::Cross { size: 5 }] {
            let kernel = Kernel::new(shape.clone());
            let naive_kernel = force_naive(&kernel);
            let blank =
                || Image::<f32, 3>::new(size, vec![0.0; size.width * size.height * 3]).unwrap();

            let mut fast = blank();
            let mut naive = blank();
            dilate(&src, &mut fast, &kernel, PaddingMode::Reflect101, [0.0; 3]).unwrap();
            dilate(
                &src,
                &mut naive,
                &naive_kernel,
                PaddingMode::Reflect101,
                [0.0; 3],
            )
            .unwrap();
            assert_eq!(
                fast.as_slice(),
                naive.as_slice(),
                "dilate mismatch for {shape:?} on RGB data"
            );

            let mut fast = blank();
            let mut naive = blank();
            erode(&src, &mut fast, &kernel, PaddingMode::Reflect101, [0.0; 3]).unwrap();
            erode(
                &src,
                &mut naive,
                &naive_kernel,
                PaddingMode::Reflect101,
                [0.0; 3],
            )
            .unwrap();
            assert_eq!(
                fast.as_slice(),
                naive.as_slice(),
                "erode mismatch for {shape:?} on RGB data"
            );
        }
    }

    #[test]
    fn open_and_close_still_correct_through_the_fast_path_with_a_cross_kernel()
    -> Result<(), ImageError> {
        let size = ImageSize {
            width: 7,
            height: 7,
        };
        let mut data = vec![0u8; 49];
        data[3 * 7 + 3] = 255; // isolated single-pixel noise at the center

        let src = Image::new(size, data)?;
        let mut dst = Image::new(size, vec![0u8; 49])?;
        let kernel = Kernel::new(KernelShape::Cross { size: 3 });
        open(&src, &mut dst, &kernel, PaddingMode::Constant, [0])?;
        assert!(
            dst.as_slice().iter().all(|&x| x == 0),
            "Opening with a Cross kernel should remove the isolated noise pixel"
        );

        let mut data = vec![0u8; 49];
        for y in 2..=4 {
            for x in 2..=4 {
                data[y * 7 + x] = 255;
            }
        }
        data[3 * 7 + 3] = 0; // hole in the center of a filled square

        let src = Image::new(size, data)?;
        let mut dst = Image::new(size, vec![0u8; 49])?;
        close(&src, &mut dst, &kernel, PaddingMode::Constant, [0])?;
        assert_eq!(
            dst.as_slice()[3 * 7 + 3],
            255,
            "Closing with a Cross kernel should fill the center hole"
        );
        Ok(())
    }

    // --- Kernel symmetry (ellipse centre was half a pixel off) -----------

    fn kernel_rows(kernel: &Kernel) -> Vec<String> {
        kernel
            .data()
            .chunks(kernel.width())
            .map(|r| r.iter().map(|&v| if v == 1 { '#' } else { '.' }).collect())
            .collect()
    }

    #[test]
    fn every_kernel_shape_is_symmetric_and_contains_its_centre_for_odd_sizes() {
        for n in (1..=21).step_by(2) {
            for shape in [
                KernelShape::Box { size: n },
                KernelShape::Cross { size: n },
                KernelShape::Ellipse {
                    width: n,
                    height: n,
                },
            ] {
                let kernel = Kernel::new(shape.clone());
                let d = kernel.data();
                let at = |y: usize, x: usize| d[y * n + x];
                assert_eq!(at(n / 2, n / 2), 1, "{shape:?}: centre not set");
                for y in 0..n {
                    for x in 0..n {
                        assert_eq!(
                            at(y, x),
                            at(y, n - 1 - x),
                            "{shape:?}: not mirror symmetric"
                        );
                        assert_eq!(at(y, x), at(n - 1 - y, x), "{shape:?}: not flip symmetric");
                        assert_eq!(at(y, x), at(x, y), "{shape:?}: not transpose symmetric");
                    }
                }
            }
        }
    }

    #[test]
    fn ellipse_kernels_are_centred_disks() {
        let rows = |n| {
            kernel_rows(&Kernel::new(KernelShape::Ellipse {
                width: n,
                height: n,
            }))
        };
        assert_eq!(rows(3), ["###", "###", "###"]);
        assert_eq!(rows(5), [".###.", "#####", "#####", "#####", ".###."]);
        assert_eq!(
            rows(7),
            [
                "..###..", ".#####.", "#######", "#######", "#######", ".#####.", "..###.."
            ]
        );
    }

    #[test]
    fn ellipse_dilation_of_a_single_pixel_is_the_kernel_centred_on_it() -> Result<(), ImageError> {
        let size = ImageSize {
            width: 9,
            height: 9,
        };
        let mut src = Image::<u8, 1>::from_size_val(size, 0)?;
        src.set_pixel(4, 4, 0, 1)?;
        let mut dst = Image::<u8, 1>::from_size_val(size, 0)?;
        let kernel = Kernel::new(KernelShape::Ellipse {
            width: 5,
            height: 5,
        });
        dilate(&src, &mut dst, &kernel, PaddingMode::Constant, [0])?;
        let k = kernel.data();
        for y in 0..9 {
            for x in 0..9 {
                let inside = (2..=6).contains(&x) && (2..=6).contains(&y);
                let expected = if inside { k[(y - 2) * 5 + (x - 2)] } else { 0 };
                assert_eq!(*dst.get_pixel(x, y, 0)?, expected, "pixel ({x}, {y})");
            }
        }
        Ok(())
    }

    // --- Label maps: every class as its own mask -------------------------

    #[derive(Clone, Copy, Debug)]
    enum Op {
        Dilate,
        Erode,
        Open,
        Close,
    }
    const OPS: [Op; 4] = [Op::Dilate, Op::Erode, Op::Open, Op::Close];

    fn labels_op(op: Op, src: &Image<u32, 1>, kernel: &Kernel) -> Image<u32, 1> {
        let mut dst = Image::<u32, 1>::from_size_val(src.size(), 0).unwrap();
        let (m, p) = (PaddingMode::Constant, [0u32]);
        match op {
            Op::Dilate => dilate_labels(src, &mut dst, kernel, m, p),
            Op::Erode => erode_labels(src, &mut dst, kernel, m, p),
            Op::Open => open_labels(src, &mut dst, kernel, m, p),
            Op::Close => close_labels(src, &mut dst, kernel, m, p),
        }
        .unwrap();
        dst
    }

    fn gray_op<T: Copy + Default + Send + Sync + PartialOrd>(
        op: Op,
        src: &Image<T, 1>,
        kernel: &Kernel,
    ) -> Image<T, 1> {
        let mut dst = Image::<T, 1>::from_size_val(src.size(), T::default()).unwrap();
        let (m, p) = (PaddingMode::Constant, [T::default()]);
        match op {
            Op::Dilate => dilate(src, &mut dst, kernel, m, p),
            Op::Erode => erode(src, &mut dst, kernel, m, p),
            Op::Open => open(src, &mut dst, kernel, m, p),
            Op::Close => close(src, &mut dst, kernel, m, p),
        }
        .unwrap();
        dst
    }

    /// The definition the label operations must meet: every class run as a
    /// binary mask on its own; a pixel keeps its own class if that class's
    /// result still covers it, otherwise it takes the highest class covering
    /// it, otherwise background.
    fn per_class_reference(op: Op, src: &Image<u32, 1>, kernel: &Kernel) -> Vec<u32> {
        let s = src.as_slice();
        let mut classes: Vec<u32> = s.iter().copied().filter(|&v| v != 0).collect();
        classes.sort_unstable();
        classes.dedup();
        let results: Vec<(u32, Vec<u8>)> = classes
            .iter()
            .map(|&c| {
                let mask: Vec<u8> = s.iter().map(|&v| (v == c) as u8).collect();
                let mask = Image::<u8, 1>::new(src.size(), mask).unwrap();
                (c, gray_op(op, &mask, kernel).as_slice().to_vec())
            })
            .collect();
        (0..s.len())
            .map(|i| {
                let covering = results.iter().filter(|(_, r)| r[i] != 0).map(|(c, _)| *c);
                if s[i] != 0 && results.iter().any(|(c, r)| *c == s[i] && r[i] != 0) {
                    s[i]
                } else {
                    covering.max().unwrap_or(0)
                }
            })
            .collect()
    }

    /// Deterministic blobs of up to `classes` classes (0 = background),
    /// touching each other in places.
    fn label_pattern(width: usize, height: usize, classes: u32, seed: u64) -> Image<u32, 1> {
        let mut state = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let mut next = move || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (state >> 33) as usize
        };
        let mut data = vec![0u32; width * height];
        for _ in 0..(width * height / 12).max(3) {
            let (cx, cy) = (next() % width, next() % height);
            let (rw, rh) = (1 + next() % 4, 1 + next() % 4);
            let class = 1 + (next() % classes as usize) as u32;
            for y in cy.saturating_sub(rh)..(cy + rh).min(height) {
                for x in cx.saturating_sub(rw)..(cx + rw).min(width) {
                    data[y * width + x] = class;
                }
            }
        }
        Image::new(ImageSize { width, height }, data).unwrap()
    }

    fn all_kernels() -> Vec<Kernel> {
        let mut kernels = Vec::new();
        for n in [1, 3, 5, 7] {
            kernels.push(Kernel::new(KernelShape::Box { size: n }));
            kernels.push(Kernel::new(KernelShape::Cross { size: n }));
            kernels.push(Kernel::new(KernelShape::Ellipse {
                width: n,
                height: n,
            }));
        }
        kernels
    }

    #[test]
    fn label_operations_equal_every_class_transformed_as_its_own_mask() {
        for (seed, classes) in [(1u64, 2u32), (2, 3), (3, 5), (4, 1)] {
            let src = label_pattern(23, 17, classes, seed);
            for kernel in all_kernels() {
                for op in OPS {
                    assert_eq!(
                        labels_op(op, &src, &kernel).as_slice(),
                        per_class_reference(op, &src, &kernel).as_slice(),
                        "{op:?}, kernel {:?}, seed {seed}",
                        kernel_rows(&kernel)
                    );
                }
            }
        }
    }

    #[test]
    fn label_operations_with_one_class_equal_the_grayscale_operations() {
        let src = label_pattern(23, 17, 1, 7);
        for kernel in all_kernels() {
            for op in OPS {
                assert_eq!(
                    labels_op(op, &src, &kernel).as_slice(),
                    gray_op(op, &src, &kernel).as_slice(),
                    "{op:?}, kernel {:?}",
                    kernel_rows(&kernel)
                );
            }
        }
    }

    #[test]
    fn label_operations_do_not_depend_on_the_class_numbers_of_touching_classes() {
        // Two touching blocks: class 1 left, class 2 right (and swapped).
        let size = ImageSize {
            width: 12,
            height: 7,
        };
        let block = |left: u32, right: u32| {
            let mut data = vec![0u32; 12 * 7];
            for y in 1..6 {
                for x in 1..11 {
                    data[y * 12 + x] = if x < 6 { left } else { right };
                }
            }
            Image::new(size, data).unwrap()
        };
        let kernel = Kernel::new(KernelShape::Box { size: 3 });
        let swap = |v: u32| match v {
            1 => 2,
            2 => 1,
            v => v,
        };
        for op in [Op::Erode, Op::Open, Op::Close] {
            let a = labels_op(op, &block(1, 2), &kernel);
            let b = labels_op(op, &block(2, 1), &kernel);
            let b_swapped: Vec<u32> = b.as_slice().iter().map(|&v| swap(v)).collect();
            assert_eq!(a.as_slice(), b_swapped.as_slice(), "{op:?}");
        }
        // Erode: both sides lose their column at the shared border.
        let eroded = labels_op(Op::Erode, &block(1, 2), &kernel);
        let row: Vec<u32> = eroded.as_slice()[3 * 12..4 * 12].to_vec();
        assert_eq!(row, [0, 0, 1, 1, 1, 0, 0, 2, 2, 2, 0, 0]);
        // Dilate never overwrites a class; the background grows.
        let dilated = labels_op(Op::Dilate, &block(1, 2), &kernel);
        let row: Vec<u32> = dilated.as_slice()[3 * 12..4 * 12].to_vec();
        assert_eq!(row, [1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2]);
        // Close keeps every labelled pixel (no gap opened between the classes).
        let closed = labels_op(Op::Close, &block(1, 2), &kernel);
        assert_eq!(closed.as_slice(), block(1, 2).as_slice());
    }

    #[test]
    fn label_dilation_gives_a_background_pixel_reached_by_two_classes_the_higher_one() {
        let size = ImageSize {
            width: 5,
            height: 1,
        };
        // x = 1 is reached by both classes, x = 3 by class 3 only, x = 4 by none.
        let src = Image::new(size, vec![1u32, 0, 3, 0, 0]).unwrap();
        let kernel = Kernel::new(KernelShape::Box { size: 3 });
        let out = labels_op(Op::Dilate, &src, &kernel);
        assert_eq!(out.as_slice(), [1, 3, 3, 3, 0]);
    }

    #[test]
    fn label_fast_paths_match_the_naive_scan() {
        let src = label_pattern(19, 13, 4, 11);
        for n in [1, 3, 5, 7] {
            for kernel in [
                Kernel::new(KernelShape::Box { size: n }),
                Kernel::new(KernelShape::Cross { size: n }),
            ] {
                let naive_kernel = force_naive(&kernel);
                for op in OPS {
                    assert_eq!(
                        labels_op(op, &src, &kernel).as_slice(),
                        labels_op(op, &src, &naive_kernel).as_slice(),
                        "{op:?}, kernel {:?}",
                        kernel_rows(&kernel)
                    );
                }
            }
        }
    }

    #[test]
    fn has_several_classes_finds_a_second_class_anywhere() {
        assert!(!has_several_classes(&[]));
        assert!(!has_several_classes(&[0, 0, 0]));
        assert!(!has_several_classes(&[0, 4, 0, 4, 4]));
        assert!(has_several_classes(&[0, 4, 0, 5]));
        // A second class only far past the first chunk.
        let mut big = vec![0u32; 300_000];
        big[10] = 1;
        big[100_000] = 1;
        assert!(!has_several_classes(&big));
        big[299_999] = 2;
        assert!(has_several_classes(&big));
    }

    #[test]
    fn transform_labels_equals_every_class_transformed_as_its_own_mask() {
        use super::super::MorphOps;
        let ops = [
            (MorphOps::Dilate, Op::Dilate),
            (MorphOps::Erode, Op::Erode),
            (MorphOps::Open, Op::Open),
            (MorphOps::Close, Op::Close),
        ];
        // One class, several classes, a second class only in the last row.
        let mut late = label_pattern(23, 17, 1, 9);
        late.as_slice_mut()[23 * 17 - 1] = 7;
        let maps = [
            label_pattern(23, 17, 1, 5),
            label_pattern(23, 17, 3, 6),
            late,
            Image::new(
                ImageSize {
                    width: 4,
                    height: 3,
                },
                vec![0u32; 12],
            )
            .unwrap(),
        ];
        for src in &maps {
            for kernel in all_kernels() {
                for (morph_op, op) in ops {
                    let mut dst = Image::<u32, 1>::from_size_val(src.size(), 0).unwrap();
                    transform_labels(morph_op, src, &mut dst, &kernel, PaddingMode::Constant, [0])
                        .unwrap();
                    assert_eq!(
                        dst.as_slice(),
                        per_class_reference(op, src, &kernel).as_slice(),
                        "{op:?}, kernel {:?}",
                        kernel_rows(&kernel)
                    );
                }
            }
        }
    }

    #[test]
    fn constant_padding_builds_the_same_buffer_as_kornia() {
        fn kornia<T: Copy + Default + Send + Sync, const C: usize>(
            src: &Image<T, C>,
            kernel: &Kernel,
            constant: [T; C],
        ) -> Vec<T> {
            let (pad_h, pad_w) = kernel.pad();
            let size = ImageSize {
                width: src.width() + 2 * pad_w,
                height: src.height() + 2 * pad_h,
            };
            let mut padded =
                Image::new(size, vec![T::default(); size.width * size.height * C]).unwrap();
            let padding = Padding2D {
                top: pad_h,
                bottom: pad_h,
                left: pad_w,
                right: pad_w,
            };
            spatial_padding(src, &mut padded, padding, PaddingMode::Constant, constant).unwrap();
            padded.as_slice().to_vec()
        }
        for (w, h) in [(1usize, 1usize), (7, 5), (16, 3)] {
            for n in [1, 3, 7] {
                let kernel = Kernel::new(KernelShape::Box { size: n });
                let gray = Image::<u8, 1>::new(
                    ImageSize {
                        width: w,
                        height: h,
                    },
                    pattern_u8(w, h),
                )
                .unwrap();
                for constant in [[0u8], [9u8]] {
                    let ours =
                        padded_copy(&gray, &kernel, PaddingMode::Constant, constant).unwrap();
                    assert_eq!(ours.as_slice(), kornia(&gray, &kernel, constant).as_slice());
                }
                let rgb_data: Vec<f32> = pattern_u8(w * 3, h).iter().map(|&v| v as f32).collect();
                let rgb = Image::<f32, 3>::new(
                    ImageSize {
                        width: w,
                        height: h,
                    },
                    rgb_data,
                )
                .unwrap();
                for constant in [[0.0f32; 3], [1.0, 2.0, 3.0]] {
                    let ours = padded_copy(&rgb, &kernel, PaddingMode::Constant, constant).unwrap();
                    assert_eq!(ours.as_slice(), kornia(&rgb, &kernel, constant).as_slice());
                }
            }
        }
    }
}
