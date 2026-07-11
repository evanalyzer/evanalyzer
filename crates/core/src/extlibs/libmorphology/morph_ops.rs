use crate::extlibs::libmorphology::{Kernel, Separable};
use kornia_image::{Image, ImageError, ImageSize, allocator::ImageAllocator};
use kornia_imgproc::padding::{Padding2D, PaddingMode, spatial_padding};
use kornia_tensor::CpuAllocator;
use rayon::prelude::*;

//use kornia_image::{Image, ImageError, ImageSize, allocator::ImageAllocator};
//use kornia_tensor::CpuAllocator;
//use rayon::prelude::*;

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
pub fn dilate<
    T: Copy + Default + Send + Sync + PartialOrd,
    const C: usize,
    A1: ImageAllocator,
    A2: ImageAllocator,
>(
    src: &Image<T, C, A1>,
    dst: &mut Image<T, C, A2>,
    kernel: &Kernel,
    padding_mode: PaddingMode,
    constant_value: [T; C],
) -> Result<(), ImageError> {
    if src.size() != dst.size() {
        return Err(ImageError::InvalidImageSize(
            dst.width(),
            dst.height(),
            src.width(),
            src.height(),
        ));
    }

    match kernel.separable() {
        Separable::Box => {
            return box_separable(src, dst, kernel, padding_mode, constant_value, |a, b| a > b);
        }
        Separable::Cross => {
            return cross_separable(src, dst, kernel, padding_mode, constant_value, |a, b| a > b);
        }
        Separable::None => {}
    }

    let width = src.width();
    let height = src.height();
    let (pad_h, pad_w) = kernel.pad();
    let k_height = kernel.height();
    let k_width = kernel.width();
    let k_data = kernel.data();

    let padded_size = ImageSize {
        width: width + 2 * pad_w,
        height: height + 2 * pad_h,
    };
    let padded_buffer = vec![T::default(); padded_size.width * padded_size.height * C];
    let mut padded = Image::new(padded_size, padded_buffer, CpuAllocator)?;

    let padding = Padding2D {
        top: pad_h,
        bottom: pad_h,
        left: pad_w,
        right: pad_w,
    };
    spatial_padding(src, &mut padded, padding, padding_mode, constant_value)?;

    // dilation
    let dst_slice = dst.as_slice_mut();
    let dst_chunks: Vec<_> = dst_slice.chunks_mut(width * C).collect();

    dst_chunks
        .into_par_iter()
        .enumerate()
        .for_each(|(h, row_chunk)| {
            for c in 0..C {
                for w in 0..width {
                    let mut max_val = T::default();

                    for kh in 0..k_height {
                        for kw in 0..k_width {
                            if k_data[kh * k_width + kw] == 1 {
                                let px = w + kw;
                                let py = h + kh;
                                if let Ok(pixel) = padded.get_pixel(px, py, c) {
                                    max_val = if *pixel > max_val { *pixel } else { max_val };
                                }
                            }
                        }
                    }

                    let idx = w * C + c;
                    row_chunk[idx] = max_val;
                }
            }
        });

    Ok(())
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
pub fn erode<
    T: Copy + Default + Send + Sync + PartialOrd,
    const C: usize,
    A1: ImageAllocator,
    A2: ImageAllocator,
>(
    src: &Image<T, C, A1>,
    dst: &mut Image<T, C, A2>,
    kernel: &Kernel,
    padding_mode: PaddingMode,
    constant_value: [T; C],
) -> Result<(), ImageError> {
    if src.size() != dst.size() {
        return Err(ImageError::InvalidImageSize(
            dst.width(),
            dst.height(),
            src.width(),
            src.height(),
        ));
    }

    match kernel.separable() {
        Separable::Box => {
            return box_separable(src, dst, kernel, padding_mode, constant_value, |a, b| a < b);
        }
        Separable::Cross => {
            return cross_separable(src, dst, kernel, padding_mode, constant_value, |a, b| a < b);
        }
        Separable::None => {}
    }

    let width = src.width();
    let height = src.height();
    let (pad_h, pad_w) = kernel.pad();
    let k_height = kernel.height();
    let k_width = kernel.width();
    let k_data = kernel.data();

    let padded_size = ImageSize {
        width: width + 2 * pad_w,
        height: height + 2 * pad_h,
    };
    let padded_buffer = vec![T::default(); padded_size.width * padded_size.height * C];
    let mut padded = Image::new(padded_size, padded_buffer, CpuAllocator)?;

    let padding = Padding2D {
        top: pad_h,
        bottom: pad_h,
        left: pad_w,
        right: pad_w,
    };
    spatial_padding(src, &mut padded, padding, padding_mode, constant_value)?;

    // erosion
    let dst_slice = dst.as_slice_mut();
    let dst_chunks: Vec<_> = dst_slice.chunks_mut(width * C).collect();

    dst_chunks
        .into_par_iter()
        .enumerate()
        .for_each(|(h, row_chunk)| {
            for c in 0..C {
                for w in 0..width {
                    let mut min_val: Option<T> = None;

                    for kh in 0..k_height {
                        for kw in 0..k_width {
                            if k_data[kh * k_width + kw] == 1 {
                                let px = w + kw;
                                let py = h + kh;
                                if let Ok(pixel) = padded.get_pixel(px, py, c) {
                                    min_val = Some(match min_val {
                                        None => *pixel,
                                        Some(v) => match v.partial_cmp(pixel) {
                                            Some(std::cmp::Ordering::Greater) => *pixel, // Pixel is smaller, take it
                                            _ => v, // v is smaller, or they are incomparable (NaN)
                                        },
                                    });
                                }
                            }
                        }
                    }

                    let idx = w * C + c;
                    row_chunk[idx] = min_val.unwrap_or_default();
                }
            }
        });

    Ok(())
}

/// O(k) fast path for a full k*k box structuring element (`Separable::Box`).
///
/// A box is the Minkowski sum of a 1*k row segment and a k*1 column segment,
/// and grayscale dilation/erosion distribute over Minkowski sums:
/// `op(f, A ⊕ B) == op(op(f, A), B)`. So instead of scanning a k*k window per
/// pixel (O(k^2) per pixel), this runs a k-wide horizontal pass into a scratch
/// buffer, then a k-tall vertical pass over that buffer (O(k) per pixel,
/// O(2k) total). `better(candidate, current)` picks the window's extremum:
/// `a > b` for dilate, `a < b` for erode.
fn box_separable<
    T: Copy + Default + Send + Sync + PartialOrd,
    const C: usize,
    A1: ImageAllocator,
    A2: ImageAllocator,
>(
    src: &Image<T, C, A1>,
    dst: &mut Image<T, C, A2>,
    kernel: &Kernel,
    padding_mode: PaddingMode,
    constant_value: [T; C],
    better: impl Fn(T, T) -> bool + Sync + Send + Copy,
) -> Result<(), ImageError> {
    if src.size() != dst.size() {
        return Err(ImageError::InvalidImageSize(
            dst.width(),
            dst.height(),
            src.width(),
            src.height(),
        ));
    }

    let width = src.width();
    let height = src.height();
    let (pad_h, pad_w) = kernel.pad();
    let k_height = kernel.height();
    let k_width = kernel.width();

    let padded_size = ImageSize {
        width: width + 2 * pad_w,
        height: height + 2 * pad_h,
    };
    let padded_buffer = vec![T::default(); padded_size.width * padded_size.height * C];
    let mut padded = Image::new(padded_size, padded_buffer, CpuAllocator)?;
    let padding = Padding2D {
        top: pad_h,
        bottom: pad_h,
        left: pad_w,
        right: pad_w,
    };
    spatial_padding(src, &mut padded, padding, padding_mode, constant_value)?;
    let padded_slice = padded.as_slice();
    let padded_stride = padded_size.width * C;

    // Pass 1: horizontal window over every padded row (still full padded
    // height - pass 2 needs to slide a vertical window over these rows).
    let mut horiz = vec![T::default(); width * padded_size.height * C];
    horiz
        .par_chunks_mut(width * C)
        .enumerate()
        .for_each(|(py, row_out)| {
            let row_in = &padded_slice[py * padded_stride..(py + 1) * padded_stride];
            for c in 0..C {
                for w in 0..width {
                    let mut val = row_in[w * C + c];
                    for kw in 1..k_width {
                        let candidate = row_in[(w + kw) * C + c];
                        if better(candidate, val) {
                            val = candidate;
                        }
                    }
                    row_out[w * C + c] = val;
                }
            }
        });

    // Pass 2: vertical window over pass 1's output.
    let dst_slice = dst.as_slice_mut();
    let dst_chunks: Vec<_> = dst_slice.chunks_mut(width * C).collect();
    dst_chunks
        .into_par_iter()
        .enumerate()
        .for_each(|(h, row_out)| {
            for c in 0..C {
                for w in 0..width {
                    let mut val = horiz[h * width * C + w * C + c];
                    for kh in 1..k_height {
                        let candidate = horiz[(h + kh) * width * C + w * C + c];
                        if better(candidate, val) {
                            val = candidate;
                        }
                    }
                    row_out[w * C + c] = val;
                }
            }
        });

    Ok(())
}

/// O(k) fast path for a plus/cross structuring element (`Separable::Cross`).
///
/// A cross is the *union* of a 1*k row segment and a k*1 column segment
/// through its center, and grayscale dilation/erosion distribute over unions:
/// `op(f, A ∪ B) == combine(op(f, A), op(f, B))` (max for dilate, min for
/// erode) - unlike a box, the two 1D passes are independent, not chained.
/// `better(candidate, current)` picks the window's extremum, matching the
/// combine direction: `a > b` for dilate, `a < b` for erode.
fn cross_separable<
    T: Copy + Default + Send + Sync + PartialOrd,
    const C: usize,
    A1: ImageAllocator,
    A2: ImageAllocator,
>(
    src: &Image<T, C, A1>,
    dst: &mut Image<T, C, A2>,
    kernel: &Kernel,
    padding_mode: PaddingMode,
    constant_value: [T; C],
    better: impl Fn(T, T) -> bool + Sync + Send + Copy,
) -> Result<(), ImageError> {
    if src.size() != dst.size() {
        return Err(ImageError::InvalidImageSize(
            dst.width(),
            dst.height(),
            src.width(),
            src.height(),
        ));
    }

    let width = src.width();
    let height = src.height();
    let (pad_h, pad_w) = kernel.pad();
    let k_height = kernel.height();
    let k_width = kernel.width();

    let padded_size = ImageSize {
        width: width + 2 * pad_w,
        height: height + 2 * pad_h,
    };
    let padded_buffer = vec![T::default(); padded_size.width * padded_size.height * C];
    let mut padded = Image::new(padded_size, padded_buffer, CpuAllocator)?;
    let padding = Padding2D {
        top: pad_h,
        bottom: pad_h,
        left: pad_w,
        right: pad_w,
    };
    spatial_padding(src, &mut padded, padding, padding_mode, constant_value)?;
    let padded_slice = padded.as_slice();
    let padded_stride = padded_size.width * C;

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
                        let candidate = row_in[(w + kw) * C + c];
                        if better(candidate, val) {
                            val = candidate;
                        }
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
                        let candidate = padded_slice[(h + kh) * padded_stride + col * C + c];
                        if better(candidate, val) {
                            val = candidate;
                        }
                    }
                    row_out[w * C + c] = val;
                }
            }
        });

    // Combine: cross = union of the two lines, so the result is the extremum
    // of the two independent passes.
    let dst_slice = dst.as_slice_mut();
    dst_slice
        .par_iter_mut()
        .zip(horiz.par_iter())
        .zip(vert.par_iter())
        .for_each(|((d, &h), &v)| {
            *d = if better(h, v) { h } else { v };
        });

    Ok(())
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
pub fn open<
    T: Copy + Default + Send + Sync + PartialOrd,
    const C: usize,
    A1: ImageAllocator,
    A2: ImageAllocator,
>(
    src: &Image<T, C, A1>,
    dst: &mut Image<T, C, A2>,
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
pub fn close<
    T: Copy + Default + Send + Sync + PartialOrd,
    const C: usize,
    A1: ImageAllocator,
    A2: ImageAllocator,
>(
    src: &Image<T, C, A1>,
    dst: &mut Image<T, C, A2>,
    kernel: &Kernel,
    padding_mode: PaddingMode,
    constant_value: [T; C],
) -> Result<(), ImageError> {
    let mut temp_img = src.clone();
    dilate(src, &mut temp_img, kernel, padding_mode, constant_value)?;
    erode(&temp_img, dst, kernel, padding_mode, constant_value)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use kornia_tensor::CpuAllocator;

    use crate::extlibs::libmorphology::KernelShape;

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
        let src = Image::new(size, data, CpuAllocator)?;
        let mut dst = Image::new(size, vec![0u8; 9], CpuAllocator)?;

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
        let src = Image::new(size, vec![255u8; 9], CpuAllocator)?;
        let mut dst = Image::new(size, vec![0u8; 9], CpuAllocator)?;

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

        let src = Image::new(size, data, CpuAllocator)?;
        let mut dst = Image::new(size, vec![0u8; 25], CpuAllocator)?;

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

        let src = Image::new(size, data, CpuAllocator)?;
        let mut dst = Image::new(size, vec![0u8; 25], CpuAllocator)?;

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
            let src = Image::<u8, 1, CpuAllocator>::new(size, data, CpuAllocator).unwrap();

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
                        let blank = || {
                            Image::<u8, 1, CpuAllocator>::new(
                                size,
                                vec![0u8; width * height],
                                CpuAllocator,
                            )
                            .unwrap()
                        };
                        let ctx = format!(
                            "{shape_name} k={k} size={width}x{height} padding={padding_mode:?}"
                        );

                        let mut fast = blank();
                        let mut naive = blank();
                        dilate(&src, &mut fast, &kernel, padding_mode, [0]).unwrap();
                        dilate(&src, &mut naive, &naive_kernel, padding_mode, [0]).unwrap();
                        assert_eq!(fast.as_slice(), naive.as_slice(), "dilate mismatch for {ctx}");

                        let mut fast = blank();
                        let mut naive = blank();
                        erode(&src, &mut fast, &kernel, padding_mode, [0]).unwrap();
                        erode(&src, &mut naive, &naive_kernel, padding_mode, [0]).unwrap();
                        assert_eq!(fast.as_slice(), naive.as_slice(), "erode mismatch for {ctx}");
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
        let src = Image::<f32, 3, CpuAllocator>::new(size, data, CpuAllocator).unwrap();

        for shape in [KernelShape::Box { size: 5 }, KernelShape::Cross { size: 5 }] {
            let kernel = Kernel::new(shape.clone());
            let naive_kernel = force_naive(&kernel);
            let blank = || {
                Image::<f32, 3, CpuAllocator>::new(
                    size,
                    vec![0.0; size.width * size.height * 3],
                    CpuAllocator,
                )
                .unwrap()
            };

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
    fn open_and_close_still_correct_through_the_fast_path_with_a_cross_kernel() -> Result<(), ImageError>
    {
        let size = ImageSize {
            width: 7,
            height: 7,
        };
        let mut data = vec![0u8; 49];
        data[3 * 7 + 3] = 255; // isolated single-pixel noise at the center

        let src = Image::new(size, data, CpuAllocator)?;
        let mut dst = Image::new(size, vec![0u8; 49], CpuAllocator)?;
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

        let src = Image::new(size, data, CpuAllocator)?;
        let mut dst = Image::new(size, vec![0u8; 49], CpuAllocator)?;
        close(&src, &mut dst, &kernel, PaddingMode::Constant, [0])?;
        assert_eq!(
            dst.as_slice()[3 * 7 + 3],
            255,
            "Closing with a Cross kernel should fill the center hole"
        );
        Ok(())
    }
}
