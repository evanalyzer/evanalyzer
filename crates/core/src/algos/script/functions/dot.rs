//! # dot
//!
//! `dot(a, b)`: the mean over the pixels of the pixel-wise product
//! `a(x) · b(x)` - for fitting how much of one image is in another, e.g. the
//! scale `a` of a structure channel `T` in a spot channel `S` (least
//! squares: `a = <S, T> / <T, T>`):
//!
//! ```rhai
//! let a = Sf.dot(T) / T.dot(T);
//! let v = dot(Sf, T);                          // the same as Sf.dot(T)
//! let v = Sf.dot(T, #{ margin: 4 });           // without a 4 px border
//! let v = Sf.dot(T, #{ margin: 4, mask: M });  // only where M > 0
//! ```
//!
//! Options:
//! - `margin` (default 0): leaves out a border this many pixels wide on
//!   each side - of every plane of a volume - e.g. the strip a shift
//!   (Translate Image) fills with padding.
//! - `mask`: counts only the pixels where the mask is above 0; a grey or a
//!   label image of the same size.
//!
//! Grey and colour images (a colour image: all its colour values), not
//! label images. Sums each product in `f64` (the product of two `f32` is
//! exact there), the rows as a compensated (Neumaier) sum - a tile has
//! millions of pixels. Allocates nothing.
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-11-17
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.

use crate::algos::script::script_bridge::ScriptImage;
use crate::image::ImageContainer;
use rhai::{Engine, EvalAltResult, Map};

/// The options `dot` takes.
const OPTIONS: [&str; 2] = ["margin", "mask"];

/// Registers `dot(a, b)` and `dot(a, b, options)` - also as methods
/// (`a.dot(b)`).
pub(crate) fn register(engine: &mut Engine) {
    engine
        .register_fn("dot", |a: &mut ScriptImage, b: ScriptImage| {
            dot_script(a, &b, &Map::new())
        })
        .register_fn(
            "dot",
            |a: &mut ScriptImage, b: ScriptImage, options: Map| dot_script(a, &b, &options),
        );
}

fn dot_script(a: &ScriptImage, b: &ScriptImage, options: &Map) -> Result<f64, Box<EvalAltResult>> {
    let mut margin = 0usize;
    let mut mask: Option<ScriptImage> = None;
    for (key, value) in options {
        match key.as_str() {
            "margin" => {
                let m = value
                    .as_int()
                    .map_err(|_| "dot: option `margin` must be a whole number".to_string())?;
                margin = usize::try_from(m)
                    .map_err(|_| format!("dot: option `margin` must not be negative, got {m}"))?;
            }
            "mask" => {
                mask = Some(
                    value
                        .clone()
                        .try_cast::<ScriptImage>()
                        .ok_or("dot: option `mask` must be an image")?,
                );
            }
            other => {
                return Err(format!(
                    "dot: unknown option `{other}`; available: {}",
                    OPTIONS.join(", ")
                )
                .into());
            }
        }
    }
    dot(&a.0, &b.0, margin, mask.as_ref().map(|m| m.0.as_ref()))
        .map_err(|e| format!("dot: {e}").into())
}

/// The mask's values, one per pixel.
#[derive(Clone, Copy)]
enum Mask<'a> {
    Gray(&'a [f32]),
    Labels(&'a [u32]),
}

impl Mask<'_> {
    #[inline]
    fn is_set(self, pixel: usize) -> bool {
        match self {
            Mask::Gray(v) => v[pixel] > 0.0,
            Mask::Labels(v) => v[pixel] > 0,
        }
    }
}

/// The mean of `a(x) · b(x)` over the pixels (and colour values) of `a`
/// and `b`, without a `margin` px border of each plane and, with a `mask`,
/// only where it is above 0.
///
/// # Errors
/// Label images, images of different sizes, colour types or depths, a mask
/// that isn't a grey or label image of the same size, and a margin or
/// mask that leaves no pixel (0 would look like a result).
pub(crate) fn dot(
    a: &ImageContainer,
    b: &ImageContainer,
    margin: usize,
    mask: Option<&ImageContainer>,
) -> Result<f64, String> {
    let (data_a, data_b, channels) = match (a, b) {
        (ImageContainer::F32Gray(a), ImageContainer::F32Gray(b)) => {
            (a.data.as_slice(), b.data.as_slice(), 1)
        }
        (ImageContainer::F32Rgb(a), ImageContainer::F32Rgb(b)) => {
            (a.data.as_slice(), b.data.as_slice(), 3)
        }
        (ImageContainer::U32(_), _) | (_, ImageContainer::U32(_)) => {
            return Err("label images can't be multiplied - use a grey or colour image".into());
        }
        _ => return Err("one image is grey, the other a colour image".into()),
    };
    let size = a.size();
    let depth = a.dimensionality().depth() as usize;
    if b.size() != size || b.dimensionality().depth() as usize != depth {
        return Err(format!(
            "the images differ in size: {} and {}",
            describe(a),
            describe(b)
        ));
    }
    let mask = match mask {
        None => None,
        Some(m) => {
            if m.size() != size || m.dimensionality().depth() as usize != depth {
                return Err(format!(
                    "the mask ({}) differs in size from the images ({})",
                    describe(m),
                    describe(a)
                ));
            }
            Some(match m {
                ImageContainer::F32Gray(m) => Mask::Gray(m.data.as_slice()),
                ImageContainer::U32(m) => Mask::Labels(m.data.as_slice()),
                ImageContainer::F32Rgb(_) => {
                    return Err("the mask must be a grey or label image, not a colour one".into());
                }
            })
        }
    };

    let width = size.width;
    let plane_height = size.height / depth.max(1);
    if 2 * margin >= width || 2 * margin >= plane_height {
        return Err(format!(
            "a margin of {margin} px leaves no pixels of a {width} x {plane_height} plane"
        ));
    }

    let mut total = NeumaierSum::default();
    let mut count = 0usize;
    for z in 0..depth {
        for y in margin..plane_height - margin {
            let row = (z * plane_height + y) * width;
            let (x0, x1) = (row + margin, row + width - margin);
            match mask {
                None => {
                    let range = x0 * channels..x1 * channels;
                    total.add(sum_of_products(&data_a[range.clone()], &data_b[range]));
                    count += (x1 - x0) * channels;
                }
                Some(mask) => {
                    let mut row_sum = 0.0f64;
                    for pixel in x0..x1 {
                        if mask.is_set(pixel) {
                            let values = pixel * channels..(pixel + 1) * channels;
                            row_sum += sum_of_products(&data_a[values.clone()], &data_b[values]);
                            count += channels;
                        }
                    }
                    total.add(row_sum);
                }
            }
        }
    }
    if count == 0 {
        return Err("the mask leaves no pixels".into());
    }
    Ok(total.value() / count as f64)
}

/// `Σ a[i] · b[i]` in `f64`, in four running sums (fewer rounding steps
/// on long rows, and the compiler can vectorize them).
#[inline]
fn sum_of_products(a: &[f32], b: &[f32]) -> f64 {
    let mut lanes = [0.0f64; 4];
    let (chunks_a, rest_a) = a.as_chunks::<4>();
    let (chunks_b, rest_b) = b.as_chunks::<4>();
    for (ca, cb) in chunks_a.iter().zip(chunks_b) {
        for k in 0..4 {
            lanes[k] += ca[k] as f64 * cb[k] as f64;
        }
    }
    let rest: f64 = rest_a
        .iter()
        .zip(rest_b)
        .map(|(p, q)| *p as f64 * *q as f64)
        .sum();
    (lanes[0] + lanes[1]) + (lanes[2] + lanes[3]) + rest
}

/// Kahan-Babuška (Neumaier) compensated sum.
#[derive(Default)]
struct NeumaierSum {
    sum: f64,
    compensation: f64,
}

impl NeumaierSum {
    #[inline]
    fn add(&mut self, value: f64) {
        let t = self.sum + value;
        if self.sum.abs() >= value.abs() {
            self.compensation += (self.sum - t) + value;
        } else {
            self.compensation += (value - t) + self.sum;
        }
        self.sum = t;
    }

    fn value(&self) -> f64 {
        self.sum + self.compensation
    }
}

/// "w x h" or "w x h x d" (a volume: its plane's size and depth).
fn describe(image: &ImageContainer) -> String {
    let size = image.size();
    let depth = image.dimensionality().depth() as usize;
    if depth > 1 {
        format!("{} x {} x {depth}", size.width, size.height / depth)
    } else {
        format!("{} x {}", size.width, size.height)
    }
}

// --- Test ------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ImageDimensionality;
    use crate::image::ManagedImage;
    use kornia_image::{Image, ImageSize};
    use rhai::Scope;
    use std::sync::Arc;

    /// A grey image `width` x `height` (of all planes) of `depth` planes.
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

    fn labels(width: usize, height: usize, values: Vec<u32>) -> ImageContainer {
        ImageContainer::U32(ManagedImage {
            data: Image::new(ImageSize { width, height }, values).unwrap(),
            tile_offset: Default::default(),
            plane: None,
            dimensionality: ImageDimensionality::Planar,
        })
    }

    fn rgb(width: usize, height: usize, values: Vec<f32>) -> ImageContainer {
        ImageContainer::F32Rgb(ManagedImage {
            data: Image::new(ImageSize { width, height }, values).unwrap(),
            tile_offset: Default::default(),
            plane: None,
            dimensionality: ImageDimensionality::Planar,
        })
    }

    fn a() -> ImageContainer {
        gray(4, 1, 1, vec![0.0, 4.0, 2.0, 0.0])
    }

    fn b() -> ImageContainer {
        gray(4, 1, 1, vec![0.0, 10.0, 5.0, 0.0])
    }

    #[test]
    fn the_mean_of_the_products() {
        // (0 + 40 + 10 + 0) / 4
        assert_eq!(dot(&a(), &b(), 0, None).unwrap(), 12.5);
    }

    #[test]
    fn the_least_squares_scale_of_one_image_in_another() {
        let bb = dot(&b(), &b(), 0, None).unwrap();
        assert_eq!(bb, 31.25);
        assert_eq!(dot(&a(), &b(), 0, None).unwrap() / bb, 0.4);
    }

    #[test]
    fn a_colour_image_counts_every_colour_value() {
        // 2 pixels: (1, 2, 3) · (1, 1, 1) and (4, 5, 6) · (2, 2, 2) -> 6 + 30 over 6.
        let x = rgb(2, 1, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let y = rgb(2, 1, vec![1.0, 1.0, 1.0, 2.0, 2.0, 2.0]);
        assert_eq!(dot(&x, &y, 0, None).unwrap(), 6.0);
    }

    #[test]
    fn a_margin_leaves_out_the_border() {
        // 5 x 5: a bright border (100), 1 inside.
        let mut values = vec![100.0f32; 25];
        for y in 1..4 {
            for x in 1..4 {
                values[y * 5 + x] = 1.0;
            }
        }
        let image = gray(5, 5, 1, values);
        assert_eq!(dot(&image, &image, 1, None).unwrap(), 1.0);
        assert!(dot(&image, &image, 0, None).unwrap() > 1.0);
    }

    #[test]
    fn a_mask_counts_only_its_pixels() {
        let mask = gray(4, 1, 1, vec![0.0, 1.0, 0.0, 0.0]);
        assert_eq!(dot(&a(), &b(), 0, Some(&mask)).unwrap(), 40.0);
        let label_mask = labels(4, 1, vec![0, 0, 7, 7]);
        assert_eq!(dot(&a(), &b(), 0, Some(&label_mask)).unwrap(), 5.0);

        let empty = gray(4, 1, 1, vec![0.0; 4]);
        let err = dot(&a(), &b(), 0, Some(&empty)).unwrap_err();
        assert!(err.contains("no pixels"), "{err}");
    }

    #[test]
    fn what_doesnt_fit_is_refused() {
        let wider = gray(5, 1, 1, vec![0.0; 5]);
        let err = dot(&a(), &wider, 0, None).unwrap_err();
        assert!(err.contains("differ in size"), "{err}");

        let err = dot(&labels(4, 1, vec![0; 4]), &b(), 0, None).unwrap_err();
        assert!(err.contains("label images"), "{err}");
        let err = dot(&a(), &labels(4, 1, vec![0; 4]), 0, None).unwrap_err();
        assert!(err.contains("label images"), "{err}");

        // 4 x 1: margin 1 already leaves no row.
        let err = dot(&a(), &b(), 1, None).unwrap_err();
        assert!(err.contains("leaves no pixels"), "{err}");
        let square = gray(4, 4, 1, vec![1.0; 16]);
        assert!(dot(&square, &square, 1, None).is_ok());
        assert!(dot(&square, &square, 2, None).is_err());

        let mask = gray(5, 1, 1, vec![1.0; 5]);
        let err = dot(&a(), &b(), 0, Some(&mask)).unwrap_err();
        assert!(err.contains("mask"), "{err}");
    }

    #[test]
    fn a_volume_has_its_margin_on_every_plane() {
        // Two 3 x 3 planes, stacked 3 x 6: 1 inside, 10 on each plane's
        // border. A margin on the stacked image would keep plane 1's last
        // and plane 2's first row (10s); per plane only the two centres.
        let plane = |inside: f32| {
            let mut v = vec![10.0f32; 9];
            v[4] = inside;
            v
        };
        let values = [plane(1.0), plane(3.0)].concat();
        let volume = gray(3, 6, 2, values);
        let ones = gray(3, 6, 2, vec![1.0; 18]);
        // (1 + 3) / 2
        assert_eq!(dot(&volume, &ones, 1, None).unwrap(), 2.0);

        let planar = gray(3, 6, 1, vec![1.0; 18]);
        let err = dot(&volume, &planar, 0, None).unwrap_err();
        assert!(err.contains("differ in size"), "{err}");
    }

    #[test]
    fn millions_of_pixels_keep_their_precision() {
        let side = 4096;
        let value = 1e-3f32;
        let image = gray(side, side, 1, vec![value; side * side]);
        // The exact product of the f32 value (1e-3 has no exact f32).
        let expected = value as f64 * value as f64;
        let got = dot(&image, &image, 0, None).unwrap();
        assert!(
            ((got - expected) / expected).abs() < 1e-12,
            "{got} vs {expected}"
        );
    }

    #[test]
    fn a_script_calls_it_as_a_function_and_as_a_method() {
        let engine = crate::algos::script::script_executor::new_engine();
        let mut scope = Scope::new();
        scope.push("Sf", ScriptImage(Arc::new(a())));
        scope.push("Ta", ScriptImage(Arc::new(b())));
        scope.push(
            "M",
            ScriptImage(Arc::new(gray(4, 1, 1, vec![0.0, 1.0, 1.0, 0.0]))),
        );
        let eval = |scope: &mut Scope, source: &str| engine.eval_with_scope::<f64>(scope, source);

        let expected = dot(&a(), &b(), 0, None).unwrap();
        assert_eq!(eval(&mut scope, "Sf.dot(Ta)").unwrap(), expected);
        assert_eq!(eval(&mut scope, "dot(Sf, Ta)").unwrap(), expected);
        assert_eq!(eval(&mut scope, "Sf.dot(Ta) / Ta.dot(Ta)").unwrap(), 0.4);
        // (40 + 10) / 2
        assert_eq!(eval(&mut scope, "Sf.dot(Ta, #{ mask: M })").unwrap(), 25.0);
        assert_eq!(
            eval(&mut scope, "dot(Sf, Ta, #{ margin: 0 })").unwrap(),
            expected
        );

        let err = eval(&mut scope, "Sf.dot(Ta, #{ margn: 4 })").unwrap_err();
        assert!(err.to_string().contains("unknown option `margn`"), "{err}");
        let err = eval(&mut scope, "Sf.dot(Ta, #{ margin: 1 })").unwrap_err();
        assert!(err.to_string().contains("leaves no pixels"), "{err}");
    }
}
