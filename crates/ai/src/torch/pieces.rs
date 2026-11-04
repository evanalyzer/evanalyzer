//! # pieces
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-11-04
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.
//!
//! Runs a fully convolutional model (U-Net, StarDist) over a large tile in
//! pieces, so its RAM is that of one piece instead of the whole tile: a
//! U-Net with 64 filters takes ~1.3 KB per pixel, ~21 GB on a 4096² tile
//! (BENCHMARK.md "AI models: RAM per inference").
//!
//! As StarDist's own `n_tiles`: each piece is run with a border of context
//! (the halo) at least as wide as the model sees around a pixel - its
//! receptive field, measured once per model ([`receptive_halo`]) - and only
//! the piece's middle is kept. Each kept pixel is then computed from the same
//! pixels as on the whole tile; the piece starts are multiples of
//! [`ALIGN`], so the model's poolings fall on the same grid. The result
//! equals the whole tile's up to float rounding (the libraries may sum in
//! another order on another size). Cellpose blends its pieces instead - it
//! must: its model takes exactly 256², too small for a full halo.
//!
//! A model that sees the whole image - it normalizes each layer by the
//! image's mean and spread (`InstanceNorm2d`, e.g. bioimage.io's
//! torch-em U-Nets) - can't be split exactly: each piece has its own
//! statistics. It runs in pieces anyway (decided 2026-11-04: else ~21 GB
//! per 4096² slide tile), with [`WHOLE_IMAGE_HALO`]: `joyful-pig` on
//! 1024² pieces flipped 0.06 % of its foreground and 0.3 % of its boundary
//! pixels (a slide's 4096² tiles already have their own statistics, per
//! tile). Logged when it happens.
//!
//! The piece side comes with the request: the analysis picks it from the
//! computer's RAM ([`crate::api::piece_side_for`]) - 2048² where it fits,
//! 1024² or 512² on a small computer. A tile of up to it per side runs as
//! one piece - as before, without a copy or a probe. Pieces are as large as
//! the RAM allows: libtorch uses many cores less well on small inputs, and
//! the halo is computed twice - 1024² pieces made a 2048² tile 1.3-1.9x
//! slower (BENCHMARK.md).

use super::Result;

/// The smallest piece side (a request's is raised to it): pieces of a few
/// hundred pixels would be mostly halo.
const MIN_PIECE: usize = 256;

/// Piece starts (and halos) are multiples of this: a U-Net halves the image
/// up to 4 times, StarDist's grid is 1-8 - a multiple of 16 keeps every
/// pooling of a piece on the whole tile's grid.
pub(crate) const ALIGN: usize = 16;

/// The halo of a model that sees the whole image (see the module docs):
/// of the halos tried on `joyful-pig` (16, 64, 128 px) the one changing the
/// fewest pixels.
pub(crate) const WHOLE_IMAGE_HALO: usize = crate::api::PLANNED_HALO;

/// Side of the image the receptive field is measured on: a halo beyond
/// half of it (the model sees further, e.g. a global normalization) runs
/// the tile whole.
const PROBE: usize = 512;

/// A model's output on an image region: `planes` planes of `width * height`
/// values, row-major - at the region's size divided by the model's grid (1
/// for a U-Net, StarDist's grid). The first plane (U-Net: foreground,
/// StarDist: probability) and the others (boundary, ray distances) apart,
/// as the answer has them - no copy to split them.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Planes {
    pub planes: usize,
    pub width: usize,
    pub height: usize,
    pub first: Vec<f32>,
    pub rest: Vec<f32>,
}

impl Planes {
    /// Plane `index`.
    pub fn plane(&self, index: usize) -> &[f32] {
        let len = self.width * self.height;
        match index {
            0 => &self.first[..len],
            i => &self.rest[(i - 1) * len..i * len],
        }
    }

    /// Plane `index`, to write.
    fn plane_mut(&mut self, index: usize) -> &mut [f32] {
        let len = self.width * self.height;
        match index {
            0 => &mut self.first[..len],
            i => &mut self.rest[(i - 1) * len..i * len],
        }
    }
}

/// The kept part `[start, end)` of each piece of at most `piece` along one
/// side of `len` pixels, and the region `[from, to)` run for it (with `halo`
/// on both sides, within the image). Starts are multiples of [`ALIGN`].
fn spans(len: usize, halo: usize, piece: usize) -> Vec<((usize, usize), (usize, usize))> {
    let count = len.div_ceil(piece).max(1);
    let step = len.div_ceil(count).div_ceil(ALIGN) * ALIGN;
    (0..count)
        .map(|i| (i * step, ((i + 1) * step).min(len)))
        .filter(|(start, end)| start < end)
        .map(|(start, end)| {
            let from = start.saturating_sub(halo);
            let to = (end + halo).min(len);
            ((start, end), (from, to))
        })
        .collect()
}

/// The piece side to run with for a request's `piece_side`: at least
/// [`MIN_PIECE`], a multiple of [`ALIGN`].
pub(crate) fn piece_side(requested: usize) -> usize {
    requested.max(MIN_PIECE).next_multiple_of(ALIGN)
}

/// Whether a `width x height` tile runs in more than one piece of `piece`.
pub(crate) fn needs_pieces(width: usize, height: usize, piece: usize) -> bool {
    width > piece || height > piece
}

/// Runs `predict` (the model on a `w x h` image) over the `width x height`
/// `pixels` in pieces of `piece` with `halo` pixels of context (see the
/// module docs) and puts the pieces' middles together. Each piece's output
/// must be its size divided by one grid for all pieces.
pub(crate) fn run(
    pixels: &[f32],
    width: usize,
    height: usize,
    halo: usize,
    piece: usize,
    mut predict: impl FnMut(&[f32], usize, usize) -> Result<Planes>,
) -> Result<Planes> {
    let mut result: Option<(usize, Planes)> = None;
    let mut buffer = Vec::new();
    for ((keep_y0, keep_y1), (y0, y1)) in spans(height, halo, piece) {
        for ((keep_x0, keep_x1), (x0, x1)) in spans(width, halo, piece) {
            let (w, h) = (x1 - x0, y1 - y0);
            buffer.clear();
            for y in y0..y1 {
                buffer.extend_from_slice(&pixels[y * width + x0..y * width + x1]);
            }
            let piece = predict(&buffer, w, h)?;
            let grid = (w / piece.width.max(1)).max(1);
            if piece.width != w / grid || piece.height != h / grid {
                return Err(format!(
                    "AI model: a {w}x{h} piece gave a {}x{} output, not its size divided by \
                     one grid",
                    piece.width, piece.height
                ));
            }
            let (_, out) = result.get_or_insert_with(|| {
                (
                    grid,
                    Planes {
                        planes: piece.planes,
                        width: width / grid,
                        height: height / grid,
                        first: vec![0.0; (width / grid) * (height / grid)],
                        rest: vec![0.0; (piece.planes - 1) * (width / grid) * (height / grid)],
                    },
                )
            });
            if piece.planes != out.planes || out.width != width / grid {
                return Err("AI model: its pieces gave different outputs".into());
            }
            // The middle, in output pixels (the last piece up to the end).
            let (mx0, my0) = (keep_x0 / grid, keep_y0 / grid);
            let mx1 = if keep_x1 == width {
                out.width
            } else {
                keep_x1 / grid
            };
            let my1 = if keep_y1 == height {
                out.height
            } else {
                keep_y1 / grid
            };
            let (px, py) = (x0 / grid, y0 / grid);
            let out_width = out.width;
            for plane in 0..out.planes {
                let source = piece.plane(plane);
                let target = out.plane_mut(plane);
                for y in my0..my1 {
                    let from = (y - py) * piece.width + (mx0 - px);
                    target[y * out_width + mx0..y * out_width + mx1]
                        .copy_from_slice(&source[from..from + (mx1 - mx0)]);
                }
            }
        }
    }
    result
        .map(|(_, planes)| planes)
        .ok_or_else(|| "AI model: an empty tile".into())
}

/// The halo `predict`'s model needs: how far a change of one pixel reaches
/// in its output (StarDist's `_compute_receptive_field`), measured on a
/// [`PROBE`]² image - empty and with one bright pixel in the middle -
/// rounded up to [`ALIGN`] with one `ALIGN` of margin. `None`: it reaches
/// (almost) everywhere - the model sees the whole image (see
/// [`halo_of`]).
pub(crate) fn receptive_halo(
    mut predict: impl FnMut(&[f32], usize, usize) -> Result<Planes>,
) -> Result<Option<usize>> {
    let center = PROBE / 2;
    let empty = vec![0.0; PROBE * PROBE];
    let mut bright = empty.clone();
    bright[center * PROBE + center] = 1.0;
    let (a, b) = (
        predict(&empty, PROBE, PROBE)?,
        predict(&bright, PROBE, PROBE)?,
    );
    if a.width != b.width
        || a.height != b.height
        || a.planes != b.planes
        || a.width == 0
        || a.planes == 0
    {
        return Err("AI model: its outputs differ in size on the same input size".into());
    }
    let grid = (PROBE / a.width).max(1);
    let mut reach = 0usize;
    for plane in 0..a.planes {
        let (pa, pb) = (a.plane(plane), b.plane(plane));
        for y in 0..a.height {
            for x in 0..a.width {
                let (va, vb) = (pa[y * a.width + x], pb[y * a.width + x]);
                // Any change, however small (as StarDist): the
                // same numbers in a pixel's window give the same result.
                if va.to_bits() != vb.to_bits() {
                    // The farthest input pixel this output pixel stands for.
                    let far = |o: usize| {
                        (o * grid)
                            .abs_diff(center)
                            .max((o * grid + grid - 1).abs_diff(center))
                    };
                    reach = reach.max(far(x)).max(far(y));
                }
            }
        }
    }
    let halo = (reach + ALIGN).div_ceil(ALIGN) * ALIGN;
    Ok((halo < center - ALIGN).then_some(halo))
}

/// The halo to run pieces with, from [`receptive_halo`]'s answer:
/// [`WHOLE_IMAGE_HALO`] for a model seeing the whole image (logged - its
/// pieces differ a little from the whole tile).
pub(crate) fn halo_of(measured: Option<usize>, model: &std::path::Path) -> usize {
    match measured {
        Some(halo) => halo,
        None => {
            log::info!(
                "AI model {} normalizes by the whole image: big tiles run in pieces with \
                 {WHOLE_IMAGE_HALO} px of context - a few pixels differ from one piece",
                model.display()
            );
            WHOLE_IMAGE_HALO
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A "model" averaging each pixel's `radius` neighbourhood (zeros
    /// outside), on a grid of `grid`: what a convolution network does to
    /// its pieces' borders.
    fn box_filter(
        radius: usize,
        grid: usize,
    ) -> impl FnMut(&[f32], usize, usize) -> Result<Planes> {
        move |pixels, w, h| {
            let (ow, oh) = (w / grid, h / grid);
            let mut data = Vec::with_capacity(2 * ow * oh);
            for plane in 0..2usize {
                for oy in 0..oh {
                    for ox in 0..ow {
                        let (cx, cy) = ((ox * grid) as i64, (oy * grid) as i64);
                        let r = radius as i64;
                        let mut sum = 0.0;
                        for y in cy - r..=cy + r {
                            for x in cx - r..=cx + r {
                                if (0..w as i64).contains(&x) && (0..h as i64).contains(&y) {
                                    sum += pixels[y as usize * w + x as usize];
                                }
                            }
                        }
                        data.push(sum * (plane + 1) as f32);
                    }
                }
            }
            let rest = data.split_off(ow * oh);
            Ok(Planes {
                planes: 2,
                width: ow,
                height: oh,
                first: data,
                rest,
            })
        }
    }

    fn image(width: usize, height: usize) -> Vec<f32> {
        (0..width * height)
            .map(|i| ((i * 7919) % 101) as f32 / 101.0)
            .collect()
    }

    #[test]
    fn a_tile_up_to_the_piece_side_runs_whole() {
        assert!(!needs_pieces(2048, 2048, 2048));
        assert!(!needs_pieces(2048, 1, 2048));
        assert!(needs_pieces(2049, 2048, 2048));
        assert!(needs_pieces(2048, 2048, 1024));
        assert_eq!(spans(4096, 112, 2048).len(), 2);
        assert_eq!(spans(4096, 112, 1024).len(), 4);
        assert_eq!(piece_side(100), MIN_PIECE);
        assert_eq!(piece_side(1000), 1008);
        assert_eq!(piece_side(2048), 2048);
    }

    #[test]
    fn spans_cover_the_side_once_with_aligned_starts_and_the_halo_within_it() {
        for (len, piece) in [1, 15, 1024, 1025, 2048, 2500, 4096, 4097]
            .into_iter()
            .flat_map(|len| [(len, 2048), (len, 1024), (len, 512)])
        {
            for halo in [0, 32, 96] {
                let spans = spans(len, halo, piece);
                assert_eq!(spans.first().unwrap().0.0, 0);
                assert_eq!(spans.last().unwrap().0.1, len);
                for pair in spans.windows(2) {
                    assert_eq!(pair[0].0.1, pair[1].0.0, "len {len}");
                }
                for ((start, end), (from, to)) in spans {
                    assert_eq!(start % ALIGN, 0);
                    assert_eq!(from % ALIGN, 0, "len {len} halo {halo}");
                    assert!(end - start <= piece, "len {len}");
                    assert_eq!(from, start.saturating_sub(halo));
                    assert_eq!(to, (end + halo).min(len));
                }
            }
        }
    }

    #[test]
    fn pieces_with_a_wide_enough_halo_give_the_whole_tile_exactly() {
        for grid in [1, 2, 4] {
            let (width, height) = (2100, 1040);
            let pixels = image(width, height);
            let whole = box_filter(5, grid)(&pixels, width, height).unwrap();
            let halo = receptive_halo(box_filter(5, grid)).unwrap().unwrap();
            assert!(halo >= 5 + grid, "grid {grid}: halo {halo}");
            let pieces = run(&pixels, width, height, halo, 1024, box_filter(5, grid)).unwrap();
            assert_eq!(pieces, whole, "grid {grid}");
        }
    }

    #[test]
    fn a_halo_narrower_than_the_model_sees_changes_the_borders() {
        let (width, height) = (1100, 64);
        let pixels = image(width, height);
        let whole = box_filter(24, 1)(&pixels, width, height).unwrap();
        let pieces = run(&pixels, width, height, 16, 1024, box_filter(24, 1)).unwrap();
        assert_ne!(pieces, whole);
    }

    #[test]
    fn a_model_seeing_everything_is_seen_as_such() {
        let global = |pixels: &[f32], w: usize, h: usize| {
            let mean = pixels.iter().sum::<f32>() / pixels.len() as f32;
            Ok(Planes {
                planes: 1,
                width: w,
                height: h,
                first: vec![mean; w * h],
                rest: Vec::new(),
            })
        };
        assert_eq!(receptive_halo(global).unwrap(), None);
    }

    /// Cells (bright discs of varying size and brightness) on a textured
    /// background - something a network's output varies on everywhere.
    fn cells(width: usize, height: usize) -> Vec<f32> {
        (0..width * height)
            .map(|i| {
                let (x, y) = ((i % width) as f32, (i / width) as f32);
                let (cx, cy) = ((x / 37.0).floor(), (y / 41.0).floor());
                let (dx, dy) = (x - cx * 37.0 - 18.0, y - cy * 41.0 - 20.0);
                let radius = 6.0 + (cx * 3.0 + cy * 5.0) % 9.0;
                let cell = if dx * dx + dy * dy < radius * radius {
                    0.5 + ((cx + cy) % 4.0) * 0.1
                } else {
                    0.0
                };
                cell + 0.05 * (x * 0.31).sin() * (y * 0.17).cos()
            })
            .collect()
    }

    /// The largest difference of `a` and `b` relative to `b`'s range, and the
    /// pixels whose first plane lands on the other side of 0.5.
    fn compare(a: &Planes, b: &Planes) -> (f32, usize) {
        assert_eq!((a.planes, a.width, a.height), (b.planes, b.width, b.height));
        let mut worst = 0f32;
        for plane in 0..a.planes {
            let (pa, pb) = (a.plane(plane), b.plane(plane));
            let range = pb.iter().fold(0f32, |m, v| m.max(v.abs())).max(1e-6);
            for (va, vb) in pa.iter().zip(pb) {
                worst = worst.max((va - vb).abs() / range);
            }
        }
        let flipped = a
            .plane(0)
            .iter()
            .zip(b.plane(0))
            .filter(|(va, vb)| (**va >= 0.5) != (**vb >= 0.5))
            .count();
        (worst, flipped)
    }

    /// Real libtorch networks laid out as StarDist's `2D_versatile_fluo`
    /// and a bioimage.io U-Net (`test_support`, irregular weights) on a
    /// tile of 3 x 2 pieces: in pieces with the measured halo, the outputs
    /// equal the whole tile's up to float rounding (measured: < 1e-7 of the
    /// range), and no pixel changes sides of 0.5 - with a too narrow halo
    /// they differ more (16 px: ~5e-5).
    #[test]
    fn real_networks_in_pieces_give_the_whole_tile() {
        use crate::torch::test_support::{bench_stardist_model, bench_unet_model};
        use tch::{CModule, Device};
        // Not multiples of 16: StarDist pads, the pieces' ends too.
        let (width, height) = (2100, 1100);
        let pixels = cells(width, height);
        let device = Device::Cpu;
        let (_stardist_dir, stardist_path) = bench_stardist_model();
        let (_unet_dir, unet_path) = bench_unet_model(32);
        let stardist = CModule::load_on_device(&stardist_path, device).unwrap();
        let unet = CModule::load_on_device(&unet_path, device).unwrap();
        let models: [(&str, Box<dyn Fn(&[f32], usize, usize) -> Result<Planes>>); 2] = [
            (
                "StarDist",
                Box::new(|p: &[f32], w, h| {
                    super::super::stardist::predict(&stardist, device, p, w, h)
                }),
            ),
            (
                "U-Net",
                Box::new(|p: &[f32], w, h| {
                    super::super::unet::predict(&unet, device, p, w, h, 0, false, 1)
                }),
            ),
        ];
        for (name, predict) in &models {
            let halo = receptive_halo(predict).unwrap().expect("a local model");
            let whole = predict(&pixels, width, height).unwrap();
            let pieces = run(&pixels, width, height, halo, 1024, predict).unwrap();
            let (worst, flipped) = compare(&pieces, &whole);
            println!(
                "{name}: halo {halo}, largest difference {worst:e} of the range, {flipped} pixels flipped"
            );
            assert!(worst < 1e-6, "{name}: {worst}");
            assert_eq!(flipped, 0, "{name}");
            // The check sees a wrong halo: far context changes a pixel
            // little, but far more than rounding does.
            let narrow = run(&pixels, width, height, ALIGN, 1024, predict).unwrap();
            let (narrow_worst, _) = compare(&narrow, &whole);
            assert!(
                narrow_worst > 20.0 * worst.max(1e-7),
                "{name}: a halo of {ALIGN}: {narrow_worst}"
            );
        }
    }

    /// A `.npy` file of little-endian `f32` (its header skipped).
    fn read_npy_f32(path: &std::path::Path) -> Vec<f32> {
        let bytes = std::fs::read(path).unwrap();
        let header = u16::from_le_bytes([bytes[8], bytes[9]]) as usize;
        bytes[10 + header..]
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    }

    /// The bioimage.io `joyful-pig` U-Net (real weights; ~116 MB, not in
    /// git: download `weights-torchscript.pt` and `test_input_0.npy` from
    /// bioimage.io into `tests/joyful-pig/`) normalizes every layer by the
    /// whole image (`InstanceNorm2d`): no halo makes its pieces exact - on a
    /// 2048 x 1536 image (its 512² test input, normalized and mirrored) a
    /// halo of 16/64/128 px flipped 0.18/0.10/0.06 % of the foreground and
    /// 0.64/0.44/0.31 % of the boundary pixels (measured 2026-11-04). The
    /// probe must see that (it runs with [`WHOLE_IMAGE_HALO`], logged).
    /// Run with:
    ///   cargo test -p evanalyzer_ai --features torch --lib real_joyful_pig -- --ignored --nocapture
    #[test]
    #[ignore]
    fn real_joyful_pig_is_seen_normalizing_by_the_whole_image() {
        use tch::{CModule, Device};
        let dir = std::path::PathBuf::from(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../tests/joyful-pig"
        ));
        let raw = read_npy_f32(&dir.join("test_input_0.npy"));
        let n = 512;
        let mean = raw.iter().sum::<f32>() / raw.len() as f32;
        let std = (raw.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / raw.len() as f32).sqrt();
        let (width, height) = (2048, 1536);
        // Mirrored copies: no seams a model would react to.
        let fold = |v: usize| {
            let v = v % (2 * n);
            if v < n { v } else { 2 * n - 1 - v }
        };
        let pixels: Vec<f32> = (0..width * height)
            .map(|i| (raw[fold(i / width) * n + fold(i % width)] - mean) / (std + 1e-6))
            .collect();
        let device = Device::Cpu;
        let model = CModule::load_on_device(dir.join("weights-torchscript.pt"), device).unwrap();
        let predict = |p: &[f32], w: usize, h: usize| {
            super::super::unet::predict(&model, device, p, w, h, 0, false, 1)
        };
        assert_eq!(receptive_halo(predict).unwrap(), None);
        // Its pieces change a few pixels - not more than measured.
        let whole = predict(&pixels, width, height).unwrap();
        let pieces = run(&pixels, width, height, WHOLE_IMAGE_HALO, 1024, predict).unwrap();
        let (_, flipped) = compare(&pieces, &whole);
        println!(
            "joyful-pig in pieces (halo {WHOLE_IMAGE_HALO}): {flipped} of {} pixels flipped",
            width * height
        );
        assert!(flipped > 0);
        assert!(flipped * 1000 < width * height, "{flipped}");
    }
}
