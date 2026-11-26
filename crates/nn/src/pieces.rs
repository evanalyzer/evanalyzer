//! # pieces
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-11-25
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.
//!
//! A fully convolutional network (U-Net, StarDist) over a large tile in
//! pieces, so its RAM is that of one piece instead of the whole tile. Each
//! piece is run with a halo at least as wide as the model's receptive field,
//! so the result equals the whole tile's up to float rounding - except for a
//! model that normalizes by the whole image (InstanceNorm).
//!
//! Ported from the removed `crates/ai/src/torch/pieces.rs` (framework-free) with its
//! tests; the sizes come from its `api.rs` as they are. Piece
//! starts are multiples of [`ALIGN`], so the model's poolings fall on the
//! whole tile's grid. A model seeing the whole image runs in pieces anyway,
//! with [`WHOLE_IMAGE_HALO`] (`joyful-pig` on 1024² pieces: 0.06 % of its
//! foreground pixels flipped).

use crate::errors::NnError;

/// The piece sides "Auto" chooses from for a model that sees a limited
/// neighbourhood - largest first: there a piece's result equals the whole
/// tile's up to float rounding, so the computer's RAM may choose
/// ([`piece_side_for`]). Larger pieces are faster (more cores used well, the
/// halo computed less often), smaller ones take less RAM.
pub const PIECE_SIDES: [usize; 3] = [2048, 1024, 512];

/// The piece side "Auto" gives a model that sees the whole image
/// (InstanceNorm): there the piece size changes the result (0.06-0.18 % of
/// the pixels), so it is the same on every computer.
pub const WHOLE_IMAGE_PIECE_SIDE: usize = 1024;

/// The halo of the pieces as planned: a model that sees the whole image gets
/// this one; the measured receptive fields were 80-112 px (U-Net, StarDist
/// layouts) - a wider one is rare (at most 240 px, the probe's limit).
pub const PLANNED_HALO: usize = 128;

/// The largest image side a U-Net or StarDist is run at on a tile side of
/// `side` with pieces of `piece`: the side itself up to `piece`, else a piece
/// with its halo on both sides (as planned, [`PLANNED_HALO`]). For RAM plans:
/// a network's RAM grows with the pixels it gets at once.
pub fn largest_model_input(side: usize, piece: usize) -> usize {
    if side <= piece {
        side
    } else {
        side.min(piece + 2 * PLANNED_HALO)
    }
}

/// The piece side for a `width x height` tile of a network taking
/// `bytes_per_pixel` of its input, for a model that sees a limited
/// neighbourhood: the largest of [`PIECE_SIDES`] whose largest input
/// ([`largest_model_input`]) takes at most a third of `ram_bytes` - the
/// computer's RAM (or the user's limit). Allowed to follow the computer: such
/// a model's pieces give the whole tile's result up to float rounding. The
/// smallest when none fits.
pub fn piece_side_for(bytes_per_pixel: f64, ram_bytes: u64, width: usize, height: usize) -> usize {
    let budget = ram_bytes as f64 / 3.0;
    PIECE_SIDES
        .into_iter()
        .find(|&piece| {
            let pixels = largest_model_input(width, piece) * largest_model_input(height, piece);
            bytes_per_pixel * pixels as f64 <= budget
        })
        .unwrap_or(PIECE_SIDES[PIECE_SIDES.len() - 1])
}

/// The pieces of a U-Net or StarDist run: one side for a model that sees a
/// limited neighbourhood, one for a model that sees the whole image - which
/// it is is known only after probing the model.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Pieces {
    /// A limited neighbourhood: the result is the same up to float rounding
    /// whatever the side.
    pub local_side: usize,
    /// The whole image (InstanceNorm): the side is part of the result.
    pub whole_image_side: usize,
}

impl Pieces {
    /// The same side for every model (a step's fixed setting).
    pub fn fixed(side: usize) -> Self {
        Self {
            local_side: side,
            whole_image_side: side,
        }
    }

    /// The larger side - what to plan RAM for before the model is known.
    pub fn largest(&self) -> usize {
        self.local_side.max(self.whole_image_side)
    }
}

/// The smallest piece side (a request's is raised to it): pieces of a few
/// hundred pixels would be mostly halo.
const MIN_PIECE: usize = 256;

/// Piece starts (and halos) are multiples of this: a U-Net halves the image
/// up to 4 times, StarDist's grid is 1-8 - a multiple of 16 keeps every
/// pooling of a piece on the whole tile's grid.
pub const ALIGN: usize = 16;

/// The halo of a model that sees the whole image (see the module docs):
/// of the halos tried on `joyful-pig` (16, 64, 128 px) the one changing the
/// fewest pixels.
pub const WHOLE_IMAGE_HALO: usize = PLANNED_HALO;

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
pub struct Planes {
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
pub fn piece_side(requested: usize) -> usize {
    requested.max(MIN_PIECE).next_multiple_of(ALIGN)
}

/// `pieces` on a GPU stating `gpu_memory` bytes, for a network taking
/// `bytes_per_pixel` of its input there: the local side (a model seeing a
/// limited neighbourhood - the same result up to float rounding whatever
/// the pieces) no larger than [`PIECE_SIDES`]' largest whose largest input
/// ([`largest_model_input`]) takes at most half of it, at least the smallest.
/// The whole-image side stays: it is part of the model's result. Without a
/// figure (`None`, the CPU) `pieces` as they are. Running out of GPU memory
/// can hang the GPU backend instead of failing (seen with burn 0.22 on
/// wgpu), so the pieces must fit before they run.
pub fn fit_gpu(
    pieces: Pieces,
    gpu_memory: Option<u64>,
    bytes_per_pixel: f64,
    width: usize,
    height: usize,
) -> Pieces {
    let Some(memory) = gpu_memory else {
        return pieces;
    };
    let budget = memory as f64 / 2.0;
    let fits = |side: usize| {
        let pixels = largest_model_input(width, side) * largest_model_input(height, side);
        bytes_per_pixel * pixels as f64 <= budget
    };
    let requested = piece_side(pieces.local_side);
    if fits(requested) {
        return pieces;
    }
    let local_side = PIECE_SIDES
        .into_iter()
        .filter(|&side| side < requested)
        .find(|&side| fits(side))
        .unwrap_or(PIECE_SIDES[PIECE_SIDES.len() - 1]);
    log::debug!(
        "AI model on a GPU of {memory} bytes: {local_side}² pieces instead of {requested}²"
    );
    Pieces {
        local_side,
        whole_image_side: pieces.whole_image_side,
    }
}

/// The smallest piece side a model that ran out of GPU memory is tried
/// again with ([`run_fitting`]).
pub const SMALLEST_RETRY_PIECE: usize = 512;

/// [`run`] as [`plan`] decided (`None`: the whole tile at once) - and when
/// the GPU runs out of memory ([`NnError::GpuOutOfMemory`]) again in pieces
/// half as large, down to [`SMALLEST_RETRY_PIECE`]. Only for a model that
/// sees a limited neighbourhood (`local_halo` gives its halo): its result is
/// the same up to float rounding whatever the pieces. A model that sees the
/// whole image keeps its pieces (they are part of its result) and gets the
/// error; the engine then runs it on the CPU.
#[allow(clippy::too_many_arguments)]
pub fn run_fitting(
    pixels: &[f32],
    channels: usize,
    width: usize,
    height: usize,
    plan: Option<(usize, usize)>,
    local_halo: impl Fn() -> Result<Option<usize>, NnError>,
    mut predict: impl FnMut(&[f32], usize, usize) -> Result<Planes, NnError>,
    model: &str,
) -> Result<Planes, NnError> {
    let (mut side, mut result) = match plan {
        Some((halo, piece)) => (
            piece,
            run(pixels, channels, width, height, halo, piece, &mut predict),
        ),
        None => (width.max(height), predict(pixels, width, height)),
    };
    loop {
        let Err(NnError::GpuOutOfMemory(why)) = &result else {
            return result;
        };
        if side <= SMALLEST_RETRY_PIECE {
            return result;
        }
        let Some(halo) = local_halo()? else {
            return result;
        };
        let smaller = piece_side((side / 2).max(SMALLEST_RETRY_PIECE));
        log::warn!(
            "AI model {model}: out of GPU memory in {side}² pieces, again in {smaller}² ({})",
            why.lines().last().unwrap_or("")
        );
        side = smaller;
        result = run(pixels, channels, width, height, halo, side, &mut predict);
    }
}

/// Whether a `width x height` tile runs in more than one piece of `piece`.
pub fn needs_pieces(width: usize, height: usize, piece: usize) -> bool {
    width > piece || height > piece
}

/// Runs `predict` (the model on a `w x h` image of `channels` planes) over
/// the `width x height` `pixels` (`channels` planes, one after the other) in
/// pieces of `piece` with `halo` pixels of context (see the
/// module docs) and puts the pieces' middles together. Each piece's output
/// must be its size divided by one grid for all pieces.
pub fn run(
    pixels: &[f32],
    channels: usize,
    width: usize,
    height: usize,
    halo: usize,
    piece: usize,
    mut predict: impl FnMut(&[f32], usize, usize) -> Result<Planes, NnError>,
) -> Result<Planes, NnError> {
    let mut result: Option<(usize, Planes)> = None;
    let mut buffer = Vec::new();
    for ((keep_y0, keep_y1), (y0, y1)) in spans(height, halo, piece) {
        for ((keep_x0, keep_x1), (x0, x1)) in spans(width, halo, piece) {
            let (w, h) = (x1 - x0, y1 - y0);
            buffer.clear();
            for plane in pixels.chunks_exact(width * height).take(channels) {
                for y in y0..y1 {
                    buffer.extend_from_slice(&plane[y * width + x0..y * width + x1]);
                }
            }
            let piece = predict(&buffer, w, h)?;
            let grid = (w / piece.width.max(1)).max(1);
            if piece.width != w / grid || piece.height != h / grid {
                return Err(NnError::Internal(format!(
                    "AI model: a {w}x{h} piece gave a {}x{} output, not its size divided by \
                     one grid",
                    piece.width, piece.height
                )));
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
                return Err(NnError::Internal(
                    "AI model: its pieces gave different outputs".into(),
                ));
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
        .ok_or_else(|| NnError::InvalidInput("AI model: an empty tile".into()))
}

/// The halo `predict`'s model needs: how far a change of one pixel reaches
/// in its output (StarDist's `_compute_receptive_field`), measured on a
/// [`PROBE`]² image - empty and with one bright pixel in the middle (of each
/// channel) -
/// rounded up to [`ALIGN`] with one `ALIGN` of margin. `None`: it reaches
/// (almost) everywhere - the model sees the whole image (see
/// [`halo_of`]).
pub fn receptive_halo(
    channels: usize,
    mut predict: impl FnMut(&[f32], usize, usize) -> Result<Planes, NnError>,
) -> Result<Option<usize>, NnError> {
    let center = PROBE / 2;
    let empty = vec![0.0; channels * PROBE * PROBE];
    let mut bright = empty.clone();
    for plane in bright.chunks_exact_mut(PROBE * PROBE) {
        plane[center * PROBE + center] = 1.0;
    }
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
        return Err(NnError::Internal(
            "AI model: its outputs differ in size on the same input size".into(),
        ));
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

/// How a `width x height` tile runs with the request's `pieces`: `None`
/// whole, else `(halo, piece side)`. `measured` gives the model's
/// [`receptive_halo`] (asked only when a tile may need pieces): a model
/// seeing a limited neighbourhood runs in `pieces.local_side` with its
/// halo, one seeing the whole image in `pieces.whole_image_side` with
/// [`WHOLE_IMAGE_HALO`] (logged - its pieces differ a little from the whole
/// tile).
pub fn plan(
    width: usize,
    height: usize,
    pieces: Pieces,
    measured: impl FnOnce() -> Result<Option<usize>, NnError>,
    model: &str,
) -> Result<Option<(usize, usize)>, NnError> {
    let (local, whole_image) = (
        piece_side(pieces.local_side),
        piece_side(pieces.whole_image_side),
    );
    if !needs_pieces(width, height, local.min(whole_image)) {
        return Ok(None);
    }
    let (halo, piece) = match measured()? {
        Some(halo) => (halo, local),
        None => {
            log::info!(
                "AI model {} normalizes by the whole image: big tiles run in {whole_image}² \
                 pieces with {WHOLE_IMAGE_HALO} px of context - a few pixels differ from one piece",
                model
            );
            (WHOLE_IMAGE_HALO, whole_image)
        }
    };
    Ok(needs_pieces(width, height, piece).then_some((halo, piece)))
}

/// `plane` (`width x height`) padded by `(left, right, top, bottom)`,
/// appended to `out`: mirrored without repeating the edge (PyTorch's
/// `reflect`); a side too small to mirror repeats its edge pixel instead
/// (`replicate`).
pub fn pad_plane(
    plane: &[f32],
    width: usize,
    height: usize,
    (left, right, top, bottom): (usize, usize, usize, usize),
    out: &mut Vec<f32>,
) {
    let reflect = left < width && right < width && top < height && bottom < height;
    let source = |i: isize, len: usize| -> usize {
        let n = len as isize;
        if reflect {
            let i = if i < 0 { -i } else { i };
            (if i >= n { 2 * (n - 1) - i } else { i }) as usize
        } else {
            i.clamp(0, n - 1) as usize
        }
    };
    for y in -(top as isize)..(height + bottom) as isize {
        let row = source(y, height) * width;
        for x in -(left as isize)..(width + right) as isize {
            out.push(plane[row + source(x, width)]);
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
    ) -> impl FnMut(&[f32], usize, usize) -> Result<Planes, NnError> {
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

    /// A tile up to the smaller side runs whole without measuring the model;
    /// beyond, a model seeing a limited neighbourhood gets the local side
    /// with its halo, one seeing the whole image its side with
    /// [`WHOLE_IMAGE_HALO`].
    #[test]
    fn the_plan_gives_each_kind_of_model_its_pieces() {
        let auto = Pieces {
            local_side: 2048,
            whole_image_side: 1024,
        };
        let model = "m";
        let never =
            || -> Result<Option<usize>, NnError> { panic!("measured a tile that runs whole") };
        assert_eq!(plan(1024, 1024, auto, never, model).unwrap(), None);
        let local = || Ok(Some(96));
        assert_eq!(
            plan(4096, 4096, auto, local, model).unwrap(),
            Some((96, 2048))
        );
        assert_eq!(
            plan(2048, 2048, auto, local, model).unwrap(),
            None,
            "fits the local side"
        );
        let whole = || Ok(None);
        assert_eq!(
            plan(4096, 4096, auto, whole, model).unwrap(),
            Some((WHOLE_IMAGE_HALO, 1024))
        );
        assert_eq!(
            plan(2048, 2048, auto, whole, model).unwrap(),
            Some((WHOLE_IMAGE_HALO, 1024))
        );
        let fixed = Pieces::fixed(512);
        assert_eq!(
            plan(1024, 1024, fixed, local, model).unwrap(),
            Some((96, 512))
        );
        assert_eq!(
            plan(1024, 1024, fixed, whole, model).unwrap(),
            Some((WHOLE_IMAGE_HALO, 512))
        );
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
            let halo = receptive_halo(1, box_filter(5, grid)).unwrap().unwrap();
            assert!(halo >= 5 + grid, "grid {grid}: halo {halo}");
            let pieces = run(&pixels, 1, width, height, halo, 1024, box_filter(5, grid)).unwrap();
            assert_eq!(pieces, whole, "grid {grid}");
        }
    }

    #[test]
    fn a_halo_narrower_than_the_model_sees_changes_the_borders() {
        let (width, height) = (1100, 64);
        let pixels = image(width, height);
        let whole = box_filter(24, 1)(&pixels, width, height).unwrap();
        let pieces = run(&pixels, 1, width, height, 16, 1024, box_filter(24, 1)).unwrap();
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
        assert_eq!(receptive_halo(1, global).unwrap(), None);
    }

    const GB: u64 = 1 << 30;

    /// "Auto" for a model seeing a limited neighbourhood: a U-Net (1.6 KB per
    /// pixel) and StarDist (0.4 KB) on a 4096² slide tile and smaller images:
    /// 2048² pieces where a third of the RAM holds one, smaller ones on a
    /// small computer; an image up to the piece side whole.
    #[test]
    fn pieces_as_large_as_a_third_of_the_ram_holds() {
        let unet = |ram: u64, side: usize| piece_side_for(1600.0, ram, side, side);
        assert_eq!(unet(64 * GB, 4096), 2048);
        assert_eq!(unet(32 * GB, 4096), 2048);
        assert_eq!(unet(16 * GB, 4096), 1024);
        assert_eq!(unet(8 * GB, 4096), 1024);
        assert_eq!(unet(4 * GB, 4096), 512);
        assert_eq!(unet(GB, 4096), 512, "the smallest when none fits");
        assert_eq!(unet(64 * GB, 2048), 2048, "whole");
        assert_eq!(unet(8 * GB, 2048), 1024);
        assert_eq!(unet(8 * GB, 1024), 2048, "whole: it fits");
        assert_eq!(piece_side_for(400.0, 8 * GB, 4096, 4096), 2048);
        let auto = Pieces {
            local_side: 2048,
            whole_image_side: WHOLE_IMAGE_PIECE_SIDE,
        };
        assert_eq!(auto.largest(), 2048);
        assert_eq!(Pieces::fixed(512).largest(), 512);
    }

    /// On a GPU of 8 GB a U-Net (825 bytes per pixel in f16) gets 1024²
    /// pieces instead of 2048² - a 2048² piece with its halo needs 4.4 GB of
    /// the 4 GB budget; 16 GB keep 2048². The whole-image side and the CPU
    /// (no figure) keep theirs.
    #[test]
    fn the_pieces_fit_into_half_the_gpu_memory() {
        let auto = Pieces {
            local_side: 2048,
            whole_image_side: WHOLE_IMAGE_PIECE_SIDE,
        };
        let fit = |memory: Option<u64>| fit_gpu(auto, memory, 825.0, 4096, 4096);
        assert_eq!(fit(Some(8 * GB)).local_side, 1024);
        assert_eq!(fit(Some(16 * GB)).local_side, 2048);
        assert_eq!(
            fit(Some(2 * GB)).local_side,
            512,
            "the smallest when none fits"
        );
        assert_eq!(fit(None), auto);
        assert_eq!(fit(Some(8 * GB)).whole_image_side, WHOLE_IMAGE_PIECE_SIDE);
        // A small image fits whole whatever was asked.
        assert_eq!(fit_gpu(auto, Some(8 * GB), 825.0, 1100, 900), auto);
    }

    /// A model the GPU has memory for only up to `max` pixels per side: a
    /// local one runs in smaller pieces - the same result as running in
    /// those pieces directly (here exactly the whole tile's, the halo being
    /// wide enough); a whole-image one keeps its pieces and gets the error.
    #[test]
    fn out_of_gpu_memory_runs_again_in_smaller_pieces_for_a_local_model() {
        let (width, height) = (2100, 1300);
        let pixels = image(width, height);
        let max = 700;
        let mut sizes = Vec::new();
        let mut limited = |p: &[f32], w: usize, h: usize| {
            sizes.push(w.max(h));
            if w.max(h) > max {
                return Err(NnError::GpuOutOfMemory("out of device memory".into()));
            }
            box_filter(5, 2)(p, w, h)
        };
        let halo = receptive_halo(1, box_filter(5, 2)).unwrap().unwrap();
        let out = run_fitting(
            &pixels,
            1,
            width,
            height,
            Some((halo, 2048)),
            || Ok(Some(halo)),
            &mut limited,
            "test",
        )
        .unwrap();
        assert_eq!(out, box_filter(5, 2)(&pixels, width, height).unwrap());
        // 2048: the whole tile (2100 wide) failed, 1024 pieces failed, 512
        // pieces (with their halo) fit.
        assert!(sizes.iter().filter(|&&s| s > max).count() >= 2, "{sizes:?}");

        let whole_image = run_fitting(
            &pixels,
            1,
            width,
            height,
            Some((WHOLE_IMAGE_HALO, 1024)),
            || Ok(None),
            |p: &[f32], w: usize, h: usize| {
                if w.max(h) > max {
                    return Err(NnError::GpuOutOfMemory("out of device memory".into()));
                }
                box_filter(5, 2)(p, w, h)
            },
            "test",
        );
        assert!(matches!(whole_image, Err(NnError::GpuOutOfMemory(_))));
    }

    #[test]
    fn the_largest_input_is_a_piece_with_its_halo_or_the_tile() {
        assert_eq!(largest_model_input(4096, 2048), 2048 + 256);
        assert_eq!(largest_model_input(4096, 1024), 1024 + 256);
        assert_eq!(largest_model_input(2048, 2048), 2048);
        assert_eq!(largest_model_input(1100, 1024), 1100);
    }
}
