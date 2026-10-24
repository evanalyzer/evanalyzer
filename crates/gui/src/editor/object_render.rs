//! Drawing objects on the image - the editor's (the project's objects and
//! the preview's) and Results' split view's (an analysis' objects) alike:
//! each object a box, a mask and a color with the objects' opacity, painted
//! in layers ([`ZOrder`]: the selected object, the selected class, the
//! classes in list order, unclassified) - fills first, then every outline
//! opaque on top - with lines from the selected object to its relatives.
//! A click picks the object drawn on top ([`pick`]). The callers only turn
//! their objects into [`ObjectDrawInstance`]s.

use bitvec::prelude::*;
use evanalyzer_app::project::ProjectExt;
use evanalyzer_cfg::core_types::ObjectClass;
use std::collections::{HashMap, HashSet};

/// The lines' opacity.
const LINK_ALPHA: f32 = 0.9;

/// An image line: from one image point to another.
pub(crate) type Link = ((f32, f32), (f32, f32));

/// Draws `objects` (layer key, instance) into `buffer` (`buf_w` × `buf_h`),
/// image pixel `p` at `p * zoom + off`: bottom to top by key - an equal key
/// keeps the given order - then `links` over them.
#[allow(clippy::too_many_arguments)]
pub(crate) fn draw_objects(
    buffer: &mut [slint::Rgba8Pixel],
    mut objects: Vec<(u32, ObjectDrawInstance)>,
    links: &[Link],
    buf_w: u32,
    buf_h: u32,
    zoom: f32,
    off_x: f32,
    off_y: f32,
) {
    // Stable: objects with an equal key keep their order.
    objects.sort_by_key(|(key, _)| *key);
    let instances: Vec<ObjectDrawInstance> = objects.into_iter().map(|(_, i)| i).collect();
    composite_object_instances_into(buffer, &instances, buf_w, buf_h, zoom, off_x, off_y, false);
    draw_links(buffer, links, LINK_RGB, buf_w, buf_h, zoom, off_x, off_y);
}

/// The object a click picks among `objects` (its layer key without the
/// selection's boost - else a selected object could never be clicked away
/// from in favour of the one under it - and whether it holds the clicked
/// pixel): the highest key, the later on a tie - the one drawn on top.
pub(crate) fn pick(objects: impl IntoIterator<Item = (u32, bool)>) -> Option<usize> {
    let mut best: Option<(usize, u32)> = None;
    for (i, (key, holds)) in objects.into_iter().enumerate() {
        if holds && best.is_none_or(|(_, k)| key >= k) {
            best = Some((i, key));
        }
    }
    best.map(|(i, _)| i)
}

/// `mask` (a bit per pixel of the inclusive `bbox`, row by row) holds the
/// image pixel `(x, y)`.
pub(crate) fn mask_holds(bbox: [u32; 4], mask: &BitSlice<u64, Lsb0>, x: u32, y: u32) -> bool {
    let [x0, y0, x1, y1] = bbox;
    if x < x0 || x > x1 || y < y0 || y > y1 {
        return false;
    }
    let width = (x1 - x0 + 1) as usize;
    let i = (y - y0) as usize * width + (x - x0) as usize;
    mask.get(i).is_some_and(|bit| *bit)
}

/// One object's pre-resolved render inputs. All project-level filtering and color
/// resolution (selection, class visibility, hide-unclassified) happens before this
/// is built, so [`composite_object_instances`] is pure pixel math with no project
/// access - that's what makes it cheap to unit test independently of the Slint/UI
/// plumbing in [`ViewportController::sync_objects_to_slint_viewport`].
pub(crate) struct ObjectDrawInstance<'a> {
    pub(crate) bbox: [u32; 4],
    pub(crate) mask: &'a bitvec::vec::BitVec<u64, bitvec::order::Lsb0>,
    pub(crate) color: slint::Rgba8Pixel,
}

/// The selected object, and its family (Relate Objects' links, design
/// canvas "Object relations in the results", board 4): its children light
/// blue, its parents yellow, the rest dimmed (a third of the opacity),
/// lines to them - in the editor and in Results' split view alike.
pub(crate) const SELECTED_RGB: [u8; 3] = [0xfc, 0xe9, 0x03];
pub(crate) const CHILD_RGB: [u8; 3] = [0x7c, 0xc4, 0xfa];
pub(crate) const PARENT_RGB: [u8; 3] = [0xfa, 0xcc, 0x15];
pub(crate) const DIMMED_RGB: [u8; 3] = [0x4a, 0x4d, 0x57];
pub(crate) const LINK_RGB: [u8; 3] = [0xf4, 0xf4, 0xf5];
/// Porter-Duff "over": blends `src` onto `dst` in place.
fn blend_pixel_over(dst: &mut slint::Rgba8Pixel, src: slint::Rgba8Pixel) {
    if dst.a == 0 || src.a == 255 {
        *dst = src;
        return;
    }
    let sa = src.a as f32 / 255.0;
    let da = dst.a as f32 / 255.0;
    let out_a = sa + da * (1.0 - sa);
    *dst = slint::Rgba8Pixel {
        r: ((src.r as f32 * sa + dst.r as f32 * da * (1.0 - sa)) / out_a) as u8,
        g: ((src.g as f32 * sa + dst.g as f32 * da * (1.0 - sa)) / out_a) as u8,
        b: ((src.b as f32 * sa + dst.b as f32 * da * (1.0 - sa)) / out_a) as u8,
        a: (out_a * 255.0) as u8,
    };
}

/// Composites pre-resolved instances (bottom -> top) into a viewport-sized RGBA buffer.
///
/// Pass 1 blends every instance's fill in order. Each instance is first rasterized
/// into a small scratch coverage buffer (its clipped screen footprint, padded by
/// 1px), so each screen pixel is blended once per object and the outline is derived
/// from the object's *own* footprint - independent of overlaps.
/// Pass 2 paints all outlines fully opaque in the same order, so an outline stays
/// visible even when another object lies on top of it.
///
/// Cost scales with visible, larger-than-a-pixel objects:
/// - off-screen bboxes are culled before touching their mask,
/// - bboxes within one screen pixel collapse to a single stamped pixel,
/// - only the visible sub-rectangle of each mask is walked,
/// - each mask row's runs of set pixels are filled at once (word-level
///   searches), not pixel by pixel.
#[cfg(test)]
fn composite_object_instances(
    instances: &[ObjectDrawInstance],
    buf_w: u32,
    buf_h: u32,
    zoom: f32,
    off_x: f32,
    off_y: f32,
) -> Vec<slint::Rgba8Pixel> {
    composite_object_instances_with(instances, buf_w, buf_h, zoom, off_x, off_y, false)
}

/// [`composite_object_instances`] - with `each_bit` every set mask pixel
/// on its own (the reference the tests check the runs against).
#[cfg(test)]
fn composite_object_instances_with(
    instances: &[ObjectDrawInstance],
    buf_w: u32,
    buf_h: u32,
    zoom: f32,
    off_x: f32,
    off_y: f32,
    each_bit: bool,
) -> Vec<slint::Rgba8Pixel> {
    let mut pixels = vec![
        slint::Rgba8Pixel {
            r: 0,
            g: 0,
            b: 0,
            a: 0
        };
        buf_w as usize * buf_h as usize
    ];
    composite_object_instances_into(
        &mut pixels,
        instances,
        buf_w,
        buf_h,
        zoom,
        off_x,
        off_y,
        each_bit,
    );
    pixels
}

/// [`composite_object_instances`] into `pixels` - a transparent
/// `buf_w * buf_h` buffer. `each_bit`: tests only (see
/// `composite_object_instances_with`).
#[allow(clippy::too_many_arguments)]
pub(crate) fn composite_object_instances_into(
    pixels: &mut [slint::Rgba8Pixel],
    instances: &[ObjectDrawInstance],
    buf_w: u32,
    buf_h: u32,
    zoom: f32,
    off_x: f32,
    off_y: f32,
    each_bit: bool,
) {
    let bw = buf_w as i32;
    let bh = buf_h as i32;
    let stride = buf_w as usize;
    debug_assert_eq!(pixels.len(), stride * buf_h as usize);
    if zoom <= 0.0 || buf_w == 0 || buf_h == 0 {
        return;
    }

    // Scratch buffers reused across instances (no per-instance allocation).
    let mut coverage: Vec<u8> = Vec::new();
    let mut x_edges: Vec<(i32, i32)> = Vec::new();
    // Outline pixels of all instances, plus (end offset, colour) per instance.
    let mut outline_idx: Vec<u32> = Vec::new();
    let mut outline_runs: Vec<(usize, slint::Rgba8Pixel)> = Vec::with_capacity(instances.len());

    for inst in instances {
        let bbox = inst.bbox;
        let bbox_w = (bbox[2] - bbox[0] + 1) as usize;
        let bbox_h = (bbox[3] - bbox[1] + 1) as usize;
        if bbox_w == 0 || bbox_h == 0 {
            continue;
        }

        // Screen footprint of the whole bbox (bbox[2]/[3] are inclusive).
        let sx0 = bbox[0] as f32 * zoom + off_x;
        let sy0 = bbox[1] as f32 * zoom + off_y;
        let sx1 = (bbox[2] + 1) as f32 * zoom + off_x;
        let sy1 = (bbox[3] + 1) as f32 * zoom + off_y;

        // Viewport culling.
        if sx1 <= 0.0 || sy1 <= 0.0 || sx0 >= buf_w as f32 || sy0 >= buf_h as f32 {
            continue;
        }

        // Sub-pixel collapse: one stamped pixel (isolated -> outline pixel).
        if sx1 - sx0 <= 1.0 && sy1 - sy0 <= 1.0 {
            let px = (((sx0 + sx1) * 0.5).max(0.0) as usize).min(stride - 1);
            let py = (((sy0 + sy1) * 0.5).max(0.0) as usize).min(buf_h as usize - 1);
            let i = py * stride + px;
            blend_pixel_over(&mut pixels[i], inst.color);
            outline_idx.push(i as u32);
            outline_runs.push((outline_idx.len(), inst.color));
            continue;
        }

        // Scratch region = footprint clipped to the viewport, padded by 1px so a
        // viewport-clipped object doesn't get a fake outline at the clip edge.
        let rx0 = (sx0.floor() as i32).max(-1);
        let ry0 = (sy0.floor() as i32).max(-1);
        let rx1 = ((sx1.floor() as i32) + 1).min(bw + 1);
        let ry1 = ((sy1.floor() as i32) + 1).min(bh + 1);
        if rx1 <= rx0 || ry1 <= ry0 {
            continue;
        }
        let rw = (rx1 - rx0) as usize;
        let rh = (ry1 - ry0) as usize;
        coverage.clear();
        coverage.resize(rw * rh, 0);

        // Visible sub-rectangle of the bbox in local image-pixel coordinates.
        let local_range = |r0: i32, r1: i32, off: f32, origin: u32, len: usize| {
            let a = (((r0 as f32 - off) / zoom).floor() as i64 - origin as i64 - 1)
                .clamp(0, len as i64) as usize;
            let b = (((r1 as f32 - off) / zoom).ceil() as i64 - origin as i64 + 1)
                .clamp(0, len as i64) as usize;
            (a, b)
        };
        let (lx0, lx1) = local_range(rx0, rx1, off_x, bbox[0], bbox_w);
        let (ly0, ly1) = local_range(ry0, ry1, off_y, bbox[1], bbox_h);

        // Per-column screen edges, computed once (shared edges => no gaps/overlap;
        // at least one pixel wide so tiny masks at low zoom don't develop holes).
        x_edges.clear();
        x_edges.extend((lx0..lx1).map(|lx| {
            let ax = (bbox[0] as usize + lx) as f32;
            let a = (ax * zoom + off_x).floor() as i32;
            let b = (((ax + 1.0) * zoom + off_x).floor() as i32).max(a + 1);
            (a.max(rx0), b.min(rx1))
        }));

        for ly in ly0..ly1 {
            let ay = (bbox[1] as usize + ly) as f32;
            let y0f = (ay * zoom + off_y).floor() as i32;
            let y1f = (((ay + 1.0) * zoom + off_y).floor() as i32).max(y0f + 1);
            let (y0, y1) = (y0f.max(ry0), y1f.min(ry1));
            if y0 >= y1 {
                continue;
            }
            let row_start = ly * bbox_w;
            let Some(row_bits) = inst.mask.get(row_start + lx0..row_start + lx1) else {
                continue;
            };
            if each_bit {
                for rel in row_bits.iter_ones() {
                    let (x0, x1) = x_edges[rel];
                    for py in (y0..y1).filter(|_| x0 < x1) {
                        let row = (py - ry0) as usize * rw;
                        coverage[row + (x0 - rx0) as usize..row + (x1 - rx0) as usize].fill(1);
                    }
                }
                continue;
            }
            // A run of set columns at once: their screen columns touch
            // (each column's right edge is at least the next one's left
            // edge, both rising), so they cover the first one's left edge
            // to the last one's right edge - one fill per screen row, not
            // one per pixel (an object without a mask is one run a row).
            let mut rel = 0;
            while let Some(start) = row_bits[rel..].first_one().map(|o| rel + o) {
                let end = row_bits[start..]
                    .first_zero()
                    .map_or(row_bits.len(), |z| start + z);
                rel = end;
                let (x0, x1) = (x_edges[start].0, x_edges[end - 1].1);
                if x0 >= x1 {
                    continue;
                }
                for py in y0..y1 {
                    let row = (py - ry0) as usize * rw;
                    coverage[row + (x0 - rx0) as usize..row + (x1 - rx0) as usize].fill(1);
                }
            }
        }

        // Resolve: blend the fill once per covered on-screen pixel and record
        // this object's own outline (covered pixel with an uncovered 4-neighbour).
        for sy in 0..rh {
            let py = ry0 + sy as i32;
            if py < 0 || py >= bh {
                continue;
            }
            for sx in 0..rw {
                let c = sy * rw + sx;
                if coverage[c] == 0 {
                    continue;
                }
                let px = rx0 + sx as i32;
                if px < 0 || px >= bw {
                    continue;
                }
                let i = py as usize * stride + px as usize;
                blend_pixel_over(&mut pixels[i], inst.color);

                let is_border = sx == 0
                    || sx + 1 == rw
                    || sy == 0
                    || sy + 1 == rh
                    || coverage[c - 1] == 0
                    || coverage[c + 1] == 0
                    || coverage[c - rw] == 0
                    || coverage[c + rw] == 0;
                if is_border {
                    outline_idx.push(i as u32);
                }
            }
        }
        outline_runs.push((outline_idx.len(), inst.color));
    }

    // Outline pass: opaque, in z-order, always on top of all fills.
    let mut start = 0;
    for (end, color) in outline_runs {
        let opaque = slint::Rgba8Pixel { a: 255, ..color };
        for &i in &outline_idx[start..end] {
            pixels[i as usize] = opaque;
        }
        start = end;
    }
}

/// Draw-order key; instances are painted in ascending key order (bigger = on top).
/// 0 = unclassified/unknown class, 1..=class_count = class stack (rank 0, the
/// first class in the list, gets the biggest value), then selected class, then
/// the selected object.
fn z_key(
    is_selected_object: bool,
    in_selected_class: bool,
    best_rank: Option<usize>,
    class_count: usize,
) -> u32 {
    if is_selected_object {
        u32::MAX
    } else if in_selected_class {
        u32::MAX - 1
    } else {
        best_rank.map_or(0, |r| (class_count - r.min(class_count - 1)) as u32)
    }
}
/// Resolved draw/pick priority for the current project state. Built once per
/// redraw or click, then queried per object (no per-object project access).
pub(crate) struct ZOrder {
    rank: HashMap<ObjectClass, usize>, // class -> index in class list (0 = on top)
    class_count: usize,
    selected_class: ObjectClass,
}

impl ZOrder {
    pub(crate) fn new<P: ProjectExt + ?Sized>(project: &P) -> Self {
        Self::of_classes(
            project.get_object_classes().iter().map(|c| c.global()),
            project.get_selected_object_class(),
        )
    }

    /// For `classes` in list order (the first on top), `selected_class` the
    /// selected one - e.g. an analysis' classes in Results' split view.
    pub(crate) fn of_classes(
        classes: impl IntoIterator<Item = ObjectClass>,
        selected_class: ObjectClass,
    ) -> Self {
        let rank: HashMap<ObjectClass, usize> = classes
            .into_iter()
            .enumerate()
            .map(|(i, c)| (c, i))
            .collect();
        Self {
            class_count: rank.len().max(1),
            rank,
            selected_class,
        }
    }

    /// Bigger = further on top.
    pub(crate) fn key(&self, classes: &HashSet<ObjectClass>, is_selected_object: bool) -> u32 {
        let in_selected_class = matches!(self.selected_class, ObjectClass::Global(_))
            && classes.contains(&self.selected_class);
        let best_rank = classes
            .iter()
            .filter_map(|c| self.rank.get(c))
            .min()
            .copied();
        z_key(
            is_selected_object,
            in_selected_class,
            best_rank,
            self.class_count,
        )
    }
}

/// Lines between image points (`links`, each from one point to another)
/// into `buffer` (`buf_w` × `buf_h`), image pixel `p` at `p * zoom + off` -
/// the editor's links between the selected object and its relatives.
pub(crate) fn draw_links(
    buffer: &mut [slint::Rgba8Pixel],
    links: &[Link],
    rgb: [u8; 3],
    buf_w: u32,
    buf_h: u32,
    zoom: f32,
    off_x: f32,
    off_y: f32,
) {
    let screen = |p: (f32, f32)| (p.0 * zoom + off_x, p.1 * zoom + off_y);
    let mut canvas = Canvas {
        buffer,
        w: buf_w as i64,
        h: buf_h as i64,
    };
    for (a, b) in links {
        canvas.line(screen(*a), screen(*b), rgb, LINK_ALPHA, true);
    }
}

/// A buffer to draw anti-aliased lines into.
pub(crate) struct Canvas<'a> {
    pub(crate) buffer: &'a mut [slint::Rgba8Pixel],
    pub(crate) w: i64,
    pub(crate) h: i64,
}

impl Canvas<'_> {
    fn plot(&mut self, x: i64, y: i64, rgb: [u8; 3], alpha: f32) {
        if x >= 0 && y >= 0 && x < self.w && y < self.h && alpha > 0.0 {
            blend(&mut self.buffer[(y * self.w + x) as usize], rgb, alpha);
        }
    }

    /// From `a` to `b` (screen pixels), anti-aliased (Xiaolin Wu) - `bold`:
    /// a second line beside it, about 2 px wide. Off screen: skipped.
    pub(crate) fn line(
        &mut self,
        a: (f32, f32),
        b: (f32, f32),
        rgb: [u8; 3],
        alpha: f32,
        bold: bool,
    ) {
        let (w, h) = (self.w as f32, self.h as f32);
        if a.0.max(b.0) < -1.0 || a.1.max(b.1) < -1.0 || a.0.min(b.0) > w || a.1.min(b.1) > h {
            return;
        }
        self.wu(a, b, rgb, alpha);
        if bold {
            // Beside it, across its direction.
            let steep = (b.1 - a.1).abs() > (b.0 - a.0).abs();
            let (dx, dy) = if steep { (1.0, 0.0) } else { (0.0, 1.0) };
            self.wu((a.0 + dx, a.1 + dy), (b.0 + dx, b.1 + dy), rgb, alpha);
        }
    }

    fn wu(&mut self, a: (f32, f32), b: (f32, f32), rgb: [u8; 3], alpha: f32) {
        let (mut x0, mut y0, mut x1, mut y1) = (a.0, a.1, b.0, b.1);
        let steep = (y1 - y0).abs() > (x1 - x0).abs();
        if steep {
            std::mem::swap(&mut x0, &mut y0);
            std::mem::swap(&mut x1, &mut y1);
        }
        if x0 > x1 {
            std::mem::swap(&mut x0, &mut x1);
            std::mem::swap(&mut y0, &mut y1);
        }
        let gradient = match x1 - x0 {
            d if d < 1e-6 => 0.0,
            d => (y1 - y0) / d,
        };
        // Only the columns on screen (a long segment zoomed in).
        let limit = if steep { self.h } else { self.w } as f32;
        let first = x0.round().max(-1.0);
        let last = x1.round().min(limit);
        let mut x = first;
        while x <= last {
            let y = y0 + gradient * (x - x0);
            let (yi, frac) = (y.floor(), y - y.floor());
            let (xi, yi) = (x as i64, yi as i64);
            if steep {
                self.plot(yi, xi, rgb, alpha * (1.0 - frac));
                self.plot(yi + 1, xi, rgb, alpha * frac);
            } else {
                self.plot(xi, yi, rgb, alpha * (1.0 - frac));
                self.plot(xi, yi + 1, rgb, alpha * frac);
            }
            x += 1.0;
        }
    }
}

/// `rgb` at `alpha` over `dst` (Porter-Duff over, straight alpha).
#[inline]
fn blend(dst: &mut slint::Rgba8Pixel, rgb: [u8; 3], alpha: f32) {
    let sa = alpha.clamp(0.0, 1.0);
    // Nothing there yet (most pixels): the colour itself.
    if dst.a == 0 {
        *dst = slint::Rgba8Pixel {
            r: rgb[0],
            g: rgb[1],
            b: rgb[2],
            a: (sa * 255.0).round() as u8,
        };
        return;
    }
    let da = dst.a as f32 / 255.0;
    let out_a = sa + da * (1.0 - sa);
    if out_a <= 0.0 {
        return;
    }
    let mix = |s: u8, d: u8| ((s as f32 * sa + d as f32 * da * (1.0 - sa)) / out_a).round() as u8;
    *dst = slint::Rgba8Pixel {
        r: mix(rgb[0], dst.r),
        g: mix(rgb[1], dst.g),
        b: mix(rgb[2], dst.b),
        a: (out_a * 255.0).round() as u8,
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rgba(r: u8, g: u8, b: u8, a: u8) -> slint::Rgba8Pixel {
        slint::Rgba8Pixel { r, g, b, a }
    }

    fn full_mask(bbox: [u32; 4]) -> BitVec<u64, Lsb0> {
        let w = (bbox[2] - bbox[0] + 1) as usize;
        let h = (bbox[3] - bbox[1] + 1) as usize;
        bitvec![u64, Lsb0; 1; w * h]
    }

    #[test]
    fn single_pixel_object_renders_at_correct_location_with_border_alpha() {
        // 3x3 bbox, only the centre bit set -> one isolated screen pixel.
        let bbox = [0u32, 0, 2, 2];
        let mut mask = bitvec![u64, Lsb0; 0; 9];
        mask.set(4, true); // local (1,1) in a 3-wide bbox
        let color = rgba(10, 20, 30, 128);
        let instances = [ObjectDrawInstance {
            bbox,
            mask: &mask,
            color,
        }];

        let pixels = composite_object_instances(&instances, 5, 5, 1.0, 0.0, 0.0);

        for y in 0..5usize {
            for x in 0..5usize {
                let i = y * 5 + x;
                if x == 1 && y == 1 {
                    // Isolated pixel: every neighbour is background, so it's a
                    // border pixel and alpha is forced to 255.
                    assert_eq!(pixels[i], rgba(10, 20, 30, 255), "at ({x},{y})");
                } else {
                    assert_eq!(pixels[i].a, 0, "expected background at ({x},{y})");
                }
            }
        }
    }

    #[test]
    fn zoomed_in_block_has_no_gaps_and_distinguishes_border_from_interior_alpha() {
        // 2x2 fully-set mask, zoom=2 -> a gapless 4x4 screen block at (2,2)-(6,6)
        // inside an 8x8 buffer, with a 2x2 interior that isn't a border.
        let bbox = [1u32, 1, 2, 2];
        let mask = full_mask(bbox);
        let color = rgba(200, 0, 0, 128);
        let instances = [ObjectDrawInstance {
            bbox,
            mask: &mask,
            color,
        }];

        let pixels = composite_object_instances(&instances, 8, 8, 2.0, 0.0, 0.0);

        for y in 0..8usize {
            for x in 0..8usize {
                let i = y * 8 + x;
                let in_block = (2..6).contains(&x) && (2..6).contains(&y);
                if !in_block {
                    assert_eq!(pixels[i].a, 0, "expected background at ({x},{y})");
                    continue;
                }
                let is_interior = (3..5).contains(&x) && (3..5).contains(&y);
                assert_eq!(pixels[i].r, 200, "at ({x},{y})");
                if is_interior {
                    // Not a border pixel: keeps the original blended alpha.
                    assert_eq!(pixels[i].a, 128, "interior alpha at ({x},{y})");
                } else {
                    assert_eq!(pixels[i].a, 255, "border alpha at ({x},{y})");
                }
            }
        }
    }

    #[test]
    fn object_entirely_outside_viewport_is_culled_and_produces_no_pixels() {
        let bbox = [1000u32, 1000, 1001, 1001];
        let mask = full_mask(bbox);
        let instances = [ObjectDrawInstance {
            bbox,
            mask: &mask,
            color: rgba(255, 255, 255, 255),
        }];

        let pixels = composite_object_instances(&instances, 8, 8, 1.0, 0.0, 0.0);

        assert!(pixels.iter().all(|p| p.a == 0));
    }

    #[test]
    fn object_straddling_the_viewport_edge_is_not_over_culled() {
        // 3x3 fully-set mask anchored at the origin, but the buffer is only 2x2:
        // the bbox extends past the buffer, yet still overlaps it and must not be
        // culled - the visible 2x2 corner should render fully.
        let bbox = [0u32, 0, 2, 2];
        let mask = full_mask(bbox);
        let instances = [ObjectDrawInstance {
            bbox,
            mask: &mask,
            color: rgba(1, 2, 3, 200),
        }];

        let pixels = composite_object_instances(&instances, 2, 2, 1.0, 0.0, 0.0);

        assert!(
            pixels.iter().all(|p| p.r == 1 && p.g == 2 && p.b == 3),
            "every pixel of the visible corner should be painted: {pixels:?}"
        );
    }

    #[test]
    fn far_zoomed_out_object_collapses_to_exactly_one_pixel_instead_of_vanishing() {
        // A small (2x2) object viewed at zoom=0.01: its screen footprint
        // (2 * 0.01 = 0.02px) is far below one pixel. Without the collapse path
        // this would either vanish (every mask pixel rounds into the same
        // sub-pixel slot that the old per-pixel clipping logic could drop) or
        // require iterating mask bits for no visual gain; with it, exactly one
        // pixel must be painted so the object doesn't silently disappear.
        let bbox = [100u32, 100, 101, 101];
        let mask = full_mask(bbox);
        let color = rgba(50, 60, 70, 222);
        let instances = [ObjectDrawInstance {
            bbox,
            mask: &mask,
            color,
        }];

        let pixels = composite_object_instances(&instances, 50, 50, 0.01, 0.0, 0.0);

        let painted: Vec<_> = pixels.iter().filter(|p| p.a != 0).collect();
        assert_eq!(
            painted.len(),
            1,
            "expected exactly one painted pixel, got {painted:?}"
        );
        assert_eq!(painted[0].r, 50);
        assert_eq!(painted[0].g, 60);
        assert_eq!(painted[0].b, 70);
        // Isolated single pixel -> border pass forces full opacity.
        assert_eq!(painted[0].a, 255);
    }

    #[test]
    fn overlapping_objects_blend_with_porter_duff_over() {
        let bbox = [0u32, 0, 0, 0]; // single-pixel bbox
        let mask = full_mask(bbox);
        let bottom = rgba(255, 0, 0, 128);
        let top = rgba(0, 0, 255, 128);
        let instances = [
            ObjectDrawInstance {
                bbox,
                mask: &mask,
                color: bottom,
            },
            ObjectDrawInstance {
                bbox,
                mask: &mask,
                color: top,
            },
        ];

        let pixels = composite_object_instances(&instances, 3, 3, 1.0, 0.0, 0.0);

        // A single pixel is all outline: outlines are painted opaque in
        // z-order after every fill, so the top object's outline wins.
        assert_eq!(pixels[0], rgba(0, 0, 255, 255));
    }

    /// The runs of set pixels are only a shortcut: for any zoom and offset
    /// they must give exactly the pixels of walking every set bit.
    #[test]
    fn runs_produce_the_same_pixels_as_walking_every_bit() {
        let mut seed = 42u64;
        let mut rnd = |m: u32| {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((seed >> 33) as u32) % m
        };
        let mut masks = Vec::new();
        let mut bboxes = Vec::new();
        for _ in 0..60 {
            let (w, h) = (1 + rnd(40), 1 + rnd(40));
            let (x, y) = (rnd(300), rnd(300));
            bboxes.push([x, y, x + w - 1, y + h - 1]);
            // Random fill, including empty and single-pixel masks.
            let density = rnd(4);
            let mask: BitVec<u64, Lsb0> = (0..w * h).map(|_| rnd(4) < density).collect();
            masks.push(mask);
        }
        let instances: Vec<ObjectDrawInstance> = bboxes
            .iter()
            .zip(&masks)
            .enumerate()
            .map(|(i, (bbox, mask))| ObjectDrawInstance {
                bbox: *bbox,
                mask,
                color: rgba((i * 37) as u8, (i * 91) as u8, 200, 100 + (i % 100) as u8),
            })
            .collect();

        for &zoom in &[0.07f32, 0.23, 0.5, 0.77, 0.999, 1.0, 1.5, 3.3] {
            for &(off_x, off_y) in &[(0.0f32, 0.0f32), (-17.3, 5.6), (11.9, -40.2)] {
                let bits =
                    composite_object_instances_with(&instances, 160, 120, zoom, off_x, off_y, true);
                let runs = composite_object_instances_with(
                    &instances, 160, 120, zoom, off_x, off_y, false,
                );
                assert!(
                    bits == runs,
                    "zoom {zoom}, offset ({off_x}, {off_y}): runs differ from bits"
                );
            }
        }
    }

    #[test]
    fn overlapping_interiors_blend_with_porter_duff_over_and_the_top_outline_wins() {
        let bbox = [1u32, 1, 3, 3]; // 3x3 -> the centre (2,2) is interior
        let mask = full_mask(bbox);
        let bottom = rgba(255, 0, 0, 128);
        let top = rgba(0, 0, 255, 128);
        let instances = [
            ObjectDrawInstance {
                bbox,
                mask: &mask,
                color: bottom,
            },
            ObjectDrawInstance {
                bbox,
                mask: &mask,
                color: top,
            },
        ];

        let pixels = composite_object_instances(&instances, 5, 5, 1.0, 0.0, 0.0);

        let mut expected = bottom;
        blend_pixel_over(&mut expected, top);
        assert_eq!(pixels[2 * 5 + 2], expected, "interior: both fills blended");
        assert_eq!(pixels[5 + 1], rgba(0, 0, 255, 255), "outline: top, opaque");
        assert_eq!(pixels[0].a, 0, "background stays empty");
    }

    #[test]
    fn a_link_is_drawn_between_its_two_points() {
        let mut buf = vec![rgba(0, 0, 0, 0); 100 * 20];
        draw_links(
            &mut buf,
            &[((5.0, 10.0), (35.0, 10.0))],
            [255, 0, 0],
            100,
            20,
            1.0,
            0.0,
            0.0,
        );
        assert!(buf[10 * 100 + 20].a > 0, "on the line");
        assert_eq!(buf[2 * 100 + 20].a, 0, "not beside it");
    }

    /// A click picks the highest layer holding the point, the later one
    /// on a tie - none when nothing holds it.
    #[test]
    fn a_click_picks_the_object_drawn_on_top() {
        assert_eq!(pick([(2, true), (5, true), (1, true)]), Some(1));
        assert_eq!(pick([(5, true), (5, true)]), Some(1), "a tie: the later");
        assert_eq!(pick([(9, false), (0, true)]), Some(1), "not holding it");
        assert_eq!(pick([(0, false)]), None);
        let ring = bitvec![u64, Lsb0; 1, 1, 1, 1, 0, 1, 1, 1, 1];
        assert!(mask_holds([10, 10, 12, 12], &ring, 10, 10));
        assert!(!mask_holds([10, 10, 12, 12], &ring, 11, 11), "its hole");
        assert!(!mask_holds([10, 10, 12, 12], &ring, 13, 10), "beside it");
    }

    /// Benchmark (not run by default): `BENCH_OBJECTS` (1045) round
    /// objects of `BENCH_SIZE` (20) px on a 2048² image, drawn fitted
    /// (zoom 0.47) and at zoom 1 into the overlay's 2432 × 1592 buffer.
    /// `cargo test --release -p evanalyzer_gui --lib bench_draw_objects -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn bench_draw_objects() {
        let env = |key: &str, default: usize| {
            std::env::var(key)
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(default)
        };
        let (n, size) = (env("BENCH_OBJECTS", 1045), env("BENCH_SIZE", 20) as u32);
        let r = size as f32 / 2.0;
        let masks: Vec<([u32; 4], BitVec<u64, Lsb0>)> = (0..n as u32)
            .map(|i| {
                let (x, y) = ((i * 7919) % (2048 - size), (i * 104_729) % (2048 - size));
                let mut mask = BitVec::<u64, Lsb0>::repeat(false, (size * size) as usize);
                for py in 0..size {
                    for px in 0..size {
                        let (dx, dy) = (px as f32 + 0.5 - r, py as f32 + 0.5 - r);
                        mask.set((py * size + px) as usize, dx * dx + dy * dy <= r * r);
                    }
                }
                ([x, y, x + size - 1, y + size - 1], mask)
            })
            .collect();
        let (w, h) = (2432u32, 1592u32);
        for (name, zoom) in [("fitted", 0.47f32), ("zoom 1", 1.0)] {
            let (mut best, mut first) = (f64::MAX, None);
            for _ in 0..20 {
                let mut buf = vec![rgba(0, 0, 0, 0); (w * h) as usize];
                let objects = masks
                    .iter()
                    .map(|(bbox, mask)| {
                        let instance = ObjectDrawInstance {
                            bbox: *bbox,
                            mask,
                            color: rgba(255, 0, 0, 200),
                        };
                        (1, instance)
                    })
                    .collect();
                let start = std::time::Instant::now();
                draw_objects(&mut buf, objects, &[], w, h, zoom, 256.0, 256.0);
                let took = start.elapsed().as_secs_f64() * 1000.0;
                first.get_or_insert(took);
                best = best.min(took);
            }
            println!(
                "draw {n} objects of {size} px, {name}: {best:.2} ms (best of 20), first {:.2} ms",
                first.unwrap_or(0.0)
            );
        }
    }
}
