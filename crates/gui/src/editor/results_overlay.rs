//! The results' split view on the image: every object of the analysis on
//! the shown image as an ellipse in its box (the results keep boxes, not
//! outlines), in its class's color or the value's heatmap color. While one
//! is selected the others are dimmed. Drawn instead of the project's
//! objects (see `ViewportController::sync_objects_to_slint_viewport`).

/// The objects to draw, in image pixels.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ResultsOverlay {
    /// `[xmin, ymin, xmax, ymax]` (inclusive) and RGB.
    pub objects: Vec<([u32; 4], [u8; 3])>,
    /// Index into `objects`.
    pub selected: Option<usize>,
}

/// Outline opacity; the inside gets a faint tint, dimmed objects a third.
const OUTLINE_ALPHA: f32 = 0.95;
const FILL_ALPHA: f32 = 0.12;
const DIMMED: f32 = 0.3;

/// Draws `overlay` into `buffer` (`buf_w` × `buf_h`, transparent before),
/// image pixel `p` at `p * zoom + off` on it.
pub fn draw_results_overlay(
    buffer: &mut [slint::Rgba8Pixel],
    overlay: &ResultsOverlay,
    buf_w: u32,
    buf_h: u32,
    zoom: f32,
    off_x: f32,
    off_y: f32,
) {
    let (w, h) = (buf_w as i64, buf_h as i64);
    for (i, (bbox, rgb)) in overlay.objects.iter().enumerate() {
        let dim = if overlay.selected.is_some_and(|s| s != i) {
            DIMMED
        } else {
            1.0
        };
        // The box's outer edges on screen (`xmax` is inclusive).
        let x0 = bbox[0] as f32 * zoom + off_x;
        let y0 = bbox[1] as f32 * zoom + off_y;
        let x1 = (bbox[2] + 1) as f32 * zoom + off_x;
        let y1 = (bbox[3] + 1) as f32 * zoom + off_y;
        if x1 < 0.0 || y1 < 0.0 || x0 >= w as f32 || y0 >= h as f32 {
            continue;
        }
        let (cx, cy) = ((x0 + x1) / 2.0, (y0 + y1) / 2.0);
        let (rx, ry) = ((x1 - x0) / 2.0, (y1 - y0) / 2.0);
        // Too small for a ring: a dot, so it doesn't vanish.
        if rx < 2.0 || ry < 2.0 {
            let (px, py) = (cx as i64, cy as i64);
            for y in py - 1..=py {
                for x in px - 1..=px {
                    if x >= 0 && y >= 0 && x < w && y < h {
                        blend(&mut buffer[(y * w + x) as usize], *rgb, OUTLINE_ALPHA * dim);
                    }
                }
            }
            continue;
        }
        // Pixels whose centre lies within ~0.75 px of the ellipse are its
        // outline (soft at the edges), inside it the tint.
        let xs = (x0.floor() as i64).max(0)..(x1.ceil() as i64).min(w);
        let ys = (y0.floor() as i64).max(0)..(y1.ceil() as i64).min(h);
        let r = rx.min(ry);
        for y in ys {
            let dy = (y as f32 + 0.5 - cy) / ry;
            for x in xs.clone() {
                let dx = (x as f32 + 0.5 - cx) / rx;
                let d = (dx * dx + dy * dy).sqrt();
                // Distance to the ellipse in screen pixels (close enough).
                let dist = (d - 1.0) * r;
                let alpha = if dist.abs() < 1.5 {
                    OUTLINE_ALPHA * (1.0 - (dist.abs() - 0.75).max(0.0) / 0.75)
                } else if dist < 0.0 {
                    FILL_ALPHA
                } else {
                    continue;
                };
                blend(&mut buffer[(y * w + x) as usize], *rgb, alpha * dim);
            }
        }
    }
}

/// `rgb` at `alpha` over `dst` (Porter-Duff over, straight alpha).
fn blend(dst: &mut slint::Rgba8Pixel, rgb: [u8; 3], alpha: f32) {
    let sa = alpha.clamp(0.0, 1.0);
    let da = dst.a as f32 / 255.0;
    let out_a = sa + da * (1.0 - sa);
    if out_a <= 0.0 {
        return;
    }
    let mix = |s: u8, d: u8| {
        ((s as f32 * sa + d as f32 * da * (1.0 - sa)) / out_a).round() as u8
    };
    *dst = slint::Rgba8Pixel {
        r: mix(rgb[0], dst.r),
        g: mix(rgb[1], dst.g),
        b: mix(rgb[2], dst.b),
        a: (out_a * 255.0).round() as u8,
    };
}

/// The view that shows `bbox` (image pixels, inclusive) in a `v_w` × `v_h`
/// viewport: centred, the object about a quarter of the view - never
/// smaller than the whole image fitted, never beyond 4×. As
/// `(zoom, offset_x, offset_y)`.
pub fn focus_view(
    v_w: f32,
    v_h: f32,
    image_w: f32,
    image_h: f32,
    bbox: [f32; 4],
) -> (f32, f32, f32) {
    let fit = (v_w / image_w.max(1.0)).min(v_h / image_h.max(1.0)) * 0.95;
    let (bw, bh) = ((bbox[2] - bbox[0] + 1.0).max(1.0), (bbox[3] - bbox[1] + 1.0).max(1.0));
    let zoom = (v_w / (bw * 4.0)).min(v_h / (bh * 4.0)).min(4.0).max(fit);
    let (cx, cy) = ((bbox[0] + bbox[2] + 1.0) / 2.0, (bbox[1] + bbox[3] + 1.0) / 2.0);
    (zoom, v_w / 2.0 - cx * zoom, v_h / 2.0 - cy * zoom)
}

/// The object under the screen point `(x, y)`: the smallest whose ellipse
/// holds it, else the nearest within 6 px of its box.
pub fn object_at(overlay: &ResultsOverlay, x: f32, y: f32, zoom: f32, off_x: f32, off_y: f32) -> Option<usize> {
    let ix = (x - off_x) / zoom;
    let iy = (y - off_y) / zoom;
    let slack = 6.0 / zoom;
    let mut best: Option<(usize, f32)> = None;
    for (i, (bbox, _)) in overlay.objects.iter().enumerate() {
        let (x0, y0) = (bbox[0] as f32, bbox[1] as f32);
        let (x1, y1) = ((bbox[2] + 1) as f32, (bbox[3] + 1) as f32);
        let (rx, ry) = ((x1 - x0) / 2.0, (y1 - y0) / 2.0);
        let (dx, dy) = ((ix - (x0 + x1) / 2.0) / rx, (iy - (y0 + y1) / 2.0) / ry);
        // Inside: ranked by size. Near the box: by distance, after any inside.
        let score = if dx * dx + dy * dy <= 1.0 {
            rx * ry
        } else {
            let gx = (x0 - ix).max(ix - x1).max(0.0);
            let gy = (y0 - iy).max(iy - y1).max(0.0);
            let gap = (gx * gx + gy * gy).sqrt();
            if gap > slack {
                continue;
            }
            f32::MAX / 2.0 + gap
        };
        if best.is_none_or(|(_, s)| score < s) {
            best = Some((i, score));
        }
    }
    best.map(|(i, _)| i)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn buffer(w: u32, h: u32) -> Vec<slint::Rgba8Pixel> {
        vec![slint::Rgba8Pixel { r: 0, g: 0, b: 0, a: 0 }; (w * h) as usize]
    }

    #[test]
    fn an_object_is_a_ring_with_a_faint_inside() {
        let overlay = ResultsOverlay {
            objects: vec![([10, 10, 29, 29], [255, 0, 0])],
            selected: None,
        };
        let mut buf = buffer(40, 40);
        draw_results_overlay(&mut buf, &overlay, 40, 40, 1.0, 0.0, 0.0);
        let at = |x: usize, y: usize| buf[y * 40 + x];
        assert!(at(10, 20).a > 200, "left edge of the ring");
        assert!(at(20, 20).a < 40 && at(20, 20).a > 0, "the faint inside");
        assert_eq!(at(10, 10).a, 0, "the box's corner is outside the ellipse");
        assert_eq!(at(20, 20).r, 255);
    }

    #[test]
    fn others_are_dimmed_while_one_is_selected() {
        let overlay = ResultsOverlay {
            objects: vec![([0, 0, 9, 9], [255, 0, 0]), ([20, 0, 29, 9], [0, 255, 0])],
            selected: Some(1),
        };
        let mut buf = buffer(30, 10);
        draw_results_overlay(&mut buf, &overlay, 30, 10, 1.0, 0.0, 0.0);
        assert!(buf[5 * 30].a < buf[5 * 30 + 20].a, "the dimmed one is fainter");
    }

    #[test]
    fn a_tiny_or_far_object_is_still_a_dot_and_off_screen_ones_are_skipped() {
        let overlay = ResultsOverlay {
            objects: vec![([100, 100, 101, 101], [0, 0, 255]), ([5000, 5000, 5100, 5100], [0, 0, 255])],
            selected: None,
        };
        let mut buf = buffer(20, 20);
        draw_results_overlay(&mut buf, &overlay, 20, 20, 0.1, 0.0, 0.0);
        assert!(buf[10 * 20 + 10].a > 0 || buf[9 * 20 + 9].a > 0, "the dot");
        assert_eq!(buf.iter().filter(|p| p.a > 0).count(), 4, "only the dot");
    }

    #[test]
    fn the_focus_centres_the_object_and_keeps_within_fit_and_4x() {
        let (zoom, ox, oy) = focus_view(800.0, 600.0, 10_000.0, 10_000.0, [4990.0, 4990.0, 5009.0, 5009.0]);
        assert_eq!(zoom, 4.0, "a small object: at most 4×");
        assert!((ox + 5000.0 * zoom - 400.0).abs() < 0.01);
        assert!((oy + 5000.0 * zoom - 300.0).abs() < 0.01);
        let (zoom, ..) = focus_view(800.0, 600.0, 1000.0, 1000.0, [0.0, 0.0, 999.0, 999.0]);
        assert!((zoom - 0.6 * 0.95).abs() < 1e-5, "a huge object: the whole image fitted");
    }

    #[test]
    fn a_click_picks_the_smallest_object_under_it_or_the_nearest() {
        let overlay = ResultsOverlay {
            objects: vec![([0, 0, 99, 99], [0; 3]), ([40, 40, 59, 59], [0; 3]), ([200, 0, 209, 9], [0; 3])],
            selected: None,
        };
        assert_eq!(object_at(&overlay, 50.0, 50.0, 1.0, 0.0, 0.0), Some(1));
        assert_eq!(object_at(&overlay, 10.0, 50.0, 1.0, 0.0, 0.0), Some(0));
        assert_eq!(object_at(&overlay, 213.0, 5.0, 1.0, 0.0, 0.0), Some(2), "4 px beside it");
        assert_eq!(object_at(&overlay, 150.0, 150.0, 1.0, 0.0, 0.0), None);
    }
}
