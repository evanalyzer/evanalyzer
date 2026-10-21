//! The results' split view on the image: every object of the analysis on
//! the shown image as an ellipse in its box (the results keep boxes, not
//! outlines), in its class's color or the value's heatmap color. The
//! selected one is drawn on top and bolder (the viewer's yellow crosshair
//! points at it); the others are dimmed meanwhile. Drawn instead of the project's
//! objects (see `ViewportController::sync_objects_to_slint_viewport`).
//! With tracks (Results › Tracks) each track's path up to the time point
//! shown goes under them as a trail, fading into the past.

use evanalyzer_app::results::ImageTracks;
use std::sync::Arc;

/// The objects to draw, in image pixels.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ResultsOverlay {
    /// `[xmin, ymin, xmax, ymax]` (inclusive) and RGB.
    pub objects: Vec<([u32; 4], [u8; 3])>,
    /// Index into `objects`.
    pub selected: Option<usize>,
    /// The tracks' trails, under the objects.
    pub trails: Option<Arc<TrailsOverlay>>,
}

/// The image's tracks as trails up to time point `t`.
#[derive(Debug, Clone, PartialEq)]
pub struct TrailsOverlay {
    pub tracks: Arc<ImageTracks>,
    /// RGB per track (`tracks.tracks`' order).
    pub colors: Vec<[u8; 3]>,
    /// The time point shown: a trail ends there.
    pub t: u32,
    /// Time points back a trail reaches - `None`: from the track's start.
    pub frames: Option<u32>,
    /// Index into `tracks.tracks`: its whole path faint, its trail white.
    pub selected: Option<usize>,
    /// Only the selected track's.
    pub only_selected: bool,
}

/// A trail's opacity at its newest end; its oldest end a quarter of it.
const TRAIL_ALPHA: f32 = 0.9;
/// The selected track's whole path, behind its trail.
const GHOST_ALPHA: f32 = 0.22;
const SELECTED_RGB: [u8; 3] = [255, 255, 255];

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
    if let Some(trails) = &overlay.trails {
        draw_trails(buffer, trails, buf_w, buf_h, zoom, off_x, off_y);
    }
    // The selected one last: on top.
    let order = (0..overlay.objects.len())
        .filter(|i| Some(*i) != overlay.selected)
        .chain(overlay.selected.filter(|s| *s < overlay.objects.len()));
    for i in order {
        let (bbox, rgb) = &overlay.objects[i];
        let selected = overlay.selected == Some(i);
        let dim = if overlay.selected.is_some() && !selected {
            DIMMED
        } else {
            1.0
        };
        // Half the outline's width.
        let half = if selected { 1.25 } else { 0.75 };
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
        // Pixels whose centre lies within ~`half` px of the ellipse are its
        // outline (soft at the edges), inside it the tint. Row by row: the
        // inside, well away from the outline, is one run (no square root
        // per pixel); only near the outline is each pixel's distance taken.
        let xs = (x0.floor() as i64).max(0)..(x1.ceil() as i64).min(w);
        let ys = (y0.floor() as i64).max(0)..(y1.ceil() as i64).min(h);
        let r = rx.min(ry);
        let band = 2.0 * half;
        for y in ys {
            let dy = (y as f32 + 0.5 - cy) / ry;
            let row = &mut buffer[(y * w) as usize..((y + 1) * w) as usize];
            // Where this row is well inside (`dist < -band`, i.e. d below
            // `a`): |x - cx| below `inner`, a pixel short to be safe.
            let a = 1.0 - band / r;
            let inner = if a > 0.0 && dy.abs() < a {
                (rx * (a * a - dy * dy).sqrt() - 1.0).max(0.0)
            } else {
                0.0
            };
            let (in0, in1) = if inner > 0.0 {
                (
                    ((cx - inner).ceil() as i64).max(xs.start),
                    ((cx + inner).floor() as i64).min(xs.end),
                )
            } else {
                (xs.start, xs.start)
            };
            let edge = |x: i64, row: &mut [slint::Rgba8Pixel]| {
                let dx = (x as f32 + 0.5 - cx) / rx;
                let d = (dx * dx + dy * dy).sqrt();
                // Distance to the ellipse in screen pixels (close enough).
                let dist = (d - 1.0) * r;
                let alpha = if dist.abs() < band {
                    OUTLINE_ALPHA * (1.0 - (dist.abs() - half).max(0.0) / half)
                } else if dist < 0.0 {
                    FILL_ALPHA
                } else {
                    return;
                };
                blend(&mut row[x as usize], *rgb, alpha * dim);
            };
            for x in xs.start..in0.max(xs.start) {
                edge(x, row);
            }
            for px in &mut row[in0.max(xs.start) as usize..in1.max(in0).max(xs.start) as usize] {
                blend(px, *rgb, FILL_ALPHA * dim);
            }
            for x in in1.max(in0).max(xs.start)..xs.end {
                edge(x, row);
            }
        }
    }
}

/// The trails of `trails` into `buffer`, as `draw_results_overlay` places
/// image pixels. Each segment from one time point to the next; the older
/// it is, the fainter.
fn draw_trails(
    buffer: &mut [slint::Rgba8Pixel],
    trails: &TrailsOverlay,
    buf_w: u32,
    buf_h: u32,
    zoom: f32,
    off_x: f32,
    off_y: f32,
) {
    let screen = |p: &evanalyzer_app::results::TrackPoint| (p.x * zoom + off_x, p.y * zoom + off_y);
    let from = trails.frames.map(|n| trails.t.saturating_sub(n));
    let span = trails.frames.unwrap_or(trails.t).max(1) as f32;
    let mut canvas = Canvas {
        buffer,
        w: buf_w as i64,
        h: buf_h as i64,
    };
    // The selected one last: on top.
    let order = (0..trails.tracks.tracks.len())
        .filter(|i| Some(*i) != trails.selected && !trails.only_selected)
        .chain(trails.selected.filter(|s| *s < trails.tracks.tracks.len()));
    for i in order {
        let track = &trails.tracks.tracks[i];
        let selected = trails.selected == Some(i);
        let rgb = match selected {
            true => SELECTED_RGB,
            false => trails.colors.get(i).copied().unwrap_or(SELECTED_RGB),
        };
        // Its whole path, faint: where it comes from and goes.
        if selected {
            for pair in track.points.windows(2) {
                canvas.line(screen(&pair[0]), screen(&pair[1]), rgb, GHOST_ALPHA, false);
            }
        }
        // Points are by time: the trail's from its first within reach to
        // the time point shown.
        let start = from.map_or(0, |from| track.points.partition_point(|p| p.t < from));
        let end = track.points.partition_point(|p| p.t <= trails.t);
        if end < start + 2 {
            continue;
        }
        for pair in track.points[start..end].windows(2) {
            let age = trails.t.saturating_sub(pair[1].t) as f32 / span;
            let alpha = TRAIL_ALPHA * (1.0 - 0.75 * age.min(1.0));
            canvas.line(screen(&pair[0]), screen(&pair[1]), rgb, alpha, selected);
        }
    }
}

/// A buffer to draw anti-aliased lines into.
struct Canvas<'a> {
    buffer: &'a mut [slint::Rgba8Pixel],
    w: i64,
    h: i64,
}

impl Canvas<'_> {
    fn plot(&mut self, x: i64, y: i64, rgb: [u8; 3], alpha: f32) {
        if x >= 0 && y >= 0 && x < self.w && y < self.h && alpha > 0.0 {
            blend(&mut self.buffer[(y * self.w + x) as usize], rgb, alpha);
        }
    }

    /// From `a` to `b` (screen pixels), anti-aliased (Xiaolin Wu) - `bold`:
    /// a second line beside it, about 2 px wide. Off screen: skipped.
    fn line(&mut self, a: (f32, f32), b: (f32, f32), rgb: [u8; 3], alpha: f32, bold: bool) {
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

/// The view that shows `bbox` (image pixels, inclusive) in a `v_w` × `v_h`
/// viewport: centred, the object about a sixth of the view (its
/// surroundings in view too) - never smaller than the whole image fitted,
/// never beyond 2.5×. As `(zoom, offset_x, offset_y)`.
pub fn focus_view(
    v_w: f32,
    v_h: f32,
    image_w: f32,
    image_h: f32,
    bbox: [f32; 4],
) -> (f32, f32, f32) {
    let fit = (v_w / image_w.max(1.0)).min(v_h / image_h.max(1.0)) * 0.95;
    let (bw, bh) = (
        (bbox[2] - bbox[0] + 1.0).max(1.0),
        (bbox[3] - bbox[1] + 1.0).max(1.0),
    );
    let zoom = (v_w / (bw * 6.0)).min(v_h / (bh * 6.0)).min(2.5).max(fit);
    let (cx, cy) = (
        (bbox[0] + bbox[2] + 1.0) / 2.0,
        (bbox[1] + bbox[3] + 1.0) / 2.0,
    );
    (zoom, v_w / 2.0 - cx * zoom, v_h / 2.0 - cy * zoom)
}

/// The object under the screen point `(x, y)`: the smallest whose ellipse
/// holds it, else the nearest within 6 px of its box.
pub fn object_at(
    overlay: &ResultsOverlay,
    x: f32,
    y: f32,
    zoom: f32,
    off_x: f32,
    off_y: f32,
) -> Option<usize> {
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
        vec![
            slint::Rgba8Pixel {
                r: 0,
                g: 0,
                b: 0,
                a: 0
            };
            (w * h) as usize
        ]
    }

    /// Two tracks along rows y = 10 and y = 30 of a 100 × 40 buffer, a
    /// point every 10 px from t = 0 (x = 0) to t = 9 (x = 90).
    fn trails(
        t: u32,
        frames: Option<u32>,
        selected: Option<usize>,
        only_selected: bool,
    ) -> ResultsOverlay {
        use evanalyzer_app::results::{TrackPath, TrackPoint};
        let track = |id: u64, y: f32| TrackPath {
            track_id: id,
            class_id: 1,
            points: (0..10)
                .map(|t| TrackPoint {
                    t,
                    x: t as f32 * 10.0,
                    y,
                    area_px: 1.0,
                })
                .collect(),
        };
        ResultsOverlay {
            objects: Vec::new(),
            selected: None,
            trails: Some(Arc::new(TrailsOverlay {
                tracks: Arc::new(ImageTracks {
                    tracks: vec![track(1, 10.0), track(2, 30.0)],
                    total_tracks: 2,
                    ..ImageTracks::default()
                }),
                colors: vec![[255, 0, 0], [0, 0, 255]],
                t,
                frames,
                selected,
                only_selected,
            })),
        }
    }

    fn alpha_at(buf: &[slint::Rgba8Pixel], x: u32, y: u32) -> u8 {
        buf[(y * 100 + x) as usize].a
    }

    /// Benchmark (not run by default): 1500 tracks of 48 time points on a
    /// 2000 × 1500 image, drawn at 1920 × 1080 - trails of 10 time points,
    /// whole tracks, the selected one alone.
    /// `cargo test --release -p evanalyzer_gui --lib bench_trails -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn bench_trails() {
        use evanalyzer_app::results::{TrackPath, TrackPoint};
        let mut seed = 1u64;
        let mut rnd = || {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (seed >> 33) as f32 / (1u64 << 31) as f32
        };
        let tracks: Vec<TrackPath> = (0..1500)
            .map(|id| {
                let (mut x, mut y) = (rnd() * 2000.0, rnd() * 1500.0);
                TrackPath {
                    track_id: id + 1,
                    class_id: 1,
                    points: (0..48)
                        .map(|t| {
                            x += (rnd() - 0.5) * 12.0;
                            y += (rnd() - 0.5) * 12.0;
                            TrackPoint {
                                t,
                                x,
                                y,
                                area_px: 100.0,
                            }
                        })
                        .collect(),
                }
            })
            .collect();
        let tracks = Arc::new(ImageTracks {
            total_tracks: tracks.len(),
            tracks,
            ..ImageTracks::default()
        });
        let overlay = |frames, selected, only_selected| ResultsOverlay {
            objects: Vec::new(),
            selected: None,
            trails: Some(Arc::new(TrailsOverlay {
                tracks: tracks.clone(),
                colors: vec![[90, 160, 255]; 1500],
                t: 30,
                frames,
                selected,
                only_selected,
            })),
        };
        for (label, overlay) in [
            (
                "trails of 10 time points",
                overlay(Some(10), Some(5), false),
            ),
            ("whole tracks up to t", overlay(None, Some(5), false)),
            ("the selected track only", overlay(Some(10), Some(5), true)),
        ] {
            let mut times = Vec::new();
            for _ in 0..15 {
                let mut buf = buffer(1920, 1080);
                let start = std::time::Instant::now();
                draw_results_overlay(&mut buf, &overlay, 1920, 1080, 0.9, 0.0, 0.0);
                times.push(start.elapsed().as_secs_f64() * 1000.0);
            }
            times.sort_by(f64::total_cmp);
            println!("{label:<28} {:>7.2} ms", times[times.len() / 2]);
        }
    }

    /// A trail ends at the time point shown; the older part is fainter.
    #[test]
    fn a_trail_reaches_the_time_point_shown_and_fades_back() {
        let mut buf = buffer(100, 40);
        draw_results_overlay(
            &mut buf,
            &trails(5, None, None, false),
            100,
            40,
            1.0,
            0.0,
            0.0,
        );
        let (old, new) = (alpha_at(&buf, 5, 10), alpha_at(&buf, 45, 10));
        assert!(old > 0 && new > old, "old {old}, new {new}");
        assert_eq!(buf[(10 * 100 + 45) as usize].r, 255);
        assert_eq!(alpha_at(&buf, 70, 10), 0, "not beyond t = 5");
        assert!(alpha_at(&buf, 45, 30) > 0, "the other track too");
    }

    /// `frames`: only that many time points back.
    #[test]
    fn a_trail_of_n_frames_starts_n_back() {
        let mut buf = buffer(100, 40);
        draw_results_overlay(
            &mut buf,
            &trails(8, Some(3), None, false),
            100,
            40,
            1.0,
            0.0,
            0.0,
        );
        assert_eq!(alpha_at(&buf, 35, 10), 0, "before t = 5");
        assert!(alpha_at(&buf, 65, 10) > 0);
    }

    /// The selected track: white, its whole path faint behind - and alone
    /// with "only selected".
    #[test]
    fn the_selected_track_is_white_with_its_whole_path() {
        let mut buf = buffer(100, 40);
        draw_results_overlay(
            &mut buf,
            &trails(5, None, Some(0), true),
            100,
            40,
            1.0,
            0.0,
            0.0,
        );
        let pixel = buf[(10 * 100 + 45) as usize];
        assert_eq!((pixel.r, pixel.g, pixel.b), (255, 255, 255));
        assert!(alpha_at(&buf, 75, 10) > 0, "its path ahead, faint");
        assert!(alpha_at(&buf, 75, 10) < alpha_at(&buf, 45, 10));
        assert_eq!(alpha_at(&buf, 45, 30), 0, "the other track hidden");
    }

    #[test]
    fn an_object_is_a_ring_with_a_faint_inside() {
        let overlay = ResultsOverlay {
            objects: vec![([10, 10, 29, 29], [255, 0, 0])],
            selected: None,
            trails: None,
        };
        let mut buf = buffer(40, 40);
        draw_results_overlay(&mut buf, &overlay, 40, 40, 1.0, 0.0, 0.0);
        let at = |x: usize, y: usize| buf[y * 40 + x];
        assert!(at(10, 20).a > 200, "left edge of the ring");
        assert!(at(20, 20).a < 40 && at(20, 20).a > 0, "the faint inside");
        assert_eq!(at(10, 10).a, 0, "the box's corner is outside the ellipse");
        assert_eq!(at(20, 20).r, 255);
    }

    /// The row runs give the pixels the per-pixel rule gives.
    #[test]
    fn the_fast_fill_matches_the_pixel_by_pixel_ellipse() {
        let overlay = ResultsOverlay {
            objects: vec![
                ([3, 7, 61, 40], [255, 0, 0]),
                ([10, 10, 13, 90], [0, 255, 0]),
            ],
            selected: None,
            trails: None,
        };
        let mut buf = buffer(80, 100);
        draw_results_overlay(&mut buf, &overlay, 80, 100, 1.0, 0.0, 0.0);
        // The reference: each pixel's own distance (as before).
        let mut want = buffer(80, 100);
        for (bbox, rgb) in &overlay.objects {
            let (x0, y0, x1, y1) = (
                bbox[0] as f32,
                bbox[1] as f32,
                (bbox[2] + 1) as f32,
                (bbox[3] + 1) as f32,
            );
            let (cx, cy, rx, ry) = (
                (x0 + x1) / 2.0,
                (y0 + y1) / 2.0,
                (x1 - x0) / 2.0,
                (y1 - y0) / 2.0,
            );
            let r = rx.min(ry);
            for y in y0.floor() as usize..(y1.ceil() as usize).min(100) {
                for x in x0.floor() as usize..(x1.ceil() as usize).min(80) {
                    let (dx, dy) = ((x as f32 + 0.5 - cx) / rx, (y as f32 + 0.5 - cy) / ry);
                    let dist = ((dx * dx + dy * dy).sqrt() - 1.0) * r;
                    let alpha = if dist.abs() < 1.5 {
                        OUTLINE_ALPHA * (1.0 - (dist.abs() - 0.75).max(0.0) / 0.75)
                    } else if dist < 0.0 {
                        FILL_ALPHA
                    } else {
                        continue;
                    };
                    blend(&mut want[y * 80 + x], *rgb, alpha);
                }
            }
        }
        assert!(buf == want, "same pixels");
    }

    #[test]
    fn others_are_dimmed_while_one_is_selected() {
        let overlay = ResultsOverlay {
            objects: vec![([0, 0, 9, 9], [255, 0, 0]), ([20, 0, 29, 9], [0, 255, 0])],
            selected: Some(1),
            trails: None,
        };
        let mut buf = buffer(30, 10);
        draw_results_overlay(&mut buf, &overlay, 30, 10, 1.0, 0.0, 0.0);
        assert!(
            buf[5 * 30].a < buf[5 * 30 + 20].a,
            "the dimmed one is fainter"
        );
        let p = buf[5 * 30 + 20];
        assert_eq!(
            [p.r, p.g, p.b],
            [0, 255, 0],
            "the selected one in its own colour"
        );
    }

    #[test]
    fn a_tiny_or_far_object_is_still_a_dot_and_off_screen_ones_are_skipped() {
        let overlay = ResultsOverlay {
            objects: vec![
                ([100, 100, 101, 101], [0, 0, 255]),
                ([5000, 5000, 5100, 5100], [0, 0, 255]),
            ],
            selected: None,
            trails: None,
        };
        let mut buf = buffer(20, 20);
        draw_results_overlay(&mut buf, &overlay, 20, 20, 0.1, 0.0, 0.0);
        assert!(buf[10 * 20 + 10].a > 0 || buf[9 * 20 + 9].a > 0, "the dot");
        assert_eq!(buf.iter().filter(|p| p.a > 0).count(), 4, "only the dot");
    }

    #[test]
    fn the_focus_centres_the_object_and_keeps_within_fit_and_2_5x() {
        let (zoom, ox, oy) = focus_view(
            800.0,
            600.0,
            10_000.0,
            10_000.0,
            [4990.0, 4990.0, 5009.0, 5009.0],
        );
        assert_eq!(zoom, 2.5, "a small object: at most 2.5×");
        assert!((ox + 5000.0 * zoom - 400.0).abs() < 0.01);
        assert!((oy + 5000.0 * zoom - 300.0).abs() < 0.01);
        let (zoom, ..) = focus_view(600.0, 600.0, 10_000.0, 10_000.0, [0.0, 0.0, 199.0, 99.0]);
        assert_eq!(zoom, 0.5, "a sixth of the view: 600 / (200 × 6)");
        let (zoom, ..) = focus_view(800.0, 600.0, 1000.0, 1000.0, [0.0, 0.0, 999.0, 999.0]);
        assert!(
            (zoom - 0.6 * 0.95).abs() < 1e-5,
            "a huge object: the whole image fitted"
        );
    }

    #[test]
    fn a_click_picks_the_smallest_object_under_it_or_the_nearest() {
        let overlay = ResultsOverlay {
            objects: vec![
                ([0, 0, 99, 99], [0; 3]),
                ([40, 40, 59, 59], [0; 3]),
                ([200, 0, 209, 9], [0; 3]),
            ],
            selected: None,
            trails: None,
        };
        assert_eq!(object_at(&overlay, 50.0, 50.0, 1.0, 0.0, 0.0), Some(1));
        assert_eq!(object_at(&overlay, 10.0, 50.0, 1.0, 0.0, 0.0), Some(0));
        assert_eq!(
            object_at(&overlay, 213.0, 5.0, 1.0, 0.0, 0.0),
            Some(2),
            "4 px beside it"
        );
        assert_eq!(object_at(&overlay, 150.0, 150.0, 1.0, 0.0, 0.0), None);
    }
}
