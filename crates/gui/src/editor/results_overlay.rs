//! # results_overlay
//!
//! The results' split view on the image: the analysis' objects on the
//! shown image, drawn and picked by the editor's code (`object_render`) -
//! in their class's color or the value's heatmap color, from the masks
//! the results keep (switched off: their boxes, filled). With tracks
//! (Results › Tracks) each track's path up to the time point shown goes
//! under them as a trail, fading into the past. Drawn instead of the
//! project's objects (see `ViewportController::sync_objects_to_slint_viewport`).
//! A division or fusion (a track's parents) is a line from the parent's
//! last point before it to the track's first, and a ring where it happened.
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-10-17
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.

use crate::editor::object_render::{
    Canvas, Link, ObjectDrawInstance, draw_objects, mask_holds, pick,
};
use bitvec::prelude::*;
use evanalyzer_app::results::ImageTracks;
use std::sync::Arc;

/// The objects to draw, in image pixels.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ResultsOverlay {
    pub objects: Vec<OverlayObject>,
    /// Each object's index in the caller's list (without the hidden ones)
    /// - empty: the same.
    pub sources: Vec<usize>,
    /// Lines from the selected object to its relatives.
    pub links: Vec<Link>,
    /// The tracks' trails, under the objects.
    pub trails: Option<Arc<TrailsOverlay>>,
}

impl ResultsOverlay {
    /// The caller's index of the object at `index` here.
    pub fn source(&self, index: usize) -> usize {
        self.sources.get(index).copied().unwrap_or(index)
    }
}

/// An object of the overlay.
#[derive(Debug, Clone, PartialEq)]
pub struct OverlayObject {
    /// `[xmin, ymin, xmax, ymax]`, inclusive.
    pub bbox: [u32; 4],
    pub rgb: [u8; 3],
    /// Outside the selected object's family: a third of the opacity.
    pub dimmed: bool,
    /// A bit per pixel of the box, row by row.
    pub mask: Arc<BitVec<u64, Lsb0>>,
    /// Its layer (`ZOrder::key`), and the same without the selection's
    /// boost for a click.
    pub layer: u32,
    pub pick: u32,
}

impl OverlayObject {
    /// In `rgb` on the lowest layer; without a mask its box, filled.
    pub fn new(bbox: [u32; 4], rgb: [u8; 3], mask: Option<Arc<BitVec<u64, Lsb0>>>) -> Self {
        let mask = mask.unwrap_or_else(|| {
            let [x0, y0, x1, y1] = bbox;
            let len = (x1 - x0 + 1) as usize * (y1 - y0 + 1) as usize;
            Arc::new(BitVec::repeat(true, len))
        });
        Self {
            bbox,
            rgb,
            dimmed: false,
            mask,
            layer: 0,
            pick: 0,
        }
    }
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
    /// The selected track's parents and children (indices into
    /// `tracks.tracks`): their whole paths faint too, shown with it.
    pub family: Vec<usize>,
    /// Only the selected track's.
    pub only_selected: bool,
}

/// A trail's opacity at its newest end; its oldest end a quarter of it.
const TRAIL_ALPHA: f32 = 0.9;
/// The selected track's whole path, behind its trail.
const GHOST_ALPHA: f32 = 0.22;
/// The ring where a division or fusion happened, in screen pixels.
const EVENT_RING_RADIUS: f32 = 4.0;
const SELECTED_TRAIL_RGB: [u8; 3] = [255, 255, 255];

/// Draws `overlay` into `buffer` (`buf_w` × `buf_h`, transparent before),
/// image pixel `p` at `p * zoom + off` on it: the trails, then the objects
/// as the editor draws its own at opacity `fill` (0..1, the viewer's slider).
#[allow(clippy::too_many_arguments)]
pub fn draw_results_overlay(
    buffer: &mut [slint::Rgba8Pixel],
    overlay: &ResultsOverlay,
    buf_w: u32,
    buf_h: u32,
    zoom: f32,
    off_x: f32,
    off_y: f32,
    fill: f32,
) {
    if let Some(trails) = &overlay.trails {
        draw_trails(buffer, trails, buf_w, buf_h, zoom, off_x, off_y);
    }
    let opacity = (fill.clamp(0.0, 1.0) * 255.0) as u8;
    let objects = overlay
        .objects
        .iter()
        .map(|object| {
            let [r, g, b] = object.rgb;
            let a = if object.dimmed { opacity / 3 } else { opacity };
            let color = slint::Rgba8Pixel { r, g, b, a };
            let instance = ObjectDrawInstance {
                bbox: object.bbox,
                mask: &object.mask,
                color,
            };
            (object.layer, instance)
        })
        .collect();
    draw_objects(
        buffer,
        objects,
        &overlay.links,
        buf_w,
        buf_h,
        zoom,
        off_x,
        off_y,
    );
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
    let tracks = &trails.tracks.tracks;
    let in_family = |i: usize| trails.family.contains(&i);
    // The family, then the selected one last: on top. Only the selected:
    // with its family.
    let order = (0..tracks.len())
        .filter(|i| Some(*i) != trails.selected && !in_family(*i) && !trails.only_selected)
        .chain(
            trails
                .family
                .iter()
                .copied()
                .filter(|i| Some(*i) != trails.selected),
        )
        .chain(trails.selected)
        .filter(|i| *i < tracks.len());
    for i in order {
        let track = &tracks[i];
        let selected = trails.selected == Some(i);
        let rgb = match selected {
            true => SELECTED_TRAIL_RGB,
            false => trails.colors.get(i).copied().unwrap_or(SELECTED_TRAIL_RGB),
        };
        // Where it came from: a line from each parent's last point before
        // it, a ring there - once its first point is in the trail.
        if let Some(first) = track.points.first()
            && first.t <= trails.t
            && from.is_none_or(|from| first.t >= from)
        {
            for parent in &track.parent_track_ids {
                let Ok(p) = tracks.binary_search_by_key(parent, |t| t.track_id) else {
                    continue; // Not read (a sample).
                };
                let before = tracks[p].points.partition_point(|q| q.t < first.t);
                let Some(at) = before.checked_sub(1).map(|k| &tracks[p].points[k]) else {
                    continue;
                };
                canvas.line(screen(at), screen(first), rgb, TRAIL_ALPHA, selected);
                ring(&mut canvas, screen(at), rgb, TRAIL_ALPHA);
            }
        }
        // Its whole path, faint: where it comes from and goes.
        if selected || in_family(i) {
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

/// A small ring around `centre` (screen pixels): where a division or
/// fusion happened.
fn ring(canvas: &mut Canvas, centre: (f32, f32), rgb: [u8; 3], alpha: f32) {
    const SIDES: usize = 12;
    let at = |k: usize| {
        let angle = k as f32 / SIDES as f32 * std::f32::consts::TAU;
        (
            centre.0 + EVENT_RING_RADIUS * angle.cos(),
            centre.1 + EVENT_RING_RADIUS * angle.sin(),
        )
    };
    for k in 0..SIDES {
        canvas.line(at(k), at(k + 1), rgb, alpha, false);
    }
}

/// The view that shows `bbox` (image pixels, inclusive) in a `v_w` × `v_h`
/// viewport: centred, the object about a 25th of the view - its
/// neighbourhood in view around it - never smaller than the whole image
/// fitted, never beyond 2.5×. As `(zoom, offset_x, offset_y)`.
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
    let zoom = (v_w / (bw * 25.0)).min(v_h / (bh * 25.0)).min(2.5).max(fit);
    let (cx, cy) = (
        (bbox[0] + bbox[2] + 1.0) / 2.0,
        (bbox[1] + bbox[3] + 1.0) / 2.0,
    );
    (zoom, v_w / 2.0 - cx * zoom, v_h / 2.0 - cy * zoom)
}

/// The object under the screen point `(x, y)`, picked as in the editor.
pub fn object_at(
    overlay: &ResultsOverlay,
    x: f32,
    y: f32,
    zoom: f32,
    off_x: f32,
    off_y: f32,
) -> Option<usize> {
    let (ix, iy) = ((x - off_x) / zoom, (y - off_y) / zoom);
    if ix < 0.0 || iy < 0.0 {
        return None;
    }
    let (ix, iy) = (ix as u32, iy as u32);
    pick(
        overlay
            .objects
            .iter()
            .map(|o| (o.pick, mask_holds(o.bbox, &o.mask, ix, iy))),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A faint inside, to tell it from the outline.
    const FILL: f32 = 0.12;

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
            parent_track_ids: Vec::new(),
        };
        ResultsOverlay {
            sources: Vec::new(),
            links: Vec::new(),
            objects: Vec::new(),
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
                family: Vec::new(),
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
                    parent_track_ids: Vec::new(),
                }
            })
            .collect();
        // The same with divisions: every 10th track a daughter of the one
        // before, from its 24th time point, starting where its mother is
        // (a short line and a ring each).
        let mut divided = tracks.clone();
        for i in (10..divided.len()).step_by(10) {
            let (mother, mother_id) = (divided[i - 1].points[23], divided[i - 1].track_id);
            let daughter = &mut divided[i];
            daughter.parent_track_ids = vec![mother_id];
            daughter.points.drain(..24);
            let (dx, dy) = (
                mother.x - daughter.points[0].x,
                mother.y - daughter.points[0].y,
            );
            for p in &mut daughter.points {
                p.x += dx + 3.0;
                p.y += dy + 3.0;
            }
        }
        let divided = Arc::new(ImageTracks {
            total_tracks: divided.len(),
            tracks: divided,
            ..ImageTracks::default()
        });
        let tracks = Arc::new(ImageTracks {
            total_tracks: tracks.len(),
            tracks,
            ..ImageTracks::default()
        });
        let with = |tracks: &Arc<ImageTracks>, frames, selected, only_selected| ResultsOverlay {
            sources: Vec::new(),
            links: Vec::new(),
            objects: Vec::new(),
            trails: Some(Arc::new(TrailsOverlay {
                tracks: tracks.clone(),
                colors: vec![[90, 160, 255]; 1500],
                t: 30,
                frames,
                selected,
                family: Vec::new(),
                only_selected,
            })),
        };
        let overlay =
            |frames, selected, only_selected| with(&tracks, frames, selected, only_selected);
        for (label, overlay) in [
            (
                "trails of 10 time points",
                overlay(Some(10), Some(5), false),
            ),
            ("whole tracks up to t", overlay(None, Some(5), false)),
            ("the selected track only", overlay(Some(10), Some(5), true)),
            ("whole, 149 divisions", with(&divided, None, Some(5), false)),
        ] {
            let mut times = Vec::new();
            for _ in 0..15 {
                let mut buf = buffer(1920, 1080);
                let start = std::time::Instant::now();
                draw_results_overlay(&mut buf, &overlay, 1920, 1080, 0.9, 0.0, 0.0, FILL);
                times.push(start.elapsed().as_secs_f64() * 1000.0);
            }
            times.sort_by(f64::total_cmp);
            println!("{label:<28} {:>7.2} ms", times[times.len() / 2]);
        }
    }

    /// Track 1 along y = 10 (t 0-9); track 2 along y = 30 from t 5 - with
    /// `parents`. Its whole trail up to t 9, `selected`/`family` as given.
    fn daughter(
        parents: Vec<u64>,
        selected: Option<usize>,
        family: Vec<usize>,
        only: bool,
    ) -> Vec<slint::Rgba8Pixel> {
        let ResultsOverlay { trails, .. } = trails(9, None, selected, only);
        let mut trails = (*trails.unwrap()).clone();
        let mut tracks = (*trails.tracks).clone();
        tracks.tracks[1].points.drain(..5);
        tracks.tracks[1].parent_track_ids = parents;
        trails.tracks = Arc::new(tracks);
        trails.family = family;
        let overlay = ResultsOverlay {
            trails: Some(Arc::new(trails)),
            ..ResultsOverlay::default()
        };
        let mut buf = buffer(100, 40);
        draw_results_overlay(&mut buf, &overlay, 100, 40, 1.0, 0.0, 0.0, FILL);
        buf
    }

    /// A daughter: a line from its mother's point before it (t 4 at
    /// x 40) to its first (t 5 at x 50), and a ring at the mother's.
    #[test]
    fn a_division_is_a_line_from_the_mother_and_a_ring() {
        let with = daughter(vec![1], None, vec![], false);
        let without = daughter(vec![], None, vec![], false);
        assert!(alpha_at(&with, 45, 20) > 0, "the line half way");
        assert_eq!(alpha_at(&without, 45, 20), 0);
        assert!(alpha_at(&with, 40, 14) > 0, "the ring's bottom");
        assert_eq!(alpha_at(&without, 40, 14), 0);
        assert_eq!(
            with[(20 * 100 + 45) as usize].b,
            255,
            "in the daughter's colour"
        );
    }

    /// Only the selected track: its family with it, the rest not.
    #[test]
    fn only_the_selected_track_shows_its_family_too() {
        let alone = daughter(vec![1], Some(1), vec![], true);
        assert_eq!(alpha_at(&alone, 20, 10), 0, "the mother isn't shown");
        let family = daughter(vec![1], Some(1), vec![0], true);
        assert!(
            alpha_at(&family, 20, 10) > 0,
            "with its family: the mother is"
        );
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
            FILL,
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
            FILL,
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
            FILL,
        );
        let pixel = buf[(10 * 100 + 45) as usize];
        assert_eq!((pixel.r, pixel.g, pixel.b), (255, 255, 255));
        assert!(alpha_at(&buf, 75, 10) > 0, "its path ahead, faint");
        assert!(alpha_at(&buf, 75, 10) < alpha_at(&buf, 45, 10));
        assert_eq!(alpha_at(&buf, 45, 30), 0, "the other track hidden");
    }

    /// A 20 x 20 ring, 5 px thick, in the box `[10, 10, 29, 29]`.
    fn ring_overlay() -> ResultsOverlay {
        let mut mask = BitVec::<u64, Lsb0>::repeat(false, 400);
        for y in 0..20 {
            for x in 0..20 {
                let inner = (5..15).contains(&x) && (5..15).contains(&y);
                mask.set(y * 20 + x, !inner);
            }
        }
        ResultsOverlay {
            objects: vec![OverlayObject::new(
                [10, 10, 29, 29],
                [255, 0, 0],
                Some(Arc::new(mask)),
            )],
            sources: Vec::new(),
            links: Vec::new(),
            trails: None,
        }
    }

    #[test]
    fn an_object_with_a_mask_is_drawn_as_its_mask() {
        let overlay = ring_overlay();
        for zoom in [1.0, 3.0] {
            let n = (40.0 * zoom) as usize;
            let mut buf = buffer(n as u32, n as u32);
            draw_results_overlay(&mut buf, &overlay, n as u32, n as u32, zoom, 0.0, 0.0, FILL);
            let at = |x: f32, y: f32| buf[(y * zoom) as usize * n + (x * zoom) as usize];
            assert!(
                at(10.0, 10.0).a > 200,
                "the box's corner: the mask's, outlined"
            );
            assert!(at(12.5, 20.0).a > 0, "inside the ring: tinted");
            assert!(at(12.5, 20.0).a < 40, "... faintly");
            // The editor's outline: the mask's last screen pixel there.
            let edge = buf[(20.0 * zoom) as usize * n + (15.0 * zoom) as usize - 1];
            assert!(edge.a > 200, "its inner edge outlined");
            assert_eq!(at(20.0, 20.0).a, 0, "the hole");
        }
    }

    /// The objects' opacity (the viewer's slider) is their insides'; the
    /// outline stays.
    #[test]
    fn the_inside_follows_the_objects_opacity() {
        let overlay = ring_overlay();
        for fill in [0.0, 0.5, 0.8] {
            // 100 wide, as `alpha_at` reads.
            let mut buf = buffer(100, 40);
            draw_results_overlay(&mut buf, &overlay, 100, 40, 1.0, 0.0, 0.0, fill);
            assert_eq!(
                alpha_at(&buf, 12, 20),
                (fill * 255.0) as u8,
                "as the editor"
            );
            assert!(alpha_at(&buf, 10, 20) > 200, "the outline");
        }
    }

    /// A click picks as in the editor: the highest layer holding the
    /// point - in a mask's hole the object below, the later one on a tie.
    #[test]
    fn a_click_picks_the_top_layer_and_not_a_masks_hole() {
        let mut overlay = ring_overlay();
        overlay.objects[0].pick = 1;
        // Under the ring a larger object without a mask (its box).
        overlay
            .objects
            .insert(0, OverlayObject::new([0, 0, 39, 39], [0, 255, 0], None));
        assert_eq!(
            object_at(&overlay, 12.0, 20.0, 1.0, 0.0, 0.0),
            Some(1),
            "on the ring"
        );
        assert_eq!(
            object_at(&overlay, 20.0, 20.0, 1.0, 0.0, 0.0),
            Some(0),
            "in its hole: the one below"
        );
        overlay.objects[1].pick = 0;
        assert_eq!(
            object_at(&overlay, 12.0, 20.0, 1.0, 0.0, 0.0),
            Some(1),
            "a tie: the later"
        );
        assert_eq!(
            object_at(&overlay, 45.0, 20.0, 1.0, 0.0, 0.0),
            None,
            "beside both"
        );
    }

    #[test]
    fn the_focus_centres_the_object_and_keeps_within_fit_and_2_5x() {
        let (zoom, ox, oy) = focus_view(
            800.0,
            600.0,
            10_000.0,
            10_000.0,
            [4999.0, 4999.0, 5000.0, 5000.0],
        );
        assert_eq!(zoom, 2.5, "a tiny object: at most 2.5×");
        assert!((ox + 5000.0 * zoom - 400.0).abs() < 0.01);
        assert!((oy + 5000.0 * zoom - 300.0).abs() < 0.01);
        let (zoom, ..) = focus_view(600.0, 600.0, 10_000.0, 10_000.0, [0.0, 0.0, 199.0, 99.0]);
        assert_eq!(zoom, 0.12, "a 25th of the view: 600 / (200 × 25)");
        let (zoom, ..) = focus_view(800.0, 600.0, 1000.0, 1000.0, [0.0, 0.0, 999.0, 999.0]);
        assert!(
            (zoom - 0.6 * 0.95).abs() < 1e-5,
            "a huge object: the whole image fitted"
        );
    }
}
