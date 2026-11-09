//! # track_motion
//!
//! How a track moved: its speeds and ratios as computer-assisted sperm
//! analysis (CASA) defines them (WHO laboratory manual, 2021),
//! its turning, how far it got from its start and its mean squared
//! displacement (MSD). Pure functions on a track's positions - Measure
//! Track Motion stores what they return.
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-11-09
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.
//!
//! ## Overview
//! All in pixels and time points (frames); a value that can't be had (a
//! track with one time point, too few steps) is NaN.
//!
//! - Path length L: the sum of the steps between consecutive points (a
//!   gap is one straight step). Displacement d: first to last point.
//! - VCL = L / T, VSL = d / T (T: last minus first time point).
//! - Average path: each point replaced by the mean of the points up to
//!   `average_half_width` before and after it - as many on both sides, so
//!   fewer near the ends and the first and last point stay (a straight
//!   path stays as long). VAP: its length / T.
//! - LIN = VSL / VCL, STR = VSL / VAP, WOB = VAP / VCL.
//! - ALH: twice the mean distance of a point from its average path point
//!   (an amplitude: from one side to the other).
//! - BCF: how often the path crosses the average path, per time point.
//! - Turning: the angle from one step to the next, + counter-clockwise as
//!   seen on the image (y down). Steps shorter than `min_step_px` are
//!   joined with the next ones first - jitter of the centroid isn't
//!   turning. Mean, mean absolute, and the sum over T (angular speed).
//! - MSD(n): the mean squared distance between the points n time points
//!   apart, for n up to T / 4 spaced evenly on the log scale (1, 2, 3, 4,
//!   5, 6, 7, 9, 12, ... - each step ×1.25: the noisy long lags of a long
//!   track would else outnumber the short ones); α and D from the
//!   straight line through log MSD over log n: MSD = 4 D n^α. Needs 10
//!   points and 2 lags.

/// A track's position at a time point, in pixels.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TrackPoint {
    pub t: i32,
    pub x: f64,
    pub y: f64,
}

/// See the module.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct MotionOptions {
    pub min_step_px: f64,
    pub average_half_width: usize,
}

/// See the module - NaN where it can't be had.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(crate) struct TrackMotion {
    pub path_length: f64,
    pub displacement: f64,
    pub max_distance: f64,
    pub vcl: f64,
    pub vsl: f64,
    pub vap: f64,
    pub lin: f64,
    pub str_: f64,
    pub wob: f64,
    pub alh: f64,
    pub bcf: f64,
    pub turning_mean: f64,
    pub turning_abs_mean: f64,
    pub angular_speed: f64,
    pub msd_alpha: f64,
    pub msd_d: f64,
}

/// The fewest points and lags an MSD fit takes.
const MSD_MIN_POINTS: usize = 10;
const MSD_MIN_LAGS: usize = 2;

fn dist(a: (f64, f64), b: (f64, f64)) -> f64 {
    (b.0 - a.0).hypot(b.1 - a.1)
}

fn ratio(a: f64, b: f64) -> f64 {
    if b > 0.0 { a / b } else { f64::NAN }
}

/// How `points` (by time, one per time point) moved - see the module.
pub(crate) fn measure(points: &[TrackPoint], options: &MotionOptions) -> TrackMotion {
    let p: Vec<(f64, f64)> = points.iter().map(|p| (p.x, p.y)).collect();
    let (Some(first), Some(last)) = (points.first(), points.last()) else {
        return TrackMotion {
            path_length: f64::NAN,
            displacement: f64::NAN,
            max_distance: f64::NAN,
            ..nan()
        };
    };
    let path_length: f64 = p.windows(2).map(|w| dist(w[0], w[1])).sum();
    let displacement = dist(p[0], p[p.len() - 1]);
    let max_distance = p.iter().map(|q| dist(p[0], *q)).fold(0.0, f64::max);
    let frames = f64::from(last.t - first.t);
    if p.len() < 2 || frames <= 0.0 {
        return TrackMotion {
            path_length,
            displacement,
            max_distance,
            ..nan()
        };
    }

    let average = average_path(&p, options.average_half_width);
    let average_length: f64 = average.windows(2).map(|w| dist(w[0], w[1])).sum();
    let vcl = path_length / frames;
    let vsl = displacement / frames;
    let vap = average_length / frames;
    let alh = 2.0
        * p.iter()
            .zip(&average)
            .map(|(a, b)| dist(*a, *b))
            .sum::<f64>()
        / p.len() as f64;
    let bcf = crossings(&p, &average) as f64 / frames;

    let angles = turning_angles(&p, options.min_step_px);
    let (turning_mean, turning_abs_mean, angular_speed) = if angles.is_empty() {
        (f64::NAN, f64::NAN, f64::NAN)
    } else {
        let n = angles.len() as f64;
        let sum: f64 = angles.iter().sum();
        (
            sum / n,
            angles.iter().map(|a| a.abs()).sum::<f64>() / n,
            sum / frames,
        )
    };
    let (msd_alpha, msd_d) = msd_fit(points);

    TrackMotion {
        path_length,
        displacement,
        max_distance,
        vcl,
        vsl,
        vap,
        lin: ratio(vsl, vcl),
        str_: ratio(vsl, vap),
        wob: ratio(vap, vcl),
        alh,
        bcf,
        turning_mean,
        turning_abs_mean,
        angular_speed,
        msd_alpha,
        msd_d,
    }
}

fn nan() -> TrackMotion {
    TrackMotion {
        path_length: f64::NAN,
        displacement: f64::NAN,
        max_distance: f64::NAN,
        vcl: f64::NAN,
        vsl: f64::NAN,
        vap: f64::NAN,
        lin: f64::NAN,
        str_: f64::NAN,
        wob: f64::NAN,
        alh: f64::NAN,
        bcf: f64::NAN,
        turning_mean: f64::NAN,
        turning_abs_mean: f64::NAN,
        angular_speed: f64::NAN,
        msd_alpha: f64::NAN,
        msd_d: f64::NAN,
    }
}

/// Each point as the mean of the points up to `half_width` before and
/// after it, as many on both sides (fewer at the ends) - a running sum, O(n).
fn average_path(p: &[(f64, f64)], half_width: usize) -> Vec<(f64, f64)> {
    let n = p.len();
    let mut prefix = Vec::with_capacity(n + 1);
    prefix.push((0.0, 0.0));
    for q in p {
        let (sx, sy) = prefix[prefix.len() - 1];
        prefix.push((sx + q.0, sy + q.1));
    }
    (0..n)
        .map(|i| {
            let half = half_width.min(i).min(n - 1 - i);
            let (lo, hi) = (i - half, i + half + 1);
            let count = (hi - lo) as f64;
            (
                (prefix[hi].0 - prefix[lo].0) / count,
                (prefix[hi].1 - prefix[lo].1) / count,
            )
        })
        .collect()
}

/// How often `p` changes the side of `average` it is on: the side of
/// each point to the average path's direction there.
fn crossings(p: &[(f64, f64)], average: &[(f64, f64)]) -> usize {
    let n = p.len();
    let mut last_side = 0.0;
    let mut count = 0;
    for i in 0..n {
        let (a, b) = if i + 1 < n {
            (average[i], average[i + 1])
        } else {
            (average[i - 1], average[i])
        };
        let direction = (b.0 - a.0, b.1 - a.1);
        let offset = (p[i].0 - average[i].0, p[i].1 - average[i].1);
        let side = (direction.0 * offset.1 - direction.1 * offset.0).signum();
        if direction == (0.0, 0.0) || offset == (0.0, 0.0) {
            continue;
        }
        if last_side != 0.0 && side != last_side {
            count += 1;
        }
        last_side = side;
    }
    count
}

/// The angles (degrees, + counter-clockwise on the image, y down) from
/// each step to the next, steps shorter than `min_step` joined first.
fn turning_angles(p: &[(f64, f64)], min_step: f64) -> Vec<f64> {
    let mut steps: Vec<(f64, f64)> = Vec::new();
    let mut anchor = p[0];
    for q in &p[1..] {
        let step = (q.0 - anchor.0, q.1 - anchor.1);
        let length = step.0.hypot(step.1);
        if length > 0.0 && length >= min_step {
            steps.push(step);
            anchor = *q;
        }
    }
    steps
        .windows(2)
        .map(|w| {
            let (a, b) = (w[0], w[1]);
            let cross = a.0 * b.1 - a.1 * b.0;
            let dot = a.0 * b.0 + a.1 * b.1;
            // y points down: a turn that is clockwise in (x, y) is
            // counter-clockwise on the screen.
            -cross.atan2(dot).to_degrees()
        })
        .collect()
}

/// α and D of MSD = 4 D n^α, fitted over n = 1 .. T / 4 - NaN with fewer
/// than `MSD_MIN_POINTS` points or `MSD_MIN_LAGS` lags.
fn msd_fit(points: &[TrackPoint]) -> (f64, f64) {
    let curve = msd_curve(points);
    if curve.len() < MSD_MIN_LAGS {
        return (f64::NAN, f64::NAN);
    }
    let xs: Vec<f64> = curve.iter().map(|(lag, _)| (*lag as f64).ln()).collect();
    let ys: Vec<f64> = curve.iter().map(|(_, msd)| msd.ln()).collect();
    let n = xs.len() as f64;
    let mean_x = xs.iter().sum::<f64>() / n;
    let mean_y = ys.iter().sum::<f64>() / n;
    let sxx: f64 = xs.iter().map(|x| (x - mean_x).powi(2)).sum();
    let sxy: f64 = xs
        .iter()
        .zip(&ys)
        .map(|(x, y)| (x - mean_x) * (y - mean_y))
        .sum();
    let alpha = sxy / sxx;
    let intercept = mean_y - alpha * mean_x;
    (alpha, intercept.exp() / 4.0)
}

/// The points α and D are fitted through: (lag, MSD in px²) for the lags
/// up to T / 4, log-spaced, that have pairs and an MSD above 0 - empty for
/// a track too short for the fit (fewer than `MSD_MIN_POINTS` points or
/// `MSD_MIN_LAGS` lags). Public for the results' MSD plot: the same values
/// the step fitted.
pub fn msd_curve(points: &[TrackPoint]) -> Vec<(usize, f64)> {
    let (Some(first), Some(last)) = (points.first(), points.last()) else {
        return Vec::new();
    };
    let span = (last.t - first.t) as usize;
    let max_lag = span / 4;
    if points.len() < MSD_MIN_POINTS || max_lag < MSD_MIN_LAGS {
        return Vec::new();
    }
    // By time point (a gap: none there).
    let mut at: Vec<Option<(f64, f64)>> = vec![None; span + 1];
    for p in points {
        at[(p.t - first.t) as usize] = Some((p.x, p.y));
    }
    let mut curve = Vec::new();
    for lag in log_spaced_lags(max_lag) {
        let (mut sum, mut pairs) = (0.0, 0usize);
        for i in 0..=span - lag {
            if let (Some(a), Some(b)) = (at[i], at[i + lag]) {
                sum += (b.0 - a.0).powi(2) + (b.1 - a.1).powi(2);
                pairs += 1;
            }
        }
        if pairs > 0 && sum > 0.0 {
            curve.push((lag, sum / pairs as f64));
        }
    }
    curve
}

/// 1 .. `max`, each ×1.25 the one before (rounded, each once).
fn log_spaced_lags(max: usize) -> Vec<usize> {
    let mut lags = Vec::new();
    let mut x = 1.0f64;
    while x.round() as usize <= max {
        let lag = x.round() as usize;
        if lags.last() != Some(&lag) {
            lags.push(lag);
        }
        x *= 1.25;
    }
    lags
}

/// The drift of an image plane: per time point, how far everything moved
/// since the first one - the median step of the `reference` tracks from
/// one time point to the next, summed up. A time point without a
/// reference step adds none. By time point, from the references' first to
/// their last.
pub(crate) fn drift(reference: &[Vec<TrackPoint>]) -> Vec<TrackPoint> {
    let (Some(t0), Some(t1)) = (
        reference
            .iter()
            .filter_map(|t| t.first())
            .map(|p| p.t)
            .min(),
        reference.iter().filter_map(|t| t.last()).map(|p| p.t).max(),
    ) else {
        return Vec::new();
    };
    let mut steps: Vec<(Vec<f64>, Vec<f64>)> =
        vec![(Vec::new(), Vec::new()); (t1 - t0) as usize + 1];
    for track in reference {
        for w in track.windows(2) {
            // Only steps of one time point: a gap's step is no frame's.
            if w[1].t - w[0].t == 1 {
                let slot = &mut steps[(w[1].t - t0) as usize];
                slot.0.push(w[1].x - w[0].x);
                slot.1.push(w[1].y - w[0].y);
            }
        }
    }
    let (mut x, mut y) = (0.0, 0.0);
    steps
        .into_iter()
        .enumerate()
        .map(|(i, (mut dx, mut dy))| {
            if !dx.is_empty() {
                x += median(&mut dx);
                y += median(&mut dy);
            }
            TrackPoint {
                t: t0 + i as i32,
                x,
                y,
            }
        })
        .collect()
}

fn median(values: &mut [f64]) -> f64 {
    values.sort_by(f64::total_cmp);
    let n = values.len();
    if n % 2 == 1 {
        values[n / 2]
    } else {
        (values[n / 2 - 1] + values[n / 2]) / 2.0
    }
}

/// `drift` at time point `t`: none before its first, its last after.
pub(crate) fn drift_at(drift: &[TrackPoint], t: i32) -> (f64, f64) {
    match (drift.first(), drift.last()) {
        (Some(first), Some(last)) => {
            if t <= first.t {
                (first.x, first.y)
            } else if t >= last.t {
                (last.x, last.y)
            } else {
                let p = drift[(t - first.t) as usize];
                (p.x, p.y)
            }
        }
        _ => (0.0, 0.0),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const OPTIONS: MotionOptions = MotionOptions {
        min_step_px: 0.0,
        average_half_width: 2,
    };

    fn track(points: impl IntoIterator<Item = (i32, f64, f64)>) -> Vec<TrackPoint> {
        points
            .into_iter()
            .map(|(t, x, y)| TrackPoint { t, x, y })
            .collect()
    }

    fn close(a: f64, b: f64, tolerance: f64) -> bool {
        (a - b).abs() <= tolerance
    }

    #[test]
    fn a_straight_line_is_linear_turns_never_and_is_directed() {
        // 3 px per time point to the right, 40 time points.
        let line = track((0..40).map(|t| (t, 3.0 * f64::from(t), 5.0)));
        let m = measure(&line, &OPTIONS);
        assert!(close(m.path_length, 117.0, 1e-9));
        assert!(close(m.displacement, 117.0, 1e-9));
        assert!(close(m.max_distance, 117.0, 1e-9));
        assert!(close(m.vcl, 3.0, 1e-9) && close(m.vsl, 3.0, 1e-9));
        assert!(close(m.lin, 1.0, 1e-9));
        assert!(close(m.vap, 3.0, 1e-9) && close(m.str_, 1.0, 1e-9) && close(m.wob, 1.0, 1e-9));
        assert!(close(m.alh, 0.0, 1e-9));
        assert_eq!(m.bcf, 0.0);
        assert!(close(m.turning_mean, 0.0, 1e-9) && close(m.turning_abs_mean, 0.0, 1e-9));
        assert!(close(m.angular_speed, 0.0, 1e-9));
        assert!(close(m.msd_alpha, 2.0, 1e-9), "{}", m.msd_alpha);
        // MSD(n) = 9 n² = 4 D n²: D = 2.25.
        assert!(close(m.msd_d, 2.25, 1e-9), "{}", m.msd_d);
    }

    #[test]
    fn a_circle_turns_one_way_and_stays_near_its_start() {
        // Counter-clockwise on the screen (y down): 10° per time point,
        // radius 20 px, four turns.
        let circle = track((0..=144).map(|t| {
            let a = f64::from(t * 10).to_radians();
            (t, 20.0 * a.cos(), -20.0 * a.sin())
        }));
        let m = measure(&circle, &OPTIONS);
        assert!(close(m.turning_mean, 10.0, 1e-9), "{}", m.turning_mean);
        assert!(close(m.turning_abs_mean, 10.0, 1e-9));
        assert!(close(m.angular_speed, 10.0 * 143.0 / 144.0, 1e-9));
        assert!(m.lin < 0.01, "back at its start: {}", m.lin);
        assert!(
            close(m.max_distance, 40.0, 1e-6),
            "the diameter: {}",
            m.max_distance
        );
        assert!(m.msd_alpha < 1.0, "bounded: {}", m.msd_alpha);
        // Clockwise: the sign flips.
        let mirrored: Vec<_> = circle
            .iter()
            .map(|p| TrackPoint { y: -p.y, ..*p })
            .collect();
        let m = measure(&mirrored, &OPTIONS);
        assert!(close(m.turning_mean, -10.0, 1e-9), "{}", m.turning_mean);
    }

    #[test]
    fn a_random_walk_turns_both_ways_and_diffuses() {
        // A fixed pseudo-random walk (no crate: a linear congruential
        // generator), steps of 1 px in a random direction.
        let mut seed: u64 = 12345;
        let mut next = || {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (seed >> 11) as f64 / (1u64 << 53) as f64
        };
        let (mut x, mut y) = (0.0, 0.0);
        let mut walk = Vec::new();
        for t in 0..4000 {
            walk.push(TrackPoint { t, x, y });
            let a = next() * std::f64::consts::TAU;
            x += a.cos();
            y += a.sin();
        }
        let m = measure(&walk, &OPTIONS);
        assert!(close(m.msd_alpha, 1.0, 0.25), "{}", m.msd_alpha);
        assert!(close(m.turning_mean, 0.0, 10.0), "{}", m.turning_mean);
        assert!(
            close(m.turning_abs_mean, 90.0, 10.0),
            "{}",
            m.turning_abs_mean
        );
        assert!(m.lin < 0.1, "{}", m.lin);
    }

    #[test]
    fn a_zigzag_crosses_its_average_path_at_every_step() {
        // Forward 2 px, 3 px to one side and the other in turn.
        let zigzag =
            track((0..21).map(|t| (t, 2.0 * f64::from(t), if t % 2 == 0 { 3.0 } else { -3.0 })));
        let m = measure(
            &zigzag,
            &MotionOptions {
                min_step_px: 0.0,
                average_half_width: 1,
            },
        );
        // Every point crosses but the first and last (they are their own
        // average): 18 crossings in 20 time points.
        assert!(close(m.bcf, 0.9, 1e-9), "{}", m.bcf);
        assert!(m.alh > 3.0 && m.alh < 12.0, "{}", m.alh);
        assert!(m.wob < 1.0 && m.vap < m.vcl);
    }

    #[test]
    fn short_steps_are_joined_before_the_turning_is_taken() {
        // Jitter of 0.1 px around a straight path of 2 px steps: without a
        // minimum step it turns a lot, with 1 px it doesn't.
        let jitter = track((0..30).flat_map(|i| {
            let x = 2.0 * f64::from(i);
            [
                (2 * i, x, 0.0),
                (2 * i + 1, x + 0.05, if i % 2 == 0 { 0.1 } else { -0.1 }),
            ]
        }));
        let raw = measure(&jitter, &OPTIONS);
        let joined = measure(
            &jitter,
            &MotionOptions {
                min_step_px: 1.0,
                ..OPTIONS
            },
        );
        assert!(raw.turning_abs_mean > 30.0, "{}", raw.turning_abs_mean);
        assert!(
            joined.turning_abs_mean < 10.0,
            "{}",
            joined.turning_abs_mean
        );
        // The path is the same: only the turning joins steps.
        assert_eq!(raw.path_length, joined.path_length);
    }

    #[test]
    fn one_time_point_has_no_speed() {
        let m = measure(&track([(4, 1.0, 1.0)]), &OPTIONS);
        assert_eq!(
            (m.path_length, m.displacement, m.max_distance),
            (0.0, 0.0, 0.0)
        );
        assert!(
            m.vcl.is_nan() && m.lin.is_nan() && m.turning_mean.is_nan() && m.msd_alpha.is_nan()
        );
        assert!(measure(&[], &OPTIONS).path_length.is_nan());
    }

    #[test]
    fn a_gap_is_one_step_and_counts_its_time() {
        // T 0 and T 4, 8 px apart: 2 px per time point.
        let m = measure(&track([(0, 0.0, 0.0), (4, 8.0, 0.0)]), &OPTIONS);
        assert_eq!((m.vcl, m.vsl), (2.0, 2.0));
    }

    #[test]
    fn the_lags_are_log_spaced() {
        assert_eq!(log_spaced_lags(12), [1, 2, 3, 4, 5, 6, 7, 9, 12]);
        assert_eq!(log_spaced_lags(1), [1]);
    }

    #[test]
    fn too_short_a_track_has_no_msd() {
        let line = track((0..9).map(|t| (t, f64::from(t), 0.0)));
        assert!(measure(&line, &OPTIONS).msd_alpha.is_nan());
    }

    #[test]
    fn the_drift_is_the_references_median_step_summed_up() {
        // Three references move +2 px in x per time point, one stray -10.
        let mut reference: Vec<Vec<TrackPoint>> = (0..3)
            .map(|k| track((0..5).map(|t| (t, f64::from(k * 50 + 2 * t), 7.0))))
            .collect();
        reference.push(track((0..5).map(|t| (t, f64::from(-10 * t), 0.0))));
        let drift = drift(&reference);
        assert_eq!(drift.len(), 5);
        assert_eq!(drift_at(&drift, 0), (0.0, 0.0));
        assert_eq!(drift_at(&drift, 3), (6.0, 0.0));
        // Outside the references' time points: their ends.
        assert_eq!(drift_at(&drift, -2), (0.0, 0.0));
        assert_eq!(drift_at(&drift, 9), (8.0, 0.0));
        assert_eq!(drift_at(&[], 3), (0.0, 0.0));
    }
}
