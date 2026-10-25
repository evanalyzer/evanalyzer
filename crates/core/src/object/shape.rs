//! # shape
//!
//! An object's shape measurements as CellProfiler's MeasureObjectSizeShape
//! gives them (cellprofiler/modules/measureobjectsizeshape.py, CellProfiler
//! 4.2.8, with scikit-image's `regionprops` and centrosome): the Feret
//! diameters, the convex area, the Euler number and the radii - computed
//! from the mask when asked for (at export), not kept on the object.
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-10-24
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.

use super::Object;

/// The radii of an object: the distances of its pixels to the nearest
/// pixel outside it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Radii {
    pub max: f64,
    pub mean: f64,
    pub median: f64,
}

/// All of an object's shape values at once (each computed from the same
/// rows and padded mask).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Shape {
    pub min_feret: f64,
    pub max_feret: f64,
    pub convex_area: usize,
    pub euler_number: i64,
    pub radii: Option<Radii>,
}

/// Buffers reused from object to object (export measures millions).
#[derive(Default)]
struct Scratch {
    padded: Vec<u8>,
    squared: Vec<i64>,
    last: Vec<usize>,
    v: Vec<usize>,
    z: Vec<(i64, i64)>,
    f: Vec<i64>,
    inside: Vec<i64>,
    points: Vec<(f64, f64)>,
}

thread_local! {
    static SCRATCH: std::cell::RefCell<Scratch> = std::cell::RefCell::new(Scratch::default());
}

impl Object {
    /// Every shape value - what the export writes.
    pub fn shape(&self) -> Shape {
        let rows = self.row_extremes();
        SCRATCH.with(|scratch| {
            let scratch = &mut *scratch.borrow_mut();
            let (min_feret, max_feret) = feret_from_rows(&rows, &mut scratch.points);
            let convex_area =
                convex_area_from_rows(&rows, self.bbox, self.area, &mut scratch.points);
            let (pw, ph) = self.pad_into(&mut scratch.padded);
            let euler_number = euler_of_padded(&scratch.padded, pw, ph);
            let radii = radii_of_padded(pw, ph, scratch);
            Shape {
                min_feret,
                max_feret,
                convex_area,
                euler_number,
                radii,
            }
        })
    }

    /// The mask as rows of 0/1 bytes with a 1-pixel border of 0 around, into
    /// `out` - returns the padded width and height.
    fn pad_into(&self, out: &mut Vec<u8>) -> (usize, usize) {
        let [x0, y0, x1, y1] = self.bbox.map(|v| v as usize);
        let (w, h) = (x1 - x0 + 1, y1 - y0 + 1);
        let (pw, ph) = (w + 2, h + 2);
        out.clear();
        out.resize(pw * ph, 0);
        for r in 0..h {
            let Some(bits) = self.mask_data.get(r * w..(r + 1) * w) else {
                break;
            };
            let row = &mut out[(r + 1) * pw + 1..(r + 1) * pw + 1 + w];
            for (o, bit) in row.iter_mut().zip(bits.iter().by_vals()) {
                *o = bit as u8;
            }
        }
        (pw, ph)
    }

    /// Per row the leftmost and rightmost pixel (only they can be hull
    /// vertices), as (y, left x, right x), top to bottom.
    fn row_extremes(&self) -> Vec<(i64, i64, i64)> {
        let [x0, y0, x1, y1] = self.bbox.map(|v| v as usize);
        let w = x1 - x0 + 1;
        (0..=y1 - y0)
            .filter_map(|r| {
                let row = self.mask_data.get(r * w..(r + 1) * w)?;
                let (left, right) = (row.first_one()?, row.last_one()?);
                Some(((y0 + r) as i64, (x0 + left) as i64, (x0 + right) as i64))
            })
            .collect()
    }

    /// The minimum and maximum Feret (caliper) diameter over the pixel
    /// centres, as centrosome's `feret_diameter` on their convex hull: the
    /// largest distance between two hull vertices, and the smallest width
    /// of the hull (rotating calipers). 0 / 0 for a single pixel.
    pub fn feret_diameters(&self) -> (f64, f64) {
        feret_from_rows(&self.row_extremes(), &mut Vec::new())
    }

    /// The number of pixels whose centre lies in (or on) the convex hull of
    /// the object's pixels - scikit-image's `convex_hull_image` sum
    /// ("ConvexArea" in CellProfiler).
    pub fn convex_area(&self) -> usize {
        convex_area_from_rows(&self.row_extremes(), self.bbox, self.area, &mut Vec::new())
    }

    /// The Euler number (objects - holes) with 8-connected pixels, as
    /// scikit-image's `euler_number` on the object's image.
    pub fn euler_number(&self) -> i64 {
        self.shape().euler_number
    }

    /// The radii: each pixel's Euclidean distance to the nearest pixel not
    /// in the object (the image edge doesn't count - CellProfiler pads the
    /// object's image), their maximum, mean and median.
    pub fn radii(&self) -> Option<Radii> {
        self.shape().radii
    }
}

/// [`Object::feret_diameters`] from the rows' extreme pixels.
fn feret_from_rows(rows: &[(i64, i64, i64)], points: &mut Vec<(f64, f64)>) -> (f64, f64) {
    points.clear();
    for &(y, left, right) in rows {
        // In (y, x) order already: the hull needs no sort.
        points.push((y as f64, left as f64));
        if right != left {
            points.push((y as f64, right as f64));
        }
    }
    let hull = hull_of_sorted(points);
    feret_of_hull(&hull)
}

/// [`Object::convex_area`] from the rows' extreme pixels: the hull of the
/// midpoints of the pixels' edges (scikit-image takes them of each row's
/// and column's first and last pixel; the rows' alone span the same hull -
/// any pixel lies between its row's two).
fn convex_area_from_rows(
    rows: &[(i64, i64, i64)],
    bbox: [u32; 4],
    area: usize,
    points: &mut Vec<(f64, f64)>,
) -> usize {
    points.clear();
    for &(y, left, right) in rows {
        let y = y as f64;
        for x in [left as f64, right as f64] {
            points.extend([(x - 0.5, y), (x + 0.5, y), (x, y - 0.5), (x, y + 0.5)]);
        }
    }
    let hull = convex_hull(points);
    if hull.len() < 3 {
        return area;
    }
    // An edge a -> b of the counter-clockwise hull: inside is where
    // cross(b - a, p - a) >= -1e-10·|b - a| (a centre on the hull, within
    // scikit-image's tolerance, counts) - for a row y: x <= c / ey for edges
    // going down (ey > 0), x >= c / ey for edges going up.
    let n = hull.len();
    let edge = |i: usize| {
        let (ax, ay) = hull[i];
        let (bx, by) = hull[(i + 1) % n];
        (ax, ay, bx - ax, by - ay)
    };
    let bound = |(ax, ay, ex, ey): (f64, f64, f64, f64), yc: f64| {
        let tol = 1e-10 * (ex * ex + ey * ey).sqrt();
        (ex * (yc - ay) + ey * ax + tol) / ey
    };
    let y_min = hull.iter().map(|p| p.1).fold(f64::MAX, f64::min);
    let y_max = hull.iter().map(|p| p.1).fold(f64::MIN, f64::max);
    // The edges bounding each side, in the order rows meet them.
    let mut right: Vec<(f64, f64, f64, f64)> = (0..n).map(edge).filter(|e| e.3 > 0.0).collect();
    let mut left: Vec<(f64, f64, f64, f64)> = (0..n).map(edge).filter(|e| e.3 < 0.0).collect();
    right.sort_by(|a, b| a.1.total_cmp(&b.1));
    left.sort_by(|a, b| (a.1 + a.3).total_cmp(&(b.1 + b.3)));
    let (mut ri, mut li) = (0usize, 0usize);
    let [x0, y0, x1, y1] = bbox.map(|v| v as i64);
    let mut count = 0usize;
    for y in y0..=y1 {
        let yc = y as f64;
        if yc < y_min - 1e-10 || yc > y_max + 1e-10 {
            continue;
        }
        // The edge spanning this row on each side.
        while ri + 1 < right.len() && right[ri].1 + right[ri].3 < yc {
            ri += 1;
        }
        while li + 1 < left.len() && left[li].1 < yc {
            li += 1;
        }
        let hi = right.get(ri).map_or(f64::MAX, |e| bound(*e, yc));
        let lo = left.get(li).map_or(f64::MIN, |e| bound(*e, yc));
        let first = lo.ceil().max(x0 as f64) as i64;
        let last = hi.floor().min(x1 as f64) as i64;
        if last >= first {
            count += (last - first + 1) as usize;
        }
    }
    count
}

/// The Euler number of a padded 0/1 mask: Gray's bit quads with 8-connected
/// pixels, (Q1 - Q3 - 2·QD) / 4 (scikit-image's `euler_number`).
fn euler_of_padded(rows: &[u8], pw: usize, ph: usize) -> i64 {
    let (mut q1, mut q3, mut qd) = (0i64, 0i64, 0i64);
    for y in 0..ph - 1 {
        let (top, bottom) = (
            &rows[y * pw..(y + 1) * pw],
            &rows[(y + 1) * pw..(y + 2) * pw],
        );
        for x in 0..pw - 1 {
            let (a, b, c, d) = (top[x], top[x + 1], bottom[x], bottom[x + 1]);
            match a + b + c + d {
                1 => q1 += 1,
                3 => q3 += 1,
                2 if a == d => qd += 1,
                _ => {}
            }
        }
    }
    (q1 - q3 - 2 * qd) / 4
}

/// The radii from the padded mask in `scratch.padded` (centrosome's
/// `median_of_labels` for the median: the middle one, or the mean of the
/// two middle ones).
fn radii_of_padded(pw: usize, ph: usize, scratch: &mut Scratch) -> Option<Radii> {
    squared_distances(pw, ph, scratch);
    let Scratch {
        padded,
        squared,
        inside,
        ..
    } = scratch;
    inside.clear();
    let mut sum = 0.0;
    for (p, d2) in padded.iter().zip(squared.iter()) {
        if *p == 1 {
            inside.push(*d2);
            sum += (*d2 as f64).sqrt();
        }
    }
    let n = inside.len();
    if n == 0 {
        return None;
    }
    let max = (*inside.iter().max()? as f64).sqrt();
    let low = (n - 1) / 2;
    let (_, at_low, above) = inside.select_nth_unstable(low);
    let at_low = (*at_low as f64).sqrt();
    let median = if n % 2 == 1 {
        at_low
    } else {
        let next = (*above.iter().min()? as f64).sqrt();
        (at_low + next) / 2.0
    };
    Some(Radii {
        max,
        mean: sum / n as f64,
        median,
    })
}

/// A fraction num / den (den > 0), or ∓∞ (den 0, num -1 / 1).
type Frac = (i64, i64);

fn frac_le((an, ad): Frac, (bn, bd): Frac) -> bool {
    match (ad, bd) {
        (0, 0) => an <= bn,
        (0, _) => an < 0,
        (_, 0) => bn > 0,
        _ => an as i128 * bd as i128 <= bn as i128 * ad as i128,
    }
}

/// The squared Euclidean distance of every cell of the padded mask in
/// `scratch.padded` (`width` x `height`) to the nearest 0 cell, into
/// `scratch.squared` - exact, in integers: Felzenszwalb & Huttenlocher's
/// lower envelope of parabolas, its breakpoints kept as fractions. Every
/// column has a 0 (the border).
fn squared_distances(width: usize, height: usize, scratch: &mut Scratch) {
    let Scratch {
        padded: inside,
        squared: d,
        last,
        v,
        z,
        f,
        ..
    } = scratch;
    const FAR: i64 = i64::MAX / 4;
    d.clear();
    d.resize(width * height, FAR);
    // Down each column: the distance to the nearest 0 in it - swept row by
    // row (in memory order), each column's last 0 kept.
    last.clear();
    last.resize(width, usize::MAX);
    for y in 0..height {
        for x in 0..width {
            let i = y * width + x;
            if inside[i] == 0 {
                last[x] = y;
            }
            if last[x] != usize::MAX {
                d[i] = (y - last[x]) as i64;
            }
        }
    }
    last.fill(usize::MAX);
    for y in (0..height).rev() {
        for x in 0..width {
            let i = y * width + x;
            if inside[i] == 0 {
                last[x] = y;
            }
            if last[x] != usize::MAX {
                d[i] = d[i].min((last[x] - y) as i64);
            }
            d[i] *= d[i];
        }
    }
    // Along each row: the lower envelope of the parabolas (x - q)² + f(q).
    v.clear();
    v.resize(width, 0);
    z.clear();
    z.resize(width + 1, (0, 0));
    f.clear();
    f.resize(width, 0);
    for y in 0..height {
        let row = &mut d[y * width..(y + 1) * width];
        f.copy_from_slice(row);
        let mut k = 0usize;
        v[0] = 0;
        z[0] = (-1, 0);
        z[1] = (1, 0);
        for q in 1..width {
            let qi = q as i64;
            let mut s;
            loop {
                let p = v[k] as i64;
                s = ((f[q] + qi * qi) - (f[v[k]] + p * p), 2 * (qi - p));
                if k > 0 && frac_le(s, z[k]) {
                    k -= 1;
                } else {
                    break;
                }
            }
            k += 1;
            v[k] = q;
            z[k] = s;
            z[k + 1] = (1, 0);
        }
        let mut k = 0usize;
        for (q, out) in row.iter_mut().enumerate() {
            let qi = q as i64;
            // z[k + 1] < q
            while !frac_le((qi, 1), z[k + 1]) {
                k += 1;
            }
            let p = v[k] as i64;
            *out = (qi - p) * (qi - p) + f[v[k]];
        }
    }
}

/// The convex hull of `points` (reordered), counter-clockwise in the
/// (x, y-down) plane, without collinear points - Andrew's monotone chain.
pub(crate) fn convex_hull(points: &mut [(f64, f64)]) -> Vec<(f64, f64)> {
    points.sort_unstable_by(|a, b| a.0.total_cmp(&b.0).then(a.1.total_cmp(&b.1)));
    hull_of_sorted(points)
}

/// Andrew's monotone chain over `points` sorted lexicographically (any two
/// coordinates; the orientation follows).
fn hull_of_sorted(points: &[(f64, f64)]) -> Vec<(f64, f64)> {
    let mut unique: Vec<(f64, f64)> = Vec::with_capacity(points.len());
    for p in points.iter() {
        if unique.last() != Some(p) {
            unique.push(*p);
        }
    }
    if unique.len() < 3 {
        return unique;
    }
    let cross = |o: (f64, f64), a: (f64, f64), b: (f64, f64)| {
        (a.0 - o.0) * (b.1 - o.1) - (a.1 - o.1) * (b.0 - o.0)
    };
    let mut hull: Vec<(f64, f64)> = Vec::with_capacity(unique.len() * 2);
    for pass in 0..2 {
        let start = hull.len();
        let iter: Box<dyn Iterator<Item = &(f64, f64)>> = if pass == 0 {
            Box::new(unique.iter())
        } else {
            Box::new(unique.iter().rev())
        };
        for &p in iter {
            while hull.len() >= start + 2
                && cross(hull[hull.len() - 2], hull[hull.len() - 1], p) <= 0.0
            {
                hull.pop();
            }
            hull.push(p);
        }
        hull.pop();
    }
    hull
}

/// The minimum width and the diameter of a convex polygon (vertices in
/// order, no three collinear) - rotating calipers: for each edge the
/// farthest vertex (it moves on with the edges), whose distance is the
/// width across that edge; each edge's ends and their farthest vertex are
/// the antipodal pairs, the longest of which is the diameter.
fn feret_of_hull(hull: &[(f64, f64)]) -> (f64, f64) {
    let d2 = |a: (f64, f64), b: (f64, f64)| (a.0 - b.0).powi(2) + (a.1 - b.1).powi(2);
    let n = hull.len();
    if n < 3 {
        let max = if n == 2 {
            d2(hull[0], hull[1]).sqrt()
        } else {
            0.0
        };
        return (0.0, max);
    }
    // Twice the area of the triangle a, b, p: the distance of p from the
    // line through a and b, times |b - a|.
    let area2 = |a: (f64, f64), b: (f64, f64), p: (f64, f64)| {
        ((b.0 - a.0) * (p.1 - a.1) - (b.1 - a.1) * (p.0 - a.0)).abs()
    };
    let (mut min, mut max2) = (f64::INFINITY, 0.0f64);
    let mut j = 1;
    for i in 0..n {
        let (a, b) = (hull[i], hull[(i + 1) % n]);
        // The farthest vertex from edge a-b (while the next one isn't nearer).
        while area2(a, b, hull[(j + 1) % n]) >= area2(a, b, hull[j % n]) {
            j += 1;
            if j > 3 * n {
                break;
            }
        }
        let far = hull[j % n];
        min = min.min(area2(a, b, far) / d2(a, b).sqrt());
        max2 = max2.max(d2(a, far)).max(d2(b, far));
        // A vertex as far as `far` (a parallel edge opposite) pairs too.
        let next = hull[(j + 1) % n];
        max2 = max2.max(d2(a, next)).max(d2(b, next));
    }
    (min, max2.sqrt())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::object::ObjectInit;
    use bitvec::prelude::*;
    use evanalyzer_cfg::core_types::ObjectId;

    #[derive(serde::Deserialize)]
    struct Case {
        width: usize,
        labels: Vec<u32>,
        area: Vec<f64>,
        perimeter: Vec<Option<f64>>,
        major_axis_length: Vec<f64>,
        minor_axis_length: Vec<f64>,
        eccentricity: Vec<f64>,
        form_factor: Vec<Option<f64>>,
        extent: Vec<f64>,
        solidity: Vec<Option<f64>>,
        euler_number: Vec<f64>,
        max_radius: Vec<f64>,
        mean_radius: Vec<f64>,
        median_radius: Vec<f64>,
        convex_area: Vec<f64>,
        min_feret_diameter: Vec<f64>,
        max_feret_diameter: Vec<f64>,
        equivalent_diameter: Vec<f64>,
    }

    fn objects(labels: &[u32], width: usize) -> Vec<Object> {
        let max = labels.iter().copied().max().unwrap_or(0);
        (1..=max)
            .map(|label| {
                let pixels: Vec<(u32, u32)> = (0..labels.len())
                    .filter(|&p| labels[p] == label)
                    .map(|p| ((p % width) as u32, (p / width) as u32))
                    .collect();
                let x0 = pixels.iter().map(|p| p.0).min().unwrap();
                let y0 = pixels.iter().map(|p| p.1).min().unwrap();
                let x1 = pixels.iter().map(|p| p.0).max().unwrap();
                let y1 = pixels.iter().map(|p| p.1).max().unwrap();
                let w = (x1 - x0 + 1) as usize;
                let mut mask = BitVec::<u64, Lsb0>::repeat(false, w * (y1 - y0 + 1) as usize);
                let (mut sx, mut sy, mut sx2, mut sy2, mut sxy) = (0u64, 0u64, 0u64, 0u64, 0u64);
                for (x, y) in &pixels {
                    mask.set((y - y0) as usize * w + (x - x0) as usize, true);
                    let (x, y) = (*x as u64, *y as u64);
                    sx += x;
                    sy += y;
                    sx2 += x * x;
                    sy2 += y * y;
                    sxy += x * y;
                }
                Object::new(ObjectInit {
                    id: ObjectId(label as u128),
                    bbox: [x0, y0, x1, y1],
                    area: pixels.len(),
                    mask_data: mask,
                    sum_x: sx,
                    sum_y: sy,
                    sum_x2: sx2,
                    sum_y2: sy2,
                    sum_xy: sxy,
                    ..Default::default()
                })
            })
            .collect()
    }

    fn close(got: f64, expected: f64, tolerance: f64, what: &str) {
        assert!(
            (got - expected).abs() <= tolerance * expected.abs().max(1.0),
            "{what}: {got} vs {expected}"
        );
    }

    /// Feret, convex area, Euler number, radii, extent and equivalent
    /// diameter as CellProfiler's MeasureObjectSizeShape (fixture from
    /// `tests/object_shape_cellprofiler.py`); the axes and eccentricity
    /// too (stored as f32).
    #[test]
    fn shape_values_are_cellprofilers() {
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/object_shape_cellprofiler.json"
        );
        let cases: Vec<Case> =
            serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
        let case = &cases[0];
        let objects = objects(&case.labels, case.width);
        assert_eq!(objects.len(), case.area.len());
        for (i, o) in objects.iter().enumerate() {
            let what = |name: &str| format!("object {}: {name}", i + 1);
            assert_eq!(o.area as f64, case.area[i]);
            let (min, max) = o.feret_diameters();
            close(min, case.min_feret_diameter[i], 1e-9, &what("min Feret"));
            close(max, case.max_feret_diameter[i], 1e-9, &what("max Feret"));
            assert_eq!(
                o.convex_area() as f64,
                case.convex_area[i],
                "{}",
                what("convex area")
            );
            assert_eq!(
                o.euler_number() as f64,
                case.euler_number[i],
                "{}",
                what("Euler number")
            );
            let r = o.radii().unwrap();
            close(r.max, case.max_radius[i], 1e-9, &what("max radius"));
            close(r.mean, case.mean_radius[i], 1e-9, &what("mean radius"));
            close(
                r.median,
                case.median_radius[i],
                1e-9,
                &what("median radius"),
            );
            let [x0, y0, x1, y1] = o.bbox;
            let bbox_area = ((x1 - x0 + 1) * (y1 - y0 + 1)) as f64;
            close(
                o.area as f64 / bbox_area,
                case.extent[i],
                1e-12,
                &what("extent"),
            );
            close(
                (4.0 * o.area as f64 / std::f64::consts::PI).sqrt(),
                case.equivalent_diameter[i],
                1e-12,
                &what("equivalent diameter"),
            );
            let e = o.get_ellipse();
            close(
                e.major as f64,
                case.major_axis_length[i],
                1e-5,
                &what("major axis"),
            );
            close(
                e.minor as f64,
                case.minor_axis_length[i],
                1e-4,
                &what("minor axis"),
            );
            close(
                e.eccentricity as f64,
                case.eccentricity[i],
                1e-4,
                &what("eccentricity"),
            );
        }
        // Different by design (ImageJ's traced perimeter, a hull over the
        // pixel corners) - printed to see by how much.
        for (i, o) in objects.iter().enumerate().take(5) {
            eprintln!(
                "object {}: perimeter {:.2} (CellProfiler {:?}), circularity {:.3} (form factor {:?}), solidity {:.3} ({:?})",
                i + 1,
                o.get_perimeter(),
                case.perimeter[i],
                o.circularity(),
                case.form_factor[i],
                o.get_solidity(),
                case.solidity[i]
            );
        }
    }

    /// Time of each shape value on one big round object (release, ignored):
    /// `cargo test --release -p evanalyzer_core --lib bench_shape -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn bench_shape() {
        let r: i64 = 700;
        let side = (2 * r + 1) as usize;
        let mut labels = vec![0u32; side * side];
        for y in 0..side {
            for x in 0..side {
                let (dx, dy) = (x as i64 - r, y as i64 - r);
                if dx * dx + dy * dy <= r * r {
                    labels[y * side + x] = 1;
                }
            }
        }
        let o = objects(&labels, side).remove(0);
        let time = |name: &str, f: &dyn Fn()| {
            let t = std::time::Instant::now();
            f();
            eprintln!("BENCH {name}: {:?} ({} px)", t.elapsed(), o.area);
        };
        time("feret", &|| {
            std::hint::black_box(o.feret_diameters());
        });
        time("convex area", &|| {
            std::hint::black_box(o.convex_area());
        });
        time("all at once (shape)", &|| {
            std::hint::black_box(o.shape());
        });
        time("solidity (existing)", &|| {
            std::hint::black_box(o.get_solidity());
        });
    }

    /// Time per object of each shape value on a small round object (r = 11,
    /// as `bench_export --object-size 24`), 100,000 times (release, ignored):
    /// `cargo test --release -p evanalyzer_core --lib bench_small_shape -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn bench_small_shape() {
        let r: i64 = 11;
        let side = (2 * r + 1) as usize;
        let mut labels = vec![0u32; side * side];
        for y in 0..side {
            for x in 0..side {
                let (dx, dy) = (x as i64 - r, y as i64 - r);
                if dx * dx + dy * dy <= r * r {
                    labels[y * side + x] = 1;
                }
            }
        }
        let o = objects(&labels, side).remove(0);
        let n = 100_000;
        let time = |name: &str, f: &dyn Fn()| {
            let t = std::time::Instant::now();
            for _ in 0..n {
                f();
            }
            eprintln!(
                "BENCH small {name}: {:.2} µs per object",
                t.elapsed().as_secs_f64() * 1e6 / n as f64
            );
        };
        time("feret", &|| {
            std::hint::black_box(o.feret_diameters());
        });
        time("convex area", &|| {
            std::hint::black_box(o.convex_area());
        });
        time("all at once (shape)", &|| {
            std::hint::black_box(o.shape());
        });
        time("solidity (existing)", &|| {
            std::hint::black_box(o.get_solidity());
        });
        let scratch = std::cell::RefCell::new(Scratch::default());
        time("row extremes", &|| {
            std::hint::black_box(o.row_extremes());
        });
        time("pad", &|| {
            std::hint::black_box(o.pad_into(&mut scratch.borrow_mut().padded));
        });
        let (pw, ph) = o.pad_into(&mut scratch.borrow_mut().padded);
        time("euler", &|| {
            std::hint::black_box(euler_of_padded(&scratch.borrow().padded, pw, ph));
        });
        time("squared distances", &|| {
            squared_distances(pw, ph, &mut scratch.borrow_mut());
        });
        time("radii", &|| {
            std::hint::black_box(radii_of_padded(pw, ph, &mut scratch.borrow_mut()));
        });
    }

    /// The rotating calipers give the brute-force diameter and width.
    #[test]
    fn calipers_match_brute_force() {
        let mut seed = 9u64;
        let mut next = move || {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((seed >> 33) % 40) as f64
        };
        for _ in 0..500 {
            let mut points: Vec<(f64, f64)> = (0..12).map(|_| (next(), next())).collect();
            let hull = convex_hull(&mut points);
            if hull.len() < 3 {
                continue;
            }
            let (min, max) = feret_of_hull(&hull);
            let d2 = |a: (f64, f64), b: (f64, f64)| (a.0 - b.0).powi(2) + (a.1 - b.1).powi(2);
            let brute_max = hull
                .iter()
                .flat_map(|a| hull.iter().map(move |b| d2(*a, *b)))
                .fold(0.0, f64::max)
                .sqrt();
            let brute_min = (0..hull.len())
                .map(|i| {
                    let (a, b) = (hull[i], hull[(i + 1) % hull.len()]);
                    hull.iter()
                        .map(|p| {
                            ((b.0 - a.0) * (p.1 - a.1) - (b.1 - a.1) * (p.0 - a.0)).abs()
                                / d2(a, b).sqrt()
                        })
                        .fold(0.0, f64::max)
                })
                .fold(f64::INFINITY, f64::min);
            assert!(
                (max - brute_max).abs() < 1e-9,
                "{hull:?}: {max} vs {brute_max}"
            );
            assert!(
                (min - brute_min).abs() < 1e-9,
                "{hull:?}: {min} vs {brute_min}"
            );
        }
    }
}
