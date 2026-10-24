//! # Euclidean feature transform
//!
//! For every pixel of a grid the nearest "feature" pixel - a port of SciPy's
//! `ndimage.distance_transform_edt(..., return_indices=True)`
//! (scipy/ndimage/src/ni_morphology.c: `_ComputeFT` / `_VoronoiFT`, Maurer,
//! Qi & Raghavan 2003), 2D, keeping its arithmetic and the order it settles
//! ties in - so objects expanded with it get CellProfiler's pixels
//! (ExpandOrShrinkObjects expands with exactly that SciPy call).
//!
//! First every column is scanned on its own (the nearest feature in the
//! column), then every row picks among its columns' candidates the lower
//! envelope of their distance parabolas.
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **AGPL-3.0**.

/// The nearest feature of every pixel of a grid (row-major), 8 bytes per
/// pixel.
pub(crate) struct NearestFeatures {
    /// (row, column) per pixel; row -1: the grid has no feature.
    ft: Vec<[i32; 2]>,
}

impl NearestFeatures {
    /// Pixel `i`'s nearest feature as (row, column).
    pub(crate) fn get(&self, i: usize) -> Option<(i64, i64)> {
        let [r, c] = self.ft[i];
        (r >= 0).then_some((r as i64, c as i64))
    }

    #[cfg(test)]
    fn len(&self) -> usize {
        self.ft.len()
    }
}

/// The nearest feature of every pixel of a `width` x `height` grid
/// (row-major) - none when the grid has no feature.
pub(crate) fn nearest_features(
    width: usize,
    height: usize,
    is_feature: impl Fn(usize, usize) -> bool,
) -> NearestFeatures {
    // (row, column) per pixel, row -1: no candidate (yet).
    let mut ft: Vec<[i32; 2]> = vec![[-1, 0]; width * height];
    let mut line: Vec<[i32; 2]> = Vec::with_capacity(width.max(height));
    let mut envelope: Vec<usize> = Vec::with_capacity(width.max(height));
    let mut copy: Vec<[i32; 2]> = Vec::with_capacity(width.max(height));

    // Along the rows of each column (SciPy's d = 0): the column's features.
    for x in 0..width {
        line.clear();
        line.extend((0..height).map(|y| {
            if is_feature(x, y) {
                [y as i32, x as i32]
            } else {
                [-1, 0]
            }
        }));
        voronoi_ft(&mut line, [0, x as i32], 0, &mut envelope, &mut copy);
        for (y, f) in line.iter().enumerate() {
            ft[y * width + x] = *f;
        }
    }
    // Along each row (d = 1): the nearest of its columns' candidates.
    for y in 0..height {
        let row = &mut ft[y * width..(y + 1) * width];
        voronoi_ft(row, [y as i32, 0], 1, &mut envelope, &mut copy);
    }
    NearestFeatures { ft }
}

/// SciPy's `_VoronoiFT` along one line (`d`: the axis it runs along, 0 =
/// down a column, 1 = along a row; `coor`: the line's coordinate on the
/// other axis): each position gets the candidate (row -1: none) nearest to
/// it. `g` and `f` are scratch space.
fn voronoi_ft(
    line: &mut [[i32; 2]],
    coor: [i32; 2],
    d: usize,
    g: &mut Vec<usize>,
    f: &mut Vec<[i32; 2]>,
) {
    let other = 1 - d;
    f.clear();
    f.extend_from_slice(line);
    let f = &*f;
    g.clear();
    for (ii, fi) in f.iter().enumerate() {
        if fi[0] < 0 {
            continue;
        }
        let fd = fi[d] as f64;
        let tw = (fi[other] - coor[other]) as f64;
        let w_r = tw * tw;
        while g.len() >= 2 {
            let idx1 = g[g.len() - 1];
            let idx2 = g[g.len() - 2];
            let f1 = f[idx1][d] as f64;
            let a = f1 - f[idx2][d] as f64;
            let b = fd - f1;
            let tu = (f[idx2][other] - coor[other]) as f64;
            let tv = (f[idx1][other] - coor[other]) as f64;
            let (u_r, v_r) = (tu * tu, tv * tv);
            let c = a + b;
            if c * v_r - b * u_r - a * w_r - a * b * c <= 0.0 {
                break;
            }
            g.pop();
        }
        g.push(ii);
    }
    if g.is_empty() {
        return;
    }
    // The squared distance from position `ii` to candidate `f[k]`.
    let distance = |k: usize, ii: usize| -> f64 {
        let along = (f[k][d] as i64 - ii as i64) as f64;
        let across = (f[k][other] - coor[other]) as f64;
        along * along + across * across
    };
    let mut l = 0;
    for (ii, out) in line.iter_mut().enumerate() {
        let mut delta1 = distance(g[l], ii);
        while l + 1 < g.len() {
            let delta2 = distance(g[l + 1], ii);
            if delta1 <= delta2 {
                break;
            }
            delta1 = delta2;
            l += 1;
        }
        *out = f[g[l]];
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Brute force: the nearest feature's squared distance.
    fn nearest_distance2(
        width: usize,
        height: usize,
        features: &[(usize, usize)],
        x: usize,
        y: usize,
    ) -> i64 {
        features
            .iter()
            .map(|&(fx, fy)| {
                let (dx, dy) = (fx as i64 - x as i64, fy as i64 - y as i64);
                dx * dx + dy * dy
            })
            .min()
            .unwrap_or_else(|| panic!("no feature in {width}x{height}"))
    }

    #[test]
    fn every_pixel_gets_a_nearest_feature() {
        let (width, height) = (23, 17);
        // A deterministic scatter of features.
        let features: Vec<(usize, usize)> = (0..30)
            .map(|i| ((i * 7 + 3) % width, (i * 11 + 5) % height))
            .collect();
        let ft = nearest_features(width, height, |x, y| features.contains(&(x, y)));
        for y in 0..height {
            for x in 0..width {
                let (r, c) = ft.get(y * width + x).unwrap();
                assert!(features.contains(&(c as usize, r as usize)));
                let (dx, dy) = (c - x as i64, r - y as i64);
                assert_eq!(
                    dx * dx + dy * dy,
                    nearest_distance2(width, height, &features, x, y),
                    "pixel ({x}, {y})"
                );
            }
        }
    }

    #[test]
    fn without_features_nothing_is_nearest() {
        let ft = nearest_features(4, 3, |_, _| false);
        assert!((0..ft.len()).all(|i| ft.get(i).is_none()));
    }
}
