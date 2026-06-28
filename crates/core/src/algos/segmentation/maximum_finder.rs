//! # maximum_finder
//!
//! **Author:** Joachim Danmayr
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **AGPL-3.0**.
//!
//! Faithful Rust port of ImageJ's `MaximumFinder` watershed segmentation
//! (`ij/plugin/filter/MaximumFinder.java`), restricted to the EDM-watershed use
//! case (`outputType = SEGMENTED`, `isEDM = true`, `strict = false`,
//! `excludeOnEdges = false`, no intensity threshold).
//!
//! The reference C++ port this is derived from lives in `docs/watershed/`.
//!
//! The algorithm:
//! 1. Find every local maximum of the (float) Euclidean distance map, using a
//!    sub-pixel "true height" estimate so that the integer-ish ridges of an EDM
//!    are compared at their interpolated heights (this is what makes a small
//!    `tolerance` such as `0.5` meaningful).
//! 2. Flood out from each maximum, highest first; maxima that protrude less than
//!    `tolerance` above the ridge connecting them to a higher maximum are merged
//!    into it. The surviving maxima and their plateaus are marked `MAX_AREA`.
//! 3. Rasterize to an 8-bit landscape (`MAX_AREA` -> 255, background -> 0) and
//!    run a level-by-level constrained dilation (the "fate table") that grows the
//!    255-regions downhill without letting two regions merge, drawing 1-pixel
//!    watershed lines at the necks.
//!
//! The returned mask is `255` inside particles and `0` on watershed lines and
//! background — exactly ImageJ's `Process > Binary > Watershed` output.

const MAXIMUM: u8 = 1; // marks local maxima (irrespective of noise tolerance)
const LISTED: u8 = 2; // marks points currently in the list
const PROCESSED: u8 = 4; // marks points processed previously
const MAX_AREA: u8 = 8; // marks areas near a maximum, within the tolerance
const EQUAL: u8 = 16; // marks contiguous maximum points of equal level
const MAX_POINT: u8 = 32; // marks a single point standing for a maximum
const ELIMINATED: u8 = 64; // marks maxima that have been eliminated before watershed

const SQRT2: f32 = 1.414_213_6;

/// Directions to the 8 neighboring pixels, clockwise: 0=N(-y), 1=NE, 2=E(+x), …7=NW.
const DIR_X_OFFSET: [i32; 8] = [0, 1, 1, 1, 0, -1, -1, -1];
const DIR_Y_OFFSET: [i32; 8] = [-1, -1, 0, 1, 1, 1, 0, -1];

struct MaximumFinder {
    width: usize,
    height: usize,
    dir_offset: [i32; 8],
    int_encode_x_mask: i64,
    int_encode_y_mask: i64,
    int_encode_shift: u32,
}

/// Runs ImageJ EDM watershed segmentation on `edm` (a float distance map of size
/// `width × height`) and returns a mask where particles are `255` and watershed
/// lines / background are `0`.
pub fn watershed_segment_edm(edm: &[f32], width: usize, height: usize, tolerance: f32) -> Vec<u8> {
    let mut mf = MaximumFinder::new(width, height);
    mf.find_maxima_segmented(edm, tolerance)
}

impl MaximumFinder {
    fn new(width: usize, height: usize) -> Self {
        let w = width as i32;
        // intEncodeShift = ceil(log2(width)) with the same loop as ImageJ.
        let mut shift = 0u32;
        let mut mult = 1i64;
        while mult < width as i64 {
            shift += 1;
            mult *= 2;
        }
        let int_encode_x_mask = mult - 1;
        Self {
            width,
            height,
            dir_offset: [-w, -w + 1, 1, w + 1, w, w - 1, -1, -w - 1],
            int_encode_x_mask,
            int_encode_y_mask: !int_encode_x_mask,
            int_encode_shift: shift,
        }
    }

    fn find_maxima_segmented(&mut self, ip: &[f32], tolerance: f32) -> Vec<u8> {
        let n = self.width * self.height;
        let mut types = vec![0u8; n];

        let mut global_min = f32::MAX;
        let mut global_max = -f32::MAX;
        for &v in ip.iter() {
            if global_min > v {
                global_min = v;
            }
            if global_max < v {
                global_max = v;
            }
        }

        if global_max <= global_min {
            return vec![0u8; n]; // flat image: nothing to segment
        }

        // `true_edm_height` is a pure function of `ip`, but both passes below
        // probe it for every pixel and again for several of its neighbors -
        // precomputing it once turns that repeated, branch-heavy recomputation
        // into a single array lookup at each call site.
        let true_heights = self.compute_true_heights(ip);

        let max_points =
            self.get_sorted_max_points(ip, &true_heights, &mut types, global_min, global_max);

        // For float EDMs the integer-encoded sort order may be off by this much.
        let max_sorting_error = 1.1 * (SQRT2 / 2.0);
        self.analyze_and_mark_maxima(
            ip,
            &true_heights,
            &mut types,
            &max_points,
            global_min,
            tolerance,
            max_sorting_error,
        );

        let mut out = self.make8bit(ip, &types, global_max);
        self.cleanup_maxima(&mut out, &mut types, &max_points);
        self.watershed_segment(&mut out);
        self.watershed_post_process(&mut out);
        out
    }

    /// Find all local maxima (using the sub-pixel "true" EDM height) and return
    /// them encoded as `value << 32 | offset`, sorted ascending by value.
    fn get_sorted_max_points(
        &self,
        ip: &[f32],
        true_heights: &[f32],
        types: &mut [u8],
        global_min: f32,
        global_max: f32,
    ) -> Vec<u64> {
        let (w, h) = (self.width, self.height);
        let mut n_max = 0usize;
        for y in 0..h {
            for x in 0..w {
                let i = x + y * w;
                let v = ip[i];
                if v == global_min {
                    continue;
                }
                let v_true = true_heights[i];
                let is_inner = y != 0 && y != h - 1 && x != 0 && x != w - 1;
                let mut is_max = true;
                for d in 0..8 {
                    if is_inner || self.is_within(x as i32, y as i32, d) {
                        let n_idx = (i as i32 + self.dir_offset[d]) as usize;
                        let v_neighbor = ip[n_idx];
                        let v_neighbor_true = true_heights[n_idx];
                        if v_neighbor > v && v_neighbor_true > v_true {
                            is_max = false;
                            break;
                        }
                    }
                }
                if is_max {
                    types[i] = MAXIMUM;
                    n_max += 1;
                }
            }
        }

        let v_factor = (2e9_f64 / (global_max - global_min) as f64) as f32;
        let mut max_points = Vec::with_capacity(n_max);
        for y in 0..h {
            for x in 0..w {
                let p = x + y * w;
                if types[p] == MAXIMUM {
                    let f_value = true_heights[p];
                    let i_value = ((f_value - global_min) * v_factor) as i64;
                    max_points.push(((i_value as u64) << 32) | p as u64);
                }
            }
        }
        max_points.sort_unstable();
        max_points
    }

    /// Flood-merge maxima within `tolerance`, marking surviving maxima's plateaus
    /// as `MAX_AREA`. Direct port of ImageJ's `analyzeAndMarkMaxima`.
    fn analyze_and_mark_maxima(
        &self,
        ip: &[f32],
        true_heights: &[f32],
        types: &mut [u8],
        max_points: &[u64],
        _global_min: f32,
        tolerance: f32,
        max_sorting_error: f32,
    ) {
        let (w, h) = (self.width, self.height);
        let n_max = max_points.len();
        let mut p_list = vec![0usize; w * h];

        for i_max in (0..n_max).rev() {
            let mut offset0 = (max_points[i_max] & 0xffff_ffff) as usize;
            if types[offset0] & PROCESSED != 0 {
                continue; // reached from a higher maximum already
            }
            let mut x0 = (offset0 % w) as i32;
            let mut y0 = (offset0 / w) as i32;
            let mut v0 = true_heights[offset0];

            loop {
                p_list[0] = offset0;
                types[offset0] |= EQUAL | LISTED;
                let mut list_len = 1usize;
                let mut list_i = 0usize;
                let mut sorting_error = false;
                let mut max_possible = true;
                let mut x_equal = x0 as f64;
                let mut y_equal = y0 as f64;
                let mut n_equal = 1i32;

                loop {
                    let offset = p_list[list_i];
                    let x = (offset % w) as i32;
                    let y = (offset / w) as i32;
                    let is_inner = y != 0 && y != h as i32 - 1 && x != 0 && x != w as i32 - 1;
                    for d in 0..8 {
                        let offset2 = (offset as i32 + self.dir_offset[d]) as usize;
                        if (is_inner || self.is_within(x, y, d)) && types[offset2] & LISTED == 0 {
                            if ip[offset2] <= 0.0 {
                                continue; // background (EDM)
                            }
                            if types[offset2] & PROCESSED != 0 {
                                max_possible = false; // reached a previously processed point
                                break;
                            }
                            let x2 = x + DIR_X_OFFSET[d];
                            let y2 = y + DIR_Y_OFFSET[d];
                            let v2 = true_heights[offset2];
                            if v2 > v0 + max_sorting_error {
                                max_possible = false; // reached a higher point
                                break;
                            } else if v2 >= v0 - tolerance {
                                if v2 > v0 {
                                    // this point should have been handled earlier
                                    sorting_error = true;
                                    offset0 = offset2;
                                    v0 = v2;
                                    x0 = x2;
                                    y0 = y2;
                                }
                                p_list[list_len] = offset2;
                                list_len += 1;
                                types[offset2] |= LISTED;
                                if v2 == v0 {
                                    types[offset2] |= EQUAL;
                                    x_equal += x2 as f64;
                                    y_equal += y2 as f64;
                                    n_equal += 1;
                                }
                            }
                        }
                    }
                    list_i += 1;
                    if list_i >= list_len {
                        break;
                    }
                }

                if sorting_error {
                    // a higher maximum was reached: reset and retry from there
                    for &off in p_list.iter().take(list_len) {
                        types[off] = 0;
                    }
                } else {
                    let reset_mask = !(if max_possible { LISTED } else { LISTED | EQUAL });
                    x_equal /= n_equal as f64;
                    y_equal /= n_equal as f64;
                    let mut min_dist2 = 1e20_f64;
                    let mut nearest_i = 0usize;
                    for list_i in 0..list_len {
                        let offset = p_list[list_i];
                        let x = (offset % w) as i32;
                        let y = (offset / w) as i32;
                        types[offset] &= reset_mask;
                        types[offset] |= PROCESSED;
                        if max_possible {
                            types[offset] |= MAX_AREA;
                            if types[offset] & EQUAL != 0 {
                                let dx = x_equal - x as f64;
                                let dy = y_equal - y as f64;
                                let dist2 = dx * dx + dy * dy;
                                if dist2 < min_dist2 {
                                    min_dist2 = dist2;
                                    nearest_i = list_i;
                                }
                            }
                        }
                    }
                    if max_possible {
                        types[p_list[nearest_i]] |= MAX_POINT;
                    }
                }

                if !sorting_error {
                    break;
                }
            }
        }

        if n_max == 0 {
            for t in types.iter_mut() {
                *t = PROCESSED | MAX_AREA;
            }
        }
    }

    /// Precomputes [`Self::true_edm_height`] for every pixel. The callers below
    /// each probe this for a pixel and several of its neighbors, so memoizing it
    /// up front avoids recomputing the same border checks and ridge tests many
    /// times over for the same coordinates.
    fn compute_true_heights(&self, ip: &[f32]) -> Vec<f32> {
        let (w, h) = (self.width, self.height);
        let mut out = vec![0.0f32; w * h];
        for y in 0..h {
            for x in 0..w {
                out[x + y * w] = self.true_edm_height(x as i32, y as i32, ip);
            }
        }
        out
    }

    /// Sub-pixel ridge/maximum height of an EDM, port of `trueEdmHeight`.
    fn true_edm_height(&self, x: i32, y: i32, ip: &[f32]) -> f32 {
        let (w, h) = (self.width as i32, self.height as i32);
        let xmax = w - 1;
        let ymax = h - 1;
        let offset = (x + y * w) as usize;
        let v = ip[offset];
        if x == 0 || y == 0 || x == xmax || y == ymax || v == 0.0 {
            v
        } else {
            let mut true_h = v + 0.5 * SQRT2; // upper bound on the true height
            let mut ridge_or_max = false;
            for d in 0..4 {
                let d2 = (d + 4) % 8;
                let v1 = ip[(offset as i32 + self.dir_offset[d]) as usize];
                let v2 = ip[(offset as i32 + self.dir_offset[d2]) as usize];
                let mut hh = if v >= v1 && v >= v2 {
                    ridge_or_max = true;
                    (v1 + v2) / 2.0
                } else {
                    v1.min(v2)
                };
                hh += if d % 2 == 0 { 1.0 } else { SQRT2 };
                if true_h > hh {
                    true_h = hh;
                }
            }
            if !ridge_or_max {
                true_h = v;
            }
            true_h
        }
    }

    /// Rasterize the EDM to an 8-bit landscape: `MAX_AREA` -> 255, background
    /// (EDM < 0.5) -> 0, everything else scaled into 1..=254. Port of `make8bit`.
    fn make8bit(&self, ip: &[f32], types: &[u8], global_max: f32) -> Vec<u8> {
        let threshold = 0.5_f64;
        let min_value = 1.0_f64;
        let gmax = global_max as f64;
        let offset = min_value - (gmax - min_value) * (1.0 / 253.0 / 2.0 - 1e-6);
        let mut factor = 253.0 / (gmax - min_value);
        if factor > 1.0 {
            factor = 1.0; // with EDM, no better resolution
        }

        let n = self.width * self.height;
        let mut out = vec![0u8; n];
        for i in 0..n {
            let raw = ip[i] as f64;
            if raw < threshold {
                out[i] = 0;
            } else if types[i] & MAX_AREA != 0 {
                out[i] = 255;
            } else {
                let v = 1 + ((raw - offset) * factor).round() as i64;
                out[i] = if v < 1 {
                    1
                } else if v <= 254 {
                    (v & 255) as u8
                } else {
                    254
                };
            }
        }
        out
    }

    /// Flatten eliminated (sub-tolerance) maxima down to their saddle level so the
    /// watershed flood has only "true" maxima to start from. Port of `cleanupMaxima`.
    fn cleanup_maxima(&self, out: &mut [u8], types: &mut [u8], max_points: &[u64]) {
        let (w, h) = (self.width, self.height);
        let n_max = max_points.len();
        let mut p_list = vec![0usize; w * h];

        for i_max in (0..n_max).rev() {
            let offset0 = (max_points[i_max] & 0xffff_ffff) as usize;
            if types[offset0] & (MAX_AREA | ELIMINATED) != 0 {
                continue;
            }
            let level = out[offset0] as i32;
            let mut lo_level = level + 1;
            p_list[0] = offset0;
            types[offset0] |= LISTED;
            let mut list_len = 1usize;
            let mut last_len = 1usize;
            let mut saddle_found = false;
            while !saddle_found && lo_level > 0 {
                lo_level -= 1;
                last_len = list_len;
                let mut list_i = 0usize;
                loop {
                    let offset = p_list[list_i];
                    let x = (offset % w) as i32;
                    let y = (offset / w) as i32;
                    let is_inner = y != 0 && y != h as i32 - 1 && x != 0 && x != w as i32 - 1;
                    for d in 0..8 {
                        let offset2 = (offset as i32 + self.dir_offset[d]) as usize;
                        if (is_inner || self.is_within(x, y, d)) && types[offset2] & LISTED == 0 {
                            if types[offset2] & MAX_AREA != 0
                                || (types[offset2] & ELIMINATED != 0
                                    && out[offset2] as i32 >= lo_level)
                            {
                                saddle_found = true; // reached (or touched) a true maximum
                                break;
                            } else if out[offset2] as i32 >= lo_level
                                && types[offset2] & ELIMINATED == 0
                            {
                                p_list[list_len] = offset2;
                                list_len += 1;
                                types[offset2] |= LISTED;
                            }
                        }
                    }
                    if saddle_found {
                        break;
                    }
                    list_i += 1;
                    if list_i >= list_len {
                        break;
                    }
                }
            }
            for &off in p_list.iter().take(list_len) {
                types[off] &= !LISTED;
            }
            for &off in p_list.iter().take(last_len) {
                out[off] = lo_level as u8;
                types[off] |= ELIMINATED;
            }
        }
    }

    /// Level-by-level constrained dilation that grows the 255-regions downhill
    /// without merging, drawing watershed lines. Port of `watershedSegment`.
    fn watershed_segment(&self, ip: &mut [u8]) {
        let (w, h) = (self.width, self.height);

        let mut histogram = [0i32; 256];
        for &v in ip.iter() {
            histogram[v as usize] += 1;
        }

        let array_size = (w * h) as i64 - histogram[0] as i64 - histogram[255] as i64;
        if array_size <= 0 {
            return;
        }

        let mut coordinates = vec![0i64; array_size as usize];
        let mut highest_value = 0usize;
        let mut max_bin_size = 0i32;
        let mut offset = 0usize;
        let mut level_start = [0usize; 256];
        for v in 1..255usize {
            level_start[v] = offset;
            offset += histogram[v] as usize;
            if histogram[v] > 0 {
                highest_value = v;
            }
            if histogram[v] > max_bin_size {
                max_bin_size = histogram[v];
            }
        }

        let mut level_offset = vec![0usize; highest_value + 1];
        for y in 0..h {
            for x in 0..w {
                let i = x + y * w;
                let v = ip[i] as usize;
                if v > 0 && v < 255 {
                    let off = level_start[v] + level_offset[v];
                    coordinates[off] = x as i64 | (y as i64) << self.int_encode_shift;
                    level_offset[v] += 1;
                }
            }
        }

        // One processLevel pass changes at most all points of a single level, so
        // `max_bin_size` is a safe upper bound (ImageJ caps lower as an
        // optimization; we keep the full bound to never index out of range).
        let mut set_point_list = vec![0usize; max_bin_size.max(1) as usize];

        let table = make_fate_table();
        let direction_sequence: [usize; 8] = [7, 3, 1, 5, 0, 4, 2, 6]; // diagonals first

        // histogram is mutated below (task lists grow), so work on a local copy.
        let mut hist = histogram;
        let mut level = highest_value as i32;
        while level >= 1 {
            let lv = level as usize;
            let mut remaining = hist[lv];
            let mut idle = 0;
            while remaining > 0 && idle < 8 {
                let mut d_index = 0usize;
                loop {
                    let n = self.process_level(
                        direction_sequence[d_index % 8],
                        ip,
                        &table,
                        level_start[lv],
                        remaining,
                        &mut coordinates,
                        &mut set_point_list,
                    );
                    remaining -= n;
                    if n > 0 {
                        idle = 0;
                    }
                    d_index += 1;
                    let done = !(remaining > 0 && idle < 8);
                    idle += 1;
                    if done {
                        break;
                    }
                }
            }

            if remaining > 0 && level > 1 {
                let mut next_level = level;
                loop {
                    next_level -= 1;
                    if !(next_level > 1 && hist[next_level as usize] == 0) {
                        break;
                    }
                }
                if next_level > 0 {
                    let nl = next_level as usize;
                    let mut new_next_level_end = level_start[nl] + hist[nl] as usize;
                    let mut p = level_start[lv];
                    for _ in 0..remaining {
                        let xy = coordinates[p];
                        let x = (xy & self.int_encode_x_mask) as i32;
                        let y = ((xy & self.int_encode_y_mask) >> self.int_encode_shift) as i32;
                        let p_offset = (x + y * w as i32) as usize;
                        let mut add_to_next = false;
                        if x == 0 || y == 0 || x == w as i32 - 1 || y == h as i32 - 1 {
                            add_to_next = true; // image border
                        } else {
                            for d in 0..8 {
                                if self.is_within(x, y, d)
                                    && ip[(p_offset as i32 + self.dir_offset[d]) as usize] == 0
                                {
                                    add_to_next = true; // border of background area
                                    break;
                                }
                            }
                        }
                        if add_to_next {
                            coordinates[new_next_level_end] = xy;
                            new_next_level_end += 1;
                        }
                        p += 1;
                    }
                    hist[nl] = (new_next_level_end - level_start[nl]) as i32;
                }
            }
            level -= 1;
        }
    }

    /// Dilate one level by one pixel in direction `pass`. Port of `processLevel`.
    #[allow(clippy::too_many_arguments)]
    fn process_level(
        &self,
        pass: usize,
        ip: &mut [u8],
        fate_table: &[i32; 256],
        level_start: usize,
        level_n_points: i32,
        coordinates: &mut [i64],
        set_point_list: &mut [usize],
    ) -> i32 {
        let (w, h) = (self.width as i32, self.height as i32);
        let xmax = w - 1;
        let ymax = h - 1;
        let mut n_changed = 0i32;
        let mut n_unchanged = 0i32;
        for i in 0..level_n_points {
            let p = level_start + i as usize;
            let xy = coordinates[p];
            let x = (xy & self.int_encode_x_mask) as i32;
            let y = ((xy & self.int_encode_y_mask) >> self.int_encode_shift) as i32;
            let offset = (x + y * w) as usize;
            let mut index = 0usize;
            if y > 0 && ip[offset - self.width] == 255 {
                index ^= 1;
            }
            if x < xmax && y > 0 && ip[offset - self.width + 1] == 255 {
                index ^= 2;
            }
            if x < xmax && ip[offset + 1] == 255 {
                index ^= 4;
            }
            if x < xmax && y < ymax && ip[offset + self.width + 1] == 255 {
                index ^= 8;
            }
            if y < ymax && ip[offset + self.width] == 255 {
                index ^= 16;
            }
            if x > 0 && y < ymax && ip[offset + self.width - 1] == 255 {
                index ^= 32;
            }
            if x > 0 && ip[offset - 1] == 255 {
                index ^= 64;
            }
            if x > 0 && y > 0 && ip[offset - self.width - 1] == 255 {
                index ^= 128;
            }
            let mask = 1 << pass;
            if fate_table[index] & mask == mask {
                set_point_list[n_changed as usize] = offset;
                n_changed += 1;
            } else {
                coordinates[level_start + n_unchanged as usize] = xy;
                n_unchanged += 1;
            }
        }
        for &off in set_point_list.iter().take(n_changed as usize) {
            ip[off] = 255;
        }
        n_changed
    }

    /// After watershed, set everything that is not a particle (value < 255) to 0.
    fn watershed_post_process(&self, ip: &mut [u8]) {
        for v in ip.iter_mut() {
            if *v < 255 {
                *v = 0;
            }
        }
    }

    /// Whether the neighbor of `(x, y)` in `direction` is inside the image,
    /// assuming `(x, y)` itself is inside. Port of `isWithin`.
    fn is_within(&self, x: i32, y: i32, direction: usize) -> bool {
        let xmax = self.width as i32 - 1;
        let ymax = self.height as i32 - 1;
        match direction {
            0 => y > 0,
            1 => x < xmax && y > 0,
            2 => x < xmax,
            3 => x < xmax && y < ymax,
            4 => y < ymax,
            5 => x > 0 && y < ymax,
            6 => x > 0,
            7 => x > 0 && y > 0,
            _ => false,
        }
    }
}

/// Lookup table for the watershed dilation, indexed by the 8-neighbor occupancy
/// bitmask. Port of `makeFateTable`. A pixel may be dilated into on `pass` if bit
/// `1 << pass` is set; neighborhoods spanning more than one region are forbidden.
fn make_fate_table() -> [i32; 256] {
    let mut table = [0i32; 256];
    let mut is_set = [false; 8];
    for item in 0..256usize {
        let mut mask = 1;
        for s in is_set.iter_mut() {
            *s = item & mask == mask;
            mask *= 2;
        }
        // dilate opposite to the direction of existing neighbors
        let mut mask = 1;
        for i in 0..8 {
            if is_set[(i + 4) % 8] {
                table[item] |= mask;
            }
            mask *= 2;
        }
        // re-derive occupancy treating set side-pixels as filling their edges
        let mut is_set2 = [false; 8];
        let mut m = 1;
        for s in is_set2.iter_mut() {
            *s = item & m == m;
            m *= 2;
        }
        let mut i = 0;
        while i < 8 {
            if is_set2[i] {
                is_set2[(i + 1) % 8] = true;
                is_set2[(i + 7) % 8] = true;
            }
            i += 2;
        }
        let mut transitions = 0;
        for i in 0..8 {
            if is_set2[i] != is_set2[(i + 1) % 8] {
                transitions += 1;
            }
        }
        if transitions >= 4 {
            table[item] = 0; // more than one region: dilation forbidden
        }
    }
    table
}
