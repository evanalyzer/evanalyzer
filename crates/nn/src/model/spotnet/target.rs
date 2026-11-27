//! # target
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-11-27
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.
//!
//! The perfect answer for training: Gaussian hills at the spots, and where the loss counts.

use crate::model::spotnet::train::{Spot, SpotLabels};

/// The perfect answer: a Gaussian hill (height 1) at every spot,
/// the max where hills overlap. `width * height` values, row-major.
pub(crate) fn gaussian_map(width: usize, height: usize, spots: &[Spot], sigma: f32) -> Vec<f32> {
    let mut out: Vec<f32> = vec![0.0; width * height];
    let radius = (3.0 * sigma).ceil() as i64;
    let two_sigma_sq = 2.0 * sigma * sigma;

    for spot in spots {
        let cx = spot.x.floor() as i64;
        let cy = spot.y.floor() as i64;
        let x0 = (cx - radius).max(0) as usize;
        let x1 = (cx + radius).min(width as i64 - 1) as usize;
        let y0 = (cy - radius).max(0) as usize;
        let y1 = (cy + radius).min(height as i64 - 1) as usize;

        for py in y0..=y1 {
            for px in x0..=x1 {
                let dx = (px as f32 + 0.5) - spot.x;
                let dy = (py as f32 + 0.5) - spot.y;
                let value = (-(dx * dx + dy * dy) / two_sigma_sq).exp();
                out[py * width + px] = out[py * width + px].max(value);
            }
        }
    }

    out
}

/// 1.0 inside the training areas, 0.0 outside: where the loss counts.
pub(crate) fn area_mask(labels: &SpotLabels) -> Vec<f32> {
    let out: Vec<f32> = labels
        .area
        .iter()
        .map(|&inside| if inside { 1.0 } else { 0.0 })
        .collect();

    out
}

#[cfg(test)]
mod tests {
    use crate::model::spotnet::{
        target::{area_mask, gaussian_map},
        train::{Spot, SpotLabels},
    };

    #[test]
    fn the_area_mask_is_one_inside_the_training_areas_and_zero_outside() {
        // 4 x 2: a 2 x 2 area on the left, the right half outside.
        let labels = SpotLabels {
            width: 4,
            height: 2,
            spots: vec![Spot { x: 3.5, y: 0.5 }],
            area: vec![true, true, false, false, true, true, false, false],
        };
        let mask = area_mask(&labels);
        assert_eq!(mask, vec![1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0]);
    }

    #[test]
    fn a_spot_is_a_hill_of_height_one_on_its_pixel() {
        let map = gaussian_map(20, 20, &[Spot { x: 5.5, y: 5.5 }], 1.5);
        print_vec(20, 20, &map);
        let at = |x: usize, y: usize| map[y * 20 + x];
        assert_eq!(at(5, 5), 1.0);
        assert!((at(6, 5) - 0.80).abs() < 0.01, "1 px beside: {}", at(6, 5));
        assert!((at(6, 6) - 0.64).abs() < 0.01, "diagonal: {}", at(6, 6));
        assert_eq!(at(15, 15), 0.0);
    }

    #[test]
    fn a_spot_at_the_border_of_a_non_square_image_stays_inside() {
        let (width, height) = (50, 20);
        let map = gaussian_map(
            width,
            height,
            &[Spot { x: 5.5, y: 18.5 }, Spot { x: 0.5, y: 0.5 }],
            1.5,
        );
        assert_eq!(map.len(), width * height);
        assert_eq!(map[18 * width + 5], 1.0);
        assert_eq!(map[0], 1.0);
    }

    #[test]
    fn close_spots_keep_a_dip_between_them() {
        // 4 px apart: the midpoint is exp(-4 / 4.5) ≈ 0.41, not a sum above 1.
        let map = gaussian_map(
            20,
            5,
            &[Spot { x: 6.5, y: 2.5 }, Spot { x: 10.5, y: 2.5 }],
            1.5,
        );
        let row = &map[2 * 20..3 * 20];
        assert_eq!(row[6], 1.0);
        assert_eq!(row[10], 1.0);
        assert!((row[8] - 0.41).abs() < 0.01, "midpoint: {}", row[8]);
        assert!(row.iter().all(|&v| v <= 1.0));
    }

    /// Prints `map` as a grid, one row per line, two decimals per value
    /// (`.` for 0, so the hills stand out). See it with
    /// `cargo test -p evanalyzer_nn --lib gaussian_map -- --nocapture`.
    fn print_vec(width: usize, height: usize, map: &Vec<f32>) {
        for y in 0..height {
            let row: Vec<String> = map[y * width..(y + 1) * width]
                .iter()
                .map(|&v| {
                    if v == 0.0 {
                        "   .".to_string()
                    } else {
                        format!("{v:4.2}")
                    }
                })
                .collect();
            println!("{}", row.join(" "));
        }
    }
}
