//! # ruler
//!
//! The Ruler tool (annotate bar, `L`): a line dragged on the image and its
//! length - in pixels and, with the image's pixel size (per axis, so also
//! right for non-square pixels), in nm, µm or mm. Shown while the tool is
//! on and kept nowhere: the next drag, another tool or another image ends
//! it (see `ViewportObjectState.ruler-*`).
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-11-05
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.

use evanalyzer_cfg::settings::images_settings::PixelSizeSettings;

/// The label of a line `dx` x `dy` image pixels long, e.g. "412 px · 20.6 µm".
pub(crate) fn ruler_label(dx: f32, dy: f32, pixel: &PixelSizeSettings) -> String {
    let px = dx.hypot(dy);
    let nm = (dx * pixel.x).hypot(dy * pixel.y);
    let px = match px >= 100.0 {
        true => format!("{px:.0}"),
        false => format!("{px:.1}"),
    };
    format!("{px} px · {}", length_text(nm))
}

/// `nm` in the unit the scale bar would use, three significant digits.
fn length_text(nm: f32) -> String {
    let (value, unit) = if nm >= 1_000_000.0 {
        (nm / 1_000_000.0, "mm")
    } else if nm >= 1_000.0 {
        (nm / 1_000.0, "µm")
    } else {
        (nm, "nm")
    };
    if value >= 100.0 {
        format!("{value:.0} {unit}")
    } else if value >= 10.0 {
        format!("{value:.1} {unit}")
    } else {
        format!("{value:.2} {unit}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pixel(x: f32, y: f32) -> PixelSizeSettings {
        PixelSizeSettings { x, y, z: None }
    }

    #[test]
    fn a_line_is_measured_in_pixels_and_in_the_unit_that_fits() {
        // 3-4-5: 5 px of 100 nm.
        assert_eq!(
            ruler_label(3.0, 4.0, &pixel(100.0, 100.0)),
            "5.0 px · 500 nm"
        );
        assert_eq!(
            ruler_label(-412.0, 0.0, &pixel(50.0, 50.0)),
            "412 px · 20.6 µm"
        );
        assert_eq!(
            ruler_label(0.0, 3000.0, &pixel(500.0, 500.0)),
            "3000 px · 1.50 mm"
        );
        assert_eq!(
            ruler_label(12.0, 0.0, &pixel(0.5, 0.5)),
            "12.0 px · 6.00 nm"
        );
    }

    /// Non-square pixels: each axis in its own size.
    #[test]
    fn each_axis_counts_with_its_own_pixel_size() {
        // 30 px of 100 nm across and 40 px of 75 nm down: 3 µm by 3 µm.
        assert_eq!(
            ruler_label(30.0, 40.0, &pixel(100.0, 75.0)),
            "50.0 px · 4.24 µm"
        );
    }

    #[test]
    fn a_line_of_no_length_is_zero() {
        assert_eq!(
            ruler_label(0.0, 0.0, &pixel(100.0, 100.0)),
            "0.0 px · 0.00 nm"
        );
    }
}
