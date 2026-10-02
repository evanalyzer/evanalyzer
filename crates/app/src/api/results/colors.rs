//! Color palettes for heatmap cells and the legend gradient.

use super::ColorSchema;

// Approximate 5-stop reproduction of the matplotlib "viridis" colormap
// (dark purple -> teal -> yellow).
const VIRIDIS_STOPS: [(f32, (u8, u8, u8)); 5] = [
    (0.0, (0x44, 0x01, 0x54)),
    (0.25, (0x3b, 0x52, 0x8b)),
    (0.5, (0x21, 0x90, 0x8d)),
    (0.75, (0x5d, 0xc9, 0x63)),
    (1.0, (0xfd, 0xe7, 0x25)),
];

// Excel's built-in "Red - Yellow - Green" 3-Color Scale conditional format —
// red at the high end, green at the low end (`t=0` is `min`, `t=1` is `max`,
// see `value_to_color`), matching how Excel's own scale reads by default.
const EXCEL_STOPS: [(f32, (u8, u8, u8)); 3] = [
    (0.0, (0x63, 0xbe, 0x7b)),
    (0.5, (0xff, 0xeb, 0x84)),
    (1.0, (0xf8, 0x69, 0x6b)),
];

// Approximate 5-stop reproductions of well-known scientific colormaps —
// same reasoning/precision level as `VIRIDIS_STOPS` above: recognizable as
// the named colormap, not a pixel-exact reproduction of it.

// matplotlib "plasma" (dark blue-purple -> magenta -> orange -> yellow).
const PLASMA_STOPS: [(f32, (u8, u8, u8)); 5] = [
    (0.0, (0x0d, 0x08, 0x87)),
    (0.25, (0x7e, 0x03, 0xa8)),
    (0.5, (0xcc, 0x47, 0x78)),
    (0.75, (0xf8, 0x94, 0x41)),
    (1.0, (0xf0, 0xf9, 0x21)),
];

// matplotlib "inferno" (black -> purple -> red -> orange -> pale yellow).
const INFERNO_STOPS: [(f32, (u8, u8, u8)); 5] = [
    (0.0, (0x00, 0x00, 0x04)),
    (0.25, (0x57, 0x10, 0x6e)),
    (0.5, (0xbc, 0x37, 0x54)),
    (0.75, (0xf9, 0x8c, 0x0a)),
    (1.0, (0xfc, 0xff, 0xa4)),
];

// matplotlib "cividis" (colorblind-friendly dark blue -> gray -> yellow).
const CIVIDIS_STOPS: [(f32, (u8, u8, u8)); 5] = [
    (0.0, (0x00, 0x20, 0x4d)),
    (0.25, (0x41, 0x4d, 0x6b)),
    (0.5, (0x7c, 0x7b, 0x78)),
    (0.75, (0xbc, 0xaf, 0x6f)),
    (1.0, (0xff, 0xea, 0x46)),
];

// matplotlib "coolwarm" (diverging blue -> near-white -> red).
const COOLWARM_STOPS: [(f32, (u8, u8, u8)); 5] = [
    (0.0, (0x3b, 0x4c, 0xc0)),
    (0.25, (0x88, 0xab, 0xfd)),
    (0.5, (0xdd, 0xdd, 0xdd)),
    (0.75, (0xf7, 0xa8, 0x89)),
    (1.0, (0xb4, 0x04, 0x26)),
];

// ColorBrewer "RdBu" diverging (dark red -> near-white -> dark blue).
const RED_BLUE_STOPS: [(f32, (u8, u8, u8)); 5] = [
    (0.0, (0x67, 0x00, 0x1f)),
    (0.25, (0xd6, 0x60, 0x4d)),
    (0.5, (0xf7, 0xf7, 0xf7)),
    (0.75, (0x43, 0x93, 0xc3)),
    (1.0, (0x05, 0x30, 0x61)),
];

// ColorBrewer "YlGnBu" sequential (pale yellow -> green -> blue -> dark navy).
const YLGNBU_STOPS: [(f32, (u8, u8, u8)); 5] = [
    (0.0, (0xff, 0xff, 0xd9)),
    (0.25, (0x7f, 0xcd, 0xbb)),
    (0.5, (0x41, 0xb6, 0xc4)),
    (0.75, (0x22, 0x5e, 0xa8)),
    (1.0, (0x08, 0x1d, 0x58)),
];

// cmocean "haline" (dark indigo -> teal -> green -> pale yellow-green),
// used for ocean salinity.
const HALINE_STOPS: [(f32, (u8, u8, u8)); 5] = [
    (0.0, (0x29, 0x18, 0x6b)),
    (0.25, (0x21, 0x6b, 0x7a)),
    (0.5, (0x2e, 0x9c, 0x82)),
    (0.75, (0x8f, 0xcb, 0x6c)),
    (1.0, (0xf6, 0xed, 0x4c)),
];

// cmocean "algae" (pale yellow-green -> mid green -> near-black dark green),
// used for algae/chlorophyll concentration.
const ALGAE_STOPS: [(f32, (u8, u8, u8)); 5] = [
    (0.0, (0xd9, 0xf0, 0xa3)),
    (0.25, (0x78, 0xc6, 0x79)),
    (0.5, (0x31, 0xa3, 0x54)),
    (0.75, (0x00, 0x68, 0x37)),
    (1.0, (0x00, 0x44, 0x1b)),
];

// cmocean "thermal" (dark navy-black -> purple -> red -> orange -> pale
// yellow), used for ocean temperature.
const THERMAL_STOPS: [(f32, (u8, u8, u8)); 5] = [
    (0.0, (0x04, 0x23, 0x33)),
    (0.25, (0x52, 0x27, 0x6b)),
    (0.5, (0xa8, 0x32, 0x7d)),
    (0.75, (0xe2, 0x72, 0x4f)),
    (1.0, (0xf2, 0xf1, 0x8d)),
];

/// Number of colors `color_scale_gradient` samples a schema at — enough for
/// the GUI's legend bar to look like a smooth gradient when it just splits
/// the stops evenly across a `HorizontalLayout`.
pub const COLOR_SCALE_GRADIENT_STOPS: usize = 12;

/// Samples `value_to_color` at `COLOR_SCALE_GRADIENT_STOPS` evenly spaced
/// points across `[0, 1]`, in `0xRRGGBB`. Lets the GUI's legend bar render
/// the exact gradient a heatmap's cells are colored with, instead of
/// reimplementing the schema's interpolation a second time in Slint.
pub fn color_scale_gradient(schema: &ColorSchema) -> [u32; COLOR_SCALE_GRADIENT_STOPS] {
    let mut stops = [0u32; COLOR_SCALE_GRADIENT_STOPS];
    for (i, stop) in stops.iter_mut().enumerate() {
        let t = i as f64 / (COLOR_SCALE_GRADIENT_STOPS - 1) as f64;
        *stop = value_to_color(t, 0.0, 1.0, schema);
    }
    stops
}

/// Maps `value` (within `[min, max]`) to a `0xRRGGBB` color under the
/// selected `ColorSchema` — the same packing `evanalyzer_cfg`'s `Class.color`
/// and `crates/gui/src/helper/color_generators.rs` already use, so the GUI
/// can unpack a heatmap cell's `bg_color` the same way it already does for
/// class colors.
pub(crate) fn value_to_color(value: f64, min: f64, max: f64, schema: &ColorSchema) -> u32 {
    let t = if max > min {
        ((value - min) / (max - min)).clamp(0.0, 1.0) as f32
    } else {
        0.5
    };
    match schema {
        ColorSchema::Viridis => lerp_palette(&VIRIDIS_STOPS, t),
        ColorSchema::Excel => lerp_palette(&EXCEL_STOPS, t),
        ColorSchema::Plasma => lerp_palette(&PLASMA_STOPS, t),
        ColorSchema::Inferno => lerp_palette(&INFERNO_STOPS, t),
        ColorSchema::Cividis => lerp_palette(&CIVIDIS_STOPS, t),
        ColorSchema::Coolwarm => lerp_palette(&COOLWARM_STOPS, t),
        ColorSchema::RedBlue => lerp_palette(&RED_BLUE_STOPS, t),
        ColorSchema::YlGnBu => lerp_palette(&YLGNBU_STOPS, t),
        ColorSchema::Haline => lerp_palette(&HALINE_STOPS, t),
        ColorSchema::Algae => lerp_palette(&ALGAE_STOPS, t),
        ColorSchema::Thermal => lerp_palette(&THERMAL_STOPS, t),
    }
}

fn lerp_palette(stops: &[(f32, (u8, u8, u8))], t: f32) -> u32 {
    let t = t.clamp(0.0, 1.0);
    for pair in stops.windows(2) {
        let (t0, c0) = pair[0];
        let (t1, c1) = pair[1];
        if t >= t0 && t <= t1 {
            let local_t = (t - t0) / (t1 - t0).max(f32::EPSILON);
            return pack_rgb(
                lerp_u8(c0.0, c1.0, local_t),
                lerp_u8(c0.1, c1.1, local_t),
                lerp_u8(c0.2, c1.2, local_t),
            );
        }
    }
    let (_, last) = *stops.last().expect("palette must have at least one stop");
    pack_rgb(last.0, last.1, last.2)
}

fn lerp_u8(a: u8, b: u8, t: f32) -> u8 {
    (a as f32 + (b as f32 - a as f32) * t).round() as u8
}

fn pack_rgb(r: u8, g: u8, b: u8) -> u32 {
    ((r as u32) << 16) | ((g as u32) << 8) | (b as u32)
}
