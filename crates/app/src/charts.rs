//! # charts
//!
//! The results' charts as pictures (design canvas "Results rework",
//! ChartBox / ChartKinds / ChartExport): a chart is built from its data
//! into an SVG document - the same one the screen shows and an export
//! writes - plus where its marks lie, so a front end can answer clicks
//! and hovers. No GUI here: the CLI or a server can draw them too.
//!
//! - [`Scale`]: linear or log axis, nice ticks.
//! - [`Theme`]: the colours and font - light or dark, as the app.
//! - [`boxplot`]: a box per group.
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-11-22
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.

mod axis;
pub mod boxplot;
mod svg;

pub use axis::Scale;

/// A colour, `0xRRGGBB`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Rgb(pub u32);

impl Rgb {
    /// `#rrggbb`.
    pub fn hex(self) -> String {
        format!("#{:06x}", self.0 & 0x00ff_ffff)
    }
}

/// The colours and the font a chart is drawn with.
#[derive(Debug, Clone, PartialEq)]
pub struct Theme {
    /// Behind the plot (none: transparent).
    pub background: Option<Rgb>,
    pub text: Rgb,
    pub text_dim: Rgb,
    pub axis: Rgb,
    pub grid: Rgb,
    pub font_family: String,
    /// Tick labels and names, px.
    pub font_size: f64,
}

impl Theme {
    /// The app's light look (Ink).
    pub fn light() -> Self {
        Self {
            background: None,
            text: Rgb(0x18181b),
            text_dim: Rgb(0x52525b),
            axis: Rgb(0xa1a1aa),
            grid: Rgb(0xececef),
            font_family: "IBM Plex Sans".into(),
            font_size: 11.0,
        }
    }

    /// The app's dark look (Ink).
    pub fn dark() -> Self {
        Self {
            background: None,
            text: Rgb(0xfafafa),
            text_dim: Rgb(0xa1a1aa),
            axis: Rgb(0x52525b),
            grid: Rgb(0x2a2a2f),
            font_family: "IBM Plex Sans".into(),
            font_size: 11.0,
        }
    }

    /// The width of `text` in this font, estimated (no font is measured:
    /// the same everywhere, also without a display).
    pub(crate) fn text_width(&self, text: &str) -> f64 {
        text.chars().count() as f64 * self.font_size * 0.58
    }
}

/// A mark of a drawn chart a click or hover can name: `index` into the
/// chart's data (e.g. the box), its rectangle in the picture's units.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Hit {
    pub index: usize,
    pub x: f64,
    pub y: f64,
    pub width: f64,
    pub height: f64,
}

/// A drawn chart: the SVG document (`width` × `height` units, as asked)
/// and its marks.
#[derive(Debug, Clone, PartialEq)]
pub struct Drawing {
    pub svg: String,
    pub hits: Vec<Hit>,
}
