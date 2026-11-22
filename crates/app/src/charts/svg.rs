//! # svg
//!
//! A small SVG writer for the charts: the elements they draw, numbers with
//! two decimals, text escaped.
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-11-22
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.

use super::{Rgb, Theme};
use std::fmt::Write;

/// How a text is placed at its point.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Anchor {
    Start,
    Middle,
    End,
}

/// An SVG document being written.
pub(crate) struct Svg {
    out: String,
}

impl Svg {
    /// A `width` × `height` document in `theme`'s font.
    pub fn new(width: f64, height: f64, theme: &Theme) -> Self {
        let mut out = String::with_capacity(16 * 1024);
        let _ = write!(
            out,
            r#"<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}" font-family="{font}" font-size="{size}">"#,
            w = n(width),
            h = n(height),
            font = escape(&theme.font_family),
            size = n(theme.font_size),
        );
        if let Some(background) = theme.background {
            let _ = write!(
                out,
                r#"<rect width="{}" height="{}" fill="{}"/>"#,
                n(width),
                n(height),
                background.hex()
            );
        }
        Self { out }
    }

    pub fn line(&mut self, x1: f64, y1: f64, x2: f64, y2: f64, color: Rgb, width: f64) {
        let _ = write!(
            self.out,
            r#"<line x1="{}" y1="{}" x2="{}" y2="{}" stroke="{}" stroke-width="{}"/>"#,
            n(x1),
            n(y1),
            n(x2),
            n(y2),
            color.hex(),
            n(width)
        );
    }

    /// A rectangle: filled with `fill` at `fill_opacity`, outlined with
    /// `stroke` (none: no outline).
    #[allow(clippy::too_many_arguments)]
    pub fn rect(
        &mut self,
        x: f64,
        y: f64,
        width: f64,
        height: f64,
        fill: Rgb,
        fill_opacity: f64,
        stroke: Option<(Rgb, f64)>,
        radius: f64,
    ) {
        let _ = write!(
            self.out,
            r#"<rect x="{}" y="{}" width="{}" height="{}" rx="{}" fill="{}" fill-opacity="{}""#,
            n(x),
            n(y),
            n(width.max(0.0)),
            n(height.max(0.0)),
            n(radius),
            fill.hex(),
            n(fill_opacity)
        );
        if let Some((color, width)) = stroke {
            let _ = write!(
                self.out,
                r#" stroke="{}" stroke-width="{}""#,
                color.hex(),
                n(width)
            );
        }
        self.out.push_str("/>");
    }

    pub fn circle(&mut self, cx: f64, cy: f64, r: f64, fill: Rgb, opacity: f64) {
        let _ = write!(
            self.out,
            r#"<circle cx="{}" cy="{}" r="{}" fill="{}" fill-opacity="{}"/>"#,
            n(cx),
            n(cy),
            n(r),
            fill.hex(),
            n(opacity)
        );
    }

    /// A filled polygon through `points`.
    pub fn polygon(&mut self, points: &[(f64, f64)], fill: Rgb, opacity: f64) {
        let points: Vec<String> = points
            .iter()
            .map(|(x, y)| format!("{},{}", n(*x), n(*y)))
            .collect();
        let _ = write!(
            self.out,
            r#"<polygon points="{}" fill="{}" fill-opacity="{}"/>"#,
            points.join(" "),
            fill.hex(),
            n(opacity)
        );
    }

    /// `text` with its baseline at `y`.
    #[allow(clippy::too_many_arguments)]
    pub fn text(
        &mut self,
        x: f64,
        y: f64,
        text: &str,
        color: Rgb,
        anchor: Anchor,
        size: Option<f64>,
        bold: bool,
    ) {
        let anchor = match anchor {
            Anchor::Start => "start",
            Anchor::Middle => "middle",
            Anchor::End => "end",
        };
        let _ = write!(
            self.out,
            r#"<text x="{}" y="{}" fill="{}" text-anchor="{anchor}""#,
            n(x),
            n(y),
            color.hex()
        );
        if let Some(size) = size {
            let _ = write!(self.out, r#" font-size="{}""#, n(size));
        }
        if bold {
            self.out.push_str(r#" font-weight="600""#);
        }
        let _ = write!(self.out, ">{}</text>", escape(text));
    }

    /// `text` turned upright (read bottom to top), centred on `x`, `y`.
    pub fn text_vertical(&mut self, x: f64, y: f64, text: &str, color: Rgb) {
        let _ = write!(
            self.out,
            r#"<text x="0" y="0" fill="{}" text-anchor="middle" transform="translate({},{}) rotate(-90)">{}</text>"#,
            color.hex(),
            n(x),
            n(y),
            escape(text)
        );
    }

    pub fn finish(mut self) -> String {
        self.out.push_str("</svg>");
        self.out
    }
}

/// A number with at most two decimals, no trailing zeros.
fn n(value: f64) -> String {
    let rounded = (value * 100.0).round() / 100.0;
    let text = format!("{rounded:.2}");
    let text = text.trim_end_matches('0').trim_end_matches('.');
    if text == "-0" { "0".to_string() } else { text.to_string() }
}

/// `text` safe in an attribute or element.
fn escape(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    for c in text.chars() {
        match c {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '"' => out.push_str("&quot;"),
            '\'' => out.push_str("&#39;"),
            c => out.push(c),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn numbers_are_short_and_text_is_escaped() {
        assert_eq!(n(12.0), "12");
        assert_eq!(n(0.125), "0.13");
        assert_eq!(n(-0.001), "0");
        assert_eq!(escape(r#"a<b & "c""#), "a&lt;b &amp; &quot;c&quot;");
    }
}
