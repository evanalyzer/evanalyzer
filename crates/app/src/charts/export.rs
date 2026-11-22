//! # export
//!
//! A drawn chart to a file (design canvas "Results rework", ChartExport):
//! the same SVG the screen shows, sized in cm - as an SVG, or rendered by
//! resvg to a PNG at the chosen dpi (written into the file, so a word
//! processor places it at its size). Laid out at 96 units per inch: 17 cm
//! are 643 units, the 11 px text about 8 pt.
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-11-22
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.

use resvg::{tiny_skia, usvg};
use std::sync::Arc;

/// Units per inch a chart is laid out in (CSS pixels).
const UNITS_PER_INCH: f64 = 96.0;

/// The layout size (units) of `width_cm` × `height_cm`.
pub fn layout_size(width_cm: f64, height_cm: f64) -> (f64, f64) {
    let per_cm = UNITS_PER_INCH / 2.54;
    (width_cm * per_cm, height_cm * per_cm)
}

/// `svg` (a chart module drawing) sized `width_cm` × `height_cm` on
/// paper - its own units stay in its view box.
pub fn svg_in_cm(svg: &str, width_cm: f64, height_cm: f64) -> String {
    let Some(start) = svg.find(" width=\"") else {
        return svg.to_string();
    };
    let Some(view_box) = svg[start..].find(" viewBox=") else {
        return svg.to_string();
    };
    format!(
        "{} width=\"{width_cm:.2}cm\" height=\"{height_cm:.2}cm\"{}",
        &svg[..start],
        &svg[start + view_box..]
    )
}

/// `svg` rendered at `dpi` for `width_cm` × `height_cm` - a PNG with its
/// dpi in it. `fonts`: the font files the text is drawn with (the app's).
pub fn png(svg: &str, width_cm: f64, height_cm: f64, dpi: f64, fonts: &[&[u8]]) -> Result<Vec<u8>, String> {
    let mut database = usvg::fontdb::Database::new();
    for font in fonts {
        database.load_font_data(font.to_vec());
    }
    database.set_sans_serif_family("IBM Plex Sans");
    let options = usvg::Options {
        fontdb: Arc::new(database),
        ..usvg::Options::default()
    };
    let tree = usvg::Tree::from_str(svg, &options).map_err(|e| format!("the chart could not be read: {e}"))?;
    let pixels_per_cm = dpi / 2.54;
    let width = (width_cm * pixels_per_cm).round().max(1.0) as u32;
    let height = (height_cm * pixels_per_cm).round().max(1.0) as u32;
    let mut pixmap = tiny_skia::Pixmap::new(width, height).ok_or("the picture is too large")?;
    let size = tree.size();
    let transform = tiny_skia::Transform::from_scale(
        width as f32 / size.width(),
        height as f32 / size.height(),
    );
    resvg::render(&tree, transform, &mut pixmap.as_mut());

    // Straight (not premultiplied) RGBA for the PNG.
    let mut rgba = Vec::with_capacity((width * height * 4) as usize);
    for pixel in pixmap.pixels() {
        let c = pixel.demultiply();
        rgba.extend_from_slice(&[c.red(), c.green(), c.blue(), c.alpha()]);
    }
    let mut out = Vec::new();
    {
        let mut encoder = png::Encoder::new(&mut out, width, height);
        encoder.set_color(png::ColorType::Rgba);
        encoder.set_depth(png::BitDepth::Eight);
        let per_metre = (dpi / 0.0254).round() as u32;
        encoder.set_pixel_dims(Some(png::PixelDimensions {
            xppu: per_metre,
            yppu: per_metre,
            unit: png::Unit::Meter,
        }));
        let mut writer = encoder.write_header().map_err(|e| e.to_string())?;
        writer.write_image_data(&rgba).map_err(|e| e.to_string())?;
    }
    Ok(out)
}

/// `rows` as CSV: comma-separated, a field quoted when it holds a comma,
/// a quote or a line break.
pub fn csv(rows: &[Vec<String>]) -> String {
    let field = |text: &String| {
        if text.contains([',', '"', '\n', '\r']) {
            format!("\"{}\"", text.replace('"', "\"\""))
        } else {
            text.clone()
        }
    };
    rows.iter()
        .map(|row| row.iter().map(field).collect::<Vec<_>>().join(","))
        .collect::<Vec<_>>()
        .join("\n")
        + "\n"
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::charts::boxplot::{BoxSpec, BoxplotLook, draw};
    use crate::charts::{Rgb, Theme};

    fn chart(width: f64, height: f64) -> String {
        let spec = BoxSpec {
            label: "A1".into(),
            color: Rgb(0x2a6bdf),
            min: 1.0,
            q1: 2.0,
            median: 3.0,
            q3: 4.0,
            max: 6.0,
            mean: 3.0,
            sd: 1.0,
            objects: 10,
            whisker_low: 1.0,
            whisker_high: 6.0,
            outliers: Vec::new(),
            count: 10,
            points: Vec::new(),
            count_noun: String::new(),
        };
        let mut theme = Theme::light();
        theme.background = Some(Rgb(0xffffff));
        draw(&[spec], &BoxplotLook::new("Area"), &theme, width, height).svg
    }

    #[test]
    fn a_png_has_the_size_of_its_cm_at_its_dpi_and_says_so() {
        let (w, h) = layout_size(17.0, 10.0);
        let bytes = png(&chart(w, h), 17.0, 10.0, 300.0, &[]).expect("png");
        let decoder = png::Decoder::new(std::io::Cursor::new(bytes));
        let reader = decoder.read_info().expect("a png");
        let info = reader.info();
        // 17 cm at 300 dpi: 2008 px.
        assert_eq!((info.width, info.height), (2008, 1181));
        let dims = info.pixel_dims.expect("its dpi");
        assert_eq!(dims.xppu, 11811, "300 dpi in pixels per metre");
    }

    #[test]
    fn an_svg_is_sized_in_cm_and_keeps_its_units() {
        let (w, h) = layout_size(8.5, 6.0);
        let svg = svg_in_cm(&chart(w, h), 8.5, 6.0);
        assert!(svg.starts_with("<svg") && svg.contains(r#"width="8.50cm" height="6.00cm""#));
        assert!(svg.contains("viewBox=\"0 0 321.26 226.77\""), "{}", &svg[..200]);
    }

    #[test]
    fn csv_quotes_what_needs_it() {
        let rows = vec![vec!["Box".to_string(), "Value".to_string()], vec!["ch1, in-cell".to_string(), "3.5".to_string()]];
        assert_eq!(csv(&rows), "Box,Value\n\"ch1, in-cell\",3.5\n");
    }
}
