//! # chart_draw
//!
//! The charts drawn by the app's chart module (`evanalyzer_app::charts`):
//! the last data kept with the chart area's size and scheme, drawn anew
//! when one of them changes - an SVG picture the view shows, the marks a
//! click names laid over it. The same picture an export writes.
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-11-22
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.

use super::*;
use crate::ChartHit;
use evanalyzer_app::charts::boxplot::{self, BoxSpec, BoxplotLook};
use evanalyzer_app::charts::{Rgb, Theme};

/// The chart area: its size (px) and whether the scheme is dark.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub(super) struct ChartCanvas {
    pub width: f64,
    pub height: f64,
    pub dark: bool,
}

/// What the charts draw now, besides their data: the boxplot's look.
pub(super) type ChartLook = evanalyzer_app::results::view_settings::BoxLook;

/// The look as Slint shows it.
pub(super) fn look_to_slint(look: &ChartLook) -> crate::ChartBoxLook {
    crate::ChartBoxLook {
        log: look.log,
        grid: look.grid,
        dots: look.dots,
        mean: look.mean,
        show_n: look.show_n,
        relative: look.relative,
        min_max: look.min_max,
        colors: i32::from(look.colors),
    }
}

/// Boxes that aren't a class (an image, a well, ...) take these, in turn.
const PALETTE: [u32; 8] = [
    0x2a6bdf, 0xe46a33, 0x2e9e5b, 0x7b4fc9, 0x14a39a, 0xc2477a, 0xb8860b, 0x5b5bd6,
];

/// The factor that turns a value named `label` into the unit the results
/// show (nm, nm², nm³ as µm, µm², µm³ - `in_micrometers`).
fn unit_factor(label: &str) -> f64 {
    if label.contains("[nm²]") {
        1e-6
    } else if label.contains("[nm³]") {
        1e-9
    } else if label.contains("[nm]") {
        1e-3
    } else {
        1.0
    }
}

/// The colour of box `index` (`class_color` 0: not a class) by the look's
/// colours: per box, one colour, or the control (the first) dark grey and
/// the others light grey.
fn box_color(index: usize, class_color: u32, colors: u8) -> Rgb {
    match colors {
        1 => Rgb(PALETTE[0]),
        2 if index == 0 => Rgb(0x52525b),
        2 => Rgb(0xa1a1aa),
        _ if class_color == 0 => Rgb(PALETTE[index % PALETTE.len()]),
        _ => Rgb(class_color & 0x00ff_ffff),
    }
}

/// The boxes for the chart module, values times `factor` (the shown unit,
/// % of control).
fn box_specs(boxes: &[BoxplotBox], factor: f64, colors: u8) -> Vec<BoxSpec> {
    boxes
        .iter()
        .enumerate()
        .map(|(i, b)| BoxSpec {
            label: b.label.clone(),
            color: box_color(i, b.color, colors),
            min: b.min * factor,
            q1: b.q1 * factor,
            median: b.median * factor,
            q3: b.q3 * factor,
            max: b.max * factor,
            mean: b.mean * factor,
            whisker_low: b.whisker_low * factor,
            whisker_high: b.whisker_high * factor,
            outliers: b.outliers.iter().map(|v| v * factor).collect(),
            // Per image: its images' values, each a dot.
            count: if b.image_count > 0 { b.image_count } else { b.object_count },
            points: b.image_values.iter().map(|v| v * factor).collect(),
            count_noun: if b.image_count > 0 { " images".into() } else { String::new() },
        })
        .collect()
}

impl ResultsStateController {
    /// The boxplot's boxes came: kept and drawn.
    pub(super) fn set_boxplot_boxes(&self, boxes: Vec<BoxplotBox>) {
        *self.chart_boxes.lock().expect("Poisoned") = boxes;
        self.draw_chart();
    }

    /// The chart drawn at the area's size and scheme, its marks pushed.
    pub(super) fn draw_chart(&self) {
        let canvas = *self.chart_canvas.lock().expect("Poisoned");
        if canvas.width < 1.0 || canvas.height < 1.0 {
            return;
        }
        let look = self.chart_look.lock().expect("Poisoned").clone();
        let (label, title) = {
            let classes = self.classes.lock().expect("Poisoned");
            let chart = self.chart_filter.lock().expect("Poisoned");
            (
                chart.column.display_label(&classes),
                super::column_panel::chart_value_label(&chart.column, chart.per_image, &classes),
            )
        };
        let mut factor = unit_factor(&label);
        let mut title = title;
        // In % of the control: the first box's median is 100.
        if look.relative {
            let boxes = self.chart_boxes.lock().expect("Poisoned");
            if let Some(first) = boxes.first().filter(|b| b.median * factor != 0.0) {
                title = format!("% of control ({})", first.label);
                factor = 100.0 / first.median;
            }
        }
        let specs = box_specs(&self.chart_boxes.lock().expect("Poisoned"), factor, look.colors);
        let theme = if canvas.dark { Theme::dark() } else { Theme::light() };
        let drawing = boxplot::draw(
            &specs,
            &BoxplotLook {
                y_title: title,
                log: look.log,
                grid: look.grid,
                dots: look.dots,
                mean: look.mean,
                show_n: look.show_n,
                min_max: look.min_max,
            },
            &theme,
            canvas.width,
            canvas.height,
        );
        let hits: Vec<(f32, f32, f32, f32, String)> = drawing
            .hits
            .iter()
            .map(|hit| {
                (
                    hit.x as f32,
                    hit.y as f32,
                    hit.width as f32,
                    hit.height as f32,
                    specs[hit.index].label.clone(),
                )
            })
            .collect();
        let svg = drawing.svg;
        let ui_weak = self.ui.clone();
        crate::helper::ui_thread::invoke_from_event_loop(move || {
            let Some(ui) = ui_weak.upgrade() else {
                return;
            };
            let state = ui.global::<ResultsState>();
            match slint::Image::load_from_svg_data(svg.as_bytes()) {
                Ok(picture) => state.set_chart_picture(picture),
                Err(_) => warn!("The chart's SVG could not be read"),
            }
            let hits: Vec<ChartHit> = hits
                .into_iter()
                .map(|(x, y, width, height, label)| ChartHit {
                    x,
                    y,
                    width,
                    height,
                    label: label.into(),
                })
                .collect();
            state.set_chart_hits(ModelRc::from(Rc::new(VecModel::from(hits))));
        })
        .ok();
    }

    pub(super) fn attach_chart_draw_callbacks(self: &Arc<Self>, ui: &AppWindow) {
        let state = ui.global::<ResultsState>();
        let manager = self.clone();
        state.on_chart_resized(move |width, height, dark| {
            let canvas = ChartCanvas {
                width: f64::from(width),
                height: f64::from(height),
                dark,
            };
            let changed = {
                let mut current = manager.chart_canvas.lock().expect("Poisoned");
                let changed = *current != canvas;
                *current = canvas;
                changed
            };
            if changed {
                manager.draw_chart();
            }
        });
        let manager = self.clone();
        state.on_chart_box_look_changed(move |look| {
            *manager.chart_look.lock().expect("Poisoned") = ChartLook {
                log: look.log,
                grid: look.grid,
                dots: look.dots,
                mean: look.mean,
                show_n: look.show_n,
                relative: look.relative,
                min_max: look.min_max,
                colors: u8::try_from(look.colors).unwrap_or(0),
            };
            manager.remember_view();
            manager.draw_chart();
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_colours_follow_the_look() {
        assert_eq!(box_color(3, 0, 0), Rgb(PALETTE[3]));
        assert_eq!(box_color(3, 0x123456, 0), Rgb(0x123456), "a class keeps its colour");
        assert_eq!(box_color(3, 0x123456, 1), Rgb(PALETTE[0]));
        assert_eq!(box_color(0, 0, 2), Rgb(0x52525b), "the control dark");
        assert_eq!(box_color(1, 0, 2), Rgb(0xa1a1aa));
    }

    #[test]
    fn nanometre_values_are_drawn_in_micrometres() {
        assert_eq!(unit_factor("Area [nm²]"), 1e-6);
        assert_eq!(unit_factor("Perimeter [nm]"), 1e-3);
        assert_eq!(unit_factor("Area [px]"), 1.0);
        let boxes = [BoxplotBox {
            label: "a".into(),
            color: 0,
            min: 1e6,
            q1: 2e6,
            median: 3e6,
            q3: 4e6,
            max: 9e6,
            whisker_low: 1e6,
            whisker_high: 5e6,
            outliers: vec![9e6],
            object_count: 7,
            image_values: Vec::new(),
            image_count: 0,
            mean: 3e6,
            sd: 1e6,
        }];
        let specs = box_specs(&boxes, 1e-6, 0);
        assert_eq!(specs[0].median, 3.0);
        assert_eq!(specs[0].outliers, [9.0]);
        assert_eq!(specs[0].color, Rgb(PALETTE[0]), "no class colour: the palette's");
    }
}
