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

/// What the charts draw now, besides their data.
#[derive(Debug, Clone, PartialEq)]
pub(super) struct ChartLook {
    pub log: bool,
    pub grid: bool,
}

impl Default for ChartLook {
    fn default() -> Self {
        Self { log: false, grid: true }
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

/// The boxes for the chart module, in the shown unit (`factor`).
fn box_specs(boxes: &[BoxplotBox], factor: f64) -> Vec<BoxSpec> {
    boxes
        .iter()
        .enumerate()
        .map(|(i, b)| BoxSpec {
            label: b.label.clone(),
            color: Rgb(if b.color == 0 { PALETTE[i % PALETTE.len()] } else { b.color & 0x00ff_ffff }),
            q1: b.q1 * factor,
            median: b.median * factor,
            q3: b.q3 * factor,
            whisker_low: b.whisker_low * factor,
            whisker_high: b.whisker_high * factor,
            outliers: b.outliers.iter().map(|v| v * factor).collect(),
            count: b.object_count,
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
        let label = {
            let classes = self.classes.lock().expect("Poisoned");
            self.chart_filter.lock().expect("Poisoned").column.display_label(&classes)
        };
        let factor = unit_factor(&label);
        let specs = box_specs(&self.chart_boxes.lock().expect("Poisoned"), factor);
        let theme = if canvas.dark { Theme::dark() } else { Theme::light() };
        let drawing = boxplot::draw(
            &specs,
            &BoxplotLook {
                y_title: super::coloc::picker_label(&label),
                log: look.log,
                grid: look.grid,
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
        state.on_chart_look_changed(move |log, grid| {
            *manager.chart_look.lock().expect("Poisoned") = ChartLook { log, grid };
            manager.draw_chart();
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
        }];
        let specs = box_specs(&boxes, 1e-6);
        assert_eq!(specs[0].median, 3.0);
        assert_eq!(specs[0].outliers, [9.0]);
        assert_eq!(specs[0].color, Rgb(PALETTE[0]), "no class colour: the palette's");
    }
}
