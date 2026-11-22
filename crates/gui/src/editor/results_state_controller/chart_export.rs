//! # chart_export
//!
//! Saving the chart shown (design canvas "Results rework", ChartExport): the
//! chart module draws it again at the size on paper - light colours, white
//! or transparent ground - and it is written as a PNG (at its dpi), an SVG
//! (in cm) or its plotted numbers as CSV, where the file browser points
//! (the backend's file system - the server's in remote mode).
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-11-22
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.

use super::chart_draw::{ChartData, unit_factor};
use super::*;
use crate::file_browser::FileRequest;
use evanalyzer_app::charts::{Rgb, Theme, export};

/// The app's fonts the PNG's text is drawn with.
const FONTS: [&[u8]; 4] = [
    include_bytes!("../../../ui/fonts/IBMPlexSans-Regular.ttf"),
    include_bytes!("../../../ui/fonts/IBMPlexSans-SemiBold.ttf"),
    include_bytes!("../../../ui/fonts/IBMPlexSans-Bold.ttf"),
    include_bytes!("../../../ui/fonts/IBMPlexMono-Regular.ttf"),
];

/// What the save dialog asked for.
#[derive(Debug, Clone, Copy, PartialEq)]
struct ExportRequest {
    format: i32,
    width_cm: f64,
    height_cm: f64,
    dpi: f64,
    transparent: bool,
}

/// A number in a CSV: full precision, "." decimals; none empty.
fn csv_number(value: f64) -> String {
    if value.is_finite() { value.to_string() } else { String::new() }
}

impl ResultsStateController {
    /// The chart shown as `request` asks - the file's bytes.
    fn export_bytes(&self, request: ExportRequest) -> Result<Vec<u8>, String> {
        if request.format == 2 {
            let rows = self.chart_csv().ok_or("No chart to save")?;
            return Ok(export::csv(&rows).into_bytes());
        }
        let mut theme = Theme::light();
        theme.background = (!request.transparent).then_some(Rgb(0xffffff));
        let (width, height) = export::layout_size(request.width_cm, request.height_cm);
        let rendered = self.render_chart(width, height, &theme).ok_or("No chart to save")?;
        if request.format == 1 {
            Ok(export::svg_in_cm(&rendered.svg, request.width_cm, request.height_cm).into_bytes())
        } else {
            export::png(&rendered.svg, request.width_cm, request.height_cm, request.dpi, &FONTS)
        }
    }

    /// The plotted numbers of the chart shown, a row each (heads first),
    /// in the units drawn.
    fn chart_csv(&self) -> Option<Vec<Vec<String>>> {
        let classes = self.classes.lock().expect("Poisoned").clone();
        let chart = self.chart_filter.lock().expect("Poisoned").clone();
        let label = chart.column.display_label(&classes);
        let factor = unit_factor(&label);
        let name = super::coloc::picker_label(&label);
        let data = self.chart_data.lock().expect("Poisoned");
        Some(match &*data {
            ChartData::None => return None,
            ChartData::Boxplot(boxes) | ChartData::Bars(boxes) => {
                let per_image = boxes.iter().any(|b| !b.image_values.is_empty());
                let mut rows = Vec::new();
                if per_image {
                    // A row per image: the dots.
                    rows.push(vec!["Group".to_string(), super::column_panel::chart_value_label(&chart.column, true, &classes)]);
                    for b in boxes {
                        for value in &b.image_values {
                            rows.push(vec![b.label.clone(), csv_number(value * factor)]);
                        }
                    }
                } else {
                    rows.push(
                        ["Group", "n", "Median", "Q1", "Q3", "Mean", "SD", "Min", "Max"]
                            .iter()
                            .map(|h| (*h).to_string())
                            .collect(),
                    );
                    for b in boxes {
                        rows.push(vec![
                            b.label.clone(),
                            b.object_count.to_string(),
                            csv_number(b.median * factor),
                            csv_number(b.q1 * factor),
                            csv_number(b.q3 * factor),
                            csv_number(b.mean * factor),
                            csv_number(b.sd * factor),
                            csv_number(b.min * factor),
                            csv_number(b.max * factor),
                        ]);
                    }
                }
                rows
            }
            ChartData::Histogram(histogram) => {
                let mut rows = vec![vec![format!("{name} from"), format!("{name} to"), "Objects".to_string()]];
                for (i, count) in histogram.counts.iter().enumerate() {
                    rows.push(vec![
                        csv_number(histogram.bin_edges[i] * factor),
                        csv_number(histogram.bin_edges[i + 1] * factor),
                        count.to_string(),
                    ]);
                }
                if histogram.below > 0 {
                    rows.push(vec![csv_number(histogram.data_min * factor), csv_number(histogram.min * factor), histogram.below.to_string()]);
                }
                if histogram.above > 0 {
                    rows.push(vec![csv_number(histogram.max * factor), csv_number(histogram.data_max * factor), histogram.above.to_string()]);
                }
                rows
            }
            ChartData::Scatter(scatter) => {
                let y_label = chart.y_column.display_label(&classes);
                let y_factor = unit_factor(&y_label);
                let mut rows = vec![vec![name, super::coloc::picker_label(&y_label), "Image".to_string()]];
                for point in &scatter.points {
                    rows.push(vec![
                        csv_number(point.x * factor),
                        csv_number(point.y * y_factor),
                        scatter.images.get(point.image as usize).cloned().unwrap_or_default(),
                    ]);
                }
                rows
            }
            ChartData::Lines(spec) => {
                let mut rows = vec![vec![
                    "Line".to_string(),
                    spec.x_title.clone(),
                    spec.y_title.clone(),
                    "Low".to_string(),
                    "High".to_string(),
                ]];
                for series in &spec.series {
                    for (x, c, lo, hi) in &series.points {
                        let opt = |v: &Option<f64>| v.map_or(String::new(), csv_number);
                        rows.push(vec![series.label.clone(), csv_number(*x), opt(c), opt(lo), opt(hi)]);
                    }
                }
                rows
            }
        })
    }

    /// The message on the Charts bar ("" none).
    fn set_export_message(&self, message: String) {
        let ui_weak = self.ui.clone();
        crate::helper::ui_thread::invoke_from_event_loop(move || {
            if let Some(ui) = ui_weak.upgrade() {
                ui.global::<ResultsState>().set_chart_export_message(message.into());
            }
        })
        .ok();
    }

    pub(super) fn attach_chart_export_callbacks(self: &Arc<Self>, ui: &AppWindow) {
        let manager = self.clone();
        let ui_weak = self.ui.clone();
        ui.global::<ResultsState>().on_chart_export_save(move || {
            let Some(ui) = ui_weak.upgrade() else {
                return;
            };
            let state = ui.global::<ResultsState>();
            let request = ExportRequest {
                format: state.get_chart_export_format(),
                width_cm: f64::from(state.get_chart_export_width()),
                height_cm: f64::from(state.get_chart_export_height()),
                dpi: f64::from(state.get_chart_export_dpi()),
                transparent: state.get_chart_export_transparent(),
            };
            let (what, extension) = match request.format {
                1 => ("SVG picture", "svg"),
                2 => ("CSV table", "csv"),
                _ => ("PNG picture", "png"),
            };
            let file_request = FileRequest::save_file("Save the chart")
                .filter(what, &[extension])
                .file_name(&format!("chart.{extension}"));
            let manager = manager.clone();
            manager._app_state.file_browser.clone().open(file_request, move |path| {
                let Some(path) = path else {
                    return;
                };
                let path = if path.extension().is_none() { path.with_extension(extension) } else { path };
                let manager = manager.clone();
                // Off the UI thread: a large PNG takes a moment.
                std::thread::spawn(move || {
                    let name = path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
                    let written = manager
                        .export_bytes(request)
                        .and_then(|bytes| manager._app_state.backend().files().write_file(&path, &bytes).map_err(|e| e.to_string()));
                    manager.set_export_message(match written {
                        Ok(()) => format!("Saved {name}"),
                        Err(error) => format!("Not saved: {error}"),
                    });
                });
            });
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn csv_numbers_keep_their_precision_and_none_is_empty() {
        assert_eq!(csv_number(0.1 + 0.2), "0.30000000000000004");
        assert_eq!(csv_number(85741096.0), "85741096");
        assert_eq!(csv_number(f64::NAN), "");
    }

    #[test]
    fn the_app_fonts_are_bundled() {
        assert!(FONTS.iter().all(|font| font.len() > 10_000));
    }
}
