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
use evanalyzer_app::charts::bar::{self, BarLook, BarSpec, ErrorBar};
use evanalyzer_app::charts::boxplot::{self, BoxSpec, BoxplotLook};
use evanalyzer_app::charts::histogram::{self, HistogramLook, HistogramSpec};
pub(super) use evanalyzer_app::charts::lines::{self, LineSeries, LinesSpec};
use evanalyzer_app::charts::scatter::{self, ScatterLook, ScatterSpec};
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
        error: i32::from(look.error),
        log_x: look.log_x,
        log_y: look.log_y,
    }
}

/// Boxes that aren't a class (an image, a well, ...) take these, in turn.
const PALETTE: [u32; 8] = [
    0x2a6bdf, 0xe46a33, 0x2e9e5b, 0x7b4fc9, 0x14a39a, 0xc2477a, 0xb8860b, 0x5b5bd6,
];

/// The colours a box can be given (the box tooltip's swatches).
pub(super) const PICKABLE: [u32; 12] = [
    0x2a6bdf, 0xe46a33, 0x2e9e5b, 0x7b4fc9, 0x14a39a, 0xc2477a, 0xd9472b, 0xe8a317, 0x5c7cfa,
    0x52525b, 0x97978f, 0x18181b,
];

/// The factor that turns a value named `label` into the unit the results
/// show (nm, nm², nm³ as µm, µm², µm³ - `in_micrometers`).
pub(super) fn unit_factor(label: &str) -> f64 {
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
fn box_specs(boxes: &[BoxplotBox], factor: f64, look: &ChartLook) -> Vec<BoxSpec> {
    boxes
        .iter()
        .enumerate()
        .map(|(i, b)| BoxSpec {
            label: b.label.clone(),
            // A colour picked for the box over the look's.
            color: look
                .box_colors
                .get(&b.label)
                .map_or_else(|| box_color(i, b.color, look.colors), |c| Rgb(*c)),
            min: b.min * factor,
            q1: b.q1 * factor,
            median: b.median * factor,
            q3: b.q3 * factor,
            max: b.max * factor,
            mean: b.mean * factor,
            sd: b.sd * factor,
            objects: b.object_count,
            whisker_low: b.whisker_low * factor,
            whisker_high: b.whisker_high * factor,
            outliers: b.outliers.iter().map(|v| v * factor).collect(),
            // Per image: its images' values, each a dot.
            count: if b.image_count > 0 {
                b.image_count
            } else {
                b.object_count
            },
            points: b.image_values.iter().map(|v| v * factor).collect(),
            count_noun: if b.image_count > 0 {
                " images".into()
            } else {
                String::new()
            },
        })
        .collect()
}

/// A box's click area made off the UI thread: x, y, width, height, its
/// name, its numbers.
type HitRow = (f32, f32, f32, f32, String, Vec<(String, String)>);

/// The numbers table's heads.
const STAT_HEADS: [&str; 8] = ["Box", "n", "Median", "Q1", "Q3", "Mean", "SD", "SEM"];

/// The numbers of a box as drawn: n, median, Q1, Q3, mean, SD, SEM.
fn box_stats(spec: &BoxSpec) -> [f64; 7] {
    let n = spec.count as f64;
    let sem = if n > 0.0 {
        spec.sd / n.sqrt()
    } else {
        f64::NAN
    };
    [n, spec.median, spec.q1, spec.q3, spec.mean, spec.sd, sem]
}

/// The numbers table as tab-separated text: the heads, a row per box -
/// full precision, "." decimals (for a spreadsheet).
fn stats_tsv(specs: &[BoxSpec]) -> String {
    let mut out = STAT_HEADS.join("\t");
    for spec in specs {
        out.push('\n');
        out.push_str(&spec.label.replace(['\t', '\n'], " "));
        for value in box_stats(spec) {
            out.push('\t');
            if value.is_finite() {
                out.push_str(&value.to_string());
            }
        }
    }
    out
}

/// A box's numbers for its tooltip, as drawn (unit, % of control).
fn box_numbers(spec: &BoxSpec, look: &ChartLook) -> Vec<(String, String)> {
    let f = |v: f64| format_number(v);
    let (whiskers, low, high) = if look.min_max {
        ("Min – max", spec.min, spec.max)
    } else {
        ("Whiskers (1.5 IQR)", spec.whisker_low, spec.whisker_high)
    };
    let sd = spec.sd;
    let mut lines = vec![
        ("Median".to_string(), f(spec.median)),
        (
            "Q1 – Q3".to_string(),
            format!("{} – {}", f(spec.q1), f(spec.q3)),
        ),
        (whiskers.to_string(), format!("{} – {}", f(low), f(high))),
        (
            "Mean ± SD".to_string(),
            format!("{} ± {}", f(spec.mean), f(sd)),
        ),
    ];
    if spec.count_noun.is_empty() {
        lines.push(("Objects".to_string(), f(spec.count as f64)));
    } else {
        lines.push(("Images".to_string(), f(spec.count as f64)));
        lines.push(("Objects".to_string(), f(spec.objects as f64)));
    }
    lines
}

/// A drawn chart: its picture, the click areas and numbers of its boxes,
/// where its dots lie.
pub(super) struct Rendered {
    pub svg: String,
    hits: Vec<HitRow>,
    stat_rows: Vec<(u32, Vec<String>)>,
    tsv: String,
    marks: Vec<evanalyzer_app::charts::Hit>,
}

impl Rendered {
    /// Only a picture.
    fn picture(svg: String) -> Self {
        Self {
            svg,
            hits: Vec::new(),
            stat_rows: Vec::new(),
            tsv: String::new(),
            marks: Vec::new(),
        }
    }
}

/// The data of the chart shown - what `draw_chart` draws from (again at a
/// new size or look).
#[derive(Default)]
pub(super) enum ChartData {
    #[default]
    None,
    Boxplot(Vec<BoxplotBox>),
    /// The boxplot's groups as bars.
    Bars(Vec<BoxplotBox>),
    Histogram(HistogramResult),
    Scatter(ScatterResult),
    /// Over time, a profile: ready in shown units.
    Lines(LinesSpec),
}

impl ResultsStateController {
    /// The boxplot's boxes came: kept and drawn.
    pub(super) fn set_boxplot_boxes(&self, boxes: Vec<BoxplotBox>) {
        *self.chart_data.lock().expect("Poisoned") = ChartData::Boxplot(boxes);
        self.draw_chart();
    }

    /// Over time's or a profile's lines came: kept and drawn.
    pub(super) fn set_lines(&self, spec: LinesSpec) {
        *self.chart_data.lock().expect("Poisoned") = ChartData::Lines(spec);
        self.draw_chart();
    }

    /// The scatter's points came: kept and drawn.
    pub(super) fn set_scatter(&self, scatter: ScatterResult) {
        *self.chart_data.lock().expect("Poisoned") = ChartData::Scatter(scatter);
        self.draw_chart();
    }

    /// The bars' groups came (the boxplot's data): kept and drawn.
    pub(super) fn set_bars(&self, boxes: Vec<BoxplotBox>) {
        *self.chart_data.lock().expect("Poisoned") = ChartData::Bars(boxes);
        self.draw_chart();
    }

    /// The histogram's bins came: kept and drawn.
    pub(super) fn set_histogram(&self, histogram: HistogramResult) {
        *self.chart_data.lock().expect("Poisoned") = ChartData::Histogram(histogram);
        self.draw_chart();
    }

    /// The chart shown, drawn at the area's size and scheme, its marks
    /// pushed.
    pub(super) fn draw_chart(&self) {
        let canvas = *self.chart_canvas.lock().expect("Poisoned");
        if canvas.width < 1.0 || canvas.height < 1.0 {
            return;
        }
        let theme = if canvas.dark {
            Theme::dark()
        } else {
            Theme::light()
        };
        let Some(rendered) = self.render_chart(canvas.width, canvas.height, &theme) else {
            return;
        };
        *self.chart_marks.lock().expect("Poisoned") = rendered.marks;
        self.push_picture(
            rendered.svg,
            rendered.hits,
            rendered.stat_rows,
            rendered.tsv,
        );
    }

    /// The chart shown, drawn `width` × `height` in `theme` (the screen's,
    /// or an export's).
    pub(super) fn render_chart(&self, width: f64, height: f64, theme: &Theme) -> Option<Rendered> {
        let canvas = ChartCanvas {
            width,
            height,
            dark: false,
        };
        let data = self.chart_data.lock().expect("Poisoned");
        Some(match &*data {
            ChartData::None => return None,
            ChartData::Boxplot(boxes) => self.draw_boxplot(boxes, canvas, theme),
            ChartData::Bars(boxes) => self.draw_bars(boxes, canvas, theme),
            ChartData::Histogram(histogram) => self.draw_histogram(histogram, canvas, theme),
            ChartData::Scatter(scatter) => self.draw_scatter(scatter, canvas, theme),
            ChartData::Lines(spec) => {
                let grid = self.chart_look.lock().expect("Poisoned").grid;
                let drawing = lines::draw(spec, grid, Rgb(PALETTE[0]), theme, width, height);
                Rendered::picture(drawing.svg)
            }
        })
    }

    /// The scatter: both values (µm), a dot per object - where they landed
    /// kept for clicks.
    fn draw_scatter(
        &self,
        scatter: &ScatterResult,
        canvas: ChartCanvas,
        theme: &Theme,
    ) -> Rendered {
        let look = self.chart_look.lock().expect("Poisoned").clone();
        let (x_label, y_label, class_color) = {
            let classes = self.classes.lock().expect("Poisoned");
            let chart = self.chart_filter.lock().expect("Poisoned");
            let class_color = chart
                .object_class
                .and_then(|c| classes.iter().find(|k| k.global() == c))
                .map_or(0, |k| k.color & 0x00ff_ffff);
            (
                chart.column.display_label(&classes),
                chart.y_column.display_label(&classes),
                class_color,
            )
        };
        let (fx, fy) = (unit_factor(&x_label), unit_factor(&y_label));
        let spec = ScatterSpec {
            points: scatter
                .points
                .iter()
                .map(|p| (p.x * fx, p.y * fy))
                .collect(),
            color: Rgb(if class_color == 0 {
                PALETTE[0]
            } else {
                class_color
            }),
            total: scatter.total_object_count,
        };
        let drawing = scatter::draw(
            &spec,
            &ScatterLook {
                x_title: super::coloc::picker_label(&x_label),
                y_title: super::coloc::picker_label(&y_label),
                log_x: look.log_x,
                log_y: look.log_y,
                grid: look.grid,
            },
            theme,
            canvas.width,
            canvas.height,
        );
        Rendered {
            marks: drawing.hits,
            ..Rendered::picture(drawing.svg)
        }
    }

    /// A click in the scatter (px): the nearest dot within 8 px - its
    /// object on its image.
    fn scatter_clicked(&self, x: f64, y: f64) {
        let Some(index) = scatter::nearest(&self.chart_marks.lock().expect("Poisoned"), x, y, 8.0)
        else {
            return;
        };
        let target = match &*self.chart_data.lock().expect("Poisoned") {
            ChartData::Scatter(scatter) => scatter.points.get(index).map(|point| {
                (
                    scatter
                        .images
                        .get(point.image as usize)
                        .cloned()
                        .unwrap_or_default(),
                    point.bbox_px,
                )
            }),
            _ => None,
        };
        let Some((rel_path, bbox_px)) = target else {
            return;
        };
        self.image_list_controller.open_image_and_highlight_object(
            &std::path::PathBuf::from(&rel_path),
            bbox_px,
            false,
        );
        let name = std::path::Path::new(&rel_path)
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or(rel_path);
        self.peek(-1, format!("Point in {name}"), false);
    }

    /// The histogram: the value's axis (µm), the counts, linear or log.
    fn draw_histogram(
        &self,
        histogram: &HistogramResult,
        canvas: ChartCanvas,
        theme: &Theme,
    ) -> Rendered {
        let look = self.chart_look.lock().expect("Poisoned").clone();
        let (label, x_title, log_counts, class_color) = {
            let classes = self.classes.lock().expect("Poisoned");
            let chart = self.chart_filter.lock().expect("Poisoned");
            let class_color = chart
                .object_class
                .and_then(|c| classes.iter().find(|k| k.global() == c))
                .map_or(0, |k| k.color & 0x00ff_ffff);
            (
                chart.column.display_label(&classes),
                super::coloc::picker_label(&chart.column.display_label(&classes)),
                chart.log_counts,
                class_color,
            )
        };
        let factor = unit_factor(&label);
        let spec = HistogramSpec {
            edges: histogram.bin_edges.iter().map(|e| e * factor).collect(),
            counts: histogram.counts.clone(),
            color: Rgb(if class_color == 0 {
                PALETTE[0]
            } else {
                class_color
            }),
            below: histogram.below,
            above: histogram.above,
            data_min: histogram.data_min * factor,
            data_max: histogram.data_max * factor,
        };
        let drawing = histogram::draw(
            &spec,
            &HistogramLook {
                x_title,
                y_title: "Objects".into(),
                log_counts,
                grid: look.grid,
            },
            theme,
            canvas.width,
            canvas.height,
        );
        Rendered::picture(drawing.svg)
    }

    /// The bars: a group's mean and error bar, its dots - the boxplot's
    /// tooltip and numbers.
    fn draw_bars(&self, boxes: &[BoxplotBox], canvas: ChartCanvas, theme: &Theme) -> Rendered {
        let look = self.chart_look.lock().expect("Poisoned").clone();
        let (label, mut title) = {
            let classes = self.classes.lock().expect("Poisoned");
            let chart = self.chart_filter.lock().expect("Poisoned");
            (
                chart.column.display_label(&classes),
                super::column_panel::chart_value_label(&chart.column, chart.per_image, &classes),
            )
        };
        let mut factor = unit_factor(&label);
        // In % of the control: the first bar's mean is 100.
        if look.relative
            && let Some(first) = boxes.first().filter(|b| b.mean * factor != 0.0)
        {
            title = format!("% of control ({})", first.label);
            factor = 100.0 / first.mean;
        }
        let error = match look.error {
            1 => ErrorBar::Sem,
            2 => ErrorBar::Ci95,
            _ => ErrorBar::Sd,
        };
        let specs = box_specs(boxes, factor, &look);
        let bars: Vec<BarSpec> = specs
            .iter()
            .map(|spec| BarSpec {
                label: spec.label.clone(),
                color: spec.color,
                mean: spec.mean,
                sd: spec.sd,
                count: spec.count,
                points: spec.points.clone(),
                count_noun: spec.count_noun.clone(),
            })
            .collect();
        let drawing = bar::draw(
            &bars,
            &BarLook {
                y_title: format!("{title} · mean ± {}", error.name()),
                error,
                grid: look.grid,
                dots: look.dots,
                show_n: look.show_n,
            },
            theme,
            canvas.width,
            canvas.height,
        );
        let hits: Vec<HitRow> = drawing
            .hits
            .iter()
            .map(|hit| {
                let spec = &specs[hit.index];
                let mut numbers = box_numbers(spec, &look);
                let half = error.half(spec.sd, spec.count);
                numbers.insert(
                    0,
                    (
                        format!("Mean ± {}", error.name()),
                        format!("{} ± {}", format_number(spec.mean), format_number(half)),
                    ),
                );
                (
                    hit.x as f32,
                    hit.y as f32,
                    hit.width as f32,
                    hit.height as f32,
                    spec.label.clone(),
                    numbers,
                )
            })
            .collect();
        let stat_rows: Vec<(u32, Vec<String>)> = specs
            .iter()
            .map(|spec| {
                let mut cells = vec![spec.label.clone()];
                cells.extend(box_stats(spec).iter().map(|v| format_number(*v)));
                (spec.color.0, cells)
            })
            .collect();
        Rendered {
            svg: drawing.svg,
            hits,
            stat_rows,
            tsv: stats_tsv(&specs),
            marks: Vec::new(),
        }
    }

    /// The boxplot: a box per group, the look's options.
    fn draw_boxplot(&self, boxes: &[BoxplotBox], canvas: ChartCanvas, theme: &Theme) -> Rendered {
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
        if look.relative
            && let Some(first) = boxes.first().filter(|b| b.median * factor != 0.0)
        {
            title = format!("% of control ({})", first.label);
            factor = 100.0 / first.median;
        }
        let specs = box_specs(boxes, factor, &look);
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
            theme,
            canvas.width,
            canvas.height,
        );
        let hits: Vec<HitRow> = drawing
            .hits
            .iter()
            .map(|hit| {
                let spec = &specs[hit.index];
                (
                    hit.x as f32,
                    hit.y as f32,
                    hit.width as f32,
                    hit.height as f32,
                    spec.label.clone(),
                    box_numbers(spec, &look),
                )
            })
            .collect();
        let svg = drawing.svg;
        let stat_rows: Vec<(u32, Vec<String>)> = specs
            .iter()
            .map(|spec| {
                let mut cells = vec![spec.label.clone()];
                cells.extend(box_stats(spec).iter().map(|v| format_number(*v)));
                (spec.color.0, cells)
            })
            .collect();
        let tsv = stats_tsv(&specs);
        Rendered {
            svg,
            hits,
            stat_rows,
            tsv,
            marks: Vec::new(),
        }
    }

    /// A drawn chart to the view: the picture, its marks, the numbers.
    fn push_picture(
        &self,
        svg: String,
        hits: Vec<HitRow>,
        stat_rows: Vec<(u32, Vec<String>)>,
        tsv: String,
    ) {
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
                .map(|(x, y, width, height, label, lines)| ChartHit {
                    x,
                    y,
                    width,
                    height,
                    label: label.into(),
                    lines: ModelRc::from(Rc::new(VecModel::from(
                        lines
                            .into_iter()
                            .map(|(label, value)| crate::ChartCard {
                                label: label.into(),
                                value: value.into(),
                            })
                            .collect::<Vec<_>>(),
                    ))),
                })
                .collect();
            state.set_chart_hits(ModelRc::from(Rc::new(VecModel::from(hits))));
            state.set_chart_stats_head(ModelRc::from(Rc::new(VecModel::from(
                STAT_HEADS
                    .iter()
                    .map(|h| slint::SharedString::from(*h))
                    .collect::<Vec<_>>(),
            ))));
            let rows: Vec<crate::ChartStatRow> = stat_rows
                .into_iter()
                .map(|(color, cells)| crate::ChartStatRow {
                    color: bg_color_to_slint(color),
                    cells: ModelRc::from(Rc::new(VecModel::from(
                        cells
                            .into_iter()
                            .map(slint::SharedString::from)
                            .collect::<Vec<_>>(),
                    ))),
                })
                .collect();
            state.set_chart_stats_rows(ModelRc::from(Rc::new(VecModel::from(rows))));
            state.set_chart_stats_tsv(tsv.into());
        })
        .ok();
    }

    pub(super) fn attach_chart_draw_callbacks(self: &Arc<Self>, ui: &AppWindow) {
        let state = ui.global::<ResultsState>();
        state.set_chart_box_palette(ModelRc::from(Rc::new(VecModel::from(
            PICKABLE
                .iter()
                .map(|c| bg_color_to_slint(*c))
                .collect::<Vec<_>>(),
        ))));
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
        state.on_chart_picture_clicked(move |x, y| {
            manager.scatter_clicked(f64::from(x), f64::from(y));
        });
        let manager = self.clone();
        state.on_chart_box_color_picked(move |label, index| {
            {
                let mut look = manager.chart_look.lock().expect("Poisoned");
                match usize::try_from(index).ok().and_then(|i| PICKABLE.get(i)) {
                    Some(color) => {
                        look.box_colors.insert(label.to_string(), *color);
                    }
                    // Back to the look's colour.
                    None => {
                        look.box_colors.remove(label.as_str());
                    }
                }
            }
            manager.remember_view();
            manager.draw_chart();
        });
        let manager = self.clone();
        state.on_chart_box_look_changed(move |look| {
            let box_colors = manager
                .chart_look
                .lock()
                .expect("Poisoned")
                .box_colors
                .clone();
            *manager.chart_look.lock().expect("Poisoned") = ChartLook {
                log: look.log,
                grid: look.grid,
                dots: look.dots,
                mean: look.mean,
                show_n: look.show_n,
                relative: look.relative,
                min_max: look.min_max,
                colors: u8::try_from(look.colors).unwrap_or(0),
                error: u8::try_from(look.error).unwrap_or(0),
                log_x: look.log_x,
                log_y: look.log_y,
                box_colors,
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
    fn a_box_names_its_numbers_per_image_with_images_and_objects() {
        let spec = BoxSpec {
            label: "A1".into(),
            color: Rgb(0),
            min: 1.0,
            q1: 2.0,
            median: 3.0,
            q3: 4.0,
            max: 9.0,
            mean: 3.5,
            sd: 1.25,
            objects: 120,
            whisker_low: 1.0,
            whisker_high: 6.0,
            outliers: vec![9.0],
            count: 16,
            points: Vec::new(),
            count_noun: " images".into(),
        };
        let lines = box_numbers(&spec, &ChartLook::default());
        let get = |label: &str| {
            lines
                .iter()
                .find(|(l, _)| l == label)
                .map(|(_, v)| v.clone())
        };
        assert_eq!(get("Median").as_deref(), Some("3"));
        assert_eq!(get("Whiskers (1.5 IQR)").as_deref(), Some("1 – 6"));
        assert_eq!(get("Images").as_deref(), Some("16"));
        assert_eq!(get("Objects").as_deref(), Some("120"));
        let min_max = ChartLook {
            min_max: true,
            ..ChartLook::default()
        };
        let lines = box_numbers(&spec, &min_max);
        assert!(lines.iter().any(|(l, v)| l == "Min – max" && v == "1 – 9"));
    }

    #[test]
    fn the_numbers_copy_as_tab_separated_text_at_full_precision() {
        let spec = BoxSpec {
            label: "A1".into(),
            color: Rgb(0),
            min: 1.0,
            q1: 2.0,
            median: 3.125,
            q3: 4.0,
            max: 9.0,
            mean: 3.5,
            sd: 2.0,
            objects: 120,
            whisker_low: 1.0,
            whisker_high: 6.0,
            outliers: Vec::new(),
            count: 16,
            points: Vec::new(),
            count_noun: " images".into(),
        };
        let tsv = stats_tsv(&[spec]);
        let lines: Vec<&str> = tsv.lines().collect();
        assert_eq!(lines[0], "Box\tn\tMedian\tQ1\tQ3\tMean\tSD\tSEM");
        assert_eq!(lines[1], "A1\t16\t3.125\t2\t4\t3.5\t2\t0.5");
    }

    #[test]
    fn the_colours_follow_the_look() {
        assert_eq!(box_color(3, 0, 0), Rgb(PALETTE[3]));
        assert_eq!(
            box_color(3, 0x123456, 0),
            Rgb(0x123456),
            "a class keeps its colour"
        );
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
        let specs = box_specs(&boxes, 1e-6, &ChartLook::default());
        let mut picked = ChartLook::default();
        picked.box_colors.insert("a".into(), 0x123456);
        assert_eq!(
            box_specs(&boxes, 1.0, &picked)[0].color,
            Rgb(0x123456),
            "a picked colour wins"
        );
        assert_eq!(specs[0].median, 3.0);
        assert_eq!(specs[0].outliers, [9.0]);
        assert_eq!(
            specs[0].color,
            Rgb(PALETTE[0]),
            "no class colour: the palette's"
        );
    }
}
