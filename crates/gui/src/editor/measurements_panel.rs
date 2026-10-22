//! The Measurements panel (`ui/editor/measurements_panel.slint`): every
//! measurement the preview made - all pipelines, steps inside modules too -
//! by step. The preview brings them back like its objects
//! (`ProjectTmpSettings::preview_measurements`); synced with them.

use crate::{AppWindow, MeasurementRow, MeasurementsState, UiState};
use evanalyzer_app::project::ProjectWithRuntime;
use evanalyzer_cfg::core_types::{MeasurementRegion, MeasurementValues, PipelineMeasurements};
use evanalyzer_cfg::settings::pipeline_settings::StepContent;
use slint::{ComponentHandle, ModelRc, VecModel};
use std::rc::Rc;

/// Fills the panel from the project: whether it measures, and the
/// preview's values. On the UI thread.
pub(crate) fn sync(ui: &AppWindow, app_state: &UiState) {
    let (has_steps, rows, excluded) = {
        let project = app_state.get_project();
        let planes = &project.tmp_settings.preview_measurements;
        (
            has_measurement_steps(&project),
            rows(planes),
            excluded(planes),
        )
    };
    let state = ui.global::<MeasurementsState>();
    state.set_has_steps(has_steps);
    state.set_rows(ModelRc::from(Rc::new(VecModel::from(rows))));
    state.set_excluded(excluded.into());
}

/// The project has a measurement step - in a pipeline or a module.
pub(crate) fn has_measurement_steps(project: &ProjectWithRuntime) -> bool {
    project
        .pipelines
        .iter()
        .flat_map(|p| p.steps.iter())
        .chain(project.modules.iter().flat_map(|m| m.steps.iter()))
        .any(|step| match &step.content {
            StepContent::Command(command) => command.measurement_name().is_some(),
            StepContent::Module(_) => false,
        })
}

/// The panel's rows: per measurement its value - an image's own, the
/// median of objects' - by step and name.
pub(crate) fn rows(planes: &[PipelineMeasurements]) -> Vec<MeasurementRow> {
    let mut rows: Vec<MeasurementRow> = planes
        .iter()
        .flat_map(|plane| plane.measurements.iter())
        .map(|m| {
            let (value, of) = match &m.values {
                MeasurementValues::Image(values) => {
                    match values.iter().find(|(r, _)| *r == MeasurementRegion::Image) {
                        Some((_, v)) => (format_value(*v), "image".to_string()),
                        None => ("–".to_string(), "no value".to_string()),
                    }
                }
                MeasurementValues::Objects(values) => {
                    let mut numbers: Vec<f64> = values
                        .iter()
                        .map(|(_, v)| *v)
                        .filter(|v| v.is_finite())
                        .collect();
                    match median(&mut numbers) {
                        Some(median) => (
                            format_value(median),
                            format!("median of {} objects", numbers.len()),
                        ),
                        None => ("–".to_string(), "no objects".to_string()),
                    }
                }
            };
            MeasurementRow {
                step: m.info.step.clone().into(),
                name: m.info.name.clone().into(),
                value: value.into(),
                of: of.into(),
            }
        })
        .collect();
    rows.sort_by(|a, b| {
        (a.step.as_str(), a.name.as_str()).cmp(&(b.step.as_str(), b.name.as_str()))
    });
    rows
}

/// Why "Exclude Image" would leave the image out of the results ("" it
/// wouldn't): the preview's planes' reasons.
pub(crate) fn excluded(planes: &[PipelineMeasurements]) -> String {
    planes
        .iter()
        .filter_map(|plane| plane.excluded.as_deref())
        .collect::<Vec<_>>()
        .join("; ")
}

fn format_value(value: f64) -> String {
    if value.is_finite() {
        format!("{value:.3}")
    } else {
        "–".to_string()
    }
}

fn median(values: &mut [f64]) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    values.sort_by(f64::total_cmp);
    let mid = values.len() / 2;
    Some(match values.len() % 2 {
        1 => values[mid],
        _ => (values[mid - 1] + values[mid]) / 2.0,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use evanalyzer_cfg::core_types::{
        Measurement, MeasurementInfo, MeasurementTarget, MeasurementUnit, ObjectId,
    };

    fn measurement(step: &str, name: &str, values: MeasurementValues) -> Measurement {
        Measurement {
            info: MeasurementInfo {
                key: format!("{step}/x"),
                name: name.into(),
                step: step.into(),
                unit: MeasurementUnit::None,
                target: match values {
                    MeasurementValues::Image(_) => MeasurementTarget::Image,
                    MeasurementValues::Objects(_) => MeasurementTarget::Object,
                },
                range: None,
                aggregatable: true,
            },
            values,
        }
    }

    /// An image's value, the median of objects', a missing one - by step.
    #[test]
    fn the_preview_says_why_the_image_would_be_excluded() {
        let plane = |excluded: Option<&str>| PipelineMeasurements {
            excluded: excluded.map(str::to_string),
            ..Default::default()
        };
        assert_eq!(excluded(&[plane(None)]), "");
        assert_eq!(
            excluded(&[plane(Some("Focus score low")), plane(None)]),
            "Focus score low"
        );
    }

    #[test]
    fn a_row_per_measurement_by_step() {
        let plane = PipelineMeasurements {
            measurements: vec![
                measurement(
                    "pearson_02",
                    "Pearson per cell",
                    MeasurementValues::Objects(vec![
                        (ObjectId(1), 0.1),
                        (ObjectId(2), 0.9),
                        (ObjectId(3), 0.4),
                    ]),
                ),
                measurement(
                    "pearson_01",
                    "Pearson",
                    MeasurementValues::Image(vec![(MeasurementRegion::Image, 0.7123)]),
                ),
                measurement("pearson_03", "Pearson", MeasurementValues::Image(vec![])),
            ],
            z_stack: 0,
            t_stack: 0,
            excluded: None,
        };
        let rows: Vec<(String, String, String)> = rows(&[plane])
            .into_iter()
            .map(|r| (r.step.into(), r.value.into(), r.of.into()))
            .collect();
        let row = |step: &str, value: &str, of: &str| (step.into(), value.into(), of.into());
        assert_eq!(
            rows,
            [
                row("pearson_01", "0.712", "image"),
                row("pearson_02", "0.400", "median of 3 objects"),
                row("pearson_03", "–", "no value"),
            ]
        );
    }
}
