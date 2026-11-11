//! # object_card
//!
//! The selected object's card (`ui/editor/object_details.slint`), by
//! section: size and shape (from its mask), intensity per channel (a colour
//! image: one row, its brightness), the values the preview's Measure steps
//! gave it, and its relations. Built for the selected object only, when it
//! is selected - nothing for the other objects.
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-11-11
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.

use crate::{CardFact, CardIntensityRow, CardStepGroup, ObjectCardSlint};
use evanalyzer_app::project::{ProjectExt, ProjectWithRuntime};
use evanalyzer_cfg::core_types::{
    MeasurementUnit, MeasurementValues, PipelineMeasurements, max_gray_value,
};
use evanalyzer_cfg::settings::images_settings::PixelSizeSettings;
use evanalyzer_cfg::settings::object_settings::{IntensitySettings, ObjectMetricSettings};
use slint::{Color, ModelRc, SharedString, VecModel};

/// How a colour (RGB) image's one channel is named.
const RGB_CHANNEL_NAME: &str = "Image (RGB)";

/// What the card's sections show beyond the object itself: how many
/// objects share its label, its relations as text (see
/// `object_list_controller::relations_text`), the colocalized classes, and
/// the image's bit depth (`None`: unknown - no intensities).
pub(crate) struct CardContext<'a> {
    pub label_count: i32,
    pub parents: &'a str,
    pub children: &'a str,
    pub coloc_with: &'a str,
    pub nr_of_bits: Option<u16>,
}

/// The card of `object`.
pub(crate) fn object_card(
    object: &ObjectMetricSettings,
    project: &ProjectWithRuntime,
    context: &CardContext,
) -> ObjectCardSlint {
    let pixel_sizes = project.get_pixel_sizes();
    let rgb = project.get_selected_image_series().is_some_and(|s| s.rgb);
    let (step_groups, step_count) =
        step_values(&object.id, &project.tmp_settings.preview_measurements);
    let relations = relations(object, context);
    let relations_summary = relations
        .iter()
        .filter(|fact| fact.label != "Same label")
        .map(|fact| fact.value.to_string())
        .chain((context.label_count > 1).then(|| format!("{} same label", context.label_count)))
        .collect::<Vec<_>>()
        .join(" · ");
    ObjectCardSlint {
        shape: model(shape(object, &pixel_sizes)),
        intensity_rows: model(intensity_rows(object, project, rgb, context.nr_of_bits)),
        step_groups: model(step_groups),
        step_count: step_count as i32,
        relations: model(relations),
        relations_summary: relations_summary.into(),
    }
}

fn model<T: Clone + 'static>(items: Vec<T>) -> ModelRc<T> {
    ModelRc::new(VecModel::from(items))
}

fn fact(label: &str, value: impl Into<SharedString>, extra: impl Into<SharedString>) -> CardFact {
    CardFact {
        label: label.into(),
        value: value.into(),
        extra: extra.into(),
    }
}

/// The pixel size in µm (x, y) - `None` while unknown (0).
fn micrometers_per_pixel(pixel_sizes: &PixelSizeSettings) -> Option<(f64, f64)> {
    (pixel_sizes.x > 0.0 && pixel_sizes.y > 0.0)
        .then(|| (pixel_sizes.x as f64 / 1000.0, pixel_sizes.y as f64 / 1000.0))
}

/// Size and shape, from the mask: area, perimeter, circularity, the fitted
/// ellipse, centre, box, plane - µm beside px once the pixel size is known.
fn shape(object: &ObjectMetricSettings, pixel_sizes: &PixelSizeSettings) -> Vec<CardFact> {
    let um = micrometers_per_pixel(pixel_sizes);
    let mut facts = vec![fact(
        "Area",
        format!("{} px²", object.area),
        um.map(|(x, y)| format!("{} µm²", number(object.area as f64 * x * y)))
            .unwrap_or_default(),
    )];
    if object.perimeter > 0.0 {
        let p = object.perimeter as f64;
        facts.push(fact(
            "Perimeter",
            format!("{} px", number(p)),
            um.map(|(x, y)| format!("{} µm", number(p * (x + y) / 2.0)))
                .unwrap_or_default(),
        ));
        let circularity = 4.0 * std::f64::consts::PI * object.area as f64 / (p * p);
        facts.push(fact(
            "Circularity",
            format!("{:.2}", circularity.min(1.0)),
            "",
        ));
    }
    if let Some((major, minor, eccentricity)) = ellipse(object) {
        facts.push(fact(
            "Ellipse",
            format!("{} × {} px", number(major), number(minor)),
            format!("ecc. {eccentricity:.2}"),
        ));
    }
    if object.area > 0 {
        let n = object.area as f64;
        facts.push(fact(
            "Centre",
            format!(
                "x {:.0} · y {:.0}",
                object.sum_x as f64 / n,
                object.sum_y as f64 / n
            ),
            "",
        ));
    }
    let w = object.bbox[2].saturating_sub(object.bbox[0]) + 1;
    let h = object.bbox[3].saturating_sub(object.bbox[1]) + 1;
    facts.push(fact("Box", format!("{w} × {h} px"), ""));
    facts.push(fact(
        "Plane",
        format!("z {} · t {}", object.z_stack, object.t_stack),
        if object.touches_edge {
            "touches edge"
        } else {
            ""
        },
    ));
    facts
}

/// The fitted ellipse's axes (full lengths, px) and eccentricity - as the
/// analysis fits it (`Object::compute_ellipse` in core), from the moments.
fn ellipse(object: &ObjectMetricSettings) -> Option<(f64, f64, f64)> {
    let n = object.area as f64;
    if n == 0.0 {
        return None;
    }
    let xc = object.sum_x as f64 / n;
    let yc = object.sum_y as f64 / n;
    let mu20 = object.sum_x2 as f64 / n - xc * xc;
    let mu02 = object.sum_y2 as f64 / n - yc * yc;
    let mu11 = object.sum_xy as f64 / n - xc * yc;
    let common = ((mu20 - mu02).powi(2) + 4.0 * mu11 * mu11).sqrt();
    let major = (8.0 * (mu20 + mu02 + common)).max(0.0).sqrt();
    let minor = (8.0 * (mu20 + mu02 - common)).max(0.0).sqrt();
    if !major.is_finite() || major <= 0.0 {
        return None;
    }
    let eccentricity = (1.0 - minor * minor / (major * major)).max(0.0).sqrt();
    Some((major, minor, eccentricity))
}

/// A number with sensible digits: 3 significant below 100, none above.
fn number(value: f64) -> String {
    if !value.is_finite() {
        "–".into()
    } else if value.abs() >= 100.0 {
        format!("{value:.0}")
    } else if value.abs() >= 10.0 {
        format!("{value:.1}")
    } else {
        format!("{value:.2}")
    }
}

/// The intensity per channel, in the image's gray values: a colour image
/// one row (its brightness, as it is measured), else one per channel.
fn intensity_rows(
    object: &ObjectMetricSettings,
    project: &ProjectWithRuntime,
    rgb: bool,
    nr_of_bits: Option<u16>,
) -> Vec<CardIntensityRow> {
    let Some(bit_max) = nr_of_bits.and_then(max_gray_value) else {
        return Vec::new();
    };
    let colors = project.get_channel_display_colors();
    let names: std::collections::BTreeMap<i32, String> = project
        .get_selected_image_series()
        .map(|series| {
            series
                .channels
                .iter()
                .map(|(idx, c)| {
                    (
                        *idx,
                        evanalyzer_app::utils::channel_display_name(&c.name, *idx),
                    )
                })
                .collect()
        })
        .unwrap_or_default();
    object
        .intensities
        .iter()
        .filter(|(channel, _)| **channel >= 0 && (!rgb || **channel == 0))
        .map(|(channel, values)| {
            let color = colors
                .get(channel)
                .map(|c| Color::from_rgb_f32(c[0], c[1], c[2]))
                .unwrap_or(Color::from_rgb_u8(255, 255, 255));
            CardIntensityRow {
                name: if rgb {
                    RGB_CHANNEL_NAME.into()
                } else {
                    names
                        .get(channel)
                        .cloned()
                        .unwrap_or_else(|| format!("Channel {channel}"))
                        .into()
                },
                color,
                rgb,
                values: model(intensity_values(values, bit_max)),
            }
        })
        .collect()
}

/// mean, sum, min, max, std, median, lower and upper quartile, MAD in gray
/// values ("–" where not measured).
fn intensity_values(values: &IntensitySettings, bit_max: f64) -> Vec<SharedString> {
    let gray = |v: f64| SharedString::from(number(v * bit_max));
    let optional = |v: Option<f64>| v.map(gray).unwrap_or_else(|| "–".into());
    vec![
        gray(values.avg_intensity),
        gray(values.sum_intensity),
        gray(values.min_intensity),
        gray(values.max_intensity),
        optional(values.std_intensity),
        optional(values.median_intensity),
        optional(values.lower_quartile_intensity),
        optional(values.upper_quartile_intensity),
        optional(values.mad_intensity),
    ]
}

/// The values the preview's Measure steps gave the object, by step (in
/// the steps' order), and how many. Profiles (several values per object,
/// e.g. Sholl) are in the results' charts, not here.
fn step_values(
    object: &evanalyzer_cfg::core_types::ObjectId,
    planes: &[PipelineMeasurements],
) -> (Vec<CardStepGroup>, usize) {
    let mut groups: Vec<(String, Vec<CardFact>)> = Vec::new();
    let mut count = 0;
    for measurement in planes.iter().flat_map(|plane| plane.measurements.iter()) {
        if measurement.info.profile {
            continue;
        }
        let MeasurementValues::Objects(values) = &measurement.values else {
            continue;
        };
        let Some(value) = values.iter().find(|v| v.object == *object) else {
            continue;
        };
        count += 1;
        let entry = fact(
            &measurement.info.name,
            with_unit(value.value, measurement.info.unit),
            "",
        );
        match groups
            .iter_mut()
            .find(|(step, _)| *step == measurement.info.step)
        {
            Some((_, facts)) => facts.push(entry),
            None => groups.push((measurement.info.step.clone(), vec![entry])),
        }
    }
    let groups = groups
        .into_iter()
        .map(|(step, facts)| CardStepGroup {
            step: step.into(),
            facts: model(facts),
        })
        .collect();
    (groups, count)
}

/// A measured value with its unit.
fn with_unit(value: f64, unit: MeasurementUnit) -> String {
    if !value.is_finite() {
        return "–".into();
    }
    let v = number(value);
    match unit {
        MeasurementUnit::None | MeasurementUnit::Fraction | MeasurementUnit::Intensity => v,
        MeasurementUnit::Count => format!("{value:.0}"),
        MeasurementUnit::Percent => format!("{v} %"),
        MeasurementUnit::Pixels => format!("{v} px"),
        MeasurementUnit::SquarePixels => format!("{v} px²"),
        MeasurementUnit::Micrometers => format!("{v} µm"),
        MeasurementUnit::SquareMicrometers => format!("{v} µm²"),
        MeasurementUnit::PixelsPerFrame => format!("{v} px/frame"),
        MeasurementUnit::SquarePixelsPerFrame => format!("{v} px²/frame"),
        MeasurementUnit::Degrees => format!("{v}°"),
        MeasurementUnit::DegreesPerFrame => format!("{v}°/frame"),
        MeasurementUnit::PerFrame => format!("{v} /frame"),
    }
}

/// Objects with the same label, parent, children, colocalized classes and
/// track - those the object has.
fn relations(object: &ObjectMetricSettings, context: &CardContext) -> Vec<CardFact> {
    let mut facts = vec![fact(
        "Same label",
        format!("{} objects", context.label_count),
        "",
    )];
    for (label, value) in [
        ("Parent", context.parents),
        ("Children", context.children),
        ("Touches", context.coloc_with),
    ] {
        if !value.is_empty() {
            facts.push(fact(label, value, ""));
        }
    }
    if !object.track.object_ids.is_empty() {
        facts.push(fact("Track", object.track.id.0.to_string(), ""));
    }
    facts
}

#[cfg(test)]
mod tests {
    use super::*;
    use evanalyzer_cfg::core_types::{Measurement, MeasurementInfo, ObjectId, ObjectValue};

    fn square(side: u64) -> ObjectMetricSettings {
        // A side × side square at the origin: its moments.
        let n = side * side;
        let s: u64 = (0..side).sum();
        let s2: u64 = (0..side).map(|v| v * v).sum();
        ObjectMetricSettings {
            area: n as usize,
            perimeter: 4.0 * side as f32,
            bbox: [0, 0, side as u32 - 1, side as u32 - 1],
            sum_x: s * side,
            sum_y: s * side,
            sum_x2: s2 * side,
            sum_y2: s2 * side,
            sum_xy: s * s,
            ..Default::default()
        }
    }

    #[test]
    fn the_shape_has_area_perimeter_ellipse_centre_and_box() {
        let unknown = PixelSizeSettings {
            x: 0.0,
            y: 0.0,
            z: None,
        };
        let facts = shape(&square(10), &unknown);
        let get = |label: &str| facts.iter().find(|f| f.label == label).cloned().unwrap();
        assert_eq!(get("Area").value, "100 px²");
        assert_eq!(
            get("Area").extra,
            "",
            "no µm while the pixel size is unknown"
        );
        assert_eq!(get("Perimeter").value, "40.0 px");
        assert_eq!(get("Centre").value, "x 4 · y 4");
        assert_eq!(get("Box").value, "10 × 10 px");
        assert!(
            get("Ellipse").extra.contains("ecc. 0.00"),
            "a square: round"
        );

        let known = PixelSizeSettings {
            x: 500.0,
            y: 500.0,
            z: None,
        };
        let facts = shape(&square(10), &known);
        assert_eq!(facts[0].extra, "25.0 µm²");
    }

    #[test]
    fn the_steps_values_of_the_object_by_step() {
        let id = ObjectId(7);
        let measurement = |step: &str, name: &str, unit, object: u128, value| Measurement {
            info: MeasurementInfo {
                key: format!("{step}/{name}"),
                name: name.into(),
                step: step.into(),
                unit,
                target: evanalyzer_cfg::core_types::MeasurementTarget::Object,
                range: None,
                aggregatable: true,
                related_objects: false,
                profile: false,
            },
            values: MeasurementValues::Objects(vec![ObjectValue::from((ObjectId(object), value))]),
        };
        let planes = vec![PipelineMeasurements {
            measurements: vec![
                measurement("shape_01", "Feret max", MeasurementUnit::Pixels, 7, 24.8),
                measurement("shape_01", "Solidity", MeasurementUnit::None, 7, 0.94),
                measurement("shape_01", "Solidity", MeasurementUnit::None, 8, 0.5),
                measurement("neighbors_01", "Neighbours", MeasurementUnit::Count, 7, 3.0),
            ],
            ..Default::default()
        }];
        let (groups, count) = step_values(&id, &planes);
        assert_eq!(count, 3);
        assert_eq!(groups.len(), 2);
        assert_eq!(groups[0].step, "shape_01");
        assert_eq!(slint::Model::row_count(&groups[0].facts), 2);
        let first = slint::Model::row_data(&groups[0].facts, 0).unwrap();
        assert_eq!(
            (first.label.as_str(), first.value.as_str()),
            ("Feret max", "24.8 px")
        );
        let neighbours = slint::Model::row_data(&groups[1].facts, 0).unwrap();
        assert_eq!(neighbours.value, "3");
    }
}
