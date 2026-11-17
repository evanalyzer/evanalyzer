//! # pipeline_measure
//!
//! The measurements of a plane: the types live in `evanalyzer_cfg` (they
//! travel to the app and over the server connection, like the objects).
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-10-22
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.

pub use evanalyzer_cfg::core_types::{
    Measurement, MeasurementRegion, MeasurementValues, ObjectValue, PipelineMeasurements,
};

use evanalyzer_cfg::core_types::{MeasurementInfo, MeasurementTarget, MeasurementUnit};

/// Builds a measurement step's outputs
/// ([`ImageAlgorithm::measurement_outputs`](crate::algos::ImageAlgorithm::measurement_outputs)):
///
/// ```ignore
/// let outputs = Outputs::of(&self.name).about(format!("channel {}", self.channel));
/// vec![
///     outputs.image("focus_score", "Focus score", MeasurementUnit::None),
///     outputs.image("percent_maximal", "Pixels at the maximum", MeasurementUnit::Percent)
///         .range(0.0, 100.0),
/// ]
/// ```
///
/// Each key is "step ID/value id" (step IDs are unique in the project), each name
/// gets what it is about in brackets: "Focus score (channel 0)".
pub struct Outputs<'a> {
    step: &'a str,
    about: Option<String>,
}

impl<'a> Outputs<'a> {
    /// The outputs of the measurement step with ID `step`.
    pub fn of(step: &'a str) -> Self {
        Self { step, about: None }
    }

    /// What every value is about, e.g. "channel 0" - added to its name.
    pub fn about(mut self, about: impl Into<String>) -> Self {
        self.about = Some(about.into());
        self
    }

    /// One value per image: `id` unique in the step (the key's end),
    /// `name` shown in the results.
    pub fn image(&self, id: &str, name: &str, unit: MeasurementUnit) -> MeasurementInfo {
        MeasurementInfo {
            key: format!("{}/{id}", self.step),
            name: match &self.about {
                Some(about) => format!("{name} ({about})"),
                None => name.to_string(),
            },
            step: self.step.to_string(),
            unit,
            target: MeasurementTarget::Image,
            range: None,
            aggregatable: true,
            related_objects: false,
            profile: false,
        }
    }

    /// One value per object - like [`Self::image`] otherwise.
    pub fn object(&self, id: &str, name: &str, unit: MeasurementUnit) -> MeasurementInfo {
        MeasurementInfo {
            target: MeasurementTarget::Object,
            ..self.image(id, name, unit)
        }
    }

    /// One value per track (a dataset step's) - like [`Self::image`] otherwise.
    pub fn track(&self, id: &str, name: &str, unit: MeasurementUnit) -> MeasurementInfo {
        MeasurementInfo {
            target: MeasurementTarget::Track,
            ..self.image(id, name, unit)
        }
    }
}

/// The measurements tile steps made, tile by tile (in a fixed tile order),
/// as one set for the image: an image value is the median over the tiles
/// that have one (per region; an even count: the mean of the middle two),
/// object values are kept together. In the order they first appear.
///
/// One tile (an image up to 4096 px) keeps its values as they are.
pub(crate) fn combine_tile_measurements(tiles: Vec<Vec<Measurement>>) -> Vec<Measurement> {
    use std::collections::HashMap;
    let mut order: Vec<MeasurementInfo> = Vec::new();
    let mut image: HashMap<String, Vec<(MeasurementRegion, Vec<f64>)>> = HashMap::new();
    let mut objects: HashMap<String, Vec<ObjectValue>> = HashMap::new();
    for measurement in tiles.into_iter().flatten() {
        let key = measurement.info.key.clone();
        if !order.iter().any(|info| info.key == key) {
            order.push(measurement.info);
        }
        match measurement.values {
            MeasurementValues::Image(values) => {
                let regions = image.entry(key).or_default();
                for (region, value) in values {
                    match regions.iter_mut().find(|(r, _)| *r == region) {
                        Some((_, list)) => list.push(value),
                        None => regions.push((region, vec![value])),
                    }
                }
            }
            MeasurementValues::Objects(values) => objects.entry(key).or_default().extend(values),
        }
    }
    order
        .into_iter()
        .map(|info| {
            let values = match objects.remove(&info.key) {
                Some(values) => MeasurementValues::Objects(values),
                None => MeasurementValues::Image(
                    image
                        .remove(&info.key)
                        .unwrap_or_default()
                        .into_iter()
                        .map(|(region, values)| (region, median(values)))
                        .collect(),
                ),
            };
            Measurement { info, values }
        })
        .collect()
}

/// The median of `values` (not empty).
fn median(mut values: Vec<f64>) -> f64 {
    values.sort_by(f64::total_cmp);
    let mid = values.len() / 2;
    if values.len() % 2 == 1 {
        values[mid]
    } else {
        (values[mid - 1] + values[mid]) / 2.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn image_measurement(key: &str, values: Vec<(MeasurementRegion, f64)>) -> Measurement {
        Measurement {
            info: Outputs::of("step").image(key, key, MeasurementUnit::None),
            values: MeasurementValues::Image(values),
        }
    }

    fn values_of(measurements: &[Measurement], key: &str) -> Vec<(MeasurementRegion, f64)> {
        match &measurements
            .iter()
            .find(|m| m.info.key == format!("step/{key}"))
            .unwrap()
            .values
        {
            MeasurementValues::Image(values) => values.clone(),
            MeasurementValues::Objects(_) => panic!("an image measurement"),
        }
    }

    #[test]
    fn one_tile_keeps_its_values() {
        let tile = vec![
            image_measurement("factor", vec![(MeasurementRegion::Image, 0.37)]),
            image_measurement("r2", vec![]),
        ];
        assert_eq!(combine_tile_measurements(vec![tile.clone()]), tile);
    }

    #[test]
    fn tiles_give_the_median_of_those_with_a_value() {
        let tile = |factor: Option<f64>, shift: f64| {
            vec![
                image_measurement(
                    "factor",
                    factor
                        .map(|f| (MeasurementRegion::Image, f))
                        .into_iter()
                        .collect(),
                ),
                image_measurement("shift", vec![(MeasurementRegion::Image, shift)]),
            ]
        };
        let combined = combine_tile_measurements(vec![
            tile(Some(0.30), 1.0),
            tile(None, 2.0),
            tile(Some(0.40), 4.0),
            tile(Some(0.36), 3.0),
        ]);
        assert_eq!(combined.len(), 2);
        assert_eq!(
            values_of(&combined, "factor"),
            [(MeasurementRegion::Image, 0.36)]
        );
        // An even count: the mean of the middle two.
        assert_eq!(
            values_of(&combined, "shift"),
            [(MeasurementRegion::Image, 2.5)]
        );
    }
}
