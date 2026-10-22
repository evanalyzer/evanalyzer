//! The measurements of a plane: the types live in `evanalyzer_cfg` (they
//! travel to the app and over the server connection, like the objects).

pub use evanalyzer_cfg::core_types::{
    Measurement, MeasurementRegion, MeasurementValues, PipelineMeasurements,
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
        }
    }
}
