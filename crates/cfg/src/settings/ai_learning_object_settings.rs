use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// One selectable object-classifier feature, mapped directly onto `Object`'s
/// own measurement methods (`crates/core/src/object.rs`) — no filter chain
/// concept here, unlike pixel features, since these are plain
/// scalar measurements, not something composed from image filters.
///
/// Deliberately excludes `object_class` (the training label itself — including
/// it would be data leakage) and centroid/position (teaches location instead
/// of shape, a classic overfitting trap for this kind of tool).
///
/// SERIALIZATION-CRITICAL: this type is stored inside every saved object
/// classifier model. Once any model has been saved, existing variants must
/// never be renamed, removed, or have their meaning changed - treat this the
/// same as a database migration, not a normal refactor. Adding a new variant
/// is safe.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum ObjectMetric {
    Area,
    Perimeter,
    Circularity,
    Solidity,
    AspectRatio,
    Roundness,
    Compactness,
    FeretDiameter,
    MinFeretDiameter,
    EllipseMajor,
    EllipseMinor,
    EllipseAngle,
    Eccentricity,
    TouchesEdge,
    /// Sum of pixel intensities in the given image channel.
    IntensitySum(i32),
    IntensityMin(i32),
    IntensityMax(i32),
    IntensityAvg(i32),
    /// How the pixel values spread in the channel (CellProfiler's
    /// MeasureObjectIntensity).
    IntensityStd(i32),
    IntensityMedian(i32),
    IntensityLowerQuartile(i32),
    IntensityUpperQuartile(i32),
    IntensityMad(i32),
    /// Mean intensity in the ring around the object (`ring_width` px wide,
    /// neighbours included) - how bright its surroundings are.
    RingIntensityAvg(i32),
    /// The object's mean intensity minus its ring's: how much it stands out
    /// from its surroundings.
    RingContrast(i32),
}

impl ObjectMetric {
    /// The image channel an intensity feature is measured in; `None` for
    /// the shape features.
    pub fn channel(&self) -> Option<i32> {
        match self {
            ObjectMetric::IntensitySum(c)
            | ObjectMetric::IntensityMin(c)
            | ObjectMetric::IntensityMax(c)
            | ObjectMetric::IntensityAvg(c)
            | ObjectMetric::IntensityStd(c)
            | ObjectMetric::IntensityMedian(c)
            | ObjectMetric::IntensityLowerQuartile(c)
            | ObjectMetric::IntensityUpperQuartile(c)
            | ObjectMetric::IntensityMad(c)
            | ObjectMetric::RingIntensityAvg(c)
            | ObjectMetric::RingContrast(c) => Some(*c),
            ObjectMetric::Area
            | ObjectMetric::Perimeter
            | ObjectMetric::Circularity
            | ObjectMetric::Solidity
            | ObjectMetric::AspectRatio
            | ObjectMetric::Roundness
            | ObjectMetric::Compactness
            | ObjectMetric::FeretDiameter
            | ObjectMetric::MinFeretDiameter
            | ObjectMetric::EllipseMajor
            | ObjectMetric::EllipseMinor
            | ObjectMetric::EllipseAngle
            | ObjectMetric::Eccentricity
            | ObjectMetric::TouchesEdge => None,
        }
    }

    /// True for the features measured on the ring around the object.
    pub fn uses_ring(&self) -> bool {
        matches!(
            self,
            ObjectMetric::RingIntensityAvg(_) | ObjectMetric::RingContrast(_)
        )
    }
}

/// The ring width when none is set: 5 px.
pub const DEFAULT_RING_WIDTH: u32 = 5;

fn default_ring_width() -> u32 {
    DEFAULT_RING_WIDTH
}

/// The closed, versioned feature recipe for an object classifier: one entry
/// per feature, in the order features are assembled into a training/inference
/// row. Bundled with the saved classifier so training and inference reproduce
/// identical inputs.
///
/// SERIALIZATION-CRITICAL — see `ObjectMetric` doc comment above.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct AiLearningObjectFeatureSettings {
    pub metrics: Vec<ObjectMetric>,
    /// Width in px of the ring around each object the ring features are
    /// measured in (every pixel at most this far from the object).
    #[serde(default = "default_ring_width")]
    pub ring_width: u32,
}

impl Default for AiLearningObjectFeatureSettings {
    fn default() -> Self {
        Self {
            metrics: Vec::new(),
            ring_width: DEFAULT_RING_WIDTH,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `SCREAMING_SNAKE_CASE` tags, matching every other enum in the
    /// codebase - not released yet, so no backward-compat aliases needed for
    /// the old bare-PascalCase tags this used to produce.
    #[test]
    fn object_metric_serializes_unit_and_tuple_variants_as_screaming_snake_case() {
        assert_eq!(serde_json::to_value(ObjectMetric::Area).unwrap(), "AREA");
        assert_eq!(
            serde_json::to_value(ObjectMetric::TouchesEdge).unwrap(),
            "TOUCHES_EDGE"
        );
        assert_eq!(
            serde_json::to_value(ObjectMetric::IntensitySum(2)).unwrap(),
            serde_json::json!({"INTENSITY_SUM": 2})
        );
    }

    #[test]
    fn intensity_features_name_their_channel_and_ring_ones_use_the_ring() {
        assert_eq!(ObjectMetric::Area.channel(), None);
        assert_eq!(ObjectMetric::TouchesEdge.channel(), None);
        assert_eq!(ObjectMetric::IntensityMad(2).channel(), Some(2));
        assert_eq!(ObjectMetric::RingContrast(1).channel(), Some(1));
        assert!(ObjectMetric::RingIntensityAvg(0).uses_ring());
        assert!(!ObjectMetric::IntensityAvg(0).uses_ring());
    }

    /// A model saved without a ring width reads as the default.
    #[test]
    fn a_missing_ring_width_is_the_default() {
        let spec: AiLearningObjectFeatureSettings =
            serde_json::from_str(r#"{"metrics": ["AREA"]}"#).unwrap();
        assert_eq!(spec.ring_width, DEFAULT_RING_WIDTH);
        assert_eq!(AiLearningObjectFeatureSettings::default().ring_width, 5);
    }
}
