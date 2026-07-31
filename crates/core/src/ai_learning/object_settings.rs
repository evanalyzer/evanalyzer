use serde::{Deserialize, Serialize};

/// One selectable object-classifier feature, mapped directly onto `Object`'s
/// own measurement methods (`crates/core/src/object.rs`) — no filter chain
/// concept here, unlike pixel features, since these are already-computed
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
#[derive(Debug, Clone, Serialize, Deserialize)]
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
}

/// The closed, versioned feature recipe for an object classifier: one entry
/// per feature, in the order features are assembled into a training/inference
/// row. Bundled with the saved classifier so training and inference reproduce
/// identical inputs.
///
/// SERIALIZATION-CRITICAL — see `ObjectMetric` doc comment above.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ObjectFeatureSpec {
    pub metrics: Vec<ObjectMetric>,
}
