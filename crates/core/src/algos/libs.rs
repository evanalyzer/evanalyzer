//! Building blocks the commands share - geometry with no pipeline types of
//! its own (each command stays one module; these are what several use).

/// For every pixel the nearest marked pixel (SciPy's exact Euclidean
/// feature transform).
pub(crate) mod feature_transform;
/// A uniform grid over bounding boxes, to find the objects near one.
pub(crate) mod spatial_grid;
