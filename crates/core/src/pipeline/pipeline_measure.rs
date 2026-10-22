//! The measurements of a plane: the types live in `evanalyzer_cfg` (they
//! travel to the app and over the server connection, like the objects).

pub use evanalyzer_cfg::core_types::{
    Measurement, MeasurementRegion, MeasurementValues, PipelineMeasurements,
};
