use evanalyzer_cfg::core_types::{MeasurementInfo, ObjectClass, ObjectId};

/// The measurements of one image plane (the pipeline cache is per plane).
#[derive(Default, Clone)]
pub struct PipelineMeasurements {
    pub measurements: Vec<Measurement>,
}

/// One measurement (e.g. "Pearson ch1 × ch2" of step "nuclei_coloc") and its
/// values.
#[derive(Clone)]
pub struct Measurement {
    pub info: MeasurementInfo,
    pub values: MeasurementValues,
}

/// Per object, or per image region - which one the info's `target` says.
#[derive(Clone)]
pub enum MeasurementValues {
    Objects(Vec<(ObjectId, f64)>),
    Image(Vec<(MeasurementRegion, f64)>),
}

/// Which pixels of the image an image measurement covers.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum MeasurementRegion {
    Image,
    Class(ObjectClass),
}
