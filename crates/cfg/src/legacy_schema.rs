//! Deserialize-only mirror of the old (`.icproj`) C++/nlohmann-json project format.
//!
//! Reverse-engineered directly from the old application's source tree (`docs/old`),
//! since no JSON Schema for that format was ever published. Every old struct used
//! `NLOHMANN_DEFINE_TYPE_INTRUSIVE_WITH_DEFAULT_EXTENDED`, which makes every single
//! field optional on deserialize (missing -> default-constructed) - `#[serde(default)]`
//! at the struct level mirrors that leniency here, since real-world `.icproj` files
//! span many schema revisions of the old app.
//!
//! Only fields actually consumed by `legacy_import` are modeled with real types;
//! command settings the importer doesn't (yet) convert are left as raw
//! `serde_json::Value` so their *presence* can still be detected and reported,
//! without needing to fully model their internals.

use serde::Deserialize;
use serde_json::Value;

#[derive(Debug, Default, Deserialize)]
#[serde(default)]
pub struct LegacySettingsMeta {
    pub name: String,
    pub notes: String,
    pub color: String,
    pub icon: String,
    pub revision: String,
    pub uid: String,
    #[serde(rename = "modifiedAt")]
    pub modified_at: Option<String>,
    pub group: Option<String>,
    pub category: Option<String>,
    pub tags: Vec<String>,
    pub author: Option<String>,
    pub organization: Option<String>,
    pub webpage: Option<String>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(default)]
pub struct LegacyExperimentSettings {
    #[serde(rename = "experimentId")]
    pub experiment_id: String,
    #[serde(rename = "experimentName")]
    pub experiment_name: String,
    pub notes: String,
}

#[derive(Debug, Default, Deserialize)]
#[serde(default)]
pub struct LegacyAddress {
    #[serde(rename = "firstName")]
    pub first_name: String,
    #[serde(rename = "lastName")]
    pub last_name: String,
    pub country: String,
    pub organization: String,
    #[serde(rename = "streetAddress")]
    pub street_address: String,
    #[serde(rename = "postalCode")]
    pub postal_code: String,
    pub city: String,
    pub email: String,
}

#[derive(Debug, Default, Deserialize)]
#[serde(default)]
pub struct LegacyResultsTemplate {
    /// One of `enums::Measurement`'s JSON strings (e.g. "AreaSize", "IntensitySum").
    #[serde(rename = "measureChannel")]
    pub measure_channel: String,
    /// `enums::Stats` JSON strings (e.g. "Avg", "Max"); a JSON array since the old
    /// type is a `std::set`.
    pub stats: Vec<String>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(default)]
pub struct LegacyClass {
    /// One of `enums::ClassId`'s JSON strings: "None", "Undefined", or a numeric
    /// string "0".."49".
    #[serde(rename = "classId")]
    pub class_id: String,
    pub name: String,
    pub color: String,
    pub notes: String,
    #[serde(rename = "defaultMeasurements")]
    pub default_measurements: Vec<LegacyResultsTemplate>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(default)]
pub struct LegacyClassification {
    pub meta: LegacySettingsMeta,
    pub classes: Vec<LegacyClass>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(default)]
pub struct LegacyPlateSetup {
    pub rows: i32,
    pub cols: i32,
    #[serde(rename = "wellImageOrder")]
    pub well_image_order: Vec<Vec<i32>>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(default)]
pub struct LegacyPlate {
    #[serde(rename = "plateId")]
    pub plate_id: u8,
    pub name: String,
    pub notes: String,
    #[serde(rename = "imageFolder")]
    pub image_folder: String,
    /// `enums::GroupBy` JSON string: "Unknown", "Off", "Directory", "Filename".
    #[serde(rename = "groupBy")]
    pub group_by: String,
    #[serde(rename = "filenameRegex")]
    pub filename_regex: String,
    #[serde(rename = "plateSetup")]
    pub plate_setup: LegacyPlateSetup,
}

#[derive(Debug, Default, Deserialize)]
#[serde(default)]
pub struct LegacyPhysicalSizeSettings {
    /// "Automatic" or "Manual".
    pub mode: String,
    #[serde(rename = "pixelSizeUnit")]
    pub pixel_size_unit: String,
    #[serde(rename = "pixelWidth")]
    pub pixel_width: f32,
    #[serde(rename = "pixelHeight")]
    pub pixel_height: f32,
}

#[derive(Debug, Default, Deserialize)]
#[serde(default)]
pub struct LegacyZStackSettings {
    #[serde(rename = "defaultZProjection")]
    pub default_z_projection: String,
}

#[derive(Debug, Default, Deserialize)]
#[serde(default)]
pub struct LegacyTStackSettings {
    #[serde(rename = "startFrame")]
    pub start_frame: i32,
    #[serde(rename = "endFrame")]
    pub end_frame: i32,
}

#[derive(Debug, Default, Deserialize)]
#[serde(default)]
pub struct LegacyProjectImageSetup {
    #[serde(rename = "imagePixelSizeSettings")]
    pub image_pixel_size_settings: LegacyPhysicalSizeSettings,
    pub series: i32,
    #[serde(rename = "tStackSettings")]
    pub t_stack_settings: LegacyTStackSettings,
    #[serde(rename = "zStackSettings")]
    pub z_stack_settings: LegacyZStackSettings,
}

#[derive(Debug, Default, Deserialize)]
#[serde(default)]
pub struct LegacyProjectSettings {
    #[serde(rename = "experimentSettings")]
    pub experiment_settings: LegacyExperimentSettings,
    pub plate: LegacyPlate,
    pub address: LegacyAddress,
    pub classification: LegacyClassification,
    /// Doubly-legacy multi-plate list, kept by the old app only for migration.
    /// Used as a fallback when `plate` itself looks unset.
    pub plates: Vec<LegacyPlate>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(default)]
pub struct LegacyPipelineSettings {
    /// "FromFile", "FromMemory", or "FromBlank".
    pub source: String,
    #[serde(rename = "cStackIndex")]
    pub c_stack_index: i32,
    /// One of `enums::ClassIdIn`'s JSON strings - may be "$", "None", "Undefined",
    /// a numeric string, or "M01".."M11".
    #[serde(rename = "defaultClassId")]
    pub default_class_id: String,
}

// ---------------------------------------------------------------------------
// Pipeline step command settings - only the commands `legacy_import` actually
// converts get a real struct; the rest are detected (for warnings) via presence
// of the corresponding `$xxx` key, holding a raw `Value`.
// ---------------------------------------------------------------------------

#[derive(Debug, Default, Deserialize)]
#[serde(default)]
pub struct LegacyBlurSettings {
    /// "Blur" (box) or "GaussianBlur".
    pub mode: String,
    #[serde(rename = "kernelSize")]
    pub kernel_size: i32,
    pub repeat: i32,
}

#[derive(Debug, Default, Deserialize)]
#[serde(default)]
pub struct LegacyEdgeDetectionCannySettings {
    #[serde(rename = "kernelSize")]
    pub kernel_size: i32,
    #[serde(rename = "thresholdMin")]
    pub threshold_min: f32,
    #[serde(rename = "thresholdMax")]
    pub threshold_max: f32,
}

#[derive(Debug, Default, Deserialize)]
#[serde(default)]
pub struct LegacyEdgeDetectionSobelSettings {
    #[serde(rename = "kernelSize")]
    pub kernel_size: i32,
}

#[derive(Debug, Default, Deserialize)]
#[serde(default)]
pub struct LegacyRollingBallSettings {
    /// "Ball" or "Paraboloid".
    #[serde(rename = "ballType")]
    pub ball_type: String,
    #[serde(rename = "ballSize")]
    pub ball_size: i32,
}

#[derive(Debug, Default, Deserialize)]
#[serde(default)]
pub struct LegacyMedianSubtractSettings {
    #[serde(rename = "kernelSize")]
    pub kernel_size: i32,
}

#[derive(Debug, Default, Deserialize)]
#[serde(default)]
pub struct LegacyEnhanceContrastSettings {
    #[serde(rename = "saturatedPixels")]
    pub saturated_pixels: f32,
    pub normalize: bool,
    #[serde(rename = "equalizeHistogram")]
    pub equalize_histogram: bool,
}

#[derive(Debug, Default, Deserialize)]
#[serde(default)]
pub struct LegacyHessianSettings {
    /// "Determinant", "EigenvaluesX", or "EigenvaluesY".
    pub mode: String,
}

#[derive(Debug, Default, Deserialize)]
#[serde(default)]
pub struct LegacyStructureTensorSettings {
    /// "Coherence", "EigenvaluesX", or "EigenvaluesY".
    pub mode: String,
    #[serde(rename = "kernelSize")]
    pub kernel_size: i32,
}

#[derive(Debug, Default, Deserialize)]
#[serde(default)]
pub struct LegacyWeightedDeviationSettings {
    #[serde(rename = "kernelSize")]
    pub kernel_size: i32,
    pub sigma: f64,
}

#[derive(Debug, Default, Deserialize)]
#[serde(default)]
pub struct LegacyRankFilterSettings {
    /// "Mean", "Min", "Max", "Variance", "Median", "Outliers", "Despeckle",
    /// "RemoveNaN", "Open", "Close", or "TopHat".
    pub mode: String,
    pub radius: f64,
}

#[derive(Debug, Default, Deserialize)]
#[serde(default)]
pub struct LegacyLaplacianSettings {
    #[serde(rename = "kernelSize")]
    pub kernel_size: i32,
}

#[derive(Debug, Default, Deserialize)]
#[serde(default)]
pub struct LegacyMorphologicalTransformSettings {
    /// "Unknown", "Erode", "Dilate", "Open", "Close", "Gradient", "TopHat",
    /// "BlackHat", or "Hitmiss".
    pub function: String,
    /// "Rectangle", "Cross", or "Ellipse".
    pub shape: String,
    #[serde(rename = "kernelSize")]
    pub kernel_size: i32,
}

#[derive(Debug, Default, Deserialize)]
#[serde(default)]
pub struct LegacyHsvColor {
    pub hue: i32,
    pub sat: i32,
    pub val: i32,
}

#[derive(Debug, Default, Deserialize)]
#[serde(default)]
pub struct LegacyColorFilterFilter {
    #[serde(rename = "colorRangeFrom")]
    pub color_range_from: LegacyHsvColor,
    #[serde(rename = "colorRangeTo")]
    pub color_range_to: LegacyHsvColor,
}

#[derive(Debug, Default, Deserialize)]
#[serde(default)]
pub struct LegacyColorFilterSettings {
    pub filter: Vec<LegacyColorFilterFilter>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(default)]
pub struct LegacyIntensityTransformationSettings {
    /// "Manual" or "Automatic".
    pub mode: String,
    pub contrast: f32,
    pub brightness: i32,
    pub gamma: i32,
}

#[derive(Debug, Default, Deserialize)]
#[serde(default)]
pub struct LegacyThresholdEntry {
    /// One of `ThresholdSettings::Methods`' JSON strings (e.g. "Otsu", "Triangle",
    /// "Percentil" [sic], "TenyiEntropy" [sic]).
    pub method: String,
    #[serde(rename = "thresholdMin")]
    pub threshold_min: f32,
    #[serde(rename = "thresholdMax")]
    pub threshold_max: f32,
    #[serde(rename = "pixelClassId")]
    pub pixel_class_id: i32,
}

#[derive(Debug, Default, Deserialize)]
#[serde(default)]
pub struct LegacyThresholdSettings {
    #[serde(rename = "modelClasses")]
    pub model_classes: Vec<LegacyThresholdEntry>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(default)]
pub struct LegacyWatershedSettings {
    #[serde(rename = "maximumFinderTolerance")]
    pub maximum_finder_tolerance: f32,
}

#[derive(Debug, Default, Deserialize)]
#[serde(default)]
pub struct LegacyImageSaverSettings {
    pub path: String,
}

#[derive(Debug, Default, Deserialize)]
#[serde(default)]
pub struct LegacyVoronoiGridSettings {
    #[serde(rename = "inputClassesPoints")]
    pub input_classes_points: Vec<String>,
    #[serde(rename = "outputClassVoronoi")]
    pub output_class_voronoi: String,
    #[serde(rename = "inputClassesMask")]
    pub input_classes_mask: Vec<String>,
    #[serde(rename = "excludeAreasWithoutPoint")]
    pub exclude_areas_without_point: bool,
    #[serde(rename = "excludeAreasAtTheEdge")]
    pub exclude_areas_at_the_edge: bool,
    #[serde(rename = "maxRadius")]
    pub max_radius: f32,
}

#[derive(Debug, Default, Deserialize)]
#[serde(default)]
pub struct LegacyIntensityFilter {
    #[serde(rename = "minIntensity")]
    pub min_intensity: f32,
    #[serde(rename = "maxIntensity")]
    pub max_intensity: f32,
}

#[derive(Debug, Default, Deserialize)]
#[serde(default)]
pub struct LegacyMetricsFilter {
    #[serde(rename = "minParticleSize")]
    pub min_particle_size: f32,
    #[serde(rename = "maxParticleSize")]
    pub max_particle_size: f32,
    #[serde(rename = "minCircularity")]
    pub min_circularity: f32,
    #[serde(rename = "excludeObjectsAtTheEdge")]
    pub exclude_objects_at_the_edge: bool,
}

#[derive(Debug, Default, Deserialize)]
#[serde(default)]
pub struct LegacyClassifierFilter {
    #[serde(rename = "outputClass")]
    pub output_class: String,
    pub intensity: LegacyIntensityFilter,
    pub metrics: LegacyMetricsFilter,
}

#[derive(Debug, Default, Deserialize)]
#[serde(default)]
pub struct LegacyClassifierObjectClass {
    pub filters: Vec<LegacyClassifierFilter>,
    #[serde(rename = "outputClassNoMatch")]
    pub output_class_no_match: String,
    #[serde(rename = "pixelClassId")]
    pub pixel_class_id: i32,
}

#[derive(Debug, Default, Deserialize)]
#[serde(default)]
pub struct LegacyClassifierSettings {
    #[serde(rename = "modelClasses")]
    pub model_classes: Vec<LegacyClassifierObjectClass>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(default)]
pub struct LegacyObjectTransformSettings {
    /// "Scale", "SnapArea", "CircleMin", "Circle", or "FitEllipse".
    pub function: String,
    #[serde(rename = "inputClasses")]
    pub input_classes: String,
    #[serde(rename = "outputClasses")]
    pub output_classes: String,
    pub factor: f32,
}

/// A pipeline step: exactly one of the `$xxx` optional fields below should be
/// populated (this is how the old format encodes "which command this step runs" -
/// there is no discriminant `type` field, unlike the new format).
#[derive(Debug, Default, Deserialize)]
#[serde(default)]
pub struct LegacyPipelineStep {
    pub disabled: bool,
    pub locked: bool,
    #[serde(rename = "breakPoint")]
    pub break_point: bool,

    // ---- Supported: image-level filters ----
    #[serde(rename = "$blur")]
    pub blur: Option<LegacyBlurSettings>,
    #[serde(rename = "$canny")]
    pub canny: Option<LegacyEdgeDetectionCannySettings>,
    #[serde(rename = "$sobel")]
    pub sobel: Option<LegacyEdgeDetectionSobelSettings>,
    #[serde(rename = "$rollingBall")]
    pub rolling_ball: Option<LegacyRollingBallSettings>,
    #[serde(rename = "$medianSubtract")]
    pub median_subtract: Option<LegacyMedianSubtractSettings>,
    #[serde(rename = "$enhanceContrast")]
    pub enhance_contrast: Option<LegacyEnhanceContrastSettings>,
    #[serde(rename = "$hessian")]
    pub hessian: Option<LegacyHessianSettings>,
    #[serde(rename = "$structureTensor")]
    pub structure_tensor: Option<LegacyStructureTensorSettings>,
    #[serde(rename = "$gaussianWeightedDev")]
    pub gaussian_weighted_dev: Option<LegacyWeightedDeviationSettings>,
    #[serde(rename = "$rank")]
    pub rank: Option<LegacyRankFilterSettings>,
    #[serde(rename = "$laplacian")]
    pub laplacian: Option<LegacyLaplacianSettings>,
    #[serde(rename = "$morphologicalTransform")]
    pub morphological_transform: Option<LegacyMorphologicalTransformSettings>,
    #[serde(rename = "$colorFilter")]
    pub color_filter: Option<LegacyColorFilterSettings>,
    #[serde(rename = "$intensityTransform")]
    pub intensity_transform: Option<LegacyIntensityTransformationSettings>,

    // ---- Supported: segmentation ----
    #[serde(rename = "$threshold")]
    pub threshold: Option<LegacyThresholdSettings>,
    #[serde(rename = "$watershed")]
    pub watershed: Option<LegacyWatershedSettings>,

    // ---- Supported: object-level ----
    #[serde(rename = "$voronoi")]
    pub voronoi: Option<LegacyVoronoiGridSettings>,
    #[serde(rename = "$classify")]
    pub classify: Option<LegacyClassifierSettings>,
    #[serde(rename = "$objectTransform")]
    pub object_transform: Option<LegacyObjectTransformSettings>,
    #[serde(rename = "$saveImage")]
    pub save_image: Option<LegacyImageSaverSettings>,

    // ---- Not (yet) converted - presence-detected only, for warnings ----
    #[serde(rename = "$thresholdAdaptive")]
    pub threshold_adaptive: Option<Value>,
    #[serde(rename = "$crop")]
    pub crop: Option<Value>,
    #[serde(rename = "$fillHoles")]
    pub fill_holes: Option<Value>,
    #[serde(rename = "$houghTransform")]
    pub hough_transform: Option<Value>,
    #[serde(rename = "$skeletonize")]
    pub skeletonize: Option<Value>,
    #[serde(rename = "$nop")]
    pub nop: Option<Value>,
    #[serde(rename = "$imageToCache")]
    pub image_to_cache: Option<Value>,
    #[serde(rename = "$imageMath")]
    pub image_math: Option<Value>,
    #[serde(rename = "$thresholdValidator")]
    pub threshold_validator: Option<Value>,
    #[serde(rename = "$noiseValidator")]
    pub noise_validator: Option<Value>,
    #[serde(rename = "$objectsToImage")]
    pub objects_to_image: Option<Value>,
    #[serde(rename = "$aiClassify")]
    pub ai_classify: Option<Value>,
    #[serde(rename = "$pixelClassify")]
    pub pixel_classify: Option<Value>,
    #[serde(rename = "$colocalization")]
    pub colocalization: Option<Value>,
    #[serde(rename = "$reclassify")]
    pub reclassify: Option<Value>,
    #[serde(rename = "$measureIntensity")]
    pub measure_intensity: Option<Value>,
    #[serde(rename = "$measureDistance")]
    pub measure_distance: Option<Value>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(default)]
pub struct LegacyPipeline {
    pub meta: LegacySettingsMeta,
    #[serde(rename = "pipelineSetup")]
    pub pipeline_setup: LegacyPipelineSettings,
    #[serde(rename = "pipelineSteps")]
    pub pipeline_steps: Vec<LegacyPipelineStep>,
    pub disabled: bool,
    pub locked: bool,
}

#[derive(Debug, Default, Deserialize)]
#[serde(default)]
pub struct LegacyAnalyzeSettings {
    #[serde(rename = "projectSettings")]
    pub project_settings: LegacyProjectSettings,
    #[serde(rename = "imageSetup")]
    pub image_setup: LegacyProjectImageSetup,
    pub pipelines: Vec<LegacyPipeline>,
    pub meta: LegacySettingsMeta,
}
