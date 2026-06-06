// @generated - do not edit by hand
use crate::core_types::{PixelUnits, SizeUnits};
use crate::modules::parameter_def::{ParamType, ParameterDef};
use crate::modules::pipeline_command_settings::*;
use crate::types::classes::{ObjectClass, SegmentationClass};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub enum CommandCategory {
    Preprocess,
    Segment,
    Object,
    Measure,
    Classify,
}

impl CommandCategory {
    /// Ordered position in the pipeline (0 = first, higher = later).
    #[allow(dead_code)]
    pub fn display_order(self) -> u8 {
        match self {
            Self::Preprocess => 0,
            Self::Segment => 1,
            Self::Object => 2,
            Self::Measure => 3,
            Self::Classify => 4,
        }
    }

    /// Which categories are valid immediately before this one in a pipeline.
    /// An empty slice means this category can start a pipeline.
    #[allow(dead_code)]
    pub fn allowed_after(self) -> &'static [CommandCategory] {
        match self {
            Self::Preprocess => &[Self::Preprocess],
            Self::Segment => &[Self::Preprocess, Self::Segment],
            Self::Object => &[Self::Segment, Self::Object],
            Self::Measure => &[Self::Object, Self::Measure],
            Self::Classify => &[Self::Measure, Self::Classify],
        }
    }

    /// The natural next category after this one, used to pre-filter the command picker.
    #[allow(dead_code)]
    pub fn suggested_next(self) -> CommandCategory {
        match self {
            Self::Preprocess => Self::Segment,
            Self::Segment => Self::Object,
            Self::Object => Self::Measure,
            Self::Measure => Self::Classify,
            Self::Classify => Self::Classify,
        }
    }
}

#[derive(Serialize, Deserialize, Debug, Clone, JsonSchema)]
#[serde(tag = "type", rename_all = "camelCase")]
pub enum PipelineCommand {
    Blur(BlurSettings),
    ClassifyRois(ClassifyRoisSettings),
    Colocalization(ColocalizationSettings),
    ColorFilterCommand(ColorFilterCommandSettings),
    ConnectedComponents(ConnectedComponentsSettings),
    DistanceTransform(DistanceTransformSettings),
    EdgeDetectionCanny(EdgeDetectionCannySettings),
    EdgeDetectionSobel(EdgeDetectionSobelSettings),
    EnhanceContrast(EnhanceContrastSettings),
    ExtractRois(ExtractRoisSettings),
    GaussianBlur(GaussianBlurSettings),
    Hessian(HessianSettings),
    ImageCache(ImageCacheSettings),
    ImageMath(ImageMathSettings),
    IntensityTransformation(IntensityTransformationSettings),
    Laplacian(LaplacianSettings),
    MedianSubtract(MedianSubtractSettings),
    MorphologicalCommand(MorphologicalCommandSettings),
    RankFilter(RankFilterSettings),
    RollingBall(RollingBallSettings),
    SaveImage(SaveImageSettings),
    StructureTensor(StructureTensorSettings),
    Threshold(ThresholdSettings),
    Voronoi(VoronoiSettings),
    Watershed(WatershedSettings),
    WeightedDeviation(WeightedDeviationSettings),
}

#[allow(dead_code)]
pub struct CommandMeta {
    pub id: i32,
    pub name: &'static str,
    pub category: CommandCategory,
    pub summary: &'static str,
    pub description: &'static str,
}

#[allow(dead_code)]
pub fn all_command_meta() -> Vec<CommandMeta> {
    vec![
        CommandMeta { id: 0, name: "Blur", category: CommandCategory::Preprocess, summary: "Smooths an image by averaging pixel intensities within a local neighborhood.", description: "This algorithm applies a uniform box filter where every pixel within the moving\nwindow contributes equally to the final value. It is a computationally fast\nmethod used for general image smoothing, blending variations, and rapid noise\nsuppression where edge precision is less critical." },
        CommandMeta { id: 1, name: "ClassifyRois", category: CommandCategory::Classify, summary: "Classifies ROIs based on morphological and intensity features.", description: "This command applies rule-based classification logic to assign object classes\nto extracted ROIs. Classification is performed using configurable criteria\nincluding area, shape descriptors, and intensity statistics." },
        CommandMeta { id: 2, name: "Colocalization", category: CommandCategory::Classify, summary: "Calculates spatial colocalization and intersections between specified object classes.", description: "This command scans the ROI cache, groups objects by their designated classes,\nand performs spatial overlap analysis. It records colocalization relationships\nbetween intersecting entities and can optionally generate new child ROIs representing\nthe precise intersection regions." },
        CommandMeta { id: 3, name: "ColorFilterCommand", category: CommandCategory::Preprocess, summary: "A command that filters an image based on a specific HSV color range.", description: "Pixels falling outside the provided [`HsvRange`] are masked\nout by setting to black.\n\n# Examples\n\n```\n# use imagec::backend::algos::{ColorFilterCommand, HsvRange};\nlet range = HsvRange {\nmin_h: 0.0,   max_h: 30.0, // Red tones\nmin_s: 0.5,   max_s: 1.0,\nmin_v: 0.5,   max_v: 1.0,\n};\n\nlet command = ColorFilterCommand { range };\n```" },
        CommandMeta { id: 4, name: "ConnectedComponents", category: CommandCategory::Object, summary: "Identifies and labels discrete objects within a binary or multi-class image.", description: "" },
        CommandMeta { id: 5, name: "DistanceTransform", category: CommandCategory::Preprocess, summary: "A command that calculates the Euclidean Distance Map (EDM) of an f32 image.", description: "This algorithm identifies pixels below a threshold as \"background\" and\ncalculates the distance of every \"foreground\" pixel to the nearest background pixel." },
        CommandMeta { id: 6, name: "EdgeDetectionCanny", category: CommandCategory::Preprocess, summary: "Extracts structural boundaries and fine edges using the multi-stage Canny algorithm.", description: "This algorithm identifies optimal edge locations by calculating spatial intensity\ngradients, suppressing non-maximum pixel responses to thin lines down to 1-pixel width,\nand applying a dual-threshold hysteresis loop to preserve weak edges connected\nto strong ones while completely rejecting isolated noise.\n\n# Examples\n\n```\n# use imagec::backend::algos::EdgeDetectionCanny;\nlet edges = EdgeDetectionCanny {\nkernel_size: 3,\nthreshold_min: 0.1,\nthreshold_max: 0.3,\n};\n```" },
        CommandMeta { id: 7, name: "EdgeDetectionSobel", category: CommandCategory::Preprocess, summary: "Extracts directional boundaries by computing spatial image intensity gradients.", description: "This algorithm applies localized 3x3 kernels to approximate the first derivative\nof pixel intensities across the horizontal and vertical axes. It highlights\nareas of sharp luminance changes, producing a continuous gradient map that\nemphasizes prominent structural edges and surface transitions.\n\n# Examples\n\n```\n# use imagec::backend::algos::EdgeDetectionSobel;\nlet filter = EdgeDetectionSobel { kernel_size: 3 };\n```" },
        CommandMeta { id: 8, name: "EnhanceContrast", category: CommandCategory::Preprocess, summary: "Configuration for contrast enhancement and histogram manipulation.", description: "This algorithm can perform linear contrast stretching, normalization,\nor histogram equalization to improve the dynamic range of an image.\n\n# Examples\n\n```\n# use imagec::backend::algos::EnhanceContrast;\nlet settings = EnhanceContrast {\nsaturated_pixels: 0.01,   // Clip 1% of outliers\nnormalize: true,          // Stretch to [0.0, 1.0]\nequalize_histogram: false,\n};\n```" },
        CommandMeta { id: 9, name: "ExtractRois", category: CommandCategory::Measure, summary: "Represents a bounded region of interest extracted from a labeled image.", description: "A command to extract spatial statistics and bounding boxes from labeled objects." },
        CommandMeta { id: 10, name: "GaussianBlur", category: CommandCategory::Preprocess, summary: "Smooths an image and reduces background noise using a Gaussian kernel.", description: "This algorithm applies a localized, bell-curve weighted blur that suppresses\nhigh-frequency pixel variations (like camera noise, salt-and-pepper artifacts,\nor dust) while preserving structural features. It is commonly used as a\npreprocessing step to optimize thresholding and edge detection tasks.\n\n# Examples\n\n```\nuse imagec::backend::algos::GaussianBlur;\n\nlet settings = GaussianBlur {\nkernel_size: 5,\nsigma: 2.0\n};\n```" },
        CommandMeta { id: 11, name: "Hessian", category: CommandCategory::Preprocess, summary: "Extracts continuous structural ridges, tubular vessels, and blobs using second-order spatial derivatives.", description: "This algorithm constructs a localized Hessian matrix for each pixel to analyze local curvature\nand intensity topography. By evaluating the eigenvalues of this matrix, it differentiates\nbetween directional ridges (like blood vessels or filaments), distinct intensity peaks (blobs),\nand flat regions, making it highly effective for curvilinear feature extraction.\n\n# Examples\n\n```\n# use imagec::backend::algos::{Hessian, HessianMode};\nlet detector = Hessian {\nmode: HessianMode::Determinant,\n};\n```" },
        CommandMeta { id: 12, name: "ImageCache", category: CommandCategory::Preprocess, summary: "A filter that acts as a synchronization point between the pipeline and a storage backend.", description: "`ImageCache` allows the pipeline to branch or \"undo\" operations by saving\nstates to a named address and reloading them as needed.\n\n# Examples\n\n```\nuse imagec::backend::core::context::{ImageCache, ImageCacheMode, ImageAddress};\nlet checkpoint = ImageCache {\nmode: ImageCacheMode::Store,\naddress: ImageAddress::from(\"pre_processed_state\"),\n};\n```" },
        CommandMeta { id: 13, name: "ImageMath", category: CommandCategory::Preprocess, summary: "A filter that performs pixel-wise mathematical operations between the current", description: "pipeline image and a secondary image stored in the cache.\n\nThis command allows for complex image blending, masking, and comparison.\n\n# Examples\n\n```\nuse imagec::backend::algos::{ImageMath, Operand};\nlet subtract_bg = ImageMath {\noperand: Operand::Subtract,\nsecond_image_address: ImageAddress::from(\"background\"),\nswap_operands: false,\n};\n```" },
        CommandMeta { id: 14, name: "IntensityTransformation", category: CommandCategory::Preprocess, summary: "Configuration for adjusting image contrast and brightness.", description: "This transformation applies a linear mapping to pixel values.\nIn [`Mode::Manual`], the output is typically calculated as:\n`output = input * contrast + brightness`." },
        CommandMeta { id: 15, name: "Laplacian", category: CommandCategory::Preprocess, summary: "Configuration for the Laplacian edge detection filter.", description: "The Laplacian is a second-order derivative operator used to find regions of\nrapid intensity change. It is particularly effective for detecting edges\nand fine details, though it is highly sensitive to noise.\n\n# Examples\n\n```\n# use imagec::backend::algos::Laplacian;\nlet filter = Laplacian { kernel_size: 3 };\n```" },
        CommandMeta { id: 16, name: "MedianSubtract", category: CommandCategory::Preprocess, summary: "A background subtraction filter that uses a median rank operator.", description: "This algorithm is highly effective for removing large-scale background\nvariations while preserving small, high-contrast features. It works by\nestimating the background as the median intensity within a local radius.\n\n# Examples\n\n```\nuse imagec::backend::algos::MedianSubtract;\nlet filter = MedianSubtract { radius: 10.0 };\n```" },
        CommandMeta { id: 17, name: "MorphologicalCommand", category: CommandCategory::Preprocess, summary: "A filter that applies mathematical morphology to an image.", description: "Morphological operations use a structuring element (kernel) to probe\nand modify the shapes within an image.\n\n# Examples\n\n```\nuse imagec::backend::algos::{MorphologicalCommand, MorphOps, KernelShapes};\nlet clean_noise = MorphologicalCommand {\nop: MorphOps::Open,\nkernel_size: 3,\nkernel_shape: KernelShapes::Ellipse,\n};\n```" },
        CommandMeta { id: 18, name: "RankFilter", category: CommandCategory::Preprocess, summary: "A filter that transforms pixels based on the statistical rank of their neighbors.", description: "Rank filters are non-linear operators used for noise reduction,\nmorphological operations, and feature enhancement.\n\nThis algorithm sorts (ranks) all pixel values within a local neighborhood\nwindow and assigns a specific percentile value to the center pixel. By selecting\ndifferent ranks, it acts as a configurable operator: the minimum rank performs\nerosion, the maximum rank performs dilation, and the median rank (50th percentile)\nprovides highly effective impulse noise suppression while preserving sharp structural edges." },
        CommandMeta { id: 19, name: "RollingBall", category: CommandCategory::Preprocess, summary: "Removes non-uniform background illumination by calculating a local intensity baseline.", description: "This algorithm models the image as a 3D intensity landscape and conceptually rolls\na sphere of a user-defined radius underneath it. The ball cannot penetrate narrow\nintensity peaks (true signal objects) but follows the sweeping, lower-frequency\ncurves of background variations. The path traced by the ball establishes a local\nbaseline map that is subtracted from the original image to isolate foreground features." },
        CommandMeta { id: 20, name: "SaveImage", category: CommandCategory::Preprocess, summary: "A command that exports the current image to a persistent file on disk.", description: "This is a **transparent command**: it does not modify the image data in the\npipeline context, nor does it perform a buffer swap. It acts as a tap\nto view the state of the image at a specific point in the pipeline.\n\n# Examples\n\n```\nuse imagec::backend::algos::SaveImage;\nlet saver = SaveImage {path:\"output/processed_cell.png\"};\n```" },
        CommandMeta { id: 21, name: "StructureTensor", category: CommandCategory::Preprocess, summary: "Analyzes local image texture, directional orientation, and corner features using a second-moment matrix.", description: "This algorithm summarizes the predominant directions of the image gradient within a local\nneighborhood, smoothing the structural data with a Gaussian window. By evaluating the\neigenvalues of the resulting matrix tensor, it distinguishes between flat areas (both eigenvalues\nnear zero), straight linear boundaries (one dominant eigenvalue indicating structural direction),\nand complex corners or intersections (two large eigenvalues).\n\n# Examples\n\n```\nuse imagec::backend::algos::{StructureTensor, Mode};\nlet settings = StructureTensor {\nmode: Mode::Coherence,\nkernel_size: 3,\nsigma: 1.5\n};\n```" },
        CommandMeta { id: 22, name: "Threshold", category: CommandCategory::Segment, summary: "A filter that segments an image into discrete classes based on intensity.", description: "This supports \"Multi-Otsu\" style behavior by allowing a vector of\n[`ThresholdSettings`]. Each pixel is evaluated against the settings to\ndetermine which `object_class_id` it belongs to.\n\n# Examples\n\n```\nuse imagec::backend::algos::{Threshold, ThresholdSettings, ThresholdMethod};\nlet binary = Threshold {\nthresholds: vec![ThresholdSettings {\nmethod: ThresholdMethod::Otsu,\nmin_threshold: 0.0,\nmax_threshold: 1.0,\nobject_class_id: ObjectLabel::Foreground,\n}]\n};\n```" },
        CommandMeta { id: 23, name: "Voronoi", category: CommandCategory::Classify, summary: "Computes a Voronoi tessellation from segmented seed objects.", description: "Each seed center expands outward until it reaches another region, the optional mask\nboundary, or the maximum radius. The resulting areas are stored as new ROIs labeled\nwith `output_class` and linked to their originating center object." },
        CommandMeta { id: 24, name: "Watershed", category: CommandCategory::Object, summary: "A morphological segmentation algorithm that splits touching objects using distance topography.", description: "The Watershed algorithm is a powerful tool for separating overlapping structures (like cells or grains).\nBy analyzing the \"shape\" of an object via a Distance Transform, it identifies centers of mass\nand establishes boundaries at the narrowest points of connection.\n\nThis implementation is adaptive:\n* It can **auto-detect** objects from grayscale intensity peaks.\n* It can **refine** existing segments if a `U32Label` image is provided as input." },
        CommandMeta { id: 25, name: "WeightedDeviation", category: CommandCategory::Preprocess, summary: "A filter that computes the Gaussian-weighted standard deviation of a local neighborhood.", description: "Unlike a standard deviation filter which treats all pixels in a window equally,\nthe Weighted Deviation uses a Gaussian kernel to give more importance to\npixels closer to the center. This is particularly effective for edge-preserving\nnoise analysis and local contrast enhancement.\n\nThis algorithm evaluates local variance by calculating two distinct Gaussian-blurred\nbaselines across the image: the weighted average of the pixel intensities, and the\nweighted average of the squared intensities. By subtracting the squared mean from\nthe mean of squares, it yields a localized, smooth statistical variance map that\nhighlights micro-textures and subtle surface boundaries without producing blocky artifacts.\n\n# Examples\n\n```\nuse imagec::backend::algos::WeightedDeviation;\nlet settings = WeightedDeviation {\nkernel_size: 7,\nsigma: 2.0,\n};\n```" },
    ]
}

#[allow(dead_code)]
pub fn default_command(id: i32) -> Option<PipelineCommand> {
    match id {
        0 => Some(PipelineCommand::Blur(BlurSettings::default())),
        1 => Some(PipelineCommand::ClassifyRois(
            ClassifyRoisSettings::default(),
        )),
        2 => Some(PipelineCommand::Colocalization(
            ColocalizationSettings::default(),
        )),
        3 => Some(PipelineCommand::ColorFilterCommand(
            ColorFilterCommandSettings::default(),
        )),
        4 => Some(PipelineCommand::ConnectedComponents(
            ConnectedComponentsSettings::default(),
        )),
        5 => Some(PipelineCommand::DistanceTransform(
            DistanceTransformSettings::default(),
        )),
        6 => Some(PipelineCommand::EdgeDetectionCanny(
            EdgeDetectionCannySettings::default(),
        )),
        7 => Some(PipelineCommand::EdgeDetectionSobel(
            EdgeDetectionSobelSettings::default(),
        )),
        8 => Some(PipelineCommand::EnhanceContrast(
            EnhanceContrastSettings::default(),
        )),
        9 => Some(PipelineCommand::ExtractRois(ExtractRoisSettings::default())),
        10 => Some(PipelineCommand::GaussianBlur(
            GaussianBlurSettings::default(),
        )),
        11 => Some(PipelineCommand::Hessian(HessianSettings::default())),
        12 => Some(PipelineCommand::ImageCache(ImageCacheSettings::default())),
        13 => Some(PipelineCommand::ImageMath(ImageMathSettings::default())),
        14 => Some(PipelineCommand::IntensityTransformation(
            IntensityTransformationSettings::default(),
        )),
        15 => Some(PipelineCommand::Laplacian(LaplacianSettings::default())),
        16 => Some(PipelineCommand::MedianSubtract(
            MedianSubtractSettings::default(),
        )),
        17 => Some(PipelineCommand::MorphologicalCommand(
            MorphologicalCommandSettings::default(),
        )),
        18 => Some(PipelineCommand::RankFilter(RankFilterSettings::default())),
        19 => Some(PipelineCommand::RollingBall(RollingBallSettings::default())),
        20 => Some(PipelineCommand::SaveImage(SaveImageSettings::default())),
        21 => Some(PipelineCommand::StructureTensor(
            StructureTensorSettings::default(),
        )),
        22 => Some(PipelineCommand::Threshold(ThresholdSettings::default())),
        23 => Some(PipelineCommand::Voronoi(VoronoiSettings::default())),
        24 => Some(PipelineCommand::Watershed(WatershedSettings::default())),
        25 => Some(PipelineCommand::WeightedDeviation(
            WeightedDeviationSettings::default(),
        )),
        _ => None,
    }
}

#[allow(dead_code)]
impl PipelineCommand {
    pub fn name(&self) -> &str {
        match self {
            Self::Blur(_) => "Blur",
            Self::ClassifyRois(_) => "ClassifyRois",
            Self::Colocalization(_) => "Colocalization",
            Self::ColorFilterCommand(_) => "ColorFilterCommand",
            Self::ConnectedComponents(_) => "ConnectedComponents",
            Self::DistanceTransform(_) => "DistanceTransform",
            Self::EdgeDetectionCanny(_) => "EdgeDetectionCanny",
            Self::EdgeDetectionSobel(_) => "EdgeDetectionSobel",
            Self::EnhanceContrast(_) => "EnhanceContrast",
            Self::ExtractRois(_) => "ExtractRois",
            Self::GaussianBlur(_) => "GaussianBlur",
            Self::Hessian(_) => "Hessian",
            Self::ImageCache(_) => "ImageCache",
            Self::ImageMath(_) => "ImageMath",
            Self::IntensityTransformation(_) => "IntensityTransformation",
            Self::Laplacian(_) => "Laplacian",
            Self::MedianSubtract(_) => "MedianSubtract",
            Self::MorphologicalCommand(_) => "MorphologicalCommand",
            Self::RankFilter(_) => "RankFilter",
            Self::RollingBall(_) => "RollingBall",
            Self::SaveImage(_) => "SaveImage",
            Self::StructureTensor(_) => "StructureTensor",
            Self::Threshold(_) => "Threshold",
            Self::Voronoi(_) => "Voronoi",
            Self::Watershed(_) => "Watershed",
            Self::WeightedDeviation(_) => "WeightedDeviation",
        }
    }

    pub fn category(&self) -> &CommandCategory {
        match self {
            Self::Blur(_) => &CommandCategory::Preprocess,
            Self::ClassifyRois(_) => &CommandCategory::Classify,
            Self::Colocalization(_) => &CommandCategory::Classify,
            Self::ColorFilterCommand(_) => &CommandCategory::Preprocess,
            Self::ConnectedComponents(_) => &CommandCategory::Object,
            Self::DistanceTransform(_) => &CommandCategory::Preprocess,
            Self::EdgeDetectionCanny(_) => &CommandCategory::Preprocess,
            Self::EdgeDetectionSobel(_) => &CommandCategory::Preprocess,
            Self::EnhanceContrast(_) => &CommandCategory::Preprocess,
            Self::ExtractRois(_) => &CommandCategory::Measure,
            Self::GaussianBlur(_) => &CommandCategory::Preprocess,
            Self::Hessian(_) => &CommandCategory::Preprocess,
            Self::ImageCache(_) => &CommandCategory::Preprocess,
            Self::ImageMath(_) => &CommandCategory::Preprocess,
            Self::IntensityTransformation(_) => &CommandCategory::Preprocess,
            Self::Laplacian(_) => &CommandCategory::Preprocess,
            Self::MedianSubtract(_) => &CommandCategory::Preprocess,
            Self::MorphologicalCommand(_) => &CommandCategory::Preprocess,
            Self::RankFilter(_) => &CommandCategory::Preprocess,
            Self::RollingBall(_) => &CommandCategory::Preprocess,
            Self::SaveImage(_) => &CommandCategory::Preprocess,
            Self::StructureTensor(_) => &CommandCategory::Preprocess,
            Self::Threshold(_) => &CommandCategory::Segment,
            Self::Voronoi(_) => &CommandCategory::Classify,
            Self::Watershed(_) => &CommandCategory::Object,
            Self::WeightedDeviation(_) => &CommandCategory::Preprocess,
        }
    }

    pub fn to_parameters(&self) -> Vec<ParameterDef> {
        match self {
            Self::Blur(_s) => vec![
                ParameterDef { name: "kernel_size".to_string(), display_name: "Kernel size".to_string(), description: "The size of the blur matrix.\n\nMust be an odd number (e.g., 3, 5, 7)".to_string(), value: format!("{}", _s.kernel_size), param_type: ParamType::Spinner, options: vec![], min: 3.0f32, max: 27.0f32, step: 2.0000f32, groups: vec![] },
            ],
            Self::ClassifyRois(_s) => vec![
                ParameterDef { name: "input_classes".to_string(), display_name: "Input Classes".to_string(), description: "Restrict classification to objects that already carry one of these classes\n\nOnly ROIs that have been assigned at least one of the listed classes by a prior\npipeline step will be evaluated against the morphological and intensity criteria below.\nLeave empty to apply the criteria to every object regardless of its current class.".to_string(), value: _s.input_classes.iter().filter_map(|c| c.to_u32()).map(|v| v.to_string()).collect::<Vec<_>>().join(","), param_type: ParamType::MultiObjClass, options: (0u32..33u32).map(|__idx| if _s.input_classes.iter().any(|c| c.to_u32().map_or(false, |v| v == __idx)) { "1".to_string() } else { "0".to_string() }).collect::<Vec<_>>(), min: 0.0f32, max: 0.0f32, step: 1.0000f32, groups: vec![] },
                ParameterDef { name: "match_handling".to_string(), display_name: "Match Handling".to_string(), description: "What to do with object class labels after criteria evaluation\n\nControls whether the output class is added or existing classes are removed,\nand whether the action is triggered on a criteria **match** or a **non-match**:\n\n- **AddOutputClassIfMatch** - append the output class to objects that pass the criteria.\n- **AddOutputClassIfNotMatch** - append the output class to objects that fail the criteria.\n- **RemoveInputClassIfMatch / NotMatch** - strip all input classes from matching / non-matching objects.\n- **RemoveOutputClassIfMatch / NotMatch** - strip the output class from matching / non-matching objects.\n- **RemoveAllClassesIfMatch / NotMatch** - clear every class label from matching / non-matching objects.".to_string(), value: format!("{:?}", _s.match_handling), param_type: ParamType::Dropdown, options: vec!["AddOutputClassIfMatch".to_string(), "AddOutputClassIfNotMatch".to_string(), "RemoveInputClassIfMatch".to_string(), "RemoveInputClassIfNotMatch".to_string(), "RemoveOutputClassIfMatch".to_string(), "RemoveOutputClassIfNotMatch".to_string(), "RemoveAllClassesIfMatch".to_string(), "RemoveAllClassesIfNotMatch".to_string()], min: 0.0f32, max: 0.0f32, step: 1.0000f32, groups: vec![] },
                ParameterDef { name: "output_class".to_string(), display_name: "Output Class".to_string(), description: "Class label assigned to (or removed from) objects by the chosen operation\n\nUsed as the target class for `AddOutputClass*` and `RemoveOutputClass*` operations.\nHas no effect when the selected operation only manipulates input classes or clears all classes.".to_string(), value: match _s.output_class.to_u32() { Some(v) => format!("{}", v), None => "-1".to_string() }, param_type: ParamType::ObjClass, options: vec![], min: 0.0f32, max: 0.0f32, step: 1.0000f32, groups: vec![] },
                ParameterDef { name: "size_unit".to_string(), display_name: "Size Unit".to_string(), description: "Unit to use for roi extraction".to_string(), value: match _s.size_unit { SizeUnits::NanoMeter => "nm".to_string(), SizeUnits::Pixels => "px".to_string() }, param_type: ParamType::SizeUnits, options: vec!["nm".to_string(), "px".to_string()], min: 0.0f32, max: 0.0f32, step: 1.0000f32, groups: vec![] },
                ParameterDef { name: "min_area".to_string(), display_name: "Min Area".to_string(), description: "Minimum area size\n\nMinimum area size of the object in selected unit (px^2 or nm^2).".to_string(), value: format!("{}", _s.min_area), param_type: ParamType::Spinner, options: vec![], min: 0.0f32, max: 2147483648.0f32, step: 1.0000f32, groups: vec![] },
                ParameterDef { name: "max_area".to_string(), display_name: "Max Area".to_string(), description: "Maximum area size\n\nMaximum area size of the object in selected unit (px^2 or nm^2).".to_string(), value: format!("{}", _s.max_area), param_type: ParamType::Spinner, options: vec![], min: 0.0f32, max: 2147483648.0f32, step: 1.0000f32, groups: vec![] },
                ParameterDef { name: "min_circularity".to_string(), display_name: "Min Circularity".to_string(), description: "Circularity range: 0 = elongated, 1 = perfect circle\n\nCircularity (sometimes called Isoperimetric Quotient) measures how efficiently a shape encloses its area relative to the length of its perimeter.\nA circle is the mathematically perfect shape for maximizing area while minimizing perimeter.\nIt is calculated with `4*Pi*AreaSize / Perimeter^2`".to_string(), value: format!("{}", _s.min_circularity), param_type: ParamType::Spinner, options: vec![], min: 0.0f32, max: 1.0f32, step: 0.1000f32, groups: vec![] },
                ParameterDef { name: "max_circularity".to_string(), display_name: "Max Circularity".to_string(), description: "Circularity range: 0 = elongated, 1 = perfect circle\n\nCircularity (sometimes called Isoperimetric Quotient) measures how efficiently a shape encloses its area relative to the length of its perimeter.\nA circle is the mathematically perfect shape for maximizing area while minimizing perimeter.\nIt is calculated with `4*Pi*AreaSize / Perimeter^2`".to_string(), value: format!("{}", _s.max_circularity), param_type: ParamType::Spinner, options: vec![], min: 0.0f32, max: 1.0f32, step: 0.1000f32, groups: vec![] },
                ParameterDef { name: "min_solidity".to_string(), display_name: "Min Solidity".to_string(), description: "Minimum Solidity/Compactness: 0 = hollow, 1 = perfect convex\n\nSolidity is a structural metric used in shape analysis to measure how \"solid\" or compact an object is.\nIt compares the actual area of an object to the area of its Convex Hull (the smallest convex polygon that can completely enclose the object,\noften visualized as a rubber band stretched around the shape).\n\nSolidity = 1.0: The object is perfectly convex (e.g., a perfect circle, a solid square, or an ellipse). It has no holes, indentations, or deep recesses.\nSolidity < 1.0: The object has irregular boundaries, deep \"bays,\" protrusions, or internal holes. The lower the value, the more jagged or structurally fragmented the object is.".to_string(), value: format!("{}", _s.min_solidity), param_type: ParamType::Spinner, options: vec![], min: 0.0f32, max: 1.0f32, step: 0.1000f32, groups: vec![] },
                ParameterDef { name: "max_solidity".to_string(), display_name: "Max Solidity".to_string(), description: "Maximum Solidity/Compactness: 0 = hollow, 1 = perfect convex\n\nSolidity is a structural metric used in shape analysis to measure how \"solid\" or compact an object is.\nIt compares the actual area of an object to the area of its Convex Hull (the smallest convex polygon that can completely enclose the object,\noften visualized as a rubber band stretched around the shape).\n\nSolidity = 1.0: The object is perfectly convex (e.g., a perfect circle, a solid square, or an ellipse). It has no holes, indentations, or deep recesses.\nSolidity < 1.0: The object has irregular boundaries, deep \"bays,\" protrusions, or internal holes. The lower the value, the more jagged or structurally fragmented the object is.".to_string(), value: format!("{}", _s.max_solidity), param_type: ParamType::Spinner, options: vec![], min: 0.0f32, max: 1.0f32, step: 0.1000f32, groups: vec![] },
                ParameterDef { name: "min_aspect_ratio".to_string(), display_name: "Min Aspect Ratio".to_string(), description: "Minimum proportional relationship between an object's width and its height\n\nThis value is calculated by the object bounding box with and height and is defined with `a = with/height`.\nThe value is without unit in the range of 0 to MAX_F32".to_string(), value: format!("{}", _s.min_aspect_ratio), param_type: ParamType::Spinner, options: vec![], min: 0.0f32, max: 2147483648.0f32, step: 1.0000f32, groups: vec![] },
                ParameterDef { name: "max_aspect_ratio".to_string(), display_name: "Max Aspect Ratio".to_string(), description: "Maximum proportional relationship between an object's width and its height\n\nThis value is calculated by the object bounding box with and height and is defined with `a = with/height`.\nThe value is without unit in the range of 0 to MAX_F32".to_string(), value: format!("{}", _s.max_aspect_ratio), param_type: ParamType::Spinner, options: vec![], min: 0.0f32, max: 2147483648.0f32, step: 1.0000f32, groups: vec![] },
                ParameterDef { name: "min_eccentricity".to_string(), display_name: "Min Eccentricity".to_string(), description: "Eccentricity: 0 = perfect circle, 1 = line\n\nEccentricity is a metric that measures how much a shape deviates from being a perfect circle.\nIt imagines the shape as an ellipse and measures how far apart its focal points are.\nIt is calculated with `sqrt(1-(b/a)^2)`".to_string(), value: format!("{}", _s.min_eccentricity), param_type: ParamType::Spinner, options: vec![], min: 0.0f32, max: 1.0f32, step: 0.1000f32, groups: vec![] },
                ParameterDef { name: "max_eccentricity".to_string(), display_name: "Max Eccentricity".to_string(), description: "Eccentricity: 0 = perfect circle, 1 = line\n\nEccentricity is a metric that measures how much a shape deviates from being a perfect circle.\nIt imagines the shape as an ellipse and measures how far apart its focal points are.\nIt is calculated with `sqrt(1-(b/a)^2)`".to_string(), value: format!("{}", _s.max_eccentricity), param_type: ParamType::Spinner, options: vec![], min: 0.0f32, max: 1.0f32, step: 0.1000f32, groups: vec![] },
                ParameterDef { name: "min_feret".to_string(), display_name: "Min Feret".to_string(), description: "Feret diameter threshold\n\nThe absolute shortest parallel distance across the object.\nThis represents the minimum sieve size a particle could pass through.\n\nIn image processing and particle size analysis, the Feret diameter (often called the caliper diameter) is a metric used to measure the size of an irregular object.\nIt mimics the action of a slide caliper, measuring the distance between two parallel tangential lines bounding the object at a specific angle.\nWhen analyzing objects or particles, applying Feret diameter thresholds allows you to filter out noise, classify objects by shape, or isolate specific structures based on their directional length rather than their total area.".to_string(), value: format!("{}", _s.min_feret), param_type: ParamType::Spinner, options: vec![], min: 0.0f32, max: 2147483648.0f32, step: 1.0000f32, groups: vec![] },
                ParameterDef { name: "max_feret".to_string(), display_name: "Max Feret".to_string(), description: "Maximum feret diameter threshold in selected unit (px or nm)\n\nThe absolute longest distance across the object at any angle.\nUsed to measure elongation or the maximum length of a particle.\n\nIn image processing and particle size analysis, the Feret diameter (often called the caliper diameter) is a metric used to measure the size of an irregular object.\nIt mimics the action of a slide caliper, measuring the distance between two parallel tangential lines bounding the object at a specific angle.\nWhen analyzing objects or particles, applying Feret diameter thresholds allows you to filter out noise, classify objects by shape, or isolate specific structures based on their directional length rather than their total area.".to_string(), value: format!("{}", _s.max_feret), param_type: ParamType::Spinner, options: vec![], min: 0.0f32, max: 2147483648.0f32, step: 1.0000f32, groups: vec![] },
                ParameterDef { name: "allow_edge_touching".to_string(), display_name: "Allow Edge Touching".to_string(), description: "Whether ROI can touch image edge".to_string(), value: format!("{}", _s.allow_edge_touching), param_type: ParamType::Toggle, options: vec![], min: 0.0f32, max: 0.0f32, step: 1.0000f32, groups: vec![] },
            ],
            Self::Colocalization(_s) => vec![
                ParameterDef { name: "classes_to_coloc".to_string(), display_name: "Classes To Coloc".to_string(), description: "Theses are the classes the coloclization should be calculated for".to_string(), value: _s.classes_to_coloc.iter().filter_map(|c| c.to_u32()).map(|v| v.to_string()).collect::<Vec<_>>().join(","), param_type: ParamType::MultiObjClass, options: (0u32..33u32).map(|__idx| if _s.classes_to_coloc.iter().any(|c| c.to_u32().map_or(false, |v| v == __idx)) { "1".to_string() } else { "0".to_string() }).collect::<Vec<_>>(), min: 0.0f32, max: 0.0f32, step: 1.0000f32, groups: vec![] },
                ParameterDef { name: "filter_classes".to_string(), display_name: "Filter Classes".to_string(), description: "Optional additional label filters.\n\nOnly classes which matches all of these filters are used for coloc calculation".to_string(), value: _s.filter_classes.iter().filter_map(|c| c.to_u32()).map(|v| v.to_string()).collect::<Vec<_>>().join(","), param_type: ParamType::MultiObjClass, options: (0u32..33u32).map(|__idx| if _s.filter_classes.iter().any(|c| c.to_u32().map_or(false, |v| v == __idx)) { "1".to_string() } else { "0".to_string() }).collect::<Vec<_>>(), min: 0.0f32, max: 0.0f32, step: 1.0000f32, groups: vec![] },
                ParameterDef { name: "class_for_overlapping_areas".to_string(), display_name: "Class For Overlapping Areas".to_string(), description: "Class of the overlapping area if needed\n\nIf defined the overlapping coloc area is added as new ROI and labeled with this class".to_string(), value: match _s.class_for_overlapping_areas.to_u32() { Some(v) => format!("{}", v), None => "-1".to_string() }, param_type: ParamType::ObjClass, options: vec![], min: 0.0f32, max: 0.0f32, step: 1.0000f32, groups: vec![] },
                ParameterDef { name: "allow_multi_object_coloc".to_string(), display_name: "Allow Multi Object Coloc".to_string(), description: "If set one object is allowed to coloc with more than one other object".to_string(), value: format!("{}", _s.allow_multi_object_coloc), param_type: ParamType::Toggle, options: vec![], min: 0.0f32, max: 0.0f32, step: 1.0000f32, groups: vec![] },
                ParameterDef { name: "size_unit".to_string(), display_name: "Size Unit".to_string(), description: "".to_string(), value: match _s.size_unit { SizeUnits::NanoMeter => "nm".to_string(), SizeUnits::Pixels => "px".to_string() }, param_type: ParamType::SizeUnits, options: vec!["nm".to_string(), "px".to_string()], min: 0.0f32, max: 0.0f32, step: 1.0000f32, groups: vec![] },
                ParameterDef { name: "min_coloc_area".to_string(), display_name: "Min Coloc Area".to_string(), description: "Minimum overlapping area size to count objects as coloc".to_string(), value: format!("{}", _s.min_coloc_area), param_type: ParamType::Number, options: vec![], min: 0.0f32, max: 0.0f32, step: 1.0000f32, groups: vec![] },
            ],
            Self::ColorFilterCommand(_s) => vec![
                ParameterDef { name: "range.min_h".to_string(), display_name: "Min H".to_string(), description: "Minimum Hue angle in degrees [0.0, 360.0].".to_string(), value: format!("{}", _s.range.min_h), param_type: ParamType::Number, options: vec![], min: 0.0f32, max: 0.0f32, step: 1.0000f32, groups: vec![] },
                ParameterDef { name: "range.max_h".to_string(), display_name: "Max H".to_string(), description: "Maximum Hue angle in degrees [0.0, 360.0].".to_string(), value: format!("{}", _s.range.max_h), param_type: ParamType::Number, options: vec![], min: 0.0f32, max: 0.0f32, step: 1.0000f32, groups: vec![] },
                ParameterDef { name: "range.min_s".to_string(), display_name: "Min S".to_string(), description: "Minimum Saturation normalized [0.0, 1.0].".to_string(), value: format!("{}", _s.range.min_s), param_type: ParamType::Number, options: vec![], min: 0.0f32, max: 0.0f32, step: 1.0000f32, groups: vec![] },
                ParameterDef { name: "range.max_s".to_string(), display_name: "Max S".to_string(), description: "Maximum Saturation normalized [0.0, 1.0].".to_string(), value: format!("{}", _s.range.max_s), param_type: ParamType::Number, options: vec![], min: 0.0f32, max: 0.0f32, step: 1.0000f32, groups: vec![] },
                ParameterDef { name: "range.min_v".to_string(), display_name: "Min V".to_string(), description: "Minimum Value (Brightness) normalized [0.0, 1.0].".to_string(), value: format!("{}", _s.range.min_v), param_type: ParamType::Number, options: vec![], min: 0.0f32, max: 0.0f32, step: 1.0000f32, groups: vec![] },
                ParameterDef { name: "range.max_v".to_string(), display_name: "Max V".to_string(), description: "Maximum Value (Brightness) normalized [0.0, 1.0].".to_string(), value: format!("{}", _s.range.max_v), param_type: ParamType::Number, options: vec![], min: 0.0f32, max: 0.0f32, step: 1.0000f32, groups: vec![] },
            ],
            Self::ConnectedComponents(_s) => vec![
            ],
            Self::DistanceTransform(_s) => vec![
                ParameterDef { name: "threshold".to_string(), display_name: "Threshold".to_string(), description: "Values less than or equal to this are treated as background (distance = 0).".to_string(), value: format!("{}", _s.threshold), param_type: ParamType::Number, options: vec![], min: 0.0f32, max: 0.0f32, step: 1.0000f32, groups: vec![] },
                ParameterDef { name: "edges_are_background".to_string(), display_name: "Edges Are Background".to_string(), description: "If true, the pixels outside the image boundary are treated as background.".to_string(), value: format!("{}", _s.edges_are_background), param_type: ParamType::Toggle, options: vec![], min: 0.0f32, max: 0.0f32, step: 1.0000f32, groups: vec![] },
            ],
            Self::EdgeDetectionCanny(_s) => vec![
                ParameterDef { name: "kernel_size".to_string(), display_name: "Kernel Size".to_string(), description: "Size of the Gaussian smoothing kernel.\n\nMust be an odd number (e.g., 3, 5). Larger values reduce\nnoise but can blur fine edge details.".to_string(), value: format!("{}", _s.kernel_size), param_type: ParamType::Number, options: vec![], min: 0.0f32, max: 0.0f32, step: 1.0000f32, groups: vec![] },
                ParameterDef { name: "threshold_min".to_string(), display_name: "Threshold Min".to_string(), description: "Lower bound for hysteresis thresholding [0.0, 1.0].\n\nEdges with a gradient intensity below this value are discarded.".to_string(), value: format!("{}", _s.threshold_min), param_type: ParamType::Number, options: vec![], min: 0.0f32, max: 0.0f32, step: 1.0000f32, groups: vec![] },
                ParameterDef { name: "threshold_max".to_string(), display_name: "Threshold Max".to_string(), description: "Upper bound for hysteresis thresholding [0.0, 1.0].\n\nEdges with a gradient intensity above this value are considered\n\"strong\" and are automatically preserved.".to_string(), value: format!("{}", _s.threshold_max), param_type: ParamType::Number, options: vec![], min: 0.0f32, max: 0.0f32, step: 1.0000f32, groups: vec![] },
            ],
            Self::EdgeDetectionSobel(_s) => vec![
                ParameterDef { name: "kernel_size".to_string(), display_name: "Kernel Size".to_string(), description: "The size of the Sobel operator window.\n\nTypically 3. Larger values (5, 7) provide a more smoothed\ngradient but result in \"thicker\" edges. Must be an odd number.".to_string(), value: format!("{}", _s.kernel_size), param_type: ParamType::Number, options: vec![], min: 0.0f32, max: 0.0f32, step: 1.0000f32, groups: vec![] },
            ],
            Self::EnhanceContrast(_s) => vec![
                ParameterDef { name: "saturated_pixels".to_string(), display_name: "Saturated Pixels".to_string(), description: "Percentage of pixels to \"clip\" from the top and bottom of the histogram.\n\nRange: [0.0, 1.0]. A value of 0.01 (1%) helps ignore hot/dead pixels\nthat would otherwise prevent effective contrast stretching.".to_string(), value: format!("{}", _s.saturated_pixels), param_type: ParamType::Number, options: vec![], min: 0.0f32, max: 0.0f32, step: 1.0000f32, groups: vec![] },
                ParameterDef { name: "normalize".to_string(), display_name: "Normalize".to_string(), description: "Whether to linearly stretch the remaining pixel intensities to fill\nthe full [0.0, 1.0] range.".to_string(), value: format!("{}", _s.normalize), param_type: ParamType::Toggle, options: vec![], min: 0.0f32, max: 0.0f32, step: 1.0000f32, groups: vec![] },
                ParameterDef { name: "equalize_histogram".to_string(), display_name: "Equalize Histogram".to_string(), description: "Whether to apply Histogram Equalization.\n\nThis redistributes pixel intensities to achieve a uniform distribution,\nwhich is highly effective for images with low contrast but high noise.".to_string(), value: format!("{}", _s.equalize_histogram), param_type: ParamType::Toggle, options: vec![], min: 0.0f32, max: 0.0f32, step: 1.0000f32, groups: vec![] },
            ],
            Self::ExtractRois(_s) => vec![
                ParameterDef { name: "max_objects_before_fail".to_string(), display_name: "Max Objects Before Fail".to_string(), description: "Maximum allowed ROIs to extract.\n\nIf this limit is exceeded the pipeline fails.\nThis is a protection against memory overload.".to_string(), value: format!("{}", _s.max_objects_before_fail), param_type: ParamType::Label, options: vec![], min: 0.0f32, max: 0.0f32, step: 1.0000f32, groups: vec![] },
            ],
            Self::GaussianBlur(_s) => vec![
                ParameterDef { name: "kernel_size".to_string(), display_name: "Kernel Size".to_string(), description: "The size of the blur matrix.\n\nMust be an odd number (e.g., 3, 5, 7).".to_string(), value: format!("{}", _s.kernel_size), param_type: ParamType::Spinner, options: vec![], min: 3.0f32, max: 27.0f32, step: 2.0000f32, groups: vec![] },
                ParameterDef { name: "sigma".to_string(), display_name: "Sigma".to_string(), description: "The standard deviation of the Gaussian kernel.\n\nHigher values create a more significant blur effect.\n$$N \\approx 6\\sigma + 1$$".to_string(), value: format!("{}", _s.sigma), param_type: ParamType::Spinner, options: vec![], min: 0.1f32, max: 5.0f32, step: 0.1000f32, groups: vec![] },
            ],
            Self::Hessian(_s) => vec![
                ParameterDef { name: "mode".to_string(), display_name: "Mode".to_string(), description: "Determines which component of the Hessian matrix structure to extract.\n\nDepending on the mode, this can highlight interest points (blobs)\nor directional features (ridges).".to_string(), value: format!("{:?}", _s.mode), param_type: ParamType::Dropdown, options: vec!["Determinant".to_string(), "EigenvaluesX".to_string(), "EigenvaluesY".to_string()], min: 0.0f32, max: 0.0f32, step: 1.0000f32, groups: vec![] },
            ],
            Self::ImageCache(_s) => vec![
                ParameterDef { name: "mode".to_string(), display_name: "Mode".to_string(), description: "Whether to save the current state to the cache or load a state from it.".to_string(), value: format!("{:?}", _s.mode), param_type: ParamType::Dropdown, options: vec!["Store".to_string(), "Load".to_string()], min: 0.0f32, max: 0.0f32, step: 1.0000f32, groups: vec![] },
            ],
            Self::ImageMath(_s) => vec![
                ParameterDef { name: "operand".to_string(), display_name: "Operand".to_string(), description: "The specific mathematical or logical operator to apply.".to_string(), value: format!("{:?}", _s.operand), param_type: ParamType::Dropdown, options: vec!["None".to_string(), "Invert".to_string(), "Add".to_string(), "Subtract".to_string(), "Multiply".to_string(), "Divide".to_string(), "AND".to_string(), "OR".to_string(), "XOR".to_string(), "MIN".to_string(), "MAX".to_string(), "Average".to_string(), "DifferenceType".to_string()], min: 0.0f32, max: 0.0f32, step: 1.0000f32, groups: vec![] },
                ParameterDef { name: "swap_operands".to_string(), display_name: "Swap Operands".to_string(), description: "If false, the calculation is `(Current Image OP Cached Image)`.\nIf true, the calculation is `(Cached Image OP Current Image)`.\n\nThis is critical for non-commutative operations like Subtraction or Division.".to_string(), value: format!("{}", _s.swap_operands), param_type: ParamType::Toggle, options: vec![], min: 0.0f32, max: 0.0f32, step: 1.0000f32, groups: vec![] },
            ],
            Self::IntensityTransformation(_s) => vec![
                ParameterDef { name: "mode".to_string(), display_name: "Mode".to_string(), description: "Determines whether to use automated enhancement or user-defined values.".to_string(), value: format!("{:?}", _s.mode), param_type: ParamType::Dropdown, options: vec!["Automatic".to_string(), "Manual".to_string()], min: 0.0f32, max: 0.0f32, step: 1.0000f32, groups: vec![] },
                ParameterDef { name: "contrast".to_string(), display_name: "Contrast".to_string(), description: "Contrast multiplier (gain).\n\nOnly active in [`Mode::Manual`].\nValues > 1.0 increase contrast, while values < 1.0 decrease it.".to_string(), value: format!("{}", _s.contrast), param_type: ParamType::Number, options: vec![], min: 0.0f32, max: 0.0f32, step: 1.0000f32, groups: vec![] },
                ParameterDef { name: "brightness".to_string(), display_name: "Brightness".to_string(), description: "Brightness offset (bias).\n\nOnly active in [`Mode::Manual`].\nPositive values brighten the image, negative values darken it.".to_string(), value: format!("{}", _s.brightness), param_type: ParamType::Number, options: vec![], min: 0.0f32, max: 0.0f32, step: 1.0000f32, groups: vec![] },
            ],
            Self::Laplacian(_s) => vec![
                ParameterDef { name: "kernel_size".to_string(), display_name: "Kernel Size".to_string(), description: "The size of the discrete Laplacian aperture.\n\nTypically 3. Larger sizes (5, 7) approximate the Laplacian of Gaussian (LoG)\nmore closely but are more computationally expensive. Must be an odd number.".to_string(), value: format!("{}", _s.kernel_size), param_type: ParamType::Number, options: vec![], min: 0.0f32, max: 0.0f32, step: 1.0000f32, groups: vec![] },
            ],
            Self::MedianSubtract(_s) => vec![
                ParameterDef { name: "radius".to_string(), display_name: "Radius".to_string(), description: "The radius of the neighborhood used to estimate the background.\n\nFeatures smaller than this radius will be preserved, while\nlarger structures will be treated as background and removed.".to_string(), value: format!("{}", _s.radius), param_type: ParamType::Number, options: vec![], min: 0.0f32, max: 0.0f32, step: 1.0000f32, groups: vec![] },
            ],
            Self::MorphologicalCommand(_s) => vec![
                ParameterDef { name: "op".to_string(), display_name: "Op".to_string(), description: "The transformation type (e.g., Dilate, Erode).".to_string(), value: format!("{:?}", _s.op), param_type: ParamType::Dropdown, options: vec!["Dilate".to_string(), "Erode".to_string(), "Open".to_string(), "Close".to_string()], min: 0.0f32, max: 0.0f32, step: 1.0000f32, groups: vec![] },
                ParameterDef { name: "kernel_size".to_string(), display_name: "Kernel Size".to_string(), description: "The diameter of the structuring element in pixels.\nMust be an odd number (e.g., 3, 5, 7).".to_string(), value: format!("{}", _s.kernel_size), param_type: ParamType::Number, options: vec![], min: 0.0f32, max: 0.0f32, step: 1.0000f32, groups: vec![] },
                ParameterDef { name: "kernel_shape".to_string(), display_name: "Kernel Shape".to_string(), description: "The geometric profile of the structuring element.".to_string(), value: format!("{:?}", _s.kernel_shape), param_type: ParamType::Dropdown, options: vec!["Box".to_string(), "Ellipse".to_string(), "Cross".to_string()], min: 0.0f32, max: 0.0f32, step: 1.0000f32, groups: vec![] },
                ParameterDef { name: "use_grayscale".to_string(), display_name: "Use Grayscale".to_string(), description: "If set the grayscale image instead of the labeld image is taken to perform a morphological transform".to_string(), value: format!("{}", _s.use_grayscale), param_type: ParamType::Toggle, options: vec![], min: 0.0f32, max: 0.0f32, step: 1.0000f32, groups: vec![] },
            ],
            Self::RankFilter(_s) => vec![
                ParameterDef { name: "radius".to_string(), display_name: "Radius".to_string(), description: "The circular radius of the neighborhood to consider.\n\nA radius of 1.0 roughly corresponds to a 3x3 square, while larger\nvalues increase the effect's strength and computational cost.".to_string(), value: format!("{}", _s.radius), param_type: ParamType::Number, options: vec![], min: 0.0f32, max: 0.0f32, step: 1.0000f32, groups: vec![] },
                ParameterDef { name: "filter_type".to_string(), display_name: "Filter Type".to_string(), description: "The specific ranking algorithm to apply to the neighborhood.".to_string(), value: format!("{:?}", _s.filter_type), param_type: ParamType::Dropdown, options: vec!["Median".to_string(), "Min".to_string(), "Max".to_string(), "Mean".to_string(), "Outliers".to_string()], min: 0.0f32, max: 0.0f32, step: 1.0000f32, groups: vec![] },
            ],
            Self::RollingBall(_s) => vec![
                ParameterDef { name: "radius".to_string(), display_name: "Radius".to_string(), description: "The radius of the ball or paraboloid in pixels.\n\nThis should be at least as large as the radius of the largest\nobject in the image that is not part of the background.".to_string(), value: format!("{}", _s.radius), param_type: ParamType::Spinner, options: vec![], min: 1.0f32, max: 64.0f32, step: 1.0000f32, groups: vec![] },
                ParameterDef { name: "ball_type".to_string(), display_name: "Ball Type".to_string(), description: "The geometric shape of the rolling structural element.".to_string(), value: format!("{:?}", _s.ball_type), param_type: ParamType::Dropdown, options: vec!["Ball".to_string(), "Paraboloid".to_string()], min: 0.0f32, max: 0.0f32, step: 1.0000f32, groups: vec![] },
                ParameterDef { name: "pre_smooth".to_string(), display_name: "Pre Smooth".to_string(), description: "".to_string(), value: format!("{}", _s.pre_smooth), param_type: ParamType::Toggle, options: vec![], min: 0.0f32, max: 0.0f32, step: 1.0000f32, groups: vec![] },
            ],
            Self::SaveImage(_s) => vec![
                ParameterDef { name: "path".to_string(), display_name: "Path".to_string(), description: "The destination filesystem path where the image will be written.".to_string(), value: _s.path.display().to_string(), param_type: ParamType::Text, options: vec![], min: 0.0f32, max: 0.0f32, step: 1.0000f32, groups: vec![] },
                ParameterDef { name: "source".to_string(), display_name: "Source".to_string(), description: "".to_string(), value: format!("{:?}", _s.source), param_type: ParamType::Dropdown, options: vec!["Image".to_string(), "InstanceMap".to_string(), "SegmentationMask".to_string()], min: 0.0f32, max: 0.0f32, step: 1.0000f32, groups: vec![] },
            ],
            Self::StructureTensor(_s) => vec![
                ParameterDef { name: "mode".to_string(), display_name: "Mode".to_string(), description: "The mathematical output to be produced by the algorithm.".to_string(), value: format!("{:?}", _s.mode), param_type: ParamType::Dropdown, options: vec!["EigenvaluesX".to_string(), "EigenvaluesY".to_string(), "Coherence".to_string()], min: 0.0f32, max: 0.0f32, step: 1.0000f32, groups: vec![] },
                ParameterDef { name: "kernel_size".to_string(), display_name: "Kernel Size".to_string(), description: "The size of the integration window used to average the local gradients.\n\nLarger windows provide more stability against noise but reduce\nspatial resolution.".to_string(), value: format!("{}", _s.kernel_size), param_type: ParamType::Number, options: vec![], min: 0.0f32, max: 0.0f32, step: 1.0000f32, groups: vec![] },
                ParameterDef { name: "sigma".to_string(), display_name: "Sigma".to_string(), description: "The standard deviation for the Gaussian weighting of the integration window.\n\nControls the spatial \"reach\" of the neighborhood analysis.".to_string(), value: format!("{}", _s.sigma), param_type: ParamType::Number, options: vec![], min: 0.0f32, max: 0.0f32, step: 1.0000f32, groups: vec![] },
            ],
            Self::Threshold(_s) => vec![
                ParameterDef { name: "thresholds".to_string(), display_name: "Thresholds".to_string(), description: "A list of thresholding layers. Overlapping ranges are resolved\nby the order of the vector (last-in priority).".to_string(), value: String::new(), param_type: ParamType::Group, options: vec![], min: 0.0f32, max: 0.0f32, step: 1.0000f32, groups: _s.thresholds.iter().map(|__item| vec![
                    ParameterDef { name: "method".to_string(), display_name: "Method".to_string(), description: "The algorithm to use (Manual or Automatic).".to_string(), value: format!("{:?}", __item.method), param_type: ParamType::Dropdown, options: vec!["None".to_string(), "Manual".to_string(), "Li".to_string(), "MinError".to_string(), "Triangle".to_string(), "Moments".to_string(), "Huang".to_string(), "Intermodes".to_string(), "IsoData".to_string(), "MaxEntropy".to_string(), "Mean".to_string(), "Minimum".to_string(), "Otsu".to_string(), "Percentile".to_string(), "RenyiEntropy".to_string(), "Shanbhag".to_string(), "Yen".to_string()], min: 0.0f32, max: 0.0f32, step: 1.0000f32, groups: vec![] },
                    ParameterDef { name: "min_threshold".to_string(), display_name: "Min Threshold".to_string(), description: "The lower intensity bound. Used directly in `Manual` mode, or as a\nfloor for auto-methods.".to_string(), value: format!("{}", __item.min_threshold), param_type: ParamType::Spinner, options: vec![], min: 0.0f32, max: 65535.0f32, step: 1.0000f32, groups: vec![] },
                    ParameterDef { name: "max_threshold".to_string(), display_name: "Max Threshold".to_string(), description: "The upper intensity bound. Used directly in `Manual` mode, or as a\nceiling for auto-methods.".to_string(), value: format!("{}", __item.max_threshold), param_type: ParamType::Spinner, options: vec![], min: 0.0f32, max: 65535.0f32, step: 1.0000f32, groups: vec![] },
                    ParameterDef { name: "unit".to_string(), display_name: "Unit".to_string(), description: "Unit used for the threshold value.\n\nbit: 0 - 255/65535\n%: 0 - 100.0\nrel: 0 - 1.0".to_string(), value: match __item.unit { PixelUnits::Bit => "bit".to_string(), PixelUnits::Percent => "%".to_string(), PixelUnits::Relative => "rel".to_string() }, param_type: ParamType::PixelUnits, options: vec!["bit".to_string(), "%".to_string(), "rel".to_string()], min: 0.0f32, max: 0.0f32, step: 1.0000f32, groups: vec![] },
                    ParameterDef { name: "object_class_id".to_string(), display_name: "Object Class Id".to_string(), description: "The classification ID assigned to pixels falling within this threshold range.".to_string(), value: format!("{}", __item.object_class_id.as_u32()), param_type: ParamType::SegClass, options: vec![], min: 0.0f32, max: 0.0f32, step: 1.0000f32, groups: vec![] },
                    ]).collect() },
            ],
            Self::Voronoi(_s) => vec![
                ParameterDef { name: "centers".to_string(), display_name: "Centers".to_string(), description: "Object class whose instances act as Voronoi seed points.".to_string(), value: match _s.centers.to_u32() { Some(v) => format!("{}", v), None => "-1".to_string() }, param_type: ParamType::ObjClass, options: vec![], min: 0.0f32, max: 0.0f32, step: 1.0000f32, groups: vec![] },
                ParameterDef { name: "center_filter_classes".to_string(), display_name: "Center Filter Classes".to_string(), description: "Additional label filters applied to center objects before tessellation.\n\nOnly center objects that carry all listed classes pass the filter.\nLeave empty to include all objects of `centers`.".to_string(), value: _s.center_filter_classes.iter().filter_map(|c| c.to_u32()).map(|v| v.to_string()).collect::<Vec<_>>().join(","), param_type: ParamType::MultiObjClass, options: (0u32..33u32).map(|__idx| if _s.center_filter_classes.iter().any(|c| c.to_u32().map_or(false, |v| v == __idx)) { "1".to_string() } else { "0".to_string() }).collect::<Vec<_>>(), min: 0.0f32, max: 0.0f32, step: 1.0000f32, groups: vec![] },
                ParameterDef { name: "mask".to_string(), display_name: "Mask".to_string(), description: "Object class used to spatially constrain the Voronoi areas.\n\nEach computed Voronoi region is intersected with the union of all mask objects,\ndiscarding pixels that fall outside the mask. Set to `Unset` to expand\nto the full image boundary instead.".to_string(), value: match _s.mask.to_u32() { Some(v) => format!("{}", v), None => "-1".to_string() }, param_type: ParamType::ObjClass, options: vec![], min: 0.0f32, max: 0.0f32, step: 1.0000f32, groups: vec![] },
                ParameterDef { name: "mask_filter_classes".to_string(), display_name: "Mask Filter Classes".to_string(), description: "Additional label filters applied to mask objects.\n\nOnly mask objects that carry all listed classes pass the filter.\nLeave empty to include all objects of `mask`.".to_string(), value: _s.mask_filter_classes.iter().filter_map(|c| c.to_u32()).map(|v| v.to_string()).collect::<Vec<_>>().join(","), param_type: ParamType::MultiObjClass, options: (0u32..33u32).map(|__idx| if _s.mask_filter_classes.iter().any(|c| c.to_u32().map_or(false, |v| v == __idx)) { "1".to_string() } else { "0".to_string() }).collect::<Vec<_>>(), min: 0.0f32, max: 0.0f32, step: 1.0000f32, groups: vec![] },
                ParameterDef { name: "output_class".to_string(), display_name: "Output Class".to_string(), description: "Object class assigned to the resulting Voronoi region ROIs.".to_string(), value: match _s.output_class.to_u32() { Some(v) => format!("{}", v), None => "-1".to_string() }, param_type: ParamType::ObjClass, options: vec![], min: 0.0f32, max: 0.0f32, step: 1.0000f32, groups: vec![] },
                ParameterDef { name: "unit".to_string(), display_name: "Unit".to_string(), description: "Unit in which `max_radius` is expressed (e.g. pixels, nm, µm).".to_string(), value: match _s.unit { SizeUnits::NanoMeter => "nm".to_string(), SizeUnits::Pixels => "px".to_string() }, param_type: ParamType::SizeUnits, options: vec!["nm".to_string(), "px".to_string()], min: 0.0f32, max: 0.0f32, step: 1.0000f32, groups: vec![] },
                ParameterDef { name: "max_radius".to_string(), display_name: "Max Radius".to_string(), description: "Maximum expansion radius for a Voronoi region.\n\nPixels farther than this distance from the nearest seed center are excluded\nfrom the region. Use `0` or a negative value to disable the limit.".to_string(), value: format!("{}", _s.max_radius), param_type: ParamType::Number, options: vec![], min: 0.0f32, max: 0.0f32, step: 1.0000f32, groups: vec![] },
                ParameterDef { name: "exclude_areas_at_the_edges".to_string(), display_name: "Exclude Areas At The Edges".to_string(), description: "Discard Voronoi regions that touch the image border.".to_string(), value: format!("{}", _s.exclude_areas_at_the_edges), param_type: ParamType::Toggle, options: vec![], min: 0.0f32, max: 0.0f32, step: 1.0000f32, groups: vec![] },
                ParameterDef { name: "exclude_areas_with_no_center".to_string(), display_name: "Exclude Areas With No Center".to_string(), description: "Discard Voronoi regions whose originating center object was filtered out or missing.".to_string(), value: format!("{}", _s.exclude_areas_with_no_center), param_type: ParamType::Toggle, options: vec![], min: 0.0f32, max: 0.0f32, step: 1.0000f32, groups: vec![] },
            ],
            Self::Watershed(_s) => vec![
                ParameterDef { name: "maximum_finder_tolerance".to_string(), display_name: "Maximum Finder Tolerance".to_string(), description: "The prominence threshold for peak detection.\n\nThis value determines how \"deep\" the valley between two peaks must be to\nkeep them as separate objects.\n\n* **Low values**: Sensitive to small variations; may cause over-segmentation (splitting one object into many).\n* **High values**: More robust to noise; may cause under-segmentation (failing to split touching objects).\n\nIn an EDM (Euclidean Distance Map), this value directly corresponds to the\npixel distance from the edge of the object.".to_string(), value: format!("{}", _s.maximum_finder_tolerance), param_type: ParamType::Spinner, options: vec![], min: 0.1f32, max: 1.0f32, step: 0.1000f32, groups: vec![] },
            ],
            Self::WeightedDeviation(_s) => vec![
                ParameterDef { name: "kernel_size".to_string(), display_name: "Kernel Size".to_string(), description: "The size of the local neighborhood window.\n\nMust be an odd number. Larger windows capture broader texture\nvariations but increase computational load.".to_string(), value: format!("{}", _s.kernel_size), param_type: ParamType::Number, options: vec![], min: 0.0f32, max: 0.0f32, step: 1.0000f32, groups: vec![] },
                ParameterDef { name: "sigma".to_string(), display_name: "Sigma".to_string(), description: "The standard deviation for the Gaussian weighting function.\n\nDefines the \"softness\" of the neighborhood boundaries. A larger\nsigma includes more of the surrounding context in the deviation calculation.".to_string(), value: format!("{}", _s.sigma), param_type: ParamType::Number, options: vec![], min: 0.0f32, max: 0.0f32, step: 1.0000f32, groups: vec![] },
            ],
        }
    }

    pub fn to_summary(&self) -> String {
        match self {
            Self::Blur(s) => format!("Kernel size: {}", format!("{:.3}", s.kernel_size)),
            Self::ClassifyRois(s) => format!("Min Area: {} · Min Circularity: {} · Min Eccentricity: {} · Max Eccentricity: {} · Allow Edge Touching: {}", format!("{:.3}", s.min_area), format!("{:.3}", s.min_circularity), format!("{:.3}", s.min_eccentricity), format!("{:.3}", s.max_eccentricity), format!("{}", s.allow_edge_touching)),
            Self::Colocalization(_) => String::new(),
            Self::ColorFilterCommand(_) => String::new(),
            Self::ConnectedComponents(_) => String::new(),
            Self::DistanceTransform(_) => String::new(),
            Self::EdgeDetectionCanny(_) => String::new(),
            Self::EdgeDetectionSobel(_) => String::new(),
            Self::EnhanceContrast(_) => String::new(),
            Self::ExtractRois(_) => String::new(),
            Self::GaussianBlur(s) => format!("Kernel Size: {} · Sigma: {}", format!("{:.3}", s.kernel_size), format!("{:.3}", s.sigma)),
            Self::Hessian(_) => String::new(),
            Self::ImageCache(_) => String::new(),
            Self::ImageMath(_) => String::new(),
            Self::IntensityTransformation(_) => String::new(),
            Self::Laplacian(_) => String::new(),
            Self::MedianSubtract(_) => String::new(),
            Self::MorphologicalCommand(_) => String::new(),
            Self::RankFilter(_) => String::new(),
            Self::RollingBall(_) => String::new(),
            Self::SaveImage(_) => String::new(),
            Self::StructureTensor(_) => String::new(),
            Self::Threshold(_) => String::new(),
            Self::Voronoi(_) => String::new(),
            Self::Watershed(_) => String::new(),
            Self::WeightedDeviation(_) => String::new(),
        }
    }

    pub fn apply_param_change(&mut self, param_name: &str, value: &str) {
        match self {
            Self::Blur(s) => {
                if param_name == "kernel_size" {
                    if let Ok(v) = value.parse::<usize>() {
                        s.kernel_size = v;
                    }
                }
            }
            Self::ClassifyRois(s) => {
                if param_name == "input_classes" {
                    if let Some(id) = value
                        .strip_prefix("toggle:")
                        .and_then(|x| x.trim().parse::<u32>().ok())
                    {
                        if s.input_classes
                            .iter()
                            .any(|c| c.to_u32().map_or(false, |v| v == id))
                        {
                            s.input_classes
                                .retain(|c| c.to_u32().map_or(true, |v| v != id));
                        } else {
                            s.input_classes.push(ObjectClass::Valid(id));
                        }
                    } else {
                        s.input_classes = value
                            .split(',')
                            .filter(|x| !x.is_empty())
                            .filter_map(|x| x.trim().parse::<u32>().ok())
                            .map(|v| ObjectClass::Valid(v))
                            .collect();
                    }
                }
                if param_name == "match_handling" {
                    s.match_handling = match value { "AddOutputClassIfMatch" => ClassificationClassifyRoisClassifyMatchHandlingSettings::AddOutputClassIfMatch, "AddOutputClassIfNotMatch" => ClassificationClassifyRoisClassifyMatchHandlingSettings::AddOutputClassIfNotMatch, "RemoveInputClassIfMatch" => ClassificationClassifyRoisClassifyMatchHandlingSettings::RemoveInputClassIfMatch, "RemoveInputClassIfNotMatch" => ClassificationClassifyRoisClassifyMatchHandlingSettings::RemoveInputClassIfNotMatch, "RemoveOutputClassIfMatch" => ClassificationClassifyRoisClassifyMatchHandlingSettings::RemoveOutputClassIfMatch, "RemoveOutputClassIfNotMatch" => ClassificationClassifyRoisClassifyMatchHandlingSettings::RemoveOutputClassIfNotMatch, "RemoveAllClassesIfMatch" => ClassificationClassifyRoisClassifyMatchHandlingSettings::RemoveAllClassesIfMatch, "RemoveAllClassesIfNotMatch" => ClassificationClassifyRoisClassifyMatchHandlingSettings::RemoveAllClassesIfNotMatch, _ => s.match_handling.clone() };
                }
                if param_name == "output_class" {
                    if value == "-1" {
                        s.output_class = ObjectClass::Unset;
                    } else if let Ok(v) = value.parse::<u32>() {
                        s.output_class = ObjectClass::Valid(v);
                    }
                }
                if param_name == "size_unit" {
                    s.size_unit = match value {
                        "nm" => SizeUnits::NanoMeter,
                        _ => SizeUnits::Pixels,
                    };
                }
                if param_name == "min_area" {
                    if let Ok(v) = value.parse::<f32>() {
                        s.min_area = v;
                    }
                }
                if param_name == "max_area" {
                    if let Ok(v) = value.parse::<f32>() {
                        s.max_area = v;
                    }
                }
                if param_name == "min_circularity" {
                    if let Ok(v) = value.parse::<f32>() {
                        s.min_circularity = v;
                    }
                }
                if param_name == "max_circularity" {
                    if let Ok(v) = value.parse::<f32>() {
                        s.max_circularity = v;
                    }
                }
                if param_name == "min_solidity" {
                    if let Ok(v) = value.parse::<f32>() {
                        s.min_solidity = v;
                    }
                }
                if param_name == "max_solidity" {
                    if let Ok(v) = value.parse::<f32>() {
                        s.max_solidity = v;
                    }
                }
                if param_name == "min_aspect_ratio" {
                    if let Ok(v) = value.parse::<f32>() {
                        s.min_aspect_ratio = v;
                    }
                }
                if param_name == "max_aspect_ratio" {
                    if let Ok(v) = value.parse::<f32>() {
                        s.max_aspect_ratio = v;
                    }
                }
                if param_name == "min_eccentricity" {
                    if let Ok(v) = value.parse::<f32>() {
                        s.min_eccentricity = v;
                    }
                }
                if param_name == "max_eccentricity" {
                    if let Ok(v) = value.parse::<f32>() {
                        s.max_eccentricity = v;
                    }
                }
                if param_name == "min_feret" {
                    if let Ok(v) = value.parse::<f32>() {
                        s.min_feret = v;
                    }
                }
                if param_name == "max_feret" {
                    if let Ok(v) = value.parse::<f32>() {
                        s.max_feret = v;
                    }
                }
                if param_name == "allow_edge_touching" {
                    s.allow_edge_touching = value == "true";
                }
            }
            Self::Colocalization(s) => {
                if param_name == "classes_to_coloc" {
                    if let Some(id) = value
                        .strip_prefix("toggle:")
                        .and_then(|x| x.trim().parse::<u32>().ok())
                    {
                        if s.classes_to_coloc
                            .iter()
                            .any(|c| c.to_u32().map_or(false, |v| v == id))
                        {
                            s.classes_to_coloc
                                .retain(|c| c.to_u32().map_or(true, |v| v != id));
                        } else {
                            s.classes_to_coloc.push(ObjectClass::Valid(id));
                        }
                    } else {
                        s.classes_to_coloc = value
                            .split(',')
                            .filter(|x| !x.is_empty())
                            .filter_map(|x| x.trim().parse::<u32>().ok())
                            .map(|v| ObjectClass::Valid(v))
                            .collect();
                    }
                }
                if param_name == "filter_classes" {
                    if let Some(id) = value
                        .strip_prefix("toggle:")
                        .and_then(|x| x.trim().parse::<u32>().ok())
                    {
                        if s.filter_classes
                            .iter()
                            .any(|c| c.to_u32().map_or(false, |v| v == id))
                        {
                            s.filter_classes
                                .retain(|c| c.to_u32().map_or(true, |v| v != id));
                        } else {
                            s.filter_classes.push(ObjectClass::Valid(id));
                        }
                    } else {
                        s.filter_classes = value
                            .split(',')
                            .filter(|x| !x.is_empty())
                            .filter_map(|x| x.trim().parse::<u32>().ok())
                            .map(|v| ObjectClass::Valid(v))
                            .collect();
                    }
                }
                if param_name == "class_for_overlapping_areas" {
                    if value == "-1" {
                        s.class_for_overlapping_areas = ObjectClass::Unset;
                    } else if let Ok(v) = value.parse::<u32>() {
                        s.class_for_overlapping_areas = ObjectClass::Valid(v);
                    }
                }
                if param_name == "allow_multi_object_coloc" {
                    s.allow_multi_object_coloc = value == "true";
                }
                if param_name == "size_unit" {
                    s.size_unit = match value {
                        "nm" => SizeUnits::NanoMeter,
                        _ => SizeUnits::Pixels,
                    };
                }
                if param_name == "min_coloc_area" {
                    if let Ok(v) = value.parse::<f32>() {
                        s.min_coloc_area = v;
                    }
                }
            }
            Self::ColorFilterCommand(s) => {
                if param_name == "range.min_h" {
                    if let Ok(v) = value.parse::<f32>() {
                        s.range.min_h = v;
                    }
                }
                if param_name == "range.max_h" {
                    if let Ok(v) = value.parse::<f32>() {
                        s.range.max_h = v;
                    }
                }
                if param_name == "range.min_s" {
                    if let Ok(v) = value.parse::<f32>() {
                        s.range.min_s = v;
                    }
                }
                if param_name == "range.max_s" {
                    if let Ok(v) = value.parse::<f32>() {
                        s.range.max_s = v;
                    }
                }
                if param_name == "range.min_v" {
                    if let Ok(v) = value.parse::<f32>() {
                        s.range.min_v = v;
                    }
                }
                if param_name == "range.max_v" {
                    if let Ok(v) = value.parse::<f32>() {
                        s.range.max_v = v;
                    }
                }
            }
            Self::ConnectedComponents(_) => {}
            Self::DistanceTransform(s) => {
                if param_name == "threshold" {
                    if let Ok(v) = value.parse::<f32>() {
                        s.threshold = v;
                    }
                }
                if param_name == "edges_are_background" {
                    s.edges_are_background = value == "true";
                }
            }
            Self::EdgeDetectionCanny(s) => {
                if param_name == "kernel_size" {
                    if let Ok(v) = value.parse::<usize>() {
                        s.kernel_size = v;
                    }
                }
                if param_name == "threshold_min" {
                    if let Ok(v) = value.parse::<f32>() {
                        s.threshold_min = v;
                    }
                }
                if param_name == "threshold_max" {
                    if let Ok(v) = value.parse::<f32>() {
                        s.threshold_max = v;
                    }
                }
            }
            Self::EdgeDetectionSobel(s) => {
                if param_name == "kernel_size" {
                    if let Ok(v) = value.parse::<usize>() {
                        s.kernel_size = v;
                    }
                }
            }
            Self::EnhanceContrast(s) => {
                if param_name == "saturated_pixels" {
                    if let Ok(v) = value.parse::<f32>() {
                        s.saturated_pixels = v;
                    }
                }
                if param_name == "normalize" {
                    s.normalize = value == "true";
                }
                if param_name == "equalize_histogram" {
                    s.equalize_histogram = value == "true";
                }
            }
            Self::ExtractRois(s) => {
                if param_name == "max_objects_before_fail" {
                    if let Ok(v) = value.parse::<i32>() {
                        s.max_objects_before_fail = v;
                    }
                }
            }
            Self::GaussianBlur(s) => {
                if param_name == "kernel_size" {
                    if let Ok(v) = value.parse::<usize>() {
                        s.kernel_size = v;
                    }
                }
                if param_name == "sigma" {
                    if let Ok(v) = value.parse::<f32>() {
                        s.sigma = v;
                    }
                }
            }
            Self::Hessian(s) => {
                if param_name == "mode" {
                    s.mode = match value {
                        "Determinant" => FiltersHessianHessianModeSettings::Determinant,
                        "EigenvaluesX" => FiltersHessianHessianModeSettings::EigenvaluesX,
                        "EigenvaluesY" => FiltersHessianHessianModeSettings::EigenvaluesY,
                        _ => s.mode.clone(),
                    };
                }
            }
            Self::ImageCache(s) => {
                if param_name == "mode" {
                    s.mode = match value {
                        "Store" => MathImageCacheImageCacheModeSettings::Store,
                        "Load" => MathImageCacheImageCacheModeSettings::Load,
                        _ => s.mode.clone(),
                    };
                }
            }
            Self::ImageMath(s) => {
                if param_name == "operand" {
                    s.operand = match value {
                        "None" => MathImageMathOperandSettings::None,
                        "Invert" => MathImageMathOperandSettings::Invert,
                        "Add" => MathImageMathOperandSettings::Add,
                        "Subtract" => MathImageMathOperandSettings::Subtract,
                        "Multiply" => MathImageMathOperandSettings::Multiply,
                        "Divide" => MathImageMathOperandSettings::Divide,
                        "AND" => MathImageMathOperandSettings::AND,
                        "OR" => MathImageMathOperandSettings::OR,
                        "XOR" => MathImageMathOperandSettings::XOR,
                        "MIN" => MathImageMathOperandSettings::MIN,
                        "MAX" => MathImageMathOperandSettings::MAX,
                        "Average" => MathImageMathOperandSettings::Average,
                        "DifferenceType" => MathImageMathOperandSettings::DifferenceType,
                        _ => s.operand.clone(),
                    };
                }
                if param_name == "swap_operands" {
                    s.swap_operands = value == "true";
                }
            }
            Self::IntensityTransformation(s) => {
                if param_name == "mode" {
                    s.mode = match value {
                        "Automatic" => {
                            FiltersIntensityTransformIntensityTransformModeSettings::Automatic
                        }
                        "Manual" => FiltersIntensityTransformIntensityTransformModeSettings::Manual,
                        _ => s.mode.clone(),
                    };
                }
                if param_name == "contrast" {
                    if let Ok(v) = value.parse::<f32>() {
                        s.contrast = v;
                    }
                }
                if param_name == "brightness" {
                    if let Ok(v) = value.parse::<f32>() {
                        s.brightness = v;
                    }
                }
            }
            Self::Laplacian(s) => {
                if param_name == "kernel_size" {
                    if let Ok(v) = value.parse::<usize>() {
                        s.kernel_size = v;
                    }
                }
            }
            Self::MedianSubtract(s) => {
                if param_name == "radius" {
                    if let Ok(v) = value.parse::<f64>() {
                        s.radius = v;
                    }
                }
            }
            Self::MorphologicalCommand(s) => {
                if param_name == "op" {
                    s.op = match value {
                        "Dilate" => MorphologyMorphologicalTransformationMorphOpsSettings::Dilate,
                        "Erode" => MorphologyMorphologicalTransformationMorphOpsSettings::Erode,
                        "Open" => MorphologyMorphologicalTransformationMorphOpsSettings::Open,
                        "Close" => MorphologyMorphologicalTransformationMorphOpsSettings::Close,
                        _ => s.op.clone(),
                    };
                }
                if param_name == "kernel_size" {
                    if let Ok(v) = value.parse::<usize>() {
                        s.kernel_size = v;
                    }
                }
                if param_name == "kernel_shape" {
                    s.kernel_shape = match value {
                        "Box" => MorphologyMorphologicalTransformationKernelShapesSettings::Box,
                        "Ellipse" => {
                            MorphologyMorphologicalTransformationKernelShapesSettings::Ellipse
                        }
                        "Cross" => MorphologyMorphologicalTransformationKernelShapesSettings::Cross,
                        _ => s.kernel_shape.clone(),
                    };
                }
                if param_name == "use_grayscale" {
                    s.use_grayscale = value == "true";
                }
            }
            Self::RankFilter(s) => {
                if param_name == "radius" {
                    if let Ok(v) = value.parse::<f64>() {
                        s.radius = v;
                    }
                }
                if param_name == "filter_type" {
                    s.filter_type = match value {
                        "Median" => FiltersRankFilterRankFilterTypeSettings::Median,
                        "Min" => FiltersRankFilterRankFilterTypeSettings::Min,
                        "Max" => FiltersRankFilterRankFilterTypeSettings::Max,
                        "Mean" => FiltersRankFilterRankFilterTypeSettings::Mean,
                        _ => s.filter_type.clone(),
                    };
                }
            }
            Self::RollingBall(s) => {
                if param_name == "radius" {
                    if let Ok(v) = value.parse::<f64>() {
                        s.radius = v;
                    }
                }
                if param_name == "ball_type" {
                    s.ball_type = match value {
                        "Ball" => FiltersRollingBallBallTypeSettings::Ball,
                        "Paraboloid" => FiltersRollingBallBallTypeSettings::Paraboloid,
                        _ => s.ball_type.clone(),
                    };
                }
                if param_name == "pre_smooth" {
                    s.pre_smooth = value == "true";
                }
            }
            Self::SaveImage(s) => {
                if param_name == "path" {
                    s.path = std::path::PathBuf::from(value);
                }
                if param_name == "source" {
                    s.source = match value {
                        "Image" => MathSaveImageImageSourceSettings::Image,
                        "InstanceMap" => MathSaveImageImageSourceSettings::InstanceMap,
                        "SegmentationMask" => MathSaveImageImageSourceSettings::SegmentationMask,
                        _ => s.source.clone(),
                    };
                }
            }
            Self::StructureTensor(s) => {
                if param_name == "mode" {
                    s.mode = match value {
                        "EigenvaluesX" => FiltersStructureTensorTensorModeSettings::EigenvaluesX,
                        "EigenvaluesY" => FiltersStructureTensorTensorModeSettings::EigenvaluesY,
                        "Coherence" => FiltersStructureTensorTensorModeSettings::Coherence,
                        _ => s.mode.clone(),
                    };
                }
                if param_name == "kernel_size" {
                    if let Ok(v) = value.parse::<usize>() {
                        s.kernel_size = v;
                    }
                }
                if param_name == "sigma" {
                    if let Ok(v) = value.parse::<f32>() {
                        s.sigma = v;
                    }
                }
            }
            Self::Threshold(s) => {
                if param_name.starts_with("thresholds.") {
                    let rest = &param_name[11..];
                    let mut _p = rest.splitn(2, '.');
                    if let (Some(_i), Some(nested_name)) = (_p.next(), _p.next()) {
                        if let Ok(_idx) = _i.parse::<usize>() {
                            if let Some(item) = s.thresholds.get_mut(_idx) {
                                if nested_name == "method" {
                                    item.method = match value { "None" => SegmentationThresholdThresholdMethodSettings::None, "Manual" => SegmentationThresholdThresholdMethodSettings::Manual, "Li" => SegmentationThresholdThresholdMethodSettings::Li, "MinError" => SegmentationThresholdThresholdMethodSettings::MinError, "Triangle" => SegmentationThresholdThresholdMethodSettings::Triangle, "Moments" => SegmentationThresholdThresholdMethodSettings::Moments, "Huang" => SegmentationThresholdThresholdMethodSettings::Huang, "Intermodes" => SegmentationThresholdThresholdMethodSettings::Intermodes, "IsoData" => SegmentationThresholdThresholdMethodSettings::IsoData, "MaxEntropy" => SegmentationThresholdThresholdMethodSettings::MaxEntropy, "Mean" => SegmentationThresholdThresholdMethodSettings::Mean, "Minimum" => SegmentationThresholdThresholdMethodSettings::Minimum, "Otsu" => SegmentationThresholdThresholdMethodSettings::Otsu, "Percentile" => SegmentationThresholdThresholdMethodSettings::Percentile, "RenyiEntropy" => SegmentationThresholdThresholdMethodSettings::RenyiEntropy, "Shanbhag" => SegmentationThresholdThresholdMethodSettings::Shanbhag, "Yen" => SegmentationThresholdThresholdMethodSettings::Yen, _ => item.method.clone() };
                                }
                                if nested_name == "min_threshold" {
                                    if let Ok(v) = value.parse::<f32>() {
                                        item.min_threshold = v;
                                    }
                                }
                                if nested_name == "max_threshold" {
                                    if let Ok(v) = value.parse::<f32>() {
                                        item.max_threshold = v;
                                    }
                                }
                                if nested_name == "unit" {
                                    item.unit = match value {
                                        "bit" => PixelUnits::Bit,
                                        "%" => PixelUnits::Percent,
                                        _ => PixelUnits::Relative,
                                    };
                                }
                                if nested_name == "object_class_id" {
                                    if let Ok(v) = value.parse::<u32>() {
                                        item.object_class_id = SegmentationClass(v);
                                    }
                                }
                            }
                        }
                    }
                }
            }
            Self::Voronoi(s) => {
                if param_name == "centers" {
                    if value == "-1" {
                        s.centers = ObjectClass::Unset;
                    } else if let Ok(v) = value.parse::<u32>() {
                        s.centers = ObjectClass::Valid(v);
                    }
                }
                if param_name == "center_filter_classes" {
                    if let Some(id) = value
                        .strip_prefix("toggle:")
                        .and_then(|x| x.trim().parse::<u32>().ok())
                    {
                        if s.center_filter_classes
                            .iter()
                            .any(|c| c.to_u32().map_or(false, |v| v == id))
                        {
                            s.center_filter_classes
                                .retain(|c| c.to_u32().map_or(true, |v| v != id));
                        } else {
                            s.center_filter_classes.push(ObjectClass::Valid(id));
                        }
                    } else {
                        s.center_filter_classes = value
                            .split(',')
                            .filter(|x| !x.is_empty())
                            .filter_map(|x| x.trim().parse::<u32>().ok())
                            .map(|v| ObjectClass::Valid(v))
                            .collect();
                    }
                }
                if param_name == "mask" {
                    if value == "-1" {
                        s.mask = ObjectClass::Unset;
                    } else if let Ok(v) = value.parse::<u32>() {
                        s.mask = ObjectClass::Valid(v);
                    }
                }
                if param_name == "mask_filter_classes" {
                    if let Some(id) = value
                        .strip_prefix("toggle:")
                        .and_then(|x| x.trim().parse::<u32>().ok())
                    {
                        if s.mask_filter_classes
                            .iter()
                            .any(|c| c.to_u32().map_or(false, |v| v == id))
                        {
                            s.mask_filter_classes
                                .retain(|c| c.to_u32().map_or(true, |v| v != id));
                        } else {
                            s.mask_filter_classes.push(ObjectClass::Valid(id));
                        }
                    } else {
                        s.mask_filter_classes = value
                            .split(',')
                            .filter(|x| !x.is_empty())
                            .filter_map(|x| x.trim().parse::<u32>().ok())
                            .map(|v| ObjectClass::Valid(v))
                            .collect();
                    }
                }
                if param_name == "output_class" {
                    if value == "-1" {
                        s.output_class = ObjectClass::Unset;
                    } else if let Ok(v) = value.parse::<u32>() {
                        s.output_class = ObjectClass::Valid(v);
                    }
                }
                if param_name == "unit" {
                    s.unit = match value {
                        "nm" => SizeUnits::NanoMeter,
                        _ => SizeUnits::Pixels,
                    };
                }
                if param_name == "max_radius" {
                    if let Ok(v) = value.parse::<f32>() {
                        s.max_radius = v;
                    }
                }
                if param_name == "exclude_areas_at_the_edges" {
                    s.exclude_areas_at_the_edges = value == "true";
                }
                if param_name == "exclude_areas_with_no_center" {
                    s.exclude_areas_with_no_center = value == "true";
                }
            }
            Self::Watershed(s) => {
                if param_name == "maximum_finder_tolerance" {
                    if let Ok(v) = value.parse::<f32>() {
                        s.maximum_finder_tolerance = v;
                    }
                }
            }
            Self::WeightedDeviation(s) => {
                if param_name == "kernel_size" {
                    if let Ok(v) = value.parse::<usize>() {
                        s.kernel_size = v;
                    }
                }
                if param_name == "sigma" {
                    if let Ok(v) = value.parse::<f32>() {
                        s.sigma = v;
                    }
                }
            }
        }
    }

    pub fn add_group_item(&mut self, param_name: &str) {
        match self {
            Self::Blur(_) => {}
            Self::ClassifyRois(_) => {}
            Self::Colocalization(_) => {}
            Self::ColorFilterCommand(_) => {}
            Self::ConnectedComponents(_) => {}
            Self::DistanceTransform(_) => {}
            Self::EdgeDetectionCanny(_) => {}
            Self::EdgeDetectionSobel(_) => {}
            Self::EnhanceContrast(_) => {}
            Self::ExtractRois(_) => {}
            Self::GaussianBlur(_) => {}
            Self::Hessian(_) => {}
            Self::ImageCache(_) => {}
            Self::ImageMath(_) => {}
            Self::IntensityTransformation(_) => {}
            Self::Laplacian(_) => {}
            Self::MedianSubtract(_) => {}
            Self::MorphologicalCommand(_) => {}
            Self::RankFilter(_) => {}
            Self::RollingBall(_) => {}
            Self::SaveImage(_) => {}
            Self::StructureTensor(_) => {}
            Self::Threshold(s) => {
                if param_name == "thresholds" {
                    if let Some(last) = s.thresholds.last().cloned() {
                        s.thresholds.push(last);
                    } else {
                        s.thresholds.push(ThresholdEntrySettings::default());
                    }
                }
            }
            Self::Voronoi(_) => {}
            Self::Watershed(_) => {}
            Self::WeightedDeviation(_) => {}
        }
    }

    pub fn remove_group_item(&mut self, param_name: &str, idx: usize) {
        match self {
            Self::Blur(_) => {}
            Self::ClassifyRois(_) => {}
            Self::Colocalization(_) => {}
            Self::ColorFilterCommand(_) => {}
            Self::ConnectedComponents(_) => {}
            Self::DistanceTransform(_) => {}
            Self::EdgeDetectionCanny(_) => {}
            Self::EdgeDetectionSobel(_) => {}
            Self::EnhanceContrast(_) => {}
            Self::ExtractRois(_) => {}
            Self::GaussianBlur(_) => {}
            Self::Hessian(_) => {}
            Self::ImageCache(_) => {}
            Self::ImageMath(_) => {}
            Self::IntensityTransformation(_) => {}
            Self::Laplacian(_) => {}
            Self::MedianSubtract(_) => {}
            Self::MorphologicalCommand(_) => {}
            Self::RankFilter(_) => {}
            Self::RollingBall(_) => {}
            Self::SaveImage(_) => {}
            Self::StructureTensor(_) => {}
            Self::Threshold(s) => {
                if param_name == "thresholds" && idx < s.thresholds.len() {
                    s.thresholds.remove(idx);
                }
            }
            Self::Voronoi(_) => {}
            Self::Watershed(_) => {}
            Self::WeightedDeviation(_) => {}
        }
    }
}
