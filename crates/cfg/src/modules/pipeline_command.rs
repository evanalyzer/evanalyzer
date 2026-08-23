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
#[serde(tag = "type", rename_all = "SCREAMING_SNAKE_CASE")]
pub enum PipelineCommand {
    #[serde(
        alias = "ai-object-classifier",
        alias = "aiObjectClassifier",
        alias = "ai_object_classifier"
    )]
    AiObjectClassifier(AiObjectClassifierSettings),
    #[serde(alias = "blur")]
    Blur(BlurSettings),
    #[serde(alias = "cellpose")]
    Cellpose(CellposeSettings),
    #[serde(
        alias = "classify-objects",
        alias = "classifyObjects",
        alias = "classify_objects"
    )]
    ClassifyObjects(ClassifyObjectsSettings),
    #[serde(alias = "colocalization")]
    Colocalization(ColocalizationSettings),
    #[serde(
        alias = "color-filter-command",
        alias = "colorFilterCommand",
        alias = "color_filter_command"
    )]
    ColorFilterCommand(ColorFilterCommandSettings),
    #[serde(
        alias = "connected-components",
        alias = "connectedComponents",
        alias = "connected_components"
    )]
    ConnectedComponents(ConnectedComponentsSettings),
    #[serde(
        alias = "distance-transform",
        alias = "distanceTransform",
        alias = "distance_transform"
    )]
    DistanceTransform(DistanceTransformSettings),
    #[serde(
        alias = "edge-detection-canny",
        alias = "edgeDetectionCanny",
        alias = "edge_detection_canny"
    )]
    EdgeDetectionCanny(EdgeDetectionCannySettings),
    #[serde(
        alias = "edge-detection-sobel",
        alias = "edgeDetectionSobel",
        alias = "edge_detection_sobel"
    )]
    EdgeDetectionSobel(EdgeDetectionSobelSettings),
    #[serde(
        alias = "enhance-contrast",
        alias = "enhanceContrast",
        alias = "enhance_contrast"
    )]
    EnhanceContrast(EnhanceContrastSettings),
    #[serde(
        alias = "extract-objects",
        alias = "extractObjects",
        alias = "extract_objects"
    )]
    ExtractObjects(ExtractObjectsSettings),
    #[serde(alias = "fill-holes", alias = "fillHoles", alias = "fill_holes")]
    FillHoles(FillHolesSettings),
    #[serde(
        alias = "gaussian-blur",
        alias = "gaussianBlur",
        alias = "gaussian_blur"
    )]
    GaussianBlur(GaussianBlurSettings),
    #[serde(alias = "hessian")]
    Hessian(HessianSettings),
    #[serde(
        alias = "illumination-correction",
        alias = "illuminationCorrection",
        alias = "illumination_correction"
    )]
    IlluminationCorrection(IlluminationCorrectionSettings),
    #[serde(alias = "image-cache", alias = "imageCache", alias = "image_cache")]
    ImageCache(ImageCacheSettings),
    #[serde(alias = "image-math", alias = "imageMath", alias = "image_math")]
    ImageMath(ImageMathSettings),
    #[serde(
        alias = "intensity-transformation",
        alias = "intensityTransformation",
        alias = "intensity_transformation"
    )]
    IntensityTransformation(IntensityTransformationSettings),
    #[serde(alias = "laplacian")]
    Laplacian(LaplacianSettings),
    #[serde(
        alias = "median-subtract",
        alias = "medianSubtract",
        alias = "median_subtract"
    )]
    MedianSubtract(MedianSubtractSettings),
    #[serde(
        alias = "morphological-command",
        alias = "morphologicalCommand",
        alias = "morphological_command"
    )]
    MorphologicalCommand(MorphologicalCommandSettings),
    #[serde(alias = "object-math", alias = "objectMath", alias = "object_math")]
    ObjectMath(ObjectMathSettings),
    #[serde(
        alias = "pixel-classifier",
        alias = "pixelClassifier",
        alias = "pixel_classifier"
    )]
    PixelClassifier(PixelClassifierSettings),
    #[serde(alias = "rank-filter", alias = "rankFilter", alias = "rank_filter")]
    RankFilter(RankFilterSettings),
    #[serde(alias = "rolling-ball", alias = "rollingBall", alias = "rolling_ball")]
    RollingBall(RollingBallSettings),
    #[serde(alias = "save-image", alias = "saveImage", alias = "save_image")]
    SaveImage(SaveImageSettings),
    #[serde(alias = "stardist")]
    Stardist(StardistSettings),
    #[serde(
        alias = "structure-tensor",
        alias = "structureTensor",
        alias = "structure_tensor"
    )]
    StructureTensor(StructureTensorSettings),
    #[serde(alias = "threshold")]
    Threshold(ThresholdSettings),
    #[serde(
        alias = "transform-objects",
        alias = "transformObjects",
        alias = "transform_objects"
    )]
    TransformObjects(TransformObjectsSettings),
    #[serde(alias = "u-net", alias = "uNet", alias = "u_net")]
    UNet(UNetSettings),
    #[serde(alias = "voronoi")]
    Voronoi(VoronoiSettings),
    #[serde(alias = "watershed")]
    Watershed(WatershedSettings),
    #[serde(
        alias = "weighted-deviation",
        alias = "weightedDeviation",
        alias = "weighted_deviation"
    )]
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
        CommandMeta {
            id: 0,
            name: "AI Object Classifier",
            category: CommandCategory::Classify,
            summary: "An object classifier trained via the app's AI training dialog (an",
            description: "`.evamodel` file - see `ai_learning::training::object::ObjectTrainingJob`),\napplied here as a pipeline classification step: every object already\npresent in `PipelineCache::object_cache` matching `origin_segmentation`/\n`input_classes` is scored independently (reusing the same feature recipe\nused at training time), then remapped through `segmentation_mapping` into\nthis project's own classes and applied via `match_handling` - the same\ninput selection `ClassifyObjects` uses, but the output class comes from\nthe model's prediction instead of a fixed rule.\n\nPredicted classes with no matching `segmentation_mapping` entry leave the\nobject untouched, mirroring how `PixelClassifier` leaves unmapped\npredictions as background - mapping only the classes you care about is a\ndeliberate simplification, not an oversight.",
        },
        CommandMeta {
            id: 1,
            name: "Blur",
            category: CommandCategory::Preprocess,
            summary: "Smooths an image by averaging pixel intensities within a local neighborhood.",
            description: "This algorithm applies a uniform box filter where every pixel within the moving\nwindow contributes equally to the final value. It is a computationally fast\nmethod used for general image smoothing, blending variations, and rapid noise\nsuppression where edge precision is less critical.",
        },
        CommandMeta {
            id: 2,
            name: "AI Cellpose Segmentation",
            category: CommandCategory::Segment,
            summary: "Instance segmentation using a pretrained Cellpose model exported as TorchScript.",
            description: "The model is fed a `[1, input_channels, H, W]` float tensor: the (normalized)\ngrayscale image is placed in channel 0 and any remaining channels are filled\nwith zeros. Standard Cellpose networks expect **two** channels (cytoplasm +\noptional nucleus), which is the default; single-channel exports use\n`input_channels = 1`. The model must return a `[1, C, H, W]` tensor with\n`C >= 3` channels: the vertical flow `dY` (channel 0), the horizontal flow\n`dX` (channel 1) and the cell-probability logits (channel 2), which is\nCellpose's spatial-gradient representation. Exports that wrap the output in a\ntuple (e.g. `(flows, style)`) are also supported — the first tensor with at\nleast three channels is used.\n\nInstances are recovered with Cellpose's *dynamics*: every pixel whose\ncell probability reaches `probability_threshold` is advected for\n`flow_iterations` Euler steps along the (down-scaled) flow field until it\nconverges to the sink at its cell's center. Pixels whose trajectories end in\nthe same sink basin — found by connected components over the final-position\ndensity map — form one instance. Instances smaller than `min_object_size`\npixels are discarded. Runs on GPU automatically if CUDA is available in the\nlinked libtorch build, otherwise falls back to CPU.",
        },
        CommandMeta {
            id: 3,
            name: "ClassifyObjects",
            category: CommandCategory::Classify,
            summary: "Classifies ROIs based on morphological and intensity features.",
            description: "This command applies rule-based classification logic to assign object classes\nto extracted ROIs. Classification is performed using configurable criteria\nincluding area, shape descriptors, and intensity statistics.",
        },
        CommandMeta {
            id: 4,
            name: "Colocalization",
            category: CommandCategory::Classify,
            summary: "Calculates spatial colocalization and intersections between specified object classes.",
            description: "This command scans the object cache, groups objects by their designated classes,\nand performs spatial overlap analysis. It records colocalization relationships\nbetween intersecting entities and can optionally generate new child ROIs representing\nthe precise intersection regions.",
        },
        CommandMeta {
            id: 5,
            name: "ColorFilterCommand",
            category: CommandCategory::Preprocess,
            summary: "A command that filters an image based on a specific HSV color range.",
            description: "Pixels falling outside the provided [`HsvRange`] are masked\nout by setting to black.\n\n# Examples\n\n```\n# use imagec::backend::algos::{ColorFilterCommand, HsvRange};\nlet range = HsvRange {\nmin_h: 0.0,   max_h: 30.0, // Red tones\nmin_s: 0.5,   max_s: 1.0,\nmin_v: 0.5,   max_v: 1.0,\n};\n\nlet command = ColorFilterCommand { range };\n```",
        },
        CommandMeta {
            id: 6,
            name: "ConnectedComponents",
            category: CommandCategory::Object,
            summary: "Identifies and labels discrete objects within a binary or multi-class image.",
            description: "",
        },
        CommandMeta {
            id: 7,
            name: "DistanceTransform",
            category: CommandCategory::Preprocess,
            summary: "A command that calculates the Euclidean Distance Map (EDM) of an f32 image.",
            description: "This algorithm identifies pixels below a threshold as \"background\" and\ncalculates the distance of every \"foreground\" pixel to the nearest background pixel.",
        },
        CommandMeta {
            id: 8,
            name: "EdgeDetectionCanny",
            category: CommandCategory::Preprocess,
            summary: "Extracts structural boundaries and fine edges using the multi-stage Canny algorithm.",
            description: "This algorithm identifies optimal edge locations by calculating spatial intensity\ngradients, suppressing non-maximum pixel responses to thin lines down to 1-pixel width,\nand applying a dual-threshold hysteresis loop to preserve weak edges connected\nto strong ones while completely rejecting isolated noise.\n\n# Examples\n\n```\n# use imagec::backend::algos::EdgeDetectionCanny;\nlet edges = EdgeDetectionCanny {\nkernel_size: 3,\nthreshold_min: 0.1,\nthreshold_max: 0.3,\n};\n```",
        },
        CommandMeta {
            id: 9,
            name: "EdgeDetectionSobel",
            category: CommandCategory::Preprocess,
            summary: "Extracts directional boundaries by computing spatial image intensity gradients.",
            description: "This algorithm applies localized 3x3 kernels to approximate the first derivative\nof pixel intensities across the horizontal and vertical axes. It highlights\nareas of sharp luminance changes, producing a continuous gradient map that\nemphasizes prominent structural edges and surface transitions.\n\n# Examples\n\n```\n# use imagec::backend::algos::EdgeDetectionSobel;\nlet filter = EdgeDetectionSobel { kernel_size: 3 };\n```",
        },
        CommandMeta {
            id: 10,
            name: "EnhanceContrast",
            category: CommandCategory::Preprocess,
            summary: "Configuration for contrast enhancement and histogram manipulation.",
            description: "This algorithm can perform linear contrast stretching, normalization,\nor histogram equalization to improve the dynamic range of an image.\n\n# Examples\n\n```\n# use imagec::backend::algos::EnhanceContrast;\nlet settings = EnhanceContrast {\nsaturated_pixels: 0.01,   // Clip 1% of outliers\nnormalize: true,          // Stretch to [0.0, 1.0]\nequalize_histogram: false,\n};\n```",
        },
        CommandMeta {
            id: 11,
            name: "ExtractObjects",
            category: CommandCategory::Measure,
            summary: "Represents a bounded object extracted from a labeled image.",
            description: "A command to extract spatial statistics and bounding boxes from labeled objects.",
        },
        CommandMeta {
            id: 12,
            name: "FillHoles",
            category: CommandCategory::Object,
            summary: "Fills enclosed background holes in the segmentation map.",
            description: "A direct port of ImageJ's `Process > Binary > Fill Holes` command\n(`ij.plugin.filter.Binary.fill`, originally contributed by Gabriel\nLandini): a background pixel counts as a \"hole\" - and is turned into\nforeground - exactly when it cannot be reached from the image border by a\npath of background pixels using 4-connectivity.\n\nLike ImageJ's own command, this treats the image as strictly binary: every\nnon-background pixel is \"foreground\" regardless of its actual label/class\nvalue, and a filled hole is stamped with a single fixed value rather than\ninheriting whatever label happens to surround it. If the segmentation map\ncarries several distinct label values, holes are not attributed back to\nthe object that encloses them - only ImageJ's original background/\nforeground distinction is reproduced here.\n\n# Algorithm (matches `ij.process.FloodFiller.fill(x, y)`)\n1. Scan every pixel on the image border; for each one that is background\n(`0`), flood-fill outward from it using 4-connectivity (up/down/left/\nright only - diagonal neighbors are **not** considered connected),\nmarking every background pixel reached this way as \"outside\".\n2. Any background pixel never marked \"outside\" is enclosed and becomes\nforeground. Every non-background pixel is copied through unchanged.\n\nThe 4-connectivity in step 1 is load-bearing, not an implementation\ndetail: a boundary that only touches itself diagonally (8-connected) does\n**not** block this flood fill, exactly mirroring ImageJ's `FloodFiller`,\nwhose own docs specify a 4-connected fill.",
        },
        CommandMeta {
            id: 13,
            name: "GaussianBlur",
            category: CommandCategory::Preprocess,
            summary: "Smooths an image and reduces background noise using a Gaussian kernel.",
            description: "This algorithm applies a localized, bell-curve weighted blur that suppresses\nhigh-frequency pixel variations (like camera noise, salt-and-pepper artifacts,\nor dust) while preserving structural features. It is commonly used as a\npreprocessing step to optimize thresholding and edge detection tasks.\n\n# Examples\n\n```\nuse imagec::backend::algos::GaussianBlur;\n\nlet settings = GaussianBlur {\nkernel_size: 5,\nsigma: 2.0\n};\n```",
        },
        CommandMeta {
            id: 14,
            name: "Hessian",
            category: CommandCategory::Preprocess,
            summary: "Extracts continuous structural ridges, tubular vessels, and blobs using second-order spatial derivatives.",
            description: "This algorithm constructs a localized Hessian matrix for each pixel to analyze local curvature\nand intensity topography. By evaluating the eigenvalues of this matrix, it differentiates\nbetween directional ridges (like blood vessels or filaments), distinct intensity peaks (blobs),\nand flat regions, making it highly effective for curvilinear feature extraction.\n\n# Examples\n\n```\n# use imagec::backend::algos::{Hessian, HessianMode};\nlet detector = Hessian {\nmode: HessianMode::Determinant,\n};\n```",
        },
        CommandMeta {
            id: 15,
            name: "IlluminationCorrection",
            category: CommandCategory::Preprocess,
            summary: "Use this when your images are brighter in the middle and dimmer toward",
            description: "the edges/corners (vignetting), or show any other smooth shading pattern\nthat repeats the same way across every tile or every image from the same\nmicroscope/camera setup - a consequence of the optics or illumination,\nnot the sample. Left uncorrected, that shading makes intensity\ncomparisons between regions of an image (or between images/wells)\nunreliable, even though it rarely stops segmentation from finding\nobjects on its own.\n\nUse [`super::rolling_ball::RollingBall`] instead when the problem is a\n*local* background glow or halo under/around individual objects (e.g.\nout-of-focus light, autofluorescence, uneven staining) that differs from\nimage to image rather than being tied to the acquisition setup -\nRollingBall strips that local floor so thresholding/segmentation works\ncleanly. The two solve different problems: RollingBall won't fix a\nglobal brightness gradient, and this filter won't remove a local halo.\n\n### How it works\n\nFlat-field (\"illumination\") correction: estimates a smooth, slowly-varying\ngain/offset field caused by uneven illumination (vignetting, dust on the\ncondenser, uneven excitation) and removes it in a single calculate+apply\nstep - equivalent to CellProfiler's `CorrectIlluminationCalculate` and\n`CorrectIlluminationApply` modules combined into one.\n\nUnlike `RollingBall`, which estimates a *local* per-object background\nbaseline via a rolling structural element, this estimates one *global*,\nlow-frequency field for the whole image/channel.",
        },
        CommandMeta {
            id: 16,
            name: "ImageCache",
            category: CommandCategory::Preprocess,
            summary: "A filter that acts as a synchronization point between the pipeline and a storage backend.",
            description: "`ImageCache` allows the pipeline to branch or \"undo\" operations by saving\nstates to a named address and reloading them as needed.\n\n# Examples\n\n```\nuse imagec::backend::core::context::{ImageCache, ImageCacheMode, ImageAddress};\nlet checkpoint = ImageCache {\nmode: ImageCacheMode::Store,\naddress: ImageAddress::from(\"pre_processed_state\"),\n};\n```",
        },
        CommandMeta {
            id: 17,
            name: "ImageMath",
            category: CommandCategory::Preprocess,
            summary: "A filter that performs pixel-wise mathematical operations between the current",
            description: "pipeline image and a secondary image stored in the cache.\n\nThis command allows for complex image blending, masking, and comparison.\n\n# Examples\n\n```\nuse imagec::backend::algos::{ImageMath, Operand};\nlet subtract_bg = ImageMath {\noperand: Operand::Subtract,\nsecond_image_address: ImageAddress::from(\"background\"),\nswap_operands: false,\n};\n```",
        },
        CommandMeta {
            id: 18,
            name: "IntensityTransformation",
            category: CommandCategory::Preprocess,
            summary: "Configuration for adjusting image contrast and brightness.",
            description: "This transformation applies a linear mapping to pixel values.\nIn [`Mode::Manual`], the output is typically calculated as:\n`output = input * contrast + brightness`.",
        },
        CommandMeta {
            id: 19,
            name: "Laplacian",
            category: CommandCategory::Preprocess,
            summary: "Configuration for the Laplacian edge detection filter.",
            description: "The Laplacian is a second-order derivative operator used to find regions of\nrapid intensity change. It is particularly effective for detecting edges\nand fine details, though it is highly sensitive to noise.\n\n# Examples\n\n```\n# use imagec::backend::algos::Laplacian;\nlet filter = Laplacian { kernel_size: 3 };\n```",
        },
        CommandMeta {
            id: 20,
            name: "MedianSubtract",
            category: CommandCategory::Preprocess,
            summary: "A background subtraction filter that uses a median rank operator.",
            description: "This algorithm is highly effective for removing large-scale background\nvariations while preserving small, high-contrast features. It works by\nestimating the background as the median intensity within a local radius.\n\n# Examples\n\n```\nuse imagec::backend::algos::MedianSubtract;\nlet filter = MedianSubtract { radius: 10.0 };\n```",
        },
        CommandMeta {
            id: 21,
            name: "MorphologicalCommand",
            category: CommandCategory::Preprocess,
            summary: "A filter that applies mathematical morphology to an image.",
            description: "Morphological operations use a structuring element (kernel) to probe\nand modify the shapes within an image.\n\n# Examples\n\n```\nuse imagec::backend::algos::{MorphologicalCommand, MorphOps, KernelShapes};\nlet clean_noise = MorphologicalCommand {\nop: MorphOps::Open,\nkernel_size: 3,\nkernel_shape: KernelShapes::Ellipse,\n};\n```",
        },
        CommandMeta {
            id: 22,
            name: "ObjectMath",
            category: CommandCategory::Classify,
            summary: "Computes a boolean set operation between two object classes, object pair by",
            description: "object pair.\n\nWhen more than one `other_class` object overlaps a given input object, all of them\nare unioned into a single \"B\" before the operation is applied, so the result\ndoesn't depend on the order they'd otherwise be combined in.",
        },
        CommandMeta {
            id: 23,
            name: "AI Pixel Classifier",
            category: CommandCategory::Segment,
            summary: "A pixel classifier trained via the app's AI training dialog",
            description: "(an`.evamodel` file - see `ai_learning::training::pixel::PixelTrainingJob`),\napplied here as a pipeline segmentation step: every pixel is classified\nindependently (reusing the same feature recipe used at training time),\nthen remapped through `segmentation_mapping` into this project's own\nclasses and written to the segmentation map - the same output shape\n`Threshold` produces, so downstream steps (extraction, classification)\ndon't need to care which one ran.\n\nPredicted classes with no matching `segmentation_mapping` entry are\nwritten as `SegmentationClass::BACKGROUND`, mirroring how `Threshold`\nleaves pixels outside every configured range as background - mapping\nonly the classes you care about is a deliberate simplification, not an\noversight.",
        },
        CommandMeta {
            id: 24,
            name: "RankFilter",
            category: CommandCategory::Preprocess,
            summary: "A filter that transforms pixels based on the statistical rank of their neighbors.",
            description: "Rank filters are non-linear operators used for noise reduction,\nmorphological operations, and feature enhancement.\n\nThis algorithm sorts (ranks) all pixel values within a local neighborhood\nwindow and assigns a specific percentile value to the center pixel. By selecting\ndifferent ranks, it acts as a configurable operator: the minimum rank performs\nerosion, the maximum rank performs dilation, and the median rank (50th percentile)\nprovides highly effective impulse noise suppression while preserving sharp structural edges.",
        },
        CommandMeta {
            id: 25,
            name: "RollingBall",
            category: CommandCategory::Preprocess,
            summary: "Removes non-uniform background illumination by calculating a local intensity baseline.",
            description: "This algorithm models the image as a 3D intensity landscape and conceptually rolls\na sphere of a user-defined radius underneath it. The ball cannot penetrate narrow\nintensity peaks (true signal objects) but follows the sweeping, lower-frequency\ncurves of background variations. The path traced by the ball establishes a local\nbaseline map that is subtracted from the original image to isolate foreground features.",
        },
        CommandMeta {
            id: 26,
            name: "SaveImage",
            category: CommandCategory::Preprocess,
            summary: "A command that exports the current image to a persistent file on disk.",
            description: "This is a **transparent command**: it does not modify the image data in the\npipeline context, nor does it perform a buffer swap. It acts as a tap\nto view the state of the image at a specific point in the pipeline.\n\n# Examples\n\n```\nuse imagec::backend::algos::SaveImage;\nlet saver = SaveImage {path:\"output/processed_cell.png\"};\n```",
        },
        CommandMeta {
            id: 27,
            name: "AI Stardist Segmentation",
            category: CommandCategory::Segment,
            summary: "Instance segmentation using a pretrained StarDist model exported as TorchScript.",
            description: "The model is expected to accept a `[1, 1, H, W]` float tensor (single-channel,\nsame normalization as the rest of the pipeline) and return two tensors:\nan object-probability map `[1, 1, H', W']` and a ray-distance map\n`[1, n_rays, H', W']` giving, for each grid cell, the distance to the object\nboundary along `n_rays` equally-spaced angles (the StarDist star-convex-polygon\nrepresentation). `H'`/`W'` may be smaller than the input size if the model\npredicts on a coarser grid; this is detected from the output shape and the\npolygons are rescaled back to image resolution automatically.\n\nSome TorchScript exports concatenate both outputs into a single\n`[1, 1 + n_rays, H', W']` tensor (channel 0 = probability, the rest =\ndistances); this is also supported.\n\nPer grid cell candidates above `probability_threshold` are converted to\nstar-convex polygons, then greedily filtered with non-maximum suppression\n(polygons whose pixel-overlap ratio with a higher-scoring candidate exceeds\n`nms_threshold` are discarded) before being rasterized into the pipeline's\nsegmentation and instance maps. Runs on GPU automatically if CUDA is\navailable in the linked libtorch build, otherwise falls back to CPU.",
        },
        CommandMeta {
            id: 28,
            name: "StructureTensor",
            category: CommandCategory::Preprocess,
            summary: "Analyzes local image texture, directional orientation, and corner features using a second-moment matrix.",
            description: "This algorithm summarizes the predominant directions of the image gradient within a local\nneighborhood, smoothing the structural data with a Gaussian window. By evaluating the\neigenvalues of the resulting matrix tensor, it distinguishes between flat areas (both eigenvalues\nnear zero), straight linear boundaries (one dominant eigenvalue indicating structural direction),\nand complex corners or intersections (two large eigenvalues).\n\n# Examples\n\n```\nuse imagec::backend::algos::{StructureTensor, Mode};\nlet settings = StructureTensor {\nmode: Mode::Coherence,\nkernel_size: 3,\nsigma: 1.5\n};\n```",
        },
        CommandMeta {
            id: 29,
            name: "Threshold",
            category: CommandCategory::Segment,
            summary: "A filter that segments an image into discrete classes based on intensity.",
            description: "This supports \"Multi-Otsu\" style behavior by allowing a vector of\n[`ThresholdSettings`]. Each pixel is evaluated against the settings to\ndetermine which `object_class_id` it belongs to.",
        },
        CommandMeta {
            id: 30,
            name: "TransformObjects",
            category: CommandCategory::Classify,
            summary: "Transforms given ROIs and either replaces the old ones or creates new ones.",
            description: "This command applies a geometric transform (scale, circle, fitted ellipse) to every object\ncarrying `input_class`. The transformed shape keeps the original object's bounding-box center.\nIf `output_class` is unset (or equal to `input_class`) the input object is replaced in place;\notherwise a new object carrying `output_class` is created alongside the untouched input object.",
        },
        CommandMeta {
            id: 31,
            name: "AI UNet Segmentation",
            category: CommandCategory::Segment,
            summary: "Semantic segmentation using a pretrained U-Net exported as TorchScript.",
            description: "The model is expected to accept a `[1, 1, H, W]` float tensor (single-channel,\nsame normalization as the rest of the pipeline) and return either a\n`[1, 1, H, W]` tensor of per-pixel foreground probabilities (the model already\napplies its final sigmoid) or a `[1, C, H, W]` tensor with more than one\nchannel, in which case `output_mode` and `foreground_channel` decide how the\nforeground probability is extracted (see [`UNetOutputMode`]). Runs on GPU\nautomatically if CUDA is available in the linked libtorch build, otherwise\nfalls back to CPU.",
        },
        CommandMeta {
            id: 32,
            name: "Voronoi",
            category: CommandCategory::Classify,
            summary: "Computes a Voronoi tessellation from segmented seed objects.",
            description: "Each seed center expands outward until it reaches another region, the optional mask\nboundary, or the maximum radius. The resulting areas are stored as new ROIs labeled\nwith `output_class` and linked to their originating center object.",
        },
        CommandMeta {
            id: 33,
            name: "Watershed",
            category: CommandCategory::Object,
            summary: "A morphological segmentation algorithm that splits touching objects using distance topography.",
            description: "This is a faithful port of ImageJ's `Process > Binary > Watershed`\n(`MaximumFinder` applied to the Euclidean distance map). Touching objects that\n`ConnectedComponents` merged into a single blob are split at their \"necks\":\nthe distance map's local maxima are the seeds, maxima protruding less than\n`maximum_finder_tolerance` above the ridge connecting them to a higher maximum\nare merged, and a constrained flood draws 1-pixel watershed lines between the\nsurviving basins. The split blob is then re-labeled into separate instances.",
        },
        CommandMeta {
            id: 34,
            name: "WeightedDeviation",
            category: CommandCategory::Preprocess,
            summary: "A filter that computes the Gaussian-weighted standard deviation of a local neighborhood.",
            description: "Unlike a standard deviation filter which treats all pixels in a window equally,\nthe Weighted Deviation uses a Gaussian kernel to give more importance to\npixels closer to the center. This is particularly effective for edge-preserving\nnoise analysis and local contrast enhancement.\n\nThis algorithm evaluates local variance by calculating two distinct Gaussian-blurred\nbaselines across the image: the weighted average of the pixel intensities, and the\nweighted average of the squared intensities. By subtracting the squared mean from\nthe mean of squares, it yields a localized, smooth statistical variance map that\nhighlights micro-textures and subtle surface boundaries without producing blocky artifacts.\n\n# Examples\n\n```\nuse imagec::backend::algos::WeightedDeviation;\nlet settings = WeightedDeviation {\nkernel_size: 7,\nsigma: 2.0,\n};\n```",
        },
    ]
}

#[allow(dead_code)]
pub fn default_command(id: i32) -> Option<PipelineCommand> {
    match id {
        0 => Some(PipelineCommand::AiObjectClassifier(
            AiObjectClassifierSettings::default(),
        )),
        1 => Some(PipelineCommand::Blur(BlurSettings::default())),
        2 => Some(PipelineCommand::Cellpose(CellposeSettings::default())),
        3 => Some(PipelineCommand::ClassifyObjects(
            ClassifyObjectsSettings::default(),
        )),
        4 => Some(PipelineCommand::Colocalization(
            ColocalizationSettings::default(),
        )),
        5 => Some(PipelineCommand::ColorFilterCommand(
            ColorFilterCommandSettings::default(),
        )),
        6 => Some(PipelineCommand::ConnectedComponents(
            ConnectedComponentsSettings::default(),
        )),
        7 => Some(PipelineCommand::DistanceTransform(
            DistanceTransformSettings::default(),
        )),
        8 => Some(PipelineCommand::EdgeDetectionCanny(
            EdgeDetectionCannySettings::default(),
        )),
        9 => Some(PipelineCommand::EdgeDetectionSobel(
            EdgeDetectionSobelSettings::default(),
        )),
        10 => Some(PipelineCommand::EnhanceContrast(
            EnhanceContrastSettings::default(),
        )),
        11 => Some(PipelineCommand::ExtractObjects(
            ExtractObjectsSettings::default(),
        )),
        12 => Some(PipelineCommand::FillHoles(FillHolesSettings::default())),
        13 => Some(PipelineCommand::GaussianBlur(
            GaussianBlurSettings::default(),
        )),
        14 => Some(PipelineCommand::Hessian(HessianSettings::default())),
        15 => Some(PipelineCommand::IlluminationCorrection(
            IlluminationCorrectionSettings::default(),
        )),
        16 => Some(PipelineCommand::ImageCache(ImageCacheSettings::default())),
        17 => Some(PipelineCommand::ImageMath(ImageMathSettings::default())),
        18 => Some(PipelineCommand::IntensityTransformation(
            IntensityTransformationSettings::default(),
        )),
        19 => Some(PipelineCommand::Laplacian(LaplacianSettings::default())),
        20 => Some(PipelineCommand::MedianSubtract(
            MedianSubtractSettings::default(),
        )),
        21 => Some(PipelineCommand::MorphologicalCommand(
            MorphologicalCommandSettings::default(),
        )),
        22 => Some(PipelineCommand::ObjectMath(ObjectMathSettings::default())),
        23 => Some(PipelineCommand::PixelClassifier(
            PixelClassifierSettings::default(),
        )),
        24 => Some(PipelineCommand::RankFilter(RankFilterSettings::default())),
        25 => Some(PipelineCommand::RollingBall(RollingBallSettings::default())),
        26 => Some(PipelineCommand::SaveImage(SaveImageSettings::default())),
        27 => Some(PipelineCommand::Stardist(StardistSettings::default())),
        28 => Some(PipelineCommand::StructureTensor(
            StructureTensorSettings::default(),
        )),
        29 => Some(PipelineCommand::Threshold(ThresholdSettings::default())),
        30 => Some(PipelineCommand::TransformObjects(
            TransformObjectsSettings::default(),
        )),
        31 => Some(PipelineCommand::UNet(UNetSettings::default())),
        32 => Some(PipelineCommand::Voronoi(VoronoiSettings::default())),
        33 => Some(PipelineCommand::Watershed(WatershedSettings::default())),
        34 => Some(PipelineCommand::WeightedDeviation(
            WeightedDeviationSettings::default(),
        )),
        _ => None,
    }
}

#[allow(dead_code)]
impl PipelineCommand {
    pub fn name(&self) -> &str {
        match self {
            Self::AiObjectClassifier(_) => "AI Object Classifier",
            Self::Blur(_) => "Blur",
            Self::Cellpose(_) => "AI Cellpose Segmentation",
            Self::ClassifyObjects(_) => "ClassifyObjects",
            Self::Colocalization(_) => "Colocalization",
            Self::ColorFilterCommand(_) => "ColorFilterCommand",
            Self::ConnectedComponents(_) => "ConnectedComponents",
            Self::DistanceTransform(_) => "DistanceTransform",
            Self::EdgeDetectionCanny(_) => "EdgeDetectionCanny",
            Self::EdgeDetectionSobel(_) => "EdgeDetectionSobel",
            Self::EnhanceContrast(_) => "EnhanceContrast",
            Self::ExtractObjects(_) => "ExtractObjects",
            Self::FillHoles(_) => "FillHoles",
            Self::GaussianBlur(_) => "GaussianBlur",
            Self::Hessian(_) => "Hessian",
            Self::IlluminationCorrection(_) => "IlluminationCorrection",
            Self::ImageCache(_) => "ImageCache",
            Self::ImageMath(_) => "ImageMath",
            Self::IntensityTransformation(_) => "IntensityTransformation",
            Self::Laplacian(_) => "Laplacian",
            Self::MedianSubtract(_) => "MedianSubtract",
            Self::MorphologicalCommand(_) => "MorphologicalCommand",
            Self::ObjectMath(_) => "ObjectMath",
            Self::PixelClassifier(_) => "AI Pixel Classifier",
            Self::RankFilter(_) => "RankFilter",
            Self::RollingBall(_) => "RollingBall",
            Self::SaveImage(_) => "SaveImage",
            Self::Stardist(_) => "AI Stardist Segmentation",
            Self::StructureTensor(_) => "StructureTensor",
            Self::Threshold(_) => "Threshold",
            Self::TransformObjects(_) => "TransformObjects",
            Self::UNet(_) => "AI UNet Segmentation",
            Self::Voronoi(_) => "Voronoi",
            Self::Watershed(_) => "Watershed",
            Self::WeightedDeviation(_) => "WeightedDeviation",
        }
    }

    pub fn category(&self) -> &CommandCategory {
        match self {
            Self::AiObjectClassifier(_) => &CommandCategory::Classify,
            Self::Blur(_) => &CommandCategory::Preprocess,
            Self::Cellpose(_) => &CommandCategory::Segment,
            Self::ClassifyObjects(_) => &CommandCategory::Classify,
            Self::Colocalization(_) => &CommandCategory::Classify,
            Self::ColorFilterCommand(_) => &CommandCategory::Preprocess,
            Self::ConnectedComponents(_) => &CommandCategory::Object,
            Self::DistanceTransform(_) => &CommandCategory::Preprocess,
            Self::EdgeDetectionCanny(_) => &CommandCategory::Preprocess,
            Self::EdgeDetectionSobel(_) => &CommandCategory::Preprocess,
            Self::EnhanceContrast(_) => &CommandCategory::Preprocess,
            Self::ExtractObjects(_) => &CommandCategory::Measure,
            Self::FillHoles(_) => &CommandCategory::Object,
            Self::GaussianBlur(_) => &CommandCategory::Preprocess,
            Self::Hessian(_) => &CommandCategory::Preprocess,
            Self::IlluminationCorrection(_) => &CommandCategory::Preprocess,
            Self::ImageCache(_) => &CommandCategory::Preprocess,
            Self::ImageMath(_) => &CommandCategory::Preprocess,
            Self::IntensityTransformation(_) => &CommandCategory::Preprocess,
            Self::Laplacian(_) => &CommandCategory::Preprocess,
            Self::MedianSubtract(_) => &CommandCategory::Preprocess,
            Self::MorphologicalCommand(_) => &CommandCategory::Preprocess,
            Self::ObjectMath(_) => &CommandCategory::Classify,
            Self::PixelClassifier(_) => &CommandCategory::Segment,
            Self::RankFilter(_) => &CommandCategory::Preprocess,
            Self::RollingBall(_) => &CommandCategory::Preprocess,
            Self::SaveImage(_) => &CommandCategory::Preprocess,
            Self::Stardist(_) => &CommandCategory::Segment,
            Self::StructureTensor(_) => &CommandCategory::Preprocess,
            Self::Threshold(_) => &CommandCategory::Segment,
            Self::TransformObjects(_) => &CommandCategory::Classify,
            Self::UNet(_) => &CommandCategory::Segment,
            Self::Voronoi(_) => &CommandCategory::Classify,
            Self::Watershed(_) => &CommandCategory::Object,
            Self::WeightedDeviation(_) => &CommandCategory::Preprocess,
        }
    }

    /// Categories that may be inserted immediately after this command.
    pub fn allowed_next(&self) -> &'static [CommandCategory] {
        match self {
            Self::AiObjectClassifier(_) => &[CommandCategory::Classify],
            Self::Blur(_) => &[CommandCategory::Segment, CommandCategory::Preprocess],
            Self::Cellpose(_) => &[CommandCategory::Measure],
            Self::ClassifyObjects(_) => &[CommandCategory::Classify],
            Self::Colocalization(_) => &[CommandCategory::Classify],
            Self::ColorFilterCommand(_) => &[CommandCategory::Segment, CommandCategory::Preprocess],
            Self::ConnectedComponents(_) => &[CommandCategory::Object, CommandCategory::Measure],
            Self::DistanceTransform(_) => &[CommandCategory::Segment, CommandCategory::Preprocess],
            Self::EdgeDetectionCanny(_) => &[CommandCategory::Segment, CommandCategory::Preprocess],
            Self::EdgeDetectionSobel(_) => &[CommandCategory::Segment, CommandCategory::Preprocess],
            Self::EnhanceContrast(_) => &[CommandCategory::Segment, CommandCategory::Preprocess],
            Self::ExtractObjects(_) => &[CommandCategory::Classify],
            Self::FillHoles(_) => &[CommandCategory::Object],
            Self::GaussianBlur(_) => &[CommandCategory::Segment, CommandCategory::Preprocess],
            Self::Hessian(_) => &[CommandCategory::Segment, CommandCategory::Preprocess],
            Self::IlluminationCorrection(_) => {
                &[CommandCategory::Segment, CommandCategory::Preprocess]
            }
            Self::ImageCache(_) => &[CommandCategory::Segment, CommandCategory::Preprocess],
            Self::ImageMath(_) => &[CommandCategory::Segment, CommandCategory::Preprocess],
            Self::IntensityTransformation(_) => {
                &[CommandCategory::Segment, CommandCategory::Preprocess]
            }
            Self::Laplacian(_) => &[CommandCategory::Segment, CommandCategory::Preprocess],
            Self::MedianSubtract(_) => &[CommandCategory::Segment, CommandCategory::Preprocess],
            Self::MorphologicalCommand(_) => {
                &[CommandCategory::Segment, CommandCategory::Preprocess]
            }
            Self::ObjectMath(_) => &[CommandCategory::Classify],
            Self::PixelClassifier(_) => &[CommandCategory::Object],
            Self::RankFilter(_) => &[CommandCategory::Segment, CommandCategory::Preprocess],
            Self::RollingBall(_) => &[CommandCategory::Segment, CommandCategory::Preprocess],
            Self::SaveImage(_) => &[CommandCategory::Segment, CommandCategory::Preprocess],
            Self::Stardist(_) => &[CommandCategory::Measure],
            Self::StructureTensor(_) => &[CommandCategory::Segment, CommandCategory::Preprocess],
            Self::Threshold(_) => &[CommandCategory::Object],
            Self::TransformObjects(_) => &[CommandCategory::Classify],
            Self::UNet(_) => &[CommandCategory::Object],
            Self::Voronoi(_) => &[CommandCategory::Classify],
            Self::Watershed(_) => &[CommandCategory::Measure],
            Self::WeightedDeviation(_) => &[CommandCategory::Segment, CommandCategory::Preprocess],
        }
    }

    pub fn to_parameters(&self) -> Vec<ParameterDef> {
        match self {
            Self::AiObjectClassifier(_s) => [vec![ParameterDef { name: "model_path".to_string(), display_name: "Model Path".to_string(), description: "Path to a trained object classifier model, saved from the AI training dialog.".to_string(), value: _s.model_path.display().to_string(), param_type: ParamType::FilePath, options: vec!["evamodel".to_string()], min: 0.0f32, max: 0.0f32, step: 1.0000f32, groups: vec![] }], vec![ParameterDef { name: "segmentation_mapping".to_string(), display_name: "Segmentation Mapping".to_string(), description: "Maps the model's predicted classes to this project's object classes.".to_string(), value: String::new(), param_type: ParamType::Group, options: vec![], min: 0.0f32, max: 0.0f32, step: 1.0000f32, groups: _s.segmentation_mapping.iter().map(|__item| [vec![ParameterDef { name: "object_class".to_string(), display_name: "Object Class".to_string(), description: "Object class predicted by the classifier model.".to_string(), value: match __item.object_class.to_u32() { Some(v) => format!("{}", v), None => "-1".to_string() }, param_type: ParamType::ObjClass, options: vec![], min: 0.0f32, max: 0.0f32, step: 1.0000f32, groups: vec![] }], vec![ParameterDef { name: "output_class".to_string(), display_name: "Output Class".to_string(), description: "The project's own object class objects predicted as `object_class`\nare assigned.".to_string(), value: match __item.output_class.to_u32() { Some(v) => format!("{}", v), None => "-1".to_string() }, param_type: ParamType::ObjClass, options: vec![], min: 0.0f32, max: 0.0f32, step: 1.0000f32, groups: vec![] }]].concat()).collect() }], vec![ParameterDef { name: "input_classes".to_string(), display_name: "Input Classes".to_string(), description: "Restrict classification to objects that already carry one of these classes\n\nOnly ROIs that have been assigned at least one of the listed classes by a prior\npipeline step will be evaluated by the model. Leave empty to apply the model to\nevery object regardless of its current class.".to_string(), value: _s.input_classes.iter().filter_map(|c| c.to_u32()).map(|v| v.to_string()).collect::<Vec<_>>().join(","), param_type: ParamType::MultiObjClass, options: (0u32..33u32).map(|__idx| if _s.input_classes.iter().any(|c| c.to_u32().map_or(false, |v| v == __idx)) { "1".to_string() } else { "0".to_string() }).collect::<Vec<_>>(), min: 0.0f32, max: 0.0f32, step: 1.0000f32, groups: vec![] }], vec![ParameterDef { name: "match_handling".to_string(), display_name: "Match Handling".to_string(), description: "What to do with object class labels after prediction\n\n- **AddOutputClassIfMatch** - append the mapped class alongside the object's existing classes.\n- **ReclassifyIfMatch** - clear every class the object carries and assign only the mapped class.".to_string(), value: match _s.match_handling { ClassificationAiObjectClassifierAiClassifyMatchHandlingSettings::AddOutputClassIfMatch => "Add class on match".to_string(), ClassificationAiObjectClassifierAiClassifyMatchHandlingSettings::ReclassifyIfMatch => "Reclassify on match".to_string() }, param_type: ParamType::Dropdown, options: vec!["Add class on match".to_string(), "Reclassify on match".to_string()], min: 0.0f32, max: 0.0f32, step: 1.0000f32, groups: vec![] }]].concat(),
            Self::Blur(_s) => vec![ParameterDef { name: "kernel_size".to_string(), display_name: "Kernel size".to_string(), description: "The size of the blur matrix.\n\nMust be an odd number (e.g., 3, 5, 7)".to_string(), value: format!("{}", _s.kernel_size), param_type: ParamType::Spinner, options: vec![], min: 3.0f32, max: 27.0f32, step: 2.0000f32, groups: vec![] }],
            Self::Cellpose(_s) => [vec![ParameterDef { name: "model_path".to_string(), display_name: "Model Path".to_string(), description: "Path to a TorchScript-exported Cellpose model (`torch.jit.script`/`torch.jit.trace`).".to_string(), value: _s.model_path.display().to_string(), param_type: ParamType::FilePath, options: vec!["pt,pth".to_string()], min: 0.0f32, max: 0.0f32, step: 1.0000f32, groups: vec![] }], vec![ParameterDef { name: "object_class_id".to_string(), display_name: "Object Class Id".to_string(), description: "The class assigned to pixels of every detected object. All other\npixels are assigned `SegmentationClass::BACKGROUND`.".to_string(), value: format!("{}", _s.object_class_id.as_u32()), param_type: ParamType::SegClass, options: vec![], min: 0.0f32, max: 0.0f32, step: 1.0000f32, groups: vec![] }], vec![ParameterDef { name: "input_channels".to_string(), display_name: "Input Channels".to_string(), description: "Number of input channels the model expects. The grayscale image goes in\nchannel 0; any further channels are zero-filled. Standard Cellpose models\ntake `2` (cytoplasm + optional nucleus); set `1` for single-channel\nexports, or higher to match a custom model.".to_string(), value: format!("{}", _s.input_channels), param_type: ParamType::Dropdown, options: vec!["1".to_string(), "2".to_string(), "3".to_string(), "4".to_string(), "5".to_string(), "6".to_string(), "7".to_string(), "8".to_string()], min: 1.0f32, max: 8.0f32, step: 1.0000f32, groups: vec![] }], vec![ParameterDef { name: "probability_threshold".to_string(), display_name: "Probability Threshold".to_string(), description: "Cell probability above which a pixel takes part in the flow dynamics and\ncan be assigned to an object. The raw cell-probability logits are passed\nthrough a sigmoid first, so this is a probability in `[0, 1]` (Cellpose's\ndefault logit threshold of `0` corresponds to `0.5`).".to_string(), value: format!("{}", _s.probability_threshold), param_type: ParamType::Spinner, options: vec![], min: 0.0f32, max: 1.0f32, step: 0.0100f32, groups: vec![] }], vec![ParameterDef { name: "flow_iterations".to_string(), display_name: "Flow Iterations".to_string(), description: "Number of Euler integration steps used to follow the flow field. Higher\nvalues let pixels of large cells reach their sink at the cost of runtime;\nCellpose's default is `200`.".to_string(), value: format!("{}", _s.flow_iterations), param_type: ParamType::Spinner, options: vec![], min: 1.0f32, max: 1000.0f32, step: 1.0000f32, groups: vec![] }], vec![ParameterDef { name: "min_object_size".to_string(), display_name: "Min Object Size".to_string(), description: "Minimum object size, in pixels. After the dynamics, any instance smaller\nthan this is removed (its pixels become background). `0` disables the filter.".to_string(), value: format!("{}", _s.min_object_size), param_type: ParamType::Spinner, options: vec![], min: 0.0f32, max: 100000.0f32, step: 1.0000f32, groups: vec![] }]].concat(),
            Self::ClassifyObjects(_s) => [vec![ParameterDef { name: "input_classes".to_string(), display_name: "Input Classes".to_string(), description: "Restrict classification to objects that already carry one of these classes\n\nOnly ROIs that have been assigned at least one of the listed classes by a prior\npipeline step will be evaluated against the morphological and intensity criteria below.\nLeave empty to apply the criteria to every object regardless of its current class.".to_string(), value: _s.input_classes.iter().filter_map(|c| c.to_u32()).map(|v| v.to_string()).collect::<Vec<_>>().join(","), param_type: ParamType::MultiObjClass, options: (0u32..33u32).map(|__idx| if _s.input_classes.iter().any(|c| c.to_u32().map_or(false, |v| v == __idx)) { "1".to_string() } else { "0".to_string() }).collect::<Vec<_>>(), min: 0.0f32, max: 0.0f32, step: 1.0000f32, groups: vec![] }], vec![ParameterDef { name: "match_handling".to_string(), display_name: "Match Handling".to_string(), description: "What to do with object class labels after criteria evaluation\n\nControls whether the output class is added or existing classes are removed,\nand whether the action is triggered on a criteria **match** or a **non-match**:\n\n- **AddOutputClassIfMatch** - append the output class to objects that pass the criteria.\n- **AddOutputClassIfNotMatch** - append the output class to objects that fail the criteria.\n- **RemoveInputClassIfMatch / NotMatch** - strip all input classes from matching / non-matching objects.\n- **RemoveOutputClassIfMatch / NotMatch** - strip the output class from matching / non-matching objects.\n- **RemoveAllClassesIfMatch / NotMatch** - clear every class label from matching / non-matching objects.".to_string(), value: match _s.match_handling { ClassificationClassifyObjectsClassifyMatchHandlingSettings::AddOutputClassIfMatch => "Add class on match".to_string(), ClassificationClassifyObjectsClassifyMatchHandlingSettings::AddOutputClassIfNotMatch => "Add class on mismatch".to_string(), ClassificationClassifyObjectsClassifyMatchHandlingSettings::RemoveInputClassIfMatch => "Remove class on match".to_string(), ClassificationClassifyObjectsClassifyMatchHandlingSettings::RemoveInputClassIfNotMatch => "Remove class on mismatch".to_string(), ClassificationClassifyObjectsClassifyMatchHandlingSettings::RemoveOutputClassIfMatch => "Remove output class on match".to_string(), ClassificationClassifyObjectsClassifyMatchHandlingSettings::RemoveOutputClassIfNotMatch => "Remove output class on mismatch".to_string(), ClassificationClassifyObjectsClassifyMatchHandlingSettings::RemoveAllClassesIfMatch => "Remove objects matching criteria".to_string(), ClassificationClassifyObjectsClassifyMatchHandlingSettings::RemoveAllClassesIfNotMatch => "Keep objects matching criteria".to_string(), ClassificationClassifyObjectsClassifyMatchHandlingSettings::ReclassifyIfMatch => "Reclassify on match".to_string(), ClassificationClassifyObjectsClassifyMatchHandlingSettings::ReclassifyIfNotMatch => "Reclassify on mismatch".to_string() }, param_type: ParamType::Dropdown, options: vec!["Add class on match".to_string(), "Add class on mismatch".to_string(), "Remove class on match".to_string(), "Remove class on mismatch".to_string(), "Remove output class on match".to_string(), "Remove output class on mismatch".to_string(), "Remove objects matching criteria".to_string(), "Keep objects matching criteria".to_string(), "Reclassify on match".to_string(), "Reclassify on mismatch".to_string()], min: 0.0f32, max: 0.0f32, step: 1.0000f32, groups: vec![] }], vec![ParameterDef { name: "output_class".to_string(), display_name: "Output Tag".to_string(), description: "Class label assigned to (or removed from) objects by the chosen operation\n\nUsed as the target class for `AddOutputClass*` and `RemoveOutputClass*` operations.\nHas no effect when the selected operation only manipulates input classes or clears all classes.".to_string(), value: match _s.output_class.to_u32() { Some(v) => format!("{}", v), None => "-1".to_string() }, param_type: ParamType::ObjClass, options: vec![], min: 0.0f32, max: 0.0f32, step: 1.0000f32, groups: vec![] }], vec![ParameterDef { name: "overlapping_with".to_string(), display_name: "Intersecting With".to_string(), description: "Additional criterion: the object must intersect an object carrying this class\n\nIf unset (the default) this filter is not applied. When set, an object only\nsatisfies the overall criteria if it also overlaps at least one object carrying this\nclass by at least `min_intersection_area`. Combine with e.g.\n`RemoveAllClassesIfMatch` to drop objects that intersect another class's objects,\nor `AddOutputClassIfMatch` to tag objects that do.".to_string(), value: match _s.overlapping_with.to_u32() { Some(v) => format!("{}", v), None => "-1".to_string() }, param_type: ParamType::ObjClass, options: vec![], min: 0.0f32, max: 0.0f32, step: 1.0000f32, groups: vec![] }], vec![ParameterDef { name: "min_intersection_area".to_string(), display_name: "Min Intersection Area".to_string(), description: "Minimum intersection area with an `overlapping_with` object, in `size_unit`\n\nHas no effect while `overlapping_with` is Unset.".to_string(), value: format!("{}", _s.min_intersection_area), param_type: ParamType::Spinner, options: vec![], min: 0.0f32, max: 2147483648.0f32, step: 1.0000f32, groups: vec![] }], vec![ParameterDef { name: "size_unit".to_string(), display_name: "Size Unit".to_string(), description: "Unit to use for object extraction".to_string(), value: match _s.size_unit { SizeUnits::NanoMeter => "nm".to_string(), SizeUnits::Pixels => "px".to_string() }, param_type: ParamType::SizeUnits, options: vec!["nm".to_string(), "px".to_string()], min: 0.0f32, max: 0.0f32, step: 1.0000f32, groups: vec![] }], vec![ParameterDef { name: "min_area".to_string(), display_name: "Min Area".to_string(), description: "Minimum area size\n\nMinimum area size of the object in selected unit (px^2 or nm^2).".to_string(), value: format!("{}", _s.min_area), param_type: ParamType::Spinner, options: vec![], min: 0.0f32, max: 2147483648.0f32, step: 1.0000f32, groups: vec![] }], vec![ParameterDef { name: "max_area".to_string(), display_name: "Max Area".to_string(), description: "Maximum area size\n\nMaximum area size of the object in selected unit (px^2 or nm^2).".to_string(), value: format!("{}", _s.max_area), param_type: ParamType::Spinner, options: vec![], min: 0.0f32, max: 2147483648.0f32, step: 1.0000f32, groups: vec![] }], vec![ParameterDef { name: "min_circularity".to_string(), display_name: "Min Circularity".to_string(), description: "Circularity range: 0 = elongated, 1 = perfect circle\n\nCircularity (sometimes called Isoperimetric Quotient) measures how efficiently a shape encloses its area relative to the length of its perimeter.\nA circle is the mathematically perfect shape for maximizing area while minimizing perimeter.\nIt is calculated with `4*Pi*AreaSize / Perimeter^2`".to_string(), value: format!("{}", _s.min_circularity), param_type: ParamType::Spinner, options: vec![], min: 0.0f32, max: 1.0f32, step: 0.1000f32, groups: vec![] }], vec![ParameterDef { name: "max_circularity".to_string(), display_name: "Max Circularity".to_string(), description: "Circularity range: 0 = elongated, 1 = perfect circle\n\nCircularity (sometimes called Isoperimetric Quotient) measures how efficiently a shape encloses its area relative to the length of its perimeter.\nA circle is the mathematically perfect shape for maximizing area while minimizing perimeter.\nIt is calculated with `4*Pi*AreaSize / Perimeter^2`".to_string(), value: format!("{}", _s.max_circularity), param_type: ParamType::Spinner, options: vec![], min: 0.0f32, max: 1.0f32, step: 0.1000f32, groups: vec![] }], vec![ParameterDef { name: "min_solidity".to_string(), display_name: "Min Solidity".to_string(), description: "Minimum Solidity/Compactness: 0 = hollow, 1 = perfect convex\n\nSolidity is a structural metric used in shape analysis to measure how \"solid\" or compact an object is.\nIt compares the actual area of an object to the area of its Convex Hull (the smallest convex polygon that can completely enclose the object,\noften visualized as a rubber band stretched around the shape).\n\nSolidity = 1.0: The object is perfectly convex (e.g., a perfect circle, a solid square, or an ellipse). It has no holes, indentations, or deep recesses.\nSolidity < 1.0: The object has irregular boundaries, deep \"bays,\" protrusions, or internal holes. The lower the value, the more jagged or structurally fragmented the object is.".to_string(), value: format!("{}", _s.min_solidity), param_type: ParamType::Spinner, options: vec![], min: 0.0f32, max: 1.0f32, step: 0.1000f32, groups: vec![] }], vec![ParameterDef { name: "max_solidity".to_string(), display_name: "Max Solidity".to_string(), description: "Maximum Solidity/Compactness: 0 = hollow, 1 = perfect convex\n\nSolidity is a structural metric used in shape analysis to measure how \"solid\" or compact an object is.\nIt compares the actual area of an object to the area of its Convex Hull (the smallest convex polygon that can completely enclose the object,\noften visualized as a rubber band stretched around the shape).\n\nSolidity = 1.0: The object is perfectly convex (e.g., a perfect circle, a solid square, or an ellipse). It has no holes, indentations, or deep recesses.\nSolidity < 1.0: The object has irregular boundaries, deep \"bays,\" protrusions, or internal holes. The lower the value, the more jagged or structurally fragmented the object is.".to_string(), value: format!("{}", _s.max_solidity), param_type: ParamType::Spinner, options: vec![], min: 0.0f32, max: 1.0f32, step: 0.1000f32, groups: vec![] }], vec![ParameterDef { name: "min_aspect_ratio".to_string(), display_name: "Min Aspect Ratio".to_string(), description: "Minimum proportional relationship between an object's width and its height\n\nThis value is calculated by the object bounding box with and height and is defined with `a = with/height`.\nThe value is without unit in the range of 0 to MAX_F32".to_string(), value: format!("{}", _s.min_aspect_ratio), param_type: ParamType::Spinner, options: vec![], min: 0.0f32, max: 2147483648.0f32, step: 1.0000f32, groups: vec![] }], vec![ParameterDef { name: "max_aspect_ratio".to_string(), display_name: "Max Aspect Ratio".to_string(), description: "Maximum proportional relationship between an object's width and its height\n\nThis value is calculated by the object bounding box with and height and is defined with `a = with/height`.\nThe value is without unit in the range of 0 to MAX_F32".to_string(), value: format!("{}", _s.max_aspect_ratio), param_type: ParamType::Spinner, options: vec![], min: 0.0f32, max: 2147483648.0f32, step: 1.0000f32, groups: vec![] }], vec![ParameterDef { name: "min_eccentricity".to_string(), display_name: "Min Eccentricity".to_string(), description: "Eccentricity: 0 = perfect circle, 1 = line\n\nEccentricity is a metric that measures how much a shape deviates from being a perfect circle.\nIt imagines the shape as an ellipse and measures how far apart its focal points are.\nIt is calculated with `sqrt(1-(b/a)^2)`".to_string(), value: format!("{}", _s.min_eccentricity), param_type: ParamType::Spinner, options: vec![], min: 0.0f32, max: 1.0f32, step: 0.1000f32, groups: vec![] }], vec![ParameterDef { name: "max_eccentricity".to_string(), display_name: "Max Eccentricity".to_string(), description: "Eccentricity: 0 = perfect circle, 1 = line\n\nEccentricity is a metric that measures how much a shape deviates from being a perfect circle.\nIt imagines the shape as an ellipse and measures how far apart its focal points are.\nIt is calculated with `sqrt(1-(b/a)^2)`".to_string(), value: format!("{}", _s.max_eccentricity), param_type: ParamType::Spinner, options: vec![], min: 0.0f32, max: 1.0f32, step: 0.1000f32, groups: vec![] }], vec![ParameterDef { name: "min_feret".to_string(), display_name: "Min Feret".to_string(), description: "Feret diameter threshold\n\nThe absolute shortest parallel distance across the object.\nThis represents the minimum sieve size a particle could pass through.\n\nIn image processing and particle size analysis, the Feret diameter (often called the caliper diameter) is a metric used to measure the size of an irregular object.\nIt mimics the action of a slide caliper, measuring the distance between two parallel tangential lines bounding the object at a specific angle.\nWhen analyzing objects or particles, applying Feret diameter thresholds allows you to filter out noise, classify objects by shape, or isolate specific structures based on their directional length rather than their total area.".to_string(), value: format!("{}", _s.min_feret), param_type: ParamType::Spinner, options: vec![], min: 0.0f32, max: 2147483648.0f32, step: 1.0000f32, groups: vec![] }], vec![ParameterDef { name: "max_feret".to_string(), display_name: "Max Feret".to_string(), description: "Maximum feret diameter threshold in selected unit (px or nm)\n\nThe absolute longest distance across the object at any angle.\nUsed to measure elongation or the maximum length of a particle.\n\nIn image processing and particle size analysis, the Feret diameter (often called the caliper diameter) is a metric used to measure the size of an irregular object.\nIt mimics the action of a slide caliper, measuring the distance between two parallel tangential lines bounding the object at a specific angle.\nWhen analyzing objects or particles, applying Feret diameter thresholds allows you to filter out noise, classify objects by shape, or isolate specific structures based on their directional length rather than their total area.".to_string(), value: format!("{}", _s.max_feret), param_type: ParamType::Spinner, options: vec![], min: 0.0f32, max: 2147483648.0f32, step: 1.0000f32, groups: vec![] }], vec![ParameterDef { name: "allow_edge_touching".to_string(), display_name: "Allow Edge Touching".to_string(), description: "Whether object can touch image edge".to_string(), value: format!("{}", _s.allow_edge_touching), param_type: ParamType::Toggle, options: vec![], min: 0.0f32, max: 0.0f32, step: 1.0000f32, groups: vec![] }]].concat(),
            Self::Colocalization(_s) => [vec![ParameterDef { name: "classes_to_coloc".to_string(), display_name: "Classes To Coloc".to_string(), description: "Theses are the classes the coloclization should be calculated for".to_string(), value: _s.classes_to_coloc.iter().filter_map(|c| c.to_u32()).map(|v| v.to_string()).collect::<Vec<_>>().join(","), param_type: ParamType::MultiObjClass, options: (0u32..33u32).map(|__idx| if _s.classes_to_coloc.iter().any(|c| c.to_u32().map_or(false, |v| v == __idx)) { "1".to_string() } else { "0".to_string() }).collect::<Vec<_>>(), min: 0.0f32, max: 0.0f32, step: 1.0000f32, groups: vec![] }], vec![ParameterDef { name: "class_for_overlapping_areas".to_string(), display_name: "Class For Overlapping Areas".to_string(), description: "Class of the overlapping area if needed\n\nIf defined the overlapping coloc area is added as new object and labeled with this class".to_string(), value: match _s.class_for_overlapping_areas.to_u32() { Some(v) => format!("{}", v), None => "-1".to_string() }, param_type: ParamType::ObjClass, options: vec![], min: 0.0f32, max: 0.0f32, step: 1.0000f32, groups: vec![] }], vec![ParameterDef { name: "multiplicity".to_string(), display_name: "Multiplicity".to_string(), description: "How many partners an object may coloc with at once.".to_string(), value: match _s.multiplicity { ClassificationColocObjectsColocMultiplicitySettings::OneToOne => "No multi coloc (1:1)".to_string(), ClassificationColocObjectsColocMultiplicitySettings::ManyToMany => "Allow multi coloc".to_string(), ClassificationColocObjectsColocMultiplicitySettings::MultiFor(_) => "Multi coloc only for selected".to_string() }, param_type: ParamType::Dropdown, options: vec!["No multi coloc (1:1)".to_string(), "Allow multi coloc".to_string(), "Multi coloc only for selected".to_string()], min: 0.0f32, max: 0.0f32, step: 1.0000f32, groups: vec![] }], match &_s.multiplicity { ClassificationColocObjectsColocMultiplicitySettings::OneToOne => vec![], ClassificationColocObjectsColocMultiplicitySettings::ManyToMany => vec![], ClassificationColocObjectsColocMultiplicitySettings::MultiFor(__inner) => vec![ParameterDef { name: "multiplicity.0".to_string(), display_name: "Multi coloc only for selected".to_string(), description: "Only objects of these classes may coloc with more than one partner;\nevery other class in `classes_to_coloc` is capped to its single\nbest-overlap match. E.g. with `classes_to_coloc: [Cell, Spot]` and\n`MultiFor([Cell])`, a cell can coloc with any number of spots, but\neach spot colocs with exactly one cell (the one it overlaps most).".to_string(), value: __inner.iter().filter_map(|c| c.to_u32()).map(|v| v.to_string()).collect::<Vec<_>>().join(","), param_type: ParamType::MultiObjClass, options: (0u32..33u32).map(|__idx| if __inner.iter().any(|c| c.to_u32().map_or(false, |v| v == __idx)) { "1".to_string() } else { "0".to_string() }).collect::<Vec<_>>(), min: 0.0f32, max: 0.0f32, step: 1.0000f32, groups: vec![] }] }, vec![ParameterDef { name: "size_unit".to_string(), display_name: "Size Unit".to_string(), description: "Size unit for the minimum coloc area size".to_string(), value: match _s.size_unit { SizeUnits::NanoMeter => "nm".to_string(), SizeUnits::Pixels => "px".to_string() }, param_type: ParamType::SizeUnits, options: vec!["nm".to_string(), "px".to_string()], min: 0.0f32, max: 0.0f32, step: 1.0000f32, groups: vec![] }], vec![ParameterDef { name: "min_coloc_area".to_string(), display_name: "Min Coloc Area".to_string(), description: "Minimum overlapping area size to count objects as coloc".to_string(), value: format!("{}", _s.min_coloc_area), param_type: ParamType::Number, options: vec![], min: 0.0f32, max: 0.0f32, step: 1.0000f32, groups: vec![] }], vec![ParameterDef { name: "exclude_classes".to_string(), display_name: "Exclude Classes".to_string(), description: "Classes an object must NOT overlap to be considered colocalized.\n\nExclude_classes is a blocklist — \"even if an object matches everything else, throw it out if it also touches one of these classes.\"\nConcretely: you're looking for objects that overlap every class in classes_to_coloc (say Class 1 and Class 2).\nWithout exclude_classes, any object satisfying that gets recorded as colocalized.\nWith exclude_classes: an object that overlaps 1 and 2 but also touches Class 3 gets dropped entirely - no colocalization recorded for it at all, even though it passed the 1-and-2 check.\nSo it's a \"match A and B, but not C\" filter\n\nExample: \"cells colocalizing with both a nucleus stain and a membrane stain, but exclude any that also overlap a dead-cell marker.\"".to_string(), value: _s.exclude_classes.iter().filter_map(|c| c.to_u32()).map(|v| v.to_string()).collect::<Vec<_>>().join(","), param_type: ParamType::MultiObjClass, options: (0u32..33u32).map(|__idx| if _s.exclude_classes.iter().any(|c| c.to_u32().map_or(false, |v| v == __idx)) { "1".to_string() } else { "0".to_string() }).collect::<Vec<_>>(), min: 0.0f32, max: 0.0f32, step: 1.0000f32, groups: vec![] }]].concat(),
            Self::ColorFilterCommand(_s) => [vec![ParameterDef { name: "range.min_h".to_string(), display_name: "Min H".to_string(), description: "Minimum Hue angle in degrees [0.0, 360.0].".to_string(), value: format!("{}", _s.range.min_h), param_type: ParamType::Number, options: vec![], min: 0.0f32, max: 0.0f32, step: 1.0000f32, groups: vec![] }], vec![ParameterDef { name: "range.max_h".to_string(), display_name: "Max H".to_string(), description: "Maximum Hue angle in degrees [0.0, 360.0].".to_string(), value: format!("{}", _s.range.max_h), param_type: ParamType::Number, options: vec![], min: 0.0f32, max: 0.0f32, step: 1.0000f32, groups: vec![] }], vec![ParameterDef { name: "range.min_s".to_string(), display_name: "Min S".to_string(), description: "Minimum Saturation normalized [0.0, 1.0].".to_string(), value: format!("{}", _s.range.min_s), param_type: ParamType::Number, options: vec![], min: 0.0f32, max: 0.0f32, step: 1.0000f32, groups: vec![] }], vec![ParameterDef { name: "range.max_s".to_string(), display_name: "Max S".to_string(), description: "Maximum Saturation normalized [0.0, 1.0].".to_string(), value: format!("{}", _s.range.max_s), param_type: ParamType::Number, options: vec![], min: 0.0f32, max: 0.0f32, step: 1.0000f32, groups: vec![] }], vec![ParameterDef { name: "range.min_v".to_string(), display_name: "Min V".to_string(), description: "Minimum Value (Brightness) normalized [0.0, 1.0].".to_string(), value: format!("{}", _s.range.min_v), param_type: ParamType::Number, options: vec![], min: 0.0f32, max: 0.0f32, step: 1.0000f32, groups: vec![] }], vec![ParameterDef { name: "range.max_v".to_string(), display_name: "Max V".to_string(), description: "Maximum Value (Brightness) normalized [0.0, 1.0].".to_string(), value: format!("{}", _s.range.max_v), param_type: ParamType::Number, options: vec![], min: 0.0f32, max: 0.0f32, step: 1.0000f32, groups: vec![] }]].concat(),
            Self::ConnectedComponents(_s) => vec![ParameterDef { name: "min_size".to_string(), display_name: "Min Size".to_string(), description: "Minimum object size, in pixels, an object must have to be kept.\n\nAfter labeling, connected components with a pixel count below this\nthreshold are discarded (their pixels are reset to background) and\nthe remaining object IDs are re-compacted to a contiguous range.\nUseful for suppressing noise/speckle artifacts before they reach\ndownstream measurement or classification steps. A value of 0 (the\ndefault) disables filtering.".to_string(), value: format!("{}", _s.min_size), param_type: ParamType::Spinner, options: vec![], min: 1.0f32, max: 0.0f32, step: 1.0000f32, groups: vec![] }],
            Self::DistanceTransform(_s) => [vec![ParameterDef { name: "threshold".to_string(), display_name: "Threshold".to_string(), description: "Values less than or equal to this are treated as background (distance = 0).".to_string(), value: format!("{}", _s.threshold), param_type: ParamType::Number, options: vec![], min: 0.0f32, max: 0.0f32, step: 1.0000f32, groups: vec![] }], vec![ParameterDef { name: "edges_are_background".to_string(), display_name: "Edges Are Background".to_string(), description: "If true, the pixels outside the image boundary are treated as background.".to_string(), value: format!("{}", _s.edges_are_background), param_type: ParamType::Toggle, options: vec![], min: 0.0f32, max: 0.0f32, step: 1.0000f32, groups: vec![] }]].concat(),
            Self::EdgeDetectionCanny(_s) => [vec![ParameterDef { name: "kernel_size".to_string(), display_name: "Kernel Size".to_string(), description: "Size of the Gaussian smoothing kernel.\n\nMust be an odd number (e.g., 3, 5). Larger values reduce\nnoise but can blur fine edge details.".to_string(), value: format!("{}", _s.kernel_size), param_type: ParamType::Number, options: vec![], min: 0.0f32, max: 0.0f32, step: 1.0000f32, groups: vec![] }], vec![ParameterDef { name: "threshold_min".to_string(), display_name: "Threshold Min".to_string(), description: "Lower bound for hysteresis thresholding [0.0, 1.0].\n\nEdges with a gradient intensity below this value are discarded.".to_string(), value: format!("{}", _s.threshold_min), param_type: ParamType::Number, options: vec![], min: 0.0f32, max: 0.0f32, step: 1.0000f32, groups: vec![] }], vec![ParameterDef { name: "threshold_max".to_string(), display_name: "Threshold Max".to_string(), description: "Upper bound for hysteresis thresholding [0.0, 1.0].\n\nEdges with a gradient intensity above this value are considered\n\"strong\" and are automatically preserved.".to_string(), value: format!("{}", _s.threshold_max), param_type: ParamType::Number, options: vec![], min: 0.0f32, max: 0.0f32, step: 1.0000f32, groups: vec![] }]].concat(),
            Self::EdgeDetectionSobel(_s) => vec![ParameterDef { name: "kernel_size".to_string(), display_name: "Kernel Size".to_string(), description: "The size of the Sobel operator window.\n\nTypically 3. Larger values (5, 7) provide a more smoothed\ngradient but result in \"thicker\" edges. Must be an odd number.".to_string(), value: format!("{}", _s.kernel_size), param_type: ParamType::Number, options: vec![], min: 0.0f32, max: 0.0f32, step: 1.0000f32, groups: vec![] }],
            Self::EnhanceContrast(_s) => [vec![ParameterDef { name: "saturated_pixels".to_string(), display_name: "Saturated Pixels".to_string(), description: "Percentage of pixels to \"clip\" from the top and bottom of the histogram.\n\nRange: [0.0, 1.0]. A value of 0.01 (1%) helps ignore hot/dead pixels\nthat would otherwise prevent effective contrast stretching.".to_string(), value: format!("{}", _s.saturated_pixels), param_type: ParamType::Number, options: vec![], min: 0.0f32, max: 0.0f32, step: 1.0000f32, groups: vec![] }], vec![ParameterDef { name: "normalize".to_string(), display_name: "Normalize".to_string(), description: "Whether to linearly stretch the remaining pixel intensities to fill\nthe full [0.0, 1.0] range.".to_string(), value: format!("{}", _s.normalize), param_type: ParamType::Toggle, options: vec![], min: 0.0f32, max: 0.0f32, step: 1.0000f32, groups: vec![] }], vec![ParameterDef { name: "equalize_histogram".to_string(), display_name: "Equalize Histogram".to_string(), description: "Whether to apply Histogram Equalization.\n\nThis redistributes pixel intensities to achieve a uniform distribution,\nwhich is highly effective for images with low contrast but high noise.".to_string(), value: format!("{}", _s.equalize_histogram), param_type: ParamType::Toggle, options: vec![], min: 0.0f32, max: 0.0f32, step: 1.0000f32, groups: vec![] }]].concat(),
            Self::ExtractObjects(_s) => vec![ParameterDef { name: "max_objects_before_fail".to_string(), display_name: "Max Objects Before Fail".to_string(), description: "Maximum allowed ROIs to extract.\n\nIf this limit is exceeded the pipeline fails.\nThis is a protection against memory overload.".to_string(), value: format!("{}", _s.max_objects_before_fail), param_type: ParamType::Label, options: vec![], min: 0.0f32, max: 0.0f32, step: 1.0000f32, groups: vec![] }],
            Self::FillHoles(_s) => vec![],
            Self::GaussianBlur(_s) => [vec![ParameterDef { name: "kernel_size".to_string(), display_name: "Kernel Size".to_string(), description: "The size of the blur matrix.\n\nMust be an odd number (e.g., 3, 5, 7).".to_string(), value: format!("{}", _s.kernel_size), param_type: ParamType::Spinner, options: vec![], min: 3.0f32, max: 27.0f32, step: 2.0000f32, groups: vec![] }], vec![ParameterDef { name: "sigma".to_string(), display_name: "Sigma".to_string(), description: "The standard deviation of the Gaussian kernel.\n\nHigher values create a more significant blur effect.\n$$N \\approx 6\\sigma + 1$$".to_string(), value: format!("{}", _s.sigma), param_type: ParamType::Spinner, options: vec![], min: 0.1f32, max: 5.0f32, step: 0.1000f32, groups: vec![] }]].concat(),
            Self::Hessian(_s) => vec![ParameterDef { name: "mode".to_string(), display_name: "Mode".to_string(), description: "Determines which component of the Hessian matrix structure to extract.\n\nDepending on the mode, this can highlight interest points (blobs)\nor directional features (ridges).".to_string(), value: match _s.mode { FiltersHessianHessianModeSettings::Determinant => "Determinant".to_string(), FiltersHessianHessianModeSettings::EigenvaluesX => "Eigenvalues X".to_string(), FiltersHessianHessianModeSettings::EigenvaluesY => "Eigenvalues Y".to_string() }, param_type: ParamType::Dropdown, options: vec!["Determinant".to_string(), "Eigenvalues X".to_string(), "Eigenvalues Y".to_string()], min: 0.0f32, max: 0.0f32, step: 1.0000f32, groups: vec![] }],
            Self::IlluminationCorrection(_s) => [vec![ParameterDef { name: "method".to_string(), display_name: "Method".to_string(), description: "How the illumination field is estimated from the image.".to_string(), value: match _s.method { FiltersIlluminationCorrectionCorrectionMethodSettings::Regular => "Regular".to_string(), FiltersIlluminationCorrectionCorrectionMethodSettings::Background => "Background".to_string() }, param_type: ParamType::Dropdown, options: vec!["Regular".to_string(), "Background".to_string()], min: 0.0f32, max: 0.0f32, step: 1.0000f32, groups: vec![] }], vec![ParameterDef { name: "block_size".to_string(), display_name: "Block Size".to_string(), description: "Block size, in pixels, used to reduce the image to a coarse\nillumination estimate before smoothing. Should be larger than the\nlargest foreground object, so objects are averaged/eroded away and\nonly the slow-varying illumination trend survives.".to_string(), value: format!("{}", _s.block_size), param_type: ParamType::Spinner, options: vec![], min: 1.0f32, max: 2000.0f32, step: 1.0000f32, groups: vec![] }], vec![ParameterDef { name: "smoothing".to_string(), display_name: "Smoothing".to_string(), description: "Smoothing applied to the block-reduced field to remove blockiness.".to_string(), value: match _s.smoothing { FiltersIlluminationCorrectionSmoothingMethodSettings::None => "None".to_string(), FiltersIlluminationCorrectionSmoothingMethodSettings::Gaussian { .. } => "Gaussian".to_string(), FiltersIlluminationCorrectionSmoothingMethodSettings::Median { .. } => "Median".to_string(), FiltersIlluminationCorrectionSmoothingMethodSettings::FitPolynomial => "Fit Polynomial".to_string() }, param_type: ParamType::Dropdown, options: vec!["None".to_string(), "Gaussian".to_string(), "Median".to_string(), "Fit Polynomial".to_string()], min: 0.0f32, max: 0.0f32, step: 1.0000f32, groups: vec![] }], match &_s.smoothing { FiltersIlluminationCorrectionSmoothingMethodSettings::None => vec![], FiltersIlluminationCorrectionSmoothingMethodSettings::Gaussian { sigma } => vec![ParameterDef { name: "smoothing.sigma".to_string(), display_name: "Sigma".to_string(), description: "Standard deviation, in block-grid units.".to_string(), value: format!("{}", sigma), param_type: ParamType::Spinner, options: vec![], min: 0.1f32, max: 20.0f32, step: 0.1000f32, groups: vec![] }], FiltersIlluminationCorrectionSmoothingMethodSettings::Median { radius } => vec![ParameterDef { name: "smoothing.radius".to_string(), display_name: "Radius".to_string(), description: "Neighborhood radius, in block-grid units.".to_string(), value: format!("{}", radius), param_type: ParamType::Spinner, options: vec![], min: 1.0f32, max: 20.0f32, step: 1.0000f32, groups: vec![] }], FiltersIlluminationCorrectionSmoothingMethodSettings::FitPolynomial => vec![] }, vec![ParameterDef { name: "apply_method".to_string(), display_name: "Apply Method".to_string(), description: "How the field is combined with the original image.".to_string(), value: match _s.apply_method { FiltersIlluminationCorrectionApplyMethodSettings::Divide => "Divide".to_string(), FiltersIlluminationCorrectionApplyMethodSettings::Subtract => "Subtract".to_string() }, param_type: ParamType::Dropdown, options: vec!["Divide".to_string(), "Subtract".to_string()], min: 0.0f32, max: 0.0f32, step: 1.0000f32, groups: vec![] }], vec![ParameterDef { name: "rescale".to_string(), display_name: "Rescale".to_string(), description: "Stretch the corrected image's intensities to fill the full\n`[0.0, 1.0]` range afterward - guards against `Divide` pushing\npreviously-dim regions above `1.0`.".to_string(), value: format!("{}", _s.rescale), param_type: ParamType::Toggle, options: vec![], min: 0.0f32, max: 0.0f32, step: 1.0000f32, groups: vec![] }]].concat(),
            Self::ImageCache(_s) => vec![ParameterDef { name: "mode".to_string(), display_name: "Mode".to_string(), description: "Whether to save the current state to the cache or load a state from it.".to_string(), value: match _s.mode { MathImageCacheImageCacheModeSettings::Store => "Store".to_string(), MathImageCacheImageCacheModeSettings::Load => "Load".to_string() }, param_type: ParamType::Dropdown, options: vec!["Store".to_string(), "Load".to_string()], min: 0.0f32, max: 0.0f32, step: 1.0000f32, groups: vec![] }],
            Self::ImageMath(_s) => [vec![ParameterDef { name: "operand".to_string(), display_name: "Operand".to_string(), description: "The specific mathematical or logical operator to apply.".to_string(), value: match _s.operand { MathImageMathOperandSettings::None => "None".to_string(), MathImageMathOperandSettings::Invert => "Invert".to_string(), MathImageMathOperandSettings::Add => "Add".to_string(), MathImageMathOperandSettings::Subtract => "Subtract".to_string(), MathImageMathOperandSettings::Multiply => "Multiply".to_string(), MathImageMathOperandSettings::Divide => "Divide".to_string(), MathImageMathOperandSettings::And => "And".to_string(), MathImageMathOperandSettings::Or => "Or".to_string(), MathImageMathOperandSettings::Xor => "Xor".to_string(), MathImageMathOperandSettings::Min => "Min".to_string(), MathImageMathOperandSettings::Max => "Max".to_string(), MathImageMathOperandSettings::Average => "Average".to_string(), MathImageMathOperandSettings::DifferenceType => "Difference Type".to_string() }, param_type: ParamType::Dropdown, options: vec!["None".to_string(), "Invert".to_string(), "Add".to_string(), "Subtract".to_string(), "Multiply".to_string(), "Divide".to_string(), "And".to_string(), "Or".to_string(), "Xor".to_string(), "Min".to_string(), "Max".to_string(), "Average".to_string(), "Difference Type".to_string()], min: 0.0f32, max: 0.0f32, step: 1.0000f32, groups: vec![] }], vec![ParameterDef { name: "swap_operands".to_string(), display_name: "Swap Operands".to_string(), description: "If false, the calculation is `(Current Image OP Cached Image)`.\nIf true, the calculation is `(Cached Image OP Current Image)`.\n\nThis is critical for non-commutative operations like Subtraction or Division.".to_string(), value: format!("{}", _s.swap_operands), param_type: ParamType::Toggle, options: vec![], min: 0.0f32, max: 0.0f32, step: 1.0000f32, groups: vec![] }]].concat(),
            Self::IntensityTransformation(_s) => [vec![ParameterDef { name: "mode".to_string(), display_name: "Mode".to_string(), description: "Determines whether to use automated enhancement or user-defined values.".to_string(), value: match _s.mode { FiltersIntensityTransformIntensityTransformModeSettings::Automatic => "Automatic".to_string(), FiltersIntensityTransformIntensityTransformModeSettings::Manual => "Manual".to_string() }, param_type: ParamType::Dropdown, options: vec!["Automatic".to_string(), "Manual".to_string()], min: 0.0f32, max: 0.0f32, step: 1.0000f32, groups: vec![] }], vec![ParameterDef { name: "contrast".to_string(), display_name: "Contrast".to_string(), description: "Contrast multiplier (gain).\n\nOnly active in [`Mode::Manual`].\nValues > 1.0 increase contrast, while values < 1.0 decrease it.".to_string(), value: format!("{}", _s.contrast), param_type: ParamType::Number, options: vec![], min: 0.0f32, max: 0.0f32, step: 1.0000f32, groups: vec![] }], vec![ParameterDef { name: "brightness".to_string(), display_name: "Brightness".to_string(), description: "Brightness offset (bias).\n\nOnly active in [`Mode::Manual`].\nPositive values brighten the image, negative values darken it.".to_string(), value: format!("{}", _s.brightness), param_type: ParamType::Number, options: vec![], min: 0.0f32, max: 0.0f32, step: 1.0000f32, groups: vec![] }]].concat(),
            Self::Laplacian(_s) => vec![ParameterDef { name: "kernel_size".to_string(), display_name: "Kernel Size".to_string(), description: "The size of the discrete Laplacian aperture.\n\nTypically 3. Larger sizes (5, 7) approximate the Laplacian of Gaussian (LoG)\nmore closely but are more computationally expensive. Must be an odd number.".to_string(), value: format!("{}", _s.kernel_size), param_type: ParamType::Number, options: vec![], min: 0.0f32, max: 0.0f32, step: 1.0000f32, groups: vec![] }],
            Self::MedianSubtract(_s) => vec![ParameterDef { name: "radius".to_string(), display_name: "Radius".to_string(), description: "The radius of the neighborhood used to estimate the background.\n\nFeatures smaller than this radius will be preserved, while\nlarger structures will be treated as background and removed.".to_string(), value: format!("{}", _s.radius), param_type: ParamType::Number, options: vec![], min: 0.0f32, max: 0.0f32, step: 1.0000f32, groups: vec![] }],
            Self::MorphologicalCommand(_s) => [vec![ParameterDef { name: "op".to_string(), display_name: "Op".to_string(), description: "The transformation type (e.g., Dilate, Erode).".to_string(), value: match _s.op { MorphologyMorphologicalTransformationMorphOpsSettings::Dilate => "Dilate".to_string(), MorphologyMorphologicalTransformationMorphOpsSettings::Erode => "Erode".to_string(), MorphologyMorphologicalTransformationMorphOpsSettings::Open => "Open".to_string(), MorphologyMorphologicalTransformationMorphOpsSettings::Close => "Close".to_string() }, param_type: ParamType::Dropdown, options: vec!["Dilate".to_string(), "Erode".to_string(), "Open".to_string(), "Close".to_string()], min: 0.0f32, max: 0.0f32, step: 1.0000f32, groups: vec![] }], vec![ParameterDef { name: "kernel_size".to_string(), display_name: "Kernel Size".to_string(), description: "The diameter of the structuring element in pixels.\nMust be an odd number (e.g., 3, 5, 7).".to_string(), value: format!("{}", _s.kernel_size), param_type: ParamType::Number, options: vec![], min: 0.0f32, max: 0.0f32, step: 1.0000f32, groups: vec![] }], vec![ParameterDef { name: "kernel_shape".to_string(), display_name: "Kernel Shape".to_string(), description: "The geometric profile of the structuring element.".to_string(), value: match _s.kernel_shape { MorphologyMorphologicalTransformationKernelShapesSettings::Box => "Box".to_string(), MorphologyMorphologicalTransformationKernelShapesSettings::Ellipse => "Ellipse".to_string(), MorphologyMorphologicalTransformationKernelShapesSettings::Cross => "Cross".to_string() }, param_type: ParamType::Dropdown, options: vec!["Box".to_string(), "Ellipse".to_string(), "Cross".to_string()], min: 0.0f32, max: 0.0f32, step: 1.0000f32, groups: vec![] }], vec![ParameterDef { name: "use_grayscale".to_string(), display_name: "Use Grayscale".to_string(), description: "If set the grayscale image instead of the labeld image is taken to perform a morphological transform".to_string(), value: format!("{}", _s.use_grayscale), param_type: ParamType::Toggle, options: vec![], min: 0.0f32, max: 0.0f32, step: 1.0000f32, groups: vec![] }]].concat(),
            Self::ObjectMath(_s) => [vec![ParameterDef { name: "operation".to_string(), display_name: "Operation".to_string(), description: "Boolean set operation to apply".to_string(), value: match _s.operation { ClassificationObjectMathObjectSetOperationSettings::And => "And".to_string(), ClassificationObjectMathObjectSetOperationSettings::Or => "Or".to_string(), ClassificationObjectMathObjectSetOperationSettings::Xor => "Xor".to_string(), ClassificationObjectMathObjectSetOperationSettings::Subtract => "Subtract".to_string() }, param_type: ParamType::Dropdown, options: vec!["And".to_string(), "Or".to_string(), "Xor".to_string(), "Subtract".to_string()], min: 0.0f32, max: 0.0f32, step: 1.0000f32, groups: vec![] }], vec![ParameterDef { name: "input_class".to_string(), display_name: "Input Class".to_string(), description: "ROIs carrying this class are the left-hand operand (\"A\").".to_string(), value: match _s.input_class.to_u32() { Some(v) => format!("{}", v), None => "-1".to_string() }, param_type: ParamType::ObjClass, options: vec![], min: 0.0f32, max: 0.0f32, step: 1.0000f32, groups: vec![] }], vec![ParameterDef { name: "other_class".to_string(), display_name: "Other Class".to_string(), description: "ROIs carrying this class are the right-hand operand (\"B\").".to_string(), value: match _s.other_class.to_u32() { Some(v) => format!("{}", v), None => "-1".to_string() }, param_type: ParamType::ObjClass, options: vec![], min: 0.0f32, max: 0.0f32, step: 1.0000f32, groups: vec![] }], vec![ParameterDef { name: "other_filter_classes".to_string(), display_name: "Other Filter Classes".to_string(), description: "Optional additional label filters applied to `other_class` objects.\n\nOnly `other_class` objects that carry all listed classes are used.".to_string(), value: _s.other_filter_classes.iter().filter_map(|c| c.to_u32()).map(|v| v.to_string()).collect::<Vec<_>>().join(","), param_type: ParamType::MultiObjClass, options: (0u32..33u32).map(|__idx| if _s.other_filter_classes.iter().any(|c| c.to_u32().map_or(false, |v| v == __idx)) { "1".to_string() } else { "0".to_string() }).collect::<Vec<_>>(), min: 0.0f32, max: 0.0f32, step: 1.0000f32, groups: vec![] }], vec![ParameterDef { name: "size_unit".to_string(), display_name: "Size Unit".to_string(), description: "Size unit for `min_overlap_area`".to_string(), value: match _s.size_unit { SizeUnits::NanoMeter => "nm".to_string(), SizeUnits::Pixels => "px".to_string() }, param_type: ParamType::SizeUnits, options: vec!["nm".to_string(), "px".to_string()], min: 0.0f32, max: 0.0f32, step: 1.0000f32, groups: vec![] }], vec![ParameterDef { name: "min_overlap_area".to_string(), display_name: "Min Overlap Area".to_string(), description: "Minimum overlap area before an `other_class` object is treated as a partner\nof an input object; objects overlapping less than this are ignored.".to_string(), value: format!("{}", _s.min_overlap_area), param_type: ParamType::Number, options: vec![], min: 0.0f32, max: 0.0f32, step: 1.0000f32, groups: vec![] }], vec![ParameterDef { name: "output_class".to_string(), display_name: "Output Class".to_string(), description: "If unset, the result replaces the input object in place.\n\nIf set, a new object carrying this class is created for each input object instead,\nleaving the input object untouched.".to_string(), value: match _s.output_class.to_u32() { Some(v) => format!("{}", v), None => "-1".to_string() }, param_type: ParamType::ObjClass, options: vec![], min: 0.0f32, max: 0.0f32, step: 1.0000f32, groups: vec![] }], vec![ParameterDef { name: "keep_unmatched".to_string(), display_name: "Keep Unmatched".to_string(), description: "When an input object has no qualifying overlapping partner: keep it unchanged in\nthe output (true), or drop it entirely - no output for it at all - (false).\n\nNote this is a policy override, not the literal mathematical result: e.g. for\n`And`, the true result of \"A and nothing\" is empty, but `keep_unmatched = true`\nstill leaves A untouched rather than emitting a zero-area object.".to_string(), value: format!("{}", _s.keep_unmatched), param_type: ParamType::Toggle, options: vec![], min: 0.0f32, max: 0.0f32, step: 1.0000f32, groups: vec![] }]].concat(),
            Self::PixelClassifier(_s) => [vec![ParameterDef { name: "model_path".to_string(), display_name: "Model Path".to_string(), description: "Path to a trained pixel classifier model, saved from the AI training dialog.".to_string(), value: _s.model_path.display().to_string(), param_type: ParamType::FilePath, options: vec!["evamodel".to_string()], min: 0.0f32, max: 0.0f32, step: 1.0000f32, groups: vec![] }], vec![ParameterDef { name: "segmentation_mapping".to_string(), display_name: "Segmentation Mapping".to_string(), description: "Maps the model's predicted classes to this project's segmentation classes.".to_string(), value: String::new(), param_type: ParamType::Group, options: vec![], min: 0.0f32, max: 0.0f32, step: 1.0000f32, groups: _s.segmentation_mapping.iter().map(|__item| [vec![ParameterDef { name: "segmentation_class".to_string(), display_name: "Segmentation Class".to_string(), description: "Segmentation class predicted by the classifier model.".to_string(), value: format!("{}", __item.segmentation_class.as_u32()), param_type: ParamType::SegClass, options: vec![], min: 0.0f32, max: 0.0f32, step: 1.0000f32, groups: vec![] }], vec![ParameterDef { name: "object_class_id".to_string(), display_name: "Object Class Id".to_string(), description: "The project's own segmentation class pixels predicted as\n`segmentation_class` are written as.".to_string(), value: format!("{}", __item.object_class_id.as_u32()), param_type: ParamType::SegClass, options: vec![], min: 0.0f32, max: 0.0f32, step: 1.0000f32, groups: vec![] }]].concat()).collect() }]].concat(),
            Self::RankFilter(_s) => [vec![ParameterDef { name: "radius".to_string(), display_name: "Radius".to_string(), description: "The circular radius of the neighborhood to consider.\n\nA radius of 1.0 roughly corresponds to a 3x3 square, while larger\nvalues increase the effect's strength and computational cost.".to_string(), value: format!("{}", _s.radius), param_type: ParamType::Number, options: vec![], min: 0.0f32, max: 0.0f32, step: 1.0000f32, groups: vec![] }], vec![ParameterDef { name: "filter_type".to_string(), display_name: "Filter Type".to_string(), description: "The specific ranking algorithm to apply to the neighborhood.".to_string(), value: match _s.filter_type { FiltersRankFilterRankFilterTypeSettings::Median => "Median".to_string(), FiltersRankFilterRankFilterTypeSettings::Min => "Min".to_string(), FiltersRankFilterRankFilterTypeSettings::Max => "Max".to_string(), FiltersRankFilterRankFilterTypeSettings::Mean => "Mean".to_string(), FiltersRankFilterRankFilterTypeSettings::Outliers(_) => "Outliers".to_string() }, param_type: ParamType::Dropdown, options: vec!["Median".to_string(), "Min".to_string(), "Max".to_string(), "Mean".to_string(), "Outliers".to_string()], min: 0.0f32, max: 0.0f32, step: 1.0000f32, groups: vec![] }], match &_s.filter_type { FiltersRankFilterRankFilterTypeSettings::Median => vec![], FiltersRankFilterRankFilterTypeSettings::Min => vec![], FiltersRankFilterRankFilterTypeSettings::Max => vec![], FiltersRankFilterRankFilterTypeSettings::Mean => vec![], FiltersRankFilterRankFilterTypeSettings::Outliers(__inner) => vec![ParameterDef { name: "filter_type.0".to_string(), display_name: "Outliers".to_string(), description: "Replaces a pixel only if it deviates from the neighborhood median\nby more than the specified threshold.".to_string(), value: format!("{}", __inner), param_type: ParamType::Number, options: vec![], min: 0.0f32, max: 0.0f32, step: 1.0000f32, groups: vec![] }] }].concat(),
            Self::RollingBall(_s) => [vec![ParameterDef { name: "radius".to_string(), display_name: "Radius".to_string(), description: "The radius of the ball or paraboloid in pixels.\n\nThis should be at least as large as the radius of the largest\nobject in the image that is not part of the background.".to_string(), value: format!("{}", _s.radius), param_type: ParamType::Spinner, options: vec![], min: 1.0f32, max: 64.0f32, step: 1.0000f32, groups: vec![] }], vec![ParameterDef { name: "ball_type".to_string(), display_name: "Ball Type".to_string(), description: "The geometric shape of the rolling structural element.".to_string(), value: match _s.ball_type { FiltersRollingBallBallTypeSettings::Ball => "Ball".to_string(), FiltersRollingBallBallTypeSettings::Paraboloid => "Paraboloid".to_string() }, param_type: ParamType::Dropdown, options: vec!["Ball".to_string(), "Paraboloid".to_string()], min: 0.0f32, max: 0.0f32, step: 1.0000f32, groups: vec![] }], vec![ParameterDef { name: "pre_smooth".to_string(), display_name: "Pre Smooth".to_string(), description: "".to_string(), value: format!("{}", _s.pre_smooth), param_type: ParamType::Toggle, options: vec![], min: 0.0f32, max: 0.0f32, step: 1.0000f32, groups: vec![] }]].concat(),
            Self::SaveImage(_s) => [vec![ParameterDef { name: "name".to_string(), display_name: "Name".to_string(), description: "Name the image should be stord under".to_string(), value: _s.name.clone(), param_type: ParamType::Text, options: vec![], min: 0.0f32, max: 0.0f32, step: 1.0000f32, groups: vec![] }], vec![ParameterDef { name: "source".to_string(), display_name: "Source".to_string(), description: "Which image from the pipeline should be stored".to_string(), value: match _s.source { MathSaveImageImageSourceSettings::Image => "Image".to_string(), MathSaveImageImageSourceSettings::InstanceMap => "Instance Map".to_string(), MathSaveImageImageSourceSettings::SegmentationMask => "Segmentation Mask".to_string() }, param_type: ParamType::Dropdown, options: vec!["Image".to_string(), "Instance Map".to_string(), "Segmentation Mask".to_string()], min: 0.0f32, max: 0.0f32, step: 1.0000f32, groups: vec![] }]].concat(),
            Self::Stardist(_s) => [vec![ParameterDef { name: "model_path".to_string(), display_name: "Model Path".to_string(), description: "Path to a TorchScript-exported StarDist model (`torch.jit.script`/`torch.jit.trace`).".to_string(), value: _s.model_path.display().to_string(), param_type: ParamType::FilePath, options: vec!["pt,pth".to_string()], min: 0.0f32, max: 0.0f32, step: 1.0000f32, groups: vec![] }], vec![ParameterDef { name: "object_class_id".to_string(), display_name: "Object Class Id".to_string(), description: "The class assigned to pixels of every detected object. All other\npixels are assigned `SegmentationClass::BACKGROUND`.".to_string(), value: format!("{}", _s.object_class_id.as_u32()), param_type: ParamType::SegClass, options: vec![], min: 0.0f32, max: 0.0f32, step: 1.0000f32, groups: vec![] }], vec![ParameterDef { name: "probability_threshold".to_string(), display_name: "Probability Threshold".to_string(), description: "Probability above which a grid cell is considered a candidate object center.".to_string(), value: format!("{}", _s.probability_threshold), param_type: ParamType::Spinner, options: vec![], min: 0.0f32, max: 1.0f32, step: 0.0100f32, groups: vec![] }], vec![ParameterDef { name: "nms_threshold".to_string(), display_name: "Nms Threshold".to_string(), description: "Pixel-overlap ratio (intersection / union) above which a lower-scoring\ncandidate polygon is suppressed in favor of an overlapping higher-scoring one.".to_string(), value: format!("{}", _s.nms_threshold), param_type: ParamType::Spinner, options: vec![], min: 0.0f32, max: 1.0f32, step: 0.0100f32, groups: vec![] }]].concat(),
            Self::StructureTensor(_s) => [vec![ParameterDef { name: "mode".to_string(), display_name: "Mode".to_string(), description: "The mathematical output to be produced by the algorithm.".to_string(), value: match _s.mode { FiltersStructureTensorTensorModeSettings::EigenvaluesX => "Eigenvalues X".to_string(), FiltersStructureTensorTensorModeSettings::EigenvaluesY => "Eigenvalues Y".to_string(), FiltersStructureTensorTensorModeSettings::Coherence => "Coherence".to_string() }, param_type: ParamType::Dropdown, options: vec!["Eigenvalues X".to_string(), "Eigenvalues Y".to_string(), "Coherence".to_string()], min: 0.0f32, max: 0.0f32, step: 1.0000f32, groups: vec![] }], vec![ParameterDef { name: "kernel_size".to_string(), display_name: "Kernel Size".to_string(), description: "The size of the integration window used to average the local gradients.\n\nLarger windows provide more stability against noise but reduce\nspatial resolution.".to_string(), value: format!("{}", _s.kernel_size), param_type: ParamType::Number, options: vec![], min: 0.0f32, max: 0.0f32, step: 1.0000f32, groups: vec![] }], vec![ParameterDef { name: "sigma".to_string(), display_name: "Sigma".to_string(), description: "The standard deviation for the Gaussian weighting of the integration window.\n\nControls the spatial \"reach\" of the neighborhood analysis.".to_string(), value: format!("{}", _s.sigma), param_type: ParamType::Number, options: vec![], min: 0.0f32, max: 0.0f32, step: 1.0000f32, groups: vec![] }]].concat(),
            Self::Threshold(_s) => vec![ParameterDef { name: "thresholds".to_string(), display_name: "Thresholds".to_string(), description: "A list of thresholding layers. Overlapping ranges are resolved\nby the order of the vector (last-in priority).".to_string(), value: String::new(), param_type: ParamType::Group, options: vec![], min: 0.0f32, max: 0.0f32, step: 1.0000f32, groups: _s.thresholds.iter().map(|__item| [vec![ParameterDef { name: "method".to_string(), display_name: "Method".to_string(), description: "The algorithm to use (Manual or Automatic).".to_string(), value: match __item.method { SegmentationThresholdThresholdMethodSettings::None => "None".to_string(), SegmentationThresholdThresholdMethodSettings::Manual => "Manual".to_string(), SegmentationThresholdThresholdMethodSettings::Li => "Li".to_string(), SegmentationThresholdThresholdMethodSettings::MinError => "Min Error".to_string(), SegmentationThresholdThresholdMethodSettings::Triangle => "Triangle".to_string(), SegmentationThresholdThresholdMethodSettings::Moments => "Moments".to_string(), SegmentationThresholdThresholdMethodSettings::Huang => "Huang".to_string(), SegmentationThresholdThresholdMethodSettings::Intermodes => "Intermodes".to_string(), SegmentationThresholdThresholdMethodSettings::IsoData => "Iso Data".to_string(), SegmentationThresholdThresholdMethodSettings::MaxEntropy => "Max Entropy".to_string(), SegmentationThresholdThresholdMethodSettings::Mean => "Mean".to_string(), SegmentationThresholdThresholdMethodSettings::Minimum => "Minimum".to_string(), SegmentationThresholdThresholdMethodSettings::Otsu { .. } => "Otsu".to_string(), SegmentationThresholdThresholdMethodSettings::Percentile => "Percentile".to_string(), SegmentationThresholdThresholdMethodSettings::RenyiEntropy => "Renyi Entropy".to_string(), SegmentationThresholdThresholdMethodSettings::Shanbhag => "Shanbhag".to_string(), SegmentationThresholdThresholdMethodSettings::Yen => "Yen".to_string() }, param_type: ParamType::Dropdown, options: vec!["None".to_string(), "Manual".to_string(), "Li".to_string(), "Min Error".to_string(), "Triangle".to_string(), "Moments".to_string(), "Huang".to_string(), "Intermodes".to_string(), "Iso Data".to_string(), "Max Entropy".to_string(), "Mean".to_string(), "Minimum".to_string(), "Otsu".to_string(), "Percentile".to_string(), "Renyi Entropy".to_string(), "Shanbhag".to_string(), "Yen".to_string()], min: 0.0f32, max: 0.0f32, step: 1.0000f32, groups: vec![] }], match &__item.method { SegmentationThresholdThresholdMethodSettings::None => vec![], SegmentationThresholdThresholdMethodSettings::Manual => vec![], SegmentationThresholdThresholdMethodSettings::Li => vec![], SegmentationThresholdThresholdMethodSettings::MinError => vec![], SegmentationThresholdThresholdMethodSettings::Triangle => vec![], SegmentationThresholdThresholdMethodSettings::Moments => vec![], SegmentationThresholdThresholdMethodSettings::Huang => vec![], SegmentationThresholdThresholdMethodSettings::Intermodes => vec![], SegmentationThresholdThresholdMethodSettings::IsoData => vec![], SegmentationThresholdThresholdMethodSettings::MaxEntropy => vec![], SegmentationThresholdThresholdMethodSettings::Mean => vec![], SegmentationThresholdThresholdMethodSettings::Minimum => vec![], SegmentationThresholdThresholdMethodSettings::Otsu { classes } => [vec![ParameterDef { name: "method.classes".to_string(), display_name: "Classes".to_string(), description: "".to_string(), value: match classes { SegmentationThresholdOtsuClassesSettings::Two => "Two".to_string(), SegmentationThresholdOtsuClassesSettings::Three { .. } => "Three".to_string() }, param_type: ParamType::Dropdown, options: vec!["Two".to_string(), "Three".to_string()], min: 0.0f32, max: 0.0f32, step: 1.0000f32, groups: vec![] }], match &classes { SegmentationThresholdOtsuClassesSettings::Two => vec![], SegmentationThresholdOtsuClassesSettings::Three { middle_class } => vec![ParameterDef { name: "method.classes.middle_class".to_string(), display_name: "Middle Class".to_string(), description: "".to_string(), value: match middle_class { SegmentationThresholdOtsuMiddleClassSettings::Foreground => "Foreground".to_string(), SegmentationThresholdOtsuMiddleClassSettings::Background => "Background".to_string() }, param_type: ParamType::Dropdown, options: vec!["Foreground".to_string(), "Background".to_string()], min: 0.0f32, max: 0.0f32, step: 1.0000f32, groups: vec![] }] }].concat(), SegmentationThresholdThresholdMethodSettings::Percentile => vec![], SegmentationThresholdThresholdMethodSettings::RenyiEntropy => vec![], SegmentationThresholdThresholdMethodSettings::Shanbhag => vec![], SegmentationThresholdThresholdMethodSettings::Yen => vec![] }, vec![ParameterDef { name: "min_threshold".to_string(), display_name: "Min Threshold".to_string(), description: "The lower intensity bound. Used directly in `Manual` mode, or as a\nfloor for auto-methods.".to_string(), value: format!("{}", __item.min_threshold), param_type: ParamType::Spinner, options: vec![], min: 0.0f32, max: 65535.0f32, step: 1.0000f32, groups: vec![] }], vec![ParameterDef { name: "max_threshold".to_string(), display_name: "Max Threshold".to_string(), description: "The upper intensity bound. Used directly in `Manual` mode, or as a\nceiling for auto-methods.".to_string(), value: format!("{}", __item.max_threshold), param_type: ParamType::Spinner, options: vec![], min: 0.0f32, max: 65535.0f32, step: 1.0000f32, groups: vec![] }], vec![ParameterDef { name: "unit".to_string(), display_name: "Unit".to_string(), description: "Unit used for the threshold value.\n\nbit: 0 - 255/65535\n%: 0 - 100.0\nrel: 0 - 1.0".to_string(), value: match __item.unit { PixelUnits::Bit => "bit".to_string(), PixelUnits::Percent => "%".to_string(), PixelUnits::Relative => "rel".to_string() }, param_type: ParamType::PixelUnits, options: vec!["bit".to_string(), "%".to_string(), "rel".to_string()], min: 0.0f32, max: 0.0f32, step: 1.0000f32, groups: vec![] }], vec![ParameterDef { name: "object_class_id".to_string(), display_name: "Object Class Id".to_string(), description: "The classification ID assigned to pixels falling within this threshold range.".to_string(), value: format!("{}", __item.object_class_id.as_u32()), param_type: ParamType::SegClass, options: vec![], min: 0.0f32, max: 0.0f32, step: 1.0000f32, groups: vec![] }]].concat()).collect() }],
            Self::TransformObjects(_s) => [vec![ParameterDef { name: "function".to_string(), display_name: "Function".to_string(), description: "Geometric transform applied to each input object".to_string(), value: match _s.function { ClassificationTransformObjectsTransformFunctionSettings::Scale { .. } => "Scale".to_string(), ClassificationTransformObjectsTransformFunctionSettings::SnapArea { .. } => "Snap Area".to_string(), ClassificationTransformObjectsTransformFunctionSettings::MinCircle { .. } => "Min Circle".to_string(), ClassificationTransformObjectsTransformFunctionSettings::DrawCircle { .. } => "Draw Circle".to_string(), ClassificationTransformObjectsTransformFunctionSettings::FittingEllipse { .. } => "Fitting Ellipse".to_string(), ClassificationTransformObjectsTransformFunctionSettings::Expand { .. } => "Expand".to_string(), ClassificationTransformObjectsTransformFunctionSettings::Shrink { .. } => "Shrink".to_string() }, param_type: ParamType::Dropdown, options: vec!["Scale".to_string(), "Snap Area".to_string(), "Min Circle".to_string(), "Draw Circle".to_string(), "Fitting Ellipse".to_string(), "Expand".to_string(), "Shrink".to_string()], min: 0.0f32, max: 0.0f32, step: 1.0000f32, groups: vec![] }], match &_s.function { ClassificationTransformObjectsTransformFunctionSettings::Scale { factor } => vec![ParameterDef { name: "function.factor".to_string(), display_name: "Factor".to_string(), description: "Unitless scale factor".to_string(), value: format!("{}", factor), param_type: ParamType::Spinner, options: vec![], min: 0.0f32, max: 65535.0f32, step: 1.0000f32, groups: vec![] }], ClassificationTransformObjectsTransformFunctionSettings::SnapArea { extra_size, unit } => [vec![ParameterDef { name: "function.extra_size".to_string(), display_name: "Extra Size".to_string(), description: "Size added on top of the object's bounding-box diameter".to_string(), value: format!("{}", extra_size), param_type: ParamType::Spinner, options: vec![], min: 0.0f32, max: 65535.0f32, step: 1.0000f32, groups: vec![] }], vec![ParameterDef { name: "function.unit".to_string(), display_name: "Unit".to_string(), description: "Unit `extra_size` is expressed in".to_string(), value: match unit { SizeUnits::NanoMeter => "nm".to_string(), SizeUnits::Pixels => "px".to_string() }, param_type: ParamType::SizeUnits, options: vec!["nm".to_string(), "px".to_string()], min: 0.0f32, max: 0.0f32, step: 1.0000f32, groups: vec![] }]].concat(), ClassificationTransformObjectsTransformFunctionSettings::MinCircle { min_diameter, unit } => [vec![ParameterDef { name: "function.min_diameter".to_string(), display_name: "Min Diameter".to_string(), description: "Minimum circle diameter".to_string(), value: format!("{}", min_diameter), param_type: ParamType::Spinner, options: vec![], min: 0.0f32, max: 65535.0f32, step: 1.0000f32, groups: vec![] }], vec![ParameterDef { name: "function.unit".to_string(), display_name: "Unit".to_string(), description: "Unit `min_diameter` is expressed in".to_string(), value: match unit { SizeUnits::NanoMeter => "nm".to_string(), SizeUnits::Pixels => "px".to_string() }, param_type: ParamType::SizeUnits, options: vec!["nm".to_string(), "px".to_string()], min: 0.0f32, max: 0.0f32, step: 1.0000f32, groups: vec![] }]].concat(), ClassificationTransformObjectsTransformFunctionSettings::DrawCircle { diameter, unit } => [vec![ParameterDef { name: "function.diameter".to_string(), display_name: "Diameter".to_string(), description: "Circle diameter (0 = use the object's bounding box)".to_string(), value: format!("{}", diameter), param_type: ParamType::Spinner, options: vec![], min: 0.0f32, max: 65535.0f32, step: 1.0000f32, groups: vec![] }], vec![ParameterDef { name: "function.unit".to_string(), display_name: "Unit".to_string(), description: "Unit `diameter` is expressed in".to_string(), value: match unit { SizeUnits::NanoMeter => "nm".to_string(), SizeUnits::Pixels => "px".to_string() }, param_type: ParamType::SizeUnits, options: vec!["nm".to_string(), "px".to_string()], min: 0.0f32, max: 0.0f32, step: 1.0000f32, groups: vec![] }]].concat(), ClassificationTransformObjectsTransformFunctionSettings::FittingEllipse { scale } => vec![ParameterDef { name: "function.scale".to_string(), display_name: "Scale".to_string(), description: "Unitless scale factor for the fitted ellipse".to_string(), value: format!("{}", scale), param_type: ParamType::Spinner, options: vec![], min: 0.0f32, max: 65535.0f32, step: 1.0000f32, groups: vec![] }], ClassificationTransformObjectsTransformFunctionSettings::Expand { margin, unit } => [vec![ParameterDef { name: "function.margin".to_string(), display_name: "Margin".to_string(), description: "Margin added on every side of the mask's contour".to_string(), value: format!("{}", margin), param_type: ParamType::Spinner, options: vec![], min: 0.0f32, max: 65535.0f32, step: 1.0000f32, groups: vec![] }], vec![ParameterDef { name: "function.unit".to_string(), display_name: "Unit".to_string(), description: "Unit `margin` is expressed in".to_string(), value: match unit { SizeUnits::NanoMeter => "nm".to_string(), SizeUnits::Pixels => "px".to_string() }, param_type: ParamType::SizeUnits, options: vec!["nm".to_string(), "px".to_string()], min: 0.0f32, max: 0.0f32, step: 1.0000f32, groups: vec![] }]].concat(), ClassificationTransformObjectsTransformFunctionSettings::Shrink { margin, unit } => [vec![ParameterDef { name: "function.margin".to_string(), display_name: "Margin".to_string(), description: "Margin removed from every side of the mask's contour".to_string(), value: format!("{}", margin), param_type: ParamType::Spinner, options: vec![], min: 0.0f32, max: 65535.0f32, step: 1.0000f32, groups: vec![] }], vec![ParameterDef { name: "function.unit".to_string(), display_name: "Unit".to_string(), description: "Unit `margin` is expressed in".to_string(), value: match unit { SizeUnits::NanoMeter => "nm".to_string(), SizeUnits::Pixels => "px".to_string() }, param_type: ParamType::SizeUnits, options: vec!["nm".to_string(), "px".to_string()], min: 0.0f32, max: 0.0f32, step: 1.0000f32, groups: vec![] }]].concat() }, vec![ParameterDef { name: "input_class".to_string(), display_name: "Input Class".to_string(), description: "ROIs carrying this class are the input to the transform".to_string(), value: match _s.input_class.to_u32() { Some(v) => format!("{}", v), None => "-1".to_string() }, param_type: ParamType::ObjClass, options: vec![], min: 0.0f32, max: 0.0f32, step: 1.0000f32, groups: vec![] }], vec![ParameterDef { name: "output_class".to_string(), display_name: "Output Class".to_string(), description: "If unset, the transformed shape replaces the input object in place.\n\nIf set, a new object carrying this class is created for each transformed input object instead,\nleaving the input object untouched.".to_string(), value: match _s.output_class.to_u32() { Some(v) => format!("{}", v), None => "-1".to_string() }, param_type: ParamType::ObjClass, options: vec![], min: 0.0f32, max: 0.0f32, step: 1.0000f32, groups: vec![] }]].concat(),
            Self::UNet(_s) => [vec![ParameterDef { name: "model_path".to_string(), display_name: "Model Path".to_string(), description: "Path to a TorchScript-exported U-Net model (`torch.jit.script`/`torch.jit.trace`).".to_string(), value: _s.model_path.display().to_string(), param_type: ParamType::FilePath, options: vec!["pt,pth".to_string()], min: 0.0f32, max: 0.0f32, step: 1.0000f32, groups: vec![] }], vec![ParameterDef { name: "object_class_id".to_string(), display_name: "Object Class Id".to_string(), description: "The class assigned to pixels whose predicted probability reaches\n`probability_threshold`. All other pixels are assigned `SegmentationClass::BACKGROUND`.".to_string(), value: format!("{}", _s.object_class_id.as_u32()), param_type: ParamType::SegClass, options: vec![], min: 0.0f32, max: 0.0f32, step: 1.0000f32, groups: vec![] }], vec![ParameterDef { name: "probability_threshold".to_string(), display_name: "Probability Threshold".to_string(), description: "Probability above which a pixel is classified as foreground.".to_string(), value: format!("{}", _s.probability_threshold), param_type: ParamType::Spinner, options: vec![], min: 0.0f32, max: 1.0f32, step: 0.0100f32, groups: vec![] }], vec![ParameterDef { name: "output_mode".to_string(), display_name: "Output Mode".to_string(), description: "How to interpret the model output when it has more than one channel.\nIgnored for single-channel outputs.".to_string(), value: match _s.output_mode { AiSegmentationUnetUNetOutputModeSettings::SoftmaxClasses => "Softmax Classes".to_string(), AiSegmentationUnetUNetOutputModeSettings::IndependentChannels => "Independent Channels".to_string() }, param_type: ParamType::Dropdown, options: vec!["Softmax Classes".to_string(), "Independent Channels".to_string()], min: 0.0f32, max: 0.0f32, step: 1.0000f32, groups: vec![] }], vec![ParameterDef { name: "foreground_channel".to_string(), display_name: "Foreground Channel".to_string(), description: "Index of the channel holding the foreground probability, used only\nwhen the model output has more than one channel. Out-of-range values\nare clamped to the last available channel.\n\n* For `SoftmaxClasses`, this is typically the last channel (e.g. `1`\nfor a 2-class background/foreground head).\n* For `IndependentChannels`, this is whichever channel the model\ndedicates to the foreground mask — commonly `0` for boundary-aware\nmodels, which conventionally output mask before boundary.".to_string(), value: format!("{}", _s.foreground_channel), param_type: ParamType::Spinner, options: vec![], min: 0.0f32, max: 16.0f32, step: 1.0000f32, groups: vec![] }], vec![ParameterDef { name: "boundary_channel".to_string(), display_name: "Boundary Channel".to_string(), description: "Index of an optional **boundary** channel for boundary-aware models\n(e.g. bioimage.io's `affable-shark` / NucleiSegmentationBoundaryModel,\nwhich outputs mask in channel 0 and boundary in channel 1). Set to `-1`\nto disable.\n\nWhen enabled, a pixel is classified as foreground only where the\nforeground probability reaches `probability_threshold` **and** the\nboundary probability stays below `boundary_threshold`. This carves the\npredicted boundaries out as thin gaps, so a following `ConnectedComponents`\nseparates touching objects directly — which is the whole point of a\nboundary model and the only way to split nuclei a plain mask merges.".to_string(), value: format!("{}", _s.boundary_channel), param_type: ParamType::Spinner, options: vec![], min: -1.0f32, max: 16.0f32, step: 1.0000f32, groups: vec![] }], vec![ParameterDef { name: "boundary_threshold".to_string(), display_name: "Boundary Threshold".to_string(), description: "Boundary probability at or above which a pixel is treated as an object\nboundary and excluded from the foreground. Only used when\n`boundary_channel` is enabled (>= 0). Lower values cut wider gaps\n(separate more aggressively); higher values cut thinner gaps.".to_string(), value: format!("{}", _s.boundary_threshold), param_type: ParamType::Spinner, options: vec![], min: 0.0f32, max: 1.0f32, step: 0.0100f32, groups: vec![] }]].concat(),
            Self::Voronoi(_s) => [vec![ParameterDef { name: "centers".to_string(), display_name: "Centers".to_string(), description: "Object class whose instances act as Voronoi seed points.".to_string(), value: match _s.centers.to_u32() { Some(v) => format!("{}", v), None => "-1".to_string() }, param_type: ParamType::ObjClass, options: vec![], min: 0.0f32, max: 0.0f32, step: 1.0000f32, groups: vec![] }], vec![ParameterDef { name: "center_filter_classes".to_string(), display_name: "Center Filter Classes".to_string(), description: "Additional label filters applied to center objects before tessellation.\n\nOnly center objects that carry all listed classes pass the filter.\nLeave empty to include all objects of `centers`.".to_string(), value: _s.center_filter_classes.iter().filter_map(|c| c.to_u32()).map(|v| v.to_string()).collect::<Vec<_>>().join(","), param_type: ParamType::MultiObjClass, options: (0u32..33u32).map(|__idx| if _s.center_filter_classes.iter().any(|c| c.to_u32().map_or(false, |v| v == __idx)) { "1".to_string() } else { "0".to_string() }).collect::<Vec<_>>(), min: 0.0f32, max: 0.0f32, step: 1.0000f32, groups: vec![] }], vec![ParameterDef { name: "mask".to_string(), display_name: "Mask".to_string(), description: "Object class used to spatially constrain the Voronoi areas.\n\nEach computed Voronoi region is intersected with the union of all mask objects,\ndiscarding pixels that fall outside the mask. Set to `Unset` to expand\nto the full image boundary instead.".to_string(), value: match _s.mask.to_u32() { Some(v) => format!("{}", v), None => "-1".to_string() }, param_type: ParamType::ObjClass, options: vec![], min: 0.0f32, max: 0.0f32, step: 1.0000f32, groups: vec![] }], vec![ParameterDef { name: "mask_filter_classes".to_string(), display_name: "Mask Filter Classes".to_string(), description: "Additional label filters applied to mask objects.\n\nOnly mask objects that carry all listed classes pass the filter.\nLeave empty to include all objects of `mask`.".to_string(), value: _s.mask_filter_classes.iter().filter_map(|c| c.to_u32()).map(|v| v.to_string()).collect::<Vec<_>>().join(","), param_type: ParamType::MultiObjClass, options: (0u32..33u32).map(|__idx| if _s.mask_filter_classes.iter().any(|c| c.to_u32().map_or(false, |v| v == __idx)) { "1".to_string() } else { "0".to_string() }).collect::<Vec<_>>(), min: 0.0f32, max: 0.0f32, step: 1.0000f32, groups: vec![] }], vec![ParameterDef { name: "output_class".to_string(), display_name: "Output Class".to_string(), description: "Object class assigned to the resulting Voronoi region ROIs.".to_string(), value: match _s.output_class.to_u32() { Some(v) => format!("{}", v), None => "-1".to_string() }, param_type: ParamType::ObjClass, options: vec![], min: 0.0f32, max: 0.0f32, step: 1.0000f32, groups: vec![] }], vec![ParameterDef { name: "unit".to_string(), display_name: "Unit".to_string(), description: "Unit in which `max_radius` is expressed (e.g. pixels, nm, µm).".to_string(), value: match _s.unit { SizeUnits::NanoMeter => "nm".to_string(), SizeUnits::Pixels => "px".to_string() }, param_type: ParamType::SizeUnits, options: vec!["nm".to_string(), "px".to_string()], min: 0.0f32, max: 0.0f32, step: 1.0000f32, groups: vec![] }], vec![ParameterDef { name: "max_radius".to_string(), display_name: "Max Radius".to_string(), description: "Maximum expansion radius for a Voronoi region.\n\nPixels farther than this distance from the nearest seed center are excluded\nfrom the region. Use `0` or a negative value to disable the limit.".to_string(), value: format!("{}", _s.max_radius), param_type: ParamType::Number, options: vec![], min: 0.0f32, max: 0.0f32, step: 1.0000f32, groups: vec![] }], vec![ParameterDef { name: "exclude_areas_at_the_edges".to_string(), display_name: "Exclude Areas At The Edges".to_string(), description: "Discard Voronoi regions that touch the image border.".to_string(), value: format!("{}", _s.exclude_areas_at_the_edges), param_type: ParamType::Toggle, options: vec![], min: 0.0f32, max: 0.0f32, step: 1.0000f32, groups: vec![] }], vec![ParameterDef { name: "exclude_areas_with_no_center".to_string(), display_name: "Exclude Areas With No Center".to_string(), description: "Discard Voronoi regions whose originating center object was filtered out or missing.".to_string(), value: format!("{}", _s.exclude_areas_with_no_center), param_type: ParamType::Toggle, options: vec![], min: 0.0f32, max: 0.0f32, step: 1.0000f32, groups: vec![] }]].concat(),
            Self::Watershed(_s) => [vec![ParameterDef { name: "maximum_finder_tolerance".to_string(), display_name: "Maximum Finder Tolerance".to_string(), description: "Prominence tolerance for the maximum finder, in pixels of distance.\n\nA local maximum of the distance map is treated as a separate object only\nif it protrudes more than this value above the ridge connecting it to a\nhigher maximum. This is ImageJ's \"prominence\"/\"noise tolerance\" parameter.\n\n* **Low values**: more sensitive; may over-segment ragged objects.\n* **High values**: more robust; may fail to split genuinely touching objects.\n\nImageJ's default of `0.5` works well for most distance maps; raise it if a\nsingle object is being split into several pieces.".to_string(), value: format!("{}", _s.maximum_finder_tolerance), param_type: ParamType::Spinner, options: vec![], min: 0.1f32, max: 20.0f32, step: 0.5000f32, groups: vec![] }], vec![ParameterDef { name: "smoothing_sigma".to_string(), display_name: "Smoothing Sigma".to_string(), description: "Standard deviation (px) of an optional Gaussian blur applied to the\ndistance map *before* the maximum finder. `0` disables it.\n\nImageJ's `trueEdmHeight` correction already handles ordinary ragged mask\nboundaries, so this is rarely needed; for extremely noisy AI masks a value\nof `1.0`–`2.0` can further suppress spurious maxima.".to_string(), value: format!("{}", _s.smoothing_sigma), param_type: ParamType::Spinner, options: vec![], min: 0.0f32, max: 10.0f32, step: 0.5000f32, groups: vec![] }], vec![ParameterDef { name: "min_object_size".to_string(), display_name: "Min Object Size".to_string(), description: "Minimum object size, in pixels. After segmentation, any object smaller than\nthis is removed (its pixels become background). `0` disables the filter.\n\nUse it to drop tiny fragments left by very ragged masks.".to_string(), value: format!("{}", _s.min_object_size), param_type: ParamType::Spinner, options: vec![], min: 0.0f32, max: 100000.0f32, step: 1.0000f32, groups: vec![] }]].concat(),
            Self::WeightedDeviation(_s) => [vec![ParameterDef { name: "kernel_size".to_string(), display_name: "Kernel Size".to_string(), description: "The size of the local neighborhood window.\n\nMust be an odd number. Larger windows capture broader texture\nvariations but increase computational load.".to_string(), value: format!("{}", _s.kernel_size), param_type: ParamType::Number, options: vec![], min: 0.0f32, max: 0.0f32, step: 1.0000f32, groups: vec![] }], vec![ParameterDef { name: "sigma".to_string(), display_name: "Sigma".to_string(), description: "The standard deviation for the Gaussian weighting function.\n\nDefines the \"softness\" of the neighborhood boundaries. A larger\nsigma includes more of the surrounding context in the deviation calculation.".to_string(), value: format!("{}", _s.sigma), param_type: ParamType::Number, options: vec![], min: 0.0f32, max: 0.0f32, step: 1.0000f32, groups: vec![] }]].concat(),
        }
    }

    pub fn to_summary(&self) -> String {
        match self {
            Self::AiObjectClassifier(_) => String::new(),
            Self::Blur(s) => format!("Kernel size: {}", format!("{:.3}", s.kernel_size)),
            Self::Cellpose(_) => String::new(),
            Self::ClassifyObjects(s) => format!(
                "Min Area: {} · Min Eccentricity: {} · Max Eccentricity: {} · Allow Edge Touching: {}",
                format!("{:.3}", s.min_area),
                format!("{:.3}", s.min_eccentricity),
                format!("{:.3}", s.max_eccentricity),
                format!("{}", s.allow_edge_touching)
            ),
            Self::Colocalization(_) => String::new(),
            Self::ColorFilterCommand(_) => String::new(),
            Self::ConnectedComponents(s) => format!("Min Size: {}", format!("{:.3}", s.min_size)),
            Self::DistanceTransform(_) => String::new(),
            Self::EdgeDetectionCanny(_) => String::new(),
            Self::EdgeDetectionSobel(_) => String::new(),
            Self::EnhanceContrast(_) => String::new(),
            Self::ExtractObjects(_) => String::new(),
            Self::FillHoles(_) => String::new(),
            Self::GaussianBlur(s) => format!(
                "Kernel Size: {} · Sigma: {}",
                format!("{:.3}", s.kernel_size),
                format!("{:.3}", s.sigma)
            ),
            Self::Hessian(_) => String::new(),
            Self::IlluminationCorrection(_) => String::new(),
            Self::ImageCache(_) => String::new(),
            Self::ImageMath(_) => String::new(),
            Self::IntensityTransformation(_) => String::new(),
            Self::Laplacian(_) => String::new(),
            Self::MedianSubtract(_) => String::new(),
            Self::MorphologicalCommand(_) => String::new(),
            Self::ObjectMath(s) => format!(
                "Operation: {}",
                match s.operation {
                    ClassificationObjectMathObjectSetOperationSettings::And => "And".to_string(),
                    ClassificationObjectMathObjectSetOperationSettings::Or => "Or".to_string(),
                    ClassificationObjectMathObjectSetOperationSettings::Xor => "Xor".to_string(),
                    ClassificationObjectMathObjectSetOperationSettings::Subtract =>
                        "Subtract".to_string(),
                }
            ),
            Self::PixelClassifier(_) => String::new(),
            Self::RankFilter(_) => String::new(),
            Self::RollingBall(_) => String::new(),
            Self::SaveImage(_) => String::new(),
            Self::Stardist(_) => String::new(),
            Self::StructureTensor(_) => String::new(),
            Self::Threshold(_) => String::new(),
            Self::TransformObjects(s) => format!(
                "Function: {}",
                match s.function {
                    ClassificationTransformObjectsTransformFunctionSettings::Scale { .. } =>
                        "Scale".to_string(),
                    ClassificationTransformObjectsTransformFunctionSettings::SnapArea {
                        ..
                    } => "Snap Area".to_string(),
                    ClassificationTransformObjectsTransformFunctionSettings::MinCircle {
                        ..
                    } => "Min Circle".to_string(),
                    ClassificationTransformObjectsTransformFunctionSettings::DrawCircle {
                        ..
                    } => "Draw Circle".to_string(),
                    ClassificationTransformObjectsTransformFunctionSettings::FittingEllipse {
                        ..
                    } => "Fitting Ellipse".to_string(),
                    ClassificationTransformObjectsTransformFunctionSettings::Expand { .. } =>
                        "Expand".to_string(),
                    ClassificationTransformObjectsTransformFunctionSettings::Shrink { .. } =>
                        "Shrink".to_string(),
                }
            ),
            Self::UNet(_) => String::new(),
            Self::Voronoi(_) => String::new(),
            Self::Watershed(_) => String::new(),
            Self::WeightedDeviation(_) => String::new(),
        }
    }

    pub fn apply_param_change(&mut self, param_name: &str, value: &str) {
        match self {
            Self::AiObjectClassifier(s) => {
                if param_name == "model_path" {
                    s.model_path = std::path::PathBuf::from(value);
                }
                if param_name.starts_with("segmentation_mapping.") {
                    let rest = &param_name[21..];
                    let mut _p = rest.splitn(2, '.');
                    if let (Some(_i), Some(nested_name)) = (_p.next(), _p.next()) {
                        if let Ok(_idx) = _i.parse::<usize>() {
                            if let Some(item) = s.segmentation_mapping.get_mut(_idx) {
                                if nested_name == "object_class" {
                                    if value == "-1" {
                                        item.object_class = ObjectClass::Unset;
                                    } else if let Ok(v) = value.parse::<u32>() {
                                        item.object_class = ObjectClass::Valid(v);
                                    }
                                }
                                if nested_name == "output_class" {
                                    if value == "-1" {
                                        item.output_class = ObjectClass::Unset;
                                    } else if let Ok(v) = value.parse::<u32>() {
                                        item.output_class = ObjectClass::Valid(v);
                                    }
                                }
                            }
                        }
                    }
                }
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
                    s.match_handling = match value { "Add class on match" => ClassificationAiObjectClassifierAiClassifyMatchHandlingSettings::AddOutputClassIfMatch, "Reclassify on match" => ClassificationAiObjectClassifierAiClassifyMatchHandlingSettings::ReclassifyIfMatch, _ => (s.match_handling).clone() };
                }
            }
            Self::Blur(s) => {
                if param_name == "kernel_size" {
                    if let Ok(v) = value.parse::<usize>() {
                        s.kernel_size = v;
                    }
                }
            }
            Self::Cellpose(s) => {
                if param_name == "model_path" {
                    s.model_path = std::path::PathBuf::from(value);
                }
                if param_name == "object_class_id" {
                    if let Ok(v) = value.parse::<u32>() {
                        s.object_class_id = SegmentationClass(v);
                    }
                }
                if param_name == "input_channels" {
                    if let Ok(v) = value.parse::<i32>() {
                        s.input_channels = v;
                    }
                }
                if param_name == "probability_threshold" {
                    if let Ok(v) = value.parse::<f32>() {
                        s.probability_threshold = v;
                    }
                }
                if param_name == "flow_iterations" {
                    if let Ok(v) = value.parse::<i32>() {
                        s.flow_iterations = v;
                    }
                }
                if param_name == "min_object_size" {
                    if let Ok(v) = value.parse::<i32>() {
                        s.min_object_size = v;
                    }
                }
            }
            Self::ClassifyObjects(s) => {
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
                    s.match_handling = match value { "Add class on match" => ClassificationClassifyObjectsClassifyMatchHandlingSettings::AddOutputClassIfMatch, "Add class on mismatch" => ClassificationClassifyObjectsClassifyMatchHandlingSettings::AddOutputClassIfNotMatch, "Remove class on match" => ClassificationClassifyObjectsClassifyMatchHandlingSettings::RemoveInputClassIfMatch, "Remove class on mismatch" => ClassificationClassifyObjectsClassifyMatchHandlingSettings::RemoveInputClassIfNotMatch, "Remove output class on match" => ClassificationClassifyObjectsClassifyMatchHandlingSettings::RemoveOutputClassIfMatch, "Remove output class on mismatch" => ClassificationClassifyObjectsClassifyMatchHandlingSettings::RemoveOutputClassIfNotMatch, "Remove objects matching criteria" => ClassificationClassifyObjectsClassifyMatchHandlingSettings::RemoveAllClassesIfMatch, "Keep objects matching criteria" => ClassificationClassifyObjectsClassifyMatchHandlingSettings::RemoveAllClassesIfNotMatch, "Reclassify on match" => ClassificationClassifyObjectsClassifyMatchHandlingSettings::ReclassifyIfMatch, "Reclassify on mismatch" => ClassificationClassifyObjectsClassifyMatchHandlingSettings::ReclassifyIfNotMatch, _ => (s.match_handling).clone() };
                }
                if param_name == "output_class" {
                    if value == "-1" {
                        s.output_class = ObjectClass::Unset;
                    } else if let Ok(v) = value.parse::<u32>() {
                        s.output_class = ObjectClass::Valid(v);
                    }
                }
                if param_name == "overlapping_with" {
                    if value == "-1" {
                        s.overlapping_with = ObjectClass::Unset;
                    } else if let Ok(v) = value.parse::<u32>() {
                        s.overlapping_with = ObjectClass::Valid(v);
                    }
                }
                if param_name == "min_intersection_area" {
                    if let Ok(v) = value.parse::<f32>() {
                        s.min_intersection_area = v;
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
                if param_name == "class_for_overlapping_areas" {
                    if value == "-1" {
                        s.class_for_overlapping_areas = ObjectClass::Unset;
                    } else if let Ok(v) = value.parse::<u32>() {
                        s.class_for_overlapping_areas = ObjectClass::Valid(v);
                    }
                }
                if param_name == "multiplicity" {
                    s.multiplicity = match value {
                        "No multi coloc (1:1)" => {
                            ClassificationColocObjectsColocMultiplicitySettings::OneToOne
                        }
                        "Allow multi coloc" => {
                            ClassificationColocObjectsColocMultiplicitySettings::ManyToMany
                        }
                        "Multi coloc only for selected" => {
                            ClassificationColocObjectsColocMultiplicitySettings::MultiFor(vec![])
                        }
                        _ => (s.multiplicity).clone(),
                    };
                }
                if let ClassificationColocObjectsColocMultiplicitySettings::MultiFor(
                    ref mut __inner,
                ) = s.multiplicity
                {
                    if param_name == "multiplicity.0" {
                        if let Some(id) = value
                            .strip_prefix("toggle:")
                            .and_then(|x| x.trim().parse::<u32>().ok())
                        {
                            if __inner
                                .iter()
                                .any(|c| c.to_u32().map_or(false, |v| v == id))
                            {
                                __inner.retain(|c| c.to_u32().map_or(true, |v| v != id));
                            } else {
                                __inner.push(ObjectClass::Valid(id));
                            }
                        } else {
                            *__inner = value
                                .split(',')
                                .filter(|x| !x.is_empty())
                                .filter_map(|x| x.trim().parse::<u32>().ok())
                                .map(ObjectClass::Valid)
                                .collect();
                        }
                    }
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
                if param_name == "exclude_classes" {
                    if let Some(id) = value
                        .strip_prefix("toggle:")
                        .and_then(|x| x.trim().parse::<u32>().ok())
                    {
                        if s.exclude_classes
                            .iter()
                            .any(|c| c.to_u32().map_or(false, |v| v == id))
                        {
                            s.exclude_classes
                                .retain(|c| c.to_u32().map_or(true, |v| v != id));
                        } else {
                            s.exclude_classes.push(ObjectClass::Valid(id));
                        }
                    } else {
                        s.exclude_classes = value
                            .split(',')
                            .filter(|x| !x.is_empty())
                            .filter_map(|x| x.trim().parse::<u32>().ok())
                            .map(|v| ObjectClass::Valid(v))
                            .collect();
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
            Self::ConnectedComponents(s) => {
                if param_name == "min_size" {
                    if let Ok(v) = value.parse::<i32>() {
                        s.min_size = v;
                    }
                }
            }
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
            Self::ExtractObjects(s) => {
                if param_name == "max_objects_before_fail" {
                    if let Ok(v) = value.parse::<i32>() {
                        s.max_objects_before_fail = v;
                    }
                }
            }
            Self::FillHoles(_) => {}
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
                        "Eigenvalues X" => FiltersHessianHessianModeSettings::EigenvaluesX,
                        "Eigenvalues Y" => FiltersHessianHessianModeSettings::EigenvaluesY,
                        _ => (s.mode).clone(),
                    };
                }
            }
            Self::IlluminationCorrection(s) => {
                if param_name == "method" {
                    s.method = match value {
                        "Regular" => FiltersIlluminationCorrectionCorrectionMethodSettings::Regular,
                        "Background" => {
                            FiltersIlluminationCorrectionCorrectionMethodSettings::Background
                        }
                        _ => (s.method).clone(),
                    };
                }
                if param_name == "block_size" {
                    if let Ok(v) = value.parse::<usize>() {
                        s.block_size = v;
                    }
                }
                if param_name == "smoothing" {
                    s.smoothing = match value {
                        "None" => FiltersIlluminationCorrectionSmoothingMethodSettings::None,
                        "Gaussian" => {
                            FiltersIlluminationCorrectionSmoothingMethodSettings::Gaussian {
                                sigma: 2.0f32,
                            }
                        }
                        "Median" => FiltersIlluminationCorrectionSmoothingMethodSettings::Median {
                            radius: 2usize,
                        },
                        "Fit Polynomial" => {
                            FiltersIlluminationCorrectionSmoothingMethodSettings::FitPolynomial
                        }
                        _ => (s.smoothing).clone(),
                    };
                }
                if let FiltersIlluminationCorrectionSmoothingMethodSettings::Gaussian {
                    ref mut sigma,
                } = s.smoothing
                {
                    if param_name == "smoothing.sigma" {
                        if let Ok(v) = value.parse::<f32>() {
                            *sigma = v;
                        }
                    }
                }
                if let FiltersIlluminationCorrectionSmoothingMethodSettings::Median {
                    ref mut radius,
                } = s.smoothing
                {
                    if param_name == "smoothing.radius" {
                        if let Ok(v) = value.parse::<usize>() {
                            *radius = v;
                        }
                    }
                }
                if param_name == "apply_method" {
                    s.apply_method = match value {
                        "Divide" => FiltersIlluminationCorrectionApplyMethodSettings::Divide,
                        "Subtract" => FiltersIlluminationCorrectionApplyMethodSettings::Subtract,
                        _ => (s.apply_method).clone(),
                    };
                }
                if param_name == "rescale" {
                    s.rescale = value == "true";
                }
            }
            Self::ImageCache(s) => {
                if param_name == "mode" {
                    s.mode = match value {
                        "Store" => MathImageCacheImageCacheModeSettings::Store,
                        "Load" => MathImageCacheImageCacheModeSettings::Load,
                        _ => (s.mode).clone(),
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
                        "And" => MathImageMathOperandSettings::And,
                        "Or" => MathImageMathOperandSettings::Or,
                        "Xor" => MathImageMathOperandSettings::Xor,
                        "Min" => MathImageMathOperandSettings::Min,
                        "Max" => MathImageMathOperandSettings::Max,
                        "Average" => MathImageMathOperandSettings::Average,
                        "Difference Type" => MathImageMathOperandSettings::DifferenceType,
                        _ => (s.operand).clone(),
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
                        _ => (s.mode).clone(),
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
                        _ => (s.op).clone(),
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
                        _ => (s.kernel_shape).clone(),
                    };
                }
                if param_name == "use_grayscale" {
                    s.use_grayscale = value == "true";
                }
            }
            Self::ObjectMath(s) => {
                if param_name == "operation" {
                    s.operation = match value {
                        "And" => ClassificationObjectMathObjectSetOperationSettings::And,
                        "Or" => ClassificationObjectMathObjectSetOperationSettings::Or,
                        "Xor" => ClassificationObjectMathObjectSetOperationSettings::Xor,
                        "Subtract" => ClassificationObjectMathObjectSetOperationSettings::Subtract,
                        _ => (s.operation).clone(),
                    };
                }
                if param_name == "input_class" {
                    if value == "-1" {
                        s.input_class = ObjectClass::Unset;
                    } else if let Ok(v) = value.parse::<u32>() {
                        s.input_class = ObjectClass::Valid(v);
                    }
                }
                if param_name == "other_class" {
                    if value == "-1" {
                        s.other_class = ObjectClass::Unset;
                    } else if let Ok(v) = value.parse::<u32>() {
                        s.other_class = ObjectClass::Valid(v);
                    }
                }
                if param_name == "other_filter_classes" {
                    if let Some(id) = value
                        .strip_prefix("toggle:")
                        .and_then(|x| x.trim().parse::<u32>().ok())
                    {
                        if s.other_filter_classes
                            .iter()
                            .any(|c| c.to_u32().map_or(false, |v| v == id))
                        {
                            s.other_filter_classes
                                .retain(|c| c.to_u32().map_or(true, |v| v != id));
                        } else {
                            s.other_filter_classes.push(ObjectClass::Valid(id));
                        }
                    } else {
                        s.other_filter_classes = value
                            .split(',')
                            .filter(|x| !x.is_empty())
                            .filter_map(|x| x.trim().parse::<u32>().ok())
                            .map(|v| ObjectClass::Valid(v))
                            .collect();
                    }
                }
                if param_name == "size_unit" {
                    s.size_unit = match value {
                        "nm" => SizeUnits::NanoMeter,
                        _ => SizeUnits::Pixels,
                    };
                }
                if param_name == "min_overlap_area" {
                    if let Ok(v) = value.parse::<f32>() {
                        s.min_overlap_area = v;
                    }
                }
                if param_name == "output_class" {
                    if value == "-1" {
                        s.output_class = ObjectClass::Unset;
                    } else if let Ok(v) = value.parse::<u32>() {
                        s.output_class = ObjectClass::Valid(v);
                    }
                }
                if param_name == "keep_unmatched" {
                    s.keep_unmatched = value == "true";
                }
            }
            Self::PixelClassifier(s) => {
                if param_name == "model_path" {
                    s.model_path = std::path::PathBuf::from(value);
                }
                if param_name.starts_with("segmentation_mapping.") {
                    let rest = &param_name[21..];
                    let mut _p = rest.splitn(2, '.');
                    if let (Some(_i), Some(nested_name)) = (_p.next(), _p.next()) {
                        if let Ok(_idx) = _i.parse::<usize>() {
                            if let Some(item) = s.segmentation_mapping.get_mut(_idx) {
                                if nested_name == "segmentation_class" {
                                    if let Ok(v) = value.parse::<u32>() {
                                        item.segmentation_class = SegmentationClass(v);
                                    }
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
                        "Outliers" => {
                            FiltersRankFilterRankFilterTypeSettings::Outliers(f32::default())
                        }
                        _ => (s.filter_type).clone(),
                    };
                }
                if let FiltersRankFilterRankFilterTypeSettings::Outliers(ref mut __inner) =
                    s.filter_type
                {
                    if param_name == "filter_type.0" {
                        if let Ok(v) = value.parse::<f32>() {
                            *__inner = v;
                        }
                    }
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
                        _ => (s.ball_type).clone(),
                    };
                }
                if param_name == "pre_smooth" {
                    s.pre_smooth = value == "true";
                }
            }
            Self::SaveImage(s) => {
                if param_name == "name" {
                    s.name = value.to_string();
                }
                if param_name == "source" {
                    s.source = match value {
                        "Image" => MathSaveImageImageSourceSettings::Image,
                        "Instance Map" => MathSaveImageImageSourceSettings::InstanceMap,
                        "Segmentation Mask" => MathSaveImageImageSourceSettings::SegmentationMask,
                        _ => (s.source).clone(),
                    };
                }
            }
            Self::Stardist(s) => {
                if param_name == "model_path" {
                    s.model_path = std::path::PathBuf::from(value);
                }
                if param_name == "object_class_id" {
                    if let Ok(v) = value.parse::<u32>() {
                        s.object_class_id = SegmentationClass(v);
                    }
                }
                if param_name == "probability_threshold" {
                    if let Ok(v) = value.parse::<f32>() {
                        s.probability_threshold = v;
                    }
                }
                if param_name == "nms_threshold" {
                    if let Ok(v) = value.parse::<f32>() {
                        s.nms_threshold = v;
                    }
                }
            }
            Self::StructureTensor(s) => {
                if param_name == "mode" {
                    s.mode = match value {
                        "Eigenvalues X" => FiltersStructureTensorTensorModeSettings::EigenvaluesX,
                        "Eigenvalues Y" => FiltersStructureTensorTensorModeSettings::EigenvaluesY,
                        "Coherence" => FiltersStructureTensorTensorModeSettings::Coherence,
                        _ => (s.mode).clone(),
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
                                    item.method = match value { "None" => SegmentationThresholdThresholdMethodSettings::None, "Manual" => SegmentationThresholdThresholdMethodSettings::Manual, "Li" => SegmentationThresholdThresholdMethodSettings::Li, "Min Error" => SegmentationThresholdThresholdMethodSettings::MinError, "Triangle" => SegmentationThresholdThresholdMethodSettings::Triangle, "Moments" => SegmentationThresholdThresholdMethodSettings::Moments, "Huang" => SegmentationThresholdThresholdMethodSettings::Huang, "Intermodes" => SegmentationThresholdThresholdMethodSettings::Intermodes, "Iso Data" => SegmentationThresholdThresholdMethodSettings::IsoData, "Max Entropy" => SegmentationThresholdThresholdMethodSettings::MaxEntropy, "Mean" => SegmentationThresholdThresholdMethodSettings::Mean, "Minimum" => SegmentationThresholdThresholdMethodSettings::Minimum, "Otsu" => SegmentationThresholdThresholdMethodSettings::Otsu { classes: SegmentationThresholdOtsuClassesSettings :: Two,  }, "Percentile" => SegmentationThresholdThresholdMethodSettings::Percentile, "Renyi Entropy" => SegmentationThresholdThresholdMethodSettings::RenyiEntropy, "Shanbhag" => SegmentationThresholdThresholdMethodSettings::Shanbhag, "Yen" => SegmentationThresholdThresholdMethodSettings::Yen, _ => (item.method).clone() };
                                }
                                if let SegmentationThresholdThresholdMethodSettings::Otsu {
                                    ref mut classes,
                                } = item.method
                                {
                                    if nested_name == "method.classes" {
                                        *classes = match value { "Two" => SegmentationThresholdOtsuClassesSettings::Two, "Three" => SegmentationThresholdOtsuClassesSettings::Three { middle_class: SegmentationThresholdOtsuMiddleClassSettings :: Background,  }, _ => (*classes).clone() };
                                    }
                                    if let SegmentationThresholdOtsuClassesSettings::Three {
                                        ref mut middle_class,
                                    } = *classes
                                    {
                                        if nested_name == "method.classes.middle_class" {
                                            *middle_class = match value { "Foreground" => SegmentationThresholdOtsuMiddleClassSettings::Foreground, "Background" => SegmentationThresholdOtsuMiddleClassSettings::Background, _ => (*middle_class).clone() };
                                        }
                                    }
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
            Self::TransformObjects(s) => {
                if param_name == "function" {
                    s.function = match value { "Scale" => ClassificationTransformObjectsTransformFunctionSettings::Scale { factor: 1.0f32,  }, "Snap Area" => ClassificationTransformObjectsTransformFunctionSettings::SnapArea { extra_size: 0.0f32, unit: SizeUnits :: NanoMeter,  }, "Min Circle" => ClassificationTransformObjectsTransformFunctionSettings::MinCircle { min_diameter: 0.0f32, unit: SizeUnits :: NanoMeter,  }, "Draw Circle" => ClassificationTransformObjectsTransformFunctionSettings::DrawCircle { diameter: 0.0f32, unit: SizeUnits :: NanoMeter,  }, "Fitting Ellipse" => ClassificationTransformObjectsTransformFunctionSettings::FittingEllipse { scale: 1.0f32,  }, "Expand" => ClassificationTransformObjectsTransformFunctionSettings::Expand { margin: 0.0f32, unit: SizeUnits :: NanoMeter,  }, "Shrink" => ClassificationTransformObjectsTransformFunctionSettings::Shrink { margin: 0.0f32, unit: SizeUnits :: NanoMeter,  }, _ => (s.function).clone() };
                }
                if let ClassificationTransformObjectsTransformFunctionSettings::Scale {
                    ref mut factor,
                } = s.function
                {
                    if param_name == "function.factor" {
                        if let Ok(v) = value.parse::<f32>() {
                            *factor = v;
                        }
                    }
                }
                if let ClassificationTransformObjectsTransformFunctionSettings::SnapArea {
                    ref mut extra_size,
                    ref mut unit,
                } = s.function
                {
                    if param_name == "function.extra_size" {
                        if let Ok(v) = value.parse::<f32>() {
                            *extra_size = v;
                        }
                    }
                    if param_name == "function.unit" {
                        *unit = match value {
                            "nm" => SizeUnits::NanoMeter,
                            _ => SizeUnits::Pixels,
                        };
                    }
                }
                if let ClassificationTransformObjectsTransformFunctionSettings::MinCircle {
                    ref mut min_diameter,
                    ref mut unit,
                } = s.function
                {
                    if param_name == "function.min_diameter" {
                        if let Ok(v) = value.parse::<f32>() {
                            *min_diameter = v;
                        }
                    }
                    if param_name == "function.unit" {
                        *unit = match value {
                            "nm" => SizeUnits::NanoMeter,
                            _ => SizeUnits::Pixels,
                        };
                    }
                }
                if let ClassificationTransformObjectsTransformFunctionSettings::DrawCircle {
                    ref mut diameter,
                    ref mut unit,
                } = s.function
                {
                    if param_name == "function.diameter" {
                        if let Ok(v) = value.parse::<f32>() {
                            *diameter = v;
                        }
                    }
                    if param_name == "function.unit" {
                        *unit = match value {
                            "nm" => SizeUnits::NanoMeter,
                            _ => SizeUnits::Pixels,
                        };
                    }
                }
                if let ClassificationTransformObjectsTransformFunctionSettings::FittingEllipse {
                    ref mut scale,
                } = s.function
                {
                    if param_name == "function.scale" {
                        if let Ok(v) = value.parse::<f32>() {
                            *scale = v;
                        }
                    }
                }
                if let ClassificationTransformObjectsTransformFunctionSettings::Expand {
                    ref mut margin,
                    ref mut unit,
                } = s.function
                {
                    if param_name == "function.margin" {
                        if let Ok(v) = value.parse::<f32>() {
                            *margin = v;
                        }
                    }
                    if param_name == "function.unit" {
                        *unit = match value {
                            "nm" => SizeUnits::NanoMeter,
                            _ => SizeUnits::Pixels,
                        };
                    }
                }
                if let ClassificationTransformObjectsTransformFunctionSettings::Shrink {
                    ref mut margin,
                    ref mut unit,
                } = s.function
                {
                    if param_name == "function.margin" {
                        if let Ok(v) = value.parse::<f32>() {
                            *margin = v;
                        }
                    }
                    if param_name == "function.unit" {
                        *unit = match value {
                            "nm" => SizeUnits::NanoMeter,
                            _ => SizeUnits::Pixels,
                        };
                    }
                }
                if param_name == "input_class" {
                    if value == "-1" {
                        s.input_class = ObjectClass::Unset;
                    } else if let Ok(v) = value.parse::<u32>() {
                        s.input_class = ObjectClass::Valid(v);
                    }
                }
                if param_name == "output_class" {
                    if value == "-1" {
                        s.output_class = ObjectClass::Unset;
                    } else if let Ok(v) = value.parse::<u32>() {
                        s.output_class = ObjectClass::Valid(v);
                    }
                }
            }
            Self::UNet(s) => {
                if param_name == "model_path" {
                    s.model_path = std::path::PathBuf::from(value);
                }
                if param_name == "object_class_id" {
                    if let Ok(v) = value.parse::<u32>() {
                        s.object_class_id = SegmentationClass(v);
                    }
                }
                if param_name == "probability_threshold" {
                    if let Ok(v) = value.parse::<f32>() {
                        s.probability_threshold = v;
                    }
                }
                if param_name == "output_mode" {
                    s.output_mode = match value {
                        "Softmax Classes" => {
                            AiSegmentationUnetUNetOutputModeSettings::SoftmaxClasses
                        }
                        "Independent Channels" => {
                            AiSegmentationUnetUNetOutputModeSettings::IndependentChannels
                        }
                        _ => (s.output_mode).clone(),
                    };
                }
                if param_name == "foreground_channel" {
                    if let Ok(v) = value.parse::<i32>() {
                        s.foreground_channel = v;
                    }
                }
                if param_name == "boundary_channel" {
                    if let Ok(v) = value.parse::<i32>() {
                        s.boundary_channel = v;
                    }
                }
                if param_name == "boundary_threshold" {
                    if let Ok(v) = value.parse::<f32>() {
                        s.boundary_threshold = v;
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
                if param_name == "smoothing_sigma" {
                    if let Ok(v) = value.parse::<f32>() {
                        s.smoothing_sigma = v;
                    }
                }
                if param_name == "min_object_size" {
                    if let Ok(v) = value.parse::<i32>() {
                        s.min_object_size = v;
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
            Self::AiObjectClassifier(s) => {
                if param_name == "segmentation_mapping" {
                    if let Some(last) = s.segmentation_mapping.last().cloned() {
                        s.segmentation_mapping.push(last);
                    } else {
                        s.segmentation_mapping
                            .push(ClassificationMappingSettings::default());
                    }
                }
            }
            Self::Blur(_) => {}
            Self::Cellpose(_) => {}
            Self::ClassifyObjects(_) => {}
            Self::Colocalization(_) => {}
            Self::ColorFilterCommand(_) => {}
            Self::ConnectedComponents(_) => {}
            Self::DistanceTransform(_) => {}
            Self::EdgeDetectionCanny(_) => {}
            Self::EdgeDetectionSobel(_) => {}
            Self::EnhanceContrast(_) => {}
            Self::ExtractObjects(_) => {}
            Self::FillHoles(_) => {}
            Self::GaussianBlur(_) => {}
            Self::Hessian(_) => {}
            Self::IlluminationCorrection(_) => {}
            Self::ImageCache(_) => {}
            Self::ImageMath(_) => {}
            Self::IntensityTransformation(_) => {}
            Self::Laplacian(_) => {}
            Self::MedianSubtract(_) => {}
            Self::MorphologicalCommand(_) => {}
            Self::ObjectMath(_) => {}
            Self::PixelClassifier(s) => {
                if param_name == "segmentation_mapping" {
                    if let Some(last) = s.segmentation_mapping.last().cloned() {
                        s.segmentation_mapping.push(last);
                    } else {
                        s.segmentation_mapping
                            .push(SegmentationMappingSettings::default());
                    }
                }
            }
            Self::RankFilter(_) => {}
            Self::RollingBall(_) => {}
            Self::SaveImage(_) => {}
            Self::Stardist(_) => {}
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
            Self::TransformObjects(_) => {}
            Self::UNet(_) => {}
            Self::Voronoi(_) => {}
            Self::Watershed(_) => {}
            Self::WeightedDeviation(_) => {}
        }
    }

    pub fn remove_group_item(&mut self, param_name: &str, idx: usize) {
        match self {
            Self::AiObjectClassifier(s) => {
                if param_name == "segmentation_mapping" && idx < s.segmentation_mapping.len() {
                    s.segmentation_mapping.remove(idx);
                }
            }
            Self::Blur(_) => {}
            Self::Cellpose(_) => {}
            Self::ClassifyObjects(_) => {}
            Self::Colocalization(_) => {}
            Self::ColorFilterCommand(_) => {}
            Self::ConnectedComponents(_) => {}
            Self::DistanceTransform(_) => {}
            Self::EdgeDetectionCanny(_) => {}
            Self::EdgeDetectionSobel(_) => {}
            Self::EnhanceContrast(_) => {}
            Self::ExtractObjects(_) => {}
            Self::FillHoles(_) => {}
            Self::GaussianBlur(_) => {}
            Self::Hessian(_) => {}
            Self::IlluminationCorrection(_) => {}
            Self::ImageCache(_) => {}
            Self::ImageMath(_) => {}
            Self::IntensityTransformation(_) => {}
            Self::Laplacian(_) => {}
            Self::MedianSubtract(_) => {}
            Self::MorphologicalCommand(_) => {}
            Self::ObjectMath(_) => {}
            Self::PixelClassifier(s) => {
                if param_name == "segmentation_mapping" && idx < s.segmentation_mapping.len() {
                    s.segmentation_mapping.remove(idx);
                }
            }
            Self::RankFilter(_) => {}
            Self::RollingBall(_) => {}
            Self::SaveImage(_) => {}
            Self::Stardist(_) => {}
            Self::StructureTensor(_) => {}
            Self::Threshold(s) => {
                if param_name == "thresholds" && idx < s.thresholds.len() {
                    s.thresholds.remove(idx);
                }
            }
            Self::TransformObjects(_) => {}
            Self::UNet(_) => {}
            Self::Voronoi(_) => {}
            Self::Watershed(_) => {}
            Self::WeightedDeviation(_) => {}
        }
    }
}
