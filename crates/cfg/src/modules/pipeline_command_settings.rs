// @generated - do not edit by hand
use crate::{
    core_types::{ImageAddress, PixelUnits, SizeUnits},
    types::classes::{ObjectClass, SegmentationClass},
};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

// ============ ENUM SETTINGS ============

///  The geometric shape used to probe the image intensity surface.
#[derive(Serialize, Deserialize, JsonSchema, Debug, Clone, PartialEq, Default)]
#[serde(rename_all = "camelCase")]
pub enum FiltersRollingBallBallTypeSettings {
    /// A spherical cap.
    ///
    /// This is the traditional ImageJ algorithm. Best for images with
    /// distinct, round features like cells or particles.
    #[default]
    Ball,
    /// A sliding parabolic surface.
    ///
    /// Mathematically smoother at the edges than the Ball, often
    /// resulting in fewer artifacts on complex gradients.
    Paraboloid,
}

///  Specifies the feature extraction method for the Hessian matrix.
///
///  The Hessian matrix describes the local second-order structure of an image,
///  often used for blob detection (LoG) or ridge extraction.
#[derive(Serialize, Deserialize, JsonSchema, Debug, Clone, PartialEq, Default)]
#[serde(rename_all = "camelCase")]
pub enum FiltersHessianHessianModeSettings {
    /// Computes the determinant: $det(H) = I_{xx}I_{yy} - I_{xy}^2$.
    ///
    /// High values typically indicate "blob-like" structures or corners.
    #[default]
    Determinant,
    /// Extracts the first (larger) eigenvalue ($\lambda_1$).
    ///
    /// Useful for detecting the maximum local curvature, identifying
    /// the principal axis of a ridge.
    EigenvaluesX,
    /// Extracts the second (smaller) eigenvalue ($\lambda_2$).
    ///
    /// Highlights secondary curvature; when both $\lambda_1$ and $\lambda_2$
    /// are large, it indicates a blob or interest point.
    EigenvaluesY,
}

///  Defines the interaction type with the persistent image storage.
#[derive(Serialize, Deserialize, JsonSchema, Debug, Clone, PartialEq, Default)]
#[serde(rename_all = "camelCase")]
pub enum MathImageCacheImageCacheModeSettings {
    /// Captures the current image from the pipeline and writes it to the cache.
    ///
    /// Used for "checkpointing" results at a specific stage in the pipeline
    /// for later comparison or retrieval.
    #[default]
    Store,
    /// Retrieves a previously stored image from the cache and injects it
    /// into the current pipeline context.
    ///
    /// This effectively replaces the current working image with the cached version.
    Load,
}

#[derive(Serialize, Deserialize, JsonSchema, Debug, Clone, PartialEq, Default)]
#[serde(rename_all = "camelCase")]
pub enum MathSaveImageImageSourceSettings {
    #[default]
    Image,
    InstanceMap,
    SegmentationMask,
}

///  Specifies how intensity adjustments are calculated.
#[derive(Serialize, Deserialize, JsonSchema, Debug, Clone, PartialEq, Default)]
#[serde(rename_all = "camelCase")]
pub enum FiltersIntensityTransformIntensityTransformModeSettings {
    /// Parameters are calculated based on image statistics (e.g., histogram analysis).
    #[default]
    Automatic,
    /// Parameters are provided explicitly by the user.
    Manual,
}

///  The geometric structure of the kernel (structuring element).
#[derive(Serialize, Deserialize, JsonSchema, Debug, Clone, PartialEq, Default)]
#[serde(rename_all = "camelCase")]
pub enum MorphologyMorphologicalTransformationKernelShapesSettings {
    /// A square/rectangular kernel. Dilates in all directions equally (8-connectivity).
    #[default]
    Box,
    /// A rounded kernel. Best for preserving the natural, circular shape of objects.
    Ellipse,
    /// A cross-shaped kernel. Only considers horizontal and vertical neighbors (4-connectivity).
    Cross,
}

///  The specific morphological transformation to perform.
///
///  Morphological operations process images based on shapes, typically used to
///  remove noise, isolate individual elements, or join disparate elements.
#[derive(Serialize, Deserialize, JsonSchema, Debug, Clone, PartialEq, Default)]
#[serde(rename_all = "camelCase")]
pub enum MorphologyMorphologicalTransformationMorphOpsSettings {
    /// Expands the bright regions of an image. Useful for filling small holes.
    #[default]
    Dilate,
    /// Shrinks the bright regions of an image. Useful for removing small noise.
    Erode,
    /// An erosion followed by a dilation. Removes small bright spots (noise)
    /// while preserving the relative size of larger objects.
    Open,
    /// A dilation followed by an erosion. Fills small dark gaps or cracks
    /// within bright objects.
    Close,
}

///  The mathematical or logical operation to perform between two images.
#[derive(Serialize, Deserialize, JsonSchema, Debug, Clone, PartialEq, Default)]
#[serde(rename_all = "camelCase")]
pub enum MathImageMathOperandSettings {
    /// No operation; typically used as a placeholder.
    #[default]
    None,
    /// Unary operation: Negates the intensities of the primary image.
    Invert,
    /// Arithmetic addition: `A + B`. (Clamped to the maximum pixel value).
    Add,
    /// Arithmetic subtraction: `A - B`. (Clamped to zero).
    Subtract,
    /// Arithmetic multiplication: `A * B`.
    Multiply,
    /// Arithmetic division: `A / B`.
    Divide,
    /// Bitwise AND operation.
    AND,
    /// Bitwise OR operation.
    OR,
    /// Bitwise XOR operation.
    XOR,
    /// Per-pixel minimum: `min(A, B)`. (Darkest Pixel).
    MIN,
    /// Per-pixel maximum: `max(A, B)`. (Brightest Pixel).
    MAX,
    /// Arithmetic mean: `(A + B) / 2`.
    Average,
    /// Absolute difference: `|A - B|`. Useful for change detection.
    DifferenceType,
}

///  Specifies the statistical operation to perform on the local pixel neighborhood.
#[derive(Serialize, Deserialize, JsonSchema, Debug, Clone, PartialEq, Default)]
#[serde(rename_all = "camelCase")]
pub enum FiltersRankFilterRankFilterTypeSettings {
    /// Selects the middle value. Excellent for removing salt-and-pepper noise
    /// while preserving sharp edges.
    #[default]
    Median,
    /// Selects the lowest intensity (Erosion). Shrinks bright regions and
    /// expands dark regions.
    Min,
    /// Selects the highest intensity (Dilation). Expands bright regions and
    /// shrinks dark regions.
    Max,
    /// Computes the average value. Acts as a box blur, smoothing the image
    /// but blurring edges.
    Mean,
    /// Replaces a pixel only if it deviates from the neighborhood median
    /// by more than the specified threshold.
    Outliers(f32),
}

///  The specific calculation to extract from the Structure Tensor.
#[derive(Serialize, Deserialize, JsonSchema, Debug, Clone, PartialEq, Default)]
#[serde(rename_all = "camelCase")]
pub enum FiltersStructureTensorTensorModeSettings {
    /// Extracts the first (primary) eigenvalue.
    ///
    /// Represents the local image intensity variation in the direction
    /// perpendicular to the edge. Useful for edge detection.
    #[default]
    EigenvaluesX,
    /// Extracts the second (secondary) eigenvalue.
    ///
    /// Represents the local image intensity variation along the edge.
    /// High values typically indicate corners or noise.
    EigenvaluesY,
    /// Computes the local anisotropy (coherence) of the image.
    ///
    /// Measures how strongly the local neighborhood is oriented.
    /// Ranges from 0 (isotropic/noise) to 1 (perfectly oriented/straight edge).
    Coherence,
}

///  The mathematical strategy used to determine the optimal global threshold.
///
///  Most methods analyze the image histogram to find a "cut-off" point that
///  best separates the foreground from the background.
#[derive(Serialize, Deserialize, JsonSchema, Debug, Clone, PartialEq, Default)]
#[serde(rename_all = "camelCase")]
pub enum SegmentationThresholdThresholdMethodSettings {
    /// No threshold applied; typically used for bypass logic.
    #[default]
    None,
    /// Uses the user-provided `min_threshold` and `max_threshold` values directly.
    Manual,
    /// Li's Minimum Cross Entropy method. Effective for images with varying backgrounds.
    Li,
    /// An iterative version of Kittler and Illingworth's minimum error thresholding.
    MinError,
    /// Zack's algorithm. Geometric method best for skewed histograms with a single clear peak.
    Triangle,
    /// Tsai's method. Preserves the moments of the original image in the binary result.
    Moments,
    /// Huang's fuzzy thresholding. Minimizes the measures of fuzziness.
    Huang,
    /// Assumes a bimodal histogram and finds the average of two peaks.
    Intermodes,
    /// Ridler-Calvard iterative clustering. Similar to Otsu but uses a different error metric.
    IsoData,
    /// Kapur's method. Uses the entropy of the histogram to find the threshold.
    MaxEntropy,
    /// Uses the average intensity of all pixels as the threshold.
    Mean,
    /// Pre-smooths the histogram until there are only two peaks; finds the minimum between them.
    Minimum,
    /// Most common method. Minimizes intra-class variance (maximizes inter-class variance).
    Otsu,
    /// Assumes a fixed percentage of pixels belong to the foreground.
    Percentile,
    /// Based on the Renyi entropy of the histogram; a generalization of MaxEntropy.
    RenyiEntropy,
    /// An extension of Kapur's method using a different coefficient for entropy.
    Shanbhag,
    /// Minimizes a cost function based on the discrepancy between two classes.
    Yen,
}

// ============ PREPROCESSING ============

///  Smooths an image by averaging pixel intensities within a local neighborhood.
///
///  This algorithm applies a uniform box filter where every pixel within the moving
///  window contributes equally to the final value. It is a computationally fast
///  method used for general image smoothing, blending variations, and rapid noise
///  suppression where edge precision is less critical.
#[derive(Serialize, Deserialize, JsonSchema, Debug, Clone)]
#[schemars(default)]
#[serde(rename_all = "camelCase")]
pub struct BlurSettings {
    ///  The size of the blur matrix.
    ///
    ///  Must be an odd number (e.g., 3, 5, 7)
    #[schemars(range(min = 3, max = 27))]
    pub kernel_size: usize,
}

impl Default for BlurSettings {
    fn default() -> Self {
        Self {
            kernel_size: 3usize,
        }
    }
}

///  A command that filters an image based on a specific HSV color range.
///
///  Pixels falling outside the provided [`HsvRange`] are masked
///  out by setting to black.
///
///  # Examples
///
///  ```
///  # use imagec::backend::algos::{ColorFilterCommand, HsvRange};
///  let range = HsvRange {
///      min_h: 0.0,   max_h: 30.0, // Red tones
///      min_s: 0.5,   max_s: 1.0,
///      min_v: 0.5,   max_v: 1.0,
///  };
///
///  let command = ColorFilterCommand { range };
///  ```
#[derive(Serialize, Deserialize, JsonSchema, Debug, Clone, Default)]
#[serde(rename_all = "camelCase")]
pub struct ColorFilterCommandSettings {
    ///  The HSV color bounds to be preserved by the filter.
    pub range: HsvRangeSettings,
}

///  A command that calculates the Euclidean Distance Map (EDM) of an f32 image.
///
///  This algorithm identifies pixels below a threshold as "background" and
///  calculates the distance of every "foreground" pixel to the nearest background pixel.
#[derive(Serialize, Deserialize, JsonSchema, Debug, Clone, Default)]
#[serde(rename_all = "camelCase")]
pub struct DistanceTransformSettings {
    ///  Values less than or equal to this are treated as background (distance = 0).
    pub threshold: f32,
    ///  If true, the pixels outside the image boundary are treated as background.
    pub edges_are_background: bool,
}

///  Extracts structural boundaries and fine edges using the multi-stage Canny algorithm.
///
///  This algorithm identifies optimal edge locations by calculating spatial intensity
///  gradients, suppressing non-maximum pixel responses to thin lines down to 1-pixel width,
///  and applying a dual-threshold hysteresis loop to preserve weak edges connected
///  to strong ones while completely rejecting isolated noise.
///
///  # Examples
///
///  ```
///  # use imagec::backend::algos::EdgeDetectionCanny;
///  let edges = EdgeDetectionCanny {
///      kernel_size: 3,
///      threshold_min: 0.1,
///      threshold_max: 0.3,
///  };
///  ```
#[derive(Serialize, Deserialize, JsonSchema, Debug, Clone, Default)]
#[serde(rename_all = "camelCase")]
pub struct EdgeDetectionCannySettings {
    ///  Size of the Gaussian smoothing kernel.
    ///
    ///  Must be an odd number (e.g., 3, 5). Larger values reduce
    ///  noise but can blur fine edge details.
    pub kernel_size: usize,
    ///  Lower bound for hysteresis thresholding [0.0, 1.0].
    ///
    ///  Edges with a gradient intensity below this value are discarded.
    pub threshold_min: f32,
    ///  Upper bound for hysteresis thresholding [0.0, 1.0].
    ///
    ///  Edges with a gradient intensity above this value are considered
    ///  "strong" and are automatically preserved.
    pub threshold_max: f32,
}

///  Extracts directional boundaries by computing spatial image intensity gradients.
///
///  This algorithm applies localized 3x3 kernels to approximate the first derivative
///  of pixel intensities across the horizontal and vertical axes. It highlights
///  areas of sharp luminance changes, producing a continuous gradient map that
///  emphasizes prominent structural edges and surface transitions.
///
///  # Examples
///
///  ```
///  # use imagec::backend::algos::EdgeDetectionSobel;
///  let filter = EdgeDetectionSobel { kernel_size: 3 };
///  ```
#[derive(Serialize, Deserialize, JsonSchema, Debug, Clone, Default)]
#[serde(rename_all = "camelCase")]
pub struct EdgeDetectionSobelSettings {
    ///  The size of the Sobel operator window.
    ///
    ///  Typically 3. Larger values (5, 7) provide a more smoothed
    ///  gradient but result in "thicker" edges. Must be an odd number.
    pub kernel_size: usize,
}

///  Configuration for contrast enhancement and histogram manipulation.
///
///  This algorithm can perform linear contrast stretching, normalization,
///  or histogram equalization to improve the dynamic range of an image.
///
///  # Examples
///
///  ```
///  # use imagec::backend::algos::EnhanceContrast;
///  let settings = EnhanceContrast {
///      saturated_pixels: 0.01,   // Clip 1% of outliers
///      normalize: true,          // Stretch to [0.0, 1.0]
///      equalize_histogram: false,
///  };
///  ```
#[derive(Serialize, Deserialize, JsonSchema, Debug, Clone, Default)]
#[serde(rename_all = "camelCase")]
pub struct EnhanceContrastSettings {
    ///  Percentage of pixels to "clip" from the top and bottom of the histogram.
    ///
    ///  Range: [0.0, 1.0]. A value of 0.01 (1%) helps ignore hot/dead pixels
    ///  that would otherwise prevent effective contrast stretching.
    pub saturated_pixels: f32,
    ///  Whether to linearly stretch the remaining pixel intensities to fill
    ///  the full [0.0, 1.0] range.
    pub normalize: bool,
    ///  Whether to apply Histogram Equalization.
    ///
    ///  This redistributes pixel intensities to achieve a uniform distribution,
    ///  which is highly effective for images with low contrast but high noise.
    pub equalize_histogram: bool,
}

///  Smooths an image and reduces background noise using a Gaussian kernel.
///
///  This algorithm applies a localized, bell-curve weighted blur that suppresses
///  high-frequency pixel variations (like camera noise, salt-and-pepper artifacts,
///  or dust) while preserving structural features. It is commonly used as a
///  preprocessing step to optimize thresholding and edge detection tasks.
///
///  # Examples
///
///  ```
///  use imagec::backend::algos::GaussianBlur;
///
///  let settings = GaussianBlur {
///      kernel_size: 5,
///      sigma: 2.0
///  };
///  ```
#[derive(Serialize, Deserialize, JsonSchema, Debug, Clone)]
#[schemars(default)]
#[serde(rename_all = "camelCase")]
pub struct GaussianBlurSettings {
    ///  The size of the blur matrix.
    ///
    ///  Must be an odd number (e.g., 3, 5, 7).
    #[schemars(range(min = 3, max = 27))]
    pub kernel_size: usize,
    ///  The standard deviation of the Gaussian kernel.
    ///
    ///  Higher values create a more significant blur effect.
    ///  $$N \approx 6\sigma + 1$$
    #[schemars(range(min = 0.1, max = 5))]
    pub sigma: f32,
}

impl Default for GaussianBlurSettings {
    fn default() -> Self {
        Self {
            kernel_size: 3usize,
            sigma: 0.34f32,
        }
    }
}

///  Extracts continuous structural ridges, tubular vessels, and blobs using second-order spatial derivatives.
///
///  This algorithm constructs a localized Hessian matrix for each pixel to analyze local curvature
///  and intensity topography. By evaluating the eigenvalues of this matrix, it differentiates
///  between directional ridges (like blood vessels or filaments), distinct intensity peaks (blobs),
///  and flat regions, making it highly effective for curvilinear feature extraction.
///
///  # Examples
///
///  ```
///  # use imagec::backend::algos::{Hessian, HessianMode};
///  let detector = Hessian {
///      mode: HessianMode::Determinant,
///  };
///  ```
#[derive(Serialize, Deserialize, JsonSchema, Debug, Clone, Default)]
#[serde(rename_all = "camelCase")]
pub struct HessianSettings {
    ///  Determines which component of the Hessian matrix structure to extract.
    ///
    ///  Depending on the mode, this can highlight interest points (blobs)
    ///  or directional features (ridges).
    pub mode: FiltersHessianHessianModeSettings,
}

///  Defines a range within the HSV (Hue, Saturation, Value) color space.
///
///  This is commonly used for color-based filtering or "chroma keying."
///
///  # Examples
///
///  ```
///  use imagec::backend::algos::HsvRange;
///
///  let green_filter = HsvRange {
///      min_h: 100.0, max_h: 140.0,
///      min_s: 0.2,   max_s: 1.0,
///      min_v: 0.2,   max_v: 1.0,
///  };
///  ```
#[derive(Serialize, Deserialize, JsonSchema, Debug, Clone, Default)]
#[serde(rename_all = "camelCase")]
pub struct HsvRangeSettings {
    ///  Minimum Hue angle in degrees [0.0, 360.0].
    pub min_h: f32,
    ///  Maximum Hue angle in degrees [0.0, 360.0].
    pub max_h: f32,
    ///  Minimum Saturation normalized [0.0, 1.0].
    pub min_s: f32,
    ///  Maximum Saturation normalized [0.0, 1.0].
    pub max_s: f32,
    ///  Minimum Value (Brightness) normalized [0.0, 1.0].
    pub min_v: f32,
    ///  Maximum Value (Brightness) normalized [0.0, 1.0].
    pub max_v: f32,
}

///  A filter that acts as a synchronization point between the pipeline and a storage backend.
///
///  `ImageCache` allows the pipeline to branch or "undo" operations by saving
///  states to a named address and reloading them as needed.
///
///  # Examples
///
///  ```
///  use imagec::backend::core::context::{ImageCache, ImageCacheMode, ImageAddress};
///  let checkpoint = ImageCache {
///      mode: ImageCacheMode::Store,
///      address: ImageAddress::from("pre_processed_state"),
///  };
///  ```
#[derive(Serialize, Deserialize, JsonSchema, Debug, Clone, Default)]
#[serde(rename_all = "camelCase")]
pub struct ImageCacheSettings {
    ///  Whether to save the current state to the cache or load a state from it.
    pub mode: MathImageCacheImageCacheModeSettings,
    ///  The unique identifier or memory slot where the image is stored.
    pub address: ImageAddress,
}

///  A filter that performs pixel-wise mathematical operations between the current
///  pipeline image and a secondary image stored in the cache.
///
///  This command allows for complex image blending, masking, and comparison.
///
///  # Examples
///
///  ```
///  use imagec::backend::algos::{ImageMath, Operand};
///  let subtract_bg = ImageMath {
///      operand: Operand::Subtract,
///      second_image_address: ImageAddress::from("background"),
///      swap_operands: false,
///  };
///  ```
#[derive(Serialize, Deserialize, JsonSchema, Debug, Clone, Default)]
#[serde(rename_all = "camelCase")]
pub struct ImageMathSettings {
    ///  The specific mathematical or logical operator to apply.
    pub operand: MathImageMathOperandSettings,
    ///  The address of the second image in the [`ImageCache`] to use for the operation.
    pub second_image_address: ImageAddress,
    ///  If false, the calculation is `(Current Image OP Cached Image)`.
    ///  If true, the calculation is `(Cached Image OP Current Image)`.
    ///
    ///  This is critical for non-commutative operations like Subtraction or Division.
    pub swap_operands: bool,
}

///  Configuration for adjusting image contrast and brightness.
///
///  This transformation applies a linear mapping to pixel values.
///  In [`Mode::Manual`], the output is typically calculated as:
///  `output = input * contrast + brightness`.
#[derive(Serialize, Deserialize, JsonSchema, Debug, Clone, Default)]
#[serde(rename_all = "camelCase")]
pub struct IntensityTransformationSettings {
    ///  Determines whether to use automated enhancement or user-defined values.
    pub mode: FiltersIntensityTransformIntensityTransformModeSettings,
    ///  Contrast multiplier (gain).
    ///
    ///  Only active in [`Mode::Manual`].
    ///  Values > 1.0 increase contrast, while values < 1.0 decrease it.
    pub contrast: f32,
    ///  Brightness offset (bias).
    ///
    ///  Only active in [`Mode::Manual`].
    ///  Positive values brighten the image, negative values darken it.
    pub brightness: f32,
}

///  Configuration for the Laplacian edge detection filter.
///
///  The Laplacian is a second-order derivative operator used to find regions of
///  rapid intensity change. It is particularly effective for detecting edges
///  and fine details, though it is highly sensitive to noise.
///
///  # Examples
///
///  ```
///  # use imagec::backend::algos::Laplacian;
///  let filter = Laplacian { kernel_size: 3 };
///  ```
#[derive(Serialize, Deserialize, JsonSchema, Debug, Clone, Default)]
#[serde(rename_all = "camelCase")]
pub struct LaplacianSettings {
    ///  The size of the discrete Laplacian aperture.
    ///
    ///  Typically 3. Larger sizes (5, 7) approximate the Laplacian of Gaussian (LoG)
    ///  more closely but are more computationally expensive. Must be an odd number.
    pub kernel_size: usize,
}

///  A background subtraction filter that uses a median rank operator.
///
///  This algorithm is highly effective for removing large-scale background
///  variations while preserving small, high-contrast features. It works by
///  estimating the background as the median intensity within a local radius.
///
///  # Examples
///
///  ```
///  use imagec::backend::algos::MedianSubtract;
///  let filter = MedianSubtract { radius: 10.0 };
///  ```
#[derive(Serialize, Deserialize, JsonSchema, Debug, Clone, Default)]
#[serde(rename_all = "camelCase")]
pub struct MedianSubtractSettings {
    ///  The radius of the neighborhood used to estimate the background.
    ///
    ///  Features smaller than this radius will be preserved, while
    ///  larger structures will be treated as background and removed.
    pub radius: f64,
}

///  A filter that applies mathematical morphology to an image.
///
///  Morphological operations use a structuring element (kernel) to probe
///  and modify the shapes within an image.
///
///  # Examples
///
///  ```
///  use imagec::backend::algos::{MorphologicalCommand, MorphOps, KernelShapes};
///  let clean_noise = MorphologicalCommand {
///      op: MorphOps::Open,
///      kernel_size: 3,
///      kernel_shape: KernelShapes::Ellipse,
///  };
///  ```
#[derive(Serialize, Deserialize, JsonSchema, Debug, Clone, Default)]
#[serde(rename_all = "camelCase")]
pub struct MorphologicalCommandSettings {
    ///  The transformation type (e.g., Dilate, Erode).
    pub op: MorphologyMorphologicalTransformationMorphOpsSettings,
    ///  The diameter of the structuring element in pixels.
    ///  Must be an odd number (e.g., 3, 5, 7).
    pub kernel_size: usize,
    ///  The geometric profile of the structuring element.
    pub kernel_shape: MorphologyMorphologicalTransformationKernelShapesSettings,
    ///  If set the grayscale image instead of the labeld image is taken to perform a morphological transform
    pub use_grayscale: bool,
}

///  A filter that transforms pixels based on the statistical rank of their neighbors.
///
///  Rank filters are non-linear operators used for noise reduction,
///  morphological operations, and feature enhancement.
///
///  This algorithm sorts (ranks) all pixel values within a local neighborhood
///  window and assigns a specific percentile value to the center pixel. By selecting
///  different ranks, it acts as a configurable operator: the minimum rank performs
///  erosion, the maximum rank performs dilation, and the median rank (50th percentile)
///  provides highly effective impulse noise suppression while preserving sharp structural edges.
#[derive(Serialize, Deserialize, JsonSchema, Debug, Clone, Default)]
#[serde(rename_all = "camelCase")]
pub struct RankFilterSettings {
    ///  The circular radius of the neighborhood to consider.
    ///
    ///  A radius of 1.0 roughly corresponds to a 3x3 square, while larger
    ///  values increase the effect's strength and computational cost.
    pub radius: f64,
    ///  The specific ranking algorithm to apply to the neighborhood.
    pub filter_type: FiltersRankFilterRankFilterTypeSettings,
}

///  Removes non-uniform background illumination by calculating a local intensity baseline.
///
///  This algorithm models the image as a 3D intensity landscape and conceptually rolls
///  a sphere of a user-defined radius underneath it. The ball cannot penetrate narrow
///  intensity peaks (true signal objects) but follows the sweeping, lower-frequency
///  curves of background variations. The path traced by the ball establishes a local
///  baseline map that is subtracted from the original image to isolate foreground features.
#[derive(Serialize, Deserialize, JsonSchema, Debug, Clone)]
#[schemars(default)]
#[serde(rename_all = "camelCase")]
pub struct RollingBallSettings {
    ///  The radius of the ball or paraboloid in pixels.
    ///
    ///  This should be at least as large as the radius of the largest
    ///  object in the image that is not part of the background.
    #[schemars(range(min = 1, max = 64))]
    pub radius: f64,
    ///  The geometric shape of the rolling structural element.
    pub ball_type: FiltersRollingBallBallTypeSettings,
    pub pre_smooth: bool,
}

impl Default for RollingBallSettings {
    fn default() -> Self {
        Self {
            radius: 4.0f64,
            ball_type: FiltersRollingBallBallTypeSettings::default(),
            pre_smooth: bool::default(),
        }
    }
}

///  A command that exports the current image to a persistent file on disk.
///
///  This is a **transparent command**: it does not modify the image data in the
///  pipeline context, nor does it perform a buffer swap. It acts as a tap
///  to view the state of the image at a specific point in the pipeline.
///
///  # Examples
///
///  ```
///  use imagec::backend::algos::SaveImage;
///  let saver = SaveImage {path:"output/processed_cell.png"};
///  ```
#[derive(Serialize, Deserialize, JsonSchema, Debug, Clone, Default)]
#[serde(rename_all = "camelCase")]
pub struct SaveImageSettings {
    ///  The destination filesystem path where the image will be written.
    pub path: PathBuf,
    pub source: MathSaveImageImageSourceSettings,
}

///  Analyzes local image texture, directional orientation, and corner features using a second-moment matrix.
///
///  This algorithm summarizes the predominant directions of the image gradient within a local
///  neighborhood, smoothing the structural data with a Gaussian window. By evaluating the
///  eigenvalues of the resulting matrix tensor, it distinguishes between flat areas (both eigenvalues
///  near zero), straight linear boundaries (one dominant eigenvalue indicating structural direction),
///  and complex corners or intersections (two large eigenvalues).
///
///  # Examples
///
///  ```
///  use imagec::backend::algos::{StructureTensor, Mode};
///  let settings = StructureTensor {
///      mode: Mode::Coherence,
///      kernel_size: 3,
///      sigma: 1.5
///  };
///  ```
#[derive(Serialize, Deserialize, JsonSchema, Debug, Clone, Default)]
#[serde(rename_all = "camelCase")]
pub struct StructureTensorSettings {
    ///  The mathematical output to be produced by the algorithm.
    pub mode: FiltersStructureTensorTensorModeSettings,
    ///  The size of the integration window used to average the local gradients.
    ///
    ///  Larger windows provide more stability against noise but reduce
    ///  spatial resolution.
    pub kernel_size: usize,
    ///  The standard deviation for the Gaussian weighting of the integration window.
    ///
    ///  Controls the spatial "reach" of the neighborhood analysis.
    pub sigma: f32,
}

///  A filter that computes the Gaussian-weighted standard deviation of a local neighborhood.
///
///  Unlike a standard deviation filter which treats all pixels in a window equally,
///  the Weighted Deviation uses a Gaussian kernel to give more importance to
///  pixels closer to the center. This is particularly effective for edge-preserving
///  noise analysis and local contrast enhancement.
///
///  This algorithm evaluates local variance by calculating two distinct Gaussian-blurred
///  baselines across the image: the weighted average of the pixel intensities, and the
///  weighted average of the squared intensities. By subtracting the squared mean from
///  the mean of squares, it yields a localized, smooth statistical variance map that
///  highlights micro-textures and subtle surface boundaries without producing blocky artifacts.
///
///  # Examples
///
///  ```
///  use imagec::backend::algos::WeightedDeviation;
///  let settings = WeightedDeviation {
///      kernel_size: 7,
///      sigma: 2.0,
///  };
///  ```
#[derive(Serialize, Deserialize, JsonSchema, Debug, Clone, Default)]
#[serde(rename_all = "camelCase")]
pub struct WeightedDeviationSettings {
    ///  The size of the local neighborhood window.
    ///
    ///  Must be an odd number. Larger windows capture broader texture
    ///  variations but increase computational load.
    pub kernel_size: usize,
    ///  The standard deviation for the Gaussian weighting function.
    ///
    ///  Defines the "softness" of the neighborhood boundaries. A larger
    ///  sigma includes more of the surrounding context in the deviation calculation.
    pub sigma: f32,
}

// ============ SEGMENTATION ============

///  A filter that segments an image into discrete classes based on intensity.
///
///  This supports "Multi-Otsu" style behavior by allowing a vector of
///  [`ThresholdSettings`]. Each pixel is evaluated against the settings to
///  determine which `object_class_id` it belongs to.
///
///  # Examples
///
///  ```
///  use imagec::backend::algos::{Threshold, ThresholdSettings, ThresholdMethod};
///  let binary = Threshold {
///      thresholds: vec![ThresholdSettings {
///          method: ThresholdMethod::Otsu,
///          min_threshold: 0.0,
///          max_threshold: 1.0,
///          object_class_id: ObjectLabel::Foreground,
///      }]
///  };
///  ```
#[derive(Serialize, Deserialize, JsonSchema, Debug, Clone, Default)]
#[serde(rename_all = "camelCase")]
pub struct ThresholdSettings {
    ///  A list of thresholding layers. Overlapping ranges are resolved
    ///  by the order of the vector (last-in priority).
    pub thresholds: Vec<ThresholdEntrySettings>,
}

///  Configuration for a single thresholding operation within a multi-threshold stack.
#[derive(Serialize, Deserialize, JsonSchema, Debug, Clone)]
#[schemars(default)]
#[serde(rename_all = "camelCase")]
pub struct ThresholdEntrySettings {
    ///  The algorithm to use (Manual or Automatic).
    pub method: SegmentationThresholdThresholdMethodSettings,
    ///  The lower intensity bound. Used directly in `Manual` mode, or as a
    ///  floor for auto-methods.
    #[schemars(range(min = 0, max = 65535))]
    pub min_threshold: f32,
    ///  The upper intensity bound. Used directly in `Manual` mode, or as a
    ///  ceiling for auto-methods.
    #[schemars(range(min = 0, max = 65535))]
    pub max_threshold: f32,
    ///  Unit used for the threshold value.
    ///
    ///  bit: 0 - 255/65535
    ///  %: 0 - 100.0
    ///  rel: 0 - 1.0
    pub unit: PixelUnits,
    ///  The classification ID assigned to pixels falling within this threshold range.
    pub object_class_id: SegmentationClass,
}

impl Default for ThresholdEntrySettings {
    fn default() -> Self {
        Self {
            method: SegmentationThresholdThresholdMethodSettings::default(),
            min_threshold: 0.0f32,
            max_threshold: 65535.0f32,
            unit: PixelUnits::Bit,
            object_class_id: SegmentationClass::default(),
        }
    }
}

// ============ OBJECT ============

///  Identifies and labels discrete objects within a binary or multi-class image.
#[derive(Serialize, Deserialize, JsonSchema, Debug, Clone, Default)]
#[serde(rename_all = "camelCase")]
pub struct ConnectedComponentsSettings {}

///  A morphological segmentation algorithm that splits touching objects using distance topography.
///
///  The Watershed algorithm is a powerful tool for separating overlapping structures (like cells or grains).
///  By analyzing the "shape" of an object via a Distance Transform, it identifies centers of mass
///  and establishes boundaries at the narrowest points of connection.
///
///  This implementation is adaptive:
///  * It can **auto-detect** objects from grayscale intensity peaks.
///  * It can **refine** existing segments if a `U32Label` image is provided as input.
#[derive(Serialize, Deserialize, JsonSchema, Debug, Clone)]
#[schemars(default)]
#[serde(rename_all = "camelCase")]
pub struct WatershedSettings {
    ///  The prominence threshold for peak detection.
    ///
    ///  This value determines how "deep" the valley between two peaks must be to
    ///  keep them as separate objects.
    ///
    ///  * **Low values**: Sensitive to small variations; may cause over-segmentation (splitting one object into many).
    ///  * **High values**: More robust to noise; may cause under-segmentation (failing to split touching objects).
    ///
    ///  In an EDM (Euclidean Distance Map), this value directly corresponds to the
    ///  pixel distance from the edge of the object.
    #[schemars(range(min = 0.1, max = 1))]
    pub maximum_finder_tolerance: f32,
}

impl Default for WatershedSettings {
    fn default() -> Self {
        Self {
            maximum_finder_tolerance: 0.5f32,
        }
    }
}

// ============ MEASURE ============

fn _serde_default_extractrois_max_objects_before_fail() -> i32 {
    100000i32
}
///  Represents a bounded region of interest extracted from a labeled image.
///  A command to extract spatial statistics and bounding boxes from labeled objects.
#[derive(Serialize, Deserialize, JsonSchema, Debug, Clone)]
#[schemars(default)]
#[serde(rename_all = "camelCase")]
pub struct ExtractRoisSettings {
    ///  Maximum allowed ROIs to extract.
    ///
    ///  If this limit is exceeded the pipeline fails.
    ///  This is a protection against memory overload.
    #[schemars(range(min = 100000, max = 100000))]
    #[serde(default = "_serde_default_extractrois_max_objects_before_fail")]
    pub max_objects_before_fail: i32,
}

impl Default for ExtractRoisSettings {
    fn default() -> Self {
        Self {
            max_objects_before_fail: 100000i32,
        }
    }
}

// ============ CLASSIFICATION ============

///  Classifies ROIs based on morphological and intensity features.
///
///  This command applies rule-based classification logic to assign object classes
///  to extracted ROIs. Classification is performed using configurable criteria
///  including area, shape descriptors, and intensity statistics.
#[derive(Serialize, Deserialize, JsonSchema, Debug, Clone)]
#[schemars(default)]
#[serde(rename_all = "camelCase")]
pub struct ClassifyRoisSettings {
    ///  Apply only to objects with this given segmentation class
    ///
    ///  The segmentation class value is assigned to each pixel in the image
    ///  after a Threshold, Pixel classifier or AI classifier.
    ///  If no seg class is selected the criteria are applied to all objects.
    pub origin_segmentation: Vec<SegmentationClass>,
    ///  Apply only to objects with this given class
    pub origin_class: Vec<ObjectClass>,
    ///  Target class for objects matching the chosen criteria.
    ///
    ///  Objects with metrics matching the chosen criteria are labeled with this additional class.
    pub target_class: ObjectClass,
    ///  Unit to use for roi extraction
    pub size_unit: SizeUnits,
    ///  Minimum area size
    ///
    ///  Minimum area size of the object in selected unit (px^2 or nm^2).
    #[schemars(range(min = 0, max = 2147483600))]
    pub min_area: f32,
    ///  Maximum area size
    ///
    ///  Maximum area size of the object in selected unit (px^2 or nm^2).
    #[schemars(range(min = 0, max = 2147483600))]
    pub max_area: f32,
    ///  Circularity range: 0 = elongated, 1 = perfect circle
    ///
    ///  Circularity (sometimes called Isoperimetric Quotient) measures how efficiently a shape encloses its area relative to the length of its perimeter.
    ///  A circle is the mathematically perfect shape for maximizing area while minimizing perimeter.
    ///  It is calculated with `4*Pi*AreaSize / Perimeter^2`
    #[schemars(range(min = 0, max = 1))]
    pub min_circularity: f32,
    ///  Circularity range: 0 = elongated, 1 = perfect circle
    ///
    ///  Circularity (sometimes called Isoperimetric Quotient) measures how efficiently a shape encloses its area relative to the length of its perimeter.
    ///  A circle is the mathematically perfect shape for maximizing area while minimizing perimeter.
    ///  It is calculated with `4*Pi*AreaSize / Perimeter^2`
    #[schemars(range(min = 0, max = 1))]
    pub max_circularity: f32,
    ///  Minimum Solidity/Compactness: 0 = hollow, 1 = perfect convex
    ///
    ///  Solidity is a structural metric used in shape analysis to measure how "solid" or compact an object is.
    ///  It compares the actual area of an object to the area of its Convex Hull (the smallest convex polygon that can completely enclose the object,
    ///  often visualized as a rubber band stretched around the shape).
    ///
    ///  Solidity = 1.0: The object is perfectly convex (e.g., a perfect circle, a solid square, or an ellipse). It has no holes, indentations, or deep recesses.
    ///  Solidity < 1.0: The object has irregular boundaries, deep "bays," protrusions, or internal holes. The lower the value, the more jagged or structurally fragmented the object is.
    #[schemars(range(min = 0, max = 1))]
    pub min_solidity: f32,
    ///  Maximum Solidity/Compactness: 0 = hollow, 1 = perfect convex
    ///
    ///  Solidity is a structural metric used in shape analysis to measure how "solid" or compact an object is.
    ///  It compares the actual area of an object to the area of its Convex Hull (the smallest convex polygon that can completely enclose the object,
    ///  often visualized as a rubber band stretched around the shape).
    ///
    ///  Solidity = 1.0: The object is perfectly convex (e.g., a perfect circle, a solid square, or an ellipse). It has no holes, indentations, or deep recesses.
    ///  Solidity < 1.0: The object has irregular boundaries, deep "bays," protrusions, or internal holes. The lower the value, the more jagged or structurally fragmented the object is.
    #[schemars(range(min = 0, max = 1))]
    pub max_solidity: f32,
    ///  Minimum proportional relationship between an object's width and its height
    ///
    ///  This value is calculated by the object bounding box with and height and is defined with `a = with/height`.
    ///  The value is without unit in the range of 0 to MAX_F32
    #[schemars(range(min = 0, max = 2147483600))]
    pub min_aspect_ratio: f32,
    ///  Maximum proportional relationship between an object's width and its height
    ///
    ///  This value is calculated by the object bounding box with and height and is defined with `a = with/height`.
    ///  The value is without unit in the range of 0 to MAX_F32
    #[schemars(range(min = 0, max = 2147483600))]
    pub max_aspect_ratio: f32,
    ///  Unit used for the intensity value.
    ///
    ///  bit: 0 - 255/65535
    ///  %: 0 - 100.0
    ///  rel: 0 - 1.0
    #[schemars(range(min = 0, max = 65535))]
    pub intensity_unit: PixelUnits,
    ///  The minimum average intensity an object must have in the selected image channel
    #[schemars(range(min = 0, max = 65535))]
    pub min_mean_intensity: f32,
    ///  The maximum average intensity an object is allowed to have in the selected image channel
    #[schemars(range(min = 0, max = 65535))]
    pub max_mean_intensity: f32,
    ///  Eccentricity: 0 = perfect circle, 1 = line
    ///
    ///  Eccentricity is a metric that measures how much a shape deviates from being a perfect circle.
    ///  It imagines the shape as an ellipse and measures how far apart its focal points are.
    ///  It is calculated with `sqrt(1-(b/a)^2)`
    #[schemars(range(min = 0, max = 1))]
    pub min_eccentricity: f32,
    ///  Eccentricity: 0 = perfect circle, 1 = line
    ///
    ///  Eccentricity is a metric that measures how much a shape deviates from being a perfect circle.
    ///  It imagines the shape as an ellipse and measures how far apart its focal points are.
    ///  It is calculated with `sqrt(1-(b/a)^2)`
    #[schemars(range(min = 0, max = 1))]
    pub max_eccentricity: f32,
    ///  Feret diameter threshold
    ///
    ///  The absolute shortest parallel distance across the object.
    ///  This represents the minimum sieve size a particle could pass through.
    ///
    ///  In image processing and particle size analysis, the Feret diameter (often called the caliper diameter) is a metric used to measure the size of an irregular object.
    ///  It mimics the action of a slide caliper, measuring the distance between two parallel tangential lines bounding the object at a specific angle.
    ///  When analyzing objects or particles, applying Feret diameter thresholds allows you to filter out noise, classify objects by shape, or isolate specific structures based on their directional length rather than their total area.
    #[schemars(range(min = 0, max = 2147483600))]
    pub min_feret: f32,
    ///  Maximum feret diameter threshold in selected unit (px or nm)
    ///
    ///  The absolute longest distance across the object at any angle.
    ///  Used to measure elongation or the maximum length of a particle.
    ///
    ///  In image processing and particle size analysis, the Feret diameter (often called the caliper diameter) is a metric used to measure the size of an irregular object.
    ///  It mimics the action of a slide caliper, measuring the distance between two parallel tangential lines bounding the object at a specific angle.
    ///  When analyzing objects or particles, applying Feret diameter thresholds allows you to filter out noise, classify objects by shape, or isolate specific structures based on their directional length rather than their total area.
    #[schemars(range(min = 0, max = 2147483600))]
    pub max_feret: f32,
    ///  Whether ROI can touch image edge
    pub allow_edge_touching: bool,
}

impl Default for ClassifyRoisSettings {
    fn default() -> Self {
        Self {
            origin_segmentation: vec![],
            origin_class: vec![],
            target_class: ObjectClass::Unset,
            size_unit: SizeUnits::NanoMeter,
            min_area: 0.0f32,
            max_area: 2147483648.0f32,
            min_circularity: 0.0f32,
            max_circularity: 1.0f32,
            min_solidity: 0.0f32,
            max_solidity: 1.0f32,
            min_aspect_ratio: 0.0f32,
            max_aspect_ratio: 2147483648.0f32,
            intensity_unit: PixelUnits::Bit,
            min_mean_intensity: 0.0f32,
            max_mean_intensity: 65535.0f32,
            min_eccentricity: 0.0f32,
            max_eccentricity: 1.0f32,
            min_feret: 0.0f32,
            max_feret: 2147483648.0f32,
            allow_edge_touching: true,
        }
    }
}

///  Calculates spatial colocalization and intersections between specified object classes.
///
///  This command scans the ROI cache, groups objects by their designated classes,
///  and performs spatial overlap analysis. It records colocalization relationships
///  between intersecting entities and can optionally generate new child ROIs representing
///  the precise intersection regions.
#[derive(Serialize, Deserialize, JsonSchema, Debug, Clone, Default)]
#[serde(rename_all = "camelCase")]
pub struct ColocalizationSettings {
    ///  Theses are the classes the coloclization should be calculated for
    pub classes_to_coloc: Vec<ObjectClass>,
    ///  Optional additional label filters.
    ///
    ///  Only classes which matches all of these filters are used for coloc calculation
    pub filter_classes: Vec<ObjectClass>,
    ///  Class of the overlapping area if needed
    ///
    ///  If defined the overlapping coloc area is added as new ROI and labeled with this class
    pub class_for_overlapping_areas: ObjectClass,
    ///  If set one object is allowed to coloc with more than one other object
    pub allow_multi_object_coloc: bool,
    pub size_unit: SizeUnits,
    ///  Minimum overlapping area size to count objects as coloc
    pub min_coloc_area: f32,
}

///  Computes a Voronoi tessellation from segmented seed objects.
///
///  Each seed center expands outward until it reaches another region, the optional mask
///  boundary, or the maximum radius. The resulting areas are stored as new ROIs labeled
///  with `output_class` and linked to their originating center object.
#[derive(Serialize, Deserialize, JsonSchema, Debug, Clone, Default)]
#[serde(rename_all = "camelCase")]
pub struct VoronoiSettings {
    ///  Object class whose instances act as Voronoi seed points.
    pub centers: ObjectClass,
    ///  Additional label filters applied to center objects before tessellation.
    ///
    ///  Only center objects that carry all listed classes pass the filter.
    ///  Leave empty to include all objects of `centers`.
    pub center_filter_classes: Vec<ObjectClass>,
    ///  Object class used to spatially constrain the Voronoi areas.
    ///
    ///  Each computed Voronoi region is intersected with the union of all mask objects,
    ///  discarding pixels that fall outside the mask. Set to `Unset` to expand
    ///  to the full image boundary instead.
    pub mask: ObjectClass,
    ///  Additional label filters applied to mask objects.
    ///
    ///  Only mask objects that carry all listed classes pass the filter.
    ///  Leave empty to include all objects of `mask`.
    pub mask_filter_classes: Vec<ObjectClass>,
    ///  Object class assigned to the resulting Voronoi region ROIs.
    pub output_class: ObjectClass,
    ///  Unit in which `max_radius` is expressed (e.g. pixels, nm, µm).
    pub unit: SizeUnits,
    ///  Maximum expansion radius for a Voronoi region.
    ///
    ///  Pixels farther than this distance from the nearest seed center are excluded
    ///  from the region. Use `0` or a negative value to disable the limit.
    pub max_radius: f32,
    ///  Discard Voronoi regions that touch the image border.
    pub exclude_areas_at_the_edges: bool,
    ///  Discard Voronoi regions whose originating center object was filtered out or missing.
    pub exclude_areas_with_no_center: bool,
}
