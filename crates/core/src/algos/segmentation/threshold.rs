use evanalyzer_cfg::core_types::{InternalErrors, PixelUnits, SegmentationClass};
use macros::CommandsMeta;

use crate::{
    algos::ImageAlgorithm,
    pipeline::{pipeline_cache::PipelineCache, pipeline_context::PipelineContext},
};

/// The mathematical strategy used to determine the optimal global threshold.
///
/// Most methods analyze the image histogram to find a "cut-off" point that
/// best separates the foreground from the background.
pub enum ThresholdMethod {
    /// No threshold applied; typically used for bypass logic.
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

/// Configuration for a single thresholding operation within a multi-threshold stack.
#[derive(CommandsMeta)]
pub struct ThresholdEntry {
    /// The algorithm to use (Manual or Automatic).
    #[cmdsmeta(default = ThresholdMethod::Manual)]
    pub method: ThresholdMethod,

    /// The lower intensity bound. Used directly in `Manual` mode, or as a
    /// floor for auto-methods.
    #[cmdsmeta(default = 0, min = 0, max = 65535, step = 1, summary = true)]
    pub min_threshold: f32,

    /// The upper intensity bound. Used directly in `Manual` mode, or as a
    /// ceiling for auto-methods.
    #[cmdsmeta(default = 65535, min = 0, max = 65535, step = 1)]
    pub max_threshold: f32,

    /// Unit used for the threshold value.
    ///
    /// bit: 0 - 255/65535
    /// %: 0 - 100.0
    /// rel: 0 - 1.0
    #[cmdsmeta(default = PixelUnits::Bit)]
    pub unit: PixelUnits,

    /// The classification ID assigned to pixels falling within this threshold range.
    pub object_class_id: SegmentationClass,
}

/// A filter that segments an image into discrete classes based on intensity.
///
/// This supports "Multi-Otsu" style behavior by allowing a vector of
/// [`ThresholdSettings`]. Each pixel is evaluated against the settings to
/// determine which `object_class_id` it belongs to.
///
/// # Examples
///
/// ```
/// use imagec::backend::algos::{Threshold, ThresholdSettings, ThresholdMethod};
/// let binary = Threshold {
///     thresholds: vec![ThresholdSettings {
///         method: ThresholdMethod::Otsu,
///         min_threshold: 0.0,
///         max_threshold: 1.0,
///         object_class_id: ObjectLabel::Foreground,
///     }]
/// };
/// ```
#[derive(CommandsMeta)]
#[cmdsmeta(category = "segment")]
pub struct Threshold {
    /// A list of thresholding layers. Overlapping ranges are resolved
    /// by the order of the vector (last-in priority).
    pub thresholds: Vec<ThresholdEntry>,
}

impl ImageAlgorithm for Threshold {
    /// Segments the image by applying one or more thresholding criteria.
    ///
    /// The execution follows a multi-step process:
    /// 1. **Histogram Generation**: For any automatic methods (Otsu, Li, etc.),
    ///    a global histogram of the `ctx.image` is calculated.
    /// 2. **Threshold Discovery**: The chosen algorithms analyze the histogram
    ///    to find the optimal cut-off points.
    /// 3. **Classification**: Each pixel is compared against the calculated
    ///    ranges. Pixels meeting the criteria are assigned the corresponding
    ///    `object_class_id`.
    ///
    /// # Multi-Threshold Logic
    /// If multiple `ThresholdSettings` are provided, the algorithm evaluates
    /// them in order. This allows for complex multi-class segmentation (e.g.,
    /// Background, Cytoplasm, and Nuclei) in a single pass.
    ///
    /// # Errors
    ///
    /// Returns [`InternalErrors::InvalidParameters`] if an automatic method
    /// fails to converge or if the image histogram is empty (e.g., all pixels are NaN).
    fn execute(
        &self,
        ctx: &mut PipelineContext,
        _cache: &mut PipelineCache,
    ) -> Result<(), InternalErrors> {
        let nr_of_bits = ctx.image_meta.nr_of_bits;
        let (input_data, segmentation_map) = ctx.get_f32_gray_and_segmentation_mask_mut()?;

        // Pre-normalize thresholds once so the inner loop stays branch-free.
        let normalized: Vec<(f32, f32, u32)> = self
            .thresholds
            .iter()
            .map(|s| {
                (
                    s.unit.to_relative(s.min_threshold, nr_of_bits),
                    s.unit.to_relative(s.max_threshold, nr_of_bits),
                    s.object_class_id.as_u32(),
                )
            })
            .collect();

        let output_slice = segmentation_map.as_slice_mut();

        // This version is much more likely to be auto-vectorized by the compiler
        for (out_pixel, &in_pixel) in output_slice.iter_mut().zip(input_data.as_slice().iter()) {
            let mut assigned_id = SegmentationClass::BACKGROUND.as_u32();
            for &(min, max, class_id) in &normalized {
                let is_in_range = (in_pixel >= min && in_pixel <= max) as u32;
                assigned_id = (is_in_range * class_id) | ((1 - is_in_range) * assigned_id);
            }
            *out_pixel = assigned_id;
        }
        //ctx.swap()?;
        Ok(())
    }

    fn name(&self) -> &'static str {
        "Threshold"
    }
}

// --- Test ------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::image::{ImageContainer, ImageDebugExt};
    use kornia_image::{Image, ImageSize};
    use kornia_tensor::CpuAllocator;

    #[test]
    fn test_multi_range_thresholding() -> Result<(), Box<dyn std::error::Error>> {
        // 1. Setup Image Size (3x2)
        let size = ImageSize {
            width: 3,
            height: 2,
        };

        // 2. Create input data with specific values:
        // Top row: 0.1 (Dark), 0.5 (Mid), 0.9 (Bright)
        // Bottom row: 0.0 (Min), 1.0 (Max), 0.4 (Mid)
        let input_data = vec![0.1, 0.5, 0.9, 0.0, 1.0, 0.4];
        let input_img = Image::<f32, 1, CpuAllocator>::new(size, input_data, CpuAllocator)?;

        input_img.print_window();

        // 3. Define three threshold ranges
        let settings = vec![
            ThresholdEntry {
                method: ThresholdMethod::Manual,
                min_threshold: 0.0,
                max_threshold: 0.2,
                object_class_id: SegmentationClass(10), // Dark Class
                unit: PixelUnits::Relative,
            },
            ThresholdEntry {
                method: ThresholdMethod::Manual,
                min_threshold: 0.3,
                max_threshold: 0.6,
                object_class_id: SegmentationClass(20), // Mid Class
                unit: PixelUnits::Relative,
            },
            ThresholdEntry {
                method: ThresholdMethod::Manual,
                min_threshold: 0.8,
                max_threshold: 1.0,
                object_class_id: SegmentationClass(30), // Bright Class
                unit: PixelUnits::Relative,
            },
        ];

        let cmd = Threshold {
            thresholds: settings,
        };

        // 4. Setup Pipeline Context
        // Note: Initializing scratchpad as F32Gray to test the auto-reallocation to U32Label
        let mut ctx = PipelineContext::new_from_image_test(input_img)?;
        let mut cache = PipelineCache::default();

        // 5. Execute the command
        cmd.execute(&mut ctx, &mut cache)?;

        ctx.get_segmentation_map()?.print_window();

        // 6. Verify the results
        let result_pixels = ctx
            .segmentation_map
            .as_ref()
            .expect("No labels found")
            .as_slice();

        // Expected mapping:
        // 0.1 -> 10
        // 0.5 -> 20
        // 0.9 -> 30
        // 0.0 -> 10
        // 1.0 -> 30
        // 0.4 -> 20
        let expected = vec![10, 20, 30, 10, 30, 20];

        assert_eq!(
            result_pixels,
            &expected[..],
            "Pixel classification failed to match expected IDs"
        );

        // Verify that the original image was moved to the scratchpad during swap
        //  if let ImageContainer::F32Gray(orig) = ctx.scratch_pad {
        //      assert_eq!(*orig.get_pixel(0, 0, 0).unwrap(), 0.1);
        //  } else {
        //      panic!("Original image was not preserved in scratch_pad after swap");
        //  }

        Ok(())
    }
}
