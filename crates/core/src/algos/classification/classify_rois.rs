//! # Classification Module
//!
//! Provides functionality for classifying ROIs based on morphological and intensity metrics.
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-05-05
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **AGPL-3.0**.
//!
//! ## Overview
//! This module implements ROI classification algorithms that assign object classes
//! to regions of interest based on configurable criteria and machine learning models.

use crate::{
    algos::ImageAlgorithm,
    image::PixelSizes,
    roi::Roi, // ... other imports
};
use evanalyzer_cfg::core_types::{
    InternalErrors,
    ObjectClass::{self, Unset},
    PixelUnits, SegmentationClass, SizeUnits,
};
use log::{debug, info, warn};
use macros::CommandsMeta;

pub enum ClassifyMatchHandling {
    AddOutputClassIfMatch,
    AddOutputClassIfNotMatch,
    RemoveInputClassIfMatch,
    RemoveInputClassIfNotMatch,
    RemoveOutputClassIfMatch,
    RemoveOutputClassIfNotMatch,
    RemoveAllClassesIfMatch,
    RemoveAllClassesIfNotMatch,
}

/// Classifies ROIs based on morphological and intensity features.
///
/// This command applies rule-based classification logic to assign object classes
/// to extracted ROIs. Classification is performed using configurable criteria
/// including area, shape descriptors, and intensity statistics.
#[derive(CommandsMeta)]
#[cmdsmeta(category = "classify")]
pub struct ClassifyRois {
    /// Apply only to objects with this given segmentation class
    ///
    /// The segmentation class value is assigned to each pixel in the image
    /// after a Threshold, Pixel classifier or AI classifier.
    /// If no seg class is selected the criteria are applied to all objects.
    #[cmdsmeta(visible = false)]
    pub origin_segmentation: Vec<SegmentationClass>,

    /// Restrict classification to objects that already carry one of these classes
    ///
    /// Only ROIs that have been assigned at least one of the listed classes by a prior
    /// pipeline step will be evaluated against the morphological and intensity criteria below.
    /// Leave empty to apply the criteria to every object regardless of its current class.
    #[cmdsmeta(display_name = "Input Classes")]
    pub input_classes: Vec<ObjectClass>,

    /// What to do with object class labels after criteria evaluation
    ///
    /// Controls whether the output class is added or existing classes are removed,
    /// and whether the action is triggered on a criteria **match** or a **non-match**:
    ///
    /// - **AddOutputClassIfMatch** - append the output class to objects that pass the criteria.
    /// - **AddOutputClassIfNotMatch** - append the output class to objects that fail the criteria.
    /// - **RemoveInputClassIfMatch / NotMatch** - strip all input classes from matching / non-matching objects.
    /// - **RemoveOutputClassIfMatch / NotMatch** - strip the output class from matching / non-matching objects.
    /// - **RemoveAllClassesIfMatch / NotMatch** - clear every class label from matching / non-matching objects.
    #[cmdsmeta(default = ClassifyMatchHandling::RemoveAllClassesIfNotMatch)]
    pub match_handling: ClassifyMatchHandling,

    /// Class label assigned to (or removed from) objects by the chosen operation
    ///
    /// Used as the target class for `AddOutputClass*` and `RemoveOutputClass*` operations.
    /// Has no effect when the selected operation only manipulates input classes or clears all classes.
    #[cmdsmeta(default = ObjectClass::Unset, display_name = "Output Class")]
    pub output_class: ObjectClass,

    /// Unit to use for roi extraction
    #[cmdsmeta(
        default = SizeUnits::NanoMeter,
    )]
    pub size_unit: SizeUnits,

    /// Minimum area size
    ///
    /// Minimum area size of the object in selected unit (px^2 or nm^2).
    #[cmdsmeta(default = 0, min = 0.0, max = 2147483648.0, summary = true)]
    pub min_area: f32,

    /// Maximum area size
    ///
    /// Maximum area size of the object in selected unit (px^2 or nm^2).
    #[cmdsmeta(default = 2147483648.0, min = 0.0, max = 2147483648.0, summary = false)]
    pub max_area: f32,

    /// Circularity range: 0 = elongated, 1 = perfect circle
    ///
    /// Circularity (sometimes called Isoperimetric Quotient) measures how efficiently a shape encloses its area relative to the length of its perimeter.
    /// A circle is the mathematically perfect shape for maximizing area while minimizing perimeter.
    /// It is calculated with `4*Pi*AreaSize / Perimeter^2`
    #[cmdsmeta(default = 0.0, min = 0.0, max = 1.0, step = 0.1, summary = true)]
    pub min_circularity: f32,

    /// Circularity range: 0 = elongated, 1 = perfect circle
    ///
    /// Circularity (sometimes called Isoperimetric Quotient) measures how efficiently a shape encloses its area relative to the length of its perimeter.
    /// A circle is the mathematically perfect shape for maximizing area while minimizing perimeter.
    /// It is calculated with `4*Pi*AreaSize / Perimeter^2`
    #[cmdsmeta(default = 1.0, min = 0.0, max = 1.0, step = 0.1, summary = false)]
    pub max_circularity: f32,

    /// Minimum Solidity/Compactness: 0 = hollow, 1 = perfect convex
    ///
    /// Solidity is a structural metric used in shape analysis to measure how "solid" or compact an object is.
    /// It compares the actual area of an object to the area of its Convex Hull (the smallest convex polygon that can completely enclose the object,
    /// often visualized as a rubber band stretched around the shape).
    ///
    /// Solidity = 1.0: The object is perfectly convex (e.g., a perfect circle, a solid square, or an ellipse). It has no holes, indentations, or deep recesses.
    /// Solidity < 1.0: The object has irregular boundaries, deep "bays," protrusions, or internal holes. The lower the value, the more jagged or structurally fragmented the object is.
    #[cmdsmeta(default = 0.0, min = 0.0, max = 1.0, step = 0.1, summary = false)]
    pub min_solidity: f32,

    /// Maximum Solidity/Compactness: 0 = hollow, 1 = perfect convex
    ///
    /// Solidity is a structural metric used in shape analysis to measure how "solid" or compact an object is.
    /// It compares the actual area of an object to the area of its Convex Hull (the smallest convex polygon that can completely enclose the object,
    /// often visualized as a rubber band stretched around the shape).
    ///
    /// Solidity = 1.0: The object is perfectly convex (e.g., a perfect circle, a solid square, or an ellipse). It has no holes, indentations, or deep recesses.
    /// Solidity < 1.0: The object has irregular boundaries, deep "bays," protrusions, or internal holes. The lower the value, the more jagged or structurally fragmented the object is.
    #[cmdsmeta(default = 1.0, min = 0.0, max = 1.0, step = 0.1, summary = false)]
    pub max_solidity: f32,

    /// Minimum proportional relationship between an object's width and its height
    ///
    /// This value is calculated by the object bounding box with and height and is defined with `a = with/height`.
    /// The value is without unit in the range of 0 to MAX_F32
    #[cmdsmeta(default = 0.0, min = 0.0, max = 2147483648.0, summary = false)]
    pub min_aspect_ratio: f32,

    /// Maximum proportional relationship between an object's width and its height
    ///
    /// This value is calculated by the object bounding box with and height and is defined with `a = with/height`.
    /// The value is without unit in the range of 0 to MAX_F32
    #[cmdsmeta(
        default = 2147483648.0,
        min = 0.0,
        max = 2147483648.0,
        step = 1.0,
        summary = false
    )]
    pub max_aspect_ratio: f32,

    /// Eccentricity: 0 = perfect circle, 1 = line
    ///
    /// Eccentricity is a metric that measures how much a shape deviates from being a perfect circle.
    /// It imagines the shape as an ellipse and measures how far apart its focal points are.
    /// It is calculated with `sqrt(1-(b/a)^2)`
    #[cmdsmeta(default = 0.0, min = 0.0, max = 1.0, step = 0.1, summary = true)]
    pub min_eccentricity: f32,

    /// Eccentricity: 0 = perfect circle, 1 = line
    ///
    /// Eccentricity is a metric that measures how much a shape deviates from being a perfect circle.
    /// It imagines the shape as an ellipse and measures how far apart its focal points are.
    /// It is calculated with `sqrt(1-(b/a)^2)`
    #[cmdsmeta(default = 1.0, min = 0.0, max = 1.0, step = 0.1, summary = true)]
    pub max_eccentricity: f32,

    /// Feret diameter threshold
    ///
    /// The absolute shortest parallel distance across the object.
    /// This represents the minimum sieve size a particle could pass through.
    ///
    /// In image processing and particle size analysis, the Feret diameter (often called the caliper diameter) is a metric used to measure the size of an irregular object.
    /// It mimics the action of a slide caliper, measuring the distance between two parallel tangential lines bounding the object at a specific angle.
    /// When analyzing objects or particles, applying Feret diameter thresholds allows you to filter out noise, classify objects by shape, or isolate specific structures based on their directional length rather than their total area.
    #[cmdsmeta(default = 0, min = 0.0, max = 2147483648.0, summary = false, step = 1)]
    pub min_feret: f32,

    /// Maximum feret diameter threshold in selected unit (px or nm)
    ///
    /// The absolute longest distance across the object at any angle.
    /// Used to measure elongation or the maximum length of a particle.
    ///
    /// In image processing and particle size analysis, the Feret diameter (often called the caliper diameter) is a metric used to measure the size of an irregular object.
    /// It mimics the action of a slide caliper, measuring the distance between two parallel tangential lines bounding the object at a specific angle.
    /// When analyzing objects or particles, applying Feret diameter thresholds allows you to filter out noise, classify objects by shape, or isolate specific structures based on their directional length rather than their total area.
    #[cmdsmeta(
        default = 2147483648.0,
        min = 0.0,
        max = 2147483648.0,
        summary = false,
        step = 1
    )]
    pub max_feret: f32,

    /// Whether ROI can touch image edge
    #[cmdsmeta(default = true, summary = true)]
    pub allow_edge_touching: bool,
}

impl Default for ClassifyRois {
    fn default() -> Self {
        Self {
            min_area: 0.0,
            max_area: 2147483648.0,
            min_circularity: 0.0,
            max_circularity: 1.0,
            min_solidity: 0.0,
            max_solidity: 1.0,
            min_aspect_ratio: 0.0,
            max_aspect_ratio: 2147483648.0,
            min_eccentricity: 0.0,
            max_eccentricity: 1.0,
            min_feret: 0.0,
            max_feret: 2147483648.0,
            allow_edge_touching: true,
            size_unit: SizeUnits::NanoMeter,
            origin_segmentation: vec![],
            output_class: Unset,
            input_classes: vec![],
            match_handling: ClassifyMatchHandling::RemoveAllClassesIfNotMatch,
        }
    }
}

impl ImageAlgorithm for ClassifyRois {
    fn execute(
        &self,
        ctx: &mut crate::pipeline::pipeline_context::PipelineContext,
        cache: &mut crate::pipeline::pipeline_cache::PipelineCache,
    ) -> Result<(), InternalErrors> {
        let px_size = ctx.pixel_sizes();

        // Iterate through all ROIs in the cache
        for roi in cache.roi_cache.values_mut() {
            // Skip ROIs that don't carry any of the required input classes
            if !self.input_classes.is_empty() && !roi.has_object_classes(&self.input_classes) {
                continue;
            }

            let matches = self.matches_criteria(roi, px_size);
            match self.match_handling {
                ClassifyMatchHandling::AddOutputClassIfMatch => {
                    if matches {
                        roi.add_object_class(self.output_class.clone());
                    }
                }
                ClassifyMatchHandling::AddOutputClassIfNotMatch => {
                    if !matches {
                        roi.add_object_class(self.output_class.clone());
                    }
                }
                ClassifyMatchHandling::RemoveInputClassIfMatch => {
                    if matches {
                        for class in &self.input_classes {
                            roi.remove_object_class(class);
                        }
                    }
                }
                ClassifyMatchHandling::RemoveInputClassIfNotMatch => {
                    if !matches {
                        for class in &self.input_classes {
                            roi.remove_object_class(class);
                        }
                    }
                }
                ClassifyMatchHandling::RemoveOutputClassIfMatch => {
                    if matches {
                        roi.remove_object_class(&self.output_class);
                    }
                }
                ClassifyMatchHandling::RemoveOutputClassIfNotMatch => {
                    if !matches {
                        roi.remove_object_class(&self.output_class);
                    }
                }
                ClassifyMatchHandling::RemoveAllClassesIfMatch => {
                    if matches {
                        roi.object_class.clear();
                    }
                }
                ClassifyMatchHandling::RemoveAllClassesIfNotMatch => {
                    if !matches {
                        roi.object_class.clear();
                    }
                }
            }
        }

        Ok(())
    }

    fn name(&self) -> &'static str {
        "ClassifyRois"
    }
}

impl ClassifyRois {
    /// Evaluates whether an ROI matches the classification criteria.
    ///
    /// # Arguments
    /// * `roi` - The ROI to evaluate
    ///
    /// # Returns
    /// `true` if the ROI matches all criteria, `false` otherwise
    fn matches_criteria(&self, roi: &Roi, pixel_sizes: &PixelSizes) -> bool {
        let pixel_area_size_nm = pixel_sizes.px_size_x * pixel_sizes.px_size_y;
        let min_area_px = self.size_unit.to_pixel(self.min_area, pixel_area_size_nm);
        let max_area_px = self.size_unit.to_pixel(self.max_area, pixel_area_size_nm);

        // Check area
        if roi.area < min_area_px || roi.area > max_area_px {
            debug!(
                "ROI {} failed area check: {} (range: {}-{})",
                roi.id, roi.area, min_area_px, max_area_px
            );
            return false;
        }

        // Check edge touching
        if roi.touches_edge && !self.allow_edge_touching {
            debug!("ROI {} touches image edge", roi.id);
            return false;
        }

        // Check circularity
        let _perimeter = roi.get_perimeter();
        let circularity = roi.circularity();
        if circularity < self.min_circularity || circularity > self.max_circularity {
            debug!(
                "ROI {} failed circularity check: {:.4} (range: {:.4}-{:.4})",
                roi.id, circularity, self.min_circularity, self.max_circularity
            );
            return false;
        }

        // Check solidity
        let solidity = roi.get_solidity();
        if solidity < self.min_solidity || solidity > self.max_solidity {
            debug!(
                "ROI {} failed solidity check: {:.4} (range: {:.4}-{:.4})",
                roi.id, solidity, self.min_solidity, self.max_solidity
            );
            return false;
        }

        // Check aspect ratio
        let aspect_ratio = roi.get_aspect_ratio();
        if aspect_ratio < self.min_aspect_ratio || aspect_ratio > self.max_aspect_ratio {
            debug!(
                "ROI {} failed aspect ratio check: {:.4} (range: {:.4}-{:.4})",
                roi.id, aspect_ratio, self.min_aspect_ratio, self.max_aspect_ratio
            );
            return false;
        }

        // Check eccentricity (from ellipse fitting)
        let ellipse = roi.get_ellipse();
        if ellipse.eccentricity < self.min_eccentricity
            || ellipse.eccentricity > self.max_eccentricity
        {
            debug!(
                "ROI {} failed eccentricity check: {:.4} (range: {:.4}-{:.4})",
                roi.id, ellipse.eccentricity, self.min_eccentricity, self.max_eccentricity
            );
            return false;
        }

        // Check Feret diameter
        let feret = roi.get_feret_diameter();
        if feret < self.min_feret || feret > self.max_feret {
            debug!(
                "ROI {} failed Feret diameter check: {:.2} (range: {:.2}-{:.2})",
                roi.id, feret, self.min_feret, self.max_feret
            );
            return false;
        }

        true
    }
}

#[cfg(test)]
mod tests {
    use bitvec::vec;

    use super::*;

    #[test]
    fn test_classification_criteria_default() {
        let criteria = ClassifyRois::default();
        assert_eq!(criteria.min_area, 0.0);
        assert_eq!(criteria.max_area, 2147483600.0);
        assert!(criteria.allow_edge_touching);
    }

    #[test]
    fn test_classifier_creation() {
        let class = ObjectClass::default();
        let classifier = ClassifyRois::default();
        assert_eq!(classifier.target_class, class);
    }
}
