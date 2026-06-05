use crate::Roi;
use crate::pipeline::pipeline_cache::PipelineCache;
use crate::storage::PipelineResultExporter;
use evanalyzer_cfg::{core_types::InternalErrors, settings::roi_settings::RoiSettings};
use std::sync::{Arc, Mutex};

/// Exports pipeline analysis results to a CSV file with comprehensive morphological,
/// intensity, and spatial relationship colocalization metrics.
///
/// This exporter includes:
/// - **Morphological metrics**: area, perimeter, circularity, solidity, aspect ratio, feret diameters
/// - **Spatial metrics**: bounding box, centroid coordinates, major axis angle
/// - **Intensity metrics**: mean, median, std dev, min, max, integrated density across channels
/// - **Classification & Lineage**: segmentation class, object class labels
/// - **Colocalization metrics**: Dynamic columns for each `ObjectClass`, containing comma-separated lists of overlapping `ObjectId`s
pub struct MemoryExporter {
    pub out_rois: Arc<Mutex<Vec<RoiSettings>>>,
}

impl PipelineResultExporter for MemoryExporter {
    fn export(&self, cache: &PipelineCache) -> Result<(), InternalErrors> {
        // Allocate space and clone data in parallel/seperately without touching the lock.
        // This keeps other threads running smoothly.
        let mut cloned_rois: Vec<Roi> = Vec::with_capacity(cache.roi_cache.len());
        for roi in cache.roi_cache.values() {
            cloned_rois.push(roi.clone());
        }

        // Acquire the lock at the absolute last second and push all items at once.
        let mut mut_cache = self
            .out_rois
            .lock()
            .map_err(|_| InternalErrors::Io("Failed to acquire MemoryExporter lock".to_string()))?;

        cloned_rois
            .iter()
            .for_each(|e| mut_cache.push(e.to_roi_settings()));

        Ok(()) // Lock is released immediately here
    }
}
