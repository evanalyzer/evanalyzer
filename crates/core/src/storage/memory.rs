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
        // Build every RoiSettings without touching the lock - this is the
        // expensive part (each conversion clones the Roi's fields, including
        // its mask), so doing it here keeps other threads running smoothly.
        let roi_settings: Vec<RoiSettings> = cache
            .roi_cache
            .values()
            .map(|roi| roi.to_roi_settings())
            .collect();

        // Acquire the lock at the absolute last second and move all items in at once.
        let mut mut_cache = self
            .out_rois
            .lock()
            .map_err(|_| InternalErrors::Io("Failed to acquire MemoryExporter lock".to_string()))?;

        mut_cache.extend(roi_settings);

        Ok(()) // Lock is released immediately here
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::roi::{Roi, RoiInit};
    use evanalyzer_cfg::core_types::ObjectId;
    use std::collections::HashSet;

    #[test]
    fn export_moves_every_roi_into_out_rois_without_dropping_or_duplicating_any() {
        let mut cache = PipelineCache::default();
        let id_a = ObjectId::next();
        let id_b = ObjectId::next();
        cache.roi_cache.insert(
            id_a.clone(),
            Roi::new(RoiInit {
                id: id_a,
                area: 10,
                bbox: [0, 0, 3, 3],
                ..Default::default()
            }),
        );
        cache.roi_cache.insert(
            id_b.clone(),
            Roi::new(RoiInit {
                id: id_b,
                area: 20,
                bbox: [1, 1, 4, 4],
                ..Default::default()
            }),
        );

        let exporter = MemoryExporter {
            out_rois: Arc::new(Mutex::new(Vec::new())),
        };
        exporter.export(&cache).unwrap();

        let exported = exporter.out_rois.lock().unwrap();
        assert_eq!(exported.len(), 2);
        let areas: HashSet<usize> = exported.iter().map(|r| r.area).collect();
        assert_eq!(areas, HashSet::from([10, 20]));
    }

    #[test]
    fn export_appends_to_existing_out_rois_rather_than_overwriting() {
        let mut cache = PipelineCache::default();
        let id = ObjectId::next();
        cache.roi_cache.insert(
            id.clone(),
            Roi::new(RoiInit {
                id,
                area: 5,
                bbox: [0, 0, 1, 1],
                ..Default::default()
            }),
        );

        let exporter = MemoryExporter {
            out_rois: Arc::new(Mutex::new(vec![])),
        };
        exporter.export(&cache).unwrap();
        exporter.export(&cache).unwrap(); // second tile's export call

        assert_eq!(
            exporter.out_rois.lock().unwrap().len(),
            2,
            "each export() call must append, not replace, the accumulated results"
        );
    }
}
