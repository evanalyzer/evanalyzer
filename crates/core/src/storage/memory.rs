use crate::pipeline::pipeline_cache::GlobalPipelineCache;
use crate::storage::stored_object::{ObjectFilter, StoredObject};
use crate::storage::{DatasetStore, PipelineResultExporter};
use evanalyzer_cfg::{
    core_types::{InternalErrors, ObjectClass, ObjectId, PipelineMeasurements, TrackId},
    settings::object_settings::ObjectMetricSettings,
};
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
#[derive(Default)]
pub struct MemoryExporter {
    pub out_objects: Arc<Mutex<Vec<ObjectMetricSettings>>>,
    /// The measurements, per plane - like the objects, for a preview.
    pub out_measurements: Arc<Mutex<Vec<PipelineMeasurements>>>,
}

impl PipelineResultExporter for MemoryExporter {
    fn export(&self, cache: &GlobalPipelineCache) -> Result<(), InternalErrors> {
        // Build every ObjectMetricSettings without touching the lock - this is the
        // expensive part (each conversion clones the Object's fields, including
        // its mask), so doing it here keeps other threads running smoothly.
        let object_settings: Vec<ObjectMetricSettings> = cache
            .object_cache
            .values()
            .map(|object| object.to_object_settings())
            .collect();

        // Acquire the lock at the absolute last second and move all items in at once.
        let mut mut_cache = self
            .out_objects
            .lock()
            .map_err(|_| InternalErrors::Io("Failed to acquire MemoryExporter lock".to_string()))?;

        mut_cache.extend(object_settings);
        drop(mut_cache);

        if !cache.measurements.measurements.is_empty() {
            self.out_measurements
                .lock()
                .map_err(|_| {
                    InternalErrors::Io("Failed to acquire MemoryExporter lock".to_string())
                })?
                .push(cache.measurements.clone());
        }
        Ok(())
    }

    fn dataset_store(&self) -> Option<&dyn DatasetStore> {
        Some(self)
    }
}

/// The objects don't record their image, so this store has one image - a
/// preview run's - with an empty `image_rel_path`, and an image filter
/// matches only that.
impl DatasetStore for MemoryExporter {
    fn get_objects(&self, filter: &ObjectFilter) -> Result<Vec<StoredObject>, InternalErrors> {
        let objects = self
            .out_objects
            .lock()
            .map_err(|_| InternalErrors::Io("Failed to acquire MemoryExporter lock".to_string()))?;
        if !filter.image_rel_path.as_deref().is_none_or(str::is_empty) {
            return Ok(Vec::new());
        }
        let mut rows: Vec<StoredObject> = objects
            .iter()
            .filter(|o| filter.t_stack.is_none_or(|t| o.t_stack == t))
            .map(object_row)
            .filter(|row| filter.matches_classes(&row.object_classes))
            .collect();
        rows.sort_by(|a, b| (a.t_stack, &a.id).cmp(&(b.t_stack, &b.id)));
        Ok(rows)
    }

    fn max_track_id_except(&self, class: ObjectClass) -> Result<u64, InternalErrors> {
        let objects = self
            .out_objects
            .lock()
            .map_err(|_| InternalErrors::Io("Failed to acquire MemoryExporter lock".to_string()))?;
        Ok(objects
            .iter()
            .filter(|o| !o.object_class.contains(&class))
            .map(|o| o.track.id.0)
            .max()
            .unwrap_or(0))
    }

    fn set_track_ids(&self, tracks: &[(ObjectId, u64)]) -> Result<(), InternalErrors> {
        let tracks: std::collections::HashMap<&ObjectId, u64> =
            tracks.iter().map(|(id, track)| (id, *track)).collect();
        let mut objects = self
            .out_objects
            .lock()
            .map_err(|_| InternalErrors::Io("Failed to acquire MemoryExporter lock".to_string()))?;
        for object in objects.iter_mut() {
            if let Some(track) = tracks.get(&object.id) {
                object.track.id = TrackId(*track);
            }
        }
        Ok(())
    }
}

/// `object` as a dataset row - the same values the results database stores.
fn object_row(object: &ObjectMetricSettings) -> StoredObject {
    let [x_min, y_min, x_max, y_max] = object.bbox;
    let channels = object
        .intensities
        .keys()
        .filter(|channel| **channel >= 0)
        .max()
        .map_or(0, |max| *max as usize + 1);
    let mut object_classes: Vec<ObjectClass> = object
        .object_class
        .iter()
        .filter(|c| **c != ObjectClass::Unset)
        .copied()
        .collect();
    object_classes.sort();
    StoredObject {
        id: object.id.clone(),
        image_rel_path: String::new(),
        t_stack: object.t_stack,
        z_stack: object.z_stack,
        centroid_px: (
            (x_min as f64 + x_max as f64) / 2.0,
            (y_min as f64 + y_max as f64) / 2.0,
        ),
        // A preview's objects don't record the pixel size.
        pixel_size_nm: None,
        track_id: Some(object.track.id.0).filter(|id| *id != 0),
        bbox_px: object.bbox,
        area_px: object.area as u64,
        object_classes,
        intensity_mean: (0..channels)
            .map(|c| object.intensities.get(&(c as i32)).map(|i| i.avg_intensity))
            .collect(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::object::{Object, ObjectInit};
    use evanalyzer_cfg::core_types::ObjectId;
    use std::collections::HashSet;

    #[test]
    fn export_moves_every_object_into_out_objects_without_dropping_or_duplicating_any() {
        let mut cache = GlobalPipelineCache::default();
        let id_a = ObjectId::next();
        let id_b = ObjectId::next();
        cache.object_cache.insert(
            id_a.clone(),
            Object::new(ObjectInit {
                id: id_a,
                area: 10,
                bbox: [0, 0, 3, 3],
                ..Default::default()
            }),
        );
        cache.object_cache.insert(
            id_b.clone(),
            Object::new(ObjectInit {
                id: id_b,
                area: 20,
                bbox: [1, 1, 4, 4],
                ..Default::default()
            }),
        );

        let exporter = MemoryExporter {
            out_objects: Arc::new(Mutex::new(Vec::new())),
            ..Default::default()
        };
        exporter.export(&cache).unwrap();

        let exported = exporter.out_objects.lock().unwrap();
        assert_eq!(exported.len(), 2);
        let areas: HashSet<usize> = exported.iter().map(|r| r.area).collect();
        assert_eq!(areas, HashSet::from([10, 20]));
    }

    #[test]
    fn export_appends_to_existing_out_objects_rather_than_overwriting() {
        let mut cache = GlobalPipelineCache::default();
        let id = ObjectId::next();
        cache.object_cache.insert(
            id.clone(),
            Object::new(ObjectInit {
                id,
                area: 5,
                bbox: [0, 0, 1, 1],
                ..Default::default()
            }),
        );

        let exporter = MemoryExporter {
            out_objects: Arc::new(Mutex::new(vec![])),
            ..Default::default()
        };
        exporter.export(&cache).unwrap();
        exporter.export(&cache).unwrap(); // second tile's export call

        assert_eq!(
            exporter.out_objects.lock().unwrap().len(),
            2,
            "each export() call must append, not replace, the accumulated results"
        );
    }

    #[test]
    fn dataset_steps_read_a_preview_runs_objects_like_the_results_database() {
        let mut cache = GlobalPipelineCache::default();
        for (id, t, class) in [(2u128, 1, 1u32), (1, 0, 1), (3, 0, 2)] {
            let mut intensities = indexmap::IndexMap::new();
            intensities.insert(
                1,
                crate::object::Intensity {
                    avg_intensity: 0.5,
                    ..Default::default()
                },
            );
            cache.object_cache.insert(
                ObjectId(id),
                Object::new(ObjectInit {
                    id: ObjectId(id),
                    object_class: HashSet::from([ObjectClass::Global(class)]),
                    area: 4,
                    bbox: [10, 20, 14, 30],
                    plane: crate::ImagePlane { z: 0, c: 0, t },
                    intensities,
                    ..Default::default()
                }),
            );
        }
        let exporter = MemoryExporter {
            out_objects: Arc::new(Mutex::new(Vec::new())),
            ..Default::default()
        };
        exporter.export(&cache).unwrap();
        let store = exporter.dataset_store().unwrap();
        let ids = |filter: ObjectFilter| -> Vec<u128> {
            store
                .get_objects(&filter)
                .unwrap()
                .iter()
                .map(|row| row.id.0)
                .collect()
        };

        assert_eq!(ids(ObjectFilter::default()), [1, 3, 2], "by frame, then id");
        let class_1 = ObjectFilter {
            object_class: Some(ObjectClass::Global(1)),
            ..Default::default()
        };
        assert_eq!(ids(class_1), [1, 2]);
        let frame_0 = ObjectFilter {
            t_stack: Some(0),
            ..Default::default()
        };
        assert_eq!(ids(frame_0), [1, 3]);
        let other_image = ObjectFilter {
            image_rel_path: Some("a.tif".into()),
            ..Default::default()
        };
        assert!(ids(other_image).is_empty(), "the preview image has no path");

        let row = &store.get_objects(&ObjectFilter::default()).unwrap()[0];
        assert_eq!(row.centroid_px, (12.0, 25.0));
        assert_eq!(row.intensity_mean, [None, Some(0.5)]);
    }
}
