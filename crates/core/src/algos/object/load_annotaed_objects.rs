//! # Load Annotated Objects Module
//!
//! Brings the objects a user annotated by hand on an image (stored in the
//! project's image settings) into the pipeline, so they are measured,
//! classified, colocalized and exported exactly like segmented objects.
//!
//! ## Overview
//! Runs once per image and plane (`ExecutionScope::WholeImage`): an annotation
//! crossing a tile border stays one object, and its intensities are measured on
//! whichever image tiles it covers. Only annotations of the plane being
//! processed are loaded (the executor fills `GlobalPipelineCache::annotated_objects`
//! accordingly).
use crate::{
    algos::{ExecutionScope, ImageAlgorithm},
    object::Object,
};
use evanalyzer_cfg::core_types::{
    CitationMetadata, InternalErrors, ObjectClass, ObjectId, SegmentationClass,
};
use macros::CommandsMeta;

/// Loads the hand-annotated objects of the image into the pipeline.
///
/// Every loaded object gets a new object id, is marked as manually annotated
/// and has its intensities measured on every channel - from there on it is
/// handled like any segmented object.
#[derive(CommandsMeta)]
#[cmdsmeta(category = "object")]
pub struct LoadAnnotatedObjects {
    /// Only load annotations carrying at least one of these classes.
    ///
    /// Leave empty to load every annotated object of the image.
    pub input_classes: Vec<ObjectClass>,

    /// Class added to every loaded object, so later steps can select them.
    /// Set to `Unset` to add none.
    pub output_class: ObjectClass,

    /// Keep the classes the objects were given while annotating.
    ///
    /// Turn off to start from `output_class` alone.
    #[cmdsmeta(default = false, visibility = Advanced)]
    pub keep_annotated_classes: bool,
}

impl ImageAlgorithm for LoadAnnotatedObjects {
    fn execute(
        &self,
        _ctx: &mut crate::pipeline::pipeline_context::PipelineContext,
        cache: &mut crate::GlobalPipelineCache,
    ) -> Result<(), InternalErrors> {
        let annotations = std::sync::Arc::clone(&cache.annotated_objects);
        let mut loaded = Vec::new();
        for annotation in annotations.iter() {
            let wanted = self.input_classes.is_empty()
                || annotation
                    .object_class
                    .iter()
                    .any(|class| self.input_classes.contains(class));
            if !wanted {
                continue;
            }

            let mut object = Object::from_object_settings(annotation.clone());
            // A fresh id: the stored one may collide with ids the pipeline
            // hands out. Links to other annotations (coloc partners, parents,
            // children, tracks) refer to those stored ids, so they're dropped.
            object.id = ObjectId::next();
            object.segmentation_class = SegmentationClass::MANUAL_ANNOTATED;
            object.colocalized_with.clear();
            object.parent_id = None;
            object.children.clear();
            object.track = Default::default();
            if !self.keep_annotated_classes {
                object.object_class.clear();
            }
            if matches!(self.output_class, ObjectClass::Valid(_)) {
                object.add_object_class(self.output_class);
            }
            // The stored values may be stale (or were never measured for an
            // object drawn by hand): measure on the image being analyzed.
            object.intensities = object.measure_intensities(cache);
            loaded.push(object);
        }

        for object in loaded {
            cache.object_cache.insert(object.id.clone(), object);
        }
        Ok(())
    }

    fn name(&self) -> &'static str {
        "Load Annotated Objects"
    }

    fn cite(&self) -> Vec<&'static CitationMetadata> {
        vec![&CitationMetadata::DANMAYR]
    }

    fn execution_scope(&self) -> ExecutionScope {
        ExecutionScope::WholeImage
    }

    fn uses_annotated_objects(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        GlobalPipelineCache, ImageContainer, ImagePlane, ImageTile, ManagedImage,
        image::{PixelSizes, Point2d},
        object::ObjectInit,
        pipeline::{pipeline::PipelineImageMeta, pipeline_context::PipelineContext},
    };
    use bitvec::prelude::*;
    use evanalyzer_cfg::settings::object_settings::ObjectMetricSettings;
    use kornia_image::{Image, ImageSize};
    use std::{path::PathBuf, sync::Arc};

    const SIZE: ImageSize = ImageSize {
        width: 10,
        height: 10,
    };
    const CHANNEL: i32 = 0;
    const VALUE: f64 = 7.0;
    const NUCLEUS: ObjectClass = ObjectClass::Valid(1);
    const CELL: ObjectClass = ObjectClass::Valid(2);
    const LOADED: ObjectClass = ObjectClass::Valid(50);
    // Far above the ObjectId::next() counter so a reused id would be visible.
    const STORED_ID: u128 = 900_000;

    fn image(value: f32) -> ImageContainer {
        ImageContainer::F32Gray(ManagedImage {
            data: Image::<f32, 1>::new(SIZE, vec![value; SIZE.width * SIZE.height]).unwrap(),
            tile_offset: Point2d { x: 0, y: 0 },
            plane: None,
        })
    }

    fn tile() -> ImageTile {
        ImageTile {
            offset_x: 0,
            offset_y: 0,
            width: SIZE.width,
            height: SIZE.height,
        }
    }

    fn make_ctx() -> PipelineContext {
        PipelineContext::new_from_image(
            PathBuf::default(),
            PipelineImageMeta {
                image_tile_info: tile(),
                full_image_width: SIZE,
                is_rgb: false,
                nr_of_bits: 8,
                pixel_sizes: PixelSizes {
                    px_size_x: 1.0,
                    px_size_y: 1.0,
                    px_size_z: 1.0,
                },
            },
            image(0.0).into(),
        )
        .unwrap()
    }

    /// A filled annotation over the inclusive `bbox`, with stale links and
    /// intensities like a stored annotation may carry.
    fn annotation(id: u128, bbox: [u32; 4], classes: &[ObjectClass]) -> ObjectMetricSettings {
        let [x_min, y_min, x_max, y_max] = bbox;
        let area = ((x_max - x_min + 1) * (y_max - y_min + 1)) as usize;
        let mut object = Object::new(ObjectInit {
            id: ObjectId(id),
            bbox,
            mask_data: BitVec::<u64, Lsb0>::repeat(true, area),
            area,
            plane: ImagePlane::default(),
            ..Default::default()
        });
        for class in classes {
            object.add_object_class(*class);
        }
        object.parent_id = Some(ObjectId(id + 1));
        object.children.push(ObjectId(id + 2));
        object.colocalized_with.insert(CELL, vec![ObjectId(id + 3)]);
        object.to_object_settings()
    }

    fn cache_with(annotations: Vec<ObjectMetricSettings>) -> GlobalPipelineCache {
        let mut cache = GlobalPipelineCache::default();
        cache.add_to_channel_cache(Arc::new(image(VALUE as f32)), CHANNEL, tile());
        cache.annotated_objects = Arc::new(annotations);
        cache
    }

    fn run(cmd: &LoadAnnotatedObjects, cache: &mut GlobalPipelineCache) -> Vec<Object> {
        cmd.execute(&mut make_ctx(), cache).unwrap();
        cache.object_cache.values().cloned().collect()
    }

    fn load_all() -> LoadAnnotatedObjects {
        LoadAnnotatedObjects {
            input_classes: vec![],
            output_class: ObjectClass::Unset,
            keep_annotated_classes: true,
        }
    }

    #[test]
    fn empty_filter_loads_every_annotation() {
        let mut cache = cache_with(vec![
            annotation(STORED_ID, [0, 0, 2, 2], &[NUCLEUS]),
            annotation(STORED_ID + 10, [5, 5, 6, 6], &[CELL]),
            annotation(STORED_ID + 20, [8, 8, 8, 8], &[]),
        ]);
        let loaded = run(&load_all(), &mut cache);
        assert_eq!(loaded.len(), 3);
        // The annotations themselves stay untouched for the next plane / run.
        assert_eq!(cache.annotated_objects.len(), 3);
    }

    #[test]
    fn input_classes_select_annotations_carrying_any_of_them() {
        let mut cache = cache_with(vec![
            annotation(STORED_ID, [0, 0, 2, 2], &[NUCLEUS]),
            annotation(STORED_ID + 10, [5, 5, 6, 6], &[CELL]),
            annotation(STORED_ID + 20, [8, 8, 8, 8], &[NUCLEUS, CELL]),
            annotation(STORED_ID + 30, [3, 3, 3, 3], &[]),
        ]);
        let cmd = LoadAnnotatedObjects {
            input_classes: vec![NUCLEUS],
            ..load_all()
        };
        let mut areas: Vec<usize> = run(&cmd, &mut cache).iter().map(|o| o.area).collect();
        areas.sort();
        assert_eq!(areas, vec![1, 9]);
    }

    #[test]
    fn loaded_objects_get_fresh_ids_and_are_marked_manual() {
        let mut cache = cache_with(vec![annotation(STORED_ID, [0, 0, 2, 2], &[NUCLEUS])]);
        let loaded = run(&load_all(), &mut cache);
        let object = &loaded[0];
        assert_ne!(object.id, ObjectId(STORED_ID));
        assert!(cache.object_cache.get(&object.id).is_some());
        assert_eq!(
            object.segmentation_class,
            SegmentationClass::MANUAL_ANNOTATED
        );
        assert_eq!(object.bbox, [0, 0, 2, 2]);
        assert_eq!(object.area, 9);
    }

    #[test]
    fn links_to_other_annotations_are_dropped() {
        let mut cache = cache_with(vec![annotation(STORED_ID, [0, 0, 2, 2], &[NUCLEUS])]);
        let object = &run(&load_all(), &mut cache)[0];
        assert!(object.colocalized_with.is_empty());
        assert!(object.parent_id.is_none());
        assert!(object.children.is_empty());
    }

    #[test]
    fn output_class_is_added_to_the_annotated_classes() {
        let mut cache = cache_with(vec![annotation(STORED_ID, [0, 0, 2, 2], &[NUCLEUS])]);
        let cmd = LoadAnnotatedObjects {
            output_class: LOADED,
            ..load_all()
        };
        let object = &run(&cmd, &mut cache)[0];
        assert!(object.has_object_class(&NUCLEUS));
        assert!(object.has_object_class(&LOADED));
    }

    #[test]
    fn annotated_classes_can_be_dropped() {
        let mut cache = cache_with(vec![annotation(STORED_ID, [0, 0, 2, 2], &[NUCLEUS, CELL])]);
        let cmd = LoadAnnotatedObjects {
            output_class: LOADED,
            keep_annotated_classes: false,
            ..load_all()
        };
        let object = &run(&cmd, &mut cache)[0];
        assert_eq!(object.object_class.len(), 1);
        assert!(object.has_object_class(&LOADED));

        // Without an output class the object ends up unclassified.
        let mut cache = cache_with(vec![annotation(STORED_ID, [0, 0, 2, 2], &[NUCLEUS])]);
        let cmd = LoadAnnotatedObjects {
            keep_annotated_classes: false,
            ..load_all()
        };
        assert!(run(&cmd, &mut cache)[0].object_class.is_empty());
    }

    #[test]
    fn intensities_are_measured_on_the_analyzed_image() {
        let mut stored = annotation(STORED_ID, [1, 1, 3, 3], &[NUCLEUS]);
        stored.intensities.clear();
        let mut cache = cache_with(vec![stored]);
        let object = &run(&load_all(), &mut cache)[0];
        let intensity = object
            .intensities
            .get(&CHANNEL)
            .expect("loaded object should be measured on the cached channel");
        assert_eq!(intensity.sum_intensity, 9.0 * VALUE as f64);
        assert_eq!(intensity.avg_intensity, VALUE);
        assert_eq!(intensity.min_intensity, VALUE);
        assert_eq!(intensity.max_intensity, VALUE);
    }

    #[test]
    fn no_annotations_leaves_the_cache_unchanged() {
        let mut cache = cache_with(vec![]);
        assert!(run(&load_all(), &mut cache).is_empty());
    }

    #[test]
    fn command_metadata() {
        let cmd = load_all();
        assert_eq!(cmd.name(), "Load Annotated Objects");
        assert_eq!(cmd.cite()[0].cite_key, "danmayr2026");
        assert!(matches!(cmd.execution_scope(), ExecutionScope::WholeImage));
        assert!(cmd.uses_annotated_objects());
    }
}
