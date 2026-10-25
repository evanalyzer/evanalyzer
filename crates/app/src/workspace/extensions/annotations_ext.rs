//! # annotations_ext
//!
//! Manual annotations (the hand-drawn objects saved per image) next to the
//! pipeline's preview objects: copying preview objects into the manual
//! annotations - a pipeline's result as the starting point of classifier
//! training data - and deleting manual annotations in bulk.
//!
//! Preview objects stay read-only; a copy is an ordinary manual annotation
//! (editable, used for training). The original stays in the preview.
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-10-19
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.

use crate::workspace::extensions::project_ext::ProjectExt;
use crate::workspace::project_owner::ProjectWithRuntime;
use evanalyzer_cfg::core_types::{ObjectClass, ObjectId, SegmentationClass};
use evanalyzer_cfg::settings::object_settings::{ObjectMetricSettings, TrackSettings};

/// Which manual annotations a bulk operation covers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AnnotationScope {
    /// The open image (its selected series).
    CurrentImage,
    /// Every image and series of the project.
    AllImages,
}

pub trait AnnotationsExt {
    /// Whether `id` is a manual annotation of the open image (not a
    /// preview object): only those can be changed or deleted.
    fn is_manual_object(&self, id: ObjectId) -> bool;

    /// Copies these preview objects of the open image into its manual
    /// annotations. `object_class`: the class the copies get, `None` keeps
    /// the preview objects' classes. An object whose exact shape is already
    /// a manual annotation (copied before) is skipped. Returns the copies'
    /// ids, in the order of `ids`.
    fn copy_preview_objects_to_manual(
        &mut self,
        ids: &[ObjectId],
        object_class: Option<ObjectClass>,
    ) -> Vec<ObjectId>;

    /// The classes of the open image's preview objects with how many have
    /// each (an object with two classes counts for both), most first;
    /// [`ObjectClass::Unset`] for those without a class.
    fn preview_object_classes(&self) -> Vec<(ObjectClass, usize)>;

    /// The classes of the manual annotations in `scope` with how many have
    /// each, as [`Self::preview_object_classes`].
    fn manual_annotation_classes(&self, scope: AnnotationScope) -> Vec<(ObjectClass, usize)>;

    /// The ids of the open image's preview objects with any of `classes`
    /// (all of them for none; [`ObjectClass::Unset`] = those without a
    /// class).
    fn preview_object_ids(&self, classes: &[ObjectClass]) -> Vec<ObjectId>;

    /// How many manual annotations [`Self::delete_manual_annotations`]
    /// would delete, and on how many images.
    fn count_manual_annotations(
        &self,
        scope: AnnotationScope,
        classes: &[ObjectClass],
    ) -> (usize, usize);

    /// Deletes the manual annotations in `scope` with any of `classes` (all
    /// of them for none; [`ObjectClass::Unset`] = those without a class).
    /// Returns how many.
    fn delete_manual_annotations(
        &mut self,
        scope: AnnotationScope,
        classes: &[ObjectClass],
    ) -> usize;
}

/// Whether `object` has one of `classes` (any object for none).
fn has_class(object: &ObjectMetricSettings, classes: &[ObjectClass]) -> bool {
    classes.is_empty()
        || classes.iter().any(|class| {
            if *class == ObjectClass::Unset {
                object.object_class.is_empty()
            } else {
                object.object_class.contains(class)
            }
        })
}

/// The classes of `objects` with how many have each (an object with two
/// classes counts for both), most first; [`ObjectClass::Unset`] for those
/// without a class.
fn class_counts<'a>(
    objects: impl Iterator<Item = &'a ObjectMetricSettings>,
) -> Vec<(ObjectClass, usize)> {
    let mut counts: Vec<(ObjectClass, usize)> = Vec::new();
    let mut add = |class: ObjectClass| match counts.iter_mut().find(|(c, _)| *c == class) {
        Some((_, n)) => *n += 1,
        None => counts.push((class, 1)),
    };
    for object in objects {
        if object.object_class.is_empty() {
            add(ObjectClass::Unset);
        }
        for class in &object.object_class {
            add(class.clone());
        }
    }
    counts.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.to_i32().cmp(&b.0.to_i32())));
    counts
}

/// The same pixels on the same plane.
pub(crate) fn same_shape(a: &ObjectMetricSettings, b: &ObjectMetricSettings) -> bool {
    a.bbox == b.bbox
        && a.z_stack == b.z_stack
        && a.c_stack == b.c_stack
        && a.t_stack == b.t_stack
        && a.mask_data == b.mask_data
}

/// A manual annotation from a preview object: same pixels and measurements,
/// a new id, without the links to other pipeline objects (colocalization,
/// parents, track) - those objects aren't manual annotations.
fn manual_copy(
    object: &ObjectMetricSettings,
    object_class: Option<&ObjectClass>,
) -> ObjectMetricSettings {
    let mut copy = object.clone();
    copy.id = ObjectId::next();
    copy.segmentation_class = SegmentationClass::MANUAL_ANNOTATED;
    if let Some(class) = object_class {
        copy.object_class = [class.clone()].into();
    }
    copy.exclude_from_training = false;
    copy.colocalized_with.clear();
    copy.parents.clear();
    copy.track = TrackSettings::default();
    copy
}

impl AnnotationsExt for ProjectWithRuntime {
    fn is_manual_object(&self, id: ObjectId) -> bool {
        self.get_objects()
            .is_some_and(|objects| objects.iter().any(|o| o.id == id))
    }

    fn copy_preview_objects_to_manual(
        &mut self,
        ids: &[ObjectId],
        object_class: Option<ObjectClass>,
    ) -> Vec<ObjectId> {
        let preview = self.get_preview_objects();
        let picked: Vec<ObjectMetricSettings> = ids
            .iter()
            .filter_map(|id| preview.iter().find(|o| o.id == *id))
            .cloned()
            .collect();
        let Some(series) = self.get_selected_image_series_mut() else {
            return Vec::new();
        };
        let mut copied = Vec::new();
        for object in &picked {
            if series.objects.iter().any(|o| same_shape(o, object)) {
                continue;
            }
            let copy = manual_copy(object, object_class.as_ref());
            copied.push(copy.id.clone());
            series.objects.push(copy);
        }
        copied
    }

    fn preview_object_classes(&self) -> Vec<(ObjectClass, usize)> {
        class_counts(self.get_preview_objects().iter())
    }

    fn manual_annotation_classes(&self, scope: AnnotationScope) -> Vec<(ObjectClass, usize)> {
        match scope {
            AnnotationScope::CurrentImage => {
                class_counts(self.get_objects().unwrap_or_default().iter())
            }
            AnnotationScope::AllImages => class_counts(
                self.settings
                    .images
                    .list
                    .values()
                    .flat_map(|image| image.series.values())
                    .flat_map(|series| series.objects.iter()),
            ),
        }
    }

    fn preview_object_ids(&self, classes: &[ObjectClass]) -> Vec<ObjectId> {
        self.get_preview_objects()
            .iter()
            .filter(|o| has_class(o, classes))
            .map(|o| o.id.clone())
            .collect()
    }

    fn count_manual_annotations(
        &self,
        scope: AnnotationScope,
        classes: &[ObjectClass],
    ) -> (usize, usize) {
        let count = |objects: &[ObjectMetricSettings]| {
            objects.iter().filter(|o| has_class(o, classes)).count()
        };
        match scope {
            AnnotationScope::CurrentImage => {
                let n = self.get_objects().map(count).unwrap_or(0);
                (n, usize::from(n > 0))
            }
            AnnotationScope::AllImages => {
                let mut objects = 0;
                let mut images = 0;
                for image in self.settings.images.list.values() {
                    let n: usize = image.series.values().map(|s| count(&s.objects)).sum();
                    objects += n;
                    images += usize::from(n > 0);
                }
                (objects, images)
            }
        }
    }

    fn delete_manual_annotations(
        &mut self,
        scope: AnnotationScope,
        classes: &[ObjectClass],
    ) -> usize {
        let mut deleted = 0;
        let mut delete = |objects: &mut Vec<ObjectMetricSettings>| {
            let before = objects.len();
            objects.retain(|o| !has_class(o, classes));
            deleted += before - objects.len();
        };
        match scope {
            AnnotationScope::CurrentImage => {
                if let Some(series) = self.get_selected_image_series_mut() {
                    delete(&mut series.objects);
                }
            }
            AnnotationScope::AllImages => {
                for image in self.settings.images.list.values_mut() {
                    for series in image.series.values_mut() {
                        delete(&mut series.objects);
                    }
                }
            }
        }
        deleted
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bitvec::prelude::*;
    use evanalyzer_cfg::settings::images_settings::{ImageEntry, SeriesSettings};
    use std::path::PathBuf;

    const A: ObjectClass = ObjectClass::Global(1);
    const B: ObjectClass = ObjectClass::Global(2);

    fn object(id: u128, classes: &[ObjectClass], x: u32) -> ObjectMetricSettings {
        ObjectMetricSettings {
            id: ObjectId(id),
            object_class: classes.iter().cloned().collect(),
            bbox: [x, 0, x + 1, 1],
            mask_data: bitvec![u64, Lsb0; 1, 0, 1, 1],
            area: 3,
            ..Default::default()
        }
    }

    /// Two images ("a.tif" open, "b.tif"), each one series.
    fn project() -> ProjectWithRuntime {
        let mut project = ProjectWithRuntime::default();
        for name in ["a.tif", "b.tif"] {
            let mut entry = ImageEntry {
                rel_path: PathBuf::from(name),
                ..Default::default()
            };
            entry.series.insert(0, SeriesSettings::default());
            project
                .settings
                .images
                .list
                .insert(PathBuf::from(name), entry);
        }
        project.set_current_image_path(&PathBuf::from("a.tif"));
        project
    }

    #[test]
    fn a_copy_is_an_editable_manual_annotation_without_pipeline_links() {
        let mut project = project();
        let mut pipeline_object = object(7, &[A], 0);
        pipeline_object.segmentation_class = SegmentationClass(3);
        pipeline_object
            .parents
            .insert(ObjectClass::Global(1), ObjectId(1));
        pipeline_object
            .colocalized_with
            .insert(B, vec![ObjectId(3)]);
        pipeline_object.exclude_from_training = true;
        project.tmp_settings.preview_objects.push(pipeline_object);
        assert!(
            !project.is_manual_object(ObjectId(7)),
            "preview objects are read-only"
        );

        let copied = project.copy_preview_objects_to_manual(&[ObjectId(7)], None);
        assert_eq!(copied.len(), 1);
        assert_ne!(copied[0], ObjectId(7), "a new id");
        assert!(project.is_manual_object(copied[0].clone()));

        let copy = &project.get_objects().unwrap()[0];
        assert_eq!(copy.segmentation_class, SegmentationClass::MANUAL_ANNOTATED);
        assert_eq!(copy.object_class, [A].into(), "its classes kept");
        assert_eq!(copy.mask_data, bitvec![u64, Lsb0; 1, 0, 1, 1]);
        assert_eq!(copy.area, 3);
        assert!(copy.parents.is_empty());
        assert!(copy.colocalized_with.is_empty());
        assert!(!copy.exclude_from_training, "a training sample");
        assert_eq!(project.get_preview_objects().len(), 1, "the original stays");
    }

    #[test]
    fn copies_can_get_one_class_instead_of_their_own() {
        let mut project = project();
        project.tmp_settings.preview_objects = vec![object(1, &[A, B], 0), object(2, &[], 5)];
        let copied = project.copy_preview_objects_to_manual(&[ObjectId(1), ObjectId(2)], Some(B));
        assert_eq!(copied.len(), 2);
        for copy in project.get_objects().unwrap() {
            assert_eq!(copy.object_class, [B].into());
        }
    }

    #[test]
    fn an_object_copied_before_is_not_copied_again() {
        let mut project = project();
        project.tmp_settings.preview_objects = vec![object(1, &[A], 0), object(2, &[A], 5)];
        assert_eq!(
            project
                .copy_preview_objects_to_manual(&[ObjectId(1)], None)
                .len(),
            1
        );
        // A new preview run gives new ids to the same shapes.
        project.tmp_settings.preview_objects = vec![object(11, &[A], 0), object(12, &[A], 5)];
        let copied = project.copy_preview_objects_to_manual(&[ObjectId(11), ObjectId(12)], None);
        assert_eq!(copied.len(), 1, "only the one not copied yet");
        assert_eq!(project.get_objects().unwrap().len(), 2);
        // Unknown ids (not preview objects) copy nothing.
        assert!(
            project
                .copy_preview_objects_to_manual(&[ObjectId(99)], None)
                .is_empty()
        );
    }

    #[test]
    fn preview_classes_and_ids_by_class() {
        let mut project = project();
        project.tmp_settings.preview_objects =
            vec![object(1, &[A], 0), object(2, &[A, B], 2), object(3, &[], 4)];
        assert_eq!(
            project.preview_object_classes(),
            vec![(A, 2), (ObjectClass::Unset, 1), (B, 1)]
        );
        assert_eq!(project.preview_object_ids(&[]).len(), 3);
        assert_eq!(project.preview_object_ids(&[B]), vec![ObjectId(2)]);
        assert_eq!(
            project.preview_object_ids(&[ObjectClass::Unset]),
            vec![ObjectId(3)]
        );
    }

    #[test]
    fn manual_annotations_are_deleted_by_class_on_this_image_or_all() {
        let mut project = project();
        project.add_object(&object(1, &[A], 0));
        project.add_object(&object(2, &[B], 2));
        project.add_object(&object(3, &[A, B], 4));
        project.set_current_image_path(&PathBuf::from("b.tif"));
        project.add_object(&object(4, &[A], 0));
        project.add_object(&object(5, &[], 2));
        project.set_current_image_path(&PathBuf::from("a.tif"));

        assert_eq!(
            project.count_manual_annotations(AnnotationScope::CurrentImage, &[A]),
            (2, 1)
        );
        assert_eq!(
            project.count_manual_annotations(AnnotationScope::AllImages, &[A]),
            (3, 2)
        );
        assert_eq!(
            project.count_manual_annotations(AnnotationScope::AllImages, &[]),
            (5, 2)
        );

        assert_eq!(
            project.manual_annotation_classes(AnnotationScope::CurrentImage),
            vec![(A, 2), (B, 2)]
        );
        assert_eq!(
            project.manual_annotation_classes(AnnotationScope::AllImages),
            vec![(A, 3), (B, 2), (ObjectClass::Unset, 1)]
        );

        // Class A on this image: objects 1 and 3 (it has A too).
        assert_eq!(
            project.delete_manual_annotations(AnnotationScope::CurrentImage, &[A]),
            2
        );
        let ids = |p: &ProjectWithRuntime| {
            p.get_objects()
                .unwrap()
                .iter()
                .map(|o| o.id.clone())
                .collect::<Vec<_>>()
        };
        assert_eq!(ids(&project), vec![ObjectId(2)]);

        // Unclassified ones on every image.
        assert_eq!(
            project.delete_manual_annotations(AnnotationScope::AllImages, &[ObjectClass::Unset]),
            1
        );
        // Then everything.
        assert_eq!(
            project.delete_manual_annotations(AnnotationScope::AllImages, &[]),
            2
        );
        assert_eq!(
            project.count_manual_annotations(AnnotationScope::AllImages, &[]),
            (0, 0)
        );
    }
}
