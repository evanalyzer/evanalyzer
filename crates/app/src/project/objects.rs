//! # object_ext
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-06-05
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.

use crate::project::ProjectWithRuntime;
use bitvec::vec::BitVec;
use evanalyzer_cfg::core_types::{ObjectId, SegmentationClass};
use evanalyzer_cfg::object_class_set_from_u32;
use evanalyzer_core::ObjectMetricSettingsExt;

use bitvec::{order::Lsb0, slice::BitSlice};
use evanalyzer_cfg::{
    core_types::{ImageDimensionality, ObjectClass},
    settings::object_settings::ObjectMetricSettings,
};

/// What the viewer shows: plane `z` of frame `t`, or (`z: None`) the
/// z-projection of frame `t`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ShownPlane {
    pub z: Option<i32>,
    pub t: i32,
}

pub trait ObjectExt {
    fn is_part_of(&self, x: u32, y: u32) -> bool;
    /// The object's mask where `shown`, `None` where it isn't shown - the
    /// same objects an analysis of that plane uses (see
    /// [`ObjectMetricSettings::z_projection`]), always of frame `t`:
    /// - a plane's object: its whole mask on its own plane and in a
    ///   z-projection;
    /// - an object drawn on a z-projection: its mask in a z-projection only;
    /// - a volume's object: its part on plane `z`, not in a z-projection.
    fn mask_on_plane(&self, shown: ShownPlane) -> Option<&BitSlice<u64, Lsb0>>;
    /// [`Self::is_part_of`] where `shown` (see [`Self::mask_on_plane`]).
    fn is_part_of_on_plane(&self, x: u32, y: u32, shown: ShownPlane) -> bool;
    fn add_object_class(&mut self, object_class: ObjectClass);
    fn remove_object_class(&mut self, object_class: &ObjectClass);
    fn remove_all_object_classes(&mut self);
}

impl ObjectExt for ObjectMetricSettings {
    /// Checks if a global coordinate (x, y) is within the object's mask.
    fn is_part_of(&self, x: u32, y: u32) -> bool {
        let [x_min, y_min, x_max, y_max] = self.bbox;

        // bbox[2]/[3] are INCLUSIVE - the last pixel coordinate inside the object.
        if x < x_min || x > x_max || y < y_min || y > y_max {
            return false;
        }

        let local_x = (x - x_min) as usize;
        let local_y = (y - y_min) as usize;
        let width = (x_max - x_min + 1) as usize;

        // Calculate index in the BitVec (Row-major order assumed)
        let bit_index = (local_y * width) + local_x;

        // Access the mask bit
        self.mask_data.get(bit_index).map(|b| *b).unwrap_or(false)
    }

    fn mask_on_plane(&self, shown: ShownPlane) -> Option<&BitSlice<u64, Lsb0>> {
        match self.dimensionality {
            ImageDimensionality::Planar => self
                .belongs_to(shown.z, shown.t)
                .then_some(&self.mask_data[..]),
            ImageDimensionality::Volume { depth } => {
                if self.t_stack != shown.t {
                    return None;
                }
                let plane = shown.z? - self.z_stack;
                if plane < 0 || plane >= depth as i32 {
                    return None;
                }
                let [x_min, y_min, x_max, y_max] = self.bbox;
                let len = (x_max - x_min + 1) as usize * (y_max - y_min + 1) as usize;
                let start = plane as usize * len;
                self.mask_data.get(start..start + len)
            }
        }
    }

    fn is_part_of_on_plane(&self, x: u32, y: u32, shown: ShownPlane) -> bool {
        let Some(mask) = self.mask_on_plane(shown) else {
            return false;
        };
        let [x_min, y_min, x_max, y_max] = self.bbox;
        if x < x_min || x > x_max || y < y_min || y > y_max {
            return false;
        }
        let width = (x_max - x_min + 1) as usize;
        let bit_index = (y - y_min) as usize * width + (x - x_min) as usize;
        mask.get(bit_index).is_some_and(|b| *b)
    }

    fn add_object_class(&mut self, object_class: ObjectClass) {
        if object_class != ObjectClass::Unset {
            self.object_class.insert(object_class);
        }
    }

    fn remove_object_class(&mut self, object_class: &ObjectClass) {
        self.object_class.remove(object_class);
    }

    fn remove_all_object_classes(&mut self) {
        self.object_class.clear();
    }
}

impl ProjectWithRuntime {
    /// Adds an object class to a specific object by its ID.
    ///
    /// This method finds the object with the given ID in the currently selected image series
    /// and adds the specified object class to it.
    ///
    /// # Arguments
    /// * `id` - The ObjectId of the object to modify
    /// * `object_class` - The ObjectClass to add to the object
    pub fn add_class_to_object(&mut self, id: ObjectId, object_class: ObjectClass) {
        if let Some(series) = self.get_selected_image_series_mut() {
            if let Some(object) = series.objects.iter_mut().find(|object| object.id == id) {
                object.add_object_class(object_class);
            }
        }
    }

    /// Removes all classes from the object and adds this single one
    ///
    /// This method finds the object with the given ID in the currently selected image series
    /// and adds the specified object class to it.
    ///
    /// # Arguments
    /// * `id` - The ObjectId of the object to modify
    /// * `object_class` - The ObjectClass to add to the object
    pub fn replace_classes_of_object(&mut self, id: ObjectId, object_class: ObjectClass) {
        if let Some(series) = self.get_selected_image_series_mut() {
            if let Some(object) = series.objects.iter_mut().find(|object| object.id == id) {
                object.remove_all_object_classes();
                object.add_object_class(object_class);
            }
        }
    }

    /// Removes an object class from a specific object by its ID.
    ///
    /// This method finds the object with the given ID in the currently selected image series
    /// and removes the specified object class from it.
    ///
    /// # Arguments
    /// * `id` - The ObjectId of the object to modify
    /// * `object_class` - The ObjectClass to remove from the object
    pub fn remove_class_from_object(&mut self, id: ObjectId, object_class: &ObjectClass) {
        if let Some(series) = self.get_selected_image_series_mut() {
            if let Some(object) = series.objects.iter_mut().find(|object| object.id == id) {
                object.remove_object_class(object_class);
            }
        }
    }

    pub fn add_object(&mut self, object: &ObjectMetricSettings) {
        if let Some(series) = self.get_selected_image_series_mut() {
            series.objects.push(object.clone());
        }
    }

    pub fn get_objects(&self) -> Option<&[ObjectMetricSettings]> {
        self.get_selected_image_series()
            .map(|series| series.objects.as_slice())
    }

    pub fn delete_object(&mut self, id: ObjectId) {
        if let Some(series) = self.get_selected_image_series_mut() {
            series.objects.retain(|object| object.id != id);
        }
    }

    pub fn get_reference_object(&self) -> Option<Vec<ObjectMetricSettings>> {
        let mut objects = Vec::new();

        // Helper to generate a circular object
        let create_circle_object = |id: u128, x_start: usize, y_start: usize| {
            let width = 5;
            let height = 5;
            let radius = 2.0f32;
            let center = 2i32;

            // BitVec stores bits efficiently. Lsb0 maps index 0 to the first bit.
            let mut mask_data = BitVec::<u64, Lsb0>::repeat(false, width * height);

            for y in 0..height {
                for x in 0..width {
                    if (x as i32 - center).pow(2) + (y as i32 - center).pow(2)
                        <= radius.powi(2) as i32
                    {
                        // Direct indexing! No manual bit arithmetic needed.
                        mask_data.set(y * width + x, true);
                    }
                }
            }
            ObjectMetricSettings {
                id: ObjectId(id),
                bbox: [
                    x_start as u32,
                    y_start as u32,
                    (x_start + width) as u32,
                    (y_start + height) as u32,
                ],
                mask_data: mask_data,
                segmentation_class: SegmentationClass(1),
                object_class: object_class_set_from_u32!(1),
                area: 31415,
                ..Default::default()
            }
        };

        let num_elements = 30000;
        let elements_per_row = 150;
        let spacing = 2048 / elements_per_row;
        for i in 0..num_elements {
            // Calculate which row and which column the current index belongs to
            let row = i / elements_per_row;
            let col = i % elements_per_row;

            // Distribute evenly by multiplying index by the calculated spacing
            let x = ((col * spacing) + (spacing / 2)) as usize;
            let y = ((row * spacing) + (spacing / 2)) as usize;

            objects.push(create_circle_object(i + 1, x, y));
        }
        Some(objects)
    }

    pub fn count_objects_for_class(&self, class_id: &ObjectClass) -> usize {
        let manual = self
            .get_objects()
            .unwrap_or_default()
            .iter()
            .filter(|r| r.object_class.contains(class_id))
            .count();
        let preview = self
            .tmp_settings
            .preview_objects
            .iter()
            .filter(|r| r.object_class.contains(class_id))
            .count();
        manual + preview
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bitvec::{order::Lsb0, vec::BitVec};
    use evanalyzer_cfg::core_types::ZProjection;

    /// A 4x3 object (bbox x_min=10,y_min=20,x_max=13,y_max=22, so width=4,
    /// height=3) whose mask is true only at local (1,1) - global (11, 21).
    fn single_pixel_object() -> ObjectMetricSettings {
        let width = 4usize;
        let height = 3usize;
        let mut mask_data = BitVec::<u64, Lsb0>::repeat(false, width * height);
        mask_data.set(1 * width + 1, true);
        ObjectMetricSettings {
            bbox: [10, 20, 13, 22],
            mask_data,
            ..Default::default()
        }
    }

    /// A 2 × 1 volume's object over z 4..=6: (10, 20) on z 4, (11, 20) on
    /// z 6, nothing on z 5.
    fn volume_object() -> ObjectMetricSettings {
        let mask_data: BitVec<u64, Lsb0> = [true, false, false, false, false, true]
            .into_iter()
            .collect();
        ObjectMetricSettings {
            bbox: [10, 20, 11, 20],
            mask_data,
            z_stack: 4,
            dimensionality: ImageDimensionality::Volume { depth: 3 },
            ..Default::default()
        }
    }

    #[test]
    fn a_volumes_object_is_cut_at_the_plane_shown() {
        let object = volume_object();
        let on = |z| {
            object
                .mask_on_plane(ShownPlane { z, t: 0 })
                .map(|m| m.iter().by_vals().collect::<Vec<_>>())
        };
        assert_eq!(on(Some(4)), Some(vec![true, false]));
        assert_eq!(on(Some(5)), Some(vec![false, false]));
        assert_eq!(on(Some(6)), Some(vec![false, true]));
        assert_eq!(on(Some(3)), None, "before its first plane");
        assert_eq!(on(Some(7)), None, "after its last plane");
        assert_eq!(on(None), None, "a projection shows no one plane");
        let plane = |z| ShownPlane { z: Some(z), t: 0 };
        assert!(object.is_part_of_on_plane(10, 20, plane(4)));
        assert!(!object.is_part_of_on_plane(10, 20, plane(6)));
        assert!(object.is_part_of_on_plane(11, 20, plane(6)));
    }

    #[test]
    fn a_planes_object_is_shown_on_its_plane_and_in_projections_of_its_frame() {
        let object = ObjectMetricSettings {
            z_stack: 3,
            t_stack: 2,
            ..single_pixel_object()
        };
        let shown = |z, t| object.mask_on_plane(ShownPlane { z, t }).map(|m| m.len());

        assert_eq!(shown(Some(3), 2), Some(12), "its plane");
        assert_eq!(shown(None, 2), Some(12), "a z-projection");
        assert_eq!(shown(Some(4), 2), None, "another plane");
        assert_eq!(shown(Some(3), 1), None, "another frame");
        assert_eq!(shown(None, 0), None, "a projection of another frame");
        assert!(object.is_part_of_on_plane(11, 21, ShownPlane { z: Some(3), t: 2 }));
        assert!(!object.is_part_of_on_plane(11, 21, ShownPlane { z: Some(0), t: 2 }));
    }

    #[test]
    fn an_object_drawn_on_a_projection_is_shown_in_projections_only() {
        let object = ObjectMetricSettings {
            z_projection: ZProjection::AvgIntensity,
            t_stack: 1,
            ..single_pixel_object()
        };
        let shown = |z, t| object.mask_on_plane(ShownPlane { z, t }).is_some();

        assert!(shown(None, 1));
        assert!(!shown(Some(0), 1), "not on plane 0, its unused z");
        assert!(!shown(None, 0), "another frame");
    }

    #[test]
    fn is_part_of_true_only_at_the_set_mask_bit() {
        let object = single_pixel_object();
        assert!(object.is_part_of(11, 21));
        assert!(!object.is_part_of(10, 20));
        assert!(!object.is_part_of(12, 21));
        assert!(!object.is_part_of(11, 22));
    }

    #[test]
    fn is_part_of_is_false_outside_the_bbox() {
        let object = single_pixel_object();
        // Below/left of the bbox.
        assert!(!object.is_part_of(9, 21));
        assert!(!object.is_part_of(11, 19));
        // Above/right of the bbox (bbox max is inclusive, so max+1 is out).
        assert!(!object.is_part_of(14, 21));
        assert!(!object.is_part_of(11, 23));
    }

    #[test]
    fn is_part_of_handles_a_bbox_at_the_coordinate_origin() {
        // x - x_min / y - y_min must not underflow when x_min/y_min are 0.
        let width = 2usize;
        let mut mask_data = BitVec::<u64, Lsb0>::repeat(false, width * width);
        mask_data.set(0, true);
        let object = ObjectMetricSettings {
            bbox: [0, 0, 1, 1],
            mask_data,
            ..Default::default()
        };
        assert!(object.is_part_of(0, 0));
        assert!(!object.is_part_of(1, 1));
    }

    #[test]
    fn add_object_class_inserts_a_valid_class_but_rejects_unset() {
        let mut object = ObjectMetricSettings::default();
        object.add_object_class(ObjectClass::Global(1));
        assert!(object.object_class.contains(&ObjectClass::Global(1)));

        object.add_object_class(ObjectClass::Unset);
        assert!(
            !object.object_class.contains(&ObjectClass::Unset),
            "Unset must never be recorded as a real class"
        );
        assert_eq!(object.object_class.len(), 1);
    }

    #[test]
    fn add_object_class_is_idempotent() {
        let mut object = ObjectMetricSettings::default();
        object.add_object_class(ObjectClass::Global(5));
        object.add_object_class(ObjectClass::Global(5));
        assert_eq!(object.object_class.len(), 1);
    }

    #[test]
    fn remove_object_class_only_removes_the_given_class() {
        let mut object = ObjectMetricSettings::default();
        object.add_object_class(ObjectClass::Global(1));
        object.add_object_class(ObjectClass::Global(2));

        object.remove_object_class(&ObjectClass::Global(1));

        assert!(!object.object_class.contains(&ObjectClass::Global(1)));
        assert!(object.object_class.contains(&ObjectClass::Global(2)));
    }

    #[test]
    fn remove_object_class_on_a_missing_class_is_a_no_op() {
        let mut object = ObjectMetricSettings::default();
        object.add_object_class(ObjectClass::Global(2));
        object.remove_object_class(&ObjectClass::Global(99));
        assert_eq!(object.object_class.len(), 1);
    }
}
