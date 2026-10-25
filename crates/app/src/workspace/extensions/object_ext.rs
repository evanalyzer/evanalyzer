//! # object_ext
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-06-05
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.

use bitvec::{order::Lsb0, slice::BitSlice};
use evanalyzer_cfg::{
    core_types::{ImageDimensionality, ObjectClass},
    settings::object_settings::ObjectMetricSettings,
};

pub trait ObjectExt {
    fn is_part_of(&self, x: u32, y: u32) -> bool;
    /// The object's mask on plane `z` (`None`: no one plane shown - a
    /// z-projection): a plane's object's whole mask (shown on every plane);
    /// a volume's object's part on `z` - `None` where it isn't on `z`.
    fn mask_on_plane(&self, z: Option<i32>) -> Option<&BitSlice<u64, Lsb0>>;
    /// [`Self::is_part_of`] on plane `z` (see [`Self::mask_on_plane`]).
    fn is_part_of_on_plane(&self, x: u32, y: u32, z: Option<i32>) -> bool;
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

    fn mask_on_plane(&self, z: Option<i32>) -> Option<&BitSlice<u64, Lsb0>> {
        match self.dimensionality {
            ImageDimensionality::Planar => Some(&self.mask_data),
            ImageDimensionality::Volume { depth } => {
                let plane = z? - self.z_stack;
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

    fn is_part_of_on_plane(&self, x: u32, y: u32, z: Option<i32>) -> bool {
        let Some(mask) = self.mask_on_plane(z) else {
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

#[cfg(test)]
mod tests {
    use super::*;
    use bitvec::{order::Lsb0, vec::BitVec};

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
                .mask_on_plane(z)
                .map(|m| m.iter().by_vals().collect::<Vec<_>>())
        };
        assert_eq!(on(Some(4)), Some(vec![true, false]));
        assert_eq!(on(Some(5)), Some(vec![false, false]));
        assert_eq!(on(Some(6)), Some(vec![false, true]));
        assert_eq!(on(Some(3)), None, "before its first plane");
        assert_eq!(on(Some(7)), None, "after its last plane");
        assert_eq!(on(None), None, "a projection shows no one plane");
        assert!(object.is_part_of_on_plane(10, 20, Some(4)));
        assert!(!object.is_part_of_on_plane(10, 20, Some(6)));
        assert!(object.is_part_of_on_plane(11, 20, Some(6)));
    }

    #[test]
    fn a_planes_object_is_shown_on_every_plane() {
        let object = single_pixel_object();
        for z in [None, Some(0), Some(9)] {
            assert_eq!(object.mask_on_plane(z).map(|m| m.len()), Some(12));
            assert!(object.is_part_of_on_plane(11, 21, z));
        }
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
