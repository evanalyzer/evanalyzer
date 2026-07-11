use evanalyzer_cfg::{core_types::ObjectClass, settings::roi_settings::RoiSettings};

pub trait RoiExt {
    fn is_part_of(&self, x: u32, y: u32) -> bool;
    fn add_object_class(&mut self, object_class: ObjectClass);
    fn remove_object_class(&mut self, object_class: &ObjectClass);
}

impl RoiExt for RoiSettings {
    /// Checks if a global coordinate (x, y) is within the ROI's mask.
    fn is_part_of(&self, x: u32, y: u32) -> bool {
        let [x_min, y_min, x_max, y_max] = self.bbox;

        // bbox[2]/[3] are INCLUSIVE - the last pixel coordinate inside the ROI.
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

    fn add_object_class(&mut self, object_class: ObjectClass) {
        if object_class != ObjectClass::Unset {
            self.object_class.insert(object_class);
        }
    }

    fn remove_object_class(&mut self, object_class: &ObjectClass) {
        self.object_class.remove(object_class);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bitvec::{order::Lsb0, vec::BitVec};

    /// A 4x3 ROI (bbox x_min=10,y_min=20,x_max=13,y_max=22, so width=4,
    /// height=3) whose mask is true only at local (1,1) - global (11, 21).
    fn single_pixel_roi() -> RoiSettings {
        let width = 4usize;
        let height = 3usize;
        let mut mask_data = BitVec::<u64, Lsb0>::repeat(false, width * height);
        mask_data.set(1 * width + 1, true);
        RoiSettings {
            bbox: [10, 20, 13, 22],
            mask_data,
            ..Default::default()
        }
    }

    #[test]
    fn is_part_of_true_only_at_the_set_mask_bit() {
        let roi = single_pixel_roi();
        assert!(roi.is_part_of(11, 21));
        assert!(!roi.is_part_of(10, 20));
        assert!(!roi.is_part_of(12, 21));
        assert!(!roi.is_part_of(11, 22));
    }

    #[test]
    fn is_part_of_is_false_outside_the_bbox() {
        let roi = single_pixel_roi();
        // Below/left of the bbox.
        assert!(!roi.is_part_of(9, 21));
        assert!(!roi.is_part_of(11, 19));
        // Above/right of the bbox (bbox max is inclusive, so max+1 is out).
        assert!(!roi.is_part_of(14, 21));
        assert!(!roi.is_part_of(11, 23));
    }

    #[test]
    fn is_part_of_handles_a_bbox_at_the_coordinate_origin() {
        // x - x_min / y - y_min must not underflow when x_min/y_min are 0.
        let width = 2usize;
        let mut mask_data = BitVec::<u64, Lsb0>::repeat(false, width * width);
        mask_data.set(0, true);
        let roi = RoiSettings {
            bbox: [0, 0, 1, 1],
            mask_data,
            ..Default::default()
        };
        assert!(roi.is_part_of(0, 0));
        assert!(!roi.is_part_of(1, 1));
    }

    #[test]
    fn add_object_class_inserts_a_valid_class_but_rejects_unset() {
        let mut roi = RoiSettings::default();
        roi.add_object_class(ObjectClass::Valid(1));
        assert!(roi.object_class.contains(&ObjectClass::Valid(1)));

        roi.add_object_class(ObjectClass::Unset);
        assert!(
            !roi.object_class.contains(&ObjectClass::Unset),
            "Unset must never be recorded as a real class"
        );
        assert_eq!(roi.object_class.len(), 1);
    }

    #[test]
    fn add_object_class_is_idempotent() {
        let mut roi = RoiSettings::default();
        roi.add_object_class(ObjectClass::Valid(5));
        roi.add_object_class(ObjectClass::Valid(5));
        assert_eq!(roi.object_class.len(), 1);
    }

    #[test]
    fn remove_object_class_only_removes_the_given_class() {
        let mut roi = RoiSettings::default();
        roi.add_object_class(ObjectClass::Valid(1));
        roi.add_object_class(ObjectClass::Valid(2));

        roi.remove_object_class(&ObjectClass::Valid(1));

        assert!(!roi.object_class.contains(&ObjectClass::Valid(1)));
        assert!(roi.object_class.contains(&ObjectClass::Valid(2)));
    }

    #[test]
    fn remove_object_class_on_a_missing_class_is_a_no_op() {
        let mut roi = RoiSettings::default();
        roi.add_object_class(ObjectClass::Valid(2));
        roi.remove_object_class(&ObjectClass::Valid(99));
        assert_eq!(roi.object_class.len(), 1);
    }
}
