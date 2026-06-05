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
