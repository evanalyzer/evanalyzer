use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// How images are grouped into wells.
#[derive(Serialize, Deserialize, Default, JsonSchema, Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum GroupingMode {
    /// The well-name pattern is detected from the image names (e.g.
    /// `A01_01.tif`, `Exp_B3_s2.tif`).
    #[default]
    Auto,
    /// `PlateSettings::grouping_regex` on the image name.
    Custom,
    /// One well per image folder - placed on the plate when the folder is
    /// named like a well (`B03/`).
    Folder,
}

/// The plate format: detected from the wells found (`Auto`) or a fixed
/// size. One list for the project settings and the results window.
#[derive(Serialize, Deserialize, Default, JsonSchema, Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum PlateSize {
    #[default]
    Auto,
    Plate1x1,
    Plate2x3,
    Plate2x4,
    Plate2x6,
    Plate3x4,
    Plate3x5,
    Plate3x6,
    Plate4x6,
    Plate6x8,
    Plate8x12,
    Plate16x24,
    Plate32x48,
    Plate48x72,
}

impl PlateSize {
    /// Every choice, in the order the dropdowns show them.
    pub const ALL: [PlateSize; 14] = [
        PlateSize::Auto,
        PlateSize::Plate1x1,
        PlateSize::Plate2x3,
        PlateSize::Plate2x4,
        PlateSize::Plate2x6,
        PlateSize::Plate3x4,
        PlateSize::Plate3x5,
        PlateSize::Plate3x6,
        PlateSize::Plate4x6,
        PlateSize::Plate6x8,
        PlateSize::Plate8x12,
        PlateSize::Plate16x24,
        PlateSize::Plate32x48,
        PlateSize::Plate48x72,
    ];

    /// `(rows, cols)`; `None` for `Auto`.
    pub const fn dimensions(self) -> Option<(usize, usize)> {
        Some(match self {
            PlateSize::Auto => return None,
            PlateSize::Plate1x1 => (1, 1),
            PlateSize::Plate2x3 => (2, 3),
            PlateSize::Plate2x4 => (2, 4),
            PlateSize::Plate2x6 => (2, 6),
            PlateSize::Plate3x4 => (3, 4),
            PlateSize::Plate3x5 => (3, 5),
            PlateSize::Plate3x6 => (3, 6),
            PlateSize::Plate4x6 => (4, 6),
            PlateSize::Plate6x8 => (6, 8),
            PlateSize::Plate8x12 => (8, 12),
            PlateSize::Plate16x24 => (16, 24),
            PlateSize::Plate32x48 => (32, 48),
            PlateSize::Plate48x72 => (48, 72),
        })
    }

    /// The fixed size with these dimensions, if there is one.
    pub fn from_dimensions(rows: usize, cols: usize) -> Option<PlateSize> {
        PlateSize::ALL
            .into_iter()
            .find(|size| size.dimensions() == Some((rows, cols)))
    }

    /// Shown in every plate size dropdown, e.g. "96 Well (8 x 12)".
    pub fn label(self) -> String {
        match self.dimensions() {
            None => "Auto".to_string(),
            Some((1, 1)) => "1 Well (1 x 1)".to_string(),
            Some((rows, cols)) => format!("{} Well ({rows} x {cols})", rows * cols),
        }
    }
}

/// The grid of images (fields) inside a well.
#[derive(Serialize, Deserialize, Default, JsonSchema, Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum WellLayout {
    /// The smallest square-ish grid that holds the highest image number of
    /// the well, images placed in number order.
    #[default]
    Auto,
    /// A fixed grid; `PlateSettings::well_image_order` says which image goes
    /// where.
    Fixed { rows: u32, cols: u32 },
}

#[derive(Serialize, Deserialize, Default, Debug, Clone, PartialEq, Eq, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct PlateSettings {
    pub grouping_mode: GroupingMode,
    /// The regex for `GroupingMode::Custom`, kept while another mode is
    /// selected. Its groups: 1 the well (`B03`), 2 the plate row (`B`), 3
    /// the plate column (`03`), 4 the image number in the well (`01`).
    pub grouping_regex: String,
    pub plate_size: PlateSize,
    pub well_layout: WellLayout,
    /// For `WellLayout::Fixed`: the image number shown at each grid
    /// position, row by row (`[1, 2, 3, 4]` = in number order). Positions
    /// without an entry stay empty.
    pub well_image_order: Vec<u32>,
}

impl PlateSettings {
    /// `well_image_order` for a fresh `rows` x `cols` grid: in number order.
    pub fn default_image_order(rows: u32, cols: u32) -> Vec<u32> {
        (1..=rows * cols).collect()
    }

    /// Sets the well layout, keeping `well_image_order` one entry per grid
    /// position: the existing entries stay, a bigger grid gets the next
    /// numbers appended, a smaller one is cut.
    pub fn set_well_layout(&mut self, layout: WellLayout) {
        self.well_layout = layout;
        if let WellLayout::Fixed { rows, cols } = layout {
            let n = (rows * cols) as usize;
            self.well_image_order.truncate(n);
            let next = self.well_image_order.len() as u32 + 1;
            self.well_image_order.extend(next..=n as u32);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_fixed_plate_size_round_trips_through_its_dimensions() {
        for size in PlateSize::ALL {
            match size.dimensions() {
                None => assert_eq!(size, PlateSize::Auto),
                Some((rows, cols)) => {
                    assert_eq!(PlateSize::from_dimensions(rows, cols), Some(size))
                }
            }
        }
        assert_eq!(PlateSize::from_dimensions(5, 5), None);
    }

    #[test]
    fn plate_size_labels_are_unique_and_readable() {
        let labels: Vec<String> = PlateSize::ALL.iter().map(|s| s.label()).collect();
        assert_eq!(labels[0], "Auto");
        assert!(labels.contains(&"96 Well (8 x 12)".to_string()));
        let mut unique = labels.clone();
        unique.sort();
        unique.dedup();
        assert_eq!(unique.len(), labels.len());
    }

    #[test]
    fn set_well_layout_keeps_one_image_order_entry_per_position() {
        let mut settings = PlateSettings::default();
        settings.set_well_layout(WellLayout::Fixed { rows: 2, cols: 2 });
        assert_eq!(settings.well_image_order, vec![1, 2, 3, 4]);
        settings.well_image_order = vec![4, 3, 2, 1];
        settings.set_well_layout(WellLayout::Fixed { rows: 2, cols: 3 });
        assert_eq!(settings.well_image_order, vec![4, 3, 2, 1, 5, 6]);
        settings.set_well_layout(WellLayout::Fixed { rows: 1, cols: 2 });
        assert_eq!(settings.well_image_order, vec![4, 3]);
        // Auto keeps the order for when a fixed grid is chosen again.
        settings.set_well_layout(WellLayout::Auto);
        assert_eq!(settings.well_image_order, vec![4, 3]);
    }

    #[test]
    fn plate_settings_serialize_auto_and_fixed_values_readably() {
        let settings = PlateSettings {
            grouping_mode: GroupingMode::Custom,
            grouping_regex: "x".into(),
            plate_size: PlateSize::Plate8x12,
            well_layout: WellLayout::Fixed { rows: 2, cols: 3 },
            well_image_order: vec![1, 2, 3],
        };
        let json = serde_json::to_value(&settings).unwrap();
        assert_eq!(json["groupingMode"], "CUSTOM");
        assert_eq!(json["plateSize"], "PLATE8X12");
        assert_eq!(json["wellLayout"]["FIXED"]["rows"], 2);
        let back: PlateSettings = serde_json::from_value(json).unwrap();
        assert_eq!(back, settings);

        let auto = serde_json::to_value(PlateSettings::default()).unwrap();
        assert_eq!(auto["groupingMode"], "AUTO");
        assert_eq!(auto["plateSize"], "AUTO");
        assert_eq!(auto["wellLayout"], "AUTO");
    }
}
