use evanalyzer_cfg::{core_types::ObjectClass, settings::classification_settings::Class};
use serde::{Deserialize, Serialize};

/// A measurement step's measurement as a column: its key in the
/// `measurements` table and how it reads.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct MeasurementColumn {
    /// The measurement's key ("pearson_01/pearson").
    pub key: String,
    /// The measurement step's name ("pearson_01").
    pub step: String,
    /// Readable ("Pearson (channel 0 × channel 1)").
    pub name: String,
    /// A value per object (else per image plane).
    pub per_object: bool,
}

#[derive(Debug, Default, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum Column {
    // The order the List shows them in (`Ord`): the image first.
    ImageName,
    ObjectId,
    TrackId,
    ObjectClass,
    Count,
    #[default]
    AreaSizePx,
    AreaSizeNm,
    PerimeterPx,
    PerimeterNm,
    Circularity,
    Solidity,
    Eccentricity,
    ColocCount(ObjectClass),
    IntensityAvg(u32),
    IntensitySum(u32),
    IntensityMin(u32),
    IntensityMax(u32),
    /// A measurement step's measurement (`measurements` table).
    Measurement(MeasurementColumn),
}

/// How a measurement column's [`Column::as_key`] starts.
const MEASUREMENT_KEY_PREFIX: &str = "measurement:";

impl Column {
    /// Stable string key (matches the underlying database column name, with
    /// a `_ch{n}` suffix for the per-channel intensity variants) used to
    /// store this variant in UI widgets that only accept strings, e.g. the
    /// slint columns dropdown. Owned (not `&'static str`) because the
    /// channel number/class name has to be formatted in.
    ///
    /// `classes` (typically `ResultsGenerator::get_object_classes()`, cached
    /// there so this is cheap to call repeatedly) resolves
    /// `ColocCount(ObjectClass::Global(id))`'s class name for the key — falls
    /// back to the raw numeric id if `classes` doesn't (yet, or any longer)
    /// recognize that id, e.g. stale GUI state after switching databases.
    pub fn as_key(&self, classes: &[Class]) -> String {
        match self {
            Column::ObjectId => "object_id".to_string(),
            Column::TrackId => "track_id".to_string(),
            Column::ImageName => "image_name".to_string(),
            Column::ObjectClass => "object_class_name".to_string(),
            Column::Count => "count".to_string(),
            Column::AreaSizePx => "area_px".to_string(),
            Column::AreaSizeNm => "area_nm2".to_string(),
            Column::PerimeterPx => "perimeter_px".to_string(),
            Column::PerimeterNm => "perimeter_nm".to_string(),
            Column::Circularity => "circularity".to_string(),
            Column::Solidity => "solidity".to_string(),
            Column::Eccentricity => "eccentricity".to_string(),
            Column::ColocCount(ObjectClass::Global(class_id)) => {
                let name = classes
                    .iter()
                    .find(|class| class.id == *class_id)
                    .map(|class| class.name.clone())
                    .unwrap_or_else(|| class_id.to_string());
                format!("n_colocalized_class_{name}")
            }
            Column::ColocCount(ObjectClass::Unset | ObjectClass::Local(_)) => {
                "n_colocalized_unset".to_string()
            }
            Column::IntensityAvg(channel) => format!("intensity_mean_gray_ch{channel}"),
            Column::IntensitySum(channel) => format!("intensity_sum_gray_ch{channel}"),
            Column::IntensityMin(channel) => format!("intensity_min_gray_ch{channel}"),
            Column::IntensityMax(channel) => format!("intensity_max_gray_ch{channel}"),
            Column::Measurement(m) => format!("{MEASUREMENT_KEY_PREFIX}{}", m.key),
        }
    }

    /// The measurement key of a [`Column::as_key`] key - `None` if it isn't
    /// a measurement's. [`Column::from_key`] can't make the column from it
    /// alone (its name comes from the results' `measurement_info`): look it
    /// up among the available columns.
    pub fn measurement_key(key: &str) -> Option<&str> {
        key.strip_prefix(MEASUREMENT_KEY_PREFIX)
    }

    /// Human-facing label for this column — every header/caption actually
    /// shown to the user (the List/Matrix table headers, XLSX export
    /// headers, `get_available_columns()`'s own `display_name`) goes
    /// through this, so renaming what's displayed only ever needs to
    /// happen here. Deliberately separate from [`Column::as_key`], which
    /// looks similar today only because it happens to reuse the database's
    /// own column names as convenient stable strings — `as_key` is a
    /// storage/round-trip key (dropdown persistence, `Column::from_key`),
    /// and changing it would silently break that persistence, so it must
    /// never be (re)used for display text.
    pub fn display_label(&self, classes: &[Class]) -> String {
        match self {
            Column::ObjectId => "Object ID".to_string(),
            Column::TrackId => "Track ID".to_string(),
            Column::ImageName => "Image".to_string(),
            Column::ObjectClass => "Class".to_string(),
            Column::Count => "Count".to_string(),
            Column::AreaSizePx => "Area [px]".to_string(),
            Column::AreaSizeNm => "Area [nm²]".to_string(),
            Column::PerimeterPx => "Perimeter [px]".to_string(),
            Column::PerimeterNm => "Perimeter [nm]".to_string(),
            Column::Circularity => "Circularity".to_string(),
            Column::Solidity => "Solidity".to_string(),
            Column::Eccentricity => "Eccentricity".to_string(),
            Column::ColocCount(class_id) => classes
                .iter()
                .find(|class| class.global() == *class_id)
                .map(|class| format!("Coloc with {}", class.name))
                .unwrap_or_else(|| match class_id {
                    ObjectClass::Global(n) => format!("Coloc with class {n}"),
                    ObjectClass::Unset | ObjectClass::Local(_) => "Coloc with unset".to_string(),
                }),
            Column::IntensityAvg(channel) => format!("Avg Intensity (Ch {channel})"),
            Column::IntensitySum(channel) => format!("Sum Intensity (Ch {channel})"),
            Column::IntensityMin(channel) => format!("Min Intensity (Ch {channel})"),
            Column::IntensityMax(channel) => format!("Max Intensity (Ch {channel})"),
            Column::Measurement(m) => format!("{} · {}", m.step, m.name),
        }
    }

    /// Whether this column can be aggregated over a group of objects (per
    /// image, well, plate or heatmap tile): everything but an object's
    /// identity (id, track, image, class). The one list the GUI's Images mode and
    /// the export's grouped/grid documents filter their columns with -
    /// must match what `column_aggregate_expr` (results_generator.rs)
    /// accepts, plus `Count` (aggregated as `COUNT(*)`).
    /// A measurement with one value per image plane (not per object): the
    /// per-image views take the image's own value; views over objects
    /// (charts, over time, heatmap, tracks) leave it out for now - over
    /// objects it would count once per object and miss images without any.
    pub fn is_image_measurement(&self) -> bool {
        matches!(self, Column::Measurement(m) if !m.per_object)
    }

    pub fn is_aggregable(&self) -> bool {
        !matches!(
            self,
            Column::ObjectId | Column::TrackId | Column::ImageName | Column::ObjectClass
        )
    }

    /// Inverse of [`Column::as_key`] — needs the same `classes` list to
    /// resolve a `"n_colocalized_class_{name}"` key back to the class's id;
    /// `None` if `name` isn't (or no longer is) a registered class.
    pub fn from_key(key: &str, classes: &[Class]) -> Option<Self> {
        if let Some(name) = key.strip_prefix("n_colocalized_class_") {
            let class_id = classes.iter().find(|class| class.name == name)?.global();
            return Some(Column::ColocCount(class_id));
        }
        if key == "n_colocalized_unset" {
            return Some(Column::ColocCount(ObjectClass::Unset));
        }
        if let Some(channel) = key.strip_prefix("intensity_mean_gray_ch") {
            return channel.parse().ok().map(Column::IntensityAvg);
        }
        if let Some(channel) = key.strip_prefix("intensity_sum_gray_ch") {
            return channel.parse().ok().map(Column::IntensitySum);
        }
        if let Some(channel) = key.strip_prefix("intensity_min_gray_ch") {
            return channel.parse().ok().map(Column::IntensityMin);
        }
        if let Some(channel) = key.strip_prefix("intensity_max_gray_ch") {
            return channel.parse().ok().map(Column::IntensityMax);
        }
        Some(match key {
            "object_id" => Column::ObjectId,
            "track_id" => Column::TrackId,
            "image_name" => Column::ImageName,
            "object_class_name" => Column::ObjectClass,
            "count" => Column::Count,
            "area_px" => Column::AreaSizePx,
            "area_nm2" => Column::AreaSizeNm,
            "perimeter_px" => Column::PerimeterPx,
            "perimeter_nm" => Column::PerimeterNm,
            "circularity" => Column::Circularity,
            "solidity" => Column::Solidity,
            "eccentricity" => Column::Eccentricity,
            _ => return None,
        })
    }
}
