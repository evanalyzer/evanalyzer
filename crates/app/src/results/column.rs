use evanalyzer_cfg::{core_types::ObjectClass, settings::classification_settings::Class};

#[derive(Debug, Default, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum Column {
    ObjectId,
    ImageName,
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
}

impl Column {
    /// Stable string key (matches the underlying database column name, with
    /// a `_ch{n}` suffix for the per-channel intensity variants) used to
    /// store this variant in UI widgets that only accept strings, e.g. the
    /// slint columns dropdown. Owned (not `&'static str`) because the
    /// channel number/class name has to be formatted in.
    ///
    /// `classes` (typically `ResultsGenerator::get_object_classes()`, cached
    /// there so this is cheap to call repeatedly) resolves
    /// `ColocCount(ObjectClass::Valid(id))`'s class name for the key — falls
    /// back to the raw numeric id if `classes` doesn't (yet, or any longer)
    /// recognize that id, e.g. stale GUI state after switching databases.
    pub fn as_key(&self, classes: &[Class]) -> String {
        match self {
            Column::ObjectId => "object_id".to_string(),
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
            Column::ColocCount(ObjectClass::Valid(class_id)) => {
                let name = classes
                    .iter()
                    .find(|class| class.id == ObjectClass::Valid(*class_id))
                    .map(|class| class.name.clone())
                    .unwrap_or_else(|| class_id.to_string());
                format!("n_colocalized_class_{name}")
            }
            Column::ColocCount(ObjectClass::Unset) => "n_colocalized_unset".to_string(),
            Column::IntensityAvg(channel) => format!("mean_scaled_ch{channel}"),
            Column::IntensitySum(channel) => format!("sum_scaled_ch{channel}"),
            Column::IntensityMin(channel) => format!("min_scaled_ch{channel}"),
            Column::IntensityMax(channel) => format!("max_scaled_ch{channel}"),
        }
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
                .find(|class| class.id == *class_id)
                .map(|class| format!("Coloc with {}", class.name))
                .unwrap_or_else(|| match class_id {
                    ObjectClass::Valid(n) => format!("Coloc with class {n}"),
                    ObjectClass::Unset => "Coloc with unset".to_string(),
                }),
            Column::IntensityAvg(channel) => format!("Avg Intensity (Ch {channel})"),
            Column::IntensitySum(channel) => format!("Sum Intensity (Ch {channel})"),
            Column::IntensityMin(channel) => format!("Min Intensity (Ch {channel})"),
            Column::IntensityMax(channel) => format!("Max Intensity (Ch {channel})"),
        }
    }

    /// Inverse of [`Column::as_key`] — needs the same `classes` list to
    /// resolve a `"n_colocalized_class_{name}"` key back to the class's id;
    /// `None` if `name` isn't (or no longer is) a registered class.
    pub fn from_key(key: &str, classes: &[Class]) -> Option<Self> {
        if let Some(name) = key.strip_prefix("n_colocalized_class_") {
            let class_id = classes.iter().find(|class| class.name == name)?.id;
            return Some(Column::ColocCount(class_id));
        }
        if key == "n_colocalized_unset" {
            return Some(Column::ColocCount(ObjectClass::Unset));
        }
        if let Some(channel) = key.strip_prefix("mean_scaled_ch") {
            return channel.parse().ok().map(Column::IntensityAvg);
        }
        if let Some(channel) = key.strip_prefix("sum_scaled_ch") {
            return channel.parse().ok().map(Column::IntensitySum);
        }
        if let Some(channel) = key.strip_prefix("min_scaled_ch") {
            return channel.parse().ok().map(Column::IntensityMin);
        }
        if let Some(channel) = key.strip_prefix("max_scaled_ch") {
            return channel.parse().ok().map(Column::IntensityMax);
        }
        Some(match key {
            "object_id" => Column::ObjectId,
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
