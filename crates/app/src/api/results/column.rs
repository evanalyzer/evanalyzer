//! # column
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-09-14
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.

use crate::api::Aggregation;
use evanalyzer_cfg::{core_types::ObjectClass, settings::classification_settings::Class};
use serde::{Deserialize, Serialize};

/// What a [`Column::Children`] column shows about a parent's children.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum ChildValue {
    /// How many children it has (0 without any).
    Count,
    /// An aggregate of one of their columns (empty without children).
    Of(Aggregation, Box<Column>),
}

/// The children of a class linked to the row's object by Relate Objects
/// (`parent_ids[parent_class]` is the row's object): their number or an
/// aggregate of their values - e.g. the spots per cell and their mean
/// intensity.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct ChildColumn {
    /// The class the children were linked through - the row's class.
    pub parent_class: ObjectClass,
    pub child_class: ObjectClass,
    pub value: ChildValue,
}

/// What a [`Column::Parent`] column shows about a row's parent.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum ParentValue {
    /// Its object id (empty without a parent).
    Id,
    /// One of its columns - e.g. its cell's area next to each spot.
    Of(Box<Column>),
}

/// The row's parent of a class (Relate Objects, `parent_ids[parent_class]`).
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct ParentColumn {
    pub parent_class: ObjectClass,
    pub value: ParentValue,
}

/// A measurement step's measurement as a column: its key in the
/// `measurements` table and how it reads.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct MeasurementColumn {
    /// The measurement's key ("pearson_01/pearson").
    pub key: String,
    /// The measurement step's ID ("pearson_01").
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
    /// A volume's object's size (3D): its voxels, in nm³; empty for a
    /// plane's object (its area).
    VolumePx,
    VolumeNm,
    /// A volume's object's surface, in nm².
    SurfaceAreaNm,
    PerimeterPx,
    PerimeterNm,
    Circularity,
    Solidity,
    Eccentricity,
    /// Feret (caliper) diameters over the pixel centres, in pixels.
    MaxFeret,
    MinFeret,
    /// The fitted ellipse's axes, in pixels.
    MajorAxis,
    MinorAxis,
    /// The diameter of a disk of the object's area, in pixels.
    EquivalentDiameter,
    /// The area over the bounding box's.
    Extent,
    ColocCount(ObjectClass),
    IntensityAvg(u32),
    IntensitySum(u32),
    IntensityMin(u32),
    IntensityMax(u32),
    /// How the pixel values spread in a channel (CellProfiler's
    /// MeasureObjectIntensity).
    IntensityStd(u32),
    IntensityMedian(u32),
    IntensityLowerQuartile(u32),
    IntensityUpperQuartile(u32),
    /// Median absolute deviation from the median.
    IntensityMad(u32),
    /// A measurement step's measurement (`measurements` table).
    Measurement(MeasurementColumn),
    /// The object a measurement's value names (its nearest neighbour, its
    /// parent): its id (`measurements.related_object_id`).
    MeasurementObject(MeasurementColumn),
    /// The row's children of a class: their number or an aggregate.
    Children(ChildColumn),
    /// The row's parent of a class: its id or one of its columns.
    Parent(ParentColumn),
}

/// The class part of a relation column's key: the class's number, or
/// `unset`.
fn class_key_part(class: &ObjectClass) -> String {
    match class {
        ObjectClass::Global(n) => n.to_string(),
        ObjectClass::Unset | ObjectClass::Local(_) => "unset".to_string(),
    }
}

fn class_from_key_part(part: &str) -> Option<ObjectClass> {
    match part {
        "unset" => Some(ObjectClass::Unset),
        n => n.parse().ok().map(ObjectClass::Global),
    }
}

/// An aggregation in a key.
fn aggregation_key(aggregation: &Aggregation) -> &'static str {
    match aggregation {
        Aggregation::Avg => "avg",
        Aggregation::Min => "min",
        Aggregation::Max => "max",
        Aggregation::Stddev => "stddev",
        Aggregation::Sum => "sum",
        Aggregation::Median => "median",
        Aggregation::Skewness => "skewness",
    }
}

fn aggregation_from_key(key: &str) -> Option<Aggregation> {
    Some(match key {
        "avg" => Aggregation::Avg,
        "min" => Aggregation::Min,
        "max" => Aggregation::Max,
        "stddev" => Aggregation::Stddev,
        "sum" => Aggregation::Sum,
        "median" => Aggregation::Median,
        "skewness" => Aggregation::Skewness,
        _ => return None,
    })
}

/// An aggregation as the user reads it ("Mean").
pub fn aggregation_label(aggregation: &Aggregation) -> &'static str {
    match aggregation {
        Aggregation::Avg => "Mean",
        Aggregation::Min => "Min",
        Aggregation::Max => "Max",
        Aggregation::Stddev => "Std dev",
        Aggregation::Sum => "Sum",
        Aggregation::Median => "Median",
        Aggregation::Skewness => "Skewness",
    }
}

/// A class's name, or "class N" for one `classes` doesn't know.
fn class_label(class: &ObjectClass, classes: &[Class]) -> String {
    classes
        .iter()
        .find(|c| c.global() == *class)
        .map(|c| c.name.clone())
        .unwrap_or_else(|| match class {
            ObjectClass::Global(n) => format!("class {n}"),
            ObjectClass::Unset | ObjectClass::Local(_) => "unset".to_string(),
        })
}

/// How a measurement column's [`Column::as_key`] starts.
const MEASUREMENT_KEY_PREFIX: &str = "measurement:";

/// How a [`Column::MeasurementObject`]'s key starts.
const MEASUREMENT_OBJECT_KEY_PREFIX: &str = "measurement_object:";

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
            Column::VolumePx => "volume_px".to_string(),
            Column::VolumeNm => "volume_nm3".to_string(),
            Column::SurfaceAreaNm => "surface_area_nm2".to_string(),
            Column::PerimeterPx => "perimeter_px".to_string(),
            Column::PerimeterNm => "perimeter_nm".to_string(),
            Column::Circularity => "circularity".to_string(),
            Column::Solidity => "solidity".to_string(),
            Column::Eccentricity => "eccentricity".to_string(),
            Column::MaxFeret => "feret_diameter_px".to_string(),
            Column::MinFeret => "min_feret_px".to_string(),
            Column::MajorAxis => "major_axis_px".to_string(),
            Column::MinorAxis => "minor_axis_px".to_string(),
            Column::EquivalentDiameter => "equivalent_diameter_px".to_string(),
            Column::Extent => "extent".to_string(),
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
            Column::IntensityStd(channel) => format!("intensity_std_gray_ch{channel}"),
            Column::IntensityMedian(channel) => format!("intensity_median_gray_ch{channel}"),
            Column::IntensityLowerQuartile(channel) => {
                format!("intensity_lower_quartile_gray_ch{channel}")
            }
            Column::IntensityUpperQuartile(channel) => {
                format!("intensity_upper_quartile_gray_ch{channel}")
            }
            Column::IntensityMad(channel) => format!("intensity_mad_gray_ch{channel}"),
            Column::Measurement(m) => format!("{MEASUREMENT_KEY_PREFIX}{}", m.key),
            Column::MeasurementObject(m) => format!("{MEASUREMENT_OBJECT_KEY_PREFIX}{}", m.key),
            // children:<parent>:<child>:count | ...:<aggregation>:<column key>
            Column::Children(c) => {
                let classes_part = format!(
                    "children:{}:{}",
                    class_key_part(&c.parent_class),
                    class_key_part(&c.child_class)
                );
                match &c.value {
                    ChildValue::Count => format!("{classes_part}:count"),
                    ChildValue::Of(aggregation, column) => format!(
                        "{classes_part}:{}:{}",
                        aggregation_key(aggregation),
                        column.as_key(classes)
                    ),
                }
            }
            // parent:<parent>:id | parent:<parent>:<column key>
            Column::Parent(p) => match &p.value {
                ParentValue::Id => format!("parent:{}:id", class_key_part(&p.parent_class)),
                ParentValue::Of(column) => format!(
                    "parent:{}:{}",
                    class_key_part(&p.parent_class),
                    column.as_key(classes)
                ),
            },
        }
    }

    /// The measurement key of a [`Column::as_key`] key - `None` if it isn't
    /// a measurement's. [`Column::from_key`] can't make the column from it
    /// alone (its name comes from the results' `measurement_info`): look it
    /// up among the available columns.
    /// Also of a [`Column::MeasurementObject`]'s key.
    pub fn measurement_key(key: &str) -> Option<&str> {
        key.strip_prefix(MEASUREMENT_KEY_PREFIX)
            .or_else(|| key.strip_prefix(MEASUREMENT_OBJECT_KEY_PREFIX))
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
            Column::VolumePx => "Volume [px³]".to_string(),
            Column::VolumeNm => "Volume [nm³]".to_string(),
            Column::SurfaceAreaNm => "Surface Area [nm²]".to_string(),
            Column::PerimeterPx => "Perimeter [px]".to_string(),
            Column::PerimeterNm => "Perimeter [nm]".to_string(),
            Column::Circularity => "Circularity".to_string(),
            Column::Solidity => "Solidity".to_string(),
            Column::Eccentricity => "Eccentricity".to_string(),
            Column::MaxFeret => "Max Feret [px]".to_string(),
            Column::MinFeret => "Min Feret [px]".to_string(),
            Column::MajorAxis => "Major Axis [px]".to_string(),
            Column::MinorAxis => "Minor Axis [px]".to_string(),
            Column::EquivalentDiameter => "Equivalent Diameter [px]".to_string(),
            Column::Extent => "Extent".to_string(),
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
            Column::IntensityStd(channel) => format!("Std Dev Intensity (Ch {channel})"),
            Column::IntensityMedian(channel) => format!("Median Intensity (Ch {channel})"),
            Column::IntensityLowerQuartile(channel) => {
                format!("Lower Quartile Intensity (Ch {channel})")
            }
            Column::IntensityUpperQuartile(channel) => {
                format!("Upper Quartile Intensity (Ch {channel})")
            }
            Column::IntensityMad(channel) => format!("MAD Intensity (Ch {channel})"),
            Column::Measurement(m) => format!("{} · {}", m.step, m.name),
            Column::MeasurementObject(m) => format!("{} · {} · object", m.step, m.name),
            Column::Children(c) => {
                let child = class_label(&c.child_class, classes);
                match &c.value {
                    ChildValue::Count => format!("{child} · Count"),
                    ChildValue::Of(aggregation, column) => format!(
                        "{child} · {} · {}",
                        aggregation_label(aggregation),
                        column.display_label(classes)
                    ),
                }
            }
            Column::Parent(p) => {
                let parent = class_label(&p.parent_class, classes);
                match &p.value {
                    ParentValue::Id => format!("Parent {parent}"),
                    ParentValue::Of(column) => {
                        format!("Parent {parent} · {}", column.display_label(classes))
                    }
                }
            }
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
            Column::ObjectId
                | Column::TrackId
                | Column::ImageName
                | Column::ObjectClass
                | Column::MeasurementObject(_)
                | Column::Parent(ParentColumn {
                    value: ParentValue::Id,
                    ..
                })
        )
    }

    /// Inverse of [`Column::as_key`] — needs the same `classes` list to
    /// resolve a `"n_colocalized_class_{name}"` key back to the class's id;
    /// `None` if `name` isn't (or no longer is) a registered class.
    pub fn from_key(key: &str, classes: &[Class]) -> Option<Self> {
        if let Some(rest) = key.strip_prefix("children:") {
            let mut parts = rest.splitn(4, ':');
            let parent_class = class_from_key_part(parts.next()?)?;
            let child_class = class_from_key_part(parts.next()?)?;
            let value = match parts.next()? {
                "count" => ChildValue::Count,
                aggregation => ChildValue::Of(
                    aggregation_from_key(aggregation)?,
                    Box::new(Column::from_key(parts.next()?, classes)?),
                ),
            };
            return Some(Column::Children(ChildColumn {
                parent_class,
                child_class,
                value,
            }));
        }
        if let Some(rest) = key.strip_prefix("parent:") {
            let (class, value) = rest.split_once(':')?;
            let value = match value {
                "id" => ParentValue::Id,
                column => ParentValue::Of(Box::new(Column::from_key(column, classes)?)),
            };
            return Some(Column::Parent(ParentColumn {
                parent_class: class_from_key_part(class)?,
                value,
            }));
        }
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
        if let Some(channel) = key.strip_prefix("intensity_std_gray_ch") {
            return channel.parse().ok().map(Column::IntensityStd);
        }
        if let Some(channel) = key.strip_prefix("intensity_median_gray_ch") {
            return channel.parse().ok().map(Column::IntensityMedian);
        }
        if let Some(channel) = key.strip_prefix("intensity_lower_quartile_gray_ch") {
            return channel.parse().ok().map(Column::IntensityLowerQuartile);
        }
        if let Some(channel) = key.strip_prefix("intensity_upper_quartile_gray_ch") {
            return channel.parse().ok().map(Column::IntensityUpperQuartile);
        }
        if let Some(channel) = key.strip_prefix("intensity_mad_gray_ch") {
            return channel.parse().ok().map(Column::IntensityMad);
        }
        Some(match key {
            "object_id" => Column::ObjectId,
            "track_id" => Column::TrackId,
            "image_name" => Column::ImageName,
            "object_class_name" => Column::ObjectClass,
            "count" => Column::Count,
            "area_px" => Column::AreaSizePx,
            "area_nm2" => Column::AreaSizeNm,
            "volume_px" => Column::VolumePx,
            "volume_nm3" => Column::VolumeNm,
            "surface_area_nm2" => Column::SurfaceAreaNm,
            "perimeter_px" => Column::PerimeterPx,
            "perimeter_nm" => Column::PerimeterNm,
            "circularity" => Column::Circularity,
            "solidity" => Column::Solidity,
            "eccentricity" => Column::Eccentricity,
            "feret_diameter_px" => Column::MaxFeret,
            "min_feret_px" => Column::MinFeret,
            "major_axis_px" => Column::MajorAxis,
            "minor_axis_px" => Column::MinorAxis,
            "equivalent_diameter_px" => Column::EquivalentDiameter,
            "extent" => Column::Extent,
            _ => return None,
        })
    }
}
