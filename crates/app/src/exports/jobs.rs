//! # jobs
//!
//! What the runner writes for one output of a data export
//! (`DataExport::outputs`): a table (`TableExport`) or a plate layout
//! (`PlateLayoutExport`) - the data export's filter with the output's own
//! settings. [`jobs_of`] makes them.
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-11-24
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.

use evanalyzer_cfg::settings::export_settings::{
    DataExport, DataOutput, ExportAggregation, HeatmapColors, ImageScope, Planes, ProfileOptions,
    SplitInto, TableFileFormat, TableSplit, TimeCourseOptions, TrackOptions,
};
use serde::{Deserialize, Serialize};

fn default_aggregations() -> Vec<ExportAggregation> {
    vec![ExportAggregation::Avg]
}

/// One output's job.
#[derive(Debug, Clone, PartialEq)]
pub enum Job {
    Table(TableExport),
    PlateLayout(PlateLayoutExport),
}

/// The jobs of `data`: one per output, in the order the outputs are given,
/// each with the data export's filter.
pub fn jobs_of(data: &DataExport) -> Vec<(DataOutput, Job)> {
    data.outputs
        .iter()
        .map(|output| (*output, job_of(data, *output)))
        .collect()
}

/// `name` in front of an automatic name, else `""` (the automatic one).
fn named(data: &DataExport, automatic: &str) -> String {
    let name = data.name.trim();
    if name.is_empty() {
        String::new()
    } else {
        format!("{name}_{automatic}")
    }
}

fn job_of(data: &DataExport, output: DataOutput) -> Job {
    let rows = match output {
        DataOutput::Objects => Some(TableRows::Object),
        DataOutput::Images => Some(TableRows::Image),
        DataOutput::Wells => Some(TableRows::Well),
        DataOutput::Folders => Some(TableRows::Folder),
        DataOutput::ColocPairs => Some(TableRows::ColocPair),
        DataOutput::Measurements => Some(TableRows::Measurement),
        DataOutput::Tracks => Some(TableRows::Track),
        DataOutput::TimeCourse => Some(TableRows::TimePoint),
        DataOutput::Profile => Some(TableRows::ProfilePosition),
        _ => None,
    };
    if let Some(rows) = rows {
        let automatic = crate::exports::naming::rows_name(rows);
        return Job::Table(TableExport {
            name: named(data, automatic),
            rows,
            format: data.format,
            classes: data.classes.clone(),
            columns: data.columns.clone(),
            aggregations: data.aggregations.clone(),
            images: data.images,
            only_images: data.only_images.clone(),
            group_regex: data.group_regex.clone(),
            empty_classes: data.empty_classes
                && matches!(rows, TableRows::Image | TableRows::Well | TableRows::Folder),
            transposed: data.transposed
                && matches!(
                    rows,
                    TableRows::Object | TableRows::Image | TableRows::Well | TableRows::Folder
                ),
            children: data
                .children
                .iter()
                .map(|link| (link.parent, link.child))
                .collect(),
            child_aggregations: data.child_aggregations.clone(),
            parents: data.parents.clone(),
            split: data.split,
            split_into: data.split_into,
            tracks: (rows == TableRows::Track)
                .then(|| data.tracks.clone())
                .flatten(),
            time_course: (rows == TableRows::TimePoint)
                .then(|| data.time_course.clone().unwrap_or_default()),
            profile: (rows == TableRows::ProfilePosition)
                .then(|| data.profile.clone().unwrap_or_default()),
            planes: if matches!(rows, TableRows::Track | TableRows::ProfilePosition) {
                Planes::default()
            } else {
                data.planes
            },
        });
    }
    let grid = match output {
        DataOutput::WellGrid => PlateGrid::Well,
        DataOutput::WellList => PlateGrid::WellList,
        DataOutput::ImageHeatmap => PlateGrid::Image,
        _ => PlateGrid::Plate,
    };
    let automatic = match grid {
        PlateGrid::Plate => "plate",
        PlateGrid::Well => "well",
        PlateGrid::WellList => "well_list",
        PlateGrid::Image => "heatmap",
    };
    Job::PlateLayout(PlateLayoutExport {
        name: named(data, automatic),
        grid,
        classes: data.classes.clone(),
        columns: data.columns.clone(),
        aggregations: data.aggregations.clone(),
        images: data.images,
        only_images: data.only_images.clone(),
        group_regex: data.group_regex.clone(),
        square_size: data.heatmap.square_size,
        colors: data.heatmap.colors,
        planes: data.planes,
    })
}

/// A table of the results (table format: `ExportSettings::table_format`).
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct TableExport {
    /// Replaces the automatic file name (`objects`, `images`, ...); `""`:
    /// the automatic one.
    #[serde(default)]
    pub name: String,
    /// What a row is.
    pub rows: TableRows,
    #[serde(default)]
    pub format: TableFileFormat,
    /// The classes in the table (class ids); `None`: all classes.
    #[serde(default)]
    pub classes: Option<Vec<u32>>,
    /// The value columns by column key (`count`, `area_nm2`,
    /// `intensity_mean_gray_ch1`, `measurement:<key>`, ...); `None`: the
    /// default columns.
    #[serde(default)]
    pub columns: Option<Vec<String>>,
    /// Grouped rows only (`Image`, `Well`, `Folder`, `Track`).
    #[serde(default = "default_aggregations")]
    pub aggregations: Vec<ExportAggregation>,
    /// Which images of the run.
    #[serde(default)]
    pub images: ImageScope,
    /// Of `images`, only these (relative paths); `None`: all of them.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub only_images: Option<Vec<String>>,
    /// The wells by this regex's first capture group on the image name;
    /// `None`: the results' plate grouping.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub group_regex: Option<String>,
    /// Grouped rows: a row also for classes without objects.
    #[serde(default)]
    pub empty_classes: bool,
    /// The classes side by side: a column per value and class instead of a
    /// row per class - `Object` and `Image` rows only.
    #[serde(default)]
    pub transposed: bool,
    /// The children linked to each object (Relate Objects), as the Results
    /// List shows them: (parent class id, child class id) - per link their
    /// count and each `child_aggregations` of each column.
    #[serde(default)]
    pub children: Vec<(u32, u32)>,
    /// How the children's values are summed up (besides their count).
    #[serde(default)]
    pub child_aggregations: Vec<ExportAggregation>,
    /// The linked objects (parents) by class id: their id and each column.
    #[serde(default)]
    pub parents: Vec<u32>,
    /// One file (or sheet) per image, well, folder or class.
    #[serde(default)]
    pub split: TableSplit,
    /// The split parts as files or as sheets of one workbook (`Xlsx` only).
    #[serde(default)]
    pub split_into: SplitInto,
    /// `rows: Track` only.
    #[serde(default)]
    pub tracks: Option<TrackOptions>,
    /// `rows: TimePoint` only: what a line is, the layout.
    #[serde(default)]
    pub time_course: Option<TimeCourseOptions>,
    /// `rows: ProfilePosition` only: which profile, what a line is.
    #[serde(default)]
    pub profile: Option<ProfileOptions>,
    /// Not for tracks and profiles (they span the planes).
    #[serde(default)]
    pub planes: Planes,
}

/// What a row of a table is.
#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum TableRows {
    /// An object.
    Object,
    /// An image and class.
    Image,
    /// A well and class.
    Well,
    /// An image folder and class.
    Folder,
    /// A pair of overlapping objects (Colocalize Objects' partners).
    ColocPair,
    /// A track.
    Track,
    /// A measurement value (image, plane, object or region, step,
    /// measurement, value, unit).
    Measurement,
    /// A time point of a line (the time course): its values summed up
    /// over the line's objects there.
    TimePoint,
    /// A position of a profile (e.g. a Sholl ring's radius): of a line, or
    /// of an object.
    ProfilePosition,
}
/// The values laid out as the plate (or well) grid - an `xlsx` workbook, a
/// sheet per class and value.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct PlateLayoutExport {
    /// Replaces the automatic file name (`plate`, `well`); `""`: the
    /// automatic one.
    #[serde(default)]
    pub name: String,
    #[serde(default)]
    pub grid: PlateGrid,
    /// Class ids; `None`: all classes.
    #[serde(default)]
    pub classes: Option<Vec<u32>>,
    /// Column keys; `None`: the count.
    #[serde(default)]
    pub columns: Option<Vec<String>>,
    /// How each column is summed up per cell: a sheet per class, column
    /// and aggregation.
    #[serde(default = "default_aggregations")]
    pub aggregations: Vec<ExportAggregation>,
    #[serde(default)]
    pub images: ImageScope,
    /// Of `images`, only these (relative paths); `None`: all of them.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub only_images: Option<Vec<String>>,
    /// The wells by this regex's first capture group on the image name;
    /// `None`: the results' plate grouping.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub group_regex: Option<String>,
    /// `grid: Image`: the side of a square in pixels (`None`: 64).
    #[serde(default)]
    pub square_size: Option<u32>,
    /// The heatmap's colours.
    #[serde(default)]
    pub colors: HeatmapColors,
    #[serde(default)]
    pub planes: Planes,
}

/// See [`PlateLayoutExport::grid`].
#[derive(Serialize, Deserialize, Debug, Default, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum PlateGrid {
    /// A cell per well.
    #[default]
    Plate,
    /// A cell per image position inside the wells.
    Well,
    /// A workbook per image: its area as a grid of squares
    /// (`square_size` pixels), the objects' values summed up per square.
    Image,
    /// The wells as a list: a row per image position (`well_list.xlsx`).
    WellList,
}
