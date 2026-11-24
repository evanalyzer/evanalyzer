// # export_settings
//
// The project's exports (`ProjectSettings::exports`): named recipes of the
// tables, images and scripts to write from a run's results, where to and
// whether after every run. The reference for every key, its values, its
// default and the files written is `docs/exports.md` - keep both in step.
//
// These are part of the project file and must keep reading: a new key or
// value comes with a default; renaming or removing one needs a schema
// version and a migration step.
//
// **Author:** Joachim Danmayr
// **Date:** 2026-11-24
//
// ## License
// Copyright 2026 Joachim Danmayr.
// Licensed under the **PolyForm Noncommercial License 1.0.0**.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// One export: what it writes (`entries`, in this order), where to
/// (`output`) and whether after every run.
#[derive(Serialize, Deserialize, Debug, JsonSchema, Clone, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct ExportSettings {
    /// Stable while the export exists, unique among the exports.
    pub id: u32,
    /// Shown in the app, `evanalyzer cli export run <name>`. Not empty,
    /// unique ignoring case.
    pub name: String,
    /// Put in front of every file and folder the export writes
    /// (`{prefix}_{name}`); `""`: none.
    #[serde(default)]
    pub prefix: String,
    /// The unit of lengths, areas and volumes in the tables - also in their
    /// column names (`area_um2`, `area_nm2`).
    #[serde(default)]
    pub length_unit: LengthUnit,
    /// Where the files go.
    #[serde(default)]
    pub output: ExportOutput,
    /// Run the export when an analysis run finishes.
    #[serde(default)]
    pub run_after_every_run: bool,
    /// The table format the files are written in - pinned, so a newer app
    /// keeps writing it until the user changes it.
    #[serde(default = "default_table_format")]
    pub table_format: u32,
    /// What to write, in this order: a script sees everything written by
    /// the entries before it.
    #[serde(default)]
    pub entries: Vec<ExportEntry>,
}

/// The table format a new export is written in.
pub const CURRENT_TABLE_FORMAT: u32 = 1;

fn default_table_format() -> u32 {
    CURRENT_TABLE_FORMAT
}

/// See [`ExportSettings::length_unit`].
#[derive(Serialize, Deserialize, Debug, Default, JsonSchema, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum LengthUnit {
    /// Micrometres (µm, µm², µm³).
    #[default]
    Um,
    /// Nanometres (nm, nm², nm³).
    Nm,
}

/// Where an export's files go.
#[derive(Serialize, Deserialize, Debug, Default, JsonSchema, Clone, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct ExportOutput {
    #[serde(default)]
    pub location: OutputLocation,
    /// `ResultsFolder`: a subfolder inside it, `""` = the results folder
    /// itself. `FixedFolder`: a path relative to the project folder (an
    /// absolute one works but isn't portable). `/` as separator, never `..`.
    #[serde(default)]
    pub folder: String,
    /// What happens to a file that is already there.
    #[serde(default)]
    pub if_exists: IfExists,
}

/// See [`ExportOutput::location`].
#[derive(Serialize, Deserialize, Debug, Default, JsonSchema, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum OutputLocation {
    /// The folder of the results database (`.evadb`) the export runs for.
    #[default]
    ResultsFolder,
    /// The same folder for every run.
    FixedFolder,
}

/// See [`ExportOutput::if_exists`] - per file.
#[derive(Serialize, Deserialize, Debug, Default, JsonSchema, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum IfExists {
    /// Replace it.
    #[default]
    Overwrite,
    /// Keep it, write the new one as `name_2.csv`, `name_3.csv`, ...
    Rename,
}

/// One thing an export writes (`"kind"` says which).
#[derive(Serialize, Deserialize, Debug, JsonSchema, Clone, PartialEq)]
#[serde(tag = "kind", rename_all = "SCREAMING_SNAKE_CASE")]
pub enum ExportEntry {
    Table(TableExport),
    PlateLayout(PlateLayoutExport),
    DrawnImages(DrawnImagesExport),
    Script(ScriptExport),
}

/// A table of the results (table format: `ExportSettings::table_format`).
#[derive(Serialize, Deserialize, Debug, JsonSchema, Clone, PartialEq)]
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
    /// Grouped rows: a row also for classes without objects.
    #[serde(default)]
    pub empty_classes: bool,
    /// The classes side by side: a column per value and class instead of a
    /// row per class - `Object` and `Image` rows only.
    #[serde(default)]
    pub transposed: bool,
    /// One file (or sheet) per image, well, folder or class.
    #[serde(default)]
    pub split: TableSplit,
    /// The split parts as files or as sheets of one workbook (`Xlsx` only).
    #[serde(default)]
    pub split_into: SplitInto,
    /// `rows: Track` only.
    #[serde(default)]
    pub tracks: Option<TrackOptions>,
}

fn default_aggregations() -> Vec<ExportAggregation> {
    vec![ExportAggregation::Avg]
}

/// What a row of a table is.
#[derive(Serialize, Deserialize, Debug, JsonSchema, Clone, Copy, PartialEq, Eq)]
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
}

/// The file format of a table.
#[derive(Serialize, Deserialize, Debug, Default, JsonSchema, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum TableFileFormat {
    #[default]
    Csv,
    Xlsx,
}

/// How grouped rows sum up their objects' values.
#[derive(Serialize, Deserialize, Debug, JsonSchema, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum ExportAggregation {
    Avg,
    Median,
    Stddev,
    Sum,
    Min,
    Max,
    Skewness,
}

/// Which images of a run.
#[derive(Serialize, Deserialize, Debug, Default, JsonSchema, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum ImageScope {
    All,
    /// The ones not left out.
    #[default]
    Included,
    LeftOut,
}

/// See [`TableExport::split`].
#[derive(Serialize, Deserialize, Debug, Default, JsonSchema, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum TableSplit {
    #[default]
    None,
    Image,
    Well,
    Folder,
    Class,
}

/// See [`TableExport::split_into`].
#[derive(Serialize, Deserialize, Debug, Default, JsonSchema, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum SplitInto {
    #[default]
    Files,
    /// One workbook, a sheet each (`Xlsx` only).
    Sheets,
}

/// Which tracks a `rows: Track` table has, and the unit of their speed.
#[derive(Serialize, Deserialize, Debug, Default, JsonSchema, Clone, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct TrackOptions {
    /// Only tracks with at least this many time points.
    #[serde(default = "default_min_points")]
    pub min_points: u32,
    /// Only tracks there from the first time point.
    #[serde(default)]
    pub from_start: bool,
    /// Leave out tracks with an object touching the image edge.
    #[serde(default)]
    pub no_edge: bool,
    #[serde(default)]
    pub speed_unit: SpeedUnit,
}

fn default_min_points() -> u32 {
    1
}

/// The unit of a track's speed.
#[derive(Serialize, Deserialize, Debug, Default, JsonSchema, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum SpeedUnit {
    /// Pixels per time point (results without a pixel size or interval).
    PxFrame,
    /// Micrometres per minute.
    #[default]
    UmMin,
    /// Micrometres per second.
    UmS,
}

/// The values laid out as the plate (or well) grid - an `xlsx` workbook, a
/// sheet per class and value.
#[derive(Serialize, Deserialize, Debug, JsonSchema, Clone, PartialEq)]
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
    #[serde(default = "default_aggregation")]
    pub aggregation: ExportAggregation,
    #[serde(default)]
    pub images: ImageScope,
}

fn default_aggregation() -> ExportAggregation {
    ExportAggregation::Avg
}

/// See [`PlateLayoutExport::grid`].
#[derive(Serialize, Deserialize, Debug, Default, JsonSchema, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum PlateGrid {
    /// A cell per well.
    #[default]
    Plate,
    /// A cell per image position inside the wells.
    Well,
}

/// A PNG per image (and plane) with the objects drawn on it.
#[derive(Serialize, Deserialize, Debug, JsonSchema, Clone, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct DrawnImagesExport {
    /// Replaces the automatic file name (`drawn_images`); `""`: the automatic one.
    #[serde(default)]
    pub name: String,
    /// The classes drawn (class ids); `None`: all classes.
    #[serde(default)]
    pub classes: Option<Vec<u32>>,
    #[serde(default)]
    pub style: DrawnStyle,
    #[serde(default)]
    pub brightness: DrawnBrightness,
    /// The longer side at most, in pixels.
    #[serde(default = "default_max_side")]
    pub max_side: u32,
    #[serde(default)]
    pub images: ImageScope,
}

fn default_max_side() -> u32 {
    2048
}

/// How an object is drawn.
#[derive(Serialize, Deserialize, Debug, Default, JsonSchema, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum DrawnStyle {
    /// Its contour, in its class's colour.
    #[default]
    Outline,
    /// Filled half transparent, its contour opaque.
    Filled,
    /// Its bounding box.
    Box,
}

/// Where each channel's display range comes from.
#[derive(Serialize, Deserialize, Debug, Default, JsonSchema, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum DrawnBrightness {
    /// As set in the viewer.
    #[default]
    Viewer,
    /// Automatic per image.
    Auto,
}

/// A script run after the entries before it, on their files:
/// `<interpreter> <file> --input <output folder> --output <output folder>/<output>`,
/// in the project folder.
#[derive(Serialize, Deserialize, Debug, JsonSchema, Clone, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct ScriptExport {
    /// Which interpreter - where it is installed is a setting of the
    /// computer (or server), never of the project.
    pub language: ScriptLanguage,
    /// Relative to the project folder, inside `scripts/`.
    pub file: String,
    /// The subfolder of the export's output folder the script writes into;
    /// `""`: the file name without its extension.
    #[serde(default)]
    pub output: String,
}

/// See [`ScriptExport::language`].
#[derive(Serialize, Deserialize, Debug, JsonSchema, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum ScriptLanguage {
    R,
    Python,
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// The example of `docs/exports.md` reads, with every default filled in.
    #[test]
    fn the_documented_example_reads() {
        let export: ExportSettings = serde_json::from_value(json!({
            "id": 3,
            "name": "Per image for R",
            "prefix": "exp1",
            "lengthUnit": "UM",
            "output": { "location": "RESULTS_FOLDER", "folder": "", "ifExists": "OVERWRITE" },
            "runAfterEveryRun": true,
            "tableFormat": 1,
            "entries": [
                { "kind": "TABLE", "rows": "OBJECT", "format": "CSV", "split": "IMAGE" },
                { "kind": "TABLE", "rows": "IMAGE", "name": "per_image",
                  "classes": [3, 5], "columns": ["count", "area_nm2"],
                  "aggregations": ["AVG", "MEDIAN"] },
                { "kind": "DRAWN_IMAGES", "classes": [3], "style": "OUTLINE" },
                { "kind": "SCRIPT", "language": "R", "file": "scripts/uptake_stats.R", "output": "stats" }
            ]
        }))
        .expect("the documented example");

        assert_eq!(export.entries.len(), 4);
        let ExportEntry::Table(objects) = &export.entries[0] else {
            panic!("a table");
        };
        assert_eq!(objects.rows, TableRows::Object);
        assert_eq!(objects.split, TableSplit::Image);
        assert_eq!(objects.aggregations, vec![ExportAggregation::Avg]);
        assert_eq!(objects.images, ImageScope::Included);
        let ExportEntry::DrawnImages(drawn) = &export.entries[2] else {
            panic!("drawn images");
        };
        assert_eq!(drawn.max_side, 2048);
        assert_eq!(drawn.brightness, DrawnBrightness::Viewer);
    }

    /// The least an export needs: its id and name - everything else has its
    /// default.
    #[test]
    fn an_export_with_only_id_and_name_has_the_defaults() {
        let export: ExportSettings =
            serde_json::from_value(json!({ "id": 1, "name": "Tables" })).unwrap();
        assert_eq!(export.prefix, "");
        assert_eq!(export.length_unit, LengthUnit::Um);
        assert_eq!(export.output, ExportOutput::default());
        assert_eq!(export.output.location, OutputLocation::ResultsFolder);
        assert_eq!(export.output.if_exists, IfExists::Overwrite);
        assert!(!export.run_after_every_run);
        assert_eq!(export.table_format, CURRENT_TABLE_FORMAT);
        assert!(export.entries.is_empty());
    }

    /// What is written reads back the same (also every entry kind).
    #[test]
    fn an_export_round_trips() {
        let export = ExportSettings {
            id: 7,
            name: "All".into(),
            prefix: "p".into(),
            length_unit: LengthUnit::Nm,
            output: ExportOutput {
                location: OutputLocation::FixedFolder,
                folder: "exports".into(),
                if_exists: IfExists::Rename,
            },
            run_after_every_run: true,
            table_format: 1,
            entries: vec![
                ExportEntry::Table(TableExport {
                    name: String::new(),
                    rows: TableRows::Track,
                    format: TableFileFormat::Xlsx,
                    classes: None,
                    columns: Some(vec!["count".into()]),
                    aggregations: vec![ExportAggregation::Median],
                    images: ImageScope::All,
                    empty_classes: true,
                    transposed: false,
                    split: TableSplit::Class,
                    split_into: SplitInto::Sheets,
                    tracks: Some(TrackOptions {
                        min_points: 3,
                        from_start: true,
                        no_edge: true,
                        speed_unit: SpeedUnit::UmS,
                    }),
                }),
                ExportEntry::PlateLayout(PlateLayoutExport {
                    name: String::new(),
                    grid: PlateGrid::Well,
                    classes: Some(vec![2]),
                    columns: None,
                    aggregation: ExportAggregation::Max,
                    images: ImageScope::LeftOut,
                }),
                ExportEntry::DrawnImages(DrawnImagesExport {
                    name: "pictures".into(),
                    classes: None,
                    style: DrawnStyle::Box,
                    brightness: DrawnBrightness::Auto,
                    max_side: 1024,
                    images: ImageScope::Included,
                }),
                ExportEntry::Script(ScriptExport {
                    language: ScriptLanguage::Python,
                    file: "scripts/plot.py".into(),
                    output: String::new(),
                }),
            ],
        };
        let json = serde_json::to_value(&export).unwrap();
        assert_eq!(json["entries"][0]["kind"], "TABLE");
        assert_eq!(json["entries"][1]["kind"], "PLATE_LAYOUT");
        assert_eq!(json["output"]["location"], "FIXED_FOLDER");
        let back: ExportSettings = serde_json::from_value(json).unwrap();
        assert_eq!(back, export);
    }

    /// An unknown kind is an error, not silently skipped: a newer app's
    /// export must never be dropped by an older one (the schema version
    /// keeps such files from being opened at all).
    #[test]
    fn an_unknown_entry_kind_is_an_error() {
        let read = serde_json::from_value::<ExportSettings>(json!({
            "id": 1, "name": "x", "entries": [{ "kind": "CHART" }]
        }));
        assert!(read.is_err());
    }
}
