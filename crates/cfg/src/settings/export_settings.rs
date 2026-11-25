// # export_settings
//
// The project's exports (`ProjectSettings::exports`): named recipes of the
// tables, images and scripts to write from a run's results, where to and
// whether after analysis. The reference for every key, its values, its
// default and the files written is `docs/exports.md` - keep both in step.
//
// These are part of the project file and must keep reading:
// - a new key comes with a default (an older app ignores it);
// - a new entry kind, or a new value of anything inside an entry (an
//   output, an aggregation, a colour scale ...), reads in an older app as
//   `ExportEntry::Unknown`: kept as it is, saved back unchanged, and that
//   export says it needs a newer version when it is checked or run - the
//   project still opens;
// - a new value of an export's own settings (`lengthUnit`, `output`),
//   renaming or removing anything needs a schema version and a migration
//   step.
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
/// (`output`) and whether after analysis.
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
    /// Run the export as the last step of every analysis ("Run after
    /// analysis"); the run dialog can switch it off for one run.
    #[serde(default)]
    pub run_after_analysis: bool,
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
    Data(DataExport),
    DrawnImages(DrawnImagesExport),
    Script(ScriptExport),
    /// An entry this version can't read - a kind or a value a newer
    /// version added: kept exactly as it is in the file (and written back
    /// so), never run.
    #[serde(untagged)]
    #[schemars(skip)]
    Unknown(serde_json::Value),
}

/// Data of the results - tables and heatmaps - with one filter: each
/// output in `outputs` is written with the same columns, values, classes,
/// planes and images (an output that can't take one of them leaves it out,
/// said in the report). Different filters: a second data export.
#[derive(Serialize, Deserialize, Debug, JsonSchema, Clone, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct DataExport {
    /// A name part in front of the files' automatic names
    /// (`ctrl_objects.csv`); `""`: the automatic names alone.
    #[serde(default)]
    pub name: String,
    /// What it writes - at least one.
    #[serde(default)]
    pub outputs: Vec<DataOutput>,
    /// The tables' file format (heatmaps are always `xlsx`).
    #[serde(default)]
    pub format: TableFileFormat,
    /// The classes (class ids); `None`: all classes.
    #[serde(default)]
    pub classes: Option<Vec<u32>>,
    /// The value columns by column key (`count`, `area_nm2`,
    /// `intensity_mean_gray_ch1`, `measurement:<key>`, ...); `None`: each
    /// output's default columns.
    #[serde(default)]
    pub columns: Option<Vec<String>>,
    /// How values are summed up where rows group objects (per image, well,
    /// folder, track, time point, heatmap cell).
    #[serde(default = "default_aggregations")]
    pub aggregations: Vec<ExportAggregation>,
    /// Which images of the run.
    #[serde(default)]
    pub images: ImageScope,
    /// Of `images`, only these (relative paths, `/` separated); `None`:
    /// all of them.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub only_images: Option<Vec<String>>,
    /// Where the images' wells come from (rows per well, a time course's
    /// well lines, the tracks' well): the first capture group of this
    /// regex on the image name; `None`: the results' plate grouping.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub group_regex: Option<String>,
    /// Which z-planes and time points (not for tracks and profiles).
    #[serde(default)]
    pub planes: Planes,
    /// Rows per image, well and folder: a row also for classes without
    /// objects.
    #[serde(default)]
    pub empty_classes: bool,
    /// Objects and per image: the classes side by side.
    #[serde(default)]
    pub transposed: bool,
    /// The children linked to each object (Relate Objects), as the Results
    /// List shows them.
    #[serde(default)]
    pub children: Vec<ChildLink>,
    /// How the children's values are summed up (besides their count).
    #[serde(default)]
    pub child_aggregations: Vec<ExportAggregation>,
    /// The linked objects (parents) by class id.
    #[serde(default)]
    pub parents: Vec<u32>,
    /// One file (or sheet) per image, well, folder or class - for the tables.
    #[serde(default)]
    pub split: TableSplit,
    /// The split parts as files or as sheets of one workbook (`Xlsx` only).
    #[serde(default)]
    pub split_into: SplitInto,
    /// The `Tracks` output's tracks and speed unit.
    #[serde(default)]
    pub tracks: Option<TrackOptions>,
    /// The `TimeCourse` output: what a line is, the layout.
    #[serde(default)]
    pub time_course: Option<TimeCourseOptions>,
    /// The `Profile` output: which profile, what a line is.
    #[serde(default)]
    pub profile: Option<ProfileOptions>,
    /// The heatmaps' colours and the image heatmap's squares.
    #[serde(default)]
    pub heatmap: HeatmapOptions,
}

/// A link from a parent class to a child class (class ids).
#[derive(Serialize, Deserialize, Debug, JsonSchema, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct ChildLink {
    pub parent: u32,
    pub child: u32,
}

/// What a data export writes - the old export dialog's ticks.
#[derive(Serialize, Deserialize, Debug, JsonSchema, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum DataOutput {
    /// A row per object: `objects`.
    Objects,
    /// A row per image and class: `images`.
    Images,
    /// A row per well and class: `wells`.
    Wells,
    /// A row per image folder and class: `folders`.
    Folders,
    /// A row per pair of overlapping objects: `coloc_pairs`.
    ColocPairs,
    /// A row per measurement value: `measurements`.
    Measurements,
    /// A row per track: `tracks`.
    Tracks,
    /// A row per time point and line: `time_course`.
    TimeCourse,
    /// A row per position of a profile: `profile`.
    Profile,
    /// The plate as a heatmap, a cell per well: `plate.xlsx`.
    Plate,
    /// The wells as a heatmap, a cell per image position: `well.xlsx`.
    WellGrid,
    /// The wells as a list, a row per image position: `well_list.xlsx`.
    WellList,
    /// A heatmap per image, its area as squares: `heatmap_<image>.xlsx`.
    ImageHeatmap,
}

/// The heatmaps of a data export.
#[derive(Serialize, Deserialize, Debug, Default, JsonSchema, Clone, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct HeatmapOptions {
    #[serde(default)]
    pub colors: HeatmapColors,
    /// The image heatmap's square side in pixels (`None`: 64).
    #[serde(default)]
    pub square_size: Option<u32>,
}

fn default_aggregations() -> Vec<ExportAggregation> {
    vec![ExportAggregation::Avg]
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
    /// The first and third quartile - time courses and tracks only.
    Q1,
    Q3,
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

/// See [`DataExport::split`].
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

/// See [`DataExport::split_into`].
#[derive(Serialize, Deserialize, Debug, Default, JsonSchema, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum SplitInto {
    #[default]
    Files,
    /// One workbook, a sheet each (`Xlsx` only).
    Sheets,
}

/// A time course (the `TimeCourse` output): what a line is, and how it is
/// laid out.
#[derive(Serialize, Deserialize, Debug, Default, JsonSchema, Clone, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct TimeCourseOptions {
    #[serde(default)]
    pub lines: TimeCourseLineKind,
    #[serde(default)]
    pub layout: TimeCourseTableLayout,
    /// The values in % of the line's first time point with objects.
    #[serde(default)]
    pub relative: bool,
    /// Track lines: each track with its parents' history (divisions,
    /// fusions), else each track on its own.
    #[serde(default)]
    pub with_parents: bool,
}

/// What a line of a time course follows.
#[derive(Serialize, Deserialize, Debug, Default, JsonSchema, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum TimeCourseLineKind {
    /// The objects of a class.
    #[default]
    Class,
    Image,
    /// The images of a well - see [`DataExport::group_regex`].
    Well,
    /// The images of a folder.
    Folder,
    Track,
}

/// A time course as rows (long: a row per time point and line - for
/// scripts) and/or as a table per value (wide: a row per time point, a
/// column per line).
#[derive(Serialize, Deserialize, Debug, Default, JsonSchema, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum TimeCourseTableLayout {
    #[default]
    Long,
    Wide,
    Both,
}

/// A profile (the `Profile` output): which, and of what.
#[derive(Serialize, Deserialize, Debug, Default, JsonSchema, Clone, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct ProfileOptions {
    /// The profile measurement's key ("sholl_01/intersections").
    #[serde(default)]
    pub key: String,
    #[serde(default)]
    pub lines: ProfileLineKind,
    #[serde(default)]
    pub content: ProfileContent,
}

/// What a line of a profile is.
#[derive(Serialize, Deserialize, Debug, Default, JsonSchema, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum ProfileLineKind {
    #[default]
    Class,
    Image,
    Well,
    /// All objects one line.
    All,
}

/// A profile's lines (n, mean, SD, median, quartiles per position) or each
/// object's values.
#[derive(Serialize, Deserialize, Debug, Default, JsonSchema, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum ProfileContent {
    #[default]
    Lines,
    Objects,
}

/// Which tracks the `Tracks` output has, and the unit of their speed.
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

/// The colours of a plate layout's heatmap, low to high.
#[derive(Serialize, Deserialize, Debug, Default, JsonSchema, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum HeatmapColors {
    /// Excel's red - yellow - green.
    #[default]
    Excel,
    Viridis,
    Plasma,
    Inferno,
    Cividis,
    Coolwarm,
    RedBlue,
    YlGnBu,
    Haline,
    Algae,
    Thermal,
}

/// Which z-planes and time points an entry writes: each a range;
/// `None`: all of them.
#[derive(Serialize, Deserialize, Debug, Default, JsonSchema, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct Planes {
    #[serde(default)]
    pub z: Option<PlaneRange>,
    #[serde(default)]
    pub t: Option<PlaneRange>,
}

/// Planes `from` to `to`, both included, counted from 0.
#[derive(Serialize, Deserialize, Debug, JsonSchema, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct PlaneRange {
    pub from: u32,
    pub to: u32,
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
    #[serde(default)]
    pub planes: Planes,
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
            "runAfterAnalysis": true,
            "tableFormat": 1,
            "entries": [
                { "kind": "DATA", "outputs": ["OBJECTS", "IMAGES", "PLATE"],
                  "format": "CSV", "split": "IMAGE",
                  "classes": [3, 5], "columns": ["count", "area_nm2"],
                  "aggregations": ["AVG", "MEDIAN"],
                  "heatmap": { "colors": "VIRIDIS" } },
                { "kind": "DATA", "name": "tracks", "outputs": ["TRACKS"] },
                { "kind": "DRAWN_IMAGES", "classes": [3], "style": "OUTLINE" },
                { "kind": "SCRIPT", "language": "R", "file": "scripts/uptake_stats.R", "output": "stats" }
            ]
        }))
        .expect("the documented example");

        assert_eq!(export.entries.len(), 4);
        let ExportEntry::Data(data) = &export.entries[0] else {
            panic!("a data export");
        };
        assert_eq!(
            data.outputs,
            vec![DataOutput::Objects, DataOutput::Images, DataOutput::Plate]
        );
        assert_eq!(data.split, TableSplit::Image);
        assert_eq!(
            data.aggregations,
            vec![ExportAggregation::Avg, ExportAggregation::Median]
        );
        assert_eq!(data.images, ImageScope::Included);
        assert_eq!(data.heatmap.colors, HeatmapColors::Viridis);
        let ExportEntry::Data(tracks) = &export.entries[1] else {
            panic!("a data export");
        };
        assert_eq!(
            tracks.aggregations,
            vec![ExportAggregation::Avg],
            "the default"
        );
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
        assert!(!export.run_after_analysis);
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
            run_after_analysis: true,
            table_format: 1,
            entries: vec![
                ExportEntry::Data(DataExport {
                    name: "ctrl".into(),
                    outputs: vec![DataOutput::Tracks, DataOutput::WellGrid],
                    format: TableFileFormat::Xlsx,
                    classes: Some(vec![2]),
                    columns: Some(vec!["count".into()]),
                    aggregations: vec![ExportAggregation::Median],
                    images: ImageScope::All,
                    only_images: Some(vec!["a/b.tif".into()]),
                    group_regex: Some("^(.+)_".into()),
                    planes: Planes::default(),
                    empty_classes: true,
                    transposed: false,
                    children: vec![ChildLink {
                        parent: 1,
                        child: 2,
                    }],
                    child_aggregations: vec![ExportAggregation::Max],
                    parents: vec![1],
                    split: TableSplit::Class,
                    split_into: SplitInto::Sheets,
                    tracks: Some(TrackOptions {
                        min_points: 3,
                        from_start: true,
                        no_edge: true,
                        speed_unit: SpeedUnit::UmS,
                    }),
                    time_course: None,
                    profile: None,
                    heatmap: HeatmapOptions {
                        colors: HeatmapColors::Thermal,
                        square_size: Some(128),
                    },
                }),
                ExportEntry::DrawnImages(DrawnImagesExport {
                    name: "pictures".into(),
                    classes: None,
                    style: DrawnStyle::Box,
                    brightness: DrawnBrightness::Auto,
                    max_side: 1024,
                    images: ImageScope::Included,
                    planes: Planes {
                        z: Some(PlaneRange { from: 0, to: 2 }),
                        t: None,
                    },
                }),
                ExportEntry::Script(ScriptExport {
                    language: ScriptLanguage::Python,
                    file: "scripts/plot.py".into(),
                    output: String::new(),
                }),
            ],
        };
        let json = serde_json::to_value(&export).unwrap();
        assert_eq!(json["entries"][0]["kind"], "DATA");
        assert_eq!(json["entries"][0]["outputs"][1], "WELL_GRID");
        assert_eq!(json["entries"][0]["heatmap"]["squareSize"], 128);
        assert_eq!(json["entries"][0]["children"][0]["parent"], 1);
        assert_eq!(json["entries"][1]["kind"], "DRAWN_IMAGES");
        assert_eq!(json["entries"][1]["planes"]["z"]["to"], 2);
        assert_eq!(json["output"]["location"], "FIXED_FOLDER");
        let back: ExportSettings = serde_json::from_value(json).unwrap();
        assert_eq!(back, export);
    }

    /// What a newer version adds reads in this one: a kind it doesn't
    /// know, or a value inside an entry (an output, an aggregation), is
    /// kept as an unknown entry - the others read as ever, and saving
    /// writes it back exactly as it was. A new key is ignored.
    #[test]
    fn entries_of_a_newer_version_are_kept_as_they_are() {
        let file = json!({
            "id": 1, "name": "x", "entries": [
                { "kind": "CHART", "file": "plot.svg", "size": [800, 600] },
                { "kind": "DATA", "outputs": ["IMAGES", "VOLUMES"] },
                { "kind": "DATA", "outputs": ["IMAGES"], "aggregations": ["AVG", "MODE"] },
                { "kind": "DATA", "outputs": ["IMAGES"], "newKey": 1 },
                { "kind": "SCRIPT", "language": "JULIA", "file": "scripts/a.jl" }
            ]
        });
        let export: ExportSettings = serde_json::from_value(file.clone()).unwrap();
        let unknown: Vec<bool> = export
            .entries
            .iter()
            .map(|entry| matches!(entry, ExportEntry::Unknown(_)))
            .collect();
        assert_eq!(unknown, [true, true, true, false, true]);

        let saved = serde_json::to_value(&export).unwrap();
        for i in [0, 1, 2, 4] {
            assert_eq!(
                saved["entries"][i], file["entries"][i],
                "entry {i} unchanged"
            );
        }
    }

    /// A new value of an export's own settings is not tolerated: that
    /// takes a schema version (see the head of this file).
    #[test]
    fn an_unknown_value_of_the_export_itself_is_an_error() {
        let read = serde_json::from_value::<ExportSettings>(json!({
            "id": 1, "name": "x", "lengthUnit": "PX", "entries": []
        }));
        assert!(read.is_err());
    }
}
