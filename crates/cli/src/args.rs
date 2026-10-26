//! # args
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-06-26
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.

use clap::{Args, Subcommand, ValueEnum};
use std::path::PathBuf;

#[derive(Subcommand)]
pub enum CliCommand {
    /// Run the project's enabled pipelines over its images and write a new results database
    Analyze(AnalyzeArgs),
    /// Print a project's images, classes and pipelines without running anything
    ProjectInfo(ProjectInfoArgs),
    /// Check that every image referenced by a project can be found on disk
    Validate(ValidateArgs),
    /// Export a results database to CSV or XLSX
    Export(ExportArgs),
    /// Print a quick summary and a page of rows from a results database
    View(ViewArgs),
    /// List the column ids available for --group-by in a results database
    Columns(ColumnsArgs),
    /// Train a pixel or object classifier from a project's labeled objects and save it under models/
    TrainClassifier(TrainClassifierArgs),
    /// List the analyses on the --remote server: the running one, and recently finished ones
    Jobs,
    /// Follow an analysis on the --remote server again, e.g. after the connection dropped
    Attach(AttachArgs),
}

#[derive(Args)]
pub struct AttachArgs {
    /// Id of the analysis, as `jobs` lists it. Default: the running one.
    #[arg(long)]
    pub job: Option<String>,
}

#[derive(Args)]
pub struct AnalyzeArgs {
    /// Project file (.evaproj) to analyze
    #[arg(long)]
    pub project: PathBuf,

    /// Directory of images to scan and use as the project's image root before running.
    /// If omitted, the project's already-saved image list is used as-is.
    #[arg(long)]
    pub images: Option<PathBuf>,

    /// Number of images to process in parallel (default: number of CPUs minus 1)
    #[arg(long)]
    pub threads: Option<usize>,

    /// Name for this analysis run, used for the results subfolder and the
    /// .evadb filename. If omitted, a random two-word name is generated.
    #[arg(long)]
    pub job_name: Option<String>,
}

#[derive(Args)]
pub struct TrainClassifierArgs {
    /// Project file (.evaproj) to train from - every object with an assigned
    /// class, across every image already in the project, is used as training
    /// data (no per-run selection, matching how `analyze` runs over every
    /// image in the project by default)
    #[arg(long)]
    pub project: PathBuf,

    /// Path to a JSON file describing the model: metadata, backend
    /// hyperparameters (Random Forest / KNN / MLP), and the feature
    /// spec + class labels to train (an `AiLearningSettings` document)
    #[arg(long)]
    pub settings: PathBuf,

    /// Name for the saved model file, under <project-dir>/models/. If
    /// omitted, uses the name from --settings's metadata.
    #[arg(long)]
    pub model_name: Option<String>,

    /// Image channel to read (pixel classifiers only)
    #[arg(long, default_value_t = 0)]
    pub channel: i32,

    /// How the analyses using the model handle z-stacks: each annotation is
    /// trained on its own plane (single-/all-stacks) or on the projection
    /// of its time frame
    #[arg(long, value_enum, default_value = "single-stack")]
    pub z_stack_handling: ZStackHandlingArg,
}

#[derive(Copy, Clone, ValueEnum)]
pub enum ZStackHandlingArg {
    SingleStack,
    AllStacks,
    MaxIntensity,
    MinIntensity,
    AvgIntensity,
    SumIntensity,
    TakeTheMiddle,
}

#[derive(Args)]
pub struct ProjectInfoArgs {
    /// Project file (.evaproj) to inspect
    #[arg(long)]
    pub project: PathBuf,

    /// Print machine-readable JSON instead of a human-readable summary
    #[arg(long)]
    pub json: bool,
}

#[derive(Args)]
pub struct ValidateArgs {
    /// Project file (.evaproj) to validate
    #[arg(long)]
    pub project: PathBuf,
}

#[derive(Args)]
pub struct ExportArgs {
    #[command(subcommand)]
    pub command: ExportCommand,
}

#[derive(Subcommand)]
pub enum ExportCommand {
    /// Export rows as a CSV file
    Csv(TableExportArgs),
    /// Export rows as an XLSX workbook
    Xlsx(TableExportArgs),
    /// Export the raw `objects` table as a single Parquet file
    ///
    /// Unlike `csv`/`xlsx`, this is an unfiltered, every-column dump of the
    /// database's `objects` table (via DuckDB's own `COPY ... TO ...
    /// (FORMAT parquet)`) — there's no column selection or image/class/z/t
    /// filtering to apply, so it takes a plainer set of arguments than
    /// `TableExportArgs`.
    Parquet(ParquetExportArgs),
    // Chart image export (histogram/scatter/heatmap PNGs) isn't wired up
    // yet: the current results backend (ResultCharts) only computes chart
    // *data* (bins/points/box stats) — actual pixel rendering only exists
    // in the GUI's Charts tab today, and the old "heatmap" here (a
    // cross-image object-centroid density map) has no equivalent in the
    // current API (ResultsGenerator::get_image_heatmap is per-image only).
    // Revisit once there's a real plotting path (`plotters` is already a
    // declared-but-unused dependency of evanalyzer_app) or a decision to
    // just export chart data as CSV/JSON instead of a rendered image.
}

#[derive(Args)]
pub struct TableExportArgs {
    /// Results database (.evadb) produced by `analyze`
    #[arg(long)]
    pub db: PathBuf,

    /// Output file path
    #[arg(long)]
    pub out: PathBuf,

    #[command(flatten)]
    pub filter: FilterArgs,

    #[command(flatten)]
    pub group: GroupArgs,

    #[command(flatten)]
    pub tracks: TrackArgs,

    /// One row per measurement value (image, plane, object or region, step,
    /// measurement, value) - the measurement steps' values, like the
    /// results' `measurements` table; --columns picks measurements by key
    /// (`measurement:<key>`), none: all
    #[arg(long, conflicts_with = "tracks")]
    pub measurements: bool,

    /// The project whose results view --view names (its .evaproj)
    #[arg(long, requires = "view")]
    pub project: Option<PathBuf>,

    /// Export as this List view of --project shows the results: its
    /// columns, classes, images and rows (a row per object, or per image
    /// with its aggregations). --images, --classes and --group-by win over
    /// it
    #[arg(long, requires = "project", conflicts_with_all = ["tracks", "measurements"])]
    pub view: Option<String>,
}

/// `--tracks`: a row per track instead of per object.
#[derive(Args, Default)]
pub struct TrackArgs {
    /// A row per track: when it was seen, how far, fast and straight it
    /// moved, the --agg statistics (default avg) of every numeric column
    /// over its time points
    #[arg(long)]
    pub tracks: bool,

    /// --tracks: only tracks with at least this many time points
    #[arg(long, default_value_t = 1, requires = "tracks")]
    pub min_points: u32,

    /// --tracks: only tracks there from the first time point
    #[arg(long, requires = "tracks")]
    pub from_start: bool,

    /// --tracks: leave out tracks with an object touching the image edge
    #[arg(long, requires = "tracks")]
    pub no_edge: bool,

    /// --tracks: the speed's unit (px-frame without a pixel size or frame
    /// interval)
    #[arg(long, value_enum, default_value = "um-min", requires = "tracks")]
    pub speed_unit: SpeedUnitKind,
}

#[derive(Copy, Clone, Default, ValueEnum)]
pub enum SpeedUnitKind {
    PxFrame,
    #[default]
    UmMin,
    UmS,
}

#[derive(Args)]
pub struct ParquetExportArgs {
    /// Results database (.evadb) produced by `analyze`
    #[arg(long)]
    pub db: PathBuf,

    /// Output file path
    #[arg(long)]
    pub out: PathBuf,
}

#[derive(Args, Default)]
pub struct FilterArgs {
    /// Restrict to these image names (repeatable)
    #[arg(long = "image")]
    pub images: Vec<String>,

    /// Restrict to these class names (repeatable)
    #[arg(long = "class")]
    pub classes: Vec<String>,

    /// Transponod output table
    #[arg(long)]
    pub transpond: Option<bool>,

    /// Only these time points: one ("3") or a range ("2-5"), 0-based.
    /// Default: all
    #[arg(long = "t", value_parser = parse_plane_range)]
    pub t_stacks: Option<std::range::Range<u32>>,

    /// Only these z-planes: one ("0") or a range ("0-4"), 0-based.
    /// Default: all
    #[arg(long = "z", value_parser = parse_plane_range)]
    pub z_stacks: Option<std::range::Range<u32>>,
}

/// `--t`/`--z`: "3" or "2-5" (both ends included) as a range.
pub(crate) fn parse_plane_range(text: &str) -> Result<std::range::Range<u32>, String> {
    let (start, end) = text.split_once('-').unwrap_or((text, text));
    let number = |value: &str| {
        value
            .trim()
            .parse::<u32>()
            .map_err(|_| format!("\"{text}\" isn't a number or a range like 2-5"))
    };
    let (start, end) = (number(start)?, number(end)?);
    if end < start {
        return Err(format!("\"{text}\": the range ends before it starts"));
    }
    Ok(std::range::Range {
        start,
        end: end + 1,
    })
}

#[derive(Args, Default)]
pub struct GroupArgs {
    /// Aggregate rows instead of exporting one row per object
    #[arg(long, value_enum)]
    pub group_by: Option<GroupByKind>,

    /// Regex on the image name for --group-by group / regex: its first
    /// capture group is the group
    #[arg(long)]
    pub group_regex: Option<String>,

    /// Statistic(s) of every numeric column when grouping (comma-separated,
    /// e.g. "avg,stdev,median"). Default: avg; for a time course (class,
    /// well, group, folder, regex, track) avg,stdev. q1/q3 only for a time
    /// course
    #[arg(long, value_enum, value_delimiter = ',')]
    pub agg: Vec<AggKind>,

    /// Time course values relative to each line's first time point (t0: in
    /// % of it)
    #[arg(long, value_enum)]
    pub normalize: Option<NormalizeKind>,

    /// Time course tables: long (a row per time point × line), wide (per
    /// value and statistic: a row per time point, a column per line; CSV:
    /// one file each, next to --out as <name>_<value>.csv) or both
    #[arg(long, value_enum, default_value = "long")]
    pub layout: LayoutKind,

    /// Additionally split each group by object class
    #[arg(long)]
    pub group_by_class: bool,
}

#[derive(Copy, Clone, ValueEnum)]
pub enum GroupByKind {
    /// A row per image and class, per time point
    Image,
    /// Time course: a line per class, every image pooled
    Class,
    /// Time course: a line per well (from the image names)
    Well,
    /// Time course: a line per group - --group-regex's first capture group
    Group,
    /// Time course: a line per folder
    Folder,
    /// Time course: same as group
    Regex,
    /// Time course: a line per track
    Track,
}

#[derive(Copy, Clone, ValueEnum)]
pub enum NormalizeKind {
    None,
    T0,
}

#[derive(Copy, Clone, Default, ValueEnum)]
pub enum LayoutKind {
    #[default]
    Long,
    Wide,
    Both,
}

#[derive(Copy, Clone, ValueEnum)]
pub enum AggKind {
    Min,
    Max,
    Avg,
    Median,
    Stdev,
    Sum,
    Q1,
    Q3,
}

#[derive(Args)]
pub struct ViewArgs {
    /// Results database (.evadb) produced by `analyze`
    #[arg(long)]
    pub db: PathBuf,

    /// Zero-based page index
    #[arg(long, default_value_t = 0)]
    pub page: usize,

    /// Rows per page
    #[arg(long, default_value_t = 25)]
    pub limit: usize,

    /// Also show per-channel intensity columns
    #[arg(long)]
    pub channels: bool,

    /// Print machine-readable JSON instead of a human-readable table
    #[arg(long)]
    pub json: bool,

    #[command(flatten)]
    pub filter: FilterArgs,

    #[command(flatten)]
    pub tracks: TrackArgs,
}

#[derive(Args)]
pub struct ColumnsArgs {
    /// Results database (.evadb) produced by `analyze`
    #[arg(long)]
    pub db: PathBuf,

    /// Print machine-readable JSON instead of a human-readable table
    #[arg(long)]
    pub json: bool,
}
