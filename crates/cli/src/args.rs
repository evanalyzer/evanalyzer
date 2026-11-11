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
    /// Run a project's pipelines over its images into a new results database
    Analyze(AnalyzeArgs),
    /// Create, inspect, check and cite project files
    #[command(subcommand)]
    Project(ProjectSubCommand),
    /// View and export results databases
    #[command(subcommand)]
    Results(ResultsSubCommand),
    /// Analyses on the --remote server
    #[command(subcommand)]
    Jobs(JobsSubCommand),
    /// Train a pixel or object classifier from a project's labeled objects
    ///
    /// The model is saved under the project's `models/` folder.
    TrainClassifier(TrainClassifierArgs),
}

#[derive(Subcommand)]
pub enum ProjectSubCommand {
    /// Create a new empty project
    New(ProjectNewArgs),
    /// Print a project's images, classes and pipelines
    Info(ProjectInfoArgs),
    /// Check a project as a run would
    ///
    /// Its images on disk, its pipelines and modules as the editor checks
    /// them, and the channels the pipelines read against the images'.
    /// Warnings are printed; errors and missing images fail.
    Validate(ValidateArgs),
    /// Write a project's methods paragraph and citations (Markdown)
    Cite(CiteArgs),
}

#[derive(Subcommand)]
pub enum ResultsSubCommand {
    /// Export a results database to CSV or XLSX
    Export(ExportArgs),
    /// Print a summary and a page of rows of a results database
    View(ViewArgs),
    /// List a results database's column ids (for --group-by)
    Columns(ColumnsArgs),
}

#[derive(Subcommand)]
pub enum JobsSubCommand {
    /// List the running and recently finished analyses
    List,
    /// Follow an analysis again, e.g. after the connection dropped
    Attach(AttachArgs),
}

#[derive(Args)]
pub struct AttachArgs {
    /// Id of the analysis, as `jobs list` shows it (default: the running one)
    #[arg(long)]
    pub job: Option<String>,
}

#[derive(Args)]
pub struct AnalyzeArgs {
    /// Project file (.evaproj) to analyze
    #[arg(long)]
    pub project: PathBuf,

    /// Image folder to scan and use as the project's image root
    ///
    /// If omitted, the project's saved image list is used as it is.
    #[arg(long)]
    pub images: Option<PathBuf>,

    /// Number of images to process in parallel (default: number of CPUs minus 1)
    #[arg(long)]
    pub threads: Option<usize>,

    /// Name of the run: its results folder and .evadb file
    ///
    /// If omitted, a random two-word name is generated.
    #[arg(long)]
    pub job_name: Option<String>,

    /// Change a step's setting for this run (repeatable)
    ///
    /// E.g. `--set "Varroa detection.9.minSize=100"`: the pipeline's name,
    /// the step's `id` in the project file, the setting's name there (or as
    /// in the editor, `min_size`) and the value. The project file stays as
    /// it is; the run's copy of it next to the results records the values
    /// it ran with.
    #[arg(long = "set", value_name = "SETTING=VALUE")]
    pub set: Vec<String>,
}

#[derive(Args)]
pub struct TrainClassifierArgs {
    /// Project file (.evaproj) to train from
    ///
    /// Every object with an assigned class, in every image of the project,
    /// is training data - like `analyze`, which runs over every image.
    #[arg(long)]
    pub project: PathBuf,

    /// JSON file describing the model
    ///
    /// An `AiLearningSettings` document: metadata, backend hyperparameters
    /// (Random Forest / KNN / MLP), the features and the class labels to
    /// train.
    #[arg(long)]
    pub settings: PathBuf,

    /// Name of the saved model file under <project-dir>/models/
    ///
    /// If omitted, the name from --settings's metadata.
    #[arg(long)]
    pub model_name: Option<String>,

    /// Image channel to read (pixel classifiers only)
    #[arg(long, default_value_t = 0)]
    pub channel: i32,

    /// How the analyses using the model handle z-stacks
    ///
    /// Each annotation is trained on its own plane (single-/all-stacks) or
    /// on the projection of its time frame.
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
pub struct ProjectNewArgs {
    /// Project file (.evaproj) to create
    #[arg(long)]
    pub project: PathBuf,
}

#[derive(Args)]
pub struct ValidateArgs {
    /// Project file (.evaproj) to validate
    #[arg(long)]
    pub project: PathBuf,

    /// Check the project with a step's setting changed (repeatable)
    ///
    /// As `analyze --set`, e.g. `--set "Varroa detection.9.minSize=100"`:
    /// checks the project as `analyze` with these overrides would run it.
    /// The project file stays as it is.
    #[arg(long = "set", value_name = "SETTING=VALUE")]
    pub set: Vec<String>,
}

#[derive(Args)]
pub struct CiteArgs {
    /// Project file (.evaproj) to cite
    #[arg(long)]
    pub project: PathBuf,

    /// Markdown file to write (default: `citation.md` next to the project)
    #[arg(long)]
    pub out: Option<PathBuf>,
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

    /// One row per measurement value
    ///
    /// Image, plane, object or region, step, measurement, value - the
    /// measurement steps' values, like the results' `measurements` table.
    /// --columns picks measurements by key (`measurement:<key>`), none: all.
    #[arg(long, conflicts_with = "tracks")]
    pub measurements: bool,

    /// The project whose results view --view names (its .evaproj)
    #[arg(long, requires = "view")]
    pub project: Option<PathBuf>,

    /// Export as this List view of --project shows the results
    ///
    /// Its columns, classes, images and rows (a row per object, or per
    /// image with its aggregations). --images, --classes and --group-by win
    /// over it.
    #[arg(long, requires = "project", conflicts_with_all = ["tracks", "measurements"])]
    pub view: Option<String>,
}

/// `--tracks`: a row per track instead of per object.
#[derive(Args, Default)]
pub struct TrackArgs {
    /// A row per track instead of per object
    ///
    /// When it was seen, how far, fast and straight it moved, and the --agg
    /// statistics (default avg) of every numeric column over its time
    /// points.
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

    /// --tracks: the speed's unit
    ///
    /// px-frame for results without a pixel size or frame interval.
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

    /// Transpose the output table
    #[arg(long)]
    pub transpond: Option<bool>,

    /// Only these time points: one ("3") or a range ("2-5"), 0-based
    ///
    /// Default: all.
    #[arg(long = "t", value_parser = parse_plane_range)]
    pub t_stacks: Option<std::range::Range<u32>>,

    /// Only these z-planes: one ("0") or a range ("0-4"), 0-based
    ///
    /// Default: all.
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

    /// Regex on the image name for --group-by group / regex
    ///
    /// Its first capture group is the group.
    #[arg(long)]
    pub group_regex: Option<String>,

    /// Statistics of every numeric column when grouping, e.g. avg,stdev
    ///
    /// Comma-separated. Default: avg; for a time course (class, well,
    /// group, folder, regex, track) avg,stdev. q1/q3 only for a time
    /// course.
    #[arg(long, value_enum, value_delimiter = ',')]
    pub agg: Vec<AggKind>,

    /// Time course values relative to each line's first time point
    ///
    /// t0: in % of it.
    #[arg(long, value_enum)]
    pub normalize: Option<NormalizeKind>,

    /// Time course table layout
    ///
    /// long: a row per time point × line. wide: per value and statistic, a
    /// row per time point and a column per line (CSV: one file each, next
    /// to --out as <name>_<value>.csv).
    #[arg(long, value_enum, default_value = "long")]
    pub layout: LayoutKind,

    /// Time course per track, with each track's parents' history
    ///
    /// A daughter's line starts with its mother's track, a fused track has
    /// a line per parent (results with divisions or fusions).
    #[arg(long)]
    pub with_parents: bool,

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
