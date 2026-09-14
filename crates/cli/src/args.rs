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

    /// Time frame to read, alongside --channel (pixel classifiers only)
    #[arg(long, default_value_t = 0)]
    pub t_stack: i32,

    /// How to handle z-stacks (pixel classifiers only)
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

    /// Restrict to colocalized (true) or non-colocalized (false) ROIs only
    #[arg(long)]
    pub colocalized: Option<bool>,
}

#[derive(Args, Default)]
pub struct GroupArgs {
    /// Aggregate rows instead of exporting one row per object
    #[arg(long, value_enum)]
    pub group_by: Option<GroupByKind>,

    /// Regex pattern used when --group-by regex; the first capture group (or the
    /// whole match if there is none) becomes the group key
    #[arg(long)]
    pub group_regex: Option<String>,

    /// Aggregate function(s) applied to every numeric column when grouping
    /// (comma-separated, e.g. "min,max,avg")
    #[arg(long, value_enum, value_delimiter = ',', default_value = "avg")]
    pub agg: Vec<AggKind>,

    /// Additionally split each group into a colocalizing / non-colocalizing row
    #[arg(long)]
    pub split_colocalized: bool,

    /// Additionally split each group by object class
    #[arg(long)]
    pub group_by_class: bool,
}

#[derive(Copy, Clone, ValueEnum)]
pub enum GroupByKind {
    Image,
    Folder,
    Regex,
}

#[derive(Copy, Clone, ValueEnum)]
pub enum AggKind {
    Min,
    Max,
    Avg,
    Median,
    Stdev,
    Sum,
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
