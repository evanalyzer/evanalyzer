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
    /// Export a results database to CSV, XLSX, or a chart image
    Export(ExportArgs),
    /// Print a quick summary and a page of rows from a results database
    View(ViewArgs),
    /// List the column ids available for --group-by / chart axes in a results database
    Columns(ColumnsArgs),
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
    /// Render a chart (histogram, scatter or heatmap) to a PNG file
    Chart(ChartArgs),
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
pub struct ChartArgs {
    #[command(subcommand)]
    pub kind: ChartKind,
}

#[derive(Subcommand)]
pub enum ChartKind {
    /// Bucket one numeric column into a histogram
    Histogram(HistogramArgs),
    /// Plot two numeric columns against each other
    Scatter(ScatterArgs),
    /// Bin object centroids into a spatial grid
    Heatmap(HeatmapArgs),
}

#[derive(Args)]
pub struct HistogramArgs {
    /// Results database (.evadb) produced by `analyze`
    #[arg(long)]
    pub db: PathBuf,

    /// Output PNG path
    #[arg(long)]
    pub out: PathBuf,

    /// Column id to plot, e.g. area_px, circularity, ch0_avg_bit
    /// (see `evanalyzer cli columns --db <file>`)
    #[arg(long)]
    pub column: String,

    #[arg(long, default_value_t = 20)]
    pub buckets: usize,

    /// Space buckets equally in log space (use for right-skewed data such as area)
    #[arg(long)]
    pub log_scale: bool,

    /// Overlay one semi-transparent histogram per group instead of one plain histogram
    #[arg(long, value_enum, default_value = "none")]
    pub color_by: ColorByKind,

    #[arg(long, default_value_t = 1000)]
    pub width: u32,

    #[arg(long, default_value_t = 700)]
    pub height: u32,

    #[command(flatten)]
    pub filter: FilterArgs,
}

#[derive(Args)]
pub struct ScatterArgs {
    /// Results database (.evadb) produced by `analyze`
    #[arg(long)]
    pub db: PathBuf,

    /// Output PNG path
    #[arg(long)]
    pub out: PathBuf,

    /// Column id for the X axis
    #[arg(long)]
    pub x: String,

    /// Column id for the Y axis
    #[arg(long)]
    pub y: String,

    #[arg(long, value_enum, default_value = "none")]
    pub color_by: ColorByKind,

    /// Cap the number of plotted points (0 = no cap)
    #[arg(long, default_value_t = 5000)]
    pub max_points: usize,

    #[arg(long, default_value_t = 1000)]
    pub width: u32,

    #[arg(long, default_value_t = 700)]
    pub height: u32,

    #[command(flatten)]
    pub filter: FilterArgs,
}

#[derive(Copy, Clone, ValueEnum)]
pub enum ColorByKind {
    None,
    Class,
    Colocalized,
}

#[derive(Args)]
pub struct HeatmapArgs {
    /// Results database (.evadb) produced by `analyze`
    #[arg(long)]
    pub db: PathBuf,

    /// Output PNG path
    #[arg(long)]
    pub out: PathBuf,

    /// "count", or a numeric column id to average per cell, e.g. area_px
    #[arg(long)]
    pub metric: String,

    /// Cell size in image pixels (square cells)
    #[arg(long, default_value_t = 256.0)]
    pub cell_size: f64,

    /// Color scheme: viridis, magma, plasma, or grayscale
    #[arg(long, default_value = "viridis")]
    pub color_scheme: String,

    /// Pin the color scale's lower bound instead of computing it from this
    /// heatmap's own data (0). Must be given together with --range-max — the
    /// point is to render multiple heatmaps under an identical color mapping
    /// so they're directly comparable.
    #[arg(long, requires = "range_max")]
    pub range_min: Option<f64>,

    /// Pin the color scale's upper bound instead of computing it from this
    /// heatmap's own data (its max cell value). Must be given together with
    /// --range-min.
    #[arg(long, requires = "range_min")]
    pub range_max: Option<f64>,

    #[arg(long, default_value_t = 1000)]
    pub width: u32,

    #[arg(long, default_value_t = 700)]
    pub height: u32,

    #[command(flatten)]
    pub filter: FilterArgs,
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
