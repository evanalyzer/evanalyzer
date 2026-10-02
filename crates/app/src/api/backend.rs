//! The backend interface itself: what a front end can ask for, and the
//! requests it sends. Implemented by `backend::local::LocalBackend` (this
//! process) and `backend::remote::RemoteBackend` (a server).

use super::{
    BoxplotFilter, BoxplotResult, ColumnEntry, DatabaseResult, FileSystem, GroupedByImageFilter,
    HistogramFilter, HistogramResult, ImageChannel, ImageEntry, ImageHeatmapFilter, ImageMeta,
    ListFilter, PixelTrainingParams, PlateFilter, PreviewRequest, ResultExport, RunningJob,
    RunningTraining, ScatterFilter, ScatterResult, StartPreviewError, StartTrainingError, View,
    WellFilter,
};
use evanalyzer_cfg::core_types::{ImageTile, InternalErrors, ZProjection};
use evanalyzer_cfg::settings::ai_learning_settings::AiLearningSettings;
use evanalyzer_cfg::settings::classification_settings::Class;
use evanalyzer_cfg::settings::project_settings::ProjectSettings;
use serde::{Deserialize, Serialize};
use std::ops::RangeInclusive;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::AtomicBool;

pub type ExportProgressFn<'a> = &'a mut dyn FnMut(&str, usize, usize);

pub trait Backend: Send + Sync {
    /// Starts a full analysis run over every image in the project, writing a
    /// results database under `<project_path>/results`.
    fn start_analysis(&self, req: AnalysisRequest) -> Result<RunningJob, InternalErrors>;

    /// Starts a preview run restricted to the tiles visible in the viewport.
    fn start_preview(&self, req: PreviewRequest) -> Result<RunningJob, StartPreviewError>;

    /// Starts training a classifier on the project's labeled data.
    fn start_training(&self, req: TrainingRequest) -> Result<RunningTraining, StartTrainingError>;

    /// Opens an image for reading its metadata and tiles. Opening is
    /// expensive (a full format parse) - callers cache the result.
    fn open_image(&self, path: &Path) -> Result<Arc<dyn ImageSource>, InternalErrors>;

    /// Opens a results database (`.evadb`) for querying - the results
    /// window and CLI `view`/`export` only go through this.
    fn open_results(&self, path: &Path) -> Result<Arc<dyn ResultsSource>, InternalErrors>;

    /// Reads just an image's metadata - much cheaper than [`Self::open_image`]
    /// (one reader instead of a pool), for scanning folders of images.
    fn read_image_meta(&self, path: &Path) -> Result<ImageMeta, InternalErrors>;

    /// Where pipeline/project templates live on the backend's machine:
    /// the user's own templates (new ones are saved there) and the ones
    /// shipped with the application.
    fn template_folders(&self) -> Result<TemplateFolders, InternalErrors>;

    /// Files on the backend's machine - what the file browser shows, and
    /// where projects, templates and models are read from and saved to.
    fn files(&self) -> &dyn FileSystem;

    /// Whether work and files are on another machine - the UI says so,
    /// e.g. in the file browser.
    fn is_remote(&self) -> bool {
        false
    }

    /// Human-readable location of this backend, for logs and status text
    /// ("local", "ws://host:7400").
    fn description(&self) -> String;

    /// Whether the backend can still be reached. A remote backend turns
    /// `false` for good once its connection drops; the UI then warns.
    fn is_connected(&self) -> bool {
        true
    }

    /// Who is logged in on a remote server (`--user`), if anyone.
    fn user(&self) -> Option<String> {
        None
    }
}

/// An opened image, readable tile by tile.
pub trait ImageSource: Send + Sync {
    /// The image's metadata (series, pyramid levels, channels).
    fn meta(&self) -> &ImageMeta;

    /// Reads one tile of every channel of `req.series`.
    fn read_tile(&self, req: &TileRequest) -> Result<Vec<ImageChannel>, InternalErrors>;
}

pub trait ResultsSource: Send + Sync {
    fn get_object_list(&self, filter: &ListFilter) -> Result<DatabaseResult, InternalErrors>;
    fn get_grouped_by_image(
        &self,
        filter: &GroupedByImageFilter,
    ) -> Result<DatabaseResult, InternalErrors>;
    fn get_group_by_plate(
        &self,
        filter: &PlateFilter,
        view: &View,
    ) -> Result<DatabaseResult, InternalErrors>;
    fn get_group_by_well(
        &self,
        filter: &WellFilter,
        view: &View,
    ) -> Result<DatabaseResult, InternalErrors>;
    fn get_image_heatmap(
        &self,
        filter: &ImageHeatmapFilter,
        view: &View,
    ) -> Result<DatabaseResult, InternalErrors>;
    fn get_images(&self) -> Result<Vec<ImageEntry>, InternalErrors>;
    /// Marks an image as excluded from (`disable`) or included in the
    /// results - persisted in the database.
    fn enable_image(&self, image_rel_path: &str, disable: bool) -> Result<(), InternalErrors>;
    fn get_object_classes(&self) -> Result<Vec<Class>, InternalErrors>;
    fn get_available_columns(&self) -> Result<Vec<ColumnEntry>, InternalErrors>;
    fn get_nr_of_z_stacks(&self) -> u32;
    fn get_nr_of_t_stacks(&self) -> u32;
    fn boxplot(&self, filter: &BoxplotFilter) -> Result<BoxplotResult, InternalErrors>;
    fn histogram(&self, filter: &HistogramFilter) -> Result<HistogramResult, InternalErrors>;
    fn scatter(&self, filter: &ScatterFilter) -> Result<ScatterResult, InternalErrors>;
    /// Writes the export's files into `export.output_dir` (on the backend's
    /// machine). Blocks; stops early with `Cancelled` once `cancel` is set.
    fn export(
        &self,
        export: &ResultExport,
        cancel: &AtomicBool,
        on_progress: ExportProgressFn,
    ) -> Result<(), InternalErrors>;
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TemplateFolders {
    /// User-created templates; saving a template defaults to this folder.
    pub user: PathBuf,
    /// Templates shipped next to the application binary (read-only).
    pub bundled: PathBuf,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct AnalysisRequest {
    pub settings: ProjectSettings,
    /// Directory the project file lives in; results go below it.
    pub project_path: PathBuf,
    pub job_name: Option<String>,
    /// Worker count override; `None` picks it from the backend host's CPU
    /// cores and RAM.
    pub threads: Option<usize>,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct TrainingRequest {
    /// The project whose labeled objects/images are the training data.
    pub project: ProjectSettings,
    pub settings: AiLearningSettings,
    pub pixel_params: PixelTrainingParams,
}

/// Which tile to read, at which pyramid level and plane.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TileRequest {
    pub series: i32,
    pub resolution_idx: i32,
    pub z_projection: ZProjection,
    pub z_range: Option<RangeInclusive<i32>>,
    pub t_stack: i32,
    pub tile: ImageTile,
}
