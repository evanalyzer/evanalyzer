//! Where compute runs. Front ends start analysis/preview/training runs and
//! read image tiles only through [`Backend`], held by
//! [`AppHandle`](crate::AppHandle) - so whether that work happens in this
//! process ([`LocalBackend`]) or on a server (`evanalyzer_net`'s remote
//! backend) is decided once at startup and invisible to the GUI and CLI.
//!
//! Everything a request or result carries is plain, serializable data, so a
//! remote backend can send it as is. The UI keeps the open project in memory
//! (with undo); every file it reads or writes goes through
//! [`Backend::files`], so it lives on whichever machine the backend runs on.

pub mod files;
pub mod local;
pub mod net;
pub mod results;

use crate::ai_learning::{PixelTrainingParams, RunningTraining, StartTrainingError};
use crate::images::{ImageChannel, ImageMeta};
use crate::job::{PreviewRequest, RunningJob, StartPreviewError};
use evanalyzer_cfg::core_types::{ImageTile, InternalErrors, ZProjection};
use evanalyzer_cfg::settings::ai_learning_settings::AiLearningSettings;
use evanalyzer_cfg::settings::project_settings::ProjectSettings;
pub use files::{DirEntry, FileSystem, LocalFileSystem, Place, PlaceKind};
pub use local::{LocalBackend, ReaderPool};
pub use results::{ExportProgressFn, LocalResults, ResultsSource};
use serde::{Deserialize, Serialize};
use std::ops::RangeInclusive;
use std::path::{Path, PathBuf};
use std::sync::Arc;

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
}

/// An opened image, readable tile by tile.
pub trait ImageSource: Send + Sync {
    /// The image's metadata (series, pyramid levels, channels).
    fn meta(&self) -> &ImageMeta;

    /// Reads one tile of every channel of `req.series`.
    fn read_tile(&self, req: &TileRequest) -> Result<Vec<ImageChannel>, InternalErrors>;
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
