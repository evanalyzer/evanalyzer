use super::RunningJob;
use evanalyzer_cfg::{
    core_types::{BreakpointSettings, InternalErrors},
    settings::project_settings::ProjectSettings,
};
use evanalyzer_core::PreviewTileSettings;
use log::{error, info};
use std::path::PathBuf;

/// Previews covering more tiles than this are rejected: at low zoom the
/// viewport of a whole-slide image can span hundreds of tiles, each
/// potentially producing huge numbers of ROIs that the viewport renderer and
/// object list can't handle responsively.
pub const MAX_PREVIEW_VISIBLE_TILES: usize = 4;

/// The visible area a preview is restricted to, in screen pixels.
#[derive(Clone, Copy, Debug, Default)]
pub struct PreviewViewport {
    /// Current pan offset (screen pixels from the image's top-left corner).
    pub offset_x: f32,
    pub offset_y: f32,
    pub viewport_width: f32,
    pub viewport_height: f32,
    /// Current zoom level (1.0 = 100 %).
    pub zoom: f32,
}

pub struct PreviewRequest {
    pub settings: ProjectSettings,
    pub project_path: PathBuf,
    pub viewport: PreviewViewport,
    /// Stop at (or snapshot) a pipeline step and stream its image back via
    /// `ProgressEvent::BreakpointReached`.
    pub breakpoint: Option<BreakpointSettings>,
}

#[derive(Debug)]
pub enum StartPreviewError {
    /// The viewport covers more than [`MAX_PREVIEW_VISIBLE_TILES`] tiles -
    /// the user has to zoom in first.
    TooManyTiles {
        tiles: usize,
    },
    Failed(InternalErrors),
}

impl From<InternalErrors> for StartPreviewError {
    fn from(e: InternalErrors) -> Self {
        StartPreviewError::Failed(e)
    }
}

/// Starts a preview run restricted to the tiles visible in `viewport`. The
/// final merged objects are returned by [`RunningJob::wait`] in
/// [`JobOutput::preview_objects`](super::JobOutput::preview_objects).
pub fn start_preview(request: PreviewRequest) -> Result<RunningJob, StartPreviewError> {
    let (mut job, out_objects) = evanalyzer_core::generate_preview_job_from_project_settings(
        request.settings,
        request.project_path,
    )?;

    let vp = request.viewport;
    job.preview_tile_settings = Some(PreviewTileSettings {
        offset_x: vp.offset_x,
        offset_y: vp.offset_y,
        viewport_width: vp.viewport_width,
        viewport_height: vp.viewport_height,
        zoom: vp.zoom,
        process_all_tiles: false,
    });

    match job.count_preview_visible_tiles() {
        Ok(tiles) if tiles > MAX_PREVIEW_VISIBLE_TILES => {
            info!(
                "Preview rejected: viewport covers {tiles} tiles (max {MAX_PREVIEW_VISIBLE_TILES})"
            );
            return Err(StartPreviewError::TooManyTiles { tiles });
        }
        // Not being able to count is no reason to block the preview - the
        // job itself reports a real read error if there is one.
        Err(e) => error!("Failed to count visible preview tiles: {e:?}"),
        Ok(_) => {}
    }

    job.breakpoint = request.breakpoint;

    let parallelism = evanalyzer_core::recommended_parallelism(job.estimate_ram_per_worker_bytes());
    Ok(RunningJob::spawn(job, parallelism, Some(out_objects)))
}
