use super::spawn_job;
use crate::api::{MAX_PREVIEW_VISIBLE_TILES, PreviewRequest, RunningJob, StartPreviewError};
use evanalyzer_core::PreviewTileSettings;
use log::{error, info};

/// Starts a preview run restricted to the tiles visible in `viewport`. The
/// final merged objects are returned by [`RunningJob::wait`] in
/// [`JobOutput::preview_objects`](super::JobOutput::preview_objects).
pub(crate) fn start_preview(request: PreviewRequest) -> Result<RunningJob, StartPreviewError> {
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
    Ok(spawn_job(job, parallelism, Some(out_objects)))
}
