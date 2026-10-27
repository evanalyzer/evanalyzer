//! # preview
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-10-10
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.

use super::spawn_job;
use crate::api::{PreviewRequest, RunningJob, StartPreviewError};
use crate::workspace::settings::ViewportMemorySettings;
use evanalyzer_core::PreviewTileSettings;
use log::{debug, error};

/// Starts a preview run restricted to the tiles visible in `viewport` (or
/// of the whole image, see [`PreviewRequest::whole_image`]). The
/// final merged objects are returned by [`RunningJob::wait`] in
/// [`JobOutput::preview_objects`](super::JobOutput::preview_objects).
pub(crate) fn start_preview(request: PreviewRequest) -> Result<RunningJob, StartPreviewError> {
    let (mut job, output) = evanalyzer_core::generate_preview_job_from_project_settings(
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
        process_all_tiles: request.whole_image,
    });

    // The viewport: as asked, else what this computer can take. The whole
    // image: no limit - all of it, as asked for.
    if !request.whole_image {
        let max = request
            .max_visible_tiles
            .unwrap_or_else(|| {
                ViewportMemorySettings::auto_preview_tiles(
                    crate::backend::local::system::cpu_ram_diagnostics().1,
                )
            })
            .max(1) as usize;
        match job.count_preview_visible_tiles() {
            Ok(tiles) if tiles > max => {
                debug!("Preview rejected: viewport covers {tiles} tiles (max {max})");
                return Err(StartPreviewError::TooManyTiles { tiles, max });
            }
            // Not being able to count is no reason to block the preview - the
            // job itself reports a real read error if there is one.
            Err(e) => error!("Failed to count visible preview tiles: {e:?}"),
            Ok(_) => {}
        }
    }

    job.breakpoint = request.breakpoint;
    job.prediction = request.prediction;
    job.breakpoint_tiles = request.breakpoint_tiles.max(1) as usize;

    let parallelism = evanalyzer_core::recommended_parallelism(job.estimate_ram_per_worker_bytes());
    Ok(spawn_job(job, parallelism, Some(output)))
}
