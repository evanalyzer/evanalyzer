//! Runs everything in this process - the default backend, and what a server
//! (`evanalyzer_net`) uses to execute the requests it receives.

use super::{AnalysisRequest, Backend, ImageSource, TileRequest, TrainingRequest};
use crate::ai_learning::{self, RunningTraining, StartTrainingError};
use crate::images::{ImageChannel, ImageMeta};
use crate::job::{self, PreviewRequest, RunningJob, StartPreviewError};
use evanalyzer_cfg::core_types::InternalErrors;
use evanalyzer_core::{ImageReader, ReadMode, recommended_reader_pool_size};
use rayon::prelude::*;
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// Stateless: every operation forwards to the `job`/`ai_learning` functions
/// that hold the actual logic.
#[derive(Debug, Default)]
pub struct LocalBackend;

impl Backend for LocalBackend {
    fn start_analysis(&self, req: AnalysisRequest) -> Result<RunningJob, InternalErrors> {
        job::start_analysis(req.settings, req.project_path, req.job_name, req.threads)
    }

    fn start_preview(&self, req: PreviewRequest) -> Result<RunningJob, StartPreviewError> {
        job::start_preview(req)
    }

    fn start_training(&self, req: TrainingRequest) -> Result<RunningTraining, StartTrainingError> {
        ai_learning::start_training(&req.project, req.settings, req.pixel_params)
    }

    fn open_image(&self, path: &Path) -> Result<Arc<dyn ImageSource>, InternalErrors> {
        Ok(Arc::new(ReaderPool::open(path)?))
    }

    fn description(&self) -> String {
        "local".into()
    }
}

/// A pool of independent readers open on the same image path, so different
/// channels/Z-slices can be read truly in parallel instead of serializing
/// through one reader's internal `Mutex` (see `evanalyzer_core::ImageReader`)
/// - one reader is safe, not concurrent. [`AppHandle`](crate::AppHandle)
/// caches the pool it gets from [`LocalBackend::open_image`] and serves both
/// metadata and tile reads from it, so a given path is parsed only once per
/// selection.
///
/// Built as the primary reader (unavoidable - channel count can only be
/// learned from a full parse) plus exactly `channel_count - 1` more, capped
/// to [`recommended_reader_pool_size`] and built in parallel - never more
/// readers than the image actually has channels.
///
/// A blind batch of `recommended_reader_pool_size` readers, trimmed to real
/// channel count only *after* building all of them, was tried and reverted:
/// it keeps wall time close to one parse *only* when I/O has enough spare
/// bandwidth to run all of them without contention. Measured on real
/// hardware it did not - building 8 readers for a 2-channel file took
/// ~1.0-1.1s wall time (bottlenecked by the slowest of 8 concurrently-
/// contending reads, most of them immediately discarded), and directly
/// stalled the render worker's first read of a newly opened image for that
/// same ~1s, visible as `read` jumping from microseconds to over a second
/// in `viewport_worker.rs`'s own timing log. Building only what's needed
/// bounds worst-case latency to roughly two sequential parses, predictably,
/// regardless of storage speed - unlike the blind-batch approach, whose
/// downside has no such bound on slower storage.
///
/// Deliberately not using `bioformats::Memoizer` here: unlike Java
/// Bio-Formats' `Memoizer` (which deep-clones the whole reader's internal
/// parsed state, e.g. TIFF IFD offset tables), this crate's `Memoizer` only
/// caches the lightweight `ImageMetadata`/`OmeMetadata` summary to disk. Its
/// `set_resolution` unconditionally forces a full real reopen regardless of
/// cache state - and every pool member needs a working resolution/pixel
/// read, not just cached summary fields - so wrapping pool members in it
/// would add a `.bfmemo` file next to every opened image for no actual
/// savings on this specific path.
pub struct ReaderPool {
    readers: Vec<Arc<ImageReader>>,
}

impl ReaderPool {
    pub fn open(path: &Path) -> Result<Self, InternalErrors> {
        let path: PathBuf = path.to_path_buf();
        let start = std::time::Instant::now();

        // The primary reader's parse is unavoidable up front - channel
        // count (needed to size the rest of the pool) can only be learned
        // from a full parse, there's no cheaper way to ask a format "how
        // many channels do you have".
        let primary = Arc::new(ImageReader::new(&path, ReadMode::SplitChannels)?);
        let channel_count = primary
            .get_image_meta()
            .series
            .values()
            .map(|s| s.nr_c_stacks.max(1) as usize)
            .max()
            .unwrap_or(1);
        let size = recommended_reader_pool_size().min(channel_count).max(1);

        // The remaining `size - 1` members are built in parallel: they each
        // pay their own full parse cost (no Memoizer cache to populate or
        // race on, unlike the old Java-backed reader), so building them one
        // at a time would serialize N full opens back to back instead of
        // overlapping them. Concurrent construction of independent readers
        // on the same path is already relied on elsewhere (see
        // `concurrent_readers_on_independent_threads_produce_consistent_results`
        // in `evanalyzer_core::image::image_reader`'s own tests).
        let readers: Vec<Arc<ImageReader>> = if size > 1 {
            let mut rest: Vec<Arc<ImageReader>> = (0..size - 1)
                .into_par_iter()
                .map(|_| ImageReader::new(&path, ReadMode::SplitChannels).map(Arc::new))
                .collect::<Result<Vec<_>, InternalErrors>>()?;
            rest.insert(0, primary);
            rest
        } else {
            vec![primary]
        };

        log::info!(
            "Built reader pool of {size} (channel count {channel_count}) for {} in {:?}",
            path.display(),
            start.elapsed()
        );
        Ok(Self { readers })
    }
}

impl ImageSource for ReaderPool {
    /// Parsed once when the pool was built.
    fn meta(&self) -> &ImageMeta {
        self.readers[0].get_image_meta()
    }

    /// Spreads the channels/Z-slices across the pool's readers.
    fn read_tile(&self, req: &TileRequest) -> Result<Vec<ImageChannel>, InternalErrors> {
        ImageReader::read_image_tile_combined_pooled(
            &self.readers,
            req.series,
            req.resolution_idx,
            req.z_projection.clone(),
            &req.z_range,
            req.t_stack,
            None,
            &req.tile,
        )
    }
}
