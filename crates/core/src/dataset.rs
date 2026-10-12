//! Dataset steps: run once after every image has been analysed, on the
//! objects of all images, frames and wells - without pixels (e.g. tracking,
//! per-plate normalisation). Image steps live in `algos`.

mod dataset_context;
pub mod tracking;

pub use self::dataset_context::{DatasetContext, DatasetStore, ObjectFilter, ObjectRow};

// Commands, flat like `crate::algos::*` - the generated code names them so.
pub use self::tracking::simple_tracker::SimpleTracker;

use evanalyzer_cfg::core_types::{CitationMetadata, InternalErrors};
use log::info;
use std::time::Instant;

pub trait DatasetAlgorithm: Send + Sync {
    /// Execute dataset processing algorithm
    fn execute(&self, ctx: &mut DatasetContext) -> Result<(), InternalErrors>;

    fn name(&self) -> &'static str;
    fn cite(&self) -> Vec<&'static CitationMetadata>;
}

/// Runs `steps` in order on `ctx`'s objects; stops at the first error or
/// when the run is cancelled.
pub(crate) fn run_dataset_steps(
    steps: &[Box<dyn DatasetAlgorithm>],
    ctx: &mut DatasetContext,
) -> Result<(), InternalErrors> {
    for step in steps {
        if ctx.is_cancelled() {
            return Err(InternalErrors::Cancelled);
        }
        let start = Instant::now();
        step.execute(ctx)?;
        info!(
            "Executed dataset step {} in {:?}",
            step.name(),
            start.elapsed()
        );
    }
    Ok(())
}

#[cfg(test)]
pub(crate) mod test_support {
    use super::*;
    use std::sync::{Arc, Mutex};

    /// A dataset step that records how many objects it saw - and fails,
    /// if asked to.
    pub(crate) struct RecordingStep {
        pub seen: Arc<Mutex<Vec<usize>>>,
        pub fail: bool,
    }

    impl DatasetAlgorithm for RecordingStep {
        fn execute(&self, ctx: &mut DatasetContext) -> Result<(), InternalErrors> {
            let objects = ctx.store.objects(&ObjectFilter::default())?;
            self.seen.lock().unwrap().push(objects.len());
            if self.fail {
                return Err(InternalErrors::Generic("the dataset step failed".into()));
            }
            Ok(())
        }

        fn name(&self) -> &'static str {
            "Recording step"
        }

        fn cite(&self) -> Vec<&'static CitationMetadata> {
            Vec::new()
        }
    }
}
