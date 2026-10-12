//! Dataset steps: run once after every image has been analysed, on the
//! objects of all images, frames and wells - without pixels (e.g. tracking,
//! per-plate normalisation). Image steps live in `algos`.

mod dataset_context;
pub mod tracking;

pub use self::dataset_context::{DatasetContext, DatasetStore, ObjectFilter, ObjectRow};

// Commands, flat like `crate::algos::*` - the generated code names them so.
pub use self::tracking::simple_tracker::SimpleTracker;

use evanalyzer_cfg::core_types::{CitationMetadata, InternalErrors};

pub trait DatasetAlgorithm: Send + Sync {
    /// Execute dataset processing algorithm
    fn execute(&self, ctx: &mut DatasetContext) -> Result<(), InternalErrors>;

    fn name(&self) -> &'static str;
    fn cite(&self) -> Vec<&'static CitationMetadata>;
}
