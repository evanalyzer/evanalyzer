use evanalyzer_cfg::core_types::CitationMetadata;
use macros::CommandsMeta;

use crate::DatasetAlgorithm;

/// Links each object to the nearest one in the next frame.
#[derive(CommandsMeta)]
#[cmdsmeta(category = "dataset")]
pub struct SimpleTracker {}

impl DatasetAlgorithm for SimpleTracker {
    fn execute(
        &self,
        ctx: &mut crate::DatasetContext,
    ) -> Result<(), evanalyzer_cfg::core_types::InternalErrors> {
        todo!("link the objects frame to frame")
    }

    fn name(&self) -> &'static str {
        "Simple Tracker"
    }

    fn cite(&self) -> Vec<&'static evanalyzer_cfg::core_types::CitationMetadata> {
        vec![&CitationMetadata::DANMAYR]
    }
}
