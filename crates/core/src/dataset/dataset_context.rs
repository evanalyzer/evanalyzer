use crate::job::job_executor::ProgressEvent;
use evanalyzer_cfg::core_types::{InternalErrors, ObjectClass, ObjectId};
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
    mpsc::Sender,
};

/// What a dataset step works with.
pub struct DatasetContext<'a> {
    /// The objects of the run, and where the step's results go.
    pub store: &'a dyn DatasetStore,
    pub progress: Option<Sender<ProgressEvent>>,
    /// Set when the user cancels - long steps check it via [`Self::is_cancelled`].
    pub cancel: Arc<AtomicBool>,
}

impl DatasetContext<'_> {
    pub fn is_cancelled(&self) -> bool {
        self.cancel.load(Ordering::Relaxed)
    }
}

/// The analysed objects, as dataset steps read them - typed, so steps don't
/// depend on the storage's table layout and can be tested in memory.
pub trait DatasetStore: Send + Sync {
    /// The objects matching `filter`, ordered by image, frame and object id.
    fn objects(&self, filter: &ObjectFilter) -> Result<Vec<ObjectRow>, InternalErrors>;
}

/// Which objects [`DatasetStore::objects`] returns. `None` matches all.
#[derive(Debug, Clone, Default)]
pub struct ObjectFilter {
    pub image_rel_path: Option<String>,
    pub t_stack: Option<i32>,
    /// Objects that carry this class (among others). `Unset` matches none:
    /// it is never stored as an object's class.
    pub object_class: Option<ObjectClass>,
}

impl ObjectFilter {
    /// Whether an object with `classes` passes the class filter.
    pub fn matches_classes(&self, classes: &[ObjectClass]) -> bool {
        self.object_class.is_none_or(|class| classes.contains(&class))
    }
}

/// One analysed object: what dataset steps need, no mask.
#[derive(Debug, Clone, PartialEq)]
pub struct ObjectRow {
    pub id: ObjectId,
    pub image_rel_path: String,
    pub t_stack: i32,
    pub z_stack: i32,
    /// Centre of the bounding box, in pixels.
    pub centroid_px: (f64, f64),
    /// x_min, y_min, x_max, y_max in pixels.
    pub bbox_px: [u32; 4],
    pub area_px: u64,
    pub object_classes: Vec<ObjectClass>,
    /// Mean intensity per channel (index = channel), normalized to 0..1;
    /// `None` for a channel that wasn't measured.
    pub intensity_mean: Vec<Option<f64>>,
}
