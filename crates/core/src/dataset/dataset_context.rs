//! # dataset_context
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-10-15
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.

use crate::job::job_executor::ProgressEvent;
use crate::storage::DatasetStore;
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
