//! # source
//!
//! What a backend offers of an analysis' results: [`ResultsSource`] - read
//! in this process (`local`) or on a server (through `crate::remote`).
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-10-10
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.

use super::{
    BoxplotFilter, BoxplotResult, ColumnEntry, DatabaseResult, GroupedByImageFilter,
    HistogramFilter, HistogramResult, ImageEntry, ImageHeatmapFilter, ImageReviewChange,
    ListFilter, PlateFilter, ResultExport, RunStatus, ScatterFilter, ScatterResult,
    TimeCourseFilter, TimeCourseResult, View, WellFilter,
};
use crate::backend::ExportProgressFn;
use evanalyzer_cfg::core_types::InternalErrors;
use evanalyzer_cfg::settings::classification_settings::Class;
use std::sync::atomic::AtomicBool;

pub trait ResultsSource: Send + Sync {
    fn get_object_list(&self, filter: &ListFilter) -> Result<DatabaseResult, InternalErrors>;
    /// Where object `object_id` lies in the object list `filter` asks for:
    /// the start (the `object_id` a page comes after) of every page before
    /// its own - it is on page `len()`, 0 the first. `None`: the list
    /// doesn't hold it, or its rows aren't objects (transposed).
    /// `filter.page.after` is ignored, `filter.page.limit` is the page size.
    fn object_list_page_starts(
        &self,
        filter: &ListFilter,
        object_id: &str,
    ) -> Result<Option<Vec<String>>, InternalErrors>;
    fn get_grouped_by_image(
        &self,
        filter: &GroupedByImageFilter,
    ) -> Result<DatabaseResult, InternalErrors>;
    fn get_group_by_plate(
        &self,
        filter: &PlateFilter,
        view: &View,
    ) -> Result<DatabaseResult, InternalErrors>;
    fn get_group_by_well(
        &self,
        filter: &WellFilter,
        view: &View,
    ) -> Result<DatabaseResult, InternalErrors>;
    fn get_image_heatmap(
        &self,
        filter: &ImageHeatmapFilter,
        view: &View,
    ) -> Result<DatabaseResult, InternalErrors>;
    fn get_images(&self) -> Result<Vec<ImageEntry>, InternalErrors>;
    /// Marks an image as excluded from (`disable`) or included in the
    /// results - persisted in the database.
    fn enable_image(&self, image_rel_path: &str, disable: bool) -> Result<(), InternalErrors>;
    /// Changes the note, marker or left-out reason of an image in these
    /// results (not in the project).
    fn change_image_review(
        &self,
        image_rel_path: &str,
        change: ImageReviewChange,
    ) -> Result<(), InternalErrors>;
    fn get_object_classes(&self) -> Result<Vec<Class>, InternalErrors>;
    fn get_available_columns(&self) -> Result<Vec<ColumnEntry>, InternalErrors>;
    fn get_nr_of_z_stacks(&self) -> u32;
    fn get_nr_of_t_stacks(&self) -> u32;
    /// The planes (time point, z-plane) the results have objects in, time
    /// point by time point - what an export or view over "all" covers.
    fn get_planes(&self) -> Result<Vec<crate::results::PlaneFilter>, InternalErrors>;
    /// How the analysis that wrote the database ended.
    fn run_status(&self) -> Result<RunStatus, InternalErrors>;
    /// The regex `grouping` uses on this database's images - for
    /// `Grouping::Auto` the pattern it detected.
    fn grouping_regex(&self, grouping: &crate::results::Grouping)
    -> Result<String, InternalErrors>;
    fn boxplot(&self, filter: &BoxplotFilter) -> Result<BoxplotResult, InternalErrors>;
    fn histogram(&self, filter: &HistogramFilter) -> Result<HistogramResult, InternalErrors>;
    fn scatter(&self, filter: &ScatterFilter) -> Result<ScatterResult, InternalErrors>;
    /// Per time point, statistics over each line's objects (`TimeCourseFilter`).
    fn time_course(&self, filter: &TimeCourseFilter) -> Result<TimeCourseResult, InternalErrors>;
    /// The profile measurements (a value per object and position, e.g. the
    /// Sholl intersections per radius) - Charts › Profile's picker.
    fn profile_columns(&self) -> Result<Vec<crate::results::MeasurementColumn>, InternalErrors>;
    /// Per position, statistics of a profile over each line's objects
    /// (`ProfileFilter`).
    fn profile(
        &self,
        filter: &crate::results::ProfileFilter,
    ) -> Result<crate::results::ProfileResult, InternalErrors>;
    /// One object's profile: (position, value), by position.
    fn object_profile(&self, key: &str, object_id: &str)
    -> Result<Vec<(f64, f64)>, InternalErrors>;
    /// Steps through the objects with a profile (`ProfileStepFilter`).
    fn profile_step(
        &self,
        filter: &crate::results::ProfileStepFilter,
    ) -> Result<crate::results::ProfileStep, InternalErrors>;
    /// The results hold traced neurons (Trace Filaments - Results › Trees
    /// is offered).
    fn has_trees(&self) -> Result<bool, InternalErrors>;
    /// The traced neurons of `images` (`None`: all) with their Measure
    /// Filament Tree values.
    fn tree_neurons(
        &self,
        images: Option<Vec<String>>,
    ) -> Result<crate::results::TreeNeurons, InternalErrors>;
    /// The segments of the neuron with soma `soma_id`.
    fn tree_segments(
        &self,
        soma_id: &str,
    ) -> Result<Vec<crate::results::TreeSegment>, InternalErrors>;
    /// The objects of one image (Results › Split with image).
    fn image_objects(
        &self,
        filter: &crate::results::ImageObjectsFilter,
    ) -> Result<Vec<crate::results::ResultObject>, InternalErrors>;
    /// A row per track (`TrackSummaryFilter`).
    fn track_summary(
        &self,
        filter: &crate::results::TrackSummaryFilter,
    ) -> Result<Vec<crate::results::TrackSummary>, InternalErrors>;
    /// A sample of the tracks, each from its start (Results › Paths).
    fn track_paths(
        &self,
        filter: &crate::results::TrackPathsFilter,
    ) -> Result<crate::results::TrackPaths, InternalErrors>;
    /// Some of the objects are tracked (Results › Tracks is offered).
    fn has_tracks(&self) -> Result<bool, InternalErrors>;
    /// A tracking step recorded divisions or fusions (the tracks' history
    /// is offered).
    fn has_lineage(&self) -> Result<bool, InternalErrors>;
    /// One image's tracks, their points by time (Results › Tracks).
    fn image_tracks(
        &self,
        filter: &crate::results::ImageTracksFilter,
    ) -> Result<crate::results::ImageTracks, InternalErrors>;
    /// Writes the export's files into `export.output_dir` (on the backend's
    /// machine). Blocks; stops early with `Cancelled` once `cancel` is set.
    fn export(
        &self,
        export: &ResultExport,
        cancel: &AtomicBool,
        on_progress: ExportProgressFn,
    ) -> Result<(), InternalErrors>;
}
