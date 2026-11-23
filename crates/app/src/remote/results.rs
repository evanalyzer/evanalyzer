//! # results
//!
//! The results calls over the network, from one list: every
//! [`ResultsSource`] call that is a question and an answer is a line in
//! [`results_calls!`] below. From it come the query and answer that travel
//! (one variant each per call), the client side ([`RemoteResults`] sends
//! the query and unpacks the answer) and the worker's side ([`answer`]
//! calls the local source). A new results call: its trait method, its code
//! in `results/local`, one line here.
//!
//! Queries and answers travel as postcard (not JSON) because results carry
//! `NaN`/`±inf` floats, which JSON can't represent. The variant order is
//! the wire format: a change bumps `PROTOCOL_VERSION`.
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-11-23
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.

use super::client::RemoteResults;
use super::client::unexpected_reply;
use crate::backend::ExportProgressFn;
use crate::results::{
    BoxplotFilter, BoxplotResult, ColumnEntry, DatabaseResult, GroupedByImageFilter, Grouping,
    HistogramFilter, HistogramResult, ImageEntry, ImageHeatmapFilter, ImageObjectsFilter,
    ImageReviewChange, ImageTracks, ImageTracksFilter, ListFilter, MeasurementColumn, PlaneFilter,
    PlateFilter, ProfileFilter, ProfileResult, ProfileStep, ProfileStepFilter, ResultExport,
    ResultObject, ResultsSource, RunStatus, ScatterFilter, ScatterResult, TimeCourseFilter,
    TimeCourseResult, TrackPaths, TrackPathsFilter, TrackSummary, TrackSummaryFilter, TreeNeurons,
    TreeSegment, View, WellFilter,
};
use evanalyzer_cfg::core_types::InternalErrors;
use evanalyzer_cfg::settings::classification_settings::Class;
use serde::{Deserialize, Serialize};
use std::sync::atomic::AtomicBool;

/// The type of an argument on the trait: `borrowed` a reference, `text` a
/// `&str`, `owned` the value.
macro_rules! trait_arg {
    (borrowed $ty:ty) => {
        &$ty
    };
    (text $ty:ty) => {
        &str
    };
    (owned $ty:ty) => {
        $ty
    };
}

/// An argument from the trait onto the wire.
macro_rules! to_wire {
    (borrowed $arg:ident) => {
        $arg.clone()
    };
    (text $arg:ident) => {
        $arg.to_string()
    };
    (owned $arg:ident) => {
        $arg
    };
}

/// An argument from the wire to the trait.
macro_rules! from_wire {
    (borrowed $arg:ident) => {
        &$arg
    };
    (text $arg:ident) => {
        &$arg
    };
    (owned $arg:ident) => {
        $arg
    };
}

/// The list: `Variant => fn trait_method(kind name: WireType, ..) -> Answer;`.
/// `counts`: calls without an error on the trait - a count, 1 when the
/// server can't answer. `by_hand`: the trait methods that aren't one
/// question and one answer.
macro_rules! results_calls {
    (
        calls {
            $( $(#[doc = $doc:literal])*
               $Variant:ident => fn $name:ident( $($kind:ident $arg:ident: $ty:ty),* ) -> $ret:ty; )*
        }
        counts {
            $( $(#[doc = $cdoc:literal])* $CVariant:ident => fn $cname:ident(); )*
        }
        by_hand { $($by_hand:item)* }
    ) => {
        /// One results-database question: a variant per call of the list.
        #[derive(Serialize, Deserialize)]
        pub(crate) enum ResultsQuery {
            $( $(#[doc = $doc])* $Variant { $($arg: $ty),* }, )*
            $( $(#[doc = $cdoc])* $CVariant, )*
        }

        /// The answer to a [`ResultsQuery`], of the same variant (failures
        /// travel as `Reply::Failed`).
        #[derive(Serialize, Deserialize)]
        pub(crate) enum ResultsAnswer {
            $( $Variant($ret), )*
            $( $CVariant(u32), )*
        }

        /// The worker's side: `query` answered from `source`.
        pub(crate) fn answer(
            source: &dyn ResultsSource,
            query: ResultsQuery,
        ) -> Result<ResultsAnswer, InternalErrors> {
            Ok(match query {
                $( ResultsQuery::$Variant { $($arg),* } => {
                    ResultsAnswer::$Variant(source.$name($(from_wire!($kind $arg)),*)?)
                } )*
                $( ResultsQuery::$CVariant => ResultsAnswer::$CVariant(source.$cname()), )*
            })
        }

        /// The client's side: each call sent as its query, its answer
        /// unpacked.
        impl ResultsSource for RemoteResults {
            $(
                fn $name(&self, $($arg: trait_arg!($kind $ty)),*) -> Result<$ret, InternalErrors> {
                    match self.query(ResultsQuery::$Variant { $($arg: to_wire!($kind $arg)),* })? {
                        ResultsAnswer::$Variant(answer) => Ok(answer),
                        _ => Err(unexpected_reply()),
                    }
                }
            )*
            $(
                fn $cname(&self) -> u32 {
                    match self.query(ResultsQuery::$CVariant) {
                        Ok(ResultsAnswer::$CVariant(n)) => n,
                        Ok(_) => {
                            log::warn!("Unexpected answer to a stack-count query");
                            1
                        }
                        Err(e) => {
                            log::warn!("Could not read stack count from the server: {e}");
                            1
                        }
                    }
                }
            )*
            $($by_hand)*
        }
    };
}

results_calls! {
    calls {
        ObjectList => fn get_object_list(borrowed filter: ListFilter) -> DatabaseResult;
        /// See `ResultsSource::object_list_page_starts`.
        ObjectListPageStarts => fn object_list_page_starts(
            borrowed filter: ListFilter,
            text object_id: String
        ) -> Option<Vec<String>>;
        GroupedByImage => fn get_grouped_by_image(borrowed filter: GroupedByImageFilter) -> DatabaseResult;
        GroupByPlate => fn get_group_by_plate(borrowed filter: PlateFilter, borrowed view: View) -> DatabaseResult;
        GroupByWell => fn get_group_by_well(borrowed filter: WellFilter, borrowed view: View) -> DatabaseResult;
        ImageHeatmap => fn get_image_heatmap(
            borrowed filter: ImageHeatmapFilter,
            borrowed view: View
        ) -> DatabaseResult;
        Images => fn get_images() -> Vec<ImageEntry>;
        EnableImage => fn enable_image(text image_rel_path: String, owned disable: bool) -> ();
        ChangeImageReview => fn change_image_review(
            text image_rel_path: String,
            owned change: ImageReviewChange
        ) -> ();
        ObjectClasses => fn get_object_classes() -> Vec<Class>;
        AvailableColumns => fn get_available_columns() -> Vec<ColumnEntry>;
        Planes => fn get_planes() -> Vec<PlaneFilter>;
        RunStatus => fn run_status() -> RunStatus;
        GroupingRegex => fn grouping_regex(borrowed grouping: Grouping) -> String;
        Boxplot => fn boxplot(borrowed filter: BoxplotFilter) -> BoxplotResult;
        Histogram => fn histogram(borrowed filter: HistogramFilter) -> HistogramResult;
        Scatter => fn scatter(borrowed filter: ScatterFilter) -> ScatterResult;
        TimeCourse => fn time_course(borrowed filter: TimeCourseFilter) -> TimeCourseResult;
        ProfileColumns => fn profile_columns() -> Vec<MeasurementColumn>;
        Profile => fn profile(borrowed filter: ProfileFilter) -> ProfileResult;
        ObjectProfile => fn object_profile(text key: String, text object_id: String) -> Vec<(f64, f64)>;
        ProfileStep => fn profile_step(borrowed filter: ProfileStepFilter) -> ProfileStep;
        HasTrees => fn has_trees() -> bool;
        TreeNeurons => fn tree_neurons(owned images: Option<Vec<String>>) -> TreeNeurons;
        TreeSegments => fn tree_segments(text soma_id: String) -> Vec<TreeSegment>;
        ImageObjects => fn image_objects(borrowed filter: ImageObjectsFilter) -> Vec<ResultObject>;
        TrackSummary => fn track_summary(borrowed filter: TrackSummaryFilter) -> Vec<TrackSummary>;
        TrackPaths => fn track_paths(borrowed filter: TrackPathsFilter) -> TrackPaths;
        HasTracks => fn has_tracks() -> bool;
        HasLineage => fn has_lineage() -> bool;
        ImageTracks => fn image_tracks(borrowed filter: ImageTracksFilter) -> ImageTracks;
    }
    counts {
        ZStacks => fn get_nr_of_z_stacks();
        TStacks => fn get_nr_of_t_stacks();
    }
    by_hand {
        /// Streams its progress and can be cancelled - see
        /// `RemoteResults::export_results`.
        fn export(
            &self,
            export: &ResultExport,
            cancel: &AtomicBool,
            on_progress: ExportProgressFn,
        ) -> Result<(), InternalErrors> {
            self.export_results(export, cancel, on_progress)
        }
    }
}
