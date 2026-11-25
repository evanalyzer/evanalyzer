//! # results
//!
//! Results databases in this process: DuckDB queries, charts and exports,
//! served through [`LocalResults`] as the `ResultsSource` for front ends.
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-10-10
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.

pub(crate) mod colorize;
mod results_charts;
pub use colorize::{colorize, colorize_thumbnail};
mod results_generator;
mod results_profile;
mod results_time_course;
mod results_tracks;
mod results_trees;
#[cfg(test)]
pub(crate) mod test_support;

pub use results_charts::ResultCharts;
pub use results_generator::ResultsGenerator;
pub(crate) use results_generator::class_display_label;

use crate::backend::ExportProgressFn;
use crate::results::{
    BoxplotFilter, BoxplotResult, ColumnEntry, DatabaseResult, GroupedByImageFilter,
    HistogramFilter, HistogramResult, ImageEntry, ImageHeatmapFilter, ImageReviewChange,
    ListFilter, PlateFilter, ResultsSource, ScatterFilter, ScatterResult, TimeCourseFilter,
    TimeCourseResult, View, WellFilter,
};
use evanalyzer_cfg::core_types::InternalErrors;
use evanalyzer_cfg::settings::classification_settings::Class;
use std::sync::Mutex;
use std::sync::atomic::AtomicBool;

/// A DuckDB results database in this process. The connection isn't `Sync`,
/// so queries take turns; an export gets its own connection and runs
/// alongside them.
pub struct LocalResults {
    database: Mutex<ResultsGenerator>,
    /// The folders an export may write to and read images from - a
    /// server's `--root`s (unrestricted on the desktop).
    files: crate::files::LocalFileSystem,
}

impl LocalResults {
    /// Exports write and read only within `files` (see `export`).
    pub fn open(
        path: std::path::PathBuf,
        files: crate::files::LocalFileSystem,
    ) -> Result<Self, InternalErrors> {
        Ok(Self {
            database: Mutex::new(ResultsGenerator::open_database(path)?),
            files,
        })
    }

    fn with<T>(&self, f: impl FnOnce(&ResultsGenerator) -> T) -> T {
        f(&self.database.lock().unwrap_or_else(|e| e.into_inner()))
    }
}

impl ResultsSource for LocalResults {
    fn get_object_list(&self, filter: &ListFilter) -> Result<DatabaseResult, InternalErrors> {
        self.with(|db| db.get_object_list(filter))
    }

    fn object_list_page_starts(
        &self,
        filter: &ListFilter,
        object_id: &str,
    ) -> Result<Option<Vec<String>>, InternalErrors> {
        self.with(|db| db.object_list_page_starts(filter, object_id))
    }

    fn get_grouped_by_image(
        &self,
        filter: &GroupedByImageFilter,
    ) -> Result<DatabaseResult, InternalErrors> {
        self.with(|db| db.get_grouped_by_image(filter))
    }

    fn get_group_by_plate(
        &self,
        filter: &PlateFilter,
        view: &View,
    ) -> Result<DatabaseResult, InternalErrors> {
        self.with(|db| db.get_group_by_plate(filter, view))
    }

    fn get_group_by_well(
        &self,
        filter: &WellFilter,
        view: &View,
    ) -> Result<DatabaseResult, InternalErrors> {
        self.with(|db| db.get_group_by_well(filter, view))
    }

    fn get_image_heatmap(
        &self,
        filter: &ImageHeatmapFilter,
        view: &View,
    ) -> Result<DatabaseResult, InternalErrors> {
        self.with(|db| db.get_image_heatmap(filter, view))
    }

    fn get_group_by_plate_multi(
        &self,
        filter: &crate::results::PlateFilterMulti,
        view: &View,
    ) -> Result<Vec<DatabaseResult>, InternalErrors> {
        self.with(|db| db.get_group_by_plate_multi(filter, view))
    }

    fn get_wells_for_plate_multi(
        &self,
        filter: &crate::results::WellsBatchFilterMulti,
        view: &View,
    ) -> Result<Vec<std::collections::HashMap<String, DatabaseResult>>, InternalErrors> {
        self.with(|db| db.get_wells_for_plate_multi(filter, view))
    }

    fn analysed_planes(&self, rel_path: &str) -> Result<Vec<(u32, u32)>, InternalErrors> {
        self.with(|db| db.analysed_planes(rel_path))
    }

    fn for_each_drawn_object(
        &self,
        rel_path: &str,
        plane: (u32, u32),
        wanted: Option<&[u32]>,
        each: &mut dyn FnMut(crate::results::DrawnObject) -> Result<(), InternalErrors>,
    ) -> Result<(), InternalErrors> {
        self.with(|db| db.for_each_drawn_object(rel_path, plane, wanted, each))
    }

    fn get_images(&self) -> Result<Vec<ImageEntry>, InternalErrors> {
        self.with(|db| db.get_images())
    }

    fn enable_image(&self, image_rel_path: &str, disable: bool) -> Result<(), InternalErrors> {
        self.with(|db| db.enable_image(image_rel_path, disable))
    }

    fn change_image_review(
        &self,
        image_rel_path: &str,
        change: ImageReviewChange,
    ) -> Result<(), InternalErrors> {
        self.with(|db| db.change_image_review(image_rel_path, &change))
    }

    fn get_object_classes(&self) -> Result<Vec<Class>, InternalErrors> {
        self.with(|db| db.get_object_classes())
    }

    fn get_available_columns(&self) -> Result<Vec<ColumnEntry>, InternalErrors> {
        self.with(|db| db.get_available_columns())
    }

    fn get_nr_of_z_stacks(&self) -> u32 {
        self.with(|db| db.get_nr_of_z_stacks())
    }

    fn get_nr_of_t_stacks(&self) -> u32 {
        self.with(|db| db.get_nr_of_t_stacks())
    }

    fn get_planes(&self) -> Result<Vec<crate::results::PlaneFilter>, InternalErrors> {
        self.with(|db| db.get_planes())
    }

    fn run_status(&self) -> Result<crate::results::RunStatus, InternalErrors> {
        self.with(|db| db.run_status())
    }

    fn grouping_regex(
        &self,
        grouping: &crate::results::Grouping,
    ) -> Result<String, InternalErrors> {
        self.with(|db| db.grouping_regex(grouping))
    }

    fn boxplot(&self, filter: &BoxplotFilter) -> Result<BoxplotResult, InternalErrors> {
        self.with(|db| ResultCharts {}.paint_boxplot(db, filter))
    }

    fn histogram(&self, filter: &HistogramFilter) -> Result<HistogramResult, InternalErrors> {
        self.with(|db| ResultCharts {}.paint_histogram(db, filter))
    }

    fn scatter(&self, filter: &ScatterFilter) -> Result<ScatterResult, InternalErrors> {
        self.with(|db| ResultCharts {}.paint_scatter(db, filter))
    }

    fn time_course(&self, filter: &TimeCourseFilter) -> Result<TimeCourseResult, InternalErrors> {
        self.with(|db| results_time_course::time_course(db, filter))
    }

    fn time_course_each(
        &self,
        filter: &TimeCourseFilter,
        each: &mut dyn FnMut(&crate::results::TimeCoursePoint) -> Result<(), InternalErrors>,
    ) -> Result<(), InternalErrors> {
        self.with(|db| results_time_course::TimeCourseQuery::prepare(db, filter)?.for_each(each))
    }

    fn profile_columns(&self) -> Result<Vec<crate::results::MeasurementColumn>, InternalErrors> {
        self.with(results_profile::profile_columns)
    }

    fn profile(
        &self,
        filter: &crate::results::ProfileFilter,
    ) -> Result<crate::results::ProfileResult, InternalErrors> {
        self.with(|db| results_profile::profile(db, filter))
    }

    fn object_profile(
        &self,
        key: &str,
        object_id: &str,
    ) -> Result<Vec<(f64, f64)>, InternalErrors> {
        self.with(|db| results_profile::object_profile(db, key, object_id))
    }

    fn profile_step(
        &self,
        filter: &crate::results::ProfileStepFilter,
    ) -> Result<crate::results::ProfileStep, InternalErrors> {
        self.with(|db| results_profile::profile_step(db, filter))
    }

    fn has_trees(&self) -> Result<bool, InternalErrors> {
        self.with(results_trees::has_trees)
    }

    fn tree_neurons(
        &self,
        images: Option<Vec<String>>,
    ) -> Result<crate::results::TreeNeurons, InternalErrors> {
        self.with(|db| results_trees::tree_neurons(db, images.as_deref()))
    }

    fn tree_segments(
        &self,
        soma_id: &str,
    ) -> Result<Vec<crate::results::TreeSegment>, InternalErrors> {
        self.with(|db| results_trees::tree_segments(db, soma_id))
    }

    fn image_objects(
        &self,
        filter: &crate::results::ImageObjectsFilter,
    ) -> Result<Vec<crate::results::ResultObject>, InternalErrors> {
        self.with(|db| ResultCharts {}.image_objects(db, filter))
    }

    fn track_summary(
        &self,
        filter: &crate::results::TrackSummaryFilter,
    ) -> Result<Vec<crate::results::TrackSummary>, InternalErrors> {
        self.with(|db| db.track_summary(filter))
    }

    fn measurement_rows(
        &self,
        filter: &crate::results::MeasurementRowsFilter,
    ) -> Result<Vec<crate::results::MeasurementRow>, InternalErrors> {
        self.with(|db| db.measurement_rows(filter))
    }

    fn track_paths(
        &self,
        filter: &crate::results::TrackPathsFilter,
    ) -> Result<crate::results::TrackPaths, InternalErrors> {
        self.with(|db| db.track_paths(filter))
    }

    fn has_tracks(&self) -> Result<bool, InternalErrors> {
        self.with(|db| ResultCharts {}.has_tracks(db))
    }

    fn has_lineage(&self) -> Result<bool, InternalErrors> {
        self.with(|db| db.has_lineage())
    }

    fn image_tracks(
        &self,
        filter: &crate::results::ImageTracksFilter,
    ) -> Result<crate::results::ImageTracks, InternalErrors> {
        self.with(|db| ResultCharts {}.image_tracks(db, filter))
    }

    fn export_parquet(
        &self,
        out_dir: &std::path::Path,
        cancel: &AtomicBool,
        on_progress: ExportProgressFn,
    ) -> Result<(), InternalErrors> {
        // Within what this backend may reach (a server: its `--root`s) -
        // the folder written to is the client's to choose.
        self.files.check_write(out_dir)?;
        if cancel.load(std::sync::atomic::Ordering::Relaxed) {
            return Err(InternalErrors::Cancelled);
        }
        on_progress("Exporting Parquet", 0, 1);
        // Its own connection, so the export doesn't block the results
        // window's queries (cloned: a second open of the file is locked on
        // Windows).
        let database = self.with(|db| db.try_clone())?;
        database.export_parquet(out_dir)?;
        on_progress("Exporting Parquet", 1, 1);
        Ok(())
    }

    fn run_export(
        &self,
        run: &crate::exports::ExportRun,
        cancel: &AtomicBool,
        on_progress: ExportProgressFn,
    ) -> Result<crate::exports::ExportReport, InternalErrors> {
        // Its own connection, as an export above: the results window's
        // queries go on meanwhile.
        let own = LocalResults {
            database: Mutex::new(self.with(|db| db.try_clone())?),
            files: self.files.clone(),
        };
        // Within what this backend may reach (a server: its `--root`s): the
        // folder the drawn images are read from is the client's to choose -
        // as the folders written to (`may_write`).
        if let Some(drawing) = &run.drawing {
            self.files.check(&drawing.image_root)?;
        }
        let may_write = |folder: &std::path::Path| self.files.check_write(folder).map(|_| ());
        crate::exports::local::run_export(&own, run, &may_write, cancel, on_progress)
    }
}

impl LocalResults {
    /// Results on an open `database`, every folder reachable - for tests.
    #[cfg(test)]
    pub(crate) fn from_generator(database: ResultsGenerator) -> Self {
        Self {
            database: Mutex::new(database),
            files: crate::files::LocalFileSystem::default(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::test_support::{ObjectSpec, seed_db};
    use super::*;
    use crate::exports::{ExportRun, ScriptPermission};
    use crate::files::LocalFileSystem;
    use crate::results::DrawnImages;

    fn run(
        results: &LocalResults,
        output: std::path::PathBuf,
        drawing: Option<DrawnImages>,
    ) -> Result<crate::exports::ExportReport, InternalErrors> {
        let export = serde_json::from_value(serde_json::json!({
            "id": 1, "name": "x", "entries": [
                { "kind": "DATA", "outputs": ["OBJECTS"], "images": "ALL" }
            ]
        }))
        .unwrap();
        results.run_export(
            &ExportRun {
                export,
                project_dir: output.clone(),
                results_dir: output.clone(),
                output: Some(output),
                scripts: ScriptPermission::Skip,
                interpreters: Default::default(),
                drawing,
            },
            &AtomicBool::new(false),
            &mut |_, _, _| {},
        )
    }

    /// A server's export (`--root`): written only within its folders, the
    /// drawn images read only from there - a client chooses both. The
    /// Parquet files the same.
    #[test]
    fn a_restricted_export_writes_and_reads_only_within_its_folders() {
        let root = tempfile::tempdir().unwrap();
        let outside = tempfile::tempdir().unwrap();
        let db = root.path().join("results.evadb");
        seed_db(&db, &[ObjectSpec::new("img1.tif", "ClassA", 1, 100)]);
        let files = LocalFileSystem::restricted_to(&[root.path().to_path_buf()]).unwrap();
        let results = LocalResults::open(db, files).unwrap();
        let no_progress = &mut |_: &str, _: usize, _: usize| {};

        for escape in [outside.path().join("out"), root.path().join("../out")] {
            assert!(
                run(&results, escape.clone(), None).is_err(),
                "{}",
                escape.display()
            );
            assert!(!escape.exists(), "{} written", escape.display());
            assert!(
                results
                    .export_parquet(&escape, &AtomicBool::new(false), no_progress)
                    .is_err()
            );
        }
        let drawing = DrawnImages {
            image_root: outside.path().to_path_buf(),
            ..Default::default()
        };
        assert!(
            run(&results, root.path().join("drawn"), Some(drawing)).is_err(),
            "images read from outside"
        );
        assert!(!root.path().join("drawn").exists());

        let report = run(&results, root.path().join("out"), None).expect("within the root");
        assert!(report.all_written(), "{report:?}");
        assert!(root.path().join("out/objects.csv").is_file());
    }
}
