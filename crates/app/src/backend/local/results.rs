//! Results databases in this process: DuckDB queries, charts and exports,
//! served through [`LocalResults`] as the `ResultsSource` for front ends.

mod results_charts;
mod results_exporter;
mod results_generator;
#[cfg(test)]
mod test_support;

pub use results_charts::ResultCharts;
pub use results_generator::ResultsGenerator;

use crate::api::{
    BoxplotFilter, BoxplotResult, ColumnEntry, DatabaseResult, ExportProgressFn,
    GroupedByImageFilter, HistogramFilter, HistogramResult, ImageEntry, ImageHeatmapFilter,
    ListFilter, PlateFilter, ResultExport, ResultsSource, ScatterFilter, ScatterResult, View,
    WellFilter,
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
}

impl LocalResults {
    pub fn open(path: std::path::PathBuf) -> Result<Self, InternalErrors> {
        Ok(Self {
            database: Mutex::new(ResultsGenerator::open_database(path)?),
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

    fn get_images(&self) -> Result<Vec<ImageEntry>, InternalErrors> {
        self.with(|db| db.get_images())
    }

    fn enable_image(&self, image_rel_path: &str, disable: bool) -> Result<(), InternalErrors> {
        self.with(|db| db.enable_image(image_rel_path, disable))
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

    fn boxplot(&self, filter: &BoxplotFilter) -> Result<BoxplotResult, InternalErrors> {
        self.with(|db| ResultCharts {}.paint_boxplot(db, filter))
    }

    fn histogram(&self, filter: &HistogramFilter) -> Result<HistogramResult, InternalErrors> {
        self.with(|db| ResultCharts {}.paint_histogram(db, filter))
    }

    fn scatter(&self, filter: &ScatterFilter) -> Result<ScatterResult, InternalErrors> {
        self.with(|db| ResultCharts {}.paint_scatter(db, filter))
    }

    fn export(
        &self,
        export: &ResultExport,
        cancel: &AtomicBool,
        on_progress: ExportProgressFn,
    ) -> Result<(), InternalErrors> {
        // Its own connection, so the (possibly long) export doesn't block
        // the results window's queries. Cloned from the open one rather than
        // opening the path again: on Windows the OS locks the file
        // exclusively even against a second handle from the same process, so
        // a second open locked the app out of its own database on every export.
        let database = self.with(|db| db.try_clone())?;
        export.start_export(&database, cancel, on_progress)
    }
}
