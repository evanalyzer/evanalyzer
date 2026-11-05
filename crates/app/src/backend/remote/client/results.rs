//! # results
//!
//! [`RemoteResults`]: a results database opened on the worker; every query
//! is one request. Survives a reconnect: opened again on the new connection.
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-10-10
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.

use super::link::{Link, Reopen};
use super::session::{Session, unexpected_reply};
use crate::api::BoxplotFilter;
use crate::api::BoxplotResult;
use crate::api::ColumnEntry;
use crate::api::DatabaseResult;
use crate::api::ExportProgressFn;
use crate::api::GroupedByImageFilter;
use crate::api::HistogramFilter;
use crate::api::HistogramResult;
use crate::api::ImageEntry;
use crate::api::ImageHeatmapFilter;
use crate::api::ListFilter;
use crate::api::PlateFilter;
use crate::api::ResultExport;
use crate::api::ResultsSource;
use crate::api::ScatterFilter;
use crate::api::ScatterResult;
use crate::api::View;
use crate::api::WellFilter;
use crate::backend::remote::wire::frame::Frame;
use crate::backend::remote::wire::protocol::{
    ClientMsg, Reply, Request, ResultsAnswer, ResultsQuery, from_postcard, to_postcard,
};
use evanalyzer_cfg::core_types::InternalErrors;
use evanalyzer_cfg::settings::classification_settings::Class;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::sync::mpsc;
use std::time::Duration;

pub(super) struct RemoteResults {
    path: PathBuf,
    handle: Reopen,
}

/// Opens the results database `path` on `session`: its handle.
fn open(session: &Session, path: &Path) -> Result<u64, InternalErrors> {
    let (id, rx) = session.request(Request::OpenResults {
        path: path.to_path_buf(),
    })?;
    let reply = session.recv(&rx);
    session.finish(id);
    match reply?.msg {
        Reply::ResultsOpened { handle } => Ok(handle),
        Reply::Failed(e) => Err(e.into_internal()),
        _ => Err(unexpected_reply()),
    }
}

impl RemoteResults {
    /// Opens `path` on the link's connection.
    pub(super) fn open(link: &Arc<Link>, path: &Path) -> Result<Self, InternalErrors> {
        let (session, generation) = link.current();
        let handle = open(&session, path)?;
        Ok(Self {
            path: path.to_path_buf(),
            handle: Reopen::new(Arc::clone(link), generation, handle),
        })
    }

    /// The connection to use and the database's handle on it.
    fn session(&self) -> Result<(Arc<Session>, u64), InternalErrors> {
        self.handle.get(|session| open(session, &self.path))
    }

    fn query(&self, query: ResultsQuery) -> Result<ResultsAnswer, InternalErrors> {
        let (session, handle) = self.session()?;
        let request = Request::QueryResults { handle };
        let (id, rx) = session.request_with_blobs(request, vec![to_postcard(&query)?])?;
        let reply = session.recv(&rx);
        session.finish(id);
        match reply? {
            Frame {
                msg: Reply::ResultsAnswer,
                blobs,
            } => from_postcard(blobs.first()),
            Frame {
                msg: Reply::Failed(e),
                ..
            } => Err(e.into_internal()),
            _ => Err(unexpected_reply()),
        }
    }

    fn table(&self, query: ResultsQuery) -> Result<DatabaseResult, InternalErrors> {
        match self.query(query)? {
            ResultsAnswer::Table(table) => Ok(table),
            _ => Err(unexpected_reply()),
        }
    }

    fn count(&self, query: ResultsQuery) -> u32 {
        match self.query(query) {
            Ok(ResultsAnswer::Count(n)) => n,
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
}

impl ResultsSource for RemoteResults {
    fn get_object_list(&self, filter: &ListFilter) -> Result<DatabaseResult, InternalErrors> {
        self.table(ResultsQuery::ObjectList(filter.clone()))
    }

    fn get_grouped_by_image(
        &self,
        filter: &GroupedByImageFilter,
    ) -> Result<DatabaseResult, InternalErrors> {
        self.table(ResultsQuery::GroupedByImage(filter.clone()))
    }

    fn get_group_by_plate(
        &self,
        filter: &PlateFilter,
        view: &View,
    ) -> Result<DatabaseResult, InternalErrors> {
        self.table(ResultsQuery::GroupByPlate(filter.clone(), view.clone()))
    }

    fn get_group_by_well(
        &self,
        filter: &WellFilter,
        view: &View,
    ) -> Result<DatabaseResult, InternalErrors> {
        self.table(ResultsQuery::GroupByWell(filter.clone(), view.clone()))
    }

    fn get_image_heatmap(
        &self,
        filter: &ImageHeatmapFilter,
        view: &View,
    ) -> Result<DatabaseResult, InternalErrors> {
        self.table(ResultsQuery::ImageHeatmap(filter.clone(), view.clone()))
    }

    fn get_images(&self) -> Result<Vec<ImageEntry>, InternalErrors> {
        match self.query(ResultsQuery::Images)? {
            ResultsAnswer::Images(images) => Ok(images),
            _ => Err(unexpected_reply()),
        }
    }

    fn enable_image(&self, image_rel_path: &str, disable: bool) -> Result<(), InternalErrors> {
        let query = ResultsQuery::EnableImage {
            image_rel_path: image_rel_path.into(),
            disable,
        };
        match self.query(query)? {
            ResultsAnswer::Done => Ok(()),
            _ => Err(unexpected_reply()),
        }
    }

    fn get_object_classes(&self) -> Result<Vec<Class>, InternalErrors> {
        match self.query(ResultsQuery::ObjectClasses)? {
            ResultsAnswer::Classes(classes) => Ok(classes),
            _ => Err(unexpected_reply()),
        }
    }

    fn get_available_columns(&self) -> Result<Vec<ColumnEntry>, InternalErrors> {
        match self.query(ResultsQuery::AvailableColumns)? {
            ResultsAnswer::Columns(columns) => Ok(columns),
            _ => Err(unexpected_reply()),
        }
    }

    fn get_nr_of_z_stacks(&self) -> u32 {
        self.count(ResultsQuery::ZStacks)
    }

    fn get_nr_of_t_stacks(&self) -> u32 {
        self.count(ResultsQuery::TStacks)
    }

    fn get_planes(&self) -> Result<Vec<crate::api::PlaneFilter>, InternalErrors> {
        match self.query(ResultsQuery::Planes)? {
            ResultsAnswer::Planes(planes) => Ok(planes),
            _ => Err(unexpected_reply()),
        }
    }

    fn run_status(&self) -> Result<crate::api::RunStatus, InternalErrors> {
        match self.query(ResultsQuery::RunStatus)? {
            ResultsAnswer::RunStatus(status) => Ok(status),
            _ => Err(unexpected_reply()),
        }
    }

    fn grouping_regex(&self, grouping: &crate::api::Grouping) -> Result<String, InternalErrors> {
        match self.query(ResultsQuery::GroupingRegex(grouping.clone()))? {
            ResultsAnswer::Text(regex) => Ok(regex),
            _ => Err(unexpected_reply()),
        }
    }

    fn boxplot(&self, filter: &BoxplotFilter) -> Result<BoxplotResult, InternalErrors> {
        match self.query(ResultsQuery::Boxplot(filter.clone()))? {
            ResultsAnswer::Boxplot(result) => Ok(result),
            _ => Err(unexpected_reply()),
        }
    }

    fn histogram(&self, filter: &HistogramFilter) -> Result<HistogramResult, InternalErrors> {
        match self.query(ResultsQuery::Histogram(filter.clone()))? {
            ResultsAnswer::Histogram(result) => Ok(result),
            _ => Err(unexpected_reply()),
        }
    }

    fn scatter(&self, filter: &ScatterFilter) -> Result<ScatterResult, InternalErrors> {
        match self.query(ResultsQuery::Scatter(filter.clone()))? {
            ResultsAnswer::Scatter(result) => Ok(result),
            _ => Err(unexpected_reply()),
        }
    }

    fn time_course(
        &self,
        filter: &crate::api::TimeCourseFilter,
    ) -> Result<crate::api::TimeCourseResult, InternalErrors> {
        match self.query(ResultsQuery::TimeCourse(filter.clone()))? {
            ResultsAnswer::TimeCourse(result) => Ok(result),
            _ => Err(unexpected_reply()),
        }
    }

    fn image_objects(
        &self,
        filter: &crate::api::ImageObjectsFilter,
    ) -> Result<Vec<crate::api::ResultObject>, InternalErrors> {
        match self.query(ResultsQuery::ImageObjects(filter.clone()))? {
            ResultsAnswer::Objects(objects) => Ok(objects),
            _ => Err(unexpected_reply()),
        }
    }

    fn track_summary(
        &self,
        filter: &crate::api::TrackSummaryFilter,
    ) -> Result<Vec<crate::api::TrackSummary>, InternalErrors> {
        match self.query(ResultsQuery::TrackSummary(filter.clone()))? {
            ResultsAnswer::TrackSummary(tracks) => Ok(tracks),
            _ => Err(unexpected_reply()),
        }
    }

    fn has_tracks(&self) -> Result<bool, InternalErrors> {
        match self.query(ResultsQuery::HasTracks)? {
            ResultsAnswer::Count(n) => Ok(n > 0),
            _ => Err(unexpected_reply()),
        }
    }

    fn has_lineage(&self) -> Result<bool, InternalErrors> {
        match self.query(ResultsQuery::HasLineage)? {
            ResultsAnswer::Count(n) => Ok(n > 0),
            _ => Err(unexpected_reply()),
        }
    }

    fn image_tracks(
        &self,
        filter: &crate::api::ImageTracksFilter,
    ) -> Result<crate::api::ImageTracks, InternalErrors> {
        match self.query(ResultsQuery::ImageTracks(filter.clone()))? {
            ResultsAnswer::Tracks(tracks) => Ok(tracks),
            _ => Err(unexpected_reply()),
        }
    }

    fn export(
        &self,
        export: &ResultExport,
        cancel: &AtomicBool,
        on_progress: ExportProgressFn,
    ) -> Result<(), InternalErrors> {
        let (session, handle) = self.session()?;
        let request = Request::ExportResults { handle };
        let (id, rx) = session.request_with_blobs(request, vec![to_postcard(export)?])?;
        let mut cancel_sent = false;
        let result = loop {
            if !cancel_sent && cancel.load(std::sync::atomic::Ordering::SeqCst) {
                cancel_sent = true;
                let _ = session.send(&ClientMsg::Cancel { id });
            }
            match rx.recv_timeout(Duration::from_millis(100)) {
                Ok(Frame {
                    msg:
                        Reply::ExportProgress {
                            message,
                            current,
                            total,
                        },
                    ..
                }) => on_progress(&message, current, total),
                Ok(Frame {
                    msg: Reply::ExportDone(result),
                    ..
                }) => break result.map_err(|e| e.into_internal()),
                Ok(Frame {
                    msg: Reply::Failed(e),
                    ..
                }) => break Err(e.into_internal()),
                Ok(_) => break Err(unexpected_reply()),
                Err(mpsc::RecvTimeoutError::Timeout) => continue,
                Err(mpsc::RecvTimeoutError::Disconnected) => {
                    break Err(session.disconnected());
                }
            }
        };
        session.finish(id);
        result
    }
}

impl Drop for RemoteResults {
    fn drop(&mut self) {
        if let Some((session, handle)) = self.handle.to_close() {
            let _ = session.send(&ClientMsg::CloseResults { handle });
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::api::*;
    use crate::backend::local::LocalBackend;
    use crate::backend::remote::{RemoteBackend, Worker};
    use crate::workspace::ProjectWithRuntime;
    use crate::workspace::extensions::images_ext::ImageListExt;
    use crate::workspace::extensions::project_ext::ProjectExt;
    use crate::workspace::settings::AppSettings;
    use evanalyzer_cfg::core_types::InternalErrors;
    use evanalyzer_cfg::settings::classification_settings::Class;
    use std::path::{Path, PathBuf};
    use std::sync::Arc;
    use std::sync::atomic::AtomicBool;
    use std::time::{Duration, Instant};

    const TOKEN: &str = "results-test-token";

    /// A worker in this process serving `backend`, and a client connected to it.
    fn connect(backend: impl Backend + 'static) -> RemoteBackend {
        let worker = Worker::bind("127.0.0.1:0", TOKEN.into()).unwrap();
        let url = format!("ws://{}", worker.local_addr().unwrap());
        std::thread::spawn(move || worker.run(Arc::new(backend)));
        RemoteBackend::connect(&url, TOKEN).unwrap()
    }

    /// The `.evadb` of a local analysis of a one-image project (plus the
    /// temp dir keeping it alive).
    fn analyzed_database() -> (tempfile::TempDir, PathBuf) {
        let dir = tempfile::tempdir().unwrap();
        let images = dir.path().join("images");
        std::fs::create_dir_all(&images).unwrap();
        std::fs::copy(
            Path::new(env!("CARGO_MANIFEST_DIR")).join("../core/tests/slice_Z0_C0_T0.tif"),
            images.join("slice.tif"),
        )
        .unwrap();
        let local = LocalBackend::default();
        let mut project = ProjectWithRuntime::default();
        project.images.root = Some(images);
        project.scan_image_folder_and_add(&local);
        assert_eq!(project.images.list.len(), 1);
        let job = local
            .start_analysis(AnalysisRequest {
                settings: project.settings,
                project_path: dir.path().to_path_buf(),
                job_name: Some("remote_results".into()),
                threads: Some(1),
                base_results: None,
            })
            .unwrap();
        job.events().iter().for_each(drop);
        let output = job.output_path().clone();
        job.wait().unwrap();
        let db = std::fs::read_dir(&output)
            .unwrap()
            .map(|e| e.unwrap().path())
            .find(|p| p.extension().is_some_and(|e| e == "evadb"))
            .expect("analysis writes an .evadb");
        (dir, db)
    }

    /// Any serializable answer, comparable.
    fn json<T: serde::Serialize>(value: T) -> serde_json::Value {
        serde_json::to_value(value).unwrap()
    }

    fn wait_until_disconnected(remote: &RemoteBackend) {
        let deadline = Instant::now() + Duration::from_secs(5);
        while remote.is_connected() && Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(5));
        }
        assert!(!remote.is_connected());
    }

    fn plane() -> PlaneFilter {
        PlaneFilter {
            z_stack: 0,
            t_stack: 0,
        }
    }

    fn histogram() -> HistogramFilter {
        HistogramFilter {
            plane: plane(),
            images: None,
            object_classes: None,
            column: Column::AreaSizePx,
            bins: 8,
        }
    }

    fn time_course() -> TimeCourseFilter {
        TimeCourseFilter {
            lines: TimeCourseLines::Class,
            grouping: Grouping::Auto,
            columns: vec![Column::AreaSizePx],
            statistics: vec![TimeCourseStat::Mean, TimeCourseStat::Sd],
            normalize: TimeCourseNormalize::None,
            images: None,
            object_classes: None,
            t_stacks: None,
            z_stacks: None,
            max_tracks: Some(10),
            history: TrackHistory::Own,
        }
    }

    fn image_tracks(image: &str) -> ImageTracksFilter {
        ImageTracksFilter {
            image_rel_path: image.into(),
            z_stack: 0,
            max_points: 100,
        }
    }

    fn image_objects(image: &str) -> ImageObjectsFilter {
        ImageObjectsFilter {
            image_rel_path: image.into(),
            plane: plane(),
            value_column: Some(Column::AreaSizePx),
        }
    }

    /// The queries `remote.rs` doesn't compare give the same answers over
    /// the connection as read locally.
    #[test]
    fn the_planes_status_charts_and_tracks_of_a_remote_database_match_a_local_read() {
        let (_dir, db) = analyzed_database();
        let remote = connect(LocalBackend::default());
        let remote_db = remote.open_results(&db).unwrap();
        let local_db = LocalBackend::default().open_results(&db).unwrap();
        let image = local_db.get_images().unwrap()[0]
            .rel_path
            .to_string_lossy()
            .into_owned();

        assert_eq!(
            remote_db.get_planes().unwrap(),
            local_db.get_planes().unwrap()
        );
        assert!(!remote_db.get_planes().unwrap().is_empty());
        assert_eq!(remote_db.run_status().unwrap(), RunStatus::Finished);
        assert_eq!(
            remote_db.has_tracks().unwrap(),
            local_db.has_tracks().unwrap()
        );
        assert_eq!(
            remote_db.has_lineage().unwrap(),
            local_db.has_lineage().unwrap()
        );
        assert_eq!(
            json(remote_db.histogram(&histogram()).unwrap()),
            json(local_db.histogram(&histogram()).unwrap())
        );
        assert_eq!(
            json(remote_db.time_course(&time_course()).unwrap()),
            json(local_db.time_course(&time_course()).unwrap())
        );
        assert_eq!(
            json(remote_db.image_objects(&image_objects(&image)).unwrap()),
            json(local_db.image_objects(&image_objects(&image)).unwrap())
        );
        let summary = TrackSummaryFilter::default();
        assert_eq!(
            remote_db.track_summary(&summary).unwrap(),
            local_db.track_summary(&summary).unwrap()
        );
        assert_eq!(
            remote_db.image_tracks(&image_tracks(&image)).unwrap(),
            local_db.image_tracks(&image_tracks(&image)).unwrap()
        );
        assert_eq!(
            json(remote_db.get_object_classes().unwrap()),
            json(local_db.get_object_classes().unwrap())
        );
    }

    #[test]
    fn opening_a_file_that_is_no_database_reports_the_worker_s_error() {
        let remote = connect(LocalBackend::default());
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("notes.evadb");
        std::fs::write(&path, "not a database").unwrap();

        let Err(remote_error) = remote.open_results(&path) else {
            panic!("a text file must not open as results");
        };
        let Err(local_error) = LocalBackend::default().open_results(&path) else {
            panic!("a text file must not open as results");
        };

        // The worker's message, marked as the server's.
        let InternalErrors::Internal(text) = remote_error else {
            panic!("expected the server's error, got {remote_error:?}");
        };
        assert_eq!(text, format!("Server: {local_error}"));
    }

    #[test]
    fn a_database_opens_again_on_the_new_connection_after_a_reconnect() {
        let (_dir, db) = analyzed_database();
        let remote = connect(LocalBackend::default());
        let remote_db = remote.open_results(&db).unwrap();
        let before = json(remote_db.get_images().unwrap());

        remote.disconnect();
        wait_until_disconnected(&remote);
        remote.reconnect().unwrap();

        assert!(remote.is_connected());
        assert_eq!(json(remote_db.get_images().unwrap()), before);
        assert_eq!(
            remote_db.get_nr_of_z_stacks(),
            LocalBackend::default()
                .open_results(&db)
                .unwrap()
                .get_nr_of_z_stacks()
        );
    }

    #[test]
    fn queries_over_a_dropped_connection_fail_and_stack_counts_fall_back_to_one() {
        let (dir, db) = analyzed_database();
        let remote = connect(LocalBackend::default());
        let remote_db = remote.open_results(&db).unwrap();

        remote.disconnect();
        wait_until_disconnected(&remote);

        let lost = |result: Result<(), InternalErrors>| {
            let Err(InternalErrors::Io(text)) = result else {
                panic!("expected a lost connection, got {result:?}");
            };
            assert!(text.contains("Lost connection"), "{text}");
        };
        lost(remote_db.get_images().map(drop));
        lost(remote_db.run_status().map(drop));
        lost(remote_db.export(
            &ResultExport {
                output_dir: dir.path().join("export"),
                ..Default::default()
            },
            &AtomicBool::new(false),
            &mut |_, _, _| {},
        ));
        assert_eq!(remote_db.get_nr_of_z_stacks(), 1);
        assert_eq!(remote_db.get_nr_of_t_stacks(), 1);
    }

    fn broken(what: &str) -> InternalErrors {
        InternalErrors::Internal(format!("{what} is broken"))
    }

    /// A results database whose every query fails.
    struct BrokenResults;

    impl ResultsSource for BrokenResults {
        fn get_object_list(&self, _: &ListFilter) -> Result<DatabaseResult, InternalErrors> {
            Err(broken("object list"))
        }
        fn get_grouped_by_image(
            &self,
            _: &GroupedByImageFilter,
        ) -> Result<DatabaseResult, InternalErrors> {
            Err(broken("grouped by image"))
        }
        fn get_group_by_plate(
            &self,
            _: &PlateFilter,
            _: &View,
        ) -> Result<DatabaseResult, InternalErrors> {
            Err(broken("plate"))
        }
        fn get_group_by_well(
            &self,
            _: &WellFilter,
            _: &View,
        ) -> Result<DatabaseResult, InternalErrors> {
            Err(broken("well"))
        }
        fn get_image_heatmap(
            &self,
            _: &ImageHeatmapFilter,
            _: &View,
        ) -> Result<DatabaseResult, InternalErrors> {
            Err(broken("image heatmap"))
        }
        fn get_images(&self) -> Result<Vec<ImageEntry>, InternalErrors> {
            Err(broken("images"))
        }
        fn enable_image(&self, _: &str, _: bool) -> Result<(), InternalErrors> {
            Err(broken("enable image"))
        }
        fn get_object_classes(&self) -> Result<Vec<Class>, InternalErrors> {
            Err(broken("classes"))
        }
        fn get_available_columns(&self) -> Result<Vec<ColumnEntry>, InternalErrors> {
            Err(broken("columns"))
        }
        fn get_nr_of_z_stacks(&self) -> u32 {
            4
        }
        fn get_nr_of_t_stacks(&self) -> u32 {
            5
        }
        fn get_planes(&self) -> Result<Vec<PlaneFilter>, InternalErrors> {
            Err(broken("planes"))
        }
        fn run_status(&self) -> Result<RunStatus, InternalErrors> {
            Err(broken("run status"))
        }
        fn grouping_regex(&self, _: &Grouping) -> Result<String, InternalErrors> {
            Err(broken("grouping regex"))
        }
        fn boxplot(&self, _: &BoxplotFilter) -> Result<BoxplotResult, InternalErrors> {
            Err(broken("boxplot"))
        }
        fn histogram(&self, _: &HistogramFilter) -> Result<HistogramResult, InternalErrors> {
            Err(broken("histogram"))
        }
        fn scatter(&self, _: &ScatterFilter) -> Result<ScatterResult, InternalErrors> {
            Err(broken("scatter"))
        }
        fn time_course(&self, _: &TimeCourseFilter) -> Result<TimeCourseResult, InternalErrors> {
            Err(broken("time course"))
        }
        fn image_objects(
            &self,
            _: &ImageObjectsFilter,
        ) -> Result<Vec<ResultObject>, InternalErrors> {
            Err(broken("image objects"))
        }
        fn track_summary(
            &self,
            _: &TrackSummaryFilter,
        ) -> Result<Vec<TrackSummary>, InternalErrors> {
            Err(broken("track summary"))
        }
        fn has_tracks(&self) -> Result<bool, InternalErrors> {
            Err(broken("has tracks"))
        }
        fn has_lineage(&self) -> Result<bool, InternalErrors> {
            Err(broken("has lineage"))
        }
        fn image_tracks(&self, _: &ImageTracksFilter) -> Result<ImageTracks, InternalErrors> {
            Err(broken("image tracks"))
        }
        fn export(
            &self,
            export: &ResultExport,
            cancel: &AtomicBool,
            on_progress: ExportProgressFn,
        ) -> Result<(), InternalErrors> {
            on_progress("halfway", 1, 2);
            if export.output_dir != Path::new(WAIT_FOR_CANCEL) {
                return Err(broken("export"));
            }
            let deadline = Instant::now() + Duration::from_secs(10);
            while !cancel.load(std::sync::atomic::Ordering::SeqCst) && Instant::now() < deadline {
                std::thread::sleep(Duration::from_millis(1));
            }
            Err(InternalErrors::Cancelled)
        }
    }

    /// Where a [`BrokenResults`] export waits until it is cancelled instead
    /// of failing at once.
    const WAIT_FOR_CANCEL: &str = "/wait/for/cancel";

    /// `LocalBackend`, except that every results database is [`BrokenResults`].
    struct BrokenResultsBackend(LocalBackend);

    impl Backend for BrokenResultsBackend {
        fn start_analysis(&self, req: AnalysisRequest) -> Result<RunningJob, InternalErrors> {
            self.0.start_analysis(req)
        }
        fn start_preview(&self, req: PreviewRequest) -> Result<RunningJob, StartPreviewError> {
            self.0.start_preview(req)
        }
        fn start_training(
            &self,
            req: TrainingRequest,
        ) -> Result<RunningTraining, StartTrainingError> {
            self.0.start_training(req)
        }
        fn open_image(&self, path: &Path) -> Result<Arc<dyn ImageSource>, InternalErrors> {
            self.0.open_image(path)
        }
        fn open_results(&self, _: &Path) -> Result<Arc<dyn ResultsSource>, InternalErrors> {
            Ok(Arc::new(BrokenResults))
        }
        fn read_image_meta(&self, path: &Path) -> Result<ImageMeta, InternalErrors> {
            self.0.read_image_meta(path)
        }
        fn template_folders(&self) -> Result<TemplateFolders, InternalErrors> {
            self.0.template_folders()
        }
        fn files(&self) -> &dyn FileSystem {
            self.0.files()
        }
        fn description(&self) -> String {
            "broken results".into()
        }
        fn system_info(&self) -> Result<SystemInfo, InternalErrors> {
            self.0.system_info()
        }
        fn image_formats(&self) -> Vec<String> {
            self.0.image_formats()
        }
        fn load_app_settings(&self) -> Result<AppSettings, InternalErrors> {
            self.0.load_app_settings()
        }
        fn save_app_settings(&self, settings: &AppSettings) -> Result<(), InternalErrors> {
            self.0.save_app_settings(settings)
        }
    }

    /// `result` failed with the worker's "`what` is broken".
    fn failed_with<T>(result: Result<T, InternalErrors>, what: &str) {
        let Err(error) = result else {
            panic!("{what}: expected the worker's error");
        };
        let text = error.to_string();
        assert!(text.contains(&format!("{what} is broken")), "{text}");
    }

    #[test]
    fn every_failing_query_on_the_worker_comes_back_as_its_error() {
        let remote = connect(BrokenResultsBackend(LocalBackend::default()));
        let db = remote.open_results(Path::new("/data/any.evadb")).unwrap();
        let image = "a.tif";

        failed_with(db.get_images(), "images");
        failed_with(db.enable_image(image, true), "enable image");
        failed_with(db.get_object_classes(), "classes");
        failed_with(db.get_available_columns(), "columns");
        failed_with(db.get_planes(), "planes");
        failed_with(db.run_status(), "run status");
        failed_with(db.grouping_regex(&Grouping::Auto), "grouping regex");
        failed_with(db.histogram(&histogram()), "histogram");
        failed_with(db.time_course(&time_course()), "time course");
        failed_with(db.image_objects(&image_objects(image)), "image objects");
        failed_with(
            db.track_summary(&TrackSummaryFilter::default()),
            "track summary",
        );
        failed_with(db.has_tracks(), "has tracks");
        failed_with(db.has_lineage(), "has lineage");
        failed_with(db.image_tracks(&image_tracks(image)), "image tracks");
        failed_with(
            db.get_object_list(&ListFilter {
                plane: plane(),
                images: None,
                object_classes: None,
                columns: vec![Column::AreaSizePx],
                with_coloc_details: false,
                related_to: None,
                page: Pagination {
                    limit: 10,
                    after: None,
                },
                transpond_table: false,
            }),
            "object list",
        );
        failed_with(
            db.get_grouped_by_image(&GroupedByImageFilter {
                plane: plane(),
                images: None,
                object_classes: None,
                columns: vec![Column::Count],
                aggregation: vec![Aggregation::Avg],
                page: Pagination {
                    limit: 10,
                    after: None,
                },
                transpond_table: false,
            }),
            "grouped by image",
        );
        let plate = PlateFilter {
            plane: plane(),
            grouping: Grouping::Auto,
            aggregation: Aggregation::Avg,
            object_class: Default::default(),
            column: Column::AreaSizePx,
            color_schema: ColorSchema::default(),
            color_scale: ColorScale::Auto,
            plate_size: PlateSize::Auto,
        };
        failed_with(db.get_group_by_plate(&plate, &View::Heatmap), "plate");
        let well = WellFilter {
            plane: plane(),
            group_name: String::new(),
            grouping: Grouping::Auto,
            aggregation: Aggregation::Avg,
            object_class: Default::default(),
            column: Column::AreaSizePx,
            color_schema: ColorSchema::default(),
            color_scale: ColorScale::Auto,
            well_size: None,
            well_order: None,
        };
        failed_with(db.get_group_by_well(&well, &View::Heatmap), "well");
        let heatmap = ImageHeatmapFilter {
            plane: plane(),
            image_rel_path: image.into(),
            aggregation: Aggregation::Avg,
            object_class: Default::default(),
            column: Column::AreaSizePx,
            color_schema: ColorSchema::default(),
            color_scale: ColorScale::Auto,
            square_size: Some(64),
        };
        failed_with(
            db.get_image_heatmap(&heatmap, &View::Heatmap),
            "image heatmap",
        );
        failed_with(
            db.boxplot(&BoxplotFilter {
                plane: plane(),
                images: None,
                object_classes: None,
                column: Column::AreaSizePx,
                group_by: Default::default(),
                grouping: Default::default(),
            }),
            "boxplot",
        );
        failed_with(
            db.scatter(&ScatterFilter {
                plane: plane(),
                images: None,
                object_classes: None,
                x_column: Column::AreaSizePx,
                y_column: Column::PerimeterPx,
                max_points: Some(10),
            }),
            "scatter",
        );
        // Counts can't fail: the worker's own numbers.
        assert_eq!(db.get_nr_of_z_stacks(), 4);
        assert_eq!(db.get_nr_of_t_stacks(), 5);
    }

    #[test]
    fn a_failing_export_reports_its_progress_and_then_the_worker_s_error() {
        let remote = connect(BrokenResultsBackend(LocalBackend::default()));
        let db = remote.open_results(Path::new("/data/any.evadb")).unwrap();
        let mut progress = Vec::new();

        let result = db.export(
            &ResultExport::default(),
            &AtomicBool::new(false),
            &mut |message, current, total| progress.push((message.to_string(), current, total)),
        );

        failed_with(result, "export");
        assert_eq!(progress, [("halfway".to_string(), 1, 2)]);
    }

    /// A [`BrokenResults`] export that waits on the worker until cancelled.
    fn waiting_export(
        db: &dyn ResultsSource,
        cancel: &AtomicBool,
        on_progress: ExportProgressFn,
    ) -> Result<(), InternalErrors> {
        db.export(
            &ResultExport {
                output_dir: PathBuf::from(WAIT_FOR_CANCEL),
                ..Default::default()
            },
            cancel,
            on_progress,
        )
    }

    #[test]
    fn cancelling_a_running_export_reaches_the_worker_and_reports_cancelled() {
        let remote = connect(BrokenResultsBackend(LocalBackend::default()));
        let db = remote.open_results(Path::new("/data/any.evadb")).unwrap();
        let cancel = AtomicBool::new(false);
        let start = Instant::now();

        // Cancelled once the worker reports it is exporting.
        let result = waiting_export(db.as_ref(), &cancel, &mut |_, _, _| {
            cancel.store(true, std::sync::atomic::Ordering::SeqCst)
        });

        assert!(
            matches!(result, Err(InternalErrors::Cancelled)),
            "{result:?}"
        );
        // Stopped by the cancel, not by the export's own time limit.
        assert!(start.elapsed() < Duration::from_secs(5));
        assert!(remote.is_connected());
    }

    /// Cancelled at once: the cancel reaches the worker right behind the
    /// request, before its thread has started the export - and stops it.
    #[test]
    fn an_export_cancelled_right_away_is_cancelled_on_the_worker() {
        let remote = connect(BrokenResultsBackend(LocalBackend::default()));
        let db = remote.open_results(Path::new("/data/any.evadb")).unwrap();
        for _ in 0..20 {
            let start = Instant::now();
            let result = waiting_export(db.as_ref(), &AtomicBool::new(true), &mut |_, _, _| {});
            assert!(
                matches!(result, Err(InternalErrors::Cancelled)),
                "{result:?}"
            );
            assert!(
                start.elapsed() < Duration::from_secs(5),
                "stopped by the cancel, not by the export's own time limit"
            );
        }
    }

    #[test]
    fn an_export_whose_connection_drops_reports_the_lost_connection() {
        let remote = connect(BrokenResultsBackend(LocalBackend::default()));
        let db = remote.open_results(Path::new("/data/any.evadb")).unwrap();

        // Dropped while the worker is exporting.
        let result = waiting_export(db.as_ref(), &AtomicBool::new(false), &mut |_, _, _| {
            remote.disconnect()
        });

        let Err(InternalErrors::Io(text)) = result else {
            panic!("expected a lost connection, got {result:?}");
        };
        assert!(text.contains("Lost connection"), "{text}");
    }
}
