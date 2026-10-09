//! [`RemoteResults`]: a results database opened on the worker; every query
//! is one request. Survives a reconnect: opened again on the new connection.

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
