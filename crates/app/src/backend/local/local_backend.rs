use crate::api::AnalysisRequest;
use crate::api::Backend;
use crate::api::FileSystem;
use crate::api::ImageMeta;
use crate::api::ImageSource;
use crate::api::PreviewRequest;
use crate::api::ResultsSource;
use crate::api::RunningJob;
use crate::api::RunningTraining;
use crate::api::StartPreviewError;
use crate::api::StartTrainingError;
use crate::api::TemplateFolders;
use crate::api::TrainingRequest;
use crate::backend::LocalFileSystem;
use crate::backend::local::image_reader::ReaderPool;
use crate::backend::local::results::LocalResults;
use evanalyzer_cfg::core_types::InternalErrors;
use evanalyzer_core::{ImageReader, ReadMode};
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// Every operation forwards to the `job`/`ai_learning` functions that hold
/// the actual logic. Optionally confined to a set of folders (a server's
/// `--root`s): then every path a request names - files, images, the project
/// and image folders - must lie inside them.
#[derive(Debug, Default)]
pub struct LocalBackend {
    files: LocalFileSystem,
}

impl LocalBackend {
    /// The template folders stay reachable too, so clients can list and
    /// save templates.
    pub fn restricted_to(roots: &[PathBuf]) -> Result<Self, InternalErrors> {
        let folders = local_template_folders();
        let template_folders: Vec<PathBuf> = [folders.user, folders.bundled]
            .into_iter()
            .filter(|folder| folder.is_dir())
            .collect();
        Ok(Self {
            files: LocalFileSystem::restricted_to(roots)?.also_allowing(&template_folders)?,
        })
    }

    /// The paths a project makes the backend read or write.
    fn check_project_paths(
        &self,
        project_path: Option<&Path>,
        settings: &evanalyzer_cfg::settings::project_settings::ProjectSettings,
    ) -> Result<(), InternalErrors> {
        if !self.files.is_restricted() {
            return Ok(());
        }
        if let Some(project_path) = project_path {
            self.files.check(project_path)?;
        }
        if let Some(root) = &settings.images.root {
            self.files.check(root)?;
        }
        Ok(())
    }
}

impl Backend for LocalBackend {
    fn start_analysis(&self, req: AnalysisRequest) -> Result<RunningJob, InternalErrors> {
        self.check_project_paths(Some(&req.project_path), &req.settings)?;
        super::job::start_analysis(req.settings, req.project_path, req.job_name, req.threads)
    }

    fn start_preview(&self, req: PreviewRequest) -> Result<RunningJob, StartPreviewError> {
        self.check_project_paths(Some(&req.project_path), &req.settings)?;
        super::job::start_preview(req)
    }

    fn start_training(&self, req: TrainingRequest) -> Result<RunningTraining, StartTrainingError> {
        self.check_project_paths(None, &req.project)
            .map_err(StartTrainingError::Failed)?;
        super::training::start_training(&req.project, req.settings, req.pixel_params)
    }

    fn open_image(&self, path: &Path) -> Result<Arc<dyn ImageSource>, InternalErrors> {
        let path = self.files.check(path)?;
        Ok(Arc::new(ReaderPool::open(&path)?))
    }

    fn open_results(&self, path: &Path) -> Result<Arc<dyn ResultsSource>, InternalErrors> {
        let path = self.files.check(path)?;
        Ok(Arc::new(LocalResults::open(path)?))
    }

    fn read_image_meta(&self, path: &Path) -> Result<ImageMeta, InternalErrors> {
        let path = self.files.check(path)?;
        let reader = ImageReader::new(&path, ReadMode::SplitChannels)?;
        Ok(reader.get_image_meta().clone())
    }

    fn template_folders(&self) -> Result<TemplateFolders, InternalErrors> {
        Ok(local_template_folders())
    }

    fn files(&self) -> &dyn FileSystem {
        &self.files
    }

    fn description(&self) -> String {
        "local".into()
    }

    fn is_remote(&self) -> bool {
        false
    }
}

fn local_template_folders() -> TemplateFolders {
    TemplateFolders {
        user: crate::backend::local::templates::get_user_templates_folder(),
        bundled: crate::backend::local::templates::get_app_templates_folder(),
    }
}
