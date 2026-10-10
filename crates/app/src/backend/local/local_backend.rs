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
use crate::api::SystemInfo;
use crate::api::TemplateFolders;
use crate::api::TrainingRequest;
use crate::backend::LocalFileSystem;
use crate::backend::local::image_reader::ReaderPool;
use crate::backend::local::results::LocalResults;
use crate::workspace::settings::{self, AppSettings};
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
    /// The user folder (templates) when set explicitly - a worker's
    /// `--home`, see [`Self::with_home`]. `None`: this machine's user's.
    user_folder: Option<PathBuf>,
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
            user_folder: None,
        })
    }

    /// Keeps the user folder (templates) in `home` - a worker serving a
    /// logged-in user - instead of this process's account. Its templates
    /// folder stays reachable when restricted to `--root`s.
    pub fn with_home(mut self, home: &Path) -> Result<Self, InternalErrors> {
        let user_folder = settings::user_folder_in(home);
        let templates = super::templates::user_templates_folder_in(&user_folder);
        if self.files.is_restricted() {
            self.files = std::mem::take(&mut self.files).also_allowing(&[templates])?;
        }
        self.user_folder = Some(user_folder);
        Ok(self)
    }

    /// Where this backend keeps the user's folder (templates, settings):
    /// the one set by [`Self::with_home`], else this account's.
    fn user_folder(&self) -> PathBuf {
        self.user_folder
            .clone()
            .unwrap_or_else(settings::get_user_folder)
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
    fn system_info(&self) -> Result<SystemInfo, InternalErrors> {
        let (cpu_cores, ram_total_bytes) = super::system::cpu_ram_diagnostics();
        Ok(SystemInfo {
            app_version: env!("CARGO_PKG_VERSION").to_string(),
            os: std::env::consts::OS.to_string(),
            cpu_cores,
            ram_total_bytes,
            cuda_available: super::system::cuda_is_available(),
        })
    }

    fn load_app_settings(&self) -> Result<AppSettings, InternalErrors> {
        Ok(settings::load_app_settings_from(
            &settings::settings_file_in(&self.user_folder()),
        ))
    }

    fn save_app_settings(&self, app_settings: &AppSettings) -> Result<(), InternalErrors> {
        let path = settings::settings_file_in(&self.user_folder());
        settings::save_app_settings_to(&path, app_settings)
            .map_err(|e| InternalErrors::Io(format!("Could not save {}: {e}", path.display())))
    }

    fn image_formats(&self) -> Vec<String> {
        super::system::SUPPORTED_IMAGE_FORMATS
            .iter()
            .map(|format| format.to_string())
            .collect()
    }

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
        Ok(match &self.user_folder {
            Some(user_folder) => TemplateFolders {
                user: super::templates::user_templates_folder_in(user_folder),
                bundled: super::templates::get_app_templates_folder(),
            },
            None => local_template_folders(),
        })
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

#[cfg(test)]
mod home_tests {
    use super::*;

    #[test]
    fn a_worker_s_templates_live_in_the_user_s_home() {
        let home = tempfile::tempdir().unwrap();

        let backend = LocalBackend::default().with_home(home.path()).unwrap();

        let templates = backend.template_folders().unwrap().user;
        assert!(
            templates.starts_with(home.path()),
            "{}",
            templates.display()
        );
        assert!(templates.ends_with("evanalyzer/templates"));
        assert!(templates.is_dir(), "created on demand");
    }

    #[test]
    fn a_worker_keeps_the_user_s_app_settings_in_their_home() {
        let home = tempfile::tempdir().unwrap();
        let backend = LocalBackend::default().with_home(home.path()).unwrap();
        assert!(
            !backend.load_app_settings().unwrap().dark_mode,
            "defaults first"
        );

        let settings = AppSettings {
            dark_mode: true,
            ..Default::default()
        };
        backend.save_app_settings(&settings).unwrap();

        let file = settings::settings_file_in(&settings::user_folder_in(home.path()));
        assert!(file.is_file(), "{}", file.display());
        assert!(backend.load_app_settings().unwrap().dark_mode);
    }

    #[test]
    fn a_restricted_worker_can_still_reach_its_templates_folder() {
        let home = tempfile::tempdir().unwrap();
        let data = tempfile::tempdir().unwrap();

        let backend = LocalBackend::restricted_to(&[data.path().to_path_buf()])
            .unwrap()
            .with_home(home.path())
            .unwrap();

        let templates = backend.template_folders().unwrap().user;
        backend
            .files()
            .write_file(&templates.join("t.evaproj.template"), b"{}")
            .expect("templates folder is reachable");
        assert!(
            backend
                .files()
                .write_file(&home.path().join("outside.txt"), b"x")
                .is_err(),
            "the rest of the home stays outside the allowed folders"
        );
    }
}

#[cfg(test)]
mod format_tests {
    use super::*;

    /// The workspace builds core with the GPL Bio-Formats readers
    /// (`bioformats-gpl` in the workspace Cargo.toml); their file extensions
    /// must then be offered too, or the file dialog and folder scans silently
    /// skip every VSI/CZI/ND2/LIF image. If the GPL readers are switched off
    /// on purpose, this test goes with them.
    #[test]
    fn the_gpl_reader_formats_are_offered() {
        let formats = LocalBackend::default().image_formats();
        for format in ["tif", "vsi", "czi", "nd2", "lif"] {
            assert!(
                formats.iter().any(|f| f == format),
                "{format} missing in {formats:?}"
            );
        }
    }
}
