//! # project
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-06-26
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.

use crate::args::{CiteArgs, ProjectInfoArgs, ProjectNewArgs, ValidateArgs};
use evanalyzer_app::analysis::check_pipeline_channels;
use evanalyzer_app::backends::Backend;
use evanalyzer_app::exporter::cite_project;
use evanalyzer_app::modules::{ModuleProblem, Severity, check_project};
use evanalyzer_app::project::{ImageListExt, ProjectWithRuntime};
use evanalyzer_app::project::{ProjectExt, load_project};
use evanalyzer_cfg::PROJECT_FILE_EXTENSIONS;
use evanalyzer_cfg::core_types::{InternalErrors, StepId};
use evanalyzer_cfg::settings::meta_data::AuthorInformation;
use evanalyzer_cfg::settings::project_settings::ProjectSettings;
use serde_json::json;

pub fn run(args: ProjectInfoArgs, backend: &dyn Backend) -> Result<(), InternalErrors> {
    let project = load_project(backend.files(), &args.project)?;

    let pipelines: Vec<_> = project
        .pipelines
        .iter()
        .map(|p| {
            let name = p.name.clone();
            (name, p.enabled, p.steps.len())
        })
        .collect();
    let class_names: Vec<&str> = project
        .classification
        .classes()
        .iter()
        .map(|c| c.name.as_str())
        .collect();
    let reachable = project.does_project_images_exist(backend.files());

    if args.json {
        let out = json!({
            "project": args.project,
            "name": project.meta.name,
            "image_root": project.images.root,
            "images": project.images.list.len(),
            "reachable": reachable,
            "classes": class_names,
            "pipelines": pipelines.iter().map(|(name, enabled, steps)| json!({
                "name": name,
                "enabled": enabled,
                "steps": steps,
            })).collect::<Vec<_>>(),
        });
        println!("{}", serde_json::to_string_pretty(&out).unwrap());
        return Ok(());
    }

    println!("Project:    {}", args.project.display());
    if !project.meta.name.is_empty() {
        println!("Name:       {}", project.meta.name);
    }
    println!(
        "Image root: {}",
        project
            .images
            .root
            .as_ref()
            .map(|p| p.display().to_string())
            .unwrap_or_else(|| "(none)".into())
    );
    println!("Images:     {}", project.images.list.len());
    println!(
        "Reachable:  {}",
        if reachable {
            "yes"
        } else {
            "NO - the image root could not be verified, check --images / project.images.root"
        }
    );

    println!("\nClasses ({}):", class_names.len());
    for name in &class_names {
        println!("  - {name}");
    }

    println!("\nPipelines ({}):", pipelines.len());
    for (name, enabled, steps) in &pipelines {
        let state = if *enabled { "enabled" } else { "disabled" };
        println!("  - {name} [{state}] - {steps} step(s)");
    }

    Ok(())
}

/// Checks the project as a run would: every image it references against
/// disk (not just the single sample `does_project_images_exist` uses to
/// detect a relinked/moved root), the pipelines and modules as the GUI
/// editor checks them, and the channels the pipelines read against the
/// images'. Warnings are printed; errors (and missing images) fail.
pub fn run_validate(args: ValidateArgs, backend: &dyn Backend) -> Result<(), InternalErrors> {
    let project = load_project(backend.files(), &args.project)?;
    let settings = &project.settings;
    let root = settings.images.root.clone();

    let missing: Vec<_> = settings
        .images
        .list
        .keys()
        .map(|rel| match &root {
            Some(root) => root.join(rel),
            None => rel.clone(),
        })
        .filter(|abs| !abs.exists())
        .collect();

    let total = settings.images.list.len();
    println!("Project:    {}", args.project.display());
    println!(
        "Image root: {}",
        root.map(|p| p.display().to_string())
            .unwrap_or_else(|| "(none)".into())
    );
    println!("Checked {total} image(s)");
    if missing.is_empty() {
        println!("All images found on disk.");
    } else {
        println!("{} image(s) missing:", missing.len());
        for path in &missing {
            println!("  - {}", path.display());
        }
    }

    let problems = check_project(settings);
    let channels = check_pipeline_channels(settings).err();
    let errors = problems
        .iter()
        .filter(|p| p.severity == Severity::Error)
        .count()
        + usize::from(channels.is_some());
    let warnings = problems.len() + usize::from(channels.is_some()) - errors;
    println!(
        "Checked {} pipeline(s) and {} module(s)",
        settings.pipelines.len(),
        settings.modules.len()
    );
    if errors + warnings == 0 {
        println!("No problems found in the pipelines.");
    }
    for problem in &problems {
        let severity = match problem.severity {
            Severity::Error => "error",
            Severity::Warning => "warning",
        };
        let detail = if problem.detail.is_empty() {
            String::new()
        } else {
            format!(" ({})", problem.detail)
        };
        println!(
            "  {severity}: {}: {}{detail}",
            problem_place(settings, problem),
            problem.message
        );
    }
    if let Some(channels) = &channels {
        println!("  error: {channels}");
    }

    let mut failures = Vec::new();
    if !missing.is_empty() {
        failures.push(format!(
            "{} of {total} image(s) could not be found on disk",
            missing.len()
        ));
    }
    if errors > 0 {
        failures.push(format!("{errors} error(s) in the pipelines"));
    }
    if failures.is_empty() {
        return Ok(());
    }
    Err(InternalErrors::InvalidArgument(failures.join(", ")))
}

/// Where a problem is, as the GUI names it: "pipeline “Nuclei”, step 2",
/// "pipeline “Nuclei”" (its image source), "module “SEGMENT”" or
/// "project".
fn problem_place(settings: &ProjectSettings, problem: &ModuleProblem) -> String {
    if let Some((pipeline_id, step_id)) = problem.step {
        let Some(pipeline) = settings.pipelines.iter().find(|p| p.id == pipeline_id) else {
            return format!("pipeline {pipeline_id}");
        };
        let position = pipeline.steps.iter().position(|s| s.id == step_id);
        return match position {
            Some(i) if step_id != StepId::UNASSIGNED => {
                format!("pipeline “{}”, step {}", pipeline.name, i + 1)
            }
            _ => format!("pipeline “{}”", pipeline.name),
        };
    }
    match &problem.module {
        Some(key) => format!("module “{key}”"),
        None => "project".into(),
    }
}

/// Writes the same file as the GUI's "Cite project": a methods paragraph,
/// the pipelines' steps and the full references.
pub fn run_cite(args: CiteArgs, backend: &dyn Backend) -> Result<(), InternalErrors> {
    let project = load_project(backend.files(), &args.project)?;
    let out = args.out.unwrap_or_else(|| {
        args.project
            .parent()
            .unwrap_or(std::path::Path::new("."))
            .join("citation.md")
    });
    cite_project(backend.files(), &project.settings, &out)?;
    println!("Citation written to {}", out.display());
    Ok(())
}

/// Creates a new empty project file through `backend` (on the `--remote`
/// server's disk with `--remote`, as every project command), with the
/// author of `user` (this computer's user settings) if one is set. Refuses
/// to overwrite an existing file.
pub fn new_project(
    args: ProjectNewArgs,
    backend: &dyn Backend,
    user: &dyn Backend,
) -> Result<(), InternalErrors> {
    let mut path = args.project;
    if path.extension().and_then(|e| e.to_str()) != Some(PROJECT_FILE_EXTENSIONS) {
        path.set_extension(PROJECT_FILE_EXTENSIONS);
    }
    if backend.files().stat(&path)?.is_some() {
        return Err(InternalErrors::InvalidArgument(format!(
            "{} already exists - choose another name or delete it first",
            path.display()
        )));
    }

    let mut project = ProjectWithRuntime::default();
    project.meta.creation_time = chrono::Utc::now();
    if let Some(author) = user
        .load_app_settings()?
        .author
        .filter(|a| !a.full_name.trim().is_empty())
    {
        project.meta.authors.push(AuthorInformation {
            full_name: author.full_name,
            organization: author.organization,
        });
    }

    project.save_project_as(backend.files(), &path)?;
    println!("Project written to {}", path.display());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::commands::test_support::TempProjectFile;

    fn run(args: ProjectInfoArgs) -> Result<(), InternalErrors> {
        super::run(args, &LocalBackend::default())
    }

    fn run_validate(args: ValidateArgs) -> Result<(), InternalErrors> {
        super::run_validate(args, &LocalBackend::default())
    }
    use evanalyzer_app::backends::local::LocalBackend;
    use evanalyzer_cfg::settings::images_settings::ImageEntry;
    use evanalyzer_cfg::settings::project_settings::ProjectSettings;

    #[test]
    fn run_reports_ok_on_a_freshly_default_project() {
        let file = TempProjectFile::new(&ProjectSettings::default());

        let result = run(ProjectInfoArgs {
            project: file.path.clone(),
            json: false,
        });

        assert!(result.is_ok());
    }

    #[test]
    fn run_json_reports_ok_on_a_freshly_default_project() {
        let file = TempProjectFile::new(&ProjectSettings::default());

        let result = run(ProjectInfoArgs {
            project: file.path.clone(),
            json: true,
        });

        assert!(result.is_ok());
    }

    #[test]
    fn run_errors_when_the_project_file_does_not_exist() {
        let result = run(ProjectInfoArgs {
            project: std::path::PathBuf::from("/nonexistent/does_not_exist.evaproj"),
            json: false,
        });

        assert!(result.is_err());
    }

    #[test]
    fn cite_writes_the_methods_text_next_to_the_project_by_default() {
        let file = TempProjectFile::new(&ProjectSettings::default());

        super::run_cite(
            CiteArgs {
                project: file.path.clone(),
                out: None,
            },
            &LocalBackend::default(),
        )
        .unwrap();

        let written = file.path.parent().unwrap().join("citation.md");
        let content = std::fs::read_to_string(written).unwrap();
        assert!(content.contains("Images were analysed with EVAnalyzer"));
        assert!(content.contains("[@danmayr2026]"));
    }

    #[test]
    fn run_validate_succeeds_when_the_project_has_no_images() {
        let file = TempProjectFile::new(&ProjectSettings::default());

        let result = run_validate(ValidateArgs {
            project: file.path.clone(),
        });

        assert!(result.is_ok());
    }

    #[test]
    fn run_validate_fails_and_lists_missing_images() {
        let mut settings = ProjectSettings::default();
        settings.images.list.insert(
            std::path::PathBuf::from("does_not_exist.tif"),
            ImageEntry::default(),
        );
        let file = TempProjectFile::new(&settings);

        let result = run_validate(ValidateArgs {
            project: file.path.clone(),
        });

        let err = result.expect_err("expected the missing image to be reported as an error");
        let InternalErrors::InvalidArgument(msg) = err else {
            panic!("expected InvalidArgument, got {err:?}");
        };
        assert!(msg.contains("1 of 1"));
    }

    /// `validate` checks the pipelines too, not only the images: a
    /// measurement ID with a space is an error the GUI editor shows.
    #[test]
    fn run_validate_fails_on_a_pipeline_problem() {
        use evanalyzer_cfg::core_types::{ImageAddress, PipelineId, StepId};
        use evanalyzer_cfg::settings::pipeline_command::PipelineCommand;
        use evanalyzer_cfg::settings::pipeline_settings::{
            PipelineSettings, PipelineStepSettings, StepContent,
        };

        let mut command = PipelineCommand::default_for_key("measure_image_colocalization").unwrap();
        *command.measurement_id_mut().unwrap() = "bad name".into();
        let mut settings = ProjectSettings::default();
        settings.pipelines.push(PipelineSettings {
            id: PipelineId(1),
            name: "Nuclei".into(),
            description: None,
            image_source: ImageAddress::Channel(0),
            enabled: true,
            steps: vec![PipelineStepSettings {
                id: StepId(1),
                enabled: true,
                content: StepContent::Command(command),
            }],
        });
        let file = TempProjectFile::new(&settings);

        let result = run_validate(ValidateArgs {
            project: file.path.clone(),
        });

        assert!(
            result.is_err(),
            "validate passed a project whose measurement ID is invalid"
        );
    }

    /// A backend whose user folder (settings) is in `home`, with `author`
    /// saved in its settings.
    fn user_with(home: &std::path::Path, author: Option<(&str, &str)>) -> LocalBackend {
        use evanalyzer_app::global::{AppSettings, UserInformation};
        let backend = LocalBackend::default().with_home(home).unwrap();
        let mut settings = AppSettings::default();
        settings.author = author.map(|(full_name, organization)| UserInformation {
            full_name: full_name.into(),
            organization: organization.into(),
        });
        backend.save_app_settings(&settings).unwrap();
        backend
    }

    /// `project new` writes a project that loads, at the path with
    /// `.evaproj` added, created now, with the user's author.
    #[test]
    fn new_project_writes_a_loadable_project_with_the_users_author() {
        let dir = tempfile::tempdir().unwrap();
        let user = user_with(dir.path(), Some(("Ada Lovelace", "Analytical Engines")));
        let before = chrono::Utc::now();

        new_project(
            ProjectNewArgs {
                project: dir.path().join("varroa"),
            },
            &user,
            &user,
        )
        .unwrap();

        let path = dir.path().join("varroa.evaproj");
        let project = load_project(user.files(), &path).unwrap();
        assert!(project.pipelines.is_empty() && project.images.list.is_empty());
        assert!(project.meta.creation_time >= before);
        let [author] = &project.meta.authors[..] else {
            panic!("one author expected, got {:?}", project.meta.authors);
        };
        assert_eq!(
            (author.full_name.as_str(), author.organization.as_str()),
            ("Ada Lovelace", "Analytical Engines")
        );
    }

    /// Without an author in the user settings (or with an empty name) the
    /// project has none.
    #[test]
    fn new_project_without_an_author_in_the_settings_has_none() {
        for author in [None, Some(("  ", "Org"))] {
            let dir = tempfile::tempdir().unwrap();
            let user = user_with(dir.path(), author);
            let path = dir.path().join("p.evaproj");

            new_project(
                ProjectNewArgs {
                    project: path.clone(),
                },
                &user,
                &user,
            )
            .unwrap();

            let project = load_project(user.files(), &path).unwrap();
            assert!(project.meta.authors.is_empty(), "{author:?}");
        }
    }

    /// An existing file is never overwritten - also when the extension is
    /// left out of the argument.
    #[test]
    fn new_project_refuses_to_overwrite_an_existing_file() {
        let dir = tempfile::tempdir().unwrap();
        let user = user_with(dir.path(), None);
        let path = dir.path().join("p.evaproj");
        std::fs::write(&path, "keep me").unwrap();

        for given in [path.clone(), dir.path().join("p")] {
            let err = new_project(ProjectNewArgs { project: given }, &user, &user)
                .expect_err("an existing project must not be overwritten");
            assert!(err.to_string().contains("already exists"), "{err}");
        }
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "keep me");
    }
}
