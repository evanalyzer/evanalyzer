use crate::args::{ProjectInfoArgs, ValidateArgs};
use evanalyzer_app::extensions::project_ext::{ProjectExt, load_project};
use evanalyzer_cfg::core_types::InternalErrors;
use serde_json::json;

pub fn run(args: ProjectInfoArgs) -> Result<(), InternalErrors> {
    let project = load_project(&args.project)?;

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
    let reachable = project.does_project_images_exist();

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

/// Checks every image referenced by the project against disk, rather than just the
/// single sample `does_project_images_exist` uses to detect a relinked/moved root.
pub fn run_validate(args: ValidateArgs) -> Result<(), InternalErrors> {
    let project = load_project(&args.project)?;
    let root = project.images.root.clone();

    let missing: Vec<_> = project
        .images
        .list
        .keys()
        .map(|rel| match &root {
            Some(root) => root.join(rel),
            None => rel.clone(),
        })
        .filter(|abs| !abs.exists())
        .collect();

    let total = project.images.list.len();
    println!("Project:    {}", args.project.display());
    println!(
        "Image root: {}",
        root.map(|p| p.display().to_string())
            .unwrap_or_else(|| "(none)".into())
    );
    println!("Checked {total} image(s)");

    if missing.is_empty() {
        println!("All images found on disk.");
        return Ok(());
    }

    println!("{} image(s) missing:", missing.len());
    for path in &missing {
        println!("  - {}", path.display());
    }
    Err(InternalErrors::InvalidArgument(format!(
        "{} of {total} image(s) could not be found on disk",
        missing.len()
    )))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::commands::test_support::TempProjectFile;
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
}
