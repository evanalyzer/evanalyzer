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
            let name = p.name.clone().unwrap_or_else(|| format!("Pipeline {}", p.id.0));
            (name, p.enabled, p.steps.len())
        })
        .collect();
    let class_names: Vec<&str> = project.classification.classes.iter().map(|c| c.name.as_str()).collect();
    let reachable = project.does_project_images_exist();

    if args.json {
        let out = json!({
            "project": args.project,
            "name": project.metadata.name,
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
    if !project.metadata.name.is_empty() {
        println!("Name:       {}", project.metadata.name);
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
        root.map(|p| p.display().to_string()).unwrap_or_else(|| "(none)".into())
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
