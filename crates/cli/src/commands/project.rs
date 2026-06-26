use crate::args::ProjectInfoArgs;
use evanalyzer_app::extensions::project_ext::{ProjectExt, load_project};
use evanalyzer_cfg::core_types::InternalErrors;

pub fn run(args: ProjectInfoArgs) -> Result<(), InternalErrors> {
    let project = load_project(&args.project)?;

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
        if project.does_project_images_exist() {
            "yes"
        } else {
            "NO - the image root could not be verified, check --images / project.images.root"
        }
    );

    println!("\nClasses ({}):", project.classification.classes.len());
    for class in &project.classification.classes {
        println!("  - {}", class.name);
    }

    println!("\nPipelines ({}):", project.pipelines.len());
    for pipeline in &project.pipelines {
        let name = pipeline
            .name
            .clone()
            .unwrap_or_else(|| format!("Pipeline {}", pipeline.id.0));
        let state = if pipeline.enabled { "enabled" } else { "disabled" };
        println!("  - {name} [{state}] - {} step(s)", pipeline.steps.len());
    }

    Ok(())
}
