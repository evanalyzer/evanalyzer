use crate::args::AnalyzeArgs;
use evanalyzer_app::extensions::project_ext::{ProjectExt, load_project};
use evanalyzer_cfg::core_types::InternalErrors;
use evanalyzer_core::ProgressEvent;
use std::io::Write;
use std::path::PathBuf;
use std::time::Instant;

pub fn run(args: AnalyzeArgs) -> Result<(), InternalErrors> {
    let mut project = load_project(&args.project)?;

    if let Some(images_dir) = &args.images {
        project.images.root = Some(images_dir.clone());
        project.scan_image_folder_and_add();
    }

    let image_count = project.images.list.len();
    if image_count == 0 {
        return Err(InternalErrors::InvalidArgument(
            "Project has no images - pass --images <dir> or add images to the project first"
                .into(),
        ));
    }

    let project_dir = args
        .project
        .parent()
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("."));

    let enabled_pipelines = project.pipelines.iter().filter(|p| p.enabled).count();
    println!("Project:   {}", args.project.display());
    println!("Images:    {image_count}");
    println!("Pipelines: {enabled_pipelines} enabled");

    let job = evanalyzer_core::generate_analyze_job_from_project_settings(
        project.settings.clone(),
        project_dir,
    )?;
    let output_path = job.output_path.clone();

    let threads = args.threads.unwrap_or_else(|| {
        std::thread::available_parallelism()
            .map(|n| n.get().saturating_sub(1).max(1))
            .unwrap_or(1)
    });
    println!("Output:    {}", output_path.display());
    println!("Running with {threads} parallel thread(s)...\n");

    let start = Instant::now();
    let (handle, rx, _cancel) = job.run_async(threads);

    let mut failed = 0usize;
    let mut total = image_count;
    for event in rx {
        match event {
            ProgressEvent::Started { total: t } => total = t,
            ProgressEvent::ImageCompleted { index, total: t, path } => {
                total = t;
                print!("\r[{index}/{t}] {}          ", path.display());
                std::io::stdout().flush().ok();
            }
            ProgressEvent::ImageFailed { path } => {
                failed += 1;
                println!("\nFAILED: {}", path.display());
            }
            ProgressEvent::Finished => println!(),
            ProgressEvent::TilesScheduled { .. }
            | ProgressEvent::TileCompleted { .. }
            | ProgressEvent::BreakpointReached { .. } => {}
        }
    }

    let result = handle.join().map_err(|_| {
        InternalErrors::Internal("Pipeline worker thread panicked".into())
    })?;
    result?;

    println!(
        "Done: {total} image(s) analyzed in {:.1?} ({failed} failed)",
        start.elapsed()
    );
    println!("Results database written under: {}", output_path.display());
    Ok(())
}
