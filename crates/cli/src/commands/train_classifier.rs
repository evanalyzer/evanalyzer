use crate::args::{TrainClassifierArgs, ZStackHandlingArg};
use evanalyzer_app::ai_learning::{PixelTrainingParams, TrainingJob, build_training_job, save_trained_model};
use evanalyzer_app::extensions::project_ext::load_project;
use evanalyzer_cfg::core_types::InternalErrors;
use evanalyzer_cfg::settings::ai_learning_settings::AiLearningSettings;
use evanalyzer_cfg::settings::images_settings::ZStackHandling;
use evanalyzer_core::TrainingProgressEvent;
use std::io::Write;
use std::path::PathBuf;
use std::sync::atomic::Ordering;
use std::time::Instant;

pub fn run(args: TrainClassifierArgs) -> Result<(), InternalErrors> {
    let project = load_project(&args.project)?;

    let settings_json = std::fs::read_to_string(&args.settings).map_err(|e| {
        InternalErrors::Io(format!(
            "Could not read settings file '{}': {e}",
            args.settings.display()
        ))
    })?;
    let settings: AiLearningSettings = serde_json::from_str(&settings_json).map_err(|e| {
        InternalErrors::InvalidArgument(format!(
            "Could not parse '{}' as AiLearningSettings: {e}",
            args.settings.display()
        ))
    })?;

    let model_name = args
        .model_name
        .clone()
        .unwrap_or_else(|| settings.metadata.name.clone());
    if model_name.trim().is_empty() {
        return Err(InternalErrors::InvalidArgument(
            "No model name given - pass --model-name or set metadata.name in --settings".into(),
        ));
    }

    let project_dir = args
        .project
        .parent()
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("."));

    let pixel_params = PixelTrainingParams {
        channel: args.channel,
        t_stack: args.t_stack,
        z_stack_handling: to_z_stack_handling(args.z_stack_handling),
    };

    let job = build_training_job(&project.settings, settings, pixel_params)?;
    let (item_kind, item_count) = match &job {
        TrainingJob::Pixel(j) => ("labeled image(s)", j.images.len()),
        TrainingJob::Object(j) => ("labeled object(s)", j.objects.len()),
    };
    if item_count == 0 {
        return Err(InternalErrors::InvalidArgument(
            "No labeled training data found in the project - assign a class to at least one object first".into(),
        ));
    }

    println!("Project:   {}", args.project.display());
    println!("Settings:  {}", args.settings.display());
    println!("Training:  {item_count} {item_kind}");

    let start = Instant::now();
    let (handle, rx, cancel) = job.run_async();

    if let Err(e) = ctrlc::set_handler(move || {
        eprintln!("\nCancelling...");
        cancel.store(true, Ordering::SeqCst);
    }) {
        eprintln!("Warning: could not install Ctrl+C handler: {e}");
    }

    for event in rx {
        print_training_progress(event);
    }

    let classifier = handle
        .join()
        .map_err(|_| InternalErrors::Internal("Training worker thread panicked".into()))??;

    let output_path = save_trained_model(&classifier, &project_dir, &model_name)?;

    println!(
        "\nDone: trained in {:.1?}. Model saved to: {}",
        start.elapsed(),
        output_path.display()
    );
    Ok(())
}

fn to_z_stack_handling(arg: ZStackHandlingArg) -> ZStackHandling {
    match arg {
        ZStackHandlingArg::SingleStack => ZStackHandling::SingleStack,
        ZStackHandlingArg::AllStacks => ZStackHandling::AllStacks,
        ZStackHandlingArg::MaxIntensity => ZStackHandling::MaxIntensity,
        ZStackHandlingArg::MinIntensity => ZStackHandling::MinIntensity,
        ZStackHandlingArg::AvgIntensity => ZStackHandling::AvgIntensity,
        ZStackHandlingArg::SumIntensity => ZStackHandling::SumIntensity,
        ZStackHandlingArg::TakeTheMiddle => ZStackHandling::TakeTheMiddle,
    }
}

fn print_training_progress(event: TrainingProgressEvent) {
    match event {
        TrainingProgressEvent::Started { total } => {
            println!("Started: {total} item(s) to process");
        }
        TrainingProgressEvent::ImageTilesScheduled {
            image_index,
            total_tiles,
        } => {
            print!("\rImage {image_index}: {total_tiles} tile(s) scheduled          ");
            std::io::stdout().flush().ok();
        }
        TrainingProgressEvent::TileProcessed {
            image_index,
            tile_index,
            total_tiles,
        } => {
            print!("\rImage {image_index}: tile {tile_index}/{total_tiles} processed          ");
            std::io::stdout().flush().ok();
        }
        TrainingProgressEvent::ItemCompleted { index, total } => {
            print!("\r[{index}/{total}] processed          ");
            std::io::stdout().flush().ok();
        }
        TrainingProgressEvent::ImageFailed { path } => {
            println!("\nFAILED to read image: {}", path.display());
        }
        TrainingProgressEvent::ObjectSkipped { index, reason } => {
            println!("\nSkipped object {index}: {reason}");
        }
        TrainingProgressEvent::Training => {
            println!("\nFitting model...");
        }
        TrainingProgressEvent::Finished => {
            println!();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::commands::test_support::TempProjectFile;
    use evanalyzer_cfg::settings::ai_learning_object_settings::AiLearningObjectFeatureSettings;
    use evanalyzer_cfg::settings::ai_learning_settings::{
        AiLearningBackendSettings, AiLearningClassifierSettings,
    };
    use evanalyzer_cfg::settings::project_settings::ProjectSettings;

    fn empty_object_settings() -> AiLearningSettings {
        AiLearningSettings {
            metadata: Default::default(),
            backend: AiLearningBackendSettings::RandomForest(Default::default()),
            classifier: AiLearningClassifierSettings::Object {
                feature_spec: AiLearningObjectFeatureSettings { metrics: vec![] },
                class_labels: vec![],
            },
        }
    }

    #[test]
    fn run_errors_when_the_project_file_does_not_exist() {
        let dir = tempfile::tempdir().unwrap();
        let settings_path = dir.path().join("settings.json");
        std::fs::write(
            &settings_path,
            serde_json::to_string(&empty_object_settings()).unwrap(),
        )
        .unwrap();

        let result = run(TrainClassifierArgs {
            project: PathBuf::from("/nonexistent/does_not_exist.evaproj"),
            settings: settings_path,
            model_name: Some("test-model".into()),
            channel: 0,
            t_stack: 0,
            z_stack_handling: ZStackHandlingArg::SingleStack,
        });

        assert!(result.is_err());
    }

    #[test]
    fn run_rejects_a_project_with_no_labeled_objects() {
        let file = TempProjectFile::new(&ProjectSettings::default());
        let settings_path = file.path.parent().unwrap().join("settings.json");
        std::fs::write(
            &settings_path,
            serde_json::to_string(&empty_object_settings()).unwrap(),
        )
        .unwrap();

        let result = run(TrainClassifierArgs {
            project: file.path.clone(),
            settings: settings_path,
            model_name: Some("test-model".into()),
            channel: 0,
            t_stack: 0,
            z_stack_handling: ZStackHandlingArg::SingleStack,
        });

        let err = result.expect_err("an empty project has no labeled objects to train from");
        let InternalErrors::InvalidArgument(msg) = err else {
            panic!("expected InvalidArgument, got {err:?}");
        };
        assert!(msg.contains("No labeled training data"));
    }

    #[test]
    fn run_errors_when_no_model_name_is_available() {
        let file = TempProjectFile::new(&ProjectSettings::default());
        let settings_path = file.path.parent().unwrap().join("settings.json");
        std::fs::write(
            &settings_path,
            serde_json::to_string(&empty_object_settings()).unwrap(),
        )
        .unwrap();

        let result = run(TrainClassifierArgs {
            project: file.path.clone(),
            settings: settings_path,
            model_name: None,
            channel: 0,
            t_stack: 0,
            z_stack_handling: ZStackHandlingArg::SingleStack,
        });

        let err = result.expect_err("no --model-name and empty metadata.name must be rejected");
        let InternalErrors::InvalidArgument(msg) = err else {
            panic!("expected InvalidArgument, got {err:?}");
        };
        assert!(msg.contains("No model name"));
    }
}
