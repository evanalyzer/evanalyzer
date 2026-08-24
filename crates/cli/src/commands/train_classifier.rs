use crate::args::{TrainClassifierArgs, ZStackHandlingArg};
use evanalyzer_app::ai_learning::{
    PixelTrainingParams, TrainingJob, build_training_job, save_trained_model,
};
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
    let raw: serde_json::Value = serde_json::from_str(&settings_json).map_err(|e| {
        InternalErrors::InvalidArgument(format!(
            "Could not parse '{}' as AiLearningSettings: {e}",
            args.settings.display()
        ))
    })?;
    let settings: AiLearningSettings = evanalyzer_cfg::load_ai_learning_settings(raw)
        .map_err(|e| InternalErrors::InvalidArgument(e.to_string()))?;

    let model_name = args
        .model_name
        .clone()
        .unwrap_or_else(|| settings.meta.name.clone());
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
        TrainingProgressEvent::Epoch {
            epoch,
            total_epochs,
            train_loss,
            val_loss,
        } => {
            match val_loss {
                Some(v) => print!(
                    "\rEpoch {}/{total_epochs} - train loss {train_loss:.4}, val loss {v:.4}          ",
                    epoch + 1
                ),
                None => print!(
                    "\rEpoch {}/{total_epochs} - train loss {train_loss:.4}          ",
                    epoch + 1
                ),
            }
            std::io::stdout().flush().ok();
        }
        TrainingProgressEvent::Finished { stats } => {
            println!("\n{stats:?}");
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
            schema_version: evanalyzer_cfg::CURRENT_AI_LEARNING_SETTINGS_SCHEMA_VERSION,
            meta: Default::default(),
            backend: AiLearningBackendSettings::RandomForest(Default::default()),
            classifier: AiLearningClassifierSettings::Object {
                feature_spec: AiLearningObjectFeatureSettings { metrics: vec![] },
                class_labels: vec![],
            },
        }
    }

    /// A project with two labeled objects (one per class) in one image -
    /// enough for `gather_labeled_objects`/`ObjectTrainingJob` to have real
    /// data to fit a classifier from, with no image files needed (object-
    /// classifier training reads only already-computed metrics).
    fn project_with_two_labeled_objects() -> ProjectSettings {
        use evanalyzer_cfg::core_types::ObjectClass;
        use evanalyzer_cfg::settings::classification_settings::Class;
        use evanalyzer_cfg::settings::images_settings::{ImageEntry, SeriesSettings};
        use evanalyzer_cfg::settings::object_settings::ObjectMetricSettings;

        let mut project = ProjectSettings::default();
        project.classification.classes_mut().push(Class {
            id: ObjectClass::Valid(1),
            name: "A".into(),
            ..Default::default()
        });
        project.classification.classes_mut().push(Class {
            id: ObjectClass::Valid(2),
            name: "B".into(),
            ..Default::default()
        });

        let mut series = SeriesSettings::default();
        series.objects.push(ObjectMetricSettings {
            area: 10,
            object_class: [ObjectClass::Valid(1)].into(),
            ..Default::default()
        });
        series.objects.push(ObjectMetricSettings {
            area: 1000,
            object_class: [ObjectClass::Valid(2)].into(),
            ..Default::default()
        });
        let mut entry = ImageEntry::default();
        entry.series.insert(0, series);
        project.images.list.insert(PathBuf::from("img.tif"), entry);

        project
    }

    fn two_class_object_settings() -> AiLearningSettings {
        use evanalyzer_cfg::core_types::ObjectClass;
        use evanalyzer_cfg::settings::ai_learning_object_settings::ObjectMetric;
        use evanalyzer_cfg::settings::ai_learning_settings::ObjectClassLabel;

        AiLearningSettings {
            schema_version: evanalyzer_cfg::CURRENT_AI_LEARNING_SETTINGS_SCHEMA_VERSION,
            meta: Default::default(),
            backend: AiLearningBackendSettings::RandomForest(Default::default()),
            classifier: AiLearningClassifierSettings::Object {
                feature_spec: AiLearningObjectFeatureSettings {
                    metrics: vec![ObjectMetric::Area],
                },
                class_labels: vec![
                    ObjectClassLabel {
                        class: ObjectClass::Valid(1),
                        name: "A".into(),
                    },
                    ObjectClassLabel {
                        class: ObjectClass::Valid(2),
                        name: "B".into(),
                    },
                ],
            },
        }
    }

    #[test]
    fn run_trains_and_saves_a_model_end_to_end() {
        let file = TempProjectFile::new(&project_with_two_labeled_objects());
        let settings_path = file.path.parent().unwrap().join("settings.json");
        std::fs::write(
            &settings_path,
            serde_json::to_string(&two_class_object_settings()).unwrap(),
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

        assert!(
            result.is_ok(),
            "expected training to succeed: {:?}",
            result.err()
        );
        let model_path = file
            .path
            .parent()
            .unwrap()
            .join("models")
            .join("test-model.evamodel");
        assert!(
            model_path.exists(),
            "trained model must be saved to <project_dir>/models/<name>.evamodel"
        );
    }

    #[test]
    fn run_falls_back_to_settings_metadata_name_when_model_name_is_not_given() {
        let file = TempProjectFile::new(&project_with_two_labeled_objects());
        let settings_path = file.path.parent().unwrap().join("settings.json");
        let mut settings = two_class_object_settings();
        settings.meta.name = "from-metadata".into();
        std::fs::write(&settings_path, serde_json::to_string(&settings).unwrap()).unwrap();

        let result = run(TrainClassifierArgs {
            project: file.path.clone(),
            settings: settings_path,
            model_name: None,
            channel: 0,
            t_stack: 0,
            z_stack_handling: ZStackHandlingArg::SingleStack,
        });

        assert!(
            result.is_ok(),
            "expected training to succeed: {:?}",
            result.err()
        );
        let model_path = file
            .path
            .parent()
            .unwrap()
            .join("models")
            .join("from-metadata.evamodel");
        assert!(
            model_path.exists(),
            "--model-name omitted must fall back to settings.meta.name"
        );
    }

    #[test]
    fn print_training_progress_does_not_panic_for_any_event_variant() {
        // Purely a "doesn't panic" smoke test - `print_training_progress`
        // only formats to stdout, so there's nothing else to assert on any
        // one event; the end-to-end tests above exercise it for real through
        // a live progress channel.
        print_training_progress(TrainingProgressEvent::Started { total: 3 });
        print_training_progress(TrainingProgressEvent::ImageTilesScheduled {
            image_index: 0,
            total_tiles: 4,
        });
        print_training_progress(TrainingProgressEvent::TileProcessed {
            image_index: 0,
            tile_index: 1,
            total_tiles: 4,
        });
        print_training_progress(TrainingProgressEvent::ItemCompleted { index: 1, total: 3 });
        print_training_progress(TrainingProgressEvent::ImageFailed {
            path: PathBuf::from("broken.tif"),
        });
        print_training_progress(TrainingProgressEvent::ObjectSkipped {
            index: 2,
            reason: "ambiguous class".into(),
        });
        print_training_progress(TrainingProgressEvent::Epoch {
            epoch: 4,
            total_epochs: 5,
            train_loss: 0.1,
            val_loss: Some(0.2),
        });
        print_training_progress(TrainingProgressEvent::Training);
        print_training_progress(TrainingProgressEvent::Finished {
            stats: evanalyzer_core::TrainingStats::RandomForest {
                n_trees: 10,
                n_samples: 100,
            },
        });
    }

    #[test]
    fn to_z_stack_handling_maps_every_variant() {
        assert!(matches!(
            to_z_stack_handling(ZStackHandlingArg::SingleStack),
            ZStackHandling::SingleStack
        ));
        assert!(matches!(
            to_z_stack_handling(ZStackHandlingArg::AllStacks),
            ZStackHandling::AllStacks
        ));
        assert!(matches!(
            to_z_stack_handling(ZStackHandlingArg::MaxIntensity),
            ZStackHandling::MaxIntensity
        ));
        assert!(matches!(
            to_z_stack_handling(ZStackHandlingArg::MinIntensity),
            ZStackHandling::MinIntensity
        ));
        assert!(matches!(
            to_z_stack_handling(ZStackHandlingArg::AvgIntensity),
            ZStackHandling::AvgIntensity
        ));
        assert!(matches!(
            to_z_stack_handling(ZStackHandlingArg::SumIntensity),
            ZStackHandling::SumIntensity
        ));
        assert!(matches!(
            to_z_stack_handling(ZStackHandlingArg::TakeTheMiddle),
            ZStackHandling::TakeTheMiddle
        ));
    }

    #[test]
    fn run_errors_when_the_settings_file_does_not_exist() {
        let file = TempProjectFile::new(&ProjectSettings::default());

        let result = run(TrainClassifierArgs {
            project: file.path.clone(),
            settings: file.path.parent().unwrap().join("does_not_exist.json"),
            model_name: Some("test-model".into()),
            channel: 0,
            t_stack: 0,
            z_stack_handling: ZStackHandlingArg::SingleStack,
        });

        let err = result.expect_err("a missing settings file must be reported, not panic");
        let InternalErrors::Io(msg) = err else {
            panic!("expected Io, got {err:?}");
        };
        assert!(msg.contains("Could not read settings file"));
    }

    #[test]
    fn run_errors_when_the_settings_file_is_malformed_json() {
        let file = TempProjectFile::new(&ProjectSettings::default());
        let settings_path = file.path.parent().unwrap().join("settings.json");
        std::fs::write(&settings_path, b"{ this is not valid json").unwrap();

        let result = run(TrainClassifierArgs {
            project: file.path.clone(),
            settings: settings_path,
            model_name: Some("test-model".into()),
            channel: 0,
            t_stack: 0,
            z_stack_handling: ZStackHandlingArg::SingleStack,
        });

        let err = result.expect_err("malformed settings JSON must be reported, not panic");
        let InternalErrors::InvalidArgument(msg) = err else {
            panic!("expected InvalidArgument, got {err:?}");
        };
        assert!(msg.contains("Could not parse"));
    }

    #[test]
    fn run_errors_when_model_name_is_whitespace_only() {
        // `--model-name` is `Some`, but trims to nothing - same rejection as
        // not passing it at all (`run_errors_when_no_model_name_is_available`),
        // exercised separately since it's a different branch of the
        // `unwrap_or_else` upstream of the shared `.trim().is_empty()` check.
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
            model_name: Some("   ".into()),
            channel: 0,
            t_stack: 0,
            z_stack_handling: ZStackHandlingArg::SingleStack,
        });

        let err = result.expect_err("a whitespace-only --model-name must be rejected");
        let InternalErrors::InvalidArgument(msg) = err else {
            panic!("expected InvalidArgument, got {err:?}");
        };
        assert!(msg.contains("No model name"));
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
