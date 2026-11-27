//! # network
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-11-27
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.
//!
//! Training a deep-learning network in this process: the project's
//! annotations of the Deep-Learning tab's classes gathered per image, the
//! core job ([`DeepLearningTrainingJob`]) built and started.

use crate::images::image_entry::ImageEntryExt;
use crate::images::image_list::ImageListExt;
use crate::jobs::CancelHandle;
use crate::jobs::local::join_job;
use crate::training::{
    NetworkTrainingRequest, RunningTraining, StartTrainingError, TrainedNetworkModel, TrainingItems,
};
use evanalyzer_cfg::core_types::{InternalErrors, ObjectClass};
use evanalyzer_cfg::settings::deep_learning_settings::DeepLearningClassSettings;
use evanalyzer_cfg::settings::object_settings::ObjectMetricSettings;
use evanalyzer_cfg::settings::project_settings::ProjectSettings;
use evanalyzer_core::{DeepLearningTrainingJob, TrainingImage};

/// The images with annotations of the training's classes (objects, training
/// areas, validation areas), each with just those annotations - the ones
/// excluded from training left out. An image of the project is opened by
/// its path joined to the project's image folder.
fn gather_images(
    project: &ProjectSettings,
    classes: &DeepLearningClassSettings,
) -> Result<Vec<TrainingImage>, InternalErrors> {
    let ours = |object: &ObjectMetricSettings| {
        let has = |class: ObjectClass| object.object_class.contains(&class);
        has(classes.objects) || has(classes.area) || classes.validation.is_some_and(has)
    };
    let mut images = Vec::new();
    for (rel_path, entry) in project.images.list.iter() {
        let active = entry.active_series(&project.images.settings);
        let Some(series) = entry.series.get(&active) else {
            continue;
        };
        let annotations: Vec<ObjectMetricSettings> = series
            .objects
            .iter()
            .filter(|o| !o.exclude_from_training && ours(o))
            .cloned()
            .collect();
        if annotations.is_empty() {
            continue;
        }
        let Some(root) = project.images.root.as_ref() else {
            return Err(InternalErrors::InvalidArgument(
                "The project has annotations but no image folder is set - set the project's \
                 image folder before training a network"
                    .into(),
            ));
        };
        images.push(TrainingImage {
            path: root.join(rel_path),
            series: active,
            labeled_objects: annotations,
        });
    }
    Ok(images)
}

/// Gathers the project's annotations for `req`'s classes and starts the
/// training. Fails up front with [`StartTrainingError::NoTrainingData`] when
/// no image has any of them.
pub(crate) fn start_network_training(
    req: NetworkTrainingRequest,
) -> Result<RunningTraining<TrainedNetworkModel>, StartTrainingError> {
    let images =
        gather_images(&req.project, &req.settings.classes).map_err(StartTrainingError::Failed)?;
    if images.is_empty() {
        return Err(StartTrainingError::NoTrainingData);
    }
    let items = TrainingItems::Images(images.len());
    let job = DeepLearningTrainingJob::from_settings(&req.settings, images, req.z_stack_handling)
        .map_err(StartTrainingError::Failed)?;
    let (handle, events, cancel) = job.run_async();
    Ok(RunningTraining::from_parts(
        events,
        CancelHandle::new(cancel),
        items,
        Box::new(move || {
            let trained = join_job(handle, "Network training worker")?;
            trained.model.to_bytes().map(TrainedNetworkModel)
        }),
    ))
}

#[cfg(test)]
mod tests;
