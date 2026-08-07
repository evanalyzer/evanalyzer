use crate::extensions::classification_ext::ClassificationExt;
use crate::extensions::object_ext::ObjectExt;
use crate::extensions::utils::{get_relative_key, is_in_root, wavelength_to_rgb_u32};
use crate::project_owner::{ProjectTmpSettings, ProjectWithRuntime};
use bitvec::{order::Lsb0, vec::BitVec};
use evanalyzer_cfg::core_types::ImageAddress;
use evanalyzer_cfg::core_types::{InternalErrors, ObjectId, PipelineId};
use evanalyzer_cfg::settings::classification_settings::ClassificationSettings;
use evanalyzer_cfg::settings::images_settings::{
    ChannelSettings, HistogramSettings, PixelSizeSettings, TStackSettings, ZStackSettings,
};
use evanalyzer_cfg::settings::meta_data::MetaData;
use evanalyzer_cfg::settings::pipeline_settings::PipelineSettings;
use evanalyzer_cfg::settings::templates::{PipelineTemplate, ProjectTemplate};
use evanalyzer_cfg::{
    PIPELINE_EXTENSIONS, PROJECT_FILE_EXTENSIONS, PROJECT_FILE_TEMPLATE_EXTENSIONS,
};
use evanalyzer_cfg::{
    core_types::ObjectClass,
    settings::{
        classification_settings::Class,
        images_settings::{ImageEntry, SeriesSettings},
        project_settings::ProjectSettings,
    },
};
use evanalyzer_cfg::{
    core_types::SegmentationClass, object_class_set_from_u32,
    settings::object_settings::ObjectMetricSettings,
};
use evanalyzer_core::{ImageMeta, ImageReader, ReadMode, SUPPORTED_IMAGE_FORMATS};
use human_sort::compare;
use log::{info, trace, warn};
use rayon::prelude::*;
use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

#[derive(Debug, PartialEq, Eq)]
pub enum ProjectAction {
    /// Everything went fine, image added.
    Success,
    /// The image is outside the root. The UI needs to ask the user what to do.
    OutSideRootConflict {
        image_path: PathBuf,
        current_root: PathBuf,
    },
    /// A legitimate failure (e.g., file doesn't exist).
    Failure(String),
}

#[derive(Debug, PartialEq, Eq)]
pub enum SelectNewProjectRootAction {
    /// Everything went fine, image added.
    Success,

    /// Image in the new project root not found
    ImageNotFound,
}

#[derive(Debug, PartialEq)]
pub enum SaveProjectActions {
    /// Everything went fine, image added.
    Success,

    /// No project file yet selected
    PleaseSelectFile,

    Error,
}

pub trait ProjectExt {
    fn add_class_to_object(&mut self, id: ObjectId, object_class: ObjectClass);
    fn remove_class_from_object(&mut self, id: ObjectId, object_class: &ObjectClass);
    fn add_object(&mut self, object: &ObjectMetricSettings);
    fn get_objects(&self) -> Option<&[ObjectMetricSettings]>;
    fn delete_object(&mut self, id: ObjectId);
    fn get_reference_object(&self) -> Option<Vec<ObjectMetricSettings>>;
    fn get_class_from_id(&self, id: &ObjectClass) -> Option<&Class>;
    fn get_current_relative_path(&self) -> Option<PathBuf>;
    fn with_current_image_mut<F, R>(&mut self, f: F) -> Option<R>
    where
        F: FnOnce(&mut ImageEntry) -> R;
    fn with_current_series_mut<F, R>(&mut self, f: F) -> Option<R>
    where
        F: FnOnce(&mut SeriesSettings) -> R;
    fn get_current_image_path_cloned(&self) -> Option<PathBuf>;
    fn get_current_rel_image_path_cloned(&self) -> Option<PathBuf>;
    fn set_current_image_path(&mut self, path: &PathBuf);
    fn rest_current_image_path(&mut self);
    fn find_image(&self, absolute_path: &PathBuf) -> Option<&ImageEntry>;

    fn find_image_mut(&mut self, path: &PathBuf) -> Option<&mut ImageEntry>;

    fn get_current_image_settings(&self) -> Option<&ImageEntry>;
    fn get_current_image_settings_mut(&mut self) -> Option<&mut ImageEntry>;
    fn get_selected_image_series_idx(&self) -> Option<i32>;
    fn get_selected_image_series(&self) -> Option<&SeriesSettings>;
    fn get_selected_image_series_mut(&mut self) -> Option<&mut SeriesSettings>;
    fn get_current_image_channel_settings(&self) -> Option<(i32, &SeriesSettings)>;
    fn get_image_channel_visibilities(&self) -> BTreeMap<i32, bool>;
    fn get_image_channel_visibilities_vec(&self) -> Vec<i32>;
    fn get_image_channel_histograms(&self) -> BTreeMap<i32, Option<HistogramSettings>>;
    fn get_histograms_from_selected_channel(&self) -> Option<&HistogramSettings>;
    fn get_selected_image_channel_idx(&self) -> i32;
    fn get_selected_series_idx(&self) -> i32;
    fn set_active_series(&mut self, selected_series: &i32);
    fn set_image_preferences(&mut self, channel_visibility: &BTreeMap<i32, bool>);
    fn set_global_preferences(&mut self, channel_visibility: &BTreeMap<i32, bool>);
    fn set_image_z_stack(&mut self, z_stack: &ZStackSettings);
    fn set_global_z_stack(&mut self, z_stack: &ZStackSettings);
    fn get_z_stack(&self) -> Option<&ZStackSettings>;
    fn get_t_stack(&self) -> Option<&TStackSettings>;
    fn get_pixel_sizes(&self) -> PixelSizeSettings;
    fn get_selected_image_channel(&self) -> Option<&ChannelSettings>;
    fn set_image_t_stack(&mut self, t_stack: &TStackSettings);
    fn set_global_t_stack(&mut self, t_stack: &TStackSettings);
    fn set_image_pixel_size_settings(&mut self, px: f32, py: f32, pz: f32);
    fn set_global_pixel_size_settings(&mut self, px: f32, py: f32, pz: f32);
    fn reset_global_pixel_size_settings(&mut self);
    fn set_image_selected_channel(&mut self, selected_channel: &i32);
    fn set_global_selected_channel(&mut self, selected_channel: &i32);
    fn set_image_histogram_settings_for_active_channel(
        &mut self,
        min: f32,
        max: f32,
        min_limit: f32,
        max_limit: f32,
    );
    fn set_image_histogram_settings_for_channel(
        &mut self,
        channel_id: i32,
        min: f32,
        max: f32,
        min_limit: f32,
        max_limit: f32,
    );
    fn auto_add_classes_based_on_image_meta(&mut self);
    fn set_selected_object_class(&mut self, object_class: ObjectClass);
    fn get_selected_object_class(&self) -> ObjectClass;
    fn delete_all_classes(&mut self);
    fn get_image_absolute_path_from_relative(&self, path: &Path) -> Option<PathBuf>;

    fn change_images_root(&mut self, new_root: &PathBuf);

    fn select_new_images_root_with_check(
        &mut self,
        new_root: &PathBuf,
    ) -> SelectNewProjectRootAction;

    fn select_new_images_root(&mut self, new_root: &PathBuf);

    fn does_project_images_exist(&self) -> bool;
    fn does_project_image_exists_at_path(&self, new_root: &PathBuf) -> bool;

    fn is_image_part_of_the_root(&self, absolute_path: &Path) -> bool;
    fn add_image_and_read_meta(&mut self, absolute_path: &Path) -> ProjectAction;
    fn add_image(&mut self, absolute_path: &Path, image_meta: &ImageMeta) -> ProjectAction;
    fn add_image_to_list(&mut self, rel_path: &Path, abs_path: &Path, image_meta: &ImageMeta);
    fn scan_image_folder_and_add(&mut self);
    fn apply_scanned_images(&mut self, found_images: Vec<(PathBuf, ImageMeta)>);
    fn collect_images_parallel(&self, dir: &Path) -> Vec<(PathBuf, ImageMeta)>;
    fn is_supported_image(&self, path: &Path) -> bool;

    fn save_project(&mut self) -> SaveProjectActions;
    fn save_project_as(&mut self, path: &PathBuf) -> Result<(), InternalErrors>;
    fn save_project_as_template(
        &mut self,
        meta: MetaData,
        path: &PathBuf,
    ) -> Result<(), InternalErrors>;
    fn save_pipeline_as_template(
        &mut self,
        meta: MetaData,
        pipeline_id: PipelineId,
        path: &PathBuf,
    ) -> Result<(), InternalErrors>;

    fn new(&self) -> Arc<ProjectWithRuntime>;
    fn new_project(&mut self, path: &PathBuf) -> Result<ProjectWithRuntime, InternalErrors>;

    fn move_pipeline_up(&mut self, pipeline_id: PipelineId);
    fn move_pipeline_down(&mut self, pipeline_id: PipelineId);

    fn enable_pipeline(&mut self, enabled: bool, pipeline_id: PipelineId);
    fn enable_pipeline_step(&mut self, enabled: bool, pipeline_id: PipelineId, step_id: usize);

    fn add_pipeline(&mut self, pipeline_settings: PipelineSettings);
    fn add_pipeline_from_template_file(&mut self, template_file: &PathBuf);

    /// Replaces the classification, plate and pipeline settings of this project
    /// with the ones from `template`. The image list and project path are kept.
    fn apply_project_template(&mut self, template: &ProjectTemplate);

    fn toggle_class_visibility(&mut self, class_id: ObjectClass);
    fn is_class_visible(&self, class_id: &ObjectClass) -> bool;
    fn count_objects_for_class(&self, class_id: &ObjectClass) -> usize;

    fn toggle_hide_unclassified_objects(&mut self);
    fn hide_unclassified_objects(&self) -> bool;
}

impl ProjectExt for ProjectWithRuntime {
    /// Adds an object class to a specific object by its ID.
    ///
    /// This method finds the object with the given ID in the currently selected image series
    /// and adds the specified object class to it.
    ///
    /// # Arguments
    /// * `id` - The ObjectId of the object to modify
    /// * `object_class` - The ObjectClass to add to the object
    fn add_class_to_object(&mut self, id: ObjectId, object_class: ObjectClass) {
        if let Some(series) = self.get_selected_image_series_mut() {
            if let Some(object) = series.objects.iter_mut().find(|object| object.id == id) {
                object.add_object_class(object_class);
            }
        }
    }

    /// Removes an object class from a specific object by its ID.
    ///
    /// This method finds the object with the given ID in the currently selected image series
    /// and removes the specified object class from it.
    ///
    /// # Arguments
    /// * `id` - The ObjectId of the object to modify
    /// * `object_class` - The ObjectClass to remove from the object
    fn remove_class_from_object(&mut self, id: ObjectId, object_class: &ObjectClass) {
        if let Some(series) = self.get_selected_image_series_mut() {
            if let Some(object) = series.objects.iter_mut().find(|object| object.id == id) {
                object.remove_object_class(object_class);
            }
        }
    }

    fn add_object(&mut self, object: &ObjectMetricSettings) {
        if let Some(series) = self.get_selected_image_series_mut() {
            series.objects.push(object.clone());
        }
    }

    fn get_objects(&self) -> Option<&[ObjectMetricSettings]> {
        self.get_selected_image_series()
            .map(|series| series.objects.as_slice())
    }

    fn delete_object(&mut self, id: ObjectId) {
        if let Some(series) = self.get_selected_image_series_mut() {
            series.objects.retain(|object| object.id != id);
        }
    }

    fn get_reference_object(&self) -> Option<Vec<ObjectMetricSettings>> {
        let mut objects = Vec::new();

        // Helper to generate a circular object
        let create_circle_object = |id: u128, x_start: usize, y_start: usize| {
            let width = 5;
            let height = 5;
            let radius = 2.0f32;
            let center = 2i32;

            // BitVec stores bits efficiently. Lsb0 maps index 0 to the first bit.
            let mut mask_data = BitVec::<u64, Lsb0>::repeat(false, width * height);

            for y in 0..height {
                for x in 0..width {
                    if (x as i32 - center).pow(2) + (y as i32 - center).pow(2)
                        <= radius.powi(2) as i32
                    {
                        // Direct indexing! No manual bit arithmetic needed.
                        mask_data.set(y * width + x, true);
                    }
                }
            }
            ObjectMetricSettings {
                id: ObjectId(id),
                bbox: [
                    x_start as u32,
                    y_start as u32,
                    (x_start + width) as u32,
                    (y_start + height) as u32,
                ],
                mask_data: mask_data,
                segmentation_class: SegmentationClass(1),
                object_class: object_class_set_from_u32!(1),
                area: 31415,
                ..Default::default()
            }
        };

        let num_elements = 30000;
        let elements_per_row = 150;
        let spacing = 2048 / elements_per_row;
        for i in 0..num_elements {
            // Calculate which row and which column the current index belongs to
            let row = i / elements_per_row;
            let col = i % elements_per_row;

            // Distribute evenly by multiplying index by the calculated spacing
            let x = ((col * spacing) + (spacing / 2)) as usize;
            let y = ((row * spacing) + (spacing / 2)) as usize;

            objects.push(create_circle_object(i + 1, x, y));
        }
        Some(objects)
    }

    fn get_class_from_id(&self, id: &ObjectClass) -> Option<&Class> {
        self.classification.classes().iter().find(|c| c.id == *id)
    }

    /// Internal helper to get the relative key of the currently active image.
    fn get_current_relative_path(&self) -> Option<PathBuf> {
        let path = self.get_current_image_path_cloned()?;
        get_relative_key(&path, self.images.root.as_ref())
    }

    /// Internal helper to execute a closure with a mutable reference to the current image entry.
    fn with_current_image_mut<F, R>(&mut self, f: F) -> Option<R>
    where
        F: FnOnce(&mut ImageEntry) -> R,
    {
        let relative = self.get_current_relative_path()?;
        self.images.list.get_mut(&relative).map(|entry| f(entry))
    }

    /// Internal helper to execute a closure with a mutable reference to the current series settings.
    fn with_current_series_mut<F, R>(&mut self, f: F) -> Option<R>
    where
        F: FnOnce(&mut SeriesSettings) -> R,
    {
        self.with_current_image_mut(|image| {
            if let Some(series) = image.series.get_mut(&image.selected_series) {
                return Some(f(series));
            }
            None
        })?
    }

    // --- Public API ---

    /// Returns a clone of the current image path if set.
    fn get_current_image_path_cloned(&self) -> Option<PathBuf> {
        self.tmp_settings.current_image.clone()
    }

    // Returns the relative image path of the currently selected image
    fn get_current_rel_image_path_cloned(&self) -> Option<PathBuf> {
        let Some(path) = self.get_current_image_path_cloned() else {
            return None;
        };
        get_relative_key(&path, self.images.root.as_ref()).clone()
    }

    /// Sets the path of the currently active image.
    fn set_current_image_path(&mut self, path: &PathBuf) {
        self.tmp_settings.current_image = Some(path.into());
    }

    fn rest_current_image_path(&mut self) {
        self.tmp_settings.current_image = None;
    }

    /// Retrieves a shared pointer to an `ImageEntry` using an absolute file path.
    fn find_image(&self, absolute_path: &PathBuf) -> Option<&ImageEntry> {
        let relative = get_relative_key(absolute_path, self.images.root.as_ref())?;
        self.images.list.get(&relative)
    }

    fn find_image_mut(&mut self, absolute_path: &PathBuf) -> Option<&mut ImageEntry> {
        let relative = get_relative_key(absolute_path, self.images.root.as_ref())?;
        self.images.list.get_mut(&relative)
    }

    /// Returns the settings for the currently active image.
    fn get_current_image_settings(&self) -> Option<&ImageEntry> {
        let path = self.get_current_image_path_cloned()?;
        self.find_image(&path)
    }

    /// Returns the settings for the currently active image.
    fn get_current_image_settings_mut(&mut self) -> Option<&mut ImageEntry> {
        let path = self.get_current_image_path_cloned()?;
        self.find_image_mut(&path)
    }

    /// Returns the active series index and its settings for the current image.
    fn get_current_image_channel_settings(&self) -> Option<(i32, &SeriesSettings)> {
        let image = self.get_current_image_settings()?;
        image
            .series
            .get(&image.selected_series)
            .map(|s| (image.selected_series, s))
    }

    /// Returns a map of channel IDs to their visibility, falling back to global settings if not set locally.
    fn get_image_channel_visibilities(&self) -> BTreeMap<i32, bool> {
        let mut visibilities = BTreeMap::new();
        if let Some((_, series)) = self.get_current_image_channel_settings() {
            for (idx, channel) in &series.channels {
                let visible = channel.visible.unwrap_or_else(|| {
                    self.images
                        .settings
                        .channels
                        .get(idx)
                        .and_then(|c| c.visible)
                        .unwrap_or(true)
                });
                visibilities.insert(*idx, visible);
            }
        }
        visibilities
    }

    /// Returns a list of channel IDs that are currently marked as visible.
    fn get_image_channel_visibilities_vec(&self) -> Vec<i32> {
        self.get_image_channel_visibilities()
            .into_iter()
            .filter_map(|(id, visible)| if visible { Some(id) } else { None })
            .collect()
    }

    /// Returns histogram settings for all channels in the current series, with global fallbacks.
    fn get_image_channel_histograms(&self) -> BTreeMap<i32, Option<HistogramSettings>> {
        let mut histograms = BTreeMap::new();
        if let Some((_, series)) = self.get_current_image_channel_settings() {
            for (idx, channel) in &series.channels {
                let hist = channel.histogram.clone().or_else(|| {
                    self.images
                        .settings
                        .channels
                        .get(idx)
                        .and_then(|c| c.histogram.clone())
                });
                histograms.insert(*idx, hist);
            }
        }
        histograms
    }

    /// Returns histogram settings for the selected image channel
    fn get_histograms_from_selected_channel(&self) -> Option<&HistogramSettings> {
        if let Some((_, series)) = self.get_current_image_channel_settings() {
            let channel = series
                .channels
                .get(&self.get_selected_image_channel_idx())?;
            return channel.histogram.as_ref();
        }
        None
    }

    /// Returns the index of the currently selected series.
    fn get_selected_series_idx(&self) -> i32 {
        self.get_current_image_settings()
            .map(|img| img.selected_series)
            .unwrap_or(0)
    }

    /// Updates the active series index for the current image.
    fn set_active_series(&mut self, selected_series: &i32) {
        self.with_current_image_mut(|image| {
            image.selected_series = *selected_series;
        });
    }

    /// Updates channel visibility preferences for the current image series.
    fn set_image_preferences(&mut self, channel_visibility: &BTreeMap<i32, bool>) {
        self.with_current_series_mut(|series| {
            for (id, visible) in channel_visibility {
                if let Some(channel_arc) = series.channels.get_mut(id) {
                    channel_arc.visible = Some(*visible);
                }
            }
        });
    }

    /// Updates global channel visibility preferences.
    fn set_global_preferences(&mut self, channel_visibility: &BTreeMap<i32, bool>) {
        for (id, visible) in channel_visibility {
            let channel_arc =
                self.images
                    .settings
                    .channels
                    .entry(*id)
                    .or_insert_with(|| ChannelSettings {
                        name: "".into(),
                        emission_wave_length: 0.0,
                        visible: None,
                        histogram: None,
                    });

            // This allows mutation by cloning if shared, or just borrowing if unique
            channel_arc.visible = Some(*visible);
        }
    }

    /// Sets the Z-stack settings for the current image series.
    fn set_image_z_stack(&mut self, z_stack: &ZStackSettings) {
        self.with_current_series_mut(|series| {
            series.z_stack = Some(z_stack.clone());
        });
    }

    /// Sets the global Z-stack settings.
    fn set_global_z_stack(&mut self, z_stack: &ZStackSettings) {
        self.images.settings.z_stack = Some(z_stack.clone());
    }

    /// Retrieves the active Z-stack settings (Local -> Global -> Default).
    fn get_z_stack(&self) -> Option<&ZStackSettings> {
        let image_z_stack = self
            .get_current_image_channel_settings()
            .and_then(|(_, s)| s.z_stack.as_ref());

        match image_z_stack {
            Some(z) => Some(z),
            None => self.images.settings.z_stack.as_ref(),
        }
    }

    /// Retrieves the active T-stack settings (Local -> Global -> Default).
    fn get_t_stack(&self) -> Option<&TStackSettings> {
        let image_t_stack = self
            .get_current_image_channel_settings()
            .and_then(|(_, s)| s.t_stack.as_ref());

        match image_t_stack {
            Some(z) => Some(z),
            None => self.images.settings.t_stack.as_ref(),
        }
    }

    /// Retrieves the active pixel size settings (Global -> Local -> Default).
    fn get_pixel_sizes(&self) -> PixelSizeSettings {
        // 1. Check Global first
        self.images
            .settings
            .pixel_sizes
            .clone()
            // 2. If Global was None, return the Local value
            .unwrap_or_else(|| {
                self.get_current_image_channel_settings()
                    .map(|(_, s)| s.pixel_sizes.clone())
                    .unwrap_or_else(|| PixelSizeSettings {
                        x: 1.0,
                        y: 1.0,
                        z: 1.0,
                    })
            })
    }

    /// Returns the index of the selected channel for the current image.
    fn get_selected_image_channel_idx(&self) -> i32 {
        self.get_current_image_channel_settings()
            .and_then(|(_, s)| s.selected_channel)
            .unwrap_or_else(|| self.images.settings.selected_channel.unwrap_or(0))
    }

    /// Returns the settings for the currently selected channel.
    fn get_selected_image_channel(&self) -> Option<&ChannelSettings> {
        let idx = self.get_selected_image_channel_idx();
        self.get_current_image_channel_settings()
            .and_then(|(_, s)| s.channels.get(&idx).clone())
    }

    /// Sets the T-stack settings for the current image series.
    fn set_image_t_stack(&mut self, t_stack: &TStackSettings) {
        self.with_current_series_mut(|series| {
            series.t_stack = Some(t_stack.clone());
        });
    }

    /// Sets the global T-stack settings.
    fn set_global_t_stack(&mut self, t_stack: &TStackSettings) {
        self.images.settings.t_stack = Some(t_stack.clone());
    }

    /// Sets the pixel size settings for the current image series.
    fn set_image_pixel_size_settings(&mut self, px: f32, py: f32, pz: f32) {
        self.with_current_series_mut(|series| {
            series.pixel_sizes = PixelSizeSettings {
                x: px,
                y: py,
                z: pz,
            };
        });
    }

    /// Sets the global pixel size settings.
    fn set_global_pixel_size_settings(&mut self, px: f32, py: f32, pz: f32) {
        self.images.settings.pixel_sizes = Some(PixelSizeSettings {
            x: px,
            y: py,
            z: pz,
        });
    }

    /// Resets the global pixel size settings.
    fn reset_global_pixel_size_settings(&mut self) {
        self.images.settings.pixel_sizes = None;
    }

    /// Sets the selected channel index for the current image series.
    fn set_image_selected_channel(&mut self, selected_channel: &i32) {
        self.with_current_series_mut(|series| {
            series.selected_channel = Some(*selected_channel);
        });
    }

    /// Sets the global selected channel index.
    fn set_global_selected_channel(&mut self, selected_channel: &i32) {
        self.images.settings.selected_channel = Some(*selected_channel);
    }

    /// Updates histogram settings for the active channel of the current image.
    fn set_image_histogram_settings_for_active_channel(
        &mut self,
        min: f32,
        max: f32,
        min_limit: f32,
        max_limit: f32,
    ) {
        let channel_id = self.get_selected_image_channel_idx();
        self.set_image_histogram_settings_for_channel(channel_id, min, max, min_limit, max_limit);
    }

    /// Updates histogram settings for a specific channel of the current image.
    fn set_image_histogram_settings_for_channel(
        &mut self,
        channel_id: i32,
        min: f32,
        max: f32,
        min_limit: f32,
        max_limit: f32,
    ) {
        // 1. Get the index outside the closure to avoid double-borrowing self
        let series_idx = self.get_selected_series_idx();

        self.with_current_series_mut(|series| {
            if let Some(channel) = series.channels.get_mut(&channel_id) {
                channel.histogram = Some(HistogramSettings {
                    min,
                    max,
                    min_limit,
                    max_limit,
                });
            } else {
                warn!(
                    "Attempted to set histogram settings for non-existent channel ID {} in series {}", channel_id, series_idx);
            }
        });
    }

    /// Returns the active series index for the current image.
    fn get_selected_image_series_idx(&self) -> Option<i32> {
        self.get_current_image_settings()
            .map(|img| img.selected_series)
    }

    fn get_selected_image_series(&self) -> Option<&SeriesSettings> {
        let Some(set) = self.get_current_image_settings() else {
            return None;
        };
        set.series.get(&set.selected_series).clone()
    }

    fn get_selected_image_series_mut(&mut self) -> Option<&mut SeriesSettings> {
        let Some(set) = self.get_current_image_settings_mut() else {
            return None;
        };
        set.series.get_mut(&set.selected_series)
    }

    fn auto_add_classes_based_on_image_meta(&mut self) {
        // Collect channel data first to avoid holding Deref borrows while mutating.
        let classes: Vec<Class> = {
            let idx = self.get_selected_series_idx();
            self.settings
                .images
                .list
                .values()
                .next()
                .and_then(|image| image.series.get(&idx))
                .map(|series_data| {
                    series_data
                        .channels
                        .values()
                        .map(|ch| Class {
                            id: ObjectClass::Unset,
                            color: wavelength_to_rgb_u32(ch.emission_wave_length),
                            name: ch.name.clone(),
                            notes: "".into(),
                        })
                        .collect()
                })
                .unwrap_or_default()
        };
        for class in classes {
            self.classification.add_class(class);
        }
    }

    fn set_selected_object_class(&mut self, object_class: ObjectClass) {
        self.tmp_settings.selected_object_class = object_class;
    }

    fn get_selected_object_class(&self) -> ObjectClass {
        self.tmp_settings.selected_object_class.clone()
    }

    fn delete_all_classes(&mut self) {
        self.classification = ClassificationSettings::new();
    }

    fn get_image_absolute_path_from_relative(&self, path: &Path) -> Option<PathBuf> {
        // Check if the path exists in our list
        if self.images.list.contains_key(path) {
            Some(match self.images.root.as_ref() {
                Some(root) => root.join(path),
                None => path.to_path_buf(),
            })
        } else {
            None
        }
    }

    /// Updates the project's image root and performs a full synchronization.
    ///
    /// This method is destructive: it clears the current [`images.list`] and resets
    /// the [`current_image`] selection to ensure no "ghost" paths from the previous
    /// root remain in memory.
    ///
    /// # Arguments
    /// * `new_root` - The absolute path to the new directory containing the project images.
    ///
    /// # Note
    /// This should typically be followed by a background directory scan to
    /// repopulate the image list based on the new root.
    fn change_images_root(&mut self, new_root: &PathBuf) {
        // Clear the image list first (Prevent UI from accessing dead paths)

        self.images.list.clear();

        // Update the root

        self.images.root = Some(new_root.clone());

        // Reset actual selected image

        self.tmp_settings.current_image = None;

        info!(
            "Project root changed to {:?}. Image list cleared.",
            new_root
        );

        // Trigger a re-scan of the new directory
        //self.spawn_directory_scan(new_root);
    }

    /// Re-links an existing project to a new location on the filesystem.
    ///
    /// Use this when a project file has been moved or shared, and the image
    /// resources are now located at a different absolute path. Unlike
    /// `change_images_root`, this preserves the existing image metadata and list,
    /// only updating the base reference point.
    ///
    /// # Arguments
    /// * `new_root` - The new absolute path where the project's images are located.
    fn select_new_images_root_with_check(
        &mut self,
        new_root: &PathBuf,
    ) -> SelectNewProjectRootAction {
        if !self.does_project_image_exists_at_path(new_root) {
            return SelectNewProjectRootAction::ImageNotFound;
        }

        self.select_new_images_root(new_root);
        SelectNewProjectRootAction::Success
    }

    fn select_new_images_root(&mut self, new_root: &PathBuf) {
        {
            self.images.root = Some(new_root.clone());
        }
        info!("Project re-linked to new root: {:?}", new_root);

        // 3. Optional: Validation
        // You might want to trigger a "Check Integrity" scan here to verify
        // that the files actually exist at the new location.
    }

    fn does_project_images_exist(&self) -> bool {
        if let Some(root) = &self.images.root {
            return self.does_project_image_exists_at_path(&root);
        };

        // No root set
        return true;
    }

    fn does_project_image_exists_at_path(&self, new_root: &PathBuf) -> bool {
        let sample_rel_path = { self.images.list.keys().next().cloned() };

        if let Some(rel) = sample_rel_path {
            let test_path = new_root.join(rel);
            if !test_path.exists() {
                // Warn the user that the first image wasn't found here
                return false;
            }
        }
        return true;
    }

    fn is_image_part_of_the_root(&self, absolute_path: &Path) -> bool {
        if let Some(existing_root) = self.images.root.as_deref() {
            if is_in_root(absolute_path, existing_root) {
                return true;
            } else {
                return false;
            }
        } else {
            // No root yet set
            return false;
        }
    }

    /// Registers a new image and its associated metadata into the project.
    ///
    /// This method generates a relative path key based on the project's base directory
    /// and performs a batch-compatible insertion into the image map.
    ///
    /// ### Workflow
    /// 1. **Relativize:** Converts the absolute path to a project-relative `PathBuf`.
    /// 2. **Initialize:** Wraps the provided `ImageMeta` into a new `ImageEntry`.
    /// 3. **Lock & Insert:** Acquires a **write lock** to update the `IndexMap`.
    ///
    /// # Thread Safety
    /// This method requires a **unique write lock** on the `images` map.
    /// If multiple threads attempt to add images simultaneously, they will be serialized.
    ///
    /// # Panics
    /// Panics if the internal `RwLock` is poisoned by a previous thread failure.
    fn add_image_and_read_meta(&mut self, absolute_path: &Path) -> ProjectAction {
        if self.is_supported_image(&absolute_path) {
            match ImageReader::new(&absolute_path.to_path_buf(), ReadMode::SplitChannels) {
                Ok(reader) => {
                    let image_meta = reader.get_image_meta();
                    self.add_image(absolute_path, &image_meta);
                    return ProjectAction::Success;
                }
                Err(ex) => {
                    return ProjectAction::Failure(ex.to_string());
                }
            }
        } else {
            return ProjectAction::Failure("Unsupported device".into());
        }
    }

    fn add_image(&mut self, absolute_path: &Path, image_meta: &ImageMeta) -> ProjectAction {
        // Determine the effective root
        let effective_root = {
            if let Some(existing_root) = self.images.root.as_deref() {
                if is_in_root(absolute_path, existing_root) {
                    Some(existing_root.to_path_buf())
                } else {
                    // Case 2: Image is outside existing root.
                    // Decision: Do you ignore it, or force a root change?
                    warn!(
                        "Image {:?} is outside current root {:?}. Ignoring.",
                        absolute_path, existing_root
                    );
                    return ProjectAction::OutSideRootConflict {
                        image_path: absolute_path.into(),
                        current_root: existing_root.into(),
                    };
                }
            } else {
                None // Case 3: No root set yet
            }
        };

        // Handle the "No Root" case by setting it
        let final_root = match effective_root {
            Some(root) => root,
            None => {
                // Set the new root in the project settings
                let parent = absolute_path
                    .parent()
                    .map(|p| p.to_path_buf())
                    .unwrap_or_else(|| PathBuf::from("."));
                info!("No root detected. Setting root to: {:?}", parent);

                self.images.root = Some(parent.clone());
                parent
            }
        };

        // Unified Success Path
        if let Some(relative_path) = get_relative_key(absolute_path, Some(&final_root)) {
            self.add_image_to_list(&relative_path, absolute_path, image_meta);
        } else {
            warn!("Failed to calculate relative path for {:?}", absolute_path);
        }
        ProjectAction::Success
    }

    /// Registers a new image into the project manifest.
    ///
    /// This method initializes default [`SeriesSettings`] and [`ChannelSettings`]
    /// based on the provided metadata. If the image (by relative path) already
    /// exists in the list, the operation is skipped to prevent duplicates.
    ///
    /// # Thread Safety
    /// Acquires a write lock on `self.images.list`. Will panic if the lock is poisoned.
    fn add_image_to_list(&mut self, rel_path: &Path, _abs_path: &Path, image_meta: &ImageMeta) {
        if self.images.list.contains_key(rel_path) {
            trace!("Image already exists at {:?}, skipping.", rel_path);
            return;
        }

        let mut file_size_approx_bytes: u64 = 0;
        //  Map Series and Channels
        let series_settings: BTreeMap<i32, SeriesSettings> = image_meta
            .series
            .iter()
            .map(|(&series_idx, data)| {
                let channels: BTreeMap<_, _> = data
                    .channels
                    .iter()
                    .map(|(ch_idx, data)| {
                        (
                            *ch_idx,
                            ChannelSettings {
                                name: data.name.clone(),
                                emission_wave_length: data.emission_wave_length.clone(),
                                visible: None,
                                histogram: None,
                            },
                        )
                    })
                    .collect();

                let (image_width, image_height, nr_bits) = match data.resolutions.get(&0) {
                    Some(res) => (res.width, res.height, res.nr_bits),
                    None => (0, 0, 0), // Fallback if resolution 0 is missing
                };

                let pixel_sizes = data.pixel_sizes.clone();
                let settings = SeriesSettings {
                    selected_channel: None,
                    image_width: image_width,
                    image_height: image_height,
                    channels,
                    pixel_sizes: PixelSizeSettings {
                        x: pixel_sizes.px_size_x,
                        y: pixel_sizes.px_size_y,
                        z: pixel_sizes.px_size_z,
                    },
                    z_stack: None,
                    t_stack: None,
                    objects: Vec::new(),
                };

                file_size_approx_bytes += image_width * image_height * nr_bits as u64;

                (series_idx, settings)
            })
            .collect();

        let img = ImageEntry {
            rel_path: rel_path.to_path_buf(),
            file_size: file_size_approx_bytes,
            selected_series: 0,
            series: series_settings,
        };

        self.images.list.insert(rel_path.to_path_buf(), img);

        info!(
            "Successfully added image to project: {}",
            rel_path.display()
        );
    }

    /// Describe this function.
    ///
    /// # Arguments
    ///
    /// - `path` (`PathBuf`) - Describe this parameter.
    ///
    /// # Returns
    ///
    /// - `Result<Vec<ImageEntry>, InternalErrors>` - Describe the return value.
    ///
    /// # Errors
    ///
    /// Describe possible errors.
    ///
    /// # Examples
    ///
    /// ```
    /// use crate::...;
    ///
    /// let _ = scan_folder();
    /// ```
    ///
    fn scan_image_folder_and_add(&mut self) {
        if let Some(root_folder) = self.settings.images.root.clone() {
            let found_images = collect_images_at_root(&root_folder);
            self.apply_scanned_images(found_images);
        }
    }

    /// Replaces the project's image list with `found_images` (as returned by
    /// the free function [`collect_images_at_root`]). Unlike
    /// `scan_image_folder_and_add`, this does no filesystem I/O itself, so
    /// callers that already scanned off-lock only need to hold the project
    /// write guard for this call, not for the scan that produced its input.
    fn apply_scanned_images(&mut self, mut found_images: Vec<(PathBuf, ImageMeta)>) {
        self.images.list.clear();

        // Natural sort
        found_images.sort_by(|a, b| {
            let path_a = a.0.to_string_lossy();
            let path_b = b.0.to_string_lossy();
            compare(&path_a, &path_b)
        });

        let start = Instant::now();
        for (path, meta) in found_images {
            self.add_image(&mut &path, &meta);
        }
        let duration = start.elapsed();
        info!("Added images {:?}", duration);
    }

    fn collect_images_parallel(&self, dir: &Path) -> Vec<(PathBuf, ImageMeta)> {
        collect_images_at_root(dir)
    }

    fn is_supported_image(&self, path: &Path) -> bool {
        is_supported_image_path(path)
    }

    fn save_project(&mut self) -> SaveProjectActions {
        // 1. Clone the path to release the borrow on 'self' immediately
        let Some(path) = self.tmp_settings.current_project.clone() else {
            return SaveProjectActions::PleaseSelectFile;
        };

        // 2. Now 'self' is free to be borrowed mutably by 'save_project_as'
        match self.save_project_as(&path) {
            Ok(_) => SaveProjectActions::Success,
            Err(_) => SaveProjectActions::Error,
        }
    }

    fn save_project_as(&mut self, path: &PathBuf) -> Result<(), InternalErrors> {
        let mut final_path = path.clone();

        // Check if the extension matches; if not, set it to evaproj
        if final_path.extension().and_then(|s| s.to_str()) != Some(PROJECT_FILE_EXTENSIONS) {
            final_path.set_extension(PROJECT_FILE_EXTENSIONS);
        }

        // Every file written by this build is stamped with the current
        // format version, regardless of what version it was loaded at (or
        // whether it was ever loaded from disk at all - e.g. a brand-new
        // project still has `schema_version: 0` from `Default`).
        self.settings.schema_version = evanalyzer_cfg::CURRENT_PROJECT_SCHEMA_VERSION;

        // Every command's file-path field (AI model paths today - see
        // `relativize_file_paths`'s doc comment for why this isn't limited to
        // those specifically) is stored relative to the project's own
        // directory on disk so the project keeps working after the folder is
        // moved or copied elsewhere - but only in the serialized copy; the
        // in-memory settings keep absolute paths so the rest of the app
        // (pipeline preview, the model-info dialog, ...) doesn't need to
        // know about the project directory to resolve them.
        let mut on_disk_settings = self.settings.clone();
        if let Some(project_dir) = final_path.parent() {
            crate::extensions::utils::relativize_file_paths(
                &mut on_disk_settings.pipelines,
                project_dir,
            );
        }

        let json = serde_json::to_string_pretty(&on_disk_settings)
            .map_err(|e| InternalErrors::ParseError(e.to_string()))?;
        fs::write(final_path.clone(), json)?;

        let _ = self.tmp_settings.current_project.insert(final_path);
        Ok(())
    }

    /// Stores the actual project as template project
    fn save_project_as_template(
        &mut self,
        meta: MetaData,
        path: &PathBuf,
    ) -> Result<(), InternalErrors> {
        let template = ProjectTemplate {
            meta,
            classification: self.classification.clone(),
            plate: self.plate.clone(),
            pipelines: self
                .pipelines
                .iter()
                .map(|pipeline| PipelineTemplate {
                    meta: MetaData {
                        name: pipeline.name.clone().unwrap_or_default(),
                        ..Default::default()
                    },
                    pipeline_steps: pipeline.steps.clone(),
                })
                .collect(),
        };

        let mut final_path = path.clone();
        if final_path.extension().and_then(|s| s.to_str()) != Some(PROJECT_FILE_TEMPLATE_EXTENSIONS)
        {
            final_path.set_extension(PROJECT_FILE_TEMPLATE_EXTENSIONS);
        }

        let json = serde_json::to_string_pretty(&template)
            .map_err(|e| InternalErrors::ParseError(e.to_string()))?;
        fs::write(final_path, json)?;
        Ok(())
    }

    /// Stores the selected pipeline as template
    fn save_pipeline_as_template(
        &mut self,
        meta: MetaData,
        pipeline_id: PipelineId,
        path: &PathBuf,
    ) -> Result<(), InternalErrors> {
        let pipeline = self
            .pipelines
            .iter()
            .find(|p| p.id == pipeline_id)
            .ok_or_else(|| InternalErrors::Internal("Pipeline not found".into()))?;

        let template = PipelineTemplate {
            meta,
            pipeline_steps: pipeline.steps.clone(),
        };

        let mut final_path = path.clone();
        if final_path.extension().and_then(|s| s.to_str()) != Some(PIPELINE_EXTENSIONS) {
            final_path.set_extension(PIPELINE_EXTENSIONS);
        }

        let json = serde_json::to_string_pretty(&template)
            .map_err(|e| InternalErrors::ParseError(e.to_string()))?;
        fs::write(final_path, json)?;
        Ok(())
    }

    /// Creates a new, empty project instance wrapped in an [`Arc`].
    ///
    /// This initializes a "blank slate" project with a `None` root and an empty
    /// image list. The resulting project is thread-safe and ready to be shared
    /// between the UI and background processing threads.
    ///
    /// # Returns
    /// An `Arc<Project>` containing default-initialized synchronization primitives
    /// (Mutexes/RwLocks) for immediate use in a multi-threaded context.
    fn new(&self) -> Arc<ProjectWithRuntime> {
        Arc::new(ProjectWithRuntime::default())
    }

    fn new_project(&mut self, path: &PathBuf) -> Result<ProjectWithRuntime, InternalErrors> {
        let mut project = ProjectWithRuntime::default();
        project.save_project_as(path)?;
        Ok(project)
    }

    fn move_pipeline_up(&mut self, pipeline_id: PipelineId) {
        if let Some(i) = self.pipelines.iter().position(|c| c.id == pipeline_id) {
            if i > 0 {
                self.pipelines.swap(i, i - 1);
            }
        }
    }
    fn move_pipeline_down(&mut self, pipeline_id: PipelineId) {
        if let Some(i) = self.pipelines.iter().position(|c| c.id == pipeline_id) {
            if i < self.pipelines.len() - 1 {
                self.pipelines.swap(i, i + 1);
            }
        }
    }

    fn enable_pipeline(&mut self, enabled: bool, pipeline_id: PipelineId) {
        if let Some(i) = self.pipelines.iter().position(|c| c.id == pipeline_id) {
            self.pipelines[i].enabled = enabled;
        }
    }

    fn enable_pipeline_step(&mut self, enabled: bool, pipeline_id: PipelineId, step_id: usize) {
        if let Some(i) = self.pipelines.iter().position(|c| c.id == pipeline_id) {
            if let Some(step) = self.pipelines[i].steps.get_mut(step_id) {
                step.enabled = enabled;
            }
        }
    }

    fn add_pipeline(&mut self, pipeline_settings: PipelineSettings) {
        self.pipelines.push(pipeline_settings);
    }
    fn add_pipeline_from_template_file(&mut self, template_file: &PathBuf) {
        let Ok(data) = fs::read_to_string(template_file) else {
            warn!("Could not read pipeline template {:?}", template_file);
            return;
        };
        let template: PipelineTemplate = match serde_json::from_str(&data) {
            Ok(template) => template,
            Err(err) => {
                warn!(
                    "Could not parse pipeline template {:?}: {}",
                    template_file, err
                );
                return;
            }
        };

        let next_id = self.pipelines.iter().map(|p| p.id.0).max().unwrap_or(0) + 1;
        let name = if template.meta.name.is_empty() {
            None
        } else {
            Some(template.meta.name)
        };

        self.add_pipeline(PipelineSettings {
            id: PipelineId(next_id),
            name,
            image_source: ImageAddress::default(),
            enabled: true,
            steps: template.pipeline_steps,
        });
    }

    fn apply_project_template(&mut self, template: &ProjectTemplate) {
        self.classification = template.classification.clone();
        self.plate = template.plate.clone();
        self.pipelines = template
            .pipelines
            .iter()
            .enumerate()
            .map(|(idx, pipeline)| {
                let name = if pipeline.meta.name.is_empty() {
                    None
                } else {
                    Some(pipeline.meta.name.clone())
                };
                PipelineSettings {
                    id: PipelineId((idx + 1) as u32),
                    name,
                    image_source: ImageAddress::default(),
                    enabled: true,
                    steps: pipeline.pipeline_steps.clone(),
                }
            })
            .collect();
    }

    fn toggle_class_visibility(&mut self, class_id: ObjectClass) {
        if self.tmp_settings.hidden_classes.contains(&class_id) {
            self.tmp_settings.hidden_classes.remove(&class_id);
        } else {
            self.tmp_settings.hidden_classes.insert(class_id);
        }
    }

    fn is_class_visible(&self, class_id: &ObjectClass) -> bool {
        !self.tmp_settings.hidden_classes.contains(class_id)
    }

    fn toggle_hide_unclassified_objects(&mut self) {
        self.tmp_settings.hide_unclassified_objects = !self.tmp_settings.hide_unclassified_objects;
    }

    fn hide_unclassified_objects(&self) -> bool {
        self.tmp_settings.hide_unclassified_objects
    }

    fn count_objects_for_class(&self, class_id: &ObjectClass) -> usize {
        let manual = self
            .get_objects()
            .unwrap_or_default()
            .iter()
            .filter(|r| r.object_class.contains(class_id))
            .count();
        let preview = self
            .tmp_settings
            .preview_objects
            .iter()
            .filter(|r| r.object_class.contains(class_id))
            .count();
        manual + preview
    }
}

/// Recursively scans `dir` in parallel for supported image files and reads
/// each one's metadata. Pure filesystem I/O - takes no project reference and
/// holds no lock, so callers on a project shared behind an `RwLock` (the GUI)
/// should run this *before* taking the write guard, then commit the result
/// with `ProjectExt::apply_scanned_images` - keeping the lock held only for
/// the fast in-memory part instead of for the whole scan (which opens an
/// `ImageReader` per file and can take seconds on a plate-sized folder).
pub fn collect_images_at_root(dir: &Path) -> Vec<(PathBuf, ImageMeta)> {
    // 1. Check if the current directory itself is named "results"
    if let Some(name) = dir.file_name().and_then(|n| n.to_str()) {
        if name.eq_ignore_ascii_case("results") {
            return vec![]; // Skip this folder and everything inside it
        }
    }

    let Ok(entries) = std::fs::read_dir(dir) else {
        return vec![];
    };

    entries
        .flatten()
        .collect::<Vec<_>>()
        .into_par_iter()
        .flat_map(|entry| {
            let path = entry.path();

            if path.is_dir() {
                // The check happens again here for subdirectories
                collect_images_at_root(&path)
            } else if is_supported_image_path(&path) {
                if let Ok(reader) = ImageReader::new(&path, ReadMode::SplitChannels) {
                    vec![(path, (*reader.get_image_meta()).clone())]
                } else {
                    vec![]
                }
            } else {
                vec![]
            }
        })
        .collect()
}

fn is_supported_image_path(path: &Path) -> bool {
    path.extension()
        .and_then(|ext| ext.to_str()) // Convert OsStr to &str
        .map(|ext_str| {
            let ext_lower = ext_str.to_lowercase();
            SUPPORTED_IMAGE_FORMATS.iter().any(|&fmt| fmt == ext_lower)
        })
        .unwrap_or(false) // Return false if no extension exists
}

pub fn load_project(path: &PathBuf) -> Result<ProjectWithRuntime, InternalErrors> {
    let data = fs::read_to_string(path.clone())?;
    let mut inner: ProjectSettings =
        serde_json::from_str(&data).map_err(|e| InternalErrors::ParseError(e.to_string()))?;

    // A file from a *newer* build may rely on fields/variants this build
    // doesn't know about yet - serde would have silently dropped them rather
    // than erroring, so proceeding could quietly discard part of the user's
    // project on the next save. Refuse instead of guessing.
    if inner.schema_version > evanalyzer_cfg::CURRENT_PROJECT_SCHEMA_VERSION {
        return Err(InternalErrors::ParseError(format!(
            "this project was saved by a newer version of EVAnalyzer (format version {}, this build supports up to version {}) - update EVAnalyzer to open it",
            inner.schema_version,
            evanalyzer_cfg::CURRENT_PROJECT_SCHEMA_VERSION
        )));
    }
    inner.migrate();

    // Reverse of the relativization `save_project_as` does on write - resolve
    // AI command `model_path`s back to absolute now, using this project
    // file's own directory, so the rest of the app can keep treating them as
    // plain absolute paths.
    if let Some(project_dir) = path.parent() {
        crate::extensions::utils::resolve_file_paths(&mut inner.pipelines, project_dir);
    }

    let mut project = ProjectWithRuntime {
        settings: inner,
        tmp_settings: ProjectTmpSettings::default(),
    };
    project.tmp_settings.current_project = Some(path.clone());
    Ok(project)
}

/// Imports an old (`.icproj`) project, converting it to the current format.
///
/// Unlike [`load_project`], the result has no `current_project` path set - the
/// imported project doesn't exist as a `.evaproj` file yet, so saving it goes
/// through the normal "please select a file" flow rather than silently
/// overwriting the legacy file.
///
/// Returns the converted project, a list of conversion warnings (commands or
/// fields the old project used that have no equivalent and were skipped or
/// approximated), and the old project's configured image folder, if any - the
/// caller is responsible for resolving that folder (e.g. relative to `path`'s
/// parent) and scanning it for images, mirroring whatever flow already adds
/// images to a new project; this function only converts settings.
pub fn import_legacy_project(
    path: &PathBuf,
) -> Result<(ProjectWithRuntime, Vec<String>, Option<String>), InternalErrors> {
    let data = fs::read_to_string(path)?;
    let outcome = evanalyzer_cfg::import_legacy_project(&data)
        .map_err(|e| InternalErrors::ParseError(e.to_string()))?;
    let project = ProjectWithRuntime {
        settings: outcome.project,
        tmp_settings: ProjectTmpSettings::default(),
    };
    Ok((project, outcome.warnings, outcome.legacy_image_folder))
}

#[cfg(test)]
mod tests {
    use super::*;
    use evanalyzer_cfg::core_types::ObjectClass;
    use evanalyzer_cfg::settings::images_settings::{TStackHandling, ZStackHandling};
    use evanalyzer_cfg::settings::pipeline_command::PipelineCommand;
    use evanalyzer_cfg::settings::pipeline_settings::PipelineStepSettings;

    /// A project with one image ("img.tif") of one series (idx 0, 2x2px)
    /// carrying two channels ("Ch0"/"Ch1"), selected as the current image -
    /// i.e. everything `get_selected_image_series`/`get_current_image_*`
    /// need to return `Some`.
    fn project_with_one_image() -> ProjectWithRuntime {
        let mut project = ProjectWithRuntime::default();
        let rel_path = PathBuf::from("img.tif");

        let channels = BTreeMap::from([
            (
                0,
                ChannelSettings {
                    name: "Ch0".into(),
                    emission_wave_length: 488.0,
                    visible: None,
                    histogram: None,
                },
            ),
            (
                1,
                ChannelSettings {
                    name: "Ch1".into(),
                    emission_wave_length: 561.0,
                    visible: None,
                    histogram: None,
                },
            ),
        ]);
        let series = SeriesSettings {
            selected_channel: None,
            image_width: 2,
            image_height: 2,
            channels,
            pixel_sizes: PixelSizeSettings {
                x: 0.5,
                y: 0.5,
                z: 1.0,
            },
            z_stack: None,
            t_stack: None,
            objects: Vec::new(),
        };
        project.images.list.insert(
            rel_path.clone(),
            ImageEntry {
                rel_path: rel_path.clone(),
                file_size: 16,
                selected_series: 0,
                series: BTreeMap::from([(0, series)]),
            },
        );
        project.set_current_image_path(&rel_path);
        project
    }

    fn class(id: u32, name: &str) -> Class {
        Class {
            id: ObjectClass::Valid(id),
            name: name.into(),
            ..Default::default()
        }
    }

    // -- objects -----------------------------------------------------------

    #[test]
    fn add_get_delete_object_round_trip() {
        let mut project = project_with_one_image();
        assert_eq!(project.get_objects().unwrap().len(), 0);

        project.add_object(&ObjectMetricSettings {
            id: ObjectId(1),
            ..Default::default()
        });
        project.add_object(&ObjectMetricSettings {
            id: ObjectId(2),
            ..Default::default()
        });
        assert_eq!(project.get_objects().unwrap().len(), 2);

        project.delete_object(ObjectId(1));
        let remaining = project.get_objects().unwrap();
        assert_eq!(remaining.len(), 1);
        assert_eq!(remaining[0].id, ObjectId(2));
    }

    #[test]
    fn object_mutations_are_a_no_op_without_a_selected_image() {
        let mut project = ProjectWithRuntime::default();
        assert!(project.get_objects().is_none());
        // Must not panic even though there's nothing to add to.
        project.add_object(&ObjectMetricSettings::default());
        project.delete_object(ObjectId(1));
        assert!(project.get_objects().is_none());
    }

    #[test]
    fn add_and_remove_class_from_object() {
        let mut project = project_with_one_image();
        project.add_object(&ObjectMetricSettings {
            id: ObjectId(1),
            ..Default::default()
        });

        project.add_class_to_object(ObjectId(1), ObjectClass::Valid(3));
        assert!(
            project.get_objects().unwrap()[0]
                .object_class
                .contains(&ObjectClass::Valid(3))
        );

        project.remove_class_from_object(ObjectId(1), &ObjectClass::Valid(3));
        assert!(
            !project.get_objects().unwrap()[0]
                .object_class
                .contains(&ObjectClass::Valid(3))
        );
    }

    // -- Classes ----------------------------------------------------------

    #[test]
    fn get_class_from_id_finds_by_id_not_position() {
        let mut project = ProjectWithRuntime::default();
        project.classification.classes_mut().push(class(1, "A"));
        project.classification.classes_mut().push(class(2, "B"));

        let found = project.get_class_from_id(&ObjectClass::Valid(2)).unwrap();
        assert_eq!(found.name, "B");
        assert!(project.get_class_from_id(&ObjectClass::Valid(99)).is_none());
    }

    #[test]
    fn delete_all_classes_clears_the_list() {
        let mut project = ProjectWithRuntime::default();
        project.classification.classes_mut().push(class(1, "A"));
        project.delete_all_classes();
        assert!(project.classification.classes().len() == 1);
        assert!(project.classification.classes()[0].id == ObjectClass::BACKGROUND);
        assert!(project.classification.classes()[0].color == 0x97978F);
        assert!(project.classification.classes()[0].name == "Background");
    }

    #[test]
    fn selected_object_class_round_trips() {
        let mut project = ProjectWithRuntime::default();
        assert_eq!(project.get_selected_object_class(), ObjectClass::default());
        project.set_selected_object_class(ObjectClass::Valid(4));
        assert_eq!(project.get_selected_object_class(), ObjectClass::Valid(4));
    }

    #[test]
    fn toggle_class_visibility_is_a_flip_flop() {
        let mut project = ProjectWithRuntime::default();
        let c = ObjectClass::Valid(1);
        assert!(
            project.is_class_visible(&c),
            "classes are visible by default"
        );

        project.toggle_class_visibility(c.clone());
        assert!(!project.is_class_visible(&c));

        project.toggle_class_visibility(c.clone());
        assert!(project.is_class_visible(&c));
    }

    #[test]
    fn toggle_hide_unclassified_objects_is_a_flip_flop() {
        let mut project = ProjectWithRuntime::default();
        let initial = project.hide_unclassified_objects();
        project.toggle_hide_unclassified_objects();
        assert_eq!(project.hide_unclassified_objects(), !initial);
        project.toggle_hide_unclassified_objects();
        assert_eq!(project.hide_unclassified_objects(), initial);
    }

    #[test]
    fn count_objects_for_class_counts_both_series_and_preview_objects() {
        let mut project = project_with_one_image();
        let target = ObjectClass::Valid(5);

        project.add_object(&ObjectMetricSettings {
            id: ObjectId(1),
            object_class: [target.clone()].into(),
            ..Default::default()
        });
        project.add_object(&ObjectMetricSettings {
            id: ObjectId(2),
            ..Default::default()
        });
        project
            .tmp_settings
            .preview_objects
            .push(ObjectMetricSettings {
                id: ObjectId(3),
                object_class: [target.clone()].into(),
                ..Default::default()
            });

        assert_eq!(project.count_objects_for_class(&target), 2);
        assert_eq!(project.count_objects_for_class(&ObjectClass::Valid(999)), 0);
    }

    // -- Pixel sizes: series-level, then global override, then reset ------

    #[test]
    fn pixel_sizes_default_to_one_when_nothing_is_set() {
        let project = ProjectWithRuntime::default();
        let px = project.get_pixel_sizes();
        assert_eq!((px.x, px.y, px.z), (1.0, 1.0, 1.0));
    }

    #[test]
    fn pixel_sizes_prefer_global_over_series_and_reset_falls_back() {
        let mut project = project_with_one_image();

        project.set_image_pixel_size_settings(0.1, 0.2, 0.3);
        let px = project.get_pixel_sizes();
        assert_eq!(
            (px.x, px.y, px.z),
            (0.1, 0.2, 0.3),
            "falls back to the series value"
        );

        project.set_global_pixel_size_settings(9.0, 9.0, 9.0);
        let px = project.get_pixel_sizes();
        assert_eq!(
            (px.x, px.y, px.z),
            (9.0, 9.0, 9.0),
            "global overrides series"
        );

        project.reset_global_pixel_size_settings();
        let px = project.get_pixel_sizes();
        assert_eq!(
            (px.x, px.y, px.z),
            (0.1, 0.2, 0.3),
            "clearing global falls back to series again"
        );
    }

    // -- Channel visibility -------------------------------------------------

    #[test]
    fn channels_are_visible_by_default() {
        let project = project_with_one_image();
        let vis = project.get_image_channel_visibilities();
        assert_eq!(vis.get(&0), Some(&true));
        assert_eq!(vis.get(&1), Some(&true));
    }

    #[test]
    fn set_image_preferences_overrides_per_channel_visibility() {
        let mut project = project_with_one_image();
        project.set_image_preferences(&BTreeMap::from([(0, false)]));

        let vis = project.get_image_channel_visibilities();
        assert_eq!(vis.get(&0), Some(&false));
        assert_eq!(
            vis.get(&1),
            Some(&true),
            "untouched channel keeps its default"
        );

        let visible_only = project.get_image_channel_visibilities_vec();
        assert!(!visible_only.contains(&0));
        assert!(visible_only.contains(&1));
    }

    #[test]
    fn set_global_preferences_is_the_fallback_when_a_channel_has_no_local_override() {
        let mut project = project_with_one_image();
        project.set_global_preferences(&BTreeMap::from([(0, false)]));
        let vis = project.get_image_channel_visibilities();
        assert_eq!(vis.get(&0), Some(&false));

        // A local override still wins over the global one.
        project.set_image_preferences(&BTreeMap::from([(0, true)]));
        let vis = project.get_image_channel_visibilities();
        assert_eq!(vis.get(&0), Some(&true));
    }

    // -- Image root / path resolution ---------------------------------------

    #[test]
    fn get_image_absolute_path_from_relative_only_resolves_known_images() {
        let mut project = project_with_one_image();
        project.images.root = Some(PathBuf::from("/data/images"));

        assert_eq!(
            project.get_image_absolute_path_from_relative(Path::new("img.tif")),
            Some(PathBuf::from("/data/images/img.tif"))
        );
        assert_eq!(
            project.get_image_absolute_path_from_relative(Path::new("unknown.tif")),
            None
        );
    }

    #[test]
    fn is_image_part_of_the_root_requires_a_root_to_be_set() {
        let mut project = ProjectWithRuntime::default();
        assert!(!project.is_image_part_of_the_root(Path::new("/data/images/img.tif")));

        project.images.root = Some(PathBuf::from("/data/images"));
        assert!(project.is_image_part_of_the_root(Path::new("/data/images/img.tif")));
        assert!(!project.is_image_part_of_the_root(Path::new("/other/img.tif")));
    }

    #[test]
    fn change_images_root_clears_the_image_list_and_current_image() {
        let mut project = project_with_one_image();
        assert!(!project.images.list.is_empty());

        let new_root = PathBuf::from("/new/root");
        project.change_images_root(&new_root);

        assert!(project.images.list.is_empty());
        assert_eq!(project.images.root, Some(new_root));
        assert_eq!(project.get_current_image_path_cloned(), None);
    }

    #[test]
    fn select_new_images_root_with_check_fails_when_the_sample_image_is_missing() {
        let mut project = project_with_one_image();
        let missing_root = PathBuf::from("/definitely/does/not/exist/anywhere");
        let action = project.select_new_images_root_with_check(&missing_root);
        assert_eq!(action, SelectNewProjectRootAction::ImageNotFound);
        // A failed check must not have changed the root.
        assert_ne!(project.images.root, Some(missing_root));
    }

    #[test]
    fn select_new_images_root_with_check_succeeds_when_the_sample_image_exists() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("img.tif"), b"fake image bytes").unwrap();

        let mut project = project_with_one_image();
        let action = project.select_new_images_root_with_check(&dir.path().to_path_buf());
        assert_eq!(action, SelectNewProjectRootAction::Success);
        assert_eq!(project.images.root, Some(dir.path().to_path_buf()));
    }

    // -- Save / load round trip ----------------------------------------------

    #[test]
    fn save_project_without_a_path_asks_the_caller_to_pick_one() {
        let mut project = ProjectWithRuntime::default();
        assert_eq!(project.save_project(), SaveProjectActions::PleaseSelectFile);
    }

    #[test]
    fn save_project_as_appends_the_extension_and_records_the_path() {
        let dir = tempfile::tempdir().unwrap();
        let mut project = ProjectWithRuntime::default();
        project.metadata.name = "Round Trip Project".into();

        // No extension given - save_project_as should append the project one.
        let requested = dir.path().join("myproject");
        project
            .save_project_as(&requested)
            .expect("save should succeed");

        let expected = dir
            .path()
            .join(format!("myproject.{}", PROJECT_FILE_EXTENSIONS));
        assert!(
            expected.exists(),
            "file should be written with the project extension appended"
        );
        assert_eq!(project.tmp_settings.current_project, Some(expected.clone()));

        // save_project() (no path arg) should now reuse that recorded path.
        project.metadata.name = "Changed After Save As".into();
        assert_eq!(project.save_project(), SaveProjectActions::Success);

        let written = std::fs::read_to_string(&expected).unwrap();
        assert!(written.contains("Changed After Save As"));
    }

    #[test]
    fn loaded_project_round_trips_objects_and_classes() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("full.evaproj");

        let mut project = project_with_one_image();
        project
            .classification
            .classes_mut()
            .push(class(1, "Nucleus"));
        project.add_object(&ObjectMetricSettings {
            id: ObjectId(1),
            area: 1234,
            object_class: [ObjectClass::Valid(1)].into(),
            ..Default::default()
        });
        project.save_project_as(&path).unwrap();

        let loaded = load_project(&path).expect("load should succeed");
        // classes()[0] is the auto-prepended Background class.
        assert_eq!(loaded.classification.classes().len(), 2);
        assert_eq!(loaded.classification.classes()[1].name, "Nucleus");
        // `load_project` doesn't select an image on its own - it only loads
        // `settings`, leaving `tmp_settings` at its defaults.
        assert_eq!(loaded.get_current_image_path_cloned(), None);
        let objects = &loaded.images.list.get(Path::new("img.tif")).unwrap().series[&0].objects;
        assert_eq!(objects.len(), 1);
        assert_eq!(objects[0].area, 1234);
    }

    #[test]
    fn save_project_as_stores_ai_model_paths_relative_to_the_project_dir() {
        use evanalyzer_cfg::settings::pipeline_command::PipelineCommand;
        use evanalyzer_cfg::settings::pipeline_command_settings::PixelClassifierSettings;
        use evanalyzer_cfg::settings::pipeline_settings::PipelineStepSettings;

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("full.evaproj");
        let absolute_model_path = dir.path().join("models").join("nuclei.evamodel");

        let mut project = ProjectWithRuntime::default();
        let mut step_pipeline = pipeline(1);
        step_pipeline.steps.push(PipelineStepSettings {
            enabled: true,
            command: PipelineCommand::PixelClassifier(PixelClassifierSettings {
                model_path: absolute_model_path.clone(),
                ..Default::default()
            }),
        });
        project.add_pipeline(step_pipeline);
        project.save_project_as(&path).unwrap();

        // On disk, the path is relative to the project directory - not the
        // absolute temp-dir path it was set from - so the project stays
        // valid if the folder is moved or copied elsewhere.
        let written = std::fs::read_to_string(&path).unwrap();
        assert!(
            written.contains("models/nuclei.evamodel")
                && !written.contains(&*absolute_model_path.to_string_lossy()),
            "expected a relative model path on disk, got: {written}"
        );

        // Reloading resolves it straight back to the original absolute path
        // - the rest of the app never has to know about relative storage.
        let loaded = load_project(&path).expect("load should succeed");
        let PipelineCommand::PixelClassifier(settings) = &loaded.pipelines[0].steps[0].command
        else {
            panic!("expected a PixelClassifier command");
        };
        assert_eq!(settings.model_path, absolute_model_path);
    }

    // -- Pipeline management ---------------------------------------------

    fn pipeline(id: u32) -> PipelineSettings {
        PipelineSettings {
            id: PipelineId(id),
            name: None,
            image_source: ImageAddress::default(),
            enabled: true,
            steps: vec![],
        }
    }

    #[test]
    fn add_pipeline_appends_to_the_list() {
        let mut project = ProjectWithRuntime::default();
        project.add_pipeline(pipeline(1));
        project.add_pipeline(pipeline(2));
        assert_eq!(
            project.pipelines.iter().map(|p| p.id).collect::<Vec<_>>(),
            vec![PipelineId(1), PipelineId(2)]
        );
    }

    #[test]
    fn move_pipeline_up_and_down_swap_neighbors_and_are_edge_noops() {
        let mut project = ProjectWithRuntime::default();
        project.add_pipeline(pipeline(1));
        project.add_pipeline(pipeline(2));
        project.add_pipeline(pipeline(3));

        project.move_pipeline_up(PipelineId(2));
        assert_eq!(
            project.pipelines.iter().map(|p| p.id.0).collect::<Vec<_>>(),
            vec![2, 1, 3]
        );

        // Already first: no-op.
        project.move_pipeline_up(PipelineId(2));
        assert_eq!(
            project.pipelines.iter().map(|p| p.id.0).collect::<Vec<_>>(),
            vec![2, 1, 3]
        );

        project.move_pipeline_down(PipelineId(2));
        assert_eq!(
            project.pipelines.iter().map(|p| p.id.0).collect::<Vec<_>>(),
            vec![1, 2, 3]
        );

        // Already last: no-op.
        project.move_pipeline_down(PipelineId(3));
        assert_eq!(
            project.pipelines.iter().map(|p| p.id.0).collect::<Vec<_>>(),
            vec![1, 2, 3]
        );

        // Unknown id: no-op, no panic.
        project.move_pipeline_up(PipelineId(99));
        project.move_pipeline_down(PipelineId(99));
        assert_eq!(
            project.pipelines.iter().map(|p| p.id.0).collect::<Vec<_>>(),
            vec![1, 2, 3]
        );
    }

    #[test]
    fn enable_pipeline_toggles_only_the_matching_pipeline() {
        let mut project = ProjectWithRuntime::default();
        project.add_pipeline(pipeline(1));
        project.add_pipeline(pipeline(2));

        project.enable_pipeline(false, PipelineId(1));
        assert!(!project.pipelines[0].enabled);
        assert!(project.pipelines[1].enabled);

        // Unknown id: no panic, no effect.
        project.enable_pipeline(false, PipelineId(99));
    }

    #[test]
    fn enable_pipeline_step_toggles_only_the_matching_step() {
        let mut project = ProjectWithRuntime::default();
        let mut p = pipeline(1);
        p.steps = vec![
            PipelineStepSettings {
                enabled: true,
                command: PipelineCommand::Blur(Default::default()),
            },
            PipelineStepSettings {
                enabled: true,
                command: PipelineCommand::Blur(Default::default()),
            },
        ];
        project.add_pipeline(p);

        project.enable_pipeline_step(false, PipelineId(1), 0);
        assert!(!project.pipelines[0].steps[0].enabled);
        assert!(project.pipelines[0].steps[1].enabled);

        // Unknown pipeline / out-of-range step: no panic.
        project.enable_pipeline_step(false, PipelineId(99), 0);
        project.enable_pipeline_step(false, PipelineId(1), 99);
    }

    #[test]
    fn add_pipeline_from_template_file_assigns_the_next_id_and_carries_the_name() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("t.evapipe");
        let template = evanalyzer_cfg::settings::templates::PipelineTemplate {
            meta: evanalyzer_cfg::settings::meta_data::MetaData {
                name: "Imported".into(),
                ..Default::default()
            },
            pipeline_steps: vec![],
        };
        std::fs::write(&path, serde_json::to_string(&template).unwrap()).unwrap();

        let mut project = ProjectWithRuntime::default();
        project.add_pipeline(pipeline(5));
        project.add_pipeline_from_template_file(&path);

        assert_eq!(project.pipelines.len(), 2);
        assert_eq!(
            project.pipelines[1].id,
            PipelineId(6),
            "next id must be max(existing) + 1"
        );
        assert_eq!(project.pipelines[1].name.as_deref(), Some("Imported"));
    }

    #[test]
    fn add_pipeline_from_template_file_is_a_noop_for_a_missing_file() {
        let mut project = ProjectWithRuntime::default();
        project.add_pipeline_from_template_file(&PathBuf::from("/does/not/exist.evapipe"));
        assert!(project.pipelines.is_empty());
    }

    #[test]
    fn add_pipeline_from_template_file_is_a_noop_for_malformed_json() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("bad.evapipe");
        std::fs::write(&path, "{ not json").unwrap();

        let mut project = ProjectWithRuntime::default();
        project.add_pipeline_from_template_file(&path);
        assert!(project.pipelines.is_empty());
    }

    #[test]
    fn apply_project_template_replaces_classification_plate_and_pipelines() {
        let mut project = ProjectWithRuntime::default();
        project.classification.classes_mut().push(class(1, "Stale"));
        project.add_pipeline(pipeline(1));

        let template = evanalyzer_cfg::settings::templates::ProjectTemplate {
            meta: Default::default(),
            classification:
                evanalyzer_cfg::settings::classification_settings::ClassificationSettings::new_from_existing(
                   vec![class(9, "Fresh")]),
            plate: Default::default(),
            pipelines: vec![evanalyzer_cfg::settings::templates::PipelineTemplate {
                meta: evanalyzer_cfg::settings::meta_data::MetaData {
                    name: "P1".into(),
                    ..Default::default()
                },
                pipeline_steps: vec![],
            }],
        };
        project.apply_project_template(&template);

        // classes()[0] is the auto-prepended Background class.
        assert_eq!(project.classification.classes().len(), 2);
        assert_eq!(project.classification.classes()[1].name, "Fresh");
        assert_eq!(project.pipelines.len(), 1);
        assert_eq!(
            project.pipelines[0].id,
            PipelineId(1),
            "ids are renumbered from 1, not carried over from the old pipelines"
        );
        assert_eq!(project.pipelines[0].name.as_deref(), Some("P1"));
    }

    // -- Image existence / root checks ------------------------------------

    #[test]
    fn does_project_images_exist_is_true_when_no_root_is_set() {
        let project = ProjectWithRuntime::default();
        assert!(project.does_project_images_exist());
    }

    #[test]
    fn does_project_images_exist_checks_the_first_listed_image_under_the_root() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("img.tif"), b"x").unwrap();

        let mut project = project_with_one_image();
        project.images.root = Some(dir.path().to_path_buf());
        assert!(project.does_project_images_exist());

        // Point the root somewhere that doesn't have img.tif.
        let other_dir = tempfile::tempdir().unwrap();
        project.images.root = Some(other_dir.path().to_path_buf());
        assert!(!project.does_project_images_exist());
    }

    #[test]
    fn does_project_image_exists_at_path_is_true_for_an_empty_image_list() {
        let project = ProjectWithRuntime::default();
        assert!(project.does_project_image_exists_at_path(&PathBuf::from("/anywhere")));
    }

    // -- add_image / add_image_to_list ------------------------------------

    fn sample_image_meta(channel_name: &str, wavelength: f32) -> ImageMeta {
        let mut resolutions = BTreeMap::new();
        resolutions.insert(
            0,
            evanalyzer_core::PyramidInfo {
                width: 4,
                height: 4,
                nr_bits: 8,
                ..Default::default()
            },
        );
        let mut channels = BTreeMap::new();
        channels.insert(
            0,
            evanalyzer_core::ChannelInfo {
                name: channel_name.into(),
                emission_wave_length: wavelength,
                ..Default::default()
            },
        );
        let mut series = BTreeMap::new();
        series.insert(
            0,
            evanalyzer_core::ImageInfo {
                nr_c_stacks: 1,
                nr_z_stacks: 1,
                nr_t_stacks: 1,
                resolutions,
                channels,
                ..Default::default()
            },
        );
        ImageMeta {
            name: "meta".into(),
            series,
            ..Default::default()
        }
    }

    #[test]
    fn add_image_sets_the_root_from_the_first_images_parent_when_unset() {
        let mut project = ProjectWithRuntime::default();
        let meta = sample_image_meta("Ch0", 488.0);
        let action = project.add_image(Path::new("/data/plate1/img.tif"), &meta);

        assert_eq!(action, ProjectAction::Success);
        assert_eq!(project.images.root, Some(PathBuf::from("/data/plate1")));
        assert!(project.images.list.contains_key(Path::new("img.tif")));
    }

    #[test]
    fn add_image_accepts_a_path_inside_an_existing_root() {
        let mut project = ProjectWithRuntime::default();
        project.images.root = Some(PathBuf::from("/data/plate1"));
        let meta = sample_image_meta("Ch0", 488.0);

        let action = project.add_image(Path::new("/data/plate1/well_A1/img.tif"), &meta);
        assert_eq!(action, ProjectAction::Success);
        assert!(
            project
                .images
                .list
                .contains_key(Path::new("well_A1/img.tif"))
        );
    }

    #[test]
    fn add_image_outside_an_existing_root_is_a_conflict_and_does_not_add_it() {
        let mut project = ProjectWithRuntime::default();
        project.images.root = Some(PathBuf::from("/data/plate1"));
        let meta = sample_image_meta("Ch0", 488.0);

        let action = project.add_image(Path::new("/other/img.tif"), &meta);
        assert!(matches!(action, ProjectAction::OutSideRootConflict { .. }));
        assert!(project.images.list.is_empty());
    }

    #[test]
    fn add_image_to_list_skips_a_path_that_is_already_present() {
        let mut project = ProjectWithRuntime::default();
        let first = sample_image_meta("Ch0", 488.0);
        let second = sample_image_meta("ShouldNotOverwrite", 600.0);

        project.add_image_to_list(Path::new("img.tif"), Path::new("/abs/img.tif"), &first);
        project.add_image_to_list(Path::new("img.tif"), Path::new("/abs/img.tif"), &second);

        let entry = project.images.list.get(Path::new("img.tif")).unwrap();
        assert_eq!(
            entry.series[&0].channels[&0].name, "Ch0",
            "the second insert must be a no-op"
        );
    }

    #[test]
    fn add_image_to_list_derives_series_and_channel_settings_from_the_metadata() {
        let mut project = ProjectWithRuntime::default();
        let meta = sample_image_meta("DAPI", 461.0);
        project.add_image_to_list(Path::new("img.tif"), Path::new("/abs/img.tif"), &meta);

        let entry = project.images.list.get(Path::new("img.tif")).unwrap();
        let series = &entry.series[&0];
        assert_eq!(series.image_width, 4);
        assert_eq!(series.image_height, 4);
        assert_eq!(series.channels[&0].name, "DAPI");
        assert_eq!(series.channels[&0].emission_wave_length, 461.0);
        assert_eq!(
            entry.file_size,
            4 * 4 * 8,
            "approximate size is width * height * bits"
        );
    }

    // -- Series / channel selection ---------------------------------------

    #[test]
    fn set_active_series_updates_the_current_images_selected_series() {
        let mut project = project_with_one_image();
        project.with_current_image_mut(|img| {
            img.series.insert(1, img.series[&0].clone());
        });
        project.set_active_series(&1);
        assert_eq!(project.get_selected_image_series_idx(), Some(1));
        assert!(project.get_selected_image_series().is_some());
    }

    #[test]
    fn get_selected_image_series_mut_allows_editing_the_active_series() {
        let mut project = project_with_one_image();
        project.get_selected_image_series_mut().unwrap().image_width = 999;
        assert_eq!(
            project.get_selected_image_series().unwrap().image_width,
            999
        );
    }

    #[test]
    fn selected_channel_prefers_local_over_global_over_zero() {
        let mut project = project_with_one_image();
        assert_eq!(
            project.get_selected_image_channel_idx(),
            0,
            "defaults to 0 with nothing set"
        );

        project.set_global_selected_channel(&1);
        assert_eq!(
            project.get_selected_image_channel_idx(),
            1,
            "falls back to the global setting"
        );

        project.set_image_selected_channel(&0);
        assert_eq!(
            project.get_selected_image_channel_idx(),
            0,
            "a local override wins over global"
        );
        assert_eq!(project.get_selected_image_channel().unwrap().name, "Ch0");
    }

    // -- Z / T stack --------------------------------------------------------

    #[test]
    fn z_stack_prefers_local_over_global() {
        let mut project = project_with_one_image();
        assert!(project.get_z_stack().is_none());

        let global = ZStackSettings {
            z_projection: ZStackHandling::MaxIntensity,
            z_range: None,
        };
        project.set_global_z_stack(&global);
        assert_eq!(
            project.get_z_stack().unwrap().z_projection,
            ZStackHandling::MaxIntensity
        );

        let local = ZStackSettings {
            z_projection: ZStackHandling::SumIntensity,
            z_range: None,
        };
        project.set_image_z_stack(&local);
        assert_eq!(
            project.get_z_stack().unwrap().z_projection,
            ZStackHandling::SumIntensity
        );
    }

    #[test]
    fn t_stack_prefers_local_over_global() {
        let mut project = project_with_one_image();
        assert!(project.get_t_stack().is_none());

        let global = TStackSettings {
            stack_handling: TStackHandling::AllStacks,
            playback_speed: 1.0,
            t_stack: 0,
        };
        project.set_global_t_stack(&global);
        assert_eq!(
            project.get_t_stack().unwrap().stack_handling,
            TStackHandling::AllStacks
        );

        let local = TStackSettings {
            stack_handling: TStackHandling::SingleStack,
            playback_speed: 1.0,
            t_stack: 3,
        };
        project.set_image_t_stack(&local);
        assert_eq!(
            project.get_t_stack().unwrap().stack_handling,
            TStackHandling::SingleStack
        );
    }

    // -- Histogram settings -------------------------------------------------

    #[test]
    fn set_image_histogram_settings_for_channel_updates_only_that_channel() {
        let mut project = project_with_one_image();
        project.set_image_histogram_settings_for_channel(0, 0.1, 0.9, 0.0, 1.0);

        let hist = project.get_image_channel_histograms();
        assert_eq!(hist[&0].as_ref().unwrap().min, 0.1);
        assert_eq!(hist[&0].as_ref().unwrap().max, 0.9);
        assert!(hist[&1].is_none());
    }

    #[test]
    fn set_image_histogram_settings_for_active_channel_targets_the_selected_channel() {
        let mut project = project_with_one_image();
        project.set_image_selected_channel(&1);
        project.set_image_histogram_settings_for_active_channel(0.2, 0.8, 0.0, 1.0);

        assert_eq!(
            project.get_histograms_from_selected_channel().unwrap().min,
            0.2
        );
        assert!(project.get_image_channel_histograms()[&0].is_none());
    }

    #[test]
    fn set_image_histogram_settings_for_an_unknown_channel_does_not_panic() {
        let mut project = project_with_one_image();
        project.set_image_histogram_settings_for_channel(99, 0.0, 1.0, 0.0, 1.0);
        // No channel 99 exists - nothing to assert beyond "did not panic".
    }

    // -- auto_add_classes_based_on_image_meta -------------------------------

    #[test]
    fn auto_add_classes_creates_one_class_per_channel_of_the_first_image() {
        let mut project = project_with_one_image();
        // A fresh project always has exactly the auto-prepended Background
        // class and nothing else.
        assert_eq!(project.classification.classes().len(), 1);

        project.auto_add_classes_based_on_image_meta();

        let names: Vec<&str> = project
            .classification
            .classes()
            .iter()
            .map(|c| c.name.as_str())
            .collect();
        // Background, plus one class per channel.
        assert_eq!(names.len(), 3);
        assert!(names.contains(&"Background"));
        assert!(names.contains(&"Ch0"));
        assert!(names.contains(&"Ch1"));
        // `add_class` (via `ClassificationExt`) assigns each new class a
        // fresh, unique id - the `ObjectClass::Unset` built in the collected
        // `Class` is only a placeholder that gets overwritten there, not the
        // final id. Background's own id is always `Valid(0)`, so all three
        // classes have a real `Valid` id here.
        let ids: Vec<u32> = project
            .classification
            .classes()
            .iter()
            .filter_map(|c| c.id.to_u32())
            .collect();
        assert_eq!(
            ids.len(),
            3,
            "Background and both new classes must have a real Valid id"
        );
    }

    // -- misc getters ---------------------------------------------------

    #[test]
    fn get_current_relative_path_and_find_image_round_trip() {
        let project = project_with_one_image();
        let rel = project.get_current_rel_image_path_cloned().unwrap();
        assert_eq!(rel, PathBuf::from("img.tif"));

        // `find_image` takes an *absolute* path and relativizes it against
        // `images.root`; with no root set, "absolute" and "relative" are the
        // same string here.
        assert!(project.find_image(&PathBuf::from("img.tif")).is_some());
        assert!(project.find_image(&PathBuf::from("missing.tif")).is_none());
    }

    #[test]
    fn find_image_mut_allows_editing_and_rest_current_image_path_clears_the_selection() {
        let mut project = project_with_one_image();
        project
            .find_image_mut(&PathBuf::from("img.tif"))
            .unwrap()
            .file_size = 777;
        assert_eq!(
            project
                .images
                .list
                .get(Path::new("img.tif"))
                .unwrap()
                .file_size,
            777
        );

        assert!(project.get_current_image_path_cloned().is_some());
        project.rest_current_image_path();
        assert!(project.get_current_image_path_cloned().is_none());
        assert!(project.get_current_image_settings().is_none());
    }

    #[test]
    fn current_image_settings_are_none_when_the_selected_path_is_not_in_the_list() {
        let mut project = project_with_one_image();
        project.set_current_image_path(&PathBuf::from("ghost.tif"));
        assert!(project.get_current_image_settings().is_none());
        assert!(project.get_current_image_settings_mut().is_none());
        assert!(project.with_current_image_mut(|_| ()).is_none());
    }

    #[test]
    fn with_current_series_mut_returns_none_when_the_selected_series_is_missing() {
        let mut project = project_with_one_image();
        project.set_active_series(&99);
        assert!(
            project
                .with_current_series_mut(|series| series.image_width)
                .is_none()
        );
    }

    // -- get_reference_object --------------------------------------------------

    #[test]
    fn get_reference_object_generates_a_grid_of_circular_objects() {
        let project = ProjectWithRuntime::default();
        let objects = project.get_reference_object().expect("always returns Some");
        assert_eq!(objects.len(), 30000);

        let first = &objects[0];
        assert_eq!(first.id, ObjectId(1));
        assert_eq!(first.area, 31415);
        assert_eq!(first.mask_data.len(), 5 * 5, "a 5x5 mask per object");
        assert_eq!(
            first.bbox[2] - first.bbox[0],
            5,
            "bbox width matches the object width"
        );
        assert_eq!(
            first.bbox[3] - first.bbox[1],
            5,
            "bbox height matches the object height"
        );
        assert_eq!(first.segmentation_class, SegmentationClass(1));

        let last = &objects[objects.len() - 1];
        assert_eq!(last.id, ObjectId(30000));
    }

    // -- get_selected_series_idx --------------------------------------------

    #[test]
    fn get_selected_series_idx_defaults_to_zero_without_a_current_image() {
        let project = ProjectWithRuntime::default();
        assert_eq!(project.get_selected_series_idx(), 0);
    }

    #[test]
    fn get_selected_series_idx_reflects_the_current_images_selected_series() {
        let mut project = project_with_one_image();
        project.with_current_image_mut(|img| {
            img.series.insert(1, img.series[&0].clone());
        });
        project.set_active_series(&1);
        assert_eq!(project.get_selected_series_idx(), 1);
    }

    // -- add_image_and_read_meta (JVM-free branch only) ----------------------

    #[test]
    fn add_image_and_read_meta_rejects_an_unsupported_extension_without_touching_the_jvm() {
        let mut project = ProjectWithRuntime::default();
        let action = project.add_image_and_read_meta(Path::new("notes.txt"));
        assert_eq!(action, ProjectAction::Failure("Unsupported device".into()));
        assert!(project.images.list.is_empty());
    }

    // -- is_supported_image / is_supported_image_path ------------------------

    #[test]
    fn is_supported_image_checks_the_extension_case_insensitively() {
        let project = ProjectWithRuntime::default();
        assert!(project.is_supported_image(Path::new("a.tif")));
        assert!(project.is_supported_image(Path::new("a.TIF")));
        assert!(!project.is_supported_image(Path::new("a.txt")));
        assert!(!project.is_supported_image(Path::new("no_extension")));
    }

    #[test]
    fn is_supported_image_path_matches_against_the_supported_formats_list() {
        assert!(is_supported_image_path(Path::new("scan.czi")));
        assert!(is_supported_image_path(Path::new("scan.CZI")));
        assert!(!is_supported_image_path(Path::new("scan.pdf")));
        assert!(!is_supported_image_path(Path::new("no_extension")));
    }

    // -- collect_images_at_root / collect_images_parallel --------------------

    #[test]
    fn collect_images_at_root_skips_a_folder_named_results_case_insensitively() {
        // The "results" check happens before any filesystem access, so this
        // is safe to call on paths that don't even exist.
        assert!(collect_images_at_root(Path::new("/any/path/Results")).is_empty());
        assert!(collect_images_at_root(Path::new("/any/path/RESULTS")).is_empty());
    }

    #[test]
    fn collect_images_at_root_returns_empty_for_a_missing_directory() {
        assert!(
            collect_images_at_root(Path::new("/definitely/does/not/exist/anywhere")).is_empty()
        );
    }

    #[test]
    fn collect_images_at_root_recurses_and_skips_unsupported_files_and_results_folders() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("notes.txt"), b"not an image").unwrap();
        let sub = dir.path().join("sub");
        std::fs::create_dir(&sub).unwrap();
        std::fs::write(sub.join("doc.docx"), b"not an image").unwrap();
        let results = dir.path().join("results");
        std::fs::create_dir(&results).unwrap();
        // Even a supported extension inside "results" must never be opened -
        // if it were, this would fail trying to actually parse the file.
        std::fs::write(results.join("fake.tif"), b"not a real tiff").unwrap();

        assert!(collect_images_at_root(dir.path()).is_empty());
    }

    #[test]
    fn collect_images_parallel_delegates_to_collect_images_at_root() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("notes.txt"), b"nope").unwrap();
        let project = ProjectWithRuntime::default();
        assert!(project.collect_images_parallel(dir.path()).is_empty());
    }

    // -- scan_image_folder_and_add / apply_scanned_images --------------------

    #[test]
    fn scan_image_folder_and_add_is_a_noop_without_a_root() {
        let mut project = ProjectWithRuntime::default();
        project.scan_image_folder_and_add();
        assert!(project.images.list.is_empty());
    }

    #[test]
    fn scan_image_folder_and_add_finds_no_images_in_an_empty_root() {
        let dir = tempfile::tempdir().unwrap();
        let mut project = ProjectWithRuntime::default();
        project.images.root = Some(dir.path().to_path_buf());
        project.scan_image_folder_and_add();
        assert!(project.images.list.is_empty());
    }

    #[test]
    fn apply_scanned_images_clears_the_old_list_and_inserts_in_natural_sort_order() {
        let dir = tempfile::tempdir().unwrap();
        let mut project = project_with_one_image();
        let meta = sample_image_meta("Ch0", 400.0);
        let found = vec![
            (dir.path().join("img10.tif"), meta.clone()),
            (dir.path().join("img2.tif"), meta),
        ];
        project.apply_scanned_images(found);

        assert!(
            !project.images.list.contains_key(Path::new("img.tif")),
            "the previous list must be cleared"
        );
        let keys: Vec<_> = project.images.list.keys().cloned().collect();
        assert_eq!(
            keys,
            vec![PathBuf::from("img2.tif"), PathBuf::from("img10.tif")],
            "natural sort orders img2 before img10"
        );
    }

    // -- save_project_as_template / save_pipeline_as_template -----------------

    #[test]
    fn save_project_as_template_writes_classification_plate_and_pipelines() {
        let dir = tempfile::tempdir().unwrap();
        let mut project = ProjectWithRuntime::default();
        project.classification.classes_mut().push(class(1, "A"));
        let mut p = pipeline(1);
        p.name = Some("MyPipe".into());
        project.add_pipeline(p);

        let meta = evanalyzer_cfg::settings::meta_data::MetaData {
            name: "Tmpl".into(),
            ..Default::default()
        };
        let path = dir.path().join("out");
        project
            .save_project_as_template(meta, &path)
            .expect("save should succeed");

        let expected = dir
            .path()
            .join(format!("out.{}", PROJECT_FILE_TEMPLATE_EXTENSIONS));
        assert!(expected.exists());
        let content = std::fs::read_to_string(&expected).unwrap();
        let parsed: ProjectTemplate = serde_json::from_str(&content).unwrap();
        // classes()[0] is the auto-prepended Background class.
        assert_eq!(parsed.classification.classes().len(), 2);
        assert_eq!(parsed.classification.classes()[1].name, "A");
        assert_eq!(parsed.pipelines.len(), 1);
        assert_eq!(parsed.pipelines[0].meta.name, "MyPipe");
    }

    #[test]
    fn save_pipeline_as_template_writes_only_the_selected_pipelines_steps() {
        let dir = tempfile::tempdir().unwrap();
        let mut project = ProjectWithRuntime::default();
        let mut p = pipeline(1);
        p.steps = vec![PipelineStepSettings {
            enabled: true,
            command: PipelineCommand::Blur(Default::default()),
        }];
        project.add_pipeline(p);
        project.add_pipeline(pipeline(2));

        let meta = evanalyzer_cfg::settings::meta_data::MetaData {
            name: "StepTmpl".into(),
            ..Default::default()
        };
        let path = dir.path().join("pipe");
        project
            .save_pipeline_as_template(meta, PipelineId(1), &path)
            .expect("save should succeed");

        let expected = dir.path().join(format!("pipe.{}", PIPELINE_EXTENSIONS));
        assert!(expected.exists());
        let content = std::fs::read_to_string(&expected).unwrap();
        let parsed: PipelineTemplate = serde_json::from_str(&content).unwrap();
        assert_eq!(parsed.pipeline_steps.len(), 1);
    }

    #[test]
    fn save_pipeline_as_template_errors_for_an_unknown_pipeline_id() {
        let dir = tempfile::tempdir().unwrap();
        let mut project = ProjectWithRuntime::default();
        project.add_pipeline(pipeline(1));

        let meta = evanalyzer_cfg::settings::meta_data::MetaData::default();
        let path = dir.path().join("pipe");
        let result = project.save_pipeline_as_template(meta, PipelineId(99), &path);
        assert!(result.is_err());
        assert!(!path.with_extension(PIPELINE_EXTENSIONS).exists());
    }

    // -- new / new_project ----------------------------------------------------

    #[test]
    fn new_returns_an_arc_wrapped_default_project() {
        let project = ProjectWithRuntime::default();
        let fresh = project.new();
        // A fresh project always has exactly the auto-prepended Background
        // class and nothing else.
        assert_eq!(fresh.classification.classes().len(), 1);
        assert_eq!(
            fresh.classification.classes()[0].id,
            ObjectClass::BACKGROUND
        );
        assert!(fresh.images.list.is_empty());
    }

    #[test]
    fn new_project_creates_and_persists_a_fresh_default_project() {
        let dir = tempfile::tempdir().unwrap();
        let mut project = ProjectWithRuntime::default();
        let path = dir.path().join("newproj");

        let created = project.new_project(&path).expect("should succeed");

        let expected = dir
            .path()
            .join(format!("newproj.{}", PROJECT_FILE_EXTENSIONS));
        assert!(expected.exists());
        assert_eq!(created.tmp_settings.current_project, Some(expected));
        // A fresh project always has exactly the auto-prepended Background
        // class and nothing else.
        assert_eq!(created.classification.classes().len(), 1);
        assert_eq!(
            created.classification.classes()[0].id,
            ObjectClass::BACKGROUND
        );
    }

    // -- load_project schema version check ------------------------------------

    #[test]
    fn load_project_rejects_a_file_written_by_a_newer_schema_version() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("future.evaproj");

        let settings = ProjectSettings::default();
        let mut value = serde_json::to_value(&settings).unwrap();
        value["schemaVersion"] =
            serde_json::json!(evanalyzer_cfg::CURRENT_PROJECT_SCHEMA_VERSION + 1);
        std::fs::write(&path, serde_json::to_string(&value).unwrap()).unwrap();

        match load_project(&path) {
            Err(err) => assert!(err.to_string().contains("newer version")),
            Ok(_) => panic!("a newer schema version must be rejected"),
        }
    }

    // -- import_legacy_project (module-level wrapper) -------------------------

    fn minimal_legacy_project_json() -> String {
        r##"{
            "meta": { "name": "Legacy Demo" },
            "projectSettings": {
                "classification": { "classes": [] },
                "plate": { "imageFolder": "images" }
            },
            "imageSetup": {},
            "pipelines": []
        }"##
        .to_string()
    }

    #[test]
    fn import_legacy_project_converts_a_valid_file_and_leaves_current_project_path_unset() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("legacy.icproj");
        std::fs::write(&path, minimal_legacy_project_json()).unwrap();

        let (project, _warnings, legacy_image_folder) =
            import_legacy_project(&path).expect("should parse");
        assert_eq!(project.metadata.name, "Legacy Demo");
        assert_eq!(legacy_image_folder, Some("images".to_string()));
        assert!(project.tmp_settings.current_project.is_none());
    }

    #[test]
    fn import_legacy_project_errors_for_malformed_json() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("bad.icproj");
        std::fs::write(&path, "{ not json").unwrap();

        assert!(import_legacy_project(&path).is_err());
    }

    #[test]
    fn import_legacy_project_errors_for_a_missing_file() {
        let result = import_legacy_project(&PathBuf::from("/does/not/exist.icproj"));
        assert!(result.is_err());
    }
}
