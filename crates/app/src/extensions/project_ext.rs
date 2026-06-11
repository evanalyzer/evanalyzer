use crate::extensions::classification_ext::ClassificationExt;
use crate::extensions::roi_ext::RoiExt;
use crate::extensions::utils::{get_relative_key, is_in_root, wavelength_to_rgb_u32};
use crate::project_owner::{ProjectTmpSettings, ProjectWithRuntime};
use bitvec::{order::Lsb0, vec::BitVec};
use evanalyzer_cfg::core_types::ImageAddress;
use evanalyzer_cfg::{PIPELINE_EXTENSIONS, PROJECT_FILE_EXTENSIONS, PROJECT_FILE_TEMPLATE_EXTENSIONS};
use evanalyzer_cfg::core_types::{InternalErrors, ObjectId, PipelineId};
use evanalyzer_cfg::settings::images_settings::{
    ChannelSettings, HistogramSettings, PixelSizeSettings, TStackSettings, ZStackSettings,
};
use evanalyzer_cfg::settings::meta_data::MetaData;
use evanalyzer_cfg::settings::pipeline_settings::PipelineSettings;
use evanalyzer_cfg::settings::templates::{PipelineTemplate, ProjectTemplate};
use evanalyzer_cfg::{
    core_types::ObjectClass,
    settings::{
        classification_settings::Class,
        images_settings::{ImageEntry, SeriesSettings},
        project_settings::ProjectSettings,
    },
};
use evanalyzer_cfg::{
    core_types::SegmentationClass, object_class_set_from_u32, settings::roi_settings::RoiSettings,
};
use evanalyzer_core::{ImageMeta, ImageReader, ReadMode, SUPPORTED_IMAGE_FORMATS};
use human_sort::compare;
use indexmap::IndexMap;
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

#[derive(PartialEq, Eq)]
pub enum SelectNewProjectRootAction {
    /// Everything went fine, image added.
    Success,

    /// Image in the new project root not found
    ImageNotFound,
}

#[derive(PartialEq)]
pub enum SaveProjectActions {
    /// Everything went fine, image added.
    Success,

    /// No project file yet selected
    PleaseSelectFile,

    Error,
}

pub trait ProjectExt {
    fn add_class_to_roi(&mut self, id: ObjectId, object_class: ObjectClass);
    fn remove_class_from_roi(&mut self, id: ObjectId, object_class: &ObjectClass);
    fn add_roi(&mut self, roi: &RoiSettings);
    fn get_rois(&self) -> Option<&[RoiSettings]>;
    fn delete_roi(&mut self, id: ObjectId);
    fn get_reference_roi(&self) -> Option<Vec<RoiSettings>>;
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
    fn count_rois_for_class(&self, class_id: &ObjectClass) -> usize;
}

impl ProjectExt for ProjectWithRuntime {
    /// Adds an object class to a specific ROI by its ID.
    ///
    /// This method finds the ROI with the given ID in the currently selected image series
    /// and adds the specified object class to it.
    ///
    /// # Arguments
    /// * `id` - The ObjectId of the ROI to modify
    /// * `object_class` - The ObjectClass to add to the ROI
    fn add_class_to_roi(&mut self, id: ObjectId, object_class: ObjectClass) {
        if let Some(series) = self.get_selected_image_series_mut() {
            if let Some(roi) = series.rois.iter_mut().find(|roi| roi.id == id) {
                roi.add_object_class(object_class);
            }
        }
    }

    /// Removes an object class from a specific ROI by its ID.
    ///
    /// This method finds the ROI with the given ID in the currently selected image series
    /// and removes the specified object class from it.
    ///
    /// # Arguments
    /// * `id` - The ObjectId of the ROI to modify
    /// * `object_class` - The ObjectClass to remove from the ROI
    fn remove_class_from_roi(&mut self, id: ObjectId, object_class: &ObjectClass) {
        if let Some(series) = self.get_selected_image_series_mut() {
            if let Some(roi) = series.rois.iter_mut().find(|roi| roi.id == id) {
                roi.remove_object_class(object_class);
            }
        }
    }

    fn add_roi(&mut self, roi: &RoiSettings) {
        if let Some(series) = self.get_selected_image_series_mut() {
            series.rois.push(roi.clone());
        }
    }

    fn get_rois(&self) -> Option<&[RoiSettings]> {
        self.get_selected_image_series()
            .map(|series| series.rois.as_slice())
    }

    fn delete_roi(&mut self, id: ObjectId) {
        if let Some(series) = self.get_selected_image_series_mut() {
            series.rois.retain(|roi| roi.id != id);
        }
    }

    fn get_reference_roi(&self) -> Option<Vec<RoiSettings>> {
        let mut rois = Vec::new();

        // Helper to generate a circular ROI
        let create_circle_roi = |id: u128, x_start: usize, y_start: usize| {
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
            RoiSettings {
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

            rois.push(create_circle_roi(i + 1, x, y));
        }
        Some(rois)
    }

    fn get_class_from_id(&self, id: &ObjectClass) -> Option<&Class> {
        self.classification.classes.iter().find(|c| c.id == *id)
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
                            measure: IndexMap::new(),
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
        self.classification.classes.clear();
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
                    rois: Vec::new(),
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
            // 1. Clear current list
            self.images.list.clear();

            // 2. Collect all valid image paths in parallel
            let mut found_images: Vec<(PathBuf, ImageMeta)> =
                self.collect_images_parallel(&root_folder);

            // Natural sort
            found_images.sort_by(|a, b| {
                let path_a = a.0.to_string_lossy();
                let path_b = b.0.to_string_lossy();
                compare(&path_a, &path_b)
            });

            // 3. Batch insert into your state (Single lock at the end)
            let start = Instant::now();
            for (path, meta) in found_images {
                // Call your internal add logic here
                // Note: Ensure your add_image logic is split so you can
                // insert pre-computed metadata without re-calculating.
                self.add_image(&mut &path, &meta);
            }
            let duration = start.elapsed();
            info!("Added images {:?}", duration);
        }
    }

    fn collect_images_parallel(&self, dir: &Path) -> Vec<(PathBuf, ImageMeta)> {
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
                    self.collect_images_parallel(&path)
                } else if self.is_supported_image(&path) {
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

    fn is_supported_image(&self, path: &Path) -> bool {
        path.extension()
            .and_then(|ext| ext.to_str()) // Convert OsStr to &str
            .map(|ext_str| {
                let ext_lower = ext_str.to_lowercase();
                SUPPORTED_IMAGE_FORMATS.iter().any(|&fmt| fmt == ext_lower)
            })
            .unwrap_or(false) // Return false if no extension exists
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

        let json = serde_json::to_string_pretty(&self.settings)
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
                warn!("Could not parse pipeline template {:?}: {}", template_file, err);
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

    fn count_rois_for_class(&self, class_id: &ObjectClass) -> usize {
        let manual = self
            .get_rois()
            .unwrap_or_default()
            .iter()
            .filter(|r| r.object_class.contains(class_id))
            .count();
        let preview = self
            .tmp_settings
            .preview_rois
            .iter()
            .filter(|r| r.object_class.contains(class_id))
            .count();
        manual + preview
    }
}

pub fn load_project(path: &PathBuf) -> Result<ProjectWithRuntime, InternalErrors> {
    let data = fs::read_to_string(path.clone())?;
    let inner: ProjectSettings =
        serde_json::from_str(&data).map_err(|e| InternalErrors::ParseError(e.to_string()))?;
    let mut project = ProjectWithRuntime {
        settings: inner,
        tmp_settings: ProjectTmpSettings::default(),
    };
    project.tmp_settings.current_project = Some(path.clone());
    Ok(project)
}
