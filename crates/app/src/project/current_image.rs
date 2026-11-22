//! # current_image
//!
//! The current image: its entry, series and channel, the plane and frame
//! shown, pixel sizes and frame interval - with the project's settings as
//! fallback.
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-11-22
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.

use crate::images::image_entry::ImageEntryExt;
use crate::images::image_list::ImageListExt;
use crate::project::ProjectWithRuntime;
use crate::project::objects::ShownPlane;
use crate::project::utils::get_relative_key;
use evanalyzer_cfg::core_types::ZProjection;
use evanalyzer_cfg::settings::images_settings::ImageEntry;
use evanalyzer_cfg::settings::images_settings::SeriesSettings;
use evanalyzer_cfg::settings::images_settings::{
    ChannelSettings, PixelSizeSettings, TStackSettings, ZStackHandling, ZStackSettings,
};
use std::path::PathBuf;

impl ProjectWithRuntime {
    /// Internal helper to get the relative key of the currently active image.
    pub fn get_current_relative_path(&self) -> Option<PathBuf> {
        let path = self.get_current_image_path_cloned()?;
        get_relative_key(&path, self.images.root.as_ref())
    }

    /// Internal helper to execute a closure with a mutable reference to the current image entry.
    pub fn with_current_image_mut<F, R>(&mut self, f: F) -> Option<R>
    where
        F: FnOnce(&mut ImageEntry) -> R,
    {
        let relative = self.get_current_relative_path()?;
        self.images.list.get_mut(&relative).map(|entry| f(entry))
    }

    /// Internal helper to execute a closure with a mutable reference to the current series settings.
    pub fn with_current_series_mut<F, R>(&mut self, f: F) -> Option<R>
    where
        F: FnOnce(&mut SeriesSettings) -> R,
    {
        let project_series = self.images.settings.selected_series;
        self.with_current_image_mut(|image| {
            let active = image.series_for(project_series);
            if let Some(series) = image.series.get_mut(&active) {
                return Some(f(series));
            }
            None
        })?
    }

    // --- Public API ---

    /// Returns a clone of the current image path if set.
    pub fn get_current_image_path_cloned(&self) -> Option<PathBuf> {
        self.tmp_settings.current_image.clone()
    }

    // Returns the relative image path of the currently selected image
    pub fn get_current_rel_image_path_cloned(&self) -> Option<PathBuf> {
        let Some(path) = self.get_current_image_path_cloned() else {
            return None;
        };
        get_relative_key(&path, self.images.root.as_ref()).clone()
    }

    /// Sets the path of the currently active image.
    pub fn set_current_image_path(&mut self, path: &PathBuf) {
        // Another image: the last one's preview objects aren't its.
        if self.tmp_settings.current_image.as_ref() != Some(path) {
            self.forget_preview_objects();
            self.tmp_settings.current_image_planes = None;
        }
        self.tmp_settings.current_image = Some(path.into());
    }

    pub fn rest_current_image_path(&mut self) {
        self.tmp_settings.current_image = None;
        self.tmp_settings.current_image_planes = None;
    }

    /// The current image's number of z-planes and frames (from its
    /// metadata): the shown plane is clamped to them, as the image reader
    /// clamps what it reads.
    pub fn set_current_image_planes(&mut self, nr_z_stacks: i32, nr_t_stacks: i32) {
        self.tmp_settings.current_image_planes = Some((nr_z_stacks, nr_t_stacks));
    }

    /// Returns the settings for the currently active image.
    pub fn get_current_image_settings(&self) -> Option<&ImageEntry> {
        let path = self.get_current_image_path_cloned()?;
        self.find_image(&path)
    }

    /// Returns the settings for the currently active image.
    pub fn get_current_image_settings_mut(&mut self) -> Option<&mut ImageEntry> {
        let path = self.get_current_image_path_cloned()?;
        self.find_image_mut(&path)
    }

    /// Returns the active series index and its settings for the current image.
    pub fn get_current_image_channel_settings(&self) -> Option<(i32, &SeriesSettings)> {
        let image = self.get_current_image_settings()?;
        let active = image.active_series(&self.images.settings);
        image.series.get(&active).map(|s| (active, s))
    }

    /// Returns the index of the currently selected series.
    pub fn get_selected_series_idx(&self) -> i32 {
        let global = &self.images.settings;
        self.get_current_image_settings()
            .map(|img| img.active_series(global))
            .unwrap_or_else(|| global.selected_series.unwrap_or(0))
    }

    /// Selects the series for every image of the project - shown and
    /// analysed (see `ImageEntry::active_series`).
    pub fn set_active_series(&mut self, selected_series: &i32) {
        // Another series: the preview's objects were the last one's.
        if self.get_selected_series_idx() != *selected_series {
            self.forget_preview_objects();
        }
        self.images.settings.selected_series = Some(*selected_series);
    }

    /// Sets the Z-stack settings for the current image series.
    pub fn set_image_z_stack(&mut self, z_stack: &ZStackSettings) {
        self.with_current_series_mut(|series| {
            series.z_stack = Some(z_stack.clone());
        });
    }

    /// Retrieves the active Z-stack settings (Local -> Global -> Default).
    pub fn get_z_stack(&self) -> Option<&ZStackSettings> {
        let image_z_stack = self
            .get_current_image_channel_settings()
            .and_then(|(_, s)| s.z_stack.as_ref());

        match image_z_stack {
            Some(z) => Some(z),
            None => self.images.settings.z_stack.as_ref(),
        }
    }

    /// The z-plane the viewer shows - `None` for a z-projection (no one
    /// plane). A volume (3D) is shown plane by plane.
    pub fn get_shown_z_plane(&self) -> Option<i32> {
        let Some(z) = self.get_z_stack() else {
            return Some(0);
        };
        match z.z_projection {
            ZStackHandling::SingleStack | ZStackHandling::AllStacks | ZStackHandling::Volume3d => {
                Some(z.z_range.as_ref().map_or(0, |r| *r.start()))
            }
            ZStackHandling::MaxIntensity
            | ZStackHandling::MinIntensity
            | ZStackHandling::AvgIntensity
            | ZStackHandling::SumIntensity
            | ZStackHandling::TakeTheMiddle => None,
        }
    }

    /// The z-projection the viewer shows - `None` for one plane.
    pub fn get_shown_z_projection(&self) -> ZProjection {
        self.get_z_stack()
            .map_or(ZProjection::None, |z| z.z_projection.z_projection())
    }

    /// The plane the viewer shows: the stored z-plane and frame, clamped to
    /// the current image's - like the image reader, which reads the last
    /// frame for a frame beyond it (e.g. frame 13 kept from a time-lapse
    /// while a single-frame image is open). Unclamped until the image's
    /// metadata was read.
    pub fn get_shown_plane(&self) -> ShownPlane {
        let z = self.get_shown_z_plane();
        let t = self.get_t_stack().map_or(0, |t| t.t_stack);
        match self.tmp_settings.current_image_planes {
            Some((nr_z, nr_t)) => ShownPlane {
                z: z.map(|z| z.min(nr_z - 1).max(0)),
                t: t.min(nr_t - 1).max(0),
            },
            None => ShownPlane { z, t },
        }
    }

    /// Retrieves the active T-stack settings (Local -> Global -> Default).
    pub fn get_t_stack(&self) -> Option<&TStackSettings> {
        let image_t_stack = self
            .get_current_image_channel_settings()
            .and_then(|(_, s)| s.t_stack.as_ref());

        match image_t_stack {
            Some(z) => Some(z),
            None => self.images.settings.t_stack.as_ref(),
        }
    }

    /// Retrieves the active pixel size settings (Global -> Local -> Default).
    pub fn get_pixel_sizes(&self) -> PixelSizeSettings {
        // 1. Check Global first
        self.images
            .settings
            .pixel_sizes
            .clone()
            // 2. If Global was None, return the Local value
            .unwrap_or_else(|| {
                self.get_current_image_channel_settings()
                    .map(|(_, s)| s.pixel_sizes.clone())
                    .unwrap_or_default()
            })
    }

    /// Returns the index of the selected channel for the current image.
    pub fn get_selected_image_channel_idx(&self) -> i32 {
        self.get_current_image_channel_settings()
            .and_then(|(_, s)| s.selected_channel)
            .unwrap_or_else(|| self.images.settings.selected_channel.unwrap_or(0))
    }

    /// Returns the settings for the currently selected channel.
    pub fn get_selected_image_channel(&self) -> Option<&ChannelSettings> {
        let idx = self.get_selected_image_channel_idx();
        self.get_current_image_channel_settings()
            .and_then(|(_, s)| s.channels.get(&idx).clone())
    }

    /// Sets the T-stack settings for the current image series.
    pub fn set_image_t_stack(&mut self, t_stack: &TStackSettings) {
        self.with_current_series_mut(|series| {
            series.t_stack = Some(t_stack.clone());
        });
    }

    /// The frame interval [s] set for the project - `None`: the image's own.
    pub fn get_frame_interval(&self) -> Option<f32> {
        self.images
            .settings
            .t_stack
            .as_ref()
            .and_then(|t| t.frame_interval)
    }

    /// The frame interval [s] to use for an image whose metadata says
    /// `image_value`: the project's if set, else the image's, else
    /// [`DEFAULT_FRAME_INTERVAL_S`]. Non-positive values count as unset.
    pub fn get_effective_frame_interval(
        &self,
        image_value: Option<f32>,
    ) -> (f32, ConfigValueSource) {
        let valid = |v: Option<f32>| v.filter(|v| v.is_finite() && *v > 0.0);
        if let Some(v) = valid(self.get_frame_interval()) {
            (v, ConfigValueSource::Project)
        } else if let Some(v) = valid(image_value) {
            (v, ConfigValueSource::Image)
        } else {
            (DEFAULT_FRAME_INTERVAL_S, ConfigValueSource::Default)
        }
    }

    /// Sets (`Some`) or clears (`None`, back to the image's own) the
    /// project's frame interval [s].
    pub fn set_frame_interval(&mut self, seconds: Option<f32>) {
        match (&mut self.images.settings.t_stack, seconds) {
            (Some(t_stack), _) => t_stack.frame_interval = seconds,
            (None, Some(_)) => {
                self.images.settings.t_stack = Some(TStackSettings {
                    frame_interval: seconds,
                    ..Default::default()
                })
            }
            (None, None) => {}
        }
    }

    /// Sets the pixel size settings for the current image series.
    ///
    /// `pz`: `None` if the z spacing is unknown.
    pub fn set_image_pixel_size_settings(&mut self, px: f32, py: f32, pz: Option<f32>) {
        self.with_current_series_mut(|series| {
            series.pixel_sizes = PixelSizeSettings {
                x: px,
                y: py,
                z: pz,
            };
        });
    }

    /// Sets the selected channel index for the current image series.
    pub fn set_image_selected_channel(&mut self, selected_channel: &i32) {
        self.with_current_series_mut(|series| {
            series.selected_channel = Some(*selected_channel);
        });
    }

    /// Returns the active series index for the current image.
    pub fn get_selected_image_series_idx(&self) -> Option<i32> {
        self.get_current_image_settings()
            .map(|img| img.active_series(&self.images.settings))
    }

    pub fn get_selected_image_series(&self) -> Option<&SeriesSettings> {
        let image = self.get_current_image_settings()?;
        image
            .series
            .get(&image.active_series(&self.images.settings))
    }

    pub fn get_selected_image_series_mut(&mut self) -> Option<&mut SeriesSettings> {
        let project_series = self.images.settings.selected_series;
        let image = self.get_current_image_settings_mut()?;
        let active = image.series_for(project_series);
        image.series.get_mut(&active)
    }
}

/// The frame interval [s] when neither the project nor the image sets one.
pub const DEFAULT_FRAME_INTERVAL_S: f32 = 1.0;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConfigValueSource {
    Project,
    Image,
    Default,
}
