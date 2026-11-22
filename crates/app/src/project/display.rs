//! # display
//!
//! How the images are shown: channel visibility, histograms, wavelengths
//! and colours, grayscale - for the current image and for all.
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-11-22
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.

use crate::project::FocusChannels;
use crate::project::ProjectWithRuntime;
use crate::project::image_entry::ImageEntryExt;
use crate::project::images::GlobalImageSettingsExt;
use crate::project::images::ImageListExt;
use crate::project::utils::{ASSIGNED_WAVELENGTHS, wavelength_colors};
use evanalyzer_cfg::core_types::ZProjection;
use evanalyzer_cfg::settings::images_settings::SeriesSettings;
use evanalyzer_cfg::settings::images_settings::{
    ChannelSettings, GlobalImageSettings, HistogramSettings,
};
use log::warn;
use std::collections::BTreeMap;

impl ProjectWithRuntime {
    /// Returns a map of channel IDs to their visibility, falling back to global settings if not set locally.
    pub fn get_image_channel_visibilities(&self) -> BTreeMap<i32, bool> {
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
        // Pipeline focus overrides the user's own settings while it is set.
        if let Some(focus) = &self.tmp_settings.pipeline_focus {
            for (idx, visible) in visibilities.iter_mut() {
                *visible = match &focus.channels {
                    FocusChannels::All => true,
                    FocusChannels::Only(channels) => channels.contains(idx),
                };
            }
        }
        // Grayscale mode shows one channel at a time - also over a focus
        // naming several, or the user's own settings after one: the selected
        // channel if visible, else the first visible (as `set_grayscale`).
        if self.images.settings.grayscale {
            let selected = self.get_selected_image_channel_idx();
            let keep = if visibilities.get(&selected) == Some(&true) {
                Some(selected)
            } else {
                visibilities
                    .iter()
                    .find(|(_, visible)| **visible)
                    .map(|(id, _)| *id)
            };
            for (id, visible) in visibilities.iter_mut() {
                *visible = Some(*id) == keep;
            }
        }
        visibilities
    }

    /// Returns a list of channel IDs that are currently marked as visible.
    pub fn get_image_channel_visibilities_vec(&self) -> Vec<i32> {
        self.get_image_channel_visibilities()
            .into_iter()
            .filter_map(|(id, visible)| if visible { Some(id) } else { None })
            .collect()
    }

    /// Returns histogram settings for all channels in the current series, with global fallbacks.
    pub fn get_image_channel_histograms(&self) -> BTreeMap<i32, Option<HistogramSettings>> {
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

    /// The scaling the selected channel is drawn with: the image's own, else
    /// the project's (see `get_image_channel_histograms`).
    pub fn get_histograms_from_selected_channel(&self) -> Option<&HistogramSettings> {
        let (_, series) = self.get_current_image_channel_settings()?;
        let idx = self.get_selected_image_channel_idx();
        let channel = series.channels.get(&idx)?;
        channel.histogram.as_ref().or_else(|| {
            self.images
                .settings
                .channels
                .get(&idx)
                .and_then(|c| c.histogram.as_ref())
        })
    }

    /// Updates channel visibility preferences for the current image series.
    /// While a pipeline is focused, the focus overlay changes instead.
    pub fn set_image_preferences(&mut self, channel_visibility: &BTreeMap<i32, bool>) {
        if self.set_focus_channels(channel_visibility) {
            return;
        }
        self.with_current_series_mut(|series| {
            for (id, visible) in channel_visibility {
                if let Some(channel_arc) = series.channels.get_mut(id) {
                    channel_arc.visible = Some(*visible);
                }
            }
        });
    }

    /// Updates global channel visibility preferences. While a pipeline is
    /// focused, the focus overlay changes instead.
    pub fn set_global_preferences(&mut self, channel_visibility: &BTreeMap<i32, bool>) {
        if self.set_focus_channels(channel_visibility) {
            return;
        }
        for (id, visible) in channel_visibility {
            let channel_arc =
                self.images
                    .settings
                    .channels
                    .entry(*id)
                    .or_insert_with(|| ChannelSettings {
                        name: "".into(),
                        emission_wave_length: None,
                        visible: None,
                        histogram: None,
                    });

            // This allows mutation by cloning if shared, or just borrowing if unique
            channel_arc.visible = Some(*visible);
        }
    }

    pub fn get_emission_wave_length(&self, channel_id: i32) -> f32 {
        // 1. A global value set for this channel wins. A global entry that
        //    only carries other settings (visibility, histogram) has none.
        self.images
            .settings
            .channels
            .get(&channel_id)
            .and_then(|channel| channel.emission_wave_length)
            // 2. Otherwise the current image's own value for the channel
            .or_else(|| {
                self.get_current_image_channel_settings()
                    .and_then(|(_, series)| series.channels.get(&channel_id))
                    .and_then(|channel| channel.emission_wave_length)
            })
            // 3. Unknown
            .unwrap_or(0.0)
    }

    /// How the images are shown, for "Images with objects drawn": the
    /// project's z-projection, channel visibility and display ranges, and
    /// (relative path -> theirs) the images whose own settings differ.
    pub fn drawn_image_display(
        &self,
    ) -> (
        crate::results::ImageDisplay,
        BTreeMap<String, crate::results::ImageDisplay>,
    ) {
        let global = &self.images.settings;
        let project = image_display(global, None);
        let own = self
            .images
            .list
            .0
            .iter()
            .filter_map(|(rel_path, entry)| {
                let series = entry.series.get(&entry.active_series(global))?;
                let display = image_display(global, Some(series));
                (display != project).then(|| (rel_path.to_string_lossy().into_owned(), display))
            })
            .collect();
        (project, own)
    }

    /// The colour each channel of the current image is shown in - see
    /// `utils::channel_colors` - or white for all of them in grayscale mode.
    pub fn get_channel_display_colors(&self) -> BTreeMap<i32, [f32; 3]> {
        let channels: Vec<(i32, f32)> = self
            .get_current_image_channel_settings()
            .map(|(_, series)| {
                series
                    .channels
                    .keys()
                    .map(|id| (*id, self.get_emission_wave_length(*id)))
                    .collect()
            })
            .unwrap_or_default();
        self.channel_display_colors(&channels)
    }

    /// [`Self::get_channel_display_colors`] for the given `(channel,
    /// emission wavelength)` list - e.g. the one the image file reports.
    pub fn channel_display_colors(&self, channels: &[(i32, f32)]) -> BTreeMap<i32, [f32; 3]> {
        if self.images.settings.grayscale {
            return channels
                .iter()
                .map(|(id, _)| (*id, [1.0, 1.0, 1.0]))
                .collect();
        }
        wavelength_colors(channels)
    }

    /// Channels without a wavelength (`channels`: channel, the image's
    /// wavelength, `<= 1.0` none) get one of [`ASSIGNED_WAVELENGTHS`] for
    /// the project - not one a channel already shows. Not for a single
    /// channel (it stays white). Whether anything was assigned.
    pub fn assign_missing_wavelengths(&mut self, channels: &[(i32, f32)]) -> bool {
        if channels.len() < 2 {
            return false;
        }
        let project_nm = |p: &Self, id: i32| {
            p.images
                .settings
                .channels
                .get(&id)
                .and_then(|c| c.emission_wave_length)
                .filter(|nm| *nm > 1.0)
        };
        let shown: Vec<f32> = channels
            .iter()
            .filter_map(|(id, nm)| project_nm(self, *id).or((*nm > 1.0).then_some(*nm)))
            .collect();
        // A colour chosen for the project - white too (0) - is not missing.
        let chosen = |p: &Self, id: i32| {
            p.images
                .settings
                .channels
                .get(&id)
                .is_some_and(|c| c.emission_wave_length.is_some())
        };
        let missing: Vec<i32> = channels
            .iter()
            .filter(|(id, nm)| *nm <= 1.0 && !chosen(self, *id))
            .map(|(id, _)| *id)
            .collect();
        if missing.is_empty() {
            return false;
        }
        // Not a colour a channel already shows (within 15 nm).
        let free: Vec<f32> = ASSIGNED_WAVELENGTHS
            .iter()
            .copied()
            .filter(|nm| !shown.iter().any(|s| (s - nm).abs() < 15.0))
            .collect();
        let palette = if free.is_empty() {
            ASSIGNED_WAVELENGTHS.to_vec()
        } else {
            free
        };
        for (id, nm) in missing.into_iter().zip(palette.into_iter().cycle()) {
            self.images
                .settings
                .set_global_emission_wavel_length(id, nm);
        }
        true
    }

    /// The open image's scaling becomes the project's, for every image: the
    /// images' own scalings are removed.
    pub fn apply_histograms_to_all_images(&mut self) {
        let current = self.get_image_channel_histograms();
        for (idx, histogram) in current {
            if let Some(histogram) = histogram {
                self.images
                    .settings
                    .channels
                    .entry(idx)
                    .or_default()
                    .histogram = Some(histogram);
            }
        }
        for image in self.images.list.values_mut() {
            for series in image.series.values_mut() {
                for channel in series.channels.values_mut() {
                    channel.histogram = None;
                }
            }
        }
    }

    /// Links (one range for all three) or unlinks a colour image's red,
    /// green and blue display ranges; linking gives all channels of the
    /// open image the selected channel's range.
    pub fn set_rgb_linked(&mut self, linked: bool) {
        self.images.settings.rgb_unlinked = !linked;
        if let (true, Some(window)) = (linked, self.get_histograms_from_selected_channel().cloned())
        {
            self.set_image_histogram_settings_for_all_channels(
                window.min,
                window.max,
                window.min_limit,
                window.max_limit,
            );
        }
    }

    /// The open image's display range for every channel - a colour image's
    /// red, green and blue while they are linked.
    pub fn set_image_histogram_settings_for_all_channels(
        &mut self,
        min: f32,
        max: f32,
        min_limit: f32,
        max_limit: f32,
    ) {
        let channels: Vec<i32> = self.get_image_channel_histograms().into_keys().collect();
        for channel in channels {
            self.set_image_histogram_settings_for_channel(channel, min, max, min_limit, max_limit);
        }
    }

    /// Switches grayscale mode. Switching it on leaves one channel visible
    /// (gray channels can't be told apart when overlaid): the selected one
    /// if it is visible, else the first visible one.
    pub fn set_grayscale(&mut self, grayscale: bool) {
        self.images.settings.grayscale = grayscale;
        if !grayscale {
            return;
        }
        let visibilities = self.get_image_channel_visibilities();
        let visible: Vec<i32> = visibilities
            .iter()
            .filter(|(_, visible)| **visible)
            .map(|(id, _)| *id)
            .collect();
        if visible.len() <= 1 {
            return;
        }
        let selected = self.get_selected_image_channel_idx();
        let keep = if visible.contains(&selected) {
            selected
        } else {
            visible[0]
        };
        let only: BTreeMap<i32, bool> = visibilities.keys().map(|id| (*id, *id == keep)).collect();
        self.set_global_preferences(&only);
    }

    /// `requested` channel visibilities as grayscale mode allows them: at
    /// most one visible - the one just switched on, which hides the others.
    /// Unchanged outside grayscale mode.
    pub fn with_one_visible_channel(&self, requested: &BTreeMap<i32, bool>) -> BTreeMap<i32, bool> {
        if !self.images.settings.grayscale {
            return requested.clone();
        }
        let visible: Vec<i32> = requested
            .iter()
            .filter(|(_, visible)| **visible)
            .map(|(id, _)| *id)
            .collect();
        if visible.len() <= 1 {
            return requested.clone();
        }
        let before = self.get_image_channel_visibilities();
        let selected = self.get_selected_image_channel_idx();
        let keep = visible
            .iter()
            .find(|id| !before.get(id).copied().unwrap_or(false))
            .copied()
            .or_else(|| visible.contains(&selected).then_some(selected))
            .unwrap_or(visible[0]);
        requested.keys().map(|id| (*id, *id == keep)).collect()
    }

    /// Updates histogram settings for the active channel of the current image.
    pub fn set_image_histogram_settings_for_active_channel(
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
    pub fn set_image_histogram_settings_for_channel(
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
}

/// How an image with `series` settings (`None`: the project's) is shown -
/// each value its own, else the project's: the z-projection its analysis
/// reads, each channel's visibility and display range (only the channels
/// not shown auto-scaled).
pub(super) fn image_display(
    global: &GlobalImageSettings,
    series: Option<&SeriesSettings>,
) -> crate::results::ImageDisplay {
    let z_projection = series
        .and_then(|s| s.z_stack.as_ref())
        .or(global.z_stack.as_ref())
        .map_or(ZProjection::None, |z| z.z_projection.z_projection());
    let mut ids: Vec<i32> = global.channels.keys().copied().collect();
    if let Some(series) = series {
        ids.extend(series.channels.keys().copied());
    }
    ids.sort_unstable();
    ids.dedup();
    let channels = ids
        .into_iter()
        .map(|id| {
            let own = series.and_then(|s| s.channels.get(&id));
            let project = global.channels.get(&id);
            let visible = own
                .and_then(|c| c.visible)
                .or(project.and_then(|c| c.visible))
                .unwrap_or(true);
            let range = own
                .and_then(|c| c.histogram.as_ref())
                .or(project.and_then(|c| c.histogram.as_ref()))
                .map(|h| (h.min, h.max));
            (id, crate::results::ChannelDisplay { visible, range })
        })
        // Shown and auto-scaled is what a channel missing here gets: left
        // out, so an image without settings of its own equals the project.
        .filter(|(_, shown)| *shown != crate::results::ChannelDisplay::default())
        .collect();
    crate::results::ImageDisplay {
        z_projection,
        channels,
    }
}
