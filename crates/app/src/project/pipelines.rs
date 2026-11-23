//! # pipelines
//!
//! The pipelines: their order, enabling them, adding one, and the pipeline
//! focus (only its channels and classes shown).
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-11-22
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.

use crate::project::ProjectWithRuntime;
use crate::project::{FocusChannels, PipelineFocus};
use evanalyzer_cfg::core_types::PipelineId;
use evanalyzer_cfg::settings::modules::ModuleDefinition;
use evanalyzer_cfg::settings::pipeline_settings::PipelineSettings;
use std::collections::BTreeMap;

impl ProjectWithRuntime {
    pub fn move_pipeline_up(&mut self, pipeline_id: PipelineId) {
        if let Some(i) = self.pipelines.iter().position(|c| c.id == pipeline_id) {
            if i > 0 {
                self.pipelines.swap(i, i - 1);
            }
        }
    }

    pub fn move_pipeline_down(&mut self, pipeline_id: PipelineId) {
        if let Some(i) = self.pipelines.iter().position(|c| c.id == pipeline_id) {
            if i < self.pipelines.len() - 1 {
                self.pipelines.swap(i, i + 1);
            }
        }
    }

    pub fn enable_pipeline(&mut self, enabled: bool, pipeline_id: PipelineId) {
        if let Some(i) = self.pipelines.iter().position(|c| c.id == pipeline_id) {
            self.pipelines[i].enabled = enabled;
        }
    }

    pub fn enable_pipeline_step(&mut self, enabled: bool, pipeline_id: PipelineId, step_id: usize) {
        if let Some(i) = self.pipelines.iter().position(|c| c.id == pipeline_id) {
            if let Some(step) = self.pipelines[i].steps.get_mut(step_id) {
                step.enabled = enabled;
            }
        }
    }

    pub fn add_pipeline(&mut self, mut pipeline_settings: PipelineSettings) {
        super::utils::assign_step_ids(&mut pipeline_settings.steps);
        self.pipelines.push(pipeline_settings);
        super::utils::assign_measurement_ids(self);
    }

    /// Focuses the given pipeline: only the image channels it reads (its
    /// source's and its load steps', also inside modules - all channels if it
    /// reads none) and its object classes are shown.
    /// Returns `false` (and clears any focus) if there is no such pipeline.
    pub fn focus_pipeline(&mut self, pipeline_id: PipelineId) -> bool {
        let Some(pipeline) = self.pipelines.iter().find(|p| p.id == pipeline_id) else {
            self.tmp_settings.pipeline_focus = None;
            return false;
        };
        let channels = focus_channels(pipeline, &self.modules);
        let pipeline_classes = evanalyzer_core::pipeline_classes(pipeline, &self.modules);
        self.tmp_settings.pipeline_focus = Some(PipelineFocus {
            pipeline_id,
            channels,
            visible_classes: pipeline_classes.iter().copied().collect(),
            pipeline_classes,
        });
        true
    }

    /// Updates the focus after the focused pipeline was edited: a changed
    /// image source or class list resets the focus to the new state (the
    /// user's tweaks made while focused are kept otherwise); a deleted
    /// pipeline ends the focus.
    pub fn refresh_pipeline_focus(&mut self) {
        let Some(focus) = self.tmp_settings.pipeline_focus.clone() else {
            return;
        };
        let Some(pipeline) = self.pipelines.iter().find(|p| p.id == focus.pipeline_id) else {
            self.tmp_settings.pipeline_focus = None;
            return;
        };
        let source_channels = focus_channels(pipeline, &self.modules);
        let source_changed = match (&source_channels, &focus.channels) {
            (FocusChannels::Only(new), FocusChannels::Only(old)) => !new.is_subset(old),
            (a, b) => a != b,
        };
        if source_changed
            || evanalyzer_core::pipeline_classes(pipeline, &self.modules) != focus.pipeline_classes
        {
            self.focus_pipeline(focus.pipeline_id);
        }
    }

    /// Ends the pipeline focus; the user's own visibility settings apply again.
    pub fn clear_pipeline_focus(&mut self) {
        self.tmp_settings.pipeline_focus = None;
    }

    pub fn focused_pipeline(&self) -> Option<PipelineId> {
        self.tmp_settings
            .pipeline_focus
            .as_ref()
            .map(|focus| focus.pipeline_id)
    }
}

/// The channels a focus on `pipeline` shows - see
/// [`ProjectWithRuntime::focus_pipeline`].
pub(super) fn focus_channels(
    pipeline: &PipelineSettings,
    modules: &[ModuleDefinition],
) -> FocusChannels {
    let channels = evanalyzer_core::pipeline_channels(pipeline, modules);
    if channels.is_empty() {
        FocusChannels::All
    } else {
        FocusChannels::Only(channels)
    }
}

impl ProjectWithRuntime {
    /// While a pipeline is focused, a channel visibility change from the UI
    /// updates the focus overlay instead of the saved settings. Returns
    /// whether it did.
    pub(super) fn set_focus_channels(&mut self, channel_visibility: &BTreeMap<i32, bool>) -> bool {
        let Some(focus) = &mut self.tmp_settings.pipeline_focus else {
            return false;
        };
        focus.channels = if channel_visibility.values().all(|&visible| visible) {
            FocusChannels::All
        } else {
            FocusChannels::Only(
                channel_visibility
                    .iter()
                    .filter(|(_, visible)| **visible)
                    .map(|(id, _)| *id)
                    .collect(),
            )
        };
        true
    }
}
