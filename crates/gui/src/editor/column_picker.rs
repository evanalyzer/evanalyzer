//! # column_picker
//!
//! The column picker (`ColumnPicker` in `ui/results/results_panel.slint`,
//! design canvas "Results rework", ColumnPanel E) for whoever picks a
//! table's columns - the Results List and the Exports tab: the same parts
//! (starting sets, groups, the intensity grid, the coloc builder) made of
//! the same columns, and the same actions on what is picked. The picked
//! columns and where they are kept are the caller's.
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-11-24
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.

pub(crate) use super::results_state_controller::coloc::{
    COLOC_BOX, ColocSettings, SectionUi, adapt_column, coloc_classes, coloc_groups,
    default_settings, offered_columns, picker_sections,
};
pub(crate) use super::results_state_controller::column_panel::ColocGroupUi;
use super::results_state_controller::column_panel::{
    self, BLOCK_SEPARATOR, coloc_with, intensity_grid, intensity_heads, preset_cards, presets,
};
use crate::{ColumnPreset, IntensityCell, MultiSelectItem};
use evanalyzer_app::results::{Column, ColumnEntry};
use evanalyzer_cfg::core_types::ObjectClass;
use evanalyzer_cfg::settings::classification_settings::Class;
use std::collections::HashMap;

/// What the picker is made of: the results' own columns, the classes, the
/// coloc settings, the sections folded by hand and the search.
pub(crate) struct PickerSource<'a> {
    /// The results' own columns (`get_available_columns`).
    pub raw: &'a [ColumnEntry],
    pub classes: &'a [Class],
    pub settings: &'a ColocSettings,
    pub open: &'a HashMap<String, bool>,
    /// The search, lower case.
    pub filter: &'a str,
}

/// Every part of the picker, ready for Slint (see `ColumnPicker`).
pub(crate) struct PickerParts {
    pub sections: Vec<SectionUi>,
    pub coloc_sections: Vec<SectionUi>,
    pub coloc_title: String,
    pub coloc_open: bool,
    pub partner_items: Vec<MultiSelectItem>,
    pub partner_summary: String,
    pub partner_order: String,
    pub count: i32,
    pub only_occurring: bool,
    pub presets: Vec<ColumnPreset>,
    pub intensity_channels: Vec<String>,
    pub intensity_stats: Vec<slint::SharedString>,
    pub intensity_cells: Vec<IntensityCell>,
    pub intensity_title: String,
    pub intensity_open: bool,
    pub coloc_groups: Vec<ColocGroupUi>,
}

impl PickerSource<'_> {
    /// The columns the picker offers: the results' own, their coloc columns
    /// as the settings make them, and the picked coloc columns they don't.
    pub(crate) fn offered(&self, selected: &[Column]) -> Vec<ColumnEntry> {
        offered_columns(self.raw, self.settings, selected, self.classes)
    }

    /// The picker's parts over `available` (see [`Self::offered`]) with
    /// `selected` picked.
    pub(crate) fn parts(&self, available: &[ColumnEntry], selected: &[Column]) -> PickerParts {
        let (sections, coloc_sections, coloc_title, coloc_open) = picker_sections(
            available,
            selected,
            self.classes,
            self.open,
            self.filter,
            false,
        );
        let partner_items: Vec<MultiSelectItem> = coloc_classes(self.raw)
            .into_iter()
            .map(|class| {
                let known = self.classes.iter().find(|c| c.global() == class);
                MultiSelectItem {
                    key: class.to_u32().unwrap_or_default().to_string().into(),
                    value: known
                        .map(|c| c.name.clone())
                        .unwrap_or_else(|| format!("class {}", class.to_u32().unwrap_or_default()))
                        .into(),
                    color: known.map_or(slint::Color::default(), |c| {
                        super::results_state_controller::class_color_to_slint(c.color)
                    }),
                    group: "".into(),
                    selected: self.settings.partners.contains(&class),
                }
            })
            .collect();
        let partner_summary = match self.settings.partners.len() {
            0 => "None".to_string(),
            n if n == partner_items.len() => "All classes".to_string(),
            _ => partner_items
                .iter()
                .filter(|item| item.selected)
                .map(|item| item.value.to_string())
                .collect::<Vec<_>>()
                .join(", "),
        };
        let partner_order = partner_items
            .iter()
            .filter(|item| item.selected)
            .map(|item| item.value.to_string())
            .collect::<Vec<_>>()
            .join(", ");
        let presets = preset_cards(&presets(available, self.raw), selected);
        let (intensity_channels, intensity_cells) =
            intensity_grid(available, selected, self.classes);
        let intensity_section = sections.iter().find(|section| section.group == "intensity");
        let intensity_title = intensity_section
            .map(|section| section.title.clone())
            .unwrap_or_default();
        let intensity_open = intensity_section.is_some_and(|section| section.open);
        let coloc_groups = column_panel::coloc_group_blocks(
            available,
            &coloc_groups(self.raw),
            selected,
            self.settings,
            self.classes,
            self.open,
            self.filter,
        );
        PickerParts {
            sections,
            coloc_sections,
            coloc_title,
            coloc_open,
            partner_items,
            partner_summary,
            partner_order,
            count: i32::from(self.settings.count_partners),
            only_occurring: self.settings.only_occurring,
            presets,
            intensity_channels,
            intensity_stats: intensity_heads(),
            intensity_cells,
            intensity_title,
            intensity_open,
            coloc_groups,
        }
    }

    /// The columns of the section `group` (a coloc block's kind:
    /// `"<group>\u{1}<block>"`) the search shows - for its All / None.
    pub(crate) fn section_columns(&self, available: &[ColumnEntry], group: &str) -> Vec<Column> {
        let (group, block) = match group.split_once(BLOCK_SEPARATOR) {
            Some((group, number)) => {
                let groups = coloc_groups(self.raw);
                let block = number
                    .parse::<usize>()
                    .ok()
                    .and_then(|n| groups.get(n.wrapping_sub(1)).cloned());
                (group, block)
            }
            None => (group, None),
        };
        available
            .iter()
            .filter(|entry| entry.group == group)
            .filter(|entry| match &block {
                Some(block) => coloc_with(&entry.key)
                    .is_some_and(|with| !with.is_empty() && with.iter().all(|c| block.contains(c))),
                None => true,
            })
            .filter(|entry| {
                self.filter.is_empty() || entry.display_name.to_lowercase().contains(self.filter)
            })
            .map(|entry| entry.key.clone())
            .collect()
    }

    /// The starting set `key`'s columns among `available` - `None`: no such
    /// set here.
    pub(crate) fn preset_columns(
        &self,
        available: &[ColumnEntry],
        key: &str,
    ) -> Option<Vec<Column>> {
        presets(available, self.raw)
            .into_iter()
            .find(|set| set.key == key)
            .map(|set| set.columns)
    }
}

/// A whole row (a channel, `by_channel`) or column (a value) of the
/// intensity grid at `index`: its columns among `available`.
pub(crate) fn intensity_line(
    available: &[ColumnEntry],
    index: usize,
    by_channel: bool,
) -> Vec<Column> {
    column_panel::intensity_line(available, index, by_channel)
}

/// Shows `columns` (`on`) or hides them in `selected` - the others stay as
/// they are; new ones go to the end.
pub(crate) fn show(selected: &mut Vec<Column>, columns: &[Column], on: bool) {
    if on {
        for column in columns {
            if !selected.contains(column) {
                selected.push(column.clone());
            }
        }
    } else {
        selected.retain(|c| !columns.contains(c));
    }
}

/// Changes the coloc settings (`change`) - the partners sorted, once each.
pub(crate) fn change_settings(
    settings: &mut ColocSettings,
    change: impl FnOnce(&mut ColocSettings),
) {
    change(settings);
    settings.partners.sort();
    settings.partners.dedup();
}

/// `selected` after the coloc settings changed to `settings`: the coloc
/// columns counted the new way, those of classes no longer chosen gone.
pub(crate) fn adapt_columns(selected: &[Column], settings: &ColocSettings) -> Vec<Column> {
    let mut columns: Vec<Column> = Vec::with_capacity(selected.len());
    for column in selected.iter().filter_map(|c| adapt_column(c, settings)) {
        if !columns.contains(&column) {
            columns.push(column);
        }
    }
    columns
}

/// A partner class's key (as the picker gives it) toggled in `settings`.
pub(crate) fn toggle_partner(settings: &mut ColocSettings, key: &str) {
    let Ok(id) = key.parse::<u32>() else {
        return;
    };
    let class = ObjectClass::Global(id);
    change_settings(settings, |settings| {
        if settings.partners.contains(&class) {
            settings.partners.retain(|c| *c != class);
        } else {
            settings.partners.push(class);
        }
    });
}

/// The Children and Linked objects lists of a table (the Results List's,
/// an export table's): the links Relate Objects made, how the children's
/// values are summed up, the parent classes - those picked ticked.
pub(crate) struct RelatedItems {
    pub children: Vec<MultiSelectItem>,
    pub children_summary: String,
    pub aggregations: Vec<MultiSelectItem>,
    pub aggregate_summary: slint::SharedString,
    pub parents: Vec<MultiSelectItem>,
    pub parent_summary: String,
}

/// [`RelatedItems`] of the results' columns `available`, with `children`,
/// `aggregations` and `parents` picked.
pub(crate) fn related_items(
    available: &[ColumnEntry],
    classes: &[Class],
    children: &[(ObjectClass, ObjectClass)],
    aggregations: &[evanalyzer_app::results::Aggregation],
    parents: &[ObjectClass],
) -> RelatedItems {
    use super::results_state_controller::{
        AGGREGATIONS, aggregation_items, link_key, list_summary,
    };
    use evanalyzer_app::results::ChildValue;
    let class_label = |class: ObjectClass| {
        classes
            .iter()
            .find(|c| c.global() == class)
            .map(|c| c.name.clone())
            .unwrap_or_else(|| format!("class {}", class.to_u32().unwrap_or_default()))
    };
    let children_items: Vec<MultiSelectItem> = available
        .iter()
        .filter_map(|entry| match &entry.key {
            Column::Children(c) if c.value == ChildValue::Count => {
                let link = (c.parent_class, c.child_class);
                Some(MultiSelectItem {
                    key: link_key(link).into(),
                    value: format!(
                        "{} in {}",
                        class_label(c.child_class),
                        class_label(c.parent_class)
                    )
                    .into(),
                    color: slint::Color::default(),
                    group: "".into(),
                    selected: children.contains(&link),
                })
            }
            _ => None,
        })
        .collect();
    let parent_items: Vec<MultiSelectItem> = available
        .iter()
        .filter_map(|entry| match &entry.key {
            Column::Parent(p) => Some(MultiSelectItem {
                key: p.parent_class.to_u32()?.to_string().into(),
                value: format!("Parent · {}", class_label(p.parent_class)).into(),
                color: slint::Color::default(),
                group: "".into(),
                selected: parents.contains(&p.parent_class),
            }),
            _ => None,
        })
        .collect();
    RelatedItems {
        children_summary: format!("{} of {}", children.len(), children_items.len()),
        parent_summary: format!("{} of {}", parents.len(), parent_items.len()),
        children: children_items,
        parents: parent_items,
        aggregations: aggregation_items(aggregations),
        aggregate_summary: list_summary(aggregations.len(), AGGREGATIONS.len(), "Aggregations"),
    }
}
