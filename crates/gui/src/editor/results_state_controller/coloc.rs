//! # coloc
//!
//! The coloc columns the results offer: with each partner class, with at
//! least a combination, with exactly a combination, with no class
//! (`Column::Coloc`, `Column::ColocCount`). The results list the partner
//! classes and the combinations that occur (`get_available_columns`); the
//! combination columns are made here from them and the coloc settings of
//! the sidebar (which partner classes, objects or partners, only the
//! combinations that occur). Every picker - List, Export, Plate, Charts -
//! offers the same columns.
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-11-17
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.

use super::*;
use crate::ColumnSection;
use evanalyzer_app::results::{ColocColumn, ColocMatch};

/// The group of the coloc columns with each class.
pub(super) const COLOC_GROUP: &str = "Coloc";
/// The group of the "with at least" columns.
pub(super) const COLOC_AT_LEAST_GROUP: &str = "Coloc · at least";
/// The group of the "with exactly" columns (and "with no class").
pub(super) const COLOC_EXACTLY_GROUP: &str = "Coloc · exactly";

/// Without "only combinations that occur", every combination is offered up
/// to this many partner classes (2^n of them) - beyond, only those that
/// occur.
const MAX_CLASSES_FOR_ALL_COMBINATIONS: usize = 8;

/// The sidebar's coloc settings.
#[derive(Debug, Clone, PartialEq)]
pub(super) struct ColocSettings {
    /// The partner classes the combinations are made of (sorted).
    pub partners: Vec<ObjectClass>,
    /// Count the partner objects, else the objects (1 / 0).
    pub count_partners: bool,
    /// Only the combinations some object has.
    pub only_occurring: bool,
}

impl Default for ColocSettings {
    fn default() -> Self {
        Self {
            partners: Vec::new(),
            count_partners: false,
            only_occurring: true,
        }
    }
}

/// Whether `column` is a coloc column (not a coloc pairs row).
pub(super) fn is_coloc_column(column: &Column) -> bool {
    matches!(column, Column::ColocCount(_) | Column::Coloc(_))
}

/// The classes that are colocalization partners somewhere (the results'
/// `ColocCount` columns).
pub(super) fn coloc_classes(raw: &[ColumnEntry]) -> Vec<ObjectClass> {
    raw.iter()
        .filter_map(|entry| match entry.key {
            Column::ColocCount(class) => Some(class),
            _ => None,
        })
        .collect()
}

/// The sets of partner classes objects have (the results' "with exactly"
/// columns among all classes).
pub(super) fn coloc_sets(raw: &[ColumnEntry]) -> Vec<Vec<ObjectClass>> {
    raw.iter()
        .filter_map(|entry| match &entry.key {
            Column::Coloc(ColocColumn {
                how: ColocMatch::Exactly,
                with,
                ..
            }) => Some(with.clone()),
            _ => None,
        })
        .collect()
}

/// The settings a fresh analysis starts with: every partner class, objects
/// - or partners if `selected` (a view's columns) counts partners.
pub(super) fn default_settings(raw: &[ColumnEntry], selected: &[Column]) -> ColocSettings {
    ColocSettings {
        partners: coloc_classes(raw),
        count_partners: selected.iter().any(counts_partners),
        only_occurring: true,
    }
}

fn counts_partners(column: &Column) -> bool {
    match column {
        Column::ColocCount(_) => true,
        Column::Coloc(c) => c.partners,
        _ => false,
    }
}

/// Every subset of `classes` (sorted like `classes`), the smaller first.
fn subsets(classes: &[ObjectClass]) -> Vec<Vec<ObjectClass>> {
    let mut all: Vec<Vec<ObjectClass>> = (0u32..(1 << classes.len()))
        .map(|bits| {
            classes
                .iter()
                .enumerate()
                .filter(|(i, _)| bits & (1 << i) != 0)
                .map(|(_, class)| *class)
                .collect()
        })
        .collect();
    all.sort_by(|a, b| a.len().cmp(&b.len()).then_with(|| a.cmp(b)));
    all
}

/// The coloc columns `settings` offer, in their order: with each class, with
/// at least, with exactly, with no class.
pub(super) fn coloc_columns(raw: &[ColumnEntry], settings: &ColocSettings) -> Vec<Column> {
    let partners = &settings.partners;
    let count = settings.count_partners;
    // What objects have of the chosen classes.
    let mut occurring: Vec<Vec<ObjectClass>> = coloc_sets(raw)
        .into_iter()
        .map(|set| set.into_iter().filter(|c| partners.contains(c)).collect())
        .collect();
    occurring.sort();
    occurring.dedup();
    let only_occurring =
        settings.only_occurring || partners.len() > MAX_CLASSES_FOR_ALL_COMBINATIONS;
    let combinations: Vec<Vec<ObjectClass>> = if only_occurring {
        // Every part of a set that occurs occurs "at least".
        let mut found: Vec<Vec<ObjectClass>> =
            occurring.iter().flat_map(|set| subsets(set)).collect();
        found.sort_by(|a, b| a.len().cmp(&b.len()).then_with(|| a.cmp(b)));
        found.dedup();
        found
    } else {
        subsets(partners)
    };

    let mut columns: Vec<Column> = partners
        .iter()
        .map(|class| Column::coloc(ColocMatch::AtLeast, vec![*class], Vec::new(), count))
        .collect();
    columns.extend(
        combinations
            .iter()
            .filter(|set| set.len() >= 2)
            .map(|set| Column::coloc(ColocMatch::AtLeast, set.clone(), Vec::new(), count)),
    );
    let exactly =
        |set: &Vec<ObjectClass>| !set.is_empty() && (!only_occurring || occurring.contains(set));
    columns.extend(
        combinations
            .iter()
            .filter(|set| exactly(set))
            .map(|set| Column::coloc(ColocMatch::Exactly, set.clone(), partners.clone(), count)),
    );
    // "No class" counts objects only (none has partners there).
    if !partners.is_empty() && !count {
        columns.push(Column::coloc(
            ColocMatch::Exactly,
            Vec::new(),
            partners.clone(),
            false,
        ));
    }
    columns
}

/// The group a coloc column is listed in.
pub(super) fn coloc_group(column: &Column) -> &'static str {
    match column {
        Column::Coloc(ColocColumn {
            how: ColocMatch::AtLeast,
            with,
            ..
        }) if with.len() >= 2 => COLOC_AT_LEAST_GROUP,
        Column::Coloc(ColocColumn {
            how: ColocMatch::Exactly,
            ..
        }) => COLOC_EXACTLY_GROUP,
        _ => COLOC_GROUP,
    }
}

/// The columns the pickers offer: the results' own (`raw`), their coloc
/// columns made by `settings` - and the chosen coloc columns `selected`
/// (e.g. a view's) that `settings` don't make, so they can be unticked.
pub(super) fn offered_columns(
    raw: &[ColumnEntry],
    settings: &ColocSettings,
    selected: &[Column],
    classes: &[Class],
) -> Vec<ColumnEntry> {
    let mut coloc = coloc_columns(raw, settings);
    for column in selected {
        if is_coloc_column(column) && !coloc.contains(column) {
            coloc.push(column.clone());
        }
    }
    let entry = |key: Column| ColumnEntry {
        display_name: key.display_label(classes),
        group: coloc_group(&key).to_string(),
        key,
    };
    let mut offered = Vec::with_capacity(raw.len() + coloc.len());
    let mut coloc = Some(coloc);
    for item in raw {
        if is_coloc_column(&item.key) {
            // In place of the results' coloc columns.
            if let Some(coloc) = coloc.take() {
                offered.extend(coloc.into_iter().map(entry));
            }
        } else {
            offered.push(item.clone());
        }
    }
    if let Some(coloc) = coloc {
        offered.extend(coloc.into_iter().map(entry));
    }
    offered
}

/// A chosen coloc column after the settings changed to `settings`: counted
/// the new way, among the new classes - `None` where it names a class no
/// longer chosen (or is "no class" while counting partners).
pub(super) fn adapt_column(column: &Column, settings: &ColocSettings) -> Option<Column> {
    let (how, with) = match column {
        Column::ColocCount(class) => (ColocMatch::AtLeast, vec![*class]),
        Column::Coloc(c) => (c.how, c.with.clone()),
        other => return Some(other.clone()),
    };
    if with.iter().any(|class| !settings.partners.contains(class)) {
        return None;
    }
    if with.is_empty() && settings.count_partners {
        return None;
    }
    Some(Column::coloc(
        how,
        with,
        settings.partners.clone(),
        settings.count_partners,
    ))
}

/// A column's name in the pickers: lengths, areas and volumes in µm, µm²
/// and µm³ - the unit the table shows them in (`in_micrometers`).
pub(super) fn picker_label(name: &str) -> String {
    name.replace("[nm²]", "[µm²]")
        .replace("[nm³]", "[µm³]")
        .replace("[nm]", "[µm]")
}

/// A model of `items` for a Slint property.
pub(super) fn model_of<T: Clone + 'static>(items: Vec<T>) -> ModelRc<T> {
    ModelRc::from(Rc::new(VecModel::from(items)))
}

/// The coloc box's own entry in `column_sections_open`.
const COLOC_BOX: &str = "coloc box";

/// A section of the sidebar's column picker, made off the UI thread.
pub(super) struct SectionUi {
    pub group: String,
    pub title: String,
    pub open: bool,
    pub items: Vec<MultiSelectItem>,
}

impl SectionUi {
    pub(super) fn into_slint(self) -> ColumnSection {
        ColumnSection {
            group: self.group.into(),
            title: self.title.into(),
            open: self.open,
            items: ModelRc::from(Rc::new(VecModel::from(self.items))),
        }
    }
}

/// The column picker's groups, in the columns' order - the coloc columns
/// apart (each class, at least, exactly), the children's and the parents'
/// columns left out (their own lists) - those the search `filter` (lower
/// case) matches; the coloc box's title and whether it is open. `single`:
/// one column is chosen (Plate, Charts) - titles without counts.
pub(super) fn picker_sections(
    available: &[ColumnEntry],
    selected: &[Column],
    classes: &[Class],
    open: &HashMap<String, bool>,
    filter: &str,
    single: bool,
) -> (Vec<SectionUi>, Vec<SectionUi>, String, bool) {
    let mut sections: Vec<SectionUi> = Vec::new();
    let mut coloc: Vec<SectionUi> = [COLOC_GROUP, COLOC_AT_LEAST_GROUP, COLOC_EXACTLY_GROUP]
        .into_iter()
        .map(|group| SectionUi {
            group: group.to_string(),
            title: String::new(),
            open: true,
            items: Vec::new(),
        })
        .collect();
    for entry in available {
        if matches!(entry.key, Column::Children(_) | Column::Parent(_)) {
            continue;
        }
        // nm as µm, as the table shows them.
        let label = picker_label(&entry.display_name);
        if !filter.is_empty() && !label.to_lowercase().contains(filter) {
            continue;
        }
        let item = MultiSelectItem {
            key: entry.key.as_key(classes).into(),
            value: label.into(),
            color: Color::default(),
            group: entry.group.as_str().into(),
            selected: selected.contains(&entry.key),
        };
        let target = if is_coloc_column(&entry.key) {
            coloc.iter_mut().find(|section| section.group == entry.group)
        } else {
            sections.iter_mut().find(|section| section.group == entry.group)
        };
        match target {
            Some(section) => section.items.push(item),
            None => sections.push(SectionUi {
                group: entry.group.clone(),
                title: String::new(),
                open: true,
                items: vec![item],
            }),
        }
    }
    // Open while searching, else as folded by hand - or while one of its
    // columns is chosen.
    let is_open = |section: &SectionUi| {
        !filter.is_empty()
            || open
                .get(&section.group)
                .copied()
                .unwrap_or_else(|| section.items.iter().any(|item| item.selected))
    };
    let title = |name: &str, items: &[MultiSelectItem]| {
        if single {
            name.to_uppercase()
        } else {
            section_title(name, items)
        }
    };
    for section in &mut sections {
        section.title = title(&section.group, &section.items);
        section.open = is_open(section);
    }
    let coloc_items: Vec<MultiSelectItem> =
        coloc.iter().flat_map(|s| s.items.iter().cloned()).collect();
    let coloc_title = title("Coloc", &coloc_items);
    let coloc_open = !filter.is_empty() || open.get(COLOC_BOX).copied().unwrap_or(true);
    for (section, name) in coloc
        .iter_mut()
        .zip(["With each class", "With at least", "With exactly"])
    {
        section.title = title(name, &section.items);
        section.open = !filter.is_empty() || open.get(&section.group).copied().unwrap_or(true);
    }
    coloc.retain(|section| !section.items.is_empty());
    (sections, coloc, coloc_title, coloc_open)
}

/// "SHAPE · 1 of 12".
fn section_title(name: &str, items: &[MultiSelectItem]) -> String {
    let chosen = items.iter().filter(|item| item.selected).count();
    format!("{} · {chosen} of {}", name.to_uppercase(), items.len())
}

impl ResultsStateController {
    /// The results' own columns (a database opened): kept, the coloc
    /// settings from the start, and the columns the pickers offer made of
    /// them - returned.
    pub(super) fn set_raw_columns(&self, raw: Vec<ColumnEntry>) -> Vec<ColumnEntry> {
        let selected = self.list_filter.lock().expect("Poisoned").columns.clone();
        *self.coloc_settings.lock().expect("Poisoned") = default_settings(&raw, &selected);
        *self.raw_columns.lock().expect("Poisoned") = raw;
        self.refresh_offered_columns();
        self.available_columns.lock().expect("Poisoned").clone()
    }

    /// `available_columns` anew from the results' columns and the coloc
    /// settings - with the coloc columns List, Plate and Charts show.
    pub(super) fn refresh_offered_columns(&self) {
        let raw = self.raw_columns.lock().expect("Poisoned").clone();
        if raw.is_empty() {
            return;
        }
        let classes = self.classes.lock().expect("Poisoned").clone();
        let settings = self.coloc_settings.lock().expect("Poisoned").clone();
        let mut selected = self.list_filter.lock().expect("Poisoned").columns.clone();
        if let Some(filter) = self.matrix_filter.lock().expect("Poisoned").as_ref() {
            selected.push(filter.column.clone());
        }
        {
            let chart = self.chart_filter.lock().expect("Poisoned");
            selected.push(chart.column.clone());
            selected.push(chart.y_column.clone());
        }
        *self.available_columns.lock().expect("Poisoned") =
            offered_columns(&raw, &settings, &selected, &classes);
    }

    /// The coloc settings as the chosen columns (a view's) count: partners
    /// if they do - then the columns offered again.
    pub(super) fn sync_coloc_with_columns(&self) {
        let selected = self.list_filter.lock().expect("Poisoned").columns.clone();
        if let Some(column) = selected.iter().find(|c| is_coloc_column(c)) {
            self.coloc_settings.lock().expect("Poisoned").count_partners = counts_partners(column);
        }
        self.refresh_offered_columns();
    }

    /// The sidebar's columns: the flat list (the summary), the sections
    /// and the coloc box, as `list_filter` chooses.
    pub(super) fn push_list_columns(&self) {
        let available = self.available_columns.lock().expect("Poisoned").clone();
        let raw = self.raw_columns.lock().expect("Poisoned").clone();
        let classes = self.classes.lock().expect("Poisoned").clone();
        let selected = self.list_filter.lock().expect("Poisoned").columns.clone();
        let settings = self.coloc_settings.lock().expect("Poisoned").clone();
        let open = self.column_sections_open.lock().expect("Poisoned").clone();
        let filter = self.column_filter.lock().expect("Poisoned").to_lowercase();

        let items = column_items(&available, &classes, |key| selected.contains(key));
        let summary = chip_summary(selected.len(), items.len());
        let groups = column_groups(&available);

        let (sections, coloc, coloc_title, coloc_open) =
            picker_sections(&available, &selected, &classes, &open, &filter, false);

        let partner_items: Vec<MultiSelectItem> = coloc_classes(&raw)
            .into_iter()
            .map(|class| {
                let known = classes.iter().find(|c| c.global() == class);
                MultiSelectItem {
                    key: class.to_u32().unwrap_or_default().to_string().into(),
                    value: known
                        .map(|c| c.name.clone())
                        .unwrap_or_else(|| format!("class {}", class.to_u32().unwrap_or_default()))
                        .into(),
                    color: known.map_or(Color::default(), |c| bg_color_to_slint(c.color)),
                    group: "".into(),
                    selected: settings.partners.contains(&class),
                }
            })
            .collect();
        let partner_summary = match settings.partners.len() {
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
        let count = i32::from(settings.count_partners);
        let only_occurring = settings.only_occurring;

        // The panel's parts: the starting sets, the intensity grid, the coloc
        // columns by kind.
        let presets =
            super::column_panel::preset_cards(&super::column_panel::presets(&available, &raw), &selected);
        let (intensity_channels, intensity_cells) =
            super::column_panel::intensity_grid(&available, &selected, &classes);
        let intensity_section = sections.iter().find(|section| section.group == "intensity");
        let intensity_title = intensity_section
            .map(|section| section.title.clone())
            .unwrap_or_default();
        let intensity_open = intensity_section.is_some_and(|section| section.open);
        let coloc_kinds =
            super::column_panel::coloc_kinds(&available, &selected, &settings, &classes, &filter);

        let ui_weak = self.ui.clone();
        crate::helper::ui_thread::invoke_from_event_loop(move || {
            let Some(ui) = ui_weak.upgrade() else {
                return;
            };
            let state = ui.global::<ResultsState>();
            let model = |items: Vec<MultiSelectItem>| ModelRc::from(Rc::new(VecModel::from(items)));
            state.set_list_columns(model(items));
            state.set_list_columns_summary(summary);
            state.set_list_columns_groups(ModelRc::from(Rc::new(VecModel::from(groups))));
            let sections: Vec<ColumnSection> =
                sections.into_iter().map(SectionUi::into_slint).collect();
            state.set_list_column_sections(ModelRc::from(Rc::new(VecModel::from(sections))));
            let coloc: Vec<ColumnSection> = coloc.into_iter().map(SectionUi::into_slint).collect();
            state.set_list_coloc_sections(ModelRc::from(Rc::new(VecModel::from(coloc))));
            state.set_list_coloc_title(coloc_title.into());
            state.set_list_coloc_open(coloc_open);
            state.set_list_coloc_partner_items(model(partner_items));
            state.set_list_coloc_partner_summary(partner_summary.into());
            state.set_list_coloc_partner_order(partner_order.into());
            state.set_list_coloc_count(count);
            state.set_list_coloc_only_occurring(only_occurring);
            state.set_list_column_presets(model_of(presets));
            state.set_list_intensity_channels(model_of(
                intensity_channels.into_iter().map(slint::SharedString::from).collect(),
            ));
            state.set_list_intensity_stats(model_of(super::column_panel::intensity_heads()));
            state.set_list_intensity_cells(model_of(intensity_cells));
            state.set_list_intensity_title(intensity_title.into());
            state.set_list_intensity_open(intensity_open);
            state.set_list_coloc_kinds(model_of(
                coloc_kinds.into_iter().map(super::column_panel::ColocKindUi::into_slint).collect(),
            ));
        })
        .ok();
        // The Plate's and the Charts' pickers offer the same columns.
        self.push_matrix_picker();
        self.push_chart_picker();
    }

    /// The coloc settings changed (`change`): the chosen coloc columns of
    /// List, Plate and Charts follow, the pickers and views anew.
    fn change_coloc_settings(self: &Arc<Self>, change: impl FnOnce(&mut ColocSettings)) {
        let settings = {
            let mut settings = self.coloc_settings.lock().expect("Poisoned");
            change(&mut settings);
            settings.partners.sort();
            settings.partners.dedup();
            settings.clone()
        };
        {
            let mut list = self.list_filter.lock().expect("Poisoned");
            let mut columns: Vec<Column> = Vec::with_capacity(list.columns.len());
            for column in list
                .columns
                .iter()
                .filter_map(|c| adapt_column(c, &settings))
            {
                if !columns.contains(&column) {
                    columns.push(column);
                }
            }
            list.columns = columns;
        }
        let matrix_changed = {
            let mut matrix = self.matrix_filter.lock().expect("Poisoned");
            match matrix.as_mut() {
                Some(filter) if is_coloc_column(&filter.column) => {
                    if let Some(column) = adapt_column(&filter.column, &settings) {
                        filter.column = column;
                    }
                    true
                }
                _ => false,
            }
        };
        let chart_changed = {
            let mut chart = self.chart_filter.lock().expect("Poisoned");
            let chart = &mut *chart;
            let mut changed = false;
            for column in [&mut chart.column, &mut chart.y_column] {
                if is_coloc_column(column) {
                    if let Some(adapted) = adapt_column(column, &settings) {
                        *column = adapted;
                    }
                    changed = true;
                }
            }
            changed
        };
        self.refresh_offered_columns();
        self.push_view_controls_in_slint();
        self.refresh_list();
        if matrix_changed {
            self.update_matrix_view();
        }
        if chart_changed {
            self.refresh_charts();
        }
    }

    /// Chooses (`selected`) or leaves out every column of a section
    /// (`group`) - those the search shows.
    fn select_column_section(self: &Arc<Self>, group: &str, selected: bool) {
        let filter = self.column_filter.lock().expect("Poisoned").to_lowercase();
        let in_section: Vec<Column> = self
            .available_columns
            .lock()
            .expect("Poisoned")
            .iter()
            .filter(|entry| entry.group == group)
            .filter(|entry| {
                filter.is_empty() || entry.display_name.to_lowercase().contains(&filter)
            })
            .map(|entry| entry.key.clone())
            .collect();
        {
            let mut list = self.list_filter.lock().expect("Poisoned");
            if selected {
                for column in in_section {
                    if !list.columns.contains(&column) {
                        list.columns.push(column);
                    }
                }
            } else {
                list.columns.retain(|column| !in_section.contains(column));
            }
        }
        self.push_list_columns();
        self.refresh_list();
    }

    pub(super) fn attach_coloc_callbacks(self: &Arc<Self>, ui: &AppWindow) {
        let state = ui.global::<ResultsState>();

        let manager = self.clone();
        state.on_list_columns_section_select(move |group, selected| {
            manager.select_column_section(group.as_str(), selected);
        });
        let manager = self.clone();
        state.on_list_columns_section_toggled(move |group, open| {
            manager
                .column_sections_open
                .lock()
                .expect("Poisoned")
                .insert(group.to_string(), open);
            manager.push_list_columns();
        });
        let manager = self.clone();
        state.on_list_columns_filter_changed(move |text| {
            *manager.column_filter.lock().expect("Poisoned") = text.to_string();
            manager.push_list_columns();
        });
        let manager = self.clone();
        state.on_list_coloc_toggled(move |open| {
            manager
                .column_sections_open
                .lock()
                .expect("Poisoned")
                .insert(COLOC_BOX.to_string(), open);
        });
        let manager = self.clone();
        state.on_list_coloc_partner_toggled(move |key| {
            let Ok(id) = key.parse::<u32>() else {
                return;
            };
            let class = ObjectClass::Global(id);
            manager.change_coloc_settings(|settings| {
                if settings.partners.contains(&class) {
                    settings.partners.retain(|c| *c != class);
                } else {
                    settings.partners.push(class);
                }
            });
        });
        let manager = self.clone();
        state.on_list_coloc_partners_all(move |on| {
            let all = coloc_classes(&manager.raw_columns.lock().expect("Poisoned"));
            manager.change_coloc_settings(|settings| {
                settings.partners = if on { all } else { Vec::new() };
            });
        });
        let manager = self.clone();
        state.on_list_coloc_count_picked(move |index| {
            manager.change_coloc_settings(|settings| settings.count_partners = index == 1);
        });
        let manager = self.clone();
        state.on_list_coloc_only_occurring_changed(move |on| {
            manager.change_coloc_settings(|settings| settings.only_occurring = on);
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn class(n: u32) -> ObjectClass {
        ObjectClass::Global(n)
    }

    /// The results of objects with partners of {2}, {2, 3} and {3, 4}.
    fn raw() -> Vec<ColumnEntry> {
        let entry = |key: Column| ColumnEntry {
            display_name: String::new(),
            group: "Coloc".into(),
            key,
        };
        let all = vec![class(2), class(3), class(4)];
        let mut raw = vec![entry(Column::AreaSizePx)];
        raw.extend(all.iter().map(|c| entry(Column::ColocCount(*c))));
        for set in [
            vec![class(2)],
            vec![class(2), class(3)],
            vec![class(3), class(4)],
        ] {
            raw.push(entry(Column::coloc(
                ColocMatch::Exactly,
                set,
                all.clone(),
                false,
            )));
        }
        raw.push(entry(Column::Circularity));
        raw
    }

    fn at_least(with: &[u32], partners: bool) -> Column {
        Column::coloc(
            ColocMatch::AtLeast,
            with.iter().map(|n| class(*n)).collect(),
            Vec::new(),
            partners,
        )
    }

    fn exactly(with: &[u32], among: &[u32]) -> Column {
        Column::coloc(
            ColocMatch::Exactly,
            with.iter().map(|n| class(*n)).collect(),
            among.iter().map(|n| class(*n)).collect(),
            false,
        )
    }

    #[test]
    fn only_the_combinations_that_occur() {
        let settings = default_settings(&raw(), &[]);
        assert_eq!(
            coloc_columns(&raw(), &settings),
            vec![
                at_least(&[2], false),
                at_least(&[3], false),
                at_least(&[4], false),
                at_least(&[2, 3], false),
                at_least(&[3, 4], false),
                exactly(&[2], &[2, 3, 4]),
                exactly(&[2, 3], &[2, 3, 4]),
                exactly(&[3, 4], &[2, 3, 4]),
                exactly(&[], &[2, 3, 4]),
            ]
        );
    }

    #[test]
    fn every_combination_when_asked() {
        let settings = ColocSettings {
            only_occurring: false,
            ..default_settings(&raw(), &[])
        };
        let columns = coloc_columns(&raw(), &settings);
        // 3 single, 4 at least (3 pairs, 1 triple), 7 exactly, no class.
        assert_eq!(columns.len(), 3 + 4 + 7 + 1);
        assert!(columns.contains(&at_least(&[2, 3, 4], false)));
        assert!(columns.contains(&exactly(&[2, 4], &[2, 3, 4])));
    }

    #[test]
    fn fewer_partner_classes_make_the_sets_smaller() {
        // Without class 4, {3, 4} is "exactly 3" among 2, 3.
        let settings = ColocSettings {
            partners: vec![class(2), class(3)],
            ..default_settings(&raw(), &[])
        };
        let columns = coloc_columns(&raw(), &settings);
        assert!(columns.contains(&exactly(&[3], &[2, 3])));
        assert!(
            !columns
                .iter()
                .any(|c| matches!(c, Column::Coloc(cc) if cc.with.contains(&class(4))))
        );
    }

    #[test]
    fn partners_are_the_coloc_count_for_a_single_class() {
        let settings = ColocSettings {
            count_partners: true,
            ..default_settings(&raw(), &[])
        };
        let columns = coloc_columns(&raw(), &settings);
        assert_eq!(columns[0], Column::ColocCount(class(2)));
        // "No class" has no partners to count.
        assert!(!columns.contains(&exactly(&[], &[2, 3, 4])));
        assert!(default_settings(&raw(), &[Column::ColocCount(class(2))]).count_partners);
    }

    #[test]
    fn offered_replaces_the_results_coloc_columns_in_place() {
        let raw = raw();
        let settings = default_settings(&raw, &[]);
        let offered = offered_columns(&raw, &settings, &[], &[]);
        assert_eq!(
            offered.first().map(|e| e.key.clone()),
            Some(Column::AreaSizePx)
        );
        assert_eq!(
            offered.last().map(|e| e.key.clone()),
            Some(Column::Circularity)
        );
        assert_eq!(offered.len(), 2 + 9);
        assert_eq!(offered[4].group, COLOC_AT_LEAST_GROUP);
        assert_eq!(offered[6].group, COLOC_EXACTLY_GROUP);
        // A chosen one the settings don't make stays offered.
        let chosen = at_least(&[2, 4], false);
        let offered = offered_columns(&raw, &settings, std::slice::from_ref(&chosen), &[]);
        assert!(offered.iter().any(|e| e.key == chosen));
    }

    #[test]
    fn chosen_columns_follow_the_settings() {
        let settings = ColocSettings {
            partners: vec![class(2), class(3)],
            count_partners: true,
            only_occurring: true,
        };
        assert_eq!(
            adapt_column(&at_least(&[2], false), &settings),
            Some(Column::ColocCount(class(2)))
        );
        assert_eq!(adapt_column(&at_least(&[2, 4], false), &settings), None);
        assert_eq!(adapt_column(&exactly(&[], &[2, 3, 4]), &settings), None);
        assert_eq!(
            adapt_column(&exactly(&[2], &[2, 3, 4]), &settings),
            Some(Column::coloc(
                ColocMatch::Exactly,
                vec![class(2)],
                vec![class(2), class(3)],
                true
            ))
        );
        assert_eq!(
            adapt_column(&Column::AreaSizePx, &settings),
            Some(Column::AreaSizePx)
        );
    }

    #[test]
    fn labels_name_the_combination() {
        let classes: Vec<Class> = Vec::new();
        assert_eq!(
            at_least(&[2], false).display_label(&classes),
            "Coloc with class 2"
        );
        assert_eq!(
            at_least(&[2, 3], false).display_label(&classes),
            "Coloc at least class 2, class 3"
        );
        assert_eq!(
            exactly(&[2, 3], &[2, 3, 4]).display_label(&classes),
            "Coloc exactly class 2, class 3"
        );
        assert_eq!(
            exactly(&[], &[2, 3]).display_label(&classes),
            "Coloc with no class"
        );
        assert_eq!(
            Column::ColocCount(class(2)).display_label(&classes),
            "Coloc with class 2"
        );
        // The key comes back as the column.
        let column = exactly(&[2, 3], &[2, 3, 4]);
        assert_eq!(
            Column::from_key(&column.as_key(&classes), &classes),
            Some(column)
        );
    }
}
