//! # column_panel
//!
//! What the List's left panel adds to the column picker (design canvas
//! "Results rework", ColumnPanel E): the starting sets ("Start from"), the
//! intensity columns as a channels × values grid, and the coloc columns
//! by kind with marks of what each needs per partner class.
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-11-22
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.

use super::coloc::{
    COLOC_AT_LEAST_GROUP, COLOC_EXACTLY_GROUP, COLOC_GROUP, ColocSettings, coloc_classes,
    is_coloc_column, model_of, picker_label,
};
use super::*;
use crate::{ColocColumnItem, ColocKind, ColumnPreset, IntensityCell};
use evanalyzer_app::results::{ColocMatch, view_settings::DEFAULT_LIST_COLUMNS};
use slint::SharedString;

/// A channel's intensity column of one value (e.g. `Column::IntensityAvg`).
type IntensityOf = fn(u32) -> Column;

/// The intensity grid's values, in its column order: the short head and
/// the column of a channel.
const INTENSITY_VALUES: [(&str, IntensityOf); 9] = [
    ("Mean", Column::IntensityAvg),
    ("Sum", Column::IntensitySum),
    ("Min", Column::IntensityMin),
    ("Max", Column::IntensityMax),
    ("SD", Column::IntensityStd),
    ("Med", Column::IntensityMedian),
    ("Q1", Column::IntensityLowerQuartile),
    ("Q3", Column::IntensityUpperQuartile),
    ("MAD", Column::IntensityMad),
];

/// The channel an intensity column is about.
fn intensity_channel(column: &Column) -> Option<u32> {
    match column {
        Column::IntensityAvg(c)
        | Column::IntensitySum(c)
        | Column::IntensityMin(c)
        | Column::IntensityMax(c)
        | Column::IntensityStd(c)
        | Column::IntensityMedian(c)
        | Column::IntensityLowerQuartile(c)
        | Column::IntensityUpperQuartile(c)
        | Column::IntensityMad(c) => Some(*c),
        _ => None,
    }
}

/// The channels the results measured intensities in, in order.
fn intensity_channels(available: &[ColumnEntry]) -> Vec<u32> {
    let mut channels: Vec<u32> = available
        .iter()
        .filter_map(|entry| intensity_channel(&entry.key))
        .collect();
    channels.sort_unstable();
    channels.dedup();
    channels
}

/// The intensity grid: the channels' names and a cell per channel and
/// value (row by row) - a value the results lack has no key.
pub(super) fn intensity_grid(
    available: &[ColumnEntry],
    selected: &[Column],
    classes: &[Class],
) -> (Vec<String>, Vec<IntensityCell>) {
    let channels = intensity_channels(available);
    let names = channels.iter().map(|c| format!("Ch {c}")).collect();
    let mut cells = Vec::with_capacity(channels.len() * INTENSITY_VALUES.len());
    for channel in &channels {
        for (_, make) in INTENSITY_VALUES {
            let column = make(*channel);
            let offered = available.iter().any(|entry| entry.key == column);
            cells.push(IntensityCell {
                key: if offered {
                    column.as_key(classes).into()
                } else {
                    "".into()
                },
                selected: offered && selected.contains(&column),
            });
        }
    }
    (names, cells)
}

/// The intensity grid's heads.
pub(super) fn intensity_heads() -> Vec<SharedString> {
    INTENSITY_VALUES
        .iter()
        .map(|(head, _)| (*head).into())
        .collect()
}

/// A chart value's name: "Objects per image" for the count per image, the
/// column's (µm) else - "Area [µm²] · mean per image" per image.
pub(super) fn chart_value_label(column: &Column, per_image: bool, classes: &[Class]) -> String {
    let name = picker_label(&column.display_label(classes));
    match (per_image, column) {
        (true, Column::Count) => "Objects per image".to_string(),
        (true, _) => format!("{name} · mean per image"),
        _ => name,
    }
}

/// A one-choice picker's parts, made off the UI thread.
struct SinglePicker {
    sections: Vec<super::coloc::SectionUi>,
    cells: Vec<IntensityCell>,
    intensity_title: String,
    intensity_open: bool,
    kinds: Vec<ColocGroupUi>,
}

/// A starting set: its key, name, what it holds and its columns.
pub(super) struct Preset {
    pub key: &'static str,
    pub title: &'static str,
    pub description: &'static str,
    pub columns: Vec<Column>,
}

/// The starting sets the results offer - Coloc only with coloc partners.
/// Their columns among `available`, in the order they are listed.
pub(super) fn presets(available: &[ColumnEntry], raw: &[ColumnEntry]) -> Vec<Preset> {
    let offered = |columns: Vec<Column>| -> Vec<Column> {
        columns
            .into_iter()
            .filter(|c| available.iter().any(|entry| entry.key == *c))
            .collect()
    };
    let basic = offered(DEFAULT_LIST_COLUMNS.to_vec());
    let identity = offered(vec![
        Column::ImageName,
        Column::ObjectId,
        Column::ObjectClass,
    ]);
    let mut shape = basic.clone();
    for entry in available.iter().filter(|entry| entry.group == "Shape") {
        if !shape.contains(&entry.key) {
            shape.push(entry.key.clone());
        }
    }
    let mut intensity = identity.clone();
    intensity.extend(
        intensity_channels(available)
            .into_iter()
            .map(Column::IntensityAvg)
            .filter(|c| available.iter().any(|entry| entry.key == *c)),
    );
    let mut sets = vec![
        Preset {
            key: "basic",
            title: "Basic",
            description: "Image, class, area, circularity",
            columns: basic,
        },
        Preset {
            key: "shape",
            title: "Shape",
            description: "Area and every shape value",
            columns: shape,
        },
    ];
    if intensity.len() > identity.len() {
        sets.push(Preset {
            key: "intensity",
            title: "Intensity",
            description: "The mean of every channel",
            columns: intensity,
        });
    }
    if !coloc_classes(raw).is_empty() {
        // With each class some object overlaps - not those no combination
        // holds (a column of zeros).
        let occurring: Vec<ObjectClass> = super::coloc::coloc_sets(raw)
            .into_iter()
            .flatten()
            .collect();
        let mut coloc = identity;
        coloc.extend(
            available
                .iter()
                .filter(|entry| entry.group == COLOC_GROUP)
                .filter(|entry| match &entry.key {
                    Column::ColocCount(class) => occurring.contains(class),
                    Column::Coloc(c) => c.with.iter().all(|class| occurring.contains(class)),
                    _ => false,
                })
                .map(|entry| entry.key.clone()),
        );
        sets.push(Preset {
            key: "coloc",
            title: "Coloc",
            description: "Overlap with each class",
            columns: coloc,
        });
    }
    sets
}

/// The starting sets for the panel - the one whose columns are exactly the
/// shown ones marked.
pub(super) fn preset_cards(sets: &[Preset], selected: &[Column]) -> Vec<ColumnPreset> {
    sets.iter()
        .map(|set| ColumnPreset {
            key: set.key.into(),
            title: set.title.into(),
            description: set.description.into(),
            active: !set.columns.is_empty()
                && set.columns.len() == selected.len()
                && set.columns.iter().all(|c| selected.contains(c)),
        })
        .collect()
}

/// What a coloc column needs of each partner class (`partners`, in
/// order): ● overlaps it, ○ must not, · either.
pub(super) fn coloc_glyph(column: &Column, partners: &[ObjectClass]) -> String {
    let (how, with) = match column {
        Column::ColocCount(class) => (ColocMatch::AtLeast, vec![*class]),
        Column::Coloc(c) => (c.how, c.with.clone()),
        _ => return String::new(),
    };
    partners
        .iter()
        .map(|class| {
            if with.contains(class) {
                '●'
            } else if how == ColocMatch::Exactly {
                '○'
            } else {
                '·'
            }
        })
        .collect()
}

/// A kind of coloc columns, made off the UI thread (a model isn't `Send`).
pub(super) struct ColocKindUi {
    group: String,
    title: &'static str,
    description: &'static str,
    count: String,
    items: Vec<ColocColumnItem>,
}

impl ColocKindUi {
    pub(super) fn into_slint(self) -> ColocKind {
        ColocKind {
            group: self.group.into(),
            title: self.title.into(),
            description: self.description.into(),
            count: self.count.into(),
            items: ModelRc::from(Rc::new(VecModel::from(self.items))),
        }
    }
}

/// Between a kind's section group and its coloc block in a kind's `group`
/// key ("Coloc · at least\u{1}2") - what All / None of a block's kind tick.
pub(super) const BLOCK_SEPARATOR: char = '\u{1}';

/// The classes a coloc column is about (empty: "with no class").
pub(super) fn coloc_with(column: &Column) -> Option<Vec<ObjectClass>> {
    match column {
        Column::ColocCount(class) => Some(vec![*class]),
        Column::Coloc(c) => Some(c.with.clone()),
        _ => None,
    }
}

/// A coloc group's block of the builder, made off the UI thread.
pub(super) struct ColocGroupUi {
    key: String,
    name: String,
    classes: String,
    open: bool,
    kinds: Vec<ColocKindUi>,
}

impl ColocGroupUi {
    pub(super) fn into_slint(self) -> crate::ColocGroup {
        crate::ColocGroup {
            key: self.key.into(),
            name: self.name.into(),
            classes: self.classes.into(),
            open: self.open,
            kinds: ModelRc::from(Rc::new(VecModel::from(
                self.kinds
                    .into_iter()
                    .map(ColocKindUi::into_slint)
                    .collect::<Vec<_>>(),
            ))),
        }
    }
}

/// A group's name: what its classes share after their "@" ("in-cell-spots"),
/// else "Group `number`"; and the classes' own parts ("ch1 · ch2 · ch3").
fn block_names(block: &[ObjectClass], classes: &[Class], number: usize) -> (String, String) {
    let names: Vec<String> = block
        .iter()
        .map(|class| {
            classes
                .iter()
                .find(|c| c.global() == *class)
                .map(|c| c.name.clone())
                .unwrap_or_else(|| format!("class {}", class.to_u32().unwrap_or_default()))
        })
        .collect();
    let parts: Vec<Option<(&str, &str)>> = names.iter().map(|name| name.split_once('@')).collect();
    match parts.first() {
        Some(Some((_, shared)))
            if !shared.is_empty()
                && parts
                    .iter()
                    .all(|p| p.is_some_and(|(_, rest)| rest == *shared)) =>
        {
            let heads: Vec<&str> = parts
                .iter()
                .filter_map(|p| p.map(|(head, _)| head))
                .collect();
            (shared.to_string(), heads.join(" · "))
        }
        _ => (format!("Group {number}"), names.join(" · ")),
    }
}

/// The coloc columns offered as the builder shows them: a block per coloc
/// group (`groups`), in it by kind (each class, at least, exactly) the
/// columns of that group's classes with marks over them; "with no class"
/// in a block of its own. Those the search `filter` (lower case) matches;
/// empty kinds and blocks left out. With one group its block has no name.
pub(super) fn coloc_group_blocks(
    available: &[ColumnEntry],
    groups: &[Vec<ObjectClass>],
    selected: &[Column],
    settings: &ColocSettings,
    classes: &[Class],
    open: &HashMap<String, bool>,
    filter: &str,
) -> Vec<ColocGroupUi> {
    let kinds_of = |block: Option<(usize, &[ObjectClass])>| -> Vec<ColocKindUi> {
        // "With no class" with several groups: marks over every class would
        // say little - none.
        let marked: Vec<ObjectClass> = match block {
            Some((_, block)) => settings
                .partners
                .iter()
                .copied()
                .filter(|c| block.contains(c))
                .collect(),
            None if groups.len() > 1 => Vec::new(),
            None => settings.partners.clone(),
        };
        [
            (COLOC_GROUP, "EACH CLASS", "One column per partner class."),
            (
                COLOC_AT_LEAST_GROUP,
                "AT LEAST",
                "Overlaps all of these classes, maybe more.",
            ),
            (
                COLOC_EXACTLY_GROUP,
                "EXACTLY",
                "These classes and no other of the group - the columns add up to the count.",
            ),
        ]
        .into_iter()
        .filter_map(|(group, title, description)| {
            let items: Vec<ColocColumnItem> = available
                .iter()
                .filter(|entry| entry.group == group && is_coloc_column(&entry.key))
                .filter(|entry| {
                    let with = coloc_with(&entry.key).unwrap_or_default();
                    match block {
                        Some((_, block)) => {
                            !with.is_empty() && with.iter().all(|c| block.contains(c))
                        }
                        None => with.is_empty(),
                    }
                })
                .filter(|entry| {
                    filter.is_empty() || entry.display_name.to_lowercase().contains(filter)
                })
                .map(|entry| ColocColumnItem {
                    key: entry.key.as_key(classes).into(),
                    name: entry.display_name.as_str().into(),
                    glyph: coloc_glyph(&entry.key, &marked).into(),
                    selected: selected.contains(&entry.key),
                })
                .collect();
            if items.is_empty() {
                return None;
            }
            let chosen = items.iter().filter(|item| item.selected).count();
            let key = match block {
                Some((number, _)) => format!("{group}{BLOCK_SEPARATOR}{number}"),
                None => group.to_string(),
            };
            Some(ColocKindUi {
                group: key,
                title,
                description,
                count: format!("{chosen} of {}", items.len()),
                items,
            })
        })
        .collect()
    };
    let named = groups.len() > 1;
    let mut blocks: Vec<ColocGroupUi> = groups
        .iter()
        .enumerate()
        .filter_map(|(i, block)| {
            let kinds = kinds_of(Some((i + 1, block)));
            if kinds.is_empty() {
                return None;
            }
            let key = format!("coloc group {}", i + 1);
            let any_chosen = kinds
                .iter()
                .any(|k| k.items.iter().any(|item| item.selected));
            let (name, members) = block_names(block, classes, i + 1);
            Some(ColocGroupUi {
                open: !filter.is_empty() || open.get(&key).copied().unwrap_or(i == 0 || any_chosen),
                key,
                name: if named { name } else { String::new() },
                classes: if named { members } else { String::new() },
                kinds,
            })
        })
        .collect();
    let none = kinds_of(None);
    if !none.is_empty() {
        blocks.push(ColocGroupUi {
            key: "coloc none".into(),
            name: String::new(),
            classes: String::new(),
            open: true,
            kinds: none,
        });
    }
    blocks
}

impl ResultsStateController {
    /// Shows `columns` (`on`) or hides them - the others stay as they are;
    /// new ones go to the end.
    fn show_columns(self: &Arc<Self>, columns: &[Column], on: bool) {
        {
            let mut list = self.list_filter.lock().expect("Poisoned");
            if on {
                for column in columns {
                    if !list.columns.contains(column) {
                        list.columns.push(column.clone());
                    }
                }
            } else {
                list.columns.retain(|c| !columns.contains(c));
            }
        }
        self.push_list_columns();
        self.refresh_list();
    }

    /// A whole row (a channel, `by_channel`) or column (a value) of the
    /// intensity grid: all shown already - hidden, else shown.
    fn toggle_intensity_line(self: &Arc<Self>, index: usize, by_channel: bool) {
        let available = self.available_columns.lock().expect("Poisoned").clone();
        let channels = intensity_channels(&available);
        let line: Vec<Column> = if by_channel {
            let Some(channel) = channels.get(index) else {
                return;
            };
            INTENSITY_VALUES
                .iter()
                .map(|(_, make)| make(*channel))
                .collect()
        } else {
            let Some((_, make)) = INTENSITY_VALUES.get(index) else {
                return;
            };
            channels.iter().map(|channel| make(*channel)).collect()
        };
        let line: Vec<Column> = line
            .into_iter()
            .filter(|c| available.iter().any(|entry| entry.key == *c))
            .collect();
        let all_shown = {
            let list = self.list_filter.lock().expect("Poisoned");
            line.iter().all(|c| list.columns.contains(c))
        };
        self.show_columns(&line, !all_shown);
    }

    /// The columns of the starting set `key` shown instead of the others
    /// (Coloc first makes every partner class chosen if none is).
    fn apply_column_preset(self: &Arc<Self>, key: &str) {
        if key == "coloc"
            && self
                .coloc_settings
                .lock()
                .expect("Poisoned")
                .partners
                .is_empty()
        {
            let raw = self.raw_columns.lock().expect("Poisoned").clone();
            let partners = coloc_classes(&raw);
            {
                let mut settings = self.coloc_settings.lock().expect("Poisoned");
                settings.partners = partners;
                settings.partners.sort();
            }
            self.refresh_offered_columns();
        }
        let available = self.available_columns.lock().expect("Poisoned").clone();
        let raw = self.raw_columns.lock().expect("Poisoned").clone();
        let Some(set) = presets(&available, &raw)
            .into_iter()
            .find(|set| set.key == key)
        else {
            return;
        };
        self.list_filter.lock().expect("Poisoned").columns = set.columns;
        self.push_list_columns();
        self.refresh_list();
    }

    /// A one-choice picker's parts (Plate, Charts): the List's groups,
    /// intensity grid and coloc kinds over `available` with `selected`
    /// chosen, those the search `filter` matches.
    fn single_picker(
        &self,
        available: &[ColumnEntry],
        selected: &[Column],
        filter: &str,
    ) -> SinglePicker {
        let classes = self.classes.lock().expect("Poisoned").clone();
        let settings = self.coloc_settings.lock().expect("Poisoned").clone();
        let open = self.column_sections_open.lock().expect("Poisoned").clone();
        let filter = filter.to_lowercase();
        let (sections, _, _, _) =
            super::coloc::picker_sections(available, selected, &classes, &open, &filter, true);
        let (_, cells) = intensity_grid(available, selected, &classes);
        let intensity_section = sections.iter().find(|section| section.group == "intensity");
        SinglePicker {
            intensity_title: intensity_section
                .map(|section| section.title.clone())
                .unwrap_or_default(),
            intensity_open: intensity_section.is_some_and(|section| section.open),
            kinds: coloc_group_blocks(
                available,
                &super::coloc::coloc_groups(&self.raw_columns.lock().expect("Poisoned")),
                selected,
                &settings,
                &classes,
                &open,
                &filter,
            ),
            sections,
            cells,
        }
    }

    /// The Plate's value picker - the columns a Plate can show (not an
    /// object's identity), its one column chosen.
    pub(super) fn push_matrix_picker(&self) {
        let available: Vec<ColumnEntry> = self
            .available_columns
            .lock()
            .expect("Poisoned")
            .iter()
            .filter(|entry| is_matrix_column(&entry.key))
            .cloned()
            .collect();
        let selected: Vec<Column> = self
            .matrix_filter
            .lock()
            .expect("Poisoned")
            .as_ref()
            .map(|filter| vec![filter.column.clone()])
            .unwrap_or_default();
        let filter = self.matrix_column_filter.lock().expect("Poisoned").clone();
        let picker = self.single_picker(&available, &selected, &filter);
        let ui_weak = self.ui.clone();
        crate::helper::ui_thread::invoke_from_event_loop(move || {
            let Some(ui) = ui_weak.upgrade() else {
                return;
            };
            let state = ui.global::<ResultsState>();
            state.set_matrix_column_sections(model_of(
                picker
                    .sections
                    .into_iter()
                    .map(super::coloc::SectionUi::into_slint)
                    .collect(),
            ));
            state.set_matrix_intensity_cells(model_of(picker.cells));
            state.set_matrix_intensity_title(picker.intensity_title.into());
            state.set_matrix_intensity_open(picker.intensity_open);
            state.set_matrix_coloc_groups(model_of(
                picker
                    .kinds
                    .into_iter()
                    .map(ColocGroupUi::into_slint)
                    .collect(),
            ));
        })
        .ok();
    }

    /// The Charts' value picker - the columns a chart can draw - with the
    /// column of the slot being picked (X, or scatter's Y) chosen.
    pub(super) fn push_chart_picker(&self) {
        // A boxplot per image offers the objects' count too.
        let per_image = {
            let chart = self.chart_filter.lock().expect("Poisoned");
            chart.per_image && matches!(chart.kind, ChartKind::Boxplot | ChartKind::Bar)
        };
        let available: Vec<ColumnEntry> = self
            .available_columns
            .lock()
            .expect("Poisoned")
            .iter()
            .filter(|entry| {
                is_chartable_column(&entry.key) || (per_image && entry.key == Column::Count)
            })
            .cloned()
            .collect();
        let y = *self.chart_value_y.lock().expect("Poisoned");
        let selected = {
            let chart = self.chart_filter.lock().expect("Poisoned");
            vec![if y {
                chart.y_column.clone()
            } else {
                chart.column.clone()
            }]
        };
        let filter = self.chart_column_filter.lock().expect("Poisoned").clone();
        let picker = self.single_picker(&available, &selected, &filter);
        let ui_weak = self.ui.clone();
        crate::helper::ui_thread::invoke_from_event_loop(move || {
            let Some(ui) = ui_weak.upgrade() else {
                return;
            };
            let state = ui.global::<ResultsState>();
            state.set_chart_column_sections(model_of(
                picker
                    .sections
                    .into_iter()
                    .map(super::coloc::SectionUi::into_slint)
                    .collect(),
            ));
            state.set_chart_intensity_cells(model_of(picker.cells));
            state.set_chart_intensity_title(picker.intensity_title.into());
            state.set_chart_intensity_open(picker.intensity_open);
            state.set_chart_coloc_groups(model_of(
                picker
                    .kinds
                    .into_iter()
                    .map(ColocGroupUi::into_slint)
                    .collect(),
            ));
        })
        .ok();
    }

    /// The value's name on the bar: "Objects per image" for the count.
    pub(super) fn push_chart_value_summary(&self) {
        let label = {
            let classes = self.classes.lock().expect("Poisoned");
            let chart = self.chart_filter.lock().expect("Poisoned");
            chart_value_label(&chart.column, chart.per_image, &classes)
        };
        let ui_weak = self.ui.clone();
        crate::helper::ui_thread::invoke_from_event_loop(move || {
            if let Some(ui) = ui_weak.upgrade() {
                ui.global::<ResultsState>()
                    .set_chart_column_summary(label.into());
            }
        })
        .ok();
    }

    /// The chart draws column `key` on the slot being picked (X, or
    /// scatter's Y).
    fn pick_chart_column(self: &Arc<Self>, key: &str) {
        let classes = self.classes.lock().expect("Poisoned").clone();
        let Some(column) = self.column_for_key(key, &classes) else {
            warn!("Unknown chart column key picked: {key}");
            return;
        };
        let per_image = self.chart_filter.lock().expect("Poisoned").per_image;
        let label = chart_value_label(&column, per_image, &classes);
        let y = *self.chart_value_y.lock().expect("Poisoned");
        {
            let mut chart = self.chart_filter.lock().expect("Poisoned");
            if y {
                chart.y_column = column;
            } else {
                chart.column = column;
                chart.count_set_aside = false;
            }
        }
        self.push_chart_picker();
        let ui_weak = self.ui.clone();
        let key = key.to_string();
        crate::helper::ui_thread::invoke_from_event_loop(move || {
            let Some(ui) = ui_weak.upgrade() else {
                return;
            };
            let state = ui.global::<ResultsState>();
            let items = if y {
                state.set_chart_y_column_summary(label.into());
                state.get_chart_y_column_items()
            } else {
                state.set_chart_column_summary(label.into());
                state.get_chart_column_items()
            };
            for i in 0..items.row_count() {
                if let Some(mut item) = items.row_data(i) {
                    let on = item.key.as_str() == key;
                    if item.selected != on {
                        item.selected = on;
                        items.set_row_data(i, item);
                    }
                }
            }
        })
        .ok();
        self.refresh_charts();
    }

    /// The Plate shows column `key` (its picker's).
    fn pick_matrix_column(self: &Arc<Self>, key: &str) {
        let classes = self.classes.lock().expect("Poisoned").clone();
        let Some(column) = self.column_for_key(key, &classes) else {
            warn!("Unknown matrix column key picked: {key}");
            return;
        };
        let label = picker_label(&column.display_label(&classes));
        self.update_matrix_filter(|filter| filter.column = column);
        self.push_matrix_picker();
        let ui_weak = self.ui.clone();
        let manager = self.clone();
        let key = key.to_string();
        crate::helper::ui_thread::invoke_from_event_loop(move || {
            let Some(ui) = ui_weak.upgrade() else {
                return;
            };
            let state = ui.global::<ResultsState>();
            state.set_matrix_column_summary(label.into());
            // Split view's dropdown follows.
            let items = state.get_matrix_column_items();
            for i in 0..items.row_count() {
                if let Some(mut item) = items.row_data(i) {
                    let on = item.key.as_str() == key;
                    if item.selected != on {
                        item.selected = on;
                        items.set_row_data(i, item);
                    }
                }
            }
            let level = state.get_matrix_level();
            manager.apply_matrix_aggregate_enabled(&ui, level);
            manager.refresh_active_matrix_view();
            manager.split_value_column_changed();
        })
        .ok();
    }

    pub(super) fn attach_column_panel_callbacks(self: &Arc<Self>, ui: &AppWindow) {
        let state = ui.global::<ResultsState>();
        let manager = self.clone();
        state.on_matrix_value_picked(move |key| manager.pick_matrix_column(key.as_str()));
        let manager = self.clone();
        state.on_matrix_columns_filter_changed(move |text| {
            *manager.matrix_column_filter.lock().expect("Poisoned") = text.to_string();
            manager.push_matrix_picker();
        });
        let manager = self.clone();
        state.on_chart_value_picked(move |key| manager.pick_chart_column(key.as_str()));
        let manager = self.clone();
        state.on_chart_value_slot_changed(move |slot| {
            *manager.chart_value_y.lock().expect("Poisoned") = slot == 1;
            manager.push_chart_picker();
        });
        let manager = self.clone();
        state.on_chart_columns_filter_changed(move |text| {
            *manager.chart_column_filter.lock().expect("Poisoned") = text.to_string();
            manager.push_chart_picker();
        });
        let manager = self.clone();
        state.on_chart_columns_section_toggled(move |group, open| {
            manager
                .column_sections_open
                .lock()
                .expect("Poisoned")
                .insert(group.to_string(), open);
            manager.push_chart_picker();
        });
        let manager = self.clone();
        state.on_matrix_columns_section_toggled(move |group, open| {
            manager
                .column_sections_open
                .lock()
                .expect("Poisoned")
                .insert(group.to_string(), open);
            manager.push_matrix_picker();
        });
        let manager = self.clone();
        state.on_list_columns_preset(move |key| manager.apply_column_preset(key.as_str()));
        let manager = self.clone();
        state.on_list_intensity_row(move |index| {
            manager.toggle_intensity_line(usize::try_from(index).unwrap_or(usize::MAX), true);
        });
        let manager = self.clone();
        state.on_list_intensity_column(move |index| {
            manager.toggle_intensity_line(usize::try_from(index).unwrap_or(usize::MAX), false);
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entry(key: Column, group: &str) -> ColumnEntry {
        ColumnEntry {
            display_name: format!("{key:?}"),
            group: group.into(),
            key,
            coloc_group: 0,
        }
    }

    #[test]
    fn the_glyph_marks_what_a_coloc_column_needs_per_class() {
        let [a, b, c] = [2, 3, 4].map(ObjectClass::Global);
        let partners = [a, b, c];
        assert_eq!(coloc_glyph(&Column::ColocCount(b), &partners), "·●·");
        let at_least = Column::coloc(ColocMatch::AtLeast, vec![a, b], partners.to_vec(), false);
        assert_eq!(coloc_glyph(&at_least, &partners), "●●·");
        let exactly = Column::coloc(ColocMatch::Exactly, vec![a, c], partners.to_vec(), false);
        assert_eq!(coloc_glyph(&exactly, &partners), "●○●");
        let none = Column::coloc(ColocMatch::Exactly, vec![], partners.to_vec(), false);
        assert_eq!(coloc_glyph(&none, &partners), "○○○");
    }

    #[test]
    fn the_intensity_grid_has_a_cell_per_channel_and_value() {
        let available = vec![
            entry(Column::IntensityAvg(1), "intensity"),
            entry(Column::IntensitySum(1), "intensity"),
            entry(Column::IntensityAvg(0), "intensity"),
        ];
        let (names, cells) = intensity_grid(&available, &[Column::IntensityAvg(1)], &[]);
        assert_eq!(names, ["Ch 0", "Ch 1"]);
        assert_eq!(cells.len(), 2 * INTENSITY_VALUES.len());
        // Channel 0: only its mean is offered.
        assert!(!cells[0].key.is_empty() && !cells[0].selected);
        assert!(
            cells[1].key.is_empty(),
            "a value the results lack has no key"
        );
        // Channel 1: the mean shown, the sum offered.
        let row = INTENSITY_VALUES.len();
        assert!(cells[row].selected);
        assert!(!cells[row + 1].key.is_empty() && !cells[row + 1].selected);
    }

    #[test]
    fn a_group_is_named_by_what_its_classes_share_else_numbered() {
        let class = |id: u32, name: &str| Class {
            id,
            name: name.into(),
            ..Class::default()
        };
        let classes = vec![
            class(4, "ch1@in-cell-spots"),
            class(5, "ch2@in-cell-spots"),
            class(7, "spot1"),
            class(8, "spot2"),
        ];
        let g = ObjectClass::Global;
        assert_eq!(
            block_names(&[g(4), g(5)], &classes, 1),
            ("in-cell-spots".to_string(), "ch1 · ch2".to_string())
        );
        // No "@": numbered, the classes in full.
        assert_eq!(
            block_names(&[g(7), g(8)], &classes, 2),
            ("Group 2".to_string(), "spot1 · spot2".to_string())
        );
    }

    #[test]
    fn the_coloc_set_leaves_out_classes_no_combination_holds() {
        let [a, b] = [2, 3].map(ObjectClass::Global);
        let mut available = vec![
            entry(Column::ImageName, "General"),
            entry(Column::ObjectId, "General"),
            entry(Column::ObjectClass, "General"),
            entry(Column::ColocCount(a), COLOC_GROUP),
            entry(Column::ColocCount(b), COLOC_GROUP),
        ];
        // Objects overlap with class 2 only.
        available.push(entry(
            Column::coloc(ColocMatch::Exactly, vec![a], vec![a, b], false),
            COLOC_EXACTLY_GROUP,
        ));
        let sets = presets(&available, &available);
        let coloc = sets.iter().find(|s| s.key == "coloc").expect("coloc set");
        assert!(coloc.columns.contains(&Column::ColocCount(a)));
        assert!(!coloc.columns.contains(&Column::ColocCount(b)));
    }

    #[test]
    fn presets_hold_only_offered_columns_and_mark_the_shown_set() {
        let available = vec![
            entry(Column::ImageName, "General"),
            entry(Column::ObjectId, "General"),
            entry(Column::ObjectClass, "General"),
            entry(Column::AreaSizePx, "Geometry"),
            entry(Column::Circularity, "Shape"),
            entry(Column::Solidity, "Shape"),
        ];
        let sets = presets(&available, &available);
        let keys: Vec<&str> = sets.iter().map(|s| s.key).collect();
        assert_eq!(keys, ["basic", "shape"], "no intensity, no coloc offered");
        assert_eq!(
            sets[0].columns,
            [
                Column::ImageName,
                Column::ObjectId,
                Column::ObjectClass,
                Column::AreaSizePx,
                Column::Circularity
            ]
        );
        assert!(sets[1].columns.contains(&Column::Solidity));
        let mut shown = sets[0].columns.clone();
        shown.reverse();
        let cards = preset_cards(&sets, &shown);
        assert!(cards[0].active, "the same columns in another order");
        assert!(!cards[1].active);
    }
}
