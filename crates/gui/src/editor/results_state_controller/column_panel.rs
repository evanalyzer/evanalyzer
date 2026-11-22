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
    is_coloc_column,
};
use super::*;
use crate::{ColocColumnItem, ColocKind, ColumnPreset, IntensityCell};
use slint::SharedString;
use evanalyzer_app::results::{ColocMatch, view_settings::DEFAULT_LIST_COLUMNS};

/// The intensity grid's values, in its column order: the short head and
/// the column of a channel.
const INTENSITY_VALUES: [(&str, fn(u32) -> Column); 9] = [
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
    INTENSITY_VALUES.iter().map(|(head, _)| (*head).into()).collect()
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
        let mut coloc = identity;
        coloc.extend(
            available
                .iter()
                .filter(|entry| entry.group == COLOC_GROUP)
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
    group: &'static str,
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

/// The coloc columns offered, by kind (each class, at least, exactly) -
/// those the search `filter` (lower case) matches; kinds without columns
/// left out.
pub(super) fn coloc_kinds(
    available: &[ColumnEntry],
    selected: &[Column],
    settings: &ColocSettings,
    classes: &[Class],
    filter: &str,
) -> Vec<ColocKindUi> {
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
            "These classes and no other - the columns add up to the count.",
        ),
    ]
    .into_iter()
    .filter_map(|(group, title, description)| {
        let items: Vec<ColocColumnItem> = available
            .iter()
            .filter(|entry| entry.group == group && is_coloc_column(&entry.key))
            .filter(|entry| filter.is_empty() || entry.display_name.to_lowercase().contains(filter))
            .map(|entry| ColocColumnItem {
                key: entry.key.as_key(classes).into(),
                name: entry.display_name.as_str().into(),
                glyph: coloc_glyph(&entry.key, &settings.partners).into(),
                selected: selected.contains(&entry.key),
            })
            .collect();
        if items.is_empty() {
            return None;
        }
        let chosen = items.iter().filter(|item| item.selected).count();
        Some(ColocKindUi {
            group,
            title,
            description,
            count: format!("{chosen} of {}", items.len()),
            items,
        })
    })
    .collect()
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
            INTENSITY_VALUES.iter().map(|(_, make)| make(*channel)).collect()
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
        if key == "coloc" && self.coloc_settings.lock().expect("Poisoned").partners.is_empty() {
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
        let Some(set) = presets(&available, &raw).into_iter().find(|set| set.key == key) else {
            return;
        };
        self.list_filter.lock().expect("Poisoned").columns = set.columns;
        self.push_list_columns();
        self.refresh_list();
    }

    pub(super) fn attach_column_panel_callbacks(self: &Arc<Self>, ui: &AppWindow) {
        let state = ui.global::<ResultsState>();
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
        assert!(cells[1].key.is_empty(), "a value the results lack has no key");
        // Channel 1: the mean shown, the sum offered.
        let row = INTENSITY_VALUES.len();
        assert!(cells[row].selected);
        assert!(!cells[row + 1].key.is_empty() && !cells[row + 1].selected);
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
