//! # present
//!
//! How a value of the results is presented, decided once: a column's name
//! in a file (table format 1), its label, its unit, and the factor its
//! stored value is multiplied by to be in that unit. The Results views, the
//! exports and the command line all present a column with this - never on
//! their own.
//!
//! Lengths are stored in nm; they are shown in µm (the Results views,
//! exports by default) or in nm (an export set to nm).
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-11-25
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.

use super::{ChildValue, ColocMatch, Column, ParentValue};
use evanalyzer_cfg::core_types::ObjectClass;
use evanalyzer_cfg::settings::classification_settings::Class;
pub use evanalyzer_cfg::settings::export_settings::LengthUnit;

/// The folder of an image directly in the image root (no folder of its
/// own) - its folder row, time course line and split part.
pub const ROOT_FOLDER: &str = "(root)";

/// A column as it is presented.
#[derive(Clone, Debug, PartialEq)]
pub struct Presented {
    /// Its name in a file: lower case, `a-z`, `0-9` and `_` only, the
    /// length unit in it (`area_um2`).
    pub name: String,
    /// As the app shows it ("Area [nm²]" - see [`Column::display_label`]).
    pub label: String,
    /// The unit of its values as presented ("µm²", "px", "gray value",
    /// "" none).
    pub unit: String,
    /// What a stored value is multiplied by to be in `unit`.
    pub factor: f64,
}

/// How `column` is presented with lengths in `unit`.
pub fn present(column: &Column, unit: LengthUnit, classes: &[Class]) -> Presented {
    let length = length_of(column);
    let (from, to, factor) = match (length, unit) {
        (None, _) | (Some(_), LengthUnit::Nm) => ("", "", 1.0),
        (Some(Length::Line), LengthUnit::Um) => ("_nm", "_um", 1e-3),
        (Some(Length::Area), LengthUnit::Um) => ("_nm2", "_um2", 1e-6),
        (Some(Length::Volume), LengthUnit::Um) => ("_nm3", "_um3", 1e-9),
    };
    let mut name = file_name(column, classes);
    if !from.is_empty()
        && let Some(at) = name.rfind(from)
    {
        name.replace_range(at..at + from.len(), to);
    }
    Presented {
        name,
        label: column.display_label(classes),
        unit: unit_of(column, length, unit),
        factor,
    }
}

/// `text` as a part of a file or column name: lower case, every character
/// but `a-z`, `0-9` and `_` a `_`.
pub fn name_part(text: &str) -> String {
    text.to_lowercase()
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '_' {
                c
            } else {
                '_'
            }
        })
        .collect()
}

/// A class's name part (see [`name_part`]), with its id where another
/// class has the same one; its id where the results don't know it.
pub fn class_part(class: &ObjectClass, classes: &[Class]) -> String {
    let ObjectClass::Global(id) = class else {
        return "unset".into();
    };
    let Some(known) = classes.iter().find(|c| c.id == *id) else {
        return format!("class_{id}");
    };
    let part = name_part(&known.name);
    let shared = classes
        .iter()
        .filter(|c| name_part(&c.name) == part)
        .count()
        > 1;
    if shared { format!("{part}_{id}") } else { part }
}

/// A unit shown in nm as the Results views show it: in µm, and the factor
/// (`("µm²", 1e-6)` for "nm²"); any other unit as it is.
pub fn in_micrometres(unit: &str) -> (&str, f64) {
    match unit {
        "nm" => ("µm", 1e-3),
        "nm²" => ("µm²", 1e-6),
        "nm³" => ("µm³", 1e-9),
        other => (other, 1.0),
    }
}

/// The file name before the length unit is set: the column's key made a
/// name part, coloc columns by their classes' names.
fn file_name(column: &Column, classes: &[Class]) -> String {
    let joined = |list: &[ObjectClass]| {
        list.iter()
            .map(|class| class_part(class, classes))
            .collect::<Vec<_>>()
            .join("_")
    };
    match column {
        // The number of partner objects of the class.
        Column::ColocCount(class) => format!("coloc_with_{}", class_part(class, classes)),
        Column::Coloc(c) => {
            let base = match (c.how, c.with.is_empty()) {
                (ColocMatch::Exactly, true) => "coloc_with_no_class".to_string(),
                (ColocMatch::AtLeast, _) if c.with.len() == 1 => {
                    format!("coloc_with_{}", joined(&c.with))
                }
                (ColocMatch::AtLeast, _) => format!("coloc_at_least_{}", joined(&c.with)),
                (ColocMatch::Exactly, false) => format!("coloc_exactly_{}", joined(&c.with)),
            };
            // An object counts 1 if it matches; with `partners` its
            // partner objects are counted.
            match c.partners {
                true => format!("{base}_partners"),
                false => format!("{base}_objects"),
            }
        }
        other => name_part(&other.as_key(classes)),
    }
}

/// A column's kind of length, if it is one (stored in nm).
#[derive(Clone, Copy)]
enum Length {
    Line,
    Area,
    Volume,
}

fn length_of(column: &Column) -> Option<Length> {
    match column {
        Column::PerimeterNm => Some(Length::Line),
        Column::AreaSizeNm | Column::SurfaceAreaNm => Some(Length::Area),
        Column::VolumeNm => Some(Length::Volume),
        Column::Children(child) => match &child.value {
            ChildValue::Of(_, inner) => length_of(inner),
            ChildValue::Count => None,
        },
        Column::Parent(parent) => match &parent.value {
            ParentValue::Of(inner) => length_of(inner),
            ParentValue::Id => None,
        },
        _ => None,
    }
}

fn unit_of(column: &Column, length: Option<Length>, unit: LengthUnit) -> String {
    let base = match unit {
        LengthUnit::Um => "µm",
        LengthUnit::Nm => "nm",
    };
    match length {
        Some(Length::Line) => base.to_string(),
        Some(Length::Area) => format!("{base}²"),
        Some(Length::Volume) => format!("{base}³"),
        None => {
            let key = column.as_key(&[]);
            if key.ends_with("_px") {
                "px".into()
            } else if key.contains("_gray") {
                "gray value".into()
            } else {
                String::new()
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::results::ColocColumn;
    use evanalyzer_cfg::settings::classification_settings::Class;

    fn classes() -> Vec<Class> {
        [(2, "Nucleus"), (7, "Spot"), (11, "Cell"), (12, "cell")]
            .into_iter()
            .map(|(id, name)| Class {
                id,
                name: name.into(),
                ..Default::default()
            })
            .collect()
    }

    #[test]
    fn lengths_are_named_and_scaled_in_the_unit() {
        let p = present(&Column::AreaSizeNm, LengthUnit::Um, &[]);
        assert_eq!(
            (p.name.as_str(), p.unit.as_str(), p.factor),
            ("area_um2", "µm²", 1e-6)
        );
        let p = present(&Column::AreaSizeNm, LengthUnit::Nm, &[]);
        assert_eq!(
            (p.name.as_str(), p.unit.as_str(), p.factor),
            ("area_nm2", "nm²", 1.0)
        );
        assert_eq!(
            present(&Column::PerimeterNm, LengthUnit::Um, &[]).name,
            "perimeter_um"
        );
        assert_eq!(
            present(&Column::VolumeNm, LengthUnit::Um, &[]).name,
            "volume_um3"
        );
        assert_eq!(
            present(&Column::SurfaceAreaNm, LengthUnit::Um, &[]).name,
            "surface_area_um2"
        );
        let p = present(&Column::AreaSizePx, LengthUnit::Um, &[]);
        assert_eq!(
            (p.name.as_str(), p.unit.as_str(), p.factor),
            ("area_px", "px", 1.0)
        );
        let p = present(&Column::IntensityAvg(1), LengthUnit::Um, &[]);
        assert_eq!(
            (p.name.as_str(), p.unit.as_str()),
            ("intensity_mean_gray_ch1", "gray value")
        );
    }

    /// Coloc columns by their classes' names - with the id where two
    /// classes' names give the same part.
    #[test]
    fn coloc_columns_are_named_by_their_classes() {
        let classes = classes();
        let name = |column: Column| present(&column, LengthUnit::Um, &classes).name;
        let g = ObjectClass::Global;
        assert_eq!(name(Column::ColocCount(g(2))), "coloc_with_nucleus");
        let coloc = |how, with: Vec<u32>, partners| {
            Column::Coloc(ColocColumn {
                how,
                with: with.into_iter().map(g).collect(),
                among: vec![g(2), g(7), g(11)],
                partners,
            })
        };
        assert_eq!(
            name(coloc(ColocMatch::Exactly, vec![2, 7], false)),
            "coloc_exactly_nucleus_spot_objects"
        );
        assert_eq!(
            name(coloc(ColocMatch::AtLeast, vec![2, 7], true)),
            "coloc_at_least_nucleus_spot_partners"
        );
        assert_eq!(
            name(coloc(ColocMatch::Exactly, vec![], false)),
            "coloc_with_no_class_objects"
        );
        assert_eq!(name(Column::ColocCount(g(11))), "coloc_with_cell_11");
        assert_eq!(name(Column::ColocCount(g(99))), "coloc_with_class_99");
    }

    #[test]
    fn micrometres_for_nm() {
        assert_eq!(in_micrometres("nm²"), ("µm²", 1e-6));
        assert_eq!(in_micrometres("px"), ("px", 1.0));
    }
}
