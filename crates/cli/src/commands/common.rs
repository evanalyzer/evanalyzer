//! # common
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-06-26
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.

use evanalyzer_app::results::Cell;
use evanalyzer_app::results::CellValue;
use evanalyzer_app::results::PlaneFilter;
use evanalyzer_app::results::ResultsSource;
use evanalyzer_cfg::core_types::{InternalErrors, ObjectClass};

/// Resolves `--image` names into the `image_rel_path`s the results engine
/// actually filters on. `[]` (no `--image` given) stays `[]`, which every
/// `*Filter` in `evanalyzer_app::results` treats as "every image", so
/// callers don't need their own empty-means-all check.
pub fn resolve_image_rel_paths(
    db: &dyn ResultsSource,
    names: &[String],
) -> Result<Vec<String>, InternalErrors> {
    if names.is_empty() {
        return Ok(Vec::new());
    }
    let all = db.get_images()?;
    let wanted: std::collections::HashSet<&str> = names.iter().map(String::as_str).collect();
    let resolved: Vec<String> = all
        .into_iter()
        .filter(|image| wanted.contains(image.name.as_str()))
        .map(|image| image.rel_path.to_string_lossy().into_owned())
        .collect();
    if resolved.len() != wanted.len() {
        return Err(InternalErrors::InvalidArgument(format!(
            "one or more --image names not found (looked for {names:?})"
        )));
    }
    Ok(resolved)
}

/// Resolves `--class` names into `ObjectClass` ids the same way
/// `resolve_image_rel_paths` resolves `--image` names — `[]` means "every
/// class".
pub fn resolve_object_classes(
    db: &dyn ResultsSource,
    names: &[String],
) -> Result<Vec<ObjectClass>, InternalErrors> {
    if names.is_empty() {
        return Ok(Vec::new());
    }
    let all = db.get_object_classes()?;
    names
        .iter()
        .map(|name| {
            all.iter()
                .find(|class| &class.name == name)
                .map(|class| class.global())
                .ok_or_else(|| InternalErrors::InvalidArgument(format!("Unknown class '{name}'")))
        })
        .collect()
}

/// A `Cell`'s plain text — shared by `table.rs` (terminal preview) and
/// `export.rs` (CSV export), both of which just want a flat string per cell
/// rather than `Cell`'s own typed/colored value.
pub fn cell_text(cell: &Cell) -> String {
    match &cell.value {
        CellValue::Empty => String::new(),
        CellValue::String(s) => s.clone(),
        CellValue::Class((s, _)) => s.clone(),
        CellValue::Float(v) => v.to_string(),
        CellValue::Integer(v) => v.to_string(),
    }
}

/// The planes `view` shows: those with objects within `t_stacks`/`z_stacks`
/// (`None`: all), time point by time point - and whether the results have
/// several time points / z-planes (then "T" / "Z" columns show which).
pub struct ShownPlanes {
    pub planes: Vec<PlaneFilter>,
    pub by_t: bool,
    pub by_z: bool,
}

pub fn shown_planes(
    db: &dyn ResultsSource,
    t_stacks: Option<std::range::Range<u32>>,
    z_stacks: Option<std::range::Range<u32>>,
) -> Result<ShownPlanes, InternalErrors> {
    let all = db.get_planes()?;
    let within = |value: u32, range: Option<std::range::Range<u32>>| {
        range.is_none_or(|range| range.start <= value && value < range.end)
    };
    Ok(ShownPlanes {
        planes: all
            .iter()
            .filter(|p| within(p.t_stack, t_stacks) && within(p.z_stack, z_stacks))
            .copied()
            .collect(),
        by_t: all.iter().any(|p| p.t_stack != all[0].t_stack),
        by_z: all.iter().any(|p| p.z_stack != all[0].z_stack),
    })
}
