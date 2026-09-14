use crate::args::{AggKind, GroupArgs, GroupByKind};
use evanalyzer_app::result::{Aggregation, Cell, CellValue, ResultsGenerator};
use evanalyzer_cfg::core_types::{InternalErrors, ObjectClass};

/// Resolves `--image` names into the `image_rel_path`s the results engine
/// actually filters on. `[]` (no `--image` given) stays `[]`, which every
/// `*Filter`/`ResultExport` in `evanalyzer_app::result` already treats as
/// "every image", so callers don't need their own empty-means-all check.
pub fn resolve_image_rel_paths(
    db: &ResultsGenerator,
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
    db: &ResultsGenerator,
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
                .map(|class| class.id)
                .ok_or_else(|| InternalErrors::InvalidArgument(format!("Unknown class '{name}'")))
        })
        .collect()
}

pub fn to_aggregation(kind: AggKind) -> Aggregation {
    match kind {
        AggKind::Min => Aggregation::Min,
        AggKind::Max => Aggregation::Max,
        AggKind::Avg => Aggregation::Avg,
        AggKind::Median => Aggregation::Median,
        AggKind::Stdev => Aggregation::Stddev,
        AggKind::Sum => Aggregation::Sum,
    }
}

/// What `GroupArgs` resolves to against the current results backend — only
/// "group by image" (`ResultsGenerator::get_grouped_by_image`) has a real
/// equivalent today. `--group-by folder`/`--group-by regex` and
/// `--split-colocalized` don't (the Plate/Well grid views' own grouping
/// regex is a coordinate-extracting concept, not a generic group-by key;
/// there's no "is this object colocalized with anything at all" row filter
/// in the current schema-level API), so those return a clear error instead
/// of silently doing something else.
pub struct ResolvedGrouping {
    pub group_by_image: bool,
    pub aggregations: Vec<Aggregation>,
}

pub fn resolve_grouping(args: &GroupArgs) -> Result<ResolvedGrouping, InternalErrors> {
    let group_by_image = match args.group_by {
        None => false,
        Some(GroupByKind::Image) => true,
        Some(GroupByKind::Folder) => {
            return Err(InternalErrors::InvalidArgument(
                "--group-by folder isn't supported by the current results backend (only \
                 --group-by image has a matching query — ResultsGenerator::get_grouped_by_image)"
                    .to_string(),
            ));
        }
        Some(GroupByKind::Regex) => {
            return Err(InternalErrors::InvalidArgument(
                "--group-by regex isn't supported for table export (that's the Plate/Well grid \
                 views' own well-coordinate regex, not a generic group-by key) — use \
                 --group-by image, or omit --group-by for a flat per-object table"
                    .to_string(),
            ));
        }
    };
    if args.split_colocalized {
        return Err(InternalErrors::InvalidArgument(
            "--split-colocalized isn't supported by the current results backend".to_string(),
        ));
    }
    // `--group-by-class` is a no-op when grouping by image:
    // `get_grouped_by_image` always splits by class (an object's class is
    // part of what it groups by, not an optional extra split), so there's
    // nothing to opt into or out of here.
    let aggregations: Vec<Aggregation> = args.agg.iter().copied().map(to_aggregation).collect();
    Ok(ResolvedGrouping {
        group_by_image,
        // clap's own `default_value = "avg"` only applies when parsing real
        // CLI args — `GroupArgs::default()` (e.g. in tests, or a caller
        // building one directly) gives an empty `Vec` instead, so this
        // covers that case the same way the flag's own default would.
        aggregations: if aggregations.is_empty() {
            vec![Aggregation::Avg]
        } else {
            aggregations
        },
    })
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::args::GroupArgs;

    #[test]
    fn resolve_grouping_defaults_to_flat_with_avg_aggregation() {
        let resolved = resolve_grouping(&GroupArgs::default()).expect("resolve_grouping");
        assert!(!resolved.group_by_image);
        assert!(resolved.aggregations == vec![Aggregation::Avg]);
    }

    #[test]
    fn resolve_grouping_maps_image_group_by_and_every_agg_kind() {
        let args = GroupArgs {
            group_by: Some(GroupByKind::Image),
            agg: vec![AggKind::Min, AggKind::Max, AggKind::Sum],
            ..Default::default()
        };
        let resolved = resolve_grouping(&args).expect("resolve_grouping");
        assert!(resolved.group_by_image);
        assert!(
            resolved.aggregations == vec![Aggregation::Min, Aggregation::Max, Aggregation::Sum]
        );
    }

    #[test]
    fn resolve_grouping_rejects_folder_and_regex_and_split_colocalized() {
        assert!(
            resolve_grouping(&GroupArgs {
                group_by: Some(GroupByKind::Folder),
                ..Default::default()
            })
            .is_err()
        );
        assert!(
            resolve_grouping(&GroupArgs {
                group_by: Some(GroupByKind::Regex),
                ..Default::default()
            })
            .is_err()
        );
        assert!(
            resolve_grouping(&GroupArgs {
                split_colocalized: true,
                ..Default::default()
            })
            .is_err()
        );
    }

    #[test]
    fn resolve_grouping_ignores_group_by_class_when_grouping_by_image() {
        // `get_grouped_by_image` always splits by class regardless of this
        // flag — resolving must not error either way.
        assert!(
            resolve_grouping(&GroupArgs {
                group_by: Some(GroupByKind::Image),
                group_by_class: false,
                ..Default::default()
            })
            .is_ok()
        );
        assert!(
            resolve_grouping(&GroupArgs {
                group_by: Some(GroupByKind::Image),
                group_by_class: true,
                ..Default::default()
            })
            .is_ok()
        );
    }
}
