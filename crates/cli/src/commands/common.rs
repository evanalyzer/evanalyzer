//! # common
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-06-26
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.

use crate::args::{AggKind, GroupArgs, GroupByKind, LayoutKind, NormalizeKind};
use evanalyzer_app::results::Cell;
use evanalyzer_app::results::CellValue;
use evanalyzer_app::results::PlaneFilter;
use evanalyzer_app::results::ResultsSource;
use evanalyzer_app::results::{
    Aggregation, Grouping, TimeCourseExport, TimeCourseLayout, TimeCourseLines,
    TimeCourseNormalize, TimeCourseStat, TrackHistory,
};
use evanalyzer_cfg::core_types::{InternalErrors, ObjectClass};

/// Resolves `--image` names into the `image_rel_path`s the results engine
/// actually filters on. `[]` (no `--image` given) stays `[]`, which every
/// `*Filter`/`ResultExport` in `evanalyzer_app::result` already treats as
/// "every image", so callers don't need their own empty-means-all check.
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

/// The aggregation of `kind` for a per-image export - none for the
/// quartiles, which only a time course has.
pub fn to_aggregation(kind: AggKind) -> Option<Aggregation> {
    match kind {
        AggKind::Min => Some(Aggregation::Min),
        AggKind::Max => Some(Aggregation::Max),
        AggKind::Avg => Some(Aggregation::Avg),
        AggKind::Median => Some(Aggregation::Median),
        AggKind::Stdev => Some(Aggregation::Stddev),
        AggKind::Sum => Some(Aggregation::Sum),
        AggKind::Q1 | AggKind::Q3 => None,
    }
}

pub fn to_time_course_stat(kind: AggKind) -> TimeCourseStat {
    match kind {
        AggKind::Min => TimeCourseStat::Min,
        AggKind::Max => TimeCourseStat::Max,
        AggKind::Avg => TimeCourseStat::Mean,
        AggKind::Median => TimeCourseStat::Median,
        AggKind::Stdev => TimeCourseStat::Sd,
        AggKind::Sum => TimeCourseStat::Sum,
        AggKind::Q1 => TimeCourseStat::Q1,
        AggKind::Q3 => TimeCourseStat::Q3,
    }
}

/// What `GroupArgs` asks for.
#[derive(Debug, PartialEq)]
pub enum ResolvedGrouping {
    /// A row per object.
    List,
    /// A row per image and class, per time point, with these aggregations.
    PerImage(Vec<Aggregation>),
    /// The time course - its lines' wells / groups by `grouping`.
    TimeCourse {
        grouping: Grouping,
        settings: TimeCourseExport,
    },
}

pub fn resolve_grouping(args: &GroupArgs) -> Result<ResolvedGrouping, InternalErrors> {
    let time_course = |lines: TimeCourseLines, grouping: Grouping| {
        let statistics: Vec<TimeCourseStat> = match args.agg.is_empty() {
            true => TimeCourseStat::DEFAULT.to_vec(),
            false => args.agg.iter().copied().map(to_time_course_stat).collect(),
        };
        ResolvedGrouping::TimeCourse {
            grouping,
            settings: TimeCourseExport {
                lines,
                statistics,
                normalize: match args.normalize {
                    Some(NormalizeKind::T0) => TimeCourseNormalize::RelativeToStart,
                    Some(NormalizeKind::None) | None => TimeCourseNormalize::None,
                },
                layout: match args.layout {
                    LayoutKind::Long => TimeCourseLayout::Long,
                    LayoutKind::Wide => TimeCourseLayout::Wide,
                    LayoutKind::Both => TimeCourseLayout::Both,
                },
                history: match args.with_parents {
                    true => TrackHistory::WithParents,
                    false => TrackHistory::Own,
                },
            },
        }
    };
    let regex = || {
        args.group_regex
            .clone()
            .filter(|regex| !regex.is_empty())
            .map(Grouping::Regex)
            .ok_or_else(|| {
                InvalidArgument(
                    "--group-by group / regex needs --group-regex (its first capture group is \
                     the group), or use --group-by folder"
                        .to_string(),
                )
            })
    };
    use InternalErrors::InvalidArgument;
    // `--group-by-class` changes nothing: every grouping splits by class.
    Ok(match args.group_by {
        None => ResolvedGrouping::List,
        Some(GroupByKind::Image) => {
            let aggregations = args
                .agg
                .iter()
                .map(|kind| {
                    to_aggregation(*kind).ok_or_else(|| {
                        InvalidArgument(
                            "q1/q3 are only for a time course (--group-by class, well, group, \
                             folder, regex or track)"
                                .to_string(),
                        )
                    })
                })
                .collect::<Result<Vec<_>, _>>()?;
            ResolvedGrouping::PerImage(match aggregations.is_empty() {
                true => vec![Aggregation::Avg],
                false => aggregations,
            })
        }
        Some(GroupByKind::Class) => time_course(TimeCourseLines::Class, Grouping::Auto),
        Some(GroupByKind::Well) => time_course(TimeCourseLines::Well, Grouping::Auto),
        Some(GroupByKind::Group | GroupByKind::Regex) => {
            time_course(TimeCourseLines::Group, regex()?)
        }
        Some(GroupByKind::Folder) => time_course(TimeCourseLines::Group, Grouping::Folder),
        Some(GroupByKind::Track) => time_course(TimeCourseLines::Track, Grouping::Auto),
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
    fn resolve_grouping_defaults_to_a_list() {
        let resolved = resolve_grouping(&GroupArgs::default()).expect("resolve_grouping");
        assert_eq!(resolved, ResolvedGrouping::List);
    }

    #[test]
    fn resolve_grouping_maps_image_group_by_and_its_aggregations() {
        let args = GroupArgs {
            group_by: Some(GroupByKind::Image),
            agg: vec![AggKind::Min, AggKind::Max, AggKind::Sum],
            ..Default::default()
        };
        assert_eq!(
            resolve_grouping(&args).unwrap(),
            ResolvedGrouping::PerImage(vec![Aggregation::Min, Aggregation::Max, Aggregation::Sum])
        );
        let default = GroupArgs {
            group_by: Some(GroupByKind::Image),
            ..Default::default()
        };
        assert_eq!(
            resolve_grouping(&default).unwrap(),
            ResolvedGrouping::PerImage(vec![Aggregation::Avg])
        );
        let quartiles = GroupArgs {
            group_by: Some(GroupByKind::Image),
            agg: vec![AggKind::Q1],
            ..Default::default()
        };
        assert!(
            resolve_grouping(&quartiles).is_err(),
            "q1 only for a time course"
        );
    }

    /// class / well / group / folder / regex / track: the time course -
    /// mean and SD unless asked otherwise.
    #[test]
    fn resolve_grouping_maps_the_time_course_groupings() {
        let resolve = |group_by, group_regex: Option<&str>| {
            resolve_grouping(&GroupArgs {
                group_by: Some(group_by),
                group_regex: group_regex.map(str::to_string),
                ..Default::default()
            })
        };
        let lines_and_grouping = |resolved: ResolvedGrouping| match resolved {
            ResolvedGrouping::TimeCourse { grouping, settings } => (settings.lines, grouping),
            other => panic!("{other:?}"),
        };
        assert_eq!(
            lines_and_grouping(resolve(GroupByKind::Class, None).unwrap()),
            (TimeCourseLines::Class, Grouping::Auto)
        );
        assert_eq!(
            lines_and_grouping(resolve(GroupByKind::Well, None).unwrap()),
            (TimeCourseLines::Well, Grouping::Auto)
        );
        assert_eq!(
            lines_and_grouping(resolve(GroupByKind::Folder, None).unwrap()),
            (TimeCourseLines::Group, Grouping::Folder)
        );
        assert_eq!(
            lines_and_grouping(resolve(GroupByKind::Group, Some("^(.*)_")).unwrap()),
            (TimeCourseLines::Group, Grouping::Regex("^(.*)_".into()))
        );
        assert_eq!(
            lines_and_grouping(resolve(GroupByKind::Track, None).unwrap()),
            (TimeCourseLines::Track, Grouping::Auto)
        );
        assert!(
            resolve(GroupByKind::Group, None).is_err(),
            "needs --group-regex"
        );
        assert!(resolve(GroupByKind::Regex, Some("")).is_err());

        let ResolvedGrouping::TimeCourse { settings, .. } =
            resolve(GroupByKind::Class, None).unwrap()
        else {
            panic!("a time course")
        };
        assert_eq!(settings.statistics, TimeCourseStat::DEFAULT);
        assert_eq!(settings.normalize, TimeCourseNormalize::None);
        assert_eq!(settings.layout, TimeCourseLayout::Long);

        let ResolvedGrouping::TimeCourse { settings, .. } = resolve_grouping(&GroupArgs {
            group_by: Some(GroupByKind::Class),
            agg: vec![AggKind::Median, AggKind::Q1, AggKind::Q3],
            normalize: Some(NormalizeKind::T0),
            layout: LayoutKind::Both,
            ..Default::default()
        })
        .unwrap() else {
            panic!("a time course")
        };
        assert_eq!(
            settings.statistics,
            [
                TimeCourseStat::Median,
                TimeCourseStat::Q1,
                TimeCourseStat::Q3
            ]
        );
        assert_eq!(settings.normalize, TimeCourseNormalize::RelativeToStart);
        assert_eq!(settings.layout, TimeCourseLayout::Both);
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
