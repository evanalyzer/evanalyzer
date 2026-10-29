//! # results_time_course
//!
//! The time course: per time point (and z-plane), statistics of values over
//! each line's objects - a class's, an image's, a well's, a group's or a
//! track's. What the "Over time" chart draws and its export writes.
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-10-21
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.

use super::results_generator::{
    ResultsGenerator, class_blocks, class_filter_sql, column_aggregate_expr, objects_per_class_sql,
    sql_string_in_list,
};
use crate::api::{
    Column, PlaneFilter, TimeCourseFilter, TimeCourseLines, TimeCourseNormalize, TimeCoursePoint,
    TimeCourseResult, TimeCourseStat, TrackHistory,
};
use duckdb::arrow::array::{Array, Float64Array, Int64Array, StringArray};
use duckdb::arrow::record_batch::RecordBatch;
use evanalyzer_cfg::core_types::{InternalErrors, ObjectClass};
use std::collections::{HashMap, HashSet};

/// The statistic a normalized one is divided by, at the line's first time
/// point: SD by the mean, the quartiles by the median, the others by
/// themselves.
fn normalization_base(stat: TimeCourseStat) -> TimeCourseStat {
    match stat {
        TimeCourseStat::Sd => TimeCourseStat::Mean,
        TimeCourseStat::Q1 | TimeCourseStat::Q3 => TimeCourseStat::Median,
        other => other,
    }
}

/// The quantile a statistic is - `None` for the others.
fn quantile(stat: TimeCourseStat) -> Option<f64> {
    match stat {
        TimeCourseStat::Median => Some(0.5),
        TimeCourseStat::Q1 => Some(0.25),
        TimeCourseStat::Q3 => Some(0.75),
        _ => None,
    }
}

pub(super) fn stat_sql(stat: TimeCourseStat, value: &str) -> String {
    let call = match stat {
        TimeCourseStat::Mean => format!("AVG({value})"),
        TimeCourseStat::Sd => format!("STDDEV_SAMP({value})"),
        TimeCourseStat::Median => format!("MEDIAN({value})"),
        TimeCourseStat::Q1 => format!("QUANTILE_CONT({value}, 0.25)"),
        TimeCourseStat::Q3 => format!("QUANTILE_CONT({value}, 0.75)"),
        TimeCourseStat::Min => format!("MIN({value})"),
        TimeCourseStat::Max => format!("MAX({value})"),
        TimeCourseStat::Sum => format!("SUM({value})"),
    };
    format!("CAST({call} AS DOUBLE)")
}

/// What every point of a time course holds - known before its rows are read.
#[derive(Debug, Clone, PartialEq)]
pub(super) struct TimeCourseHead {
    pub value_names: Vec<String>,
    pub by_z: bool,
    pub frame_interval_s: Option<f64>,
    /// The last time point asked for - with the frame interval, how long the
    /// series runs.
    pub last_t: Option<u32>,
    /// The results have tracks: the points' `n_tracks` count.
    pub tracked: bool,
}

/// How the rows of the query become points.
struct Plan {
    /// Run in turn - their rows in order one after the other.
    sql: Vec<String>,
    lines: TimeCourseLines,
    with_image: bool,
    class_names: HashMap<u32, String>,
    n_columns: usize,
    stats: Vec<TimeCourseStat>,
    /// `stats` and, normalized, what they're divided by.
    computed: Vec<TimeCourseStat>,
    normalized: bool,
    /// Track lines with their parents' history: the lines each track of a
    /// family is in (by track id) - `None` otherwise, and without lineage.
    lineage: Option<HashMap<u64, Vec<String>>>,
}

/// At most this many lines end at one track: each fusion before it doubles
/// them.
const MAX_LINES_PER_TRACK: usize = 64;

/// Track lines with their parents' history: the lines each track of a
/// family (`parents`: a track's parents) is in - a line per way from a
/// track without parents to one without children, labelled "#5 · from #1
/// › #3" (the last track, then the ones before it). `leaves`: only the
/// lines ending at those. Tracks of no family aren't listed: their line is
/// their own.
fn lineage_lines(
    parents: &HashMap<u64, Vec<u64>>,
    leaves: Option<&HashSet<u64>>,
) -> HashMap<u64, Vec<String>> {
    let with_children: HashSet<u64> = parents.values().flatten().copied().collect();
    let mut ends: Vec<u64> = parents
        .keys()
        .copied()
        .filter(|t| !with_children.contains(t) && leaves.is_none_or(|l| l.contains(t)))
        .collect();
    ends.sort_unstable();
    let mut lines: HashMap<u64, Vec<String>> = HashMap::new();
    for end in ends {
        // Every way up, depth first: (track, the way from `end` to it).
        let mut stack = vec![vec![end]];
        let mut found = 0;
        while let Some(way) = stack.pop() {
            let track = *way.last().expect("a way has a track");
            match parents.get(&track).filter(|p| !p.is_empty()) {
                Some(of) => {
                    for parent in of.iter().rev() {
                        // A broken file's loop isn't followed.
                        if !way.contains(parent) {
                            let mut longer = way.clone();
                            longer.push(*parent);
                            stack.push(longer);
                        }
                    }
                }
                None => {
                    let before: Vec<String> =
                        way[1..].iter().rev().map(|t| format!("#{t}")).collect();
                    let label = format!("#{end} · from {}", before.join(" › "));
                    for t in &way {
                        lines.entry(*t).or_default().push(label.clone());
                    }
                    found += 1;
                    if found == MAX_LINES_PER_TRACK {
                        break;
                    }
                }
            }
        }
    }
    lines
}

/// The time course `filter` asks for, ready to run. Every time point the run
/// analysed (within `filter`'s ranges) has a point per line and class - with
/// no objects there, N objects 0, Sum 0 and no other value. Track lines only
/// have points where the track has an object. Failed and disabled images are
/// left out.
///
/// Its points are read a chunk at a time ([`Self::for_each`]): a per-track
/// time course has millions - an export writes them as they come.
pub(super) struct TimeCourseQuery<'a> {
    database: &'a ResultsGenerator,
    pub head: TimeCourseHead,
    /// `None`: no points (nothing selected).
    plan: Option<Plan>,
}

impl<'a> TimeCourseQuery<'a> {
    pub fn prepare(
        database: &'a ResultsGenerator,
        filter: &TimeCourseFilter,
    ) -> Result<Self, InternalErrors> {
        let classes = database.get_object_classes()?;
        let columns: Vec<&Column> = filter
            .columns
            .iter()
            .filter(|column| {
                column.is_aggregable()
                    && **column != Column::Count
                    && !column.is_image_measurement()
            })
            .collect();
        let stats = &filter.statistics;
        let normalized = filter.normalize == TimeCourseNormalize::RelativeToStart;
        let value_names: Vec<String> = columns
            .iter()
            .flat_map(|column| {
                let label = column.display_label(&classes);
                stats.iter().map(move |stat| match normalized {
                    true => format!("{label} {} [% of start]", stat.label()),
                    false => format!("{label} {}", stat.label()),
                })
            })
            .collect();

        let all = database.get_planes()?;
        let by_z = all.iter().any(|p| p.z_stack != all[0].z_stack);
        let within = |value: u32, range: &Option<std::range::Range<u32>>| {
            range.is_none_or(|range| range.start <= value && value < range.end)
        };
        let planes: Vec<PlaneFilter> = all
            .iter()
            .filter(|p| within(p.t_stack, &filter.t_stacks) && within(p.z_stack, &filter.z_stacks))
            .copied()
            .collect();
        let mut head = TimeCourseHead {
            value_names,
            by_z,
            frame_interval_s: None,
            last_t: planes.iter().map(|p| p.t_stack).max(),
            tracked: database
                .connection()
                .query_row(
                    "SELECT EXISTS (SELECT 1 FROM objects WHERE COALESCE(track_id, 0) <> 0)",
                    [],
                    |row| row.get(0),
                )
                .map_err(|e| InternalErrors::Io(e.to_string()))?,
        };
        let selected_ids: Option<Vec<u32>> = filter.object_classes.as_ref().map(|wanted| {
            wanted
                .iter()
                .filter_map(|class| match class {
                    ObjectClass::Global(n) => Some(*n),
                    ObjectClass::Unset | ObjectClass::Local(_) => None,
                })
                .collect()
        });
        if planes.is_empty()
            || columns.is_empty()
            || stats.is_empty()
            || selected_ids.as_ref().is_some_and(Vec::is_empty)
            || filter.images.as_ref().is_some_and(Vec::is_empty)
        {
            return Ok(Self {
                database,
                head,
                plan: None,
            });
        }

        // The images counted: analysed (not failed), not left out, of a well
        // / group when the lines are those.
        let grouping = database.grouping_sql(&filter.grouping)?;
        let line_of_image = match filter.lines {
            TimeCourseLines::Class | TimeCourseLines::Track => "''".to_string(),
            TimeCourseLines::Image => "i.image_rel_path".to_string(),
            TimeCourseLines::Well | TimeCourseLines::Group => format!(
                "regexp_extract(i.{}, '{}', 1)",
                grouping.column, grouping.regex
            ),
        };
        let mut image_conditions = vec!["NOT i.disabled".to_string(), "i.successful".to_string()];
        let mut object_conditions = vec!["(t_stack, z_stack) IN (SELECT t, z FROM p)".to_string()];
        if let Some(paths) = &filter.images {
            image_conditions.push(format!(
                "i.image_rel_path IN ({})",
                sql_string_in_list(paths)
            ));
            object_conditions.push(format!("image_rel_path IN ({})", sql_string_in_list(paths)));
        }
        if matches!(filter.lines, TimeCourseLines::Well | TimeCourseLines::Group) {
            image_conditions.push(format!("{line_of_image} <> ''"));
        }
        let image_where = image_conditions.join(" AND ");
        head.frame_interval_s = frame_interval(database, &image_where);

        let mut class_post = vec!["true".to_string()];
        if let Some(ids) = &selected_ids {
            object_conditions.push(class_filter_sql("object_class_id", ids));
            class_post.push(format!(
                "class_id IN ({})",
                ids.iter()
                    .map(u32::to_string)
                    .collect::<Vec<_>>()
                    .join(", ")
            ));
        }
        // With their parents' history: the families (results with lineage
        // only - one small query; every other time course is as before).
        let parents: Option<HashMap<u64, Vec<u64>>> = match (filter.lines, filter.history) {
            (TimeCourseLines::Track, TrackHistory::WithParents) if database.has_lineage()? => Some(
                database
                    .all_track_parents()?
                    .into_values()
                    .flatten()
                    .collect(),
            ),
            _ => None,
        };
        let mut lineage = None;
        if filter.lines == TimeCourseLines::Track {
            object_conditions.push("COALESCE(track_id, 0) <> 0".to_string());
            // A sample of the tracks - the same each time - picked in the
            // database: with many tracks, reading every one only to draw a few
            // hundred takes seconds.
            match (filter.max_tracks, &parents) {
                (Some(n), None) => object_conditions.push(format!(
                    "track_id IN (SELECT track_id FROM (SELECT DISTINCT track_id FROM objects WHERE {}) \
                     USING SAMPLE reservoir({n} ROWS) REPEATABLE (7))",
                    object_conditions.join(" AND ")
                )),
                // A sample of the lines: of the tracks without children,
                // then the tracks before them - whole lines.
                (Some(n), Some(parents)) => {
                    let sampled = sample_line_ends(
                        database,
                        &planes,
                        &object_conditions.join(" AND "),
                        parents,
                        n,
                    )?;
                    let mut tracks: HashSet<u64> = sampled.iter().copied().collect();
                    let mut up: Vec<u64> = sampled.clone();
                    while let Some(track) = up.pop() {
                        for parent in parents.get(&track).into_iter().flatten() {
                            if tracks.insert(*parent) {
                                up.push(*parent);
                            }
                        }
                    }
                    object_conditions.push(format!("track_id IN ({})", id_list(&tracks)));
                    lineage = Some(lineage_lines(parents, Some(&sampled.into_iter().collect())));
                }
                (None, Some(parents)) => lineage = Some(lineage_lines(parents, None)),
                (None, None) => {}
            }
        }
        let mut row_classes = class_blocks(selected_ids.as_deref(), &classes);
        if selected_ids.is_none() {
            row_classes.retain(|id| ObjectClass::Global(*id) != ObjectClass::BACKGROUND);
        }

        let mut computed: Vec<TimeCourseStat> = stats.clone();
        if normalized {
            for stat in stats {
                let base = normalization_base(*stat);
                if !computed.contains(&base) {
                    computed.push(base);
                }
            }
        }
        let values_sql = columns
            .iter()
            .enumerate()
            .map(|(i, column)| Ok(format!("{} AS v{i}", column_aggregate_expr(column)?)))
            .collect::<Result<Vec<_>, InternalErrors>>()?
            .join(", ");
        // A value's quantiles in one aggregate, `q{i}`: each would keep its
        // own copy of every value (median + IQR: 730 instead of 390 MB on
        // 6.6M objects).
        let quantiles: Vec<f64> = computed.iter().filter_map(|s| quantile(*s)).collect();
        let stats_sql = (0..columns.len())
            .flat_map(|i| {
                let value = format!("obj.v{i}");
                let list = (!quantiles.is_empty()).then(|| {
                    let at = quantiles
                        .iter()
                        .map(f64::to_string)
                        .collect::<Vec<_>>()
                        .join(", ");
                    format!("QUANTILE_CONT({value}, [{at}])::DOUBLE[] AS q{i}")
                });
                computed
                    .iter()
                    .enumerate()
                    .filter(|(_, stat)| quantile(**stat).is_none())
                    .map(move |(k, stat)| format!("{} AS s{i}_{k}", stat_sql(*stat, &value)))
                    .chain(list)
            })
            .collect::<Vec<_>>()
            .join(", ");
        let stat_names = (0..columns.len())
            .flat_map(|i| {
                let quantiles = &quantiles;
                computed
                    .iter()
                    .enumerate()
                    .map(move |(k, stat)| match quantile(*stat) {
                        Some(q) => {
                            let at = quantiles.iter().position(|x| *x == q).expect("a quantile");
                            format!("a.q{i}[{}]", at + 1)
                        }
                        None => format!("a.s{i}_{k}"),
                    })
            })
            .collect::<Vec<_>>()
            .join(", ");

        let planes_sql = |planes: &[PlaneFilter]| {
            planes
                .iter()
                .map(|p| format!("({}, {})", p.t_stack, p.z_stack))
                .collect::<Vec<_>>()
                .join(", ")
        };
        let measured = match database.analysed_planes_recorded() {
            true => {
                "list_contains(img.t_stacks_analyzed, p.t) AND list_contains(img.z_stacks_analyzed, p.z)"
            }
            // Results from before the run recorded its planes: within the
            // image's own sizes (the planes are the ones with objects).
            false => "p.z < img.z_stacks AND p.t < img.t_stacks",
        };
        // A track has at most one object per plane and class (the tracker
        // links each object once, track ids are unique across images): its
        // objects there are its tracks. A distinct count would hold every
        // (plane, track) pair in memory - hundreds of MB.
        let (line_of_object, counts) = match filter.lines {
            TimeCourseLines::Track => (
                "obj.track_id",
                "1 AS n_images, COUNT(*) AS n_objects, 1 AS n_tracks",
            ),
            _ => (
                "img.line",
                "COUNT(DISTINCT obj.image_rel_path) AS n_images, COUNT(*) AS n_objects, \
                 COUNT(obj.track_id) AS n_tracks",
            ),
        };
        let objects = objects_per_class_sql(&object_conditions.join(" AND "));
        let common = |planes_sql: String| {
            format!(
                "WITH p(t, z) AS (VALUES {planes_sql}),\n\
             img AS (SELECT i.*, {line_of_image} AS line FROM images i WHERE {image_where}),\n\
             obj AS (SELECT o.t_stack AS t, o.z_stack AS z, o.image_rel_path, o.class_id, NULLIF(o.track_id, 0) AS track_id, \
                     {values_sql} FROM ({objects}) o WHERE {}),\n\
             a AS (SELECT obj.t, obj.z, {line_of_object} AS line, obj.class_id, \
                     MIN(img.image_name) AS image, {counts}, {stats_sql} \
                   FROM obj JOIN img USING (image_rel_path) GROUP BY 1, 2, 3, 4)",
                class_post.join(" AND ")
            )
        };
        // The columns' types fixed: read as Arrow arrays of those.
        let sql = match filter.lines {
            // A track is where its objects are. Every track: a query per
            // plane - grouping and sorting millions of rows at once takes
            // GBs, a plane's are a few MB (the order is the same).
            TimeCourseLines::Track => {
                let chunks: Vec<&[PlaneFilter]> = match filter.max_tracks {
                    Some(_) => vec![&planes[..]],
                    None => planes.chunks(1).collect(),
                };
                chunks
                    .into_iter()
                    .map(|planes| {
                        let common = common(planes_sql(planes));
                        format!(
                            "{common}\n\
                 SELECT a.t::BIGINT, a.z::BIGINT, a.line::VARCHAR, a.image::VARCHAR, \
                        a.class_id::BIGINT, a.n_images::BIGINT, a.n_objects::BIGINT, \
                        a.n_tracks::BIGINT, {stat_names}\n\
                 FROM a ORDER BY a.t, a.z, a.line, a.class_id"
                        )
                    })
                    .collect()
            }
            // Every analysed plane x line x class - also without objects.
            _ => vec![format!(
                "{},\n\
                 m AS (SELECT p.t, p.z, img.line, MIN(img.image_name) AS image, COUNT(*) AS n_images \
                       FROM p, img WHERE {measured} \
                       GROUP BY 1, 2, 3),\n\
                 f AS (SELECT m.*, c.class_id FROM m, (SELECT UNNEST({}) AS class_id) c)\n\
                 SELECT COALESCE(f.t, a.t)::BIGINT, COALESCE(f.z, a.z)::BIGINT, \
                        COALESCE(f.line, a.line)::VARCHAR, COALESCE(f.image, a.image)::VARCHAR, \
                        COALESCE(f.class_id, a.class_id)::BIGINT, \
                        COALESCE(f.n_images, a.n_images)::BIGINT, COALESCE(a.n_objects, 0)::BIGINT, \
                        COALESCE(a.n_tracks, 0)::BIGINT, {stat_names}\n\
                 FROM f FULL JOIN a ON f.t = a.t AND f.z = a.z AND f.line = a.line \
                     AND f.class_id = a.class_id\n\
                 ORDER BY 1, 2, 3, 5",
                common(planes_sql(&planes)),
                sql_u32_array(&row_classes)
            )],
        };
        Ok(Self {
            database,
            head,
            plan: Some(Plan {
                sql,
                lines: filter.lines,
                with_image: matches!(
                    filter.lines,
                    TimeCourseLines::Image | TimeCourseLines::Track
                ),
                class_names: classes.iter().map(|c| (c.id, c.name.clone())).collect(),
                n_columns: columns.len(),
                stats: stats.clone(),
                computed,
                normalized,
                lineage,
            }),
        })
    }

    /// Each point, in order: time point by time point, then z-plane, line and
    /// class - track lines with their parents' history after a plane's
    /// other lines (each line in time order). Read from the database a chunk
    /// at a time.
    pub fn for_each(
        &self,
        mut visit: impl FnMut(&TimeCoursePoint) -> Result<(), InternalErrors>,
    ) -> Result<(), InternalErrors> {
        let Some(plan) = &self.plan else {
            return Ok(());
        };
        let err = |e: duckdb::Error| InternalErrors::Io(e.to_string());
        let n_stats = plan.n_columns * plan.computed.len();
        // Without objects a sum is 0, every other statistic none.
        let value_of = |n_objects: u64, computed_values: &[Option<f64>], i: usize, stat| {
            let k = plan
                .computed
                .iter()
                .position(|s| *s == stat)
                .expect("computed");
            match (n_objects, stat) {
                (0, TimeCourseStat::Sum) => Some(0.0),
                _ => computed_values[i * plan.computed.len() + k],
            }
        };
        // Normalized: each line's first time point with objects is 100 % -
        // the rows come time point by time point, so it's the first seen.
        let mut starts: HashMap<(u32, String, String, u32), Vec<Option<f64>>> = HashMap::new();
        // A point's values (normalized), then to `visit`.
        let mut finish = |point: &mut TimeCoursePoint,
                          computed_values: &[Option<f64>]|
         -> Result<(), InternalErrors> {
            let start = match plan.normalized {
                false => None,
                true => {
                    let key = (
                        point.z,
                        point.line.clone(),
                        point.image.clone(),
                        point.class_id,
                    );
                    if point.n_objects > 0 && !starts.contains_key(&key) {
                        starts.insert(key.clone(), computed_values.to_vec());
                    }
                    starts.get(&key)
                }
            };
            point.values.clear();
            point.values.extend(
                (0..plan.n_columns)
                    .flat_map(|i| plan.stats.iter().map(move |stat| (i, *stat)))
                    .map(|(i, stat)| {
                        let value = value_of(point.n_objects, computed_values, i, stat);
                        match (plan.normalized, start) {
                            (false, _) => value,
                            (true, None) => None,
                            (true, Some(start)) => {
                                let base =
                                    value_of(point.n_objects, start, i, normalization_base(stat));
                                match (value, base) {
                                    (Some(value), Some(base)) if base != 0.0 => {
                                        Some(value / base * 100.0)
                                    }
                                    _ => None,
                                }
                            }
                        }
                    }),
            );
            visit(point)
        };
        // With their parents' history: a chunk's points of families, each
        // in each of its lines - after the chunk's other points, in order.
        // Only they are copied and sorted.
        let mut lined: Vec<(TimeCoursePoint, Vec<Option<f64>>)> = Vec::new();

        let mut computed_values = vec![None; n_stats];
        let mut point = TimeCoursePoint::default();
        for sql in &plan.sql {
            let mut statement = self.database.connection().prepare(sql).map_err(err)?;
            // Streamed: DuckDB hands the rows over a chunk at a time instead of
            // holding them all. `step` (unlike the iterator) returns its errors.
            drop(statement.stream_arrow([]).map_err(err)?);
            while let Some(chunk) = statement.step().map_err(err)? {
                let batch = RecordBatch::from(&chunk);
                let int = |i: usize| as_i64(&batch, i);
                let text = |i: usize| as_text(&batch, i);
                let (t, z, line, image, class_id) = (int(0)?, int(1)?, text(2)?, text(3)?, int(4)?);
                let (n_images, n_objects, n_tracks) = (int(5)?, int(6)?, int(7)?);
                let stat_columns = (0..n_stats)
                    .map(|k| as_f64(&batch, 8 + k))
                    .collect::<Result<Vec<_>, _>>()?;
                for row in 0..batch.num_rows() {
                    for (value, column) in computed_values.iter_mut().zip(&stat_columns) {
                        *value = column.is_valid(row).then(|| column.value(row));
                    }
                    let t = t.value(row) as u32;
                    let class_id = class_id.value(row) as u32;
                    let line = line
                        .is_valid(row)
                        .then(|| line.value(row))
                        .unwrap_or_default();
                    // Overwritten in place: no allocation per row.
                    point.t = t;
                    point.z = z.value(row) as u32;
                    point.time_s = self
                        .head
                        .frame_interval_s
                        .map(|interval| f64::from(t) * interval);
                    point.line.clear();
                    match plan.lines {
                        TimeCourseLines::Class | TimeCourseLines::Image => {}
                        TimeCourseLines::Track => {
                            point.line.push('#');
                            point.line.push_str(line);
                        }
                        TimeCourseLines::Well | TimeCourseLines::Group => point.line.push_str(line),
                    }
                    point.image.clear();
                    if plan.with_image && image.is_valid(row) {
                        point.image.push_str(image.value(row));
                    }
                    if point.class_id != class_id || point.class_name.is_empty() {
                        point.class_id = class_id;
                        point.class_name.clear();
                        match plan.class_names.get(&class_id) {
                            Some(name) => point.class_name.push_str(name),
                            None => point.class_name.push_str(&class_id.to_string()),
                        }
                    }
                    point.n_images = n_images.value(row) as u32;
                    point.n_objects = n_objects.value(row) as u64;
                    point.n_tracks = n_tracks.value(row) as u64;
                    match &plan.lineage {
                        None => finish(&mut point, &computed_values)?,
                        Some(lineage) => {
                            match line.parse::<u64>().ok().and_then(|id| lineage.get(&id)) {
                                None => finish(&mut point, &computed_values)?,
                                Some(lines) => {
                                    for label in lines {
                                        let mut in_line = point.clone();
                                        in_line.line.clear();
                                        in_line.line.push_str(label);
                                        lined.push((in_line, computed_values.clone()));
                                    }
                                }
                            }
                        }
                    }
                }
            }
            // A query is a plane (or the whole sample): its family lines.
            lined.sort_by(|(a, _), (b, _)| {
                (a.t, a.z, &a.line, a.class_id).cmp(&(b.t, b.z, &b.line, b.class_id))
            });
            for (mut point, computed_values) in lined.drain(..) {
                finish(&mut point, &computed_values)?;
            }
        }
        Ok(())
    }

    /// Every point, kept.
    pub fn collect(self) -> Result<TimeCourseResult, InternalErrors> {
        let mut points = Vec::new();
        self.for_each(|point| {
            points.push(point.clone());
            Ok(())
        })?;
        Ok(TimeCourseResult {
            value_names: self.head.value_names,
            by_z: self.head.by_z,
            frame_interval_s: self.head.frame_interval_s,
            points,
        })
    }
}

fn column_err(i: usize, wanted: &str) -> InternalErrors {
    InternalErrors::Internal(format!("time course column {i} is not {wanted}"))
}

fn as_i64(batch: &RecordBatch, i: usize) -> Result<&Int64Array, InternalErrors> {
    batch
        .column(i)
        .as_any()
        .downcast_ref()
        .ok_or_else(|| column_err(i, "BIGINT"))
}

fn as_f64(batch: &RecordBatch, i: usize) -> Result<&Float64Array, InternalErrors> {
    batch
        .column(i)
        .as_any()
        .downcast_ref()
        .ok_or_else(|| column_err(i, "DOUBLE"))
}

fn as_text(batch: &RecordBatch, i: usize) -> Result<&StringArray, InternalErrors> {
    batch
        .column(i)
        .as_any()
        .downcast_ref()
        .ok_or_else(|| column_err(i, "VARCHAR"))
}

/// The time course `filter` asks for - see [`TimeCourseQuery`].
pub(super) fn time_course(
    database: &ResultsGenerator,
    filter: &TimeCourseFilter,
) -> Result<TimeCourseResult, InternalErrors> {
    TimeCourseQuery::prepare(database, filter)?.collect()
}

impl ResultsGenerator {
    /// The time course `filter` asks for - see [`TimeCourseQuery`].
    pub fn time_course(
        &self,
        filter: &TimeCourseFilter,
    ) -> Result<TimeCourseResult, InternalErrors> {
        time_course(self, filter)
    }
}

/// The frame interval the counted images share - `None` when one has none,
/// they differ, or the results are from before it was recorded.
fn frame_interval(database: &ResultsGenerator, image_where: &str) -> Option<f64> {
    let sql = format!(
        "SELECT COUNT(DISTINCT frame_interval_s), MIN(frame_interval_s), \
                COUNT(*) FILTER (WHERE frame_interval_s IS NULL) FROM images i WHERE {image_where}"
    );
    let (distinct, interval, missing): (i64, Option<f64>, i64) = database
        .connection()
        .query_row(&sql, [], |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)))
        .ok()?;
    (distinct == 1 && missing == 0)
        .then_some(interval)
        .flatten()
}

/// A repeatable sample of `n` of the tracks (matching `object_where`, on
/// `planes`) that no other track came from - where a line with its
/// parents' history ends. Those with children are left out here, not in
/// SQL: a `NOT IN` list of thousands is checked on every row (~180 ms on
/// 1.1M objects).
fn sample_line_ends(
    database: &ResultsGenerator,
    planes: &[PlaneFilter],
    object_where: &str,
    parents: &HashMap<u64, Vec<u64>>,
    n: usize,
) -> Result<Vec<u64>, InternalErrors> {
    let err = |e: duckdb::Error| InternalErrors::Io(e.to_string());
    let with_children: HashSet<u64> = parents.values().flatten().copied().collect();
    let planes = planes
        .iter()
        .map(|p| format!("({}, {})", p.t_stack, p.z_stack))
        .collect::<Vec<_>>()
        .join(", ");
    let sql = format!(
        "WITH p(t, z) AS (VALUES {planes}) \
         SELECT DISTINCT track_id FROM objects WHERE {object_where}"
    );
    let mut stmt = database.connection().prepare(&sql).map_err(err)?;
    let mut ends: Vec<u64> = stmt
        .query_map([], |row| row.get::<_, u64>(0))
        .map_err(err)?
        .filter(|id| id.as_ref().map_or(true, |id| !with_children.contains(id)))
        .collect::<Result<_, _>>()
        .map_err(err)?;
    // The same each time: the n with the lowest hash.
    let hash = |id: u64| {
        let mut x = id.wrapping_add(0x9E37_79B9_7F4A_7C15);
        x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        x ^ (x >> 31)
    };
    if ends.len() > n {
        ends.select_nth_unstable_by_key(n, |id| hash(*id));
        ends.truncate(n);
    }
    ends.sort_unstable();
    Ok(ends)
}

/// Track ids as "1, 2, 3" (sorted), for `IN (...)`.
fn id_list(ids: &HashSet<u64>) -> String {
    let mut ids: Vec<u64> = ids.iter().copied().collect();
    ids.sort_unstable();
    ids.iter()
        .map(u64::to_string)
        .collect::<Vec<_>>()
        .join(", ")
}

/// `[1, 2, 3]` as an `INTEGER[]` literal.
fn sql_u32_array(ids: &[u32]) -> String {
    format!(
        "[{}]::INTEGER[]",
        ids.iter()
            .map(u32::to_string)
            .collect::<Vec<_>>()
            .join(", ")
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::Grouping;
    use crate::backend::local::results::test_support::{ObjectSpec, seed_db};

    /// `objects`, with every image analysed on time points `0..times` (one
    /// z-plane), `frame_interval_s` apart.
    fn open(objects: &[ObjectSpec], times: i32, frame_interval_s: Option<f64>) -> ResultsGenerator {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("results.evadb");
        seed_db(&path, objects);
        {
            let conn = duckdb::Connection::open(&path).expect("open db");
            conn.execute_batch(
                "ALTER TABLE images ADD COLUMN t_stacks_analyzed INTEGER[];
                 ALTER TABLE images ADD COLUMN z_stacks_analyzed INTEGER[];
                 ALTER TABLE images ADD COLUMN frame_interval_s DOUBLE;",
            )
            .unwrap();
            conn.execute(
                "UPDATE images SET t_stacks = ?, t_stacks_analyzed = CAST(range(0, ?) AS INTEGER[]), \
                 z_stacks_analyzed = [0], frame_interval_s = ?",
                duckdb::params![times, times, frame_interval_s],
            )
            .unwrap();
        }
        std::mem::forget(dir);
        ResultsGenerator::open_database(path).expect("open database")
    }

    /// img1 and img2 at T0, img1 again at T1, nothing at T2.
    fn series() -> Vec<ObjectSpec> {
        vec![
            ObjectSpec::new("img1.tif", "ClassA", 1, 100),
            ObjectSpec::new("img2.tif", "ClassA", 1, 200),
            ObjectSpec::new("img1.tif", "ClassA", 1, 300).at_plane(0, 1),
        ]
    }

    fn filter(lines: TimeCourseLines, statistics: &[TimeCourseStat]) -> TimeCourseFilter {
        TimeCourseFilter {
            lines,
            grouping: Grouping::Auto,
            columns: vec![Column::AreaSizePx],
            statistics: statistics.to_vec(),
            normalize: TimeCourseNormalize::None,
            images: None,
            object_classes: None,
            t_stacks: None,
            z_stacks: None,
            max_tracks: None,
            history: TrackHistory::Own,
        }
    }

    /// (t, line or image, N images, N objects, values) per point.
    fn summary(result: &TimeCourseResult) -> Vec<(u32, String, u32, u64, Vec<Option<f64>>)> {
        result
            .points
            .iter()
            .map(|p| {
                let line = if p.line.is_empty() {
                    p.image.clone()
                } else {
                    p.line.clone()
                };
                (p.t, line, p.n_images, p.n_objects, p.values.clone())
            })
            .collect()
    }

    fn rounded(values: &[Option<f64>]) -> Vec<Option<f64>> {
        values
            .iter()
            .map(|v| v.map(|v| (v * 100.0).round() / 100.0))
            .collect()
    }

    /// A class line: every analysed time point - the empty one too, with
    /// N objects 0, sum 0 and no mean.
    #[test]
    fn a_class_line_has_every_analysed_time_point() {
        let db = open(&series(), 3, None);
        let stats = [
            TimeCourseStat::Mean,
            TimeCourseStat::Sd,
            TimeCourseStat::Sum,
        ];
        let result = time_course(&db, &filter(TimeCourseLines::Class, &stats)).unwrap();
        assert_eq!(
            result.value_names,
            ["Area [px] mean", "Area [px] SD", "Area [px] sum"]
        );
        let rows: Vec<_> = summary(&result)
            .into_iter()
            .map(|(t, line, n_images, n_objects, values)| {
                (t, line, n_images, n_objects, rounded(&values))
            })
            .collect();
        assert_eq!(
            rows,
            [
                (
                    0,
                    String::new(),
                    2,
                    2,
                    vec![Some(150.0), Some(70.71), Some(300.0)]
                ),
                (1, String::new(), 2, 1, vec![Some(300.0), None, Some(300.0)]),
                (2, String::new(), 2, 0, vec![None, None, Some(0.0)]),
            ]
        );
        assert!(result.points.iter().all(|p| p.class_name == "ClassA"));
    }

    /// Image lines: each image at each time point, with or without objects.
    #[test]
    fn image_lines_have_each_image_at_each_time_point() {
        let db = open(&series(), 2, None);
        let result = time_course(
            &db,
            &filter(TimeCourseLines::Image, &[TimeCourseStat::Mean]),
        )
        .unwrap();
        assert_eq!(
            summary(&result),
            [
                (0, "img1.tif".to_string(), 1, 1, vec![Some(100.0)]),
                (0, "img2.tif".to_string(), 1, 1, vec![Some(200.0)]),
                (1, "img1.tif".to_string(), 1, 1, vec![Some(300.0)]),
                (1, "img2.tif".to_string(), 1, 0, vec![None]),
            ]
        );
    }

    /// Wells from the image names: a well's images pooled.
    #[test]
    fn well_lines_pool_the_wells_images() {
        let db = open(
            &[
                ObjectSpec::new("A1_01.tif", "ClassA", 1, 10),
                ObjectSpec::new("A1_02.tif", "ClassA", 1, 30),
                ObjectSpec::new("B2_01.tif", "ClassA", 1, 100),
            ],
            1,
            None,
        );
        let result =
            time_course(&db, &filter(TimeCourseLines::Well, &[TimeCourseStat::Mean])).unwrap();
        assert_eq!(
            summary(&result),
            [
                (0, "A1".to_string(), 2, 2, vec![Some(20.0)]),
                (0, "B2".to_string(), 1, 1, vec![Some(100.0)]),
            ]
        );
    }

    /// Track lines: where the track has objects - untracked objects (track
    /// 0) are no track.
    /// 1 divides into 2 and 3 after T 1; 3 and 4 fuse into 5 at T 3; 6
    /// has no family. Areas: the track id times 10, plus the time point.
    fn family() -> Vec<ObjectSpec> {
        let at = |track: u64, t: i32| {
            ObjectSpec::new("img1.tif", "ClassA", 1, track * 10 + t as u64)
                .in_track(track)
                .at_plane(0, t)
        };
        vec![
            at(1, 0),
            at(1, 1),
            at(2, 2),
            at(3, 2),
            at(4, 2),
            at(5, 3),
            at(6, 0),
            at(6, 3),
        ]
    }

    fn open_family() -> ResultsGenerator {
        let db = open(&family(), 4, None);
        db.connection()
            .execute_batch(
                "ALTER TABLE objects ADD COLUMN parent_track_ids UBIGINT[];
                 UPDATE objects SET parent_track_ids = [1] WHERE track_id IN (2, 3);
                 UPDATE objects SET parent_track_ids = [3, 4] WHERE track_id = 5;",
            )
            .unwrap();
        db
    }

    /// (line, t, area) of each point.
    fn lines_of(result: &TimeCourseResult) -> Vec<(String, u32, Option<f64>)> {
        result
            .points
            .iter()
            .map(|p| (p.line.clone(), p.t, p.values[0]))
            .collect()
    }

    #[test]
    fn a_line_per_way_from_the_first_ancestor_to_a_track_without_children() {
        let parents = HashMap::from([(2, vec![1]), (3, vec![1]), (5, vec![3, 4])]);
        let lines = lineage_lines(&parents, None);
        assert_eq!(lines[&2], ["#2 · from #1"]);
        assert_eq!(
            lines[&1],
            ["#2 · from #1", "#5 · from #1 › #3"],
            "in every line through it"
        );
        assert_eq!(lines[&3], ["#5 · from #1 › #3"]);
        assert_eq!(lines[&4], ["#5 · from #4"]);
        assert_eq!(
            lines[&5],
            ["#5 · from #1 › #3", "#5 · from #4"],
            "a fusion: a line per parent"
        );

        let only = lineage_lines(&parents, Some(&HashSet::from([2])));
        assert_eq!(only[&1], ["#2 · from #1"]);
        assert!(!only.contains_key(&5));

        let looped = lineage_lines(&HashMap::from([(1, vec![2]), (2, vec![1])]), None);
        assert!(
            looped.is_empty(),
            "no track without children: no line, no hang"
        );
    }

    /// Own track (default): as without lineage. With parents' history:
    /// each line from the first ancestor on, in time order, a track's
    /// points in every line through it; a track without family its own.
    #[test]
    fn track_lines_with_their_parents_history() {
        let db = open_family();
        let own = db
            .time_course(&filter(TimeCourseLines::Track, &[TimeCourseStat::Mean]))
            .unwrap();
        let lines: HashSet<String> = own.points.iter().map(|p| p.line.clone()).collect();
        assert_eq!(lines.len(), 6, "a line per track: {lines:?}");

        let with = db
            .time_course(&TimeCourseFilter {
                history: TrackHistory::WithParents,
                ..filter(TimeCourseLines::Track, &[TimeCourseStat::Mean])
            })
            .unwrap();
        let line = |label: &str| -> Vec<(u32, Option<f64>)> {
            lines_of(&with)
                .into_iter()
                .filter(|(l, ..)| l == label)
                .map(|(_, t, v)| (t, v))
                .collect()
        };
        assert_eq!(
            line("#2 · from #1"),
            [(0, Some(10.0)), (1, Some(11.0)), (2, Some(22.0))]
        );
        assert_eq!(
            line("#5 · from #1 › #3"),
            [
                (0, Some(10.0)),
                (1, Some(11.0)),
                (2, Some(32.0)),
                (3, Some(53.0))
            ]
        );
        assert_eq!(line("#5 · from #4"), [(2, Some(42.0)), (3, Some(53.0))]);
        assert_eq!(line("#6"), [(0, Some(60.0)), (3, Some(63.0))]);
        assert!(line("#1").is_empty(), "a mother has no line of its own");
        let times: Vec<u32> = with.points.iter().map(|p| p.t).collect();
        assert!(times.is_sorted(), "time point by time point: {times:?}");
    }

    /// Relative to the first time point: the line's - its first ancestor's.
    #[test]
    fn a_line_with_history_is_relative_to_its_first_ancestor() {
        let with = open_family()
            .time_course(&TimeCourseFilter {
                history: TrackHistory::WithParents,
                normalize: TimeCourseNormalize::RelativeToStart,
                ..filter(TimeCourseLines::Track, &[TimeCourseStat::Mean])
            })
            .unwrap();
        let daughter = lines_of(&with)
            .into_iter()
            .find(|(l, t, _)| l == "#2 · from #1" && *t == 2)
            .unwrap();
        assert_eq!(
            rounded(&[daughter.2]),
            [Some(220.0)],
            "22 of the mother's 10"
        );
    }

    /// A sample of lines is whole lines: their ends sampled, the tracks
    /// before them added.
    #[test]
    fn a_sample_of_lines_with_history_has_whole_lines() {
        let with = open_family()
            .time_course(&TimeCourseFilter {
                history: TrackHistory::WithParents,
                max_tracks: Some(2),
                ..filter(TimeCourseLines::Track, &[TimeCourseStat::Mean])
            })
            .unwrap();
        let mut by_line: HashMap<String, Vec<u32>> = HashMap::new();
        for p in &with.points {
            by_line.entry(p.line.clone()).or_default().push(p.t);
        }
        assert!(!by_line.is_empty() && by_line.len() <= 3, "{by_line:?}");
        for (label, times) in &by_line {
            if label.contains("from #1") {
                assert_eq!(times[0], 0, "{label} starts with the mother: {times:?}");
            }
        }
    }

    /// Results without lineage: with parents' history is the same as own
    /// track - no other query.
    #[test]
    fn without_lineage_the_history_changes_nothing() {
        let db = open(&family(), 4, None);
        let own = db
            .time_course(&filter(TimeCourseLines::Track, &[TimeCourseStat::Mean]))
            .unwrap();
        let with = db
            .time_course(&TimeCourseFilter {
                history: TrackHistory::WithParents,
                ..filter(TimeCourseLines::Track, &[TimeCourseStat::Mean])
            })
            .unwrap();
        assert_eq!(own.points, with.points);
        let class = db
            .time_course(&TimeCourseFilter {
                history: TrackHistory::WithParents,
                ..filter(TimeCourseLines::Class, &[TimeCourseStat::Mean])
            })
            .unwrap();
        let class_own = db
            .time_course(&filter(TimeCourseLines::Class, &[TimeCourseStat::Mean]))
            .unwrap();
        assert_eq!(class.points, class_own.points, "class lines never change");
    }

    #[test]
    fn track_lines_follow_each_track() {
        let db = open(
            &[
                ObjectSpec::new("img1.tif", "ClassA", 1, 100).in_track(7),
                ObjectSpec::new("img1.tif", "ClassA", 1, 120)
                    .in_track(7)
                    .at_plane(0, 1),
                ObjectSpec::new("img1.tif", "ClassA", 1, 50)
                    .in_track(9)
                    .at_plane(0, 1),
                ObjectSpec::new("img1.tif", "ClassA", 1, 999),
            ],
            2,
            None,
        );
        let result = time_course(
            &db,
            &filter(TimeCourseLines::Track, &[TimeCourseStat::Mean]),
        )
        .unwrap();
        assert_eq!(
            summary(&result),
            [
                (0, "#7".to_string(), 1, 1, vec![Some(100.0)]),
                (1, "#7".to_string(), 1, 1, vec![Some(120.0)]),
                (1, "#9".to_string(), 1, 1, vec![Some(50.0)]),
            ]
        );
        assert!(
            result
                .points
                .iter()
                .all(|p| p.image == "img1.tif" && p.n_tracks == 1)
        );
    }

    /// Every track is read a plane at a time: normalized to its own first
    /// time point across those reads, numbered tracks in numeric order.
    #[test]
    fn track_lines_normalize_to_each_tracks_start() {
        let db = open(
            &[
                ObjectSpec::new("img1.tif", "ClassA", 1, 100).in_track(10),
                ObjectSpec::new("img1.tif", "ClassA", 1, 150)
                    .in_track(10)
                    .at_plane(0, 1),
                ObjectSpec::new("img1.tif", "ClassA", 1, 40)
                    .in_track(9)
                    .at_plane(0, 1),
                ObjectSpec::new("img1.tif", "ClassA", 1, 60)
                    .in_track(9)
                    .at_plane(0, 2),
            ],
            3,
            None,
        );
        let result = time_course(
            &db,
            &TimeCourseFilter {
                normalize: TimeCourseNormalize::RelativeToStart,
                ..filter(TimeCourseLines::Track, &[TimeCourseStat::Mean])
            },
        )
        .unwrap();
        assert_eq!(
            summary(&result),
            [
                (0, "#10".to_string(), 1, 1, vec![Some(100.0)]),
                (1, "#9".to_string(), 1, 1, vec![Some(100.0)]),
                (1, "#10".to_string(), 1, 1, vec![Some(150.0)]),
                (2, "#9".to_string(), 1, 1, vec![Some(150.0)]),
            ]
        );
    }

    /// At most `max_tracks` tracks: a sample, the same every time.
    #[test]
    fn track_lines_can_be_a_repeatable_sample() {
        let objects: Vec<ObjectSpec> = (1..=20)
            .flat_map(|track| {
                [0, 1].map(|t| {
                    ObjectSpec::new("img1.tif", "ClassA", 1, 100)
                        .in_track(track)
                        .at_plane(0, t)
                })
            })
            .collect();
        let db = open(&objects, 2, None);
        let tracks = |max_tracks| {
            let mut sample = filter(TimeCourseLines::Track, &[TimeCourseStat::Mean]);
            sample.max_tracks = max_tracks;
            let result = time_course(&db, &sample).unwrap();
            let mut lines: Vec<String> = result.points.iter().map(|p| p.line.clone()).collect();
            lines.dedup();
            lines.sort();
            lines.dedup();
            (lines, result.points.len())
        };
        assert_eq!(tracks(None).0.len(), 20);
        let (sample, points) = tracks(Some(5));
        assert_eq!(sample.len(), 5);
        assert_eq!(points, 10, "every time point of the sampled tracks");
        assert_eq!(tracks(Some(5)).0, sample, "the same sample");
    }

    /// Relative to the start: the line's first time point with objects is
    /// 100 %; SD in % of that mean; an empty point has none.
    #[test]
    fn normalized_values_are_percent_of_the_start() {
        let db = open(&series(), 3, None);
        let mut normalized = filter(
            TimeCourseLines::Class,
            &[TimeCourseStat::Mean, TimeCourseStat::Sd],
        );
        normalized.normalize = TimeCourseNormalize::RelativeToStart;
        let result = time_course(&db, &normalized).unwrap();
        assert_eq!(result.value_names[0], "Area [px] mean [% of start]");
        let values: Vec<_> = result.points.iter().map(|p| rounded(&p.values)).collect();
        assert_eq!(
            values,
            [
                vec![Some(100.0), Some(47.14)],
                vec![Some(200.0), None],
                vec![None, None]
            ]
        );
    }

    /// The frame interval the images share gives each point its time.
    #[test]
    fn a_shared_frame_interval_gives_the_time() {
        let db = open(&series(), 3, Some(30.0));
        let result = time_course(
            &db,
            &filter(TimeCourseLines::Class, &[TimeCourseStat::Mean]),
        )
        .unwrap();
        assert_eq!(result.frame_interval_s, Some(30.0));
        let times: Vec<_> = result.points.iter().map(|p| p.time_s).collect();
        assert_eq!(times, [Some(0.0), Some(30.0), Some(60.0)]);

        let unknown = time_course(
            &open(&series(), 3, None),
            &filter(TimeCourseLines::Class, &[TimeCourseStat::Mean]),
        )
        .unwrap();
        assert_eq!(unknown.frame_interval_s, None);
        assert!(unknown.points.iter().all(|p| p.time_s.is_none()));
    }

    /// Median and quartiles, and a time point range.
    #[test]
    fn quartiles_and_a_time_point_range() {
        let db = open(
            &[
                ObjectSpec::new("img1.tif", "ClassA", 1, 10),
                ObjectSpec::new("img1.tif", "ClassA", 1, 20),
                ObjectSpec::new("img1.tif", "ClassA", 1, 30),
                ObjectSpec::new("img1.tif", "ClassA", 1, 40),
                ObjectSpec::new("img1.tif", "ClassA", 1, 50),
                ObjectSpec::new("img1.tif", "ClassA", 1, 7).at_plane(0, 1),
            ],
            2,
            None,
        );
        let mut quartiles = filter(
            TimeCourseLines::Class,
            &[
                TimeCourseStat::Q1,
                TimeCourseStat::Median,
                TimeCourseStat::Q3,
            ],
        );
        quartiles.t_stacks = Some(std::range::Range { start: 0, end: 1 });
        let result = time_course(&db, &quartiles).unwrap();
        assert_eq!(result.points.len(), 1, "only T0");
        assert_eq!(
            result.points[0].values,
            [Some(20.0), Some(30.0), Some(40.0)]
        );
    }
}
