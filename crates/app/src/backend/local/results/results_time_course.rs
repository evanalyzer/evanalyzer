//! The time course: per time point (and z-plane), statistics of values over
//! each line's objects - a class's, an image's, a well's, a group's or a
//! track's. What the "Over time" chart draws and its export writes.

use super::results_generator::{
    ResultsGenerator, class_blocks, class_filter_sql, column_aggregate_expr, objects_per_class_sql,
    sql_string_in_list,
};
use crate::api::{
    Column, PlaneFilter, TimeCourseFilter, TimeCourseLines, TimeCourseNormalize, TimeCoursePoint,
    TimeCourseResult, TimeCourseStat,
};
use duckdb::arrow::array::{Array, Float64Array, Int64Array, StringArray};
use duckdb::arrow::record_batch::RecordBatch;
use evanalyzer_cfg::core_types::{InternalErrors, ObjectClass};
use std::collections::HashMap;

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

fn stat_sql(stat: TimeCourseStat, value: &str) -> String {
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
            .filter(|column| column.is_aggregable() && **column != Column::Count)
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
        if filter.lines == TimeCourseLines::Track {
            object_conditions.push("COALESCE(track_id, 0) <> 0".to_string());
            // A sample of the tracks - the same each time - picked in the
            // database: with many tracks, reading every one only to draw a few
            // hundred takes seconds.
            if let Some(n) = filter.max_tracks {
                object_conditions.push(format!(
                    "track_id IN (SELECT track_id FROM (SELECT DISTINCT track_id FROM objects WHERE {}) \
                     USING SAMPLE reservoir({n} ROWS) REPEATABLE (7))",
                    object_conditions.join(" AND ")
                ));
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
            }),
        })
    }

    /// Each point, in order: time point by time point, then z-plane, line and
    /// class. Read from the database a chunk at a time.
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
                    let start = match plan.normalized {
                        false => None,
                        true => {
                            let key = (point.z, point.line.clone(), point.image.clone(), class_id);
                            if point.n_objects > 0 && !starts.contains_key(&key) {
                                starts.insert(key.clone(), computed_values.clone());
                            }
                            starts.get(&key)
                        }
                    };
                    point.values.clear();
                    point.values.extend(
                        (0..plan.n_columns)
                            .flat_map(|i| plan.stats.iter().map(move |stat| (i, *stat)))
                            .map(|(i, stat)| {
                                let value = value_of(point.n_objects, &computed_values, i, stat);
                                match (plan.normalized, start) {
                                    (false, _) => value,
                                    (true, None) => None,
                                    (true, Some(start)) => {
                                        let base = value_of(
                                            point.n_objects,
                                            start,
                                            i,
                                            normalization_base(stat),
                                        );
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
                    visit(&point)?;
                }
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
