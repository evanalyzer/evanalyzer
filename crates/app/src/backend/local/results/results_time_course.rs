//! The time course: per time point (and z-plane), statistics of values over
//! each line's objects - a class's, an image's, a well's, a group's or a
//! track's. What the "Over time" chart draws and its export writes.

use super::results_generator::{
    ResultsGenerator, class_blocks, class_filter_sql, column_aggregate_expr,
    objects_per_class_sql, sql_string_in_list,
};
use crate::api::{
    Column, PlaneFilter, TimeCourseFilter, TimeCourseLines, TimeCourseNormalize, TimeCoursePoint,
    TimeCourseResult, TimeCourseStat,
};
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

/// The time course `filter` asks for. Every time point the run analysed
/// (within `filter`'s ranges) has a point per line and class - with no
/// objects there, N objects 0, Sum 0 and no other value. Track lines only
/// have points where the track has an object. Failed and disabled images
/// are left out.
pub(super) fn time_course(
    database: &ResultsGenerator,
    filter: &TimeCourseFilter,
) -> Result<TimeCourseResult, InternalErrors> {
    let err = |e: duckdb::Error| InternalErrors::Io(e.to_string());
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
    let empty = |frame_interval_s| TimeCourseResult {
        value_names: value_names.clone(),
        by_z,
        frame_interval_s,
        points: Vec::new(),
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
        return Ok(empty(None));
    }

    // The images counted: analysed (not failed), not left out, of a well /
    // group when the lines are those.
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
        image_conditions.push(format!("i.image_rel_path IN ({})", sql_string_in_list(paths)));
        object_conditions.push(format!("image_rel_path IN ({})", sql_string_in_list(paths)));
    }
    if matches!(filter.lines, TimeCourseLines::Well | TimeCourseLines::Group) {
        image_conditions.push(format!("{line_of_image} <> ''"));
    }
    let image_where = image_conditions.join(" AND ");
    let frame_interval_s = frame_interval(database, &image_where);

    let mut class_post = vec!["true".to_string()];
    if let Some(ids) = &selected_ids {
        object_conditions.push(class_filter_sql("object_class_id", ids));
        class_post.push(format!(
            "class_id IN ({})",
            ids.iter().map(u32::to_string).collect::<Vec<_>>().join(", ")
        ));
    }
    if filter.lines == TimeCourseLines::Track {
        object_conditions.push("COALESCE(track_id, 0) <> 0".to_string());
    }
    let mut row_classes = class_blocks(selected_ids.as_deref(), &classes);
    if selected_ids.is_none() {
        row_classes.retain(|id| ObjectClass::Global(*id) != ObjectClass::BACKGROUND);
    }

    // The statistics computed: the asked-for ones and, normalized, what
    // they're divided by.
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
    let stats_sql = (0..columns.len())
        .flat_map(|i| {
            computed
                .iter()
                .enumerate()
                .map(move |(k, stat)| format!("{} AS s{i}_{k}", stat_sql(*stat, &format!("obj.v{i}"))))
        })
        .collect::<Vec<_>>()
        .join(", ");
    let stat_names = (0..columns.len())
        .flat_map(|i| (0..computed.len()).map(move |k| format!("a.s{i}_{k}")))
        .collect::<Vec<_>>()
        .join(", ");

    let planes_sql = planes
        .iter()
        .map(|p| format!("({}, {})", p.t_stack, p.z_stack))
        .collect::<Vec<_>>()
        .join(", ");
    let measured = match database.analysed_planes_recorded() {
        true => {
            "list_contains(img.t_stacks_analyzed, p.t) AND list_contains(img.z_stacks_analyzed, p.z)"
        }
        // Results from before the run recorded its planes: within the
        // image's own sizes (the planes are the ones with objects).
        false => "p.z < img.z_stacks AND p.t < img.t_stacks",
    };
    let line_of_object = match filter.lines {
        TimeCourseLines::Track => "CAST(obj.track_id AS VARCHAR)",
        _ => "img.line",
    };
    let objects = objects_per_class_sql(&object_conditions.join(" AND "));
    let common = format!(
        "WITH p(t, z) AS (VALUES {planes_sql}),\n\
         img AS (SELECT i.*, {line_of_image} AS line FROM images i WHERE {image_where}),\n\
         obj AS (SELECT o.t_stack AS t, o.z_stack AS z, o.image_rel_path, o.class_id, NULLIF(o.track_id, 0) AS track_id, \
                 {values_sql} FROM ({objects}) o WHERE {}),\n\
         a AS (SELECT obj.t, obj.z, {line_of_object} AS line, obj.class_id, \
                 MIN(img.image_name) AS image, COUNT(DISTINCT obj.image_rel_path) AS n_images, \
                 COUNT(*) AS n_objects, COUNT(DISTINCT obj.track_id) AS n_tracks, {stats_sql} \
               FROM obj JOIN img USING (image_rel_path) GROUP BY 1, 2, 3, 4)",
        class_post.join(" AND ")
    );
    let sql = match filter.lines {
        // A track is where its objects are.
        TimeCourseLines::Track => format!(
            "{common}\n\
             SELECT a.t, a.z, a.line, a.image, a.class_id, a.n_images, a.n_objects, a.n_tracks, \
                    {stat_names}\n\
             FROM a ORDER BY 1, 2, TRY_CAST(a.line AS BIGINT), 5"
        ),
        // Every analysed plane x line x class - also without objects.
        _ => format!(
            "{common},\n\
             m AS (SELECT p.t, p.z, img.line, MIN(img.image_name) AS image, COUNT(*) AS n_images \
                   FROM p, img WHERE {measured} \
                   GROUP BY 1, 2, 3),\n\
             f AS (SELECT m.*, c.class_id FROM m, (SELECT UNNEST({}) AS class_id) c)\n\
             SELECT COALESCE(f.t, a.t), COALESCE(f.z, a.z), COALESCE(f.line, a.line), \
                    COALESCE(f.image, a.image), COALESCE(f.class_id, a.class_id), \
                    COALESCE(f.n_images, a.n_images), COALESCE(a.n_objects, 0), \
                    COALESCE(a.n_tracks, 0), {stat_names}\n\
             FROM f FULL JOIN a ON f.t = a.t AND f.z = a.z AND f.line = a.line \
                 AND f.class_id = a.class_id\n\
             ORDER BY 1, 2, 3, 5",
            sql_u32_array(&row_classes)
        ),
    };

    let class_names: HashMap<u32, String> =
        classes.iter().map(|c| (c.id, c.name.clone())).collect();
    let with_image = matches!(filter.lines, TimeCourseLines::Image | TimeCourseLines::Track);
    let n_stats = columns.len() * computed.len();
    let mut statement = database.connection().prepare(&sql).map_err(err)?;
    let raw: Vec<(TimeCoursePoint, Vec<Option<f64>>)> = statement
        .query_map([], |row| {
            let class_id = row.get::<_, i64>(4)? as u32;
            let line: String = row.get::<_, Option<String>>(2)?.unwrap_or_default();
            let image: String = row.get::<_, Option<String>>(3)?.unwrap_or_default();
            let t = row.get::<_, i64>(0)? as u32;
            let computed_values = (0..n_stats)
                .map(|k| row.get::<_, Option<f64>>(8 + k))
                .collect::<Result<Vec<_>, _>>()?;
            Ok((
                TimeCoursePoint {
                    t,
                    z: row.get::<_, i64>(1)? as u32,
                    time_s: frame_interval_s.map(|interval| f64::from(t) * interval),
                    line: match filter.lines {
                        TimeCourseLines::Class | TimeCourseLines::Image => String::new(),
                        TimeCourseLines::Track => format!("#{line}"),
                        TimeCourseLines::Well | TimeCourseLines::Group => line,
                    },
                    image: if with_image { image } else { String::new() },
                    class_id,
                    class_name: class_names
                        .get(&class_id)
                        .cloned()
                        .unwrap_or_else(|| class_id.to_string()),
                    n_images: row.get::<_, i64>(5)? as u32,
                    n_objects: row.get::<_, i64>(6)? as u64,
                    n_tracks: row.get::<_, i64>(7)? as u64,
                    values: Vec::new(),
                },
                computed_values,
            ))
        })
        .map_err(err)?
        .collect::<Result<Vec<_>, _>>()
        .map_err(err)?;

    // Without objects a sum is 0, every other statistic none.
    let value_of = |point: &TimeCoursePoint, computed_values: &[Option<f64>], i: usize, stat: TimeCourseStat| {
        let k = computed.iter().position(|s| *s == stat).expect("computed");
        match (point.n_objects, stat) {
            (0, TimeCourseStat::Sum) => Some(0.0),
            _ => computed_values[i * computed.len() + k],
        }
    };
    // Normalized: each line's first time point with objects is 100 %.
    let mut starts: HashMap<(u32, String, String, u32), Vec<Option<f64>>> = HashMap::new();
    if normalized {
        for (point, computed_values) in &raw {
            if point.n_objects > 0 {
                starts
                    .entry((point.z, point.line.clone(), point.image.clone(), point.class_id))
                    .or_insert_with(|| computed_values.clone());
            }
        }
    }
    let points = raw
        .iter()
        .map(|(point, computed_values)| {
            let start = starts.get(&(point.z, point.line.clone(), point.image.clone(), point.class_id));
            let values = (0..columns.len())
                .flat_map(|i| stats.iter().map(move |stat| (i, *stat)))
                .map(|(i, stat)| {
                    let value = value_of(point, computed_values, i, stat);
                    match (normalized, start) {
                        (false, _) => value,
                        (true, None) => None,
                        (true, Some(start)) => {
                            let base = value_of(point, start, i, normalization_base(stat));
                            match (value, base) {
                                (Some(value), Some(base)) if base != 0.0 => Some(value / base * 100.0),
                                _ => None,
                            }
                        }
                    }
                })
                .collect();
            TimeCoursePoint {
                values,
                ..point.clone()
            }
        })
        .collect();

    Ok(TimeCourseResult {
        value_names,
        by_z,
        frame_interval_s,
        points,
    })
}

impl ResultsGenerator {
    /// The time course `filter` asks for - see [`time_course`].
    pub fn time_course(&self, filter: &TimeCourseFilter) -> Result<TimeCourseResult, InternalErrors> {
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
    (distinct == 1 && missing == 0).then_some(interval).flatten()
}

/// `[1, 2, 3]` as an `INTEGER[]` literal.
fn sql_u32_array(ids: &[u32]) -> String {
    format!(
        "[{}]::INTEGER[]",
        ids.iter().map(u32::to_string).collect::<Vec<_>>().join(", ")
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
        }
    }

    /// (t, line or image, N images, N objects, values) per point.
    fn summary(result: &TimeCourseResult) -> Vec<(u32, String, u32, u64, Vec<Option<f64>>)> {
        result
            .points
            .iter()
            .map(|p| {
                let line = if p.line.is_empty() { p.image.clone() } else { p.line.clone() };
                (p.t, line, p.n_images, p.n_objects, p.values.clone())
            })
            .collect()
    }

    fn rounded(values: &[Option<f64>]) -> Vec<Option<f64>> {
        values.iter().map(|v| v.map(|v| (v * 100.0).round() / 100.0)).collect()
    }

    /// A class line: every analysed time point - the empty one too, with
    /// N objects 0, sum 0 and no mean.
    #[test]
    fn a_class_line_has_every_analysed_time_point() {
        let db = open(&series(), 3, None);
        let stats = [TimeCourseStat::Mean, TimeCourseStat::Sd, TimeCourseStat::Sum];
        let result = time_course(&db, &filter(TimeCourseLines::Class, &stats)).unwrap();
        assert_eq!(result.value_names, ["Area [px] mean", "Area [px] SD", "Area [px] sum"]);
        let rows: Vec<_> = summary(&result)
            .into_iter()
            .map(|(t, line, n_images, n_objects, values)| (t, line, n_images, n_objects, rounded(&values)))
            .collect();
        assert_eq!(
            rows,
            [
                (0, String::new(), 2, 2, vec![Some(150.0), Some(70.71), Some(300.0)]),
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
        let result = time_course(&db, &filter(TimeCourseLines::Image, &[TimeCourseStat::Mean])).unwrap();
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
        let result = time_course(&db, &filter(TimeCourseLines::Well, &[TimeCourseStat::Mean])).unwrap();
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
                ObjectSpec::new("img1.tif", "ClassA", 1, 120).in_track(7).at_plane(0, 1),
                ObjectSpec::new("img1.tif", "ClassA", 1, 50).in_track(9).at_plane(0, 1),
                ObjectSpec::new("img1.tif", "ClassA", 1, 999),
            ],
            2,
            None,
        );
        let result = time_course(&db, &filter(TimeCourseLines::Track, &[TimeCourseStat::Mean])).unwrap();
        assert_eq!(
            summary(&result),
            [
                (0, "#7".to_string(), 1, 1, vec![Some(100.0)]),
                (1, "#7".to_string(), 1, 1, vec![Some(120.0)]),
                (1, "#9".to_string(), 1, 1, vec![Some(50.0)]),
            ]
        );
        assert!(result.points.iter().all(|p| p.image == "img1.tif" && p.n_tracks == 1));
    }

    /// Relative to the start: the line's first time point with objects is
    /// 100 %; SD in % of that mean; an empty point has none.
    #[test]
    fn normalized_values_are_percent_of_the_start() {
        let db = open(&series(), 3, None);
        let mut normalized = filter(TimeCourseLines::Class, &[TimeCourseStat::Mean, TimeCourseStat::Sd]);
        normalized.normalize = TimeCourseNormalize::RelativeToStart;
        let result = time_course(&db, &normalized).unwrap();
        assert_eq!(result.value_names[0], "Area [px] mean [% of start]");
        let values: Vec<_> = result.points.iter().map(|p| rounded(&p.values)).collect();
        assert_eq!(
            values,
            [vec![Some(100.0), Some(47.14)], vec![Some(200.0), None], vec![None, None]]
        );
    }

    /// The frame interval the images share gives each point its time.
    #[test]
    fn a_shared_frame_interval_gives_the_time() {
        let db = open(&series(), 3, Some(30.0));
        let result = time_course(&db, &filter(TimeCourseLines::Class, &[TimeCourseStat::Mean])).unwrap();
        assert_eq!(result.frame_interval_s, Some(30.0));
        let times: Vec<_> = result.points.iter().map(|p| p.time_s).collect();
        assert_eq!(times, [Some(0.0), Some(30.0), Some(60.0)]);

        let unknown = time_course(&open(&series(), 3, None), &filter(TimeCourseLines::Class, &[TimeCourseStat::Mean])).unwrap();
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
            &[TimeCourseStat::Q1, TimeCourseStat::Median, TimeCourseStat::Q3],
        );
        quartiles.t_stacks = Some(std::range::Range { start: 0, end: 1 });
        let result = time_course(&db, &quartiles).unwrap();
        assert_eq!(result.points.len(), 1, "only T0");
        assert_eq!(result.points[0].values, [Some(20.0), Some(30.0), Some(40.0)]);
    }
}
