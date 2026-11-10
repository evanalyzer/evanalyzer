//! # results_profile
//!
//! Profile measurements in the results (results format 2): a value per
//! object and position, e.g. Measure Object Sholl's intersections per
//! ring radius. What Charts › Profile draws: per line (a class, an image, a
//! well, all objects) and position the statistics over the line's objects,
//! and one object's own curve.
//!
//! An object without a value at a position of its line counts as 0 there:
//! a neuron that doesn't reach a radius crosses it 0 times. The rows are
//! filled up in the database (each object × each position of its line), so
//! median and quartiles count those zeros too, not only the mean.
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-11-09
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.

use super::results_generator::{ResultsGenerator, class_filter_sql, sql_string_in_list};
use crate::api::{
    MeasurementColumn, ProfileFilter, ProfileLines, ProfileObject, ProfilePoint, ProfileResult,
    ProfileStep, ProfileStepFilter,
};
use evanalyzer_cfg::core_types::{InternalErrors, ObjectClass};
use std::collections::HashMap;

fn err(e: duckdb::Error) -> InternalErrors {
    InternalErrors::Io(e.to_string())
}

/// The profile measurements of the results, by step and name - none in
/// results of format 1.
pub(super) fn profile_columns(
    db: &ResultsGenerator,
) -> Result<Vec<MeasurementColumn>, InternalErrors> {
    if !db.has_profiles()? {
        return Ok(Vec::new());
    }
    let mut stmt = db
        .connection()
        .prepare(
            "SELECT key, step, measurement FROM measurement_info \
             WHERE coalesce(profile, false) ORDER BY step, measurement, key",
        )
        .map_err(err)?;
    let columns = stmt
        .query_map([], |row| {
            Ok(MeasurementColumn {
                key: row.get(0)?,
                step: row.get(1)?,
                name: row.get(2)?,
                per_object: true,
            })
        })
        .map_err(err)?
        .collect::<Result<Vec<_>, _>>()
        .map_err(err)?;
    Ok(columns)
}

/// One object's profile: (position, value), by position.
pub(super) fn object_profile(
    db: &ResultsGenerator,
    key: &str,
    object_id: &str,
) -> Result<Vec<(f64, f64)>, InternalErrors> {
    if !db.has_profiles()? {
        return Ok(Vec::new());
    }
    let mut stmt = db
        .connection()
        .prepare(
            "SELECT position, coalesce(value, 0) FROM measurements \
             WHERE key = ? AND object_id = TRY_CAST(? AS UUID) AND position IS NOT NULL \
             ORDER BY position",
        )
        .map_err(err)?;
    let rows = stmt
        .query_map([key, object_id], |row| Ok((row.get(0)?, row.get(1)?)))
        .map_err(err)?
        .collect::<Result<Vec<_>, _>>()
        .map_err(err)?;
    Ok(rows)
}

/// The objects with the profile `filter.key` (of its classes, on the
/// enabled images), numbered in the order ‹ › steps through them: by
/// image, then top to bottom, left to right. Columns `n, id, image, centre,
/// x0, y0, x1, y1`.
fn profile_objects_sql(filter: &ProfileStepFilter) -> String {
    let mut conditions = vec![
        format!("m.key = '{}'", filter.key.replace('\'', "''")),
        "m.region = 'object'".to_string(),
        "m.position IS NOT NULL".to_string(),
        "NOT i.disabled".to_string(),
        "i.successful".to_string(),
    ];
    if let Some(classes) = &filter.object_classes {
        let ids: Vec<u32> = classes
            .iter()
            .filter_map(|class| match class {
                ObjectClass::Global(n) => Some(*n),
                _ => None,
            })
            .collect();
        conditions.push(match ids.is_empty() {
            true => "false".to_string(),
            false => class_filter_sql("m.object_class_id", &ids),
        });
    }
    format!(
        "WITH p AS (
            SELECT m.object_id, any_value(m.image_rel_path) AS image,
                any_value(m.related_object_id) AS centre
            FROM measurements m JOIN images i ON i.image_rel_path = m.image_rel_path
            WHERE {conditions}
            GROUP BY m.object_id
         ),
         b AS (
            SELECT object_id, min(bbox_xmin_px) AS x0, min(bbox_ymin_px) AS y0,
                max(bbox_xmax_px) AS x1, max(bbox_ymax_px) AS y1
            FROM objects WHERE object_id IN (SELECT object_id FROM p)
            GROUP BY object_id
         )
         SELECT row_number() OVER (ORDER BY p.image, b.y0, b.x0, p.object_id) AS n,
             p.object_id::VARCHAR AS id, p.image, p.centre::VARCHAR AS centre,
             b.x0, b.y0, b.x1, b.y1
         FROM p LEFT JOIN b ON b.object_id = p.object_id",
        conditions = conditions.join(" AND "),
    )
}

/// A step through the objects with a profile (`ProfileStepFilter`): none
/// in results of format 1.
pub(super) fn profile_step(
    db: &ResultsGenerator,
    filter: &ProfileStepFilter,
) -> Result<ProfileStep, InternalErrors> {
    if !db.has_profiles()? || filter.key.is_empty() {
        return Ok(ProfileStep::default());
    }
    let objects = profile_objects_sql(filter);
    let (total, current): (i64, Option<i64>) = db
        .connection()
        .query_row(
            &format!("SELECT count(*), max(CASE WHEN id = ? THEN n END) FROM ({objects})"),
            [filter.from.as_deref().unwrap_or("")],
            |row| Ok((row.get(0)?, row.get(1)?)),
        )
        .map_err(err)?;
    let target = match (current, filter.delta) {
        _ if total == 0 => None,
        (Some(n), delta) => Some((n + delta as i64).clamp(1, total)),
        (None, 0) => None,
        (None, delta) if delta < 0 => Some(total),
        (None, _) => Some(1),
    };
    let Some(target) = target else {
        return Ok(ProfileStep {
            position: 0,
            total: total as usize,
            object: None,
        });
    };
    let object = db
        .connection()
        .query_row(
            &format!("SELECT id, image, centre, x0, y0, x1, y1 FROM ({objects}) WHERE n = ?"),
            [target],
            |row| {
                let bbox: [Option<u32>; 4] = [row.get(3)?, row.get(4)?, row.get(5)?, row.get(6)?];
                Ok(ProfileObject {
                    object_id: row.get(0)?,
                    image_rel_path: row.get(1)?,
                    centre_id: row.get(2)?,
                    bbox_px: match bbox {
                        [Some(x0), Some(y0), Some(x1), Some(y1)] => Some([x0, y0, x1, y1]),
                        _ => None,
                    },
                })
            },
        )
        .map_err(err)?;
    Ok(ProfileStep {
        position: target as usize,
        total: total as usize,
        object: Some(object),
    })
}

/// The SQL of a profile chart's lines (see the module's doc): its columns
/// `line, class_id, position, n, mean, sd, median, q1, q3`, by line.
fn profile_sql(db: &ResultsGenerator, filter: &ProfileFilter) -> Result<String, InternalErrors> {
    let grouping = db.grouping_sql(&filter.grouping)?;
    let line = match filter.lines {
        ProfileLines::Class | ProfileLines::All => "''".to_string(),
        ProfileLines::Image => "m.image_rel_path".to_string(),
        ProfileLines::Well => format!(
            "regexp_extract(i.{}, '{}', 1)",
            grouping.column, grouping.regex
        ),
    };
    let mut conditions = vec![
        format!("m.key = '{}'", filter.key.replace('\'', "''")),
        "m.region = 'object'".to_string(),
        "m.position IS NOT NULL".to_string(),
        "NOT i.disabled".to_string(),
        "i.successful".to_string(),
    ];
    if let Some(images) = &filter.images {
        conditions.push(format!(
            "m.image_rel_path IN ({})",
            sql_string_in_list(images)
        ));
    }
    if filter.lines == ProfileLines::Well {
        conditions.push(format!("{line} <> ''"));
    }
    let selected: Option<Vec<u32>> = filter.object_classes.as_ref().map(|classes| {
        classes
            .iter()
            .filter_map(|class| match class {
                ObjectClass::Global(n) => Some(*n),
                _ => None,
            })
            .collect()
    });
    // Class lines: a line per class of the object (an object of two
    // classes is in both); else one class id 0 per row.
    let (class_column, unnest) = match filter.lines {
        ProfileLines::Class => ("c.class_id", ", unnest(m.object_class_id) AS c(class_id)"),
        _ => ("0", ""),
    };
    if let Some(ids) = &selected {
        conditions.push(class_filter_sql("m.object_class_id", ids));
    }
    let mut class_conditions = vec!["true".to_string()];
    if filter.lines == ProfileLines::Class {
        match &selected {
            Some(ids) => class_conditions.push(format!(
                "class_id IN ({})",
                ids.iter()
                    .map(u32::to_string)
                    .collect::<Vec<_>>()
                    .join(", ")
            )),
            // The background class is no line of its own.
            None => class_conditions.push(format!(
                "class_id <> {}",
                match ObjectClass::BACKGROUND {
                    ObjectClass::Global(n) => n,
                    _ => 0,
                }
            )),
        }
    }
    Ok(format!(
        "WITH v AS (
            SELECT {line} AS line, {class_column} AS class_id, m.object_id, m.position, m.value
            FROM measurements m JOIN images i ON i.image_rel_path = m.image_rel_path {unnest}
            WHERE {conditions}
         ),
         v2 AS (SELECT * FROM v WHERE {class_conditions}),
         objects AS (SELECT DISTINCT line, class_id, object_id FROM v2),
         positions AS (SELECT DISTINCT line, class_id, position FROM v2),
         filled AS (
            SELECT o.line, o.class_id, p.position, coalesce(v2.value, 0) AS value
            FROM objects o
            JOIN positions p ON p.line = o.line AND p.class_id = o.class_id
            LEFT JOIN v2 ON v2.line = o.line AND v2.class_id = o.class_id
                AND v2.object_id = o.object_id AND v2.position = p.position
         )
         SELECT line, class_id, position, count(*) AS n, avg(value), stddev_samp(value),
             quantile_cont(value, 0.5), quantile_cont(value, 0.25), quantile_cont(value, 0.75)
         FROM filled GROUP BY line, class_id, position
         ORDER BY class_id, line, position",
        conditions = conditions.join(" AND "),
        class_conditions = class_conditions.join(" AND "),
    ))
}

/// A profile chart's lines (`ProfileFilter`): at most `max_lines`, the
/// first by class and name; none in results of format 1.
pub(super) fn profile(
    db: &ResultsGenerator,
    filter: &ProfileFilter,
) -> Result<ProfileResult, InternalErrors> {
    if !db.has_profiles()?
        || filter.key.is_empty()
        || filter.images.as_ref().is_some_and(Vec::is_empty)
        || filter.object_classes.as_ref().is_some_and(Vec::is_empty)
    {
        return Ok(ProfileResult::default());
    }
    let names: HashMap<u32, String> = db
        .get_object_classes()?
        .into_iter()
        .map(|class| (class.id, class.name))
        .collect();
    let sql = profile_sql(db, filter)?;
    let mut stmt = db.connection().prepare(&sql).map_err(err)?;
    let rows = stmt
        .query_map([], |row| {
            let class_id: i64 = row.get(1)?;
            Ok(ProfilePoint {
                line: row.get(0)?,
                class_id: class_id as u32,
                class_name: String::new(),
                position: row.get(2)?,
                n_objects: row.get::<_, i64>(3)? as u64,
                mean: row.get(4)?,
                sd: row.get(5)?,
                median: row.get(6)?,
                q1: row.get(7)?,
                q3: row.get(8)?,
            })
        })
        .map_err(err)?
        .collect::<Result<Vec<_>, _>>()
        .map_err(err)?;
    let mut result = ProfileResult::default();
    let mut kept: Vec<(u32, String)> = Vec::new();
    for mut point in rows {
        let line = (point.class_id, point.line.clone());
        if !kept.contains(&line) {
            result.lines_total += 1;
            if kept.len() < filter.max_lines {
                kept.push(line.clone());
            }
        }
        if !kept.contains(&line) {
            continue;
        }
        if filter.lines == ProfileLines::Class {
            point.class_name = names
                .get(&point.class_id)
                .cloned()
                .unwrap_or_else(|| format!("class {}", point.class_id));
        }
        result.points.push(point);
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::super::test_support::{ObjectSpec, profile_database, seed_db};
    use super::*;
    use crate::api::Grouping;

    fn database() -> ResultsGenerator {
        profile_database()
    }

    fn filter(lines: ProfileLines) -> ProfileFilter {
        ProfileFilter {
            key: "sholl_01/intersections".into(),
            lines,
            grouping: Grouping::Auto,
            images: None,
            object_classes: None,
            max_lines: 24,
        }
    }

    #[test]
    fn only_profiles_are_listed() {
        let keys: Vec<String> = profile_columns(&database())
            .unwrap()
            .into_iter()
            .map(|c| c.key)
            .collect();
        assert_eq!(keys, ["sholl_01/intersections"]);
    }

    #[test]
    fn a_missing_radius_counts_as_zero() {
        let result = profile(&database(), &filter(ProfileLines::Class)).unwrap();
        let a: Vec<(f64, u64, Option<f64>)> = result
            .points
            .iter()
            .filter(|p| p.class_name == "A")
            .map(|p| (p.position, p.n_objects, p.mean))
            .collect();
        // At r 30 only the second neuron crosses (2): with the first's 0,
        // the mean is 1 over both neurons.
        assert_eq!(
            a,
            [
                (10.0, 2, Some(3.0)),
                (20.0, 2, Some(5.0)),
                (30.0, 2, Some(1.0))
            ]
        );
        let at30 = result
            .points
            .iter()
            .find(|p| p.class_name == "A" && p.position == 30.0)
            .unwrap();
        assert_eq!(at30.median, Some(1.0));
        assert_eq!(at30.q1, Some(0.5));
        assert_eq!(result.lines_total, 2);
    }

    #[test]
    fn lines_per_image_and_all() {
        let images = profile(&database(), &filter(ProfileLines::Image)).unwrap();
        let lines: Vec<&str> = images.points.iter().map(|p| p.line.as_str()).collect();
        assert_eq!(lines, ["A01_1.tif", "A01_1.tif", "A01_1.tif", "B01_1.tif"]);
        let all = profile(&database(), &filter(ProfileLines::All)).unwrap();
        // All three neurons at r 10: (2 + 4 + 1) / 3.
        let at10 = all.points.iter().find(|p| p.position == 10.0).unwrap();
        assert_eq!(at10.n_objects, 3);
        assert!((at10.mean.unwrap() - 7.0 / 3.0).abs() < 1e-12);
        assert_eq!(all.lines_total, 1);
    }

    #[test]
    fn the_classes_filter_and_the_line_limit() {
        let mut only_b = filter(ProfileLines::Class);
        only_b.object_classes = Some(vec![ObjectClass::Global(2)]);
        let result = profile(&database(), &only_b).unwrap();
        assert!(result.points.iter().all(|p| p.class_name == "B"));
        let mut one = filter(ProfileLines::Class);
        one.max_lines = 1;
        let result = profile(&database(), &one).unwrap();
        assert!(result.points.iter().all(|p| p.class_name == "A"));
        assert_eq!(result.lines_total, 2);
    }

    #[test]
    fn one_objects_curve() {
        let curve = object_profile(
            &database(),
            "sholl_01/intersections",
            "00000000-0000-0000-0000-000000000001",
        )
        .unwrap();
        assert_eq!(curve, [(10.0, 4.0), (20.0, 6.0), (30.0, 2.0)]);
    }

    fn step(from: Option<usize>, delta: i32) -> ProfileStepFilter {
        ProfileStepFilter {
            key: "sholl_01/intersections".into(),
            object_classes: None,
            from: from.map(|i| format!("00000000-0000-0000-0000-{i:012}")),
            delta,
        }
    }

    fn landed(step: &ProfileStep) -> (usize, usize, Option<String>) {
        (
            step.position,
            step.total,
            step.object.as_ref().map(|o| o.object_id[24..].to_string()),
        )
    }

    #[test]
    fn stepping_goes_through_the_objects_by_image_and_stops_at_the_ends() {
        let db = database();
        // The two neurons of A01_1 (same box: by id), then B01_1's.
        assert_eq!(
            landed(&profile_step(&db, &step(None, 1)).unwrap()),
            (1, 3, Some("000000000000".into()))
        );
        assert_eq!(
            landed(&profile_step(&db, &step(None, -1)).unwrap()),
            (3, 3, Some("000000000002".into()))
        );
        assert_eq!(
            landed(&profile_step(&db, &step(Some(0), 1)).unwrap()),
            (2, 3, Some("000000000001".into()))
        );
        assert_eq!(
            landed(&profile_step(&db, &step(Some(2), 1)).unwrap()),
            (3, 3, Some("000000000002".into()))
        );
        assert_eq!(
            landed(&profile_step(&db, &step(Some(0), -1)).unwrap()),
            (1, 3, Some("000000000000".into()))
        );
        // Where an object is, and the count without one.
        assert_eq!(
            landed(&profile_step(&db, &step(Some(1), 0)).unwrap()),
            (2, 3, Some("000000000001".into()))
        );
        assert_eq!(
            landed(&profile_step(&db, &step(None, 0)).unwrap()),
            (0, 3, None)
        );
        // An object without the profile: like none.
        assert_eq!(
            landed(&profile_step(&db, &step(Some(9), 1)).unwrap()),
            (1, 3, Some("000000000000".into()))
        );
    }

    #[test]
    fn a_step_names_the_objects_image_box_and_centre() {
        let db = database();
        db.connection()
            .execute_batch(
                "UPDATE measurements SET related_object_id = '00000000-0000-0000-0000-000000000007' \
                 WHERE object_id = '00000000-0000-0000-0000-000000000002'",
            )
            .unwrap();
        let last = profile_step(&db, &step(None, -1)).unwrap().object.unwrap();
        assert_eq!(last.image_rel_path, "B01_1.tif");
        assert_eq!(last.bbox_px, Some([0, 0, 10, 10]));
        assert_eq!(
            last.centre_id.as_deref(),
            Some("00000000-0000-0000-0000-000000000007")
        );
        let first = profile_step(&db, &step(None, 1)).unwrap().object.unwrap();
        assert_eq!(first.centre_id, None);
    }

    #[test]
    fn stepping_keeps_to_the_classes_asked_for() {
        let db = database();
        let mut only_b = step(None, 1);
        only_b.object_classes = Some(vec![ObjectClass::Global(2)]);
        assert_eq!(
            landed(&profile_step(&db, &only_b).unwrap()),
            (1, 1, Some("000000000002".into()))
        );
        let mut none = step(None, 1);
        none.object_classes = Some(vec![ObjectClass::Unset]);
        assert_eq!(landed(&profile_step(&db, &none).unwrap()), (0, 0, None));
    }

    #[test]
    fn results_of_format_1_have_no_profiles() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("results.evadb");
        seed_db(&path, &[ObjectSpec::new("a.tif", "A", 1, 10)]);
        let db = ResultsGenerator::open_database(path).unwrap();
        assert!(profile_columns(&db).unwrap().is_empty());
        assert_eq!(
            profile(&db, &filter(ProfileLines::Class)).unwrap(),
            ProfileResult::default()
        );
        assert_eq!(
            profile_step(&db, &step(None, 1)).unwrap(),
            ProfileStep::default()
        );
    }
}
