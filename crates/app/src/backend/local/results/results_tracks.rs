//! # results_tracks
//!
//! One row per track: when it was seen, how far and how straight it moved,
//! its measurements over its time points - Results › Tracks and its export.
//! A track never leaves its image: one query per image, so a results file
//! with many thousands of tracks is read an image's worth at a time.
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-10-21
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.

use super::results_generator::{
    ResultsGenerator, class_filter_sql, column_aggregate_expr, sql_string_in_list,
};
use super::results_time_course::stat_sql;
use crate::api::{Column, TrackSummary, TrackSummaryFilter};
use evanalyzer_cfg::core_types::{InternalErrors, ObjectClass};
use std::collections::HashMap;

/// What every row holds - known before the rows are read.
#[derive(Debug, Clone, PartialEq)]
pub(super) struct TrackSummaryHead {
    /// "Area [px] mean" per entry of `TrackSummary::values`.
    pub value_names: Vec<String>,
    /// Several z-planes: a Z column.
    pub by_z: bool,
    /// Every tracked object has a pixel size: lengths can be in µm.
    pub um: bool,
    /// Every image has a frame interval: times can be in minutes.
    pub timed: bool,
    /// The well / group column's name.
    pub group_label: &'static str,
}

/// An image whose tracks are read.
struct Image {
    rel_path: String,
    name: String,
    group: String,
    s_per_frame: Option<f64>,
}

/// The tracks `filter` asks for, ready to read (see the module).
pub(super) struct TrackSummaryQuery<'a> {
    database: &'a ResultsGenerator,
    pub head: TrackSummaryHead,
    images: Vec<Image>,
    class_names: HashMap<i32, String>,
    /// For an image: its rows (bound: the image's path).
    sql: String,
    first_t: u32,
    n_values: usize,
}

impl<'a> TrackSummaryQuery<'a> {
    pub fn prepare(
        database: &'a ResultsGenerator,
        filter: &TrackSummaryFilter,
    ) -> Result<Self, InternalErrors> {
        let err = |e: duckdb::Error| InternalErrors::Io(e.to_string());
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
        let value_names: Vec<String> = columns
            .iter()
            .flat_map(|column| {
                let label = column.display_label(&classes);
                filter
                    .statistics
                    .iter()
                    .map(move |stat| format!("{label} {}", stat.label()))
            })
            .collect();
        let planes = database.get_planes()?;
        let by_z = planes.iter().any(|p| p.z_stack != planes[0].z_stack);
        let first_t = planes.iter().map(|p| p.t_stack).min().unwrap_or(0);

        // The images: analysed, not left out - their well / group and frame
        // interval (unknown in results from before it was recorded).
        let grouping = database.grouping_sql(&filter.grouping)?;
        let mut image_where = "NOT i.disabled AND i.successful".to_string();
        if let Some(paths) = &filter.images {
            image_where += &format!(" AND i.image_rel_path IN ({})", sql_string_in_list(paths));
        }
        let group_expr = format!(
            "regexp_extract(i.{}, '{}', 1)",
            grouping.column, grouping.regex
        );
        let read_images = |interval: &str| -> Result<Vec<Image>, duckdb::Error> {
            let sql = format!(
                "SELECT i.image_rel_path, i.image_name, {group_expr}, {interval} \
                 FROM images i WHERE {image_where} ORDER BY i.image_rel_path"
            );
            let mut stmt = database.connection().prepare(&sql)?;
            stmt.query_map([], |row| {
                Ok(Image {
                    rel_path: row.get(0)?,
                    name: row.get(1)?,
                    group: row.get::<_, Option<String>>(2)?.unwrap_or_default(),
                    s_per_frame: row.get::<_, Option<f64>>(3)?.filter(|s| *s > 0.0),
                })
            })?
            .collect()
        };
        let images = match read_images("i.frame_interval_s") {
            Ok(images) => images,
            Err(_) => read_images("NULL::DOUBLE").map_err(err)?,
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
        let mut object_conditions = vec!["COALESCE(track_id, 0) <> 0".to_string()];
        if let Some(ids) = &selected_ids {
            object_conditions.push(match ids.is_empty() {
                true => "false".to_string(),
                false => class_filter_sql("object_class_id", ids),
            });
        }
        let object_where = object_conditions.join(" AND ");
        let um = database
            .connection()
            .query_row(
                &format!(
                    "SELECT COALESCE(bool_and(COALESCE(pixel_size_x_nm, 0) > 0), false) \
                     FROM objects WHERE {object_where}"
                ),
                [],
                |row| row.get(0),
            )
            .map_err(err)?;
        let timed = !images.is_empty() && images.iter().all(|image| image.s_per_frame.is_some());

        let values_sql = columns
            .iter()
            .enumerate()
            .map(|(i, column)| Ok(format!(", {} AS v{i}", column_aggregate_expr(column)?)))
            .collect::<Result<String, InternalErrors>>()?;
        let stats_sql: String = (0..columns.len())
            .flat_map(|i| {
                filter
                    .statistics
                    .iter()
                    .map(move |stat| format!(", {}", stat_sql(*stat, &format!("v{i}"))))
            })
            .collect();
        let mut having = vec![format!("COUNT(*) >= {}", filter.min_points.max(1))];
        if filter.from_start {
            having.push(format!("MIN(t) = {first_t}"));
        }
        if filter.no_edge {
            having.push("NOT bool_or(edge)".to_string());
        }
        // A track's steps: from each point to the next (per z-plane - the
        // tracker links within one).
        let sql = format!(
            "WITH o AS (SELECT track_id, t_stack AS t, z_stack AS z, \
                    centroid_x_px AS x, centroid_y_px AS y, COALESCE(touches_edge, false) AS edge, \
                    COALESCE(object_class_id[1], -1) AS cls, NULLIF(pixel_size_x_nm, 0) AS nm{values_sql} \
                FROM objects WHERE image_rel_path = ? AND {object_where}), \
             s AS (SELECT *, sqrt(power(x - lag(x) OVER w, 2) + power(y - lag(y) OVER w, 2)) AS step \
                FROM o WINDOW w AS (PARTITION BY track_id, z ORDER BY t)) \
             SELECT track_id::UBIGINT, MIN(z)::BIGINT, arg_min(cls, t)::BIGINT, MIN(t)::BIGINT, \
                MAX(t)::BIGINT, COUNT(*)::BIGINT, COALESCE(SUM(step), 0)::DOUBLE, \
                sqrt(power(arg_max(x, t) - arg_min(x, t), 2) + power(arg_max(y, t) - arg_min(y, t), 2))::DOUBLE, \
                bool_or(edge), MIN(nm)::DOUBLE{stats_sql} \
             FROM s GROUP BY track_id HAVING {} ORDER BY track_id",
            having.join(" AND ")
        );
        Ok(Self {
            database,
            head: TrackSummaryHead {
                value_names,
                by_z,
                um,
                timed,
                group_label: match filter.grouping {
                    crate::api::Grouping::Folder => "Folder",
                    _ => "Well",
                },
            },
            images,
            class_names: classes
                .iter()
                .map(|c| (c.id as i32, c.name.clone()))
                .collect(),
            sql,
            first_t,
            n_values: columns.len() * filter.statistics.len(),
        })
    }

    /// Each track, image by image, by track id within an image.
    pub fn for_each(
        &self,
        mut visit: impl FnMut(&TrackSummary) -> Result<(), InternalErrors>,
    ) -> Result<(), InternalErrors> {
        let err = |e: duckdb::Error| InternalErrors::Io(e.to_string());
        let mut stmt = self.database.connection().prepare(&self.sql).map_err(err)?;
        for image in &self.images {
            let rows = stmt
                .query_map([&image.rel_path], |row| {
                    let class_id = row.get::<_, i64>(2)? as i32;
                    let first_t = row.get::<_, i64>(3)? as u32;
                    Ok(TrackSummary {
                        image: image.name.clone(),
                        group: image.group.clone(),
                        track_id: row.get(0)?,
                        z: row.get::<_, i64>(1)? as u32,
                        class_id,
                        class_name: self
                            .class_names
                            .get(&class_id)
                            .cloned()
                            .unwrap_or_else(|| class_id.to_string()),
                        first_t,
                        last_t: row.get::<_, i64>(4)? as u32,
                        points: row.get::<_, i64>(5)? as u32,
                        path_px: row.get(6)?,
                        displacement_px: row.get::<_, Option<f64>>(7)?.unwrap_or(0.0),
                        touches_edge: row.get(8)?,
                        from_start: first_t == self.first_t,
                        um_per_px: row.get::<_, Option<f64>>(9)?.map(|nm| nm / 1000.0),
                        s_per_frame: image.s_per_frame,
                        values: (0..self.n_values)
                            .map(|k| row.get::<_, Option<f64>>(10 + k))
                            .collect::<Result<_, _>>()?,
                    })
                })
                .map_err(err)?;
            for track in rows {
                visit(&track.map_err(err)?)?;
            }
        }
        Ok(())
    }

    /// Every track, kept.
    pub fn collect(&self) -> Result<Vec<TrackSummary>, InternalErrors> {
        let mut tracks = Vec::new();
        self.for_each(|track| {
            tracks.push(track.clone());
            Ok(())
        })?;
        Ok(tracks)
    }
}

impl ResultsGenerator {
    /// One row per track - see [`TrackSummaryFilter`].
    pub fn track_summary(
        &self,
        filter: &TrackSummaryFilter,
    ) -> Result<Vec<TrackSummary>, InternalErrors> {
        TrackSummaryQuery::prepare(self, filter)?.collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::{Grouping, TimeCourseStat};
    use crate::backend::local::results::test_support::{ObjectSpec, seed_db};

    /// The results of `objects` over `times` time points, then `sql` on
    /// them (edges, pixel sizes, a frame interval).
    fn open(objects: &[ObjectSpec], times: i32, sql: &str) -> ResultsGenerator {
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
                 z_stacks_analyzed = [0]",
                duckdb::params![times, times],
            )
            .unwrap();
            conn.execute_batch(sql).unwrap();
        }
        std::mem::forget(dir);
        ResultsGenerator::open_database(path).expect("open database")
    }

    /// Track 7 in img1: (0, 0) → (3, 4) → (3, 0), areas 10, 20, 30. Track 8
    /// in img2 from T 1: one point. Track 9 in img1, ClassB: T 0 and T 1.
    fn tracks() -> Vec<ObjectSpec> {
        vec![
            ObjectSpec::new("img1.tif", "ClassA", 1, 10)
                .in_track(7)
                .at_centroid(0.0, 0.0),
            ObjectSpec::new("img1.tif", "ClassA", 1, 20)
                .in_track(7)
                .at_plane(0, 1)
                .at_centroid(3.0, 4.0),
            ObjectSpec::new("img1.tif", "ClassA", 1, 30)
                .in_track(7)
                .at_plane(0, 2)
                .at_centroid(3.0, 0.0),
            ObjectSpec::new("img2.tif", "ClassA", 1, 50)
                .in_track(8)
                .at_plane(0, 1),
            ObjectSpec::new("img1.tif", "ClassB", 2, 5).in_track(9),
            ObjectSpec::new("img1.tif", "ClassB", 2, 7)
                .in_track(9)
                .at_plane(0, 1),
            ObjectSpec::new("img1.tif", "ClassA", 1, 99),
        ]
    }

    fn filter() -> TrackSummaryFilter {
        TrackSummaryFilter {
            grouping: Grouping::Auto,
            columns: vec![Column::AreaSizePx],
            statistics: vec![TimeCourseStat::Mean, TimeCourseStat::Max],
            ..TrackSummaryFilter::default()
        }
    }

    /// A row per track - image by image, by id: its time points, path
    /// point to point, distance first to last, speed, straightness and
    /// its measurements' statistics. Untracked objects aren't one.
    #[test]
    fn a_track_is_summed_up_from_its_points() {
        let db = open(&tracks(), 3, "");
        let rows = db.track_summary(&filter()).unwrap();
        let ids: Vec<u64> = rows.iter().map(|r| r.track_id).collect();
        assert_eq!(ids, [7, 9, 8]);
        let seven = &rows[0];
        assert_eq!(
            (seven.image.as_str(), seven.class_name.as_str()),
            ("img1.tif", "ClassA")
        );
        assert_eq!((seven.first_t, seven.last_t, seven.points), (0, 2, 3));
        assert_eq!((seven.path_px, seven.displacement_px), (9.0, 3.0));
        assert_eq!(seven.speed_px_per_frame(), Some(4.5));
        assert_eq!(seven.straightness(), Some(3.0 / 9.0));
        assert_eq!(seven.values, [Some(20.0), Some(30.0)]);
        assert!(seven.from_start && !seven.touches_edge);
        // The fixture's pixel size: 1 nm; no frame interval.
        assert_eq!((seven.um_per_px, seven.s_per_frame), (Some(0.001), None));
        assert_eq!(rows[1].class_name, "ClassB");
        let eight = &rows[2];
        assert_eq!(
            (eight.points, eight.path_px, eight.from_start),
            (1, 0.0, false)
        );
        assert_eq!(
            (eight.speed_px_per_frame(), eight.straightness()),
            (None, None)
        );
    }

    /// At least N time points, from the first time point, not touching the
    /// edge, of a class.
    #[test]
    fn tracks_can_be_left_out() {
        let db = open(
            &tracks(),
            3,
            "UPDATE objects SET touches_edge = (track_id = 9 AND t_stack = 1)",
        );
        let ids = |filter: TrackSummaryFilter| -> Vec<u64> {
            db.track_summary(&filter)
                .unwrap()
                .iter()
                .map(|r| r.track_id)
                .collect()
        };
        assert_eq!(
            ids(TrackSummaryFilter {
                min_points: 2,
                ..filter()
            }),
            [7, 9]
        );
        assert_eq!(
            ids(TrackSummaryFilter {
                from_start: true,
                ..filter()
            }),
            [7, 9]
        );
        assert_eq!(
            ids(TrackSummaryFilter {
                no_edge: true,
                ..filter()
            }),
            [7, 8]
        );
        assert_eq!(
            ids(TrackSummaryFilter {
                object_classes: Some(vec![ObjectClass::Global(2)]),
                ..filter()
            }),
            [9]
        );
        assert_eq!(
            ids(TrackSummaryFilter {
                images: Some(vec!["img2.tif".into()]),
                ..filter()
            }),
            [8]
        );
    }

    /// A pixel size and a frame interval everywhere: µm and seconds can
    /// be shown; one image without: not.
    #[test]
    fn units_come_from_the_pixel_size_and_frame_interval() {
        let db = open(
            &tracks(),
            3,
            "UPDATE objects SET pixel_size_x_nm = 500; UPDATE images SET frame_interval_s = 30;",
        );
        let query = TrackSummaryQuery::prepare(&db, &filter()).unwrap();
        assert!(query.head.um && query.head.timed);
        let rows = query.collect().unwrap();
        assert_eq!(
            (rows[0].um_per_px, rows[0].s_per_frame),
            (Some(0.5), Some(30.0))
        );
        assert_eq!(query.head.value_names, ["Area [px] mean", "Area [px] max"]);

        let db = open(
            &tracks(),
            3,
            "UPDATE images SET frame_interval_s = 30 WHERE image_rel_path = 'img1.tif'; \
             UPDATE objects SET pixel_size_x_nm = NULL WHERE track_id = 8;",
        );
        let head = TrackSummaryQuery::prepare(&db, &filter()).unwrap().head;
        assert!(!head.um && !head.timed);
    }
}
