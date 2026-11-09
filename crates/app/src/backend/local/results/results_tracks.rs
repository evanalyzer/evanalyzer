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
use crate::api::{
    Column, TrackFromStart, TrackLineage, TrackMotion, TrackPaths, TrackPathsFilter, TrackSummary,
    TrackSummaryFilter, drift_at,
};
use evanalyzer_cfg::core_types::{InternalErrors, ObjectClass};
use std::collections::{BTreeMap, HashMap};

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
    /// A tracking step recorded divisions or fusions: the rows' lineage.
    pub lineage: bool,
    /// Measure Track Motion measured tracks: the rows' motion values.
    pub motion: bool,
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
    /// Per image, its tracks' parents (with lineage) - read in one go: few
    /// tracks have any.
    parents: HashMap<String, HashMap<u64, Vec<u64>>>,
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
        let lineage = database.has_lineage()?;
        let parents = match lineage {
            true => database.all_track_parents()?,
            false => HashMap::new(),
        };

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
        // How a track moved isn't computed here: Measure Track Motion
        // stored it (read apart, per image).
        let sql = format!(
            "WITH o AS (SELECT track_id, t_stack AS t, z_stack AS z, \
                    COALESCE(touches_edge, false) AS edge, \
                    COALESCE(object_class_id[1], -1) AS cls, NULLIF(pixel_size_x_nm, 0) AS nm{values_sql} \
                FROM objects WHERE image_rel_path = ? AND {object_where}) \
             SELECT track_id::UBIGINT, MIN(z)::BIGINT, arg_min(cls, t)::BIGINT, MIN(t)::BIGINT, \
                MAX(t)::BIGINT, COUNT(*)::BIGINT, bool_or(edge), MIN(nm)::DOUBLE{stats_sql} \
             FROM o GROUP BY track_id HAVING {} ORDER BY track_id",
            having.join(" AND ")
        );
        let motion = database.has_track_motion()?;
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
                lineage,
                motion,
            },
            images,
            class_names: classes
                .iter()
                .map(|c| (c.id as i32, c.name.clone()))
                .collect(),
            sql,
            first_t,
            n_values: columns.len() * filter.statistics.len(),
            parents,
        })
    }

    /// Each track, image by image, by track id within an image.
    pub fn for_each(
        &self,
        mut visit: impl FnMut(&TrackSummary) -> Result<(), InternalErrors>,
    ) -> Result<(), InternalErrors> {
        self.for_each_in_image(|_, track| visit(track))
    }

    /// [`Self::for_each`] with each track's image's path.
    fn for_each_in_image(
        &self,
        mut visit: impl FnMut(&str, &TrackSummary) -> Result<(), InternalErrors>,
    ) -> Result<(), InternalErrors> {
        let err = |e: duckdb::Error| InternalErrors::Io(e.to_string());
        let mut stmt = self.database.connection().prepare(&self.sql).map_err(err)?;
        for image in &self.images {
            // A track never leaves its image: its family neither.
            let lineages = match self.parents.get(&image.rel_path) {
                Some(parents) => Lineages::new(parents.clone()),
                None => Lineages::default(),
            };
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
                        motion: TrackMotion::default(),
                        touches_edge: row.get(6)?,
                        from_start: first_t == self.first_t,
                        um_per_px: row.get::<_, Option<f64>>(7)?.map(|nm| nm / 1000.0),
                        s_per_frame: image.s_per_frame,
                        values: (0..self.n_values)
                            .map(|k| row.get::<_, Option<f64>>(8 + k))
                            .collect::<Result<_, _>>()?,
                        lineage: TrackLineage::default(),
                    })
                })
                .map_err(err)?;
            let mut motions = match self.head.motion {
                true => self.database.track_motion(&image.rel_path)?,
                false => HashMap::new(),
            };
            for track in rows {
                let mut track = track.map_err(err)?;
                track.lineage = lineages.of(track.track_id);
                if let Some(motion) = motions.remove(&track.track_id) {
                    track.motion = motion;
                }
                visit(&image.rel_path, &track)?;
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

    /// At most `max_tracks` of the tracks, an even sample (every k-th, the
    /// same each time), each from its start minus its image's stored drift
    /// - see [`TrackPathsFilter`]. Only the sample's points are read, an
    /// image at a time.
    pub fn paths(&self, max_tracks: usize) -> Result<TrackPaths, InternalErrors> {
        let err = |e: duckdb::Error| InternalErrors::Io(e.to_string());
        // Every track's key (small: no values), then the sample.
        let mut all: Vec<(String, TrackFromStart, u32)> = Vec::new();
        self.for_each_in_image(|path, track| {
            all.push((
                path.to_string(),
                TrackFromStart {
                    image: track.image.clone(),
                    group: track.group.clone(),
                    class_id: track.class_id,
                    track_id: track.track_id,
                    um_per_px: track.um_per_px,
                    points: Vec::new(),
                },
                track.z,
            ));
            Ok(())
        })?;
        let total_tracks = all.len();
        let keep = max_tracks.min(total_tracks);
        let mut sample: Vec<(String, TrackFromStart, u32)> = match keep < total_tracks {
            true => (0..keep)
                .map(|i| all[i * total_tracks / keep].clone())
                .collect(),
            false => all,
        };
        let mut stmt = self
            .database
            .connection()
            .prepare(
                "SELECT track_id, t_stack, centroid_x_px, centroid_y_px FROM objects \
                 WHERE image_rel_path = ? AND z_stack = ? AND track_id = ANY(?::UBIGINT[]) \
                 ORDER BY track_id, t_stack",
            )
            .map_err(err)?;
        // By image and plane: the sample is in image order already.
        let mut start = 0;
        while start < sample.len() {
            let (path, z) = (sample[start].0.clone(), sample[start].2);
            let end = start
                + sample[start..]
                    .iter()
                    .take_while(|(p, _, tz)| *p == path && *tz == z)
                    .count();
            let ids: Vec<String> = sample[start..end]
                .iter()
                .map(|(_, t, _)| t.track_id.to_string())
                .collect();
            let index: HashMap<u64, usize> =
                (start..end).map(|i| (sample[i].1.track_id, i)).collect();
            let drift = self.database.stored_drift(&path, z)?;
            let mut first: HashMap<u64, (f64, f64)> = HashMap::new();
            let mut rows = stmt
                .query(duckdb::params![path, z, format!("[{}]", ids.join(","))])
                .map_err(err)?;
            while let Some(row) = rows.next().map_err(err)? {
                let track_id: u64 = row.get(0).map_err(err)?;
                let t = row.get::<_, i64>(1).map_err(err)? as u32;
                let (dx, dy) = drift_at(&drift, t);
                let x = row.get::<_, Option<f64>>(2).map_err(err)?.unwrap_or(0.0) - dx;
                let y = row.get::<_, Option<f64>>(3).map_err(err)?.unwrap_or(0.0) - dy;
                let Some(&i) = index.get(&track_id) else {
                    continue;
                };
                let origin = *first.entry(track_id).or_insert((x, y));
                sample[i]
                    .1
                    .points
                    .push(((x - origin.0) as f32, (y - origin.1) as f32));
            }
            start = end;
        }
        Ok(TrackPaths {
            tracks: sample.into_iter().map(|(_, track, _)| track).collect(),
            total_tracks,
        })
    }
}

/// An image's family trees, from each track's parents.
#[derive(Debug, Default)]
pub(super) struct Lineages {
    parents: HashMap<u64, Vec<u64>>,
    children: HashMap<u64, Vec<u64>>,
}

impl Lineages {
    pub(super) fn new(parents: HashMap<u64, Vec<u64>>) -> Self {
        let mut children: HashMap<u64, Vec<u64>> = HashMap::new();
        for (&child, of) in &parents {
            for &parent in of {
                children.entry(parent).or_default().push(child);
            }
        }
        for list in children.values_mut() {
            list.sort_unstable();
        }
        Self { parents, children }
    }

    /// Track `id`'s place in its tree.
    pub(super) fn of(&self, id: u64) -> TrackLineage {
        if self.parents.is_empty() {
            return TrackLineage {
                root: id,
                ..TrackLineage::default()
            };
        }
        let children = self.children.get(&id).cloned().unwrap_or_default();
        let fuses = children
            .iter()
            .any(|child| self.parents.get(child).is_some_and(|p| p.len() > 1));
        let (root, generation) = self.ancestry(id);
        TrackLineage {
            parents: self.parents.get(&id).cloned().unwrap_or_default(),
            children,
            root,
            generation,
            fuses,
        }
    }

    /// Its root (the lowest of its ancestors without parents) and its
    /// generation (the longest way up) - walked up, so a family's depth
    /// (divisions in a movie: tens) is the cost.
    fn ancestry(&self, id: u64) -> (u64, u32) {
        // (track, generation below it) - every way up, depth first.
        let mut stack = vec![(id, 0u32)];
        let (mut root, mut generation) = (u64::MAX, 0);
        let mut seen: BTreeMap<u64, u32> = BTreeMap::new();
        while let Some((track, depth)) = stack.pop() {
            // Each track once at its deepest - a fusion's parents can share
            // ancestors (and a broken file can't loop forever).
            if seen.get(&track).is_some_and(|&d| d >= depth) || depth as usize > self.parents.len()
            {
                continue;
            }
            seen.insert(track, depth);
            match self.parents.get(&track).filter(|p| !p.is_empty()) {
                None => {
                    root = root.min(track);
                    generation = generation.max(depth);
                }
                Some(parents) => stack.extend(parents.iter().map(|&p| (p, depth + 1))),
            }
        }
        (root, generation)
    }
}

impl ResultsGenerator {
    /// Measure Track Motion stored values (`measurements` rows with
    /// `region = 'track'`) - not in results without it or from before it.
    /// Asked once per opened file: the values are written by the run, not
    /// while the results are open.
    pub fn has_track_motion(&self) -> Result<bool, InternalErrors> {
        if let Some(cached) = *self.track_motion_cache.borrow() {
            return Ok(cached);
        }
        let has = self.has_column("measurements", "track_id")?
            && self
                .connection()
                .query_row(
                    "SELECT EXISTS (SELECT 1 FROM measurements WHERE region = 'track')",
                    [],
                    |row| row.get(0),
                )
                .map_err(|e| InternalErrors::Io(e.to_string()))?;
        *self.track_motion_cache.borrow_mut() = Some(has);
        Ok(has)
    }

    /// [`Self::track_motion`] and [`Self::stored_drift`] of an image plane
    /// in one query (Results › Tracks reads both for every image shown).
    pub(super) fn track_motion_and_drift(
        &self,
        image_rel_path: &str,
        z: u32,
    ) -> Result<(HashMap<u64, TrackMotion>, Vec<(u32, f64, f64)>), InternalErrors> {
        let err = |e: duckdb::Error| InternalErrors::Io(e.to_string());
        if !self.has_track_motion()? {
            return Ok((HashMap::new(), Vec::new()));
        }
        let mut stmt = self
            .connection()
            .prepare_cached(
                "SELECT track_id, t_stack, step, key, value FROM measurements \
                 WHERE image_rel_path = ? AND (region = 'track' OR (region = 'image' \
                     AND z_stack = ? AND (key LIKE '%/drift_x' OR key LIKE '%/drift_y')))",
            )
            .map_err(err)?;
        let mut rows = stmt
            .query(duckdb::params![image_rel_path, z])
            .map_err(err)?;
        let mut motions: HashMap<u64, TrackMotion> = HashMap::new();
        // Drift by step, then only the steps that measured tracks here.
        let mut drift: HashMap<String, BTreeMap<u32, (f64, f64)>> = HashMap::new();
        let mut track_steps: Vec<String> = Vec::new();
        while let Some(row) = rows.next().map_err(err)? {
            let track: Option<u64> = row.get(0).map_err(err)?;
            let step: String = row.get(2).map_err(err)?;
            let key: String = row.get(3).map_err(err)?;
            // "motion_01/vcl": the value's id after the step's.
            let id = key.split_once('/').map_or(key.as_str(), |(_, id)| id);
            let value: Option<f64> = row.get(4).map_err(err)?;
            match track {
                Some(track) => {
                    motions.entry(track).or_default().set(id, value);
                    if !track_steps.contains(&step) {
                        track_steps.push(step);
                    }
                }
                None => {
                    let t = row.get::<_, i64>(1).map_err(err)? as u32;
                    let at = drift.entry(step).or_default().entry(t).or_default();
                    match id {
                        "drift_x" => at.0 = value.unwrap_or(0.0),
                        _ => at.1 = value.unwrap_or(0.0),
                    }
                }
            }
        }
        let drift = track_steps
            .iter()
            .find_map(|step| drift.remove(step))
            .map(|by_t| by_t.into_iter().map(|(t, (x, y))| (t, x, y)).collect())
            .unwrap_or_default();
        Ok((motions, drift))
    }

    /// What Measure Track Motion stored of each track of `image_rel_path`,
    /// by track id - none without it. Two such steps on one track: either's
    /// value.
    pub(super) fn track_motion(
        &self,
        image_rel_path: &str,
    ) -> Result<HashMap<u64, TrackMotion>, InternalErrors> {
        let err = |e: duckdb::Error| InternalErrors::Io(e.to_string());
        let mut stmt = self
            .connection()
            .prepare(
                "SELECT track_id, split_part(key, '/', 2), value FROM measurements \
                 WHERE region = 'track' AND image_rel_path = ? AND track_id IS NOT NULL",
            )
            .map_err(err)?;
        let rows = stmt
            .query_map([image_rel_path], |row| {
                Ok((
                    row.get::<_, u64>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, Option<f64>>(2)?,
                ))
            })
            .map_err(err)?;
        let mut motions: HashMap<u64, TrackMotion> = HashMap::new();
        for row in rows {
            let (track, id, value) = row.map_err(err)?;
            motions.entry(track).or_default().set(&id, value);
        }
        Ok(motions)
    }

    /// The results hold divisions or fusions (`objects.parent_track_ids`,
    /// set by a tracking step) - not in results from before it.
    pub fn has_lineage(&self) -> Result<bool, InternalErrors> {
        if !self.has_object_column("parent_track_ids")? {
            return Ok(false);
        }
        self.connection()
            .query_row(
                "SELECT EXISTS (SELECT 1 FROM objects WHERE parent_track_ids IS NOT NULL)",
                [],
                |row| row.get(0),
            )
            .map_err(|e| InternalErrors::Io(e.to_string()))
    }

    /// Per image, the parents of each of its tracks that has any - none in
    /// results without lineage.
    pub(super) fn all_track_parents(
        &self,
    ) -> Result<HashMap<String, HashMap<u64, Vec<u64>>>, InternalErrors> {
        let err = |e: duckdb::Error| InternalErrors::Io(e.to_string());
        if !self.has_object_column("parent_track_ids")? {
            return Ok(HashMap::new());
        }
        let mut stmt = self
            .connection()
            .prepare(
                "SELECT image_rel_path, track_id, UNNEST(parents) FROM (SELECT image_rel_path, \
                     track_id, any_value(parent_track_ids) AS parents FROM objects \
                     WHERE parent_track_ids IS NOT NULL GROUP BY image_rel_path, track_id) \
                 ORDER BY 1, 2, 3",
            )
            .map_err(err)?;
        let mut parents: HashMap<String, HashMap<u64, Vec<u64>>> = HashMap::new();
        let rows = stmt
            .query_map([], |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, u64>(1)?,
                    row.get::<_, u64>(2)?,
                ))
            })
            .map_err(err)?;
        for row in rows {
            let (image, track, parent) = row.map_err(err)?;
            parents
                .entry(image)
                .or_default()
                .entry(track)
                .or_default()
                .push(parent);
        }
        Ok(parents)
    }

    /// The parents of each track of `image_rel_path` that has any - none
    /// in results without lineage.
    pub(super) fn track_parents(
        &self,
        image_rel_path: &str,
    ) -> Result<HashMap<u64, Vec<u64>>, InternalErrors> {
        let err = |e: duckdb::Error| InternalErrors::Io(e.to_string());
        if !self.has_object_column("parent_track_ids")? {
            return Ok(HashMap::new());
        }
        // The same on every object of a track: any one's.
        let mut stmt = self
            .connection()
            .prepare(
                "SELECT track_id, UNNEST(parents) FROM (SELECT track_id, \
                     any_value(parent_track_ids) AS parents FROM objects \
                     WHERE image_rel_path = ? AND parent_track_ids IS NOT NULL \
                     GROUP BY track_id) ORDER BY 1, 2",
            )
            .map_err(err)?;
        let mut parents: HashMap<u64, Vec<u64>> = HashMap::new();
        let rows = stmt
            .query_map([image_rel_path], |row| {
                Ok((row.get::<_, u64>(0)?, row.get::<_, u64>(1)?))
            })
            .map_err(err)?;
        for row in rows {
            let (track, parent) = row.map_err(err)?;
            parents.entry(track).or_default().push(parent);
        }
        Ok(parents)
    }

    /// One row per track - see [`TrackSummaryFilter`].
    pub fn track_summary(
        &self,
        filter: &TrackSummaryFilter,
    ) -> Result<Vec<TrackSummary>, InternalErrors> {
        TrackSummaryQuery::prepare(self, filter)?.collect()
    }

    /// A sample of the tracks, each from its start - see [`TrackPathsFilter`].
    pub fn track_paths(&self, filter: &TrackPathsFilter) -> Result<TrackPaths, InternalErrors> {
        let summary = TrackSummaryFilter {
            columns: Vec::new(),
            ..filter.summary.clone()
        };
        TrackSummaryQuery::prepare(self, &summary)?.paths(filter.max_tracks)
    }

    /// The drift Measure Track Motion stored for an image plane (the
    /// `drift_x` / `drift_y` image values of the step that measured its
    /// tracks): (t, x, y) by time - empty without drift correction.
    pub(super) fn stored_drift(
        &self,
        image_rel_path: &str,
        z: u32,
    ) -> Result<Vec<(u32, f64, f64)>, InternalErrors> {
        let err = |e: duckdb::Error| InternalErrors::Io(e.to_string());
        if !self.has_track_motion()? {
            return Ok(Vec::new());
        }
        let mut stmt = self
            .connection()
            .prepare(
                "SELECT t_stack, \
                     max(value) FILTER (WHERE key LIKE '%/drift_x'), \
                     max(value) FILTER (WHERE key LIKE '%/drift_y') \
                 FROM measurements \
                 WHERE region = 'image' AND image_rel_path = ? AND z_stack = ? \
                     AND (key LIKE '%/drift_x' OR key LIKE '%/drift_y') \
                     AND step IN (SELECT DISTINCT step FROM measurements \
                         WHERE region = 'track' AND image_rel_path = ?) \
                 GROUP BY t_stack ORDER BY t_stack",
            )
            .map_err(err)?;
        let rows = stmt
            .query_map(duckdb::params![image_rel_path, z, image_rel_path], |row| {
                Ok((
                    row.get::<_, i64>(0)? as u32,
                    row.get::<_, Option<f64>>(1)?.unwrap_or(0.0),
                    row.get::<_, Option<f64>>(2)?.unwrap_or(0.0),
                ))
            })
            .map_err(err)?;
        rows.collect::<Result<_, _>>().map_err(err)
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

    /// A row per track - image by image, by id: its time points and its
    /// measurements' statistics. Untracked objects aren't one. Without
    /// Measure Track Motion it has no motion values (none is computed here).
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
        assert!(!seven.motion.is_measured());
        assert_eq!(
            (seven.speed_px_per_frame(), seven.straightness()),
            (None, None)
        );
        assert_eq!(seven.values, [Some(20.0), Some(30.0)]);
        assert!(seven.from_start && !seven.touches_edge);
        // The fixture's pixel size: 1 nm; no frame interval.
        assert_eq!((seven.um_per_px, seven.s_per_frame), (Some(0.001), None));
        assert_eq!(rows[1].class_name, "ClassB");
        let eight = &rows[2];
        assert_eq!((eight.points, eight.from_start), (1, false));
    }

    /// Measure Track Motion's stored values (`region = 'track'` rows) are
    /// each track's motion - the head says there are some.
    #[test]
    fn a_track_shows_the_motion_measure_track_motion_stored() {
        let db = open(
            &tracks(),
            3,
            "CREATE TABLE measurements (image_name VARCHAR, image_rel_path VARCHAR, \
             z_stack INTEGER, t_stack INTEGER, region VARCHAR, object_id UUID, \
             related_object_id UUID, object_class_name VARCHAR, object_class_id INTEGER[], \
             step VARCHAR, measurement VARCHAR, unit VARCHAR, key VARCHAR, value DOUBLE, \
             position DOUBLE, track_id UBIGINT);
             INSERT INTO measurements VALUES
                 ('img1.tif', 'img1.tif', 0, 0, 'track', NULL, NULL, NULL, [1], 'motion_01', \
                  'VCL', 'PixelsPerFrame', 'motion_01/vcl', 4.5, NULL, 7),
                 ('img1.tif', 'img1.tif', 0, 0, 'track', NULL, NULL, NULL, [1], 'motion_01', \
                  'LIN', 'Fraction', 'motion_01/lin', 0.25, NULL, 7),
                 ('img2.tif', 'img2.tif', 0, 1, 'track', NULL, NULL, NULL, [1], 'motion_01', \
                  'Path length', 'Pixels', 'motion_01/path_length', 0.0, NULL, 8),
                 ('img1.tif', 'img1.tif', 0, 0, 'image', NULL, NULL, NULL, NULL, 'motion_01', \
                  'Drift X', 'Pixels', 'motion_01/drift_x', 3.0, NULL, NULL);",
        );
        let query = TrackSummaryQuery::prepare(&db, &filter()).unwrap();
        assert!(query.head.motion);
        let rows = query.collect().unwrap();
        let seven = rows.iter().find(|r| r.track_id == 7).unwrap();
        assert_eq!(
            (seven.speed_px_per_frame(), seven.straightness()),
            (Some(4.5), Some(0.25))
        );
        assert_eq!(seven.motion.path_length, None);
        let eight = rows.iter().find(|r| r.track_id == 8).unwrap();
        assert_eq!(eight.motion.path_length, Some(0.0));
        assert!(
            !rows
                .iter()
                .find(|r| r.track_id == 9)
                .unwrap()
                .motion
                .is_measured()
        );
        // µm per minute: 1 nm pixels, a frame every 30 s.
        let um_per_min = seven.motion.in_units(
            TrackMotion::VCL,
            crate::workspace::settings::SpeedUnit::UmPerMin,
            Some(0.001),
            Some(30.0),
        );
        assert!((um_per_min.unwrap() - 4.5 * 0.001 * 2.0).abs() < 1e-12);
    }

    /// Paths: each picked track from its start, minus the drift the step
    /// stored for its image; beyond `max_tracks` an even sample.
    #[test]
    fn paths_start_at_zero_without_the_stored_drift() {
        let db = open(
            &tracks(),
            3,
            "CREATE TABLE measurements (image_name VARCHAR, image_rel_path VARCHAR, \
             z_stack INTEGER, t_stack INTEGER, region VARCHAR, object_id UUID, \
             related_object_id UUID, object_class_name VARCHAR, object_class_id INTEGER[], \
             step VARCHAR, measurement VARCHAR, unit VARCHAR, key VARCHAR, value DOUBLE, \
             position DOUBLE, track_id UBIGINT);
             INSERT INTO measurements VALUES
                 ('img1.tif', 'img1.tif', 0, 0, 'track', NULL, NULL, NULL, [1], 'motion_01', \
                  'VCL', 'PixelsPerFrame', 'motion_01/vcl', 4.5, NULL, 7),
                 ('img1.tif', 'img1.tif', 0, 0, 'image', NULL, NULL, NULL, NULL, 'motion_01', \
                  'Drift X', 'Pixels', 'motion_01/drift_x', 0.0, NULL, NULL),
                 ('img1.tif', 'img1.tif', 0, 1, 'image', NULL, NULL, NULL, NULL, 'motion_01', \
                  'Drift X', 'Pixels', 'motion_01/drift_x', 1.0, NULL, NULL),
                 ('img1.tif', 'img1.tif', 0, 2, 'image', NULL, NULL, NULL, NULL, 'motion_01', \
                  'Drift X', 'Pixels', 'motion_01/drift_x', 2.0, NULL, NULL);",
        );
        // As the step stores it: from 0 at the references' first time point.
        assert_eq!(
            db.stored_drift("img1.tif", 0).unwrap(),
            [(0, 0.0, 0.0), (1, 1.0, 0.0), (2, 2.0, 0.0)]
        );
        let (motions, drift) = db.track_motion_and_drift("img1.tif", 0).unwrap();
        assert_eq!(motions[&7].vcl, Some(4.5));
        assert_eq!(drift, [(0, 0.0, 0.0), (1, 1.0, 0.0), (2, 2.0, 0.0)]);
        let filter = |max_tracks| TrackPathsFilter {
            summary: filter(),
            max_tracks,
        };
        let paths = db.track_paths(&filter(10)).unwrap();
        assert_eq!(paths.total_tracks, 3);
        let seven = paths.tracks.iter().find(|t| t.track_id == 7).unwrap();
        // (0,0) (3,4) (3,0), minus drift x 0, 1, 2.
        assert_eq!(seven.points, [(0.0, 0.0), (2.0, 4.0), (1.0, 0.0)]);
        let eight = paths.tracks.iter().find(|t| t.track_id == 8).unwrap();
        assert_eq!(eight.points, [(0.0, 0.0)], "no drift in its image");
        let sampled = db.track_paths(&filter(2)).unwrap();
        assert_eq!((sampled.tracks.len(), sampled.total_tracks), (2, 3));
        assert_eq!(
            sampled,
            db.track_paths(&filter(2)).unwrap(),
            "the same sample each time"
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

    /// Track 1 divides into 2 and 3 at T 2; 3 and 4 fuse into 5 at T 4.
    fn family() -> Vec<ObjectSpec> {
        let at = |track: u64, t: i32| {
            ObjectSpec::new("img1.tif", "ClassA", 1, 10)
                .in_track(track)
                .at_plane(0, t)
        };
        vec![
            at(1, 0),
            at(1, 1),
            at(2, 2),
            at(2, 3),
            at(3, 2),
            at(3, 3),
            at(4, 2),
            at(4, 3),
            at(5, 4),
        ]
    }

    const FAMILY_SQL: &str = "ALTER TABLE objects ADD COLUMN parent_track_ids UBIGINT[];
         UPDATE objects SET parent_track_ids = [1] WHERE track_id IN (2, 3);
         UPDATE objects SET parent_track_ids = [3, 4] WHERE track_id = 5;";

    /// Each row knows its parents and children, its tree's root and how
    /// many divisions and fusions lie between, and how it starts and ends.
    #[test]
    fn a_track_knows_its_family() {
        use crate::api::LineageEvent::{Division, Fusion, None as Appears};
        let db = open(&family(), 5, FAMILY_SQL);
        let query = TrackSummaryQuery::prepare(&db, &filter()).unwrap();
        assert!(query.head.lineage);
        let rows = query.collect().unwrap();
        let of = |id: u64| &rows.iter().find(|r| r.track_id == id).unwrap().lineage;

        assert_eq!(
            (of(1).parents.clone(), of(1).children.clone()),
            (vec![], vec![2, 3])
        );
        assert_eq!(
            (of(1).starts_with(), of(1).ends_with()),
            (Appears, Division)
        );
        assert_eq!((of(1).root, of(1).generation), (1, 0));

        assert_eq!(of(2).parents, [1]);
        assert_eq!(
            (of(2).starts_with(), of(2).ends_with()),
            (Division, Appears)
        );
        assert_eq!((of(2).root, of(2).generation), (1, 1));

        assert_eq!(of(3).ends_with(), Fusion, "its child has another parent");
        assert_eq!((of(4).root, of(4).ends_with()), (4, Fusion));

        let five = of(5);
        assert_eq!(
            (five.parents.clone(), five.starts_with()),
            (vec![3, 4], Fusion)
        );
        assert_eq!(
            (five.root, five.generation),
            (1, 2),
            "the lowest root, the longest way up"
        );
    }

    /// Results without lineage (no column, or no parents): no lineage,
    /// every track its own root.
    #[test]
    fn results_without_divisions_have_no_lineage() {
        for sql in [
            "",
            "ALTER TABLE objects ADD COLUMN parent_track_ids UBIGINT[];",
        ] {
            let db = open(&family(), 5, sql);
            let query = TrackSummaryQuery::prepare(&db, &filter()).unwrap();
            assert!(!query.head.lineage, "{sql:?}");
            let rows = query.collect().unwrap();
            assert!(
                rows.iter()
                    .all(|r| r.lineage.parents.is_empty() && r.lineage.children.is_empty())
            );
            assert!(
                rows.iter().all(|r| r.lineage.root == r.track_id),
                "its own root"
            );
        }
    }

    #[test]
    fn a_broken_family_with_a_loop_still_ends() {
        let lineages = Lineages::new(HashMap::from([(1, vec![2]), (2, vec![1])]));
        let _ = lineages.of(1);
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
