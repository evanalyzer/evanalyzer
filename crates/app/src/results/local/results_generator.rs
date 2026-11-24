//! # results_generator
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-09-14
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.

use crate::results::plate_grouping::*;
use crate::results::value_to_color;
use crate::results::*;
use duckdb::Connection;
use duckdb::types::Value;
use evanalyzer_cfg::{
    core_types::{InternalErrors, ObjectClass},
    settings::{classification_settings::Class, images_settings::ImageMarker},
};
use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::path::PathBuf;

/// How a query extracts the well of an image (see [`Grouping`]): a regex
/// (already quote-escaped for a SQL string literal) on `column` of the
/// `objects`/`images` row.
pub(super) struct GroupingSql {
    pub(super) column: &'static str,
    pub(super) regex: String,
}

pub struct ResultsGenerator {
    database: evanalyzer_core::ResultsConnection,
    classes_cache: RefCell<Option<Vec<Class>>>,
    /// The partner class sets objects have, and the coloc groups
    /// (`get_coloc_class_sets`, `get_coloc_groups`) - read once.
    coloc_classes_cache: RefCell<Option<(Vec<Vec<ObjectClass>>, Vec<Vec<ObjectClass>>)>>,
    /// See `classes_with_objects`.
    classes_with_objects_cache: RefCell<Option<Vec<u32>>>,
    /// The pattern `Grouping::Auto` picked for this database's images.
    auto_grouping_cache: RefCell<Option<String>>,
    /// The `objects` table's columns - fixed when the results are written.
    object_columns_cache: RefCell<Option<HashSet<String>>>,
    /// Whether the results hold a volume's objects (`has_volumes`).
    volumes_cache: RefCell<Option<bool>>,
    /// Whether the results can hold profiles (format 2, `has_profiles`).
    profiles_cache: RefCell<Option<bool>>,
    /// Whether Measure Track Motion stored values (`has_track_motion`).
    pub(super) track_motion_cache: RefCell<Option<bool>>,
    /// The (parent class, child class) pairs (`get_relations`) - fixed when
    /// the results are written; asked for by the columns and Trees.
    relations_cache: RefCell<Option<Vec<(ObjectClass, ObjectClass)>>>,
}

mod columns;
mod grouped;
mod images;
mod list;
mod measurement_rows;
mod plate;
mod sql;
#[cfg(test)]
mod tests;

// What the views share, reachable here (and in `tests`) as before.
pub(crate) use self::sql::class_display_label;
use self::{list::*, sql::*};
pub(super) use self::{
    sql::ImageSelection, sql::class_blocks, sql::class_filter_sql, sql::column_aggregate_expr,
    sql::objects_per_class_sql, sql::sql_int_array_literal, sql::sql_string_in_list,
};

impl ResultsGenerator {
    pub fn open_database(path: PathBuf) -> Result<Self, InternalErrors> {
        let database = evanalyzer_core::open_results_database(&path)?;
        evanalyzer_core::reject_newer_results(&database)?;
        evanalyzer_core::add_review_columns(&database)?;
        Ok(Self {
            database,
            classes_cache: RefCell::new(None),
            coloc_classes_cache: RefCell::new(None),
            classes_with_objects_cache: RefCell::new(None),
            auto_grouping_cache: RefCell::new(None),
            object_columns_cache: RefCell::new(None),
            volumes_cache: RefCell::new(None),
            profiles_cache: RefCell::new(None),
            relations_cache: RefCell::new(None),
            track_motion_cache: RefCell::new(None),
        })
    }

    /// Whether the results can hold profiles (format 2:
    /// `measurement_info.profile` and `measurements.position`) - one query,
    /// cached: the columns of a results file don't change.
    pub(super) fn has_profiles(&self) -> Result<bool, InternalErrors> {
        if let Some(cached) = *self.profiles_cache.borrow() {
            return Ok(cached);
        }
        let both: bool = self
            .database
            .query_row(
                "SELECT count(*) = 2 FROM information_schema.columns WHERE \
                 (table_name = 'measurement_info' AND column_name = 'profile') OR \
                 (table_name = 'measurements' AND column_name = 'position')",
                [],
                |row| row.get(0),
            )
            .map_err(|e| InternalErrors::Io(e.to_string()))?;
        *self.profiles_cache.borrow_mut() = Some(both);
        Ok(both)
    }

    /// The regex `grouping` uses on this database's images - for
    /// `Grouping::Auto` the pattern detected (see
    /// [`AUTO_GROUPING_PATTERNS`]).
    pub fn grouping_regex(&self, grouping: &Grouping) -> Result<String, InternalErrors> {
        Ok(match grouping {
            Grouping::Regex(regex) if !regex.trim().is_empty() => regex.clone(),
            Grouping::Regex(_) | Grouping::Auto => self.detect_grouping_regex()?,
            Grouping::Folder => FOLDER_GROUPING_REGEX.to_string(),
        })
    }

    pub(super) fn grouping_sql(&self, grouping: &Grouping) -> Result<GroupingSql, InternalErrors> {
        Ok(GroupingSql {
            column: match grouping {
                Grouping::Folder => "image_rel_path",
                _ => "image_name",
            },
            regex: self.grouping_regex(grouping)?.replace('\'', "''"),
        })
    }

    /// The first of [`AUTO_GROUPING_PATTERNS`] that matches the most image
    /// names (the first one if none matches any). Cached: the images of a
    /// results database don't change.
    fn detect_grouping_regex(&self) -> Result<String, InternalErrors> {
        if let Some(cached) = self.auto_grouping_cache.borrow().as_ref() {
            return Ok(cached.clone());
        }
        let counts = AUTO_GROUPING_PATTERNS
            .iter()
            .map(|pattern| {
                format!(
                    "count(*) FILTER (WHERE regexp_matches(image_name, '{}'))",
                    pattern.replace('\'', "''")
                )
            })
            .collect::<Vec<_>>()
            .join(", ");
        let matches: Vec<i64> = self
            .database
            .query_row(&format!("SELECT {counts} FROM images"), [], |row| {
                (0..AUTO_GROUPING_PATTERNS.len())
                    .map(|i| row.get::<_, i64>(i))
                    .collect()
            })
            .map_err(|e| InternalErrors::Io(e.to_string()))?;
        let mut best = 0;
        for (i, count) in matches.iter().enumerate() {
            if *count > matches[best] {
                best = i;
            }
        }
        let regex = AUTO_GROUPING_PATTERNS[best].to_string();
        *self.auto_grouping_cache.borrow_mut() = Some(regex.clone());
        Ok(regex)
    }

    /// Whether the run recorded which planes of each image it analysed
    /// (`images.t_stacks_analyzed`/`z_stacks_analyzed`) - results written before that
    /// don't have them.
    pub(super) fn analysed_planes_recorded(&self) -> bool {
        self.database
            .query_row(
                "SELECT COUNT(*) FROM images WHERE t_stacks_analyzed IS NOT NULL",
                [],
                |row| row.get::<_, i64>(0),
            )
            .is_ok_and(|count| count > 0)
    }

    /// SQL condition that the `images` row `alias` was analysed
    /// successfully on `plane` - i.e. a missing object there really means
    /// "none found", so a Count/Sum of 0 is a real result (see
    /// [`fill_zero_sql`]). A failed image, or a plane the run never
    /// analysed, has no value.
    ///
    /// The run records the planes it analysed of each image
    /// (`images.t_stacks_analyzed`/`z_stacks_analyzed`) - also those where nothing was
    /// found. Results
    /// from before that have them taken from the objects themselves: every
    /// plane between the lowest and highest Z/T any object was found on (a
    /// Z-projection run only produces plane 0), and within the image's own
    /// plane count. So there a plane at the edge of the analysed range on
    /// which no image found anything at all, or a run without a single
    /// object, counts as not analysed (empty).
    pub(super) fn measured_on_plane_sql(
        &self,
        alias: &str,
        plane: &PlaneFilter,
    ) -> Result<String, InternalErrors> {
        let (z, t) = (plane.z_stack as i64, plane.t_stack as i64);
        if self.analysed_planes_recorded() {
            return Ok(format!(
                "({alias}.successful AND list_contains({alias}.t_stacks_analyzed, {t}) \
                 AND list_contains({alias}.z_stacks_analyzed, {z}))"
            ));
        }
        let range: Option<(i64, i64, i64, i64)> = self
            .database
            .query_row(
                "SELECT MIN(z_stack), MAX(z_stack), MIN(t_stack), MAX(t_stack) FROM objects",
                [],
                |row| {
                    Ok(match (row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?) {
                        (Some(z_min), Some(z_max), Some(t_min), Some(t_max)) => {
                            Some((z_min, z_max, t_min, t_max))
                        }
                        _ => None,
                    })
                },
            )
            .map_err(|e| InternalErrors::Io(e.to_string()))?;
        Ok(match range {
            Some((z_min, z_max, t_min, t_max))
                if (z_min..=z_max).contains(&z) && (t_min..=t_max).contains(&t) =>
            {
                format!(
                    "({alias}.successful AND {z} < {alias}.z_stacks AND {t} < {alias}.t_stacks)"
                )
            }
            _ => "false".to_string(),
        })
    }

    /// A second, independent connection to the same already-open database -
    /// for handing to a background thread (e.g. a potentially long-running
    /// export) that shouldn't have to either hold this `ResultsGenerator`'s
    /// caller's lock for its own duration, or reopen the underlying file at
    /// the OS level. Reopening the same path with a fresh `open_database`
    /// call instead of cloning was the original approach: it works on
    /// Linux/macOS, but not Windows, where the OS enforces exclusive-by-
    /// default file locking even for a second handle opened by the same
    /// process - the app ended up locking itself out of its own database
    /// on every export.
    pub fn try_clone(&self) -> Result<Self, InternalErrors> {
        Ok(Self {
            database: self.database.try_clone()?,
            classes_cache: RefCell::new(None),
            coloc_classes_cache: RefCell::new(None),
            classes_with_objects_cache: RefCell::new(None),
            auto_grouping_cache: RefCell::new(None),
            object_columns_cache: RefCell::new(None),
            volumes_cache: RefCell::new(None),
            profiles_cache: RefCell::new(None),
            relations_cache: RefCell::new(None),
            track_motion_cache: RefCell::new(None),
        })
    }

    /// Raw DB handle
    pub(super) fn connection(&self) -> &Connection {
        &self.database
    }

    /// The one folder SQL on this database may write to (see
    /// `evanalyzer_core::storage::duckdb::export_staging_dir`); `None` in
    /// memory.
    pub(super) fn export_staging_dir(&self) -> Option<std::path::PathBuf> {
        self.database.export_staging_dir()
    }

    /// Whether these results store parents (`parent_ids`) - not those from
    /// before Relate Objects.
    pub fn has_parent_ids(&self) -> Result<bool, InternalErrors> {
        self.has_object_column("parent_ids")
    }

    /// The results keep the objects' masks (results from before they did,
    /// or written with them switched off, have no or empty `mask`s).
    pub fn has_masks(&self) -> Result<bool, InternalErrors> {
        self.has_object_column("mask")
    }

    /// Whether the results have the z columns (`ndim`, `bbox_zmin_px`, …) -
    /// not results from before them.
    pub(super) fn has_volume_columns(&self) -> Result<bool, InternalErrors> {
        self.has_object_column("ndim")
    }

    /// Whether these results hold a volume's objects (3D) - none in results
    /// from before the z columns. Cached: the objects don't change.
    fn has_volumes(&self) -> Result<bool, InternalErrors> {
        if let Some(cached) = *self.volumes_cache.borrow() {
            return Ok(cached);
        }
        let has = self.has_object_column("ndim")?
            && self
                .database
                .query_row(
                    "SELECT EXISTS (SELECT 1 FROM objects WHERE ndim = 3)",
                    [],
                    |row| row.get(0),
                )
                .map_err(|e| InternalErrors::Io(e.to_string()))?;
        *self.volumes_cache.borrow_mut() = Some(has);
        Ok(has)
    }

    /// The condition for objects on z-plane `z` (of `objects` aliased as
    /// `alias`, "" for none): a volume's object is on each plane from its
    /// `bbox_zmin_px` to its `bbox_zmax_px`. Results from before the z
    /// columns only have `z_stack`.
    pub(super) fn object_on_plane_sql(
        &self,
        alias: &str,
        z: u32,
    ) -> Result<String, InternalErrors> {
        let column = |name: &str| match alias {
            "" => name.to_string(),
            alias => format!("{alias}.{name}"),
        };
        Ok(match self.has_object_column("bbox_zmin_px")? {
            true => format!(
                "{z} BETWEEN {} AND {}",
                column("bbox_zmin_px"),
                column("bbox_zmax_px")
            ),
            false => format!("{} = {z}", column("z_stack")),
        })
    }

    pub(super) fn has_object_column(&self, column: &str) -> Result<bool, InternalErrors> {
        if let Some(columns) = &*self.object_columns_cache.borrow() {
            return Ok(columns.contains(column));
        }
        let err = |e: duckdb::Error| InternalErrors::Io(e.to_string());
        let mut stmt = self
            .database
            .prepare(
                "SELECT column_name FROM information_schema.columns \
                 WHERE table_name = 'objects'",
            )
            .map_err(err)?;
        let columns: HashSet<String> = stmt
            .query_map([], |row| row.get(0))
            .map_err(err)?
            .collect::<Result<_, _>>()
            .map_err(err)?;
        let has = columns.contains(column);
        *self.object_columns_cache.borrow_mut() = Some(columns);
        Ok(has)
    }

    /// Whether `table` has `column` - one added by a later results format.
    pub(super) fn has_column(&self, table: &str, column: &str) -> Result<bool, InternalErrors> {
        self.database
            .query_row(
                "SELECT count(*) > 0 FROM information_schema.columns \
                 WHERE table_name = ? AND column_name = ?",
                [table, column],
                |row| row.get(0),
            )
            .map_err(|e| InternalErrors::Io(e.to_string()))
    }

    pub fn get_nr_of_c_stacks(&self) -> u32 {
        let max_stacks: u32 = self
            .database
            .query_row("SELECT MAX(c_stacks) FROM images;", [], |row| row.get(0))
            .unwrap_or(1);
        max_stacks
    }

    pub fn get_nr_of_z_stacks(&self) -> u32 {
        let max_stack: u32 = self
            .database
            .query_row("SELECT MAX(z_stacks) FROM images;", [], |row| row.get(0))
            .unwrap_or(1);
        max_stack
    }

    /// The planes (time point, z-plane) the run analysed - also those where
    /// nothing was found - ordered by time point, then z-plane. Results
    /// from before the run recorded them (`images.t_stacks_analyzed`/`z_stacks_analyzed`):
    /// the planes with objects.
    pub fn get_planes(&self) -> Result<Vec<PlaneFilter>, InternalErrors> {
        // Each image's time points x its z-planes.
        let planes = match self.analysed_planes_recorded() {
            true => {
                "SELECT t AS t_stack, UNNEST(z_stacks_analyzed) AS z_stack \
                 FROM (SELECT UNNEST(t_stacks_analyzed) AS t, z_stacks_analyzed FROM images \
                       WHERE t_stacks_analyzed IS NOT NULL)"
            }
            false => "SELECT t_stack, z_stack FROM objects",
        };
        let mut statement = self
            .database
            .prepare(&format!(
                "SELECT DISTINCT t_stack, z_stack FROM ({planes}) \
                 WHERE t_stack >= 0 AND z_stack >= 0 ORDER BY t_stack, z_stack;"
            ))
            .map_err(|e| InternalErrors::Io(e.to_string()))?;
        let planes = statement
            .query_map([], |row| {
                Ok(PlaneFilter {
                    t_stack: row.get::<_, i64>(0)? as u32,
                    z_stack: row.get::<_, i64>(1)? as u32,
                })
            })
            .map_err(|e| InternalErrors::Io(e.to_string()))?
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| InternalErrors::Io(e.to_string()))?;
        Ok(planes)
    }

    pub fn get_nr_of_t_stacks(&self) -> u32 {
        let max_stack: u32 = self
            .database
            .query_row("SELECT MAX(t_stacks) FROM images;", [], |row| row.get(0))
            .unwrap_or(1);
        max_stack
    }

    /// From the `run` table the analysis writes; `Unknown` for a database
    /// written before it existed.
    pub fn run_status(&self) -> Result<crate::results::RunStatus, InternalErrors> {
        use crate::results::RunStatus;
        let has_table: bool = self
            .database
            .query_row(
                "SELECT count(*) > 0 FROM information_schema.tables WHERE table_name = 'run'",
                [],
                |row| row.get(0),
            )
            .map_err(|e| InternalErrors::Io(e.to_string()))?;
        if !has_table {
            return Ok(RunStatus::Unknown);
        }
        let row: Option<(String, Option<String>)> = self
            .database
            .query_row("SELECT status, message FROM run LIMIT 1", [], |row| {
                Ok((row.get(0)?, row.get(1)?))
            })
            .map(Some)
            .or_else(|e| match e {
                duckdb::Error::QueryReturnedNoRows => Ok(None),
                e => Err(InternalErrors::Io(e.to_string())),
            })?;
        Ok(match row {
            None => RunStatus::Unknown,
            Some((status, message)) => match status.as_str() {
                "finished" => RunStatus::Finished,
                "cancelled" => RunStatus::Cancelled,
                "failed" => RunStatus::Failed(message.unwrap_or_default()),
                _ => RunStatus::NotFinished,
            },
        })
    }
}
