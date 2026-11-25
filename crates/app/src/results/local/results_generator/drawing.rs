//! # drawing
//!
//! What a picture of an image with its objects drawn needs from the
//! results: the planes its analysis ran on, and each object of a plane -
//! box, classes, mask - one at a time (a whole slide's objects never in
//! RAM at once). The drawing itself is the export's (`exports::local`).
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-11-25
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.

use super::ResultsGenerator;
use super::sql::class_filter_sql;
use crate::results::DrawnObject;
use evanalyzer_cfg::core_types::InternalErrors;

impl ResultsGenerator {
    /// The planes `(z, t)` of image `rel_path` its analysis ran on, by t
    /// then z. Results that don't record them: the planes its objects are
    /// on. Empty: none (an image without objects in older results).
    pub fn analysed_planes(&self, rel_path: &str) -> Result<Vec<(u32, u32)>, InternalErrors> {
        let err = |e: duckdb::Error| InternalErrors::Io(e.to_string());
        let source = match self.analysed_planes_recorded() {
            true => {
                "SELECT t AS t_stack, UNNEST(z_stacks_analyzed) AS z_stack \
                 FROM (SELECT UNNEST(t_stacks_analyzed) AS t, z_stacks_analyzed FROM images \
                       WHERE image_rel_path = ? AND t_stacks_analyzed IS NOT NULL)"
            }
            false => "SELECT t_stack, z_stack FROM objects WHERE image_rel_path = ?",
        };
        let mut stmt = self
            .database
            .prepare(&format!(
                "SELECT DISTINCT z_stack, t_stack FROM ({source}) \
                 WHERE t_stack >= 0 AND z_stack >= 0 ORDER BY t_stack, z_stack"
            ))
            .map_err(err)?;
        stmt.query_map([rel_path], |row| {
            Ok((row.get::<_, i64>(0)? as u32, row.get::<_, i64>(1)? as u32))
        })
        .map_err(err)?
        .collect::<Result<Vec<_>, _>>()
        .map_err(err)
    }

    /// Each object of image `rel_path` on plane `(z, t)` - of the classes
    /// `wanted` (`None`: all) - handed to `each` one at a time.
    pub fn for_each_drawn_object(
        &self,
        rel_path: &str,
        (z, t): (u32, u32),
        wanted: Option<&[u32]>,
        each: &mut dyn FnMut(DrawnObject) -> Result<(), InternalErrors>,
    ) -> Result<(), InternalErrors> {
        let err = |e: duckdb::Error| InternalErrors::Io(e.to_string());
        let mask_expr = match self.has_masks()? {
            true => "mask",
            false => "NULL::BLOB",
        };
        let volume_expr = match self.has_volume_columns()? {
            true => {
                "CASE WHEN ndim = 3 THEN bbox_zmin_px END, \
                     CASE WHEN ndim = 3 THEN bbox_zmax_px - bbox_zmin_px + 1 END"
            }
            false => "NULL::UINTEGER, NULL::UINTEGER",
        };
        let class_filter = match wanted {
            Some([]) => return Ok(()),
            Some(ids) => format!(" AND {}", class_filter_sql("object_class_id", ids)),
            None => String::new(),
        };
        let sql = format!(
            "SELECT bbox_xmin_px, bbox_ymin_px, bbox_xmax_px, bbox_ymax_px, \
             CAST(object_class_id AS INTEGER[]), {mask_expr}, {volume_expr} \
             FROM objects WHERE image_rel_path = ? AND {} AND t_stack = ?{class_filter}",
            self.object_on_plane_sql("", z)?
        );
        let mut stmt = self.database.prepare(&sql).map_err(err)?;
        let mut rows = stmt
            .query(duckdb::params![rel_path, t as i64])
            .map_err(err)?;
        while let Some(row) = rows.next().map_err(err)? {
            each(DrawnObject {
                bbox: [
                    row.get(0).map_err(err)?,
                    row.get(1).map_err(err)?,
                    row.get(2).map_err(err)?,
                    row.get(3).map_err(err)?,
                ],
                classes: int_list(row.get(4).map_err(err)?),
                mask: row.get(5).map_err(err)?,
                volume: (row.get(6).map_err(err)?, row.get(7).map_err(err)?),
            })?;
        }
        Ok(())
    }
}

/// A DuckDB `INTEGER[]` as a list - empty for NULL.
fn int_list(value: duckdb::types::Value) -> Vec<i32> {
    match value {
        duckdb::types::Value::List(values) => values
            .into_iter()
            .filter_map(|v| match v {
                duckdb::types::Value::Int(i) => Some(i),
                _ => None,
            })
            .collect(),
        _ => Vec::new(),
    }
}
