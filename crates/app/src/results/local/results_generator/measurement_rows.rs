//! # measurement_rows
//!
//! The measurement values one by one (`measurements` table) - for an
//! export's table of measurements.
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-11-24
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.

use super::ResultsGenerator;
use super::sql::sql_string_in_list;
use crate::results::{MeasurementRow, MeasurementRowsFilter};
use evanalyzer_cfg::core_types::InternalErrors;

impl ResultsGenerator {
    /// A page of the measurement values `filter` asks for, by image, time
    /// point, z-plane, step, key, region and object. None in results from
    /// before measurement steps existed.
    pub fn measurement_rows(
        &self,
        filter: &MeasurementRowsFilter,
    ) -> Result<Vec<MeasurementRow>, InternalErrors> {
        let err = |e: duckdb::Error| InternalErrors::Io(e.to_string());
        if !self.has_table("measurements")? {
            return Ok(Vec::new());
        }
        let mut conditions = vec!["true".to_string()];
        if let Some(images) = &filter.images {
            conditions.push(if images.is_empty() {
                "false".to_string()
            } else {
                format!("image_rel_path IN ({})", sql_string_in_list(images))
            });
        }
        if let Some(keys) = &filter.keys {
            conditions.push(if keys.is_empty() {
                "false".to_string()
            } else {
                format!("key IN ({})", sql_string_in_list(keys))
            });
        }
        let with_position = self.has_column("measurements", "position")?;
        let position = if with_position {
            "position"
        } else {
            "NULL::DOUBLE"
        };
        let sql = format!(
            "SELECT image_rel_path, t_stack, z_stack, region, object_id::VARCHAR, \
             related_object_id::VARCHAR, object_class_name, step, measurement, unit, key, \
             value, {position} \
             FROM measurements WHERE {} \
             ORDER BY image_rel_path, t_stack, z_stack, step, key, region, object_id, {position} \
             LIMIT {} OFFSET {}",
            conditions.join(" AND "),
            filter.limit.max(1),
            filter.offset,
        );
        let mut stmt = self.connection().prepare(&sql).map_err(err)?;
        let rows = stmt
            .query_map([], |row| {
                Ok(MeasurementRow {
                    image: row.get(0)?,
                    t: row.get(1)?,
                    z: row.get(2)?,
                    region: row.get(3)?,
                    object_id: row.get(4)?,
                    related_object_id: row.get(5)?,
                    class: row.get(6)?,
                    step: row.get(7)?,
                    measurement: row.get(8)?,
                    unit: row.get(9)?,
                    key: row.get(10)?,
                    value: row.get(11)?,
                    position: row.get(12)?,
                })
            })
            .map_err(err)?
            .collect::<Result<Vec<_>, _>>()
            .map_err(err)?;
        Ok(rows)
    }

    /// Whether the database has the table `table`.
    fn has_table(&self, table: &str) -> Result<bool, InternalErrors> {
        self.connection()
            .query_row(
                "SELECT count(*) > 0 FROM information_schema.tables WHERE table_name = ?",
                [table],
                |row| row.get(0),
            )
            .map_err(|e| InternalErrors::Io(e.to_string()))
    }
}
