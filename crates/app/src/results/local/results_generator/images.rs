//! # images
//!
//! The analysed images: their list, leaving one out, its note and marker.
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-11-22
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.

use super::*;

impl ResultsGenerator {
    pub fn get_images(&self) -> Result<Vec<ImageEntry>, InternalErrors> {
        let err = |e: duckdb::Error| InternalErrors::Io(e.to_string());
        let mut stmt = self
            .database
            .prepare(
                "SELECT image_name, image_rel_path, disabled, disabled_reason, disabled_by, \
                 coalesce(note, ''), marker FROM images ORDER BY image_name",
            )
            .map_err(err)?;
        let map = stmt
            .query_map([], |row| {
                Ok(ImageEntry {
                    name: row.get(0)?,
                    rel_path: PathBuf::from(row.get::<_, String>(1)?),
                    disabled: row.get(2)?,
                    disabled_reason: row.get(3)?,
                    disabled_by: row.get(4)?,
                    note: row.get(5)?,
                    marker: row
                        .get::<_, Option<i64>>(6)?
                        .and_then(ImageMarker::from_number),
                })
            })
            .map_err(err)?
            .collect::<Result<Vec<_>, _>>()
            .map_err(err);
        map
    }

    /// Every enabled image's `image_rel_path`, in the order the list and
    /// grouped-by-image queries sort by.
    pub(in crate::results::local) fn enabled_image_rel_paths(
        &self,
    ) -> Result<Vec<String>, InternalErrors> {
        let err = |e: duckdb::Error| InternalErrors::Io(e.to_string());
        let mut stmt = self
            .database
            .prepare("SELECT image_rel_path FROM images WHERE NOT disabled ORDER BY image_rel_path")
            .map_err(err)?;
        let paths = stmt
            .query_map([], |row| row.get(0))
            .map_err(err)?
            .collect::<Result<Vec<String>, _>>()
            .map_err(err);
        paths
    }

    /// Enables or disables `image_rel_path` (the `images` table's primary
    /// key) for statistics: a disabled image's own value is still shown
    /// everywhere it appears, but is excluded from any aggregate combining
    /// it with other images (see `get_group_by_plate`'s `value`/
    /// `any_disabled` and `Cell::disabled`). A no-op if `image_rel_path`
    /// doesn't match any row. Left out here, it is the user's: `disabled_by`
    /// 'results', without a reason until one is given
    /// ([`Self::change_image_review`]); taken back in, the reason goes.
    /// Leaving out an image already left out changes nothing.
    pub fn enable_image(&self, image_rel_path: &str, disable: bool) -> Result<(), InternalErrors> {
        let err = |e: duckdb::Error| InternalErrors::Io(e.to_string());
        let sql = if disable {
            "UPDATE images SET disabled = true, disabled_reason = NULL, disabled_by = 'results' \
             WHERE image_rel_path = ? AND NOT disabled"
        } else {
            "UPDATE images SET disabled = false, disabled_reason = NULL, disabled_by = NULL \
             WHERE image_rel_path = ?"
        };
        self.database
            .execute(sql, duckdb::params![image_rel_path])
            .map_err(err)?;
        Ok(())
    }

    /// Changes the note, marker or left-out reason of `image_rel_path` in
    /// these results. A reason only for a left-out image.
    pub fn change_image_review(
        &self,
        image_rel_path: &str,
        change: &ImageReviewChange,
    ) -> Result<(), InternalErrors> {
        let err = |e: duckdb::Error| InternalErrors::Io(e.to_string());
        let changed = match change {
            ImageReviewChange::Note(note) => self.database.execute(
                "UPDATE images SET note = ? WHERE image_rel_path = ?",
                duckdb::params![(!note.is_empty()).then_some(note.as_str()), image_rel_path],
            ),
            ImageReviewChange::Marker(marker) => self.database.execute(
                "UPDATE images SET marker = ? WHERE image_rel_path = ?",
                duckdb::params![marker.map(ImageMarker::number), image_rel_path],
            ),
            ImageReviewChange::LeftOutReason(reason) => self.database.execute(
                "UPDATE images SET disabled_reason = ? WHERE image_rel_path = ? AND disabled",
                duckdb::params![reason.as_deref().filter(|r| !r.is_empty()), image_rel_path],
            ),
        };
        changed.map_err(err)?;
        Ok(())
    }
}
