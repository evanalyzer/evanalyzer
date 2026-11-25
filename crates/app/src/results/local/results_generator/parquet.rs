//! # parquet
//!
//! The results' raw tables as Parquet files: `objects.parquet` and, where
//! there are measurements, `measurements.parquet` - as they are stored, for
//! tools that read Parquet (`cli results export parquet`).
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-11-25
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.

use super::ResultsGenerator;
use evanalyzer_cfg::core_types::InternalErrors;
use std::path::{Path, PathBuf};

impl ResultsGenerator {
    /// Writes the raw tables into `out_dir` as Parquet; the files written.
    pub fn export_parquet(&self, out_dir: &Path) -> Result<Vec<PathBuf>, InternalErrors> {
        // SQL may write below the results file's staging folder only (no
        // file access otherwise - see `lock_down` in the core's duckdb.rs):
        // the files go there first, in a folder of this export's own (two
        // exports of one file at once), then to where they were asked for.
        let staging = self.export_staging_dir().ok_or_else(|| {
            InternalErrors::Internal("results in memory can't be written as Parquet".into())
        })?;
        static EXPORTS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let n = EXPORTS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let folder = staging.join(format!("parquet-{}-{n}", std::process::id()));
        std::fs::create_dir_all(&folder).map_err(|e| {
            InternalErrors::Io(format!("could not create {}: {e}", folder.display()))
        })?;
        let mut tables = vec![("objects", "objects.parquet")];
        // The measurements beside them, as they are (one row per value).
        if !self.get_measurement_columns()?.is_empty() {
            tables.push(("measurements", "measurements.parquet"));
        }
        let mut files = Vec::new();
        let written = tables.iter().try_for_each(|(table, file)| {
            let staged = folder.join(file);
            // DuckDB's `COPY` takes the destination as a string literal in
            // the SQL itself (not a bindable parameter): `'` doubled.
            let literal = staged.to_string_lossy().replace('\'', "''");
            // Built into DuckDB, loaded only here: no connection carries it
            // (~10 MB) when nothing is written as Parquet.
            self.database
                .execute_batch(&format!(
                    "LOAD parquet; COPY {table} TO '{literal}' (FORMAT parquet);"
                ))
                .map_err(|e| InternalErrors::Io(e.to_string()))?;
            let to = out_dir.join(file);
            move_file(&staged, &to)?;
            files.push(to);
            Ok::<(), InternalErrors>(())
        });
        // Nothing stays behind: this export's folder, and the staging folder
        // once no other export uses it.
        let _ = std::fs::remove_dir_all(&folder);
        let _ = std::fs::remove_dir(&staging);
        written?;
        Ok(files)
    }
}

/// Moves `from` to `to` - copied and removed where a rename can't (another
/// disk).
fn move_file(from: &Path, to: &Path) -> Result<(), InternalErrors> {
    let err =
        |e: std::io::Error| InternalErrors::Io(format!("could not write {}: {e}", to.display()));
    if std::fs::rename(from, to).is_ok() {
        return Ok(());
    }
    std::fs::copy(from, to).map_err(err)?;
    let _ = std::fs::remove_file(from);
    Ok(())
}
