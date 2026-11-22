//! # tables
//!
//! The object list and the measurements as tables (CSV, XLSX, Parquet),
//! page by page.
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-11-22
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.

use super::*;

impl ResultExport {
    // `list.xlsx`/`list.csv`: every object, ordered exactly as requested —
    // every object for image0 at t0, then image1 at t0, ..., then image0 at
    // t1, image1 at t1, .... `get_list`'s own SQL only orders by `object_id`
    // and only ever filters one (z, t) pair at a time, so that ordering is
    // produced here by calling it once per (z, t, image) triple, in
    // `z`/`t`/name-sorted order, and concatenating — not by a single broader
    // query. For XLSX, a second "List (Coloc Details)" sheet is added
    // alongside it when `with_list_coloc_details` is also set; for CSV
    // (which has no concept of a second sheet in the same file) that becomes
    // a second `list_coloc_details.csv` document instead.
    //
    // `with_list_one_file_per_image` splits this into `list_{image}.xlsx`/
    // `.csv` per image instead — each one its own independent single-image,
    // multi-t/z document — so a dataset whose combined row count would blow
    // past Excel's 1,048,576-row-per-sheet limit in one shared file still
    // exports cleanly, as long as no single image alone exceeds it (CSV has
    // no such limit, but honors the same flag for output-shape parity
    // between formats).
    pub(super) fn export_list(
        &self,
        database: &ResultsGenerator,
        cancel: &AtomicBool,
        on_progress: ExportProgress,
    ) -> Result<(), InternalErrors> {
        let images = resolve_images(database, self)?;
        let object_classes = if self.object_classes.is_empty() {
            None
        } else {
            Some(self.object_classes.clone())
        };

        if self.with_list_one_file_per_image {
            for (image_idx, image) in images.iter().enumerate() {
                check_cancelled(cancel)?;
                on_progress(
                    &format!("Exporting List: {image}"),
                    image_idx + 1,
                    images.len(),
                );
                let single_image = std::slice::from_ref(image);
                let stem = sanitize_filename_component(&image_stub(image));
                self.write_list_document(
                    database,
                    cancel,
                    ImageSelection::Listed(single_image),
                    &object_classes,
                    &format!("list_{stem}"),
                )?;
            }
            return Ok(());
        }

        on_progress("Exporting List view", 0, 1);
        self.write_list_document(
            database,
            cancel,
            self.image_selection(&images),
            &object_classes,
            "list",
        )?;
        on_progress("Exporting List view", 1, 1);
        Ok(())
    }

    /// Writes one List document (plus, when `with_list_coloc_details` is
    /// set, its coloc-details companion) under `file_stem` — `list.xlsx` /
    /// `list.xlsx`'s "List (Coloc Details)" sheet for XLSX, or
    /// `{file_stem}.csv` / `{file_stem}_coloc_details.csv` for CSV. Shared by
    /// both branches of `export_list` (whole-database and
    /// one-file-per-image) so they can't drift on how a document's content is
    /// built, only on which `images` slice and `file_stem` they pass in.
    pub(super) fn write_list_document(
        &self,
        database: &ResultsGenerator,
        cancel: &AtomicBool,
        images: ImageSelection<'_>,
        object_classes: &Option<Vec<ObjectClass>>,
        file_stem: &str,
    ) -> Result<(), InternalErrors> {
        match self.format {
            ExportFormat::XLSX => {
                let mut workbook = Workbook::new();
                // `write_list_sheet` only ever writes strictly top-to-bottom
                // (one page's rows after another, never revisiting an
                // earlier row) - exactly what "constant memory" mode
                // requires, and what lets it flush each row to a tempfile
                // as soon as the next one is written instead of keeping
                // every cell in memory for the life of the `Workbook`. A
                // standard `add_worksheet()` here would otherwise scale
                // memory with the row count writen - `rust_xlsxwriter`'s
                // own numbers put a 1M-row-ish sheet at ~200+ MB *before*
                // accounting for column count (cost is roughly per-cell),
                // which a single large image's object list can blow past
                // well under Excel's 1,048,576-row sheet cap.
                let sheet = workbook
                    .add_worksheet_with_constant_memory()
                    .map_err(xlsx_err)?;
                sheet.set_name("List").map_err(xlsx_err)?;
                write_list_sheet(sheet, database, cancel, self, images, object_classes, false)?;

                if self.with_list_coloc_details {
                    let coloc_sheet = workbook
                        .add_worksheet_with_constant_memory()
                        .map_err(xlsx_err)?;
                    coloc_sheet
                        .set_name("List (Coloc Details)")
                        .map_err(xlsx_err)?;
                    write_list_sheet(
                        coloc_sheet,
                        database,
                        cancel,
                        self,
                        images,
                        object_classes,
                        true,
                    )?;
                }

                workbook
                    .save(self.output_path(&format!("{file_stem}.xlsx")))
                    .map_err(xlsx_err)?;
            }
            ExportFormat::CSV => {
                write_list_csv(
                    database,
                    cancel,
                    self,
                    images,
                    object_classes,
                    false,
                    &self.output_path(&format!("{file_stem}.csv")),
                )?;
                if self.with_list_coloc_details {
                    write_list_csv(
                        database,
                        cancel,
                        self,
                        images,
                        object_classes,
                        true,
                        &self.output_path(&format!("{file_stem}_coloc_details.csv")),
                    )?;
                }
            }
            // `start_export` returns early for `Parquet` before ever calling
            // `export_list`/`write_list_document` - reachable only if
            // something calls this directly, so this stays a real error
            // rather than a panic.
            ExportFormat::Parquet => {
                return Err(InternalErrors::Internal(
                    "Parquet export doesn't support the List view — call export_as_parquet directly instead".to_string(),
                ));
            }
        }
        Ok(())
    }

    // `plate.xlsx` and `well.xlsx`: one tab per object class in each
    // document, every tab stacking one square, colored grid block per
    // (column, aggregation) combination (and, one level further down in
    // `well.xlsx`, per well on top of that) — see `write_grid_block`.
    // Independently toggled by `with_plate_view_heatmap`/
    // `with_well_view_heatmap` - either can run without the other, each
    // skipping that side's own query and file entirely when off. The
    // flat-list form of the same data (see `export_plate_and_well_as_flat_list`)
    // is a separate pass into separate files, not appended here — this
    // function only ever needs `View::Heatmap` data.
    //
    // Only the first z/t in `self.z_stacks`/`self.t_stacks` is rendered:
    // unlike the List export, a plate/well grid is inherently a single
    // plane's snapshot (matching the GUI's own Matrix view, which shows one
    // z/t at a time via the global stepper, not a time series) — there's no
    // "stack every t one after another" equivalent for a grid the way
    // there is for a flat table.
    // `time_course.csv`/`.xlsx`: per time point (and z-plane), the
    // statistics of `columns` over each line's objects - long (a row per
    // time point × line × class) and/or wide (per value and statistic: a row
    // per time point, a column per line; CSV a file each, XLSX a sheet).
    /// `tracks.csv` / a "Tracks" sheet: a row per track (see
    /// `TrackColumns`), written as the tracks are read.
    /// `measurements.{csv,xlsx}`: one row per measurement value with what
    /// it is about - the results' `measurements` table, its columns named
    /// the same, of the chosen images and planes and the measurements among
    /// `columns` (none chosen: all). The long layout R or Python read
    /// without reshaping, and every new measurement fits without new
    /// columns.
    pub(super) fn export_measurements(
        &self,
        database: &ResultsGenerator,
        cancel: &AtomicBool,
        on_progress: ExportProgress,
    ) -> Result<(), InternalErrors> {
        let err = |e: duckdb::Error| InternalErrors::Io(e.to_string());
        check_cancelled(cancel)?;
        on_progress("Exporting Measurements", 0, 1);
        if database.get_measurement_columns()?.is_empty() {
            return Err(InternalErrors::InvalidArgument(
                "These results have no measurements - a pipeline needs a measurement step"
                    .to_string(),
            ));
        }
        let images = resolve_images(database, self)?;
        let mut conditions = vec![match images.is_empty() {
            true => "false".to_string(),
            false => format!("image_rel_path IN ({})", sql_string_in_list(&images)),
        }];
        let keys: Vec<String> = self
            .columns
            .iter()
            .filter_map(|column| match column {
                Column::Measurement(m) => Some(m.key.clone()),
                _ => None,
            })
            .collect();
        if !keys.is_empty() {
            conditions.push(format!("key IN ({})", sql_string_in_list(&keys)));
        }
        for (column, range) in [("t_stack", &self.t_stacks), ("z_stack", &self.z_stacks)] {
            if let Some(range) = range {
                conditions.push(format!(
                    "{column} >= {} AND {column} < {}",
                    range.start, range.end
                ));
            }
        }
        // `position` (a profile's, e.g. a Sholl ring's radius) since format 2.
        let with_position = database.has_column("measurements", "position")?;
        let mut columns = vec![
            "image_name",
            "image_rel_path",
            "z_stack",
            "t_stack",
            "region",
            "object_id",
            "related_object_id",
            "object_class_name",
            "step",
            "measurement",
            "unit",
            "key",
            "value",
        ];
        if with_position {
            columns.push("position");
        }
        let sql = format!(
            "SELECT image_name, image_rel_path, z_stack, t_stack, region, object_id::VARCHAR, \
             related_object_id::VARCHAR, object_class_name, step, measurement, unit, key, value\
             {} FROM measurements WHERE {} \
             ORDER BY image_name, t_stack, z_stack, step, key, region, object_id{}",
            if with_position { ", position" } else { "" },
            conditions.join(" AND "),
            if with_position { ", position" } else { "" },
        );
        let mut stmt = database.connection().prepare(&sql).map_err(err)?;
        let mut each_row = |out: &mut dyn FieldSink| -> Result<(), InternalErrors> {
            for name in &columns {
                out.text(name)?;
            }
            out.end_row()?;
            let mut rows = stmt.query([]).map_err(err)?;
            let mut n = 0usize;
            while let Some(row) = rows.next().map_err(err)? {
                n += 1;
                if n.is_multiple_of(65_536) {
                    check_cancelled(cancel)?;
                }
                for i in 0..columns.len() {
                    match i {
                        // z: NULL for a value of a whole volume (an empty cell).
                        2 => {
                            out.optional(row.get::<_, Option<i32>>(i).map_err(err)?.map(f64::from))?
                        }
                        3 => out.number(row.get::<_, i32>(i).map_err(err)? as f64)?,
                        12 | 13 => out.optional(row.get::<_, Option<f64>>(i).map_err(err)?)?,
                        _ => match row.get::<_, Option<String>>(i).map_err(err)? {
                            Some(text) => out.text(&text)?,
                            None => out.empty()?,
                        },
                    }
                }
                out.end_row()?;
            }
            Ok(())
        };
        match self.format {
            ExportFormat::XLSX => {
                let mut workbook = Workbook::new();
                let worksheet = workbook
                    .add_worksheet_with_constant_memory()
                    .map_err(xlsx_err)?;
                worksheet.set_name("Measurements").map_err(xlsx_err)?;
                each_row(&mut FieldsSheet::new(worksheet))?;
                workbook
                    .save(self.output_path("measurements.xlsx"))
                    .map_err(xlsx_err)?;
            }
            ExportFormat::CSV => {
                let path = self.output_path("measurements.csv");
                each_row(&mut FieldsCsv::create(&path)?)?;
            }
            // Written with the objects (`export_as_parquet`).
            ExportFormat::Parquet => {}
        }
        on_progress("Exporting Measurements", 1, 1);
        Ok(())
    }

    /// `objects.parquet`: the entire `objects` table, every column, no z/t/
    /// image/class filtering and no column selection — DuckDB's own `COPY`
    /// writes the file directly, so this bypasses `get_object_list`/
    /// `stream_list_pages` entirely (unlike every other export above).
    /// Meant for a downstream tool that reads Parquet natively rather than
    /// for a human to open, so there's no equivalent of `with_list_view`'s
    /// column/plane scoping to apply here.
    pub(super) fn export_as_parquet(
        &self,
        database: &ResultsGenerator,
        cancel: &AtomicBool,
        on_progress: ExportProgress,
    ) -> Result<(), InternalErrors> {
        check_cancelled(cancel)?;
        on_progress("Exporting Parquet", 0, 1);
        // SQL may write below the results file's staging folder only (no
        // file access otherwise - see `lock_down` in the core's duckdb.rs):
        // the files go there first, in a folder of this export's own (two
        // exports of one file at once), then to where they were asked for.
        let staging = database.export_staging_dir().ok_or_else(|| {
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
        if !database.get_measurement_columns()?.is_empty() {
            tables.push(("measurements", "measurements.parquet"));
        }
        let written = tables.iter().try_for_each(|(table, file)| {
            let staged = folder.join(file);
            // DuckDB's `COPY` takes the destination as a string literal in
            // the SQL itself (not a bindable parameter): `'` doubled.
            let literal = staged.to_string_lossy().replace('\'', "''");
            // Built into DuckDB, loaded only here: no connection carries it
            // (~10 MB) when nothing is written as Parquet.
            database
                .connection()
                .execute_batch(&format!(
                    "LOAD parquet; COPY {table} TO '{literal}' (FORMAT parquet);"
                ))
                .map_err(|e| InternalErrors::Io(e.to_string()))?;
            move_file(&staged, &self.output_path(file))
        });
        // Nothing stays behind: this export's folder, and the staging folder
        // once no other export uses it.
        let _ = std::fs::remove_dir_all(&folder);
        let _ = std::fs::remove_dir(&staging);
        written?;
        on_progress("Exporting Parquet", 1, 1);
        Ok(())
    }
}

// Flat-pivot (plate_list.xlsx/well_list.xlsx) value columns - not the
// square-grid width above, this is a plain table so there's no per-column
// budget to share across many columns.
pub(super) const LIST_VALUE_COL_PX: u32 = 90;

// `get_list` only ever hands back one bounded page (`ListFilter.page`) —
// same keyset-pagination shape `results_state_controller.rs` pages through
// for the GUI's List view (see its own `update_list_view`) — so an export,
// which needs every matching row rather than one page of them, walks every
// page via the same cursor-from-last-row-id trick.
//
// Every requested image is folded into ONE combined `images` filter here
// (not called once per image) - a `WHERE image_rel_path IN (...)` scoped
// to a single image still has to plan and run a full `objects` scan of its
// own, and that cost is dominated by the *whole* table's size, not just
// the matching image's row count, on a database that holds far more than
// what's being exported (the common case). So an N-image export previously
// meant N scans each paying that overhead; batching turns it into
// `total_matching_rows / PAGE_SIZE` queries instead of `images.len()`.
// `write_list_document`'s callers pass either every requested image at
// once (the shared-file List export) or a single-element slice
// (`export_list`'s `with_list_one_file_per_image` branch, calling it once
// per image) - either way this is the one shared entry point for turning a
// `ListFilter` sweep into pages.
//
// `on_page` runs once per page in z/t-then-row-id order, rather than this
// merging every page into one in-memory `DatabaseResult` first (unlike
// `fetch_all_grouped_by_image_rows`'s approach, which suits its much
// smaller image-grouped row count): peak RAM for a List export stays
// bounded by one page's worth of rows (`PAGE_SIZE`) regardless of how many
// images or planes the export spans, instead of growing with the combined
// row count now that a page can span every image at once.
pub(super) fn stream_list_pages(
    database: &ResultsGenerator,
    cancel: &AtomicBool,
    export: &ResultExport,
    images: ImageSelection<'_>,
    object_classes: &Option<Vec<ObjectClass>>,
    with_coloc_details: bool,
    planes: &ExportPlanes,
    mut on_page: impl FnMut(&DatabaseResult, bool, PlaneFilter) -> Result<(), InternalErrors>,
) -> Result<(), InternalErrors> {
    const PAGE_SIZE: i32 = 20_000;
    // Matches the old per-image loop's behavior when there's nothing to
    // scan at all: no query, no header, an empty document - rather than
    // running one (z, t) query per plane for a filter that's already known
    // to match nothing.
    if images.matches_nothing() {
        return Ok(());
    }

    let mut header_written = false;

    // Time point by time point, then z-plane (`ExportPlanes`).
    for &plane in &planes.planes {
        let mut cursor: Option<String> = None;
        loop {
            check_cancelled(cancel)?;
            let filter = ListFilter {
                plane,
                // `images` below instead: no copy of every path per page.
                images: None,
                object_classes: object_classes.clone(),
                columns: export.columns.clone(),
                with_coloc_details,
                related_to: None,
                page: Pagination {
                    limit: PAGE_SIZE,
                    after: cursor.take(),
                },
                transpond_table: export.transpond_table,
                sort: None,
            };
            let page = database.get_object_list_of(&filter, images)?;
            let is_last_page = page.source_object_count < PAGE_SIZE as usize;
            cursor = page.row_names.last().cloned();

            on_page(&page, !header_written, plane)?;
            header_written = true;

            if is_last_page {
                break;
            }
        }
    }

    Ok(())
}

// Writes every object matching `export`'s filters, in
// z/t/image-name-sorted order (see `ResultExport::export_list`), as one
// continuous table starting at the worksheet's top row.
pub(super) fn write_list_sheet(
    worksheet: &mut Worksheet,
    database: &ResultsGenerator,
    cancel: &AtomicBool,
    export: &ResultExport,
    images: ImageSelection<'_>,
    object_classes: &Option<Vec<ObjectClass>>,
    with_coloc_details: bool,
) -> Result<(), InternalErrors> {
    let header_format = Format::new().set_bold().set_font_size(EXPORT_FONT_SIZE);
    let mut next_row: u32 = 0;
    let planes = ExportPlanes::of(database, export)?;

    stream_list_pages(
        database,
        cancel,
        export,
        images,
        object_classes,
        with_coloc_details,
        &planes,
        |page, is_first_page, plane| {
            if is_first_page {
                let header = planes
                    .header()
                    .into_iter()
                    .chain(page.column_names.iter().cloned());
                for (col_idx, name) in header.enumerate() {
                    worksheet
                        .write_with_format(next_row, col_idx as u16, name.as_str(), &header_format)
                        .map_err(xlsx_err)?;
                }
                next_row += 1;
            }

            let plane_cells = planes.cells(plane);
            for row in &page.rows {
                if next_row as usize >= XLSX_MAX_ROWS {
                    return Err(InternalErrors::InvalidArgument(format!(
                        "The List has more than {} rows - more than an XLSX sheet holds: \
                         export it as CSV, or one file per image",
                        XLSX_MAX_ROWS - 1
                    )));
                }
                for (col_idx, cell) in plane_cells.iter().chain(row.iter()).enumerate() {
                    write_cell(worksheet, next_row, col_idx as u16, cell, None)?;
                }
                next_row += 1;
            }
            Ok(())
        },
    )
}

/// CSV sibling of `write_list_sheet`: same combined-images/per-page sweep
/// (so the two formats can never disagree on row order or content), written
/// straight to `path` a page at a time instead of into an XLSX worksheet.
pub(super) fn write_list_csv(
    database: &ResultsGenerator,
    cancel: &AtomicBool,
    export: &ResultExport,
    images: ImageSelection<'_>,
    object_classes: &Option<Vec<ObjectClass>>,
    with_coloc_details: bool,
    path: &Path,
) -> Result<(), InternalErrors> {
    let mut out = create_csv_writer(path)?;
    let write_err = |e: std::io::Error| {
        InternalErrors::Internal(format!("could not write {}: {e}", path.display()))
    };
    let planes = ExportPlanes::of(database, export)?;
    let mut line = Vec::new();

    stream_list_pages(
        database,
        cancel,
        export,
        images,
        object_classes,
        with_coloc_details,
        &planes,
        |page, is_first_page, plane| {
            use std::io::Write;
            if is_first_page {
                let header: Vec<String> = planes
                    .header()
                    .into_iter()
                    .chain(page.column_names.iter().cloned())
                    .collect();
                write_csv_row(&mut out, &header).map_err(write_err)?;
            }
            // Each row into one reused line - no string per cell: a large
            // List has millions of rows.
            let plane_cells = planes.cells(plane);
            for row in &page.rows {
                line.clear();
                for (i, cell) in plane_cells.iter().chain(row.iter()).enumerate() {
                    if i > 0 {
                        line.push(b',');
                    }
                    write_csv_cell(&mut line, cell);
                }
                line.push(b'\n');
                out.write_all(&line).map_err(write_err)?;
            }
            Ok(())
        },
    )
}
