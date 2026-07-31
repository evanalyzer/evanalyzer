use crate::results::plate_matrix::{
    PlateMatrixResult, WellMatrixResult, compute_plate_matrix, compute_well_matrix, resolve_range,
    row_label,
};
use crate::results::results_chart::HeatmapColorScheme;
use crate::results::results_loader::{
    AggFunc, ColumnSpec, DatabaseFilter, GroupBy, GroupConfig, ResultsLoader,
    aggregate_objects_sql, build_coloc_detail_column_specs, coloc_partner_ids,
    discover_coloc_detail_columns, flatten_coloc_rows, to_display_row, to_object_filter,
};
use evanalyzer_cfg::core_types::InternalErrors;
use evanalyzer_core::{DuckDbReader, ObjectRow};
use rust_xlsxwriter::{Color, Format, Workbook};
use std::{collections::HashSet, path::Path, sync::Arc};

/// Source rows accumulated per colocalization-partner lookup while streaming
/// the colocalization detail export - each batch does one `IN (...)` query
/// for every partner id its rows reference, so this bounds both that query's
/// size and how many source rows are held in memory at once. Not used by the
/// plain per-object/grouped export, which streams the DB's own row-at-a-time
/// cursor straight through to the CSV/XLSX writer (see `export_rows`).
const COLOC_PARTNER_BATCH_SIZE: usize = 5_000;

pub struct ResultsExporter {
    results_loader: Arc<ResultsLoader>,
}

impl ResultsExporter {
    pub fn new(results_loader: Arc<ResultsLoader>) -> Self {
        Self { results_loader }
    }

    /// Exports the rows matching `filter` as a CSV file to `export_path`.
    /// When `group.group_by` is not `None`, the aggregated/grouped rows are
    /// exported instead of the per-object rows (mirroring the table view).
    pub fn export_to_csv(
        &self,
        filter: DatabaseFilter,
        group: &GroupConfig,
        base_specs: &[ColumnSpec],
        export_path: &Path,
    ) -> Result<(), InternalErrors> {
        let mut writer =
            csv::Writer::from_path(export_path).map_err(|e| InternalErrors::Io(e.to_string()))?;

        self.export_rows(filter, group, base_specs, |row| {
            writer
                .write_record(row)
                .map_err(|e| InternalErrors::Io(e.to_string()))
        })?;

        writer
            .flush()
            .map_err(|e| InternalErrors::Io(e.to_string()))?;
        Ok(())
    }

    /// Exports the rows matching `filter` as an XLSX file to `export_path`.
    /// When `group.group_by` is not `None`, the aggregated/grouped rows are
    /// exported instead of the per-object rows (mirroring the table view).
    pub fn export_to_xlsx(
        &self,
        filter: DatabaseFilter,
        group: &GroupConfig,
        base_specs: &[ColumnSpec],
        export_path: &Path,
    ) -> Result<(), InternalErrors> {
        let err = |e: rust_xlsxwriter::XlsxError| InternalErrors::Io(e.to_string());
        let mut workbook = Workbook::new();
        // `export_rows` always writes strictly in row order (never revisits
        // an earlier row), so "constant memory" mode applies cleanly here —
        // it flushes each row to a temp file as the next one is written
        // instead of buffering the whole sheet, keeping memory flat
        // regardless of row count.
        let sheet = workbook.add_worksheet_with_constant_memory();
        sheet.set_name("Results").map_err(err)?;
        sheet.set_freeze_panes(1, 0).map_err(err)?;

        self.export_rows(filter, group, base_specs, xlsx_row_writer(sheet))?;

        workbook.save(export_path).map_err(err)?;
        Ok(())
    }

    /// Exports the colocalization detail flat table to CSV:
    /// one row per (source object, colocalized partner) pair.
    ///
    /// `visible_labels`, when given, restricts the exported columns to just
    /// those labels (columns are discovered fresh from `filter` regardless,
    /// since coloc-detail columns depend on which partner classes/channels
    /// are actually present — this only trims which of the discovered
    /// columns get written out). `None` exports every discovered column.
    pub fn export_coloc_detail_to_csv(
        &self,
        filter: DatabaseFilter,
        visible_labels: Option<&HashSet<String>>,
        export_path: &Path,
    ) -> Result<(), InternalErrors> {
        let mut writer =
            csv::Writer::from_path(export_path).map_err(|e| InternalErrors::Io(e.to_string()))?;
        self.export_coloc_detail_rows(filter, visible_labels, |row| {
            writer
                .write_record(row)
                .map_err(|e| InternalErrors::Io(e.to_string()))
        })?;
        writer
            .flush()
            .map_err(|e| InternalErrors::Io(e.to_string()))?;
        Ok(())
    }

    /// Exports the colocalization detail flat table to XLSX:
    /// one row per (source object, colocalized partner) pair.
    /// See `export_coloc_detail_to_csv` for `visible_labels`.
    pub fn export_coloc_detail_to_xlsx(
        &self,
        filter: DatabaseFilter,
        visible_labels: Option<&HashSet<String>>,
        export_path: &Path,
    ) -> Result<(), InternalErrors> {
        let err = |e: rust_xlsxwriter::XlsxError| InternalErrors::Io(e.to_string());
        let mut workbook = Workbook::new();
        let sheet = workbook.add_worksheet_with_constant_memory();
        sheet.set_name("Coloc Detail").map_err(err)?;
        sheet.set_freeze_panes(1, 0).map_err(err)?;

        self.export_coloc_detail_rows(filter, visible_labels, xlsx_row_writer(sheet))?;

        workbook.save(export_path).map_err(err)?;
        Ok(())
    }

    /// Exports one Matrix (Plate) view as a plain-text grid to CSV: a header
    /// row of column numbers, a header column of row letters (`A`, `B`, ...,
    /// `AA`, ... via [`row_label`]), and each well's aggregated value at its
    /// plate position — no color (CSV can't carry it), mirroring
    /// `export_matrix_to_xlsx` otherwise. Recomputes the matrix fresh from
    /// `filter` (the same way `export_to_csv` recomputes grouped rows fresh
    /// rather than reusing cached UI state), so each batch's own class/image
    /// filter is respected.
    #[allow(clippy::too_many_arguments)]
    pub fn export_matrix_to_csv(
        &self,
        filter: DatabaseFilter,
        group_by: GroupBy,
        regex: &str,
        agg: AggFunc,
        metric: &ColumnSpec,
        plate_rows: usize,
        plate_cols: usize,
        export_path: &Path,
    ) -> Result<(), InternalErrors> {
        let result =
            self.compute_matrix(filter, group_by, regex, agg, metric, plate_rows, plate_cols)?;

        let mut writer =
            csv::Writer::from_path(export_path).map_err(|e| InternalErrors::Io(e.to_string()))?;
        let mut header = vec![String::new()];
        header.extend((1..=result.cols).map(|c| c.to_string()));
        writer
            .write_record(&header)
            .map_err(|e| InternalErrors::Io(e.to_string()))?;
        for r in 0..result.rows {
            let mut row = vec![row_label(r)];
            for c in 0..result.cols {
                let value = result.cells[r * result.cols + c].value;
                row.push(value.map(|v| format!("{v:.3}")).unwrap_or_default());
            }
            writer
                .write_record(&row)
                .map_err(|e| InternalErrors::Io(e.to_string()))?;
        }
        writer
            .flush()
            .map_err(|e| InternalErrors::Io(e.to_string()))?;
        Ok(())
    }

    /// Exports one Matrix (Plate) view to XLSX: same grid shape as
    /// `export_matrix_to_csv`, plus each well's cell colored via
    /// `color_scheme` over `[range_min, range_max]` (or auto-ranged to
    /// `[0, max(values)]` when `range_auto`) — the same color mapping the
    /// live Matrix view uses (see `resolve_range`), so the exported sheet
    /// looks the same as what's on screen.
    #[allow(clippy::too_many_arguments)]
    pub fn export_matrix_to_xlsx(
        &self,
        filter: DatabaseFilter,
        group_by: GroupBy,
        regex: &str,
        agg: AggFunc,
        metric: &ColumnSpec,
        plate_rows: usize,
        plate_cols: usize,
        color_scheme: HeatmapColorScheme,
        range_auto: bool,
        range_min: f64,
        range_max: f64,
        export_path: &Path,
    ) -> Result<(), InternalErrors> {
        let result =
            self.compute_matrix(filter, group_by, regex, agg, metric, plate_rows, plate_cols)?;
        let values: Vec<f64> = result.cells.iter().filter_map(|c| c.value).collect();
        let (lo, hi) = resolve_range(&values, range_auto, range_min, range_max);

        let err = |e: rust_xlsxwriter::XlsxError| InternalErrors::Io(e.to_string());
        let mut workbook = Workbook::new();
        let sheet = workbook.add_worksheet();
        sheet.set_name("Matrix").map_err(err)?;

        let header_fmt = Format::new().set_bold();
        for c in 0..result.cols {
            sheet
                .write_number_with_format(0, (c + 1) as u16, (c + 1) as f64, &header_fmt)
                .map_err(err)?;
        }
        for r in 0..result.rows {
            sheet
                .write_with_format(r as u32 + 1, 0, row_label(r), &header_fmt)
                .map_err(err)?;
            for c in 0..result.cols {
                let Some(v) = result.cells[r * result.cols + c].value else {
                    continue;
                };
                let t = ((v - lo) / (hi - lo)).clamp(0.0, 1.0);
                let (cr, cg, cb) = color_scheme.color_rgb(t);
                let cell_fmt = Format::new()
                    .set_background_color(Color::RGB(
                        ((cr as u32) << 16) | ((cg as u32) << 8) | cb as u32,
                    ))
                    .set_font_color(Color::White);
                sheet
                    .write_number_with_format(r as u32 + 1, (c + 1) as u16, v, &cell_fmt)
                    .map_err(err)?;
            }
        }

        workbook.save(export_path).map_err(err)?;
        Ok(())
    }

    /// Loads every object matching `filter` and computes the plate matrix —
    /// the shared first step of both `export_matrix_to_csv`/`_xlsx`.
    #[allow(clippy::too_many_arguments)]
    fn compute_matrix(
        &self,
        filter: DatabaseFilter,
        group_by: GroupBy,
        regex: &str,
        agg: AggFunc,
        metric: &ColumnSpec,
        plate_rows: usize,
        plate_cols: usize,
    ) -> Result<PlateMatrixResult, InternalErrors> {
        let objects = self.results_loader.get_objects(DatabaseFilter {
            needs_intensities: true,
            page_size: 0,
            ..filter
        })?;
        Ok(compute_plate_matrix(
            &objects, group_by, regex, agg, metric, plate_rows, plate_cols,
        ))
    }

    /// Loads every object matching `filter` once, computes the plate matrix to
    /// discover which wells actually have data (`count > 0`), then computes
    /// each of those wells' field-of-view grid (see [`compute_well_matrix`]) -
    /// the shared first step of `export_well_matrices_to_csv`/`_xlsx`. A well
    /// whose regex has no usable sub-position data is silently skipped
    /// (mirrors the live Matrix view's own well drill-down).
    #[allow(clippy::too_many_arguments)]
    fn compute_well_matrices(
        &self,
        filter: DatabaseFilter,
        group_by: GroupBy,
        regex: &str,
        agg: AggFunc,
        metric: &ColumnSpec,
        plate_rows: usize,
        plate_cols: usize,
        well_rows: usize,
        well_cols: usize,
        well_image_order: &[i32],
    ) -> Result<Vec<WellMatrixResult>, InternalErrors> {
        let objects = self.results_loader.get_objects(DatabaseFilter {
            needs_intensities: true,
            page_size: 0,
            ..filter
        })?;
        let plate = compute_plate_matrix(
            &objects, group_by, regex, agg, metric, plate_rows, plate_cols,
        );
        Ok(plate
            .cells
            .iter()
            .filter(|c| c.count > 0 && !c.label.is_empty())
            .filter_map(|c| {
                compute_well_matrix(
                    &objects,
                    regex,
                    &c.label,
                    agg,
                    metric,
                    well_rows,
                    well_cols,
                    well_image_order,
                )
            })
            .collect())
    }

    /// Exports one CSV file per well actually present in the plate (see
    /// `compute_well_matrices`) into `folder`, named
    /// `<base_name>_<well_label>.csv` - the Well-view counterpart to
    /// `export_matrix_to_csv`'s single plate-wide grid. Errors if no well has
    /// usable sub-position data (regex has no group 4, or nothing matched)
    /// rather than silently writing nothing.
    #[allow(clippy::too_many_arguments)]
    pub fn export_well_matrices_to_csv(
        &self,
        filter: DatabaseFilter,
        group_by: GroupBy,
        regex: &str,
        agg: AggFunc,
        metric: &ColumnSpec,
        plate_rows: usize,
        plate_cols: usize,
        well_rows: usize,
        well_cols: usize,
        well_image_order: &[i32],
        folder: &Path,
        base_name: &str,
    ) -> Result<(), InternalErrors> {
        let wells = self.compute_well_matrices(
            filter,
            group_by,
            regex,
            agg,
            metric,
            plate_rows,
            plate_cols,
            well_rows,
            well_cols,
            well_image_order,
        )?;
        if wells.is_empty() {
            return Err(InternalErrors::Io(
                "No wells with sub-position data found - check the regex has a 4th capture group."
                    .to_string(),
            ));
        }

        for well in &wells {
            let export_path = folder.join(format!(
                "{base_name}_{}.csv",
                sanitize_component(&well.well_label)
            ));
            let mut writer = csv::Writer::from_path(&export_path)
                .map_err(|e| InternalErrors::Io(e.to_string()))?;
            let mut header = vec![String::new()];
            header.extend((1..=well.cols).map(|c| c.to_string()));
            writer
                .write_record(&header)
                .map_err(|e| InternalErrors::Io(e.to_string()))?;
            for r in 0..well.rows {
                let mut row = vec![row_label(r)];
                for c in 0..well.cols {
                    let value = well.cells[r * well.cols + c].value;
                    row.push(value.map(|v| format!("{v:.3}")).unwrap_or_default());
                }
                writer
                    .write_record(&row)
                    .map_err(|e| InternalErrors::Io(e.to_string()))?;
            }
            writer
                .flush()
                .map_err(|e| InternalErrors::Io(e.to_string()))?;
        }
        Ok(())
    }

    /// Exports every well actually present in the plate (see
    /// `compute_well_matrices`) into one XLSX workbook, one worksheet per well
    /// (named after the well label) - the Well-view counterpart to
    /// `export_matrix_to_xlsx`'s single plate-wide grid, each sheet colored
    /// the same way. Errors if no well has usable sub-position data.
    #[allow(clippy::too_many_arguments)]
    pub fn export_well_matrices_to_xlsx(
        &self,
        filter: DatabaseFilter,
        group_by: GroupBy,
        regex: &str,
        agg: AggFunc,
        metric: &ColumnSpec,
        plate_rows: usize,
        plate_cols: usize,
        well_rows: usize,
        well_cols: usize,
        well_image_order: &[i32],
        color_scheme: HeatmapColorScheme,
        range_auto: bool,
        range_min: f64,
        range_max: f64,
        export_path: &Path,
    ) -> Result<(), InternalErrors> {
        let wells = self.compute_well_matrices(
            filter,
            group_by,
            regex,
            agg,
            metric,
            plate_rows,
            plate_cols,
            well_rows,
            well_cols,
            well_image_order,
        )?;
        if wells.is_empty() {
            return Err(InternalErrors::Io(
                "No wells with sub-position data found - check the regex has a 4th capture group."
                    .to_string(),
            ));
        }

        let err = |e: rust_xlsxwriter::XlsxError| InternalErrors::Io(e.to_string());
        let mut workbook = Workbook::new();
        let mut used_sheet_names = HashSet::new();
        let header_fmt = Format::new().set_bold();

        for well in &wells {
            let values: Vec<f64> = well.cells.iter().filter_map(|c| c.value).collect();
            let (lo, hi) = resolve_range(&values, range_auto, range_min, range_max);
            let sheet_name = unique_sheet_name(&well.well_label, &mut used_sheet_names);
            let sheet = workbook.add_worksheet();
            sheet.set_name(sheet_name).map_err(err)?;

            for c in 0..well.cols {
                sheet
                    .write_number_with_format(0, (c + 1) as u16, (c + 1) as f64, &header_fmt)
                    .map_err(err)?;
            }
            for r in 0..well.rows {
                sheet
                    .write_with_format(r as u32 + 1, 0, row_label(r), &header_fmt)
                    .map_err(err)?;
                for c in 0..well.cols {
                    let Some(v) = well.cells[r * well.cols + c].value else {
                        continue;
                    };
                    let t = ((v - lo) / (hi - lo)).clamp(0.0, 1.0);
                    let (cr, cg, cb) = color_scheme.color_rgb(t);
                    let cell_fmt = Format::new()
                        .set_background_color(Color::RGB(
                            ((cr as u32) << 16) | ((cg as u32) << 8) | cb as u32,
                        ))
                        .set_font_color(Color::White);
                    sheet
                        .write_number_with_format(r as u32 + 1, (c + 1) as u16, v, &cell_fmt)
                        .map_err(err)?;
                }
            }
        }

        workbook.save(export_path).map_err(err)?;
        Ok(())
    }

    // -------------------------------------------------------------------------

    /// Streams the rows matching `filter` (or grouped/aggregated rows, when
    /// `group.group_by != GroupBy::None`) to `emit_row` — the header labels
    /// first, then each data row — instead of materializing the whole
    /// matching result set as one `Vec` in memory.
    ///
    /// `base_specs` are the per-object column specs from the table (carrying the
    /// current visibility selection), so the export mirrors what is shown:
    /// - grouped → one aggregated row per group over the visible metrics
    ///   (always a single, small pass — the aggregated result set is never
    ///   large regardless of how many ROIs it summarizes);
    /// - otherwise → per-object rows for the visible columns only, via a single
    ///   DB cursor over every matching row (see `DuckDbReader::stream_objects`)
    ///   instead of a fresh `LIMIT`/`OFFSET` query per page — the entire
    ///   matching set is sorted/scanned exactly once, no matter how many
    ///   rows it contains, and Rust-side memory stays at one row at a time.
    fn export_rows(
        &self,
        filter: DatabaseFilter,
        group: &GroupConfig,
        base_specs: &[ColumnSpec],
        mut emit_row: impl FnMut(&[String]) -> Result<(), InternalErrors>,
    ) -> Result<(), InternalErrors> {
        if group.group_by != GroupBy::None {
            // Aggregation is computed directly in DuckDB (see
            // `aggregate_objects_sql`) instead of fetching every matching row.
            let (specs, display_rows) =
                aggregate_objects_sql(&self.results_loader, filter, group, base_specs)?;
            let headers: Vec<String> = specs.iter().map(|c| c.label.clone()).collect();
            emit_row(&headers)?;
            for row in &display_rows {
                emit_row(&row.values)?;
            }
            return Ok(());
        }

        let headers: Vec<String> = base_specs
            .iter()
            .filter(|c| c.visible)
            .map(|c| c.label.clone())
            .collect();
        emit_row(&headers)?;

        let reader = self.results_loader.open_reader()?;
        let object_filter = to_object_filter(DatabaseFilter {
            needs_intensities: true,
            ..filter
        });
        let mut row_idx = 0usize;
        reader.stream_objects(&object_filter, |object| {
            let display = to_display_row(row_idx, &object, base_specs);
            row_idx += 1;
            let values: Vec<String> = base_specs
                .iter()
                .zip(display.values.iter())
                .filter(|(col, _)| col.visible)
                .map(|(_, v)| v.clone())
                .collect();
            emit_row(&values)
        })
    }

    /// Streams the colocalization detail flat table to `emit_row` — the header
    /// labels first, then each flattened row — via a single DB cursor over
    /// every matching source object (see `DuckDbReader::stream_objects`), fetching
    /// only the colocalization partner ROIs each batch of
    /// `COLOC_PARTNER_BATCH_SIZE` source rows actually references (via
    /// `DatabaseFilter::object_id_filter`) instead of every object in the image.
    /// The source stream and every partner lookup share one open connection
    /// (see `ResultsLoader::open_reader`), so a large export pays for exactly
    /// one connection instead of one per batch.
    fn export_coloc_detail_rows(
        &self,
        filter: DatabaseFilter,
        visible_labels: Option<&HashSet<String>>,
        mut emit_row: impl FnMut(&[String]) -> Result<(), InternalErrors>,
    ) -> Result<(), InternalErrors> {
        let (channels, coloc_partner_classes) =
            discover_coloc_detail_columns(&self.results_loader, &filter)?;
        let mut specs = build_coloc_detail_column_specs(&channels, &coloc_partner_classes);
        if let Some(labels) = visible_labels {
            for spec in specs.iter_mut() {
                spec.visible = labels.contains(&spec.label);
            }
        }
        let headers: Vec<String> = specs
            .iter()
            .filter(|c| c.visible)
            .map(|c| c.label.clone())
            .collect();
        emit_row(&headers)?;

        let reader = self.results_loader.open_reader()?;
        let object_filter = to_object_filter(DatabaseFilter {
            needs_intensities: true,
            ..filter
        });

        let mut batch: Vec<ObjectRow> = Vec::with_capacity(COLOC_PARTNER_BATCH_SIZE);
        reader.stream_objects(&object_filter, |object| {
            batch.push(object);
            if batch.len() >= COLOC_PARTNER_BATCH_SIZE {
                flush_coloc_detail_batch(&reader, &mut batch, &specs, &mut emit_row)?;
            }
            Ok(())
        })?;
        flush_coloc_detail_batch(&reader, &mut batch, &specs, &mut emit_row)
    }
}

/// Replaces characters illegal in a filename or XLSX sheet name (`/ \ : * ? "
/// < > |` plus the sheet-name-only `[ ]`) with `_`. Falls back to `"well"` if
/// nothing usable is left.
fn sanitize_component(name: &str) -> String {
    let cleaned: String = name
        .trim()
        .chars()
        .map(|c| {
            if matches!(
                c,
                '/' | '\\' | ':' | '*' | '?' | '"' | '<' | '>' | '|' | '[' | ']'
            ) {
                '_'
            } else {
                c
            }
        })
        .collect();
    let cleaned = cleaned.trim().to_string();
    if cleaned.is_empty() || cleaned.chars().all(|c| c == '_') {
        "well".to_string()
    } else {
        cleaned
    }
}

/// Sanitizes `label` into a valid XLSX sheet name (see [`sanitize_component`]),
/// truncated to Excel's 31-character limit, and disambiguated against
/// `used` with a numeric suffix on collision (well labels are normally
/// already unique, so this is a defensive fallback rather than the common
/// case).
fn unique_sheet_name(label: &str, used: &mut HashSet<String>) -> String {
    let base: String = sanitize_component(label).chars().take(31).collect();
    if used.insert(base.clone()) {
        return base;
    }
    for n in 2..1000 {
        let suffix = format!("_{n}");
        let cut = 31usize.saturating_sub(suffix.chars().count());
        let candidate: String = base.chars().take(cut).collect::<String>() + &suffix;
        if used.insert(candidate.clone()) {
            return candidate;
        }
    }
    base
}

/// Resolves colocalization partners for one accumulated batch of source ROIs
/// against `reader` (the same connection `export_coloc_detail_rows`'s source
/// stream is still open on — see `DuckDbReader::stream_objects`'s doc comment
/// for why nesting a second query here is safe), flattens and emits the
/// batch's rows, then clears it for the next batch. A no-op on an empty
/// batch (the final flush after a source count that divides evenly).
fn flush_coloc_detail_batch(
    reader: &DuckDbReader,
    batch: &mut Vec<ObjectRow>,
    specs: &[ColumnSpec],
    emit_row: &mut dyn FnMut(&[String]) -> Result<(), InternalErrors>,
) -> Result<(), InternalErrors> {
    if batch.is_empty() {
        return Ok(());
    }

    let ids = coloc_partner_ids(batch);
    let partner_batch = if ids.is_empty() {
        vec![]
    } else {
        reader.get_objects(&to_object_filter(DatabaseFilter {
            object_id_filter: Some(ids),
            page_size: 0,
            needs_intensities: true,
            ..Default::default()
        }))?
    };

    for row in flatten_coloc_rows(batch, &partner_batch, specs) {
        let values: Vec<String> = specs
            .iter()
            .zip(row.values.iter())
            .filter(|(col, _)| col.visible)
            .map(|(_, v)| v.clone())
            .collect();
        emit_row(&values)?;
    }
    batch.clear();
    Ok(())
}

/// Builds an `emit_row` closure that writes each row into `sheet` in turn —
/// the first call (the header row) bold and at row 0, every later call as a
/// data row at the next row index. Numeric-looking strings are written as
/// actual Excel numbers (so they can be sorted/filtered); empty cells are
/// skipped entirely rather than writing an empty string.
fn xlsx_row_writer(
    sheet: &mut rust_xlsxwriter::Worksheet,
) -> impl FnMut(&[String]) -> Result<(), InternalErrors> + '_ {
    let err = |e: rust_xlsxwriter::XlsxError| InternalErrors::Io(e.to_string());
    let bold = Format::new().set_bold();
    let mut next_row: u32 = 0;
    move |row: &[String]| -> Result<(), InternalErrors> {
        let xlsx_row = next_row;
        next_row += 1;
        for (col, value) in row.iter().enumerate() {
            if value.is_empty() {
                continue;
            }
            let xlsx_col = col as u16;
            if xlsx_row == 0 {
                sheet
                    .write_with_format(xlsx_row, xlsx_col, value, &bold)
                    .map_err(err)?;
                continue;
            }
            // Write numeric strings as actual numbers so Excel can sort/filter them.
            if let Ok(n) = value.parse::<f64>() {
                sheet.write_number(xlsx_row, xlsx_col, n).map_err(err)?;
            } else {
                sheet.write_string(xlsx_row, xlsx_col, value).map_err(err)?;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::results::results_loader::{AggFunc, build_column_specs};
    use crate::results::test_support::{
        decode_object_id_idx, seed_large_coloc_db, seed_large_results_db, seed_plate_results_db,
        seed_results_db,
    };
    use calamine::{Data, Reader, open_workbook_auto};

    fn parse_csv(path: &Path) -> Vec<Vec<String>> {
        let mut reader = csv::Reader::from_path(path).expect("read csv back");
        let mut rows = vec![
            reader
                .headers()
                .expect("csv header row")
                .iter()
                .map(String::from)
                .collect::<Vec<_>>(),
        ];
        for record in reader.records() {
            rows.push(
                record
                    .expect("csv data row")
                    .iter()
                    .map(String::from)
                    .collect(),
            );
        }
        rows
    }

    /// Reads an XLSX workbook's first sheet back as strings, mirroring
    /// `parse_csv`'s shape (header row first) so both formats can be
    /// asserted on with the same test logic. Numeric cells (written by
    /// `xlsx_row_writer` as real numbers, not strings) are rendered without
    /// a trailing `.0` when they're integral, matching the CSV writer's text
    /// output for the same value.
    fn parse_xlsx(path: &Path) -> Vec<Vec<String>> {
        let mut workbook: calamine::Sheets<_> = open_workbook_auto(path).expect("open xlsx back");
        let range = workbook
            .worksheet_range_at(0)
            .expect("sheet 0 present")
            .expect("read sheet 0");
        range
            .rows()
            .map(|row| {
                row.iter()
                    .map(|cell| match cell {
                        Data::Float(f) if f.fract() == 0.0 => format!("{f:.0}"),
                        Data::Int(i) => i.to_string(),
                        Data::Float(f) => f.to_string(),
                        Data::Empty => String::new(),
                        other => other.to_string(),
                    })
                    .collect()
            })
            .collect()
    }

    fn col_index(header: &[String], label: &str) -> usize {
        header
            .iter()
            .position(|h| h == label)
            .unwrap_or_else(|| panic!("no {label:?} column in {header:?}"))
    }

    #[test]
    fn export_to_csv_writes_one_row_per_object_with_the_right_values() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("results.duckdb");
        seed_results_db(&db_path);

        let loader = Arc::new(ResultsLoader::new(db_path));
        let exporter = ResultsExporter::new(loader);
        let specs = build_column_specs(&[0], &[]);

        let csv_path = dir.path().join("out.csv");
        exporter
            .export_to_csv(
                DatabaseFilter::default(),
                &GroupConfig::default(),
                &specs,
                &csv_path,
            )
            .expect("export should succeed");

        let rows = parse_csv(&csv_path);
        assert_eq!(rows.len(), 3, "header + 2 object rows");

        let header = &rows[0];
        let class_col = col_index(header, "Class");
        let image_col = col_index(header, "Image");
        let area_col = col_index(header, "Area (px\u{00B2})");
        let ch0_avg_col = col_index(header, "Ch0 Avg (bit)");

        let by_image: std::collections::HashMap<&str, &Vec<String>> = rows[1..]
            .iter()
            .map(|r| (r[image_col].as_str(), r))
            .collect();

        let row1 = by_image["img1.tif"];
        assert_eq!(row1[class_col], "ClassA");
        assert_eq!(row1[area_col], "100");
        assert_eq!(row1[ch0_avg_col], "127.0");

        let row2 = by_image["img2.tif"];
        assert_eq!(row2[class_col], "ClassB");
        assert_eq!(row2[area_col], "200");
    }

    #[test]
    fn export_to_csv_only_includes_visible_columns() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("results.duckdb");
        seed_results_db(&db_path);

        let loader = Arc::new(ResultsLoader::new(db_path));
        let exporter = ResultsExporter::new(loader);
        let mut specs = build_column_specs(&[0], &[]);
        for s in specs.iter_mut() {
            if s.id == "class" {
                s.visible = false;
            }
        }

        let csv_path = dir.path().join("out.csv");
        exporter
            .export_to_csv(
                DatabaseFilter::default(),
                &GroupConfig::default(),
                &specs,
                &csv_path,
            )
            .expect("export should succeed");

        let rows = parse_csv(&csv_path);
        assert!(
            !rows[0].contains(&"Class".to_string()),
            "hidden column must not appear in the header"
        );
    }

    #[test]
    fn export_to_csv_respects_the_image_filter() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("results.duckdb");
        seed_results_db(&db_path);

        let loader = Arc::new(ResultsLoader::new(db_path));
        let exporter = ResultsExporter::new(loader);
        let specs = build_column_specs(&[0], &[]);

        let csv_path = dir.path().join("out.csv");
        let filter = DatabaseFilter {
            image_filter: Some(vec!["img1.tif".to_string()]),
            ..Default::default()
        };
        exporter
            .export_to_csv(filter, &GroupConfig::default(), &specs, &csv_path)
            .unwrap();

        let rows = parse_csv(&csv_path);
        assert_eq!(rows.len(), 2, "header + only the one matching object");
    }

    #[test]
    fn export_to_csv_grouped_writes_one_aggregated_row_per_image() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("results.duckdb");
        seed_results_db(&db_path);

        let loader = Arc::new(ResultsLoader::new(db_path));
        let exporter = ResultsExporter::new(loader);
        let specs = build_column_specs(&[0], &[]);

        let group = GroupConfig {
            group_by: GroupBy::Image,
            aggs: vec![AggFunc::Avg],
            ..Default::default()
        };
        let csv_path = dir.path().join("grouped.csv");
        exporter
            .export_to_csv(DatabaseFilter::default(), &group, &specs, &csv_path)
            .unwrap();

        let rows = parse_csv(&csv_path);
        // One image per source object here, so grouping by image still yields
        // one row per object - this exercises the `group.group_by != None`
        // branch (`aggregate_objects_sql`) end-to-end rather than proving a
        // specific row count.
        assert_eq!(
            rows.len(),
            3,
            "header + one aggregated row per distinct image"
        );
    }

    #[test]
    fn export_to_csv_grouped_computes_the_correct_aggregated_values() {
        // Unlike `export_to_csv_grouped_writes_one_aggregated_row_per_image`
        // above (which only checks row *count*, since its fixture puts one
        // object per image), this seeds real multi-object groups so the
        // exported numbers actually exercise the aggregation math, not just
        // the grouping/row-shape plumbing.
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("results.duckdb");
        // n=6, cycling 3 images x 2 classes, area_px = row index:
        // img1={0,3} sum=3 avg=1.5; img2={1,4} sum=5 avg=2.5; img3={2,5} sum=7 avg=3.5.
        seed_large_results_db(&db_path, 6);

        let loader = Arc::new(ResultsLoader::new(db_path));
        let exporter = ResultsExporter::new(loader);
        let specs = build_column_specs(&[], &[]);

        let group = GroupConfig {
            group_by: GroupBy::Image,
            aggs: vec![AggFunc::Sum, AggFunc::Avg],
            ..Default::default()
        };
        let csv_path = dir.path().join("grouped_values.csv");
        exporter
            .export_to_csv(DatabaseFilter::default(), &group, &specs, &csv_path)
            .unwrap();

        let rows = parse_csv(&csv_path);
        let header = &rows[0];
        let area_sum_col = header
            .iter()
            .position(|h| h.contains("Area") && h.contains("sum"))
            .expect("an Area [sum] column must be present");
        let area_avg_col = header
            .iter()
            .position(|h| h.contains("Area") && h.contains("avg"))
            .expect("an Area [avg] column must be present");
        let image_col = header.iter().position(|h| h == "Image").unwrap();

        let mut by_image: std::collections::HashMap<String, (String, String)> = rows[1..]
            .iter()
            .map(|r| {
                (
                    r[image_col].clone(),
                    (r[area_sum_col].clone(), r[area_avg_col].clone()),
                )
            })
            .collect();

        assert_eq!(
            by_image.remove("img1.tif"),
            Some(("3.0".to_string(), "1.5".to_string()))
        );
        assert_eq!(
            by_image.remove("img2.tif"),
            Some(("5.0".to_string(), "2.5".to_string()))
        );
        assert_eq!(
            by_image.remove("img3.tif"),
            Some(("7.0".to_string(), "3.5".to_string()))
        );
    }

    #[test]
    fn export_to_xlsx_writes_a_readable_workbook() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("results.duckdb");
        seed_results_db(&db_path);

        let loader = Arc::new(ResultsLoader::new(db_path));
        let exporter = ResultsExporter::new(loader);
        let specs = build_column_specs(&[0], &[]);

        let xlsx_path = dir.path().join("out.xlsx");
        exporter
            .export_to_xlsx(
                DatabaseFilter::default(),
                &GroupConfig::default(),
                &specs,
                &xlsx_path,
            )
            .expect("export should succeed");

        let bytes = std::fs::read(&xlsx_path).expect("xlsx file should exist");
        assert!(
            bytes.len() > 100,
            "workbook should have real content, not just a stub file"
        );
        // XLSX is a ZIP container - "PK\x03\x04" is the local-file-header
        // magic every valid ZIP (and therefore every valid XLSX) starts with.
        assert_eq!(&bytes[0..4], b"PK\x03\x04", "not a valid XLSX/ZIP file");
    }

    #[test]
    fn export_to_xlsx_writes_numeric_columns_as_real_excel_numbers_not_text() {
        // `xlsx_row_writer`'s whole point is writing numeric-looking cells as
        // actual Excel numbers (`write_number`) rather than text
        // (`write_string`), so Excel formulas like SUM/AVERAGE work directly
        // over an exported column. Every previous XLSX test reads cells back
        // through `parse_xlsx`, which stringifies everything - a regression
        // that silently switched to `write_string` for every column would
        // still pass those. This reads the raw `calamine::Data` variant
        // instead, so it actually distinguishes the two.
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("results.duckdb");
        seed_results_db(&db_path);

        let loader = Arc::new(ResultsLoader::new(db_path));
        let exporter = ResultsExporter::new(loader);
        let specs = build_column_specs(&[], &[]);

        let xlsx_path = dir.path().join("types.xlsx");
        exporter
            .export_to_xlsx(
                DatabaseFilter::default(),
                &GroupConfig::default(),
                &specs,
                &xlsx_path,
            )
            .unwrap();

        let mut workbook = open_workbook_auto(&xlsx_path).expect("open exported xlsx");
        let sheet = workbook.worksheet_range("Results").expect("Results sheet");
        let header: Vec<String> = sheet
            .rows()
            .next()
            .unwrap()
            .iter()
            .map(|c| c.to_string())
            .collect();
        let area_col = header
            .iter()
            .position(|h| h.starts_with("Area (px"))
            .expect("Area column header");
        let image_col = header.iter().position(|h| h == "Image").unwrap();

        let data_row = sheet.rows().nth(1).expect("one data row");
        assert!(
            matches!(data_row[area_col], Data::Float(_) | Data::Int(_)),
            "numeric column must be a real Excel number, got {:?}",
            data_row[area_col]
        );
        assert!(
            matches!(&data_row[image_col], Data::String(_)),
            "image name must stay a text cell, got {:?}",
            data_row[image_col]
        );
    }

    #[test]
    fn export_to_csv_round_trips_values_containing_commas_and_quotes() {
        // The `csv` crate handles RFC 4180 escaping itself, but this proves
        // the actual export+reparse round trip preserves a class name with
        // characters that would corrupt a naive comma-joined line - a
        // realistic case since class names are free-form user text.
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("results.duckdb");
        seed_results_db(&db_path);

        let image_name = "img \"weird\", name.tif";
        let class_name = "Class, \"A\"";
        let object_class_json = serde_json::to_string(&vec![class_name]).unwrap();
        let conn = duckdb::Connection::open(&db_path).unwrap();
        conn.execute(
            "INSERT INTO objects (
                image_name, image_rel_path, object_id, seg_class_name, seg_class_id,
                object_class_name, object_class_id, track_id,
                centroid_x_px, centroid_y_px, centroid_x_nm, centroid_y_nm,
                bbox_xmin_px, bbox_ymin_px, bbox_xmax_px, bbox_ymax_px,
                bbox_xmin_nm, bbox_ymin_nm, bbox_xmax_nm, bbox_ymax_nm,
                area_px, area_nm2, perimeter_px, perimeter_nm,
                circularity, solidity, aspect_ratio, roundness, compactness,
                major_axis_px, minor_axis_px, touches_edge,
                pixel_size_x_nm, pixel_size_y_nm, pixel_size_z_nm,
                intensities_json, coloc_json
            ) VALUES (
                ?, ?, '00000000-0000-0000-0000-000000000099',
                ?, 1, ?, '[1]', 0,
                0, 0, 0, 0, 0, 0, 10, 10, 0, 0, 0, 0,
                50, 50.0, 40, 40, 1.0, 1.0, 1.0, 1.0, 1.0, 10, 10, false,
                1.0, 1.0, 1.0, '{}', '{}'
            )",
            duckdb::params![image_name, image_name, class_name, object_class_json],
        )
        .unwrap();
        drop(conn);

        let loader = Arc::new(ResultsLoader::new(db_path));
        let exporter = ResultsExporter::new(loader);
        let specs = build_column_specs(&[], &[]);
        let csv_path = dir.path().join("special_chars.csv");
        exporter
            .export_to_csv(
                DatabaseFilter::default(),
                &GroupConfig::default(),
                &specs,
                &csv_path,
            )
            .unwrap();

        let rows = parse_csv(&csv_path);
        let header = &rows[0];
        let image_col = header.iter().position(|h| h == "Image").unwrap();
        let class_col = header.iter().position(|h| h == "Class").unwrap();
        let row = rows[1..]
            .iter()
            .find(|r| r[image_col] == image_name)
            .expect("the special-character row must round-trip through export");
        assert_eq!(row[class_col], class_name);
    }

    // -------------------------------------------------------------------------
    // Matrix (Plate) export tests.

    fn area_metric() -> ColumnSpec {
        ColumnSpec {
            id: "area_px".to_string(),
            label: "Area (px\u{00B2})".to_string(),
            filterable: false,
            visible: true,
        }
    }

    #[test]
    fn export_matrix_to_csv_places_wells_by_folder_and_formats_values() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("results.duckdb");
        seed_plate_results_db(&db_path);

        let loader = Arc::new(ResultsLoader::new(db_path));
        let exporter = ResultsExporter::new(loader);

        let csv_path = dir.path().join("plate.csv");
        exporter
            .export_matrix_to_csv(
                DatabaseFilter::default(),
                GroupBy::Folder,
                "",
                AggFunc::Avg,
                &area_metric(),
                2,
                2,
                &csv_path,
            )
            .expect("matrix export should succeed");

        let rows = parse_csv(&csv_path);
        assert_eq!(
            rows,
            vec![
                vec!["".to_string(), "1".to_string(), "2".to_string()],
                vec!["A".to_string(), "10.000".to_string(), "".to_string()],
                vec!["B".to_string(), "".to_string(), "20.000".to_string()],
            ],
            "folder \"A1\" -> row A/col 1 (value 10), folder \"B2\" -> row B/col 2 (value 20)"
        );
    }

    #[test]
    fn export_matrix_to_csv_regex_grouping_produces_the_same_grid_as_folder_grouping() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("results.duckdb");
        seed_plate_results_db(&db_path);

        let loader = Arc::new(ResultsLoader::new(db_path));
        let exporter = ResultsExporter::new(loader);

        let csv_path = dir.path().join("plate.csv");
        exporter
            .export_matrix_to_csv(
                DatabaseFilter::default(),
                GroupBy::Regex,
                r"^([A-Z]\d+)_",
                AggFunc::Avg,
                &area_metric(),
                2,
                2,
                &csv_path,
            )
            .expect("matrix export should succeed");

        let rows = parse_csv(&csv_path);
        assert_eq!(
            rows,
            vec![
                vec!["".to_string(), "1".to_string(), "2".to_string()],
                vec!["A".to_string(), "10.000".to_string(), "".to_string()],
                vec!["B".to_string(), "".to_string(), "20.000".to_string()],
            ]
        );
    }

    #[test]
    fn export_matrix_to_xlsx_writes_the_same_values_as_csv_with_a_distinct_fill_per_cell() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("results.duckdb");
        seed_plate_results_db(&db_path);

        let loader = Arc::new(ResultsLoader::new(db_path));
        let exporter = ResultsExporter::new(loader);

        let xlsx_path = dir.path().join("plate.xlsx");
        exporter
            .export_matrix_to_xlsx(
                DatabaseFilter::default(),
                GroupBy::Folder,
                "",
                AggFunc::Avg,
                &area_metric(),
                2,
                2,
                HeatmapColorScheme::Viridis,
                true,
                0.0,
                1.0,
                &xlsx_path,
            )
            .expect("matrix export should succeed");

        let rows = parse_xlsx(&xlsx_path);
        assert_eq!(
            rows[0],
            vec!["".to_string(), "1".to_string(), "2".to_string()]
        );
        assert_eq!(
            rows[1],
            vec!["A".to_string(), "10".to_string(), "".to_string()]
        );
        assert_eq!(
            rows[2],
            vec!["B".to_string(), "".to_string(), "20".to_string()]
        );

        let bytes = std::fs::read(&xlsx_path).expect("xlsx file should exist");
        assert_eq!(&bytes[0..4], b"PK\x03\x04", "not a valid XLSX/ZIP file");
        // Per-cell fill coloring (via `HeatmapColorScheme::color_rgb`) is
        // covered by live GUI verification rather than here - inspecting it
        // would need a zip/XML reader this crate doesn't otherwise depend on,
        // out of proportion to what a unit test should pull in.
    }

    #[test]
    fn export_matrix_to_xlsx_writes_the_same_rounded_value_the_table_view_shows() {
        // `compute_plate_matrix`/`compute_well_matrix` (shared by the Table
        // view, the GUI's live Matrix grid, and both Matrix export formats)
        // already round every metric to `metric_precision(column_id)`
        // decimals *before* any of these ever sees it - by re-parsing the
        // `aggregate_rows`-formatted display string rather than keeping a
        // full-precision float. `export_matrix_to_xlsx` writes that value
        // via plain `write_number` with no further `.set_num_format(...)`
        // rounding on top, so it must land in the workbook exactly as
        // Table shows it (unlike `export_matrix_to_csv`, which reformats
        // to a fixed `{:.3}`, and the GUI's on-screen Matrix cell, which
        // reformats to a fixed `{:.1}` - both *additional* roundings on top
        // of this same first one, see the two tests below). Uses the raw
        // `calamine::Data` value (not the `parse_xlsx` test helper, which
        // itself special-cases whole-ish floats for display and would hide
        // a real mismatch here).
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("results.duckdb");
        seed_plate_results_db(&db_path); // establishes the schema + wells A1/B2
        let conn = duckdb::Connection::open(&db_path).unwrap();
        // A new well "A2" (row 0, col 1 - within the 2x2 plate below, and
        // untouched by seed_plate_results_db) with 3 objects: area_px =
        // 10, 11, 11 -> avg = 32/3 = 10.6666... - genuinely non-terminating,
        // so "full precision" and "rounded to N decimals" can't coincide.
        for (idx, area) in [10u64, 11, 11].into_iter().enumerate() {
            conn.execute(
                "INSERT INTO objects (
                    image_name, image_rel_path, object_id, seg_class_name, seg_class_id,
                    object_class_name, object_class_id, track_id,
                    centroid_x_px, centroid_y_px, centroid_x_nm, centroid_y_nm,
                    bbox_xmin_px, bbox_ymin_px, bbox_xmax_px, bbox_ymax_px,
                    bbox_xmin_nm, bbox_ymin_nm, bbox_xmax_nm, bbox_ymax_nm,
                    area_px, area_nm2, perimeter_px, perimeter_nm,
                    circularity, solidity, aspect_ratio, roundness, compactness,
                    major_axis_px, minor_axis_px, touches_edge,
                    pixel_size_x_nm, pixel_size_y_nm, pixel_size_z_nm,
                    intensities_json, coloc_json
                ) VALUES (
                    'A2_extra.tif', 'A2/A2_extra.tif', ?, 'ClassA', 1,
                    '[\"ClassA\"]', '[1]', 0,
                    0, 0, 0, 0, 0, 0, 10, 10, 0, 0, 0, 0,
                    ?, ?, 40, 40, 1.0, 1.0, 1.0, 1.0, 1.0, 10, 10, false,
                    1.0, 1.0, 1.0, '{}', '{}'
                )",
                duckdb::params![
                    format!("00000000-0000-0000-0000-0000000001{idx:02}"),
                    area,
                    area as f64
                ],
            )
            .unwrap();
        }
        drop(conn);

        let loader = Arc::new(ResultsLoader::new(db_path));
        let exporter = ResultsExporter::new(loader);
        let xlsx_path = dir.path().join("plate_precision.xlsx");
        exporter
            .export_matrix_to_xlsx(
                DatabaseFilter::default(),
                GroupBy::Folder,
                "",
                AggFunc::Avg,
                &area_metric(),
                2,
                2,
                HeatmapColorScheme::Viridis,
                true,
                0.0,
                1.0,
                &xlsx_path,
            )
            .unwrap();

        let mut workbook = open_workbook_auto(&xlsx_path).unwrap();
        let sheet = workbook.worksheet_range("Matrix").unwrap();
        // Row "A" (sheet row 1), col "2" (sheet col 2: row-label col 0 + 1-indexed col 1 + 1).
        let cell = sheet.get_value((1, 2)).expect("well A2's cell value");
        let Data::Float(v) = cell else {
            panic!("expected a numeric cell, got {cell:?}");
        };
        // True average is 32/3 = 10.6666..., but `area_px`'s
        // `metric_precision` is 1 decimal, so the value every consumer
        // (Table included) actually works with is the already-rounded 10.7.
        assert!(
            (v - 10.7).abs() < 1e-9,
            "expected the metric_precision-rounded 10.7 (matching what the Table view shows), got {v}"
        );
    }

    #[test]
    fn export_matrix_to_csv_pads_to_a_fixed_3_decimals_even_when_the_table_view_shows_fewer() {
        // Same fixture/value as the XLSX test above (10.7, already rounded
        // to `area_px`'s 1-decimal `metric_precision`) - `export_matrix_to_csv`
        // additionally reformats every value to a fixed `{:.3}`, so the CSV
        // cell reads "10.700" even though the Table view (and the XLSX
        // export) show "10.7". Same numeric value, cosmetically different
        // text - documented so a future reader doesn't mistake the extra
        // trailing zeros for a real precision difference.
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("results.duckdb");
        seed_plate_results_db(&db_path);
        let conn = duckdb::Connection::open(&db_path).unwrap();
        for (idx, area) in [10u64, 11, 11].into_iter().enumerate() {
            conn.execute(
                "INSERT INTO objects (
                    image_name, image_rel_path, object_id, seg_class_name, seg_class_id,
                    object_class_name, object_class_id, track_id,
                    centroid_x_px, centroid_y_px, centroid_x_nm, centroid_y_nm,
                    bbox_xmin_px, bbox_ymin_px, bbox_xmax_px, bbox_ymax_px,
                    bbox_xmin_nm, bbox_ymin_nm, bbox_xmax_nm, bbox_ymax_nm,
                    area_px, area_nm2, perimeter_px, perimeter_nm,
                    circularity, solidity, aspect_ratio, roundness, compactness,
                    major_axis_px, minor_axis_px, touches_edge,
                    pixel_size_x_nm, pixel_size_y_nm, pixel_size_z_nm,
                    intensities_json, coloc_json
                ) VALUES (
                    'A2_extra.tif', 'A2/A2_extra.tif', ?, 'ClassA', 1,
                    '[\"ClassA\"]', '[1]', 0,
                    0, 0, 0, 0, 0, 0, 10, 10, 0, 0, 0, 0,
                    ?, ?, 40, 40, 1.0, 1.0, 1.0, 1.0, 1.0, 10, 10, false,
                    1.0, 1.0, 1.0, '{}', '{}'
                )",
                duckdb::params![
                    format!("00000000-0000-0000-0000-0000000002{idx:02}"),
                    area,
                    area as f64
                ],
            )
            .unwrap();
        }
        drop(conn);

        let loader = Arc::new(ResultsLoader::new(db_path));
        let exporter = ResultsExporter::new(loader);
        let csv_path = dir.path().join("plate_precision.csv");
        exporter
            .export_matrix_to_csv(
                DatabaseFilter::default(),
                GroupBy::Folder,
                "",
                AggFunc::Avg,
                &area_metric(),
                2,
                2,
                &csv_path,
            )
            .unwrap();

        let rows = parse_csv(&csv_path);
        assert_eq!(rows[1][2], "10.700", "well A2 (row A, col 2)");
    }

    /// A well regex with the 4th capture group `compute_well_matrix` needs
    /// (sub-position) - matches `seed_plate_results_db`'s "A1_01.tif" /
    /// "B2_01.tif" image names exactly: group 1 = well id ("A1"/"B2"),
    /// group 4 = sub-position ("01").
    const WELL_REGEX: &str = r"^(([A-Z])(\d+))_(\d+)";

    #[test]
    fn export_well_matrices_to_csv_writes_one_file_per_well() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("results.duckdb");
        seed_plate_results_db(&db_path);

        let loader = Arc::new(ResultsLoader::new(db_path));
        let exporter = ResultsExporter::new(loader);

        exporter
            .export_well_matrices_to_csv(
                DatabaseFilter::default(),
                GroupBy::Regex,
                WELL_REGEX,
                AggFunc::Avg,
                &area_metric(),
                2,
                2,
                1,
                1,
                &[1],
                dir.path(),
                "wells",
            )
            .expect("well matrix export should succeed");

        let rows_a1 = parse_csv(&dir.path().join("wells_A1.csv"));
        assert_eq!(
            rows_a1,
            vec![
                vec!["".to_string(), "1".to_string()],
                vec!["A".to_string(), "10.000".to_string()]
            ]
        );
        let rows_b2 = parse_csv(&dir.path().join("wells_B2.csv"));
        assert_eq!(
            rows_b2,
            vec![
                vec!["".to_string(), "1".to_string()],
                vec!["A".to_string(), "20.000".to_string()]
            ]
        );
    }

    #[test]
    fn export_well_matrices_to_csv_errors_when_no_well_has_sub_position_data() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("results.duckdb");
        seed_plate_results_db(&db_path);

        let loader = Arc::new(ResultsLoader::new(db_path));
        let exporter = ResultsExporter::new(loader);

        let result = exporter.export_well_matrices_to_csv(
            DatabaseFilter::default(),
            GroupBy::Regex,
            r"^([A-Z]\d+)_", // only 1 capture group - no sub-position
            AggFunc::Avg,
            &area_metric(),
            2,
            2,
            1,
            1,
            &[1],
            dir.path(),
            "wells",
        );
        assert!(
            result.is_err(),
            "no well has usable sub-position data - should error, not write nothing"
        );
    }

    #[test]
    fn export_well_matrices_to_xlsx_writes_one_sheet_per_well() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("results.duckdb");
        seed_plate_results_db(&db_path);

        let loader = Arc::new(ResultsLoader::new(db_path));
        let exporter = ResultsExporter::new(loader);

        let xlsx_path = dir.path().join("wells.xlsx");
        exporter
            .export_well_matrices_to_xlsx(
                DatabaseFilter::default(),
                GroupBy::Regex,
                WELL_REGEX,
                AggFunc::Avg,
                &area_metric(),
                2,
                2,
                1,
                1,
                &[1],
                HeatmapColorScheme::Viridis,
                true,
                0.0,
                1.0,
                &xlsx_path,
            )
            .expect("well matrix export should succeed");

        let mut workbook: calamine::Sheets<_> =
            open_workbook_auto(&xlsx_path).expect("open xlsx back");
        let mut sheet_names = workbook.sheet_names().to_vec();
        sheet_names.sort();
        assert_eq!(sheet_names, vec!["A1".to_string(), "B2".to_string()]);

        let sheet_a1 = workbook.worksheet_range("A1").expect("read sheet A1");
        let rows_a1: Vec<Vec<String>> = sheet_a1
            .rows()
            .map(|row| {
                row.iter()
                    .map(|cell| match cell {
                        Data::Float(f) if f.fract() == 0.0 => format!("{f:.0}"),
                        Data::Empty => String::new(),
                        other => other.to_string(),
                    })
                    .collect()
            })
            .collect();
        assert_eq!(
            rows_a1,
            vec![
                vec!["".to_string(), "1".to_string()],
                vec!["A".to_string(), "10".to_string()]
            ]
        );
    }

    // -------------------------------------------------------------------------
    // Large-dataset characterization tests.
    //
    // These seed far more rows than any GUI page or DB round-trip holds and
    // assert every seeded row appears in the export exactly once, in the
    // expected order - proving there's no off-by-one, duplicate, or
    // dropped-row bug when a result set spans many DB round-trips. Originally
    // written against the per-page `LIMIT`/`OFFSET` re-query implementation
    // (each round-trip a separate page) and left unchanged across the move to
    // `DuckDbReader::stream_objects`'s single-cursor implementation (each
    // round-trip a chunk of one continuous scan) - passing before and after
    // is exactly what proves the two implementations return identical
    // results.

    #[test]
    fn export_to_csv_across_a_large_result_set_returns_every_row_exactly_once_in_order() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("results.duckdb");
        let n = COLOC_PARTNER_BATCH_SIZE * 2 + 345;
        seed_large_results_db(&db_path, n);

        let loader = Arc::new(ResultsLoader::new(db_path));
        let exporter = ResultsExporter::new(loader);
        let specs = build_column_specs(&[0], &[]);

        let csv_path = dir.path().join("out.csv");
        exporter
            .export_to_csv(
                DatabaseFilter::default(),
                &GroupConfig::default(),
                &specs,
                &csv_path,
            )
            .expect("export should succeed");

        let rows = parse_csv(&csv_path);
        assert_eq!(
            rows.len(),
            n + 1,
            "header + one row per seeded object, none dropped or duplicated"
        );

        let area_col = col_index(&rows[0], "Area (px\u{00B2})");
        let areas: Vec<u64> = rows[1..]
            .iter()
            .map(|r| r[area_col].parse().unwrap())
            .collect();
        let expected: Vec<u64> = (0..n as u64).collect();
        assert_eq!(
            areas, expected,
            "rows must come back in ascending object_id order across every export page"
        );
    }

    #[test]
    fn export_to_xlsx_across_a_large_result_set_returns_every_row_exactly_once_in_order() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("results.duckdb");
        let n = COLOC_PARTNER_BATCH_SIZE * 2 + 345;
        seed_large_results_db(&db_path, n);

        let loader = Arc::new(ResultsLoader::new(db_path));
        let exporter = ResultsExporter::new(loader);
        let specs = build_column_specs(&[0], &[]);

        let xlsx_path = dir.path().join("out.xlsx");
        exporter
            .export_to_xlsx(
                DatabaseFilter::default(),
                &GroupConfig::default(),
                &specs,
                &xlsx_path,
            )
            .expect("export should succeed");

        let rows = parse_xlsx(&xlsx_path);
        assert_eq!(
            rows.len(),
            n + 1,
            "header + one row per seeded object, none dropped or duplicated"
        );

        let area_col = col_index(&rows[0], "Area (px\u{00B2})");
        let areas: Vec<u64> = rows[1..]
            .iter()
            .map(|r| r[area_col].parse().unwrap())
            .collect();
        let expected: Vec<u64> = (0..n as u64).collect();
        assert_eq!(
            areas, expected,
            "rows must come back in ascending object_id order across every export page"
        );
    }

    #[test]
    fn export_coloc_detail_to_csv_across_a_large_result_set_resolves_every_partner_correctly() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("results.duckdb");
        let n_sources = COLOC_PARTNER_BATCH_SIZE + 777;
        let n_partners = 500;
        seed_large_coloc_db(&db_path, n_sources, n_partners);

        let loader = Arc::new(ResultsLoader::new(db_path));
        let exporter = ResultsExporter::new(loader);

        let csv_path = dir.path().join("coloc_detail.csv");
        let filter = DatabaseFilter {
            class_filter: Some(vec!["ClassA".to_string()]),
            ..Default::default()
        };
        exporter
            .export_coloc_detail_to_csv(filter, None, &csv_path)
            .expect("export should succeed");

        let rows = parse_csv(&csv_path);
        assert_eq!(
            rows.len(),
            n_sources + 1,
            "header + exactly one flattened row per source object"
        );

        let object_id_col = col_index(&rows[0], "object ID");
        let partner_id_col = col_index(&rows[0], "Coloc ClassB object ID");

        let mut seen_source_idx = vec![false; n_sources];
        for row in &rows[1..] {
            let source_idx = decode_object_id_idx(&row[object_id_col]);
            let partner_idx = decode_object_id_idx(&row[partner_id_col]);
            assert_eq!(
                partner_idx,
                source_idx % n_partners,
                "row for source {source_idx} resolved the wrong partner across an export page boundary"
            );
            assert!(
                !seen_source_idx[source_idx],
                "source object {source_idx} exported more than once"
            );
            seen_source_idx[source_idx] = true;
        }
        assert!(
            seen_source_idx.into_iter().all(|seen| seen),
            "every source object must be exported exactly once"
        );
    }

    #[test]
    fn export_coloc_detail_to_xlsx_writes_a_readable_workbook_and_respects_the_column_filter() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("results.duckdb");
        seed_large_coloc_db(&db_path, 2, 1);

        let loader = Arc::new(ResultsLoader::new(db_path));
        let exporter = ResultsExporter::new(loader);

        let xlsx_path = dir.path().join("coloc_detail.xlsx");
        let visible: HashSet<String> = ["object ID", "Coloc ClassB object ID"]
            .into_iter()
            .map(String::from)
            .collect();
        // Scoped to ClassA (the source class) only - otherwise the ClassB
        // partner objects seeded alongside the sources would themselves be
        // treated as (partner-less) source rows too, same as the large-scale
        // CSV test above.
        let filter = DatabaseFilter {
            class_filter: Some(vec!["ClassA".to_string()]),
            ..Default::default()
        };
        exporter
            .export_coloc_detail_to_xlsx(filter, Some(&visible), &xlsx_path)
            .expect("export should succeed");

        let rows = parse_xlsx(&xlsx_path);
        assert_eq!(
            rows.len(),
            3,
            "header + one flattened row per source object"
        );
        assert_eq!(
            rows[0],
            vec![
                "object ID".to_string(),
                "Coloc ClassB object ID".to_string()
            ],
            "hidden columns (Image, Class, Area, ...) must not appear in the header"
        );

        let bytes = std::fs::read(&xlsx_path).expect("xlsx file should exist");
        assert_eq!(&bytes[0..4], b"PK\x03\x04", "not a valid XLSX/ZIP file");
    }

    #[test]
    fn export_coloc_detail_rows_with_no_colocalization_data_still_exports_object_rows() {
        // `seed_results_db`'s two objects both carry an empty `coloc_json`
        // ('{}') and no `coloc_stats` rows - the shape of a project whose
        // colocalization pipeline step has never run. The exporter must
        // still succeed (zero partner classes discovered, zero partner ids
        // resolved per batch) rather than erroring or panicking.
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("results.duckdb");
        seed_results_db(&db_path);

        let loader = Arc::new(ResultsLoader::new(db_path));
        let exporter = ResultsExporter::new(loader);

        let csv_path = dir.path().join("coloc_detail.csv");
        exporter
            .export_coloc_detail_to_csv(DatabaseFilter::default(), None, &csv_path)
            .expect("export should succeed even with no colocalization data at all");

        let rows = parse_csv(&csv_path);
        assert_eq!(
            rows.len(),
            3,
            "header + one row per object, no partner columns"
        );
        assert!(
            !rows[0].iter().any(|h| h.starts_with("Coloc ")),
            "no partner classes were ever recorded, so no Coloc-prefixed column should exist"
        );
    }
}
