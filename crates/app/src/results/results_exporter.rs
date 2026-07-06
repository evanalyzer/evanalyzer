use crate::results::results_loader::{
    aggregate_rois_sql, build_coloc_detail_column_specs, coloc_partner_ids,
    discover_coloc_detail_columns, flatten_coloc_rows, to_display_row, ColumnSpec, DatabaseFilter,
    GroupBy, GroupConfig, ResultsLoader,
};
use evanalyzer_cfg::core_types::InternalErrors;
use rust_xlsxwriter::{Format, Workbook};
use std::{collections::HashSet, path::Path, sync::Arc};

/// Rows per DB round-trip while exporting. Larger than the GUI's page size
/// (`PAGE_SIZE = 500`) since export has no competing memory pressure from a
/// live UI — fewer, bigger round-trips is a pure win here.
const EXPORT_PAGE_SIZE: usize = 5_000;

pub struct ResultsExporter {
    results_loader: Arc<ResultsLoader>,
}

impl ResultsExporter {
    pub fn new(results_loader: Arc<ResultsLoader>) -> Self {
        Self { results_loader }
    }

    /// Exports the rows matching `filter` as a CSV file to `export_path`.
    /// When `group.group_by` is not `None`, the aggregated/grouped rows are
    /// exported instead of the per-ROI rows (mirroring the table view).
    pub fn export_to_csv(
        &self,
        filter: DatabaseFilter,
        group: &GroupConfig,
        base_specs: &[ColumnSpec],
        export_path: &Path,
    ) -> Result<(), InternalErrors> {
        let mut writer = csv::Writer::from_path(export_path)
            .map_err(|e| InternalErrors::Io(e.to_string()))?;

        self.export_rows(filter, group, base_specs, |row| {
            writer.write_record(row).map_err(|e| InternalErrors::Io(e.to_string()))
        })?;

        writer.flush().map_err(|e| InternalErrors::Io(e.to_string()))?;
        Ok(())
    }

    /// Exports the rows matching `filter` as an XLSX file to `export_path`.
    /// When `group.group_by` is not `None`, the aggregated/grouped rows are
    /// exported instead of the per-ROI rows (mirroring the table view).
    pub fn export_to_xlsx(
        &self,
        filter: DatabaseFilter,
        group: &GroupConfig,
        base_specs: &[ColumnSpec],
        export_path: &Path,
    ) -> Result<(), InternalErrors> {
        let err = |e: rust_xlsxwriter::XlsxError| InternalErrors::Io(e.to_string());
        let mut workbook = Workbook::new();
        let sheet = workbook.add_worksheet();
        sheet.set_name("Results").map_err(err)?;
        sheet.set_freeze_panes(1, 0).map_err(err)?;

        self.export_rows(filter, group, base_specs, xlsx_row_writer(sheet))?;

        workbook.save(export_path).map_err(err)?;
        Ok(())
    }

    /// Exports the colocalization detail flat table to CSV:
    /// one row per (source ROI, colocalized partner) pair.
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
        let mut writer = csv::Writer::from_path(export_path)
            .map_err(|e| InternalErrors::Io(e.to_string()))?;
        self.export_coloc_detail_rows(filter, visible_labels, |row| {
            writer.write_record(row).map_err(|e| InternalErrors::Io(e.to_string()))
        })?;
        writer.flush().map_err(|e| InternalErrors::Io(e.to_string()))?;
        Ok(())
    }

    /// Exports the colocalization detail flat table to XLSX:
    /// one row per (source ROI, colocalized partner) pair.
    /// See `export_coloc_detail_to_csv` for `visible_labels`.
    pub fn export_coloc_detail_to_xlsx(
        &self,
        filter: DatabaseFilter,
        visible_labels: Option<&HashSet<String>>,
        export_path: &Path,
    ) -> Result<(), InternalErrors> {
        let err = |e: rust_xlsxwriter::XlsxError| InternalErrors::Io(e.to_string());
        let mut workbook = Workbook::new();
        let sheet = workbook.add_worksheet();
        sheet.set_name("Coloc Detail").map_err(err)?;
        sheet.set_freeze_panes(1, 0).map_err(err)?;

        self.export_coloc_detail_rows(filter, visible_labels, xlsx_row_writer(sheet))?;

        workbook.save(export_path).map_err(err)?;
        Ok(())
    }

    // -------------------------------------------------------------------------

    /// Streams the rows matching `filter` (or grouped/aggregated rows, when
    /// `group.group_by != GroupBy::None`) to `emit_row` — the header labels
    /// first, then each data row, one page at a time — instead of
    /// materializing the whole matching result set as one `Vec` in memory.
    ///
    /// `base_specs` are the per-ROI column specs from the table (carrying the
    /// current visibility selection), so the export mirrors what is shown:
    /// - grouped → one aggregated row per group over the visible metrics
    ///   (always a single, small pass — the aggregated result set is never
    ///   large regardless of how many ROIs it summarizes);
    /// - otherwise → per-ROI rows for the visible columns only, paged.
    fn export_rows(
        &self,
        filter: DatabaseFilter,
        group: &GroupConfig,
        base_specs: &[ColumnSpec],
        mut emit_row: impl FnMut(&[String]) -> Result<(), InternalErrors>,
    ) -> Result<(), InternalErrors> {
        if group.group_by != GroupBy::None {
            // Aggregation is computed directly in DuckDB (see
            // `aggregate_rois_sql`) instead of fetching every matching row.
            let (specs, display_rows) = aggregate_rois_sql(&self.results_loader, filter, group, base_specs)?;
            let headers: Vec<String> = specs.iter().map(|c| c.label.clone()).collect();
            emit_row(&headers)?;
            for row in &display_rows {
                emit_row(&row.values)?;
            }
            return Ok(());
        }

        let headers: Vec<String> =
            base_specs.iter().filter(|c| c.visible).map(|c| c.label.clone()).collect();
        emit_row(&headers)?;

        let mut page = 0;
        loop {
            let rois_page = self.results_loader.get_rois(DatabaseFilter {
                page_size: EXPORT_PAGE_SIZE,
                page,
                needs_intensities: true,
                ..filter.clone()
            })?;
            if rois_page.is_empty() {
                break;
            }
            let done = rois_page.len() < EXPORT_PAGE_SIZE;
            for (i, roi) in rois_page.iter().enumerate() {
                let display = to_display_row(page * EXPORT_PAGE_SIZE + i, roi, base_specs);
                let values: Vec<String> = base_specs
                    .iter()
                    .zip(display.values.iter())
                    .filter(|(col, _)| col.visible)
                    .map(|(_, v)| v.clone())
                    .collect();
                emit_row(&values)?;
            }
            if done {
                break;
            }
            page += 1;
        }

        Ok(())
    }

    /// Streams the colocalization detail flat table to `emit_row` — the header
    /// labels first, then each flattened row — one page of source ROIs at a
    /// time, fetching only the colocalization partner ROIs that page's
    /// `coloc_json` actually references (via
    /// `DatabaseFilter::object_id_filter`), instead of every ROI in the image.
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
        let headers: Vec<String> =
            specs.iter().filter(|c| c.visible).map(|c| c.label.clone()).collect();
        emit_row(&headers)?;

        let mut page = 0;
        loop {
            let source_page = self.results_loader.get_rois(DatabaseFilter {
                page_size: EXPORT_PAGE_SIZE,
                page,
                needs_intensities: true,
                ..filter.clone()
            })?;
            if source_page.is_empty() {
                break;
            }
            let done = source_page.len() < EXPORT_PAGE_SIZE;

            let ids = coloc_partner_ids(&source_page);
            let partner_page = if ids.is_empty() {
                vec![]
            } else {
                self.results_loader.get_rois(DatabaseFilter {
                    object_id_filter: Some(ids),
                    page_size: 0,
                    needs_intensities: true,
                    ..Default::default()
                })?
            };

            for row in flatten_coloc_rows(&source_page, &partner_page, &specs) {
                let values: Vec<String> = specs
                    .iter()
                    .zip(row.values.iter())
                    .filter(|(col, _)| col.visible)
                    .map(|(_, v)| v.clone())
                    .collect();
                emit_row(&values)?;
            }
            if done {
                break;
            }
            page += 1;
        }

        Ok(())
    }
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
                sheet.write_with_format(xlsx_row, xlsx_col, value, &bold).map_err(err)?;
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
