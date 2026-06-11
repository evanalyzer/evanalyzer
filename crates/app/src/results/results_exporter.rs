use crate::results::results_loader::{
    aggregate_rows, to_display_row, ColumnSpec, DatabaseFilter, GroupBy, GroupConfig, ResultsLoader,
};
use evanalyzer_cfg::core_types::InternalErrors;
use rust_xlsxwriter::{Format, Workbook};
use std::{path::Path, sync::Arc};

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
        let (headers, rows) = self.prepare_data(filter, group, base_specs)?;

        let mut writer = csv::Writer::from_path(export_path)
            .map_err(|e| InternalErrors::Io(e.to_string()))?;

        writer
            .write_record(&headers)
            .map_err(|e| InternalErrors::Io(e.to_string()))?;

        for row in &rows {
            writer
                .write_record(row)
                .map_err(|e| InternalErrors::Io(e.to_string()))?;
        }

        writer
            .flush()
            .map_err(|e| InternalErrors::Io(e.to_string()))?;
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
        let (headers, rows) = self.prepare_data(filter, group, base_specs)?;
        let err = |e: rust_xlsxwriter::XlsxError| InternalErrors::Io(e.to_string());

        let mut workbook = Workbook::new();
        let sheet = workbook.add_worksheet();
        sheet.set_name("Results").map_err(err)?;
        sheet.set_freeze_panes(1, 0).map_err(err)?;

        let bold = Format::new().set_bold();

        for (col, label) in headers.iter().enumerate() {
            sheet
                .write_with_format(0, col as u16, label.as_str(), &bold)
                .map_err(err)?;
        }

        for (row_idx, row) in rows.iter().enumerate() {
            let xlsx_row = (row_idx + 1) as u32;
            for (col, value) in row.iter().enumerate() {
                let xlsx_col = col as u16;
                if value.is_empty() {
                    continue;
                }
                // Write numeric strings as actual numbers so Excel can sort/filter them.
                if let Ok(n) = value.parse::<f64>() {
                    sheet.write_number(xlsx_row, xlsx_col, n).map_err(err)?;
                } else {
                    sheet.write_string(xlsx_row, xlsx_col, value).map_err(err)?;
                }
            }
        }

        workbook.save(export_path).map_err(err)?;
        Ok(())
    }

    // -------------------------------------------------------------------------

    /// Loads all matching rows and returns (header labels, rows-of-strings).
    ///
    /// `base_specs` are the per-ROI column specs from the table (carrying the
    /// current visibility selection), so the export mirrors what is shown:
    /// - grouped → one aggregated row per group over the visible metrics;
    /// - otherwise → per-ROI rows for the visible columns only.
    fn prepare_data(
        &self,
        filter: DatabaseFilter,
        group: &GroupConfig,
        base_specs: &[ColumnSpec],
    ) -> Result<(Vec<String>, Vec<Vec<String>>), InternalErrors> {
        let rois = self.results_loader.get_rois(DatabaseFilter {
            page_size: 0, // fetch all rows
            needs_intensities: true,
            ..filter
        })?;

        // Grouped export: one aggregated row per group (over visible metrics).
        if group.group_by != GroupBy::None {
            let (specs, display_rows) = aggregate_rows(&rois, group, base_specs);
            let headers = specs.iter().map(|c| c.label.clone()).collect();
            let rows = display_rows.into_iter().map(|d| d.values).collect();
            return Ok((headers, rows));
        }

        let headers: Vec<String> = base_specs
            .iter()
            .filter(|c| c.visible)
            .map(|c| c.label.clone())
            .collect();

        let rows: Vec<Vec<String>> = rois
            .iter()
            .enumerate()
            .map(|(i, roi)| {
                let display = to_display_row(i, roi, base_specs);
                base_specs
                    .iter()
                    .zip(display.values.iter())
                    .filter(|(col, _)| col.visible)
                    .map(|(_, v)| v.clone())
                    .collect()
            })
            .collect();

        Ok((headers, rows))
    }
}
