use crate::results::results_loader::{
    aggregate_rois_sql, build_coloc_detail_column_specs, coloc_partner_ids,
    discover_coloc_detail_columns, flatten_coloc_rows, to_display_row, to_roi_filter, ColumnSpec,
    DatabaseFilter, GroupBy, GroupConfig, ResultsLoader,
};
use evanalyzer_cfg::core_types::InternalErrors;
use evanalyzer_core::{DuckDbReader, RoiRow};
use rust_xlsxwriter::{Format, Workbook};
use std::{collections::HashSet, path::Path, sync::Arc};

/// Source rows accumulated per colocalization-partner lookup while streaming
/// the colocalization detail export - each batch does one `IN (...)` query
/// for every partner id its rows reference, so this bounds both that query's
/// size and how many source rows are held in memory at once. Not used by the
/// plain per-ROI/grouped export, which streams the DB's own row-at-a-time
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
        let sheet = workbook.add_worksheet_with_constant_memory();
        sheet.set_name("Coloc Detail").map_err(err)?;
        sheet.set_freeze_panes(1, 0).map_err(err)?;

        self.export_coloc_detail_rows(filter, visible_labels, xlsx_row_writer(sheet))?;

        workbook.save(export_path).map_err(err)?;
        Ok(())
    }

    // -------------------------------------------------------------------------

    /// Streams the rows matching `filter` (or grouped/aggregated rows, when
    /// `group.group_by != GroupBy::None`) to `emit_row` — the header labels
    /// first, then each data row — instead of materializing the whole
    /// matching result set as one `Vec` in memory.
    ///
    /// `base_specs` are the per-ROI column specs from the table (carrying the
    /// current visibility selection), so the export mirrors what is shown:
    /// - grouped → one aggregated row per group over the visible metrics
    ///   (always a single, small pass — the aggregated result set is never
    ///   large regardless of how many ROIs it summarizes);
    /// - otherwise → per-ROI rows for the visible columns only, via a single
    ///   DB cursor over every matching row (see `DuckDbReader::stream_rois`)
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

        let reader = self.results_loader.open_reader()?;
        let roi_filter = to_roi_filter(DatabaseFilter { needs_intensities: true, ..filter });
        let mut row_idx = 0usize;
        reader.stream_rois(&roi_filter, |roi| {
            let display = to_display_row(row_idx, &roi, base_specs);
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
    /// every matching source ROI (see `DuckDbReader::stream_rois`), fetching
    /// only the colocalization partner ROIs each batch of
    /// `COLOC_PARTNER_BATCH_SIZE` source rows actually references (via
    /// `DatabaseFilter::object_id_filter`) instead of every ROI in the image.
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
        let headers: Vec<String> =
            specs.iter().filter(|c| c.visible).map(|c| c.label.clone()).collect();
        emit_row(&headers)?;

        let reader = self.results_loader.open_reader()?;
        let roi_filter = to_roi_filter(DatabaseFilter { needs_intensities: true, ..filter });

        let mut batch: Vec<RoiRow> = Vec::with_capacity(COLOC_PARTNER_BATCH_SIZE);
        reader.stream_rois(&roi_filter, |roi| {
            batch.push(roi);
            if batch.len() >= COLOC_PARTNER_BATCH_SIZE {
                flush_coloc_detail_batch(&reader, &mut batch, &specs, &mut emit_row)?;
            }
            Ok(())
        })?;
        flush_coloc_detail_batch(&reader, &mut batch, &specs, &mut emit_row)
    }
}

/// Resolves colocalization partners for one accumulated batch of source ROIs
/// against `reader` (the same connection `export_coloc_detail_rows`'s source
/// stream is still open on — see `DuckDbReader::stream_rois`'s doc comment
/// for why nesting a second query here is safe), flattens and emits the
/// batch's rows, then clears it for the next batch. A no-op on an empty
/// batch (the final flush after a source count that divides evenly).
fn flush_coloc_detail_batch(
    reader: &DuckDbReader,
    batch: &mut Vec<RoiRow>,
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
        reader.get_rois(&to_roi_filter(DatabaseFilter {
            object_id_filter: Some(ids),
            page_size: 0,
            needs_intensities: true,
            ..Default::default()
        }))?
    };

    for row in flatten_coloc_rows(batch, &partner_batch, specs) {
        let values: Vec<String> =
            specs.iter().zip(row.values.iter()).filter(|(col, _)| col.visible).map(|(_, v)| v.clone()).collect();
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::results::results_loader::{build_column_specs, AggFunc};
    use crate::results::test_support::{
        decode_object_id_idx, seed_large_coloc_db, seed_large_results_db, seed_results_db,
    };
    use calamine::{open_workbook_auto, Data, Reader};

    fn parse_csv(path: &Path) -> Vec<Vec<String>> {
        let mut reader = csv::Reader::from_path(path).expect("read csv back");
        let mut rows = vec![reader.headers().expect("csv header row").iter().map(String::from).collect::<Vec<_>>()];
        for record in reader.records() {
            rows.push(record.expect("csv data row").iter().map(String::from).collect());
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
        let range = workbook.worksheet_range_at(0).expect("sheet 0 present").expect("read sheet 0");
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
        header.iter().position(|h| h == label).unwrap_or_else(|| panic!("no {label:?} column in {header:?}"))
    }

    #[test]
    fn export_to_csv_writes_one_row_per_roi_with_the_right_values() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("results.duckdb");
        seed_results_db(&db_path);

        let loader = Arc::new(ResultsLoader::new(db_path));
        let exporter = ResultsExporter::new(loader);
        let specs = build_column_specs(&[0], &[]);

        let csv_path = dir.path().join("out.csv");
        exporter
            .export_to_csv(DatabaseFilter::default(), &GroupConfig::default(), &specs, &csv_path)
            .expect("export should succeed");

        let rows = parse_csv(&csv_path);
        assert_eq!(rows.len(), 3, "header + 2 ROI rows");

        let header = &rows[0];
        let class_col = col_index(header, "Class");
        let image_col = col_index(header, "Image");
        let area_col = col_index(header, "Area (px\u{00B2})");
        let ch0_avg_col = col_index(header, "Ch0 Avg (bit)");

        let by_image: std::collections::HashMap<&str, &Vec<String>> =
            rows[1..].iter().map(|r| (r[image_col].as_str(), r)).collect();

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
            .export_to_csv(DatabaseFilter::default(), &GroupConfig::default(), &specs, &csv_path)
            .expect("export should succeed");

        let rows = parse_csv(&csv_path);
        assert!(!rows[0].contains(&"Class".to_string()), "hidden column must not appear in the header");
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
        let filter = DatabaseFilter { image_filter: Some(vec!["img1.tif".to_string()]), ..Default::default() };
        exporter.export_to_csv(filter, &GroupConfig::default(), &specs, &csv_path).unwrap();

        let rows = parse_csv(&csv_path);
        assert_eq!(rows.len(), 2, "header + only the one matching ROI");
    }

    #[test]
    fn export_to_csv_grouped_writes_one_aggregated_row_per_image() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("results.duckdb");
        seed_results_db(&db_path);

        let loader = Arc::new(ResultsLoader::new(db_path));
        let exporter = ResultsExporter::new(loader);
        let specs = build_column_specs(&[0], &[]);

        let group = GroupConfig { group_by: GroupBy::Image, aggs: vec![AggFunc::Avg], ..Default::default() };
        let csv_path = dir.path().join("grouped.csv");
        exporter.export_to_csv(DatabaseFilter::default(), &group, &specs, &csv_path).unwrap();

        let rows = parse_csv(&csv_path);
        // One image per source ROI here, so grouping by image still yields
        // one row per ROI - this exercises the `group.group_by != None`
        // branch (`aggregate_rois_sql`) end-to-end rather than proving a
        // specific row count.
        assert_eq!(rows.len(), 3, "header + one aggregated row per distinct image");
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
            .export_to_xlsx(DatabaseFilter::default(), &GroupConfig::default(), &specs, &xlsx_path)
            .expect("export should succeed");

        let bytes = std::fs::read(&xlsx_path).expect("xlsx file should exist");
        assert!(bytes.len() > 100, "workbook should have real content, not just a stub file");
        // XLSX is a ZIP container - "PK\x03\x04" is the local-file-header
        // magic every valid ZIP (and therefore every valid XLSX) starts with.
        assert_eq!(&bytes[0..4], b"PK\x03\x04", "not a valid XLSX/ZIP file");
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
    // `DuckDbReader::stream_rois`'s single-cursor implementation (each
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
            .export_to_csv(DatabaseFilter::default(), &GroupConfig::default(), &specs, &csv_path)
            .expect("export should succeed");

        let rows = parse_csv(&csv_path);
        assert_eq!(rows.len(), n + 1, "header + one row per seeded ROI, none dropped or duplicated");

        let area_col = col_index(&rows[0], "Area (px\u{00B2})");
        let areas: Vec<u64> = rows[1..].iter().map(|r| r[area_col].parse().unwrap()).collect();
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
            .export_to_xlsx(DatabaseFilter::default(), &GroupConfig::default(), &specs, &xlsx_path)
            .expect("export should succeed");

        let rows = parse_xlsx(&xlsx_path);
        assert_eq!(rows.len(), n + 1, "header + one row per seeded ROI, none dropped or duplicated");

        let area_col = col_index(&rows[0], "Area (px\u{00B2})");
        let areas: Vec<u64> = rows[1..].iter().map(|r| r[area_col].parse().unwrap()).collect();
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
        let filter = DatabaseFilter { class_filter: Some(vec!["ClassA".to_string()]), ..Default::default() };
        exporter.export_coloc_detail_to_csv(filter, None, &csv_path).expect("export should succeed");

        let rows = parse_csv(&csv_path);
        assert_eq!(rows.len(), n_sources + 1, "header + exactly one flattened row per source ROI");

        let roi_id_col = col_index(&rows[0], "ROI ID");
        let partner_id_col = col_index(&rows[0], "Coloc ClassB ROI ID");

        let mut seen_source_idx = vec![false; n_sources];
        for row in &rows[1..] {
            let source_idx = decode_object_id_idx(&row[roi_id_col]);
            let partner_idx = decode_object_id_idx(&row[partner_id_col]);
            assert_eq!(
                partner_idx,
                source_idx % n_partners,
                "row for source {source_idx} resolved the wrong partner across an export page boundary"
            );
            assert!(!seen_source_idx[source_idx], "source ROI {source_idx} exported more than once");
            seen_source_idx[source_idx] = true;
        }
        assert!(seen_source_idx.into_iter().all(|seen| seen), "every source ROI must be exported exactly once");
    }
}
