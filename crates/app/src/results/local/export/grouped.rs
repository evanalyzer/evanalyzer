//! # grouped
//!
//! Grouped by image: a row per image and class, written block by block.
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-11-22
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.

use super::*;

impl ResultExport {
    // `grouped_by_image.xlsx`: one row per image, one column per (selected
    // column × selected aggregation) combination — `get_grouped_by_image`'s
    // own shape, just paginated through in full rather than exposed live.
    // Single plane only (like `export_plate_and_well`/`export_heatmap`):
    // grouping by image is itself a per-plane aggregate, so there's no
    // meaningful "stack every t/z" equivalent the way there is for the
    // per-object List export.
    pub(super) fn export_grouped_by_image(
        &self,
        database: &ResultsGenerator,
        cancel: &AtomicBool,
        on_progress: ExportProgress,
    ) -> Result<(), InternalErrors> {
        check_cancelled(cancel)?;
        on_progress("Exporting Grouped Image List", 0, 1);

        // Group by image always lists every enabled image: the image
        // filter is not applied here.
        let object_classes = if self.object_classes.is_empty() {
            None
        } else {
            Some(self.object_classes.clone())
        };
        let columns: Vec<Column> = self
            .columns
            .iter()
            .filter(|column| column.is_aggregable())
            .cloned()
            .collect();
        // Each plane's rows, time point by time point, behind its "T"/"Z".
        let planes = ExportPlanes::of(database, self)?;
        // Each row ends with its image's marker and note - when an image
        // has one.
        let reviews = ImageReviews::of(database)?;
        let filter_of = |plane: PlaneFilter| GroupedByImageFilter {
            plane,
            images: crate::results::ImageSet::All,
            object_classes: object_classes.clone(),
            columns: columns.clone(),
            aggregation: self.aggregations.clone(),
            page: Pagination {
                limit: 0,
                after: None,
            },
            transpond_table: self.transpond_table,
            grouping: None,
            empty_classes: true,
            sort: None,
        };

        // Written page by page as the rows come (100,000 images × 5
        // classes: 500,000 rows - kept in RAM, they took GBs).
        match self.format {
            ExportFormat::XLSX => {
                let mut workbook = Workbook::new();
                let sheet = workbook
                    .add_worksheet_with_constant_memory()
                    .map_err(xlsx_err)?;
                sheet.set_name("Grouped by Image").map_err(xlsx_err)?;
                let header_format = Format::new().set_bold().set_font_size(EXPORT_FONT_SIZE);
                let mut next_row: u32 = 0;
                for &plane in &planes.planes {
                    let plane_cells = planes.cells(plane);
                    stream_grouped_by_image_pages(
                        database,
                        cancel,
                        &filter_of(plane),
                        GROUPED_ROWS_PER_PAGE,
                        |page| {
                            if next_row == 0 {
                                let header = planes
                                    .header()
                                    .into_iter()
                                    .chain(page.column_names.iter().cloned())
                                    .chain(reviews.header());
                                for (col_idx, name) in header.enumerate() {
                                    sheet
                                        .write_with_format(
                                            0,
                                            col_idx as u16,
                                            name.as_str(),
                                            &header_format,
                                        )
                                        .map_err(xlsx_err)?;
                                }
                                next_row = 1;
                            }
                            for (row, (rel_path, _)) in page.rows.iter().zip(&page.row_locations) {
                                if next_row as usize >= XLSX_MAX_ROWS {
                                    return Err(InternalErrors::InvalidArgument(format!(
                                        "Grouped by image has more than {} rows - more than \
                                         an XLSX sheet holds: export it as CSV",
                                        XLSX_MAX_ROWS - 1
                                    )));
                                }
                                let review = reviews.cells(rel_path);
                                for (col_idx, cell) in
                                    plane_cells.iter().chain(row).chain(&review).enumerate()
                                {
                                    write_cell(sheet, next_row, col_idx as u16, cell, None)?;
                                }
                                next_row += 1;
                            }
                            Ok(())
                        },
                    )?;
                }
                workbook
                    .save(self.output_path("grouped_by_image.xlsx"))
                    .map_err(xlsx_err)?;
            }
            ExportFormat::CSV => {
                let path = self.output_path("grouped_by_image.csv");
                let mut out = create_csv_writer(&path)?;
                let write_err = |e: std::io::Error| {
                    InternalErrors::Internal(format!("could not write {}: {e}", path.display()))
                };
                let mut header_written = false;
                for &plane in &planes.planes {
                    let plane_cells = planes.cells(plane);
                    stream_grouped_by_image_pages(
                        database,
                        cancel,
                        &filter_of(plane),
                        GROUPED_ROWS_PER_PAGE,
                        |page| {
                            if !header_written {
                                let header: Vec<String> = planes
                                    .header()
                                    .into_iter()
                                    .chain(page.column_names.iter().cloned())
                                    .chain(reviews.header())
                                    .collect();
                                write_csv_row(&mut out, &header).map_err(write_err)?;
                                header_written = true;
                            }
                            for (row, (rel_path, _)) in page.rows.iter().zip(&page.row_locations) {
                                let review = reviews.cells(rel_path);
                                let cells: Vec<String> = plane_cells
                                    .iter()
                                    .chain(row)
                                    .chain(&review)
                                    .map(cell_text)
                                    .collect();
                                write_csv_row(&mut out, &cells).map_err(write_err)?;
                            }
                            Ok(())
                        },
                    )?;
                }
                use std::io::Write;
                out.flush().map_err(write_err)?;
            }
            // See the identical arm in `write_list_document` above.
            ExportFormat::Parquet => {
                return Err(InternalErrors::Internal(
                    "Parquet export doesn't support the Grouped-by-Image view — call export_as_parquet directly instead".to_string(),
                ));
            }
        }
        on_progress("Exporting Grouped Image List", 1, 1);
        Ok(())
    }
}

/// About how many rows a block of a grouped-by-image export has.
pub(super) const GROUPED_ROWS_PER_PAGE: usize = 20_000;

// Every enabled image's rows of `get_grouped_by_image` (the filter's
// `images` and `page` are ignored), a block of images at a time: each
// block's objects are aggregated once (a cursor over the rows aggregated
// every object after it again, each page), and `on_page` gets each block's
// rows as they come, so RAM stays at one block however many images there
// are. Called once with no rows (its column names) when no image is
// enabled.
pub(super) fn stream_grouped_by_image_pages(
    database: &ResultsGenerator,
    cancel: &AtomicBool,
    base_filter: &GroupedByImageFilter,
    rows_per_page: usize,
    mut on_page: impl FnMut(&DatabaseResult) -> Result<(), InternalErrors>,
) -> Result<(), InternalErrors> {
    let filter = GroupedByImageFilter {
        page: Pagination {
            limit: i32::MAX,
            after: None,
        },
        ..base_filter.clone()
    };
    let images = database.enabled_image_rel_paths()?;
    if images.is_empty() {
        return on_page(&database.get_grouped_by_image_of(&filter, ImageSelection::Listed(&[]))?);
    }
    let classes = database.get_object_classes()?.len().max(1);
    // About `rows_per_page` rows per block: a row per image and class.
    for block in images.chunks((rows_per_page / classes).max(1)) {
        check_cancelled(cancel)?;
        let (first, last) = (&block[0], &block[block.len() - 1]);
        let range = ImageSelection::EnabledRange { first, last };
        on_page(&database.get_grouped_by_image_of(&filter, range)?)?;
    }
    Ok(())
}
