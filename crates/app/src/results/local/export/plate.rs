//! # plate
//!
//! The plate, its wells and the image heatmaps: as grids or as a flat list.
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-11-22
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.

use super::*;

impl ResultExport {
    pub(super) fn export_plate_and_well(
        &self,
        database: &ResultsGenerator,
        cancel: &AtomicBool,
        on_progress: ExportProgress,
    ) -> Result<(), InternalErrors> {
        if !self.with_plate_view_heatmap && !self.with_well_view_heatmap {
            return Ok(());
        }

        let classes_all = database.get_object_classes()?;
        let target_classes: Vec<ObjectClass> = if self.object_classes.is_empty() {
            classes_all.iter().map(|class| class.global()).collect()
        } else {
            self.object_classes.clone()
        };
        let available_columns = database.get_available_columns()?;
        let planes = ExportPlanes::of(database, self)?;

        let aggregable_columns: Vec<&Column> = self
            .columns
            .iter()
            .filter(|column| column.is_aggregable())
            .collect();
        let no_combos = target_classes.is_empty()
            || aggregable_columns.is_empty()
            || self.aggregations.is_empty();

        // Plate and well each run their own independent query + document,
        // rather than one combined pass over both - `get_group_by_plate_multi`/
        // `get_wells_for_plate_multi` already batch every (class, column,
        // aggregation) combination into `target_classes.len()` scans each
        // (see their own doc comments), so splitting them here just means
        // skipping one side's scan and file entirely when its flag is off,
        // instead of always paying for both and discarding whichever one
        // wasn't wanted.
        if self.with_plate_view_heatmap {
            let mut plate_workbook = Workbook::new();
            let mut plate_names = SheetNamer::new();
            // A tab per class, plane by plane.
            for &plane in &planes.planes {
                let mut plate_grids_iter = if no_combos {
                    Vec::new().into_iter()
                } else {
                    let plate_filter = PlateFilterMulti {
                        plane,
                        grouping: self.grouping.clone(),
                        aggregation: self.aggregations.clone(),
                        object_class: target_classes.clone(),
                        column: aggregable_columns.iter().map(|c| (*c).clone()).collect(),
                        color_schema: self.color_schema.clone(),
                        color_scale: self.color_scale.clone(),
                        plate_size: self.plate_size,
                    };
                    database
                        .get_group_by_plate_multi(&plate_filter, &View::Heatmap)?
                        .into_iter()
                };

                for (class_idx, class) in target_classes.iter().enumerate() {
                    check_cancelled(cancel)?;
                    let class_label = class_display_label(*class, &classes_all);
                    on_progress(
                        &format!("Exporting Plate: {class_label}"),
                        class_idx + 1,
                        target_classes.len(),
                    );

                    let plate_sheet = plate_workbook.add_worksheet();
                    plate_sheet
                        .set_name(plate_names.unique(&planes.tab_name(plane, &class_label)))
                        .map_err(xlsx_err)?;
                    let mut plate_row = 0u32;

                    for column in &aggregable_columns {
                        let column_label =
                            column_display_name(column, &available_columns, &classes_all);
                        for aggregation in &self.aggregations {
                            let plate_grid = plate_grids_iter.next().ok_or_else(|| {
                                InternalErrors::Io(
                                    "plate export: plate result count desynced from class x \
                                 column x aggregation combinations"
                                        .to_string(),
                                )
                            })?;
                            let caption =
                                format!("{column_label} — {}", aggregation_label(aggregation));
                            plate_row =
                                write_grid_block(plate_sheet, plate_row, &caption, &plate_grid)?;
                        }
                    }
                }
            }

            plate_workbook
                .save(self.output_path("plate.xlsx"))
                .map_err(xlsx_err)?;
        }

        if self.with_well_view_heatmap {
            let mut well_workbook = Workbook::new();
            let mut well_names = SheetNamer::new();
            // A tab per class, plane by plane.
            for &plane in &planes.planes {
                let mut well_heatmaps_iter = if no_combos {
                    Vec::new().into_iter()
                } else {
                    let wells_filter = WellsBatchFilterMulti {
                        plane,
                        grouping: self.grouping.clone(),
                        aggregation: self.aggregations.clone(),
                        object_class: target_classes.clone(),
                        column: aggregable_columns.iter().map(|c| (*c).clone()).collect(),
                        color_schema: self.color_schema.clone(),
                        color_scale: self.color_scale.clone(),
                        well_size: self.well_size,
                        well_order: self.well_order.clone(),
                    };
                    database
                        .get_wells_for_plate_multi(&wells_filter, &View::Heatmap)?
                        .into_iter()
                };

                for (class_idx, class) in target_classes.iter().enumerate() {
                    check_cancelled(cancel)?;
                    let class_label = class_display_label(*class, &classes_all);
                    on_progress(
                        &format!("Exporting Well: {class_label}"),
                        class_idx + 1,
                        target_classes.len(),
                    );

                    let well_sheet = well_workbook.add_worksheet();
                    well_sheet
                        .set_name(well_names.unique(&planes.tab_name(plane, &class_label)))
                        .map_err(xlsx_err)?;
                    let mut well_row = 0u32;

                    for column in &aggregable_columns {
                        let column_label =
                            column_display_name(column, &available_columns, &classes_all);
                        for aggregation in &self.aggregations {
                            let mut well_heatmaps = well_heatmaps_iter.next().ok_or_else(|| {
                                InternalErrors::Io(
                                    "well export: well result count desynced from class x \
                                 column x aggregation combinations"
                                        .to_string(),
                                )
                            })?;
                            let caption =
                                format!("{column_label} — {}", aggregation_label(aggregation));

                            // Sorted for deterministic, well-id-ordered output -
                            // `well_heatmaps` is a `HashMap`, so its own
                            // iteration order isn't meaningful on its own.
                            let mut well_ids: Vec<String> = well_heatmaps.keys().cloned().collect();
                            well_ids.sort();

                            for well_id in &well_ids {
                                let well_caption = format!("{caption} — Well {well_id}");
                                let Some(well_grid) = well_heatmaps.remove(well_id) else {
                                    continue;
                                };
                                well_row = write_grid_block(
                                    well_sheet,
                                    well_row,
                                    &well_caption,
                                    &well_grid,
                                )?;
                            }
                        }
                    }
                }
            }

            well_workbook
                .save(self.output_path("well.xlsx"))
                .map_err(xlsx_err)?;
        }

        Ok(())
    }

    // `plate_list.xlsx` and `well_list.xlsx`: the same aggregated data as
    // `export_plate_and_well`, but pivoted into one plain table per document
    // instead of a grid-per-class-tab — one row per well (or, in
    // `well_list.xlsx`, per well+field), one column per (class, column,
    // aggregation) combination, so every class sits side by side in the
    // same tab rather than needing its own. `well_list.xlsx` also carries
    // an "Image" column (the source image for that field), independent of
    // which class/column/aggregation combination is being looked at.
    // Independently toggled by `with_plate_view_list`/`with_well_view_list`,
    // same as `export_plate_and_well`'s own heatmap flags.
    //
    // Only ever needs `View::List` data - never overlaps in query cost with
    // `export_plate_and_well`'s `View::Heatmap`-only fetches above, even
    // though both run in the same export when this is enabled.
    pub(super) fn export_plate_and_well_as_flat_list(
        &self,
        database: &ResultsGenerator,
        cancel: &AtomicBool,
        on_progress: ExportProgress,
    ) -> Result<(), InternalErrors> {
        if !self.with_plate_view_list && !self.with_well_view_list {
            return Ok(());
        }

        let classes_all = database.get_object_classes()?;
        let target_classes: Vec<ObjectClass> = if self.object_classes.is_empty() {
            classes_all.iter().map(|class| class.global()).collect()
        } else {
            self.object_classes.clone()
        };
        let available_columns = database.get_available_columns()?;
        let planes = ExportPlanes::of(database, self)?;
        let aggregable_columns: Vec<&Column> = self
            .columns
            .iter()
            .filter(|column| column.is_aggregable())
            .collect();
        let combo_count = target_classes.len() * aggregable_columns.len() * self.aggregations.len();
        let no_combos = combo_count == 0;

        // Shared by both outputs and independent of which data actually
        // gets fetched below - built once, up front, purely from the
        // requested class/column/aggregation combinations.
        let mut combo_labels: Vec<String> = Vec::with_capacity(combo_count);
        for class in &target_classes {
            let class_label = class_display_label(*class, &classes_all);
            for column in &aggregable_columns {
                let column_label = column_display_name(column, &available_columns, &classes_all);
                for aggregation in &self.aggregations {
                    combo_labels.push(format!(
                        "{class_label} — {column_label} — {}",
                        aggregation_label(aggregation)
                    ));
                }
            }
        }

        // Plate and well each run their own independent query + document -
        // see the identical reasoning in `export_plate_and_well`.
        if self.with_plate_view_list {
            // Plane by plane: each plane's wells behind its "T"/"Z".
            let mut rows: Vec<(Vec<String>, Vec<Option<f64>>, bool)> = Vec::new();
            for &plane in &planes.planes {
                let mut plate_values: HashMap<String, Vec<Option<f64>>> = HashMap::new();
                // Whether at least one image in that well is disabled - see
                // `Cell::disabled`. Independent of class/column/aggregation (it
                // only depends on which images belong to the well), but merged
                // with `|=` across every combo anyway rather than assumed, so a
                // future combo-dependent source of `disabled` wouldn't silently
                // under-report it.
                let mut plate_disabled: HashMap<String, bool> = HashMap::new();
                let mut plate_lists_iter = if no_combos {
                    Vec::new().into_iter()
                } else {
                    let plate_filter = PlateFilterMulti {
                        plane,
                        grouping: self.grouping.clone(),
                        aggregation: self.aggregations.clone(),
                        object_class: target_classes.clone(),
                        column: aggregable_columns.iter().map(|c| (*c).clone()).collect(),
                        color_schema: self.color_schema.clone(),
                        color_scale: self.color_scale.clone(),
                        plate_size: self.plate_size,
                    };
                    database
                        .get_group_by_plate_multi(&plate_filter, &View::List)?
                        .into_iter()
                };

                let mut combo_idx = 0usize;
                for (class_idx, class) in target_classes.iter().enumerate() {
                    check_cancelled(cancel)?;
                    on_progress(
                        &format!(
                            "Exporting Flat List (Plate): {}",
                            class_display_label(*class, &classes_all)
                        ),
                        class_idx + 1,
                        target_classes.len(),
                    );
                    for _ in &aggregable_columns {
                        for _ in &self.aggregations {
                            let plate_list = plate_lists_iter.next().ok_or_else(|| {
                                InternalErrors::Io(
                                    "plate flat list export: plate result count desynced from \
                                 class x column x aggregation combinations"
                                        .to_string(),
                                )
                            })?;
                            for (well_id, row) in plate_list.row_names.iter().zip(&plate_list.rows)
                            {
                                // Seeded unconditionally (not only when a value
                                // is found) so a well that has images but zero
                                // matching objects for *every* combo still gets
                                // a row - written as "-" per cell, not silently
                                // dropped for having nothing to show.
                                let entry = plate_values
                                    .entry(well_id.clone())
                                    .or_insert_with(|| vec![None; combo_count]);
                                if let Some(value) = row.get(1).and_then(cell_to_f64) {
                                    entry[combo_idx] = Some(value);
                                }
                                if let Some(cell) = row.get(1) {
                                    *plate_disabled.entry(well_id.clone()).or_insert(false) |=
                                        cell.disabled;
                                }
                            }
                            combo_idx += 1;
                        }
                    }
                }

                let mut plate_rows: Vec<(String, Vec<Option<f64>>)> =
                    plate_values.into_iter().collect();
                plate_rows.sort_by(|(a, _), (b, _)| a.cmp(b));
                rows.extend(plate_rows.into_iter().map(|(well_id, values)| {
                    let disabled = plate_disabled.get(&well_id).copied().unwrap_or(false);
                    let mut keys = planes.texts(plane);
                    keys.push(well_id);
                    (keys, values, disabled)
                }));
            }
            let key_headers: Vec<String> = planes
                .header()
                .into_iter()
                .chain(["Well".to_string()])
                .collect();
            write_flat_pivot(
                &self.output_path("plate_list.xlsx"),
                "Plate",
                &key_headers.iter().map(String::as_str).collect::<Vec<_>>(),
                &combo_labels,
                rows.into_iter(),
            )?;
        }

        if self.with_well_view_list {
            // Plane by plane, like the plate's.
            let mut rows: Vec<(Vec<String>, Vec<Option<f64>>, bool)> = Vec::new();
            for &plane in &planes.planes {
                let mut well_values: HashMap<(String, String), Vec<Option<f64>>> = HashMap::new();
                let mut well_images: HashMap<(String, String), String> = HashMap::new();
                // Whether that specific field's own image is disabled - see
                // `Cell::disabled` on `get_group_by_well`'s per-field cells.
                let mut well_disabled: HashMap<(String, String), bool> = HashMap::new();
                let mut well_lists_iter = if no_combos {
                    Vec::new().into_iter()
                } else {
                    let wells_filter = WellsBatchFilterMulti {
                        plane,
                        grouping: self.grouping.clone(),
                        aggregation: self.aggregations.clone(),
                        object_class: target_classes.clone(),
                        column: aggregable_columns.iter().map(|c| (*c).clone()).collect(),
                        color_schema: self.color_schema.clone(),
                        color_scale: self.color_scale.clone(),
                        well_size: self.well_size,
                        well_order: self.well_order.clone(),
                    };
                    database
                        .get_wells_for_plate_multi(&wells_filter, &View::List)?
                        .into_iter()
                };

                let mut combo_idx = 0usize;
                for (class_idx, class) in target_classes.iter().enumerate() {
                    check_cancelled(cancel)?;
                    on_progress(
                        &format!(
                            "Exporting Flat List (Well): {}",
                            class_display_label(*class, &classes_all)
                        ),
                        class_idx + 1,
                        target_classes.len(),
                    );
                    for _ in &aggregable_columns {
                        for _ in &self.aggregations {
                            let well_lists = well_lists_iter.next().ok_or_else(|| {
                                InternalErrors::Io(
                                    "well flat list export: well result count desynced from \
                                 class x column x aggregation combinations"
                                        .to_string(),
                                )
                            })?;
                            for (well_id, result) in &well_lists {
                                for (field_idx, row) in result.row_names.iter().zip(&result.rows) {
                                    let key = (well_id.clone(), field_idx.clone());
                                    // Seeded unconditionally - see the matching
                                    // comment on the plate loop above.
                                    let entry = well_values
                                        .entry(key.clone())
                                        .or_insert_with(|| vec![None; combo_count]);
                                    if let Some(value) = row.get(1).and_then(cell_to_f64) {
                                        entry[combo_idx] = Some(value);
                                    }
                                    if let Some(cell) = row.get(1) {
                                        *well_disabled.entry(key.clone()).or_insert(false) |=
                                            cell.disabled;
                                        if let Some((image_name, _)) = &cell.search_key {
                                            well_images
                                                .entry(key)
                                                .or_insert_with(|| image_name.clone());
                                        }
                                    }
                                }
                            }
                            combo_idx += 1;
                        }
                    }
                }

                let mut well_rows: Vec<((String, String), Vec<Option<f64>>)> =
                    well_values.into_iter().collect();
                well_rows.sort_by(|(a, _), (b, _)| {
                    a.0.cmp(&b.0).then_with(|| {
                        a.1.parse::<u32>()
                            .ok()
                            .cmp(&b.1.parse::<u32>().ok())
                            .then_with(|| a.1.cmp(&b.1))
                    })
                });
                rows.extend(well_rows.into_iter().map(|((well_id, field_idx), values)| {
                    let key = (well_id.clone(), field_idx.clone());
                    let image = well_images.get(&key).cloned().unwrap_or_default();
                    let disabled = well_disabled.get(&key).copied().unwrap_or(false);
                    let mut keys = planes.texts(plane);
                    keys.extend([well_id, field_idx, image]);
                    (keys, values, disabled)
                }));
            }
            let key_headers: Vec<String> = planes
                .header()
                .into_iter()
                .chain(["Well", "Field", "Image"].map(String::from))
                .collect();
            write_flat_pivot(
                &self.output_path("well_list.xlsx"),
                "Well",
                &key_headers.iter().map(String::as_str).collect::<Vec<_>>(),
                &combo_labels,
                rows.into_iter(),
            )?;
        }

        Ok(())
    }

    // One `heatmap_{image}.xlsx` per image — same one-tab-per-class,
    // stacked-column/aggregation-blocks shape as `plate.xlsx`/`well.xlsx`
    // above, just partitioned by image into separate files instead of
    // sharing one document, since a heatmap is inherently local to a single
    // image. Same single-plane caveat as `export_plate_and_well` applies:
    // only the first z/t in the given ranges is rendered.
    pub(super) fn export_heatmap(
        &self,
        database: &ResultsGenerator,
        cancel: &AtomicBool,
        on_progress: ExportProgress,
    ) -> Result<(), InternalErrors> {
        let classes_all = database.get_object_classes()?;
        let target_classes: Vec<ObjectClass> = if self.object_classes.is_empty() {
            classes_all.iter().map(|class| class.global()).collect()
        } else {
            self.object_classes.clone()
        };
        let available_columns = database.get_available_columns()?;
        let images = resolve_images(database, self)?;
        let planes = ExportPlanes::of(database, self)?;

        for (image_idx, image) in images.iter().enumerate() {
            check_cancelled(cancel)?;
            on_progress(
                &format!("Exporting Heatmap: {image}"),
                image_idx + 1,
                images.len(),
            );
            let mut workbook = Workbook::new();
            let mut names = SheetNamer::new();

            // A tab per class, plane by plane.
            for &plane in &planes.planes {
                for class in &target_classes {
                    let class_label = class_display_label(*class, &classes_all);
                    let sheet = workbook.add_worksheet();
                    sheet
                        .set_name(names.unique(&planes.tab_name(plane, &class_label)))
                        .map_err(xlsx_err)?;

                    let mut row = 0u32;
                    // An image measurement has no place within the image.
                    for column in self
                        .columns
                        .iter()
                        .filter(|column| column.is_aggregable() && !column.is_image_measurement())
                    {
                        let column_label =
                            column_display_name(column, &available_columns, &classes_all);
                        for aggregation in &self.aggregations {
                            let caption =
                                format!("{column_label} — {}", aggregation_label(aggregation));
                            let filter = ImageHeatmapFilter {
                                plane,
                                image_rel_path: image.clone(),
                                aggregation: aggregation.clone(),
                                object_class: *class,
                                column: column.clone(),
                                color_schema: self.color_schema.clone(),
                                color_scale: self.color_scale.clone(),
                                square_size: self.square_size,
                            };
                            let grid = database.get_image_heatmap(&filter, &View::Heatmap)?;
                            row = write_grid_block(sheet, row, &caption, &grid)?;
                        }
                    }
                }
            }

            let file_name = format!(
                "heatmap_{}.xlsx",
                sanitize_filename_component(&image_stub(image))
            );
            workbook
                .save(self.output_path(&file_name))
                .map_err(xlsx_err)?;
        }
        Ok(())
    }
}

pub(super) const GRID_CELL_PX: u32 = 60;

pub(super) const GRID_HEADER_PX: u32 = 24;

pub(super) const GRID_NUM_FORMAT: &str = "0.00";

pub(super) const EMPTY_CELL_BG: u32 = 0xE0E0E0;

/// Writes one square, colored grid block (`result`, a `View::Heatmap`
/// `DatabaseResult`) starting at `start_row`: a bold caption, a header row
/// of `result.column_names`, then one row per `result.row_names` with that
/// row's label in column 0 and its cells across — see `write_cell` for how
/// a cell's value/color are actually rendered. Returns the next free row
/// (one blank row after the block).
pub(super) fn write_grid_block(
    worksheet: &mut Worksheet,
    start_row: u32,
    caption: &str,
    result: &DatabaseResult,
) -> Result<u32, InternalErrors> {
    let bold = Format::new().set_bold().set_font_size(EXPORT_FONT_SIZE);
    // Row/col header cells (the top row of column labels and the left
    // column of row labels) — solid black fill, white bold text, so they
    // read as a distinct axis rather than blending into the data cells.
    let header_format = Format::new()
        .set_bold()
        .set_background_color(Color::Black)
        .set_font_color(Color::White)
        .set_font_size(EXPORT_FONT_SIZE);
    worksheet
        .write_with_format(start_row, 0, caption, &bold)
        .map_err(xlsx_err)?;

    let header_row = start_row + 1;
    worksheet
        .set_row_height_pixels(header_row, GRID_HEADER_PX)
        .map_err(xlsx_err)?;
    worksheet
        .set_column_width_pixels(0, GRID_HEADER_PX)
        .map_err(xlsx_err)?;
    for (col_idx, col_name) in result.column_names.iter().enumerate() {
        let col = (col_idx + 1) as u16;
        worksheet
            .write_with_format(header_row, col, col_name.as_str(), &header_format)
            .map_err(xlsx_err)?;
        worksheet
            .set_column_width_pixels(col, GRID_CELL_PX)
            .map_err(xlsx_err)?;
    }

    for (row_idx, row_name) in result.row_names.iter().enumerate() {
        let row = header_row + 1 + row_idx as u32;
        worksheet
            .write_with_format(row, 0, row_name.as_str(), &header_format)
            .map_err(xlsx_err)?;
        worksheet
            .set_row_height_pixels(row, GRID_CELL_PX)
            .map_err(xlsx_err)?;
        for (col_idx, cell) in result.rows[row_idx].iter().enumerate() {
            let col = (col_idx + 1) as u16;
            write_cell(worksheet, row, col, cell, Some(GRID_NUM_FORMAT))?;
        }
    }

    Ok(header_row + 1 + result.row_names.len() as u32 + 1)
}

/// Writes one pivoted flat-list document: a single sheet named
/// `sheet_name`, headed by `key_labels` (e.g. `["Well"]` or
/// `["Well", "Field", "Image"]`) followed by `combo_labels` (one column per
/// class/column/aggregation combination), then one row per `rows` entry —
/// each a `(key values, one Some(value)-or-None per combo_labels column,
/// disabled)` triple. Every row here corresponds to a real well/field (the
/// caller only ever seeds one for a key it actually saw - see
/// `export_plate_and_well_as_flat_list`), so a `None` here only ever means
/// "this well/field has no matching data for this particular combination"
/// (e.g. no objects of some other class), never "no such well/field at
/// all" - written as `"-"`, matching `write_cell`'s own convention for the
/// same case in the grid export. `disabled` (at least one image behind
/// this row is disabled - see `Cell::disabled`) strikes the whole row
/// through. Unlike `write_grid_block`, this is a plain table, not a matrix
/// - no square sizing or per-cell color.
pub(super) fn write_flat_pivot(
    path: &Path,
    sheet_name: &str,
    key_labels: &[&str],
    combo_labels: &[String],
    rows: impl Iterator<Item = (Vec<String>, Vec<Option<f64>>, bool)>,
) -> Result<(), InternalErrors> {
    let mut workbook = Workbook::new();
    let sheet = workbook.add_worksheet();
    sheet.set_name(sheet_name).map_err(xlsx_err)?;
    let bold = Format::new().set_bold().set_font_size(EXPORT_FONT_SIZE);
    // `GRID_NUM_FORMAT` bounds an AVG/etc.'s raw full-precision float to 2
    // decimals, same reasoning as the grid export - otherwise
    // `LIST_VALUE_COL_PX` still isn't enough room for it.
    let plain = Format::new()
        .set_font_size(EXPORT_FONT_SIZE)
        .set_num_format(GRID_NUM_FORMAT);
    let strikethrough = Format::new()
        .set_font_strikethrough()
        .set_font_size(EXPORT_FONT_SIZE)
        .set_num_format(GRID_NUM_FORMAT);

    for (col_idx, label) in key_labels.iter().enumerate() {
        sheet
            .write_with_format(0, col_idx as u16, *label, &bold)
            .map_err(xlsx_err)?;
    }
    let key_cols = key_labels.len();
    for (i, label) in combo_labels.iter().enumerate() {
        let col = (key_cols + i) as u16;
        sheet
            .write_with_format(0, col, label.as_str(), &bold)
            .map_err(xlsx_err)?;
        sheet
            .set_column_width_pixels(col, LIST_VALUE_COL_PX)
            .map_err(xlsx_err)?;
    }

    for (row_idx, (keys, values, disabled)) in rows.enumerate() {
        let row = (row_idx + 1) as u32;
        let format = if disabled { &strikethrough } else { &plain };
        for (col_idx, key) in keys.iter().enumerate() {
            sheet
                .write_string_with_format(row, col_idx as u16, key.as_str(), format)
                .map_err(xlsx_err)?;
        }
        for (i, value) in values.iter().enumerate() {
            let col = (key_cols + i) as u16;
            match value {
                Some(v) => {
                    sheet
                        .write_number_with_format(row, col, *v, format)
                        .map_err(xlsx_err)?;
                }
                None => {
                    sheet
                        .write_string_with_format(row, col, "-", format)
                        .map_err(xlsx_err)?;
                }
            }
        }
    }

    workbook.save(path).map_err(xlsx_err)?;
    Ok(())
}
