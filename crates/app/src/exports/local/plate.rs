//! # plate
//!
//! The plate, its wells and the image heatmaps: as grids (XLSX) or as a
//! flat list - a data export's PLATE, WELL_GRID, WELL_LIST and
//! IMAGE_HEATMAP outputs (see `reused.rs`, which names the files).
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-11-22
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.

use crate::results::local::class_display_label;
use crate::results::{
    Aggregation, Cell, CellValue, ColorScale, ColorSchema, Column, ColumnEntry, DatabaseResult,
    ExportProgress, Grouping, ImageHeatmapFilter, PlaneFilter, PlateFilterMulti, PlateSize,
    ResultsSource, View, WellSize, WellsBatchFilterMulti,
};
use evanalyzer_cfg::core_types::{InternalErrors, ObjectClass};
use rust_xlsxwriter::{Color, Format, FormatAlign, Workbook, Worksheet, XlsxError};
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};

#[cfg(test)]
mod tests;

/// What a plate layout writes, and how: its grids or lists, of which
/// images, classes, columns and aggregations, planes and colours.
#[derive(Clone, Default)]
pub(super) struct PlateExport {
    pub output_dir: PathBuf,
    pub t_stacks: Option<std::range::Range<u32>>,
    pub z_stacks: Option<std::range::Range<u32>>,
    /// The images; empty: every image not left out.
    pub image_rel_paths: Vec<String>,
    pub columns: Vec<Column>,
    pub color_schema: ColorSchema,
    pub color_scale: ColorScale,
    pub grouping: Grouping,
    pub object_classes: Vec<ObjectClass>,
    pub aggregations: Vec<Aggregation>,
    pub plate_size: PlateSize,
    pub well_size: Option<WellSize>,
    pub well_order: Option<Vec<u32>>,
    pub square_size: Option<usize>,
    pub with_plate_view_heatmap: bool,
    pub with_well_view_heatmap: bool,
    pub with_plate_view_list: bool,
    pub with_well_view_list: bool,
    pub with_heatmap: bool,
}

impl PlateExport {
    /// Writes what is asked for into `output_dir` (`plate.xlsx`,
    /// `well.xlsx`, `plate_list.xlsx`, `well_list.xlsx`, `heatmap_<image>.xlsx`).
    pub(super) fn write(
        &self,
        database: &dyn ResultsSource,
        cancel: &AtomicBool,
    ) -> Result<(), InternalErrors> {
        let mut progress = |_: &str, _: usize, _: usize| {};
        let any = self.with_plate_view_heatmap
            || self.with_well_view_heatmap
            || self.with_plate_view_list
            || self.with_well_view_list
            || self.with_heatmap;
        if any {
            std::fs::create_dir_all(&self.output_dir).map_err(|e| {
                InternalErrors::Io(format!(
                    "Could not create {}: {e}",
                    self.output_dir.display()
                ))
            })?;
        }
        if self.with_plate_view_heatmap || self.with_well_view_heatmap {
            self.export_plate_and_well(database, cancel, &mut progress)?;
        }
        if self.with_plate_view_list || self.with_well_view_list {
            self.export_plate_and_well_as_flat_list(database, cancel, &mut progress)?;
        }
        if self.with_heatmap {
            self.export_heatmap(database, cancel, &mut progress)?;
        }
        Ok(())
    }

    fn output_path(&self, file_name: &str) -> PathBuf {
        self.output_dir.join(file_name)
    }

    /// The images: the picked ones, else every image not left out.
    fn images(&self, database: &dyn ResultsSource) -> Result<Vec<String>, InternalErrors> {
        let all = database.get_images()?;
        let wanted: HashSet<&str> = self.image_rel_paths.iter().map(String::as_str).collect();
        Ok(all
            .into_iter()
            .filter(|image| match wanted.is_empty() {
                true => !image.disabled,
                false => wanted.contains(image.rel_path.to_string_lossy().as_ref()),
            })
            .map(|image| image.rel_path.to_string_lossy().into_owned())
            .collect())
    }
}

impl PlateExport {
    fn export_plate_and_well(
        &self,
        database: &dyn ResultsSource,
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
    fn export_plate_and_well_as_flat_list(
        &self,
        database: &dyn ResultsSource,
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
    fn export_heatmap(
        &self,
        database: &dyn ResultsSource,
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
        let images = self.images(database)?;
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

const GRID_CELL_PX: u32 = 60;

const GRID_HEADER_PX: u32 = 24;

const GRID_NUM_FORMAT: &str = "0.00";

const EMPTY_CELL_BG: u32 = 0xE0E0E0;

/// Writes one square, colored grid block (`result`, a `View::Heatmap`
/// `DatabaseResult`) starting at `start_row`: a bold caption, a header row
/// of `result.column_names`, then one row per `result.row_names` with that
/// row's label in column 0 and its cells across — see `write_cell` for how
/// a cell's value/color are actually rendered. Returns the next free row
/// (one blank row after the block).
fn write_grid_block(
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
fn write_flat_pivot(
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

// -- Shared cell and sheet helpers --

fn xlsx_err(error: XlsxError) -> InternalErrors {
    InternalErrors::Internal(format!("XLSX export error: {error}"))
}

/// The font size, background fill, strikethrough and/or number format
/// `write_cell` applies for a given `Cell` - split out from `write_cell`
/// itself so the formatting logic can be unit-tested directly against
/// `Format` (which implements `Eq`) instead of only indirectly through a
/// written-and-reread `.xlsx` file. Every cell gets `EXPORT_FONT_SIZE` at
/// minimum. `bg_color` when it's set (`0` is every non-colored cell's
/// sentinel throughout `results_generator.rs`) takes priority since it's
/// real data (e.g. a class badge's own color); `EMPTY_CELL_BG` (see above)
/// comes next; `alternating_color` only ever paints `ALTERNATING_ROW_BG` as
/// a last-resort fallback, for a cell that has no color of its own to show.
/// `disabled` additionally strikes the text through, so a value derived
/// from (or, for a well/plate cell, affected by) a disabled image visibly
/// reads as "shown but excluded from the statistics" rather than looking
/// like any other value. `num_format` (e.g. `"0.00"`, used by the plate/
/// well/heatmap grids - see `write_grid_block`) is applied last, along with
/// centering the cell both horizontally and vertically - `num_format` is
/// only ever passed for a grid cell, never a flat list/grouped-by-image
/// one, so it doubles as "this is a grid cell" for alignment purposes too.
/// The number format keeps a long float from forcing the column far wider
/// than `GRID_CELL_PX` to avoid Excel's `#####` "too narrow to display"
/// placeholder.
/// Font color of a cell from a failed image - the GUI's light-theme
/// `Theme.warn`.
const FAILED_FONT_COLOR: u32 = 0xC07800;

fn cell_format(cell: &Cell, num_format: Option<&str>) -> Format {
    let mut format = Format::new().set_font_size(EXPORT_FONT_SIZE);
    if cell.disabled {
        // Gray out a disabled cell regardless of any heatmap/class color it
        // would otherwise carry - a struck-through value on top of its
        // normal color still read as "just another colored cell" at a
        // glance, same complaint as the GUI grid before it switched
        // disabled tiles to a flat gray fill instead of their heatmap
        // color.
        format = format.set_background_color(Color::RGB(EMPTY_CELL_BG));
        format = format.set_font_strikethrough();
    } else if cell.bg_color != 0 {
        format = format.set_background_color(Color::RGB(cell.bg_color));
    } else if matches!(cell.value, CellValue::Empty) && cell.search_key.is_some() {
        format = format.set_background_color(Color::RGB(EMPTY_CELL_BG));
    } else if cell.alternating_color {
        format = format.set_background_color(Color::RGB(ALTERNATING_ROW_BG));
    }
    if cell.failed || cell.any_failed {
        // The image's analysis stopped with an error (or, for a well, one
        // of its images' did): its values are incomplete / it's left out -
        // same warning color the GUI uses.
        format = format
            .set_font_color(Color::RGB(FAILED_FONT_COLOR))
            .set_italic();
    }
    if let Some(num_format) = num_format {
        format = format
            .set_num_format(num_format)
            .set_align(FormatAlign::Center)
            .set_align(FormatAlign::VerticalCenter);
    }
    format
}

/// Writes one `Cell` — its value, typed appropriately (`write_string`/
/// `write_number`, not everything flattened to text, so the sheet stays
/// sortable/usable as real data) rather than pre-formatted display text —
/// plus whatever `cell_format` computes for it. `num_format` is forwarded
/// straight to `cell_format` (see its doc comment) - pass `None` outside a
/// plate/well/heatmap grid. A real well/field with no value (`CellValue::Empty`
/// with a `search_key` - see `EMPTY_CELL_BG`) is written as `"-"`; a grid
/// position with no well/field at all is left a true blank.
fn write_cell(
    worksheet: &mut Worksheet,
    row: u32,
    col: u16,
    cell: &Cell,
    num_format: Option<&str>,
) -> Result<(), InternalErrors> {
    let format = cell_format(cell, num_format);

    match &cell.value {
        CellValue::Empty => {
            if cell.search_key.is_some() {
                // A real well/field with no value (see `EMPTY_CELL_BG`
                // above) - label it "-" rather than leaving the filled
                // cell looking blank.
                worksheet
                    .write_string_with_format(row, col, "-", &format)
                    .map_err(xlsx_err)?;
            } else {
                worksheet.write_blank(row, col, &format).map_err(xlsx_err)?;
            }
        }
        CellValue::String(s) | CellValue::Class((s, _)) => {
            worksheet
                .write_string_with_format(row, col, s, &format)
                .map_err(xlsx_err)?;
        }
        CellValue::Float(value) => {
            worksheet
                .write_number_with_format(row, col, *value as f64, &format)
                .map_err(xlsx_err)?;
        }
        CellValue::Integer(value) => {
            worksheet
                .write_number_with_format(row, col, *value as f64, &format)
                .map_err(xlsx_err)?;
        }
    }
    Ok(())
}

/// `Column`'s human-readable label, matching `get_available_columns()`'s
/// own `display_name` for that column — falls back to the raw db key
/// (`Column::as_key`) for a column this database's `available_columns`
/// doesn't list - a coloc combination (`Column::Coloc`) or stale export
/// settings - its own label with the class names.
fn column_display_name(
    column: &Column,
    available_columns: &[ColumnEntry],
    classes: &[evanalyzer_cfg::settings::classification_settings::Class],
) -> String {
    available_columns
        .iter()
        .find(|entry| entry.key == *column)
        .map(|entry| entry.display_name.clone())
        .unwrap_or_else(|| column.display_label(classes))
}

fn aggregation_label(aggregation: &Aggregation) -> &'static str {
    match aggregation {
        Aggregation::Avg => "Average",
        Aggregation::Min => "Minimum",
        Aggregation::Max => "Maximum",
        Aggregation::Stddev => "Std Dev",
        Aggregation::Sum => "Sum",
        Aggregation::Median => "Median",
        Aggregation::Skewness => "Skewness",
    }
}

fn sanitize_filename_component(name: &str) -> String {
    name.chars()
        .map(|c| {
            if c.is_alphanumeric() || c == '-' || c == '_' {
                c
            } else {
                '_'
            }
        })
        .collect()
}

/// Extracts a `View::List` value cell's number — always `CellValue::Float`
/// in practice (see `well_fields_to_result`/`get_group_by_plate`'s own
/// `View::List` arms), but matching `Integer` too costs nothing and keeps
/// this from silently going quiet if that ever changes.
fn cell_to_f64(cell: &Cell) -> Option<f64> {
    match &cell.value {
        CellValue::Float(value) => Some(*value as f64),
        CellValue::Integer(value) => Some(*value as f64),
        _ => None,
    }
}

fn plain_cell(value: CellValue) -> Cell {
    Cell {
        value,
        bg_color: 0,
        alternating_color: false,
        search_key: None,
        disabled: false,
        any_disabled: false,
        failed: false,
        any_failed: false,
    }
}

const EXPORT_FONT_SIZE: f64 = 10.0;

/// Light gray Excel gives every other coloc-detail row (`Cell::alternating_color`)
/// so the fanned-out rows belonging to one source object stay visually
/// grouped — matches `Theme.list-row-alt-bg`'s role in the GUI's own List
/// view, just as a plain hex constant here since XLSX formatting has no
/// theme to pull from.
const ALTERNATING_ROW_BG: u32 = 0xF1F1F1;

/// The planes an export covers: the ones with objects (`get_planes`) within
/// the chosen time points and z-planes (`None`: all), time point by time
/// point, then z-plane. Whether the results have several time points
/// (`by_t`) or z-planes (`by_z`) decides the extra "T"/"Z" columns of the
/// lists and the plane in the grids' tab names - results of one plane look
/// as they always did.
struct ExportPlanes {
    planes: Vec<PlaneFilter>,
    by_t: bool,
    by_z: bool,
}

impl ExportPlanes {
    fn of(database: &dyn ResultsSource, export: &PlateExport) -> Result<Self, InternalErrors> {
        let all = database.get_planes()?;
        let within = |value: u32, range: &Option<std::range::Range<u32>>| {
            range.is_none_or(|range| range.start <= value && value < range.end)
        };
        let mut planes: Vec<PlaneFilter> = all
            .iter()
            .filter(|p| within(p.t_stack, &export.t_stacks) && within(p.z_stack, &export.z_stacks))
            .copied()
            .collect();
        if planes.is_empty() {
            // No objects there: still the first chosen plane - the
            // per-image and plate exports list their images then, with
            // nothing found.
            planes.push(PlaneFilter {
                t_stack: export.t_stacks.map_or(0, |range| range.start),
                z_stack: export.z_stacks.map_or(0, |range| range.start),
            });
        }
        Ok(Self {
            by_t: all.iter().any(|p| p.t_stack != all[0].t_stack),
            by_z: all.iter().any(|p| p.z_stack != all[0].z_stack),
            planes,
        })
    }

    /// The lists' extra columns: "T", "Z" - each only when the results have
    /// several.
    fn header(&self) -> Vec<String> {
        let mut header = Vec::new();
        if self.by_t {
            header.push("T".to_string());
        }
        if self.by_z {
            header.push("Z".to_string());
        }
        header
    }

    /// `plane` in the lists' extra columns (see [`Self::header`]).
    fn cells(&self, plane: PlaneFilter) -> Vec<Cell> {
        let mut cells = Vec::new();
        if self.by_t {
            cells.push(plain_cell(CellValue::Integer(plane.t_stack as i32)));
        }
        if self.by_z {
            cells.push(plain_cell(CellValue::Integer(plane.z_stack as i32)));
        }
        cells
    }

    /// `plane` in the flat lists' extra key columns (see [`Self::header`]).
    fn texts(&self, plane: PlaneFilter) -> Vec<String> {
        self.cells(plane).iter().map(cell_text).collect()
    }

    /// `plane` before a grid's tab name ("T1 Nuclei", "T1 Z2 Nuclei") - first,
    /// so a long class name is what Excel's 31 characters cut, not the plane.
    fn tab_name(&self, plane: PlaneFilter, name: &str) -> String {
        let mut parts = Vec::new();
        if self.by_t {
            parts.push(format!("T{}", plane.t_stack));
        }
        if self.by_z {
            parts.push(format!("Z{}", plane.z_stack));
        }
        parts.push(name.to_string());
        parts.join(" ")
    }
}

/// A usable base filename for `image_rel_path`'s heatmap document — its
/// file stem (no directory, no extension), since the rel path's own
/// separators/extension aren't valid or wanted in a sibling file's name.
fn image_stub(image_rel_path: &str) -> String {
    Path::new(image_rel_path)
        .file_stem()
        .map(|stem| stem.to_string_lossy().into_owned())
        .unwrap_or_else(|| image_rel_path.to_string())
}

/// Check for cancle the export process
fn check_cancelled(cancel: &AtomicBool) -> Result<(), InternalErrors> {
    if cancel.load(Ordering::Relaxed) {
        Err(InternalErrors::Cancelled)
    } else {
        Ok(())
    }
}

/// Plain-text form of a `Cell`'s value for CSV, dropping its color/formatting
/// (CSV has no cells to color) — matches the CLI's identically-named helper
/// in `crates/cli/src/commands/common.rs`, kept separate there since it's
/// also used by that crate's table/JSON rendering, unrelated to export.
fn cell_text(cell: &Cell) -> String {
    match &cell.value {
        CellValue::Empty => String::new(),
        CellValue::String(s) => s.clone(),
        CellValue::Class((s, _)) => s.clone(),
        CellValue::Float(v) => v.to_string(),
        CellValue::Integer(v) => v.to_string(),
    }
}

/// Hands out sanitized, unique-within-one-workbook Excel sheet names.
/// Excel rejects a sheet name that's empty, over 31 characters, contains
/// `[ ] : * ? / \`, or duplicates another sheet in the same workbook —
/// `unique` fixes up all four so a class's own (arbitrary, user-entered)
/// name is always safe to use directly as a tab name.
struct SheetNamer {
    used: HashSet<String>,
}

impl SheetNamer {
    fn new() -> Self {
        Self {
            used: HashSet::new(),
        }
    }

    fn unique(&mut self, wanted: &str) -> String {
        let mut sanitized: String = wanted
            .chars()
            .map(|c| if "[]:*?/\\".contains(c) { '_' } else { c })
            .collect();
        sanitized = sanitized.trim().to_string();
        if sanitized.is_empty() {
            sanitized = "Sheet".to_string();
        }
        if sanitized.chars().count() > 31 {
            sanitized = sanitized.chars().take(31).collect();
        }

        let mut candidate = sanitized.clone();
        let mut suffix_n = 2;
        while self.used.contains(&candidate) {
            let suffix = format!(" ({suffix_n})");
            let max_base = 31usize.saturating_sub(suffix.chars().count());
            let base: String = sanitized.chars().take(max_base).collect();
            candidate = format!("{base}{suffix}");
            suffix_n += 1;
        }
        self.used.insert(candidate.clone());
        candidate
    }
}

const LIST_VALUE_COL_PX: u32 = 90;
