use super::results_generator::class_display_label;
use crate::result::{
    Aggregation, Cell, CellValue, ColorScale, ColorSchema, Column, ColumnEntry, DatabaseResult,
    GroupedByImageFilter, ImageHeatmapFilter, ListFilter, Pagination, PlaneFilter, PlateDimensions,
    PlateFilter, ResultsGenerator, View, WellSize, WellsBatchFilter,
};
use evanalyzer_cfg::core_types::{InternalErrors, ObjectClass};
use rust_xlsxwriter::{Color, Format, Workbook, Worksheet, XlsxError};
use std::collections::HashMap;
use std::collections::HashSet;
use std::path::Path;
use std::path::PathBuf;
use std::range::Range;

/// Light gray Excel gives every other coloc-detail row (`Cell::alternating_color`)
/// so the fanned-out rows belonging to one source object stay visually
/// grouped — matches `Theme.list-row-alt-bg`'s role in the GUI's own List
/// view, just as a plain hex constant here since XLSX formatting has no
/// theme to pull from.
const ALTERNATING_ROW_BG: u32 = 0xF1F1F1;

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub enum ExportFormat {
    #[default]
    XLSX,
    CSV,
    Parquet,
}

/// One export step/document completing — `current`/`total` describe
/// progress *within* the step named by `message` (e.g. "Plate/Well: ch2@spot"
/// at `current=3, total=22` for the 3rd of 22 classes), not overall export
/// progress across List/Plate/Well/Heatmap combined, since those phases
/// have no shared unit to make a single running percentage meaningful.
pub type ExportProgress<'a> = &'a mut dyn FnMut(&str, usize, usize);

#[derive(Default, Clone)]
pub struct ResultExport {
    /// Directory the export writes its file(s) into — created if missing.
    /// Every document below lives directly under it (`list.xlsx`,
    /// `plate.xlsx`, `well.xlsx`, `heatmap_{image}.xlsx`).
    pub output_dir: PathBuf,
    pub format: ExportFormat,
    pub t_stacks: Range<u32>,
    pub z_stacks: Range<u32>,
    /// `[]` means every image in the database (see `resolve_images`).
    pub image_rel_paths: Vec<String>,

    // Variants
    pub columns: Vec<Column>,
    pub color_schema: ColorSchema,
    pub color_scale: ColorScale,
    pub grouping_regex: String,
    /// `[]` means every object class registered in the database (see
    /// `export_plate_and_well`/`export_heatmap`).
    pub object_classes: Vec<ObjectClass>,

    // Matrix view options
    pub aggregations: Vec<Aggregation>,
    pub plate_dimension: Option<PlateDimensions>,
    pub well_size: Option<WellSize>,
    pub well_order: Option<Vec<u32>>,
    pub square_size: Option<usize>,

    // What to export
    pub with_list_view: bool,
    pub with_list_coloc_details: bool,
    /// `list_{image}.xlsx` per image instead of one shared `list.xlsx` —
    /// needed once a single image's own object count risks Excel's
    /// 1,048,576-row-per-sheet limit (a large plate scan can comfortably
    /// exceed that combined across images, even if no single image does).
    pub with_list_one_file_per_image: bool,
    /// `grouped_by_image.xlsx`: one row per image, one column per
    /// (selected column × selected aggregation) combination — see
    /// `export_grouped_by_image`. Independent of `with_list_view`: it's a
    /// separate document, not a variant of the per-object list.
    pub with_grouped_by_image_list: bool,
    pub with_plate_view: bool,
    pub with_plates_and_wells_as_list: bool,
    pub with_heatmap: bool,
}

impl ResultExport {
    /// `on_progress(message, current, total)` is called throughout to
    /// report what's happening — see `ExportProgress`'s doc comment for what
    /// `current`/`total` are relative to. Called from a background thread
    /// (this can take a while for a large database), so `on_progress`
    /// itself must not touch UI state directly — the caller's closure
    /// should just forward each call through `slint::invoke_from_event_loop`
    /// or equivalent.
    pub fn start_export(
        &self,
        database: &ResultsGenerator,
        on_progress: ExportProgress,
    ) -> Result<(), InternalErrors> {
        if matches!(self.format, ExportFormat::CSV)
            && (self.with_plate_view || self.with_plates_and_wells_as_list || self.with_heatmap)
        {
            return Err(InternalErrors::Internal(
                "CSV export only supports the List and Grouped-by-Image views — use XLSX for Plate/Well/Heatmap".to_string(),
            ));
        }

        std::fs::create_dir_all(&self.output_dir).map_err(|e| {
            InternalErrors::Internal(format!(
                "Could not create export directory {:?}: {e}",
                self.output_dir
            ))
        })?;

        // Parquet ignores every `with_*`/column/filter setting below - it's
        // always a single raw dump of the whole `objects` table (see
        // `export_as_parquet`'s doc comment), so it's handled as its own
        // early return rather than threaded through the List/Grouped/Plate/
        // Well/Heatmap dispatch those settings drive.
        if matches!(self.format, ExportFormat::Parquet) {
            return self.export_as_parquet(database, &mut *on_progress);
        }

        if self.with_list_view {
            self.export_list(database, &mut *on_progress)?;
        }
        if self.with_grouped_by_image_list {
            self.export_grouped_by_image(database, &mut *on_progress)?;
        }
        // Single flag drives both documents — see the doc comment on
        // `export_plate_and_well` for why plate and well are always
        // exported together rather than needing their own toggle each.
        if self.with_plate_view {
            self.export_plate_and_well(database, &mut *on_progress)?;
        }
        if self.with_plates_and_wells_as_list {
            self.export_plate_and_well_as_flat_list(database, &mut *on_progress)?;
        }
        if self.with_heatmap {
            self.export_heatmap(database, &mut *on_progress)?;
        }
        Ok(())
    }

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
    fn export_list(
        &self,
        database: &ResultsGenerator,
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
                on_progress(
                    &format!("Exporting List: {image}"),
                    image_idx + 1,
                    images.len(),
                );
                let single_image = std::slice::from_ref(image);
                let stem = sanitize_filename_component(&image_stub(image));
                self.write_list_document(
                    database,
                    single_image,
                    &object_classes,
                    &format!("list_{stem}"),
                )?;
            }
            return Ok(());
        }

        on_progress("Exporting List view", 0, 1);
        self.write_list_document(database, &images, &object_classes, "list")?;
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
    fn write_list_document(
        &self,
        database: &ResultsGenerator,
        images: &[String],
        object_classes: &Option<Vec<ObjectClass>>,
        file_stem: &str,
    ) -> Result<(), InternalErrors> {
        match self.format {
            ExportFormat::XLSX => {
                let mut workbook = Workbook::new();
                let sheet = workbook.add_worksheet();
                sheet.set_name("List").map_err(xlsx_err)?;
                write_list_sheet(sheet, database, self, images, object_classes, false)?;

                if self.with_list_coloc_details {
                    let coloc_sheet = workbook.add_worksheet();
                    coloc_sheet
                        .set_name("List (Coloc Details)")
                        .map_err(xlsx_err)?;
                    write_list_sheet(coloc_sheet, database, self, images, object_classes, true)?;
                }

                workbook
                    .save(self.output_dir.join(format!("{file_stem}.xlsx")))
                    .map_err(xlsx_err)?;
            }
            ExportFormat::CSV => {
                write_list_csv(
                    database,
                    self,
                    images,
                    object_classes,
                    false,
                    &self.output_dir.join(format!("{file_stem}.csv")),
                )?;
                if self.with_list_coloc_details {
                    write_list_csv(
                        database,
                        self,
                        images,
                        object_classes,
                        true,
                        &self
                            .output_dir
                            .join(format!("{file_stem}_coloc_details.csv")),
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

    // `grouped_by_image.xlsx`: one row per image, one column per (selected
    // column × selected aggregation) combination — `get_grouped_by_image`'s
    // own shape, just paginated through in full rather than exposed live.
    // Single plane only (like `export_plate_and_well`/`export_heatmap`):
    // grouping by image is itself a per-plane aggregate, so there's no
    // meaningful "stack every t/z" equivalent the way there is for the
    // per-object List export.
    fn export_grouped_by_image(
        &self,
        database: &ResultsGenerator,
        on_progress: ExportProgress,
    ) -> Result<(), InternalErrors> {
        on_progress("Exporting Grouped Image List", 0, 1);

        let images = resolve_images(database, self)?;
        let object_classes = if self.object_classes.is_empty() {
            None
        } else {
            Some(self.object_classes.clone())
        };
        let columns: Vec<Column> = self
            .columns
            .iter()
            .filter(|column| is_aggregable(column))
            .cloned()
            .collect();

        let base_filter = GroupedByImageFilter {
            plane: PlaneFilter {
                z_stack: self.z_stacks.start,
                t_stack: self.t_stacks.start,
            },
            images: Some(images),
            object_classes,
            columns,
            aggregation: self.aggregations.clone(),
            page: Pagination {
                limit: 0,
                after: None,
            },
        };
        let result = fetch_all_grouped_by_image_rows(database, &base_filter)?;

        match self.format {
            ExportFormat::XLSX => {
                let mut workbook = Workbook::new();
                let sheet = workbook.add_worksheet();
                sheet.set_name("Grouped by Image").map_err(xlsx_err)?;
                let header_format = Format::new().set_bold();
                for (col_idx, name) in result.column_names.iter().enumerate() {
                    sheet
                        .write_with_format(0, col_idx as u16, name.as_str(), &header_format)
                        .map_err(xlsx_err)?;
                }
                for (row_idx, row) in result.rows.iter().enumerate() {
                    let row_n = (row_idx + 1) as u32;
                    for (col_idx, cell) in row.iter().enumerate() {
                        write_cell(sheet, row_n, col_idx as u16, cell)?;
                    }
                }

                workbook
                    .save(self.output_dir.join("grouped_by_image.xlsx"))
                    .map_err(xlsx_err)?;
            }
            ExportFormat::CSV => {
                write_csv(&result, &self.output_dir.join("grouped_by_image.csv"))?;
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

    // `plate.xlsx` + `well.xlsx`: one tab per object class in each
    // document, every tab stacking one square, colored grid block per
    // (column, aggregation) combination (and, one level further down in
    // `well.xlsx`, per well on top of that) — see `write_grid_block`. Both
    // documents come from the same flag: a plate view without its wells (or
    // vice versa) isn't a meaningful export on its own, so there's no
    // separate toggle for each. The flat-list form of the same data (see
    // `export_plate_and_well_as_flat_list`) is a separate pass into
    // separate files, not appended here — this function only ever needs
    // `View::Heatmap` data.
    //
    // Only the first z/t in `self.z_stacks`/`self.t_stacks` is rendered:
    // unlike the List export, a plate/well grid is inherently a single
    // plane's snapshot (matching the GUI's own Matrix view, which shows one
    // z/t at a time via the global stepper, not a time series) — there's no
    // "stack every t one after another" equivalent for a grid the way
    // there is for a flat table.
    fn export_plate_and_well(
        &self,
        database: &ResultsGenerator,
        on_progress: ExportProgress,
    ) -> Result<(), InternalErrors> {
        let classes_all = database.get_object_classes()?;
        let target_classes: Vec<ObjectClass> = if self.object_classes.is_empty() {
            classes_all.iter().map(|class| class.id).collect()
        } else {
            self.object_classes.clone()
        };
        let available_columns = database.get_available_columns()?;
        let z = self.z_stacks.start;
        let t = self.t_stacks.start;

        let mut plate_workbook = Workbook::new();
        let mut plate_names = SheetNamer::new();
        let mut well_workbook = Workbook::new();
        let mut well_names = SheetNamer::new();

        for (class_idx, class) in target_classes.iter().enumerate() {
            let class_label = class_display_label(*class, &classes_all);
            on_progress(
                &format!("Exporting Plate/Well: {class_label}"),
                class_idx + 1,
                target_classes.len(),
            );

            let plate_sheet = plate_workbook.add_worksheet();
            plate_sheet
                .set_name(plate_names.unique(&class_label))
                .map_err(xlsx_err)?;
            let well_sheet = well_workbook.add_worksheet();
            well_sheet
                .set_name(well_names.unique(&class_label))
                .map_err(xlsx_err)?;

            let mut plate_row = 0u32;
            let mut well_row = 0u32;

            for column in self.columns.iter().filter(|column| is_aggregable(column)) {
                let column_label = column_display_name(column, &available_columns);

                // One query covering *every* requested aggregation, rather
                // than one query per aggregation — the WHERE/GROUP BY here
                // is identical across all of them, only the aggregate
                // function itself differs, so this is the difference
                // between (with all 7 aggregations selected) 7 full scans
                // and 1. Combined with `get_wells_for_plate_multi_agg`'s own
                // per-well batching below, a full "every class, every
                // column, every aggregation" export goes from
                // classes × columns × aggregations × wells scans down to
                // classes × columns.
                let plate_filter = PlateFilter {
                    plane: PlaneFilter {
                        z_stack: z,
                        t_stack: t,
                    },
                    grouping_regex: self.grouping_regex.clone(),
                    // Ignored by `get_group_by_plate_multi_agg` (it takes
                    // `self.aggregations` separately) - `PlateFilter` still
                    // needs some value structurally.
                    aggregation: Aggregation::default(),
                    object_class: *class,
                    column: column.clone(),
                    color_schema: self.color_schema.clone(),
                    color_scale: self.color_scale.clone(),
                    matrix_dimension: self.plate_dimension,
                };
                let plate_grids = database.get_group_by_plate_multi_agg(
                    &plate_filter,
                    &self.aggregations,
                    &View::Heatmap,
                )?;

                let wells_filter = WellsBatchFilter {
                    plane: PlaneFilter {
                        z_stack: z,
                        t_stack: t,
                    },
                    grouping_regex: self.grouping_regex.clone(),
                    aggregation: Aggregation::default(),
                    object_class: *class,
                    column: column.clone(),
                    color_schema: self.color_schema.clone(),
                    color_scale: self.color_scale.clone(),
                    well_size: self.well_size,
                    well_order: self.well_order.clone(),
                };
                let well_heatmaps_per_agg = database.get_wells_for_plate_multi_agg(
                    &wells_filter,
                    &self.aggregations,
                    &View::Heatmap,
                )?;

                for ((aggregation, plate_grid), mut well_heatmaps) in self
                    .aggregations
                    .iter()
                    .zip(plate_grids)
                    .zip(well_heatmaps_per_agg)
                {
                    let caption = format!("{column_label} — {}", aggregation_label(aggregation));
                    plate_row = write_grid_block(plate_sheet, plate_row, &caption, &plate_grid)?;

                    // Sorted for deterministic, well-id-ordered output -
                    // `well_heatmaps` is a `HashMap`, so its own iteration
                    // order isn't meaningful on its own.
                    let mut well_ids: Vec<String> = well_heatmaps.keys().cloned().collect();
                    well_ids.sort();

                    for well_id in &well_ids {
                        let well_caption = format!("{caption} — Well {well_id}");
                        let Some(well_grid) = well_heatmaps.remove(well_id) else {
                            continue;
                        };
                        well_row =
                            write_grid_block(well_sheet, well_row, &well_caption, &well_grid)?;
                    }
                }
            }
        }

        plate_workbook
            .save(self.output_dir.join("plate.xlsx"))
            .map_err(xlsx_err)?;
        well_workbook
            .save(self.output_dir.join("well.xlsx"))
            .map_err(xlsx_err)?;
        Ok(())
    }

    // `plate_list.xlsx` + `well_list.xlsx`: the same aggregated data as
    // `export_plate_and_well`, but pivoted into one plain table per document
    // instead of a grid-per-class-tab — one row per well (or, in
    // `well_list.xlsx`, per well+field), one column per (class, column,
    // aggregation) combination, so every class sits side by side in the
    // same tab rather than needing its own. `well_list.xlsx` also carries
    // an "Image" column (the source image for that field), independent of
    // which class/column/aggregation combination is being looked at.
    //
    // Only ever needs `View::List` data - never overlaps in query cost with
    // `export_plate_and_well`'s `View::Heatmap`-only fetches above, even
    // though both run in the same export when this is enabled.
    fn export_plate_and_well_as_flat_list(
        &self,
        database: &ResultsGenerator,
        on_progress: ExportProgress,
    ) -> Result<(), InternalErrors> {
        let classes_all = database.get_object_classes()?;
        let target_classes: Vec<ObjectClass> = if self.object_classes.is_empty() {
            classes_all.iter().map(|class| class.id).collect()
        } else {
            self.object_classes.clone()
        };
        let available_columns = database.get_available_columns()?;
        let z = self.z_stacks.start;
        let t = self.t_stacks.start;
        let aggregable_columns: Vec<&Column> = self
            .columns
            .iter()
            .filter(|column| is_aggregable(column))
            .collect();
        let combo_count = target_classes.len() * aggregable_columns.len() * self.aggregations.len();

        let mut combo_labels: Vec<String> = Vec::with_capacity(combo_count);
        let mut plate_values: HashMap<String, Vec<Option<f64>>> = HashMap::new();
        let mut well_values: HashMap<(String, String), Vec<Option<f64>>> = HashMap::new();
        let mut well_images: HashMap<(String, String), String> = HashMap::new();
        let mut combo_idx = 0usize;

        for (class_idx, class) in target_classes.iter().enumerate() {
            let class_label = class_display_label(*class, &classes_all);
            on_progress(
                &format!("Exporting Flat List: {class_label}"),
                class_idx + 1,
                target_classes.len(),
            );

            for column in &aggregable_columns {
                let column_label = column_display_name(column, &available_columns);

                // Same one-query-for-every-aggregation batching as
                // `export_plate_and_well` - see its comment.
                let plate_filter = PlateFilter {
                    plane: PlaneFilter {
                        z_stack: z,
                        t_stack: t,
                    },
                    grouping_regex: self.grouping_regex.clone(),
                    aggregation: Aggregation::default(),
                    object_class: *class,
                    column: (*column).clone(),
                    color_schema: self.color_schema.clone(),
                    color_scale: self.color_scale.clone(),
                    matrix_dimension: self.plate_dimension,
                };
                let plate_lists = database.get_group_by_plate_multi_agg(
                    &plate_filter,
                    &self.aggregations,
                    &View::List,
                )?;

                let wells_filter = WellsBatchFilter {
                    plane: PlaneFilter {
                        z_stack: z,
                        t_stack: t,
                    },
                    grouping_regex: self.grouping_regex.clone(),
                    aggregation: Aggregation::default(),
                    object_class: *class,
                    column: (*column).clone(),
                    color_schema: self.color_schema.clone(),
                    color_scale: self.color_scale.clone(),
                    well_size: self.well_size,
                    well_order: self.well_order.clone(),
                };
                let well_lists_per_agg = database.get_wells_for_plate_multi_agg(
                    &wells_filter,
                    &self.aggregations,
                    &View::List,
                )?;

                for ((aggregation, plate_list), well_lists) in self
                    .aggregations
                    .iter()
                    .zip(plate_lists)
                    .zip(well_lists_per_agg)
                {
                    combo_labels.push(format!(
                        "{class_label} — {column_label} — {}",
                        aggregation_label(aggregation)
                    ));

                    for (well_id, row) in plate_list.row_names.iter().zip(&plate_list.rows) {
                        if let Some(value) = row.get(1).and_then(cell_to_f64) {
                            plate_values
                                .entry(well_id.clone())
                                .or_insert_with(|| vec![None; combo_count])[combo_idx] =
                                Some(value);
                        }
                    }

                    for (well_id, result) in &well_lists {
                        for (field_idx, row) in result.row_names.iter().zip(&result.rows) {
                            let key = (well_id.clone(), field_idx.clone());
                            if let Some(value) = row.get(1).and_then(cell_to_f64) {
                                well_values
                                    .entry(key.clone())
                                    .or_insert_with(|| vec![None; combo_count])[combo_idx] =
                                    Some(value);
                            }
                            if let Some((image_name, _)) =
                                row.get(1).and_then(|cell| cell.search_key.as_ref())
                            {
                                well_images.entry(key).or_insert_with(|| image_name.clone());
                            }
                        }
                    }

                    combo_idx += 1;
                }
            }
        }

        let mut plate_rows: Vec<(String, Vec<Option<f64>>)> = plate_values.into_iter().collect();
        plate_rows.sort_by(|(a, _), (b, _)| a.cmp(b));
        write_flat_pivot(
            &self.output_dir.join("plate_list.xlsx"),
            "Plate",
            &["Well"],
            &combo_labels,
            plate_rows
                .into_iter()
                .map(|(well_id, values)| (vec![well_id], values)),
        )?;

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
        write_flat_pivot(
            &self.output_dir.join("well_list.xlsx"),
            "Well",
            &["Well", "Field", "Image"],
            &combo_labels,
            well_rows.into_iter().map(|((well_id, field_idx), values)| {
                let image = well_images
                    .get(&(well_id.clone(), field_idx.clone()))
                    .cloned()
                    .unwrap_or_default();
                (vec![well_id, field_idx, image], values)
            }),
        )?;

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
        database: &ResultsGenerator,
        on_progress: ExportProgress,
    ) -> Result<(), InternalErrors> {
        let classes_all = database.get_object_classes()?;
        let target_classes: Vec<ObjectClass> = if self.object_classes.is_empty() {
            classes_all.iter().map(|class| class.id).collect()
        } else {
            self.object_classes.clone()
        };
        let available_columns = database.get_available_columns()?;
        let images = resolve_images(database, self)?;
        let z = self.z_stacks.start;
        let t = self.t_stacks.start;

        for (image_idx, image) in images.iter().enumerate() {
            on_progress(
                &format!("Exporting Heatmap: {image}"),
                image_idx + 1,
                images.len(),
            );
            let mut workbook = Workbook::new();
            let mut names = SheetNamer::new();

            for class in &target_classes {
                let class_label = class_display_label(*class, &classes_all);
                let sheet = workbook.add_worksheet();
                sheet
                    .set_name(names.unique(&class_label))
                    .map_err(xlsx_err)?;

                let mut row = 0u32;
                for column in self.columns.iter().filter(|column| is_aggregable(column)) {
                    let column_label = column_display_name(column, &available_columns);
                    for aggregation in &self.aggregations {
                        let caption =
                            format!("{column_label} — {}", aggregation_label(aggregation));
                        let filter = ImageHeatmapFilter {
                            plane: PlaneFilter {
                                z_stack: z,
                                t_stack: t,
                            },
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

            let file_name = format!(
                "heatmap_{}.xlsx",
                sanitize_filename_component(&image_stub(image))
            );
            workbook
                .save(self.output_dir.join(file_name))
                .map_err(xlsx_err)?;
        }
        Ok(())
    }

    /// `objects.parquet`: the entire `objects` table, every column, no z/t/
    /// image/class filtering and no column selection — DuckDB's own `COPY`
    /// writes the file directly, so this bypasses `get_object_list`/
    /// `fetch_all_list_rows` entirely (unlike every other export above).
    /// Meant for a downstream tool that reads Parquet natively rather than
    /// for a human to open, so there's no equivalent of `with_list_view`'s
    /// column/plane scoping to apply here.
    fn export_as_parquet(
        &self,
        database: &ResultsGenerator,
        on_progress: ExportProgress,
    ) -> Result<(), InternalErrors> {
        on_progress("Exporting Parquet", 0, 1);
        let path = self.output_dir.join("objects.parquet");
        // DuckDB's `COPY` takes the destination as a single-quoted string
        // literal inside the SQL text itself (not a bindable parameter), so
        // any literal `'` in the path has to be escaped the same way a SQL
        // string literal would be (doubling it) rather than passed through
        // `params![]`.
        let path_literal = path.to_string_lossy().replace('\'', "''");
        let sql = format!("COPY objects TO '{path_literal}' (FORMAT parquet);");
        database
            .connection()
            .execute_batch(&sql)
            .map_err(|e| InternalErrors::Io(e.to_string()))?;
        on_progress("Exporting Parquet", 1, 1);
        Ok(())
    }
}

fn xlsx_err(error: XlsxError) -> InternalErrors {
    InternalErrors::Internal(format!("XLSX export error: {error}"))
}

/// The images an export should cover, in the same image-name-sorted order
/// `get_images()` returns them in — regardless of what order
/// `ResultExport::image_rel_paths` happens to list them in, so "sorted by
/// image name" (see `export_list`'s doc comment) holds even when the caller
/// supplies an explicit subset. `[]` means every image in the database,
/// except ones the user has disabled — an explicit list is taken as
/// overriding that (the caller asked for exactly these, disabled or not).
fn resolve_images(
    database: &ResultsGenerator,
    export: &ResultExport,
) -> Result<Vec<String>, InternalErrors> {
    let all = database.get_images()?;
    if export.image_rel_paths.is_empty() {
        Ok(all
            .into_iter()
            .filter(|image| !image.disabled)
            .map(|image| image.rel_path.to_string_lossy().into_owned())
            .collect())
    } else {
        let wanted: HashSet<&str> = export.image_rel_paths.iter().map(String::as_str).collect();
        Ok(all
            .into_iter()
            .map(|image| image.rel_path.to_string_lossy().into_owned())
            .filter(|rel_path| wanted.contains(rel_path.as_str()))
            .collect())
    }
}

// `get_list` only ever hands back one bounded page (`ListFilter.page`) —
// same keyset-pagination shape `results_state_controller.rs` pages through
// for the GUI's List view (see its own `update_list_view`) — so an export,
// which needs every matching row rather than one page of them, walks every
// page via the same cursor-from-last-row-id trick and concatenates them.
// Compared against `source_object_count` (not `rows.len()`), matching that
// same reasoning: `ListFilter::with_coloc_details` fan-out can multiply
// `rows.len()` past `PAGE_SIZE` on what's still the final page.
fn fetch_all_list_rows(
    database: &ResultsGenerator,
    base_filter: &ListFilter,
) -> Result<DatabaseResult, InternalErrors> {
    const PAGE_SIZE: i32 = 20_000;

    let mut merged = DatabaseResult {
        column_names: Vec::new(),
        row_names: Vec::new(),
        rows: Vec::new(),
        min: 0.0,
        max: 0.0,
        source_object_count: 0,
        row_locations: Vec::new(),
    };
    let mut cursor: Option<String> = None;
    let mut first_page = true;

    loop {
        let filter = ListFilter {
            page: Pagination {
                limit: PAGE_SIZE,
                after: cursor.take(),
            },
            ..base_filter.clone()
        };
        let mut page = database.get_object_list(&filter)?;
        let is_last_page = page.source_object_count < PAGE_SIZE as usize;
        cursor = page.row_names.last().cloned();

        if first_page {
            merged.column_names = std::mem::take(&mut page.column_names);
            first_page = false;
        }
        merged.row_names.extend(page.row_names);
        merged.rows.extend(page.rows);
        merged.row_locations.extend(page.row_locations);
        merged.source_object_count += page.source_object_count;

        if is_last_page {
            break;
        }
    }

    Ok(merged)
}

// Same keyset-pagination walk as `fetch_all_list_rows`, just over
// `get_grouped_by_image`'s (image-grouped, not per-object) pages instead —
// its cursor is the last page's final `image_rel_path`, exactly like
// `get_object_list`'s own row-id cursor.
fn fetch_all_grouped_by_image_rows(
    database: &ResultsGenerator,
    base_filter: &GroupedByImageFilter,
) -> Result<DatabaseResult, InternalErrors> {
    const PAGE_SIZE: i32 = 20_000;

    let mut merged = DatabaseResult {
        column_names: Vec::new(),
        row_names: Vec::new(),
        rows: Vec::new(),
        min: f32::INFINITY,
        max: f32::NEG_INFINITY,
        source_object_count: 0,
        row_locations: Vec::new(),
    };
    let mut cursor: Option<String> = None;
    let mut first_page = true;

    loop {
        let filter = GroupedByImageFilter {
            page: Pagination {
                limit: PAGE_SIZE,
                after: cursor.take(),
            },
            ..base_filter.clone()
        };
        let mut page = database.get_grouped_by_image(&filter)?;
        let is_last_page = page.source_object_count < PAGE_SIZE as usize;
        cursor = page.row_names.last().cloned();

        if first_page {
            merged.column_names = std::mem::take(&mut page.column_names);
            first_page = false;
        }
        merged.min = merged.min.min(page.min);
        merged.max = merged.max.max(page.max);
        merged.row_names.extend(page.row_names);
        merged.rows.extend(page.rows);
        merged.source_object_count += page.source_object_count;

        if is_last_page {
            break;
        }
    }

    if !merged.min.is_finite() || !merged.max.is_finite() {
        merged.min = 0.0;
        merged.max = 0.0;
    }

    Ok(merged)
}

// Writes every object matching `export`'s filters, in
// z/t/image-name-sorted order (see `ResultExport::export_list`), as one
// continuous table starting at the worksheet's top row.
fn write_list_sheet(
    worksheet: &mut Worksheet,
    database: &ResultsGenerator,
    export: &ResultExport,
    images: &[String],
    object_classes: &Option<Vec<ObjectClass>>,
    with_coloc_details: bool,
) -> Result<(), InternalErrors> {
    let header_format = Format::new().set_bold();
    let mut next_row: u32 = 0;
    let mut header_written = false;

    for z in export.z_stacks {
        for t in export.t_stacks {
            for image in images {
                let base_filter = ListFilter {
                    plane: PlaneFilter {
                        z_stack: z,
                        t_stack: t,
                    },
                    images: Some(vec![image.clone()]),
                    object_classes: object_classes.clone(),
                    columns: export.columns.clone(),
                    with_coloc_details,
                    page: Pagination {
                        limit: 0,
                        after: None,
                    },
                };
                let result = fetch_all_list_rows(database, &base_filter)?;

                if !header_written {
                    for (col_idx, name) in result.column_names.iter().enumerate() {
                        worksheet
                            .write_with_format(
                                next_row,
                                col_idx as u16,
                                name.as_str(),
                                &header_format,
                            )
                            .map_err(xlsx_err)?;
                    }
                    next_row += 1;
                    header_written = true;
                }

                for row in &result.rows {
                    for (col_idx, cell) in row.iter().enumerate() {
                        write_cell(worksheet, next_row, col_idx as u16, cell)?;
                    }
                    next_row += 1;
                }
            }
        }
    }

    Ok(())
}

/// CSV sibling of `write_list_sheet`: same z/t/image sweep and per-plane
/// fetch (so the two formats can never disagree on row order or content),
/// written straight to `path` a plane at a time instead of into an XLSX
/// worksheet.
fn write_list_csv(
    database: &ResultsGenerator,
    export: &ResultExport,
    images: &[String],
    object_classes: &Option<Vec<ObjectClass>>,
    with_coloc_details: bool,
    path: &Path,
) -> Result<(), InternalErrors> {
    let mut out = create_csv_writer(path)?;
    let write_err = |e: std::io::Error| {
        InternalErrors::Internal(format!("could not write {}: {e}", path.display()))
    };
    let mut header_written = false;

    for z in export.z_stacks {
        for t in export.t_stacks {
            for image in images {
                let base_filter = ListFilter {
                    plane: PlaneFilter {
                        z_stack: z,
                        t_stack: t,
                    },
                    images: Some(vec![image.clone()]),
                    object_classes: object_classes.clone(),
                    columns: export.columns.clone(),
                    with_coloc_details,
                    page: Pagination {
                        limit: 0,
                        after: None,
                    },
                };
                let result = fetch_all_list_rows(database, &base_filter)?;

                if !header_written {
                    write_csv_row(&mut out, &result.column_names).map_err(write_err)?;
                    header_written = true;
                }
                for row in &result.rows {
                    let cells: Vec<String> = row.iter().map(cell_text).collect();
                    write_csv_row(&mut out, &cells).map_err(write_err)?;
                }
            }
        }
    }

    Ok(())
}

/// Writes `result` as CSV to `path` in one shot — `result.column_names` as
/// the header, then one line per row. Used by whichever export step already
/// has its whole result in memory (e.g. `export_grouped_by_image`, which
/// pages through `fetch_all_grouped_by_image_rows` and merges before
/// writing); `write_list_csv` above streams instead, since a List export
/// covers every z/t/image plane and its per-plane fetch is already the
/// natural place to write each batch of rows.
fn write_csv(result: &DatabaseResult, path: &Path) -> Result<(), InternalErrors> {
    let mut out = create_csv_writer(path)?;
    let write_err = |e: std::io::Error| {
        InternalErrors::Internal(format!("could not write {}: {e}", path.display()))
    };
    write_csv_row(&mut out, &result.column_names).map_err(write_err)?;
    for row in &result.rows {
        let cells: Vec<String> = row.iter().map(cell_text).collect();
        write_csv_row(&mut out, &cells).map_err(write_err)?;
    }
    Ok(())
}

fn create_csv_writer(path: &Path) -> Result<std::io::BufWriter<std::fs::File>, InternalErrors> {
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        std::fs::create_dir_all(parent).map_err(|e| {
            InternalErrors::Internal(format!("could not create {}: {e}", parent.display()))
        })?;
    }
    let file = std::fs::File::create(path).map_err(|e| {
        InternalErrors::Internal(format!("could not create {}: {e}", path.display()))
    })?;
    Ok(std::io::BufWriter::new(file))
}

fn write_csv_row(out: &mut impl std::io::Write, fields: &[String]) -> std::io::Result<()> {
    let line: Vec<String> = fields.iter().map(|f| csv_escape(f)).collect();
    writeln!(out, "{}", line.join(","))
}

fn csv_escape(field: &str) -> String {
    if field.contains(',') || field.contains('"') || field.contains('\n') || field.contains('\r') {
        format!("\"{}\"", field.replace('"', "\"\""))
    } else {
        field.to_string()
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

/// Excel cell size (both width and height, in pixels) every plate/well/
/// heatmap grid block is laid out at, so its cells read as squares — the
/// grid's own values are unitless relative to a real image/plate scale, so
/// there's no "correct" size to derive them from; this just needs to be
/// visually square and legible.
const GRID_CELL_PX: u32 = 40;

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
    let bold = Format::new().set_bold();
    // Row/col header cells (the top row of column labels and the left
    // column of row labels) — solid black fill, white bold text, so they
    // read as a distinct axis rather than blending into the data cells.
    let header_format = Format::new()
        .set_bold()
        .set_background_color(Color::Black)
        .set_font_color(Color::White);
    worksheet
        .write_with_format(start_row, 0, caption, &bold)
        .map_err(xlsx_err)?;

    let header_row = start_row + 1;
    worksheet
        .set_row_height_pixels(header_row, GRID_CELL_PX)
        .map_err(xlsx_err)?;
    worksheet
        .set_column_width_pixels(0, GRID_CELL_PX)
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
            write_cell(worksheet, row, col, cell)?;
        }
    }

    Ok(header_row + 1 + result.row_names.len() as u32 + 1)
}

/// Writes one pivoted flat-list document: a single sheet named
/// `sheet_name`, headed by `key_labels` (e.g. `["Well"]` or
/// `["Well", "Field", "Image"]`) followed by `combo_labels` (one column per
/// class/column/aggregation combination), then one row per `rows` entry —
/// each a `(key values, one Some(value)-or-None per combo_labels column)`
/// pair. A `None` (that key had no matching data for that particular
/// combination — e.g. a well with no objects of some other class) is
/// written as a plain `"-"`, matching how a missing cell reads everywhere
/// else in these exports. Unlike `write_grid_block`, this is a plain table,
/// not a matrix - no square sizing or per-cell color.
fn write_flat_pivot(
    path: &Path,
    sheet_name: &str,
    key_labels: &[&str],
    combo_labels: &[String],
    rows: impl Iterator<Item = (Vec<String>, Vec<Option<f64>>)>,
) -> Result<(), InternalErrors> {
    let mut workbook = Workbook::new();
    let sheet = workbook.add_worksheet();
    sheet.set_name(sheet_name).map_err(xlsx_err)?;
    let bold = Format::new().set_bold();

    for (col_idx, label) in key_labels.iter().enumerate() {
        sheet
            .write_with_format(0, col_idx as u16, *label, &bold)
            .map_err(xlsx_err)?;
    }
    let key_cols = key_labels.len();
    for (i, label) in combo_labels.iter().enumerate() {
        sheet
            .write_with_format(0, (key_cols + i) as u16, label.as_str(), &bold)
            .map_err(xlsx_err)?;
    }

    for (row_idx, (keys, values)) in rows.enumerate() {
        let row = (row_idx + 1) as u32;
        for (col_idx, key) in keys.iter().enumerate() {
            sheet
                .write(row, col_idx as u16, key.as_str())
                .map_err(xlsx_err)?;
        }
        for (i, value) in values.iter().enumerate() {
            let col = (key_cols + i) as u16;
            match value {
                Some(v) => {
                    sheet.write(row, col, *v).map_err(xlsx_err)?;
                }
                None => {
                    sheet.write(row, col, "-").map_err(xlsx_err)?;
                }
            }
        }
    }

    workbook.save(path).map_err(xlsx_err)?;
    Ok(())
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

/// Writes one `Cell` — its value, typed appropriately (`write_string`/
/// `write_number`, not everything flattened to text, so the sheet stays
/// sortable/usable as real data) rather than pre-formatted display text,
/// plus a background fill: `bg_color` when it's set (`0` is every
/// non-colored cell's sentinel throughout `results_generator.rs`, e.g. a
/// `CellValue::Empty` grid gap, so it's left with Excel's default fill
/// rather than painted black) takes priority since it's real data (e.g. a
/// class badge's own color) — `alternating_color` only ever paints
/// `ALTERNATING_ROW_BG` as a fallback, for a cell that has no color of its
/// own to show.
fn write_cell(
    worksheet: &mut Worksheet,
    row: u32,
    col: u16,
    cell: &Cell,
) -> Result<(), InternalErrors> {
    let format = if cell.bg_color != 0 {
        Some(Format::new().set_background_color(Color::RGB(cell.bg_color)))
    } else if cell.alternating_color {
        Some(Format::new().set_background_color(Color::RGB(ALTERNATING_ROW_BG)))
    } else {
        None
    };

    match &cell.value {
        CellValue::Empty => {
            if let Some(format) = &format {
                worksheet.write_blank(row, col, format).map_err(xlsx_err)?;
            }
        }
        CellValue::String(s) | CellValue::Class((s, _)) => match &format {
            Some(format) => {
                worksheet
                    .write_string_with_format(row, col, s, format)
                    .map_err(xlsx_err)?;
            }
            None => {
                worksheet.write_string(row, col, s).map_err(xlsx_err)?;
            }
        },
        CellValue::Float(value) => match &format {
            Some(format) => {
                worksheet
                    .write_number_with_format(row, col, *value as f64, format)
                    .map_err(xlsx_err)?;
            }
            None => {
                worksheet
                    .write_number(row, col, *value as f64)
                    .map_err(xlsx_err)?;
            }
        },
        CellValue::Integer(value) => match &format {
            Some(format) => {
                worksheet
                    .write_number_with_format(row, col, *value as f64, format)
                    .map_err(xlsx_err)?;
            }
            None => {
                worksheet
                    .write_number(row, col, *value as f64)
                    .map_err(xlsx_err)?;
            }
        },
    }
    Ok(())
}

/// Whether `column` can drive a plate/well/heatmap grid at all — the
/// inverse of `column_aggregate_expr`'s (in results_generator.rs) "cannot be
/// aggregated for the plate view yet" arm, kept in sync with it by
/// inspecting the exact same variants. `self.columns` is shared with the
/// List export, where every column is valid (one row per object, nothing to
/// aggregate), so a grid export must filter it down to this subset itself
/// rather than assume every selected column applies.
fn is_aggregable(column: &Column) -> bool {
    !matches!(
        column,
        Column::ObjectId
            | Column::ImageName
            | Column::ObjectClass
            | Column::IntensityAvg(_)
            | Column::IntensitySum(_)
            | Column::IntensityMin(_)
            | Column::IntensityMax(_)
    )
}

/// `Column`'s human-readable label, matching `get_available_columns()`'s
/// own `display_name` for that column — falls back to the raw db key
/// (`Column::as_key`) for a column this database's `available_columns`
/// doesn't (or no longer) list, e.g. stale export settings.
fn column_display_name(column: &Column, available_columns: &[ColumnEntry]) -> String {
    available_columns
        .iter()
        .find(|entry| entry.key == *column)
        .map(|entry| entry.display_name.clone())
        .unwrap_or_else(|| column.display_label(&[]))
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

/// A usable base filename for `image_rel_path`'s heatmap document — its
/// file stem (no directory, no extension), since the rel path's own
/// separators/extension aren't valid or wanted in a sibling file's name.
fn image_stub(image_rel_path: &str) -> String {
    Path::new(image_rel_path)
        .file_stem()
        .map(|stem| stem.to_string_lossy().into_owned())
        .unwrap_or_else(|| image_rel_path.to_string())
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

#[cfg(test)]
mod tests {
    use super::*;
    use calamine::Reader as _;
    use crate::results::test_support::{ObjectSpec, seed_db};

    #[test]
    fn csv_escape_quotes_fields_containing_commas_or_quotes() {
        assert_eq!(csv_escape("plain"), "plain");
        assert_eq!(csv_escape("a,b"), "\"a,b\"");
        assert_eq!(csv_escape("a\"b"), "\"a\"\"b\"");
    }

    #[test]
    fn csv_escape_leaves_a_plain_field_untouched() {
        assert_eq!(csv_escape("plain text"), "plain text");
        assert_eq!(csv_escape(""), "");
    }

    #[test]
    fn csv_escape_quotes_a_field_containing_a_newline() {
        assert_eq!(csv_escape("a\nb"), "\"a\nb\"");
        assert_eq!(csv_escape("a\rb"), "\"a\rb\"");
    }

    fn cell_str(value: &str) -> Cell {
        Cell {
            value: CellValue::String(value.to_string()),
            bg_color: 0,
            alternating_color: false,
            search_key: None,
        }
    }

    #[test]
    fn cell_text_reads_back_every_cell_value_variant_as_plain_text() {
        assert_eq!(cell_text(&cell_str("hi")), "hi");
        assert_eq!(
            cell_text(&Cell {
                value: CellValue::Empty,
                bg_color: 0,
                alternating_color: false,
                search_key: None,
            }),
            ""
        );
        assert_eq!(
            cell_text(&Cell {
                value: CellValue::Float(1.5),
                bg_color: 0,
                alternating_color: false,
                search_key: None,
            }),
            "1.5"
        );
        assert_eq!(
            cell_text(&Cell {
                value: CellValue::Integer(7),
                bg_color: 0,
                alternating_color: false,
                search_key: None,
            }),
            "7"
        );
        assert_eq!(
            cell_text(&Cell {
                value: CellValue::Class(("ClassA".to_string(), 0)),
                bg_color: 0,
                alternating_color: false,
                search_key: None,
            }),
            "ClassA"
        );
    }

    /// Opens a fresh `ResultsGenerator` over a temp `.evadb` seeded with
    /// `objects` (see `test_support::seed_db`). Leaks the backing `TempDir`
    /// (the returned generator only holds an open `Connection`, not the
    /// directory) - acceptable for a short-lived test process.
    fn open(objects: &[ObjectSpec]) -> (ResultsGenerator, PathBuf) {
        let dir = tempfile::tempdir().expect("tempdir");
        let db_path = dir.path().join("results.evadb");
        seed_db(&db_path, objects);
        let out_dir = dir.path().join("out");
        std::mem::forget(dir);
        (
            ResultsGenerator::open_database(db_path).expect("open database"),
            out_dir,
        )
    }

    fn no_progress() -> impl FnMut(&str, usize, usize) {
        |_message: &str, _current: usize, _total: usize| {}
    }

    fn full_range() -> (Range<u32>, Range<u32>) {
        (Range { start: 0, end: 1 }, Range { start: 0, end: 1 })
    }

    /// End-to-end proof that `start_export`'s CSV path (shared by the CLI
    /// and the GUI export dialog) actually writes a real
    /// `list.csv`/`grouped_by_image.csv` with the expected content — not
    /// just that a caller's own call site happens to work.
    #[test]
    fn start_export_writes_csv_for_list_and_grouped_by_image() {
        let (database, out_dir) = open(&[
            ObjectSpec::new("img1.tif", "ClassA", 1, 100),
            ObjectSpec::new("img2.tif", "ClassB", 2, 200),
        ]);
        let columns: Vec<Column> = database
            .get_available_columns()
            .expect("available columns")
            .into_iter()
            .map(|entry| entry.key)
            .collect();
        let (z_stacks, t_stacks) = full_range();

        let list_export = ResultExport {
            output_dir: out_dir.clone(),
            format: ExportFormat::CSV,
            z_stacks,
            t_stacks,
            columns: columns.clone(),
            with_list_view: true,
            ..Default::default()
        };
        list_export
            .start_export(&database, &mut no_progress())
            .expect("csv list export");
        let list_csv = std::fs::read_to_string(out_dir.join("list.csv")).expect("read list.csv");
        let mut lines = list_csv.lines();
        assert!(lines.next().unwrap().contains("Class"));
        assert_eq!(lines.count(), 2, "expected 2 data rows");
        assert!(list_csv.contains("ClassA"));
        assert!(list_csv.contains("ClassB"));

        let grouped_export = ResultExport {
            output_dir: out_dir.clone(),
            format: ExportFormat::CSV,
            z_stacks,
            t_stacks,
            columns,
            aggregations: vec![Aggregation::Avg],
            with_grouped_by_image_list: true,
            ..Default::default()
        };
        grouped_export
            .start_export(&database, &mut no_progress())
            .expect("csv grouped export");
        let grouped_csv = std::fs::read_to_string(out_dir.join("grouped_by_image.csv"))
            .expect("read grouped_by_image.csv");
        assert_eq!(
            grouped_csv.lines().count(),
            3,
            "expected a header plus one row per image: {grouped_csv}"
        );
    }

    #[test]
    fn start_export_writes_a_valid_xlsx_for_list_and_grouped_by_image() {
        let (database, out_dir) = open(&[
            ObjectSpec::new("img1.tif", "ClassA", 1, 100),
            ObjectSpec::new("img2.tif", "ClassB", 2, 200),
        ]);
        let columns: Vec<Column> = database
            .get_available_columns()
            .expect("available columns")
            .into_iter()
            .map(|entry| entry.key)
            .collect();
        let (z_stacks, t_stacks) = full_range();

        let export = ResultExport {
            output_dir: out_dir.clone(),
            format: ExportFormat::XLSX,
            z_stacks,
            t_stacks,
            columns,
            aggregations: vec![Aggregation::Avg],
            with_list_view: true,
            with_grouped_by_image_list: true,
            ..Default::default()
        };
        export
            .start_export(&database, &mut no_progress())
            .expect("xlsx export");

        for name in ["list.xlsx", "grouped_by_image.xlsx"] {
            let bytes =
                std::fs::read(out_dir.join(name)).unwrap_or_else(|e| panic!("read {name}: {e}"));
            assert!(
                bytes.len() > 4,
                "{name} is too small: {} bytes",
                bytes.len()
            );
            assert_eq!(&bytes[..4], b"PK\x03\x04", "{name} is not a zip/xlsx file");
        }
    }

    #[test]
    fn start_export_with_list_coloc_details_writes_a_second_csv_document() {
        let (database, out_dir) = open(&[ObjectSpec::new("img1.tif", "ClassA", 1, 100)]);
        let columns: Vec<Column> = database
            .get_available_columns()
            .expect("available columns")
            .into_iter()
            .map(|entry| entry.key)
            .collect();
        let (z_stacks, t_stacks) = full_range();

        let export = ResultExport {
            output_dir: out_dir.clone(),
            format: ExportFormat::CSV,
            z_stacks,
            t_stacks,
            columns,
            with_list_view: true,
            with_list_coloc_details: true,
            ..Default::default()
        };
        export
            .start_export(&database, &mut no_progress())
            .expect("csv export with coloc details");

        assert!(out_dir.join("list.csv").exists());
        assert!(out_dir.join("list_coloc_details.csv").exists());
    }

    #[test]
    fn start_export_with_one_file_per_image_writes_a_document_per_image() {
        let (database, out_dir) = open(&[
            ObjectSpec::new("img1.tif", "ClassA", 1, 100),
            ObjectSpec::new("img2.tif", "ClassB", 2, 200),
        ]);
        let columns: Vec<Column> = database
            .get_available_columns()
            .expect("available columns")
            .into_iter()
            .map(|entry| entry.key)
            .collect();
        let (z_stacks, t_stacks) = full_range();

        let export = ResultExport {
            output_dir: out_dir.clone(),
            format: ExportFormat::CSV,
            z_stacks,
            t_stacks,
            columns,
            with_list_view: true,
            with_list_one_file_per_image: true,
            ..Default::default()
        };
        export
            .start_export(&database, &mut no_progress())
            .expect("csv per-image export");

        let img1 = std::fs::read_to_string(out_dir.join("list_img1.csv")).expect("list_img1.csv");
        let img2 = std::fs::read_to_string(out_dir.join("list_img2.csv")).expect("list_img2.csv");
        assert!(img1.contains("ClassA") && !img1.contains("ClassB"));
        assert!(img2.contains("ClassB") && !img2.contains("ClassA"));
    }

    #[test]
    fn start_export_rejects_csv_combined_with_plate_view() {
        let (database, out_dir) = open(&[ObjectSpec::new("img1.tif", "ClassA", 1, 100)]);
        let (z_stacks, t_stacks) = full_range();
        let export = ResultExport {
            output_dir: out_dir,
            format: ExportFormat::CSV,
            z_stacks,
            t_stacks,
            with_plate_view: true,
            ..Default::default()
        };
        let result = export.start_export(&database, &mut no_progress());
        assert!(result.is_err());
    }

    #[test]
    fn start_export_rejects_csv_combined_with_heatmap() {
        let (database, out_dir) = open(&[ObjectSpec::new("img1.tif", "ClassA", 1, 100)]);
        let (z_stacks, t_stacks) = full_range();
        let export = ResultExport {
            output_dir: out_dir,
            format: ExportFormat::CSV,
            z_stacks,
            t_stacks,
            with_heatmap: true,
            ..Default::default()
        };
        let result = export.start_export(&database, &mut no_progress());
        assert!(result.is_err());
    }

    /// The Parquet path (`export_as_parquet`) executes DuckDB's own
    /// `COPY objects TO '<path>' (FORMAT parquet)` directly against the raw
    /// `objects` table — this proves it actually produces a file DuckDB
    /// itself can read back, with every seeded row intact, and (unlike
    /// CSV/XLSX) completely ignoring `columns`/`with_list_view`/filters.
    #[test]
    fn start_export_parquet_dumps_the_whole_objects_table_ignoring_other_settings() {
        let (database, out_dir) = open(&[
            ObjectSpec::new("img1.tif", "ClassA", 1, 100),
            ObjectSpec::new("img2.tif", "ClassB", 2, 200),
            ObjectSpec::new("img2.tif", "ClassB", 2, 300),
        ]);
        let export = ResultExport {
            output_dir: out_dir.clone(),
            format: ExportFormat::Parquet,
            // Deliberately left at defaults / unset: `columns` is empty,
            // every `with_*` flag is false - Parquet must ignore all of it.
            ..Default::default()
        };
        export
            .start_export(&database, &mut no_progress())
            .expect("parquet export");

        let parquet_path = out_dir.join("objects.parquet");
        let bytes = std::fs::read(&parquet_path).expect("read objects.parquet");
        assert!(bytes.len() > 8);
        assert_eq!(&bytes[..4], b"PAR1", "missing leading PAR1 magic");
        assert_eq!(
            &bytes[bytes.len() - 4..],
            b"PAR1",
            "missing trailing PAR1 magic"
        );

        // Read it back through a *separate* DuckDB connection - proof the
        // file is genuinely valid Parquet, not just magic-byte-shaped.
        let verify = duckdb::Connection::open_in_memory().expect("open verify connection");
        let count: i64 = verify
            .query_row(
                &format!(
                    "SELECT COUNT(*) FROM read_parquet('{}')",
                    parquet_path.to_string_lossy().replace('\'', "''")
                ),
                [],
                |row| row.get(0),
            )
            .expect("query parquet row count");
        assert_eq!(count, 3, "expected every seeded object, unfiltered");
    }

    /// `write_list_document`'s `Parquet` arm is unreachable through
    /// `start_export` (which returns early for Parquet before ever calling
    /// `export_list`) - this proves it's still a real, safe error rather
    /// than a panic if something calls it directly.
    #[test]
    fn write_list_document_rejects_parquet_directly() {
        let (database, out_dir) = open(&[ObjectSpec::new("img1.tif", "ClassA", 1, 100)]);
        let export = ResultExport {
            output_dir: out_dir,
            format: ExportFormat::Parquet,
            ..Default::default()
        };
        let result = export.write_list_document(&database, &[], &None, "list");
        assert!(result.is_err());
    }

    /// Same defensive-arm proof as `write_list_document_rejects_parquet_directly`,
    /// for `export_grouped_by_image`'s equivalent early return.
    #[test]
    fn export_grouped_by_image_rejects_parquet_directly() {
        let (database, out_dir) = open(&[ObjectSpec::new("img1.tif", "ClassA", 1, 100)]);
        let export = ResultExport {
            output_dir: out_dir,
            format: ExportFormat::Parquet,
            ..Default::default()
        };
        let result = export.export_grouped_by_image(&database, &mut no_progress());
        assert!(result.is_err());
    }

    #[test]
    fn resolve_images_defaults_to_every_non_disabled_image_sorted_by_name() {
        let (database, out_dir) = open(&[
            ObjectSpec::new("img2.tif", "ClassA", 1, 100),
            ObjectSpec::new("img1.tif", "ClassA", 1, 100),
        ]);
        let export = ResultExport {
            output_dir: out_dir,
            ..Default::default()
        };
        let images = resolve_images(&database, &export).expect("resolve images");
        assert_eq!(images, vec!["img1.tif".to_string(), "img2.tif".to_string()]);
    }

    #[test]
    fn resolve_images_with_an_explicit_list_returns_only_those_images() {
        let (database, out_dir) = open(&[
            ObjectSpec::new("img1.tif", "ClassA", 1, 100),
            ObjectSpec::new("img2.tif", "ClassA", 1, 100),
        ]);
        let export = ResultExport {
            output_dir: out_dir,
            image_rel_paths: vec!["img2.tif".to_string()],
            ..Default::default()
        };
        let images = resolve_images(&database, &export).expect("resolve images");
        assert_eq!(images, vec!["img2.tif".to_string()]);
    }

    #[test]
    fn is_aggregable_excludes_identity_and_intensity_columns() {
        assert!(!is_aggregable(&Column::ObjectId));
        assert!(!is_aggregable(&Column::ImageName));
        assert!(!is_aggregable(&Column::ObjectClass));
        assert!(!is_aggregable(&Column::IntensityAvg(0)));
        assert!(is_aggregable(&Column::AreaSizePx));
        assert!(is_aggregable(&Column::Circularity));
    }

    #[test]
    fn image_stub_strips_directory_and_extension() {
        assert_eq!(image_stub("folder/img1.tif"), "img1");
        assert_eq!(image_stub("img2.ome.tif"), "img2.ome");
        assert_eq!(image_stub("no_extension"), "no_extension");
    }

    #[test]
    fn sanitize_filename_component_replaces_unsafe_characters() {
        assert_eq!(sanitize_filename_component("a/b:c"), "a_b_c");
        assert_eq!(sanitize_filename_component("plain-name_1"), "plain-name_1");
    }

    #[test]
    fn sheet_namer_deduplicates_and_truncates_long_names() {
        let mut namer = SheetNamer::new();
        assert_eq!(namer.unique("Class A"), "Class A");
        assert_eq!(namer.unique("Class A"), "Class A (2)");
        assert_eq!(namer.unique("Class A"), "Class A (3)");

        let long_name = "x".repeat(40);
        let sanitized = namer.unique(&long_name);
        assert!(sanitized.chars().count() <= 31);
    }

    #[test]
    fn sheet_namer_rejects_excel_forbidden_characters() {
        let mut namer = SheetNamer::new();
        let name = namer.unique("a[b]:c*d?e/f\\g");
        assert!(!name.contains(['[', ']', ':', '*', '?', '/', '\\']));
    }

    // -- calamine smoke test: confirms the reader API used below -------------

    #[test]
    fn calamine_can_read_back_a_real_exported_xlsx() {
        use calamine::{Data, Reader, Xlsx, open_workbook};

        let (database, out_dir) = open(&[ObjectSpec::new("img1.tif", "ClassA", 1, 100)]);
        let columns: Vec<Column> = database
            .get_available_columns()
            .expect("available columns")
            .into_iter()
            .map(|entry| entry.key)
            .collect();
        let (z_stacks, t_stacks) = full_range();
        let export = ResultExport {
            output_dir: out_dir.clone(),
            format: ExportFormat::XLSX,
            z_stacks,
            t_stacks,
            columns,
            with_list_view: true,
            ..Default::default()
        };
        export
            .start_export(&database, &mut no_progress())
            .expect("xlsx export");

        let mut workbook: Xlsx<_> =
            open_workbook(out_dir.join("list.xlsx")).expect("open exported xlsx");
        let range = workbook.worksheet_range("List").expect("List sheet");
        assert_eq!(range.get_value((0, 0)), Some(&Data::String("Object ID".to_string())));
    }

    // -- plate/well/heatmap exports: plausibility against the same values --
    // `ResultsGenerator` reports through its own query methods.
    //
    // Each of these re-derives the *expected* grid by calling the exact same
    // `ResultsGenerator` method `export_plate_and_well`/`export_heatmap`
    // calls internally, then reads the real produced XLSX file back with
    // `calamine` and asserts every header/row-label/value cell matches - this
    // is what actually exercises `write_grid_block`'s XLSX-writing logic
    // (previously untested), and proves the exported file faithfully
    // reflects the grid `ResultsGenerator` computed rather than the two
    // silently drifting apart.

    fn data_f64(data: Option<&calamine::Data>) -> f64 {
        match data {
            Some(calamine::Data::Float(v)) => *v,
            Some(calamine::Data::Int(v)) => *v as f64,
            other => panic!("expected a numeric cell, got {other:?}"),
        }
    }

    /// Asserts that `range`'s grid starting at `start_row` (as written by
    /// `write_grid_block`: a caption row, then a header row of
    /// `expected.column_names`, then one row per `expected.row_names`)
    /// matches `expected` cell-for-cell.
    fn assert_grid_matches_at(
        range: &calamine::Range<calamine::Data>,
        start_row: u32,
        expected: &DatabaseResult,
    ) {
        let header_row = start_row + 1;
        for (col_idx, name) in expected.column_names.iter().enumerate() {
            let col = (col_idx + 1) as u32;
            assert_eq!(
                range.get_value((header_row, col)),
                Some(&calamine::Data::String(name.clone())),
                "column header at ({header_row}, {col})"
            );
        }
        for (row_idx, row_name) in expected.row_names.iter().enumerate() {
            let row = header_row + 1 + row_idx as u32;
            assert_eq!(
                range.get_value((row, 0)),
                Some(&calamine::Data::String(row_name.clone())),
                "row label at ({row}, 0)"
            );
            for (col_idx, cell) in expected.rows[row_idx].iter().enumerate() {
                let col = (col_idx + 1) as u32;
                match &cell.value {
                    CellValue::Float(v) => {
                        let got = data_f64(range.get_value((row, col)));
                        assert!(
                            (got - *v as f64).abs() < 1e-6,
                            "value at ({row}, {col}): expected {v}, got {got}"
                        );
                    }
                    CellValue::Empty => {
                        assert!(
                            matches!(
                                range.get_value((row, col)),
                                None | Some(calamine::Data::Empty)
                            ),
                            "expected an empty cell at ({row}, {col})"
                        );
                    }
                    _ => {}
                }
            }
        }
    }

    #[test]
    fn start_export_xlsx_plate_grid_matches_get_group_by_plate_heatmap() {
        let (database, out_dir) = open(&[
            ObjectSpec::new("A1_01.tif", "ClassA", 1, 10),
            ObjectSpec::new("A1_02.tif", "ClassA", 1, 20),
            ObjectSpec::new("B2_01.tif", "ClassA", 1, 100),
        ]);
        let export = ResultExport {
            output_dir: out_dir.clone(),
            format: ExportFormat::XLSX,
            columns: vec![Column::AreaSizePx],
            aggregations: vec![Aggregation::Avg],
            with_plate_view: true,
            ..Default::default()
        };
        export
            .start_export(&database, &mut no_progress())
            .expect("plate export");

        // Same filter `export_plate_and_well` builds internally for this
        // (single class, single column, single aggregation) configuration.
        let expected = database
            .get_group_by_plate(
                &PlateFilter {
                    plane: PlaneFilter {
                        z_stack: 0,
                        t_stack: 0,
                    },
                    grouping_regex: String::new(),
                    aggregation: Aggregation::Avg,
                    object_class: ObjectClass::Valid(1),
                    column: Column::AreaSizePx,
                    color_schema: ColorSchema::default(),
                    color_scale: ColorScale::default(),
                    matrix_dimension: None,
                },
                &View::Heatmap,
            )
            .expect("expected plate grid");

        let mut workbook: calamine::Xlsx<_> =
            calamine::open_workbook(out_dir.join("plate.xlsx")).expect("open plate.xlsx");
        let range = workbook.worksheet_range("ClassA").expect("ClassA sheet");
        assert_grid_matches_at(&range, 0, &expected);

        // Belt-and-suspenders on the actual numbers, independent of the
        // `DatabaseResult` plumbing above.
        assert_eq!(data_f64(range.get_value((2, 1))), 15.0, "A1 = avg(10, 20)");
        assert_eq!(data_f64(range.get_value((3, 2))), 100.0, "B2");
    }

    #[test]
    fn start_export_xlsx_well_grid_matches_get_wells_for_plate_heatmap() {
        let (database, out_dir) = open(&[
            ObjectSpec::new("A1_01.tif", "ClassA", 1, 10),
            ObjectSpec::new("A1_02.tif", "ClassA", 1, 20),
        ]);
        let export = ResultExport {
            output_dir: out_dir.clone(),
            format: ExportFormat::XLSX,
            columns: vec![Column::AreaSizePx],
            aggregations: vec![Aggregation::Avg],
            with_plate_view: true,
            ..Default::default()
        };
        export
            .start_export(&database, &mut no_progress())
            .expect("well export");

        let expected_by_well = database
            .get_wells_for_plate(
                &WellsBatchFilter {
                    plane: PlaneFilter {
                        z_stack: 0,
                        t_stack: 0,
                    },
                    grouping_regex: String::new(),
                    aggregation: Aggregation::Avg,
                    object_class: ObjectClass::Valid(1),
                    column: Column::AreaSizePx,
                    color_schema: ColorSchema::default(),
                    color_scale: ColorScale::default(),
                    well_size: None,
                    well_order: None,
                },
                &View::Heatmap,
            )
            .expect("expected well grid");
        assert_eq!(expected_by_well.len(), 1, "only well A1 was seeded");
        let expected = &expected_by_well["A1"];

        let mut workbook: calamine::Xlsx<_> =
            calamine::open_workbook(out_dir.join("well.xlsx")).expect("open well.xlsx");
        let range = workbook.worksheet_range("ClassA").expect("ClassA sheet");
        assert_grid_matches_at(&range, 0, expected);
        assert_eq!(data_f64(range.get_value((2, 1))), 10.0, "field 01");
        assert_eq!(data_f64(range.get_value((2, 2))), 20.0, "field 02");
    }

    #[test]
    fn start_export_xlsx_heatmap_grid_matches_get_image_heatmap() {
        let (database, out_dir) = open(&[
            ObjectSpec::new("img1.tif", "ClassA", 1, 10)
                .at_centroid(10.0, 10.0)
                .with_image_size(100, 100),
            ObjectSpec::new("img1.tif", "ClassA", 1, 20)
                .at_centroid(60.0, 10.0)
                .with_image_size(100, 100),
        ]);
        let export = ResultExport {
            output_dir: out_dir.clone(),
            format: ExportFormat::XLSX,
            columns: vec![Column::AreaSizePx],
            aggregations: vec![Aggregation::Avg],
            square_size: Some(50),
            with_heatmap: true,
            ..Default::default()
        };
        export
            .start_export(&database, &mut no_progress())
            .expect("heatmap export");

        let expected = database
            .get_image_heatmap(
                &ImageHeatmapFilter {
                    plane: PlaneFilter {
                        z_stack: 0,
                        t_stack: 0,
                    },
                    image_rel_path: "img1.tif".to_string(),
                    aggregation: Aggregation::Avg,
                    object_class: ObjectClass::Valid(1),
                    column: Column::AreaSizePx,
                    color_schema: ColorSchema::default(),
                    color_scale: ColorScale::default(),
                    square_size: Some(50),
                },
                &View::Heatmap,
            )
            .expect("expected image heatmap grid");

        let mut workbook: calamine::Xlsx<_> =
            calamine::open_workbook(out_dir.join("heatmap_img1.xlsx")).expect("open heatmap_img1.xlsx");
        let range = workbook.worksheet_range("ClassA").expect("ClassA sheet");
        assert_grid_matches_at(&range, 0, &expected);
        assert_eq!(data_f64(range.get_value((2, 1))), 10.0, "R0C0");
        assert_eq!(data_f64(range.get_value((2, 2))), 20.0, "R0C1");
    }

    #[test]
    fn start_export_xlsx_plate_list_and_well_list_report_the_seeded_average() {
        let (database, out_dir) = open(&[
            ObjectSpec::new("A1_01.tif", "ClassA", 1, 10),
            ObjectSpec::new("A1_02.tif", "ClassA", 1, 20),
        ]);
        let export = ResultExport {
            output_dir: out_dir.clone(),
            format: ExportFormat::XLSX,
            columns: vec![Column::AreaSizePx],
            aggregations: vec![Aggregation::Avg],
            with_plates_and_wells_as_list: true,
            ..Default::default()
        };
        export
            .start_export(&database, &mut no_progress())
            .expect("flat-pivot export");

        let mut plate_wb: calamine::Xlsx<_> =
            calamine::open_workbook(out_dir.join("plate_list.xlsx")).expect("open plate_list.xlsx");
        let plate_range = plate_wb.worksheet_range("Plate").expect("Plate sheet");
        assert_eq!(
            plate_range.get_value((0, 0)),
            Some(&calamine::Data::String("Well".to_string()))
        );
        assert_eq!(
            plate_range.get_value((1, 0)),
            Some(&calamine::Data::String("A1".to_string()))
        );
        assert_eq!(data_f64(plate_range.get_value((1, 1))), 15.0);

        let mut well_wb: calamine::Xlsx<_> =
            calamine::open_workbook(out_dir.join("well_list.xlsx")).expect("open well_list.xlsx");
        let well_range = well_wb.worksheet_range("Well").expect("Well sheet");
        assert_eq!(
            well_range.get_value((0, 0)),
            Some(&calamine::Data::String("Well".to_string()))
        );
        // Two data rows: field "01" (value 10) and field "02" (value 20),
        // sorted by (well, field-as-number) - see `export_plate_and_well_as_flat_list`.
        assert_eq!(data_f64(well_range.get_value((1, 3))), 10.0);
        assert_eq!(data_f64(well_range.get_value((2, 3))), 20.0);
    }

    #[test]
    fn start_export_runs_every_xlsx_document_type_together() {
        let (database, out_dir) = open(&[
            ObjectSpec::new("A1_01.tif", "ClassA", 1, 10)
                .at_centroid(10.0, 10.0)
                .with_image_size(100, 100),
            ObjectSpec::new("B2_01.tif", "ClassA", 1, 20)
                .at_centroid(10.0, 10.0)
                .with_image_size(100, 100),
        ]);
        let columns: Vec<Column> = database
            .get_available_columns()
            .expect("available columns")
            .into_iter()
            .map(|entry| entry.key)
            .collect();
        let (z_stacks, t_stacks) = full_range();
        let export = ResultExport {
            output_dir: out_dir.clone(),
            format: ExportFormat::XLSX,
            z_stacks,
            t_stacks,
            columns,
            aggregations: vec![Aggregation::Avg],
            with_list_view: true,
            with_grouped_by_image_list: true,
            with_plate_view: true,
            with_plates_and_wells_as_list: true,
            with_heatmap: true,
            ..Default::default()
        };
        export
            .start_export(&database, &mut no_progress())
            .expect("full xlsx export");

        for name in [
            "list.xlsx",
            "grouped_by_image.xlsx",
            "plate.xlsx",
            "well.xlsx",
            "plate_list.xlsx",
            "well_list.xlsx",
            "heatmap_A1_01.xlsx",
            "heatmap_B2_01.xlsx",
        ] {
            let bytes = std::fs::read(out_dir.join(name)).unwrap_or_else(|e| panic!("read {name}: {e}"));
            assert_eq!(&bytes[..4], b"PK\x03\x04", "{name} is not a zip/xlsx file");
        }
    }

    #[test]
    fn start_export_object_classes_filter_restricts_plate_well_flat_list_and_heatmap() {
        let (database, out_dir) = open(&[
            ObjectSpec::new("A1_01.tif", "ClassA", 1, 10)
                .at_centroid(10.0, 10.0)
                .with_image_size(100, 100),
            ObjectSpec::new("A1_01.tif", "ClassB", 2, 999)
                .at_centroid(10.0, 10.0)
                .with_image_size(100, 100),
        ]);
        let export = ResultExport {
            output_dir: out_dir.clone(),
            format: ExportFormat::XLSX,
            columns: vec![Column::AreaSizePx],
            aggregations: vec![Aggregation::Avg],
            object_classes: vec![ObjectClass::Valid(1)],
            square_size: Some(50),
            with_plate_view: true,
            with_plates_and_wells_as_list: true,
            with_heatmap: true,
            ..Default::default()
        };
        export
            .start_export(&database, &mut no_progress())
            .expect("filtered export");

        for file in ["plate.xlsx", "well.xlsx", "heatmap_A1_01.xlsx"] {
            let workbook: calamine::Xlsx<_> =
                calamine::open_workbook(out_dir.join(file)).unwrap_or_else(|e| panic!("open {file}: {e}"));
            let sheets = workbook.sheet_names();
            assert_eq!(
                sheets,
                vec!["ClassA".to_string()],
                "{file} must only have a sheet for the selected class, not ClassB: {sheets:?}"
            );
        }

        let mut plate_list_wb: calamine::Xlsx<_> =
            calamine::open_workbook(out_dir.join("plate_list.xlsx")).expect("open plate_list.xlsx");
        let plate_list_range = plate_list_wb.worksheet_range("Plate").expect("Plate sheet");
        let header = plate_list_range.get_value((0, 1)).expect("combo header");
        assert!(
            matches!(header, calamine::Data::String(s) if s.contains("ClassA") && !s.contains("ClassB")),
            "plate_list.xlsx combo header must only reference the selected class: {header:?}"
        );
    }

    #[test]
    fn start_export_with_list_coloc_details_xlsx_writes_a_second_sheet_with_shaded_rows() {
        let objects = vec![
            ObjectSpec::new("img1.tif", "ClassA", 1, 10)
                .with_coloc(r#"{"2":["00000000-0000-0000-0000-000000000001"]}"#),
            ObjectSpec::new("img1.tif", "ClassB", 2, 99),
        ];
        let (database, out_dir) = open(&objects);
        let (z_stacks, t_stacks) = full_range();
        let export = ResultExport {
            output_dir: out_dir.clone(),
            format: ExportFormat::XLSX,
            z_stacks,
            t_stacks,
            columns: vec![
                Column::ObjectClass,
                Column::AreaSizePx,
                Column::ColocCount(ObjectClass::Valid(2)),
            ],
            with_list_view: true,
            with_list_coloc_details: true,
            ..Default::default()
        };
        export
            .start_export(&database, &mut no_progress())
            .expect("xlsx export with coloc details");

        let mut workbook: calamine::Xlsx<_> =
            calamine::open_workbook(out_dir.join("list.xlsx")).expect("open list.xlsx");
        let sheets = workbook.sheet_names();
        assert!(sheets.contains(&"List".to_string()));
        assert!(sheets.contains(&"List (Coloc Details)".to_string()));
        let coloc_range = workbook
            .worksheet_range("List (Coloc Details)")
            .expect("coloc details sheet");
        // 2 fanned-out rows (one per source object) plus the header.
        assert_eq!(coloc_range.rows().count(), 3);
    }

    #[test]
    fn start_export_reports_an_error_when_the_output_directory_cannot_be_created() {
        let (database, _out_dir) = open(&[ObjectSpec::new("img1.tif", "ClassA", 1, 100)]);
        let dir = tempfile::tempdir().expect("tempdir");
        let blocking_file = dir.path().join("blocker");
        std::fs::write(&blocking_file, b"not a directory").expect("write blocking file");
        let export = ResultExport {
            // A regular file can't have a subdirectory created under it.
            output_dir: blocking_file.join("sub"),
            format: ExportFormat::CSV,
            with_list_view: true,
            ..Default::default()
        };
        let result = export.start_export(&database, &mut no_progress());
        assert!(result.is_err());
    }

    #[test]
    fn start_export_xlsx_save_failure_surfaces_as_an_internal_error() {
        let (database, out_dir) = open(&[ObjectSpec::new("img1.tif", "ClassA", 1, 100)]);
        // Pre-create a *directory* at the exact path `list.xlsx` would be
        // saved to, so `Workbook::save` fails instead of succeeding.
        std::fs::create_dir_all(out_dir.join("list.xlsx")).expect("pre-create blocking directory");
        let columns: Vec<Column> = database
            .get_available_columns()
            .expect("available columns")
            .into_iter()
            .map(|entry| entry.key)
            .collect();
        let export = ResultExport {
            output_dir: out_dir,
            format: ExportFormat::XLSX,
            columns,
            with_list_view: true,
            ..Default::default()
        };
        let result = export.start_export(&database, &mut no_progress());
        assert!(result.is_err());
    }

    #[test]
    fn start_export_grouped_by_image_with_a_non_matching_class_filter_is_an_empty_but_valid_document() {
        let (database, out_dir) = open(&[ObjectSpec::new("img1.tif", "ClassA", 1, 100)]);
        let export = ResultExport {
            output_dir: out_dir.clone(),
            format: ExportFormat::CSV,
            columns: vec![Column::AreaSizePx],
            aggregations: vec![Aggregation::Avg],
            // No class 99 was ever seeded - a non-empty selection matching
            // zero rows, exercising `fetch_all_grouped_by_image_rows`'s
            // "no rows at all" min/max fallback (as opposed to an
            // explicitly-empty `Some(vec![])` selection, which short-
            // circuits earlier).
            object_classes: vec![ObjectClass::Valid(99)],
            with_grouped_by_image_list: true,
            ..Default::default()
        };
        export
            .start_export(&database, &mut no_progress())
            .expect("export with a non-matching filter should still succeed");
        let content =
            std::fs::read_to_string(out_dir.join("grouped_by_image.csv")).expect("read csv");
        assert_eq!(content.lines().count(), 1, "header only, no data rows");
    }

    #[test]
    fn aggregation_label_covers_every_variant() {
        assert_eq!(aggregation_label(&Aggregation::Avg), "Average");
        assert_eq!(aggregation_label(&Aggregation::Min), "Minimum");
        assert_eq!(aggregation_label(&Aggregation::Max), "Maximum");
        assert_eq!(aggregation_label(&Aggregation::Stddev), "Std Dev");
        assert_eq!(aggregation_label(&Aggregation::Sum), "Sum");
        assert_eq!(aggregation_label(&Aggregation::Median), "Median");
        assert_eq!(aggregation_label(&Aggregation::Skewness), "Skewness");
    }

    #[test]
    fn cell_to_f64_reads_both_numeric_cell_variants_and_rejects_others() {
        let float_cell = Cell {
            value: CellValue::Float(1.5),
            bg_color: 0,
            alternating_color: false,
            search_key: None,
        };
        let int_cell = Cell {
            value: CellValue::Integer(7),
            bg_color: 0,
            alternating_color: false,
            search_key: None,
        };
        let string_cell = Cell {
            value: CellValue::String("x".to_string()),
            bg_color: 0,
            alternating_color: false,
            search_key: None,
        };
        assert_eq!(cell_to_f64(&float_cell), Some(1.5));
        assert_eq!(cell_to_f64(&int_cell), Some(7.0));
        assert_eq!(cell_to_f64(&string_cell), None);
    }

    #[test]
    fn sheet_namer_falls_back_to_a_generic_name_for_a_blank_class_name() {
        let mut namer = SheetNamer::new();
        assert_eq!(namer.unique("   "), "Sheet");
    }
}
