//! # table
//!
//! A `TABLE` entry in table format 1 (`docs/exports.md` §3): the rows the
//! results' queries give, with column names made for scripts (the column
//! key, the unit in it), lengths in the export's unit, the leading columns
//! first, in CSV or XLSX - split into a file (or sheet) per image, folder
//! or class.
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-11-24
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.

use super::Output;
use crate::exports::jobs::{TableExport, TableRows};
use crate::exports::naming;
use crate::exports::types::EntryReport;
use crate::results::view_settings::resolve_columns;
use crate::results::{
    Aggregation, Cell, CellValue, ChildValue, Column, GroupedByImageFilter, Grouping, ListFilter,
    Pagination, ParentValue, PlaneFilter, ResultsSource,
};
use evanalyzer_cfg::core_types::{InternalErrors, ObjectClass};
use evanalyzer_cfg::settings::classification_settings::Class;
use evanalyzer_cfg::settings::export_settings::TimeCourseTableLayout;
use evanalyzer_cfg::settings::export_settings::{
    ExportAggregation, ExportSettings, ImageScope, LengthUnit, SplitInto, TableFileFormat,
    TableSplit,
};
use rust_xlsxwriter::{Workbook, Worksheet};
use std::collections::BTreeMap;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};

/// Rows per query page.
const PAGE_SIZE: i32 = 20_000;

/// The value columns a table has without `columns`.
pub(crate) const DEFAULT_OBJECT_COLUMNS: [Column; 3] =
    [Column::AreaSizePx, Column::AreaSizeNm, Column::Circularity];
pub(crate) const DEFAULT_GROUPED_COLUMNS: [Column; 3] =
    [Column::Count, Column::AreaSizeNm, Column::Circularity];

/// Writes `table`, its files into `report`.
pub(super) fn write(
    results: &dyn ResultsSource,
    export: &ExportSettings,
    table: &TableExport,
    output: &Output,
    report: &mut EntryReport,
    cancel: &AtomicBool,
) -> Result<(), InternalErrors> {
    let rows: Box<dyn Rows> = match table.rows {
        TableRows::Track => Box::new(super::tracks::TrackRows::new(
            results,
            export,
            table,
            &mut report.notes,
        )?),
        TableRows::Measurement => Box::new(super::measurements::MeasurementTable::new(
            results,
            table,
            &mut report.notes,
        )?),
        TableRows::ProfilePosition => Box::new(super::series::ProfileRows::new(results, table)?),
        TableRows::TimePoint => {
            let rows =
                super::series::TimeCourseRows::new(results, export, table, &mut report.notes)?;
            let name = naming::with_prefix(export, &naming::table_name(table));
            let layout = rows.layout();
            if layout != TimeCourseTableLayout::Long {
                rows.write_wide(results, &name, table.format, output, report, cancel)?;
            }
            if layout == TimeCourseTableLayout::Wide {
                return Ok(());
            }
            Box::new(rows)
        }
        _ => Box::new(Plan::new(results, export, table, &mut report.notes)?),
    };
    let name = naming::with_prefix(export, &naming::table_name(table));
    let parts = parts(results, table, rows.images(), rows.classes())?;

    match (table.format, table.split_into) {
        (TableFileFormat::Csv, _) => {
            for part in &parts {
                let file = output.file(&format!("{}.csv", part.file_name(&name)));
                let path = output.path(&file);
                let mut sink = CsvSink::create(&path)?;
                rows.write_part(results, part, &mut sink, cancel)?;
                sink.finish()?;
                report.files.push(file);
            }
        }
        (TableFileFormat::Xlsx, SplitInto::Files) => {
            for part in &parts {
                let file = output.file(&format!("{}.xlsx", part.file_name(&name)));
                write_workbook(
                    &output.path(&file),
                    rows.as_ref(),
                    results,
                    std::slice::from_ref(part),
                    cancel,
                )?;
                report.files.push(file);
            }
        }
        (TableFileFormat::Xlsx, SplitInto::Sheets) => {
            let file = output.file(&format!("{name}.xlsx"));
            write_workbook(&output.path(&file), rows.as_ref(), results, &parts, cancel)?;
            report.files.push(file);
        }
    }
    Ok(())
}

/// The rows of a table: what a part of it holds, written into a sink.
pub(super) trait Rows {
    /// The images the table covers (relative paths, `/` separated).
    fn images(&self) -> &[String];
    /// The classes the table covers; `None`: all.
    fn classes(&self) -> Option<Vec<ObjectClass>>;
    /// The header, then the rows of `part`.
    fn write_part(
        &self,
        results: &dyn ResultsSource,
        part: &Part,
        sink: &mut dyn Sink,
        cancel: &AtomicBool,
    ) -> Result<(), InternalErrors>;
    /// The `About` sheet's rows: each column's name, label and unit.
    fn about(&self) -> Vec<[String; 3]>;
}

/// One file (or sheet) of a table: its name part and what it covers.
pub(super) struct Part {
    /// `None`: the table isn't split.
    pub(super) suffix: Option<String>,
    pub(super) images: Vec<String>,
    pub(super) classes: Option<Vec<ObjectClass>>,
}

impl Part {
    fn file_name(&self, name: &str) -> String {
        match &self.suffix {
            Some(suffix) => format!("{name}_{suffix}"),
            None => name.to_string(),
        }
    }

    fn sheet_name(&self) -> String {
        self.suffix.clone().unwrap_or_else(|| "Table".to_string())
    }
}

/// The parts `table` is split into.
fn parts(
    results: &dyn ResultsSource,
    table: &TableExport,
    images: &[String],
    classes: Option<Vec<ObjectClass>>,
) -> Result<Vec<Part>, InternalErrors> {
    let all = |images: Vec<String>, suffix| Part {
        suffix,
        images,
        classes: classes.clone(),
    };
    Ok(match table.split {
        TableSplit::None => vec![all(images.to_vec(), None)],
        TableSplit::Image => {
            let names = naming::unique_image_parts(&images);
            images
                .iter()
                .map(|image| all(vec![image.clone()], Some(names[image].clone())))
                .collect()
        }
        TableSplit::Folder => {
            let mut folders: BTreeMap<String, Vec<String>> = BTreeMap::new();
            for image in images {
                let folder = image
                    .rsplit_once('/')
                    .map(|(folder, _)| folder.to_string())
                    .unwrap_or_default();
                folders.entry(folder).or_default().push(image.clone());
            }
            folders
                .into_iter()
                .map(|(folder, images)| {
                    let suffix = if folder.is_empty() {
                        "_".to_string()
                    } else {
                        naming::sanitize(&folder)
                    };
                    all(images, Some(suffix))
                })
                .collect()
        }
        TableSplit::Class => {
            let wanted: Vec<(u32, String)> = results
                .get_object_classes()?
                .into_iter()
                .filter(|class| class.id != 0)
                .filter(|class| {
                    classes
                        .as_ref()
                        .is_none_or(|classes| classes.contains(&ObjectClass::Global(class.id)))
                })
                .map(|class| (class.id, class.name))
                .collect();
            let names = naming::unique_parts(&wanted);
            wanted
                .iter()
                .map(|(id, _)| Part {
                    suffix: Some(names[id].clone()),
                    images: images.to_vec(),
                    classes: Some(vec![ObjectClass::Global(*id)]),
                })
                .collect()
        }
        TableSplit::Well => {
            // The well as the results group their images: the grouping's
            // pattern on the image's name (its first capture group).
            let pattern = results.grouping_regex(&Grouping::Auto)?;
            let regex = regex::Regex::new(&pattern).map_err(|e| {
                InternalErrors::Internal(format!("the well pattern “{pattern}” doesn't work: {e}"))
            })?;
            let mut wells: BTreeMap<String, Vec<String>> = BTreeMap::new();
            for image in images {
                let name = image.rsplit('/').next().unwrap_or(image);
                let well = regex
                    .captures(name)
                    .and_then(|captures| captures.get(1))
                    .map(|m| m.as_str().to_string())
                    .unwrap_or_default();
                wells.entry(well).or_default().push(image.clone());
            }
            wells
                .into_iter()
                .map(|(well, images)| {
                    let suffix = if well.is_empty() {
                        "_".to_string()
                    } else {
                        naming::sanitize(&well)
                    };
                    all(images, Some(suffix))
                })
                .collect()
        }
    })
}

/// What a table's files are made of, worked out once.
struct Plan {
    rows: TableRows,
    unit: LengthUnit,
    /// The value columns, in the format's order (`Column`'s order).
    values: Vec<Column>,
    aggregations: Vec<Aggregation>,
    classes: Option<Vec<ObjectClass>>,
    db_classes: Vec<Class>,
    images: Vec<String>,
    planes: Vec<PlaneFilter>,
    several_z: bool,
    several_t: bool,
    empty_classes: bool,
    /// The classes side by side (`TableExport::transposed`; object and
    /// image rows).
    transposed: bool,
    /// Object rows with their track (`track_id` ticked): a `track_id`
    /// column after `object_id`.
    track_id: bool,
    /// Per class id its name part for the transposed column names.
    class_parts: BTreeMap<u32, String>,
}

/// A written column of a transposed table: what it is, where its cell is
/// in a query's row.
struct Placed {
    column: ValueColumn,
    at: usize,
}

impl Plan {
    fn new(
        results: &dyn ResultsSource,
        export: &ExportSettings,
        table: &TableExport,
        notes: &mut Vec<String>,
    ) -> Result<Self, InternalErrors> {
        let db_classes = results.get_object_classes()?;
        let grouped = matches!(
            table.rows,
            TableRows::Image | TableRows::Well | TableRows::Folder
        );
        let mut values: Vec<Column> = match &table.columns {
            None if grouped => DEFAULT_GROUPED_COLUMNS.to_vec(),
            None => DEFAULT_OBJECT_COLUMNS.to_vec(),
            Some(keys) => {
                let available = results.get_available_columns()?;
                let (columns, missing) = resolve_columns(keys, &db_classes, &available);
                if !missing.is_empty() {
                    notes.push(format!(
                        "these results have no column {}",
                        missing.join(", ")
                    ));
                }
                columns
            }
        };
        // The children and linked objects, as the Results List adds them:
        // per link its count and each value of each column, per parent class
        // its id and each column.
        if !table.children.is_empty() || !table.parents.is_empty() {
            let children: Vec<(ObjectClass, ObjectClass)> = table
                .children
                .iter()
                .map(|(p, c)| (ObjectClass::Global(*p), ObjectClass::Global(*c)))
                .collect();
            let child_aggregations: Vec<Aggregation> = table
                .child_aggregations
                .iter()
                .map(|a| aggregation(*a))
                .collect();
            let parents: Vec<ObjectClass> = table
                .parents
                .iter()
                .map(|p| ObjectClass::Global(*p))
                .collect();
            values = crate::results::view_settings::with_related_columns(
                &values,
                &children,
                &child_aggregations,
                &parents,
            );
        }
        let track_id = !grouped && values.contains(&Column::TrackId);
        // The leading columns are always there, in their place.
        values.retain(|column| !is_leading(column));
        if !grouped {
            values.retain(|column| *column != Column::Count);
        }
        values.sort();
        values.dedup();

        let images = scope_images(results, table.images)?;
        let mut planes = results.get_planes()?;
        planes.retain(|p| super::in_planes(&table.planes, p.z_stack, p.t_stack));
        if planes.is_empty() && table.planes == Default::default() {
            planes.push(PlaneFilter {
                z_stack: 0,
                t_stack: 0,
            });
        }
        Ok(Self {
            rows: table.rows,
            unit: export.length_unit,
            values,
            aggregations: table.aggregations.iter().map(|a| aggregation(*a)).collect(),
            classes: table
                .classes
                .as_ref()
                .map(|ids| ids.iter().map(|id| ObjectClass::Global(*id)).collect()),
            images,
            planes,
            several_z: results.get_nr_of_z_stacks() > 1,
            several_t: results.get_nr_of_t_stacks() > 1,
            empty_classes: table.empty_classes,
            transposed: table.transposed
                && matches!(
                    table.rows,
                    TableRows::Object | TableRows::Image | TableRows::Well | TableRows::Folder
                ),
            track_id,
            class_parts: naming::unique_parts(
                &db_classes
                    .iter()
                    // Column names are lower case (§3.3).
                    .map(|class| (class.id, class.name.to_lowercase()))
                    .collect::<Vec<_>>(),
            ),
            db_classes,
        })
    }

    /// The classes a transposed table puts side by side, in id order: those
    /// asked for (and known), else every class but Background (which no
    /// object ever gets). The query is asked for exactly these, so its
    /// blocks are these - every file of the export has the same columns,
    /// a class without objects included.
    fn blocks(&self, classes: Option<&[ObjectClass]>) -> Vec<u32> {
        let known = |id: &u32| self.db_classes.iter().any(|class| class.id == *id);
        let mut ids: Vec<u32> = match classes {
            Some(classes) => classes
                .iter()
                .filter_map(|class| match class {
                    ObjectClass::Global(id) => Some(*id),
                    _ => None,
                })
                .filter(known)
                .collect(),
            None => self
                .db_classes
                .iter()
                .map(|class| class.id)
                .filter(|id| ObjectClass::Global(*id) != ObjectClass::BACKGROUND)
                .collect(),
        };
        ids.sort_unstable();
        ids.dedup();
        ids
    }

    /// A transposed query's cells per class: for image rows a cell per
    /// value and aggregation, for object rows the id and the values (in
    /// `Column`'s order) - and the written columns among them.
    fn transposed_columns(&self, blocks: &[u32]) -> (usize, Vec<Placed>) {
        let block: Vec<ValueColumn> = if self.grouped() {
            self.value_columns()
        } else {
            let mut columns = self.list_columns();
            columns.retain(|column| *column != Column::ObjectClass);
            columns.sort();
            columns
                .iter()
                .map(|column| {
                    if *column == Column::ObjectId {
                        return ValueColumn {
                            name: "object_id".into(),
                            label: "Object".into(),
                            unit: String::new(),
                            factor: 1.0,
                            keep: true,
                        };
                    }
                    if *column == Column::TrackId {
                        return ValueColumn {
                            name: "track_id".into(),
                            label: "Track".into(),
                            unit: String::new(),
                            factor: 1.0,
                            keep: true,
                        };
                    }
                    let (name, factor) = column_name(column, self.unit, &self.db_classes);
                    ValueColumn {
                        name,
                        label: column.display_label(&self.db_classes),
                        unit: unit_label(column, self.unit),
                        factor,
                        keep: true,
                    }
                })
                .collect()
        };
        let per_block = block.len();
        let mut placed = Vec::new();
        for (b, id) in blocks.iter().enumerate() {
            let part = self
                .class_parts
                .get(id)
                .cloned()
                .unwrap_or_else(|| id.to_string());
            let class = self
                .db_classes
                .iter()
                .find(|class| class.id == *id)
                .map_or_else(|| id.to_string(), |class| class.name.clone());
            for (j, column) in block.iter().enumerate() {
                if !column.keep {
                    continue;
                }
                placed.push(Placed {
                    column: ValueColumn {
                        name: format!("{}_{part}", column.name),
                        label: format!("{} · {class}", column.label),
                        unit: column.unit.clone(),
                        factor: column.factor,
                        keep: true,
                    },
                    at: 1 + b * per_block + j,
                });
            }
        }
        (per_block, placed)
    }

    /// The leading columns of a transposed table: the image (well,
    /// folder), the plane.
    fn transposed_leading_header(&self) -> Vec<String> {
        let mut header = vec![
            match self.rows {
                TableRows::Well => "well",
                TableRows::Folder => "folder",
                _ => "image",
            }
            .to_string(),
        ];
        if self.several_t {
            header.push("t".into());
        }
        if self.several_z {
            header.push("z".into());
        }
        header
    }

    /// [`Plan::write_part`] with the classes side by side.
    fn write_part_transposed(
        &self,
        results: &dyn ResultsSource,
        part: &Part,
        sink: &mut dyn Sink,
        cancel: &AtomicBool,
    ) -> Result<(), InternalErrors> {
        let blocks = self.blocks(part.classes.as_deref());
        let (per_block, placed) = self.transposed_columns(&blocks);
        let header: Vec<Out> = self
            .transposed_leading_header()
            .into_iter()
            .chain(placed.iter().map(|p| p.column.name.clone()))
            .map(Out::Text)
            .collect();
        sink.row(&header)?;
        if part.images.is_empty() || blocks.is_empty() {
            return Ok(());
        }
        let classes: Option<Vec<ObjectClass>> =
            Some(blocks.iter().map(|id| ObjectClass::Global(*id)).collect());
        let mut line = Vec::new();
        for plane in &self.planes {
            let mut cursor: Option<String> = None;
            loop {
                if cancel.load(Ordering::Relaxed) {
                    return Err(InternalErrors::Cancelled);
                }
                let after = cursor.take();
                let page = if self.grouped() {
                    results.get_grouped_by_image(&GroupedByImageFilter {
                        plane: *plane,
                        images: Some(part.images.clone()),
                        object_classes: classes.clone(),
                        columns: self.values.clone(),
                        aggregation: self.aggregations.clone(),
                        page: Pagination {
                            limit: PAGE_SIZE,
                            after,
                        },
                        transpond_table: true,
                        grouping: self.grouping(),
                        empty_classes: true,
                        sort: None,
                    })?
                } else {
                    results.get_object_list(&ListFilter {
                        plane: *plane,
                        images: Some(part.images.clone()),
                        object_classes: classes.clone(),
                        columns: self.list_columns(),
                        with_coloc_details: false,
                        related_to: None,
                        page: Pagination {
                            limit: PAGE_SIZE,
                            after,
                        },
                        transpond_table: true,
                        sort: None,
                    })?
                };
                // The classes side by side as worked out here - else the
                // values would land under another class's names.
                if page.column_names.len() != 1 + blocks.len() * per_block {
                    return Err(InternalErrors::Internal(format!(
                        "the transposed table has {} columns, {} expected",
                        page.column_names.len(),
                        1 + blocks.len() * per_block
                    )));
                }
                for (i, row) in page.rows.iter().enumerate() {
                    line.clear();
                    // Wells and folders: the row's name (several images).
                    line.push(match self.grouping() {
                        Some(_) => row.first().map_or(Out::Empty, |cell| value(cell, 1.0)),
                        None => page
                            .row_locations
                            .get(i)
                            .map_or(Out::Empty, |(path, _)| Out::Text(path.replace('\\', "/"))),
                    });
                    if self.several_t {
                        line.push(Out::Number(f64::from(plane.t_stack)));
                    }
                    if self.several_z {
                        line.push(Out::Number(f64::from(plane.z_stack)));
                    }
                    for p in &placed {
                        line.push(
                            row.get(p.at)
                                .map_or(Out::Empty, |cell| value(cell, p.column.factor)),
                        );
                    }
                    sink.row(&line)?;
                }
                if page.source_object_count < PAGE_SIZE as usize {
                    break;
                }
                cursor = page.row_names.last().cloned();
                if cursor.is_none() {
                    break;
                }
            }
        }
        Ok(())
    }

    /// How well and folder rows are grouped (none for the others).
    fn grouping(&self) -> Option<Grouping> {
        match self.rows {
            TableRows::Well => Some(Grouping::Auto),
            TableRows::Folder => Some(Grouping::Folder),
            _ => None,
        }
    }

    fn grouped(&self) -> bool {
        matches!(
            self.rows,
            TableRows::Image | TableRows::Well | TableRows::Folder
        )
    }

    /// The leading columns' names: where the row belongs, then the plane.
    fn leading_header(&self) -> Vec<String> {
        let mut header: Vec<String> = match self.rows {
            TableRows::Image => vec!["image".into(), "class".into()],
            TableRows::Well => vec!["well".into(), "class".into()],
            TableRows::Folder => vec!["folder".into(), "class".into()],
            _ if self.track_id => vec![
                "image".into(),
                "class".into(),
                "object_id".into(),
                "track_id".into(),
            ],
            _ => vec!["image".into(), "class".into(), "object_id".into()],
        };
        if self.several_t {
            header.push("t".into());
        }
        if self.several_z {
            header.push("z".into());
        }
        header
    }

    /// The value columns as written: for grouped rows a column per value
    /// and aggregation - where the count is the same for every aggregation
    /// (images), only the first is kept.
    fn value_columns(&self) -> Vec<ValueColumn> {
        let mut out = Vec::new();
        for column in &self.values {
            let (name, factor) = column_name(column, self.unit, &self.db_classes);
            let label = column.display_label(&self.db_classes);
            let unit = unit_label(column, self.unit);
            if !self.grouped() {
                out.push(ValueColumn {
                    name,
                    label,
                    unit,
                    factor,
                    keep: true,
                });
                continue;
            }
            for (i, aggregation) in self.aggregations.iter().enumerate() {
                let suffix = aggregation_suffix(aggregation);
                let (name, label) = match (column, self.rows) {
                    (Column::Count, TableRows::Image) => ("count".to_string(), label.clone()),
                    (Column::Count, _) => (
                        format!("count_per_image_{suffix}"),
                        format!("{label} per image ({suffix})"),
                    ),
                    _ => (format!("{name}_{suffix}"), format!("{label} ({suffix})")),
                };
                let keep = !(*column == Column::Count && self.rows == TableRows::Image && i > 0);
                out.push(ValueColumn {
                    name,
                    label,
                    unit: unit.clone(),
                    factor,
                    keep,
                });
            }
        }
        out
    }

    /// The rows of `part`, plane by plane, page by page, into `sink`.
    fn write_part(
        &self,
        results: &dyn ResultsSource,
        part: &Part,
        sink: &mut dyn Sink,
        cancel: &AtomicBool,
    ) -> Result<(), InternalErrors> {
        if self.transposed {
            return self.write_part_transposed(results, part, sink, cancel);
        }
        let values = self.value_columns();
        let header: Vec<Out> = self
            .leading_header()
            .into_iter()
            .chain(values.iter().filter(|v| v.keep).map(|v| v.name.clone()))
            .map(Out::Text)
            .collect();
        sink.row(&header)?;
        if part.images.is_empty() {
            return Ok(());
        }
        let layout = self.layout();
        let mut line = Vec::new();
        for plane in &self.planes {
            let mut cursor: Option<String> = None;
            loop {
                if cancel.load(Ordering::Relaxed) {
                    return Err(InternalErrors::Cancelled);
                }
                let after = cursor.take();
                let page = if self.grouped() {
                    results.get_grouped_by_image(&GroupedByImageFilter {
                        plane: *plane,
                        images: Some(part.images.clone()),
                        object_classes: part.classes.clone(),
                        columns: self.values.clone(),
                        aggregation: self.aggregations.clone(),
                        page: Pagination {
                            limit: PAGE_SIZE,
                            after,
                        },
                        transpond_table: false,
                        grouping: self.grouping(),
                        empty_classes: self.empty_classes,
                        sort: None,
                    })?
                } else {
                    results.get_object_list(&ListFilter {
                        plane: *plane,
                        images: Some(part.images.clone()),
                        object_classes: part.classes.clone(),
                        columns: self.list_columns(),
                        with_coloc_details: self.rows == TableRows::ColocPair,
                        related_to: None,
                        page: Pagination {
                            limit: PAGE_SIZE,
                            after,
                        },
                        transpond_table: false,
                        sort: None,
                    })?
                };
                for (i, row) in page.rows.iter().enumerate() {
                    line.clear();
                    self.leading_cells(&layout, &page.row_locations, i, row, *plane, &mut line);
                    for (v, at) in values.iter().zip(&layout.values) {
                        if v.keep {
                            line.push(
                                row.get(*at)
                                    .map_or(Out::Empty, |cell| value(cell, v.factor)),
                            );
                        }
                    }
                    // Coloc details: the partners' values after the columns.
                    for cell in row.iter().skip(layout.extras_from) {
                        line.push(value(cell, 1.0));
                    }
                    sink.row(&line)?;
                }
                if page.source_object_count < PAGE_SIZE as usize {
                    break;
                }
                cursor = page.row_names.last().cloned();
                if cursor.is_none() {
                    break;
                }
            }
        }
        Ok(())
    }

    /// Where in a query's row the leading cells and each value column are:
    /// the queries return their columns in `Column`'s order.
    fn layout(&self) -> Layout {
        if self.grouped() {
            // Place and class first, then a cell per value and aggregation.
            let count = self.values.len() * self.aggregations.len();
            return Layout {
                class: 1,
                id: None,
                track: None,
                values: (2..2 + count).collect(),
                extras_from: 2 + count,
            };
        }
        let mut sorted = self.list_columns();
        sorted.sort();
        let at = |wanted: &Column| sorted.iter().position(|c| c == wanted).unwrap_or(0);
        Layout {
            class: at(&Column::ObjectClass),
            id: Some(at(&Column::ObjectId)),
            track: self.track_id.then(|| at(&Column::TrackId)),
            values: self.values.iter().map(at).collect(),
            extras_from: sorted.len(),
        }
    }

    /// The object list's columns: the class and the id first (the image is
    /// the row's location), then the values - the query returns them in
    /// `Column`'s order, which is why the class and id are asked for in a
    /// fixed place (see `leading_cells`).
    fn list_columns(&self) -> Vec<Column> {
        let mut columns = vec![Column::ObjectClass, Column::ObjectId];
        if self.track_id {
            columns.push(Column::TrackId);
        }
        columns.extend(self.values.iter().cloned());
        columns
    }

    fn leading_cells(
        &self,
        layout: &Layout,
        locations: &[(String, Option<[u32; 4]>)],
        i: usize,
        row: &[Cell],
        plane: PlaneFilter,
        line: &mut Vec<Out>,
    ) {
        let cell = |at: usize| row.get(at).map_or(Out::Empty, |cell| value(cell, 1.0));
        if self.grouped() {
            let place = match (row.first(), self.rows) {
                (Some(first), TableRows::Image) => first
                    .search_key
                    .as_ref()
                    .map(|(_, path)| Out::Text(path.replace('\\', "/")))
                    .unwrap_or_else(|| value(first, 1.0)),
                (Some(first), _) => value(first, 1.0),
                (None, _) => Out::Empty,
            };
            line.push(place);
            line.push(cell(layout.class));
        } else {
            line.push(
                locations
                    .get(i)
                    .map_or(Out::Empty, |(path, _)| Out::Text(path.replace('\\', "/"))),
            );
            line.push(cell(layout.class));
            line.push(layout.id.map_or(Out::Empty, cell));
            if let Some(track) = layout.track {
                line.push(cell(track));
            }
        }
        if self.several_t {
            line.push(Out::Number(f64::from(plane.t_stack)));
        }
        if self.several_z {
            line.push(Out::Number(f64::from(plane.z_stack)));
        }
    }

    /// The `About` sheet's rows: each column's name, label and unit.
    fn about(&self) -> Vec<[String; 3]> {
        let mut rows = vec![[
            "column".to_string(),
            "label".to_string(),
            "unit".to_string(),
        ]];
        let leading = if self.transposed {
            self.transposed_leading_header()
        } else {
            self.leading_header()
        };
        for name in leading {
            let label = match name.as_str() {
                "image" => "Image (relative path)",
                "class" => "Class",
                "object_id" => "Object",
                "track_id" => "Track",
                "well" => "Well",
                "folder" => "Folder",
                "t" => "Time point (from 0)",
                "z" => "Z-plane (from 0)",
                _ => "",
            };
            rows.push([name, label.to_string(), String::new()]);
        }
        if self.transposed {
            let blocks = self.blocks(self.classes.as_deref());
            for placed in self.transposed_columns(&blocks).1 {
                let column = placed.column;
                rows.push([column.name, column.label, column.unit]);
            }
            return rows;
        }
        for column in self.value_columns().into_iter().filter(|v| v.keep) {
            rows.push([column.name, column.label, column.unit]);
        }
        rows
    }
}

/// A value column as written.
struct ValueColumn {
    name: String,
    label: String,
    unit: String,
    factor: f64,
    keep: bool,
}

/// See [`Plan::layout`].
struct Layout {
    class: usize,
    id: Option<usize>,
    /// The track's cell (`Plan::track_id`).
    track: Option<usize>,
    /// Per value column (and aggregation), its cell.
    values: Vec<usize>,
    /// The first cell after the asked-for columns (coloc details).
    extras_from: usize,
}

/// Columns that are always there, first - never a value column.
pub(crate) fn is_leading(column: &Column) -> bool {
    matches!(
        column,
        Column::ImageName | Column::ObjectClass | Column::ObjectId | Column::TrackId
    )
}

fn aggregation(aggregation: ExportAggregation) -> Aggregation {
    match aggregation {
        ExportAggregation::Avg => Aggregation::Avg,
        ExportAggregation::Median => Aggregation::Median,
        ExportAggregation::Stddev => Aggregation::Stddev,
        ExportAggregation::Sum => Aggregation::Sum,
        ExportAggregation::Min => Aggregation::Min,
        ExportAggregation::Max => Aggregation::Max,
        ExportAggregation::Skewness => Aggregation::Skewness,
    }
}

fn aggregation_suffix(aggregation: &Aggregation) -> &'static str {
    match aggregation {
        Aggregation::Avg => "avg",
        Aggregation::Median => "median",
        Aggregation::Stddev => "stddev",
        Aggregation::Sum => "sum",
        Aggregation::Min => "min",
        Aggregation::Max => "max",
        Aggregation::Skewness => "skewness",
    }
}

/// What kind of length a column holds (also through a child's or parent's
/// value) - the results store them in nm.
#[derive(Clone, Copy, PartialEq)]
enum Length {
    Line,
    Area,
    Volume,
}

fn length_of(column: &Column) -> Option<Length> {
    match column {
        Column::PerimeterNm => Some(Length::Line),
        Column::AreaSizeNm | Column::SurfaceAreaNm => Some(Length::Area),
        Column::VolumeNm => Some(Length::Volume),
        Column::Children(child) => match &child.value {
            ChildValue::Of(_, inner) => length_of(inner),
            ChildValue::Count => None,
        },
        Column::Parent(parent) => match &parent.value {
            ParentValue::Of(inner) => length_of(inner),
            ParentValue::Id => None,
        },
        _ => None,
    }
}

/// A column's format 1 name and the factor its values are multiplied by:
/// its key with anything but `a-z`, `0-9` and `_` turned into `_`, lengths
/// in the export's unit.
pub(super) fn column_name(column: &Column, unit: LengthUnit, classes: &[Class]) -> (String, f64) {
    let mut name: String = column
        .as_key(classes)
        .to_lowercase()
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '_' {
                c
            } else {
                '_'
            }
        })
        .collect();
    let Some(length) = length_of(column) else {
        return (name, 1.0);
    };
    if unit == LengthUnit::Nm {
        return (name, 1.0);
    }
    let (from, to, factor) = match length {
        Length::Line => ("perimeter_nm", "perimeter_um", 1e-3),
        Length::Area => ("_nm2", "_um2", 1e-6),
        Length::Volume => ("_nm3", "_um3", 1e-9),
    };
    if let Some(at) = name.rfind(from) {
        name.replace_range(at..at + from.len(), to);
    }
    (name, factor)
}

/// The unit of a column's values for the About sheet.
pub(super) fn unit_label(column: &Column, unit: LengthUnit) -> String {
    let base = match unit {
        LengthUnit::Um => "µm",
        LengthUnit::Nm => "nm",
    };
    match length_of(column) {
        Some(Length::Line) => base.to_string(),
        Some(Length::Area) => format!("{base}²"),
        Some(Length::Volume) => format!("{base}³"),
        None => {
            let key = column.as_key(&[]);
            if key.ends_with("_px") {
                "px".into()
            } else if key.contains("_gray") {
                "gray value".into()
            } else {
                String::new()
            }
        }
    }
}

/// One written value.
#[derive(Clone, Debug, PartialEq)]
pub(super) enum Out {
    Empty,
    Text(String),
    Number(f64),
}

pub(super) fn value(cell: &Cell, factor: f64) -> Out {
    match &cell.value {
        CellValue::Empty => Out::Empty,
        CellValue::String(text) => Out::Text(text.clone()),
        CellValue::Class((name, _)) => Out::Text(name.clone()),
        CellValue::Float(v) if v.is_finite() => Out::Number(f64::from(*v) * factor),
        CellValue::Float(_) => Out::Empty,
        CellValue::Integer(v) => Out::Number(f64::from(*v) * factor),
    }
}

/// Where a table's rows go.
pub(super) trait Sink {
    fn row(&mut self, cells: &[Out]) -> Result<(), InternalErrors>;
}

/// A CSV file: UTF-8, `,`, `.` decimals, `"` quoting where needed, `\n`.
pub(super) struct CsvSink {
    out: std::io::BufWriter<std::fs::File>,
    path: PathBuf,
    line: Vec<u8>,
}

impl CsvSink {
    pub(super) fn create(path: &Path) -> Result<Self, InternalErrors> {
        let file = std::fs::File::create(path)
            .map_err(|e| InternalErrors::Io(format!("Could not create {}: {e}", path.display())))?;
        Ok(Self {
            out: std::io::BufWriter::new(file),
            path: path.to_path_buf(),
            line: Vec::new(),
        })
    }

    pub(super) fn finish(mut self) -> Result<(), InternalErrors> {
        self.out.flush().map_err(|e| {
            InternalErrors::Io(format!("Could not write {}: {e}", self.path.display()))
        })
    }
}

impl Sink for CsvSink {
    fn row(&mut self, cells: &[Out]) -> Result<(), InternalErrors> {
        self.line.clear();
        for (i, cell) in cells.iter().enumerate() {
            if i > 0 {
                self.line.push(b',');
            }
            match cell {
                Out::Empty => {}
                Out::Number(v) => self.line.extend_from_slice(number_text(*v).as_bytes()),
                Out::Text(text) => {
                    if text.contains([',', '"', '\n', '\r']) {
                        self.line.push(b'"');
                        self.line
                            .extend_from_slice(text.replace('"', "\"\"").as_bytes());
                        self.line.push(b'"');
                    } else {
                        self.line.extend_from_slice(text.as_bytes());
                    }
                }
            }
        }
        self.line.push(b'\n');
        self.out.write_all(&self.line).map_err(|e| {
            InternalErrors::Io(format!("Could not write {}: {e}", self.path.display()))
        })
    }
}

/// A number as written: whole numbers without a decimal point, others as
/// short as they read back the same, never with thousands separators.
fn number_text(v: f64) -> String {
    if v.fract() == 0.0 && v.abs() < 1e15 {
        format!("{}", v as i64)
    } else {
        // f32 is what the results hold: printed at its precision, not as
        // a long f64 tail (0.1f32 is 0.10000000149011612 as f64).
        format!("{}", v as f32)
    }
}

/// A sheet of a workbook, written top to bottom.
struct SheetSink<'a> {
    sheet: &'a mut Worksheet,
    row: u32,
}

impl Sink for SheetSink<'_> {
    fn row(&mut self, cells: &[Out]) -> Result<(), InternalErrors> {
        for (col, cell) in cells.iter().enumerate() {
            let col = col as u16;
            match cell {
                Out::Empty => {}
                Out::Number(v) => {
                    self.sheet
                        .write_number(self.row, col, *v)
                        .map_err(xlsx_error)?;
                }
                Out::Text(text) => {
                    self.sheet
                        .write_string(self.row, col, text)
                        .map_err(xlsx_error)?;
                }
            }
        }
        self.row += 1;
        Ok(())
    }
}

fn xlsx_error(e: rust_xlsxwriter::XlsxError) -> InternalErrors {
    InternalErrors::Io(format!("Could not write the workbook: {e}"))
}

/// A workbook of `tables`: a sheet each, named by it.
pub(super) fn write_sheets(
    path: &Path,
    tables: &[(String, Vec<Vec<Out>>)],
) -> Result<(), InternalErrors> {
    let mut workbook = Workbook::new();
    let mut used: Vec<String> = Vec::new();
    for (name, rows) in tables {
        let sheet = workbook.add_worksheet();
        sheet
            .set_name(sheet_name(name, &mut used))
            .map_err(xlsx_error)?;
        let mut sink = SheetSink { sheet, row: 0 };
        for row in rows {
            sink.row(row)?;
        }
    }
    workbook
        .save(path)
        .map_err(|e| InternalErrors::Io(format!("Could not write {}: {e}", path.display())))
}

/// A workbook with a sheet per part and the `About` sheet.
fn write_workbook(
    path: &Path,
    rows: &dyn Rows,
    results: &dyn ResultsSource,
    parts: &[Part],
    cancel: &AtomicBool,
) -> Result<(), InternalErrors> {
    let mut workbook = Workbook::new();
    let mut used: Vec<String> = Vec::new();
    for part in parts {
        let sheet = workbook
            .add_worksheet_with_constant_memory()
            .map_err(xlsx_error)?;
        sheet
            .set_name(sheet_name(&part.sheet_name(), &mut used))
            .map_err(xlsx_error)?;
        rows.write_part(results, part, &mut SheetSink { sheet, row: 0 }, cancel)?;
    }
    let about = workbook.add_worksheet();
    about.set_name("About").map_err(xlsx_error)?;
    for (row, cells) in rows.about().iter().enumerate() {
        for (col, text) in cells.iter().enumerate() {
            about
                .write_string(row as u32, col as u16, text)
                .map_err(xlsx_error)?;
        }
    }
    workbook
        .save(path)
        .map_err(|e| InternalErrors::Io(format!("Could not write {}: {e}", path.display())))
}

/// A sheet name Excel takes (at most 31 characters, none of `[]:*?/\`),
/// unique in its workbook and never `About`.
fn sheet_name(wanted: &str, used: &mut Vec<String>) -> String {
    let clean: String = wanted
        .chars()
        .map(|c| if "[]:*?/\\".contains(c) { '_' } else { c })
        .collect();
    let base: String = clean.chars().take(31).collect();
    let mut name = base.clone();
    let mut n = 2;
    while used.iter().any(|u| u.eq_ignore_ascii_case(&name)) || name.eq_ignore_ascii_case("About") {
        let suffix = format!("_{n}");
        name = format!(
            "{}{suffix}",
            base.chars().take(31 - suffix.len()).collect::<String>()
        );
        n += 1;
    }
    used.push(name.clone());
    name
}

/// The images of a run in `scope` (relative paths, `/` separated).
pub(super) fn scope_images(
    results: &dyn ResultsSource,
    scope: ImageScope,
) -> Result<Vec<String>, InternalErrors> {
    Ok(results
        .get_images()?
        .into_iter()
        .filter(|image| match scope {
            ImageScope::All => true,
            ImageScope::Included => !image.disabled,
            ImageScope::LeftOut => image.disabled,
        })
        .map(|image| image.rel_path.to_string_lossy().replace('\\', "/"))
        .collect())
}

impl Rows for Plan {
    fn images(&self) -> &[String] {
        &self.images
    }

    fn classes(&self) -> Option<Vec<ObjectClass>> {
        self.classes.clone()
    }

    fn write_part(
        &self,
        results: &dyn ResultsSource,
        part: &Part,
        sink: &mut dyn Sink,
        cancel: &AtomicBool,
    ) -> Result<(), InternalErrors> {
        Plan::write_part(self, results, part, sink, cancel)
    }

    fn about(&self) -> Vec<[String; 3]> {
        Plan::about(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lengths_are_named_and_converted_in_the_export_unit() {
        let (name, factor) = column_name(&Column::AreaSizeNm, LengthUnit::Um, &[]);
        assert_eq!((name.as_str(), factor), ("area_um2", 1e-6));
        let (name, factor) = column_name(&Column::AreaSizeNm, LengthUnit::Nm, &[]);
        assert_eq!((name.as_str(), factor), ("area_nm2", 1.0));
        assert_eq!(
            column_name(&Column::PerimeterNm, LengthUnit::Um, &[]).0,
            "perimeter_um"
        );
        assert_eq!(
            column_name(&Column::VolumeNm, LengthUnit::Um, &[]).0,
            "volume_um3"
        );
        assert_eq!(
            column_name(&Column::SurfaceAreaNm, LengthUnit::Um, &[]).0,
            "surface_area_um2"
        );
        assert_eq!(
            column_name(&Column::AreaSizePx, LengthUnit::Um, &[]),
            ("area_px".into(), 1.0)
        );
        assert_eq!(
            column_name(&Column::IntensityAvg(1), LengthUnit::Um, &[]).0,
            "intensity_mean_gray_ch1"
        );
    }

    #[test]
    fn numbers_are_written_plainly() {
        assert_eq!(number_text(3.0), "3");
        assert_eq!(number_text(0.1f32 as f64), "0.1");
        assert_eq!(number_text(-2.5), "-2.5");
        assert_eq!(number_text(1234567.0), "1234567");
    }

    #[test]
    fn sheet_names_fit_excel_and_stay_unique() {
        let mut used = Vec::new();
        assert_eq!(sheet_name("a/b", &mut used), "a_b");
        assert_eq!(sheet_name("a:b", &mut used), "a_b_2");
        assert_eq!(sheet_name("About", &mut used), "About_2");
        assert_eq!(sheet_name(&"x".repeat(40), &mut used).len(), 31);
    }
}
