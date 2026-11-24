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
use crate::exports::naming;
use crate::exports::types::EntryReport;
use crate::results::view_settings::resolve_columns;
use crate::results::{
    Aggregation, Cell, CellValue, ChildValue, Column, GroupedByImageFilter, Grouping, ListFilter,
    Pagination, ParentValue, PlaneFilter, ResultsSource,
};
use evanalyzer_cfg::core_types::{InternalErrors, ObjectClass};
use evanalyzer_cfg::settings::classification_settings::Class;
use evanalyzer_cfg::settings::export_settings::{
    ExportAggregation, ExportSettings, ImageScope, LengthUnit, SplitInto, TableExport,
    TableFileFormat, TableRows, TableSplit,
};
use rust_xlsxwriter::{Workbook, Worksheet};
use std::collections::BTreeMap;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};

/// Rows per query page.
const PAGE_SIZE: i32 = 20_000;

/// The value columns a table has without `columns`.
const DEFAULT_OBJECT_COLUMNS: [Column; 3] =
    [Column::AreaSizePx, Column::AreaSizeNm, Column::Circularity];
const DEFAULT_GROUPED_COLUMNS: [Column; 3] =
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
    let plan = Plan::new(results, export, table, &mut report.notes)?;
    let name = naming::with_prefix(
        export,
        &naming::entry_name(
            &evanalyzer_cfg::settings::export_settings::ExportEntry::Table(table.clone()),
        )
        .unwrap_or_default(),
    );
    let parts = parts(results, table, &plan)?;

    match (table.format, table.split_into) {
        (TableFileFormat::Csv, _) => {
            for part in &parts {
                let file = output.file(&format!("{}.csv", part.file_name(&name)));
                let path = output.path(&file);
                let mut sink = CsvSink::create(&path)?;
                plan.write_part(results, part, &mut sink, cancel)?;
                sink.finish()?;
                report.files.push(file);
            }
        }
        (TableFileFormat::Xlsx, SplitInto::Files) => {
            for part in &parts {
                let file = output.file(&format!("{}.xlsx", part.file_name(&name)));
                write_workbook(
                    &output.path(&file),
                    &plan,
                    results,
                    std::slice::from_ref(part),
                    cancel,
                )?;
                report.files.push(file);
            }
        }
        (TableFileFormat::Xlsx, SplitInto::Sheets) => {
            let file = output.file(&format!("{name}.xlsx"));
            write_workbook(&output.path(&file), &plan, results, &parts, cancel)?;
            report.files.push(file);
        }
    }
    Ok(())
}

/// One file (or sheet) of a table: its name part and what it covers.
struct Part {
    /// `None`: the table isn't split.
    suffix: Option<String>,
    images: Vec<String>,
    classes: Option<Vec<ObjectClass>>,
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
    plan: &Plan,
) -> Result<Vec<Part>, InternalErrors> {
    let all = |images: Vec<String>, suffix| Part {
        suffix,
        images,
        classes: plan.classes.clone(),
    };
    Ok(match table.split {
        TableSplit::None => vec![all(plan.images.clone(), None)],
        TableSplit::Image => {
            let names = naming::unique_image_parts(&plan.images);
            plan.images
                .iter()
                .map(|image| all(vec![image.clone()], Some(names[image].clone())))
                .collect()
        }
        TableSplit::Folder => {
            let mut folders: BTreeMap<String, Vec<String>> = BTreeMap::new();
            for image in &plan.images {
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
                    plan.classes
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
                    images: plan.images.clone(),
                    classes: Some(vec![ObjectClass::Global(*id)]),
                })
                .collect()
        }
        TableSplit::Well => {
            return Err(InternalErrors::InvalidArgument(
                "a file per well isn't possible yet".into(),
            ));
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
}

impl Plan {
    fn new(
        results: &dyn ResultsSource,
        export: &ExportSettings,
        table: &TableExport,
        notes: &mut Vec<String>,
    ) -> Result<Self, InternalErrors> {
        if matches!(table.rows, TableRows::Track | TableRows::Measurement) {
            return Err(InternalErrors::InvalidArgument(format!(
                "tables of {} aren't possible yet",
                match table.rows {
                    TableRows::Track => "tracks",
                    _ => "measurements",
                }
            )));
        }
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
        // The leading columns are always there, in their place.
        values.retain(|column| !is_leading(column));
        if !grouped {
            values.retain(|column| *column != Column::Count);
        }
        values.sort();
        values.dedup();

        let images: Vec<String> = results
            .get_images()?
            .into_iter()
            .filter(|image| match table.images {
                ImageScope::All => true,
                ImageScope::Included => !image.disabled,
                ImageScope::LeftOut => image.disabled,
            })
            .map(|image| image.rel_path.to_string_lossy().replace('\\', "/"))
            .collect();
        let mut planes = results.get_planes()?;
        if planes.is_empty() {
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
            db_classes,
            images,
            planes,
            several_z: results.get_nr_of_z_stacks() > 1,
            several_t: results.get_nr_of_t_stacks() > 1,
            empty_classes: table.empty_classes,
        })
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
                        grouping: match self.rows {
                            TableRows::Well => Some(Grouping::Auto),
                            TableRows::Folder => Some(Grouping::Folder),
                            _ => None,
                        },
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
        for name in self.leading_header() {
            let label = match name.as_str() {
                "image" => "Image (relative path)",
                "class" => "Class",
                "object_id" => "Object",
                "well" => "Well",
                "folder" => "Folder",
                "t" => "Time point (from 0)",
                "z" => "Z-plane (from 0)",
                _ => "",
            };
            rows.push([name, label.to_string(), String::new()]);
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
    /// Per value column (and aggregation), its cell.
    values: Vec<usize>,
    /// The first cell after the asked-for columns (coloc details).
    extras_from: usize,
}

/// Columns that are always there, first - never a value column.
fn is_leading(column: &Column) -> bool {
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
fn column_name(column: &Column, unit: LengthUnit, classes: &[Class]) -> (String, f64) {
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
fn unit_label(column: &Column, unit: LengthUnit) -> String {
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
enum Out {
    Empty,
    Text(String),
    Number(f64),
}

fn value(cell: &Cell, factor: f64) -> Out {
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
trait Sink {
    fn row(&mut self, cells: &[Out]) -> Result<(), InternalErrors>;
}

/// A CSV file: UTF-8, `,`, `.` decimals, `"` quoting where needed, `\n`.
struct CsvSink {
    out: std::io::BufWriter<std::fs::File>,
    path: PathBuf,
    line: Vec<u8>,
}

impl CsvSink {
    fn create(path: &Path) -> Result<Self, InternalErrors> {
        let file = std::fs::File::create(path)
            .map_err(|e| InternalErrors::Io(format!("Could not create {}: {e}", path.display())))?;
        Ok(Self {
            out: std::io::BufWriter::new(file),
            path: path.to_path_buf(),
            line: Vec::new(),
        })
    }

    fn finish(mut self) -> Result<(), InternalErrors> {
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

/// A workbook with a sheet per part and the `About` sheet.
fn write_workbook(
    path: &Path,
    plan: &Plan,
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
        plan.write_part(results, part, &mut SheetSink { sheet, row: 0 }, cancel)?;
    }
    let about = workbook.add_worksheet();
    about.set_name("About").map_err(xlsx_error)?;
    for (row, cells) in plan.about().iter().enumerate() {
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
