//! # format
//!
//! Writing CSV and XLSX: fields and cells, number formats, sheet names,
//! files moved into place; text that would run as a formula is kept text.
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-11-22
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.

use super::*;

/// A time course table's cell.
#[derive(Debug, Clone, PartialEq)]
pub(super) enum Field {
    Text(String),
    Number(f64),
    Empty,
}

/// A table: its header, then its rows.
pub(super) type Fields = Vec<Vec<Field>>;

/// Where a table's rows go, a field at a time - nothing per row allocated:
/// a per-track time course has millions of rows.
pub(super) trait FieldSink {
    fn text(&mut self, text: &str) -> Result<(), InternalErrors>;
    fn number(&mut self, number: f64) -> Result<(), InternalErrors>;
    fn empty(&mut self) -> Result<(), InternalErrors>;
    fn end_row(&mut self) -> Result<(), InternalErrors>;

    fn optional(&mut self, value: Option<f64>) -> Result<(), InternalErrors> {
        match value {
            Some(number) => self.number(number),
            None => self.empty(),
        }
    }

    fn row(&mut self, row: &[Field]) -> Result<(), InternalErrors> {
        for field in row {
            match field {
                Field::Text(text) => self.text(text)?,
                Field::Number(number) => self.number(*number)?,
                Field::Empty => self.empty()?,
            }
        }
        self.end_row()
    }
}

/// Collects one row's fields.
impl FieldSink for Vec<Field> {
    fn text(&mut self, text: &str) -> Result<(), InternalErrors> {
        self.push(Field::Text(text.to_string()));
        Ok(())
    }
    fn number(&mut self, number: f64) -> Result<(), InternalErrors> {
        self.push(Field::Number(number));
        Ok(())
    }
    fn empty(&mut self) -> Result<(), InternalErrors> {
        self.push(Field::Empty);
        Ok(())
    }
    fn end_row(&mut self) -> Result<(), InternalErrors> {
        Ok(())
    }
}

/// Takes rows and drops them - nothing to write.
pub(super) struct Discard;

impl FieldSink for Discard {
    fn text(&mut self, _: &str) -> Result<(), InternalErrors> {
        Ok(())
    }
    fn number(&mut self, _: f64) -> Result<(), InternalErrors> {
        Ok(())
    }
    fn empty(&mut self) -> Result<(), InternalErrors> {
        Ok(())
    }
    fn end_row(&mut self) -> Result<(), InternalErrors> {
        Ok(())
    }
}

pub(super) fn write_fields_csv(table: &Fields, path: &Path) -> Result<(), InternalErrors> {
    let mut out = FieldsCsv::create(path)?;
    table.iter().try_for_each(|row| out.row(row))
}

/// A CSV file written a field at a time: into one line buffer, reused.
pub(super) struct FieldsCsv<'a> {
    pub(super) out: std::io::BufWriter<std::fs::File>,
    pub(super) path: &'a Path,
    pub(super) line: Vec<u8>,
    /// Fields in `line` so far.
    pub(super) fields: usize,
}

impl<'a> FieldsCsv<'a> {
    pub(super) fn create(path: &'a Path) -> Result<Self, InternalErrors> {
        Ok(Self {
            out: create_csv_writer(path)?,
            path,
            line: Vec::new(),
            fields: 0,
        })
    }

    pub(super) fn separate(&mut self) {
        if self.fields > 0 {
            self.line.push(b',');
        }
        self.fields += 1;
    }
}

impl FieldSink for FieldsCsv<'_> {
    fn text(&mut self, text: &str) -> Result<(), InternalErrors> {
        self.separate();
        write_csv_field(&mut self.line, text).expect("writing to memory");
        Ok(())
    }

    fn number(&mut self, number: f64) -> Result<(), InternalErrors> {
        use std::io::Write;
        self.separate();
        // Whole numbers (T, counts) as integers - the same text as `f64`'s
        // own, written several times faster.
        match number.fract() == 0.0 && number.abs() < 1e15 && !number.is_sign_negative() {
            true => write!(self.line, "{}", number as u64).expect("writing to memory"),
            false => write_f64(&mut self.line, number),
        }
        Ok(())
    }

    fn empty(&mut self) -> Result<(), InternalErrors> {
        self.separate();
        Ok(())
    }

    fn end_row(&mut self) -> Result<(), InternalErrors> {
        use std::io::Write;
        self.line.push(b'\n');
        let written = self.out.write_all(&self.line);
        self.line.clear();
        self.fields = 0;
        written.map_err(|e| {
            InternalErrors::Internal(format!("could not write {}: {e}", self.path.display()))
        })
    }
}

/// `number` like `f64` prints it - the shortest text that reads back the
/// same, no exponent - through `zmij`, several times faster. Where 17 digits
/// are needed and two last digits both read back, `zmij` takes the closer
/// one, `f64` not always: the value is the same. Where `zmij` would use an
/// exponent (very small or large values), as `f64` prints it.
pub(super) fn write_f64(out: &mut Vec<u8>, number: f64) {
    write_float(out, number, number.is_finite());
}

/// `write_f64` for an `f64` or `f32` (`finite`: its `is_finite()`).
pub(super) fn write_float(
    out: &mut Vec<u8>,
    number: impl zmij::Float + std::fmt::Display,
    finite: bool,
) {
    use std::io::Write;
    let mut buffer = zmij::Buffer::new();
    let text = match finite {
        true => buffer.format_finite(number),
        false => "e",
    };
    match text.contains('e') {
        true => write!(out, "{number}").expect("writing to memory"),
        false => out.extend_from_slice(text.strip_suffix(".0").unwrap_or(text).as_bytes()),
    }
}

pub(super) fn write_fields_sheet(
    worksheet: &mut Worksheet,
    table: &Fields,
) -> Result<(), InternalErrors> {
    let mut sheet = FieldsSheet::new(worksheet);
    table.iter().try_for_each(|row| sheet.row(row))
}

/// A worksheet written a field at a time, the first row in bold.
pub(super) struct FieldsSheet<'a> {
    pub(super) worksheet: &'a mut Worksheet,
    pub(super) header_format: Format,
    pub(super) format: Format,
    pub(super) next_row: usize,
    pub(super) next_column: u16,
}

impl<'a> FieldsSheet<'a> {
    pub(super) fn new(worksheet: &'a mut Worksheet) -> Self {
        Self {
            worksheet,
            header_format: Format::new().set_bold().set_font_size(EXPORT_FONT_SIZE),
            format: Format::new().set_font_size(EXPORT_FONT_SIZE),
            next_row: 0,
            next_column: 0,
        }
    }

    /// The next cell: its row and column - an error beyond the last row.
    pub(super) fn cell(&mut self) -> Result<(u32, u16), InternalErrors> {
        if self.next_row >= XLSX_MAX_ROWS {
            return Err(InternalErrors::InvalidArgument(format!(
                "More than {} rows - more than an XLSX sheet holds: export it as CSV",
                XLSX_MAX_ROWS - 1
            )));
        }
        self.next_column += 1;
        Ok((self.next_row as u32, self.next_column - 1))
    }
}

impl FieldSink for FieldsSheet<'_> {
    fn text(&mut self, text: &str) -> Result<(), InternalErrors> {
        let (r, c) = self.cell()?;
        let format = match r {
            0 => &self.header_format,
            _ => &self.format,
        };
        self.worksheet
            .write_with_format(r, c, text, format)
            .map_err(xlsx_err)?;
        Ok(())
    }

    fn number(&mut self, number: f64) -> Result<(), InternalErrors> {
        let (r, c) = self.cell()?;
        self.worksheet
            .write_with_format(r, c, number, &self.format)
            .map_err(xlsx_err)?;
        Ok(())
    }

    fn empty(&mut self) -> Result<(), InternalErrors> {
        self.cell()?;
        Ok(())
    }

    fn end_row(&mut self) -> Result<(), InternalErrors> {
        self.next_row += 1;
        self.next_column = 0;
        Ok(())
    }
}

pub(super) fn plain_cell(value: CellValue) -> Cell {
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

/// `from` to `to`: renamed on the same disk, copied (then removed) to
/// another one.
pub(super) fn move_file(from: &Path, to: &Path) -> Result<(), InternalErrors> {
    let err =
        |e: std::io::Error| InternalErrors::Io(format!("could not write {}: {e}", to.display()));
    if std::fs::rename(from, to).is_ok() {
        return Ok(());
    }
    std::fs::copy(from, to).map_err(err)?;
    let _ = std::fs::remove_file(from);
    Ok(())
}

pub(super) fn xlsx_err(error: XlsxError) -> InternalErrors {
    InternalErrors::Internal(format!("XLSX export error: {error}"))
}

pub(super) fn create_csv_writer(
    path: &Path,
) -> Result<std::io::BufWriter<std::fs::File>, InternalErrors> {
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

pub(super) fn write_csv_row(
    out: &mut impl std::io::Write,
    fields: &[String],
) -> std::io::Result<()> {
    for (i, field) in fields.iter().enumerate() {
        if i > 0 {
            out.write_all(b",")?;
        }
        write_csv_field(out, field)?;
    }
    out.write_all(b"\n")
}

/// `field`, quoted when it holds a comma, quote or line break. Text that
/// Excel or LibreOffice would run as a formula (a class or image name
/// starting with `=`, `+`, `-`, `@`, tab or CR, from a shared project)
/// gets a `'` in front - numbers (`-0.5`) stay as they are.
pub(super) fn write_csv_field(out: &mut impl std::io::Write, field: &str) -> std::io::Result<()> {
    let guarded;
    let field = match runs_as_formula(field) {
        true => {
            guarded = format!("'{field}");
            guarded.as_str()
        }
        false => field,
    };
    match field.contains([',', '"', '\n', '\r']) {
        true => write!(out, "\"{}\"", field.replace('"', "\"\"")),
        false => out.write_all(field.as_bytes()),
    }
}

/// Whether a spreadsheet would read `field` as a formula: it starts with a
/// formula character and isn't a number.
pub(super) fn runs_as_formula(field: &str) -> bool {
    field.starts_with(['=', '+', '-', '@', '\t', '\r']) && field.trim().parse::<f64>().is_err()
}

/// `cell` as CSV - the text `cell_text` gives it (floats through `zmij`,
/// see `write_f64`), quoted where needed.
pub(super) fn write_csv_cell(out: &mut Vec<u8>, cell: &Cell) {
    use std::io::Write;
    match &cell.value {
        CellValue::Empty => {}
        CellValue::String(s) | CellValue::Class((s, _)) => {
            write_csv_field(out, s).expect("writing to memory")
        }
        CellValue::Float(v) => write_float(out, *v, v.is_finite()),
        CellValue::Integer(v) => write!(out, "{v}").expect("writing to memory"),
    }
}

/// Plain-text form of a `Cell`'s value for CSV, dropping its color/formatting
/// (CSV has no cells to color) — matches the CLI's identically-named helper
/// in `crates/cli/src/commands/common.rs`, kept separate there since it's
/// also used by that crate's table/JSON rendering, unrelated to export.
pub(super) fn cell_text(cell: &Cell) -> String {
    match &cell.value {
        CellValue::Empty => String::new(),
        CellValue::String(s) => s.clone(),
        CellValue::Class((s, _)) => s.clone(),
        CellValue::Float(v) => v.to_string(),
        CellValue::Integer(v) => v.to_string(),
    }
}

/// Extracts a `View::List` value cell's number — always `CellValue::Float`
/// in practice (see `well_fields_to_result`/`get_group_by_plate`'s own
/// `View::List` arms), but matching `Integer` too costs nothing and keeps
/// this from silently going quiet if that ever changes.
pub(super) fn cell_to_f64(cell: &Cell) -> Option<f64> {
    match &cell.value {
        CellValue::Float(value) => Some(*value as f64),
        CellValue::Integer(value) => Some(*value as f64),
        _ => None,
    }
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
pub(super) const FAILED_FONT_COLOR: u32 = 0xC07800;

pub(super) fn cell_format(cell: &Cell, num_format: Option<&str>) -> Format {
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
pub(super) fn write_cell(
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
pub(super) fn column_display_name(
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

pub(super) fn aggregation_label(aggregation: &Aggregation) -> &'static str {
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

pub(super) fn sanitize_filename_component(name: &str) -> String {
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
pub(super) struct SheetNamer {
    pub(super) used: HashSet<String>,
}

impl SheetNamer {
    pub(super) fn new() -> Self {
        Self {
            used: HashSet::new(),
        }
    }

    pub(super) fn unique(&mut self, wanted: &str) -> String {
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
