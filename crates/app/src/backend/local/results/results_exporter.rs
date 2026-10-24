use super::results_generator::{ResultsGenerator, class_display_label, sql_string_in_list};
use super::results_time_course::{TimeCourseHead, TimeCourseQuery};
use super::results_tracks::{TrackSummaryHead, TrackSummaryQuery};
use crate::api::{
    Aggregation, Cell, CellValue, Column, ColumnEntry, DatabaseResult, GroupedByImageFilter,
    ImageHeatmapFilter, ListFilter, Pagination, PlaneFilter, PlateFilterMulti, View,
    WellsBatchFilterMulti,
};
use crate::api::{ExportFormat, ExportProgress, ResultExport};
use crate::api::{
    TimeCourseFilter, TimeCourseLayout, TimeCourseLines, TimeCoursePoint, TimeCourseResult,
    TrackSummary, TrackSummaryFilter,
};
use crate::workspace::settings::SpeedUnit;
use evanalyzer_cfg::core_types::{InternalErrors, ObjectClass};
use rust_xlsxwriter::{Color, Format, FormatAlign, Workbook, Worksheet, XlsxError};
use std::collections::HashMap;
use std::collections::HashSet;
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};

/// Light gray Excel gives every other coloc-detail row (`Cell::alternating_color`)
/// so the fanned-out rows belonging to one source object stay visually
/// grouped — matches `Theme.list-row-alt-bg`'s role in the GUI's own List
/// view, just as a plain hex constant here since XLSX formatting has no
/// theme to pull from.
const ALTERNATING_ROW_BG: u32 = 0xF1F1F1;

const GRID_CELL_PX: u32 = 60;

const GRID_HEADER_PX: u32 = 24;

const EXPORT_FONT_SIZE: f64 = 10.0;

const GRID_NUM_FORMAT: &str = "0.00";

const EMPTY_CELL_BG: u32 = 0xE0E0E0;

// Flat-pivot (plate_list.xlsx/well_list.xlsx) value columns - not the
// square-grid width above, this is a plain table so there's no per-column
// budget to share across many columns.
const LIST_VALUE_COL_PX: u32 = 90;

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
    fn of(database: &ResultsGenerator, export: &ResultExport) -> Result<Self, InternalErrors> {
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

/// A time course table's cell.
#[derive(Debug, Clone, PartialEq)]
enum Field {
    Text(String),
    Number(f64),
    Empty,
}

/// A table: its header, then its rows.
type Fields = Vec<Vec<Field>>;

/// Where a table's rows go, a field at a time - nothing per row allocated:
/// a per-track time course has millions of rows.
trait FieldSink {
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
struct Discard;

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

/// The T, Z (several z-planes) and time (a frame interval) columns - the
/// time unit (seconds, minutes or hours, by how long the series runs) found
/// once per table.
struct TimeAxis {
    by_z: bool,
    unit: Option<(&'static str, f64)>,
}

impl TimeAxis {
    fn new(head: &TimeCourseHead) -> Self {
        let last = head
            .frame_interval_s
            .zip(head.last_t)
            .map(|(interval, t)| interval * f64::from(t));
        Self {
            by_z: head.by_z,
            unit: last.map(|last| match last {
                l if l >= 7200.0 => ("Time [h]", 3600.0),
                l if l >= 120.0 => ("Time [min]", 60.0),
                _ => ("Time [s]", 1.0),
            }),
        }
    }

    fn header(&self, out: &mut dyn FieldSink) -> Result<(), InternalErrors> {
        out.text("T")?;
        if self.by_z {
            out.text("Z")?;
        }
        if let Some((label, _)) = self.unit {
            out.text(label)?;
        }
        Ok(())
    }

    fn fields(
        &self,
        point: &TimeCoursePoint,
        out: &mut dyn FieldSink,
    ) -> Result<(), InternalErrors> {
        out.number(point.t as f64)?;
        if self.by_z {
            out.number(point.z as f64)?;
        }
        if let (Some((_, per)), Some(time)) = (self.unit, point.time_s) {
            out.number(time / per)?;
        }
        Ok(())
    }
}

fn optional(value: Option<f64>) -> Field {
    value.map_or(Field::Empty, Field::Number)
}

/// An XLSX sheet's rows, the header included.
const XLSX_MAX_ROWS: usize = 1_048_576;

/// The long table: a row per time point (z-plane) × line × class - the
/// line's own columns, the counts, then the values. Written a row at a time
/// as the points come (a per-track time course has millions).
struct TimeCourseLong<'a> {
    head: &'a TimeCourseHead,
    lines: TimeCourseLines,
    axis: TimeAxis,
}

impl<'a> TimeCourseLong<'a> {
    fn new(head: &'a TimeCourseHead, lines: TimeCourseLines) -> Self {
        Self {
            head,
            lines,
            axis: TimeAxis::new(head),
        }
    }

    fn header(&self, out: &mut dyn FieldSink) -> Result<(), InternalErrors> {
        self.axis.header(out)?;
        let line_columns: &[&str] = match self.lines {
            TimeCourseLines::Class => &[],
            TimeCourseLines::Image => &["Image"],
            TimeCourseLines::Well => &["Well"],
            TimeCourseLines::Group => &["Group"],
            TimeCourseLines::Track => &["Track", "Image"],
        };
        for column in line_columns
            .iter()
            .chain(&["Class", "N images", "N objects"])
        {
            out.text(column)?;
        }
        if self.head.tracked {
            out.text("N tracks")?;
        }
        for name in &self.head.value_names {
            out.text(name)?;
        }
        out.end_row()
    }

    fn row(&self, point: &TimeCoursePoint, out: &mut dyn FieldSink) -> Result<(), InternalErrors> {
        self.axis.fields(point, out)?;
        match self.lines {
            TimeCourseLines::Class => {}
            TimeCourseLines::Image => out.text(&point.image)?,
            TimeCourseLines::Well | TimeCourseLines::Group => out.text(&point.line)?,
            TimeCourseLines::Track => {
                out.text(&point.line)?;
                out.text(&point.image)?;
            }
        }
        out.text(&point.class_name)?;
        out.number(point.n_images as f64)?;
        out.number(point.n_objects as f64)?;
        if self.head.tracked {
            out.number(point.n_tracks as f64)?;
        }
        for value in &point.values {
            out.optional(*value)?;
        }
        out.end_row()
    }
}

/// The tracks table: who (image, well, class, track), when (time points,
/// duration), how it moved (path, distance, speed, straightness), whether
/// it touches the edge or is there from the start, then the measurements.
/// Lengths in µm with a pixel size, else px; durations in minutes with a
/// frame interval, else time points; the speed in the unit asked for when
/// it can be.
struct TrackColumns<'a> {
    head: &'a TrackSummaryHead,
    speed_unit: SpeedUnit,
}

impl<'a> TrackColumns<'a> {
    fn new(head: &'a TrackSummaryHead, speed_unit: SpeedUnit) -> Self {
        Self {
            head,
            speed_unit: match head.um && head.timed {
                true => speed_unit,
                false => SpeedUnit::PxPerFrame,
            },
        }
    }

    fn length_unit(&self) -> &'static str {
        if self.head.um { "µm" } else { "px" }
    }

    fn header(&self, out: &mut dyn FieldSink) -> Result<(), InternalErrors> {
        let length = self.length_unit();
        let speed = match self.speed_unit {
            SpeedUnit::PxPerFrame => "px/time point",
            SpeedUnit::UmPerMin => "µm/min",
            SpeedUnit::UmPerS => "µm/s",
        };
        out.text("Image")?;
        out.text(self.head.group_label)?;
        if self.head.by_z {
            out.text("Z")?;
        }
        for column in ["Class", "Track", "First T", "Last T", "Time points"] {
            out.text(column)?;
        }
        out.text(if self.head.timed {
            "Duration [min]"
        } else {
            "Duration [time points]"
        })?;
        out.text(&format!("Path length [{length}]"))?;
        out.text(&format!("Distance moved [{length}]"))?;
        out.text(&format!("Mean speed [{speed}]"))?;
        for column in ["Straightness", "Touches edge", "From first T"] {
            out.text(column)?;
        }
        for name in &self.head.value_names {
            out.text(name)?;
        }
        out.end_row()
    }

    fn row(&self, track: &TrackSummary, out: &mut dyn FieldSink) -> Result<(), InternalErrors> {
        let frames = f64::from(track.last_t.saturating_sub(track.first_t));
        let um = track.um_per_px.filter(|_| self.head.um);
        let length = |px: f64| um.map_or(px, |um| px * um);
        out.text(&track.image)?;
        out.text(&track.group)?;
        if self.head.by_z {
            out.number(f64::from(track.z))?;
        }
        out.text(&track.class_name)?;
        out.number(track.track_id as f64)?;
        out.number(f64::from(track.first_t))?;
        out.number(f64::from(track.last_t))?;
        out.number(f64::from(track.points))?;
        match track.s_per_frame.filter(|_| self.head.timed) {
            Some(s) => out.number(frames * s / 60.0)?,
            None => out.number(frames)?,
        }
        out.number(length(track.path_px))?;
        out.number(length(track.displacement_px))?;
        out.optional(track.speed_in(self.speed_unit))?;
        out.optional(track.straightness())?;
        out.text(if track.touches_edge { "TRUE" } else { "FALSE" })?;
        out.text(if track.from_start { "TRUE" } else { "FALSE" })?;
        for value in &track.values {
            out.optional(*value)?;
        }
        out.end_row()
    }
}

/// The wide tables, one per value and statistic (named like it): a row per
/// time point (z-plane), a column per line ("A1 · Nuclei"; the class alone
/// for class lines, without it when there's one class).
fn time_course_wide(
    result: &TimeCourseResult,
    lines: TimeCourseLines,
    axis: &TimeAxis,
) -> Result<Vec<(String, Fields)>, InternalErrors> {
    let one_class = result
        .points
        .iter()
        .all(|p| p.class_id == result.points.first().map_or(0, |f| f.class_id));
    let label = |p: &TimeCoursePoint| {
        let line = match lines {
            TimeCourseLines::Class => String::new(),
            TimeCourseLines::Image => p.image.clone(),
            TimeCourseLines::Track => format!("{} {}", p.line, p.image),
            TimeCourseLines::Well | TimeCourseLines::Group => p.line.clone(),
        };
        match (
            line.is_empty(),
            one_class && lines != TimeCourseLines::Class,
        ) {
            (true, _) => p.class_name.clone(),
            (false, true) => line,
            (false, false) => format!("{line} · {}", p.class_name),
        }
    };
    // The columns and planes in order of first appearance - looked up in
    // maps: a time course can have hundreds of thousands of points.
    let mut columns: Vec<String> = Vec::new();
    let mut column_of: HashMap<String, usize> = HashMap::new();
    let mut planes: Vec<(u32, u32)> = Vec::new();
    let mut plane_of: HashMap<(u32, u32), usize> = HashMap::new();
    let mut at: HashMap<(usize, usize), &TimeCoursePoint> = HashMap::new();
    for point in &result.points {
        let name = label(point);
        let column = *column_of.entry(name.clone()).or_insert_with(|| {
            columns.push(name);
            columns.len() - 1
        });
        let plane = *plane_of.entry((point.t, point.z)).or_insert_with(|| {
            planes.push((point.t, point.z));
            planes.len() - 1
        });
        at.insert((plane, column), point);
    }
    result
        .value_names
        .iter()
        .enumerate()
        .map(|(v, value_name)| {
            let mut header = Vec::new();
            axis.header(&mut header)?;
            header.extend(columns.iter().map(|c| Field::Text(c.clone())));
            let mut table = vec![header];
            for plane in 0..planes.len() {
                let first = (0..columns.len())
                    .find_map(|c| at.get(&(plane, c)))
                    .expect("a point of this plane");
                let mut row = Vec::new();
                axis.fields(first, &mut row)?;
                row.extend((0..columns.len()).map(|c| {
                    at.get(&(plane, c))
                        .map_or(Field::Empty, |p| optional(p.values[v]))
                }));
                table.push(row);
            }
            Ok((value_name.clone(), table))
        })
        .collect()
}

fn write_fields_csv(table: &Fields, path: &Path) -> Result<(), InternalErrors> {
    let mut out = FieldsCsv::create(path)?;
    table.iter().try_for_each(|row| out.row(row))
}

/// A CSV file written a field at a time: into one line buffer, reused.
struct FieldsCsv<'a> {
    out: std::io::BufWriter<std::fs::File>,
    path: &'a Path,
    line: Vec<u8>,
    /// Fields in `line` so far.
    fields: usize,
}

impl<'a> FieldsCsv<'a> {
    fn create(path: &'a Path) -> Result<Self, InternalErrors> {
        Ok(Self {
            out: create_csv_writer(path)?,
            path,
            line: Vec::new(),
            fields: 0,
        })
    }

    fn separate(&mut self) {
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
fn write_f64(out: &mut Vec<u8>, number: f64) {
    write_float(out, number, number.is_finite());
}

/// `write_f64` for an `f64` or `f32` (`finite`: its `is_finite()`).
fn write_float(out: &mut Vec<u8>, number: impl zmij::Float + std::fmt::Display, finite: bool) {
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

fn write_fields_sheet(worksheet: &mut Worksheet, table: &Fields) -> Result<(), InternalErrors> {
    let mut sheet = FieldsSheet::new(worksheet);
    table.iter().try_for_each(|row| sheet.row(row))
}

/// A worksheet written a field at a time, the first row in bold.
struct FieldsSheet<'a> {
    worksheet: &'a mut Worksheet,
    header_format: Format,
    format: Format,
    next_row: usize,
    next_column: u16,
}

impl<'a> FieldsSheet<'a> {
    fn new(worksheet: &'a mut Worksheet) -> Self {
        Self {
            worksheet,
            header_format: Format::new().set_bold().set_font_size(EXPORT_FONT_SIZE),
            format: Format::new().set_font_size(EXPORT_FONT_SIZE),
            next_row: 0,
            next_column: 0,
        }
    }

    /// The next cell: its row and column - an error beyond the last row.
    fn cell(&mut self) -> Result<(u32, u16), InternalErrors> {
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

/// A cell holding just `value`, no formatting.
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

impl ResultExport {
    /// Where an export document named `file_name` goes: inside
    /// `output_dir`, prefixed with `outputfile_prefix` (`{prefix}_{file_name}`)
    /// when one is set. The prefix is sanitized like every other file name
    /// part, so it can't point outside `output_dir`.
    fn output_path(&self, file_name: &str) -> std::path::PathBuf {
        match self
            .outputfile_prefix
            .as_deref()
            .map(str::trim)
            .filter(|prefix| !prefix.is_empty())
        {
            Some(prefix) => self.output_dir.join(format!(
                "{}_{file_name}",
                sanitize_filename_component(prefix)
            )),
            None => self.output_dir.join(file_name),
        }
    }

    /// `on_progress(message, current, total)` is called throughout to
    /// report what's happening — see `ExportProgress`'s doc comment for what
    /// `current`/`total` are relative to. Called from a background thread
    /// (this can take a while for a large database), so `on_progress`
    /// itself must not touch UI state directly — the caller's closure
    /// should just forward each call through `slint::invoke_from_event_loop`
    /// or equivalent.
    ///
    /// `cancel`: checked between documents and, within a large document,
    /// between images/classes/planes (see `check_cancelled`) — set it to
    /// `true` from another thread (e.g. a dialog's Cancel button) to stop
    /// the export at the next such checkpoint, returning
    /// `Err(InternalErrors::Cancelled)` rather than finishing. A fresh
    /// `AtomicBool::new(false)` is fine for a caller that never cancels
    /// (the CLI, today).
    pub fn start_export(
        &self,
        database: &ResultsGenerator,
        cancel: &AtomicBool,
        on_progress: ExportProgress,
    ) -> Result<(), InternalErrors> {
        if matches!(self.format, ExportFormat::CSV)
            && (self.with_plate_view_heatmap
                || self.with_well_view_heatmap
                || self.with_plate_view_list
                || self.with_well_view_list
                || self.with_heatmap)
        {
            return Err(InternalErrors::Internal(
                "CSV export only supports the List and Grouped-by-Image views — use XLSX for Plate/Well/Heatmap".to_string(),
            ));
        }
        check_cancelled(cancel)?;

        std::fs::create_dir_all(&self.output_dir).map_err(|e| {
            InternalErrors::Internal(format!(
                "Could not create export directory {:?}: {e}",
                self.output_dir
            ))
        })?;

        // Parquet ignores every `with_*`/column/filter setting below - it's
        // always a single raw dump of the whole `objects` table
        if matches!(self.format, ExportFormat::Parquet) {
            return self.export_as_parquet(database, cancel, &mut *on_progress);
        }

        if self.with_list_view {
            self.export_list(database, cancel, &mut *on_progress)?;
        }
        if self.with_grouped_by_image_list {
            self.export_grouped_by_image(database, cancel, &mut *on_progress)?;
        }
        if self.with_time_course {
            self.export_time_course(database, cancel, &mut *on_progress)?;
        }
        if self.with_track_summary {
            self.export_track_summary(database, cancel, &mut *on_progress)?;
        }
        if self.with_measurements {
            self.export_measurements(database, cancel, &mut *on_progress)?;
        }
        if self.with_plate_view_heatmap || self.with_well_view_heatmap {
            self.export_plate_and_well(database, cancel, &mut *on_progress)?;
        }
        if self.with_plate_view_list || self.with_well_view_list {
            self.export_plate_and_well_as_flat_list(database, cancel, &mut *on_progress)?;
        }
        if self.with_heatmap {
            self.export_heatmap(database, cancel, &mut *on_progress)?;
        }
        if let Some(drawn) = &self.drawn_images {
            let images = resolve_images(database, self)?;
            // The first exported plane's objects.
            let first = ExportPlanes::of(database, self)?.planes[0];
            super::drawn_images::export_drawn_images(
                database,
                drawn,
                &images,
                &self.output_dir,
                (first.z_stack, first.t_stack),
                cancel,
                &mut *on_progress,
            )?;
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
        cancel: &AtomicBool,
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
                check_cancelled(cancel)?;
                on_progress(
                    &format!("Exporting List: {image}"),
                    image_idx + 1,
                    images.len(),
                );
                let single_image = std::slice::from_ref(image);
                let stem = sanitize_filename_component(&image_stub(image));
                self.write_list_document(
                    database,
                    cancel,
                    single_image,
                    &object_classes,
                    &format!("list_{stem}"),
                )?;
            }
            return Ok(());
        }

        on_progress("Exporting List view", 0, 1);
        self.write_list_document(database, cancel, &images, &object_classes, "list")?;
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
        cancel: &AtomicBool,
        images: &[String],
        object_classes: &Option<Vec<ObjectClass>>,
        file_stem: &str,
    ) -> Result<(), InternalErrors> {
        match self.format {
            ExportFormat::XLSX => {
                let mut workbook = Workbook::new();
                // `write_list_sheet` only ever writes strictly top-to-bottom
                // (one page's rows after another, never revisiting an
                // earlier row) - exactly what "constant memory" mode
                // requires, and what lets it flush each row to a tempfile
                // as soon as the next one is written instead of keeping
                // every cell in memory for the life of the `Workbook`. A
                // standard `add_worksheet()` here would otherwise scale
                // memory with the row count writen - `rust_xlsxwriter`'s
                // own numbers put a 1M-row-ish sheet at ~200+ MB *before*
                // accounting for column count (cost is roughly per-cell),
                // which a single large image's object list can blow past
                // well under Excel's 1,048,576-row sheet cap.
                let sheet = workbook.add_worksheet_with_constant_memory();
                sheet.set_name("List").map_err(xlsx_err)?;
                write_list_sheet(sheet, database, cancel, self, images, object_classes, false)?;

                if self.with_list_coloc_details {
                    let coloc_sheet = workbook.add_worksheet_with_constant_memory();
                    coloc_sheet
                        .set_name("List (Coloc Details)")
                        .map_err(xlsx_err)?;
                    write_list_sheet(
                        coloc_sheet,
                        database,
                        cancel,
                        self,
                        images,
                        object_classes,
                        true,
                    )?;
                }

                workbook
                    .save(self.output_path(&format!("{file_stem}.xlsx")))
                    .map_err(xlsx_err)?;
            }
            ExportFormat::CSV => {
                write_list_csv(
                    database,
                    cancel,
                    self,
                    images,
                    object_classes,
                    false,
                    &self.output_path(&format!("{file_stem}.csv")),
                )?;
                if self.with_list_coloc_details {
                    write_list_csv(
                        database,
                        cancel,
                        self,
                        images,
                        object_classes,
                        true,
                        &self.output_path(&format!("{file_stem}_coloc_details.csv")),
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
        cancel: &AtomicBool,
        on_progress: ExportProgress,
    ) -> Result<(), InternalErrors> {
        check_cancelled(cancel)?;
        on_progress("Exporting Grouped Image List", 0, 1);

        // Group by image always returns a list of all images.
        // Image filter is not applied here.
        // let images = resolve_images(database, self)?;
        let images = resolve_all_images(database)?;
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
        let mut result: Option<DatabaseResult> = None;
        for &plane in &planes.planes {
            check_cancelled(cancel)?;
            let base_filter = GroupedByImageFilter {
                plane,
                images: Some(images.clone()),
                object_classes: object_classes.clone(),
                columns: columns.clone(),
                aggregation: self.aggregations.clone(),
                page: Pagination {
                    limit: 0,
                    after: None,
                },
                transpond_table: self.transpond_table,
            };
            let page = fetch_all_grouped_by_image_rows(database, &base_filter)?;
            let merged = result.get_or_insert_with(|| DatabaseResult {
                column_names: planes
                    .header()
                    .into_iter()
                    .chain(page.column_names.iter().cloned())
                    .collect(),
                row_names: Vec::new(),
                rows: Vec::new(),
                min: page.min,
                max: page.max,
                source_object_count: 0,
                row_locations: Vec::new(),
            });
            merged.min = merged.min.min(page.min);
            merged.max = merged.max.max(page.max);
            merged.source_object_count += page.source_object_count;
            merged.row_names.extend(page.row_names);
            merged.rows.extend(
                page.rows
                    .into_iter()
                    .map(|row| planes.cells(plane).into_iter().chain(row).collect()),
            );
        }
        let result = result.expect("at least one plane");

        match self.format {
            ExportFormat::XLSX => {
                let mut workbook = Workbook::new();
                let sheet = workbook.add_worksheet();
                sheet.set_name("Grouped by Image").map_err(xlsx_err)?;
                write_database_result_sheet(sheet, &result)?;

                workbook
                    .save(self.output_path("grouped_by_image.xlsx"))
                    .map_err(xlsx_err)?;
            }
            ExportFormat::CSV => {
                write_csv(&result, &self.output_path("grouped_by_image.csv"))?;
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

    // `plate.xlsx` and `well.xlsx`: one tab per object class in each
    // document, every tab stacking one square, colored grid block per
    // (column, aggregation) combination (and, one level further down in
    // `well.xlsx`, per well on top of that) — see `write_grid_block`.
    // Independently toggled by `with_plate_view_heatmap`/
    // `with_well_view_heatmap` - either can run without the other, each
    // skipping that side's own query and file entirely when off. The
    // flat-list form of the same data (see `export_plate_and_well_as_flat_list`)
    // is a separate pass into separate files, not appended here — this
    // function only ever needs `View::Heatmap` data.
    //
    // Only the first z/t in `self.z_stacks`/`self.t_stacks` is rendered:
    // unlike the List export, a plate/well grid is inherently a single
    // plane's snapshot (matching the GUI's own Matrix view, which shows one
    // z/t at a time via the global stepper, not a time series) — there's no
    // "stack every t one after another" equivalent for a grid the way
    // there is for a flat table.
    // `time_course.csv`/`.xlsx`: per time point (and z-plane), the
    // statistics of `columns` over each line's objects - long (a row per
    // time point × line × class) and/or wide (per value and statistic: a row
    // per time point, a column per line; CSV a file each, XLSX a sheet).
    /// `tracks.csv` / a "Tracks" sheet: a row per track (see
    /// `TrackColumns`), written as the tracks are read.
    /// `measurements.{csv,xlsx}`: one row per measurement value with what
    /// it is about - the results' `measurements` table, its columns named
    /// the same, of the chosen images and planes and the measurements among
    /// `columns` (none chosen: all). The long layout R or Python read
    /// without reshaping, and every new measurement fits without new
    /// columns.
    fn export_measurements(
        &self,
        database: &ResultsGenerator,
        cancel: &AtomicBool,
        on_progress: ExportProgress,
    ) -> Result<(), InternalErrors> {
        let err = |e: duckdb::Error| InternalErrors::Io(e.to_string());
        check_cancelled(cancel)?;
        on_progress("Exporting Measurements", 0, 1);
        if database.get_measurement_columns()?.is_empty() {
            return Err(InternalErrors::InvalidArgument(
                "These results have no measurements - a pipeline needs a measurement step"
                    .to_string(),
            ));
        }
        let images = resolve_images(database, self)?;
        let mut conditions = vec![match images.is_empty() {
            true => "false".to_string(),
            false => format!("image_rel_path IN ({})", sql_string_in_list(&images)),
        }];
        let keys: Vec<String> = self
            .columns
            .iter()
            .filter_map(|column| match column {
                Column::Measurement(m) => Some(m.key.clone()),
                _ => None,
            })
            .collect();
        if !keys.is_empty() {
            conditions.push(format!("key IN ({})", sql_string_in_list(&keys)));
        }
        for (column, range) in [("t_stack", &self.t_stacks), ("z_stack", &self.z_stacks)] {
            if let Some(range) = range {
                conditions.push(format!(
                    "{column} >= {} AND {column} < {}",
                    range.start, range.end
                ));
            }
        }
        const COLUMNS: [&str; 13] = [
            "image_name",
            "image_rel_path",
            "z_stack",
            "t_stack",
            "region",
            "object_id",
            "related_object_id",
            "object_class_name",
            "step",
            "measurement",
            "unit",
            "key",
            "value",
        ];
        let sql = format!(
            "SELECT image_name, image_rel_path, z_stack, t_stack, region, object_id::VARCHAR, \
             related_object_id::VARCHAR, object_class_name, step, measurement, unit, key, value \
             FROM measurements WHERE {} \
             ORDER BY image_name, t_stack, z_stack, step, key, region, object_id",
            conditions.join(" AND ")
        );
        let mut stmt = database.connection().prepare(&sql).map_err(err)?;
        let mut each_row = |out: &mut dyn FieldSink| -> Result<(), InternalErrors> {
            for name in COLUMNS {
                out.text(name)?;
            }
            out.end_row()?;
            let mut rows = stmt.query([]).map_err(err)?;
            let mut n = 0usize;
            while let Some(row) = rows.next().map_err(err)? {
                n += 1;
                if n.is_multiple_of(65_536) {
                    check_cancelled(cancel)?;
                }
                for i in 0..COLUMNS.len() {
                    match i {
                        2 | 3 => out.number(row.get::<_, i32>(i).map_err(err)? as f64)?,
                        12 => out.optional(row.get::<_, Option<f64>>(i).map_err(err)?)?,
                        _ => match row.get::<_, Option<String>>(i).map_err(err)? {
                            Some(text) => out.text(&text)?,
                            None => out.empty()?,
                        },
                    }
                }
                out.end_row()?;
            }
            Ok(())
        };
        match self.format {
            ExportFormat::XLSX => {
                let mut workbook = Workbook::new();
                let worksheet = workbook.add_worksheet_with_constant_memory();
                worksheet.set_name("Measurements").map_err(xlsx_err)?;
                each_row(&mut FieldsSheet::new(worksheet))?;
                workbook
                    .save(self.output_path("measurements.xlsx"))
                    .map_err(xlsx_err)?;
            }
            ExportFormat::CSV => {
                let path = self.output_path("measurements.csv");
                each_row(&mut FieldsCsv::create(&path)?)?;
            }
            // Written with the objects (`export_as_parquet`).
            ExportFormat::Parquet => {}
        }
        on_progress("Exporting Measurements", 1, 1);
        Ok(())
    }

    fn export_track_summary(
        &self,
        database: &ResultsGenerator,
        cancel: &AtomicBool,
        on_progress: ExportProgress,
    ) -> Result<(), InternalErrors> {
        check_cancelled(cancel)?;
        on_progress("Exporting Tracks", 0, 1);
        let settings = &self.track_summary;
        let filter = TrackSummaryFilter {
            grouping: self.grouping.clone(),
            columns: self.columns.clone(),
            statistics: settings.statistics.clone(),
            images: (!self.image_rel_paths.is_empty()).then(|| self.image_rel_paths.clone()),
            object_classes: (!self.object_classes.is_empty()).then(|| self.object_classes.clone()),
            min_points: settings.min_points,
            from_start: settings.from_start,
            no_edge: settings.no_edge,
        };
        let query = TrackSummaryQuery::prepare(database, &filter)?;
        let columns = TrackColumns::new(&query.head, settings.speed_unit);
        let each_track = |out: &mut dyn FieldSink| {
            columns.header(out)?;
            let mut n = 0usize;
            query.for_each(|track| {
                n += 1;
                if n.is_multiple_of(65_536) {
                    check_cancelled(cancel)?;
                }
                columns.row(track, out)
            })
        };
        match self.format {
            ExportFormat::XLSX => {
                let mut workbook = Workbook::new();
                let worksheet = workbook.add_worksheet_with_constant_memory();
                worksheet.set_name("Tracks").map_err(xlsx_err)?;
                each_track(&mut FieldsSheet::new(worksheet))?;
                workbook
                    .save(self.output_path("tracks.xlsx"))
                    .map_err(xlsx_err)?;
            }
            ExportFormat::CSV => {
                let path = self.output_path("tracks.csv");
                each_track(&mut FieldsCsv::create(&path)?)?;
            }
            ExportFormat::Parquet => {
                return Err(InternalErrors::Internal(
                    "Parquet export doesn't support the tracks".to_string(),
                ));
            }
        }
        on_progress("Exporting Tracks", 1, 1);
        Ok(())
    }

    fn export_time_course(
        &self,
        database: &ResultsGenerator,
        cancel: &AtomicBool,
        on_progress: ExportProgress,
    ) -> Result<(), InternalErrors> {
        check_cancelled(cancel)?;
        on_progress("Exporting Time course", 0, 1);
        let settings = &self.time_course;
        let write_long = settings.layout != TimeCourseLayout::Wide;
        let write_wide = settings.layout != TimeCourseLayout::Long;
        if write_wide && settings.lines == TimeCourseLines::Track {
            return Err(InternalErrors::InvalidArgument(
                "A wide time course per track would have a column per track - export it long"
                    .to_string(),
            ));
        }
        let filter = TimeCourseFilter {
            lines: settings.lines,
            grouping: self.grouping.clone(),
            columns: self.columns.clone(),
            statistics: settings.statistics.clone(),
            normalize: settings.normalize,
            images: (!self.image_rel_paths.is_empty()).then(|| self.image_rel_paths.clone()),
            object_classes: (!self.object_classes.is_empty()).then(|| self.object_classes.clone()),
            t_stacks: self.t_stacks,
            z_stacks: self.z_stacks,
            max_tracks: None,
        };
        let query = TimeCourseQuery::prepare(database, &filter)?;
        let long = TimeCourseLong::new(&query.head, settings.lines);
        // The long table is written as the points come; they're only kept
        // for the wide tables (never per track: refused above).
        let mut kept = Vec::new();
        let mut each_point = |out: &mut dyn FieldSink| {
            if write_long {
                long.header(out)?;
            }
            let mut n = 0usize;
            query.for_each(|point| {
                n += 1;
                if n.is_multiple_of(65_536) {
                    check_cancelled(cancel)?;
                }
                if write_long {
                    long.row(point, out)?;
                }
                if write_wide {
                    kept.push(point.clone());
                }
                Ok(())
            })
        };
        let wide = |kept: Vec<TimeCoursePoint>| {
            if !write_wide {
                return Ok(Vec::new());
            }
            let result = TimeCourseResult {
                value_names: query.head.value_names.clone(),
                by_z: query.head.by_z,
                frame_interval_s: query.head.frame_interval_s,
                points: kept,
            };
            time_course_wide(&result, settings.lines, &long.axis)
        };
        match self.format {
            ExportFormat::XLSX => {
                let mut workbook = Workbook::new();
                let mut names = SheetNamer::new();
                if write_long {
                    // Constant memory: rows go to a temporary file.
                    let worksheet = workbook.add_worksheet_with_constant_memory();
                    worksheet.set_name(names.unique("Long")).map_err(xlsx_err)?;
                    each_point(&mut FieldsSheet::new(worksheet))?;
                } else {
                    each_point(&mut Discard)?;
                }
                check_cancelled(cancel)?;
                for (name, table) in wide(kept)? {
                    let sheet = workbook.add_worksheet();
                    // Excel has no [ ] in tab names: "Area (px) mean".
                    let name = name.replace('[', "(").replace(']', ")");
                    sheet.set_name(names.unique(&name)).map_err(xlsx_err)?;
                    write_fields_sheet(sheet, &table)?;
                }
                workbook
                    .save(self.output_path("time_course.xlsx"))
                    .map_err(xlsx_err)?;
            }
            ExportFormat::CSV => {
                if write_long {
                    let path = self.output_path("time_course.csv");
                    each_point(&mut FieldsCsv::create(&path)?)?;
                } else {
                    each_point(&mut Discard)?;
                }
                check_cancelled(cancel)?;
                for (name, table) in wide(kept)? {
                    let file = format!(
                        "time_course_{}.csv",
                        sanitize_filename_component(name.as_str())
                    );
                    write_fields_csv(&table, &self.output_path(&file))?;
                }
            }
            ExportFormat::Parquet => {
                return Err(InternalErrors::Internal(
                    "Parquet export doesn't support the time course".to_string(),
                ));
            }
        }
        on_progress("Exporting Time course", 1, 1);
        Ok(())
    }

    fn export_plate_and_well(
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
                        let column_label = column_display_name(column, &available_columns);
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
                        let column_label = column_display_name(column, &available_columns);
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
                let column_label = column_display_name(column, &available_columns);
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
                        let column_label = column_display_name(column, &available_columns);
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

    /// `objects.parquet`: the entire `objects` table, every column, no z/t/
    /// image/class filtering and no column selection — DuckDB's own `COPY`
    /// writes the file directly, so this bypasses `get_object_list`/
    /// `stream_list_pages` entirely (unlike every other export above).
    /// Meant for a downstream tool that reads Parquet natively rather than
    /// for a human to open, so there's no equivalent of `with_list_view`'s
    /// column/plane scoping to apply here.
    fn export_as_parquet(
        &self,
        database: &ResultsGenerator,
        cancel: &AtomicBool,
        on_progress: ExportProgress,
    ) -> Result<(), InternalErrors> {
        check_cancelled(cancel)?;
        on_progress("Exporting Parquet", 0, 1);
        let path = self.output_path("objects.parquet");
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
        // The measurements beside them, as they are (one row per value).
        if !database.get_measurement_columns()?.is_empty() {
            let path = self.output_path("measurements.parquet");
            let path_literal = path.to_string_lossy().replace('\'', "''");
            database
                .connection()
                .execute_batch(&format!(
                    "COPY measurements TO '{path_literal}' (FORMAT parquet);"
                ))
                .map_err(|e| InternalErrors::Io(e.to_string()))?;
        }
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

fn resolve_all_images(database: &ResultsGenerator) -> Result<Vec<String>, InternalErrors> {
    let all = database.get_images()?;
    Ok(all
        .into_iter()
        .filter(|image| !image.disabled)
        .map(|image| image.rel_path.to_string_lossy().into_owned())
        .collect())
}

// `get_list` only ever hands back one bounded page (`ListFilter.page`) —
// same keyset-pagination shape `results_state_controller.rs` pages through
// for the GUI's List view (see its own `update_list_view`) — so an export,
// which needs every matching row rather than one page of them, walks every
// page via the same cursor-from-last-row-id trick.
//
// Every requested image is folded into ONE combined `images` filter here
// (not called once per image) - a `WHERE image_rel_path IN (...)` scoped
// to a single image still has to plan and run a full `objects` scan of its
// own, and that cost is dominated by the *whole* table's size, not just
// the matching image's row count, on a database that holds far more than
// what's being exported (the common case). So an N-image export previously
// meant N scans each paying that overhead; batching turns it into
// `total_matching_rows / PAGE_SIZE` queries instead of `images.len()`.
// `write_list_document`'s callers pass either every requested image at
// once (the shared-file List export) or a single-element slice
// (`export_list`'s `with_list_one_file_per_image` branch, calling it once
// per image) - either way this is the one shared entry point for turning a
// `ListFilter` sweep into pages.
//
// `on_page` runs once per page in z/t-then-row-id order, rather than this
// merging every page into one in-memory `DatabaseResult` first (unlike
// `fetch_all_grouped_by_image_rows`'s approach, which suits its much
// smaller image-grouped row count): peak RAM for a List export stays
// bounded by one page's worth of rows (`PAGE_SIZE`) regardless of how many
// images or planes the export spans, instead of growing with the combined
// row count now that a page can span every image at once.
fn stream_list_pages(
    database: &ResultsGenerator,
    cancel: &AtomicBool,
    export: &ResultExport,
    images: &[String],
    object_classes: &Option<Vec<ObjectClass>>,
    with_coloc_details: bool,
    planes: &ExportPlanes,
    mut on_page: impl FnMut(&DatabaseResult, bool, PlaneFilter) -> Result<(), InternalErrors>,
) -> Result<(), InternalErrors> {
    const PAGE_SIZE: i32 = 20_000;
    // Matches the old per-image loop's behavior when there's nothing to
    // scan at all: no query, no header, an empty document - rather than
    // running one (z, t) query per plane for a filter that's already known
    // to match nothing.
    if images.is_empty() {
        return Ok(());
    }

    let mut header_written = false;

    // Time point by time point, then z-plane (`ExportPlanes`).
    for &plane in &planes.planes {
        let mut cursor: Option<String> = None;
        loop {
            check_cancelled(cancel)?;
            let filter = ListFilter {
                plane,
                images: Some(images.to_vec()),
                object_classes: object_classes.clone(),
                columns: export.columns.clone(),
                with_coloc_details,
                related_to: None,
                page: Pagination {
                    limit: PAGE_SIZE,
                    after: cursor.take(),
                },
                transpond_table: export.transpond_table,
            };
            let page = database.get_object_list(&filter)?;
            let is_last_page = page.source_object_count < PAGE_SIZE as usize;
            cursor = page.row_names.last().cloned();

            on_page(&page, !header_written, plane)?;
            header_written = true;

            if is_last_page {
                break;
            }
        }
    }

    Ok(())
}

// Same keyset-pagination walk as `stream_list_pages`, just over
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
    cancel: &AtomicBool,
    export: &ResultExport,
    images: &[String],
    object_classes: &Option<Vec<ObjectClass>>,
    with_coloc_details: bool,
) -> Result<(), InternalErrors> {
    let header_format = Format::new().set_bold().set_font_size(EXPORT_FONT_SIZE);
    let mut next_row: u32 = 0;
    let planes = ExportPlanes::of(database, export)?;

    stream_list_pages(
        database,
        cancel,
        export,
        images,
        object_classes,
        with_coloc_details,
        &planes,
        |page, is_first_page, plane| {
            if is_first_page {
                let header = planes
                    .header()
                    .into_iter()
                    .chain(page.column_names.iter().cloned());
                for (col_idx, name) in header.enumerate() {
                    worksheet
                        .write_with_format(next_row, col_idx as u16, name.as_str(), &header_format)
                        .map_err(xlsx_err)?;
                }
                next_row += 1;
            }

            let plane_cells = planes.cells(plane);
            for row in &page.rows {
                if next_row as usize >= XLSX_MAX_ROWS {
                    return Err(InternalErrors::InvalidArgument(format!(
                        "The List has more than {} rows - more than an XLSX sheet holds: \
                         export it as CSV, or one file per image",
                        XLSX_MAX_ROWS - 1
                    )));
                }
                for (col_idx, cell) in plane_cells.iter().chain(row.iter()).enumerate() {
                    write_cell(worksheet, next_row, col_idx as u16, cell, None)?;
                }
                next_row += 1;
            }
            Ok(())
        },
    )
}

/// CSV sibling of `write_list_sheet`: same combined-images/per-page sweep
/// (so the two formats can never disagree on row order or content), written
/// straight to `path` a page at a time instead of into an XLSX worksheet.
fn write_list_csv(
    database: &ResultsGenerator,
    cancel: &AtomicBool,
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
    let planes = ExportPlanes::of(database, export)?;
    let mut line = Vec::new();

    stream_list_pages(
        database,
        cancel,
        export,
        images,
        object_classes,
        with_coloc_details,
        &planes,
        |page, is_first_page, plane| {
            use std::io::Write;
            if is_first_page {
                let header: Vec<String> = planes
                    .header()
                    .into_iter()
                    .chain(page.column_names.iter().cloned())
                    .collect();
                write_csv_row(&mut out, &header).map_err(write_err)?;
            }
            // Each row into one reused line - no string per cell: a large
            // List has millions of rows.
            let plane_cells = planes.cells(plane);
            for row in &page.rows {
                line.clear();
                for (i, cell) in plane_cells.iter().chain(row.iter()).enumerate() {
                    if i > 0 {
                        line.push(b',');
                    }
                    write_csv_cell(&mut line, cell);
                }
                line.push(b'\n');
                out.write_all(&line).map_err(write_err)?;
            }
            Ok(())
        },
    )
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

/// XLSX sibling of `write_csv`: writes `result` into `worksheet` in one
/// shot from an already-fetched, in-memory `DatabaseResult` - a bold
/// header row at row 0, then one row per `result.rows` entry below it.
/// Used by whichever export step already has its whole result in memory
/// (`export_grouped_by_image`); `write_list_sheet` writes straight from
/// `stream_list_pages`'s own pages instead, since a List export's combined
/// row count is the thing that RAM-bounded streaming exists for.
fn write_database_result_sheet(
    worksheet: &mut Worksheet,
    result: &DatabaseResult,
) -> Result<(), InternalErrors> {
    let header_format = Format::new().set_bold().set_font_size(EXPORT_FONT_SIZE);
    for (col_idx, name) in result.column_names.iter().enumerate() {
        worksheet
            .write_with_format(0, col_idx as u16, name.as_str(), &header_format)
            .map_err(xlsx_err)?;
    }
    for (row_idx, row) in result.rows.iter().enumerate() {
        let row_n = (row_idx + 1) as u32;
        for (col_idx, cell) in row.iter().enumerate() {
            write_cell(worksheet, row_n, col_idx as u16, cell, None)?;
        }
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
    for (i, field) in fields.iter().enumerate() {
        if i > 0 {
            out.write_all(b",")?;
        }
        write_csv_field(out, field)?;
    }
    out.write_all(b"\n")
}

/// `field`, quoted when it holds a comma, quote or line break.
fn write_csv_field(out: &mut impl std::io::Write, field: &str) -> std::io::Result<()> {
    match field.contains([',', '"', '\n', '\r']) {
        true => write!(out, "\"{}\"", field.replace('"', "\"\"")),
        false => out.write_all(field.as_bytes()),
    }
}

/// `cell` as CSV - the text `cell_text` gives it (floats through `zmij`,
/// see `write_f64`), quoted where needed.
fn write_csv_cell(out: &mut Vec<u8>, cell: &Cell) {
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
fn cell_text(cell: &Cell) -> String {
    match &cell.value {
        CellValue::Empty => String::new(),
        CellValue::String(s) => s.clone(),
        CellValue::Class((s, _)) => s.clone(),
        CellValue::Float(v) => v.to_string(),
        CellValue::Integer(v) => v.to_string(),
    }
}

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

/// Check for cancle the export process
fn check_cancelled(cancel: &AtomicBool) -> Result<(), InternalErrors> {
    if cancel.load(Ordering::Relaxed) {
        Err(InternalErrors::Cancelled)
    } else {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::super::test_support::{ObjectSpec, seed_db};
    use super::*;
    use crate::api::{ColorScale, ColorSchema};
    use crate::api::{Grouping, PlateSize, TimeCourseExport};
    use crate::api::{PlateFilter, WellsBatchFilter};
    use calamine::Reader as _;
    use std::path::PathBuf;
    use std::range::Range;

    /// `write_f64` writes text that reads back as the same value, without
    /// an exponent, no longer than what `f64` prints - mostly the very same
    /// text. Small, large, negative, whole, odd values and many random ones.
    #[test]
    fn numbers_read_back_as_written_without_an_exponent() {
        let mut values = vec![
            0.0,
            -0.0,
            1.0,
            -2.0,
            0.1,
            1.5e-7,
            1e-5,
            9.99e-6,
            123456.789,
            1e15,
            1e16,
            1.7e308,
            f64::MIN_POSITIVE,
            5e-324,
            f64::NAN,
            f64::INFINITY,
            f64::NEG_INFINITY,
            2.0 / 3.0,
        ];
        let mut seed = 0x9E37_79B9_7F4A_7C15u64;
        for _ in 0..200_000 {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            values.push(f64::from_bits(seed));
            values.push((seed % 1_000_000) as f64 / 1000.0);
            values.push(f64::from_bits(seed).abs() % 2e14);
            values.push(f64::from_bits(seed).abs() % 1e3);
        }
        let mut same = 0;
        for value in &values {
            let mut out = Vec::new();
            write_f64(&mut out, *value);
            let text = String::from_utf8(out).unwrap();
            let printed = value.to_string();
            let back: f64 = text.parse().unwrap();
            assert!(
                back.to_bits() == value.to_bits() || value.is_nan(),
                "{value:e}: {text}"
            );
            assert!(
                !text.contains('e') && text.len() <= printed.len(),
                "{value:e}: {text}"
            );
            same += usize::from(text == printed);
        }
        assert!(
            same * 100 >= values.len() * 99,
            "{same} of {}",
            values.len()
        );
    }

    /// List cells: `f32` values read back exactly, like `f32` prints them
    /// (but for the last of 9 digits), text quoted where needed.
    #[test]
    fn csv_cells_read_back_as_their_values() {
        let text = |value| {
            let mut out = Vec::new();
            write_csv_cell(&mut out, &plain_cell(value));
            String::from_utf8(out).unwrap()
        };
        let mut seed = 0x2545_F491u32;
        for _ in 0..200_000 {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            let value = f32::from_bits(seed);
            let written = text(CellValue::Float(value));
            let back: f32 = written.parse().unwrap();
            assert!(
                back.to_bits() == value.to_bits() || value.is_nan(),
                "{value:e}: {written}"
            );
            assert!(!written.contains('e') && written.len() <= value.to_string().len());
        }
        assert_eq!(text(CellValue::Float(150.0)), "150");
        assert_eq!(text(CellValue::Integer(-3)), "-3");
        assert_eq!(text(CellValue::String("a,b".into())), "\"a,b\"");
        assert_eq!(text(CellValue::Empty), "");
    }

    /// One field as `write_csv_field` writes it.
    fn csv_escape(field: &str) -> String {
        let mut out = Vec::new();
        write_csv_field(&mut out, field).unwrap();
        String::from_utf8(out).unwrap()
    }

    /// Fields go out as before: empty ones (the first too) between commas,
    /// whole numbers without a fraction, the others as `f64` prints them.
    #[test]
    fn csv_fields_are_written_like_their_text() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("fields.csv");
        write_fields_csv(
            &vec![
                vec![Field::Empty, Field::Text("a,b".into()), Field::Number(3.0)],
                vec![Field::Number(0.1), Field::Empty, Field::Number(-2.0)],
                vec![Field::Number(-0.0), Field::Number(1e300), Field::Empty],
            ],
            &path,
        )
        .unwrap();
        assert_eq!(
            std::fs::read_to_string(&path).unwrap(),
            format!(",\"a,b\",3\n0.1,,-2\n-0,{},\n", 1e300)
        );
    }

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
            disabled: false,
            any_disabled: false,
            failed: false,
            any_failed: false,
        }
    }

    #[test]
    fn cell_format_is_just_the_font_size_for_a_plain_enabled_cell() {
        assert_eq!(
            cell_format(&cell_str("x"), None),
            Format::new().set_font_size(EXPORT_FONT_SIZE)
        );
    }

    #[test]
    fn cell_format_grays_and_strikes_through_a_disabled_cell_with_no_other_formatting() {
        assert_eq!(
            cell_format(
                &Cell {
                    disabled: true,
                    ..cell_str("x")
                },
                None
            ),
            Format::new()
                .set_font_size(EXPORT_FONT_SIZE)
                .set_background_color(Color::RGB(EMPTY_CELL_BG))
                .set_font_strikethrough()
        );
    }

    #[test]
    fn cell_format_grays_out_a_disabled_cell_instead_of_its_own_background_color() {
        // A disabled cell must not keep its heatmap/class color - otherwise
        // it still reads as "just another colored cell" at a glance, same
        // as the GUI grid's flat gray fill for a disabled tile.
        assert_eq!(
            cell_format(
                &Cell {
                    bg_color: 0x112233,
                    disabled: true,
                    ..cell_str("x")
                },
                None
            ),
            Format::new()
                .set_font_size(EXPORT_FONT_SIZE)
                .set_background_color(Color::RGB(EMPTY_CELL_BG))
                .set_font_strikethrough()
        );
    }

    #[test]
    fn cell_format_grays_out_a_disabled_cell_instead_of_the_alternating_row_background() {
        assert_eq!(
            cell_format(
                &Cell {
                    alternating_color: true,
                    disabled: true,
                    ..cell_str("x")
                },
                None
            ),
            Format::new()
                .set_font_size(EXPORT_FONT_SIZE)
                .set_background_color(Color::RGB(EMPTY_CELL_BG))
                .set_font_strikethrough()
        );
    }

    #[test]
    fn cell_format_fills_an_existing_but_empty_grid_cell() {
        assert_eq!(
            cell_format(
                &Cell {
                    value: CellValue::Empty,
                    search_key: Some(("A1".to_string(), "A1".to_string())),
                    ..cell_str("x")
                },
                None
            ),
            Format::new()
                .set_font_size(EXPORT_FONT_SIZE)
                .set_background_color(Color::RGB(EMPTY_CELL_BG))
        );
    }

    #[test]
    fn cell_format_leaves_a_truly_absent_grid_position_with_just_the_font_size() {
        assert_eq!(
            cell_format(
                &Cell {
                    value: CellValue::Empty,
                    search_key: None,
                    ..cell_str("x")
                },
                None
            ),
            Format::new().set_font_size(EXPORT_FONT_SIZE),
            "no well/field sits here at all, so it must stay blank like write_blank's no-op"
        );
    }

    #[test]
    fn cell_format_applies_a_num_format_and_centers_a_grid_cell() {
        assert_eq!(
            cell_format(&cell_str("x"), Some("0.00")),
            Format::new()
                .set_font_size(EXPORT_FONT_SIZE)
                .set_num_format("0.00")
                .set_align(FormatAlign::Center)
                .set_align(FormatAlign::VerticalCenter)
        );
    }

    /// `write_cell` itself (not just `cell_format`) - an existing-but-empty
    /// grid cell (a real well/field with no value) must read back as the
    /// text "-", not a truly blank cell (that's reserved for a grid
    /// position with no well/field at all).
    #[test]
    fn write_cell_labels_an_existing_but_empty_grid_cell_as_a_dash() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("cell.xlsx");
        let mut workbook = Workbook::new();
        let sheet = workbook.add_worksheet();
        write_cell(
            sheet,
            0,
            0,
            &Cell {
                value: CellValue::Empty,
                search_key: Some(("A1".to_string(), "A1".to_string())),
                ..cell_str("unused")
            },
            None,
        )
        .expect("write existing-but-empty cell");
        write_cell(
            sheet,
            0,
            1,
            &Cell {
                value: CellValue::Empty,
                search_key: None,
                ..cell_str("unused")
            },
            None,
        )
        .expect("write truly absent cell");
        workbook.save(&path).expect("save workbook");

        let mut wb: calamine::Xlsx<_> = calamine::open_workbook(&path).expect("open cell.xlsx");
        let range = wb.worksheet_range_at(0).expect("sheet").expect("range");
        assert_eq!(
            range.get_value((0, 0)),
            Some(&calamine::Data::String("-".to_string())),
            "a real well/field with no value reads as \"-\""
        );
        assert!(
            matches!(range.get_value((0, 1)), None | Some(calamine::Data::Empty)),
            "a grid position with no well/field at all stays truly blank"
        );
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
                disabled: false,
                any_disabled: false,
                failed: false,
                any_failed: false,
            }),
            ""
        );
        assert_eq!(
            cell_text(&Cell {
                value: CellValue::Float(1.5),
                bg_color: 0,
                alternating_color: false,
                search_key: None,
                disabled: false,
                any_disabled: false,
                failed: false,
                any_failed: false,
            }),
            "1.5"
        );
        assert_eq!(
            cell_text(&Cell {
                value: CellValue::Integer(7),
                bg_color: 0,
                alternating_color: false,
                search_key: None,
                disabled: false,
                any_disabled: false,
                failed: false,
                any_failed: false,
            }),
            "7"
        );
        assert_eq!(
            cell_text(&Cell {
                value: CellValue::Class(("ClassA".to_string(), 0)),
                bg_color: 0,
                alternating_color: false,
                search_key: None,
                disabled: false,
                any_disabled: false,
                failed: false,
                any_failed: false,
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

    /// Same as `open`, plus one extra `images` row with no objects at all
    /// (a well/field that has an image but zero detected objects) -
    /// `ResultsGenerator`'s own `database` connection is private even to
    /// this sibling test module, and `seed_db`/`ObjectSpec` only ever
    /// create an image alongside a seeded object, so the raw insert has to
    /// happen through its own connection, before `ResultsGenerator` ever
    /// opens the file (matching how `seed_db` itself closes its connection
    /// before returning).
    fn open_with_empty_image(
        objects: &[ObjectSpec],
        empty_image_name: &str,
    ) -> (ResultsGenerator, PathBuf) {
        let dir = tempfile::tempdir().expect("tempdir");
        let db_path = dir.path().join("results.evadb");
        seed_db(&db_path, objects);
        {
            let conn = duckdb::Connection::open(&db_path).expect("open db for extra image");
            conn.execute(
                "INSERT INTO images (image_name, image_rel_path, width, height, c_stacks, z_stacks, t_stacks) \
                 VALUES (?, ?, 100, 100, 1, 1, 1)",
                duckdb::params![empty_image_name, empty_image_name],
            )
            .expect("insert empty image");
        }
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

    fn no_cancel() -> AtomicBool {
        AtomicBool::new(false)
    }

    /// Every z-plane and time point (`None`: all).
    fn full_range() -> (Option<Range<u32>>, Option<Range<u32>>) {
        (None, None)
    }

    /// End-to-end proof that `start_export`'s CSV path (shared by the CLI
    /// and the GUI export dialog) actually writes a real
    /// `list.csv`/`grouped_by_image.csv` with the expected content — not
    /// just that a caller's own call site happens to work.
    #[test]
    fn transposed_export_writes_the_classes_side_by_side() {
        let (database, out_dir) = open(&[
            ObjectSpec::new("img1.tif", "ClassA", 1, 100),
            ObjectSpec::new("img1.tif", "ClassB", 2, 200),
            ObjectSpec::new("img2.tif", "ClassA", 1, 300),
        ]);
        let (z_stacks, t_stacks) = full_range();
        ResultExport {
            output_dir: out_dir.clone(),
            format: ExportFormat::CSV,
            z_stacks,
            t_stacks,
            columns: vec![Column::AreaSizePx],
            aggregations: vec![Aggregation::Avg],
            with_list_view: true,
            with_grouped_by_image_list: true,
            transpond_table: true,
            ..Default::default()
        }
        .start_export(&database, &no_cancel(), &mut no_progress())
        .expect("transposed export");

        let list = std::fs::read_to_string(out_dir.join("list.csv")).unwrap();
        let mut lines = list.lines();
        assert_eq!(
            lines.next().unwrap(),
            "image,Area [px] (ClassA),Area [px] (ClassB)"
        );
        assert_eq!(lines.count(), 2, "one row per image here: {list}");

        let grouped = std::fs::read_to_string(out_dir.join("grouped_by_image.csv")).unwrap();
        let mut lines = grouped.lines();
        assert_eq!(
            lines.next().unwrap(),
            "image,Area [px] (AVG) (ClassA),Area [px] (AVG) (ClassB)"
        );
        // img2 has no ClassB objects: an empty field, not 0.
        assert!(
            lines.any(|line| line.starts_with("img2.tif,300") && line.ends_with(',')),
            "{grouped}"
        );
    }

    #[test]
    fn output_file_prefix_is_prepended_to_every_document() {
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
        ResultExport {
            output_dir: out_dir.clone(),
            outputfile_prefix: Some("exp1".into()),
            format: ExportFormat::XLSX,
            z_stacks,
            t_stacks,
            columns,
            aggregations: vec![Aggregation::Avg],
            with_list_view: true,
            with_grouped_by_image_list: true,
            with_heatmap: true,
            ..Default::default()
        }
        .start_export(&database, &no_cancel(), &mut no_progress())
        .expect("prefixed export");

        let mut written: Vec<String> = std::fs::read_dir(&out_dir)
            .unwrap()
            .map(|entry| entry.unwrap().file_name().to_string_lossy().into_owned())
            .collect();
        written.sort();
        assert!(
            written.contains(&"exp1_list.xlsx".to_string()),
            "{written:?}"
        );
        assert!(
            written.contains(&"exp1_grouped_by_image.xlsx".to_string()),
            "{written:?}"
        );
        assert!(
            written.iter().any(|f| f.starts_with("exp1_heatmap_")),
            "{written:?}"
        );
        assert!(
            written.iter().all(|f| f.starts_with("exp1_")),
            "{written:?}"
        );
    }

    #[test]
    fn output_path_sanitizes_the_prefix_and_ignores_a_blank_one() {
        let export = |prefix: Option<&str>| ResultExport {
            output_dir: PathBuf::from("/out"),
            outputfile_prefix: prefix.map(str::to_string),
            ..Default::default()
        };
        assert_eq!(
            export(None).output_path("list.csv"),
            PathBuf::from("/out/list.csv")
        );
        assert_eq!(
            export(Some("  ")).output_path("list.csv"),
            PathBuf::from("/out/list.csv")
        );
        assert_eq!(
            export(Some(" run 2 ")).output_path("list.csv"),
            PathBuf::from("/out/run_2_list.csv")
        );
        // Can't escape the output folder.
        assert_eq!(
            export(Some("../../etc/x")).output_path("list.csv"),
            PathBuf::from("/out/______etc_x_list.csv")
        );
    }

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
            .start_export(&database, &no_cancel(), &mut no_progress())
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
            .start_export(&database, &no_cancel(), &mut no_progress())
            .expect("csv grouped export");
        let grouped_csv = std::fs::read_to_string(out_dir.join("grouped_by_image.csv"))
            .expect("read grouped_by_image.csv");
        // One row per (image, class) - including the class an image has no
        // objects of: a count of 0, and no averages.
        assert_eq!(
            grouped_csv.lines().count(),
            5,
            "expected a header plus one row per image and class: {grouped_csv}"
        );
        assert!(grouped_csv.contains("img1.tif,ClassB,0,,"), "{grouped_csv}");
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
            .start_export(&database, &no_cancel(), &mut no_progress())
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
            .start_export(&database, &no_cancel(), &mut no_progress())
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
            .start_export(&database, &no_cancel(), &mut no_progress())
            .expect("csv per-image export");

        let img1 = std::fs::read_to_string(out_dir.join("list_img1.csv")).expect("list_img1.csv");
        let img2 = std::fs::read_to_string(out_dir.join("list_img2.csv")).expect("list_img2.csv");
        assert!(img1.contains("ClassA") && !img1.contains("ClassB"));
        assert!(img2.contains("ClassB") && !img2.contains("ClassA"));
    }

    #[test]
    fn start_export_rejects_csv_combined_with_plate_view_heatmap() {
        let (database, out_dir) = open(&[ObjectSpec::new("img1.tif", "ClassA", 1, 100)]);
        let (z_stacks, t_stacks) = full_range();
        let export = ResultExport {
            output_dir: out_dir,
            format: ExportFormat::CSV,
            z_stacks,
            t_stacks,
            with_plate_view_heatmap: true,
            ..Default::default()
        };
        let result = export.start_export(&database, &no_cancel(), &mut no_progress());
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
        let result = export.start_export(&database, &no_cancel(), &mut no_progress());
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
            .start_export(&database, &no_cancel(), &mut no_progress())
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
        let result = export.write_list_document(&database, &no_cancel(), &[], &None, "list");
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
        let result = export.export_grouped_by_image(&database, &no_cancel(), &mut no_progress());
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

    /// "Images with objects drawn": a PNG per image, of the real file.
    #[test]
    fn images_with_objects_drawn_are_written_as_png() {
        let name = "multi-channel-4D-series.ome.tif";
        let (database, out_dir) = open(&[ObjectSpec::new(name, "ClassA", 1, 100)]);
        let (z_stacks, t_stacks) = full_range();
        let export = ResultExport {
            output_dir: out_dir.clone(),
            format: ExportFormat::XLSX,
            z_stacks,
            t_stacks,
            drawn_images: Some(crate::api::DrawnImages {
                image_root: PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../core/tests"),
                series: 0,
                max_side: 4096,
                colors: [(0, [1.0, 0.0, 0.0])].into(),
            }),
            ..Default::default()
        };
        export
            .start_export(&database, &no_cancel(), &mut no_progress())
            .expect("export");
        let png = out_dir
            .join("images")
            .join("multi-channel-4D-series.ome.png");
        let picture = image::open(&png).expect("a png").to_rgb8();
        assert!(picture.width() > 0 && picture.height() > 0);
    }

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
            .start_export(&database, &no_cancel(), &mut no_progress())
            .expect("xlsx export");

        let mut workbook: Xlsx<_> =
            open_workbook(out_dir.join("list.xlsx")).expect("open exported xlsx");
        let range = workbook.worksheet_range("List").expect("List sheet");
        // The image first, as in the List.
        assert_eq!(
            range.get_value((0, 0)),
            Some(&Data::String("Image".to_string()))
        );
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

    /// A time series: img1 and img2 at time point 0, img1 again at 1.
    fn time_series() -> (ResultsGenerator, PathBuf) {
        open(&[
            ObjectSpec::new("img1.tif", "ClassA", 1, 100),
            ObjectSpec::new("img2.tif", "ClassA", 1, 200),
            ObjectSpec::new("img1.tif", "ClassA", 1, 300).at_plane(0, 1),
        ])
    }

    /// The CSV `name` of an export of `export`'s settings, as rows of cells.
    fn exported_csv(export: ResultExport, name: &str) -> Vec<Vec<String>> {
        let out_dir = export.output_dir.clone();
        let (database, _) = time_series();
        export
            .start_export(&database, &no_cancel(), &mut no_progress())
            .expect("export");
        std::fs::read_to_string(out_dir.join(name))
            .expect(name)
            .lines()
            .map(|line| line.split(',').map(str::to_string).collect())
            .collect()
    }

    fn out_dir() -> PathBuf {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("out");
        std::mem::forget(dir);
        path
    }

    /// Every time point by default, time point by time point, each row with
    /// its "T" - no "Z": the results have one z-plane.
    #[test]
    fn a_time_series_list_has_every_time_point_with_its_t() {
        let rows = exported_csv(
            ResultExport {
                output_dir: out_dir(),
                format: ExportFormat::CSV,
                columns: vec![Column::AreaSizePx],
                with_list_view: true,
                ..Default::default()
            },
            "list.csv",
        );
        assert_eq!(rows[0][0], "T");
        assert_ne!(rows[0][1], "Z");
        let t_and_area: Vec<(&str, &str)> = rows[1..]
            .iter()
            .map(|row| (row[0].as_str(), row.last().unwrap().as_str()))
            .collect();
        assert_eq!(t_and_area, [("0", "100"), ("0", "200"), ("1", "300")]);
    }

    /// The per-image export: a row per image at each of its time points
    /// (img2 has only time point 0), time point by time point.
    #[test]
    fn the_grouped_export_has_a_row_per_time_point_and_image() {
        let rows = exported_csv(
            ResultExport {
                output_dir: out_dir(),
                format: ExportFormat::CSV,
                columns: vec![Column::AreaSizePx],
                aggregations: vec![Aggregation::Avg],
                with_grouped_by_image_list: true,
                ..Default::default()
            },
            "grouped_by_image.csv",
        );
        assert_eq!(rows[0], ["T", "image", "class", "Area [px] (AVG)"]);
        let t_image_area: Vec<(&str, &str, &str)> = rows[1..]
            .iter()
            .map(|row| (row[0].as_str(), row[1].as_str(), row[3].as_str()))
            .collect();
        assert_eq!(
            t_image_area,
            [
                ("0", "img1.tif", "100"),
                ("0", "img2.tif", "200"),
                ("1", "img1.tif", "300")
            ]
        );
    }

    /// A chosen range: only its time points - still with their "T".
    #[test]
    fn a_time_point_range_exports_only_those() {
        let rows = exported_csv(
            ResultExport {
                output_dir: out_dir(),
                format: ExportFormat::CSV,
                t_stacks: Some(Range { start: 1, end: 2 }),
                columns: vec![Column::AreaSizePx],
                with_list_view: true,
                ..Default::default()
            },
            "list.csv",
        );
        assert_eq!(rows[0][0], "T");
        assert_eq!(rows.len(), 2, "the header and time point 1's object");
        assert_eq!(
            (rows[1][0].as_str(), rows[1].last().unwrap().as_str()),
            ("1", "300")
        );
    }

    /// Results of one plane: no "T"/"Z" columns - exports as before.
    #[test]
    fn results_of_one_plane_have_no_plane_columns() {
        let (database, out_dir) = open(&[ObjectSpec::new("img1.tif", "ClassA", 1, 100)]);
        ResultExport {
            output_dir: out_dir.clone(),
            format: ExportFormat::CSV,
            columns: vec![Column::AreaSizePx],
            with_list_view: true,
            ..Default::default()
        }
        .start_export(&database, &no_cancel(), &mut no_progress())
        .unwrap();
        let csv = std::fs::read_to_string(out_dir.join("list.csv")).unwrap();
        let header = csv.lines().next().unwrap();
        assert!(
            !header.starts_with("T,") && !header.starts_with("Z,"),
            "{header}"
        );
    }

    /// `open`, with the planes a run records analysing
    /// (`images.t_stacks_analyzed`/`z_stacks_analyzed`): every image at every time point
    /// `0..times` (one z-plane) - also those where nothing was found.
    fn open_analysed(objects: &[ObjectSpec], times: i32) -> (ResultsGenerator, PathBuf) {
        let dir = tempfile::tempdir().expect("tempdir");
        let db_path = dir.path().join("results.evadb");
        seed_db(&db_path, objects);
        {
            let conn = duckdb::Connection::open(&db_path).expect("open db");
            conn.execute_batch(
                "ALTER TABLE images ADD COLUMN t_stacks_analyzed INTEGER[];
                 ALTER TABLE images ADD COLUMN z_stacks_analyzed INTEGER[];",
            )
            .unwrap();
            conn.execute(
                "UPDATE images SET t_stacks = ?, \
                 t_stacks_analyzed = CAST(range(0, ?) AS INTEGER[]), z_stacks_analyzed = [0]",
                duckdb::params![times, times],
            )
            .unwrap();
        }
        let out_dir = dir.path().join("out");
        std::mem::forget(dir);
        (
            ResultsGenerator::open_database(db_path).expect("open database"),
            out_dir,
        )
    }

    /// Every analysed time point - also one where no image found anything -
    /// and every image at each: Count 0 where it found nothing.
    #[test]
    fn the_grouped_export_has_every_analysed_time_point_and_image() {
        let (database, out_dir) = open_analysed(
            &[
                ObjectSpec::new("img1.tif", "ClassA", 1, 100),
                ObjectSpec::new("img2.tif", "ClassA", 1, 200),
                ObjectSpec::new("img1.tif", "ClassA", 1, 300).at_plane(0, 1),
            ],
            3,
        );
        ResultExport {
            output_dir: out_dir.clone(),
            format: ExportFormat::CSV,
            columns: vec![Column::Count],
            aggregations: vec![Aggregation::Sum],
            with_grouped_by_image_list: true,
            ..Default::default()
        }
        .start_export(&database, &no_cancel(), &mut no_progress())
        .expect("export");
        let csv = std::fs::read_to_string(out_dir.join("grouped_by_image.csv")).unwrap();
        let rows: Vec<Vec<&str>> = csv.lines().map(|line| line.split(',').collect()).collect();
        assert_eq!(rows[0][..3], ["T", "image", "class"]);
        let t_image_count: Vec<(&str, &str, &str)> = rows[1..]
            .iter()
            .map(|row| (row[0], row[1], row[3]))
            .collect();
        assert_eq!(
            t_image_count,
            [
                ("0", "img1.tif", "1"),
                ("0", "img2.tif", "1"),
                ("1", "img1.tif", "1"),
                ("1", "img2.tif", "0"),
                ("2", "img1.tif", "0"),
                ("2", "img2.tif", "0"),
            ]
        );
    }

    fn time_course_export(format: ExportFormat, layout: TimeCourseLayout) -> PathBuf {
        let (database, out_dir) = open_analysed(
            &[
                ObjectSpec::new("img1.tif", "ClassA", 1, 100),
                ObjectSpec::new("img2.tif", "ClassA", 1, 200),
                ObjectSpec::new("img1.tif", "ClassA", 1, 300).at_plane(0, 1),
            ],
            3,
        );
        ResultExport {
            output_dir: out_dir.clone(),
            format,
            columns: vec![Column::AreaSizePx],
            with_time_course: true,
            time_course: TimeCourseExport {
                layout,
                ..Default::default()
            },
            ..Default::default()
        }
        .start_export(&database, &no_cancel(), &mut no_progress())
        .expect("time course export");
        out_dir
    }

    /// Long: a row per time point and class line - the empty T2 too; wide:
    /// per value and statistic, a row per time point, a column per line.
    #[test]
    fn the_time_course_exports_long_and_wide_csv() {
        let out_dir = time_course_export(ExportFormat::CSV, TimeCourseLayout::Both);
        let long = std::fs::read_to_string(out_dir.join("time_course.csv")).unwrap();
        assert_eq!(
            long.lines().collect::<Vec<_>>(),
            [
                "T,Class,N images,N objects,Area [px] mean,Area [px] SD",
                "0,ClassA,2,2,150,70.71067811865476",
                "1,ClassA,2,1,300,",
                "2,ClassA,2,0,,",
            ]
        );
        let wide = std::fs::read_to_string(out_dir.join("time_course_Area__px__mean.csv")).unwrap();
        assert_eq!(
            wide.lines().collect::<Vec<_>>(),
            ["T,ClassA", "0,150", "1,300", "2,"]
        );
        assert!(out_dir.join("time_course_Area__px__SD.csv").exists());
    }

    /// XLSX: the long sheet and a sheet per value and statistic - or only
    /// what the layout asks for.
    #[test]
    fn the_time_course_xlsx_has_a_sheet_per_table() {
        let sheets = |layout| {
            let out_dir = time_course_export(ExportFormat::XLSX, layout);
            let workbook: calamine::Xlsx<_> =
                calamine::open_workbook(out_dir.join("time_course.xlsx")).expect("open");
            workbook.sheet_names()
        };
        assert_eq!(
            sheets(TimeCourseLayout::Both),
            ["Long", "Area (px) mean", "Area (px) SD"]
        );
        assert_eq!(sheets(TimeCourseLayout::Long), ["Long"]);
        assert_eq!(
            sheets(TimeCourseLayout::Wide),
            ["Area (px) mean", "Area (px) SD"]
        );
    }

    /// An XLSX sheet holds 1,048,576 rows: one more is refused with a hint
    /// to export CSV - not cut off.
    #[test]
    fn an_xlsx_sheet_refuses_rows_beyond_its_limit() {
        let mut workbook = Workbook::new();
        let mut sheet = FieldsSheet::new(workbook.add_worksheet_with_constant_memory());
        sheet.next_row = XLSX_MAX_ROWS - 1;
        sheet.row(&[Field::Number(1.0)]).expect("the last row");
        let error = sheet.row(&[Field::Number(2.0)]).expect_err("beyond");
        assert!(error.to_string().contains("export it as CSV"), "{error}");
    }

    /// Tracks: a row per track - who, when, how it moved (µm with the
    /// fixture's pixel size; time points without a frame interval: the
    /// speed per time point, whatever unit was asked for), its values.
    #[test]
    fn the_tracks_export_has_a_row_per_track() {
        let (database, out_dir) = open_analysed(
            &[
                ObjectSpec::new("img1.tif", "ClassA", 1, 10)
                    .in_track(7)
                    .at_centroid(0.0, 0.0),
                ObjectSpec::new("img1.tif", "ClassA", 1, 30)
                    .in_track(7)
                    .at_plane(0, 1)
                    .at_centroid(3.0, 4.0),
                ObjectSpec::new("img1.tif", "ClassA", 1, 99),
            ],
            2,
        );
        ResultExport {
            output_dir: out_dir.clone(),
            format: ExportFormat::CSV,
            columns: vec![Column::AreaSizePx],
            with_track_summary: true,
            ..Default::default()
        }
        .start_export(&database, &no_cancel(), &mut no_progress())
        .expect("tracks export");
        let csv = std::fs::read_to_string(out_dir.join("tracks.csv")).unwrap();
        let lines: Vec<&str> = csv.lines().collect();
        assert_eq!(
            lines[0],
            "Image,Well,Class,Track,First T,Last T,Time points,Duration [time points],\
             Path length [µm],Distance moved [µm],Mean speed [px/time point],Straightness,\
             Touches edge,From first T,Area [px] mean"
        );
        let row: Vec<&str> = lines[1].split(',').collect();
        assert_eq!(row[0], "img1.tif");
        assert_eq!(
            row[2..],
            [
                "ClassA", "7", "0", "1", "2", "1", "0.005", "0.005", "5", "1", "FALSE", "TRUE",
                "20"
            ]
        );
        assert_eq!(lines.len(), 2, "the untracked object isn't a track");
    }

    /// A wide table per track would have a column per track: refused with
    /// a hint to export it long.
    #[test]
    fn a_wide_time_course_per_track_is_refused() {
        let (database, out_dir) =
            open_analysed(&[ObjectSpec::new("img1.tif", "ClassA", 1, 100)], 1);
        let export = |layout| {
            ResultExport {
                output_dir: out_dir.clone(),
                format: ExportFormat::CSV,
                columns: vec![Column::AreaSizePx],
                with_time_course: true,
                time_course: TimeCourseExport {
                    lines: TimeCourseLines::Track,
                    layout,
                    ..Default::default()
                },
                ..Default::default()
            }
            .start_export(&database, &no_cancel(), &mut no_progress())
        };
        let error = export(TimeCourseLayout::Both).expect_err("wide per track");
        assert!(error.to_string().contains("export it long"), "{error}");
        export(TimeCourseLayout::Long).expect("long per track");
        assert!(out_dir.join("time_course.csv").exists());
    }

    /// A plate grid for every analysed time point - the one where nothing
    /// was found too.
    #[test]
    fn plate_grids_get_a_tab_for_an_empty_time_point_too() {
        let (database, out_dir) =
            open_analysed(&[ObjectSpec::new("A1_01.tif", "ClassA", 1, 10)], 2);
        ResultExport {
            output_dir: out_dir.clone(),
            format: ExportFormat::XLSX,
            columns: vec![Column::AreaSizePx],
            aggregations: vec![Aggregation::Avg],
            with_plate_view_heatmap: true,
            ..Default::default()
        }
        .start_export(&database, &no_cancel(), &mut no_progress())
        .expect("plate export");
        let workbook: calamine::Xlsx<_> =
            calamine::open_workbook(out_dir.join("plate.xlsx")).expect("open plate.xlsx");
        assert_eq!(workbook.sheet_names(), ["T0 ClassA", "T1 ClassA"]);
    }

    /// A plate grid per time point: a tab each, the time point first.
    #[test]
    fn plate_grids_get_a_tab_per_time_point() {
        let (database, out_dir) = open(&[
            ObjectSpec::new("A1_01.tif", "ClassA", 1, 10),
            ObjectSpec::new("A1_01.tif", "ClassA", 1, 30).at_plane(0, 1),
        ]);
        ResultExport {
            output_dir: out_dir.clone(),
            format: ExportFormat::XLSX,
            columns: vec![Column::AreaSizePx],
            aggregations: vec![Aggregation::Avg],
            with_plate_view_heatmap: true,
            ..Default::default()
        }
        .start_export(&database, &no_cancel(), &mut no_progress())
        .expect("plate export");
        let mut workbook: calamine::Xlsx<_> =
            calamine::open_workbook(out_dir.join("plate.xlsx")).expect("open plate.xlsx");
        assert_eq!(workbook.sheet_names(), ["T0 ClassA", "T1 ClassA"]);
        let t1 = workbook.worksheet_range("T1 ClassA").unwrap();
        assert_eq!(data_f64(t1.get_value((2, 1))), 30.0, "A1 at time point 1");
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
            with_plate_view_heatmap: true,
            ..Default::default()
        };
        export
            .start_export(&database, &no_cancel(), &mut no_progress())
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
                    grouping: Grouping::Auto,
                    aggregation: Aggregation::Avg,
                    object_class: ObjectClass::Global(1),
                    column: Column::AreaSizePx,
                    color_schema: ColorSchema::default(),
                    color_scale: ColorScale::default(),
                    plate_size: PlateSize::Auto,
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
            with_well_view_heatmap: true,
            ..Default::default()
        };
        export
            .start_export(&database, &no_cancel(), &mut no_progress())
            .expect("well export");

        let expected_by_well = database
            .get_wells_for_plate(
                &WellsBatchFilter {
                    plane: PlaneFilter {
                        z_stack: 0,
                        t_stack: 0,
                    },
                    grouping: Grouping::Auto,
                    aggregation: Aggregation::Avg,
                    object_class: ObjectClass::Global(1),
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
            .start_export(&database, &no_cancel(), &mut no_progress())
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
                    object_class: ObjectClass::Global(1),
                    column: Column::AreaSizePx,
                    color_schema: ColorSchema::default(),
                    color_scale: ColorScale::default(),
                    square_size: Some(50),
                },
                &View::Heatmap,
            )
            .expect("expected image heatmap grid");

        let mut workbook: calamine::Xlsx<_> =
            calamine::open_workbook(out_dir.join("heatmap_img1.xlsx"))
                .expect("open heatmap_img1.xlsx");
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
            with_plate_view_list: true,
            with_well_view_list: true,
            ..Default::default()
        };
        export
            .start_export(&database, &no_cancel(), &mut no_progress())
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

    /// Intensity columns are aggregable like every other measurement - they
    /// were once silently dropped from every grouped/grid document.
    #[test]
    fn start_export_includes_intensity_columns_in_grouped_and_plate_documents() {
        use super::super::test_support::CH0_INTENSITIES_JSON;
        let (database, out_dir) = open(&[
            ObjectSpec::new("A1_01.tif", "ClassA", 1, 10).with_intensities(CH0_INTENSITIES_JSON),
            ObjectSpec::new("A1_02.tif", "ClassA", 1, 20).with_intensities(CH0_INTENSITIES_JSON),
        ]);
        let export = ResultExport {
            output_dir: out_dir.clone(),
            format: ExportFormat::XLSX,
            columns: vec![Column::AreaSizePx, Column::IntensityAvg(0)],
            aggregations: vec![Aggregation::Sum],
            with_grouped_by_image_list: true,
            with_plate_view_list: true,
            with_plate_view_heatmap: true,
            ..Default::default()
        };
        export
            .start_export(&database, &no_cancel(), &mut no_progress())
            .expect("export");

        // Plate list: one value column per column x aggregation.
        let mut plate_wb: calamine::Xlsx<_> =
            calamine::open_workbook(out_dir.join("plate_list.xlsx")).expect("open plate_list.xlsx");
        let plate_range = plate_wb.worksheet_range("Plate").expect("Plate sheet");
        let header: Vec<String> = (0..plate_range.width() as u32)
            .filter_map(|c| match plate_range.get_value((0, c)) {
                Some(calamine::Data::String(s)) => Some(s.clone()),
                _ => None,
            })
            .collect();
        let intensity_col = header
            .iter()
            .position(|h| h.contains("Avg Intensity (Ch 0)"))
            .unwrap_or_else(|| panic!("no intensity column in the plate list: {header:?}"));
        assert_eq!(
            data_f64(plate_range.get_value((1, intensity_col as u32))),
            254.0,
            "A1: sum of both objects' mean intensity"
        );

        // Grouped by image: the intensity column per image.
        let mut grouped_wb: calamine::Xlsx<_> =
            calamine::open_workbook(out_dir.join("grouped_by_image.xlsx"))
                .expect("open grouped_by_image.xlsx");
        let sheet = grouped_wb.sheet_names()[0].clone();
        let grouped_range = grouped_wb.worksheet_range(&sheet).expect("grouped sheet");
        let header: Vec<String> = (0..grouped_range.width() as u32)
            .filter_map(|c| match grouped_range.get_value((0, c)) {
                Some(calamine::Data::String(s)) => Some(s.clone()),
                _ => None,
            })
            .collect();
        let intensity_col = header
            .iter()
            .position(|h| h.contains("Avg Intensity (Ch 0)"))
            .unwrap_or_else(|| panic!("no intensity column grouped by image: {header:?}"));
        assert_eq!(
            data_f64(grouped_range.get_value((1, intensity_col as u32))),
            127.0
        );

        // Plate heatmap: a grid per column, the intensity one included.
        let mut heatmap_wb: calamine::Xlsx<_> =
            calamine::open_workbook(out_dir.join("plate.xlsx")).expect("open plate.xlsx");
        let sheet = heatmap_wb.sheet_names()[0].clone();
        let heatmap_range = heatmap_wb.worksheet_range(&sheet).expect("heatmap sheet");
        let has_intensity_caption = heatmap_range.cells().any(|(_, _, value)| {
            matches!(value, calamine::Data::String(s) if s.contains("Avg Intensity (Ch 0)"))
        });
        assert!(
            has_intensity_caption,
            "no intensity grid in the plate heatmap"
        );
    }

    /// A well with an image but zero matching objects must still get a row
    /// in the flat-pivot list export (not vanish for having nothing to
    /// aggregate), with `"-"` in its value cell rather than a misleading `0`.
    #[test]
    fn start_export_xlsx_plate_list_writes_a_dash_for_a_well_with_no_objects() {
        let (database, out_dir) = open_with_empty_image(
            &[ObjectSpec::new("A1_01.tif", "ClassA", 1, 10)],
            "B2_01.tif",
        );
        let export = ResultExport {
            output_dir: out_dir.clone(),
            format: ExportFormat::XLSX,
            columns: vec![Column::AreaSizePx],
            aggregations: vec![Aggregation::Avg],
            with_plate_view_list: true,
            ..Default::default()
        };
        export
            .start_export(&database, &no_cancel(), &mut no_progress())
            .expect("flat-pivot export");

        let mut plate_wb: calamine::Xlsx<_> =
            calamine::open_workbook(out_dir.join("plate_list.xlsx")).expect("open plate_list.xlsx");
        let plate_range = plate_wb.worksheet_range("Plate").expect("Plate sheet");
        assert_eq!(
            plate_range.get_value((1, 0)),
            Some(&calamine::Data::String("A1".to_string()))
        );
        assert_eq!(data_f64(plate_range.get_value((1, 1))), 10.0);
        assert_eq!(
            plate_range.get_value((2, 0)),
            Some(&calamine::Data::String("B2".to_string())),
            "B2 has an image but no objects and must still get a row"
        );
        assert_eq!(
            plate_range.get_value((2, 1)),
            Some(&calamine::Data::String("-".to_string())),
            "B2 has no matching objects, so its cell must read \"-\", not 0"
        );
    }

    /// A well made up only of disabled images must still get a row (its
    /// images are real, just excluded from the statistics) - written as a
    /// dash like any other well with no matching data, not struck through:
    /// a well itself is never flagged disabled, only its individual images
    /// are (see `plate_groups_to_result`).
    #[test]
    fn start_export_xlsx_plate_list_still_includes_a_fully_disabled_well() {
        let (database, out_dir) = open(&[
            ObjectSpec::new("A1_01.tif", "ClassA", 1, 10),
            ObjectSpec::new("B2_01.tif", "ClassA", 1, 100),
        ]);
        database
            .enable_image("B2_01.tif", true)
            .expect("disable B2");
        let export = ResultExport {
            output_dir: out_dir.clone(),
            format: ExportFormat::XLSX,
            columns: vec![Column::AreaSizePx],
            aggregations: vec![Aggregation::Avg],
            with_plate_view_list: true,
            ..Default::default()
        };
        export
            .start_export(&database, &no_cancel(), &mut no_progress())
            .expect("flat-pivot export");

        let mut plate_wb: calamine::Xlsx<_> =
            calamine::open_workbook(out_dir.join("plate_list.xlsx")).expect("open plate_list.xlsx");
        let plate_range = plate_wb.worksheet_range("Plate").expect("Plate sheet");
        assert_eq!(
            plate_range.get_value((2, 0)),
            Some(&calamine::Data::String("B2".to_string())),
            "B2 must still appear even though its only image is disabled"
        );
        assert_eq!(
            plate_range.get_value((2, 1)),
            Some(&calamine::Data::String("-".to_string())),
            "B2's disabled image must not contribute to the average"
        );
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
            with_plate_view_heatmap: true,
            with_well_view_heatmap: true,
            with_plate_view_list: true,
            with_well_view_list: true,
            with_heatmap: true,
            ..Default::default()
        };
        export
            .start_export(&database, &no_cancel(), &mut no_progress())
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
            let bytes =
                std::fs::read(out_dir.join(name)).unwrap_or_else(|e| panic!("read {name}: {e}"));
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
            object_classes: vec![ObjectClass::Global(1)],
            square_size: Some(50),
            with_plate_view_heatmap: true,
            with_well_view_heatmap: true,
            with_plate_view_list: true,
            with_well_view_list: true,
            with_heatmap: true,
            ..Default::default()
        };
        export
            .start_export(&database, &no_cancel(), &mut no_progress())
            .expect("filtered export");

        for file in ["plate.xlsx", "well.xlsx", "heatmap_A1_01.xlsx"] {
            let workbook: calamine::Xlsx<_> = calamine::open_workbook(out_dir.join(file))
                .unwrap_or_else(|e| panic!("open {file}: {e}"));
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
                Column::ColocCount(ObjectClass::Global(2)),
            ],
            with_list_view: true,
            with_list_coloc_details: true,
            ..Default::default()
        };
        export
            .start_export(&database, &no_cancel(), &mut no_progress())
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
        let result = export.start_export(&database, &no_cancel(), &mut no_progress());
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
        let result = export.start_export(&database, &no_cancel(), &mut no_progress());
        assert!(result.is_err());
    }

    #[test]
    fn start_export_grouped_by_image_with_a_class_without_objects_gives_each_image_an_empty_row() {
        let (database, out_dir) = open(&[ObjectSpec::new("img1.tif", "ClassA", 1, 100)]);
        let export = ResultExport {
            output_dir: out_dir.clone(),
            format: ExportFormat::CSV,
            columns: vec![Column::AreaSizePx],
            aggregations: vec![Aggregation::Avg],
            // No class 99 object was ever seeded - a selection matching
            // zero objects, exercising `fetch_all_grouped_by_image_rows`'s
            // "no values at all" min/max fallback (as opposed to an
            // explicitly-empty `Some(vec![])` selection, which short-
            // circuits earlier). The analysed image still gets its row:
            // "none found" is a result, and there's no average to show.
            object_classes: vec![ObjectClass::Global(99)],
            with_grouped_by_image_list: true,
            ..Default::default()
        };
        export
            .start_export(&database, &no_cancel(), &mut no_progress())
            .expect("export with a non-matching filter should still succeed");
        let content =
            std::fs::read_to_string(out_dir.join("grouped_by_image.csv")).expect("read csv");
        let lines: Vec<&str> = content.lines().collect();
        assert_eq!(lines.len(), 2, "header plus the image's row: {content}");
        assert!(lines[1].starts_with("img1.tif,"), "{content}");
        assert!(
            lines[1].ends_with(','),
            "no average without objects: {content}"
        );
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
            disabled: false,
            any_disabled: false,
            failed: false,
            any_failed: false,
        };
        let int_cell = Cell {
            value: CellValue::Integer(7),
            bg_color: 0,
            alternating_color: false,
            search_key: None,
            disabled: false,
            any_disabled: false,
            failed: false,
            any_failed: false,
        };
        let string_cell = Cell {
            value: CellValue::String("x".to_string()),
            bg_color: 0,
            alternating_color: false,
            search_key: None,
            disabled: false,
            any_disabled: false,
            failed: false,
            any_failed: false,
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

    // -- cancellation -----------------------------------------------------

    #[test]
    fn start_export_returns_cancelled_immediately_when_already_cancelled() {
        let (database, out_dir) = open(&[ObjectSpec::new("img1.tif", "ClassA", 1, 100)]);
        let (z_stacks, t_stacks) = full_range();
        let export = ResultExport {
            output_dir: out_dir.clone(),
            format: ExportFormat::CSV,
            z_stacks,
            t_stacks,
            with_list_view: true,
            ..Default::default()
        };
        let cancel = AtomicBool::new(true);
        let result = export.start_export(&database, &cancel, &mut no_progress());
        assert!(matches!(result, Err(InternalErrors::Cancelled)));
        // Nothing should have been written - cancellation is checked before
        // even creating the output directory.
        assert!(!out_dir.join("list.csv").exists());
    }

    #[test]
    fn start_export_stops_partway_through_a_one_file_per_image_export_once_cancelled() {
        let (database, out_dir) = open(&[
            ObjectSpec::new("img1.tif", "ClassA", 1, 10),
            ObjectSpec::new("img2.tif", "ClassA", 1, 20),
            ObjectSpec::new("img3.tif", "ClassA", 1, 30),
        ]);
        let (z_stacks, t_stacks) = full_range();
        let export = ResultExport {
            output_dir: out_dir.clone(),
            format: ExportFormat::CSV,
            z_stacks,
            t_stacks,
            columns: vec![Column::AreaSizePx],
            with_list_view: true,
            with_list_one_file_per_image: true,
            ..Default::default()
        };
        let cancel = AtomicBool::new(false);
        // Cancel as soon as the very first image's document has been
        // written, from inside the progress callback - proves cancellation
        // is actually observed *during* the export, not just checked once
        // up front.
        let mut seen_images = 0usize;
        let result = export.start_export(&database, &cancel, &mut |_message, _current, _total| {
            seen_images += 1;
            cancel.store(true, Ordering::Relaxed);
        });
        assert!(matches!(result, Err(InternalErrors::Cancelled)));
        assert_eq!(
            seen_images, 1,
            "should stop right after the first image's progress callback"
        );
        assert!(
            out_dir.join("list_img1.csv").exists(),
            "the first image's file was already written"
        );
        assert!(
            !out_dir.join("list_img2.csv").exists(),
            "must not start the second image"
        );
    }

    #[test]
    fn cell_format_marks_a_failed_images_cell() {
        let cell = |failed, any_failed| Cell {
            value: CellValue::Float(1.0),
            bg_color: 0,
            alternating_color: false,
            search_key: None,
            disabled: false,
            any_disabled: false,
            failed,
            any_failed,
        };
        let plain = cell_format(&cell(false, false), None);

        assert_ne!(cell_format(&cell(true, false), None), plain);
        assert_ne!(cell_format(&cell(false, true), None), plain);
        assert_eq!(cell_format(&cell(false, false), None), plain);
    }

    #[test]
    fn plate_and_well_documents_are_written_even_without_any_class_column_combination() {
        let (database, out_dir) = open(&[ObjectSpec::new("img1.tif", "ClassA", 1, 100)]);
        let (z_stacks, t_stacks) = full_range();
        let export = ResultExport {
            output_dir: out_dir.clone(),
            format: ExportFormat::XLSX,
            z_stacks,
            t_stacks,
            columns: vec![Column::AreaSizePx],
            // No aggregation: nothing to compute per class and column.
            aggregations: Vec::new(),
            with_plate_view_heatmap: true,
            with_well_view_heatmap: true,
            with_plate_view_list: true,
            with_well_view_list: true,
            ..Default::default()
        };

        export
            .start_export(&database, &no_cancel(), &mut no_progress())
            .expect("xlsx export without combinations");

        for name in [
            "plate.xlsx",
            "well.xlsx",
            "plate_list.xlsx",
            "well_list.xlsx",
        ] {
            assert!(out_dir.join(name).is_file(), "{name} missing");
        }
    }

    #[test]
    fn plate_and_well_exports_do_nothing_when_none_of_their_views_is_wanted() {
        let (database, out_dir) = open(&[ObjectSpec::new("img1.tif", "ClassA", 1, 100)]);
        let export = ResultExport {
            output_dir: out_dir.clone(),
            format: ExportFormat::XLSX,
            ..Default::default()
        };

        export
            .export_plate_and_well(&database, &no_cancel(), &mut no_progress())
            .unwrap();
        export
            .export_plate_and_well_as_flat_list(&database, &no_cancel(), &mut no_progress())
            .unwrap();

        assert!(!out_dir.exists(), "no document written");
    }

    #[test]
    fn listing_no_images_never_queries_a_page() {
        let (database, _) = open(&[ObjectSpec::new("img1.tif", "ClassA", 1, 100)]);
        let (z_stacks, t_stacks) = full_range();
        let export = ResultExport {
            z_stacks,
            t_stacks,
            columns: vec![Column::AreaSizePx],
            ..Default::default()
        };
        let mut pages = 0;

        stream_list_pages(
            &database,
            &no_cancel(),
            &export,
            &[],
            &None,
            false,
            &ExportPlanes::of(&database, &export).unwrap(),
            |_, _, _| {
                pages += 1;
                Ok(())
            },
        )
        .unwrap();

        assert_eq!(pages, 0);
    }

    #[test]
    fn grouped_rows_without_values_report_a_zero_range_instead_of_infinities() {
        let (database, _) = open(&[ObjectSpec::new("img1.tif", "ClassA", 1, 100)]);
        let filter = GroupedByImageFilter {
            plane: PlaneFilter {
                z_stack: 0,
                t_stack: 0,
            },
            images: Some(vec!["not-in-the-database.tif".into()]),
            object_classes: None,
            columns: vec![Column::AreaSizePx],
            aggregation: vec![Aggregation::Avg],
            page: Pagination {
                limit: 1,
                after: None,
            },
            transpond_table: false,
        };

        let merged = fetch_all_grouped_by_image_rows(&database, &filter).unwrap();

        assert!(merged.rows.is_empty());
        assert_eq!((merged.min, merged.max), (0.0, 0.0));
    }

    #[test]
    fn a_csv_document_that_cannot_be_created_is_an_error_naming_it() {
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("a-file");
        std::fs::write(&file, b"").unwrap();

        // Its folder would have to be inside a file.
        let Err(e) = create_csv_writer(&file.join("list.csv")) else {
            panic!("expected an error");
        };
        assert!(e.to_string().contains("could not create"), "{e}");

        // The document's own path is a folder.
        let Err(e) = create_csv_writer(dir.path()) else {
            panic!("expected an error");
        };
        assert!(e.to_string().contains("could not create"), "{e}");
    }

    // -- measurements -----------------------------------------------------

    /// `img1.tif`'s two time points with an image measurement each, and one
    /// object's value; `img2.tif` with a value too.
    fn with_measurements() -> (ResultsGenerator, PathBuf, String) {
        let (generator, out_dir) = open(&[
            ObjectSpec::new("img1.tif", "ClassA", 1, 10),
            ObjectSpec::new("img2.tif", "ClassA", 1, 20),
        ]);
        let object: String = generator
            .connection()
            .query_row(
                "SELECT object_id::VARCHAR FROM objects WHERE area_px = 10",
                [],
                |row| row.get(0),
            )
            .unwrap();
        generator
            .connection()
            .execute_batch(&format!(
                "CREATE TABLE measurement_info (key VARCHAR PRIMARY KEY, measurement VARCHAR, \
                 step VARCHAR, unit VARCHAR, target VARCHAR, range_min DOUBLE, \
                 range_max DOUBLE, aggregatable BOOLEAN, object_index INTEGER);
                 CREATE TABLE measurements (image_name VARCHAR, image_rel_path VARCHAR, \
                 z_stack INTEGER, t_stack INTEGER, region VARCHAR, object_id UUID, \
                 related_object_id UUID, object_class_name VARCHAR, object_class_id INTEGER[], \
                 step VARCHAR, measurement VARCHAR, unit VARCHAR, key VARCHAR, value DOUBLE);
                 INSERT INTO measurement_info VALUES
                     ('p_01/p', 'Pearson', 'p_01', 'None', 'Image', -1, 1, true, NULL),
                     ('q_01/q', 'Pearson', 'q_01', 'None', 'Object', -1, 1, true, 1);
                 INSERT INTO measurements VALUES
                     ('img1.tif', 'img1.tif', 0, 1, 'image', NULL, NULL, NULL, NULL, 'p_01', 'Pearson', 'None', 'p_01/p', NULL),
                     ('img1.tif', 'img1.tif', 0, 0, 'image', NULL, NULL, NULL, NULL, 'p_01', 'Pearson', 'None', 'p_01/p', 0.5),
                     ('img1.tif', 'img1.tif', 0, 0, 'object', '{object}', '{object}', '[\"ClassA\"]', [1], 'q_01', 'Pearson', 'None', 'q_01/q', 0.25),
                     ('img2.tif', 'img2.tif', 0, 0, 'image', NULL, NULL, NULL, NULL, 'p_01', 'Pearson', 'None', 'p_01/p', 0.75);
                 UPDATE objects SET measurement_values = [0.25] WHERE object_id = '{object}';"
            ))
            .unwrap();
        (generator, out_dir, object)
    }

    /// One row per value, the columns named like the database's; a value
    /// that isn't there stays empty.
    #[test]
    fn the_measurements_export_has_a_row_per_value() {
        let (generator, out_dir, object) = with_measurements();
        ResultExport {
            output_dir: out_dir.clone(),
            format: ExportFormat::CSV,
            with_measurements: true,
            ..Default::default()
        }
        .start_export(&generator, &no_cancel(), &mut no_progress())
        .expect("export");
        let csv = std::fs::read_to_string(out_dir.join("measurements.csv")).unwrap();
        let lines: Vec<&str> = csv.lines().collect();
        assert_eq!(
            lines,
            [
                "image_name,image_rel_path,z_stack,t_stack,region,object_id,related_object_id,\
                 object_class_name,step,measurement,unit,key,value",
                "img1.tif,img1.tif,0,0,image,,,,p_01,Pearson,None,p_01/p,0.5",
                &format!(
                    "img1.tif,img1.tif,0,0,object,{object},{object},\"[\"\"ClassA\"\"]\",q_01,Pearson,None,q_01/q,0.25"
                ),
                "img1.tif,img1.tif,0,1,image,,,,p_01,Pearson,None,p_01/p,",
                "img2.tif,img2.tif,0,0,image,,,,p_01,Pearson,None,p_01/p,0.75",
            ]
        );
    }

    /// The measurements chosen among the columns, of the chosen images and
    /// time points.
    #[test]
    fn the_measurements_export_keeps_to_the_chosen_ones() {
        let (generator, out_dir, _) = with_measurements();
        let p = generator
            .get_measurement_columns()
            .unwrap()
            .into_iter()
            .find(|m| m.key == "p_01/p")
            .unwrap();
        ResultExport {
            output_dir: out_dir.clone(),
            format: ExportFormat::CSV,
            with_measurements: true,
            columns: vec![Column::AreaSizePx, Column::Measurement(p)],
            image_rel_paths: vec!["img1.tif".into()],
            t_stacks: Some(Range { start: 0, end: 1 }),
            ..Default::default()
        }
        .start_export(&generator, &no_cancel(), &mut no_progress())
        .expect("export");
        let csv = std::fs::read_to_string(out_dir.join("measurements.csv")).unwrap();
        assert_eq!(
            csv.lines().skip(1).collect::<Vec<_>>(),
            ["img1.tif,img1.tif,0,0,image,,,,p_01,Pearson,None,p_01/p,0.5"]
        );
    }

    #[test]
    fn results_without_measurements_say_so_and_parquet_writes_them_too() {
        let (generator, out_dir) = open(&[ObjectSpec::new("img1.tif", "ClassA", 1, 10)]);
        let err = ResultExport {
            output_dir: out_dir.clone(),
            format: ExportFormat::CSV,
            with_measurements: true,
            ..Default::default()
        }
        .start_export(&generator, &no_cancel(), &mut no_progress())
        .err()
        .expect("refused")
        .to_string();
        assert!(err.contains("no measurements"), "{err}");

        let (generator, out_dir, _) = with_measurements();
        ResultExport {
            output_dir: out_dir.clone(),
            format: ExportFormat::Parquet,
            ..Default::default()
        }
        .start_export(&generator, &no_cancel(), &mut no_progress())
        .expect("export");
        let rows: i64 = generator
            .connection()
            .query_row(
                &format!(
                    "SELECT count(*) FROM '{}'",
                    out_dir.join("measurements.parquet").display()
                ),
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(rows, 4);
    }
}
