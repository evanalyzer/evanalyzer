//! # series
//!
//! Values over time and over a radius: the time course, the profiles and
//! the track summary.
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-11-22
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.

use super::*;

impl ResultExport {
    pub(super) fn export_track_summary(
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
                let worksheet = workbook
                    .add_worksheet_with_constant_memory()
                    .map_err(xlsx_err)?;
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

    pub(super) fn export_time_course(
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
            history: settings.history,
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
                    let worksheet = workbook
                        .add_worksheet_with_constant_memory()
                        .map_err(xlsx_err)?;
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

    /// A profile measurement (`ProfileExport`, e.g. the Sholl
    /// intersections), of the chosen images and classes:
    /// - the chart's lines: `profile_lines.csv` (long: a row per line ×
    ///   position with n, mean, SD, median, Q1, Q3) and/or
    ///   `profile_lines_<statistic>.csv` (wide: a row per position, a column
    ///   per line); XLSX: a sheet each,
    /// - each object's profile: `profile_objects.csv` (long: a row per
    ///   object × position, streamed) and/or `profile_objects_wide.csv` (a
    ///   row per object, a column per position, a position it doesn't reach
    ///   as 0 - as in the chart).
    pub(super) fn export_profile(
        &self,
        database: &ResultsGenerator,
        cancel: &AtomicBool,
        on_progress: ExportProgress,
    ) -> Result<(), InternalErrors> {
        check_cancelled(cancel)?;
        on_progress("Exporting Profile", 0, 1);
        let settings = &self.profile;
        if settings.key.is_empty() {
            return Err(InternalErrors::InvalidArgument(
                "Choose the profile measurement to export".to_string(),
            ));
        }
        let images = (!self.image_rel_paths.is_empty()).then(|| self.image_rel_paths.clone());
        let classes = (!self.object_classes.is_empty()).then(|| self.object_classes.clone());
        let write_long = settings.layout != TimeCourseLayout::Wide;
        let write_wide = settings.layout != TimeCourseLayout::Long;
        let mut tables: Vec<(String, Fields)> = Vec::new();
        let mut long_rows: Option<Fields> = None;
        match settings.content {
            ProfileExportContent::Lines => {
                let filter = ProfileFilter {
                    key: settings.key.clone(),
                    lines: settings.lines,
                    grouping: self.grouping.clone(),
                    images,
                    object_classes: classes,
                    max_lines: usize::MAX,
                };
                let result = super::super::results_profile::profile(database, &filter)?;
                let label = |p: &ProfilePoint| match settings.lines {
                    ProfileLines::Class => p.class_name.clone(),
                    ProfileLines::All => "all".to_string(),
                    _ => p.line.clone(),
                };
                if write_long {
                    let mut rows: Fields = vec![
                        [
                            "line",
                            "position",
                            "n_objects",
                            "mean",
                            "sd",
                            "median",
                            "q1",
                            "q3",
                        ]
                        .map(|h| Field::Text(h.to_string()))
                        .to_vec(),
                    ];
                    for p in &result.points {
                        let opt = |v: Option<f64>| v.map_or(Field::Empty, Field::Number);
                        rows.push(vec![
                            Field::Text(label(p)),
                            Field::Number(p.position),
                            Field::Number(p.n_objects as f64),
                            opt(p.mean),
                            opt(p.sd),
                            opt(p.median),
                            opt(p.q1),
                            opt(p.q3),
                        ]);
                    }
                    long_rows = Some(rows);
                }
                if write_wide {
                    let mut lines: Vec<String> = Vec::new();
                    let mut positions: Vec<f64> = Vec::new();
                    for p in &result.points {
                        if !lines.contains(&label(p)) {
                            lines.push(label(p));
                        }
                        if !positions.contains(&p.position) {
                            positions.push(p.position);
                        }
                    }
                    positions.sort_by(f64::total_cmp);
                    type Stat = fn(&ProfilePoint) -> Option<f64>;
                    let stats: [(&str, Stat); 6] = [
                        ("mean", |p| p.mean),
                        ("sd", |p| p.sd),
                        ("median", |p| p.median),
                        ("q1", |p| p.q1),
                        ("q3", |p| p.q3),
                        ("n_objects", |p| Some(p.n_objects as f64)),
                    ];
                    for (name, stat) in stats {
                        let mut rows: Fields = vec![
                            std::iter::once(Field::Text("position".to_string()))
                                .chain(lines.iter().map(|l| Field::Text(l.clone())))
                                .collect(),
                        ];
                        for &position in &positions {
                            let mut row = vec![Field::Number(position)];
                            for line in &lines {
                                let value = result
                                    .points
                                    .iter()
                                    .find(|p| p.position == position && label(p) == *line)
                                    .and_then(stat);
                                row.push(value.map_or(Field::Empty, Field::Number));
                            }
                            rows.push(row);
                        }
                        tables.push((format!("profile_lines_{name}"), rows));
                    }
                }
                self.write_profile_tables("profile_lines", long_rows, tables)?;
            }
            ProfileExportContent::Objects => {
                self.export_profile_objects(
                    database, images, classes, write_long, write_wide, cancel,
                )?;
            }
        }
        on_progress("Exporting Profile", 1, 1);
        Ok(())
    }

    /// Each object's profile (see `export_profile`): long streamed, wide
    /// a row per object.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn export_profile_objects(
        &self,
        database: &ResultsGenerator,
        images: Option<Vec<String>>,
        classes: Option<Vec<ObjectClass>>,
        write_long: bool,
        write_wide: bool,
        cancel: &AtomicBool,
    ) -> Result<(), InternalErrors> {
        let err = |e: duckdb::Error| InternalErrors::Io(e.to_string());
        if !database.has_column("measurements", "position")? {
            return Err(InternalErrors::InvalidArgument(
                "These results hold no profiles (written before format 2)".to_string(),
            ));
        }
        let mut conditions = vec![
            format!("m.key = '{}'", self.profile.key.replace('\'', "''")),
            "m.region = 'object'".to_string(),
            "m.position IS NOT NULL".to_string(),
            "NOT i.disabled".to_string(),
            "i.successful".to_string(),
        ];
        if let Some(images) = &images {
            conditions.push(format!(
                "m.image_rel_path IN ({})",
                sql_string_in_list(images)
            ));
        }
        if let Some(classes) = &classes {
            let ids: Vec<u32> = classes
                .iter()
                .filter_map(|c| match c {
                    ObjectClass::Global(n) => Some(*n),
                    _ => None,
                })
                .collect();
            conditions.push(class_filter_sql("m.object_class_id", &ids));
        }
        let sql = format!(
            "SELECT m.image_name, m.image_rel_path, m.object_id::VARCHAR, m.object_class_name, \
             m.position, m.value FROM measurements m \
             JOIN images i ON i.image_rel_path = m.image_rel_path WHERE {} \
             ORDER BY m.image_name, m.object_id, m.position",
            conditions.join(" AND ")
        );
        let mut stmt = database.connection().prepare(&sql).map_err(err)?;
        let header = [
            "image_name",
            "image_rel_path",
            "object_id",
            "object_class_name",
        ];
        // Wide: per object its values by position.
        let mut wide: Vec<([String; 4], Vec<(f64, f64)>)> = Vec::new();
        let mut each_row = |out: &mut dyn FieldSink| -> Result<(), InternalErrors> {
            for name in header.iter().chain(&["position", "value"]) {
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
                let text = |i: usize| -> Result<String, InternalErrors> {
                    Ok(row
                        .get::<_, Option<String>>(i)
                        .map_err(err)?
                        .unwrap_or_default())
                };
                let id = [text(0)?, text(1)?, text(2)?, text(3)?];
                let position: f64 = row.get(4).map_err(err)?;
                let value: Option<f64> = row.get(5).map_err(err)?;
                for field in &id {
                    out.text(field)?;
                }
                out.number(position)?;
                out.optional(value)?;
                out.end_row()?;
                if write_wide {
                    match wide.last_mut() {
                        Some((last, values)) if *last == id => {
                            values.push((position, value.unwrap_or(0.0)))
                        }
                        _ => wide.push((id, vec![(position, value.unwrap_or(0.0))])),
                    }
                }
            }
            Ok(())
        };
        let long_name = "profile_objects";
        match self.format {
            ExportFormat::XLSX => {
                let mut workbook = Workbook::new();
                let mut names = SheetNamer::new();
                if write_long {
                    let worksheet = workbook
                        .add_worksheet_with_constant_memory()
                        .map_err(xlsx_err)?;
                    worksheet.set_name(names.unique("Long")).map_err(xlsx_err)?;
                    each_row(&mut FieldsSheet::new(worksheet))?;
                } else {
                    each_row(&mut Discard)?;
                }
                if write_wide {
                    let sheet = workbook.add_worksheet();
                    sheet.set_name(names.unique("Wide")).map_err(xlsx_err)?;
                    write_fields_sheet(sheet, &profile_objects_wide(&header, &wide))?;
                }
                workbook
                    .save(self.output_path(&format!("{long_name}.xlsx")))
                    .map_err(xlsx_err)?;
            }
            ExportFormat::CSV => {
                if write_long {
                    each_row(&mut FieldsCsv::create(
                        &self.output_path(&format!("{long_name}.csv")),
                    )?)?;
                } else {
                    each_row(&mut Discard)?;
                }
                if write_wide {
                    write_fields_csv(
                        &profile_objects_wide(&header, &wide),
                        &self.output_path(&format!("{long_name}_wide.csv")),
                    )?;
                }
            }
            // Written with the objects (`export_as_parquet`): the raw
            // `measurements` table, its `position` column included.
            ExportFormat::Parquet => {}
        }
        Ok(())
    }

    /// The profile's line tables: `name.csv` (long) and a CSV per wide
    /// table, or one workbook with a sheet each.
    pub(super) fn write_profile_tables(
        &self,
        name: &str,
        long: Option<Fields>,
        wide: Vec<(String, Fields)>,
    ) -> Result<(), InternalErrors> {
        match self.format {
            ExportFormat::XLSX => {
                let mut workbook = Workbook::new();
                let mut names = SheetNamer::new();
                if let Some(long) = &long {
                    let sheet = workbook.add_worksheet();
                    sheet.set_name(names.unique("Long")).map_err(xlsx_err)?;
                    write_fields_sheet(sheet, long)?;
                }
                for (table, rows) in &wide {
                    let sheet = workbook.add_worksheet();
                    let short = table.trim_start_matches("profile_lines_");
                    sheet.set_name(names.unique(short)).map_err(xlsx_err)?;
                    write_fields_sheet(sheet, rows)?;
                }
                workbook
                    .save(self.output_path(&format!("{name}.xlsx")))
                    .map_err(xlsx_err)?;
            }
            ExportFormat::CSV => {
                if let Some(long) = &long {
                    write_fields_csv(long, &self.output_path(&format!("{name}.csv")))?;
                }
                for (table, rows) in &wide {
                    write_fields_csv(rows, &self.output_path(&format!("{table}.csv")))?;
                }
            }
            ExportFormat::Parquet => {}
        }
        Ok(())
    }
}

/// The T, Z (several z-planes) and time (a frame interval) columns - the
/// time unit (seconds, minutes or hours, by how long the series runs) found
/// once per table.
pub(super) struct TimeAxis {
    pub(super) by_z: bool,
    pub(super) unit: Option<(&'static str, f64)>,
}

impl TimeAxis {
    pub(super) fn new(head: &TimeCourseHead) -> Self {
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

    pub(super) fn header(&self, out: &mut dyn FieldSink) -> Result<(), InternalErrors> {
        out.text("T")?;
        if self.by_z {
            out.text("Z")?;
        }
        if let Some((label, _)) = self.unit {
            out.text(label)?;
        }
        Ok(())
    }

    pub(super) fn fields(
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

pub(super) fn optional(value: Option<f64>) -> Field {
    value.map_or(Field::Empty, Field::Number)
}

/// The long table: a row per time point (z-plane) × line × class - the
/// line's own columns, the counts, then the values. Written a row at a time
/// as the points come (a per-track time course has millions).
pub(super) struct TimeCourseLong<'a> {
    pub(super) head: &'a TimeCourseHead,
    pub(super) lines: TimeCourseLines,
    pub(super) axis: TimeAxis,
}

impl<'a> TimeCourseLong<'a> {
    pub(super) fn new(head: &'a TimeCourseHead, lines: TimeCourseLines) -> Self {
        Self {
            head,
            lines,
            axis: TimeAxis::new(head),
        }
    }

    pub(super) fn header(&self, out: &mut dyn FieldSink) -> Result<(), InternalErrors> {
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

    pub(super) fn row(
        &self,
        point: &TimeCoursePoint,
        out: &mut dyn FieldSink,
    ) -> Result<(), InternalErrors> {
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
/// duration), how it moved (Measure Track Motion's values, only with that
/// step), whether it touches the edge or is there from the start, then the
/// measurements. Lengths in µm with a pixel size, else px; durations in
/// minutes with a frame interval, else time points; speeds (and the other
/// values per time) in the unit asked for when it can be.
pub(super) struct TrackColumns<'a> {
    pub(super) head: &'a TrackSummaryHead,
    pub(super) speed_unit: SpeedUnit,
}

impl<'a> TrackColumns<'a> {
    pub(super) fn new(head: &'a TrackSummaryHead, speed_unit: SpeedUnit) -> Self {
        Self {
            head,
            speed_unit: match head.um && head.timed {
                true => speed_unit,
                false => SpeedUnit::PxPerFrame,
            },
        }
    }

    pub(super) fn length_unit(&self) -> &'static str {
        if self.head.um { "µm" } else { "px" }
    }

    pub(super) fn header(&self, out: &mut dyn FieldSink) -> Result<(), InternalErrors> {
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
        for column in ["Class", "Track"] {
            out.text(column)?;
        }
        if self.head.lineage {
            for column in LINEAGE_COLUMNS {
                out.text(column)?;
            }
        }
        for column in ["First T", "Last T", "Time points"] {
            out.text(column)?;
        }
        out.text(if self.head.timed {
            "Duration [min]"
        } else {
            "Duration [time points]"
        })?;
        if self.head.motion {
            let time = match self.speed_unit {
                SpeedUnit::PxPerFrame => "time point",
                SpeedUnit::UmPerMin => "min",
                SpeedUnit::UmPerS => "s",
            };
            let area = if self.head.um && self.speed_unit != SpeedUnit::PxPerFrame {
                "µm²"
            } else {
                "px²"
            };
            for (id, _, _) in TrackMotion::VALUES {
                out.text(&match id {
                    "path_length" => format!("Path length [{length}]"),
                    "displacement" => format!("Distance moved [{length}]"),
                    "max_distance" => format!("Max distance from start [{length}]"),
                    "vcl" => format!("VCL (mean speed) [{speed}]"),
                    "vsl" => format!("VSL [{speed}]"),
                    "vap" => format!("VAP [{speed}]"),
                    "lin" => "LIN (straightness)".to_string(),
                    "str" => "STR".to_string(),
                    "wob" => "WOB".to_string(),
                    "alh" => format!("ALH [{length}]"),
                    "bcf" => format!("BCF [1/{time}]"),
                    "turning" => "Turning angle mean [°]".to_string(),
                    "turning_abs" => "Turning angle mean absolute [°]".to_string(),
                    "angular_speed" => format!("Angular speed [°/{time}]"),
                    "msd_alpha" => "MSD alpha".to_string(),
                    "msd_d" => format!("MSD D [{area}/{time}]"),
                    other => other.to_string(),
                })?;
            }
        }
        for column in ["Touches edge", "From first T"] {
            out.text(column)?;
        }
        for name in &self.head.value_names {
            out.text(name)?;
        }
        out.end_row()
    }

    pub(super) fn row(
        &self,
        track: &TrackSummary,
        out: &mut dyn FieldSink,
    ) -> Result<(), InternalErrors> {
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
        if self.head.lineage {
            let lineage = &track.lineage;
            out.text(&id_list(&lineage.parents))?;
            out.text(&id_list(&lineage.children))?;
            out.number(lineage.root as f64)?;
            out.number(f64::from(lineage.generation))?;
            out.text(lineage.starts_with().label())?;
            out.text(lineage.ends_with().label())?;
        }
        out.number(f64::from(track.first_t))?;
        out.number(f64::from(track.last_t))?;
        out.number(f64::from(track.points))?;
        match track.s_per_frame.filter(|_| self.head.timed) {
            Some(s) => out.number(frames * s / 60.0)?,
            None => out.number(frames)?,
        }
        if self.head.motion {
            for (i, (_, _, kind)) in TrackMotion::VALUES.iter().enumerate() {
                out.optional(match kind {
                    // Lengths in µm whenever the pixel size is known.
                    MotionKind::Length => track.motion.get(i).map(length),
                    _ => track.motion.in_units(
                        i,
                        self.speed_unit,
                        track.um_per_px,
                        track.s_per_frame,
                    ),
                })?;
            }
        }
        out.text(if track.touches_edge { "TRUE" } else { "FALSE" })?;
        out.text(if track.from_start { "TRUE" } else { "FALSE" })?;
        for value in &track.values {
            out.optional(*value)?;
        }
        out.end_row()
    }
}

/// The family tree columns after "Track" (results with divisions or
/// fusions): the parents are the tree's edges - with each track's first
/// and last T enough to draw it. "Lineage": the tree's root track.
pub(super) const LINEAGE_COLUMNS: [&str; 6] = [
    "Parent tracks",
    "Child tracks",
    "Lineage",
    "Generation",
    "Starts with",
    "Ends with",
];

/// Track ids as "3;7" - not "," (the CSV's separator); empty for none.
pub(super) fn id_list(ids: &[u64]) -> String {
    ids.iter().map(u64::to_string).collect::<Vec<_>>().join(";")
}

/// The wide tables, one per value and statistic (named like it): a row per
/// time point (z-plane), a column per line ("A1 · Nuclei"; the class alone
/// for class lines, without it when there's one class).
pub(super) fn time_course_wide(
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

/// Each object's profile wide: a row per object, a column per position
/// (all the objects' positions, by value; one an object doesn't reach: 0).
pub(super) fn profile_objects_wide(
    header: &[&str; 4],
    objects: &[([String; 4], Vec<(f64, f64)>)],
) -> Fields {
    let mut positions: Vec<f64> = objects
        .iter()
        .flat_map(|(_, values)| values.iter().map(|v| v.0))
        .collect();
    positions.sort_by(f64::total_cmp);
    positions.dedup();
    let mut rows: Fields = vec![
        header
            .iter()
            .map(|h| Field::Text(h.to_string()))
            .chain(positions.iter().map(|p| Field::Text(format!("r_{p}"))))
            .collect(),
    ];
    for (id, values) in objects {
        let mut row: Vec<Field> = id.iter().map(|t| Field::Text(t.clone())).collect();
        for &position in &positions {
            let value = values.iter().find(|v| v.0 == position).map_or(0.0, |v| v.1);
            row.push(Field::Number(value));
        }
        rows.push(row);
    }
    rows
}
