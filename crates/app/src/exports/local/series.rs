//! # series
//!
//! Values over time and over a position in table format 1: a `TABLE` of
//! `TIME_POINT` rows (the time course: per time point and line, the values
//! summed up over the line's objects there) and of `PROFILE_POSITION` rows
//! (a profile, e.g. the Sholl intersections per radius: per line and
//! position its statistics, or each object's values).
//!
//! The time course comes point by point (`ResultsSource::time_course_each`:
//! a course per track has millions); its wide tables (a column per line)
//! keep the points of one table only.
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-11-24
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.

use super::Output;
use super::table::{Out, Part, Rows, Sink, column_name, scope_images, unit_label};
use super::tracks::{statistic, suffix};
use crate::exports::naming;
use crate::exports::types::EntryReport;
use crate::results::view_settings::resolve_columns;
use crate::results::{
    Column, Grouping, MeasurementRowsFilter, ProfileFilter, ProfileLines, ResultsSource,
    TimeCourseFilter, TimeCourseLines, TimeCourseNormalize, TimeCoursePoint, TrackHistory,
};
use evanalyzer_cfg::core_types::{InternalErrors, ObjectClass};
use evanalyzer_cfg::settings::classification_settings::Class;
use evanalyzer_cfg::settings::export_settings::{
    ExportSettings, LengthUnit, ProfileContent, ProfileLineKind, TableExport, TableFileFormat,
    TimeCourseLineKind, TimeCourseOptions, TimeCourseTableLayout,
};
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicBool, Ordering};

/// Rows per measurement query page (a profile's object values).
const PAGE_SIZE: u32 = 100_000;

fn classes_of(table: &TableExport) -> Option<Vec<ObjectClass>> {
    table
        .classes
        .as_ref()
        .map(|ids| ids.iter().map(|id| ObjectClass::Global(*id)).collect())
}

/// A value column of a time course: (format 1 name, label, unit, factor).
struct ValueColumn {
    name: String,
    label: String,
    unit: String,
    factor: f64,
}

pub(super) struct TimeCourseRows {
    images: Vec<String>,
    classes: Option<Vec<ObjectClass>>,
    options: TimeCourseOptions,
    columns: Vec<Column>,
    statistics: Vec<crate::results::TimeCourseStat>,
    /// Per column and statistic, in the query's order.
    values: Vec<ValueColumn>,
    several_z: bool,
    planes: evanalyzer_cfg::settings::export_settings::Planes,
}

impl TimeCourseRows {
    pub(super) fn new(
        results: &dyn ResultsSource,
        export: &ExportSettings,
        table: &TableExport,
        notes: &mut Vec<String>,
    ) -> Result<Self, InternalErrors> {
        let db_classes = results.get_object_classes()?;
        let mut columns = match &table.columns {
            None => crate::exports::default_columns(table.rows),
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
        // As the time course sums up: values of the objects (not the count,
        // not an image's own measurements).
        let left_out: Vec<String> = columns
            .iter()
            .filter(|c| !c.is_aggregable() || **c == Column::Count || c.is_image_measurement())
            .map(|c| c.as_key(&db_classes))
            .collect();
        if !left_out.is_empty() {
            notes.push(format!(
                "a time course sums up the objects' values - left out: {}",
                left_out.join(", ")
            ));
        }
        columns.retain(|c| c.is_aggregable() && *c != Column::Count && !c.is_image_measurement());
        columns.sort();
        columns.dedup();
        let pairs: Vec<_> = table
            .aggregations
            .iter()
            .filter_map(|a| statistic(*a).map(|s| (s, suffix(*a))))
            .collect();
        if pairs.len() < table.aggregations.len() {
            notes.push("a time course has no skewness - left out".into());
        }
        let options = table.time_course.clone().unwrap_or_default();
        let values = values_of(
            &columns,
            &pairs,
            export.length_unit,
            &db_classes,
            options.relative,
        );
        Ok(Self {
            images: scope_images(results, table.images)?,
            classes: classes_of(table),
            statistics: pairs.iter().map(|(s, _)| *s).collect(),
            columns,
            values,
            options,
            several_z: results.get_nr_of_z_stacks() > 1,
            planes: table.planes,
        })
    }

    pub(super) fn layout(&self) -> TimeCourseTableLayout {
        self.options.layout
    }

    fn filter(&self, part: &Part) -> TimeCourseFilter {
        TimeCourseFilter {
            lines: match self.options.lines {
                TimeCourseLineKind::Class => TimeCourseLines::Class,
                TimeCourseLineKind::Image => TimeCourseLines::Image,
                TimeCourseLineKind::Well => TimeCourseLines::Well,
                TimeCourseLineKind::Track => TimeCourseLines::Track,
            },
            grouping: Grouping::Auto,
            columns: self.columns.clone(),
            statistics: self.statistics.clone(),
            normalize: if self.options.relative {
                TimeCourseNormalize::RelativeToStart
            } else {
                TimeCourseNormalize::None
            },
            images: Some(part.images.clone()),
            object_classes: part.classes.clone(),
            t_stacks: super::plane_range(self.planes.t),
            z_stacks: super::plane_range(self.planes.z),
            max_tracks: None,
            history: if self.options.with_parents {
                TrackHistory::WithParents
            } else {
                TrackHistory::Own
            },
        }
    }

    /// Where a row belongs: (format 1 name, label).
    fn line_columns(&self) -> Vec<(&'static str, &'static str)> {
        let mut columns = match self.options.lines {
            TimeCourseLineKind::Class => vec![],
            TimeCourseLineKind::Image => vec![("image", "Image (relative path)")],
            TimeCourseLineKind::Well => vec![("well", "Well")],
            TimeCourseLineKind::Track => {
                vec![("track_id", "Track"), ("image", "Image (relative path)")]
            }
        };
        columns.push(("class", "Class"));
        columns.push(("t", "Time point (from 0)"));
        if self.several_z {
            columns.push(("z", "Z-plane (from 0)"));
        }
        columns.extend([
            (
                "time_s",
                "Time since the first time point (s; empty without a frame interval)",
            ),
            ("n_images", "Images of the line at this time point"),
            ("n_objects", "Objects of the line at this time point"),
            ("n_tracks", "Tracks among them (0 without tracking)"),
        ]);
        columns
    }

    fn line_cells(&self, point: &TimeCoursePoint, out: &mut Vec<Out>) {
        let text = |s: &str| Out::Text(s.replace('\\', "/"));
        match self.options.lines {
            TimeCourseLineKind::Class => {}
            TimeCourseLineKind::Image => out.push(text(&point.image)),
            TimeCourseLineKind::Well => out.push(text(&point.line)),
            TimeCourseLineKind::Track => {
                out.push(
                    point
                        .line
                        .trim_start_matches('#')
                        .parse::<f64>()
                        .map_or_else(|_| text(&point.line), Out::Number),
                );
                out.push(text(&point.image));
            }
        }
        out.push(Out::Text(point.class_name.clone()));
        out.push(Out::Number(f64::from(point.t)));
        if self.several_z {
            out.push(Out::Number(f64::from(point.z)));
        }
        out.push(point.time_s.map_or(Out::Empty, Out::Number));
        out.push(Out::Number(f64::from(point.n_images)));
        out.push(Out::Number(point.n_objects as f64));
        out.push(Out::Number(point.n_tracks as f64));
    }

    fn value_cells(&self, point: &TimeCoursePoint, out: &mut Vec<Out>) {
        for (value, column) in point.values.iter().zip(&self.values) {
            out.push(
                value
                    .filter(|v| v.is_finite())
                    .map_or(Out::Empty, |v| Out::Number(v * column.factor)),
            );
        }
    }

    /// The wide tables (unsplit): per value a table of a row per time point
    /// (and z-plane), a column per line - into `<name>_wide.xlsx` (a sheet
    /// each) or `<name>_wide_<value>.csv`.
    pub(super) fn write_wide(
        &self,
        results: &dyn ResultsSource,
        name: &str,
        format: TableFileFormat,
        output: &Output,
        report: &mut EntryReport,
        cancel: &AtomicBool,
    ) -> Result<(), InternalErrors> {
        let part = Part {
            suffix: None,
            images: self.images.clone(),
            classes: self.classes.clone(),
        };
        // The lines in the order they come, each named once.
        let mut lines: Vec<(String, String)> = Vec::new();
        let mut rows: BTreeMap<(u32, u32), (Option<f64>, BTreeMap<usize, Vec<Option<f64>>>)> =
            BTreeMap::new();
        let mut n = 0usize;
        results.time_course_each(&self.filter(&part), &mut |point| {
            n += 1;
            if n.is_multiple_of(65_536) && cancel.load(Ordering::Relaxed) {
                return Err(InternalErrors::Cancelled);
            }
            let line = (
                match self.options.lines {
                    TimeCourseLineKind::Class => String::new(),
                    TimeCourseLineKind::Image => point.image.clone(),
                    _ => point.line.clone(),
                },
                point.class_name.clone(),
            );
            let index = match lines.iter().position(|l| *l == line) {
                Some(i) => i,
                None => {
                    lines.push(line);
                    lines.len() - 1
                }
            };
            let row = rows.entry((point.t, point.z)).or_default();
            row.0 = row.0.or(point.time_s);
            row.1.insert(index, point.values.clone());
            Ok(())
        })?;
        let names: Vec<String> = {
            let mut used: Vec<String> = Vec::new();
            lines
                .iter()
                .map(|(line, class)| {
                    let wanted = if line.is_empty() {
                        naming::sanitize(class)
                    } else {
                        format!("{}_{}", naming::sanitize(line), naming::sanitize(class))
                    }
                    .to_lowercase();
                    let name = naming::free_name(&wanted, |n| used.iter().any(|u| u == n));
                    used.push(name.clone());
                    name
                })
                .collect()
        };
        let mut tables: Vec<(String, Vec<Vec<Out>>)> = Vec::new();
        for (v, column) in self.values.iter().enumerate() {
            let mut header = vec![Out::Text("t".into())];
            if self.several_z {
                header.push(Out::Text("z".into()));
            }
            header.push(Out::Text("time_s".into()));
            header.extend(names.iter().map(|n| Out::Text(n.clone())));
            let mut table = vec![header];
            for ((t, z), (time_s, by_line)) in &rows {
                let mut row = vec![Out::Number(f64::from(*t))];
                if self.several_z {
                    row.push(Out::Number(f64::from(*z)));
                }
                row.push(time_s.map_or(Out::Empty, Out::Number));
                for i in 0..lines.len() {
                    row.push(
                        by_line
                            .get(&i)
                            .and_then(|values| values.get(v).copied().flatten())
                            .filter(|x| x.is_finite())
                            .map_or(Out::Empty, |x| Out::Number(x * column.factor)),
                    );
                }
                table.push(row);
            }
            tables.push((column.name.clone(), table));
        }
        match format {
            TableFileFormat::Csv => {
                for (value, table) in &tables {
                    let file = output.file(&format!("{name}_wide_{value}.csv"));
                    let mut sink = super::table::CsvSink::create(&output.path(&file))?;
                    for row in table {
                        sink.row(row)?;
                    }
                    sink.finish()?;
                    report.files.push(file);
                }
            }
            TableFileFormat::Xlsx => {
                let file = output.file(&format!("{name}_wide.xlsx"));
                super::table::write_sheets(&output.path(&file), &tables)?;
                report.files.push(file);
            }
        }
        Ok(())
    }
}

/// The value columns of a time course: per column, per statistic.
fn values_of(
    columns: &[Column],
    pairs: &[(crate::results::TimeCourseStat, &'static str)],
    unit: LengthUnit,
    classes: &[Class],
    relative: bool,
) -> Vec<ValueColumn> {
    let mut values = Vec::new();
    for column in columns {
        let (name, factor) = column_name(column, unit, classes);
        let label = column.display_label(classes);
        let column_unit = unit_label(column, unit);
        for (statistic, suffix) in pairs {
            values.push(if relative {
                ValueColumn {
                    name: format!("{name}_{suffix}_pct"),
                    label: format!("{label} ({}, % of the first time point)", statistic.label()),
                    unit: "%".into(),
                    factor: 1.0,
                }
            } else {
                ValueColumn {
                    name: format!("{name}_{suffix}"),
                    label: format!("{label} ({})", statistic.label()),
                    unit: column_unit.clone(),
                    factor,
                }
            });
        }
    }
    values
}

impl Rows for TimeCourseRows {
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
        let header: Vec<Out> = self
            .line_columns()
            .into_iter()
            .map(|(name, _)| Out::Text(name.into()))
            .chain(self.values.iter().map(|v| Out::Text(v.name.clone())))
            .collect();
        sink.row(&header)?;
        if part.images.is_empty() {
            return Ok(());
        }
        let mut line = Vec::new();
        let mut n = 0usize;
        results.time_course_each(&self.filter(part), &mut |point| {
            n += 1;
            if n.is_multiple_of(65_536) && cancel.load(Ordering::Relaxed) {
                return Err(InternalErrors::Cancelled);
            }
            line.clear();
            self.line_cells(point, &mut line);
            self.value_cells(point, &mut line);
            sink.row(&line)
        })
    }

    fn about(&self) -> Vec<[String; 3]> {
        let mut rows = vec![["column".into(), "label".into(), "unit".into()]];
        for (name, label) in self.line_columns() {
            rows.push([name.into(), label.into(), String::new()]);
        }
        for value in &self.values {
            rows.push([value.name.clone(), value.label.clone(), value.unit.clone()]);
        }
        rows
    }
}

pub(super) struct ProfileRows {
    images: Vec<String>,
    classes: Option<Vec<ObjectClass>>,
    key: String,
    lines: ProfileLineKind,
    content: ProfileContent,
    /// The class names per id - an object's value names its class.
    class_names: Vec<(u32, String)>,
}

impl ProfileRows {
    pub(super) fn new(
        results: &dyn ResultsSource,
        table: &TableExport,
    ) -> Result<Self, InternalErrors> {
        let options = table.profile.clone().unwrap_or_default();
        Ok(Self {
            images: scope_images(results, table.images)?,
            classes: classes_of(table),
            key: options.key,
            lines: options.lines,
            content: options.content,
            class_names: results
                .get_object_classes()?
                .into_iter()
                .map(|class| (class.id, class.name))
                .collect(),
        })
    }

    /// The columns: (format 1 name, label).
    fn columns(&self) -> Vec<(&'static str, &'static str)> {
        match self.content {
            ProfileContent::Objects => vec![
                ("image", "Image (relative path)"),
                ("class", "Class"),
                ("object_id", "Object"),
                ("position", "Position (e.g. a ring's radius, px)"),
                ("value", "Value"),
            ],
            ProfileContent::Lines => {
                let mut columns = match self.lines {
                    ProfileLineKind::Class => vec![("class", "Class")],
                    ProfileLineKind::Image => vec![("image", "Image (relative path)")],
                    ProfileLineKind::Well => vec![("well", "Well")],
                    ProfileLineKind::All => vec![],
                };
                columns.extend([
                    ("position", "Position (e.g. a ring's radius, px)"),
                    ("n_objects", "Objects of the line"),
                    ("mean", "Mean (an object without a value here counts 0)"),
                    ("sd", "Standard deviation"),
                    ("median", "Median"),
                    ("q1", "First quartile"),
                    ("q3", "Third quartile"),
                ]);
                columns
            }
        }
    }
}

impl Rows for ProfileRows {
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
        let header: Vec<Out> = self
            .columns()
            .into_iter()
            .map(|(name, _)| Out::Text(name.into()))
            .collect();
        sink.row(&header)?;
        if part.images.is_empty() {
            return Ok(());
        }
        let number = |v: Option<f64>| v.filter(|v| v.is_finite()).map_or(Out::Empty, Out::Number);
        match self.content {
            ProfileContent::Lines => {
                let result = results.profile(&ProfileFilter {
                    key: self.key.clone(),
                    lines: match self.lines {
                        ProfileLineKind::Class => ProfileLines::Class,
                        ProfileLineKind::Image => ProfileLines::Image,
                        ProfileLineKind::Well => ProfileLines::Well,
                        ProfileLineKind::All => ProfileLines::All,
                    },
                    grouping: Grouping::Auto,
                    images: Some(part.images.clone()),
                    object_classes: part.classes.clone(),
                    max_lines: usize::MAX,
                })?;
                for point in result.points {
                    let mut row = match self.lines {
                        ProfileLineKind::Class => vec![Out::Text(point.class_name.clone())],
                        ProfileLineKind::Image | ProfileLineKind::Well => {
                            vec![Out::Text(point.line.replace('\\', "/"))]
                        }
                        ProfileLineKind::All => vec![],
                    };
                    row.extend([
                        Out::Number(point.position),
                        Out::Number(point.n_objects as f64),
                        number(point.mean),
                        number(point.sd),
                        number(point.median),
                        number(point.q1),
                        number(point.q3),
                    ]);
                    sink.row(&row)?;
                }
            }
            ProfileContent::Objects => {
                let mut offset = 0u64;
                loop {
                    if cancel.load(Ordering::Relaxed) {
                        return Err(InternalErrors::Cancelled);
                    }
                    let page = results.measurement_rows(&MeasurementRowsFilter {
                        images: Some(part.images.clone()),
                        keys: Some(vec![self.key.clone()]),
                        limit: PAGE_SIZE,
                        offset,
                    })?;
                    let count = page.len();
                    for row in page {
                        let in_classes = match (&part.classes, &row.class) {
                            (Some(classes), Some(class)) => {
                                self.class_names.iter().any(|(id, n)| {
                                    n == class && classes.contains(&ObjectClass::Global(*id))
                                })
                            }
                            _ => true,
                        };
                        if !in_classes {
                            continue;
                        }
                        sink.row(&[
                            Out::Text(row.image.replace('\\', "/")),
                            row.class.map_or(Out::Empty, Out::Text),
                            row.object_id.map_or(Out::Empty, Out::Text),
                            number(row.position),
                            number(row.value),
                        ])?;
                    }
                    if count < PAGE_SIZE as usize {
                        return Ok(());
                    }
                    offset += count as u64;
                }
            }
        }
        Ok(())
    }

    fn about(&self) -> Vec<[String; 3]> {
        let mut rows = vec![["column".into(), "label".into(), "unit".into()]];
        for (name, label) in self.columns() {
            rows.push([name.into(), label.into(), String::new()]);
        }
        rows.push([
            "profile".into(),
            format!("The profile measurement: {}", self.key),
            String::new(),
        ]);
        rows
    }
}
