//! # tracks
//!
//! A `TABLE` of `TRACK` rows in table format 1: a row per track - where
//! (image, well, class), when (time points, duration), how it moved
//! (Measure Track Motion's values) and its family (with divisions or
//! fusions), then the statistics of the asked-for columns over its time
//! points. Lengths in µm (or nm) where the pixel size is known, else px;
//! speeds in the unit asked for where it can be.
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-11-24
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.

use super::table::{Out, Part, Rows, Sink, column_name, scope_images, unit_label};
use crate::results::view_settings::resolve_columns;
use crate::results::{
    Column, Grouping, MotionKind, ResultsSource, SpeedUnit, TimeCourseStat, TrackMotion,
    TrackSummary, TrackSummaryFilter,
};
use evanalyzer_cfg::core_types::{InternalErrors, ObjectClass};
use evanalyzer_cfg::settings::export_settings::{
    ExportAggregation, ExportSettings, LengthUnit, TableExport, TrackOptions,
};
use std::sync::atomic::{AtomicBool, Ordering};

/// The tracks of a run, read once, written part by part.
pub(super) struct TrackRows {
    images: Vec<String>,
    classes: Option<Vec<ObjectClass>>,
    tracks: Vec<TrackSummary>,
    /// (format 1 name, label, unit, factor) per column and statistic.
    values: Vec<(String, String, String, f64)>,
    unit: LengthUnit,
    speed: SpeedUnit,
    /// Lengths in µm (or nm): the pixel size is known.
    um: bool,
    /// Durations in minutes: the frame interval is known.
    timed: bool,
    motion: bool,
    lineage: bool,
    several_z: bool,
}

impl TrackRows {
    pub(super) fn new(
        results: &dyn ResultsSource,
        export: &ExportSettings,
        table: &TableExport,
        notes: &mut Vec<String>,
    ) -> Result<Self, InternalErrors> {
        let images = scope_images(results, table.images)?;
        let classes: Option<Vec<ObjectClass>> = table
            .classes
            .as_ref()
            .map(|ids| ids.iter().map(|id| ObjectClass::Global(*id)).collect());
        let db_classes = results.get_object_classes()?;
        let mut columns = match &table.columns {
            None => Vec::new(),
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
        columns.retain(|column| {
            !matches!(
                column,
                Column::ImageName
                    | Column::ObjectClass
                    | Column::ObjectId
                    | Column::TrackId
                    | Column::Count
            )
        });
        columns.sort();
        columns.dedup();
        let mut statistics = Vec::new();
        for aggregation in &table.aggregations {
            match statistic(*aggregation) {
                Some(stat) => statistics.push((stat, suffix(*aggregation))),
                None => notes.push("tracks have no skewness - left out".into()),
            }
        }
        let options = table.tracks.clone().unwrap_or(TrackOptions {
            min_points: 1,
            ..TrackOptions::default()
        });
        let tracks = results.track_summary(&TrackSummaryFilter {
            grouping: Grouping::Auto,
            columns: columns.clone(),
            statistics: statistics.iter().map(|(stat, _)| *stat).collect(),
            images: Some(images.clone()),
            object_classes: classes.clone(),
            min_points: options.min_points,
            from_start: options.from_start,
            no_edge: options.no_edge,
        })?;
        let um = tracks.iter().any(|track| track.um_per_px.is_some());
        let timed = tracks.iter().any(|track| track.s_per_frame.is_some());
        let speed = match (um && timed, options.speed_unit) {
            (false, _) => SpeedUnit::PxPerFrame,
            (true, evanalyzer_cfg::settings::export_settings::SpeedUnit::PxFrame) => {
                SpeedUnit::PxPerFrame
            }
            (true, evanalyzer_cfg::settings::export_settings::SpeedUnit::UmMin) => {
                SpeedUnit::UmPerMin
            }
            (true, evanalyzer_cfg::settings::export_settings::SpeedUnit::UmS) => SpeedUnit::UmPerS,
        };
        let mut values = Vec::new();
        for column in &columns {
            let (name, factor) = column_name(column, export.length_unit, &db_classes);
            let label = column.display_label(&db_classes);
            let unit = unit_label(column, export.length_unit);
            for (_, suffix) in &statistics {
                values.push((
                    format!("{name}_{suffix}"),
                    format!("{label} ({suffix})"),
                    unit.clone(),
                    factor,
                ));
            }
        }
        Ok(Self {
            images,
            classes,
            motion: tracks.iter().any(|track| track.motion.is_measured()),
            lineage: results.has_lineage()?,
            tracks,
            values,
            unit: export.length_unit,
            speed,
            um,
            timed,
            several_z: results.get_nr_of_z_stacks() > 1,
        })
    }

    /// The length unit's name part and the factor from µm.
    fn length(&self) -> (&'static str, f64) {
        match (self.um, self.unit) {
            (false, _) => ("px", 1.0),
            (true, LengthUnit::Um) => ("um", 1.0),
            (true, LengthUnit::Nm) => ("nm", 1000.0),
        }
    }

    fn time(&self) -> &'static str {
        match self.speed {
            SpeedUnit::PxPerFrame => "time_point",
            SpeedUnit::UmPerMin => "min",
            SpeedUnit::UmPerS => "s",
        }
    }

    /// The columns: (format 1 name, label, unit).
    fn columns(&self) -> Vec<(String, String, String)> {
        let text = |name: &str, label: &str, unit: &str| {
            (name.to_string(), label.to_string(), unit.to_string())
        };
        let mut columns = vec![
            text("image", "Image (relative path)", ""),
            text("well", "Well", ""),
            text("class", "Class", ""),
            text("track_id", "Track", ""),
        ];
        if self.several_z {
            columns.push(text("z", "Z-plane (from 0)", ""));
        }
        if self.lineage {
            columns.extend([
                text("parent_track_ids", "Parent tracks (;-separated)", ""),
                text("child_track_ids", "Child tracks (;-separated)", ""),
                text("lineage", "Lineage (its root track)", ""),
                text("generation", "Generation", ""),
                text("starts_with", "Starts with", ""),
                text("ends_with", "Ends with", ""),
            ]);
        }
        columns.extend([
            text("first_t", "First time point (from 0)", ""),
            text("last_t", "Last time point (from 0)", ""),
            text("time_points", "Time points with an object", ""),
        ]);
        columns.push(if self.timed {
            text("duration_min", "Duration", "min")
        } else {
            text("duration_time_points", "Duration", "time points")
        });
        if self.motion {
            let (len, _) = self.length();
            let len_unit = match len {
                "um" => "µm",
                other => other,
            };
            let time = self.time();
            let speed = format!(
                "{}_per_{time}",
                if self.speed == SpeedUnit::PxPerFrame {
                    "px"
                } else {
                    "um"
                }
            );
            let area = if self.speed == SpeedUnit::PxPerFrame {
                "px2"
            } else {
                "um2"
            };
            for (id, label, kind) in TrackMotion::VALUES {
                let (name, unit) = match kind {
                    MotionKind::Length => (format!("{id}_{len}"), len_unit.to_string()),
                    MotionKind::Speed => (format!("{id}_{speed}"), speed.replace('_', " ")),
                    MotionKind::Ratio => (id.to_string(), String::new()),
                    MotionKind::Angle => (format!("{id}_deg"), "°".into()),
                    MotionKind::AngularSpeed => {
                        (format!("{id}_deg_per_{time}"), format!("°/{time}"))
                    }
                    MotionKind::Frequency => (format!("{id}_per_{time}"), format!("1/{time}")),
                    MotionKind::Diffusion => {
                        (format!("{id}_{area}_per_{time}"), format!("{area}/{time}"))
                    }
                };
                columns.push((name, label.to_string(), unit));
            }
        }
        columns.extend([
            text("touches_edge", "Touches the image edge", ""),
            text("from_start", "There from the first time point", ""),
        ]);
        for (name, label, unit, _) in &self.values {
            columns.push((name.clone(), label.clone(), unit.clone()));
        }
        columns
    }

    fn row(&self, track: &TrackSummary) -> Vec<Out> {
        let number = |v: f64| Out::Number(v);
        let optional = |v: Option<f64>| v.map_or(Out::Empty, Out::Number);
        let truth = |b: bool| Out::Text(if b { "TRUE" } else { "FALSE" }.into());
        let ids =
            |ids: &[u64]| Out::Text(ids.iter().map(u64::to_string).collect::<Vec<_>>().join(";"));
        let mut row = vec![
            Out::Text(track.image_rel_path.clone()),
            Out::Text(track.group.clone()),
            Out::Text(track.class_name.clone()),
            number(track.track_id as f64),
        ];
        if self.several_z {
            row.push(number(f64::from(track.z)));
        }
        if self.lineage {
            let lineage = &track.lineage;
            row.extend([
                ids(&lineage.parents),
                ids(&lineage.children),
                number(lineage.root as f64),
                number(f64::from(lineage.generation)),
                Out::Text(lineage.starts_with().label().into()),
                Out::Text(lineage.ends_with().label().into()),
            ]);
        }
        let frames = f64::from(track.last_t.saturating_sub(track.first_t));
        row.extend([
            number(f64::from(track.first_t)),
            number(f64::from(track.last_t)),
            number(f64::from(track.points)),
        ]);
        row.push(match track.s_per_frame.filter(|_| self.timed) {
            Some(s) => number(frames * s / 60.0),
            None => number(frames),
        });
        if self.motion {
            let (_, nm_factor) = self.length();
            for (i, (_, _, kind)) in TrackMotion::VALUES.iter().enumerate() {
                row.push(optional(match kind {
                    // Lengths in µm (nm) whenever the pixel size is known.
                    MotionKind::Length => {
                        track
                            .motion
                            .get(i)
                            .map(|px| match track.um_per_px.filter(|_| self.um) {
                                Some(um) => px * um * nm_factor,
                                None => px,
                            })
                    }
                    _ => track
                        .motion
                        .in_units(i, self.speed, track.um_per_px, track.s_per_frame),
                }));
            }
        }
        row.push(truth(track.touches_edge));
        row.push(truth(track.from_start));
        for ((_, _, _, factor), value) in self.values.iter().zip(&track.values) {
            row.push(optional(value.map(|v| v * factor)));
        }
        row
    }
}

impl Rows for TrackRows {
    fn images(&self) -> &[String] {
        &self.images
    }

    fn classes(&self) -> Option<Vec<ObjectClass>> {
        self.classes.clone()
    }

    fn write_part(
        &self,
        _results: &dyn ResultsSource,
        part: &Part,
        sink: &mut dyn Sink,
        cancel: &AtomicBool,
    ) -> Result<(), InternalErrors> {
        let header: Vec<Out> = self
            .columns()
            .into_iter()
            .map(|(name, ..)| Out::Text(name))
            .collect();
        sink.row(&header)?;
        for (n, track) in self.tracks.iter().enumerate() {
            if n % 4096 == 0 && cancel.load(Ordering::Relaxed) {
                return Err(InternalErrors::Cancelled);
            }
            let in_part = part.images.contains(&track.image_rel_path)
                && part.classes.as_ref().is_none_or(|classes| {
                    classes.contains(&ObjectClass::Global(track.class_id.max(0) as u32))
                });
            if in_part {
                sink.row(&self.row(track))?;
            }
        }
        Ok(())
    }

    fn about(&self) -> Vec<[String; 3]> {
        let mut rows = vec![[
            "column".to_string(),
            "label".to_string(),
            "unit".to_string(),
        ]];
        rows.extend(
            self.columns()
                .into_iter()
                .map(|(name, label, unit)| [name, label, unit]),
        );
        rows
    }
}

pub(super) fn statistic(aggregation: ExportAggregation) -> Option<TimeCourseStat> {
    Some(match aggregation {
        ExportAggregation::Avg => TimeCourseStat::Mean,
        ExportAggregation::Median => TimeCourseStat::Median,
        ExportAggregation::Stddev => TimeCourseStat::Sd,
        ExportAggregation::Sum => TimeCourseStat::Sum,
        ExportAggregation::Min => TimeCourseStat::Min,
        ExportAggregation::Max => TimeCourseStat::Max,
        ExportAggregation::Skewness => return None,
    })
}

pub(super) fn suffix(aggregation: ExportAggregation) -> &'static str {
    match aggregation {
        ExportAggregation::Avg => "avg",
        ExportAggregation::Median => "median",
        ExportAggregation::Stddev => "stddev",
        ExportAggregation::Sum => "sum",
        ExportAggregation::Min => "min",
        ExportAggregation::Max => "max",
        ExportAggregation::Skewness => "skewness",
    }
}
