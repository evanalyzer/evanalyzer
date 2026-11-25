//! # measurements
//!
//! A `TABLE` of `MEASUREMENT` rows in table format 1: a row per measurement
//! value with what it is about - the long layout R or Python read without
//! reshaping, where every new measurement fits without new columns. The
//! unit is a column of its own (`unit`), values as measured.
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-11-24
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.

use super::table::{Out, Part, Rows, Sink, scope_images};
use crate::exports::jobs::TableExport;
use crate::results::{Column, MeasurementRow, MeasurementRowsFilter, ResultsSource};
use evanalyzer_cfg::core_types::{InternalErrors, ObjectClass};
use std::sync::atomic::{AtomicBool, Ordering};

/// Rows per query page.
const PAGE_SIZE: u32 = 100_000;

pub(super) struct MeasurementTable {
    images: Vec<String>,
    classes: Option<Vec<ObjectClass>>,
    /// The class names per id - a measurement row names its class.
    class_names: Vec<(u32, String)>,
    /// `None`: all measurements.
    keys: Option<Vec<String>>,
    several_z: bool,
    several_t: bool,
    planes: evanalyzer_cfg::settings::export_settings::Planes,
    /// The results have profiles (a value per position).
    position: bool,
}

impl MeasurementTable {
    pub(super) fn new(
        results: &dyn ResultsSource,
        table: &TableExport,
        notes: &mut Vec<String>,
    ) -> Result<Self, InternalErrors> {
        let keys = table.columns.as_ref().map(|columns| {
            let (keys, others): (Vec<&String>, Vec<&String>) = columns
                .iter()
                .partition(|key| Column::measurement_key(key).is_some());
            if !others.is_empty() {
                let others: Vec<&str> = others.iter().map(|key| key.as_str()).collect();
                notes.push(format!(
                    "a table of measurements has measurements only - left out: {}",
                    others.join(", ")
                ));
            }
            keys.iter()
                .filter_map(|key| Column::measurement_key(key).map(String::from))
                .collect()
        });
        Ok(Self {
            images: scope_images(results, table.images)?,
            classes: table
                .classes
                .as_ref()
                .map(|ids| ids.iter().map(|id| ObjectClass::Global(*id)).collect()),
            class_names: results
                .get_object_classes()?
                .into_iter()
                .map(|class| (class.id, class.name))
                .collect(),
            keys,
            planes: table.planes,
            several_z: results.get_nr_of_z_stacks() > 1,
            several_t: results.get_nr_of_t_stacks() > 1,
            position: !results.profile_columns()?.is_empty(),
        })
    }

    /// The columns: (format 1 name, label).
    fn columns(&self) -> Vec<(&'static str, &'static str)> {
        let mut columns = vec![("image", "Image (relative path)")];
        if self.several_t {
            columns.push(("t", "Time point (from 0)"));
        }
        if self.several_z {
            columns.push(("z", "Z-plane (from 0; empty: the whole volume)"));
        }
        columns.extend([
            (
                "region",
                "What was measured: the image, an object, a track, ...",
            ),
            ("object_id", "Object"),
            ("related_object_id", "The other object (a relation's)"),
            ("class", "Class"),
            ("step", "Pipeline step"),
            ("measurement", "Measurement"),
            ("key", "Measurement key"),
            ("value", "Value"),
            ("unit", "Unit of the value"),
        ]);
        if self.position {
            columns.push(("position", "Position (a profile's, e.g. a ring's radius)"));
        }
        columns
    }

    /// Whether `row` belongs to `part`'s classes: rows without a class (of
    /// an image) always do.
    fn in_classes(&self, part: &Part, row: &MeasurementRow) -> bool {
        let (Some(classes), Some(class)) = (&part.classes, &row.class) else {
            return true;
        };
        self.class_names
            .iter()
            .any(|(id, name)| name == class && classes.contains(&ObjectClass::Global(*id)))
    }

    fn row(&self, row: MeasurementRow) -> Vec<Out> {
        let text = |value: Option<String>| value.map_or(Out::Empty, Out::Text);
        let mut out = vec![Out::Text(row.image.replace('\\', "/"))];
        if self.several_t {
            out.push(Out::Number(f64::from(row.t)));
        }
        if self.several_z {
            out.push(row.z.map_or(Out::Empty, |z| Out::Number(f64::from(z))));
        }
        out.extend([
            text(row.region),
            text(row.object_id),
            text(row.related_object_id),
            text(row.class),
            text(row.step),
            text(row.measurement),
            Out::Text(row.key),
            row.value
                .filter(|v| v.is_finite())
                .map_or(Out::Empty, Out::Number),
            text(row.unit),
        ]);
        if self.position {
            out.push(row.position.map_or(Out::Empty, Out::Number));
        }
        out
    }
}

impl Rows for MeasurementTable {
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
            .map(|(name, _)| Out::Text(name.to_string()))
            .collect();
        sink.row(&header)?;
        if part.images.is_empty() {
            return Ok(());
        }
        let mut offset = 0u64;
        loop {
            if cancel.load(Ordering::Relaxed) {
                return Err(InternalErrors::Cancelled);
            }
            let page = results.measurement_rows(&MeasurementRowsFilter {
                images: Some(part.images.clone()),
                keys: self.keys.clone(),
                limit: PAGE_SIZE,
                offset,
            })?;
            let count = page.len();
            for row in page {
                // A whole-volume value (no z) is of every z-plane.
                let in_planes = super::in_planes(
                    &self.planes,
                    row.z
                        .and_then(|z| u32::try_from(z).ok())
                        .unwrap_or(self.planes.z.map_or(0, |range| range.from)),
                    u32::try_from(row.t).unwrap_or(0),
                );
                if in_planes && self.in_classes(part, &row) {
                    sink.row(&self.row(row))?;
                }
            }
            if count < PAGE_SIZE as usize {
                return Ok(());
            }
            offset += count as u64;
        }
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
                .map(|(name, label)| [name.to_string(), label.to_string(), String::new()]),
        );
        rows
    }
}
