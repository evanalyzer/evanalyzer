//! # results_exporter
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-06-05
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.

use super::results_generator::{
    ImageSelection, ResultsGenerator, class_display_label, class_filter_sql, sql_string_in_list,
};
use super::results_time_course::{TimeCourseHead, TimeCourseQuery};
use super::results_tracks::{TrackSummaryHead, TrackSummaryQuery};
use crate::results::{
    Aggregation, Cell, CellValue, Column, ColumnEntry, DatabaseResult, GroupedByImageFilter,
    ImageHeatmapFilter, ListFilter, Pagination, PlaneFilter, PlateFilterMulti, View,
    WellsBatchFilterMulti,
};
use crate::results::{ExportFormat, ExportProgress, ResultExport};
use crate::results::{
    MotionKind, TimeCourseFilter, TimeCourseLayout, TimeCourseLines, TimeCoursePoint,
    TimeCourseResult, TrackMotion, TrackSummary, TrackSummaryFilter,
};
use crate::results::{ProfileExportContent, ProfileFilter, ProfileLines, ProfilePoint};
use crate::user::settings::SpeedUnit;
use evanalyzer_cfg::core_types::{InternalErrors, ObjectClass};
use evanalyzer_cfg::settings::images_settings::ImageMarker;
use rust_xlsxwriter::{Color, Format, FormatAlign, Workbook, Worksheet, XlsxError};
use std::collections::HashMap;
use std::collections::HashSet;
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};

pub(super) mod drawn_images;
mod format;
mod grouped;
mod plate;
mod series;
mod tables;
#[cfg(test)]
mod tests;

// What the parts share, reachable here (and in `tests`) as before.
use self::{format::*, plate::*, tables::*};

/// Light gray Excel gives every other coloc-detail row (`Cell::alternating_color`)
/// so the fanned-out rows belonging to one source object stay visually
/// grouped — matches `Theme.list-row-alt-bg`'s role in the GUI's own List
/// view, just as a plain hex constant here since XLSX formatting has no
/// theme to pull from.
const ALTERNATING_ROW_BG: u32 = 0xF1F1F1;

const EXPORT_FONT_SIZE: f64 = 10.0;

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

/// An XLSX sheet's rows, the header included.
const XLSX_MAX_ROWS: usize = 1_048_576;

/// A cell holding just `value`, no formatting.
/// The images' marker and note (only those with one), for the image-level
/// tables of an export - the results' own copy.
struct ImageReviews(HashMap<String, (Option<ImageMarker>, String)>);

impl ImageReviews {
    fn of(database: &ResultsGenerator) -> Result<Self, InternalErrors> {
        Ok(Self(
            database
                .get_images()?
                .into_iter()
                .filter(|image| image.marker.is_some() || !image.note.is_empty())
                .map(|image| {
                    (
                        image.rel_path.to_string_lossy().into_owned(),
                        (image.marker, image.note),
                    )
                })
                .collect(),
        ))
    }

    /// The columns: none when no image has a marker or a note - such an
    /// export stays as it was.
    fn header(&self) -> Vec<String> {
        if self.0.is_empty() {
            return Vec::new();
        }
        vec!["image_marker".to_string(), "image_note".to_string()]
    }

    /// The marker's colour name and the note of the image at `rel_path`
    /// (empty without).
    fn cells(&self, rel_path: &str) -> Vec<Cell> {
        if self.0.is_empty() {
            return Vec::new();
        }
        let text = |text: &str| {
            plain_cell(if text.is_empty() {
                CellValue::Empty
            } else {
                CellValue::String(text.to_string())
            })
        };
        match self.0.get(rel_path) {
            Some((marker, note)) => vec![text(marker.map_or("", ImageMarker::name)), text(note)],
            None => vec![text(""), text("")],
        }
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
        if self.with_profile {
            self.export_profile(database, cancel, &mut *on_progress)?;
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
            // Each image's planes within the exported ones, named by their
            // Z / T when the results have several.
            let planes = ExportPlanes::of(database, self)?;
            drawn_images::export_drawn_images(
                database,
                drawn,
                &self.object_classes,
                &images,
                &self.output_dir,
                drawn_images::DrawnPlanes {
                    z_stacks: self.z_stacks,
                    t_stacks: self.t_stacks,
                    by_z: planes.by_z,
                    by_t: planes.by_t,
                },
                cancel,
                &mut *on_progress,
            )?;
        }
        Ok(())
    }
}

/// The images an export should cover, in the same image-name-sorted order
/// `get_images()` returns them in — regardless of what order
/// `ResultExport::image_rel_paths` happens to list them in, so "sorted by
/// image name" (see `export_list`'s doc comment) holds even when the caller
/// supplies an explicit subset. `[]` means every image in the database,
/// except ones the user has disabled — an explicit list is taken as
/// overriding that (the caller asked for exactly these, disabled or not).
impl ResultExport {
    /// The images of `resolved` (from [`resolve_images`]) for a query: every
    /// enabled image as a condition when none is picked (no list of every
    /// path in each page's SQL), else the picked ones.
    fn image_selection<'a>(&self, resolved: &'a [String]) -> ImageSelection<'a> {
        if self.image_rel_paths.is_empty() && !resolved.is_empty() {
            ImageSelection::Enabled
        } else {
            ImageSelection::Listed(resolved)
        }
    }
}

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
