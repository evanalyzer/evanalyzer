//! # reused
//!
//! The entries the results' own export already draws - the plate layout
//! and the drawn images: written by it into a scratch folder, then moved
//! to their names in table format 1's naming (`docs/exports.md` §2).
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-11-24
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.

use super::Output;
use super::table::scope_images;
use crate::exports::jobs::{PlateGrid, PlateLayoutExport};
use crate::exports::naming;
use crate::exports::types::EntryReport;
use crate::results::view_settings::resolve_columns;
use crate::results::{
    Aggregation, Column, DrawBrightness, DrawStyle, DrawnImages, ExportFormat, ResultExport,
    ResultsSource,
};
use evanalyzer_cfg::core_types::{InternalErrors, ObjectClass};
use evanalyzer_cfg::settings::export_settings::{
    DrawnBrightness, DrawnImagesExport, DrawnStyle, ExportAggregation, ExportEntry, ExportSettings,
    HeatmapColors,
};
use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicBool;

/// A plate (or well) layout: `plate.xlsx` / `well.xlsx` of the results'
/// export, under the entry's name.
pub(super) fn plate_layout(
    results: &dyn ResultsSource,
    export: &ExportSettings,
    plate: &PlateLayoutExport,
    output: &Output,
    report: &mut EntryReport,
    cancel: &AtomicBool,
) -> Result<(), InternalErrors> {
    let db_classes = results.get_object_classes()?;
    let columns = match &plate.columns {
        None => vec![Column::Count],
        Some(keys) => {
            let available = results.get_available_columns()?;
            let (columns, missing) = resolve_columns(keys, &db_classes, &available);
            if !missing.is_empty() {
                report.notes.push(format!(
                    "these results have no column {}",
                    missing.join(", ")
                ));
            }
            columns
        }
    };
    let images = scope_images(results, plate.images)?;
    if plate.grid == PlateGrid::Image {
        return image_heatmaps(
            results, export, plate, columns, images, output, report, cancel,
        );
    }
    let scratch = Scratch::new(output)?;
    let job = ResultExport {
        output_dir: scratch.dir.clone(),
        format: ExportFormat::XLSX,
        image_rel_paths: images,
        columns,
        object_classes: classes(&plate.classes),
        aggregations: plate.aggregations.iter().map(|a| aggregation(*a)).collect(),
        color_schema: color_schema(plate.colors),
        z_stacks: super::plane_range(plate.planes.z),
        t_stacks: super::plane_range(plate.planes.t),
        with_plate_view_heatmap: plate.grid == PlateGrid::Plate,
        with_well_view_heatmap: plate.grid == PlateGrid::Well,
        with_well_view_list: plate.grid == PlateGrid::WellList,
        ..Default::default()
    };
    results.export(&job, cancel, &mut |_, _, _| {})?;
    let written = match plate.grid {
        PlateGrid::Plate => "plate.xlsx",
        PlateGrid::WellList => "well_list.xlsx",
        _ => "well.xlsx",
    };
    let name = naming::with_prefix(export, &naming::plate_name(plate));
    let file = output.file(&format!("{name}.xlsx"));
    scratch.take(Path::new(written), &output.path(&file))?;
    report.files.push(file);
    Ok(())
}

/// A plate layout of `grid: Image`: a workbook per image, its area as a
/// grid of squares - `<name>_<image>.xlsx`, the image by its shortest
/// unique name. Written image by image (one at a time in the scratch
/// folder: which file is whose is never guessed).
#[allow(clippy::too_many_arguments)]
fn image_heatmaps(
    results: &dyn ResultsSource,
    export: &ExportSettings,
    plate: &PlateLayoutExport,
    columns: Vec<Column>,
    images: Vec<String>,
    output: &Output,
    report: &mut EntryReport,
    cancel: &AtomicBool,
) -> Result<(), InternalErrors> {
    let name = naming::with_prefix(export, &naming::plate_name(plate));
    let parts = naming::unique_image_parts(&images);
    for image in &images {
        if cancel.load(std::sync::atomic::Ordering::Relaxed) {
            return Err(InternalErrors::Cancelled);
        }
        let scratch = Scratch::new(output)?;
        let job = ResultExport {
            output_dir: scratch.dir.clone(),
            format: ExportFormat::XLSX,
            image_rel_paths: vec![image.clone()],
            columns: columns.clone(),
            object_classes: classes(&plate.classes),
            aggregations: plate.aggregations.iter().map(|a| aggregation(*a)).collect(),
            color_schema: color_schema(plate.colors),
            z_stacks: super::plane_range(plate.planes.z),
            t_stacks: super::plane_range(plate.planes.t),
            with_heatmap: true,
            square_size: Some(plate.square_size.unwrap_or(64).max(1) as usize),
            ..Default::default()
        };
        results.export(&job, cancel, &mut |_, _, _| {})?;
        let Some(written) = files_below(&scratch.dir)?.into_iter().next() else {
            continue;
        };
        let file = output.file(&format!("{name}_{}.xlsx", parts[image]));
        let relative = written
            .strip_prefix(&scratch.dir)
            .unwrap_or(&written)
            .to_path_buf();
        scratch.take(&relative, &output.path(&file))?;
        report.files.push(file);
    }
    Ok(())
}

/// The images with their objects drawn: a PNG per image (and plane),
/// flat, named by the image's shortest unique name.
pub(super) fn drawn_images(
    results: &dyn ResultsSource,
    export: &ExportSettings,
    drawn: &DrawnImagesExport,
    drawing: &DrawnImages,
    output: &Output,
    report: &mut EntryReport,
    cancel: &AtomicBool,
) -> Result<(), InternalErrors> {
    let images = scope_images(results, drawn.images)?;
    let scratch = Scratch::new(output)?;
    let job = ResultExport {
        output_dir: scratch.dir.clone(),
        image_rel_paths: images.clone(),
        object_classes: classes(&drawn.classes),
        z_stacks: super::plane_range(drawn.planes.z),
        t_stacks: super::plane_range(drawn.planes.t),
        drawn_images: Some(DrawnImages {
            max_side: drawn.max_side,
            style: match drawn.style {
                DrawnStyle::Outline => DrawStyle::Outline,
                DrawnStyle::Filled => DrawStyle::Filled,
                DrawnStyle::Box => DrawStyle::Box,
            },
            brightness: match drawn.brightness {
                DrawnBrightness::Viewer => DrawBrightness::Viewer,
                DrawnBrightness::Auto => DrawBrightness::Auto,
            },
            ..drawing.clone()
        }),
        ..Default::default()
    };
    results.export(&job, cancel, &mut |_, _, _| {})?;

    // The export wrote `images/<folder>/<stem>[_z…][_t…].png`: the image is
    // the longest path the file starts with.
    let name = naming::with_prefix(
        export,
        &naming::entry_names(&ExportEntry::DrawnImages(drawn.clone()))[0],
    );
    let parts = naming::unique_image_parts(&images);
    let mut bases: Vec<(String, &String)> = images
        .iter()
        .map(|image| {
            let base = image
                .rsplit_once('.')
                .filter(|(stem, _)| !stem.ends_with('/') && !stem.is_empty())
                .map_or(image.as_str(), |(stem, _)| stem);
            (base.to_string(), image)
        })
        .collect();
    bases.sort_by_key(|(base, _)| std::cmp::Reverse(base.len()));
    let root = scratch.dir.join("images");
    for written in files_below(&root)? {
        let relative = written
            .strip_prefix(&root)
            .unwrap_or(&written)
            .to_string_lossy()
            .replace('\\', "/");
        let Some(without_ext) = relative.strip_suffix(".png") else {
            continue;
        };
        let Some((base, image)) = bases.iter().find(|(base, _)| {
            without_ext.strip_prefix(base.as_str()).is_some_and(|rest| {
                rest.is_empty() || rest.starts_with("_z") || rest.starts_with("_t")
            })
        }) else {
            continue;
        };
        let plane = &without_ext[base.len()..];
        let file = output.file(&format!("{name}_{}{plane}.png", parts[*image]));
        scratch.take(
            Path::new(&format!("images/{relative}")),
            &output.path(&file),
        )?;
        report.files.push(file);
    }
    report.files.sort();
    Ok(())
}

fn classes(classes: &Option<Vec<u32>>) -> Vec<ObjectClass> {
    classes
        .as_ref()
        .map(|ids| ids.iter().map(|id| ObjectClass::Global(*id)).collect())
        .unwrap_or_default()
}

fn color_schema(colors: HeatmapColors) -> crate::results::ColorSchema {
    use crate::results::ColorSchema;
    match colors {
        HeatmapColors::Excel => ColorSchema::Excel,
        HeatmapColors::Viridis => ColorSchema::Viridis,
        HeatmapColors::Plasma => ColorSchema::Plasma,
        HeatmapColors::Inferno => ColorSchema::Inferno,
        HeatmapColors::Cividis => ColorSchema::Cividis,
        HeatmapColors::Coolwarm => ColorSchema::Coolwarm,
        HeatmapColors::RedBlue => ColorSchema::RedBlue,
        HeatmapColors::YlGnBu => ColorSchema::YlGnBu,
        HeatmapColors::Haline => ColorSchema::Haline,
        HeatmapColors::Algae => ColorSchema::Algae,
        HeatmapColors::Thermal => ColorSchema::Thermal,
    }
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

/// Every file below `dir`.
fn files_below(dir: &Path) -> Result<Vec<PathBuf>, InternalErrors> {
    let mut found = Vec::new();
    let mut todo = vec![dir.to_path_buf()];
    while let Some(dir) = todo.pop() {
        let Ok(entries) = std::fs::read_dir(&dir) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                todo.push(path);
            } else {
                found.push(path);
            }
        }
    }
    found.sort();
    Ok(found)
}

/// A scratch folder inside the output folder (so files only move, never
/// copy across disks), removed when done.
struct Scratch {
    dir: PathBuf,
}

impl Scratch {
    fn new(output: &Output) -> Result<Self, InternalErrors> {
        let dir = output.path(Path::new(&format!(
            ".evanalyzer-export-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_or(0, |d| d.as_nanos())
        )));
        std::fs::create_dir_all(&dir)
            .map_err(|e| InternalErrors::Io(format!("Could not create {}: {e}", dir.display())))?;
        Ok(Self { dir })
    }

    /// Moves the scratch file `relative` to `to`.
    fn take(&self, relative: &Path, to: &Path) -> Result<(), InternalErrors> {
        let from = self.dir.join(relative);
        std::fs::rename(&from, to).map_err(|e| {
            InternalErrors::Io(format!(
                "Could not move {} to {}: {e}",
                from.display(),
                to.display()
            ))
        })
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.dir);
    }
}
