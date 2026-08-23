//! Converts an old (`.icproj`) project into the current [`ProjectSettings`] format.
//!
//! The old C++/nlohmann-json application used a structurally different format -
//! different top-level shape, a `$xxx`-optional-field encoding for "which pipeline
//! command runs in this step" instead of a `type` tag, and several commands that
//! don't exist (or whose semantics changed) in this codebase's reimplementation.
//!
//! This module does a best-effort, field-by-field conversion of everything that has
//! a confident equivalent, and reports anything it had to drop, approximate, or
//! skip via [`LegacyImportOutcome::warnings`] rather than silently losing it or
//! failing the whole import. See `legacy_schema` for the source format.
//!
//! Two import-time expansions are structural, not just field renames:
//! - The old app extracted ROIs automatically as part of running `$threshold` /
//!   `$watershed`; this codebase requires explicit `ConnectedComponents` +
//!   `ExtractObjects` steps afterwards. These are inserted automatically right after
//!   the last segmentation step in a run of them.
//! - The old `$classify` step held a *list* of model classes, each with a *list* of
//!   filters; this codebase's `ClassifyObjects` is one filter per step. Each
//!   (model class × filter) pair becomes its own `ClassifyObjects` step.

use crate::core_types::{ImageAddress, ObjectClass, PipelineId, SegmentationClass, SizeUnits};
use crate::legacy_schema::*;
use crate::settings::classification_settings::{Class, ClassificationSettings};
use crate::settings::images_settings::{
    GlobalImageSettings, PixelSizeSettings, ZStackHandling, ZStackSettings,
};
use crate::settings::meta_data::MetaData;
use crate::settings::pipeline_command::PipelineCommand;
use crate::settings::pipeline_command_settings::*;
use crate::settings::pipeline_settings::{PipelineSettings, PipelineStepSettings};
use crate::settings::plate_settings::{GroupingMode, PlateSettings};
use crate::settings::project_settings::ProjectSettings;

#[derive(Debug, thiserror::Error)]
pub enum LegacyImportError {
    #[error("failed to parse legacy project JSON: {0}")]
    Json(#[from] serde_json::Error),
}

/// Result of importing a legacy `.icproj` project.
pub struct LegacyImportOutcome {
    pub project: ProjectSettings,
    /// Notes on anything dropped, approximated, or skipped during conversion -
    /// e.g. unsupported pipeline commands, or fields with no new equivalent.
    /// Empty means the import had no caveats.
    pub warnings: Vec<String>,
    /// The old project's configured image folder (resolved relative to nothing -
    /// it's whatever was stored in the `.icproj`), if any. The returned
    /// [`ProjectSettings::images`] is always empty: populating it requires
    /// scanning disk for real files, which is the caller's job (mirroring
    /// whatever "add images from folder" flow already exists for new projects).
    pub legacy_image_folder: Option<String>,
}

/// Parses an old `.icproj` JSON document and converts it into the current
/// [`ProjectSettings`] format. See the module docs for what is and isn't
/// supported.
pub fn import_legacy_project(json: &str) -> Result<LegacyImportOutcome, LegacyImportError> {
    let old: LegacyAnalyzeSettings = serde_json::from_str(json)?;
    let mut warnings = Vec::new();
    let project = convert_project(&old, &mut warnings);
    let legacy_image_folder = resolve_plate(&old.project_settings, &mut warnings)
        .image_folder
        .clone();
    let legacy_image_folder = (!legacy_image_folder.is_empty()).then_some(legacy_image_folder);
    Ok(LegacyImportOutcome {
        project,
        warnings,
        legacy_image_folder,
    })
}

fn convert_project(old: &LegacyAnalyzeSettings, warnings: &mut Vec<String>) -> ProjectSettings {
    let mut images = crate::settings::images_settings::ImageSettings::default();
    images.settings = convert_global_image_settings(&old.image_setup);

    ProjectSettings {
        // Built fresh from the current-shape fields below, so it's already
        // current - no migration needed, unlike a file loaded from disk.
        schema_version: crate::CURRENT_PROJECT_SCHEMA_VERSION,
        metadata: convert_metadata(old),
        classification: convert_classification(&old.project_settings.classification, warnings),
        plate: convert_plate(&old.project_settings, warnings),
        images,
        pipelines: convert_pipelines(old, warnings),
        tile_merge: crate::settings::project_settings::TileMergeSettings::default(),
    }
}

// ---------------------------------------------------------------------------
// Metadata
// ---------------------------------------------------------------------------

fn convert_metadata(old: &LegacyAnalyzeSettings) -> MetaData {
    let m = &old.meta;
    let name = if !m.name.is_empty() {
        m.name.clone()
    } else {
        old.project_settings
            .experiment_settings
            .experiment_name
            .clone()
    };
    let description = if !m.notes.is_empty() {
        m.notes.clone()
    } else {
        old.project_settings.experiment_settings.notes.clone()
    };
    let creation_time = m
        .modified_at
        .as_deref()
        .and_then(|s| chrono::DateTime::parse_from_rfc3339(s).ok())
        .map(|dt| dt.with_timezone(&chrono::Utc))
        .unwrap_or_else(chrono::Utc::now);
    let authors = m
        .author
        .as_deref()
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(|s| vec![s.to_string()])
        .unwrap_or_default();

    MetaData {
        name,
        short_description: String::new(),
        description,
        authors,
        author_organization: m.organization.clone().unwrap_or_default(),
        creation_time,
        category: String::new(),
        tags: Vec::new(),
        // Stamped for real by `save_project_as` once the imported project is
        // actually saved - this legacy file predates the field entirely.
        app_version: String::new(),
    }
}

// ---------------------------------------------------------------------------
// Class IDs
// ---------------------------------------------------------------------------

/// Parses a legacy `ClassId`/`ClassIdIn` numeric JSON string ("0".."49") into the
/// raw value. Returns `None` for anything else (including "None", "Undefined",
/// "$", and the "M01".."M11" temp-class forms).
fn parse_numeric_class_id(s: &str) -> Option<u32> {
    s.parse::<u32>().ok()
}

/// Resolves a legacy class-id field to the new `ObjectClass`, substituting
/// `pipeline_default` for the "$" placeholder (meaning "this pipeline's own
/// default class", set via `PipelineSettings::defaultClassId` in the old format).
fn resolve_class(
    s: &str,
    pipeline_default: ObjectClass,
    context: &str,
    field: &str,
    warnings: &mut Vec<String>,
) -> ObjectClass {
    match s {
        "$" => pipeline_default,
        "" | "None" | "Undefined" => ObjectClass::Unset,
        other => match parse_numeric_class_id(other) {
            Some(n) => ObjectClass::Valid(n),
            None => {
                warnings.push(format!(
                    "{context}: {field} uses unsupported legacy class id '{other}' (temp/model class) - left unset"
                ));
                ObjectClass::Unset
            }
        },
    }
}

// ---------------------------------------------------------------------------
// Classification
// ---------------------------------------------------------------------------

fn convert_classification(
    old: &LegacyClassification,
    warnings: &mut Vec<String>,
) -> ClassificationSettings {
    let classes = old
        .classes
        .iter()
        .map(|c| convert_class(c, warnings))
        .collect();
    ClassificationSettings::new_from_existing(classes)
}

fn convert_class(old: &LegacyClass, warnings: &mut Vec<String>) -> Class {
    let id = resolve_class(
        &old.class_id,
        ObjectClass::Unset,
        "classification",
        "classId",
        warnings,
    );
    let color = parse_hex_color(&old.color).unwrap_or(0x9933FF);
    Class {
        id,
        color,
        name: old.name.clone(),
        notes: old.notes.clone(),
    }
}

fn parse_hex_color(s: &str) -> Option<u32> {
    u32::from_str_radix(s.trim_start_matches('#'), 16).ok()
}

// ---------------------------------------------------------------------------
// Plate
// ---------------------------------------------------------------------------

/// Old projects predating the singular `plate` field only populated the legacy
/// `plates` list. Falls back to its first entry in that case.
fn resolve_plate<'a>(
    old: &'a LegacyProjectSettings,
    warnings: &mut Vec<String>,
) -> &'a LegacyPlate {
    if !old.plate.image_folder.is_empty() {
        return &old.plate;
    }
    if let Some(first) = old.plates.first() {
        warnings.push(
            "project only had the legacy multi-plate list ('plates'); only the first plate was imported"
                .to_string(),
        );
        return first;
    }
    &old.plate
}

fn convert_plate(old: &LegacyProjectSettings, warnings: &mut Vec<String>) -> PlateSettings {
    let plate = resolve_plate(old, warnings);

    let grouping_mode = match plate.group_by.as_str() {
        "Directory" => GroupingMode::FolderName,
        "Filename" => GroupingMode::FileName,
        _ => GroupingMode::NoGrouping,
    };

    let well_image_order: Vec<i32> = plate
        .plate_setup
        .well_image_order
        .iter()
        .flatten()
        .copied()
        .collect();

    let default = PlateSettings::default();
    PlateSettings {
        grouping_mode,
        grouping_regex: plate.filename_regex.clone(),
        // The old format has no concept of multiple physical plates per project -
        // that's `wellRows`/`wellCols` below (the per-well image tiling grid).
        plate_cols: 1,
        plate_rows: 1,
        well_cols: if plate.plate_setup.cols > 0 {
            plate.plate_setup.cols
        } else {
            default.well_cols
        },
        well_rows: if plate.plate_setup.rows > 0 {
            plate.plate_setup.rows
        } else {
            default.well_rows
        },
        well_image_order: if well_image_order.is_empty() {
            default.well_image_order
        } else {
            well_image_order
        },
    }
}

// ---------------------------------------------------------------------------
// Image setup (pixel size, z-projection)
// ---------------------------------------------------------------------------

fn legacy_unit_to_nm_factor(unit: &str) -> f32 {
    match unit {
        "nm" => 1.0,
        "um" => 1_000.0,
        "mm" => 1_000_000.0,
        "cm" => 10_000_000.0,
        "m" => 1_000_000_000.0,
        "km" => 1_000_000_000_000.0,
        // "Px" / "Undefined": no physical unit was configured - treat the raw
        // value as already nm-equivalent rather than guessing.
        _ => 1.0,
    }
}

fn map_z_projection(old: &str) -> ZStackHandling {
    match old {
        "MaxIntensity" => ZStackHandling::MaxIntensity,
        "MinIntensity" => ZStackHandling::MinIntensity,
        "AvgIntensity" => ZStackHandling::AvgIntensity,
        "TakeMiddle" => ZStackHandling::TakeTheMiddle,
        _ => ZStackHandling::SingleStack,
    }
}

fn convert_global_image_settings(old: &LegacyProjectImageSetup) -> GlobalImageSettings {
    let mut settings = GlobalImageSettings::default();

    if old.image_pixel_size_settings.mode == "Manual" {
        let px = &old.image_pixel_size_settings;
        let factor = legacy_unit_to_nm_factor(&px.pixel_size_unit);
        settings.pixel_sizes = Some(PixelSizeSettings {
            x: px.pixel_width * factor,
            y: px.pixel_height * factor,
            z: 1.0,
        });
    }

    settings.z_stack = Some(ZStackSettings {
        z_projection: map_z_projection(&old.z_stack_settings.default_z_projection),
        z_range: None,
    });

    settings
}

// ---------------------------------------------------------------------------
// Pipelines
// ---------------------------------------------------------------------------

fn convert_pipelines(
    old: &LegacyAnalyzeSettings,
    warnings: &mut Vec<String>,
) -> Vec<PipelineSettings> {
    old.pipelines
        .iter()
        .enumerate()
        .map(|(idx, p)| convert_pipeline(idx as u32, p, warnings))
        .collect()
}

fn convert_pipeline(id: u32, old: &LegacyPipeline, warnings: &mut Vec<String>) -> PipelineSettings {
    let display_name = if old.meta.name.is_empty() {
        format!("pipeline #{id}")
    } else {
        old.meta.name.clone()
    };
    let context = format!("pipeline '{display_name}'");

    let pipeline_default_class = resolve_class(
        &old.pipeline_setup.default_class_id,
        ObjectClass::Unset,
        &context,
        "defaultClassId",
        warnings,
    );

    let image_source = if old.pipeline_setup.source == "FromFile" {
        ImageAddress::Channel(old.pipeline_setup.c_stack_index)
    } else {
        ImageAddress::Scratchpad
    };

    let mut steps: Vec<PipelineStepSettings> = Vec::new();
    // True right after converting a `$threshold`/`$watershed` step: the old app
    // extracted ROIs as a side effect of those; this app needs explicit
    // ConnectedComponents + ExtractObjects steps, inserted just before whatever
    // comes next (or at the end of the pipeline).
    let mut pending_segmentation = false;

    for step in &old.pipeline_steps {
        let is_segmentation = step.threshold.is_some() || step.watershed.is_some();

        if pending_segmentation && !is_segmentation {
            push_segmentation_followup(&mut steps);
            pending_segmentation = false;
        }

        let enabled = !step.disabled;
        for command in convert_step(step, pipeline_default_class, &context, warnings) {
            steps.push(PipelineStepSettings { enabled, command });
        }

        if is_segmentation {
            pending_segmentation = true;
        }
    }
    if pending_segmentation {
        push_segmentation_followup(&mut steps);
    }

    PipelineSettings {
        id: PipelineId(id),
        name: (!old.meta.name.is_empty()).then(|| old.meta.name.clone()),
        image_source,
        enabled: !old.disabled,
        steps,
    }
}

fn push_segmentation_followup(steps: &mut Vec<PipelineStepSettings>) {
    steps.push(PipelineStepSettings {
        enabled: true,
        command: PipelineCommand::ConnectedComponents(ConnectedComponentsSettings { min_size: 0 }),
    });
    steps.push(PipelineStepSettings {
        enabled: true,
        command: PipelineCommand::ExtractObjects(ExtractObjectsSettings::default()),
    });
}

/// Converts one legacy pipeline step. Most commands produce exactly one new
/// command; `$classify` can expand into several (one per old model-class ×
/// filter pair); unsupported commands produce none (after recording a warning).
fn convert_step(
    step: &LegacyPipelineStep,
    pipeline_default_class: ObjectClass,
    context: &str,
    warnings: &mut Vec<String>,
) -> Vec<PipelineCommand> {
    if let Some(s) = &step.blur {
        return vec![convert_blur(s, context, warnings)];
    }
    if let Some(s) = &step.canny {
        return vec![convert_canny(s)];
    }
    if let Some(s) = &step.sobel {
        return vec![convert_sobel(s, context, warnings)];
    }
    if let Some(s) = &step.rolling_ball {
        return vec![convert_rolling_ball(s)];
    }
    if let Some(s) = &step.median_subtract {
        return vec![convert_median_subtract(s)];
    }
    if let Some(s) = &step.enhance_contrast {
        return vec![convert_enhance_contrast(s)];
    }
    if let Some(s) = &step.hessian {
        return vec![convert_hessian(s, context, warnings)];
    }
    if let Some(s) = &step.structure_tensor {
        return vec![convert_structure_tensor(s, context, warnings)];
    }
    if let Some(s) = &step.gaussian_weighted_dev {
        return vec![convert_weighted_deviation(s)];
    }
    if let Some(s) = &step.rank {
        return vec![convert_rank_filter(s, context, warnings)];
    }
    if let Some(s) = &step.laplacian {
        return vec![convert_laplacian(s, context, warnings)];
    }
    if let Some(s) = &step.morphological_transform {
        return convert_morphological(s, context, warnings);
    }
    if let Some(s) = &step.color_filter {
        return convert_color_filter(s, context, warnings);
    }
    if let Some(s) = &step.intensity_transform {
        return vec![convert_intensity_transform(s, context, warnings)];
    }
    if let Some(s) = &step.threshold {
        return convert_threshold(s, context, warnings);
    }
    if let Some(s) = &step.watershed {
        return vec![convert_watershed(s)];
    }
    if let Some(s) = &step.voronoi {
        return vec![convert_voronoi(
            s,
            pipeline_default_class,
            context,
            warnings,
        )];
    }
    if let Some(s) = &step.classify {
        return convert_classifier(s, context, warnings);
    }
    if let Some(s) = &step.object_transform {
        return convert_object_transform(s, pipeline_default_class, context, warnings);
    }
    if let Some(s) = &step.save_image {
        return vec![convert_save_image(s)];
    }

    warn_unsupported_step(step, context, warnings);
    vec![]
}

fn warn_unsupported_step(step: &LegacyPipelineStep, context: &str, warnings: &mut Vec<String>) {
    let unsupported: &[(&str, bool)] = &[
        ("$thresholdAdaptive", step.threshold_adaptive.is_some()),
        ("$crop", step.crop.is_some()),
        ("$fillHoles", step.fill_holes.is_some()),
        ("$houghTransform", step.hough_transform.is_some()),
        ("$skeletonize", step.skeletonize.is_some()),
        ("$nop", step.nop.is_some()),
        ("$imageToCache", step.image_to_cache.is_some()),
        ("$imageMath", step.image_math.is_some()),
        ("$thresholdValidator", step.threshold_validator.is_some()),
        ("$noiseValidator", step.noise_validator.is_some()),
        ("$objectsToImage", step.objects_to_image.is_some()),
        ("$aiClassify", step.ai_classify.is_some()),
        ("$pixelClassify", step.pixel_classify.is_some()),
        ("$colocalization", step.colocalization.is_some()),
        ("$reclassify", step.reclassify.is_some()),
        ("$measureIntensity", step.measure_intensity.is_some()),
        ("$measureDistance", step.measure_distance.is_some()),
    ];
    for (name, present) in unsupported {
        if *present {
            warnings.push(format!(
                "{context}: step '{name}' has no equivalent command in the new format - skipped"
            ));
        }
    }
}

// ---- image-level filters ----

fn convert_blur(
    s: &LegacyBlurSettings,
    context: &str,
    warnings: &mut Vec<String>,
) -> PipelineCommand {
    if s.mode == "GaussianBlur" {
        // Old had no explicit sigma for Gaussian blur (it derived one from the
        // kernel size internally); approximate via the standard kernel<->sigma
        // relationship documented on the new GaussianBlur (N ~= 6*sigma + 1).
        let sigma = ((s.kernel_size as f32 - 1.0) / 6.0).max(0.1);
        warnings.push(format!(
            "{context}: '$blur' (Gaussian) had no sigma in the legacy format - estimated sigma={sigma:.2} from kernel size"
        ));
        PipelineCommand::GaussianBlur(GaussianBlurSettings {
            kernel_size: s.kernel_size.max(1) as usize,
            sigma,
        })
    } else {
        if s.repeat != 1 {
            warnings.push(format!(
                "{context}: '$blur' repeat={} has no equivalent - applied once",
                s.repeat
            ));
        }
        PipelineCommand::Blur(BlurSettings {
            kernel_size: s.kernel_size.max(1) as usize,
        })
    }
}

fn convert_canny(s: &LegacyEdgeDetectionCannySettings) -> PipelineCommand {
    // Old thresholds are in raw [0-255] intensity space; the new command's
    // thresholds are normalized [0.0, 1.0].
    PipelineCommand::EdgeDetectionCanny(EdgeDetectionCannySettings {
        kernel_size: s.kernel_size.max(1) as usize,
        threshold_min: s.threshold_min / 255.0,
        threshold_max: s.threshold_max / 255.0,
    })
}

fn convert_sobel(
    s: &LegacyEdgeDetectionSobelSettings,
    context: &str,
    warnings: &mut Vec<String>,
) -> PipelineCommand {
    warnings.push(format!(
        "{context}: '$sobel' weight function / derivative order have no equivalent - only kernel size was kept"
    ));
    PipelineCommand::EdgeDetectionSobel(EdgeDetectionSobelSettings {
        kernel_size: s.kernel_size.max(1) as usize,
    })
}

fn convert_rolling_ball(s: &LegacyRollingBallSettings) -> PipelineCommand {
    let ball_type = match s.ball_type.as_str() {
        "Paraboloid" => FiltersRollingBallBallTypeSettings::Paraboloid,
        _ => FiltersRollingBallBallTypeSettings::Ball,
    };
    PipelineCommand::RollingBall(RollingBallSettings {
        radius: s.ball_size as f64,
        ball_type,
        pre_smooth: false,
    })
}

fn convert_median_subtract(s: &LegacyMedianSubtractSettings) -> PipelineCommand {
    // Old "kernel size" (diameter) -> new "radius".
    let radius = ((s.kernel_size as f64 - 1.0) / 2.0).max(0.0);
    PipelineCommand::MedianSubtract(MedianSubtractSettings { radius })
}

fn convert_enhance_contrast(s: &LegacyEnhanceContrastSettings) -> PipelineCommand {
    PipelineCommand::EnhanceContrast(EnhanceContrastSettings {
        saturated_pixels: s.saturated_pixels,
        normalize: s.normalize,
        equalize_histogram: s.equalize_histogram,
    })
}

fn convert_hessian(
    s: &LegacyHessianSettings,
    context: &str,
    warnings: &mut Vec<String>,
) -> PipelineCommand {
    let mode = match s.mode.as_str() {
        "EigenvaluesX" => FiltersHessianHessianModeSettings::EigenvaluesX,
        "EigenvaluesY" => FiltersHessianHessianModeSettings::EigenvaluesY,
        "Determinant" => FiltersHessianHessianModeSettings::Determinant,
        other => {
            warnings.push(format!(
                "{context}: '$hessian' mode '{other}' unrecognized - defaulted to Determinant"
            ));
            FiltersHessianHessianModeSettings::Determinant
        }
    };
    PipelineCommand::Hessian(HessianSettings { mode })
}

fn convert_structure_tensor(
    s: &LegacyStructureTensorSettings,
    context: &str,
    warnings: &mut Vec<String>,
) -> PipelineCommand {
    let mode = match s.mode.as_str() {
        "EigenvaluesX" => FiltersStructureTensorTensorModeSettings::EigenvaluesX,
        "EigenvaluesY" => FiltersStructureTensorTensorModeSettings::EigenvaluesY,
        "Coherence" => FiltersStructureTensorTensorModeSettings::Coherence,
        other => {
            warnings.push(format!(
                "{context}: '$structureTensor' mode '{other}' unrecognized - defaulted to Coherence"
            ));
            FiltersStructureTensorTensorModeSettings::Coherence
        }
    };
    PipelineCommand::StructureTensor(StructureTensorSettings {
        mode,
        kernel_size: s.kernel_size.max(1) as usize,
        // Old had no explicit sigma for this filter either.
        sigma: 1.5,
    })
}

fn convert_weighted_deviation(s: &LegacyWeightedDeviationSettings) -> PipelineCommand {
    PipelineCommand::WeightedDeviation(WeightedDeviationSettings {
        kernel_size: s.kernel_size.max(1) as usize,
        sigma: s.sigma as f32,
    })
}

fn convert_rank_filter(
    s: &LegacyRankFilterSettings,
    context: &str,
    warnings: &mut Vec<String>,
) -> PipelineCommand {
    let filter_type = match s.mode.as_str() {
        "Mean" => FiltersRankFilterRankFilterTypeSettings::Mean,
        "Min" => FiltersRankFilterRankFilterTypeSettings::Min,
        "Max" => FiltersRankFilterRankFilterTypeSettings::Max,
        "Median" => FiltersRankFilterRankFilterTypeSettings::Median,
        "Outliers" => FiltersRankFilterRankFilterTypeSettings::Outliers(50.0),
        other => {
            warnings.push(format!(
                "{context}: '$rank' mode '{other}' has no equivalent - defaulted to Median"
            ));
            FiltersRankFilterRankFilterTypeSettings::Median
        }
    };
    PipelineCommand::RankFilter(RankFilterSettings {
        radius: s.radius,
        filter_type,
    })
}

fn convert_laplacian(
    s: &LegacyLaplacianSettings,
    context: &str,
    warnings: &mut Vec<String>,
) -> PipelineCommand {
    let _ = (context, warnings); // `repeat` (if != 1) is silently dropped; not worth a warning per-step.
    PipelineCommand::Laplacian(LaplacianSettings {
        kernel_size: s.kernel_size.max(1) as usize,
    })
}

fn convert_morphological(
    s: &LegacyMorphologicalTransformSettings,
    context: &str,
    warnings: &mut Vec<String>,
) -> Vec<PipelineCommand> {
    let op = match s.function.as_str() {
        "Dilate" => MorphologyMorphologicalTransformationMorphOpsSettings::Dilate,
        "Erode" => MorphologyMorphologicalTransformationMorphOpsSettings::Erode,
        "Open" => MorphologyMorphologicalTransformationMorphOpsSettings::Open,
        "Close" => MorphologyMorphologicalTransformationMorphOpsSettings::Close,
        other => {
            warnings.push(format!(
                "{context}: '$morphologicalTransform' function '{other}' has no equivalent - step skipped"
            ));
            return vec![];
        }
    };
    let kernel_shape = match s.shape.as_str() {
        "Cross" => MorphologyMorphologicalTransformationKernelShapesSettings::Cross,
        "Ellipse" => MorphologyMorphologicalTransformationKernelShapesSettings::Ellipse,
        _ => MorphologyMorphologicalTransformationKernelShapesSettings::Box,
    };
    vec![PipelineCommand::MorphologicalCommand(
        MorphologicalCommandSettings {
            op,
            kernel_size: s.kernel_size.max(1) as usize,
            kernel_shape,
            use_grayscale: false,
        },
    )]
}

fn convert_color_filter(
    s: &LegacyColorFilterSettings,
    context: &str,
    warnings: &mut Vec<String>,
) -> Vec<PipelineCommand> {
    let Some(first) = s.filter.first() else {
        return vec![];
    };
    if s.filter.len() > 1 {
        warnings.push(format!(
            "{context}: '$colorFilter' had {} color ranges - only the first was imported",
            s.filter.len()
        ));
    }
    // Old hue is on a [0, 255] scale (255 == 360deg); sat/val are [0, 255] -> [0.0, 1.0].
    let to_degrees = |h: i32| h as f32 * 360.0 / 255.0;
    let to_unit = |v: i32| (v as f32 / 255.0).clamp(0.0, 1.0);
    vec![PipelineCommand::ColorFilterCommand(
        ColorFilterCommandSettings {
            range: HsvRangeSettings {
                min_h: to_degrees(first.color_range_from.hue),
                max_h: to_degrees(first.color_range_to.hue),
                min_s: to_unit(first.color_range_from.sat),
                max_s: to_unit(first.color_range_to.sat),
                min_v: to_unit(first.color_range_from.val),
                max_v: to_unit(first.color_range_to.val),
            },
        },
    )]
}

fn convert_intensity_transform(
    s: &LegacyIntensityTransformationSettings,
    context: &str,
    warnings: &mut Vec<String>,
) -> PipelineCommand {
    if s.gamma != 0 {
        warnings.push(format!(
            "{context}: '$intensityTransform' gamma={} has no equivalent - dropped",
            s.gamma
        ));
    }
    let mode = match s.mode.as_str() {
        "Automatic" => FiltersIntensityTransformIntensityTransformModeSettings::Automatic,
        _ => FiltersIntensityTransformIntensityTransformModeSettings::Manual,
    };
    PipelineCommand::IntensityTransformation(IntensityTransformationSettings {
        mode,
        contrast: s.contrast,
        // Old brightness is a raw 16-bit-ish offset; the new pipeline works on
        // normalized [0,1] images.
        brightness: s.brightness as f32 / 65535.0,
    })
}

// ---- segmentation ----

fn map_threshold_method(old: &str) -> SegmentationThresholdThresholdMethodSettings {
    use SegmentationThresholdThresholdMethodSettings as M;
    match old {
        "Manual" => M::Manual,
        "Li" => M::Li,
        "MinError" => M::MinError,
        "Triangle" => M::Triangle,
        "Moments" => M::Moments,
        "Huang" => M::Huang,
        "Intermodes" => M::Intermodes,
        "IsoData" => M::IsoData,
        "MaxEntropy" => M::MaxEntropy,
        "Mean" => M::Mean,
        "Minimum" => M::Minimum,
        // Legacy projects predate three-class Otsu; they always meant the
        // original two-class split.
        "Otsu" => M::Otsu {
            classes: SegmentationThresholdOtsuClassesSettings::Two,
        },
        // Old's JSON strings were misspelled ("Percentil", "TenyiEntropy").
        "Percentil" => M::Percentile,
        "TenyiEntropy" => M::RenyiEntropy,
        "Shanbhag" => M::Shanbhag,
        "Yen" => M::Yen,
        _ => M::None,
    }
}

fn convert_threshold(
    s: &LegacyThresholdSettings,
    context: &str,
    warnings: &mut Vec<String>,
) -> Vec<PipelineCommand> {
    if s.model_classes.is_empty() {
        return vec![];
    }
    let thresholds = s
        .model_classes
        .iter()
        .map(|t| {
            if t.method == "" {
                // empty string is the "NONE" sentinel, not an error.
            } else if map_threshold_method(&t.method)
                == SegmentationThresholdThresholdMethodSettings::None
                && t.method != "None"
            {
                warnings.push(format!(
                    "{context}: '$threshold' method '{}' unrecognized - defaulted to None",
                    t.method
                ));
            }
            ThresholdEntrySettings {
                method: map_threshold_method(&t.method),
                min_threshold: t.threshold_min,
                max_threshold: t.threshold_max,
                unit: crate::core_types::PixelUnits::Bit,
                object_class_id: SegmentationClass(t.pixel_class_id.max(0) as u32),
                value_source: SegmentationThresholdThresholdValueSourceSettings::ActualImage,
            }
        })
        .collect();
    vec![PipelineCommand::Threshold(ThresholdSettings { thresholds })]
}

fn convert_watershed(s: &LegacyWatershedSettings) -> PipelineCommand {
    // NOTE: old's tolerance was documented as a [0,1] fraction; the new tolerance
    // is an absolute pixel distance. There's no principled conversion between the
    // two without re-deriving it from the actual distance map, so the raw number
    // is carried over as a starting point - it is very likely too small and
    // should be re-tuned by hand.
    PipelineCommand::Watershed(WatershedSettings {
        maximum_finder_tolerance: s.maximum_finder_tolerance.max(0.1),
        smoothing_sigma: 0.0,
        min_object_size: 0,
        // Old watershed only ever seeded from the distance map - the new
        // `Intensity` option didn't exist yet, so this preserves old behavior.
        seed_source: SegmentationWatershedSeedSourceSettings::DistanceMap,
    })
}

// ---- object-level ----

fn convert_voronoi(
    s: &LegacyVoronoiGridSettings,
    pipeline_default_class: ObjectClass,
    context: &str,
    warnings: &mut Vec<String>,
) -> PipelineCommand {
    let mut points = s.input_classes_points.iter();
    let centers = points
        .next()
        .map(|c| {
            resolve_class(
                c,
                pipeline_default_class,
                context,
                "voronoi centers",
                warnings,
            )
        })
        .unwrap_or(ObjectClass::Unset);
    let center_filter_classes: Vec<ObjectClass> = points
        .map(|c| {
            resolve_class(
                c,
                pipeline_default_class,
                context,
                "voronoi centers (extra)",
                warnings,
            )
        })
        .collect();
    if !center_filter_classes.is_empty() {
        warnings.push(format!(
            "{context}: '$voronoi' had {} center classes - only the first was used as `centers`, the rest as filters",
            s.input_classes_points.len()
        ));
    }

    let mut masks = s.input_classes_mask.iter();
    let mask = masks
        .next()
        .map(|c| resolve_class(c, pipeline_default_class, context, "voronoi mask", warnings))
        .unwrap_or(ObjectClass::Unset);
    let mask_filter_classes: Vec<ObjectClass> = masks
        .map(|c| {
            resolve_class(
                c,
                pipeline_default_class,
                context,
                "voronoi mask (extra)",
                warnings,
            )
        })
        .collect();

    let output_class = resolve_class(
        &s.output_class_voronoi,
        pipeline_default_class,
        context,
        "voronoi output class",
        warnings,
    );

    PipelineCommand::Voronoi(VoronoiSettings {
        centers,
        center_filter_classes,
        mask,
        mask_filter_classes,
        output_class,
        unit: SizeUnits::Pixels,
        max_radius: s.max_radius,
        exclude_areas_at_the_edges: s.exclude_areas_at_the_edge,
        exclude_areas_with_no_center: s.exclude_areas_without_point,
    })
}

fn convert_classifier(
    s: &LegacyClassifierSettings,
    context: &str,
    warnings: &mut Vec<String>,
) -> Vec<PipelineCommand> {
    let mut out = Vec::new();
    for model_class in &s.model_classes {
        if model_class.output_class_no_match != "Undefined"
            && !model_class.output_class_no_match.is_empty()
        {
            warnings.push(format!(
                "{context}: '$classify' 'no match' class assignment has no equivalent - only 'match' classification was imported"
            ));
        }
        for filter in &model_class.filters {
            if filter.intensity.min_intensity >= 0.0 || filter.intensity.max_intensity >= 0.0 {
                warnings.push(format!(
                    "{context}: '$classify' intensity filter has no equivalent in ClassifyObjects - dropped"
                ));
            }
            let output_class = resolve_class(
                &filter.output_class,
                ObjectClass::Unset,
                context,
                "classify output class",
                warnings,
            );
            out.push(PipelineCommand::ClassifyObjects(ClassifyObjectsSettings {
                origin_segmentation: vec![SegmentationClass(
                    model_class.pixel_class_id.max(0) as u32
                )],
                input_classes: vec![],
                match_handling:
                    ClassificationClassifyObjectsClassifyMatchHandlingSettings::AddOutputClassIfMatch,
                output_class,
                overlapping_with: ObjectClass::Unset,
                min_intersection_area: 0.0,
                size_unit: SizeUnits::Pixels,
                min_area: filter.metrics.min_particle_size.max(0.0),
                max_area: if filter.metrics.max_particle_size >= 0.0 {
                    filter.metrics.max_particle_size
                } else {
                    2_147_483_600.0
                },
                min_circularity: filter.metrics.min_circularity.max(0.0),
                max_circularity: 1.0,
                min_solidity: 0.0,
                max_solidity: 1.0,
                min_aspect_ratio: 0.0,
                max_aspect_ratio: 2_147_483_600.0,
                min_eccentricity: 0.0,
                max_eccentricity: 1.0,
                min_feret: 0.0,
                max_feret: 2_147_483_600.0,
                allow_edge_touching: !filter.metrics.exclude_objects_at_the_edge,
            }));
        }
    }
    out
}

fn convert_object_transform(
    s: &LegacyObjectTransformSettings,
    pipeline_default_class: ObjectClass,
    context: &str,
    warnings: &mut Vec<String>,
) -> Vec<PipelineCommand> {
    let input_class = resolve_class(
        &s.input_classes,
        pipeline_default_class,
        context,
        "objectTransform input",
        warnings,
    );
    let output_class = resolve_class(
        &s.output_classes,
        pipeline_default_class,
        context,
        "objectTransform output",
        warnings,
    );

    let function = match s.function.as_str() {
        "Scale" => {
            ClassificationTransformObjectsTransformFunctionSettings::Scale { factor: s.factor }
        }
        "SnapArea" => ClassificationTransformObjectsTransformFunctionSettings::SnapArea {
            extra_size: s.factor,
            unit: SizeUnits::Pixels,
        },
        // Old's `factor` for circle functions is documented as a *radius*; the
        // new fields take a *diameter*.
        "CircleMin" => ClassificationTransformObjectsTransformFunctionSettings::MinCircle {
            min_diameter: s.factor * 2.0,
            unit: SizeUnits::Pixels,
        },
        "Circle" => ClassificationTransformObjectsTransformFunctionSettings::DrawCircle {
            diameter: s.factor * 2.0,
            unit: SizeUnits::Pixels,
        },
        "FitEllipse" => {
            ClassificationTransformObjectsTransformFunctionSettings::FittingEllipse { scale: 1.0 }
        }
        other => {
            warnings.push(format!(
                "{context}: '$objectTransform' function '{other}' unrecognized - step skipped"
            ));
            return vec![];
        }
    };

    vec![PipelineCommand::TransformObjects(
        TransformObjectsSettings {
            function,
            input_class,
            output_class,
        },
    )]
}

fn convert_save_image(s: &LegacyImageSaverSettings) -> PipelineCommand {
    let name = std::path::Path::new(&s.path)
        .file_stem()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_else(|| "image".to_string());
    PipelineCommand::SaveImage(SaveImageSettings {
        name,
        source: MathSaveImageImageSourceSettings::Image,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_project_json() -> String {
        r##"{
            "meta": { "name": "Legacy Demo", "notes": "Imported from old app", "author": "Joachim Danmayr", "organization": "evanalyzer.org" },
            "projectSettings": {
                "experimentSettings": { "experimentName": "Fallback name", "notes": "fallback notes" },
                "classification": {
                    "classes": [
                        { "classId": "1", "name": "nucleus", "color": "#3399FF", "notes": "",
                          "defaultMeasurements": [ { "measureChannel": "AreaSize", "stats": ["Avg", "Sum"] } ] },
                        { "classId": "None", "name": "unset", "color": "#BFBFBF", "notes": "" }
                    ]
                },
                "plate": {
                    "imageFolder": "images",
                    "groupBy": "Directory",
                    "filenameRegex": "_((.)([0-9]+))_([0-9]+)",
                    "plateSetup": { "rows": 2, "cols": 2, "wellImageOrder": [[1, 2], [3, 4]] }
                }
            },
            "imageSetup": {
                "imagePixelSizeSettings": { "mode": "Manual", "pixelSizeUnit": "um", "pixelWidth": 0.5, "pixelHeight": 0.5 },
                "zStackSettings": { "defaultZProjection": "MaxIntensity" }
            },
            "pipelines": [
                {
                    "meta": { "name": "Nuclei" },
                    "pipelineSetup": { "source": "FromFile", "cStackIndex": 0, "defaultClassId": "1" },
                    "pipelineSteps": [
                        { "$rollingBall": { "ballType": "Paraboloid", "ballSize": 4 } },
                        { "$blur": { "mode": "Blur", "kernelSize": 3, "repeat": 1 } },
                        { "$threshold": { "modelClasses": [ { "method": "Triangle", "thresholdMin": 200, "thresholdMax": 65535, "pixelClassId": 1 } ] } },
                        { "$watershed": { "maximumFinderTolerance": 0.5 } },
                        { "$classify": {
                            "modelClasses": [ {
                                "filters": [ { "outputClass": "1", "intensity": {"minIntensity": -1, "maxIntensity": -1}, "metrics": { "minParticleSize": 50.0, "maxParticleSize": -1, "minCircularity": -1, "excludeObjectsAtTheEdge": false } } ],
                                "outputClassNoMatch": "Undefined",
                                "pixelClassId": 1
                            } ]
                        } },
                        { "$houghTransform": { "someField": 1 } },
                        { "$saveImage": { "path": "output/nucleus.png" } }
                    ]
                }
            ]
        }"##
        .to_string()
    }

    #[test]
    fn imports_metadata_classification_and_plate() {
        let outcome = import_legacy_project(&sample_project_json()).expect("should parse");
        let p = &outcome.project;

        assert_eq!(p.metadata.name, "Legacy Demo");
        assert_eq!(p.metadata.authors, vec!["Joachim Danmayr".to_string()]);
        assert_eq!(p.metadata.author_organization, "evanalyzer.org");

        // classes()[0] is always the auto-prepended Background class - see
        // `ClassificationSettings::new_from_existing`.
        assert_eq!(p.classification.classes().len(), 3);
        assert_eq!(p.classification.classes()[0].id, ObjectClass::BACKGROUND);
        assert_eq!(p.classification.classes()[1].id, ObjectClass::Valid(1));
        assert_eq!(p.classification.classes()[1].color, 0x3399FF);
        assert_eq!(p.classification.classes()[2].id, ObjectClass::Unset);

        assert_eq!(p.plate.grouping_mode, GroupingMode::FolderName);
        assert_eq!(p.plate.well_rows, 2);
        assert_eq!(p.plate.well_cols, 2);
        assert_eq!(p.plate.well_image_order, vec![1, 2, 3, 4]);
        assert_eq!(p.plate.plate_rows, 1);
        assert_eq!(p.plate.plate_cols, 1);

        assert_eq!(outcome.legacy_image_folder, Some("images".to_string()));
    }

    #[test]
    fn imports_pixel_size_in_nanometers() {
        let outcome = import_legacy_project(&sample_project_json()).unwrap();
        let px = outcome
            .project
            .images
            .settings
            .pixel_sizes
            .expect("pixel size set");
        assert_eq!(px.x, 500.0); // 0.5 um -> 500 nm
        assert_eq!(px.y, 500.0);
    }

    #[test]
    fn expands_threshold_and_watershed_into_segmentation_plus_extraction() {
        let outcome = import_legacy_project(&sample_project_json()).unwrap();
        let steps = &outcome.project.pipelines[0].steps;

        let names: Vec<&str> = steps
            .iter()
            .map(|s| match &s.command {
                PipelineCommand::RollingBall(_) => "RollingBall",
                PipelineCommand::Blur(_) => "Blur",
                PipelineCommand::Threshold(_) => "Threshold",
                PipelineCommand::Watershed(_) => "Watershed",
                PipelineCommand::ConnectedComponents(_) => "ConnectedComponents",
                PipelineCommand::ExtractObjects(_) => "ExtractObjects",
                PipelineCommand::ClassifyObjects(_) => "ClassifyObjects",
                PipelineCommand::SaveImage(_) => "SaveImage",
                _ => "Other",
            })
            .collect();

        assert_eq!(
            names,
            vec![
                "RollingBall",
                "Blur",
                "Threshold",
                "Watershed",
                "ConnectedComponents",
                "ExtractObjects",
                "ClassifyObjects",
                "SaveImage",
            ]
        );
    }

    #[test]
    fn classify_objects_carries_the_metrics_filter() {
        let outcome = import_legacy_project(&sample_project_json()).unwrap();
        let classify = outcome.project.pipelines[0]
            .steps
            .iter()
            .find_map(|s| match &s.command {
                PipelineCommand::ClassifyObjects(c) => Some(c),
                _ => None,
            })
            .expect("a ClassifyObjects step");

        assert_eq!(classify.min_area, 50.0);
        assert_eq!(classify.output_class, ObjectClass::Valid(1));
        assert_eq!(classify.origin_segmentation, vec![SegmentationClass(1)]);
    }

    #[test]
    fn reports_a_warning_for_unsupported_commands() {
        let outcome = import_legacy_project(&sample_project_json()).unwrap();
        assert!(
            outcome
                .warnings
                .iter()
                .any(|w| w.contains("$houghTransform")),
            "expected a warning about the unsupported houghTransform step, got: {:#?}",
            outcome.warnings
        );
    }

    #[test]
    fn save_image_uses_the_file_stem_as_name() {
        let outcome = import_legacy_project(&sample_project_json()).unwrap();
        let save = outcome.project.pipelines[0]
            .steps
            .iter()
            .find_map(|s| match &s.command {
                PipelineCommand::SaveImage(c) => Some(c),
                _ => None,
            })
            .expect("a SaveImage step");
        assert_eq!(save.name, "nucleus");
    }

    #[test]
    fn dollar_placeholder_resolves_to_the_pipelines_default_class() {
        let json = r##"{
            "meta": { "name": "X" },
            "projectSettings": { "classification": { "classes": [] }, "plate": {} },
            "imageSetup": {},
            "pipelines": [ {
                "meta": { "name": "P" },
                "pipelineSetup": { "source": "FromMemory", "defaultClassId": "7" },
                "pipelineSteps": [
                    { "$objectTransform": { "function": "Scale", "inputClasses": "$", "outputClasses": "$", "factor": 1.2 } }
                ]
            } ]
        }"##;
        let outcome = import_legacy_project(json).unwrap();
        let transform = outcome.project.pipelines[0]
            .steps
            .iter()
            .find_map(|s| match &s.command {
                PipelineCommand::TransformObjects(c) => Some(c),
                _ => None,
            })
            .expect("a TransformObjects step");
        assert_eq!(transform.input_class, ObjectClass::Valid(7));
        assert_eq!(transform.output_class, ObjectClass::Valid(7));
    }

    #[test]
    fn unparseable_json_returns_an_error_not_a_panic() {
        let result = import_legacy_project("{ not valid json");
        assert!(result.is_err());
    }

    fn pipeline_with_step(step_json: &str) -> String {
        format!(
            r##"{{
                "meta": {{ "name": "X" }},
                "projectSettings": {{ "classification": {{ "classes": [] }}, "plate": {{}} }},
                "imageSetup": {{}},
                "pipelines": [ {{
                    "meta": {{ "name": "P" }},
                    "pipelineSetup": {{ "source": "FromMemory", "defaultClassId": "1" }},
                    "pipelineSteps": [ {step_json} ]
                }} ]
            }}"##
        )
    }

    fn only_command(outcome: &LegacyImportOutcome) -> &PipelineCommand {
        &outcome.project.pipelines[0].steps[0].command
    }

    #[test]
    fn gaussian_blur_mode_maps_to_the_gaussian_command_with_an_estimated_sigma() {
        let json = pipeline_with_step(
            r#"{ "$blur": { "mode": "GaussianBlur", "kernelSize": 7, "repeat": 1 } }"#,
        );
        let outcome = import_legacy_project(&json).unwrap();
        match only_command(&outcome) {
            PipelineCommand::GaussianBlur(s) => {
                assert_eq!(s.kernel_size, 7);
                // (7-1)/6 = 1.0
                assert!((s.sigma - 1.0).abs() < 1e-6, "sigma was {}", s.sigma);
            }
            other => panic!("expected GaussianBlur, got {other:?}"),
        }
        assert!(
            outcome
                .warnings
                .iter()
                .any(|w| w.contains("estimated sigma"))
        );
    }

    #[test]
    fn box_blur_mode_maps_to_the_plain_blur_command() {
        let json =
            pipeline_with_step(r#"{ "$blur": { "mode": "Blur", "kernelSize": 5, "repeat": 1 } }"#);
        let outcome = import_legacy_project(&json).unwrap();
        match only_command(&outcome) {
            PipelineCommand::Blur(s) => assert_eq!(s.kernel_size, 5),
            other => panic!("expected Blur, got {other:?}"),
        }
    }

    #[test]
    fn misspelled_legacy_threshold_methods_map_to_the_correctly_spelled_new_variants() {
        let json = pipeline_with_step(
            r#"{ "$threshold": { "modelClasses": [
                { "method": "Percentil", "thresholdMin": 1, "thresholdMax": 2, "pixelClassId": 1 },
                { "method": "TenyiEntropy", "thresholdMin": 3, "thresholdMax": 4, "pixelClassId": 2 }
            ] } }"#,
        );
        let outcome = import_legacy_project(&json).unwrap();
        match only_command(&outcome) {
            PipelineCommand::Threshold(s) => {
                assert_eq!(s.thresholds.len(), 2);
                assert_eq!(
                    s.thresholds[0].method,
                    SegmentationThresholdThresholdMethodSettings::Percentile
                );
                assert_eq!(
                    s.thresholds[1].method,
                    SegmentationThresholdThresholdMethodSettings::RenyiEntropy
                );
            }
            other => panic!("expected Threshold, got {other:?}"),
        }
        // Correctly-spelled, recognized methods should not trigger a warning.
        assert!(!outcome.warnings.iter().any(|w| w.contains("unrecognized")));
    }

    #[test]
    fn color_filter_converts_hue_scale_and_normalizes_sat_val() {
        let json = pipeline_with_step(
            r#"{ "$colorFilter": { "filter": [
                { "colorRangeFrom": {"hue": 0, "sat": 0, "val": 0}, "colorRangeTo": {"hue": 255, "sat": 255, "val": 255} }
            ], "grayScaleConvertMode": "Linear" } }"#,
        );
        let outcome = import_legacy_project(&json).unwrap();
        match only_command(&outcome) {
            PipelineCommand::ColorFilterCommand(s) => {
                assert!((s.range.min_h - 0.0).abs() < 1e-6);
                assert!((s.range.max_h - 360.0).abs() < 1e-3);
                assert!((s.range.min_s - 0.0).abs() < 1e-6);
                assert!((s.range.max_s - 1.0).abs() < 1e-6);
                assert!((s.range.max_v - 1.0).abs() < 1e-6);
            }
            other => panic!("expected ColorFilterCommand, got {other:?}"),
        }
    }

    #[test]
    fn legacy_multi_plate_list_is_used_when_the_singular_plate_is_unset() {
        let json = r##"{
            "meta": { "name": "X" },
            "projectSettings": {
                "classification": { "classes": [] },
                "plate": {},
                "plates": [ { "imageFolder": "old_images", "groupBy": "Filename", "plateSetup": { "rows": 1, "cols": 1, "wellImageOrder": [[1]] } } ]
            },
            "imageSetup": {},
            "pipelines": []
        }"##;
        let outcome = import_legacy_project(json).unwrap();
        assert_eq!(outcome.legacy_image_folder, Some("old_images".to_string()));
        assert_eq!(outcome.project.plate.grouping_mode, GroupingMode::FileName);
        assert!(
            outcome
                .warnings
                .iter()
                .any(|w| w.contains("legacy multi-plate list"))
        );
    }

    #[test]
    fn classify_with_multiple_filters_expands_into_one_classify_objects_step_per_filter() {
        let json = pipeline_with_step(
            r#"{ "$classify": { "modelClasses": [ {
                "filters": [
                    { "outputClass": "1", "intensity": {"minIntensity": -1, "maxIntensity": -1}, "metrics": { "minParticleSize": -1, "maxParticleSize": -1, "minCircularity": -1, "excludeObjectsAtTheEdge": false } },
                    { "outputClass": "2", "intensity": {"minIntensity": -1, "maxIntensity": -1}, "metrics": { "minParticleSize": -1, "maxParticleSize": -1, "minCircularity": -1, "excludeObjectsAtTheEdge": true } }
                ],
                "outputClassNoMatch": "Undefined",
                "pixelClassId": 3
            } ] } }"#,
        );
        let outcome = import_legacy_project(&json).unwrap();
        let steps = &outcome.project.pipelines[0].steps;
        assert_eq!(steps.len(), 2);
        for step in steps {
            match &step.command {
                PipelineCommand::ClassifyObjects(c) => {
                    assert_eq!(c.origin_segmentation, vec![SegmentationClass(3)]);
                }
                other => panic!("expected ClassifyObjects, got {other:?}"),
            }
        }
    }

    #[test]
    fn parse_numeric_class_id_accepts_plain_digits_and_rejects_everything_else() {
        assert_eq!(parse_numeric_class_id("7"), Some(7));
        assert_eq!(parse_numeric_class_id("0"), Some(0));
        assert_eq!(
            parse_numeric_class_id("M01"),
            None,
            "temp-class ids have no numeric equivalent"
        );
        assert_eq!(parse_numeric_class_id("$"), None);
        assert_eq!(parse_numeric_class_id(""), None);
    }

    #[test]
    fn resolve_class_substitutes_the_pipeline_default_for_the_dollar_placeholder() {
        let mut warnings = Vec::new();
        let resolved = resolve_class("$", ObjectClass::Valid(5), "ctx", "field", &mut warnings);
        assert_eq!(resolved, ObjectClass::Valid(5));
        assert!(warnings.is_empty());
    }

    #[test]
    fn resolve_class_treats_none_undefined_and_empty_as_unset_without_a_warning() {
        let mut warnings = Vec::new();
        for s in ["", "None", "Undefined"] {
            assert_eq!(
                resolve_class(s, ObjectClass::Valid(5), "ctx", "field", &mut warnings),
                ObjectClass::Unset
            );
        }
        assert!(warnings.is_empty());
    }

    #[test]
    fn resolve_class_parses_a_plain_number_directly() {
        let mut warnings = Vec::new();
        assert_eq!(
            resolve_class("12", ObjectClass::Valid(5), "ctx", "field", &mut warnings),
            ObjectClass::Valid(12)
        );
        assert!(warnings.is_empty());
    }

    #[test]
    fn resolve_class_warns_and_falls_back_to_unset_for_an_unsupported_temp_class_id() {
        let mut warnings = Vec::new();
        let resolved = resolve_class(
            "M01",
            ObjectClass::Valid(5),
            "my-context",
            "my-field",
            &mut warnings,
        );
        assert_eq!(resolved, ObjectClass::Unset);
        assert_eq!(warnings.len(), 1);
        assert!(warnings[0].contains("my-context"));
        assert!(warnings[0].contains("my-field"));
        assert!(warnings[0].contains("M01"));
    }

    #[test]
    fn parse_hex_color_accepts_a_leading_hash_or_bare_hex() {
        assert_eq!(parse_hex_color("#ff0000"), Some(0xff0000));
        assert_eq!(parse_hex_color("00ff00"), Some(0x00ff00));
        assert_eq!(
            parse_hex_color("#0000FF"),
            Some(0x0000ff),
            "hex parsing must be case-insensitive"
        );
    }

    #[test]
    fn parse_hex_color_rejects_non_hex_input() {
        assert_eq!(parse_hex_color("not-a-color"), None);
        assert_eq!(parse_hex_color(""), None);
    }

    #[test]
    fn legacy_unit_to_nm_factor_covers_every_known_unit() {
        assert_eq!(legacy_unit_to_nm_factor("nm"), 1.0);
        assert_eq!(legacy_unit_to_nm_factor("um"), 1_000.0);
        assert_eq!(legacy_unit_to_nm_factor("mm"), 1_000_000.0);
        assert_eq!(legacy_unit_to_nm_factor("cm"), 10_000_000.0);
        assert_eq!(legacy_unit_to_nm_factor("m"), 1_000_000_000.0);
        assert_eq!(legacy_unit_to_nm_factor("km"), 1_000_000_000_000.0);
    }

    #[test]
    fn legacy_unit_to_nm_factor_treats_unitless_values_as_already_nm() {
        assert_eq!(legacy_unit_to_nm_factor("Px"), 1.0);
        assert_eq!(legacy_unit_to_nm_factor("Undefined"), 1.0);
        assert_eq!(legacy_unit_to_nm_factor("bogus"), 1.0);
    }

    #[test]
    fn map_z_projection_covers_every_known_mode_and_falls_back_to_single_stack() {
        assert_eq!(
            map_z_projection("MaxIntensity"),
            ZStackHandling::MaxIntensity
        );
        assert_eq!(
            map_z_projection("MinIntensity"),
            ZStackHandling::MinIntensity
        );
        assert_eq!(
            map_z_projection("AvgIntensity"),
            ZStackHandling::AvgIntensity
        );
        assert_eq!(
            map_z_projection("TakeMiddle"),
            ZStackHandling::TakeTheMiddle
        );
        assert_eq!(map_z_projection("SingleStack"), ZStackHandling::SingleStack);
        assert_eq!(
            map_z_projection("SomethingUnknown"),
            ZStackHandling::SingleStack
        );
    }

    #[test]
    fn map_threshold_method_covers_the_misspelled_legacy_names_and_falls_back_to_none() {
        use SegmentationThresholdThresholdMethodSettings as M;
        assert_eq!(map_threshold_method("Percentil"), M::Percentile);
        assert_eq!(map_threshold_method("TenyiEntropy"), M::RenyiEntropy);
        assert_eq!(
            map_threshold_method("Otsu"),
            M::Otsu {
                classes: SegmentationThresholdOtsuClassesSettings::Two
            }
        );
        assert_eq!(map_threshold_method("SomethingUnknown"), M::None);
    }

    // ---- metadata fallbacks ----

    #[test]
    fn metadata_falls_back_to_experiment_name_and_notes_when_meta_fields_are_empty() {
        let json = r##"{
            "meta": {},
            "projectSettings": {
                "experimentSettings": { "experimentName": "Fallback Name", "notes": "fallback notes" },
                "classification": { "classes": [] },
                "plate": {}
            },
            "imageSetup": {},
            "pipelines": []
        }"##;
        let outcome = import_legacy_project(json).unwrap();
        assert_eq!(outcome.project.metadata.name, "Fallback Name");
        assert_eq!(outcome.project.metadata.description, "fallback notes");
        assert!(outcome.project.metadata.authors.is_empty());
        assert_eq!(outcome.project.metadata.author_organization, "");
    }

    #[test]
    fn metadata_parses_a_valid_rfc3339_modified_at_timestamp() {
        let json = r##"{
            "meta": { "name": "X", "modifiedAt": "2020-01-02T03:04:05Z" },
            "projectSettings": { "classification": { "classes": [] }, "plate": {} },
            "imageSetup": {},
            "pipelines": []
        }"##;
        let outcome = import_legacy_project(json).unwrap();
        let expected = chrono::DateTime::parse_from_rfc3339("2020-01-02T03:04:05Z")
            .unwrap()
            .with_timezone(&chrono::Utc);
        assert_eq!(outcome.project.metadata.creation_time, expected);
    }

    #[test]
    fn metadata_falls_back_to_now_for_an_unparseable_modified_at() {
        let json = r##"{
            "meta": { "name": "X", "modifiedAt": "not-a-date" },
            "projectSettings": { "classification": { "classes": [] }, "plate": {} },
            "imageSetup": {},
            "pipelines": []
        }"##;
        let before = chrono::Utc::now();
        let outcome = import_legacy_project(json).unwrap();
        let after = chrono::Utc::now();
        assert!(outcome.project.metadata.creation_time >= before);
        assert!(outcome.project.metadata.creation_time <= after);
    }

    // ---- classification edge cases ----

    #[test]
    fn class_color_falls_back_to_a_default_when_unparseable() {
        let json = r##"{
            "meta": { "name": "X" },
            "projectSettings": {
                "classification": { "classes": [ { "classId": "1", "name": "c", "color": "not-a-color", "notes": "" } ] },
                "plate": {}
            },
            "imageSetup": {},
            "pipelines": []
        }"##;
        let outcome = import_legacy_project(json).unwrap();
        // classes()[0] is the auto-prepended Background class; the imported
        // one (with the unparseable color) lands at index 1.
        assert_eq!(outcome.project.classification.classes()[1].color, 0x9933FF);
    }

    // ---- plate edge cases ----

    #[test]
    fn plate_group_by_off_falls_back_to_no_grouping() {
        let json = r##"{
            "meta": { "name": "X" },
            "projectSettings": { "classification": { "classes": [] },
                "plate": { "imageFolder": "img", "groupBy": "Off" } },
            "imageSetup": {},
            "pipelines": []
        }"##;
        let outcome = import_legacy_project(json).unwrap();
        assert_eq!(
            outcome.project.plate.grouping_mode,
            GroupingMode::NoGrouping
        );
    }

    #[test]
    fn plate_well_dimensions_fall_back_to_defaults_when_unset() {
        let json = r##"{
            "meta": { "name": "X" },
            "projectSettings": { "classification": { "classes": [] },
                "plate": { "imageFolder": "img" } },
            "imageSetup": {},
            "pipelines": []
        }"##;
        let outcome = import_legacy_project(json).unwrap();
        let default = PlateSettings::default();
        assert_eq!(outcome.project.plate.well_cols, default.well_cols);
        assert_eq!(outcome.project.plate.well_rows, default.well_rows);
        assert_eq!(
            outcome.project.plate.well_image_order,
            default.well_image_order
        );
    }

    #[test]
    fn no_plate_configured_at_all_yields_no_image_folder_and_no_warning() {
        let json = r##"{
            "meta": { "name": "X" },
            "projectSettings": { "classification": { "classes": [] }, "plate": {} },
            "imageSetup": {},
            "pipelines": []
        }"##;
        let outcome = import_legacy_project(json).unwrap();
        assert_eq!(outcome.legacy_image_folder, None);
        assert!(outcome.warnings.is_empty());
    }

    // ---- image setup edge cases ----

    #[test]
    fn automatic_pixel_size_mode_leaves_pixel_sizes_unset() {
        let json = r##"{
            "meta": { "name": "X" },
            "projectSettings": { "classification": { "classes": [] }, "plate": {} },
            "imageSetup": { "imagePixelSizeSettings": { "mode": "Automatic", "pixelSizeUnit": "um", "pixelWidth": 1.0, "pixelHeight": 1.0 } },
            "pipelines": []
        }"##;
        let outcome = import_legacy_project(json).unwrap();
        assert!(outcome.project.images.settings.pixel_sizes.is_none());
    }

    // ---- pipeline-level edge cases ----

    #[test]
    fn pipeline_without_a_name_uses_a_numbered_placeholder_in_warning_context() {
        let json = r##"{
            "meta": { "name": "X" },
            "projectSettings": { "classification": { "classes": [] }, "plate": {} },
            "imageSetup": {},
            "pipelines": [ { "pipelineSetup": { "source": "FromMemory", "defaultClassId": "1" },
                "pipelineSteps": [ { "$houghTransform": {} } ] } ]
        }"##;
        let outcome = import_legacy_project(json).unwrap();
        assert_eq!(outcome.project.pipelines[0].name, None);
        assert!(outcome.warnings.iter().any(|w| w.contains("pipeline #0")));
    }

    #[test]
    fn disabled_pipeline_and_disabled_step_are_carried_over() {
        let json = r##"{
            "meta": { "name": "X" },
            "projectSettings": { "classification": { "classes": [] }, "plate": {} },
            "imageSetup": {},
            "pipelines": [ { "meta": { "name": "P" }, "disabled": true,
                "pipelineSetup": { "source": "FromMemory", "defaultClassId": "1" },
                "pipelineSteps": [ { "disabled": true, "$laplacian": { "kernelSize": 3 } } ] } ]
        }"##;
        let outcome = import_legacy_project(json).unwrap();
        assert!(!outcome.project.pipelines[0].enabled);
        assert!(!outcome.project.pipelines[0].steps[0].enabled);
    }

    // ---- previously-untested filter/segmentation/object converters ----

    #[test]
    fn canny_converts_kernel_size_and_normalizes_thresholds_to_the_unit_range() {
        let json = pipeline_with_step(
            r#"{ "$canny": { "kernelSize": 5, "thresholdMin": 51.0, "thresholdMax": 204.0 } }"#,
        );
        let outcome = import_legacy_project(&json).unwrap();
        match only_command(&outcome) {
            PipelineCommand::EdgeDetectionCanny(s) => {
                assert_eq!(s.kernel_size, 5);
                assert!((s.threshold_min - 0.2).abs() < 1e-6);
                assert!((s.threshold_max - 0.8).abs() < 1e-6);
            }
            other => panic!("expected EdgeDetectionCanny, got {other:?}"),
        }
    }

    #[test]
    fn sobel_keeps_only_the_kernel_size_and_warns_about_dropped_fields() {
        let json = pipeline_with_step(r#"{ "$sobel": { "kernelSize": 3 } }"#);
        let outcome = import_legacy_project(&json).unwrap();
        match only_command(&outcome) {
            PipelineCommand::EdgeDetectionSobel(s) => assert_eq!(s.kernel_size, 3),
            other => panic!("expected EdgeDetectionSobel, got {other:?}"),
        }
        assert!(outcome.warnings.iter().any(|w| w.contains("$sobel")));
    }

    #[test]
    fn rolling_ball_defaults_to_the_ball_type_for_a_ball_value() {
        let json =
            pipeline_with_step(r#"{ "$rollingBall": { "ballType": "Ball", "ballSize": 10 } }"#);
        let outcome = import_legacy_project(&json).unwrap();
        match only_command(&outcome) {
            PipelineCommand::RollingBall(s) => {
                assert_eq!(s.ball_type, FiltersRollingBallBallTypeSettings::Ball);
                assert_eq!(s.radius, 10.0);
                assert!(!s.pre_smooth);
            }
            other => panic!("expected RollingBall, got {other:?}"),
        }
    }

    #[test]
    fn median_subtract_converts_kernel_diameter_to_radius() {
        let json = pipeline_with_step(r#"{ "$medianSubtract": { "kernelSize": 5 } }"#);
        let outcome = import_legacy_project(&json).unwrap();
        match only_command(&outcome) {
            PipelineCommand::MedianSubtract(s) => assert_eq!(s.radius, 2.0),
            other => panic!("expected MedianSubtract, got {other:?}"),
        }
    }

    #[test]
    fn median_subtract_clamps_radius_to_zero_for_a_small_kernel() {
        let json = pipeline_with_step(r#"{ "$medianSubtract": { "kernelSize": 0 } }"#);
        let outcome = import_legacy_project(&json).unwrap();
        match only_command(&outcome) {
            PipelineCommand::MedianSubtract(s) => assert_eq!(s.radius, 0.0),
            other => panic!("expected MedianSubtract, got {other:?}"),
        }
    }

    #[test]
    fn enhance_contrast_fields_pass_through_unchanged() {
        let json = pipeline_with_step(
            r#"{ "$enhanceContrast": { "saturatedPixels": 0.35, "normalize": true, "equalizeHistogram": false } }"#,
        );
        let outcome = import_legacy_project(&json).unwrap();
        match only_command(&outcome) {
            PipelineCommand::EnhanceContrast(s) => {
                assert_eq!(s.saturated_pixels, 0.35);
                assert!(s.normalize);
                assert!(!s.equalize_histogram);
            }
            other => panic!("expected EnhanceContrast, got {other:?}"),
        }
    }

    #[test]
    fn hessian_maps_every_documented_mode() {
        let cases = [
            (
                "EigenvaluesX",
                FiltersHessianHessianModeSettings::EigenvaluesX,
            ),
            (
                "EigenvaluesY",
                FiltersHessianHessianModeSettings::EigenvaluesY,
            ),
            (
                "Determinant",
                FiltersHessianHessianModeSettings::Determinant,
            ),
        ];
        for (legacy, expected) in cases {
            let json =
                pipeline_with_step(&format!(r#"{{ "$hessian": {{ "mode": "{legacy}" }} }}"#));
            let outcome = import_legacy_project(&json).unwrap();
            match only_command(&outcome) {
                PipelineCommand::Hessian(s) => assert_eq!(s.mode, expected, "mode {legacy}"),
                other => panic!("expected Hessian, got {other:?}"),
            }
        }
    }

    #[test]
    fn hessian_defaults_to_determinant_for_an_unrecognized_mode_with_a_warning() {
        let json = pipeline_with_step(r#"{ "$hessian": { "mode": "Bogus" } }"#);
        let outcome = import_legacy_project(&json).unwrap();
        match only_command(&outcome) {
            PipelineCommand::Hessian(s) => {
                assert_eq!(s.mode, FiltersHessianHessianModeSettings::Determinant)
            }
            other => panic!("expected Hessian, got {other:?}"),
        }
        assert!(
            outcome
                .warnings
                .iter()
                .any(|w| w.contains("$hessian") && w.contains("Bogus"))
        );
    }

    #[test]
    fn structure_tensor_maps_every_documented_mode_and_keeps_kernel_size() {
        let cases = [
            (
                "EigenvaluesX",
                FiltersStructureTensorTensorModeSettings::EigenvaluesX,
            ),
            (
                "EigenvaluesY",
                FiltersStructureTensorTensorModeSettings::EigenvaluesY,
            ),
            (
                "Coherence",
                FiltersStructureTensorTensorModeSettings::Coherence,
            ),
        ];
        for (legacy, expected) in cases {
            let json = pipeline_with_step(&format!(
                r#"{{ "$structureTensor": {{ "mode": "{legacy}", "kernelSize": 7 }} }}"#
            ));
            let outcome = import_legacy_project(&json).unwrap();
            match only_command(&outcome) {
                PipelineCommand::StructureTensor(s) => {
                    assert_eq!(s.mode, expected, "mode {legacy}");
                    assert_eq!(s.kernel_size, 7);
                    assert_eq!(s.sigma, 1.5);
                }
                other => panic!("expected StructureTensor, got {other:?}"),
            }
        }
    }

    #[test]
    fn structure_tensor_defaults_to_coherence_for_an_unrecognized_mode_with_a_warning() {
        let json =
            pipeline_with_step(r#"{ "$structureTensor": { "mode": "Bogus", "kernelSize": 3 } }"#);
        let outcome = import_legacy_project(&json).unwrap();
        match only_command(&outcome) {
            PipelineCommand::StructureTensor(s) => {
                assert_eq!(s.mode, FiltersStructureTensorTensorModeSettings::Coherence)
            }
            other => panic!("expected StructureTensor, got {other:?}"),
        }
        assert!(
            outcome
                .warnings
                .iter()
                .any(|w| w.contains("$structureTensor"))
        );
    }

    #[test]
    fn gaussian_weighted_deviation_keeps_kernel_size_and_sigma() {
        let json =
            pipeline_with_step(r#"{ "$gaussianWeightedDev": { "kernelSize": 9, "sigma": 2.5 } }"#);
        let outcome = import_legacy_project(&json).unwrap();
        match only_command(&outcome) {
            PipelineCommand::WeightedDeviation(s) => {
                assert_eq!(s.kernel_size, 9);
                assert!((s.sigma - 2.5).abs() < 1e-6);
            }
            other => panic!("expected WeightedDeviation, got {other:?}"),
        }
    }

    #[test]
    fn rank_filter_maps_every_documented_mode() {
        let cases = [
            ("Mean", FiltersRankFilterRankFilterTypeSettings::Mean),
            ("Min", FiltersRankFilterRankFilterTypeSettings::Min),
            ("Max", FiltersRankFilterRankFilterTypeSettings::Max),
            ("Median", FiltersRankFilterRankFilterTypeSettings::Median),
        ];
        for (legacy, expected) in cases {
            let json = pipeline_with_step(&format!(
                r#"{{ "$rank": {{ "mode": "{legacy}", "radius": 3.0 }} }}"#
            ));
            let outcome = import_legacy_project(&json).unwrap();
            match only_command(&outcome) {
                PipelineCommand::RankFilter(s) => {
                    assert_eq!(s.filter_type, expected, "mode {legacy}");
                    assert_eq!(s.radius, 3.0);
                }
                other => panic!("expected RankFilter, got {other:?}"),
            }
        }
    }

    #[test]
    fn rank_filter_outliers_mode_gets_a_default_threshold() {
        let json = pipeline_with_step(r#"{ "$rank": { "mode": "Outliers", "radius": 2.0 } }"#);
        let outcome = import_legacy_project(&json).unwrap();
        match only_command(&outcome) {
            PipelineCommand::RankFilter(s) => assert_eq!(
                s.filter_type,
                FiltersRankFilterRankFilterTypeSettings::Outliers(50.0)
            ),
            other => panic!("expected RankFilter, got {other:?}"),
        }
    }

    #[test]
    fn rank_filter_defaults_to_median_for_an_unrecognized_mode_with_a_warning() {
        let json = pipeline_with_step(r#"{ "$rank": { "mode": "Variance", "radius": 1.0 } }"#);
        let outcome = import_legacy_project(&json).unwrap();
        match only_command(&outcome) {
            PipelineCommand::RankFilter(s) => {
                assert_eq!(
                    s.filter_type,
                    FiltersRankFilterRankFilterTypeSettings::Median
                )
            }
            other => panic!("expected RankFilter, got {other:?}"),
        }
        assert!(outcome.warnings.iter().any(|w| w.contains("$rank")));
    }

    #[test]
    fn laplacian_keeps_the_kernel_size() {
        let json = pipeline_with_step(r#"{ "$laplacian": { "kernelSize": 3 } }"#);
        let outcome = import_legacy_project(&json).unwrap();
        match only_command(&outcome) {
            PipelineCommand::Laplacian(s) => assert_eq!(s.kernel_size, 3),
            other => panic!("expected Laplacian, got {other:?}"),
        }
    }

    #[test]
    fn morphological_transform_maps_every_documented_function_and_shape() {
        let cases = [
            (
                "Dilate",
                MorphologyMorphologicalTransformationMorphOpsSettings::Dilate,
            ),
            (
                "Erode",
                MorphologyMorphologicalTransformationMorphOpsSettings::Erode,
            ),
            (
                "Open",
                MorphologyMorphologicalTransformationMorphOpsSettings::Open,
            ),
            (
                "Close",
                MorphologyMorphologicalTransformationMorphOpsSettings::Close,
            ),
        ];
        for (legacy, expected) in cases {
            let json = pipeline_with_step(&format!(
                r#"{{ "$morphologicalTransform": {{ "function": "{legacy}", "shape": "Cross", "kernelSize": 5 }} }}"#
            ));
            let outcome = import_legacy_project(&json).unwrap();
            match only_command(&outcome) {
                PipelineCommand::MorphologicalCommand(s) => {
                    assert_eq!(s.op, expected, "function {legacy}");
                    assert_eq!(
                        s.kernel_shape,
                        MorphologyMorphologicalTransformationKernelShapesSettings::Cross
                    );
                    assert_eq!(s.kernel_size, 5);
                    assert!(!s.use_grayscale);
                }
                other => panic!("expected MorphologicalCommand, got {other:?}"),
            }
        }
    }

    #[test]
    fn morphological_transform_shape_falls_back_to_box_for_rectangle_or_unknown() {
        let json = pipeline_with_step(
            r#"{ "$morphologicalTransform": { "function": "Erode", "shape": "Rectangle", "kernelSize": 3 } }"#,
        );
        let outcome = import_legacy_project(&json).unwrap();
        match only_command(&outcome) {
            PipelineCommand::MorphologicalCommand(s) => assert_eq!(
                s.kernel_shape,
                MorphologyMorphologicalTransformationKernelShapesSettings::Box
            ),
            other => panic!("expected MorphologicalCommand, got {other:?}"),
        }
    }

    #[test]
    fn morphological_transform_unrecognized_function_is_skipped_with_a_warning() {
        let json = pipeline_with_step(
            r#"{ "$morphologicalTransform": { "function": "Gradient", "shape": "Box", "kernelSize": 3 } }"#,
        );
        let outcome = import_legacy_project(&json).unwrap();
        assert!(outcome.project.pipelines[0].steps.is_empty());
        assert!(
            outcome
                .warnings
                .iter()
                .any(|w| w.contains("$morphologicalTransform") && w.contains("Gradient"))
        );
    }

    #[test]
    fn intensity_transform_automatic_mode_scales_brightness_into_the_unit_range() {
        let json = pipeline_with_step(
            r#"{ "$intensityTransform": { "mode": "Automatic", "contrast": 1.5, "brightness": 32767, "gamma": 0 } }"#,
        );
        let outcome = import_legacy_project(&json).unwrap();
        match only_command(&outcome) {
            PipelineCommand::IntensityTransformation(s) => {
                assert_eq!(
                    s.mode,
                    FiltersIntensityTransformIntensityTransformModeSettings::Automatic
                );
                assert_eq!(s.contrast, 1.5);
                assert!((s.brightness - 0.5).abs() < 1e-3);
            }
            other => panic!("expected IntensityTransformation, got {other:?}"),
        }
        assert!(outcome.warnings.is_empty());
    }

    #[test]
    fn intensity_transform_manual_mode_and_gamma_warning() {
        let json = pipeline_with_step(
            r#"{ "$intensityTransform": { "mode": "Manual", "contrast": 1.0, "brightness": 0, "gamma": 2 } }"#,
        );
        let outcome = import_legacy_project(&json).unwrap();
        match only_command(&outcome) {
            PipelineCommand::IntensityTransformation(s) => assert_eq!(
                s.mode,
                FiltersIntensityTransformIntensityTransformModeSettings::Manual
            ),
            other => panic!("expected IntensityTransformation, got {other:?}"),
        }
        assert!(outcome.warnings.iter().any(|w| w.contains("gamma")));
    }

    #[test]
    fn voronoi_uses_the_first_center_and_mask_class_and_carries_the_rest_as_filters() {
        let json = pipeline_with_step(
            r#"{ "$voronoi": {
                "inputClassesPoints": ["1", "2"],
                "outputClassVoronoi": "3",
                "inputClassesMask": ["4", "5"],
                "excludeAreasWithoutPoint": true,
                "excludeAreasAtTheEdge": false,
                "maxRadius": 12.5
            } }"#,
        );
        let outcome = import_legacy_project(&json).unwrap();
        match only_command(&outcome) {
            PipelineCommand::Voronoi(s) => {
                assert_eq!(s.centers, ObjectClass::Valid(1));
                assert_eq!(s.center_filter_classes, vec![ObjectClass::Valid(2)]);
                assert_eq!(s.mask, ObjectClass::Valid(4));
                assert_eq!(s.mask_filter_classes, vec![ObjectClass::Valid(5)]);
                assert_eq!(s.output_class, ObjectClass::Valid(3));
                assert_eq!(s.max_radius, 12.5);
                assert!(s.exclude_areas_with_no_center);
                assert!(!s.exclude_areas_at_the_edges);
            }
            other => panic!("expected Voronoi, got {other:?}"),
        }
        assert!(
            outcome
                .warnings
                .iter()
                .any(|w| w.contains("$voronoi") && w.contains("2 center classes"))
        );
    }

    #[test]
    fn voronoi_with_a_single_center_and_no_mask_has_no_extra_center_warning() {
        let json = pipeline_with_step(
            r#"{ "$voronoi": { "inputClassesPoints": ["1"], "outputClassVoronoi": "2", "inputClassesMask": [] } }"#,
        );
        let outcome = import_legacy_project(&json).unwrap();
        match only_command(&outcome) {
            PipelineCommand::Voronoi(s) => {
                assert_eq!(s.mask, ObjectClass::Unset);
                assert!(s.mask_filter_classes.is_empty());
            }
            other => panic!("expected Voronoi, got {other:?}"),
        }
        assert!(
            !outcome
                .warnings
                .iter()
                .any(|w| w.contains("center classes"))
        );
    }

    #[test]
    fn color_filter_with_multiple_ranges_keeps_only_the_first_and_warns() {
        let json = pipeline_with_step(
            r#"{ "$colorFilter": { "filter": [
                { "colorRangeFrom": {"hue": 0, "sat": 0, "val": 0}, "colorRangeTo": {"hue": 128, "sat": 128, "val": 128} },
                { "colorRangeFrom": {"hue": 10, "sat": 10, "val": 10}, "colorRangeTo": {"hue": 20, "sat": 20, "val": 20} }
            ] } }"#,
        );
        let outcome = import_legacy_project(&json).unwrap();
        assert!(
            outcome
                .warnings
                .iter()
                .any(|w| w.contains("2 color ranges"))
        );
    }

    #[test]
    fn color_filter_with_no_ranges_produces_no_command() {
        let json = pipeline_with_step(r#"{ "$colorFilter": { "filter": [] } }"#);
        let outcome = import_legacy_project(&json).unwrap();
        assert!(outcome.project.pipelines[0].steps.is_empty());
    }

    #[test]
    fn box_blur_with_a_repeat_other_than_one_is_applied_once_with_a_warning() {
        let json =
            pipeline_with_step(r#"{ "$blur": { "mode": "Blur", "kernelSize": 3, "repeat": 3 } }"#);
        let outcome = import_legacy_project(&json).unwrap();
        match only_command(&outcome) {
            PipelineCommand::Blur(s) => assert_eq!(s.kernel_size, 3),
            other => panic!("expected Blur, got {other:?}"),
        }
        assert!(outcome.warnings.iter().any(|w| w.contains("repeat=3")));
    }

    #[test]
    fn threshold_with_no_model_classes_produces_no_threshold_command_but_still_extracts_objects() {
        let json = pipeline_with_step(r#"{ "$threshold": { "modelClasses": [] } }"#);
        let outcome = import_legacy_project(&json).unwrap();
        let steps = &outcome.project.pipelines[0].steps;
        assert_eq!(steps.len(), 2);
        assert!(matches!(
            steps[0].command,
            PipelineCommand::ConnectedComponents(_)
        ));
        assert!(matches!(
            steps[1].command,
            PipelineCommand::ExtractObjects(_)
        ));
    }

    #[test]
    fn threshold_empty_or_none_method_is_the_none_sentinel_without_a_warning() {
        let json = pipeline_with_step(
            r#"{ "$threshold": { "modelClasses": [
                { "method": "", "thresholdMin": 1, "thresholdMax": 2, "pixelClassId": 1 },
                { "method": "None", "thresholdMin": 1, "thresholdMax": 2, "pixelClassId": 2 }
            ] } }"#,
        );
        let outcome = import_legacy_project(&json).unwrap();
        match &outcome.project.pipelines[0].steps[0].command {
            PipelineCommand::Threshold(s) => {
                assert!(
                    s.thresholds
                        .iter()
                        .all(|t| t.method == SegmentationThresholdThresholdMethodSettings::None)
                );
            }
            other => panic!("expected Threshold, got {other:?}"),
        }
        assert!(!outcome.warnings.iter().any(|w| w.contains("unrecognized")));
    }

    #[test]
    fn threshold_unrecognized_nonempty_method_warns_and_defaults_to_none() {
        let json = pipeline_with_step(
            r#"{ "$threshold": { "modelClasses": [
                { "method": "Bogus", "thresholdMin": 1, "thresholdMax": 2, "pixelClassId": 1 }
            ] } }"#,
        );
        let outcome = import_legacy_project(&json).unwrap();
        assert!(
            outcome
                .warnings
                .iter()
                .any(|w| w.contains("Bogus") && w.contains("unrecognized"))
        );
    }

    #[test]
    fn watershed_tolerance_is_clamped_to_a_minimum_of_0_1() {
        let json = pipeline_with_step(r#"{ "$watershed": { "maximumFinderTolerance": 0.0 } }"#);
        let outcome = import_legacy_project(&json).unwrap();
        match &outcome.project.pipelines[0].steps[0].command {
            PipelineCommand::Watershed(s) => assert_eq!(s.maximum_finder_tolerance, 0.1),
            other => panic!("expected Watershed, got {other:?}"),
        }
    }

    #[test]
    fn watershed_tolerance_above_the_minimum_is_kept_as_is() {
        let json = pipeline_with_step(r#"{ "$watershed": { "maximumFinderTolerance": 5.0 } }"#);
        let outcome = import_legacy_project(&json).unwrap();
        match &outcome.project.pipelines[0].steps[0].command {
            PipelineCommand::Watershed(s) => assert_eq!(s.maximum_finder_tolerance, 5.0),
            other => panic!("expected Watershed, got {other:?}"),
        }
    }

    #[test]
    fn classify_warns_about_no_match_class_and_intensity_filters() {
        let json = pipeline_with_step(
            r#"{ "$classify": { "modelClasses": [ {
                "filters": [ { "outputClass": "2", "intensity": {"minIntensity": 5, "maxIntensity": -1}, "metrics": { "minParticleSize": 1, "maxParticleSize": -1, "minCircularity": -1, "excludeObjectsAtTheEdge": false } } ],
                "outputClassNoMatch": "3",
                "pixelClassId": 1
            } ] } }"#,
        );
        let outcome = import_legacy_project(&json).unwrap();
        assert!(outcome.warnings.iter().any(|w| w.contains("no match")));
        assert!(
            outcome
                .warnings
                .iter()
                .any(|w| w.contains("intensity filter"))
        );
    }

    #[test]
    fn object_transform_maps_snap_area_and_circle_functions() {
        let json = pipeline_with_step(
            r#"{ "$objectTransform": { "function": "SnapArea", "inputClasses": "1", "outputClasses": "2", "factor": 3.0 } }"#,
        );
        let outcome = import_legacy_project(&json).unwrap();
        match only_command(&outcome) {
            PipelineCommand::TransformObjects(s) => assert_eq!(
                s.function,
                ClassificationTransformObjectsTransformFunctionSettings::SnapArea {
                    extra_size: 3.0,
                    unit: SizeUnits::Pixels,
                }
            ),
            other => panic!("expected TransformObjects, got {other:?}"),
        }

        let json = pipeline_with_step(
            r#"{ "$objectTransform": { "function": "CircleMin", "inputClasses": "1", "outputClasses": "2", "factor": 3.0 } }"#,
        );
        let outcome = import_legacy_project(&json).unwrap();
        match only_command(&outcome) {
            PipelineCommand::TransformObjects(s) => assert_eq!(
                s.function,
                ClassificationTransformObjectsTransformFunctionSettings::MinCircle {
                    min_diameter: 6.0,
                    unit: SizeUnits::Pixels,
                }
            ),
            other => panic!("expected TransformObjects, got {other:?}"),
        }

        let json = pipeline_with_step(
            r#"{ "$objectTransform": { "function": "Circle", "inputClasses": "1", "outputClasses": "2", "factor": 3.0 } }"#,
        );
        let outcome = import_legacy_project(&json).unwrap();
        match only_command(&outcome) {
            PipelineCommand::TransformObjects(s) => assert_eq!(
                s.function,
                ClassificationTransformObjectsTransformFunctionSettings::DrawCircle {
                    diameter: 6.0,
                    unit: SizeUnits::Pixels,
                }
            ),
            other => panic!("expected TransformObjects, got {other:?}"),
        }
    }

    #[test]
    fn object_transform_maps_fit_ellipse_with_a_fixed_scale() {
        let json = pipeline_with_step(
            r#"{ "$objectTransform": { "function": "FitEllipse", "inputClasses": "1", "outputClasses": "2", "factor": 9.0 } }"#,
        );
        let outcome = import_legacy_project(&json).unwrap();
        match only_command(&outcome) {
            PipelineCommand::TransformObjects(s) => assert_eq!(
                s.function,
                ClassificationTransformObjectsTransformFunctionSettings::FittingEllipse {
                    scale: 1.0
                }
            ),
            other => panic!("expected TransformObjects, got {other:?}"),
        }
    }

    #[test]
    fn object_transform_unrecognized_function_is_skipped_with_a_warning() {
        let json = pipeline_with_step(
            r#"{ "$objectTransform": { "function": "Bogus", "inputClasses": "1", "outputClasses": "2", "factor": 1.0 } }"#,
        );
        let outcome = import_legacy_project(&json).unwrap();
        assert!(outcome.project.pipelines[0].steps.is_empty());
        assert!(
            outcome
                .warnings
                .iter()
                .any(|w| w.contains("$objectTransform") && w.contains("Bogus"))
        );
    }

    #[test]
    fn save_image_falls_back_to_a_default_name_when_the_path_has_no_file_stem() {
        let json = pipeline_with_step(r#"{ "$saveImage": { "path": "" } }"#);
        let outcome = import_legacy_project(&json).unwrap();
        match only_command(&outcome) {
            PipelineCommand::SaveImage(s) => assert_eq!(s.name, "image"),
            other => panic!("expected SaveImage, got {other:?}"),
        }
    }

    #[test]
    fn warn_unsupported_step_reports_every_unsupported_command_present() {
        let json = pipeline_with_step(
            r#"{ "$crop": {}, "$fillHoles": {}, "$nop": {}, "$aiClassify": {}, "$colocalization": {} }"#,
        );
        let outcome = import_legacy_project(&json).unwrap();
        assert!(outcome.project.pipelines[0].steps.is_empty());
        for name in [
            "$crop",
            "$fillHoles",
            "$nop",
            "$aiClassify",
            "$colocalization",
        ] {
            assert!(
                outcome.warnings.iter().any(|w| w.contains(name)),
                "missing warning for {name}, got: {:#?}",
                outcome.warnings
            );
        }
    }
}
