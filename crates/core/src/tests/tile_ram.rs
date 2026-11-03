//! # Tile commands: RAM
//!
//! What a command running on a tile needs beyond the worker's own buffers
//! (working image, scratch pad, segmentation and instance map - planned by
//! `memory::analysis::estimate_ram_budget`): each command with its default
//! settings on a synthetic tile (cells, after a threshold, connected
//! components and the objects), on a small and a four times larger one -
//! what it needs more per extra pixel - must stay within what the command
//! declares (`ImageAlgorithm::ram_per_tile_pixel`): the planner adds the
//! largest of a job's commands to each worker (TODO "RAM budget for the
//! commands", step 3).
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-11-03
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.

use crate::ImageDimensionality;
use crate::algos::ExecutionScope;
use crate::image::{ImageContainer, ManagedImage, Point2d};
use crate::job::algos_from_config::into_algorithm;
use crate::pipeline::pipeline::PipelineImageMeta;
use crate::pipeline::pipeline_cache::GlobalPipelineCache;
use crate::pipeline::pipeline_context::PipelineContext;
use crate::tests::counting_allocator;
use evanalyzer_cfg::core_types::{ImageTile, SegmentationClass};
use evanalyzer_cfg::settings::pipeline_command::{PipelineCommand, default_command};
use evanalyzer_cfg::settings::pipeline_command_settings::*;
use kornia_image::{Image, ImageSize};
use std::sync::Arc;

/// The sides of the small and the large tile (pixels).
const SMALL: usize = 512;
const LARGE: usize = 1024;
/// One cell per `SPACING`² pixels.
const SPACING: usize = 32;

/// The settings to run `command` with: its defaults, or (where they don't
/// run) valid ones.
fn configured(command: PipelineCommand) -> PipelineCommand {
    use PipelineCommand as C;
    match command {
        C::EdgeDetectionSobel(_) => {
            C::EdgeDetectionSobel(EdgeDetectionSobelSettings { kernel_size: 3 })
        }
        C::StructureTensor(s) => C::StructureTensor(StructureTensorSettings {
            kernel_size: 5,
            sigma: 1.0,
            ..s
        }),
        C::WeightedDeviation(_) => C::WeightedDeviation(WeightedDeviationSettings {
            kernel_size: 5,
            sigma: 1.0,
        }),
        other => other,
    }
}

/// Commands reading a color (RGB) image.
fn needs_rgb(command: &PipelineCommand) -> bool {
    matches!(
        command,
        PipelineCommand::ColorDeconvolution(_) | PipelineCommand::ColorFilterCommand(_)
    )
}

/// A tile of `side`² pixels (16-bit gray values: cells at ~4000 on ~500)
/// after a threshold, connected components and the objects - and the cache
/// with it as channel 0. `rgb`: the image as color (0..1 per channel).
fn tile(
    side: usize,
    rgb: bool,
    output: &std::path::Path,
) -> (PipelineContext, GlobalPipelineCache) {
    let size = ImageSize {
        width: side,
        height: side,
    };
    let values: Vec<f32> = (0..side * side)
        .map(|i| {
            let (x, y) = (i % side, i / side);
            let (cx, cy) = (x % SPACING, y % SPACING);
            let cell = (4..28).contains(&cx) && (4..28).contains(&cy);
            (if cell { 4000.0 } else { 500.0 }) + ((x * 7 + y * 13) % 97) as f32
        })
        .collect();
    let image = || match rgb {
        false => Arc::new(ImageContainer::F32Gray(ManagedImage {
            data: Image::new(size, values.clone()).unwrap(),
            tile_offset: Point2d { x: 0, y: 0 },
            plane: None,
            dimensionality: ImageDimensionality::Planar,
        })),
        true => Arc::new(ImageContainer::F32Rgb(ManagedImage {
            data: Image::<f32, 3>::new(
                size,
                values
                    .iter()
                    .flat_map(|v| [v / 5000.0, v / 7000.0, v / 9000.0])
                    .collect(),
            )
            .unwrap(),
            tile_offset: Point2d { x: 0, y: 0 },
            plane: None,
            dimensionality: ImageDimensionality::Planar,
        })),
    };
    let tile = ImageTile {
        offset_x: 0,
        offset_y: 0,
        width: side,
        height: side,
    };
    let meta = PipelineImageMeta {
        image_tile_info: tile,
        full_image_width: size,
        is_rgb: rgb,
        nr_of_bits: 16,
        pixel_sizes: Default::default(),
    };
    let mut cache = GlobalPipelineCache::default();
    cache.image_meta.nr_of_bits = 16;
    cache.image_meta.full_image_width = size;
    cache.add_to_channel_cache(image(), 0, tile);
    let mut ctx = PipelineContext::new_from_image(output.to_path_buf(), meta, image()).unwrap();
    for command in [
        PipelineCommand::Threshold(ThresholdSettings {
            thresholds: vec![ThresholdEntrySettings {
                min_threshold: 1500.0,
                segmentation_class: SegmentationClass(1),
                ..Default::default()
            }],
        }),
        PipelineCommand::ConnectedComponents(Default::default()),
        PipelineCommand::ExtractObjects(ExtractObjectsSettings {
            max_objects_before_fail: 1_000_000,
            ..Default::default()
        }),
    ] {
        // A color image goes into a color command as it is.
        if rgb {
            break;
        }
        into_algorithm(command)
            .unwrap()
            .run(&mut ctx, &mut cache)
            .unwrap();
    }
    (ctx, cache)
}

/// The most bytes `command` has in use more than before it ran, on a fresh
/// tile of `side`² pixels (one thread); `Err`: it failed there.
fn peak_bytes(command: &PipelineCommand, side: usize) -> Result<usize, String> {
    let algorithm = into_algorithm(command.clone()).map_err(|e| e.to_string())?;
    // Where Save Image writes (removed after).
    let output = tempfile::tempdir().map_err(|e| e.to_string())?;
    let (mut ctx, mut cache) = tile(side, needs_rgb(command), output.path());
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .unwrap();
    let before = counting_allocator::in_use();
    counting_allocator::reset_peak();
    pool.install(|| algorithm.run(&mut ctx, &mut cache))
        .map_err(|e| e.to_string())?;
    Ok(counting_allocator::peak() - before)
}

/// What a command may need more per pixel than it declares (measuring
/// noise: a few small buffers grow with the tile too).
const TOLERANCE: f64 = 0.25;

/// Tile commands that can't run here, and why.
fn not_measured(command: &PipelineCommand) -> Option<&'static str> {
    use PipelineCommand as C;
    match command {
        C::Cellpose(_) | C::Stardist(_) | C::UNet(_) | C::Yolov5(_) | C::PixelClassifier(_) => {
            Some("needs a model (and libtorch) - TODO: measure the AI models")
        }
        _ => None,
    }
}

/// Every tile command's RAM per tile pixel beyond the worker's buffers,
/// against what it declares - in a process of its own (the allocator counts
/// the whole process). Run by `tile_commands_need_what_they_declare`.
#[test]
#[ignore]
fn measure_tile_commands() {
    let mut failed = Vec::new();
    for command in (0..).map_while(default_command).map(configured) {
        let Ok(algorithm) = into_algorithm(command.clone()) else {
            continue;
        };
        if algorithm.execution_scope() != ExecutionScope::Tile {
            continue;
        }
        let name = algorithm.name();
        if let Some(why) = not_measured(&command) {
            println!("TILE {name}: not measured - {why}");
            continue;
        }
        let declared = algorithm.ram_per_tile_pixel();
        let start = std::time::Instant::now();
        match (peak_bytes(&command, SMALL), peak_bytes(&command, LARGE)) {
            (Ok(small), Ok(large)) => {
                let per_pixel =
                    large.saturating_sub(small) as f64 / (LARGE * LARGE - SMALL * SMALL) as f64;
                println!(
                    "TILE {name}: {:.1} MB / {:.1} MB, {per_pixel:.2} bytes per pixel, declared \
                     {declared} ({:.1} s)",
                    small as f64 / 1e6,
                    large as f64 / 1e6,
                    start.elapsed().as_secs_f64()
                );
                if per_pixel > declared + TOLERANCE {
                    failed.push(format!(
                        "{name}: {per_pixel:.2} bytes per tile pixel, declares {declared} \
                         (`ram_per_tile_pixel`)"
                    ));
                }
            }
            (Err(e), _) | (_, Err(e)) => failed.push(format!(
                "{name}: doesn't run with its settings here ({e}) - add them to `configured`"
            )),
        }
    }
    assert!(failed.is_empty(), "{}", failed.join("\n"));
}

/// Each tile command needs no more RAM per tile pixel than it declares -
/// the planner counts on it (see the module docs).
#[test]
fn tile_commands_need_what_they_declare() {
    let output = std::process::Command::new(std::env::current_exe().unwrap())
        .args([
            "--exact",
            "tests::tile_ram::measure_tile_commands",
            "--ignored",
            "--nocapture",
            "--test-threads=1",
        ])
        .output()
        .unwrap();
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    println!("{stdout}");
    assert!(
        output.status.success() && stdout.contains("1 passed"),
        "{stdout}\n{stderr}"
    );
}
