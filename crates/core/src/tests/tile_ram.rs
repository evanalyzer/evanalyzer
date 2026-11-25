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
        // The median's RAM doesn't depend on its radius; 30 px would take
        // minutes in a debug build.
        C::SubtractReferenceChannel(s) => {
            C::SubtractReferenceChannel(SubtractReferenceChannelSettings {
                background: MathSubtractReferenceChannelBackgroundSettings::Median { radius: 5.0 },
                // The test image has one channel: its own reference.
                reference_channel: evanalyzer_cfg::core_types::ImageChannelIdx(0),
                ..s
            })
        }
        // The defaults shift by 0 (a copy): measure the interpolation.
        C::TranslateImage(s) => C::TranslateImage(TranslateImageSettings {
            dy: 0.5,
            dx: -1.2,
            interpolation: SpartialTransformTranslateImageInterpolationSettings::Cubic,
            ..s
        }),
        C::WeightedDeviation(_) => C::WeightedDeviation(WeightedDeviationSettings {
            kernel_size: 5,
            sigma: 1.0,
        }),
        C::UNet(s) => C::UNet(UNetSettings {
            model_path: real_model(UNET),
            ..s
        }),
        C::Stardist(s) => C::Stardist(StardistSettings {
            model_path: real_model(STARDIST),
            ..s
        }),
        C::Cellpose(s) => C::Cellpose(CellposeSettings {
            model_path: real_model(CELLPOSE),
            ..s
        }),
        other => other,
    }
}

/// The AI steps run on the real models, converted to EVAnalyzer's files
/// (git-ignored, see `docs/nn_burn_migration_plan.md`): what the analysis
/// keeps of an inference depends on the outputs (StarDist's candidates) -
/// random weights would detect anything. Without them, not measured.
const UNET: &str = "tests/romantic-shark/romantic-shark.safetensors";
const STARDIST: &str = "tests/stardist/2D_versatile_fluo.safetensors";
const CELLPOSE: &str = "tests/cellpose-sam/cpsam_v2.safetensors";

fn real_model(file: &str) -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join(file)
}

/// Commands reading a color (RGB) image.
fn needs_rgb(command: &PipelineCommand) -> bool {
    matches!(
        command,
        PipelineCommand::ColorDeconvolution(_)
            | PipelineCommand::ColorFilterCommand(_)
            | PipelineCommand::ColorToGrayscale(_)
    )
}

/// A tile of `side`² pixels (16-bit gray values: cells at ~4000 on ~500)
/// after a threshold, connected components and the objects - and the cache
/// with it as channel 0. `rgb`: the image as color (0..1 per channel).
fn tile(
    side: usize,
    rgb: bool,
    normalized: bool,
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
        // The AI models were trained on normalized images (~0..1, as a
        // normalizing step before them gives).
        .map(|v| if normalized { v / 5000.0 } else { v })
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
            id: String::new(),
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
    let ai = is_ai_model(command);
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .unwrap();
    if ai {
        // Loads the model first (kept by the engine for the run): its
        // weights are the step's shared RAM, not per tile pixel.
        let (mut ctx, mut cache) = tile(64, false, true, output.path());
        pool.install(|| algorithm.run(&mut ctx, &mut cache))
            .map_err(|e| e.to_string())?;
    }
    let (mut ctx, mut cache) = tile(side, needs_rgb(command), ai, output.path());
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
        C::UNet(_) if !real_model(UNET).exists() => Some("needs the converted U-Net (see `UNET`)"),
        C::Stardist(_) if !real_model(STARDIST).exists() => {
            Some("needs the converted StarDist (see `STARDIST`)")
        }
        C::Cellpose(_) if !real_model(CELLPOSE).exists() => {
            Some("needs the converted Cellpose-SAM (see `CELLPOSE`)")
        }
        C::PixelClassifier(_) => Some("needs a trained classifier"),
        _ => None,
    }
}

/// Every tile command's RAM per tile pixel beyond the worker's buffers,
/// against what it declares - in a process of its own (the allocator counts
/// the whole process). Run by `tile_commands_need_what_they_declare`; the AI
/// model steps by `measure_ai_tile_commands`.
#[test]
#[ignore]
fn measure_tile_commands() {
    measure(|command| !is_ai_model(command));
}

/// The AI model steps: real networks, ~0.7 TFLOP per Cellpose-SAM tile -
/// hours in a debug build, so not part of
/// `tile_commands_need_what_they_declare`. Release build, with the converted
/// models (`UNET`, `STARDIST`, `CELLPOSE`):
/// `cargo test --release --target x86_64-unknown-linux-gnu -p evanalyzer_core --lib
/// measure_ai_tile_commands -- --ignored --nocapture --test-threads=1`.
#[test]
#[ignore]
fn measure_ai_tile_commands() {
    // The CPU: on a GPU the network's memory is the GPU's, not counted here
    // (a process of its own - the first configuration counts).
    crate::ai_devices::configure_ai_devices(crate::ai_devices::AiDeviceOptions {
        devices: crate::ai_devices::AiDeviceSelection::Cpu,
        ..Default::default()
    });
    measure(is_ai_model);
}

fn is_ai_model(command: &PipelineCommand) -> bool {
    use PipelineCommand as C;
    matches!(command, C::UNet(_) | C::Stardist(_) | C::Cellpose(_))
}

/// Every tile command `wanted` takes: its RAM per tile pixel against what
/// it declares.
fn measure(wanted: fn(&PipelineCommand) -> bool) {
    let mut failed = Vec::new();
    for command in (0..).map_while(default_command).map(configured) {
        if !wanted(&command) {
            continue;
        }
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
        // An AI model step: its network's activations are its shared RAM
        // (one inference at a time), on top of what it keeps per worker.
        let network = if is_ai_model(&command) {
            let shared = |side| algorithm.shared_ram_bytes(side, side) as f64;
            (shared(LARGE) - shared(SMALL)) / (LARGE * LARGE - SMALL * SMALL) as f64
        } else {
            0.0
        };
        let declared = algorithm.ram_per_tile_pixel() + network;
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
                // The networks' activations: up to 1 % off (another
                // summation of buffers per backend version).
                if per_pixel > declared + TOLERANCE + 0.01 * network {
                    failed.push(format!(
                        "{name}: {per_pixel:.2} bytes per tile pixel, declares {declared} \
                         (`ram_per_tile_pixel` + the network's per pixel)"
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
