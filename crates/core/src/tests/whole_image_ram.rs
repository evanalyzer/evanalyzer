//! # Whole-image commands: RAM
//!
//! A whole-image command runs once over a whole slide, after its tiles. Its
//! RAM must not grow with the slide's pixels: the tile cache keeps the
//! channels within the RAM budget (spilling the rest to disk), but a buffer
//! a command allocates for every pixel of the image (Propagation's 16 bytes
//! per pixel, 15 GB on a 1-gigapixel slide) - or the tiles it holds on to
//! all at once - are outside that budget. On a liver slide that closed the
//! app without a crash log (2026-10-29: out of memory aborts, no panic).
//!
//! Every whole-image command runs on a small and a four times larger
//! synthetic slide; what it needs more on the large one, per extra pixel,
//! must stay below [`MAX_BYTES_PER_PIXEL`]. Objects grow with the slide too
//! (one cell per 64 × 64 pixels), their own RAM is part of it. A new
//! whole-image command fails the test until it is added to [`configured`].
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-10-29
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.

use crate::ImageDimensionality;
use crate::algos::ExecutionScope;
use crate::image::{ImageContainer, ManagedImage, Point2d};
use crate::job::algos_from_config::into_algorithm;
use crate::object::{Object, ObjectInit};
use crate::pipeline::pipeline::PipelineImageMeta;
use crate::pipeline::pipeline_cache::GlobalPipelineCache;
use crate::pipeline::pipeline_context::PipelineContext;
use crate::pipeline::scope::Scope;
use crate::tests::counting_allocator;
use bitvec::prelude::*;
use evanalyzer_cfg::core_types::{ImageChannelIdx, ImageTile, ObjectClass, ObjectId, SizeUnits};
use evanalyzer_cfg::settings::pipeline_command::{PipelineCommand, default_command};
use evanalyzer_cfg::settings::pipeline_command_settings::*;
use kornia_image::{Image, ImageSize};
use std::sync::Arc;

/// The most RAM a whole-image command may need more per pixel of a larger
/// slide (its own objects and values included). Before 2026-10-29:
/// Propagation ~20, Voronoi 4, Measure Image Colocalization 8 (both channels'
/// tiles held), Measure Image Texture 4.
const MAX_BYTES_PER_PIXEL: f64 = 1.0;

/// The sides of the small and the large slide (pixels).
const SMALL: usize = 1024;
const LARGE: usize = 2048;
/// The side of a tile.
const TILE: usize = 512;
/// The tile cache's cap: both slides' channels spill to disk (a tile is
/// 1 MB), so the cache's own RAM is the same on both.
const CACHE_BYTES: u64 = 4 << 20;
/// One cell (with its nucleus and a spot) per `SPACING`² pixels.
const SPACING: usize = 64;

const NUCLEUS: ObjectClass = ObjectClass::Global(1);
const CELL: ObjectClass = ObjectClass::Global(2);
const SPOT: ObjectClass = ObjectClass::Global(3);
const OUTPUT: ObjectClass = ObjectClass::Global(9);

/// The whole-image commands left out, and why.
fn not_measured(command: &PipelineCommand) -> Option<&'static str> {
    match command {
        PipelineCommand::AiObjectClassifier(_) => {
            Some("needs a trained model; measures its rings tile by tile (measure_intensities_of)")
        }
        // The power spectrum's window is at most 4096² pixels (`MAX_SPECTRUM_SIDE`):
        // up to ~670 MB, but not more on a larger slide - the test slides are smaller.
        PipelineCommand::MeasureImageQuality(_) => Some(
            "the spectrum's window is capped at 4096² pixels (bounded, larger than the test slides)",
        ),
        _ => None,
    }
}

/// Each whole-image command, set up to work on the synthetic slide's
/// nuclei, cells, spots and channels 0 and 1. `None`: not set up yet.
fn configured(command: &PipelineCommand) -> Option<PipelineCommand> {
    use PipelineCommand as C;
    let id = "ram_01".to_string();
    Some(match command {
        C::ExcludeImage(_) => C::ExcludeImage(ExcludeImageSettings::default()),
        C::MeasureImageAreaOccupied(_) => {
            C::MeasureImageAreaOccupied(MeasureImageAreaOccupiedSettings {
                id,
                object_class: CELL,
            })
        }
        C::MeasureImageColocalization(_) => {
            C::MeasureImageColocalization(MeasureImageColocalizationSettings {
                id,
                channel_a: ImageChannelIdx(0),
                channel_b: ImageChannelIdx(1),
                ..Default::default()
            })
        }
        C::MeasureImageIntensity(_) => C::MeasureImageIntensity(MeasureImageIntensitySettings {
            id,
            channel: ImageChannelIdx(0),
        }),
        C::MeasureImageQuality(_) => C::MeasureImageQuality(MeasureImageQualitySettings {
            id,
            channel: ImageChannelIdx(0),
        }),
        C::MeasureImageTexture(_) => C::MeasureImageTexture(MeasureImageTextureSettings {
            id,
            channel: ImageChannelIdx(0),
            ..Default::default()
        }),
        C::MeasureObjectColocalization(_) => {
            C::MeasureObjectColocalization(MeasureObjectColocalizationSettings {
                id,
                objects: CELL,
                channel_a: ImageChannelIdx(0),
                channel_b: ImageChannelIdx(1),
                ..Default::default()
            })
        }
        C::MeasureObjectDistanceToParent(_) => {
            C::MeasureObjectDistanceToParent(MeasureObjectDistanceToParentSettings {
                id,
                children: SPOT,
                parent_class: CELL,
            })
        }
        // "Expand until adjacent" grows a label raster of the whole image -
        // up to 128 megapixels, refused above (TODO): "Within a distance".
        C::MeasureObjectNeighbors(_) => C::MeasureObjectNeighbors(MeasureObjectNeighborsSettings {
            id,
            objects: CELL,
            neighbors: CELL,
            method: MeasureObjectNeighborsNeighborMethodSettings::WithinDistance,
            distance: 5,
        }),
        C::MeasureObjectShape(_) => {
            C::MeasureObjectShape(MeasureObjectShapeSettings { id, objects: CELL })
        }
        C::MeasureObjectTexture(_) => C::MeasureObjectTexture(MeasureObjectTextureSettings {
            id,
            objects: CELL,
            channel: ImageChannelIdx(0),
            ..Default::default()
        }),
        C::ClassifyObjects(_) => C::ClassifyObjects(ClassifyObjectsSettings {
            input_classes: vec![SPOT],
            output_class: OUTPUT,
            overlapping_with: CELL,
            ..Default::default()
        }),
        C::ColocalizeObjects(_) => C::ColocalizeObjects(ColocalizeObjectsSettings {
            classes_to_coloc: vec![CELL, SPOT],
            class_for_overlapping_areas: OUTPUT,
            ..Default::default()
        }),
        C::LoadAnnotatedObjects(_) => C::LoadAnnotatedObjects(LoadAnnotatedObjectsSettings {
            output_class: OUTPUT,
            ..Default::default()
        }),
        C::ObjectMath(_) => C::ObjectMath(ObjectMathSettings {
            input_class: CELL,
            other_class: NUCLEUS,
            output_class: OUTPUT,
            ..Default::default()
        }),
        C::Propagation(_) => C::Propagation(PropagationSettings {
            seeds: NUCLEUS,
            growth_channel: ImageChannelIdx(0),
            mask: CELL,
            regularization: 0.05,
            output_class: OUTPUT,
            exclude_at_edges: false,
        }),
        C::RelateObjects(_) => C::RelateObjects(RelateObjectsSettings {
            child_classes: vec![SPOT],
            parent_classes: vec![CELL],
            ..Default::default()
        }),
        C::TransformObjects(_) => C::TransformObjects(TransformObjectsSettings {
            input_class: CELL,
            output_class: OUTPUT,
            ..Default::default()
        }),
        C::Voronoi(_) => C::Voronoi(VoronoiSettings {
            centers: NUCLEUS,
            output_class: OUTPUT,
            unit: SizeUnits::Pixels,
            max_radius: 40.0,
            ..Default::default()
        }),
        _ => return None,
    })
}

/// A synthetic slide of `side`² pixels: channels 0 and 1 in tiles (cells
/// brighter than the background, with a ramp so values differ), capped
/// so the tiles spill; nuclei, cells and spots.
fn slide(side: usize) -> GlobalPipelineCache {
    let mut cache = GlobalPipelineCache::default();
    cache.image_cache.set_capacity_bytes(CACHE_BYTES);
    cache.image_meta.nr_of_bits = 16;
    cache.image_meta.full_image_width = ImageSize {
        width: side,
        height: side,
    };
    let value = |channel: usize, x: usize, y: usize| {
        let (cx, cy) = (x % SPACING, y % SPACING);
        let in_cell = (8..56).contains(&cx) && (8..56).contains(&cy);
        let base = if in_cell { 0.4 } else { 0.05 };
        base + ((x * 7 + y * 13 + channel * 5) % 97) as f32 / 1000.0
    };
    for ty in (0..side).step_by(TILE) {
        for tx in (0..side).step_by(TILE) {
            for channel in 0..2 {
                let data: Vec<f32> = (0..TILE * TILE)
                    .map(|i| value(channel, tx + i % TILE, ty + i / TILE))
                    .collect();
                let image = ImageContainer::F32Gray(ManagedImage {
                    data: Image::new(
                        ImageSize {
                            width: TILE,
                            height: TILE,
                        },
                        data,
                    )
                    .unwrap(),
                    tile_offset: Point2d { x: tx, y: ty },
                    plane: None,
                    dimensionality: ImageDimensionality::Planar,
                });
                let tile = ImageTile {
                    offset_x: tx,
                    offset_y: ty,
                    width: TILE,
                    height: TILE,
                };
                cache.add_to_channel_cache(Arc::new(image), channel as i32, tile);
            }
        }
    }
    let mut id = 0u128;
    let mut add = |cache: &mut GlobalPipelineCache, bbox: [u32; 4], class: ObjectClass| {
        id += 1;
        let (w, h) = (
            (bbox[2] - bbox[0] + 1) as usize,
            (bbox[3] - bbox[1] + 1) as usize,
        );
        let mut object = Object::new(ObjectInit {
            id: ObjectId(id),
            bbox,
            mask_data: BitVec::<u64, Lsb0>::repeat(true, w * h),
            area: w * h,
            ..Default::default()
        });
        object.intensities = object.measure_intensities(cache);
        object.add_object_class(class, Scope::Global).unwrap();
        cache.object_cache.insert(object.id.clone(), object);
    };
    for y in (0..side).step_by(SPACING) {
        for x in (0..side).step_by(SPACING) {
            let (x, y) = (x as u32, y as u32);
            add(&mut cache, [x + 8, y + 8, x + 55, y + 55], CELL);
            add(&mut cache, [x + 28, y + 28, x + 35, y + 35], NUCLEUS);
            add(&mut cache, [x + 12, y + 12, x + 14, y + 14], SPOT);
        }
    }
    cache
}

/// The most bytes `command` has in use more than before it ran, on (a copy
/// of) `slide`, `side`² pixels.
fn peak_bytes(command: &PipelineCommand, slide: &GlobalPipelineCache, side: usize) -> usize {
    let algorithm = into_algorithm(command.clone()).unwrap();
    let mut cache = slide.clone();
    let mut meta = PipelineImageMeta::default();
    meta.full_image_width = ImageSize {
        width: side,
        height: side,
    };
    meta.nr_of_bits = 16;
    let mut ctx = PipelineContext::without_image(std::path::PathBuf::new(), meta).unwrap();
    // One thread: work in parallel holds a tile or a window per thread -
    // bounded, but more of them on the larger slide.
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .unwrap();
    let before = counting_allocator::in_use();
    counting_allocator::reset_peak();
    if let Err(e) = pool.install(|| algorithm.execute(&mut ctx, &mut cache)) {
        panic!("{}: {e}", algorithm.name());
    }
    counting_allocator::peak() - before
}

/// Measures every whole-image command (see the module docs) - in a process
/// of its own: the allocator counts the whole process, other tests must not
/// run alongside. Run by `whole_image_commands_need_no_ram_per_pixel`.
#[test]
#[ignore]
fn measure_whole_image_commands() {
    let (small_slide, large_slide) = (slide(SMALL), slide(LARGE));
    let mut failed = Vec::new();
    for command in (0..).map_while(default_command) {
        let Ok(algorithm) = into_algorithm(command.clone()) else {
            continue;
        };
        if algorithm.execution_scope() != ExecutionScope::WholeImage {
            continue;
        }
        let name = algorithm.name();
        if let Some(why) = not_measured(&command) {
            println!("{name}: not measured - {why}");
            continue;
        }
        let Some(command) = configured(&command) else {
            failed.push(format!(
                "{name}: a new whole-image command - add it to `configured`"
            ));
            continue;
        };
        let start = std::time::Instant::now();
        let small = peak_bytes(&command, &small_slide, SMALL);
        let large = peak_bytes(&command, &large_slide, LARGE);
        let per_pixel = large.saturating_sub(small) as f64 / (LARGE * LARGE - SMALL * SMALL) as f64;
        println!(
            "{name}: {:.1} MB / {:.1} MB, {per_pixel:.3} bytes per pixel ({:.1} s)",
            small as f64 / 1e6,
            large as f64 / 1e6,
            start.elapsed().as_secs_f64()
        );
        if per_pixel > MAX_BYTES_PER_PIXEL {
            failed.push(format!(
                "{name}: {per_pixel:.2} bytes more per pixel of a larger slide (at most \
                 {MAX_BYTES_PER_PIXEL})"
            ));
        }
    }
    assert!(failed.is_empty(), "{}", failed.join("\n"));
}

/// No whole-image command needs RAM per pixel of the slide (see the module
/// docs).
#[test]
fn whole_image_commands_need_no_ram_per_pixel() {
    let output = std::process::Command::new(std::env::current_exe().unwrap())
        .args([
            "--exact",
            "tests::whole_image_ram::measure_whole_image_commands",
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
