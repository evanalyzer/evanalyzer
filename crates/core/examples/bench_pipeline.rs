// TEMPORARY benchmark for the performance review: synthetic 16-bit images
// with cell-like spots, pipeline Gaussian blur -> threshold -> extract
// objects, run through the real job generator + JobExecutor.
//
// Usage: cargo run --release -p evanalyzer_core --example bench_pipeline -- \
//     <workdir> [--images N] [--size PX] [--threads N] [--spots N] [--classify N]
//
// `--classify N` adds N Classify Objects steps (class filtering by size).
// `--source 1` starts from the scratchpad, `--source 2` from no image, both
// with a Load Image Channel step first (0: the channel itself, the default).
// `--shared 1`: pipeline 1 stores channel 0 in shared memory slot 1, the
// detection pipeline starts from that slot (an image passed between pipelines).
// `--module 1` adds a module instance that turns the larger objects
// into local (internal) objects - they don't reach the results.

use evanalyzer_cfg::core_types::{ImageAddress, SegmentationClass};
use evanalyzer_cfg::core_types::{ObjectClass, SizeUnits};
use evanalyzer_cfg::settings::images_settings::ImageEntry;
use evanalyzer_cfg::settings::pipeline_command::PipelineCommand;
use evanalyzer_cfg::settings::pipeline_command_settings::{
    ClassifyObjectsSettings, ExtractObjectsSettings, GaussianBlurSettings,
    ObjectClassifyObjectsClassifyMatchHandlingSettings, ThresholdEntrySettings, ThresholdSettings,
};
use evanalyzer_cfg::settings::pipeline_settings::{PipelineSettings, PipelineStepSettings};
use evanalyzer_cfg::settings::project_settings::ProjectSettings;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::time::Instant;

fn arg(name: &str, default: usize) -> usize {
    let args: Vec<String> = std::env::args().collect();
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1))
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn make_image(path: &PathBuf, size: u32, spots: u32, seed: u64) {
    let mut state = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
    let mut rnd = || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (state >> 33) as u32
    };
    let mut img = image::ImageBuffer::<image::Luma<u16>, Vec<u16>>::new(size, size);
    for p in img.pixels_mut() {
        p.0[0] = 200 + (rnd() % 100) as u16;
    }
    for _ in 0..spots {
        let (cx, cy) = (rnd() % size, rnd() % size);
        let r = 4 + (rnd() % 8);
        for y in cy.saturating_sub(r)..(cy + r).min(size) {
            for x in cx.saturating_sub(r)..(cx + r).min(size) {
                let (dx, dy) = (x as i64 - cx as i64, y as i64 - cy as i64);
                if dx * dx + dy * dy <= (r * r) as i64 {
                    img.get_pixel_mut(x, y).0[0] = 3000 + (rnd() % 500) as u16;
                }
            }
        }
    }
    img.save(path).unwrap();
}

fn main() {
    env_logger::init();
    let workdir = PathBuf::from(std::env::args().nth(1).expect("workdir"));
    let images = arg("--images", 8);
    let size = arg("--size", 4096) as u32;
    let threads = arg("--threads", 0);
    let spots = arg("--spots", 20000) as u32;
    let blurs = arg("--blurs", 1);
    let classify = arg("--classify", 0);
    let module = arg("--module", 0);
    let source = arg("--source", 0);
    let shared = arg("--shared", 0);
    let images_dir = workdir.join(format!("images_{size}_{spots}"));
    std::fs::create_dir_all(&images_dir).unwrap();

    let mut project = ProjectSettings::default();
    project.images.root = Some(images_dir.clone());
    for i in 0..images {
        let name = PathBuf::from(format!("img_{i:03}.tif"));
        let path = images_dir.join(&name);
        if !path.exists() {
            make_image(&path, size, spots, i as u64 + 1);
        }
        project.images.list.insert(
            name.clone(),
            ImageEntry {
                rel_path: name,
                ..Default::default()
            },
        );
    }
    project.classification.classes_mut().push(
        evanalyzer_cfg::settings::classification_settings::Class {
            id: 1,
            color: 0xFF0000,
            name: "Spots".into(),
            notes: String::new(),
        },
    );
    let step = |command| PipelineStepSettings {
        id: evanalyzer_cfg::core_types::StepId::UNASSIGNED,
        enabled: true,
        content: evanalyzer_cfg::settings::pipeline_settings::StepContent::Command(command),
    };
    project.pipelines.push(PipelineSettings {
        id: evanalyzer_cfg::core_types::PipelineId(1),
        name: "bench".into(),
        description: None,
        image_source: match source {
            _ if shared > 0 => {
                ImageAddress::Memory(evanalyzer_cfg::core_types::MemoryId::ProjectCache(1))
            }
            1 => ImageAddress::Scratchpad,
            2 => ImageAddress::None,
            _ => ImageAddress::Channel(0),
        },
        enabled: true,
        steps: (source > 0)
            .then(|| {
                step(PipelineCommand::LoadChannel(
                    evanalyzer_cfg::settings::pipeline_command_settings::LoadChannelSettings {
                        channel: evanalyzer_cfg::core_types::ImageChannelIdx(0),
                        projection: evanalyzer_cfg::settings::pipeline_command_settings::MathLoadChannelChannelZProjectionSettings::Auto,
                        z_index: 0,
                    },
                ))
            })
            .into_iter()
            .chain((0..blurs).map(|_| {
                step(PipelineCommand::GaussianBlur(GaussianBlurSettings {
                    kernel_size: 5,
                    sigma: 1.0,
                }))
            }))
            .chain([
                step(PipelineCommand::Threshold(ThresholdSettings {
                    thresholds: vec![ThresholdEntrySettings {
                        min_threshold: 1500.0,
                        segmentation_class: SegmentationClass(1),
                        ..Default::default()
                    }],
                })),
                step(PipelineCommand::ConnectedComponents(Default::default())),
                step(PipelineCommand::ExtractObjects(ExtractObjectsSettings {
                    max_objects_before_fail: 100000,
                    ..Default::default()
                })),
            ])
            .chain((0..classify).map(|i| {
                step(PipelineCommand::ClassifyObjects(ClassifyObjectsSettings {
                    input_classes: vec![ObjectClass::Global(1)],
                    match_handling:
                        ObjectClassifyObjectsClassifyMatchHandlingSettings::AddOutputClassIfMatch,
                    output_class: ObjectClass::Global(2 + i as u32),
                    size_unit: SizeUnits::Pixels,
                    min_area: 100.0,
                    ..Default::default()
                }))
            }))
            .chain((module > 0).then(|| PipelineStepSettings {
                id: evanalyzer_cfg::core_types::StepId::UNASSIGNED,
                enabled: true,
                content: evanalyzer_cfg::settings::pipeline_settings::StepContent::Module(
                    evanalyzer_cfg::settings::modules::ModuleInstance {
                        key: "INTERNAL".into(),
                        values: Default::default(),
                    },
                ),
            }))
            .collect(),
    });
    if module > 0 {
        project.modules.push(internal_objects_module());
    }
    if shared > 0 {
        // The storing pipeline runs first.
        project.pipelines.insert(
            0,
            PipelineSettings {
                id: evanalyzer_cfg::core_types::PipelineId(2),
                name: "store".into(),
                description: None,
                image_source: ImageAddress::Channel(0),
                enabled: true,
                steps: vec![step(PipelineCommand::ImageCache(
                    evanalyzer_cfg::settings::pipeline_command_settings::ImageCacheSettings {
                        mode: evanalyzer_cfg::settings::pipeline_command_settings::MathImageCacheImageCacheModeSettings::Store,
                        address: ImageAddress::Memory(evanalyzer_cfg::core_types::MemoryId::ProjectCache(1)),
                    },
                ))],
            },
        );
    }

    let job = evanalyzer_core::generate_analyze_job_from_project_settings(
        project,
        workdir.clone(),
        Some(format!("bench_t{threads}")),
    )
    .unwrap();
    let parallelism = if threads == 0 {
        std::thread::available_parallelism().unwrap().get()
    } else {
        threads
    };
    let (tx, rx) = std::sync::mpsc::channel();
    let start = Instant::now();
    let result = job.run(parallelism, tx, Arc::new(AtomicBool::new(false)));
    let elapsed = start.elapsed().as_secs_f64();
    drop(rx);
    drop(job);
    let db = std::fs::read_dir(workdir.join("results"))
        .unwrap()
        .flatten()
        .flat_map(|d| std::fs::read_dir(d.path()).unwrap().flatten())
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|e| e == "evadb"))
        .max_by_key(|p| std::fs::metadata(p).unwrap().modified().unwrap())
        .unwrap();
    let conn = duckdb::Connection::open(&db).unwrap();
    let objects: i64 = conn
        .query_row("SELECT count(*) FROM objects", [], |r| r.get(0))
        .unwrap();
    println!("objects in db: {objects}");
    let mpx = images as f64 * (size as f64 * size as f64) / 1e6;
    println!(
        "threads={parallelism} images={images} size={size} result={:?} time={elapsed:.2}s  {:.2} img/s  {:.1} MPx/s",
        result.err(),
        images as f64 / elapsed,
        mpx / elapsed
    );
}

/// A module that reclassifies the larger objects into a local class and
/// classifies those once more - internal objects only.
fn internal_objects_module() -> evanalyzer_cfg::settings::modules::ModuleDefinition {
    let classify = |id: u32, input: ObjectClass, handling, output: ObjectClass, min_area: f32| {
        PipelineStepSettings {
            id: evanalyzer_cfg::core_types::StepId(id),
            enabled: true,
            content: evanalyzer_cfg::settings::pipeline_settings::StepContent::Command(
                PipelineCommand::ClassifyObjects(ClassifyObjectsSettings {
                    input_classes: vec![input],
                    match_handling: handling,
                    output_class: output,
                    size_unit: SizeUnits::Pixels,
                    min_area,
                    ..Default::default()
                }),
            ),
        }
    };
    let local_class = |id: u32| evanalyzer_cfg::settings::classification_settings::Class {
        id,
        color: 0,
        name: format!("Local {id}"),
        notes: String::new(),
    };
    evanalyzer_cfg::settings::modules::ModuleDefinition {
        key: "INTERNAL".into(),
        meta: Default::default(),
        classes: vec![local_class(1), local_class(2)],
        steps: vec![
            classify(
                1,
                ObjectClass::Global(1),
                ObjectClassifyObjectsClassifyMatchHandlingSettings::ReclassifyIfMatch,
                ObjectClass::Local(1),
                100.0,
            ),
            classify(
                2,
                ObjectClass::Local(1),
                ObjectClassifyObjectsClassifyMatchHandlingSettings::AddOutputClassIfMatch,
                ObjectClass::Local(2),
                150.0,
            ),
        ],
        parameters: vec![],
        origin: None,
    }
}
