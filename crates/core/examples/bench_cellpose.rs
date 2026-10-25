//! # bench_cellpose
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-10-19
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.

// Benchmark: a Cellpose pipeline (Cellpose -> extract objects) over
// synthetic 16-bit images with cell-like spots, run through the real job
// generator + JobExecutor. Prints the wall time and the process's peak RAM.
// See BENCHMARK.md ("AI models: Cellpose").
//
// Usage: cargo run --release -p evanalyzer_core --features libtorch --example bench_cellpose -- \
//     <workdir> --model <model.pt> [--images N] [--size PX] [--threads N] [--channels N] [--repeat N]
//
// `--repeat N` runs the job N times in one process: the first run includes
// setting up the engine, loading the model and libtorch's warm-up; the
// others are the steady state (their median is printed).
//
// `--features libtorch` runs the model through the plugin (build it with
// `cargo build-ai-plugin --release`, point at it with EVANALYZER_AI_PLUGIN),
// `--features libtorch-in-process` with libtorch linked in.

use evanalyzer_cfg::core_types::{ImageAddress, SegmentationClass};
use evanalyzer_cfg::settings::images_settings::ImageEntry;
use evanalyzer_cfg::settings::pipeline_command::PipelineCommand;
use evanalyzer_cfg::settings::pipeline_command_settings::{
    CellposeSettings, ExtractObjectsSettings,
};
use evanalyzer_cfg::settings::pipeline_settings::{PipelineSettings, PipelineStepSettings};
use evanalyzer_cfg::settings::project_settings::ProjectSettings;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::time::Instant;

fn arg(name: &str) -> Option<String> {
    let args: Vec<String> = std::env::args().collect();
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1))
        .cloned()
}

fn num(name: &str, default: usize) -> usize {
    arg(name).and_then(|v| v.parse().ok()).unwrap_or(default)
}

fn make_image(path: &PathBuf, size: u32, seed: u64) {
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
    // About one cell (radius 8-15 px) per 60 x 60 px.
    for _ in 0..(size / 60).pow(2) {
        let (cx, cy) = (rnd() % size, rnd() % size);
        let r = 8 + (rnd() % 8);
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

/// `VmHWM` (peak resident memory) of this process, in MB.
fn peak_rss_mb() -> u64 {
    std::fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|s| {
            s.lines()
                .find(|l| l.starts_with("VmHWM:"))
                .and_then(|l| l.split_whitespace().nth(1)?.parse::<u64>().ok())
        })
        .map_or(0, |kb| kb / 1024)
}

fn main() {
    env_logger::init();
    let workdir = PathBuf::from(std::env::args().nth(1).expect("workdir"));
    let model = PathBuf::from(arg("--model").expect("--model <model.pt>"));
    let images = num("--images", 4);
    let size = num("--size", 1024) as u32;
    let threads = num("--threads", 0);
    let channels = num("--channels", 2) as i32;
    let images_dir = workdir.join(format!("cells_{size}"));
    std::fs::create_dir_all(&images_dir).unwrap();

    let mut project = ProjectSettings::default();
    project.images.root = Some(images_dir.clone());
    for i in 0..images {
        let name = PathBuf::from(format!("img_{i:03}.tif"));
        let path = images_dir.join(&name);
        if !path.exists() {
            make_image(&path, size, i as u64 + 1);
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
            name: "Cells".into(),
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
        name: "cellpose".into(),
        description: None,
        image_source: ImageAddress::Channel(0),
        enabled: true,
        steps: vec![
            step(PipelineCommand::Cellpose(CellposeSettings {
                model_path: model,
                segmentation_class: SegmentationClass(1),
                input_channels: channels,
                ..Default::default()
            })),
            step(PipelineCommand::ExtractObjects(ExtractObjectsSettings {
                max_objects_before_fail: 1_000_000,
                ..Default::default()
            })),
        ],
    });

    let parallelism = if threads == 0 {
        std::thread::available_parallelism().unwrap().get()
    } else {
        threads
    };
    let repeat = num("--repeat", 1).max(1);
    let mut times = Vec::new();
    for run in 0..repeat {
        // A results folder of its own per run (the name has a 1 s timestamp).
        let name = format!(
            "bench_cellpose_{}_{run}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        );
        let job = evanalyzer_core::generate_analyze_job_from_project_settings(
            project.clone(),
            workdir.clone(),
            Some(name),
        )
        .unwrap();
        let (tx, rx) = std::sync::mpsc::channel();
        let start = Instant::now();
        let result = job.run(parallelism, tx, Arc::new(AtomicBool::new(false)));
        let elapsed = start.elapsed().as_secs_f64();
        drop(rx);
        drop(job);
        if let Err(e) = result {
            panic!("run {run} failed: {e:?}");
        }
        println!(
            "run {run}: {elapsed:.2}s ({:.2} s/img){}",
            elapsed / images as f64,
            if run == 0 {
                "  - with start-up and warm-up"
            } else {
                ""
            }
        );
        times.push(elapsed);
    }
    let mut steady = times[1.min(times.len() - 1)..].to_vec();
    steady.sort_by(f64::total_cmp);
    println!(
        "threads={parallelism} images={images} size={size} first={:.2}s steady median={:.2}s  peak RSS {} MB",
        times[0],
        steady[steady.len() / 2],
        peak_rss_mb()
    );
}
