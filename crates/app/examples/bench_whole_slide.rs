//! # bench_whole_slide
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-10-28
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.

// Benchmark for the whole-slide merge: one tiled slide through the same path
// as `evanalyzer cli analyze` (folder scan, then `start_analysis`), pipeline
// max projection -> Gaussian blur -> Otsu -> objects -> Voronoi cells (a
// whole-image step after the tile merge). Reports wall time, peak RSS and the
// peak size of the image cache's spill files (named ones in the temp folder
// and unnamed ones the process holds open).
//
// The spill files go to the system temp folder - set TMPDIR to an empty
// folder so its size is the image cache's alone.
//
// Usage: TMPDIR=<empty dir> cargo run --release -p evanalyzer_app --example bench_whole_slide -- \
//     <images dir> <work dir> [--threads N] [--channel C] [--coloc 1]
//
// `--coloc 1` adds Measure Object Colocalization of the Voronoi cells
// (channels 0 and 1, threshold 15 %, Costes "Faster"). `--voronoi-radius R`
// (default 50 px; 0: no limit - the cells cover the whole slide).

use evanalyzer_app::analysis::AnalysisRequest;
use evanalyzer_app::backends::Backend;
use evanalyzer_app::backends::local::LocalBackend;
use evanalyzer_app::project::{ImageListExt, ProjectExt, ProjectWithRuntime};
use evanalyzer_cfg::core_types::{
    ImageAddress, ImageChannelIdx, ObjectClass, PipelineId, SegmentationClass, SizeUnits, StepId,
};
use evanalyzer_cfg::settings::classification_settings::Class;
use evanalyzer_cfg::settings::pipeline_command::PipelineCommand;
use evanalyzer_cfg::settings::pipeline_command_settings::{
    ExtractObjectsSettings, GaussianBlurSettings, LoadChannelSettings,
    MathLoadChannelChannelZProjectionSettings, MeasureColocalizationCostesMethodSettings,
    MeasureObjectColocalizationSettings, SegmentationThresholdThresholdMethodSettings,
    ThresholdEntrySettings, ThresholdSettings, VoronoiSettings,
};
use evanalyzer_cfg::settings::pipeline_settings::{
    PipelineSettings, PipelineStepSettings, StepContent,
};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::{Duration, Instant};

fn arg(name: &str, default: usize) -> usize {
    let args: Vec<String> = std::env::args().collect();
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1))
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn proc_status_mb(key: &str) -> u64 {
    std::fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|s| {
            s.lines()
                .find(|l| l.starts_with(key))
                .and_then(|l| l.split_whitespace().nth(1)?.parse::<u64>().ok())
        })
        .unwrap_or(0)
        / 1024
}

fn dir_bytes(dir: &Path) -> u64 {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return 0;
    };
    entries
        .flatten()
        .map(|e| match e.file_type() {
            Ok(t) if t.is_dir() => dir_bytes(&e.path()),
            _ => e.metadata().map(|m| m.len()).unwrap_or(0),
        })
        .sum()
}

/// Unnamed (already deleted) files under `dir` this process holds open -
/// spill files the OS deletes by itself, invisible to `dir_bytes`. Linux.
fn open_deleted_bytes(dir: &Path) -> u64 {
    let Ok(entries) = std::fs::read_dir("/proc/self/fd") else {
        return 0;
    };
    entries
        .flatten()
        .filter(|e| {
            std::fs::read_link(e.path()).is_ok_and(|target| {
                target.starts_with(dir) && target.to_string_lossy().ends_with(" (deleted)")
            })
        })
        .map(|e| std::fs::metadata(e.path()).map(|m| m.len()).unwrap_or(0))
        .sum()
}

fn main() {
    let images_dir = PathBuf::from(std::env::args().nth(1).expect("images dir"));
    let workdir = PathBuf::from(std::env::args().nth(2).expect("work dir"));
    let threads = arg("--threads", 0);
    let channel = arg("--channel", 0) as u32;
    let coloc = arg("--coloc", 0) == 1;
    let voronoi_radius = arg("--voronoi-radius", 50) as f32;
    std::fs::create_dir_all(&workdir).unwrap();
    let backend = LocalBackend::default();

    let mut project = ProjectWithRuntime::default();
    project.images.root = Some(images_dir);
    project.scan_image_folder_and_add(&backend);
    for entry in project.images.list.values() {
        for (idx, series) in &entry.series {
            println!(
                "{:?} series {idx}: {} x {}, {} channel(s)",
                entry.rel_path,
                series.image_width,
                series.image_height,
                series.channels.len()
            );
        }
    }
    let classes = project.classification.classes_mut();
    for (id, name) in [(1, "Nuclei"), (2, "Cells")] {
        classes.push(Class {
            id,
            color: 0xFF0000,
            name: name.into(),
            notes: String::new(),
        });
    }
    let step = |command| PipelineStepSettings {
        id: StepId::UNASSIGNED,
        enabled: true,
        content: StepContent::Command(command),
    };
    project.pipelines.push(PipelineSettings {
        id: PipelineId(1),
        name: "bench".into(),
        description: None,
        image_source: ImageAddress::None,
        enabled: true,
        steps: vec![
            step(PipelineCommand::LoadChannel(LoadChannelSettings {
                channel: ImageChannelIdx(channel),
                projection: MathLoadChannelChannelZProjectionSettings::MaxIntensity,
                z_index: 0,
            })),
            step(PipelineCommand::GaussianBlur(GaussianBlurSettings {
                kernel_size: 5,
                sigma: 1.0,
            })),
            step(PipelineCommand::Threshold(ThresholdSettings {
                thresholds: vec![ThresholdEntrySettings {
                    method: SegmentationThresholdThresholdMethodSettings::Otsu {
                        classes: Default::default(),
                    },
                    max_threshold: 65535.0,
                    segmentation_class: SegmentationClass(1),
                    ..Default::default()
                }],
            })),
            step(PipelineCommand::ConnectedComponents(Default::default())),
            step(PipelineCommand::ExtractObjects(ExtractObjectsSettings {
                max_objects_before_fail: 10_000_000,
                ..Default::default()
            })),
            step(PipelineCommand::Voronoi(VoronoiSettings {
                centers: ObjectClass::Global(1),
                mask: ObjectClass::Unset,
                output_class: ObjectClass::Global(2),
                unit: SizeUnits::Pixels,
                max_radius: voronoi_radius,
                ..Default::default()
            })),
        ]
        .into_iter()
        .chain(coloc.then(|| {
            step(PipelineCommand::MeasureObjectColocalization(
                MeasureObjectColocalizationSettings {
                    id: "object_coloc_01".into(),
                    objects: ObjectClass::Global(2),
                    channel_a: ImageChannelIdx(0),
                    channel_b: ImageChannelIdx(1),
                    threshold_percent: 15.0,
                    costes: MeasureColocalizationCostesMethodSettings::Faster,
                },
            ))
        }))
        .collect(),
    });

    // The spill files' peak, sampled while the run goes on.
    let tmp = std::env::temp_dir();
    let spill_peak = Arc::new(AtomicU64::new(0));
    // DuckDB's own memory (`duckdb_memory()`) of the results file, sampled
    // too - once the run created it.
    let duckdb_peak = Arc::new(AtomicU64::new(0));
    let results_dir: Arc<std::sync::Mutex<Option<PathBuf>>> = Default::default();
    let stop = Arc::new(AtomicBool::new(false));
    let sampler = {
        let (spill_peak, stop) = (spill_peak.clone(), stop.clone());
        let (duckdb_peak, results_dir) = (duckdb_peak.clone(), results_dir.clone());
        std::thread::spawn(move || {
            // `BENCH_TRACE=1`: RSS and spill size over time, on stderr.
            let trace = std::env::var("BENCH_TRACE").is_ok();
            let start = Instant::now();
            while !stop.load(Ordering::Relaxed) {
                let spill = dir_bytes(&tmp) + open_deleted_bytes(&tmp);
                spill_peak.fetch_max(spill, Ordering::Relaxed);
                let db = results_dir.lock().unwrap().as_ref().and_then(|dir| {
                    std::fs::read_dir(dir)
                        .ok()?
                        .flatten()
                        .map(|e| e.path())
                        .find(|p| p.extension().is_some_and(|e| e == "evadb"))
                });
                if let Some(db) = db
                    && let Ok(conn) = evanalyzer_core::open_results_database(&db)
                    && let Ok(bytes) = conn.query_row(
                        "SELECT sum(memory_usage_bytes)::BIGINT FROM duckdb_memory()",
                        [],
                        |r| r.get::<_, i64>(0),
                    )
                {
                    duckdb_peak.fetch_max(bytes as u64, Ordering::Relaxed);
                }
                if trace {
                    eprintln!(
                        "trace t={:.1}s rss={} MB spill={} MB",
                        start.elapsed().as_secs_f64(),
                        proc_status_mb("VmRSS:"),
                        spill / 1_000_000
                    );
                }
                std::thread::sleep(Duration::from_millis(500));
            }
        })
    };

    let start = Instant::now();
    let job = backend
        .start_analysis(AnalysisRequest {
            settings: project.settings.clone(),
            project_path: workdir.clone(),
            job_name: Some("bench_whole_slide".into()),
            threads: (threads > 0).then_some(threads),
            base_results: None,
        })
        .unwrap();
    println!("threads: {}", job.parallelism());
    let output = job.output_path().clone();
    *results_dir.lock().unwrap() = Some(output.clone());
    for _ in job.events() {}
    let result = job.wait();
    let elapsed = start.elapsed().as_secs_f64();
    stop.store(true, Ordering::Relaxed);
    sampler.join().unwrap();

    let db = std::fs::read_dir(&output)
        .unwrap()
        .flatten()
        .map(|e| e.path())
        .find(|p| p.extension().is_some_and(|e| e == "evadb"))
        .expect("a results database");
    // The results file as the run left it (the process still running, as
    // the app after a run).
    let file_mb = |p: &Path| std::fs::metadata(p).map(|m| m.len()).unwrap_or(0) / 1_000_000;
    let mut wal = db.clone().into_os_string();
    wal.push(".wal");
    println!(
        "results: file={} MB wal={} MB duckdb_peak={} MB",
        file_mb(&db),
        file_mb(Path::new(&wal)),
        duckdb_peak.load(Ordering::Relaxed) / 1_000_000
    );
    let conn = duckdb::Connection::open(&db).unwrap();
    let (objects, area): (i64, i64) = conn
        .query_row(
            "SELECT count(*), coalesce(sum(area_px), 0)::BIGINT FROM objects",
            [],
            |r| Ok((r.get(0)?, r.get(1)?)),
        )
        .unwrap();
    // To compare runs: identical inputs must give identical values.
    let (values, value_sum): (i64, f64) = conn
        .query_row(
            "SELECT count(*), coalesce(sum(value) FILTER (WHERE isfinite(value)), 0) FROM measurements",
            [],
            |r| Ok((r.get(0)?, r.get(1)?)),
        )
        .unwrap_or((0, 0.0));
    // Exact and independent of the row order (the sum above isn't).
    let value_hash: u64 = conn
        .query_row(
            "SELECT coalesce(bit_xor(hash(key, value::VARCHAR)), 0) FROM measurements",
            [],
            |r| r.get(0),
        )
        .unwrap_or(0);
    println!("measurements={values} sum={value_sum:.9} hash={value_hash:016x}");
    println!(
        "result={:?} time={elapsed:.1}s peak_rss={} MB spill_peak={} MB objects={objects} sum_area_px={area}",
        result.err(),
        proc_status_mb("VmHWM:"),
        spill_peak.load(Ordering::Relaxed) / 1_000_000,
    );
}
