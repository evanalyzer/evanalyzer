// Writes a large time series results database for benchmarks (not part of
// the test suite) - through the real `DuckDbExporter`, the way an analysis
// writes one: per image, each time point's objects, then the image with its
// analysed time points and frame interval.
//
// A plate: wells A01..H12, `--fields` images per well ("B03_02.tif"), each
// `--time-points` time points `--frame-interval` seconds apart, with
// `--objects` tracked objects per image and time point (some tracks start
// late or end early). Three classes - Nuclei, Cells, Spots in cells; area
// and intensity change over time, differently per plate column.
//
// Usage: cargo run --release -p evanalyzer_core --example make_time_series -- \
//     <out.evadb> [--wells N] [--fields N] [--time-points N] [--objects N] [--frame-interval S]

use bitvec::prelude::*;
use evanalyzer_cfg::core_types::{ObjectClass, ObjectId, TrackId};
use evanalyzer_cfg::settings::meta_data::MetaData;
use evanalyzer_core::{
    DuckDbExporter, GlobalPipelineCache, ImagePlane, ImageSummary, Intensity, Object, ObjectInit,
    PipelineResultExporter, Track,
};
use indexmap::IndexMap;
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::time::Instant;

fn arg_value(name: &str, default: usize) -> usize {
    let args: Vec<String> = std::env::args().collect();
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1))
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

/// A repeatable pseudo-random number in [0, 1) from `seed`.
fn noise(seed: u64) -> f64 {
    let mut x = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    x ^= x >> 31;
    x = x.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x ^= x >> 29;
    (x % 1_000_000) as f64 / 1_000_000.0
}

const CLASSES: [(u32, &str, f64); 3] = [
    (1, "Nuclei", 120.0),
    (2, "Cells", 900.0),
    (3, "Spots in cells", 12.0),
];

fn main() {
    let out = PathBuf::from(std::env::args().nth(1).expect(
        "usage: make_time_series <out.evadb> [--wells N] [--fields N] [--time-points N] \
         [--objects N] [--frame-interval S]",
    ));
    let wells = arg_value("--wells", 48);
    let fields = arg_value("--fields", 2);
    let times = arg_value("--time-points", 48);
    let per_image = arg_value("--objects", 1500);
    let frame_interval = arg_value("--frame-interval", 300) as f64;
    let _ = std::fs::remove_file(&out);
    let _ = std::fs::remove_file(format!("{}.wal", out.display()));

    let class_names: HashMap<ObjectClass, (String, u32)> = CLASSES
        .iter()
        .map(|(id, name, _)| {
            (
                ObjectClass::Global(*id),
                (name.to_string(), 0x2A6BDF + id * 0x101010),
            )
        })
        .collect();
    let exporter =
        DuckDbExporter::new(&out, class_names, &MetaData::default()).expect("create exporter");

    let start = Instant::now();
    let mut next_id: u128 = 1;
    let mut objects_written: u64 = 0;
    let analysed_t: Vec<i32> = (0..times as i32).collect();
    for image in 0..wells * fields {
        let well = image / fields;
        let (row, col) = (well / 12, well % 12);
        let name = format!(
            "{}{:02}_{:02}.tif",
            (b'A' + row as u8) as char,
            col + 1,
            image % fields + 1
        );
        // Each column grows at its own rate - something to see per well.
        let growth = 0.1 + col as f64 * 0.05;
        for t in 0..times {
            let mut cache = GlobalPipelineCache::default();
            cache.image_rel_path = PathBuf::from(&name);
            cache.image_meta.nr_of_bits = 16;
            for k in 0..per_image {
                let track = (image * per_image + k) as u64 + 1;
                // 10 % of the tracks start late, 10 % end early.
                let first = if noise(track) < 0.1 {
                    (noise(track + 7) * times as f64 / 2.0) as usize
                } else {
                    0
                };
                let last = if noise(track + 3) < 0.1 {
                    times / 2 + (noise(track + 11) * times as f64 / 2.0) as usize
                } else {
                    times
                };
                if t < first || t >= last {
                    continue;
                }
                let (class_id, _, base) = CLASSES[(noise(track + 5) * 10.0) as usize % 5 / 2];
                let mut object_class: HashSet<ObjectClass> =
                    HashSet::from([ObjectClass::Global(class_id)]);
                if class_id == 3 && noise(track + 13) < 0.2 {
                    object_class.insert(ObjectClass::Global(2));
                }
                let progress = t as f64 / times.max(2) as f64;
                let area =
                    (base * (1.0 + growth * progress) * (0.7 + 0.6 * noise(track * 31 + t as u64)))
                        .max(1.0) as usize;
                let (cx, cy) = (
                    (noise(track + 17) * 3800.0 + t as f64 * 2.0 * (noise(track + 19) - 0.5))
                        as u64
                        + 100,
                    (noise(track + 23) * 3800.0 + t as f64 * 2.0 * (noise(track + 29) - 0.5))
                        as u64
                        + 100,
                );
                let brightness = 1.0 - 0.4 * progress * (class_id as f64 / 3.0);
                let intensities: IndexMap<i32, Intensity> = (0..2)
                    .map(|channel| {
                        let mean =
                            (300.0 + 900.0 * noise(track + channel as u64 * 101)) * brightness;
                        (
                            channel,
                            Intensity {
                                sum_intensity: mean * area as f64,
                                min_intensity: mean * 0.5,
                                max_intensity: mean * 1.8,
                                avg_intensity: mean,
                                pixel_values: Vec::new(),
                            },
                        )
                    })
                    .collect();
                let side = (area as f64).sqrt().ceil() as u32;
                let id = ObjectId(next_id);
                next_id += 1;
                cache.object_cache.insert(
                    id.clone(),
                    Object::new(ObjectInit {
                        id,
                        object_class,
                        intensities,
                        track: Track {
                            id: TrackId(track),
                            ..Default::default()
                        },
                        bbox: [cx as u32, cy as u32, cx as u32 + side, cy as u32 + side],
                        mask_data: BitVec::<u64, Lsb0>::repeat(true, (side * side) as usize),
                        area,
                        sum_x: cx * area as u64,
                        sum_y: cy * area as u64,
                        plane: ImagePlane {
                            z: 0,
                            c: 0,
                            t: t as i32,
                        },
                        ..Default::default()
                    }),
                );
                objects_written += 1;
            }
            exporter.export(&cache).expect("export");
        }
        exporter
            .finalize_image(
                Path::new(&name),
                &ImageSummary {
                    width: 4096,
                    height: 4096,
                    nr_c_stacks: 2,
                    nr_z_stacks: 1,
                    nr_t_stacks: times as u32,
                    t_stacks_analyzed: &analysed_t,
                    z_stacks_analyzed: &[0],
                    frame_interval_s: Some(frame_interval),
                },
                None,
            )
            .expect("finalize");
        if image % 8 == 7 {
            eprintln!("{} of {} images", image + 1, wells * fields);
        }
    }
    drop(exporter);
    evanalyzer_core::open_results_database(&out)
        .expect("reopen")
        .execute_batch("CHECKPOINT")
        .expect("checkpoint");
    let size_mb = std::fs::metadata(&out).map_or(0.0, |m| m.len() as f64 / 1e6);
    println!(
        "wrote {objects_written} objects: {} images ({wells} wells x {fields}) x {times} time points, \
         {per_image} tracks each, in {:.1} s - {size_mb:.0} MB",
        wells * fields,
        start.elapsed().as_secs_f64()
    );
}
