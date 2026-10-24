// Benchmark (not part of the test suite) for writing results: drives the
// real `DuckDbExporter` with synthetic objects the way an analysis does -
// one `export()` + `finalize_image()` per image - and reports wall time,
// objects per second, peak RSS (`/proc/self/status`'s `VmHWM`, Linux-only)
// and the resulting file size. Run it against two checkouts with the same
// arguments to compare a change to the write path.
//
// Objects are mostly single-class, every 20th in two classes, with six
// intensity channels (values varying per object), and every 4th colocalizes with 1-2 objects of another
// class - roughly the shape of the real 5.6M-object reference file (6
// channels, 1.4M objects with colocalization).
//
// Usage: cargo run --release -p evanalyzer_core --example bench_export -- \
//     <out.evadb> [--images N] [--objects-per-image N] [--object-size N]
//
// `--object-size N` (default 4: a solid 4 × 4 square): a round object with
// a ragged edge in an N × N box, like a real cell or nucleus - what object
// masks cost depends on their shape. `BENCH_MASKS=0` doesn't store masks.
// `--spread 0`: objects without intensity distributions (std, median, ...;
// default 1: with, as a pipeline's objects have them).

use bitvec::prelude::*;
use evanalyzer_cfg::core_types::{ObjectClass, ObjectId};
use evanalyzer_cfg::settings::meta_data::MetaData;
use evanalyzer_core::PipelineResultExporter;
use evanalyzer_core::{DuckDbExporter, GlobalPipelineCache, Intensity, Object, ObjectInit};
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

fn peak_rss_mb() -> f64 {
    std::fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|status| {
            status
                .lines()
                .find_map(|line| line.strip_prefix("VmHWM:"))
                .and_then(|rest| rest.split_whitespace().next())
                .and_then(|kb| kb.parse::<f64>().ok())
        })
        .map_or(0.0, |kb| kb / 1024.0)
}

/// A round mask with a ragged edge in a `size` × `size` box (`size` <= 4:
/// solid), and its area.
fn mask(id: u128, size: u32) -> (BitVec<u64, Lsb0>, usize) {
    if size <= 4 {
        return (
            BitVec::repeat(true, (size * size) as usize),
            (size * size) as usize,
        );
    }
    let r = size as f32 / 2.0;
    // Every object its own outline, like real ones (a handful of shapes
    // would let DuckDB store each only once).
    let h = (id as u64)
        .wrapping_mul(0x9E37_79B9_7F4A_7C15)
        .rotate_left(29)
        .wrapping_mul(0xBF58_476D_1CE4_E5B9);
    let bumps = 3 + (h % 5) as u32;
    let phase = ((h >> 8) % 628) as f32 / 100.0;
    let depth = 0.06 + ((h >> 20) % 10) as f32 / 100.0;
    let mut mask = BitVec::<u64, Lsb0>::repeat(false, (size * size) as usize);
    let mut area = 0;
    for y in 0..size {
        for x in 0..size {
            let (dx, dy) = (x as f32 + 0.5 - r, y as f32 + 0.5 - r);
            let angle = dy.atan2(dx);
            let ragged = r * (0.85 + depth * (angle * bumps as f32 + phase).sin());
            if (dx * dx + dy * dy).sqrt() <= ragged {
                mask.set((y * size + x) as usize, true);
                area += 1;
            }
        }
    }
    (mask, area)
}

/// `spread`: each channel with its intensity distribution, as a pipeline's
/// objects have it (else none, as a stored annotation's).
fn object(id: u128, size: u32, spread: bool) -> Object {
    let class = (id % 21) as u32 + 1;
    let mut object_class: HashSet<ObjectClass> = HashSet::from([ObjectClass::Global(class)]);
    if id % 20 == 0 {
        object_class.insert(ObjectClass::Global(class % 21 + 1));
    }
    let intensities: IndexMap<i32, Intensity> = (0..6)
        .map(|channel| {
            // Varies per object and channel like real measurements -
            // identical values would compress unrealistically well.
            let v =
                ((id as u64).wrapping_mul(2654435761) % 10_000) as f64 / 10_000.0 + channel as f64;
            (
                channel,
                Intensity {
                    sum_intensity: 1.348531 * v,
                    min_intensity: (0.004471 * v),
                    max_intensity: (0.025757 * v),
                    avg_intensity: (0.008481 * v),
                    std_intensity: if spread { 0.002113 * v } else { f64::NAN },
                    median_intensity: if spread { 0.008302 * v } else { f64::NAN },
                    lower_quartile_intensity: if spread { 0.006917 * v } else { f64::NAN },
                    upper_quartile_intensity: if spread { 0.009875 * v } else { f64::NAN },
                    mad_intensity: if spread { 0.001414 * v } else { f64::NAN },
                },
            )
        })
        .collect();
    let mut colocalized_with: IndexMap<ObjectClass, Vec<ObjectId>> = IndexMap::new();
    if id % 4 == 0 && id > 2 {
        let partners = if id % 8 == 0 {
            vec![ObjectId(id - 1), ObjectId(id - 2)]
        } else {
            vec![ObjectId(id - 1)]
        };
        colocalized_with.insert(ObjectClass::Global(class % 21 + 2), partners);
    }
    let x = (id % 1000) as u32 * size;
    let y = (id / 1000 % 1000) as u32 * size;
    let (mask_data, area) = mask(id, size);
    Object::new(ObjectInit {
        id: ObjectId(id),
        object_class,
        intensities,
        colocalized_with,
        bbox: [x, y, x + size - 1, y + size - 1],
        mask_data,
        area,
        ..Default::default()
    })
}

fn main() {
    let out = PathBuf::from(
        std::env::args()
            .nth(1)
            .expect("usage: bench_export <out.evadb> [--images N] [--objects-per-image N]"),
    );
    let images = arg_value("--images", 1776);
    let per_image = arg_value("--objects-per-image", 2150);
    let size = arg_value("--object-size", 4) as u32;
    let spread = arg_value("--spread", 1) != 0;
    let _ = std::fs::remove_file(&out);
    let _ = std::fs::remove_file(format!("{}.wal", out.display()));

    let class_names: HashMap<ObjectClass, (String, u32)> = (1..=22)
        .map(|c| (ObjectClass::Global(c), (format!("Class {c}"), 0)))
        .collect();
    let exporter = DuckDbExporter::new(
        &out,
        class_names,
        &MetaData {
            name: "Experiment name".into(),
            short_description: "Short description".into(),
            description: "Long description".into(),
            authors: vec![],
            creation_time: chrono::DateTime::from_timestamp_nanos(1791318919124935845),
            version: None,
            category: "Test category".into(),
            tags: vec![],
            app_version: "v1.0.0".into(),
        },
    )
    .expect("create exporter")
    .with_masks(std::env::var("BENCH_MASKS").as_deref() != Ok("0"));

    // Objects are built outside the timed section; only writing is timed.
    let mut write_time = std::time::Duration::ZERO;
    let mut next_id: u128 = 1;
    for image in 0..images {
        let mut cache = GlobalPipelineCache::default();
        cache.image_rel_path = PathBuf::from(format!("A{}_{:02}.tif", image % 12 + 1, image));
        cache.image_meta.nr_of_bits = 16;
        for _ in 0..per_image {
            cache
                .object_cache
                .insert(ObjectId(next_id), object(next_id, size, spread));
            next_id += 1;
        }
        let start = Instant::now();
        exporter.export(&cache).expect("export");
        exporter
            .finalize_image(
                Path::new(&cache.image_rel_path),
                &evanalyzer_core::ImageSummary {
                    width: 4096,
                    height: 4096,
                    nr_c_stacks: 3,
                    nr_z_stacks: 1,
                    nr_t_stacks: 1,
                    t_stacks_analyzed: &[0],
                    z_stacks_analyzed: &[0],
                    frame_interval_s: None,
                    excluded: None,
                },
                None,
            )
            .expect("finalize");
        write_time += start.elapsed();
    }
    drop(exporter);

    // The exporter's connection stays open in the shared connection cache,
    // so recent writes may still sit in the WAL; checkpoint them into the
    // main file so both sizes compare the same thing.
    let size_mb = |path: &Path| std::fs::metadata(path).map_or(0.0, |m| m.len() as f64 / 1e6);
    let wal = PathBuf::from(format!("{}.wal", out.display()));
    let (file_before, wal_before) = (size_mb(&out), size_mb(&wal));
    let checkpoint = Instant::now();
    evanalyzer_core::open_results_database(&out)
        .expect("reopen")
        .execute_batch("CHECKPOINT")
        .expect("checkpoint");
    let checkpoint_time = checkpoint.elapsed();

    let objects = images * per_image;
    println!(
        "wrote {objects} objects ({images} images x {per_image}) in {:.2} s \
         ({:.0} objects/s), peak RSS {:.0} MB; on disk: file {:.0} MB + WAL {:.0} MB, \
         after CHECKPOINT ({:.1} s) file {:.0} MB + WAL {:.0} MB",
        write_time.as_secs_f64(),
        objects as f64 / write_time.as_secs_f64(),
        peak_rss_mb(),
        file_before,
        wal_before,
        checkpoint_time.as_secs_f64(),
        size_mb(&out),
        size_mb(&wal)
    );
}
