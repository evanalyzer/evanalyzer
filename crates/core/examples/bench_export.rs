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
// Usage: cargo run --release -p evanalyzer_core --features ai --example bench_export -- \
//     <out.evadb> [--images N] [--objects-per-image N]

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

fn object(id: u128) -> Object {
    let class = (id % 21) as u32 + 1;
    let mut object_class: HashSet<ObjectClass> = HashSet::from([ObjectClass::Valid(class)]);
    if id % 20 == 0 {
        object_class.insert(ObjectClass::Valid(class % 21 + 1));
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
                    pixel_values: Vec::new(),
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
        colocalized_with.insert(ObjectClass::Valid(class % 21 + 2), partners);
    }
    let x = (id % 1000) as u32 * 4;
    let y = (id / 1000 % 1000) as u32 * 4;
    Object::new(ObjectInit {
        id: ObjectId(id),
        object_class,
        intensities,
        colocalized_with,
        bbox: [x, y, x + 3, y + 3],
        mask_data: BitVec::<u64, Lsb0>::repeat(true, 16),
        area: 16,
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
    let _ = std::fs::remove_file(&out);
    let _ = std::fs::remove_file(format!("{}.wal", out.display()));

    let class_names: HashMap<ObjectClass, (String, u32)> = (1..=22)
        .map(|c| (ObjectClass::Valid(c), (format!("Class {c}"), 0)))
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
            category: "Test category".into(),
            tags: vec![],
            app_version: "v1.0.0".into(),
        },
    )
    .expect("create exporter");

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
                .insert(ObjectId(next_id), object(next_id));
            next_id += 1;
        }
        let start = Instant::now();
        exporter.export(&cache).expect("export");
        exporter
            .finalize_image(Path::new(&cache.image_rel_path), 4096, 4096, 3, 1, 1, None)
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
