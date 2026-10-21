// Benchmark (not part of the test suite) of the time course - the "Over
// time" chart's query and its export - against a time series results
// database (make one with core's `make_time_series` example). Each query is
// the median of `--iters` runs after a warm-up run; each export runs once.
// Peak RSS at the end (`VmHWM`, Linux).
//
// Usage: cargo run --release -p evanalyzer_app --example bench_time_course -- <db.evadb> [--iters N]

use evanalyzer_app::results::{
    Column, ExportFormat, Grouping, LocalResultsGenerator, ResultExport, TimeCourseExport,
    TimeCourseFilter, TimeCourseLayout, TimeCourseLines, TimeCourseNormalize, TimeCourseStat,
};
use std::path::PathBuf;
use std::sync::atomic::AtomicBool;
use std::time::Instant;

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

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let db_path = PathBuf::from(args.get(1).expect("usage: bench_time_course <db.evadb> [--iters N]"));
    let iters: usize = args
        .iter()
        .position(|a| a == "--iters")
        .and_then(|i| args.get(i + 1))
        .and_then(|v| v.parse().ok())
        .unwrap_or(5);

    let start = Instant::now();
    let database = LocalResultsGenerator::open_database(db_path.clone()).expect("open database");
    println!("open: {:.1} ms", start.elapsed().as_secs_f64() * 1000.0);

    let filter = |lines, statistics: &[TimeCourseStat], normalize| TimeCourseFilter {
        lines,
        grouping: Grouping::Auto,
        columns: vec![Column::AreaSizePx],
        statistics: statistics.to_vec(),
        normalize,
        images: None,
        object_classes: None,
        t_stacks: None,
        z_stacks: None,
    };
    let mean_sd = &TimeCourseStat::DEFAULT[..];
    let median_iqr = &[TimeCourseStat::Median, TimeCourseStat::Q1, TimeCourseStat::Q3][..];
    let queries = [
        ("chart: line per class, mean ± SD", filter(TimeCourseLines::Class, mean_sd, TimeCourseNormalize::None)),
        ("chart: line per class, median + IQR", filter(TimeCourseLines::Class, median_iqr, TimeCourseNormalize::None)),
        ("chart: line per class, relative", filter(TimeCourseLines::Class, mean_sd, TimeCourseNormalize::RelativeToStart)),
        ("chart: line per image", filter(TimeCourseLines::Image, mean_sd, TimeCourseNormalize::None)),
        ("chart: line per well", filter(TimeCourseLines::Well, mean_sd, TimeCourseNormalize::None)),
        ("chart: line per track (faint tracks)", filter(TimeCourseLines::Track, &[TimeCourseStat::Mean], TimeCourseNormalize::None)),
    ];
    println!("\n{:<42} {:>10} {:>10}", "query", "median ms", "points");
    for (label, filter) in &queries {
        let mut times = Vec::new();
        let mut points = 0;
        for i in 0..=iters {
            let start = Instant::now();
            let result = database.time_course(filter).expect("time course");
            if i > 0 {
                times.push(start.elapsed().as_secs_f64() * 1000.0);
            }
            points = result.points.len();
        }
        times.sort_by(f64::total_cmp);
        println!("{label:<42} {:>10.1} {points:>10}", times[times.len() / 2]);
    }

    let out_root = std::env::temp_dir().join(format!("evanalyzer_bench_time_course_{}", std::process::id()));
    let export = |label: &str, format: ExportFormat, lines: TimeCourseLines, layout: TimeCourseLayout| {
        let out_dir = out_root.join(label.replace(' ', "_"));
        std::fs::create_dir_all(&out_dir).unwrap();
        let settings = ResultExport {
            output_dir: out_dir.clone(),
            format,
            columns: vec![Column::AreaSizePx, Column::IntensityAvg(0), Column::IntensityAvg(1)],
            with_time_course: true,
            time_course: TimeCourseExport {
                lines,
                layout,
                ..Default::default()
            },
            ..Default::default()
        };
        let start = Instant::now();
        settings
            .start_export(&database, &AtomicBool::new(false), &mut |_, _, _| {})
            .expect("export");
        let elapsed = start.elapsed().as_secs_f64() * 1000.0;
        let (files, bytes) = std::fs::read_dir(&out_dir)
            .unwrap()
            .flatten()
            .fold((0, 0u64), |(n, b), e| (n + 1, b + e.metadata().map_or(0, |m| m.len())));
        println!("{label:<42} {elapsed:>10.1} {files:>6} files {:>8.2} MB", bytes as f64 / 1e6);
    };
    println!("\n{:<42} {:>10}", "export (3 values, mean + SD)", "ms");
    export("CSV long+wide, line per class", ExportFormat::CSV, TimeCourseLines::Class, TimeCourseLayout::Both);
    export("CSV long+wide, line per well", ExportFormat::CSV, TimeCourseLines::Well, TimeCourseLayout::Both);
    export("XLSX long+wide, line per image", ExportFormat::XLSX, TimeCourseLines::Image, TimeCourseLayout::Both);
    export("CSV long, line per track", ExportFormat::CSV, TimeCourseLines::Track, TimeCourseLayout::Long);
    let _ = std::fs::remove_dir_all(&out_root);
    println!("\npeak RSS {:.0} MB", peak_rss_mb());
}
