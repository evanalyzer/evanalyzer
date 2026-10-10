// Throwaway benchmark, run against a real, large `.evadb` file, covering the
// "images LEFT JOIN objects, correctly including images with zero matching
// objects and excluding disabled images" fix applied to `get_group_by_plate`,
// `get_group_by_plate_multi`, `get_group_by_well`, `get_wells_for_plate` and
// `get_wells_for_plate_multi`.
//
// For the plate view specifically, three query shapes are compared:
//   - "two SELECTs+merge": the pre-fix approach's *correct* reconstruction
//     (the in-tree WIP it replaced had a bug - its second query's result
//     just overwrote the first instead of merging - so it isn't a fair
//     baseline to time as-is).
//   - "join-of-aggregates": one query, but `objects` is filtered+aggregated
//     down to ~1 row/group *before* joining onto `images`.
//   - `get_group_by_plate` itself, which uses the join-of-aggregates shape -
//     a flat `images LEFT JOIN objects ON ... AND <filter>` was tried first
//     and was ~4x slower (~165ms vs ~37ms on this dataset), since the filter
//     living in the join's `ON` (required to keep an empty-image row) stops
//     DuckDB from filtering `objects` down before the join.
//
// The well-level functions key their join on `image_rel_path` (`images`'
// primary key, a 1:1 lookup rather than the plate view's many-to-one regex
// bucketing) so there's no equivalent "naive join" failure mode to compare
// against - they're timed directly, including one pre-fix-shaped flat-scan
// baseline for `get_wells_for_plate` to confirm the fix didn't cost anything
// (it measured *faster*: pre-aggregating by the plain `image_rel_path`
// column avoids running the regex extraction over every one of `objects`'
// rows just to group by it).
//
// Usage: cargo run --release -p evanalyzer_app --example bench_group_by_plate -- <path.evadb> [--iters N]

use duckdb::Connection;
use evanalyzer_app::results::Aggregation;
use evanalyzer_app::results::ColorScale;
use evanalyzer_app::results::ColorSchema;
use evanalyzer_app::results::Column;
use evanalyzer_app::results::LocalResultsGenerator;
use evanalyzer_app::results::PlaneFilter;
use evanalyzer_app::results::PlateFilter;
use evanalyzer_app::results::PlateFilterMulti;
use evanalyzer_app::results::View;
use evanalyzer_app::results::{Grouping, PlateSize};
use evanalyzer_cfg::core_types::ObjectClass;
use std::path::PathBuf;
use std::time::Instant;

fn arg_value(name: &str, default: usize) -> usize {
    let args: Vec<String> = std::env::args().collect();
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1))
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

/// Reconstructs the pre-fix "two SELECTs, merged in Rust" strategy, fixed to
/// actually merge correctly: seed every group from `images` with `None`,
/// then overlay the aggregated value for every group that has matching
/// objects. Deliberately does *not* filter `disabled` images (that check
/// didn't exist in the pre-fix code) so this is an apples-to-apples timing
/// comparison against the join, not a comparison of the two bugs it fixes.
fn two_selects_merge(conn: &Connection, regex: &str) -> Vec<(String, String, String, Option<f64>)> {
    let mut stmt = conn
        .prepare(&format!(
            "SELECT\n\
                regexp_extract(image_name, '{regex}', 1) AS group_prefix,\n\
                regexp_extract(image_name, '{regex}', 2) AS row,\n\
                regexp_extract(image_name, '{regex}', 3) AS col\n\
             FROM images\n\
             GROUP BY group_prefix, row, col\n\
             ORDER BY group_prefix"
        ))
        .unwrap();
    let mut groups: Vec<(String, String, String, Option<f64>)> = stmt
        .query_map([], |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?, None)))
        .unwrap()
        .collect::<Result<_, _>>()
        .unwrap();

    let mut stmt = conn
        .prepare(&format!(
            "SELECT\n\
                regexp_extract(image_name, '{regex}', 1) AS group_prefix,\n\
                regexp_extract(image_name, '{regex}', 2) AS row,\n\
                regexp_extract(image_name, '{regex}', 3) AS col,\n\
                AVG(area_px) AS value\n\
             FROM objects\n\
             WHERE z_stack = 0 AND t_stack = 0\n\
             GROUP BY group_prefix, row, col\n\
             ORDER BY group_prefix"
        ))
        .unwrap();
    let object_groups: Vec<(String, String, String, Option<f64>)> = stmt
        .query_map([], |row| {
            Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?))
        })
        .unwrap()
        .collect::<Result<_, _>>()
        .unwrap();

    for (prefix, row, col, value) in object_groups {
        if let Some(entry) = groups
            .iter_mut()
            .find(|(p, r, c, _)| *p == prefix && *r == row && *c == col)
        {
            entry.3 = value;
        }
    }
    groups
}

/// A single query, but structured so DuckDB filters/aggregates `objects`
/// down to (at most) one row per group *before* ever joining - unlike the
/// production `get_group_by_plate` query, which puts the filter in a
/// `LEFT JOIN ... ON` against the raw, unfiltered `objects` table.
fn join_of_aggregates(
    conn: &Connection,
    regex: &str,
) -> Vec<(String, String, String, Option<f64>)> {
    let sql = format!(
        "SELECT img.group_prefix, img.row, img.col, agg.value\n\
         FROM (\n\
             SELECT\n\
                 regexp_extract(image_name, '{regex}', 1) AS group_prefix,\n\
                 regexp_extract(image_name, '{regex}', 2) AS row,\n\
                 regexp_extract(image_name, '{regex}', 3) AS col\n\
             FROM images\n\
             WHERE NOT disabled\n\
             GROUP BY group_prefix, row, col\n\
         ) img\n\
         LEFT JOIN (\n\
             SELECT\n\
                 regexp_extract(o.image_name, '{regex}', 1) AS group_prefix,\n\
                 regexp_extract(o.image_name, '{regex}', 2) AS row,\n\
                 regexp_extract(o.image_name, '{regex}', 3) AS col,\n\
                 AVG(o.area_px) AS value\n\
             FROM objects o\n\
             JOIN images i ON i.image_rel_path = o.image_rel_path\n\
             WHERE NOT i.disabled AND o.z_stack = 0 AND o.t_stack = 0\n\
             GROUP BY group_prefix, row, col\n\
         ) agg USING (group_prefix, row, col)\n\
         ORDER BY img.group_prefix"
    );
    let mut stmt = conn.prepare(&sql).unwrap();
    stmt.query_map([], |row| {
        Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?))
    })
    .unwrap()
    .collect::<Result<_, _>>()
    .unwrap()
}

fn main() {
    let path = std::env::args()
        .nth(1)
        .expect("usage: bench_group_by_plate <path.evadb> [--iters N]");
    let iters = arg_value("--iters", 5).max(1);

    let filter = PlateFilter {
        plane: PlaneFilter {
            z_stack: 0,
            t_stack: 0,
        },
        grouping: Grouping::Auto,
        aggregation: Aggregation::Avg,
        object_class: ObjectClass::Unset,
        column: Column::AreaSizePx,
        color_schema: ColorSchema::default(),
        color_scale: ColorScale::default(),
        plate_size: PlateSize::Auto,
    };

    let generator = LocalResultsGenerator::open_database(PathBuf::from(&path)).expect("open db");
    let mut join_total = std::time::Duration::ZERO;
    let mut join_rows = 0;
    for _ in 0..iters {
        let start = Instant::now();
        let result = generator.get_group_by_plate(&filter, &View::List).unwrap();
        join_total += start.elapsed();
        join_rows = result.row_names.len();
    }

    let conn = Connection::open(&path).expect("open db (raw)");
    // Mirrors `results_generator.rs`'s private `DEFAULT_GROUPING_REGEX`.
    let regex = r"^(([A-H])([0-9]{1,2}))_([0-9]+)\.([a-zA-Z0-9]+)$";
    let mut two_select_total = std::time::Duration::ZERO;
    let mut two_select_rows = 0;
    for _ in 0..iters {
        let start = Instant::now();
        let groups = two_selects_merge(&conn, regex);
        two_select_total += start.elapsed();
        two_select_rows = groups.len();
    }

    println!(
        "get_group_by_plate: avg {:?} over {iters} iters ({join_rows} groups)",
        join_total / iters as u32
    );
    println!(
        "two SELECTs+merge:  avg {:?} over {iters} iters ({two_select_rows} groups)",
        two_select_total / iters as u32
    );

    let mut join_agg_total = std::time::Duration::ZERO;
    let mut join_agg_rows = 0;
    for _ in 0..iters {
        let start = Instant::now();
        let groups = join_of_aggregates(&conn, regex);
        join_agg_total += start.elapsed();
        join_agg_rows = groups.len();
    }
    println!(
        "join-of-aggregates: avg {:?} over {iters} iters ({join_agg_rows} groups)",
        join_agg_total / iters as u32
    );

    // Sibling group/well queries, fixed the same way (filter+aggregate
    // `objects` before joining onto `images`) - timed directly through the
    // real generator methods rather than a hand-rolled comparison query,
    // since their join key (`image_rel_path`, `images`' primary key) is a
    // simple 1:1 lookup rather than the plate view's many-to-one regex
    // bucketing, so there's no separate "naive" baseline worth reconstructing.
    let multi_filter = PlateFilterMulti {
        plane: filter.plane.clone(),
        grouping: Grouping::Auto,
        aggregation: vec![Aggregation::Avg, Aggregation::Sum],
        object_class: vec![ObjectClass::Unset],
        column: vec![Column::AreaSizePx, Column::PerimeterPx],
        color_schema: ColorSchema::default(),
        color_scale: ColorScale::default(),
        plate_size: PlateSize::Auto,
    };
    let start = Instant::now();
    for _ in 0..iters {
        generator
            .get_group_by_plate_multi(&multi_filter, &View::List)
            .unwrap();
    }
    println!(
        "get_group_by_plate_multi: avg {:?} over {iters} iters",
        start.elapsed() / iters as u32
    );

    let well_filter = evanalyzer_app::results::WellFilter {
        plane: filter.plane.clone(),
        group_name: "A2".to_string(),
        grouping: Grouping::Auto,
        aggregation: Aggregation::Avg,
        object_class: ObjectClass::Unset,
        column: Column::AreaSizePx,
        color_schema: ColorSchema::default(),
        color_scale: ColorScale::default(),
        well_size: None,
        well_order: None,
    };
    let start = Instant::now();
    for _ in 0..iters {
        generator
            .get_group_by_well(&well_filter, &View::List)
            .unwrap();
    }
    println!(
        "get_group_by_well: avg {:?} over {iters} iters",
        start.elapsed() / iters as u32
    );

    let wells_filter = evanalyzer_app::results::WellsBatchFilter {
        plane: filter.plane.clone(),
        grouping: Grouping::Auto,
        aggregation: Aggregation::Avg,
        object_class: ObjectClass::Unset,
        column: Column::AreaSizePx,
        color_schema: ColorSchema::default(),
        color_scale: ColorScale::default(),
        well_size: None,
        well_order: None,
    };
    let start = Instant::now();
    for _ in 0..iters {
        generator
            .get_wells_for_plate(&wells_filter, &View::List)
            .unwrap();
    }
    println!(
        "get_wells_for_plate: avg {:?} over {iters} iters",
        start.elapsed() / iters as u32
    );

    // Pre-fix baseline: one flat scan straight `FROM objects`, no `images`
    // join at all - misses empty-object images and leaks disabled images'
    // objects, but costs nothing extra. Timed to show what the correctness
    // fix above actually costs on this dataset.
    let start = Instant::now();
    for _ in 0..iters {
        let sql = format!(
            "SELECT\n\
                regexp_extract(image_name, '{regex}', 1) AS group_prefix,\n\
                regexp_extract(image_name, '{regex}', 4) AS idx,\n\
                image_rel_path,\n\
                image_name,\n\
                AVG(area_px) AS value\n\
             FROM objects\n\
             WHERE z_stack = 0 AND t_stack = 0\n\
             GROUP BY group_prefix, idx, image_rel_path, image_name\n\
             ORDER BY group_prefix, idx"
        );
        let mut stmt = conn.prepare(&sql).unwrap();
        let _rows: Vec<(String, String, String, String, Option<f64>)> = stmt
            .query_map([], |row| {
                Ok((
                    row.get(0)?,
                    row.get(1)?,
                    row.get(2)?,
                    row.get(3)?,
                    row.get(4)?,
                ))
            })
            .unwrap()
            .collect::<Result<_, _>>()
            .unwrap();
    }
    println!(
        "get_wells_for_plate (pre-fix, flat scan): avg {:?} over {iters} iters",
        start.elapsed() / iters as u32
    );

    let wells_multi_filter = evanalyzer_app::results::WellsBatchFilterMulti {
        plane: filter.plane.clone(),
        grouping: Grouping::Auto,
        aggregation: vec![Aggregation::Avg, Aggregation::Sum],
        object_class: vec![ObjectClass::Unset],
        column: vec![Column::AreaSizePx, Column::PerimeterPx],
        color_schema: ColorSchema::default(),
        color_scale: ColorScale::default(),
        well_size: None,
        well_order: None,
    };
    let start = Instant::now();
    for _ in 0..iters {
        generator
            .get_wells_for_plate_multi(&wells_multi_filter, &View::List)
            .unwrap();
    }
    println!(
        "get_wells_for_plate_multi: avg {:?} over {iters} iters",
        start.elapsed() / iters as u32
    );
}
