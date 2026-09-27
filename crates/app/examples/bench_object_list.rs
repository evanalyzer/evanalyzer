// Throwaway benchmark, run against a real, large `.evadb` file, covering the
// `objects o LEFT JOIN images i ON i.image_rel_path = o.image_rel_path`
// added to `get_object_list`'s per-page row fetch (needed so every List row
// can carry `Cell::disabled`/`ObjectRow::disabled` - see
// `object_select_clause` in results_generator.rs, which now appends
// `COALESCE(i.disabled, false)` to every object query).
//
// The per-page fetch is already narrowed to `object_id IN (<this page's
// ids>)` before the join ever runs (see the two-step id-then-full-row fetch
// in `get_object_list`), so in principle the join should cost close to
// nothing regardless of table size - this confirms that on a real
// multi-million-object database instead of just assuming it from the query
// shape.
//
// Compares, each timed over several pages walked via keyset pagination
// (mirrors real GUI scrolling through the List view, not just re-fetching
// page 1 over and over):
//   - `get_object_list` itself (post-fix, with the join).
//   - the same two-step id-then-full-row query, hand-rolled without the
//     `images` join at all (the pre-fix shape) - the baseline the join is
//     measured against.
//   - the `images` join alone, isolated from `get_object_list`'s other
//     per-column logic, timed directly as raw SQL against the same page of
//     ids the no-join baseline just fetched.
//
// Usage: cargo run --release -p evanalyzer_app --example bench_object_list -- [path.evadb] [--iters N] [--pages N] [--page-size N]

use duckdb::Connection;
use evanalyzer_app::result::{Column, ListFilter, Pagination, PlaneFilter, ResultsGenerator};
use std::path::PathBuf;
use std::time::{Duration, Instant};

fn arg_value(name: &str, default: usize) -> usize {
    let args: Vec<String> = std::env::args().collect();
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1))
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

/// The pre-fix query shape: the same two-step id-then-full-row fetch as
/// `get_object_list`, but `FROM objects` alone - no `images` join, so no
/// `disabled` column and no way to strike a disabled image's row through.
/// Used only to measure what the join costs, not as a feature-complete
/// alternative. Returns this page's ids (for the join-only comparison below
/// and for advancing the cursor) alongside how long the full fetch took.
fn fetch_page_no_join(
    conn: &Connection,
    z_stack: i64,
    t_stack: i64,
    after: Option<&str>,
    limit: i32,
) -> (Vec<String>, Duration) {
    let mut conditions = vec![format!("z_stack = {z_stack}"), format!("t_stack = {t_stack}")];
    if let Some(cursor) = after {
        conditions.push(format!("object_id > '{cursor}'::UUID"));
    }
    let where_clause = format!("WHERE {}", conditions.join(" AND "));

    let start = Instant::now();
    let key_sql =
        format!("SELECT object_id FROM objects {where_clause} ORDER BY object_id LIMIT {limit}");
    let mut key_stmt = conn.prepare(&key_sql).unwrap();
    let ids: Vec<String> = key_stmt
        .query_map([], |row| row.get::<_, String>(0))
        .unwrap()
        .collect::<Result<_, _>>()
        .unwrap();
    if ids.is_empty() {
        return (ids, start.elapsed());
    }
    let in_list = ids
        .iter()
        .map(|id| format!("'{id}'::UUID"))
        .collect::<Vec<_>>()
        .join(", ");
    let sql = format!(
        "SELECT o.object_id, o.image_name, o.object_class_name, o.area_nm2\n\
         FROM objects o\n\
         WHERE o.object_id IN ({in_list})\n\
         ORDER BY o.object_id"
    );
    let mut stmt = conn.prepare(&sql).unwrap();
    let _rows: Vec<(String, String, String, f64)> = stmt
        .query_map([], |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?)))
        .unwrap()
        .collect::<Result<_, _>>()
        .unwrap();
    (ids, start.elapsed())
}

/// Just the join itself, isolated from `get_object_list`'s other per-column
/// logic - the same `object_id IN (...)` narrowing `fetch_page_no_join` just
/// used, joined onto `images` for exactly the `disabled` column
/// `object_select_clause` added.
fn fetch_page_with_join_only(conn: &Connection, ids: &[String]) -> Duration {
    if ids.is_empty() {
        return Duration::ZERO;
    }
    let in_list = ids
        .iter()
        .map(|id| format!("'{id}'::UUID"))
        .collect::<Vec<_>>()
        .join(", ");
    let sql = format!(
        "SELECT o.object_id, o.image_name, o.object_class_name, o.area_nm2, COALESCE(i.disabled, false)\n\
         FROM objects o LEFT JOIN images i ON i.image_rel_path = o.image_rel_path\n\
         WHERE o.object_id IN ({in_list})\n\
         ORDER BY o.object_id"
    );
    let start = Instant::now();
    let mut stmt = conn.prepare(&sql).unwrap();
    let _rows: Vec<(String, String, String, f64, bool)> = stmt
        .query_map([], |row| {
            Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?, row.get(4)?))
        })
        .unwrap()
        .collect::<Result<_, _>>()
        .unwrap();
    start.elapsed()
}

fn main() {
    let default_path = "tests/20260831_Uptake1_Exp0706_PH1.evadb".to_string();
    let path = std::env::args()
        .nth(1)
        .filter(|a| !a.starts_with("--"))
        .unwrap_or(default_path);
    let iters = arg_value("--iters", 3).max(1);
    let pages = arg_value("--pages", 10).max(1);
    let page_size = arg_value("--page-size", 500) as i32;

    let conn = Connection::open(&path).expect("open db (raw)");
    // Whichever plane actually has the most objects - an arbitrary z=0/t=0
    // guess can land on an empty plane on a real acquisition and time
    // nothing at all.
    let (z_stack, t_stack): (i64, i64) = conn
        .query_row(
            "SELECT z_stack, t_stack FROM objects GROUP BY 1, 2 ORDER BY COUNT(*) DESC LIMIT 1",
            [],
            |row| Ok((row.get(0)?, row.get(1)?)),
        )
        .expect("pick the plane with the most objects");
    let total_objects: i64 = conn
        .query_row(
            "SELECT COUNT(*) FROM objects WHERE z_stack = ? AND t_stack = ?",
            duckdb::params![z_stack, t_stack],
            |row| row.get(0),
        )
        .unwrap();
    println!(
        "db: {path}\nplane: z={z_stack} t={t_stack}  ({total_objects} objects)  page size: {page_size}"
    );

    let generator = ResultsGenerator::open_database(PathBuf::from(&path)).expect("open db");
    let columns = vec![
        Column::ObjectId,
        Column::ImageName,
        Column::ObjectClass,
        Column::AreaSizeNm,
    ];

    for iter in 0..iters {
        println!("\n== iteration {} ==", iter + 1);

        let mut join_total = Duration::ZERO;
        let mut join_pages = 0u32;
        let mut after: Option<String> = None;
        for _ in 0..pages {
            let filter = ListFilter {
                plane: PlaneFilter {
                    z_stack: z_stack as u32,
                    t_stack: t_stack as u32,
                },
                images: None,
                object_classes: None,
                columns: columns.clone(),
                with_coloc_details: false,
                page: Pagination {
                    limit: page_size,
                    after: after.clone(),
                },
            };
            let start = Instant::now();
            let result = generator.get_object_list(&filter).unwrap();
            join_total += start.elapsed();
            join_pages += 1;
            let Some(last) = result.row_names.last() else {
                break;
            };
            let got_full_page = result.row_names.len() == page_size as usize;
            after = Some(last.clone());
            if !got_full_page {
                break;
            }
        }
        println!(
            "get_object_list (with join):   avg {:?}/page over {join_pages} pages",
            join_total / join_pages.max(1)
        );

        let mut no_join_total = Duration::ZERO;
        let mut join_only_total = Duration::ZERO;
        let mut no_join_pages = 0u32;
        let mut after: Option<String> = None;
        for _ in 0..pages {
            let (ids, elapsed) =
                fetch_page_no_join(&conn, z_stack, t_stack, after.as_deref(), page_size);
            no_join_total += elapsed;
            if ids.is_empty() {
                break;
            }
            join_only_total += fetch_page_with_join_only(&conn, &ids);
            no_join_pages += 1;
            let got_full_page = ids.len() == page_size as usize;
            after = Some(ids.last().unwrap().clone());
            if !got_full_page {
                break;
            }
        }
        println!(
            "raw, no images join:           avg {:?}/page over {no_join_pages} pages",
            no_join_total / no_join_pages.max(1)
        );
        println!(
            "images join alone (isolated):  avg {:?}/page over {no_join_pages} pages",
            join_only_total / no_join_pages.max(1)
        );
    }
}
