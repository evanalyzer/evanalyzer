//! # compare_exporters
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-11-25
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.

// Throwaway check (not part of the test suite) before the old results
// export's table writers are removed: the same results written by the old
// exporter (`ResultsSource::export`: list + grouped by image) and by the
// new one (`run_export`: OBJECTS + IMAGES), compared value by value.
//
// Usage: cargo run --release -p evanalyzer_app --example compare_exporters -- <path.evadb> <scratch dir>

use evanalyzer_app::backend::Backend;
use evanalyzer_app::backend::local::LocalBackend;
use evanalyzer_app::exports::{ExportRun, ScriptPermission};
use evanalyzer_app::results::{Aggregation, Column, ExportFormat, ResultExport};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicBool;

/// A CSV file as rows of cells (quoted cells with commas kept whole).
fn csv(path: &Path) -> Vec<Vec<String>> {
    let text = std::fs::read_to_string(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    text.lines()
        .map(|line| {
            let mut cells = Vec::new();
            let mut cell = String::new();
            let mut quoted = false;
            for c in line.chars() {
                match c {
                    '"' => quoted = !quoted,
                    ',' if !quoted => cells.push(std::mem::take(&mut cell)),
                    c => cell.push(c),
                }
            }
            cells.push(cell);
            cells
        })
        .collect()
}

/// Two cells the same: as text, or as numbers within float rounding.
fn same(a: &str, b: &str) -> bool {
    if a == b {
        return true;
    }
    match (a.parse::<f64>(), b.parse::<f64>()) {
        (Ok(a), Ok(b)) => (a - b).abs() <= 1e-6 * a.abs().max(b.abs()).max(1.0),
        _ => false,
    }
}

fn column(header: &[String], name: &str) -> usize {
    header
        .iter()
        .position(|h| h == name)
        .unwrap_or_else(|| panic!("no column {name} in {header:?}"))
}

fn main() {
    let mut args = std::env::args().skip(1);
    let db = PathBuf::from(args.next().expect("usage: compare_exporters <db> <scratch dir>"));
    let scratch = PathBuf::from(args.next().expect("a scratch dir"));
    let (old_dir, new_dir) = (scratch.join("old"), scratch.join("new"));
    std::fs::create_dir_all(&old_dir).unwrap();
    std::fs::create_dir_all(&new_dir).unwrap();
    let backend = LocalBackend::default();
    let results = backend.open_results(&db).expect("open results");
    let cancel = AtomicBool::new(false);

    let start = std::time::Instant::now();
    results
        .export(
            &ResultExport {
                output_dir: old_dir.clone(),
                format: ExportFormat::CSV,
                columns: vec![
                    Column::ObjectClass,
                    Column::ObjectId,
                    Column::Count,
                    Column::AreaSizePx,
                    Column::Circularity,
                ],
                aggregations: vec![Aggregation::Avg, Aggregation::Median],
                with_list_view: true,
                with_grouped_by_image_list: true,
                ..Default::default()
            },
            &cancel,
            &mut |_, _, _| {},
        )
        .expect("old export");
    println!("old exporter: {:.1} s", start.elapsed().as_secs_f64());

    let export = serde_json::from_value(serde_json::json!({
        "id": 1, "name": "Compare", "entries": [
            { "kind": "DATA", "outputs": ["OBJECTS"], "columns": ["area_px", "circularity"] },
            { "kind": "DATA", "outputs": ["IMAGES"], "columns": ["count", "area_px", "circularity"],
              "aggregations": ["AVG", "MEDIAN"], "emptyClasses": true }
        ]
    }))
    .unwrap();
    let start = std::time::Instant::now();
    let report = results
        .run_export(
            &ExportRun {
                export,
                project_dir: scratch.clone(),
                results_dir: db.parent().unwrap().to_path_buf(),
                output: Some(new_dir.clone()),
                scripts: ScriptPermission::Skip,
                interpreters: Default::default(),
                drawing: None,
            },
            &cancel,
            &mut |_, _, _| {},
        )
        .expect("new export");
    println!("new exporter: {:.1} s", start.elapsed().as_secs_f64());
    assert!(report.all_written(), "{report:?}");

    // -- A row per object: by object id --
    let old = csv(&old_dir.join("list.csv"));
    let new = csv(&new_dir.join("objects.csv"));
    println!("old list header: {:?}", old[0]);
    println!("new objects header: {:?}", new[0]);
    let (old_id, old_class) = (column(&old[0], "Object ID"), column(&old[0], "Class"));
    let old_values = [column(&old[0], "Area [px]"), column(&old[0], "Circularity")];
    let (new_id, new_class) = (column(&new[0], "object_id"), column(&new[0], "class"));
    let new_values = [column(&new[0], "area_px"), column(&new[0], "circularity")];
    let by_id: BTreeMap<&str, &Vec<String>> =
        old[1..].iter().map(|row| (row[old_id].as_str(), row)).collect();
    assert_eq!(old.len(), new.len(), "as many objects");
    assert_eq!(by_id.len(), old.len() - 1, "unique object ids");
    let mut compared = 0usize;
    for row in &new[1..] {
        let other = by_id
            .get(row[new_id].as_str())
            .unwrap_or_else(|| panic!("object {} not in the old list", row[new_id]));
        assert_eq!(row[new_class], other[old_class], "class of {}", row[new_id]);
        for (n, o) in new_values.iter().zip(old_values) {
            assert!(
                same(&row[*n], &other[o]),
                "object {}: {} != {}",
                row[new_id],
                row[*n],
                other[o]
            );
            compared += 1;
        }
    }
    println!("objects: {} rows, {compared} values equal", new.len() - 1);

    // -- A row per image and class: by (image, class) --
    let old = csv(&old_dir.join("grouped_by_image.csv"));
    let new = csv(&new_dir.join("images.csv"));
    println!("old grouped header: {:?}", old[0]);
    println!("new images header: {:?}", new[0]);
    // Old: image name, class, then per column and aggregation; new: the
    // image's relative path, class, count once, then the same order.
    let old_first = 2;
    let new_first = 2;
    // Count is the same for every aggregation: the old table repeats its
    // column, the new one has it once.
    let old_cols: Vec<usize> = (old_first..old[0].len())
        .filter(|i| !old[0][old_first..*i].contains(&old[0][*i]))
        .filter(|i| !matches!(old[0][*i].to_lowercase().as_str(), "marker" | "note"))
        .collect();
    let new_cols: Vec<usize> = (new_first..new[0].len()).collect();
    assert_eq!(old_cols.len(), new_cols.len(), "as many value columns");
    let file_name = |path: &str| path.rsplit('/').next().unwrap_or(path).to_string();
    let key = |row: &Vec<String>| (file_name(&row[0]), row[1].clone());
    let by_key: BTreeMap<(String, String), &Vec<String>> =
        old[1..].iter().map(|row| (key(row), row)).collect();
    assert_eq!(by_key.len(), old.len() - 1, "unique (image, class) in the old table");
    assert_eq!(old.len(), new.len(), "as many image rows");
    let mut compared = 0usize;
    for row in &new[1..] {
        let other = by_key
            .get(&key(row))
            .unwrap_or_else(|| panic!("row {:?} not in the old table", key(row)));
        for (n, o) in new_cols.iter().zip(&old_cols) {
            assert!(
                same(&row[*n], &other[*o]),
                "{:?} column {}: {} != {}",
                key(row),
                new[0][*n],
                row[*n],
                other[*o]
            );
            compared += 1;
        }
    }
    println!("images: {} rows, {compared} values equal", new.len() - 1);
    println!("OK");
}
