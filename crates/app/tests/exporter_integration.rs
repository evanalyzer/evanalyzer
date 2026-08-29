//! End-to-end exporter tests against a real DuckDB result file, covering the
//! streaming rewrite in `results_exporter.rs` (Part A of the memory-usage
//! fix): the header row must come first in CSV output (a real bug the
//! streaming rewrite could easily reintroduce, since `csv::Writer` writes in
//! call order), and the coloc-detail export's bounded per-page partner fetch
//! must still resolve partner object data correctly.

use bitvec::prelude::*;
use evanalyzer_app::result::{DatabaseFilter, GroupConfig, ResultsExporter, ResultsLoader};
use evanalyzer_cfg::core_types::{ObjectClass, ObjectId};
use evanalyzer_core::{DuckDbExporter, Object, ObjectInit, GlobalPipelineCache, PipelineResultExporter};
use std::sync::Arc;

fn make_filled_object(id: u128, bbox: [u32; 4], class: ObjectClass) -> Object {
    let [x_min, y_min, x_max, y_max] = bbox;
    let w = (x_max - x_min + 1) as usize;
    let h = (y_max - y_min + 1) as usize;
    let area = w * h;
    let mask_data = BitVec::<u64, Lsb0>::repeat(true, area);
    let mut object = Object::new(ObjectInit {
        id: ObjectId(id),
        bbox,
        mask_data,
        area,
        ..Default::default()
    });
    object.add_object_class(class);
    object
}

/// Writes `objects` to a fresh DuckDB result file and returns a `ResultsLoader`
/// pointed at it, plus the `TempDir` (must stay alive for the file to exist).
fn export_fixture(objects: Vec<Object>) -> (tempfile::TempDir, ResultsLoader) {
    let dir = tempfile::TempDir::new().expect("failed to create temp dir");
    let db_path = dir.path().join("results.duckdb");

    let mut cache = GlobalPipelineCache::default();
    // The exporter rejects an implausible bit depth (guards against
    // `1u64 << nr_of_bits` overflowing); these tests only care about
    // object data, so just supply a plausible value.
    cache.image_meta.nr_of_bits = 16;
    for object in objects {
        cache.object_cache.insert(object.id.clone(), object);
    }

    let exporter = DuckDbExporter::new(&db_path, std::collections::HashMap::new())
        .expect("exporter init failed");
    exporter.export(&cache).expect("export failed");
    drop(exporter);

    (dir, ResultsLoader::new(db_path))
}

#[test]
fn csv_export_writes_header_row_first() {
    const CLASS_A: ObjectClass = ObjectClass::Valid(1);
    let object = make_filled_object(1, [0, 0, 1, 1], CLASS_A);
    let (_db_dir, loader) = export_fixture(vec![object]);

    let out_dir = tempfile::TempDir::new().unwrap();
    let csv_path = out_dir.path().join("out.csv");

    let exporter = ResultsExporter::new(Arc::new(loader));
    exporter
        .export_to_csv(
            DatabaseFilter::default(),
            &GroupConfig::default(),
            &evanalyzer_app::result::build_column_specs(&[], &[]),
            &csv_path,
        )
        .expect("csv export failed");

    let content = std::fs::read_to_string(&csv_path).unwrap();
    let mut lines = content.lines();
    let header = lines.next().expect("csv must have at least a header line");
    assert_eq!(
        header, "object ID,Image,Class,Area (px²),Area (nm²),Circularity,Colocalized",
        "header row must be the first line written, not the last"
    );
    let data_line = lines.next().expect("csv must have one data row");
    assert!(
        data_line.contains("class_1"),
        "data row should follow the header"
    );
}

#[test]
fn coloc_detail_csv_export_resolves_bounded_partner_fetch() {
    const CLASS_SOURCE: ObjectClass = ObjectClass::Valid(1);
    const CLASS_PARTNER: ObjectClass = ObjectClass::Valid(2);

    let mut source = make_filled_object(1, [0, 0, 1, 1], CLASS_SOURCE);
    source.add_colocalizing_object(CLASS_PARTNER, ObjectId(2));
    let partner = make_filled_object(2, [5, 5, 6, 6], CLASS_PARTNER);

    let (_db_dir, loader) = export_fixture(vec![source, partner]);

    let out_dir = tempfile::TempDir::new().unwrap();
    let csv_path = out_dir.path().join("coloc_detail.csv");

    let exporter = ResultsExporter::new(Arc::new(loader));
    exporter
        .export_coloc_detail_to_csv(DatabaseFilter::default(), None, &csv_path)
        .expect("coloc detail csv export failed");

    let content = std::fs::read_to_string(&csv_path).unwrap();
    let mut lines = content.lines();
    let header = lines.next().expect("csv must have a header line");
    assert!(
        header.starts_with("object ID,Image,Class"),
        "header row must come first"
    );
    assert!(
        header.contains("Coloc class_2 object ID"),
        "partner class column group must be present"
    );

    let data_line = lines.next().expect("csv must have one flattened row");
    // The source row's own object_id (a UUID string) should appear, and the
    // bounded per-page partner fetch (via object_id_filter) must have
    // resolved partner id 2's real object_id into the row too.
    let source_id = ObjectId(1).to_string();
    let partner_id = ObjectId(2).to_string();
    assert!(
        data_line.contains(&source_id),
        "source object id missing from row"
    );
    assert!(
        data_line.contains(&partner_id),
        "partner object id missing from row — bounded partner fetch failed to resolve it"
    );
}

#[test]
fn coloc_detail_csv_export_visible_labels_restricts_columns() {
    const CLASS_SOURCE: ObjectClass = ObjectClass::Valid(1);
    const CLASS_PARTNER: ObjectClass = ObjectClass::Valid(2);

    let mut source = make_filled_object(1, [0, 0, 1, 1], CLASS_SOURCE);
    source.add_colocalizing_object(CLASS_PARTNER, ObjectId(2));
    let partner = make_filled_object(2, [5, 5, 6, 6], CLASS_PARTNER);

    let (_db_dir, loader) = export_fixture(vec![source, partner]);

    let out_dir = tempfile::TempDir::new().unwrap();
    let csv_path = out_dir.path().join("coloc_detail_filtered.csv");

    let visible: std::collections::HashSet<String> = ["object ID", "Image", "Class"]
        .into_iter()
        .map(String::from)
        .collect();

    let exporter = ResultsExporter::new(Arc::new(loader));
    exporter
        .export_coloc_detail_to_csv(DatabaseFilter::default(), Some(&visible), &csv_path)
        .expect("coloc detail csv export failed");

    let content = std::fs::read_to_string(&csv_path).unwrap();
    let header = content.lines().next().expect("csv must have a header line");
    assert_eq!(
        header, "object ID,Image,Class",
        "only the requested columns should be written, in spec order"
    );
    assert!(
        !header.contains("Coloc class_2"),
        "partner columns not in visible_labels must be dropped"
    );
}
