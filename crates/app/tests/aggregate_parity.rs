//! Parity tests for Part B of the memory-usage fix: `aggregate_objects_sql` (the
//! new SQL-pushdown grouped/aggregated query, computed in DuckDB) must produce
//! byte-for-byte the same `(ColumnSpec, DisplayRow)` output as `aggregate_rows`
//! (the existing, thoroughly-unit-tested in-memory Rust implementation) for
//! the same fixture and `GroupConfig` — `aggregate_rows` is the ground-truth
//! spec for grouping semantics; these tests are the guard against the SQL
//! path silently drifting from it.

use bitvec::prelude::*;
use evanalyzer_app::result::{
    ColumnSpec, DatabaseFilter, GroupBy, GroupConfig, ResultsLoader, aggregate_objects_sql,
    aggregate_rows, build_column_specs,
};
use evanalyzer_cfg::core_types::{ObjectClass, ObjectId};
use evanalyzer_core::{
    DuckDbExporter, Intensity, Object, ObjectInit, GlobalPipelineCache, PipelineResultExporter,
};
use indexmap::IndexMap;
use std::path::PathBuf;

fn make_object(id: u128, bbox: [u32; 4], classes: &[ObjectClass]) -> Object {
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
    for class in classes {
        object.add_object_class(*class);
    }
    object
}

fn with_channel0(mut object: Object, min: f32, max: f32, avg: f32) -> Object {
    object.intensities = IndexMap::from([(
        0,
        Intensity {
            sum_intensity: 0.0,
            min_intensity: min,
            max_intensity: max,
            avg_intensity: avg,
            pixel_values: vec![],
        },
    )]);
    object
}

/// Exports each `(image_rel_path, objects)` pair as its own image (mirroring real
/// usage — one `PipelineCache`/`export()` call per image) into a fresh DuckDB
/// file, and returns a `ResultsLoader` over it.
fn export_fixture(images: Vec<(&str, Vec<Object>)>) -> (tempfile::TempDir, ResultsLoader) {
    let dir = tempfile::TempDir::new().expect("failed to create temp dir");
    let db_path = dir.path().join("results.duckdb");

    let exporter = DuckDbExporter::new(&db_path, std::collections::HashMap::new())
        .expect("exporter init failed");
    for (image_rel_path, objects) in images {
        let mut cache = GlobalPipelineCache {
            image_rel_path: PathBuf::from(image_rel_path),
            ..Default::default()
        };
        // The exporter rejects an implausible bit depth (guards against
        // `1u64 << nr_of_bits` overflowing); these tests only care about
        // object/aggregation data, so just supply a plausible value.
        cache.image_meta.nr_of_bits = 16;
        for object in objects {
            cache.object_cache.insert(object.id.clone(), object);
        }
        exporter.export(&cache).expect("export failed");
    }
    drop(exporter);

    (dir, ResultsLoader::new(db_path))
}

/// Runs both `aggregate_rows` (over every object fetched via `get_objects`) and
/// `aggregate_objects_sql`, and asserts they produce the same column specs
/// (id/label) and the same row values, in the same order.
fn assert_parity(loader: &ResultsLoader, config: &GroupConfig, base_specs: &[ColumnSpec]) {
    let all_objects = loader
        .get_objects(DatabaseFilter {
            page_size: 0,
            ..Default::default()
        })
        .unwrap();
    let (rust_specs, rust_rows) = aggregate_rows(&all_objects, config, base_specs);
    let (sql_specs, sql_rows) =
        aggregate_objects_sql(loader, DatabaseFilter::default(), config, base_specs).unwrap();

    let spec_ids = |specs: &[ColumnSpec]| {
        specs
            .iter()
            .map(|c| (c.id.clone(), c.label.clone()))
            .collect::<Vec<_>>()
    };
    assert_eq!(
        spec_ids(&rust_specs),
        spec_ids(&sql_specs),
        "column specs must match exactly"
    );

    let row_values = |rows: &[evanalyzer_app::result::DisplayRow]| {
        rows.iter().map(|r| r.values.clone()).collect::<Vec<_>>()
    };
    assert_eq!(
        row_values(&rust_rows),
        row_values(&sql_rows),
        "row values must match exactly (Rust ground-truth vs SQL pushdown)"
    );
}

#[test]
fn parity_group_by_image_multiple_aggregates() {
    let img_a = vec![
        with_channel0(make_object(1, [0, 0, 9, 9], &[]), 10.0, 100.0, 50.0), // area_px=100
        with_channel0(make_object(2, [0, 0, 19, 9], &[]), 20.0, 200.0, 100.0), // area_px=200
    ];
    let img_b = vec![make_object(3, [0, 0, 29, 9], &[])]; // area_px=300, no channel data

    let (_dir, loader) = export_fixture(vec![("img_a.tif", img_a), ("img_b.tif", img_b)]);

    let base_specs = build_column_specs(&[0], &[]);
    let config = GroupConfig {
        group_by: GroupBy::Image,
        regex: String::new(),
        aggs: vec![
            evanalyzer_app::result::AggFunc::Min,
            evanalyzer_app::result::AggFunc::Max,
            evanalyzer_app::result::AggFunc::Avg,
            evanalyzer_app::result::AggFunc::Median,
            evanalyzer_app::result::AggFunc::Stdev,
            evanalyzer_app::result::AggFunc::Sum,
        ],
        split_colocalized: false,
        group_by_class: false,
    };

    assert_parity(&loader, &config, &base_specs);
}

#[test]
fn parity_group_by_folder() {
    let a1 = vec![make_object(1, [0, 0, 9, 9], &[])];
    let a2 = vec![make_object(2, [0, 0, 9, 9], &[])];
    let b1 = vec![make_object(3, [0, 0, 9, 9], &[])];
    let root1 = vec![make_object(4, [0, 0, 9, 9], &[])];

    let (_dir, loader) = export_fixture(vec![
        ("plate1/wellA/img1.tif", a1),
        ("plate1/wellA/img2.tif", a2),
        ("plate1/wellB/img3.tif", b1),
        ("root_img.tif", root1),
    ]);

    let base_specs = build_column_specs(&[], &[]);
    let config = GroupConfig {
        group_by: GroupBy::Folder,
        regex: String::new(),
        aggs: vec![evanalyzer_app::result::AggFunc::Avg],
        split_colocalized: false,
        group_by_class: false,
    };

    assert_parity(&loader, &config, &base_specs);
}

#[test]
fn parity_group_by_regex_with_capture_group_and_non_matching_row() {
    let a1 = vec![make_object(1, [0, 0, 9, 9], &[])];
    let a2 = vec![make_object(2, [0, 0, 9, 9], &[])];
    let control = vec![make_object(3, [0, 0, 9, 9], &[])];

    let (_dir, loader) = export_fixture(vec![
        ("A1_01.tif", a1),
        ("A1_02.tif", a2),
        ("control.tif", control),
    ]);

    let base_specs = build_column_specs(&[], &[]);
    let config = GroupConfig {
        group_by: GroupBy::Regex,
        regex: r"^([A-Z]\d+)_".to_string(),
        aggs: vec![evanalyzer_app::result::AggFunc::Avg],
        split_colocalized: false,
        group_by_class: false,
    };

    assert_parity(&loader, &config, &base_specs);
}

#[test]
fn parity_group_by_regex_invalid_pattern_yields_no_groups() {
    let (_dir, loader) = export_fixture(vec![("img.tif", vec![make_object(1, [0, 0, 9, 9], &[])])]);

    let base_specs = build_column_specs(&[], &[]);
    let config = GroupConfig {
        group_by: GroupBy::Regex,
        regex: "(unclosed".to_string(),
        aggs: vec![evanalyzer_app::result::AggFunc::Avg],
        split_colocalized: false,
        group_by_class: false,
    };

    assert_parity(&loader, &config, &base_specs);
}

#[test]
fn parity_group_by_class_fans_out_multi_class_object() {
    const CLASS_A: ObjectClass = ObjectClass::Valid(1);
    const CLASS_B: ObjectClass = ObjectClass::Valid(2);

    let multi = make_object(1, [0, 0, 9, 9], &[CLASS_A, CLASS_B]);
    let single = make_object(2, [0, 0, 19, 9], &[CLASS_A]);

    let (_dir, loader) = export_fixture(vec![("img.tif", vec![multi, single])]);

    let base_specs = build_column_specs(&[], &[]);
    let config = GroupConfig {
        group_by: GroupBy::Image,
        regex: String::new(),
        aggs: vec![evanalyzer_app::result::AggFunc::Avg],
        split_colocalized: false,
        group_by_class: true,
    };

    assert_parity(&loader, &config, &base_specs);
}

#[test]
fn parity_split_colocalized() {
    let mut coloc = make_object(1, [0, 0, 9, 9], &[]);
    coloc.add_colocalizing_object(ObjectClass::Valid(9), ObjectId(99));
    let not_coloc = make_object(2, [0, 0, 9, 9], &[]);

    let (_dir, loader) = export_fixture(vec![("img.tif", vec![coloc, not_coloc])]);

    let base_specs = build_column_specs(&[], &[]);
    let config = GroupConfig {
        group_by: GroupBy::Image,
        regex: String::new(),
        aggs: vec![evanalyzer_app::result::AggFunc::Avg],
        split_colocalized: true,
        group_by_class: false,
    };

    assert_parity(&loader, &config, &base_specs);
}

#[test]
fn parity_coloc_partner_count_is_aggregatable_metric() {
    let mut object1 = make_object(1, [0, 0, 9, 9], &[]);
    object1.add_colocalizing_object(ObjectClass::Valid(2), ObjectId(101));
    let mut object2 = make_object(2, [0, 0, 9, 9], &[]);
    object2.add_colocalizing_object(ObjectClass::Valid(2), ObjectId(101));
    object2.add_colocalizing_object(ObjectClass::Valid(2), ObjectId(102));

    let (_dir, loader) = export_fixture(vec![("img.tif", vec![object1, object2])]);

    let coloc_partner_classes = loader.get_coloc_partner_class_names().unwrap();
    let base_specs = build_column_specs(&[], &coloc_partner_classes);
    let config = GroupConfig {
        group_by: GroupBy::Image,
        regex: String::new(),
        aggs: vec![evanalyzer_app::result::AggFunc::Max],
        split_colocalized: false,
        group_by_class: false,
    };

    assert_parity(&loader, &config, &base_specs);
}

#[test]
fn parity_respects_column_visibility() {
    let objects = vec![
        make_object(1, [0, 0, 9, 9], &[]),
        make_object(2, [0, 0, 19, 9], &[]),
    ];
    let (_dir, loader) = export_fixture(vec![("img.tif", objects)]);

    let mut base_specs = build_column_specs(&[], &[]);
    for spec in base_specs.iter_mut() {
        spec.visible = spec.id == "area_px";
    }
    let config = GroupConfig {
        group_by: GroupBy::Image,
        regex: String::new(),
        aggs: vec![evanalyzer_app::result::AggFunc::Avg],
        split_colocalized: false,
        group_by_class: false,
    };

    assert_parity(&loader, &config, &base_specs);
}

#[test]
fn parity_string_ordering_mixed_case_and_non_ascii() {
    // DuckDB's default VARCHAR ORDER BY collation vs Rust's byte-wise `Ord for
    // str` could in principle disagree on row order for non-trivial names —
    // this pins down that they don't, for mixed-case ASCII and non-ASCII
    // (UTF-8 multi-byte) image names.
    let images = vec![
        ("Zebra.tif", vec![make_object(1, [0, 0, 9, 9], &[])]),
        ("apple.tif", vec![make_object(2, [0, 0, 9, 9], &[])]),
        ("Apple.tif", vec![make_object(3, [0, 0, 9, 9], &[])]),
        ("\u{00e9}clair.tif", vec![make_object(4, [0, 0, 9, 9], &[])]), // "éclair.tif"
        ("zebra.tif", vec![make_object(5, [0, 0, 9, 9], &[])]),
    ];
    let (_dir, loader) = export_fixture(images);

    let base_specs = build_column_specs(&[], &[]);
    let config = GroupConfig {
        group_by: GroupBy::Image,
        regex: String::new(),
        aggs: vec![evanalyzer_app::result::AggFunc::Avg],
        split_colocalized: false,
        group_by_class: false,
    };

    assert_parity(&loader, &config, &base_specs);
}

#[test]
fn parity_empty_result_set() {
    let (_dir, loader) = export_fixture(vec![("img.tif", vec![make_object(1, [0, 0, 9, 9], &[])])]);

    let base_specs = build_column_specs(&[], &[]);
    let config = GroupConfig {
        group_by: GroupBy::Image,
        regex: String::new(),
        aggs: vec![evanalyzer_app::result::AggFunc::Avg],
        split_colocalized: false,
        group_by_class: false,
    };

    // A filter that matches nothing (unknown image).
    let all_objects = loader
        .get_objects(DatabaseFilter {
            image_filter: Some(vec!["does_not_exist.tif".to_string()]),
            ..Default::default()
        })
        .unwrap();
    assert!(all_objects.is_empty());
    let (rust_specs, rust_rows) = aggregate_rows(&all_objects, &config, &base_specs);
    let (sql_specs, sql_rows) = aggregate_objects_sql(
        &loader,
        DatabaseFilter {
            image_filter: Some(vec!["does_not_exist.tif".to_string()]),
            ..Default::default()
        },
        &config,
        &base_specs,
    )
    .unwrap();

    assert!(rust_rows.is_empty());
    assert!(sql_rows.is_empty());
    assert_eq!(
        rust_specs.iter().map(|c| c.id.clone()).collect::<Vec<_>>(),
        sql_specs.iter().map(|c| c.id.clone()).collect::<Vec<_>>()
    );
}
