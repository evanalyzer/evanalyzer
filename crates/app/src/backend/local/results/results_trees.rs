//! # results_trees
//!
//! Traced neurons in the results (Trace Filaments, Measure Filament Tree):
//! Results › Trees' list of neurons and the segments of one.
//!
//! The results don't record which classes a trace used; they are read off
//! the objects: the segment class is the one whose objects have a parent of
//! their own class (a segment continues another), the soma class the other
//! parent class of the root segments (those without one of their own).
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-11-09
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.

use super::results_generator::{ResultsGenerator, sql_string_in_list};
use crate::api::{TREE_NEURON_VALUES, TreeNeuron, TreeNeurons, TreeSegment};
use evanalyzer_cfg::core_types::InternalErrors;
use std::collections::HashMap;

fn err(e: duckdb::Error) -> InternalErrors {
    InternalErrors::Io(e.to_string())
}

/// The segment values of Measure Filament Tree, by the end of their key.
const SEGMENT_KEYS: [&str; 5] = [
    "length",
    "path_distance",
    "branch_level",
    "strahler_order",
    "straightness",
];

/// The classes of the traced trees: (segment class, soma class); `None`:
/// no traced trees.
pub(super) fn tree_classes(db: &ResultsGenerator) -> Result<Option<(i64, i64)>, InternalErrors> {
    if !db.has_parent_ids()? {
        return Ok(None);
    }
    let conn = db.connection();
    let segment: Option<i64> = conn
        .query_row(
            "SELECT c FROM (SELECT unnest(object_class_id) AS c, parent_ids FROM objects \
             WHERE cardinality(parent_ids) > 0) WHERE list_contains(map_keys(parent_ids), c) \
             GROUP BY c ORDER BY count(*) DESC, c LIMIT 1",
            [],
            |row| row.get(0),
        )
        .map(Some)
        .or_else(|e| match e {
            duckdb::Error::QueryReturnedNoRows => Ok(None),
            e => Err(err(e)),
        })?;
    let Some(segment) = segment else {
        return Ok(None);
    };
    let soma: Option<i64> = conn
        .query_row(
            "SELECT k FROM (SELECT unnest(map_keys(parent_ids)) AS k FROM objects \
             WHERE list_contains(object_class_id, ?) \
             AND NOT list_contains(map_keys(parent_ids), ?)) \
             WHERE k <> ? GROUP BY k ORDER BY count(*) DESC, k LIMIT 1",
            [segment, segment, segment],
            |row| row.get(0),
        )
        .map(Some)
        .or_else(|e| match e {
            duckdb::Error::QueryReturnedNoRows => Ok(None),
            e => Err(err(e)),
        })?;
    Ok(soma.map(|soma| (segment, soma)))
}

/// The results hold traced trees (Results › Trees is offered): a class
/// linked to a parent of its own class - read off the relations the column
/// list reads anyway (cached), so asking costs nothing when results open.
pub(super) fn has_trees(db: &ResultsGenerator) -> Result<bool, InternalErrors> {
    Ok(db
        .get_relations()?
        .iter()
        .any(|(parent, child)| parent == child))
}

/// The key of Measure Filament Tree's value `id` (the first such step):
/// `None` when no step measured it.
fn tree_key(db: &ResultsGenerator, id: &str) -> Result<Option<String>, InternalErrors> {
    let exists: bool = db
        .connection()
        .query_row(
            "SELECT count(*) > 0 FROM information_schema.tables WHERE table_name = 'measurement_info'",
            [],
            |row| row.get(0),
        )
        .map_err(err)?;
    if !exists {
        return Ok(None);
    }
    db.connection()
        .query_row(
            "SELECT key FROM measurement_info WHERE key LIKE ? ORDER BY key LIMIT 1",
            [format!("%/{id}")],
            |row| row.get(0),
        )
        .map(Some)
        .or_else(|e| match e {
            duckdb::Error::QueryReturnedNoRows => Ok(None),
            e => Err(err(e)),
        })
}

/// The values of `keys` of the objects in `ids` (`object_id -> key -> value`).
fn object_values(
    db: &ResultsGenerator,
    keys: &[String],
    ids: &[String],
) -> Result<HashMap<String, HashMap<String, f64>>, InternalErrors> {
    let mut out: HashMap<String, HashMap<String, f64>> = HashMap::new();
    if keys.is_empty() || ids.is_empty() {
        return Ok(out);
    }
    let sql = format!(
        "SELECT object_id::VARCHAR, key, value FROM measurements \
         WHERE region = 'object' AND value IS NOT NULL AND key IN ({}) \
         AND object_id::VARCHAR IN ({})",
        sql_string_in_list(keys),
        sql_string_in_list(ids)
    );
    let mut stmt = db.connection().prepare(&sql).map_err(err)?;
    let rows = stmt
        .query_map([], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, f64>(2)?,
            ))
        })
        .map_err(err)?;
    for row in rows {
        let (id, key, value) = row.map_err(err)?;
        out.entry(id).or_default().insert(key, value);
    }
    Ok(out)
}

/// The traced neurons (of `images`, `None`: all), by image and soma - the
/// failed and left out images' not.
pub(super) fn tree_neurons(
    db: &ResultsGenerator,
    images: Option<&[String]>,
) -> Result<TreeNeurons, InternalErrors> {
    let Some((segment, soma)) = tree_classes(db)? else {
        return Ok(TreeNeurons::default());
    };
    let image_filter = match images {
        Some(images) if images.is_empty() => return Ok(TreeNeurons::default()),
        Some(images) => format!("AND o.image_rel_path IN ({})", sql_string_in_list(images)),
        None => String::new(),
    };
    let sql = format!(
        "WITH s AS (
            SELECT CAST(o.parent_ids[{soma}] AS VARCHAR) AS soma, count(*) AS n
            FROM objects o JOIN images i ON i.image_rel_path = o.image_rel_path
            WHERE list_contains(o.object_class_id, {segment}) AND o.parent_ids[{soma}] IS NOT NULL
                AND NOT i.disabled AND i.successful {image_filter}
            GROUP BY 1
         )
         SELECT s.soma, o.image_rel_path, o.image_name, s.n,
             o.bbox_xmin_px, o.bbox_ymin_px, o.bbox_xmax_px, o.bbox_ymax_px
         FROM s JOIN objects o ON o.object_id::VARCHAR = s.soma
         ORDER BY o.image_name, o.image_rel_path, o.bbox_ymin_px, o.bbox_xmin_px, s.soma"
    );
    let mut stmt = db.connection().prepare(&sql).map_err(err)?;
    let mut neurons: Vec<TreeNeuron> = stmt
        .query_map([], |row| {
            Ok(TreeNeuron {
                soma_id: row.get(0)?,
                image_rel_path: row.get(1)?,
                image_name: row.get(2)?,
                segments: row.get::<_, i64>(3)? as u64,
                soma_bbox_px: [row.get(4)?, row.get(5)?, row.get(6)?, row.get(7)?],
                values: Vec::new(),
            })
        })
        .map_err(err)?
        .collect::<Result<_, _>>()
        .map_err(err)?;
    let keys: Vec<Option<String>> = TREE_NEURON_VALUES
        .iter()
        .map(|(id, _)| tree_key(db, id))
        .collect::<Result<_, _>>()?;
    let measured = keys.iter().any(Option::is_some);
    let present: Vec<String> = keys.iter().flatten().cloned().collect();
    let ids: Vec<String> = neurons.iter().map(|n| n.soma_id.clone()).collect();
    let values = object_values(db, &present, &ids)?;
    for neuron in &mut neurons {
        let own = values.get(&neuron.soma_id);
        neuron.values = keys
            .iter()
            .map(|key| {
                key.as_ref()
                    .and_then(|k| own.and_then(|v| v.get(k)).copied())
            })
            .collect();
    }
    Ok(TreeNeurons { neurons, measured })
}

/// The segments of the neuron with soma `soma_id`, by object id.
pub(super) fn tree_segments(
    db: &ResultsGenerator,
    soma_id: &str,
) -> Result<Vec<TreeSegment>, InternalErrors> {
    let Some((segment, soma)) = tree_classes(db)? else {
        return Ok(Vec::new());
    };
    let sql = format!(
        "SELECT object_id::VARCHAR, CAST(parent_ids[{segment}] AS VARCHAR), area_px,
             bbox_xmin_px, bbox_ymin_px, bbox_xmax_px, bbox_ymax_px
         FROM objects
         WHERE list_contains(object_class_id, {segment})
             AND parent_ids[{soma}] = TRY_CAST(? AS UUID)
         ORDER BY object_id"
    );
    let mut stmt = db.connection().prepare(&sql).map_err(err)?;
    let mut segments: Vec<TreeSegment> = stmt
        .query_map([soma_id], |row| {
            Ok(TreeSegment {
                object_id: row.get(0)?,
                parent: row.get(1)?,
                length: row.get::<_, Option<i64>>(2)?.unwrap_or(0) as f64,
                bbox_px: [row.get(3)?, row.get(4)?, row.get(5)?, row.get(6)?],
                ..Default::default()
            })
        })
        .map_err(err)?
        .collect::<Result<_, _>>()
        .map_err(err)?;
    let keys: Vec<Option<String>> = SEGMENT_KEYS
        .iter()
        .map(|id| tree_key(db, id))
        .collect::<Result<_, _>>()?;
    let present: Vec<String> = keys.iter().flatten().cloned().collect();
    let ids: Vec<String> = segments.iter().map(|s| s.object_id.clone()).collect();
    let values = object_values(db, &present, &ids)?;
    for s in &mut segments {
        let own = values.get(&s.object_id);
        let get = |i: usize| {
            keys[i]
                .as_ref()
                .and_then(|k| own.and_then(|v| v.get(k)).copied())
        };
        if let Some(length) = get(0) {
            s.length = length;
        }
        s.path_distance = get(1);
        s.branch_level = get(2);
        s.strahler_order = get(3);
        s.straightness = get(4);
    }
    Ok(segments)
}

#[cfg(test)]
mod tests {
    use super::super::test_support::{ObjectSpec, seed_db};
    use super::*;

    /// Two neurons: soma 0 (class 1) with a root segment 2 and its two
    /// children 3, 4 (class 3); soma 1 with one segment 5. Measured: the
    /// first neuron's total length, segment 3's branch level.
    fn database(measured: bool) -> ResultsGenerator {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("results.evadb");
        let mut specs = vec![
            ObjectSpec::new("A01_1.tif", "Soma", 1, 100),
            ObjectSpec::new("B01_1.tif", "Soma", 1, 100),
        ];
        for _ in 0..4 {
            specs.push(ObjectSpec::new("A01_1.tif", "Segment", 3, 20));
        }
        specs[5] = ObjectSpec::new("B01_1.tif", "Segment", 3, 30);
        seed_db(&path, &specs);
        std::mem::forget(dir);
        let db = ResultsGenerator::open_database(path).unwrap();
        let id = |i: usize| format!("00000000-0000-0000-0000-{i:012}");
        let mut sql = String::new();
        for (child, parents) in [
            (2, format!("MAP {{1: '{}'::UUID}}", id(0))),
            (
                3,
                format!("MAP {{1: '{}'::UUID, 3: '{}'::UUID}}", id(0), id(2)),
            ),
            (
                4,
                format!("MAP {{1: '{}'::UUID, 3: '{}'::UUID}}", id(0), id(2)),
            ),
            (5, format!("MAP {{1: '{}'::UUID}}", id(1))),
        ] {
            sql.push_str(&format!(
                "UPDATE objects SET parent_ids = {parents} WHERE object_id = '{}';",
                id(child)
            ));
        }
        if measured {
            sql.push_str(&format!(
                "CREATE TABLE measurement_info (key VARCHAR PRIMARY KEY, measurement VARCHAR, \
                 step VARCHAR, unit VARCHAR, target VARCHAR, range_min DOUBLE, range_max DOUBLE, \
                 aggregatable BOOLEAN, object_index INTEGER);
                 CREATE TABLE measurements (image_name VARCHAR, image_rel_path VARCHAR, \
                 z_stack INTEGER, t_stack INTEGER, region VARCHAR, object_id UUID, \
                 related_object_id UUID, object_class_name VARCHAR, object_class_id INTEGER[], \
                 step VARCHAR, measurement VARCHAR, unit VARCHAR, key VARCHAR, value DOUBLE);
                 INSERT INTO measurement_info VALUES
                     ('tree_01/total_length', 'Neuron total length', 'tree_01', 'Pixels', 'Object', NULL, NULL, true, 1),
                     ('tree_01/branch_level', 'Branch level', 'tree_01', 'Count', 'Object', NULL, NULL, true, 2);
                 INSERT INTO measurements VALUES
                     ('A01_1.tif', 'A01_1.tif', 0, 0, 'object', '{}', NULL, NULL, [1], 'tree_01', 'x', 'Pixels', 'tree_01/total_length', 60),
                     ('A01_1.tif', 'A01_1.tif', 0, 0, 'object', '{}', NULL, NULL, [3], 'tree_01', 'x', 'Count', 'tree_01/branch_level', 2);",
                id(0),
                id(3)
            ));
        }
        db.connection().execute_batch(&sql).unwrap();
        db
    }

    #[test]
    fn the_tree_classes_are_read_off_the_parents() {
        assert_eq!(tree_classes(&database(false)).unwrap(), Some((3, 1)));
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("results.evadb");
        seed_db(&path, &[ObjectSpec::new("a.tif", "A", 1, 10)]);
        let plain = ResultsGenerator::open_database(path).unwrap();
        assert!(!has_trees(&plain).unwrap());
    }

    #[test]
    fn the_neurons_with_their_values() {
        let neurons = tree_neurons(&database(true), None).unwrap();
        assert!(neurons.measured);
        let rows: Vec<(String, u64, Option<f64>)> = neurons
            .neurons
            .iter()
            .map(|n| (n.image_name.clone(), n.segments, n.values[0]))
            .collect();
        assert_eq!(
            rows,
            [
                ("A01_1.tif".to_string(), 3, Some(60.0)),
                ("B01_1.tif".to_string(), 1, None)
            ]
        );
        let only_b = tree_neurons(&database(true), Some(&["B01_1.tif".to_string()])).unwrap();
        assert_eq!(only_b.neurons.len(), 1);
    }

    #[test]
    fn a_neurons_segments_with_their_parents() {
        let db = database(true);
        let soma = "00000000-0000-0000-0000-000000000000";
        let segments = tree_segments(&db, soma).unwrap();
        let parents: Vec<Option<String>> = segments.iter().map(|s| s.parent.clone()).collect();
        let root = "00000000-0000-0000-0000-000000000002".to_string();
        assert_eq!(parents, [None, Some(root.clone()), Some(root)]);
        assert_eq!(segments[1].branch_level, Some(2.0));
        // Not measured: the length is the pixel count.
        assert_eq!(segments[0].length, 20.0);
    }

    #[test]
    fn traced_but_not_measured() {
        let neurons = tree_neurons(&database(false), None).unwrap();
        assert!(!neurons.measured);
        assert_eq!(neurons.neurons.len(), 2);
        assert!(neurons.neurons[0].values.iter().all(Option::is_none));
    }
}
