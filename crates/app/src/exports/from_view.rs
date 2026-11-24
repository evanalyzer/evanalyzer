//! # from_view
//!
//! A Results view as an export entry: what "Export this view" writes and
//! what "Save as export…" keeps - a List as a table, a Plate as a plate
//! layout, Tracks as a table of tracks. Charts, Paths and Trees have no
//! entry (exports write tables, images and script results).
//!
//! The entry is a copy of what the view shows now: later changes to the
//! view don't change it.
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-11-24
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.

use crate::results::view_settings::{
    DEFAULT_LIST_COLUMNS, ViewSettings, resolve_columns, with_related_columns,
};
use crate::results::{Aggregation, ColumnEntry};
use evanalyzer_cfg::core_types::ObjectClass;
use evanalyzer_cfg::settings::classification_settings::Class;
use evanalyzer_cfg::settings::export_settings::{
    ExportAggregation, ExportEntry, ImageScope, PlateGrid, PlateLayoutExport, TableExport,
    TableRows,
};
use evanalyzer_cfg::settings::project_settings::{ResultsView, ResultsViewKind};

/// `view` as an export entry - `None` for a kind exports don't write.
/// `classes` and `available` are the open results' (to name the related
/// columns a List shows besides its own).
pub fn entry_from_view(
    view: &ResultsView,
    classes: &[Class],
    available: &[ColumnEntry],
) -> Option<ExportEntry> {
    let settings = ViewSettings::read(&view.settings);
    match view.kind {
        ResultsViewKind::List => {
            let list = &settings.list;
            let rows = if list.coloc_details {
                TableRows::ColocPair
            } else {
                match list.rows_per_group.as_str() {
                    "wells" => TableRows::Well,
                    "folders" => TableRows::Folder,
                    _ if list.rows_per_image => TableRows::Image,
                    _ => TableRows::Object,
                }
            };
            let related = !list.children.is_empty() || !list.parents.is_empty();
            let columns = match (&list.columns, related) {
                (None, false) => None,
                (keys, _) => {
                    let keys: Vec<String> = match keys {
                        Some(keys) => keys.clone(),
                        None => DEFAULT_LIST_COLUMNS
                            .iter()
                            .map(|column| column.as_key(classes))
                            .collect(),
                    };
                    let (base, missing) = resolve_columns(&keys, classes, available);
                    let children: Vec<(ObjectClass, ObjectClass)> = list
                        .children
                        .iter()
                        .map(|(p, c)| (ObjectClass::Global(*p), ObjectClass::Global(*c)))
                        .collect();
                    let parents: Vec<ObjectClass> = list
                        .parents
                        .iter()
                        .map(|p| ObjectClass::Global(*p))
                        .collect();
                    let all =
                        with_related_columns(&base, &children, &list.child_aggregations, &parents);
                    let mut keys: Vec<String> =
                        all.iter().map(|column| column.as_key(classes)).collect();
                    // Kept: the results may lack a column the view names.
                    keys.extend(missing);
                    Some(keys)
                }
            };
            let mut aggregations: Vec<ExportAggregation> =
                list.aggregations.iter().map(export_aggregation).collect();
            aggregations.dedup();
            if aggregations.is_empty() {
                aggregations.push(ExportAggregation::Avg);
            }
            let mut table = table(rows);
            table.classes = list.classes.clone();
            table.columns = columns;
            table.aggregations = aggregations;
            table.empty_classes = list.empty_classes;
            table.transposed =
                list.transposed && matches!(rows, TableRows::Object | TableRows::Image);
            table.images = match list.images.as_str() {
                "all" => ImageScope::All,
                "disabled" => ImageScope::LeftOut,
                _ => ImageScope::Included,
            };
            Some(ExportEntry::Table(table))
        }
        ResultsViewKind::Plate => {
            let plate = &settings.plate;
            Some(ExportEntry::PlateLayout(PlateLayoutExport {
                name: String::new(),
                grid: PlateGrid::Plate,
                classes: plate.class.map(|class| vec![class]),
                columns: plate.column.clone().map(|column| vec![column]),
                aggregation: export_aggregation(&plate.aggregation),
                images: ImageScope::Included,
            }))
        }
        ResultsViewKind::Tracks => {
            let mut table = table(TableRows::Track);
            table.columns = settings.tracks.columns.clone();
            Some(ExportEntry::Table(table))
        }
        ResultsViewKind::Charts | ResultsViewKind::Paths | ResultsViewKind::Trees => None,
    }
}

/// A table of `rows` with its defaults.
fn table(rows: TableRows) -> TableExport {
    let rows = serde_json::to_value(rows).expect("rows as JSON");
    serde_json::from_value(serde_json::json!({ "rows": rows })).expect("a table with defaults")
}

fn export_aggregation(aggregation: &Aggregation) -> ExportAggregation {
    match aggregation {
        Aggregation::Avg => ExportAggregation::Avg,
        Aggregation::Min => ExportAggregation::Min,
        Aggregation::Max => ExportAggregation::Max,
        Aggregation::Stddev => ExportAggregation::Stddev,
        Aggregation::Sum => ExportAggregation::Sum,
        Aggregation::Median => ExportAggregation::Median,
        Aggregation::Skewness => ExportAggregation::Skewness,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn view(kind: &str, settings: serde_json::Value) -> ResultsView {
        serde_json::from_value(json!({ "id": 1, "name": "V", "kind": kind, "settings": settings }))
            .unwrap()
    }

    fn table_of(entry: Option<ExportEntry>) -> TableExport {
        match entry {
            Some(ExportEntry::Table(table)) => table,
            _ => panic!("a table"),
        }
    }

    /// A List per image: its classes, columns, values, images, side by side.
    #[test]
    fn a_list_per_image_is_a_table_of_images() {
        let list = view(
            "list",
            json!({ "list": {
                "classes": [2], "columns": ["count", "area_px"], "aggregations": ["Max", "Avg"],
                "rowsPerImage": true, "emptyClasses": true, "transposed": true, "images": "all"
            }}),
        );
        let table = table_of(entry_from_view(&list, &[], &[]));
        assert_eq!(table.rows, TableRows::Image);
        assert_eq!(table.classes, Some(vec![2]));
        assert_eq!(
            table.columns,
            Some(vec!["count".to_string(), "area_px".to_string()])
        );
        assert_eq!(
            table.aggregations,
            vec![ExportAggregation::Max, ExportAggregation::Avg]
        );
        assert!(table.empty_classes && table.transposed);
        assert_eq!(table.images, ImageScope::All);
    }

    /// Rows per well, per folder, coloc pairs; the default columns stay
    /// the default ones; transposed only where a table can be.
    #[test]
    fn the_rows_follow_the_list() {
        let rows = |settings: serde_json::Value| {
            table_of(entry_from_view(
                &view("list", json!({ "list": settings })),
                &[],
                &[],
            ))
        };
        let well = rows(json!({ "rowsPerGroup": "wells", "transposed": true }));
        assert_eq!(well.rows, TableRows::Well);
        assert!(!well.transposed);
        assert_eq!(well.columns, None);
        assert_eq!(
            rows(json!({ "rowsPerGroup": "folders" })).rows,
            TableRows::Folder
        );
        assert_eq!(
            rows(json!({ "colocDetails": true })).rows,
            TableRows::ColocPair
        );
        assert_eq!(rows(json!({})).rows, TableRows::Object);
        // A List shows every image unless it says otherwise.
        assert_eq!(rows(json!({})).images, ImageScope::All);
        assert_eq!(
            rows(json!({ "images": "enabled" })).images,
            ImageScope::Included
        );
    }

    /// A Plate is a plate layout of its value; Tracks a table of tracks;
    /// a chart is no entry.
    #[test]
    fn plates_tracks_and_charts() {
        let plate = view(
            "plate",
            json!({ "plate": { "class": 3, "column": "area_px", "aggregation": "Median" } }),
        );
        let Some(ExportEntry::PlateLayout(layout)) = entry_from_view(&plate, &[], &[]) else {
            panic!("a plate layout");
        };
        assert_eq!(layout.classes, Some(vec![3]));
        assert_eq!(layout.columns, Some(vec!["area_px".to_string()]));
        assert_eq!(layout.aggregation, ExportAggregation::Median);

        let tracks = view("tracks", json!({ "tracks": { "columns": ["vcl"] } }));
        let table = table_of(entry_from_view(&tracks, &[], &[]));
        assert_eq!(table.rows, TableRows::Track);
        assert_eq!(table.columns, Some(vec!["vcl".to_string()]));

        assert!(entry_from_view(&view("charts", json!({})), &[], &[]).is_none());
    }
}
