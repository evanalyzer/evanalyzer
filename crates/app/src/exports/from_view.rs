//! # from_view
//!
//! A Results view as an export entry: what "Export this view" writes and
//! what "Save as export…" keeps - a List as a table, a Plate as a plate
//! layout, Tracks as a table of tracks, a chart as the table behind it (a
//! time course, a profile, the values it shows), Paths as their tracks,
//! Trees as the measurements of the traced neurons.
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

use crate::exports::jobs::{PlateGrid, PlateLayoutExport, TableExport, TableRows};
use crate::results::view_settings::ViewSettings;
use crate::results::{Aggregation, ProfileLines, TimeCourseLines, TrackHistory};
use evanalyzer_cfg::settings::export_settings::{
    ChildLink, DataExport, DataOutput, ExportAggregation, ExportEntry, HeatmapColors,
    HeatmapOptions, ImageScope, ProfileLineKind, ProfileOptions, TimeCourseLineKind,
    TimeCourseOptions,
};
use evanalyzer_cfg::settings::project_settings::{ResultsView, ResultsViewKind};

/// `view` as an export entry - `None` for a kind exports don't write.
pub fn entry_from_view(view: &ResultsView) -> Option<ExportEntry> {
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
            let mut aggregations: Vec<ExportAggregation> =
                list.aggregations.iter().map(export_aggregation).collect();
            aggregations.dedup();
            if aggregations.is_empty() {
                aggregations.push(ExportAggregation::Avg);
            }
            let mut table = table(rows);
            table.classes = list.classes.clone();
            // Its own columns, and the children and linked objects as it
            // shows them - the runner adds their columns the same way.
            table.columns = list.columns.clone();
            table.children = list.children.clone();
            table.child_aggregations = list
                .child_aggregations
                .iter()
                .map(export_aggregation)
                .collect();
            table.parents = list.parents.clone();
            table.aggregations = aggregations;
            table.empty_classes = list.empty_classes;
            table.transposed = list.transposed
                && matches!(
                    rows,
                    TableRows::Object | TableRows::Image | TableRows::Well | TableRows::Folder
                );
            table.images = match list.images.as_str() {
                "all" => ImageScope::All,
                "disabled" => ImageScope::LeftOut,
                _ => ImageScope::Included,
            };
            Some(ExportEntry::Data(data_of_table(table)))
        }
        ResultsViewKind::Plate => {
            let plate = &settings.plate;
            Some(ExportEntry::Data(data_of_plate(PlateLayoutExport {
                name: String::new(),
                grid: PlateGrid::Plate,
                classes: plate.class.map(|class| vec![class]),
                columns: plate.column.clone().map(|column| vec![column]),
                aggregations: vec![export_aggregation(&plate.aggregation)],
                images: ImageScope::Included,
                only_images: None,
                group_regex: None,
                square_size: None,
                colors: colors(&plate.color_schema),
                planes: Default::default(),
            })))
        }
        ResultsViewKind::Tracks => {
            let mut table = table(TableRows::Track);
            table.columns = settings.tracks.columns.clone();
            Some(ExportEntry::Data(data_of_table(table)))
        }
        ResultsViewKind::Charts => {
            let chart = &settings.chart;
            let columns = |keys: Vec<Option<String>>| {
                let keys: Vec<String> = keys.into_iter().flatten().collect();
                (!keys.is_empty()).then_some(keys)
            };
            let classes = chart.class.map(|class| vec![class]);
            let mut table = match chart.kind.as_str() {
                "overTime" => {
                    let mut table = table(TableRows::TimePoint);
                    table.aggregations = if chart.time_median {
                        vec![ExportAggregation::Median]
                    } else {
                        vec![ExportAggregation::Avg, ExportAggregation::Stddev]
                    };
                    table.time_course = Some(TimeCourseOptions {
                        lines: match chart.time_lines {
                            TimeCourseLines::Class => TimeCourseLineKind::Class,
                            TimeCourseLines::Image => TimeCourseLineKind::Image,
                            TimeCourseLines::Well | TimeCourseLines::Group => {
                                TimeCourseLineKind::Well
                            }
                            TimeCourseLines::Track => TimeCourseLineKind::Track,
                        },
                        layout: Default::default(),
                        relative: chart.time_relative,
                        with_parents: chart.time_history == TrackHistory::WithParents,
                    });
                    table.columns = columns(vec![chart.column.clone()]);
                    table
                }
                "profile" => {
                    let mut table = table(TableRows::ProfilePosition);
                    table.profile = Some(ProfileOptions {
                        key: chart.profile_key.clone().unwrap_or_default(),
                        lines: match chart.profile_lines {
                            ProfileLines::Class => ProfileLineKind::Class,
                            ProfileLines::Image => ProfileLineKind::Image,
                            ProfileLines::Well => ProfileLineKind::Well,
                            ProfileLines::All => ProfileLineKind::All,
                        },
                        content: Default::default(),
                    });
                    table
                }
                // A boxplot: its values per image (each image a dot) or per
                // object.
                "boxplot" if chart.per_image => {
                    let mut table = table(TableRows::Image);
                    table.columns = columns(vec![chart.column.clone()]);
                    table
                }
                // A histogram, a scatter, a boxplot of objects: the objects'
                // values.
                _ => {
                    let mut table = table(TableRows::Object);
                    table.columns = columns(vec![chart.column.clone(), chart.y_column.clone()]);
                    table
                }
            };
            table.classes = classes;
            table.images = ImageScope::All;
            Some(ExportEntry::Data(data_of_table(table)))
        }
        // The tracks the paths are of.
        ResultsViewKind::Paths => Some(ExportEntry::Data(data_of_table(table(TableRows::Track)))),
        // The traced neurons' values are measurements (Measure Filament Tree).
        ResultsViewKind::Trees => Some(ExportEntry::Data(data_of_table(table(
            TableRows::Measurement,
        )))),
    }
}

/// A data export of `table`'s one output, with its filter.
fn data_of_table(table: TableExport) -> DataExport {
    let output = match table.rows {
        TableRows::Object => DataOutput::Objects,
        TableRows::Image => DataOutput::Images,
        TableRows::Well => DataOutput::Wells,
        TableRows::Folder => DataOutput::Folders,
        TableRows::ColocPair => DataOutput::ColocPairs,
        TableRows::Measurement => DataOutput::Measurements,
        TableRows::Track => DataOutput::Tracks,
        TableRows::TimePoint => DataOutput::TimeCourse,
        TableRows::ProfilePosition => DataOutput::Profile,
    };
    DataExport {
        name: String::new(),
        outputs: vec![output],
        format: table.format,
        classes: table.classes,
        columns: table.columns,
        aggregations: table.aggregations,
        images: table.images,
        only_images: table.only_images,
        group_regex: table.group_regex,
        planes: table.planes,
        empty_classes: table.empty_classes,
        transposed: table.transposed,
        children: table
            .children
            .iter()
            .map(|(parent, child)| ChildLink {
                parent: *parent,
                child: *child,
            })
            .collect(),
        child_aggregations: table.child_aggregations,
        parents: table.parents,
        split: table.split,
        split_into: table.split_into,
        tracks: table.tracks,
        time_course: table.time_course,
        profile: table.profile,
        heatmap: Default::default(),
    }
}

/// A data export of `plate`'s one output, with its filter.
fn data_of_plate(plate: PlateLayoutExport) -> DataExport {
    let mut data = data_of_table(table(TableRows::Image));
    data.outputs = vec![match plate.grid {
        PlateGrid::Plate => DataOutput::Plate,
        PlateGrid::Well => DataOutput::WellGrid,
        PlateGrid::WellList => DataOutput::WellList,
        PlateGrid::Image => DataOutput::ImageHeatmap,
    }];
    data.classes = plate.classes;
    data.columns = plate.columns;
    data.aggregations = plate.aggregations;
    data.images = plate.images;
    data.only_images = plate.only_images;
    data.group_regex = plate.group_regex;
    data.planes = plate.planes;
    data.heatmap = HeatmapOptions {
        colors: plate.colors,
        square_size: plate.square_size,
    };
    data
}

/// A table of `rows` with its defaults.
fn table(rows: TableRows) -> TableExport {
    let rows = serde_json::to_value(rows).expect("rows as JSON");
    serde_json::from_value(serde_json::json!({ "rows": rows })).expect("a table with defaults")
}

fn colors(schema: &crate::results::ColorSchema) -> HeatmapColors {
    use crate::results::ColorSchema;
    match schema {
        ColorSchema::Excel => HeatmapColors::Excel,
        ColorSchema::Viridis => HeatmapColors::Viridis,
        ColorSchema::Plasma => HeatmapColors::Plasma,
        ColorSchema::Inferno => HeatmapColors::Inferno,
        ColorSchema::Cividis => HeatmapColors::Cividis,
        ColorSchema::Coolwarm => HeatmapColors::Coolwarm,
        ColorSchema::RedBlue => HeatmapColors::RedBlue,
        ColorSchema::YlGnBu => HeatmapColors::YlGnBu,
        ColorSchema::Haline => HeatmapColors::Haline,
        ColorSchema::Algae => HeatmapColors::Algae,
        ColorSchema::Thermal => HeatmapColors::Thermal,
    }
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
        let Some(ExportEntry::Data(data)) = entry else {
            panic!("a data export");
        };
        match crate::exports::jobs::jobs_of(&data).remove(0).1 {
            crate::exports::jobs::Job::Table(table) => table,
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
        let table = table_of(entry_from_view(&list));
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

    /// The children and linked objects as the List shows them - kept as
    /// such, not spelled out.
    #[test]
    fn a_lists_children_and_linked_objects_are_kept() {
        let list = view(
            "list",
            json!({ "list": {
                "columns": ["area_px"], "children": [[2, 1]], "childAggregations": ["Max"],
                "parents": [3]
            }}),
        );
        let table = table_of(entry_from_view(&list));
        assert_eq!(table.columns, Some(vec!["area_px".to_string()]));
        assert_eq!(table.children, vec![(2, 1)]);
        assert_eq!(table.child_aggregations, vec![ExportAggregation::Max]);
        assert_eq!(table.parents, vec![3]);
    }

    /// Rows per well, per folder, coloc pairs; the default columns stay
    /// the default ones; transposed only where a table can be.
    #[test]
    fn the_rows_follow_the_list() {
        let rows = |settings: serde_json::Value| {
            table_of(entry_from_view(&view("list", json!({ "list": settings }))))
        };
        let well = rows(json!({ "rowsPerGroup": "wells", "transposed": true }));
        assert_eq!(well.rows, TableRows::Well);
        assert!(well.transposed, "wells side by side as in the List");
        assert!(!rows(json!({ "colocDetails": true, "transposed": true })).transposed);
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
        let Some(ExportEntry::Data(data)) = entry_from_view(&plate) else {
            panic!("a data export");
        };
        let crate::exports::jobs::Job::PlateLayout(layout) =
            crate::exports::jobs::jobs_of(&data).remove(0).1
        else {
            panic!("a plate layout");
        };
        assert_eq!(layout.classes, Some(vec![3]));
        assert_eq!(layout.columns, Some(vec!["area_px".to_string()]));
        assert_eq!(layout.aggregations, vec![ExportAggregation::Median]);

        let tracks = view("tracks", json!({ "tracks": { "columns": ["vcl"] } }));
        let table = table_of(entry_from_view(&tracks));
        assert_eq!(table.rows, TableRows::Track);
        assert_eq!(table.columns, Some(vec!["vcl".to_string()]));

        // A chart: the table behind it.
        let over_time = view(
            "charts",
            json!({ "chart": { "kind": "overTime", "column": "area_px", "timeLines": "Track",
                               "timeRelative": true } }),
        );
        let table = table_of(entry_from_view(&over_time));
        assert_eq!(table.rows, TableRows::TimePoint);
        assert_eq!(table.columns, Some(vec!["area_px".to_string()]));
        let course = table.time_course.unwrap();
        assert_eq!(course.lines, TimeCourseLineKind::Track);
        assert!(course.relative);
        let scatter = view(
            "charts",
            json!({ "chart": { "kind": "scatter", "column": "area_px", "yColumn": "circularity" } }),
        );
        let table = table_of(entry_from_view(&scatter));
        assert_eq!(table.rows, TableRows::Object);
        assert_eq!(
            table.columns,
            Some(vec!["area_px".to_string(), "circularity".to_string()])
        );
        assert_eq!(
            table_of(entry_from_view(&view("paths", json!({})))).rows,
            TableRows::Track
        );
    }
}
