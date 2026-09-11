use std::cell::RefCell;
use std::path::PathBuf;

use duckdb::Connection;
use duckdb::types::Value;
use evanalyzer_cfg::{
    core_types::{InternalErrors, ObjectClass},
    settings::classification_settings::Class,
};

pub struct ResultsGenerator {
    database: Connection,
    classes_cache: RefCell<Option<Vec<Class>>>,
}

#[derive(Clone)]
pub enum View {
    List,
    Heatmap,
}

#[derive(Clone)]
pub enum Aggregation {
    Avg,
    Min,
    Max,
    Stddev,
    Count,
}

#[derive(Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum Column {
    ObjectId,
    ImageName,
    ObjectClass,
    AreaSizePx,
    AreaSizeNm,
    PerimeterPx,
    PerimeterNm,
    Circularity,
    Solidity,
    Eccentricity,
    ColocCount,
    IntensityAvg,
    IntensitySum,
    IntensityMin,
    IntensityMax,
}

impl Column {
    /// Stable string key (matches the underlying database column name) used
    /// to store this variant in UI widgets that only accept strings, e.g. the
    /// slint columns dropdown.
    pub fn as_key(&self) -> &'static str {
        match self {
            Column::ObjectId => "object_id",
            Column::ImageName => "image_name",
            Column::ObjectClass => "object_class_name",
            Column::AreaSizePx => "area_px",
            Column::AreaSizeNm => "area_nm2",
            Column::PerimeterPx => "perimeter_px",
            Column::PerimeterNm => "perimeter_nm",
            Column::Circularity => "circularity",
            Column::Solidity => "solidity",
            Column::Eccentricity => "eccentricity",
            Column::ColocCount => "n_colocalized",
            Column::IntensityAvg => "mean_raw",
            Column::IntensitySum => "sum_raw",
            Column::IntensityMin => "min_raw",
            Column::IntensityMax => "max_raw",
        }
    }

    /// Inverse of [`Column::as_key`].
    pub fn from_key(key: &str) -> Option<Self> {
        Some(match key {
            "object_id" => Column::ObjectId,
            "image_name" => Column::ImageName,
            "object_class_name" => Column::ObjectClass,
            "area_px" => Column::AreaSizePx,
            "area_nm2" => Column::AreaSizeNm,
            "perimeter_px" => Column::PerimeterPx,
            "perimeter_nm" => Column::PerimeterNm,
            "circularity" => Column::Circularity,
            "solidity" => Column::Solidity,
            "eccentricity" => Column::Eccentricity,
            "n_colocalized" => Column::ColocCount,
            "mean_raw" => Column::IntensityAvg,
            "sum_raw" => Column::IntensitySum,
            "min_raw" => Column::IntensityMin,
            "max_raw" => Column::IntensityMax,
            _ => return None,
        })
    }
}

#[derive(Clone)]
pub struct PlaneFilter {
    pub z_stack: u32,
    pub t_stack: u32,
}

#[derive(Clone)]
pub struct Pagination {
    pub limit: i32,
    pub offset: i32,
}

#[derive(Clone)]
pub struct GroupFilter {
    pub plane: PlaneFilter,
    pub grouping_regex: String,
    pub aggregation: Aggregation,
    pub object_class: ObjectClass,
    pub column: Column,
}

#[derive(Clone)]
pub struct ListFilter {
    pub plane: PlaneFilter,
    pub images: Option<Vec<String>>,
    pub object_classes: Option<Vec<ObjectClass>>,
    pub columns: Vec<Column>,
    pub page: Pagination,
}

pub enum Cell {
    String(String),
    Float(f32),
    Integer(i32),
    /// Object class with color
    Class((String, u32)),
}

pub struct ColumnEntry {
    pub display_name: String,
    pub key: Column,
    pub group: String,
}

pub struct ImageEntry {
    pub name: String,
    pub rel_path: PathBuf,
    pub disabled: bool,
}

pub struct DatabaseResult {
    pub column_names: Vec<String>,
    pub row_names: Vec<String>,
    /// One row with its colums
    pub rows: Vec<Vec<Cell>>,
}

impl ResultsGenerator {
    pub fn open_database(path: PathBuf) -> Result<Self, InternalErrors> {
        let to_io_err = |e: duckdb::Error| InternalErrors::Io(e.to_string());
        let database = Connection::open(&path).map_err(to_io_err)?;
        Ok(Self {
            database,
            classes_cache: RefCell::new(None),
        })
    }

    pub fn get_list(
        &self,
        filter: &ListFilter,
        _view: &View,
    ) -> Result<DatabaseResult, InternalErrors> {
        let err = |e: duckdb::Error| InternalErrors::Io(e.to_string());
        let mut ordered_columns = filter.columns.clone();
        ordered_columns.sort();
        let column_names: Vec<String> = ordered_columns
            .iter()
            .map(|c| c.as_key().to_string())
            .collect();
        let empty_result = |column_names: Vec<String>| DatabaseResult {
            column_names,
            row_names: vec![],
            rows: vec![],
        };

        // `ListFilter.images` carries the rel-paths the GUI's image picker
        // keys on, but the `objects` table only has `image_name` to filter
        // on — translate one to the other via the images table. `Some([])`
        // (an active filter matching nothing, whether the user selected
        // zero images or none of their selections resolved to a real image)
        // means zero rows, same convention as the class filter below.
        let image_names = match &filter.images {
            Some(rel_paths) => {
                let images = self.get_images()?;
                let names: Vec<String> = rel_paths
                    .iter()
                    .filter_map(|rel_path| {
                        images
                            .iter()
                            .find(|image| image.rel_path.to_str() == Some(rel_path.as_str()))
                            .map(|image| image.name.clone())
                    })
                    .collect();
                if names.is_empty() {
                    return Ok(empty_result(column_names.clone()));
                }
                Some(names)
            }
            None => None,
        };

        // `ListFilter.object_classes` already carries `ObjectClass` ids, but
        // `get_object_classes()` (cached — see `classes_cache`) still doubles
        // as validation: an id that no longer names a registered class (e.g.
        // stale GUI state after switching databases) is dropped rather than
        // matched against `object_class_id` blindly.
        let classes = self.get_object_classes()?;
        let class_ids = match &filter.object_classes {
            Some(wanted) => {
                let ids: Vec<u32> = wanted
                    .iter()
                    .filter_map(|id| classes.iter().find(|class| &class.id == id))
                    .filter_map(|class| match class.id {
                        ObjectClass::Valid(n) => Some(n),
                        ObjectClass::Unset => None,
                    })
                    .collect();
                if ids.is_empty() {
                    return Ok(empty_result(column_names.clone()));
                }
                Some(ids)
            }
            None => None,
        };

        let mut conditions = vec![
            format!("z_stack = {}", filter.plane.z_stack),
            format!("t_stack = {}", filter.plane.t_stack),
        ];
        if let Some(names) = &image_names {
            conditions.push(format!("image_rel_path IN ({})", sql_string_in_list(names)));
        }
        if let Some(ids) = &class_ids {
            conditions.push(format!(
                "list_has_any(CAST(object_class_id AS INTEGER[]), {})",
                sql_int_array_literal(ids)
            ));
        }
        let where_clause = format!("WHERE {}", conditions.join(" AND "));

        let limit = filter.page.limit.max(0);
        let offset = filter.page.offset.max(0);
        let sql = format!(
            "SELECT object_id, image_name, CAST(object_class_name AS VARCHAR[]), seg_class_name,\n\
                    area_px, area_nm2, perimeter_px, perimeter_nm,\n\
                    circularity, solidity, eccentricity,\n\
                    coloc_json, intensities_json\n\
             FROM objects {where_clause}\n\
             ORDER BY object_id\n\
             LIMIT {limit} OFFSET {offset}"
        );

        let mut stmt = self.database.prepare(&sql).map_err(err)?;
        let objects = stmt
            .query_map([], |row| {
                Ok(ObjectRow {
                    object_id: row.get(0)?,
                    image_name: row.get(1)?,
                    object_class_name: extract_string_list(row.get::<_, Value>(2)?),
                    seg_class_name: row.get(3)?,
                    area_px: row.get(4)?,
                    area_nm2: row.get(5)?,
                    perimeter_px: row.get(6)?,
                    perimeter_nm: row.get(7)?,
                    circularity: row.get(8)?,
                    solidity: row.get(9)?,
                    eccentricity: row.get(10)?,
                    coloc_json: row.get::<_, Option<String>>(11)?.unwrap_or_default(),
                    intensities_json: row.get::<_, Option<String>>(12)?.unwrap_or_default(),
                })
            })
            .map_err(err)?
            .collect::<Result<Vec<_>, _>>()
            .map_err(err)?;

        let row_names = objects
            .iter()
            .map(|object| object.object_id.clone())
            .collect();
        let rows = objects
            .iter()
            .map(|object| {
                ordered_columns
                    .iter()
                    .map(|column| cell_for_column(column, object, &classes))
                    .collect()
            })
            .collect();

        Ok(DatabaseResult {
            column_names,
            row_names,
            rows,
        })
    }
    pub fn get_group_by_well(
        &self,
        filter: &GroupFilter,
        view: &View,
    ) -> Result<DatabaseResult, InternalErrors> {
        Err(("not implemented").into())
    }
    pub fn get_group_by_plate(
        &self,
        filter: &GroupFilter,
        view: &View,
    ) -> Result<DatabaseResult, InternalErrors> {
        Err(("not implemented").into())
    }
    pub fn get_coloc_objects(&self) {}

    pub fn get_histogram(&self) {}
    pub fn get_scatter(&self) {}
    pub fn get_heatmap(&self) {}

    pub fn get_images(&self) -> Result<Vec<ImageEntry>, InternalErrors> {
        let err = |e: duckdb::Error| InternalErrors::Io(e.to_string());
        let mut stmt = self
            .database
            .prepare("SELECT image_name, image_rel_path, disabled FROM images ORDER BY image_name")
            .map_err(err)?;
        let map = stmt
            .query_map([], |row| {
                Ok(ImageEntry {
                    name: row.get(0)?,
                    rel_path: PathBuf::from(row.get::<_, String>(1)?),
                    disabled: row.get(2)?,
                })
            })
            .map_err(err)?
            .collect::<Result<Vec<_>, _>>()
            .map_err(err);
        map
    }

    /// Snapshot of the `classes` table. Cached after the first call for this
    /// opened database — see `classes_cache` — since it's read on essentially
    /// every `get_list` call (both to translate the class filter and to
    /// resolve display colors) but the registry itself only ever changes by
    /// opening a different database.
    pub fn get_object_classes(&self) -> Result<Vec<Class>, InternalErrors> {
        if let Some(cached) = self.classes_cache.borrow().as_ref() {
            return Ok(cached.clone());
        }

        let err = |e: duckdb::Error| InternalErrors::Io(e.to_string());
        let mut stmt = self
            .database
            .prepare("SELECT class_id, name, color FROM classes ORDER BY class_id")
            .map_err(err)?;
        let classes = stmt
            .query_map([], |row| {
                let class_id: u32 = row.get(0)?;
                let color: Option<u32> = row.get(2)?;
                Ok(Class {
                    id: ObjectClass::Valid(class_id),
                    name: row.get(1)?,
                    color: color.unwrap_or(0),
                    notes: String::new(),
                })
            })
            .map_err(err)?
            .collect::<Result<Vec<_>, _>>()
            .map_err(err)?;

        *self.classes_cache.borrow_mut() = Some(classes.clone());
        Ok(classes)
    }

    pub fn get_available_columns(&self) -> Result<Vec<ColumnEntry>, InternalErrors> {
        let ret = vec![
            ColumnEntry {
                display_name: "Object ID".into(),
                key: Column::ObjectId,
                group: "General".into(),
            },
            ColumnEntry {
                display_name: "Image".into(),
                key: Column::ImageName,
                group: "General".into(),
            },
            ColumnEntry {
                display_name: "Class".into(),
                key: Column::ObjectClass,
                group: "General".into(),
            },
            ColumnEntry {
                display_name: "Area [px]".into(),
                key: Column::AreaSizePx,
                group: "Geometry".into(),
            },
            ColumnEntry {
                display_name: "Area [nm²]".into(),
                key: Column::AreaSizeNm,
                group: "Geometry".into(),
            },
            ColumnEntry {
                display_name: "Perimeter [px]".into(),
                key: Column::PerimeterPx,
                group: "Geometry".into(),
            },
            ColumnEntry {
                display_name: "Perimeter [nm]".into(),
                key: Column::PerimeterNm,
                group: "Geometry".into(),
            },
            ColumnEntry {
                display_name: "Circularity".into(),
                key: Column::Circularity,
                group: "Shape".into(),
            },
            ColumnEntry {
                display_name: "Solidity".into(),
                key: Column::Solidity,
                group: "Shape".into(),
            },
            ColumnEntry {
                display_name: "Eccentricity".into(),
                key: Column::Eccentricity,
                group: "Shape".into(),
            },
            ColumnEntry {
                display_name: "Coloc count".into(),
                key: Column::ColocCount,
                group: "Coloc".into(),
            },
            ColumnEntry {
                display_name: "Avg Intensity".into(),
                key: Column::IntensityAvg,
                group: "intensity".into(),
            },
            ColumnEntry {
                display_name: "Sum Intensity".into(),
                key: Column::IntensitySum,
                group: "intensity".into(),
            },
            ColumnEntry {
                display_name: "Min Intensity".into(),
                key: Column::IntensityMin,
                group: "intensity".into(),
            },
            ColumnEntry {
                display_name: "Max Intensity".into(),
                key: Column::IntensityMax,
                group: "intensity".into(),
            },
        ];

        Ok(ret)
    }

    pub fn get_nr_of_c_stacks(&self) -> u32 {
        let max_stack: u32 = self
            .database
            .query_row("SELECT MAX(c_stack) FROM objects;", [], |row| row.get(0))
            .unwrap_or(1);
        max_stack
    }

    pub fn get_nr_of_z_stacks(&self) -> u32 {
        let max_stack: u32 = self
            .database
            .query_row("SELECT MAX(z_stack) FROM objects;", [], |row| row.get(0))
            .unwrap_or(1);
        max_stack
    }

    pub fn get_nr_of_t_stacks(&self) -> u32 {
        let max_stack: u32 = self
            .database
            .query_row("SELECT MAX(t_stack) FROM objects;", [], |row| row.get(0))
            .unwrap_or(1);
        max_stack
    }
}

/// One row of the `objects` table, as fetched by `get_list`'s hand-written
/// SQL — only the columns needed to fill in any `Column` variant (see
/// `cell_for_column`), not every column the table has.
struct ObjectRow {
    object_id: String,
    image_name: String,
    object_class_name: Vec<String>,
    seg_class_name: Option<String>,
    area_px: u64,
    area_nm2: f64,
    perimeter_px: f64,
    perimeter_nm: f64,
    circularity: f64,
    solidity: f64,
    eccentricity: f64,
    coloc_json: String,
    intensities_json: String,
}

/// Escapes and comma-joins string literals for a SQL `IN (...)` list.
fn sql_string_in_list(values: &[String]) -> String {
    values
        .iter()
        .map(|v| format!("'{}'", v.replace('\'', "''")))
        .collect::<Vec<_>>()
        .join(", ")
}

/// Comma-joins integers into a DuckDB list literal, e.g. `[1, 2]`, for
/// `list_has_any(...)`.
fn sql_int_array_literal(values: &[u32]) -> String {
    format!(
        "[{}]",
        values
            .iter()
            .map(|v| v.to_string())
            .collect::<Vec<_>>()
            .join(", ")
    )
}

/// Converts a DuckDB list/array value (as returned for a `VARCHAR[]`
/// column) into a `Vec<String>`, dropping any non-text elements.
fn extract_string_list(value: Value) -> Vec<String> {
    match value {
        Value::List(items) | Value::Array(items) => items
            .into_iter()
            .filter_map(|item| match item {
                Value::Text(s) => Some(s),
                _ => None,
            })
            .collect(),
        _ => vec![],
    }
}

fn cell_for_column(column: &Column, object: &ObjectRow, classes: &[Class]) -> Cell {
    match column {
        Column::ObjectId => Cell::String(object.object_id.clone()),
        Column::ImageName => Cell::String(object.image_name.clone()),
        Column::ObjectClass => {
            let label = if object.object_class_name.is_empty() {
                object.seg_class_name.clone().unwrap_or_default()
            } else {
                object.object_class_name.join(", ")
            };
            let color = object
                .object_class_name
                .first()
                .and_then(|name| classes.iter().find(|class| &class.name == name))
                .map(|class| class.color)
                .unwrap_or(0);
            Cell::Class((label, color))
        }
        Column::AreaSizePx => Cell::Integer(object.area_px as i32),
        Column::AreaSizeNm => Cell::Float(object.area_nm2 as f32),
        Column::PerimeterPx => Cell::Float(object.perimeter_px as f32),
        Column::PerimeterNm => Cell::Float(object.perimeter_nm as f32),
        Column::Circularity => Cell::Float(object.circularity as f32),
        Column::Solidity => Cell::Float(object.solidity as f32),
        Column::Eccentricity => Cell::Float(object.eccentricity as f32),
        Column::ColocCount => Cell::Integer(coloc_count(&object.coloc_json)),
        // Only channel 0: `ListFilter` has no channel selector yet for the
        // flat list view (unlike the per-channel columns the image editor's
        // object list already supports), so this is a placeholder until one
        // exists rather than a real multi-channel aggregate.
        Column::IntensityAvg => Cell::Float(intensity_stat(&object.intensities_json, "mean_raw")),
        Column::IntensitySum => Cell::Float(intensity_stat(&object.intensities_json, "sum_raw")),
        Column::IntensityMin => Cell::Float(intensity_stat(&object.intensities_json, "min_raw")),
        Column::IntensityMax => Cell::Float(intensity_stat(&object.intensities_json, "max_raw")),
    }
}

/// Total number of colocalization partners across every target class, from
/// the raw `{"<class>": [<object ids>], ...}` shape `coloc_json` stores.
fn coloc_count(coloc_json: &str) -> i32 {
    let Ok(serde_json::Value::Object(partners)) = serde_json::from_str(coloc_json) else {
        return 0;
    };
    partners
        .values()
        .map(|ids| ids.as_array().map_or(0, |ids| ids.len()))
        .sum::<usize>() as i32
}

/// One channel-0 stat out of the raw `{"<channel>": {"mean_raw": ..., ...},
/// ...}` shape `intensities_json` stores (see `intensities_to_json` in
/// evanalyzer_core, whose stat key names this mirrors exactly).
fn intensity_stat(intensities_json: &str, stat: &str) -> f32 {
    let Ok(value) = serde_json::from_str::<serde_json::Value>(intensities_json) else {
        return 0.0;
    };
    value
        .get("0")
        .and_then(|channel| channel.get(stat))
        .and_then(|v| v.as_f64())
        .unwrap_or(0.0) as f32
}
