use duckdb::Connection;
use duckdb::types::Value;
use evanalyzer_cfg::{
    core_types::{InternalErrors, ObjectClass},
    settings::classification_settings::Class,
};
use std::collections::HashMap;
use std::path::PathBuf;
use std::{cell::RefCell, default};

const DEFAULT_GROUPING_REGEX: &str = r"^(([A-H])([0-9]{1,2}))_([0-9]+)\.([a-zA-Z0-9]+)$";

pub struct ResultsGenerator {
    database: Connection,
    classes_cache: RefCell<Option<Vec<Class>>>,
    // Cached after the first call, same reasoning as `classes_cache` (see
    // `get_object_classes_with_at_least_coloc`) — a fresh `ResultsGenerator`
    // per opened database (see `open_database`) means this never needs
    // invalidating, only ever populating once.
    coloc_classes_cache: RefCell<Option<Vec<ObjectClass>>>,
}

#[derive(Clone)]
pub enum View {
    List,
    Heatmap,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlateDimensions {
    PLate2x3,
    Plate3x4,
    Plate4x6,
    Plate6x8,
    Plate8x12,
    Plate16x24,
    Plate32x48,
}

impl PlateDimensions {
    /// Returns the matrix dimensions as a `(rows, columns)` tuple.
    pub const fn dimensions(&self) -> (usize, usize) {
        match self {
            Self::PLate2x3 => (2, 3),
            Self::Plate3x4 => (3, 4),
            Self::Plate4x6 => (4, 6),
            Self::Plate6x8 => (6, 8),
            Self::Plate8x12 => (8, 12),
            Self::Plate16x24 => (16, 24),
            Self::Plate32x48 => (32, 48),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WellSize {
    pub rows: usize,
    pub cols: usize,
}

#[derive(Default, Clone)]
pub enum Aggregation {
    #[default]
    Avg,
    Min,
    Max,
    Stddev,
    Sum,
}

#[derive(Default, Clone, PartialEq, Eq)]
pub enum ColorSchema {
    #[default]
    Excel,
    Viridis,
}

#[derive(Default, Clone)]
pub enum ColorScale {
    #[default]
    Auto,
    Manual(f32, f32),
}

#[derive(Default, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum Column {
    ObjectId,
    ImageName,
    ObjectClass,
    #[default]
    AreaSizePx,
    AreaSizeNm,
    PerimeterPx,
    PerimeterNm,
    Circularity,
    Solidity,
    Eccentricity,
    ColocCount(ObjectClass),
    IntensityAvg(u32),
    IntensitySum(u32),
    IntensityMin(u32),
    IntensityMax(u32),
}

impl Column {
    /// Stable string key (matches the underlying database column name, with
    /// a `_ch{n}` suffix for the per-channel intensity variants) used to
    /// store this variant in UI widgets that only accept strings, e.g. the
    /// slint columns dropdown. Owned (not `&'static str`) because the
    /// channel number/class name has to be formatted in.
    ///
    /// `classes` (typically `ResultsGenerator::get_object_classes()`, cached
    /// there so this is cheap to call repeatedly) resolves
    /// `ColocCount(ObjectClass::Valid(id))`'s class name for the key — falls
    /// back to the raw numeric id if `classes` doesn't (yet, or any longer)
    /// recognize that id, e.g. stale GUI state after switching databases.
    pub fn as_key(&self, classes: &[Class]) -> String {
        match self {
            Column::ObjectId => "object_id".to_string(),
            Column::ImageName => "image_name".to_string(),
            Column::ObjectClass => "object_class_name".to_string(),
            Column::AreaSizePx => "area_px".to_string(),
            Column::AreaSizeNm => "area_nm2".to_string(),
            Column::PerimeterPx => "perimeter_px".to_string(),
            Column::PerimeterNm => "perimeter_nm".to_string(),
            Column::Circularity => "circularity".to_string(),
            Column::Solidity => "solidity".to_string(),
            Column::Eccentricity => "eccentricity".to_string(),
            Column::ColocCount(ObjectClass::Valid(class_id)) => {
                let name = classes
                    .iter()
                    .find(|class| class.id == ObjectClass::Valid(*class_id))
                    .map(|class| class.name.clone())
                    .unwrap_or_else(|| class_id.to_string());
                format!("n_colocalized_class_{name}")
            }
            Column::ColocCount(ObjectClass::Unset) => "n_colocalized_unset".to_string(),
            Column::IntensityAvg(channel) => format!("mean_raw_ch{channel}"),
            Column::IntensitySum(channel) => format!("sum_raw_ch{channel}"),
            Column::IntensityMin(channel) => format!("min_raw_ch{channel}"),
            Column::IntensityMax(channel) => format!("max_raw_ch{channel}"),
        }
    }

    /// Inverse of [`Column::as_key`] — needs the same `classes` list to
    /// resolve a `"n_colocalized_class_{name}"` key back to the class's id;
    /// `None` if `name` isn't (or no longer is) a registered class.
    pub fn from_key(key: &str, classes: &[Class]) -> Option<Self> {
        if let Some(name) = key.strip_prefix("n_colocalized_class_") {
            let class_id = classes.iter().find(|class| class.name == name)?.id;
            return Some(Column::ColocCount(class_id));
        }
        if key == "n_colocalized_unset" {
            return Some(Column::ColocCount(ObjectClass::Unset));
        }
        if let Some(channel) = key.strip_prefix("mean_raw_ch") {
            return channel.parse().ok().map(Column::IntensityAvg);
        }
        if let Some(channel) = key.strip_prefix("sum_raw_ch") {
            return channel.parse().ok().map(Column::IntensitySum);
        }
        if let Some(channel) = key.strip_prefix("min_raw_ch") {
            return channel.parse().ok().map(Column::IntensityMin);
        }
        if let Some(channel) = key.strip_prefix("max_raw_ch") {
            return channel.parse().ok().map(Column::IntensityMax);
        }
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
    /// Keyset cursor: `None` fetches the first page; `Some(id)` fetches the page starting right after that `object_id`.
    pub after: Option<String>,
}

#[derive(Clone)]
pub struct PlateFilter {
    pub plane: PlaneFilter,
    // Grouping regex, requires follwoing regex output (e.g. A1_01.vsi)
    // - Group1: the match of the group (e.g. A1)
    // - Group2: the match of the plate row (e.g. A)
    // - Group3: the match of the plate col (e.g. 1)
    // - Group4: the match of the image index (e.g. 01)
    pub grouping_regex: String,
    pub aggregation: Aggregation,
    pub object_class: ObjectClass,
    pub column: Column,
    pub color_schema: ColorSchema,
    pub color_scale: ColorScale,
    pub matrix_dimension: Option<PlateDimensions>,
}

#[derive(Clone)]
pub struct WellFilter {
    pub plane: PlaneFilter,
    // Name of the group to displax
    pub group_name: String,
    // Grouping regex, requires follwoing regex output (e.g. A1_01.vsi)
    // - Group1: the match of the group (e.g. A1)
    // - Group2: the match of the plate row (e.g. A)
    // - Group3: the match of the plate col (e.g. 1)
    // - Group4: the match of the image index (e.g. 01)
    pub grouping_regex: String,
    pub aggregation: Aggregation,
    pub object_class: ObjectClass,
    pub column: Column,
    pub color_schema: ColorSchema,
    pub color_scale: ColorScale,
    /// Grid dimensions to lay the well's fields out in. `None` assumes a
    /// 4x4 well (the common case for a plate imager's per-well field
    /// count) rather than fitting to whatever fields were actually found,
    /// so a well missing a field still shows a gap at that field's real
    /// position instead of the grid silently shrinking.
    pub well_size: Option<WellSize>,
    /// Maps grid position -> field index for non-trivial (e.g. snake)
    /// acquisition patterns: `well_order[position]` is the field `idx`
    /// (the regex's 4th capture group, e.g. the "01" in "A1_01.vsi") shown
    /// at that row-major grid position. `None` uses `idx` as the position
    /// directly (1-based: idx 1 -> position 0, top-left).
    pub well_order: Option<Vec<u32>>,
}

#[derive(Clone)]
pub struct ImageHeatmapFilter {
    pub plane: PlaneFilter,
    /// Image for which the heatmpa should be generated for
    pub image_rel_path: String,
    pub aggregation: Aggregation,
    pub object_class: ObjectClass,
    pub column: Column,
    pub color_schema: ColorSchema,
    pub color_scale: ColorScale,
    /// Size of a heatmap element (Default 256)
    pub square_size: Option<usize>,
}

#[derive(Clone)]
pub struct ListFilter {
    pub plane: PlaneFilter,
    pub images: Option<Vec<String>>,
    pub object_classes: Option<Vec<ObjectClass>>,
    pub columns: Vec<Column>,
    pub page: Pagination,
}

pub enum CellValue {
    Empty,
    String(String),
    Float(f32),
    Integer(i32),
    /// Object class with color
    Class((String, u32)),
}

pub struct Cell {
    pub value: CellValue,
    /// Cell background color
    pub bg_color: u32,
    /// Optional search (display name, key) (Group name of plate and image_rel_path in well view)
    pub search_key: Option<(String, String)>,
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
    pub min: f32,
    pub max: f32,
}

impl ResultsGenerator {
    pub fn open_database(path: PathBuf) -> Result<Self, InternalErrors> {
        let to_io_err = |e: duckdb::Error| InternalErrors::Io(e.to_string());
        let database = Connection::open(&path).map_err(to_io_err)?;
        Ok(Self {
            database,
            classes_cache: RefCell::new(None),
            coloc_classes_cache: RefCell::new(None),
        })
    }

    pub fn get_list(
        &self,
        filter: &ListFilter,
        _view: &View,
    ) -> Result<DatabaseResult, InternalErrors> {
        let err = |e: duckdb::Error| InternalErrors::Io(e.to_string());
        // Fetched up front (cached — see `classes_cache`) since
        // `column_names` below already needs it to resolve a `ColocCount`
        // column's class name, ahead of where it's also used to validate/
        // translate `filter.object_classes`.
        let classes = self.get_object_classes()?;
        let mut ordered_columns = filter.columns.clone();
        ordered_columns.sort();
        let column_names: Vec<String> = ordered_columns
            .iter()
            .map(|c| c.as_key(&classes).to_string())
            .collect();
        let empty_result = |column_names: Vec<String>| DatabaseResult {
            column_names,
            row_names: vec![],
            rows: vec![],
            min: 0.0,
            max: 0.0,
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
        // `classes` (fetched above) still doubles as validation: an id that
        // no longer names a registered class (e.g. stale GUI state after
        // switching databases) is dropped rather than matched against
        // `object_class_id` blindly.
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
        // Keyset pagination (see the doc comment on `Pagination::after`):
        // narrowing to `object_id > cursor` here, in the same WHERE clause
        // DuckDB already zone-map-prunes on, is what lets it skip whole row
        // groups below the cursor instead of sorting/reading the full table.
        if let Some(cursor) = &filter.page.after {
            conditions.push(format!(
                "object_id > '{}'::UUID",
                cursor.replace('\'', "''")
            ));
        }
        let where_clause = format!("WHERE {}", conditions.join(" AND "));

        // Two-step fetch: first find just the `object_id`s of this page —
        // a query that only ever touches the (fixed-width, cheap) filter
        // columns and `object_id` itself, never the wide/JSON columns below
        // — then re-fetch full rows filtered to exactly those ids. Doing it
        // in one wide `SELECT ... WHERE ... ORDER BY object_id LIMIT n`
        // forces DuckDB to decode every selected column (including whatever
        // of `coloc_json`/`intensities_json` was requested) for every row
        // that matches the WHERE clause before it can even start sorting —
        // on this app's tables that's routinely the *entire* table, since
        // z/t-plane and image/class filters often don't narrow anything.
        // Splitting it lets the second query's `object_id IN (...)` use
        // DuckDB's per-row-group zone maps to skip straight to the row
        // groups that actually contain those ids (measured this dropping a
        // ~5.5M-row table's per-page cost from single-digit GB to
        // single-digit MB, first page included).
        let limit = filter.page.limit.max(0);
        let key_sql = format!(
            "SELECT object_id FROM objects {where_clause} ORDER BY object_id LIMIT {limit}"
        );
        let mut key_stmt = self.database.prepare(&key_sql).map_err(err)?;
        let ids: Vec<String> = key_stmt
            .query_map([], |row| row.get::<_, String>(0))
            .map_err(err)?
            .collect::<Result<Vec<_>, _>>()
            .map_err(err)?;
        if ids.is_empty() {
            return Ok(empty_result(column_names));
        }
        let where_clause = format!("WHERE object_id IN ({})", sql_string_in_list(&ids));

        // Only pull the source columns `ordered_columns` actually needs.
        // DuckDB is columnar: a column replaced by a constant here is never
        // read off disk or carried through the sort, so an unselected
        // `coloc_json`/`intensities_json` (each row's biggest fields, since
        // every other field is a fixed-width number or a short string) costs
        // nothing instead of being materialized for every row that matches
        // the WHERE clause before LIMIT/OFFSET trims it down to one page —
        // this was blowing up RAM on tables with hundreds of thousands of
        // objects even though only `LIST_PAGE_SIZE` rows ever reach the GUI.
        let need_image_name = ordered_columns.contains(&Column::ImageName);
        let need_class = ordered_columns.contains(&Column::ObjectClass);
        let need_area_px = ordered_columns.contains(&Column::AreaSizePx);
        let need_area_nm2 = ordered_columns.contains(&Column::AreaSizeNm);
        let need_perimeter_px = ordered_columns.contains(&Column::PerimeterPx);
        let need_perimeter_nm = ordered_columns.contains(&Column::PerimeterNm);
        let need_circularity = ordered_columns.contains(&Column::Circularity);
        let need_solidity = ordered_columns.contains(&Column::Solidity);
        let need_eccentricity = ordered_columns.contains(&Column::Eccentricity);
        let need_coloc = ordered_columns
            .iter()
            .any(|c| matches!(c, Column::ColocCount(_)));
        let need_intensities = ordered_columns.iter().any(|c| {
            matches!(
                c,
                Column::IntensityAvg(_)
                    | Column::IntensitySum(_)
                    | Column::IntensityMin(_)
                    | Column::IntensityMax(_)
            )
        });

        let select_image_name = if need_image_name { "image_name" } else { "''" };
        let select_object_class_name = if need_class {
            "CAST(object_class_name AS VARCHAR[])"
        } else {
            "CAST(NULL AS VARCHAR[])"
        };
        let select_seg_class_name = if need_class {
            "seg_class_name"
        } else {
            "NULL::VARCHAR"
        };
        let select_area_px = if need_area_px {
            "area_px"
        } else {
            "0::UBIGINT"
        };
        let select_area_nm2 = if need_area_nm2 {
            "area_nm2"
        } else {
            "0.0::DOUBLE"
        };
        let select_perimeter_px = if need_perimeter_px {
            "perimeter_px"
        } else {
            "0.0::DOUBLE"
        };
        let select_perimeter_nm = if need_perimeter_nm {
            "perimeter_nm"
        } else {
            "0.0::DOUBLE"
        };
        let select_circularity = if need_circularity {
            "circularity"
        } else {
            "0.0::DOUBLE"
        };
        let select_solidity = if need_solidity {
            "solidity"
        } else {
            "0.0::DOUBLE"
        };
        let select_eccentricity = if need_eccentricity {
            "eccentricity"
        } else {
            "0.0::DOUBLE"
        };
        let select_coloc_json = if need_coloc {
            "coloc_json"
        } else {
            "NULL::VARCHAR"
        };
        let select_intensities_json = if need_intensities {
            "intensities_json"
        } else {
            "NULL::VARCHAR"
        };

        let sql = format!(
            "SELECT object_id, {select_image_name}, {select_object_class_name}, {select_seg_class_name},\n\
                    {select_area_px}, {select_area_nm2}, {select_perimeter_px}, {select_perimeter_nm},\n\
                    {select_circularity}, {select_solidity}, {select_eccentricity},\n\
                    {select_coloc_json}, {select_intensities_json}\n\
             FROM objects {where_clause}\n\
             ORDER BY object_id"
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

        // Not really meaningful across `ordered_columns` (area/circularity/
        // eccentricity/... are different units mixed in one row), unlike the
        // single-column plate view below — left at 0 rather than guessing at
        // a cross-column range.
        Ok(DatabaseResult {
            column_names,
            row_names,
            rows,
            min: 0.0,
            max: 0.0,
        })
    }

    // First step: return the grouped/aggregated rows as a plain flat table
    // (group key + aggregated value), same `DatabaseResult` shape as
    // `get_list`. Turning that into the plate grid's actual rows/cols of
    // wells (`MatrixCell`s, row/col letters, etc.) is a separate step in the
    // GUI once this data is available.
    pub fn get_group_by_plate(
        &self,
        filter: &PlateFilter,
        view: &View,
    ) -> Result<DatabaseResult, InternalErrors> {
        let err = |e: duckdb::Error| InternalErrors::Io(e.to_string());
        // `column_aggregate_expr` below rejects `Column::ColocCount(_)`
        // (can't be aggregated for this view), so `filter.column.as_key()`
        // further down can never actually need to resolve a class name in
        // practice — fetched anyway (cached, cheap) so that stays true by
        // construction rather than by relying on that ordering.
        let classes = self.get_object_classes()?;
        let value_expr = column_aggregate_expr(&filter.column)?;
        let agg_fn = aggregation_sql_fn(&filter.aggregation);

        // Same default as the example query this is modeled on: everything
        // before the first `_` in `image_name` (e.g. "A1_field1.tif" -> "A1")
        // — used whenever the GUI's regex box (`filter.grouping_regex`) is
        // still empty.
        let regex = if filter.grouping_regex.trim().is_empty() {
            DEFAULT_GROUPING_REGEX
        } else {
            filter.grouping_regex.as_str()
        };

        let mut conditions = vec![
            format!("z_stack = {}", filter.plane.z_stack),
            format!("t_stack = {}", filter.plane.t_stack),
        ];
        if let ObjectClass::Valid(id) = filter.object_class {
            conditions.push(format!(
                "list_has_any(CAST(object_class_id AS INTEGER[]), {})",
                sql_int_array_literal(&[id])
            ));
        }
        let where_clause = format!("WHERE {}", conditions.join(" AND "));

        let sql = format!(
            "SELECT\n\
                regexp_extract(image_name, '{regex}', 1) AS group_prefix,\n\
                regexp_extract(image_name, '{regex}', 2) AS row,\n\
                regexp_extract(image_name, '{regex}', 3) AS col,\n\
                {agg_fn}({value_expr}) AS value\n\
             FROM objects\n\
             {where_clause}\n\
             GROUP BY group_prefix, row, col\n\
             ORDER BY group_prefix",
            regex = regex.replace('\'', "''"),
        );

        let mut stmt = self.database.prepare(&sql).map_err(err)?;

        match view {
            View::List => {
                let groups: Vec<(String, Option<f64>)> = stmt
                    .query_map([], |row| Ok((row.get(0)?, row.get(3)?)))
                    .map_err(err)?
                    .collect::<Result<Vec<_>, _>>()
                    .map_err(err)?;

                let mut min = f64::INFINITY;
                let mut max = f64::NEG_INFINITY;
                for (_, value) in &groups {
                    if let Some(value) = value {
                        min = min.min(*value);
                        max = max.max(*value);
                    }
                }
                if !min.is_finite() || !max.is_finite() {
                    min = 0.0;
                    max = 0.0;
                }

                let column_names = vec!["group".to_string(), filter.column.as_key(&classes)];
                let row_names = groups.iter().map(|(key, _)| key.clone()).collect();
                let rows = groups
                    .into_iter()
                    .map(|(key, value)| {
                        // `key` is the group/well id (e.g. "A1") itself, so
                        // it's its own search key — used by the GUI to
                        // navigate into that group/well.
                        let search_key = Some((key.clone(), key.clone()));
                        vec![
                            Cell {
                                value: CellValue::String(key),
                                bg_color: 0,
                                search_key: search_key.clone(),
                            },
                            Cell {
                                value: CellValue::Float(value.unwrap_or(0.0) as f32),
                                bg_color: 0,
                                search_key,
                            },
                        ]
                    })
                    .collect();
                Ok(DatabaseResult {
                    column_names,
                    row_names,
                    rows,
                    min: min as f32,
                    max: max as f32,
                })
            }
            View::Heatmap => {
                // `row`/`col` (capture groups 2/3 of the same regex — e.g.
                // "A"/"1" out of well id "A1") are the grid's two axes here,
                // unlike `View::List` which only needed the full group key
                // (capture group 1).
                let cells: Vec<(String, String, String, Option<f64>)> = stmt
                    .query_map([], |row| {
                        Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?))
                    })
                    .map_err(err)?
                    .collect::<Result<Vec<_>, _>>()
                    .map_err(err)?;

                // Real 0-based (row, col) well coordinates ("A" -> 0, "1" ->
                // 0, ...), not just distinct-and-sorted keys — needed so the
                // grid always lines up with a real plate's row/column
                // numbering (see `matrix_dimension` below) instead of
                // silently compressing when a row or column has no objects
                // at all. `group_prefix` (e.g. "A1") rides along per cell so
                // it can be returned as `Cell::search_key` below.
                let mut values: HashMap<(usize, usize), (f64, String)> = HashMap::new();
                let mut max_row = None;
                let mut max_col = None;
                for (group_prefix, row, col, value) in &cells {
                    let (Some(row), Some(col)) =
                        (row_letter_to_index(row), col_number_to_index(col))
                    else {
                        continue;
                    };
                    max_row = Some(max_row.map_or(row, |m: usize| m.max(row)));
                    max_col = Some(max_col.map_or(col, |m: usize| m.max(col)));
                    if let Some(value) = value {
                        values.insert((row, col), (*value, group_prefix.clone()));
                    }
                }

                // Given: use it exactly, so the caller can request e.g. a
                // 384-well layout even if this particular plate only has
                // objects in a handful of wells. Not given: the smallest
                // standard plate size that still fits every well this query
                // actually found.
                let dimensions = filter
                    .matrix_dimension
                    .unwrap_or_else(|| best_matching_dimensions(max_row, max_col));
                let (rows, cols) = dimensions.dimensions();

                let (range_min, range_max) = match filter.color_scale {
                    ColorScale::Manual(min, max) => (min as f64, max as f64),
                    ColorScale::Auto => {
                        let mut min = f64::INFINITY;
                        let mut max = f64::NEG_INFINITY;
                        for (value, _) in values.values() {
                            min = min.min(*value);
                            max = max.max(*value);
                        }
                        if min.is_finite() && max.is_finite() {
                            (min, max)
                        } else {
                            (0.0, 0.0)
                        }
                    }
                };

                let grid_rows = (0..rows)
                    .map(|row| {
                        (0..cols)
                            .map(|col| match values.get(&(row, col)) {
                                Some((value, group_prefix)) => Cell {
                                    value: CellValue::Float(*value as f32),
                                    bg_color: value_to_color(
                                        *value,
                                        range_min,
                                        range_max,
                                        &filter.color_schema,
                                    ),
                                    search_key: Some((group_prefix.clone(), group_prefix.clone())),
                                },
                                // No object matched this well at all — leave
                                // it empty rather than showing a misleading 0
                                // or a value from some other well.
                                None => Cell {
                                    value: CellValue::Empty,
                                    bg_color: 0,
                                    search_key: None,
                                },
                            })
                            .collect()
                    })
                    .collect();

                Ok(DatabaseResult {
                    column_names: (1..=cols).map(|col| col.to_string()).collect(),
                    row_names: (0..rows).map(row_index_to_letter).collect(),
                    rows: grid_rows,
                    // Same range the cells were colored against above, so the
                    // GUI's color bar always matches what's actually painted
                    // rather than recomputing (and potentially disagreeing
                    // with) it from the returned cells.
                    min: range_min as f32,
                    max: range_max as f32,
                })
            }
        }
    }

    // Second drill level: the fields (individual images) inside one well
    // (`filter.group_name`, e.g. "A1"). Mirrors `get_group_by_plate` in
    // shape and view handling, just one level deeper — group key here is
    // the field index (regex capture group 4, e.g. the "01" in
    // "A1_01.vsi"), not the well id.
    //
    // The example query this is modeled on filtered with
    // `WHERE group_prefix = 'A1'`, but `group_prefix` is a `SELECT`-list
    // alias (itself a `regexp_extract(...)` call) — DuckDB (like standard
    // SQL) evaluates `WHERE` before `SELECT`, so a bare alias reference
    // there is not visible yet. Re-running the same `regexp_extract(...)`
    // call directly in the `WHERE` clause below gets the same filter
    // without that alias problem.
    pub fn get_group_by_well(
        &self,
        filter: &WellFilter,
        view: &View,
    ) -> Result<DatabaseResult, InternalErrors> {
        let err = |e: duckdb::Error| InternalErrors::Io(e.to_string());
        let classes = self.get_object_classes()?;
        let value_expr = column_aggregate_expr(&filter.column)?;
        let agg_fn = aggregation_sql_fn(&filter.aggregation);

        let regex = if filter.grouping_regex.trim().is_empty() {
            DEFAULT_GROUPING_REGEX
        } else {
            filter.grouping_regex.as_str()
        };
        let regex = regex.replace('\'', "''");

        let mut conditions = vec![
            format!("z_stack = {}", filter.plane.z_stack),
            format!("t_stack = {}", filter.plane.t_stack),
            format!(
                "regexp_extract(image_name, '{regex}', 1) = '{}'",
                filter.group_name.replace('\'', "''")
            ),
        ];
        if let ObjectClass::Valid(id) = filter.object_class {
            conditions.push(format!(
                "list_has_any(CAST(object_class_id AS INTEGER[]), {})",
                sql_int_array_literal(&[id])
            ));
        }
        let where_clause = format!("WHERE {}", conditions.join(" AND "));

        let sql = format!(
            "SELECT\n\
                regexp_extract(image_name, '{regex}', 4) AS idx,\n\
                image_rel_path,\n\
                image_name,\n\
                {agg_fn}({value_expr}) AS value\n\
             FROM objects\n\
             {where_clause}\n\
             GROUP BY idx, image_rel_path, image_name\n\
             ORDER BY idx"
        );

        let mut stmt = self.database.prepare(&sql).map_err(err)?;
        // (idx, image_rel_path, image_name, value) — `image_rel_path` and
        // `image_name` are carried through into `Cell::search_key` on every
        // cell for this field so the GUI can select/open the underlying
        // image from a well-view tile (see `ImageEntry`, which the GUI
        // matches images against by `rel_path`).
        let fields: Vec<(String, String, String, Option<f64>)> = stmt
            .query_map([], |row| {
                Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?))
            })
            .map_err(err)?
            .collect::<Result<Vec<_>, _>>()
            .map_err(err)?;

        match view {
            View::List => {
                let mut min = f64::INFINITY;
                let mut max = f64::NEG_INFINITY;
                for (_, _, _, value) in &fields {
                    if let Some(value) = value {
                        min = min.min(*value);
                        max = max.max(*value);
                    }
                }
                if !min.is_finite() || !max.is_finite() {
                    min = 0.0;
                    max = 0.0;
                }

                let column_names = vec!["field".to_string(), filter.column.as_key(&classes)];
                let row_names = fields.iter().map(|(idx, ..)| idx.clone()).collect();
                let rows = fields
                    .into_iter()
                    .map(|(idx, image_rel_path, image_name, value)| {
                        let search_key = Some((image_name, image_rel_path));
                        vec![
                            Cell {
                                value: CellValue::String(idx),
                                bg_color: 0,
                                search_key: search_key.clone(),
                            },
                            Cell {
                                value: CellValue::Float(value.unwrap_or(0.0) as f32),
                                bg_color: 0,
                                search_key,
                            },
                        ]
                    })
                    .collect();
                Ok(DatabaseResult {
                    column_names,
                    row_names,
                    rows,
                    min: min as f32,
                    max: max as f32,
                })
            }
            View::Heatmap => {
                // No `well_order` (see the doc comment on
                // `WellFilter::well_order`): a field's `idx` (1-based) is
                // its position directly, in row-major reading order —
                // idx 1 -> (0, 0), idx 2 -> (0, 1), .... Given a
                // `well_order`, it's a lookup table instead: the value at
                // `well_order[position]` names which field idx sits at
                // that (row-major) grid position, letting a well be laid
                // out in a non-trivial (e.g. snake) acquisition pattern.
                let well_size = filter.well_size.unwrap_or(WellSize { rows: 4, cols: 4 });
                let (rows, cols) = (well_size.rows, well_size.cols);

                let mut values: HashMap<usize, (f64, String, String)> = HashMap::new();
                for (idx_str, image_rel_path, image_name, value) in &fields {
                    let Ok(idx) = idx_str.parse::<u32>() else {
                        continue;
                    };
                    let position = match &filter.well_order {
                        Some(order) => order.iter().position(|&field_idx| field_idx == idx),
                        None => idx.checked_sub(1).map(|p| p as usize),
                    };
                    let Some(position) = position else {
                        continue;
                    };
                    if let Some(value) = value {
                        values.insert(
                            position,
                            (*value, image_name.clone(), image_rel_path.clone()),
                        );
                    }
                }

                let (range_min, range_max) = match filter.color_scale {
                    ColorScale::Manual(min, max) => (min as f64, max as f64),
                    ColorScale::Auto => {
                        let mut min = f64::INFINITY;
                        let mut max = f64::NEG_INFINITY;
                        for (value, ..) in values.values() {
                            min = min.min(*value);
                            max = max.max(*value);
                        }
                        if min.is_finite() && max.is_finite() {
                            (min, max)
                        } else {
                            (0.0, 0.0)
                        }
                    }
                };

                let grid_rows = (0..rows)
                    .map(|row| {
                        (0..cols)
                            .map(|col| match values.get(&(row * cols + col)) {
                                Some((value, image_name, image_rel_path)) => Cell {
                                    value: CellValue::Float(*value as f32),
                                    bg_color: value_to_color(
                                        *value,
                                        range_min,
                                        range_max,
                                        &filter.color_schema,
                                    ),
                                    search_key: Some((image_name.clone(), image_rel_path.clone())),
                                },
                                // No field occupies this grid position —
                                // leave it empty rather than showing a
                                // misleading 0 or another field's value.
                                None => Cell {
                                    value: CellValue::Empty,
                                    bg_color: 0,
                                    search_key: None,
                                },
                            })
                            .collect()
                    })
                    .collect();

                Ok(DatabaseResult {
                    column_names: (1..=cols).map(|col| col.to_string()).collect(),
                    row_names: (1..=rows).map(|row| row.to_string()).collect(),
                    rows: grid_rows,
                    min: range_min as f32,
                    max: range_max as f32,
                })
            }
        }
    }

    // Third drill level: a spatial heatmap over one image's own pixels
    // (`filter.image_rel_path`, e.g. "A1_01.vsi") — mirrors
    // `get_group_by_plate`/`get_group_by_well` in shape and view handling,
    // just with the grid binned by `square_size`-pixel tiles of the image
    // instead of grouped by a regex-derived key. `centroid_x_px`/
    // `centroid_y_px` (already computed per object, see `duckdb.rs`'s
    // exporter) give each object's tile via integer-divide-by-`square_size`;
    // the image's own `width`/`height` (from the `images` table — see
    // `finalize_image`) size the grid so every tile is represented even if
    // it has no objects at all, the same way `get_group_by_plate`'s
    // `PlateDimensions` fill unmatched wells with `CellValue::Empty` rather
    // than silently compressing the grid.
    pub fn get_image_heatmap(
        &self,
        filter: &ImageHeatmapFilter,
        view: &View,
    ) -> Result<DatabaseResult, InternalErrors> {
        let err = |e: duckdb::Error| InternalErrors::Io(e.to_string());
        let classes = self.get_object_classes()?;
        let value_expr = column_aggregate_expr(&filter.column)?;
        let agg_fn = aggregation_sql_fn(&filter.aggregation);
        let square_size = filter.square_size.unwrap_or(256).max(1);
        let image_rel_path = filter.image_rel_path.replace('\'', "''");

        let (width, height): (u32, u32) = self
            .database
            .query_row(
                &format!(
                    "SELECT width, height FROM images WHERE image_rel_path = '{image_rel_path}'"
                ),
                [],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .map_err(err)?;
        let cols = (width as usize).div_ceil(square_size).max(1);
        let rows = (height as usize).div_ceil(square_size).max(1);

        let mut conditions = vec![
            format!("z_stack = {}", filter.plane.z_stack),
            format!("t_stack = {}", filter.plane.t_stack),
            format!("image_rel_path = '{image_rel_path}'"),
        ];
        if let ObjectClass::Valid(id) = filter.object_class {
            conditions.push(format!(
                "list_has_any(CAST(object_class_id AS INTEGER[]), {})",
                sql_int_array_literal(&[id])
            ));
        }
        let where_clause = format!("WHERE {}", conditions.join(" AND "));

        let sql = format!(
            "SELECT\n\
                CAST(centroid_x_px / {square_size} AS INTEGER) AS col,\n\
                CAST(centroid_y_px / {square_size} AS INTEGER) AS row,\n\
                {agg_fn}({value_expr}) AS value\n\
             FROM objects\n\
             {where_clause}\n\
             GROUP BY col, row\n\
             ORDER BY row, col"
        );

        let mut stmt = self.database.prepare(&sql).map_err(err)?;
        let raw_cells: Vec<(i64, i64, Option<f64>)> = stmt
            .query_map([], |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)))
            .map_err(err)?
            .collect::<Result<Vec<_>, _>>()
            .map_err(err)?;

        // Clamp rather than drop: an object whose centroid sits exactly on
        // (or, from floating-point slop, just past) the image's far edge
        // would otherwise floor-divide into a `cols`/`rows`-th tile that the
        // grid (sized from `width`/`height`) doesn't have a slot for.
        let mut values: HashMap<(usize, usize), f64> = HashMap::new();
        for (col, row, value) in &raw_cells {
            let (Some(value), Ok(col), Ok(row)) =
                (value, usize::try_from(*col), usize::try_from(*row))
            else {
                continue;
            };
            values.insert((row.min(rows - 1), col.min(cols - 1)), *value);
        }

        match view {
            View::List => {
                let mut min = f64::INFINITY;
                let mut max = f64::NEG_INFINITY;
                for value in values.values() {
                    min = min.min(*value);
                    max = max.max(*value);
                }
                if !min.is_finite() || !max.is_finite() {
                    min = 0.0;
                    max = 0.0;
                }

                let mut sorted: Vec<_> = values.iter().collect();
                sorted.sort_by_key(|(pos, _)| *pos);

                let column_names = vec!["square".to_string(), filter.column.as_key(&classes)];
                let row_names = sorted
                    .iter()
                    .map(|((row, col), _)| format!("R{row}C{col}"))
                    .collect();
                let rows_out = sorted
                    .into_iter()
                    .map(|((row, col), value)| {
                        // No further drill level exists below the image
                        // heatmap, so — like the plate's well cells — a
                        // square is its own search key.
                        let key = format!("R{row}C{col}");
                        let search_key = Some((key.clone(), key.clone()));
                        vec![
                            Cell {
                                value: CellValue::String(key),
                                bg_color: 0,
                                search_key: search_key.clone(),
                            },
                            Cell {
                                value: CellValue::Float(*value as f32),
                                bg_color: 0,
                                search_key,
                            },
                        ]
                    })
                    .collect();
                Ok(DatabaseResult {
                    column_names,
                    row_names,
                    rows: rows_out,
                    min: min as f32,
                    max: max as f32,
                })
            }
            View::Heatmap => {
                let (range_min, range_max) = match filter.color_scale {
                    ColorScale::Manual(min, max) => (min as f64, max as f64),
                    ColorScale::Auto => {
                        let mut min = f64::INFINITY;
                        let mut max = f64::NEG_INFINITY;
                        for value in values.values() {
                            min = min.min(*value);
                            max = max.max(*value);
                        }
                        if min.is_finite() && max.is_finite() {
                            (min, max)
                        } else {
                            (0.0, 0.0)
                        }
                    }
                };

                let grid_rows = (0..rows)
                    .map(|row| {
                        (0..cols)
                            .map(|col| match values.get(&(row, col)) {
                                Some(value) => {
                                    let key = format!("R{row}C{col}");
                                    Cell {
                                        value: CellValue::Float(*value as f32),
                                        bg_color: value_to_color(
                                            *value,
                                            range_min,
                                            range_max,
                                            &filter.color_schema,
                                        ),
                                        search_key: Some((key.clone(), key)),
                                    }
                                }
                                // No object fell into this tile at all —
                                // leave it empty rather than showing a
                                // misleading 0 or a neighboring tile's value.
                                None => Cell {
                                    value: CellValue::Empty,
                                    bg_color: 0,
                                    search_key: None,
                                },
                            })
                            .collect()
                    })
                    .collect();

                Ok(DatabaseResult {
                    column_names: (0..cols).map(|col| col.to_string()).collect(),
                    row_names: (0..rows).map(|row| row.to_string()).collect(),
                    rows: grid_rows,
                    min: range_min as f32,
                    max: range_max as f32,
                })
            }
        }
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

    /// Every class id that appears as a colocalization partner in at least
    /// one object's `coloc_json` — i.e. the candidates for a
    /// `Column::ColocCount(class)` column, mirroring how
    /// `get_available_columns` enumerates one Avg/Sum/Min/Max intensity
    /// column per `get_nr_of_c_stacks()` channel. `coloc_json` is keyed by
    /// class id directly (see `coloc_to_json` in evanalyzer_core's
    /// duckdb.rs), so this just needs the distinct keys across every
    /// non-empty `coloc_json` — no join against `classes` required to
    /// recover the id itself, only to resolve display names later in
    /// `get_available_columns`.
    ///
    /// Cached after the first call for this opened database (see
    /// `coloc_classes_cache`), same reasoning as `get_object_classes`: this
    /// scans every non-empty `coloc_json` in the table, and the set of
    /// classes ever recorded as a coloc partner can't change without
    /// re-exporting (i.e. opening a different database).
    pub fn get_object_classes_with_at_least_coloc(
        &self,
    ) -> Result<Vec<ObjectClass>, InternalErrors> {
        if let Some(cached) = self.coloc_classes_cache.borrow().as_ref() {
            return Ok(cached.clone());
        }

        let err = |e: duckdb::Error| InternalErrors::Io(e.to_string());
        let mut stmt = self
            .database
            .prepare(
                "SELECT DISTINCT UNNEST(json_keys(coloc_json)) AS class_key \
                 FROM objects \
                 WHERE coloc_json IS NOT NULL AND CAST(coloc_json AS VARCHAR) != '{}'",
            )
            .map_err(err)?;
        let keys: Vec<String> = stmt
            .query_map([], |row| row.get::<_, String>(0))
            .map_err(err)?
            .collect::<Result<Vec<_>, _>>()
            .map_err(err)?;

        let mut classes: Vec<ObjectClass> = keys
            .into_iter()
            .filter_map(|key| key.parse::<u32>().ok())
            .map(ObjectClass::Valid)
            .collect();
        classes.sort();
        classes.dedup();

        *self.coloc_classes_cache.borrow_mut() = Some(classes.clone());
        Ok(classes)
    }

    pub fn get_available_columns(&self) -> Result<Vec<ColumnEntry>, InternalErrors> {
        let mut ret = vec![
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
        ];

        // Coloc count is measured per candidate partner class (like
        // intensity is measured per channel below), so there's one column
        // per class that actually shows up as a colocalization partner
        // somewhere in this database, rather than a single shared "total
        // across every class" column.
        let classes = self.get_object_classes()?;
        for class_id in self.get_object_classes_with_at_least_coloc()? {
            let display_name = classes
                .iter()
                .find(|class| class.id == class_id)
                .map(|class| format!("Coloc with {}", class.name))
                .unwrap_or_else(|| match class_id {
                    ObjectClass::Valid(n) => format!("Coloc with class {n}"),
                    ObjectClass::Unset => "Coloc with unset".to_string(),
                });
            ret.push(ColumnEntry {
                display_name,
                key: Column::ColocCount(class_id),
                group: "Coloc".into(),
            });
        }

        // Intensity is measured per image channel, so there's one Avg/Sum/
        // Min/Max column per channel rather than a single shared one.
        for channel in 0..self.get_nr_of_c_stacks() {
            ret.push(ColumnEntry {
                display_name: format!("Avg Intensity (Ch {channel})"),
                key: Column::IntensityAvg(channel),
                group: "intensity".into(),
            });
            ret.push(ColumnEntry {
                display_name: format!("Sum Intensity (Ch {channel})"),
                key: Column::IntensitySum(channel),
                group: "intensity".into(),
            });
            ret.push(ColumnEntry {
                display_name: format!("Min Intensity (Ch {channel})"),
                key: Column::IntensityMin(channel),
                group: "intensity".into(),
            });
            ret.push(ColumnEntry {
                display_name: format!("Max Intensity (Ch {channel})"),
                key: Column::IntensityMax(channel),
                group: "intensity".into(),
            });
        }

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
    // Only the class badge carries a background color today; every other
    // column renders on the table's normal row background.
    let no_bg = Cell {
        value: CellValue::String(String::new()),
        bg_color: 0,
        search_key: None,
    };
    match column {
        Column::ObjectId => Cell {
            value: CellValue::String(object.object_id.clone()),
            ..no_bg
        },
        Column::ImageName => Cell {
            value: CellValue::String(object.image_name.clone()),
            ..no_bg
        },
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
            Cell {
                value: CellValue::Class((label, color)),
                bg_color: color,
                search_key: None,
            }
        }
        Column::AreaSizePx => Cell {
            value: CellValue::Integer(object.area_px as i32),
            ..no_bg
        },
        Column::AreaSizeNm => Cell {
            value: CellValue::Float(object.area_nm2 as f32),
            ..no_bg
        },
        Column::PerimeterPx => Cell {
            value: CellValue::Float(object.perimeter_px as f32),
            ..no_bg
        },
        Column::PerimeterNm => Cell {
            value: CellValue::Float(object.perimeter_nm as f32),
            ..no_bg
        },
        Column::Circularity => Cell {
            value: CellValue::Float(object.circularity as f32),
            ..no_bg
        },
        Column::Solidity => Cell {
            value: CellValue::Float(object.solidity as f32),
            ..no_bg
        },
        Column::Eccentricity => Cell {
            value: CellValue::Float(object.eccentricity as f32),
            ..no_bg
        },
        Column::ColocCount(class) => Cell {
            value: CellValue::Integer(coloc_count_for_class(&object.coloc_json, *class)),
            ..no_bg
        },
        Column::IntensityAvg(channel) => Cell {
            value: CellValue::Float(intensity_stat(
                &object.intensities_json,
                *channel,
                "mean_raw",
            )),
            ..no_bg
        },
        Column::IntensitySum(channel) => Cell {
            value: CellValue::Float(intensity_stat(
                &object.intensities_json,
                *channel,
                "sum_raw",
            )),
            ..no_bg
        },
        Column::IntensityMin(channel) => Cell {
            value: CellValue::Float(intensity_stat(
                &object.intensities_json,
                *channel,
                "min_raw",
            )),
            ..no_bg
        },
        Column::IntensityMax(channel) => Cell {
            value: CellValue::Float(intensity_stat(
                &object.intensities_json,
                *channel,
                "max_raw",
            )),
            ..no_bg
        },
    }
}

/// Number of `class`-colocalizing partners a object has, from the raw
/// `{"<class_id>": [<object ids>], ...}` shape `coloc_json` stores (see
/// `coloc_to_json` in evanalyzer_core's duckdb.rs) — keyed by the target
/// class's numeric id, not its name. `0` if `class` never shows up as a key
/// at all (no colocalization with that class recorded for this object).
fn coloc_count_for_class(coloc_json: &str, class: ObjectClass) -> i32 {
    let Ok(serde_json::Value::Object(partners)) = serde_json::from_str(coloc_json) else {
        return 0;
    };
    let key = match class {
        ObjectClass::Valid(n) => n.to_string(),
        ObjectClass::Unset => "unset".to_string(),
    };
    partners
        .get(&key)
        .and_then(|v| v.as_array())
        .map_or(0, |ids| ids.len() as i32)
}

/// One channel's stat out of the raw `{"<channel>": {"mean_raw": ..., ...},
/// ...}` shape `intensities_json` stores (see `intensities_to_json` in
/// evanalyzer_core, whose stat key names this mirrors exactly).
fn intensity_stat(intensities_json: &str, channel: u32, stat: &str) -> f32 {
    let Ok(value) = serde_json::from_str::<serde_json::Value>(intensities_json) else {
        return 0.0;
    };
    value
        .get(channel.to_string())
        .and_then(|channel| channel.get(stat))
        .and_then(|v| v.as_f64())
        .unwrap_or(0.0) as f32
}

/// SQL scalar expression for a `Column`, to be wrapped in an aggregate
/// function by `get_group_by_plate`/`get_group_by_well`/`get_image_heatmap`.
/// Plain numeric `objects` columns are a direct column reference;
/// `ColocCount` is a `json_array_length` extraction (see
/// `coloc_count_for_class`, which does the same lookup per-row in Rust for
/// `get_list`). The per-channel intensity columns still need their own
/// JSON-extraction SQL (see `intensity_stat`, which — like `coloc_count_for_class`
/// before this — only handles this per-row in Rust today, not as a groupable
/// SQL expression), left for a follow-up.
fn column_aggregate_expr(column: &Column) -> Result<String, InternalErrors> {
    Ok(match column {
        Column::AreaSizePx => "area_px".to_string(),
        Column::AreaSizeNm => "area_nm2".to_string(),
        Column::PerimeterPx => "perimeter_px".to_string(),
        Column::PerimeterNm => "perimeter_nm".to_string(),
        Column::Circularity => "circularity".to_string(),
        Column::Solidity => "solidity".to_string(),
        Column::Eccentricity => "eccentricity".to_string(),
        // Same shape as `coloc_partner_count_expr` in evanalyzer_core's
        // duckdb.rs: `coloc_json` is a native `JSON` column, keyed by class
        // id (see `coloc_to_json`), so `->` always receives well-formed
        // JSON — no string-literal-cast guard needed here.
        Column::ColocCount(ObjectClass::Valid(class_id)) => {
            format!("COALESCE(json_array_length(coloc_json -> '{class_id}'), 0)")
        }
        Column::ColocCount(ObjectClass::Unset) => {
            "COALESCE(json_array_length(coloc_json -> 'unset'), 0)".to_string()
        }
        Column::ObjectId
        | Column::ImageName
        | Column::ObjectClass
        | Column::IntensityAvg(_)
        | Column::IntensitySum(_)
        | Column::IntensityMin(_)
        | Column::IntensityMax(_) => {
            // No classes list handy here (this is a plain error-message
            // helper, not a `ResultsGenerator` method) — `as_key` already
            // falls back to the raw numeric id for `ColocCount` when it
            // can't resolve a name, which is fine for an error message.
            return Err(InternalErrors::InvalidArgument(format!(
                "column {} cannot be aggregated for the plate view yet",
                column.as_key(&[])
            )));
        }
    })
}

fn aggregation_sql_fn(aggregation: &Aggregation) -> &'static str {
    match aggregation {
        Aggregation::Avg => "AVG",
        Aggregation::Min => "MIN",
        Aggregation::Max => "MAX",
        Aggregation::Stddev => "STDDEV_SAMP",
        Aggregation::Sum => "SUM",
    }
}

// Approximate 5-stop reproduction of the matplotlib "viridis" colormap
// (dark purple -> teal -> yellow).
const VIRIDIS_STOPS: [(f32, (u8, u8, u8)); 5] = [
    (0.0, (0x44, 0x01, 0x54)),
    (0.25, (0x3b, 0x52, 0x8b)),
    (0.5, (0x21, 0x90, 0x8d)),
    (0.75, (0x5d, 0xc9, 0x63)),
    (1.0, (0xfd, 0xe7, 0x25)),
];

// Excel's built-in "Red - Yellow - Green" 3-Color Scale conditional format.
const EXCEL_STOPS: [(f32, (u8, u8, u8)); 3] = [
    (0.0, (0xf8, 0x69, 0x6b)),
    (0.5, (0xff, 0xeb, 0x84)),
    (1.0, (0x63, 0xbe, 0x7b)),
];

/// Every standard plate size, smallest first — `best_matching_dimensions`
/// relies on this order to find the smallest one that fits.
const ALL_PLATE_DIMENSIONS: [PlateDimensions; 7] = [
    PlateDimensions::PLate2x3,
    PlateDimensions::Plate3x4,
    PlateDimensions::Plate4x6,
    PlateDimensions::Plate6x8,
    PlateDimensions::Plate8x12,
    PlateDimensions::Plate16x24,
    PlateDimensions::Plate32x48,
];

/// The smallest standard plate size whose row/column count covers every well
/// this query actually found (`max_row`/`max_col`, both 0-based). Falls back
/// to the largest known size if even that doesn't fit (a plate bigger than
/// any standard format, or a `grouping_regex` extracting something that
/// isn't really a well id).
fn best_matching_dimensions(max_row: Option<usize>, max_col: Option<usize>) -> PlateDimensions {
    let needed_rows = max_row.map_or(1, |row| row + 1);
    let needed_cols = max_col.map_or(1, |col| col + 1);
    ALL_PLATE_DIMENSIONS
        .into_iter()
        .find(|dimensions| {
            let (rows, cols) = dimensions.dimensions();
            rows >= needed_rows && cols >= needed_cols
        })
        .unwrap_or(PlateDimensions::Plate32x48)
}

/// Parses a well's row letters ("A", "B", ..., "Z", "AA", "AB", ...) into a
/// 0-based row index, using the same bijective base-26 scheme spreadsheet
/// column letters use. `None` if `letters` isn't purely alphabetic (e.g. the
/// `grouping_regex` didn't actually match a well id).
fn row_letter_to_index(letters: &str) -> Option<usize> {
    if letters.is_empty() || !letters.chars().all(|c| c.is_ascii_alphabetic()) {
        return None;
    }
    let mut index: usize = 0;
    for c in letters.chars() {
        let digit = (c.to_ascii_uppercase() as u8 - b'A') as usize + 1;
        index = index * 26 + digit;
    }
    Some(index - 1)
}

/// Inverse of [`row_letter_to_index`].
fn row_index_to_letter(index: usize) -> String {
    let mut n = index + 1;
    let mut letters = Vec::new();
    while n > 0 {
        let rem = (n - 1) % 26;
        letters.push((b'A' + rem as u8) as char);
        n = (n - 1) / 26;
    }
    letters.iter().rev().collect()
}

/// Parses a well's column number ("1", "2", ...) into a 0-based column
/// index. `None` if `digits` isn't a positive integer.
fn col_number_to_index(digits: &str) -> Option<usize> {
    digits.parse::<usize>().ok()?.checked_sub(1)
}

/// Number of colors `color_scale_gradient` samples a schema at — enough for
/// the GUI's legend bar to look like a smooth gradient when it just splits
/// the stops evenly across a `HorizontalLayout`.
pub const COLOR_SCALE_GRADIENT_STOPS: usize = 12;

/// Samples `value_to_color` at `COLOR_SCALE_GRADIENT_STOPS` evenly spaced
/// points across `[0, 1]`, in `0xRRGGBB`. Lets the GUI's legend bar render
/// the exact gradient a heatmap's cells are colored with, instead of
/// reimplementing the schema's interpolation a second time in Slint.
pub fn color_scale_gradient(schema: &ColorSchema) -> [u32; COLOR_SCALE_GRADIENT_STOPS] {
    let mut stops = [0u32; COLOR_SCALE_GRADIENT_STOPS];
    for (i, stop) in stops.iter_mut().enumerate() {
        let t = i as f64 / (COLOR_SCALE_GRADIENT_STOPS - 1) as f64;
        *stop = value_to_color(t, 0.0, 1.0, schema);
    }
    stops
}

/// Maps `value` (within `[min, max]`) to a `0xRRGGBB` color under the
/// selected `ColorSchema` — the same packing `evanalyzer_cfg`'s `Class.color`
/// and `crates/gui/src/helper/color_generators.rs` already use, so the GUI
/// can unpack a heatmap cell's `bg_color` the same way it already does for
/// class colors.
fn value_to_color(value: f64, min: f64, max: f64, schema: &ColorSchema) -> u32 {
    let t = if max > min {
        ((value - min) / (max - min)).clamp(0.0, 1.0) as f32
    } else {
        0.5
    };
    match schema {
        ColorSchema::Viridis => lerp_palette(&VIRIDIS_STOPS, t),
        ColorSchema::Excel => lerp_palette(&EXCEL_STOPS, t),
    }
}

fn lerp_palette(stops: &[(f32, (u8, u8, u8))], t: f32) -> u32 {
    let t = t.clamp(0.0, 1.0);
    for pair in stops.windows(2) {
        let (t0, c0) = pair[0];
        let (t1, c1) = pair[1];
        if t >= t0 && t <= t1 {
            let local_t = (t - t0) / (t1 - t0).max(f32::EPSILON);
            return pack_rgb(
                lerp_u8(c0.0, c1.0, local_t),
                lerp_u8(c0.1, c1.1, local_t),
                lerp_u8(c0.2, c1.2, local_t),
            );
        }
    }
    let (_, last) = *stops.last().expect("palette must have at least one stop");
    pack_rgb(last.0, last.1, last.2)
}

fn lerp_u8(a: u8, b: u8, t: f32) -> u8 {
    (a as f32 + (b as f32 - a as f32) * t).round() as u8
}

fn pack_rgb(r: u8, g: u8, b: u8) -> u32 {
    ((r as u32) << 16) | ((g as u32) << 8) | (b as u32)
}
