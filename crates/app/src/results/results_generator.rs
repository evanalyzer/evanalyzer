use crate::result::Column;
use duckdb::Connection;
use duckdb::types::Value;
use evanalyzer_cfg::{
    core_types::{InternalErrors, ObjectClass},
    settings::classification_settings::Class,
};
use std::cell::RefCell;
use std::collections::HashMap;
use std::path::PathBuf;

const DEFAULT_GROUPING_REGEX: &str = r"^(([A-H])([0-9]{1,2}))_([0-9]+)\.([a-zA-Z0-9]+)$";

pub struct ResultsGenerator {
    database: Connection,
    classes_cache: RefCell<Option<Vec<Class>>>,
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

#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub enum Aggregation {
    #[default]
    Avg,
    Min,
    Max,
    Stddev,
    Sum,
    Median,
    Skewness,
}

#[derive(Default, Clone, PartialEq, Eq)]
pub enum ColorSchema {
    #[default]
    Excel,
    Viridis,
    Plasma,
    Inferno,
    Cividis,
    Coolwarm,
    RedBlue,
    YlGnBu,
    Haline,
    Algae,
    Thermal,
}

#[derive(Default, Clone)]
pub enum ColorScale {
    #[default]
    Auto,
    Manual(f32, f32),
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
    // Name of the group to display
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

/// Same shape as `WellFilter` minus `group_name` — `get_wells_for_plate`
/// answers for every well at once, so there's no single well to name.
#[derive(Clone)]
pub struct WellsBatchFilter {
    pub plane: PlaneFilter,
    pub grouping_regex: String,
    pub aggregation: Aggregation,
    pub object_class: ObjectClass,
    pub column: Column,
    pub color_schema: ColorSchema,
    pub color_scale: ColorScale,
    pub well_size: Option<WellSize>,
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
    pub with_coloc_details: bool,
    pub page: Pagination,
}

#[derive(Clone)]
pub struct GroupedByImageFilter {
    pub plane: PlaneFilter,
    pub images: Option<Vec<String>>,
    pub object_classes: Option<Vec<ObjectClass>>,
    pub columns: Vec<Column>,
    pub aggregation: Vec<Aggregation>,
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
    /// If true this cell should be displayed in alternating color, the ui desides on itself what is alternating
    pub alternating_color: bool,
    /// Optional search (display name, key) (Group name of plate and image_rel_path in well view)
    pub search_key: Option<(String, String)>,
}

#[derive(Clone)]
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
    /// How many source rows this page's query actually matched
    pub source_object_count: usize,
    /// For navigation
    pub row_locations: Vec<(String, [u32; 4])>,
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

    /// Raw DB handle
    pub(super) fn connection(&self) -> &Connection {
        &self.database
    }

    pub fn get_object_list(&self, filter: &ListFilter) -> Result<DatabaseResult, InternalErrors> {
        let err = |e: duckdb::Error| InternalErrors::Io(e.to_string());
        // Fetched up front (cached — see `classes_cache`) since
        // `column_names` below already needs it to resolve a `ColocCount`
        // column's class name, ahead of where it's also used to validate/
        // translate `filter.object_classes`.
        let classes = self.get_object_classes()?;
        let mut ordered_columns = filter.columns.clone();
        ordered_columns.sort();
        let mut column_names: Vec<String> = ordered_columns
            .iter()
            .map(|c| c.display_label(&classes))
            .collect();

        // `ListFilter::with_coloc_details`: every selected `ColocCount(class)`
        // column crossed with every selected plain-measurement column (see
        // `is_resolvable_metric`) adds one more header — that combination's
        // value resolved on the class's colocalizing partner object instead
        // of the source object. Needs at least one of each to mean anything;
        // with only coloc-class columns selected (no measurement to resolve)
        // or vice versa, list rows are built the same as when the flag is
        // off.
        let coloc_class_columns: Vec<ObjectClass> = ordered_columns
            .iter()
            .filter_map(|c| match c {
                Column::ColocCount(class) => Some(*class),
                _ => None,
            })
            .collect();
        let metric_columns: Vec<Column> = ordered_columns
            .iter()
            .filter(|c| is_resolvable_metric(c))
            .cloned()
            .collect();
        let details_active = filter.with_coloc_details
            && !coloc_class_columns.is_empty()
            && !metric_columns.is_empty();
        if details_active {
            for class in &coloc_class_columns {
                for metric in &metric_columns {
                    column_names.push(format!(
                        "{} coloc {}",
                        class_display_label(*class, &classes),
                        metric.display_label(&classes)
                    ));
                }
            }
        }

        let empty_result = |column_names: Vec<String>| DatabaseResult {
            column_names,
            row_names: vec![],
            rows: vec![],
            min: 0.0,
            max: 0.0,
            source_object_count: 0,
            row_locations: vec![],
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
        let needs = ObjectColumnNeeds::for_columns(&ordered_columns);
        let sql = format!(
            "SELECT {}\n FROM objects {where_clause}\n ORDER BY object_id",
            object_select_clause(needs)
        );

        let mut stmt = self.database.prepare(&sql).map_err(err)?;
        let objects: Vec<ObjectRow> = stmt
            .query_map([], map_object_row)
            .map_err(err)?
            .collect::<Result<Vec<_>, _>>()
            .map_err(err)?;

        let (row_names, rows, row_locations) = if details_active {
            self.build_coloc_detail_rows(
                &objects,
                &ordered_columns,
                &coloc_class_columns,
                &metric_columns,
                &classes,
            )?
        } else {
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
            let row_locations = objects.iter().map(object_location).collect();
            (row_names, rows, row_locations)
        };

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
            source_object_count: objects.len(),
            row_locations,
        })
    }

    /// Returns a list of (image, class) groups with the object metrics
    /// (columns) grouped by `image_rel_path` *and* `object_class_id` — an
    /// object can belong to more than one class at once (`object_class_id`
    /// is itself an array column, see evanalyzer_core's duckdb.rs), so this
    /// unnests it and produces one row per class an image actually has
    /// objects of, rather than lumping every class together into a single
    /// per-image aggregate (which would silently mix unrelated classes'
    /// values together whenever more than one class is present/selected).
    ///
    /// One output column per (`filter.columns` entry) x (`filter.aggregation`
    /// entry) — e.g. 2 columns x 3 aggregations = 6 output columns, one row
    /// per (image, class) — mirroring `ResultExport`'s flat-list export
    /// (results_exporter.rs), just exposed here as a live, paginated List
    /// view mode instead of a one-shot export. No `grouping_regex` (unlike
    /// `get_group_by_plate`/`get_group_by_well`): grouped directly by each
    /// object's own `image_rel_path`, nothing derived from it.
    pub fn get_grouped_by_image(
        &self,
        filter: &GroupedByImageFilter,
    ) -> Result<DatabaseResult, InternalErrors> {
        let err = |e: duckdb::Error| InternalErrors::Io(e.to_string());
        let classes = self.get_object_classes()?;
        // Fixed, selection-order-independent column order — same reasoning
        // as `get_object_list`'s own `ordered_columns.sort()`: `filter.columns`
        // reflects whatever order the caller happens to track selections in
        // (e.g. the GUI's toggle-on/toggle-off `Vec`), which shifts around
        // as columns are (de)selected and would otherwise reorder the
        // output out from under the user.
        let mut ordered_columns = filter.columns.clone();
        ordered_columns.sort();

        let mut column_names = vec!["image".to_string(), "class".to_string()];
        let mut value_exprs = Vec::new();
        for column in &ordered_columns {
            for aggregation in &filter.aggregation {
                let (agg_fn, value_expr) = aggregate_sql(column, aggregation)?;
                column_names.push(format!("{} ({agg_fn})", column.display_label(&classes)));
                value_exprs.push(format!(
                    "{agg_fn}({value_expr}) AS value_{}",
                    value_exprs.len()
                ));
            }
        }

        let empty_result = || DatabaseResult {
            column_names: column_names.clone(),
            row_names: vec![],
            rows: vec![],
            min: 0.0,
            max: 0.0,
            source_object_count: 0,
            row_locations: vec![],
        };
        // Nothing selected to aggregate - no query can produce a
        // meaningful answer, same as `get_object_list`'s empty-filter
        // short-circuits below.
        if value_exprs.is_empty() {
            return Ok(empty_result());
        }

        let mut conditions = vec![
            format!("z_stack = {}", filter.plane.z_stack),
            format!("t_stack = {}", filter.plane.t_stack),
        ];
        if let Some(rel_paths) = &filter.images {
            if rel_paths.is_empty() {
                return Ok(empty_result());
            }
            conditions.push(format!(
                "image_rel_path IN ({})",
                sql_string_in_list(rel_paths)
            ));
        }
        if let Some(wanted) = &filter.object_classes {
            let ids: Vec<u32> = wanted
                .iter()
                .filter_map(|id| match id {
                    ObjectClass::Valid(n) => Some(*n),
                    ObjectClass::Unset => None,
                })
                .collect();
            if ids.is_empty() {
                return Ok(empty_result());
            }
            conditions.push(format!(
                "class_id IN ({})",
                ids.iter()
                    .map(|id| id.to_string())
                    .collect::<Vec<_>>()
                    .join(", ")
            ));
        }
        // Keyset pagination over the *groups* (one (image, class) pair =
        // one row here), not over individual objects like
        // `get_object_list` - ordered the same way (`image_rel_path`, then
        // `class_id`), so a row-value comparison against the last page's
        // final group can never split a group across pages.
        if let Some(cursor) = &filter.page.after {
            let (cursor_path, cursor_class) = cursor
                .split_once('\u{1}')
                .unwrap_or((cursor.as_str(), "-1"));
            conditions.push(format!(
                "(image_rel_path, class_id) > ('{}', {})",
                cursor_path.replace('\'', "''"),
                cursor_class.parse::<i64>().unwrap_or(-1)
            ));
        }
        let where_clause = format!("WHERE {}", conditions.join(" AND "));
        let limit = filter.page.limit.max(0);
        let value_exprs_sql = value_exprs.join(",\n                ");

        // `object_class_id` is a per-object array (multi-class objects
        // exist), so it's unnested into one `class_id` per (object, class)
        // pair *before* grouping — an object with no class at all
        // (`object_class_id = []`) contributes no `class_id` row and so
        // never appears in the output, same as every other view in this
        // file that filters by class rather than treating "no class" as a
        // group of its own.
        let sql = format!(
            "SELECT\n\
                image_rel_path,\n\
                MIN(image_name) AS image_name,\n\
                class_id,\n\
                {value_exprs_sql}\n\
             FROM objects, UNNEST(CAST(object_class_id AS INTEGER[])) AS u(class_id)\n\
             {where_clause}\n\
             GROUP BY image_rel_path, class_id\n\
             ORDER BY image_rel_path, class_id\n\
             LIMIT {limit}"
        );

        let mut stmt = self.database.prepare(&sql).map_err(err)?;
        let n = value_exprs.len();
        let groups: Vec<(String, String, u32, Vec<Option<f64>>)> = stmt
            .query_map([], |row| {
                let image_rel_path: String = row.get(0)?;
                let image_name: String = row.get(1)?;
                let class_id: u32 = row.get(2)?;
                let mut values = Vec::with_capacity(n);
                for i in 0..n {
                    values.push(row.get::<_, Option<f64>>(3 + i)?);
                }
                Ok((image_rel_path, image_name, class_id, values))
            })
            .map_err(err)?
            .collect::<Result<Vec<_>, _>>()
            .map_err(err)?;

        let mut min = f64::INFINITY;
        let mut max = f64::NEG_INFINITY;
        for (_, _, _, values) in &groups {
            for value in values.iter().flatten() {
                min = min.min(*value);
                max = max.max(*value);
            }
        }
        if !min.is_finite() || !max.is_finite() {
            min = 0.0;
            max = 0.0;
        }

        let row_names = groups
            .iter()
            .map(|(rel_path, _, class_id, _)| format!("{rel_path}\u{1}{class_id}"))
            .collect();
        let source_object_count = groups.len();
        let rows: Vec<Vec<Cell>> = groups
            .into_iter()
            .map(|(image_rel_path, image_name, class_id, values)| {
                // Lets the GUI navigate straight to the source image, same
                // as any other image-bearing search key in this file.
                let search_key = Some((image_name.clone(), image_rel_path));
                let object_class = ObjectClass::Valid(class_id);
                let label = class_display_label(object_class, &classes);
                let color = classes
                    .iter()
                    .find(|class| class.id == object_class)
                    .map(|class| class.color)
                    .unwrap_or(0);
                let mut cells = vec![
                    Cell {
                        value: CellValue::String(image_name),
                        bg_color: 0,
                        alternating_color: false,
                        search_key: search_key.clone(),
                    },
                    Cell {
                        value: CellValue::Class((label, color)),
                        bg_color: color,
                        alternating_color: false,
                        search_key: search_key.clone(),
                    },
                ];
                for value in values {
                    cells.push(Cell {
                        value: CellValue::Float(value.unwrap_or(0.0) as f32),
                        bg_color: 0,
                        alternating_color: false,
                        search_key: search_key.clone(),
                    });
                }
                cells
            })
            .collect();

        Ok(DatabaseResult {
            column_names,
            row_names,
            rows,
            min: min as f32,
            max: max as f32,
            source_object_count,
            row_locations: Vec::new(),
        })
    }

    // `ListFilter::with_coloc_details`: fan out each source object into one
    // row per (selected coloc-class, colocalizing partner) pair, appending
    // one resolved-metric cell per `coloc_class_columns` x `metric_columns`
    // combination — see the confirmed design on `ListFilter::with_coloc_details`
    // above `get_list`. Independent per-class fan-out: a row built from a
    // partner of class A shows "-" for every other selected class's metric
    // cells even if that other class also has real partners elsewhere for
    // the same source object (those get their own separate rows instead).
    fn build_coloc_detail_rows(
        &self,
        objects: &[ObjectRow],
        ordered_columns: &[Column],
        coloc_class_columns: &[ObjectClass],
        metric_columns: &[Column],
        classes: &[Class],
    ) -> Result<(Vec<String>, Vec<Vec<Cell>>, Vec<(String, [u32; 4])>), InternalErrors> {
        let err = |e: duckdb::Error| InternalErrors::Io(e.to_string());
        let dash = |alternating_color: bool| Cell {
            value: CellValue::String("-".to_string()),
            bg_color: 0,
            alternating_color,
            search_key: None,
        };

        let mut partner_ids: std::collections::HashSet<String> = std::collections::HashSet::new();
        let mut partners_by_object: HashMap<String, HashMap<ObjectClass, Vec<String>>> =
            HashMap::new();
        for object in objects {
            let parsed: Option<serde_json::Value> = serde_json::from_str(&object.coloc_json).ok();
            let mut per_class: HashMap<ObjectClass, Vec<String>> = HashMap::new();
            for class in coloc_class_columns {
                let key = coloc_class_key(*class);
                let ids: Vec<String> = parsed
                    .as_ref()
                    .and_then(|value| value.get(&key))
                    .and_then(|value| value.as_array())
                    .map(|arr| {
                        arr.iter()
                            .filter_map(|value| value.as_str().map(str::to_string))
                            .collect()
                    })
                    .unwrap_or_default();
                partner_ids.extend(ids.iter().cloned());
                per_class.insert(*class, ids);
            }
            partners_by_object.insert(object.object_id.clone(), per_class);
        }

        let partner_rows: HashMap<String, ObjectRow> = if partner_ids.is_empty() {
            HashMap::new()
        } else {
            let ids: Vec<String> = partner_ids.into_iter().collect();
            let partner_needs = ObjectColumnNeeds::for_columns(metric_columns);
            let partner_sql = format!(
                "SELECT {} FROM objects WHERE object_id IN ({})",
                object_select_clause(partner_needs),
                sql_string_in_list(&ids)
            );
            let mut partner_stmt = self.database.prepare(&partner_sql).map_err(err)?;
            partner_stmt
                .query_map([], map_object_row)
                .map_err(err)?
                .collect::<Result<Vec<_>, _>>()
                .map_err(err)?
                .into_iter()
                .map(|row| (row.object_id.clone(), row))
                .collect()
        };

        let mut row_names = Vec::new();
        let mut rows = Vec::new();
        let mut row_locations = Vec::new();
        // Flips every time the source object changes (not every fanned-out
        // row) so every row belonging to the same source object shares one
        // shade and neighboring objects alternate — lets the GUI shade by
        // object group instead of by row parity, which would cut a group in
        // half arbitrarily whenever it has an even number of partner rows.
        let mut alternating_color = false;
        for object in objects {
            alternating_color = !alternating_color;
            let per_class_partners = &partners_by_object[&object.object_id];
            let mut fan_specs: Vec<(ObjectClass, Option<&str>)> = Vec::new();
            for class in coloc_class_columns {
                for partner_id in &per_class_partners[class] {
                    fan_specs.push((*class, Some(partner_id.as_str())));
                }
            }
            if fan_specs.is_empty() {
                fan_specs.push((coloc_class_columns[0], None));
            }

            for (active_class, partner_id) in fan_specs {
                let mut cells: Vec<Cell> = ordered_columns
                    .iter()
                    .map(|column| {
                        let mut cell = cell_for_column(column, object, classes);
                        cell.alternating_color = alternating_color;
                        cell
                    })
                    .collect();
                for class in coloc_class_columns {
                    for metric in metric_columns {
                        let mut cell = if *class == active_class {
                            partner_id
                                .and_then(|id| partner_rows.get(id))
                                .map(|partner_row| cell_for_column(metric, partner_row, classes))
                                .unwrap_or_else(|| dash(alternating_color))
                        } else {
                            dash(alternating_color)
                        };
                        cell.alternating_color = alternating_color;
                        cells.push(cell);
                    }
                }
                row_names.push(object.object_id.clone());
                rows.push(cells);
                // The row is fanned out over this object's colocalizing
                // partners, but it's still fundamentally a row *about* the
                // source object — navigating from it should go to the
                // source object's own location, not a partner's.
                row_locations.push(object_location(object));
            }
        }
        Ok((row_names, rows, row_locations))
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
        let (agg_fn, value_expr) = aggregate_sql(&filter.column, &filter.aggregation)?;

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
        let groups: Vec<(String, String, String, Option<f64>)> = stmt
            .query_map([], |row| {
                Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?))
            })
            .map_err(err)?
            .collect::<Result<Vec<_>, _>>()
            .map_err(err)?;

        Ok(plate_groups_to_result(
            groups,
            &filter.column,
            &classes,
            filter.matrix_dimension,
            &filter.color_schema,
            &filter.color_scale,
            view,
        ))
    }

    // Batched form of `get_group_by_plate` across every requested
    // aggregation at once, in one query instead of one per aggregation —
    // same reasoning as `get_wells_for_plate` batching across wells: the
    // WHERE/GROUP BY here is identical for every aggregation, only the
    // aggregate function itself differs, so computing e.g. AVG, MIN, MAX,
    // SUM, STDDEV, MEDIAN and SKEWNESS of the same column all in one SELECT
    // means one full scan instead of seven. Returns one `DatabaseResult` per
    // `aggregations` entry, same order.
    pub fn get_group_by_plate_multi_agg(
        &self,
        filter: &PlateFilter,
        aggregations: &[Aggregation],
        view: &View,
    ) -> Result<Vec<DatabaseResult>, InternalErrors> {
        let err = |e: duckdb::Error| InternalErrors::Io(e.to_string());
        let classes = self.get_object_classes()?;

        let mut value_exprs = Vec::with_capacity(aggregations.len());
        for aggregation in aggregations {
            let (agg_fn, value_expr) = aggregate_sql(&filter.column, aggregation)?;
            value_exprs.push(format!(
                "{agg_fn}({value_expr}) AS value_{}",
                value_exprs.len()
            ));
        }
        let value_exprs_sql = value_exprs.join(",\n                ");

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
                {value_exprs_sql}\n\
             FROM objects\n\
             {where_clause}\n\
             GROUP BY group_prefix, row, col\n\
             ORDER BY group_prefix",
            regex = regex.replace('\'', "''"),
        );

        let mut stmt = self.database.prepare(&sql).map_err(err)?;
        let n = aggregations.len();
        let raw: Vec<(String, String, String, Vec<Option<f64>>)> = stmt
            .query_map([], |row| {
                let group_prefix: String = row.get(0)?;
                let group_row: String = row.get(1)?;
                let group_col: String = row.get(2)?;
                let mut values = Vec::with_capacity(n);
                for i in 0..n {
                    values.push(row.get::<_, Option<f64>>(3 + i)?);
                }
                Ok((group_prefix, group_row, group_col, values))
            })
            .map_err(err)?
            .collect::<Result<Vec<_>, _>>()
            .map_err(err)?;

        Ok((0..n)
            .map(|i| {
                let groups: Vec<(String, String, String, Option<f64>)> = raw
                    .iter()
                    .map(|(g, r, c, values)| (g.clone(), r.clone(), c.clone(), values[i]))
                    .collect();
                plate_groups_to_result(
                    groups,
                    &filter.column,
                    &classes,
                    filter.matrix_dimension,
                    &filter.color_schema,
                    &filter.color_scale,
                    view,
                )
            })
            .collect())
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
        let (agg_fn, value_expr) = aggregate_sql(&filter.column, &filter.aggregation)?;

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

        Ok(well_fields_to_result(
            fields,
            &filter.column,
            &classes,
            filter.well_size,
            &filter.well_order,
            &filter.color_schema,
            &filter.color_scale,
            view,
        ))
    }

    // Batched form of `get_group_by_well`: every well's fields in one query
    // (grouped by well *and* field, rather than one query per well behind a
    // `WHERE ... = '{group_name}'` filter) — the well filter can't use an
    // index (it's a `regexp_extract` match per row), so calling
    // `get_group_by_well` once per well means scanning the whole table once
    // per well. An export iterating every well for every (class, column,
    // aggregation) combination turns that into thousands of full scans;
    // this does the same work with exactly one scan per (class, column,
    // aggregation) instead, by asking for every well's answer at once and
    // partitioning the single result set client-side. Keyed by well/group
    // id (e.g. "A1"), matching `WellFilter::group_name`.
    pub fn get_wells_for_plate(
        &self,
        filter: &WellsBatchFilter,
        view: &View,
    ) -> Result<HashMap<String, DatabaseResult>, InternalErrors> {
        let err = |e: duckdb::Error| InternalErrors::Io(e.to_string());
        let classes = self.get_object_classes()?;
        let (agg_fn, value_expr) = aggregate_sql(&filter.column, &filter.aggregation)?;

        let regex = if filter.grouping_regex.trim().is_empty() {
            DEFAULT_GROUPING_REGEX
        } else {
            filter.grouping_regex.as_str()
        };
        let regex = regex.replace('\'', "''");

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
                regexp_extract(image_name, '{regex}', 4) AS idx,\n\
                image_rel_path,\n\
                image_name,\n\
                {agg_fn}({value_expr}) AS value\n\
             FROM objects\n\
             {where_clause}\n\
             GROUP BY group_prefix, idx, image_rel_path, image_name\n\
             ORDER BY group_prefix, idx"
        );

        let mut stmt = self.database.prepare(&sql).map_err(err)?;
        let mut fields_by_well: HashMap<String, Vec<(String, String, String, Option<f64>)>> =
            HashMap::new();
        let rows = stmt
            .query_map([], |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, String>(2)?,
                    row.get::<_, String>(3)?,
                    row.get::<_, Option<f64>>(4)?,
                ))
            })
            .map_err(err)?
            .collect::<Result<Vec<_>, _>>()
            .map_err(err)?;
        for (group_prefix, idx, image_rel_path, image_name, value) in rows {
            fields_by_well.entry(group_prefix).or_default().push((
                idx,
                image_rel_path,
                image_name,
                value,
            ));
        }

        Ok(fields_by_well
            .into_iter()
            .map(|(well_id, fields)| {
                let result = well_fields_to_result(
                    fields,
                    &filter.column,
                    &classes,
                    filter.well_size,
                    &filter.well_order,
                    &filter.color_schema,
                    &filter.color_scale,
                    view,
                );
                (well_id, result)
            })
            .collect())
    }

    // Batched form of `get_wells_for_plate` across every requested
    // aggregation at once (same one-query-instead-of-N reasoning as
    // `get_group_by_plate_multi_agg`) — combined with `get_wells_for_plate`'s
    // own per-well batching, this is the difference between, say, 54 wells
    // × 7 aggregations = 378 full scans and exactly 1, for one (class,
    // column) combination. Returns one `HashMap<well_id, DatabaseResult>`
    // per `aggregations` entry, same order.
    pub fn get_wells_for_plate_multi_agg(
        &self,
        filter: &WellsBatchFilter,
        aggregations: &[Aggregation],
        view: &View,
    ) -> Result<Vec<HashMap<String, DatabaseResult>>, InternalErrors> {
        let err = |e: duckdb::Error| InternalErrors::Io(e.to_string());
        let classes = self.get_object_classes()?;

        let mut value_exprs = Vec::with_capacity(aggregations.len());
        for aggregation in aggregations {
            let (agg_fn, value_expr) = aggregate_sql(&filter.column, aggregation)?;
            value_exprs.push(format!(
                "{agg_fn}({value_expr}) AS value_{}",
                value_exprs.len()
            ));
        }
        let value_exprs_sql = value_exprs.join(",\n                ");

        let regex = if filter.grouping_regex.trim().is_empty() {
            DEFAULT_GROUPING_REGEX
        } else {
            filter.grouping_regex.as_str()
        };
        let regex = regex.replace('\'', "''");

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
                regexp_extract(image_name, '{regex}', 4) AS idx,\n\
                image_rel_path,\n\
                image_name,\n\
                {value_exprs_sql}\n\
             FROM objects\n\
             {where_clause}\n\
             GROUP BY group_prefix, idx, image_rel_path, image_name\n\
             ORDER BY group_prefix, idx"
        );

        let mut stmt = self.database.prepare(&sql).map_err(err)?;
        let n = aggregations.len();
        let mut fields_by_well: HashMap<String, Vec<(String, String, String, Vec<Option<f64>>)>> =
            HashMap::new();
        let rows = stmt
            .query_map([], |row| {
                let group_prefix: String = row.get(0)?;
                let idx: String = row.get(1)?;
                let image_rel_path: String = row.get(2)?;
                let image_name: String = row.get(3)?;
                let mut values = Vec::with_capacity(n);
                for i in 0..n {
                    values.push(row.get::<_, Option<f64>>(4 + i)?);
                }
                Ok((group_prefix, idx, image_rel_path, image_name, values))
            })
            .map_err(err)?
            .collect::<Result<Vec<_>, _>>()
            .map_err(err)?;
        for (group_prefix, idx, image_rel_path, image_name, values) in rows {
            fields_by_well.entry(group_prefix).or_default().push((
                idx,
                image_rel_path,
                image_name,
                values,
            ));
        }

        Ok((0..n)
            .map(|agg_idx| {
                fields_by_well
                    .iter()
                    .map(|(well_id, fields)| {
                        let per_agg_fields: Vec<(String, String, String, Option<f64>)> = fields
                            .iter()
                            .map(|(idx, rel_path, name, values)| {
                                (idx.clone(), rel_path.clone(), name.clone(), values[agg_idx])
                            })
                            .collect();
                        let result = well_fields_to_result(
                            per_agg_fields,
                            &filter.column,
                            &classes,
                            filter.well_size,
                            &filter.well_order,
                            &filter.color_schema,
                            &filter.color_scale,
                            view,
                        );
                        (well_id.clone(), result)
                    })
                    .collect::<HashMap<String, DatabaseResult>>()
            })
            .collect())
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
        let (agg_fn, value_expr) = aggregate_sql(&filter.column, &filter.aggregation)?;
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

                let column_names =
                    vec!["square".to_string(), filter.column.display_label(&classes)];
                let row_names = sorted
                    .iter()
                    .map(|((row, col), _)| format!("R{row}C{col}"))
                    .collect();
                let rows_out: Vec<Vec<Cell>> = sorted
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
                                alternating_color: false,
                                search_key: search_key.clone(),
                            },
                            Cell {
                                value: CellValue::Float(*value as f32),
                                bg_color: 0,
                                alternating_color: false,
                                search_key,
                            },
                        ]
                    })
                    .collect();
                let source_object_count = rows_out.len();
                Ok(DatabaseResult {
                    column_names,
                    row_names,
                    rows: rows_out,
                    min: min as f32,
                    max: max as f32,
                    source_object_count,
                    row_locations: Vec::new(),
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

                let grid_rows: Vec<Vec<Cell>> = (0..rows)
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
                                        alternating_color: false,
                                        search_key: Some((key.clone(), key)),
                                    }
                                }
                                // No object fell into this tile at all —
                                // leave it empty rather than showing a
                                // misleading 0 or a neighboring tile's value.
                                None => Cell {
                                    value: CellValue::Empty,
                                    bg_color: 0,
                                    alternating_color: false,
                                    search_key: None,
                                },
                            })
                            .collect()
                    })
                    .collect();

                let source_object_count = grid_rows.len();
                Ok(DatabaseResult {
                    column_names: (0..cols).map(|col| col.to_string()).collect(),
                    row_names: (0..rows).map(|row| row.to_string()).collect(),
                    rows: grid_rows,
                    min: range_min as f32,
                    max: range_max as f32,
                    source_object_count,
                    row_locations: Vec::new(),
                })
            }
        }
    }

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
        let classes = self.get_object_classes()?;
        let entry = |key: Column, group: &str| ColumnEntry {
            display_name: key.display_label(&classes),
            key,
            group: group.into(),
        };
        let mut ret = vec![
            entry(Column::ObjectId, "General"),
            entry(Column::ImageName, "General"),
            entry(Column::ObjectClass, "General"),
            entry(Column::Count, "General"),
            entry(Column::AreaSizePx, "Geometry"),
            entry(Column::AreaSizeNm, "Geometry"),
            entry(Column::PerimeterPx, "Geometry"),
            entry(Column::PerimeterNm, "Geometry"),
            entry(Column::Circularity, "Shape"),
            entry(Column::Solidity, "Shape"),
            entry(Column::Eccentricity, "Shape"),
        ];

        // Coloc count is measured per candidate partner class (like
        // intensity is measured per channel below), so there's one column
        // per class that actually shows up as a colocalization partner
        // somewhere in this database, rather than a single shared "total
        // across every class" column.
        for class_id in self.get_object_classes_with_at_least_coloc()? {
            ret.push(entry(Column::ColocCount(class_id), "Coloc"));
        }

        // Intensity is measured per image channel, so there's one Avg/Sum/
        // Min/Max column per channel rather than a single shared one.
        for channel in 0..self.get_nr_of_c_stacks() {
            ret.push(entry(Column::IntensityAvg(channel), "intensity"));
            ret.push(entry(Column::IntensitySum(channel), "intensity"));
            ret.push(entry(Column::IntensityMin(channel), "intensity"));
            ret.push(entry(Column::IntensityMax(channel), "intensity"));
        }

        Ok(ret)
    }

    pub fn get_nr_of_c_stacks(&self) -> u32 {
        let max_stacks: u32 = self
            .database
            .query_row("SELECT MAX(c_stacks) FROM images;", [], |row| row.get(0))
            .unwrap_or(1);
        max_stacks
    }

    pub fn get_nr_of_z_stacks(&self) -> u32 {
        let max_stack: u32 = self
            .database
            .query_row("SELECT MAX(z_stacks) FROM images;", [], |row| row.get(0))
            .unwrap_or(1);
        max_stack
    }

    pub fn get_nr_of_t_stacks(&self) -> u32 {
        let max_stack: u32 = self
            .database
            .query_row("SELECT MAX(t_stacks) FROM images;", [], |row| row.get(0))
            .unwrap_or(1);
        max_stack
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Builds a `ResultsGenerator` over an in-memory database with just the
    /// columns `get_available_columns`/`get_nr_of_c_stacks` actually touch.
    /// `objects.c_stack` is deliberately fixed at `0` for every row (as real
    /// pipeline extraction does, see `extract_objects.rs`) to prove channel
    /// discovery no longer depends on it — the real source is `images.c_stacks`,
    /// populated per image from real metadata (see `finalize_image` in
    /// evanalyzer_core's duckdb.rs), so this takes one `c_stacks` value per
    /// "image" row directly rather than inferring it from object data.
    fn generator_with_c_stacks(c_stacks_per_image: &[u32]) -> ResultsGenerator {
        let database = Connection::open_in_memory().unwrap();
        database
            .execute_batch(
                "CREATE TABLE classes (class_id INTEGER, name VARCHAR, color UINTEGER);
                 CREATE TABLE objects (
                     c_stack INTEGER,
                     coloc_json JSON,
                     intensities_json JSON
                 );
                 CREATE TABLE images (
                     image_rel_path VARCHAR,
                     c_stacks UINTEGER
                 );",
            )
            .unwrap();
        for (i, c_stacks) in c_stacks_per_image.iter().enumerate() {
            database
                .execute(
                    "INSERT INTO images (image_rel_path, c_stacks) VALUES (?, ?)",
                    duckdb::params![format!("image_{i}.tif"), c_stacks],
                )
                .unwrap();
        }
        ResultsGenerator {
            database,
            classes_cache: RefCell::new(None),
            coloc_classes_cache: RefCell::new(None),
        }
    }

    #[test]
    fn nr_of_c_stacks_reads_the_real_per_image_channel_count() {
        let generator = generator_with_c_stacks(&[3]);
        assert_eq!(generator.get_nr_of_c_stacks(), 3);
    }

    #[test]
    fn nr_of_c_stacks_is_the_max_across_every_image() {
        let generator = generator_with_c_stacks(&[1, 3, 2]);
        assert_eq!(generator.get_nr_of_c_stacks(), 3);
    }

    #[test]
    fn available_columns_include_one_intensity_group_per_measured_channel() {
        let generator = generator_with_c_stacks(&[3]);
        let columns = generator.get_available_columns().unwrap();
        let intensity_avg_columns = columns
            .iter()
            .filter(|c| matches!(c.key, Column::IntensityAvg(_)))
            .count();
        assert_eq!(intensity_avg_columns, 3);
    }

    #[test]
    fn nr_of_c_stacks_falls_back_to_one_when_images_table_is_empty() {
        let generator = generator_with_c_stacks(&[]);
        assert_eq!(generator.get_nr_of_c_stacks(), 1);
    }

    use crate::results::test_support::{ObjectSpec, seed_db};

    /// Opens a fresh `ResultsGenerator` over a temp `.evadb` seeded with
    /// `objects` (see `test_support::seed_db`). Leaks the backing `TempDir`
    /// (the returned generator only holds an open `Connection`, not the
    /// directory) - acceptable for a short-lived test process.
    fn open(objects: &[ObjectSpec]) -> ResultsGenerator {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("results.evadb");
        seed_db(&path, objects);
        std::mem::forget(dir);
        ResultsGenerator::open_database(path.into()).unwrap()
    }

    fn plane() -> PlaneFilter {
        PlaneFilter {
            z_stack: 0,
            t_stack: 0,
        }
    }

    /// A limit generous enough to fetch every row a test seeds in one page
    /// — unlike `fetch_all_list_rows`/`fetch_all_grouped_by_image_rows`
    /// (results_exporter.rs), which page through `get_object_list`/
    /// `get_grouped_by_image` themselves and only ever pass `limit: 0` as a
    /// template immediately overwritten before use, a direct call here with
    /// `limit: 0` would produce a literal `LIMIT 0` — zero rows.
    fn no_page() -> Pagination {
        Pagination {
            limit: 1000,
            after: None,
        }
    }

    // -- get_object_list --------------------------------------------------

    #[test]
    fn get_object_list_returns_one_row_per_object_with_requested_columns() {
        let generator = open(&[
            ObjectSpec::new("img1.tif", "ClassA", 1, 100),
            ObjectSpec::new("img2.tif", "ClassB", 2, 200),
        ]);
        let result = generator
            .get_object_list(&ListFilter {
                plane: plane(),
                images: None,
                object_classes: None,
                columns: vec![Column::ObjectClass, Column::AreaSizePx],
                with_coloc_details: false,
                page: no_page(),
            })
            .unwrap();

        assert_eq!(
            result.column_names,
            vec!["Class".to_string(), "Area [px]".to_string()]
        );
        assert_eq!(result.rows.len(), 2);
        assert_eq!(result.source_object_count, 2);
        assert_eq!(result.row_names.len(), 2);
        assert_eq!(result.row_locations.len(), 2);
    }

    #[test]
    fn get_object_list_image_filter_restricts_to_the_selected_image() {
        let generator = open(&[
            ObjectSpec::new("img1.tif", "ClassA", 1, 100),
            ObjectSpec::new("img2.tif", "ClassB", 2, 200),
        ]);
        let result = generator
            .get_object_list(&ListFilter {
                plane: plane(),
                images: Some(vec!["img1.tif".to_string()]),
                object_classes: None,
                columns: vec![Column::ImageName],
                with_coloc_details: false,
                page: no_page(),
            })
            .unwrap();
        assert_eq!(result.source_object_count, 1);
    }

    #[test]
    fn get_object_list_class_filter_restricts_to_the_selected_class() {
        let generator = open(&[
            ObjectSpec::new("img1.tif", "ClassA", 1, 100),
            ObjectSpec::new("img1.tif", "ClassB", 2, 200),
        ]);
        let result = generator
            .get_object_list(&ListFilter {
                plane: plane(),
                images: None,
                object_classes: Some(vec![ObjectClass::Valid(2)]),
                columns: vec![Column::ObjectClass],
                with_coloc_details: false,
                page: no_page(),
            })
            .unwrap();
        assert_eq!(result.source_object_count, 1);
    }

    #[test]
    fn get_object_list_with_an_explicitly_empty_image_selection_is_empty() {
        let generator = open(&[ObjectSpec::new("img1.tif", "ClassA", 1, 100)]);
        let result = generator
            .get_object_list(&ListFilter {
                plane: plane(),
                images: Some(vec![]),
                object_classes: None,
                columns: vec![Column::ObjectClass],
                with_coloc_details: false,
                page: no_page(),
            })
            .unwrap();
        assert_eq!(result.source_object_count, 0);
        assert!(result.rows.is_empty());
    }

    #[test]
    fn get_object_list_plane_filter_excludes_objects_on_other_planes() {
        let generator = open(&[
            ObjectSpec::new("img1.tif", "ClassA", 1, 100),
            ObjectSpec::new("img1.tif", "ClassA", 1, 200).at_plane(1, 0),
        ]);
        let result = generator
            .get_object_list(&ListFilter {
                plane: PlaneFilter {
                    z_stack: 1,
                    t_stack: 0,
                },
                images: None,
                object_classes: None,
                columns: vec![Column::AreaSizePx],
                with_coloc_details: false,
                page: no_page(),
            })
            .unwrap();
        assert_eq!(result.source_object_count, 1);
    }

    #[test]
    fn get_object_list_pagination_cursor_returns_the_next_page() {
        let objects: Vec<ObjectSpec> = (0..5)
            .map(|i| ObjectSpec::new("img1.tif", "ClassA", 1, i))
            .collect();
        let generator = open(&objects);

        let first_page = generator
            .get_object_list(&ListFilter {
                plane: plane(),
                images: None,
                object_classes: None,
                columns: vec![Column::AreaSizePx],
                with_coloc_details: false,
                page: Pagination {
                    limit: 2,
                    after: None,
                },
            })
            .unwrap();
        assert_eq!(first_page.rows.len(), 2);
        let cursor = first_page.row_names.last().cloned();

        let second_page = generator
            .get_object_list(&ListFilter {
                plane: plane(),
                images: None,
                object_classes: None,
                columns: vec![Column::AreaSizePx],
                with_coloc_details: false,
                page: Pagination {
                    limit: 2,
                    after: cursor.clone(),
                },
            })
            .unwrap();
        assert_eq!(second_page.rows.len(), 2);
        assert_ne!(
            first_page.row_names, second_page.row_names,
            "second page must not repeat the first page's rows"
        );
        assert!(
            !second_page.row_names.contains(cursor.as_ref().unwrap()),
            "the cursor row itself must not repeat on the next page"
        );
    }

    #[test]
    fn get_object_list_intensity_column_reads_the_seeded_channel_value() {
        use crate::results::test_support::CH0_INTENSITIES_JSON;
        let generator =
            open(&[ObjectSpec::new("img1.tif", "ClassA", 1, 100)
                .with_intensities(CH0_INTENSITIES_JSON)]);
        let result = generator
            .get_object_list(&ListFilter {
                plane: plane(),
                images: None,
                object_classes: None,
                columns: vec![Column::IntensityAvg(0)],
                with_coloc_details: false,
                page: no_page(),
            })
            .unwrap();
        assert_eq!(result.rows.len(), 1);
        match &result.rows[0][0].value {
            CellValue::Float(v) => assert_eq!(*v, 127.0),
            _ => panic!("expected a float cell"),
        }
    }

    // -- get_grouped_by_image ---------------------------------------------

    #[test]
    fn get_grouped_by_image_averages_the_selected_column_per_image_and_class() {
        let generator = open(&[
            ObjectSpec::new("img1.tif", "ClassA", 1, 10),
            ObjectSpec::new("img1.tif", "ClassA", 1, 20),
            ObjectSpec::new("img2.tif", "ClassA", 1, 100),
        ]);
        let result = generator
            .get_grouped_by_image(&GroupedByImageFilter {
                plane: plane(),
                images: None,
                object_classes: None,
                columns: vec![Column::AreaSizePx],
                aggregation: vec![Aggregation::Avg],
                page: no_page(),
            })
            .unwrap();

        assert_eq!(result.rows.len(), 2, "one row per (image, class) group");
        assert_eq!(result.column_names[0], "image");
        assert_eq!(result.column_names[1], "class");
        assert_eq!(result.column_names[2], "Area [px] (AVG)");
    }

    #[test]
    fn get_grouped_by_image_with_no_columns_selected_is_empty() {
        let generator = open(&[ObjectSpec::new("img1.tif", "ClassA", 1, 10)]);
        let result = generator
            .get_grouped_by_image(&GroupedByImageFilter {
                plane: plane(),
                images: None,
                object_classes: None,
                columns: vec![],
                aggregation: vec![Aggregation::Avg],
                page: no_page(),
            })
            .unwrap();
        assert!(result.rows.is_empty());
    }

    #[test]
    fn get_grouped_by_image_with_an_explicitly_empty_image_selection_is_empty() {
        let generator = open(&[ObjectSpec::new("img1.tif", "ClassA", 1, 10)]);
        let result = generator
            .get_grouped_by_image(&GroupedByImageFilter {
                plane: plane(),
                images: Some(vec![]),
                object_classes: None,
                columns: vec![Column::AreaSizePx],
                aggregation: vec![Aggregation::Avg],
                page: no_page(),
            })
            .unwrap();
        assert!(result.rows.is_empty());
    }

    // -- get_group_by_plate -------------------------------------------------

    #[test]
    fn get_group_by_plate_groups_by_the_default_regex_and_averages_per_well() {
        // Default regex expects `<well>_<field>.<ext>`, e.g. "A1_01.tif".
        let generator = open(&[
            ObjectSpec::new("A1_01.tif", "ClassA", 1, 10),
            ObjectSpec::new("A1_02.tif", "ClassA", 1, 20),
            ObjectSpec::new("B2_01.tif", "ClassA", 1, 100),
        ]);
        let result = generator
            .get_group_by_plate(
                &PlateFilter {
                    plane: plane(),
                    grouping_regex: String::new(),
                    aggregation: Aggregation::Avg,
                    object_class: ObjectClass::Unset,
                    column: Column::AreaSizePx,
                    color_schema: ColorSchema::default(),
                    color_scale: ColorScale::default(),
                    matrix_dimension: None,
                },
                &View::List,
            )
            .unwrap();

        assert_eq!(result.row_names.len(), 2);
        let a1_idx = result
            .row_names
            .iter()
            .position(|n| n == "A1")
            .expect("A1 group");
        let b2_idx = result
            .row_names
            .iter()
            .position(|n| n == "B2")
            .expect("B2 group");
        let cell_f64 = |cell: &Cell| match &cell.value {
            CellValue::Float(v) => *v as f64,
            _ => panic!("expected a float cell"),
        };
        assert_eq!(cell_f64(&result.rows[a1_idx][1]), 15.0, "avg(10, 20)");
        assert_eq!(cell_f64(&result.rows[b2_idx][1]), 100.0);
    }

    // -- get_images ---------------------------------------------------------

    #[test]
    fn get_images_returns_every_seeded_image_sorted_by_name() {
        let generator = open(&[
            ObjectSpec::new("b.tif", "ClassA", 1, 10),
            ObjectSpec::new("a.tif", "ClassA", 1, 10),
        ]);
        let images = generator.get_images().unwrap();
        let names: Vec<String> = images.iter().map(|i| i.name.clone()).collect();
        assert_eq!(names, vec!["a.tif".to_string(), "b.tif".to_string()]);
        assert!(images.iter().all(|i| !i.disabled));
    }

    // -- Column key/label round trips --------------------------------------

    #[test]
    fn column_as_key_and_from_key_round_trip_for_plain_columns() {
        let classes: Vec<Class> = vec![];
        for column in [
            Column::ObjectId,
            Column::ImageName,
            Column::ObjectClass,
            Column::Count,
            Column::AreaSizePx,
            Column::AreaSizeNm,
            Column::PerimeterPx,
            Column::PerimeterNm,
            Column::Circularity,
            Column::Solidity,
            Column::Eccentricity,
        ] {
            let key = column.as_key(&classes);
            assert_eq!(Column::from_key(&key, &classes), Some(column));
        }
    }

    #[test]
    fn column_as_key_and_from_key_round_trip_for_intensity_channels() {
        let classes: Vec<Class> = vec![];
        for column in [
            Column::IntensityAvg(0),
            Column::IntensitySum(1),
            Column::IntensityMin(2),
            Column::IntensityMax(3),
        ] {
            let key = column.as_key(&classes);
            assert_eq!(Column::from_key(&key, &classes), Some(column));
        }
    }

    #[test]
    fn column_as_key_and_from_key_round_trip_for_coloc_count() {
        let classes = vec![Class {
            id: ObjectClass::Valid(1),
            name: "ClassA".to_string(),
            color: 0,
            notes: String::new(),
        }];
        let column = Column::ColocCount(ObjectClass::Valid(1));
        let key = column.as_key(&classes);
        assert_eq!(key, "n_colocalized_class_ClassA");
        assert_eq!(Column::from_key(&key, &classes), Some(column));

        let unset = Column::ColocCount(ObjectClass::Unset);
        assert_eq!(unset.as_key(&classes), "n_colocalized_unset");
        assert_eq!(
            Column::from_key("n_colocalized_unset", &classes),
            Some(unset)
        );
    }

    #[test]
    fn column_from_key_returns_none_for_an_unrecognized_key() {
        assert_eq!(Column::from_key("not_a_real_column", &[]), None);
    }

    #[test]
    fn column_display_label_is_stable_and_distinct_from_its_key() {
        let classes: Vec<Class> = vec![];
        assert_eq!(Column::AreaSizePx.display_label(&classes), "Area [px]");
        assert_eq!(Column::AreaSizePx.as_key(&classes), "area_px");
        assert_eq!(
            Column::IntensityAvg(2).display_label(&classes),
            "Avg Intensity (Ch 2)"
        );
    }

    // -- class_display_label -------------------------------------------------

    #[test]
    fn class_display_label_falls_back_to_a_generic_name_for_an_unknown_class_id() {
        let classes: Vec<Class> = vec![];
        assert_eq!(
            class_display_label(ObjectClass::Valid(7), &classes),
            "class 7"
        );
        assert_eq!(class_display_label(ObjectClass::Unset, &classes), "unset");
    }

    #[test]
    fn class_display_label_uses_the_registered_class_name_when_known() {
        let classes = vec![Class {
            id: ObjectClass::Valid(1),
            name: "Nuclei".to_string(),
            color: 0,
            notes: String::new(),
        }];
        assert_eq!(
            class_display_label(ObjectClass::Valid(1), &classes),
            "Nuclei"
        );
    }

    // -- get_available_columns -------------------------------------------------

    #[test]
    fn get_available_columns_only_lists_coloc_count_for_classes_that_actually_colocalize() {
        let objects = vec![
            ObjectSpec::new("img1.tif", "ClassA", 1, 10)
                .with_coloc(r#"{"2":["00000000-0000-0000-0000-000000000001"]}"#),
            ObjectSpec::new("img1.tif", "ClassB", 2, 20),
        ];
        let generator = open(&objects);
        let columns = generator.get_available_columns().unwrap();
        let coloc_columns: Vec<&ColumnEntry> = columns
            .iter()
            .filter(|c| matches!(c.key, Column::ColocCount(_)))
            .collect();
        assert_eq!(coloc_columns.len(), 1);
    }

    // -- with_coloc_details fan-out ----------------------------------------

    #[test]
    fn get_object_list_with_coloc_details_resolves_the_partners_own_metric() {
        // Object A colocalizes with class 2, partnered with object B (the
        // second seeded object, id index 1). Both a coloc-class column
        // (`ColocCount(2)`) and a resolvable metric (`AreaSizePx`) must be
        // selected together for `with_coloc_details` fan-out to activate
        // (see `is_resolvable_metric`/`details_active`).
        let objects = vec![
            ObjectSpec::new("img1.tif", "ClassA", 1, 10)
                .with_coloc(r#"{"2":["00000000-0000-0000-0000-000000000001"]}"#),
            ObjectSpec::new("img1.tif", "ClassB", 2, 99),
        ];
        let generator = open(&objects);

        let result = generator
            .get_object_list(&ListFilter {
                plane: plane(),
                images: None,
                object_classes: None,
                columns: vec![
                    Column::ColocCount(ObjectClass::Valid(2)),
                    Column::AreaSizePx,
                ],
                with_coloc_details: true,
                page: no_page(),
            })
            .unwrap();

        assert_eq!(result.column_names.len(), 3);
        assert!(result.column_names[2].contains("coloc"));
        assert_eq!(result.rows.len(), 2, "one fanned-out row per source object");

        // Column order is `Column`'s declared `Ord` (`AreaSizePx` before
        // `ColocCount`), so column 0 is the object's own area, column 1 its
        // coloc count, column 2 the cross-resolved partner metric.
        let int_cell = |cell: &Cell| match &cell.value {
            CellValue::Integer(v) => *v,
            _ => panic!("expected an integer cell"),
        };
        let row_a = &result.rows[0];
        assert_eq!(int_cell(&row_a[0]), 10, "object A's own area");
        assert_eq!(
            int_cell(&row_a[1]),
            1,
            "object A colocalizes with exactly 1 class-2 object"
        );
        assert_eq!(
            int_cell(&row_a[2]),
            99,
            "resolved onto B, the colocalizing partner"
        );

        let row_b = &result.rows[1];
        assert_eq!(int_cell(&row_b[0]), 99, "object B's own area");
        assert_eq!(
            int_cell(&row_b[1]),
            0,
            "object B itself has no colocalizing partners"
        );
        assert!(
            matches!(row_b[2].value, CellValue::String(ref s) if s == "-"),
            "no partner to resolve onto for B"
        );
    }

    // -- plate/well/heatmap grouping correctness & cross-view plausibility --
    //
    // These check that the same underlying data reported through different
    // "views" of the same aggregate (List vs Heatmap; a single well's own
    // query vs the batched every-well query used by exports; a single
    // aggregation vs the multi-aggregation batch) always agree — exactly the
    // kind of drift a hand-rolled SQL string per view/batch variant could
    // silently introduce.

    fn plate_filter(column: Column) -> PlateFilter {
        PlateFilter {
            plane: plane(),
            grouping_regex: String::new(),
            aggregation: Aggregation::Avg,
            object_class: ObjectClass::Unset,
            column,
            color_schema: ColorSchema::default(),
            color_scale: ColorScale::default(),
            matrix_dimension: None,
        }
    }

    fn well_filter(group_name: &str, column: Column) -> WellFilter {
        WellFilter {
            plane: plane(),
            group_name: group_name.to_string(),
            grouping_regex: String::new(),
            aggregation: Aggregation::Avg,
            object_class: ObjectClass::Unset,
            column,
            color_schema: ColorSchema::default(),
            color_scale: ColorScale::default(),
            well_size: None,
            well_order: None,
        }
    }

    fn wells_batch_filter(column: Column) -> WellsBatchFilter {
        WellsBatchFilter {
            plane: plane(),
            grouping_regex: String::new(),
            aggregation: Aggregation::Avg,
            object_class: ObjectClass::Unset,
            column,
            color_schema: ColorSchema::default(),
            color_scale: ColorScale::default(),
            well_size: None,
            well_order: None,
        }
    }

    fn float_cell(cell: &Cell) -> f64 {
        match &cell.value {
            CellValue::Float(v) => *v as f64,
            _ => panic!("expected a float cell"),
        }
    }

    #[test]
    fn plate_list_and_plate_heatmap_report_the_same_values_at_matching_positions() {
        // Well A1 (row 0, col 0): avg(10, 20) = 15. Well B2 (row 1, col 1): 100.
        let generator = open(&[
            ObjectSpec::new("A1_01.tif", "ClassA", 1, 10),
            ObjectSpec::new("A1_02.tif", "ClassA", 1, 20),
            ObjectSpec::new("B2_01.tif", "ClassA", 1, 100),
        ]);
        let filter = plate_filter(Column::AreaSizePx);

        let list = generator.get_group_by_plate(&filter, &View::List).unwrap();
        let heatmap = generator
            .get_group_by_plate(&filter, &View::Heatmap)
            .unwrap();

        let list_value = |well: &str| {
            let idx = list.row_names.iter().position(|n| n == well).unwrap();
            float_cell(&list.rows[idx][1])
        };
        assert_eq!(list_value("A1"), 15.0);
        assert_eq!(list_value("B2"), 100.0);

        // Heatmap row_names are letters ("A", "B", ...), column_names are
        // 1-based numbers ("1", "2", ...) - A1 is heatmap[0][0], B2 is
        // heatmap[1][1].
        assert_eq!(heatmap.row_names[0], "A");
        assert_eq!(heatmap.column_names[0], "1");
        assert_eq!(float_cell(&heatmap.rows[0][0]), list_value("A1"));
        assert_eq!(float_cell(&heatmap.rows[1][1]), list_value("B2"));
        // Every other cell in range has no matching well - must stay empty,
        // not a stray 0.
        assert!(matches!(heatmap.rows[0][1].value, CellValue::Empty));
    }

    #[test]
    fn plate_object_class_filter_restricts_the_grouped_objects() {
        let generator = open(&[
            ObjectSpec::new("A1_01.tif", "ClassA", 1, 10),
            ObjectSpec::new("A1_02.tif", "ClassB", 2, 1000),
        ]);
        let mut filter = plate_filter(Column::AreaSizePx);
        filter.object_class = ObjectClass::Valid(1);
        let list = generator.get_group_by_plate(&filter, &View::List).unwrap();
        assert_eq!(list.row_names, vec!["A1".to_string()]);
        assert_eq!(
            float_cell(&list.rows[0][1]),
            10.0,
            "ClassB's 1000 must be excluded"
        );
    }

    #[test]
    fn well_list_and_well_heatmap_report_the_same_values_at_matching_positions() {
        // Field "01" -> position 0 -> heatmap (row 0, col 0); field "02" ->
        // position 1 -> heatmap (row 0, col 1) (default well_size 4x4, no
        // well_order: idx-1 read row-major).
        let generator = open(&[
            ObjectSpec::new("A1_01.tif", "ClassA", 1, 10),
            ObjectSpec::new("A1_02.tif", "ClassA", 1, 20),
        ]);
        let filter = well_filter("A1", Column::AreaSizePx);

        let list = generator.get_group_by_well(&filter, &View::List).unwrap();
        let heatmap = generator
            .get_group_by_well(&filter, &View::Heatmap)
            .unwrap();

        assert_eq!(list.row_names, vec!["01".to_string(), "02".to_string()]);
        assert_eq!(float_cell(&list.rows[0][1]), 10.0);
        assert_eq!(float_cell(&list.rows[1][1]), 20.0);

        assert_eq!(float_cell(&heatmap.rows[0][0]), 10.0);
        assert_eq!(float_cell(&heatmap.rows[0][1]), 20.0);
        assert!(matches!(heatmap.rows[1][0].value, CellValue::Empty));
    }

    #[test]
    fn wells_for_plate_batched_matches_group_by_well_single_call() {
        let generator = open(&[
            ObjectSpec::new("A1_01.tif", "ClassA", 1, 10),
            ObjectSpec::new("A1_02.tif", "ClassA", 1, 20),
            ObjectSpec::new("B2_01.tif", "ClassA", 1, 100),
        ]);

        let batched = generator
            .get_wells_for_plate(&wells_batch_filter(Column::AreaSizePx), &View::List)
            .unwrap();
        assert_eq!(batched.len(), 2, "one entry per distinct well");

        for well in ["A1", "B2"] {
            let single = generator
                .get_group_by_well(&well_filter(well, Column::AreaSizePx), &View::List)
                .unwrap();
            let batch_result = &batched[well];
            assert_eq!(batch_result.row_names, single.row_names, "well {well}");
            for (b_row, s_row) in batch_result.rows.iter().zip(&single.rows) {
                assert_eq!(float_cell(&b_row[1]), float_cell(&s_row[1]), "well {well}");
            }
        }
    }

    #[test]
    fn group_by_plate_multi_agg_matches_group_by_plate_per_aggregation() {
        let generator = open(&[
            ObjectSpec::new("A1_01.tif", "ClassA", 1, 10),
            ObjectSpec::new("A1_02.tif", "ClassA", 1, 20),
            ObjectSpec::new("A1_03.tif", "ClassA", 1, 30),
        ]);
        let aggregations = vec![Aggregation::Avg, Aggregation::Min, Aggregation::Max];
        let multi = generator
            .get_group_by_plate_multi_agg(
                &plate_filter(Column::AreaSizePx),
                &aggregations,
                &View::List,
            )
            .unwrap();
        assert_eq!(multi.len(), 3);

        for (aggregation, batched_result) in aggregations.iter().zip(&multi) {
            let mut filter = plate_filter(Column::AreaSizePx);
            filter.aggregation = aggregation.clone();
            let single = generator.get_group_by_plate(&filter, &View::List).unwrap();
            assert_eq!(
                float_cell(&batched_result.rows[0][1]),
                float_cell(&single.rows[0][1]),
                "aggregation {aggregation:?} disagrees between multi_agg batch and single call",
            );
        }
        // Sanity on the actual numbers, not just internal agreement.
        assert_eq!(float_cell(&multi[0].rows[0][1]), 20.0); // avg
        assert_eq!(float_cell(&multi[1].rows[0][1]), 10.0); // min
        assert_eq!(float_cell(&multi[2].rows[0][1]), 30.0); // max
    }

    #[test]
    fn wells_for_plate_multi_agg_matches_wells_for_plate_per_aggregation() {
        let generator = open(&[
            ObjectSpec::new("A1_01.tif", "ClassA", 1, 10),
            ObjectSpec::new("A1_02.tif", "ClassA", 1, 20),
        ]);
        let aggregations = vec![Aggregation::Avg, Aggregation::Sum];
        let multi = generator
            .get_wells_for_plate_multi_agg(
                &wells_batch_filter(Column::AreaSizePx),
                &aggregations,
                &View::List,
            )
            .unwrap();
        assert_eq!(multi.len(), 2);

        for (aggregation, batched_by_well) in aggregations.iter().zip(&multi) {
            let mut filter = wells_batch_filter(Column::AreaSizePx);
            filter.aggregation = aggregation.clone();
            let single_by_well = generator.get_wells_for_plate(&filter, &View::List).unwrap();
            for (well_id, single_result) in &single_by_well {
                let batch_result = &batched_by_well[well_id];
                for (b_row, s_row) in batch_result.rows.iter().zip(&single_result.rows) {
                    assert_eq!(
                        float_cell(&b_row[1]),
                        float_cell(&s_row[1]),
                        "well {well_id}, aggregation {aggregation:?}",
                    );
                }
            }
        }
    }

    #[test]
    fn image_heatmap_list_and_heatmap_report_the_same_values_at_matching_positions() {
        // A 100x100 image, 50px tiles -> a 2x2 grid. One object per tile.
        let generator = open(&[
            ObjectSpec::new("img1.tif", "ClassA", 1, 10)
                .at_centroid(10.0, 10.0)
                .with_image_size(100, 100),
            ObjectSpec::new("img1.tif", "ClassA", 1, 20)
                .at_centroid(60.0, 10.0)
                .with_image_size(100, 100),
        ]);
        let filter = ImageHeatmapFilter {
            plane: plane(),
            image_rel_path: "img1.tif".to_string(),
            aggregation: Aggregation::Avg,
            object_class: ObjectClass::Unset,
            column: Column::AreaSizePx,
            color_schema: ColorSchema::default(),
            color_scale: ColorScale::default(),
            square_size: Some(50),
        };

        let list = generator.get_image_heatmap(&filter, &View::List).unwrap();
        let heatmap = generator
            .get_image_heatmap(&filter, &View::Heatmap)
            .unwrap();

        assert_eq!(heatmap.rows.len(), 2, "2x2 grid");
        assert_eq!(heatmap.rows[0].len(), 2);
        assert_eq!(float_cell(&heatmap.rows[0][0]), 10.0, "R0C0");
        assert_eq!(float_cell(&heatmap.rows[0][1]), 20.0, "R0C1");
        assert!(matches!(heatmap.rows[1][0].value, CellValue::Empty));

        assert_eq!(list.row_names, vec!["R0C0".to_string(), "R0C1".to_string()]);
        assert_eq!(
            float_cell(&list.rows[0][1]),
            float_cell(&heatmap.rows[0][0])
        );
        assert_eq!(
            float_cell(&list.rows[1][1]),
            float_cell(&heatmap.rows[0][1])
        );
    }

    #[test]
    fn image_heatmap_clamps_a_centroid_on_the_far_edge_into_the_last_tile() {
        // A 100x100 image, 50px tiles -> valid tile indices 0..=1. A centroid
        // sitting exactly on the image's far edge (100.0) floor-divides to
        // tile index 2, which the 2x2 grid has no slot for - must clamp into
        // the last tile (index 1) rather than being dropped or panicking.
        let generator = open(&[ObjectSpec::new("img1.tif", "ClassA", 1, 10)
            .at_centroid(100.0, 100.0)
            .with_image_size(100, 100)]);
        let filter = ImageHeatmapFilter {
            plane: plane(),
            image_rel_path: "img1.tif".to_string(),
            aggregation: Aggregation::Avg,
            object_class: ObjectClass::Unset,
            column: Column::AreaSizePx,
            color_schema: ColorSchema::default(),
            color_scale: ColorScale::default(),
            square_size: Some(50),
        };
        let heatmap = generator
            .get_image_heatmap(&filter, &View::Heatmap)
            .unwrap();
        assert_eq!(float_cell(&heatmap.rows[1][1]), 10.0);
    }

    // -- small pure helpers ---------------------------------------------------

    #[test]
    fn row_letter_index_round_trips_including_double_letters() {
        for (letters, index) in [("A", 0), ("B", 1), ("Z", 25), ("AA", 26), ("AB", 27)] {
            assert_eq!(row_letter_to_index(letters), Some(index));
            assert_eq!(row_index_to_letter(index), letters);
        }
    }

    #[test]
    fn row_letter_to_index_rejects_non_alphabetic_input() {
        assert_eq!(row_letter_to_index(""), None);
        assert_eq!(row_letter_to_index("A1"), None);
    }

    #[test]
    fn col_number_to_index_is_one_based() {
        assert_eq!(col_number_to_index("1"), Some(0));
        assert_eq!(col_number_to_index("12"), Some(11));
        assert_eq!(
            col_number_to_index("0"),
            None,
            "0 has no 0-based predecessor"
        );
        assert_eq!(col_number_to_index("abc"), None);
    }

    #[test]
    fn best_matching_dimensions_picks_the_smallest_plate_that_fits() {
        // Needs 8 rows (max_row index 7) x 10 cols (max_col index 9) - the
        // smallest standard size with at least 8 rows and 10 cols is 8x12
        // (6x8 falls short on rows: 6 < 8).
        assert_eq!(
            best_matching_dimensions(Some(7), Some(9)),
            PlateDimensions::Plate8x12
        );
        assert_eq!(
            best_matching_dimensions(None, None),
            PlateDimensions::PLate2x3
        );
    }

    #[test]
    fn plate_dimensions_reports_correct_row_col_counts_for_every_size() {
        assert_eq!(PlateDimensions::PLate2x3.dimensions(), (2, 3));
        assert_eq!(PlateDimensions::Plate3x4.dimensions(), (3, 4));
        assert_eq!(PlateDimensions::Plate4x6.dimensions(), (4, 6));
        assert_eq!(PlateDimensions::Plate6x8.dimensions(), (6, 8));
        assert_eq!(PlateDimensions::Plate8x12.dimensions(), (8, 12));
        assert_eq!(PlateDimensions::Plate16x24.dimensions(), (16, 24));
        assert_eq!(PlateDimensions::Plate32x48.dimensions(), (32, 48));
    }

    #[test]
    fn color_scale_gradient_and_value_to_color_cover_every_schema() {
        for schema in [
            ColorSchema::Excel,
            ColorSchema::Viridis,
            ColorSchema::Plasma,
            ColorSchema::Inferno,
            ColorSchema::Cividis,
            ColorSchema::Coolwarm,
            ColorSchema::RedBlue,
            ColorSchema::YlGnBu,
            ColorSchema::Haline,
            ColorSchema::Algae,
            ColorSchema::Thermal,
        ] {
            let stops = color_scale_gradient(&schema);
            assert_eq!(stops.len(), COLOR_SCALE_GRADIENT_STOPS);
            assert_ne!(
                stops[0],
                stops[COLOR_SCALE_GRADIENT_STOPS - 1],
                "gradient should not be flat"
            );
        }
    }

    #[test]
    fn column_display_label_coloc_count_falls_back_to_a_numeric_class_label() {
        let classes: Vec<Class> = vec![];
        assert_eq!(
            Column::ColocCount(ObjectClass::Valid(7)).display_label(&classes),
            "Coloc with class 7"
        );
        assert_eq!(
            Column::ColocCount(ObjectClass::Unset).display_label(&classes),
            "Coloc with unset"
        );
    }

    #[test]
    fn get_object_list_with_an_explicitly_empty_class_selection_is_empty() {
        let generator = open(&[ObjectSpec::new("img1.tif", "ClassA", 1, 10)]);
        let result = generator
            .get_object_list(&ListFilter {
                plane: plane(),
                images: None,
                object_classes: Some(vec![]),
                columns: vec![Column::ObjectClass],
                with_coloc_details: false,
                page: no_page(),
            })
            .unwrap();
        assert_eq!(result.source_object_count, 0);
        assert!(result.rows.is_empty());
    }

    #[test]
    fn get_object_list_on_a_plane_with_no_objects_at_all_is_empty() {
        let generator = open(&[ObjectSpec::new("img1.tif", "ClassA", 1, 10)]);
        let result = generator
            .get_object_list(&ListFilter {
                plane: PlaneFilter {
                    z_stack: 99,
                    t_stack: 99,
                },
                images: None,
                object_classes: None,
                columns: vec![Column::AreaSizePx],
                with_coloc_details: false,
                page: no_page(),
            })
            .unwrap();
        assert_eq!(result.source_object_count, 0);
        assert!(result.rows.is_empty());
    }

    #[test]
    fn get_grouped_by_image_object_classes_filter_restricts_to_the_selected_class() {
        let generator = open(&[
            ObjectSpec::new("img1.tif", "ClassA", 1, 10),
            ObjectSpec::new("img1.tif", "ClassB", 2, 999),
        ]);
        let result = generator
            .get_grouped_by_image(&GroupedByImageFilter {
                plane: plane(),
                images: None,
                object_classes: Some(vec![ObjectClass::Valid(1)]),
                columns: vec![Column::AreaSizePx],
                aggregation: vec![Aggregation::Avg],
                page: no_page(),
            })
            .unwrap();
        assert_eq!(
            result.rows.len(),
            1,
            "only ClassA's group should be reported"
        );
    }

    #[test]
    fn get_grouped_by_image_with_an_explicitly_empty_class_selection_is_empty() {
        let generator = open(&[ObjectSpec::new("img1.tif", "ClassA", 1, 10)]);
        let result = generator
            .get_grouped_by_image(&GroupedByImageFilter {
                plane: plane(),
                images: None,
                object_classes: Some(vec![]),
                columns: vec![Column::AreaSizePx],
                aggregation: vec![Aggregation::Avg],
                page: no_page(),
            })
            .unwrap();
        assert!(result.rows.is_empty());
    }

    #[test]
    fn get_grouped_by_image_on_a_plane_with_no_objects_reports_a_zeroed_range() {
        let generator = open(&[ObjectSpec::new("img1.tif", "ClassA", 1, 10)]);
        let result = generator
            .get_grouped_by_image(&GroupedByImageFilter {
                plane: PlaneFilter {
                    z_stack: 99,
                    t_stack: 99,
                },
                images: None,
                object_classes: None,
                columns: vec![Column::AreaSizePx],
                aggregation: vec![Aggregation::Avg],
                page: no_page(),
            })
            .unwrap();
        assert!(result.rows.is_empty());
        assert_eq!(result.min, 0.0);
        assert_eq!(result.max, 0.0);
    }

    #[test]
    fn get_grouped_by_image_pagination_cursor_returns_the_next_page() {
        let generator = open(&[
            ObjectSpec::new("img1.tif", "ClassA", 1, 10),
            ObjectSpec::new("img2.tif", "ClassA", 1, 20),
            ObjectSpec::new("img3.tif", "ClassA", 1, 30),
        ]);
        let base = GroupedByImageFilter {
            plane: plane(),
            images: None,
            object_classes: None,
            columns: vec![Column::AreaSizePx],
            aggregation: vec![Aggregation::Avg],
            page: Pagination {
                limit: 1,
                after: None,
            },
        };
        let first = generator.get_grouped_by_image(&base).unwrap();
        assert_eq!(first.rows.len(), 1);
        let cursor = first.row_names.last().cloned();

        let second = generator
            .get_grouped_by_image(&GroupedByImageFilter {
                page: Pagination {
                    limit: 1,
                    after: cursor,
                },
                ..base
            })
            .unwrap();
        assert_eq!(second.rows.len(), 1);
        assert_ne!(first.row_names, second.row_names);
    }

    #[test]
    fn get_object_list_with_coloc_details_and_no_colocalizing_partners_still_fills_every_row() {
        // Both a coloc-class column and a resolvable metric are selected
        // (activating `with_coloc_details`'s fan-out), but neither object
        // colocalizes with anything at all - `partners_by_object`/
        // `partner_rows` must handle "not a single partner anywhere"
        // gracefully rather than only "some partners, some not".
        let generator = open(&[
            ObjectSpec::new("img1.tif", "ClassA", 1, 10),
            ObjectSpec::new("img1.tif", "ClassB", 2, 20),
        ]);
        let result = generator
            .get_object_list(&ListFilter {
                plane: plane(),
                images: None,
                object_classes: None,
                columns: vec![
                    Column::ColocCount(ObjectClass::Valid(2)),
                    Column::AreaSizePx,
                ],
                with_coloc_details: true,
                page: no_page(),
            })
            .unwrap();
        assert_eq!(
            result.rows.len(),
            2,
            "one row per source object, no fan-out"
        );
        for row in &result.rows {
            assert!(matches!(row.last().unwrap().value, CellValue::String(ref s) if s == "-"));
        }
    }

    #[test]
    fn get_object_list_with_coloc_details_dashes_the_inactive_coloc_class_column() {
        // Two coloc-class columns selected at once: object A colocalizes
        // with class 2 only, so its fanned-out row(s) must resolve the
        // class-2 cross-cell but dash the class-3 one, never the reverse.
        let objects = vec![
            ObjectSpec::new("img1.tif", "ClassA", 1, 10)
                .with_coloc(r#"{"2":["00000000-0000-0000-0000-000000000001"]}"#),
            ObjectSpec::new("img1.tif", "ClassB", 2, 99),
        ];
        let generator = open(&objects);
        let result = generator
            .get_object_list(&ListFilter {
                plane: plane(),
                images: None,
                object_classes: None,
                columns: vec![
                    Column::AreaSizePx,
                    Column::ColocCount(ObjectClass::Valid(2)),
                    Column::ColocCount(ObjectClass::Valid(3)),
                ],
                with_coloc_details: true,
                page: no_page(),
            })
            .unwrap();

        // Column order (sorted): AreaSizePx, ColocCount(2), ColocCount(3),
        // then the coloc-detail cross-columns in the same
        // coloc-class-then-metric order: (2, AreaSizePx), (3, AreaSizePx).
        let object_a_row = &result.rows[0];
        let dash = |cell: &Cell| matches!(cell.value, CellValue::String(ref s) if s == "-");
        assert!(
            !dash(&object_a_row[3]),
            "class 2's cross-cell should resolve, not dash"
        );
        assert!(
            dash(&object_a_row[4]),
            "class 3's cross-cell must dash - A doesn't colocalize with it"
        );
    }

    #[test]
    fn get_group_by_plate_honors_a_custom_grouping_regex() {
        // A single well under this custom `(plate)-(well)(idx)` scheme -
        // proves the non-default `grouping_regex` branch is actually used
        // (rather than falling back to `DEFAULT_GROUPING_REGEX`, which
        // wouldn't match this image name at all and would report nothing).
        let generator = open(&[ObjectSpec::new("plateX-well1.tif", "ClassA", 1, 42)]);
        let result = generator
            .get_group_by_plate(
                &PlateFilter {
                    plane: plane(),
                    grouping_regex: r"^(plateX)-(well)(\d+)\.".to_string(),
                    aggregation: Aggregation::Avg,
                    object_class: ObjectClass::Unset,
                    column: Column::AreaSizePx,
                    color_schema: ColorSchema::default(),
                    color_scale: ColorScale::default(),
                    matrix_dimension: None,
                },
                &View::List,
            )
            .unwrap();
        assert_eq!(result.row_names, vec!["plateX".to_string()]);
        let cell_f64 = |cell: &Cell| match &cell.value {
            CellValue::Float(v) => *v as f64,
            _ => panic!("expected a float cell"),
        };
        assert_eq!(cell_f64(&result.rows[0][1]), 42.0);
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
    // Always fetched (unlike every field above, gated by `ObjectColumnNeeds`
    // on whether its `Column` is actually selected/displayed) - needed by
    // the GUI to navigate to and highlight this object in its source image
    // (see `DatabaseResult::row_locations`) regardless of which columns the
    // user chose to show.
    image_rel_path: String,
    bbox_xmin_px: u32,
    bbox_ymin_px: u32,
    bbox_xmax_px: u32,
    bbox_ymax_px: u32,
}

/// Which of `ObjectRow`'s source columns a given column selection actually
/// needs — shared by `get_list`'s main fetch (needs from `ordered_columns`)
/// and its coloc-detail partner fetch (needs from just `metric_columns`,
/// see `Column::with_coloc_details` on `ListFilter`), so both build their
/// `SELECT` list and parse rows the exact same (bug-for-bug consistent) way.
#[derive(Default, Clone, Copy)]
struct ObjectColumnNeeds {
    image_name: bool,
    class: bool,
    area_px: bool,
    area_nm2: bool,
    perimeter_px: bool,
    perimeter_nm: bool,
    circularity: bool,
    solidity: bool,
    eccentricity: bool,
    coloc: bool,
    intensities: bool,
}

impl ObjectColumnNeeds {
    fn for_columns(columns: &[Column]) -> Self {
        Self {
            image_name: columns.contains(&Column::ImageName),
            class: columns.contains(&Column::ObjectClass),
            area_px: columns.contains(&Column::AreaSizePx),
            area_nm2: columns.contains(&Column::AreaSizeNm),
            perimeter_px: columns.contains(&Column::PerimeterPx),
            perimeter_nm: columns.contains(&Column::PerimeterNm),
            circularity: columns.contains(&Column::Circularity),
            solidity: columns.contains(&Column::Solidity),
            eccentricity: columns.contains(&Column::Eccentricity),
            coloc: columns.iter().any(|c| matches!(c, Column::ColocCount(_))),
            intensities: columns.iter().any(|c| {
                matches!(
                    c,
                    Column::IntensityAvg(_)
                        | Column::IntensitySum(_)
                        | Column::IntensityMin(_)
                        | Column::IntensityMax(_)
                )
            }),
        }
    }
}

/// The comma-joined `SELECT` column list `get_list` queries `objects`
/// with — a column not in `need` becomes a cheap constant instead of a real
/// column reference (see the column-pruning comment on `get_list`), so
/// `map_object_row` below can always read the same fixed positions
/// regardless of which are real. `image_rel_path`/the four `bbox_*_px`
/// columns are the exception: small fixed-width columns, always selected
/// for real regardless of `need`, since the GUI needs an object's location
/// to navigate to and highlight it (see `DatabaseResult::row_locations`)
/// independent of which columns are actually displayed.
fn object_select_clause(need: ObjectColumnNeeds) -> String {
    let select_image_name = if need.image_name { "image_name" } else { "''" };
    let select_object_class_name = if need.class {
        "CAST(object_class_name AS VARCHAR[])"
    } else {
        "CAST(NULL AS VARCHAR[])"
    };
    let select_seg_class_name = if need.class {
        "seg_class_name"
    } else {
        "NULL::VARCHAR"
    };
    let select_area_px = if need.area_px {
        "area_px"
    } else {
        "0::UBIGINT"
    };
    let select_area_nm2 = if need.area_nm2 {
        "area_nm2"
    } else {
        "0.0::DOUBLE"
    };
    let select_perimeter_px = if need.perimeter_px {
        "perimeter_px"
    } else {
        "0.0::DOUBLE"
    };
    let select_perimeter_nm = if need.perimeter_nm {
        "perimeter_nm"
    } else {
        "0.0::DOUBLE"
    };
    let select_circularity = if need.circularity {
        "circularity"
    } else {
        "0.0::DOUBLE"
    };
    let select_solidity = if need.solidity {
        "solidity"
    } else {
        "0.0::DOUBLE"
    };
    let select_eccentricity = if need.eccentricity {
        "eccentricity"
    } else {
        "0.0::DOUBLE"
    };
    let select_coloc_json = if need.coloc {
        "coloc_json"
    } else {
        "NULL::VARCHAR"
    };
    let select_intensities_json = if need.intensities {
        "intensities_json"
    } else {
        "NULL::VARCHAR"
    };
    format!(
        "object_id, {select_image_name}, {select_object_class_name}, {select_seg_class_name},\n\
                {select_area_px}, {select_area_nm2}, {select_perimeter_px}, {select_perimeter_nm},\n\
                {select_circularity}, {select_solidity}, {select_eccentricity},\n\
                {select_coloc_json}, {select_intensities_json},\n\
                image_rel_path, bbox_xmin_px, bbox_ymin_px, bbox_xmax_px, bbox_ymax_px"
    )
}

/// Inverse of `object_select_clause`'s fixed column position order —
/// shared so the main and partner fetches in `get_list` can never drift.
fn map_object_row(row: &duckdb::Row<'_>) -> duckdb::Result<ObjectRow> {
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
        image_rel_path: row.get(13)?,
        bbox_xmin_px: row.get(14)?,
        bbox_ymin_px: row.get(15)?,
        bbox_xmax_px: row.get(16)?,
        bbox_ymax_px: row.get(17)?,
    })
}

/// `coloc_json`'s key for `class` (see `coloc_to_json` in evanalyzer_core's
/// duckdb.rs) — shared by `coloc_count_for_class` and `get_list`'s
/// coloc-detail partner resolution so both agree on the same lookup.
fn coloc_class_key(class: ObjectClass) -> String {
    match class {
        ObjectClass::Valid(n) => n.to_string(),
        ObjectClass::Unset => "unset".to_string(),
    }
}

/// `DatabaseResult::row_locations`' entry for `object` — its source image
/// (by rel path) and pixel bounding box, for the GUI to navigate to and
/// highlight it.
fn object_location(object: &ObjectRow) -> (String, [u32; 4]) {
    (
        object.image_rel_path.clone(),
        [
            object.bbox_xmin_px,
            object.bbox_ymin_px,
            object.bbox_xmax_px,
            object.bbox_ymax_px,
        ],
    )
}

/// Whether `column` names a per-object value that can be meaningfully
/// resolved on a *different* object — i.e. a coloc partner's own value for
/// that same column, per `ListFilter::with_coloc_details`. Includes
/// `ObjectId` deliberately (even though it's identity, not a measurement):
/// without it there'd be no way to tell *which* partner object a fanned-out
/// coloc-detail row is actually about, only which class it belongs to.
/// `ImageName`/`ObjectClass`/`ColocCount` stay excluded — a coloc partner is
/// always in the same image as its source object (so `ImageName` would
/// just repeat the source row's own value), the partner's class is already
/// implied by which `coloc_class_columns` combination produced the row, and
/// resolving `ColocCount` on the partner would mean its *own* colocalization
/// counts, not this relationship.
fn is_resolvable_metric(column: &Column) -> bool {
    matches!(
        column,
        Column::ObjectId
            | Column::AreaSizePx
            | Column::AreaSizeNm
            | Column::PerimeterPx
            | Column::PerimeterNm
            | Column::Circularity
            | Column::Solidity
            | Column::Eccentricity
            | Column::IntensityAvg(_)
            | Column::IntensitySum(_)
            | Column::IntensityMin(_)
            | Column::IntensityMax(_)
    )
}

/// Display label for a `coloc_json`/`object_class_name`-adjacent class,
/// e.g. for a coloc-detail column header — the class's registered name, or
/// `"class {n}"` if `n` isn't (or no longer is) a recognized id.
pub(crate) fn class_display_label(class: ObjectClass, classes: &[Class]) -> String {
    match class {
        ObjectClass::Valid(n) => classes
            .iter()
            .find(|c| c.id == ObjectClass::Valid(n))
            .map(|c| c.name.clone())
            .unwrap_or_else(|| format!("class {n}")),
        ObjectClass::Unset => "unset".to_string(),
    }
}

/// Escapes and comma-joins string literals for a SQL `IN (...)` list.
///
/// `pub(super)`: also used by `results_charts.rs`'s boxplot query, which
/// needs raw SQL access this crate keeps otherwise private to this module.
pub(super) fn sql_string_in_list(values: &[String]) -> String {
    values
        .iter()
        .map(|v| format!("'{}'", v.replace('\'', "''")))
        .collect::<Vec<_>>()
        .join(", ")
}

/// Comma-joins integers into a DuckDB list literal, e.g. `[1, 2]`, for
/// `list_has_any(...)`. `pub(super)`: see `sql_string_in_list`.
pub(super) fn sql_int_array_literal(values: &[u32]) -> String {
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
        alternating_color: false,
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
                alternating_color: false,
                search_key: None,
            }
        }
        Column::Count => Cell {
            value: CellValue::Integer(1),
            ..no_bg
        },
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
                "mean_scaled",
            )),
            ..no_bg
        },
        Column::IntensitySum(channel) => Cell {
            value: CellValue::Float(intensity_stat(
                &object.intensities_json,
                *channel,
                "sum_scaled",
            )),
            ..no_bg
        },
        Column::IntensityMin(channel) => Cell {
            value: CellValue::Float(intensity_stat(
                &object.intensities_json,
                *channel,
                "min_scaled",
            )),
            ..no_bg
        },
        Column::IntensityMax(channel) => Cell {
            value: CellValue::Float(intensity_stat(
                &object.intensities_json,
                *channel,
                "max_scaled",
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
/// function by `get_group_by_plate`/`get_group_by_well`/`get_image_heatmap`
/// — also `results_charts.rs`'s histogram/scatter/boxplot queries (hence
/// `pub(super)`), which aggregate nothing themselves but still need a bare
/// per-object value expression to bucket/plot/box.
/// Plain numeric `objects` columns are a direct column reference;
/// `ColocCount` is a `json_array_length` extraction (see
/// `coloc_count_for_class`, which does the same lookup per-row in Rust for
/// `get_list`). The per-channel intensity columns still need their own
/// JSON-extraction SQL (see `intensity_stat`, which — like `coloc_count_for_class`
/// before this — only handles this per-row in Rust today, not as a groupable
/// SQL expression), left for a follow-up.
pub(super) fn column_aggregate_expr(column: &Column) -> Result<String, InternalErrors> {
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
        // Handled by `aggregate_sql` before this function is ever called
        // with `Column::Count` — `COUNT(*)` doesn't fit the "aggregate
        // function wraps a per-row scalar expression" shape every other
        // arm here does, since it counts rows rather than reading a column
        // off them. Kept here (rather than left unreachable) only so this
        // match stays exhaustive.
        Column::Count
        | Column::ObjectId
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
        Aggregation::Median => "MEDIAN",
        Aggregation::Skewness => "SKEWNESS",
    }
}

/// The `{agg_fn}({value_expr})` pair `get_group_by_plate`/`get_group_by_well`/
/// `get_image_heatmap` plug into their `SELECT`. `Column::Count` ("number of
/// objects", not a per-object measurement) is special-cased to a flat
/// `COUNT(*)`, ignoring `aggregation` entirely — averaging or summing a count
/// across an already-single-valued group wouldn't mean anything the count
/// itself doesn't already say more plainly. Every other column defers to the
/// existing `aggregation_sql_fn`/`column_aggregate_expr`.
fn aggregate_sql(
    column: &Column,
    aggregation: &Aggregation,
) -> Result<(&'static str, String), InternalErrors> {
    if matches!(column, Column::Count) {
        return Ok(("COUNT", "*".to_string()));
    }
    Ok((
        aggregation_sql_fn(aggregation),
        column_aggregate_expr(column)?,
    ))
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

// Excel's built-in "Red - Yellow - Green" 3-Color Scale conditional format —
// red at the high end, green at the low end (`t=0` is `min`, `t=1` is `max`,
// see `value_to_color`), matching how Excel's own scale reads by default.
const EXCEL_STOPS: [(f32, (u8, u8, u8)); 3] = [
    (0.0, (0x63, 0xbe, 0x7b)),
    (0.5, (0xff, 0xeb, 0x84)),
    (1.0, (0xf8, 0x69, 0x6b)),
];

// Approximate 5-stop reproductions of well-known scientific colormaps —
// same reasoning/precision level as `VIRIDIS_STOPS` above: recognizable as
// the named colormap, not a pixel-exact reproduction of it.

// matplotlib "plasma" (dark blue-purple -> magenta -> orange -> yellow).
const PLASMA_STOPS: [(f32, (u8, u8, u8)); 5] = [
    (0.0, (0x0d, 0x08, 0x87)),
    (0.25, (0x7e, 0x03, 0xa8)),
    (0.5, (0xcc, 0x47, 0x78)),
    (0.75, (0xf8, 0x94, 0x41)),
    (1.0, (0xf0, 0xf9, 0x21)),
];

// matplotlib "inferno" (black -> purple -> red -> orange -> pale yellow).
const INFERNO_STOPS: [(f32, (u8, u8, u8)); 5] = [
    (0.0, (0x00, 0x00, 0x04)),
    (0.25, (0x57, 0x10, 0x6e)),
    (0.5, (0xbc, 0x37, 0x54)),
    (0.75, (0xf9, 0x8c, 0x0a)),
    (1.0, (0xfc, 0xff, 0xa4)),
];

// matplotlib "cividis" (colorblind-friendly dark blue -> gray -> yellow).
const CIVIDIS_STOPS: [(f32, (u8, u8, u8)); 5] = [
    (0.0, (0x00, 0x20, 0x4d)),
    (0.25, (0x41, 0x4d, 0x6b)),
    (0.5, (0x7c, 0x7b, 0x78)),
    (0.75, (0xbc, 0xaf, 0x6f)),
    (1.0, (0xff, 0xea, 0x46)),
];

// matplotlib "coolwarm" (diverging blue -> near-white -> red).
const COOLWARM_STOPS: [(f32, (u8, u8, u8)); 5] = [
    (0.0, (0x3b, 0x4c, 0xc0)),
    (0.25, (0x88, 0xab, 0xfd)),
    (0.5, (0xdd, 0xdd, 0xdd)),
    (0.75, (0xf7, 0xa8, 0x89)),
    (1.0, (0xb4, 0x04, 0x26)),
];

// ColorBrewer "RdBu" diverging (dark red -> near-white -> dark blue).
const RED_BLUE_STOPS: [(f32, (u8, u8, u8)); 5] = [
    (0.0, (0x67, 0x00, 0x1f)),
    (0.25, (0xd6, 0x60, 0x4d)),
    (0.5, (0xf7, 0xf7, 0xf7)),
    (0.75, (0x43, 0x93, 0xc3)),
    (1.0, (0x05, 0x30, 0x61)),
];

// ColorBrewer "YlGnBu" sequential (pale yellow -> green -> blue -> dark navy).
const YLGNBU_STOPS: [(f32, (u8, u8, u8)); 5] = [
    (0.0, (0xff, 0xff, 0xd9)),
    (0.25, (0x7f, 0xcd, 0xbb)),
    (0.5, (0x41, 0xb6, 0xc4)),
    (0.75, (0x22, 0x5e, 0xa8)),
    (1.0, (0x08, 0x1d, 0x58)),
];

// cmocean "haline" (dark indigo -> teal -> green -> pale yellow-green),
// used for ocean salinity.
const HALINE_STOPS: [(f32, (u8, u8, u8)); 5] = [
    (0.0, (0x29, 0x18, 0x6b)),
    (0.25, (0x21, 0x6b, 0x7a)),
    (0.5, (0x2e, 0x9c, 0x82)),
    (0.75, (0x8f, 0xcb, 0x6c)),
    (1.0, (0xf6, 0xed, 0x4c)),
];

// cmocean "algae" (pale yellow-green -> mid green -> near-black dark green),
// used for algae/chlorophyll concentration.
const ALGAE_STOPS: [(f32, (u8, u8, u8)); 5] = [
    (0.0, (0xd9, 0xf0, 0xa3)),
    (0.25, (0x78, 0xc6, 0x79)),
    (0.5, (0x31, 0xa3, 0x54)),
    (0.75, (0x00, 0x68, 0x37)),
    (1.0, (0x00, 0x44, 0x1b)),
];

// cmocean "thermal" (dark navy-black -> purple -> red -> orange -> pale
// yellow), used for ocean temperature.
const THERMAL_STOPS: [(f32, (u8, u8, u8)); 5] = [
    (0.0, (0x04, 0x23, 0x33)),
    (0.25, (0x52, 0x27, 0x6b)),
    (0.5, (0xa8, 0x32, 0x7d)),
    (0.75, (0xe2, 0x72, 0x4f)),
    (1.0, (0xf2, 0xf1, 0x8d)),
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

/// Turns raw `(group_prefix, row, col, value)` plate-group rows into a
/// `DatabaseResult` — shared by `get_group_by_plate` (one aggregation per
/// call) and `get_group_by_plate_multi_agg` (every requested aggregation in
/// one batched query, calling this once per aggregation over its own slice
/// of that batch) so the two agree on exactly the same List/Heatmap shape.
fn plate_groups_to_result(
    groups: Vec<(String, String, String, Option<f64>)>,
    column: &Column,
    classes: &[Class],
    matrix_dimension: Option<PlateDimensions>,
    color_schema: &ColorSchema,
    color_scale: &ColorScale,
    view: &View,
) -> DatabaseResult {
    match view {
        View::List => {
            let mut min = f64::INFINITY;
            let mut max = f64::NEG_INFINITY;
            for (_, _, _, value) in &groups {
                if let Some(value) = value {
                    min = min.min(*value);
                    max = max.max(*value);
                }
            }
            if !min.is_finite() || !max.is_finite() {
                min = 0.0;
                max = 0.0;
            }

            let column_names = vec!["group".to_string(), column.display_label(classes)];
            let row_names = groups.iter().map(|(key, ..)| key.clone()).collect();
            let rows: Vec<Vec<Cell>> = groups
                .into_iter()
                .map(|(key, _row, _col, value)| {
                    // `key` is the group/well id (e.g. "A1") itself, so
                    // it's its own search key — used by the GUI to
                    // navigate into that group/well.
                    let search_key = Some((key.clone(), key.clone()));
                    vec![
                        Cell {
                            value: CellValue::String(key),
                            bg_color: 0,
                            alternating_color: false,
                            search_key: search_key.clone(),
                        },
                        Cell {
                            value: CellValue::Float(value.unwrap_or(0.0) as f32),
                            bg_color: 0,
                            alternating_color: false,
                            search_key,
                        },
                    ]
                })
                .collect();
            let source_object_count = rows.len();
            DatabaseResult {
                column_names,
                row_names,
                rows,
                min: min as f32,
                max: max as f32,
                source_object_count,
                row_locations: Vec::new(),
            }
        }
        View::Heatmap => {
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
            for (group_prefix, row, col, value) in &groups {
                let (Some(row), Some(col)) = (row_letter_to_index(row), col_number_to_index(col))
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
            let dimensions =
                matrix_dimension.unwrap_or_else(|| best_matching_dimensions(max_row, max_col));
            let (rows, cols) = dimensions.dimensions();

            let (range_min, range_max) = match color_scale {
                ColorScale::Manual(min, max) => (*min as f64, *max as f64),
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

            let grid_rows: Vec<Vec<Cell>> = (0..rows)
                .map(|row| {
                    (0..cols)
                        .map(|col| match values.get(&(row, col)) {
                            Some((value, group_prefix)) => Cell {
                                value: CellValue::Float(*value as f32),
                                bg_color: value_to_color(
                                    *value,
                                    range_min,
                                    range_max,
                                    color_schema,
                                ),
                                alternating_color: false,
                                search_key: Some((group_prefix.clone(), group_prefix.clone())),
                            },
                            // No object matched this well at all — leave
                            // it empty rather than showing a misleading 0
                            // or a value from some other well.
                            None => Cell {
                                value: CellValue::Empty,
                                bg_color: 0,
                                alternating_color: false,
                                search_key: None,
                            },
                        })
                        .collect()
                })
                .collect();

            let source_object_count = grid_rows.len();
            DatabaseResult {
                column_names: (1..=cols).map(|col| col.to_string()).collect(),
                row_names: (0..rows).map(row_index_to_letter).collect(),
                rows: grid_rows,
                // Same range the cells were colored against above, so the
                // GUI's color bar always matches what's actually painted
                // rather than recomputing (and potentially disagreeing
                // with) it from the returned cells.
                min: range_min as f32,
                max: range_max as f32,
                source_object_count,
                row_locations: Vec::new(),
            }
        }
    }
}

/// Turns one well's raw `(idx, image_rel_path, image_name, value)` field
/// rows into a `DatabaseResult` — shared by `get_group_by_well` (one well
/// per call) and `get_wells_for_plate` (every well in one batched query,
/// calling this once per well over its slice of that batch) so the two
/// agree on exactly the same List/Heatmap shape.
fn well_fields_to_result(
    fields: Vec<(String, String, String, Option<f64>)>,
    column: &Column,
    classes: &[Class],
    well_size: Option<WellSize>,
    well_order: &Option<Vec<u32>>,
    color_schema: &ColorSchema,
    color_scale: &ColorScale,
    view: &View,
) -> DatabaseResult {
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

            let column_names = vec!["field".to_string(), column.display_label(classes)];
            let row_names = fields.iter().map(|(idx, ..)| idx.clone()).collect();
            let rows: Vec<Vec<Cell>> = fields
                .into_iter()
                .map(|(idx, image_rel_path, image_name, value)| {
                    let search_key = Some((image_name, image_rel_path));
                    vec![
                        Cell {
                            value: CellValue::String(idx),
                            bg_color: 0,
                            alternating_color: false,
                            search_key: search_key.clone(),
                        },
                        Cell {
                            value: CellValue::Float(value.unwrap_or(0.0) as f32),
                            bg_color: 0,
                            alternating_color: false,
                            search_key,
                        },
                    ]
                })
                .collect();
            let source_object_count = rows.len();
            DatabaseResult {
                column_names,
                row_names,
                rows,
                min: min as f32,
                max: max as f32,
                source_object_count,
                row_locations: Vec::new(),
            }
        }
        View::Heatmap => {
            // No `well_order` (see the doc comment on
            // `WellFilter::well_order`): a field's `idx` (1-based) is its
            // position directly, in row-major reading order — idx 1 -> (0,
            // 0), idx 2 -> (0, 1), .... Given a `well_order`, it's a lookup
            // table instead: the value at `well_order[position]` names
            // which field idx sits at that (row-major) grid position,
            // letting a well be laid out in a non-trivial (e.g. snake)
            // acquisition pattern.
            let well_size = well_size.unwrap_or(WellSize { rows: 4, cols: 4 });
            let (rows, cols) = (well_size.rows, well_size.cols);

            let mut values: HashMap<usize, (f64, String, String)> = HashMap::new();
            for (idx_str, image_rel_path, image_name, value) in &fields {
                let Ok(idx) = idx_str.parse::<u32>() else {
                    continue;
                };
                let position = match well_order {
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

            let (range_min, range_max) = match color_scale {
                ColorScale::Manual(min, max) => (*min as f64, *max as f64),
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

            let grid_rows: Vec<Vec<Cell>> = (0..rows)
                .map(|row| {
                    (0..cols)
                        .map(|col| match values.get(&(row * cols + col)) {
                            Some((value, image_name, image_rel_path)) => Cell {
                                value: CellValue::Float(*value as f32),
                                bg_color: value_to_color(
                                    *value,
                                    range_min,
                                    range_max,
                                    color_schema,
                                ),
                                alternating_color: false,
                                search_key: Some((image_name.clone(), image_rel_path.clone())),
                            },
                            // No field occupies this grid position — leave
                            // it empty rather than showing a misleading 0
                            // or another field's value.
                            None => Cell {
                                value: CellValue::Empty,
                                bg_color: 0,
                                alternating_color: false,
                                search_key: None,
                            },
                        })
                        .collect()
                })
                .collect();

            let source_object_count = grid_rows.len();
            DatabaseResult {
                column_names: (1..=cols).map(|col| col.to_string()).collect(),
                row_names: (1..=rows).map(|row| row.to_string()).collect(),
                rows: grid_rows,
                min: range_min as f32,
                max: range_max as f32,
                source_object_count,
                row_locations: Vec::new(),
            }
        }
    }
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
        ColorSchema::Plasma => lerp_palette(&PLASMA_STOPS, t),
        ColorSchema::Inferno => lerp_palette(&INFERNO_STOPS, t),
        ColorSchema::Cividis => lerp_palette(&CIVIDIS_STOPS, t),
        ColorSchema::Coolwarm => lerp_palette(&COOLWARM_STOPS, t),
        ColorSchema::RedBlue => lerp_palette(&RED_BLUE_STOPS, t),
        ColorSchema::YlGnBu => lerp_palette(&YLGNBU_STOPS, t),
        ColorSchema::Haline => lerp_palette(&HALINE_STOPS, t),
        ColorSchema::Algae => lerp_palette(&ALGAE_STOPS, t),
        ColorSchema::Thermal => lerp_palette(&THERMAL_STOPS, t),
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
