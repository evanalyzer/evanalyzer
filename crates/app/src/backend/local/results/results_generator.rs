use crate::api::value_to_color;
use crate::api::*;
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

impl ResultsGenerator {
    pub fn open_database(path: PathBuf) -> Result<Self, InternalErrors> {
        let database = evanalyzer_core::open_results_database(&path)?;
        Ok(Self {
            database,
            classes_cache: RefCell::new(None),
            coloc_classes_cache: RefCell::new(None),
        })
    }

    /// SQL condition that the `images` row `alias` was analysed
    /// successfully on `plane` - i.e. a missing object there really means
    /// "none found", so a Count/Sum of 0 is a real result (see
    /// [`fill_zero_sql`]). A failed image, or a plane the run never
    /// analysed, has no value.
    ///
    /// Which planes a run analysed isn't stored; it's taken from the
    /// objects themselves: every plane between the lowest and highest Z/T
    /// any object was found on (a Z-projection run only produces plane 0),
    /// and within the image's own plane count. So a plane at the edge of
    /// the analysed range on which no image found anything at all, or a
    /// run without a single object, counts as not analysed (empty).
    fn measured_on_plane_sql(
        &self,
        alias: &str,
        plane: &PlaneFilter,
    ) -> Result<String, InternalErrors> {
        let (z, t) = (plane.z_stack as i64, plane.t_stack as i64);
        let range: Option<(i64, i64, i64, i64)> = self
            .database
            .query_row(
                "SELECT MIN(z_stack), MAX(z_stack), MIN(t_stack), MAX(t_stack) FROM objects",
                [],
                |row| {
                    Ok(match (row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?) {
                        (Some(z_min), Some(z_max), Some(t_min), Some(t_max)) => {
                            Some((z_min, z_max, t_min, t_max))
                        }
                        _ => None,
                    })
                },
            )
            .map_err(|e| InternalErrors::Io(e.to_string()))?;
        Ok(match range {
            Some((z_min, z_max, t_min, t_max))
                if (z_min..=z_max).contains(&z) && (t_min..=t_max).contains(&t) =>
            {
                format!(
                    "({alias}.successful AND {z} < {alias}.z_stacks AND {t} < {alias}.t_stacks)"
                )
            }
            _ => "false".to_string(),
        })
    }

    /// A second, independent connection to the same already-open database -
    /// for handing to a background thread (e.g. a potentially long-running
    /// export) that shouldn't have to either hold this `ResultsGenerator`'s
    /// caller's lock for its own duration, or reopen the underlying file at
    /// the OS level. Reopening the same path with a fresh `open_database`
    /// call instead of cloning was the original approach: it works on
    /// Linux/macOS, but not Windows, where the OS enforces exclusive-by-
    /// default file locking even for a second handle opened by the same
    /// process - the app ended up locking itself out of its own database
    /// on every export.
    pub fn try_clone(&self) -> Result<Self, InternalErrors> {
        let to_io_err = |e: duckdb::Error| InternalErrors::Io(e.to_string());
        Ok(Self {
            database: self.database.try_clone().map_err(to_io_err)?,
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
        if filter.transpond_table {
            let blocks = class_blocks(class_ids.as_deref(), &classes);
            return self.get_object_list_transposed(
                filter,
                &conditions,
                &blocks,
                &ordered_columns,
                &classes,
            );
        }
        if let Some(ids) = &class_ids {
            conditions.push(object_class_filter_sql("object_class_id", ids));
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
            "SELECT {}\n FROM objects o LEFT JOIN images i ON i.image_rel_path = o.image_rel_path\n {where_clause}\n ORDER BY o.object_id",
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
        let mut ordered_columns = filter.columns.clone();
        ordered_columns.sort();
        // One (column, aggregation) statistic per value, in output order.
        let stats: Vec<(&Column, &Aggregation)> = ordered_columns
            .iter()
            .flat_map(|column| filter.aggregation.iter().map(move |agg| (column, agg)))
            .collect();

        let mut column_names = vec!["image".to_string(), "class".to_string()];
        for (column, aggregation) in &stats {
            let (agg_fn, _) = aggregate_sql(column, aggregation)?;
            column_names.push(format!("{} ({agg_fn})", column.display_label(&classes)));
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
        if stats.is_empty() {
            return Ok(empty_result());
        }

        // Conditions on the objects before their class lists are unpacked
        // (`pre`) and on the unpacked (object, class) rows (`post`) - see
        // `per_image_class_sql`.
        let mut pre = vec![
            format!("z_stack = {}", filter.plane.z_stack),
            format!("t_stack = {}", filter.plane.t_stack),
        ];
        let mut post = Vec::new();
        let mut image_conditions = vec!["true".to_string()];
        if let Some(rel_paths) = &filter.images {
            if rel_paths.is_empty() {
                return Ok(empty_result());
            }
            let in_list = format!("image_rel_path IN ({})", sql_string_in_list(rel_paths));
            pre.push(in_list.clone());
            image_conditions.push(in_list);
        }
        let selected_ids: Option<Vec<u32>> = filter.object_classes.as_ref().map(|wanted| {
            wanted
                .iter()
                .filter_map(|id| match id {
                    ObjectClass::Valid(n) => Some(*n),
                    ObjectClass::Unset => None,
                })
                .collect()
        });
        if let Some(ids) = &selected_ids {
            if ids.is_empty() {
                return Ok(empty_result());
            }
            pre.push(object_class_filter_sql("object_class_id", ids));
            post.push(format!("class_id IN ({})", sql_u32_list(ids)));
        }
        // The classes every analysed image gets a row (or, transposed, a
        // column block) for, even without objects of that class: its Count
        // is a real 0 then. The selected classes, or every class of the
        // project the results were written for - except Background, which
        // every project has and no object ever gets, so it would only add a
        // 0 row per image (still shown when explicitly selected).
        let mut row_classes = class_blocks(selected_ids.as_deref(), &classes);
        if selected_ids.is_none() {
            row_classes.retain(|id| ObjectClass::Valid(*id) != ObjectClass::BACKGROUND);
        }
        let measured = self.measured_on_plane_sql("images", &filter.plane)?;
        if filter.transpond_table {
            return self.get_grouped_by_image_transposed(
                filter,
                &stats,
                pre,
                image_conditions,
                &measured,
                &row_classes,
                &classes,
            );
        }
        // Keyset pagination over the *groups* (one (image, class) pair =
        // one row here), not over individual objects like
        // `get_object_list` - ordered the same way (`image_rel_path`, then
        // `class_id`), so a row-value comparison against the last page's
        // final group can never split a group across pages. Applied to the
        // objects too, so later pages don't re-aggregate earlier ones.
        let mut key_conditions = vec!["true".to_string()];
        if let Some(cursor) = &filter.page.after {
            let (cursor_path, cursor_class) = cursor
                .split_once('\u{1}')
                .unwrap_or((cursor.as_str(), "-1"));
            let cursor_path = cursor_path.replace('\'', "''");
            let cursor = format!(
                "('{cursor_path}', {})",
                cursor_class.parse::<i64>().unwrap_or(-1)
            );
            pre.push(format!("image_rel_path >= '{cursor_path}'"));
            post.push(format!("(image_rel_path, class_id) > {cursor}"));
            key_conditions.push(format!("(keys.image_rel_path, keys.class_id) > {cursor}"));
        }
        let per_image_class = per_image_class_sql(&stats, &pre, &post)?;
        let image_where = image_conditions.join(" AND ");
        let key_where = key_conditions.join(" AND ");
        let limit = filter.page.limit.max(0);
        let value_cols_sql = stats
            .iter()
            .enumerate()
            .map(|(i, (column, aggregation))| {
                fill_zero_sql(
                    &format!("agg.value_{i}"),
                    column,
                    aggregation,
                    "agg.n_objects IS NULL",
                    "img.measured",
                )
            })
            .collect::<Vec<_>>()
            .join(",\n                ");
        // Every analysed image x every row class, plus whatever (image,
        // class) pairs actually have objects - the latter also covers an
        // image missing from `images` (a run that crashed before
        // finalizing it) or a class missing from `classes`.
        let measured_keys = if row_classes.is_empty() {
            String::new()
        } else {
            format!(
                "SELECT image_rel_path, class_id\n\
                 FROM img, (SELECT UNNEST({}) AS class_id)\n\
                 WHERE img.measured\n\
                 UNION\n",
                sql_int_array_literal(&row_classes)
            )
        };

        let sql = format!(
            "WITH agg AS (\n\
                 {per_image_class}\n\
             ), img AS (\n\
                 SELECT image_rel_path, image_name, NOT successful AS failed,\n\
                     {measured} AS measured\n\
                 FROM images\n\
                 WHERE {image_where}\n\
             ), keys AS (\n\
                 {measured_keys}\
                 SELECT image_rel_path, class_id FROM agg\n\
             )\n\
             SELECT\n\
                 keys.image_rel_path,\n\
                 COALESCE(img.image_name, agg.image_name),\n\
                 keys.class_id,\n\
                 COALESCE(img.failed, false),\n\
                 {value_cols_sql}\n\
             FROM keys\n\
             LEFT JOIN img ON img.image_rel_path = keys.image_rel_path\n\
             LEFT JOIN agg ON agg.image_rel_path = keys.image_rel_path\n\
                 AND agg.class_id = keys.class_id\n\
             WHERE {key_where}\n\
             ORDER BY keys.image_rel_path, keys.class_id\n\
             LIMIT {limit}"
        );

        let mut stmt = self.database.prepare(&sql).map_err(err)?;
        let n = stats.len();
        let groups: Vec<(String, String, u32, bool, Vec<Option<f64>>)> = stmt
            .query_map([], |row| {
                let mut values = Vec::with_capacity(n);
                for i in 0..n {
                    values.push(row.get::<_, Option<f64>>(4 + i)?);
                }
                Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?, values))
            })
            .map_err(err)?
            .collect::<Result<Vec<_>, _>>()
            .map_err(err)?;

        let (min, max) = value_range(groups.iter().flat_map(|(_, _, _, _, values)| values.iter()));
        let row_names = groups
            .iter()
            .map(|(rel_path, _, class_id, _, _)| format!("{rel_path}\u{1}{class_id}"))
            .collect();
        let source_object_count = groups.len();
        let rows: Vec<Vec<Cell>> = groups
            .into_iter()
            .map(|(image_rel_path, image_name, class_id, failed, values)| {
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
                        search_key: search_key.clone(),
                        failed,
                        ..plain_cell()
                    },
                    Cell {
                        value: CellValue::Class((label, color)),
                        bg_color: color,
                        search_key: search_key.clone(),
                        failed,
                        ..plain_cell()
                    },
                ];
                cells.extend(values.into_iter().map(|value| Cell {
                    // No value (an average without objects, a spread of a
                    // single object, ...): empty, not 0 - see
                    // `fill_zero_sql` for when 0 *is* the answer.
                    value: value.map_or(CellValue::Empty, |v| CellValue::Float(v as f32)),
                    search_key: search_key.clone(),
                    failed,
                    ..plain_cell()
                }));
                cells
            })
            .collect();

        Ok(DatabaseResult {
            column_names,
            row_names,
            rows,
            min,
            max,
            source_object_count,
            row_locations: Vec::new(),
        })
    }

    /// `get_grouped_by_image` with `transpond_table`: one row per image, and
    /// per class in `blocks` one column per statistic - the (image, class)
    /// rows of the normal view placed side by side.
    ///
    /// The query only aggregates per (image, class), exactly like the normal
    /// view, for one page of images; placing a class's values side by side
    /// happens here. Doing that in SQL - one `agg(...) FILTER (WHERE
    /// class_id = c)` per output column, over the objects or even over the
    /// grouped rows - cost DuckDB far more than the aggregation itself
    /// (~345 ms / ~125 ms vs ~45 ms for 21 classes x 4 statistics on a
    /// 3.8M-object file). Pages over images (`row_names` holds each row's
    /// `image_rel_path`, the next page's cursor).
    #[allow(clippy::too_many_arguments)]
    fn get_grouped_by_image_transposed(
        &self,
        filter: &GroupedByImageFilter,
        stats: &[(&Column, &Aggregation)],
        mut pre: Vec<String>,
        mut image_conditions: Vec<String>,
        measured: &str,
        blocks: &[u32],
        classes: &[Class],
    ) -> Result<DatabaseResult, InternalErrors> {
        let err = |e: duckdb::Error| InternalErrors::Io(e.to_string());
        let mut column_names = vec!["image".to_string()];
        for class_id in blocks {
            let class_label = class_display_label(ObjectClass::Valid(*class_id), classes);
            for (column, aggregation) in stats {
                let (agg_fn, _) = aggregate_sql(column, aggregation)?;
                column_names.push(format!(
                    "{} ({agg_fn}) ({class_label})",
                    column.display_label(classes)
                ));
            }
        }
        if blocks.is_empty() || stats.is_empty() {
            return Ok(empty_database_result(column_names));
        }

        let post = vec![format!("class_id IN ({})", sql_u32_list(blocks))];
        if let Some(cursor) = &filter.page.after {
            let after = format!("image_rel_path > '{}'", cursor.replace('\'', "''"));
            pre.push(after.clone());
            image_conditions.push(after);
        }
        let per_image_class = per_image_class_sql(stats, &pre, &post)?;
        let limit = filter.page.limit.max(0);
        let values = (0..stats.len())
            .map(|i| format!("agg.value_{i}"))
            .collect::<Vec<_>>()
            .join(", ");
        // The page's images: every analysed one, plus any with objects (an
        // image missing from `images` - a run that crashed before
        // finalizing it - still shows), so an analysed image without
        // objects gets its row, with a Count of 0 per class
        // (`zero_when_no_objects`).
        let sql = format!(
            "WITH agg AS (\n\
                 {per_image_class}\n\
             ), img AS (\n\
                 SELECT image_rel_path, image_name, NOT successful AS failed,\n\
                     {measured} AS measured\n\
                 FROM images\n\
                 WHERE {image_where}\n\
             ), page AS (\n\
                 SELECT image_rel_path FROM (\n\
                     SELECT image_rel_path FROM img WHERE measured\n\
                     UNION\n\
                     SELECT image_rel_path FROM agg\n\
                 )\n\
                 ORDER BY image_rel_path\n\
                 LIMIT {limit}\n\
             )\n\
             SELECT page.image_rel_path, COALESCE(img.image_name, agg.image_name),\n\
                 COALESCE(img.failed, false), COALESCE(img.measured, false),\n\
                 agg.class_id, {values}\n\
             FROM page\n\
             LEFT JOIN img ON img.image_rel_path = page.image_rel_path\n\
             LEFT JOIN agg ON agg.image_rel_path = page.image_rel_path\n\
             ORDER BY page.image_rel_path",
            image_where = image_conditions.join(" AND "),
        );
        let n = stats.len();
        let mut stmt = self.database.prepare(&sql).map_err(err)?;
        // One result row per (image, class) with objects, or a single
        // class-less one for an image without any, ordered by image. Read
        // as a stream and pivoted one image at a time, so only the output
        // rows are ever held, never the grouped rows on top of them.
        let mut result_rows = stmt.query([]).map_err(err)?;
        // (image, name, failed, values in output column order)
        let mut groups: Vec<(String, String, bool, Vec<Option<f64>>)> = Vec::new();
        // The image being collected: its row so far and whether it was
        // analysed, plus which block classes it had objects of.
        let mut current: Option<(String, String, bool, Vec<Option<f64>>, bool)> = None;
        let mut seen_classes: Vec<bool> = vec![false; blocks.len()];
        let finish = |(path, name, failed, mut values, measured): (
            String,
            String,
            bool,
            Vec<Option<f64>>,
            bool,
        ),
                      seen_classes: &[bool]| {
            // A block class without any object in an analysed image: 0
            // where that's the real answer (`zero_when_no_objects`).
            for (block, seen) in seen_classes.iter().enumerate() {
                if *seen || !measured {
                    continue;
                }
                for (i, (column, aggregation)) in stats.iter().enumerate() {
                    if zero_when_no_objects(column, aggregation) {
                        values[block * n + i] = Some(0.0);
                    }
                }
            }
            (path, name, failed, values)
        };
        while let Some(row) = result_rows.next().map_err(err)? {
            let image_rel_path: String = row.get(0).map_err(err)?;
            if current.as_ref().is_none_or(|c| c.0 != image_rel_path) {
                if let Some(done) = current.take() {
                    groups.push(finish(done, &seen_classes));
                }
                seen_classes.fill(false);
                current = Some((
                    image_rel_path,
                    row.get(1).map_err(err)?,
                    row.get(2).map_err(err)?,
                    vec![None; blocks.len() * n],
                    row.get(3).map_err(err)?,
                ));
            }
            let class_id: Option<u32> = row.get(4).map_err(err)?;
            let Some(block) = class_id.and_then(|id| blocks.iter().position(|b| *b == id)) else {
                continue;
            };
            seen_classes[block] = true;
            let values = &mut current.as_mut().expect("set above").3;
            for i in 0..n {
                values[block * n + i] = row.get(5 + i).map_err(err)?;
            }
        }
        if let Some(done) = current.take() {
            groups.push(finish(done, &seen_classes));
        }

        let (min, max) = value_range(groups.iter().flat_map(|(_, _, _, values)| values.iter()));
        let row_names = groups.iter().map(|(path, ..)| path.clone()).collect();
        let source_object_count = groups.len();
        let rows = groups
            .into_iter()
            .map(|(image_rel_path, image_name, failed, values)| {
                let search_key = Some((image_name.clone(), image_rel_path));
                let mut cells = vec![Cell {
                    value: CellValue::String(image_name),
                    search_key: search_key.clone(),
                    failed,
                    ..plain_cell()
                }];
                cells.extend(values.into_iter().map(|value| Cell {
                    // No value of that class in this image: empty, unless
                    // 0 is the real answer (`zero_when_no_objects`).
                    value: value.map_or(CellValue::Empty, |v| CellValue::Float(v as f32)),
                    search_key: search_key.clone(),
                    failed,
                    ..plain_cell()
                }));
                cells
            })
            .collect();
        Ok(DatabaseResult {
            column_names,
            row_names,
            rows,
            min,
            max,
            source_object_count,
            row_locations: Vec::new(),
        })
    }

    /// `get_object_list` with `transpond_table`: the classes side by side.
    /// Per image, the n-th object of each class in `blocks` (ordered by
    /// object id) shares row n, with one block of columns per class; a class
    /// with fewer objects in that image leaves its block empty in the extra
    /// rows. `ImageName`/`ObjectClass` aren't repeated per block - the image
    /// is the first column and the class is the block.
    ///
    /// Two queries, like the normal list: the row layout comes straight from
    /// SQL (`row_number() OVER (PARTITION BY image, class)` gives every
    /// object its row, and the page is cut on (image, row) keys), touching
    /// only narrow id/filter columns; then the page's objects are fetched in
    /// full by id. Pages over (image, row): `row_names` holds
    /// `"{image_rel_path}\u{1}{row}"`, the next page's cursor.
    fn get_object_list_transposed(
        &self,
        filter: &ListFilter,
        base_conditions: &[String],
        blocks: &[u32],
        ordered_columns: &[Column],
        classes: &[Class],
    ) -> Result<DatabaseResult, InternalErrors> {
        let err = |e: duckdb::Error| InternalErrors::Io(e.to_string());
        let block_columns: Vec<Column> = ordered_columns
            .iter()
            .filter(|c| !matches!(c, Column::ImageName | Column::ObjectClass))
            .cloned()
            .collect();
        let mut column_names = vec!["image".to_string()];
        for class_id in blocks {
            let class_label = class_display_label(ObjectClass::Valid(*class_id), classes);
            for column in &block_columns {
                column_names.push(format!("{} ({class_label})", column.display_label(classes)));
            }
        }
        if blocks.is_empty() {
            return Ok(empty_database_result(column_names));
        }

        let mut page_condition = String::new();
        let mut cursor_path = None;
        if let Some(cursor) = &filter.page.after {
            let (path, row) = cursor.split_once('\u{1}').unwrap_or((cursor.as_str(), "0"));
            let path = path.replace('\'', "''");
            let rows_done = row.parse::<i64>().unwrap_or(0);
            page_condition = format!("WHERE (image_rel_path, rn) > ('{path}', {rows_done})");
            cursor_path = Some(path);
        }
        let limit = filter.page.limit.max(0);
        // Without the class condition: it would need the UNNEST, which makes
        // finding the next images 5x slower. An image without objects of the
        // selected classes just contributes no rows below.
        let image_conditions = {
            let mut conditions = base_conditions.to_vec();
            if let Some(path) = &cursor_path {
                conditions.push(format!("image_rel_path >= '{path}'"));
            }
            conditions.join(" AND ")
        };
        let mut conditions = base_conditions.to_vec();
        conditions.push(format!("class_id IN ({})", sql_u32_list(blocks)));
        let conditions = conditions.join(" AND ");

        // Numbering objects into rows needs a sort per (image, class), so
        // only the next few images get numbered: start with 4 and widen 4x
        // while the page isn't full and there are more images - a 500-row
        // page usually needs one or two. Numbering every remaining image
        // instead made a page ~20x slower on 2.4M objects.
        let mut batch = 4;
        let slots = loop {
            let images: Vec<String> = {
                let sql = format!(
                    "SELECT DISTINCT image_rel_path FROM objects WHERE {image_conditions}\n\
                     ORDER BY image_rel_path LIMIT {batch}"
                );
                let mut stmt = self.database.prepare(&sql).map_err(err)?;
                stmt.query_map([], |row| row.get(0))
                    .map_err(err)?
                    .collect::<Result<_, _>>()
                    .map_err(err)?
            };
            if images.is_empty() {
                return Ok(empty_database_result(column_names));
            }
            let sql = format!(
                "WITH ranked AS (\n\
                    SELECT image_rel_path, image_name, class_id, object_id,\n\
                           row_number() OVER (PARTITION BY image_rel_path, class_id ORDER BY object_id) AS rn\n\
                    FROM objects, UNNEST(CAST(object_class_id AS INTEGER[])) AS u(class_id)\n\
                    WHERE {conditions} AND image_rel_path IN ({})\n\
                 ), page AS (\n\
                    SELECT DISTINCT image_rel_path, rn FROM ranked {page_condition}\n\
                    ORDER BY image_rel_path, rn LIMIT {limit}\n\
                 )\n\
                 SELECT r.image_rel_path, r.image_name, r.rn, r.class_id, r.object_id::VARCHAR\n\
                 FROM ranked r JOIN page p ON r.image_rel_path = p.image_rel_path AND r.rn = p.rn\n\
                 ORDER BY r.image_rel_path, r.rn, r.class_id",
                sql_string_in_list(&images)
            );
            let mut stmt = self.database.prepare(&sql).map_err(err)?;
            let slots: Vec<(String, String, i64, u32, String)> = stmt
                .query_map([], |row| {
                    Ok((
                        row.get(0)?,
                        row.get(1)?,
                        row.get(2)?,
                        row.get(3)?,
                        row.get(4)?,
                    ))
                })
                .map_err(err)?
                .collect::<Result<Vec<_>, _>>()
                .map_err(err)?;
            let mut rows: Vec<(&str, i64)> =
                slots.iter().map(|slot| (slot.0.as_str(), slot.2)).collect();
            rows.dedup();
            let page_full = rows.len() >= limit as usize;
            let no_more_images = images.len() < batch;
            if page_full || no_more_images {
                break slots;
            }
            batch *= 4;
        };
        if slots.is_empty() {
            return Ok(empty_database_result(column_names));
        }

        // Full rows for exactly this page's objects (an object of several
        // classes appears in several blocks but is fetched once).
        let mut ids: Vec<String> = slots.iter().map(|slot| slot.4.clone()).collect();
        ids.sort();
        ids.dedup();
        let needs = ObjectColumnNeeds::for_columns(&block_columns);
        let sql = format!(
            "SELECT {}\n FROM objects o LEFT JOIN images i ON i.image_rel_path = o.image_rel_path\n WHERE o.object_id IN ({})",
            object_select_clause(needs),
            sql_string_in_list(&ids)
        );
        let mut stmt = self.database.prepare(&sql).map_err(err)?;
        let objects: HashMap<String, ObjectRow> = stmt
            .query_map([], map_object_row)
            .map_err(err)?
            .map(|row| row.map(|object| (object.object_id.clone(), object)))
            .collect::<Result<_, _>>()
            .map_err(err)?;

        let mut row_names = Vec::new();
        let mut rows: Vec<Vec<Cell>> = Vec::new();
        let mut row_locations = Vec::new();
        let mut index = 0;
        while index < slots.len() {
            let (image_rel_path, image_name, rn, _, _) = &slots[index];
            let end = slots[index..]
                .iter()
                .position(|slot| &slot.0 != image_rel_path || slot.2 != *rn)
                .map_or(slots.len(), |offset| index + offset);
            let row_objects: Vec<(u32, &ObjectRow)> = slots[index..end]
                .iter()
                .filter_map(|slot| objects.get(&slot.4).map(|object| (slot.3, object)))
                .collect();
            let search_key = Some((image_name.clone(), image_rel_path.clone()));
            let mut cells = vec![Cell {
                value: CellValue::String(image_name.clone()),
                search_key,
                disabled: row_objects.iter().any(|(_, object)| object.disabled),
                ..plain_cell()
            }];
            for class_id in blocks {
                match row_objects.iter().find(|(class, _)| class == class_id) {
                    Some((_, object)) => cells.extend(
                        block_columns
                            .iter()
                            .map(|column| cell_for_column(column, object, classes)),
                    ),
                    None => cells.extend(block_columns.iter().map(|_| plain_cell())),
                }
            }
            // Clicking a row opens its image at the row's first object.
            row_locations.push(match row_objects.first() {
                Some((_, object)) => object_location(object),
                None => (image_rel_path.clone(), [0; 4]),
            });
            row_names.push(format!("{image_rel_path}\u{1}{rn}"));
            rows.push(cells);
            index = end;
        }
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
        let dash = |alternating_color: bool, disabled: bool| Cell {
            value: CellValue::String("-".to_string()),
            bg_color: 0,
            alternating_color,
            search_key: None,
            disabled,
            any_disabled: false,
            failed: false,
            any_failed: false,
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
                "SELECT {} FROM objects o LEFT JOIN images i ON i.image_rel_path = o.image_rel_path WHERE o.object_id IN ({})",
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
                                .unwrap_or_else(|| dash(alternating_color, object.disabled))
                        } else {
                            dash(alternating_color, object.disabled)
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

        let mut object_conditions = vec![
            format!("o.z_stack = {}", filter.plane.z_stack),
            format!("o.t_stack = {}", filter.plane.t_stack),
        ];
        if let ObjectClass::Valid(id) = filter.object_class {
            object_conditions.push(object_class_filter_sql("o.object_class_id", &[id]));
        }
        let object_where = object_conditions.join(" AND ");
        // `column_aggregate_expr`'s bare column names (e.g. "area_px") need
        // qualifying against `objects o` now that this SELECT also has
        // `images i` in scope; `Column::Count`'s "*" needs no such prefix.
        let value_expr_sql = if value_expr == "*" {
            value_expr
        } else {
            format!("o.{value_expr}")
        };

        // Two nested queries rather than one flat `images LEFT JOIN
        // objects`: a flat join would put the z/t-stack + class filter in
        // the `ON` clause (it has to, to keep an image with zero matching
        // objects instead of dropping its row), which forces DuckDB to
        // build the join over every one of `objects`' rows before it can
        // apply that filter. Benchmarked on a real ~1.8k-image/5.6M-object
        // database, that flat join took ~165ms/call; filtering+aggregating
        // `objects` down to (at most) one row per group *first*, in its own
        // subquery, then `LEFT JOIN`ing that tiny result onto the group list
        // from `images`, took ~40ms - see `examples/bench_group_by_plate.rs`.
        //
        // A disabled image's objects are always excluded from `value` (the
        // aggregate), and so are a failed image's (`NOT successful`: its
        // analysis stopped with an error, so its objects are incomplete and
        // would undercount the well), but the image itself is never
        // dropped: every well
        // still appears (even one made up only of disabled images, via
        // `img` never filtering on `disabled`), and `any_disabled` -
        // `bool_or(disabled)` per well - tells the caller whether at least
        // one of that well's images was excluded from `value`, so the UI can
        // mark the well without hiding it.
        //
        // `any_measured`: at least one enabled image of the well was
        // analysed on this plane, so a well without objects has a real
        // Count/Sum of 0 (`fill_zero_sql`); a well made up only of
        // disabled, failed or not-analysed images stays empty.
        let value = fill_zero_sql(
            "agg.value",
            &filter.column,
            &filter.aggregation,
            "agg.n_objects IS NULL",
            "img.any_measured",
        );
        let measured = self.measured_on_plane_sql("images", &filter.plane)?;
        let sql = format!(
            "SELECT img.group_prefix, img.row, img.col, {value}, img.any_disabled, img.any_failed\n\
             FROM (\n\
                 SELECT\n\
                     regexp_extract(image_name, '{regex}', 1) AS group_prefix,\n\
                     regexp_extract(image_name, '{regex}', 2) AS row,\n\
                     regexp_extract(image_name, '{regex}', 3) AS col,\n\
                     bool_or(disabled) AS any_disabled,\n\
                     bool_or(NOT successful) AS any_failed,\n\
                     bool_or(NOT disabled AND {measured}) AS any_measured\n\
                 FROM images\n\
                 GROUP BY group_prefix, row, col\n\
             ) img\n\
             LEFT JOIN (\n\
                 SELECT\n\
                     regexp_extract(o.image_name, '{regex}', 1) AS group_prefix,\n\
                     regexp_extract(o.image_name, '{regex}', 2) AS row,\n\
                     regexp_extract(o.image_name, '{regex}', 3) AS col,\n\
                     COUNT(*) AS n_objects,\n\
                     {agg_fn}({value_expr_sql}) AS value\n\
                 FROM objects o\n\
                 JOIN images i ON i.image_rel_path = o.image_rel_path\n\
                 WHERE NOT i.disabled AND i.successful AND {object_where}\n\
                 GROUP BY group_prefix, row, col\n\
             ) agg USING (group_prefix, row, col)\n\
             ORDER BY img.group_prefix",
            regex = regex.replace('\'', "''"),
        );

        let mut stmt = self.database.prepare(&sql).map_err(err)?;
        let groups: Vec<(String, String, String, Option<f64>, ImageFlags)> = stmt
            .query_map([], |row| {
                Ok((
                    row.get(0)?,
                    row.get(1)?,
                    row.get(2)?,
                    row.get(3)?,
                    ImageFlags {
                        disabled: row.get(4)?,
                        failed: row.get(5)?,
                    },
                ))
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

    // Batched form of `get_group_by_plate` across every requested column,
    // aggregation *and* object class at once. The WHERE/GROUP BY is
    // identical across every `column` x `aggregation` combination for a
    // given class - only the aggregate expression itself differs - so all
    // of them are computed in one SELECT per class, same reasoning as
    // `get_wells_for_plate_multi_agg` batching every aggregation. Each
    // class in `filter.object_class` still needs its own query (its own
    // `list_has_any`/no-filter WHERE clause), so this is
    // `object_class.len()` scans total rather than
    // `object_class.len() * column.len() * aggregation.len()`.
    //
    // Returns one `DatabaseResult` per (class, column, aggregation) combo,
    // flattened in that nesting order - i.e. `filter.object_class[0]`'s
    // results (each of its columns, each of that column's aggregations)
    // come first, then `filter.object_class[1]`'s, etc.
    pub fn get_group_by_plate_multi(
        &self,
        filter: &PlateFilterMulti,
        view: &View,
    ) -> Result<Vec<DatabaseResult>, InternalErrors> {
        let err = |e: duckdb::Error| InternalErrors::Io(e.to_string());
        let classes = self.get_object_classes()?;

        let mut value_exprs = Vec::with_capacity(filter.column.len() * filter.aggregation.len());
        let mut agg_value_cols = Vec::with_capacity(value_exprs.capacity());
        for column in &filter.column {
            for aggregation in &filter.aggregation {
                agg_value_cols.push(fill_zero_sql(
                    &format!("agg.value_{}", agg_value_cols.len()),
                    column,
                    aggregation,
                    "agg.n_objects IS NULL",
                    "img.any_measured",
                ));
                let (agg_fn, value_expr) = aggregate_sql(column, aggregation)?;
                // Qualified against `objects o` - see `get_group_by_plate`.
                let value_expr_sql = if value_expr == "*" {
                    value_expr
                } else {
                    format!("o.{value_expr}")
                };
                value_exprs.push(format!(
                    "{agg_fn}({value_expr_sql}) AS value_{}",
                    value_exprs.len()
                ));
            }
        }
        let n = value_exprs.len();
        let value_exprs_sql = value_exprs.join(",\n                    ");

        let regex = if filter.grouping_regex.trim().is_empty() {
            DEFAULT_GROUPING_REGEX
        } else {
            filter.grouping_regex.as_str()
        };
        let regex = regex.replace('\'', "''");

        let measured = self.measured_on_plane_sql("images", &filter.plane)?;
        let mut results = Vec::with_capacity(filter.object_class.len() * n);
        for object_class in &filter.object_class {
            let mut object_conditions = vec![
                format!("o.z_stack = {}", filter.plane.z_stack),
                format!("o.t_stack = {}", filter.plane.t_stack),
            ];
            if let ObjectClass::Valid(id) = object_class {
                object_conditions.push(object_class_filter_sql("o.object_class_id", &[*id]));
            }
            let object_where = object_conditions.join(" AND ");
            let agg_value_cols = agg_value_cols.join(", ");

            // Same "filter+aggregate `objects` before joining" shape as
            // `get_group_by_plate` - see its comment for why (benchmarked
            // ~4x faster than a flat `images LEFT JOIN objects` on a real
            // multi-million-object database) and for `any_disabled`/
            // `any_measured`.
            let sql = format!(
                "SELECT img.group_prefix, img.row, img.col, {agg_value_cols}, img.any_disabled,\n\
                     img.any_failed\n\
                 FROM (\n\
                     SELECT\n\
                         regexp_extract(image_name, '{regex}', 1) AS group_prefix,\n\
                         regexp_extract(image_name, '{regex}', 2) AS row,\n\
                         regexp_extract(image_name, '{regex}', 3) AS col,\n\
                         bool_or(disabled) AS any_disabled,\n\
                         bool_or(NOT successful) AS any_failed,\n\
                         bool_or(NOT disabled AND {measured}) AS any_measured\n\
                     FROM images\n\
                     GROUP BY group_prefix, row, col\n\
                 ) img\n\
                 LEFT JOIN (\n\
                     SELECT\n\
                         regexp_extract(o.image_name, '{regex}', 1) AS group_prefix,\n\
                         regexp_extract(o.image_name, '{regex}', 2) AS row,\n\
                         regexp_extract(o.image_name, '{regex}', 3) AS col,\n\
                         COUNT(*) AS n_objects,\n\
                         {value_exprs_sql}\n\
                     FROM objects o\n\
                     JOIN images i ON i.image_rel_path = o.image_rel_path\n\
                     WHERE NOT i.disabled AND i.successful AND {object_where}\n\
                     GROUP BY group_prefix, row, col\n\
                 ) agg USING (group_prefix, row, col)\n\
                 ORDER BY img.group_prefix"
            );

            let mut stmt = self.database.prepare(&sql).map_err(err)?;
            let raw: Vec<(String, String, String, Vec<Option<f64>>, ImageFlags)> = stmt
                .query_map([], |row| {
                    let group_prefix: String = row.get(0)?;
                    let group_row: String = row.get(1)?;
                    let group_col: String = row.get(2)?;
                    let mut values = Vec::with_capacity(n);
                    for i in 0..n {
                        values.push(row.get::<_, Option<f64>>(3 + i)?);
                    }
                    let flags = ImageFlags {
                        disabled: row.get(3 + n)?,
                        failed: row.get(4 + n)?,
                    };
                    Ok((group_prefix, group_row, group_col, values, flags))
                })
                .map_err(err)?
                .collect::<Result<Vec<_>, _>>()
                .map_err(err)?;

            let mut combo = 0;
            for column in &filter.column {
                for _aggregation in &filter.aggregation {
                    let groups: Vec<(String, String, String, Option<f64>, ImageFlags)> = raw
                        .iter()
                        .map(|(g, r, c, values, flags)| {
                            (g.clone(), r.clone(), c.clone(), values[combo], *flags)
                        })
                        .collect();
                    results.push(plate_groups_to_result(
                        groups,
                        column,
                        &classes,
                        filter.matrix_dimension,
                        &filter.color_schema,
                        &filter.color_scale,
                        view,
                    ));
                    combo += 1;
                }
            }
        }

        Ok(results)
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

        let mut object_conditions = vec![
            format!("z_stack = {}", filter.plane.z_stack),
            format!("t_stack = {}", filter.plane.t_stack),
        ];
        if let ObjectClass::Valid(id) = filter.object_class {
            object_conditions.push(object_class_filter_sql("object_class_id", &[id]));
        }
        let object_where = object_conditions.join(" AND ");

        // Driven from `images` (one row per field, `image_rel_path` is its
        // primary key - no fan-out, unlike the plate view's regex-bucketed
        // groups) so a field with zero matching objects still gets a tile.
        // `objects` is filtered and aggregated down to one row per image
        // *before* the join, same reasoning as `get_group_by_plate`. A
        // disabled image's field is shown like any other (its own value is
        // still its own real aggregate - "disabled" only matters once
        // several images get combined into one statistic, which never
        // happens at this single-image granularity), just flagged via
        // `i.disabled` so the UI can mark it as excluded from any
        // *plate*-level statistic.
        //
        // A field without objects gets a Count/Sum of 0 if it was analysed
        // on this plane (`fill_zero_sql`) - its own value, so disabled or
        // not doesn't matter here either.
        let value = fill_zero_sql(
            "agg.value",
            &filter.column,
            &filter.aggregation,
            "agg.n_objects IS NULL",
            &self.measured_on_plane_sql("i", &filter.plane)?,
        );
        let sql = format!(
            "SELECT\n\
                regexp_extract(i.image_name, '{regex}', 4) AS idx,\n\
                i.image_rel_path,\n\
                i.image_name,\n\
                {value},\n\
                i.disabled,\n\
                NOT i.successful\n\
             FROM images i\n\
             LEFT JOIN (\n\
                 SELECT image_rel_path, COUNT(*) AS n_objects, {agg_fn}({value_expr}) AS value\n\
                 FROM objects\n\
                 WHERE {object_where}\n\
                 GROUP BY image_rel_path\n\
             ) agg ON agg.image_rel_path = i.image_rel_path\n\
             WHERE regexp_extract(i.image_name, '{regex}', 1) = '{group_name}'\n\
             ORDER BY idx",
            group_name = filter.group_name.replace('\'', "''"),
        );

        let mut stmt = self.database.prepare(&sql).map_err(err)?;
        // (idx, image_rel_path, image_name, value, disabled) —
        // `image_rel_path` and `image_name` are carried through into
        // `Cell::search_key` on every cell for this field so the GUI can
        // select/open the underlying image from a well-view tile (see
        // `ImageEntry`, which the GUI matches images against by `rel_path`).
        let fields: Vec<(String, String, String, Option<f64>, ImageFlags)> = stmt
            .query_map([], |row| {
                Ok((
                    row.get(0)?,
                    row.get(1)?,
                    row.get(2)?,
                    row.get(3)?,
                    ImageFlags {
                        disabled: row.get(4)?,
                        failed: row.get(5)?,
                    },
                ))
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

        let mut object_conditions = vec![
            format!("z_stack = {}", filter.plane.z_stack),
            format!("t_stack = {}", filter.plane.t_stack),
        ];
        if let ObjectClass::Valid(id) = filter.object_class {
            object_conditions.push(object_class_filter_sql("object_class_id", &[id]));
        }
        let object_where = object_conditions.join(" AND ");

        // Same shape as `get_group_by_well` (one row per field/image,
        // objects filtered+aggregated before the join, disabled images
        // shown - not dropped - and flagged via `i.disabled`), just without
        // the single-well filter - every well's fields in one query.
        let value = fill_zero_sql(
            "agg.value",
            &filter.column,
            &filter.aggregation,
            "agg.n_objects IS NULL",
            &self.measured_on_plane_sql("i", &filter.plane)?,
        );
        let sql = format!(
            "SELECT\n\
                regexp_extract(i.image_name, '{regex}', 1) AS group_prefix,\n\
                regexp_extract(i.image_name, '{regex}', 4) AS idx,\n\
                i.image_rel_path,\n\
                i.image_name,\n\
                {value},\n\
                i.disabled,\n\
                NOT i.successful\n\
             FROM images i\n\
             LEFT JOIN (\n\
                 SELECT image_rel_path, COUNT(*) AS n_objects, {agg_fn}({value_expr}) AS value\n\
                 FROM objects\n\
                 WHERE {object_where}\n\
                 GROUP BY image_rel_path\n\
             ) agg ON agg.image_rel_path = i.image_rel_path\n\
             ORDER BY group_prefix, idx"
        );

        let mut stmt = self.database.prepare(&sql).map_err(err)?;
        let mut fields_by_well: HashMap<
            String,
            Vec<(String, String, String, Option<f64>, ImageFlags)>,
        > = HashMap::new();
        let rows = stmt
            .query_map([], |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, String>(2)?,
                    row.get::<_, String>(3)?,
                    row.get::<_, Option<f64>>(4)?,
                    ImageFlags {
                        disabled: row.get(5)?,
                        failed: row.get(6)?,
                    },
                ))
            })
            .map_err(err)?
            .collect::<Result<Vec<_>, _>>()
            .map_err(err)?;
        for (group_prefix, idx, image_rel_path, image_name, value, flags) in rows {
            fields_by_well.entry(group_prefix).or_default().push((
                idx,
                image_rel_path,
                image_name,
                value,
                flags,
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

    // Batched form of `get_wells_for_plate` across every requested column,
    // aggregation *and* object class at once - same reasoning as
    // `get_group_by_plate_multi`: the WHERE/GROUP BY is identical across
    // every `column` x `aggregation` combination for a given class, so all
    // of them are computed in one SELECT per class (54 wells x 7
    // aggregations x 5 columns = 1890 full scans collapsing to exactly 1
    // per class, rather than one scan per (class, column, aggregation)
    // combination). Each class in `filter.object_class` still needs its own
    // query (its own `list_has_any`/no-filter WHERE clause).
    //
    // Returns one `HashMap<well_id, DatabaseResult>` per (class, column,
    // aggregation) combo, flattened in that nesting order - matching
    // `get_group_by_plate_multi`'s own ordering.
    pub fn get_wells_for_plate_multi(
        &self,
        filter: &WellsBatchFilterMulti,
        view: &View,
    ) -> Result<Vec<HashMap<String, DatabaseResult>>, InternalErrors> {
        let err = |e: duckdb::Error| InternalErrors::Io(e.to_string());
        let classes = self.get_object_classes()?;

        let measured = self.measured_on_plane_sql("i", &filter.plane)?;
        let mut value_exprs = Vec::with_capacity(filter.column.len() * filter.aggregation.len());
        let mut agg_value_cols = Vec::with_capacity(value_exprs.capacity());
        for column in &filter.column {
            for aggregation in &filter.aggregation {
                agg_value_cols.push(fill_zero_sql(
                    &format!("agg.value_{}", agg_value_cols.len()),
                    column,
                    aggregation,
                    "agg.n_objects IS NULL",
                    &measured,
                ));
                let (agg_fn, value_expr) = aggregate_sql(column, aggregation)?;
                value_exprs.push(format!(
                    "{agg_fn}({value_expr}) AS value_{}",
                    value_exprs.len()
                ));
            }
        }
        let n = value_exprs.len();
        let value_exprs_sql = value_exprs.join(",\n                ");
        let agg_value_cols = agg_value_cols.join(", ");

        let regex = if filter.grouping_regex.trim().is_empty() {
            DEFAULT_GROUPING_REGEX
        } else {
            filter.grouping_regex.as_str()
        };
        let regex = regex.replace('\'', "''");

        let mut results = Vec::with_capacity(filter.object_class.len() * n);
        for object_class in &filter.object_class {
            let mut object_conditions = vec![
                format!("z_stack = {}", filter.plane.z_stack),
                format!("t_stack = {}", filter.plane.t_stack),
            ];
            if let ObjectClass::Valid(id) = object_class {
                object_conditions.push(object_class_filter_sql("object_class_id", &[*id]));
            }
            let object_where = object_conditions.join(" AND ");

            // Same shape as `get_wells_for_plate` (one row per field/image,
            // objects filtered+aggregated before the join, disabled images
            // shown - not dropped - and flagged via `i.disabled`), batched
            // across every column x aggregation combo like
            // `get_group_by_plate_multi`.
            let sql = format!(
                "SELECT\n\
                    regexp_extract(i.image_name, '{regex}', 1) AS group_prefix,\n\
                    regexp_extract(i.image_name, '{regex}', 4) AS idx,\n\
                    i.image_rel_path,\n\
                    i.image_name,\n\
                    {agg_value_cols},\n\
                    i.disabled,\n\
                    NOT i.successful\n\
                 FROM images i\n\
                 LEFT JOIN (\n\
                     SELECT image_rel_path, COUNT(*) AS n_objects, {value_exprs_sql}\n\
                     FROM objects\n\
                     WHERE {object_where}\n\
                     GROUP BY image_rel_path\n\
                 ) agg ON agg.image_rel_path = i.image_rel_path\n\
                 ORDER BY group_prefix, idx"
            );

            let mut stmt = self.database.prepare(&sql).map_err(err)?;
            let mut fields_by_well: HashMap<
                String,
                Vec<(String, String, String, Vec<Option<f64>>, ImageFlags)>,
            > = HashMap::new();
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
                    let flags = ImageFlags {
                        disabled: row.get(4 + n)?,
                        failed: row.get(5 + n)?,
                    };
                    Ok((group_prefix, idx, image_rel_path, image_name, values, flags))
                })
                .map_err(err)?
                .collect::<Result<Vec<_>, _>>()
                .map_err(err)?;
            for (group_prefix, idx, image_rel_path, image_name, values, flags) in rows {
                fields_by_well.entry(group_prefix).or_default().push((
                    idx,
                    image_rel_path,
                    image_name,
                    values,
                    flags,
                ));
            }

            let mut combo = 0;
            for column in &filter.column {
                for _aggregation in &filter.aggregation {
                    let by_well: HashMap<String, DatabaseResult> = fields_by_well
                        .iter()
                        .map(|(well_id, fields)| {
                            let per_combo_fields: Vec<(
                                String,
                                String,
                                String,
                                Option<f64>,
                                ImageFlags,
                            )> = fields
                                .iter()
                                .map(|(idx, rel_path, name, values, flags)| {
                                    (
                                        idx.clone(),
                                        rel_path.clone(),
                                        name.clone(),
                                        values[combo],
                                        *flags,
                                    )
                                })
                                .collect();
                            let result = well_fields_to_result(
                                per_combo_fields,
                                column,
                                &classes,
                                filter.well_size,
                                &filter.well_order,
                                &filter.color_schema,
                                &filter.color_scale,
                                view,
                            );
                            (well_id.clone(), result)
                        })
                        .collect();
                    results.push(by_well);
                    combo += 1;
                }
            }
        }

        Ok(results)
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

        let (width, height, measured): (u32, u32, bool) = self
            .database
            .query_row(
                &format!(
                    "SELECT width, height, {}\n\
                     FROM images WHERE image_rel_path = '{image_rel_path}'",
                    self.measured_on_plane_sql("images", &filter.plane)?
                ),
                [],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
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
            conditions.push(object_class_filter_sql("object_class_id", &[id]));
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
        // A square without objects of an analysed image: a Count/Sum of 0
        // is the real answer there (see `fill_zero_sql`). `raw_cells` has a
        // row for every square with objects, value or not, so only squares
        // without any object are filled.
        if measured && zero_when_no_objects(&filter.column, &filter.aggregation) {
            let with_objects: std::collections::HashSet<(usize, usize)> = raw_cells
                .iter()
                .filter_map(|(col, row, _)| {
                    let (col, row) = (usize::try_from(*col).ok()?, usize::try_from(*row).ok()?);
                    Some((row.min(rows - 1), col.min(cols - 1)))
                })
                .collect();
            for row in 0..rows {
                for col in 0..cols {
                    if !with_objects.contains(&(row, col)) {
                        values.insert((row, col), 0.0);
                    }
                }
            }
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
                                disabled: false,
                                any_disabled: false,
                                failed: false,
                                any_failed: false,
                            },
                            Cell {
                                value: CellValue::Float(*value as f32),
                                bg_color: 0,
                                alternating_color: false,
                                search_key,
                                disabled: false,
                                any_disabled: false,
                                failed: false,
                                any_failed: false,
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
                                        disabled: false,
                                        any_disabled: false,
                                        failed: false,
                                        any_failed: false,
                                    }
                                }
                                // No value for this tile (no objects in a
                                // statistic that needs some, or the image
                                // wasn't analysed on this plane) - leave it
                                // empty rather than showing a misleading 0
                                // or a neighboring tile's value.
                                None => Cell {
                                    value: CellValue::Empty,
                                    bg_color: 0,
                                    alternating_color: false,
                                    search_key: None,
                                    disabled: false,
                                    any_disabled: false,
                                    failed: false,
                                    any_failed: false,
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

    /// Enables or disables `image_rel_path` (the `images` table's primary
    /// key) for statistics: a disabled image's own value is still shown
    /// everywhere it appears, but is excluded from any aggregate combining
    /// it with other images (see `get_group_by_plate`'s `value`/
    /// `any_disabled` and `Cell::disabled`). A no-op if `image_rel_path`
    /// doesn't match any row.
    pub fn enable_image(&self, image_rel_path: &str, disable: bool) -> Result<(), InternalErrors> {
        let err = |e: duckdb::Error| InternalErrors::Io(e.to_string());
        self.database
            .execute(
                "UPDATE images SET disabled = ? WHERE image_rel_path = ?",
                duckdb::params![disable, image_rel_path],
            )
            .map_err(err)?;
        Ok(())
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
    disabled: bool,
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
    let select_image_name = if need.image_name {
        "o.image_name"
    } else {
        "''"
    };
    let select_object_class_name = if need.class {
        "CAST(o.object_class_name AS VARCHAR[])"
    } else {
        "CAST(NULL AS VARCHAR[])"
    };
    let select_seg_class_name = if need.class {
        "o.seg_class_name"
    } else {
        "NULL::VARCHAR"
    };
    let select_area_px = if need.area_px {
        "o.area_px"
    } else {
        "0::UBIGINT"
    };
    let select_area_nm2 = if need.area_nm2 {
        "o.area_nm2"
    } else {
        "0.0::DOUBLE"
    };
    let select_perimeter_px = if need.perimeter_px {
        "o.perimeter_px"
    } else {
        "0.0::DOUBLE"
    };
    let select_perimeter_nm = if need.perimeter_nm {
        "o.perimeter_nm"
    } else {
        "0.0::DOUBLE"
    };
    let select_circularity = if need.circularity {
        "o.circularity"
    } else {
        "0.0::DOUBLE"
    };
    let select_solidity = if need.solidity {
        "o.solidity"
    } else {
        "0.0::DOUBLE"
    };
    let select_eccentricity = if need.eccentricity {
        "o.eccentricity"
    } else {
        "0.0::DOUBLE"
    };
    let select_coloc_json = if need.coloc {
        "o.coloc_json"
    } else {
        "NULL::VARCHAR"
    };
    let select_intensities_json = if need.intensities {
        "o.intensities_json"
    } else {
        "NULL::VARCHAR"
    };
    format!(
        "o.object_id, {select_image_name}, {select_object_class_name}, {select_seg_class_name},\n\
                {select_area_px}, {select_area_nm2}, {select_perimeter_px}, {select_perimeter_nm},\n\
                {select_circularity}, {select_solidity}, {select_eccentricity},\n\
                {select_coloc_json}, {select_intensities_json},\n\
                o.image_rel_path, o.bbox_xmin_px, o.bbox_ymin_px, o.bbox_xmax_px, o.bbox_ymax_px,\n\
                COALESCE(i.disabled, false)"
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
        disabled: row.get(18)?,
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

/// The classes a transposed view puts side by side, in id order: the
/// selected ones (`selected`, already validated), or every registered class.
fn class_blocks(selected: Option<&[u32]>, classes: &[Class]) -> Vec<u32> {
    let mut ids: Vec<u32> = match selected {
        Some(ids) => ids.to_vec(),
        None => classes
            .iter()
            .filter_map(|class| match class.id {
                ObjectClass::Valid(n) => Some(n),
                ObjectClass::Unset => None,
            })
            .collect(),
    };
    ids.sort_unstable();
    ids.dedup();
    ids
}

/// SQL condition that the object's `column` (its `object_class_id`, a JSON
/// list stored as text, e.g. `"[1, 3]"`) contains any of `ids`.
///
/// Parsing that text into a list for every object is what made a class
/// filter expensive (~47 ms vs ~10 ms on a 3.8M-object file); a results
/// file only has a handful of distinct class combinations, so they're
/// parsed once each and the objects matched by their unparsed text.
pub(super) fn object_class_filter_sql(column: &str, ids: &[u32]) -> String {
    format!(
        "{column} IN (\
         SELECT class_list FROM (SELECT DISTINCT object_class_id AS class_list FROM objects) \
         WHERE list_has_any(CAST(class_list AS INTEGER[]), {}))",
        sql_int_array_literal(ids)
    )
}

/// The objects grouped per (image, class), one `value_{i}` per statistic
/// in `stats` plus `n_objects` - the first stage of both per-image views.
///
/// `object_class_id` is a per-object list (multi-class objects exist),
/// unpacked into one `class_id` per (object, class) pair before grouping;
/// an object without any class contributes no row. `pre` filters the
/// objects *before* that unpacking - DuckDB otherwise parses every object's
/// class list first (~48 ms vs ~30 ms on a 3.8M-object file) - and `post`
/// the unpacked rows (conditions on `class_id`).
fn per_image_class_sql(
    stats: &[(&Column, &Aggregation)],
    pre: &[String],
    post: &[String],
) -> Result<String, InternalErrors> {
    let mut values = Vec::with_capacity(stats.len());
    for (i, (column, aggregation)) in stats.iter().enumerate() {
        let (agg_fn, value_expr) = aggregate_sql(column, aggregation)?;
        values.push(format!("{agg_fn}({value_expr}) AS value_{i}"));
    }
    let post = if post.is_empty() {
        "true".to_string()
    } else {
        post.join(" AND ")
    };
    Ok(format!(
        "SELECT image_rel_path, MIN(image_name) AS image_name, class_id,\n\
             COUNT(*) AS n_objects, {values}\n\
         FROM (SELECT * FROM objects WHERE {pre}) AS o,\n\
             UNNEST(CAST(o.object_class_id AS INTEGER[])) AS u(class_id)\n\
         WHERE {post}\n\
         GROUP BY image_rel_path, class_id",
        values = values.join(", "),
        pre = pre.join(" AND "),
    ))
}

/// Comma-joined integers for a SQL `IN (...)` list.
fn sql_u32_list(values: &[u32]) -> String {
    values
        .iter()
        .map(|v| v.to_string())
        .collect::<Vec<_>>()
        .join(", ")
}

/// An empty, unstyled cell - also the filler for a class block with no
/// object in a transposed row.
fn plain_cell() -> Cell {
    Cell {
        value: CellValue::Empty,
        bg_color: 0,
        alternating_color: false,
        search_key: None,
        disabled: false,
        any_disabled: false,
        failed: false,
        any_failed: false,
    }
}

fn empty_database_result(column_names: Vec<String>) -> DatabaseResult {
    DatabaseResult {
        column_names,
        row_names: vec![],
        rows: vec![],
        min: 0.0,
        max: 0.0,
        source_object_count: 0,
        row_locations: vec![],
    }
}

/// Min and max of the present values, `(0, 0)` if there are none.
fn value_range<'a>(values: impl Iterator<Item = &'a Option<f64>>) -> (f32, f32) {
    let (min, max) = values
        .flatten()
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), v| {
            (lo.min(*v), hi.max(*v))
        });
    if min.is_finite() && max.is_finite() {
        (min as f32, max as f32)
    } else {
        (0.0, 0.0)
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
        disabled: object.disabled,
        any_disabled: false,
        failed: false,
        any_failed: false,
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
                disabled: object.disabled,
                any_disabled: false,
                failed: false,
                any_failed: false,
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

/// Whether `column`/`aggregation` has a real value for a group without any
/// objects: a count of 0, and an (empty) sum of 0. Every other statistic
/// (average, min, max, median, spread) doesn't exist without values and
/// stays empty - a 0 there would be a made-up measurement.
fn zero_when_no_objects(column: &Column, aggregation: &Aggregation) -> bool {
    matches!(column, Column::Count) || matches!(aggregation, Aggregation::Sum)
}

/// `value` (an aggregate over a group's objects), but 0 instead of NULL
/// when the group has no objects (`no_objects`) on images that were
/// analysed (`measured`, see `ResultsGenerator::measured_on_plane_sql`) -
/// for the statistics [`zero_when_no_objects`] allows. Keyed on "no
/// objects" rather than on NULL alone, so a sum over objects whose own
/// value is missing (e.g. nm areas without pixel sizes) stays empty.
fn fill_zero_sql(
    value: &str,
    column: &Column,
    aggregation: &Aggregation,
    no_objects: &str,
    measured: &str,
) -> String {
    if zero_when_no_objects(column, aggregation) {
        format!("CASE WHEN {no_objects} AND {measured} THEN 0 ELSE {value} END")
    } else {
        value.to_string()
    }
}

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

/// Why an image behind a plate/well cell is marked: disabled by the user,
/// or failed (its analysis stopped with an error, so its objects are
/// incomplete). For a plate well, whether *any* of its images is; neither
/// kind contributes to a well's value.
#[derive(Debug, Clone, Copy, Default)]
struct ImageFlags {
    disabled: bool,
    failed: bool,
}

/// Turns raw `(group_prefix, row, col, value, flags)` plate-group
/// rows into a `DatabaseResult` — shared by `get_group_by_plate` (one
/// aggregation per call) and `get_group_by_plate_multi_agg` (every requested
/// aggregation in one batched query, calling this once per aggregation over
/// its own slice of that batch) so the two agree on exactly the same
/// List/Heatmap shape. `flags` says whether at least one image in that
/// well/group is disabled or failed — `value` itself never includes such an
/// image's objects (see `get_group_by_plate`). A well is never itself
/// rendered as "disabled"/"failed" though (only individual images are - a
/// well with a mix of images still shows its normal heatmap color), so the
/// flags only go into `Cell::any_disabled`/`Cell::any_failed` here.
fn plate_groups_to_result(
    groups: Vec<(String, String, String, Option<f64>, ImageFlags)>,
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
            for (_, _, _, value, _) in &groups {
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
                .map(|(key, _row, _col, value, flags)| {
                    // `key` is the group/well id (e.g. "A1") itself, so
                    // it's its own search key — used by the GUI to
                    // navigate into that group/well. A well itself is
                    // never "disabled" - only the individual images inside
                    // it are - so `any_disabled` must not gray/strike this
                    // row; it only ever excluded a disabled image's
                    // objects from `value` above. It's still carried into
                    // `Cell::any_disabled` though, so the GUI can mark the
                    // well as containing a disabled image without
                    // recoloring it.
                    let search_key = Some((key.clone(), key.clone()));
                    vec![
                        Cell {
                            value: CellValue::String(key),
                            bg_color: 0,
                            alternating_color: false,
                            search_key: search_key.clone(),
                            disabled: false,
                            any_disabled: flags.disabled,
                            failed: false,
                            any_failed: flags.failed,
                        },
                        Cell {
                            value: value.map_or(CellValue::Empty, |v| CellValue::Float(v as f32)),
                            bg_color: 0,
                            alternating_color: false,
                            search_key,
                            disabled: false,
                            any_disabled: flags.disabled,
                            failed: false,
                            any_failed: flags.failed,
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
            // it can be returned as `Cell::search_key` below. Every well
            // with valid coordinates is inserted regardless of whether it
            // has a value, so a well made up only of disabled images still
            // reports `any_disabled` even though its cell renders empty.
            let mut values: HashMap<(usize, usize), (Option<f64>, String, ImageFlags)> =
                HashMap::new();
            let mut max_row = None;
            let mut max_col = None;
            for (group_prefix, row, col, value, flags) in &groups {
                let (Some(row), Some(col)) = (row_letter_to_index(row), col_number_to_index(col))
                else {
                    continue;
                };
                max_row = Some(max_row.map_or(row, |m: usize| m.max(row)));
                max_col = Some(max_col.map_or(col, |m: usize| m.max(col)));
                values.insert((row, col), (*value, group_prefix.clone(), *flags));
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
                    for (value, _, _) in values.values() {
                        if let Some(value) = value {
                            min = min.min(*value);
                            max = max.max(*value);
                        }
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
                            // A well itself is never "disabled" - only the
                            // individual images inside it are - so a well
                            // with a mix of enabled/disabled images still
                            // renders its heatmap color here, same as any
                            // other well; `any_disabled` only ever excluded
                            // a disabled image's objects from `value`. It's
                            // still carried into `Cell::any_disabled` so the
                            // GUI can badge the well without recoloring it.
                            Some((value, group_prefix, flags)) => Cell {
                                value: value
                                    .map_or(CellValue::Empty, |v| CellValue::Float(v as f32)),
                                bg_color: value.map_or(0, |v| {
                                    value_to_color(v, range_min, range_max, color_schema)
                                }),
                                alternating_color: false,
                                search_key: Some((group_prefix.clone(), group_prefix.clone())),
                                disabled: false,
                                any_disabled: flags.disabled,
                                failed: false,
                                any_failed: flags.failed,
                            },
                            // No well at all matched this grid position —
                            // leave it empty rather than showing a
                            // misleading 0 or a value from some other well.
                            None => Cell {
                                value: CellValue::Empty,
                                bg_color: 0,
                                alternating_color: false,
                                search_key: None,
                                disabled: false,
                                any_disabled: false,
                                failed: false,
                                any_failed: false,
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
    fields: Vec<(String, String, String, Option<f64>, ImageFlags)>,
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
            for (_, _, _, value, _) in &fields {
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
                .map(|(idx, image_rel_path, image_name, value, flags)| {
                    let search_key = Some((image_name, image_rel_path));
                    vec![
                        Cell {
                            value: CellValue::String(idx),
                            bg_color: 0,
                            alternating_color: false,
                            search_key: search_key.clone(),
                            disabled: flags.disabled,
                            any_disabled: false,
                            failed: flags.failed,
                            any_failed: false,
                        },
                        Cell {
                            value: value.map_or(CellValue::Empty, |v| CellValue::Float(v as f32)),
                            bg_color: 0,
                            alternating_color: false,
                            search_key,
                            disabled: flags.disabled,
                            any_disabled: false,
                            failed: flags.failed,
                            any_failed: false,
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

            // Every field gets a grid position regardless of whether it has
            // a value, same reasoning as `plate_groups_to_result`'s Heatmap
            // arm - a field genuinely has no objects is a different thing
            // from "no field at all sits here", and only the latter should
            // render as if the tile doesn't exist.
            let mut values: HashMap<usize, (Option<f64>, String, String, ImageFlags)> =
                HashMap::new();
            for (idx_str, image_rel_path, image_name, value, disabled) in &fields {
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
                values.insert(
                    position,
                    (
                        *value,
                        image_name.clone(),
                        image_rel_path.clone(),
                        *disabled,
                    ),
                );
            }

            let (range_min, range_max) = match color_scale {
                ColorScale::Manual(min, max) => (*min as f64, *max as f64),
                ColorScale::Auto => {
                    let mut min = f64::INFINITY;
                    let mut max = f64::NEG_INFINITY;
                    for (value, _, _, flags) in values.values() {
                        if let (Some(value), false) = (value, flags.disabled) {
                            min = min.min(*value);
                            max = max.max(*value);
                        }
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
                            Some((value, image_name, image_rel_path, flags)) => Cell {
                                value: value
                                    .map_or(CellValue::Empty, |v| CellValue::Float(v as f32)),
                                bg_color: value.map_or(0, |v| {
                                    value_to_color(v, range_min, range_max, color_schema)
                                }),
                                alternating_color: false,
                                search_key: Some((image_name.clone(), image_rel_path.clone())),
                                disabled: flags.disabled,
                                any_disabled: false,
                                failed: flags.failed,
                                any_failed: false,
                            },
                            // No field occupies this grid position at all -
                            // leave it empty rather than showing a
                            // misleading 0 or another field's value.
                            None => Cell {
                                value: CellValue::Empty,
                                bg_color: 0,
                                alternating_color: false,
                                search_key: None,
                                disabled: false,
                                any_disabled: false,
                                failed: false,
                                any_failed: false,
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

#[cfg(test)]
mod tests {
    use super::*;

    /// Performance of the List view, normal vs. transposed, on a large
    /// synthetic database. Not a correctness test - run explicitly:
    ///
    /// ```sh
    /// BENCH_IMAGES=400 BENCH_CLASSES=4 BENCH_PER_CLASS=300 \
    ///   cargo test --release -p evanalyzer_app --lib bench_transposed_list \
    ///   -- --ignored --nocapture
    /// ```
    ///
    /// Measures what a user actually waits for: the first page, paging on
    /// (GUI scrolling, 500 rows/page like `LIST_PAGE_SIZE`), and walking
    /// every page (what an export does). For grouped-by-image it also times
    /// transposing in Rust after the normal query, the alternative to doing
    /// it in SQL.
    #[test]
    #[ignore]
    fn bench_transposed_list() {
        use super::super::test_support::seed_synthetic_db;
        use std::time::{Duration, Instant};
        let env = |name: &str, default: u32| {
            std::env::var(name)
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(default)
        };
        let (images, classes, per_class) = (
            env("BENCH_IMAGES", 400),
            env("BENCH_CLASSES", 4),
            env("BENCH_PER_CLASS", 300),
        );
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("bench.evadb");
        let start = Instant::now();
        seed_synthetic_db(&path, images, classes, per_class);
        let generator = ResultsGenerator::open_database(path).unwrap();
        println!(
            "\n{} objects ({images} images x {classes} classes x {per_class}), seeded in {:?}",
            images * classes * per_class,
            start.elapsed()
        );
        let columns = vec![
            Column::ImageName,
            Column::ObjectClass,
            Column::AreaSizePx,
            Column::Circularity,
            Column::IntensityAvg(0),
        ];
        const PAGE: i32 = 500;
        let generator = &generator;
        let columns = &columns;

        // Walks up to `max_pages` pages; returns (first page, avg per page,
        // pages, rows).
        let walk = |fetch: &dyn Fn(Option<String>) -> DatabaseResult, max_pages: usize| {
            let mut cursor = None;
            let mut times = Vec::new();
            let mut rows = 0;
            for _ in 0..max_pages {
                let start = Instant::now();
                let page = fetch(cursor.take());
                times.push(start.elapsed());
                rows += page.rows.len();
                cursor = page.row_names.last().cloned();
                if page.rows.len() < PAGE as usize {
                    break;
                }
            }
            let total: Duration = times.iter().sum();
            (
                times[0],
                total / times.len() as u32,
                times.len(),
                rows,
                total,
            )
        };
        let report = |name: &str, r: (Duration, Duration, usize, usize, Duration)| {
            println!(
                "{name:<34} first {:>9.2?}  avg/page {:>9.2?}  {:>4} pages {:>8} rows  total {:>9.2?}",
                r.0, r.1, r.2, r.3, r.4
            );
        };
        let list = |transpond_table: bool| {
            move |after: Option<String>| {
                generator
                    .get_object_list(&ListFilter {
                        plane: plane(),
                        images: None,
                        object_classes: None,
                        columns: columns.clone(),
                        with_coloc_details: false,
                        page: Pagination { limit: PAGE, after },
                        transpond_table,
                    })
                    .unwrap()
            }
        };
        let grouped = |transpond_table: bool| {
            move |after: Option<String>| {
                generator
                    .get_grouped_by_image(&GroupedByImageFilter {
                        plane: plane(),
                        images: None,
                        object_classes: None,
                        columns: vec![Column::Count, Column::AreaSizePx, Column::Circularity],
                        aggregation: vec![Aggregation::Avg, Aggregation::Max],
                        page: Pagination { limit: PAGE, after },
                        transpond_table,
                    })
                    .unwrap()
            }
        };

        for transposed in [false, true] {
            let label = if transposed { "transposed" } else { "normal" };
            report(
                &format!("object list {label} (20 pages)"),
                walk(&list(transposed), 20),
            );
        }
        for transposed in [false, true] {
            let label = if transposed { "transposed" } else { "normal" };
            report(
                &format!("grouped {label} (all pages)"),
                walk(&grouped(transposed), usize::MAX),
            );
        }

        // The alternative: normal grouped query, transposed in Rust.
        let start = Instant::now();
        let mut cursor = None;
        let mut per_image: HashMap<String, HashMap<String, Vec<String>>> = HashMap::new();
        loop {
            let page = grouped(false)(cursor.take());
            for row in &page.rows {
                let image = match &row[0].value {
                    CellValue::String(s) => s.clone(),
                    _ => String::new(),
                };
                let class = match &row[1].value {
                    CellValue::Class((name, _)) => name.clone(),
                    _ => String::new(),
                };
                let values = row[2..]
                    .iter()
                    .map(|cell| match cell.value {
                        CellValue::Float(v) => v.to_string(),
                        _ => String::new(),
                    })
                    .collect();
                per_image.entry(image).or_default().insert(class, values);
            }
            cursor = page.row_names.last().cloned();
            if page.rows.len() < PAGE as usize {
                break;
            }
        }
        println!(
            "{:<34} total {:>9.2?}  ({} image rows)",
            "grouped normal + pivot in Rust",
            start.elapsed(),
            per_image.len()
        );
    }

    /// Opening results the analysis in this process just wrote must reuse
    /// the analysis' connection. A separate `Connection::open` of the same
    /// file is refused on Windows ("file is being used by another process");
    /// on Linux it is allowed but becomes a second, stale database instance
    /// - which this test detects: it can't see rows written afterwards.
    #[test]
    fn results_share_the_connection_the_analysis_wrote_with() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("results.evadb");
        let writer = evanalyzer_core::open_results_database(&path).unwrap();
        writer
            .execute_batch("CREATE TABLE t (x INTEGER); INSERT INTO t VALUES (1);")
            .unwrap();

        let results = ResultsGenerator::open_database(path.clone()).unwrap();
        writer.execute("INSERT INTO t VALUES (2)", []).unwrap();

        let count: i64 = results
            .connection()
            .query_row("SELECT count(*) FROM t", [], |row| row.get(0))
            .unwrap();
        assert_eq!(count, 2, "results view must see the analysis' writes");
    }

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

    use super::super::test_support::{ObjectSpec, seed_db};

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
                transpond_table: false,
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

    /// Every cell in a row belongs to the same source object, so every one
    /// of them - not just an image-name/path column - must carry that
    /// object's own image's disabled flag, letting the GUI strike the whole
    /// row through regardless of which columns are actually shown.
    #[test]
    fn get_object_list_flags_every_cell_of_a_disabled_images_row() {
        let generator = open(&[
            ObjectSpec::new("img1.tif", "ClassA", 1, 100),
            ObjectSpec::new("img2.tif", "ClassB", 2, 200),
        ]);
        generator.enable_image("img2.tif", true).unwrap();

        let result = generator
            .get_object_list(&ListFilter {
                plane: plane(),
                images: None,
                object_classes: None,
                columns: vec![Column::ImageName, Column::ObjectClass, Column::AreaSizePx],
                with_coloc_details: false,
                page: no_page(),
                transpond_table: false,
            })
            .unwrap();

        let img1_row = result
            .rows
            .iter()
            .position(|row| matches!(&row[0].value, CellValue::String(s) if s == "img1.tif"))
            .expect("img1's row");
        let img2_row = result
            .rows
            .iter()
            .position(|row| matches!(&row[0].value, CellValue::String(s) if s == "img2.tif"))
            .expect("img2's row");

        assert!(
            result.rows[img1_row].iter().all(|cell| !cell.disabled),
            "img1 is enabled, so none of its cells should be struck through"
        );
        assert!(
            result.rows[img2_row].iter().all(|cell| cell.disabled),
            "img2 is disabled, so every cell of its row - not just the image name - must be flagged"
        );
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
                transpond_table: false,
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
                transpond_table: false,
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
                transpond_table: false,
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
                transpond_table: false,
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
                transpond_table: false,
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
                transpond_table: false,
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
        use super::super::test_support::CH0_INTENSITIES_JSON;
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
                transpond_table: false,
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
                transpond_table: false,
            })
            .unwrap();

        assert_eq!(result.rows.len(), 2, "one row per (image, class) group");
        assert_eq!(result.column_names[0], "image");
        assert_eq!(result.column_names[1], "class");
        assert_eq!(result.column_names[2], "Area [px] (AVG)");
    }

    // -- transposed (classes side by side) -------------------------------

    fn grouped_transposed(
        generator: &ResultsGenerator,
        object_classes: Option<Vec<ObjectClass>>,
        page: Pagination,
    ) -> DatabaseResult {
        generator
            .get_grouped_by_image(&GroupedByImageFilter {
                plane: plane(),
                images: None,
                object_classes,
                columns: vec![Column::Count, Column::AreaSizePx],
                aggregation: vec![Aggregation::Avg],
                page,
                transpond_table: true,
            })
            .unwrap()
    }

    fn cell_value(cell: &Cell) -> Option<f64> {
        match &cell.value {
            CellValue::Float(v) => Some(*v as f64),
            CellValue::Empty => None,
            _ => panic!("expected a float or empty cell"),
        }
    }

    #[test]
    fn transposed_grouped_puts_each_class_of_an_image_in_one_row() {
        let generator = open(&[
            ObjectSpec::new("img1.tif", "ClassA", 1, 10),
            ObjectSpec::new("img1.tif", "ClassA", 1, 20),
            ObjectSpec::new("img1.tif", "ClassB", 2, 100),
            ObjectSpec::new("img2.tif", "ClassA", 1, 50),
        ]);
        let result = grouped_transposed(&generator, None, no_page());

        assert_eq!(
            result.column_names,
            [
                "image",
                "Count (COUNT) (ClassA)",
                "Area [px] (AVG) (ClassA)",
                "Count (COUNT) (ClassB)",
                "Area [px] (AVG) (ClassB)",
            ]
        );
        assert_eq!(result.rows.len(), 2, "one row per image");
        let values = |row: &[Cell]| row[1..].iter().map(cell_value).collect::<Vec<_>>();
        assert_eq!(
            values(&result.rows[0]),
            [Some(2.0), Some(15.0), Some(1.0), Some(100.0)]
        );
        // img2 has no ClassB objects: no count of 0 pretending to be data
        // for the average, just empty.
        assert_eq!(values(&result.rows[1])[..2], [Some(1.0), Some(50.0)]);
        assert_eq!(values(&result.rows[1])[3], None);
    }

    #[test]
    fn transposed_grouped_matches_the_normal_view_value_for_value() {
        let generator = open(&[
            ObjectSpec::new("img1.tif", "ClassA", 1, 10),
            ObjectSpec::new("img1.tif", "ClassB", 2, 30),
            ObjectSpec::new("img1.tif", "ClassB", 2, 50),
            ObjectSpec::new("img2.tif", "ClassB", 2, 70),
        ]);
        let normal = generator
            .get_grouped_by_image(&GroupedByImageFilter {
                plane: plane(),
                images: None,
                object_classes: None,
                columns: vec![Column::Count, Column::AreaSizePx],
                aggregation: vec![Aggregation::Avg],
                page: no_page(),
                transpond_table: false,
            })
            .unwrap();
        let transposed = grouped_transposed(&generator, None, no_page());
        // Every (image, class) row of the normal view appears in the
        // transposed row of its image, in its class' block.
        for row in &normal.rows {
            let image = match &row[0].value {
                CellValue::String(s) => s.clone(),
                _ => unreachable!(),
            };
            let block = match &row[1].value {
                CellValue::Class((name, _)) if name == "ClassA" => 0,
                _ => 1,
            };
            let wide = transposed
                .rows
                .iter()
                .find(|r| matches!(&r[0].value, CellValue::String(s) if *s == image))
                .unwrap();
            for (i, cell) in row[2..].iter().enumerate() {
                assert_eq!(cell_value(cell), cell_value(&wide[1 + block * 2 + i]));
            }
        }
    }

    #[test]
    fn transposed_grouped_pages_by_image_and_respects_the_class_filter() {
        let generator = open(&[
            ObjectSpec::new("img1.tif", "ClassA", 1, 10),
            ObjectSpec::new("img2.tif", "ClassB", 2, 20),
            ObjectSpec::new("img3.tif", "ClassA", 1, 30),
        ]);
        let first = grouped_transposed(
            &generator,
            None,
            Pagination {
                limit: 2,
                after: None,
            },
        );
        assert_eq!(first.row_names, ["img1.tif", "img2.tif"]);
        let next = grouped_transposed(
            &generator,
            None,
            Pagination {
                limit: 2,
                after: first.row_names.last().cloned(),
            },
        );
        assert_eq!(next.row_names, ["img3.tif"]);

        let only_a = grouped_transposed(&generator, Some(vec![ObjectClass::Valid(1)]), no_page());
        assert_eq!(only_a.column_names.len(), 1 + 2, "image + one class block");
        // img2 has no ClassA objects, but was analysed: it keeps its row,
        // with a ClassA count of 0 and no average area.
        assert_eq!(only_a.row_names, ["img1.tif", "img2.tif", "img3.tif"]);
        assert_eq!(cell_value(&only_a.rows[1][1]), Some(0.0));
        assert!(matches!(only_a.rows[1][2].value, CellValue::Empty));
    }

    fn list_transposed(
        generator: &ResultsGenerator,
        object_classes: Option<Vec<ObjectClass>>,
        page: Pagination,
    ) -> DatabaseResult {
        generator
            .get_object_list(&ListFilter {
                plane: plane(),
                images: None,
                object_classes,
                columns: vec![Column::ImageName, Column::ObjectClass, Column::AreaSizePx],
                with_coloc_details: false,
                page,
                transpond_table: true,
            })
            .unwrap()
    }

    fn area(cell: &Cell) -> Option<u64> {
        match &cell.value {
            CellValue::Integer(v) => Some(*v as u64),
            CellValue::Float(v) => Some(*v as u64),
            CellValue::Empty => None,
            other => panic!(
                "unexpected area cell {:?}",
                matches!(other, CellValue::String(_))
            ),
        }
    }

    #[test]
    fn transposed_list_puts_the_classes_of_an_image_side_by_side() {
        // ObjectSpec ids are assigned in seeding order, so within an image
        // and class the n-th seeded object is the n-th row.
        let generator = open(&[
            ObjectSpec::new("img1.tif", "ClassA", 1, 11),
            ObjectSpec::new("img1.tif", "ClassA", 1, 12),
            ObjectSpec::new("img1.tif", "ClassA", 1, 13),
            ObjectSpec::new("img1.tif", "ClassB", 2, 21),
            ObjectSpec::new("img2.tif", "ClassB", 2, 22),
        ]);
        let result = list_transposed(&generator, None, no_page());

        assert_eq!(
            result.column_names,
            ["image", "Area [px] (ClassA)", "Area [px] (ClassB)"]
        );
        let rows: Vec<(String, Option<u64>, Option<u64>)> = result
            .rows
            .iter()
            .map(|row| {
                let image = match &row[0].value {
                    CellValue::String(s) => s.clone(),
                    _ => unreachable!(),
                };
                (image, area(&row[1]), area(&row[2]))
            })
            .collect();
        assert_eq!(
            rows,
            [
                ("img1.tif".into(), Some(11), Some(21)),
                ("img1.tif".into(), Some(12), None),
                ("img1.tif".into(), Some(13), None),
                ("img2.tif".into(), None, Some(22)),
            ]
        );
        assert_eq!(result.row_locations.len(), 4, "every row can be clicked");
        assert_eq!(result.source_object_count, 5);
    }

    #[test]
    fn transposed_list_pages_without_gaps_or_repeats_across_images() {
        let mut specs = Vec::new();
        for (image, n) in [("img1.tif", 3), ("img2.tif", 2), ("img3.tif", 4)] {
            for i in 0..n {
                specs.push(ObjectSpec::new(image, "ClassA", 1, 100 + i));
                specs.push(ObjectSpec::new(image, "ClassB", 2, 200 + i));
            }
        }
        let generator = open(&specs);
        let all = list_transposed(&generator, None, no_page());
        assert_eq!(all.rows.len(), 9);

        let mut paged = Vec::new();
        let mut after = None;
        loop {
            let page = list_transposed(&generator, None, Pagination { limit: 2, after });
            if page.rows.is_empty() {
                break;
            }
            after = page.row_names.last().cloned();
            paged.extend(page.row_names);
        }
        assert_eq!(paged, all.row_names);
    }

    #[test]
    fn transposed_list_with_a_class_filter_shows_only_that_block() {
        let generator = open(&[
            ObjectSpec::new("img1.tif", "ClassA", 1, 11),
            ObjectSpec::new("img1.tif", "ClassB", 2, 21),
            ObjectSpec::new("img1.tif", "ClassB", 2, 22),
        ]);
        let result = list_transposed(&generator, Some(vec![ObjectClass::Valid(2)]), no_page());
        assert_eq!(result.column_names, ["image", "Area [px] (ClassB)"]);
        assert_eq!(result.rows.len(), 2);
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
                transpond_table: false,
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
                transpond_table: false,
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

    /// An image that was analyzed but produced zero objects (e.g. an empty
    /// well) must still show up as its own group rather than being silently
    /// absent from the matrix, since "absent" and "present but empty" mean
    /// different things in a plate view. The row survives *and* its value
    /// cell is `CellValue::Empty` (not a misleading `0.0`), same convention
    /// as `Heatmap` and as the export's own "None"-for-empty-cell text.
    #[test]
    fn get_group_by_plate_includes_a_well_with_no_objects_as_an_empty_group() {
        let generator = open(&[ObjectSpec::new("A1_01.tif", "ClassA", 1, 10)]);
        generator
            .database
            .execute(
                "INSERT INTO images (image_name, image_rel_path, width, height, c_stacks, z_stacks, t_stacks) \
                 VALUES ('B2_01.tif', 'B2_01.tif', 100, 100, 1, 1, 1)",
                [],
            )
            .unwrap();

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

        assert_eq!(result.row_names.len(), 2, "B2 must still appear");
        let b2_idx = result
            .row_names
            .iter()
            .position(|n| n == "B2")
            .expect("B2 group");
        assert!(
            matches!(result.rows[b2_idx][1].value, CellValue::Empty),
            "a well with no objects has no average to show"
        );

        // The Heatmap view must place a real, selectable (empty) tile at
        // B2's grid position - not skip it as if no well were there at all.
        let heatmap = generator
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
                &View::Heatmap,
            )
            .unwrap();
        // A1 -> (row 0, col 0), B2 -> (row 1, col 1).
        assert!(
            matches!(heatmap.rows[1][1].value, CellValue::Empty),
            "B2 has no objects, so no number to show"
        );
        assert!(
            heatmap.rows[1][1].search_key.is_some(),
            "B2 must still be a real, clickable well"
        );
        assert!(
            heatmap.rows[1][2].search_key.is_none(),
            "position (1, 2) has no well at all and must stay non-clickable"
        );
    }

    /// A disabled image's objects must not contribute to its well's
    /// aggregate, but the well itself is never dropped - even one made up
    /// only of disabled images still appears (as an empty cell, same as any
    /// other well with no contributing objects). The well itself is never
    /// flagged `Cell::disabled` though - only individual images are - so a
    /// well made up only of disabled images renders the same as any other
    /// empty well.
    #[test]
    fn get_group_by_plate_ignores_objects_from_a_disabled_image() {
        let generator = open(&[
            ObjectSpec::new("A1_01.tif", "ClassA", 1, 10),
            ObjectSpec::new("B2_01.tif", "ClassA", 1, 100),
        ]);
        generator.enable_image("B2_01.tif", true).unwrap();

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

        assert_eq!(
            result.row_names,
            vec!["A1".to_string(), "B2".to_string()],
            "B2 must still appear even though every one of its images is disabled"
        );
        let a1_idx = result.row_names.iter().position(|n| n == "A1").unwrap();
        let b2_idx = result.row_names.iter().position(|n| n == "B2").unwrap();
        assert!(
            !result.rows[a1_idx][1].disabled,
            "A1 has no disabled images"
        );
        assert!(
            !result.rows[b2_idx][1].disabled,
            "the well itself is never flagged disabled, only individual images are"
        );
        assert!(
            matches!(result.rows[b2_idx][1].value, CellValue::Empty),
            "B2's disabled image must not contribute to the average, leaving no value to show"
        );
    }

    /// The mixed case: a well with both an enabled and a disabled image
    /// must still average only the enabled one's objects, but the well
    /// itself must not be flagged `disabled` - only individual images are,
    /// so the well keeps its normal heatmap color.
    #[test]
    fn get_group_by_plate_flags_a_well_with_a_mix_of_enabled_and_disabled_images() {
        let generator = open(&[
            ObjectSpec::new("A1_01.tif", "ClassA", 1, 10),
            ObjectSpec::new("A1_02.tif", "ClassA", 1, 1000),
        ]);
        generator.enable_image("A1_02.tif", true).unwrap();

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

        assert_eq!(result.row_names, vec!["A1".to_string()]);
        assert!(
            !result.rows[0][1].disabled,
            "the well itself must not be flagged disabled just because one of its images is"
        );
        assert!(
            result.rows[0][1].any_disabled,
            "the well must still be flagged any_disabled so the GUI can badge it"
        );
        assert!(
            matches!(result.rows[0][1].value, CellValue::Float(v) if v == 10.0),
            "only the enabled image's objects (10) must be averaged, not the disabled one's 1000"
        );

        let heatmap = generator
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
                &View::Heatmap,
            )
            .unwrap();
        let a1_cell = &heatmap.rows[0][0];
        assert!(
            !a1_cell.disabled,
            "the heatmap's well cell must not be flagged disabled either"
        );
        assert!(
            a1_cell.any_disabled,
            "the heatmap's well cell must carry any_disabled too, for the same corner badge"
        );
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

    #[test]
    fn enable_image_toggles_the_disabled_flag() {
        let generator = open(&[ObjectSpec::new("a.tif", "ClassA", 1, 10)]);
        assert!(!generator.get_images().unwrap()[0].disabled);

        generator.enable_image("a.tif", true).unwrap();
        assert!(generator.get_images().unwrap()[0].disabled);

        generator.enable_image("a.tif", false).unwrap();
        assert!(!generator.get_images().unwrap()[0].disabled);
    }

    #[test]
    fn enable_image_on_an_unknown_path_is_a_no_op() {
        let generator = open(&[ObjectSpec::new("a.tif", "ClassA", 1, 10)]);
        generator.enable_image("does-not-exist.tif", true).unwrap();
        assert!(!generator.get_images().unwrap()[0].disabled);
    }

    /// `try_clone` is the fix for a real Windows bug: exporting used to
    /// reopen the `.evadb` file by path on a background thread while the
    /// original `ResultsGenerator` connection was still open, which Windows
    /// (unlike Linux/macOS) refuses - "used by another process", reporting
    /// the app's own PID. `try_clone` gets a second, independent connection
    /// to the *same already-open* database instead of reopening the file,
    /// so this pins down that both connections stay live and see the same
    /// data at the same time - the exact scenario a background export needs.
    #[test]
    fn try_clone_gives_an_independent_connection_to_the_same_live_database() {
        let generator = open(&[ObjectSpec::new("a.tif", "ClassA", 1, 10)]);
        let cloned = generator.try_clone().expect("clone connection");

        // Both connections are usable at the same time...
        assert_eq!(generator.get_images().unwrap().len(), 1);
        assert_eq!(cloned.get_images().unwrap().len(), 1);

        // ...and see the same underlying data, not two separate files: a
        // write through one is visible through the other.
        cloned.enable_image("a.tif", true).unwrap();
        assert!(
            generator.get_images().unwrap()[0].disabled,
            "the clone must share the original's already-open database, not a second file"
        );
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
                transpond_table: false,
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

    /// Mirrors `get_group_by_plate_includes_a_well_with_no_objects_as_an_empty_group`
    /// one level down: a field (image) with zero objects must still get its
    /// own tile in the well view instead of silently vanishing.
    #[test]
    fn get_group_by_well_includes_a_field_with_no_objects_as_an_empty_tile() {
        let generator = open(&[ObjectSpec::new("A1_01.tif", "ClassA", 1, 10)]);
        generator
            .database
            .execute(
                "INSERT INTO images (image_name, image_rel_path, width, height, c_stacks, z_stacks, t_stacks) \
                 VALUES ('A1_02.tif', 'A1_02.tif', 100, 100, 1, 1, 1)",
                [],
            )
            .unwrap();

        let list = generator
            .get_group_by_well(&well_filter("A1", Column::AreaSizePx), &View::List)
            .unwrap();

        assert_eq!(list.row_names, vec!["01".to_string(), "02".to_string()]);
        assert_eq!(float_cell(&list.rows[0][1]), 10.0);
        assert!(
            matches!(list.rows[1][1].value, CellValue::Empty),
            "field with no objects has no average to show"
        );

        // The Heatmap view must place a real, selectable (empty) tile at
        // field 02's grid position - not skip it as if no field were
        // there at all (that's reserved for a position no field occupies).
        let heatmap = generator
            .get_group_by_well(&well_filter("A1", Column::AreaSizePx), &View::Heatmap)
            .unwrap();
        // Default 4x4 well, no `well_order`: idx 1 -> position 0 (row 0,
        // col 0), idx 2 -> position 1 (row 0, col 1).
        assert!(
            matches!(heatmap.rows[0][1].value, CellValue::Empty),
            "field 02 has no objects, so no number to show"
        );
        assert!(
            heatmap.rows[0][1].search_key.is_some(),
            "field 02 must still be a real, clickable tile"
        );
        assert!(
            heatmap.rows[0][2].search_key.is_none(),
            "position (0, 2) has no field at all and must stay non-clickable"
        );
    }

    /// A disabled image's *own* field must still show its real value (a
    /// single image's own aggregate is never affected by its own disabled
    /// flag - only a value that combines several images, like the plate
    /// view's well average, excludes it), just flagged via `Cell::disabled`.
    /// Mirrors `get_group_by_plate_flags_a_well_with_a_mix_of_enabled_and_disabled_images`
    /// one level down.
    #[test]
    fn get_wells_for_plate_flags_but_keeps_a_disabled_images_field() {
        let generator = open(&[
            ObjectSpec::new("A1_01.tif", "ClassA", 1, 10),
            ObjectSpec::new("A1_02.tif", "ClassA", 1, 20),
        ]);
        generator.enable_image("A1_02.tif", true).unwrap();

        let batched = generator
            .get_wells_for_plate(&wells_batch_filter(Column::AreaSizePx), &View::List)
            .unwrap();
        assert_eq!(
            batched["A1"].row_names,
            vec!["01".to_string(), "02".to_string()],
            "the disabled field must still appear"
        );
        assert!(!batched["A1"].rows[0][1].disabled, "field 01 is enabled");
        assert!(batched["A1"].rows[1][1].disabled, "field 02 is disabled");
        assert!(
            matches!(batched["A1"].rows[1][1].value, CellValue::Float(v) if v == 20.0),
            "a disabled image's own field still shows its own real value"
        );

        let single = generator
            .get_group_by_well(&well_filter("A1", Column::AreaSizePx), &View::List)
            .unwrap();
        assert_eq!(single.row_names, vec!["01".to_string(), "02".to_string()]);
        assert!(single.rows[1][1].disabled);
    }

    /// A disabled field's own value is still shown (see the test above),
    /// but must not skew the Auto color range every *other* field's tile is
    /// colored against - disabling an outlier should change how the
    /// remaining fields compare to each other, not leave them exactly where
    /// they were as if nothing happened.
    #[test]
    fn get_group_by_well_heatmap_excludes_a_disabled_fields_value_from_the_auto_color_range() {
        let generator = open(&[
            ObjectSpec::new("A1_01.tif", "ClassA", 1, 10),
            ObjectSpec::new("A1_02.tif", "ClassA", 1, 20),
            ObjectSpec::new("A1_03.tif", "ClassA", 1, 1000),
        ]);
        generator.enable_image("A1_03.tif", true).unwrap();

        let heatmap = generator
            .get_group_by_well(&well_filter("A1", Column::AreaSizePx), &View::Heatmap)
            .unwrap();

        // Default 4x4 well grid, no `well_order`: field "01" -> (0, 0),
        // "02" -> (0, 1), "03" -> (0, 2).
        let schema = ColorSchema::default();
        assert_eq!(
            heatmap.rows[0][0].bg_color,
            value_to_color(10.0, 10.0, 20.0, &schema),
            "10 must be colored as the range's own min, ignoring the disabled 1000"
        );
        assert_eq!(
            heatmap.rows[0][1].bg_color,
            value_to_color(20.0, 10.0, 20.0, &schema),
            "20 must be colored as the range's own max, ignoring the disabled 1000"
        );
    }

    #[test]
    fn group_by_plate_multi_agg_matches_group_by_plate_per_aggregation() {
        let generator = open(&[
            ObjectSpec::new("A1_01.tif", "ClassA", 1, 10),
            ObjectSpec::new("A1_02.tif", "ClassA", 1, 20),
            ObjectSpec::new("A1_03.tif", "ClassA", 1, 30),
        ]);
        generator.enable_image("A1_03.tif", true).unwrap();
        let aggregations = vec![Aggregation::Avg, Aggregation::Min, Aggregation::Max];
        let multi_filter = PlateFilterMulti {
            plane: plane(),
            grouping_regex: String::new(),
            aggregation: aggregations.clone(),
            object_class: vec![ObjectClass::Unset],
            column: vec![Column::AreaSizePx],
            color_schema: ColorSchema::default(),
            color_scale: ColorScale::default(),
            matrix_dimension: None,
        };
        let multi = generator
            .get_group_by_plate_multi(&multi_filter, &View::List)
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
            assert_eq!(
                batched_result.rows[0][1].disabled, single.rows[0][1].disabled,
                "aggregation {aggregation:?}'s disabled flag disagrees between multi_agg batch and single call",
            );
            assert!(
                !batched_result.rows[0][1].disabled,
                "the well itself must not be flagged disabled just because A1_03 is"
            );
        }
        // Sanity on the actual numbers (A1_03's 30 excluded from every
        // aggregate since it's disabled), not just internal agreement.
        assert_eq!(float_cell(&multi[0].rows[0][1]), 15.0); // avg(10, 20)
        assert_eq!(float_cell(&multi[1].rows[0][1]), 10.0); // min(10, 20)
        assert_eq!(float_cell(&multi[2].rows[0][1]), 20.0); // max(10, 20)
    }

    #[test]
    fn wells_for_plate_multi_agg_matches_wells_for_plate_per_aggregation() {
        let generator = open(&[
            ObjectSpec::new("A1_01.tif", "ClassA", 1, 10),
            ObjectSpec::new("A1_02.tif", "ClassA", 1, 20),
        ]);
        generator.enable_image("A1_02.tif", true).unwrap();
        let aggregations = vec![Aggregation::Avg, Aggregation::Sum];
        let multi_filter = WellsBatchFilterMulti {
            plane: plane(),
            grouping_regex: String::new(),
            aggregation: aggregations.clone(),
            object_class: vec![ObjectClass::Unset],
            column: vec![Column::AreaSizePx],
            color_schema: ColorSchema::default(),
            color_scale: ColorScale::default(),
            well_size: None,
            well_order: None,
        };
        let multi = generator
            .get_wells_for_plate_multi(&multi_filter, &View::List)
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
                    assert_eq!(
                        b_row[1].disabled, s_row[1].disabled,
                        "well {well_id}, aggregation {aggregation:?}",
                    );
                }
            }
        }
        // A1_02 (field "02") is disabled - its own field still shows its
        // own value (20), and only field "01" (value 10) is unflagged.
        assert!(!multi[0]["A1"].rows[0][1].disabled);
        assert!(multi[0]["A1"].rows[1][1].disabled);
        assert_eq!(float_cell(&multi[0]["A1"].rows[1][1]), 20.0);
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
                transpond_table: false,
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
                transpond_table: false,
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
                transpond_table: false,
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
                transpond_table: false,
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
                transpond_table: false,
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
            transpond_table: false,
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
                transpond_table: false,
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
                transpond_table: false,
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

    // -- groups without objects: 0 vs. empty ---------------------------------
    //
    // A Count or Sum of a group without objects is 0 when its images were
    // analysed on the selected plane; every other statistic stays empty, and
    // so does everything of a failed, disabled-only or not-analysed group.

    /// How an image row without objects of its own is seeded.
    struct ImageRow {
        name: &'static str,
        successful: bool,
        disabled: bool,
        z_stacks: u32,
    }

    impl ImageRow {
        fn analysed(name: &'static str) -> Self {
            Self {
                name,
                successful: true,
                disabled: false,
                z_stacks: 1,
            }
        }
    }

    fn add_image(generator: &ResultsGenerator, image: ImageRow) {
        generator
            .database
            .execute(
                "INSERT INTO images (image_name, image_rel_path, successful, disabled, width, \
                 height, c_stacks, z_stacks, t_stacks) VALUES (?, ?, ?, ?, 512, 512, 1, ?, 1)",
                duckdb::params![
                    image.name,
                    image.name,
                    image.successful,
                    image.disabled,
                    image.z_stacks
                ],
            )
            .unwrap();
    }

    fn plate_value(generator: &ResultsGenerator, filter: &PlateFilter, well: &str) -> Option<f64> {
        let result = generator.get_group_by_plate(filter, &View::List).unwrap();
        let idx = result
            .row_names
            .iter()
            .position(|n| n == well)
            .unwrap_or_else(|| panic!("well {well} missing: {:?}", result.row_names));
        as_float(&result.rows[idx][1].value)
    }

    fn as_float(value: &CellValue) -> Option<f64> {
        match value {
            CellValue::Float(v) => Some(*v as f64),
            CellValue::Empty => None,
            _ => panic!("expected a float or empty cell"),
        }
    }

    #[test]
    fn plate_count_and_sum_are_zero_for_a_well_whose_images_found_nothing() {
        let generator = open(&[ObjectSpec::new("A1_01.tif", "ClassA", 1, 10)]);
        add_image(&generator, ImageRow::analysed("B2_01.tif"));

        let count = plate_value(&generator, &plate_filter(Column::Count), "B2");
        let sum = plate_value(
            &generator,
            &PlateFilter {
                aggregation: Aggregation::Sum,
                ..plate_filter(Column::AreaSizePx)
            },
            "B2",
        );
        let avg = plate_value(&generator, &plate_filter(Column::AreaSizePx), "B2");

        assert_eq!(count, Some(0.0));
        assert_eq!(sum, Some(0.0));
        assert_eq!(avg, None, "no average without objects");
        assert_eq!(
            plate_value(&generator, &plate_filter(Column::Count), "A1"),
            Some(1.0)
        );
    }

    #[test]
    fn plate_count_stays_empty_for_a_well_of_only_failed_or_only_disabled_images() {
        let generator = open(&[ObjectSpec::new("A1_01.tif", "ClassA", 1, 10)]);
        add_image(
            &generator,
            ImageRow {
                successful: false,
                ..ImageRow::analysed("B2_01.tif")
            },
        );
        add_image(
            &generator,
            ImageRow {
                disabled: true,
                ..ImageRow::analysed("C3_01.tif")
            },
        );

        let filter = plate_filter(Column::Count);
        assert_eq!(plate_value(&generator, &filter, "B2"), None);
        assert_eq!(plate_value(&generator, &filter, "C3"), None);
    }

    #[test]
    fn plate_count_is_zero_when_at_least_one_enabled_image_of_the_well_was_analysed() {
        let generator = open(&[ObjectSpec::new("A1_01.tif", "ClassA", 1, 10)]);
        add_image(
            &generator,
            ImageRow {
                successful: false,
                ..ImageRow::analysed("B2_01.tif")
            },
        );
        add_image(&generator, ImageRow::analysed("B2_02.tif"));

        let value = plate_value(&generator, &plate_filter(Column::Count), "B2");

        assert_eq!(value, Some(0.0));
    }

    fn count_on_plane(z_stack: u32) -> PlateFilter {
        PlateFilter {
            plane: PlaneFilter {
                z_stack,
                t_stack: 0,
            },
            ..plate_filter(Column::Count)
        }
    }

    #[test]
    fn plate_count_stays_empty_on_a_plane_the_run_did_not_analyse() {
        // A Z-projection run: 3 planes in the file, objects only on plane 0.
        let generator = open(&[ObjectSpec::new("A1_01.tif", "ClassA", 1, 10)]);
        add_image(
            &generator,
            ImageRow {
                z_stacks: 3,
                ..ImageRow::analysed("B2_01.tif")
            },
        );

        assert_eq!(plate_value(&generator, &count_on_plane(0), "B2"), Some(0.0));
        assert_eq!(plate_value(&generator, &count_on_plane(1), "B2"), None);
    }

    #[test]
    fn planes_between_the_lowest_and_highest_object_plane_count_as_analysed() {
        let generator = open(&[
            ObjectSpec::new("A1_01.tif", "ClassA", 1, 10).at_plane(0, 0),
            ObjectSpec::new("A1_01.tif", "ClassA", 1, 10).at_plane(2, 0),
        ]);
        add_image(
            &generator,
            ImageRow {
                z_stacks: 3,
                ..ImageRow::analysed("B2_01.tif")
            },
        );
        // Has only one plane itself - plane 1 doesn't exist for it.
        add_image(&generator, ImageRow::analysed("C3_01.tif"));

        // Plane 1: nothing found anywhere, but inside the analysed range.
        assert_eq!(plate_value(&generator, &count_on_plane(1), "A1"), Some(0.0));
        assert_eq!(plate_value(&generator, &count_on_plane(1), "B2"), Some(0.0));
        assert_eq!(plate_value(&generator, &count_on_plane(1), "C3"), None);
    }

    #[test]
    fn a_run_without_any_object_has_no_values_at_all() {
        let generator = open(&[]);
        add_image(&generator, ImageRow::analysed("B2_01.tif"));

        assert_eq!(plate_value(&generator, &count_on_plane(0), "B2"), None);
    }

    #[test]
    fn plate_sum_of_objects_without_a_value_stays_empty_rather_than_zero() {
        // Objects exist, but none has an nm area (no pixel size): the sum
        // is unknown, not 0.
        let generator = open(&[ObjectSpec::new("A1_01.tif", "ClassA", 1, 10)]);
        generator
            .database
            .execute("UPDATE objects SET area_nm2 = NULL", [])
            .unwrap();
        let filter = PlateFilter {
            aggregation: Aggregation::Sum,
            ..plate_filter(Column::AreaSizeNm)
        };

        assert_eq!(plate_value(&generator, &filter, "A1"), None);
    }

    #[test]
    fn plate_multi_fills_zero_only_for_count_and_sum() {
        let generator = open(&[ObjectSpec::new("A1_01.tif", "ClassA", 1, 10)]);
        add_image(&generator, ImageRow::analysed("B2_01.tif"));
        let results = generator
            .get_group_by_plate_multi(
                &PlateFilterMulti {
                    plane: plane(),
                    grouping_regex: String::new(),
                    aggregation: vec![Aggregation::Avg, Aggregation::Sum],
                    object_class: vec![ObjectClass::Unset],
                    column: vec![Column::Count, Column::AreaSizePx],
                    color_schema: ColorSchema::default(),
                    color_scale: ColorScale::default(),
                    matrix_dimension: None,
                },
                &View::List,
            )
            .unwrap();
        let b2 = |result: &DatabaseResult| {
            let idx = result.row_names.iter().position(|n| n == "B2").unwrap();
            as_float(&result.rows[idx][1].value)
        };

        // (Count, Avg), (Count, Sum), (area, Avg), (area, Sum)
        let values: Vec<Option<f64>> = results.iter().map(b2).collect();
        assert_eq!(values, [Some(0.0), Some(0.0), None, Some(0.0)]);
    }

    #[test]
    fn well_fields_without_objects_count_zero_unless_their_analysis_failed() {
        let generator = open(&[ObjectSpec::new("A1_01.tif", "ClassA", 1, 10)]);
        add_image(&generator, ImageRow::analysed("A1_02.tif"));
        add_image(
            &generator,
            ImageRow {
                successful: false,
                ..ImageRow::analysed("A1_03.tif")
            },
        );
        // A disabled field's own value is still its own real result.
        add_image(
            &generator,
            ImageRow {
                disabled: true,
                ..ImageRow::analysed("A1_04.tif")
            },
        );
        let filter = well_filter("A1", Column::Count);

        let single = generator.get_group_by_well(&filter, &View::List).unwrap();
        let values: Vec<Option<f64>> = single
            .rows
            .iter()
            .map(|row| as_float(&row[1].value))
            .collect();
        assert_eq!(values, [Some(1.0), Some(0.0), None, Some(0.0)]);

        let batched = generator
            .get_wells_for_plate(&wells_batch_filter(Column::Count), &View::List)
            .unwrap();
        let batched_values: Vec<Option<f64>> = batched["A1"]
            .rows
            .iter()
            .map(|row| as_float(&row[1].value))
            .collect();
        assert_eq!(batched_values, values);

        let multi = generator
            .get_wells_for_plate_multi(
                &WellsBatchFilterMulti {
                    plane: plane(),
                    grouping_regex: String::new(),
                    aggregation: vec![Aggregation::Avg],
                    object_class: vec![ObjectClass::Unset],
                    column: vec![Column::Count, Column::AreaSizePx],
                    color_schema: ColorSchema::default(),
                    color_scale: ColorScale::default(),
                    well_size: None,
                    well_order: None,
                },
                &View::List,
            )
            .unwrap();
        let multi_values = |combo: usize| -> Vec<Option<f64>> {
            multi[combo]["A1"]
                .rows
                .iter()
                .map(|row| as_float(&row[1].value))
                .collect()
        };
        assert_eq!(multi_values(0), values);
        assert_eq!(multi_values(1), [Some(10.0), None, None, None]);
    }

    fn grouped(generator: &ResultsGenerator, aggregation: Aggregation) -> DatabaseResult {
        generator
            .get_grouped_by_image(&GroupedByImageFilter {
                plane: plane(),
                images: None,
                object_classes: None,
                columns: vec![Column::Count, Column::AreaSizePx],
                aggregation: vec![aggregation],
                page: no_page(),
                transpond_table: false,
            })
            .unwrap()
    }

    /// `(image, class label, values...)` per row.
    fn grouped_rows(result: &DatabaseResult) -> Vec<(String, String, Vec<Option<f64>>)> {
        result
            .rows
            .iter()
            .map(|row| {
                let text = |cell: &Cell| match &cell.value {
                    CellValue::String(s) => s.clone(),
                    CellValue::Class((label, _)) => label.clone(),
                    _ => panic!("expected text"),
                };
                (
                    text(&row[0]),
                    text(&row[1]),
                    row[2..].iter().map(|c| as_float(&c.value)).collect(),
                )
            })
            .collect()
    }

    #[test]
    fn grouped_by_image_lists_every_analysed_image_and_class_with_count_zero() {
        let generator = open(&[
            ObjectSpec::new("img1.tif", "ClassA", 1, 10),
            ObjectSpec::new("img2.tif", "ClassB", 2, 20),
        ]);
        add_image(&generator, ImageRow::analysed("img3.tif"));
        add_image(
            &generator,
            ImageRow {
                successful: false,
                ..ImageRow::analysed("img4.tif")
            },
        );

        let rows = grouped_rows(&grouped(&generator, Aggregation::Avg));

        let row = |image: &str, class: &str, count, area| {
            (image.to_string(), class.to_string(), vec![count, area])
        };
        assert_eq!(
            rows,
            [
                row("img1.tif", "ClassA", Some(1.0), Some(10.0)),
                row("img1.tif", "ClassB", Some(0.0), None),
                row("img2.tif", "ClassA", Some(0.0), None),
                row("img2.tif", "ClassB", Some(1.0), Some(20.0)),
                row("img3.tif", "ClassA", Some(0.0), None),
                row("img3.tif", "ClassB", Some(0.0), None),
                // img4.tif failed and found nothing: no row at all.
            ]
        );
    }

    #[test]
    fn grouped_by_image_shows_a_missing_spread_as_empty_not_zero() {
        // A sample standard deviation needs two values; one object has none.
        let generator = open(&[ObjectSpec::new("img1.tif", "ClassA", 1, 10)]);

        let rows = grouped_rows(&grouped(&generator, Aggregation::Stddev));

        assert_eq!(rows[0].2, [Some(1.0), None]);
    }

    #[test]
    fn grouped_by_image_pages_through_images_without_objects_without_gaps() {
        let generator = open(&[ObjectSpec::new("img1.tif", "ClassA", 1, 10)]);
        add_image(&generator, ImageRow::analysed("img2.tif"));
        add_image(&generator, ImageRow::analysed("img3.tif"));
        let page = |after: Option<String>| {
            generator
                .get_grouped_by_image(&GroupedByImageFilter {
                    plane: plane(),
                    images: None,
                    object_classes: None,
                    columns: vec![Column::Count],
                    aggregation: vec![Aggregation::Avg],
                    page: Pagination { limit: 2, after },
                    transpond_table: false,
                })
                .unwrap()
        };

        let mut seen = Vec::new();
        let mut after = None;
        loop {
            let result = page(after);
            seen.extend(grouped_rows(&result).into_iter().map(|(image, ..)| image));
            match result.row_names.last() {
                Some(last) if result.rows.len() == 2 => after = Some(last.clone()),
                _ => break,
            }
        }

        assert_eq!(seen, ["img1.tif", "img2.tif", "img3.tif"]);
    }

    #[test]
    fn grouped_by_image_has_no_rows_on_a_plane_the_run_did_not_analyse() {
        let generator = open(&[ObjectSpec::new("img1.tif", "ClassA", 1, 10)]);
        add_image(
            &generator,
            ImageRow {
                z_stacks: 3,
                ..ImageRow::analysed("img2.tif")
            },
        );

        let result = generator
            .get_grouped_by_image(&GroupedByImageFilter {
                plane: PlaneFilter {
                    z_stack: 1,
                    t_stack: 0,
                },
                images: None,
                object_classes: None,
                columns: vec![Column::Count],
                aggregation: vec![Aggregation::Avg],
                page: no_page(),
                transpond_table: false,
            })
            .unwrap();

        assert!(result.rows.is_empty(), "{:?}", result.row_names);
    }

    #[test]
    fn grouped_by_image_leaves_out_background_unless_it_is_selected() {
        let generator = open(&[ObjectSpec::new("img1.tif", "ClassA", 1, 10)]);
        generator
            .database
            .execute(
                "INSERT INTO classes (class_id, name, color) VALUES (0, 'Background', 0)",
                [],
            )
            .unwrap();
        generator.classes_cache.replace(None);

        let all = grouped_rows(&grouped(&generator, Aggregation::Avg));
        let selected = grouped_rows(
            &generator
                .get_grouped_by_image(&GroupedByImageFilter {
                    plane: plane(),
                    images: None,
                    object_classes: Some(vec![ObjectClass::BACKGROUND]),
                    columns: vec![Column::Count],
                    aggregation: vec![Aggregation::Avg],
                    page: no_page(),
                    transpond_table: false,
                })
                .unwrap(),
        );

        let classes: Vec<&str> = all.iter().map(|(_, class, _)| class.as_str()).collect();
        assert_eq!(classes, ["ClassA"]);
        assert_eq!(selected.len(), 1);
        assert_eq!(selected[0].1, "Background");
        assert_eq!(selected[0].2, [Some(0.0)]);
    }

    #[test]
    fn transposed_grouped_shows_an_analysed_image_without_objects_with_count_zero() {
        let generator = open(&[ObjectSpec::new("img1.tif", "ClassA", 1, 10)]);
        add_image(&generator, ImageRow::analysed("img2.tif"));
        add_image(
            &generator,
            ImageRow {
                successful: false,
                ..ImageRow::analysed("img3.tif")
            },
        );

        let result = grouped_transposed(&generator, None, no_page());

        assert_eq!(result.row_names, ["img1.tif", "img2.tif"]);
        assert_eq!(cell_value(&result.rows[1][1]), Some(0.0), "Count");
        assert!(
            matches!(result.rows[1][2].value, CellValue::Empty),
            "Avg area"
        );
    }

    fn heatmap(
        generator: &ResultsGenerator,
        image: &str,
        column: Column,
        aggregation: Aggregation,
    ) -> DatabaseResult {
        generator
            .get_image_heatmap(
                &ImageHeatmapFilter {
                    plane: plane(),
                    image_rel_path: image.to_string(),
                    aggregation,
                    object_class: ObjectClass::Unset,
                    column,
                    color_schema: ColorSchema::default(),
                    color_scale: ColorScale::default(),
                    square_size: Some(256),
                },
                &View::Heatmap,
            )
            .unwrap()
    }

    #[test]
    fn image_heatmap_counts_zero_in_squares_without_objects() {
        let generator = open(&[ObjectSpec::new("img1.tif", "ClassA", 1, 10)
            .with_image_size(512, 256)
            .at_centroid(10.0, 10.0)]);

        let count = heatmap(&generator, "img1.tif", Column::Count, Aggregation::Avg);
        let avg = heatmap(&generator, "img1.tif", Column::AreaSizePx, Aggregation::Avg);

        let values = |result: &DatabaseResult| -> Vec<Option<f64>> {
            result.rows[0].iter().map(|c| as_float(&c.value)).collect()
        };
        assert_eq!(values(&count), [Some(1.0), Some(0.0)]);
        assert_eq!(values(&avg), [Some(10.0), None]);
    }

    #[test]
    fn image_heatmap_of_a_failed_image_stays_empty() {
        let generator = open(&[ObjectSpec::new("img1.tif", "ClassA", 1, 10)]);
        add_image(
            &generator,
            ImageRow {
                successful: false,
                ..ImageRow::analysed("img2.tif")
            },
        );

        let count = heatmap(&generator, "img2.tif", Column::Count, Aggregation::Avg);

        assert!(
            count
                .rows
                .iter()
                .flatten()
                .all(|c| matches!(c.value, CellValue::Empty))
        );
    }

    // -- failed images ------------------------------------------------------
    //
    // An image whose analysis stopped with an error only has the objects
    // found before the error: excluded from a well's value like a disabled
    // image, and marked wherever it's shown on its own.

    fn mark_failed(generator: &ResultsGenerator, image: &str) {
        generator
            .database
            .execute(
                "UPDATE images SET successful = false WHERE image_rel_path = ?",
                [image],
            )
            .unwrap();
    }

    #[test]
    fn plate_leaves_out_a_failed_images_objects_and_marks_the_well() {
        let generator = open(&[
            ObjectSpec::new("A1_01.tif", "ClassA", 1, 10),
            ObjectSpec::new("A1_02.tif", "ClassA", 1, 100),
            ObjectSpec::new("B2_01.tif", "ClassA", 1, 20),
        ]);
        mark_failed(&generator, "A1_02.tif");

        let result = generator
            .get_group_by_plate(&plate_filter(Column::AreaSizePx), &View::List)
            .unwrap();
        let a1 = result.row_names.iter().position(|n| n == "A1").unwrap();
        let b2 = result.row_names.iter().position(|n| n == "B2").unwrap();

        assert_eq!(as_float(&result.rows[a1][1].value), Some(10.0));
        assert!(result.rows[a1][1].any_failed);
        assert!(!result.rows[a1][1].failed, "a well itself never failed");
        assert!(!result.rows[b2][1].any_failed);
        assert_eq!(
            plate_value(&generator, &plate_filter(Column::Count), "A1"),
            Some(1.0)
        );

        let multi = generator
            .get_group_by_plate_multi(
                &PlateFilterMulti {
                    plane: plane(),
                    grouping_regex: String::new(),
                    aggregation: vec![Aggregation::Avg],
                    object_class: vec![ObjectClass::Unset],
                    column: vec![Column::AreaSizePx],
                    color_schema: ColorSchema::default(),
                    color_scale: ColorScale::default(),
                    matrix_dimension: None,
                },
                &View::Heatmap,
            )
            .unwrap();
        // A1 -> (row 0, col 0)
        assert_eq!(as_float(&multi[0].rows[0][0].value), Some(10.0));
        assert!(multi[0].rows[0][0].any_failed);
    }

    #[test]
    fn well_shows_a_failed_fields_own_value_but_marks_it() {
        let generator = open(&[
            ObjectSpec::new("A1_01.tif", "ClassA", 1, 10),
            ObjectSpec::new("A1_02.tif", "ClassA", 1, 100),
        ]);
        mark_failed(&generator, "A1_02.tif");

        let single = generator
            .get_group_by_well(&well_filter("A1", Column::AreaSizePx), &View::List)
            .unwrap();
        let batched = generator
            .get_wells_for_plate(&wells_batch_filter(Column::AreaSizePx), &View::Heatmap)
            .unwrap();

        assert_eq!(as_float(&single.rows[1][1].value), Some(100.0));
        assert!(single.rows[1][1].failed);
        assert!(!single.rows[0][1].failed);
        // Field 02 -> heatmap position 1 (row 0, col 1).
        assert!(batched["A1"].rows[0][1].failed);
        assert!(!batched["A1"].rows[0][0].failed);
    }

    #[test]
    fn grouped_by_image_marks_every_cell_of_a_failed_images_rows() {
        let generator = open(&[
            ObjectSpec::new("img1.tif", "ClassA", 1, 10),
            ObjectSpec::new("img2.tif", "ClassA", 1, 20),
        ]);
        mark_failed(&generator, "img2.tif");

        let normal = grouped(&generator, Aggregation::Avg);
        let transposed = grouped_transposed(&generator, None, no_page());

        for result in [&normal, &transposed] {
            assert!(result.rows[0].iter().all(|cell| !cell.failed));
            assert!(result.rows[1].iter().all(|cell| cell.failed));
        }
        // Its objects (found before the error) are still shown.
        assert_eq!(cell_value(&transposed.rows[1][2]), Some(20.0));
    }

    #[test]
    fn transposed_grouped_places_each_classs_values_in_its_own_block() {
        let generator = open(&[
            ObjectSpec::new("img1.tif", "ClassA", 1, 10),
            ObjectSpec::new("img1.tif", "ClassA", 1, 30),
            ObjectSpec::new("img1.tif", "ClassB", 2, 5),
            ObjectSpec::new("img2.tif", "ClassB", 2, 7),
        ]);

        let result = grouped_transposed(&generator, None, no_page());

        let values = |row: usize| -> Vec<Option<f64>> {
            result.rows[row][1..]
                .iter()
                .map(|c| as_float(&c.value))
                .collect()
        };
        // (Count, Avg area) for ClassA, then for ClassB.
        assert_eq!(values(0), [Some(2.0), Some(20.0), Some(1.0), Some(5.0)]);
        assert_eq!(values(1), [Some(0.0), None, Some(1.0), Some(7.0)]);
    }

    #[test]
    fn class_filter_matches_objects_with_several_classes() {
        let generator = open(&[
            ObjectSpec::new("A1_01.tif", "ClassA", 1, 10),
            ObjectSpec::new("A1_01.tif", "ClassB", 2, 20),
        ]);
        // The first object is in both classes.
        generator
            .database
            .execute(
                "UPDATE objects SET object_class_id = '[1, 2]' WHERE area_px = 10",
                [],
            )
            .unwrap();
        let count_of = |id| PlateFilter {
            object_class: ObjectClass::Valid(id),
            ..plate_filter(Column::Count)
        };

        assert_eq!(plate_value(&generator, &count_of(1), "A1"), Some(1.0));
        assert_eq!(plate_value(&generator, &count_of(2), "A1"), Some(2.0));
        assert_eq!(plate_value(&generator, &count_of(3), "A1"), Some(0.0));
    }
}
