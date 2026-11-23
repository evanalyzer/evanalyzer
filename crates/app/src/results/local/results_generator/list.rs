//! # list
//!
//! The object list: a row per object (or per object and partner), its pages
//! and sort, the values of related objects and measurements, transposed.
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-11-22
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.

use super::*;

impl ResultsGenerator {
    /// The WHERE conditions of the object list `filter` asks for that the
    /// transposed table shares - plane, frame and `images` - and the class
    /// ids it keeps (`None`: every class); `None` when it matches nothing.
    /// The rest: [`push_object_list_conditions`].
    pub(super) fn object_list_base_conditions(
        &self,
        filter: &ListFilter,
        images: ImageSelection<'_>,
        classes: &[Class],
    ) -> Result<Option<ListConditions>, InternalErrors> {
        // `ListFilter.images` carries the rel-paths the GUI's image picker
        // keys on - the objects' `image_rel_path` (not `image_name`, the file
        // name only: images of the same name in two folders). `Some([])`
        // (an active filter matching nothing) means zero rows, same
        // convention as the class filter below; a path that names no image
        // simply matches no object.
        if images.matches_nothing() {
            return Ok(None);
        }

        // `ListFilter.object_classes` already carries `ObjectClass` ids, but
        // `classes` still doubles as validation: an id that no longer names
        // a registered class (e.g. stale GUI state after switching
        // databases) is dropped rather than matched against
        // `object_class_id` blindly.
        let class_ids = match &filter.object_classes {
            Some(wanted) => {
                let ids: Vec<u32> = wanted
                    .iter()
                    .filter_map(|id| classes.iter().find(|class| &class.global() == id))
                    .map(|class| class.id)
                    .collect();
                if ids.is_empty() {
                    return Ok(None);
                }
                Some(ids)
            }
            None => None,
        };

        let mut conditions = vec![
            self.object_on_plane_sql("", filter.plane.z_stack)?,
            format!("t_stack = {}", filter.plane.t_stack),
        ];
        if let Some(condition) = images.objects_condition() {
            conditions.push(condition);
        }
        Ok(Some((conditions, class_ids)))
    }

    /// Where object `object_id` lies in the object list `filter` asks for
    /// (one row per object, in `object_id` order, `filter.page.limit` per
    /// page): the start of every page before its own - the `object_id`
    /// each page comes after - so it is on page `len()` (0: the first).
    /// `None` when the list doesn't hold it (filtered out, another plane)
    /// or its rows aren't objects (transposed). `filter.page.after` is
    /// ignored.
    pub fn object_list_page_starts(
        &self,
        filter: &ListFilter,
        object_id: &str,
    ) -> Result<Option<Vec<String>>, InternalErrors> {
        let err = |e: duckdb::Error| InternalErrors::Io(e.to_string());
        if filter.transpond_table {
            return Ok(None);
        }
        let classes = self.get_object_classes()?;
        let Some((mut conditions, class_ids)) =
            self.object_list_base_conditions(filter, ImageSelection::of(&filter.images), &classes)?
        else {
            return Ok(None);
        };
        push_object_list_conditions(&mut conditions, class_ids.as_deref(), &filter.related_to);
        let where_clause = conditions.join(" AND ");
        let id = object_id.replace('\'', "''");
        let held: i64 = self
            .database
            .query_row(
                &format!(
                    "SELECT COUNT(*) FROM objects WHERE {where_clause} AND object_id = '{id}'::UUID"
                ),
                [],
                |row| row.get(0),
            )
            .map_err(err)?;
        if held == 0 {
            return Ok(None);
        }
        let limit = filter.page.limit.max(1);
        // Sorted by a column: pages start at row offsets (`sort_offset`) -
        // the starts of the pages before its row under the same order.
        if let Some(sort) = &filter.sort
            && let Some(expr) = object_sort_expr(sort)
        {
            let direction = if sort.descending { "DESC" } else { "ASC" };
            let row: Option<i64> = self
                .database
                .query_row(
                    &format!(
                        "SELECT rn FROM (SELECT object_id, row_number() OVER \
                         (ORDER BY ({expr}) {direction} NULLS LAST, object_id) - 1 AS rn \
                         FROM objects WHERE {where_clause}) WHERE object_id = '{id}'::UUID"
                    ),
                    [],
                    |row| row.get(0),
                )
                .map_err(err)?;
            let Some(row) = row else {
                return Ok(None);
            };
            let pages_before = row as usize / limit as usize;
            return Ok(Some(
                (1..=pages_before)
                    .map(|page| (page * limit as usize).to_string())
                    .collect(),
            ));
        }
        // Every `limit`-th object before it: the last of each page before
        // its own (the next page starts after it). Only the id is read.
        let sql = format!(
            "SELECT object_id::VARCHAR FROM \
             (SELECT object_id, row_number() OVER (ORDER BY object_id) AS rn FROM objects \
              WHERE {where_clause} AND object_id < '{id}'::UUID) \
             WHERE rn % {limit} = 0 ORDER BY rn"
        );
        let mut statement = self.database.prepare(&sql).map_err(err)?;
        let starts = statement
            .query_map([], |row| row.get::<_, String>(0))
            .map_err(err)?
            .collect::<Result<Vec<_>, _>>()
            .map_err(err)?;
        Ok(Some(starts))
    }

    pub fn get_object_list(&self, filter: &ListFilter) -> Result<DatabaseResult, InternalErrors> {
        self.get_object_list_of(filter, ImageSelection::of(&filter.images))
    }

    /// [`get_object_list`](Self::get_object_list) of `images` instead of
    /// `filter.images` (which is then ignored).
    pub(in crate::results::local) fn get_object_list_of(
        &self,
        filter: &ListFilter,
        images: ImageSelection<'_>,
    ) -> Result<DatabaseResult, InternalErrors> {
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

        // `ListFilter::with_coloc_details`: every class of a selected coloc
        // column (`ColocCount`, a combination's `Coloc` - each class once)
        // crossed with every selected plain-measurement column (see
        // `is_resolvable_metric`) adds one more header — that combination's
        // value resolved on the class's colocalizing partner object instead
        // of the source object. Needs at least one of each to mean anything;
        // with only coloc-class columns selected (no measurement to resolve)
        // or vice versa, list rows are built the same as when the flag is
        // off.
        let mut coloc_class_columns: Vec<ObjectClass> = Vec::new();
        for column in &ordered_columns {
            let classes: &[ObjectClass] = match column {
                Column::ColocCount(class) => std::slice::from_ref(class),
                Column::Coloc(c) => &c.with,
                _ => &[],
            };
            for class in classes {
                if !coloc_class_columns.contains(class) {
                    coloc_class_columns.push(*class);
                }
            }
        }
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

        let Some((mut conditions, class_ids)) =
            self.object_list_base_conditions(filter, images, &classes)?
        else {
            return Ok(empty_result(column_names.clone()));
        };
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
        push_object_list_conditions(&mut conditions, class_ids.as_deref(), &filter.related_to);
        // Keyset pagination (see the doc comment on `Pagination::after`):
        // narrowing to `object_id > cursor` here, in the same WHERE clause
        // DuckDB already zone-map-prunes on, is what lets it skip whole row
        // groups below the cursor instead of sorting/reading the full table.
        // Sorted by a column: over all the matching objects, paged by offset.
        let sort_expr = filter.sort.as_ref().and_then(object_sort_expr);
        if sort_expr.is_none()
            && let Some(cursor) = &filter.page.after
        {
            conditions.push(format!(
                "object_id > '{}'::UUID",
                cursor.replace('\'', "''")
            ));
        }
        let where_clause = format!("WHERE {}", conditions.join(" AND "));

        // Two-step fetch: first find just the first and last `object_id` of
        // this page — a query that only ever touches the (fixed-width,
        // cheap) filter columns and `object_id` itself, never the wide/JSON
        // columns below — then fetch the full rows of the same filter
        // between those two ids: exactly this page's rows, since the page is
        // the first `limit` matching ids in order. Doing it in one wide
        // `SELECT ... WHERE ... ORDER BY object_id LIMIT n` forces DuckDB to
        // decode every selected column (including whatever of the
        // colocalization map/intensity lists was requested) for every row
        // that matches the WHERE clause before it can even start sorting —
        // on this app's tables that's routinely the *entire* table, since
        // z/t-plane and image/class filters often don't narrow anything.
        // The id range lets DuckDB's per-row-group zone maps skip straight
        // to the row groups holding the page (measured this dropping a
        // ~5.5M-row table's per-page cost from single-digit GB to
        // single-digit MB, first page included). A range, not the page's
        // ids as an `IN (...)` list: DuckDB parsed and planned a 20,000-id
        // list for every export page - 39 s of a 6.6M-object List export.
        let limit = filter.page.limit.max(0);
        // Sorted: the page's ids in their order (a short list - a GUI page),
        // the rows put in that order below.
        let mut sorted_ids: Option<Vec<String>> = None;
        let where_clause = if let (Some(expr), Some(sort)) = (&sort_expr, &filter.sort) {
            let offset = sort_offset(&filter.page);
            let direction = if sort.descending { "DESC" } else { "ASC" };
            let mut stmt = self
                .database
                .prepare(&format!(
                    "SELECT object_id::VARCHAR FROM objects {where_clause} \
                     ORDER BY ({expr}) {direction} NULLS LAST, object_id \
                     LIMIT {limit} OFFSET {offset}"
                ))
                .map_err(err)?;
            let ids: Vec<String> = stmt
                .query_map([], |row| row.get(0))
                .map_err(err)?
                .collect::<Result<_, _>>()
                .map_err(err)?;
            if ids.is_empty() {
                return Ok(empty_result(column_names));
            }
            let list = ids
                .iter()
                .map(|id| format!("'{}'::UUID", id.replace('\'', "''")))
                .collect::<Vec<_>>()
                .join(", ");
            sorted_ids = Some(ids);
            format!("WHERE object_id IN ({list})")
        } else {
            let key_sql = format!(
                "SELECT MIN(object_id)::VARCHAR, MAX(object_id)::VARCHAR FROM \
                 (SELECT object_id FROM objects {where_clause} ORDER BY object_id LIMIT {limit})"
            );
            let (first, last): (Option<String>, Option<String>) = self
                .database
                .query_row(&key_sql, [], |row| Ok((row.get(0)?, row.get(1)?)))
                .map_err(err)?;
            let (Some(first), Some(last)) = (first, last) else {
                return Ok(empty_result(column_names));
            };
            format!("{where_clause} AND object_id BETWEEN '{first}'::UUID AND '{last}'::UUID")
        };

        // Only pull the source columns `ordered_columns` actually needs.
        // DuckDB is columnar: a column replaced by a constant here is never
        // read off disk or carried through the sort, so an unselected
        // `coloc_partner_ids`/`intensity_*` (each row's biggest fields, since
        // every other field is a fixed-width number or a short string) costs
        // nothing instead of being materialized for every row that matches
        // the WHERE clause before LIMIT/OFFSET trims it down to one page —
        // this was blowing up RAM on tables with hundreds of thousands of
        // objects even though only `LIST_PAGE_SIZE` rows ever reach the GUI.
        let needs = ObjectColumnNeeds::for_columns(&ordered_columns);
        let sql = format!(
            "SELECT {}\n FROM (SELECT * FROM objects {where_clause}) o \
             LEFT JOIN images i ON i.image_rel_path = o.image_rel_path\n ORDER BY o.object_id",
            object_select_clause(&needs)?
        );
        let value_keys = needs.value_keys();

        let mut stmt = self.database.prepare(&sql).map_err(err)?;
        let mut objects: Vec<ObjectRow> = stmt
            .query_map([], |row| map_object_row(row, &value_keys))
            .map_err(err)?
            .collect::<Result<Vec<_>, _>>()
            .map_err(err)?;
        if let Some(ids) = &sorted_ids {
            let place: HashMap<&str, usize> = ids
                .iter()
                .enumerate()
                .map(|(i, id)| (id.as_str(), i))
                .collect();
            objects.sort_by_key(|o| {
                place
                    .get(o.object_id.as_str())
                    .copied()
                    .unwrap_or(usize::MAX)
            });
        }
        self.add_measurement_values(objects.iter_mut(), &ordered_columns)?;
        self.add_measurement_objects(objects.iter_mut(), &ordered_columns)?;
        self.add_related_values(objects.iter_mut(), &ordered_columns)?;

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

    /// Reads the values of the children's / parent's columns among `columns`
    /// for `objects` (a page of the list) - the same SQL as the per-image,
    /// plate and chart views, so the numbers agree. One query for the page.
    pub(super) fn add_related_values<'a>(
        &self,
        objects: impl Iterator<Item = &'a mut ObjectRow>,
        columns: &[Column],
    ) -> Result<(), InternalErrors> {
        let err = |e: duckdb::Error| InternalErrors::Io(e.to_string());
        let related: Vec<&Column> = columns
            .iter()
            .filter(|c| matches!(c, Column::Children(_) | Column::Parent(_)))
            .collect();
        if related.is_empty() {
            return Ok(());
        }
        let mut objects: HashMap<String, &mut ObjectRow> =
            objects.map(|o| (o.object_id.clone(), o)).collect();
        if objects.is_empty() {
            return Ok(());
        }
        let ids = sql_string_in_list(&objects.keys().cloned().collect::<Vec<_>>());
        let expressions = related
            .iter()
            .map(|c| related_value_expr(c))
            .collect::<Result<Vec<_>, _>>()?;
        let sql = format!(
            "SELECT object_id::VARCHAR, {} FROM objects WHERE object_id IN ({ids})",
            expressions
                .iter()
                .map(|e| format!("CAST({e} AS VARCHAR)"))
                .collect::<Vec<_>>()
                .join(", ")
        );
        let keys: Vec<String> = related.iter().map(|c| c.as_key(&[])).collect();
        let texts: Vec<bool> = related
            .iter()
            .map(|c| {
                matches!(
                    c,
                    Column::Parent(ParentColumn {
                        value: ParentValue::Id,
                        ..
                    })
                )
            })
            .collect();
        let mut stmt = self.database.prepare(&sql).map_err(err)?;
        let mut rows = stmt.query([]).map_err(err)?;
        while let Some(row) = rows.next().map_err(err)? {
            let id: String = row.get(0).map_err(err)?;
            let Some(object) = objects.get_mut(&id) else {
                continue;
            };
            for (i, key) in keys.iter().enumerate() {
                let text: Option<String> = row.get(i + 1).map_err(err)?;
                let value = text.and_then(|t| {
                    if texts[i] {
                        Some(RelatedValue::Text(t))
                    } else {
                        t.parse().ok().map(RelatedValue::Number)
                    }
                });
                object.related.insert(key.clone(), value);
            }
        }
        Ok(())
    }

    /// Reads the values of the measurement columns among `columns` for
    /// `objects` (a page of the list): each object's own, or its image
    /// plane's for an image measurement. One query for the page.
    pub(super) fn add_measurement_values<'a>(
        &self,
        objects: impl Iterator<Item = &'a mut ObjectRow>,
        columns: &[Column],
    ) -> Result<(), InternalErrors> {
        let err = |e: duckdb::Error| InternalErrors::Io(e.to_string());
        let keys = |per_object: bool| -> Vec<String> {
            columns
                .iter()
                .filter_map(|c| match c {
                    Column::Measurement(m) if m.per_object == per_object => Some(m.key.clone()),
                    _ => None,
                })
                .collect()
        };
        let (object_keys, image_keys) = (keys(true), keys(false));
        if object_keys.is_empty() && image_keys.is_empty() {
            return Ok(());
        }
        let mut objects: HashMap<String, &mut ObjectRow> =
            objects.map(|o| (o.object_id.clone(), o)).collect();
        if objects.is_empty() {
            return Ok(());
        }
        let ids = sql_string_in_list(&objects.keys().cloned().collect::<Vec<_>>());
        let mut parts = Vec::new();
        if !object_keys.is_empty() {
            parts.push(format!(
                "SELECT object_id::VARCHAR, key, value FROM measurements \
                 WHERE region = 'object' AND value IS NOT NULL \
                 AND key IN ({}) AND object_id IN ({ids})",
                sql_string_in_list(&object_keys)
            ));
        }
        if !image_keys.is_empty() {
            parts.push(format!(
                "SELECT o.object_id::VARCHAR, m.key, m.value FROM objects o \
                 JOIN measurements m ON m.image_rel_path = o.image_rel_path \
                 AND m.z_stack IS NOT DISTINCT FROM o.z_stack AND m.t_stack = o.t_stack \
                 WHERE m.region = 'image' AND m.value IS NOT NULL \
                 AND m.key IN ({}) AND o.object_id IN ({ids})",
                sql_string_in_list(&image_keys)
            ));
        }
        let mut stmt = self
            .database
            .prepare(&parts.join(" UNION ALL "))
            .map_err(err)?;
        let values = stmt
            .query_map([], |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, f64>(2)?,
                ))
            })
            .map_err(err)?;
        for value in values {
            let (id, key, value) = value.map_err(err)?;
            if let Some(object) = objects.get_mut(&id) {
                object.measurements.insert(key, value);
            }
        }
        Ok(())
    }

    /// Reads the objects the [`Column::MeasurementObject`] columns among
    /// `columns` name for `objects` (a page of the list): their ids, kept
    /// with the related values. One query for the page.
    pub(super) fn add_measurement_objects<'a>(
        &self,
        objects: impl Iterator<Item = &'a mut ObjectRow>,
        columns: &[Column],
    ) -> Result<(), InternalErrors> {
        let err = |e: duckdb::Error| InternalErrors::Io(e.to_string());
        let wanted: Vec<&MeasurementColumn> = columns
            .iter()
            .filter_map(|c| match c {
                Column::MeasurementObject(m) => Some(m),
                _ => None,
            })
            .collect();
        if wanted.is_empty() {
            return Ok(());
        }
        let mut objects: HashMap<String, &mut ObjectRow> =
            objects.map(|o| (o.object_id.clone(), o)).collect();
        if objects.is_empty() {
            return Ok(());
        }
        let ids = sql_string_in_list(&objects.keys().cloned().collect::<Vec<_>>());
        let keys: Vec<String> = wanted.iter().map(|m| m.key.clone()).collect();
        let sql = format!(
            "SELECT object_id::VARCHAR, key, related_object_id::VARCHAR FROM measurements \
             WHERE region = 'object' AND related_object_id IS NOT NULL \
             AND key IN ({}) AND object_id IN ({ids})",
            sql_string_in_list(&keys)
        );
        let mut stmt = self.database.prepare(&sql).map_err(err)?;
        let rows = stmt
            .query_map([], |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, String>(2)?,
                ))
            })
            .map_err(err)?;
        for row in rows {
            let (id, key, related) = row.map_err(err)?;
            let Some(m) = wanted.iter().find(|m| m.key == key) else {
                continue;
            };
            if let Some(object) = objects.get_mut(&id) {
                let column = Column::MeasurementObject((*m).clone());
                object
                    .related
                    .insert(column.as_key(&[]), Some(RelatedValue::Text(related)));
            }
        }
        Ok(())
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
    pub(super) fn get_object_list_transposed(
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
            let class_label = class_display_label(ObjectClass::Global(*class_id), classes);
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
        let conditions = base_conditions.join(" AND ");
        let block_ids = sql_u32_list(blocks);

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
                    FROM ({}) AS u\n\
                    WHERE class_id IN ({block_ids})\n\
                 ), page AS (\n\
                    SELECT DISTINCT image_rel_path, rn FROM ranked {page_condition}\n\
                    ORDER BY image_rel_path, rn LIMIT {limit}\n\
                 )\n\
                 SELECT r.image_rel_path, r.image_name, r.rn, r.class_id, r.object_id::VARCHAR\n\
                 FROM ranked r JOIN page p ON r.image_rel_path = p.image_rel_path AND r.rn = p.rn\n\
                 ORDER BY r.image_rel_path, r.rn, r.class_id",
                objects_per_class_sql(&format!(
                    "{conditions} AND image_rel_path IN ({})",
                    sql_string_in_list(&images)
                )),
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
            object_select_clause(&needs)?,
            sql_string_in_list(&ids)
        );
        let value_keys = needs.value_keys();
        let mut stmt = self.database.prepare(&sql).map_err(err)?;
        let mut objects: HashMap<String, ObjectRow> = stmt
            .query_map([], |row| map_object_row(row, &value_keys))
            .map_err(err)?
            .map(|row| row.map(|object| (object.object_id.clone(), object)))
            .collect::<Result<_, _>>()
            .map_err(err)?;
        self.add_measurement_values(objects.values_mut(), &block_columns)?;
        self.add_measurement_objects(objects.values_mut(), &block_columns)?;
        self.add_related_values(objects.values_mut(), &block_columns)?;

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
                None => (image_rel_path.clone(), None),
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
    pub(super) fn build_coloc_detail_rows(
        &self,
        objects: &[ObjectRow],
        ordered_columns: &[Column],
        coloc_class_columns: &[ObjectClass],
        metric_columns: &[Column],
        classes: &[Class],
    ) -> Result<(Vec<String>, Vec<Vec<Cell>>, Vec<(String, Option<[u32; 4]>)>), InternalErrors>
    {
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
            let mut per_class: HashMap<ObjectClass, Vec<String>> = HashMap::new();
            for class in coloc_class_columns {
                let ids = coloc_partners(object, *class).cloned().unwrap_or_default();
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
                object_select_clause(&partner_needs)?,
                sql_string_in_list(&ids)
            );
            let value_keys = partner_needs.value_keys();
            let mut partner_stmt = self.database.prepare(&partner_sql).map_err(err)?;
            partner_stmt
                .query_map([], |row| map_object_row(row, &value_keys))
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
}

/// One row of the `objects` table, as fetched by `get_list`'s hand-written
/// SQL — only the columns needed to fill in any `Column` variant (see
/// `cell_for_column`), not every column the table has.
pub(super) struct ObjectRow {
    pub(super) object_id: String,
    pub(super) image_name: String,
    pub(super) object_class_name: Vec<String>,
    pub(super) seg_class_name: Option<String>,
    pub(super) area_px: u64,
    pub(super) area_nm2: f64,
    pub(super) perimeter_px: f64,
    pub(super) perimeter_nm: f64,
    pub(super) circularity: f64,
    pub(super) solidity: f64,
    pub(super) eccentricity: f64,
    /// The object's track, `None` if it isn't part of one (stored as 0).
    pub(super) track_id: Option<u64>,
    /// Colocalization partners: (partner class key, partner object ids) -
    /// see `coloc_partner_ids` in evanalyzer_core's duckdb.rs.
    pub(super) coloc: Vec<(i32, Vec<String>)>,
    /// Per-channel intensities in gray values, `[sum, mean, min, max]`,
    /// each indexed by channel (`None` where not measured).
    pub(super) intensities: [Vec<Option<f64>>; 4],
    // Always fetched (unlike every field above, gated by `ObjectColumnNeeds`
    // on whether its `Column` is actually selected/displayed) - needed by
    // the GUI to navigate to and highlight this object in its source image
    // (see `DatabaseResult::row_locations`) regardless of which columns the
    // user chose to show.
    pub(super) image_rel_path: String,
    pub(super) bbox_xmin_px: u32,
    pub(super) bbox_ymin_px: u32,
    pub(super) bbox_xmax_px: u32,
    pub(super) bbox_ymax_px: u32,
    pub(super) disabled: bool,
    /// The selected plain number columns' values (see
    /// [`is_value_column`]), by column key - read with the row.
    pub(super) values: HashMap<String, Option<f64>>,
    /// The selected measurements' values, by key - filled after the rows
    /// are read (`ResultsGenerator::add_measurement_values`).
    pub(super) measurements: HashMap<String, f64>,
    /// The selected children's / parent's values, by column key (`None`:
    /// no value) - filled with the measurements.
    pub(super) related: HashMap<String, Option<RelatedValue>>,
}

/// A [`Column::Children`] / [`Column::Parent`] value of a list row.
#[derive(Clone)]
pub(super) enum RelatedValue {
    Number(f64),
    Text(String),
}

/// Which of `ObjectRow`'s source columns a given column selection actually
/// needs — shared by `get_list`'s main fetch (needs from `ordered_columns`)
/// and its coloc-detail partner fetch (needs from just `metric_columns`,
/// see `Column::with_coloc_details` on `ListFilter`), so both build their
/// `SELECT` list and parse rows the exact same (bug-for-bug consistent) way.
#[derive(Default, Clone)]
pub(super) struct ObjectColumnNeeds {
    pub(super) image_name: bool,
    pub(super) class: bool,
    pub(super) area_px: bool,
    pub(super) area_nm2: bool,
    pub(super) perimeter_px: bool,
    pub(super) perimeter_nm: bool,
    pub(super) circularity: bool,
    pub(super) solidity: bool,
    pub(super) eccentricity: bool,
    pub(super) track_id: bool,
    pub(super) coloc: bool,
    pub(super) intensities: bool,
    /// The plain number columns among the selection, read as
    /// `column_aggregate_expr` after the fixed columns.
    pub(super) values: Vec<Column>,
}

impl ObjectColumnNeeds {
    pub(super) fn for_columns(columns: &[Column]) -> Self {
        Self {
            image_name: columns.contains(&Column::ImageName),
            // A coloc column also needs the object's classes: its coloc
            // with its own class has no value.
            class: columns.contains(&Column::ObjectClass)
                || columns
                    .iter()
                    .any(|c| matches!(c, Column::ColocCount(_) | Column::Coloc(_))),
            area_px: columns.contains(&Column::AreaSizePx),
            area_nm2: columns.contains(&Column::AreaSizeNm),
            perimeter_px: columns.contains(&Column::PerimeterPx),
            perimeter_nm: columns.contains(&Column::PerimeterNm),
            circularity: columns.contains(&Column::Circularity),
            solidity: columns.contains(&Column::Solidity),
            eccentricity: columns.contains(&Column::Eccentricity),
            track_id: columns.contains(&Column::TrackId),
            coloc: columns
                .iter()
                .any(|c| matches!(c, Column::ColocCount(_) | Column::Coloc(_))),
            intensities: columns.iter().any(|c| {
                matches!(
                    c,
                    Column::IntensityAvg(_)
                        | Column::IntensitySum(_)
                        | Column::IntensityMin(_)
                        | Column::IntensityMax(_)
                )
            }),
            values: columns
                .iter()
                .filter(|c| is_value_column(c))
                .cloned()
                .collect(),
        }
    }

    /// The `values` columns' keys, in `SELECT` order.
    pub(super) fn value_keys(&self) -> Vec<String> {
        self.values.iter().map(|c| c.as_key(&[])).collect()
    }
}

/// A column that is one number per object, read with
/// [`column_aggregate_expr`] in the list too (instead of a field of its
/// own in `ObjectRow`): the newer per-object columns.
pub(super) fn is_value_column(column: &Column) -> bool {
    matches!(
        column,
        Column::VolumePx
            | Column::VolumeNm
            | Column::SurfaceAreaNm
            | Column::MaxFeret
            | Column::MinFeret
            | Column::MajorAxis
            | Column::MinorAxis
            | Column::EquivalentDiameter
            | Column::Extent
            | Column::IntensityStd(_)
            | Column::IntensityMedian(_)
            | Column::IntensityLowerQuartile(_)
            | Column::IntensityUpperQuartile(_)
            | Column::IntensityMad(_)
    )
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
pub(super) fn object_select_clause(need: &ObjectColumnNeeds) -> Result<String, InternalErrors> {
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
    // 0 is "not tracked" (`TrackId`'s default; real tracks start at 1).
    let select_track_id = if need.track_id {
        "NULLIF(o.track_id, 0)"
    } else {
        "NULL::UBIGINT"
    };
    let select_coloc = if need.coloc {
        "map_keys(o.coloc_partner_ids), CAST(map_values(o.coloc_partner_ids) AS VARCHAR[][])"
    } else {
        "NULL::INTEGER[], NULL::VARCHAR[][]"
    };
    let select_intensities = if need.intensities {
        "o.intensity_sum_gray, o.intensity_mean_gray, o.intensity_min_gray, \
         o.intensity_max_gray"
    } else {
        "NULL::DOUBLE[], NULL::DOUBLE[], NULL::DOUBLE[], NULL::DOUBLE[]"
    };
    let mut select_values = String::new();
    for column in &need.values {
        select_values.push_str(&format!(
            ", CAST(({}) AS DOUBLE)",
            column_aggregate_expr(column)?
        ));
    }
    Ok(format!(
        "o.object_id, {select_image_name}, {select_object_class_name}, {select_seg_class_name},\n\
                {select_area_px}, {select_area_nm2}, {select_perimeter_px}, {select_perimeter_nm},\n\
                {select_circularity}, {select_solidity}, {select_eccentricity},\n\
                {select_coloc}, {select_intensities},\n\
                o.image_rel_path, o.bbox_xmin_px, o.bbox_ymin_px, o.bbox_xmax_px, o.bbox_ymax_px,\n\
                COALESCE(i.disabled, false), {select_track_id}{select_values}"
    ))
}

/// Inverse of `object_select_clause`'s fixed column position order —
/// shared so the main and partner fetches in `get_list` can never drift.
/// `value_keys`: [`ObjectColumnNeeds::value_keys`], the columns after the
/// fixed ones.
pub(super) fn map_object_row(
    row: &duckdb::Row<'_>,
    value_keys: &[String],
) -> duckdb::Result<ObjectRow> {
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
        coloc: extract_int_list(row.get::<_, Value>(11)?)
            .into_iter()
            .zip(match row.get::<_, Value>(12)? {
                Value::List(lists) | Value::Array(lists) => {
                    lists.into_iter().map(extract_string_list).collect()
                }
                _ => Vec::new(),
            })
            .collect(),
        intensities: [
            extract_f64_list(row.get::<_, Value>(13)?),
            extract_f64_list(row.get::<_, Value>(14)?),
            extract_f64_list(row.get::<_, Value>(15)?),
            extract_f64_list(row.get::<_, Value>(16)?),
        ],
        image_rel_path: row.get(17)?,
        bbox_xmin_px: row.get(18)?,
        bbox_ymin_px: row.get(19)?,
        bbox_xmax_px: row.get(20)?,
        bbox_ymax_px: row.get(21)?,
        disabled: row.get(22)?,
        track_id: row.get(23)?,
        values: value_keys
            .iter()
            .enumerate()
            .map(|(i, key)| Ok((key.clone(), row.get::<_, Option<f64>>(24 + i)?)))
            .collect::<duckdb::Result<_>>()?,
        measurements: HashMap::new(),
        related: HashMap::new(),
    })
}

/// `DatabaseResult::row_locations`' entry for `object` — its source image
/// (by rel path) and pixel bounding box, for the GUI to navigate to and
/// highlight it.
pub(super) fn object_location(object: &ObjectRow) -> (String, Option<[u32; 4]>) {
    (
        object.image_rel_path.clone(),
        Some([
            object.bbox_xmin_px,
            object.bbox_ymin_px,
            object.bbox_xmax_px,
            object.bbox_ymax_px,
        ]),
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
pub(super) fn is_resolvable_metric(column: &Column) -> bool {
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
    ) || is_value_column(column)
}

/// An empty, unstyled cell - also the filler for a class block with no
/// object in a transposed row.
pub(super) fn plain_cell() -> Cell {
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

pub(super) fn empty_database_result(column_names: Vec<String>) -> DatabaseResult {
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
pub(super) fn value_range<'a>(values: impl Iterator<Item = &'a Option<f64>>) -> (f32, f32) {
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

pub(super) fn cell_for_column(column: &Column, object: &ObjectRow, classes: &[Class]) -> Cell {
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
        Column::TrackId => Cell {
            value: object
                .track_id
                .map_or(CellValue::Empty, |id| CellValue::String(id.to_string())),
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
        // A coloc of the object's own class wasn't measured (a class isn't
        // colocalized with itself): no value, not 0.
        Column::ColocCount(class) if is_own_class(object, std::slice::from_ref(class), classes) => {
            Cell {
                value: CellValue::Empty,
                ..no_bg
            }
        }
        Column::Coloc(c) if is_own_class(object, &c.with, classes) => Cell {
            value: CellValue::Empty,
            ..no_bg
        },
        Column::ColocCount(class) => Cell {
            value: CellValue::Integer(
                coloc_partners(object, *class).map_or(0, |ids| ids.len() as i32),
            ),
            ..no_bg
        },
        Column::Coloc(c) => Cell {
            value: CellValue::Integer(coloc_value(c, |class| {
                coloc_partners(object, class).map_or(0, |ids| ids.len())
            }) as i32),
            ..no_bg
        },
        Column::IntensitySum(channel) => Cell {
            value: CellValue::Float(intensity_stat(object, IntensityStat::Sum, *channel)),
            ..no_bg
        },
        Column::IntensityAvg(channel) => Cell {
            value: CellValue::Float(intensity_stat(object, IntensityStat::Mean, *channel)),
            ..no_bg
        },
        Column::IntensityMin(channel) => Cell {
            value: CellValue::Float(intensity_stat(object, IntensityStat::Min, *channel)),
            ..no_bg
        },
        Column::IntensityMax(channel) => Cell {
            value: CellValue::Float(intensity_stat(object, IntensityStat::Max, *channel)),
            ..no_bg
        },
        Column::VolumePx
        | Column::VolumeNm
        | Column::SurfaceAreaNm
        | Column::MaxFeret
        | Column::MinFeret
        | Column::MajorAxis
        | Column::MinorAxis
        | Column::EquivalentDiameter
        | Column::Extent
        | Column::IntensityStd(_)
        | Column::IntensityMedian(_)
        | Column::IntensityLowerQuartile(_)
        | Column::IntensityUpperQuartile(_)
        | Column::IntensityMad(_) => Cell {
            value: object
                .values
                .get(&column.as_key(&[]))
                .copied()
                .flatten()
                .map_or(CellValue::Empty, |v| CellValue::Float(v as f32)),
            ..no_bg
        },
        Column::Measurement(m) => Cell {
            value: object
                .measurements
                .get(&m.key)
                .map_or(CellValue::Empty, |v| CellValue::Float(*v as f32)),
            ..no_bg
        },
        Column::Children(_) | Column::Parent(_) | Column::MeasurementObject(_) => {
            let count = matches!(
                column,
                Column::Children(ChildColumn {
                    value: ChildValue::Count,
                    ..
                })
            );
            Cell {
                value: match object.related.get(&column.as_key(&[])).cloned().flatten() {
                    Some(RelatedValue::Number(v)) if count => CellValue::Integer(v as i32),
                    Some(RelatedValue::Number(v)) => CellValue::Float(v as f32),
                    Some(RelatedValue::Text(text)) => CellValue::String(text),
                    None if count => CellValue::Integer(0),
                    None => CellValue::Empty,
                },
                ..no_bg
            }
        }
    }
}

/// The `coloc_partner_ids` map key of `class` - its id, or -1 for
/// partners without a class (see evanalyzer_core's duckdb.rs).
pub(super) fn coloc_class_key(class: ObjectClass) -> i32 {
    match class {
        ObjectClass::Global(n) => n as i32,
        ObjectClass::Unset | ObjectClass::Local(_) => -1,
    }
}

/// `object`'s colocalization partners of `class`, if it has any.
pub(super) fn coloc_partners(object: &ObjectRow, class: ObjectClass) -> Option<&Vec<String>> {
    let key = coloc_class_key(class);
    object
        .coloc
        .iter()
        .find(|(partner_class, _)| *partner_class == key)
        .map(|(_, ids)| ids)
}

/// A [`Column::Coloc`] value from an object's number of partners per
/// class (`partners_of`) - what [`coloc_value_expr`] computes in SQL.
pub(super) fn coloc_value(c: &ColocColumn, partners_of: impl Fn(ObjectClass) -> usize) -> usize {
    let matches = c.with.iter().all(|class| partners_of(*class) > 0)
        && (c.how == ColocMatch::AtLeast
            || c.among
                .iter()
                .filter(|class| !c.with.contains(class))
                .all(|class| partners_of(*class) == 0));
    match (matches, c.partners) {
        (false, _) => 0,
        (true, false) => 1,
        (true, true) => c.with.iter().map(|class| partners_of(*class)).sum(),
    }
}

/// Whether one of `with` is a class of `object` - its coloc with its own
/// class, never measured.
pub(super) fn is_own_class(object: &ObjectRow, with: &[ObjectClass], classes: &[Class]) -> bool {
    with.iter().any(|class| {
        classes
            .iter()
            .find(|c| c.global() == *class)
            .is_some_and(|c| object.object_class_name.contains(&c.name))
    })
}

/// `expr` - or NULL for an object of one of `with`'s classes (its coloc
/// with its own class isn't measured: no value, left out of aggregates).
pub(super) fn unless_own_class(with: &[ObjectClass], expr: String) -> String {
    let ids: Vec<String> = with
        .iter()
        .filter_map(|class| match class {
            ObjectClass::Global(id) => Some(id.to_string()),
            _ => None,
        })
        .collect();
    if ids.is_empty() {
        return expr;
    }
    format!(
        "(CASE WHEN list_has_any(object_class_id, [{}]) THEN NULL ELSE {expr} END)",
        ids.join(", ")
    )
}

/// SQL for a [`Column::Coloc`] value of the row's object (see
/// [`coloc_value`]): 1 / 0 for whether it matches, or its partners of the
/// classes when it does.
pub(super) fn coloc_value_expr(c: &ColocColumn) -> String {
    let count = |class: &ObjectClass| {
        format!(
            "COALESCE(len(coloc_partner_ids[{}]), 0)",
            coloc_class_key(*class)
        )
    };
    let mut conditions: Vec<String> = c
        .with
        .iter()
        .map(|class| format!("{} > 0", count(class)))
        .collect();
    if c.how == ColocMatch::Exactly {
        conditions.extend(
            c.among
                .iter()
                .filter(|class| !c.with.contains(class))
                .map(|class| format!("{} = 0", count(class))),
        );
    }
    let condition = if conditions.is_empty() {
        "true".to_string()
    } else {
        conditions.join(" AND ")
    };
    let value = if c.partners && !c.with.is_empty() {
        c.with.iter().map(count).collect::<Vec<_>>().join(" + ")
    } else {
        "1".to_string()
    };
    format!("(CASE WHEN {condition} THEN {value} ELSE 0 END)")
}

/// Position of a statistic in `ObjectRow::intensities`.
#[derive(Clone, Copy)]
pub(super) enum IntensityStat {
    Sum,
    Mean,
    Min,
    Max,
}

/// One channel's statistic in gray values; 0 for a channel that wasn't
/// measured (as before, when it came from JSON).
pub(super) fn intensity_stat(object: &ObjectRow, stat: IntensityStat, channel: u32) -> f32 {
    object.intensities[stat as usize]
        .get(channel as usize)
        .copied()
        .flatten()
        .unwrap_or(0.0) as f32
}

/// SQL scalar expression for a `Column`, to be wrapped in an aggregate
/// function by `get_group_by_plate`/`get_group_by_well`/`get_image_heatmap`
/// — also `results_charts.rs`'s histogram/scatter/boxplot queries (hence
/// `pub(super)`), which aggregate nothing themselves but still need a bare
/// per-object value expression to bucket/plot/box.
/// Plain numeric `objects` columns are a direct column reference;
/// `ColocCount` looks the class up in `coloc_partner_ids` and intensities
/// pick their channel out of the per-channel lists (the same values
/// `coloc_partners`/`intensity_stat` read per row for the list view).
/// Unqualified column names: none of them exists in `images` as well, so
/// they stay unambiguous in queries joining it.
/// The SQL to sort a list's objects by (`sort`'s column): its value per
/// object, the image's path for Image. `None`: not sortable this way.
pub(super) fn object_sort_expr(sort: &SortBy) -> Option<String> {
    match &sort.column {
        Column::ImageName => Some("image_rel_path".to_string()),
        Column::Count | Column::ObjectId | Column::ObjectClass => None,
        Column::Measurement(_) | Column::Children(_) | Column::Parent(_) => None,
        column => column_aggregate_expr(column).ok(),
    }
}

/// A grouped list's sort: the statistic it sorts by (its index in
/// `stats`) and whether descending - `None` without a sort or for one the
/// rows don't have.
pub(super) fn stat_sort(
    filter: &GroupedByImageFilter,
    stats: &[(&Column, &Aggregation)],
) -> Option<(usize, bool)> {
    let sort = filter.sort.as_ref()?;
    stats
        .iter()
        .position(|(column, aggregation)| {
            **column == sort.column && sort.aggregation.as_ref().is_none_or(|a| a == *aggregation)
        })
        .map(|i| (i, sort.descending))
}

/// With a sort, the rows before the page (`Pagination::after`).
pub(super) fn sort_offset(page: &Pagination) -> u64 {
    page.after
        .as_deref()
        .and_then(|after| after.parse().ok())
        .unwrap_or(0)
}

/// An object list's WHERE conditions and the class ids it keeps (`None`:
/// every class) - see [`ResultsGenerator::object_list_base_conditions`].
pub(super) type ListConditions = (Vec<String>, Option<Vec<u32>>);

/// The WHERE conditions of the object list's rows beyond
/// [`ResultsGenerator::object_list_base_conditions`]: its classes
/// (`class_ids`, `None`: all) and the objects it is drilled down to.
pub(super) fn push_object_list_conditions(
    conditions: &mut Vec<String>,
    class_ids: Option<&[u32]>,
    related_to: &Option<RelatedTo>,
) {
    if let Some(ids) = class_ids {
        conditions.push(class_filter_sql("object_class_id", ids));
    }
    match related_to {
        Some(RelatedTo::ChildrenOf(parent_class, parent_id)) => conditions.push(format!(
            "parent_ids[{}] = '{}'::UUID",
            coloc_class_key(*parent_class),
            parent_id.replace('\'', "''")
        )),
        Some(RelatedTo::PartnersOf(class, id)) => conditions.push(format!(
            "object_id IN (SELECT UNNEST(coloc_partner_ids[{}]) FROM objects \
             WHERE object_id = '{}'::UUID)",
            coloc_class_key(*class),
            id.replace('\'', "''")
        )),
        Some(RelatedTo::RelatedOf(key, id)) => conditions.push(format!(
            "object_id IN (SELECT related_object_id FROM measurements \
             WHERE region = 'object' AND key = '{}' AND object_id = '{}'::UUID)",
            key.replace('\'', "''"),
            id.replace('\'', "''")
        )),
        Some(RelatedTo::ParentOf(class, id)) => conditions.push(format!(
            "object_id IN (SELECT parent_ids[{}] FROM objects WHERE object_id = '{}'::UUID)",
            coloc_class_key(*class),
            id.replace('\'', "''")
        )),
        None => {}
    }
}
