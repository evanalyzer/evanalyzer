//! # plate
//!
//! The plate, its wells and an image's heatmap: a value per well, per image
//! of a well, per tile of an image.
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-11-22
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.

use super::*;

impl ResultsGenerator {
    /// Per-well statistics over the *per-image object counts* - what
    /// `Column::Count` means in the plate view: "Avg" is the average number
    /// of objects per image of the well, "Min" the smallest count of any
    /// image, "Sum" the total, ...
    ///
    /// Counted are the enabled, successfully analysed images that were
    /// analysed on the plane (see `measured_on_plane_sql`), *including* those
    /// without objects (count 0) - leaving those out would inflate every
    /// average. A well without such an image has no row (-> empty).
    ///
    /// `values`: (output column name, SQL aggregate fn) pairs, one per
    /// statistic. `regex_sql` must already be quote-escaped.
    pub(super) fn plate_count_sql(
        &self,
        grouping: &GroupingSql,
        plane: &PlaneFilter,
        object_class: ObjectClass,
        values: &[(String, &'static str)],
    ) -> Result<String, InternalErrors> {
        let mut conditions = vec![
            self.object_on_plane_sql("", plane.z_stack)?,
            format!("t_stack = {}", plane.t_stack),
        ];
        if let ObjectClass::Global(id) = object_class {
            conditions.push(class_filter_sql("object_class_id", &[id]));
        }
        let object_where = conditions.join(" AND ");
        let measured = self.measured_on_plane_sql("i", plane)?;
        let (group_col, regex_sql) = (grouping.column, grouping.regex.as_str());
        let aggregates = values
            .iter()
            .map(|(name, agg_fn)| format!("{agg_fn}(CAST(n AS DOUBLE)) AS {name}"))
            .collect::<Vec<_>>()
            .join(", ");
        Ok(format!(
            "SELECT group_prefix, row, col, {aggregates}\n\
             FROM (\n\
                 SELECT\n\
                     regexp_extract(i.{group_col}, '{regex_sql}', 1) AS group_prefix,\n\
                     regexp_extract(i.{group_col}, '{regex_sql}', 2) AS row,\n\
                     regexp_extract(i.{group_col}, '{regex_sql}', 3) AS col,\n\
                     COALESCE(c.n, 0) AS n\n\
                 FROM images i\n\
                 LEFT JOIN (\n\
                     SELECT image_rel_path, COUNT(*) AS n\n\
                     FROM objects\n\
                     WHERE {object_where}\n\
                     GROUP BY image_rel_path\n\
                 ) c ON c.image_rel_path = i.image_rel_path\n\
                 WHERE NOT i.disabled AND i.successful AND {measured}\n\
             )\n\
             GROUP BY group_prefix, row, col"
        ))
    }

    /// Per-well statistics over the *image measurements* of the well's
    /// images - like `plate_count_sql`: each enabled, successfully analysed
    /// image counts once, whatever its number of objects; one without a
    /// value is left out. `values`: (output column name, measurement key,
    /// SQL aggregate fn) per statistic.
    pub(super) fn plate_image_measurement_sql(
        &self,
        grouping: &GroupingSql,
        plane: &PlaneFilter,
        values: &[(String, String, &'static str)],
    ) -> Result<String, InternalErrors> {
        let measured = self.measured_on_plane_sql("i", plane)?;
        let (group_col, regex_sql) = (grouping.column, grouping.regex.as_str());
        let per_image = values
            .iter()
            .enumerate()
            .map(|(j, (_, key, _))| {
                format!(
                    "AVG(value) FILTER (WHERE key = '{}') AS v_{j}",
                    key.replace('\'', "''")
                )
            })
            .collect::<Vec<_>>()
            .join(", ");
        let image_values = (0..values.len())
            .map(|j| format!("m.v_{j}"))
            .collect::<Vec<_>>()
            .join(", ");
        let aggregates = values
            .iter()
            .enumerate()
            .map(|(j, (name, _, agg_fn))| format!("{agg_fn}(v_{j}) AS {name}"))
            .collect::<Vec<_>>()
            .join(", ");
        Ok(format!(
            "SELECT group_prefix, row, col, {aggregates}\n\
             FROM (\n\
                 SELECT\n\
                     regexp_extract(i.{group_col}, '{regex_sql}', 1) AS group_prefix,\n\
                     regexp_extract(i.{group_col}, '{regex_sql}', 2) AS row,\n\
                     regexp_extract(i.{group_col}, '{regex_sql}', 3) AS col,\n\
                     {image_values}\n\
                 FROM images i\n\
                 LEFT JOIN (\n\
                     SELECT image_rel_path, {per_image}\n\
                     FROM measurements\n\
                     WHERE region = 'image' AND (z_stack = {z} OR z_stack IS NULL) \
                         AND t_stack = {t}\n\
                     GROUP BY image_rel_path\n\
                 ) m ON m.image_rel_path = i.image_rel_path\n\
                 WHERE NOT i.disabled AND i.successful AND {measured}\n\
             )\n\
             GROUP BY group_prefix, row, col",
            z = plane.z_stack,
            t = plane.t_stack,
        ))
    }

    pub fn get_group_by_plate(
        &self,
        filter: &PlateFilter,
        view: &View,
    ) -> Result<DatabaseResult, InternalErrors> {
        let err = |e: duckdb::Error| InternalErrors::Io(e.to_string());
        let classes = self.get_object_classes()?;

        let grouping = self.grouping_sql(&filter.grouping)?;
        let (group_col, regex_sql) = (grouping.column, grouping.regex.as_str());
        let measured = self.measured_on_plane_sql("images", &filter.plane)?;

        // `value`: the SELECT expression for the well's value, `agg_sql`: the
        // subquery producing it (aliased `agg`, keyed by well).
        //
        // `Column::Count` is aggregated over the per-image counts of a well
        // (see `plate_count_sql`); every other column over the objects of the
        // well directly, filtered+aggregated before the join with `images`
        // (see the benchmark note in the old version of this function: a flat
        // join was ~4x slower).
        let (value, agg_sql) = if matches!(filter.column, Column::Count) {
            let agg_sql = self.plate_count_sql(
                &grouping,
                &filter.plane,
                filter.object_class,
                &[("value".to_string(), aggregation_sql_fn(&filter.aggregation))],
            )?;
            ("agg.value".to_string(), agg_sql)
        } else if let Column::Measurement(m) = &filter.column
            && !m.per_object
        {
            // An image measurement: over the well's images' own values.
            let agg_sql = self.plate_image_measurement_sql(
                &grouping,
                &filter.plane,
                &[(
                    "value".to_string(),
                    m.key.clone(),
                    aggregation_sql_fn(&filter.aggregation),
                )],
            )?;
            ("agg.value".to_string(), agg_sql)
        } else {
            let (agg_fn, value_expr) = aggregate_sql(&filter.column, &filter.aggregation)?;
            let mut object_conditions = vec![
                self.object_on_plane_sql("o", filter.plane.z_stack)?,
                format!("o.t_stack = {}", filter.plane.t_stack),
            ];
            if let ObjectClass::Global(id) = filter.object_class {
                object_conditions.push(class_filter_sql("o.object_class_id", &[id]));
            }
            let object_where = object_conditions.join(" AND ");
            let value = fill_zero_sql(
                "agg.value",
                &filter.column,
                &filter.aggregation,
                "agg.n_objects IS NULL",
                "img.any_measured",
            );
            let agg_sql = format!(
                "SELECT\n\
                     regexp_extract(o.{group_col}, '{regex_sql}', 1) AS group_prefix,\n\
                     regexp_extract(o.{group_col}, '{regex_sql}', 2) AS row,\n\
                     regexp_extract(o.{group_col}, '{regex_sql}', 3) AS col,\n\
                     COUNT(*) AS n_objects,\n\
                     {agg_fn}({value_expr}) AS value\n\
                 FROM objects o\n\
                 JOIN images i ON i.image_rel_path = o.image_rel_path\n\
                 WHERE NOT i.disabled AND i.successful AND {object_where}\n\
                 GROUP BY group_prefix, row, col"
            );
            (value, agg_sql)
        };

        // A disabled image's objects are excluded from `value`, and so are a
        // failed image's (`NOT successful`), but the image itself is never
        // dropped: every well still appears, `any_disabled`/`any_failed` tell
        // the caller that at least one of its images was excluded.
        // `any_measured`: at least one enabled image of the well was analysed
        // on this plane, so a well without objects has a real Sum of 0
        // (`fill_zero_sql`).
        let sql = format!(
            "SELECT img.group_prefix, img.row, img.col, {value}, img.any_disabled, img.any_failed\n\
             FROM (\n\
                 SELECT\n\
                     regexp_extract({group_col}, '{regex_sql}', 1) AS group_prefix,\n\
                     regexp_extract({group_col}, '{regex_sql}', 2) AS row,\n\
                     regexp_extract({group_col}, '{regex_sql}', 3) AS col,\n\
                     bool_or(disabled) AS any_disabled,\n\
                     bool_or(NOT successful) AS any_failed,\n\
                     bool_or(NOT disabled AND {measured}) AS any_measured\n\
                 FROM images\n\
                 GROUP BY group_prefix, row, col\n\
             ) img\n\
             LEFT JOIN (\n\
                 {agg_sql}\n\
             ) agg USING (group_prefix, row, col)\n\
             ORDER BY img.group_prefix"
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
            filter.plate_size,
            &filter.color_schema,
            &filter.color_scale,
            view,
        ))
    }

    // Batched form of `get_group_by_plate` across every requested column,
    // aggregation *and* object class at once: one SELECT per class.
    // `Column::Count` combos are aggregated over per-image counts (see
    // `plate_count_sql`) in their own subquery (`cnt`), joined next to the
    // regular per-object one (`agg`) that all other combos share.
    //
    // Returns one `DatabaseResult` per (class, column, aggregation) combo,
    // flattened in that nesting order.
    pub fn get_group_by_plate_multi(
        &self,
        filter: &PlateFilterMulti,
        view: &View,
    ) -> Result<Vec<DatabaseResult>, InternalErrors> {
        let err = |e: duckdb::Error| InternalErrors::Io(e.to_string());
        let classes = self.get_object_classes()?;

        // Per combo (in column x aggregation order): its output expression,
        // plus the aggregate definitions for the `agg` (per-object) and `cnt`
        // (per-image count) subqueries. `k` is the combo index and names
        // `value_{k}` in both.
        let mut value_cols: Vec<String> = Vec::new();
        let mut object_value_exprs: Vec<String> = Vec::new();
        let mut count_values: Vec<(String, &'static str)> = Vec::new();
        // Per-image measurement statistics (an image's own value).
        let mut image_values: Vec<(String, String, &'static str)> = Vec::new();
        for column in &filter.column {
            for aggregation in &filter.aggregation {
                let k = value_cols.len();
                if matches!(column, Column::Count) {
                    value_cols.push(format!("cnt.value_{k}"));
                    count_values.push((format!("value_{k}"), aggregation_sql_fn(aggregation)));
                } else if let Column::Measurement(m) = column
                    && !m.per_object
                {
                    value_cols.push(format!("imv.value_{k}"));
                    image_values.push((
                        format!("value_{k}"),
                        m.key.clone(),
                        aggregation_sql_fn(aggregation),
                    ));
                } else {
                    value_cols.push(fill_zero_sql(
                        &format!("agg.value_{k}"),
                        column,
                        aggregation,
                        "agg.n_objects IS NULL",
                        "img.any_measured",
                    ));
                    let (agg_fn, value_expr) = aggregate_sql(column, aggregation)?;
                    object_value_exprs.push(format!("{agg_fn}({value_expr}) AS value_{k}"));
                }
            }
        }
        let n = value_cols.len();
        let value_cols_sql = value_cols.join(", ");
        let object_values_sql = object_value_exprs.join(",\n                    ");

        let grouping = self.grouping_sql(&filter.grouping)?;
        let (group_col, regex) = (grouping.column, grouping.regex.as_str());

        let measured = self.measured_on_plane_sql("images", &filter.plane)?;
        let mut results = Vec::with_capacity(filter.object_class.len() * n);
        for object_class in &filter.object_class {
            // Per-object statistics (everything but Count).
            let agg_join = if object_value_exprs.is_empty() {
                String::new()
            } else {
                let mut object_conditions = vec![
                    self.object_on_plane_sql("o", filter.plane.z_stack)?,
                    format!("o.t_stack = {}", filter.plane.t_stack),
                ];
                if let ObjectClass::Global(id) = object_class {
                    object_conditions.push(class_filter_sql("o.object_class_id", &[*id]));
                }
                let object_where = object_conditions.join(" AND ");
                format!(
                    "LEFT JOIN (\n\
                         SELECT\n\
                             regexp_extract(o.{group_col}, '{regex}', 1) AS group_prefix,\n\
                             regexp_extract(o.{group_col}, '{regex}', 2) AS row,\n\
                             regexp_extract(o.{group_col}, '{regex}', 3) AS col,\n\
                             COUNT(*) AS n_objects,\n\
                             {object_values_sql}\n\
                         FROM objects o\n\
                         JOIN images i ON i.image_rel_path = o.image_rel_path\n\
                         WHERE NOT i.disabled AND i.successful AND {object_where}\n\
                         GROUP BY group_prefix, row, col\n\
                     ) agg USING (group_prefix, row, col)\n"
                )
            };
            // Per-image count statistics (Count).
            let cnt_join = if count_values.is_empty() {
                String::new()
            } else {
                let cnt_sql =
                    self.plate_count_sql(&grouping, &filter.plane, *object_class, &count_values)?;
                format!(
                    "LEFT JOIN (\n{cnt_sql}\n) cnt\n\
                         ON cnt.group_prefix = img.group_prefix\n\
                         AND cnt.row = img.row AND cnt.col = img.col\n"
                )
            };

            // Image measurement statistics - the same for every class.
            let imv_join = if image_values.is_empty() {
                String::new()
            } else {
                let imv_sql =
                    self.plate_image_measurement_sql(&grouping, &filter.plane, &image_values)?;
                format!(
                    "LEFT JOIN (\n{imv_sql}\n) imv\n\
                         ON imv.group_prefix = img.group_prefix\n\
                         AND imv.row = img.row AND imv.col = img.col\n"
                )
            };

            let sql = format!(
                "SELECT img.group_prefix, img.row, img.col, {value_cols_sql}, img.any_disabled,\n\
                 img.any_failed\n\
                 FROM (\n\
                     SELECT\n\
                         regexp_extract({group_col}, '{regex}', 1) AS group_prefix,\n\
                         regexp_extract({group_col}, '{regex}', 2) AS row,\n\
                         regexp_extract({group_col}, '{regex}', 3) AS col,\n\
                         bool_or(disabled) AS any_disabled,\n\
                         bool_or(NOT successful) AS any_failed,\n\
                         bool_or(NOT disabled AND {measured}) AS any_measured\n\
                     FROM images\n\
                     GROUP BY group_prefix, row, col\n\
                 ) img\n\
                 {agg_join}\
                 {cnt_join}\
                 {imv_join}\
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
                        filter.plate_size,
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

        let grouping = self.grouping_sql(&filter.grouping)?;
        let (group_col, regex) = (grouping.column, grouping.regex.as_str());

        let mut object_conditions = vec![
            self.object_on_plane_sql("", filter.plane.z_stack)?,
            format!("t_stack = {}", filter.plane.t_stack),
        ];
        if let ObjectClass::Global(id) = filter.object_class {
            object_conditions.push(class_filter_sql("object_class_id", &[id]));
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
        let value = if is_image_measurement(&filter.column) {
            "imm.m_0".to_string()
        } else {
            fill_zero_sql(
                "agg.value",
                &filter.column,
                &filter.aggregation,
                "agg.n_objects IS NULL",
                &self.measured_on_plane_sql("i", &filter.plane)?,
            )
        };
        // An image measurement: the image's own value (`imm`), not one over
        // its objects.
        let agg_value = match is_image_measurement(&filter.column) {
            true => "NULL".to_string(),
            false => format!("{agg_fn}({value_expr})"),
        };
        let imm_join =
            image_measurement_join(&[(&filter.column, &filter.aggregation)], &filter.plane);
        let sql = format!(
            "SELECT\n\
                regexp_extract(i.{group_col}, '{regex}', 4) AS idx,\n\
                i.image_rel_path,\n\
                i.image_name,\n\
                {value},\n\
                i.disabled,\n\
                NOT i.successful\n\
             FROM images i\n\
             LEFT JOIN (\n\
                 SELECT image_rel_path, COUNT(*) AS n_objects, {agg_value} AS value\n\
                 FROM objects\n\
                 WHERE {object_where}\n\
                 GROUP BY image_rel_path\n\
             ) agg ON agg.image_rel_path = i.image_rel_path\n\
             {imm_join}\
             WHERE regexp_extract(i.{group_col}, '{regex}', 1) = '{group_name}'\n\
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

        let grouping = self.grouping_sql(&filter.grouping)?;
        let (group_col, regex) = (grouping.column, grouping.regex.as_str());

        let mut object_conditions = vec![
            self.object_on_plane_sql("", filter.plane.z_stack)?,
            format!("t_stack = {}", filter.plane.t_stack),
        ];
        if let ObjectClass::Global(id) = filter.object_class {
            object_conditions.push(class_filter_sql("object_class_id", &[id]));
        }
        let object_where = object_conditions.join(" AND ");

        // Same shape as `get_group_by_well` (one row per field/image,
        // objects filtered+aggregated before the join, disabled images
        // shown - not dropped - and flagged via `i.disabled`), just without
        // the single-well filter - every well's fields in one query.
        let value = if is_image_measurement(&filter.column) {
            "imm.m_0".to_string()
        } else {
            fill_zero_sql(
                "agg.value",
                &filter.column,
                &filter.aggregation,
                "agg.n_objects IS NULL",
                &self.measured_on_plane_sql("i", &filter.plane)?,
            )
        };
        // An image measurement: the image's own value (`imm`), not one over
        // its objects.
        let agg_value = match is_image_measurement(&filter.column) {
            true => "NULL".to_string(),
            false => format!("{agg_fn}({value_expr})"),
        };
        let imm_join =
            image_measurement_join(&[(&filter.column, &filter.aggregation)], &filter.plane);
        let sql = format!(
            "SELECT\n\
                regexp_extract(i.{group_col}, '{regex}', 1) AS group_prefix,\n\
                regexp_extract(i.{group_col}, '{regex}', 4) AS idx,\n\
                i.image_rel_path,\n\
                i.image_name,\n\
                {value},\n\
                i.disabled,\n\
                NOT i.successful\n\
             FROM images i\n\
             LEFT JOIN (\n\
                 SELECT image_rel_path, COUNT(*) AS n_objects, {agg_value} AS value\n\
                 FROM objects\n\
                 WHERE {object_where}\n\
                 GROUP BY image_rel_path\n\
             ) agg ON agg.image_rel_path = i.image_rel_path\n\
             {imm_join}\
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
        let mut stats: Vec<(&Column, &Aggregation)> = Vec::new();
        for column in &filter.column {
            for aggregation in &filter.aggregation {
                let k = agg_value_cols.len();
                stats.push((column, aggregation));
                // An image measurement: the image's own value (`imm`).
                if is_image_measurement(column) {
                    agg_value_cols.push(format!("imm.m_{k}"));
                    value_exprs.push(format!("NULL AS value_{k}"));
                    continue;
                }
                agg_value_cols.push(fill_zero_sql(
                    &format!("agg.value_{k}"),
                    column,
                    aggregation,
                    "agg.n_objects IS NULL",
                    &measured,
                ));
                let (agg_fn, value_expr) = aggregate_sql(column, aggregation)?;
                value_exprs.push(format!("{agg_fn}({value_expr}) AS value_{k}"));
            }
        }
        let imm_join = image_measurement_join(&stats, &filter.plane);
        let n = value_exprs.len();
        let value_exprs_sql = value_exprs.join(",\n                ");
        let agg_value_cols = agg_value_cols.join(", ");

        let grouping = self.grouping_sql(&filter.grouping)?;
        let (group_col, regex) = (grouping.column, grouping.regex.as_str());

        let mut results = Vec::with_capacity(filter.object_class.len() * n);
        for object_class in &filter.object_class {
            let mut object_conditions = vec![
                self.object_on_plane_sql("", filter.plane.z_stack)?,
                format!("t_stack = {}", filter.plane.t_stack),
            ];
            if let ObjectClass::Global(id) = object_class {
                object_conditions.push(class_filter_sql("object_class_id", &[*id]));
            }
            let object_where = object_conditions.join(" AND ");

            // Same shape as `get_wells_for_plate` (one row per field/image,
            // objects filtered+aggregated before the join, disabled images
            // shown - not dropped - and flagged via `i.disabled`), batched
            // across every column x aggregation combo like
            // `get_group_by_plate_multi`.
            let sql = format!(
                "SELECT\n\
                    regexp_extract(i.{group_col}, '{regex}', 1) AS group_prefix,\n\
                    regexp_extract(i.{group_col}, '{regex}', 4) AS idx,\n\
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
                 {imm_join}\
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
            self.object_on_plane_sql("", filter.plane.z_stack)?,
            format!("t_stack = {}", filter.plane.t_stack),
            format!("image_rel_path = '{image_rel_path}'"),
        ];
        if let ObjectClass::Global(id) = filter.object_class {
            conditions.push(class_filter_sql("object_class_id", &[id]));
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

    /// Marks every cell of a row whose image is disabled (`row_locations`
    /// names each row's image) - an image row then looks like the rows of a
    /// disabled image's objects.
    pub(super) fn flag_disabled_image_rows(
        &self,
        rows: &mut [Vec<Cell>],
        row_locations: &[(String, Option<[u32; 4]>)],
    ) -> Result<(), InternalErrors> {
        let err = |e: duckdb::Error| InternalErrors::Io(e.to_string());
        let mut stmt = self
            .database
            .prepare("SELECT image_rel_path FROM images WHERE disabled")
            .map_err(err)?;
        let disabled: std::collections::HashSet<String> = stmt
            .query_map([], |row| row.get(0))
            .map_err(err)?
            .collect::<Result<_, _>>()
            .map_err(err)?;
        if disabled.is_empty() {
            return Ok(());
        }
        for (row, (rel_path, _)) in rows.iter_mut().zip(row_locations) {
            if disabled.contains(rel_path) {
                row.iter_mut().for_each(|cell| cell.disabled = true);
            }
        }
        Ok(())
    }
}

/// The plate sizes `PlateSize::Auto` chooses from (the standard formats),
/// smallest first - `best_matching_dimensions` relies on this order to find
/// the smallest one that fits.
pub(super) const AUTO_PLATE_SIZES: [PlateSize; 8] = [
    PlateSize::Plate2x3,
    PlateSize::Plate3x4,
    PlateSize::Plate4x6,
    PlateSize::Plate6x8,
    PlateSize::Plate8x12,
    PlateSize::Plate16x24,
    PlateSize::Plate32x48,
    PlateSize::Plate48x72,
];

/// The smallest standard plate size whose row/column count covers every well
/// this query actually found (`max_row`/`max_col`, both 0-based), as
/// `(rows, cols)`. Falls back to the largest known size if even that doesn't
/// fit (a plate bigger than any standard format, or a grouping extracting
/// something that isn't really a well id).
pub(crate) fn best_matching_dimensions(
    max_row: Option<usize>,
    max_col: Option<usize>,
) -> (usize, usize) {
    let needed_rows = max_row.map_or(1, |row| row + 1);
    let needed_cols = max_col.map_or(1, |col| col + 1);
    AUTO_PLATE_SIZES
        .into_iter()
        .filter_map(PlateSize::dimensions)
        .find(|(rows, cols)| *rows >= needed_rows && *cols >= needed_cols)
        .unwrap_or((48, 72))
}

/// The `(rows, cols)` grid for a well whose highest image number is
/// `max_idx`, for `WellSize` Auto: the smallest square-ish grid (as many or
/// one more columns than rows) holding that many images.
pub(super) fn auto_well_size(max_idx: u32) -> WellSize {
    let n = max_idx.max(1) as usize;
    let cols = (n as f64).sqrt().ceil() as usize;
    WellSize {
        rows: n.div_ceil(cols),
        cols,
    }
}

/// Parses a well's row letters ("A", "B", ..., "Z", "AA", "AB", ...) into a
/// 0-based row index, using the same bijective base-26 scheme spreadsheet
/// column letters use. `None` if `letters` isn't purely alphabetic (e.g. the
/// `grouping_regex` didn't actually match a well id).
pub(crate) fn row_letter_to_index(letters: &str) -> Option<usize> {
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
pub(crate) fn row_index_to_letter(index: usize) -> String {
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
pub(crate) fn col_number_to_index(digits: &str) -> Option<usize> {
    digits.parse::<usize>().ok()?.checked_sub(1)
}

/// Why an image behind a plate/well cell is marked: disabled by the user,
/// or failed (its analysis stopped with an error, so its objects are
/// incomplete). For a plate well, whether *any* of its images is; neither
/// kind contributes to a well's value.
#[derive(Debug, Clone, Copy, Default)]
pub(super) struct ImageFlags {
    pub(super) disabled: bool,
    pub(super) failed: bool,
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
pub(super) fn plate_groups_to_result(
    groups: Vec<(String, String, String, Option<f64>, ImageFlags)>,
    column: &Column,
    classes: &[Class],
    plate_size: PlateSize,
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
            let (rows, cols) = plate_size
                .dimensions()
                .unwrap_or_else(|| best_matching_dimensions(max_row, max_col));

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
pub(super) fn well_fields_to_result(
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
            // An empty image number (a pattern for one image per well) is
            // image 1.
            let field_idx = |idx: &str| {
                if idx.is_empty() {
                    Some(1)
                } else {
                    idx.parse::<u32>().ok()
                }
            };
            let well_size = well_size.unwrap_or_else(|| {
                auto_well_size(
                    fields
                        .iter()
                        .filter_map(|(idx, ..)| field_idx(idx))
                        .max()
                        .unwrap_or(1),
                )
            });
            let (rows, cols) = (well_size.rows, well_size.cols);

            // Every field gets a grid position regardless of whether it has
            // a value, same reasoning as `plate_groups_to_result`'s Heatmap
            // arm - a field genuinely has no objects is a different thing
            // from "no field at all sits here", and only the latter should
            // render as if the tile doesn't exist.
            let mut values: HashMap<usize, (Option<f64>, String, String, ImageFlags)> =
                HashMap::new();
            for (idx_str, image_rel_path, image_name, value, disabled) in &fields {
                let Some(idx) = field_idx(idx_str) else {
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
