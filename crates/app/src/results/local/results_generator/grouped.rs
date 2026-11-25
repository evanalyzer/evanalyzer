//! # grouped
//!
//! Grouped by image or by group: a row per image (or group) and class,
//! aggregated, also transposed.
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-11-22
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.

use super::*;

impl ResultsGenerator {
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
    /// per (image, class) — the Results List's rows per image, and the
    /// exports' `images` table (read block by block). No `grouping_regex` (unlike
    /// `get_group_by_plate`/`get_group_by_well`): grouped directly by each
    /// object's own `image_rel_path`, nothing derived from it.
    pub fn get_grouped_by_image(
        &self,
        filter: &GroupedByImageFilter,
    ) -> Result<DatabaseResult, InternalErrors> {
        self.get_grouped_by_image_of(filter, ImageSelection::of(&filter.images))
    }

    /// [`get_grouped_by_image`](Self::get_grouped_by_image) of `images`
    /// instead of `filter.images` (which is then ignored).
    pub(in crate::results::local) fn get_grouped_by_image_of(
        &self,
        filter: &GroupedByImageFilter,
        images: ImageSelection<'_>,
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
            self.object_on_plane_sql("", filter.plane.z_stack)?,
            format!("t_stack = {}", filter.plane.t_stack),
        ];
        let mut post = Vec::new();
        let mut image_conditions = vec!["true".to_string()];
        if images.matches_nothing() {
            return Ok(empty_result());
        }
        if let Some(condition) = images.objects_condition() {
            pre.push(condition);
        }
        if let Some(condition) = images.images_condition() {
            image_conditions.push(condition);
        }
        let selected_ids: Option<Vec<u32>> = filter.object_classes.as_ref().map(|wanted| {
            wanted
                .iter()
                .filter_map(|id| match id {
                    ObjectClass::Global(n) => Some(*n),
                    ObjectClass::Unset | ObjectClass::Local(_) => None,
                })
                .collect()
        });
        if let Some(ids) = &selected_ids {
            if ids.is_empty() {
                return Ok(empty_result());
            }
            pre.push(class_filter_sql("object_class_id", ids));
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
            row_classes.retain(|id| ObjectClass::Global(*id) != ObjectClass::BACKGROUND);
        }
        // Only the classes with objects: no row of zeros per empty class.
        if !filter.empty_classes {
            let with_objects = self.classes_with_objects()?;
            row_classes.retain(|id| with_objects.contains(id));
        }
        let measured = self.measured_on_plane_sql("images", &filter.plane)?;
        if let Some(grouping) = &filter.grouping {
            if filter.transpond_table {
                return self.get_grouped_by_group_transposed(
                    filter,
                    grouping,
                    &stats,
                    &pre,
                    &post,
                    &image_conditions,
                    &row_classes,
                    &classes,
                );
            }
            return self.get_grouped_by_group(
                filter,
                grouping,
                &stats,
                &pre,
                &post,
                &image_conditions,
                &row_classes,
                &classes,
            );
        }
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
        let sort = stat_sort(filter, &stats);
        if sort.is_none()
            && let Some(cursor) = &filter.page.after
        {
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
                if is_image_measurement(column) {
                    return format!("imm.m_{i} AS v_{i}");
                }
                let value = fill_zero_sql(
                    &format!("agg.value_{i}"),
                    column,
                    aggregation,
                    "agg.n_objects IS NULL",
                    "img.measured",
                );
                format!("{value} AS v_{i}")
            })
            .collect::<Vec<_>>()
            .join(",\n                ");
        let (imm_cte, imm_join) = match image_measurements_sql(&stats, &filter.plane) {
            Some(sql) => (
                format!("), imm AS (\n{sql}\n"),
                "LEFT JOIN imm ON imm.image_rel_path = keys.image_rel_path\n",
            ),
            None => (String::new(), ""),
        };
        // Every analysed image x every row class, plus whatever (image,
        // class) pairs actually have objects - the latter also covers an
        // image missing from `images` (a run that crashed before
        // finalizing it) or a class missing from `classes`.
        let measured_keys = if row_classes.is_empty() || !filter.empty_classes {
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
             {imm_cte})\n\
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
             {imm_join}\
             WHERE {key_where}\n\
             ORDER BY {order}keys.image_rel_path, keys.class_id\n\
             LIMIT {limit}{offset}",
            order = sort.as_ref().map_or(String::new(), |(i, desc)| format!(
                "v_{i} {} NULLS LAST, ",
                if *desc { "DESC" } else { "ASC" }
            )),
            offset = match sort {
                Some(_) => format!(" OFFSET {}", sort_offset(&filter.page)),
                None => String::new(),
            },
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
        // A row is an image: clicking it shows the whole image.
        let row_locations: Vec<(String, Option<[u32; 4]>)> = groups
            .iter()
            .map(|(rel_path, ..)| (rel_path.clone(), None))
            .collect();
        let source_object_count = groups.len();
        let rows: Vec<Vec<Cell>> = groups
            .into_iter()
            .map(|(image_rel_path, image_name, class_id, failed, values)| {
                // Lets the GUI navigate straight to the source image, same
                // as any other image-bearing search key in this file.
                let search_key = Some((image_name.clone(), image_rel_path));
                let object_class = ObjectClass::Global(class_id);
                let label = class_display_label(object_class, &classes);
                let color = classes
                    .iter()
                    .find(|class| class.global() == object_class)
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

        let mut rows = rows;
        self.flag_disabled_image_rows(&mut rows, &row_locations)?;
        Ok(DatabaseResult {
            column_names,
            row_names,
            rows,
            min,
            max,
            source_object_count,
            row_locations,
        })
    }

    /// The classes that any object has (cached: the objects of a results
    /// database don't change).
    pub(super) fn classes_with_objects(&self) -> Result<Vec<u32>, InternalErrors> {
        if let Some(cached) = self.classes_with_objects_cache.borrow().as_ref() {
            return Ok(cached.clone());
        }
        let err = |e: duckdb::Error| InternalErrors::Io(e.to_string());
        let mut stmt = self
            .database
            .prepare(
                "SELECT DISTINCT class_id FROM \
                 (SELECT UNNEST(object_class_id) AS class_id FROM objects) \
                 WHERE class_id IS NOT NULL ORDER BY class_id",
            )
            .map_err(err)?;
        let ids: Vec<u32> = stmt
            .query_map([], |row| row.get::<_, u32>(0))
            .map_err(err)?
            .collect::<Result<_, _>>()
            .map_err(err)?;
        *self.classes_with_objects_cache.borrow_mut() = Some(ids.clone());
        Ok(ids)
    }

    /// `get_grouped_by_image` with a `grouping`: a row per (group, class) -
    /// the group a well (the grouping's pattern on the image names) or a
    /// folder. The values over the objects of the group's images; Count over
    /// the images' object counts (each analysed image, also one without
    /// objects - as the plate view counts: "mean" is the objects per image,
    /// "sum" all of them). Left-out and failed images add nothing, their
    /// group is marked. Images the pattern doesn't fit form the group "No
    /// group" (for folders: "Top folder", the images right in the image
    /// folder).
    #[allow(clippy::too_many_arguments)]
    pub(super) fn get_grouped_by_group(
        &self,
        filter: &GroupedByImageFilter,
        grouping: &Grouping,
        stats: &[(&Column, &Aggregation)],
        pre: &[String],
        post: &[String],
        image_conditions: &[String],
        row_classes: &[u32],
        classes: &[Class],
    ) -> Result<DatabaseResult, InternalErrors> {
        let err = |e: duckdb::Error| InternalErrors::Io(e.to_string());
        let group_sql = self.grouping_sql(grouping)?;
        let (group_col, regex) = (group_sql.column, group_sql.regex.as_str());
        let group_name = match grouping {
            Grouping::Folder => "folder",
            _ => "well",
        };
        let mut column_names = vec![group_name.to_string(), "class".to_string()];
        for (column, aggregation) in stats {
            // Count: over the images' counts - per image.
            if matches!(column, Column::Count) {
                column_names.push(format!(
                    "Count per image ({})",
                    aggregation_sql_fn(aggregation)
                ));
                continue;
            }
            let (agg_fn, _) = aggregate_sql(column, aggregation)?;
            column_names.push(format!("{} ({agg_fn})", column.display_label(classes)));
        }
        let measured = self.measured_on_plane_sql("images", &filter.plane)?;
        let post_where = if post.is_empty() {
            "true".to_string()
        } else {
            post.join(" AND ")
        };
        // Per statistic: over the objects (`agg`), the per-image counts
        // (`counts`) or the images' own values (`imm`).
        let mut object_values = Vec::new();
        let mut count_values = Vec::new();
        let mut image_values = Vec::new();
        let mut select_values = Vec::new();
        for (i, (column, aggregation)) in stats.iter().enumerate() {
            let agg_fn = aggregation_sql_fn(aggregation);
            if matches!(column, Column::Count) {
                count_values.push(format!(
                    "{agg_fn}(CAST(COALESCE(p.n, 0) AS DOUBLE)) AS c_{i}"
                ));
                select_values.push(format!("counts.c_{i}"));
            } else if let Column::Measurement(m) = column
                && !m.per_object
            {
                image_values.push(format!(
                    "{agg_fn}(m.value) FILTER (WHERE m.key = '{}') AS m_{i}",
                    m.key.replace('\'', "''")
                ));
                select_values.push(format!("imm.m_{i}"));
            } else {
                let (agg_fn, value_expr) = aggregate_sql(column, aggregation)?;
                object_values.push(format!("{agg_fn}({value_expr}) AS value_{i}"));
                select_values.push(fill_zero_sql(
                    &format!("agg.value_{i}"),
                    column,
                    aggregation,
                    "agg.n_objects IS NULL",
                    "flags.any_measured",
                ));
            }
        }
        let object_values = object_values
            .iter()
            .map(|v| format!(", {v}"))
            .collect::<String>();
        let counts_cte = if count_values.is_empty() {
            String::new()
        } else {
            format!(
                ", per_img AS (\n\
                     SELECT image_rel_path, class_id, COUNT(*) AS n FROM u\n\
                     WHERE {post_where} GROUP BY image_rel_path, class_id\n\
                 ), counts AS (\n\
                     SELECT k.grp, k.class_id, {}\n\
                     FROM keys k\n\
                     JOIN img ON img.grp = k.grp AND img.counted\n\
                     LEFT JOIN per_img p ON p.image_rel_path = img.image_rel_path\n\
                         AND p.class_id = k.class_id\n\
                     GROUP BY k.grp, k.class_id\n\
                 )",
                count_values.join(", ")
            )
        };
        let counts_join = if count_values.is_empty() {
            ""
        } else {
            "LEFT JOIN counts ON counts.grp = keys.grp AND counts.class_id = keys.class_id\n"
        };
        let imm_cte = if image_values.is_empty() {
            String::new()
        } else {
            format!(
                ", imm AS (\n\
                     SELECT img.grp, {}\n\
                     FROM measurements m JOIN img ON img.image_rel_path = m.image_rel_path\n\
                     WHERE m.region = 'image' AND (m.z_stack = {z} OR m.z_stack IS NULL)\n\
                         AND m.t_stack = {t} AND img.counted\n\
                     GROUP BY img.grp\n\
                 )",
                image_values.join(", "),
                z = filter.plane.z_stack,
                t = filter.plane.t_stack,
            )
        };
        let imm_join = if image_values.is_empty() {
            ""
        } else {
            "LEFT JOIN imm ON imm.grp = keys.grp\n"
        };
        // Every group x every row class (classes without objects too, when
        // asked), plus the (group, class) pairs with objects.
        let all_keys = if row_classes.is_empty() || !filter.empty_classes {
            String::new()
        } else {
            format!(
                "SELECT DISTINCT grp, class_id\n\
                 FROM img, (SELECT UNNEST({}) AS class_id)\n\
                 WHERE img.counted\n\
                 UNION\n",
                sql_int_array_literal(row_classes)
            )
        };
        // In plate order: by row (A before AA), column number, then name -
        // the cursor carries the last row's place in it.
        let sort_key = "length(groups.sort_row), groups.sort_row, groups.sort_col, keys.grp, \
                        keys.class_id";
        let mut key_where = "true".to_string();
        let sort = stat_sort(filter, stats);
        if sort.is_none()
            && let Some(cursor) = &filter.page.after
        {
            let parts: Vec<&str> = cursor.split('\u{1}').collect();
            let text = |i: usize| parts.get(i).copied().unwrap_or("").replace('\'', "''");
            let number = |i: usize| {
                parts
                    .get(i)
                    .and_then(|p| p.parse::<i64>().ok())
                    .unwrap_or(-1)
            };
            key_where = format!(
                "({sort_key}) > ({}, '{}', {}, '{}', {})",
                text(2).chars().count(),
                text(2),
                number(3),
                text(0),
                number(1)
            );
        }
        let limit = filter.page.limit.max(0);
        let sql = format!(
            "WITH img AS (\n\
                 SELECT image_rel_path,\n\
                     COALESCE(regexp_extract({group_col}, '{regex}', 1), '') AS grp,\n\
                     COALESCE(regexp_extract({group_col}, '{regex}', 2), '') AS sort_row,\n\
                     COALESCE(TRY_CAST(regexp_extract({group_col}, '{regex}', 3) AS BIGINT), -1)\n\
                         AS sort_col,\n\
                     disabled, NOT successful AS failed,\n\
                     NOT disabled AND successful AND {measured} AS counted\n\
                 FROM images\n\
                 WHERE {image_where}\n\
             ), u AS (\n\
                 {per_class}\n\
             ), agg AS (\n\
                 SELECT img.grp, u.class_id, COUNT(*) AS n_objects{object_values}\n\
                 FROM u JOIN img ON img.image_rel_path = u.image_rel_path\n\
                 WHERE {post_where} AND NOT img.disabled AND NOT img.failed\n\
                 GROUP BY img.grp, u.class_id\n\
             ), keys AS (\n\
                 {all_keys}\
                 SELECT grp, class_id FROM agg\n\
             ), groups AS (\n\
                 SELECT grp, MIN(sort_row) AS sort_row, MIN(sort_col) AS sort_col\n\
                 FROM img GROUP BY grp\n\
             ), flags AS (\n\
                 SELECT grp, bool_or(disabled) AS any_disabled, bool_or(failed) AS any_failed,\n\
                     bool_or(counted) AS any_measured\n\
                 FROM img GROUP BY grp\n\
             ){counts_cte}{imm_cte}\n\
             SELECT keys.grp, keys.class_id, COALESCE(flags.any_disabled, false),\n\
                 COALESCE(flags.any_failed, false), groups.sort_row, groups.sort_col{select}\n\
             FROM keys\n\
             JOIN groups ON groups.grp = keys.grp\n\
             LEFT JOIN agg ON agg.grp = keys.grp AND agg.class_id = keys.class_id\n\
             LEFT JOIN flags ON flags.grp = keys.grp\n\
             {counts_join}{imm_join}\
             WHERE {key_where}\n\
             ORDER BY {order}{sort_key}\n\
             LIMIT {limit}{offset}",
            order = sort.as_ref().map_or(String::new(), |(i, desc)| format!(
                "v_{i} {} NULLS LAST, ",
                if *desc { "DESC" } else { "ASC" }
            )),
            offset = match sort {
                Some(_) => format!(" OFFSET {}", sort_offset(&filter.page)),
                None => String::new(),
            },
            image_where = image_conditions.join(" AND "),
            per_class = objects_per_class_sql(&pre.join(" AND ")),
            select = select_values
                .iter()
                .enumerate()
                .map(|(i, v)| format!(",\n                 {v} AS v_{i}"))
                .collect::<String>(),
        );
        let mut stmt = self.database.prepare(&sql).map_err(err)?;
        let n = stats.len();
        type GroupRow = (String, u32, bool, bool, (String, i64), Vec<Option<f64>>);
        let groups: Vec<GroupRow> = stmt
            .query_map([], |row| {
                let mut values = Vec::with_capacity(n);
                for i in 0..n {
                    values.push(row.get::<_, Option<f64>>(6 + i)?);
                }
                Ok((
                    row.get(0)?,
                    row.get(1)?,
                    row.get(2)?,
                    row.get(3)?,
                    (row.get(4)?, row.get(5)?),
                    values,
                ))
            })
            .map_err(err)?
            .collect::<Result<Vec<_>, _>>()
            .map_err(err)?;

        let (min, max) = value_range(groups.iter().flat_map(|(.., values)| values.iter()));
        // The next page's cursor: the row's group, class and plate place.
        let row_names = groups
            .iter()
            .map(|(group, class_id, _, _, (sort_row, sort_col), _)| {
                format!("{group}\u{1}{class_id}\u{1}{sort_row}\u{1}{sort_col}")
            })
            .collect();
        // A row is several images: no one image to show.
        let row_locations = groups.iter().map(|_| (String::new(), None)).collect();
        let source_object_count = groups.len();
        let rows = groups
            .into_iter()
            .map(|(group, class_id, any_disabled, failed, _, values)| {
                let object_class = ObjectClass::Global(class_id);
                let label = class_display_label(object_class, classes);
                let color = classes
                    .iter()
                    .find(|class| class.global() == object_class)
                    .map(|class| class.color)
                    .unwrap_or(0);
                // Images the pattern doesn't fit; for folders: those right in
                // the image folder.
                let name = match (group.is_empty(), grouping) {
                    (false, _) => group,
                    (true, Grouping::Folder) => "Top folder".to_string(),
                    (true, _) => "No group".to_string(),
                };
                let mut cells = vec![
                    Cell {
                        value: CellValue::String(name),
                        any_disabled,
                        failed,
                        ..plain_cell()
                    },
                    Cell {
                        value: CellValue::Class((label, color)),
                        bg_color: color,
                        any_disabled,
                        failed,
                        ..plain_cell()
                    },
                ];
                cells.extend(values.into_iter().map(|value| Cell {
                    value: value.map_or(CellValue::Empty, |v| CellValue::Float(v as f32)),
                    any_disabled,
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
            row_locations,
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
    pub(super) fn get_grouped_by_image_transposed(
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
            let class_label = class_display_label(ObjectClass::Global(*class_id), classes);
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
        let values = stats
            .iter()
            .enumerate()
            .map(|(i, (column, _))| match is_image_measurement(column) {
                true => format!("imm.m_{i}"),
                false => format!("agg.value_{i}"),
            })
            .collect::<Vec<_>>()
            .join(", ");
        let (imm_cte, imm_join) = match image_measurements_sql(stats, &filter.plane) {
            Some(sql) => (
                format!("), imm AS (\n{sql}\n"),
                "LEFT JOIN imm ON imm.image_rel_path = page.image_rel_path\n",
            ),
            None => (String::new(), ""),
        };
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
             {imm_cte})\n\
             SELECT page.image_rel_path, COALESCE(img.image_name, agg.image_name),\n\
                 COALESCE(img.failed, false), COALESCE(img.measured, false),\n\
                 agg.class_id, {values}\n\
             FROM page\n\
             LEFT JOIN img ON img.image_rel_path = page.image_rel_path\n\
             LEFT JOIN agg ON agg.image_rel_path = page.image_rel_path\n\
             {imm_join}\
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
            let values = &mut current.as_mut().expect("set above").3;
            // An image measurement is the image's: in every class's block,
            // also of an image without objects (its row has no class).
            for (i, (column, _)) in stats.iter().enumerate() {
                if is_image_measurement(column) {
                    let value: Option<f64> = row.get(5 + i).map_err(err)?;
                    for block in 0..blocks.len() {
                        values[block * n + i] = value;
                    }
                }
            }
            let class_id: Option<u32> = row.get(4).map_err(err)?;
            let Some(block) = class_id.and_then(|id| blocks.iter().position(|b| *b == id)) else {
                continue;
            };
            seen_classes[block] = true;
            for (i, (column, _)) in stats.iter().enumerate() {
                if !is_image_measurement(column) {
                    values[block * n + i] = row.get(5 + i).map_err(err)?;
                }
            }
        }
        if let Some(done) = current.take() {
            groups.push(finish(done, &seen_classes));
        }

        let (min, max) = value_range(groups.iter().flat_map(|(_, _, _, values)| values.iter()));
        let row_names = groups.iter().map(|(path, ..)| path.clone()).collect();
        // A row is an image: clicking it shows the whole image.
        let row_locations: Vec<(String, Option<[u32; 4]>)> = groups
            .iter()
            .map(|(path, ..)| (path.clone(), None))
            .collect();
        let source_object_count = groups.len();
        let rows: Vec<Vec<Cell>> = groups
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
        let mut rows = rows;
        self.flag_disabled_image_rows(&mut rows, &row_locations)?;
        Ok(DatabaseResult {
            column_names,
            row_names,
            rows,
            min,
            max,
            source_object_count,
            row_locations,
        })
    }

    /// `get_grouped_by_group` with `transpond_table`: one row per well (or
    /// folder), and per class of `row_classes` one column per statistic -
    /// the (group, class) rows of the normal view placed side by side.
    ///
    /// Runs the normal query for up to `limit` groups' worth of rows (a
    /// group has at most a row per class) and places them here; on a full
    /// page the last group may lack classes still to come, so it is left
    /// for the next page (unless it's the only one). Pages over groups
    /// (`row_names` holds each group's last (group, class) cursor); a full
    /// page reports `limit` as `source_object_count`, so a caller asks on.
    #[allow(clippy::too_many_arguments)]
    fn get_grouped_by_group_transposed(
        &self,
        filter: &GroupedByImageFilter,
        grouping: &Grouping,
        stats: &[(&Column, &Aggregation)],
        pre: &[String],
        post: &[String],
        image_conditions: &[String],
        row_classes: &[u32],
        classes: &[Class],
    ) -> Result<DatabaseResult, InternalErrors> {
        let blocks = row_classes;
        let group_name = match grouping {
            Grouping::Folder => "folder",
            _ => "well",
        };
        let mut column_names = vec![group_name.to_string()];
        for class_id in blocks {
            let class_label = class_display_label(ObjectClass::Global(*class_id), classes);
            for (column, aggregation) in stats {
                let name = if matches!(column, Column::Count) {
                    format!("Count per image ({})", aggregation_sql_fn(aggregation))
                } else {
                    let (agg_fn, _) = aggregate_sql(column, aggregation)?;
                    format!("{} ({agg_fn})", column.display_label(classes))
                };
                column_names.push(format!("{name} ({class_label})"));
            }
        }
        if blocks.is_empty() || stats.is_empty() {
            return Ok(empty_database_result(column_names));
        }
        let limit = filter.page.limit.max(0);
        let rows_limit = (limit as i64)
            .saturating_mul(blocks.len() as i64)
            .min(i64::from(i32::MAX)) as i32;
        let by_class = GroupedByImageFilter {
            page: Pagination {
                limit: rows_limit,
                after: filter.page.after.clone(),
            },
            transpond_table: false,
            sort: None,
            ..filter.clone()
        };
        let page = self.get_grouped_by_group(
            &by_class,
            grouping,
            stats,
            pre,
            post,
            image_conditions,
            row_classes,
            classes,
        )?;
        let full = page.rows.len() as i64 >= i64::from(rows_limit);
        let n = stats.len();
        // Per group: its key, name cell, last cursor, values per block.
        let mut groups: Vec<(String, Cell, String, Vec<Cell>)> = Vec::new();
        for (row, cursor) in page.rows.into_iter().zip(page.row_names) {
            let mut parts = cursor.split('\u{1}');
            let key = parts.next().unwrap_or("").to_string();
            let class_id: Option<u32> = parts.next().and_then(|p| p.parse().ok());
            let mut cells = row.into_iter();
            let name = cells.next().unwrap_or_else(plain_cell);
            let _class = cells.next();
            if groups.last().is_none_or(|g| g.0 != key) {
                let (any_disabled, failed) = (name.any_disabled, name.failed);
                let empty = (0..blocks.len() * n)
                    .map(|_| Cell {
                        value: CellValue::Empty,
                        any_disabled,
                        failed,
                        ..plain_cell()
                    })
                    .collect();
                groups.push((key, name, String::new(), empty));
            }
            let group = groups.last_mut().expect("pushed above");
            group.2 = cursor;
            let Some(block) = class_id.and_then(|id| blocks.iter().position(|b| *b == id)) else {
                continue;
            };
            for (i, cell) in cells.take(n).enumerate() {
                group.3[block * n + i] = cell;
            }
        }
        if full && groups.len() > 1 {
            groups.pop();
        }
        let source_object_count = if full { limit as usize } else { groups.len() };
        let row_names = groups.iter().map(|g| g.2.clone()).collect();
        let row_locations = groups.iter().map(|_| (String::new(), None)).collect();
        let rows: Vec<Vec<Cell>> = groups
            .into_iter()
            .map(|(_, name, _, values)| std::iter::once(name).chain(values).collect())
            .collect();
        let values: Vec<Option<f64>> = rows
            .iter()
            .flat_map(|row| row.iter().skip(1))
            .map(|cell| match cell.value {
                CellValue::Float(v) => Some(f64::from(v)),
                _ => None,
            })
            .collect();
        let (min, max) = value_range(values.iter());
        Ok(DatabaseResult {
            column_names,
            row_names,
            rows,
            min,
            max,
            source_object_count,
            row_locations,
        })
    }
}
