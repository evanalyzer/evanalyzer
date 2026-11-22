//! # sql
//!
//! SQL the views share: class and image filters, image measurements,
//! aggregates, value expressions, list literals.
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-11-22
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.

use super::*;

/// Display label for a `coloc_partner_ids`/`object_class_name`-adjacent class,
/// e.g. for a coloc-detail column header — the class's registered name, or
/// `"class {n}"` if `n` isn't (or no longer is) a recognized id.
pub(crate) fn class_display_label(class: ObjectClass, classes: &[Class]) -> String {
    match class {
        ObjectClass::Global(n) => classes
            .iter()
            .find(|c| c.id == n)
            .map(|c| c.name.clone())
            .unwrap_or_else(|| format!("class {n}")),
        ObjectClass::Unset | ObjectClass::Local(_) => "unset".to_string(),
    }
}

/// The classes a transposed view puts side by side, in id order: the
/// selected ones (`selected`, already validated), or every registered class.
pub(in crate::results::local) fn class_blocks(
    selected: Option<&[u32]>,
    classes: &[Class],
) -> Vec<u32> {
    let mut ids: Vec<u32> = match selected {
        Some(ids) => ids.to_vec(),
        None => classes.iter().map(|class| class.id).collect(),
    };
    ids.sort_unstable();
    ids.dedup();
    ids
}

/// The objects matching `conditions`, one row per (object, class) - every
/// column of `objects` plus the unpacked `class_id` (none for an object
/// without a class).
///
/// Unpacked in the `SELECT` list, not as `FROM objects, UNNEST(...)`: with
/// the latter, any later condition on `class_id` (even in `HAVING`) made
/// DuckDB pick a plan taking ~1 s and ~2.8 GB instead of ~24 ms and ~90 MB
/// on a 3.8M-object file.
pub(in crate::results::local) fn objects_per_class_sql(conditions: &str) -> String {
    format!(
        "SELECT *, UNNEST(object_class_id) AS class_id \
         FROM objects WHERE {conditions}"
    )
}

/// SQL condition that the object's class list `column` (its
/// `object_class_id`) contains any of `ids` - one `list_contains` per id
/// (~3 ms vs ~20 ms for a single `list_has_any` on a 3.8M-object file).
pub(in crate::results::local) fn class_filter_sql(column: &str, ids: &[u32]) -> String {
    let any = ids
        .iter()
        .map(|id| format!("list_contains({column}, {id})"))
        .collect::<Vec<_>>()
        .join(" OR ");
    format!("({any})")
}

/// The objects grouped per (image, class), one `value_{i}` per statistic
/// in `stats` plus `n_objects` - the first stage of both per-image views.
///
/// `object_class_id` is a per-object list (multi-class objects exist),
/// unpacked into one `class_id` per (object, class) pair before grouping;
/// an object without any class contributes no row. `pre` filters the
/// objects before that, `post` the (object, class) rows (conditions on
/// `class_id`).
pub(super) fn per_image_class_sql(
    stats: &[(&Column, &Aggregation)],
    pre: &[String],
    post: &[String],
) -> Result<String, InternalErrors> {
    let mut values = Vec::with_capacity(stats.len());
    for (i, (column, aggregation)) in stats.iter().enumerate() {
        // An image's own value comes from `image_measurements_sql`.
        if is_image_measurement(column) {
            values.push(format!("NULL AS value_{i}"));
            continue;
        }
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
         FROM ({per_class}) AS u\n\
         WHERE {post}\n\
         GROUP BY image_rel_path, class_id",
        values = values.join(", "),
        per_class = objects_per_class_sql(&pre.join(" AND ")),
    ))
}

/// Whether `column` is a value per image plane (not per object).
pub(super) fn is_image_measurement(column: &Column) -> bool {
    column.is_image_measurement()
}

/// The image measurements among `stats` per image of `plane` - one `m_{i}`
/// per such statistic (its aggregation over the image's value: the value
/// itself, no spread) - for the per-image views to join as `imm`. Also an
/// image without objects has its value. `None` without image measurements.
pub(super) fn image_measurements_sql(
    stats: &[(&Column, &Aggregation)],
    plane: &PlaneFilter,
) -> Option<String> {
    let values: Vec<String> = stats
        .iter()
        .enumerate()
        .filter_map(|(i, (column, aggregation))| match column {
            Column::Measurement(m) if !m.per_object => Some(format!(
                "{}(value) FILTER (WHERE key = '{}') AS m_{i}",
                aggregation_sql_fn(aggregation),
                m.key.replace('\'', "''")
            )),
            _ => None,
        })
        .collect();
    (!values.is_empty()).then(|| {
        format!(
            "SELECT image_rel_path, {}\n\
             FROM measurements\n\
             WHERE region = 'image' AND (z_stack = {} OR z_stack IS NULL) AND t_stack = {}\n\
             GROUP BY image_rel_path",
            values.join(", "),
            plane.z_stack,
            plane.t_stack
        )
    })
}

/// For the views with a row per image (`images i`): the join adding the
/// image measurements among `stats` as `imm.m_{i}` - empty without any.
pub(super) fn image_measurement_join(
    stats: &[(&Column, &Aggregation)],
    plane: &PlaneFilter,
) -> String {
    match image_measurements_sql(stats, plane) {
        Some(sql) => {
            format!("LEFT JOIN (\n{sql}\n) imm ON imm.image_rel_path = i.image_rel_path\n")
        }
        None => String::new(),
    }
}

/// Comma-joined integers for a SQL `IN (...)` list.
pub(super) fn sql_u32_list(values: &[u32]) -> String {
    values
        .iter()
        .map(|v| v.to_string())
        .collect::<Vec<_>>()
        .join(", ")
}

/// The images a list or grouped-by-image query reads.
#[derive(Clone, Copy, Debug)]
pub(in crate::results::local) enum ImageSelection<'a> {
    /// Every image.
    All,
    /// Every image that isn't disabled - an export of all images: a
    /// condition on the `images` table, not a list of every path parsed
    /// with each page (100,000 images: ~4 MB of SQL per page).
    Enabled,
    /// These images (`image_rel_path`s); none matches nothing.
    Listed(&'a [String]),
    /// The enabled images from `first` to `last` (`image_rel_path`s, in
    /// the database's order, both included) - a block of an export's
    /// images: the objects outside it are filtered out before they are
    /// aggregated, and rows of the same file stay together (zone maps).
    EnabledRange { first: &'a str, last: &'a str },
}

impl<'a> ImageSelection<'a> {
    /// A filter's `images`: `None` is every image.
    pub(in crate::results::local) fn of(images: &'a Option<Vec<String>>) -> Self {
        match images {
            Some(rel_paths) => Self::Listed(rel_paths),
            None => Self::All,
        }
    }

    pub(in crate::results::local) fn matches_nothing(&self) -> bool {
        matches!(self, Self::Listed(rel_paths) if rel_paths.is_empty())
    }

    /// The condition on the objects' `image_rel_path`, if any.
    pub(super) fn objects_condition(&self) -> Option<String> {
        match self {
            Self::All => None,
            Self::Enabled => Some(
                "image_rel_path IN (SELECT image_rel_path FROM images WHERE NOT disabled)"
                    .to_string(),
            ),
            Self::Listed(rel_paths) => Some(format!(
                "image_rel_path IN ({})",
                sql_string_in_list(rel_paths)
            )),
            Self::EnabledRange { .. } => Some(format!(
                "{} AND {}",
                self.range_condition(),
                Self::Enabled.objects_condition()?
            )),
        }
    }

    /// `image_rel_path BETWEEN first AND last` of an `EnabledRange`.
    pub(super) fn range_condition(&self) -> String {
        match self {
            Self::EnabledRange { first, last } => format!(
                "image_rel_path BETWEEN '{}' AND '{}'",
                first.replace('\'', "''"),
                last.replace('\'', "''")
            ),
            _ => "true".to_string(),
        }
    }

    /// The condition on the `images` table itself, if any.
    pub(super) fn images_condition(&self) -> Option<String> {
        match self {
            Self::All => None,
            Self::Enabled => Some("NOT disabled".to_string()),
            Self::Listed(_) => self.objects_condition(),
            Self::EnabledRange { .. } => {
                Some(format!("NOT disabled AND {}", self.range_condition()))
            }
        }
    }
}

/// Escapes and comma-joins string literals for a SQL `IN (...)` list.
///
/// `pub(super)`: also used by `results_charts.rs`'s boxplot query, which
/// needs raw SQL access this crate keeps otherwise private to this module.
pub(in crate::results::local) fn sql_string_in_list(values: &[String]) -> String {
    values
        .iter()
        .map(|v| format!("'{}'", v.replace('\'', "''")))
        .collect::<Vec<_>>()
        .join(", ")
}

/// Comma-joins integers into a DuckDB list literal, e.g. `[1, 2]`, for
/// `list_has_any(...)`. `pub(super)`: see `sql_string_in_list`.
pub(in crate::results::local) fn sql_int_array_literal(values: &[u32]) -> String {
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
pub(super) fn extract_int_list(value: Value) -> Vec<i32> {
    match value {
        Value::List(items) | Value::Array(items) => items
            .into_iter()
            .filter_map(|item| match item {
                Value::Int(v) => Some(v),
                _ => None,
            })
            .collect(),
        _ => vec![],
    }
}

/// A `DOUBLE[]` value, keeping each NULL element as `None` so positions
/// (= channels) stay aligned.
pub(super) fn extract_f64_list(value: Value) -> Vec<Option<f64>> {
    match value {
        Value::List(items) | Value::Array(items) => items
            .into_iter()
            .map(|item| match item {
                Value::Double(v) => Some(v),
                _ => None,
            })
            .collect(),
        _ => vec![],
    }
}

pub(super) fn extract_string_list(value: Value) -> Vec<String> {
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

pub(in crate::results::local) fn column_aggregate_expr(
    column: &Column,
) -> Result<String, InternalErrors> {
    Ok(match column {
        Column::AreaSizePx => "area_px".to_string(),
        Column::AreaSizeNm => "area_nm2".to_string(),
        Column::VolumePx => "volume_px".to_string(),
        Column::VolumeNm => "volume_nm3".to_string(),
        Column::SurfaceAreaNm => "surface_area_nm2".to_string(),
        Column::PerimeterPx => "perimeter_px".to_string(),
        Column::PerimeterNm => "perimeter_nm".to_string(),
        Column::Circularity => "circularity".to_string(),
        Column::Solidity => "solidity".to_string(),
        Column::Eccentricity => "eccentricity".to_string(),
        Column::MaxFeret => "feret_diameter_px".to_string(),
        Column::MinFeret => "min_feret_px".to_string(),
        Column::MajorAxis => "major_axis_px".to_string(),
        Column::MinorAxis => "minor_axis_px".to_string(),
        // The diameter of a disk of the object's area.
        Column::EquivalentDiameter => "sqrt(4.0 * area_px / pi())".to_string(),
        // The area over the bounding box's.
        Column::Extent => "area_px / ((bbox_xmax_px - bbox_xmin_px + 1) \
             * (bbox_ymax_px - bbox_ymin_px + 1))::DOUBLE"
            .to_string(),
        Column::ColocCount(class) => unless_own_class(
            std::slice::from_ref(class),
            format!(
                "COALESCE(len(coloc_partner_ids[{}]), 0)",
                coloc_class_key(*class)
            ),
        ),
        Column::Coloc(c) => unless_own_class(&c.with, coloc_value_expr(c)),
        // Gray values, like the list view; NULL (left out of every
        // aggregate) for a channel the object wasn't measured in.
        Column::IntensitySum(channel) => format!("intensity_sum_gray[{}]", channel + 1),
        Column::IntensityAvg(channel) => format!("intensity_mean_gray[{}]", channel + 1),
        Column::IntensityMin(channel) => format!("intensity_min_gray[{}]", channel + 1),
        Column::IntensityMax(channel) => format!("intensity_max_gray[{}]", channel + 1),
        Column::IntensityStd(channel) => format!("intensity_std_gray[{}]", channel + 1),
        Column::IntensityMedian(channel) => format!("intensity_median_gray[{}]", channel + 1),
        Column::IntensityLowerQuartile(channel) => {
            format!("intensity_lower_quartile_gray[{}]", channel + 1)
        }
        Column::IntensityUpperQuartile(channel) => {
            format!("intensity_upper_quartile_gray[{}]", channel + 1)
        }
        Column::IntensityMad(channel) => format!("intensity_mad_gray[{}]", channel + 1),
        Column::Measurement(m) => measurement_value_expr(m),
        Column::Children(c) => children_value_expr(c)?,
        Column::Parent(ParentColumn {
            parent_class,
            value: ParentValue::Of(column),
        }) => format!(
            "(SELECT pv.v FROM (SELECT object_id AS oid, {} AS v FROM objects) pv \
             WHERE pv.oid = parent_ids[{}])",
            column_aggregate_expr(column)?,
            coloc_class_key(*parent_class)
        ),
        Column::Parent(ParentColumn {
            value: ParentValue::Id,
            ..
        }) => {
            return Err(InternalErrors::InvalidArgument(
                "a parent's id cannot be aggregated".into(),
            ));
        }
        // Handled by `aggregate_sql` before this function is ever called
        // with `Column::Count` — `COUNT(*)` doesn't fit the "aggregate
        // function wraps a per-row scalar expression" shape every other
        // arm here does, since it counts rows rather than reading a column
        // off them. Kept here (rather than left unreachable) only so this
        // match stays exhaustive.
        Column::Count
        | Column::ObjectId
        | Column::TrackId
        | Column::ImageName
        | Column::ObjectClass
        | Column::MeasurementObject(_) => {
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

/// A [`Column::Children`] value for the object of the row: the number of
/// its children of the class, or an aggregate of their values. Like
/// [`measurement_value_expr`] it refers to the row only by `object_id` (the
/// inner query renames its own), so it works whatever `objects` is called.
pub(super) fn children_value_expr(c: &ChildColumn) -> Result<String, InternalErrors> {
    let (value, aggregate) = match &c.value {
        ChildValue::Count => ("1".to_string(), "count"),
        ChildValue::Of(aggregation, column) => (
            column_aggregate_expr(column)?,
            aggregation_sql_fn(aggregation),
        ),
    };
    let child_filter = match c.child_class {
        ObjectClass::Global(n) => format!(" AND list_contains(object_class_id, {n})"),
        ObjectClass::Unset | ObjectClass::Local(_) => String::new(),
    };
    Ok(format!(
        "(SELECT {aggregate}(cv.v) FROM (SELECT parent_ids[{parent}] AS pid, {value} AS v \
         FROM objects WHERE parent_ids[{parent}] IS NOT NULL{child_filter}) cv \
         WHERE cv.pid = object_id)",
        parent = coloc_class_key(c.parent_class),
    ))
}

/// The SQL of a [`Column::Children`] / [`Column::Parent`] column for a list
/// row: the parent's id as text, everything else as [`column_aggregate_expr`].
pub(super) fn related_value_expr(column: &Column) -> Result<String, InternalErrors> {
    match column {
        Column::Parent(ParentColumn {
            parent_class,
            value: ParentValue::Id,
        }) => Ok(format!(
            "CAST(parent_ids[{}] AS VARCHAR)",
            coloc_class_key(*parent_class)
        )),
        _ => column_aggregate_expr(column),
    }
}

/// A measurement's value for the object of the row - its own (a value per
/// object) or its image plane's (a value per image). NULL without a value.
///
/// An object's own value is read from its `measurement_values` list, at the
/// measurement's `object_index` (one lookup for the whole query): a join
/// with `measurements` per column took ~2 s and ~4 GB for ten columns over
/// 5.6M objects, the list ~80 ms and ~290 MB (BENCHMARK.md). An image's
/// value refers to the row only by `object_id`, which no other table of
/// these queries has, so it works whatever `objects` is called there (`o`,
/// `objects`, joined with `images`).
pub(super) fn measurement_value_expr(m: &MeasurementColumn) -> String {
    let key = m.key.replace('\'', "''");
    if m.per_object {
        format!(
            "measurement_values[(SELECT object_index FROM measurement_info WHERE key = '{key}')]"
        )
    } else {
        format!(
            "(SELECT mv.v FROM (SELECT ob.object_id AS oid, m.value AS v \
             FROM objects ob JOIN measurements m ON m.image_rel_path = ob.image_rel_path \
             AND m.z_stack IS NOT DISTINCT FROM ob.z_stack AND m.t_stack = ob.t_stack \
             WHERE m.key = '{key}' AND m.region = 'image') mv WHERE mv.oid = object_id)"
        )
    }
}

pub(super) fn aggregation_sql_fn(aggregation: &Aggregation) -> &'static str {
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
pub(super) fn aggregate_sql(
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
pub(super) fn zero_when_no_objects(column: &Column, aggregation: &Aggregation) -> bool {
    // An image measurement is the image's own value, objects or not.
    !is_image_measurement(column)
        && (matches!(column, Column::Count) || matches!(aggregation, Aggregation::Sum))
}

/// `value` (an aggregate over a group's objects), but 0 instead of NULL
/// when the group has no objects (`no_objects`) on images that were
/// analysed (`measured`, see `ResultsGenerator::measured_on_plane_sql`) -
/// for the statistics [`zero_when_no_objects`] allows. Keyed on "no
/// objects" rather than on NULL alone, so a sum over objects whose own
/// value is missing (e.g. nm areas without pixel sizes) stays empty.
pub(super) fn fill_zero_sql(
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
