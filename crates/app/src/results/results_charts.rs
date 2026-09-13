use super::results_generator::{
    Column, PlaneFilter, ResultsGenerator, class_display_label, column_aggregate_expr,
    sql_int_array_literal, sql_string_in_list,
};
use duckdb::types::Value;
use evanalyzer_cfg::core_types::{InternalErrors, ObjectClass};

/// One histogram: `column`'s value distribution across every object
/// matching `plane`/`images`/`object_classes`, bucketed into `bins`
/// equal-width bins — same plane/images/class scoping every other results
/// view uses (see `ListFilter`/`PlateFilter` in results_generator.rs).
#[derive(Clone)]
pub struct HistogramFilter {
    pub plane: PlaneFilter,
    /// `None` means every image in the database.
    pub images: Option<Vec<String>>,
    /// `None` means every object class registered in the database.
    pub object_classes: Option<Vec<ObjectClass>>,
    /// The measurement to bucket.
    pub column: Column,
    /// Number of equal-width bins to bucket `column`'s values into.
    pub bins: usize,
}

/// One scatter plot: every matched object's `(x_column, y_column)` pair,
/// one point per object.
#[derive(Clone)]
pub struct ScatterFilter {
    pub plane: PlaneFilter,
    /// `None` means every image in the database.
    pub images: Option<Vec<String>>,
    /// `None` means every object class registered in the database.
    pub object_classes: Option<Vec<ObjectClass>>,
    pub x_column: Column,
    pub y_column: Column,
    /// Caps how many points are actually plotted — a scatter of every
    /// object in a large database is neither readable nor cheap to render.
    /// `None` means no cap (every matched object).
    pub max_points: Option<usize>,
}

/// One boxplot: `column`'s distribution (min/Q1/median/Q3/max, plus
/// outliers) — one box per class in `object_classes` (or every registered
/// class if `None`), so distributions across classes can be compared side
/// by side in a single chart.
#[derive(Clone)]
pub struct BoxplotFilter {
    pub plane: PlaneFilter,
    /// `None` means every image in the database.
    pub images: Option<Vec<String>>,
    /// `None` means every object class registered in the database — one
    /// box per class.
    pub object_classes: Option<Vec<ObjectClass>>,
    pub column: Column,
}

/// One equal-width-binned histogram, ready for the GUI to draw straight
/// from `bin_edges`/`counts` — `bin_edges` has `counts.len() + 1` entries
/// (edge `i` / edge `i+1` bound bin `i`).
pub struct HistogramResult {
    pub bin_edges: Vec<f64>,
    pub counts: Vec<u64>,
    pub min: f64,
    pub max: f64,
}

#[derive(Clone, Copy)]
pub struct ScatterPoint {
    pub x: f64,
    pub y: f64,
}

/// `points` may be a random sample of `total_object_count` rather than
/// every one of them — see `ScatterFilter::max_points`.
pub struct ScatterResult {
    pub points: Vec<ScatterPoint>,
    pub x_min: f64,
    pub x_max: f64,
    pub y_min: f64,
    pub y_max: f64,
    pub total_object_count: usize,
}

/// One class's box: min/Q1/median/Q3/max plus any Tukey outliers (values
/// beyond 1.5x the interquartile range from Q1/Q3).
pub struct BoxplotBox {
    pub label: String,
    pub color: u32,
    pub min: f64,
    pub q1: f64,
    pub median: f64,
    pub q3: f64,
    pub max: f64,
    pub outliers: Vec<f64>,
    pub object_count: usize,
}

pub struct BoxplotResult {
    pub boxes: Vec<BoxplotBox>,
}

pub struct ResultCharts {}

impl ResultCharts {
    /// One box (min/Q1/median/Q3/max + Tukey outliers) per object class —
    /// `object_class_id` is itself an array column (an object can belong to
    /// more than one class, see `results_generator.rs`'s `get_grouped_by_image`
    /// doc comment), so this unnests it the same way, grouping by the
    /// exploded class id rather than filtering to a single one, so every
    /// requested class gets its own box in one query instead of one query
    /// per class.
    pub fn paint_boxplot(
        &self,
        database: &ResultsGenerator,
        filter: &BoxplotFilter,
    ) -> Result<BoxplotResult, InternalErrors> {
        let err = |e: duckdb::Error| InternalErrors::Io(e.to_string());
        let classes = database.get_object_classes()?;
        let value_expr = chart_value_expr(&filter.column)?;

        let mut conditions = vec![
            format!("z_stack = {}", filter.plane.z_stack),
            format!("t_stack = {}", filter.plane.t_stack),
        ];
        if let Some(images) = &filter.images {
            if images.is_empty() {
                return Ok(BoxplotResult { boxes: Vec::new() });
            }
            conditions.push(format!(
                "image_rel_path IN ({})",
                sql_string_in_list(images)
            ));
        }
        let where_clause = format!("WHERE {}", conditions.join(" AND "));

        let class_filter = match &filter.object_classes {
            Some(wanted) => {
                let ids: Vec<u32> = wanted
                    .iter()
                    .filter_map(|id| match id {
                        ObjectClass::Valid(n) => Some(*n),
                        ObjectClass::Unset => None,
                    })
                    .collect();
                if ids.is_empty() {
                    return Ok(BoxplotResult { boxes: Vec::new() });
                }
                format!(
                    "WHERE class_id IN ({})",
                    ids.iter()
                        .map(u32::to_string)
                        .collect::<Vec<_>>()
                        .join(", ")
                )
            }
            None => String::new(),
        };

        let sql = format!(
            "WITH exploded AS (\n\
                 SELECT class_id, {value_expr} AS v\n\
                 FROM objects, UNNEST(CAST(object_class_id AS INTEGER[])) AS u(class_id)\n\
                 {where_clause}\n\
             ),\n\
             filtered AS (\n\
                 SELECT * FROM exploded {class_filter}\n\
             ),\n\
             stats AS (\n\
                 SELECT class_id,\n\
                        MIN(v) AS min_v,\n\
                        quantile_cont(v, 0.25) AS q1,\n\
                        quantile_cont(v, 0.5) AS med,\n\
                        quantile_cont(v, 0.75) AS q3,\n\
                        MAX(v) AS max_v,\n\
                        COUNT(*) AS n\n\
                 FROM filtered\n\
                 GROUP BY class_id\n\
             )\n\
             SELECT s.class_id, s.min_v, s.q1, s.med, s.q3, s.max_v, s.n,\n\
                    LIST(f.v) FILTER (\n\
                        WHERE f.v < s.q1 - 1.5 * (s.q3 - s.q1) OR f.v > s.q3 + 1.5 * (s.q3 - s.q1)\n\
                    ) AS outliers\n\
             FROM stats s\n\
             JOIN filtered f ON f.class_id = s.class_id\n\
             GROUP BY s.class_id, s.min_v, s.q1, s.med, s.q3, s.max_v, s.n\n\
             ORDER BY s.class_id"
        );

        let mut stmt = database.connection().prepare(&sql).map_err(err)?;
        #[allow(clippy::type_complexity)]
        let rows: Vec<(u32, f64, f64, f64, f64, f64, i64, Value)> = stmt
            .query_map([], |row| {
                Ok((
                    row.get(0)?,
                    row.get(1)?,
                    row.get(2)?,
                    row.get(3)?,
                    row.get(4)?,
                    row.get(5)?,
                    row.get(6)?,
                    row.get(7)?,
                ))
            })
            .map_err(err)?
            .collect::<Result<Vec<_>, _>>()
            .map_err(err)?;

        let boxes = rows
            .into_iter()
            .map(|(class_id, min_v, q1, median, q3, max_v, n, outliers)| {
                let object_class = ObjectClass::Valid(class_id);
                let color = classes
                    .iter()
                    .find(|class| class.id == object_class)
                    .map(|class| class.color)
                    .unwrap_or(0);
                BoxplotBox {
                    label: class_display_label(object_class, &classes),
                    color,
                    min: min_v,
                    q1,
                    median,
                    q3,
                    max: max_v,
                    outliers: extract_f64_list(outliers),
                    object_count: n as usize,
                }
            })
            .collect();

        Ok(BoxplotResult { boxes })
    }

    /// One equal-width-binned histogram of `filter.column`'s values across
    /// every object matching `plane`/`images`/`object_classes` — a single
    /// combined distribution, not split per class (unlike `paint_boxplot`
    /// above), matching the Charts toolbar's own single-select CLASS filter
    /// (a filter, not a group-by axis, for this view).
    pub fn paint_histogram(
        &self,
        database: &ResultsGenerator,
        filter: &HistogramFilter,
    ) -> Result<HistogramResult, InternalErrors> {
        let err = |e: duckdb::Error| InternalErrors::Io(e.to_string());
        let value_expr = chart_value_expr(&filter.column)?;
        let bins = filter.bins.max(1);
        let Some(where_clause) =
            chart_where_clause(&filter.plane, &filter.images, &filter.object_classes)
        else {
            return Ok(HistogramResult {
                bin_edges: (0..=bins).map(|i| i as f64).collect(),
                counts: vec![0; bins],
                min: 0.0,
                max: 0.0,
            });
        };

        let bounds_sql = format!(
            "SELECT MIN(v), MAX(v), COUNT(*) FROM (SELECT {value_expr} AS v FROM objects {where_clause})"
        );
        let (min, max, count): (Option<f64>, Option<f64>, i64) = database
            .connection()
            .query_row(&bounds_sql, [], |row| {
                Ok((row.get(0)?, row.get(1)?, row.get(2)?))
            })
            .map_err(err)?;
        let (Some(min), Some(max)) = (min, max) else {
            return Ok(HistogramResult {
                bin_edges: (0..=bins).map(|i| i as f64).collect(),
                counts: vec![0; bins],
                min: 0.0,
                max: 0.0,
            });
        };

        let mut counts = vec![0u64; bins];
        if max > min {
            // 0-based bucket index, clamped into `[0, bins-1]` so the
            // exact max value (which would otherwise floor to `bins`,
            // one past the end) lands in the last bin instead.
            let bucket_sql = format!(
                "SELECT LEAST({bins} - 1, CAST(FLOOR({bins}::DOUBLE * (v - {min}) / ({max} - {min})) AS BIGINT)) AS bucket,\n\
                        COUNT(*) AS cnt\n\
                 FROM (SELECT {value_expr} AS v FROM objects {where_clause})\n\
                 GROUP BY bucket"
            );
            let mut stmt = database.connection().prepare(&bucket_sql).map_err(err)?;
            let buckets: Vec<(i64, i64)> = stmt
                .query_map([], |row| Ok((row.get(0)?, row.get(1)?)))
                .map_err(err)?
                .collect::<Result<Vec<_>, _>>()
                .map_err(err)?;
            for (bucket, cnt) in buckets {
                if let Some(slot) = usize::try_from(bucket).ok().and_then(|b| counts.get_mut(b)) {
                    *slot = cnt as u64;
                }
            }
        } else {
            // Every matched object has the same value — a single spike,
            // not a division by zero.
            counts[0] = count as u64;
        }

        let bin_edges = (0..=bins)
            .map(|i| min + (max - min) * (i as f64 / bins as f64))
            .collect();

        Ok(HistogramResult {
            bin_edges,
            counts,
            min,
            max,
        })
    }

    /// Every matched object's `(x_column, y_column)` pair, one point per
    /// object — randomly sampled down to `max_points` (DuckDB's own
    /// reservoir sampling) when the match set is larger than that, since a
    /// scatter of hundreds of thousands of points is neither readable nor
    /// cheap to render.
    pub fn paint_scatter(
        &self,
        database: &ResultsGenerator,
        filter: &ScatterFilter,
    ) -> Result<ScatterResult, InternalErrors> {
        let err = |e: duckdb::Error| InternalErrors::Io(e.to_string());
        let x_expr = chart_value_expr(&filter.x_column)?;
        let y_expr = chart_value_expr(&filter.y_column)?;
        let Some(where_clause) =
            chart_where_clause(&filter.plane, &filter.images, &filter.object_classes)
        else {
            return Ok(ScatterResult {
                points: Vec::new(),
                x_min: 0.0,
                x_max: 0.0,
                y_min: 0.0,
                y_max: 0.0,
                total_object_count: 0,
            });
        };

        let count_sql = format!("SELECT COUNT(*) FROM objects {where_clause}");
        let total_object_count: i64 = database
            .connection()
            .query_row(&count_sql, [], |row| row.get(0))
            .map_err(err)?;

        let sample_clause = match filter.max_points {
            Some(max_points) if (total_object_count as usize) > max_points => {
                format!("USING SAMPLE {max_points} ROWS (reservoir)")
            }
            _ => String::new(),
        };
        let points_sql = format!(
            "SELECT {x_expr} AS x, {y_expr} AS y FROM objects {where_clause} {sample_clause}"
        );
        let mut stmt = database.connection().prepare(&points_sql).map_err(err)?;
        let points: Vec<ScatterPoint> = stmt
            .query_map([], |row| {
                Ok(ScatterPoint {
                    x: row.get(0)?,
                    y: row.get(1)?,
                })
            })
            .map_err(err)?
            .collect::<Result<Vec<_>, _>>()
            .map_err(err)?;

        let mut x_min = f64::INFINITY;
        let mut x_max = f64::NEG_INFINITY;
        let mut y_min = f64::INFINITY;
        let mut y_max = f64::NEG_INFINITY;
        for point in &points {
            x_min = x_min.min(point.x);
            x_max = x_max.max(point.x);
            y_min = y_min.min(point.y);
            y_max = y_max.max(point.y);
        }
        if !x_min.is_finite() || !x_max.is_finite() {
            x_min = 0.0;
            x_max = 0.0;
        }
        if !y_min.is_finite() || !y_max.is_finite() {
            y_min = 0.0;
            y_max = 0.0;
        }

        Ok(ScatterResult {
            points,
            x_min,
            x_max,
            y_min,
            y_max,
            total_object_count: total_object_count as usize,
        })
    }
}

/// `column_aggregate_expr`, wrapped with a chart-specific error message —
/// that function's own wording ("cannot be aggregated for the plate view
/// yet") is misleading when the failure actually surfaced from a
/// histogram/scatter/boxplot request; the set of supported columns is the
/// same either way (identity columns and per-channel intensity aren't
/// resolvable to a single per-object SQL expression yet).
fn chart_value_expr(column: &Column) -> Result<String, InternalErrors> {
    column_aggregate_expr(column).map_err(|_| {
        InternalErrors::InvalidArgument(format!(
            "column {} cannot be charted yet",
            column.as_key(&[])
        ))
    })
}

/// The plane/images/object_classes `WHERE` clause `paint_histogram`/
/// `paint_scatter` share (a single combined match set — `paint_boxplot`
/// groups by class itself and builds its own, since it needs
/// `object_class_id` exploded via `UNNEST` rather than filtered with
/// `list_has_any`). `None` means an empty `Some(vec![])` filter (no
/// image/class actually selected) that can never match anything — the
/// caller should short-circuit to an empty result instead of building an
/// invalid `IN ()`.
fn chart_where_clause(
    plane: &PlaneFilter,
    images: &Option<Vec<String>>,
    object_classes: &Option<Vec<ObjectClass>>,
) -> Option<String> {
    let mut conditions = vec![
        format!("z_stack = {}", plane.z_stack),
        format!("t_stack = {}", plane.t_stack),
    ];
    if let Some(images) = images {
        if images.is_empty() {
            return None;
        }
        conditions.push(format!(
            "image_rel_path IN ({})",
            sql_string_in_list(images)
        ));
    }
    if let Some(wanted) = object_classes {
        let ids: Vec<u32> = wanted
            .iter()
            .filter_map(|id| match id {
                ObjectClass::Valid(n) => Some(*n),
                ObjectClass::Unset => None,
            })
            .collect();
        if ids.is_empty() {
            return None;
        }
        conditions.push(format!(
            "list_has_any(CAST(object_class_id AS INTEGER[]), {})",
            sql_int_array_literal(&ids)
        ));
    }
    Some(format!("WHERE {}", conditions.join(" AND ")))
}

/// Converts a DuckDB list/array value (as returned for `LIST(...)`'s own
/// result column, e.g. `paint_boxplot`'s outliers) into a `Vec<f64>`,
/// dropping any non-numeric elements.
fn extract_f64_list(value: Value) -> Vec<f64> {
    match value {
        Value::List(items) | Value::Array(items) => {
            items.into_iter().filter_map(value_to_f64).collect()
        }
        _ => vec![],
    }
}

/// Every numeric `Value` variant DuckDB might hand back for a `v` (e.g.
/// `object_class_id`-adjacent measurement columns span `UBIGINT`/`DOUBLE`/
/// `BIGINT` depending on which one) as `f64` — unlike `row.get::<_, f64>()`
/// on a *typed* column (which converts transparently), a `LIST(...)`
/// aggregate's elements come back as untyped `Value`s needing this done by
/// hand.
fn value_to_f64(value: Value) -> Option<f64> {
    match value {
        Value::Double(v) => Some(v),
        Value::Float(v) => Some(v as f64),
        Value::TinyInt(v) => Some(v as f64),
        Value::SmallInt(v) => Some(v as f64),
        Value::Int(v) => Some(v as f64),
        Value::BigInt(v) => Some(v as f64),
        Value::HugeInt(v) => Some(v as f64),
        Value::UTinyInt(v) => Some(v as f64),
        Value::USmallInt(v) => Some(v as f64),
        Value::UInt(v) => Some(v as f64),
        Value::UBigInt(v) => Some(v as f64),
        Value::UHugeInt(v) => Some(v as f64),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::results::results_generator::ResultsGenerator;
    use crate::results::test_support::{ObjectSpec, seed_db};

    fn open(objects: &[ObjectSpec]) -> ResultsGenerator {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("results.evadb");
        seed_db(&path, objects);
        // Leaking the tempdir keeps the backing file alive for the life of
        // the returned `ResultsGenerator` (which only holds an open
        // `duckdb::Connection`, not the directory) - acceptable for a test
        // process that exits shortly after.
        std::mem::forget(dir);
        ResultsGenerator::open_database(path).expect("open database")
    }

    fn plane() -> PlaneFilter {
        PlaneFilter {
            z_stack: 0,
            t_stack: 0,
        }
    }

    // -- paint_boxplot --------------------------------------------------

    #[test]
    fn paint_boxplot_computes_correct_quartiles_and_flags_a_tukey_outlier() {
        // Sorted [10, 20, 30, 40, 1000]: quantile_cont at p=0.25/0.5/0.75
        // over 5 values lands exactly on indices 1/2/3 (no interpolation
        // needed), so q1=20, median=30, q3=40 - IQR=20, outlier bounds
        // [20-30, 40+30] = [-10, 70], so 1000 is a Tukey outlier but still
        // counts toward MIN/MAX (10..1000).
        let objects = [10u64, 20, 30, 40, 1000]
            .into_iter()
            .map(|area| ObjectSpec::new("img1.tif", "ClassA", 1, area))
            .collect::<Vec<_>>();
        let generator = open(&objects);
        let result = ResultCharts {}
            .paint_boxplot(
                &generator,
                &BoxplotFilter {
                    plane: plane(),
                    images: None,
                    object_classes: None,
                    column: Column::AreaSizePx,
                },
            )
            .expect("boxplot");

        assert_eq!(result.boxes.len(), 1);
        let b = &result.boxes[0];
        assert_eq!(b.label, "ClassA");
        assert_eq!(b.object_count, 5);
        assert_eq!(b.min, 10.0);
        assert_eq!(b.q1, 20.0);
        assert_eq!(b.median, 30.0);
        assert_eq!(b.q3, 40.0);
        assert_eq!(b.max, 1000.0);
        assert_eq!(b.outliers, vec![1000.0]);
    }

    #[test]
    fn paint_boxplot_gives_each_class_its_own_box() {
        let mut objects = Vec::new();
        for area in [10u64, 20, 30, 40, 50] {
            objects.push(ObjectSpec::new("img1.tif", "ClassA", 1, area));
        }
        for area in [100u64, 200, 300] {
            objects.push(ObjectSpec::new("img1.tif", "ClassB", 2, area));
        }
        let generator = open(&objects);
        let result = ResultCharts {}
            .paint_boxplot(
                &generator,
                &BoxplotFilter {
                    plane: plane(),
                    images: None,
                    object_classes: None,
                    column: Column::AreaSizePx,
                },
            )
            .expect("boxplot");

        assert_eq!(result.boxes.len(), 2);
        let a = result.boxes.iter().find(|b| b.label == "ClassA").unwrap();
        let b = result.boxes.iter().find(|b| b.label == "ClassB").unwrap();
        assert_eq!(a.object_count, 5);
        assert_eq!(b.object_count, 3);
        assert_eq!(b.median, 200.0);
    }

    #[test]
    fn paint_boxplot_object_classes_filter_restricts_to_selected_classes() {
        let mut objects = Vec::new();
        for area in [10u64, 20, 30] {
            objects.push(ObjectSpec::new("img1.tif", "ClassA", 1, area));
        }
        for area in [100u64, 200, 300] {
            objects.push(ObjectSpec::new("img1.tif", "ClassB", 2, area));
        }
        let generator = open(&objects);
        let result = ResultCharts {}
            .paint_boxplot(
                &generator,
                &BoxplotFilter {
                    plane: plane(),
                    images: None,
                    object_classes: Some(vec![ObjectClass::Valid(1)]),
                    column: Column::AreaSizePx,
                },
            )
            .expect("boxplot");

        assert_eq!(result.boxes.len(), 1);
        assert_eq!(result.boxes[0].label, "ClassA");
    }

    #[test]
    fn paint_boxplot_with_an_explicitly_empty_image_selection_is_empty() {
        let objects = [ObjectSpec::new("img1.tif", "ClassA", 1, 10)];
        let generator = open(&objects);
        let result = ResultCharts {}
            .paint_boxplot(
                &generator,
                &BoxplotFilter {
                    plane: plane(),
                    images: Some(vec![]),
                    object_classes: None,
                    column: Column::AreaSizePx,
                },
            )
            .expect("boxplot");
        assert!(result.boxes.is_empty());
    }

    #[test]
    fn paint_boxplot_with_an_explicitly_empty_class_selection_is_empty() {
        let objects = [ObjectSpec::new("img1.tif", "ClassA", 1, 10)];
        let generator = open(&objects);
        let result = ResultCharts {}
            .paint_boxplot(
                &generator,
                &BoxplotFilter {
                    plane: plane(),
                    images: None,
                    object_classes: Some(vec![]),
                    column: Column::AreaSizePx,
                },
            )
            .expect("boxplot");
        assert!(result.boxes.is_empty());
    }

    #[test]
    fn paint_boxplot_rejects_a_column_that_cannot_be_charted() {
        let objects = [ObjectSpec::new("img1.tif", "ClassA", 1, 10)];
        let generator = open(&objects);
        let result = ResultCharts {}.paint_boxplot(
            &generator,
            &BoxplotFilter {
                plane: plane(),
                images: None,
                object_classes: None,
                column: Column::ObjectId,
            },
        );
        assert!(result.is_err());
    }

    // -- paint_histogram --------------------------------------------------

    #[test]
    fn paint_histogram_buckets_a_single_repeated_value_into_one_bin() {
        let objects: Vec<ObjectSpec> = (0..5)
            .map(|_| ObjectSpec::new("img1.tif", "ClassA", 1, 42))
            .collect();
        let generator = open(&objects);
        let result = ResultCharts {}
            .paint_histogram(
                &generator,
                &HistogramFilter {
                    plane: plane(),
                    images: None,
                    object_classes: None,
                    column: Column::AreaSizePx,
                    bins: 3,
                },
            )
            .expect("histogram");

        assert_eq!(result.min, 42.0);
        assert_eq!(result.max, 42.0);
        assert_eq!(result.counts, vec![5, 0, 0]);
        assert_eq!(result.bin_edges.len(), 4);
    }

    #[test]
    fn paint_histogram_spreads_values_across_bins_and_conserves_total_count() {
        let objects: Vec<ObjectSpec> = (0..10)
            .map(|i| ObjectSpec::new("img1.tif", "ClassA", 1, i * 10))
            .collect();
        let generator = open(&objects);
        let result = ResultCharts {}
            .paint_histogram(
                &generator,
                &HistogramFilter {
                    plane: plane(),
                    images: None,
                    object_classes: None,
                    column: Column::AreaSizePx,
                    bins: 5,
                },
            )
            .expect("histogram");

        assert_eq!(result.min, 0.0);
        assert_eq!(result.max, 90.0);
        assert_eq!(result.counts.iter().sum::<u64>(), 10);
        assert_eq!(result.counts.len(), 5);
        assert_eq!(result.bin_edges.len(), 6);
        assert_eq!(result.bin_edges[0], 0.0);
        assert_eq!(result.bin_edges[5], 90.0);
    }

    #[test]
    fn paint_histogram_with_an_explicitly_empty_image_selection_is_empty() {
        let objects = [ObjectSpec::new("img1.tif", "ClassA", 1, 10)];
        let generator = open(&objects);
        let result = ResultCharts {}
            .paint_histogram(
                &generator,
                &HistogramFilter {
                    plane: plane(),
                    images: Some(vec![]),
                    object_classes: None,
                    column: Column::AreaSizePx,
                    bins: 4,
                },
            )
            .expect("histogram");
        assert_eq!(result.counts, vec![0; 4]);
        assert_eq!(result.min, 0.0);
        assert_eq!(result.max, 0.0);
    }

    #[test]
    fn paint_histogram_bins_is_clamped_to_at_least_one() {
        let objects = [ObjectSpec::new("img1.tif", "ClassA", 1, 10)];
        let generator = open(&objects);
        let result = ResultCharts {}
            .paint_histogram(
                &generator,
                &HistogramFilter {
                    plane: plane(),
                    images: None,
                    object_classes: None,
                    column: Column::AreaSizePx,
                    bins: 0,
                },
            )
            .expect("histogram");
        assert_eq!(result.counts.len(), 1);
        assert_eq!(result.counts[0], 1);
    }

    #[test]
    fn paint_histogram_image_filter_restricts_to_the_selected_image() {
        let objects = [
            ObjectSpec::new("img1.tif", "ClassA", 1, 10),
            ObjectSpec::new("img2.tif", "ClassA", 1, 20),
        ];
        let generator = open(&objects);
        let result = ResultCharts {}
            .paint_histogram(
                &generator,
                &HistogramFilter {
                    plane: plane(),
                    images: Some(vec!["img1.tif".to_string()]),
                    object_classes: None,
                    column: Column::AreaSizePx,
                    bins: 1,
                },
            )
            .expect("histogram");
        assert_eq!(result.counts, vec![1]);
        assert_eq!(result.min, 10.0);
        assert_eq!(result.max, 10.0);
    }

    #[test]
    fn paint_histogram_plane_filter_excludes_objects_on_other_planes() {
        let objects = [
            ObjectSpec::new("img1.tif", "ClassA", 1, 10),
            ObjectSpec::new("img1.tif", "ClassA", 1, 20).at_plane(1, 0),
        ];
        let generator = open(&objects);
        let result = ResultCharts {}
            .paint_histogram(
                &generator,
                &HistogramFilter {
                    plane: PlaneFilter {
                        z_stack: 0,
                        t_stack: 0,
                    },
                    images: None,
                    object_classes: None,
                    column: Column::AreaSizePx,
                    bins: 1,
                },
            )
            .expect("histogram");
        assert_eq!(result.counts, vec![1]);
        assert_eq!(result.min, 10.0);
        assert_eq!(result.max, 10.0);
    }

    // -- paint_scatter ----------------------------------------------------

    #[test]
    fn paint_scatter_returns_one_point_per_object_on_the_diagonal() {
        let objects: Vec<ObjectSpec> = (1..=5)
            .map(|i| ObjectSpec::new("img1.tif", "ClassA", 1, i * 10))
            .collect();
        let generator = open(&objects);
        let result = ResultCharts {}
            .paint_scatter(
                &generator,
                &ScatterFilter {
                    plane: plane(),
                    images: None,
                    object_classes: None,
                    x_column: Column::AreaSizePx,
                    y_column: Column::AreaSizePx,
                    max_points: None,
                },
            )
            .expect("scatter");

        assert_eq!(result.points.len(), 5);
        assert_eq!(result.total_object_count, 5);
        for point in &result.points {
            assert_eq!(point.x, point.y, "x/y should match: both read area_px");
        }
        assert_eq!(result.x_min, 10.0);
        assert_eq!(result.x_max, 50.0);
        assert_eq!(result.y_min, 10.0);
        assert_eq!(result.y_max, 50.0);
    }

    #[test]
    fn paint_scatter_max_points_caps_the_point_count_but_not_the_reported_total() {
        let objects: Vec<ObjectSpec> = (1..=20)
            .map(|i| ObjectSpec::new("img1.tif", "ClassA", 1, i))
            .collect();
        let generator = open(&objects);
        let result = ResultCharts {}
            .paint_scatter(
                &generator,
                &ScatterFilter {
                    plane: plane(),
                    images: None,
                    object_classes: None,
                    x_column: Column::AreaSizePx,
                    y_column: Column::AreaSizePx,
                    max_points: Some(5),
                },
            )
            .expect("scatter");

        assert_eq!(result.total_object_count, 20);
        assert!(
            result.points.len() <= 5,
            "expected at most 5 sampled points, got {}",
            result.points.len()
        );
    }

    #[test]
    fn paint_scatter_with_an_explicitly_empty_class_selection_is_empty() {
        let objects = [ObjectSpec::new("img1.tif", "ClassA", 1, 10)];
        let generator = open(&objects);
        let result = ResultCharts {}
            .paint_scatter(
                &generator,
                &ScatterFilter {
                    plane: plane(),
                    images: None,
                    object_classes: Some(vec![]),
                    x_column: Column::AreaSizePx,
                    y_column: Column::AreaSizePx,
                    max_points: None,
                },
            )
            .expect("scatter");
        assert!(result.points.is_empty());
        assert_eq!(result.total_object_count, 0);
    }

    #[test]
    fn paint_scatter_rejects_a_column_that_cannot_be_charted() {
        let objects = [ObjectSpec::new("img1.tif", "ClassA", 1, 10)];
        let generator = open(&objects);
        let result = ResultCharts {}.paint_scatter(
            &generator,
            &ScatterFilter {
                plane: plane(),
                images: None,
                object_classes: None,
                x_column: Column::IntensityAvg(0),
                y_column: Column::AreaSizePx,
                max_points: None,
            },
        );
        assert!(result.is_err());
    }
}
