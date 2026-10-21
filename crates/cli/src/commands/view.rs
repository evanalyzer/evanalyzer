use crate::args::{ColumnsArgs, ViewArgs};
use crate::commands::common::{
    cell_text, resolve_image_rel_paths, resolve_object_classes, shown_planes,
};
use crate::table::print_object_table;
use evanalyzer_app::backends::Backend;
use evanalyzer_app::results::Cell;
use evanalyzer_app::results::CellValue;
use evanalyzer_app::results::Column;
use evanalyzer_app::results::DatabaseResult;
use evanalyzer_app::results::ListFilter;
use evanalyzer_app::results::Pagination;
use evanalyzer_app::results::PlaneFilter;
use evanalyzer_cfg::core_types::InternalErrors;
use serde_json::json;

/// Whether `column` is a per-channel intensity column — excluded from the
/// default view unless `--channels` is given, same split the old CLI's own
/// `--channels` flag drew.
fn is_intensity_column(column: &Column) -> bool {
    matches!(
        column,
        Column::IntensityAvg(_)
            | Column::IntensitySum(_)
            | Column::IntensityMin(_)
            | Column::IntensityMax(_)
    )
}

/// A cell holding just `value` - a row's "T"/"Z".
fn plane_cell(value: u32) -> Cell {
    Cell {
        value: CellValue::Integer(value as i32),
        bg_color: 0,
        alternating_color: false,
        search_key: None,
        disabled: false,
        any_disabled: false,
        failed: false,
        any_failed: false,
    }
}

/// The rows `view` shows: page `args.page` of `base_filter`'s objects
/// over every plane (`--t`/`--z` choose), time point by time point, each
/// row with its "T"/"Z" when the results have several. The list's paging is
/// keyset-based (a cursor from the previous page), so earlier pages are
/// walked through - fine for a quick look, `export` is for the whole table.
fn rows_page(
    db: &dyn evanalyzer_app::results::ResultsSource,
    base_filter: &ListFilter,
    args: &ViewArgs,
) -> Result<DatabaseResult, InternalErrors> {
    let limit = args.limit.max(1);
    let shown = shown_planes(db, args.filter.t_stacks, args.filter.z_stacks)?;
    let skip = args.page * limit;
    let mut column_names: Option<Vec<String>> = None;
    let mut rows: Vec<Vec<Cell>> = Vec::new();
    let mut seen = 0usize;
    'planes: for &plane in &shown.planes {
        let mut cursor: Option<String> = None;
        loop {
            let page = db.get_object_list(&ListFilter {
                plane,
                page: Pagination {
                    limit: limit as i32,
                    after: cursor.take(),
                },
                ..base_filter.clone()
            })?;
            column_names.get_or_insert_with(|| page.column_names.clone());
            let is_last_page = page.source_object_count < limit;
            cursor = page.row_names.last().cloned();
            for row in page.rows {
                if seen >= skip {
                    let mut cells = Vec::new();
                    if shown.by_t {
                        cells.push(plane_cell(plane.t_stack));
                    }
                    if shown.by_z {
                        cells.push(plane_cell(plane.z_stack));
                    }
                    cells.extend(row);
                    rows.push(cells);
                    if rows.len() == limit {
                        break 'planes;
                    }
                }
                seen += 1;
            }
            if is_last_page || cursor.is_none() {
                break;
            }
        }
    }
    let column_names = match column_names {
        Some(names) => names,
        // Nothing in the chosen planes: still the columns.
        None => db.get_object_list(base_filter)?.column_names,
    };
    let mut header = Vec::new();
    if shown.by_t {
        header.push("T".to_string());
    }
    if shown.by_z {
        header.push("Z".to_string());
    }
    header.extend(column_names);
    Ok(DatabaseResult {
        column_names: header,
        row_names: Vec::new(),
        source_object_count: rows.len(),
        rows,
        min: 0.0,
        max: 0.0,
        row_locations: Vec::new(),
    })
}

pub fn run(args: ViewArgs, backend: &dyn Backend) -> Result<(), InternalErrors> {
    let db = backend.open_results(&args.db)?;
    let db = db.as_ref();

    if args.filter.colocalized.is_some() {
        return Err(InternalErrors::InvalidArgument(
            "--colocalized isn't supported by the current results backend".to_string(),
        ));
    }

    let images = db.get_images()?;
    let classes = db.get_object_classes()?;
    let image_names: Vec<String> = images.iter().map(|image| image.name.clone()).collect();
    let class_names: Vec<String> = classes.iter().map(|class| class.name.clone()).collect();

    let image_rel_paths = resolve_image_rel_paths(db, &args.filter.images)?;
    let object_classes = resolve_object_classes(db, &args.filter.classes)?;
    let columns: Vec<Column> = db
        .get_available_columns()?
        .into_iter()
        .filter(|entry| args.channels || !is_intensity_column(&entry.key))
        .map(|entry| entry.key)
        .collect();

    let limit = args.limit.max(1);
    let base_filter = ListFilter {
        plane: PlaneFilter {
            z_stack: 0,
            t_stack: 0,
        },
        images: (!image_rel_paths.is_empty()).then_some(image_rel_paths),
        object_classes: (!object_classes.is_empty()).then_some(object_classes),
        columns,
        with_coloc_details: false,
        page: Pagination {
            limit: limit as i32,
            after: None,
        },
        transpond_table: args.filter.transpond.unwrap_or_else(|| false),
    };

    let result = rows_page(db, &base_filter, &args)?;

    if args.json {
        let rows: Vec<_> = result
            .rows
            .iter()
            .map(|row| {
                serde_json::Value::Object(
                    result
                        .column_names
                        .iter()
                        .zip(row.iter())
                        .map(|(name, cell)| (name.clone(), json!(cell_text(cell))))
                        .collect(),
                )
            })
            .collect();
        let out = json!({
            "db": args.db,
            "images": image_names,
            "classes": class_names,
            "t_stack_range": [0, db.get_nr_of_t_stacks()],
            "z_stack_range": [0, db.get_nr_of_z_stacks()],
            "page": args.page,
            "limit": args.limit,
            "matched": result.source_object_count,
            "rows": rows,
        });
        println!("{}", serde_json::to_string_pretty(&out).unwrap());
        return Ok(());
    }

    println!("Database: {}", args.db.display());
    println!(
        "Images:   {} ({})",
        image_names.len(),
        summarize(&image_names)
    );
    println!(
        "Classes:  {} ({})",
        class_names.len(),
        summarize(&class_names)
    );
    println!("T-stack:  0..{}", db.get_nr_of_t_stacks());
    println!("Z-stack:  0..{}", db.get_nr_of_z_stacks());
    match db.run_status() {
        Ok(status) => {
            if let Some(warning) = status.warning() {
                println!("WARNING:  {warning}");
            }
        }
        Err(e) => eprintln!("Warning: could not read how the analysis ended: {e}"),
    }
    println!();

    if result.rows.is_empty() {
        println!("(no rows match)");
        return Ok(());
    }

    print_object_table(&result);
    println!(
        "\nPage {} - {} row(s) shown (of {} matched). Use --page/--limit to page through more, --channels to add intensities.",
        args.page,
        result.rows.len(),
        result.source_object_count
    );
    Ok(())
}

pub fn run_columns(args: ColumnsArgs, backend: &dyn Backend) -> Result<(), InternalErrors> {
    let db = backend.open_results(&args.db)?;
    let classes = db.get_object_classes()?;
    let columns = db.get_available_columns()?;

    if args.json {
        let out: Vec<_> = columns
            .iter()
            .map(|entry| {
                json!({
                    "id": entry.key.as_key(&classes),
                    "label": entry.display_name,
                    "group": entry.group,
                })
            })
            .collect();
        println!("{}", serde_json::to_string_pretty(&out).unwrap());
        return Ok(());
    }

    println!("{:<28} {:<32} {}", "ID", "LABEL", "GROUP");
    for entry in &columns {
        println!(
            "{:<28} {:<32} {}",
            entry.key.as_key(&classes),
            entry.display_name,
            entry.group
        );
    }
    Ok(())
}

fn summarize(names: &[String]) -> String {
    const MAX_SHOWN: usize = 6;
    if names.len() <= MAX_SHOWN {
        return names.join(", ");
    }
    format!(
        "{}, ... +{} more",
        names[..MAX_SHOWN].join(", "),
        names.len() - MAX_SHOWN
    )
}

#[cfg(test)]
mod tests {
    use evanalyzer_app::backends::local::LocalBackend;

    use super::*;

    fn run(args: ViewArgs) -> Result<(), InternalErrors> {
        super::run(args, &LocalBackend::default())
    }

    fn run_columns(args: ColumnsArgs) -> Result<(), InternalErrors> {
        super::run_columns(args, &LocalBackend::default())
    }
    use crate::args::{ColumnsArgs, FilterArgs, ViewArgs, parse_plane_range};
    use crate::commands::test_support::TempResultsDb;

    fn names(n: usize) -> Vec<String> {
        (0..n).map(|i| format!("img{i}")).collect()
    }

    fn view_args(db: &std::path::Path, json: bool, channels: bool) -> ViewArgs {
        ViewArgs {
            db: db.to_path_buf(),
            page: 0,
            limit: 25,
            channels,
            json,
            filter: FilterArgs::default(),
        }
    }

    /// The seeded database's rows as `view` shows them with `args`: (T, Z,
    /// area) each - img1's object at t0/z0, img2's at t1/z1.
    fn shown(args: &ViewArgs) -> (Vec<String>, Vec<(String, String, String)>) {
        let source = LocalBackend::default().open_results(&args.db).unwrap();
        let base = ListFilter {
            plane: PlaneFilter {
                z_stack: 0,
                t_stack: 0,
            },
            images: None,
            object_classes: None,
            columns: vec![Column::AreaSizePx],
            with_coloc_details: false,
            page: Pagination {
                limit: args.limit as i32,
                after: None,
            },
            transpond_table: false,
        };
        let result = rows_page(source.as_ref(), &base, args).unwrap();
        let rows = result
            .rows
            .iter()
            .map(|row| (cell_text(&row[0]), cell_text(&row[1]), cell_text(row.last().unwrap())))
            .collect();
        (result.column_names, rows)
    }

    fn row(t: &str, z: &str, area: &str) -> (String, String, String) {
        (t.to_string(), z.to_string(), area.to_string())
    }

    /// Every plane by default, each row with its T and Z - not only t0/z0.
    #[test]
    fn view_shows_every_plane_with_its_t_and_z() {
        let db = TempResultsDb::seeded();
        let (header, rows) = shown(&view_args(&db.path, true, false));
        assert_eq!(header[..2], ["T", "Z"]);
        assert_eq!(rows, [row("0", "0", "100"), row("1", "1", "200")]);
    }

    /// `--t`/`--z` choose the planes; `--page` counts rows across them.
    #[test]
    fn view_t_z_and_page_choose_the_rows() {
        let db = TempResultsDb::seeded();
        let mut args = view_args(&db.path, true, false);
        args.filter.t_stacks = Some(parse_plane_range("1").unwrap());
        assert_eq!(shown(&args).1, [row("1", "1", "200")]);

        let mut args = view_args(&db.path, true, false);
        args.filter.z_stacks = Some(parse_plane_range("0-0").unwrap());
        assert_eq!(shown(&args).1, [row("0", "0", "100")]);

        let mut args = view_args(&db.path, true, false);
        args.limit = 1;
        args.page = 1;
        assert_eq!(shown(&args).1, [row("1", "1", "200")], "the second row");
    }

    #[test]
    fn a_plane_range_is_one_number_or_both_ends_included() {
        let range = |text: &str| parse_plane_range(text).map(|r| (r.start, r.end));
        assert_eq!(range("3"), Ok((3, 4)));
        assert_eq!(range("2-5"), Ok((2, 6)));
        assert_eq!(range(" 0 - 1 "), Ok((0, 2)));
        assert!(range("5-2").is_err());
        assert!(range("a").is_err());
    }

    #[test]
    fn run_prints_a_human_readable_page_of_rows_for_a_seeded_database() {
        let db = TempResultsDb::seeded();

        let result = run(view_args(&db.path, false, false));

        assert!(
            result.is_ok(),
            "run failed: {:?}",
            result.err().map(|e| e.to_string())
        );
    }

    #[test]
    fn run_prints_json_for_a_seeded_database() {
        let db = TempResultsDb::seeded();

        let result = run(view_args(&db.path, true, false));

        assert!(
            result.is_ok(),
            "run failed: {:?}",
            result.err().map(|e| e.to_string())
        );
    }

    #[test]
    fn run_reports_no_rows_match_when_the_filter_excludes_every_row() {
        let db = TempResultsDb::seeded();
        let mut args = view_args(&db.path, false, false);
        args.filter.images = vec!["nonexistent.tif".to_string()];

        let result = run(args);

        // An unknown --image name is now rejected up front (see
        // resolve_image_rel_paths) rather than silently matching nothing,
        // so this is an error, not an empty "(no rows match)" success.
        assert!(result.is_err());
    }

    #[test]
    fn run_with_channels_discovers_intensity_columns_from_the_seeded_channel_0_data() {
        let db = TempResultsDb::seeded();

        let result = run(view_args(&db.path, false, true));

        assert!(
            result.is_ok(),
            "run failed: {:?}",
            result.err().map(|e| e.to_string())
        );
    }

    #[test]
    fn run_columns_lists_plain_columns_for_a_seeded_database() {
        let db = TempResultsDb::seeded();

        let result = run_columns(ColumnsArgs {
            db: db.path.clone(),
            json: false,
        });

        assert!(result.is_ok());
    }

    #[test]
    fn run_columns_lists_json_columns_for_a_seeded_database() {
        let db = TempResultsDb::seeded();

        let result = run_columns(ColumnsArgs {
            db: db.path.clone(),
            json: true,
        });

        assert!(result.is_ok());
    }

    #[test]
    fn summarize_empty_list_is_an_empty_string() {
        assert_eq!(summarize(&[]), "");
    }

    #[test]
    fn summarize_lists_every_name_up_to_the_shown_limit() {
        assert_eq!(summarize(&names(6)), "img0, img1, img2, img3, img4, img5");
    }

    #[test]
    fn summarize_truncates_and_counts_the_remainder_past_the_limit() {
        assert_eq!(
            summarize(&names(9)),
            "img0, img1, img2, img3, img4, img5, ... +3 more"
        );
    }
}
