use crate::args::{ColumnsArgs, ViewArgs};
use crate::commands::common::{cell_text, resolve_image_rel_paths, resolve_object_classes};
use crate::table::print_object_table;
use evanalyzer_app::result::{Column, ListFilter, Pagination, PlaneFilter, ResultsGenerator};
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

/// Walks forward from the first page to find the keyset cursor for
/// `target_page` (0-based) — `ListFilter`'s own pagination is keyset-based
/// (a cursor from the *previous* page), not offset-based, so a stateless
/// one-shot CLI invocation has no cursor to resume from and must re-walk
/// every earlier page. Fine for `view`'s "quick preview" use case (small
/// `--page`, typically 0); an unbounded deep `--page` would rescan a lot -
/// use `export` for anything that needs the whole table.
fn cursor_for_page(
    db: &ResultsGenerator,
    base: &ListFilter,
    target_page: usize,
) -> Result<Option<String>, InternalErrors> {
    let mut cursor = None;
    for _ in 0..target_page {
        let probe = ListFilter {
            page: Pagination {
                limit: base.page.limit,
                after: cursor.take(),
            },
            ..base.clone()
        };
        let page = db.get_object_list(&probe)?;
        cursor = page.row_names.last().cloned();
        if cursor.is_none() {
            break;
        }
    }
    Ok(cursor)
}

pub fn run(args: ViewArgs) -> Result<(), InternalErrors> {
    let db = ResultsGenerator::open_database(args.db.clone())?;

    if args.filter.colocalized.is_some() {
        return Err(InternalErrors::InvalidArgument(
            "--colocalized isn't supported by the current results backend".to_string(),
        ));
    }

    let images = db.get_images()?;
    let classes = db.get_object_classes()?;
    let image_names: Vec<String> = images.iter().map(|image| image.name.clone()).collect();
    let class_names: Vec<String> = classes.iter().map(|class| class.name.clone()).collect();

    let image_rel_paths = resolve_image_rel_paths(&db, &args.filter.images)?;
    let object_classes = resolve_object_classes(&db, &args.filter.classes)?;
    let columns: Vec<Column> = db
        .get_available_columns()?
        .into_iter()
        .filter(|entry| args.channels || !is_intensity_column(&entry.key))
        .map(|entry| entry.key)
        .collect();

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
            limit: args.limit.max(1) as i32,
            after: None,
        },
    };
    let cursor = cursor_for_page(&db, &base_filter, args.page)?;
    let result = db.get_object_list(&ListFilter {
        page: Pagination {
            limit: base_filter.page.limit,
            after: cursor,
        },
        ..base_filter
    })?;

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

pub fn run_columns(args: ColumnsArgs) -> Result<(), InternalErrors> {
    let db = ResultsGenerator::open_database(args.db.clone())?;
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
    use super::*;
    use crate::args::{ColumnsArgs, FilterArgs, ViewArgs};
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

    #[test]
    fn run_prints_a_human_readable_page_of_rows_for_a_seeded_database() {
        let db = TempResultsDb::seeded();

        let result = run(view_args(&db.path, false, false));

        assert!(result.is_ok(), "run failed: {:?}", result.err().map(|e| e.to_string()));
    }

    #[test]
    fn run_prints_json_for_a_seeded_database() {
        let db = TempResultsDb::seeded();

        let result = run(view_args(&db.path, true, false));

        assert!(result.is_ok(), "run failed: {:?}", result.err().map(|e| e.to_string()));
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

        assert!(result.is_ok(), "run failed: {:?}", result.err().map(|e| e.to_string()));
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
