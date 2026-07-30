use crate::args::{ColumnsArgs, ViewArgs};
use crate::commands::common::{build_database_filter, discover_columns};
use crate::table::print_object_table;
use evanalyzer_app::result::{
    ResultsLoader, build_column_specs, discover_channels, plottable_columns, to_display_row,
};
use evanalyzer_cfg::core_types::InternalErrors;
use serde_json::json;

pub fn run(args: ViewArgs) -> Result<(), InternalErrors> {
    let loader = ResultsLoader::new(&args.db);

    let image_names = loader.get_image_names()?;
    let class_names = loader.get_class_names()?;
    let t_range = loader.get_t_stack_range().unwrap_or(None);
    let z_range = loader.get_z_stack_range().unwrap_or(None);

    let filter = build_database_filter(&args.filter, args.channels, args.page, args.limit);
    let objects = loader.get_objects(filter)?;

    let (channels, coloc_partner_classes) = if args.channels {
        (discover_channels(&objects), loader.get_coloc_partner_class_names()?)
    } else {
        (vec![], vec![])
    };
    let specs = build_column_specs(&channels, &coloc_partner_classes);

    if args.json {
        let rows: Vec<_> = objects
            .iter()
            .enumerate()
            .map(|(i, object)| {
                let display = to_display_row(i, object, &specs);
                serde_json::Value::Object(
                    specs
                        .iter()
                        .zip(display.values.iter())
                        .map(|(spec, v)| (spec.id.clone(), json!(v)))
                        .collect(),
                )
            })
            .collect();
        let out = json!({
            "db": args.db,
            "images": image_names,
            "classes": class_names,
            "t_stack_range": t_range,
            "z_stack_range": z_range,
            "page": args.page,
            "limit": args.limit,
            "rows": rows,
        });
        println!("{}", serde_json::to_string_pretty(&out).unwrap());
        return Ok(());
    }

    println!("Database: {}", args.db.display());
    println!("Images:   {} ({})", image_names.len(), summarize(&image_names));
    println!("Classes:  {} ({})", class_names.len(), summarize(&class_names));
    if let Some((min, max)) = t_range {
        println!("T-stack:  {min}..{max}");
    }
    if let Some((min, max)) = z_range {
        println!("Z-stack:  {min}..{max}");
    }
    println!();

    if objects.is_empty() {
        println!("(no rows match)");
        return Ok(());
    }

    print_object_table(&specs, &objects);
    println!(
        "\nPage {} - {} row(s) shown. Use --page/--limit to page through more, --channels to add intensities.",
        args.page,
        objects.len()
    );
    Ok(())
}

pub fn run_columns(args: ColumnsArgs) -> Result<(), InternalErrors> {
    let loader = ResultsLoader::new(&args.db);
    let specs = discover_columns(&loader)?;
    let plottable: std::collections::HashSet<&str> =
        plottable_columns(&specs).iter().map(|c| c.id.as_str()).collect();

    if args.json {
        let out: Vec<_> = specs
            .iter()
            .map(|c| {
                json!({
                    "id": c.id,
                    "label": c.label,
                    "numeric": plottable.contains(c.id.as_str()),
                })
            })
            .collect();
        println!("{}", serde_json::to_string_pretty(&out).unwrap());
        return Ok(());
    }

    println!("{:<36} {:<32} {}", "ID", "LABEL", "NUMERIC");
    for c in &specs {
        let numeric = if plottable.contains(c.id.as_str()) { "yes" } else { "" };
        println!("{:<36} {:<32} {numeric}", c.id, c.label);
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

        assert!(result.is_ok());
    }

    #[test]
    fn run_prints_json_for_a_seeded_database() {
        let db = TempResultsDb::seeded();

        let result = run(view_args(&db.path, true, false));

        assert!(result.is_ok());
    }

    #[test]
    fn run_reports_no_rows_match_when_the_filter_excludes_every_row() {
        let db = TempResultsDb::seeded();
        let mut args = view_args(&db.path, false, false);
        args.filter.images = vec!["nonexistent.tif".to_string()];

        // Exercise the same lookup `run` uses to decide it hit the
        // "(no rows match)" branch, so the test doesn't just take `is_ok()`
        // on faith - it proves the fixture+filter combination really does
        // produce zero rows before checking `run` handles that cleanly.
        let loader = ResultsLoader::new(&db.path);
        let filter = build_database_filter(&args.filter, args.channels, args.page, args.limit);
        let objects = loader.get_objects(filter).expect("get_objects");
        assert!(objects.is_empty());

        let result = run(args);

        assert!(result.is_ok());
    }

    #[test]
    fn run_with_channels_discovers_intensity_columns_from_the_seeded_channel_0_data() {
        let db = TempResultsDb::seeded();

        // Same discovery path `run` takes internally when `--channels` is
        // set: fetch the (intensity-carrying) objects, then discover which
        // channel indices actually appear.
        let loader = ResultsLoader::new(&db.path);
        let filter = build_database_filter(&FilterArgs::default(), true, 0, 25);
        let objects = loader.get_objects(filter).expect("get_objects");
        let channels = discover_channels(&objects);
        assert_eq!(channels, vec![0]);

        let result = run(view_args(&db.path, false, true));

        assert!(result.is_ok());
    }

    #[test]
    fn run_columns_lists_plain_columns_for_a_seeded_database() {
        let db = TempResultsDb::seeded();

        let result = run_columns(ColumnsArgs { db: db.path.clone(), json: false });

        assert!(result.is_ok());
    }

    #[test]
    fn run_columns_lists_json_columns_for_a_seeded_database() {
        let db = TempResultsDb::seeded();

        // Same column-discovery call `run_columns` makes internally - assert
        // the fixed `object_id`/`image`/`class` columns and the seeded
        // channel-0 intensity column are all present, and that at least one
        // of them is flagged plottable/numeric like the JSON output does.
        let loader = ResultsLoader::new(&db.path);
        let specs = discover_columns(&loader).expect("discover_columns");
        let ids: Vec<&str> = specs.iter().map(|c| c.id.as_str()).collect();
        assert!(ids.contains(&"object_id"));
        assert!(ids.contains(&"image"));
        assert!(ids.contains(&"class"));
        assert!(ids.iter().any(|id| id.starts_with("ch0_")));

        let plottable: std::collections::HashSet<&str> =
            plottable_columns(&specs).iter().map(|c| c.id.as_str()).collect();
        assert!(!plottable.is_empty());

        let result = run_columns(ColumnsArgs { db: db.path.clone(), json: true });

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
