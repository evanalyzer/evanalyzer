use crate::args::{ColumnsArgs, ViewArgs};
use crate::commands::common::{build_database_filter, discover_columns};
use crate::table::print_roi_table;
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
    let rois = loader.get_rois(filter)?;

    let (channels, coloc_partner_classes) = if args.channels {
        (discover_channels(&rois), loader.get_coloc_partner_class_names()?)
    } else {
        (vec![], vec![])
    };
    let specs = build_column_specs(&channels, &coloc_partner_classes);

    if args.json {
        let rows: Vec<_> = rois
            .iter()
            .enumerate()
            .map(|(i, roi)| {
                let display = to_display_row(i, roi, &specs);
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

    if rois.is_empty() {
        println!("(no rows match)");
        return Ok(());
    }

    print_roi_table(&specs, &rois);
    println!(
        "\nPage {} - {} row(s) shown. Use --page/--limit to page through more, --channels to add intensities.",
        args.page,
        rois.len()
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

    fn names(n: usize) -> Vec<String> {
        (0..n).map(|i| format!("img{i}")).collect()
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
