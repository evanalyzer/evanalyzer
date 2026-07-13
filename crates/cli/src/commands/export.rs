use crate::args::{
    ChartKind, ColorByKind, ExportArgs, ExportCommand, HeatmapArgs, HistogramArgs, ScatterArgs,
    TableExportArgs,
};
use crate::commands::common::{build_database_filter, build_group_config, discover_columns};
use evanalyzer_app::result::{
    ColorBy, HeatmapColorScheme, HeatmapMetric, ResultsExporter, ResultsLoader, compute_heatmap,
    compute_histogram, compute_scatter, save_heatmap_png, save_histogram_png, save_scatter_png,
};
use evanalyzer_cfg::core_types::InternalErrors;
use std::sync::Arc;

pub fn run(args: ExportArgs) -> Result<(), InternalErrors> {
    match args.command {
        ExportCommand::Csv(table) => export_table(table, true),
        ExportCommand::Xlsx(table) => export_table(table, false),
        ExportCommand::Chart(chart) => match chart.kind {
            ChartKind::Histogram(a) => export_histogram(a),
            ChartKind::Scatter(a) => export_scatter(a),
            ChartKind::Heatmap(a) => export_heatmap(a),
        },
    }
}

fn export_table(args: TableExportArgs, csv: bool) -> Result<(), InternalErrors> {
    let loader = Arc::new(ResultsLoader::new(&args.db));
    let specs = discover_columns(&loader)?;
    let filter = build_database_filter(&args.filter, true, 0, 0);
    let group = build_group_config(&args.group);

    let exporter = ResultsExporter::new(loader);
    if csv {
        exporter.export_to_csv(filter, &group, &specs, &args.out)?;
    } else {
        exporter.export_to_xlsx(filter, &group, &specs, &args.out)?;
    }
    println!("Exported to {}", args.out.display());
    Ok(())
}

fn export_histogram(args: HistogramArgs) -> Result<(), InternalErrors> {
    let loader = ResultsLoader::new(&args.db);
    let specs = discover_columns(&loader)?;
    let filter = build_database_filter(&args.filter, true, 0, 0);
    let rois = loader.get_rois(filter)?;

    let data = compute_histogram(
        &rois,
        &args.column,
        &specs,
        args.buckets,
        args.log_scale,
        to_color_by(args.color_by),
    )
    .ok_or_else(|| column_not_found_error(&args.column, &args.db))?;
    save_histogram_png(&data, args.width, args.height, &args.out)?;
    println!("Saved histogram to {}", args.out.display());
    Ok(())
}

fn export_scatter(args: ScatterArgs) -> Result<(), InternalErrors> {
    let loader = ResultsLoader::new(&args.db);
    let specs = discover_columns(&loader)?;
    let filter = build_database_filter(&args.filter, true, 0, 0);
    let rois = loader.get_rois(filter)?;

    let data =
        compute_scatter(&rois, &args.x, &args.y, to_color_by(args.color_by), &specs, args.max_points)
        .ok_or_else(|| column_not_found_error(&format!("{} / {}", args.x, args.y), &args.db))?;
    save_scatter_png(&data, args.width, args.height, &args.out)?;
    println!("Saved scatter plot to {}", args.out.display());
    Ok(())
}

fn export_heatmap(args: HeatmapArgs) -> Result<(), InternalErrors> {
    let loader = ResultsLoader::new(&args.db);
    let specs = discover_columns(&loader)?;
    let filter = build_database_filter(&args.filter, true, 0, 0);
    let rois = loader.get_rois(filter)?;

    let metric = if args.metric == "count" {
        HeatmapMetric::Count
    } else {
        HeatmapMetric::Average(args.metric.clone())
    };

    let data = compute_heatmap(&rois, &metric, &specs, args.cell_size)
        .ok_or_else(|| column_not_found_error(&args.metric, &args.db))?;
    let scheme = HeatmapColorScheme::from_label(&args.color_scheme);
    save_heatmap_png(&data, scheme, args.width, args.height, &args.out)?;
    println!("Saved heatmap to {}", args.out.display());
    Ok(())
}

fn to_color_by(kind: ColorByKind) -> ColorBy {
    match kind {
        ColorByKind::None => ColorBy::None,
        ColorByKind::Class => ColorBy::Class,
        ColorByKind::Colocalized => ColorBy::Colocalized,
    }
}

fn column_not_found_error(column: &str, db: &std::path::Path) -> InternalErrors {
    InternalErrors::InvalidArgument(format!(
        "No data for column '{column}': check it exists and matches the active filter \
         (see `evanalyzer cli columns --db {}`)",
        db.display()
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn to_color_by_maps_every_kind_to_its_matching_variant() {
        assert!(matches!(to_color_by(ColorByKind::None), ColorBy::None));
        assert!(matches!(to_color_by(ColorByKind::Class), ColorBy::Class));
        assert!(matches!(
            to_color_by(ColorByKind::Colocalized),
            ColorBy::Colocalized
        ));
    }

    #[test]
    fn column_not_found_error_names_the_column_and_db_path_in_the_message() {
        let err = column_not_found_error("area", std::path::Path::new("/tmp/results.evadb"));

        let InternalErrors::InvalidArgument(msg) = err else {
            panic!("expected InvalidArgument, got {err:?}");
        };
        assert!(msg.contains("area"));
        assert!(msg.contains("/tmp/results.evadb"));
    }
}
