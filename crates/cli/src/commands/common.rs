use crate::args::{AggKind, FilterArgs, GroupArgs, GroupByKind};
use evanalyzer_app::result::{
    AggFunc, ColumnSpec, DatabaseFilter, GroupBy, GroupConfig, ResultsLoader, build_column_specs,
    discover_channels,
};
use evanalyzer_cfg::core_types::InternalErrors;

/// Rows sampled to discover which channel/colocalization-partner columns exist.
/// Mirrors the GUI's first-page sample (`results_table_controller.rs`) rather than
/// scanning the whole table, since this only needs to know which columns *exist*.
const COLUMN_DISCOVERY_SAMPLE: usize = 2000;

pub fn build_database_filter(
    filter: &FilterArgs,
    needs_intensities: bool,
    page: usize,
    page_size: usize,
) -> DatabaseFilter {
    DatabaseFilter {
        image_filter: (!filter.images.is_empty()).then(|| filter.images.clone()),
        class_filter: (!filter.classes.is_empty()).then(|| filter.classes.clone()),
        coloc_filter: filter
            .colocalized
            .map(|yes| vec![if yes { "Yes" } else { "No" }.to_string()]),
        page_size,
        page,
        needs_intensities,
        ..Default::default()
    }
}

pub fn build_group_config(args: &GroupArgs) -> GroupConfig {
    let group_by = match args.group_by {
        None => GroupBy::None,
        Some(GroupByKind::Image) => GroupBy::Image,
        Some(GroupByKind::Folder) => GroupBy::Folder,
        Some(GroupByKind::Regex) => GroupBy::Regex,
    };
    let aggs: Vec<AggFunc> = args
        .agg
        .iter()
        .map(|a| match a {
            AggKind::Min => AggFunc::Min,
            AggKind::Max => AggFunc::Max,
            AggKind::Avg => AggFunc::Avg,
            AggKind::Median => AggFunc::Median,
            AggKind::Stdev => AggFunc::Stdev,
            AggKind::Sum => AggFunc::Sum,
        })
        .collect();
    GroupConfig {
        group_by,
        regex: args.group_regex.clone().unwrap_or_default(),
        aggs: if aggs.is_empty() { vec![AggFunc::Avg] } else { aggs },
        split_colocalized: args.split_colocalized,
        group_by_class: args.group_by_class,
    }
}

/// Discovers the full set of columns (fixed + per-channel + coloc-partner) present
/// in `loader`'s database, for use as `ResultsExporter`/`compute_*` chart inputs.
pub fn discover_columns(loader: &ResultsLoader) -> Result<Vec<ColumnSpec>, InternalErrors> {
    let sample = loader.get_rois(DatabaseFilter {
        page_size: COLUMN_DISCOVERY_SAMPLE,
        ..Default::default()
    })?;
    let channels = discover_channels(&sample);
    let coloc_partner_classes = loader.get_coloc_partner_class_names()?;
    Ok(build_column_specs(&channels, &coloc_partner_classes))
}
