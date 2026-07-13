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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::args::GroupByKind;

    #[test]
    fn database_filter_leaves_image_and_class_filters_unset_when_args_are_empty() {
        let filter = build_database_filter(&FilterArgs::default(), true, 2, 50);

        assert_eq!(filter.image_filter, None);
        assert_eq!(filter.class_filter, None);
        assert_eq!(filter.coloc_filter, None);
        assert_eq!(filter.page, 2);
        assert_eq!(filter.page_size, 50);
        assert!(filter.needs_intensities);
    }

    #[test]
    fn database_filter_forwards_image_and_class_lists_when_present() {
        let args = FilterArgs {
            images: vec!["a.tif".to_string(), "b.tif".to_string()],
            classes: vec!["Nucleus".to_string()],
            colocalized: None,
        };

        let filter = build_database_filter(&args, false, 0, 0);

        assert_eq!(
            filter.image_filter,
            Some(vec!["a.tif".to_string(), "b.tif".to_string()])
        );
        assert_eq!(filter.class_filter, Some(vec!["Nucleus".to_string()]));
        assert!(!filter.needs_intensities);
    }

    #[test]
    fn database_filter_translates_colocalized_true_and_false_to_yes_no_strings() {
        let yes = FilterArgs {
            colocalized: Some(true),
            ..Default::default()
        };
        let no = FilterArgs {
            colocalized: Some(false),
            ..Default::default()
        };

        assert_eq!(
            build_database_filter(&yes, true, 0, 0).coloc_filter,
            Some(vec!["Yes".to_string()])
        );
        assert_eq!(
            build_database_filter(&no, true, 0, 0).coloc_filter,
            Some(vec!["No".to_string()])
        );
    }

    #[test]
    fn group_config_defaults_to_group_by_none_and_avg_when_no_agg_given() {
        let config = build_group_config(&GroupArgs::default());

        assert_eq!(config.group_by, GroupBy::None);
        assert_eq!(config.aggs, vec![AggFunc::Avg]);
        assert_eq!(config.regex, "");
        assert!(!config.split_colocalized);
        assert!(!config.group_by_class);
    }

    #[test]
    fn group_config_maps_every_group_by_kind_to_its_groupby_variant() {
        let by = |kind| {
            build_group_config(&GroupArgs {
                group_by: Some(kind),
                ..Default::default()
            })
            .group_by
        };

        assert_eq!(by(GroupByKind::Image), GroupBy::Image);
        assert_eq!(by(GroupByKind::Folder), GroupBy::Folder);
        assert_eq!(by(GroupByKind::Regex), GroupBy::Regex);
    }

    #[test]
    fn group_config_maps_every_agg_kind_and_preserves_given_order() {
        let args = GroupArgs {
            agg: vec![
                AggKind::Min,
                AggKind::Max,
                AggKind::Avg,
                AggKind::Median,
                AggKind::Stdev,
                AggKind::Sum,
            ],
            ..Default::default()
        };

        let config = build_group_config(&args);

        assert_eq!(
            config.aggs,
            vec![
                AggFunc::Min,
                AggFunc::Max,
                AggFunc::Avg,
                AggFunc::Median,
                AggFunc::Stdev,
                AggFunc::Sum,
            ]
        );
    }

    #[test]
    fn group_config_forwards_regex_and_split_and_group_by_class_flags() {
        let args = GroupArgs {
            group_regex: Some("well_(\\w+)".to_string()),
            split_colocalized: true,
            group_by_class: true,
            ..Default::default()
        };

        let config = build_group_config(&args);

        assert_eq!(config.regex, "well_(\\w+)");
        assert!(config.split_colocalized);
        assert!(config.group_by_class);
    }
}
