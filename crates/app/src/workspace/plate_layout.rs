//! The project's images on a plate - which well each image belongs to, by
//! the project's plate settings, before anything is analysed (the images
//! panel's plate view). The same grouping the results queries use.

use crate::backend::local::results::{
    AUTO_GROUPING_PATTERNS, FOLDER_GROUPING_REGEX, best_matching_dimensions, col_number_to_index,
    row_index_to_letter, row_letter_to_index,
};
use evanalyzer_cfg::settings::plate_settings::{GroupingMode, PlateSettings};
use regex::Regex;
use std::collections::BTreeMap;
use std::path::Path;

/// The plate: its size and the wells that have images.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct PlateLayout {
    pub rows: usize,
    pub cols: usize,
    /// Row by row; only wells with images.
    pub wells: Vec<WellImages>,
    /// Images no well was found for (or outside the plate size).
    pub unplaced: usize,
}

/// A well and its images.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct WellImages {
    /// 0-based.
    pub row: usize,
    pub col: usize,
    /// "C3".
    pub name: String,
    /// Indices into the image list given, by image number in the well.
    pub images: Vec<usize>,
}

/// Groups `images` (paths relative to the image folder, `/` or `\`
/// separated) into wells as `plate` says.
pub fn plate_layout(images: &[&Path], plate: &PlateSettings) -> PlateLayout {
    let folder = plate.grouping_mode == GroupingMode::Folder;
    let texts: Vec<String> = images
        .iter()
        .map(|path| {
            if folder {
                path.to_string_lossy().into_owned()
            } else {
                path.file_name()
                    .map(|name| name.to_string_lossy().into_owned())
                    .unwrap_or_default()
            }
        })
        .collect();
    let Some(regex) = grouping_regex(plate, &texts) else {
        return PlateLayout {
            unplaced: images.len(),
            ..Default::default()
        };
    };

    // (row, col) -> [(image number, index)]
    let mut wells: BTreeMap<(usize, usize), Vec<(u64, usize)>> = BTreeMap::new();
    let mut unplaced = 0;
    for (index, text) in texts.iter().enumerate() {
        let place = regex.captures(text).and_then(|caps| {
            let row = row_letter_to_index(caps.get(2)?.as_str())?;
            let col = col_number_to_index(caps.get(3)?.as_str())?;
            let number = caps
                .get(4)
                .and_then(|m| m.as_str().parse::<u64>().ok())
                .unwrap_or(0);
            Some(((row, col), number))
        });
        match place {
            Some((well, number)) => wells.entry(well).or_default().push((number, index)),
            None => unplaced += 1,
        }
    }

    let (rows, cols) = plate.plate_size.dimensions().unwrap_or_else(|| {
        best_matching_dimensions(
            wells.keys().map(|(row, _)| *row).max(),
            wells.keys().map(|(_, col)| *col).max(),
        )
    });
    let mut placed = Vec::with_capacity(wells.len());
    for ((row, col), mut members) in wells {
        if row >= rows || col >= cols {
            unplaced += members.len();
            continue;
        }
        members.sort();
        placed.push(WellImages {
            row,
            col,
            name: format!("{}{}", row_index_to_letter(row), col + 1),
            images: members.into_iter().map(|(_, index)| index).collect(),
        });
    }
    PlateLayout {
        rows,
        cols,
        wells: placed,
        unplaced,
    }
}

/// The indices of `texts` (image names, or relative paths for the folder
/// grouping) whose regex `part` (1 the well, 2 its row, 3 its column - see
/// `PlateSettings::grouping_regex`) is `label`. Nothing for a regex that
/// doesn't compile.
pub fn texts_in_group(texts: &[&str], regex: &str, part: usize, label: &str) -> Vec<usize> {
    let Ok(regex) = Regex::new(regex) else {
        return Vec::new();
    };
    texts
        .iter()
        .enumerate()
        .filter(|(_, text)| {
            regex
                .captures(text)
                .and_then(|caps| caps.get(part))
                .is_some_and(|m| m.as_str() == label)
        })
        .map(|(i, _)| i)
        .collect()
}

/// The regex for `plate`: the custom one, the folder one, or for Auto the
/// first of the common patterns that matches the most of `texts`. `None`
/// for a custom regex that doesn't compile.
fn grouping_regex(plate: &PlateSettings, texts: &[String]) -> Option<Regex> {
    match plate.grouping_mode {
        GroupingMode::Custom if !plate.grouping_regex.trim().is_empty() => {
            Regex::new(&plate.grouping_regex).ok()
        }
        GroupingMode::Folder => Regex::new(FOLDER_GROUPING_REGEX).ok(),
        GroupingMode::Custom | GroupingMode::Auto => {
            let candidates: Vec<Regex> = AUTO_GROUPING_PATTERNS
                .iter()
                .filter_map(|pattern| Regex::new(pattern).ok())
                .collect();
            let mut best = 0;
            let mut best_count = 0;
            for (i, regex) in candidates.iter().enumerate() {
                let count = texts.iter().filter(|text| regex.is_match(text)).count();
                if count > best_count {
                    best = i;
                    best_count = count;
                }
            }
            candidates.into_iter().nth(best)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use evanalyzer_cfg::settings::plate_settings::PlateSize;
    use std::path::PathBuf;

    fn layout(names: &[&str], plate: &PlateSettings) -> PlateLayout {
        let paths: Vec<PathBuf> = names.iter().map(PathBuf::from).collect();
        let refs: Vec<&Path> = paths.iter().map(PathBuf::as_path).collect();
        plate_layout(&refs, plate)
    }

    #[test]
    fn auto_groups_by_the_well_in_the_name_and_sorts_by_image_number() {
        let plate = PlateSettings::default();
        let got = layout(
            &["B03_02.tif", "A01_01.tif", "B03_01.tif", "notes.tif"],
            &plate,
        );
        assert_eq!(
            (got.rows, got.cols),
            (2, 3),
            "smallest standard plate holding B3"
        );
        assert_eq!(got.unplaced, 1);
        assert_eq!(got.wells.len(), 2);
        assert_eq!(got.wells[0].name, "A1");
        assert_eq!(got.wells[0].images, vec![1]);
        assert_eq!(got.wells[1].name, "B3");
        assert_eq!((got.wells[1].row, got.wells[1].col), (1, 2));
        assert_eq!(got.wells[1].images, vec![2, 0], "image 01 before 02");
    }

    #[test]
    fn auto_uses_only_the_file_name() {
        let got = layout(&["plate1/C12_01.tif"], &PlateSettings::default());
        assert_eq!(got.wells[0].name, "C12");
    }

    #[test]
    fn a_fixed_plate_size_wins_and_wells_outside_it_are_unplaced() {
        let plate = PlateSettings {
            plate_size: PlateSize::Plate2x3,
            ..Default::default()
        };
        let got = layout(&["A01_01.tif", "H12_01.tif"], &plate);
        assert_eq!((got.rows, got.cols), (2, 3));
        assert_eq!(got.wells.len(), 1);
        assert_eq!(got.unplaced, 1);
    }

    #[test]
    fn folder_mode_takes_the_folder_as_the_well() {
        let plate = PlateSettings {
            grouping_mode: GroupingMode::Folder,
            ..Default::default()
        };
        let got = layout(&["B02/img_3.tif", "B02/img_1.tif", "misc/x.tif"], &plate);
        assert_eq!(got.wells.len(), 1);
        assert_eq!(got.wells[0].name, "B2");
        assert_eq!(got.wells[0].images, vec![1, 0]);
        assert_eq!(
            got.unplaced, 1,
            "a folder not named like a well isn't on the plate"
        );
    }

    #[test]
    fn a_custom_regex_that_does_not_compile_places_nothing() {
        let plate = PlateSettings {
            grouping_mode: GroupingMode::Custom,
            grouping_regex: "((".into(),
            ..Default::default()
        };
        let got = layout(&["A01_01.tif"], &plate);
        assert_eq!(got.wells.len(), 0);
        assert_eq!(got.unplaced, 1);
    }

    #[test]
    fn texts_in_group_picks_the_images_of_a_row_column_or_well() {
        let regex = AUTO_GROUPING_PATTERNS[0];
        let names = ["A01_01.tif", "A02_01.tif", "B01_01.tif"];
        assert_eq!(texts_in_group(&names, regex, 2, "A"), vec![0, 1]);
        assert_eq!(texts_in_group(&names, regex, 3, "01"), vec![0, 2]);
        assert_eq!(texts_in_group(&names, regex, 1, "B01"), vec![2]);
        assert!(texts_in_group(&names, "((", 1, "A01").is_empty());
    }

    #[test]
    fn no_images_is_an_empty_smallest_plate() {
        let got = layout(&[], &PlateSettings::default());
        assert_eq!((got.rows, got.cols), (2, 3));
        assert!(got.wells.is_empty());
    }
}
