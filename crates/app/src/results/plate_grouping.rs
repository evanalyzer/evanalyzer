//! # plate_grouping
//!
//! How images are grouped into the wells of a plate: the well-name
//! patterns, the folder rule, the plate size that fits, row letters and
//! column numbers - the same for the results queries and for the project's
//! plate view before any analysis.
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-11-22
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.

use crate::results::PlateSize;

/// The well-name patterns `Grouping::Auto` chooses from, in order of
/// preference (the first wins a tie). Each captures 1 the well (`B03`), 2 its
/// row (`B`), 3 its column (`03`) and 4 the image number in the well (`01`,
/// empty for one image per well).
pub(crate) const AUTO_GROUPING_PATTERNS: [&str; 4] = [
    // A01_01.tif, B3-s2.tif, C12 f003.ome.tif
    r"^(([A-Za-z]{1,2})([0-9]{1,3}))[_\- ]+[A-Za-z]*([0-9]+)\.[A-Za-z0-9.]+$",
    // Exp1_B03_01.tif, plate 2 - Well A1 - field 3.tif
    r"^.*[_\- ](([A-Za-z]{1,2})([0-9]{1,3}))[_\- ]+[A-Za-z]*([0-9]+)\.[A-Za-z0-9.]+$",
    // A01.tif (one image per well)
    r"^(([A-Za-z]{1,2})([0-9]{1,3}))()\.[A-Za-z0-9.]+$",
    // Exp1_B03.tif (one image per well)
    r"^.*[_\- ](([A-Za-z]{1,2})([0-9]{1,3}))()\.[A-Za-z0-9.]+$",
];

/// `Grouping::Folder`, on `image_rel_path` (`/` or `\` separated): the
/// folder an image is in is its well, with a row and column when the folder
/// is named like a well (`B03`); the image number is the last number in the
/// file name.
pub(crate) const FOLDER_GROUPING_REGEX: &str = r"^(?:.*[/\\])?(([A-Za-z]{1,2})([0-9]{1,3})|[^/\\]+)[/\\](?:[^/\\]*?([0-9]+))?[^/\\0-9]*\.[^/\\.]+$";

/// The plate sizes `PlateSize::Auto` chooses from (the standard formats),
/// smallest first - `best_matching_dimensions` relies on this order to find
/// the smallest one that fits.
pub(crate) const AUTO_PLATE_SIZES: [PlateSize; 8] = [
    PlateSize::Plate2x3,
    PlateSize::Plate3x4,
    PlateSize::Plate4x6,
    PlateSize::Plate6x8,
    PlateSize::Plate8x12,
    PlateSize::Plate16x24,
    PlateSize::Plate32x48,
    PlateSize::Plate48x72,
];

/// The smallest standard plate size whose row/column count covers every well
/// this query actually found (`max_row`/`max_col`, both 0-based), as
/// `(rows, cols)`. Falls back to the largest known size if even that doesn't
/// fit (a plate bigger than any standard format, or a grouping extracting
/// something that isn't really a well id).
pub(crate) fn best_matching_dimensions(
    max_row: Option<usize>,
    max_col: Option<usize>,
) -> (usize, usize) {
    let needed_rows = max_row.map_or(1, |row| row + 1);
    let needed_cols = max_col.map_or(1, |col| col + 1);
    AUTO_PLATE_SIZES
        .into_iter()
        .filter_map(PlateSize::dimensions)
        .find(|(rows, cols)| *rows >= needed_rows && *cols >= needed_cols)
        .unwrap_or((48, 72))
}

/// Parses a well's row letters ("A", "B", ..., "Z", "AA", "AB", ...) into a
/// 0-based row index, using the same bijective base-26 scheme spreadsheet
/// column letters use. `None` if `letters` isn't purely alphabetic (e.g. the
/// `grouping_regex` didn't actually match a well id).
pub(crate) fn row_letter_to_index(letters: &str) -> Option<usize> {
    if letters.is_empty() || !letters.chars().all(|c| c.is_ascii_alphabetic()) {
        return None;
    }
    let mut index: usize = 0;
    for c in letters.chars() {
        let digit = (c.to_ascii_uppercase() as u8 - b'A') as usize + 1;
        index = index * 26 + digit;
    }
    Some(index - 1)
}

/// Inverse of [`row_letter_to_index`].
pub(crate) fn row_index_to_letter(index: usize) -> String {
    let mut n = index + 1;
    let mut letters = Vec::new();
    while n > 0 {
        let rem = (n - 1) % 26;
        letters.push((b'A' + rem as u8) as char);
        n = (n - 1) / 26;
    }
    letters.iter().rev().collect()
}

/// Parses a well's column number ("1", "2", ...) into a 0-based column
/// index. `None` if `digits` isn't a positive integer.
pub(crate) fn col_number_to_index(digits: &str) -> Option<usize> {
    digits.parse::<usize>().ok()?.checked_sub(1)
}
