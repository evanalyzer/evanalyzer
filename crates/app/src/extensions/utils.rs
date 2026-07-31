use pathdiff::diff_paths;
use std::{
    fs,
    path::{Path, PathBuf},
};

pub fn get_relative_key(image_path: &Path, images_root: Option<&PathBuf>) -> Option<PathBuf> {
    match images_root {
        Some(root) => diff_paths(image_path, root),
        None => Some(image_path.to_path_buf()),
    }
}

pub fn get_file_size(path: &Path) -> std::io::Result<u64> {
    let metadata = fs::metadata(path)?;
    Ok(metadata.len())
}

pub fn is_in_root(image_path: &Path, data_root: &Path) -> bool {
    // Basic check: Does the image path begin with the data_root string?
    image_path.starts_with(data_root)
}

pub fn wavelength_to_rgb_u32(wavelength: f32) -> u32 {
    let color = wavelength_to_rgb_float(wavelength);
    let ret_color: u32 = ((color[0] * 255.0) as u32) << 16
        | ((color[1] * 255.0) as u32) << 8
        | (color[2] * 255.0) as u32;
    ret_color
}

/// Converts a wavelength in nm to an RGB [f32; 3] color.
/// Returns [0.0, 0.0, 0.0] if the wavelength is outside the visible spectrum.
pub fn wavelength_to_rgb_float(wavelength: f32) -> [f32; 3] {
    let (mut r, mut g, mut b) = (0.0, 0.0, 0.0);

    // Images without given emission wave length have default value 0.
    // In this case we show a grayscale value. Because of float we assume < 1
    if wavelength <= 1.0 {
        return [1.0, 1.0, 1.0];
    }

    // Pure red
    if wavelength == 635.0 {
        return [1.0, 0.0, 0.0];
    }

    // Pure green
    if wavelength == 532.0 {
        return [0.0, 1.0, 0.0];
    }

    // Pure blue
    if wavelength == 450.0 {
        return [0.0, 0.0, 1.0];
    }

    // Calculate base RGB components
    if (380.0..440.0).contains(&wavelength) {
        r = -(wavelength - 440.0) / (440.0 - 380.0);
        b = 1.0;
    } else if (440.0..490.0).contains(&wavelength) {
        g = (wavelength - 440.0) / (490.0 - 440.0);
        b = 1.0;
    } else if (490.0..510.0).contains(&wavelength) {
        g = 1.0;
        b = -(wavelength - 510.0) / (510.0 - 490.0);
    } else if (510.0..580.0).contains(&wavelength) {
        r = (wavelength - 510.0) / (580.0 - 510.0);
        g = 1.0;
    } else if (580.0..645.0).contains(&wavelength) {
        r = 1.0;
        g = -(wavelength - 645.0) / (645.0 - 580.0);
    } else if (645.0..781.0).contains(&wavelength) {
        r = 1.0;
    }

    // Factor for intensity fade-out at the edges of the spectrum
    let factor = if (380.0..420.0).contains(&wavelength) {
        0.3 + 0.7 * (wavelength - 380.0) / (420.0 - 380.0)
    } else if (420.0..701.0).contains(&wavelength) {
        1.0
    } else if (701.0..781.0).contains(&wavelength) {
        0.3 + 0.7 * (780.0 - wavelength) / (780.0 - 700.0)
    } else {
        0.0
    };

    // Apply intensity factor
    [r * factor, g * factor, b * factor]
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- get_relative_key ----

    #[test]
    fn get_relative_key_returns_the_diff_from_root_when_a_root_is_given() {
        let root = PathBuf::from("/data/plate1");
        let image = Path::new("/data/plate1/well_A1/img.tif");
        assert_eq!(
            get_relative_key(image, Some(&root)),
            Some(PathBuf::from("well_A1/img.tif"))
        );
    }

    #[test]
    fn get_relative_key_returns_the_full_path_when_there_is_no_root() {
        let image = Path::new("/data/plate1/img.tif");
        assert_eq!(get_relative_key(image, None), Some(image.to_path_buf()));
    }

    #[test]
    fn get_relative_key_diffs_via_parent_segments_even_across_unrelated_absolute_roots() {
        // Two absolute paths always produce a relative diff (via `..`
        // segments), even with nothing in common - `diff_paths` only
        // returns `None` on an absolute/relative mismatch (see the next
        // test), not for "unrelated" paths.
        let root = PathBuf::from("/data/plate1");
        let image = Path::new("/other/img.tif");
        assert_eq!(
            get_relative_key(image, Some(&root)),
            Some(PathBuf::from("../../other/img.tif"))
        );
    }

    #[test]
    fn get_relative_key_returns_none_for_a_relative_image_path_against_an_absolute_root() {
        // `diff_paths` returns `None` exactly when the two paths' "is
        // absolute" status differs and the candidate path is the relative
        // one - it has no way to express a relative-to-root diff for a path
        // that isn't anchored anywhere itself.
        let root = PathBuf::from("/data/plate1");
        let image = Path::new("relative/img.tif");
        assert_eq!(get_relative_key(image, Some(&root)), None);
    }

    // ---- get_file_size ----

    #[test]
    fn get_file_size_returns_the_byte_length_of_an_existing_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("f.bin");
        std::fs::write(&path, [0u8; 42]).unwrap();
        assert_eq!(get_file_size(&path).unwrap(), 42);
    }

    #[test]
    fn get_file_size_errors_for_a_missing_file() {
        assert!(get_file_size(Path::new("/does/not/exist.bin")).is_err());
    }

    // ---- is_in_root ----

    #[test]
    fn is_in_root_true_for_a_path_under_the_root() {
        assert!(is_in_root(
            Path::new("/data/plate1/img.tif"),
            Path::new("/data/plate1")
        ));
    }

    #[test]
    fn is_in_root_false_for_a_path_outside_the_root() {
        assert!(!is_in_root(
            Path::new("/other/img.tif"),
            Path::new("/data/plate1")
        ));
    }

    #[test]
    fn is_in_root_false_for_a_sibling_directory_with_a_shared_prefix_string() {
        // `starts_with` is a path-component comparison, not a raw string
        // prefix, so "/data/plate10" must not be considered inside
        // "/data/plate1".
        assert!(!is_in_root(
            Path::new("/data/plate10/img.tif"),
            Path::new("/data/plate1")
        ));
    }

    // ---- wavelength_to_rgb_float / wavelength_to_rgb_u32 ----

    #[test]
    fn wavelength_at_or_below_one_is_treated_as_unset_and_returns_white() {
        assert_eq!(wavelength_to_rgb_float(0.0), [1.0, 1.0, 1.0]);
        assert_eq!(wavelength_to_rgb_float(1.0), [1.0, 1.0, 1.0]);
    }

    #[test]
    fn wavelength_pure_colors_are_exact() {
        assert_eq!(wavelength_to_rgb_float(635.0), [1.0, 0.0, 0.0]);
        assert_eq!(wavelength_to_rgb_float(532.0), [0.0, 1.0, 0.0]);
        assert_eq!(wavelength_to_rgb_float(450.0), [0.0, 0.0, 1.0]);
    }

    #[test]
    fn wavelength_outside_the_visible_spectrum_is_black() {
        assert_eq!(wavelength_to_rgb_float(300.0), [0.0, 0.0, 0.0]);
        assert_eq!(wavelength_to_rgb_float(800.0), [0.0, 0.0, 0.0]);
    }

    #[test]
    fn wavelength_to_rgb_u32_packs_each_float_channel_into_its_own_byte() {
        // Regression test: the blue byte was previously packed from
        // `color[0]` (red) instead of `color[2]` (blue), so a pure-blue
        // wavelength rendered as black (0x000000) instead of 0x0000FF.
        assert_eq!(wavelength_to_rgb_u32(635.0), 0x00FF0000, "pure red");
        assert_eq!(wavelength_to_rgb_u32(532.0), 0x0000FF00, "pure green");
        assert_eq!(wavelength_to_rgb_u32(450.0), 0x000000FF, "pure blue");
    }

    #[test]
    fn wavelength_to_rgb_u32_white_for_an_unset_wavelength() {
        assert_eq!(wavelength_to_rgb_u32(0.0), 0x00FFFFFF);
    }
}
