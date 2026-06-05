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
        | (color[0] * 255.0) as u32;
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
