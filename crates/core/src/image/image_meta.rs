use std::collections::BTreeMap;

#[derive(Default, Clone)]
pub struct ChannelInfo {
    pub id: String,
    pub name: String,
    pub emission_wave_length: f32, // Emission wave length in nm
    pub contrast_method: String,
}

#[derive(Default, Clone)]
pub struct PyramidInfo {
    pub nr_bits: u8,
    pub color_channels: u8, // Is either 1, 3 or 4
    pub is_rgb: bool,
    pub width: u64,
    pub height: u64,
    pub tile_width: u64,
    pub tile_height: u64,
    pub is_interleaved: bool,
    pub is_little_endian: bool,
}

#[derive(Default, Clone)]
pub struct PixelSizes {
    pub px_size_x: f32, // Pixel x size in nm
    pub px_size_y: f32, // Pixel y size in nm
    pub px_size_z: f32, // Pixel z size in nm
}

#[derive(Default, Clone)]
pub struct ImageInfo {
    pub nr_c_stacks: i32,
    pub nr_z_stacks: i32,
    pub nr_t_stacks: i32,
    pub pixel_sizes: PixelSizes,
    pub resolutions: BTreeMap<i32, PyramidInfo>, // Array of resolutions in case of a pyamid image
    pub channels: BTreeMap<i32, ChannelInfo>, // Contains the channel information <channelIdx | channelinfo>
}

#[derive(Default, Clone)]
pub struct Objective {
    pub manufacturer: String,
    pub model: String,
    pub magnification: f32,
}

#[derive(Default, Clone)]
pub struct ImageMeta {
    pub name: String,
    pub objective: Objective,
    pub series: BTreeMap<i32, ImageInfo>, // Image series
}

#[derive(Default, Copy, Debug, Clone, PartialEq, Eq, Hash)]
pub struct ImagePlane {
    pub z: i32,
    pub c: i32,
    pub t: i32,
}

#[derive(Default, Clone)]
pub struct ImageTile {
    pub offset_x: usize,
    pub offset_y: usize,
    pub width: usize,
    pub height: usize,
}
