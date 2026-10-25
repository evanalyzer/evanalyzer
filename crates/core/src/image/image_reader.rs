//! # image_reader
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-06-05
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.

use crate::ImageDimensionality;
use crate::image::Point2d;
use crate::image::image_meta::{ImageMeta, ImagePlane, ImageTile};
use crate::image::image_ome_parser::{build_image_meta, channel_select, is_signed, scopeio_err};
use crate::utils::converters::wavelength_to_rgb_float;
use evanalyzer_cfg::core_types::{InternalErrors, ZProjection};
use kornia_image::{Image, ImageSize};
use log::debug;
use rayon::prelude::*;
use scopeio::{ChannelSelect, OutputSpec, OutputType, Region, RegionRequest};
use std::ops::RangeInclusive;
use std::ops::{Deref, DerefMut};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

/// Extensions scopeio lists that would make a folder scan pick up files
/// that aren't images of their own: `xml`/`dat` are only images under the
/// fixed file names of Operetta and scan^R, `img` is the data half of a
/// NIfTI/Analyze `.hdr` + `.img` pair, and the OME companion files point at
/// TIFFs the scan finds anyway.
const SCAN_EXCLUDED_EXTENSIONS: &[&str] = &["xml", "dat", "img", "ome.xml", "companion.ome"];

/// File extensions `ImageReader` can open, lower case without the dot -
/// the extensions of every format scopeio reads as a file (not folder
/// formats such as OME-Zarr), without [`SCAN_EXCLUDED_EXTENSIONS`].
pub fn supported_image_formats() -> Vec<String> {
    let mut extensions: Vec<String> = scopeio::supported_formats()
        .iter()
        .filter(|format| !format.opens_folder)
        .flat_map(|format| format.extensions.iter())
        .filter(|ext| !SCAN_EXCLUDED_EXTENSIONS.contains(ext))
        .map(|ext| ext.to_string())
        .collect();
    extensions.sort();
    extensions.dedup();
    extensions
}

#[derive(Clone)]
pub struct ManagedImage<T, const C: usize> {
    pub data: Image<T, C>,
    /// The x/y offset from the top left of the tile which was loaded
    pub tile_offset: Point2d,
    /// Image plane info this image was extracted from - of a volume, its
    /// first plane.
    pub plane: Option<ImagePlane>,
    /// One plane or a volume. A volume's planes lie one below the other in
    /// `data` (z-major): `data` is `depth` planes high.
    pub dimensionality: ImageDimensionality,
    ///// The size of the original image (not the tile)
    //pub full_image_size: ImageSize,
    ///// Image bit depth: 8, 16, 32
    //pub nr_bits: u8,
    //// Sizes of the image pixels in nm
    //pub pixel_sizes: PixelSizes,
}

impl<T, const C: usize> Deref for ManagedImage<T, C> {
    type Target = Image<T, C>;

    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

impl<T, const C: usize> DerefMut for ManagedImage<T, C> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.data
    }
}

#[derive(Clone)]
pub enum ImageContainer {
    F32Gray(ManagedImage<f32, 1>),
    F32Rgb(ManagedImage<f32, 3>),
    U32(ManagedImage<u32, 1>),
}

impl std::fmt::Debug for ImageContainer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::F32Gray(img) => f
                .debug_struct("F32Gray")
                .field("width", &img.width())
                .field("height", &img.height())
                .field("channels", &1)
                .finish(),
            Self::F32Rgb(img) => f
                .debug_struct("F32Rgb")
                .field("width", &img.width())
                .field("height", &img.height())
                .field("channels", &3)
                .finish(),
            Self::U32(img) => f
                .debug_struct("U32")
                .field("width", &img.width())
                .field("height", &img.height())
                .field("channels", &3)
                .finish(),
        }
    }
}

impl ImageContainer {
    pub fn clone_empty(&self) -> Self {
        match self {
            ImageContainer::F32Gray(img) => {
                let new_img = kornia_image::Image::from_size_val(img.size(), 0.0)
                    .expect("Failed to allocate scratch buffer");
                ImageContainer::F32Gray(ManagedImage {
                    data: new_img,
                    tile_offset: img.tile_offset.clone(),
                    plane: img.plane.clone(),
                    dimensionality: img.dimensionality,
                })
            }
            ImageContainer::F32Rgb(img) => {
                let new_img = kornia_image::Image::from_size_val(img.size(), 0.0)
                    .expect("Failed to allocate scratch buffer");
                ImageContainer::F32Rgb(ManagedImage {
                    data: new_img,
                    tile_offset: img.tile_offset.clone(),
                    plane: img.plane.clone(),
                    dimensionality: img.dimensionality,
                })
            }
            ImageContainer::U32(img) => {
                let new_img = kornia_image::Image::from_size_val(img.size(), 0u32)
                    .expect("Failed to allocate scratch buffer");
                ImageContainer::U32(ManagedImage {
                    data: new_img,
                    tile_offset: img.tile_offset.clone(),
                    plane: img.plane.clone(),
                    dimensionality: img.dimensionality,
                })
            }
        }
    }

    /// Returns the dimensions of the underlying image.
    pub fn size(&self) -> ImageSize {
        match self {
            Self::F32Gray(img) => img.size(),
            Self::F32Rgb(img) => img.size(),
            Self::U32(img) => img.size(),
        }
    }

    pub fn nr_color_channels(&self) -> usize {
        match self {
            Self::F32Gray(_img) => 1,
            Self::F32Rgb(_img) => 3,
            Self::U32(_img) => 1,
        }
    }

    pub fn as_f32_slice(&self) -> Option<&[f32]> {
        match self {
            Self::F32Gray(img) => Some(img.as_slice()),
            Self::F32Rgb(img) => Some(img.as_slice()),
            Self::U32(_) => None,
        }
    }

    pub fn get_image_memory_usage(&self) -> usize {
        match self {
            Self::F32Gray(img) => img.size().height * img.size().width * 4,
            Self::F32Rgb(img) => img.size().height * img.size().width * 12,
            Self::U32(img) => img.size().height * img.size().width * 4,
        }
    }

    pub fn tile_offset(&self) -> Point2d {
        match self {
            Self::F32Gray(img) => return img.tile_offset,
            Self::F32Rgb(img) => return img.tile_offset,
            Self::U32(img) => return img.tile_offset,
        }
    }

    pub fn plane(&self) -> Option<ImagePlane> {
        match self {
            Self::F32Gray(img) => return img.plane,
            Self::F32Rgb(img) => return img.plane,
            Self::U32(img) => return img.plane,
        }
    }

    pub fn dimensionality(&self) -> ImageDimensionality {
        match self {
            Self::F32Gray(img) => img.dimensionality,
            Self::F32Rgb(img) => img.dimensionality,
            Self::U32(img) => img.dimensionality,
        }
    }
}

impl<T, const C: usize> ManagedImage<T, C> {
    /// The size of one plane - of a planar image its whole size.
    pub fn plane_size(&self) -> ImageSize {
        let size = self.data.size();
        ImageSize {
            width: size.width,
            height: size.height / self.dimensionality.depth() as usize,
        }
    }

    /// The samples of plane `z` (counted from the image's first plane).
    pub fn plane_slice(&self, z: usize) -> &[T] {
        let len = self.plane_size().width * self.plane_size().height * C;
        &self.data.as_slice()[z * len..(z + 1) * len]
    }
}

#[derive(Debug, Clone)]
pub struct ImageChannel {
    pub image: Arc<ImageContainer>,
    pub color: [f32; 3], // The LUT (e.g., [1.0, 0.0, 0.0] for Red)
    pub is_visible: bool,
    pub c_stack: i32,
    pub name: String,
    pub is_rgb: bool,
}

/// Computes the byte size of a `width x height` tile buffer, checking for
/// overflow along the way. A pre-flight sanity check before asking
/// scopeio to actually allocate and fill a region this size.
///
/// `width`/`height`/`color_channels` ultimately come from file-declared
/// metadata (`PyramidInfo`'s fields are read straight from whatever
/// scopeio reports) - a corrupted or malicious file can claim
/// dimensions large enough to overflow a plain `width * height * channels *
/// bytes_per_pixel` multiplication, silently wrapping to a tiny buffer size
/// instead of erroring - so every step is checked and any overflow is turned
/// into a clean error instead.
fn checked_tile_buffer_size(
    width: usize,
    height: usize,
    color_channels: u8,
    nr_bytes_per_channel: usize,
) -> Result<usize, InternalErrors> {
    width
        .checked_mul(height)
        .and_then(|v| v.checked_mul(color_channels as usize))
        .and_then(|v| v.checked_mul(nr_bytes_per_channel))
        .ok_or_else(|| {
            InternalErrors::ImageReadError(format!(
                "Tile size overflow: {width}x{height}, {color_channels}ch, \
                 {nr_bytes_per_channel}B/px exceeds the maximum representable buffer size - \
                 the file's declared dimensions are likely corrupt"
            ))
        })
}

pub struct ImageReader {
    // scopeio readers are stateless and `Send + Sync`: reads from many
    // threads at once share this one handle without locking.
    inner: Arc<dyn scopeio::ImageReader>,
    pub(crate) read_mode: ReadMode,
    pub image_meta: Arc<ImageMeta>,
    pub(crate) current_path: PathBuf,
}

#[derive(PartialEq, Eq, Debug, Clone, Copy)]
pub enum ReadMode {
    Default,
    SplitChannels, // Split RGB channels of an image to three individual channels
}

/// Image reader on top of scopeio (format auto-detected from the file's
/// content and extension).
impl ImageReader {
    /// Creates a new `ImageReader` instance and opens the underlying file.
    ///
    /// # Arguments
    /// * `path` - The filesystem path to the image to be opened.
    /// * `mode` - `SplitChannels` exposes an RGB image as three individual
    ///   grayscale channels instead of one packed RGB channel.
    ///
    /// # Errors
    /// Returns an error if the path doesn't exist, scopeio doesn't recognize
    /// the file, or the format's own parsing fails.
    pub fn new(path: &PathBuf, mode: ReadMode) -> Result<Self, InternalErrors> {
        if !path.exists() {
            return Err(InternalErrors::Io(format!(
                "File '{}' does not exist",
                path.display()
            )));
        }

        let total_start = Instant::now();

        // The real per-format parse (TIFF IFD walk, OME-XML, proprietary
        // tile-table parsing, etc.) happens in here - for most formats this
        // dominates `ImageReader::new`'s total cost, not the metadata walk
        // below, so it's timed separately.
        let open_start = Instant::now();
        let inner = scopeio::open(path).map_err(scopeio_err)?;
        let open_duration = open_start.elapsed();

        let meta_start = Instant::now();
        let image_meta = build_image_meta(inner.as_ref(), path, mode)?;
        let meta_duration = meta_start.elapsed();

        let total_duration = total_start.elapsed();
        debug!(
            "Opened {} in {:?} (format parse: {:?}, metadata walk: {:?})",
            path.display(),
            total_duration,
            open_duration,
            meta_duration
        );

        Ok(Self {
            inner,
            read_mode: mode,
            image_meta: Arc::new(image_meta),
            current_path: path.clone(),
        })
    }

    pub fn get_image_meta(&self) -> &ImageMeta {
        return self.image_meta.as_ref();
    }

    pub fn get_current_image_path(&self) -> &PathBuf {
        return &self.current_path;
    }

    /// Reset pixel sizes to default
    ///
    /// Restore the image pixel sizes from the original image meta data.
    pub fn get_pixel_sizes_from_meta(
        &self,
        series: &i32,
    ) -> Result<(f32, f32, f32), InternalErrors> {
        if let Some(series_data) = self.image_meta.series.get(&series) {
            return Ok((
                series_data.pixel_sizes.px_size_x,
                series_data.pixel_sizes.px_size_y,
                series_data.pixel_sizes.px_size_z,
            ));
        }
        Ok((1.0, 1.0, 1.0))
    }

    /// Returns the image size of the loaded image
    ///
    /// # Arguments
    ///
    /// - `&self` (`undefined`) - Describe this parameter.
    /// - `series` (`i32`) - Image series
    ///
    /// # Returns
    ///
    /// - `Result<ImageSize, InternalErrors>` - Describe the return value.
    ///
    /// # Errors
    ///
    /// Describe possible errors.
    ///
    /// # Examples
    ///
    /// ```
    /// use crate::...;
    ///
    /// let _ = get_image_size();
    /// ```
    pub fn get_image_size(&self, series: i32) -> Result<ImageSize, InternalErrors> {
        // Get the series or return error
        let series_info = self.image_meta.series.get(&series).ok_or_else(|| {
            InternalErrors::ImageReadError(format!("Series {} does not exist", series))
        })?;
        // Get pyramid
        let pyramid_info = series_info.resolutions.get(&0).ok_or_else(|| {
            InternalErrors::ImageReadError(format!(
                "Pyramid {} does not exist in series {}",
                0, series
            ))
        })?;

        Ok(ImageSize {
            width: pyramid_info.width as usize,
            height: pyramid_info.height as usize,
        })
    }

    /// Reads a tile (an arbitrary rectangular region) from the image.
    fn read_image_tile(
        &self,
        series: i32,
        resolution_idx: i32,
        image_plane: &ImagePlane,
        image_tile: &ImageTile,
    ) -> Result<ImageContainer, InternalErrors> {
        // 1. Get the series or return error
        let series_info = self.image_meta.series.get(&series).ok_or_else(|| {
            InternalErrors::ImageReadError(format!("Series {} does not exist", series))
        })?;

        // 2. Get the channel or return error
        let _channel_info = series_info.channels.get(&image_plane.c).ok_or_else(|| {
            InternalErrors::ImageReadError(format!(
                "Channel {} does not exist in series {}",
                image_plane.c, series
            ))
        })?;

        // 3. Get pyramid
        let pyramid_info = series_info
            .resolutions
            .get(&resolution_idx)
            .ok_or_else(|| {
                InternalErrors::ImageReadError(format!(
                    "Pyramid {} does not exist in series {}",
                    resolution_idx, series
                ))
            })?;

        let mut width = image_tile.width;
        let mut height = image_tile.height;
        if width == 0 || height == 0 {
            width = pyramid_info.width as usize;
            height = pyramid_info.height as usize;
        }

        // Bail before doing any buffer allocation or read call at all if the
        // metadata declares an implausible bit depth: anything above 32 is
        // outside every real scientific image format (and refuses f64 data)
        // and would make `(1u64 << nr_bits) - 1` below a shift-by-too-large,
        // silently wrong in a release build rather than an error.
        if !(1..=32).contains(&pyramid_info.nr_bits) {
            return Err(InternalErrors::ImageReadError(format!(
                "Series {series}, resolution {resolution_idx} reports an implausible bit depth \
                 of {} (expected 1-32, missing or corrupt BitsPerPixel metadata) - cannot read tile",
                pyramid_info.nr_bits
            )));
        }

        // Overflow guard before requesting the region - see
        // `checked_tile_buffer_size`'s own doc comment.
        let nr_bytes = (pyramid_info.nr_bits as f32 / 8.0).ceil() as usize;
        checked_tile_buffer_size(width, height, pyramid_info.color_channels, nr_bytes)?;

        let series_meta = self
            .inner
            .series_meta(series as usize)
            .map_err(scopeio_err)?;
        let select = channel_select(series_meta, self.read_mode, image_plane.c.max(0) as u32);
        self.read_region_f32(
            series,
            resolution_idx,
            image_plane,
            select,
            image_tile,
            ImageSize { width, height },
        )
    }

    /// The request for `select` of one z/t plane as f32 samples (see
    /// [`Self::read_region_f32`]).
    fn region_request(
        &self,
        series: i32,
        resolution_idx: i32,
        plane: &ImagePlane,
        select: ChannelSelect,
        image_tile: &ImageTile,
        size: ImageSize,
    ) -> Result<RegionRequest, InternalErrors> {
        let series_meta = self
            .inner
            .series_meta(series as usize)
            .map_err(scopeio_err)?;
        // An out-of-range resolution level falls back to the coarsest level
        // that exists, rather than erroring.
        let level = (resolution_idx as usize).min(series_meta.levels.len().saturating_sub(1));
        let dtype = if is_signed(series_meta.pixel_type) {
            OutputType::F32NormalizedShifted
        } else {
            OutputType::F32SignificantBits
        };
        Ok(RegionRequest {
            series: series as usize,
            level,
            z: plane.z.max(0) as u32,
            t: plane.t.max(0) as u32,
            channels: select,
            region: Region::new(
                image_tile.offset_x as u64,
                image_tile.offset_y as u64,
                size.width as u64,
                size.height as u64,
            ),
            output: OutputSpec {
                dtype,
                ..Default::default()
            },
        })
    }

    /// Reads planes `z_range` of channel `image_plane.c` at time
    /// `image_plane.t` (its `z` is ignored) of `image_tile` as one volume:
    /// the planes one below the other (z-major), no projection - gray or RGB
    /// as the plane would be. Each plane is decoded straight into its part
    /// of the volume, the planes in parallel. A tile of size 0 is the whole
    /// image.
    pub fn read_image_tile_volume(
        &self,
        series: i32,
        resolution_idx: i32,
        image_plane: &ImagePlane,
        z_range: RangeInclusive<i32>,
        image_tile: &ImageTile,
    ) -> Result<ImageContainer, InternalErrors> {
        let series_info = self.image_meta.series.get(&series).ok_or_else(|| {
            InternalErrors::ImageReadError(format!("Series {} does not exist", series))
        })?;
        series_info.channels.get(&image_plane.c).ok_or_else(|| {
            InternalErrors::ImageReadError(format!(
                "Channel {} does not exist in series {}",
                image_plane.c, series
            ))
        })?;
        let pyramid_info = series_info
            .resolutions
            .get(&resolution_idx)
            .ok_or_else(|| {
                InternalErrors::ImageReadError(format!(
                    "Pyramid {} does not exist in series {}",
                    resolution_idx, series
                ))
            })?;
        if z_range.is_empty() || *z_range.start() < 0 || *z_range.end() >= series_info.nr_z_stacks {
            return Err(InternalErrors::ImageReadError(format!(
                "Z-planes {}..={} are not in series {series} (it has {} planes)",
                z_range.start(),
                z_range.end(),
                series_info.nr_z_stacks
            )));
        }
        if !(1..=32).contains(&pyramid_info.nr_bits) {
            return Err(InternalErrors::ImageReadError(format!(
                "Series {series}, resolution {resolution_idx} reports an implausible bit depth \
                 of {} (expected 1-32, missing or corrupt BitsPerPixel metadata) - cannot read tile",
                pyramid_info.nr_bits
            )));
        }
        let (width, height) = match (image_tile.width, image_tile.height) {
            (0, _) | (_, 0) => (pyramid_info.width as usize, pyramid_info.height as usize),
            size => size,
        };
        let depth = (z_range.end() - z_range.start() + 1) as usize;
        let nr_bytes = (pyramid_info.nr_bits as f32 / 8.0).ceil() as usize;
        checked_tile_buffer_size(width, height * depth, pyramid_info.color_channels, nr_bytes)?;

        let series_meta = self
            .inner
            .series_meta(series as usize)
            .map_err(scopeio_err)?;
        let select = channel_select(series_meta, self.read_mode, image_plane.c.max(0) as u32);
        let size = ImageSize { width, height };
        let request = |z: i32| {
            let plane = ImagePlane { z, ..*image_plane };
            self.region_request(
                series,
                resolution_idx,
                &plane,
                select.clone(),
                image_tile,
                size,
            )
        };
        let channels = self
            .inner
            .output_shape(&request(*z_range.start())?)
            .map_err(scopeio_err)?
            .channels;
        let plane_len = width * height * channels as usize;
        let mut data = vec![0f32; plane_len * depth];
        data.par_chunks_mut(plane_len)
            .enumerate()
            .try_for_each(|(i, plane)| {
                let z = z_range.start() + i as i32;
                let out = scopeio::PixelViewMut::contiguous(
                    plane,
                    width as u64,
                    height as u64,
                    channels,
                    scopeio::Layout::Hwc,
                );
                self.inner
                    .read_region_into(&request(z)?, out)
                    .map_err(scopeio_err)
            })?;

        let tile_offset = Point2d {
            x: image_tile.offset_x,
            y: image_tile.offset_y,
        };
        let plane = Some(ImagePlane {
            z: *z_range.start(),
            ..*image_plane
        });
        let dimensionality = ImageDimensionality::Volume {
            depth: depth as u32,
        };
        let stacked = ImageSize {
            width,
            height: height * depth,
        };
        Ok(match channels {
            3 => ImageContainer::F32Rgb(ManagedImage {
                data: Image::new(stacked, data).map_err(InternalErrors::from_kornia)?,
                tile_offset,
                plane,
                dimensionality,
            }),
            _ => ImageContainer::F32Gray(ManagedImage {
                data: Image::new(stacked, data).map_err(InternalErrors::from_kornia)?,
                tile_offset,
                plane,
                dimensionality,
            }),
        })
    }

    /// Reads `select` of one z/t plane as f32 samples, normalized by
    /// scopeio in the same pass as the read: integers by their significant
    /// bits (the largest value becomes 1.0), signed ones shifted by the
    /// type's minimum first as in ImageJ (int8: -128 -> 0.0, 127 -> 1.0),
    /// floats as stored. One gray or RGB image (alpha dropped) at `plane`.
    fn read_region_f32(
        &self,
        series: i32,
        resolution_idx: i32,
        plane: &ImagePlane,
        select: ChannelSelect,
        image_tile: &ImageTile,
        size: ImageSize,
    ) -> Result<ImageContainer, InternalErrors> {
        let request =
            self.region_request(series, resolution_idx, plane, select, image_tile, size)?;
        let image = self.inner.read_region(&request).map_err(scopeio_err)?;
        let channels = image.channels as usize;
        let data: Vec<f32> = image.into_vec().map_err(scopeio_err)?;

        let tile_offset = Point2d {
            x: image_tile.offset_x,
            y: image_tile.offset_y,
        };
        let plane = Some(*plane);
        if channels == 3 {
            let data = Image::<f32, 3>::new(size, data).map_err(InternalErrors::from_kornia)?;
            return Ok(ImageContainer::F32Rgb(ManagedImage {
                data,
                tile_offset,
                plane,
                dimensionality: ImageDimensionality::Planar,
            }));
        }
        let data = Image::<f32, 1>::new(size, data).map_err(InternalErrors::from_kornia)?;
        Ok(ImageContainer::F32Gray(ManagedImage {
            data,
            tile_offset,
            plane,
            dimensionality: ImageDimensionality::Planar,
        }))
    }

    /// Reads every requested channel's (possibly Z-projected) tile using
    /// `self` alone, one channel after the other. Used by callers that
    /// already parallelize at a coarser level (e.g. `job_executor.rs`
    /// parallelizes across tiles), where adding a second layer of
    /// parallelism here would just add rayon dispatch for no gain.
    pub fn read_image_tile_combined(
        &self,
        series: i32,
        resolution_idx: i32,
        z_projection: ZProjection,
        z_range: &Option<RangeInclusive<i32>>,
        t_stack: i32,
        c_stacks_in: Option<&Vec<i32>>,
        image_tile: &ImageTile,
    ) -> Result<Vec<ImageChannel>, InternalErrors> {
        self.read_image_tile_combined_impl(
            series,
            resolution_idx,
            z_projection,
            z_range,
            t_stack,
            c_stacks_in,
            image_tile,
            false,
        )
    }

    /// Same as [`Self::read_image_tile_combined`], but reads the channels in
    /// parallel (the reader is thread-safe). For callers that read one tile
    /// at a time, e.g. the viewer.
    pub fn read_image_tile_combined_parallel(
        &self,
        series: i32,
        resolution_idx: i32,
        z_projection: ZProjection,
        z_range: &Option<RangeInclusive<i32>>,
        t_stack: i32,
        c_stacks_in: Option<&Vec<i32>>,
        image_tile: &ImageTile,
    ) -> Result<Vec<ImageChannel>, InternalErrors> {
        self.read_image_tile_combined_impl(
            series,
            resolution_idx,
            z_projection,
            z_range,
            t_stack,
            c_stacks_in,
            image_tile,
            true,
        )
    }

    /// Shared implementation behind [`Self::read_image_tile_combined`] and
    /// [`Self::read_image_tile_combined_parallel`]. `parallel` selects
    /// rayon's work-stealing iteration over the channel list vs. a plain
    /// sequential one - see the two public wrappers for why that choice is
    /// per-caller, not automatic.
    fn read_image_tile_combined_impl(
        &self,
        series: i32,
        resolution_idx: i32,
        z_projection: ZProjection,
        z_range: &Option<RangeInclusive<i32>>,
        t_stack: i32,
        c_stacks_in: Option<&Vec<i32>>,
        image_tile: &ImageTile,
        parallel: bool,
    ) -> Result<Vec<ImageChannel>, InternalErrors> {
        let series_info = self.image_meta.series.get(&series).ok_or_else(|| {
            InternalErrors::ImageReadError(format!("Series {} does not exist", series))
        })?;

        let c_stacks = match c_stacks_in {
            Some(stacks) => stacks.clone(),
            None => (0..series_info.nr_c_stacks).collect(),
        };
        let pyramid_info = series_info
            .resolutions
            .get(&resolution_idx)
            .ok_or_else(|| {
                InternalErrors::ImageReadError(format!(
                    "Pyramid {} does not exist in series {}",
                    resolution_idx, series
                ))
            })?;

        // Maximum intensity projection
        let max_proj = |dst: &mut [f32], src: &[f32]| {
            dst.iter_mut().zip(src.iter()).for_each(|(d, s)| {
                if *s > *d {
                    *d = *s;
                }
            });
        };

        // Minimum intensity projection
        let min_proj = |dst: &mut [f32], src: &[f32]| {
            dst.iter_mut().zip(src.iter()).for_each(|(d, s)| {
                if *s < *d {
                    *d = *s;
                }
            });
        };

        // Sum intensity projection
        let sum_proj = |dst: &mut [f32], src: &[f32]| {
            dst.iter_mut().zip(src.iter()).for_each(|(d, s)| {
                *d += *s;
            });
        };

        let z_stack_range = match z_range {
            Some(range) => {
                // We use inclusive range here to match the input logic
                *range.start()
                    ..=*range
                        .end()
                        .min(&(series_info.nr_z_stacks.saturating_sub(1) as i32))
            }
            None => {
                // We convert this to an inclusive range so it matches the type above
                0..=(series_info.nr_z_stacks.saturating_sub(1) as i32)
            }
        };
        // Only used by `ZProjection::TakeTheMiddle`, computed once up front.
        let z_stack_mid = z_stack_range.start() + (z_stack_range.end() - z_stack_range.start()) / 2;

        let selected_channels: Vec<&i32> = c_stacks
            .iter()
            .filter_map(|c_stack_to_read| {
                if c_stack_to_read >= &series_info.nr_c_stacks {
                    log::warn!(
                        "Channel {c_stack_to_read} was requested, but {} has only {} \
                         channel(s) (0-{}) - skipped, a pipeline reading it gets no image",
                        self.current_path.display(),
                        series_info.nr_c_stacks,
                        series_info.nr_c_stacks - 1
                    );
                    return None;
                }
                if self.read_mode == ReadMode::Default
                    && pyramid_info.is_rgb
                    && c_stack_to_read > &0
                {
                    return None;
                }
                Some(c_stack_to_read)
            })
            .collect();

        // Reads one channel's (possibly Z-projected) tile; safe to call
        // concurrently across channels.
        let read_channel = |c_stack_to_read: &i32| -> Result<ImageChannel, InternalErrors> {
            // --- PREP READ PARAMETERS ---
            let t_read = t_stack.min(series_info.nr_t_stacks.saturating_sub(1));
            let c_read = *c_stack_to_read.min(&series_info.nr_c_stacks.saturating_sub(1));

            // --- INITIAL IMAGE LOAD ---
            let mut image = self.read_image_tile(
                series,
                resolution_idx,
                &ImagePlane {
                    z: z_stack_range.start().clone(),
                    c: c_read,
                    t: t_read,
                },
                image_tile,
            )?;

            // --- Z-PROJECTION LOGIC ---
            if z_projection != ZProjection::None && !pyramid_info.is_rgb {
                if let ImageContainer::F32Gray(mut gray_image) = image {
                    for z in (z_stack_range.start() + 1)..=z_stack_range.end().clone() {
                        let image_tmp = self.read_image_tile(
                            series,
                            resolution_idx,
                            &ImagePlane {
                                z,
                                c: c_read,
                                t: t_read,
                            },
                            image_tile,
                        )?;

                        if let ImageContainer::F32Gray(image_tmp_gray) = image_tmp {
                            let src = image_tmp_gray.as_slice();
                            let dst = gray_image.as_slice_mut();

                            match z_projection {
                                ZProjection::MaxIntensity => max_proj(dst, src),
                                ZProjection::MinIntensity => min_proj(dst, src),
                                ZProjection::AvgIntensity | ZProjection::SumIntensity => {
                                    sum_proj(dst, src)
                                }
                                ZProjection::TakeTheMiddle => {
                                    if z == z_stack_mid {
                                        dst.copy_from_slice(src);
                                    }
                                }
                                ZProjection::None => {}
                            }
                        }
                    }

                    if z_projection == ZProjection::AvgIntensity {
                        let n_inv = 1.0 / (series_info.nr_z_stacks) as f32;
                        gray_image
                            .as_slice_mut()
                            .iter_mut()
                            .for_each(|p| *p *= n_inv);
                    }
                    image = ImageContainer::F32Gray(gray_image);
                }
            }

            self.image_channel(series, *c_stack_to_read, pyramid_info.is_rgb, image)
        };

        let resulting_images: Vec<ImageChannel> = if parallel {
            selected_channels
                .into_par_iter()
                .map(read_channel)
                .collect::<Result<Vec<_>, InternalErrors>>()?
        } else {
            // The caller parallelizes at a coarser level already (see
            // `read_image_tile_combined`).
            selected_channels
                .into_iter()
                .map(read_channel)
                .collect::<Result<Vec<_>, InternalErrors>>()?
        };

        Ok(resulting_images)
    }

    /// Channel `c`'s `image` with its name and display colour.
    fn image_channel(
        &self,
        series: i32,
        c: i32,
        is_rgb: bool,
        image: ImageContainer,
    ) -> Result<ImageChannel, InternalErrors> {
        let channel_meta = self.image_meta.series[&series]
            .channels
            .get(&c)
            .ok_or_else(|| {
                InternalErrors::ImageReadError(format!(
                    "Channel {c} does not exist in series {series}"
                ))
            })?;
        Ok(ImageChannel {
            image: Arc::new(image),
            color: wavelength_to_rgb_float(channel_meta.emission_wave_length),
            is_visible: true,
            c_stack: c,
            name: channel_meta.name.clone(),
            is_rgb,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::relative_eq;
    use std::fs;
    use std::thread;
    use sysinfo::{ProcessesToUpdate, System, get_current_pid};

    #[test]
    fn checked_tile_buffer_size_computes_the_plain_product_for_ordinary_dimensions() {
        let size = checked_tile_buffer_size(1920, 1080, 3, 1).unwrap();
        assert_eq!(size, 1920 * 1080 * 3);
    }

    #[test]
    fn checked_tile_buffer_size_errors_instead_of_wrapping_on_overflow() {
        // Emulates a corrupt file declaring an astronomically large tile -
        // the naive `width * height * channels * bytes` would silently wrap
        // to a small number here instead of erroring.
        let result = checked_tile_buffer_size(usize::MAX / 2, 3, 4, 1);
        assert!(result.is_err());
    }

    #[test]
    fn checked_tile_buffer_size_errors_at_exactly_usize_max_plus_one() {
        assert!(checked_tile_buffer_size(usize::MAX, 2, 1, 1).is_err());
        // One below the overflow boundary still succeeds.
        assert!(checked_tile_buffer_size(usize::MAX, 1, 1, 1).is_ok());
    }

    fn read_raw_data(path: &str, bits: i32) -> Vec<f32> {
        let reference_data_u8 = fs::read(path).unwrap();

        if bits == 8 {
            let reference_data_f32: Vec<f32> = reference_data_u8
                .into_iter()
                .map(|x| x as f32 / 255.0)
                .collect();
            return reference_data_f32;
        } else {
            if bits == 32 {
                let read_raw_data_u32: Vec<i32> = reference_data_u8
                    .chunks_exact(4) // Take 4 bytes at a time
                    .map(|chunk| {
                        // Convert 4 bytes (u8) into a [u8; 4] array,
                        // then into an f32 using Native Endianness (usually Little Endian)
                        i32::from_ne_bytes(chunk.try_into().unwrap())
                    })
                    .collect();

                let reference_data_f32: Vec<f32> = read_raw_data_u32
                    .into_iter()
                    .map(|x| x as f32 / 1.0)
                    .collect();
                return reference_data_f32;
            } else {
                return vec![];
            }
        }
    }

    fn compare_data(wanted: &Vec<f32>, is_data: &[f32], epsilon: f32) {
        for n in 0..wanted.len() {
            let actual = is_data.get(n).unwrap();
            let expected = wanted.get(n).unwrap();

            assert!(
                relative_eq!(actual, expected, epsilon = epsilon),
                "Normalization failed! Got {}, expected {} (diff was greater than {} at pixel {})",
                actual,
                expected,
                epsilon,
                n
            );
        }
    }

    #[test]
    fn test_no_projection_z0() {
        let reference_data_f32 = read_raw_data(
            concat!(env!("CARGO_MANIFEST_DIR"), "/tests/slice_Z0_C0_T0.raw"),
            8,
        );

        let reader = ImageReader::new(
            &concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/tests/multi-channel-4D-series.ome.tif"
            )
            .into(),
            ReadMode::Default,
        )
        .unwrap();
        let result = reader
            .read_image_tile_combined(
                0,
                0,
                ZProjection::None,
                &None,
                0,
                Some(&vec![0]),
                &ImageTile {
                    offset_x: 0,
                    offset_y: 0,
                    width: 0,
                    height: 0,
                },
            )
            .unwrap();

        for image_channel in result {
            match &*image_channel.image {
                ImageContainer::F32Gray(image) => {
                    let slice = image.as_slice();
                    compare_data(&reference_data_f32, &slice, 1e-6);
                }
                ImageContainer::F32Rgb(_) => todo!(),
                ImageContainer::U32(_) => todo!(),
            }
        }
    }

    #[test]
    fn test_no_projection_z1() {
        let reference_data_f32 = read_raw_data(
            concat!(env!("CARGO_MANIFEST_DIR"), "/tests/slice_Z1_C0_T0.raw"),
            8,
        );

        let reader = ImageReader::new(
            &concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/tests/multi-channel-4D-series.ome.tif"
            )
            .into(),
            ReadMode::Default,
        )
        .unwrap();
        let result = reader
            .read_image_tile_combined(
                0,
                0,
                ZProjection::None,
                &Some(1..=1),
                0,
                Some(&vec![0]),
                &ImageTile {
                    offset_x: 0,
                    offset_y: 0,
                    width: 0,
                    height: 0,
                },
            )
            .unwrap();

        for image_channel in result {
            match &*image_channel.image {
                ImageContainer::F32Gray(image) => {
                    let slice = image.as_slice();
                    compare_data(&reference_data_f32, &slice, 1e-6);
                }
                ImageContainer::F32Rgb(_) => todo!(),
                ImageContainer::U32(_) => todo!(),
            }
        }
    }

    #[test]
    fn test_maximum_intensity_projection() {
        let reader = ImageReader::new(
            &concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/tests/multi-channel-4D-series.ome.tif"
            )
            .into(),
            ReadMode::Default,
        )
        .unwrap();
        let result = reader
            .read_image_tile_combined(
                0,
                0,
                ZProjection::MaxIntensity,
                &None,
                0,
                Some(&vec![0]),
                &ImageTile {
                    offset_x: 0,
                    offset_y: 0,
                    width: 0,
                    height: 0,
                },
            )
            .unwrap();

        let reference_data_f32 = read_raw_data(
            concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/tests/slice_Z0_C0_T0_max_intensity.raw"
            ),
            8,
        );

        for image_channel in result {
            match &*image_channel.image {
                ImageContainer::F32Gray(image) => {
                    let slice = image.as_slice();
                    compare_data(&reference_data_f32, &slice, 1e-6);
                }
                ImageContainer::F32Rgb(_) => todo!(),
                ImageContainer::U32(_) => todo!(),
            }
        }
    }

    fn fixture_path() -> PathBuf {
        concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/multi-channel-4D-series.ome.tif"
        )
        .into()
    }

    fn full_tile() -> ImageTile {
        ImageTile {
            offset_x: 0,
            offset_y: 0,
            width: 0,
            height: 0,
        }
    }

    #[test]
    fn a_nonexistent_channel_is_skipped_not_an_error() {
        // `read_image_tile_combined_impl` filters requested channels the image
        // doesn't have (as it filters channels > 0 of an RGB image): the
        // result simply holds no tile for them.
        let reader = ImageReader::new(&fixture_path(), ReadMode::Default).unwrap();
        let channels = reader
            .read_image_tile_combined(
                0,
                0,
                ZProjection::None,
                &None,
                0,
                Some(&vec![99]),
                &full_tile(),
            )
            .unwrap();
        assert!(channels.is_empty());
    }

    #[test]
    fn an_implausible_bit_depth_is_refused_before_reading() {
        // Corrupt metadata (e.g. BitsPerPixel 40) must be an error, not a
        // silently wrong normalization.
        let mut reader = ImageReader::new(&fixture_path(), ReadMode::Default).unwrap();
        let meta = Arc::make_mut(&mut reader.image_meta);
        for info in meta.series.values_mut() {
            for resolution in info.resolutions.values_mut() {
                resolution.nr_bits = 40;
            }
        }
        let err = reader
            .read_image_tile_combined(0, 0, ZProjection::None, &None, 0, None, &full_tile())
            .unwrap_err();
        assert!(
            matches!(&err, InternalErrors::ImageReadError(msg) if msg.contains("implausible bit depth")),
            "{err:?}"
        );
    }

    #[test]
    fn read_image_tile_combined_errors_for_a_nonexistent_series() {
        let reader = ImageReader::new(&fixture_path(), ReadMode::Default).unwrap();
        let err = reader
            .read_image_tile_combined(99, 0, ZProjection::None, &None, 0, None, &full_tile())
            .unwrap_err();
        assert!(
            matches!(err, InternalErrors::ImageReadError(msg) if msg.contains("Series 99 does not exist"))
        );
    }

    #[test]
    fn read_image_tile_combined_errors_for_a_nonexistent_resolution() {
        let reader = ImageReader::new(&fixture_path(), ReadMode::Default).unwrap();
        let err = reader
            .read_image_tile_combined(0, 99, ZProjection::None, &None, 0, None, &full_tile())
            .unwrap_err();
        assert!(
            matches!(err, InternalErrors::ImageReadError(msg) if msg.contains("Pyramid 99 does not exist"))
        );
    }

    #[test]
    fn get_image_size_errors_for_a_nonexistent_series() {
        let reader = ImageReader::new(&fixture_path(), ReadMode::Default).unwrap();
        let err = reader.get_image_size(99).unwrap_err();
        assert!(
            matches!(err, InternalErrors::ImageReadError(msg) if msg.contains("Series 99 does not exist"))
        );
    }

    #[test]
    fn get_image_size_returns_the_real_dimensions_for_an_existing_series() {
        let reader = ImageReader::new(&fixture_path(), ReadMode::Default).unwrap();
        let size = reader.get_image_size(0).unwrap();
        assert!(size.width > 0);
        assert!(size.height > 0);
    }

    #[test]
    fn get_pixel_sizes_from_meta_returns_the_real_values_for_an_existing_series() {
        let reader = ImageReader::new(&fixture_path(), ReadMode::Default).unwrap();
        let (x, y, z) = reader.get_pixel_sizes_from_meta(&0).unwrap();
        let expected = &reader.get_image_meta().series[&0].pixel_sizes;
        assert_eq!(
            (x, y, z),
            (expected.px_size_x, expected.px_size_y, expected.px_size_z)
        );
    }

    #[test]
    fn get_pixel_sizes_from_meta_falls_back_to_1_1_1_for_a_nonexistent_series() {
        let reader = ImageReader::new(&fixture_path(), ReadMode::Default).unwrap();
        assert_eq!(
            reader.get_pixel_sizes_from_meta(&99).unwrap(),
            (1.0, 1.0, 1.0)
        );
    }

    #[test]
    fn read_image_tile_combined_silently_drops_a_channel_index_past_the_last_real_channel() {
        // Unlike a nonexistent series/resolution (hard errors), requesting a
        // channel index >= `nr_c_stacks` is filtered out of the result
        // rather than failing the whole read - useful for a caller that
        // requests a fixed set of channel slots without knowing up front how
        // many real channels a given image has.
        let reader = ImageReader::new(&fixture_path(), ReadMode::Default).unwrap();
        let nr_c_stacks = reader.get_image_meta().series[&0].nr_c_stacks;
        let result = reader
            .read_image_tile_combined(
                0,
                0,
                ZProjection::None,
                &None,
                0,
                Some(&vec![0, nr_c_stacks]),
                &full_tile(),
            )
            .unwrap();

        assert_eq!(
            result.len(),
            1,
            "only channel 0 is real - the out-of-range channel must be silently dropped, not errored"
        );
        assert_eq!(result[0].c_stack, 0);
    }

    /// A volume of planes 1..=3 is those planes one below the other - each
    /// the same as the plane read on its own.
    #[test]
    fn a_volume_is_its_planes_one_below_the_other() {
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/multi-channel-4D-series.ome.tif"
        );
        let reader = ImageReader::new(&path.into(), ReadMode::Default).unwrap();
        let tile = ImageTile::default();
        let channel = ImagePlane { z: 0, c: 1, t: 0 };

        let volume = reader
            .read_image_tile_volume(0, 0, &channel, 1..=3, &tile)
            .unwrap();
        let ImageContainer::F32Gray(volume) = volume else {
            panic!("a gray volume");
        };
        assert_eq!(
            volume.dimensionality,
            ImageDimensionality::Volume { depth: 3 }
        );
        assert_eq!(volume.plane, Some(ImagePlane { z: 1, c: 1, t: 0 }));
        let plane_size = volume.plane_size();
        assert_eq!(volume.size().height, plane_size.height * 3);
        for (i, z) in (1..=3).enumerate() {
            let ImageContainer::F32Gray(plane) = reader
                .read_image_tile(0, 0, &ImagePlane { z, ..channel }, &tile)
                .unwrap()
            else {
                panic!("a gray plane");
            };
            assert_eq!(plane.size(), plane_size);
            assert_eq!(volume.plane_slice(i), plane.as_slice(), "plane {z}");
        }
        assert_ne!(
            volume.plane_slice(0),
            volume.plane_slice(2),
            "the planes differ"
        );
    }

    #[test]
    fn a_volume_outside_the_planes_is_refused() {
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/multi-channel-4D-series.ome.tif"
        );
        let reader = ImageReader::new(&path.into(), ReadMode::Default).unwrap();
        let planes = reader.get_image_meta().series[&0].nr_z_stacks;
        let read = |z| {
            reader.read_image_tile_volume(0, 0, &ImagePlane::default(), z, &ImageTile::default())
        };
        assert!(read(0..=planes).is_err(), "one plane too many");
        assert!(read(-1..=0).is_err());
        assert!(read(0..=planes - 1).is_ok());
    }

    /// Fiji's "Fly Brain" sample (RGB, 57 planes), downloaded by hand to
    /// `tests/3D/` (gitignored, licence unchecked): read as one RGB volume.
    #[test]
    #[ignore = "needs tests/3D/flybrain.tif (local only)"]
    fn the_fly_brain_sample_reads_as_an_rgb_volume() {
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../tests/3D/flybrain.tif");
        let reader = ImageReader::new(&path.into(), ReadMode::Default).unwrap();
        let series = &reader.get_image_meta().series[&0];
        assert_eq!(series.nr_z_stacks, 57, "{series:?}");
        let started = std::time::Instant::now();
        let volume = reader
            .read_image_tile_volume(0, 0, &ImagePlane::default(), 0..=56, &ImageTile::default())
            .unwrap();
        eprintln!(
            "flybrain: {:?}, pixel sizes {:?} nm, read in {:?}",
            volume,
            series.pixel_sizes,
            started.elapsed()
        );
        assert_eq!(
            volume.dimensionality(),
            ImageDimensionality::Volume { depth: 57 }
        );
        assert!(matches!(volume, ImageContainer::F32Rgb(_)), "{volume:?}");
    }

    #[test]
    fn take_the_middle_projection_returns_the_middle_slice_not_the_first() {
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/multi-channel-4D-series.ome.tif"
        );
        let tile = ImageTile {
            offset_x: 0,
            offset_y: 0,
            width: 0,
            height: 0,
        };

        let reader = ImageReader::new(&path.into(), ReadMode::Default).unwrap();

        let nr_z_stacks = reader.get_image_meta().series[&0].nr_z_stacks;
        assert!(
            nr_z_stacks > 2,
            "test fixture needs more than 2 Z-slices to distinguish 'first' from 'middle'"
        );
        let z_mid = (nr_z_stacks - 1) / 2;

        // Under test: the full-range TakeTheMiddle projection.
        let projected = reader
            .read_image_tile_combined(
                0,
                0,
                ZProjection::TakeTheMiddle,
                &None,
                0,
                Some(&vec![0]),
                &tile,
            )
            .unwrap();

        // Ground truth: the single slice at the exact middle Z-index, read
        // directly with no projection at all.
        let expected_mid = reader
            .read_image_tile_combined(
                0,
                0,
                ZProjection::None,
                &Some(z_mid..=z_mid),
                0,
                Some(&vec![0]),
                &tile,
            )
            .unwrap();

        // The first slice - to prove the fixture actually varies across Z,
        // otherwise "returns slice 0" (the bug) and "returns the middle
        // slice" (the fix) would be indistinguishable by this test.
        let first_slice = reader
            .read_image_tile_combined(
                0,
                0,
                ZProjection::None,
                &Some(0..=0),
                0,
                Some(&vec![0]),
                &tile,
            )
            .unwrap();

        for ((proj_ch, mid_ch), first_ch) in projected
            .iter()
            .zip(expected_mid.iter())
            .zip(first_slice.iter())
        {
            match (&*proj_ch.image, &*mid_ch.image, &*first_ch.image) {
                (
                    ImageContainer::F32Gray(proj),
                    ImageContainer::F32Gray(mid),
                    ImageContainer::F32Gray(first),
                ) => {
                    assert_eq!(
                        proj.as_slice(),
                        mid.as_slice(),
                        "TakeTheMiddle should return exactly the middle Z-slice's pixels"
                    );
                    assert_ne!(
                        mid.as_slice(),
                        first.as_slice(),
                        "test fixture's middle and first Z-slices are identical - this test can't \
                         distinguish the fix from the bug it targets"
                    );
                }
                other => panic!("unexpected variant combination: {other:?}"),
            }
        }
    }

    #[test]
    fn the_supported_formats_come_from_scopeio_without_companion_files() {
        let formats = supported_image_formats();
        for ext in ["tif", "ome.tif", "vsi", "czi", "nd2", "lif", "svs", "png"] {
            assert!(
                formats.iter().any(|f| f == ext),
                "{ext} missing in {formats:?}"
            );
        }
        for ext in ["xml", "dat", "img", "zarr"] {
            assert!(
                !formats.iter().any(|f| f == ext),
                "{ext} must not be scanned"
            );
        }
    }

    /// Split RGB channels come from one read per colour channel and equal
    /// the samples of the packed RGB read, in the requested order.
    #[test]
    fn split_rgb_channels_are_the_samples_of_the_packed_read() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("rgb.png");
        let (w, h) = (7u32, 5u32);
        image::RgbImage::from_fn(w, h, |x, y| {
            image::Rgb([(x * 30) as u8, (y * 40) as u8, (x * y * 5) as u8])
        })
        .save(&path)
        .unwrap();

        let packed = ImageReader::new(&path, ReadMode::Default).unwrap();
        let packed = packed
            .read_image_tile_combined(0, 0, ZProjection::None, &None, 0, None, &full_tile())
            .unwrap();
        let rgb = packed[0].image.as_f32_slice().unwrap();

        let split = ImageReader::new(&path, ReadMode::SplitChannels).unwrap();
        for parallel in [false, true] {
            let order = vec![2, 0, 1];
            let channels = if parallel {
                split.read_image_tile_combined_parallel(
                    0,
                    0,
                    ZProjection::None,
                    &None,
                    0,
                    Some(&order),
                    &full_tile(),
                )
            } else {
                split.read_image_tile_combined(
                    0,
                    0,
                    ZProjection::None,
                    &None,
                    0,
                    Some(&order),
                    &full_tile(),
                )
            }
            .unwrap();
            assert_eq!(
                channels.iter().map(|c| c.c_stack).collect::<Vec<_>>(),
                order,
                "the requested order"
            );
            for channel in &channels {
                let sample = channel.c_stack as usize;
                let expected: Vec<f32> = rgb.iter().skip(sample).step_by(3).copied().collect();
                assert_eq!(
                    channel.image.as_f32_slice().unwrap(),
                    expected,
                    "sample {sample}"
                );
                assert_eq!(channel.image.plane().unwrap().c, channel.c_stack);
            }
        }
    }

    #[test]
    fn parallel_read_matches_sequential_read() {
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/multi-channel-4D-series.ome.tif"
        );
        let tile = ImageTile {
            offset_x: 0,
            offset_y: 0,
            width: 0,
            height: 0,
        };

        // One reader, read sequentially and with every channel in parallel.
        let reader = ImageReader::new(&path.into(), ReadMode::SplitChannels).unwrap();
        let mut sequential = reader
            .read_image_tile_combined(0, 0, ZProjection::MaxIntensity, &None, 0, None, &tile)
            .unwrap();
        let mut pooled = reader
            .read_image_tile_combined_parallel(
                0,
                0,
                ZProjection::MaxIntensity,
                &None,
                0,
                None,
                &tile,
            )
            .unwrap();

        assert_eq!(
            sequential.len(),
            pooled.len(),
            "must read the same number of channels"
        );
        assert!(
            sequential.len() > 1,
            "test needs a genuinely multi-channel fixture"
        );

        // rayon's completion order isn't guaranteed to match input order.
        sequential.sort_by_key(|c| c.c_stack);
        pooled.sort_by_key(|c| c.c_stack);

        for (seq_ch, pooled_ch) in sequential.iter().zip(pooled.iter()) {
            assert_eq!(seq_ch.c_stack, pooled_ch.c_stack);
            match (&*seq_ch.image, &*pooled_ch.image) {
                (ImageContainer::F32Gray(a), ImageContainer::F32Gray(b)) => {
                    assert_eq!(
                        a.as_slice(),
                        b.as_slice(),
                        "channel {} pixels differ between sequential and parallel reads",
                        seq_ch.c_stack
                    );
                }
                other => panic!("unexpected variant combination: {other:?}"),
            }
        }
    }

    #[test]
    fn test_minimum_intensity_projection() {
        let reader = ImageReader::new(
            &concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/tests/multi-channel-4D-series.ome.tif"
            )
            .into(),
            ReadMode::Default,
        )
        .unwrap();
        let result = reader
            .read_image_tile_combined(
                0,
                0,
                ZProjection::MinIntensity,
                &None,
                0,
                Some(&vec![0]),
                &ImageTile {
                    offset_x: 0,
                    offset_y: 0,
                    width: 0,
                    height: 0,
                },
            )
            .unwrap();

        let reference_data_f32 = read_raw_data(
            concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/tests/slice_Z0_C0_T0_min_intensity.raw"
            ),
            8,
        );

        for image_channel in result {
            match &*image_channel.image {
                ImageContainer::F32Gray(image) => {
                    let slice = image.as_slice();
                    compare_data(&reference_data_f32, &slice, 1e-6);
                }
                ImageContainer::F32Rgb(_) => todo!(),
                ImageContainer::U32(_) => todo!(),
            }
        }
    }

    #[test]
    fn test_average_intensity_projection() {
        let reader = ImageReader::new(
            &concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/tests/multi-channel-4D-series.ome.tif"
            )
            .into(),
            ReadMode::Default,
        )
        .unwrap();
        let result = reader
            .read_image_tile_combined(
                0,
                0,
                ZProjection::AvgIntensity,
                &None,
                0,
                Some(&vec![0]),
                &ImageTile {
                    offset_x: 0,
                    offset_y: 0,
                    width: 0,
                    height: 0,
                },
            )
            .unwrap();

        let reference_data_f32 = read_raw_data(
            concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/tests/slice_Z0_C0_T0_avg_intensity.raw"
            ),
            8,
        );

        for image_channel in result {
            match &*image_channel.image {
                ImageContainer::F32Gray(image) => {
                    let slice = image.as_slice();
                    compare_data(&reference_data_f32, &slice, 1e-2);
                }
                ImageContainer::F32Rgb(_) => todo!(),
                ImageContainer::U32(_) => todo!(),
            }
        }
    }

    #[test]
    fn clone_empty_u32_returns_matching_variant_and_size() {
        let size = ImageSize {
            width: 4,
            height: 3,
        };
        let source = ImageContainer::U32(ManagedImage {
            data: Image::<u32, 1>::new(size, vec![7u32; size.width * size.height]).unwrap(),
            tile_offset: Point2d { x: 1, y: 2 },
            plane: None,
            dimensionality: ImageDimensionality::Planar,
        });

        let empty = source.clone_empty();

        match empty {
            ImageContainer::U32(img) => {
                assert_eq!(img.size().width, size.width);
                assert_eq!(img.size().height, size.height);
                assert_eq!(img.as_slice().len(), size.width * size.height);
                assert!(img.as_slice().iter().all(|&v| v == 0));
            }
            other => panic!(
                "clone_empty() on a U32 container must return a U32 container, got {other:?}"
            ),
        }
    }

    #[test]
    fn u32_container_accessors_match_their_f32_counterparts() {
        // `nr_color_channels`, `as_f32_slice`, `get_image_memory_usage`,
        // `tile_offset` and `plane` are all per-variant match arms - every
        // other test in this file exercises `F32Gray`/`F32Rgb` containers,
        // leaving the `U32` arm of each one untested.
        let size = ImageSize {
            width: 4,
            height: 3,
        };
        let plane = Some(ImagePlane { z: 1, c: 2, t: 3 });
        let container = ImageContainer::U32(ManagedImage {
            data: Image::<u32, 1>::new(size, vec![7u32; size.width * size.height]).unwrap(),
            tile_offset: Point2d { x: 5, y: 6 },
            plane,
            dimensionality: ImageDimensionality::Planar,
        });

        assert_eq!(container.nr_color_channels(), 1);
        assert!(
            container.as_f32_slice().is_none(),
            "a U32 container has no f32 data to slice"
        );
        assert_eq!(
            container.get_image_memory_usage(),
            size.width * size.height * 4
        );
        assert_eq!(container.tile_offset(), Point2d { x: 5, y: 6 });
        assert_eq!(container.plane(), plane);
    }

    /// Regression test for a real-world bug: `G7_03.vsi`'s series 0 declares
    /// 3 real channels (`nr_c_stacks == 3`), but the VSI reader's own OME
    /// metadata only described 1 of them - `build_image_meta` used to trust
    /// that shorter list and silently drop channels 1/2, which then failed
    /// with "Channel 1 does not exist in series 0" the moment the viewer
    /// tried to read them. Reads every channel of both series to prove the
    /// fix holds for the whole file, not just series 0's metadata shape.
    #[test]
    #[ignore = "requires tests/G7_03.vsi, a real VSI fixture too large to commit to git \
                (see .gitignore) - run locally with `cargo test -- --ignored` when present"]
    fn every_declared_channel_reads_successfully_even_when_ome_metadata_under_reports_them() {
        let path: PathBuf = concat!(env!("CARGO_MANIFEST_DIR"), "/../../tests/G7_03.vsi").into();
        let reader = ImageReader::new(&path, ReadMode::Default).unwrap();

        let tile = ImageTile {
            offset_x: 0,
            offset_y: 0,
            width: 0,
            height: 0,
        };

        for (series, info) in reader.image_meta.series.iter() {
            let c_stacks: Vec<i32> = (0..info.nr_c_stacks).collect();
            let result = reader
                .read_image_tile_combined(
                    *series,
                    0,
                    ZProjection::None,
                    &None,
                    0,
                    Some(&c_stacks),
                    &tile,
                )
                .unwrap_or_else(|e| panic!("series {series} must read every channel: {e}"));
            assert_eq!(
                result.len(),
                c_stacks.len(),
                "series {series} must return one ImageChannel per declared channel"
            );
        }
    }

    #[test]
    fn test_bigger_image_with_z_stack() {
        let reader = ImageReader::new(
            &concat!(env!("CARGO_MANIFEST_DIR"), "/tests/muliple_z_stacks.nd2").into(),
            ReadMode::Default,
        )
        .unwrap();
        let result = reader
            .read_image_tile_combined(
                0,
                0,
                ZProjection::None,
                &None,
                0,
                Some(&vec![0]),
                &ImageTile {
                    offset_x: 0,
                    offset_y: 0,
                    width: 0,
                    height: 0,
                },
            )
            .unwrap();

        for image_channel in result {
            match &*image_channel.image {
                ImageContainer::F32Gray(_) => {}
                ImageContainer::F32Rgb(_) => todo!(),
                ImageContainer::U32(_) => todo!(),
            }
        }
    }

    /// Resident set size of this process, in bytes, or `None` if it can't be
    /// determined on this platform.
    fn current_process_rss_bytes() -> Option<u64> {
        let pid = get_current_pid().ok()?;
        let mut sys = System::new();
        sys.refresh_processes(ProcessesToUpdate::Some(&[pid]), true);
        sys.process(pid).map(|p| p.memory())
    }

    /// Sanity check: repeatedly opening, reading from, and dropping readers
    /// must keep working over many cycles without erroring or leaking a
    /// meaningful amount of memory.
    #[test]
    fn repeated_open_read_close_does_not_error_or_blow_up_memory() {
        let path: PathBuf = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/multi-channel-4D-series.ome.tif"
        )
        .into();
        let tile = ImageTile {
            offset_x: 0,
            offset_y: 0,
            width: 0,
            height: 0,
        };

        let open_read_drop = || {
            let reader = ImageReader::new(&path, ReadMode::Default).unwrap();
            let _ = reader
                .read_image_tile_combined(0, 0, ZProjection::None, &None, 0, Some(&vec![0]), &tile)
                .unwrap();
            // `reader` (and its underlying format reader) drops here.
        };

        for _ in 0..10 {
            open_read_drop();
        }

        let baseline_bytes = current_process_rss_bytes();

        const ITERATIONS: usize = 200;
        for _ in 0..ITERATIONS {
            open_read_drop();
        }

        let (Some(baseline_bytes), Some(after_bytes)) =
            (baseline_bytes, current_process_rss_bytes())
        else {
            eprintln!("skipping RSS sanity check: process memory unavailable on this platform");
            return;
        };
        let growth_mb = after_bytes.saturating_sub(baseline_bytes) as f64 / 1_000_000.0;

        // Wide bound: this only guards against a qualitatively different
        // failure mode (e.g. leaking a whole tile buffer's worth of memory
        // per iteration), not normal allocator/heap fluctuation.
        assert!(
            growth_mb < 1500.0,
            "process RSS grew by {growth_mb:.1} MB over {ITERATIONS} open/read/close cycles \
             (baseline {baseline_bytes} bytes, after {after_bytes} bytes) - well beyond normal \
             growth for this workload"
        );
    }

    /// Confirms opening/reading a reader is safe to
    /// enter concurrently from several genuinely independent OS threads at
    /// once - e.g. a GUI thread and a background worker thread each opening
    /// their own reader - rather than only from rayon's own worker pool
    /// (already covered by `pooled_parallel_read_matches_sequential_read`).
    /// Every thread opens an independent reader on the same file and all
    /// must see identical channel data.
    #[test]
    fn concurrent_readers_on_independent_threads_produce_consistent_results() {
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/multi-channel-4D-series.ome.tif"
        );

        let handles: Vec<_> = (0..8)
            .map(|_| {
                let path = path.to_string();
                thread::spawn(move || {
                    let reader = ImageReader::new(&path.into(), ReadMode::Default).unwrap();
                    reader
                        .read_image_tile_combined(
                            0,
                            0,
                            ZProjection::None,
                            &None,
                            0,
                            Some(&vec![0]),
                            &ImageTile {
                                offset_x: 0,
                                offset_y: 0,
                                width: 0,
                                height: 0,
                            },
                        )
                        .unwrap()
                })
            })
            .collect();

        let results: Vec<Vec<ImageChannel>> = handles
            .into_iter()
            .map(|h| h.join().expect("reader thread panicked"))
            .collect();

        assert!(
            results.iter().all(|r| !r.is_empty()),
            "every thread must read at least one channel"
        );

        let first = &results[0];
        for other in &results[1..] {
            assert_eq!(
                other.len(),
                first.len(),
                "channel counts must match across threads"
            );
            for (a, b) in first.iter().zip(other.iter()) {
                assert_eq!(a.c_stack, b.c_stack);
                match (&*a.image, &*b.image) {
                    (ImageContainer::F32Gray(x), ImageContainer::F32Gray(y)) => {
                        assert_eq!(
                            x.as_slice(),
                            y.as_slice(),
                            "channel {} pixels differ between independent threads",
                            a.c_stack
                        );
                    }
                    other => panic!("unexpected variant combination: {other:?}"),
                }
            }
        }
    }

    #[test]
    fn get_current_image_path_returns_the_path_the_reader_was_opened_with() {
        let reader = ImageReader::new(&fixture_path(), ReadMode::Default).unwrap();
        assert_eq!(reader.get_current_image_path(), &fixture_path());
    }

    #[test]
    fn get_image_size_errors_when_the_series_has_no_full_resolution_level() {
        let mut reader = ImageReader::new(&fixture_path(), ReadMode::Default).unwrap();
        let meta = Arc::make_mut(&mut reader.image_meta);
        meta.series.get_mut(&0).unwrap().resolutions.remove(&0);
        let err = reader.get_image_size(0).unwrap_err();
        assert!(
            matches!(&err, InternalErrors::ImageReadError(msg) if msg.contains("Pyramid 0 does not exist in series 0")),
            "{err:?}"
        );
    }

    /// `read_image_tile` checks series, channel and resolution itself - the
    /// public entry points filter most of these out before calling it, so
    /// its own guards are exercised directly.
    #[test]
    fn read_image_tile_errors_for_a_nonexistent_series_channel_or_resolution() {
        let reader = ImageReader::new(&fixture_path(), ReadMode::Default).unwrap();
        let plane = |c| ImagePlane { z: 0, c, t: 0 };

        let err = reader
            .read_image_tile(99, 0, &plane(0), &full_tile())
            .unwrap_err();
        assert!(
            matches!(&err, InternalErrors::ImageReadError(msg) if msg.contains("Series 99 does not exist")),
            "{err:?}"
        );

        let err = reader
            .read_image_tile(0, 0, &plane(99), &full_tile())
            .unwrap_err();
        assert!(
            matches!(&err, InternalErrors::ImageReadError(msg) if msg.contains("Channel 99 does not exist in series 0")),
            "{err:?}"
        );

        let err = reader
            .read_image_tile(0, 99, &plane(0), &full_tile())
            .unwrap_err();
        assert!(
            matches!(&err, InternalErrors::ImageReadError(msg) if msg.contains("Pyramid 99 does not exist in series 0")),
            "{err:?}"
        );
    }

    /// An RGB series in `ReadMode::Default` exposes one packed channel:
    /// requested channels past 0 are dropped, and no Z projection is applied
    /// (the first plane of the Z range is returned as is).
    #[test]
    fn an_rgb_series_in_default_mode_reads_only_channel_zero_without_z_projection() {
        let plain = ImageReader::new(&fixture_path(), ReadMode::Default).unwrap();
        assert!(
            plain.get_image_meta().series[&0].nr_c_stacks >= 2
                && plain.get_image_meta().series[&0].nr_z_stacks >= 2,
            "the fixture must have several channels and Z planes"
        );
        let first_plane = plain
            .read_image_tile_combined(
                0,
                0,
                ZProjection::None,
                &None,
                0,
                Some(&vec![0]),
                &full_tile(),
            )
            .unwrap();

        let mut rgb = ImageReader::new(&fixture_path(), ReadMode::Default).unwrap();
        let meta = Arc::make_mut(&mut rgb.image_meta);
        for info in meta.series.values_mut() {
            for resolution in info.resolutions.values_mut() {
                resolution.is_rgb = true;
            }
        }
        let result = rgb
            .read_image_tile_combined(
                0,
                0,
                ZProjection::MaxIntensity,
                &None,
                0,
                Some(&vec![0, 1]),
                &full_tile(),
            )
            .unwrap();

        assert_eq!(result.len(), 1, "only the packed channel 0 is read");
        assert_eq!(result[0].c_stack, 0);
        assert!(result[0].is_rgb);
        assert_eq!(
            result[0].image.as_f32_slice().unwrap(),
            first_plane[0].image.as_f32_slice().unwrap(),
            "an RGB series is never Z-projected"
        );
    }
}
