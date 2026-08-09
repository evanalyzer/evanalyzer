use crate::converters::wavelength_to_rgb_float;
use crate::image::image_meta::{ImageMeta, ImagePlane, ImageTile};
use crate::image::java::{JAVA_WRAPPER, ensure_java_wrapper};
use evanalyzer_cfg::core_types::InternalErrors;
use jni::errors::Error as JniError;
use jni::objects::{JMethodID, JObject, JValue};
use jni::refs::Global;
use jni::signature::{Primitive, ReturnType};
use kornia_apriltag::utils::Point2d;
use kornia_image::{Image, ImageSize};
use kornia_tensor::CpuAllocator;
use log::info;
use rayon::prelude::*;
use std::ops::RangeInclusive;
use std::ops::{Deref, DerefMut};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

pub const SUPPORTED_IMAGE_FORMATS: &[&str] = &[
    "tif", "tiff", "btif", "btiff", "btf", "jpg", "jpeg", "vsi", "ics", "czi", "nd2", "lif", "lei",
    "fli", "scn", "sxm", "lim", "oir", "top", "stk", "nd", "bip", "fli", "msr", "dm3", "dm4",
    "img", "cr2", "ch5", "dib", "ims", "pic", "raw", "1sc", "std", "spc", "avi", "cif", "sif",
    "aim", "svs", "arf", "sld",
];

#[derive(Default, Debug, Clone, PartialEq, Eq, Hash)]
pub enum ZProjection {
    #[default]
    None,
    MaxIntensity,
    MinIntensity,
    AvgIntensity,
    SumIntensity,
    TakeTheMiddle,
}
#[derive(Clone)]
pub struct ManagedImage<T, const C: usize> {
    pub data: Image<T, C, CpuAllocator>,
    /// The x/y offset from the top left of the tile which was loaded
    pub tile_offset: Point2d,
    /// Image plane info this image was extracted from
    pub plane: Option<ImagePlane>,
    ///// The size of the original image (not the tile)
    //pub full_image_size: ImageSize,
    ///// Image bit depth: 8, 16, 32
    //pub nr_bits: u8,
    //// Sizes of the image pixels in nm
    //pub pixel_sizes: PixelSizes,
}

impl<T, const C: usize> Deref for ManagedImage<T, C> {
    type Target = Image<T, C, CpuAllocator>;

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
                let new_img = kornia_image::Image::from_size_val(img.size(), 0.0, CpuAllocator)
                    .expect("Failed to allocate scratch buffer");
                ImageContainer::F32Gray(ManagedImage {
                    data: new_img,
                    tile_offset: img.tile_offset.clone(),
                    plane: img.plane.clone(),
                })
            }
            ImageContainer::F32Rgb(img) => {
                let new_img = kornia_image::Image::from_size_val(img.size(), 0.0, CpuAllocator)
                    .expect("Failed to allocate scratch buffer");
                ImageContainer::F32Rgb(ManagedImage {
                    data: new_img,
                    tile_offset: img.tile_offset.clone(),
                    plane: img.plane.clone(),
                })
            }
            ImageContainer::U32(img) => {
                let new_img = kornia_image::Image::from_size_val(img.size(), 0u32, CpuAllocator)
                    .expect("Failed to allocate scratch buffer");
                ImageContainer::U32(ManagedImage {
                    data: new_img,
                    tile_offset: img.tile_offset.clone(),
                    plane: img.plane.clone(),
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
/// overflow along the way.
///
/// `width`/`height`/`color_channels` ultimately come from file-declared
/// metadata (`PyramidInfo`'s fields are read straight from whatever the
/// image file/BioFormats reports) - a corrupted or malicious file can claim
/// dimensions large enough to overflow a plain `width * height * channels *
/// bytes_per_pixel` multiplication. Wrapping silently there would compute an
/// undersized buffer that the JNI call still fills with the *real*
/// width*height tile - i.e. a heap buffer overflow, not just a bad error
/// message - so every step is checked and any overflow is turned into a
/// clean error instead.
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
    // pub(crate) (rather than private) so sibling modules - e.g.
    // image_ome_parser's tests, which construct a JVM-free `ImageReader` via
    // struct literal to unit-test `parse_ome_xml` in isolation - can build
    // one without going through `new()`'s JNI call.
    pub(crate) wrapper_instance: Option<Global<JObject<'static>>>,
    pub(crate) read_mode: ReadMode,
    pub image_meta: Arc<ImageMeta>,
    pub(crate) current_path: PathBuf,
}

#[derive(PartialEq, Eq, Debug)]
pub enum ReadMode {
    Default,
    SplitChannels, // Split RGB channels of an image to three individual channels
}

/// A high-performance image reader that interfaces with Bio-Formats via JNI.
///
/// `ImageReader` manages the lifecycle of a Java-side Bio-Formats reader instance.
/// It uses a lifetime `'a` to ensure it does not outlive the [`JavaWrapper`]
/// providing the JVM and cached method IDs.
impl ImageReader {
    /// Creates a new `ImageReader` instance and initializes the Java-side object.
    ///
    /// # Arguments
    /// * `wrapper` - A reference to the initialized JVM wrapper containing cached IDs.
    /// * `image_path` - The filesystem path to the image to be opened.
    /// * `mode` - If true an RGB image is represented by three individual channels
    ///
    /// # Errors
    /// Returns an error if the JVM is not initialized, if the Bio-Formats class is missing,
    /// or if the Java constructor throws an exception.
    pub fn new(path: &PathBuf, mode: ReadMode) -> Result<Self, InternalErrors> {
        let Some(image_path) = path.to_str() else {
            return Err(InternalErrors::Internal("Wrong path".into()));
        };

        if !path.exists() {
            return Err(InternalErrors::Io(format!(
                "File '{:?}' not existing",
                path
            )));
        }

        // Prepare split channel
        let split_rgb_channel = match mode {
            ReadMode::Default => false,
            ReadMode::SplitChannels => true,
        };

        // Initial Checks. Starts the JVM lazily on first use if it hasn't
        // been warmed up already (see the background warmup spawned by the
        // GUI at startup, in `evanalyzer_gui::run`).
        let wrapper = ensure_java_wrapper()?;

        let jvm = wrapper.jvm.as_ref().ok_or("JVM not initialized")?;
        let class = wrapper
            .m_bioformats_class
            .as_ref()
            .ok_or("Class not loaded")?;
        let constructor_raw = wrapper.m_constructor.ok_or("Constructor ID not cached")?;

        // Attach the thread (permanently) and create the Java object.
        // jni 0.22's `attach_current_thread` only ever hands out an `Env`
        // reference within a closure (never an owned guard), so every JNI
        // call that needs it happens inside this closure.
        let global_instance = jvm
            .attach_current_thread(|env| -> jni::errors::Result<_> {
                let path_arg = env.new_string(image_path)?;

                // Create the Java Object instance
                let instance = unsafe {
                    let method_id = JMethodID::from_raw(constructor_raw);

                    // It's safer to use the 'env' to create the object
                    let obj = env.new_object_unchecked(
                        class,
                        method_id,
                        &[
                            jni::objects::JValue::from(&path_arg).as_jni(),
                            jni::objects::JValue::from(split_rgb_channel).as_jni(),
                        ],
                    )?;

                    // Check if the constructor threw an exception (e.g., IOException)
                    if env.exception_check() {
                        env.exception_describe(); // Prints the stack trace to stderr
                        env.exception_clear();
                        return Err(JniError::JavaException);
                    }
                    obj
                };

                // Wrap the local reference into a Global ref - this ensures
                // the Java object stays alive even after this closure exits.
                env.new_global_ref(instance)
            })
            .map_err(|e| match e {
                JniError::JavaException => {
                    InternalErrors::JvmError("Java constructor threw an exception".to_string())
                }
                e => InternalErrors::JvmError(e.to_string()),
            })?;

        // Return the fully constructed struct
        let mut reader = Self {
            read_mode: mode,
            wrapper_instance: Some(global_instance),
            image_meta: Arc::new(ImageMeta::default()),
            current_path: image_path.into(),
        };
        let start = Instant::now();
        reader.image_meta = Arc::new(reader.read_image_meta()?);
        let duration = start.elapsed();
        info!("Executed ReadImageMeta in {:?}", duration);
        Ok(reader)
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

    /// Read image meta information (OME xml is used)
    ///
    /// # Arguments
    ///
    /// - `&self` (`undefined`) - Describe this parameter.
    ///
    /// # Returns
    ///
    /// - `Result<ImageMeta, Box<dyn std::error::Error>>` - Describe the return value.
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
    /// let _ = read_image_meta();
    /// ```
    fn read_image_meta(&self) -> Result<ImageMeta, InternalErrors> {
        let wrapper = ensure_java_wrapper()?;

        let jvm = wrapper
            .jvm
            .as_ref()
            .ok_or("JVM not initialized")
            .map_err(|e| InternalErrors::JvmError(e.to_string()))?;
        let instance = self
            .wrapper_instance
            .as_ref()
            .ok_or("Reader instance is null")?;
        let method_id_raw = wrapper.m_get_image_properties.ok_or("Method ID missing")?;

        let rust_string = jvm
            .attach_current_thread(|env| -> jni::errors::Result<String> {
                let method_id = unsafe { JMethodID::from_raw(method_id_raw) };
                let result = unsafe {
                    env.call_method_unchecked(instance, method_id, ReturnType::Object, &[])?
                };
                let jstring_obj = result.l()?;
                let jstring = env.cast_local::<jni::objects::JString>(jstring_obj)?;
                let s = jstring.try_to_string(env)?;
                Ok(s)
            })
            .map_err(|e| InternalErrors::JvmError(e.to_string()))?;

        self.parse_ome_xml(rust_string.as_str())
    }

    /// Reads a tile from the image directly into a Rust buffer.
    ///
    /// # Safety
    /// This function performs a zero-copy transfer by creating a `DirectByteBuffer`.
    /// The caller must ensure the `JavaWrapper` remains valid for the duration of the call.
    fn read_image_tile(
        &self,
        series: i32,
        resolution_idx: i32,
        image_plane: &ImagePlane,
        image_tile: &ImageTile,
        byte_scratch: &mut Vec<u8>,
    ) -> Result<ImageContainer, InternalErrors> {
        // 1. Setup Environment and Instance
        let wrapper = ensure_java_wrapper()?;
        let jvm = wrapper.jvm.as_ref().ok_or("JVM not initialized")?;
        let instance = self
            .wrapper_instance
            .as_ref()
            .ok_or("Reader instance is null")?;

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

        // 3. Get pyramid of full image
        let _pyramid_info_full = series_info.resolutions.get(&0).ok_or_else(|| {
            InternalErrors::ImageReadError(format!(
                "Pyramid {} does not exist in series {}",
                resolution_idx, series
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

        // Bail before doing any buffer allocation or JNI call at all if the
        // metadata declares an implausible bit depth (see the matching guard
        // in `decode_image`, which this also protects against reaching).
        // `0` means BitsPerPixel was never set; anything above 32 is outside
        // every real scientific image format and would make
        // `(1u64 << nr_bits) - 1` (in `decode_image`) a shift-by-too-large,
        // silently wrong in a release build rather than an error.
        if !(1..=32).contains(&pyramid_info.nr_bits) {
            return Err(InternalErrors::ImageReadError(format!(
                "Series {series}, resolution {resolution_idx} reports an implausible bit depth \
                 of {} (expected 1-32, missing or corrupt BitsPerPixel metadata) - cannot read tile",
                pyramid_info.nr_bits
            )));
        }

        // Prepare Buffer - Use Boxed Slice for more stable heap allocation
        let nr_bytes = (pyramid_info.nr_bits as f32 / 8.0).ceil() as usize;
        let buffer_size: usize =
            checked_tile_buffer_size(width, height, pyramid_info.color_channels, nr_bytes)?;

        // Check if JAVA VM has enough reserved memory for loading the image
        let required_bytes = buffer_size as u64;
        let available_bytes = wrapper.m_reserved_ram;
        if required_bytes > available_bytes {
            return Err(InternalErrors::JvmError(
                format!(
                    "JVM Memory Limit Exceeded: Requested {}, but only {} is reserved.",
                    required_bytes, available_bytes
                )
                .into(),
            ));
        }

        // Reuse the caller's scratch allocation when it's already the right
        // size (e.g. across Z-slices of the same tile/channel, where width,
        // height, bit depth and channel count never change) instead of
        // allocating a fresh buffer on every call. The Java call below always
        // overwrites the full `buffer_size` region, so stale bytes left over
        // from a previous read are never observed.
        if byte_scratch.len() != buffer_size {
            byte_scratch.resize(buffer_size, 0u8);
        }
        let buffer = byte_scratch;

        // Resolve Method ID
        let method_id_raw = wrapper.m_read_image_tile.ok_or("Method ID missing")?;

        // Create Direct View (Zero-Copy), invoke the Java call on INSTANCE
        // (not class) and check for exceptions - all inside the attach
        // closure, since jni 0.22 only ever hands out an `Env` reference
        // within one.
        // IMPORTANT: 'buffer' must not be dropped while 'direct_buffer' is in use by Java.
        jvm.attach_current_thread(|env| -> jni::errors::Result<()> {
            let direct_buffer =
                unsafe { env.new_direct_byte_buffer(buffer.as_mut_ptr(), buffer_size)? };

            let method_id = unsafe { JMethodID::from_raw(method_id_raw) };

            unsafe {
                env.call_method_unchecked(
                    instance,
                    method_id,
                    ReturnType::Primitive(Primitive::Void),
                    &[
                        JValue::from(&direct_buffer).as_jni(),
                        JValue::from(series).as_jni(),
                        JValue::from(resolution_idx).as_jni(),
                        JValue::from(image_plane.z).as_jni(),
                        JValue::from(image_plane.c).as_jni(),
                        JValue::from(image_plane.t).as_jni(),
                        JValue::from(image_tile.offset_x as i32).as_jni(),
                        JValue::from(image_tile.offset_y as i32).as_jni(),
                        JValue::from(width as i32).as_jni(),
                        JValue::from(height as i32).as_jni(),
                    ],
                )?;
            }

            // Check for Java Exceptions (Crucial for JNI debugging)
            if env.exception_check() {
                env.exception_describe(); // Prints stack trace to stderr
                env.exception_clear();
                return Err(JniError::JavaException);
            }
            Ok(())
        })
        .map_err(|e| match e {
            JniError::JavaException => {
                InternalErrors::ImageReadError("Java exception during tile read".to_string())
            }
            e => InternalErrors::JvmError(e.to_string()),
        })?;

        decode_image(
            buffer.as_slice(),
            pyramid_info.is_interleaved,
            pyramid_info.is_little_endian,
            ImageSize {
                width: width,
                height: height,
            },
            pyramid_info.nr_bits,
            pyramid_info.color_channels,
            image_tile.clone(),
            image_plane.clone(),
        )
    }

    /// Reads every requested channel's (possibly Z-projected) tile using
    /// `self` alone - every channel shares the one BioFormats reader,
    /// serialized through its `synchronized(formatReader)` block (see
    /// `BioFormatsWrapper.java`). Used by callers that already parallelize
    /// at a coarser level (e.g. `job_executor.rs` parallelizes across
    /// tiles), where adding a second layer of parallelism here would just
    /// contend rayon workers against that same lock for no gain.
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
        Self::read_image_tile_combined_impl(
            std::slice::from_ref(&self),
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

    /// Same as [`Self::read_image_tile_combined`], but distributes the
    /// independent (channel, Z-slice) reads across `pool` in parallel
    /// instead of running them one at a time on a single reader.
    ///
    /// Each `ImageReader` wraps its own BioFormats `IFormatReader` object,
    /// which is not safe to call from multiple threads at once - see the
    /// `synchronized(formatReader)` block in `BioFormatsWrapper.java`, which
    /// only guarantees no corruption, not concurrency. `pool` must contain
    /// genuinely distinct `ImageReader` instances opened on the same path
    /// (e.g. from a reader pool built after the first instance's `setId()`
    /// call has populated the BioFormats `Memoizer` cache, so the rest are
    /// cheap to construct), not clones of the same one - each channel's
    /// reads are pinned to one pool member (`pool[i % pool.len()]`), so two
    /// channels never share a reader concurrently.
    pub fn read_image_tile_combined_pooled(
        pool: &[Arc<ImageReader>],
        series: i32,
        resolution_idx: i32,
        z_projection: ZProjection,
        z_range: &Option<RangeInclusive<i32>>,
        t_stack: i32,
        c_stacks_in: Option<&Vec<i32>>,
        image_tile: &ImageTile,
    ) -> Result<Vec<ImageChannel>, InternalErrors> {
        if pool.is_empty() {
            return Err(InternalErrors::Internal("Reader pool is empty".into()));
        }
        let refs: Vec<&ImageReader> = pool.iter().map(Arc::as_ref).collect();
        Self::read_image_tile_combined_impl(
            &refs,
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
    /// [`Self::read_image_tile_combined_pooled`]. `pool[0]` is used for
    /// metadata lookups (every pool member was opened on the same path, so
    /// their metadata is equivalent); each channel's reads are pinned to
    /// `pool[i % pool.len()]`. `parallel` selects rayon's work-stealing
    /// iteration over the channel list vs. a plain sequential one - see the
    /// two public wrappers for why that choice is per-caller, not automatic.
    fn read_image_tile_combined_impl(
        pool: &[&ImageReader],
        series: i32,
        resolution_idx: i32,
        z_projection: ZProjection,
        z_range: &Option<RangeInclusive<i32>>,
        t_stack: i32,
        c_stacks_in: Option<&Vec<i32>>,
        image_tile: &ImageTile,
        parallel: bool,
    ) -> Result<Vec<ImageChannel>, InternalErrors> {
        let primary = pool[0];
        let series_info = primary.image_meta.series.get(&series).ok_or_else(|| {
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
                    return None;
                }
                if primary.read_mode == ReadMode::Default
                    && pyramid_info.is_rgb
                    && c_stack_to_read > &0
                {
                    return None;
                }
                Some(c_stack_to_read)
            })
            .collect();

        // Reads one channel's (possibly Z-projected) tile from
        // `pool[i % pool.len()]`. Each channel gets its own scratch buffer
        // and pinned reader, so this is safe to call concurrently across
        // channels - two channels never share a reader.
        let read_channel =
            |i: usize, c_stack_to_read: &i32| -> Result<ImageChannel, InternalErrors> {
                let reader = pool[i % pool.len()];

                // --- PREP READ PARAMETERS ---
                let t_read = t_stack.min(series_info.nr_t_stacks.saturating_sub(1));
                let c_read = *c_stack_to_read.min(&series_info.nr_c_stacks.saturating_sub(1));

                // Reused for every read below (the initial load and every
                // Z-slice): width/height/bit depth/channel count are constant
                // across a single channel's Z-stack, so this scratch buffer
                // only ever grows once instead of reallocating per slice.
                let mut byte_scratch: Vec<u8> = Vec::new();

                // --- INITIAL IMAGE LOAD ---
                let mut image = reader.read_image_tile(
                    series,
                    resolution_idx,
                    &ImagePlane {
                        z: z_stack_range.start().clone(),
                        c: c_read,
                        t: t_read,
                    },
                    image_tile,
                    &mut byte_scratch,
                )?;

                // --- Z-PROJECTION LOGIC ---
                if z_projection != ZProjection::None && !pyramid_info.is_rgb {
                    if let ImageContainer::F32Gray(mut gray_image) = image {
                        for z in (z_stack_range.start() + 1)..=z_stack_range.end().clone() {
                            let image_tmp = reader.read_image_tile(
                                series,
                                resolution_idx,
                                &ImagePlane {
                                    z,
                                    c: c_read,
                                    t: t_read,
                                },
                                image_tile,
                                &mut byte_scratch,
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

                // --- METADATA & FINAL OBJECT ---
                let channel_meta = series_info.channels.get(&c_stack_to_read).ok_or_else(|| {
                    InternalErrors::ImageReadError(format!("Series {} does not exist", series))
                })?;

                Ok(ImageChannel {
                    image: Arc::new(image),
                    color: wavelength_to_rgb_float(channel_meta.emission_wave_length),
                    is_visible: true,
                    c_stack: *c_stack_to_read,
                    name: channel_meta.name.clone(),
                    is_rgb: pyramid_info.is_rgb,
                })
            };

        let resulting_images: Vec<ImageChannel> = if parallel {
            selected_channels
                .into_par_iter()
                .enumerate()
                .map(|(i, c)| read_channel(i, c))
                .collect::<Result<Vec<_>, InternalErrors>>()?
        } else {
            // BioFormats IFormatReader is NOT thread-safe: it holds mutable
            // internal state (current series, plane, file position). With a
            // single-element pool this loop always hits the same reader, so
            // it stays sequential rather than paying rayon dispatch
            // overhead for tasks that would just serialize on the reader's
            // Java-side lock anyway.
            selected_channels
                .into_iter()
                .enumerate()
                .map(|(i, c)| read_channel(i, c))
                .collect::<Result<Vec<_>, InternalErrors>>()?
        };

        Ok(resulting_images)
    }
}

/// Decode the image based on the image meta data
///
/// # Arguments
///
/// - `buffer` (`&[u8]`) - Describe this parameter.
///
/// # Returns
///
/// - `Result<ImageContainer, InternalErrors>` - Describe the return value.
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
/// let _ = decode_image();
/// ```
fn decode_image(
    buffer: &[u8],
    is_interleaved: bool,
    is_little_endian: bool,
    image_size: ImageSize,
    nr_bits: u8,
    color_channels: u8,
    image_tile: ImageTile,
    plane: ImagePlane,
) -> Result<ImageContainer, InternalErrors> {
    // `nr_bits == 0` means the source metadata never declared a bit depth
    // (e.g. a pyramid sub-resolution omitting `BitsPerPixel` - plausible for
    // real-world files, not just corrupt ones). Every decode path below
    // divides by or chunks on a byte-per-sample count derived from `nr_bits`,
    // which is zero in that case: `par_chunks_exact(0)` panics unconditionally
    // and `buffer.len() / (bytes_per_sample * source_planes)` divides by zero.
    // The upper bound matters too: `(1u64 << nr_bits) - 1` below is a
    // shift-by-too-large for anything beyond 63 (undefined in principle, and
    // in a release build - no `overflow-checks` - silently wrong rather than
    // a panic), and `read_le`/`read_be` panic outright once
    // `bytes_per_sample = (nr_bits + 7) / 8` exceeds 8. No real image format
    // needs more than 32 bits per sample, so reject anything past that here
    // rather than let a corrupt/implausible value reach either failure mode.
    if !(1..=32).contains(&nr_bits) {
        return Err(InternalErrors::ImageReadError(format!(
            "Image reports an implausible bit depth of {nr_bits} (expected 1-32, missing or \
             corrupt BitsPerPixel metadata) - cannot decode"
        )));
    }

    let max_val = (1u64 << nr_bits) - 1;
    let inv_divisor = 1.0 / (max_val as f32);

    let final_data = match (color_channels, is_interleaved) {
        // Grayscale pass through
        (1, _) => decode_samples_parallel(buffer, nr_bits, is_little_endian, inv_divisor),

        // RGB interleaved (still correct layout: RGBRGB...)
        (3, true) => decode_samples_parallel(buffer, nr_bits, is_little_endian, inv_divisor),

        // RGB interleaved with alpha channel -> Remove alpha channel
        (4, true) => {
            let raw_f32 = decode_samples_parallel(buffer, nr_bits, is_little_endian, inv_divisor);
            raw_f32
                .chunks_exact(4)
                .flat_map(|rgba| [rgba[0], rgba[1], rgba[2]]) // Nimm nur R, G, B
                .collect()
        }

        // RGB planar (RRR...GGG...BBB...) -> interleaved (RGBRGB...), decoded
        // directly from the source bytes: no full-size intermediate buffer is
        // ever materialized (unlike the interleaved paths above, which can
        // reuse the parallel decode as-is since no reordering is needed).
        (3, false) => {
            decode_planar_to_interleaved(buffer, nr_bits, is_little_endian, inv_divisor, 3)
        }

        // RGBA planar with alpha channel -> RGB interleaved, alpha plane bytes
        // are never even read.
        (4, false) => {
            decode_planar_to_interleaved(buffer, nr_bits, is_little_endian, inv_divisor, 4)
        }

        _ => return Err(InternalErrors::ImageReadError("".to_string())),
    };

    // Convert to korina-rs image tensor
    if color_channels >= 3 {
        let img = Image::<f32, 3, CpuAllocator>::new(image_size, final_data, CpuAllocator)
            .map_err(InternalErrors::from_kornia)?;
        Ok(ImageContainer::F32Rgb(ManagedImage {
            data: img,
            tile_offset: Point2d {
                x: image_tile.offset_x,
                y: image_tile.offset_y,
            },
            plane: Some(plane),
        }))
    } else {
        let img = Image::<f32, 1, CpuAllocator>::new(image_size, final_data, CpuAllocator)
            .map_err(InternalErrors::from_kornia)?;
        Ok(ImageContainer::F32Gray(ManagedImage {
            data: img,
            tile_offset: Point2d {
                x: image_tile.offset_x,
                y: image_tile.offset_y,
            },
            plane: Some(plane),
        }))
    }
}

/// Converts a raw byte buffer into normalized `f32` samples in parallel, one
/// sample per pixel-channel, preserving the buffer's original sample order.
/// Use direct integer constructors for the two most common bit depths to
/// avoid the 8-byte stack-buffer allocation that read_le/read_be would do.
fn decode_samples_parallel(
    buffer: &[u8],
    nr_bits: u8,
    is_little_endian: bool,
    inv_divisor: f32,
) -> Vec<f32> {
    match nr_bits {
        8 => buffer.par_iter().map(|&b| b as f32 * inv_divisor).collect(),
        16 => buffer
            .par_chunks_exact(2)
            .map(|c| {
                let v = if is_little_endian {
                    u16::from_le_bytes([c[0], c[1]])
                } else {
                    u16::from_be_bytes([c[0], c[1]])
                };
                v as f32 * inv_divisor
            })
            .collect(),
        _ => {
            let bytes_per_sample = (nr_bits as usize + 7) / 8;
            buffer
                .par_chunks_exact(bytes_per_sample)
                .map(|chunk| {
                    let val = if is_little_endian {
                        read_le(chunk)
                    } else {
                        read_be(chunk)
                    };
                    val as f32 * inv_divisor
                })
                .collect()
        }
    }
}

/// Reads and normalizes a single sample at `byte_offset`, matching
/// [`decode_samples_parallel`]'s per-bit-depth semantics exactly.
fn sample_f32(
    buffer: &[u8],
    byte_offset: usize,
    nr_bits: u8,
    is_little_endian: bool,
    inv_divisor: f32,
) -> f32 {
    match nr_bits {
        8 => buffer[byte_offset] as f32 * inv_divisor,
        16 => {
            let v = if is_little_endian {
                u16::from_le_bytes([buffer[byte_offset], buffer[byte_offset + 1]])
            } else {
                u16::from_be_bytes([buffer[byte_offset], buffer[byte_offset + 1]])
            };
            v as f32 * inv_divisor
        }
        _ => {
            let bytes_per_sample = (nr_bits as usize + 7) / 8;
            let chunk = &buffer[byte_offset..byte_offset + bytes_per_sample];
            let val = if is_little_endian {
                read_le(chunk)
            } else {
                read_be(chunk)
            };
            val as f32 * inv_divisor
        }
    }
}

/// Decodes a planar (RRR...GGG...BBB...[AAA...]) byte buffer directly into an
/// interleaved RGB `f32` buffer. `source_planes` is 3 for RGB or 4 for RGBA
/// source data; only the first three planes are ever read, so an alpha plane
/// (if present) is dropped without its bytes ever being touched. Unlike the
/// interleaved decode path, this never materializes a full-size normalized
/// copy of the source planes before reordering them. The per-pixel work is
/// independent, so it's parallelized the same way the byte decode it
/// replaces already was.
fn decode_planar_to_interleaved(
    buffer: &[u8],
    nr_bits: u8,
    is_little_endian: bool,
    inv_divisor: f32,
    source_planes: usize,
) -> Vec<f32> {
    let bytes_per_sample = (nr_bits as usize + 7) / 8;
    let n = buffer.len() / (bytes_per_sample * source_planes);
    let plane_stride = n * bytes_per_sample;

    let mut interleaved = vec![0f32; n * 3];
    interleaved
        .par_chunks_mut(3)
        .enumerate()
        .for_each(|(i, px)| {
            let base = i * bytes_per_sample;
            px[0] = sample_f32(buffer, base, nr_bits, is_little_endian, inv_divisor);
            px[1] = sample_f32(
                buffer,
                plane_stride + base,
                nr_bits,
                is_little_endian,
                inv_divisor,
            );
            px[2] = sample_f32(
                buffer,
                plane_stride * 2 + base,
                nr_bits,
                is_little_endian,
                inv_divisor,
            );
        });
    interleaved
}

// Helper functions for byte read
fn read_le(chunk: &[u8]) -> u64 {
    let mut buf = [0u8; 8];
    buf[..chunk.len()].copy_from_slice(chunk);
    u64::from_le_bytes(buf)
}

fn read_be(chunk: &[u8]) -> u64 {
    let mut buf = [0u8; 8];
    let start = 8 - chunk.len();
    buf[start..].copy_from_slice(chunk);
    u64::from_be_bytes(buf)
}

/// Destroy the java instance
///
/// # Arguments
///
/// - `&mut self` (`undefined`) - Describe this parameter.
///
/// # Examples
///
/// ```
/// use crate::...;
///
/// let _ = drop();
/// ```
impl Drop for ImageReader {
    fn drop(&mut self) {
        // Nothing to close - either the JVM close-out already ran once (this
        // `take()`s the field), or this reader never had a live Java object
        // to begin with (e.g. one built directly via struct literal for a
        // unit test that only exercises pure-Rust parsing code, bypassing
        // `new()`'s JNI call entirely). Either way, there is no live object
        // for the JVM lookup below to act on, so skip it - a `Drop` impl
        // must never panic, and requiring an initialized JVM just to drop a
        // reader with nothing to clean up would do exactly that.
        let Some(instance) = self.wrapper_instance.take() else {
            return;
        };
        let wrapper = JAVA_WRAPPER
            .get()
            .expect("Java Runtime not initialized, call init_java_wrapper");
        let (Some(jvm), Some(close_raw)) = (&wrapper.jvm, wrapper.m_close) else {
            return;
        };
        let _ = jvm.attach_current_thread(|env| -> jni::errors::Result<()> {
            let method_id = unsafe { JMethodID::from_raw(close_raw) };
            unsafe {
                let _ = env.call_method_unchecked(
                    &instance,
                    method_id,
                    ReturnType::Primitive(Primitive::Void), // Equivalent to 'Void' in CallVoidMethod
                    &[],                                    // No arguments)
                );
            }
            Ok(())
        });
        // `instance` (a `Global<JObject<'static>>`) drops here; jni 0.22's
        // `Global` attaches the thread internally as needed for
        // `DeleteGlobalRef`, so no explicit attachment is required for that.
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::init_java_wrapper;
    use approx::relative_eq;
    use jni::objects::Reference;
    use jni::{jni_sig, jni_str};
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
        init_java_wrapper(1000000000).unwrap();
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
        init_java_wrapper(1000000000).unwrap();
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
        init_java_wrapper(1000000000).unwrap();

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

    #[test]
    fn take_the_middle_projection_returns_the_middle_slice_not_the_first() {
        init_java_wrapper(1000000000).unwrap();

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
    fn pooled_parallel_read_matches_sequential_read() {
        init_java_wrapper(1000000000).unwrap();

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

        // Sequential baseline: one reader, every channel's Z-stack read one
        // (z, c) plane at a time, exactly as read_image_tile_combined always has.
        let sequential_reader = ImageReader::new(&path.into(), ReadMode::SplitChannels).unwrap();
        let mut sequential = sequential_reader
            .read_image_tile_combined(0, 0, ZProjection::MaxIntensity, &None, 0, None, &tile)
            .unwrap();

        // Pooled: independent readers (not clones - each opened separately,
        // same as ReaderPool does), channels distributed round-robin across
        // them and read concurrently.
        let pool: Vec<Arc<ImageReader>> = (0..3)
            .map(|_| Arc::new(ImageReader::new(&path.into(), ReadMode::SplitChannels).unwrap()))
            .collect();
        let mut pooled = ImageReader::read_image_tile_combined_pooled(
            &pool,
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
                        "channel {} pixels differ between sequential and pooled reads",
                        seq_ch.c_stack
                    );
                }
                other => panic!("unexpected variant combination: {other:?}"),
            }
        }
    }

    #[test]
    fn test_minimum_intensity_projection() {
        init_java_wrapper(1000000000).unwrap();

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
        init_java_wrapper(1000000000).unwrap();

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
    fn decode_image_planar_rgb_8bit_interleaves_correctly() {
        // 2x2 image, 8-bit planar RGB: R plane, then G plane, then B plane.
        let buffer: Vec<u8> = vec![
            10, 20, 30, 40, // R
            50, 60, 70, 80, // G
            90, 100, 110, 120, // B
        ];

        let result = decode_image(
            &buffer,
            false,
            true,
            ImageSize {
                width: 2,
                height: 2,
            },
            8,
            3,
            ImageTile::default(),
            ImagePlane { z: 0, c: 0, t: 0 },
        )
        .unwrap();

        match result {
            ImageContainer::F32Rgb(img) => {
                let expected: Vec<f32> = [10, 50, 90, 20, 60, 100, 30, 70, 110, 40, 80, 120]
                    .iter()
                    .map(|&v| v as f32 / 255.0)
                    .collect();
                compare_data(&expected, img.as_slice(), 1e-6);
            }
            other => panic!("expected F32Rgb, got {other:?}"),
        }
    }

    #[test]
    fn decode_image_planar_rgba_16bit_drops_alpha_and_interleaves() {
        // 2x1 image, 16-bit little-endian planar RGBA: R, G, B, A planes.
        let samples: [u16; 8] = [
            1000, 2000, // R
            3000, 4000, // G
            5000, 6000, // B
            9999, 9999, // A (must be dropped)
        ];
        let mut buffer = Vec::with_capacity(samples.len() * 2);
        for s in samples {
            buffer.extend_from_slice(&s.to_le_bytes());
        }

        let result = decode_image(
            &buffer,
            false,
            true,
            ImageSize {
                width: 2,
                height: 1,
            },
            16,
            4,
            ImageTile::default(),
            ImagePlane { z: 0, c: 0, t: 0 },
        )
        .unwrap();

        match result {
            ImageContainer::F32Rgb(img) => {
                let inv = 1.0 / 65535.0;
                let expected: Vec<f32> = [1000, 3000, 5000, 2000, 4000, 6000]
                    .iter()
                    .map(|&v| v as f32 * inv)
                    .collect();
                compare_data(&expected, img.as_slice(), 1e-6);
            }
            other => panic!("expected F32Rgb, got {other:?}"),
        }
    }

    #[test]
    fn decode_image_grayscale_8bit_normalizes_samples() {
        let buffer: Vec<u8> = vec![0, 128, 255, 64];

        let result = decode_image(
            &buffer,
            true,
            true,
            ImageSize {
                width: 2,
                height: 2,
            },
            8,
            1,
            ImageTile::default(),
            ImagePlane { z: 0, c: 0, t: 0 },
        )
        .unwrap();

        match result {
            ImageContainer::F32Gray(img) => {
                let expected: Vec<f32> = buffer.iter().map(|&v| v as f32 / 255.0).collect();
                compare_data(&expected, img.as_slice(), 1e-6);
            }
            other => panic!("expected F32Gray, got {other:?}"),
        }
    }

    #[test]
    fn decode_image_interleaved_rgb_16bit_little_endian_preserves_order() {
        // 1x2 image, 16-bit little-endian interleaved RGB: already RGBRGB..., so
        // decoding must preserve sample order (unlike the planar paths, which
        // reorder).
        let samples: [u16; 6] = [1000, 2000, 3000, 4000, 5000, 6000];
        let mut buffer = Vec::with_capacity(samples.len() * 2);
        for s in samples {
            buffer.extend_from_slice(&s.to_le_bytes());
        }

        let result = decode_image(
            &buffer,
            true,
            true,
            ImageSize {
                width: 2,
                height: 1,
            },
            16,
            3,
            ImageTile::default(),
            ImagePlane { z: 0, c: 0, t: 0 },
        )
        .unwrap();

        match result {
            ImageContainer::F32Rgb(img) => {
                let inv = 1.0 / 65535.0;
                let expected: Vec<f32> = samples.iter().map(|&v| v as f32 * inv).collect();
                compare_data(&expected, img.as_slice(), 1e-6);
            }
            other => panic!("expected F32Rgb, got {other:?}"),
        }
    }

    #[test]
    fn decode_image_interleaved_rgba_8bit_drops_alpha_channel() {
        // 1x2 image, 8-bit interleaved RGBA: RGBA RGBA.
        let buffer: Vec<u8> = vec![
            10, 20, 30, 255, // pixel 0 RGBA
            40, 50, 60, 128, // pixel 1 RGBA
        ];

        let result = decode_image(
            &buffer,
            true,
            true,
            ImageSize {
                width: 2,
                height: 1,
            },
            8,
            4,
            ImageTile::default(),
            ImagePlane { z: 0, c: 0, t: 0 },
        )
        .unwrap();

        match result {
            ImageContainer::F32Rgb(img) => {
                let expected: Vec<f32> = [10, 20, 30, 40, 50, 60]
                    .iter()
                    .map(|&v| v as f32 / 255.0)
                    .collect();
                compare_data(&expected, img.as_slice(), 1e-6);
            }
            other => panic!("expected F32Rgb, got {other:?}"),
        }
    }

    #[test]
    fn decode_image_unsupported_channel_count_returns_error() {
        let buffer: Vec<u8> = vec![0, 0, 0, 0];
        let result = decode_image(
            &buffer,
            true,
            true,
            ImageSize {
                width: 2,
                height: 1,
            },
            8,
            2, // neither grayscale (1) nor RGB(A) (3/4)
            ImageTile::default(),
            ImagePlane { z: 0, c: 0, t: 0 },
        );
        assert!(result.is_err());
    }

    #[test]
    fn decode_image_zero_bit_depth_returns_error_instead_of_panicking() {
        // Regression test: metadata that omits `BitsPerPixel` (plausible for
        // pyramid sub-resolutions in real, non-corrupt files - see
        // image_ome_parser.rs) surfaces as `nr_bits == 0`. Before this guard,
        // the grayscale/RGB-interleaved path panicked unconditionally on
        // `buffer.par_chunks_exact(0)` and the planar path divided by zero -
        // both a hard crash on opening a legitimate file.
        let buffer: Vec<u8> = vec![0, 0, 0, 0];
        let result = decode_image(
            &buffer,
            true,
            true,
            ImageSize {
                width: 2,
                height: 1,
            },
            0, // nr_bits
            1,
            ImageTile::default(),
            ImagePlane { z: 0, c: 0, t: 0 },
        );
        assert!(result.is_err());
    }

    #[test]
    fn decode_image_implausible_bit_depth_returns_error_instead_of_silently_wrong_output() {
        // Regression test: a corrupt/implausible `BitsPerPixel` (e.g. 200,
        // parsed straight from XML with no range check) previously made
        // `(1u64 << nr_bits) - 1` a shift-by-too-large - undefined in
        // principle, and silently wrong (not an error) in a release build
        // since this crate doesn't enable `overflow-checks`. Values above 64
        // additionally panic outright in `read_le`/`read_be`. Both must now
        // be rejected up front instead.
        let buffer: Vec<u8> = vec![0, 0, 0, 0];
        for nr_bits in [64, 200, 255] {
            let result = decode_image(
                &buffer,
                true,
                true,
                ImageSize {
                    width: 2,
                    height: 1,
                },
                nr_bits,
                1,
                ImageTile::default(),
                ImagePlane { z: 0, c: 0, t: 0 },
            );
            assert!(
                result.is_err(),
                "nr_bits={nr_bits} should be rejected, not silently produce wrong output"
            );
        }
    }

    #[test]
    fn decode_samples_parallel_generic_bit_depth_matches_16bit_semantics() {
        // nr_bits = 12 falls through to the generic (non-8/16) branch, which
        // reads via read_le/read_be into a u64 instead of u16::from_le_bytes -
        // for a 2-byte chunk both must normalize to the same value.
        let buffer: Vec<u8> = vec![0xFF, 0x00, 0x00, 0x10]; // LE: 0x00FF, 0x1000
        let inv_divisor = 1.0 / 4095.0;

        let little_endian = decode_samples_parallel(&buffer, 12, true, inv_divisor);
        assert_eq!(
            little_endian,
            vec![0x00FF as f32 * inv_divisor, 0x1000 as f32 * inv_divisor]
        );

        let big_endian = decode_samples_parallel(&buffer, 12, false, inv_divisor);
        assert_eq!(
            big_endian,
            vec![0xFF00 as f32 * inv_divisor, 0x0010 as f32 * inv_divisor]
        );
    }

    #[test]
    fn read_le_reconstructs_value_from_a_partial_byte_chunk() {
        assert_eq!(read_le(&[0x01, 0x02]), 0x0201);
        assert_eq!(read_le(&[0xFF]), 0xFF);
        assert_eq!(read_le(&[0x00, 0x00, 0x01]), 0x010000);
    }

    #[test]
    fn read_be_reconstructs_value_from_a_partial_byte_chunk() {
        assert_eq!(read_be(&[0x01, 0x02]), 0x0102);
        assert_eq!(read_be(&[0xFF]), 0xFF);
        assert_eq!(read_be(&[0x01, 0x00, 0x00]), 0x010000);
    }

    #[test]
    fn clone_empty_u32_returns_matching_variant_and_size() {
        let size = ImageSize {
            width: 4,
            height: 3,
        };
        let source = ImageContainer::U32(ManagedImage {
            data: Image::<u32, 1, CpuAllocator>::new(
                size,
                vec![7u32; size.width * size.height],
                CpuAllocator,
            )
            .unwrap(),
            tile_offset: Point2d { x: 1, y: 2 },
            plane: None,
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
    fn test_bigger_image_with_z_stack() {
        init_java_wrapper(1000000000).unwrap();

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

    /*
    #[test]
    fn test_sum_intensity_projection() {
        init_java_wrapper(1000000000).unwrap();

        let reader =
            ImageReader::new(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/multi-channel-4D-series.ome.tif"))
                .unwrap();
        let result = reader
            .read_image_tile_combined(
                0,
                ZProjection::SumIntensity,
                0,
                0,
                vec![0],
                ImageTile {
                    offset_x: 0,
                    offset_y: 0,
                    width: 0,
                    height: 0,
                },
            )
            .unwrap();

        let reference_data_f32 = read_raw_data(
            concat!(env!("CARGO_MANIFEST_DIR"), "/tests/slice_Z0_C0_T0_sum_intensity.raw"),
            32,
        );

        for image_channel in result {
            match image_channel.image {
                ImageContainer::F32Gray(image) => {
                    let slice = image.as_slice();
                    compare_data(&reference_data_f32, &slice, 1e-1);
                }
                ImageContainer::F32Rgb(_) => todo!(),
                ImageContainer::U32(_) => todo!(),
            }
        }
    }*/

    /// Resident set size of this process, in bytes, or `None` if it can't be
    /// determined on this platform.
    fn current_process_rss_bytes() -> Option<u64> {
        let pid = get_current_pid().ok()?;
        let mut sys = System::new();
        sys.refresh_processes(ProcessesToUpdate::Some(&[pid]), true);
        sys.process(pid).map(|p| p.memory())
    }

    /// Precise regression test for the jni 0.21 -> 0.22 port: `wrapper_instance`
    /// is now a `Global<JObject<'static>>` (renamed/generic from `GlobalRef`),
    /// whose `Drop` impl is responsible for calling `DeleteGlobalRef` (see
    /// `ImageReader`'s own `Drop` impl). If a future change ever left it
    /// un-dropped (e.g. captured into a cycle, or the delete call silently
    /// skipped), the underlying BioFormats Java object would stay pinned
    /// forever, even after the `ImageReader` itself is gone.
    ///
    /// This wraps the reader's Java object in a `java.lang.ref.WeakReference`
    /// *before* dropping the reader, then proves the opposite: after the
    /// reader drops and a full GC runs, the weak reference clears - meaning
    /// nothing on the Rust side still holds a strong reference to it.
    ///
    /// Note: an OS-level RSS measurement was tried first and rejected - a
    /// baseline loop against the pre-port (jni 0.21) code showed the JVM's
    /// own heap high-water mark alone grows by ~300MB over 200 open/read/close
    /// cycles on this fixture, and `System.gc()` does not shrink it back
    /// (committed heap pages aren't returned to the OS). That's normal,
    /// pre-existing JVM/BioFormats behavior unrelated to the Rust binding,
    /// but it makes RSS growth useless as a leak signal here - a genuine
    /// per-object leak would be lost in that noise. Reachability, checked via
    /// `WeakReference`, is unaffected by that noise and directly targets the
    /// exact mechanism the migration touched.
    #[test]
    fn dropping_a_reader_releases_its_global_reference_for_gc() {
        init_java_wrapper(1_000_000_000).unwrap();
        let path: PathBuf = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/multi-channel-4D-series.ome.tif"
        )
        .into();

        let reader = ImageReader::new(&path, ReadMode::Default).unwrap();
        let wrapper = ensure_java_wrapper().unwrap();
        let jvm = wrapper.jvm.as_ref().unwrap();

        // Grab the raw pointer (a plain `Copy` value, not borrowed from
        // `reader`) up front so building the `WeakReference` below doesn't
        // need to hold a borrow of `reader` across the later `drop(reader)`.
        let target_raw = reader
            .wrapper_instance
            .as_ref()
            .expect("reader must have a live Java instance")
            .as_raw();

        // Wrap the reader's underlying Java object in a `WeakReference`, held
        // via our own separate `Global` so it survives independently of
        // `reader`.
        let weak_ref: Global<JObject<'static>> = jvm
            .attach_current_thread(|env| -> jni::errors::Result<_> {
                // Safety: `target_raw` is still valid here - the reader (and
                // its global ref) hasn't been dropped yet.
                let target = unsafe { JObject::from_raw(env, target_raw) };
                let local_weak = env.new_object(
                    jni_str!("java/lang/ref/WeakReference"),
                    jni_sig!("(Ljava/lang/Object;)V"),
                    &[JValue::from(&target)],
                )?;
                env.new_global_ref(local_weak)
            })
            .unwrap();

        // Drop the reader - must release its `Global<JObject>` (and call the
        // Java-side `close()`), leaving nothing else holding a strong
        // reference to the wrapped Java object.
        drop(reader);

        // A `WeakReference` only clears once the JVM actually runs a
        // collection, so force a few, giving the collector a moment each
        // time in case reference processing lags behind the GC itself.
        let is_cleared = jvm
            .attach_current_thread(|env| -> jni::errors::Result<bool> {
                for _ in 0..5 {
                    env.call_static_method(
                        jni_str!("java/lang/System"),
                        jni_str!("gc"),
                        jni_sig!("()V"),
                        &[],
                    )?;
                    thread::sleep(std::time::Duration::from_millis(50));
                    let referent = env
                        .call_method(
                            &weak_ref,
                            jni_str!("get"),
                            jni_sig!("()Ljava/lang/Object;"),
                            &[],
                        )?
                        .l()?;
                    if referent.is_null() {
                        return Ok(true);
                    }
                }
                Ok(false)
            })
            .unwrap();

        assert!(
            is_cleared,
            "WeakReference to the BioFormats instance was not cleared after dropping the \
             ImageReader and forcing GC - something is still holding a strong reference to it \
             (possible Global-ref leak in the jni 0.22 port)"
        );
    }

    /// Companion sanity check to the precise leak test above: repeatedly
    /// opening, reading from, and dropping readers must keep working over
    /// many cycles without erroring (e.g. a real global-ref leak would
    /// eventually make the JVM log "JNI ERROR: global reference table
    /// overflow" and abort). Growth itself isn't asserted tightly here (see
    /// the note on the test above for why RSS is noisy for this workload) -
    /// only that the process stays well clear of runaway growth.
    #[test]
    fn repeated_open_read_close_does_not_error_or_blow_up_memory() {
        init_java_wrapper(1_000_000_000).unwrap();
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
            // `reader` (and its `wrapper_instance` global ref) drops here.
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

        // Wide bound: a *leak-free* run of this loop was observed to grow
        // ~200-300MB (JVM heap high-water mark, not returned to the OS -
        // see the note on the test above). This only guards against a
        // qualitatively different failure mode (e.g. leaking a whole tile
        // buffer's worth of memory per iteration).
        assert!(
            growth_mb < 1500.0,
            "process RSS grew by {growth_mb:.1} MB over {ITERATIONS} open/read/close cycles \
             (baseline {baseline_bytes} bytes, after {after_bytes} bytes) - well beyond normal \
             JVM heap high-water-mark growth for this workload"
        );
    }

    /// Confirms the ported `attach_current_thread` closures (see
    /// `ImageReader::new`, `read_image_meta`, `read_image_tile` in
    /// image_reader.rs, and `JavaWrapper::init` in java.rs) are safe to
    /// enter concurrently from several genuinely independent OS threads at
    /// once - e.g. a GUI thread and a background worker thread each opening
    /// their own reader - rather than only from rayon's own worker pool
    /// (already covered by `pooled_parallel_read_matches_sequential_read`).
    /// Every thread opens an independent reader on the same file and all
    /// must see identical channel data.
    #[test]
    fn concurrent_readers_on_independent_threads_produce_consistent_results() {
        init_java_wrapper(1_000_000_000).unwrap();
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
}
