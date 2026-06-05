use crate::converters::wavelength_to_rgb_float;
use crate::image::image_meta::{ImageMeta, ImagePlane, ImageTile};
use crate::image::java::JAVA_WRAPPER;
use evanalyzer_cfg::core_types::InternalErrors;
use jni::objects::{GlobalRef, JMethodID, JValue};
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
                let new_img = kornia_image::Image::from_size_val(img.size(), 0.0, CpuAllocator)
                    .expect("Failed to allocate scratch buffer");
                ImageContainer::F32Rgb(ManagedImage {
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

pub struct ImageReader {
    wrapper_instance: Option<GlobalRef>,
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

        // Initial Checks
        let wrapper = JAVA_WRAPPER
            .get()
            .expect("Java Runtime not initialized, call init_java_wrapper");

        let jvm = wrapper.jvm.as_ref().ok_or("JVM not initialized")?;
        let class = wrapper
            .m_bioformats_class
            .as_ref()
            .ok_or("Class not loaded")?;
        let constructor_raw = wrapper.m_constructor.ok_or("Constructor ID not cached")?;

        // Attach thread and prepare arguments
        let mut env = jvm
            .attach_current_thread_permanently()
            .map_err(|e| InternalErrors::JvmError(e.to_string()))?;
        let path_arg = env
            .new_string(image_path)
            .map_err(|e| InternalErrors::JvmError(e.to_string()))?;

        // Create the Java Object instance
        let instance = unsafe {
            let method_id = JMethodID::from_raw(constructor_raw);

            // It's safer to use the 'env' to create the object
            let obj = env
                .new_object_unchecked(
                    class,
                    method_id,
                    &[
                        jni::objects::JValue::from(&path_arg).as_jni(),
                        jni::objects::JValue::from(split_rgb_channel).as_jni(),
                    ],
                )
                .map_err(|e| InternalErrors::JvmError(e.to_string()))?;

            // Check if the constructor threw an exception (e.g., IOException)
            if env
                .exception_check()
                .map_err(|e| InternalErrors::JvmError(e.to_string()))?
            {
                env.exception_describe()
                    .map_err(|e| InternalErrors::JvmError(e.to_string()))?; // Prints the stack trace to stderr
                env.exception_clear()
                    .map_err(|e| InternalErrors::JvmError(e.to_string()))?;
                return Err("Java constructor threw an exception".into());
            }
            obj
        };

        // Wrap the local reference into a GlobalRef
        // This ensures the Java object stays alive even after 'env' is dropped
        let global_instance = env
            .new_global_ref(instance)
            .map_err(|e| InternalErrors::JvmError(e.to_string()))?;

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
        let wrapper = JAVA_WRAPPER
            .get()
            .expect("Java Runtime not initialized, call init_java_wrapper");

        let jvm = wrapper
            .jvm
            .as_ref()
            .ok_or("JVM not initialized")
            .map_err(|e| InternalErrors::JvmError(e.to_string()))?;
        let instance = self
            .wrapper_instance
            .as_ref()
            .ok_or("Reader instance is null")?;
        let mut env = jvm
            .attach_current_thread_permanently()
            .map_err(|e| InternalErrors::JvmError(e.to_string()))?;
        let method_id_raw = wrapper.m_get_image_properties.ok_or("Method ID missing")?;
        let method_id = unsafe { JMethodID::from_raw(method_id_raw) };

        unsafe {
            let result = env
                .call_method_unchecked(instance, method_id, ReturnType::Object, &[])
                .map_err(|e| InternalErrors::JvmError(e.to_string()))?;
            let jstring_obj = result
                .l()
                .map_err(|e| InternalErrors::JvmError(e.to_string()))?;
            let rust_string: String = env
                .get_string(&jstring_obj.into())
                .map_err(|e| InternalErrors::JvmError(e.to_string()))?
                .into();
            return Ok(self.parse_ome_xml(rust_string.as_str())?);
        }
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
    ) -> Result<ImageContainer, InternalErrors> {
        // 1. Setup Environment and Instance
        let wrapper = JAVA_WRAPPER
            .get()
            .expect("Java Runtime not initialized, call init_java_wrapper");
        let jvm = wrapper.jvm.as_ref().ok_or("JVM not initialized")?;
        let instance = self
            .wrapper_instance
            .as_ref()
            .ok_or("Reader instance is null")?;
        let mut env = jvm
            .attach_current_thread_permanently()
            .map_err(|e| InternalErrors::JvmError(e.to_string()))?;

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

        // Prepare Buffer - Use Boxed Slice for more stable heap allocation
        let nr_bytes = (pyramid_info.nr_bits as f32 / 8.0).ceil() as usize;
        let buffer_size: usize = width * height * (pyramid_info.color_channels as usize) * nr_bytes;

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

        let mut buffer = vec![0u8; buffer_size].into_boxed_slice();

        // Create Direct View (Zero-Copy)
        // IMPORTANT: 'buffer' must not be dropped while 'direct_buffer' is in use by Java.
        let direct_buffer = unsafe {
            env.new_direct_byte_buffer(buffer.as_mut_ptr(), buffer_size)
                .map_err(|e| InternalErrors::JvmError(e.to_string()))?
        };

        // Resolve Method ID
        let method_id_raw = wrapper.m_read_image_tile.ok_or("Method ID missing")?;
        let method_id = unsafe { JMethodID::from_raw(method_id_raw) };

        // Invoke Call on INSTANCE (not class)
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
            )
            .map_err(|e| InternalErrors::JvmError(e.to_string()))?;
        }

        // Check for Java Exceptions (Crucial for JNI debugging)
        if env
            .exception_check()
            .map_err(|e| InternalErrors::JvmError(e.to_string()))?
        {
            env.exception_describe().ok(); // Prints stack trace to stderr
            env.exception_clear().ok();
            return Err(InternalErrors::ImageReadError(
                "Java exception during tile read".to_string(),
            ));
        }
        decode_image(
            buffer,
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

    /// Describe this function.
    ///
    /// # Arguments
    ///
    /// - `series` (`i32`) - Describe this parameter.
    /// - `z_projection` (`ZProjection`) - Describe this parameter.
    /// - `z_stack` (`i32`) - Describe this parameter.
    /// - `t_stack` (`i32`) - Describe this parameter.
    /// - `c_stacks` (`Vec<i32>`) - Describe this parameter.
    /// - `image_tile` (`ImageTile`) - Describe this parameter.
    ///
    /// # Examples
    ///
    /// ```
    /// use crate::...;
    ///
    /// let _ = read_image_tile_combined();
    /// ```
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
        // BioFormats IFormatReader is NOT thread-safe: it holds mutable internal
        // state (current series, plane, file position).  Calling readImageTile
        // on the same Java instance from multiple threads corrupts that state.
        let resulting_images: Vec<ImageChannel> = c_stacks
            .iter()
            .filter_map(|c_stack_to_read| {
                if c_stack_to_read >= &series_info.nr_c_stacks {
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
            .map(|c_stack_to_read| {
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
                    &image_tile,
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
                                &image_tile,
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
                                    _ => {}
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
            })
            .collect::<Result<Vec<_>, InternalErrors>>()?; // Collects into a Result, bubble up error if any

        Ok(resulting_images)
    }
}

/// Decode the image based on the image meta data
///
/// # Arguments
///
/// - `buffer` (`Box<[u8]>`) - Describe this parameter.
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
    buffer: Box<[u8]>,
    is_interleaved: bool,
    is_little_endian: bool,
    image_size: ImageSize,
    nr_bits: u8,
    color_channels: u8,
    image_tile: ImageTile,
    plane: ImagePlane,
) -> Result<ImageContainer, InternalErrors> {
    let max_val = (1u64 << nr_bits) - 1;
    let inv_divisor = 1.0 / (max_val as f32);

    // Convert raw bytes to normalised f32 in parallel.
    // Use direct integer constructors for the two most common bit depths to
    // avoid the 8-byte stack-buffer allocation that read_le/read_be would do.
    let raw_f32: Vec<f32> = match nr_bits {
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
    };

    let final_data = match (color_channels, is_interleaved) {
        // Grayscal pass through
        (1, _) => raw_f32,

        // RGB interleaved (still correct layout: RGBRGB...)
        (3, true) => raw_f32,

        // RGB interleaved with alpha channel -> Remove alpha channel
        (4, true) => {
            raw_f32
                .chunks_exact(4)
                .flat_map(|rgba| [rgba[0], rgba[1], rgba[2]]) // Nimm nur R, G, B
                .collect()
        }

        // RGB planar (RRR...GGG...BBB...) -> convert to interleaved (RGBRGB...)
        (3, false) => {
            let n = raw_f32.len() / 3;
            let mut interleaved = Vec::with_capacity(raw_f32.len());
            for i in 0..n {
                interleaved.push(raw_f32[i]); // R
                interleaved.push(raw_f32[i + n]); // G
                interleaved.push(raw_f32[i + 2 * n]); // B
            }
            interleaved
        }

        // RGBA planar with alpha channel -> RGB interleaved remove alpha
        (4, false) => {
            let n = raw_f32.len() / 4;
            let mut interleaved = Vec::with_capacity(n * 3);
            for i in 0..n {
                interleaved.push(raw_f32[i]); // R
                interleaved.push(raw_f32[i + n]); // G
                interleaved.push(raw_f32[i + 2 * n]); // B
            }
            interleaved
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
        let wrapper = JAVA_WRAPPER
            .get()
            .expect("Java Runtime not initialized, call init_java_wrapper");
        let instance = self.wrapper_instance.take();
        if let Some(jvm) = &wrapper.jvm {
            if let Ok(mut env) = jvm.attach_current_thread() {
                if let (Some(obj), Some(close_raw)) = (&instance, wrapper.m_close) {
                    let method_id = unsafe { JMethodID::from_raw(close_raw) };
                    unsafe {
                        let _ = env.call_method_unchecked(
                            obj,
                            method_id,
                            ReturnType::Primitive(Primitive::Void), // Equivalent to 'Void' in CallVoidMethod
                            &[],                                    // No arguments)
                        );
                    }
                }
                // Explicitly drop the GlobalRef while the thread is attached
                drop(instance);
                return;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::init_java_wrapper;
    use approx::relative_eq;
    use std::fs;

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
            "/workspaces/evanalyzer/crates/core/tests/slice_Z0_C0_T0.raw",
            8,
        );

        let reader = ImageReader::new(
            &"/workspaces/evanalyzer/crates/core/tests/multi-channel-4D-series.ome.tif".into(),
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
            "/workspaces/evanalyzer/crates/core/tests/slice_Z1_C0_T0.raw",
            8,
        );

        let reader = ImageReader::new(
            &"/workspaces/evanalyzer/crates/core/tests/multi-channel-4D-series.ome.tif".into(),
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
            &"/workspaces/evanalyzer/crates/core/tests/multi-channel-4D-series.ome.tif".into(),
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
            "/workspaces/evanalyzer/crates/core/tests/slice_Z0_C0_T0_max_intensity.raw",
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
    fn test_minimum_intensity_projection() {
        init_java_wrapper(1000000000).unwrap();

        let reader = ImageReader::new(
            &"/workspaces/evanalyzer/crates/core/tests/multi-channel-4D-series.ome.tif".into(),
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
            "/workspaces/evanalyzer/crates/core/tests/slice_Z0_C0_T0_min_intensity.raw",
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
            &"/workspaces/evanalyzer/crates/core/tests/multi-channel-4D-series.ome.tif".into(),
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
            "/workspaces/evanalyzer/crates/core/tests/slice_Z0_C0_T0_avg_intensity.raw",
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
    fn test_bigger_image_with_z_stack() {
        init_java_wrapper(1000000000).unwrap();

        let reader = ImageReader::new(
            &"/workspaces/evanalyzer/crates/core/tests/muliple_z_stacks.nd2".into(),
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
            ImageReader::new("/workspaces/evanalyzer/crates/core/tests/multi-channel-4D-series.ome.tif")
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
            "/workspaces/evanalyzer/crates/core/tests/slice_Z0_C0_T0_sum_intensity.raw",
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
}
