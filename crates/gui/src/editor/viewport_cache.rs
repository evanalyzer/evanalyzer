use crate::UiState;
use crate::editor::viewport_controller::ViewportState;
use clru::{CLruCache, WeightScale};
use evanalyzer_app::extensions::project_ext::ProjectExt;
use evanalyzer_cfg::core_types::InternalErrors;
use evanalyzer_cfg::settings::images_settings::ZStackHandling;
use evanalyzer_core::{
    ImageChannel, ImageContainer, ImageTile, ManagedImage, PyramidInfo, ZProjection,
};
use kornia_image::allocator::CpuAllocator;
use kornia_image::{Image, InterpolationMode};
use kornia_imgproc::resize;
use log::error;
use std::collections::BTreeMap;
use std::num::NonZeroUsize;
use std::ops::RangeInclusive;
use std::sync::{Arc, Mutex, RwLock};
pub(crate) type TileCache = Mutex<
    CLruCache<TileKey, Arc<CachedTile>, std::collections::hash_map::RandomState, TileWeightScale>,
>;

const LOW_RES_MAX_WIDTH_AND_HEIGHT: u64 = 1024;
const TILE_SIZE_TO_LOAD: f32 = 1024.0;

#[derive(Debug, Clone, Default)]
pub struct ReadContext {
    pub zoomed_w: f32,
    pub zoomed_h: f32,
    pub zoom: f32,
    pub draw_x: f32,
    pub draw_y: f32,
    pub offset_x: f32,
    pub offset_y: f32,
    pub read_off_x: usize,
    pub read_off_y: usize,
    pub res_idx: i32,
    pub image_w: usize,
    pub image_h: usize,
    pub bit_depth: u8,
    pub _nr_color_channels: u8,
    pub viewport_width: f32,
    pub viewport_height: f32,
    pub full_image_w: usize,
    pub full_image_h: usize,
}

#[derive(Hash, Eq, PartialEq, Clone, Debug)]
pub struct TileKey {
    pub series: i32,
    pub level: i32,
    pub x: usize, // offset_x
    pub y: usize, // offset_y
    pub width: usize,
    pub height: usize,
    pub z_projection: ZProjection,
    pub z_range: Option<RangeInclusive<i32>>,
    pub t: i32,
}

pub struct TileWeightScale;

impl WeightScale<TileKey, Arc<CachedTile>> for TileWeightScale {
    fn weight(&self, _key: &TileKey, value: &Arc<CachedTile>) -> usize {
        value.size_in_bytes()
    }
}

pub struct CachedTile {
    pub data: Arc<Vec<ImageChannel>>,
}

impl CachedTile {
    pub fn size_in_bytes(&self) -> usize {
        let mut memory_size: usize = 0;
        for val in self.data.iter() {
            memory_size += val.image.get_image_memory_usage();
        }

        memory_size
    }
}

pub(crate) type RenderSource = Arc<Vec<ImageChannel>>;

pub struct ViewportCache {
    pub(crate) app_state: Arc<UiState>,
    pub(crate) cache_high_res: TileCache,
    pub(crate) cache_low_res: TileCache,
    pub active_high_res_data: RwLock<Option<(RenderSource, ReadContext)>>,
}

impl ViewportCache {
    pub fn new(app_state: Arc<UiState>) -> Self {
        let high_res_capacity = NonZeroUsize::new(1 * 1024 * 1024 * 1024).unwrap(); // 1 GB budget
        let low_res_capacity = NonZeroUsize::new(256 * 1024 * 1024).unwrap(); // 256 MB budget

        Self {
            app_state,
            cache_high_res: Mutex::new(CLruCache::with_config(
                clru::CLruCacheConfig::new(high_res_capacity).with_scale(TileWeightScale),
            )),
            cache_low_res: Mutex::new(CLruCache::with_config(
                clru::CLruCacheConfig::new(low_res_capacity).with_scale(TileWeightScale),
            )),
            active_high_res_data: RwLock::new(None),
        }
    }

    pub fn get_image_references(&self) -> (Vec<(i32, Arc<ImageContainer>)>, ReadContext) {
        let guard = self.active_high_res_data.read().expect("Poisoned");
        let Some((render_source, read_context)) = &*guard else {
            return (vec![], ReadContext::default());
        };
        (
            render_source
                .iter()
                .map(|ch| (ch.c_stack, ch.image.clone()))
                .collect(),
            read_context.clone(),
        )
    }

    /// Read an image tile either from disk or from the cache
    ///
    /// This function reads an image tile from the disk or the cache
    pub fn read_image_tile_combined(
        &self,
        series: i32,
        z_projection: ZProjection,
        z_range: Option<RangeInclusive<i32>>,
        t_stack: i32,
        //  c_stacks: &Vec<i32>,
        fit_to_screen: bool,
        is_new_image: bool,
        is_low_res: bool,
        view_port_state: &ViewportState,
    ) -> Result<(RenderSource, ReadContext), InternalErrors> {
        // Prepare image
        let (v_w, v_h, mut off_x, mut off_y, mut zoom, path) = {
            let p = self.app_state.get_project().get_current_image_path_cloned();
            let path = p
                .as_ref()
                .ok_or_else(|| InternalErrors::Internal("No path".into()))?
                .clone();
            (
                view_port_state.viewport_width,
                view_port_state.viewport_height,
                view_port_state.offset_x,
                view_port_state.offset_y,
                view_port_state.zoom,
                path,
            )
        };

        let reader = self.app_state.get_or_create_reader(&path)?;
        let meta = reader.get_image_meta();
        let s_info = meta
            .series
            .get(&series)
            .ok_or_else(|| InternalErrors::Internal("Series missing".into()))?;

        // Determine Pyramid Level & Scaling
        let highest_res = &s_info.resolutions[&0];

        let wanted_w: f32;
        let wanted_h: f32;

        if is_new_image {
            if is_low_res {
                let mut cache = self.cache_low_res.lock().unwrap();
                cache.clear();
            } else {
                let mut cache = self.cache_high_res.lock().unwrap();
                cache.clear();
            }
        }
        if fit_to_screen {
            // Viewport not yet laid out - bail so we never store zoom=0 in state.
            // update_viewport_size_in_viewport_state will re-trigger once dimensions arrive.
            if v_w <= 0.0 || v_h <= 0.0 {
                return Err(InternalErrors::Internal(
                    "Viewport not yet sized; deferring fit-to-screen render. Resize the window to update".into(),
                ));
            }

            // Fit to screen: Calculate the zoom (Window / Image)
            zoom = (v_w / highest_res.width as f32).min(v_h / highest_res.height as f32);

            // Add a 5% margin so the image doesn't touch the window edges
            zoom *= 0.95;

            // Calculate the "Wanted" size at this new zoom level
            wanted_w = highest_res.width as f32 * zoom;
            wanted_h = highest_res.height as f32 * zoom;

            // Center the image
            off_x = (v_w - wanted_w) / 2.0;
            off_y = (v_h - wanted_h) / 2.0;
        } else {
            // Recalculate wanted_w/h for the current zoom if not a new reader
            wanted_w = highest_res.width as f32 * zoom;
            wanted_h = highest_res.height as f32 * zoom;
        }

        let (res_idx, pyramid_w, pyramid_h) = look_for_best_matching_resolution_index(
            wanted_w as usize,
            wanted_h as usize,
            &s_info.resolutions,
            is_low_res,
        );

        // If this is the low res image, scale down to 1024x1024
        let (scale_x, scale_y, read_off_x, read_off_y, request_w, request_h, image_w, image_h) =
            match is_low_res {
                true => {
                    // Low resolution image
                    let ratio = pyramid_w / pyramid_h;
                    let image_w: f32 = LOW_RES_MAX_WIDTH_AND_HEIGHT as f32;
                    let image_h: f32 = LOW_RES_MAX_WIDTH_AND_HEIGHT as f32 * ratio;
                    let scale_x = wanted_w / image_w;
                    let scale_y = wanted_h / image_h;

                    if !scale_x.is_finite()
                        || scale_x <= 0.0
                        || !scale_y.is_finite()
                        || scale_y <= 0.0
                    {
                        return Err(InternalErrors::Internal(
                            "Degenerate low-res tile scale".into(),
                        ));
                    }

                    let max_tiles_x = (pyramid_w / TILE_SIZE_TO_LOAD).floor().max(0.0) as usize;
                    let max_tiles_y = (pyramid_h / TILE_SIZE_TO_LOAD).floor().max(0.0) as usize;

                    let read_off_x = ((-off_x / scale_x) / TILE_SIZE_TO_LOAD)
                        .floor()
                        .max(0.0)
                        .min(max_tiles_x as f32) as usize
                        * TILE_SIZE_TO_LOAD as usize;
                    let read_off_y = ((-off_y / scale_y) / TILE_SIZE_TO_LOAD)
                        .floor()
                        .max(0.0)
                        .min(max_tiles_y as f32) as usize
                        * TILE_SIZE_TO_LOAD as usize;

                    (
                        scale_x as f32,
                        scale_y as f32,
                        read_off_x as usize,
                        read_off_y as usize,
                        pyramid_w as usize,
                        pyramid_h as usize,
                        image_w as usize,
                        image_h as usize,
                    )
                }
                false => {
                    // High resolution image
                    let scale_x = wanted_w / pyramid_w;
                    let scale_y = wanted_h / pyramid_h;

                    if !scale_x.is_finite()
                        || scale_x <= 0.0
                        || !scale_y.is_finite()
                        || scale_y <= 0.0
                    {
                        return Err(InternalErrors::Internal(
                            "Degenerate high-res tile scale".into(),
                        ));
                    }

                    let max_tiles_x = (pyramid_w / TILE_SIZE_TO_LOAD).floor().max(0.0) as usize;
                    let max_tiles_y = (pyramid_h / TILE_SIZE_TO_LOAD).floor().max(0.0) as usize;

                    let read_off_x = ((-off_x / scale_x) / TILE_SIZE_TO_LOAD)
                        .floor()
                        .max(0.0)
                        .min(max_tiles_x as f32) as usize
                        * TILE_SIZE_TO_LOAD as usize;
                    let read_off_y = ((-off_y / scale_y) / TILE_SIZE_TO_LOAD)
                        .floor()
                        .max(0.0)
                        .min(max_tiles_y as f32) as usize
                        * TILE_SIZE_TO_LOAD as usize;

                    // Calculate Tile Dimensions
                    let request_w = (((v_w / scale_x).ceil() as usize)
                        + TILE_SIZE_TO_LOAD as usize)
                        .next_multiple_of(TILE_SIZE_TO_LOAD as usize)
                        .min((pyramid_w as usize).saturating_sub(read_off_x));

                    let request_h = (((v_h / scale_y).ceil() as usize)
                        + TILE_SIZE_TO_LOAD as usize)
                        .next_multiple_of(TILE_SIZE_TO_LOAD as usize)
                        .min((pyramid_h as usize).saturating_sub(read_off_y));
                    (
                        scale_x, scale_y, read_off_x, read_off_y, request_w, request_h, request_w,
                        request_h,
                    )
                }
            };

        let cache_key = TileKey {
            series,
            level: res_idx,
            x: read_off_x,
            y: read_off_y,
            width: image_w as usize,
            height: image_h as usize,
            z_projection: z_projection.clone(),
            z_range: z_range.clone(),
            t: t_stack,
        };

        let (draw_x, draw_y) =
            self.calc_draw_pos(cache_key.x, cache_key.y, scale_x, scale_y, off_x, off_y);

        let mut ctx = ReadContext {
            zoomed_w: cache_key.width as f32 * scale_x,
            zoomed_h: cache_key.height as f32 * scale_y,
            zoom: zoom,
            draw_x: draw_x,
            draw_y: draw_y,
            offset_x: off_x,
            offset_y: off_y,
            read_off_x: cache_key.x,
            read_off_y: cache_key.y,
            res_idx: res_idx,
            image_w: image_w,
            image_h: image_h,
            bit_depth: highest_res.nr_bits,
            _nr_color_channels: highest_res.color_channels,
            viewport_width: v_w,
            viewport_height: v_h,
            full_image_w: highest_res.width as usize,
            full_image_h: highest_res.height as usize,
        };

        // Read image from cache
        if let Some((found_key, cached_tile)) = self.find_in_cache(&cache_key, is_low_res) {
            let (draw_x, draw_y) =
                self.calc_draw_pos(found_key.x, found_key.y, scale_x, scale_y, off_x, off_y);

            ctx = ReadContext {
                zoomed_w: found_key.width as f32 * scale_x,
                zoomed_h: found_key.height as f32 * scale_y,
                zoom: zoom,
                draw_x: draw_x,
                draw_y: draw_y,
                offset_x: off_x,
                offset_y: off_y,
                read_off_x: found_key.x,
                read_off_y: found_key.y,
                res_idx: res_idx,
                image_w: found_key.width,
                image_h: found_key.height,
                bit_depth: highest_res.nr_bits,
                _nr_color_channels: highest_res.color_channels,
                viewport_width: v_w,
                viewport_height: v_h,
                full_image_w: highest_res.width as usize,
                full_image_h: highest_res.height as usize,
            };

            return Ok((cached_tile.data.clone(), ctx));
        }

        // Read image from disk
        let mut loaded = reader.read_image_tile_combined(
            series,
            ctx.res_idx,
            z_projection,
            &z_range,
            t_stack,
            None,
            &ImageTile {
                offset_x: ctx.read_off_x,
                offset_y: ctx.read_off_y,
                width: request_w as usize,
                height: request_h as usize,
            },
        )?;

        // Scale down if it is low res
        if is_low_res {
            loaded = scale_image(loaded, ctx.image_w, ctx.image_h)?;
        }

        // Put to cache and return
        let shared_data = Arc::new(loaded);
        let cached_tile = Arc::new(CachedTile {
            data: Arc::clone(&shared_data),
        });

        if is_low_res {
            let mut guard = self.cache_low_res.lock().unwrap();
            let ret = guard.put_with_weight(cache_key.clone(), cached_tile);
            if ret.is_err() {
                error!("Not enough space for low-res tile!");
            }
        } else {
            let mut guard = self.cache_high_res.lock().unwrap();
            let ret = guard.put_with_weight(cache_key.clone(), Arc::clone(&cached_tile));
            if ret.is_err() {
                error!("Not enough space!");
            }
        }

        Ok((Arc::clone(&shared_data), ctx))
    }

    /// Create a new image reader or use an existing if the image is the same
    ///
    /// This creates an image reader (Java wrapper).

    #[inline(always)]
    fn calc_draw_pos(
        &self,
        rx: usize,
        ry: usize,
        sx: f32,
        sy: f32,
        ox: f32,
        oy: f32,
    ) -> (f32, f32) {
        ((rx as f32).mul_add(sx, ox), (ry as f32).mul_add(sy, oy))
    }

    /// Is looking for an image tile in the cache
    ///
    /// Is looking for an image in the cache, first looking for a exact match
    /// if not found a spheral search is done to look for an image in the cache which
    /// overlaps with a still existing tile in the cache
    fn find_in_cache(
        &self,
        req: &TileKey,
        low_resolution: bool,
    ) -> Option<(TileKey, Arc<CachedTile>)> {
        if low_resolution {
            let mut cache = self.cache_low_res.lock().unwrap();

            if let Some(tile) = cache.get(req) {
                return Some((req.clone(), Arc::clone(tile)));
            }

            let mut found_key: Option<TileKey> = None;

            for (key, _) in cache.iter() {
                if key.level == req.level
                    && key.series == req.series
                    && key.t == req.t
                    && key.z_range == req.z_range
                    && key.z_projection == req.z_projection
                    && key.x <= req.x
                    && key.y <= req.y
                    && (key.x + key.width) >= (req.x + req.width)
                    && (key.y + key.height) >= (req.y + req.height)
                {
                    found_key = Some(key.clone());
                    break;
                }
            }

            // If found spatially, call .get() to refresh its priority in the LRU cache
            if let Some(key) = found_key {
                if let Some(tile) = cache.get(&key) {
                    return Some((key, Arc::clone(tile)));
                }
            }

            return None;
        } else {
            let mut cache = self.cache_high_res.lock().unwrap();

            if let Some(tile) = cache.get(req) {
                return Some((req.clone(), Arc::clone(tile)));
            }

            let mut found_key: Option<TileKey> = None;

            for (key, _) in cache.iter() {
                if key.level == req.level
                    && key.series == req.series
                    && key.t == req.t
                    && key.z_range == req.z_range
                    && key.z_projection == req.z_projection
                    && key.x <= req.x
                    && key.y <= req.y
                    && (key.x + key.width) >= (req.x + req.width)
                    && (key.y + key.height) >= (req.y + req.height)
                {
                    found_key = Some(key.clone());
                    break;
                }
            }

            // If found spatially, call .get() to refresh its priority in the LRU cache
            if let Some(key) = found_key {
                if let Some(tile) = cache.get(&key) {
                    return Some((key, Arc::clone(tile)));
                }
            }

            return None;
        }
    }
}

/// Looks for the best matching pyramid index
///
/// This method takes the wanted image size and width and returns the
/// index of the pyramid image which is nearest to the wanted width and height
fn look_for_best_matching_resolution_index(
    width: usize,
    height: usize,
    resolutions: &BTreeMap<i32, PyramidInfo>,
    low_resolution: bool,
) -> (i32, f32, f32) {
    let (target_w, target_h) = if low_resolution {
        (LOW_RES_MAX_WIDTH_AND_HEIGHT, LOW_RES_MAX_WIDTH_AND_HEIGHT)
    } else {
        (width as u64, height as u64)
    };

    for (level, res) in resolutions.iter().rev() {
        if res.width > target_w && res.height > target_h {
            return (*level, res.width as f32, res.height as f32);
        }
    }

    let max_res = resolutions.get(&0);
    match max_res {
        Some(res) => {
            return (0, res.width as f32, res.height as f32);
        }
        None => {
            return (0, 0.0, 0.0);
        }
    }
}

/// Scale images from the given vector
///
/// The images in the image channel are scaled to the given width and height
pub(crate) fn scale_image(
    channels: Vec<ImageChannel>,
    new_w: usize,
    new_h: usize,
) -> Result<Vec<ImageChannel>, InternalErrors> {
    let new_size = kornia_image::ImageSize {
        width: new_w as usize,
        height: new_h as usize,
    };

    channels
        .iter()
        .map(|channel| {
            let resized_container = match &*channel.image {
                ImageContainer::F32Gray(img) => {
                    let mut dst = Image::<f32, 1, CpuAllocator>::new(
                        new_size,
                        vec![0.0; new_size.width * new_size.height],
                        CpuAllocator,
                    )
                    .map_err(InternalErrors::from_kornia)?;

                    resize::resize_native(img, &mut dst, InterpolationMode::Nearest)
                        .map_err(InternalErrors::from_kornia)?;
                    Ok::<ImageContainer, InternalErrors>(ImageContainer::F32Gray(ManagedImage {
                        data: dst,
                        tile_offset: channel.image.tile_offset(),
                        plane: channel.image.plane(),
                    }))
                }
                ImageContainer::F32Rgb(img) => {
                    let mut dst = Image::<f32, 3, CpuAllocator>::new(
                        new_size,
                        vec![0.0; new_size.width * new_size.height * 3],
                        CpuAllocator,
                    )
                    .map_err(InternalErrors::from_kornia)?;

                    resize::resize_native(img, &mut dst, InterpolationMode::Nearest)
                        .map_err(InternalErrors::from_kornia)?;

                    Ok::<ImageContainer, InternalErrors>(ImageContainer::F32Rgb(ManagedImage {
                        data: dst,
                        tile_offset: channel.image.tile_offset(),
                        plane: channel.image.plane(),
                    }))
                }
                ImageContainer::U32(img) => {
                    Ok::<ImageContainer, InternalErrors>(ImageContainer::U32(img.clone()))
                }
            }?;

            Ok::<ImageChannel, InternalErrors>(ImageChannel {
                image: Arc::new(resized_container),
                color: channel.color,
                is_visible: channel.is_visible,
                c_stack: channel.c_stack,
                name: channel.name.clone(),
                is_rgb: channel.is_rgb,
            })
        })
        .collect::<Result<Vec<ImageChannel>, InternalErrors>>()
}

pub fn to_z_projection(z_handling: ZStackHandling) -> ZProjection {
    match z_handling {
        ZStackHandling::SingleStack => ZProjection::None,
        ZStackHandling::AllStacks => ZProjection::None,
        ZStackHandling::MaxIntensity => ZProjection::MaxIntensity,
        ZStackHandling::MinIntensity => ZProjection::MinIntensity,
        ZStackHandling::AvgIntensity => ZProjection::AvgIntensity,
        ZStackHandling::SumIntensity => ZProjection::SumIntensity,
        ZStackHandling::TakeTheMiddle => ZProjection::TakeTheMiddle,
    }
}
