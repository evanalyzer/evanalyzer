use crate::UiState;
use crate::editor::view_channel::{Samples, ViewChannel};
use crate::editor::viewport_controller::ViewportState;
use crate::helper::size_formater::BYTES_PER_MB;
use clru::{CLruCache, WeightScale};
use evanalyzer_app::global::ViewportMemorySettings;
use evanalyzer_app::images::ImageChannel;
use evanalyzer_app::images::ImageContainer;
use evanalyzer_app::images::ImageSource;
use evanalyzer_app::images::ManagedImage;
use evanalyzer_app::images::Point2d;
use evanalyzer_app::images::PyramidInfo;
use evanalyzer_app::images::TileRequest;
use evanalyzer_app::project::ProjectExt;
use evanalyzer_cfg::core_types::InternalErrors;
use evanalyzer_cfg::core_types::{ImageTile, ZProjection};
use evanalyzer_cfg::settings::images_settings::ZStackHandling;
use kornia_image::{Image, InterpolationMode};
use kornia_imgproc::resize;
use log::error;
use rayon::prelude::*;
use std::collections::{BTreeMap, HashMap};
use std::num::NonZeroUsize;
use std::ops::RangeInclusive;
use std::sync::{Arc, Mutex, RwLock};

type InnerCache =
    CLruCache<TileKey, Arc<CachedTile>, std::collections::hash_map::RandomState, TileWeightScale>;

const LOW_RES_MAX_WIDTH_AND_HEIGHT: u64 = 1024;

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
    pub bit_depth: u16,
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
    /// Every how many pixels of the region one was kept (1: all).
    pub subsample: usize,
    pub z_projection: ZProjection,
    pub z_range: Option<RangeInclusive<i32>>,
    pub t: i32,
}

impl TileKey {
    /// Size of the tile's image: the region, subsampled.
    fn output_size(&self) -> (usize, usize) {
        let step = self.subsample.max(1);
        (self.width.div_ceil(step), self.height.div_ceil(step))
    }
}

pub struct TileWeightScale;

impl WeightScale<TileKey, Arc<CachedTile>> for TileWeightScale {
    fn weight(&self, _key: &TileKey, value: &Arc<CachedTile>) -> usize {
        value.size_in_bytes()
    }
}

pub struct CachedTile {
    pub(crate) data: RenderSource,
}

impl CachedTile {
    pub fn size_in_bytes(&self) -> usize {
        self.data.iter().map(ViewChannel::size_in_bytes).sum()
    }
}

/// A tile's channels as the viewport keeps them - compact where exact (see
/// [`ViewChannel`]).
pub(crate) type RenderSource = Arc<Vec<ViewChannel>>;

/// Everything about a cached tile except its spatial position - the part of
/// a request that must match exactly, used to scope the spatial
/// "does a bigger cached tile already cover this" search to a small
/// candidate set instead of the whole cache.
#[derive(Hash, Eq, PartialEq, Clone, Debug)]
struct TileGroupKey {
    series: i32,
    level: i32,
    subsample: usize,
    t: i32,
    z_projection: ZProjection,
    z_range: Option<RangeInclusive<i32>>,
}

impl From<&TileKey> for TileGroupKey {
    fn from(key: &TileKey) -> Self {
        Self {
            series: key.series,
            level: key.level,
            subsample: key.subsample,
            t: key.t,
            z_projection: key.z_projection.clone(),
            z_range: key.z_range.clone(),
        }
    }
}

/// Groups every key currently in a [`InnerCache`] by [`TileGroupKey`], so a
/// spatial-superset search only scans tiles that could possibly match
/// instead of the entire cache.
///
/// Kept in sync on insert: a plain insert appends to its group; anything
/// that could have evicted another entry (the cache didn't grow by exactly
/// one) triggers a full rebuild from the cache's current contents - `clru`
/// has no eviction callback, so this is the only way to guarantee no stale
/// entry lingers forever. A stale entry between an insert and its resync is
/// harmless either way: [`IndexedTileCache::find_spatial_superset`] always
/// re-checks presence via `cache.get()` before trusting a candidate.
#[derive(Default)]
struct TileGroupIndex {
    by_group: HashMap<TileGroupKey, Vec<TileKey>>,
}

impl TileGroupIndex {
    fn record(&mut self, key: &TileKey) {
        self.by_group
            .entry(TileGroupKey::from(key))
            .or_default()
            .push(key.clone());
    }

    fn rebuild_from(&mut self, cache: &InnerCache) {
        self.by_group.clear();
        for (key, _) in cache.iter() {
            self.record(key);
        }
    }

    fn clear(&mut self) {
        self.by_group.clear();
    }

    fn candidates(&self, group: &TileGroupKey) -> &[TileKey] {
        self.by_group.get(group).map(Vec::as_slice).unwrap_or(&[])
    }
}

/// A weight-bounded LRU tile cache plus a [`TileGroupIndex`] kept in sync
/// with it.
struct IndexedTileCache {
    cache: InnerCache,
    index: TileGroupIndex,
}

impl IndexedTileCache {
    fn new(capacity: NonZeroUsize) -> Self {
        Self {
            cache: CLruCache::with_config(
                clru::CLruCacheConfig::new(capacity).with_scale(TileWeightScale),
            ),
            index: TileGroupIndex::default(),
        }
    }

    fn clear(&mut self) {
        self.cache.clear();
        self.index.clear();
    }

    /// Changes the weight budget, evicting least recently used tiles until
    /// the rest fits.
    fn resize(&mut self, capacity: NonZeroUsize) {
        self.cache.resize(capacity);
        self.index.rebuild_from(&self.cache);
    }

    /// Bytes currently held.
    fn weight(&self) -> usize {
        self.cache.weight()
    }

    fn get_exact(&mut self, key: &TileKey) -> Option<Arc<CachedTile>> {
        self.cache.get(key).cloned()
    }

    /// Finds a cached tile in `req`'s exact (series, level, t, z) group whose
    /// spatial bounds fully contain `req`, refreshing its LRU priority.
    fn find_spatial_superset(&mut self, req: &TileKey) -> Option<(TileKey, Arc<CachedTile>)> {
        let group = TileGroupKey::from(req);
        let found_key = self
            .index
            .candidates(&group)
            .iter()
            .find(|key| {
                key.x <= req.x
                    && key.y <= req.y
                    && (key.x + key.width) >= (req.x + req.width)
                    && (key.y + key.height) >= (req.y + req.height)
            })
            .cloned()?;
        // Also guards against a stale index entry: if it was already
        // evicted, this returns None and the caller falls back to disk.
        let tile = self.cache.get(&found_key).cloned()?;
        Some((found_key, tile))
    }

    fn insert(&mut self, key: TileKey, value: Arc<CachedTile>) -> Result<(), ()> {
        let len_before = self.cache.len();
        match self.cache.put_with_weight(key.clone(), value) {
            Ok(_) => {
                if self.cache.len() == len_before + 1 {
                    self.index.record(&key);
                } else {
                    // Either a replace (len unchanged) or an eviction made
                    // room for this insert - resync fully rather than trust
                    // partial bookkeeping.
                    self.index.rebuild_from(&self.cache);
                }
                Ok(())
            }
            Err(_) => Err(()),
        }
    }
}

pub struct ViewportCache {
    pub(crate) app_state: Arc<UiState>,
    cache_high_res: Mutex<IndexedTileCache>,
    cache_low_res: Mutex<IndexedTileCache>,
    pub active_high_res_data: RwLock<Option<(RenderSource, ReadContext)>>,
}

impl ViewportCache {
    /// A cache with the limits of this computer's viewport memory settings.
    pub fn new(app_state: Arc<UiState>) -> Self {
        let (tile_mb, overview_mb) = effective_cache_mb(&app_state.viewport_memory_settings());
        let high_res_capacity = mb_to_bytes(tile_mb);
        let low_res_capacity = mb_to_bytes(overview_mb);

        Self {
            app_state,
            cache_high_res: Mutex::new(IndexedTileCache::new(high_res_capacity)),
            cache_low_res: Mutex::new(IndexedTileCache::new(low_res_capacity)),
            active_high_res_data: RwLock::new(None),
        }
    }

    /// Changes the limits (in MB) of the tile and the overview cache; a
    /// lower limit evicts the least recently used entries right away.
    pub fn set_limits(&self, tile_cache_mb: u32, overview_cache_mb: u32) {
        self.cache_high_res
            .lock()
            .unwrap()
            .resize(mb_to_bytes(tile_cache_mb));
        self.cache_low_res
            .lock()
            .unwrap()
            .resize(mb_to_bytes(overview_cache_mb));
    }

    /// Empties both caches. What is on screen stays (it holds its own
    /// references); anything else is read from disk again when needed.
    pub fn clear(&self) {
        self.cache_high_res.lock().unwrap().clear();
        self.cache_low_res.lock().unwrap().clear();
    }

    /// Bytes held by the (tile, overview) cache.
    pub fn held_bytes(&self) -> (usize, usize) {
        (
            self.cache_high_res.lock().unwrap().weight(),
            self.cache_low_res.lock().unwrap().weight(),
        )
    }

    pub fn get_image_references(&self) -> (Vec<(i32, Arc<ImageContainer>)>, ReadContext) {
        let guard = self.active_high_res_data.read().expect("Poisoned");
        let Some((render_source, read_context)) = &*guard else {
            return (vec![], ReadContext::default());
        };
        (
            render_source
                .iter()
                .filter_map(|ch| Some((ch.c_stack, ch.to_container().ok()?)))
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

        let source = self.app_state.get_image_source(&path)?;
        let meta = source.meta();
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

        if !is_low_res {
            return self.read_sharp_view(
                &*source,
                GridParams {
                    series,
                    res_idx,
                    z_projection,
                    z_range,
                    t_stack,
                    pyramid_w,
                    pyramid_h,
                    wanted_w,
                    wanted_h,
                    zoom,
                    off_x,
                    off_y,
                    v_w,
                    v_h,
                    bit_depth: highest_res.nr_bits,
                    color_channels: highest_res.color_channels,
                    full_w: highest_res.width as usize,
                    full_h: highest_res.height as usize,
                },
            );
        }

        // `key_w`/`key_h`: the region read, in pixels of pyramid level
        // `res_idx` (the overview: its output size). `subsample`: every how
        // many of those pixels one is kept (see `read_subsampled`).
        let (
            scale_x,
            scale_y,
            read_off_x,
            read_off_y,
            request_w,
            request_h,
            key_w,
            key_h,
            subsample,
        ) = match is_low_res {
            true => {
                // The overview: the whole image, at most
                // LOW_RES_MAX_WIDTH_AND_HEIGHT wide, keeping its shape.
                let ratio = pyramid_w / pyramid_h;
                let image_w: f32 = LOW_RES_MAX_WIDTH_AND_HEIGHT as f32;
                let image_h: f32 = (LOW_RES_MAX_WIDTH_AND_HEIGHT as f32 / ratio).max(1.0);
                let scale_x = wanted_w / image_w;
                let scale_y = wanted_h / image_h;

                if !scale_x.is_finite() || scale_x <= 0.0 || !scale_y.is_finite() || scale_y <= 0.0
                {
                    return Err(InternalErrors::Internal(
                        "Degenerate low-res tile scale".into(),
                    ));
                }

                // The overview is always the whole image, so it starts
                // at the origin wherever the view is - the viewport
                // places it with the live zoom/offset.
                (
                    scale_x,
                    scale_y,
                    0,
                    0,
                    pyramid_w as usize,
                    pyramid_h as usize,
                    image_w as usize,
                    image_h as usize,
                    1,
                )
            }
            false => unreachable!("sharp views are read through the grid above"),
        };

        let cache_key = TileKey {
            series,
            level: res_idx,
            x: read_off_x,
            y: read_off_y,
            width: key_w,
            height: key_h,
            subsample,
            z_projection: z_projection.clone(),
            z_range: z_range.clone(),
            t: t_stack,
        };

        // The render context of a tile cached (or about to be) under `key`.
        let context_for = |key: &TileKey| {
            let (draw_x, draw_y) = self.calc_draw_pos(key.x, key.y, scale_x, scale_y, off_x, off_y);
            let (image_w, image_h) = key.output_size();
            ReadContext {
                // Each output pixel covers `subsample` pixels of the level.
                zoomed_w: (image_w * key.subsample) as f32 * scale_x,
                zoomed_h: (image_h * key.subsample) as f32 * scale_y,
                zoom,
                draw_x,
                draw_y,
                offset_x: off_x,
                offset_y: off_y,
                read_off_x: key.x,
                read_off_y: key.y,
                res_idx,
                image_w,
                image_h,
                bit_depth: highest_res.nr_bits,
                _nr_color_channels: highest_res.color_channels,
                viewport_width: v_w,
                viewport_height: v_h,
                full_image_w: highest_res.width as usize,
                full_image_h: highest_res.height as usize,
            }
        };

        // Read image from cache
        if let Some((found_key, cached_tile)) = self.find_in_cache(&cache_key, is_low_res) {
            return Ok((cached_tile.data.clone(), context_for(&found_key)));
        }
        let ctx = context_for(&cache_key);

        let request = TileRequest {
            series,
            resolution_idx: ctx.res_idx,
            z_projection,
            z_range: z_range.clone(),
            t_stack,
            tile: ImageTile {
                offset_x: ctx.read_off_x,
                offset_y: ctx.read_off_y,
                width: request_w,
                height: request_h,
            },
        };
        // The overview: subsampled down to about its size first (the only
        // level may be the full image), then scaled exactly.
        let step = (request_w / key_w).min(request_h / key_h).max(1);
        let loaded = read_subsampled(&*source, &request, step)?;
        let loaded = scale_image(loaded, key_w, key_h)?;

        // Keep the samples compact (exactly - see `ViewChannel`); the
        // reader's f32 copy is dropped right here.
        let shared_data: RenderSource = Arc::new(
            loaded
                .iter()
                .map(|channel| ViewChannel::from_image_channel(channel, highest_res.nr_bits))
                .collect(),
        );
        drop(loaded);

        // Put to cache and return
        let cached_tile = Arc::new(CachedTile {
            data: Arc::clone(&shared_data),
        });

        let mut guard = self.cache_low_res.lock().unwrap();
        if guard.insert(cache_key.clone(), cached_tile).is_err() {
            error!("Not enough space for low-res tile!");
        }

        Ok((Arc::clone(&shared_data), ctx))
    }

    /// Create a new image reader or use an existing if the image is the same
    ///
    /// This creates an image reader.

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

    /// The sharp view for the current viewport, assembled from grid cells
    /// of the pyramid level that fits the zoom (see [`read_grid_view`]).
    fn read_sharp_view(
        &self,
        source: &dyn ImageSource,
        p: GridParams,
    ) -> Result<(RenderSource, ReadContext), InternalErrors> {
        // Screen pixels per level pixel.
        let scale_x = p.wanted_w / p.pyramid_w;
        let scale_y = p.wanted_h / p.pyramid_h;
        if !scale_x.is_finite() || scale_x <= 0.0 || !scale_y.is_finite() || scale_y <= 0.0 {
            return Err(InternalErrors::Internal(
                "Degenerate high-res tile scale".into(),
            ));
        }
        // No stored level is near the wanted size (a coarse pyramid): keep
        // only every `subsample`-th pixel.
        let subsample = ((1.0 / scale_x.max(scale_y)).floor() as usize).max(1);
        let view = GridView {
            series: p.series,
            level: p.res_idx,
            z_projection: p.z_projection,
            z_range: p.z_range,
            t: p.t_stack,
            subsample,
            level_w: p.pyramid_w as usize,
            level_h: p.pyramid_h as usize,
            visible: [
                -p.off_x / scale_x,
                -p.off_y / scale_y,
                (p.v_w - p.off_x) / scale_x,
                (p.v_h - p.off_y) / scale_y,
            ],
        };
        let (data, region) = read_grid_view(&self.cache_high_res, source, &view, p.bit_depth)?;
        let ctx = ReadContext {
            // Each output pixel covers `subsample` level pixels.
            zoomed_w: (region.out_w * subsample) as f32 * scale_x,
            zoomed_h: (region.out_h * subsample) as f32 * scale_y,
            zoom: p.zoom,
            draw_x: (region.x as f32).mul_add(scale_x, p.off_x),
            draw_y: (region.y as f32).mul_add(scale_y, p.off_y),
            offset_x: p.off_x,
            offset_y: p.off_y,
            read_off_x: region.x,
            read_off_y: region.y,
            res_idx: p.res_idx,
            image_w: region.out_w,
            image_h: region.out_h,
            bit_depth: p.bit_depth,
            _nr_color_channels: p.color_channels,
            viewport_width: p.v_w,
            viewport_height: p.v_h,
            full_image_w: p.full_w,
            full_image_h: p.full_h,
        };
        Ok((data, ctx))
    }

    /// Is looking for an image tile in the cache
    ///
    /// Is looking for an image in the cache, first looking for a exact match
    /// if not found a spatial search is done to look for an image in the
    /// cache which overlaps with a still existing tile in the cache. The
    /// spatial search only scans tiles in `req`'s exact (series, level, t, z)
    /// group ([`TileGroupIndex`]) instead of the whole cache.
    fn find_in_cache(
        &self,
        req: &TileKey,
        low_resolution: bool,
    ) -> Option<(TileKey, Arc<CachedTile>)> {
        let mut cache = if low_resolution {
            self.cache_low_res.lock().unwrap()
        } else {
            self.cache_high_res.lock().unwrap()
        };

        if let Some(tile) = cache.get_exact(req) {
            return Some((req.clone(), tile));
        }

        cache.find_spatial_superset(req)
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
        // At least the wanted size - an exact match is the level itself.
        if res.width >= target_w && res.height >= target_h {
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

/// Size of the blocks [`read_subsampled`] reads at a time - a multiple of
/// the usual 256/512 px file tiles, so no file tile is decoded twice.
const SUBSAMPLE_BLOCK_W: usize = 2048;
const SUBSAMPLE_BLOCK_H: usize = 512;

/// Reads `request`'s region keeping only every `step`-th pixel in both
/// directions (nearest neighbour). With `step` > 1 the region is read block
/// by block and each block is thinned out right away, so RAM holds one
/// block plus the (`step`^2 times smaller) result - never the whole region
/// at full size, which for an image without a pyramid can be many GB.
fn read_subsampled(
    source: &dyn ImageSource,
    request: &TileRequest,
    step: usize,
) -> Result<Vec<ImageChannel>, InternalErrors> {
    if step <= 1 {
        return source.read_tile(request);
    }
    let region = request.tile;
    let out_w = region.width.div_ceil(step);
    let out_h = region.height.div_ceil(step);
    // Per channel: (samples per pixel, output samples), set by the first block.
    let mut outputs: Vec<(usize, Vec<f32>)> = Vec::new();
    let mut template: Vec<ImageChannel> = Vec::new();

    for by in (0..region.height).step_by(SUBSAMPLE_BLOCK_H) {
        let bh = SUBSAMPLE_BLOCK_H.min(region.height - by);
        // First row of this block on the sampling grid.
        let first_row = (step - by % step) % step;
        if first_row >= bh {
            continue;
        }
        for bx in (0..region.width).step_by(SUBSAMPLE_BLOCK_W) {
            let bw = SUBSAMPLE_BLOCK_W.min(region.width - bx);
            let first_col = (step - bx % step) % step;
            if first_col >= bw {
                continue;
            }
            let mut block_request = request.clone();
            block_request.tile = ImageTile {
                offset_x: region.offset_x + bx,
                offset_y: region.offset_y + by,
                width: bw,
                height: bh,
            };
            let block = source.read_tile(&block_request)?;
            if outputs.is_empty() {
                outputs = block
                    .iter()
                    .map(|c| {
                        let n = samples_per_pixel(&c.image);
                        (n, vec![0.0f32; out_w * out_h * n])
                    })
                    .collect();
                template = block.clone();
            }
            for (channel, (n, out)) in block.iter().zip(outputs.iter_mut()) {
                let Some(src) = f32_samples(&channel.image) else {
                    continue;
                };
                let n = *n;
                for ry in (first_row..bh).step_by(step) {
                    let oy = (by + ry) / step;
                    for rx in (first_col..bw).step_by(step) {
                        let ox = (bx + rx) / step;
                        let s = (ry * bw + rx) * n;
                        let d = (oy * out_w + ox) * n;
                        out[d..d + n].copy_from_slice(&src[s..s + n]);
                    }
                }
            }
        }
    }

    let size = kornia_image::ImageSize {
        width: out_w,
        height: out_h,
    };
    template
        .into_iter()
        .zip(outputs)
        .map(|(channel, (n, data))| {
            let tile_offset = Point2d {
                x: region.offset_x,
                y: region.offset_y,
            };
            let plane = channel.image.plane();
            let image = if n == 3 {
                ImageContainer::F32Rgb(ManagedImage {
                    data: Image::<f32, 3>::new(size, data).map_err(InternalErrors::from_kornia)?,
                    tile_offset,
                    plane,
                })
            } else {
                ImageContainer::F32Gray(ManagedImage {
                    data: Image::<f32, 1>::new(size, data).map_err(InternalErrors::from_kornia)?,
                    tile_offset,
                    plane,
                })
            };
            Ok(ImageChannel {
                image: Arc::new(image),
                ..channel
            })
        })
        .collect()
}

fn samples_per_pixel(image: &ImageContainer) -> usize {
    match image {
        ImageContainer::F32Rgb(_) => 3,
        _ => 1,
    }
}

fn f32_samples(image: &ImageContainer) -> Option<&[f32]> {
    match image {
        ImageContainer::F32Gray(img) => Some(img.as_slice()),
        ImageContainer::F32Rgb(img) => Some(img.as_slice()),
        ImageContainer::U32(_) => None,
    }
}

/// What [`ViewportCache::read_sharp_view`] needs from the read request.
struct GridParams {
    series: i32,
    res_idx: i32,
    z_projection: ZProjection,
    z_range: Option<RangeInclusive<i32>>,
    t_stack: i32,
    pyramid_w: f32,
    pyramid_h: f32,
    wanted_w: f32,
    wanted_h: f32,
    zoom: f32,
    off_x: f32,
    off_y: f32,
    v_w: f32,
    v_h: f32,
    bit_depth: u16,
    color_channels: u8,
    full_w: usize,
    full_h: usize,
}

/// Edge length, in output pixels, of the grid cells the sharp view is read
/// and cached in. A multiple of the usual 256/512 px file tiles.
const GRID_CELL: usize = 512;

/// How many runs of missing cells [`read_grid_view`] reads at once.
const PARALLEL_RUNS: usize = 4;

/// What a sharp view of one plane at one pyramid level needs, in that
/// level's pixels.
struct GridView {
    series: i32,
    level: i32,
    z_projection: ZProjection,
    z_range: Option<RangeInclusive<i32>>,
    t: i32,
    /// Every how many level pixels one is kept (coarse pyramids).
    subsample: usize,
    level_w: usize,
    level_h: usize,
    /// The visible part of the level: x0, y0, x1, y1 (may reach outside).
    visible: [f32; 4],
}

/// Where an assembled view lies: its top-left cell corner and size, in
/// level pixels / output pixels.
#[derive(Debug, PartialEq)]
struct GridRegion {
    x: usize,
    y: usize,
    out_w: usize,
    out_h: usize,
}

impl GridView {
    /// Cell edge in level pixels.
    fn cell(&self) -> usize {
        GRID_CELL * self.subsample.max(1)
    }

    /// The cell columns/rows covering the visible part, clamped to the
    /// image (at least one cell). No margin around it: measured, a ring of
    /// cells around the view (sharp during a pan of up to one cell) about
    /// doubled the time and RAM per view - panning shows the overview in
    /// the uncovered strip until the re-render instead.
    fn cell_range(&self) -> (std::ops::Range<usize>, std::ops::Range<usize>) {
        let cell = self.cell() as f32;
        let (nx, ny) = (
            self.level_w.div_ceil(self.cell()),
            self.level_h.div_ceil(self.cell()),
        );
        let range = |a: f32, b: f32, n: usize| {
            let first = ((a / cell).floor()).clamp(0.0, (n - 1) as f32) as usize;
            let end = ((b / cell).ceil()).clamp(1.0, n as f32) as usize;
            first..end.max(first + 1)
        };
        let [x0, y0, x1, y1] = self.visible;
        (range(x0, x1, nx), range(y0, y1, ny))
    }

    fn key(&self, cx: usize, cy: usize) -> TileKey {
        let cell = self.cell();
        let (x, y) = (cx * cell, cy * cell);
        TileKey {
            series: self.series,
            level: self.level,
            x,
            y,
            width: cell.min(self.level_w - x),
            height: cell.min(self.level_h - y),
            subsample: self.subsample.max(1),
            z_projection: self.z_projection.clone(),
            z_range: self.z_range.clone(),
            t: self.t,
        }
    }
}

/// The sharp view `view` asks for, assembled from grid cells: cached cells
/// are reused, missing ones read - a row's run of neighbouring missing cells
/// in one read - compacted and cached one by one. So a pan reads only the
/// cells it uncovers, and every pixel is cached once however the views
/// overlap.
fn read_grid_view(
    cache: &Mutex<IndexedTileCache>,
    source: &dyn ImageSource,
    view: &GridView,
    bits: u16,
) -> Result<(RenderSource, GridRegion), InternalErrors> {
    let (cols, rows) = view.cell_range();
    let mut cells: Vec<Vec<Option<Arc<CachedTile>>>> = {
        let mut cache = cache.lock().unwrap();
        rows.clone()
            .map(|cy| {
                cols.clone()
                    .map(|cx| cache.get_exact(&view.key(cx, cy)))
                    .collect()
            })
            .collect()
    };

    // Read the missing ones, a run of neighbours per read - up to
    // `PARALLEL_RUNS` runs at once, so the reader keeps more cores busy
    // while the f32 copies in flight stay bounded.
    let step = view.subsample.max(1);
    let mut runs = Vec::new();
    for (ry, row) in cells.iter().enumerate() {
        let mut rx = 0;
        while rx < row.len() {
            if row[rx].is_some() {
                rx += 1;
                continue;
            }
            let start = rx;
            while rx < row.len() && row[rx].is_none() {
                rx += 1;
            }
            runs.push((ry, start..rx));
        }
    }
    for batch in runs.chunks(PARALLEL_RUNS) {
        let read: Vec<Vec<(usize, usize, TileKey, Arc<CachedTile>)>> = batch
            .par_iter()
            .map(|(ry, run)| {
                let cy = rows.start + ry;
                let first = view.key(cols.start + run.start, cy);
                let last = view.key(cols.start + run.end - 1, cy);
                let request = TileRequest {
                    series: view.series,
                    resolution_idx: view.level,
                    z_projection: view.z_projection.clone(),
                    z_range: view.z_range.clone(),
                    t_stack: view.t,
                    tile: ImageTile {
                        offset_x: first.x,
                        offset_y: first.y,
                        width: last.x + last.width - first.x,
                        height: first.height,
                    },
                };
                let channels = read_subsampled(source, &request, step)?;
                run.clone()
                    .map(|i| {
                        let key = view.key(cols.start + i, cy);
                        let col0 = (key.x - first.x) / step;
                        let (out_w, _) = key.output_size();
                        let data = channels
                            .iter()
                            .map(|c| {
                                Ok(ViewChannel::from_image_channel(
                                    &crop_columns(c, col0, out_w)?,
                                    bits,
                                ))
                            })
                            .collect::<Result<Vec<_>, InternalErrors>>()?;
                        let tile = Arc::new(CachedTile {
                            data: Arc::new(data),
                        });
                        Ok((*ry, i, key, tile))
                    })
                    .collect()
            })
            .collect::<Result<_, InternalErrors>>()?;
        let mut cache = cache.lock().unwrap();
        for (ry, i, key, tile) in read.into_iter().flatten() {
            if cache.insert(key, tile.clone()).is_err() {
                error!("Not enough space for a viewport tile!");
            }
            cells[ry][i] = Some(tile);
        }
    }

    // Assemble them into one view.
    let cells: Vec<Vec<Arc<CachedTile>>> = cells
        .into_iter()
        .map(|row| {
            row.into_iter()
                .map(|c| c.expect("every cell read"))
                .collect()
        })
        .collect();
    let first = view.key(cols.start, rows.start);
    let last = view.key(cols.end - 1, rows.end - 1);
    let region = GridRegion {
        x: first.x,
        y: first.y,
        out_w: (last.x + last.width - first.x).div_ceil(step),
        out_h: (last.y + last.height - first.y).div_ceil(step),
    };
    let channels = cells[0][0].data.len();
    let assembled = (0..channels)
        .into_par_iter()
        .map(|c| assemble_channel(&cells, c, GRID_CELL, region.out_w, region.out_h))
        .collect::<Result<Vec<_>, InternalErrors>>()?;
    Ok((Arc::new(assembled), region))
}

/// Columns `first..first + width` of `channel` (gray or interleaved RGB).
fn crop_columns(
    channel: &ImageChannel,
    first: usize,
    width: usize,
) -> Result<ImageChannel, InternalErrors> {
    let n = samples_per_pixel(&channel.image);
    let src = f32_samples(&channel.image)
        .ok_or_else(|| InternalErrors::ImageReadError("label image in a viewport tile".into()))?;
    let size = channel.image.size();
    let mut data = Vec::with_capacity(width * size.height * n);
    for row in src.chunks_exact(size.width * n) {
        data.extend_from_slice(&row[first * n..(first + width) * n]);
    }
    let new_size = kornia_image::ImageSize {
        width,
        height: size.height,
    };
    let offset = channel.image.tile_offset();
    let tile_offset = Point2d {
        x: offset.x + first,
        y: offset.y,
    };
    let plane = channel.image.plane();
    let image = if n == 3 {
        ImageContainer::F32Rgb(ManagedImage {
            data: Image::<f32, 3>::new(new_size, data).map_err(InternalErrors::from_kornia)?,
            tile_offset,
            plane,
        })
    } else {
        ImageContainer::F32Gray(ManagedImage {
            data: Image::<f32, 1>::new(new_size, data).map_err(InternalErrors::from_kornia)?,
            tile_offset,
            plane,
        })
    };
    Ok(ImageChannel {
        image: Arc::new(image),
        color: channel.color,
        is_visible: channel.is_visible,
        c_stack: channel.c_stack,
        name: channel.name.clone(),
        is_rgb: channel.is_rgb,
    })
}

/// Channel `c` of the `cells` grid (rows of cells, each `cell` output
/// pixels wide/high except at the image edge) as one `out_w` x `out_h`
/// channel - in the cells' compact type if they all share one.
fn assemble_channel(
    cells: &[Vec<Arc<CachedTile>>],
    c: usize,
    cell: usize,
    out_w: usize,
    out_h: usize,
) -> Result<ViewChannel, InternalErrors> {
    let first = &cells[0][0].data[c];
    let all = || cells.iter().flatten().map(|t| &t.data[c]);
    // Copies each cell's rows into `out` (one value per pixel).
    fn copy<T: Copy + Default + Send + Sync>(
        cells: &[Vec<Arc<CachedTile>>],
        c: usize,
        cell: usize,
        out_w: usize,
        out_h: usize,
        get: impl Fn(&ViewChannel) -> Option<&[T]> + Sync,
    ) -> Option<Vec<T>> {
        let mut out = vec![T::default(); out_w * out_h];
        // One band of cell rows per task.
        out.par_chunks_mut(cell * out_w)
            .zip(cells.par_iter())
            .try_for_each(|(band, row)| {
                for (gx, tile) in row.iter().enumerate() {
                    let ch = &tile.data[c];
                    let src = get(ch)?;
                    for y in 0..ch.height {
                        let d = y * out_w + gx * cell;
                        band[d..d + ch.width]
                            .copy_from_slice(&src[y * ch.width..(y + 1) * ch.width]);
                    }
                }
                Some(())
            })?;
        Some(out)
    }
    let samples = match &first.samples {
        Samples::U8 { inv, .. } if all().all(|ch| matches!(ch.samples, Samples::U8 { .. })) => {
            copy(cells, c, cell, out_w, out_h, |ch| match &ch.samples {
                Samples::U8 { data, .. } => Some(&data[..]),
                _ => None,
            })
            .map(|d| Samples::U8 {
                data: d.into(),
                inv: *inv,
            })
        }
        Samples::U16 { inv, .. } if all().all(|ch| matches!(ch.samples, Samples::U16 { .. })) => {
            copy(cells, c, cell, out_w, out_h, |ch| match &ch.samples {
                Samples::U16 { data, .. } => Some(&data[..]),
                _ => None,
            })
            .map(|d| Samples::U16 {
                data: d.into(),
                inv: *inv,
            })
        }
        _ => None,
    };
    let samples = match samples {
        Some(s) => s,
        // Mixed or float cells: as f32.
        None => {
            let mut out = vec![0.0f32; out_w * out_h];
            for (gy, row) in cells.iter().enumerate() {
                for (gx, tile) in row.iter().enumerate() {
                    let ch = &tile.data[c];
                    let src = ch.slice().ok_or_else(|| {
                        InternalErrors::ImageReadError("unexpected viewport tile layout".into())
                    })?;
                    for y in 0..ch.height {
                        for x in 0..ch.width {
                            out[(gy * cell + y) * out_w + gx * cell + x] =
                                src.get(y * ch.width + x);
                        }
                    }
                }
            }
            let image = Image::<f32, 1>::new(
                kornia_image::ImageSize {
                    width: out_w,
                    height: out_h,
                },
                out,
            )
            .map_err(InternalErrors::from_kornia)?;
            Samples::Image(Arc::new(ImageContainer::F32Gray(ManagedImage {
                data: image,
                tile_offset: first.tile_offset,
                plane: None,
            })))
        }
    };
    Ok(ViewChannel {
        samples,
        width: out_w,
        height: out_h,
        tile_offset: first.tile_offset,
        color: first.color,
        c_stack: first.c_stack,
        name: first.name.clone(),
        is_rgb: first.is_rgb,
    })
}

/// This computer's total RAM in bytes - read once, it doesn't change.
pub(crate) fn this_computer_ram_bytes() -> u64 {
    static RAM: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *RAM.get_or_init(|| evanalyzer_app::backends::local::cpu_ram_diagnostics().1)
}

/// The cache sizes (tile, overview) in MB `settings` give on this computer.
pub(crate) fn effective_cache_mb(settings: &ViewportMemorySettings) -> (u32, u32) {
    settings.effective_cache_mb(this_computer_ram_bytes())
}

/// A cache limit in MB as the byte budget `clru` takes (at least 1 MB).
fn mb_to_bytes(mb: u32) -> NonZeroUsize {
    NonZeroUsize::new(mb.max(1) as usize * BYTES_PER_MB as usize).expect("at least 1 MB")
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
                    let mut dst =
                        Image::<f32, 1>::new(new_size, vec![0.0; new_size.width * new_size.height])
                            .map_err(InternalErrors::from_kornia)?;

                    resize::resize(img, &mut dst, InterpolationMode::Nearest)
                        .map_err(InternalErrors::from_kornia)?;
                    Ok::<ImageContainer, InternalErrors>(ImageContainer::F32Gray(ManagedImage {
                        data: dst,
                        tile_offset: channel.image.tile_offset(),
                        plane: channel.image.plane(),
                    }))
                }
                ImageContainer::F32Rgb(img) => {
                    let mut dst = Image::<f32, 3>::new(
                        new_size,
                        vec![0.0; new_size.width * new_size.height * 3],
                    )
                    .map_err(InternalErrors::from_kornia)?;

                    resize::resize(img, &mut dst, InterpolationMode::Nearest)
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

#[cfg(test)]
mod tests {
    use super::*;
    use evanalyzer_app::images::Point2d;
    use evanalyzer_cfg::core_types::ImagePlane;
    use kornia_image::ImageSize;

    fn key(series: i32, level: i32, t: i32, x: usize, y: usize, w: usize, h: usize) -> TileKey {
        TileKey {
            series,
            level,
            x,
            y,
            width: w,
            height: h,
            subsample: 1,
            z_projection: ZProjection::None,
            z_range: None,
            t,
        }
    }

    /// A tile with a real, size-proportional weight (w*h*4 bytes for a single
    /// F32Gray channel), so capacity/eviction behaves like production.
    fn weighted_tile(w: usize, h: usize) -> Arc<CachedTile> {
        let image = Image::<f32, 1>::new(
            ImageSize {
                width: w,
                height: h,
            },
            vec![0.0f32; w * h],
        )
        .unwrap();
        let container = ImageContainer::F32Gray(ManagedImage {
            data: image,
            tile_offset: Point2d { x: 0, y: 0 },
            plane: Some(ImagePlane { z: 0, c: 0, t: 0 }),
        });
        Arc::new(CachedTile {
            data: Arc::new(vec![ViewChannel::wrap(&ImageChannel {
                image: Arc::new(container),
                color: [1.0, 1.0, 1.0],
                is_visible: true,
                c_stack: 0,
                name: "ch0".to_string(),
                is_rgb: false,
            })]),
        })
    }

    #[test]
    fn exact_match_is_found_without_a_spatial_search() {
        let mut cache = IndexedTileCache::new(NonZeroUsize::new(1_000_000).unwrap());
        let k = key(0, 0, 0, 0, 0, 10, 10);
        cache.insert(k.clone(), weighted_tile(10, 10)).unwrap();

        assert!(cache.get_exact(&k).is_some());
    }

    #[test]
    fn spatial_superset_is_found_within_the_same_group() {
        let mut cache = IndexedTileCache::new(NonZeroUsize::new(1_000_000).unwrap());
        let big = key(0, 0, 0, 0, 0, 100, 100);
        cache.insert(big.clone(), weighted_tile(100, 100)).unwrap();

        // A smaller request fully inside `big`, same group, no exact key match.
        let small_req = key(0, 0, 0, 10, 10, 20, 20);
        assert!(cache.get_exact(&small_req).is_none());
        let (found_key, _) = cache
            .find_spatial_superset(&small_req)
            .expect("must find the containing tile");
        assert_eq!(found_key, big);
    }

    #[test]
    fn spatial_search_never_matches_a_different_group() {
        let mut cache = IndexedTileCache::new(NonZeroUsize::new(1_000_000).unwrap());
        // Same spatial footprint, but a different `t` - stands in for any of
        // (series, level, t, z_projection, z_range) differing.
        let other_group = key(0, 0, /* t = */ 1, 0, 0, 100, 100);
        cache.insert(other_group, weighted_tile(100, 100)).unwrap();

        let req = key(0, 0, /* t = */ 0, 10, 10, 20, 20);
        assert!(cache.find_spatial_superset(&req).is_none());
    }

    #[test]
    fn candidates_are_restricted_to_the_requested_group_even_with_many_groups_cached() {
        let mut cache = IndexedTileCache::new(NonZeroUsize::new(10_000_000).unwrap());
        // Populate 20 different `t` groups with one tile each, plus the group
        // we'll actually query - this is exactly the scenario the linear
        // scan used to pay for on every miss.
        for t in 0..20 {
            cache
                .insert(key(0, 0, t, 0, 0, 50, 50), weighted_tile(50, 50))
                .unwrap();
        }
        let target_group = key(0, 0, 999, 0, 0, 50, 50);
        cache
            .insert(target_group.clone(), weighted_tile(50, 50))
            .unwrap();

        let group = TileGroupKey::from(&target_group);
        let candidates = cache.index.candidates(&group);
        assert_eq!(
            candidates.len(),
            1,
            "the group index must not return tiles from other (series, level, t, z) groups"
        );
        assert_eq!(candidates[0], target_group);
    }

    #[test]
    fn evicted_tiles_are_never_returned_as_spatial_matches() {
        // Small enough capacity that inserting a second tile evicts the first.
        let one_tile_weight = 50 * 50 * 4; // F32Gray, w*h*4 bytes
        let mut cache = IndexedTileCache::new(NonZeroUsize::new(one_tile_weight + 10).unwrap());

        let first = key(0, 0, 0, 0, 0, 50, 50);
        cache.insert(first.clone(), weighted_tile(50, 50)).unwrap();
        assert!(cache.get_exact(&first).is_some());

        // Second tile in the same group forces eviction of `first` (LRU, over budget).
        let second = key(0, 0, 0, 1000, 1000, 50, 50);
        cache.insert(second, weighted_tile(50, 50)).unwrap();
        assert!(
            cache.get_exact(&first).is_none(),
            "sanity check: first must actually have been evicted"
        );

        // A request `first` would have spatially covered must not resolve to
        // it now that it's gone.
        let req_within_first = key(0, 0, 0, 10, 10, 5, 5);
        assert!(
            cache.find_spatial_superset(&req_within_first).is_none(),
            "an evicted tile must never be returned, even transiently through the index"
        );
    }

    #[test]
    fn clear_empties_both_the_cache_and_the_index() {
        let mut cache = IndexedTileCache::new(NonZeroUsize::new(1_000_000).unwrap());
        let k = key(0, 0, 0, 0, 0, 10, 10);
        cache.insert(k.clone(), weighted_tile(10, 10)).unwrap();
        assert!(cache.get_exact(&k).is_some());

        cache.clear();

        assert!(cache.get_exact(&k).is_none());
        let group = TileGroupKey::from(&k);
        assert!(cache.index.candidates(&group).is_empty());
    }

    #[test]
    fn a_lower_limit_evicts_the_least_recently_used_tiles_and_keeps_the_index_in_sync() {
        // 100x100 F32 = 40_000 bytes per tile (clru adds 1 per entry);
        // room for three.
        let mut cache = IndexedTileCache::new(NonZeroUsize::new(130_000).unwrap());
        let keys: Vec<_> = (0..3).map(|i| key(0, 0, 0, i * 100, 0, 100, 100)).collect();
        for k in &keys {
            cache.insert(k.clone(), weighted_tile(100, 100)).unwrap();
        }
        cache.get_exact(&keys[0]); // now the most recently used

        // Room for two.
        cache.resize(NonZeroUsize::new(90_000).unwrap());

        assert_eq!(cache.weight(), 80_000);
        assert!(cache.get_exact(&keys[0]).is_some(), "recently used stays");
        assert!(cache.get_exact(&keys[1]).is_none(), "oldest goes");
        let group = TileGroupKey::from(&keys[1]);
        assert!(!cache.index.candidates(&group).contains(&keys[1]));
    }

    /// A virtual gray image whose value at (x, y) is known, recording the
    /// largest block read from it.
    #[derive(Default)]
    struct PatternSource {
        meta: evanalyzer_app::images::ImageMeta,
        largest_read: std::sync::Mutex<usize>,
        pixels_read: std::sync::Mutex<usize>,
    }

    /// Normalized like the reader does for 8-bit data (`raw * (1 / 255)`).
    fn pattern_value(x: usize, y: usize) -> f32 {
        ((x * 31 + y * 17) % 256) as f32 * (1.0 / 255.0)
    }

    impl ImageSource for PatternSource {
        fn meta(&self) -> &evanalyzer_app::images::ImageMeta {
            &self.meta
        }

        fn read_tile(&self, req: &TileRequest) -> Result<Vec<ImageChannel>, InternalErrors> {
            let t = req.tile;
            let mut largest = self.largest_read.lock().unwrap();
            *largest = (*largest).max(t.width * t.height);
            *self.pixels_read.lock().unwrap() += t.width * t.height;
            let data = (0..t.height)
                .flat_map(|y| {
                    (0..t.width).map(move |x| pattern_value(t.offset_x + x, t.offset_y + y))
                })
                .collect();
            let image = Image::<f32, 1>::new(
                ImageSize {
                    width: t.width,
                    height: t.height,
                },
                data,
            )
            .unwrap();
            Ok(vec![ImageChannel {
                image: Arc::new(ImageContainer::F32Gray(ManagedImage {
                    data: image,
                    tile_offset: Point2d {
                        x: t.offset_x,
                        y: t.offset_y,
                    },
                    plane: None,
                })),
                color: [1.0, 1.0, 1.0],
                is_visible: true,
                c_stack: 0,
                name: "ch0".into(),
                is_rgb: false,
            }])
        }
    }

    fn pattern_request(x: usize, y: usize, w: usize, h: usize) -> TileRequest {
        TileRequest {
            series: 0,
            resolution_idx: 0,
            z_projection: ZProjection::None,
            z_range: None,
            t_stack: 0,
            tile: ImageTile {
                offset_x: x,
                offset_y: y,
                width: w,
                height: h,
            },
        }
    }

    #[test]
    fn a_subsampled_read_keeps_every_nth_pixel_and_reads_only_small_blocks() {
        let source = PatternSource::default();
        let step = 3;
        let (x0, y0, w, h) = (1024, 512, 5000, 1300);

        let channels = read_subsampled(&source, &pattern_request(x0, y0, w, h), step).unwrap();

        let ImageContainer::F32Gray(img) = &*channels[0].image else {
            panic!("gray image expected");
        };
        let (out_w, out_h) = (w.div_ceil(step), h.div_ceil(step));
        assert_eq!((img.width(), img.height()), (out_w, out_h));
        let px = img.as_slice();
        for (ox, oy) in [
            (0, 0),
            (1, 0),
            (682, 170),
            (out_w - 1, out_h - 1),
            (700, 171),
        ] {
            assert_eq!(
                px[oy * out_w + ox],
                pattern_value(x0 + ox * step, y0 + oy * step),
                "output ({ox}, {oy})"
            );
        }
        assert!(
            *source.largest_read.lock().unwrap() <= SUBSAMPLE_BLOCK_W * SUBSAMPLE_BLOCK_H,
            "never the whole region at once"
        );
    }

    #[test]
    fn without_subsampling_the_region_is_read_in_one_piece() {
        let source = PatternSource::default();
        let channels = read_subsampled(&source, &pattern_request(0, 0, 3000, 700), 1).unwrap();
        assert_eq!(channels[0].image.size().width, 3000);
        assert_eq!(*source.largest_read.lock().unwrap(), 3000 * 700);
    }

    #[test]
    fn a_subsampled_tile_is_sized_and_grouped_by_its_step() {
        let mut k = key(0, 0, 0, 0, 0, 5000, 1300);
        k.subsample = 3;
        assert_eq!(k.output_size(), (1667, 434));
        assert_ne!(
            TileGroupKey::from(&k),
            TileGroupKey::from(&key(0, 0, 0, 0, 0, 5000, 1300))
        );
    }

    /// A view of a 10,000 x 8,000 level showing level pixels
    /// `x0..x1` x `y0..y1`.
    fn grid_view(subsample: usize, visible: [f32; 4]) -> GridView {
        GridView {
            series: 0,
            level: 0,
            z_projection: ZProjection::None,
            z_range: None,
            t: 0,
            subsample,
            level_w: 10_000,
            level_h: 8_000,
            visible,
        }
    }

    fn view_values(data: &RenderSource) -> (usize, Vec<f32>) {
        let ch = &data[0];
        let s = ch.slice().unwrap();
        (ch.width, (0..s.len()).map(|i| s.get(i)).collect())
    }

    #[test]
    fn a_sharp_view_is_assembled_from_the_grid_cells_it_shows() {
        let cache = Mutex::new(IndexedTileCache::new(NonZeroUsize::new(1 << 30).unwrap()));
        let source = PatternSource::default();
        // Visible 1100..2900 x 700..1800: cells 2..=5 x 1..=3.
        let view = grid_view(1, [1100.0, 700.0, 2900.0, 1800.0]);

        let (data, region) = read_grid_view(&cache, &source, &view, 8).unwrap();

        assert_eq!(
            region,
            GridRegion {
                x: 1024,
                y: 512,
                out_w: 4 * 512,
                out_h: 3 * 512,
            }
        );
        let (w, values) = view_values(&data);
        assert_eq!(w, region.out_w);
        for (ox, oy) in [(0, 0), (511, 0), (512, 1), (1777, 1300), (2047, 1535)] {
            let expected = pattern_value(region.x + ox, region.y + oy);
            assert_eq!(values[oy * w + ox], expected, "({ox}, {oy})");
        }
        assert!(
            matches!(data[0].samples, Samples::U8 { .. }),
            "8-bit stays compact"
        );
    }

    #[test]
    fn panning_reads_only_the_cells_it_uncovers_and_caches_each_once() {
        let cache = Mutex::new(IndexedTileCache::new(NonZeroUsize::new(1 << 30).unwrap()));
        let source = PatternSource::default();
        read_grid_view(
            &cache,
            &source,
            &grid_view(1, [1100.0, 700.0, 2900.0, 1800.0]),
            8,
        )
        .unwrap();
        let first_read = *source.pixels_read.lock().unwrap();
        let first_weight = cache.lock().unwrap().weight();
        assert_eq!(first_read, 4 * 3 * 512 * 512);

        // One cell to the right: one new column of 3 cells.
        read_grid_view(
            &cache,
            &source,
            &grid_view(1, [1612.0, 700.0, 3412.0, 1800.0]),
            8,
        )
        .unwrap();
        assert_eq!(
            *source.pixels_read.lock().unwrap() - first_read,
            3 * 512 * 512
        );
        // Each pixel cached once: one column (1 byte per pixel) more.
        assert_eq!(cache.lock().unwrap().weight() - first_weight, 3 * 512 * 512);

        // Back again: nothing read.
        let before = *source.pixels_read.lock().unwrap();
        read_grid_view(
            &cache,
            &source,
            &grid_view(1, [1100.0, 700.0, 2900.0, 1800.0]),
            8,
        )
        .unwrap();
        assert_eq!(*source.pixels_read.lock().unwrap(), before);
    }

    #[test]
    fn cells_at_the_image_edge_are_clipped() {
        let cache = Mutex::new(IndexedTileCache::new(NonZeroUsize::new(1 << 30).unwrap()));
        let source = PatternSource::default();
        // The bottom-right corner, partly outside the image.
        let (data, region) = read_grid_view(
            &cache,
            &source,
            &grid_view(1, [9000.0, 7000.0, 11_000.0, 9000.0]),
            8,
        )
        .unwrap();
        // Cells from the one holding 9000/7000 to the edge.
        assert_eq!((region.x, region.y), (8704, 6656));
        assert_eq!((region.out_w, region.out_h), (10_000 - 8704, 8_000 - 6656));
        let (w, values) = view_values(&data);
        assert_eq!(values.last(), Some(&pattern_value(9_999, 7_999)));
        assert_eq!(values[w - 1], pattern_value(9_999, 6656));
    }

    #[test]
    fn a_coarse_pyramid_level_is_subsampled_cell_by_cell() {
        let cache = Mutex::new(IndexedTileCache::new(NonZeroUsize::new(1 << 30).unwrap()));
        let source = PatternSource::default();
        let (data, region) = read_grid_view(
            &cache,
            &source,
            &grid_view(2, [0.0, 0.0, 3000.0, 2000.0]),
            8,
        )
        .unwrap();
        // Cells are 1024 level px (512 output px); 0..3 x 0..2.
        assert_eq!(
            (region.x, region.y, region.out_w, region.out_h),
            (0, 0, 1536, 1024)
        );
        let (w, values) = view_values(&data);
        for (ox, oy) in [(0, 0), (513, 7), (1535, 1023)] {
            assert_eq!(
                values[oy * w + ox],
                pattern_value(ox * 2, oy * 2),
                "({ox}, {oy})"
            );
        }
    }

    #[test]
    fn a_cache_limit_of_zero_mb_still_is_a_valid_budget() {
        assert_eq!(mb_to_bytes(0).get(), 1024 * 1024);
        assert_eq!(mb_to_bytes(1024).get(), 1024 * 1024 * 1024);
    }

    // -- to_z_projection ---------------------------------------------------------

    #[test]
    fn to_z_projection_maps_every_z_stack_handling_variant() {
        let cases = [
            (ZStackHandling::SingleStack, ZProjection::None),
            (ZStackHandling::AllStacks, ZProjection::None),
            (ZStackHandling::MaxIntensity, ZProjection::MaxIntensity),
            (ZStackHandling::MinIntensity, ZProjection::MinIntensity),
            (ZStackHandling::AvgIntensity, ZProjection::AvgIntensity),
            (ZStackHandling::SumIntensity, ZProjection::SumIntensity),
            (ZStackHandling::TakeTheMiddle, ZProjection::TakeTheMiddle),
        ];
        for (handling, expected) in cases {
            assert_eq!(
                to_z_projection(handling.clone()),
                expected,
                "for {handling:?}"
            );
        }
    }

    // -- look_for_best_matching_resolution_index ---------------------------------

    fn pyramid(width: u64, height: u64) -> PyramidInfo {
        PyramidInfo {
            width,
            height,
            ..Default::default()
        }
    }

    #[test]
    fn best_matching_resolution_full_res_picks_the_smallest_level_that_still_covers_the_viewport() {
        let mut resolutions = BTreeMap::new();
        resolutions.insert(0, pyramid(4000, 3000));
        resolutions.insert(1, pyramid(2000, 1500));
        resolutions.insert(2, pyramid(500, 375)); // too small to cover 800x600

        let (level, w, h) = look_for_best_matching_resolution_index(800, 600, &resolutions, false);

        assert_eq!(level, 1);
        assert_eq!((w, h), (2000.0, 1500.0));
    }

    #[test]
    fn best_matching_resolution_falls_back_to_level_0_when_every_level_is_too_small() {
        let mut resolutions = BTreeMap::new();
        resolutions.insert(0, pyramid(200, 150));

        let (level, w, h) = look_for_best_matching_resolution_index(800, 600, &resolutions, false);

        assert_eq!(level, 0);
        assert_eq!((w, h), (200.0, 150.0));
    }

    #[test]
    fn best_matching_resolution_of_an_empty_pyramid_returns_a_zero_sized_default() {
        let resolutions = BTreeMap::new();

        let (level, w, h) = look_for_best_matching_resolution_index(800, 600, &resolutions, false);

        assert_eq!((level, w, h), (0, 0.0, 0.0));
    }

    #[test]
    fn best_matching_resolution_low_res_mode_targets_the_fixed_low_res_cap_not_the_viewport_size() {
        let mut resolutions = BTreeMap::new();
        // Bigger than the viewport request (10000x10000) but still within
        // the fixed low-res cap - low_resolution=true must target the cap
        // (LOW_RES_MAX_WIDTH_AND_HEIGHT), not the (huge) requested size.
        resolutions.insert(0, pyramid(2000, 2000));

        let (level, w, h) =
            look_for_best_matching_resolution_index(10_000, 10_000, &resolutions, true);

        assert_eq!(level, 0);
        assert_eq!((w, h), (2000.0, 2000.0));
    }

    // -- calc_draw_pos / find_in_cache --------------------------------------------

    #[test]
    fn calc_draw_pos_scales_and_offsets_the_tile_position() {
        let cache = ViewportCache::new(crate::editor::test_support::test_ui_state());

        let (x, y) = cache.calc_draw_pos(10, 20, 2.0, 3.0, 1.0, 1.0);

        assert_eq!((x, y), (21.0, 61.0));
    }

    #[test]
    fn find_in_cache_finds_an_exact_match_after_inserting_into_the_high_res_cache() {
        let cache = ViewportCache::new(crate::editor::test_support::test_ui_state());
        let k = key(0, 0, 0, 0, 0, 10, 10);
        cache
            .cache_high_res
            .lock()
            .unwrap()
            .insert(k.clone(), weighted_tile(10, 10))
            .unwrap();

        let found = cache.find_in_cache(&k, false);

        assert!(found.is_some());
    }

    #[test]
    fn find_in_cache_low_resolution_flag_searches_the_low_res_cache_not_the_high_res_one() {
        let cache = ViewportCache::new(crate::editor::test_support::test_ui_state());
        let k = key(0, 0, 0, 0, 0, 10, 10);
        cache
            .cache_high_res
            .lock()
            .unwrap()
            .insert(k.clone(), weighted_tile(10, 10))
            .unwrap();

        // Inserted into high-res only - a low-res lookup must not find it.
        assert!(cache.find_in_cache(&k, true).is_none());
        assert!(cache.find_in_cache(&k, false).is_some());
    }
}
