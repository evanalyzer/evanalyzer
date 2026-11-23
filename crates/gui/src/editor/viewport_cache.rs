//! # viewport_cache
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-06-05
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.

use crate::UiState;
use crate::editor::view_channel::{Samples, ViewChannel};
use crate::editor::viewport_controller::ViewportState;
use crate::helper::size_formater::BYTES_PER_MB;
use clru::{CLruCache, WeightScale};
use evanalyzer_app::images::ImageChannel;
use evanalyzer_app::images::ImageContainer;
use evanalyzer_app::images::ImageSource;
use evanalyzer_app::images::ManagedImage;
use evanalyzer_app::images::Point2d;
use evanalyzer_app::images::PyramidInfo;
use evanalyzer_app::images::TileRequest;
use evanalyzer_app::user::ViewportMemorySettings;
use evanalyzer_cfg::core_types::ImageDimensionality;
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
use std::path::PathBuf;
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
    /// The image file the tile is of: a tile of another image never
    /// matches, even before the caches are emptied for it (the overview
    /// and the sharp view are read by two workers, each emptying its cache
    /// when it gets to the new image).
    pub image: Arc<PathBuf>,
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
    image: Arc<PathBuf>,
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
            image: Arc::clone(&key.image),
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
    /// The view last assembled from this cache's grid cells (see
    /// [`read_grid_view`]).
    last_view: Option<AssembledView>,
    /// Grid cells being read right now (see [`read_grid_cells`]): another
    /// reader needing one waits for it on `ready` instead of reading it a
    /// second time.
    in_flight: std::collections::HashSet<TileKey>,
    ready: Arc<std::sync::Condvar>,
}

/// A sharp view assembled from grid cells, and which cells it was made of.
/// The same cells again (a contrast drag, a channel toggle: only the look
/// changes) give the same view - it is returned instead of copying every
/// cell again. The cells are held weakly: their pixels are freed when the
/// cache evicts them, only their small allocation stays, so an address is
/// never reused for another cell while it is compared.
struct AssembledView {
    cells: Vec<std::sync::Weak<CachedTile>>,
    region: GridRegion,
    data: RenderSource,
}

impl AssembledView {
    /// The view if `cells` (rows of cells, as read) and `region` are what
    /// it was assembled from.
    fn reuse(&self, cells: &[Vec<Arc<CachedTile>>], region: &GridRegion) -> Option<RenderSource> {
        let same = self.region == *region
            && self.cells.len() == cells.iter().map(Vec::len).sum::<usize>()
            && self
                .cells
                .iter()
                .zip(cells.iter().flatten())
                .all(|(kept, cell)| std::ptr::eq(kept.as_ptr(), Arc::as_ptr(cell)));
        same.then(|| Arc::clone(&self.data))
    }
}

impl IndexedTileCache {
    fn new(capacity: NonZeroUsize) -> Self {
        Self {
            cache: CLruCache::with_config(
                clru::CLruCacheConfig::new(capacity).with_scale(TileWeightScale),
            ),
            index: TileGroupIndex::default(),
            last_view: None,
            in_flight: Default::default(),
            ready: Arc::new(std::sync::Condvar::new()),
        }
    }

    fn clear(&mut self) {
        self.cache.clear();
        self.index.clear();
        self.last_view = None;
        // A read still under way finds its cells' keys gone (harmless); a
        // reader waiting for one reads it itself.
        self.in_flight.clear();
        self.ready.notify_all();
    }

    /// Only `image`'s tiles kept (another image opens): the others freed.
    /// Its own were read for this opening - by the other worker, too (the
    /// caches are emptied when the opening is dispatched, see
    /// [`ViewportCache::clear_tiles`]).
    fn retain_image(&mut self, image: &std::path::Path) {
        self.cache.retain(|key, _| key.image.as_path() == image);
        self.index.rebuild_from(&self.cache);
        self.last_view = None;
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
    /// The results' split view: opening this image (path) shows this box
    /// (image pixels) instead of the whole image - see
    /// [`crate::editor::results_overlay::focus_view`].
    pub focus: Mutex<Option<(PathBuf, [f32; 4])>>,
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
            focus: Mutex::new(None),
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

    /// The (tile, overview) cache's limits in bytes.
    #[cfg(test)]
    pub fn capacity_bytes(&self) -> (usize, usize) {
        (
            self.cache_high_res.lock().unwrap().cache.capacity(),
            self.cache_low_res.lock().unwrap().cache.capacity(),
        )
    }

    /// Nothing shown any more (another project): every cached tile and
    /// overview, and the sharp view's data, are freed.
    pub fn clear(&self) {
        self.clear_tiles();
        if let Ok(mut active) = self.active_high_res_data.write() {
            *active = None;
        }
    }

    /// Every cached tile and overview freed - an image opens: it is read
    /// anew (also when it is the same file again).
    pub fn clear_tiles(&self) {
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

    /// The sharp view's channels expanded to `f32`, and where it lies (the
    /// benchmarks print them; elsewhere [`Self::read_context`] does).
    #[cfg(test)]
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

    /// Where the sharp view on screen lies - [`Self::get_image_references`]'s
    /// context alone, without expanding the pixels (a stroke's end only
    /// needs the view's position).
    pub fn read_context(&self) -> ReadContext {
        let guard = self.active_high_res_data.read().expect("Poisoned");
        guard
            .as_ref()
            .map(|(_, context)| context.clone())
            .unwrap_or_default()
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

        // Another image: the last one's tiles go. This one's stay - the
        // other worker may have read them for this opening already (both
        // caches were emptied when it was dispatched).
        if is_new_image {
            if is_low_res {
                self.cache_low_res.lock().unwrap().retain_image(&path);
            } else {
                self.cache_high_res.lock().unwrap().retain_image(&path);
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

            let focus = self
                .focus
                .lock()
                .unwrap()
                .as_ref()
                .filter(|(focus_path, _)| *focus_path == path)
                .map(|(_, bbox)| *bbox);
            if let Some(bbox) = focus {
                (zoom, off_x, off_y) = crate::editor::results_overlay::focus_view(
                    v_w,
                    v_h,
                    highest_res.width as f32,
                    highest_res.height as f32,
                    bbox,
                );
                wanted_w = highest_res.width as f32 * zoom;
                wanted_h = highest_res.height as f32 * zoom;
            } else {
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
            }
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

        let image = Arc::new(path.clone());
        if !is_low_res {
            return self.read_sharp_view(
                &*source,
                GridParams {
                    image,
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
            image: Arc::clone(&image),
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
        // Of an image without a coarser level, the sharp view (fitted)
        // reads the same level: its cells give the same samples, read once
        // for both workers (else the file was decoded twice).
        let shared = SharedOverview {
            image: Arc::clone(&image),
            series,
            z_projection: request.z_projection,
            z_range: z_range.clone(),
            t: t_stack,
            step,
            bits: highest_res.nr_bits,
        };
        let from_cells = (res_idx == 0)
            .then(|| {
                sharp_subsample_shared_with_overview(
                    &s_info.resolutions,
                    wanted_w,
                    wanted_h,
                    v_w,
                    v_h,
                    step,
                )
            })
            .flatten()
            .and_then(|subsample| {
                // Not one value per pixel (interleaved, labels) or a read
                // error: read as before.
                self.overview_from_sharp_cells(&*source, &shared, subsample)
                    .inspect_err(|e| log::debug!("Overview from the sharp cells: {e:?}"))
                    .ok()
            });
        let loaded = match from_cells {
            Some(loaded) => loaded,
            None => read_subsampled(&*source, &request, step)?,
        };
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

    /// The overview's samples (`shared.step` x `shared.step` subsampled,
    /// as [`read_subsampled`] gives them) taken from the sharp view's grid
    /// cells of level 0 at `subsample` (a divisor of the step) - cached ones
    /// reused, the rest read (and cached for the sharp view).
    fn overview_from_sharp_cells(
        &self,
        source: &dyn ImageSource,
        shared: &SharedOverview,
        subsample: usize,
    ) -> Result<Vec<ImageChannel>, InternalErrors> {
        let level = source
            .meta()
            .series
            .get(&shared.series)
            .and_then(|s| s.resolutions.get(&0))
            .ok_or_else(|| InternalErrors::Internal("Level 0 missing".into()))?;
        let (full_w, full_h) = (level.width as usize, level.height as usize);
        let view = GridView {
            image: Arc::clone(&shared.image),
            series: shared.series,
            level: 0,
            z_projection: shared.z_projection,
            z_range: shared.z_range.clone(),
            t: shared.t,
            subsample,
            level_w: full_w,
            level_h: full_h,
            visible: [0.0, 0.0, full_w as f32, full_h as f32],
        };
        let (cells, region) = read_grid_cells(&self.cache_high_res, source, &view, shared.bits)?;
        overview_samples(
            &cells,
            &region,
            shared.step / subsample,
            shared.step,
            full_w,
            full_h,
        )
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
            image: p.image,
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

/// What the overview of one plane needs to be taken from the sharp view's
/// cells (see [`ViewportCache::overview_from_sharp_cells`]).
struct SharedOverview {
    image: Arc<PathBuf>,
    series: i32,
    z_projection: ZProjection,
    z_range: Option<RangeInclusive<i32>>,
    t: i32,
    /// The overview's subsampling of level 0.
    step: usize,
    bits: u16,
}

/// The subsample the sharp view reads level 0 with for a view of
/// `wanted_w` x `wanted_h` screen pixels (computed exactly as
/// [`ViewportCache::read_sharp_view`] does) - if the overview, which reads
/// level 0 subsampled by `step`, can take its samples from those cells:
/// the sharp view reads level 0 too, at the subsample the fitted view of a
/// `v_w` x `v_h` viewport uses (zoomed in further, its cells of the whole
/// image would be far more than the overview needs), and that divides
/// `step`. `None`: the overview is read on its own, as before.
fn sharp_subsample_shared_with_overview(
    resolutions: &BTreeMap<i32, PyramidInfo>,
    wanted_w: f32,
    wanted_h: f32,
    v_w: f32,
    v_h: f32,
    step: usize,
) -> Option<usize> {
    let full = resolutions.get(&0)?;
    let subsample_for = |wanted_w: f32, wanted_h: f32| {
        let (level, level_w, level_h) = look_for_best_matching_resolution_index(
            wanted_w as usize,
            wanted_h as usize,
            resolutions,
            false,
        );
        let (scale_x, scale_y) = (wanted_w / level_w, wanted_h / level_h);
        let valid = scale_x.is_finite() && scale_x > 0.0 && scale_y.is_finite() && scale_y > 0.0;
        (level == 0 && valid).then(|| ((1.0 / scale_x.max(scale_y)).floor() as usize).max(1))
    };
    let now = subsample_for(wanted_w, wanted_h)?;
    // The fitted view's, as `read_image_tile_combined` fits.
    let mut zoom = (v_w / full.width as f32).min(v_h / full.height as f32);
    zoom *= 0.95;
    let fitted = subsample_for(full.width as f32 * zoom, full.height as f32 * zoom)?;
    (now == fitted && step.is_multiple_of(now)).then_some(now)
}

/// The overview's samples from `cells` (the grid of all of level 0 at a
/// subsample, `region` where they lie): every `ratio`-th of them - level-0
/// pixel `(x * step, y * step)` at `(x, y)`, as [`read_subsampled`] keeps
/// them, for a `full_w` x `full_h` level.
fn overview_samples(
    cells: &[Vec<Arc<CachedTile>>],
    region: &GridRegion,
    ratio: usize,
    step: usize,
    full_w: usize,
    full_h: usize,
) -> Result<Vec<ImageChannel>, InternalErrors> {
    if region.x != 0 || region.y != 0 || ratio == 0 {
        return Err(InternalErrors::Internal(
            "Overview grid not of the whole image".into(),
        ));
    }
    let (out_w, out_h) = (full_w.div_ceil(step), full_h.div_ceil(step));
    let size = kornia_image::ImageSize {
        width: out_w,
        height: out_h,
    };
    (0..cells[0][0].data.len())
        .map(|c| {
            let first = &cells[0][0].data[c];
            let mut out = vec![0.0f32; out_w * out_h];
            out.par_chunks_mut(out_w)
                .enumerate()
                .try_for_each(|(oy, row)| {
                    let gy = oy * ratio;
                    let (cy, ly) = (gy / GRID_CELL, gy % GRID_CELL);
                    for (ox, value) in row.iter_mut().enumerate() {
                        let gx = ox * ratio;
                        let ch = &cells.get(cy)?.get(gx / GRID_CELL)?.data[c];
                        let samples = ch.slice()?;
                        let lx = gx % GRID_CELL;
                        (lx < ch.width && ly < ch.height).then_some(())?;
                        let i = ly * ch.width + lx;
                        (i < samples.len()).then_some(())?;
                        *value = samples.get(i);
                    }
                    Some(())
                })
                .ok_or_else(|| {
                    InternalErrors::ImageReadError(
                        "overview cell without one value per pixel".into(),
                    )
                })?;
            Ok(ImageChannel {
                image: Arc::new(ImageContainer::F32Gray(ManagedImage {
                    data: Image::<f32, 1>::new(size, out).map_err(InternalErrors::from_kornia)?,
                    tile_offset: Point2d { x: 0, y: 0 },
                    plane: None,
                    dimensionality: ImageDimensionality::Planar,
                })),
                color: first.color,
                is_visible: true,
                c_stack: first.c_stack,
                name: first.name.clone(),
                is_rgb: first.is_rgb,
            })
        })
        .collect()
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
pub(crate) fn read_subsampled(
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
                    dimensionality: ImageDimensionality::Planar,
                })
            } else {
                ImageContainer::F32Gray(ManagedImage {
                    data: Image::<f32, 1>::new(size, data).map_err(InternalErrors::from_kornia)?,
                    tile_offset,
                    plane,
                    dimensionality: ImageDimensionality::Planar,
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
    image: Arc<PathBuf>,
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
    /// The image file (see [`TileKey::image`]).
    image: Arc<PathBuf>,
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
#[derive(Debug, PartialEq, Clone, Copy)]
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
            image: Arc::clone(&self.image),
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

/// The sharp view `view` asks for, assembled from grid cells (see
/// [`read_grid_cells`]).
fn read_grid_view(
    cache: &Mutex<IndexedTileCache>,
    source: &dyn ImageSource,
    view: &GridView,
    bits: u16,
) -> Result<(RenderSource, GridRegion), InternalErrors> {
    let (cells, region) = read_grid_cells(cache, source, view, bits)?;
    // The same cells as the last view (only the look changed): that view.
    if let Some(data) = cache
        .lock()
        .unwrap()
        .last_view
        .as_ref()
        .and_then(|view| view.reuse(&cells, &region))
    {
        return Ok((data, region));
    }
    let assembled: RenderSource = match single_compact_cell(&cells) {
        // One cell, already compact: the view is that cell, uncopied.
        Some(data) => data,
        None => {
            let channels = cells[0][0].data.len();
            Arc::new(
                (0..channels)
                    .into_par_iter()
                    .map(|c| assemble_channel(&cells, c, GRID_CELL, region.out_w, region.out_h))
                    .collect::<Result<Vec<_>, InternalErrors>>()?,
            )
        }
    };
    cache.lock().unwrap().last_view = Some(AssembledView {
        cells: cells.iter().flatten().map(Arc::downgrade).collect(),
        region,
        data: Arc::clone(&assembled),
    });
    Ok((assembled, region))
}

/// Rows of grid cells.
type CellGrid = Vec<Vec<Arc<CachedTile>>>;

/// Rows of grid cells, those not read yet `None`.
type PartialCellGrid = [Vec<Option<Arc<CachedTile>>>];

/// Cells this reader claimed for reading (in the cache's `in_flight`):
/// released when read - or when the read fails or panics (on drop), so a
/// reader waiting for one never waits forever.
struct Claims<'a> {
    cache: &'a Mutex<IndexedTileCache>,
    keys: Vec<TileKey>,
}

impl Claims<'_> {
    /// `keys` read (or given up): no longer claimed, the waiters woken.
    fn release(&mut self, keys: &[TileKey]) {
        let mut cache = self.cache.lock().unwrap_or_else(|e| e.into_inner());
        for key in keys {
            cache.in_flight.remove(key);
        }
        self.keys.retain(|k| !keys.contains(k));
        cache.ready.notify_all();
    }
}

impl Drop for Claims<'_> {
    fn drop(&mut self) {
        if !self.keys.is_empty() {
            let keys = std::mem::take(&mut self.keys);
            self.release(&keys);
        }
    }
}

/// The grid cells `view` shows (rows of cells) and where they lie: cached
/// cells are reused, missing ones read - a row's run of neighbouring missing
/// cells in one read - compacted and cached one by one. So a pan reads only
/// the cells it uncovers, and every pixel is cached once however the views
/// overlap. A cell another reader is reading right now (the other worker:
/// the overview and the sharp view of an image without a pyramid share
/// cells) is waited for, not read a second time.
fn read_grid_cells(
    cache: &Mutex<IndexedTileCache>,
    source: &dyn ImageSource,
    view: &GridView,
    bits: u16,
) -> Result<(CellGrid, GridRegion), InternalErrors> {
    let (cols, rows) = view.cell_range();
    let mut claims = Claims {
        cache,
        keys: Vec::new(),
    };
    // Cached; else claimed here (read below); else read by another reader
    // right now (`theirs`, waited for after).
    let mut theirs: Vec<(usize, usize)> = Vec::new();
    let mut cells: Vec<Vec<Option<Arc<CachedTile>>>> = {
        let mut cache = cache.lock().unwrap();
        rows.clone()
            .enumerate()
            .map(|(ry, cy)| {
                cols.clone()
                    .enumerate()
                    .map(|(rx, cx)| {
                        let key = view.key(cx, cy);
                        let cached = cache.get_exact(&key);
                        if cached.is_none() {
                            if cache.in_flight.contains(&key) {
                                theirs.push((ry, rx));
                            } else {
                                cache.in_flight.insert(key.clone());
                                claims.keys.push(key);
                            }
                        }
                        cached
                    })
                    .collect()
            })
            .collect()
    };
    let is_theirs = |ry: usize, rx: usize| theirs.contains(&(ry, rx));

    // Read the claimed ones, a run of neighbours per read - up to
    // `PARALLEL_RUNS` runs at once, so the reader keeps more cores busy
    // while the f32 copies in flight stay bounded.
    let step = view.subsample.max(1);
    let mut runs = Vec::new();
    for (ry, row) in cells.iter().enumerate() {
        let mut rx = 0;
        while rx < row.len() {
            if row[rx].is_some() || is_theirs(ry, rx) {
                rx += 1;
                continue;
            }
            let start = rx;
            while rx < row.len() && row[rx].is_none() && !is_theirs(ry, rx) {
                rx += 1;
            }
            runs.push((ry, start..rx));
        }
    }
    read_runs(
        cache,
        source,
        view,
        bits,
        &rows,
        &cols,
        &runs,
        &mut cells,
        &mut claims,
    )?;

    // The cells another reader was reading: once it is done, from the cache
    // - or read here if it didn't keep them (its read failed, no room).
    for (ry, rx) in theirs {
        let key = view.key(cols.start + rx, rows.start + ry);
        let mut guard = cache.lock().unwrap();
        loop {
            if let Some(tile) = guard.get_exact(&key) {
                cells[ry][rx] = Some(tile);
                break;
            }
            if !guard.in_flight.contains(&key) {
                guard.in_flight.insert(key.clone());
                claims.keys.push(key.clone());
                drop(guard);
                read_runs(
                    cache,
                    source,
                    view,
                    bits,
                    &rows,
                    &cols,
                    &[(ry, rx..rx + 1)],
                    &mut cells,
                    &mut claims,
                )?;
                break;
            }
            let ready = Arc::clone(&guard.ready);
            guard = ready.wait(guard).unwrap();
        }
    }

    let cells: CellGrid = cells
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
    Ok((cells, region))
}

/// Reads `runs` (`(row, cells)` of `view`'s cell range `rows` x `cols`),
/// caches each cell, puts it into `cells` and releases its claim.
#[allow(clippy::too_many_arguments)]
fn read_runs(
    cache: &Mutex<IndexedTileCache>,
    source: &dyn ImageSource,
    view: &GridView,
    bits: u16,
    rows: &std::ops::Range<usize>,
    cols: &std::ops::Range<usize>,
    runs: &[(usize, std::ops::Range<usize>)],
    cells: &mut PartialCellGrid,
    claims: &mut Claims<'_>,
) -> Result<(), InternalErrors> {
    let step = view.subsample.max(1);
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
        let mut read_keys = Vec::new();
        {
            let mut cache = cache.lock().unwrap();
            for (ry, i, key, tile) in read.into_iter().flatten() {
                if cache.insert(key.clone(), tile.clone()).is_err() {
                    error!("Not enough space for a viewport tile!");
                }
                cells[ry][i] = Some(tile);
                read_keys.push(key);
            }
        }
        claims.release(&read_keys);
    }
    Ok(())
}

/// The one cell of a one-cell view, if all its channels are compact
/// (`u8`/`u16`) - what [`assemble_channel`] would copy it into, value for
/// value. Other data goes through `assemble_channel` (which turns float
/// cells into one `f32` image and refuses interleaved ones).
fn single_compact_cell(cells: &[Vec<Arc<CachedTile>>]) -> Option<RenderSource> {
    let [row] = cells else {
        return None;
    };
    let [cell] = row.as_slice() else {
        return None;
    };
    cell.data
        .iter()
        .all(|ch| matches!(ch.samples, Samples::U8 { .. } | Samples::U16 { .. }))
        .then(|| Arc::clone(&cell.data))
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
            dimensionality: ImageDimensionality::Planar,
        })
    } else {
        ImageContainer::F32Gray(ManagedImage {
            data: Image::<f32, 1>::new(new_size, data).map_err(InternalErrors::from_kornia)?,
            tile_offset,
            plane,
            dimensionality: ImageDimensionality::Planar,
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

/// Sample types whose all-zero bytes are a valid value (`0`) - so a buffer
/// of them can be allocated zeroed (lazily, by the system) instead of
/// written once before it is filled.
trait ZeroIsValid: Copy + Send + Sync {}
impl ZeroIsValid for u8 {}
impl ZeroIsValid for u16 {}

/// A shared buffer of `n` zeros.
fn zeroed_shared<T: ZeroIsValid>(n: usize) -> Arc<[T]> {
    let zeroed = Arc::<[T]>::new_zeroed_slice(n);
    // SAFETY: all-zero bytes are a valid `T` (see `ZeroIsValid`: `u8` and
    // `u16` only).
    unsafe { zeroed.assume_init() }
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
    // Copies each cell's rows into `out` (one value per pixel) - made in
    // place in the shared buffer the view keeps (not a `Vec` copied into
    // one afterwards).
    fn copy<T: ZeroIsValid>(
        cells: &[Vec<Arc<CachedTile>>],
        c: usize,
        cell: usize,
        out_w: usize,
        out_h: usize,
        get: impl Fn(&ViewChannel) -> Option<&[T]> + Sync,
    ) -> Option<Arc<[T]>> {
        let mut out = zeroed_shared::<T>(out_w * out_h);
        let slice = Arc::get_mut(&mut out).expect("a new buffer is not shared");
        // One band of cell rows per task.
        slice
            .par_chunks_mut(cell * out_w)
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
            .map(|data| Samples::U8 { data, inv: *inv })
        }
        Samples::U16 { inv, .. } if all().all(|ch| matches!(ch.samples, Samples::U16 { .. })) => {
            copy(cells, c, cell, out_w, out_h, |ch| match &ch.samples {
                Samples::U16 { data, .. } => Some(&data[..]),
                _ => None,
            })
            .map(|data| Samples::U16 { data, inv: *inv })
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
                dimensionality: ImageDimensionality::Planar,
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
    *RAM.get_or_init(|| evanalyzer_app::backend::local::cpu_ram_diagnostics().1)
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
                        dimensionality: channel.image.dimensionality(),
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
                        dimensionality: channel.image.dimensionality(),
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
        // The viewer shows a volume plane by plane.
        ZStackHandling::Volume3d => ZProjection::None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use evanalyzer_app::images::Point2d;
    use evanalyzer_cfg::core_types::ImagePlane;
    use kornia_image::ImageSize;

    /// The image the tests' tiles are of.
    fn test_image() -> Arc<PathBuf> {
        Arc::new(PathBuf::from("/test/image.tif"))
    }

    fn key(series: i32, level: i32, t: i32, x: usize, y: usize, w: usize, h: usize) -> TileKey {
        TileKey {
            image: test_image(),
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
            dimensionality: ImageDimensionality::Planar,
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
                    dimensionality: ImageDimensionality::Planar,
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
            image: test_image(),
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

    /// Every value of `data`'s first channel against the pattern.
    fn assert_shows_pattern(data: &RenderSource, region: &GridRegion) {
        let (w, values) = view_values(data);
        assert_eq!(w, region.out_w);
        assert_eq!(values.len(), region.out_w * region.out_h);
        for (i, v) in values.iter().enumerate() {
            let (ox, oy) = (i % w, i / w);
            assert_eq!(
                v.to_bits(),
                pattern_value(region.x + ox, region.y + oy).to_bits(),
                "({ox}, {oy})"
            );
        }
    }

    /// The same view again (a contrast drag, a channel toggle) is the view
    /// already assembled - not a copy of every cell again - and shows the
    /// same pixels. Another view, or the same cells read again after they
    /// left the cache, are assembled anew.
    #[test]
    fn the_same_cells_again_give_the_view_already_assembled() {
        let cache = Mutex::new(IndexedTileCache::new(NonZeroUsize::new(1 << 30).unwrap()));
        let source = PatternSource::default();
        let view = grid_view(1, [1100.0, 700.0, 2900.0, 1800.0]);

        let (first, region) = read_grid_view(&cache, &source, &view, 8).unwrap();
        let (again, region_again) = read_grid_view(&cache, &source, &view, 8).unwrap();
        assert!(Arc::ptr_eq(&first, &again), "the same view, uncopied");
        assert_eq!(region, region_again);
        assert_shows_pattern(&again, &region);

        // Panned by one cell: other cells, a new view with their pixels.
        let panned = grid_view(1, [1100.0 + 512.0, 700.0, 2900.0 + 512.0, 1800.0]);
        let (moved, moved_region) = read_grid_view(&cache, &source, &panned, 8).unwrap();
        assert!(!Arc::ptr_eq(&first, &moved));
        assert_ne!(moved_region, region);
        assert_shows_pattern(&moved, &moved_region);

        // Back: the cells are still cached, but the last view is the panned
        // one - assembled again, the same pixels.
        let (back, back_region) = read_grid_view(&cache, &source, &view, 8).unwrap();
        assert!(!Arc::ptr_eq(&moved, &back));
        assert_eq!(back_region, region);
        assert_shows_pattern(&back, &back_region);

        // The cache emptied (another image): the cells are read again -
        // a new view even though it lies at the same place.
        cache.lock().unwrap().clear();
        assert!(cache.lock().unwrap().last_view.is_none());
        let (fresh, fresh_region) = read_grid_view(&cache, &source, &view, 8).unwrap();
        assert!(!Arc::ptr_eq(&back, &fresh));
        assert_shows_pattern(&fresh, &fresh_region);
    }

    /// A cell evicted (its pixels freed) and read again is another cell:
    /// the view kept from the evicted one is not taken for it.
    #[test]
    fn a_cell_read_again_after_eviction_is_not_taken_for_the_old_one() {
        // Room for one 512 x 512 8-bit cell, not two.
        let cache = Mutex::new(IndexedTileCache::new(
            NonZeroUsize::new(512 * 512 + 1024).unwrap(),
        ));
        let source = PatternSource::default();
        let view = grid_view(1, [10.0, 10.0, 500.0, 500.0]);
        let other = grid_view(1, [600.0, 10.0, 1000.0, 500.0]);

        let (first, _) = read_grid_view(&cache, &source, &view, 8).unwrap();
        let cell = cache.lock().unwrap().get_exact(&view.key(0, 0)).unwrap();
        drop(cell);
        // Another cell evicts the first one.
        read_grid_view(&cache, &source, &other, 8).unwrap();
        assert!(cache.lock().unwrap().get_exact(&view.key(0, 0)).is_none());
        let reads = *source.pixels_read.lock().unwrap();

        let (again, region) = read_grid_view(&cache, &source, &view, 8).unwrap();
        assert!(*source.pixels_read.lock().unwrap() > reads, "read again");
        assert!(!Arc::ptr_eq(&first, &again));
        assert_shows_pattern(&again, &region);
    }

    /// A view of one compact cell is that cell itself (nothing copied) -
    /// with exactly what assembling it would give.
    #[test]
    fn a_one_cell_view_is_the_cell_itself() {
        let cache = Mutex::new(IndexedTileCache::new(NonZeroUsize::new(1 << 30).unwrap()));
        let source = PatternSource::default();
        let view = grid_view(1, [600.0, 600.0, 900.0, 900.0]);

        let (data, region) = read_grid_view(&cache, &source, &view, 8).unwrap();

        let cell = cache.lock().unwrap().get_exact(&view.key(1, 1)).unwrap();
        assert!(Arc::ptr_eq(&data, &cell.data), "the cached cell, uncopied");
        let assembled = assemble_channel(
            &[vec![cell.clone()]],
            0,
            GRID_CELL,
            region.out_w,
            region.out_h,
        )
        .unwrap();
        let ch = &data[0];
        assert_eq!(
            (ch.width, ch.height, ch.tile_offset, ch.c_stack, ch.is_rgb),
            (
                assembled.width,
                assembled.height,
                assembled.tile_offset,
                assembled.c_stack,
                assembled.is_rgb
            )
        );
        assert_eq!(
            view_values(&data),
            view_values(&Arc::new(vec![assembled])),
            "the same values"
        );
        assert_shows_pattern(&data, &region);
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

    /// The overview takes the sharp view's cells only where both read level
    /// 0 - the sharp view at the fitted view's subsample, which divides the
    /// overview's step.
    #[test]
    fn the_overview_shares_the_sharp_cells_only_at_the_fitted_subsample() {
        let single = |w: u64, h: u64| BTreeMap::from([(0, pyramid(w, h))]);
        let fit = |w: f32, h: f32| {
            let mut zoom = (1920.0 / w).min(1080.0 / h);
            zoom *= 0.95;
            (w * zoom, h * zoom)
        };
        // 2048 x 2048 in 1080p: fitted at every pixel (zoom 0.5), step 2.
        let (fw, fh) = fit(2048.0, 2048.0);
        let shared = sharp_subsample_shared_with_overview;
        assert_eq!(
            shared(&single(2048, 2048), fw, fh, 1920.0, 1080.0, 2),
            Some(1)
        );
        // Zoomed in to 100%: still every pixel - the same cells.
        assert_eq!(
            shared(&single(2048, 2048), 2048.0, 2048.0, 1920.0, 1080.0, 2),
            Some(1)
        );
        // 3000 x 3000: every 2nd pixel fitted, step 2.
        let (fw, fh) = fit(3000.0, 3000.0);
        assert_eq!(
            shared(&single(3000, 3000), fw, fh, 1920.0, 1080.0, 2),
            Some(2)
        );
        // ... zoomed in to 100%, every pixel: not the fitted cells.
        assert_eq!(
            shared(&single(3000, 3000), 3000.0, 3000.0, 1920.0, 1080.0, 2),
            None
        );
        // 4096 x 4096: every 3rd pixel fitted, step 4 - no divisor.
        let (fw, fh) = fit(4096.0, 4096.0);
        assert_eq!(shared(&single(4096, 4096), fw, fh, 1920.0, 1080.0, 4), None);
        // A pyramid whose coarser level the sharp view reads: nothing shared.
        let pyramid_levels = BTreeMap::from([
            (0, pyramid(8000, 8000)),
            (1, pyramid(2000, 2000)),
            (2, pyramid(500, 500)),
        ]);
        let (fw, fh) = fit(8000.0, 8000.0);
        assert_eq!(shared(&pyramid_levels, fw, fh, 1920.0, 1080.0, 7), None);
        // A degenerate view.
        assert_eq!(
            shared(&single(2048, 2048), 0.0, 0.0, 1920.0, 1080.0, 2),
            None
        );
    }

    /// A view of all of a `w` x `h` level 0 at `subsample`.
    fn whole_view(w: usize, h: usize, subsample: usize) -> GridView {
        GridView {
            subsample,
            level_w: w,
            level_h: h,
            visible: [0.0, 0.0, w as f32, h as f32],
            ..grid_view(subsample, [0.0; 4])
        }
    }

    /// Taken from the sharp view's cells, the overview's samples are
    /// bit for bit what reading it on its own gave - for every subsample
    /// that divides the step.
    #[test]
    fn the_overview_from_the_sharp_cells_is_what_reading_it_gives() {
        let (w, h) = (1500usize, 1100usize);
        for (step, subsample) in [(1, 1), (2, 1), (2, 2), (3, 1), (3, 3), (4, 2), (6, 3)] {
            let cache = Mutex::new(IndexedTileCache::new(NonZeroUsize::new(1 << 30).unwrap()));
            let source = PatternSource::default();
            let (cells, region) =
                read_grid_cells(&cache, &source, &whole_view(w, h, subsample), 8).unwrap();
            let from_cells =
                overview_samples(&cells, &region, step / subsample, step, w, h).unwrap();
            let read = read_subsampled(&source, &pattern_request(0, 0, w, h), step).unwrap();

            assert_eq!(from_cells.len(), read.len());
            for (a, b) in from_cells.iter().zip(&read) {
                assert_eq!(
                    a.image.size(),
                    b.image.size(),
                    "step {step}, subsample {subsample}"
                );
                let values = |c: &ImageChannel| match &*c.image {
                    ImageContainer::F32Gray(img) => img
                        .as_slice()
                        .iter()
                        .map(|v| v.to_bits())
                        .collect::<Vec<_>>(),
                    _ => panic!("gray"),
                };
                assert_eq!(values(a), values(b), "step {step}, subsample {subsample}");
                assert_eq!(a.image.tile_offset(), b.image.tile_offset());
                assert_eq!(
                    (a.c_stack, &a.name, a.color, a.is_rgb),
                    (b.c_stack, &b.name, b.color, b.is_rgb)
                );
            }
        }
    }

    /// A source that answers slowly (so two readers overlap), counting the
    /// pixels read; optionally failing or panicking on its first read.
    #[derive(Default)]
    struct SlowSource {
        inner: PatternSource,
        reads: std::sync::atomic::AtomicUsize,
        fail_first: bool,
        panic_first: bool,
    }

    impl ImageSource for SlowSource {
        fn meta(&self) -> &evanalyzer_app::images::ImageMeta {
            self.inner.meta()
        }

        fn read_tile(&self, req: &TileRequest) -> Result<Vec<ImageChannel>, InternalErrors> {
            let n = self.reads.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            std::thread::sleep(std::time::Duration::from_millis(20));
            if n == 0 && self.panic_first {
                panic!("the reader fails");
            }
            if n == 0 && self.fail_first {
                return Err(InternalErrors::ImageReadError("unreadable".into()));
            }
            self.inner.read_tile(req)
        }
    }

    /// Runs `f` on its own thread, failing the test if it takes over 20 s
    /// (a reader waiting forever).
    fn within_time<T: Send + 'static>(f: impl FnOnce() -> T + Send + 'static) -> T {
        let (done, result) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            let _ = done.send(f());
        });
        result
            .recv_timeout(std::time::Duration::from_secs(20))
            .expect("a reader waited forever")
    }

    /// Two readers wanting the same cells at once (the overview and the
    /// sharp view of an image without a pyramid) read each cell once: one
    /// reads, the other waits for it - and both get the same cells.
    #[test]
    fn two_readers_of_the_same_cells_read_each_once() {
        let cache = Arc::new(Mutex::new(IndexedTileCache::new(
            NonZeroUsize::new(1 << 30).unwrap(),
        )));
        let source = Arc::new(SlowSource::default());
        let view = Arc::new(whole_view(1500, 1100, 1));
        let read =
            |cache: Arc<Mutex<IndexedTileCache>>, source: Arc<SlowSource>, view: Arc<GridView>| {
                std::thread::spawn(move || read_grid_cells(&cache, &*source, &view, 8).unwrap())
            };
        let a = read(cache.clone(), source.clone(), view.clone());
        let b = read(cache.clone(), source.clone(), view.clone());
        let ((a, _), (b, _)) = within_time(move || (a.join().unwrap(), b.join().unwrap()));

        assert_eq!(
            *source.inner.pixels_read.lock().unwrap(),
            1500 * 1100,
            "once"
        );
        for (x, y) in a.iter().flatten().zip(b.iter().flatten()) {
            assert!(Arc::ptr_eq(x, y), "the same cells");
        }
        assert!(cache.lock().unwrap().in_flight.is_empty());
    }

    /// The reader others wait for fails or panics: its cells are released,
    /// and a waiting reader reads them itself - nobody waits forever.
    #[test]
    fn a_failing_or_panicking_reader_leaves_no_one_waiting() {
        for panics in [false, true] {
            let cache = Arc::new(Mutex::new(IndexedTileCache::new(
                NonZeroUsize::new(1 << 30).unwrap(),
            )));
            let source = Arc::new(SlowSource {
                fail_first: !panics,
                panic_first: panics,
                ..Default::default()
            });
            // One cell: the first reader claims it, the second waits.
            let view = Arc::new(grid_view(1, [10.0, 10.0, 400.0, 400.0]));
            let first = {
                let (cache, source, view) = (cache.clone(), source.clone(), view.clone());
                std::thread::spawn(move || read_grid_cells(&cache, &*source, &view, 8).map(|_| ()))
            };
            std::thread::sleep(std::time::Duration::from_millis(5));
            let second = {
                let (cache, source, view) = (cache.clone(), source.clone(), view.clone());
                std::thread::spawn(move || read_grid_cells(&cache, &*source, &view, 8).map(|_| ()))
            };
            let (first, second) =
                within_time(move || (first.join(), second.join().expect("no panic")));
            match first {
                Ok(result) => assert!(result.is_err() && !panics),
                Err(_) => assert!(panics, "the first reader panicked"),
            }
            assert!(second.is_ok(), "read by the second reader");
            assert!(cache.lock().unwrap().in_flight.is_empty());
            assert!(cache.lock().unwrap().get_exact(&view.key(0, 0)).is_some());
        }
    }

    /// A real image without a pyramid (the fixture), fitted: its overview is
    /// taken from the sharp view's cells (they are in the tile cache after
    /// it) and is, value for value, the overview read on its own as before
    /// - for every plane.
    #[test]
    fn an_image_without_a_pyramid_gets_the_overview_it_got_when_read_alone() {
        let app_state = crate::editor::test_support::test_ui_state_with_project(
            crate::editor::test_support::project_with_fixture_image(),
        );
        let cache = ViewportCache::new(app_state.clone());
        let state = ViewportState {
            viewport_width: 400.0,
            viewport_height: 300.0,
            zoom: 1.0,
            offset_x: 0.0,
            offset_y: 0.0,
            mouse_pos_x: 0.0,
            mouse_pos_y: 0.0,
        };
        let path = app_state
            .get_project()
            .get_current_image_path_cloned()
            .unwrap();
        let source = app_state.get_image_source(&path).unwrap();
        let level = source.meta().series[&0].resolutions[&0].clone();
        assert_eq!(
            source.meta().series[&0].resolutions.len(),
            1,
            "the fixture has no pyramid"
        );
        let planes = source.meta().series[&0].nr_z_stacks.max(1) as i32;
        let (w, h) = (level.width as usize, level.height as usize);

        for z in 0..planes {
            let z_range = Some(z..=z);
            let (data, _) = cache
                .read_image_tile_combined(
                    0,
                    ZProjection::None,
                    z_range.clone(),
                    0,
                    true,
                    z == 0,
                    true,
                    &state,
                )
                .unwrap();
            assert!(
                cache.held_bytes().0 > 0,
                "read as sharp-view cells (plane {z})"
            );

            // As before: read alone, scaled, compacted.
            let ratio = w as f32 / h as f32;
            let (key_w, key_h) = (1024usize, (1024.0 / ratio).max(1.0) as usize);
            let step = (w / key_w).min(h / key_h).max(1);
            let request = TileRequest {
                series: 0,
                resolution_idx: 0,
                z_projection: ZProjection::None,
                z_range,
                t_stack: 0,
                tile: ImageTile {
                    offset_x: 0,
                    offset_y: 0,
                    width: w,
                    height: h,
                },
            };
            let alone = scale_image(
                read_subsampled(&*source, &request, step).unwrap(),
                key_w,
                key_h,
            )
            .unwrap()
            .iter()
            .map(|c| ViewChannel::from_image_channel(c, level.nr_bits))
            .collect::<Vec<_>>();

            assert_eq!(data.len(), alone.len());
            for (a, b) in data.iter().zip(&alone) {
                assert_eq!((a.width, a.height), (b.width, b.height));
                assert_eq!(
                    (a.tile_offset, a.c_stack, &a.name, a.color, a.is_rgb),
                    (b.tile_offset, b.c_stack, &b.name, b.color, b.is_rgb)
                );
                let (sa, sb) = (a.slice().unwrap(), b.slice().unwrap());
                assert_eq!(sa.len(), sb.len());
                assert!(
                    (0..sa.len()).all(|i| sa.get(i).to_bits() == sb.get(i).to_bits()),
                    "plane {z}, channel {}",
                    a.c_stack
                );
                assert_eq!(
                    std::mem::discriminant(&a.samples),
                    std::mem::discriminant(&b.samples)
                );
            }
        }
    }

    /// The cells of one image never stand in for another's, even before the
    /// cache is emptied for it.
    #[test]
    fn cells_of_another_image_are_never_taken() {
        let cache = Mutex::new(IndexedTileCache::new(NonZeroUsize::new(1 << 30).unwrap()));
        let source = PatternSource::default();
        let view = grid_view(1, [10.0, 10.0, 400.0, 400.0]);
        read_grid_cells(&cache, &source, &view, 8).unwrap();
        let reads = *source.pixels_read.lock().unwrap();
        let other = GridView {
            image: Arc::new(PathBuf::from("/test/other.tif")),
            ..grid_view(1, [10.0, 10.0, 400.0, 400.0])
        };
        read_grid_cells(&cache, &source, &other, 8).unwrap();
        assert!(
            *source.pixels_read.lock().unwrap() > reads,
            "read for the other image"
        );

        // Another image opens: the last one's tiles go, its own stay.
        cache
            .lock()
            .unwrap()
            .retain_image(std::path::Path::new("/test/other.tif"));
        assert!(cache.lock().unwrap().get_exact(&view.key(0, 0)).is_none());
        assert!(cache.lock().unwrap().get_exact(&other.key(0, 0)).is_some());
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

    // -- other sample layouts (RGB, labels, 16 bit) -------------------------------

    fn rgb_channel(w: usize, h: usize, offset: Point2d) -> ImageChannel {
        // Pixel i holds (i, i + 0.25, i + 0.5) / 1000.
        let data = (0..w * h)
            .flat_map(|i| {
                let v = i as f32 / 1000.0;
                [v, v + 0.00025, v + 0.0005]
            })
            .collect();
        ImageChannel {
            image: Arc::new(ImageContainer::F32Rgb(ManagedImage {
                data: Image::<f32, 3>::new(
                    ImageSize {
                        width: w,
                        height: h,
                    },
                    data,
                )
                .unwrap(),
                tile_offset: offset,
                plane: None,
                dimensionality: ImageDimensionality::Planar,
            })),
            color: [1.0, 1.0, 1.0],
            is_visible: true,
            c_stack: 0,
            name: "rgb".into(),
            is_rgb: true,
        }
    }

    fn label_channel(w: usize, h: usize) -> ImageChannel {
        ImageChannel {
            image: Arc::new(ImageContainer::U32(ManagedImage {
                data: Image::<u32, 1>::new(
                    ImageSize {
                        width: w,
                        height: h,
                    },
                    (0..(w * h) as u32).collect(),
                )
                .unwrap(),
                tile_offset: Point2d { x: 0, y: 0 },
                plane: None,
                dimensionality: ImageDimensionality::Planar,
            })),
            color: [1.0, 1.0, 1.0],
            is_visible: true,
            c_stack: 1,
            name: "labels".into(),
            is_rgb: false,
        }
    }

    /// A virtual image whose tiles are RGB and/or label channels.
    struct LayoutSource {
        meta: evanalyzer_app::images::ImageMeta,
        rgb: bool,
        labels: bool,
    }

    impl ImageSource for LayoutSource {
        fn meta(&self) -> &evanalyzer_app::images::ImageMeta {
            &self.meta
        }

        fn read_tile(&self, req: &TileRequest) -> Result<Vec<ImageChannel>, InternalErrors> {
            let t = req.tile;
            let mut channels = Vec::new();
            if self.rgb {
                channels.push(rgb_channel(
                    t.width,
                    t.height,
                    Point2d {
                        x: t.offset_x,
                        y: t.offset_y,
                    },
                ));
            }
            if self.labels {
                channels.push(label_channel(t.width, t.height));
            }
            Ok(channels)
        }
    }

    fn layout_source(rgb: bool, labels: bool) -> LayoutSource {
        LayoutSource {
            meta: Default::default(),
            rgb,
            labels,
        }
    }

    #[test]
    fn a_subsampled_rgb_read_keeps_all_three_samples_of_every_kept_pixel() {
        let source = layout_source(true, false);
        let channels = read_subsampled(&source, &pattern_request(4, 6, 10, 6), 2).unwrap();

        let ImageContainer::F32Rgb(img) = &*channels[0].image else {
            panic!("RGB stays RGB");
        };
        assert_eq!((img.width(), img.height()), (5, 3));
        assert_eq!(channels[0].image.tile_offset(), Point2d { x: 4, y: 6 });
        assert!(channels[0].is_rgb);
        let px = img.as_slice();
        // Output (1, 1) is source pixel (2, 2) of the 10 px wide region.
        let i = 2 * 10 + 2;
        let v = i as f32 / 1000.0;
        assert_eq!(
            &px[(5 + 1) * 3..(5 + 1) * 3 + 3],
            &[v, v + 0.00025, v + 0.0005]
        );
    }

    #[test]
    fn a_subsampled_read_of_a_label_channel_keeps_its_size_without_copying_its_values() {
        let source = layout_source(false, true);
        let channels = read_subsampled(&source, &pattern_request(0, 0, 9, 4), 3).unwrap();

        assert_eq!(channels.len(), 1);
        assert_eq!(channels[0].c_stack, 1);
        let size = channels[0].image.size();
        assert_eq!((size.width, size.height), (3, 2));
    }

    #[test]
    fn blocks_holding_no_pixel_of_the_sampling_grid_are_never_read() {
        let source = PatternSource::default();
        // One column past the first block column, one row past the first
        // block row - neither holds a pixel of the every-3rd grid.
        let (w, h) = (SUBSAMPLE_BLOCK_W + 1, SUBSAMPLE_BLOCK_H + 1);
        let channels = read_subsampled(&source, &pattern_request(0, 0, w, h), 3).unwrap();

        let size = channels[0].image.size();
        assert_eq!((size.width, size.height), (w.div_ceil(3), h.div_ceil(3)));
        assert_eq!(
            *source.pixels_read.lock().unwrap(),
            SUBSAMPLE_BLOCK_W * SUBSAMPLE_BLOCK_H,
            "only the first block is read"
        );
    }

    #[test]
    fn cropping_rgb_columns_keeps_whole_pixels_and_moves_the_offset() {
        let channel = rgb_channel(4, 2, Point2d { x: 10, y: 20 });

        let cropped = crop_columns(&channel, 1, 2).unwrap();

        let ImageContainer::F32Rgb(img) = &*cropped.image else {
            panic!("RGB stays RGB");
        };
        assert_eq!((img.width(), img.height()), (2, 2));
        assert_eq!(cropped.image.tile_offset(), Point2d { x: 11, y: 20 });
        // Row 1, column 0 of the crop = source pixel 1 * 4 + 1.
        let v = 5.0 / 1000.0;
        assert_eq!(&img.as_slice()[6..9], &[v, v + 0.00025, v + 0.0005]);
        assert_eq!(cropped.name, "rgb");
        assert!(cropped.is_rgb);
    }

    #[test]
    fn scaling_keeps_rgb_channels_rgb_and_label_channels_untouched() {
        let channels = vec![
            rgb_channel(4, 4, Point2d { x: 3, y: 5 }),
            label_channel(4, 4),
        ];

        let scaled = scale_image(channels, 2, 2).unwrap();

        let ImageContainer::F32Rgb(rgb) = &*scaled[0].image else {
            panic!("RGB stays RGB");
        };
        assert_eq!((rgb.width(), rgb.height()), (2, 2));
        assert_eq!(scaled[0].image.tile_offset(), Point2d { x: 3, y: 5 });
        let ImageContainer::U32(labels) = &*scaled[1].image else {
            panic!("labels stay labels");
        };
        assert_eq!((labels.width(), labels.height()), (4, 4), "never resampled");
        assert_eq!(scaled[1].c_stack, 1);
    }

    fn view_cell(samples: Samples, w: usize, h: usize) -> Arc<CachedTile> {
        Arc::new(CachedTile {
            data: Arc::new(vec![ViewChannel {
                samples,
                width: w,
                height: h,
                tile_offset: Point2d { x: 0, y: 0 },
                color: [0.0, 1.0, 0.0],
                c_stack: 2,
                name: "c".into(),
                is_rgb: false,
            }]),
        })
    }

    fn u16_cell(values: &[u16]) -> Arc<CachedTile> {
        view_cell(
            Samples::U16 {
                data: values.into(),
                inv: 1.0 / 65535.0,
            },
            2,
            2,
        )
    }

    #[test]
    fn sixteen_bit_cells_are_assembled_without_widening_them() {
        let cells = vec![vec![u16_cell(&[1, 2, 3, 4]), u16_cell(&[5, 6, 7, 8])]];

        let assembled = assemble_channel(&cells, 0, 2, 4, 2).unwrap();

        let Samples::U16 { data, inv } = &assembled.samples else {
            panic!("16-bit stays compact");
        };
        assert_eq!(&data[..], &[1, 2, 5, 6, 3, 4, 7, 8]);
        assert_eq!(*inv, 1.0 / 65535.0);
        assert_eq!((assembled.width, assembled.height), (4, 2));
        assert_eq!(assembled.c_stack, 2);
        assert_eq!(assembled.color, [0.0, 1.0, 0.0]);
    }

    #[test]
    fn cells_of_mixed_sample_types_are_assembled_as_floats() {
        let u8_cell = view_cell(
            Samples::U8 {
                data: vec![255u8, 0, 0, 255].into(),
                inv: 1.0 / 255.0,
            },
            2,
            2,
        );
        let cells = vec![vec![u8_cell, u16_cell(&[0, 65535, 65535, 0])]];

        let assembled = assemble_channel(&cells, 0, 2, 4, 2).unwrap();

        assert!(matches!(assembled.samples, Samples::Image(_)));
        let s = assembled.slice().unwrap();
        let values: Vec<f32> = (0..s.len()).map(|i| s.get(i)).collect();
        assert_eq!(values, vec![1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0]);
    }

    #[test]
    fn cells_that_cannot_be_rendered_fail_the_assembly_instead_of_panicking() {
        let rgb = ViewChannel::wrap(&rgb_channel(2, 2, Point2d { x: 0, y: 0 }));
        let cells = vec![vec![Arc::new(CachedTile {
            data: Arc::new(vec![rgb]),
        })]];

        assert!(assemble_channel(&cells, 0, 2, 2, 2).is_err());
    }

    #[test]
    fn a_view_too_big_for_the_cache_is_still_returned_just_not_kept() {
        let cache = Mutex::new(IndexedTileCache::new(NonZeroUsize::new(1).unwrap()));
        let source = PatternSource::default();

        let (data, region) =
            read_grid_view(&cache, &source, &grid_view(1, [0.0, 0.0, 600.0, 100.0]), 8).unwrap();

        assert_eq!((region.out_w, region.out_h), (1024, 512));
        let (w, values) = view_values(&data);
        assert_eq!(values[3 * w + 700], pattern_value(700, 3));
        assert_eq!(cache.lock().unwrap().weight(), 0, "nothing cached");
    }

    // -- read_image_tile_combined (fixture image file) ----------------------------

    fn fixture_cache() -> ViewportCache {
        ViewportCache::new(crate::editor::test_support::test_ui_state_with_project(
            crate::editor::test_support::project_with_fixture_image(),
        ))
    }

    fn viewport(width: f32, height: f32, zoom: f32) -> ViewportState {
        ViewportState {
            viewport_width: width,
            viewport_height: height,
            zoom,
            offset_x: 0.0,
            offset_y: 0.0,
            mouse_pos_x: 0.0,
            mouse_pos_y: 0.0,
        }
    }

    fn read(
        cache: &ViewportCache,
        fit_to_screen: bool,
        is_low_res: bool,
        state: &ViewportState,
    ) -> Result<(RenderSource, ReadContext), InternalErrors> {
        cache.read_image_tile_combined(
            0,
            ZProjection::None,
            Some(0..=0),
            0,
            fit_to_screen,
            false,
            is_low_res,
            state,
        )
    }

    fn error_of(result: Result<(RenderSource, ReadContext), InternalErrors>) -> String {
        match result {
            Ok(_) => panic!("an error expected"),
            Err(e) => format!("{e:?}"),
        }
    }

    #[test]
    fn fitting_to_a_viewport_not_yet_laid_out_is_deferred() {
        let cache = fixture_cache();
        for is_low_res in [false, true] {
            for state in [viewport(0.0, 300.0, 1.0), viewport(400.0, 0.0, 1.0)] {
                let error = error_of(read(&cache, true, is_low_res, &state));
                assert!(error.contains("not yet sized"), "{error}");
            }
        }
    }

    #[test]
    fn a_zoom_of_zero_is_refused_rather_than_drawn() {
        let cache = fixture_cache();
        let state = viewport(400.0, 300.0, 0.0);
        let sharp = error_of(read(&cache, false, false, &state));
        assert!(sharp.contains("Degenerate high-res"), "{sharp}");
        let overview = error_of(read(&cache, false, true, &state));
        assert!(overview.contains("Degenerate low-res"), "{overview}");
        assert_eq!(cache.held_bytes(), (0, 0));
    }

    #[test]
    fn an_overview_too_big_for_the_cache_is_still_drawn() {
        let cache = fixture_cache();
        *cache.cache_low_res.lock().unwrap() = IndexedTileCache::new(NonZeroUsize::new(1).unwrap());

        let (data, ctx) = read(&cache, true, true, &viewport(400.0, 300.0, 1.0)).unwrap();

        assert!(!data.is_empty());
        assert!(ctx.image_w > 0 && ctx.image_h > 0);
        assert_eq!(cache.held_bytes().1, 0, "not kept");
    }
}
