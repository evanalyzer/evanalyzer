//! # image_cache
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-09-03
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.

use crate::ImageDimensionality;
use clru::{CLruCache, CLruCacheConfig, WeightScale};
use log::{debug, warn};

use crate::image::Point2d;
use crate::{ImageContainer, ImagePlane, ManagedImage, pipeline::pipeline_cache::CacheAddress};
use kornia_image::{Image, ImageSize};
use std::{
    collections::{HashMap, hash_map::RandomState},
    fs::File,
    io::{self, BufReader, BufWriter, Read, Write},
    num::NonZeroUsize,
    path::Path,
    sync::{Arc, Mutex},
    time::Instant,
};

/// On-disk tag for which `ImageContainer` variant a spilled image is -
/// written first so `read_image` knows how to reconstruct it.
const TAG_F32_GRAY: u8 = 0;
const TAG_F32_RGB: u8 = 1;
const TAG_U32: u8 = 2;

/// Bytes `write_header` writes before the pixels: tag, width, height, tile
/// offset x/y, has-plane flag, z/c/t.
const HEADER_BYTES: u64 = 1 + 4 * 8 + 1 + 3 * 4;

/// Default cap on `hot_cache`'s combined pixel-data size. Bounds memory
/// use regardless of how many tiles/objects a whole-slide image produces -
/// once this is hit, the least-recently-used entries are evicted to disk
/// (see `spill_to_disk`) to make room.
/// TODO: make this configurable based on the user's available RAM.
const DEFAULT_HOT_CACHE_CAPACITY_BYTES: usize = 1024 * 1024 * 1024; // 1 GiB

/// Weighs a `hot_cache` entry by its actual pixel-data size (see
/// `ImageContainer::get_image_memory_usage`), so `DEFAULT_HOT_CACHE_CAPACITY_BYTES`
/// is a real memory bound rather than an entry count - a 4096x4096 RGB tile
/// and a 512x512 gray tile don't count the same.
struct ImageWeight;

impl WeightScale<CacheAddress, Arc<ImageContainer>> for ImageWeight {
    fn weight(&self, _key: &CacheAddress, value: &Arc<ImageContainer>) -> usize {
        value.get_image_memory_usage()
    }
}

type HotCache = CLruCache<CacheAddress, Arc<ImageContainer>, RandomState, ImageWeight>;

fn new_hot_cache(capacity_bytes: usize) -> HotCache {
    // `.max(1)`: `NonZeroUsize` can't hold 0 - a real caller should never
    // ask for a zero-byte cache, but this avoids a panic if one ever does,
    // same as this whole module's other `.max(1)` guards.
    let capacity = NonZeroUsize::new(capacity_bytes.max(1)).unwrap();
    CLruCache::with_config(CLruCacheConfig::new(capacity).with_scale(ImageWeight))
}

/// The one spill file of a cache and every clone of it: an unnamed temp
/// file the OS deletes by itself when the last handle closes - also when
/// the process is killed or crashes, so a whole slide's spill can't be left
/// behind. One file (one handle) however many images spill, so a big slide
/// can't run into the open-file limit (256 on macOS by default).
///
/// Spilled images are written one after the other, each at its own offset,
/// with positional reads/writes - several threads use the file at once
/// without a shared file position. A freed range ([`SpillSlot`] dropped) is
/// reused by the next image that fits; the file only shrinks when it's
/// deleted.
struct SpillFile {
    file: File,
    space: Mutex<SpillSpace>,
}

#[derive(Default)]
struct SpillSpace {
    /// The file's end: where the next image goes when no free range fits.
    end: u64,
    /// Freed ranges (offset, length).
    free: Vec<(u64, u64)>,
}

impl SpillFile {
    /// In `dir`, or the system temp folder (`None`).
    fn new(dir: Option<&Path>) -> io::Result<Self> {
        let file = match dir {
            Some(dir) => tempfile::tempfile_in(dir)?,
            None => tempfile::tempfile()?,
        };
        Ok(Self {
            file,
            space: Mutex::new(SpillSpace::default()),
        })
    }

    /// A range of `len` bytes: the first free one it fits in, else at the end.
    fn allocate(&self, len: u64) -> u64 {
        let mut space = self.space.lock().expect("Poisned thread");
        if let Some(i) = space.free.iter().position(|&(_, free)| free >= len) {
            let (offset, free) = space.free.swap_remove(i);
            if free > len {
                space.free.push((offset + len, free - len));
            }
            return offset;
        }
        let offset = space.end;
        space.end += len;
        offset
    }

    fn release(&self, offset: u64, len: u64) {
        let mut space = self.space.lock().expect("Poisned thread");
        space.free.push((offset, len));
    }
}

/// Where one spilled image is in its [`SpillFile`]. Shared (`Arc`) by every
/// cache index that holds the image - clones and merges copy the index - so
/// its range is only freed when the last of them drops it: no copy can read
/// a range that another image already reuses.
struct SpillSlot {
    file: Arc<SpillFile>,
    offset: u64,
    len: u64,
}

impl Drop for SpillSlot {
    fn drop(&mut self) {
        self.file.release(self.offset, self.len);
    }
}

/// [`Read`]/[`Write`] at a position of a file, without moving its shared
/// file position.
struct FileAt<'a> {
    file: &'a File,
    position: u64,
}

impl Write for FileAt<'_> {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        #[cfg(unix)]
        let written = std::os::unix::fs::FileExt::write_at(self.file, buf, self.position)?;
        #[cfg(windows)]
        let written = std::os::windows::fs::FileExt::seek_write(self.file, buf, self.position)?;
        self.position += written as u64;
        Ok(written)
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

impl Read for FileAt<'_> {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        #[cfg(unix)]
        let read = std::os::unix::fs::FileExt::read_at(self.file, buf, self.position)?;
        #[cfg(windows)]
        let read = std::os::windows::fs::FileExt::seek_read(self.file, buf, self.position)?;
        self.position += read as u64;
        Ok(read)
    }
}

/// `image_index` and `hot_cache` behind one lock, not two: `spill_to_disk`
/// needs to check-then-insert into `image_index` in lockstep with evicting
/// from `hot_cache` (see its own doc comment), and `get`/`insert` can race
/// each other across threads on the same `ImageCache` (e.g. rayon-parallel
/// object measurement in the whole-image phase all reading the same merged
/// cache) - a single lock makes that whole sequence atomic instead of
/// leaving a window where the two could disagree.
struct ImageCacheInner {
    // Addresses that have actually been written to the spill file, and
    // where. An address only appears here once it's been spilled out of
    // `hot_cache` (see `spill_to_disk`) - most images live and die entirely
    // in `hot_cache` and never touch disk at all.
    image_index: HashMap<CacheAddress, Arc<SpillSlot>>,
    // Size-bounded, least-recently-used store of resident images. Every
    // stored address is either only here (never spilled), or here *and* in
    // `image_index` (spilled once, then loaded back by `get`).
    hot_cache: HotCache,
}

pub struct ImageCache {
    inner: Mutex<ImageCacheInner>,
    // Shared by every clone descended from the same `new()` call: real
    // usage (`job_executor.rs`) clones a `global_cache` once per tile, each
    // tile spills into its clone, and every clone is merged back via
    // `extend` - which moves the index entries ([`SpillSlot`]s, each holding
    // the file too). The file closes, and the OS deletes it, once the last
    // clone and the last slot are gone.
    spill: Arc<SpillFile>,
}

/// Can't `#[derive(Clone)]`: `Mutex<T>` doesn't implement `Clone` even when
/// `T` does (a lock can't be duplicated without briefly acquiring it, and
/// std won't do that implicitly - a poisoned lock would make the clone
/// itself fallible in a way `Clone::clone` can't express).
///
/// Unconditionally safe regardless of how many entries `self` already has:
/// the spill file is shared (see the field doc above), and the clone's index
/// shares each spilled image's slot with `self`'s.
impl Clone for ImageCache {
    fn clone(&self) -> Self {
        // `CLruCache` isn't `Clone` either, so rebuild one from `self`'s
        // entries rather than cloning the container directly - preserving
        // `self`'s actual configured capacity, not the default.
        let source = self.inner.lock().expect("Poisned thread");
        let mut hot_cache = new_hot_cache(source.hot_cache.capacity());
        for (address, image) in source.hot_cache.iter() {
            let _ = hot_cache.put_with_weight(*address, Arc::clone(image));
        }
        Self {
            inner: Mutex::new(ImageCacheInner {
                image_index: source.image_index.clone(),
                hot_cache,
            }),
            spill: Arc::clone(&self.spill),
        }
    }
}

impl ImageCache {
    /// Creates a cache with its own spill file, with `hot_cache` capped at
    /// [`DEFAULT_HOT_CACHE_CAPACITY_BYTES`]. For production pipeline runs,
    /// prefer [`with_capacity_bytes`](Self::with_capacity_bytes) with a
    /// budget sized against actual parallelism/available RAM (see
    /// `memory::analysis::recommended_image_cache_bytes`) - this default exists for
    /// callers (tests, ad-hoc/GUI-preview caches) that don't have that
    /// context.
    pub fn new() -> io::Result<Self> {
        Self::with_capacity_bytes(DEFAULT_HOT_CACHE_CAPACITY_BYTES as u64)
    }

    /// Creates a cache with its own spill file in the system temp folder,
    /// with `hot_cache` capped at `capacity_bytes`. Fails only if the OS
    /// can't create a temp file (no permissions, no temp folder) - callers
    /// should treat that as fatal for whatever operation needed a
    /// disk-backed cache in the first place.
    pub fn with_capacity_bytes(capacity_bytes: u64) -> io::Result<Self> {
        Ok(Self {
            inner: Mutex::new(ImageCacheInner {
                image_index: Default::default(),
                hot_cache: new_hot_cache(capacity_bytes as usize),
            }),
            spill: Arc::new(SpillFile::new(None)?),
        })
    }

    /// Changes `hot_cache`'s cap to `capacity_bytes`. Entries over a smaller
    /// cap are spilled to disk, not dropped.
    pub fn set_capacity_bytes(&mut self, capacity_bytes: u64) {
        let inner = self.inner.get_mut().expect("Poisned thread");
        let capacity = NonZeroUsize::new((capacity_bytes as usize).max(1)).unwrap();
        while inner.hot_cache.len() + inner.hot_cache.weight() >= capacity.get() {
            let Some((address, image)) = inner.hot_cache.pop_back() else {
                break;
            };
            spill_to_disk(&self.spill, &mut inner.image_index, address, &image);
        }
        inner.hot_cache.resize(capacity);
    }

    /// Inserts `img` at `address`, returning the previously stored image (if
    /// any) - matches `HashMap::insert`.
    ///
    /// Only ever caches in `hot_cache` here - never writes to disk directly.
    /// Most images (anything not part of a whole-slide-sized run) live their
    /// entire life in `hot_cache` and are dropped without ever touching
    /// disk. A disk write only happens later, and only for whichever entries
    /// `hot_cache`'s size limit actually forces out - see `spill_to_disk`.
    pub fn insert(
        &mut self,
        address: CacheAddress,
        img: Arc<ImageContainer>,
    ) -> Option<Arc<ImageContainer>> {
        let mut inner = self.inner.lock().expect("Poisned thread");
        // Replacing this address invalidates what was spilled for the old
        // value - its slot is freed for reuse once no other copy holds it.
        inner.image_index.remove(&address);
        let ImageCacheInner {
            image_index,
            hot_cache,
        } = &mut *inner;
        hot_cache_insert(&self.spill, image_index, hot_cache, address, img.clone());
        Some(img)
    }

    /// Moves every entry out of `other` and into `self` - matches
    /// `Extend::extend`. Safe to call even after `other` is dropped
    /// immediately afterwards: each spilled entry's slot holds its spill
    /// file, so what `other` spilled stays readable. `other`'s `hot_cache`
    /// entries are re-inserted through the same size-bounded path `insert`
    /// uses, so merging can still spill to disk if the combined result needs
    /// the room.
    pub fn extend(&mut self, other: ImageCache) {
        let ImageCacheInner {
            image_index: other_index,
            hot_cache: other_hot,
        } = other.inner.into_inner().expect("Poisned thread");

        let mut inner = self.inner.lock().expect("Poisned thread");
        inner.image_index.extend(other_index);
        let ImageCacheInner {
            image_index,
            hot_cache,
        } = &mut *inner;
        for (address, image) in other_hot {
            hot_cache_insert(&self.spill, image_index, hot_cache, address, image);
        }
    }

    /// Matches `HashMap::contains_key`: `&self`, key by reference. An
    /// address counts whether it's spilled to disk, still only resident in
    /// `hot_cache`, or both.
    pub fn contains_key(&self, address: &CacheAddress) -> bool {
        let inner = self.inner.lock().expect("Poisned thread");
        inner.image_index.contains_key(address) || inner.hot_cache.contains(address)
    }

    /// Matches `HashMap::len`: `&self`. Counts each address once, even if
    /// it happens to be both spilled to disk *and* currently resident in
    /// `hot_cache` (loaded back by a `get` after being spilled once).
    pub fn len(&self) -> usize {
        self.keys().count()
    }

    /// Matches `HashMap::is_empty`: `&self`.
    pub fn is_empty(&self) -> bool {
        let inner = self.inner.lock().expect("Poisned thread");
        inner.image_index.is_empty() && inner.hot_cache.is_empty()
    }

    /// Matches `HashMap::keys`: every registered address, without loading
    /// any pixel data from disk. Unlike before, this can't stay a zero-copy
    /// borrow of a single map: an address may only exist in `hot_cache`
    /// (never spilled), so this briefly locks `hot_cache` to collect the
    /// union with `image_index`, fully materialized before returning (a
    /// borrowed iterator can't outlive the lock guard). Returns owned
    /// `CacheAddress`es (it's `Copy`) rather than references, for the same
    /// reason.
    pub fn keys(&self) -> impl Iterator<Item = CacheAddress> + use<> {
        let inner = self.inner.lock().expect("Poisned thread");
        let mut addresses: Vec<CacheAddress> = inner.image_index.keys().copied().collect();
        for (address, _) in inner.hot_cache.iter() {
            if !inner.image_index.contains_key(address) {
                addresses.push(*address);
            }
        }
        addresses.into_iter()
    }

    /// Returns the image at `address`, transparently loading it from disk
    /// on a miss and caching it back in `hot_cache` - so a repeat `get` for
    /// the same address doesn't pay the disk read again (until it's evicted
    /// for space). A hit also bumps the entry to most-recently-used, which
    /// is why this needs a lock even for a read.
    ///
    /// Returns an owned `Arc<ImageContainer>`, not `&Arc<ImageContainer>`:
    /// a `&self` method can't return a reference that outlives the guard
    /// borrowed to produce it - that guard is a local value, dropped at the
    /// end of this call. Cloning the `Arc` (a cheap refcount bump, not a
    /// data copy - ~10ns regardless of image size) is the standard way
    /// around that, and lets the lock be held only for the lookup itself,
    /// never across the disk read below.
    pub fn get(&self, address: &CacheAddress) -> Option<Arc<ImageContainer>> {
        let slot = {
            let mut inner = self.inner.lock().expect("Poisned thread");
            if let Some(image) = inner.hot_cache.get(address) {
                return Some(image.clone());
            }
            Arc::clone(inner.image_index.get(address)?)
        };

        let image = load_from_disk(&slot)?;
        let mut inner = self.inner.lock().expect("Poisned thread");
        let ImageCacheInner {
            image_index,
            hot_cache,
        } = &mut *inner;
        hot_cache_insert(&self.spill, image_index, hot_cache, *address, image.clone());
        Some(image)
    }

    /// Keeps only the entries for which `f` returns `true`, evicting the
    /// rest from both `hot_cache` and `image_index` - a spilled entry's
    /// slot is freed for reuse once no other copy holds it.
    ///
    /// Unlike `HashMap::retain`, `f` only receives the key, not `&mut V`:
    /// this cache is disk-backed, so a disk-only entry has no in-memory
    /// `Arc<ImageContainer>` to hand the closure without loading it first -
    /// pure wasted I/O for a filter, and this cache's one real caller
    /// (`clear_pipeline_context`) only ever inspects the key anyway.
    pub fn retain<F>(&mut self, mut f: F)
    where
        F: FnMut(&CacheAddress) -> bool,
    {
        let to_remove: Vec<CacheAddress> = self.keys().filter(|address| !f(address)).collect();

        let mut inner = self.inner.lock().expect("Poisned thread");
        for address in to_remove {
            inner.hot_cache.pop(&address);
            inner.image_index.remove(&address);
        }
    }

    /// Matches `HashMap::iter`: an iterator of `(&K, &V)` pairs in
    /// unspecified order.
    // pub fn iter(&self) -> impl Iterator<Item = (&CacheAddress, &Arc<ImageContainer>)> {
    //     self.image_index.iter()
    // }

    /// Bytes [`Self::write_image`] writes for `image`.
    fn encoded_len(image: &ImageContainer) -> u64 {
        let pixels = match image {
            ImageContainer::F32Gray(img) => std::mem::size_of_val(img.as_slice()),
            ImageContainer::F32Rgb(img) => std::mem::size_of_val(img.as_slice()),
            ImageContainer::U32(img) => std::mem::size_of_val(img.as_slice()),
        };
        HEADER_BYTES + pixels as u64
    }

    fn write_image(w: &mut impl Write, image: &ImageContainer) -> io::Result<()> {
        match image {
            ImageContainer::F32Gray(img) => {
                Self::write_header(w, TAG_F32_GRAY, img.size(), &img.tile_offset, &img.plane)?;
                w.write_all(as_bytes(img.as_slice()))?;
            }
            ImageContainer::F32Rgb(img) => {
                Self::write_header(w, TAG_F32_RGB, img.size(), &img.tile_offset, &img.plane)?;
                w.write_all(as_bytes(img.as_slice()))?;
            }
            ImageContainer::U32(img) => {
                Self::write_header(w, TAG_U32, img.size(), &img.tile_offset, &img.plane)?;
                w.write_all(as_bytes(img.as_slice()))?;
            }
        }
        w.flush()
    }

    fn write_header(
        w: &mut impl Write,
        tag: u8,
        size: ImageSize,
        tile_offset: &Point2d,
        plane: &Option<ImagePlane>,
    ) -> io::Result<()> {
        w.write_all(&[tag])?;
        w.write_all(&(size.width as u64).to_ne_bytes())?;
        w.write_all(&(size.height as u64).to_ne_bytes())?;
        w.write_all(&(tile_offset.x as u64).to_ne_bytes())?;
        w.write_all(&(tile_offset.y as u64).to_ne_bytes())?;
        match plane {
            Some(p) => {
                w.write_all(&[1])?;
                w.write_all(&p.z.to_ne_bytes())?;
                w.write_all(&p.c.to_ne_bytes())?;
                w.write_all(&p.t.to_ne_bytes())?;
            }
            // Keep the header a fixed size regardless of whether `plane` is
            // set, so `read_header` doesn't need its own branch to match.
            None => w.write_all(&[0; 1 + 12])?,
        }
        Ok(())
    }

    fn read_image(r: &mut impl Read) -> io::Result<ImageContainer> {
        let mut tag = [0u8; 1];
        r.read_exact(&mut tag)?;
        let width = read_u64(r)? as usize;
        let height = read_u64(r)? as usize;
        let tile_offset = Point2d {
            x: read_u64(r)? as usize,
            y: read_u64(r)? as usize,
        };
        let mut has_plane = [0u8; 1];
        r.read_exact(&mut has_plane)?;
        let plane = if has_plane[0] != 0 {
            Some(ImagePlane {
                z: read_i32(r)?,
                c: read_i32(r)?,
                t: read_i32(r)?,
            })
        } else {
            let mut skip = [0u8; 12];
            r.read_exact(&mut skip)?;
            None
        };
        let size = ImageSize { width, height };

        let to_io_err = |e: std::string::String| io::Error::new(io::ErrorKind::InvalidData, e);

        match tag[0] {
            TAG_F32_GRAY => {
                let data = read_pod_vec(r, width * height, 0f32)?;
                Ok(ImageContainer::F32Gray(ManagedImage {
                    data: Image::<f32, 1>::new(size, data)
                        .map_err(|e| to_io_err(format!("{e:?}")))?,
                    tile_offset,
                    plane,
                    dimensionality: ImageDimensionality::Planar,
                }))
            }
            TAG_F32_RGB => {
                let data = read_pod_vec(r, width * height * 3, 0f32)?;
                Ok(ImageContainer::F32Rgb(ManagedImage {
                    data: Image::<f32, 3>::new(size, data)
                        .map_err(|e| to_io_err(format!("{e:?}")))?,
                    tile_offset,
                    plane,
                    dimensionality: ImageDimensionality::Planar,
                }))
            }
            TAG_U32 => {
                let data = read_pod_vec(r, width * height, 0u32)?;
                Ok(ImageContainer::U32(ManagedImage {
                    data: Image::<u32, 1>::new(size, data)
                        .map_err(|e| to_io_err(format!("{e:?}")))?,
                    tile_offset,
                    plane,
                    dimensionality: ImageDimensionality::Planar,
                }))
            }
            other => Err(to_io_err(format!("unknown image cache tag {other}"))),
        }
    }
}

/// No manual `Drop` needed here: the spill file is closed - and deleted by
/// the OS - once the last clone and the last [`SpillSlot`] sharing it drop.
impl Default for ImageCache {
    fn default() -> Self {
        Self::new().expect("failed to create the spill file for ImageCache::default()")
    }
}

/// Reads back an image `spill_to_disk` wrote into `slot`. Returns `None`
/// (logged) on any I/O or format error, since a cache miss is always a safe
/// fallback, but a hard error here would take down the whole pipeline over
/// what's just a scratch file.
fn load_from_disk(slot: &SpillSlot) -> Option<Arc<ImageContainer>> {
    let start = Instant::now();
    let mut r = BufReader::new(FileAt {
        file: &slot.file.file,
        position: slot.offset,
    })
    .take(slot.len);
    match ImageCache::read_image(&mut r) {
        Ok(image) => {
            debug!(
                "Loaded image ({} bytes) from the spill file at {} in {:?}",
                image.get_image_memory_usage(),
                slot.offset,
                start.elapsed()
            );
            Some(Arc::new(image))
        }
        Err(e) => {
            warn!(
                "Could not load image from the spill file at {}: {}",
                slot.offset, e
            );
            None
        }
    }
}

/// Puts `img` into `hot_cache`, evicting (and spilling to disk via
/// `spill_to_disk`) whatever least-recently-used entries are needed to make
/// room - or, if `img` alone is too big to ever fit, writing it straight to
/// disk without holding it in memory at all. A free function (not an
/// `ImageCache` method) so it can take `image_index`/`hot_cache` as
/// independent `&mut` borrows of one already-locked `ImageCacheInner`,
/// alongside `&self.spill` at the same call site.
fn hot_cache_insert(
    spill: &Arc<SpillFile>,
    image_index: &mut HashMap<CacheAddress, Arc<SpillSlot>>,
    hot_cache: &mut HotCache,
    address: CacheAddress,
    img: Arc<ImageContainer>,
) {
    let weight = img.get_image_memory_usage();
    if weight >= hot_cache.capacity() {
        warn!(
            "Image at {:?} ({} bytes) is larger than the hot cache's entire capacity \
             ({} bytes) - writing it straight to disk instead of holding it in memory.",
            address,
            weight,
            hot_cache.capacity()
        );
        spill_to_disk(spill, image_index, address, &img);
        return;
    }
    // Mirrors `CLruCache::put_with_weight`'s own "make room" formula
    // exactly (see its source), so this pre-eviction leaves the cache in a
    // state where the `put_with_weight` call below never needs to evict
    // anything further itself.
    while hot_cache.len() + hot_cache.weight() + weight >= hot_cache.capacity() {
        // unwrap: the loop condition and the `weight >= capacity` check
        // above together guarantee `hot_cache` isn't empty yet - each
        // iteration strictly reduces `len() + weight()`, and it can't reach
        // zero while the condition (which `weight` alone already satisfies
        // being under) still holds.
        let (evicted_address, evicted_image) = hot_cache.pop_back().unwrap();
        spill_to_disk(spill, image_index, evicted_address, &evicted_image);
    }
    let _ = hot_cache.put_with_weight(address, img);
}

/// Writes `img` into a free range of the spill file and records its slot in
/// `image_index` - unless `image_index` already has a slot for `address`,
/// meaning it was written once before (e.g. loaded back into `hot_cache` by
/// `get`, then evicted again) and that copy is still current: the only way
/// an address's value changes is through `ImageCache::insert`, which always
/// drops the stale slot first.
fn spill_to_disk(
    spill: &Arc<SpillFile>,
    image_index: &mut HashMap<CacheAddress, Arc<SpillSlot>>,
    address: CacheAddress,
    img: &ImageContainer,
) {
    if image_index.contains_key(&address) {
        return;
    }
    let len = ImageCache::encoded_len(img);
    // Freed again (dropped) if the write fails.
    let slot = SpillSlot {
        file: Arc::clone(spill),
        offset: spill.allocate(len),
        len,
    };
    let start = Instant::now();
    let mut w = BufWriter::new(FileAt {
        file: &spill.file,
        position: slot.offset,
    });
    if let Err(e) = ImageCache::write_image(&mut w, img) {
        warn!(
            "Could not write image to the spill file at {}: {}",
            slot.offset, e
        );
        return;
    }
    drop(w);
    debug!(
        "Spilled image at {:?} ({} bytes) to the spill file at {} in {:?}",
        address,
        img.get_image_memory_usage(),
        slot.offset,
        start.elapsed()
    );
    image_index.insert(address, Arc::new(slot));
}

/// Reinterprets a pixel slice as raw bytes in host-native order, for a
/// direct bulk write with no per-element encoding. Sound for the only types
/// this is ever called with (`f32`, `u32`): both are `Copy`, have no
/// padding, and every bit pattern is valid, so viewing their bytes directly
/// can't produce an invalid value; `u8` has no alignment requirement.
fn as_bytes<T: Copy>(data: &[T]) -> &[u8] {
    // SAFETY: see doc comment above.
    unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, std::mem::size_of_val(data)) }
}

fn read_u64(r: &mut impl Read) -> io::Result<u64> {
    let mut buf = [0u8; 8];
    r.read_exact(&mut buf)?;
    Ok(u64::from_ne_bytes(buf))
}

fn read_i32(r: &mut impl Read) -> io::Result<i32> {
    let mut buf = [0u8; 4];
    r.read_exact(&mut buf)?;
    Ok(i32::from_ne_bytes(buf))
}

/// Reads `count` raw, host-native-order `T`s directly into a freshly
/// allocated `Vec<T>` - the read side of `as_bytes`, sound for the same
/// reason: the vec's own allocation is already correctly aligned for `T`
/// (we just allocated it as `Vec<T>`), and any bit pattern is a valid `T`
/// for the POD types (`f32`, `u32`) this is ever called with. `fill` is
/// only there to make the element type inferrable at the call site.
fn read_pod_vec<T: Copy>(r: &mut impl Read, count: usize, fill: T) -> io::Result<Vec<T>> {
    let mut data = vec![fill; count];
    // SAFETY: see doc comment above.
    let bytes = unsafe {
        std::slice::from_raw_parts_mut(
            data.as_mut_ptr() as *mut u8,
            count * std::mem::size_of::<T>(),
        )
    };
    r.read_exact(bytes)?;
    Ok(data)
}

#[cfg(test)]
mod tests {
    use super::*;
    use evanalyzer_cfg::core_types::MemoryId;

    fn gray_container(width: usize, height: usize, data: Vec<f32>) -> Arc<ImageContainer> {
        Arc::new(ImageContainer::F32Gray(ManagedImage {
            data: Image::<f32, 1>::new(ImageSize { width, height }, data).unwrap(),
            tile_offset: Point2d { x: 0, y: 0 },
            plane: None,
            dimensionality: ImageDimensionality::Planar,
        }))
    }

    /// Bypasses `new`'s fixed `DEFAULT_HOT_CACHE_CAPACITY_BYTES`, so eviction
    /// can be exercised without allocating gigabytes of test data.
    fn cache_with_capacity(bytes: usize) -> ImageCache {
        cache_in(bytes, None)
    }

    /// [`cache_with_capacity`] with its spill file in `dir`.
    fn cache_in(bytes: usize, dir: Option<&Path>) -> ImageCache {
        let capacity = NonZeroUsize::new(bytes).unwrap();
        ImageCache {
            inner: Mutex::new(ImageCacheInner {
                image_index: Default::default(),
                hot_cache: CLruCache::with_config(
                    CLruCacheConfig::new(capacity).with_scale(ImageWeight),
                ),
            }),
            spill: Arc::new(SpillFile::new(dir).unwrap()),
        }
    }

    /// How many images `cache` holds in its spill file.
    fn spilled(cache: &ImageCache) -> usize {
        cache.inner.lock().unwrap().image_index.len()
    }

    fn spill_file_len(cache: &ImageCache) -> u64 {
        cache.spill.file.metadata().unwrap().len()
    }

    /// A 4x4 gray image in the spill file: header + 64 bytes of pixels.
    const SPILLED_4X4: u64 = HEADER_BYTES + 64;

    #[test]
    fn insert_under_capacity_never_touches_disk() {
        let mut cache = cache_with_capacity(DEFAULT_HOT_CACHE_CAPACITY_BYTES);

        cache.insert(
            CacheAddress::Scratchpad,
            gray_container(2, 2, vec![1.0, 2.0, 3.0, 4.0]),
        );

        assert_eq!(spilled(&cache), 0);
        assert_eq!(cache.len(), 1);
        assert!(cache.contains_key(&CacheAddress::Scratchpad));

        let fetched = cache.get(&CacheAddress::Scratchpad).unwrap();
        assert_eq!(
            fetched.as_f32_slice(),
            Some([1.0, 2.0, 3.0, 4.0].as_slice())
        );
        // A `get` hit never needs to write anything either.
        assert_eq!(spilled(&cache), 0);
        assert_eq!(spill_file_len(&cache), 0);
    }

    #[test]
    fn eviction_spills_the_least_recently_used_entry_to_disk() {
        // Each 4x4 f32 gray image is 4*4*4 = 64 bytes; cap the cache at 100
        // bytes so a second insert can't fit alongside the first.
        let mut cache = cache_with_capacity(100);
        let first = CacheAddress::Memory(MemoryId::PipelineContext(1));
        let second = CacheAddress::Memory(MemoryId::PipelineContext(2));

        cache.insert(first, gray_container(4, 4, vec![1.0; 16]));
        assert_eq!(spilled(&cache), 0);

        cache.insert(second, gray_container(4, 4, vec![2.0; 16]));

        // `first` no longer fits alongside `second` - it was evicted and
        // spilled to disk. `second` (just inserted, most-recently-used)
        // stays resident, so only one image is spilled.
        assert_eq!(spilled(&cache), 1);
        assert_eq!(cache.len(), 2);

        // Still transparently retrievable, reloaded from the spill file.
        let fetched = cache.get(&first).unwrap();
        assert_eq!(fetched.as_f32_slice(), Some([1.0; 16].as_slice()));
    }

    #[test]
    fn a_smaller_capacity_spills_what_no_longer_fits() {
        let mut cache = cache_with_capacity(200);
        let first = CacheAddress::Memory(MemoryId::PipelineContext(1));
        let second = CacheAddress::Memory(MemoryId::PipelineContext(2));
        cache.insert(first, gray_container(4, 4, vec![1.0; 16]));
        cache.insert(second, gray_container(4, 4, vec![2.0; 16]));
        assert_eq!(spilled(&cache), 0);

        cache.set_capacity_bytes(100);

        // The least recently used went to disk, not lost.
        assert_eq!(spilled(&cache), 1);
        assert_eq!(cache.len(), 2);
        assert_eq!(
            cache.get(&first).unwrap().as_f32_slice(),
            Some([1.0; 16].as_slice())
        );
    }

    #[test]
    fn a_larger_capacity_keeps_more_in_memory() {
        let mut cache = cache_with_capacity(100);

        cache.set_capacity_bytes(200);
        cache.insert(
            CacheAddress::Memory(MemoryId::PipelineContext(1)),
            gray_container(4, 4, vec![1.0; 16]),
        );
        cache.insert(
            CacheAddress::Memory(MemoryId::PipelineContext(2)),
            gray_container(4, 4, vec![2.0; 16]),
        );

        assert_eq!(spilled(&cache), 0);
    }

    /// The whole slide's merge: every tile's cache is under its own cap,
    /// together they are over the merged cache's one cap - the rest spills.
    #[test]
    fn tiles_merged_into_one_cache_spill_over_its_one_capacity() {
        // As the job executor: the merged cache and every tile's cache are
        // copies of one empty cache.
        let global = cache_with_capacity(200);
        let mut merged = global.clone();
        merged.set_capacity_bytes(100);
        for id in 1..=2 {
            let mut tile = global.clone();
            tile.set_capacity_bytes(200);
            tile.insert(
                CacheAddress::Memory(MemoryId::PipelineContext(id)),
                gray_container(4, 4, vec![id as f32; 16]),
            );
            assert_eq!(spilled(&tile), 0, "a tile alone fits");
            merged.extend(tile);
        }

        assert_eq!(spilled(&merged), 1);
        assert_eq!(merged.len(), 2);
        let first = CacheAddress::Memory(MemoryId::PipelineContext(1));
        assert_eq!(
            merged.get(&first).unwrap().as_f32_slice(),
            Some([1.0; 16].as_slice())
        );
    }

    #[test]
    fn retain_removes_hot_only_entries_without_touching_disk() {
        let mut cache = cache_with_capacity(DEFAULT_HOT_CACHE_CAPACITY_BYTES);
        cache.insert(CacheAddress::Scratchpad, gray_container(1, 1, vec![9.0]));

        cache.retain(|_| false);

        assert!(cache.is_empty());
        assert_eq!(spill_file_len(&cache), 0);
    }

    #[test]
    fn insert_frees_the_stale_spill_of_the_address_it_replaces() {
        let mut cache = cache_with_capacity(100);
        let first = CacheAddress::Memory(MemoryId::PipelineContext(1));
        let second = CacheAddress::Memory(MemoryId::PipelineContext(2));

        // Force `first` to be spilled by inserting `second` right behind it.
        cache.insert(first, gray_container(4, 4, vec![1.0; 16]));
        cache.insert(second, gray_container(4, 4, vec![2.0; 16]));
        assert_eq!(spilled(&cache), 1);
        assert_eq!(spill_file_len(&cache), SPILLED_4X4);

        // Re-inserting `first` with a fresh value frees its old spill -
        // `second`, spilled to make room, reuses it: the file doesn't grow.
        cache.insert(first, gray_container(4, 4, vec![3.0; 16]));
        assert_eq!(spilled(&cache), 1);
        assert_eq!(spill_file_len(&cache), SPILLED_4X4);

        let fetched = cache.get(&first).unwrap();
        assert_eq!(fetched.as_f32_slice(), Some([3.0; 16].as_slice()));
        assert_eq!(
            cache.get(&second).unwrap().as_f32_slice(),
            Some([2.0; 16].as_slice())
        );
    }

    /// The spill file has no name: nothing in its folder, so nothing is left
    /// behind when the process is killed - the OS deletes it.
    #[test]
    fn spilled_images_leave_no_file_in_the_folder() {
        let dir = tempfile::tempdir().unwrap();
        let mut cache = cache_in(100, Some(dir.path()));
        let first = CacheAddress::Memory(MemoryId::PipelineContext(1));
        cache.insert(first, gray_container(4, 4, vec![1.0; 16]));
        cache.insert(
            CacheAddress::Memory(MemoryId::PipelineContext(2)),
            gray_container(4, 4, vec![2.0; 16]),
        );
        assert_eq!(spilled(&cache), 1);

        assert_eq!(std::fs::read_dir(dir.path()).unwrap().count(), 0);
        assert_eq!(
            cache.get(&first).unwrap().as_f32_slice(),
            Some([1.0; 16].as_slice())
        );
    }

    /// A copy shares the spilled image's slot: when the original drops the
    /// image and spills others, the copy still reads its own pixels - the
    /// range isn't reused while any copy holds it.
    #[test]
    fn a_copy_still_reads_a_spilled_image_the_original_dropped() {
        let mut original = cache_with_capacity(100);
        let first = CacheAddress::Memory(MemoryId::PipelineContext(1));
        original.insert(first, gray_container(4, 4, vec![1.0; 16]));
        original.insert(
            CacheAddress::Memory(MemoryId::PipelineContext(2)),
            gray_container(4, 4, vec![2.0; 16]),
        );
        let copy = original.clone();

        original.retain(|_| false);
        for id in 3..=5 {
            original.insert(
                CacheAddress::Memory(MemoryId::PipelineContext(id)),
                gray_container(4, 4, vec![id as f32; 16]),
            );
        }

        assert_eq!(
            copy.get(&first).unwrap().as_f32_slice(),
            Some([1.0; 16].as_slice())
        );
    }

    #[test]
    fn several_threads_spill_and_read_the_one_file_at_once() {
        let global = cache_with_capacity(100);
        std::thread::scope(|scope| {
            for id in 0..8u32 {
                let mut tile = global.clone();
                scope.spawn(move || {
                    for k in 0..4u32 {
                        tile.insert(
                            CacheAddress::Memory(MemoryId::PipelineContext(id * 10 + k)),
                            gray_container(4, 4, vec![(id * 10 + k) as f32; 16]),
                        );
                    }
                    for k in 0..4u32 {
                        let value = (id * 10 + k) as f32;
                        let address = CacheAddress::Memory(MemoryId::PipelineContext(id * 10 + k));
                        assert_eq!(
                            tile.get(&address).unwrap().as_f32_slice(),
                            Some([value; 16].as_slice())
                        );
                    }
                });
            }
        });
    }

    // ---- disk round trip of every image type ----

    fn round_trip(image: &ImageContainer) -> ImageContainer {
        let mut bytes = Vec::new();
        ImageCache::write_image(&mut bytes, image).unwrap();
        assert_eq!(bytes.len() as u64, ImageCache::encoded_len(image));
        ImageCache::read_image(&mut bytes.as_slice()).unwrap()
    }

    #[test]
    fn an_rgb_image_survives_the_disk_round_trip_with_offset_and_plane() {
        let data: Vec<f32> = (0..2 * 3 * 3).map(|v| v as f32 * 0.5).collect();
        let plane = Some(ImagePlane { z: 2, c: -1, t: 7 });
        let original = ImageContainer::F32Rgb(ManagedImage {
            data: Image::<f32, 3>::new(
                ImageSize {
                    width: 2,
                    height: 3,
                },
                data.clone(),
            )
            .unwrap(),
            tile_offset: Point2d { x: 640, y: 1280 },
            plane: plane.clone(),
            dimensionality: ImageDimensionality::Planar,
        });
        match round_trip(&original) {
            ImageContainer::F32Rgb(img) => {
                assert_eq!(
                    img.size(),
                    ImageSize {
                        width: 2,
                        height: 3
                    }
                );
                assert_eq!(img.as_slice(), data.as_slice());
                assert_eq!((img.tile_offset.x, img.tile_offset.y), (640, 1280));
                assert_eq!(img.plane, plane);
            }
            other => panic!("expected an RGB image, got {other:?}"),
        }
    }

    #[test]
    fn a_label_image_survives_the_disk_round_trip() {
        let labels = vec![0u32, 1, 4_000_000_000, 7];
        let original = ImageContainer::U32(ManagedImage {
            data: Image::<u32, 1>::new(
                ImageSize {
                    width: 2,
                    height: 2,
                },
                labels.clone(),
            )
            .unwrap(),
            tile_offset: Point2d { x: 3, y: 5 },
            plane: None,
            dimensionality: ImageDimensionality::Planar,
        });
        match round_trip(&original) {
            ImageContainer::U32(img) => {
                assert_eq!(img.as_slice(), labels.as_slice());
                assert_eq!((img.tile_offset.x, img.tile_offset.y), (3, 5));
                assert_eq!(img.plane, None);
            }
            other => panic!("expected a label image, got {other:?}"),
        }
    }

    #[test]
    fn a_damaged_or_unknown_spilled_image_is_a_cache_miss_not_a_crash() {
        let mut bytes = Vec::new();
        ImageCache::write_image(&mut bytes, &gray_container(8, 8, vec![1.0; 64])).unwrap();

        // Truncated: the header promises more pixels than there are.
        let truncated = &bytes[..bytes.len() / 2];
        assert!(ImageCache::read_image(&mut &truncated[..]).is_err());

        // Unknown type tag.
        let mut unknown = bytes.clone();
        unknown[0] = 99;
        let err = ImageCache::read_image(&mut unknown.as_slice()).unwrap_err();
        assert!(
            err.to_string().contains("unknown image cache tag 99"),
            "{err}"
        );

        // A slot past the file's end.
        let spill = Arc::new(SpillFile::new(None).unwrap());
        let slot = SpillSlot {
            file: Arc::clone(&spill),
            offset: 1000,
            len: SPILLED_4X4,
        };
        assert!(load_from_disk(&slot).is_none());
    }
}
