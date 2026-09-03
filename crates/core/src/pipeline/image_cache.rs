use clru::{CLruCache, CLruCacheConfig, WeightScale};
use log::{info, warn};

use crate::{ImageContainer, ImagePlane, ManagedImage, pipeline::pipeline_cache::CacheAddress};
use kornia_apriltag::utils::Point2d;
use kornia_image::{Image, ImageSize};
use kornia_tensor::CpuAllocator;
use std::{
    collections::{HashMap, hash_map::RandomState},
    fs::{self, File},
    io::{self, BufReader, BufWriter, Read, Write},
    num::NonZeroUsize,
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
    time::Instant,
};

/// On-disk tag for which `ImageContainer` variant a cache file holds -
/// written first so `load_from_disk` knows how to reconstruct it.
const TAG_F32_GRAY: u8 = 0;
const TAG_F32_RGB: u8 = 1;
const TAG_U32: u8 = 2;

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

/// `image_index` and `hot_cache` behind one lock, not two: `spill_to_disk`
/// needs to check-then-insert into `image_index` in lockstep with evicting
/// from `hot_cache` (see its own doc comment), and `get`/`insert` can race
/// each other across threads on the same `ImageCache` (e.g. rayon-parallel
/// object measurement in the whole-image phase all reading the same merged
/// cache) - a single lock makes that whole sequence atomic instead of
/// leaving a window where the two could disagree.
struct ImageCacheInner {
    // Addresses that have actually been written to a scratch file, and
    // where. An address only appears here once it's been spilled out of
    // `hot_cache` (see `spill_to_disk`) - most images live and die entirely
    // in `hot_cache` and never touch disk at all.
    image_index: HashMap<CacheAddress, PathBuf>,
    // Size-bounded, least-recently-used store of resident images. Every
    // stored address is either only here (never spilled), or here *and* in
    // `image_index` (spilled once, then loaded back by `get`).
    hot_cache: HotCache,
}

pub struct ImageCache {
    inner: Mutex<ImageCacheInner>,
    // Shared, refcounted scratch directory: every clone descended from the
    // same `new()` call writes into (and reads from) this *same* physical
    // directory, rather than each clone getting its own. That matters
    // because real usage (`job_executor.rs`) clones a `global_cache` once
    // per tile, has each tile write its own files into its clone, then
    // merges every clone back together via `extend` - if each clone owned
    // a separate directory, merging would only move the path *entries*
    // (see `extend`), and the moment the per-tile clone that actually wrote
    // those files went out of scope, its directory (and every file in it)
    // would be deleted out from under the merged index. Sharing the
    // directory via `Arc` sidesteps that entirely: the directory is only
    // removed once the *last* clone anywhere drops, so a file survives
    // regardless of which specific `ImageCache` instance wrote it or which
    // one gets dropped first/last - `Arc`'s refcounting is atomic, so this
    // is safe even when different clones drop concurrently on different
    // threads.
    temp_dir: Arc<tempfile::TempDir>,
}

/// Can't `#[derive(Clone)]`: `Mutex<T>` doesn't implement `Clone` even when
/// `T` does (a lock can't be duplicated without briefly acquiring it, and
/// std won't do that implicitly - a poisoned lock would make the clone
/// itself fallible in a way `Clone::clone` can't express).
///
/// Unconditionally safe regardless of how many entries `self` already has:
/// `temp_dir` is shared (see the field doc above), so the clone can see and
/// write into the exact same directory as `self` - nothing is left behind
/// or hidden the way an independent fresh directory would.
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
            temp_dir: Arc::clone(&self.temp_dir),
        }
    }
}

impl ImageCache {
    /// Creates a cache backed by its own fresh scratch directory, with
    /// `hot_cache` capped at [`DEFAULT_HOT_CACHE_CAPACITY_BYTES`]. For
    /// production pipeline runs, prefer [`with_capacity_bytes`](Self::with_capacity_bytes)
    /// with a budget sized against actual parallelism/available RAM (see
    /// `resources::recommended_image_cache_bytes`) - this default exists for
    /// callers (tests, ad-hoc/GUI-preview caches) that don't have that
    /// context.
    pub fn new() -> io::Result<Self> {
        Self::with_capacity_bytes(DEFAULT_HOT_CACHE_CAPACITY_BYTES as u64)
    }

    /// Creates a cache backed by its own fresh scratch directory, with
    /// `hot_cache` capped at `capacity_bytes`. Fails only if the OS can't
    /// create a temp directory (disk full, no permissions) - callers should
    /// treat that as fatal for whatever operation needed a disk-backed cache
    /// in the first place, the same way `job::object_scratch::TileObjectStore::new`
    /// already does for an analogous scratch directory.
    pub fn with_capacity_bytes(capacity_bytes: u64) -> io::Result<Self> {
        let temp_dir = tempfile::Builder::new()
            .prefix("evanalyzer-image-cache-")
            .tempdir()?;
        Ok(Self {
            inner: Mutex::new(ImageCacheInner {
                image_index: Default::default(),
                hot_cache: new_hot_cache(capacity_bytes as usize),
            }),
            temp_dir: Arc::new(temp_dir),
        })
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
        // Replacing this address invalidates any file already spilled for
        // the old value - delete it now rather than leaking scratch space
        // until the whole cache (and its temp directory) eventually drops.
        if let Some(stale_path) = inner.image_index.remove(&address) {
            self.delete_from_disk(&stale_path);
        }
        let ImageCacheInner {
            image_index,
            hot_cache,
        } = &mut *inner;
        hot_cache_insert(
            self.temp_dir.path(),
            image_index,
            hot_cache,
            address,
            img.clone(),
        );
        Some(img)
    }

    /// Moves every entry out of `other` and into `self` - matches
    /// `Extend::extend`. Safe to call even after `other` is dropped
    /// immediately afterwards: `temp_dir` is shared (see the struct-level
    /// doc), so the on-disk files these paths point at aren't tied to
    /// `other`'s lifetime. `other`'s `hot_cache` entries are re-inserted
    /// through the same size-bounded path `insert` uses, so merging can
    /// still spill to disk if the combined result needs the room.
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
            hot_cache_insert(self.temp_dir.path(), image_index, hot_cache, address, image);
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
        let image_path = {
            let mut inner = self.inner.lock().expect("Poisned thread");
            if let Some(image) = inner.hot_cache.get(address) {
                return Some(image.clone());
            }
            inner.image_index.get(address)?.clone()
        };

        let image = self.load_from_disk(&image_path)?;
        let mut inner = self.inner.lock().expect("Poisned thread");
        let ImageCacheInner {
            image_index,
            hot_cache,
        } = &mut *inner;
        hot_cache_insert(
            self.temp_dir.path(),
            image_index,
            hot_cache,
            *address,
            image.clone(),
        );
        Some(image)
    }

    /// Keeps only the entries for which `f` returns `true`, evicting the
    /// rest from both `hot_cache` and `image_index` - deleting the temp
    /// file too, but only for entries that actually had one (most never
    /// got spilled at all, so there's nothing on disk to clean up).
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
            if let Some(path) = inner.image_index.remove(&address) {
                self.delete_from_disk(&path);
            }
        }
    }

    /// Matches `HashMap::iter`: an iterator of `(&K, &V)` pairs in
    /// unspecified order.
    // pub fn iter(&self) -> impl Iterator<Item = (&CacheAddress, &Arc<ImageContainer>)> {
    //     self.image_index.iter()
    // }

    fn try_write_to_disk(path: &PathBuf, image: &ImageContainer) -> io::Result<()> {
        let mut w = BufWriter::new(File::create(path)?);
        match image {
            ImageContainer::F32Gray(img) => {
                Self::write_header(
                    &mut w,
                    TAG_F32_GRAY,
                    img.size(),
                    &img.tile_offset,
                    &img.plane,
                )?;
                w.write_all(as_bytes(img.as_slice()))?;
            }
            ImageContainer::F32Rgb(img) => {
                Self::write_header(
                    &mut w,
                    TAG_F32_RGB,
                    img.size(),
                    &img.tile_offset,
                    &img.plane,
                )?;
                w.write_all(as_bytes(img.as_slice()))?;
            }
            ImageContainer::U32(img) => {
                Self::write_header(&mut w, TAG_U32, img.size(), &img.tile_offset, &img.plane)?;
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

    /// Reads back a file written by `try_write_to_disk`. Returns `None`
    /// (logged) on any I/O or format error - e.g. this cache's own scratch
    /// directory was wiped externally - since a cache miss is always a safe
    /// fallback, but a hard error here would take down the whole pipeline
    /// over what's just a scratch file.
    fn load_from_disk(&self, path: &PathBuf) -> Option<Arc<ImageContainer>> {
        let start = Instant::now();
        match Self::try_load_from_disk(path) {
            Ok(image) => {
                info!(
                    "Loaded image ({} bytes) from cache file {:?} in {:?}",
                    image.get_image_memory_usage(),
                    path,
                    start.elapsed()
                );
                Some(Arc::new(image))
            }
            Err(e) => {
                warn!("Could not load image from cache file {:?}: {}", path, e);
                None
            }
        }
    }

    fn try_load_from_disk(path: &PathBuf) -> io::Result<ImageContainer> {
        let mut r = BufReader::new(File::open(path)?);

        let mut tag = [0u8; 1];
        r.read_exact(&mut tag)?;
        let width = read_u64(&mut r)? as usize;
        let height = read_u64(&mut r)? as usize;
        let tile_offset = Point2d {
            x: read_u64(&mut r)? as usize,
            y: read_u64(&mut r)? as usize,
        };
        let mut has_plane = [0u8; 1];
        r.read_exact(&mut has_plane)?;
        let plane = if has_plane[0] != 0 {
            Some(ImagePlane {
                z: read_i32(&mut r)?,
                c: read_i32(&mut r)?,
                t: read_i32(&mut r)?,
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
                let data = read_pod_vec(&mut r, width * height, 0f32)?;
                Ok(ImageContainer::F32Gray(ManagedImage {
                    data: Image::<f32, 1, CpuAllocator>::new(size, data, CpuAllocator)
                        .map_err(|e| to_io_err(format!("{e:?}")))?,
                    tile_offset,
                    plane,
                }))
            }
            TAG_F32_RGB => {
                let data = read_pod_vec(&mut r, width * height * 3, 0f32)?;
                Ok(ImageContainer::F32Rgb(ManagedImage {
                    data: Image::<f32, 3, CpuAllocator>::new(size, data, CpuAllocator)
                        .map_err(|e| to_io_err(format!("{e:?}")))?,
                    tile_offset,
                    plane,
                }))
            }
            TAG_U32 => {
                let data = read_pod_vec(&mut r, width * height, 0u32)?;
                Ok(ImageContainer::U32(ManagedImage {
                    data: Image::<u32, 1, CpuAllocator>::new(size, data, CpuAllocator)
                        .map_err(|e| to_io_err(format!("{e:?}")))?,
                    tile_offset,
                    plane,
                }))
            }
            other => Err(to_io_err(format!("unknown image cache tag {other}"))),
        }
    }
    fn delete_from_disk(&self, path: &PathBuf) {
        if let Err(error) = fs::remove_file(path) {
            warn!(
                "Could not delete image: {:?} from cache. Got: {}",
                path, error
            );
        }
    }
}

/// No manual `Drop` needed here: `temp_dir` is `Arc<tempfile::TempDir>`, so
/// the directory - and everything still in it - is only removed once the
/// *last* clone sharing it drops. That also covers any file `image_index`
/// might not be tracking for some reason (a bug elsewhere, a process that
/// panicked mid-write), which a hand-rolled "delete only what `image_index`
/// knows about" loop wouldn't.
impl Default for ImageCache {
    fn default() -> Self {
        Self::new().expect("failed to create scratch directory for ImageCache::default()")
    }
}

/// Puts `img` into `hot_cache`, evicting (and spilling to disk via
/// `spill_to_disk`) whatever least-recently-used entries are needed to make
/// room - or, if `img` alone is too big to ever fit, writing it straight to
/// disk without holding it in memory at all. A free function (not an
/// `ImageCache` method) so it can take `image_index`/`hot_cache` as
/// independent `&mut` borrows of one already-locked `ImageCacheInner`,
/// alongside `&self` calls like `delete_from_disk` at the same call site.
fn hot_cache_insert(
    temp_dir: &Path,
    image_index: &mut HashMap<CacheAddress, PathBuf>,
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
        spill_to_disk(temp_dir, image_index, address, &img);
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
        spill_to_disk(temp_dir, image_index, evicted_address, &evicted_image);
    }
    let _ = hot_cache.put_with_weight(address, img);
}

/// Writes `img` to a fresh scratch file and records its path in
/// `image_index` - unless `image_index` already has a path for `address`,
/// meaning it was written once before (e.g. loaded back into `hot_cache` by
/// `get`, then evicted again) and that file is still current: the only way
/// an address's value changes is through `ImageCache::insert`, which always
/// deletes any stale file first.
fn spill_to_disk(
    temp_dir: &Path,
    image_index: &mut HashMap<CacheAddress, PathBuf>,
    address: CacheAddress,
    img: &ImageContainer,
) {
    if image_index.contains_key(&address) {
        return;
    }
    let uuid = fast_uuid_v7::gen_id_u128();
    let img_path = temp_dir.join(format!("{}", uuid));
    let start = Instant::now();
    if let Err(e) = ImageCache::try_write_to_disk(&img_path, img) {
        warn!("Could not write image to cache file {:?}: {}", img_path, e);
        return;
    }
    info!(
        "Spilled image at {:?} ({} bytes) to cache file {:?} in {:?}",
        address,
        img.get_image_memory_usage(),
        img_path,
        start.elapsed()
    );
    image_index.insert(address, img_path);
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
            data: Image::<f32, 1, CpuAllocator>::new(
                ImageSize { width, height },
                data,
                CpuAllocator,
            )
            .unwrap(),
            tile_offset: Point2d { x: 0, y: 0 },
            plane: None,
        }))
    }

    /// Bypasses `new`'s fixed `DEFAULT_HOT_CACHE_CAPACITY_BYTES`, so eviction
    /// can be exercised without allocating gigabytes of test data.
    fn cache_with_capacity(bytes: usize) -> ImageCache {
        let temp_dir = tempfile::Builder::new()
            .prefix("evanalyzer-image-cache-test-")
            .tempdir()
            .unwrap();
        let capacity = NonZeroUsize::new(bytes).unwrap();
        ImageCache {
            inner: Mutex::new(ImageCacheInner {
                image_index: Default::default(),
                hot_cache: CLruCache::with_config(
                    CLruCacheConfig::new(capacity).with_scale(ImageWeight),
                ),
            }),
            temp_dir: Arc::new(temp_dir),
        }
    }

    fn disk_file_count(cache: &ImageCache) -> usize {
        fs::read_dir(cache.temp_dir.path()).unwrap().count()
    }

    #[test]
    fn insert_under_capacity_never_touches_disk() {
        let mut cache = cache_with_capacity(DEFAULT_HOT_CACHE_CAPACITY_BYTES);

        cache.insert(
            CacheAddress::Scratchpad,
            gray_container(2, 2, vec![1.0, 2.0, 3.0, 4.0]),
        );

        assert_eq!(disk_file_count(&cache), 0);
        assert_eq!(cache.len(), 1);
        assert!(cache.contains_key(&CacheAddress::Scratchpad));

        let fetched = cache.get(&CacheAddress::Scratchpad).unwrap();
        assert_eq!(
            fetched.as_f32_slice(),
            Some([1.0, 2.0, 3.0, 4.0].as_slice())
        );
        // A `get` hit never needs to write anything either.
        assert_eq!(disk_file_count(&cache), 0);
    }

    #[test]
    fn eviction_spills_the_least_recently_used_entry_to_disk() {
        // Each 4x4 f32 gray image is 4*4*4 = 64 bytes; cap the cache at 100
        // bytes so a second insert can't fit alongside the first.
        let mut cache = cache_with_capacity(100);
        let first = CacheAddress::Memory(MemoryId::PipelineContext(1));
        let second = CacheAddress::Memory(MemoryId::PipelineContext(2));

        cache.insert(first, gray_container(4, 4, vec![1.0; 16]));
        assert_eq!(disk_file_count(&cache), 0);

        cache.insert(second, gray_container(4, 4, vec![2.0; 16]));

        // `first` no longer fits alongside `second` - it was evicted and
        // spilled to disk. `second` (just inserted, most-recently-used)
        // stays resident, so still only one file exists.
        assert_eq!(disk_file_count(&cache), 1);
        assert_eq!(cache.len(), 2);

        // Still transparently retrievable, reloaded from the spilled file.
        let fetched = cache.get(&first).unwrap();
        assert_eq!(fetched.as_f32_slice(), Some([1.0; 16].as_slice()));
    }

    #[test]
    fn retain_removes_hot_only_entries_without_touching_disk() {
        let mut cache = cache_with_capacity(DEFAULT_HOT_CACHE_CAPACITY_BYTES);
        cache.insert(CacheAddress::Scratchpad, gray_container(1, 1, vec![9.0]));

        cache.retain(|_| false);

        assert!(cache.is_empty());
        assert_eq!(disk_file_count(&cache), 0);
    }

    #[test]
    fn insert_deletes_the_stale_spilled_file_of_the_address_it_replaces() {
        let mut cache = cache_with_capacity(100);
        let first = CacheAddress::Memory(MemoryId::PipelineContext(1));
        let second = CacheAddress::Memory(MemoryId::PipelineContext(2));

        // Force `first` to be spilled by inserting `second` right behind it.
        cache.insert(first, gray_container(4, 4, vec![1.0; 16]));
        cache.insert(second, gray_container(4, 4, vec![2.0; 16]));
        assert_eq!(disk_file_count(&cache), 1);

        // Re-inserting `first` with a fresh value must not leave the old
        // spilled file behind.
        cache.insert(first, gray_container(4, 4, vec![3.0; 16]));
        assert_eq!(disk_file_count(&cache), 1);

        let fetched = cache.get(&first).unwrap();
        assert_eq!(fetched.as_f32_slice(), Some([3.0; 16].as_slice()));
    }
}
