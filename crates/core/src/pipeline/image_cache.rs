use log::warn;

use crate::{ImageContainer, ImagePlane, ManagedImage, pipeline::pipeline_cache::CacheAddress};
use kornia_apriltag::utils::Point2d;
use kornia_image::{Image, ImageSize};
use kornia_tensor::CpuAllocator;
use std::{
    collections::HashMap,
    fs::{self, File},
    io::{self, BufReader, BufWriter, Read, Write},
    path::PathBuf,
    sync::{Arc, RwLock},
};

/// On-disk tag for which `ImageContainer` variant a cache file holds -
/// written first so `load_from_disk` knows how to reconstruct it.
const TAG_F32_GRAY: u8 = 0;
const TAG_F32_RGB: u8 = 1;
const TAG_U32: u8 = 2;

pub struct ImageCache {
    // Image index which holds all stored images and the file name under which it was stored
    image_index: HashMap<CacheAddress, PathBuf>,
    // Images in the hot cache which do not need to be loaded from disk right now
    hot_cache: RwLock<HashMap<CacheAddress, Arc<ImageContainer>>>,
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

/// Can't `#[derive(Clone)]`: `RwLock<T>` doesn't implement `Clone` even when
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
        Self {
            image_index: self.image_index.clone(),
            hot_cache: RwLock::new(self.hot_cache.read().expect("Poisned thread").clone()),
            temp_dir: Arc::clone(&self.temp_dir),
        }
    }
}

impl ImageCache {
    /// Creates a cache backed by its own fresh scratch directory. Fails
    /// only if the OS can't create a temp directory (disk full, no
    /// permissions) - callers should treat that as fatal for whatever
    /// operation needed a disk-backed cache in the first place, the same
    /// way `job::object_scratch::TileObjectStore::new` already does for an
    /// analogous scratch directory.
    pub fn new() -> io::Result<Self> {
        let temp_dir = tempfile::Builder::new()
            .prefix("evanalyzer-image-cache-")
            .tempdir()?;
        Ok(Self {
            image_index: Default::default(),
            hot_cache: Default::default(),
            temp_dir: Arc::new(temp_dir),
        })
    }

    /// Inserts `img` at `address`, returning the previously stored image (if
    /// any) - matches `HashMap::insert`.
    ///
    /// Writes through to disk immediately (so the entry can later be evicted
    /// from `hot_cache` without losing it - see `retain`/eviction) *and*
    /// caches it in `hot_cache` right away: without that, the very next
    /// `get()` for this address - typically moments later, e.g. a
    /// tile-scoped pipeline reading back the channel image it was just
    /// handed - would force an immediate disk round-trip for data that's
    /// still sitting right here in memory. `hot_cache` has no eviction
    /// policy yet (nothing bounds its size), so until one exists this means
    /// every inserted image does stay resident for the cache's lifetime,
    /// same as a plain in-memory `HashMap` would - disk-backing today only
    /// gets you the *ability* to evict-and-reload later, not automatic
    /// memory bounding.
    pub fn insert(
        &mut self,
        address: CacheAddress,
        img: Arc<ImageContainer>,
    ) -> Option<Arc<ImageContainer>> {
        let uuid = fast_uuid_v7::gen_id_u128();
        let img_path = self.generate_path(&uuid);
        self.write_to_disk(&img_path, img.clone());
        self.image_index.insert(address, img_path);
        //  self.hot_cache
        //      .write()
        //      .expect("Poisned thread")
        //      .insert(address, img.clone());
        Some(img)
    }

    /// Moves every entry out of `other` and into `self` - matches
    /// `Extend::extend`. Safe to call even after `other` is dropped
    /// immediately afterwards: `temp_dir` is shared (see the struct-level
    /// doc), so the on-disk files these paths point at aren't tied to
    /// `other`'s lifetime. Also merges `hot_cache`, so entries that were
    /// hot in `other` (e.g. a tile's just-processed channel images) stay
    /// hot in the merged cache instead of forcing a cold disk reload on the
    /// next `get`.
    pub fn extend(&mut self, mut other: ImageCache) {
        self.image_index
            .extend(std::mem::take(&mut other.image_index));
        if let Ok(other_hot) = other.hot_cache.into_inner() {
            self.hot_cache
                .write()
                .expect("Poisned thread")
                .extend(other_hot);
        }
    }

    /// Matches `HashMap::contains_key`: `&self`, key by reference.
    pub fn contains_key(&self, address: &CacheAddress) -> bool {
        self.image_index.contains_key(address)
    }

    /// Matches `HashMap::len`: `&self`.
    pub fn len(&self) -> usize {
        self.image_index.len()
    }

    /// Matches `HashMap::is_empty`: `&self`.
    pub fn is_empty(&self) -> bool {
        self.image_index.is_empty()
    }

    /// Matches `HashMap::keys`: every registered address, without loading
    /// any pixel data - cheap, since it only reads `image_index` (always
    /// in memory), never touches `hot_cache` or disk. Callers that only
    /// need to know *which* entries exist (e.g. to find the ones relevant
    /// to a query before deciding which to actually load) should use this
    /// instead of `get`-ing everything.
    pub fn keys(&self) -> impl Iterator<Item = &CacheAddress> {
        self.image_index.keys()
    }

    /// Returns the image at `address`, transparently loading it from disk
    /// and caching it in `hot_cache` on a miss - so a repeat `get` for the
    /// same address doesn't pay the disk read again.
    ///
    /// Returns an owned `Arc<ImageContainer>`, not `&Arc<ImageContainer>`:
    /// `hot_cache` is behind an `RwLock` for thread-safety, and a `&self`
    /// method can't return a reference that outlives the `RwLockReadGuard`
    /// borrowed to produce it - that guard is a local value, dropped at the
    /// end of this call. Cloning the `Arc` (a cheap refcount bump, not a
    /// data copy - ~10ns regardless of image size) is the standard way
    /// around that, and lets the lock be held only for the lookup itself.
    pub fn get(&self, address: &CacheAddress) -> Option<Arc<ImageContainer>> {
        if let Some(image) = self.hot_cache.read().expect("Poisned thread").get(address) {
            return Some(image.clone());
        }

        let image_path = self.image_index.get(address)?.clone();
        let image = self.load_from_disk(&image_path)?;
        // self.hot_cache
        //     .write()
        //     .expect("Poisned thread")
        //     .insert(*address, image.clone());
        Some(image)
    }

    /// Keeps only the entries for which `f` returns `true`, evicting the
    /// rest from both the hot cache and the on-disk index - and deleting
    /// their temp file, so a dropped entry doesn't leak `temp_folder`
    /// space forever.
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
        let to_remove: Vec<CacheAddress> = self
            .image_index
            .keys()
            .filter(|address| !f(address))
            .copied()
            .collect();

        for address in to_remove {
            self.hot_cache
                .write()
                .expect("Poisned thread")
                .remove(&address);
            if let Some(path) = self.image_index.remove(&address) {
                self.delete_from_disk(&path);
            }
        }
    }

    /// Matches `HashMap::iter`: an iterator of `(&K, &V)` pairs in
    /// unspecified order.
    // pub fn iter(&self) -> impl Iterator<Item = (&CacheAddress, &Arc<ImageContainer>)> {
    //     self.image_index.iter()
    // }

    /// Writes `image` to `path` as raw bytes: a small fixed header (variant,
    /// size, tile offset, plane) followed by the pixel buffer verbatim, in
    /// host-native byte order - no compression, no per-element encoding.
    /// This is a scratch file for the current process only (deleted on
    /// eviction or `Drop`, see `delete_from_disk`), never read by another
    /// run or another machine, so there's no reason to pay for a
    /// self-describing or portable format. Failure just leaves this
    /// address unwritten - logged, not propagated, since a slow/full disk
    /// shouldn't take down the whole pipeline over a cache write.
    fn write_to_disk(&self, path: &PathBuf, image: Arc<ImageContainer>) {
        if let Err(e) = Self::try_write_to_disk(path, &image) {
            warn!("Could not write image to cache file {:?}: {}", path, e);
        }
    }

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

    /// Reads back a file written by `write_to_disk`. Returns `None` (logged)
    /// on any I/O or format error - e.g. this cache's own `temp_folder` was
    /// wiped externally - since a cache miss is always a safe fallback, but
    /// a hard error here would take down the whole pipeline over what's
    /// just a scratch file.
    fn load_from_disk(&self, path: &PathBuf) -> Option<Arc<ImageContainer>> {
        match Self::try_load_from_disk(path) {
            Ok(image) => Some(Arc::new(image)),
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

    fn generate_path(&self, name: &u128) -> PathBuf {
        self.temp_dir.path().join(format!("{}", name))
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
