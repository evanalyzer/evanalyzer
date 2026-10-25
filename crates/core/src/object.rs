//! # object
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-06-05
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.

use crate::ImageDimensionality;
use crate::ImagePlane;
use crate::ImageTile;
use crate::pipeline::pipeline_cache::GlobalPipelineCache;
use crate::pipeline::scope::Scope;
use bitvec::prelude::*;
use evanalyzer_cfg::core_types::InternalErrors;
use evanalyzer_cfg::{
    core_types::{ObjectClass, ObjectId, SegmentationClass, TrackId},
    settings::object_settings::{ObjectMetricSettings, TrackSettings},
};
use indexmap::IndexMap;
use kornia_image::ImageSize;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;

/// An object's (or annotation's) intensities over its mask - one
/// implementation for the pipeline and stored annotations.
pub mod intensities;
pub mod regions;
/// Shape measurements computed on demand (Feret, convex area, Euler number,
/// radii).
pub mod shape;

/// An object's intensities in one channel. How the values spread (std to
/// MAD, as CellProfiler's MeasureObjectIntensity - see
/// [`Intensity::set_spread`]) is NaN where the pixel values weren't at hand
/// (a stored annotation's intensities) - the results show no value then.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Intensity {
    /// Sum of all pixel intensities in the object
    pub sum_intensity: f64,
    /// Minimum pixel intensity in the object
    pub min_intensity: f64,
    /// Maximum pixel intensity in the object
    pub max_intensity: f64,
    /// Average pixel intensity in the object
    pub avg_intensity: f64,
    /// Standard deviation
    pub std_intensity: f64,
    /// Median intensity
    pub median_intensity: f64,
    pub lower_quartile_intensity: f64,
    pub upper_quartile_intensity: f64,
    /// Median absolute deviation: the median of the distances from the median.
    pub mad_intensity: f64,
}

/// Zero sums, the spread not measured (NaN).
impl Default for Intensity {
    fn default() -> Self {
        Self {
            sum_intensity: 0.0,
            min_intensity: 0.0,
            max_intensity: 0.0,
            avg_intensity: 0.0,
            std_intensity: f64::NAN,
            median_intensity: f64::NAN,
            lower_quartile_intensity: f64::NAN,
            upper_quartile_intensity: f64::NAN,
            mad_intensity: f64::NAN,
        }
    }
}

#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct Track {
    pub id: TrackId,
    pub object_ids: Vec<ObjectId>,     // Ordered list of ROIs over time
    pub parent_track: Option<TrackId>, // If created by division
}

#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct Object {
    // Global unique object ID
    pub id: ObjectId,

    // Semantic class after threshold
    pub segmentation_class: SegmentationClass,

    // Dedicated class after classify object.
    object_class: HashSet<ObjectClass>,

    /// The scope the object lives in
    scope: Scope,

    // Colocalization
    pub colocalized_with: IndexMap<ObjectClass, Vec<ObjectId>>,

    /// The objects this one belongs to, at most one per class: parent
    /// class -> parent object. Only Relate Objects links (see
    /// [`set_parent`](Self::set_parent)); its children are the objects
    /// pointing here, so they aren't stored.
    parents: IndexMap<ObjectClass, ObjectId>,

    // Tracking
    pub track: Track,

    // Are size
    pub area: usize,

    // Bounding box
    pub bbox: [u32; 4], // x_min, y_min, x_max, y_max

    pub mask_data: BitVec<u64, Lsb0>, // The mask data (relative to BBox)

    // True if it touches the edge
    pub touches_edge: bool,

    // Accumulators for moments & intensity
    pub sum_x: u64,
    pub sum_y: u64,
    pub sum_x2: u64,
    pub sum_y2: u64,
    pub sum_xy: u64,

    // Intensities
    pub intensities: IndexMap<i32, Intensity>, // Intensity values for each image channel

    // Image plane information
    pub plane: ImagePlane,

    /// One plane (`plane.z`) or a volume of `depth` planes from `plane.z` on -
    /// then `bbox` is the box of every plane, `mask_data` covers them one
    /// after the other (z-major) and `area` counts the voxels.
    pub dimensionality: ImageDimensionality,

    /// z moments of a volume, z counted from `plane.z` (0 for one plane).
    pub sum_z: u64,
    pub sum_z2: u64,
    pub sum_xz: u64,
    pub sum_yz: u64,

    /// The tile this object was extracted from (by `ExtractObjects`)
    pub source_tile: ImageTile,

    // --- Precomputed geometry metrics ---
    // Filled by `finalize_geometry()` at object creation time — which runs on the
    // parallel extraction/segmentation workers — and read back through
    // `get_perimeter()` / `get_ellipse()`. The perimeter is an O(bbox area)
    // boundary walk; computing it here keeps it off the single-threaded DB writer,
    // where a lazy computation would stall every other tile's insert. Both derive
    // purely from the immutable mask geometry (mask_data, bbox, area, moments),
    // which never changes after extraction, so the stored values stay valid for
    // the object's lifetime. Default to 0 / empty until finalize_geometry runs.
    //
    // These are intentionally private (no `pub`): keeping them module-private makes
    // a `Object { .. }` struct literal illegal outside this module, which forces all
    // external construction through [`Object::new`] — the one path that is guaranteed
    // to call [`finalize_geometry`](Self::finalize_geometry). Callers read them via
    // `get_perimeter()` / `get_ellipse()` / `get_centroid()`.
    perimeter: f32,
    ellipse: FittingEllipse,
    centroid: (f32, f32),
}

/// Caller-settable fields for building a [`Object`] via [`Object::new`].
///
/// Mirrors every field of [`Object`] except the derived geometry metrics
/// (`perimeter`, `ellipse`), which [`Object::new`] computes for you. Build it with
/// struct-update syntax and pass it to [`Object::new`]:
///
/// ```ignore
/// let object = Object::new(ObjectInit { id, bbox, mask_data, area, ..Default::default() });
/// ```
#[derive(Debug, Default, Clone)]
pub struct ObjectInit {
    pub id: ObjectId,
    pub segmentation_class: SegmentationClass,
    pub object_class: HashSet<ObjectClass>,
    pub scope: Scope,
    pub colocalized_with: IndexMap<ObjectClass, Vec<ObjectId>>,
    /// Kept links of an object rebuilt from another (a stored object, a
    /// transformed one) - new links only come from Relate Objects.
    pub parents: IndexMap<ObjectClass, ObjectId>,
    pub track: Track,
    pub area: usize,
    pub bbox: [u32; 4],
    pub mask_data: BitVec<u64, Lsb0>,
    pub touches_edge: bool,
    pub sum_x: u64,
    pub sum_y: u64,
    pub sum_x2: u64,
    pub sum_y2: u64,
    pub sum_xy: u64,
    pub intensities: IndexMap<i32, Intensity>,
    pub plane: ImagePlane,
    pub dimensionality: ImageDimensionality,
    pub sum_z: u64,
    pub sum_z2: u64,
    pub sum_xz: u64,
    pub sum_yz: u64,
    pub source_tile: ImageTile,
}

#[derive(Debug, Default, Clone, Copy, Serialize, Deserialize)]
pub struct FittingEllipse {
    /// The length of the longest diameter (2a).
    /// ImageJ refers to this as 'Major'.
    pub major: f32,

    /// The length of the shortest diameter (2b).
    /// ImageJ refers to this as 'Minor'.
    pub minor: f32,

    /// The angle of the major axis relative to the x-axis.
    /// Typically stored in radians (-π/2 to π/2) or degrees (0-180).
    pub angle: f32,

    /// Optional: How 'squashed' the ellipse is.
    /// Calculated as sqrt(1 - (minor^2 / major^2)).
    pub eccentricity: f32,
}

/// A boolean set operation between two object masks, used by [`Object::combine_geometry`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BooleanOp {
    /// Intersection: pixels present in both operands.
    And,
    /// Union: pixels present in either operand.
    Or,
    /// Symmetric difference: pixels present in exactly one operand.
    Xor,
    /// Set difference: pixels present in the first operand but not the second.
    Subtract,
}

/// Rasterized mask geometry produced by a shape transform (see [`Object::scaled_geometry`],
/// [`Object::circle_geometry`], [`Object::fitting_ellipse_geometry`]).
///
/// Carries only the geometric fields of a [`Object`] — identity, classes, lineage and plane are
/// deliberately left out, since callers decide whether the transformed shape replaces the
/// source object in place or becomes a brand-new object, and need to set those fields themselves
/// via [`ObjectInit`].
#[derive(Debug, Default, Clone)]
pub struct TransformedGeometry {
    pub bbox: [u32; 4],
    pub mask_data: BitVec<u64, Lsb0>,
    pub area: usize,
    pub touches_edge: bool,
    pub sum_x: u64,
    pub sum_y: u64,
    pub sum_x2: u64,
    pub sum_y2: u64,
    pub sum_xy: u64,
}

impl Object {
    /// Builds a fully-finalized object from [`ObjectInit`].
    ///
    /// This is the only public way to construct a [`Object`]: the geometry metrics
    /// (`perimeter`, `ellipse`) are computed here via
    /// [`finalize_geometry`](Self::finalize_geometry), so they can never be left
    /// uncomputed by a forgotten call.
    ///
    /// For ROIs assembled incrementally (e.g. pixel-by-pixel extraction where the
    /// mask and moments are filled in after construction), call
    /// [`finalize_geometry`](Self::finalize_geometry) again once accumulation is
    /// complete — the eager finalize here only reflects the geometry present in
    /// `init`.
    pub fn new(init: ObjectInit) -> Self {
        let mut object = Object {
            id: init.id,
            segmentation_class: init.segmentation_class,
            object_class: init.object_class,
            scope: init.scope,
            colocalized_with: init.colocalized_with,
            parents: init.parents,
            track: init.track,
            area: init.area,
            bbox: init.bbox,
            mask_data: init.mask_data,
            touches_edge: init.touches_edge,
            sum_x: init.sum_x,
            sum_y: init.sum_y,
            sum_x2: init.sum_x2,
            sum_y2: init.sum_y2,
            sum_xy: init.sum_xy,
            intensities: init.intensities,
            plane: init.plane,
            dimensionality: init.dimensionality,
            sum_z: init.sum_z,
            sum_z2: init.sum_z2,
            sum_xz: init.sum_xz,
            sum_yz: init.sum_yz,
            source_tile: init.source_tile,
            perimeter: 0.0,
            ellipse: FittingEllipse::default(),
            centroid: (0.0, 0.0),
        };
        object.finalize_geometry();
        object
    }

    /// An order by where the object lies: its first pixel in reading order
    /// (top to bottom, left to right - the order scipy/scikit-image number
    /// labels, so CellProfiler's "lower label" ties come out the same), then
    /// box, area and mask. The same in every run, unlike the ids (random
    /// within a millisecond). For tie-breaks: two objects differ here unless
    /// they cover the very same pixels.
    pub fn position_cmp(&self, other: &Self) -> std::cmp::Ordering {
        let first = |o: &Self| {
            (
                o.bbox[1],
                o.bbox[0] as usize + o.mask_data.first_one().unwrap_or(0),
            )
        };
        (first(self), self.bbox, self.area)
            .cmp(&(first(other), other.bbox, other.area))
            .then_with(|| self.mask_data.cmp(&other.mask_data))
    }

    /// Checks if a global coordinate (x, y) is within the object's mask.
    pub fn is_part_of(&self, x: u32, y: u32) -> bool {
        let [x_min, y_min, x_max, y_max] = self.bbox;

        // bbox[2]/[3] are INCLUSIVE - the last pixel coordinate inside the object.
        if x < x_min || x > x_max || y < y_min || y > y_max {
            return false;
        }

        let local_x = (x - x_min) as usize;
        let local_y = (y - y_min) as usize;
        let width = (x_max - x_min + 1) as usize;

        // Calculate index in the BitVec (Row-major order assumed)
        let bit_index = (local_y * width) + local_x;

        // Access the mask bit
        self.mask_data.get(bit_index).map(|b| *b).unwrap_or(false)
    }

    /// Adds `object_class`, as named by a command running in `scope`. An
    /// object without classes takes the class's scope (so reclassifying -
    /// clear, then add - moves it); adding a class of another scope than the
    /// object's is refused - an object never mixes local and global classes.
    pub fn add_object_class(
        &mut self,
        object_class: ObjectClass,
        scope: Scope,
    ) -> Result<(), InternalErrors> {
        if object_class == ObjectClass::Unset {
            return Ok(());
        }
        let class_scope = scope.of_class(&object_class);
        if self.object_class.is_empty() {
            self.scope = class_scope;
        } else if self.scope != class_scope {
            return Err(InternalErrors::InvalidArgument(format!(
                "object {} can't get class {} - an object can't have a module's \
                 local classes and project classes at the same time",
                self.id,
                object_class.to_text()
            )));
        }
        self.object_class.insert(object_class);
        Ok(())
    }

    /// Adds an object to the object which colocalzis with it
    ///
    /// The class is the object class of the other object which was used to
    /// calc the colocalization.
    pub fn add_colocalizing_object(&mut self, coloc_class: ObjectClass, object_id: ObjectId) {
        let coloc_per_class = self.colocalized_with.entry(coloc_class).or_default();
        coloc_per_class.push(object_id);
        coloc_per_class.sort();
        coloc_per_class.dedup();
    }

    /// The objects this one belongs to: parent class -> parent object.
    pub fn parents(&self) -> &IndexMap<ObjectClass, ObjectId> {
        &self.parents
    }

    /// The parent of class `class`, if linked.
    pub fn parent(&self, class: &ObjectClass) -> Option<&ObjectId> {
        self.parents.get(class)
    }

    /// Links this object to `parent` as its parent of class `class`
    /// (`None`: unlinks it). Crate-only: Relate Objects is the one step that
    /// links objects.
    pub(crate) fn set_parent(&mut self, class: ObjectClass, parent: Option<ObjectId>) {
        match parent {
            Some(parent) => {
                self.parents.insert(class, parent);
            }
            None => {
                self.parents.shift_remove(&class);
            }
        }
    }

    /// Keeps only the links `keep` picks (e.g. drops those to removed
    /// objects).
    pub(crate) fn retain_parents(&mut self, mut keep: impl FnMut(&ObjectClass, &ObjectId) -> bool) {
        self.parents.retain(|class, parent| keep(class, parent));
    }

    /// Whether the object is of `object_class`, as named by a command
    /// running in `scope`: a local class matches only objects of that scope.
    pub fn is_of(&self, object_class: &ObjectClass, scope: Scope) -> bool {
        self.scope == scope.of_class(object_class) && self.object_class.contains(object_class)
    }

    /// Whether the object is of at least one of `object_classes` - how every
    /// "only objects of these classes" list of a pipeline command selects
    /// its objects. See [`Self::is_of`].
    pub fn is_of_any(&self, object_classes: &[ObjectClass], scope: Scope) -> bool {
        object_classes.iter().any(|class| self.is_of(class, scope))
    }

    /// Removes `object_class`, as named by a command running in `scope` (a
    /// local class of another scope isn't this object's).
    pub fn remove_object_class(&mut self, object_class: &ObjectClass, scope: Scope) {
        if self.is_of(object_class, scope) {
            self.object_class.remove(object_class);
        }
    }

    /// Removes every class. The object keeps its scope until it gets a
    /// class again.
    pub fn clear_object_classes(&mut self) {
        self.object_class.clear();
    }

    /// Whether a command running in `scope` sees the object at all: the
    /// project's objects, and those of its own scope - what "all objects"
    /// means for a command (e.g. an empty class filter).
    pub fn is_visible_in(&self, scope: Scope) -> bool {
        self.scope == Scope::Global || self.scope == scope
    }

    /// The object's classes - local ones only mean something together with
    /// [`Self::scope`].
    pub fn object_classes(&self) -> &HashSet<ObjectClass> {
        &self.object_class
    }

    /// The scope the object lives in.
    pub fn scope(&self) -> Scope {
        self.scope
    }

    /// Computes and stores the geometry metrics (`perimeter`, `ellipse`) from the
    /// current mask. Call once after the mask/moments are fully assembled — i.e.
    /// at object creation, on the parallel extraction workers — so later reads
    /// (classification, DB export) are free field accesses. The metrics depend
    /// only on the immutable geometry, so a single call is enough for the object's life.
    ///
    /// `pub(crate)` on purpose: one-shot construction should go through [`Object::new`],
    /// which calls this for you. It stays reachable inside the crate for the
    /// incremental extraction path, which must re-finalize after the mask and
    /// moments are fully accumulated.
    pub(crate) fn finalize_geometry(&mut self) {
        self.perimeter = self.compute_perimeter();
        self.ellipse = self.compute_ellipse();
        self.centroid = self.compute_centroid();
    }

    /// Returns the fitted ellipse (major/minor axes, angle, eccentricity)
    /// precomputed by [`finalize_geometry`](Self::finalize_geometry).
    pub fn get_ellipse(&self) -> FittingEllipse {
        self.ellipse
    }

    fn compute_ellipse(&self) -> FittingEllipse {
        let n = self.area as f64;
        if n == 0.0 {
            return FittingEllipse::default();
        }

        let xc = self.sum_x as f64 / n;
        let yc = self.sum_y as f64 / n;

        let mu20 = (self.sum_x2 as f64 / n) - xc.powi(2);
        let mu02 = (self.sum_y2 as f64 / n) - yc.powi(2);
        let mu11 = (self.sum_xy as f64 / n) - (xc * yc);

        let common = ((mu20 - mu02).powi(2) + 4.0 * mu11.powi(2)).sqrt();

        let major = (8.0 * (mu20 + mu02 + common)).sqrt() as f32;
        let minor = (8.0 * (mu20 + mu02 - common)).sqrt() as f32;

        // Calculate Eccentricity
        // We use .max(0.0) to prevent NaN from tiny floating point inaccuracies
        let eccentricity = if major > 0.0 {
            (1.0 - (minor.powi(2) / major.powi(2))).max(0.0).sqrt()
        } else {
            0.0
        };

        FittingEllipse {
            major,
            minor,
            angle: 0.5 * (2.0 * mu11).atan2(mu20 - mu02) as f32,
            eccentricity,
        }
    }

    pub fn circularity(&self) -> f32 {
        let perimeter = self.get_perimeter();
        // Same guard as get_roundness() below, which computes the identical
        // formula - without it, a zero-perimeter object (only possible
        // alongside zero area) would divide by zero and write NaN/inf into
        // CSV export and ML training feature vectors instead of erroring.
        if perimeter > 0.0 {
            (4.0 * std::f32::consts::PI * self.area as f32) / (perimeter * perimeter)
        } else {
            0.0
        }
    }

    /// Returns the object perimeter in pixels, precomputed by
    /// [`finalize_geometry`](Self::finalize_geometry).
    ///
    /// The underlying [`compute_perimeter`](Self::compute_perimeter) is an
    /// O(bbox area) boundary walk — by far the most expensive object metric — which
    /// is why it is computed once at creation rather than on demand.
    pub fn get_perimeter(&self) -> f32 {
        self.perimeter
    }

    /// Calculates the perimeter of the object using ImageJ's actual algorithm:
    /// an 8-connected boundary trace (`ij.gui.Wand.traceEdge`, the same one
    /// `ParticleAnalyzer` uses to build its outline polygon), followed by
    /// `ij.gui.PolygonRoi.getTracedPerimeter`'s corner-corrected length
    /// formula.
    ///
    /// A prior version of this function instead summed each foreground
    /// pixel's background-facing edges (orthogonal neighbors weighted 1.0,
    /// diagonal weighted √2) - that measures something close to *twice* the
    /// true boundary length for any non-trivial shape (verified against a
    /// from-scratch Rust port of the real Wand/PolygonRoi algorithm above,
    /// cross-checked by hand-tracing a single pixel and a 2×2 square): a
    /// circle of radius 80px came out with perimeter ≈962px (old formula)
    /// vs. ≈529px (ImageJ) against an ideal 2πr≈503px, i.e. circularity
    /// ≈0.27 instead of ImageJ's ≈0.90 for a shape that's visibly round.
    ///
    /// # Returns
    /// The perimeter value in pixels. A perfect circle with radius r has
    /// perimeter ≈ 2πr, though ImageJ's own correction formula systematically
    /// undercorrects small/jagged shapes - a single isolated pixel comes out
    /// as 2√2 (≈2.83), not the geometrically "obvious" 4. This isn't a bug
    /// here; it's ImageJ's real, documented behavior, faithfully reproduced
    /// for parity with results measured in ImageJ itself.
    fn compute_perimeter(&self) -> f32 {
        let [x_min, y_min, x_max, y_max] = self.bbox;
        if self.area == 0 || x_max < x_min || y_max < y_min {
            return 0.0;
        }

        // Mask stride uses the INCLUSIVE bbox convention (width = x_max - x_min + 1),
        // matching how the mask is built everywhere else.
        let width = (x_max - x_min + 1) as usize;
        let height = (y_max - y_min + 1) as usize;
        if width == 0 || height == 0 {
            return 0.0;
        }

        // Materialize the mask into a bool grid padded with a 1-pixel false border,
        // so the boundary trace below never needs a bounds check: any vertex
        // coordinate the trace can reach maps to a real (possibly-border) grid cell.
        let pw = width + 2;
        let ph = height + 2;
        let mut grid = vec![false; pw * ph];
        for y in 0..height {
            let src = y * width;
            let dst = (y + 1) * pw + 1;
            for x in 0..width {
                if self.mask_data.get(src + x).map(|b| *b).unwrap_or(false) {
                    grid[dst + x] = true;
                }
            }
        }

        let inside = |x: i64, y: i64| -> bool {
            if x < 0 || y < 0 || x as usize >= pw || y as usize >= ph {
                false
            } else {
                grid[y as usize * pw + x as usize]
            }
        };
        // Pixel at vertex (x, y) in a given walking `direction`, matching
        // `ij.gui.Wand`'s vertex/direction convention exactly: 0 -> pixel
        // (x, y); 1 -> pixel (x, y-1); 2 -> pixel (x-1, y-1); 3 -> pixel (x-1, y).
        let inside_dir = |x: i64, y: i64, direction: i64| -> bool {
            match direction.rem_euclid(4) {
                0 => inside(x, y),
                1 => inside(x, y - 1),
                2 => inside(x - 1, y - 1),
                _ => inside(x - 1, y),
            }
        };

        // Start at the first foreground pixel in row-major grid order - always
        // exists since `self.area > 0` was already checked above.
        let Some(start_idx) = grid.iter().position(|&b| b) else {
            return 0.0;
        };
        let start_x = (start_idx % pw) as i64;
        let start_y = (start_idx / pw) as i64;

        // `inside(start_x, start_y)` is always true (it's the foreground pixel
        // we just found), so this always takes the `1` branch - ImageJ's other
        // branch only matters when starting from a caller-supplied point that
        // might be outside, which never happens here.
        let start_direction: i64 = 1;
        let mut x = start_x;
        let mut y = start_y;
        let mut direction = start_direction;

        let mut xp: Vec<i64> = Vec::new();
        let mut yp: Vec<i64> = Vec::new();
        loop {
            // Try turning left first (direction + 1), then straight, then
            // right - the first of those with an inside pixel wins. Mirrors
            // `Wand.traceEdge`'s 8-connected search exactly, including that
            // the last candidate is taken on faith without being re-tested:
            // if neither "left" nor "straight" has an inside pixel, "right"
            // is topologically guaranteed to (we're standing on a real
            // boundary edge), so ImageJ doesn't bother checking it either.
            let mut new_direction = direction + 1;
            loop {
                if inside_dir(x, y, new_direction) {
                    break;
                }
                new_direction -= 1;
                if new_direction < direction {
                    break;
                }
            }
            if new_direction != direction {
                xp.push(x);
                yp.push(y);
            }
            match new_direction.rem_euclid(4) {
                0 => x += 1,
                1 => y -= 1,
                2 => x -= 1,
                _ => y += 1,
            }
            direction = new_direction;
            if x == start_x && y == start_y && direction.rem_euclid(4) == start_direction {
                break;
            }
        }

        let n = xp.len();
        if n < 4 {
            return 0.0;
        }

        // `ij.gui.PolygonRoi.getTracedPerimeter`: walk the traced polygon,
        // summing the horizontal/vertical run lengths and counting corners,
        // then apply the correction that turns a naive (overestimated) crack-
        // boundary length into ImageJ's actual reported perimeter.
        let mut sumdx = 0i64;
        let mut sumdy = 0i64;
        let mut n_corners = 0i64;
        let mut dx1 = xp[0] - xp[n - 1];
        let mut dy1 = yp[0] - yp[n - 1];
        let mut side1 = dx1.abs() + dy1.abs();
        let mut corner = false;
        for i in 0..n {
            let next_i = (i + 1) % n;
            let dx2 = xp[next_i] - xp[i];
            let dy2 = yp[next_i] - yp[i];
            sumdx += dx1.abs();
            sumdy += dy1.abs();
            let side2 = dx2.abs() + dy2.abs();
            if side1 > 1 || !corner {
                corner = true;
                n_corners += 1;
            } else {
                corner = false;
            }
            dx1 = dx2;
            dy1 = dy2;
            side1 = side2;
        }

        (sumdx + sumdy) as f32 - n_corners as f32 * (2.0 - std::f32::consts::SQRT_2)
    }

    /// Calculates solidity: pixel area divided by the area of the convex hull of the
    /// mask. Values range from 0 to 1 (1 = perfectly convex; lower means more concave
    /// boundaries, bays or holes).
    ///
    /// The hull is taken over the *corners* of the mask's pixel squares, so the pixel
    /// union is always contained in its hull and solidity never exceeds 1. Only the
    /// leftmost/rightmost pixel of each row can be a hull vertex, so we feed just those
    /// extreme corners to the hull — O(height) hull points instead of O(area).
    pub fn get_solidity(&self) -> f32 {
        let [x_min, y_min, x_max, y_max] = self.bbox;
        if self.area == 0 || x_max < x_min || y_max < y_min {
            return 0.0;
        }
        let width = (x_max - x_min + 1) as usize;
        let height = (y_max - y_min + 1) as usize;

        // Per row, collect the outer corners of the leftmost and rightmost set pixels.
        let mut points: Vec<(f64, f64)> = Vec::with_capacity(height * 4);
        for ry in 0..height {
            let row = ry * width;
            let mut left: Option<usize> = None;
            let mut right = 0usize;
            for rx in 0..width {
                if self.mask_data.get(row + rx).map(|b| *b).unwrap_or(false) {
                    if left.is_none() {
                        left = Some(rx);
                    }
                    right = rx;
                }
            }
            if let Some(left) = left {
                let (y0, y1) = (ry as f64, (ry + 1) as f64);
                points.push((left as f64, y0));
                points.push((left as f64, y1));
                points.push(((right + 1) as f64, y0));
                points.push(((right + 1) as f64, y1));
            }
        }

        let hull_area = convex_hull_area(&mut points);
        if hull_area <= 0.0 {
            // Degenerate (single pixel / thin line); the pixel union is its own hull.
            return 1.0;
        }
        (self.area as f32 / hull_area as f32).min(1.0)
    }

    /// Calculates aspect ratio: major axis / minor axis.
    pub fn get_aspect_ratio(&self) -> f32 {
        let ellipse = self.get_ellipse();
        if ellipse.minor > 0.0 {
            ellipse.major / ellipse.minor
        } else {
            1.0
        }
    }

    /// Calculates roundness: 4π × Area / Perimeter².
    pub fn get_roundness(&self, perimeter: f32) -> f32 {
        if perimeter > 0.0 {
            (4.0 * std::f32::consts::PI * self.area as f32) / (perimeter * perimeter)
        } else {
            0.0
        }
    }

    /// Calculates compactness: Perimeter² / Area.
    pub fn get_compactness(&self, perimeter: f32) -> f32 {
        if self.area > 0 {
            (perimeter * perimeter) / self.area as f32
        } else {
            0.0
        }
    }

    /// The centroid: the mean position of the object's pixels (its centre
    /// of mass, not weighted by intensity) - as CellProfiler's and Fiji's.
    /// The bounding box's centre for an object without pixels. Precomputed
    /// by [`finalize_geometry`](Self::finalize_geometry).
    pub fn get_centroid(&self) -> (f32, f32) {
        self.centroid
    }

    /// The mean pixel position, from the mask a word at a time: the sum of
    /// the set bits' positions (six popcounts per 64-bit word) and of their
    /// rows (one count per row) give `Σx = Σposition - width·Σy` - exact,
    /// without visiting each pixel.
    fn compute_centroid(&self) -> (f32, f32) {
        let [x_min, y_min, x_max, y_max] = self.bbox;
        let width = (x_max - x_min + 1) as usize;
        let height = (y_max - y_min + 1) as usize;
        let mask = &self.mask_data[..self.mask_data.len().min(width * height)];

        // Bits whose position (0..64) has bit k set, for k = 0..6.
        const BIT_OF_POSITION: [u64; 6] = [
            0xAAAA_AAAA_AAAA_AAAA,
            0xCCCC_CCCC_CCCC_CCCC,
            0xF0F0_F0F0_F0F0_F0F0,
            0xFF00_FF00_FF00_FF00,
            0xFFFF_0000_FFFF_0000,
            0xFFFF_FFFF_0000_0000,
        ];
        let (mut count, mut sum_position) = (0u64, 0u64);
        for (i, chunk) in mask.chunks(64).enumerate() {
            let word: u64 = chunk.load_le();
            let ones = word.count_ones() as u64;
            let within: u64 = BIT_OF_POSITION
                .iter()
                .enumerate()
                .map(|(k, m)| ((word & m).count_ones() as u64) << k)
                .sum();
            count += ones;
            sum_position += ones * 64 * i as u64 + within;
        }
        if count == 0 {
            return self.get_bbox_center();
        }
        let sum_y: u64 = mask
            .chunks(width)
            .enumerate()
            .map(|(y, row)| y as u64 * row.count_ones() as u64)
            .sum();
        let sum_x = sum_position - width as u64 * sum_y;
        (
            (x_min as f64 + sum_x as f64 / count as f64) as f32,
            (y_min as f64 + sum_y as f64 / count as f64) as f32,
        )
    }

    /// The centre of the bounding box - the centroid only for symmetric
    /// shapes.
    pub fn get_bbox_center(&self) -> (f32, f32) {
        let [x_min, y_min, x_max, y_max] = self.bbox;
        let x_center = (x_min as f32 + x_max as f32) / 2.0;
        let y_center = (y_min as f32 + y_max as f32) / 2.0;
        (x_center, y_center)
    }

    /// Returns Feret diameter (longest distance between boundary points):
    /// the largest distance between two pixel centres, as CellProfiler's
    /// MaxFeretDiameter (see [`Object::feret_diameters`]). Was the bounding
    /// box's diagonal until 2026-10-24.
    pub fn get_feret_diameter(&self) -> f32 {
        self.feret_diameters().1 as f32
    }

    /// Returns minimum Feret diameter: the smallest caliper width over the
    /// pixel centres, as CellProfiler's MinFeretDiameter. Was the fitted
    /// ellipse's minor axis until 2026-10-24.
    pub fn get_min_feret_diameter(&self) -> f32 {
        self.feret_diameters().0 as f32
    }

    pub fn to_object_settings(&self) -> ObjectMetricSettings {
        ObjectMetricSettings {
            id: self.id.clone(),
            segmentation_class: self.segmentation_class,
            object_class: self.object_class.clone(),
            // A freshly extracted/measured `Object` has no notion of
            // training exclusion - that's a manual decision made later on
            // the already-persisted `ObjectMetricSettings`, in the AI
            // training dialog, not something re-derived here.
            exclude_from_training: false,
            colocalized_with: self.colocalized_with.clone(),
            parents: self.parents.clone(),
            track: TrackSettings {
                id: self.track.id.clone(),
                object_ids: self.track.object_ids.clone(),
                parent_track: self.track.parent_track.clone(),
            },
            area: self.area,
            bbox: self.bbox,
            mask_data: self.mask_data.clone(),
            touches_edge: self.touches_edge,
            sum_x: self.sum_x,
            sum_y: self.sum_y,
            sum_x2: self.sum_x2,
            sum_y2: self.sum_y2,
            sum_xy: self.sum_xy,
            intensities: self
                .intensities
                .iter()
                .map(|(k, v)| (*k, v.to_settings()))
                .collect(),
            perimeter: self.perimeter,
            z_stack: self.plane.z,
            c_stack: self.plane.c,
            t_stack: self.plane.t,
            dimensionality: self.dimensionality,
            sum_z: self.sum_z,
            sum_z2: self.sum_z2,
            sum_xz: self.sum_xz,
            sum_yz: self.sum_yz,
        }
    }

    /// A manually annotated object from its shape alone (no pixels to
    /// measure - e.g. an imported annotation): area, moments, edge flag and
    /// outline from the set pixels of `mask` (relative to `bbox` = `[x1, y1,
    /// x2, y2]`, inclusive), no intensities. On `plane` of an image of
    /// `full_image_size`.
    pub fn from_mask_geometry(
        full_image_size: &ImageSize,
        mask: BitVec<u64, Lsb0>,
        bbox: [u32; 4],
        plane: ImagePlane,
        object_class: HashSet<ObjectClass>,
    ) -> Self {
        let mut object = Object::new(ObjectInit {
            id: ObjectId::next(),
            segmentation_class: SegmentationClass::MANUAL_ANNOTATED,
            object_class,
            intensities: IndexMap::new(),
            bbox,
            mask_data: mask,
            ..Default::default()
        });
        object.plane = plane;
        let width = (bbox[2] - bbox[0] + 1) as usize;
        for (i, set) in object.mask_data.clone().iter().by_vals().enumerate() {
            if !set {
                continue;
            }
            let x = bbox[0] as usize + i % width;
            let y = bbox[1] as usize + i / width;
            object.area += 1;
            object.sum_x += x as u64;
            object.sum_y += y as u64;
            object.sum_x2 += (x * x) as u64;
            object.sum_y2 += (y * y) as u64;
            object.sum_xy += (x * y) as u64;
            if x == 0 || y == 0 || x + 1 >= full_image_size.width || y + 1 >= full_image_size.height
            {
                object.touches_edge = true;
            }
        }
        object.finalize_geometry();
        object
    }

    pub fn from_object_settings(s: ObjectMetricSettings) -> Self {
        Object::new(ObjectInit {
            id: s.id,
            segmentation_class: s.segmentation_class,
            object_class: s.object_class,
            // Stored and annotated objects only have project classes.
            scope: Scope::Global,
            colocalized_with: s.colocalized_with,
            parents: s.parents,
            track: Track {
                id: s.track.id,
                object_ids: s.track.object_ids,
                parent_track: s.track.parent_track,
            },
            area: s.area,
            bbox: s.bbox,
            mask_data: s.mask_data.clone(),
            touches_edge: s.touches_edge,
            sum_x: s.sum_x,
            sum_y: s.sum_y,
            sum_x2: s.sum_x2,
            sum_y2: s.sum_y2,
            sum_xy: s.sum_xy,
            intensities: s
                .intensities
                .into_iter()
                .map(|(k, v)| (k, Intensity::from_settings(&v)))
                .collect(),
            plane: ImagePlane {
                z: s.z_stack,
                c: s.c_stack,
                t: s.t_stack,
            },
            dimensionality: s.dimensionality,
            sum_z: s.sum_z,
            sum_z2: s.sum_z2,
            sum_xz: s.sum_xz,
            sum_yz: s.sum_yz,
            source_tile: ImageTile::default(),
        })
    }

    /// How many pixels this object shares with `other` - the area of
    /// [`overlaps`](Self::overlaps), without building the intersection.
    pub fn overlap_area(&self, other: &Self) -> usize {
        let [s_xmin, s_ymin, s_xmax, _] = self.bbox;
        let [o_xmin, o_ymin, o_xmax, _] = other.bbox;
        let (x_min, y_min) = (s_xmin.max(o_xmin), s_ymin.max(o_ymin));
        let (x_max, y_max) = (s_xmax.min(o_xmax), self.bbox[3].min(other.bbox[3]));
        // bbox[2]/[3] are INCLUSIVE.
        if x_min > x_max || y_min > y_max {
            return 0;
        }
        let s_width = (s_xmax - s_xmin + 1) as usize;
        let o_width = (o_xmax - o_xmin + 1) as usize;
        let mut area = 0;
        for y in y_min..=y_max {
            let s_row = (y - s_ymin) as usize * s_width;
            let o_row = (y - o_ymin) as usize * o_width;
            for x in x_min..=x_max {
                let s = self.mask_data.get(s_row + (x - s_xmin) as usize);
                let o = other.mask_data.get(o_row + (x - o_xmin) as usize);
                if s.is_some_and(|b| *b) && o.is_some_and(|b| *b) {
                    area += 1;
                }
            }
        }
        area
    }

    /// Computes the intersection with another object.
    /// Returns `Some(Object)` representing the overlapping region, or `None` if there is no overlap.
    pub fn overlaps(&self, other: &Self) -> Option<Self> {
        // 1. Extract and find the overlap of the two bounding boxes
        let [s_xmin, s_ymin, s_xmax, s_ymax] = self.bbox;
        let [o_xmin, o_ymin, o_xmax, o_ymax] = other.bbox;

        let overlap_xmin = s_xmin.max(o_xmin);
        let overlap_ymin = s_ymin.max(o_ymin);
        let overlap_xmax = s_xmax.min(o_xmax);
        let overlap_ymax = s_ymax.min(o_ymax);

        // bbox[2]/[3] are INCLUSIVE. Two inclusive ranges overlap only when min <= other's max.
        if overlap_xmin > overlap_xmax || overlap_ymin > overlap_ymax {
            return None;
        }

        // Inclusive extents: width = xmax - xmin + 1
        let overlap_width = (overlap_xmax - overlap_xmin + 1) as usize;
        let overlap_height = (overlap_ymax - overlap_ymin + 1) as usize;

        let mut overlap_mask = BitVec::<u64, Lsb0>::repeat(false, overlap_width * overlap_height);

        let s_width = (s_xmax - s_xmin + 1) as usize;
        let o_width = (o_xmax - o_xmin + 1) as usize;

        let mut area = 0;
        let mut sum_x = 0u64;
        let mut sum_y = 0u64;
        let mut sum_x2 = 0u64;
        let mut sum_y2 = 0u64;
        let mut sum_xy = 0u64;

        // 2. Scan only inside the intersecting bounding box window using global coordinates
        for y in overlap_ymin..=overlap_ymax {
            for x in overlap_xmin..=overlap_xmax {
                // Map the global (x,y) index back to local coordinates for both self and other masks
                let s_local_x = (x - s_xmin) as usize;
                let s_local_y = (y - s_ymin) as usize;
                let s_bit_idx = (s_local_y * s_width) + s_local_x;

                let o_local_x = (x - o_xmin) as usize;
                let o_local_y = (y - o_ymin) as usize;
                let o_bit_idx = (o_local_y * o_width) + o_local_x;

                // Evaluate masks safely
                let s_active = self.mask_data.get(s_bit_idx).map(|b| *b).unwrap_or(false);
                let o_active = other.mask_data.get(o_bit_idx).map(|b| *b).unwrap_or(false);

                if s_active && o_active {
                    // Map the global coordinate to our new, localized intersection mask coordinate
                    let overlap_local_x = (x - overlap_xmin) as usize;
                    let overlap_local_y = (y - overlap_ymin) as usize;
                    let overlap_bit_idx = (overlap_local_y * overlap_width) + overlap_local_x;

                    overlap_mask.set(overlap_bit_idx, true);

                    // Update area and geometry components dynamically
                    area += 1;
                    let x_u64 = x as u64;
                    let y_u64 = y as u64;

                    sum_x += x_u64;
                    sum_y += y_u64;
                    sum_x2 += x_u64 * x_u64;
                    sum_y2 += y_u64 * y_u64;
                    sum_xy += x_u64 * y_u64;
                }
            }
        }

        // If masks did not share any active matching foreground pixels, return None
        if area == 0 {
            return None;
        }

        // 3. Assemble and return the brand-new structural intersection object
        Some(Object::new(ObjectInit {
            id: ObjectId::next(), // Typically initialized empty or generated by your core engine later
            segmentation_class: self.segmentation_class.clone(),
            area,
            bbox: [overlap_xmin, overlap_ymin, overlap_xmax, overlap_ymax],
            mask_data: overlap_mask,
            touches_edge: self.touches_edge || other.touches_edge, // Inherits edge vulnerability flags
            sum_x,
            sum_y,
            sum_x2,
            sum_y2,
            sum_xy,
            plane: self.plane.clone(),
            ..Default::default()
        }))
    }

    /// Samples per-channel pixel intensities for this object's mask against the channel images
    /// held in `cache`. For ROIs assembled by [`ExtractObjects`](crate::algos::classification::extract_objects::ExtractObjects)
    /// this is redundant (it already measures intensities in the same pass that builds the
    /// mask). It exists for ROIs synthesized later in the pipeline — e.g. the intersection
    /// ROIs colocalization creates via [`overlaps`](Self::overlaps), Voronoi regions, or an
    /// object rebuilt by `TransformObjects`/`ObjectMath` — which have a valid mask/bbox but
    /// never had a chance to sample pixel data.
    ///
    /// `cache.image_cache` can hold channel data from *more than one* source tile at once
    /// (the whole-image phase merges every tile's own channel views together), and this
    /// object's mask may itself span more than one of them - e.g. a `TileMerge`d object, or
    /// a Voronoi region seeded near a tile boundary. So each mask pixel is matched against
    /// whichever loaded tile actually covers it, per channel, rather than assuming a single
    /// "current" tile the way a `Tile`-scoped command's own `ctx` would. A pixel not covered
    /// by any loaded tile for a given channel (e.g. a tile that was never processed, as
    /// happens for an off-screen tile in a partial preview run) is simply skipped for that
    /// channel - not an error, just missing data.
    pub fn measure_intensities(&self, cache: &GlobalPipelineCache) -> IndexMap<i32, Intensity> {
        // Only resolves (and, if needed, disk-loads - see `ImageCache`'s own doc
        // comments) the tiles whose bounds actually overlap this object's bbox,
        // not every channel entry the cache happens to be holding: once the
        // whole-image phase merges every tile's channel views together, this
        // object's mask - all `measure_intensities` ever samples - typically
        // only touches a handful of those tiles.
        let channel_views = cache.resolve_channel_views_for_bbox(self.bbox);
        // In `resolve_channel_views_for_bbox`'s order: the result's channel
        // order follows it (existing callers/tests rely on `IndexMap` order).
        let tiles: Vec<intensities::ChannelTile<'_>> = channel_views
            .iter()
            .filter_map(|((channel, tile), _, container)| {
                intensities::ChannelTile::from_container(*channel, *tile, container)
            })
            .collect();
        let mut acc = intensities::IntensityAccumulator::default();
        intensities::MaskIntensities::add_intensities(self, &tiles, &mut acc);
        acc.finish()
    }

    /// Measures every object of `objects` (into its `intensities`) as
    /// [`Self::measure_intensities`] does one by one, but tile by tile: each
    /// channel tile is fetched from the cache once - not once per object,
    /// which on a whole slide read the same tiles from disk again and again -
    /// and the objects in parallel. An object inside one tile is measured
    /// while that tile is loaded; one over several keeps its sums (bounded:
    /// gray values are counted) until its last tile.
    pub fn measure_intensities_of(objects: &mut [Object], cache: &GlobalPipelineCache) {
        use rayon::prelude::*;
        let mut keys: Vec<(i32, ImageTile)> = cache.channel_keys().collect();
        keys.sort_by_key(|(channel, t)| (t.offset_y, t.offset_x, *channel));
        let tiles: Vec<ImageTile> = keys.chunk_by(|a, b| a.1 == b.1).map(|c| c[0].1).collect();
        // The tiles in rows (the same offset_y), to find an object's quickly.
        let rows: Vec<&[ImageTile]> = tiles.chunk_by(|a, b| a.offset_y == b.offset_y).collect();
        let row_starts: Vec<usize> = rows
            .iter()
            .scan(0, |start, row| {
                let s = *start;
                *start += row.len();
                Some(s)
            })
            .collect();
        // Each object's tiles (indices into `tiles`, in order).
        let object_tiles: Vec<Vec<usize>> = objects
            .par_iter()
            .map(|o| {
                let [x0, y0, x1, y1] = o.bbox.map(|v| v as usize);
                let mut found = Vec::new();
                for (row, start) in rows.iter().zip(&row_starts) {
                    let t = row[0];
                    if t.offset_y > y1 || y0 >= t.offset_y + t.height {
                        continue;
                    }
                    for (k, t) in row.iter().enumerate() {
                        if t.offset_x <= x1 && x0 < t.offset_x + t.width {
                            found.push(start + k);
                        }
                    }
                }
                found
            })
            .collect();
        let mut by_tile: Vec<Vec<usize>> = vec![Vec::new(); tiles.len()];
        for (i, found) in object_tiles.iter().enumerate() {
            for &t in found {
                by_tile[t].push(i);
            }
        }

        let mut results: Vec<IndexMap<i32, Intensity>> = vec![IndexMap::new(); objects.len()];
        // Objects over several tiles: one accumulator per channel, so a huge
        // object's channels are added in parallel too.
        let mut channels: Vec<i32> = keys.iter().map(|(channel, _)| *channel).collect();
        channels.sort_unstable();
        channels.dedup();
        let mut spanning: Vec<(usize, i32, intensities::IntensityAccumulator)> = object_tiles
            .iter()
            .enumerate()
            .filter(|(_, found)| found.len() > 1)
            .flat_map(|(i, _)| {
                channels
                    .iter()
                    .map(move |&c| (i, c, intensities::IntensityAccumulator::default()))
            })
            .collect();
        let objects_ref: &[Object] = objects;
        for (t, (tile, members)) in tiles.iter().zip(&by_tile).enumerate() {
            if members.is_empty() {
                continue;
            }
            let containers: Vec<(i32, std::sync::Arc<crate::ImageContainer>)> = keys
                .iter()
                .filter(|(_, k)| k == tile)
                .filter_map(|key| {
                    let address = crate::pipeline::pipeline_cache::CacheAddress::Channel(*key);
                    Some((key.0, cache.image_cache.get(&address)?))
                })
                .collect();
            let channel_tiles: Vec<intensities::ChannelTile<'_>> = containers
                .iter()
                .filter_map(|(channel, image)| {
                    intensities::ChannelTile::from_container(*channel, *tile, image)
                })
                .collect();
            let measured: Vec<(usize, IndexMap<i32, Intensity>)> = members
                .par_iter()
                .filter(|&&i| object_tiles[i].len() == 1)
                .map(|&i| {
                    let mut acc = intensities::IntensityAccumulator::default();
                    intensities::MaskIntensities::add_intensities(
                        &objects_ref[i],
                        &channel_tiles,
                        &mut acc,
                    );
                    (i, acc.finish())
                })
                .collect();
            for (i, intensities) in measured {
                results[i] = intensities;
            }
            spanning
                .par_iter_mut()
                .filter(|(i, ..)| object_tiles[*i].contains(&t))
                .for_each(|(i, channel, acc)| {
                    let own: Vec<intensities::ChannelTile<'_>> = channel_tiles
                        .iter()
                        .filter(|c| c.channel == *channel)
                        .copied()
                        .collect();
                    intensities::MaskIntensities::add_intensities(&objects_ref[*i], &own, acc)
                });
            // Done after their last tile: finished, their sums freed.
            let (done, rest): (Vec<_>, Vec<_>) = spanning
                .into_iter()
                .partition(|(i, ..)| object_tiles[*i].last() == Some(&t));
            spanning = rest;
            let finished: Vec<(usize, IndexMap<i32, Intensity>)> = done
                .into_par_iter()
                .map(|(i, _, acc)| (i, acc.finish()))
                .collect();
            // In channel order (as `done` is): the same order as an object's
            // inside one tile.
            for (i, intensities) in finished {
                results[i].extend(intensities);
            }
        }
        for (object, intensities) in objects.iter_mut().zip(results) {
            object.intensities = intensities;
        }
    }

    /// Builds the rasterized geometry of `self` scaled by `(scale_x, scale_y)` around its
    /// bounding-box center.
    ///
    /// Each target pixel is back-mapped through the inverse scale and tested against the
    /// *original* mask via [`is_part_of`](Self::is_part_of), so irregular/concave shapes scale
    /// faithfully instead of just growing or shrinking the bounding box.
    pub fn scaled_geometry(
        &self,
        scale_x: f32,
        scale_y: f32,
        image_size: ImageSize,
    ) -> TransformedGeometry {
        let (cx, cy) = self.get_bbox_center();
        let [x_min, y_min, x_max, y_max] = self.bbox;
        let project = |v: f32, c: f32, s: f32| c + (v - c) * s;
        let bbox_f = [
            project(x_min as f32, cx, scale_x),
            project(y_min as f32, cy, scale_y),
            project(x_max as f32, cx, scale_x),
            project(y_max as f32, cy, scale_y),
        ];

        rasterize_geometry(bbox_f, image_size, self.touches_edge, |gx, gy| {
            if scale_x == 0.0 || scale_y == 0.0 {
                return false;
            }
            let src_x = cx + (gx - cx) / scale_x;
            let src_y = cy + (gy - cy) / scale_y;
            if src_x < 0.0 || src_y < 0.0 {
                return false;
            }
            self.is_part_of(src_x as u32, src_y as u32)
        })
    }

    /// Builds the rasterized geometry of a filled circle with the given `diameter`, centered
    /// on `self`'s bounding-box center.
    pub fn circle_geometry(&self, diameter: f32, image_size: ImageSize) -> TransformedGeometry {
        let (cx, cy) = self.get_bbox_center();
        // A radius of 0 would rasterize to an empty mask; floor it to a single pixel.
        let radius = (diameter / 2.0).max(0.5);
        let bbox_f = [cx - radius, cy - radius, cx + radius, cy + radius];

        rasterize_geometry(bbox_f, image_size, self.touches_edge, |gx, gy| {
            let dx = gx - cx;
            let dy = gy - cy;
            dx * dx + dy * dy <= radius * radius
        })
    }

    /// Builds the rasterized geometry of the ellipse fitted to `self`'s mask moments (see
    /// [`get_ellipse`](Self::get_ellipse)), scaled by `scale` and centered on `self`'s
    /// bounding-box center.
    pub fn fitting_ellipse_geometry(
        &self,
        scale: f32,
        image_size: ImageSize,
    ) -> TransformedGeometry {
        let (cx, cy) = self.get_bbox_center();
        let ellipse = self.get_ellipse();
        // Degenerate ROIs (e.g. a single pixel) have zero moments; floor the semi-axes to a
        // single pixel so they still rasterize to something instead of vanishing.
        let semi_major = (ellipse.major / 2.0 * scale).max(0.5);
        let semi_minor = (ellipse.minor / 2.0 * scale).max(0.5);
        let (sin_a, cos_a) = ellipse.angle.sin_cos();

        // Axis-aligned bounding box of a rotated ellipse.
        let half_w = ((semi_major * cos_a).powi(2) + (semi_minor * sin_a).powi(2)).sqrt();
        let half_h = ((semi_major * sin_a).powi(2) + (semi_minor * cos_a).powi(2)).sqrt();
        let bbox_f = [cx - half_w, cy - half_h, cx + half_w, cy + half_h];

        rasterize_geometry(bbox_f, image_size, self.touches_edge, |gx, gy| {
            let dx = gx - cx;
            let dy = gy - cy;
            let rx = dx * cos_a + dy * sin_a;
            let ry = -dx * sin_a + dy * cos_a;
            (rx / semi_major).powi(2) + (ry / semi_minor).powi(2) <= 1.0
        })
    }

    /// Boolean set operation between `self` ("A") and the union of `others` ("B"):
    /// intersection (AND), union (OR), symmetric difference (XOR), or set difference
    /// (Subtract, A \ B). Used for object-pair "object math" - unlike a pixel-level
    /// binary-image operation, this keeps the result tied to exactly the input object(s)
    /// it came from, and only ever touches the
    /// small bbox window spanned by the operands instead of rasterizing the whole
    /// tile/image.
    ///
    /// AND and Subtract can only ever remove pixels from `self`, so the result stays
    /// within `self`'s own bbox. OR and XOR can extend into `others`' territory too,
    /// so the working window covers the union of every operand's bbox.
    pub fn combine_geometry(
        &self,
        others: &[&Object],
        op: BooleanOp,
        image_size: ImageSize,
    ) -> TransformedGeometry {
        let [mut x_min, mut y_min, mut x_max, mut y_max] = self.bbox;
        let grows_beyond_self = matches!(op, BooleanOp::Or | BooleanOp::Xor);
        if grows_beyond_self {
            for o in others {
                let [oxmin, oymin, oxmax, oymax] = o.bbox;
                x_min = x_min.min(oxmin);
                y_min = y_min.min(oymin);
                x_max = x_max.max(oxmax);
                y_max = y_max.max(oymax);
            }
        }
        let bbox_f = [x_min as f32, y_min as f32, x_max as f32, y_max as f32];
        let base_touches_edge =
            self.touches_edge || (grows_beyond_self && others.iter().any(|o| o.touches_edge));

        rasterize_geometry(bbox_f, image_size, base_touches_edge, |gx, gy| {
            let (px, py) = (gx as u32, gy as u32);
            let in_a = self.is_part_of(px, py);
            let in_b = others.iter().any(|o| o.is_part_of(px, py));
            match op {
                BooleanOp::And => in_a && in_b,
                BooleanOp::Or => in_a || in_b,
                BooleanOp::Xor => in_a != in_b,
                BooleanOp::Subtract => in_a && !in_b,
            }
        })
    }

    /// Grows `self`'s mask outward by `margin` pixels using a disk-shaped structuring
    /// element - the standard flat dilation used by most bio-image tools (e.g.
    /// ImageJ's "Enlarge"): a pixel is included if it lies within `margin` of any
    /// pixel already in the mask.
    pub fn dilated_geometry(&self, margin: f32, image_size: ImageSize) -> TransformedGeometry {
        let margin = margin.max(0.0);
        let [x_min, y_min, x_max, y_max] = self.bbox;
        let bbox_f = [
            x_min as f32 - margin,
            y_min as f32 - margin,
            x_max as f32 + margin,
            y_max as f32 + margin,
        ];
        rasterize_geometry(bbox_f, image_size, self.touches_edge, |gx, gy| {
            disk_hits_mask(
                self, gx as i64, gy as i64, margin, /* require_all */ false,
            )
        })
    }

    /// Shrinks `self`'s mask inward by `margin` pixels using a disk-shaped structuring
    /// element - the standard flat erosion: a pixel survives only if every pixel
    /// within `margin` of it is also part of the original mask.
    pub fn eroded_geometry(&self, margin: f32, image_size: ImageSize) -> TransformedGeometry {
        let margin = margin.max(0.0);
        let [x_min, y_min, x_max, y_max] = self.bbox;
        // Erosion only ever shrinks - the result can't extend past the original bbox.
        let bbox_f = [x_min as f32, y_min as f32, x_max as f32, y_max as f32];
        rasterize_geometry(bbox_f, image_size, self.touches_edge, |gx, gy| {
            disk_hits_mask(
                self, gx as i64, gy as i64, margin, /* require_all */ true,
            )
        })
    }
}

/// Tests the disk of radius `margin` centered at pixel `(px, py)` against `object`'s mask.
///
/// `require_all = false` (dilation): true if *any* pixel in the disk is part of the mask.
/// `require_all = true` (erosion): true only if *every* pixel in the disk is part of the mask.
fn disk_hits_mask(object: &Object, px: i64, py: i64, margin: f32, require_all: bool) -> bool {
    let r = margin.ceil() as i64;
    let margin_sq = margin * margin;
    for dy in -r..=r {
        for dx in -r..=r {
            if (dx * dx + dy * dy) as f32 > margin_sq {
                continue;
            }
            let (nx, ny) = (px + dx, py + dy);
            let inside = nx >= 0 && ny >= 0 && object.is_part_of(nx as u32, ny as u32);
            if require_all {
                if !inside {
                    return false;
                }
            } else if inside {
                return true;
            }
        }
    }
    require_all
}

/// Rasterizes `inside_shape` (a predicate over continuous pixel-center coordinates) within
/// `bbox_f`, clipped to `image_size`, into a [`TransformedGeometry`].
///
/// `base_touches_edge` is OR-ed with whatever clipping against `image_size` introduces, so a
/// shape that already touched the edge before the transform keeps that flag even if the
/// transformed shape happens to land fully inside the image.
fn rasterize_geometry(
    bbox_f: [f32; 4],
    image_size: ImageSize,
    base_touches_edge: bool,
    mut inside_shape: impl FnMut(f32, f32) -> bool,
) -> TransformedGeometry {
    if image_size.width == 0 || image_size.height == 0 {
        return TransformedGeometry {
            touches_edge: true,
            ..Default::default()
        };
    }
    let max_x = (image_size.width - 1) as f32;
    let max_y = (image_size.height - 1) as f32;

    let touches_edge = base_touches_edge
        || bbox_f[0] < 0.0
        || bbox_f[1] < 0.0
        || bbox_f[2] > max_x
        || bbox_f[3] > max_y;

    let x_min = bbox_f[0].floor().clamp(0.0, max_x) as u32;
    let y_min = bbox_f[1].floor().clamp(0.0, max_y) as u32;
    let x_max = bbox_f[2].ceil().clamp(0.0, max_x) as u32;
    let y_max = bbox_f[3].ceil().clamp(0.0, max_y) as u32;

    if x_max < x_min || y_max < y_min {
        return TransformedGeometry {
            bbox: [x_min, y_min, x_min, y_min],
            touches_edge,
            ..Default::default()
        };
    }

    let width = (x_max - x_min + 1) as usize;
    let height = (y_max - y_min + 1) as usize;
    let mut mask_data = BitVec::<u64, Lsb0>::repeat(false, width * height);

    let mut area = 0usize;
    let (mut sum_x, mut sum_y, mut sum_x2, mut sum_y2, mut sum_xy) = (0u64, 0u64, 0u64, 0u64, 0u64);
    // Tracks the actual extent of set pixels so the candidate scan window (which floor/ceil
    // padded outward to not miss any boundary pixel) can be cropped down to a tight bbox below
    // — callers like `get_feret_diameter`/`get_centroid` read `bbox` directly, so a loose box
    // would silently skew those metrics even though the mask itself is correct.
    let (mut tight_x_min, mut tight_x_max) = (u32::MAX, 0u32);
    let (mut tight_y_min, mut tight_y_max) = (u32::MAX, 0u32);

    for ly in 0..height {
        let gy = y_min + ly as u32;
        for lx in 0..width {
            let gx = x_min + lx as u32;
            if inside_shape(gx as f32 + 0.5, gy as f32 + 0.5) {
                mask_data.set(ly * width + lx, true);
                area += 1;
                let (xu, yu) = (gx as u64, gy as u64);
                sum_x += xu;
                sum_y += yu;
                sum_x2 += xu * xu;
                sum_y2 += yu * yu;
                sum_xy += xu * yu;
                tight_x_min = tight_x_min.min(gx);
                tight_x_max = tight_x_max.max(gx);
                tight_y_min = tight_y_min.min(gy);
                tight_y_max = tight_y_max.max(gy);
            }
        }
    }

    if area == 0 {
        return TransformedGeometry {
            bbox: [x_min, y_min, x_min, y_min],
            touches_edge,
            ..Default::default()
        };
    }

    if (tight_x_min, tight_y_min, tight_x_max, tight_y_max) == (x_min, y_min, x_max, y_max) {
        return TransformedGeometry {
            bbox: [x_min, y_min, x_max, y_max],
            mask_data,
            area,
            touches_edge,
            sum_x,
            sum_y,
            sum_x2,
            sum_y2,
            sum_xy,
        };
    }

    let tight_width = (tight_x_max - tight_x_min + 1) as usize;
    let tight_height = (tight_y_max - tight_y_min + 1) as usize;
    let mut tight_mask = BitVec::<u64, Lsb0>::repeat(false, tight_width * tight_height);
    for ly in 0..tight_height {
        let src_row = (tight_y_min - y_min) as usize + ly;
        let dst_row = ly * tight_width;
        for lx in 0..tight_width {
            let src_idx = src_row * width + (tight_x_min - x_min) as usize + lx;
            if mask_data[src_idx] {
                tight_mask.set(dst_row + lx, true);
            }
        }
    }

    TransformedGeometry {
        bbox: [tight_x_min, tight_y_min, tight_x_max, tight_y_max],
        mask_data: tight_mask,
        area,
        touches_edge,
        sum_x,
        sum_y,
        sum_x2,
        sum_y2,
        sum_xy,
    }
}

/// Area of the convex hull of `points`, via Andrew's monotone chain + the shoelace
/// formula. `points` is sorted (and may be reordered) in place. Returns 0.0 for fewer
/// than three non-collinear points.
fn convex_hull_area(points: &mut [(f64, f64)]) -> f64 {
    if points.len() < 3 {
        return 0.0;
    }
    points.sort_by(|a, b| {
        a.0.partial_cmp(&b.0)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
    });

    // Cross product of OA x OB; > 0 means a counter-clockwise turn.
    let cross = |o: (f64, f64), a: (f64, f64), b: (f64, f64)| {
        (a.0 - o.0) * (b.1 - o.1) - (a.1 - o.1) * (b.0 - o.0)
    };

    let mut hull: Vec<(f64, f64)> = Vec::with_capacity(points.len() + 1);
    // Lower hull.
    for &p in points.iter() {
        while hull.len() >= 2 && cross(hull[hull.len() - 2], hull[hull.len() - 1], p) <= 0.0 {
            hull.pop();
        }
        hull.push(p);
    }
    // Upper hull.
    let lower = hull.len() + 1;
    for &p in points.iter().rev() {
        while hull.len() >= lower && cross(hull[hull.len() - 2], hull[hull.len() - 1], p) <= 0.0 {
            hull.pop();
        }
        hull.push(p);
    }
    hull.pop(); // last point equals the first

    if hull.len() < 3 {
        return 0.0;
    }
    // Shoelace formula.
    let mut twice_area = 0.0;
    for i in 0..hull.len() {
        let (x1, y1) = hull[i];
        let (x2, y2) = hull[(i + 1) % hull.len()];
        twice_area += x1 * y2 - x2 * y1;
    }
    twice_area.abs() / 2.0
}

#[cfg(test)]
mod tests {
    use super::*;
    use evanalyzer_cfg::core_types::ObjectId;

    /// A drawn shape that doesn't fill its bounding box (an oval, a
    /// polygon, a brush stroke): only its own pixels count.
    #[test]
    fn an_annotation_is_built_from_its_masks_pixels() {
        // 3x3 box at (10, 20), a plus sign: 5 of 9 pixels.
        let mask: BitVec<u64, Lsb0> = [false, true, false, true, true, true, false, true, false]
            .into_iter()
            .collect();
        let object = Object::from_mask_geometry(
            &ImageSize {
                width: 100,
                height: 100,
            },
            mask,
            [10, 20, 12, 22],
            ImagePlane { z: 2, c: 1, t: 3 },
            HashSet::from([ObjectClass::Global(4)]),
        );
        assert_eq!(object.area, 5);
        assert_eq!(object.get_centroid(), (11.0, 21.0));
        assert_eq!(object.get_bbox_center(), (11.0, 21.0), "symmetric");
        // Moments: x 11,10,11,12,11 / y 20,21,21,21,22.
        assert_eq!((object.sum_x, object.sum_y), (55, 105));
        assert_eq!(
            object.sum_xy,
            11 * 20 + 10 * 21 + 11 * 21 + 12 * 21 + 11 * 22
        );
        assert!(!object.touches_edge);
        assert!(
            object.intensities.is_empty(),
            "measured on the image file later"
        );
        assert_eq!(
            object.segmentation_class,
            SegmentationClass::MANUAL_ANNOTATED
        );
        let settings = object.to_object_settings();
        assert_eq!(
            (settings.z_stack, settings.c_stack, settings.t_stack),
            (2, 1, 3)
        );
    }

    /// An L: the centroid is the mean of its pixels, not its box's centre.
    #[test]
    fn the_centroid_is_the_mean_pixel_position_not_the_box_centre() {
        // A full first column and a full last row of a 5×5 box at (10, 20).
        let mut mask = BitVec::<u64, Lsb0>::repeat(false, 25);
        for i in 0..5 {
            mask.set(i * 5, true);
            mask.set(20 + i, true);
        }
        let object = Object::new(ObjectInit {
            bbox: [10, 20, 14, 24],
            mask_data: mask,
            area: 9,
            ..Default::default()
        });
        // x: 0,0,0,0,0 + 1,2,3,4 → 10/9; y: 0..4 + 4,4,4,4 → 26/9.
        let (x, y) = object.get_centroid();
        assert!((x - (10.0 + 10.0 / 9.0)).abs() < 1e-4, "{x}");
        assert!((y - (20.0 + 26.0 / 9.0)).abs() < 1e-4, "{y}");
        assert_eq!(object.get_bbox_center(), (12.0, 22.0));
    }

    /// The word-at-a-time centroid equals the mean of the pixels visited
    /// one by one, for masks of every width (rows across word borders).
    #[test]
    fn the_centroid_matches_the_pixel_by_pixel_mean() {
        let mut state = 0x9E37_79B9_u32;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            state
        };
        for _ in 0..500 {
            let (w, h) = (1 + next() % 90, 1 + next() % 40);
            let (x0, y0) = (next() % 1000, next() % 1000);
            let density = 1 + next() % 4;
            let mask: BitVec<u64, Lsb0> = (0..w * h).map(|_| next() % density == 0).collect();
            let object = Object::new(ObjectInit {
                bbox: [x0, y0, x0 + w - 1, y0 + h - 1],
                area: mask.count_ones(),
                mask_data: mask.clone(),
                ..Default::default()
            });
            let n = mask.count_ones() as f64;
            if n == 0.0 {
                continue;
            }
            let mx = mask.iter_ones().map(|i| (i as u32 % w) as f64).sum::<f64>() / n;
            let my = mask.iter_ones().map(|i| (i as u32 / w) as f64).sum::<f64>() / n;
            let (cx, cy) = object.get_centroid();
            assert!((cx as f64 - (x0 as f64 + mx)).abs() < 1e-3, "{w}×{h}");
            assert!((cy as f64 - (y0 as f64 + my)).abs() < 1e-3, "{w}×{h}");
        }
    }

    #[test]
    fn an_object_without_pixels_has_its_box_centre_as_centroid() {
        let object = Object::new(ObjectInit {
            bbox: [0, 0, 2, 4],
            mask_data: BitVec::<u64, Lsb0>::repeat(false, 15),
            ..Default::default()
        });
        assert_eq!(object.get_centroid(), (1.0, 2.0));
    }

    #[test]
    fn an_annotation_at_the_image_border_touches_its_edge() {
        let object = Object::from_mask_geometry(
            &ImageSize {
                width: 4,
                height: 4,
            },
            BitVec::<u64, Lsb0>::repeat(true, 2),
            [2, 3, 3, 3],
            ImagePlane { z: 0, c: 0, t: 0 },
            HashSet::new(),
        );
        assert!(object.touches_edge);
        assert!(object.get_perimeter() > 0.0);
    }

    /// Build a finalized object from a rectangular `rows × cols` boolean pattern.
    fn object_from_pattern(pattern: &[&[bool]]) -> Object {
        let height = pattern.len();
        let width = pattern[0].len();
        let mut mask = BitVec::<u64, Lsb0>::repeat(false, width * height);
        let mut area = 0usize;
        for (y, row) in pattern.iter().enumerate() {
            for (x, &on) in row.iter().enumerate() {
                if on {
                    mask.set(y * width + x, true);
                    area += 1;
                }
            }
        }
        Object::new(ObjectInit {
            id: ObjectId(1),
            bbox: [0, 0, (width - 1) as u32, (height - 1) as u32],
            mask_data: mask,
            area,
            ..Default::default()
        })
    }

    /// A filled square object, `side` pixels wide, with top-left corner at `(x, y)`.
    fn square_object_at(x: u32, y: u32, side: u32) -> Object {
        let area = (side * side) as usize;
        Object::new(ObjectInit {
            id: ObjectId(1),
            bbox: [x, y, x + side - 1, y + side - 1],
            mask_data: BitVec::<u64, Lsb0>::repeat(true, area),
            area,
            ..Default::default()
        })
    }

    const LARGE: ImageSize = ImageSize {
        width: 1000,
        height: 1000,
    };

    #[test]
    fn circularity_of_a_zero_area_object_is_zero_not_nan() {
        // circularity() and get_roundness() compute the identical formula,
        // but only get_roundness() guarded perimeter == 0 - a zero-area
        // object (the only way perimeter can be 0) would otherwise divide
        // by zero here instead of returning a sentinel.
        let object = Object::new(ObjectInit {
            id: ObjectId(1),
            bbox: [0, 0, 0, 0],
            mask_data: BitVec::<u64, Lsb0>::new(),
            area: 0,
            ..Default::default()
        });
        assert_eq!(
            object.get_perimeter(),
            0.0,
            "test assumption: zero-area implies zero perimeter"
        );
        assert_eq!(object.circularity(), 0.0);
        assert!(!object.circularity().is_nan());
    }

    #[test]
    fn dilated_geometry_grows_bbox_by_exactly_the_margin() {
        let object = square_object_at(100, 100, 4); // bbox [100,100,103,103]
        let grown = object.dilated_geometry(2.0, LARGE);
        assert_eq!(grown.bbox, [98, 98, 105, 105]);
        assert!(grown.area > 16);
    }

    #[test]
    fn dilated_geometry_zero_margin_is_identity() {
        let object = square_object_at(100, 100, 4);
        let grown = object.dilated_geometry(0.0, LARGE);
        assert_eq!(grown.bbox, object.bbox);
        assert_eq!(grown.area, object.area);
    }

    #[test]
    fn eroded_geometry_shrinks_bbox_by_exactly_the_margin() {
        let object = square_object_at(100, 100, 8); // bbox [100,100,107,107]
        let shrunk = object.eroded_geometry(2.0, LARGE);
        assert_eq!(shrunk.bbox, [102, 102, 105, 105]);
        assert!(shrunk.area > 0 && shrunk.area <= 16);
    }

    #[test]
    fn eroded_geometry_past_the_whole_mask_is_empty() {
        let object = square_object_at(100, 100, 4);
        let shrunk = object.eroded_geometry(10.0, LARGE);
        assert_eq!(shrunk.area, 0);
    }

    #[test]
    fn combine_geometry_subtract_removes_only_the_overlapping_pixels() {
        let a = square_object_at(10, 10, 10); // [10,10,19,19], area 100
        let b = square_object_at(13, 13, 4); // [13,13,16,16], area 16, inside a
        let result = a.combine_geometry(&[&b], BooleanOp::Subtract, LARGE);
        assert_eq!(result.area, 100 - 16);
    }

    #[test]
    fn combine_geometry_subtract_with_no_overlap_is_unchanged() {
        let a = square_object_at(10, 10, 10);
        let b = square_object_at(50, 50, 4); // far away, no overlap
        let result = a.combine_geometry(&[&b], BooleanOp::Subtract, LARGE);
        assert_eq!(result.area, a.area);
        assert_eq!(result.bbox, a.bbox);
    }

    #[test]
    fn combine_geometry_subtract_unions_multiple_subtrahends() {
        let a = square_object_at(10, 10, 10); // [10,10,19,19], area 100
        let b1 = square_object_at(10, 10, 3); // corner, area 9
        let b2 = square_object_at(16, 16, 3); // opposite corner, area 9, no overlap with b1
        let result = a.combine_geometry(&[&b1, &b2], BooleanOp::Subtract, LARGE);
        assert_eq!(result.area, 100 - 9 - 9);
    }

    #[test]
    fn combine_geometry_and_is_the_overlap_only() {
        let a = square_object_at(10, 10, 10); // [10,10,19,19], area 100
        let b = square_object_at(15, 15, 10); // [15,15,24,24], area 100, overlaps a in [15,15,19,19]
        let result = a.combine_geometry(&[&b], BooleanOp::And, LARGE);
        assert_eq!(result.area, 25); // 5x5 overlap
        assert_eq!(result.bbox, [15, 15, 19, 19]);
    }

    #[test]
    fn combine_geometry_and_with_no_overlap_is_empty() {
        let a = square_object_at(10, 10, 10);
        let b = square_object_at(50, 50, 4);
        let result = a.combine_geometry(&[&b], BooleanOp::And, LARGE);
        assert_eq!(result.area, 0);
    }

    #[test]
    fn combine_geometry_and_unions_multiple_partners_before_intersecting() {
        let a = square_object_at(10, 10, 10); // [10,10,19,19], area 100
        let b1 = square_object_at(10, 10, 3); // corner, area 9, fully inside a
        let b2 = square_object_at(16, 16, 3); // opposite corner, area 9, fully inside a
        let result = a.combine_geometry(&[&b1, &b2], BooleanOp::And, LARGE);
        assert_eq!(
            result.area,
            9 + 9,
            "AND with the union of both partners, not just one"
        );
    }

    #[test]
    fn combine_geometry_or_extends_beyond_self_bbox() {
        let a = square_object_at(10, 10, 10); // [10,10,19,19], area 100
        let b = square_object_at(15, 15, 10); // [15,15,24,24], area 100, overlap 25
        let result = a.combine_geometry(&[&b], BooleanOp::Or, LARGE);
        assert_eq!(
            result.bbox,
            [10, 10, 24, 24],
            "union bbox must cover both operands"
        );
        assert_eq!(result.area, 100 + 100 - 25);
    }

    #[test]
    fn combine_geometry_or_with_no_overlap_is_just_self() {
        let a = square_object_at(10, 10, 10);
        let b = square_object_at(50, 50, 4);
        // No overlap means "no partner" at the command level, but the raw geometry
        // primitive itself still computes a true union if asked.
        let result = a.combine_geometry(&[&b], BooleanOp::Or, LARGE);
        assert_eq!(result.area, a.area + b.area);
    }

    #[test]
    fn combine_geometry_xor_excludes_the_overlap() {
        let a = square_object_at(10, 10, 10); // area 100
        let b = square_object_at(15, 15, 10); // area 100, overlap 25
        let result = a.combine_geometry(&[&b], BooleanOp::Xor, LARGE);
        assert_eq!(
            result.area,
            100 + 100 - 2 * 25,
            "XOR must exclude the overlap entirely"
        );
        let result_object = Object::new(ObjectInit {
            id: ObjectId(1),
            bbox: result.bbox,
            mask_data: result.mask_data,
            area: result.area,
            ..Default::default()
        });
        assert!(
            !result_object.is_part_of(17, 17),
            "a pixel in both operands must be excluded"
        );
    }

    #[test]
    fn measure_intensities_samples_correct_pixel_in_a_multi_tile_image() {
        // Regression test: when the full image is larger than the current tile (the
        // whole-slide-image / multi-tile case), sampling must use the tile's own
        // pixel grid directly - not scale coordinates by tile_size/full_image_size,
        // which truncates to 0 for any real multi-tile image and made every
        // synthesized object (e.g. colocalization intersections, Voronoi regions)
        // sample channel pixel 0 regardless of its actual tile-local position.
        use crate::image::ManagedImage;
        use crate::image::Point2d;
        use crate::pipeline::pipeline_cache::GlobalPipelineCache;
        use crate::{ImageContainer, ImagePlane};
        use kornia_image::{Image, ImageSize};
        use std::sync::Arc;

        let tile_size = ImageSize {
            width: 15,
            height: 20,
        };
        let offset = Point2d { x: 10, y: 15 };

        let mut cache = GlobalPipelineCache::default();

        // Distinct intensities at two different tile-local positions.
        let mut intensity = vec![0.0f32; 15 * 20];
        intensity[0] = 10.0; // local (0,0)
        intensity[5 * 15 + 7] = 99.0; // local (7,5)
        let channel = Arc::new(ImageContainer::F32Gray(ManagedImage {
            data: Image::<f32, 1>::from_size_slice(tile_size, &intensity).unwrap(),
            tile_offset: offset,
            plane: Some(ImagePlane { z: 0, c: 0, t: 0 }),
            dimensionality: ImageDimensionality::Planar,
        }));
        // The cache key's `ImageTile` is what `measure_intensities` now matches mask
        // pixels against - it must reflect this channel's *real* tile position/size,
        // not a placeholder, since a mismatch here means "no tile covers this pixel".
        cache.add_to_channel_cache(
            channel,
            0,
            ImageTile {
                offset_x: offset.x,
                offset_y: offset.y,
                width: tile_size.width,
                height: tile_size.height,
            },
        );

        // A 1-pixel object at the global position corresponding to tile-local (7,5).
        let object = Object::new(ObjectInit {
            id: ObjectId(1),
            bbox: [10 + 7, 15 + 5, 10 + 7, 15 + 5],
            mask_data: BitVec::<u64, Lsb0>::repeat(true, 1),
            area: 1,
            ..Default::default()
        });

        let intensities = object.measure_intensities(&cache);

        assert_eq!(
            intensities.get(&0).unwrap().sum_intensity,
            99.0,
            "object at tile-local (7,5) must sample its own pixel, not pixel (0,0)"
        );
    }

    /// Tile by tile gives every object what measuring it alone gives -
    /// inside one tile, over two and over four, two channels, objects big
    /// enough for counted gray values among them.
    #[test]
    fn measure_intensities_of_matches_measuring_one_by_one() {
        use crate::ImageContainer;
        use crate::image::ManagedImage;
        use crate::image::Point2d;
        use crate::pipeline::pipeline_cache::GlobalPipelineCache;
        use kornia_image::Image;
        use std::sync::Arc;

        const SIZE: usize = 400;
        const TILE: usize = 160;
        let mut cache = GlobalPipelineCache::default();
        for channel in 0..2 {
            for (x0, y0) in (0..3).flat_map(|ty| (0..3).map(move |tx| (tx * TILE, ty * TILE))) {
                let (w, h) = ((SIZE - x0).min(TILE), (SIZE - y0).min(TILE));
                let samples: Vec<f32> = (0..w * h)
                    .map(|i| {
                        let (x, y) = (x0 + i % w, y0 + i / w);
                        ((x * 31 + y * 17 + channel * 7) % 4000) as f32
                    })
                    .collect();
                let image = ImageContainer::F32Gray(ManagedImage {
                    data: Image::<f32, 1>::from_size_slice(
                        ImageSize {
                            width: w,
                            height: h,
                        },
                        &samples,
                    )
                    .unwrap(),
                    tile_offset: Point2d { x: x0, y: y0 },
                    plane: None,
                    dimensionality: ImageDimensionality::Planar,
                });
                let tile = ImageTile {
                    offset_x: x0,
                    offset_y: y0,
                    width: w,
                    height: h,
                };
                cache.add_to_channel_cache(Arc::new(image), channel as i32, tile);
            }
        }
        // (bbox) - every other mask pixel set.
        let boxes: [[u32; 4]; 5] = [
            [10, 10, 40, 30],     // inside one tile
            [150, 20, 170, 60],   // over two (left/right)
            [20, 150, 50, 170],   // over two (top/bottom)
            [140, 140, 180, 175], // over four
            [0, 0, 399, 399],     // the whole image (80,000 px): counted gray values
        ];
        let mut objects: Vec<Object> = boxes
            .iter()
            .enumerate()
            .map(|(i, &bbox)| {
                let w = (bbox[2] - bbox[0] + 1) as usize;
                let h = (bbox[3] - bbox[1] + 1) as usize;
                let mask: BitVec<u64, Lsb0> = (0..w * h).map(|k| k % 2 == 0).collect();
                Object::new(ObjectInit {
                    id: ObjectId(i as u128 + 1),
                    bbox,
                    area: mask.count_ones(),
                    mask_data: mask,
                    ..Default::default()
                })
            })
            .collect();
        let one_by_one: Vec<IndexMap<i32, Intensity>> = objects
            .iter()
            .map(|o| o.measure_intensities(&cache))
            .collect();
        Object::measure_intensities_of(&mut objects, &cache);
        for (object, expected) in objects.iter().zip(&one_by_one) {
            assert_eq!(object.intensities.len(), 2, "object {:?}", object.id);
            for (channel, e) in expected {
                let got = &object.intensities[channel];
                let what = format!("object {:?} channel {channel}", object.id);
                // The sums in another tile order: equal up to rounding.
                assert!(
                    (got.sum_intensity - e.sum_intensity).abs() <= 1e-9 * e.sum_intensity,
                    "{what}"
                );
                assert_eq!(
                    (got.min_intensity, got.max_intensity),
                    (e.min_intensity, e.max_intensity),
                    "{what}"
                );
                assert_eq!(got.median_intensity, e.median_intensity, "{what}");
                assert_eq!(
                    got.lower_quartile_intensity, e.lower_quartile_intensity,
                    "{what}"
                );
                assert_eq!(
                    got.upper_quartile_intensity, e.upper_quartile_intensity,
                    "{what}"
                );
                assert_eq!(got.mad_intensity, e.mad_intensity, "{what}");
                assert!(
                    (got.std_intensity - e.std_intensity).abs() <= 1e-9 * e.std_intensity,
                    "{what}"
                );
            }
        }
    }

    #[test]
    fn measure_intensities_skips_mask_pixels_outside_the_current_tile() {
        // Regression test: `TransformObjects` (Expand/Scale/Circle/...) clamps a object's mask
        // against the *full* image, not the tile it's being processed in, so a transformed
        // object near a non-origin tile's top/left edge can legally have mask pixels that fall
        // outside the tile currently loaded in `cache`. Before the fix, `x_abs - tile_offset`
        // underflowed for those pixels (silently wrapping in release builds) and indexed the
        // channel slice miles out of bounds, panicking. Now those pixels must simply be
        // skipped rather than sampled.
        use crate::image::ManagedImage;
        use crate::image::Point2d;
        use crate::pipeline::pipeline_cache::GlobalPipelineCache;
        use crate::{ImageContainer, ImagePlane};
        use kornia_image::{Image, ImageSize};
        use std::sync::Arc;

        let tile_size = ImageSize {
            width: 15,
            height: 20,
        };
        let offset = Point2d { x: 10, y: 15 };

        let mut cache = GlobalPipelineCache::default();

        let mut intensity = vec![7.0f32; 15 * 20];
        intensity[5 * 15 + 7] = 99.0; // tile-local (7,5)
        let channel = Arc::new(ImageContainer::F32Gray(ManagedImage {
            data: Image::<f32, 1>::from_size_slice(tile_size, &intensity).unwrap(),
            tile_offset: offset,
            plane: Some(ImagePlane { z: 0, c: 0, t: 0 }),
            dimensionality: ImageDimensionality::Planar,
        }));
        // The cache key's `ImageTile` is what `measure_intensities` now matches mask
        // pixels against - it must reflect this channel's *real* tile position/size,
        // not a placeholder, since a mismatch here means "no tile covers this pixel".
        cache.add_to_channel_cache(
            channel,
            0,
            ImageTile {
                offset_x: offset.x,
                offset_y: offset.y,
                width: tile_size.width,
                height: tile_size.height,
            },
        );

        // A 4x4 mask straddling the tile's top-left corner: global x in [8, 11], y in
        // [13, 16] against a tile that only starts at (10, 15). Only the bottom-right
        // 2x2 quadrant (global (10,15)..(11,16), i.e. tile-local (0,0)..(1,1)) is in-tile.
        let mask = BitVec::<u64, Lsb0>::repeat(true, 4 * 4);
        let object = Object::new(ObjectInit {
            id: ObjectId(1),
            bbox: [8, 13, 11, 16],
            mask_data: mask,
            area: 16,
            ..Default::default()
        });

        // Must not panic, and must only have sampled the four in-tile pixels (tile-local
        // (0,0), (1,0), (0,1), (1,1)), all value 7.0 - away from the 99.0 marker.
        let intensities = object.measure_intensities(&cache);
        let intensity = intensities.get(&0).unwrap();
        assert_eq!(intensity.sum_intensity, 28.0);
        assert_eq!(intensity.max_intensity, 7.0);
    }

    #[test]
    fn perimeter_uses_inclusive_stride_for_2x2_square() {
        // 2×2 filled block, traced as a single 2×2 square with 4 corners:
        // sumdx=sumdy=4, so perimeter = 8 - 4·(2-√2) = 4√2. Cross-checked
        // against a from-scratch Rust port of ij.gui.Wand/PolygonRoi (see
        // `compute_perimeter`'s doc comment) and by hand-tracing the walk.
        let object = object_from_pattern(&[&[true, true], &[true, true]]);
        let expected = 4.0 * std::f32::consts::SQRT_2;
        assert!(
            (object.get_perimeter() - expected).abs() < 1e-3,
            "got {}, expected {}",
            object.get_perimeter(),
            expected
        );
    }

    #[test]
    fn perimeter_single_pixel() {
        // One pixel traced as a 1×1 square with 4 corners: sumdx=sumdy=2
        // (each edge of the square is walked once in x and once in y over
        // the full loop), so perimeter = 4 - 4·(2-√2) = 4√2 - 4 = 2√2.
        // This is ImageJ's real, documented behavior for a single-pixel ROI
        // (its corner correction undercorrects maximally-jagged shapes) -
        // not a bug in this port. Cross-checked against a from-scratch Rust
        // port of ij.gui.Wand/PolygonRoi and by hand-tracing the walk.
        let object = object_from_pattern(&[&[true]]);
        let expected = 2.0 * std::f32::consts::SQRT_2;
        assert!(
            (object.get_perimeter() - expected).abs() < 1e-3,
            "got {}, expected {}",
            object.get_perimeter(),
            expected
        );
    }

    #[test]
    fn circularity_of_a_rasterized_circle_matches_imagejs_traced_perimeter_algorithm() {
        // A rasterized disc of radius 20px (bounding box 46x46, centered),
        // matching the shape produced by pixel-membership-test rasterization
        // (bit set iff pixel-center distance <= radius). ImageJ's actual
        // Wand/PolygonRoi traced-perimeter algorithm reports circularity
        // ~0.92 for this exact raster (verified with a from-scratch Rust
        // port of that algorithm, itself cross-checked by hand-tracing a
        // single pixel and a 2x2 square - see `compute_perimeter`'s doc
        // comment). The prior (buggy) perimeter formula reported ~0.27 for
        // the same shape - roughly double the true boundary length, since
        // circularity ~ 1/perimeter^2.
        let radius: f64 = 20.0;
        let size = (radius * 2.0 + 6.0) as usize;
        let center = size as f64 / 2.0;
        let pattern: Vec<Vec<bool>> = (0..size)
            .map(|y| {
                (0..size)
                    .map(|x| {
                        let dx = x as f64 + 0.5 - center;
                        let dy = y as f64 + 0.5 - center;
                        dx * dx + dy * dy <= radius * radius
                    })
                    .collect()
            })
            .collect();
        let pattern_refs: Vec<&[bool]> = pattern.iter().map(|row| row.as_slice()).collect();
        let object = object_from_pattern(&pattern_refs);

        let circularity = object.circularity();
        assert!(
            circularity > 0.85,
            "a visibly round object's circularity should be close to 1.0 (ImageJ reports ~0.92 \
             for this exact raster), got {circularity} - the perimeter calculation likely \
             overestimates the boundary length again"
        );
    }

    #[test]
    fn solidity_is_one_for_filled_square() {
        let object = object_from_pattern(&[
            &[true, true, true],
            &[true, true, true],
            &[true, true, true],
        ]);
        // Hull area == pixel area (9) → perfectly solid.
        assert!((object.get_solidity() - 1.0).abs() < 1e-4);
    }

    #[test]
    fn solidity_below_one_for_concave_plus_shape() {
        // A plus/cross: concave, so the convex hull is larger than the pixel area.
        let object = object_from_pattern(&[
            &[false, true, false],
            &[true, true, true],
            &[false, true, false],
        ]);
        let s = object.get_solidity();
        assert!(
            s > 0.0 && s < 1.0,
            "plus shape solidity should be in (0,1): {s}"
        );
    }

    #[test]
    fn solidity_one_for_single_pixel() {
        let object = object_from_pattern(&[&[true]]);
        assert_eq!(object.get_solidity(), 1.0);
    }

    // ---- Classes and scopes ----

    fn scoped(n: u32) -> Scope {
        Scope::Scoped(crate::pipeline::scope::ScopeId(n))
    }

    #[test]
    fn an_object_takes_the_scope_of_its_first_class_and_never_mixes() {
        let mut object = Object::new(ObjectInit::default());
        object
            .add_object_class(ObjectClass::Local(1), scoped(3))
            .unwrap();
        assert_eq!(object.scope(), scoped(3));
        assert!(object.is_of(&ObjectClass::Local(1), scoped(3)));
        assert!(
            !object.is_of(&ObjectClass::Local(1), scoped(4)),
            "another instance's"
        );
        assert!(
            object
                .add_object_class(ObjectClass::Global(1), scoped(3))
                .is_err()
        );
        object
            .add_object_class(ObjectClass::Local(2), scoped(3))
            .unwrap();

        // Removing every class keeps it internal; its next class decides again.
        object.clear_object_classes();
        assert_eq!(object.scope(), scoped(3));
        object
            .add_object_class(ObjectClass::Global(5), scoped(3))
            .unwrap();
        assert_eq!(object.scope(), Scope::Global);
        assert!(object.is_of(&ObjectClass::Global(5), Scope::Global));
        assert!(
            object.is_of(&ObjectClass::Global(5), scoped(3)),
            "seen from inside a module"
        );
    }

    #[test]
    fn a_local_class_of_another_scope_is_not_removed() {
        let mut object = Object::new(ObjectInit::default());
        object
            .add_object_class(ObjectClass::Local(1), scoped(1))
            .unwrap();
        object.remove_object_class(&ObjectClass::Local(1), scoped(2));
        assert!(object.is_of(&ObjectClass::Local(1), scoped(1)));
        object.remove_object_class(&ObjectClass::Local(1), scoped(1));
        assert!(object.object_classes().is_empty());
    }

    #[test]
    fn a_command_sees_the_projects_objects_and_those_of_its_scope() {
        let mut local = Object::new(ObjectInit::default());
        local
            .add_object_class(ObjectClass::Local(1), scoped(1))
            .unwrap();
        assert!(local.is_visible_in(scoped(1)));
        assert!(!local.is_visible_in(scoped(2)));
        assert!(!local.is_visible_in(Scope::Global));
        let global = Object::new(ObjectInit::default());
        assert!(global.is_visible_in(scoped(2)));
    }
}
