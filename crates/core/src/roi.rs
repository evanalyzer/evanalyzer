use bitvec::prelude::*;
use evanalyzer_cfg::{
    core_types::{ObjectClass, ObjectId, SegmentationClass, TrackId},
    settings::roi_settings::{IntensitySettings, RoiSettings, TrackSettings},
};
use indexmap::IndexMap;
use kornia_image::ImageSize;
use std::collections::HashSet;

use crate::pipeline::{
    pipeline_cache::{sample_channel_pixel, PipelineCache},
    pipeline_context::PipelineContext,
};
use crate::ImagePlane;

#[derive(Debug, Default, Clone)]
pub struct Intensity {
    /// Sum of all pixel intensities in the ROI
    pub sum_intensity: f64,
    /// Minimum pixel intensity in the ROI
    pub min_intensity: f32,
    /// Maximum pixel intensity in the ROI
    pub max_intensity: f32,
    /// Average pixel intensity in the ROI
    pub avg_intensity: f32,
    /// All pixel values (used for computing median and std_dev)
    pub pixel_values: Vec<f32>,
}

#[derive(Debug, Default, Clone)]
pub struct Track {
    pub id: TrackId,
    pub roi_ids: Vec<ObjectId>,        // Ordered list of ROIs over time
    pub parent_track: Option<TrackId>, // If created by division
}

#[derive(Debug, Default, Clone)]
pub struct Roi {
    // Global unique object ID
    pub id: ObjectId,

    // Semantic class after threshold
    pub segmentation_class: SegmentationClass,

    // Dedicated class after classify roi
    pub object_class: HashSet<ObjectClass>,

    // Colocalization
    pub colocalized_with: IndexMap<ObjectClass, Vec<ObjectId>>,

    // Relation
    pub parent_id: Option<ObjectId>, // Who owns me?
    pub children: Vec<ObjectId>,     // Who is part of me?

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

    // --- Precomputed geometry metrics ---
    // Filled by `finalize_geometry()` at ROI creation time — which runs on the
    // parallel extraction/segmentation workers — and read back through
    // `get_perimeter()` / `get_ellipse()`. The perimeter is an O(bbox area)
    // boundary walk; computing it here keeps it off the single-threaded DB writer,
    // where a lazy computation would stall every other tile's insert. Both derive
    // purely from the immutable mask geometry (mask_data, bbox, area, moments),
    // which never changes after extraction, so the stored values stay valid for
    // the ROI's lifetime. Default to 0 / empty until finalize_geometry runs.
    //
    // These are intentionally private (no `pub`): keeping them module-private makes
    // a `Roi { .. }` struct literal illegal outside this module, which forces all
    // external construction through [`Roi::new`] — the one path that is guaranteed
    // to call [`finalize_geometry`](Self::finalize_geometry). Callers read them via
    // `get_perimeter()` / `get_ellipse()`.
    perimeter: f32,
    ellipse: FittingEllipse,
}

/// Caller-settable fields for building a [`Roi`] via [`Roi::new`].
///
/// Mirrors every field of [`Roi`] except the derived geometry metrics
/// (`perimeter`, `ellipse`), which [`Roi::new`] computes for you. Build it with
/// struct-update syntax and pass it to [`Roi::new`]:
///
/// ```ignore
/// let roi = Roi::new(RoiInit { id, bbox, mask_data, area, ..Default::default() });
/// ```
#[derive(Debug, Default, Clone)]
pub struct RoiInit {
    pub id: ObjectId,
    pub segmentation_class: SegmentationClass,
    pub object_class: HashSet<ObjectClass>,
    pub colocalized_with: IndexMap<ObjectClass, Vec<ObjectId>>,
    pub parent_id: Option<ObjectId>,
    pub children: Vec<ObjectId>,
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
}

#[derive(Debug, Default, Clone, Copy)]
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

/// Rasterized mask geometry produced by a shape transform (see [`Roi::scaled_geometry`],
/// [`Roi::circle_geometry`], [`Roi::fitting_ellipse_geometry`]).
///
/// Carries only the geometric fields of a [`Roi`] — identity, classes, lineage and plane are
/// deliberately left out, since callers decide whether the transformed shape replaces the
/// source ROI in place or becomes a brand-new ROI, and need to set those fields themselves
/// via [`RoiInit`].
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

impl Roi {
    /// Builds a fully-finalized ROI from [`RoiInit`].
    ///
    /// This is the only public way to construct a [`Roi`]: the geometry metrics
    /// (`perimeter`, `ellipse`) are computed here via
    /// [`finalize_geometry`](Self::finalize_geometry), so they can never be left
    /// uncomputed by a forgotten call.
    ///
    /// For ROIs assembled incrementally (e.g. pixel-by-pixel extraction where the
    /// mask and moments are filled in after construction), call
    /// [`finalize_geometry`](Self::finalize_geometry) again once accumulation is
    /// complete — the eager finalize here only reflects the geometry present in
    /// `init`.
    pub fn new(init: RoiInit) -> Self {
        let mut roi = Roi {
            id: init.id,
            segmentation_class: init.segmentation_class,
            object_class: init.object_class,
            colocalized_with: init.colocalized_with,
            parent_id: init.parent_id,
            children: init.children,
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
            perimeter: 0.0,
            ellipse: FittingEllipse::default(),
        };
        roi.finalize_geometry();
        roi
    }

    /// Checks if a global coordinate (x, y) is within the ROI's mask.
    pub fn is_part_of(&self, x: u32, y: u32) -> bool {
        let [x_min, y_min, x_max, y_max] = self.bbox;

        // bbox[2]/[3] are INCLUSIVE - the last pixel coordinate inside the ROI.
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

    pub fn add_object_class(&mut self, object_class: ObjectClass) {
        if object_class != ObjectClass::Unset {
            self.object_class.insert(object_class);
        }
    }

    /// Adds an object to the ROI which colocalzis with it
    ///
    /// The class is the object class of the other object which was used to
    /// calc the colocalization.
    pub fn add_colocalizing_object(&mut self, coloc_class: ObjectClass, object_id: ObjectId) {
        let coloc_per_class = self.colocalized_with.entry(coloc_class).or_default();
        coloc_per_class.push(object_id);
        coloc_per_class.sort();
        coloc_per_class.dedup();
    }

    pub fn has_object_class(&self, object_class: &ObjectClass) -> bool {
        self.object_class.contains(object_class)
    }

    pub fn has_object_classes(&self, object_classes: &[ObjectClass]) -> bool {
        object_classes
            .iter()
            .all(|class| self.has_object_class(class))
    }

    pub fn remove_object_class(&mut self, object_class: &ObjectClass) {
        self.object_class.remove(object_class);
    }

    /// Computes and stores the geometry metrics (`perimeter`, `ellipse`) from the
    /// current mask. Call once after the mask/moments are fully assembled — i.e.
    /// at ROI creation, on the parallel extraction workers — so later reads
    /// (classification, DB export) are free field accesses. The metrics depend
    /// only on the immutable geometry, so a single call is enough for the ROI's life.
    ///
    /// `pub(crate)` on purpose: one-shot construction should go through [`Roi::new`],
    /// which calls this for you. It stays reachable inside the crate for the
    /// incremental extraction path, which must re-finalize after the mask and
    /// moments are fully accumulated.
    pub(crate) fn finalize_geometry(&mut self) {
        self.perimeter = self.compute_perimeter();
        self.ellipse = self.compute_ellipse();
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
        (4.0 * std::f32::consts::PI * self.area as f32) / (perimeter * perimeter)
    }

    /// Returns the ROI perimeter in pixels, precomputed by
    /// [`finalize_geometry`](Self::finalize_geometry).
    ///
    /// The underlying [`compute_perimeter`](Self::compute_perimeter) is an
    /// O(bbox area) boundary walk — by far the most expensive ROI metric — which
    /// is why it is computed once at creation rather than on demand.
    pub fn get_perimeter(&self) -> f32 {
        self.perimeter
    }

    /// Calculates the perimeter of the ROI using ImageJ's algorithm.
    ///
    /// This method computes the perimeter by analyzing the boundary of the mask.
    /// It counts the number of transitions between foreground and background pixels,
    /// accounting for diagonal adjacencies. The calculation follows ImageJ's approach:
    /// - Horizontal and vertical edges contribute 1.0 to the perimeter
    /// - Diagonal edges contribute sqrt(2) ≈ 1.414 to the perimeter
    ///
    /// # Returns
    /// The perimeter value in pixels. A perfect circle with radius r has perimeter ≈ 2πr.
    fn compute_perimeter(&self) -> f32 {
        let [x_min, y_min, x_max, y_max] = self.bbox;
        if self.area == 0 || x_max < x_min || y_max < y_min {
            return 0.0;
        }

        // Mask stride uses the INCLUSIVE bbox convention (width = x_max - x_min + 1),
        // matching how the mask is built everywhere else. (The previous version dropped
        // the +1, reading misaligned bits and undercounting multi-row ROIs.)
        let width = (x_max - x_min + 1) as usize;
        let height = (y_max - y_min + 1) as usize;
        if width == 0 || height == 0 {
            return 0.0;
        }

        // Materialize the mask into a bool grid padded with a 1-pixel false border.
        // The border lets the 8-neighbor scan below run without per-neighbor bounds
        // checks, BitVec bit-addressing or Option handling — all of which dominated the
        // boundary walk, the single most expensive ROI metric.
        let pw = width + 2;
        let mut grid = vec![false; pw * (height + 2)];
        for y in 0..height {
            let src = y * width;
            let dst = (y + 1) * pw + 1;
            for x in 0..width {
                if self.mask_data.get(src + x).map(|b| *b).unwrap_or(false) {
                    grid[dst + x] = true;
                }
            }
        }

        let mut perimeter = 0.0f32;
        const SQRT2: f32 = std::f32::consts::SQRT_2;

        // Out-of-bounds neighbors are the false border cells, so they count as "outside",
        // exactly like the original. Orthogonal boundary edges weigh 1.0, diagonal SQRT2.
        for y in 1..=height {
            let row = y * pw;
            for x in 1..=width {
                if !grid[row + x] {
                    continue;
                }
                let orth = (!grid[row + x - 1] as u32)
                    + (!grid[row + x + 1] as u32)
                    + (!grid[row - pw + x] as u32)
                    + (!grid[row + pw + x] as u32);
                let diag = (!grid[row - pw + x - 1] as u32)
                    + (!grid[row - pw + x + 1] as u32)
                    + (!grid[row + pw + x - 1] as u32)
                    + (!grid[row + pw + x + 1] as u32);
                perimeter += orth as f32 + diag as f32 * SQRT2;
            }
        }

        // ImageJ divides by 2 because each boundary edge is counted from both sides.
        perimeter / 2.0
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

    /// Calculates centroid coordinates (center of mass).
    pub fn get_centroid(&self) -> (f32, f32) {
        let [x_min, y_min, x_max, y_max] = self.bbox;
        let x_center = (x_min as f32 + x_max as f32) / 2.0;
        let y_center = (y_min as f32 + y_max as f32) / 2.0;
        (x_center, y_center)
    }

    /// Returns Feret diameter (longest distance between boundary points).
    pub fn get_feret_diameter(&self) -> f32 {
        let [x_min, y_min, x_max, y_max] = self.bbox;
        let dx = (x_max - x_min) as f32;
        let dy = (y_max - y_min) as f32;
        (dx * dx + dy * dy).sqrt()
    }

    /// Returns minimum Feret diameter (perpendicular to max).
    pub fn get_min_feret_diameter(&self) -> f32 {
        let ellipse = self.get_ellipse();
        ellipse.minor
    }

    pub fn to_roi_settings(&self) -> RoiSettings {
        RoiSettings {
            id: self.id.clone(),
            segmentation_class: self.segmentation_class,
            object_class: self.object_class.clone(),
            colocalized_with: self.colocalized_with.clone(),
            parent_id: self.parent_id.clone(),
            children: self.children.clone(),
            track: TrackSettings {
                id: self.track.id.clone(),
                roi_ids: self.track.roi_ids.clone(),
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
                .map(|(k, v)| {
                    (
                        *k,
                        IntensitySettings {
                            sum_intensity: v.sum_intensity,
                            min_intensity: v.min_intensity,
                            max_intensity: v.max_intensity,
                            avg_intensity: v.avg_intensity,
                            pixel_values: vec![],
                        },
                    )
                })
                .collect(),
            perimeter: self.perimeter,
            z_stack: self.plane.z,
            c_stack: self.plane.c,
            t_stack: self.plane.t,
        }
    }

    pub fn from_roi_settings(s: RoiSettings) -> Self {
        Roi::new(RoiInit {
            id: s.id,
            segmentation_class: s.segmentation_class,
            object_class: s.object_class,
            colocalized_with: s.colocalized_with,
            parent_id: s.parent_id,
            children: s.children,
            track: Track {
                id: s.track.id,
                roi_ids: s.track.roi_ids,
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
                .map(|(k, v)| {
                    (
                        k,
                        Intensity {
                            sum_intensity: v.sum_intensity,
                            min_intensity: v.min_intensity,
                            max_intensity: v.max_intensity,
                            avg_intensity: v.avg_intensity,
                            pixel_values: Vec::new(),
                        },
                    )
                })
                .collect(),
            plane: ImagePlane {
                z: s.z_stack,
                c: s.c_stack,
                t: s.t_stack,
            },
        })
    }

    /// Computes the intersection with another ROI.
    /// Returns `Some(Roi)` representing the overlapping region, or `None` if there is no overlap.
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

        // 3. Assemble and return the brand-new structural intersection ROI
        Some(Roi::new(RoiInit {
            id: ObjectId::next(), // Typically initialized empty or generated by your core engine later
            segmentation_class: self.segmentation_class.clone(),
            parent_id: Some(self.id.clone()), // Tracks lineage ancestry
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

    /// Samples per-channel pixel intensities for this ROI's mask against the channel images
    /// held in `cache`. For ROIs assembled by [`ExtractRois`](crate::algos::classification::extract_rois::ExtractRois)
    /// this is redundant (it already measures intensities in the same pass that builds the
    /// mask). It exists for ROIs synthesized later in the pipeline — e.g. the intersection
    /// ROIs colocalization creates via [`overlaps`](Self::overlaps) — which have a valid
    /// mask/bbox but never had a chance to sample pixel data.
    pub fn measure_intensities(
        &self,
        ctx: &PipelineContext,
        cache: &PipelineCache,
    ) -> IndexMap<i32, Intensity> {
        let tile_offset = ctx.get_image_tile_offset();
        let size = ctx.get_image_size();
        let full_image_size = ctx.full_image_size();
        let origin_width = size.width;
        let zoom_x = size.width / full_image_size.width;
        let zoom_y = size.height / full_image_size.height;

        let channel_views = cache.image_cache.resolve_channel_views();

        let [xmin, ymin, xmax, ymax] = self.bbox;
        let rw = (xmax - xmin + 1) as usize;
        let rh = (ymax - ymin + 1) as usize;

        let mut sum = vec![0f64; channel_views.len()];
        let mut min = vec![f32::MAX; channel_views.len()];
        let mut max = vec![f32::MIN; channel_views.len()];
        let mut area = 0usize;

        for ry in 0..rh {
            for rx in 0..rw {
                if !self
                    .mask_data
                    .get(ry * rw + rx)
                    .map(|b| *b)
                    .unwrap_or(false)
                {
                    continue;
                }
                area += 1;

                let x_abs = xmin as usize + rx;
                let y_abs = ymin as usize + ry;
                let tile_x = x_abs - tile_offset.x;
                let tile_y = y_abs - tile_offset.y;
                let sample = (tile_y * zoom_y) * origin_width + tile_x * zoom_x;

                for (ci, (_, is_rgb, slice)) in channel_views.iter().enumerate() {
                    let val = sample_channel_pixel(*is_rgb, slice, sample);
                    sum[ci] += val as f64;
                    if val < min[ci] {
                        min[ci] = val;
                    }
                    if val > max[ci] {
                        max[ci] = val;
                    }
                }
            }
        }

        let n = area.max(1) as f64;
        channel_views
            .iter()
            .enumerate()
            .map(|(ci, (ch_idx, _, _))| {
                (
                    *ch_idx,
                    Intensity {
                        sum_intensity: sum[ci],
                        min_intensity: min[ci],
                        max_intensity: max[ci],
                        avg_intensity: (sum[ci] / n) as f32,
                        pixel_values: Vec::new(),
                    },
                )
            })
            .collect()
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
        let (cx, cy) = self.get_centroid();
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
        let (cx, cy) = self.get_centroid();
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
        let (cx, cy) = self.get_centroid();
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

    /// Build a finalized ROI from a rectangular `rows × cols` boolean pattern.
    fn roi_from_pattern(pattern: &[&[bool]]) -> Roi {
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
        Roi::new(RoiInit {
            id: ObjectId(1),
            bbox: [0, 0, (width - 1) as u32, (height - 1) as u32],
            mask_data: mask,
            area,
            ..Default::default()
        })
    }

    #[test]
    fn perimeter_uses_inclusive_stride_for_2x2_square() {
        // 2×2 filled block. With the (now correct) inclusive stride the boundary walk
        // visits all four pixels; the weighted ImageJ scheme yields 4 + 6·√2.
        let roi = roi_from_pattern(&[&[true, true], &[true, true]]);
        let expected = 4.0 + 6.0 * std::f32::consts::SQRT_2;
        assert!(
            (roi.get_perimeter() - expected).abs() < 1e-3,
            "got {}, expected {}",
            roi.get_perimeter(),
            expected
        );
    }

    #[test]
    fn perimeter_single_pixel() {
        // One pixel: all 8 neighbors are outside → (4·1 + 4·√2) / 2 = 2 + 2·√2.
        // (The old buggy stride returned 0 here because width collapsed to 0.)
        let roi = roi_from_pattern(&[&[true]]);
        let expected = 2.0 + 2.0 * std::f32::consts::SQRT_2;
        assert!((roi.get_perimeter() - expected).abs() < 1e-3);
    }

    #[test]
    fn solidity_is_one_for_filled_square() {
        let roi = roi_from_pattern(&[
            &[true, true, true],
            &[true, true, true],
            &[true, true, true],
        ]);
        // Hull area == pixel area (9) → perfectly solid.
        assert!((roi.get_solidity() - 1.0).abs() < 1e-4);
    }

    #[test]
    fn solidity_below_one_for_concave_plus_shape() {
        // A plus/cross: concave, so the convex hull is larger than the pixel area.
        let roi = roi_from_pattern(&[
            &[false, true, false],
            &[true, true, true],
            &[false, true, false],
        ]);
        let s = roi.get_solidity();
        assert!(
            s > 0.0 && s < 1.0,
            "plus shape solidity should be in (0,1): {s}"
        );
    }

    #[test]
    fn solidity_one_for_single_pixel() {
        let roi = roi_from_pattern(&[&[true]]);
        assert_eq!(roi.get_solidity(), 1.0);
    }
}
