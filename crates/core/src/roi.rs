use bitvec::prelude::*;
use evanalyzer_cfg::{
    core_types::{ObjectClass, ObjectId, SegmentationClass, TrackId},
    settings::roi_settings::{IntensitySettings, RoiSettings, TrackSettings},
};
use indexmap::IndexMap;
use std::collections::HashSet;

use crate::ImagePlane;

#[derive(Debug, Default, Clone)]
pub struct Intensity {
    /// Sum of all pixel intensities in the ROI
    pub sum_intensity: f64,
    /// Minimum pixel intensity in the ROI
    pub min_intensity: f32,
    /// Maximum pixel intensity in the ROI
    pub max_intensity: f32,
    /// Median pixel intensity in the ROI
    pub median_intensity: Option<f32>,
    /// Standard deviation of pixel intensities
    pub std_dev: Option<f32>,
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
        let width = (x_max - x_min) as usize;
        let height = (y_max - y_min) as usize;

        if width == 0 || height == 0 || self.area == 0 {
            return 0.0;
        }

        let mut perimeter = 0.0;
        const SQRT2: f32 = 1.414_213_562_373_095_0;

        // Iterate through each pixel in the bounding box
        for y in 0..height {
            for x in 0..width {
                let bit_index = (y * width) + x;
                let is_inside = self.mask_data.get(bit_index).map(|b| *b).unwrap_or(false);

                if !is_inside {
                    continue;
                }

                // Check 8-connected neighbors
                // For each neighbor that is outside the mask, we have a boundary
                for dy in -1..=1i32 {
                    for dx in -1..=1i32 {
                        if dx == 0 && dy == 0 {
                            continue;
                        }

                        let nx = (x as i32) + dx;
                        let ny = (y as i32) + dy;

                        // Check if neighbor is outside bounds or outside the mask
                        let is_neighbor_inside =
                            if nx >= 0 && nx < width as i32 && ny >= 0 && ny < height as i32 {
                                let neighbor_index = ((ny as usize) * width) + (nx as usize);
                                self.mask_data
                                    .get(neighbor_index)
                                    .map(|b| *b)
                                    .unwrap_or(false)
                            } else {
                                false // Out of bounds is considered "outside"
                            };

                        if !is_neighbor_inside {
                            // This neighbor is outside, so we have a boundary edge
                            if dx == 0 || dy == 0 {
                                // Horizontal or vertical edge
                                perimeter += 1.0;
                            } else {
                                // Diagonal edge
                                perimeter += SQRT2;
                            }
                        }
                    }
                }
            }
        }

        // ImageJ divides by 2 because we count each edge twice (from both sides)
        perimeter / 2.0
    }

    /// Calculates solidity: ratio of area to convex hull area.
    /// Values range from 0 to 1 (1 = perfectly convex).
    pub fn get_solidity(&self) -> f32 {
        // Approximate using area / (perimeter² / 4π)
        // A perfect circle has the minimum perimeter for given area
        let min_perimeter_sq = (4.0 * std::f32::consts::PI * self.area as f32).powi(2);
        (self.area as f32 * 4.0 * std::f32::consts::PI) / min_perimeter_sq
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
                            median_intensity: v.median_intensity,
                            std_dev: v.std_dev,
                            pixel_values: vec![],
                        },
                    )
                })
                .collect(),
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
                            median_intensity: v.median_intensity,
                            std_dev: v.std_dev,
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
}
