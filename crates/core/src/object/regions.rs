//! # regions
//!
//! Measuring masks on an image read region by region - manual annotations
//! (drawn, copied or imported) and the rings around objects alike: the
//! regions to read, and the ring geometry the AI object classifier uses in
//! training and in the pipeline step.
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-10-25
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.

use super::intensities::{ChannelTile, IntensityAccumulator, MaskIntensities};
use crate::ai_learning::utils::{bbox_overlaps_tile, tile_grid};
use crate::algos::libs::feature_transform::nearest_features;
use crate::image::ImageChannel;
use bitvec::prelude::{BitVec, Lsb0};
use evanalyzer_cfg::core_types::{ImageTile, InternalErrors};
use evanalyzer_cfg::settings::object_settings::ObjectMetricSettings;
use std::collections::BTreeMap;

/// Side of the cells the reads are planned in: a cell with masks is read
/// cut to them - small reads also when the image is on a server.
const CELL: usize = 1024;

/// The regions to read for `members` (indices into `objects`) of an image
/// of `full_width` x `full_height`: each grid cell with masks, cut to the
/// bounding box of the mask pixels in it.
pub(crate) fn plan_reads(
    objects: &[ObjectMetricSettings],
    members: &[usize],
    full_width: usize,
    full_height: usize,
) -> Vec<ImageTile> {
    tile_grid(full_width, full_height, CELL)
        .into_iter()
        .filter_map(|cell| {
            let (mut x0, mut y0, mut x1, mut y1) = (usize::MAX, usize::MAX, 0, 0);
            for &i in members {
                let b = objects[i].bbox;
                if !bbox_overlaps_tile(b, &cell) {
                    continue;
                }
                x0 = x0.min((b[0] as usize).max(cell.offset_x));
                y0 = y0.min((b[1] as usize).max(cell.offset_y));
                x1 = x1.max((b[2] as usize).min(cell.offset_x + cell.width - 1));
                y1 = y1.max((b[3] as usize).min(cell.offset_y + cell.height - 1));
            }
            (x0 <= x1 && y0 <= y1).then(|| ImageTile {
                offset_x: x0,
                offset_y: y0,
                width: x1 - x0 + 1,
                height: y1 - y0 + 1,
            })
        })
        .collect()
}

/// Measures `objects` of an image of `full_width` x `full_height`, each on
/// its own z and t plane: every channel under each mask. `read(z, t,
/// region)` reads all channels of a region at full resolution; only regions
/// with masks are read, one at a time.
pub fn measure_on_planes(
    objects: &[ObjectMetricSettings],
    full_width: usize,
    full_height: usize,
    mut read: impl FnMut(i32, i32, ImageTile) -> Result<Vec<ImageChannel>, InternalErrors>,
) -> Result<Vec<IntensityAccumulator>, InternalErrors> {
    // The objects of each plane together: one read per region and plane.
    let mut by_plane: BTreeMap<(i32, i32), Vec<usize>> = BTreeMap::new();
    for (i, object) in objects.iter().enumerate() {
        by_plane
            .entry((object.z_stack.max(0), object.t_stack.max(0)))
            .or_default()
            .push(i);
    }
    let mut sums = vec![IntensityAccumulator::default(); objects.len()];
    for ((z, t), members) in &by_plane {
        for region in plan_reads(objects, members, full_width, full_height) {
            let channels = read(*z, *t, region)?;
            let tiles: Vec<ChannelTile<'_>> = channels
                .iter()
                .filter_map(|c| ChannelTile::from_container(c.c_stack, region, &c.image))
                .collect();
            for &i in members {
                if bbox_overlaps_tile(objects[i].bbox, &region) {
                    objects[i].add_intensities(&tiles, &mut sums[i]);
                }
            }
        }
    }
    Ok(sums)
}

/// The ring around a mask (`mask` relative to `bbox` = `[x1, y1, x2, y2]`,
/// inclusive): every pixel at most `width` px away from it (Euclidean),
/// minus the mask itself, cut to the image of `full_width` x
/// `full_height`. Neighbouring objects in the ring count - it is the
/// surroundings as they are. Returns the ring's mask and bounding box (the
/// object's grown by `width`).
pub fn ring_mask(
    mask: &BitVec<u64, Lsb0>,
    bbox: [u32; 4],
    width: u32,
    full_width: usize,
    full_height: usize,
) -> (BitVec<u64, Lsb0>, [u32; 4]) {
    let [x0, y0, x1, y1] = bbox.map(|v| v as usize);
    let w = width as usize;
    let (rx0, ry0) = (x0.saturating_sub(w), y0.saturating_sub(w));
    let rx1 = (x1 + w).min(full_width.max(1) - 1);
    let ry1 = (y1 + w).min(full_height.max(1) - 1);
    let (rw, rh) = (rx1 - rx0 + 1, ry1 - ry0 + 1);
    let mask_width = x1 - x0 + 1;
    let in_mask = |x: usize, y: usize| {
        // (x, y) in ring coordinates.
        let (gx, gy) = (x + rx0, y + ry0);
        (x0..=x1).contains(&gx)
            && (y0..=y1).contains(&gy)
            && mask
                .get((gy - y0) * mask_width + (gx - x0))
                .is_some_and(|b| *b)
    };
    let nearest = nearest_features(rw, rh, in_mask);
    let max_d2 = (w * w) as i64;
    let ring: BitVec<u64, Lsb0> = (0..rw * rh)
        .map(|i| {
            let (x, y) = (i % rw, i / rw);
            !in_mask(x, y)
                && nearest.get(i).is_some_and(|(r, c)| {
                    let (dy, dx) = (r - y as i64, c - x as i64);
                    dx * dx + dy * dy <= max_d2
                })
        })
        .collect();
    (ring, [rx0 as u32, ry0 as u32, rx1 as u32, ry1 as u32])
}

/// The ring (see [`ring_mask`]) around `object`, on its plane, to measure
/// like the object itself.
pub fn ring_of(
    object: &ObjectMetricSettings,
    width: u32,
    full_width: usize,
    full_height: usize,
) -> ObjectMetricSettings {
    let (mask_data, bbox) = ring_mask(
        &object.mask_data,
        object.bbox,
        width,
        full_width,
        full_height,
    );
    ObjectMetricSettings {
        bbox,
        area: mask_data.count_ones(),
        mask_data,
        z_stack: object.z_stack,
        t_stack: object.t_stack,
        ..Default::default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bitvec::bitvec;

    fn object(bbox: [u32; 4]) -> ObjectMetricSettings {
        ObjectMetricSettings {
            bbox,
            ..Default::default()
        }
    }

    #[test]
    fn only_the_regions_with_masks_are_read() {
        let objects = vec![
            object([10, 20, 30, 40]),
            object([50, 25, 60, 30]),
            object([1500, 10, 1510, 12]),
        ];
        let tiles = plan_reads(&objects, &[0, 1, 2], 4000, 3000);
        assert_eq!(
            tiles,
            vec![
                ImageTile {
                    offset_x: 10,
                    offset_y: 20,
                    width: 51,
                    height: 21
                },
                ImageTile {
                    offset_x: 1500,
                    offset_y: 10,
                    width: 11,
                    height: 3
                },
            ]
        );
        // An object over a cell border: read in both cells, each part once.
        let tiles = plan_reads(&[object([1000, 0, 1100, 5])], &[0], 4000, 3000);
        assert_eq!(tiles.len(), 2);
        assert_eq!(tiles.iter().map(|t| t.width).sum::<usize>(), 101);
    }

    /// Rows of a mask as text, `#` set.
    fn draw(mask: &BitVec<u64, Lsb0>, width: usize) -> Vec<String> {
        mask.iter()
            .by_vals()
            .collect::<Vec<_>>()
            .chunks(width)
            .map(|row| row.iter().map(|&b| if b { '#' } else { '.' }).collect())
            .collect()
    }

    #[test]
    fn a_ring_is_every_pixel_within_its_width_of_the_object_but_not_the_object() {
        // One pixel at (10, 10), ring 2 px: a disc of radius 2 without its centre.
        let (ring, bbox) = ring_mask(&bitvec![u64, Lsb0; 1], [10, 10, 10, 10], 2, 100, 100);
        assert_eq!(bbox, [8, 8, 12, 12]);
        assert_eq!(
            draw(&ring, 5),
            vec!["..#..", ".###.", "##.##", ".###.", "..#.."]
        );
    }

    #[test]
    fn a_ring_follows_the_shape_and_stops_at_the_image_border() {
        // An L in the top-left corner of a 6 x 6 image, ring 1 px.
        let l = bitvec![u64, Lsb0; 1, 0, 1, 1];
        let (ring, bbox) = ring_mask(&l, [0, 0, 1, 1], 1, 6, 6);
        assert_eq!(bbox, [0, 0, 2, 2]);
        // The L's own pixels and the far corner (2, 2) (√2 px away) stay out;
        // (1, 0), the L's notch, is in.
        assert_eq!(draw(&ring, 3), vec![".#.", "..#", "##."]);
    }

    #[test]
    fn the_ring_keeps_the_objects_plane() {
        let mut o = object([5, 5, 5, 5]);
        o.mask_data = bitvec![u64, Lsb0; 1];
        o.z_stack = 3;
        o.t_stack = 2;
        let ring = ring_of(&o, 1, 20, 20);
        assert_eq!((ring.z_stack, ring.t_stack, ring.area), (3, 2, 4));
    }
}
