//! # annotation_measure
//!
//! Measuring manual annotations on their image - drawn, copied or imported
//! ones alike: every channel under each mask, at full resolution, through
//! the backend's [`ImageSource`] (this computer's files or the server's).
//! What to read and the pixel math are core's ([`measure_on_planes`], the
//! same as for training an object classifier); here only reading them.
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-10-19
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.

use crate::api::{ImageSource, TileRequest};
use evanalyzer_cfg::core_types::{InternalErrors, ZProjection};
use evanalyzer_cfg::settings::object_settings::ObjectMetricSettings;
use evanalyzer_core::measure_on_planes;

/// Measures `objects` (annotations of `series`) on `source`, each on its own
/// image - its plane, or the z-projection it was drawn on, of its frame:
/// replaces their intensities. Reads only the regions
/// with annotations, one at a time.
pub fn measure_on_source(
    source: &dyn ImageSource,
    series: i32,
    objects: &mut [ObjectMetricSettings],
) -> Result<(), InternalErrors> {
    if objects.is_empty() {
        return Ok(());
    }
    let full = source
        .meta()
        .series
        .get(&series)
        .and_then(|s| s.resolutions.get(&0))
        .ok_or_else(|| InternalErrors::Generic(format!("The image has no series {series}")))?;
    let sums = measure_on_planes(
        objects,
        full.width as usize,
        full.height as usize,
        |z_projection, z, t, tile| {
            source.read_tile(&TileRequest {
                series,
                resolution_idx: 0,
                z_projection,
                // A projection over every plane, as the pipeline reads it.
                z_range: (z_projection == ZProjection::None).then_some(z..=z),
                t_stack: t,
                tile,
            })
        },
    )?;
    for (object, sum) in objects.iter_mut().zip(sums) {
        object.intensities = sum.finish_settings();
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::local::image_reader::open_image_source;
    use bitvec::prelude::*;
    use evanalyzer_cfg::core_types::ImagePlane;
    use evanalyzer_cfg::core_types::ImageTile;
    use evanalyzer_core::ChannelTile;
    use kornia_image::ImageSize;
    use std::collections::HashSet;
    use std::path::PathBuf;

    fn fixture() -> PathBuf {
        PathBuf::from(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../core/tests/slice_Z0_C0_T0.tif"
        ))
    }

    #[test]
    fn an_annotation_is_measured_on_its_own_pixels_of_the_image() {
        let source = open_image_source(&fixture()).unwrap();
        let full = source.meta().series[&0].resolutions[&0].clone();
        let (w, h) = (full.width as usize, full.height as usize);
        // The whole image at once, for the expected values.
        let whole = source
            .read_tile(&TileRequest {
                series: 0,
                resolution_idx: 0,
                z_projection: ZProjection::None,
                z_range: Some(0..=0),
                t_stack: 0,
                tile: ImageTile {
                    offset_x: 0,
                    offset_y: 0,
                    width: w,
                    height: h,
                },
            })
            .unwrap();

        // A plus-shaped 3x3 mask somewhere inside, and a 1-pixel object.
        let plus = bitvec![u64, Lsb0; 0, 1, 0, 1, 1, 1, 0, 1, 0];
        let (x, y) = ((w / 3) as u32, (h / 2) as u32);
        let shapes = [
            (plus, [x, y, x + 2, y + 2]),
            (bitvec![u64, Lsb0; 1], [5, 7, 5, 7]),
        ];
        let mut annotations: Vec<ObjectMetricSettings> = shapes
            .iter()
            .map(|(mask, bbox)| {
                evanalyzer_core::Object::from_mask_geometry(
                    &ImageSize {
                        width: w,
                        height: h,
                    },
                    mask.clone(),
                    *bbox,
                    ImagePlane { z: 0, c: 0, t: 0 },
                    HashSet::new(),
                )
                .to_object_settings()
            })
            .collect();
        measure_on_source(&source, 0, &mut annotations).unwrap();

        for ((mask, bbox), measured) in shapes.iter().zip(&annotations) {
            assert!(!measured.intensities.is_empty());
            let width = (bbox[2] - bbox[0] + 1) as usize;
            for channel in &whole {
                let expected_tile = ChannelTile::from_container(
                    channel.c_stack,
                    ImageTile {
                        offset_x: 0,
                        offset_y: 0,
                        width: w,
                        height: h,
                    },
                    &channel.image,
                )
                .unwrap();
                let values: Vec<f64> = mask
                    .iter()
                    .by_vals()
                    .enumerate()
                    .filter(|(_, set)| *set)
                    .map(|(i, _)| {
                        let px = bbox[0] as usize + i % width;
                        let py = bbox[1] as usize + i / width;
                        evanalyzer_core::sample_channel_pixel(
                            expected_tile.is_rgb,
                            expected_tile.samples,
                            py * w + px,
                        ) as f64
                    })
                    .collect();
                let got = &measured.intensities[&channel.c_stack];
                let sum: f64 = values.iter().sum();
                assert!((got.sum_intensity - sum).abs() < 1e-6);
                assert_eq!(
                    got.min_intensity,
                    values.iter().cloned().fold(f64::MAX, f64::min)
                );
                assert_eq!(
                    got.max_intensity,
                    values.iter().cloned().fold(f64::MIN, f64::max)
                );
                assert!((got.avg_intensity - sum / values.len() as f64).abs() < 1e-9);
            }
        }
        assert!(
            measure_on_source(&source, 7, &mut annotations).is_err(),
            "no series 7"
        );
    }

    /// What measuring a drawn annotation costs on a large image: time per
    /// annotation (opened image reused, as the app does) for a small, a
    /// medium and a large shape, and the RAM it takes. Run with:
    ///   EVANALYZER_BENCH_SLIDE=/path/slide.tif cargo test --release -p evanalyzer_app \\
    ///     --lib bench_measure_drawn -- --ignored --nocapture
    #[test]
    #[ignore]
    fn bench_measure_drawn() {
        use std::time::Instant;
        let path = PathBuf::from(std::env::var("EVANALYZER_BENCH_SLIDE").unwrap());
        let source = open_image_source(&path).unwrap();
        let full = source.meta().series[&0].resolutions[&0].clone();
        let (w, h) = (full.width as usize, full.height as usize);
        let rss = || {
            std::fs::read_to_string("/proc/self/status")
                .unwrap()
                .lines()
                .find(|l| l.starts_with("VmRSS:"))
                .and_then(|l| l.split_whitespace().nth(1)?.parse::<u64>().ok())
                .unwrap_or(0)
                / 1024
        };
        println!("image {w} x {h}, RSS at start {} MB", rss());
        for side in [40u32, 400, 2000] {
            let mut times = Vec::new();
            for i in 0..5u32 {
                // A filled circle at a new place each time (no cache help).
                let (x0, y0) = (1000 + i * 7000, 1000 + i * 5000);
                let r = side as f64 / 2.0;
                let mask: BitVec<u64, Lsb0> = (0..side * side)
                    .map(|k| {
                        let (x, y) = ((k % side) as f64 + 0.5 - r, (k / side) as f64 + 0.5 - r);
                        x * x + y * y <= r * r
                    })
                    .collect();
                let mut objects = vec![
                    evanalyzer_core::Object::from_mask_geometry(
                        &ImageSize {
                            width: w,
                            height: h,
                        },
                        mask,
                        [x0, y0, x0 + side - 1, y0 + side - 1],
                        ImagePlane { z: 0, c: 0, t: 0 },
                        HashSet::new(),
                    )
                    .to_object_settings(),
                ];
                let start = Instant::now();
                measure_on_source(&source, 0, &mut objects).unwrap();
                times.push(start.elapsed().as_secs_f64() * 1000.0);
                if i == 4 {
                    // The read alone, for comparison.
                    let start = Instant::now();
                    let region = ImageTile {
                        offset_x: x0 as usize,
                        offset_y: y0 as usize,
                        width: side as usize,
                        height: side as usize,
                    };
                    let _ = source
                        .read_tile(&TileRequest {
                            series: 0,
                            resolution_idx: 0,
                            z_projection: ZProjection::None,
                            z_range: Some(0..=0),
                            t_stack: 0,
                            tile: region,
                        })
                        .unwrap();
                    println!(
                        "      read alone: {:.1} ms",
                        start.elapsed().as_secs_f64() * 1000.0
                    );
                }
            }
            times.sort_by(f64::total_cmp);
            println!(
                "{side:>5} px shape: median {:.1} ms (min {:.1}, max {:.1}), RSS {} MB",
                times[2],
                times[0],
                times[4],
                rss()
            );
        }
    }
}
