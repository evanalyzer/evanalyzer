//! # Area occupied
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-10-22
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **AGPL-3.0**.

use crate::algos::{ExecutionScope, GlobalPipelineCache, ImageAlgorithm, PipelineContext};
use crate::object::Object;
use crate::pipeline::pipeline_measure::{MeasurementRegion, Outputs};
use evanalyzer_cfg::core_types::{
    CitationMetadata, InternalErrors, MeasurementInfo, MeasurementTarget, MeasurementUnit,
    ObjectClass,
};
use macros::CommandsMeta;
use std::collections::{HashMap, HashSet};

/// Measures how much of the image the objects of a class cover.
///
/// The area in pixels and µm² (with a known pixel size) and the part of the
/// image (0..1). Where objects overlap, the area counts once.
#[derive(CommandsMeta)]
#[cmdsmeta(category = "measure", display_name = "Area Occupied")]
pub struct AreaOccupied {
    /// The step's ID, unique in the project - its values' keys start with
    /// it ("area_occupied_01/...").
    #[cmdsmeta(measurement_id = "area_occupied", display_name = "ID", summary = true)]
    pub id: String,

    /// The objects whose area is measured.
    #[cmdsmeta(summary = true)]
    pub object_class: ObjectClass,
}

impl ImageAlgorithm for AreaOccupied {
    fn execute(
        &self,
        ctx: &mut PipelineContext,
        cache: &mut GlobalPipelineCache,
    ) -> Result<(), InternalErrors> {
        if self.object_class == ObjectClass::Unset {
            return Err(InternalErrors::InvalidArgument(
                "Area occupied: choose the objects' class".into(),
            ));
        }
        let covered_px = covered_pixels(
            cache
                .object_cache
                .values()
                .filter(|o| o.is_of(&self.object_class, ctx.scope)),
        ) as f64;
        let image_px = analysed_pixels(cache) as f64;
        let px = &cache.image_meta.pixel_sizes;
        // nm² per pixel -> µm²; no value without a pixel size.
        let um2_per_px = (px.px_size_x > 0.0 && px.px_size_y > 0.0)
            .then(|| px.px_size_x as f64 * px.px_size_y as f64 / 1e6);

        let values = [
            ("area_px", Some(covered_px)),
            ("area_um2", um2_per_px.map(|f| covered_px * f)),
            ("fraction", (image_px > 0.0).then(|| covered_px / image_px)),
        ];
        for (id, value) in values {
            cache.add_image_measurement(
                self.measurement_output(id),
                value.map(|v| (MeasurementRegion::Image, v)),
            );
        }
        Ok(())
    }

    fn name(&self) -> &'static str {
        "Area occupied"
    }

    fn cite(&self) -> Vec<&'static CitationMetadata> {
        vec![&CitationMetadata::DANMAYR]
    }

    /// The objects of the whole image, after they are all found.
    fn execution_scope(&self) -> ExecutionScope {
        ExecutionScope::WholeImage
    }

    fn measures(&self) -> bool {
        true
    }

    fn measurement_outputs(&self) -> Vec<MeasurementInfo> {
        let outputs = Outputs::of(&self.id);
        vec![
            outputs.image("area_px", "Area occupied", MeasurementUnit::SquarePixels),
            outputs.image(
                "area_um2",
                "Area occupied",
                MeasurementUnit::SquareMicrometers,
            ),
            outputs
                .image(
                    "fraction",
                    "Part of the image occupied",
                    MeasurementUnit::Fraction,
                )
                .range(0.0, 1.0),
        ]
    }
}

/// The pixels at least one of `objects` covers: their masks' runs per row,
/// merged - exact also for overlapping objects, without a bitmap of the
/// whole image.
fn covered_pixels<'a>(objects: impl Iterator<Item = &'a Object>) -> u64 {
    let mut runs: HashMap<u32, Vec<(u32, u32)>> = HashMap::new();
    for object in objects {
        let [x_min, y_min, x_max, y_max] = object.bbox;
        if x_max < x_min || y_max < y_min {
            continue;
        }
        // bbox[2]/[3] are inclusive; the mask is row-major over the bbox.
        let width = (x_max - x_min + 1) as usize;
        for row in 0..=(y_max - y_min) {
            let start = row as usize * width;
            let Some(bits) = object.mask_data.get(start..start + width) else {
                break;
            };
            let row_runs = runs.entry(y_min + row).or_default();
            let mut run_start = None;
            for (x, bit) in bits.iter().by_vals().enumerate() {
                match (bit, run_start) {
                    (true, None) => run_start = Some(x as u32),
                    (false, Some(s)) => {
                        row_runs.push((x_min + s, x_min + x as u32));
                        run_start = None;
                    }
                    _ => {}
                }
            }
            if let Some(s) = run_start {
                row_runs.push((x_min + s, x_min + width as u32));
            }
        }
    }
    runs.into_values()
        .map(|mut row| {
            // Half-open runs, sorted: count each pixel once.
            row.sort_unstable();
            let (mut covered, mut end) = (0u64, 0u32);
            for (s, e) in row {
                let s = s.max(end);
                if e > s {
                    covered += (e - s) as u64;
                    end = e;
                }
            }
            covered
        })
        .sum()
}

/// The pixels the analysis looked at: its tiles (in a preview the visible
/// part), else the whole image.
fn analysed_pixels(cache: &GlobalPipelineCache) -> u64 {
    let tiles: HashSet<(usize, usize, usize, usize)> = cache
        .channel_keys()
        .map(|(_, t)| (t.offset_x, t.offset_y, t.width, t.height))
        .collect();
    if tiles.is_empty() {
        let size = &cache.image_meta.full_image_width;
        return (size.width * size.height) as u64;
    }
    tiles.into_iter().map(|(_, _, w, h)| (w * h) as u64).sum()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ImageTile;
    use crate::image::ImageContainer;
    use crate::image::{ManagedImage, Point2d};
    use crate::object::ObjectInit;
    use crate::pipeline::pipeline::PipelineImageMeta;
    use crate::pipeline::pipeline_measure::MeasurementValues;
    use bitvec::prelude::*;
    use kornia_image::{Image, ImageSize};
    use std::sync::Arc;

    /// A filled `w`×`h` rectangle at (`x`, `y`).
    fn rect(x: u32, y: u32, w: u32, h: u32) -> Object {
        Object::new(ObjectInit {
            bbox: [x, y, x + w - 1, y + h - 1],
            mask_data: bitvec![u64, Lsb0; 1; (w * h) as usize],
            area: (w * h) as usize,
            ..Default::default()
        })
    }

    #[test]
    fn overlapping_objects_count_once() {
        // Two 4x4 squares overlapping in a 2x2 corner: 16 + 16 - 4.
        let a = rect(0, 0, 4, 4);
        let b = rect(2, 2, 4, 4);
        assert_eq!(covered_pixels([&a, &b].into_iter()), 28);
        assert_eq!(covered_pixels([&a, &a].into_iter()), 16);
        assert_eq!(covered_pixels(std::iter::empty()), 0);
    }

    /// Only the mask's pixels, not the bbox's: a ring around a hole.
    #[test]
    fn holes_in_the_mask_arent_covered() {
        let mut ring = rect(10, 10, 3, 3);
        ring.mask_data.set(4, false);
        assert_eq!(covered_pixels([&ring].into_iter()), 8);
    }

    fn run(step: &AreaOccupied, cache: &mut GlobalPipelineCache) -> Result<(), InternalErrors> {
        let mut ctx =
            PipelineContext::without_image(std::path::PathBuf::new(), PipelineImageMeta::default())
                .unwrap();
        step.execute(&mut ctx, cache)
    }

    #[test]
    fn measures_area_and_part_of_the_image() {
        let class = ObjectClass::Global(1);
        let mut cache = GlobalPipelineCache::default();
        // A 10x10 image of 1 µm pixels.
        cache.add_to_channel_cache(
            Arc::new(ImageContainer::F32Gray(ManagedImage {
                data: Image::new(
                    ImageSize {
                        width: 10,
                        height: 10,
                    },
                    vec![0.0; 100],
                )
                .unwrap(),
                tile_offset: Point2d { x: 0, y: 0 },
                plane: None,
            })),
            0,
            ImageTile {
                offset_x: 0,
                offset_y: 0,
                width: 10,
                height: 10,
            },
        );
        cache.image_meta.pixel_sizes.px_size_x = 1000.0;
        cache.image_meta.pixel_sizes.px_size_y = 1000.0;
        let mut ctx =
            PipelineContext::without_image(std::path::PathBuf::new(), PipelineImageMeta::default())
                .unwrap();
        for (id, (mut object, of_class)) in [(rect(0, 0, 4, 5), true), (rect(5, 5, 2, 2), false)]
            .into_iter()
            .enumerate()
        {
            object.id = evanalyzer_cfg::core_types::ObjectId(id as u128 + 1);
            if of_class {
                object.add_object_class(class, ctx.scope).unwrap();
            }
            cache.object_cache.insert(object.id.clone(), object);
        }
        let step = AreaOccupied {
            id: "nuclei_area".into(),
            object_class: class,
        };
        step.execute(&mut ctx, &mut cache).unwrap();

        let values: Vec<(String, Option<f64>)> = cache
            .measurements
            .measurements
            .iter()
            .map(|m| match &m.values {
                MeasurementValues::Image(v) => (m.info.key.clone(), v.first().map(|(_, v)| *v)),
                MeasurementValues::Objects(_) => panic!("an image measurement"),
            })
            .collect();
        assert_eq!(
            values,
            [
                ("nuclei_area/area_px".into(), Some(20.0)),
                ("nuclei_area/area_um2".into(), Some(20.0)),
                ("nuclei_area/fraction".into(), Some(0.2)),
            ]
        );
    }

    #[test]
    fn without_a_pixel_size_theres_no_um2() {
        let mut cache = GlobalPipelineCache::default();
        let step = AreaOccupied {
            id: "a".into(),
            object_class: ObjectClass::Global(1),
        };
        run(&step, &mut cache).unwrap();
        let um2 = &cache.measurements.measurements[1];
        assert_eq!(um2.info.unit, MeasurementUnit::SquareMicrometers);
        assert_eq!(um2.values, MeasurementValues::Image(vec![]));
    }

    #[test]
    fn the_class_must_be_chosen() {
        let mut cache = GlobalPipelineCache::default();
        let step = AreaOccupied {
            id: "a".into(),
            object_class: ObjectClass::Unset,
        };
        assert!(run(&step, &mut cache).is_err());
    }
}
