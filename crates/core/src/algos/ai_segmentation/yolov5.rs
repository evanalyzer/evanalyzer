//! # yolov5
//!
//! **Author:** Joachim Danmayr
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **AGPL-3.0**.

use std::path::PathBuf;

use evanalyzer_ai_engine::{Detection, Image as AiImage, Request, Response, Task, box_iou};
use evanalyzer_cfg::core_types::{CitationMetadata, InternalErrors, SegmentationClass};
use kornia_image::Image;
use macros::CommandsMeta;

use crate::{
    ImageContainer, ai_devices,
    algos::{ExecutionScope, ImageAlgorithm},
    pipeline::{pipeline_cache::GlobalPipelineCache, pipeline_context::PipelineContext},
};

/// Which project segmentation class the objects of one model class get.
#[derive(CommandsMeta)]
pub struct YoloClassMapping {
    /// Index of the class in the model (`0` = its first class).
    #[cmdsmeta(default = 0, min = 0, max = 1000, step = 1)]
    pub model_class: i32,
    /// The project's segmentation class objects of `model_class` are written as.
    #[cmdsmeta(default = SegmentationClass(1))]
    pub segmentation_class: SegmentationClass,
}

/// Instance segmentation (or detection) with a YOLOv5 model exported as TorchScript.
///
/// [AI YOLOv5 Segmentation] -> [Extract Objects]
///
/// Takes a YOLOv5 TorchScript export (`export.py --include torchscript`) with
/// a fixed 640x640 input. Segmentation models (`yolov5*-seg`) give every
/// object its mask; plain detection models give filled boxes. Any number of
/// model classes is supported.
///
/// Tiles larger than 640 px are analyzed in overlapping 640x640 windows at
/// full resolution (small tiles are padded), so small objects stay
/// detectable; objects seen by several windows are merged. Gray images are
/// given to the model as RGB with equal channels.
#[derive(CommandsMeta)]
#[cmdsmeta(
    category = "segment",
    next = "extract",
    display_name = "AI YOLOv5 Segmentation"
)]
pub struct Yolov5 {
    /// Path to a YOLOv5 model exported as TorchScript.
    #[cmdsmeta(file_extensions = "pt,torchscript")]
    pub model_path: PathBuf,

    /// Maps the model's classes to this project's segmentation classes;
    /// objects of classes not listed are dropped. Leave empty to write model
    /// class `i` as segmentation class `i + 1`, for every class.
    pub class_mapping: Vec<YoloClassMapping>,

    /// Minimum confidence (objectness x class score) of a detection.
    #[cmdsmeta(default = 0.25, min = 0.0, max = 1.0, step = 0.01)]
    pub confidence_threshold: f32,

    /// Detections of the same class overlapping more than this (box
    /// intersection over union) are merged into the more confident one.
    #[cmdsmeta(default = 0.45, min = 0.0, max = 1.0, step = 0.01, visibility = Advanced)]
    pub iou_threshold: f32,

    /// Mask probability above which a pixel belongs to its object
    /// (segmentation models only).
    #[cmdsmeta(default = 0.5, min = 0.0, max = 1.0, step = 0.01, visibility = Advanced)]
    pub mask_threshold: f32,

    /// Factor the image is scaled by before it is given to the model, the
    /// masks are scaled back afterwards. Use it when the model was trained on
    /// downscaled images: `640 / training image size`, e.g. `0.3125` for
    /// 2048 px images YOLOv5 shrank to 640. `1` = full resolution.
    #[cmdsmeta(default = 1.0, min = 0.05, max = 4.0, step = 0.01, optional = true)]
    pub image_scale: f32,

    /// Overlap of neighboring 640x640 windows, in pixels. Must be larger
    /// than the biggest object, so every object lies completely inside some
    /// window.
    #[cmdsmeta(default = 128, min = 0, max = 512, step = 8, visibility = Advanced)]
    pub window_overlap: i32,

    /// Objects with fewer pixels than this (after overlapping objects were
    /// resolved) are removed. `0` keeps every object.
    #[cmdsmeta(default = 15, min = 0, max = 100000, step = 1)]
    pub min_object_size: i32,
}

impl ImageAlgorithm for Yolov5 {
    fn execute(
        &self,
        ctx: &mut PipelineContext,
        _cache: &mut GlobalPipelineCache,
    ) -> Result<(), InternalErrors> {
        let size = ctx.get_image_size();
        let (tile_width, tile_height) = (size.width, size.height);
        // Everything up to the label maps runs on the scaled image.
        let scale = self.image_scale.clamp(0.05, 4.0) as f64;
        let width = ((tile_width as f64 * scale).round() as usize).max(1);
        let height = ((tile_height as f64 * scale).round() as usize).max(1);
        let rgb = Self::rgb_planes(&ctx.image)?.concat();
        let request = Request {
            model: self.model_path.clone(),
            task: Task::Yolov5 {
                width,
                height,
                window_overlap: self.window_overlap,
                confidence_threshold: self.confidence_threshold,
                iou_threshold: self.iou_threshold,
                mask_threshold: self.mask_threshold,
            },
        };
        let image = AiImage {
            width: tile_width,
            height: tile_height,
            channels: 3,
            data: &rgb,
        };
        let Response::Yolov5 { detections } = ai_devices::infer(&request, image)? else {
            return Err(InternalErrors::Generic(
                "YOLOv5: the AI engine gave another kind of answer".into(),
            ));
        };
        let detections = Self::nms(detections, self.iou_threshold);

        let min_size = (self.min_object_size.max(0) as f64 * scale * scale).round() as usize;
        let (mut segmentation, mut instances) =
            self.rasterize(&detections, width, height, min_size);
        if (width, height) != (tile_width, tile_height) {
            let up = |labels: &[u32]| {
                super::resize_labels_nearest(labels, width, height, tile_width, tile_height)
            };
            segmentation = up(&segmentation);
            instances = up(&instances);
        }
        let to_map = |data: Vec<u32>| {
            Image::<u32, 1>::new(size, data)
                .map_err(|e| InternalErrors::Generic(format!("YOLOv5: {e}")))
        };
        ctx.segmentation_map = Some(to_map(segmentation)?);
        ctx.instance_map = Some(to_map(instances)?);
        Ok(())
    }

    fn name(&self) -> &'static str {
        "YOLOv5"
    }

    fn cite(&self) -> Vec<&'static CitationMetadata> {
        vec![&CitationMetadata {
            cite_key: "jocher2022yolov5",
            title: "ultralytics/yolov5: v7.0 - YOLOv5 SOTA Realtime Instance Segmentation",
            authors: &["Glenn Jocher", "Ayush Chaurasia", "Alex Stoken"],
            year: 2022,
            container: Some("Zenodo"),
            doi: Some("10.5281/zenodo.3908559"),
            url: Some("https://doi.org/10.5281/zenodo.3908559"),
            pages: None,
        }]
    }

    fn execution_scope(&self) -> ExecutionScope {
        ExecutionScope::Tile
    }
}

impl Yolov5 {
    /// The image as three `width * height` planes (R, G, B); a gray image
    /// gives three equal planes.
    fn rgb_planes(image: &ImageContainer) -> Result<[Vec<f32>; 3], InternalErrors> {
        match image {
            ImageContainer::F32Gray(img) => {
                let plane = img.data.as_slice().to_vec();
                Ok([plane.clone(), plane.clone(), plane])
            }
            ImageContainer::F32Rgb(img) => {
                let data = img.data.as_slice();
                let plane = |c: usize| data.iter().skip(c).step_by(3).copied().collect();
                Ok([plane(0), plane(1), plane(2)])
            }
            other => Err(InternalErrors::FormatMismatch {
                expected: "F32Gray or F32Rgb".into(),
                found: format!("{other:?}"),
            }),
        }
    }

    /// Greedy class-aware NMS over all windows' detections (box IoU): of
    /// overlapping detections of one class only the most confident stays.
    fn nms(mut detections: Vec<Detection>, iou_threshold: f32) -> Vec<Detection> {
        detections.sort_by(|a, b| b.score.total_cmp(&a.score));
        let mut kept: Vec<Detection> = Vec::new();
        for d in detections {
            if !kept
                .iter()
                .any(|k| k.class == d.class && box_iou(&k.bbox, &d.bbox) > iou_threshold)
            {
                kept.push(d);
            }
        }
        kept
    }

    /// The project segmentation class of a model class, `None` = drop it.
    fn segmentation_class(&self, model_class: usize) -> Option<u32> {
        if self.class_mapping.is_empty() {
            return Some(model_class as u32 + 1);
        }
        self.class_mapping
            .iter()
            .find(|m| m.model_class.max(0) as usize == model_class)
            .map(|m| m.segmentation_class.as_u32())
    }

    /// Paints the detections (`detections` sorted by descending score) into
    /// segmentation and instance maps: more confident objects lie on top,
    /// objects left smaller than `min_size` pixels are removed, instance ids
    /// are `1..=n`.
    fn rasterize(
        &self,
        detections: &[Detection],
        width: usize,
        height: usize,
        min_size: usize,
    ) -> (Vec<u32>, Vec<u32>) {
        let mut owner = vec![0u32; width * height]; // detection index + 1
        for (i, d) in detections.iter().enumerate().rev() {
            if self.segmentation_class(d.class).is_none() {
                continue;
            }
            let [rx, ry, rw, rh] = d.mask_rect;
            for y in 0..rh {
                for x in 0..rw {
                    let filled = d.mask.as_ref().map(|m| m[y * rw + x]).unwrap_or(true);
                    if filled && ry + y < height && rx + x < width {
                        owner[(ry + y) * width + rx + x] = i as u32 + 1;
                    }
                }
            }
        }

        let mut sizes = vec![0usize; detections.len() + 1];
        for &o in &owner {
            sizes[o as usize] += 1;
        }
        let mut instance_id = vec![0u32; detections.len() + 1];
        let mut next = 1;
        for i in 1..=detections.len() {
            if sizes[i] > 0 && sizes[i] >= min_size {
                instance_id[i] = next;
                next += 1;
            }
        }

        let mut segmentation = vec![0u32; width * height];
        let mut instances = vec![0u32; width * height];
        for (p, &o) in owner.iter().enumerate() {
            let id = instance_id[o as usize];
            if id != 0 {
                instances[p] = id;
                segmentation[p] = self
                    .segmentation_class(detections[o as usize - 1].class)
                    .unwrap_or(0);
            }
        }
        (segmentation, instances)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(feature = "libtorch-in-process")]
    use evanalyzer_ai_engine::torch::test_support::{
        tch::{Kind, Tensor},
        trace_and_save_model,
    };
    use kornia_image::ImageSize;

    fn yolo() -> Yolov5 {
        Yolov5 {
            model_path: PathBuf::new(),
            class_mapping: vec![],
            confidence_threshold: 0.25,
            iou_threshold: 0.45,
            mask_threshold: 0.5,
            image_scale: 1.0,
            window_overlap: 128,
            min_object_size: 0,
        }
    }

    fn boxed(score: f32, class: usize, bbox: [f32; 4]) -> Detection {
        let rect = [
            bbox[0] as usize,
            bbox[1] as usize,
            (bbox[2] - bbox[0]) as usize,
            (bbox[3] - bbox[1]) as usize,
        ];
        Detection {
            score,
            class,
            bbox,
            mask: None,
            mask_rect: rect,
        }
    }

    #[test]
    fn an_empty_mapping_writes_every_model_class_one_up() {
        let cmd = yolo();
        assert_eq!(cmd.segmentation_class(0), Some(1));
        assert_eq!(cmd.segmentation_class(7), Some(8));
    }

    #[test]
    fn a_mapping_keeps_only_the_listed_classes() {
        let cmd = Yolov5 {
            class_mapping: vec![YoloClassMapping {
                model_class: 2,
                segmentation_class: SegmentationClass(5),
            }],
            ..yolo()
        };
        assert_eq!(cmd.segmentation_class(2), Some(5));
        assert_eq!(cmd.segmentation_class(0), None);
    }

    #[test]
    fn nms_is_class_aware_and_keeps_the_most_confident() {
        let kept = Yolov5::nms(
            vec![
                boxed(0.6, 0, [0.0, 0.0, 10.0, 10.0]),
                boxed(0.9, 0, [1.0, 0.0, 11.0, 10.0]),
                boxed(0.5, 1, [0.0, 0.0, 10.0, 10.0]),
            ],
            0.45,
        );
        let scores: Vec<f32> = kept.iter().map(|d| d.score).collect();
        assert_eq!(scores, vec![0.9, 0.5]);
    }

    #[test]
    fn rasterize_puts_confident_objects_on_top_and_drops_small_and_unmapped_ones() {
        // Sorted by score as `nms` leaves them.
        let detections = vec![
            boxed(0.9, 0, [2.0, 0.0, 6.0, 4.0]),  // on top of the next one
            boxed(0.8, 0, [0.0, 0.0, 4.0, 4.0]),  // keeps 8 of its 16 px
            boxed(0.7, 1, [8.0, 0.0, 10.0, 1.0]), // 2 px: too small
        ];
        let cmd = Yolov5 {
            min_object_size: 3,
            ..yolo()
        };
        let (seg, inst) = cmd.rasterize(&detections, 10, 4, 3);
        assert_eq!(
            inst[0], 2,
            "the less confident object keeps its uncovered part"
        );
        assert_eq!(
            inst[3], 1,
            "the overlap belongs to the more confident object"
        );
        assert_eq!(inst[8], 0, "objects below min_object_size are removed");
        assert_eq!(seg[3], 1);
        assert_eq!(inst.iter().copied().max(), Some(2));

        let mapped = Yolov5 {
            class_mapping: vec![YoloClassMapping {
                model_class: 1,
                segmentation_class: SegmentationClass(4),
            }],
            ..yolo()
        };
        let (seg, inst) = mapped.rasterize(&detections, 10, 4, 0);
        assert_eq!(inst[3], 0, "class 0 isn't mapped: dropped");
        assert_eq!((seg[8], inst[8]), (4, 1));
    }

    #[test]
    fn rgb_planes_split_rgb_and_repeat_gray() {
        let rgb = crate::pipeline::pipeline_context::PipelineContext::new_from_image_test_rgb(
            Image::<f32, 3>::new(
                ImageSize {
                    width: 2,
                    height: 1,
                },
                vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            )
            .unwrap(),
        )
        .unwrap();
        let planes = Yolov5::rgb_planes(&rgb.image).unwrap();
        assert_eq!(planes, [vec![1.0, 4.0], vec![2.0, 5.0], vec![3.0, 6.0]]);

        let gray = PipelineContext::new_from_image_test(
            Image::<f32, 1>::new(
                ImageSize {
                    width: 2,
                    height: 1,
                },
                vec![0.25, 0.75],
            )
            .unwrap(),
        )
        .unwrap();
        let planes = Yolov5::rgb_planes(&gray.image).unwrap();
        assert!(planes.iter().all(|p| p == &vec![0.25, 0.75]));
    }

    // ---- execute() with a traced detection model ----

    #[cfg(feature = "libtorch-in-process")]
    /// A detection model (single output) that always reports one 100x100
    /// box of class 1 at the window centre.
    fn centre_box_model() -> (tempfile::TempDir, PathBuf) {
        trace_and_save_model(3, 640, 640, |x| {
            // Built from the input with plain numbers: a tensor created inside
            // the traced closure (e.g. `Tensor::from_slice`) is saved as zeros.
            let one = x.mean(Kind::Float) * 0.0 + 1.0;
            let values = [320.0, 320.0, 100.0, 100.0, 0.9, 0.1, 0.9];
            let columns: Vec<Tensor> = values.iter().map(|&v| &one * v).collect();
            Tensor::stack(&columns, 0).reshape([1, 1, 7])
        })
    }

    fn objects(ctx: &PipelineContext) -> Vec<(u32, usize)> {
        let inst = ctx.get_instance_map().unwrap().as_slice();
        let seg = ctx.get_segmentation_map().unwrap().as_slice();
        let max = inst.iter().copied().max().unwrap_or(0);
        (1..=max)
            .map(|id| {
                let px: Vec<usize> = (0..inst.len()).filter(|&i| inst[i] == id).collect();
                (seg[px[0]], px.len())
            })
            .collect()
    }

    fn gray(width: usize, height: usize) -> PipelineContext {
        PipelineContext::new_from_image_test(
            Image::<f32, 1>::new(ImageSize { width, height }, vec![0.5; width * height]).unwrap(),
        )
        .unwrap()
    }

    #[cfg(feature = "libtorch-in-process")]
    #[test]
    fn execute_segments_a_gray_tile_with_a_detection_model() {
        let (_dir, model_path) = centre_box_model();
        let cmd = Yolov5 {
            model_path,
            ..yolo()
        };
        let mut ctx = gray(640, 640);
        cmd.execute(&mut ctx, &mut GlobalPipelineCache::default())
            .unwrap();
        // Model class 1 -> segmentation class 2; a filled 100x100 box.
        assert_eq!(objects(&ctx), vec![(2, 100 * 100)]);
    }

    #[cfg(feature = "libtorch-in-process")]
    #[test]
    fn execute_runs_one_detection_per_window_on_a_large_tile() {
        // 1000x700: windows at x 0 and 360, y 0 and 60 - each reports its own
        // centre box, so four objects at distinct places.
        let (_dir, model_path) = centre_box_model();
        let cmd = Yolov5 {
            model_path,
            ..yolo()
        };
        let mut ctx = gray(1000, 700);
        cmd.execute(&mut ctx, &mut GlobalPipelineCache::default())
            .unwrap();
        let found = objects(&ctx);
        assert_eq!(found.len(), 4, "{found:?}");
        assert!(found.iter().all(|&(class, _)| class == 2));
    }

    #[cfg(feature = "libtorch-in-process")]
    #[test]
    fn execute_scales_the_image_and_the_masks_back() {
        // A 1280x1280 RGB tile at scale 0.5 is one 640 window; the 100 px box
        // there is 200 px in the tile.
        let (_dir, model_path) = centre_box_model();
        let cmd = Yolov5 {
            model_path,
            image_scale: 0.5,
            ..yolo()
        };
        let mut ctx = PipelineContext::new_from_image_test_rgb(
            Image::<f32, 3>::new(
                ImageSize {
                    width: 1280,
                    height: 1280,
                },
                vec![0.5; 1280 * 1280 * 3],
            )
            .unwrap(),
        )
        .unwrap();
        cmd.execute(&mut ctx, &mut GlobalPipelineCache::default())
            .unwrap();
        assert_eq!(objects(&ctx), vec![(2, 200 * 200)]);
    }

    #[cfg(feature = "libtorch-in-process")]
    #[test]
    fn execute_drops_classes_the_mapping_leaves_out() {
        let (_dir, model_path) = centre_box_model();
        let cmd = Yolov5 {
            model_path,
            class_mapping: vec![YoloClassMapping {
                model_class: 0,
                segmentation_class: SegmentationClass(3),
            }],
            ..yolo()
        };
        let mut ctx = gray(640, 640);
        cmd.execute(&mut ctx, &mut GlobalPipelineCache::default())
            .unwrap();
        assert!(objects(&ctx).is_empty());
    }

    #[cfg(feature = "libtorch-in-process")]
    #[test]
    fn execute_explains_a_model_that_cannot_be_loaded() {
        let dir = tempfile::tempdir().unwrap();
        let cmd = Yolov5 {
            model_path: dir.path().join("missing.pt"),
            ..yolo()
        };
        let err = cmd
            .execute(&mut gray(64, 64), &mut GlobalPipelineCache::default())
            .unwrap_err();
        assert!(matches!(err, InternalErrors::Generic(msg) if msg.contains("TorchScript")));
    }

    /// Reads a little-endian `.npy` file of 4-byte values.
    fn read_npy_4byte(path: &std::path::Path) -> Vec<[u8; 4]> {
        let bytes = std::fs::read(path)
            .unwrap_or_else(|e| panic!("{}: {e} - see the test's doc comment", path.display()));
        let header_len = u16::from_le_bytes([bytes[8], bytes[9]]) as usize;
        bytes[10 + header_len..]
            .chunks_exact(4)
            .map(|c| [c[0], c[1], c[2], c[3]])
            .collect()
    }

    #[cfg(feature = "libtorch-in-process")]
    /// The University of Salzburg brightfield YOLOv5-seg model on a 640x640
    /// crop of `tests/cellpose_sam/B5_0009.jpg`, compared with YOLOv5's own
    /// post-processing (`non_max_suppression` + `process_mask`), re-implemented
    /// in `make_reference.py` next to the model. Model and data aren't in git
    /// (190 MB); they live in `tests/university_of_sbg_brightfield_cell_segmentation_v3/`.
    /// Run with:
    ///   cargo test -p evanalyzer_core --features libtorch-in-process --lib real_yolov5 -- --ignored
    #[test]
    #[ignore]
    fn real_yolov5_seg_matches_the_yolov5_reference() {
        let dir = PathBuf::from(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../tests/university_of_sbg_brightfield_cell_segmentation_v3"
        ));
        let crop: Vec<f32> = read_npy_4byte(&dir.join("test_crop_640_rgb.npy"))
            .into_iter()
            .map(f32::from_le_bytes)
            .collect();
        let reference: Vec<u32> = read_npy_4byte(&dir.join("test_crop_640_reference_labels.npy"))
            .into_iter()
            .map(|b| i32::from_le_bytes(b) as u32)
            .collect();
        let cmd = Yolov5 {
            model_path: dir.join("weights.pt"),
            ..yolo()
        };
        let mut ctx = PipelineContext::new_from_image_test_rgb(
            Image::<f32, 3>::new(
                ImageSize {
                    width: 640,
                    height: 640,
                },
                crop,
            )
            .unwrap(),
        )
        .unwrap();
        cmd.execute(&mut ctx, &mut GlobalPipelineCache::default())
            .unwrap();
        let instances = ctx.get_instance_map().unwrap().as_slice();

        let objects = |labels: &[u32]| {
            let mut ids: Vec<u32> = labels.iter().copied().filter(|&l| l != 0).collect();
            ids.sort();
            ids.dedup();
            ids.len()
        };
        // Object ids may differ - compare foreground and object count.
        let differing = instances
            .iter()
            .zip(&reference)
            .filter(|(a, b)| (**a == 0) != (**b == 0))
            .count();
        println!(
            "objects: rust {}, reference {}; foreground differs at {differing} px",
            objects(instances),
            objects(&reference)
        );
        assert_eq!(objects(instances), objects(&reference));
        assert_eq!(
            differing, 0,
            "{differing} pixels differ from YOLOv5's own result"
        );
    }

    #[test]
    fn yolov5_is_named_cites_its_paper_and_runs_per_tile() {
        let cmd = yolo();
        assert_eq!(cmd.name(), "YOLOv5");
        assert_eq!(cmd.execution_scope(), ExecutionScope::Tile);
        let cites = cmd.cite();
        assert_eq!(cites.len(), 1);
        assert_eq!(cites[0].cite_key, "jocher2022yolov5");
    }

    #[test]
    fn rgb_planes_rejects_a_label_image() {
        let size = ImageSize {
            width: 2,
            height: 2,
        };
        let labels = ImageContainer::U32(crate::ManagedImage {
            data: Image::<u32, 1>::new(size, vec![1; 4]).unwrap(),
            tile_offset: crate::image::Point2d { x: 0, y: 0 },
            plane: None,
        });
        let err = Yolov5::rgb_planes(&labels).unwrap_err();
        assert!(
            matches!(&err, InternalErrors::FormatMismatch { expected, .. } if expected.contains("F32Gray or F32Rgb")),
            "{err:?}"
        );
    }

    #[cfg(feature = "libtorch-in-process")]
    #[test]
    fn execute_reports_a_model_whose_prediction_has_too_few_columns() {
        // A model returning `[1, 1, 5]`: a box and objectness, no class score.
        let (_dir, model_path) = trace_and_save_model(3, 640, 640, |x| {
            let one = x.mean(Kind::Float) * 0.0 + 1.0;
            let columns: Vec<Tensor> = [320.0, 320.0, 100.0, 100.0, 0.9]
                .iter()
                .map(|&v| &one * v)
                .collect();
            Tensor::stack(&columns, 0).reshape([1, 1, 5])
        });
        let cmd = Yolov5 {
            model_path,
            ..yolo()
        };
        let err = cmd
            .execute(&mut gray(64, 64), &mut GlobalPipelineCache::default())
            .unwrap_err();
        assert!(
            matches!(&err, InternalErrors::Generic(msg) if msg.contains("too few")),
            "{err:?}"
        );
    }
}
