//! # yolov5
//!
//! **Author:** Joachim Danmayr
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **AGPL-3.0**.

use std::path::PathBuf;

use evanalyzer_cfg::core_types::{CitationMetadata, InternalErrors, SegmentationClass};
use kornia_image::Image;
use macros::CommandsMeta;
use tch::{CModule, Device, IValue, Kind, Tensor};

use crate::{
    ImageContainer,
    algos::{ExecutionScope, ImageAlgorithm, ai_segmentation::model_cache::load_cached_model},
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
    next = "measure",
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
    #[cmdsmeta(default = 0.45, min = 0.0, max = 1.0, step = 0.01)]
    pub iou_threshold: f32,

    /// Mask probability above which a pixel belongs to its object
    /// (segmentation models only).
    #[cmdsmeta(default = 0.5, min = 0.0, max = 1.0, step = 0.01)]
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
    #[cmdsmeta(default = 128, min = 0, max = 512, step = 8)]
    pub window_overlap: i32,

    /// Objects with fewer pixels than this (after overlapping objects were
    /// resolved) are removed. `0` keeps every object.
    #[cmdsmeta(default = 15, min = 0, max = 100000, step = 1)]
    pub min_object_size: i32,
}

/// One kept detection, in tile coordinates.
struct Detection {
    score: f32,
    class: usize,
    /// `[x1, y1, x2, y2]`, exclusive upper bounds.
    bbox: [f32; 4],
    /// Pixel mask over `mask_rect` (row-major), `None` for a filled box.
    mask: Option<Vec<bool>>,
    /// Integer rectangle `[x, y, width, height]` the mask covers.
    mask_rect: [usize; 4],
}

impl ImageAlgorithm for Yolov5 {
    fn execute(
        &self,
        ctx: &mut PipelineContext,
        _cache: &mut GlobalPipelineCache,
    ) -> Result<(), InternalErrors> {
        let device = Device::cuda_if_available();
        let model = load_cached_model(&self.model_path, || {
            CModule::load_on_device(&self.model_path, device)
        })
        .map_err(|e| {
            InternalErrors::Generic(format!(
                "Failed to load YOLOv5 model from {}: {e}. The model must be a TorchScript \
                 export (yolov5 export.py --include torchscript), not a training checkpoint.",
                self.model_path.display()
            ))
        })?;

        let size = ctx.get_image_size();
        let (tile_width, tile_height) = (size.width, size.height);
        // Everything up to the label maps runs on the scaled image.
        let scale = self.image_scale.clamp(0.05, 4.0) as f64;
        let width = ((tile_width as f64 * scale).round() as usize).max(1);
        let height = ((tile_height as f64 * scale).round() as usize).max(1);
        let mut rgb = Self::rgb_planes(&ctx.image)?;
        if (width, height) != (tile_width, tile_height) {
            rgb = Self::resize_planes(rgb, tile_width, tile_height, width, height)?;
        }

        let mut detections = Vec::new();
        for &y0 in &Self::window_starts(height, self.window_overlap) {
            for &x0 in &Self::window_starts(width, self.window_overlap) {
                let window = Self::window_tensor(&rgb, width, height, x0, y0).to_device(device);
                let output =
                    tch::no_grad(|| model.forward_is(&[IValue::Tensor(window)])).map_err(|e| {
                        InternalErrors::Generic(format!("YOLOv5 inference failed: {e}"))
                    })?;
                let (pred, proto) = Self::split_outputs(output)?;
                detections.extend(self.decode_window(
                    &pred,
                    proto.as_ref(),
                    x0,
                    y0,
                    width,
                    height,
                )?);
            }
        }
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
    /// The input size YOLOv5 TorchScript exports are traced at.
    const WINDOW: usize = 640;
    /// YOLOv5's letterbox padding gray (114 of 255).
    const PAD_VALUE: f32 = 114.0 / 255.0;
    /// Detections kept per window after NMS (YOLOv5's `max_det`).
    const MAX_DETECTIONS: usize = 300;
    /// A box this close (px) to a window edge inside the tile is cut off by
    /// that edge; the overlapping neighbor window sees the object whole.
    const EDGE_MARGIN: f32 = 2.0;

    /// How far (px) a mask may reach past its box: one prototype pixel
    /// (640 / 160 = 4 px for YOLOv5's 160x160 prototypes).
    const MASK_SPILL: f32 = 4.0;

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

    /// Resizes the R, G, B planes from `src` to `dst` size (bilinear, with
    /// antialiasing when shrinking - as image libraries resize).
    fn resize_planes(
        planes: [Vec<f32>; 3],
        src_width: usize,
        src_height: usize,
        dst_width: usize,
        dst_height: usize,
    ) -> Result<[Vec<f32>; 3], InternalErrors> {
        let err = |e: tch::TchError| InternalErrors::Generic(format!("YOLOv5 resize failed: {e}"));
        let data: Vec<f32> = planes.concat();
        let resized = Tensor::from_slice(&data)
            .reshape([1, 3, src_height as i64, src_width as i64])
            .f_internal_upsample_bilinear2d_aa(
                [dst_height as i64, dst_width as i64],
                false,
                None,
                None,
            )
            .map_err(err)?;
        let flat = Vec::<f32>::try_from(&resized.reshape([-1])).map_err(err)?;
        let n = dst_width * dst_height;
        Ok([
            flat[..n].to_vec(),
            flat[n..2 * n].to_vec(),
            flat[2 * n..].to_vec(),
        ])
    }

    /// Left/top edges of the windows covering `len` pixels with `overlap`
    /// px of overlap: one window for `len <= 640`, else evenly spread
    /// windows, the last one flush with the end.
    fn window_starts(len: usize, overlap: i32) -> Vec<usize> {
        if len <= Self::WINDOW {
            return vec![0];
        }
        let stride = Self::WINDOW.saturating_sub(overlap.max(0) as usize).max(1);
        let span = len - Self::WINDOW;
        let count = span.div_ceil(stride) + 1;
        (0..count)
            .map(|i| if i + 1 == count { span } else { i * stride })
            .collect()
    }

    /// The `[1, 3, 640, 640]` model input for the window at (`x0`, `y0`);
    /// parts outside the tile are padded with YOLOv5's gray.
    fn window_tensor(
        rgb: &[Vec<f32>; 3],
        width: usize,
        height: usize,
        x0: usize,
        y0: usize,
    ) -> Tensor {
        let w = Self::WINDOW;
        let mut data = vec![Self::PAD_VALUE; 3 * w * w];
        let cols = (width - x0).min(w);
        for (c, plane) in rgb.iter().enumerate() {
            for y in 0..(height - y0).min(w) {
                let src = (y0 + y) * width + x0;
                let dst = c * w * w + y * w;
                data[dst..dst + cols].copy_from_slice(&plane[src..src + cols]);
            }
        }
        Tensor::from_slice(&data).reshape([1, 3, w as i64, w as i64])
    }

    /// The prediction tensor (`[1, N, 5 + classes + mask coefficients]`) and,
    /// for segmentation models, the mask prototypes (`[1, M, h, w]`).
    fn split_outputs(output: IValue) -> Result<(Tensor, Option<Tensor>), InternalErrors> {
        let mut tensors = Vec::new();
        fn collect(value: IValue, out: &mut Vec<Tensor>) {
            match value {
                IValue::Tensor(t) => out.push(t),
                IValue::Tuple(items) | IValue::GenericList(items) => {
                    items.into_iter().for_each(|v| collect(v, out))
                }
                _ => {}
            }
        }
        collect(output, &mut tensors);
        let pred = tensors
            .iter()
            .find(|t| t.dim() == 3)
            .map(|t| t.shallow_clone());
        let proto = tensors
            .iter()
            .find(|t| t.dim() == 4)
            .map(|t| t.shallow_clone());
        let pred = pred.ok_or_else(|| {
            InternalErrors::Generic(
                "YOLOv5 model returned no [1, N, C] prediction tensor - is this a YOLOv5 export?"
                    .into(),
            )
        })?;
        Ok((pred, proto))
    }

    /// Turns one window's raw output into detections in tile coordinates:
    /// confidence filter and per-window NMS as YOLOv5's `non_max_suppression`,
    /// boxes cut by an inner window edge dropped, masks built for the rest.
    fn decode_window(
        &self,
        pred: &Tensor,
        proto: Option<&Tensor>,
        x0: usize,
        y0: usize,
        width: usize,
        height: usize,
    ) -> Result<Vec<Detection>, InternalErrors> {
        let err =
            |e: tch::TchError| InternalErrors::Generic(format!("YOLOv5 inference failed: {e}"));
        let pred = pred
            .f_to_device(Device::Cpu)
            .map_err(err)?
            .to_kind(Kind::Float);
        let sizes = pred.size();
        let (n, cols) = (sizes[1] as usize, sizes[2] as usize);
        let n_masks = proto.map(|p| p.size()[1] as usize).unwrap_or(0);
        if cols < 6 + n_masks {
            return Err(InternalErrors::Generic(format!(
                "YOLOv5 prediction has {cols} columns, too few for {n_masks} mask coefficients"
            )));
        }
        let n_classes = cols - 5 - n_masks;
        let values = Vec::<f32>::try_from(&pred.reshape([-1])).map_err(err)?;

        let w = Self::WINDOW as f32;
        let mut candidates: Vec<(Detection, Vec<f32>)> = Vec::new();
        for row in values.chunks_exact(cols).take(n) {
            let objectness = row[4];
            if objectness <= self.confidence_threshold {
                continue;
            }
            let (class, class_score) =
                row[5..5 + n_classes]
                    .iter()
                    .enumerate()
                    .fold(
                        (0, f32::MIN),
                        |best, (i, &s)| if s > best.1 { (i, s) } else { best },
                    );
            let score = objectness * class_score;
            if score <= self.confidence_threshold {
                continue;
            }
            let (cx, cy, bw, bh) = (row[0], row[1], row[2], row[3]);
            let bbox = [
                (cx - bw / 2.0).clamp(0.0, w),
                (cy - bh / 2.0).clamp(0.0, w),
                (cx + bw / 2.0).clamp(0.0, w),
                (cy + bh / 2.0).clamp(0.0, w),
            ];
            candidates.push((
                Detection {
                    score,
                    class,
                    bbox,
                    mask: None,
                    mask_rect: [0; 4],
                },
                row[5 + n_classes..].to_vec(),
            ));
        }

        // Per-window NMS on window coordinates (as YOLOv5), keeping the
        // coefficients alongside.
        candidates.sort_by(|a, b| b.0.score.total_cmp(&a.0.score));
        let mut kept: Vec<(Detection, Vec<f32>)> = Vec::new();
        for candidate in candidates {
            if kept.len() >= Self::MAX_DETECTIONS {
                break;
            }
            let suppressed = kept.iter().any(|(k, _)| {
                k.class == candidate.0.class && iou(&k.bbox, &candidate.0.bbox) > self.iou_threshold
            });
            if !suppressed {
                kept.push(candidate);
            }
        }

        // Boxes touching a window edge that lies inside the tile are cut off
        // there - the neighboring window (overlap) sees them whole.
        let inner = [
            x0 > 0,
            y0 > 0,
            x0 + Self::WINDOW < width,
            y0 + Self::WINDOW < height,
        ];
        kept.retain(|(d, _)| {
            !((inner[0] && d.bbox[0] <= Self::EDGE_MARGIN)
                || (inner[1] && d.bbox[1] <= Self::EDGE_MARGIN)
                || (inner[2] && d.bbox[2] >= w - Self::EDGE_MARGIN)
                || (inner[3] && d.bbox[3] >= w - Self::EDGE_MARGIN))
        });

        let masks = match proto {
            Some(proto) if !kept.is_empty() => Some(self.masks(proto, &kept)?),
            _ => None,
        };

        let (tile_w, tile_h) = (width as f32, height as f32);
        let mut out = Vec::with_capacity(kept.len());
        for (i, (mut d, _)) in kept.into_iter().enumerate() {
            // Window -> tile coordinates; padding beyond the tile is cut away.
            let bbox = [
                (d.bbox[0] + x0 as f32).min(tile_w),
                (d.bbox[1] + y0 as f32).min(tile_h),
                (d.bbox[2] + x0 as f32).min(tile_w),
                (d.bbox[3] + y0 as f32).min(tile_h),
            ];
            // A mask is cut to its box at prototype resolution and only then
            // scaled up, so its smoothed edge reaches past the box by up to
            // one prototype pixel - copy that much around the box too.
            let spill = if masks.is_some() {
                Self::MASK_SPILL
            } else {
                0.0
            };
            let rx = (d.bbox[0] - spill).max(0.0).floor() as usize;
            let ry = (d.bbox[1] - spill).max(0.0).floor() as usize;
            let rx2 = ((d.bbox[2] + spill).ceil() as usize)
                .min(Self::WINDOW)
                .min(width - x0);
            let ry2 = ((d.bbox[3] + spill).ceil() as usize)
                .min(Self::WINDOW)
                .min(height - y0);
            if rx2 <= rx || ry2 <= ry {
                continue;
            }
            let (rw, rh) = (rx2 - rx, ry2 - ry);
            d.mask = masks.as_ref().map(|m| {
                let mut mask = Vec::with_capacity(rw * rh);
                for y in ry..ry2 {
                    let row = i * Self::WINDOW * Self::WINDOW + y * Self::WINDOW;
                    mask.extend_from_slice(&m[row + rx..row + rx2]);
                }
                mask
            });
            d.mask_rect = [rx + x0, ry + y0, rw, rh];
            d.bbox = bbox;
            out.push(d);
        }
        Ok(out)
    }

    /// Masks of the kept detections at window resolution (`[k][640 * 640]`,
    /// flattened), as YOLOv5's `process_mask(upsample=True)`: the
    /// coefficients weight the prototypes, the sigmoid of that is cut to the
    /// box (at prototype resolution), scaled up bilinearly and thresholded.
    fn masks(
        &self,
        proto: &Tensor,
        kept: &[(Detection, Vec<f32>)],
    ) -> Result<Vec<bool>, InternalErrors> {
        let err = |e: tch::TchError| InternalErrors::Generic(format!("YOLOv5 mask failed: {e}"));
        let proto = proto
            .f_to_device(Device::Cpu)
            .map_err(err)?
            .to_kind(Kind::Float);
        let s = proto.size();
        let (m, ph, pw) = (s[1], s[2], s[3]);
        let k = kept.len() as i64;
        let coefficients: Vec<f32> = kept.iter().flat_map(|(_, c)| c.iter().copied()).collect();
        let coefficients = Tensor::from_slice(&coefficients).reshape([k, m]);
        let masks = coefficients
            .matmul(&proto.reshape([m, ph * pw]))
            .sigmoid()
            .reshape([k, ph, pw]);

        // Cut to the box at prototype resolution.
        let (sx, sy) = (
            pw as f32 / Self::WINDOW as f32,
            ph as f32 / Self::WINDOW as f32,
        );
        let mut inside = vec![0f32; (k * ph * pw) as usize];
        for (i, (d, _)) in kept.iter().enumerate() {
            let [x1, y1, x2, y2] = d.bbox;
            for y in 0..ph as usize {
                let py = y as f32;
                if py < y1 * sy || py >= y2 * sy {
                    continue;
                }
                for x in 0..pw as usize {
                    let px = x as f32;
                    if px >= x1 * sx && px < x2 * sx {
                        inside[i * (ph * pw) as usize + y * pw as usize + x] = 1.0;
                    }
                }
            }
        }
        let masks = masks * Tensor::from_slice(&inside).reshape([k, ph, pw]);
        let w = Self::WINDOW as i64;
        let upsampled = masks
            .unsqueeze(0)
            .upsample_bilinear2d([w, w], false, None, None)
            .squeeze_dim(0)
            .gt(self.mask_threshold as f64);
        Vec::<bool>::try_from(&upsampled.reshape([-1])).map_err(err)
    }

    /// Greedy class-aware NMS over all windows' detections (box IoU): of
    /// overlapping detections of one class only the most confident stays.
    fn nms(mut detections: Vec<Detection>, iou_threshold: f32) -> Vec<Detection> {
        detections.sort_by(|a, b| b.score.total_cmp(&a.score));
        let mut kept: Vec<Detection> = Vec::new();
        for d in detections {
            if !kept
                .iter()
                .any(|k| k.class == d.class && iou(&k.bbox, &d.bbox) > iou_threshold)
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

/// Intersection over union of two `[x1, y1, x2, y2]` boxes.
fn iou(a: &[f32; 4], b: &[f32; 4]) -> f32 {
    let iw = (a[2].min(b[2]) - a[0].max(b[0])).max(0.0);
    let ih = (a[3].min(b[3]) - a[1].max(b[1])).max(0.0);
    let intersection = iw * ih;
    let union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - intersection;
    if union <= 0.0 {
        0.0
    } else {
        intersection / union
    }
}

#[cfg(test)]
mod tests {
    use super::*;
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
    fn window_starts_cover_the_tile_with_the_last_window_flush_with_the_end() {
        assert_eq!(Yolov5::window_starts(500, 128), vec![0]);
        assert_eq!(Yolov5::window_starts(640, 128), vec![0]);
        assert_eq!(Yolov5::window_starts(1000, 128), vec![0, 360]);
        assert_eq!(Yolov5::window_starts(2048, 128), vec![0, 512, 1024, 1408]);
        for len in [641, 1000, 2048, 3001] {
            let starts = Yolov5::window_starts(len, 128);
            assert_eq!(*starts.last().unwrap() + 640, len);
            for pair in starts.windows(2) {
                assert!(
                    pair[1] - pair[0] <= 512,
                    "{len}: gap leaves less than 128 px overlap"
                );
            }
        }
    }

    #[test]
    fn iou_of_boxes() {
        assert_eq!(iou(&[0.0, 0.0, 10.0, 10.0], &[0.0, 0.0, 10.0, 10.0]), 1.0);
        assert_eq!(iou(&[0.0, 0.0, 10.0, 10.0], &[20.0, 20.0, 30.0, 30.0]), 0.0);
        assert!((iou(&[0.0, 0.0, 10.0, 10.0], &[5.0, 0.0, 15.0, 10.0]) - 1.0 / 3.0).abs() < 1e-6);
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

    /// The University of Salzburg brightfield YOLOv5-seg model on a 640x640
    /// crop of `tests/cellpose_sam/B5_0009.jpg`, compared with YOLOv5's own
    /// post-processing (`non_max_suppression` + `process_mask`), re-implemented
    /// in `make_reference.py` next to the model. Model and data aren't in git
    /// (190 MB); they live in `tests/university_of_sbg_brightfield_cell_segmentation_v3/`.
    /// Run with:
    ///   cargo test -p evanalyzer_core --features ai --lib real_yolov5 -- --ignored
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
}
