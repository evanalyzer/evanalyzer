//! # stardist
//!
//! **Author:** Joachim Danmayr
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **AGPL-3.0**.

use std::path::PathBuf;

use evanalyzer_cfg::core_types::{InternalErrors, SegmentationClass};
use macros::CommandsMeta;
use tch::{CModule, Device, IValue, Kind, Tensor};

use crate::{
    algos::ImageAlgorithm,
    pipeline::{pipeline_cache::PipelineCache, pipeline_context::PipelineContext},
};

/// Instance segmentation using a pretrained StarDist model exported as TorchScript.
///
/// The model is expected to accept a `[1, 1, H, W]` float tensor (single-channel,
/// same normalization as the rest of the pipeline) and return two tensors:
/// an object-probability map `[1, 1, H', W']` and a ray-distance map
/// `[1, n_rays, H', W']` giving, for each grid cell, the distance to the object
/// boundary along `n_rays` equally-spaced angles (the StarDist star-convex-polygon
/// representation). `H'`/`W'` may be smaller than the input size if the model
/// predicts on a coarser grid; this is detected from the output shape and the
/// polygons are rescaled back to image resolution automatically.
///
/// Some TorchScript exports concatenate both outputs into a single
/// `[1, 1 + n_rays, H', W']` tensor (channel 0 = probability, the rest =
/// distances); this is also supported.
///
/// Per grid cell candidates above `probability_threshold` are converted to
/// star-convex polygons, then greedily filtered with non-maximum suppression
/// (polygons whose pixel-overlap ratio with a higher-scoring candidate exceeds
/// `nms_threshold` are discarded) before being rasterized into the pipeline's
/// segmentation and instance maps. Runs on GPU automatically if CUDA is
/// available in the linked libtorch build, otherwise falls back to CPU.
#[derive(CommandsMeta)]
#[cmdsmeta(category = "segment", next = "measure", display_name = "AI Stardist Segmentation")]
pub struct Stardist {
    /// Path to a TorchScript-exported StarDist model (`torch.jit.script`/`torch.jit.trace`).
    #[cmdsmeta(file_extensions = "pt,pth")]
    pub model_path: PathBuf,

    /// The class assigned to pixels of every detected object. All other
    /// pixels are assigned `SegmentationClass::BACKGROUND`.
    #[cmdsmeta(default = SegmentationClass(1))]
    pub object_class_id: SegmentationClass,

    /// Probability above which a grid cell is considered a candidate object center.
    #[cmdsmeta(default = 0.5, min = 0.0, max = 1.0, step = 0.01)]
    pub probability_threshold: f32,

    /// Pixel-overlap ratio (intersection / union) above which a lower-scoring
    /// candidate polygon is suppressed in favor of an overlapping higher-scoring one.
    #[cmdsmeta(default = 0.3, min = 0.0, max = 1.0, step = 0.01)]
    pub nms_threshold: f32,
}

/// One star-convex polygon candidate, produced from a single grid cell.
struct Candidate {
    score: f32,
    /// Inclusive pixel bounds `[min_x, min_y, max_x, max_y]`, clipped to the image.
    bbox: [i64; 4],
    /// Rasterized fill mask local to `bbox`, row-major, width = `bbox` width.
    mask: Vec<bool>,
    /// Number of `true` entries in `mask` (i.e. the polygon's pixel area).
    area: usize,
}

impl ImageAlgorithm for Stardist {
    fn execute(
        &self,
        ctx: &mut PipelineContext,
        _cache: &mut PipelineCache,
    ) -> Result<(), InternalErrors> {
        let device = Device::cuda_if_available();
        let model = CModule::load_on_device(&self.model_path, device).map_err(|e| {
            InternalErrors::Generic(format!(
                "Failed to load StarDist model from {}: {e}",
                self.model_path.display()
            ))
        })?;

        let (input_image, segmentation_map, instance_map) =
            ctx.get_f32_gray_segmentation_and_instances_mut()?;
        let size = input_image.size();
        let (width, height) = (size.width, size.height);

        let input = Tensor::from_slice(input_image.as_slice())
            .to_device(device)
            .to_kind(Kind::Float)
            .reshape([1, 1, height as i64, width as i64]);

        let (prob_tensor, dist_tensor) = Self::split_outputs(&model, input)?;

        let prob_sizes = prob_tensor.size();
        let dist_sizes = dist_tensor.size();
        let grid_h = prob_sizes[prob_sizes.len() - 2] as usize;
        let grid_w = prob_sizes[prob_sizes.len() - 1] as usize;
        let n_rays = dist_sizes[dist_sizes.len() - 3] as usize;

        let prob_flat: Vec<f32> = prob_tensor
            .f_to_device(Device::Cpu)
            .and_then(|t| t.f_reshape([(grid_h * grid_w) as i64]))
            .map_err(|e| InternalErrors::Generic(format!("StarDist inference failed: {e}")))
            .and_then(|t| {
                Vec::try_from(&t)
                    .map_err(|e| InternalErrors::Generic(format!("StarDist inference failed: {e}")))
            })?;

        let dist_flat: Vec<f32> = dist_tensor
            .f_to_device(Device::Cpu)
            .and_then(|t| t.f_reshape([(n_rays * grid_h * grid_w) as i64]))
            .map_err(|e| InternalErrors::Generic(format!("StarDist inference failed: {e}")))
            .and_then(|t| {
                Vec::try_from(&t)
                    .map_err(|e| InternalErrors::Generic(format!("StarDist inference failed: {e}")))
            })?;

        let candidates = self.build_candidates(
            &prob_flat, &dist_flat, grid_h, grid_w, n_rays, width, height,
        );
        let kept = Self::non_max_suppress(candidates, self.nms_threshold);

        let foreground_class = self.object_class_id.as_u32();
        let seg_slice = segmentation_map.as_slice_mut();
        let inst_slice = instance_map.as_slice_mut();

        for (instance_id, candidate) in (1u32..).zip(kept.iter()) {
            let [min_x, min_y, max_x, max_y] = candidate.bbox;
            let local_w = (max_x - min_x + 1) as usize;
            for y in min_y..=max_y {
                for x in min_x..=max_x {
                    let local_idx = (y - min_y) as usize * local_w + (x - min_x) as usize;
                    if !candidate.mask[local_idx] {
                        continue;
                    }
                    let global_idx = y as usize * width + x as usize;
                    // First (highest-scoring) candidate to claim a pixel wins.
                    if inst_slice[global_idx] != 0 {
                        continue;
                    }
                    inst_slice[global_idx] = instance_id;
                    seg_slice[global_idx] = foreground_class;
                }
            }
        }

        Ok(())
    }

    fn name(&self) -> &'static str {
        "Stardist"
    }
}

impl Stardist {
    /// Runs the model and splits its output into `(probability, distance)` tensors,
    /// supporting both the two-separate-tensors and the single-concatenated-tensor
    /// TorchScript export conventions.
    fn split_outputs(model: &CModule, input: Tensor) -> Result<(Tensor, Tensor), InternalErrors> {
        let output = model
            .forward_is(&[IValue::Tensor(input)])
            .map_err(|e| InternalErrors::Generic(format!("StarDist inference failed: {e}")))?;

        let tensors: Vec<Tensor> = match output {
            IValue::Tensor(t) => vec![t],
            IValue::Tuple(items) | IValue::GenericList(items) => items
                .into_iter()
                .filter_map(|v| match v {
                    IValue::Tensor(t) => Some(t),
                    _ => None,
                })
                .collect(),
            other => {
                return Err(InternalErrors::Generic(format!(
                    "StarDist model returned an unsupported output type: {other:?}"
                )));
            }
        };

        match tensors.as_slice() {
            [single] => {
                let sizes = single.size();
                if sizes.len() < 3 {
                    return Err(InternalErrors::Generic(
                        "StarDist model output has too few dimensions".into(),
                    ));
                }
                let channel_dim = sizes.len() as i64 - 3;
                let channels = sizes[sizes.len() - 3];
                if channels < 2 {
                    return Err(InternalErrors::Generic(
                        "StarDist model output has fewer than 2 channels; expected a probability channel plus ray-distance channels".into(),
                    ));
                }
                let prob = single.narrow(channel_dim, 0, 1);
                let dist = single.narrow(channel_dim, 1, channels - 1);
                Ok((prob, dist))
            }
            [_, _, ..] => {
                let prob_idx = tensors.iter().position(|t| {
                    let sizes = t.size();
                    sizes.len() >= 3 && sizes[sizes.len() - 3] == 1
                });
                match prob_idx {
                    Some(idx) => {
                        let dist_idx = if idx == 0 { 1 } else { 0 };
                        Ok((tensors[idx].shallow_clone(), tensors[dist_idx].shallow_clone()))
                    }
                    None => Err(InternalErrors::Generic(
                        "StarDist model returned multiple outputs but none has a single-channel probability map".into(),
                    )),
                }
            }
            [] => Err(InternalErrors::Generic(
                "StarDist model returned no tensor outputs".into(),
            )),
        }
    }

    /// Builds one star-convex polygon candidate per grid cell whose probability
    /// reaches `probability_threshold`, rescaling coordinates back to image
    /// resolution if the model predicts on a coarser grid than the input.
    fn build_candidates(
        &self,
        prob_flat: &[f32],
        dist_flat: &[f32],
        grid_h: usize,
        grid_w: usize,
        n_rays: usize,
        width: usize,
        height: usize,
    ) -> Vec<Candidate> {
        let scale_y = height as f32 / grid_h as f32;
        let scale_x = width as f32 / grid_w as f32;
        let dist_scale = (scale_y + scale_x) * 0.5;

        let angles: Vec<(f32, f32)> = (0..n_rays)
            .map(|k| {
                let angle = 2.0 * std::f32::consts::PI * k as f32 / n_rays as f32;
                (angle.cos(), angle.sin())
            })
            .collect();

        let mut candidates = Vec::new();
        for gy in 0..grid_h {
            for gx in 0..grid_w {
                let idx = gy * grid_w + gx;
                let score = prob_flat[idx];
                if score < self.probability_threshold {
                    continue;
                }

                let cy = (gy as f32 + 0.5) * scale_y;
                let cx = (gx as f32 + 0.5) * scale_x;

                let polygon: Vec<(f32, f32)> = angles
                    .iter()
                    .enumerate()
                    .map(|(k, (cos_a, sin_a))| {
                        let r = dist_flat[k * grid_h * grid_w + idx] * dist_scale;
                        (cx + r * cos_a, cy + r * sin_a)
                    })
                    .collect();

                if let Some(candidate) = Self::build_candidate(score, &polygon, width, height) {
                    candidates.push(candidate);
                }
            }
        }
        candidates
    }

    fn build_candidate(
        score: f32,
        polygon: &[(f32, f32)],
        width: usize,
        height: usize,
    ) -> Option<Candidate> {
        let (mut min_x, mut min_y, mut max_x, mut max_y) = (f32::MAX, f32::MAX, f32::MIN, f32::MIN);
        for &(x, y) in polygon {
            min_x = min_x.min(x);
            min_y = min_y.min(y);
            max_x = max_x.max(x);
            max_y = max_y.max(y);
        }

        let bbox = [
            (min_x.floor() as i64).clamp(0, width as i64 - 1),
            (min_y.floor() as i64).clamp(0, height as i64 - 1),
            (max_x.ceil() as i64).clamp(0, width as i64 - 1),
            (max_y.ceil() as i64).clamp(0, height as i64 - 1),
        ];
        if bbox[2] < bbox[0] || bbox[3] < bbox[1] {
            return None;
        }

        let mask = Self::rasterize_polygon(polygon, bbox);
        let area = mask.iter().filter(|&&b| b).count();
        if area == 0 {
            return None;
        }

        Some(Candidate {
            score,
            bbox,
            mask,
            area,
        })
    }

    /// Fills a simple (non-self-intersecting) polygon using an even-odd scanline
    /// rasterizer, sampling at pixel centers. Star-convex polygons (rays cast from
    /// a single interior center) are always simple, so this is exact.
    fn rasterize_polygon(polygon: &[(f32, f32)], bbox: [i64; 4]) -> Vec<bool> {
        let [min_x, min_y, max_x, max_y] = bbox;
        let local_w = (max_x - min_x + 1) as usize;
        let local_h = (max_y - min_y + 1) as usize;
        let mut mask = vec![false; local_w * local_h];

        let n = polygon.len();
        for row in 0..local_h {
            let py = min_y as f32 + row as f32 + 0.5;
            let mut crossings: Vec<f32> = Vec::new();
            for i in 0..n {
                let (x0, y0) = polygon[i];
                let (x1, y1) = polygon[(i + 1) % n];
                if (y0 <= py && y1 > py) || (y1 <= py && y0 > py) {
                    let t = (py - y0) / (y1 - y0);
                    crossings.push(x0 + t * (x1 - x0));
                }
            }
            crossings.sort_by(|a, b| a.partial_cmp(b).unwrap());

            for pair in crossings.chunks_exact(2) {
                let x_start = (pair[0].round() as i64 - min_x).clamp(0, local_w as i64 - 1);
                let x_end = (pair[1].round() as i64 - min_x).clamp(0, local_w as i64 - 1);
                for x in x_start..=x_end {
                    mask[row * local_w + x as usize] = true;
                }
            }
        }
        mask
    }

    fn bbox_overlaps(a: &[i64; 4], b: &[i64; 4]) -> bool {
        a[0] <= b[2] && b[0] <= a[2] && a[1] <= b[3] && b[1] <= a[3]
    }

    /// Pixel-overlap ratio (intersection / union) between two candidates' masks.
    fn iou(a: &Candidate, b: &Candidate) -> f32 {
        let min_x = a.bbox[0].max(b.bbox[0]);
        let min_y = a.bbox[1].max(b.bbox[1]);
        let max_x = a.bbox[2].min(b.bbox[2]);
        let max_y = a.bbox[3].min(b.bbox[3]);
        if max_x < min_x || max_y < min_y {
            return 0.0;
        }

        let a_w = a.bbox[2] - a.bbox[0] + 1;
        let b_w = b.bbox[2] - b.bbox[0] + 1;

        let mut intersection = 0usize;
        for y in min_y..=max_y {
            for x in min_x..=max_x {
                let a_idx = ((y - a.bbox[1]) * a_w + (x - a.bbox[0])) as usize;
                let b_idx = ((y - b.bbox[1]) * b_w + (x - b.bbox[0])) as usize;
                if a.mask[a_idx] && b.mask[b_idx] {
                    intersection += 1;
                }
            }
        }
        if intersection == 0 {
            return 0.0;
        }

        let union = a.area + b.area - intersection;
        intersection as f32 / union as f32
    }

    /// Greedy NMS: candidates are visited in descending score order; a candidate
    /// is kept unless it overlaps (by more than `nms_threshold`) a higher-scoring
    /// candidate that was already kept. Returns the kept candidates, still sorted
    /// by descending score (highest-scoring objects are painted first).
    fn non_max_suppress(candidates: Vec<Candidate>, nms_threshold: f32) -> Vec<Candidate> {
        let mut order: Vec<usize> = (0..candidates.len()).collect();
        order.sort_by(|&a, &b| candidates[b].score.partial_cmp(&candidates[a].score).unwrap());

        let mut suppressed = vec![false; candidates.len()];
        let mut keep_order = Vec::new();

        for &i in &order {
            if suppressed[i] {
                continue;
            }
            keep_order.push(i);
            for &j in &order {
                if j == i || suppressed[j] || !Self::bbox_overlaps(&candidates[i].bbox, &candidates[j].bbox)
                {
                    continue;
                }
                if Self::iou(&candidates[i], &candidates[j]) > nms_threshold {
                    suppressed[j] = true;
                }
            }
        }

        let mut slots: Vec<Option<Candidate>> = candidates.into_iter().map(Some).collect();
        keep_order
            .into_iter()
            .map(|i| slots[i].take().unwrap())
            .collect()
    }
}
