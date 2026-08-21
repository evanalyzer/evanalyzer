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
    algos::{ImageAlgorithm, ai_segmentation::model_cache::load_cached_model},
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
#[cmdsmeta(
    category = "segment",
    next = "measure",
    display_name = "AI Stardist Segmentation"
)]
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
        let model = load_cached_model(&self.model_path, || {
            CModule::load_on_device(&self.model_path, device)
        })
        .map_err(|e| {
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
        Self::write_instances(
            &kept,
            width,
            foreground_class,
            segmentation_map.as_slice_mut(),
            instance_map.as_slice_mut(),
        );

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
                // NaN fails every `<` comparison, so a NaN score would
                // otherwise sail past the threshold check below (`NaN <
                // threshold` is always false) and later reach the score sort
                // in `non_max_suppress`, which can't order it.
                if !score.is_finite() || score < self.probability_threshold {
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
        // A single degenerate ray distance (model output NaN/inf for one grid
        // cell, e.g. from a blank/degenerate input tile) poisons this whole
        // star-convex polygon. `f32::min`/`max` below silently ignore NaN
        // operands (so the bbox wouldn't catch it), but the raw polygon is
        // also fed to `rasterize_polygon`'s scanline sort, which can't order
        // a NaN crossing - reject the candidate outright instead of letting
        // a corrupted shape (or a panic) through.
        if polygon
            .iter()
            .any(|(x, y)| !x.is_finite() || !y.is_finite())
        {
            return None;
        }

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
            // `build_candidate` already rejects any polygon with a non-finite
            // vertex, so `crossings` shouldn't contain NaN here - `unwrap_or`
            // is defense in depth against a future caller of this function
            // skipping that check, not the primary fix.
            crossings.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

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
        // `build_candidates` already skips non-finite scores before a
        // `Candidate` is ever built, so `unwrap_or` here is defense in depth
        // against a future caller constructing candidates some other way,
        // not the primary fix.
        order.sort_by(|&a, &b| {
            candidates[b]
                .score
                .partial_cmp(&candidates[a].score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let mut suppressed = vec![false; candidates.len()];
        let mut keep_order = Vec::new();

        for &i in &order {
            if suppressed[i] {
                continue;
            }
            keep_order.push(i);
            for &j in &order {
                if j == i
                    || suppressed[j]
                    || !Self::bbox_overlaps(&candidates[i].bbox, &candidates[j].bbox)
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

    /// Rasterizes the kept candidates into `seg_slice`/`inst_slice`.
    ///
    /// `seg_slice`/`inst_slice` may be reused buffers carrying stale labels
    /// from an earlier segmentation pass in the same pipeline, so every pixel
    /// not covered by a surviving candidate is explicitly reset to
    /// background (0) up front - the per-candidate loop below only ever
    /// writes pixels it actually claims.
    fn write_instances(
        kept: &[Candidate],
        width: usize,
        foreground_class: u32,
        seg_slice: &mut [u32],
        inst_slice: &mut [u32],
    ) {
        seg_slice.fill(0);
        inst_slice.fill(0);

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
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algos::ai_segmentation::test_support::trace_and_save_model;
    use kornia_image::{Image, ImageSize};
    use kornia_tensor::CpuAllocator;

    fn stardist(probability_threshold: f32, nms_threshold: f32) -> Stardist {
        Stardist {
            model_path: PathBuf::new(),
            object_class_id: SegmentationClass(1),
            probability_threshold,
            nms_threshold,
        }
    }

    fn gray_ctx(width: usize, height: usize, values: Vec<f32>) -> PipelineContext {
        let img =
            Image::<f32, 1, CpuAllocator>::new(ImageSize { width, height }, values, CpuAllocator)
                .unwrap();
        PipelineContext::new_from_image_test(img).unwrap()
    }

    // ---- execute() - real TorchScript load + inference, see `test_support`.
    // Only the "single concatenated [1, 1+n_rays, H, W] tensor" output
    // convention is exercisable this way - `create_by_tracing` can't produce
    // a saveable module with the "two separate tensors" convention (see
    // `test_support`'s doc comment), so `split_outputs`'s `[_, _, ..]` branch
    // stays untested here. ----

    #[test]
    fn execute_errors_when_the_model_path_does_not_exist() {
        let dir = tempfile::tempdir().unwrap();
        let cmd = Stardist {
            model_path: dir.path().join("missing.pt"),
            ..stardist(0.5, 0.3)
        };
        let mut ctx = gray_ctx(2, 2, vec![0.0; 4]);
        let mut cache = PipelineCache::default();

        let err = cmd.execute(&mut ctx, &mut cache).unwrap_err();
        assert!(matches!(err, InternalErrors::Generic(_)));
    }

    #[test]
    fn execute_end_to_end_rasterizes_a_candidate_around_a_high_probability_pixel() {
        // Grid resolution == image resolution (no downsampling), so the
        // scale factor is 1 and ray distances are plain pixel radii. Channel
        // 0 (probability) is the input value directly - a single interior
        // pixel is 1.0 (comfortably above the default 0.5 threshold), every
        // other pixel is 0.0. The 4 ray-distance channels are a constant
        // radius-2 disc around that one surviving grid cell; the exact
        // polygon shape is already covered precisely by
        // `build_candidates_scales_ray_distance_by_the_grid_to_image_ratio`
        // et al. above, so this only checks that real inference correctly
        // drives *some* object into existence around the intended pixel and
        // leaves a far corner untouched.
        const N_RAYS: i64 = 4;
        let (_dir, model_path) = trace_and_save_model(1, 6, 6, |x| {
            let prob = x.shallow_clone();
            let radius = x.zeros_like() + 2.0;
            let mut channels = vec![prob];
            channels.extend((0..N_RAYS).map(|_| radius.shallow_clone()));
            Tensor::cat(&channels, 1)
        });
        let cmd = Stardist {
            model_path,
            ..stardist(0.5, 0.3)
        };

        let mut values = vec![0.0f32; 36];
        values[3 * 6 + 3] = 1.0; // row 3, col 3
        let mut ctx = gray_ctx(6, 6, values);
        let mut cache = PipelineCache::default();
        cmd.execute(&mut ctx, &mut cache).unwrap();

        let seg = ctx.get_segmentation_map().unwrap();
        let inst = ctx.get_instance_map().unwrap();
        assert_eq!(
            seg.as_slice()[3 * 6 + 3],
            1,
            "the high-probability pixel itself must be claimed by the object"
        );
        assert_ne!(inst.as_slice()[3 * 6 + 3], 0);
        assert_eq!(
            seg.as_slice()[0],
            0,
            "a far corner outside the radius-2 disc must stay background"
        );
        assert_eq!(inst.as_slice()[0], 0);
    }

    #[test]
    fn execute_errors_when_the_model_output_has_too_few_dimensions() {
        let (_dir, model_path) = trace_and_save_model(1, 1, 2, |x| x.squeeze_dim(0).squeeze_dim(0));
        let cmd = Stardist {
            model_path,
            ..stardist(0.5, 0.3)
        };
        let mut ctx = gray_ctx(2, 1, vec![0.0, 1.0]);
        let mut cache = PipelineCache::default();

        let err = cmd.execute(&mut ctx, &mut cache).unwrap_err();
        assert!(matches!(err, InternalErrors::Generic(msg) if msg.contains("too few dimensions")));
    }

    #[test]
    fn execute_errors_when_the_model_output_has_fewer_than_two_channels() {
        let (_dir, model_path) = trace_and_save_model(1, 1, 2, |x| x.shallow_clone());
        let cmd = Stardist {
            model_path,
            ..stardist(0.5, 0.3)
        };
        let mut ctx = gray_ctx(2, 1, vec![0.0, 1.0]);
        let mut cache = PipelineCache::default();

        let err = cmd.execute(&mut ctx, &mut cache).unwrap_err();
        assert!(
            matches!(err, InternalErrors::Generic(msg) if msg.contains("fewer than 2 channels"))
        );
    }

    fn square(min_x: i64, min_y: i64, side: i64, score: f32) -> Candidate {
        let bbox = [min_x, min_y, min_x + side - 1, min_y + side - 1];
        let w = side as usize;
        Candidate {
            score,
            bbox,
            mask: vec![true; w * w],
            area: w * w,
        }
    }

    // ---- rasterize_polygon ----

    #[test]
    fn rasterize_polygon_fills_a_simple_square() {
        // A square strictly larger than the 4x4 bbox on every side, so every
        // sampled pixel center (0.5..3.5) is unambiguously interior - corners
        // landing exactly on a pixel-center sample line would instead
        // exercise the scanline rounding convention, which isn't what this
        // test is about.
        let polygon = [(-1.0, -1.0), (5.0, -1.0), (5.0, 5.0), (-1.0, 5.0)];
        let bbox = [0, 0, 3, 3];
        let mask = Stardist::rasterize_polygon(&polygon, bbox);
        assert_eq!(mask.len(), 16);
        assert!(
            mask.iter().all(|&b| b),
            "every pixel center lies inside the square"
        );
    }

    #[test]
    fn rasterize_polygon_leaves_pixels_outside_a_triangle_unfilled() {
        // Right triangle spanning a 4x4 bbox: only the lower-left half should fill.
        let polygon = [(0.0, 0.0), (0.0, 4.0), (4.0, 4.0)];
        let bbox = [0, 0, 3, 3];
        let mask = Stardist::rasterize_polygon(&polygon, bbox);
        let local_w = 4usize;
        // Top-right corner pixel (row 0, col 3) is well outside the triangle.
        assert!(!mask[0 * local_w + 3]);
        // Bottom-left corner pixel (row 3, col 0) is well inside the triangle.
        assert!(mask[3 * local_w + 0]);
    }

    // ---- build_candidate ----

    #[test]
    fn build_candidate_clips_bbox_to_image_bounds() {
        let polygon = [(-5.0, -5.0), (5.0, -5.0), (5.0, 5.0), (-5.0, 5.0)];
        let candidate = Stardist::build_candidate(0.9, &polygon, 4, 4)
            .expect("a large square overlapping the image must produce a candidate");
        assert_eq!(
            candidate.bbox,
            [0, 0, 3, 3],
            "bbox must be clamped to the image dimensions"
        );
    }

    #[test]
    fn build_candidate_returns_none_for_a_zero_area_polygon() {
        // All three points on one horizontal line - zero area, no scanline crossing pair.
        let polygon = [(1.0, 1.0), (2.0, 1.0), (3.0, 1.0)];
        assert!(Stardist::build_candidate(0.9, &polygon, 10, 10).is_none());
    }

    #[test]
    fn build_candidate_returns_none_when_entirely_outside_the_image() {
        let polygon = [(-10.0, -10.0), (-6.0, -10.0), (-6.0, -6.0), (-10.0, -6.0)];
        assert!(Stardist::build_candidate(0.9, &polygon, 10, 10).is_none());
    }

    // ---- build_candidates ----

    #[test]
    fn build_candidates_only_keeps_grid_cells_above_the_probability_threshold() {
        let algo = stardist(0.5, 0.3);
        // 1x2 grid; only the second cell clears the threshold.
        let prob_flat = [0.1, 0.9];
        // 4 rays, both cells given the same modest radius.
        let dist_flat = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        let candidates = algo.build_candidates(&prob_flat, &dist_flat, 1, 2, 4, 20, 10);
        assert_eq!(candidates.len(), 1);
        assert_eq!(candidates[0].score, 0.9);
    }

    #[test]
    fn build_candidates_scales_ray_distance_by_the_grid_to_image_ratio() {
        let algo = stardist(0.5, 0.3);
        // 1x1 grid over a 10x10 image (scale factor 10 in both axes) with a
        // unit ray distance - the resulting polygon should span roughly
        // radius-10 around the image center, clipped to the image bounds.
        let prob_flat = [1.0];
        let n_rays = 8;
        let dist_flat = vec![1.0f32; n_rays];
        let candidates = algo.build_candidates(&prob_flat, &dist_flat, 1, 1, n_rays, 10, 10);
        assert_eq!(candidates.len(), 1);
        let bbox = candidates[0].bbox;
        // The unscaled radius (1px) would collapse to a single pixel; scaled
        // by ~10x it must span a large fraction of the 10x10 image instead.
        assert!(
            bbox[2] - bbox[0] > 4,
            "bbox must reflect the scaled radius, not the raw 1px ray length"
        );
    }

    // ---- bbox_overlaps ----

    #[test]
    fn bbox_overlaps_true_for_touching_boxes() {
        assert!(Stardist::bbox_overlaps(&[0, 0, 3, 3], &[3, 3, 6, 6]));
    }

    #[test]
    fn bbox_overlaps_false_for_disjoint_boxes() {
        assert!(!Stardist::bbox_overlaps(&[0, 0, 3, 3], &[4, 4, 6, 6]));
    }

    // ---- iou ----

    #[test]
    fn iou_is_zero_for_disjoint_candidates() {
        let a = square(0, 0, 4, 0.9);
        let b = square(10, 10, 4, 0.5);
        assert_eq!(Stardist::iou(&a, &b), 0.0);
    }

    #[test]
    fn iou_is_zero_when_bboxes_overlap_but_the_masks_share_no_pixel() {
        // A occupies only the top-left 2x2 of its 4x4 bbox [0,0,3,3]; B
        // occupies only the bottom-right 2x2 of its 4x4 bbox [2,2,5,5]. The
        // two *bboxes* overlap (in [2,3]x[2,3]), but neither mask is filled
        // there - `iou`'s own bbox fast-path (checked by
        // `iou_is_zero_for_disjoint_candidates` above) can't catch this; only
        // the mask intersection loop finding zero true pixels does.
        let mut a_mask = vec![false; 16];
        for row in 0..2 {
            for col in 0..2 {
                a_mask[row * 4 + col] = true;
            }
        }
        let a = Candidate {
            score: 0.9,
            bbox: [0, 0, 3, 3],
            mask: a_mask,
            area: 4,
        };
        let mut b_mask = vec![false; 16];
        for row in 2..4 {
            for col in 2..4 {
                b_mask[row * 4 + col] = true;
            }
        }
        let b = Candidate {
            score: 0.5,
            bbox: [2, 2, 5, 5],
            mask: b_mask,
            area: 4,
        };
        assert_eq!(Stardist::iou(&a, &b), 0.0);
    }

    #[test]
    fn iou_is_one_for_identical_candidates() {
        let a = square(0, 0, 4, 0.9);
        let b = square(0, 0, 4, 0.5);
        assert_eq!(Stardist::iou(&a, &b), 1.0);
    }

    #[test]
    fn iou_computes_the_intersection_over_union_ratio() {
        // Two 4x4 squares (area 16 each) overlapping in a 2x4 strip (area 8).
        // union = 16 + 16 - 8 = 24, iou = 8/24.
        let a = square(0, 0, 4, 0.9);
        let b = square(2, 0, 4, 0.5);
        let iou = Stardist::iou(&a, &b);
        assert!((iou - 8.0 / 24.0).abs() < 1e-6, "got {iou}");
    }

    // ---- non_max_suppress ----

    #[test]
    fn non_max_suppress_keeps_every_candidate_when_none_overlap() {
        let candidates = vec![
            square(0, 0, 2, 0.9),
            square(10, 10, 2, 0.8),
            square(20, 20, 2, 0.7),
        ];
        let kept = Stardist::non_max_suppress(candidates, 0.3);
        assert_eq!(kept.len(), 3);
    }

    #[test]
    fn non_max_suppress_drops_the_lower_scoring_of_two_heavily_overlapping_candidates() {
        // Identical boxes (iou = 1.0) - well above any positive threshold.
        let candidates = vec![square(0, 0, 4, 0.4), square(0, 0, 4, 0.9)];
        let kept = Stardist::non_max_suppress(candidates, 0.3);
        assert_eq!(kept.len(), 1);
        assert_eq!(
            kept[0].score, 0.9,
            "the higher-scoring candidate must survive"
        );
    }

    #[test]
    fn non_max_suppress_keeps_both_when_overlap_is_below_threshold() {
        // 4x4 squares shifted by 3px overlap in a 1x4 strip: iou = 4/(16+16-4) = 4/28 ≈ 0.14.
        let candidates = vec![square(0, 0, 4, 0.9), square(3, 0, 4, 0.8)];
        let kept = Stardist::non_max_suppress(candidates, 0.3);
        assert_eq!(
            kept.len(),
            2,
            "0.14 overlap must not exceed a 0.3 suppression threshold"
        );
    }

    #[test]
    fn non_max_suppress_returns_candidates_sorted_by_descending_score() {
        let candidates = vec![
            square(0, 0, 2, 0.3),
            square(50, 50, 2, 0.9),
            square(100, 100, 2, 0.6),
        ];
        let kept = Stardist::non_max_suppress(candidates, 0.3);
        let scores: Vec<f32> = kept.iter().map(|c| c.score).collect();
        assert_eq!(scores, vec![0.9, 0.6, 0.3]);
    }

    // ---- write_instances ----

    #[test]
    fn write_instances_resets_stale_buffers_when_no_candidates_survive() {
        // Regression: `seg_slice`/`inst_slice` may be reused buffers from an
        // earlier segmentation pass. Finding zero surviving candidates must
        // still reset them to background, not preserve stale labels.
        let mut seg = vec![9u32; 9]; // stale sentinel, 3x3 image
        let mut inst = vec![9u32; 9];
        Stardist::write_instances(&[], 3, 7, &mut seg, &mut inst);
        assert_eq!(seg, vec![0; 9]);
        assert_eq!(inst, vec![0; 9]);
    }

    #[test]
    fn write_instances_resets_stale_pixels_outside_every_surviving_candidate() {
        // A single 2x2 candidate in the top-left corner of a 4x4 image;
        // every other pixel must end up reset to background even though it
        // starts out with a stale nonzero sentinel.
        let candidates = vec![square(0, 0, 2, 0.9)];
        let mut seg = vec![9u32; 16];
        let mut inst = vec![9u32; 16];
        Stardist::write_instances(&candidates, 4, 7, &mut seg, &mut inst);

        for y in 0..4 {
            for x in 0..4 {
                let idx = y * 4 + x;
                if x < 2 && y < 2 {
                    assert_eq!(inst[idx], 1, "pixel ({x},{y}) should belong to instance 1");
                    assert_eq!(
                        seg[idx], 7,
                        "pixel ({x},{y}) should carry the foreground class"
                    );
                } else {
                    assert_eq!(
                        inst[idx], 0,
                        "pixel ({x},{y}) should be reset to background"
                    );
                    assert_eq!(seg[idx], 0, "pixel ({x},{y}) should be reset to background");
                }
            }
        }
    }

    #[test]
    fn write_instances_skips_the_false_cells_of_a_non_rectangular_mask() {
        // A 2x2 bbox candidate whose mask is true only on its diagonal -
        // unlike every other `write_instances` test above, which uses
        // `square()`'s fully-true mask and so never exercises the per-cell
        // `!candidate.mask[local_idx]` skip.
        let candidate = Candidate {
            score: 0.9,
            bbox: [0, 0, 1, 1],
            mask: vec![true, false, false, true], // (0,0) and (1,1) only
            area: 2,
        };
        let mut seg = vec![0u32; 4];
        let mut inst = vec![0u32; 4];
        Stardist::write_instances(&[candidate], 2, 7, &mut seg, &mut inst);

        assert_eq!(
            inst,
            vec![1, 0, 0, 1],
            "only the diagonal cells are claimed"
        );
        assert_eq!(seg, vec![7, 0, 0, 7]);
    }

    #[test]
    fn write_instances_first_candidate_wins_a_contested_pixel() {
        // Two overlapping 2x2 squares passed directly (bypassing
        // `non_max_suppress`, which would normally drop one of them) - the
        // pixel they both claim must go to whichever candidate appears
        // first in `kept`, not be overwritten by the second.
        let candidates = vec![square(0, 0, 2, 0.9), square(1, 0, 2, 0.5)];
        let mut seg = vec![0u32; 9]; // 3x3 image
        let mut inst = vec![0u32; 9];
        Stardist::write_instances(&candidates, 3, 7, &mut seg, &mut inst);

        // Column 1, rows 0-1 is covered by both candidates - instance 1
        // (the first candidate) must own it.
        assert_eq!(
            inst[0 * 3 + 1],
            1,
            "the first candidate must win the contested pixel"
        );
        assert_eq!(inst[1 * 3 + 1], 1);
        // The first candidate's own exclusive column (x=0) is untouched by the contest.
        assert_eq!(inst[0 * 3 + 0], 1);
        // The second candidate's exclusive column (x=2) still gets its own instance id.
        assert_eq!(inst[0 * 3 + 2], 2);
    }

    // ---- NaN model output no longer panics ----
    //
    // A degenerate/blank tile or an incompatible model export can make the
    // TorchScript model emit NaN. Fixed at three points: `build_candidates`
    // skips non-finite scores before a `Candidate` is even built,
    // `build_candidate` rejects any polygon with a non-finite vertex, and
    // both `partial_cmp(...).unwrap()` sort call sites were changed to
    // `unwrap_or(Ordering::Equal)` as defense in depth for callers that
    // bypass the two filters above (as these tests deliberately do, to
    // exercise that fallback directly).

    #[test]
    fn rasterize_polygon_does_not_panic_on_a_nan_vertex() {
        let polygon = [(0.0, 0.0), (f32::NAN, 4.0), (4.0, 4.0)];
        // No assertion on the resulting mask's shape - a NaN vertex makes the
        // polygon meaningless, but `rasterize_polygon` (unlike
        // `build_candidate`, which rejects this polygon outright) is a
        // low-level primitive that must at least not crash.
        let _ = Stardist::rasterize_polygon(&polygon, [0, 0, 3, 3]);
    }

    #[test]
    fn non_max_suppress_does_not_panic_when_a_candidate_score_is_nan() {
        let candidates = vec![square(0, 0, 2, f32::NAN), square(10, 10, 2, 0.5)];
        let kept = Stardist::non_max_suppress(candidates, 0.3);
        assert_eq!(kept.len(), 2, "non-overlapping candidates are both kept");
    }

    #[test]
    fn build_candidates_skips_a_nan_score_grid_cell() {
        let algo = stardist(0.5, 0.3);
        // 1x2 grid: cell 0 has a NaN score (e.g. degenerate model output),
        // cell 1 clears the threshold normally.
        let prob_flat = [f32::NAN, 0.9];
        let dist_flat = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        let candidates = algo.build_candidates(&prob_flat, &dist_flat, 1, 2, 4, 20, 10);
        assert_eq!(candidates.len(), 1);
        assert_eq!(candidates[0].score, 0.9);
    }

    #[test]
    fn build_candidate_returns_none_for_a_polygon_with_a_nan_vertex() {
        let polygon = [(1.0, 1.0), (f32::NAN, 5.0), (5.0, 5.0), (1.0, 5.0)];
        assert!(Stardist::build_candidate(0.9, &polygon, 10, 10).is_none());
    }

    #[test]
    fn build_candidate_returns_none_for_a_polygon_with_an_infinite_vertex() {
        let polygon = [(1.0, 1.0), (f32::INFINITY, 5.0), (5.0, 5.0), (1.0, 5.0)];
        assert!(Stardist::build_candidate(0.9, &polygon, 10, 10).is_none());
    }
}
