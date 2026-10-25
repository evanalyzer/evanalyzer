//! # yolov5
//!
//! **Author:** Joachim Danmayr
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.
//!
//! YOLOv5 inference: the RGB image resized to the size the model runs at,
//! run in overlapping 640 x 640 windows, every window's output decoded into
//! detections (with masks for segmentation exports) in image coordinates.
//! The global NMS across windows is the caller's.

use super::{Result, scheduler::TorchScheduler};
use crate::api::{Detection, Image, Response, box_iou};
use std::path::Path;
use tch::{Device, IValue, Kind, Tensor};

/// The thresholds of one run.
struct Decoder {
    confidence_threshold: f32,
    iou_threshold: f32,
    mask_threshold: f32,
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn run(
    scheduler: &TorchScheduler,
    model_path: &Path,
    width: usize,
    height: usize,
    window_overlap: i32,
    confidence_threshold: f32,
    iou_threshold: f32,
    mask_threshold: f32,
    image: Image<'_>,
) -> Result<Response> {
    if image.channels != 3 {
        return Err(format!(
            "YOLOv5 needs an RGB image, got {} channels",
            image.channels
        ));
    }
    let (tile_width, tile_height) = (image.width, image.height);
    let n = tile_width * tile_height;
    let decoder = Decoder {
        confidence_threshold,
        iou_threshold,
        mask_threshold,
    };
    scheduler.run(
        model_path,
        |e| {
            format!(
                "Failed to load YOLOv5 model from {}: {e}. The model must be a TorchScript \
                 export (yolov5 export.py --include torchscript), not a training checkpoint.",
                model_path.display()
            )
        },
        |model, device| {
            let mut rgb = [
                image.data[..n].to_vec(),
                image.data[n..2 * n].to_vec(),
                image.data[2 * n..3 * n].to_vec(),
            ];
            if (width, height) != (tile_width, tile_height) {
                rgb = resize_planes(rgb, tile_width, tile_height, width, height)?;
            }

            let mut detections = Vec::new();
            for &y0 in &window_starts(height, window_overlap) {
                for &x0 in &window_starts(width, window_overlap) {
                    let window = window_tensor(&rgb, width, height, x0, y0).to_device(device);
                    let output = tch::no_grad(|| model.forward_is(&[IValue::Tensor(window)]))
                        .map_err(|e| format!("YOLOv5 inference failed: {e}"))?;
                    let (pred, proto) = split_outputs(output)?;
                    detections.extend(decoder.decode_window(
                        &pred,
                        proto.as_ref(),
                        x0,
                        y0,
                        width,
                        height,
                    )?);
                }
            }
            Ok(Response::Yolov5 { detections })
        },
    )
}

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

/// Resizes the R, G, B planes from `src` to `dst` size (bilinear, with
/// antialiasing when shrinking - as image libraries resize).
fn resize_planes(
    planes: [Vec<f32>; 3],
    src_width: usize,
    src_height: usize,
    dst_width: usize,
    dst_height: usize,
) -> Result<[Vec<f32>; 3]> {
    let err = |e: tch::TchError| format!("YOLOv5 resize failed: {e}");
    let data: Vec<f32> = planes.concat();
    let resized = Tensor::from_slice(&data)
        .reshape([1, 3, src_height as i64, src_width as i64])
        .f_internal_upsample_bilinear2d_aa([dst_height as i64, dst_width as i64], false, None, None)
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
    if len <= WINDOW {
        return vec![0];
    }
    let stride = WINDOW.saturating_sub(overlap.max(0) as usize).max(1);
    let span = len - WINDOW;
    let count = span.div_ceil(stride) + 1;
    (0..count)
        .map(|i| if i + 1 == count { span } else { i * stride })
        .collect()
}

/// The `[1, 3, 640, 640]` model input for the window at (`x0`, `y0`);
/// parts outside the tile are padded with YOLOv5's gray.
fn window_tensor(rgb: &[Vec<f32>; 3], width: usize, height: usize, x0: usize, y0: usize) -> Tensor {
    let w = WINDOW;
    let mut data = vec![PAD_VALUE; 3 * w * w];
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
fn split_outputs(output: IValue) -> Result<(Tensor, Option<Tensor>)> {
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
        "YOLOv5 model returned no [1, N, C] prediction tensor - is this a YOLOv5 export?"
            .to_string()
    })?;
    Ok((pred, proto))
}

impl Decoder {
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
    ) -> Result<Vec<Detection>> {
        let err = |e: tch::TchError| format!("YOLOv5 inference failed: {e}");
        let pred = pred
            .f_to_device(Device::Cpu)
            .map_err(err)?
            .to_kind(Kind::Float);
        let sizes = pred.size();
        let (n, cols) = (sizes[1] as usize, sizes[2] as usize);
        let n_masks = proto.map(|p| p.size()[1] as usize).unwrap_or(0);
        if cols < 6 + n_masks {
            return Err(format!(
                "YOLOv5 prediction has {cols} columns, too few for {n_masks} mask coefficients"
            ));
        }
        let n_classes = cols - 5 - n_masks;
        let values = Vec::<f32>::try_from(&pred.reshape([-1])).map_err(err)?;

        let w = WINDOW as f32;
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
            if kept.len() >= MAX_DETECTIONS {
                break;
            }
            let suppressed = kept.iter().any(|(k, _)| {
                k.class == candidate.0.class
                    && box_iou(&k.bbox, &candidate.0.bbox) > self.iou_threshold
            });
            if !suppressed {
                kept.push(candidate);
            }
        }

        // Boxes touching a window edge that lies inside the tile are cut off
        // there - the neighboring window (overlap) sees them whole.
        let inner = [x0 > 0, y0 > 0, x0 + WINDOW < width, y0 + WINDOW < height];
        kept.retain(|(d, _)| {
            !((inner[0] && d.bbox[0] <= EDGE_MARGIN)
                || (inner[1] && d.bbox[1] <= EDGE_MARGIN)
                || (inner[2] && d.bbox[2] >= w - EDGE_MARGIN)
                || (inner[3] && d.bbox[3] >= w - EDGE_MARGIN))
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
            let spill = if masks.is_some() { MASK_SPILL } else { 0.0 };
            let rx = (d.bbox[0] - spill).max(0.0).floor() as usize;
            let ry = (d.bbox[1] - spill).max(0.0).floor() as usize;
            let rx2 = ((d.bbox[2] + spill).ceil() as usize)
                .min(WINDOW)
                .min(width - x0);
            let ry2 = ((d.bbox[3] + spill).ceil() as usize)
                .min(WINDOW)
                .min(height - y0);
            if rx2 <= rx || ry2 <= ry {
                continue;
            }
            let (rw, rh) = (rx2 - rx, ry2 - ry);
            d.mask = masks.as_ref().map(|m| {
                let mut mask = Vec::with_capacity(rw * rh);
                for y in ry..ry2 {
                    let row = i * WINDOW * WINDOW + y * WINDOW;
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
    fn masks(&self, proto: &Tensor, kept: &[(Detection, Vec<f32>)]) -> Result<Vec<bool>> {
        let err = |e: tch::TchError| format!("YOLOv5 mask failed: {e}");
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
        let (sx, sy) = (pw as f32 / WINDOW as f32, ph as f32 / WINDOW as f32);
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
        let w = WINDOW as i64;
        let upsampled = masks
            .unsqueeze(0)
            .upsample_bilinear2d([w, w], false, None, None)
            .squeeze_dim(0)
            .gt(self.mask_threshold as f64);
        Vec::<bool>::try_from(&upsampled.reshape([-1])).map_err(err)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn decoder() -> Decoder {
        Decoder {
            confidence_threshold: 0.25,
            iou_threshold: 0.45,
            mask_threshold: 0.5,
        }
    }

    /// One prediction row: box (centre x/y, width, height, in window px),
    /// objectness, `classes` class scores and `coefficients`.
    fn row(
        cx: f32,
        cy: f32,
        w: f32,
        h: f32,
        obj: f32,
        classes: &[f32],
        coefficients: &[f32],
    ) -> Vec<f32> {
        let mut r = vec![cx, cy, w, h, obj];
        r.extend_from_slice(classes);
        r.extend_from_slice(coefficients);
        r
    }

    fn pred(rows: &[Vec<f32>]) -> Tensor {
        let cols = rows[0].len() as i64;
        let flat: Vec<f32> = rows.iter().flatten().copied().collect();
        Tensor::from_slice(&flat).reshape([1, rows.len() as i64, cols])
    }

    /// `[1, 1, 160, 160]` prototypes of value `v`.
    fn proto(v: f32) -> Tensor {
        Tensor::full([1, 1, 160, 160], v as f64, (Kind::Float, Device::Cpu))
    }

    #[test]
    fn window_starts_cover_the_tile_with_the_last_window_flush_with_the_end() {
        assert_eq!(window_starts(500, 128), vec![0]);
        assert_eq!(window_starts(640, 128), vec![0]);
        assert_eq!(window_starts(1000, 128), vec![0, 360]);
        assert_eq!(window_starts(2048, 128), vec![0, 512, 1024, 1408]);
        for len in [641, 1000, 2048, 3001] {
            let starts = window_starts(len, 128);
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
    fn decode_window_places_a_detection_and_its_mask_in_tile_coordinates() {
        // A 100x100 box at (100..200, 140..240) in the window at (512, 0)
        // of a 2000x700 tile; coefficient 5 on all-one prototypes: a full mask.
        let p = pred(&[row(150.0, 190.0, 100.0, 100.0, 0.9, &[0.1, 0.8], &[5.0])]);
        let out = decoder()
            .decode_window(&p, Some(&proto(1.0)), 512, 0, 2000, 700)
            .unwrap();
        assert_eq!(out.len(), 1);
        let d = &out[0];
        assert_eq!(d.class, 1);
        assert!((d.score - 0.72).abs() < 1e-6, "objectness x class score");
        assert_eq!(d.bbox, [612.0, 140.0, 712.0, 240.0]);
        let [rx, ry, rw, rh] = d.mask_rect;
        assert!(rx <= 612 && ry <= 140 && rx + rw >= 712 && ry + rh >= 240);
        let filled = d.mask.as_ref().unwrap().iter().filter(|&&m| m).count();
        assert!(
            (filled as i64 - 100 * 100).abs() <= 2 * 400,
            "mask about the box: {filled}"
        );
    }

    #[test]
    fn decode_window_drops_unconfident_and_overlapping_detections() {
        let p = pred(&[
            row(100.0, 100.0, 40.0, 40.0, 0.2, &[0.9], &[]), // objectness too low
            row(300.0, 300.0, 40.0, 40.0, 0.9, &[0.2], &[]), // 0.18 overall: too low
            row(500.0, 500.0, 40.0, 40.0, 0.9, &[0.9], &[]), // kept
            row(502.0, 500.0, 40.0, 40.0, 0.8, &[0.9], &[]), // same class, overlaps: merged
        ]);
        let out = decoder().decode_window(&p, None, 0, 0, 640, 640).unwrap();
        assert_eq!(out.len(), 1);
        assert!((out[0].score - 0.81).abs() < 1e-6);
        assert!(
            out[0].mask.is_none(),
            "a detection model gives filled boxes"
        );
    }

    #[test]
    fn decode_window_drops_boxes_cut_by_an_inner_window_edge_only() {
        // A box touching the window's left edge: cut off if that edge lies
        // inside the tile (window at x 512), real if it is the tile border.
        let p = pred(&[row(20.0, 300.0, 40.0, 40.0, 0.9, &[0.9], &[])]);
        assert!(
            decoder()
                .decode_window(&p, None, 512, 0, 2000, 640)
                .unwrap()
                .is_empty()
        );
        assert_eq!(
            decoder()
                .decode_window(&p, None, 0, 0, 2000, 640)
                .unwrap()
                .len(),
            1
        );
    }

    #[test]
    fn decode_window_clips_boxes_to_a_small_tile() {
        // A 300x200 tile in a padded window: a box reaching into the padding
        // is cut at the tile border.
        let p = pred(&[row(280.0, 100.0, 80.0, 40.0, 0.9, &[0.9], &[])]);
        let d = &decoder().decode_window(&p, None, 0, 0, 300, 200).unwrap()[0];
        assert_eq!(d.bbox, [240.0, 80.0, 300.0, 120.0]);
        assert_eq!(d.mask_rect, [240, 80, 60, 40]);
    }

    #[test]
    fn decode_window_rejects_a_prediction_with_too_few_columns() {
        let p = pred(&[vec![0.0; 5]]);
        assert!(
            decoder()
                .decode_window(&p, Some(&proto(1.0)), 0, 0, 640, 640)
                .is_err()
        );
    }

    #[test]
    fn negative_mask_coefficients_give_an_empty_mask() {
        let p = pred(&[row(320.0, 320.0, 100.0, 100.0, 0.9, &[0.9], &[-5.0])]);
        let d = &decoder()
            .decode_window(&p, Some(&proto(1.0)), 0, 0, 640, 640)
            .unwrap()[0];
        assert!(d.mask.as_ref().unwrap().iter().all(|&m| !m));
    }

    #[test]
    fn split_outputs_finds_prediction_and_prototypes() {
        let tuple = IValue::Tuple(vec![
            IValue::Tensor(proto(1.0)),
            IValue::Tensor(pred(&[vec![0.0; 6]])),
        ]);
        let (p, prototypes) = split_outputs(tuple).unwrap();
        assert_eq!(p.dim(), 3);
        assert_eq!(prototypes.unwrap().dim(), 4);

        let (_, none) = split_outputs(IValue::Tensor(pred(&[vec![0.0; 6]]))).unwrap();
        assert!(none.is_none(), "a detection model has no prototypes");

        assert!(split_outputs(IValue::Tensor(proto(1.0))).is_err());
    }

    #[test]
    fn split_outputs_skips_values_that_are_not_tensors() {
        let output = IValue::Tuple(vec![
            IValue::Int(7),
            IValue::GenericList(vec![IValue::Tensor(pred(&[vec![0.0; 6]]))]),
            IValue::Double(1.5),
        ]);
        let (p, prototypes) = split_outputs(output).unwrap();
        assert_eq!(p.dim(), 3);
        assert!(prototypes.is_none());
    }

    #[test]
    fn decode_window_keeps_at_most_max_detections_per_window() {
        // 21 x 21 = 441 separate 10x10 boxes (30 px apart, so NMS keeps all),
        // more than YOLOv5's `max_det`: only the most confident 300 stay.
        let mut rows = Vec::new();
        for gy in 0..21 {
            for gx in 0..21 {
                let i = gy * 21 + gx;
                let obj = 0.99 - i as f32 * 0.001;
                rows.push(row(
                    15.0 + gx as f32 * 30.0,
                    15.0 + gy as f32 * 30.0,
                    10.0,
                    10.0,
                    obj,
                    &[1.0],
                    &[],
                ));
            }
        }
        let out = decoder()
            .decode_window(&pred(&rows), None, 0, 0, 640, 640)
            .unwrap();
        assert_eq!(out.len(), MAX_DETECTIONS);
        let lowest_kept = out.iter().map(|d| d.score).fold(f32::MAX, f32::min);
        assert!(
            (lowest_kept - (0.99 - 299.0 * 0.001)).abs() < 1e-5,
            "the 300 most confident are kept, lowest {lowest_kept}"
        );
    }

    #[test]
    fn decode_window_drops_a_box_lying_entirely_in_the_padding() {
        // A 300x200 tile in a padded window: a box at x 400..440 is past the
        // tile's right border, so nothing of it is left to rasterize.
        let p = pred(&[row(420.0, 100.0, 40.0, 40.0, 0.9, &[0.9], &[])]);
        assert!(
            decoder()
                .decode_window(&p, None, 0, 0, 300, 200)
                .unwrap()
                .is_empty()
        );
    }
}
