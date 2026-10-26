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

// The model's input/output conventions.

/// The input size the TorchScript export was traced at: the model takes
/// exactly one `[1, 3, 640, 640]` image.
const WINDOW: usize = 640;
/// Fill value for the parts of a window past the tile border. YOLOv5
/// models are trained on images padded to their input size with gray 114
/// of 255, so this is the value the model knows as "no image here" - any
/// other value could look like structure to it.
const PAD_VALUE: f32 = 114.0 / 255.0;
/// At most this many detections (the most confident ones) are kept per
/// window after NMS - bounds the mask work per window. A window holding
/// more objects loses its least confident ones.
const MAX_DETECTIONS: usize = 300;
/// Columns a prediction row starts with: box centre x, centre y, width and
/// height (window px), then the objectness. The class scores follow, then
/// (segmentation models) one coefficient per mask prototype.
const BOX_COLUMNS: usize = 5;
/// A box this close (px) to a window edge inside the tile is cut off by
/// that edge; the overlapping neighbor window sees the object whole.
const EDGE_MARGIN: f32 = 2.0;

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
/// parts outside the tile get [`PAD_VALUE`].
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

/// A prediction row that passed the confidence filter.
struct Candidate {
    /// Objectness x score of `class`.
    score: f32,
    class: usize,
    /// `[x1, y1, x2, y2]` in window px, clipped to the window.
    bbox: [f32; 4],
    /// Index of the row in the prediction, for its mask coefficients.
    row: usize,
}

impl Decoder {
    /// Turns one window's raw output into detections in tile coordinates.
    ///
    /// Steps: rows below the confidence threshold are dropped and the rest
    /// decoded into boxes ([`Self::candidate`]), overlapping boxes of a
    /// class reduced to the most confident one ([`greedy_nms`]), boxes cut
    /// by a window edge inside the tile dropped, and for segmentation
    /// models every survivor's mask assembled from the prototypes
    /// ([`MaskAssembler`]). Finally window coordinates become tile
    /// coordinates, with everything in the padding past the tile cut away.
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
        let (rows, columns) = (sizes[1] as usize, sizes[2] as usize);
        let coefficient_count = proto.map(|p| p.size()[1] as usize).unwrap_or(0);
        // At least one class score besides box, objectness and coefficients.
        if columns < BOX_COLUMNS + 1 + coefficient_count {
            return Err(format!(
                "YOLOv5 prediction has {columns} columns, too few for {coefficient_count} mask \
                 coefficients"
            ));
        }
        let class_count = columns - BOX_COLUMNS - coefficient_count;
        let values = Vec::<f32>::try_from(&pred.reshape([-1])).map_err(err)?;
        let row = |i: usize| &values[i * columns..(i + 1) * columns];

        let candidates: Vec<Candidate> = (0..rows)
            .filter_map(|i| self.candidate(i, row(i), class_count))
            .collect();
        let mut kept = greedy_nms(candidates, self.iou_threshold, MAX_DETECTIONS);

        // A box reaching (within `EDGE_MARGIN`) a window edge that lies
        // inside the tile is most likely an object cut off by that edge; the
        // neighboring window overlaps far enough to see it whole. Edges on
        // the tile border are real borders and keep their boxes.
        let w = WINDOW as f32;
        let inner = [x0 > 0, y0 > 0, x0 + WINDOW < width, y0 + WINDOW < height];
        kept.retain(|c| {
            let cut = [
                c.bbox[0] <= EDGE_MARGIN,
                c.bbox[1] <= EDGE_MARGIN,
                c.bbox[2] >= w - EDGE_MARGIN,
                c.bbox[3] >= w - EDGE_MARGIN,
            ];
            !(0..4).any(|side| inner[side] && cut[side])
        });

        let assembler = match proto {
            Some(proto) if !kept.is_empty() => {
                Some(MaskAssembler::new(proto, self.mask_threshold)?)
            }
            _ => None,
        };
        // A mask is cropped to its box at prototype resolution and only then
        // scaled up, so its interpolated edge can reach up to one prototype
        // pixel past the box: its rectangle is that much larger.
        let spill = assembler.as_ref().map_or(0.0, MaskAssembler::stride);
        // The window's pixels that show the tile, not padding.
        let (visible_w, visible_h) = ((width - x0).min(WINDOW), (height - y0).min(WINDOW));

        let mut scratch = MaskScratch::default();
        let mut out = Vec::with_capacity(kept.len());
        for c in kept {
            let [x1, y1, x2, y2] = c.bbox;
            let rect = [
                (x1 - spill).max(0.0).floor() as usize,
                (y1 - spill).max(0.0).floor() as usize,
                ((x2 + spill).ceil() as usize).min(visible_w),
                ((y2 + spill).ceil() as usize).min(visible_h),
            ];
            if rect[2] <= rect[0] || rect[3] <= rect[1] {
                continue; // entirely in the padding
            }
            let mask = assembler.as_ref().map(|a| {
                let coefficients = &row(c.row)[BOX_COLUMNS + class_count..];
                a.mask(coefficients, &c.bbox, rect, &mut scratch)
            });
            out.push(Detection {
                score: c.score,
                class: c.class,
                bbox: [
                    (x1 + x0 as f32).min(width as f32),
                    (y1 + y0 as f32).min(height as f32),
                    (x2 + x0 as f32).min(width as f32),
                    (y2 + y0 as f32).min(height as f32),
                ],
                mask,
                mask_rect: [
                    rect[0] + x0,
                    rect[1] + y0,
                    rect[2] - rect[0],
                    rect[3] - rect[1],
                ],
            });
        }
        Ok(out)
    }

    /// Row `index` as a candidate, `None` when it isn't confident enough.
    ///
    /// The confidence of a detection is its objectness (probability that
    /// the box holds an object) times the score of its best class
    /// (probability of that class, given an object) - the class-specific
    /// confidence of the YOLO detector family. Ties between class scores go
    /// to the lower class index. The box comes as centre and size and is
    /// turned into corners, clipped to the window.
    fn candidate(&self, index: usize, row: &[f32], class_count: usize) -> Option<Candidate> {
        let objectness = row[BOX_COLUMNS - 1];
        // Class scores are probabilities (<= 1), so the confidence can't
        // exceed the objectness: skip the class scan for hopeless rows.
        if objectness <= self.confidence_threshold {
            return None;
        }
        let mut class = 0;
        let mut class_score = f32::MIN;
        for (i, &s) in row[BOX_COLUMNS..BOX_COLUMNS + class_count]
            .iter()
            .enumerate()
        {
            if s > class_score {
                (class, class_score) = (i, s);
            }
        }
        let score = objectness * class_score;
        if score <= self.confidence_threshold {
            return None;
        }
        let (cx, cy, half_w, half_h) = (row[0], row[1], row[2] / 2.0, row[3] / 2.0);
        let w = WINDOW as f32;
        Some(Candidate {
            score,
            class,
            bbox: [
                (cx - half_w).clamp(0.0, w),
                (cy - half_h).clamp(0.0, w),
                (cx + half_w).clamp(0.0, w),
                (cy + half_h).clamp(0.0, w),
            ],
            row: index,
        })
    }
}

/// Greedy non-maximum suppression within each class, keeping at most
/// `limit` boxes: candidates are visited by descending score (ties in row
/// order) and a candidate is kept unless a kept box of its class overlaps
/// it by more than `iou_threshold` (intersection over union). Boxes of
/// different classes never suppress each other (the semantics of
/// torchvision's `batched_nms`). The result is sorted by descending score.
fn greedy_nms(mut candidates: Vec<Candidate>, iou_threshold: f32, limit: usize) -> Vec<Candidate> {
    // Stable sort: equal scores keep the model's row order.
    candidates.sort_by(|a, b| b.score.total_cmp(&a.score));
    let mut kept: Vec<Candidate> = Vec::with_capacity(limit.min(candidates.len()));
    for c in candidates {
        if kept.len() == limit {
            break;
        }
        let suppressed = kept
            .iter()
            .any(|k| k.class == c.class && box_iou(&k.bbox, &c.bbox) > iou_threshold);
        if !suppressed {
            kept.push(c);
        }
    }
    kept
}

/// Builds instance masks the YOLACT way (Bolya et al., "YOLACT: Real-time
/// Instance Segmentation", ICCV 2019, Sec. 3): the network predicts a small
/// set of prototype masks for the whole window plus, per detection, one
/// coefficient per prototype. A detection's mask is the sigmoid of the
/// coefficient-weighted sum of the prototypes (`M = sigmoid(P C^T)`, Sec.
/// 3.3), cropped to the detection's box at prototype resolution (zero
/// outside), scaled up bilinearly to window resolution and thresholded.
///
/// Everything is evaluated only where it can matter: the weighted sum over
/// the prototype pixels around the box, the upsampling over the
/// detection's rectangle - no full-window mask per detection.
struct MaskAssembler {
    /// The prototypes on the CPU, `[count][height * width]`.
    prototypes: Vec<f32>,
    width: usize,
    height: usize,
    threshold: f32,
}

/// Buffers reused from one mask to the next.
#[derive(Default)]
struct MaskScratch {
    /// The cropped assembled mask at prototype resolution, over the
    /// prototype pixels the upsampling reads.
    assembled: Vec<f32>,
    /// Per output column: the two prototype columns it interpolates
    /// (relative to the `assembled` area) and the weight of the second.
    columns: Vec<(usize, usize, f32)>,
}

impl MaskAssembler {
    /// Takes the `[1, count, height, width]` prototype tensor.
    fn new(proto: &Tensor, threshold: f32) -> Result<Self> {
        let err = |e: tch::TchError| format!("YOLOv5 mask failed: {e}");
        let proto = proto
            .f_to_device(Device::Cpu)
            .map_err(err)?
            .to_kind(Kind::Float);
        let s = proto.size();
        let (height, width) = (s[2] as usize, s[3] as usize);
        if width == 0 || height == 0 {
            return Err("YOLOv5 model returned empty mask prototypes".into());
        }
        Ok(Self {
            prototypes: Vec::<f32>::try_from(&proto.reshape([-1])).map_err(err)?,
            width,
            height,
            threshold,
        })
    }

    /// One prototype pixel in window px (the larger of both axes).
    fn stride(&self) -> f32 {
        WINDOW as f32 / self.width.min(self.height) as f32
    }

    /// The mask of a detection with `coefficients` and `bbox` (window px)
    /// over `rect` (`[x1, y1, x2, y2]`, window px, exclusive), row-major.
    fn mask(
        &self,
        coefficients: &[f32],
        bbox: &[f32; 4],
        rect: [usize; 4],
        scratch: &mut MaskScratch,
    ) -> Vec<bool> {
        let (sx, sy) = (
            self.width as f32 / WINDOW as f32,
            self.height as f32 / WINDOW as f32,
        );
        // Prototype pixels the bilinear upsampling of `rect` reads.
        let (read_x0, read_x1) = (
            sample(rect[0], sx, self.width).0,
            sample(rect[2] - 1, sx, self.width).1 + 1,
        );
        let (read_y0, read_y1) = (
            sample(rect[1], sy, self.height).0,
            sample(rect[3] - 1, sy, self.height).1 + 1,
        );
        let area_w = read_x1 - read_x0;

        // The crop: prototype pixel `p` lies in the box when its index,
        // scaled to window px, does (`x1 <= p / sx < x2`). Only the part of
        // the crop that is read gets assembled; the rest stays zero.
        let crop = |lo: f32, hi: f32, len: usize, read: (usize, usize)| {
            let start = (lo.ceil().max(0.0) as usize).clamp(read.0, read.1);
            let end = (hi.ceil().max(0.0) as usize).min(len).clamp(start, read.1);
            start..end
        };
        let crop_x = crop(bbox[0] * sx, bbox[2] * sx, self.width, (read_x0, read_x1));
        let crop_y = crop(bbox[1] * sy, bbox[3] * sy, self.height, (read_y0, read_y1));

        let assembled = &mut scratch.assembled;
        assembled.clear();
        assembled.resize(area_w * (read_y1 - read_y0), 0.0);
        // Weighted sum, one prototype at a time (contiguous rows).
        let plane_len = self.width * self.height;
        for (&c, plane) in coefficients
            .iter()
            .zip(self.prototypes.chunks_exact(plane_len))
        {
            for py in crop_y.clone() {
                let src = &plane[py * self.width + crop_x.start..py * self.width + crop_x.end];
                let at = (py - read_y0) * area_w + crop_x.start - read_x0;
                for (dst, &p) in assembled[at..at + src.len()].iter_mut().zip(src) {
                    *dst += c * p;
                }
            }
        }
        for py in crop_y.clone() {
            let at = (py - read_y0) * area_w + crop_x.start - read_x0;
            for v in &mut assembled[at..at + crop_x.len()] {
                *v = 1.0 / (1.0 + (-*v).exp());
            }
        }

        // Bilinear upsampling to window px over `rect`, then the threshold.
        scratch.columns.clear();
        scratch.columns.extend((rect[0]..rect[2]).map(|x| {
            let (i0, i1, t) = sample(x, sx, self.width);
            (i0 - read_x0, i1 - read_x0, t)
        }));
        let mut mask = Vec::with_capacity((rect[2] - rect[0]) * (rect[3] - rect[1]));
        for y in rect[1]..rect[3] {
            let (j0, j1, s) = sample(y, sy, self.height);
            let top = &assembled[(j0 - read_y0) * area_w..][..area_w];
            let bottom = &assembled[(j1 - read_y0) * area_w..][..area_w];
            mask.extend(scratch.columns.iter().map(|&(i0, i1, t)| {
                let v = (1.0 - s) * ((1.0 - t) * top[i0] + t * top[i1])
                    + s * ((1.0 - t) * bottom[i0] + t * bottom[i1]);
                v > self.threshold
            }));
        }
        mask
    }
}

/// Where output pixel `i` samples an axis of `len` source pixels scaled by
/// `scale` (source px per output px): pixel centres at `+ 0.5` on both
/// grids, positions before the first centre clamped to it. Gives the two
/// neighboring source pixels and the weight of the second.
fn sample(i: usize, scale: f32, len: usize) -> (usize, usize, f32) {
    let u = ((i as f32 + 0.5) * scale - 0.5).max(0.0);
    let i0 = (u as usize).min(len - 1);
    (i0, (i0 + 1).min(len - 1), u - i0 as f32)
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
        // more than `MAX_DETECTIONS`: only the most confident 300 stay.
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

    #[test]
    fn sample_uses_pixel_centres_and_clamps_at_both_ends() {
        // 160 -> 640: output pixel 0 lies before the first source centre.
        assert_eq!(sample(0, 0.25, 160), (0, 1, 0.0));
        assert_eq!(sample(2, 0.25, 160), (0, 1, 0.125));
        assert_eq!(sample(5, 0.25, 160), (0, 1, 0.875));
        assert_eq!(sample(6, 0.25, 160), (1, 2, 0.125));
        assert_eq!(sample(639, 0.25, 160), (159, 159, 0.375));
    }

    #[test]
    fn greedy_nms_keeps_tied_scores_in_row_order() {
        let candidate = |row, x| Candidate {
            score: 0.5,
            class: 0,
            bbox: [x, 0.0, x + 10.0, 10.0],
            row,
        };
        let kept = greedy_nms(
            vec![candidate(0, 0.0), candidate(1, 1.0), candidate(2, 50.0)],
            0.45,
            MAX_DETECTIONS,
        );
        let rows: Vec<usize> = kept.iter().map(|c| c.row).collect();
        assert_eq!(rows, vec![0, 2]);
    }

    /// The mask the straightforward way: the whole window's assembled,
    /// cropped mask, upsampled pixel by pixel - for checking that
    /// `MaskAssembler::mask` (which only evaluates around the box) agrees.
    fn full_window_mask(a: &MaskAssembler, coefficients: &[f32], bbox: &[f32; 4]) -> Vec<f32> {
        let (pw, ph) = (a.width, a.height);
        let (sx, sy) = (pw as f32 / WINDOW as f32, ph as f32 / WINDOW as f32);
        let mut cropped = vec![0f32; pw * ph];
        for y in 0..ph {
            for x in 0..pw {
                let inside = x as f32 >= bbox[0] * sx
                    && (x as f32) < bbox[2] * sx
                    && y as f32 >= bbox[1] * sy
                    && (y as f32) < bbox[3] * sy;
                if inside {
                    let z: f32 = coefficients
                        .iter()
                        .enumerate()
                        .map(|(k, c)| c * a.prototypes[k * pw * ph + y * pw + x])
                        .sum();
                    cropped[y * pw + x] = 1.0 / (1.0 + (-z).exp());
                }
            }
        }
        let mut out = vec![0f32; WINDOW * WINDOW];
        for y in 0..WINDOW {
            let (j0, j1, s) = sample(y, sy, ph);
            for x in 0..WINDOW {
                let (i0, i1, t) = sample(x, sx, pw);
                out[y * WINDOW + x] = (1.0 - s)
                    * ((1.0 - t) * cropped[j0 * pw + i0] + t * cropped[j0 * pw + i1])
                    + s * ((1.0 - t) * cropped[j1 * pw + i0] + t * cropped[j1 * pw + i1]);
            }
        }
        out
    }

    #[test]
    fn masks_match_the_full_window_computation() {
        // Varying prototypes; boxes in the middle, at the window borders
        // and at fractional positions.
        let values: Vec<f32> = (0..3 * 160 * 160)
            .map(|i| ((i * 7919 % 211) as f32 / 211.0 - 0.5) * 4.0)
            .collect();
        let proto = Tensor::from_slice(&values).reshape([1, 3, 160, 160]);
        let a = MaskAssembler::new(&proto, 0.5).unwrap();
        let coefficients = [1.5, -0.7, 2.0];
        let mut scratch = MaskScratch::default();
        for bbox in [
            [100.0, 140.0, 200.0, 240.0],
            [0.0, 0.0, 37.3, 22.9],
            [601.5, 590.2, 640.0, 640.0],
            [13.1, 400.7, 14.2, 401.1],
        ] {
            let spill = a.stride();
            let rect = [
                (bbox[0] - spill).max(0.0).floor() as usize,
                (bbox[1] - spill).max(0.0).floor() as usize,
                ((bbox[2] + spill).ceil() as usize).min(WINDOW),
                ((bbox[3] + spill).ceil() as usize).min(WINDOW),
            ];
            let mask = a.mask(&coefficients, &bbox, rect, &mut scratch);
            let full = full_window_mask(&a, &coefficients, &bbox);
            let rw = rect[2] - rect[0];
            for y in 0..WINDOW {
                for x in 0..WINDOW {
                    let in_rect =
                        (rect[0]..rect[2]).contains(&x) && (rect[1]..rect[3]).contains(&y);
                    let expected = full[y * WINDOW + x] > 0.5;
                    let got = in_rect && mask[(y - rect[1]) * rw + x - rect[0]];
                    assert_eq!(got, expected, "{bbox:?} at ({x}, {y})");
                }
            }
        }
    }
}
