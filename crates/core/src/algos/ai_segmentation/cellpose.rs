//! # cellpose
//!
//! **Author:** Joachim Danmayr
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.
//!
//! Portions ported from Cellpose, Copyright © 2020 Howard Hughes Medical
//! Institute, BSD-3-Clause - see the third-party notices (About > Licenses).

use std::path::PathBuf;

use evanalyzer_ai::{Image, Request, Response, Task};
use evanalyzer_cfg::core_types::{CitationMetadata, InternalErrors, SegmentationClass};
use macros::CommandsMeta;

use crate::{
    ai_devices,
    algos::{ExecutionScope, ImageAlgorithm},
    pipeline::{pipeline_cache::GlobalPipelineCache, pipeline_context::PipelineContext},
};

/// Instance segmentation using a Cellpose-SAM model
///
/// [AI Cellpose Segmentation] -> [Extract Objects]
///
/// Object segmentation using Cellpose-SAM model which can be downloaded from
/// https://evanalyzer.org/downloads/#ai-models
///
/// Cellpose-SAM is a biological segmentation model that integrates the pretrained transformer
/// architecture of Meta's Segment Anything Model (SAM) with the Cellpose framework to accurately
/// predict vector flow fields for dense cellular structures.
/// By combining these methods, it achieves "superhuman generalization," outperforming the average
/// accuracy of human annotators and reaching near-optimal cell masking performance. [1]
///
/// [1] Pachitariu, M., Rariden, M., & Stringer, C. (2025). Cellpose-SAM: superhuman generalization for cellular segmentation. bioRxiv. doi.org
///
#[derive(CommandsMeta)]
#[cmdsmeta(
    category = "segment",
    next = "extract",
    display_name = "AI Cellpose Segmentation"
)]
pub struct Cellpose {
    /// Path to a TorchScript-exported Cellpose model (`torch.jit.script`/`torch.jit.trace`).
    #[cmdsmeta(file_extensions = "pt,pth")]
    pub model_path: PathBuf,

    /// The class assigned to pixels of every detected object. All other
    /// pixels are assigned `SegmentationClass::BACKGROUND`.
    #[cmdsmeta(default = SegmentationClass(1))]
    pub segmentation_class: SegmentationClass,

    /// Number of input channels the model expects. The grayscale image goes in
    /// channel 0; any further channels are zero-filled. Cellpose-SAM's
    /// patch-embedding convolution only has weights for up to 3 input
    /// channels: `2` (cytoplasm + optional nucleus) is standard, `1` is for
    /// single-channel exports.
    #[cmdsmeta(default = 2, min = 1, max = 3, step = 1, visibility = Advanced)]
    pub input_channels: i32,

    /// Cell probability above which a pixel takes part in the flow dynamics and
    /// can be assigned to an object. The raw cell-probability logits are passed
    /// through a sigmoid first, so this is a probability in `[0, 1]` (Cellpose's
    /// default logit threshold of `0` corresponds to `0.5`).
    #[cmdsmeta(default = 0.5, min = 0.0, max = 1.0, step = 0.01)]
    pub probability_threshold: f32,

    /// Number of Euler integration steps used to follow the flow field. Higher
    /// values let pixels of large cells reach their sink at the cost of runtime;
    /// Cellpose's default is `200`.
    #[cmdsmeta(default = 200, min = 1, max = 1000, step = 1, visibility = Advanced)]
    pub flow_iterations: i32,

    /// Minimum object size, in pixels. After the dynamics, any instance smaller
    /// than this is removed (its pixels become background). `0` disables the filter.
    #[cmdsmeta(default = 15, min = 0, max = 100000, step = 1)]
    pub min_object_size: i32,

    /// Longest image side, in pixels, the image is scaled down to before
    /// segmentation; the masks are scaled back up to the original size
    /// afterwards. Smaller values make large cells look like the cell sizes
    /// the model was trained on and are faster. The scale is taken from the
    /// full image, so every tile is scaled the same. `0` keeps the full
    /// resolution (the Cellpose web demo uses `1000`).
    #[cmdsmeta(default = 0, min = 0, max = 100000, step = 1, optional = true, visibility = Advanced)]
    pub max_resize: i32,

    /// Flow error threshold: an object whose shape doesn't match the flows
    /// the model predicted (mean squared error above this value) is removed.
    /// Increase to keep more objects, decrease to keep only clean ones. `0`
    /// disables the check (Cellpose's default is `0.4`).
    #[cmdsmeta(default = 0.4, min = 0.0, max = 10.0, step = 0.01, optional = true, visibility = Advanced)]
    pub flow_threshold: f32,

    /// Build the objects from the flows exactly like Cellpose does: pixels
    /// follow the interpolated flows, an object only starts where more than
    /// 10 pixels end up together, and pixels that reach no such spot become
    /// background. Off, every spot any pixel ends up at starts an object,
    /// which can join touching cells. Cellpose also fills the holes inside
    /// each object - add a Fill Object Holes step after this one for that.
    #[cmdsmeta(default = true, optional = true, visibility = Advanced)]
    pub cellpose_postprocessing: bool,

    /// Copy the gray image into every input channel instead of filling the
    /// extra channels with zeros. With `input_channels = 3` this matches
    /// Cellpose run on an RGB image whose channels are (nearly) equal.
    #[cmdsmeta(default = true, optional = true, visibility = Advanced)]
    pub replicate_gray_channel: bool,
}

impl ImageAlgorithm for Cellpose {
    fn execute(
        &self,
        ctx: &mut PipelineContext,
        _cache: &mut GlobalPipelineCache,
    ) -> Result<(), InternalErrors> {
        let scale = self.resize_factor(ctx.full_image_size());
        let (input_image, segmentation_map, instance_map) =
            ctx.get_f32_gray_segmentation_and_instances_mut()?;
        let size = input_image.size();
        let (width, height) = (size.width, size.height);
        // The size the model and the dynamics run at.
        let net_width = ((width as f64 * scale).round() as usize).max(1);
        let net_height = ((height as f64 * scale).round() as usize).max(1);
        let resized = (net_width, net_height) != (width, height);
        let (w, h) = (net_width, net_height);

        let request = Request {
            model: self.model_path.clone(),
            task: Task::Cellpose {
                net_width,
                net_height,
                input_channels: self.input_channels,
                replicate_gray_channel: self.replicate_gray_channel,
            },
        };
        let image = Image {
            width,
            height,
            channels: 1,
            data: input_image.as_slice(),
        };
        let Response::Cellpose {
            flow_y,
            flow_x,
            cell_prob,
        } = ai_devices::infer(&request, image)?
        else {
            return Err(InternalErrors::Generic(
                "Cellpose: the AI engine gave another kind of answer".into(),
            ));
        };

        let mut labels = self.masks_from_flows(&flow_y, &flow_x, &cell_prob, w, h);
        if resized {
            labels = super::resize_labels_nearest(&labels, w, h, width, height);
        }

        self.write_instances(
            &labels,
            segmentation_map.as_slice_mut(),
            instance_map.as_slice_mut(),
        );

        Ok(())
    }

    fn name(&self) -> &'static str {
        "Cellpose"
    }

    fn cite(&self) -> Vec<&'static CitationMetadata> {
        vec![&CitationMetadata {
            cite_key: "pachitariu2025cellposesam",
            title: "Cellpose-SAM: superhuman generalization for cellular segmentation",
            authors: &["Marius Pachitariu", "Michael Rariden", "Carsen Stringer"],
            year: 2025,
            container: Some("bioRxiv"),
            doi: Some("10.1101/2025.04.28.651001"),
            url: Some("https://doi.org/10.1101/2025.04.28.651001"),
            pages: None,
        }]
    }

    fn execution_scope(&self) -> ExecutionScope {
        ExecutionScope::Tile
    }
}

impl Cellpose {
    /// Cellpose scales the training flows by `5`, so the predicted flows are
    /// divided by the same factor before integration to keep each Euler step
    /// near one pixel.
    const FLOW_SCALE: f32 = 5.0;

    /// Turns the predicted flows and cell probabilities (at the size the
    /// model ran at) into an instance label map (`0` = background) - the
    /// dynamics, mask building and the flow error check.
    fn masks_from_flows(
        &self,
        flow_y: &[f32],
        flow_x: &[f32],
        cell_prob: &[f32],
        width: usize,
        height: usize,
    ) -> Vec<u32> {
        // Pixels above the cell-probability threshold take part in the dynamics.
        let is_cell: Vec<bool> = cell_prob
            .iter()
            .map(|&p| p >= self.probability_threshold)
            .collect();

        let mut labels = if self.cellpose_postprocessing {
            let final_positions =
                self.follow_flows_interpolated(flow_y, flow_x, &is_cell, width, height);
            Self::masks_from_seeds(&final_positions, &is_cell, width, height)
        } else {
            let final_positions = self.follow_flows(flow_y, flow_x, &is_cell, width, height);
            Self::label_sinks(&final_positions, &is_cell, width, height)
        };
        if self.flow_threshold > 0.0 {
            Self::remove_bad_flow_masks(
                &mut labels,
                flow_y,
                flow_x,
                width,
                height,
                self.flow_threshold,
            );
        }
        labels
    }

    /// Cellpose's dynamics (`dynamics.steps_interp`): every cell pixel takes
    /// `flow_iterations` Euler steps along the flows divided by `FLOW_SCALE`,
    /// sampled with bilinear interpolation. Like Cellpose, the flows are zero
    /// outside the cell pixels, and the sampling reproduces its
    /// `grid_sample(align_corners=False)` on `[0, L-1]`-normalized
    /// coordinates: position `p` samples pixel `p * L / (L - 1) - 0.5`,
    /// pixels outside the image count as zero. Returns each pixel's final
    /// `(y, x)`, truncated to whole pixels (non-cell pixels keep their own).
    fn follow_flows_interpolated(
        &self,
        flow_y: &[f32],
        flow_x: &[f32],
        is_cell: &[bool],
        width: usize,
        height: usize,
    ) -> Vec<(usize, usize)> {
        use rayon::prelude::*;
        let niter = self.flow_iterations.max(1) as usize;
        let masked = |flow: &[f32]| -> Vec<f32> {
            flow.iter()
                .zip(is_cell)
                .map(|(&f, &c)| if c { f / Self::FLOW_SCALE } else { 0.0 })
                .collect()
        };
        let (fy, fx) = (masked(flow_y), masked(flow_x));
        // Sampling coordinate of position `p` along an axis of length `len`.
        let to_sample = |p: f32, len: usize| -> f32 {
            if len < 2 {
                0.0
            } else {
                p * len as f32 / (len - 1) as f32 - 0.5
            }
        };
        let bilinear = |field: &[f32], uy: f32, ux: f32| -> f32 {
            let (y0, x0) = (uy.floor(), ux.floor());
            let (ty, tx) = (uy - y0, ux - x0);
            let mut value = 0.0;
            for (dy, wy) in [(0, 1.0 - ty), (1, ty)] {
                for (dx, wx) in [(0, 1.0 - tx), (1, tx)] {
                    let (yy, xx) = (y0 as i64 + dy, x0 as i64 + dx);
                    if yy >= 0 && xx >= 0 && (yy as usize) < height && (xx as usize) < width {
                        value += wy * wx * field[yy as usize * width + xx as usize];
                    }
                }
            }
            value
        };
        let (max_y, max_x) = ((height - 1) as f32, (width - 1) as f32);
        (0..width * height)
            .into_par_iter()
            .map(|idx| {
                let (y, x) = (idx / width, idx % width);
                if !is_cell[idx] {
                    return (y, x);
                }
                let (mut py, mut px) = (y as f32, x as f32);
                for _ in 0..niter {
                    let (uy, ux) = (to_sample(py, height), to_sample(px, width));
                    let step_y = bilinear(&fy, uy, ux);
                    let step_x = bilinear(&fx, uy, ux);
                    py = (py + step_y).clamp(0.0, max_y);
                    px = (px + step_x).clamp(0.0, max_x);
                }
                (py as usize, px as usize)
            })
            .collect()
    }

    /// Cellpose's mask building (`dynamics.get_masks_torch`): a histogram of
    /// where the cell pixels ended up is searched for peaks (local maxima in
    /// a 5x5 window with more than 10 pixels); each peak grows for 5 steps
    /// (3x3) into the histogram bins holding more than 2 pixels, within 5
    /// pixels of the peak. A cell pixel takes the label of the peak region
    /// it ended up in, or `0` if it reached none. Where regions overlap, the
    /// peak with more pixels wins. Labels are `1..=n` in order of first
    /// appearance. (Cellpose's removal of objects above 40 % of the image is
    /// left to a size filter in the pipeline.)
    fn masks_from_seeds(
        final_positions: &[(usize, usize)],
        is_cell: &[bool],
        width: usize,
        height: usize,
    ) -> Vec<u32> {
        const PAD: usize = 20;
        let (hw, hh) = (width + 2 * PAD, height + 2 * PAD);
        let mut hist = vec![0u32; hw * hh];
        for (idx, &(y, x)) in final_positions.iter().enumerate() {
            if is_cell[idx] {
                hist[(y + PAD) * hw + x + PAD] += 1;
            }
        }
        let local_max = Self::max_filter(&hist, hw, hh, 2);
        let mut seeds: Vec<usize> = (0..hw * hh)
            .filter(|&i| hist[i] > 10 && hist[i] == local_max[i])
            .collect();
        seeds.sort_by_key(|&i| hist[i]); // stable: row-major among equals
        if seeds.is_empty() {
            return vec![0; width * height];
        }

        // Grow every seed inside its 11x11 neighborhood; seeds never sit
        // closer than PAD to the histogram border, so the window always fits.
        const R: usize = 5;
        const SIDE: usize = 2 * R + 1;
        let mut region = vec![0u32; hw * hh];
        let mut mask = [0u32; SIDE * SIDE];
        for (k, &seed) in seeds.iter().enumerate() {
            let (sy, sx) = (seed / hw, seed % hw);
            let bin = |wy: usize, wx: usize| hist[(sy + wy - R) * hw + sx + wx - R];
            mask.fill(0);
            mask[R * SIDE + R] = 1;
            for _ in 0..5 {
                mask = Self::max_filter(&mask, SIDE, SIDE, 1).try_into().unwrap();
                for wy in 0..SIDE {
                    for wx in 0..SIDE {
                        if bin(wy, wx) <= 2 {
                            mask[wy * SIDE + wx] = 0;
                        }
                    }
                }
            }
            for wy in 0..SIDE {
                for wx in 0..SIDE {
                    if mask[wy * SIDE + wx] != 0 {
                        let i = (sy + wy - R) * hw + sx + wx - R;
                        region[i] = region[i].max(k as u32 + 1);
                    }
                }
            }
        }

        let mut labels = vec![0u32; width * height];
        for (idx, &(y, x)) in final_positions.iter().enumerate() {
            if is_cell[idx] {
                labels[idx] = region[(y + PAD) * hw + x + PAD];
            }
        }
        Self::renumber(&mut labels);
        labels
    }

    /// Maximum over a `(2 * radius + 1)`² window (cut off at the borders).
    fn max_filter(values: &[u32], width: usize, height: usize, radius: usize) -> Vec<u32> {
        let mut rows = vec![0u32; values.len()];
        for y in 0..height {
            for x in 0..width {
                let (x0, x1) = (x.saturating_sub(radius), (x + radius).min(width - 1));
                rows[y * width + x] = values[y * width + x0..=y * width + x1]
                    .iter()
                    .copied()
                    .max()
                    .unwrap_or(0);
            }
        }
        let mut out = vec![0u32; values.len()];
        for y in 0..height {
            let (y0, y1) = (y.saturating_sub(radius), (y + radius).min(height - 1));
            for x in 0..width {
                out[y * width + x] = (y0..=y1).map(|yy| rows[yy * width + x]).max().unwrap_or(0);
            }
        }
        out
    }

    /// Renumbers labels to `1..=n` in order of first appearance (row-major).
    fn renumber(labels: &mut [u32]) {
        let mut map = std::collections::HashMap::new();
        for l in labels.iter_mut() {
            if *l != 0 {
                let next = map.len() as u32 + 1;
                *l = *map.entry(*l).or_insert(next);
            }
        }
    }

    /// Factor the image is scaled by before segmentation: shrinks the full
    /// image's longest side to `max_resize`, never enlarges it.
    fn resize_factor(&self, full_image: kornia_image::ImageSize) -> f64 {
        let longest = full_image.width.max(full_image.height);
        if self.max_resize <= 0 || longest <= self.max_resize as usize {
            return 1.0;
        }
        self.max_resize as f64 / longest as f64
    }

    /// Cellpose's flow-error quality control (`dynamics.remove_bad_flow_masks`):
    /// recomputes the flows each object's shape implies (`masks_to_flows`, a
    /// diffusion from the object's center) and removes every object whose
    /// mean squared difference to the predicted flows (divided by
    /// `FLOW_SCALE`) is above `threshold`. Removed objects become background
    /// (`0`); the surviving labels are left as they are.
    fn remove_bad_flow_masks(
        labels: &mut [u32],
        flow_y: &[f32],
        flow_x: &[f32],
        width: usize,
        height: usize,
        threshold: f32,
    ) {
        let (implied_y, implied_x) = Self::masks_to_flows(labels, width, height);
        let max_label = labels.iter().copied().max().unwrap_or(0) as usize;
        let mut error_sum = vec![0f64; max_label + 1];
        let mut count = vec![0usize; max_label + 1];
        for (i, &label) in labels.iter().enumerate() {
            if label == 0 {
                continue;
            }
            let dy = implied_y[i] - flow_y[i] as f64 / Self::FLOW_SCALE as f64;
            let dx = implied_x[i] - flow_x[i] as f64 / Self::FLOW_SCALE as f64;
            error_sum[label as usize] += dy * dy + dx * dx;
            count[label as usize] += 1;
        }
        let bad: Vec<bool> = (0..=max_label)
            .map(|l| count[l] > 0 && error_sum[l] / count[l] as f64 > threshold as f64)
            .collect();
        for label in labels.iter_mut() {
            if bad[*label as usize] {
                *label = 0;
            }
        }
    }

    /// Unit flow vectors (dY, dX) per pixel implied by the label map - a port
    /// of Cellpose's `masks_to_flows_gpu`, kept faithful (including its
    /// quirks) so the flow error matches Cellpose's. Background pixels get
    /// `0`. Labels must be `1..=n` without gaps.
    fn masks_to_flows(labels: &[u32], width: usize, height: usize) -> (Vec<f64>, Vec<f64>) {
        use rayon::prelude::*;
        let n_labels = labels.iter().copied().max().unwrap_or(0) as usize;
        let mut flow_y = vec![0f64; width * height];
        let mut flow_x = vec![0f64; width * height];
        if n_labels == 0 {
            return (flow_y, flow_x);
        }

        // Bounding boxes and mean positions, both relative to the box like
        // Cellpose's `find_objects` slices (the rounding of the mean depends on it).
        let mut bbox = vec![[usize::MAX, usize::MAX, 0usize, 0usize]; n_labels + 1];
        for y in 0..height {
            for x in 0..width {
                let l = labels[y * width + x] as usize;
                if l == 0 {
                    continue;
                }
                let b = &mut bbox[l];
                b[0] = b[0].min(y);
                b[1] = b[1].min(x);
                b[2] = b[2].max(y);
                b[3] = b[3].max(x);
            }
        }
        // Padded by one pixel on every side, so every neighbor index is valid.
        let pw = width + 2;
        let padded = |y: usize, x: usize| (y + 1) * pw + (x + 1);
        let mut centers = vec![0usize; n_labels + 1];
        let mut max_ext = 0usize;
        for l in 1..=n_labels {
            let [y0, x0, y1, x1] = bbox[l];
            if y0 == usize::MAX {
                continue;
            }
            let (mut sy, mut sx, mut n) = (0usize, 0usize, 0usize);
            for y in y0..=y1 {
                for x in x0..=x1 {
                    if labels[y * width + x] as usize == l {
                        sy += y - y0;
                        sx += x - x0;
                        n += 1;
                    }
                }
            }
            let ym = (sy as f64 / n as f64).round_ties_even() as usize;
            let xm = (sx as f64 / n as f64).round_ties_even() as usize;
            let (mut cy, mut cx) = (y0 + ym, x0 + xm);
            if cy > y1 || cx > x1 || labels[cy * width + cx] as usize != l {
                // The mean lies outside the object: take its closest pixel,
                // the first one in row-major order on a tie.
                let mut best = usize::MAX;
                for y in y0..=y1 {
                    for x in x0..=x1 {
                        if labels[y * width + x] as usize != l {
                            continue;
                        }
                        let d = (y - y0).abs_diff(ym).pow(2) + (x - x0).abs_diff(xm).pow(2);
                        if d < best {
                            best = d;
                            (cy, cx) = (y, x);
                        }
                    }
                }
            }
            centers[l] = padded(cy, cx);
            max_ext = max_ext.max((y1 - y0 + 1) + (x1 - x0 + 1) + 2);
        }

        // Mask pixels with their 9 neighbors (center, up, down, left, right,
        // then the diagonals) and which of those belong to the same object.
        const OFFSETS: [(isize, isize); 9] = [
            (0, 0),
            (-1, 0),
            (1, 0),
            (0, -1),
            (0, 1),
            (-1, -1),
            (-1, 1),
            (1, -1),
            (1, 1),
        ];
        let label_at = |y: isize, x: isize| -> u32 {
            if y < 0 || x < 0 || y >= height as isize || x >= width as isize {
                0
            } else {
                labels[y as usize * width + x as usize]
            }
        };
        let mut pixels: Vec<(usize, usize)> = Vec::new();
        let mut neighbors: Vec<[usize; 9]> = Vec::new();
        let mut same: Vec<[bool; 9]> = Vec::new();
        for y in 0..height {
            for x in 0..width {
                let l = labels[y * width + x];
                if l == 0 {
                    continue;
                }
                let mut nb = [0usize; 9];
                let mut sm = [false; 9];
                for (k, (dy, dx)) in OFFSETS.iter().enumerate() {
                    let (ny, nx) = (y as isize + dy, x as isize + dx);
                    nb[k] = ((ny + 1) as usize) * pw + (nx + 1) as usize;
                    sm[k] = label_at(ny, nx) == l;
                }
                pixels.push((y, x));
                neighbors.push(nb);
                same.push(sm);
            }
        }

        // Diffusion from the centers: every step adds 1 at each center, then
        // replaces each pixel by the sum of its same-object neighbors / 9.
        let mut heat = vec![0f64; pw * (height + 2)];
        let mut next = vec![0f64; pixels.len()];
        for _ in 0..2 * max_ext {
            for &c in &centers[1..] {
                heat[c] += 1.0;
            }
            next.par_iter_mut().enumerate().for_each(|(i, v)| {
                let nb = &neighbors[i];
                let sm = &same[i];
                *v = (0..9).filter(|&k| sm[k]).map(|k| heat[nb[k]]).sum::<f64>() / 9.0;
            });
            for (i, nb) in neighbors.iter().enumerate() {
                heat[nb[0]] = next[i];
            }
        }

        // Normalized gradient. Like Cellpose, it reads the raw neighbor
        // values, also those of a touching object.
        for (i, &(y, x)) in pixels.iter().enumerate() {
            let nb = &neighbors[i];
            let dy = heat[nb[2]] - heat[nb[1]];
            let dx = heat[nb[4]] - heat[nb[3]];
            let norm = 1e-60 + (dy * dy + dx * dx).sqrt();
            flow_y[y * width + x] = dy / norm;
            flow_x[y * width + x] = dx / norm;
        }
        (flow_y, flow_x)
    }

    /// Advects each cell pixel along the flow field for `flow_iterations` Euler
    /// steps (Cellpose's `steps2D`), sampling the flow at the current integer
    /// position and clamping to the image bounds. Returns, for every pixel, the
    /// flattened index of its final position (cell pixels converge to the sink
    /// at their object's center; non-cell pixels keep their own index).
    fn follow_flows(
        &self,
        flow_y: &[f32],
        flow_x: &[f32],
        is_cell: &[bool],
        width: usize,
        height: usize,
    ) -> Vec<usize> {
        let niter = self.flow_iterations.max(1) as usize;
        let max_x = width as f32 - 1.0;
        let max_y = height as f32 - 1.0;

        let mut final_pos = vec![0usize; width * height];
        for y in 0..height {
            for x in 0..width {
                let idx = y * width + x;
                if !is_cell[idx] {
                    final_pos[idx] = idx;
                    continue;
                }

                let (mut py, mut px) = (y as f32, x as f32);
                for _ in 0..niter {
                    let sample = py.round() as usize * width + px.round() as usize;
                    py = (py + flow_y[sample] / Self::FLOW_SCALE).clamp(0.0, max_y);
                    px = (px + flow_x[sample] / Self::FLOW_SCALE).clamp(0.0, max_x);
                }
                final_pos[idx] = py.round() as usize * width + px.round() as usize;
            }
        }
        final_pos
    }

    /// Groups cell pixels into instances by their convergence sinks. A density
    /// map of all final positions is built and its occupied cells are labeled
    /// with 8-connected connected components; every cell pixel then inherits the
    /// label of the sink it landed in. Returns a per-pixel instance label
    /// (`0` = background).
    fn label_sinks(
        final_positions: &[usize],
        is_cell: &[bool],
        width: usize,
        height: usize,
    ) -> Vec<u32> {
        // Mark the sink cells (final positions of cell pixels).
        let mut is_sink = vec![false; width * height];
        for (idx, &pos) in final_positions.iter().enumerate() {
            if is_cell[idx] {
                is_sink[pos] = true;
            }
        }

        // 8-connected connected components over the sink cells.
        let mut sink_label = vec![0u32; width * height];
        let mut next_label = 1u32;
        let mut stack: Vec<usize> = Vec::new();
        for start in 0..(width * height) {
            if !is_sink[start] || sink_label[start] != 0 {
                continue;
            }
            sink_label[start] = next_label;
            stack.push(start);
            while let Some(p) = stack.pop() {
                let (cx, cy) = (p % width, p / width);
                for dy in -1i64..=1 {
                    for dx in -1i64..=1 {
                        if dx == 0 && dy == 0 {
                            continue;
                        }
                        let nx = cx as i64 + dx;
                        let ny = cy as i64 + dy;
                        if nx < 0 || ny < 0 || nx >= width as i64 || ny >= height as i64 {
                            continue;
                        }
                        let np = ny as usize * width + nx as usize;
                        if is_sink[np] && sink_label[np] == 0 {
                            sink_label[np] = next_label;
                            stack.push(np);
                        }
                    }
                }
            }
            next_label += 1;
        }

        // Propagate the sink label back to every cell pixel.
        let mut labels = vec![0u32; width * height];
        for (idx, &pos) in final_positions.iter().enumerate() {
            if is_cell[idx] {
                labels[idx] = sink_label[pos];
            }
        }
        labels
    }

    /// Drops instances smaller than `min_object_size`, renumbers the survivors to
    /// contiguous IDs starting at `1`, and rasterizes them into the segmentation
    /// and instance maps.
    fn write_instances(&self, labels: &[u32], seg_slice: &mut [u32], inst_slice: &mut [u32]) {
        let max_label = labels.iter().copied().max().unwrap_or(0) as usize;
        if max_label == 0 {
            // `seg_slice`/`inst_slice` may be reused buffers carrying stale
            // labels from an earlier segmentation pass in the same pipeline
            // (e.g. re-running Cellpose, or running it after another
            // instance-map-writing step) - per this command's documented
            // contract ("all other pixels are assigned BACKGROUND"), finding
            // zero cells must still reset the maps rather than leaving
            // whatever was there before.
            seg_slice.fill(0);
            inst_slice.fill(0);
            return;
        }

        let mut sizes = vec![0usize; max_label + 1];
        for &label in labels {
            sizes[label as usize] += 1;
        }

        let min_size = self.min_object_size.max(0) as usize;
        // Map original labels to compacted instance IDs, dropping small objects.
        let mut remap = vec![0u32; max_label + 1];
        let mut next_id = 1u32;
        for label in 1..=max_label {
            if sizes[label] >= min_size.max(1) {
                remap[label] = next_id;
                next_id += 1;
            }
        }

        let foreground_class = self.segmentation_class.as_u32();
        for (i, &label) in labels.iter().enumerate() {
            let instance_id = remap[label as usize];
            inst_slice[i] = instance_id;
            seg_slice[i] = if instance_id == 0 {
                0
            } else {
                foreground_class
            };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(feature = "libtorch-in-process")]
    use evanalyzer_ai::torch::test_support::{tch::Tensor, trace_and_save_model};

    /// The tile size Cellpose(-SAM) runs at: test models are traced at it.
    #[cfg(feature = "libtorch-in-process")]
    const TILE_SIZE: i64 = 256;
    use kornia_image::{Image, ImageSize};

    fn cellpose(min_object_size: i32) -> Cellpose {
        Cellpose {
            model_path: PathBuf::new(),
            segmentation_class: SegmentationClass(7),
            input_channels: 2,
            probability_threshold: 0.5,
            flow_iterations: 10,
            min_object_size,
            max_resize: 0,
            flow_threshold: 0.0,
            cellpose_postprocessing: false,
            replicate_gray_channel: false,
        }
    }

    fn gray_ctx(width: usize, height: usize, values: Vec<f32>) -> PipelineContext {
        let img = Image::<f32, 1>::new(ImageSize { width, height }, values).unwrap();
        PipelineContext::new_from_image_test(img).unwrap()
    }

    // ---- execute() - real TorchScript load + inference, see `test_support` ----

    #[test]
    fn execute_errors_when_the_model_path_does_not_exist() {
        let dir = tempfile::tempdir().unwrap();
        let cmd = Cellpose {
            model_path: dir.path().join("missing.pt"),
            ..cellpose(0)
        };
        let mut ctx = gray_ctx(2, 2, vec![0.0; 4]);
        let mut cache = GlobalPipelineCache::default();

        let err = cmd.execute(&mut ctx, &mut cache).unwrap_err();
        assert!(matches!(err, InternalErrors::Generic(_)));
    }

    #[cfg(feature = "libtorch-in-process")]
    /// dY=0, dX=0 everywhere (a cell pixel never moves, so it is its own
    /// sink), cell-probability logit derived directly from the input pixel
    /// value: `value * 20 - 10`, i.e. an input near `1.0` saturates the
    /// post-sigmoid probability near `1.0` (cell) and near `0.0` saturates
    /// near `0.0` (background) - comfortably on either side of the default
    /// 0.5 threshold. Two *adjacent* cell pixels each become their own sink
    /// at zero flow, but those sinks are themselves 8-connected, so
    /// `label_sinks` still merges them into one instance - this is the real
    /// tensor-plumbing test, the merge logic itself is covered directly by
    /// `label_sinks_gives_the_same_label_to_pixels_converging_to_adjacent_sinks`.
    ///
    /// Always traced at `TILE_SIZE` (256x256): `run_model_tiled`
    /// invokes the model at that fixed size regardless of the logical image
    /// size passed to `execute()` (see `gray_ctx`), matching how a real
    /// Cellpose-SAM export behaves.
    fn flow_free_cellpose_model(in_channels: i64) -> (tempfile::TempDir, std::path::PathBuf) {
        trace_and_save_model(in_channels, TILE_SIZE, TILE_SIZE, |x| {
            let image_channel = x.narrow(1, 0, 1);
            let flow_y = image_channel.zeros_like();
            let flow_x = image_channel.zeros_like();
            let cell_logit = image_channel * 20.0 - 10.0;
            Tensor::cat(&[flow_y, flow_x, cell_logit], 1)
        })
    }

    #[cfg(feature = "libtorch-in-process")]
    #[test]
    fn execute_end_to_end_merges_adjacent_cell_pixels_into_one_instance() {
        let (_dir, model_path) = flow_free_cellpose_model(2);
        let cmd = Cellpose {
            model_path,
            input_channels: 2,
            min_object_size: 0,
            ..cellpose(0)
        };
        let mut ctx = gray_ctx(3, 1, vec![0.0, 1.0, 1.0]);
        let mut cache = GlobalPipelineCache::default();
        cmd.execute(&mut ctx, &mut cache).unwrap();

        let seg = ctx.get_segmentation_map().unwrap();
        assert_eq!(seg.as_slice(), &[0u32, 7, 7]);
        let inst = ctx.get_instance_map().unwrap();
        assert_eq!(inst.as_slice()[0], 0);
        assert_eq!(
            inst.as_slice()[1],
            inst.as_slice()[2],
            "two adjacent cell pixels must share one instance id"
        );
        assert_ne!(inst.as_slice()[1], 0);
    }

    #[cfg(feature = "libtorch-in-process")]
    #[test]
    fn execute_single_input_channel_skips_the_zero_fill_padding() {
        // input_channels = 1 takes the `image` tensor directly (no
        // concatenated zero channels) - a model traced for a genuine 1-channel
        // input exercises that branch instead of the zero-fill one above.
        let (_dir, model_path) = flow_free_cellpose_model(1);
        let cmd = Cellpose {
            model_path,
            input_channels: 1,
            min_object_size: 0,
            ..cellpose(0)
        };
        let mut ctx = gray_ctx(2, 1, vec![0.0, 1.0]);
        let mut cache = GlobalPipelineCache::default();
        cmd.execute(&mut ctx, &mut cache).unwrap();

        let seg = ctx.get_segmentation_map().unwrap();
        assert_eq!(seg.as_slice(), &[0u32, 7]);
    }

    #[cfg(feature = "libtorch-in-process")]
    #[test]
    fn execute_errors_when_the_model_output_has_too_few_dimensions() {
        let (_dir, model_path) = trace_and_save_model(2, TILE_SIZE, TILE_SIZE, |x| {
            // Collapses [1,2,256,256] down to rank 2, well short of the
            // required `[1, C, 256, 256]`.
            x.narrow(1, 0, 1).squeeze_dim(0).squeeze_dim(0)
        });
        let cmd = Cellpose {
            model_path,
            min_object_size: 0,
            ..cellpose(0)
        };
        let mut ctx = gray_ctx(2, 1, vec![0.0, 1.0]);
        let mut cache = GlobalPipelineCache::default();

        let err = cmd.execute(&mut ctx, &mut cache).unwrap_err();
        assert!(matches!(err, InternalErrors::Generic(msg) if msg.contains("too few dimensions")));
    }

    #[cfg(feature = "libtorch-in-process")]
    #[test]
    fn execute_errors_when_the_model_output_has_fewer_than_three_channels() {
        let (_dir, model_path) = trace_and_save_model(2, TILE_SIZE, TILE_SIZE, |x| {
            let c = x.narrow(1, 0, 1);
            Tensor::cat(&[c.shallow_clone(), c.shallow_clone()], 1)
        });
        let cmd = Cellpose {
            model_path,
            min_object_size: 0,
            ..cellpose(0)
        };
        let mut ctx = gray_ctx(2, 1, vec![0.0, 1.0]);
        let mut cache = GlobalPipelineCache::default();

        let err = cmd.execute(&mut ctx, &mut cache).unwrap_err();
        assert!(
            matches!(err, InternalErrors::Generic(msg) if msg.contains("fewer than 3 channels"))
        );
    }

    #[cfg(feature = "libtorch-in-process")]
    #[test]
    fn execute_errors_when_the_model_output_tile_size_is_wrong() {
        // A real Cellpose-SAM export always returns a 256x256 tile (its ViT
        // encoder can't run at any other size); a model that doesn't must be
        // rejected rather than silently misaligned during stitching.
        let (_dir, model_path) = trace_and_save_model(2, TILE_SIZE, TILE_SIZE, |x| {
            let c = x.narrow(1, 0, 1);
            let cropped = c.narrow(3, 0, 128); // half the tile width
            Tensor::cat(
                &[cropped.shallow_clone(), cropped.shallow_clone(), cropped],
                1,
            )
        });
        let cmd = Cellpose {
            model_path,
            min_object_size: 0,
            ..cellpose(0)
        };
        let mut ctx = gray_ctx(4, 1, vec![0.0; 4]);
        let mut cache = GlobalPipelineCache::default();

        let err = cmd.execute(&mut ctx, &mut cache).unwrap_err();
        assert!(
            matches!(err, InternalErrors::Generic(msg) if msg.contains("expected exactly 256x256"))
        );
    }

    // ---- tiling ----

    // ---- follow_flows ----

    #[test]
    fn follow_flows_leaves_non_cell_pixels_at_their_own_index() {
        let algo = cellpose(0);
        let flow_y = vec![1.0; 9];
        let flow_x = vec![1.0; 9];
        let is_cell = vec![false; 9];
        let final_pos = algo.follow_flows(&flow_y, &flow_x, &is_cell, 3, 3);
        assert_eq!(final_pos, (0..9).collect::<Vec<_>>());
    }

    #[test]
    fn follow_flows_converges_every_cell_pixel_to_a_single_sink() {
        // 1D row of 5 pixels; the flow field points every pixel toward x=2 at
        // exactly one pixel per Euler step (FLOW_SCALE cancels the /5 in
        // follow_flows), so a handful of iterations is enough to converge
        // from either end.
        let algo = cellpose(0);
        let flow_y = vec![0.0; 5];
        let flow_x = vec![5.0, 5.0, 0.0, -5.0, -5.0];
        let is_cell = vec![true; 5];
        let final_pos = algo.follow_flows(&flow_y, &flow_x, &is_cell, 5, 1);
        assert_eq!(final_pos, vec![2, 2, 2, 2, 2]);
    }

    #[test]
    fn follow_flows_clamps_trajectories_to_the_image_bounds() {
        // Every pixel pushed hard off the right/bottom edge must clamp to the
        // last valid row/column, not wrap or index out of bounds.
        let algo = cellpose(0);
        let flow_y = vec![100.0; 4];
        let flow_x = vec![100.0; 4];
        let is_cell = vec![true; 4];
        let final_pos = algo.follow_flows(&flow_y, &flow_x, &is_cell, 2, 2);
        // width=2, height=2 -> bottom-right pixel is index 1*2+1 = 3.
        assert_eq!(final_pos, vec![3, 3, 3, 3]);
    }

    // ---- label_sinks ----

    #[test]
    fn label_sinks_gives_the_same_label_to_pixels_converging_to_adjacent_sinks() {
        // Two adjacent (8-connected) sink cells merge into one connected
        // component, so cell pixels converging to either must share a label.
        let final_positions = vec![0, 1];
        let is_cell = vec![true, true];
        let labels = Cellpose::label_sinks(&final_positions, &is_cell, 2, 1);
        assert_ne!(labels[0], 0);
        assert_eq!(labels[0], labels[1]);
    }

    #[test]
    fn label_sinks_gives_different_labels_to_far_apart_sinks() {
        let final_positions = vec![0, 0, 2];
        let is_cell = vec![true, true, true];
        let labels = Cellpose::label_sinks(&final_positions, &is_cell, 3, 1);
        assert_eq!(labels[0], labels[1]);
        assert_ne!(labels[0], labels[2]);
        assert_ne!(labels[2], 0);
    }

    #[test]
    fn label_sinks_never_labels_a_non_cell_pixel() {
        let final_positions = vec![1, 1, 1];
        let is_cell = vec![false, true, true];
        let labels = Cellpose::label_sinks(&final_positions, &is_cell, 3, 1);
        assert_eq!(
            labels[0], 0,
            "non-cell pixels must stay unlabeled regardless of final_positions"
        );
        assert_ne!(labels[1], 0);
        assert_eq!(labels[1], labels[2]);
    }

    #[test]
    fn label_sinks_returns_all_zero_when_no_pixel_is_a_cell() {
        let final_positions = vec![0, 1, 2, 3];
        let is_cell = vec![false, false, false, false];
        let labels = Cellpose::label_sinks(&final_positions, &is_cell, 2, 2);
        assert_eq!(labels, vec![0, 0, 0, 0]);
    }

    // ---- max_resize ----

    #[test]
    fn resize_factor_shrinks_the_longest_full_image_side_only() {
        let size = |width, height| kornia_image::ImageSize { width, height };
        let cmd = |max_resize| Cellpose {
            max_resize,
            ..cellpose(0)
        };
        assert_eq!(cmd(0).resize_factor(size(2048, 1024)), 1.0, "0 disables it");
        assert_eq!(
            cmd(1000).resize_factor(size(800, 600)),
            1.0,
            "never enlarges"
        );
        assert_eq!(cmd(1000).resize_factor(size(2000, 4000)), 0.25);
    }

    #[cfg(feature = "libtorch-in-process")]
    #[test]
    fn execute_with_max_resize_runs_at_low_resolution_and_scales_masks_back() {
        // A 300x300 image shrunk to 150x150 (max_resize 150): the masks come
        // back at full size with the left/right split preserved.
        let (_dir, model_path) = flow_free_cellpose_model(2);
        let cmd = Cellpose {
            model_path,
            max_resize: 150,
            ..cellpose(0)
        };
        let (width, height) = (300usize, 300usize);
        let mut values = vec![0f32; width * height];
        for y in 0..height {
            for x in 150..width {
                values[y * width + x] = 1.0;
            }
        }
        let mut ctx = gray_ctx(width, height, values);
        let mut cache = GlobalPipelineCache::default();
        cmd.execute(&mut ctx, &mut cache).unwrap();

        let seg = ctx.get_segmentation_map().unwrap().as_slice();
        assert_eq!(seg.len(), width * height);
        for y in [0, 149, 299] {
            assert_eq!(seg[y * width + 10], 0, "background stays background");
            assert_eq!(seg[y * width + 290], 7, "the cell side is segmented");
        }
    }

    // ---- flow_threshold ----

    /// `cellpose.dynamics.masks_to_flows_gpu` (Cellpose 4.2.1) on this label
    /// map: two touching squares, a C shape whose mean lies outside it, a
    /// single pixel next to another object, and a 3x3 square.
    mod reference {
        pub const LABELS: [u32; 108] = [
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 0, 0, 0, 0, 0, 1, 1, 1, 1,
            2, 2, 2, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 0, 0,
            0, 0, 3, 3, 3, 3, 3, 0, 0, 0, 0, 5, 5, 5, 3, 0, 0, 0, 0, 0, 0, 0, 4, 5, 5, 5, 3, 0, 0,
            0, 0, 0, 0, 0, 0, 5, 5, 5, 3, 3, 3, 3, 3, 0, 0, 0, 0, 0, 0, 0,
        ];
        pub const EXPECTED_Y: [f64; 108] = [
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.707107, 0.974547,
            0.996067, 0.977018, 0.993298, 1.0, 0.895948, 0.0, 0.0, 0.0, 0.0, 0.0, 0.224183,
            0.707107, 0.978536, 0.891251, 0.962258, 1.0, 0.521388, 0.0, 0.0, 0.0, 0.0, 0.0,
            -0.088608, -0.206078, -0.707107, -0.300207, -0.417556, -1.0, -0.087281, 0.0, 0.0, 0.0,
            0.0, 0.0, -0.478715, -0.869184, -0.995737, -0.967643, -0.986389, -1.0, -0.716851, 0.0,
            0.0, 0.0, 0.0, 0.989622, -0.999995, -0.999999, -1.0, -1.0, 0.0, 0.0, 0.0, 0.0,
            0.707107, 1.0, 0.707107, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.707107, -1.0, -0.707107, -0.143741,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ];
        pub const EXPECTED_X: [f64; 108] = [
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.707107, 0.224183,
            -0.088608, -0.213157, 0.115583, 0.0, -0.444159, 0.0, 0.0, 0.0, 0.0, 0.0, 0.974547,
            0.707107, -0.206078, -0.45351, 0.272139, 0.0, -0.85332, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.996067, 0.978536, -0.707107, -0.953874, 0.908651, 0.0, -0.996184, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.87797, 0.494488, -0.092234, -0.252321, 0.164426, 0.0, -0.697226, 0.0, 0.0, 0.0,
            0.0, 0.143692, -0.003081, -0.00127, -0.000144, -2.4e-05, 0.0, 0.0, 0.0, 0.0, 0.707107,
            0.0, -0.707107, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, -1.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.707107, 0.0, -0.707107, 0.989615, 1.0, -1.0, -1.0,
            -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ];
    }

    #[test]
    fn masks_to_flows_matches_cellpose() {
        let (fy, fx) = Cellpose::masks_to_flows(&reference::LABELS, 12, 9);
        for i in 0..reference::LABELS.len() {
            assert!(
                (fy[i] - reference::EXPECTED_Y[i]).abs() < 1e-4
                    && (fx[i] - reference::EXPECTED_X[i]).abs() < 1e-4,
                "pixel ({}, {}): got ({}, {}), Cellpose has ({}, {})",
                i % 12,
                i / 12,
                fy[i],
                fx[i],
                reference::EXPECTED_Y[i],
                reference::EXPECTED_X[i]
            );
        }
    }

    #[test]
    fn remove_bad_flow_masks_keeps_objects_whose_flows_match_and_drops_the_rest() {
        let (w, h) = (12, 9);
        let (fy, fx) = Cellpose::masks_to_flows(&reference::LABELS, w, h);
        // Predicted flows (x FLOW_SCALE) that match every object exactly ...
        let mut pred_y: Vec<f32> = fy.iter().map(|v| (v * 5.0) as f32).collect();
        let mut pred_x: Vec<f32> = fx.iter().map(|v| (v * 5.0) as f32).collect();
        // ... except object 5, whose flows point the wrong way.
        for i in 0..w * h {
            if reference::LABELS[i] == 5 {
                pred_y[i] = -pred_y[i];
                pred_x[i] = -pred_x[i];
            }
        }
        let mut labels = reference::LABELS.to_vec();
        Cellpose::remove_bad_flow_masks(&mut labels, &pred_y, &pred_x, w, h, 0.4);
        for i in 0..w * h {
            let expected = if reference::LABELS[i] == 5 {
                0
            } else {
                reference::LABELS[i]
            };
            assert_eq!(labels[i], expected, "pixel {i}");
        }
    }

    #[test]
    fn masks_to_flows_of_an_empty_label_map_is_zero() {
        let (fy, fx) = Cellpose::masks_to_flows(&[0; 6], 3, 2);
        assert!(fy.iter().chain(&fx).all(|&v| v == 0.0));
    }

    // ---- cellpose_postprocessing ----

    /// Asserts both label maps describe the same objects - the label
    /// numbers themselves may differ.
    fn assert_same_objects(got: &[u32], expected: &[u32]) {
        let mismatches = count_object_mismatches(got, expected);
        assert_eq!(mismatches, 0, "{mismatches} pixels differ from Cellpose");
    }

    fn count_object_mismatches(got: &[u32], expected: &[u32]) -> usize {
        let mut forward = std::collections::HashMap::new();
        let mut backward = std::collections::HashMap::new();
        let mut mismatches = 0;
        for (&g, &e) in got.iter().zip(expected) {
            let ok = (g == 0) == (e == 0)
                && (g == 0
                    || (*forward.entry(g).or_insert(e) == e
                        && *backward.entry(e).or_insert(g) == g));
            if !ok {
                mismatches += 1;
            }
        }
        mismatches
    }

    #[test]
    fn cellpose_postprocessing_reproduces_cellpose_masks() {
        // Flows/probabilities and the masks Cellpose 4.2.1 builds from them -
        // see `source` in the fixture.
        let fixture: serde_json::Value = serde_json::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/cellpose_postprocessing.json"
        )))
        .unwrap();
        let floats = |key: &str| -> Vec<f32> {
            fixture[key]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_f64().unwrap() as f32)
                .collect()
        };
        let width = fixture["width"].as_u64().unwrap() as usize;
        let height = fixture["height"].as_u64().unwrap() as usize;
        let cell_prob: Vec<f32> = floats("cellprob_logit")
            .iter()
            .map(|l| 1.0 / (1.0 + (-l).exp()))
            .collect();
        let expected: Vec<u32> = fixture["labels"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as u32)
            .collect();

        // What the template's Cellpose -> Fill Object Holes steps do:
        // `execute`'s mask building and size filter, then the hole filling.
        let segment = |cmd: &Cellpose| -> Vec<u32> {
            let labels = cmd.masks_from_flows(
                &floats("flow_y"),
                &floats("flow_x"),
                &cell_prob,
                width,
                height,
            );
            let mut segmentation = vec![0u32; width * height];
            let mut instances = vec![0u32; width * height];
            cmd.write_instances(&labels, &mut segmentation, &mut instances);
            crate::algos::morphology::fill_object_holes::fill_instance_holes(
                &mut instances,
                width,
                height,
            );
            instances
        };

        let cmd = Cellpose {
            flow_iterations: 200,
            min_object_size: 15,
            cellpose_postprocessing: true,
            ..cellpose(15)
        };
        assert_same_objects(&segment(&cmd), &expected);

        // The fixture tells the two modes apart: without the post-processing
        // the result differs from Cellpose.
        let cmd = Cellpose {
            cellpose_postprocessing: false,
            ..cmd
        };
        let mismatches = count_object_mismatches(&segment(&cmd), &expected);
        println!("default mode: {mismatches} pixels differ from Cellpose");
        assert!(mismatches > 0);
    }

    #[test]
    fn masks_from_seeds_drops_pixels_that_reach_no_crowded_spot() {
        // 12 pixels end at (5,5): a seed. 3 pixels end alone far away: no seed.
        let (w, h) = (20, 1);
        let mut final_positions: Vec<(usize, usize)> = (0..w).map(|x| (0, x)).collect();
        let is_cell = vec![true; w];
        for p in final_positions.iter_mut().take(12) {
            *p = (0, 5);
        }
        let labels = Cellpose::masks_from_seeds(&final_positions, &is_cell, w, h);
        assert!(labels[..12].iter().all(|&l| l == 1));
        assert!(labels[12..].iter().all(|&l| l == 0), "{labels:?}");
    }

    #[test]
    fn masks_from_seeds_without_any_crowded_spot_is_empty() {
        let final_positions: Vec<(usize, usize)> = (0..4).map(|x| (0, x)).collect();
        let labels = Cellpose::masks_from_seeds(&final_positions, &[true; 4], 4, 1);
        assert_eq!(labels, vec![0; 4]);
    }

    #[cfg(feature = "libtorch-in-process")]
    #[test]
    fn execute_with_replicate_gray_channel_feeds_the_image_into_every_channel() {
        // The model's cell logit comes from channel 1 only: zero-filled, no
        // cell is found; replicated, the bright pixel is.
        let (_dir, model_path) = trace_and_save_model(2, TILE_SIZE, TILE_SIZE, |x| {
            let second = x.narrow(1, 1, 1);
            let zeros = second.zeros_like();
            Tensor::cat(&[zeros.shallow_clone(), zeros, second * 20.0 - 10.0], 1)
        });
        let run = |replicate_gray_channel| {
            let cmd = Cellpose {
                model_path: model_path.clone(),
                replicate_gray_channel,
                ..cellpose(0)
            };
            let mut ctx = gray_ctx(2, 1, vec![0.0, 1.0]);
            cmd.execute(&mut ctx, &mut GlobalPipelineCache::default())
                .unwrap();
            ctx.get_segmentation_map().unwrap().as_slice().to_vec()
        };
        assert_eq!(run(false), vec![0, 0]);
        assert_eq!(run(true), vec![0, 7]);
    }

    // ---- write_instances ----

    #[test]
    fn write_instances_drops_objects_smaller_than_min_object_size() {
        let algo = cellpose(2);
        // label 1: 3 pixels (kept), label 2: 1 pixel (dropped).
        let labels = vec![1, 1, 1, 2, 0, 0];
        let mut seg = vec![0u32; 6];
        let mut inst = vec![0u32; 6];
        algo.write_instances(&labels, &mut seg, &mut inst);

        assert_eq!(inst, vec![1, 1, 1, 0, 0, 0]);
        assert_eq!(seg, vec![7, 7, 7, 0, 0, 0]);
    }

    #[test]
    fn write_instances_renumbers_surviving_labels_to_contiguous_ids() {
        let algo = cellpose(2);
        // label 1 (3px) and label 3 (3px) survive; label 2 (1px) is dropped,
        // so the surviving instance IDs must be 1 and 2, not 1 and 3.
        let labels = vec![1, 1, 1, 2, 3, 3, 3];
        let mut seg = vec![0u32; 7];
        let mut inst = vec![0u32; 7];
        algo.write_instances(&labels, &mut seg, &mut inst);

        assert_eq!(inst, vec![1, 1, 1, 0, 2, 2, 2]);
    }

    #[test]
    fn write_instances_resets_stale_buffers_for_an_all_background_label_map() {
        // Regression: `seg_slice`/`inst_slice` may be reused buffers carrying
        // stale nonzero labels from an earlier segmentation pass in the same
        // pipeline. Per this command's documented contract ("all other
        // pixels are assigned BACKGROUND"), finding zero cells this run must
        // reset them to 0, not silently preserve whatever was there before.
        let algo = cellpose(0);
        let labels = vec![0, 0, 0, 0];
        let mut seg = vec![9u32; 4]; // stale sentinel from a previous pass
        let mut inst = vec![9u32; 4];
        algo.write_instances(&labels, &mut seg, &mut inst);

        assert_eq!(
            seg,
            vec![0, 0, 0, 0],
            "stale segmentation labels must be reset to background"
        );
        assert_eq!(
            inst,
            vec![0, 0, 0, 0],
            "stale instance ids must be reset to background"
        );
    }

    #[test]
    fn write_instances_resets_stale_buffers_for_pixels_outside_any_surviving_object() {
        // Same stale-buffer scenario, but with a mix of surviving and
        // dropped/background pixels: every pixel not covered by a surviving
        // object must be explicitly reset, not just the ones that happen to
        // be background in `labels`.
        let algo = cellpose(2);
        // label 1: 3px (kept), label 2: 1px (dropped for being < min_size),
        // remaining 2 pixels are background (label 0).
        let labels = vec![1, 1, 1, 2, 0, 0];
        let mut seg = vec![9u32; 6];
        let mut inst = vec![9u32; 6];
        algo.write_instances(&labels, &mut seg, &mut inst);

        assert_eq!(inst, vec![1, 1, 1, 0, 0, 0]);
        assert_eq!(seg, vec![7, 7, 7, 0, 0, 0]);
    }

    #[test]
    fn write_instances_min_object_size_zero_still_keeps_a_single_pixel_object() {
        let algo = cellpose(0);
        let labels = vec![1, 0, 0];
        let mut seg = vec![0u32; 3];
        let mut inst = vec![0u32; 3];
        algo.write_instances(&labels, &mut seg, &mut inst);

        assert_eq!(
            inst[0], 1,
            "min_object_size = 0 must disable the size filter, not drop everything"
        );
        assert_eq!(seg[0], 7);
    }
}
