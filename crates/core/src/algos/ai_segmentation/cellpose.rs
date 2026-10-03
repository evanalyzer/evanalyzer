//! # cellpose
//!
//! **Author:** Joachim Danmayr
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **AGPL-3.0**.

use std::path::PathBuf;

use evanalyzer_cfg::core_types::{CitationMetadata, InternalErrors, SegmentationClass};
use macros::CommandsMeta;
use tch::{CModule, Device, IValue, Kind, Tensor};

use crate::{
    algos::{ExecutionScope, ImageAlgorithm, ai_segmentation::model_cache::load_cached_model},
    pipeline::{pipeline_cache::GlobalPipelineCache, pipeline_context::PipelineContext},
};

/// Instance segmentation using a Cellpose-SAM model exported as TorchScript
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
    next = "measure",
    display_name = "AI Cellpose Segmentation"
)]
pub struct Cellpose {
    /// Path to a TorchScript-exported Cellpose model (`torch.jit.script`/`torch.jit.trace`).
    #[cmdsmeta(file_extensions = "pt,pth")]
    pub model_path: PathBuf,

    /// The class assigned to pixels of every detected object. All other
    /// pixels are assigned `SegmentationClass::BACKGROUND`.
    #[cmdsmeta(default = SegmentationClass(1))]
    pub object_class_id: SegmentationClass,

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
        let device = Device::cuda_if_available();
        let model = load_cached_model(&self.model_path, || {
            CModule::load_on_device(&self.model_path, device)
        })
        .map_err(|e| {
            InternalErrors::Generic(format!(
                "Failed to load Cellpose model from {}: {e}. The model must be a \
                 TorchScript export (torch.jit.script/trace), not a raw weights file. \
                 A bioimage.io `pytorch_state_dict` (.pth) holds only weights and no \
                 graph — load it into the Cellpose architecture in Python and re-save \
                 it with torch.jit before using it here.",
                self.model_path.display()
            ))
        })?;

        let scale = self.resize_factor(ctx.full_image_size());
        let (input_image, segmentation_map, instance_map) =
            ctx.get_f32_gray_segmentation_and_instances_mut()?;
        let size = input_image.size();
        let (width, height) = (size.width, size.height);
        // The size the model and the dynamics run at.
        let net_width = ((width as f64 * scale).round() as usize).max(1);
        let net_height = ((height as f64 * scale).round() as usize).max(1);
        let resized = (net_width, net_height) != (width, height);

        let image = Tensor::from_slice(input_image.as_slice())
            .to_device(device)
            .to_kind(Kind::Float)
            .reshape([1, 1, height as i64, width as i64]);

        // The image is the first channel; standard Cellpose models expect a
        // second (nucleus) channel, and custom models may want more. Zero-fill
        // (or replicate the image into) any extra channels so the tensor
        // matches the model's input width.
        let in_channels = self.input_channels.max(1) as i64;
        let input = if in_channels <= 1 {
            image
        } else if self.replicate_gray_channel {
            image.repeat([1, in_channels, 1, 1])
        } else {
            let extra = Tensor::zeros(
                [1, in_channels - 1, height as i64, width as i64],
                (Kind::Float, device),
            );
            Tensor::cat(&[image, extra], 1)
        };
        let input = if resized {
            input.upsample_bilinear2d([net_height as i64, net_width as i64], false, None, None)
        } else {
            input
        };

        // Cellpose-SAM can only run on exactly 256x256 tiles (see the struct
        // doc comment) - `run_model_tiled` hides that behind the same
        // `[1, C, H, W]` contract `run_model` used to expose directly.
        let output = Self::run_model_tiled(&model, &input, device)?;

        // `run_model_tiled` always returns a `[1, C, height, width]` tensor,
        // so the channel dimension is fixed at index 1.
        const CHANNEL_DIM: i64 = 1;
        let (w, h) = (net_width, net_height);
        let flow_y = Self::channel_to_vec(&output.narrow(CHANNEL_DIM, 0, 1), w, h)?;
        let flow_x = Self::channel_to_vec(&output.narrow(CHANNEL_DIM, 1, 1), w, h)?;
        let cell_prob = Self::channel_to_vec(&output.narrow(CHANNEL_DIM, 2, 1).sigmoid(), w, h)?;

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

    /// Runs the model and returns the flow/probability tensor, supporting both a
    /// bare tensor output and exports that wrap it in a tuple/list (e.g.
    /// `(flows, style)`).
    fn run_model(model: &CModule, input: Tensor) -> Result<Tensor, InternalErrors> {
        let output = model
            .forward_is(&[IValue::Tensor(input)])
            .map_err(|e| InternalErrors::Generic(format!("Cellpose inference failed: {e}")))?;

        let tensor = match output {
            IValue::Tensor(t) => Some(t),
            IValue::Tuple(items) | IValue::GenericList(items) => items
                .into_iter()
                .filter_map(|v| match v {
                    IValue::Tensor(t) => Some(t),
                    _ => None,
                })
                .find(|t| {
                    let s = t.size();
                    s.len() >= 3 && s[s.len() - 3] >= 3
                }),
            other => {
                return Err(InternalErrors::Generic(format!(
                    "Cellpose model returned an unsupported output type: {other:?}"
                )));
            }
        };

        tensor.ok_or_else(|| {
            InternalErrors::Generic(
                "Cellpose model returned no tensor with at least 3 channels".into(),
            )
        })
    }

    /// Cellpose-SAM's ViT encoder bakes its positional embeddings for a fixed
    /// token grid at export time (see the struct doc comment) - the exported
    /// graph only accepts exactly `TILE_SIZE x TILE_SIZE` input.
    const TILE_SIZE: i64 = 256;

    /// Fraction of a tile that overlaps its neighbor, matching Cellpose's own
    /// default (`tile_overlap=0.1` in `models.CellposeModel.eval`).
    const TILE_OVERLAP: f32 = 0.1;

    /// Runs `model` over `input` (`[1, C, height, width]`, any `height`/`width`)
    /// by padding it up to at least `TILE_SIZE` per side, splitting it into
    /// overlapping `TILE_SIZE x TILE_SIZE` tiles, running each tile through
    /// `run_model`, and blending the results back into a single
    /// `[1, C, height, width]` tensor with a feathered (sigmoid taper) weight
    /// per tile - the same approach Cellpose's own `transforms.average_tiles`
    /// uses, so a segmentation spanning a tile boundary doesn't show a seam.
    ///
    /// A single tile that covers the whole (padded) image is the common case
    /// for images no bigger than `TILE_SIZE`; the taper weight cancels out
    /// exactly there (it's the only tile contributing to every pixel), so
    /// small images are unaffected by the blending.
    fn run_model_tiled(
        model: &CModule,
        input: &Tensor,
        device: Device,
    ) -> Result<Tensor, InternalErrors> {
        // Without this, every tile's forward pass keeps its autograd graph
        // (all of the ViT encoder's intermediate activations) alive, and the
        // in-place `acc_region += ...` accumulation below chains each tile's
        // graph onto the last - so a large image's memory use grows with
        // *every* tile instead of being bounded by one tile's peak, which
        // reliably exhausts GPU memory on anything but a tiny image.
        tch::no_grad(|| Self::run_model_tiled_inner(model, input, device))
    }

    fn run_model_tiled_inner(
        model: &CModule,
        input: &Tensor,
        device: Device,
    ) -> Result<Tensor, InternalErrors> {
        let sizes = input.size();
        let (orig_h, orig_w) = (sizes[2], sizes[3]);

        let pad_h = (Self::TILE_SIZE - orig_h).max(0);
        let pad_w = (Self::TILE_SIZE - orig_w).max(0);
        // `Tensor::pad`'s list is ordered from the last dimension inward:
        // [left, right, top, bottom] pads W then H. Only the bottom/right
        // edges are padded - unlike Cellpose's own centered padding, this is
        // an implementation simplification, not a correctness requirement:
        // the padding is zero context for the network either way, and the
        // exact split of "extra" border pixels doesn't affect the result
        // (the output is cropped back to `orig_h`/`orig_w` afterwards).
        let padded = input.pad([0, pad_w, 0, pad_h], "constant", 0.0);
        let padded_h = orig_h + pad_h;
        let padded_w = orig_w + pad_w;

        let ys = Self::tile_starts(padded_h);
        let xs = Self::tile_starts(padded_w);
        let mask = Self::taper_mask(device);

        let mut acc: Option<Tensor> = None;
        let norm = Tensor::zeros([1, 1, padded_h, padded_w], (Kind::Float, device));

        for &y in &ys {
            for &x in &xs {
                let tile = padded
                    .narrow(2, y, Self::TILE_SIZE)
                    .narrow(3, x, Self::TILE_SIZE);
                let tile_out = Self::run_model(model, tile)?;

                let tsizes = tile_out.size();
                if tsizes.len() < 4 {
                    return Err(InternalErrors::Generic(
                        "Cellpose model output has too few dimensions; expected \
                         `[1, C, 256, 256]`"
                            .into(),
                    ));
                }
                let tile_channels = tsizes[tsizes.len() - 3];
                if tile_channels < 3 {
                    return Err(InternalErrors::Generic(
                        "Cellpose model output has fewer than 3 channels; expected \
                         `[dY, dX, cellprob]`"
                            .into(),
                    ));
                }
                if tsizes[tsizes.len() - 2] != Self::TILE_SIZE
                    || tsizes[tsizes.len() - 1] != Self::TILE_SIZE
                {
                    return Err(InternalErrors::Generic(format!(
                        "Cellpose-SAM model output tile is {}x{}, expected exactly \
                         256x256 - the exported model must be traced at a fixed \
                         256x256 input (see docs/convert_cellpose.py)",
                        tsizes[tsizes.len() - 1],
                        tsizes[tsizes.len() - 2],
                    )));
                }

                let tile_out = tile_out.to_kind(Kind::Float).to_device(device);
                let acc = acc.get_or_insert_with(|| {
                    Tensor::zeros(
                        [1, tile_channels, padded_h, padded_w],
                        (Kind::Float, device),
                    )
                });

                let mut acc_region =
                    acc.narrow(2, y, Self::TILE_SIZE)
                        .narrow(3, x, Self::TILE_SIZE);
                acc_region += &tile_out * &mask;
                let mut norm_region =
                    norm.narrow(2, y, Self::TILE_SIZE)
                        .narrow(3, x, Self::TILE_SIZE);
                norm_region += &mask;
            }
        }

        let acc =
            acc.ok_or_else(|| InternalErrors::Generic("Cellpose produced no output tiles".into()))?;
        let stitched = acc / &norm;
        Ok(stitched.narrow(2, 0, orig_h).narrow(3, 0, orig_w))
    }

    /// Start offsets (top or left) of the `TILE_SIZE`-wide tiles covering
    /// `padded_len` pixels with `TILE_OVERLAP` fractional overlap, matching
    /// Cellpose's own `transforms.make_tiles`. A single tile starting at `0`
    /// covers the whole span whenever `padded_len <= TILE_SIZE`.
    fn tile_starts(padded_len: i64) -> Vec<i64> {
        if padded_len <= Self::TILE_SIZE {
            return vec![0];
        }
        let overlap = Self::TILE_OVERLAP.clamp(0.05, 0.5);
        let n =
            (((1.0 + 2.0 * overlap) * padded_len as f32) / Self::TILE_SIZE as f32).ceil() as i64;
        if n <= 1 {
            return vec![0];
        }
        let span = (padded_len - Self::TILE_SIZE) as f32;
        (0..n)
            .map(|i| (span * i as f32 / (n - 1) as f32) as i64)
            .collect()
    }

    /// The `[1, 1, TILE_SIZE, TILE_SIZE]` feathered blend weight Cellpose's
    /// `transforms._taper_mask` uses: a separable sigmoid taper that's ~1 near
    /// the tile's center and decays (without ever reaching exactly `0`) toward
    /// its edges, so overlapping tiles blend smoothly instead of showing a
    /// seam at the boundary.
    fn taper_mask(device: Device) -> Tensor {
        const SIG: f32 = 7.5;
        let size = Self::TILE_SIZE as usize;
        let center = (Self::TILE_SIZE as f32 - 1.0) / 2.0;
        let mask1d: Vec<f32> = (0..size)
            .map(|i| {
                let xm = (i as f32 - center).abs();
                1.0 / (1.0 + ((xm - (Self::TILE_SIZE as f32 / 2.0 - 20.0)) / SIG).exp())
            })
            .collect();
        let mut mask2d = vec![0f32; size * size];
        for y in 0..size {
            for x in 0..size {
                mask2d[y * size + x] = mask1d[y] * mask1d[x];
            }
        }
        Tensor::from_slice(&mask2d).to_device(device).reshape([
            1,
            1,
            Self::TILE_SIZE,
            Self::TILE_SIZE,
        ])
    }

    /// Moves a single-channel `[1, 1, H, W]` tensor to the CPU and flattens it
    /// into a `width * height` vector.
    fn channel_to_vec(
        tensor: &Tensor,
        width: usize,
        height: usize,
    ) -> Result<Vec<f32>, InternalErrors> {
        tensor
            .f_to_device(Device::Cpu)
            .and_then(|out| out.f_reshape([(width * height) as i64]))
            .map_err(|e| InternalErrors::Generic(format!("Cellpose inference failed: {e}")))
            .and_then(|out| {
                Vec::try_from(&out)
                    .map_err(|e| InternalErrors::Generic(format!("Cellpose inference failed: {e}")))
            })
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

        let foreground_class = self.object_class_id.as_u32();
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
    use crate::algos::ai_segmentation::test_support::trace_and_save_model;
    use kornia_image::{Image, ImageSize};

    fn cellpose(min_object_size: i32) -> Cellpose {
        Cellpose {
            model_path: PathBuf::new(),
            object_class_id: SegmentationClass(7),
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
    /// Always traced at `Cellpose::TILE_SIZE` (256x256): `run_model_tiled`
    /// invokes the model at that fixed size regardless of the logical image
    /// size passed to `execute()` (see `gray_ctx`), matching how a real
    /// Cellpose-SAM export behaves.
    fn flow_free_cellpose_model(in_channels: i64) -> (tempfile::TempDir, std::path::PathBuf) {
        trace_and_save_model(in_channels, Cellpose::TILE_SIZE, Cellpose::TILE_SIZE, |x| {
            let image_channel = x.narrow(1, 0, 1);
            let flow_y = image_channel.zeros_like();
            let flow_x = image_channel.zeros_like();
            let cell_logit = image_channel * 20.0 - 10.0;
            Tensor::cat(&[flow_y, flow_x, cell_logit], 1)
        })
    }

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

    #[test]
    fn execute_errors_when_the_model_output_has_too_few_dimensions() {
        let (_dir, model_path) =
            trace_and_save_model(2, Cellpose::TILE_SIZE, Cellpose::TILE_SIZE, |x| {
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

    #[test]
    fn execute_errors_when_the_model_output_has_fewer_than_three_channels() {
        let (_dir, model_path) =
            trace_and_save_model(2, Cellpose::TILE_SIZE, Cellpose::TILE_SIZE, |x| {
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

    #[test]
    fn execute_errors_when_the_model_output_tile_size_is_wrong() {
        // A real Cellpose-SAM export always returns a 256x256 tile (its ViT
        // encoder can't run at any other size); a model that doesn't must be
        // rejected rather than silently misaligned during stitching.
        let (_dir, model_path) =
            trace_and_save_model(2, Cellpose::TILE_SIZE, Cellpose::TILE_SIZE, |x| {
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

    #[test]
    fn tile_starts_returns_a_single_zero_start_for_a_tile_sized_or_smaller_canvas() {
        assert_eq!(Cellpose::tile_starts(1), vec![0]);
        assert_eq!(Cellpose::tile_starts(Cellpose::TILE_SIZE), vec![0]);
    }

    #[test]
    fn tile_starts_covers_a_larger_canvas_with_overlap() {
        // Ly=500 > TILE_SIZE: ny = ceil(1.2 * 500 / 256) = 3, tile starts
        // evenly spaced across [0, 500-256] = [0, 244], matching Cellpose's
        // own `transforms.make_tiles`.
        let starts = Cellpose::tile_starts(500);
        assert_eq!(starts, vec![0, 122, 244]);
        for &s in &starts {
            assert!(s >= 0 && s + Cellpose::TILE_SIZE <= 500);
        }
    }

    #[test]
    fn taper_mask_peaks_at_the_center_and_decays_but_never_reaches_zero_at_the_edges() {
        let mask = Cellpose::taper_mask(Device::Cpu);
        assert_eq!(
            mask.size(),
            vec![1, 1, Cellpose::TILE_SIZE, Cellpose::TILE_SIZE]
        );

        let at = |y: i64, x: i64| -> f64 {
            f64::try_from(mask.narrow(2, y, 1).narrow(3, x, 1).reshape([1])).unwrap()
        };
        let center = at(127, 127);
        let corner = at(0, 0);
        assert!(center > 0.99, "center weight should be ~1.0, got {center}");
        assert!(corner > 0.0, "taper mask must never reach exactly 0");
        assert!(
            corner < center,
            "corner weight ({corner}) should be far smaller than the center ({center})"
        );
    }

    #[test]
    fn run_model_tiled_reconstructs_exact_per_pixel_values_across_a_multi_tile_image() {
        // A pixel-wise (position-independent) toy model: whatever the tiling
        // grid or blend weights do, the "true" value at a given pixel is the
        // same in every tile that covers it, so a correct blend must
        // reconstruct it exactly - this isolates the padding/tiling/stitching
        // logic itself from the flow dynamics already covered above.
        let (_dir, model_path) = flow_free_cellpose_model(2);
        let model = tch::CModule::load_on_device(&model_path, Device::Cpu).unwrap();

        // 300x300 forces a 2x2 tile grid (Cellpose::tile_starts(300) has two
        // starts per axis) with a large overlap region; the left half is
        // background (0.0) and the right half is foreground (1.0), so the
        // split sits inside the overlap and exercises the blend directly.
        let (width, height): (i64, i64) = (300, 300);
        let mut values = vec![0f32; (width * height) as usize];
        for y in 0..height {
            for x in 150..width {
                values[(y * width + x) as usize] = 1.0;
            }
        }
        let image = Tensor::from_slice(&values)
            .to_kind(Kind::Float)
            .reshape([1, 1, height, width]);
        let input = Tensor::cat(&[image.shallow_clone(), image.zeros_like()], 1);

        let output = Cellpose::run_model_tiled(&model, &input, Device::Cpu).unwrap();
        assert_eq!(output.size(), vec![1, 3, height, width]);

        let cell_prob: Vec<f32> =
            Vec::try_from(&output.narrow(1, 2, 1).sigmoid().reshape([height * width])).unwrap();
        for y in 0..height as usize {
            for x in 0..width as usize {
                let p = cell_prob[y * width as usize + x];
                if x < 150 {
                    assert!(p < 0.01, "expected background at ({x},{y}), got {p}");
                } else {
                    assert!(p > 0.99, "expected foreground at ({x},{y}), got {p}");
                }
            }
        }
    }

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

    #[test]
    fn execute_with_replicate_gray_channel_feeds_the_image_into_every_channel() {
        // The model's cell logit comes from channel 1 only: zero-filled, no
        // cell is found; replicated, the bright pixel is.
        let (_dir, model_path) =
            trace_and_save_model(2, Cellpose::TILE_SIZE, Cellpose::TILE_SIZE, |x| {
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
