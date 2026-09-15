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
/// (see `docs/convert_cellpose.py`).
///
/// Cellpose-SAM's SAM-derived ViT encoder bakes its positional embeddings for
/// a **fixed 256x256 token grid** at export time, so the exported graph can
/// only be run on exactly 256x256 tiles — Cellpose's own Python
/// implementation enforces the same limit (`bsize != 256 is not supported
/// for cpsam`). This command hides that constraint: the (normalized) image is
/// padded and split into overlapping 256x256 tiles internally, each tile is
/// run through the model, and the outputs are blended back together with the
/// same feathered (sigmoid taper) weighting Cellpose's own tiling uses
/// (`transforms.average_tiles`), so a segmentation spanning a tile boundary
/// doesn't show a seam.
///
/// Each tile is a `[1, input_channels, 256, 256]` float tensor: the
/// (normalized) grayscale image goes in channel 0 and any remaining channels
/// are zero-filled. Cellpose-SAM's patch-embedding convolution only has
/// weights for up to 3 input channels, so `input_channels` must be `1`-`3`
/// (`2`, cytoplasm + optional nucleus, is standard). The model must return a
/// `[1, C, 256, 256]` tensor per tile with `C >= 3` channels: the vertical
/// flow `dY` (channel 0), the horizontal flow `dX` (channel 1) and the
/// cell-probability logits (channel 2), which is Cellpose's spatial-gradient
/// representation. Exports that wrap the output in a tuple (e.g.
/// `(flows, style)`) are also supported — the first tensor with at least
/// three channels is used.
///
/// Instances are recovered with Cellpose's *dynamics*: every pixel whose
/// cell probability reaches `probability_threshold` is advected for
/// `flow_iterations` Euler steps along the (down-scaled) flow field until it
/// converges to the sink at its cell's center. Pixels whose trajectories end in
/// the same sink basin — found by connected components over the final-position
/// density map — form one instance. Instances smaller than `min_object_size`
/// pixels are discarded. Runs on GPU automatically if CUDA is available in the
/// linked libtorch build, otherwise falls back to CPU.
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
    #[cmdsmeta(default = 2, min = 1, max = 3, step = 1)]
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
    #[cmdsmeta(default = 200, min = 1, max = 1000, step = 1)]
    pub flow_iterations: i32,

    /// Minimum object size, in pixels. After the dynamics, any instance smaller
    /// than this is removed (its pixels become background). `0` disables the filter.
    #[cmdsmeta(default = 15, min = 0, max = 100000, step = 1)]
    pub min_object_size: i32,
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

        let (input_image, segmentation_map, instance_map) =
            ctx.get_f32_gray_segmentation_and_instances_mut()?;
        let size = input_image.size();
        let (width, height) = (size.width, size.height);

        let image = Tensor::from_slice(input_image.as_slice())
            .to_device(device)
            .to_kind(Kind::Float)
            .reshape([1, 1, height as i64, width as i64]);

        // The image is the first channel; standard Cellpose models expect a
        // second (nucleus) channel, and custom models may want more. Zero-fill
        // any extra channels so the tensor matches the model's input width.
        let in_channels = self.input_channels.max(1) as i64;
        let input = if in_channels <= 1 {
            image
        } else {
            let extra = Tensor::zeros(
                [1, in_channels - 1, height as i64, width as i64],
                (Kind::Float, device),
            );
            Tensor::cat(&[image, extra], 1)
        };

        // Cellpose-SAM can only run on exactly 256x256 tiles (see the struct
        // doc comment) - `run_model_tiled` hides that behind the same
        // `[1, C, H, W]` contract `run_model` used to expose directly.
        let output = Self::run_model_tiled(&model, &input, device)?;

        // `run_model_tiled` always returns a `[1, C, height, width]` tensor,
        // so the channel dimension is fixed at index 1.
        const CHANNEL_DIM: i64 = 1;
        let flow_y = Self::channel_to_vec(&output.narrow(CHANNEL_DIM, 0, 1), width, height)?;
        let flow_x = Self::channel_to_vec(&output.narrow(CHANNEL_DIM, 1, 1), width, height)?;
        let cell_prob =
            Self::channel_to_vec(&output.narrow(CHANNEL_DIM, 2, 1).sigmoid(), width, height)?;

        // Pixels above the cell-probability threshold take part in the dynamics.
        let is_cell: Vec<bool> = cell_prob
            .iter()
            .map(|&p| p >= self.probability_threshold)
            .collect();

        let final_positions = self.follow_flows(&flow_y, &flow_x, &is_cell, width, height);

        let labels = Self::label_sinks(&final_positions, &is_cell, width, height);

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

    fn cite(&self) -> Option<&'static CitationMetadata> {
        Some(&CitationMetadata {
            cite_key: "stringer2021cellpose",
            title: "Cellpose: a generalist algorithm for cellular segmentation",
            authors: &[
                "Carsen Stringer",
                "Tim Wang",
                "Michalis Michaelos",
                "Marius Pachitariu",
            ],
            year: 2021,
            container: Some("Nature Methods"),
            doi: Some("10.1038/s41592-020-01018-x"),
            url: Some("https://doi.org/10.1038/s41592-020-01018-x"),
            pages: Some("100-106"),
        })
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
    fn run_model_tiled(model: &CModule, input: &Tensor, device: Device) -> Result<Tensor, InternalErrors> {
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

                let mut acc_region = acc.narrow(2, y, Self::TILE_SIZE).narrow(3, x, Self::TILE_SIZE);
                acc_region += &tile_out * &mask;
                let mut norm_region = norm.narrow(2, y, Self::TILE_SIZE).narrow(3, x, Self::TILE_SIZE);
                norm_region += &mask;
            }
        }

        let acc = acc.ok_or_else(|| {
            InternalErrors::Generic("Cellpose produced no output tiles".into())
        })?;
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
        let n = (((1.0 + 2.0 * overlap) * padded_len as f32) / Self::TILE_SIZE as f32).ceil() as i64;
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
        Tensor::from_slice(&mask2d)
            .to_device(device)
            .reshape([1, 1, Self::TILE_SIZE, Self::TILE_SIZE])
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
    use kornia_tensor::CpuAllocator;

    fn cellpose(min_object_size: i32) -> Cellpose {
        Cellpose {
            model_path: PathBuf::new(),
            object_class_id: SegmentationClass(7),
            input_channels: 2,
            probability_threshold: 0.5,
            flow_iterations: 10,
            min_object_size,
        }
    }

    fn gray_ctx(width: usize, height: usize, values: Vec<f32>) -> PipelineContext {
        let img =
            Image::<f32, 1, CpuAllocator>::new(ImageSize { width, height }, values, CpuAllocator)
                .unwrap();
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
        assert_eq!(mask.size(), vec![1, 1, Cellpose::TILE_SIZE, Cellpose::TILE_SIZE]);

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

        let cell_prob: Vec<f32> = Vec::try_from(
            &output
                .narrow(1, 2, 1)
                .sigmoid()
                .reshape([height * width]),
        )
        .unwrap();
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
