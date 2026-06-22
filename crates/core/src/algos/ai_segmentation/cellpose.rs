//! # cellpose
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

/// Instance segmentation using a pretrained Cellpose model exported as TorchScript.
///
/// The model is expected to accept a `[1, 1, H, W]` float tensor (single-channel,
/// same normalization as the rest of the pipeline) and return a `[1, C, H, W]`
/// tensor with `C >= 3` channels: the vertical flow `dY` (channel 0), the
/// horizontal flow `dX` (channel 1) and the cell-probability logits (channel 2),
/// which is Cellpose's spatial-gradient representation. Exports that wrap the
/// output in a tuple (e.g. `(flows, style)`) are also supported — the first
/// tensor with at least three channels is used.
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
#[cmdsmeta(category = "segment", next = "measure", display_name = "AI Cellpose Segmentation")]
pub struct Cellpose {
    /// Path to a TorchScript-exported Cellpose model (`torch.jit.script`/`torch.jit.trace`).
    #[cmdsmeta(file_extensions = "pt,pth")]
    pub model_path: PathBuf,

    /// The class assigned to pixels of every detected object. All other
    /// pixels are assigned `SegmentationClass::BACKGROUND`.
    #[cmdsmeta(default = SegmentationClass(1))]
    pub object_class_id: SegmentationClass,

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
        _cache: &mut PipelineCache,
    ) -> Result<(), InternalErrors> {
        let device = Device::cuda_if_available();
        let model = CModule::load_on_device(&self.model_path, device).map_err(|e| {
            InternalErrors::Generic(format!(
                "Failed to load Cellpose model from {}: {e}",
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

        let output = Self::run_model(&model, input)?;
        let out_sizes = output.size();
        if out_sizes.len() < 4 {
            return Err(InternalErrors::Generic(
                "Cellpose model output has too few dimensions; expected `[1, C, H, W]`".into(),
            ));
        }
        let channels = out_sizes[out_sizes.len() - 3];
        if channels < 3 {
            return Err(InternalErrors::Generic(
                "Cellpose model output has fewer than 3 channels; expected `[dY, dX, cellprob]`"
                    .into(),
            ));
        }
        let out_h = out_sizes[out_sizes.len() - 2] as usize;
        let out_w = out_sizes[out_sizes.len() - 1] as usize;
        if out_h != height || out_w != width {
            return Err(InternalErrors::Generic(format!(
                "Cellpose model output resolution {out_w}x{out_h} does not match the input \
                 resolution {width}x{height}"
            )));
        }

        let channel_dim = out_sizes.len() as i64 - 3;
        let flow_y = Self::channel_to_vec(&output.narrow(channel_dim, 0, 1), width, height)?;
        let flow_x = Self::channel_to_vec(&output.narrow(channel_dim, 1, 1), width, height)?;
        let cell_prob = Self::channel_to_vec(
            &output.narrow(channel_dim, 2, 1).sigmoid(),
            width,
            height,
        )?;

        // Pixels above the cell-probability threshold take part in the dynamics.
        let is_cell: Vec<bool> = cell_prob
            .iter()
            .map(|&p| p >= self.probability_threshold)
            .collect();

        let final_positions =
            self.follow_flows(&flow_y, &flow_x, &is_cell, width, height);

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
            if instance_id == 0 {
                continue;
            }
            inst_slice[i] = instance_id;
            seg_slice[i] = foreground_class;
        }
    }
}
