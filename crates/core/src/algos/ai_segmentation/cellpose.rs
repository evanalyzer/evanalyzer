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

/// Instance segmentation using a pretrained Cellpose model exported as TorchScript.
///
/// The model is fed a `[1, input_channels, H, W]` float tensor: the (normalized)
/// grayscale image is placed in channel 0 and any remaining channels are filled
/// with zeros. Standard Cellpose networks expect **two** channels (cytoplasm +
/// optional nucleus), which is the default; single-channel exports use
/// `input_channels = 1`. The model must return a `[1, C, H, W]` tensor with
/// `C >= 3` channels: the vertical flow `dY` (channel 0), the horizontal flow
/// `dX` (channel 1) and the cell-probability logits (channel 2), which is
/// Cellpose's spatial-gradient representation. Exports that wrap the output in a
/// tuple (e.g. `(flows, style)`) are also supported — the first tensor with at
/// least three channels is used.
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
    /// channel 0; any further channels are zero-filled. Standard Cellpose models
    /// take `2` (cytoplasm + optional nucleus); set `1` for single-channel
    /// exports, or higher to match a custom model.
    #[cmdsmeta(default = 2, min = 1, max = 8, step = 1)]
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
        let cell_prob =
            Self::channel_to_vec(&output.narrow(channel_dim, 2, 1).sigmoid(), width, height)?;

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
    fn flow_free_cellpose_model(
        in_channels: i64,
        width: i64,
        height: i64,
    ) -> (tempfile::TempDir, std::path::PathBuf) {
        trace_and_save_model(in_channels, height, width, |x| {
            let image_channel = x.narrow(1, 0, 1);
            let flow_y = image_channel.zeros_like();
            let flow_x = image_channel.zeros_like();
            let cell_logit = image_channel * 20.0 - 10.0;
            Tensor::cat(&[flow_y, flow_x, cell_logit], 1)
        })
    }

    #[test]
    fn execute_end_to_end_merges_adjacent_cell_pixels_into_one_instance() {
        let (_dir, model_path) = flow_free_cellpose_model(2, 3, 1);
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
        let (_dir, model_path) = flow_free_cellpose_model(1, 2, 1);
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
        let (_dir, model_path) = trace_and_save_model(2, 1, 2, |x| {
            // Collapses [1,2,1,2] down to rank 2, well short of the required
            // `[1, C, H, W]`.
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
        let (_dir, model_path) = trace_and_save_model(2, 1, 2, |x| {
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
    fn execute_errors_when_the_model_output_resolution_does_not_match_the_input() {
        let (_dir, model_path) = trace_and_save_model(2, 1, 4, |x| {
            let c = x.narrow(1, 0, 1);
            let cropped = c.narrow(3, 0, 2); // half the input width
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
            matches!(err, InternalErrors::Generic(msg) if msg.contains("does not match the input resolution"))
        );
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
