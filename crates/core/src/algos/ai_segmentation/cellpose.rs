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

use evanalyzer_cfg::core_types::{CitationMetadata, InternalErrors, SegmentationClass};
use evanalyzer_nn::model::cellpose_sam::{CellposeSam, CellposeSamParams};
use evanalyzer_nn::{Image, Model};
use macros::CommandsMeta;

use super::InputNormalization;
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
    /// The Cellpose-SAM model file (EVAnalyzer's `.evanet` or `.safetensors`, converted
    /// once from Cellpose's own, e.g. `cpsam_v2`).
    #[cmdsmeta(file_extensions = NN_MODEL_EXTENSIONS)]
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

    /// How the image is scaled before the model sees it: by default the
    /// tile's 1st and 99th percentile become 0 and 1, as Cellpose does.
    #[cmdsmeta(default = InputNormalization::ModelDefault, optional = true, visibility = Advanced)]
    pub normalization: InputNormalization,
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

        let params = CellposeSamParams {
            normalization: self.normalization.to_nn(),
            net_width,
            net_height,
            input_channels: self.input_channels.clamp(1, 3) as usize,
            replicate_gray_channel: self.replicate_gray_channel,
            probability_threshold: self.probability_threshold,
            flow_iterations: self.flow_iterations.max(1) as usize,
            flow_threshold: self.flow_threshold,
            min_object_size: self.min_object_size.max(0) as usize,
            cellpose_dynamics: self.cellpose_postprocessing,
        };
        let image = Image {
            width,
            height,
            channels: 1,
            data: input_image.as_slice(),
        };
        ai_devices::run::<CellposeSam>(
            &self.model_path,
            image,
            &params,
            instance_map.as_slice_mut(),
        )?;
        super::fill_class(
            instance_map.as_slice(),
            segmentation_map.as_slice_mut(),
            self.segmentation_class,
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

    /// What the step keeps per worker: the flows and the cell probability,
    /// and the dynamics' planes - 42 bytes per pixel (`tests::tile_ram`;
    /// the image is only scaled down, so per tile pixel at most).
    fn ram_per_tile_pixel(&self) -> f64 {
        <CellposeSam as Model>::POST_BYTES_PER_PIXEL
    }

    /// One inference at a time on the CPU: the network on one 256² piece
    /// (Cellpose-SAM's ViT; estimated from a 1024² run: ~1.7 GB peak with
    /// its 1.2 GB of weights - BENCHMARK.md "AI models: Cellpose through the
    /// plugin", libtorch) plus the pieces blended over the image, 46 bytes per pixel
    /// (measured with a stand-in: 766 MB at 4096²), and the model's
    /// weights. The image is only ever scaled down (`max_resize`), so the
    /// tile's pixels bound it.
    fn shared_ram_bytes(&self, tile_width: usize, tile_height: usize) -> u64 {
        const NETWORK_BYTES: u64 = 512 << 20;
        const BLEND_BYTES_PER_PIXEL: u64 = <CellposeSam as Model>::NET_BYTES_PER_PIXEL as u64;
        NETWORK_BYTES
            + BLEND_BYTES_PER_PIXEL * (tile_width * tile_height) as u64
            + super::model_bytes(&self.model_path)
    }
}

impl Cellpose {
    /// Factor the image is scaled by before segmentation: shrinks the full
    /// image's longest side to `max_resize`, never enlarges it.
    fn resize_factor(&self, full_image: kornia_image::ImageSize) -> f64 {
        let longest = full_image.width.max(full_image.height);
        if self.max_resize <= 0 || longest <= self.max_resize as usize {
            return 1.0;
        }
        self.max_resize as f64 / longest as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
            normalization: InputNormalization::ModelDefault,
        }
    }

    fn gray_ctx(width: usize, height: usize, values: Vec<f32>) -> PipelineContext {
        let img = Image::<f32, 1>::new(ImageSize { width, height }, values).unwrap();
        PipelineContext::new_from_image_test(img).unwrap()
    }

    // ---- execute(): a real model file (a tiny random Cellpose-SAM of 32²
    // tiles, nn's `test_support`) through the engine ----

    /// The step runs the model file over several tiles and writes objects:
    /// every instance pixel has the step's class.
    #[test]
    fn execute_runs_the_model_file_and_writes_instances() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("cellpose.evanet");
        let config = evanalyzer_nn::test_support::tiny_cellpose_sam_config();
        std::fs::write(
            &path,
            evanalyzer_nn::test_support::cellpose_sam_file(&config),
        )
        .unwrap();
        let (w, h) = (70, 45);
        let pixels: Vec<f32> = (0..w * h)
            .map(|i| ((i * 7919) % 101) as f32 / 101.0)
            .collect();
        let cmd = Cellpose {
            model_path: path,
            probability_threshold: 0.0,
            replicate_gray_channel: true,
            ..cellpose(0)
        };
        let mut ctx = gray_ctx(w, h, pixels);
        cmd.execute(&mut ctx, &mut GlobalPipelineCache::default())
            .unwrap();
        let classes = ctx.get_segmentation_map().unwrap().as_slice().to_vec();
        let instances = ctx.get_instance_map().unwrap().as_slice().to_vec();
        assert_eq!(instances.len(), w * h);
        for (class, instance) in classes.iter().zip(&instances) {
            if *instance > 0 {
                assert_eq!(*class, 7);
            }
        }
    }

    #[test]
    fn execute_errors_when_the_model_path_does_not_exist() {
        let dir = tempfile::tempdir().unwrap();
        let cmd = Cellpose {
            model_path: dir.path().join("missing.evanet"),
            ..cellpose(0)
        };
        let mut ctx = gray_ctx(2, 2, vec![0.0; 4]);
        let mut cache = GlobalPipelineCache::default();

        let err = cmd.execute(&mut ctx, &mut cache).unwrap_err();
        assert!(matches!(err, InternalErrors::Generic(_)));
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

        // What the template's Cellpose -> Fill Object Holes steps do: nn's
        // mask building and size filter, then the hole filling.
        let segment = |cellpose_dynamics: bool| -> Vec<u32> {
            use evanalyzer_nn::model::cellpose_sam::{CellposeSamPlan, CellposeSamRaw};
            let raw = CellposeSamRaw {
                flow_y: floats("flow_y"),
                flow_x: floats("flow_x"),
                cell_prob: cell_prob.clone(),
            };
            let plan = CellposeSamPlan {
                image: (width, height),
                net: (width, height),
            };
            let params = CellposeSamParams {
                normalization: evanalyzer_nn::Normalization::None,
                net_width: width,
                net_height: height,
                input_channels: 2,
                replicate_gray_channel: false,
                probability_threshold: 0.5,
                flow_iterations: 200,
                flow_threshold: 0.0,
                min_object_size: 15,
                cellpose_dynamics,
            };
            let mut instances = vec![0u32; width * height];
            CellposeSam::post(raw, plan, &params, &mut instances).unwrap();
            crate::algos::morphology::fill_object_holes::fill_instance_holes(
                &mut instances,
                width,
                height,
            );
            instances
        };

        assert_same_objects(&segment(true), &expected);

        // The fixture tells the two modes apart: without the post-processing
        // the result differs from Cellpose.
        let mismatches = count_object_mismatches(&segment(false), &expected);
        println!("default mode: {mismatches} pixels differ from Cellpose");
        assert!(mismatches > 0);
    }
}
