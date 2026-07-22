#[cfg(test)]
mod repro_slow_watershed {
    use crate::algos::ImageAlgorithm;
    use crate::algos::filters::blur::Blur;
    use crate::algos::filters::rolling_ball::{BallType, RollingBall};
    use crate::algos::segmentation::connected_components::ConnectedComponents;
    use crate::algos::segmentation::threshold::{Threshold, ThresholdEntry, ThresholdMethod};
    use crate::algos::segmentation::watershed::Watershed;
    use crate::image::ImageContainer;
    use crate::init_java_wrapper;
    use crate::pipeline::pipeline_cache::PipelineCache;
    use crate::pipeline::pipeline_context::PipelineContext;
    use crate::{ImageReader, ReadMode, ZProjection};
    use crate::{ImageTile, image::PixelSizes, pipeline::pipeline::PipelineImageMeta};
    use evanalyzer_cfg::core_types::{PixelUnits, SegmentationClass};
    use std::time::{Duration, Instant};

    /// Requires a local, non-repo test file (`docs/slowwatershed.tif`, a real
    /// ~176MB microscopy image) that reproduced a genuine pathological bug:
    /// `Watershed` took 200+ seconds on this data because `DistanceTransform`
    /// left stale, non-zero garbage in background pixels of the EDM (a reused
    /// scratch buffer wasn't zeroed - see the fix and its regression test in
    /// `spartial_transform::edm::tests`), which fooled the maxima finder into
    /// treating ~87,000 background pixels as local maxima instead of ~800.
    /// Kept `#[ignore]`d since the fixture isn't checked in; run manually with
    /// `cargo test --release -- --ignored repro_real_slow_watershed_pipeline`
    /// after placing the file, to guard against this class of bug regressing.
    #[test]
    #[ignore]
    fn repro_real_slow_watershed_pipeline() {
        init_java_wrapper(2_000_000_000).unwrap();

        let path: std::path::PathBuf = "/workspaces/evanalyzer/docs/slowwatershed.tif".into();
        let reader = ImageReader::new(&path, ReadMode::Default).expect("failed to open tiff");

        let result = reader
            .read_image_tile_combined(
                0,              // series
                0,              // resolution_idx
                ZProjection::MaxIntensity,
                &Some(0..=0),   // z_range: exactly z=0
                0,              // t_stack
                Some(&vec![1]), // channel 1
                &ImageTile {
                    offset_x: 0,
                    offset_y: 0,
                    width: 0,
                    height: 0,
                },
            )
            .expect("failed to read tile");

        let channel = result.into_iter().next().expect("no channel returned");
        let ImageContainer::F32Gray(img) = &*channel.image else {
            panic!("expected F32Gray");
        };
        let size = img.size();

        let mut ctx = PipelineContext::new_from_image(
            std::path::PathBuf::default(),
            PipelineImageMeta {
                image_tile_info: ImageTile {
                    offset_x: 0,
                    offset_y: 0,
                    width: size.width,
                    height: size.height,
                },
                full_image_width: size,
                is_rgb: false,
                nr_of_bits: 16,
                pixel_sizes: PixelSizes {
                    px_size_x: 1.0,
                    px_size_y: 1.0,
                    px_size_z: 1.0,
                },
            },
            channel.image.clone(),
        )
        .unwrap();
        let mut cache = PipelineCache::default();

        macro_rules! run_step {
            ($name:expr, $cmd:expr) => {{
                $cmd.execute(&mut ctx, &mut cache).expect($name);
            }};
        }

        run_step!(
            "RollingBall",
            RollingBall {
                radius: 4.0,
                ball_type: BallType::Paraboloid,
                pre_smooth: false,
            }
        );
        run_step!("Blur1", Blur { kernel_size: 3 });
        run_step!("Blur2", Blur { kernel_size: 3 });
        run_step!(
            "Threshold",
            Threshold {
                thresholds: vec![ThresholdEntry {
                    method: ThresholdMethod::Triangle,
                    min_threshold: 0.0,
                    max_threshold: 65535.0,
                    unit: PixelUnits::Bit,
                    object_class_id: SegmentationClass(1),
                }],
            }
        );
        run_step!(
            "ConnectedComponents",
            ConnectedComponents { min_size_px: 0 }
        );

        let t0 = Instant::now();
        run_step!(
            "Watershed",
            Watershed {
                maximum_finder_tolerance: 0.5,
                smoothing_sigma: 0.0,
                min_object_size: 0,
            }
        );
        let elapsed = t0.elapsed();
        assert!(
            elapsed < Duration::from_secs(10),
            "Watershed took {elapsed:?} - expected well under a second; \
             this indicates the stale-scratch-buffer EDM bug (or something \
             like it) has regressed"
        );
    }
}
