#[cfg(test)]
mod repro_slow_watershed_tmp {
    use crate::algos::ImageAlgorithm;
    use crate::algos::filters::blur::Blur;
    use crate::algos::filters::rolling_ball::{BallType, RollingBall};
    use crate::algos::segmentation::connected_components::ConnectedComponents;
    use crate::algos::segmentation::threshold::{Threshold, ThresholdEntry, ThresholdMethod};
    use crate::algos::segmentation::watershed::Watershed;
    use crate::image::{ImageContainer, ImageDebugExt};
    use crate::init_java_wrapper;
    use crate::{ImageReader, ReadMode, ZProjection};
    use crate::pipeline::pipeline_cache::PipelineCache;
    use crate::pipeline::pipeline_context::PipelineContext;
    use crate::{ImageTile, image::PixelSizes, pipeline::pipeline::PipelineImageMeta};
    use evanalyzer_cfg::core_types::{PixelUnits, SegmentationClass};
    use std::time::Instant;

    #[test]
    fn repro_real_slow_watershed_pipeline() {
        init_java_wrapper(2_000_000_000).unwrap();

        let path: std::path::PathBuf = "/workspaces/evanalyzer/docs/slowwatershed.tif".into();
        let reader = ImageReader::new(&path, ReadMode::Default).expect("failed to open tiff");

        let t_load = Instant::now();
        let result = reader
            .read_image_tile_combined(
                0,                      // series
                0,                      // resolution_idx
                ZProjection::MaxIntensity,
                &Some(0..=0),           // z_range: exactly z=0
                0,                      // t_stack
                Some(&vec![1]),         // channel 1
                &ImageTile {
                    offset_x: 0,
                    offset_y: 0,
                    width: 0,
                    height: 0,
                },
            )
            .expect("failed to read tile");
        println!("Image load took {:?}", t_load.elapsed());

        let channel = result.into_iter().next().expect("no channel returned");
        let ImageContainer::F32Gray(img) = &*channel.image else {
            panic!("expected F32Gray");
        };
        let size = img.size();
        println!("Loaded image: {}x{}", size.width, size.height);

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
                let t0 = Instant::now();
                $cmd.execute(&mut ctx, &mut cache).expect($name);
                println!("{} took {:?}", $name, t0.elapsed());
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
        run_step!(
            "Watershed",
            Watershed {
                maximum_finder_tolerance: 0.5,
                smoothing_sigma: 0.0,
                min_object_size: 0,
            }
        );
    }
}
