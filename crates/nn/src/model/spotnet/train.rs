//! # train
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-11-27
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.
//!
//! Training a SpotNet from clicked spots and training areas.

use crate::{
    NnError, TrainControl, TrainStats,
    model::{
        Dataset,
        spotnet::network::{SpotNet, SpotNetConfig},
    },
};
use burn::Tensor;
use burn::tensor::activation::log_sigmoid;

/// One clicked spot: its centre in px (a pixel's centre is +0.5).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Spot {
    pub x: f32,
    pub y: f32,
}

/// What the user drew on one training image.
#[derive(Debug, Clone, PartialEq)]
pub struct SpotLabels {
    pub width: usize,
    pub height: usize,
    pub spots: Vec<Spot>,
    /// Per pixel: inside a training area. Spots outside are ignored.
    pub area: Vec<bool>,
}

/// Settings of one training.
#[derive(Debug, Clone, PartialEq)]
pub struct SpotNetTrainParams {
    pub epochs: usize,
    /// Patches per epoch (an image gives many patches).
    pub patches_per_epoch: usize,
    pub batch_size: usize,
    /// Side of a square patch in px, e.g. 128.
    pub patch_size: usize,
    pub learning_rate: f64,
    /// Width of the Gaussian hills in px, e.g. 1.5.
    pub sigma: f32,
}

/// Trains a new SpotNet on `data`.
pub fn train(
    data: &dyn Dataset<SpotLabels>,
    config: &SpotNetConfig,
    params: &SpotNetTrainParams,
    control: &TrainControl<'_>,
) -> Result<(SpotNet, TrainStats), NnError> {
    // 1. Prepare:   device = autodiff CPU, model = SpotNet::new(config, control.seed, &device),
    //               optimizer = Adam.
    // 2. Per epoch:
    //    a. pick `patches_per_epoch` random patches inside the training areas
    //    b. per batch of `batch_size` patches:
    //         image patch → input tensor  [B, in_channels, P, P]
    //         gaussian_map patch → target [B, 1, P, P]
    //         area_mask patch → mask      [B, 1, P, P]
    //         prediction = model.forward(input)
    //         loss = masked_loss(prediction, target, mask)
    //         grads = loss.backward() → optimizer step
    //    c. report the epoch's loss via control.progress, check control.is_cancelled()
    // 3. Return model.valid() (the model without autodiff) and the stats.
    todo!()
}

/// How much more a pixel on a spot centre counts than background in the
/// loss (`weight = 1 + SPOT_WEIGHT * target`): almost every pixel is
/// background, so without it "never a spot" is nearly right - and the
/// network learns exactly that.
const SPOT_WEIGHT: f32 = 10.0;

/// How far the prediction is from the perfect answer, only where mask = 1:
/// the binary cross-entropy of `logits` (the network before its sigmoid,
/// [`SpotNet::forward_logits`]) against `target`, weighted towards the
/// spots ([`SPOT_WEIGHT`]) and averaged over the weighted mask pixels.
///
/// Cross-entropy, not squared error: after a sigmoid the squared error's
/// gradient vanishes once the output is near 0 - a network that says "no
/// spot" everywhere stays stuck there. Here the gradient is `p - target`,
/// so a wrong answer is corrected however sure the network was.
///
/// A mask without any pixel (a patch outside every training area) gives 0,
/// never NaN.
fn masked_loss(logits: Tensor<4>, target: Tensor<4>, mask: Tensor<4>) -> Tensor<1> {
    // -[t ln(p) + (1 - t) ln(1 - p)], with ln(p) = log_sigmoid(z) and
    // ln(1 - p) = log_sigmoid(-z): stable even for z = -50.
    let per_pixel = -(target.clone() * log_sigmoid(logits.clone())
        + (-target.clone() + 1.0) * log_sigmoid(-logits));
    let weight = (target * SPOT_WEIGHT + 1.0) * mask;
    let total = weight.clone().sum().clamp_min(1.0);
    (per_pixel * weight).sum() / total
}

#[cfg(test)]
mod tests {
    use crate::model::spotnet::network::{SpotNet, SpotNetConfig};
    use crate::model::spotnet::target::gaussian_map;
    use crate::model::spotnet::train::Spot;

    use super::masked_loss;
    use burn::Tensor;
    use burn::optim::{AdamConfig, GradientsParams};
    use burn::tensor::activation::sigmoid;
    use burn::tensor::{Device, TensorData};

    /// A `[1, 1, 2, 2]` tensor from four values (row-major).
    fn patch(values: [f32; 4]) -> Tensor<4> {
        Tensor::<1>::from_floats(values, &Device::flex()).reshape([1, 1, 2, 2])
    }

    fn loss(logits: [f32; 4], target: [f32; 4], mask: [f32; 4]) -> f32 {
        masked_loss(patch(logits), patch(target), patch(mask)).into_scalar()
    }

    /// The logit of probability `p`: `sigmoid(logit(p)) == p`.
    fn logit(p: f32) -> f32 {
        (p / (1.0 - p)).ln()
    }

    #[test]
    fn a_sure_right_answer_has_almost_no_loss() {
        let target = [0.0, 1.0, 0.0, 1.0];
        let l = loss([-20.0, 20.0, -20.0, 20.0], target, [1.0; 4]);
        assert!(l < 1e-6, "{l}");
    }

    #[test]
    fn on_a_hill_the_target_value_itself_has_the_smallest_loss() {
        // A target of 0.6 (beside a spot centre): 0.6 beats 0.5 and 0.7.
        let at = |p: f32| loss([logit(p); 4], [0.6; 4], [1.0; 4]);
        assert!(at(0.6) < at(0.5), "{} vs {}", at(0.6), at(0.5));
        assert!(at(0.6) < at(0.7), "{} vs {}", at(0.6), at(0.7));
    }

    #[test]
    fn errors_outside_the_training_areas_do_not_count() {
        // Right inside the mask, completely wrong on the two pixels outside.
        let inside_only = loss(
            [-20.0, -20.0, 20.0, -20.0],
            [0.0, 0.0, 1.0, 0.0],
            [1.0, 1.0, 1.0, 1.0],
        );
        let with_outside = loss(
            [20.0, 20.0, 20.0, -20.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0, 1.0],
        );
        assert!(
            inside_only < 1e-6 && with_outside < 1e-6,
            "{inside_only}, {with_outside}"
        );
    }

    #[test]
    fn an_unsure_answer_on_background_costs_ln_2() {
        // 50 % everywhere (logit 0) where there is no spot: -ln(0.5) = ln 2.
        let l = loss([0.0; 4], [0.0; 4], [1.0; 4]);
        assert!((l - std::f32::consts::LN_2).abs() < 1e-6, "{l}");
    }

    #[test]
    fn a_missed_spot_costs_eleven_times_a_false_alarm() {
        // The same mistake (3 logits the wrong way) once on the spot centre,
        // once on background: the spot counts 1 + SPOT_WEIGHT = 11 times.
        let target = [1.0, 0.0, 0.0, 0.0];
        let missed = loss([-3.0, -20.0, -20.0, -20.0], target, [1.0; 4]);
        let false_alarm = loss([20.0, 3.0, -20.0, -20.0], target, [1.0; 4]);
        assert!(
            (missed / false_alarm - 11.0).abs() < 0.01,
            "{missed} / {false_alarm}"
        );
    }

    #[test]
    fn a_patch_without_training_area_has_no_loss_and_no_nan() {
        assert_eq!(loss([0.0; 4], [1.0; 4], [0.0; 4]), 0.0);
    }

    #[test]
    /// The "overfit one image" check: can SpotNet learn a single image by
    /// heart? If not, the network, the loss or the loop is broken.
    ///
    /// Kept small for the debug build (a few seconds): a 32 x 32 image, 8
    /// channels instead of 16, a learning rate of 1e-2, and a stop as soon
    /// as the answer is right - checked every 10 steps.
    fn spotnet_learns_one_image() {
        const SIZE: usize = 32;
        const MAX_STEPS: usize = 300;
        let spots = vec![
            Spot { x: 8.5, y: 8.5 },
            Spot { x: 22.5, y: 12.5 },
            Spot { x: 14.5, y: 24.5 },
            Spot { x: 1.5, y: 18.5 }, // near the left edge
        ];

        let target = gaussian_map(SIZE, SIZE, &spots, 1.5);
        let mask = vec![1.0f32; SIZE * SIZE]; // the whole image is a training area
        let device = Device::flex().autodiff();
        let image = test_image(SIZE, SIZE, &spots);

        let shape = [1, 1, SIZE, SIZE];
        let x = Tensor::<4>::from_data(TensorData::new(image, shape), &device);
        let target_t = Tensor::<4>::from_data(TensorData::new(target.clone(), shape), &device);
        let mask = Tensor::<4>::from_data(TensorData::new(mask, shape), &device);

        let config = SpotNetConfig {
            in_channels: 1,
            channels: 8,
            ..SpotNetConfig::default()
        };
        // Fixed start weights: the same run every time, also in parallel tests.
        let mut model = SpotNet::new(&config, 7, &device);
        let mut optim = AdamConfig::new().init();

        // Right = above 0.5 on every spot centre, below 0.1 wherever the
        // target is 0 (beyond every hill).
        let right = |prediction: &[f32]| {
            let spots_found = spots
                .iter()
                .all(|s| prediction[s.y as usize * SIZE + s.x as usize] > 0.5);
            let background_dark = target
                .iter()
                .zip(prediction)
                .all(|(&t, &p)| t > 0.0 || p < 0.1);
            spots_found && background_dark
        };

        let mut first = None;
        let mut last = f32::NAN;
        let mut steps = 0;
        let mut learned = false;
        while steps < MAX_STEPS {
            let logits = model.forward_logits(x.clone());
            if steps % 10 == 0 {
                let prediction = sigmoid(logits.clone())
                    .into_data()
                    .try_to_vec::<f32>()
                    .expect("f32 output");
                if right(&prediction) {
                    learned = true;
                    break;
                }
            }
            let loss = masked_loss(logits, target_t.clone(), mask.clone());
            // Read the value before `backward()` consumes the loss.
            last = loss.clone().into_scalar();
            first.get_or_insert(last);
            if steps % 20 == 0 {
                println!("step {steps:3}: loss {last:.5}");
            }
            let grads = GradientsParams::from_grads(loss.backward(), &model);
            model = optim.step(1e-2, model, grads);
            steps += 1;
        }
        let first = first.expect("at least one step ran");
        println!("{steps} steps: first loss {first:.5}, last {last:.5}");

        assert!(
            learned,
            "not learned in {MAX_STEPS} steps (loss {first} → {last})"
        );
        assert!(last < first, "the loss did not fall: {first} → {last}");
    }

    /// A 1-channel test image like a microscope would take it: background
    /// 0.1, a bright blob (sigma 2 px, so ~10 px wide, peak +0.8) at each
    /// spot - added up, as light from close spots adds up too.
    fn test_image(width: usize, height: usize, spots: &[Spot]) -> Vec<f32> {
        const BACKGROUND: f32 = 0.1;
        const BRIGHTNESS: f32 = 0.8;
        const SIGMA: f32 = 2.0;
        let two_sigma_sq = 2.0 * SIGMA * SIGMA;

        let mut out = vec![0.0; width * height];
        for py in 0..height {
            for px in 0..width {
                let mut value = BACKGROUND;
                for spot in spots {
                    let dx = (px as f32 + 0.5) - spot.x;
                    let dy = (py as f32 + 0.5) - spot.y;
                    value += BRIGHTNESS * (-(dx * dx + dy * dy) / two_sigma_sq).exp();
                }
                out[py * width + px] = value;
            }
        }
        out
    }

    #[test]
    fn the_test_image_has_blobs_on_a_background() {
        let image = test_image(20, 10, &[Spot { x: 5.5, y: 4.5 }]);
        let at = |x: usize, y: usize| image[y * 20 + x];
        assert!((at(5, 4) - 0.9).abs() < 1e-6, "centre: {}", at(5, 4));
        assert!(
            at(6, 4) < at(5, 4) && at(6, 4) > 0.5,
            "beside: {}",
            at(6, 4)
        );
        assert!((at(19, 9) - 0.1).abs() < 1e-3, "far away: {}", at(19, 9));
    }
}
