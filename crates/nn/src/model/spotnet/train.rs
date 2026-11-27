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

use super::validate::{Tally, ValidationScore};
use crate::{
    NnError, TrainControl, TrainStats,
    augment::{AugmentParams, Augmentation},
    model::{
        Dataset, OwnedImage,
        spotnet::{
            network::{SpotNet, SpotNetConfig},
            target::{area_mask, gaussian_map},
        },
    },
    train::{MIN_IMAGES_FOR_AUTO_SPLIT, SeededUniform, TrainProgress, Validation},
};
use burn::Tensor;
use burn::module::Module;
use burn::optim::{AdamConfig, GradientsParams};
use burn::tensor::activation::log_sigmoid;
use burn::tensor::{Device, TensorData};

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
    /// Per pixel: inside a validation area the user drew - never trained on
    /// (all `false` without any). Its spots must be annotated as completely
    /// as in a training area.
    pub validation: Vec<bool>,
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
    /// The random changes of every patch.
    pub augment: AugmentParams,
    /// What the training checks itself on.
    pub validation: Validation,
}

/// A finished training.
#[derive(Debug, Clone)]
pub struct SpotNetTraining {
    /// The best model: of the epoch with the most spots found right on the
    /// held-back areas (F1, then the lower loss), the last one without
    /// validation.
    pub model: SpotNet,
    pub stats: TrainStats,
    pub validation: ValidationOutcome,
}

/// How a training was checked.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ValidationOutcome {
    /// The score of the kept model on the held-back areas.
    Scored {
        epoch: usize,
        score: ValidationScore,
    },
    /// Switched off.
    Off,
    /// `Validation::Auto` without drawn validation areas and with fewer than
    /// [`MIN_IMAGES_FOR_AUTO_SPLIT`] images: nothing to hold back.
    TooFewImages { images: usize },
}

/// Trains a new SpotNet on `data`, on the CPU (the GPU follows later).
///
/// Image by image - only one image is in RAM at a time: per epoch every
/// image with pixels to train on gives its share of `patches_per_epoch`
/// random patches (each centred on such a pixel), trained in batches of
/// `batch_size`. Images smaller than `patch_size` give patches of the whole
/// image. Every patch gets its own random changes (`params.augment`); its
/// target is computed from its moved spots.
///
/// After every epoch the model is checked on the held-back areas
/// (`params.validation`, see [`plan`]) and the best epoch's model kept
/// (see [`better`]); `stats.best_val_loss` is that epoch's loss.
/// Start weights, patches, changes and the held-back images come from
/// `control.seed`: the same data and seed give the same model.
pub fn train(
    data: &dyn Dataset<SpotLabels>,
    config: &SpotNetConfig,
    params: &SpotNetTrainParams,
    control: &TrainControl<'_>,
) -> Result<SpotNetTraining, NnError> {
    check_params(params)?;
    let plan = plan(data, config, params.validation, control.seed)?;
    let patches_per_image = params.patches_per_epoch.div_ceil(plan.train.len());
    let margin = config.receptive_field() / 2;

    let device = Device::flex().autodiff();
    let mut model = SpotNet::new(config, control.seed, &device);
    let mut optim = AdamConfig::new().init();
    // Its own stream (seed + 1): the patches don't depend on how many
    // random numbers the start weights took.
    let mut random = SeededUniform::new(control.seed.wrapping_add(1));

    let mut final_train_loss = f32::NAN;
    let mut final_val_loss = None;
    let mut best: Option<(usize, ValidationScore, SpotNet)> = None;
    let mut epochs_run = 0;
    for epoch in 0..params.epochs {
        let (mut loss_sum, mut batches) = (0.0f32, 0usize);
        for &index in &plan.train {
            if control.is_cancelled() {
                return Err(NnError::Cancelled);
            }
            let (image, labels) = data.get(index)?;
            let sample = Sample::new(&image, &labels, params.sigma, plan.drawn_areas);
            let size = params.patch_size.min(image.width).min(image.height);
            let patches: Vec<Patch> = (0..patches_per_image)
                .map(|_| Patch {
                    origin: patch_origin(
                        &sample.area_pixels,
                        image.width,
                        image.height,
                        size,
                        &mut random,
                    ),
                    augmentation: Augmentation::random(
                        &params.augment,
                        image.channels,
                        &mut random,
                    ),
                })
                .collect();
            for batch in patches.chunks(params.batch_size) {
                let (x, target, mask) = sample.batch(batch, size, &device);
                let loss = masked_loss(model.forward_logits(x), target, mask);
                loss_sum += loss.clone().into_scalar::<f32>();
                batches += 1;
                let grads = GradientsParams::from_grads(loss.backward(), &model);
                model = optim.step(params.learning_rate, model, grads);
            }
        }
        final_train_loss = loss_sum / batches.max(1) as f32;
        epochs_run = epoch + 1;

        if !plan.validate.is_empty() {
            let checked = model.valid();
            let mut tally = Tally::default();
            for &index in &plan.validate {
                if control.is_cancelled() {
                    return Err(NnError::Cancelled);
                }
                let (image, labels) = data.get(index)?;
                let held_back = if plan.drawn_areas {
                    &labels.validation
                } else {
                    &labels.area
                };
                let tile = params.patch_size + 2 * margin;
                tally.add(
                    &checked,
                    &image,
                    &labels.spots,
                    held_back,
                    params.sigma,
                    tile,
                    margin,
                );
            }
            if let Some(score) = tally.score() {
                final_val_loss = Some(score.loss);
                if best.as_ref().is_none_or(|(_, b, _)| better(&score, b)) {
                    best = Some((epoch, score, checked));
                }
            }
        }
        (control.progress)(TrainProgress::Epoch {
            epoch,
            total_epochs: params.epochs,
            train_loss: final_train_loss,
            val_loss: final_val_loss,
        });
    }

    let stats = TrainStats::Epochs {
        epochs_run,
        total_epochs: params.epochs,
        final_train_loss,
        final_val_loss,
        best_val_loss: best.as_ref().map(|(_, score, _)| score.loss),
        best_val_epoch: best.as_ref().map(|(epoch, _, _)| *epoch),
    };
    Ok(match best {
        Some((epoch, score, model)) => SpotNetTraining {
            model,
            stats,
            validation: ValidationOutcome::Scored { epoch, score },
        },
        None => SpotNetTraining {
            model: model.valid(),
            stats,
            validation: plan.outcome,
        },
    })
}

/// Whether `score` beats `best`: more spots found right (F1), on a tie the
/// lower loss. Not the loss alone - a model still unsure everywhere has a
/// lower loss than one that is sure and slightly off, while it finds
/// nothing yet.
fn better(score: &ValidationScore, best: &ValidationScore) -> bool {
    let (f1, best_f1) = (score.f1(), best.f1());
    f1 > best_f1 || (f1 == best_f1 && score.loss < best.loss)
}

/// Refuses settings no training can run with.
fn check_params(params: &SpotNetTrainParams) -> Result<(), NnError> {
    let bad = |what: &str| Err(NnError::InvalidInput(format!("SpotNet training: {what}")));
    if params.epochs == 0 {
        return bad("at least one epoch");
    }
    if params.patches_per_epoch == 0 || params.batch_size == 0 || params.patch_size == 0 {
        return bad("patches per epoch, batch size and patch size must be at least 1");
    }
    if !(params.learning_rate > 0.0) {
        return bad("the learning rate must be above 0");
    }
    if !(params.sigma > 0.0) {
        return bad("sigma must be above 0");
    }
    if let Validation::Auto { fraction } = params.validation
        && !(fraction > 0.0 && fraction < 1.0)
    {
        return bad("the share held back for validation must be between 0 and 1");
    }
    Ok(())
}

/// Which images a training trains and checks on.
#[derive(Debug, PartialEq)]
struct Plan {
    /// Images with pixels to train on.
    train: Vec<usize>,
    /// Images to check on (empty: no check).
    validate: Vec<usize>,
    /// Checked on the drawn validation areas (never trained on); else the
    /// held-back images' training areas.
    drawn_areas: bool,
    /// How the training ends up checked when `validate` is empty.
    outcome: ValidationOutcome,
}

/// Checks every sample once (sizes, channels) and splits them:
///
/// - drawn validation areas (`Areas`, or `Auto` when there are any) are
///   checked on, and the training uses only training-area pixels outside
///   them;
/// - `Auto` without drawn areas holds back `fraction` of the images with a
///   training area (at least one, chosen from `seed`) from
///   [`MIN_IMAGES_FOR_AUTO_SPLIT`] images on - whole images, so nothing of
///   them leaks into the training through a patch's surroundings;
/// - `Off` trains on every training area.
///
/// Loads each image once.
fn plan(
    data: &dyn Dataset<SpotLabels>,
    config: &SpotNetConfig,
    validation: Validation,
    seed: u64,
) -> Result<Plan, NnError> {
    // Per image: has training pixels outside drawn validation areas, has
    // training pixels at all, has drawn validation areas.
    let mut images = Vec::new();
    for index in 0..data.len() {
        let (image, labels) = data.get(index)?;
        let bad = |what: String| {
            Err(NnError::InvalidInput(format!(
                "training image {}: {what}",
                index + 1
            )))
        };
        if image.channels != config.in_channels {
            return bad(format!(
                "{} channels, the network takes {}",
                image.channels, config.in_channels
            ));
        }
        if image.data.len() != image.channels * image.width * image.height {
            return bad("the pixels don't match its size".into());
        }
        if labels.width != image.width || labels.height != image.height {
            return bad("the annotations don't match the image size".into());
        }
        if labels.area.len() != labels.width * labels.height {
            return bad("the training area doesn't match the image size".into());
        }
        if labels.validation.len() != labels.width * labels.height {
            return bad("the validation area doesn't match the image size".into());
        }
        let outside_validation = labels
            .area
            .iter()
            .zip(&labels.validation)
            .any(|(&a, &v)| a && !v);
        let any_area = labels.area.iter().any(|&a| a);
        let drawn = labels.validation.iter().any(|&v| v);
        images.push((index, outside_validation, any_area, drawn));
    }

    let any_drawn = images.iter().any(|i| i.3);
    let use_drawn = match validation {
        Validation::Areas if !any_drawn => {
            return Err(NnError::InvalidInput(
                "no validation area: draw one, or let the training choose (Auto)".into(),
            ));
        }
        Validation::Areas => true,
        Validation::Auto { .. } => any_drawn,
        Validation::Off => false,
    };
    let no_training_area = || {
        NnError::InvalidInput("no training area: draw at least one around annotated spots".into())
    };

    if use_drawn {
        let train: Vec<usize> = images.iter().filter(|i| i.1).map(|i| i.0).collect();
        if train.is_empty() {
            return Err(no_training_area());
        }
        return Ok(Plan {
            train,
            validate: images.iter().filter(|i| i.3).map(|i| i.0).collect(),
            drawn_areas: true,
            outcome: ValidationOutcome::Off,
        });
    }

    let mut train: Vec<usize> = images.iter().filter(|i| i.2).map(|i| i.0).collect();
    if train.is_empty() {
        return Err(no_training_area());
    }
    let Validation::Auto { fraction } = validation else {
        return Ok(Plan {
            train,
            validate: Vec::new(),
            drawn_areas: false,
            outcome: ValidationOutcome::Off,
        });
    };
    if train.len() < MIN_IMAGES_FOR_AUTO_SPLIT {
        return Ok(Plan {
            outcome: ValidationOutcome::TooFewImages {
                images: train.len(),
            },
            train,
            validate: Vec::new(),
            drawn_areas: false,
        });
    }
    // A seeded shuffle (its own stream, seed + 2); the first ones are held
    // back, kept in their order.
    let mut random = SeededUniform::new(seed.wrapping_add(2));
    for i in (1..train.len()).rev() {
        train.swap(i, random.index(i + 1));
    }
    let held_back = ((train.len() as f32 * fraction).round() as usize).clamp(1, train.len() - 1);
    let mut validate: Vec<usize> = train.drain(..held_back).collect();
    validate.sort_unstable();
    train.sort_unstable();
    Ok(Plan {
        train,
        validate,
        drawn_areas: false,
        outcome: ValidationOutcome::Off,
    })
}

/// Where a patch is cut and how it is changed.
struct Patch {
    /// Top-left corner in the image.
    origin: (usize, usize),
    augmentation: Augmentation,
}

/// One loaded training image with its spots and mask.
struct Sample<'a> {
    image: &'a OwnedImage,
    spots: &'a [Spot],
    sigma: f32,
    mask: Vec<f32>,
    /// The pixels inside a training area (row-major indices) - where a
    /// patch may be centred.
    area_pixels: Vec<u32>,
}

impl<'a> Sample<'a> {
    /// `without_validation`: the drawn validation areas are not trained on.
    fn new(
        image: &'a OwnedImage,
        labels: &'a SpotLabels,
        sigma: f32,
        without_validation: bool,
    ) -> Self {
        let train: Vec<bool> = if without_validation {
            labels
                .area
                .iter()
                .zip(&labels.validation)
                .map(|(&a, &v)| a && !v)
                .collect()
        } else {
            labels.area.clone()
        };
        Self {
            image,
            spots: &labels.spots,
            sigma,
            mask: area_mask(&train),
            area_pixels: (0..train.len() as u32)
                .filter(|&i| train[i as usize])
                .collect(),
        }
    }

    /// The patches as tensors: input `[B, C, size, size]`, target and mask
    /// `[B, 1, size, size]`. Each patch is cut, changed by its augmentation
    /// (the image, the mask and its spots alike), and its target computed
    /// from the moved spots.
    fn batch(
        &self,
        patches: &[Patch],
        size: usize,
        device: &Device,
    ) -> (Tensor<4>, Tensor<4>, Tensor<4>) {
        let (width, height, channels) = (self.image.width, self.image.height, self.image.channels);
        let (mut x, mut target, mut mask) = (Vec::new(), Vec::new(), Vec::new());
        for patch in patches {
            let (x0, y0) = patch.origin;
            let a = &patch.augmentation;
            let image = cut_patch(&self.image.data, width, height, channels, x0, y0, size);
            x.extend(a.image(&image, size, size, channels));
            let area = cut_patch(&self.mask, width, height, 1, x0, y0, size);
            mask.extend(a.label(&area, size, size));
            let spots: Vec<Spot> = self
                .spots_near(x0, y0, size)
                .map(|s| {
                    let (x, y) = a.point(s.x - x0 as f32, s.y - y0 as f32, size, size);
                    Spot { x, y }
                })
                .collect();
            target.extend(gaussian_map(size, size, &spots, self.sigma));
        }
        let b = patches.len();
        let tensor = |values: Vec<f32>, c: usize| {
            Tensor::<4>::from_data(TensorData::new(values, [b, c, size, size]), device)
        };
        (tensor(x, channels), tensor(target, 1), tensor(mask, 1))
    }

    /// The spots whose hill reaches into the patch at (`x0`, `y0`) - also
    /// those just outside it.
    fn spots_near(&self, x0: usize, y0: usize, size: usize) -> impl Iterator<Item = &Spot> {
        let reach = (3.0 * self.sigma).ceil() + 1.0;
        let (left, top) = (x0 as f32 - reach, y0 as f32 - reach);
        let (right, bottom) = ((x0 + size) as f32 + reach, (y0 + size) as f32 + reach);
        self.spots
            .iter()
            .filter(move |s| (left..right).contains(&s.x) && (top..bottom).contains(&s.y))
    }
}

/// The `size x size` square at (`x0`, `y0`) of every channel of `data`
/// (`channels` planes of `width x height`, one after the other):
/// `channels * size * size` values. The square must lie inside the image.
fn cut_patch(
    data: &[f32],
    width: usize,
    height: usize,
    channels: usize,
    x0: usize,
    y0: usize,
    size: usize,
) -> Vec<f32> {
    let mut out = Vec::with_capacity(channels * size * size);
    for c in 0..channels {
        let plane = &data[c * width * height..(c + 1) * width * height];
        for y in y0..y0 + size {
            out.extend_from_slice(&plane[y * width + x0..y * width + x0 + size]);
        }
    }
    out
}

/// The top-left corner of a `size x size` patch centred on a random pixel of
/// `area_pixels`, moved inside the image where it would stick out.
fn patch_origin(
    area_pixels: &[u32],
    width: usize,
    height: usize,
    size: usize,
    random: &mut SeededUniform,
) -> (usize, usize) {
    let pixel = area_pixels[random.index(area_pixels.len())] as usize;
    let (cx, cy) = (pixel % width, pixel / width);
    let x0 = cx.saturating_sub(size / 2).min(width - size);
    let y0 = cy.saturating_sub(size / 2).min(height - size);
    (x0, y0)
}

/// How much more a pixel on a spot centre counts than background in the
/// loss (`weight = 1 + SPOT_WEIGHT * target`): almost every pixel is
/// background, so without it "never a spot" is nearly right - and the
/// network learns exactly that.
pub(crate) const SPOT_WEIGHT: f32 = 10.0;

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
    use crate::model::spotnet::network::SpotNetConfig;
    use crate::model::spotnet::target::gaussian_map;
    use crate::model::spotnet::train::Spot;

    use super::{
        Patch, Plan, Sample, SpotLabels, SpotNetTrainParams, SpotNetTraining, ValidationOutcome,
        cut_patch, masked_loss, patch_origin, plan, train,
    };
    use crate::NnError;
    use crate::augment::{AugmentParams, Augmentation};
    use crate::model::{Dataset, OwnedImage};
    use crate::train::{SeededUniform, TrainControl, TrainDevice, Validation};
    use burn::Tensor;
    use burn::tensor::{Device, TensorData};
    use std::sync::atomic::{AtomicBool, Ordering};

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

    /// Training images held in memory.
    struct Images(Vec<(OwnedImage, SpotLabels)>);

    impl Dataset<SpotLabels> for Images {
        fn len(&self) -> usize {
            self.0.len()
        }
        fn get(&self, index: usize) -> Result<(OwnedImage, SpotLabels), NnError> {
            Ok(self.0[index].clone())
        }
    }

    /// One `size x size` test image with `spots`, all of it a training area.
    fn one_image(size: usize, spots: Vec<Spot>) -> Images {
        let image = OwnedImage {
            width: size,
            height: size,
            channels: 1,
            data: test_image(size, size, &spots),
        };
        let labels = SpotLabels {
            width: size,
            height: size,
            spots,
            area: vec![true; size * size],
            validation: vec![false; size * size],
        };
        Images(vec![(image, labels)])
    }

    fn params(epochs: usize) -> SpotNetTrainParams {
        SpotNetTrainParams {
            epochs,
            patches_per_epoch: 1,
            batch_size: 1,
            patch_size: 128,
            learning_rate: 1e-2,
            sigma: 1.5,
            augment: AugmentParams::none(),
            validation: Validation::Off,
        }
    }

    fn small_config() -> SpotNetConfig {
        SpotNetConfig {
            in_channels: 1,
            channels: 8,
            ..SpotNetConfig::default()
        }
    }

    /// `f` with a quiet control of `seed`; `cancel` set from the start.
    fn with_control<R>(seed: u64, cancel: bool, f: impl FnOnce(&TrainControl<'_>) -> R) -> R {
        let progress = |_| {};
        let cancel = AtomicBool::new(cancel);
        f(&TrainControl {
            progress: &progress,
            cancel: &cancel,
            seed,
            device: TrainDevice::Cpu,
            cores: 1,
        })
    }

    #[test]
    /// The "overfit one image" check: can SpotNet learn a single image by
    /// heart through `train()`? If not, the network, the loss or the loop is
    /// broken. Kept small for the debug build (a few seconds): 32 x 32, 8
    /// channels, a learning rate of 1e-2, one patch (the whole image) per
    /// epoch - so an epoch is one step.
    fn spotnet_learns_one_image() {
        const SIZE: usize = 32;
        let spots = vec![
            Spot { x: 8.5, y: 8.5 },
            Spot { x: 22.5, y: 12.5 },
            Spot { x: 14.5, y: 24.5 },
            Spot { x: 1.5, y: 18.5 }, // near the left edge
        ];
        let data = one_image(SIZE, spots.clone());
        let SpotNetTraining { model, stats, .. } = with_control(7, false, |control| {
            train(&data, &small_config(), &params(150), control)
        })
        .unwrap();
        println!("{stats:?}");

        let device = Device::flex();
        let x = Tensor::<4>::from_data(
            TensorData::new(data.0[0].0.data.clone(), [1, 1, SIZE, SIZE]),
            &device,
        );
        let prediction = model.forward(x).into_data().try_to_vec::<f32>().unwrap();
        // Right = above 0.5 on every spot centre, below 0.1 wherever the
        // target is 0 (beyond every hill).
        for s in &spots {
            let p = prediction[s.y as usize * SIZE + s.x as usize];
            assert!(p > 0.5, "no hill at spot ({}, {}): {p}", s.x, s.y);
        }
        let target = gaussian_map(SIZE, SIZE, &spots, 1.5);
        let brightest_background = target
            .iter()
            .zip(&prediction)
            .filter(|&(&t, _)| t == 0.0)
            .map(|(_, &p)| p)
            .fold(0.0f32, f32::max);
        assert!(
            brightest_background < 0.1,
            "background not dark: {brightest_background}"
        );
    }

    #[test]
    fn an_augmented_patch_keeps_its_spots_on_its_blobs() {
        // The whole 24 x 24 image as one patch, rotated and flipped at
        // random: every hill top of the target (1.0) lies on a blob centre of
        // the image (0.9 = background 0.1 + 0.8), one per spot.
        let spots = vec![
            Spot { x: 4.5, y: 6.5 },
            Spot { x: 17.5, y: 3.5 },
            Spot { x: 12.5, y: 19.5 },
        ];
        let data = one_image(24, spots.clone());
        let (image, labels) = &data.0[0];
        let sample = Sample::new(image, labels, 1.5, false);
        let params = AugmentParams {
            brightness: (1.0, 1.0),
            ..AugmentParams::default()
        };
        let mut random = SeededUniform::new(2);
        for _ in 0..30 {
            let patch = Patch {
                origin: (0, 0),
                augmentation: Augmentation::random(&params, 1, &mut random),
            };
            let (x, target, _) = sample.batch(&[patch], 24, &Device::flex());
            let x = x.into_data().try_to_vec::<f32>().unwrap();
            let target = target.into_data().try_to_vec::<f32>().unwrap();
            let tops: Vec<usize> = (0..target.len()).filter(|&i| target[i] == 1.0).collect();
            assert_eq!(tops.len(), spots.len(), "{tops:?}");
            for i in tops {
                assert!(x[i] > 0.89, "hill top {i} on {}, not on a blob", x[i]);
            }
        }
    }

    #[test]
    fn a_spot_just_outside_a_patch_still_reaches_into_it() {
        // Spot 2 px left of the 8 x 8 patch at (8, 8): its hill reaches the
        // patch's left column (exp(-4 / 4.5) = 0.41 at distance 2).
        let data = one_image(24, vec![Spot { x: 6.5, y: 10.5 }]);
        let (image, labels) = &data.0[0];
        let sample = Sample::new(image, labels, 1.5, false);
        let patch = Patch {
            origin: (8, 8),
            augmentation: Augmentation::random(
                &AugmentParams::none(),
                1,
                &mut SeededUniform::new(1),
            ),
        };
        let (_, target, _) = sample.batch(&[patch], 8, &Device::flex());
        let target = target.into_data().try_to_vec::<f32>().unwrap();
        assert!((target[2 * 8] - 0.41).abs() < 0.01, "{}", target[2 * 8]);
    }

    #[test]
    fn the_same_seed_trains_the_same_model() {
        // With every augmentation on: the changes come from the seed too.
        let data = one_image(16, vec![Spot { x: 8.5, y: 8.5 }]);
        let augmented = SpotNetTrainParams {
            augment: AugmentParams::default(),
            ..params(3)
        };
        let run = |seed| {
            with_control(seed, false, |control| {
                train(&data, &small_config(), &augmented, control)
            })
            .unwrap()
            .stats
        };
        assert_eq!(run(3), run(3));
        assert_ne!(run(3), run(4));
    }

    #[test]
    fn a_training_without_any_training_area_is_refused() {
        let mut data = one_image(16, vec![Spot { x: 8.5, y: 8.5 }]);
        data.0[0].1.area = vec![false; 16 * 16];
        let result = with_control(1, false, |c| train(&data, &small_config(), &params(1), c));
        assert!(matches!(result, Err(NnError::InvalidInput(m)) if m.contains("no training area")));
    }

    #[test]
    fn an_image_with_other_channels_than_the_network_is_refused() {
        let data = one_image(16, vec![]);
        let config = SpotNetConfig::default(); // 2 channels, the image has 1
        let result = with_control(1, false, |c| train(&data, &config, &params(1), c));
        assert!(matches!(result, Err(NnError::InvalidInput(m)) if m.contains("1 channels")));
    }

    #[test]
    fn a_training_area_of_the_wrong_size_is_refused() {
        let mut data = one_image(16, vec![]);
        data.0[0].1.area.pop();
        let result = with_control(1, false, |c| train(&data, &small_config(), &params(1), c));
        assert!(matches!(result, Err(NnError::InvalidInput(m)) if m.contains("training area")));
    }

    #[test]
    fn a_cancelled_training_stops() {
        let data = one_image(16, vec![Spot { x: 8.5, y: 8.5 }]);
        let result = with_control(1, true, |c| train(&data, &small_config(), &params(5), c));
        assert!(matches!(result, Err(NnError::Cancelled)));
    }

    #[test]
    fn progress_is_reported_once_per_epoch() {
        let data = one_image(16, vec![Spot { x: 8.5, y: 8.5 }]);
        let epochs = std::sync::Mutex::new(Vec::new());
        let progress = |p| {
            if let crate::train::TrainProgress::Epoch { epoch, .. } = p {
                epochs.lock().unwrap().push(epoch);
            }
        };
        let cancel = AtomicBool::new(false);
        let control = TrainControl {
            progress: &progress,
            cancel: &cancel,
            seed: 1,
            device: TrainDevice::Cpu,
            cores: 1,
        };
        train(&data, &small_config(), &params(3), &control).unwrap();
        assert_eq!(*epochs.lock().unwrap(), vec![0, 1, 2]);
        assert!(!cancel.load(Ordering::Relaxed));
    }

    #[test]
    fn a_patch_is_the_square_of_every_channel() {
        // 4 x 4, two channels, each value its own index: 0..16 and 16..32.
        let data: Vec<f32> = (0..32).map(|v| v as f32).collect();
        let patch = cut_patch(&data, 4, 4, 2, 1, 2, 2);
        assert_eq!(patch, vec![9.0, 10.0, 13.0, 14.0, 25.0, 26.0, 29.0, 30.0]);
    }

    #[test]
    fn a_patch_lies_inside_the_image_and_over_the_training_area() {
        // 40 x 20, the training area only in the bottom-right corner.
        let (width, height, size) = (40, 20, 8);
        let area: Vec<u32> = (0..(width * height) as u32)
            .filter(|&i| i as usize % width >= 36 && i as usize / width >= 17)
            .collect();
        let mut random = SeededUniform::new(5);
        for _ in 0..200 {
            let (x0, y0) = patch_origin(&area, width, height, size, &mut random);
            assert!(x0 + size <= width && y0 + size <= height, "({x0}, {y0})");
            let covers = area.iter().any(|&i| {
                let (x, y) = (i as usize % width, i as usize / width);
                (x0..x0 + size).contains(&x) && (y0..y0 + size).contains(&y)
            });
            assert!(covers, "({x0}, {y0}) misses the area");
        }
    }

    /// `n` copies of a 16 x 16 image with one spot, all of it a training area.
    fn images(n: usize) -> Images {
        let one = one_image(16, vec![Spot { x: 8.5, y: 8.5 }]);
        Images(vec![one.0[0].clone(); n])
    }

    fn split(data: &Images, validation: Validation, seed: u64) -> Result<Plan, NnError> {
        plan(data, &small_config(), validation, seed)
    }

    #[test]
    fn auto_holds_back_a_fifth_of_the_images_chosen_by_the_seed() {
        let data = images(10);
        let plan = split(&data, Validation::default(), 3).unwrap();
        assert_eq!(plan.validate.len(), 2);
        assert_eq!(plan.train.len(), 8);
        assert!(
            plan.validate.iter().all(|i| !plan.train.contains(i)),
            "{plan:?}"
        );
        assert!(!plan.drawn_areas);
        assert_eq!(plan, split(&data, Validation::default(), 3).unwrap());
        assert_ne!(
            plan.validate,
            split(&data, Validation::default(), 4).unwrap().validate
        );
    }

    #[test]
    fn auto_holds_back_at_least_one_image_and_never_all() {
        let tiny = split(&images(5), Validation::Auto { fraction: 0.01 }, 1).unwrap();
        assert_eq!((tiny.validate.len(), tiny.train.len()), (1, 4));
        let huge = split(&images(5), Validation::Auto { fraction: 0.99 }, 1).unwrap();
        assert_eq!((huge.validate.len(), huge.train.len()), (4, 1));
    }

    #[test]
    fn auto_with_too_few_images_trains_on_all_and_says_so() {
        let plan = split(&images(4), Validation::default(), 1).unwrap();
        assert_eq!(plan.train.len(), 4);
        assert!(plan.validate.is_empty());
        assert_eq!(plan.outcome, ValidationOutcome::TooFewImages { images: 4 });
    }

    #[test]
    fn drawn_validation_areas_win_over_the_image_split() {
        let mut data = images(6);
        data.0[2].1.validation[0] = true;
        for validation in [Validation::default(), Validation::Areas] {
            let plan = split(&data, validation, 1).unwrap();
            assert!(plan.drawn_areas);
            assert_eq!(plan.validate, vec![2]);
            assert_eq!(
                plan.train.len(),
                6,
                "the rest of image 2 is still trained on"
            );
        }
    }

    #[test]
    fn areas_without_any_drawn_are_refused() {
        let result = split(&images(6), Validation::Areas, 1);
        assert!(
            matches!(result, Err(NnError::InvalidInput(m)) if m.contains("no validation area"))
        );
    }

    #[test]
    fn off_checks_nothing() {
        let mut data = images(6);
        data.0[2].1.validation[0] = true;
        let plan = split(&data, Validation::Off, 1).unwrap();
        assert!(plan.validate.is_empty());
        assert_eq!(plan.outcome, ValidationOutcome::Off);
    }

    #[test]
    fn a_drawn_validation_area_is_never_trained_on() {
        // The left half validation, the right half training: no training
        // pixel - and so no patch centre - lies in the left half.
        let mut data = one_image(16, vec![]);
        let labels = &mut data.0[0].1;
        for i in 0..16 * 16 {
            labels.validation[i] = i % 16 < 8;
        }
        let (image, labels) = &data.0[0];
        let sample = Sample::new(image, labels, 1.5, true);
        assert_eq!(sample.area_pixels.len(), 16 * 8);
        assert!(sample.area_pixels.iter().all(|&i| i % 16 >= 8));
    }

    #[test]
    fn a_trained_model_is_checked_on_its_validation_area() {
        // One 32 x 32 image: the top half training (4 spots), the bottom
        // half validation (2 spots), augmentation on. The model never sees
        // the bottom spots' labels - finding them shows it learned spots,
        // not places.
        const SIZE: usize = 32;
        let spots = vec![
            Spot { x: 5.5, y: 4.5 },
            Spot { x: 14.5, y: 10.5 },
            Spot { x: 21.5, y: 4.5 },
            Spot { x: 28.5, y: 11.5 },
            Spot { x: 10.5, y: 23.5 },
            Spot { x: 23.5, y: 25.5 },
        ];
        let mut data = one_image(SIZE, spots);
        let labels = &mut data.0[0].1;
        for i in 0..SIZE * SIZE {
            labels.validation[i] = i / SIZE >= 16;
        }
        let training = SpotNetTrainParams {
            augment: AugmentParams::default(),
            validation: Validation::Areas,
            ..params(150)
        };
        let result =
            with_control(7, false, |c| train(&data, &small_config(), &training, c)).unwrap();
        let ValidationOutcome::Scored { epoch, score } = result.validation else {
            panic!("not validated: {:?}", result.validation)
        };
        println!("best epoch {epoch}: {score:?}, f1 {}", score.f1());
        assert_eq!(score.annotated, 2);
        assert_eq!(score.f1(), 1.0, "{score:?}");
        let crate::TrainStats::Epochs {
            best_val_epoch,
            best_val_loss,
            ..
        } = result.stats
        else {
            panic!()
        };
        assert_eq!(best_val_epoch, Some(epoch));
        assert_eq!(best_val_loss, Some(score.loss));
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
