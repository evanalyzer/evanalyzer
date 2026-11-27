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

use super::validate::Tally;
use crate::{
    NnError, TrainControl, TrainStats,
    augment::Augmentation,
    model::{
        Dataset, OwnedImage,
        spotnet::{
            file::{SpotNetModel, SpotNetSettings},
            network::{SpotNet, SpotNetConfig},
            normalize::PercentileNormalization,
            target::{area_mask, gaussian_map},
        },
    },
    pieces::Pieces,
    train::{
        HeldBack, HeldBackScore, MIN_IMAGES_FOR_AUTO_SPLIT, NetworkTrainParams, SeededUniform,
        TrainProgress, Trained, Validation,
    },
};
use burn::Tensor;
use burn::module::Module;
use burn::optim::{AdamConfig, GradientsParams};
use burn::tensor::activation::log_sigmoid;
use burn::tensor::{Device, TensorData};
use std::collections::BTreeMap;

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

/// Settings of one SpotNet training: every network's plus SpotNet's own.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct SpotNetTrainParams {
    pub common: NetworkTrainParams,
    /// The architecture of a new model (a fine-tuned one keeps its file's).
    pub network: SpotNetConfig,
    /// Width of the Gaussian hills in px, e.g. 1.5 (a fine-tuned model keeps
    /// its file's).
    pub sigma: f32,
    /// How every input channel is brought to 0..1 - stored in the model
    /// file, prediction normalises the same way (a fine-tuned model keeps
    /// its file's).
    pub normalization: PercentileNormalization,
}

impl SpotNetTrainParams {
    /// A new SpotNet of `in_channels` (1: the spot channel, 2: plus a
    /// reference channel) trained with `common`, everything else SpotNet's
    /// own defaults: the default network (a ~20 px view for spots of 4-10
    /// px), hills of 1.5 px, each channel's 1st / 99.8th percentile as 0 / 1.
    pub fn new(common: NetworkTrainParams, in_channels: usize) -> Self {
        Self {
            common,
            network: SpotNetConfig {
                in_channels,
                ..SpotNetConfig::default()
            },
            sigma: 1.5,
            normalization: PercentileNormalization::default(),
        }
    }
}

/// Trains a new SpotNet on `data`, on the CPU (the GPU follows later).
///
/// Image by image - only one image is in RAM at a time: per epoch every
/// image with pixels to train on gives its share of `patches_per_epoch`
/// random patches (each centred on such a pixel), trained in batches of
/// `batch_size`. Images smaller than `patch_size` give patches of the whole
/// image. Every patch gets its own random changes (`params.common.augment`); its
/// target is computed from its moved spots.
///
/// After every epoch the model is checked on the held-back areas
/// (`params.common.validation`, see [`plan`]) and the best epoch's model kept
/// (see [`better`]); `stats.best_val_loss` is that epoch's loss.
/// Start weights, patches, changes and the held-back images come from
/// `control.seed`: the same data and seed give the same model.
///
/// `start`: `None` = new weights (`params.network`), `Some` = fine-tune that
/// model - its architecture, target width and normalisation stay.
pub fn train(
    start: Option<&SpotNetModel>,
    data: &dyn Dataset<SpotLabels>,
    params: &SpotNetTrainParams,
    control: &TrainControl<'_>,
) -> Result<Trained<SpotNetModel>, NnError> {
    check_params(params)?;
    let settings = match start {
        Some(start) => start.settings.clone(),
        None => SpotNetSettings {
            network: params.network.clone(),
            normalization: params.normalization,
            sigma: params.sigma,
        },
    };
    let config = &settings.network;
    config
        .check()
        .map_err(|e| NnError::InvalidInput(e.to_string()))?;
    let common = &params.common;
    let plan = plan(data, config, common.validation, control.seed)?;
    let patches_per_image = common.patches_per_epoch.div_ceil(plan.train.len());
    let margin = config.receptive_field() / 2;

    let device = Device::flex().autodiff();
    let mut model = match start {
        // The file's weights onto the training device (with gradients).
        Some(start) => SpotNetModel::from_bytes(&start.to_bytes()?, &device)?.network,
        None => SpotNet::new(config, control.seed, &device),
    };
    let mut optim = AdamConfig::new().init();
    // Its own stream (seed + 1): the patches don't depend on how many
    // random numbers the start weights took.
    let mut random = SeededUniform::new(control.seed.wrapping_add(1));

    let mut final_train_loss = f32::NAN;
    let mut final_val_loss = None;
    // Validation pieces: a training patch plus its halo on every side.
    let pieces = Pieces::fixed(common.patch_size + 2 * margin);
    let mut best: Option<(usize, HeldBackScore, SpotNetModel)> = None;
    let mut epochs_run = 0;
    for epoch in 0..common.epochs {
        let (mut loss_sum, mut batches) = (0.0f32, 0usize);
        for &index in &plan.train {
            if control.is_cancelled() {
                return Err(NnError::Cancelled);
            }
            let (mut image, labels) = data.get(index)?;
            settings.normalization.apply(&mut image);
            let sample = Sample::new(&image, &labels, settings.sigma, plan.drawn_areas);
            let size = common.patch_size.min(image.width).min(image.height);
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
                        &common.augment,
                        image.channels,
                        &mut random,
                    ),
                })
                .collect();
            for batch in patches.chunks(common.batch_size) {
                let (x, target, mask) = sample.batch(batch, size, &device);
                let loss = masked_loss(model.forward_logits(x), target, mask);
                loss_sum += loss.clone().into_scalar::<f32>();
                batches += 1;
                let grads = GradientsParams::from_grads(loss.backward(), &model);
                model = optim.step(common.learning_rate, model, grads);
            }
        }
        final_train_loss = loss_sum / batches.max(1) as f32;
        epochs_run = epoch + 1;
        let mut epoch_score = None;

        if !plan.validate.is_empty() {
            let checked = SpotNetModel {
                network: model.valid(),
                settings: settings.clone(),
                info: BTreeMap::new(),
                device: Device::flex(),
            };
            let mut tally = Tally::default();
            for &index in &plan.validate {
                if control.is_cancelled() {
                    return Err(NnError::Cancelled);
                }
                let (mut image, labels) = data.get(index)?;
                settings.normalization.apply(&mut image);
                let held_back = if plan.drawn_areas {
                    &labels.validation
                } else {
                    &labels.area
                };
                tally.add(&checked, &image, &labels.spots, held_back, pieces)?;
            }
            if let Some(score) = tally.score() {
                final_val_loss = Some(score.loss);
                epoch_score = Some(score);
                if best.as_ref().is_none_or(|(_, b, _)| better(&score, b)) {
                    best = Some((epoch, score, checked));
                }
            }
        }
        (control.progress)(TrainProgress::Epoch {
            epoch,
            total_epochs: common.epochs,
            train_loss: final_train_loss,
            val_loss: final_val_loss,
            held_back: epoch_score,
        });
        let best_so_far = best.as_ref().map(|(e, score, _)| (*e, score.f1()));
        if stop_early(common.patience, epoch, best_so_far) {
            break;
        }
    }

    let stats = TrainStats::Epochs {
        epochs_run,
        total_epochs: common.epochs,
        final_train_loss,
        final_val_loss,
        best_val_loss: best.as_ref().map(|(_, score, _)| score.loss),
        best_val_epoch: best.as_ref().map(|(epoch, _, _)| *epoch),
    };
    let (mut model, validation) = match best {
        Some((epoch, score, model)) => (model, HeldBack::Scored { epoch, score }),
        None => (
            SpotNetModel {
                network: model.valid(),
                settings,
                info: BTreeMap::new(),
                device: Device::flex(),
            },
            plan.outcome,
        ),
    };
    model.info = BTreeMap::from([
        ("seed".into(), control.seed.to_string()),
        (
            "training".into(),
            serde_json::to_string(params).expect("params serialize"),
        ),
        ("epochs_run".into(), epochs_run.to_string()),
    ]);
    Ok(Trained {
        model,
        stats,
        held_back: validation,
    })
}

/// Whether `score` beats `best`: more spots found right (F1), on a tie the
/// lower loss. Not the loss alone - a model still unsure everywhere has a
/// lower loss than one that is sure and slightly off, while it finds
/// nothing yet.
fn better(score: &HeldBackScore, best: &HeldBackScore) -> bool {
    let (f1, best_f1) = (score.f1(), best.f1());
    f1 > best_f1 || (f1 == best_f1 && score.loss < best.loss)
}

/// Whether to stop after `epoch`: `patience` epochs have passed since the
/// best check (`best`: its epoch and F1) without a better one.
///
/// Counted only once the best model finds spots (F1 above 0): before that
/// the loss of the held-back areas may rise for many epochs while the
/// network forms its first filters - stopping there would end a training
/// right before it starts to find anything.
fn stop_early(patience: Option<usize>, epoch: usize, best: Option<(usize, f32)>) -> bool {
    match (patience, best) {
        (Some(patience), Some((best_epoch, f1))) if f1 > 0.0 => epoch - best_epoch >= patience,
        _ => false,
    }
}

/// Refuses settings no training can run with.
fn check_params(params: &SpotNetTrainParams) -> Result<(), NnError> {
    params.common.check()?;
    params.normalization.check()?;
    if !(params.sigma > 0.0) {
        return Err(NnError::InvalidInput(
            "SpotNet training: sigma must be above 0".into(),
        ));
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
    outcome: HeldBack,
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
            outcome: HeldBack::Off,
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
            outcome: HeldBack::Off,
        });
    };
    if train.len() < MIN_IMAGES_FOR_AUTO_SPLIT {
        return Ok(Plan {
            outcome: HeldBack::TooFewImages {
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
        outcome: HeldBack::Off,
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
    use crate::model::spotnet::normalize::PercentileNormalization;
    use crate::model::spotnet::target::gaussian_map;
    use crate::model::spotnet::train::Spot;

    use super::{
        Patch, Plan, Sample, SpotLabels, SpotNetTrainParams, cut_patch, masked_loss, patch_origin,
        plan, stop_early, train,
    };
    use crate::NnError;
    use crate::augment::{AugmentParams, Augmentation};
    use crate::model::spotnet::file::SpotNetModel;
    use crate::model::{Dataset, OwnedImage};
    use crate::train::{
        HeldBack, NetworkTrainParams, SeededUniform, TrainControl, TrainDevice, Trained, Validation,
    };
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
            common: NetworkTrainParams {
                epochs,
                patches_per_epoch: 1,
                batch_size: 1,
                patch_size: 128,
                learning_rate: 1e-2,
                augment: AugmentParams::none(),
                validation: Validation::Off,
                patience: None,
            },
            network: small_config(),
            sigma: 1.5,
            normalization: PercentileNormalization::default(),
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
        let Trained { model, stats, .. } = with_control(7, false, |control| {
            train(None, &data, &params(150), control)
        })
        .unwrap();
        println!("{stats:?}");

        let device = Device::flex();
        // The input as the training saw it: normalised.
        let mut input = data.0[0].0.clone();
        model.settings.normalization.apply(&mut input);
        let x = Tensor::<4>::from_data(TensorData::new(input.data, [1, 1, SIZE, SIZE]), &device);
        let prediction = model
            .network
            .forward(x)
            .into_data()
            .try_to_vec::<f32>()
            .unwrap();
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
        let mut augmented = params(3);
        augmented.common.augment = AugmentParams::default();
        let run = |seed| {
            with_control(seed, false, |control| {
                train(None, &data, &augmented, control)
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
        let result = with_control(1, false, |c| train(None, &data, &params(1), c));
        assert!(matches!(result, Err(NnError::InvalidInput(m)) if m.contains("no training area")));
    }

    #[test]
    fn an_image_with_other_channels_than_the_network_is_refused() {
        let data = one_image(16, vec![]);
        let mut training = params(1);
        training.network = SpotNetConfig::default(); // 2 channels, the image has 1
        let result = with_control(1, false, |c| train(None, &data, &training, c));
        assert!(matches!(result, Err(NnError::InvalidInput(m)) if m.contains("1 channels")));
    }

    #[test]
    fn a_training_area_of_the_wrong_size_is_refused() {
        let mut data = one_image(16, vec![]);
        data.0[0].1.area.pop();
        let result = with_control(1, false, |c| train(None, &data, &params(1), c));
        assert!(matches!(result, Err(NnError::InvalidInput(m)) if m.contains("training area")));
    }

    #[test]
    fn a_cancelled_training_stops() {
        let data = one_image(16, vec![Spot { x: 8.5, y: 8.5 }]);
        let result = with_control(1, true, |c| train(None, &data, &params(5), c));
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
        train(None, &data, &params(3), &control).unwrap();
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
        assert_eq!(plan.outcome, HeldBack::TooFewImages { images: 4 });
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
        assert_eq!(plan.outcome, HeldBack::Off);
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

    /// One 32 x 32 image: the top half training (4 spots), the bottom half
    /// validation (2 spots), augmentation on - trained up to 150 epochs.
    fn train_on_half_with_validation(patience: Option<usize>) -> Trained<SpotNetModel> {
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
        let mut training = params(150);
        training.common.augment = AugmentParams::default();
        training.common.validation = Validation::Areas;
        training.common.patience = patience;
        with_control(7, false, |c| train(None, &data, &training, c)).unwrap()
    }

    #[test]
    fn a_trained_model_is_checked_on_its_validation_area() {
        // The model never sees the bottom spots' labels - finding them shows
        // it learned spots, not places.
        let result = train_on_half_with_validation(None);
        let HeldBack::Scored { epoch, score } = result.held_back else {
            panic!("not validated: {:?}", result.held_back)
        };
        println!("best epoch {epoch}: {score:?}, f1 {}", score.f1());
        assert_eq!(score.annotated, 2);
        assert_eq!(score.f1(), 1.0, "{score:?}");
        let crate::TrainStats::Epochs {
            best_val_epoch,
            best_val_loss,
            epochs_run,
            ..
        } = result.stats
        else {
            panic!()
        };
        assert_eq!(epochs_run, 150);
        assert_eq!(best_val_epoch, Some(epoch));
        assert_eq!(best_val_loss, Some(score.loss));
    }

    #[test]
    fn new_params_are_spotnets_defaults_for_the_channels() {
        let common = params(1).common;
        let p = SpotNetTrainParams::new(common.clone(), 2);
        assert_eq!(
            p.network,
            SpotNetConfig {
                in_channels: 2,
                ..SpotNetConfig::default()
            }
        );
        assert_eq!(
            (p.sigma, p.normalization),
            (1.5, PercentileNormalization::default())
        );
        assert_eq!(p.common, common);
    }

    #[test]
    fn every_epoch_reports_its_held_back_score() {
        let data = one_image(32, vec![Spot { x: 8.5, y: 6.5 }, Spot { x: 12.5, y: 24.5 }]);
        let mut data = data;
        for i in 0..32 * 32 {
            data.0[0].1.validation[i] = i / 32 >= 16;
        }
        let mut training = params(3);
        training.common.validation = Validation::Areas;
        let scores = std::sync::Mutex::new(Vec::new());
        let progress = |p| {
            if let crate::train::TrainProgress::Epoch { held_back, .. } = p {
                scores.lock().unwrap().push(held_back);
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
        train(None, &data, &training, &control).unwrap();
        let scores = scores.into_inner().unwrap();
        assert_eq!(scores.len(), 3);
        assert!(
            scores.iter().all(|s| s.is_some_and(|s| s.annotated == 1)),
            "{scores:?}"
        );
    }

    #[test]
    fn a_training_stops_after_patience_epochs_without_a_better_check() {
        let result = train_on_half_with_validation(Some(10));
        let HeldBack::Scored { epoch, score } = result.held_back else {
            panic!("not validated: {:?}", result.held_back)
        };
        let crate::TrainStats::Epochs { epochs_run, .. } = result.stats else {
            panic!()
        };
        println!(
            "stopped after {epochs_run} epochs, best {epoch}: f1 {}",
            score.f1()
        );
        assert!(score.f1() > 0.0);
        assert!(epochs_run < 150, "never stopped");
        assert_eq!(epochs_run, epoch + 10 + 1, "10 epochs after the best one");
    }

    #[test]
    fn early_stopping_waits_for_patience_epochs_after_the_best() {
        // Best at epoch 5 finding spots: stop at 15 with a patience of 10.
        assert!(!stop_early(Some(10), 14, Some((5, 0.8))));
        assert!(stop_early(Some(10), 15, Some((5, 0.8))));
    }

    #[test]
    fn early_stopping_never_stops_before_anything_is_found_or_without_patience() {
        // F1 0: the network may still be forming its filters.
        assert!(!stop_early(Some(10), 90, Some((5, 0.0))));
        assert!(!stop_early(None, 90, Some((5, 0.8))));
        assert!(!stop_early(Some(10), 90, None));
    }

    #[test]
    fn a_trained_model_saved_and_loaded_predicts_the_same_and_knows_its_training() {
        let data = one_image(16, vec![Spot { x: 8.5, y: 8.5 }]);
        let trained = with_control(9, false, |c| train(None, &data, &params(3), c))
            .unwrap()
            .model;
        let dir = tempfile::TempDir::new().unwrap();
        let path = dir.path().join("spots.evanet");
        trained.save(&path).unwrap();
        let loaded =
            crate::model::spotnet::file::SpotNetModel::read(&path, &Device::flex()).unwrap();

        assert_eq!(loaded.settings, trained.settings);
        assert_eq!(loaded.info["seed"], "9");
        assert!(
            loaded.info["training"].contains("\"epochs\":3"),
            "{}",
            loaded.info["training"]
        );
        let predict = |m: &crate::model::spotnet::file::SpotNetModel| {
            let mut input = data.0[0].0.clone();
            m.settings.normalization.apply(&mut input);
            let x = Tensor::<4>::from_data(
                TensorData::new(input.data, [1, 1, 16, 16]),
                &Device::flex(),
            );
            m.network
                .forward(x)
                .into_data()
                .try_to_vec::<f32>()
                .unwrap()
        };
        assert_eq!(predict(&loaded), predict(&trained));
    }

    #[test]
    fn a_patience_of_zero_is_refused() {
        let data = one_image(16, vec![Spot { x: 8.5, y: 8.5 }]);
        let mut training = params(1);
        training.common.patience = Some(0);
        let result = with_control(1, false, |c| train(None, &data, &training, c));
        assert!(matches!(result, Err(NnError::InvalidInput(m)) if m.contains("patience")));
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
