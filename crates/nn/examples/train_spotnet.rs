//! # train_spotnet
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-11-27
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.
//!
//! Dev tool until the app's training job exists: trains a SpotNet on
//! generated images (a noisy background, bright blobs of 4-10 px at random
//! places, some touching) and writes the model file - to try the
//! "AI SpotNet Spot Detection" step in the app.
//!
//!   cargo build-linux -p evanalyzer_nn --example train_spotnet
//!   target/x86_64-unknown-linux-gnu/release/examples/train_spotnet <out.evanet> [epochs] [seed]
//!
//! Spot channel only (1 input channel); 6 images of 256 x 256, a fifth held
//! back for validation, augmentation on, early stop after 20 epochs without
//! a better check. Prints the loss and the held-back score per epoch.

use evanalyzer_nn::NnError;
use evanalyzer_nn::augment::AugmentParams;
use evanalyzer_nn::model::spotnet::{
    PercentileNormalization, Spot, SpotLabels, SpotNetConfig, SpotNetTrainParams, train,
};
use evanalyzer_nn::model::{Dataset, OwnedImage};
use evanalyzer_nn::train::{
    HeldBack, NetworkTrainParams, TrainControl, TrainDevice, TrainProgress, Validation,
};
use std::sync::atomic::AtomicBool;

const SIDE: usize = 256;
const IMAGES: usize = 6;

/// Uniform values in `[0, 1)` from a seed (xorshift64).
struct Random(u64);

impl Random {
    fn unit(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 40) as f32 / (1u64 << 24) as f32
    }
}

/// Generated images with their spots, all of each a training area.
struct Generated(Vec<(OwnedImage, SpotLabels)>);

impl Dataset<SpotLabels> for Generated {
    fn len(&self) -> usize {
        self.0.len()
    }
    fn get(&self, index: usize) -> Result<(OwnedImage, SpotLabels), NnError> {
        Ok(self.0[index].clone())
    }
}

/// One image: background ~100 with noise, 40 spots of 4-10 px and random
/// brightness, every fifth one touching the one before (16-bit-like values).
fn image(random: &mut Random) -> (OwnedImage, SpotLabels) {
    let mut spots: Vec<(Spot, f32, f32)> = Vec::new();
    for i in 0..40 {
        let (x, y) = if i % 5 == 4 {
            let (prev, _, _) = spots[i - 1];
            (prev.x + 5.0, prev.y)
        } else {
            (
                8.0 + random.unit() * (SIDE as f32 - 16.0),
                8.0 + random.unit() * (SIDE as f32 - 16.0),
            )
        };
        let diameter = 4.0 + random.unit() * 6.0;
        let brightness = 300.0 + random.unit() * 1200.0;
        spots.push((Spot { x, y }, diameter / 2.5, brightness));
    }
    let mut data = vec![0.0f32; SIDE * SIDE];
    for (i, v) in data.iter_mut().enumerate() {
        let (px, py) = ((i % SIDE) as f32 + 0.5, (i / SIDE) as f32 + 0.5);
        let mut value = 100.0 + (random.unit() - 0.5) * 40.0;
        for (s, sigma, brightness) in &spots {
            let r2 = (px - s.x).powi(2) + (py - s.y).powi(2);
            if r2 < 100.0 {
                value += brightness * (-r2 / (2.0 * sigma * sigma)).exp();
            }
        }
        *v = value;
    }
    let image = OwnedImage {
        width: SIDE,
        height: SIDE,
        channels: 1,
        data,
    };
    let labels = SpotLabels {
        width: SIDE,
        height: SIDE,
        spots: spots.into_iter().map(|(s, _, _)| s).collect(),
        area: vec![true; SIDE * SIDE],
        validation: vec![false; SIDE * SIDE],
    };
    (image, labels)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let Some(out) = args.get(1) else {
        eprintln!("usage: train_spotnet <out.evanet> [epochs] [seed]");
        std::process::exit(2);
    };
    let epochs: usize = args
        .get(2)
        .map_or(200, |e| e.parse().expect("epochs: a number"));
    let seed: u64 = args
        .get(3)
        .map_or(1, |s| s.parse().expect("seed: a number"));

    let mut random = Random(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).max(1));
    let data = Generated((0..IMAGES).map(|_| image(&mut random)).collect());
    let params = SpotNetTrainParams {
        common: NetworkTrainParams {
            epochs,
            patches_per_epoch: 32,
            batch_size: 8,
            patch_size: 64,
            learning_rate: 1e-3,
            augment: AugmentParams::default(),
            validation: Validation::default(),
            patience: Some(20),
        },
        network: SpotNetConfig {
            in_channels: 1,
            ..SpotNetConfig::default()
        },
        sigma: 1.5,
        normalization: PercentileNormalization::default(),
    };
    let progress = |p| {
        if let TrainProgress::Epoch {
            epoch,
            train_loss,
            held_back,
            ..
        } = p
        {
            match held_back {
                Some(s) => println!(
                    "epoch {epoch:4}: loss {train_loss:.4}, held back: found {} of {}, {} wrongly",
                    s.matched,
                    s.annotated,
                    s.wrongly_found()
                ),
                None => println!("epoch {epoch:4}: loss {train_loss:.4}"),
            }
        }
    };
    let cancel = AtomicBool::new(false);
    let control = TrainControl {
        progress: &progress,
        cancel: &cancel,
        seed,
        device: TrainDevice::Cpu,
        cores: std::thread::available_parallelism().map_or(1, |n| n.get()),
    };
    let start = std::time::Instant::now();
    let trained = train(None, &data, &params, &control).expect("training");
    println!(
        "trained in {:.1} s: {:?}",
        start.elapsed().as_secs_f64(),
        trained.stats
    );
    if let HeldBack::Scored { epoch, score } = trained.held_back {
        println!(
            "kept epoch {epoch}: found {} of {} held-back spots, {} wrongly (F1 {:.3})",
            score.matched,
            score.annotated,
            score.wrongly_found(),
            score.f1()
        );
    }
    trained
        .model
        .save(std::path::Path::new(out))
        .expect("writing the model file");
    println!("written {out}");
}
