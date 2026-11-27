//! # ai_devices
//!
//! Which devices AI models (Cellpose-SAM, StarDist, U-Net) run on, set once
//! at startup from the command line (`--ai-devices`, `--gpu-slots`), and
//! [`run`]: a model run on the engine they set up (`evanalyzer_nn`).
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-10-10
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.

use evanalyzer_cfg::core_types::InternalErrors;
use evanalyzer_nn::{DeviceSelection, EngineConfig, Image, Model, Written};
use std::fmt;
use std::str::FromStr;
use std::sync::OnceLock;

/// Parallel inferences per GPU unless set otherwise. One never overflows a
/// GPU's memory (overflowing makes the Windows driver silently page into
/// system RAM, which all but stops inference); GPUs with memory to spare
/// can run more, overlapping one tile's transfers with another's compute.
pub const DEFAULT_GPU_SLOTS: usize = 1;

/// The devices AI inference may use.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum AiDeviceSelection {
    /// Every discrete GPU, or the CPU when there is none.
    #[default]
    Auto,
    /// Never use a GPU.
    Cpu,
    /// These discrete GPUs (numbered from 0 in the order the system lists them); ones that don't exist
    /// are ignored, and with none left the CPU is used.
    Gpus(Vec<usize>),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AiDeviceOptions {
    pub devices: AiDeviceSelection,
    /// Parallel inferences per GPU, at least 1 (see [`parse_gpu_slots`]).
    pub gpu_slots: usize,
    /// Where the GPU keeps its autotune results and compiled kernels across
    /// runs (`<user folder>/cache/gpu`, see [`evanalyzer_nn::configure_gpu`]);
    /// `None`: nothing kept.
    pub gpu_cache: Option<std::path::PathBuf>,
}

impl Default for AiDeviceOptions {
    fn default() -> Self {
        Self {
            devices: AiDeviceSelection::default(),
            gpu_slots: DEFAULT_GPU_SLOTS,
            gpu_cache: None,
        }
    }
}

static OPTIONS: OnceLock<AiDeviceOptions> = OnceLock::new();

/// Sets the AI device options for this process. Call it once at startup,
/// before the first AI step runs; returns `false` (and changes nothing) if
/// options were already set or AI inference already started.
pub fn configure_ai_devices(options: AiDeviceOptions) -> bool {
    // Only a path: nothing is opened or written until a model runs on a GPU.
    if options.devices != AiDeviceSelection::Cpu {
        evanalyzer_nn::configure_gpu(options.gpu_cache.as_deref());
    }
    OPTIONS
        .set(AiDeviceOptions {
            gpu_slots: options.gpu_slots.max(1),
            ..options
        })
        .is_ok()
}

/// The configured options; the defaults once AI inference has started
/// without any being set.
pub(crate) fn ai_device_options() -> &'static AiDeviceOptions {
    OPTIONS.get_or_init(AiDeviceOptions::default)
}

/// Model `M` from `file` on `image` with the engine (`evanalyzer_nn::run`:
/// pre-processing, the network, post-processing into `out`), set up with
/// these options on the first call.
pub(crate) fn run<M: Model>(
    file: &std::path::Path,
    image: Image<'_>,
    params: &M::Params,
    out: &mut [u32],
) -> Result<Written, InternalErrors> {
    evanalyzer_nn::run::<M>(file, image, params, out, engine_config)
        .map_err(|e| run_error::<M>(file, e))
}

/// A failed run of `M` as the user reads it: when the model file is the
/// reason (missing, no model, another kind of model), what it is and which
/// step runs it - like the card at the step's file field; else nn's message.
fn run_error<M: Model>(file: &std::path::Path, error: evanalyzer_nn::NnError) -> InternalErrors {
    use crate::algos::ai_segmentation::model_file::{
        ModelFileNote, ModelKind, model_file_note, read_start,
    };
    use evanalyzer_nn::NnError;
    let about_the_file = matches!(
        error,
        NnError::Io(_) | NnError::UnsupportedArchitecture(_) | NnError::InvalidModelFile(_)
    );
    if about_the_file && let Some(kind) = ModelKind::of_arch(M::ARCH) {
        let start = read_start(file);
        match model_file_note(kind, file, start.as_deref().map_err(Clone::clone)) {
            ModelFileNote::WrongKind(problem) | ModelFileNote::Unusable(problem) => {
                return InternalErrors::Generic(format!("{}: {}", kind.step(), problem.message));
            }
            // The right kind, its contents don't fit: nn's message says what.
            ModelFileNote::Fits(_) => {}
        }
    }
    InternalErrors::Generic(error.to_string())
}

fn engine_config() -> EngineConfig {
    let options = ai_device_options();
    EngineConfig {
        devices: device_selection(&options.devices),
        gpu_slots: options.gpu_slots,
        cores: crate::memory::cpu_cores(),
    }
}

fn device_selection(devices: &AiDeviceSelection) -> DeviceSelection {
    match devices {
        AiDeviceSelection::Auto => DeviceSelection::Auto,
        AiDeviceSelection::Cpu => DeviceSelection::Cpu,
        AiDeviceSelection::Gpus(gpus) => DeviceSelection::Gpus(gpus.clone()),
    }
}

/// How many GPUs AI models run on with this process' options - 0 for
/// `--ai-devices cpu` without looking for any; else see [`gpus`].
pub(crate) fn used_gpu_count() -> usize {
    match OPTIONS.get().map(|o| &o.devices) {
        Some(AiDeviceSelection::Cpu) => 0,
        _ => gpus().iter().filter(|gpu| gpu.used).count(),
    }
}

/// The computer's GPUs and which ones AI models run on with this process'
/// options (`--ai-devices`; nothing is fixed by asking). Opens the discrete
/// GPUs once: slow, keep it off startup paths.
pub fn gpus() -> Vec<evanalyzer_nn::GpuInfo> {
    let devices = OPTIONS.get().map(|o| o.devices.clone()).unwrap_or_default();
    evanalyzer_nn::gpus(&device_selection(&devices))
}

/// Parses `--gpu-slots`: a number of at least 1.
pub fn parse_gpu_slots(s: &str) -> Result<usize, String> {
    match s.trim().parse::<usize>() {
        Ok(n) if n >= 1 => Ok(n),
        _ => Err(format!("`{s}` is not a number of at least 1")),
    }
}

impl FromStr for AiDeviceSelection {
    type Err = String;

    /// `auto`, `cpu`, or GPU indexes like `0,2`.
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let s = s.trim();
        if s.eq_ignore_ascii_case("auto") {
            return Ok(Self::Auto);
        }
        if s.eq_ignore_ascii_case("cpu") {
            return Ok(Self::Cpu);
        }
        let mut gpus = Vec::new();
        for part in s.split(',') {
            let index: usize = part.trim().parse().map_err(|_| {
                format!("`{s}` is not `auto`, `cpu` or a list of GPU indexes like `0,2`")
            })?;
            if !gpus.contains(&index) {
                gpus.push(index);
            }
        }
        Ok(Self::Gpus(gpus))
    }
}

impl fmt::Display for AiDeviceSelection {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Auto => f.write_str("auto"),
            Self::Cpu => f.write_str("cpu"),
            Self::Gpus(gpus) => {
                let list: Vec<String> = gpus.iter().map(usize::to_string).collect();
                f.write_str(&list.join(","))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn device_selection_parses_and_round_trips() {
        assert_eq!("auto".parse(), Ok(AiDeviceSelection::Auto));
        assert_eq!("CPU".parse(), Ok(AiDeviceSelection::Cpu));
        assert_eq!(" 2, 0,2 ".parse(), Ok(AiDeviceSelection::Gpus(vec![2, 0])));
        assert!("gpu".parse::<AiDeviceSelection>().is_err());
        assert!("0,,1".parse::<AiDeviceSelection>().is_err());
        for s in ["auto", "cpu", "0,3"] {
            assert_eq!(s.parse::<AiDeviceSelection>().unwrap().to_string(), s);
        }
    }

    #[test]
    fn the_device_options_reach_the_engine() {
        let config = engine_config();
        assert!(config.gpu_slots >= 1);
        assert!(config.cores >= 1);
    }

    #[test]
    fn gpu_slots_are_at_least_one() {
        assert_eq!(parse_gpu_slots("3"), Ok(3));
        assert!(parse_gpu_slots("0").is_err());
        assert!(parse_gpu_slots("auto").is_err());
        assert_eq!(AiDeviceOptions::default().gpu_slots, DEFAULT_GPU_SLOTS);
    }
}
