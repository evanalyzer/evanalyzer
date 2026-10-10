//! Which devices AI models (Cellpose, StarDist, U-Net, YOLOv5) run on, set
//! once at startup from the command line (`--ai-devices`, `--gpu-slots`).

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
    /// Every CUDA GPU, or the CPU when there is none.
    #[default]
    Auto,
    /// Never use a GPU.
    Cpu,
    /// These CUDA GPUs (indexes as CUDA numbers them); ones that don't exist
    /// are ignored, and with none left the CPU is used.
    Gpus(Vec<usize>),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AiDeviceOptions {
    pub devices: AiDeviceSelection,
    /// Parallel inferences per GPU, at least 1 (see [`parse_gpu_slots`]).
    pub gpu_slots: usize,
}

impl Default for AiDeviceOptions {
    fn default() -> Self {
        Self {
            devices: AiDeviceSelection::default(),
            gpu_slots: DEFAULT_GPU_SLOTS,
        }
    }
}

static OPTIONS: OnceLock<AiDeviceOptions> = OnceLock::new();

/// Sets the AI device options for this process. Call it once at startup,
/// before the first AI step runs; returns `false` (and changes nothing) if
/// options were already set or AI inference already started.
pub fn configure_ai_devices(options: AiDeviceOptions) -> bool {
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
    fn gpu_slots_are_at_least_one() {
        assert_eq!(parse_gpu_slots("3"), Ok(3));
        assert!(parse_gpu_slots("0").is_err());
        assert!(parse_gpu_slots("auto").is_err());
        assert_eq!(AiDeviceOptions::default().gpu_slots, DEFAULT_GPU_SLOTS);
    }
}
