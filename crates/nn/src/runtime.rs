//! # runtime
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-11-25
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.
//!
//! Backend and device, chosen once per process on the first model run -
//! nothing is set up before (idle RAM, startup).
//!
//! P3: the GPU is wgpu (Vulkan / DirectX 12 / Metal). The CPU backend
//! (burn-flex) when there is no adapter, only a software one (llvmpipe,
//! lavapipe, WARP) or its set-up fails; a device lost or out of memory
//! during a run → that inference again on the CPU, logged once.

/// Which devices model runs may use.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum DeviceSelection {
    /// Every GPU there is, else the CPU.
    #[default]
    Auto,
    Cpu,
    /// These GPUs (indexes); ones that don't exist are ignored.
    Gpus(Vec<usize>),
}

/// How the engine sets itself up, once per process.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EngineConfig {
    pub devices: DeviceSelection,
    /// Model runs at once per GPU (at least 1).
    pub gpu_slots: usize,
    /// CPU cores the engine may use.
    pub cores: usize,
}

/// A device model runs use, and how many at once.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeviceInfo {
    /// E.g. "CPU", "NVIDIA GeForce RTX 4070 (Vulkan)".
    pub name: String,
    pub slots: usize,
}

/// What the engine is, after it set itself up - for the log and the
/// settings dialog.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EngineInfo {
    /// E.g. "burn 0.22 (wgpu)".
    pub name: String,
    pub devices: Vec<DeviceInfo>,
}
