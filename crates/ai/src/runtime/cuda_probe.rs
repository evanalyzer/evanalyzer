//! # cuda_probe
//!
//! Which CUDA GPUs the NVIDIA driver sees - asked of the driver itself
//! (`libcuda.so.1` / `nvcuda.dll`), so the CPU plugin can be chosen without
//! loading libtorch, and the CUDA one only where it can work. A GPU older
//! than the shipped libtorch has kernels for isn't usable: libtorch would
//! find it and then fail on the first model ("no kernel image is available").
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-10-19
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.

use std::sync::OnceLock;

/// The oldest GPUs the CUDA libtorch the release ships has kernels for:
/// cu130 builds for compute capability 7.5 (Turing: RTX 20xx, GTX 16xx) and
/// newer. Goes with `CUDA_VARIANT` in libs/download.sh.
const MIN_COMPUTE_CAPABILITY: (u32, u32) = (7, 5);

/// The CUDA GPUs of this computer that can run models, as CUDA device
/// indexes (the ones libtorch uses too); none without an NVIDIA driver (and
/// on macOS).
pub(crate) fn usable_gpus() -> &'static [usize] {
    static USABLE: OnceLock<Vec<usize>> = OnceLock::new();
    USABLE.get_or_init(|| {
        let capabilities = probe();
        log::debug!("CUDA driver: GPUs with compute capability {capabilities:?}");
        usable(&capabilities)
    })
}

/// The indexes of `capabilities` new enough for this build; a warning for
/// each one that isn't.
fn usable(capabilities: &[(u32, u32)]) -> Vec<usize> {
    let (min_major, min_minor) = MIN_COMPUTE_CAPABILITY;
    let mut gpus = Vec::new();
    for (index, &(major, minor)) in capabilities.iter().enumerate() {
        if (major, minor) >= MIN_COMPUTE_CAPABILITY {
            gpus.push(index);
        } else {
            log::warn!(
                "GPU {index} (compute capability {major}.{minor}) is too old for this build's \
                 CUDA (it needs {min_major}.{min_minor} or newer) - AI models run on the CPU instead"
            );
        }
    }
    gpus
}

/// The compute capability (major, minor) of each GPU, by CUDA device index.
#[cfg(any(target_os = "linux", target_os = "windows"))]
fn probe() -> Vec<(u32, u32)> {
    #[cfg(target_os = "linux")]
    const DRIVER: &str = "libcuda.so.1";
    #[cfg(target_os = "windows")]
    const DRIVER: &str = "nvcuda.dll";

    type CuInit = unsafe extern "C" fn(flags: u32) -> i32;
    type CuDeviceGetCount = unsafe extern "C" fn(count: *mut i32) -> i32;
    type CuDeviceGet = unsafe extern "C" fn(device: *mut i32, ordinal: i32) -> i32;
    type CuDeviceGetAttribute =
        unsafe extern "C" fn(value: *mut i32, attribute: i32, device: i32) -> i32;
    // CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR / _MINOR
    const MAJOR: i32 = 75;
    const MINOR: i32 = 76;

    // SAFETY: the NVIDIA driver library; its functions are called with
    // the arguments the CUDA driver API documents. The library stays loaded
    // (libtorch uses it later anyway).
    unsafe {
        let Ok(driver) = libloading::Library::new(DRIVER) else {
            return Vec::new();
        };
        let (Ok(init), Ok(device_count), Ok(device_get), Ok(attribute)) = (
            driver.get::<CuInit>(b"cuInit\0"),
            driver.get::<CuDeviceGetCount>(b"cuDeviceGetCount\0"),
            driver.get::<CuDeviceGet>(b"cuDeviceGet\0"),
            driver.get::<CuDeviceGetAttribute>(b"cuDeviceGetAttribute\0"),
        ) else {
            return Vec::new();
        };
        let mut count = 0;
        let mut capabilities = Vec::new();
        if init(0) == 0 && device_count(&mut count) == 0 {
            for ordinal in 0..count {
                let (mut device, mut major, mut minor) = (0, 0, 0);
                // A GPU whose capability can't be read counts as 0.0 (unusable).
                if device_get(&mut device, ordinal) != 0
                    || attribute(&mut major, MAJOR, device) != 0
                    || attribute(&mut minor, MINOR, device) != 0
                {
                    (major, minor) = (0, 0);
                }
                capabilities.push((major.max(0) as u32, minor.max(0) as u32));
            }
        }
        std::mem::forget(driver);
        capabilities
    }
}

#[cfg(not(any(target_os = "linux", target_os = "windows")))]
fn probe() -> Vec<(u32, u32)> {
    Vec::new()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_probe_answers_without_a_driver_too() {
        // On a computer without an NVIDIA driver (CI, the dev container)
        // there are none; either way it must not fail and stays the same.
        assert_eq!(usable_gpus(), usable_gpus());
    }

    #[test]
    fn gpus_older_than_the_build_supports_are_not_usable() {
        assert_eq!(usable(&[]), Vec::<usize>::new());
        // Pascal (GTX 1060), Turing, Ampere, Volta, Blackwell, unreadable.
        assert_eq!(
            usable(&[(6, 1), (7, 5), (8, 6), (7, 0), (12, 0), (0, 0)]),
            vec![1, 2, 4]
        );
    }
}
