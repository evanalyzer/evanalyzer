//! How many CUDA GPUs the NVIDIA driver sees - asked of the driver itself
//! (`libcuda.so.1` / `nvcuda.dll`), so the CPU plugin can be chosen without
//! loading libtorch, and the CUDA one only where it can work.

use std::sync::OnceLock;

/// CUDA GPUs of this computer; 0 without an NVIDIA driver (and on macOS).
pub(crate) fn gpu_count() -> usize {
    static COUNT: OnceLock<usize> = OnceLock::new();
    *COUNT.get_or_init(|| {
        let count = probe();
        log::debug!("CUDA driver: {count} GPU(s)");
        count
    })
}

#[cfg(any(target_os = "linux", target_os = "windows"))]
fn probe() -> usize {
    #[cfg(target_os = "linux")]
    const DRIVER: &str = "libcuda.so.1";
    #[cfg(target_os = "windows")]
    const DRIVER: &str = "nvcuda.dll";

    type CuInit = unsafe extern "C" fn(flags: u32) -> i32;
    type CuDeviceGetCount = unsafe extern "C" fn(count: *mut i32) -> i32;

    // SAFETY: the NVIDIA driver library; its two functions are called with
    // the arguments the CUDA driver API documents. The library stays loaded
    // (libtorch uses it later anyway).
    unsafe {
        let Ok(driver) = libloading::Library::new(DRIVER) else {
            return 0;
        };
        let (Ok(init), Ok(device_count)) = (
            driver.get::<CuInit>(b"cuInit\0"),
            driver.get::<CuDeviceGetCount>(b"cuDeviceGetCount\0"),
        ) else {
            return 0;
        };
        let mut count = 0;
        let found = init(0) == 0 && device_count(&mut count) == 0;
        std::mem::forget(driver);
        if found { count.max(0) as usize } else { 0 }
    }
}

#[cfg(not(any(target_os = "linux", target_os = "windows")))]
fn probe() -> usize {
    0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_probe_answers_without_a_driver_too() {
        // On a computer without an NVIDIA driver (CI, the dev container)
        // this is 0; either way it must not fail and stays the same.
        assert_eq!(gpu_count(), gpu_count());
    }
}
