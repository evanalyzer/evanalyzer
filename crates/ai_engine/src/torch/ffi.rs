//! # ffi
//!
//! The C functions of the plugin library ([`crate::api::ffi`]): one
//! [`TorchEngine`] per process, set up by `eva_ai_init`. No panic crosses
//! into the app - each one becomes an error message.
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-10-19
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.

use super::TorchEngine;
use crate::api::{
    API_VERSION, EngineConfig, Image, InferenceEngine, Request,
    ffi::{Buf, LogFn, decode, encode},
};
use std::{
    panic::{AssertUnwindSafe, catch_unwind},
    sync::OnceLock,
};

static ENGINE: OnceLock<TorchEngine> = OnceLock::new();
static LOG: OnceLock<LogFn> = OnceLock::new();

/// Forwards this library's log lines (it has its own copy of `log`) to the
/// app.
struct Forward;

impl log::Log for Forward {
    fn enabled(&self, metadata: &log::Metadata) -> bool {
        metadata.level() <= log::max_level()
    }

    fn log(&self, record: &log::Record) {
        if !self.enabled(record.metadata()) {
            return;
        }
        if let Some(log) = LOG.get() {
            let target = record.target();
            let message = record.args().to_string();
            log(
                record.level() as u8,
                target.as_ptr(),
                target.len(),
                message.as_ptr(),
                message.len(),
            );
        }
    }

    fn flush(&self) {}
}

fn level_filter(level: u8) -> log::LevelFilter {
    match level {
        0 => log::LevelFilter::Off,
        1 => log::LevelFilter::Error,
        2 => log::LevelFilter::Warn,
        3 => log::LevelFilter::Info,
        4 => log::LevelFilter::Debug,
        _ => log::LevelFilter::Trace,
    }
}

/// Runs `f`, a panic turned into an error.
fn guarded<T>(f: impl FnOnce() -> Result<T, String>) -> Result<T, String> {
    catch_unwind(AssertUnwindSafe(f)).unwrap_or_else(|panic| {
        let message = panic
            .downcast_ref::<String>()
            .cloned()
            .or_else(|| panic.downcast_ref::<&str>().map(|s| s.to_string()))
            .unwrap_or_else(|| "unknown error".into());
        Err(format!("AI inference crashed: {message}"))
    })
}

/// Writes `result` to `out` / `out_err`: 0 on success, 1 on an error.
///
/// # Safety
/// `out` and `out_err` must be valid for writes.
unsafe fn answer(result: Result<Vec<u8>, String>, out: *mut Buf, out_err: *mut Buf) -> i32 {
    match result {
        Ok(bytes) => {
            unsafe { out.write(Buf::from_vec(bytes)) };
            0
        }
        Err(message) => {
            unsafe { out_err.write(Buf::from_vec(message.into_bytes())) };
            1
        }
    }
}

/// See [`crate::api::ffi::ApiVersionFn`].
#[unsafe(no_mangle)]
pub extern "C" fn eva_ai_api_version() -> u32 {
    API_VERSION
}

/// See [`crate::api::ffi::InitFn`].
///
/// # Safety
/// `config` must point at `config_len` bytes; `out_info` and `out_err` must
/// be valid for writes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn eva_ai_init(
    config: *const u8,
    config_len: usize,
    log: LogFn,
    max_log_level: u8,
    out_info: *mut Buf,
    out_err: *mut Buf,
) -> i32 {
    let bytes = unsafe { slice(config, config_len) };
    let result = guarded(|| {
        if LOG.set(log).is_ok() && log::set_logger(&Forward).is_ok() {
            log::set_max_level(level_filter(max_log_level));
        }
        let config: EngineConfig = decode(bytes)?;
        let engine = ENGINE.get_or_init(|| TorchEngine::new(&config));
        encode(&engine.info())
    });
    unsafe { answer(result, out_info, out_err) }
}

/// See [`crate::api::ffi::InferFn`].
///
/// # Safety
/// `request` must point at `request_len` bytes, `data` at `data_len`
/// floats; `out` and `out_err` must be valid for writes.
#[unsafe(no_mangle)]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn eva_ai_infer(
    request: *const u8,
    request_len: usize,
    width: usize,
    height: usize,
    channels: usize,
    data: *const f32,
    data_len: usize,
    out: *mut Buf,
    out_err: *mut Buf,
) -> i32 {
    let request_bytes = unsafe { slice(request, request_len) };
    let data = unsafe { slice(data, data_len) };
    let result = guarded(|| {
        let engine = ENGINE
            .get()
            .ok_or("AI inference: the plugin was not set up (eva_ai_init)")?;
        let request: Request = decode(request_bytes)?;
        let image = Image {
            width,
            height,
            channels,
            data,
        };
        encode(&engine.infer(&request, image)?)
    });
    unsafe { answer(result, out, out_err) }
}

/// See [`crate::api::ffi::FreeFn`].
///
/// # Safety
/// `buf` must come from this library and be given back once.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn eva_ai_free(buf: Buf) {
    drop(unsafe { buf.into_vec() });
}

/// `len` values at `ptr` (none for a null pointer).
///
/// # Safety
/// A non-null `ptr` must point at `len` valid values.
unsafe fn slice<'a, T>(ptr: *const T, len: usize) -> &'a [T] {
    if ptr.is_null() || len == 0 {
        &[]
    } else {
        unsafe { std::slice::from_raw_parts(ptr, len) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::{DeviceSelection, EngineInfo, Response, Task};
    use crate::torch::test_support::trace_and_save_model;

    extern "C" fn no_log(_: u8, _: *const u8, _: usize, _: *const u8, _: usize) {}

    /// The answer of a C call: the bytes on success, else the message.
    fn bytes(code: i32, out: Buf, err: Buf) -> Result<Vec<u8>, String> {
        let (out, err) = unsafe { (out.into_vec(), err.into_vec()) };
        if code == 0 {
            Ok(out)
        } else {
            Err(String::from_utf8(err).unwrap())
        }
    }

    fn init() -> EngineInfo {
        let config = encode(&EngineConfig {
            devices: DeviceSelection::Cpu,
            gpu_slots: 1,
            cores: 2,
        })
        .unwrap();
        let (mut info, mut err) = (Buf::EMPTY, Buf::EMPTY);
        let code = unsafe {
            eva_ai_init(
                config.as_ptr(),
                config.len(),
                no_log,
                0,
                &mut info,
                &mut err,
            )
        };
        decode(&bytes(code, info, err).unwrap()).unwrap()
    }

    fn infer(
        request: &Request,
        width: usize,
        height: usize,
        data: &[f32],
    ) -> Result<Response, String> {
        let request = encode(request).unwrap();
        let (mut out, mut err) = (Buf::EMPTY, Buf::EMPTY);
        let code = unsafe {
            eva_ai_infer(
                request.as_ptr(),
                request.len(),
                width,
                height,
                1,
                data.as_ptr(),
                data.len(),
                &mut out,
                &mut err,
            )
        };
        bytes(code, out, err).map(|b| decode(&b).unwrap())
    }

    #[test]
    fn the_c_functions_set_up_the_engine_and_run_a_model() {
        assert_eq!(eva_ai_api_version(), API_VERSION);
        let info = init();
        assert!(info.name.starts_with("libtorch"), "{}", info.name);
        assert_eq!(info.devices[0].name, "CPU");

        // A one-channel "U-Net" giving the image back: the probabilities are
        // the pixels.
        let (_dir, model) = trace_and_save_model(1, 32, 32, |x| x.shallow_clone());
        let request = Request {
            model,
            task: Task::UNet {
                foreground_channel: 0,
                softmax: false,
                boundary_channel: -1,
            },
        };
        let pixels: Vec<f32> = (0..4).map(|v| v as f32 / 4.0).collect();
        match infer(&request, 2, 2, &pixels).unwrap() {
            Response::UNet {
                probabilities,
                boundary,
            } => {
                assert_eq!(probabilities, pixels);
                assert!(boundary.is_none());
            }
            other => panic!("{other:?}"),
        }

        // Errors come back as messages, not panics.
        let missing = Request {
            model: "/no/such/model.pt".into(),
            ..request
        };
        let err = infer(&missing, 2, 2, &pixels).unwrap_err();
        assert!(err.contains("Failed to load U-Net model"), "{err}");
        let err = infer(&missing, 4, 4, &pixels).unwrap_err();
        assert!(err.contains("16 expected"), "{err}");
    }

    #[test]
    fn a_panic_becomes_an_error() {
        let err = guarded::<()>(|| panic!("boom")).unwrap_err();
        assert_eq!(err, "AI inference crashed: boom");
    }
}
