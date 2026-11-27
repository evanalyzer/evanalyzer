//! # renderer
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-11-19
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.
//!
//! Which Slint renderer draws the window: chosen before the window exists
//! ([`choose`]) and logged once it draws its first frame ([`log_active`]).
//! Slint falls back to the next compiled-in renderer without saying so, so
//! the log line is the only way to tell from a user's log whether the
//! graphics card or the CPU draws. A GPU renderer that ended up on a CPU
//! adapter (remote desktop, virtual machine, no driver) also gets a warning
//! and "Slow drawing" in the status bar (`RendererState`).

use crate::{AppWindow, RendererState};
use evanalyzer_app::backend::Backend;
use slint::{ComponentHandle, GraphicsAPI, RenderingState, SetRenderingNotifierError};
use std::cell::Cell;
use std::rc::Rc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

/// Why a renderer was asked for - part of the log line.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Request {
    /// Slint's own `SLINT_BACKEND` was set in the environment.
    SlintBackend,
    /// Settings › Viewport › Draw in software.
    SoftwareSetting,
    /// Nothing asked for: Slint's default order with fallbacks.
    Default,
}

impl Request {
    fn reason(self) -> String {
        match self {
            Request::SlintBackend => format!(
                "SLINT_BACKEND={}",
                std::env::var("SLINT_BACKEND").unwrap_or_default()
            ),
            Request::SoftwareSetting => "Settings › Viewport › Draw in software".to_string(),
            Request::Default => "default".to_string(),
        }
    }
}

/// Settings › Viewport › Draw in software: Slint's own software renderer
/// instead of the graphics card - chosen before the window exists (so it
/// takes effect at the next start). `SLINT_BACKEND` set in the environment
/// wins.
pub(crate) fn choose() -> Request {
    if std::env::var_os("SLINT_BACKEND").is_some() {
        return Request::SlintBackend;
    }
    let software = evanalyzer_app::backend::local::LocalBackend::default()
        .load_app_settings()
        .map(|settings| settings.viewport.software_renderer)
        .unwrap_or(false);
    if !software {
        if cfg!(target_os = "windows") {
            direct3d_only();
        }
        return Request::Default;
    }
    match slint::BackendSelector::new()
        .renderer_name("software".into())
        .select()
    {
        Ok(()) => Request::SoftwareSetting,
        Err(e) => {
            log::warn!("Could not draw in software, using the default renderer: {e}");
            Request::Default
        }
    }
}

/// Windows: wgpu on Direct3D 12 only, never Vulkan. Slint's default also
/// enumerates Vulkan, which loads the Vulkan loader, every installed driver
/// and every implicit layer (overlays of Steam, OBS, RivaTuner, ...) - a
/// broken one crashed the app without a message (2026-11-19, a user's PC:
/// Vulkan crashed, Direct3D 12 worked). Without Direct3D 12 (before
/// Windows 10, a missing driver) Slint falls back to the software renderer.
/// Compiled everywhere, called only on Windows.
fn direct3d_only() {
    use slint::wgpu_30::{WGPUConfiguration, WGPUSettings, wgpu};
    let mut settings = WGPUSettings::default();
    settings.backends = wgpu::Backends::DX12;
    if let Err(e) = slint::BackendSelector::new()
        .require_wgpu_30(WGPUConfiguration::Automatic(settings))
        .select()
    {
        log::warn!("Could not ask for Direct3D 12, using the default renderer: {e}");
    }
}

/// Whether Slint's software renderer draws the window - set once by
/// [`log_active`] before the first frame, read by the viewport. That
/// renderer stores scene coordinates as i16 and steps through scaled images
/// in 8.8 fixed point (rounded down: an image stretched 8x ends up ~2 %
/// too wide), so the viewport hands it screen-sized frames instead of
/// letting it scale. `false` until known (and without a window, in tests).
static SOFTWARE: AtomicBool = AtomicBool::new(false);

/// See [`SOFTWARE`].
pub(crate) fn is_software() -> bool {
    SOFTWARE.load(Ordering::Relaxed)
}

/// How long after the start a GPU renderer that never reported its
/// graphics API is logged as such.
const UNREPORTED_AFTER: Duration = Duration::from_secs(5);

/// Logs one info line with the renderer that draws `window`, e.g.
/// `Renderer: GPU via wgpu - Vulkan, Intel(R) Graphics (ARL) (integrated
/// GPU) [default]` or `Renderer: software (CPU) - fallback, no GPU renderer
/// could start [default]`.
///
/// Only the software renderer refuses a rendering notifier; the GPU
/// renderers report their graphics API once, before the first frame. One
/// that accepts the notifier but reports nothing is logged after
/// [`UNREPORTED_AFTER`]. The refusal also sets [`is_software`].
///
/// A GPU renderer on a CPU adapter: a warning too, and `RendererState.slow`
/// for the status bar.
pub(crate) fn log_active(ui: &AppWindow, request: Request) {
    let reported = Rc::new(Cell::new(false));
    let seen = Rc::clone(&reported);
    let ui_weak = ui.as_weak();
    let result = ui.window().set_rendering_notifier(move |state, api| {
        if matches!(state, RenderingState::RenderingSetup) && !seen.replace(true) {
            let (text, cpu) = describe(api);
            log::info!("Renderer: GPU via {text} [{}]", request.reason());
            if cpu {
                log::warn!("{SLOW_WARNING}");
                // Not while the frame is drawn: after it.
                let ui_weak = ui_weak.clone();
                slint::Timer::single_shot(Duration::ZERO, move || {
                    if let Some(ui) = ui_weak.upgrade() {
                        let state = ui.global::<RendererState>();
                        state.set_description(text.into());
                        state.set_slow(true);
                    }
                });
            }
        }
    });
    match result {
        Ok(()) => {
            slint::Timer::single_shot(UNREPORTED_AFTER, move || {
                if !reported.get() {
                    log::info!(
                        "Renderer: GPU renderer without a reported graphics API [{}]",
                        request.reason()
                    );
                }
            });
        }
        Err(SetRenderingNotifierError::Unsupported) => {
            SOFTWARE.store(true, Ordering::Relaxed);
            let why = match request {
                Request::Default => "fallback, no GPU renderer could start",
                Request::SoftwareSetting | Request::SlintBackend => "as requested",
            };
            log::info!("Renderer: software (CPU) - {why} [{}]", request.reason());
        }
        Err(e) => log::debug!("Renderer not logged: {e}"),
    }
}

/// The console line for a GPU renderer on a CPU adapter.
const SLOW_WARNING: &str = "Drawing is slow: the graphics card can't be used here (remote \
    desktop, virtual machine or no driver), so the screen is computed through CPU emulation. \
    Turn on Settings › Viewport › Draw in software and restart for better performance.";

/// Graphics API and device, and whether that device is the CPU, e.g. `wgpu - Vulkan, Intel(R) Graphics (ARL)
/// (integrated GPU)` or `OpenGL - Mesa Intel(R) Graphics (ARL), 4.6 (Core
/// Profile) Mesa 25.0.7`. A CPU device (llvmpipe, lavapipe, WARP) shows up
/// here as such: those draw on the CPU although Slint sees a GPU API.
fn describe(api: &GraphicsAPI<'_>) -> (String, bool) {
    match api {
        GraphicsAPI::NativeOpenGL { get_proc_address } => {
            let renderer = gl_string(*get_proc_address, GL_RENDERER);
            let version = gl_string(*get_proc_address, GL_VERSION);
            let cpu = is_cpu_gl_renderer(&renderer);
            let kind = if cpu { " (CPU emulation)" } else { "" };
            (format!("OpenGL - {renderer}{kind}, {version}"), cpu)
        }
        GraphicsAPI::WGPU30 { device, .. } => {
            let info = device.adapter_info();
            let kind = match info.device_type {
                slint::wgpu_30::wgpu::DeviceType::IntegratedGpu => "integrated GPU",
                slint::wgpu_30::wgpu::DeviceType::DiscreteGpu => "discrete GPU",
                slint::wgpu_30::wgpu::DeviceType::VirtualGpu => "virtual GPU",
                slint::wgpu_30::wgpu::DeviceType::Cpu => "CPU emulation",
                slint::wgpu_30::wgpu::DeviceType::Other => "other device",
            };
            let cpu = info.device_type == slint::wgpu_30::wgpu::DeviceType::Cpu;
            (
                format!("wgpu - {:?}, {} ({kind})", info.backend, info.name),
                cpu,
            )
        }
        _ => ("an unknown graphics API".to_string(), false),
    }
}

/// An OpenGL renderer string of a driver that computes on the CPU (Mesa's
/// llvmpipe/softpipe/swrast, Windows' GDI Generic, SwiftShader).
fn is_cpu_gl_renderer(renderer: &str) -> bool {
    let renderer = renderer.to_ascii_lowercase();
    [
        "llvmpipe",
        "softpipe",
        "swrast",
        "gdi generic",
        "swiftshader",
    ]
    .iter()
    .any(|name| renderer.contains(name))
}

const GL_RENDERER: u32 = 0x1F01;
const GL_VERSION: u32 = 0x1F02;

/// `glGetString(name)` through the renderer's own loader; the GL context
/// is current while the rendering notifier runs.
fn gl_string(
    get_proc_address: &dyn Fn(&std::ffi::CStr) -> *const std::ffi::c_void,
    name: u32,
) -> String {
    let ptr = get_proc_address(c"glGetString");
    if ptr.is_null() {
        return "?".to_string();
    }
    // SAFETY: `glGetString` has this signature on every platform; the
    // pointer came from the context's own loader and the context is
    // current. The returned string is static and NUL-terminated, or null.
    unsafe {
        let gl_get_string: extern "system" fn(u32) -> *const std::ffi::c_char =
            std::mem::transmute(ptr);
        let text = gl_get_string(name);
        if text.is_null() {
            "?".to_string()
        } else {
            std::ffi::CStr::from_ptr(text)
                .to_string_lossy()
                .into_owned()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::is_cpu_gl_renderer;

    #[test]
    fn cpu_gl_renderers_are_recognised() {
        assert!(is_cpu_gl_renderer("llvmpipe (LLVM 19.1.7, 256 bits)"));
        assert!(is_cpu_gl_renderer("GDI Generic"));
        assert!(is_cpu_gl_renderer("Google SwiftShader"));
        assert!(!is_cpu_gl_renderer("Mesa Intel(R) Arc(tm) Graphics (MTL)"));
        assert!(!is_cpu_gl_renderer("NVIDIA GeForce RTX 4070/PCIe/SSE2"));
    }
}
