fn main() {
    relink_torch_cuda();
}

/// torch-sys asks the linker for `-ltorch_cuda`, but on Linux rustc always
/// inserts `-Wl,--as-needed` ahead of the native libs and never turns it back
/// off. Since nothing in tch's C++ shim has a symbol that's only defined in
/// `libtorch_cuda.so` (CUDA support registers itself via static initializers
/// once the .so is actually loaded, not via ordinary symbol resolution), the
/// linker silently drops it from `NEEDED` and the binary is CPU-only at
/// runtime even though it linked "successfully".
///
/// The usual fix — re-assert `-Wl,--no-as-needed -ltorch_cuda` later on the
/// command line — only works with linkers that re-evaluate each `-l`
/// occurrence independently (e.g. GNU `ld.bfd`). `mold` (the project-wide
/// linker, set via `.cargo/config.toml` for link speed) instead dedupes
/// repeated references to the same shared object and keeps only the first,
/// `--as-needed`-affected decision, so the reassertion is a no-op under mold.
/// Overriding `-fuse-ld` to `bfd` *after* the existing `-fuse-ld=mold` flag
/// (gcc honors the last `-fuse-ld` it sees) switches just this final binary
/// link to bfd, where the reassertion works. This only affects the final
/// `evanalyzer` executable's link step — compiling the intermediate rlibs
/// doesn't invoke a system linker at all, so it costs nothing in build time.
///
/// Gated on the CUDA build of libtorch actually being present, so CPU-only
/// builds are unaffected.
///
/// Windows has the same problem in a different form: an import library only
/// adds its DLL to the executable's import table when a symbol from it is
/// actually referenced, so `torch_cuda.dll` is left out and never loaded
/// (CUDA unavailable at runtime, although every CUDA DLL ships next to the
/// exe). There, `/INCLUDE:` forces a reference to a function `torch_cuda`
/// exports - `int at::cuda::warp_size()`, in its MSVC-mangled name.
fn relink_torch_cuda() {
    let Some(libtorch) = std::env::var_os("LIBTORCH") else {
        return;
    };
    println!("cargo:rerun-if-env-changed=LIBTORCH");
    let lib = std::path::Path::new(&libtorch).join("lib");
    match std::env::var("CARGO_CFG_TARGET_OS").as_deref() {
        Ok("linux") if lib.join("libtorch_cuda.so").exists() => {
            println!("cargo:rustc-link-arg-bins=-Wl,--no-as-needed");
            println!("cargo:rustc-link-arg-bins=-ltorch_cuda");
            println!("cargo:rustc-link-arg-bins=-fuse-ld=bfd");
        }
        Ok("windows") if lib.join("torch_cuda.lib").exists() => {
            println!("cargo:rustc-link-arg-bins=/INCLUDE:?warp_size@cuda@at@@YAHXZ");
        }
        _ => {}
    }
}
