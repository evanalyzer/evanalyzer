fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=PROFILE");

    let manifest_dir = std::path::PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());

    // Watch every .slint file in this directory tree individually
    watch_slint_files(&manifest_dir);

    // Debug builds (and so the tests) keep the element names, so tests can
    // find elements and drive the real UI with pointer events
    // (`i_slint_backend_testing::ElementHandle`); release builds don't need
    // them.
    let debug_build = std::env::var("PROFILE").as_deref() == Ok("debug");
    let config = slint_build::CompilerConfiguration::new()
        .with_include_paths(vec![manifest_dir.clone()])
        .with_style("fluent".into())
        .with_debug_info(debug_build);

    slint_build::compile_with_config(manifest_dir.join("main.slint"), config)
        .expect("Slint compilation failed");
}

fn watch_slint_files(dir: &std::path::Path) {
    if let Ok(entries) = std::fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                watch_slint_files(&path);
            } else if path.extension().map_or(false, |e| e == "slint") {
                println!("cargo:rerun-if-changed={}", path.display());
            }
        }
    }
}
