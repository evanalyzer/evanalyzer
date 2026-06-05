#[path = "build/java_wrapper_builder.rs"]
mod java_wrapper;

fn main() {
    println!("cargo:rerun-if-changed=java/src/BioFormatsWrapper.java");
    println!("cargo:rerun-if-changed=java/src/bioformats.jar");

    // Execute
    if let Err(e) = java_wrapper::compile() {
        eprintln!("Java Compilation Error: {}", e);
        std::process::exit(1);
    }
}
