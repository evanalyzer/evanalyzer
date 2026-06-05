use std::{fs, process::Command};

pub fn compile() -> Result<(), Box<dyn std::error::Error>> {
    let java_src_dir = "java/src";
    let status = Command::new("javac")
        .current_dir(java_src_dir)
        .args(&[
            "-source",
            "1.8",
            "-target",
            "1.8",
            "-cp",
            "bioformats.jar",
            "BioFormatsWrapper.java",
        ])
        .status()?;

    if !status.success() {
        return Err("Java compilation failed (javac returned non-zero exit code)".into());
    }

    // Logic for copying files...
    fs::copy(
        "java/src/BioFormatsWrapper.class",
        "java/BioFormatsWrapper.class",
    )
    .ok();

    Ok(())
}
