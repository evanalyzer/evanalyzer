//! # check_templates
//!
//! Checks that every project template (`.evatemplate`) and library module
//! (`.evamodule`) under a folder is usable - not only readable: made into a
//! project as the app does (a module used twice and inside another module,
//! with its default values), checked as the editor checks it, its run's job
//! made, and with `--image` a preview run on that image. The same check as
//! `evanalyzer cli project validate --project <folder>`, without building
//! the GUI - for CI here and in the templates repository.
//!
//! Usage: `cargo run -p evanalyzer_app --example check_templates -- <folder> [--image <file>]`
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-11-21
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.

use std::path::PathBuf;

fn main() {
    let mut folder: Option<PathBuf> = None;
    let mut image: Option<PathBuf> = None;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--image" => image = args.next().map(PathBuf::from),
            _ if folder.is_none() => folder = Some(PathBuf::from(arg)),
            _ => {
                eprintln!("usage: check_templates <folder> [--image <file>]");
                std::process::exit(2);
            }
        }
    }
    let Some(folder) = folder else {
        eprintln!("usage: check_templates <folder> [--image <file>]");
        std::process::exit(2);
    };
    if let Err(e) = evanalyzer_app::modules::validate_templates(&folder, image.as_deref()) {
        eprintln!("{e}");
        std::process::exit(1);
    }
}
