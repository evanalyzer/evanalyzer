//! Validates that every `.evapipe`/`.evapt` file under a given directory
//! (searched recursively) still deserializes as `PipelineTemplate`/
//! `ProjectTemplate` - the same types and the same `serde_json::from_str`
//! call the GUI/CLI actually use to load them. This is stronger than
//! validating against `docs/pipeline_template.schema.json`/
//! `docs/project_template.schema.json` with a generic JSON Schema
//! validator: a `#[serde(alias = ...)]` or an enum's tagging behavior can
//! change what actually deserializes without necessarily showing up as a
//! schema-shape difference.
//!
//! Templates now live in a separate repository
//! (evanalyzer/evanalyzer-templates), so this can't be a `#[cfg(test)]` in
//! this crate the way it was before - there's nothing at a fixed relative
//! path to read. Instead it's a small standalone binary: CI (in both
//! repositories - see `.github/workflows/check-templates.yml`) fetches the
//! templates and points this at the checkout.
//!
//! Usage: `cargo run -p evanalyzer_cfg --example check_templates -- <dir>`

use evanalyzer_cfg::settings::templates::{PipelineTemplate, ProjectTemplate};
use std::path::Path;

fn main() {
    let mut args = std::env::args().skip(1);
    let Some(root) = args.next() else {
        eprintln!("usage: check_templates <directory>");
        std::process::exit(2);
    };
    let root = Path::new(&root);

    let mut checked_evapipe = 0;
    let mut checked_evapt = 0;
    let mut failures = Vec::new();

    walk(root, &mut |path| {
        let is_evapipe = path.extension().and_then(|e| e.to_str()) == Some("evapipe");
        let is_evapt = path.extension().and_then(|e| e.to_str()) == Some("evapt");
        if !is_evapipe && !is_evapt {
            return;
        }

        let Ok(json) = std::fs::read_to_string(path) else {
            failures.push(format!("{}: could not read file", path.display()));
            return;
        };

        if is_evapipe {
            checked_evapipe += 1;
            if let Err(e) = serde_json::from_str::<PipelineTemplate>(&json) {
                failures.push(format!(
                    "{}: failed to deserialize as PipelineTemplate: {e}",
                    path.display()
                ));
            }
        } else {
            checked_evapt += 1;
            if let Err(e) = serde_json::from_str::<ProjectTemplate>(&json) {
                failures.push(format!(
                    "{}: failed to deserialize as ProjectTemplate: {e}",
                    path.display()
                ));
            }
        }
    });

    println!("checked {checked_evapipe} .evapipe file(s), {checked_evapt} .evapt file(s)");

    if checked_evapipe == 0 && checked_evapt == 0 {
        eprintln!(
            "no .evapipe/.evapt files found under {} - check the path/checkout is correct",
            root.display()
        );
        std::process::exit(1);
    }

    if !failures.is_empty() {
        eprintln!("\n{} file(s) failed to deserialize:\n", failures.len());
        for failure in &failures {
            eprintln!("  {failure}");
        }
        std::process::exit(1);
    }
}

fn walk(dir: &Path, on_file: &mut impl FnMut(&Path)) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            if path.file_name().and_then(|n| n.to_str()) == Some(".git") {
                continue;
            }
            walk(&path, on_file);
        } else {
            on_file(&path);
        }
    }
}
