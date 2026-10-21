//! Validates that every `.evatemplate`/`.evamodule` file under a given directory
//! (searched recursively) still deserializes as `ProjectTemplate`/
//! `ModuleTemplate` - the same types and the same `serde_json::from_str`
//! call the GUI/CLI actually use to load them. This is stronger than
//! validating against `docs/project_template.schema.json`/
//! `docs/module_template.schema.json` with a generic JSON Schema
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

use evanalyzer_cfg::settings::templates::{ModuleTemplate, ProjectTemplate};
use std::path::Path;

fn main() {
    let mut args = std::env::args().skip(1);
    let Some(root) = args.next() else {
        eprintln!("usage: check_templates <directory>");
        std::process::exit(2);
    };
    let root = Path::new(&root);

    let mut checked_evatemplate = 0;
    let mut checked_evamodule = 0;
    let mut failures = Vec::new();

    walk(root, &mut |path| {
        let is_evatemplate = path.extension().and_then(|e| e.to_str()) == Some("evatemplate");
        let is_evamodule = path.extension().and_then(|e| e.to_str()) == Some("evamodule");
        if !is_evatemplate && !is_evamodule {
            return;
        }

        let Ok(json) = std::fs::read_to_string(path) else {
            failures.push(format!("{}: could not read file", path.display()));
            return;
        };

        if is_evamodule {
            checked_evamodule += 1;
            if let Err(e) = serde_json::from_str::<ModuleTemplate>(&json) {
                failures.push(format!(
                    "{}: failed to deserialize as ModuleTemplate: {e}",
                    path.display()
                ));
            }
        } else {
            checked_evatemplate += 1;
            if let Err(e) = serde_json::from_str::<ProjectTemplate>(&json) {
                failures.push(format!(
                    "{}: failed to deserialize as ProjectTemplate: {e}",
                    path.display()
                ));
            }
        }
    });

    println!(
        "checked {checked_evatemplate} .evatemplate file(s), {checked_evamodule} .evamodule file(s)"
    );

    if checked_evatemplate == 0 && checked_evamodule == 0 {
        eprintln!(
            "no .evatemplate/.evamodule files found under {} - check the path/checkout is correct",
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
