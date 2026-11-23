//! # feature_rules
//!
//! Who may use what in the app crate (see `lib.rs`), until separate crates
//! make the compiler check it:
//! - The features that do the work (`results`, `images`, `files`, `jobs`,
//!   `training`) use no project feature and nothing of `remote`.
//! - The features of the open project (`project`, `modules`, `templates`,
//!   `annotations`, `ai`, `user`) reach the work only through the `Backend`
//!   trait and the features' types: no feature's `local` code, no local
//!   backend or file system, nothing of `remote`.
//! - The remote client (`remote/client`) only sends: no `local` code.
//!
//! Tests are left out: they may set up a local backend.
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-11-22
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.

use std::path::{Path, PathBuf};

const WORKING: &[&str] = &["results", "images", "files", "jobs", "training"];
const PROJECT: &[&str] = &[
    "project",
    "modules",
    "templates",
    "annotations",
    "ai",
    "user",
];

/// What a working feature may not name.
const NOT_IN_WORKING: &[&str] = &[
    "crate::remote",
    "crate::project",
    "crate::modules",
    "crate::templates",
    "crate::annotations",
    "crate::ai",
    "crate::user",
];
/// What a project feature may not name.
const NOT_IN_PROJECT: &[&str] = &[
    "crate::remote",
    "crate::results::local",
    "crate::images::local",
    "crate::files::local",
    "crate::jobs::local",
    "crate::training::local",
    "crate::backend::local",
    "LocalBackend",
    "LocalFileSystem",
];
/// What the remote client may not name.
const NOT_IN_REMOTE_CLIENT: &[&str] = &["::local", "LocalBackend", "LocalFileSystem"];

fn rust_files(path: &Path, out: &mut Vec<PathBuf>) {
    if path.is_dir() {
        for entry in std::fs::read_dir(path).expect("app sources") {
            rust_files(&entry.expect("a directory entry").path(), out);
        }
    } else if path.extension().is_some_and(|ext| ext == "rs") {
        out.push(path.to_path_buf());
    }
}

/// The files of `feature`: `feature.rs` and everything below `feature/`,
/// test files left out.
fn feature_files(src: &Path, feature: &str) -> Vec<PathBuf> {
    let mut files = vec![src.join(format!("{feature}.rs"))];
    rust_files(&src.join(feature), &mut files);
    files.retain(|f| f.exists() && f.file_name().is_none_or(|n| n != "tests.rs"));
    files
}

/// `(line number, line)` of `file` before its tests, comments left out.
fn code_lines(file: &Path) -> Vec<(usize, String)> {
    let source = std::fs::read_to_string(file).expect("a source file");
    source
        .lines()
        .take_while(|line| !line.starts_with("#[cfg(test)]"))
        .enumerate()
        .filter(|(_, line)| !line.trim_start().starts_with("//"))
        .map(|(i, line)| (i + 1, line.to_string()))
        .collect()
}

fn names(text: &str, name: &str) -> bool {
    text.match_indices(name).any(|(at, _)| {
        let after = text[at + name.len()..].chars().next();
        !after.is_some_and(|c| c.is_alphanumeric() || c == '_')
    })
}

fn broken(files: &[PathBuf], forbidden: &[&str], src: &Path) -> Vec<String> {
    let mut found = Vec::new();
    for file in files {
        for (n, line) in code_lines(file) {
            for name in forbidden {
                if names(&line, name) {
                    let rel = file.strip_prefix(src).unwrap_or(file).display();
                    found.push(format!("{rel}:{n}: {} - names {name}", line.trim()));
                }
            }
        }
    }
    found
}

#[test]
fn features_use_only_what_they_may() {
    let src = Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
    let mut found = Vec::new();
    for feature in WORKING {
        found.extend(broken(&feature_files(&src, feature), NOT_IN_WORKING, &src));
    }
    for feature in PROJECT {
        found.extend(broken(&feature_files(&src, feature), NOT_IN_PROJECT, &src));
    }
    found.extend(broken(
        &feature_files(&src, "remote/client"),
        NOT_IN_REMOTE_CLIENT,
        &src,
    ));
    assert!(
        found.is_empty(),
        "a feature uses what it may not (see this test's comment):\n{}",
        found.join("\n")
    );
}
