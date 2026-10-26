//! # third_party_licenses
//!
//! Third-party dependency license data for the About dialog.
//!
//! `generated/third_party_licenses.json` is produced by cargo-about (the
//! exact command is in `about.toml` at the repo root, template:
//! `docs/about.hbs`; CI fails when it's out of date) and
//! committed to the repo. It is not regenerated at build time so a plain
//! `cargo build` never depends on `cargo-about` being installed.
//!
//! `third_party_notices.json` (hand-maintained, same format) adds what
//! isn't a crate: code ported from other projects, the fonts and icons
//! compiled in, and the native libraries shipped next to the binary
//! (libtorch, DuckDB, Skia, NVIDIA's CUDA libraries). Its entries have an
//! empty `version` where there is none.
//!
//! Grouped by exact license *text*, not just SPDX id: copyright lines differ
//! crate to crate, so e.g. "MIT" covers many distinct text variants, each
//! listed separately here with the crates that use that exact text.
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-08-07
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.

use serde::Deserialize;

#[derive(Deserialize)]
struct RawGroup {
    id: String,
    name: String,
    text: String,
    crates: Vec<RawCrate>,
}

#[derive(Deserialize)]
struct RawCrate {
    name: String,
    version: String,
}

pub struct ThirdPartyGroup {
    pub id: String,
    pub name: String,
    /// Pre-joined, sorted "name version, name version, ...".
    pub crates: String,
    /// Split on blank lines, like `license_text::LICENSE_TEXT` - a single
    /// multi-KB Text element has previously rendered garbled in the
    /// software-rendering fallback (see state.slint AppInfoState).
    pub text_paragraphs: Vec<String>,
}

/// Distinct crates covered (not distinct license-text groups, which double
/// count a crate whose license text happens to also appear via another
/// group - that can't happen with cargo-about's own grouping, but the
/// caller wants "how many packages", not "how many license texts").
pub fn load() -> (Vec<ThirdPartyGroup>, usize) {
    const RAW: &str = include_str!("generated/third_party_licenses.json");
    const NOTICES: &str = include_str!("third_party_notices.json");
    let mut raw: Vec<RawGroup> =
        serde_json::from_str(RAW).expect("generated/third_party_licenses.json is malformed");
    let notices: Vec<RawGroup> =
        serde_json::from_str(NOTICES).expect("third_party_notices.json is malformed");
    raw.extend(notices);

    let package_count: usize = raw.iter().map(|g| g.crates.len()).sum();

    let groups = raw
        .into_iter()
        .map(|g| {
            let mut crates = g.crates;
            crates.sort_by(|a, b| a.name.cmp(&b.name).then(a.version.cmp(&b.version)));
            let crates = crates
                .into_iter()
                .map(|c| match c.version.as_str() {
                    "" => c.name,
                    version => format!("{} {version}", c.name),
                })
                .collect::<Vec<_>>()
                .join(", ");

            ThirdPartyGroup {
                id: g.id,
                name: g.name,
                crates,
                text_paragraphs: g.text.split("\n\n").map(str::to_owned).collect(),
            }
        })
        .collect();

    (groups, package_count)
}

/// The About dialog's license list as plain text: `THIRD-PARTY-NOTICES.txt`
/// at the repo root, which the release archives carry next to the binaries
/// and native libraries (some licenses, e.g. NVIDIA's, ask for the text with
/// the files, not only inside the program).
pub fn notices_text() -> String {
    let (groups, _) = load();
    let mut text = String::from(
        "EVAnalyzer - third-party notices\n\n\
         EVAnalyzer itself is licensed under the PolyForm Noncommercial License \
         1.0.0 (see LICENSE). The components below keep their own licenses.\n",
    );
    for group in groups {
        text.push_str(&format!(
            "\n{rule}\n{}\nUsed by: {}\n{rule}\n\n{}\n",
            group.name,
            group.crates,
            group.text_paragraphs.join("\n\n").trim_end(),
            rule = "=".repeat(78),
        ));
    }
    text
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `THIRD-PARTY-NOTICES.txt` matches the license data. Rewrite it with
    /// `UPDATE_NOTICES=1 cargo test -p evanalyzer_gui notices_file` after
    /// regenerating the JSON.
    #[test]
    fn the_notices_file_is_up_to_date() {
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../THIRD-PARTY-NOTICES.txt");
        let expected = notices_text();
        if std::env::var_os("UPDATE_NOTICES").is_some() {
            std::fs::write(path, &expected).unwrap();
        }
        let found = std::fs::read_to_string(path).unwrap_or_default();
        assert!(
            found == expected,
            "THIRD-PARTY-NOTICES.txt is out of date - run `UPDATE_NOTICES=1 cargo test -p evanalyzer_gui notices_file`"
        );
    }

    #[test]
    fn the_notices_are_listed_with_the_crates() {
        let (groups, count) = load();
        let generated: Vec<RawGroup> =
            serde_json::from_str(include_str!("generated/third_party_licenses.json")).unwrap();
        let notices: Vec<RawGroup> =
            serde_json::from_str(include_str!("third_party_notices.json")).unwrap();
        assert!(!notices.is_empty());
        let crates = |g: &[RawGroup]| g.iter().map(|g| g.crates.len()).sum::<usize>();
        assert_eq!(count, crates(&generated) + crates(&notices));
        for notice in &notices {
            assert!(
                !notice.text.trim().is_empty(),
                "{} has no text",
                notice.name
            );
            assert!(
                groups.iter().any(|g| g.name == notice.name),
                "{} is missing",
                notice.name
            );
        }
        // No version: just the name, no trailing space.
        let phosphor = groups.iter().find(|g| g.name.contains("Phosphor")).unwrap();
        assert_eq!(phosphor.crates, "Phosphor Icons");
    }
}
