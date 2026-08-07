//! Third-party dependency license data for the About dialog.
//!
//! `generated/third_party_licenses.json` is produced by
//! `cargo about generate about.hbs -o crates/gui/src/generated/third_party_licenses.json`
//! (config: `about.toml`, template: `about.hbs`, both at the repo root) and
//! committed to the repo - CI regenerates and commits it on pushes to main
//! alongside the coverage badge, see `.github/workflows/ci.yml`. It is not
//! regenerated at build time so a plain `cargo build` never depends on
//! `cargo-about` being installed.
//!
//! Grouped by exact license *text*, not just SPDX id: copyright lines differ
//! crate to crate, so e.g. "MIT" covers many distinct text variants, each
//! listed separately here with the crates that use that exact text.

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
    let raw: Vec<RawGroup> =
        serde_json::from_str(RAW).expect("generated/third_party_licenses.json is malformed");

    let package_count: usize = raw.iter().map(|g| g.crates.len()).sum();

    let groups = raw
        .into_iter()
        .map(|g| {
            let mut crates = g.crates;
            crates.sort_by(|a, b| a.name.cmp(&b.name).then(a.version.cmp(&b.version)));
            let crates = crates
                .into_iter()
                .map(|c| format!("{} {}", c.name, c.version))
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
