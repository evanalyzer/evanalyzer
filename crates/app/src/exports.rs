//! # exports
//!
//! The project's exports (`ProjectSettings::exports`, see `docs/exports.md`):
//! named recipes of the tables, plate layouts, drawn images and scripts to
//! write from a run's results.
//!
//! - [`check`]: what is wrong with a project's exports - for
//!   `project validate`, the editor and before every run.
//! - [`naming`]: the file names an export writes (automatic names, the
//!   prefix, unique names per image).
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-11-24
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.

pub mod check;
pub mod naming;

use evanalyzer_cfg::settings::export_settings::ExportSettings;

/// The export `id_or_name` names among `exports`: by id if it is a number
/// one of them has, else by name ignoring case.
pub fn find_export<'a>(
    exports: &'a [ExportSettings],
    id_or_name: &str,
) -> Option<&'a ExportSettings> {
    let wanted = id_or_name.trim();
    wanted
        .parse::<u32>()
        .ok()
        .and_then(|id| exports.iter().find(|export| export.id == id))
        .or_else(|| {
            exports
                .iter()
                .find(|export| export.name.trim().eq_ignore_ascii_case(wanted))
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn export(id: u32, name: &str) -> ExportSettings {
        serde_json::from_value(serde_json::json!({ "id": id, "name": name })).unwrap()
    }

    /// By id, else by name ignoring case - a name that is a number still
    /// finds its export when no id matches.
    #[test]
    fn an_export_is_found_by_id_or_name() {
        let exports = [export(3, "Per image"), export(7, "2024")];
        assert_eq!(find_export(&exports, "3").unwrap().id, 3);
        assert_eq!(find_export(&exports, "per IMAGE ").unwrap().id, 3);
        assert_eq!(find_export(&exports, "2024").unwrap().id, 7);
        assert!(find_export(&exports, "9").is_none());
        assert!(find_export(&exports, "nothing").is_none());
    }
}
