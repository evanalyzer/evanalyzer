//! # migration_v4_to_v5
//!
//! Migrates a [`ProjectSettings`](crate::settings::project_settings::ProjectSettings)
//! document from schema version 4 to version 5.
//!
//! Version 5 adds the project's exports (`exports`, see `docs/exports.md`).
//! Nothing in a version 4 file changes: the step exists so that an app that
//! doesn't know exports refuses a version 5 file instead of opening it and
//! saving it back without them.
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-11-24
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.

use serde_json::Value;

pub fn migrate_from_v4_to_v5(_raw: &mut Value) {}

#[cfg(test)]
mod tests {
    use crate::load_config::load_project_settings;
    use crate::settings::project_settings::ProjectSettings;

    /// A version 4 project reads as before, without exports.
    #[test]
    fn a_version_4_project_reads_unchanged_without_exports() {
        let mut raw = serde_json::to_value(ProjectSettings::default()).unwrap();
        raw["schemaVersion"] = serde_json::json!(4);
        raw.as_object_mut().unwrap().remove("exports");
        let before = raw.clone();

        let settings = load_project_settings(raw).expect("a version 4 project");
        assert!(settings.exports.is_empty());
        let mut after = serde_json::to_value(&settings).unwrap();
        after.as_object_mut().unwrap().remove("exports");
        assert_eq!(after["classification"], before["classification"]);
        assert_eq!(after["pipelines"], before["pipelines"]);
    }
}
