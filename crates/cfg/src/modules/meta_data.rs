use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Debug, Clone, Default, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct AuthorInformation {
    /// Full name of the author
    pub full_name: String,

    /// Author organization
    pub organization: String,
}

#[derive(Serialize, Deserialize, Debug, Clone, Default, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct MetaData {
    /// Name of the module
    pub name: String,

    /// A short one line description of the module
    pub short_description: String,

    /// A long detailed description of the module
    pub description: String,

    /// Every author, in display order (the first entry is treated as the
    /// primary author in template/pipeline picker dialogs, the rest as
    /// co-authors). Free-form display strings, not split into first/last.
    #[serde(default)]
    pub authors: Vec<AuthorInformation>,

    /// Creation time
    pub creation_time: chrono::DateTime<chrono::Utc>,

    /// The author's version of this component or template
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub version: Option<u32>,

    /// Category used to group templates in pickers (e.g. "Cell Biology / Uptake")
    #[serde(default)]
    pub category: String,

    /// Free-form keywords used to search/filter templates
    #[serde(default)]
    pub tags: Vec<String>,

    /// Version of EVAnalyzer that last wrote this file (`CARGO_PKG_VERSION`
    /// of the writing build). Empty for files written before this field
    /// existed, or built by hand rather than saved from the app.
    #[serde(default)]
    pub app_version: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_version_is_optional_and_only_written_when_set() {
        let meta: MetaData =
            serde_json::from_str(r#"{"name":"X","shortDescription":"","description":"","creationTime":"2026-10-14T00:00:00Z"}"#)
                .unwrap();
        assert_eq!(meta.version, None);
        assert!(!serde_json::to_string(&meta).unwrap().contains("version\""));

        let versioned = MetaData {
            version: Some(3),
            ..meta
        };
        let json = serde_json::to_string(&versioned).unwrap();
        assert!(json.contains(r#""version":3"#), "{json}");
        assert_eq!(
            serde_json::from_str::<MetaData>(&json).unwrap().version,
            Some(3)
        );
    }
}
