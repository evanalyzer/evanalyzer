use schemars::JsonSchema;
use serde::Serialize;

#[derive(Serialize, Debug, Clone, Default, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct CitationMetadata {
    pub cite_key: &'static str,
    pub title: &'static str,
    pub authors: &'static [&'static str],
    pub year: u16,
    pub container: Option<&'static str>, // Journal or Conference Name
    pub doi: Option<&'static str>,
    pub url: Option<&'static str>,
    pub pages: Option<&'static str>,
}
