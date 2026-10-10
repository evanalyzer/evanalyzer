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

impl CitationMetadata {
    /// EVAnalyzer itself - for commands implemented here without a
    /// published method behind them.
    ///
    /// An associated `const` (not a function) so `cite()` can return
    /// `Some(&CitationMetadata::DANMAYR)`: a reference to a constant is
    /// promoted to `'static`, a reference to a function's return value isn't.
    pub const DANMAYR: CitationMetadata = CitationMetadata {
        cite_key: "danmayr2026",
        title: "EVAnalyzer: Enhanced Visual Analyzer",
        authors: &["Joachim Danmayr"],
        year: 2026,
        container: None,
        doi: None,
        url: Some("https://evanalyzer.org"),
        pages: None,
    };

    /// ImageJ - for commands ported from ImageJ's source.
    pub const IMAGEJ: CitationMetadata = CitationMetadata {
        cite_key: "schneider2012imagej",
        title: "NIH Image to ImageJ: 25 years of image analysis",
        authors: &[
            "Caroline A. Schneider",
            "Wayne S. Rasband",
            "Kevin W. Eliceiri",
        ],
        year: 2012,
        container: Some("Nature Methods"),
        doi: Some("10.1038/nmeth.2089"),
        url: Some("https://doi.org/10.1038/nmeth.2089"),
        pages: Some("671-675"),
    };

    /// CellProfiler - for commands following CellProfiler's methods.
    pub const CELLPROFILER: CitationMetadata = CitationMetadata {
        cite_key: "mcquin2018cellprofiler",
        title: "CellProfiler 3.0: Next-generation image processing for biology",
        authors: &[
            "Claire McQuin",
            "Allen Goodman",
            "Vasiliy Chernyshev",
            "Lee Kamentsky",
            "Beth A. Cimini",
            "Kyle W. Karhohs",
            "Minh Doan",
            "Liya Ding",
            "Susanne M. Rafelski",
            "Derek Thirstrup",
            "Winfried Wiegraebe",
            "Shantanu Singh",
            "Tim Becker",
            "Juan C. Caicedo",
            "Anne E. Carpenter",
        ],
        year: 2018,
        container: Some("PLOS Biology"),
        doi: Some("10.1371/journal.pbio.2005970"),
        url: Some("https://doi.org/10.1371/journal.pbio.2005970"),
        pages: Some("e2005970"),
    };
}
