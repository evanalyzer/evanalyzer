#[allow(dead_code)]
#[derive(Debug, Clone, PartialEq)]
pub enum ParamType {
    Number,
    Text,
    Dropdown,
    Toggle,
    Slider,
    /// Integer with +/- step buttons; min/max/step are populated from cmdsmeta.
    Spinner,
    Group,
    /// u32 displayed as a class-aware dropdown (Background / Manual / named classes)
    ObjClass,
    /// u32 displayed as a segmentation-class dropdown (same layout, "Seg." prefix)
    SegClass,
    /// comma-separated u32 list; displayed as a multi-select class picker
    MultiObjClass,
    /// comma-separated u32 list; displayed as a multi-select segmentation-class picker
    MultiSegClass,
    /// PixelUnits enum - options populated from serde names (bit / % / rel)
    PixelUnits,
    /// SizeUnits enum - options populated from serde names (nm / px / …)
    SizeUnits,
    /// Read-only text label - value is displayed but the field is not editable.
    Label,
    /// PathBuf displayed as a text field with a "Browse…" button that opens a
    /// native file picker. `options` holds the allowed file extensions (empty
    /// means any file); the dialog's initial directory is derived from the
    /// field's current value.
    FilePath,
    /// `ImageAddress` displayed as a source picker (channel / memory slot /
    /// scratchpad). `value` is `ImageAddress::to_param_value`'s string
    /// (e.g. "channel:2"); `options` holds the same pre-split as
    /// `[kind, number]`, since Slint can't split strings.
    ImageAddress,
    /// `ImageChannelIdx` displayed as the channel picker (channel name +
    /// color of the open image). `value` is the channel index.
    ImageChannel,
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct ParameterDef {
    pub name: String,
    pub display_name: String,
    /// First doc-comment line of the original Rust field; empty when undocumented.
    pub description: String,
    pub value: String,
    pub param_type: ParamType,
    pub options: Vec<String>,
    pub min: f32,
    pub max: f32,
    pub step: f32,
    /// Non-empty only when param_type == Group.
    /// Each inner Vec is one item in the list (e.g. one ThresholdEntry).
    pub groups: Vec<Vec<ParameterDef>>,
    /// The setting's default, formatted like `value`; empty when unknown. The UI
    /// counts advanced settings whose value differs from it as "changed".
    pub default_value: String,
    /// `visibility = Advanced`: shown only while the step's advanced settings
    /// are expanded.
    pub advanced: bool,
    /// Per entry of `options` (dropdowns): whether that option is advanced.
    /// Empty when no option is.
    pub option_advanced: Vec<bool>,
}
