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
}
