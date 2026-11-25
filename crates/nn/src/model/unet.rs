use crate::model::{Model, Pieces};

pub struct UNet {
    /// Channel of the foreground probability (when the output has more
    /// than one; out-of-range values are clamped).
    foreground_channel: i32,
    /// The channels are class scores: softmax over them first. Else
    /// they are independent probability maps.
    softmax: bool,
    /// A boundary-probability channel (boundary-aware models), `-1` none.
    boundary_channel: i32,
    /// How large a tile runs in pieces.
    pieces: Pieces,
}

pub struct UNetResponse {
    /// Foreground probability per pixel (`width * height`).
    probabilities: Vec<f32>,
    /// Boundary probability per pixel, if a boundary channel was asked
    /// for and exists.
    boundary: Option<Vec<f32>>,
}

impl Model for UNet {
    type Output = UNetResponse;

    fn load(
        file: &std::path::Path,
        config: &super::EngineConfig,
    ) -> Result<Self, crate::errors::NnError> {
        todo!()
    }

    fn infer(image: super::Image<'_>) -> Result<Self::Output, crate::errors::NnError> {
        todo!()
    }
}
