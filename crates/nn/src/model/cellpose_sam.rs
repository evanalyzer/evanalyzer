use crate::model::{Model, stardist};

pub struct CellposeSam {
    /// The size the model runs at (the image is resized to it).
    net_width: usize,
    net_height: usize,
    /// Channels the model takes (at least 1).
    input_channels: i32,
    /// Copy the gray image into every channel instead of zero-filling.
    replicate_gray_channel: bool,
}

pub struct CellposeSamResponse {
    flow_y: Vec<f32>,
    flow_x: Vec<f32>,
    cell_prob: Vec<f32>,
}

impl Model for CellposeSam {
    type Output = CellposeSamResponse;

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
