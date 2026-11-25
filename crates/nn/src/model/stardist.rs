use crate::model::{Model, Pieces};

pub struct Stardist {
    /// How large a tile runs in pieces.
    pieces: Pieces,
}

pub struct StardistResponse {
    /// `grid_height * grid_width` probabilities, `n_rays` planes of ray
    /// distances of that size.
    prob: Vec<f32>,
    dist: Vec<f32>,
    grid_height: usize,
    grid_width: usize,
    n_rays: usize,
}

impl Model for Stardist {
    type Output = StardistResponse;

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
