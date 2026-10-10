//! Image work the UI does on pixels it already has.

use crate::api::ImageContainer;
use bitvec::prelude::{BitVec, Lsb0};
use evanalyzer_cfg::core_types::ObjectClass;
use evanalyzer_cfg::settings::object_settings::ObjectMetricSettings;
use evanalyzer_core::Object;
use kornia_image::ImageSize;
use std::sync::Arc;

/// Builds a manually annotated object from a painted `mask` (covering
/// `bbox` = `[x1, y1, x2, y2]` in full-image pixels), measuring its
/// intensities on every channel in `images`. `origin_image` is the channel
/// the user painted on; it provides the object's plane.
pub fn object_from_mask(
    full_image_width: usize,
    full_image_height: usize,
    mask: BitVec<u64, Lsb0>,
    bbox: [u32; 4],
    origin_image: &ImageContainer,
    images: &[(i32, Arc<ImageContainer>)],
    object_class: ObjectClass,
) -> ObjectMetricSettings {
    Object::from_mask(
        &ImageSize {
            width: full_image_width,
            height: full_image_height,
        },
        mask,
        bbox,
        origin_image,
        images,
        object_class,
    )
    .to_object_settings()
}
