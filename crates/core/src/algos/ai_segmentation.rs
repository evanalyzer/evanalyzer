pub mod cellpose;
pub(crate) mod model_cache;
pub mod pixel_classifier;
pub mod stardist;
#[cfg(test)]
pub(crate) mod test_support;
pub mod unet;
pub mod yolov5;

/// Nearest-neighbor resize of a label map (OpenCV's `INTER_NEAREST`, as used
/// by Cellpose and YOLOv5 to scale masks back up): target pixel `d` takes
/// source pixel `floor(d * src / dst)`.
pub(crate) fn resize_labels_nearest(
    labels: &[u32],
    src_width: usize,
    src_height: usize,
    dst_width: usize,
    dst_height: usize,
) -> Vec<u32> {
    let src_x: Vec<usize> = (0..dst_width)
        .map(|x| (x * src_width / dst_width).min(src_width - 1))
        .collect();
    let mut out = Vec::with_capacity(dst_width * dst_height);
    for y in 0..dst_height {
        let row = (y * src_height / dst_height).min(src_height - 1) * src_width;
        out.extend(src_x.iter().map(|&x| labels[row + x]));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resize_labels_nearest_matches_opencv_inter_nearest() {
        // 2x2 -> 4x3: column d takes floor(d * 2 / 4), row d floor(d * 2 / 3).
        let labels = vec![1, 2, 3, 4];
        let out = resize_labels_nearest(&labels, 2, 2, 4, 3);
        assert_eq!(out, vec![1, 1, 2, 2, 1, 1, 2, 2, 3, 3, 4, 4]);
        // Shrinking takes every other pixel.
        let out = resize_labels_nearest(&[1, 2, 3, 4], 4, 1, 2, 1);
        assert_eq!(out, vec![1, 3]);
    }
}
