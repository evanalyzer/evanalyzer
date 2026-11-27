//! # rotate
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-11-27
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.
//!
//! Rotations by multiples of 90°: they only reorder pixels, so they are
//! lossless - labels stay exact, nothing is interpolated. A microscope image
//! has no "up", so every rotation is as real as the original.
//!
//! Image coordinates: x to the right, y down; a rotation turns clockwise as
//! seen on screen. A 90° or 270° rotation swaps width and height.

use crate::train::SeededUniform;

/// A rotation by a multiple of 90°, clockwise.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Rotation {
    None,
    Quarter,
    Half,
    ThreeQuarters,
}

impl Rotation {
    const ALL: [Self; 4] = [Self::None, Self::Quarter, Self::Half, Self::ThreeQuarters];

    /// One of the four, each equally likely.
    pub(crate) fn random(random: &mut SeededUniform) -> Self {
        Self::ALL[random.index(Self::ALL.len())]
    }

    /// Width and height after the rotation.
    pub(crate) fn size(self, width: usize, height: usize) -> (usize, usize) {
        match self {
            Self::None | Self::Half => (width, height),
            Self::Quarter | Self::ThreeQuarters => (height, width),
        }
    }

    /// One plane of `width x height` values (row-major), rotated - pixels
    /// of any kind: intensities, object ids, classes, mask values.
    pub(crate) fn plane<T: Copy>(self, data: &[T], width: usize, height: usize) -> Vec<T> {
        debug_assert_eq!(data.len(), width * height);
        if self == Self::None {
            return data.to_vec();
        }
        let (new_width, new_height) = self.size(width, height);
        let mut out = Vec::with_capacity(data.len());
        // For each pixel of the result, the pixel of `data` it comes from.
        for ny in 0..new_height {
            for nx in 0..new_width {
                let (x, y) = match self {
                    Self::None => (nx, ny),
                    Self::Quarter => (ny, height - 1 - nx),
                    Self::Half => (width - 1 - nx, height - 1 - ny),
                    Self::ThreeQuarters => (width - 1 - ny, nx),
                };
                out.push(data[y * width + x]);
            }
        }
        out
    }

    /// `channels` planes of `width x height` one after the other (an
    /// image), each rotated the same way.
    pub(crate) fn planes<T: Copy>(
        self,
        data: &[T],
        width: usize,
        height: usize,
        channels: usize,
    ) -> Vec<T> {
        debug_assert_eq!(data.len(), channels * width * height);
        data.chunks(width * height)
            .flat_map(|plane| self.plane(plane, width, height))
            .collect()
    }

    /// A point (in px, a pixel's centre at +0.5) of a `width x height`
    /// image, where it lies after the rotation - on the pixel that
    /// [`Self::plane`] moves its pixel to.
    pub(crate) fn point(self, x: f32, y: f32, width: usize, height: usize) -> (f32, f32) {
        let (w, h) = (width as f32, height as f32);
        match self {
            Self::None => (x, y),
            Self::Quarter => (h - y, x),
            Self::Half => (w - x, h - y),
            Self::ThreeQuarters => (y, w - x),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::Rotation;
    use crate::train::SeededUniform;

    // 3 x 2:      0 1 2
    //             3 4 5
    const IMAGE: [u32; 6] = [0, 1, 2, 3, 4, 5];

    #[test]
    fn a_quarter_turns_clockwise_and_swaps_width_and_height() {
        // 2 x 3:  3 0
        //         4 1
        //         5 2
        assert_eq!(Rotation::Quarter.size(3, 2), (2, 3));
        assert_eq!(
            Rotation::Quarter.plane(&IMAGE, 3, 2),
            vec![3, 0, 4, 1, 5, 2]
        );
    }

    #[test]
    fn a_half_turns_the_image_upside_down() {
        assert_eq!(Rotation::Half.size(3, 2), (3, 2));
        assert_eq!(Rotation::Half.plane(&IMAGE, 3, 2), vec![5, 4, 3, 2, 1, 0]);
    }

    #[test]
    fn three_quarters_turn_counter_clockwise() {
        // 2 x 3:  2 5
        //         1 4
        //         0 3
        assert_eq!(Rotation::ThreeQuarters.size(3, 2), (2, 3));
        assert_eq!(
            Rotation::ThreeQuarters.plane(&IMAGE, 3, 2),
            vec![2, 5, 1, 4, 0, 3]
        );
    }

    #[test]
    fn no_rotation_keeps_the_image() {
        assert_eq!(Rotation::None.plane(&IMAGE, 3, 2), IMAGE.to_vec());
    }

    #[test]
    fn four_quarters_are_the_original_again() {
        let (mut data, mut w, mut h) = (IMAGE.to_vec(), 3, 2);
        for _ in 0..4 {
            data = Rotation::Quarter.plane(&data, w, h);
            (w, h) = Rotation::Quarter.size(w, h);
        }
        assert_eq!((data, w, h), (IMAGE.to_vec(), 3, 2));
    }

    #[test]
    fn every_channel_is_rotated_the_same_way() {
        // Two channels of 3 x 2: 0..6 and 10..16.
        let image: Vec<u32> = (0..6).chain(10..16).collect();
        assert_eq!(
            Rotation::Quarter.planes(&image, 3, 2, 2),
            vec![3, 0, 4, 1, 5, 2, 13, 10, 14, 11, 15, 12]
        );
    }

    #[test]
    fn a_point_moves_with_its_pixel() {
        // A non-square 5 x 3 image; every pixel's value is its index. The
        // rotated centre of a pixel lies on the rotated pixel with its value.
        let (width, height) = (5, 3);
        let image: Vec<usize> = (0..width * height).collect();
        for rotation in Rotation::ALL {
            let rotated = rotation.plane(&image, width, height);
            let (new_width, _) = rotation.size(width, height);
            for y in 0..height {
                for x in 0..width {
                    let (px, py) = rotation.point(x as f32 + 0.5, y as f32 + 0.5, width, height);
                    let moved_to = py.floor() as usize * new_width + px.floor() as usize;
                    assert_eq!(rotated[moved_to], y * width + x, "{rotation:?} ({x}, {y})");
                }
            }
        }
    }

    #[test]
    fn random_rotations_are_all_four_and_reproducible() {
        let draw = |seed| {
            let mut random = SeededUniform::new(seed);
            (0..200)
                .map(|_| Rotation::random(&mut random))
                .collect::<Vec<_>>()
        };
        let rotations = draw(9);
        for rotation in Rotation::ALL {
            assert!(rotations.contains(&rotation), "{rotation:?} never drawn");
        }
        assert_eq!(rotations, draw(9));
    }
}
