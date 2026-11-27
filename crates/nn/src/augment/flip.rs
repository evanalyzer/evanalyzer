//! # flip
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-11-27
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.
//!
//! Mirroring: left-right and/or top-bottom. Like the rotations it only
//! reorders pixels - lossless, labels stay exact. A mirrored cell is as real
//! as the original (a microscope image has no handedness).
//!
//! Together with [`Rotation`](super::rotate::Rotation) a random flip gives
//! all 8 orientations of a square equally often: both mirrors at once are a
//! 180° turn, so half of the draws mirror and half don't - and a mirror
//! combined with a random rotation is each of the 4 mirror images alike.

use crate::train::SeededUniform;

/// Which ways to mirror. Both at once = a 180° rotation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(crate) struct Flip {
    /// Left-right: x becomes `width - 1 - x`.
    pub(crate) horizontal: bool,
    /// Top-bottom: y becomes `height - 1 - y`.
    pub(crate) vertical: bool,
}

impl Flip {
    /// Each mirror on or off with 50 %, independently.
    pub(crate) fn random(random: &mut SeededUniform) -> Self {
        Self {
            horizontal: random.index(2) == 1,
            vertical: random.index(2) == 1,
        }
    }

    /// One plane of `width x height` values (row-major), mirrored - pixels of
    /// any kind: intensities, object ids, classes, mask values. The size
    /// stays.
    pub(crate) fn plane<T: Copy>(self, data: &[T], width: usize, height: usize) -> Vec<T> {
        debug_assert_eq!(data.len(), width * height);
        let mut out = Vec::with_capacity(data.len());
        for y in 0..height {
            let from_y = if self.vertical { height - 1 - y } else { y };
            let row = &data[from_y * width..(from_y + 1) * width];
            if self.horizontal {
                out.extend(row.iter().rev());
            } else {
                out.extend_from_slice(row);
            }
        }
        out
    }

    /// `channels` planes of `width x height` one after the other (an image),
    /// each mirrored the same way.
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
    /// image, where it lies after the flip - on the pixel that
    /// [`Self::plane`] moves its pixel to.
    pub(crate) fn point(self, x: f32, y: f32, width: usize, height: usize) -> (f32, f32) {
        let x = if self.horizontal { width as f32 - x } else { x };
        let y = if self.vertical { height as f32 - y } else { y };
        (x, y)
    }
}

#[cfg(test)]
mod tests {
    use super::Flip;
    use crate::augment::rotate::Rotation;
    use crate::train::SeededUniform;
    use std::collections::HashMap;

    // 3 x 2:      0 1 2
    //             3 4 5
    const IMAGE: [u32; 6] = [0, 1, 2, 3, 4, 5];

    const HORIZONTAL: Flip = Flip {
        horizontal: true,
        vertical: false,
    };
    const VERTICAL: Flip = Flip {
        horizontal: false,
        vertical: true,
    };
    const BOTH: Flip = Flip {
        horizontal: true,
        vertical: true,
    };
    const ALL: [Flip; 4] = [
        Flip {
            horizontal: false,
            vertical: false,
        },
        HORIZONTAL,
        VERTICAL,
        BOTH,
    ];

    #[test]
    fn horizontal_mirrors_left_and_right() {
        assert_eq!(HORIZONTAL.plane(&IMAGE, 3, 2), vec![2, 1, 0, 5, 4, 3]);
    }

    #[test]
    fn vertical_mirrors_top_and_bottom() {
        assert_eq!(VERTICAL.plane(&IMAGE, 3, 2), vec![3, 4, 5, 0, 1, 2]);
    }

    #[test]
    fn both_mirrors_are_a_half_turn() {
        assert_eq!(BOTH.plane(&IMAGE, 3, 2), Rotation::Half.plane(&IMAGE, 3, 2));
    }

    #[test]
    fn no_flip_keeps_the_image() {
        assert_eq!(Flip::default().plane(&IMAGE, 3, 2), IMAGE.to_vec());
    }

    #[test]
    fn mirroring_twice_is_the_original_again() {
        for flip in ALL {
            let twice = flip.plane(&flip.plane(&IMAGE, 3, 2), 3, 2);
            assert_eq!(twice, IMAGE.to_vec(), "{flip:?}");
        }
    }

    #[test]
    fn every_channel_is_mirrored_the_same_way() {
        // Two channels of 3 x 2: 0..6 and 10..16.
        let image: Vec<u32> = (0..6).chain(10..16).collect();
        assert_eq!(
            HORIZONTAL.planes(&image, 3, 2, 2),
            vec![2, 1, 0, 5, 4, 3, 12, 11, 10, 15, 14, 13]
        );
    }

    #[test]
    fn a_point_moves_with_its_pixel() {
        // A non-square 5 x 3 image; every pixel's value is its index.
        let (width, height) = (5, 3);
        let image: Vec<usize> = (0..width * height).collect();
        for flip in ALL {
            let flipped = flip.plane(&image, width, height);
            for y in 0..height {
                for x in 0..width {
                    let (px, py) = flip.point(x as f32 + 0.5, y as f32 + 0.5, width, height);
                    let moved_to = py.floor() as usize * width + px.floor() as usize;
                    assert_eq!(flipped[moved_to], y * width + x, "{flip:?} ({x}, {y})");
                }
            }
        }
    }

    #[test]
    fn a_random_flip_and_rotation_give_all_8_orientations_alike() {
        // A 2 x 2 image with 4 different values has 8 distinct orientations;
        // 8000 draws should hit each about 1000 times.
        let image = [0u8, 1, 2, 3];
        let mut random = SeededUniform::new(11);
        let mut seen: HashMap<Vec<u8>, usize> = HashMap::new();
        for _ in 0..8000 {
            let flipped = Flip::random(&mut random).plane(&image, 2, 2);
            let turned = Rotation::random(&mut random).plane(&flipped, 2, 2);
            *seen.entry(turned).or_default() += 1;
        }
        assert_eq!(seen.len(), 8, "{seen:?}");
        for (orientation, &count) in &seen {
            assert!((850..1150).contains(&count), "{orientation:?}: {count}");
        }
    }

    #[test]
    fn random_flips_are_all_four_and_reproducible() {
        let draw = |seed| {
            let mut random = SeededUniform::new(seed);
            (0..200)
                .map(|_| Flip::random(&mut random))
                .collect::<Vec<_>>()
        };
        let flips = draw(9);
        for flip in ALL {
            assert!(flips.contains(&flip), "{flip:?} never drawn");
        }
        assert_eq!(flips, draw(9));
    }
}
