//! # channel_shift
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-11-27
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.
//!
//! Channels shifted against each other by a few whole pixels: real channels
//! are never perfectly aligned (a filter change, chromatic shift, a moving
//! sample). Trained with this the network learns a tolerant comparison of
//! its channels - e.g. "a peak in the spot channel without a structure
//! within a few pixels of the reference" - instead of a pixel-exact one,
//! which breaks as soon as the channels are 1 px apart.
//!
//! Channel 0 stays where it is - the labels belong to it, so put the channel
//! the annotations were made on first (SpotNet: the spot channel). Every
//! other channel gets its own random shift. The input only: labels are never
//! touched.
//!
//! Whole pixels: lossless, nothing interpolated. Pixels shifted in from
//! outside the patch repeat its border (no black stripe a network could
//! learn as an edge).

use crate::train::SeededUniform;

/// One shift `(dx, dy)` per channel, in px; channel 0 always `(0, 0)`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ChannelShift {
    shifts: Vec<(isize, isize)>,
}

impl ChannelShift {
    /// For every channel but the first, `dx` and `dy` each drawn on their
    /// own from `-max..=max` (every whole value equally likely, 0 included).
    pub(crate) fn random(channels: usize, max: usize, random: &mut SeededUniform) -> Self {
        let mut draw = || random.index(2 * max + 1) as isize - max as isize;
        let shifts = (0..channels)
            .map(|c| if c == 0 { (0, 0) } else { (draw(), draw()) })
            .collect();
        Self { shifts }
    }

    #[cfg(test)]
    /// The shift of each channel: its content moves `dx` px to the right and
    /// `dy` px down.
    pub(crate) fn shifts(&self) -> &[(isize, isize)] {
        &self.shifts
    }

    /// `data` (one `width x height` plane per channel, one after the other)
    /// with every channel moved by its shift, in place.
    pub(crate) fn apply(&self, data: &mut [f32], width: usize, height: usize) {
        debug_assert_eq!(data.len(), self.shifts.len() * width * height);
        let mut source = Vec::new();
        for (plane, &(dx, dy)) in data.chunks_mut(width * height).zip(&self.shifts) {
            if (dx, dy) == (0, 0) {
                continue;
            }
            source.clear();
            source.extend_from_slice(plane);
            // A pixel of the result shows the source pixel `(dx, dy)` before
            // it, clamped into the patch: the border repeats.
            let from = |v: usize, d: isize, len: usize| {
                (v as isize - d).clamp(0, len as isize - 1) as usize
            };
            for y in 0..height {
                let sy = from(y, dy, height);
                for x in 0..width {
                    plane[y * width + x] = source[sy * width + from(x, dx, width)];
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::ChannelShift;
    use crate::train::SeededUniform;

    // Two channels of 4 x 2:   0  1  2  3      10 11 12 13
    //                          4  5  6  7      14 15 16 17
    fn image() -> Vec<f32> {
        (0..8).chain(10..18).map(|v| v as f32).collect()
    }

    #[test]
    fn the_second_channel_moves_right_and_its_border_repeats() {
        let shift = ChannelShift {
            shifts: vec![(0, 0), (1, 0)],
        };
        let mut data = image();
        shift.apply(&mut data, 4, 2);
        let expected: Vec<f32> = [0, 1, 2, 3, 4, 5, 6, 7, 10, 10, 11, 12, 14, 14, 15, 16]
            .iter()
            .map(|&v| v as f32)
            .collect();
        assert_eq!(data, expected);
    }

    #[test]
    fn a_shift_up_and_left_moves_the_content_up_and_left() {
        let shift = ChannelShift {
            shifts: vec![(0, 0), (-2, -1)],
        };
        let mut data = image();
        shift.apply(&mut data, 4, 2);
        // Row 0 shows row 1 two pixels further right (border repeated).
        assert_eq!(
            &data[8..],
            &[16.0, 17.0, 17.0, 17.0, 16.0, 17.0, 17.0, 17.0]
        );
    }

    #[test]
    fn the_first_channel_never_moves() {
        let mut random = SeededUniform::new(2);
        for _ in 0..100 {
            let shift = ChannelShift::random(3, 3, &mut random);
            assert_eq!(shift.shifts()[0], (0, 0));
        }
    }

    #[test]
    fn a_single_channel_stays_as_it_is() {
        let shift = ChannelShift::random(1, 3, &mut SeededUniform::new(1));
        let mut data: Vec<f32> = (0..8).map(|v| v as f32).collect();
        shift.apply(&mut data, 4, 2);
        assert_eq!(data, (0..8).map(|v| v as f32).collect::<Vec<_>>());
    }

    #[test]
    fn a_maximum_of_zero_shifts_nothing() {
        let shift = ChannelShift::random(2, 0, &mut SeededUniform::new(1));
        assert_eq!(shift.shifts(), &[(0, 0), (0, 0)]);
    }

    #[test]
    fn every_shift_up_to_the_maximum_is_drawn_and_none_beyond() {
        let mut random = SeededUniform::new(6);
        let mut seen = std::collections::HashSet::new();
        for _ in 0..2000 {
            let (dx, dy) = ChannelShift::random(2, 2, &mut random).shifts()[1];
            assert!(dx.abs() <= 2 && dy.abs() <= 2, "({dx}, {dy})");
            seen.insert((dx, dy));
        }
        assert_eq!(seen.len(), 25, "all of -2..=2 x -2..=2");
    }

    #[test]
    fn the_same_seed_draws_the_same_shifts() {
        let draw = |seed| ChannelShift::random(3, 3, &mut SeededUniform::new(seed));
        assert_eq!(draw(4), draw(4));
        assert_ne!(draw(4), draw(5));
    }
}
