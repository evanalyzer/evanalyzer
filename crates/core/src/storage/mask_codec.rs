//! # Object masks in the results
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-10-23
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **AGPL-3.0**.
//!
//! ## Overview
//! An object's mask (relative to its bounding box, row by row) as the
//! `objects.mask` BLOB: the lengths of its runs, alternating unset and set
//! pixels and starting with an unset run (0 when the first pixel is set),
//! each a LEB128 varint. A spot is a few bytes, a large solid region not
//! much more - bits per pixel would grow with its box.

use bitvec::prelude::*;

/// `mask` as run lengths (see the module docs).
pub fn encode_mask(mask: &BitSlice<u64, Lsb0>) -> Vec<u8> {
    let mut out = Vec::new();
    let mut pos = 0;
    let len = mask.len();
    while pos < len {
        let zeros = mask[pos..].first_one().unwrap_or(len - pos);
        push_varint(&mut out, zeros as u64);
        pos += zeros;
        if pos >= len {
            break;
        }
        let ones = mask[pos..].first_zero().unwrap_or(len - pos);
        push_varint(&mut out, ones as u64);
        pos += ones;
    }
    out
}

/// The mask of `len` pixels `bytes` holds - `None` if they don't fit it.
pub fn decode_mask(bytes: &[u8], len: usize) -> Option<BitVec<u64, Lsb0>> {
    let mut mask = BitVec::<u64, Lsb0>::repeat(false, len);
    let (mut pos, mut set, mut i) = (0usize, false, 0usize);
    while i < bytes.len() {
        let (run, used) = read_varint(&bytes[i..])?;
        i += used;
        let end = pos.checked_add(run as usize).filter(|end| *end <= len)?;
        if set {
            mask[pos..end].fill(true);
        }
        pos = end;
        set = !set;
    }
    Some(mask)
}

fn push_varint(out: &mut Vec<u8>, mut value: u64) {
    loop {
        let byte = (value & 0x7f) as u8;
        value >>= 7;
        if value == 0 {
            out.push(byte);
            return;
        }
        out.push(byte | 0x80);
    }
}

/// A varint at the start of `bytes`: its value and how many bytes it took.
fn read_varint(bytes: &[u8]) -> Option<(u64, usize)> {
    let mut value = 0u64;
    for (i, byte) in bytes.iter().enumerate().take(10) {
        value |= u64::from(byte & 0x7f) << (7 * i);
        if byte & 0x80 == 0 {
            return Some((value, i + 1));
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn masks_survive_a_round_trip() {
        let mut state = 0x2545_F491_u32;
        let mut next = move |n: u32| {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            state % n
        };
        for _ in 0..500 {
            let len = 1 + next(5000) as usize;
            let density = 1 + next(6);
            let runs = next(2) == 0;
            let mut mask = BitVec::<u64, Lsb0>::repeat(false, len);
            let mut on = false;
            for i in 0..len {
                // Long runs (like real objects) or scattered pixels.
                if runs {
                    if next(40) == 0 {
                        on = !on;
                    }
                    mask.set(i, on);
                } else {
                    mask.set(i, next(density) == 0);
                }
            }
            let bytes = encode_mask(&mask);
            assert_eq!(decode_mask(&bytes, len).as_ref(), Some(&mask), "len {len}");
        }
    }

    #[test]
    fn a_solid_mask_is_two_runs_and_an_empty_one_one() {
        let solid = BitVec::<u64, Lsb0>::repeat(true, 10_000);
        assert_eq!(encode_mask(&solid), [0, 0x90, 0x4e], "0 unset, 10000 set");
        let empty = BitVec::<u64, Lsb0>::repeat(false, 300);
        assert_eq!(encode_mask(&empty), [0xac, 0x02]);
        assert_eq!(encode_mask(&BitVec::<u64, Lsb0>::new()), Vec::<u8>::new());
    }

    #[test]
    fn bytes_that_dont_fit_the_mask_are_refused() {
        let solid = BitVec::<u64, Lsb0>::repeat(true, 100);
        let bytes = encode_mask(&solid);
        assert!(decode_mask(&bytes, 50).is_none(), "runs past its end");
        assert!(decode_mask(&[0x80], 10).is_none(), "an unfinished varint");
    }
}
