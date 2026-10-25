//! # make_test_slide
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-10-15
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.

// Writes a synthetic whole slide image for the viewport benchmarks
// (`bench_viewport_*` tests in `viewport_worker.rs`): an 8-bit RGB,
// brightfield-like slide stored like a real scanner file - a tiled BigTIFF,
// 512 x 512 JPEG tiles, with a SubIFD pyramid (each level half the size of
// the one above, down to below 1024 px). `--flat` writes only the full
// resolution (no pyramid), as plain big TIFFs often are.
//
// The pixels are computed per tile from their position (pink tissue with
// purple nuclei on a white background, plus noise), so the image never has to
// fit in RAM and JPEG sizes/decode cost resemble a real H&E slide.
//
// Usage: cargo run --release -p evanalyzer_gui --example make_test_slide -- <out.tif> [--width W] [--height H] [--flat]

use image::ImageEncoder;
use image::codecs::jpeg::JpegEncoder;
use rayon::prelude::*;
use std::fs::File;
use std::io::{BufWriter, Seek, SeekFrom, Write};

const TILE: u32 = 512;

fn arg_value(name: &str, default: u32) -> u32 {
    let args: Vec<String> = std::env::args().collect();
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1))
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

/// Cheap deterministic hash noise in 0..1.
fn hash(x: u32, y: u32) -> f32 {
    let mut h = x.wrapping_mul(0x9E37_79B1) ^ y.wrapping_mul(0x85EB_CA77);
    h ^= h >> 15;
    h = h.wrapping_mul(0x2C1B_3C6D);
    h ^= h >> 12;
    (h & 0xFFFF) as f32 / 65535.0
}

/// The colour at full-resolution position (`x`, `y`).
fn pixel(x: f32, y: f32) -> [u8; 3] {
    // Tissue: smooth blobs where the sum of a few waves is high.
    let t = (x * 0.00021).sin() * (y * 0.00017).cos()
        + 0.6 * (x * 0.00063 + y * 0.00041).sin()
        + 0.3 * (y * 0.0013).sin();
    let noise = hash(x as u32, y as u32);
    if t < 0.15 {
        let v = 236.0 + noise * 12.0;
        return [v as u8, v as u8, (v - 2.0) as u8];
    }
    // Nuclei: a jittered 40 px grid of discs.
    let (cx, cy) = ((x / 40.0).floor(), (y / 40.0).floor());
    let jx = hash(cx as u32, cy as u32 + 7) * 20.0 + 10.0;
    let jy = hash(cx as u32 + 3, cy as u32) * 20.0 + 10.0;
    let (dx, dy) = (x - (cx * 40.0 + jx), y - (cy * 40.0 + jy));
    if dx * dx + dy * dy < 64.0 {
        return [
            (95.0 + noise * 30.0) as u8,
            (60.0 + noise * 20.0) as u8,
            (140.0 + noise * 30.0) as u8,
        ];
    }
    [
        (225.0 - noise * 25.0) as u8,
        (150.0 - noise * 30.0) as u8,
        (190.0 - noise * 20.0) as u8,
    ]
}

/// One JPEG tile of a level whose pixels are `scale` full-resolution
/// pixels wide. Edge tiles are padded to full tile size, as TIFF requires.
fn encode_tile(tx: u32, ty: u32, scale: f32, level_w: u32, level_h: u32) -> Vec<u8> {
    let mut rgb = vec![255u8; (TILE * TILE * 3) as usize];
    for y in 0..TILE {
        let ly = ty * TILE + y;
        if ly >= level_h {
            break;
        }
        for x in 0..TILE {
            let lx = tx * TILE + x;
            if lx >= level_w {
                break;
            }
            let p = pixel(lx as f32 * scale, ly as f32 * scale);
            let i = ((y * TILE + x) * 3) as usize;
            rgb[i..i + 3].copy_from_slice(&p);
        }
    }
    let mut out = Vec::new();
    JpegEncoder::new_with_quality(&mut out, 85)
        .write_image(&rgb, TILE, TILE, image::ExtendedColorType::Rgb8)
        .expect("JPEG encoding");
    out
}

/// A BigTIFF IFD entry: (tag, type, count, inline value or offset).
type Entry = (u16, u16, u64, u64);
const SHORT: u16 = 3;
const LONG: u16 = 4;
const LONG8: u16 = 16;
const IFD8: u16 = 18;

/// Writes `entries` (sorted by tag) as an IFD at the current position and
/// returns its offset. Arrays longer than 8 bytes must already be written,
/// with their offset as the value.
fn write_ifd(w: &mut BufWriter<File>, mut entries: Vec<Entry>, next: u64) -> u64 {
    entries.sort_by_key(|e| e.0);
    let at = w.stream_position().unwrap();
    w.write_all(&(entries.len() as u64).to_le_bytes()).unwrap();
    for (tag, typ, count, value) in entries {
        w.write_all(&tag.to_le_bytes()).unwrap();
        w.write_all(&typ.to_le_bytes()).unwrap();
        w.write_all(&count.to_le_bytes()).unwrap();
        w.write_all(&value.to_le_bytes()).unwrap();
    }
    w.write_all(&next.to_le_bytes()).unwrap();
    at
}

fn write_u64s(w: &mut BufWriter<File>, values: &[u64]) -> u64 {
    let at = w.stream_position().unwrap();
    for v in values {
        w.write_all(&v.to_le_bytes()).unwrap();
    }
    at
}

/// Writes one level's tiles and returns the IFD entries describing it.
fn write_level(
    w: &mut BufWriter<File>,
    level_w: u32,
    level_h: u32,
    scale: f32,
    reduced: bool,
) -> Vec<Entry> {
    let (nx, ny) = (level_w.div_ceil(TILE), level_h.div_ceil(TILE));
    let mut offsets = Vec::with_capacity((nx * ny) as usize);
    let mut counts = Vec::with_capacity((nx * ny) as usize);
    // Encode a row of tiles in parallel, write them in order.
    for ty in 0..ny {
        let row: Vec<Vec<u8>> = (0..nx)
            .into_par_iter()
            .map(|tx| encode_tile(tx, ty, scale, level_w, level_h))
            .collect();
        for tile in row {
            offsets.push(w.stream_position().unwrap());
            counts.push(tile.len() as u64);
            w.write_all(&tile).unwrap();
        }
    }
    let offsets_at = write_u64s(w, &offsets);
    let counts_at = write_u64s(w, &counts);
    // BitsPerSample 8,8,8 fits inline (3 x u16).
    let bps = 8u64 | (8u64 << 16) | (8u64 << 32);
    vec![
        (254, LONG, 1, reduced as u64),
        (256, LONG, 1, level_w as u64),
        (257, LONG, 1, level_h as u64),
        (258, SHORT, 3, bps),
        (259, SHORT, 1, 7), // JPEG
        (262, SHORT, 1, 2), // RGB
        (277, SHORT, 1, 3),
        (284, SHORT, 1, 1),
        (322, LONG, 1, TILE as u64),
        (323, LONG, 1, TILE as u64),
        (324, LONG8, offsets.len() as u64, offsets_at),
        (325, LONG8, counts.len() as u64, counts_at),
    ]
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let out = args.get(1).expect("output path").clone();
    let width = arg_value("--width", 60_000);
    let height = arg_value("--height", 40_000);
    let flat = args.iter().any(|a| a == "--flat");

    let mut w = BufWriter::new(File::create(&out).expect("create output"));
    // BigTIFF header: "II", 43, offset size 8, 0, first IFD (patched below).
    w.write_all(b"II").unwrap();
    w.write_all(&43u16.to_le_bytes()).unwrap();
    w.write_all(&8u16.to_le_bytes()).unwrap();
    w.write_all(&0u16.to_le_bytes()).unwrap();
    w.write_all(&0u64.to_le_bytes()).unwrap();

    let start = std::time::Instant::now();
    let mut level0 = write_level(&mut w, width, height, 1.0, false);

    // Reduced levels, each an IFD of its own, linked from level 0's SubIFDs.
    let mut sub_ifds = Vec::new();
    if !flat {
        let (mut lw, mut lh, mut scale) = (width, height, 1.0f32);
        while lw >= 1024 || lh >= 1024 {
            lw = lw.div_ceil(2);
            lh = lh.div_ceil(2);
            scale *= 2.0;
            let entries = write_level(&mut w, lw, lh, scale, true);
            sub_ifds.push(write_ifd(&mut w, entries, 0));
            eprintln!("level {} written: {lw} x {lh}", sub_ifds.len());
        }
    }
    if !sub_ifds.is_empty() {
        let at = write_u64s(&mut w, &sub_ifds);
        level0.push((
            330,
            IFD8,
            sub_ifds.len() as u64,
            if sub_ifds.len() == 1 { sub_ifds[0] } else { at },
        ));
    }
    let first_ifd = write_ifd(&mut w, level0, 0);
    w.seek(SeekFrom::Start(8)).unwrap();
    w.write_all(&first_ifd.to_le_bytes()).unwrap();
    w.flush().unwrap();

    let size = std::fs::metadata(&out).unwrap().len();
    eprintln!(
        "{out}: {width} x {height} RGB, {} levels, {:.0} MB, written in {:.1?}",
        sub_ifds.len() + 1,
        size as f64 / 1e6,
        start.elapsed()
    );
}
