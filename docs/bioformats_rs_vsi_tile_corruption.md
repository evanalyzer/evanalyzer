# `bioformats` crate: VSI/CellSens full-resolution (ETS) tile decoding produces repeated/corrupted pixel data

- **Crate**: [`bioformats`](https://github.com/henriksson-lab/bioformats-rs) `0.1.8`, and confirmed still present on [`joda01/bioformats-rs@flatten-pyramid`](https://github.com/joda01/bioformats-rs/tree/flatten-pyramid) commit `81518d0` (which fixes the separate pyramid-flattening issue in `docs/bioformats_rs_vsi_flattened_pyramid.md`, but doesn't touch tile decoding)
- **Affected reader**: `CellSensReader`'s `EtsVolume` (Olympus VSI / cellSens external `.ets` pyramid tile stores), `crates/formats/flim2.rs`
- **Found while**: verifying the fix for `docs/bioformats_rs_vsi_flattened_pyramid.md` against a real whole-slide VSI file
- **Severity**: high — full-resolution pixel data for VSI/CellSens files is visibly wrong, not just imprecise. Thumbnail/overview resolution levels are unaffected.
- **Status**: not fixed upstream at time of writing; no app-side workaround is possible (see "Why this can't be worked around downstream" below)

## Summary

Reading any region of a VSI file's full-resolution (`.ets`-backed) pyramid level returns pixel data with the source tile(s) visibly repeated across the requested region, instead of genuinely distinct stitched content. It looks like a stride/geometry mismatch somewhere in the per-tile decode or the tile-to-output copy math, not a "wrong tile selected" bug — position-dependent content *does* change across the full image, but each individual read still contains an internal repeat at a period much smaller than the requested region (and smaller than what any true tile-grid stitching bug alone would explain).

## Reproduction

Fixture: `tests/fixtures/test_image_slide_scanner_rgb.vsi` (checked into this app's repo — H&E-stained tissue, RGB, 8-bit, one `.ets` pyramid with 2 flattened levels; see the companion pyramid-flattening doc for how the two levels were originally exposed).

```rust
// Cargo.toml: bioformats = "0.1.8"
use bioformats::common::reader::FormatReader;

fn main() {
    let path = std::path::Path::new("test_image_slide_scanner_rgb.vsi");
    let mut reader = bioformats::registry::open_reader_boxed(path).unwrap();

    // Series 1 here is the full-resolution (4096x3000) flattened pyramid
    // level - see docs/bioformats_rs_vsi_flattened_pyramid.md for why it's
    // series 1 and not a resolution of series 0.
    reader.set_series(1).unwrap();

    // Any region with real tissue reproduces this - e.g.:
    let region = reader.open_bytes_region(0, 0, 1200, 512, 512).unwrap();
    // `region` visibly repeats a small source patch ~9 times in a 3x3 grid
    // when rendered, instead of holding 512x512 of distinct content.

    // The corruption isn't only at the 512x512 assembly level: reading a
    // single tile-sized region (170x170, picked to roughly match the
    // visible repeat period above) *already* contains a finer internal
    // repeat, and two adjacent 170x170 reads *do* differ from each other -
    // so this isn't simply "the same tile is returned for every position",
    // it's wrong within one tile's own decode/copy.
    let _tile_a = reader.open_bytes_region(0, 1800, 1200, 170, 170).unwrap();
    let _tile_b = reader.open_bytes_region(0, 1970, 1200, 170, 170).unwrap(); // one tile right
}
```

### What a 512x512 read from real tissue actually looks like

Rendered as RGB (via the `image` crate, from `evanalyzer_core::ImageReader`'s decode of the exact same bytes `open_bytes_region` returns): a small patch of tissue, tiled 3x3 across the 512x512 output. The thumbnail/overview resolution level of the same file, by contrast, renders as a completely clean, artifact-free H&E section.

## What was ruled out

- **Not a pyramid-level mixup.** This reproduces on a single `open_bytes_region` call to one series/resolution - it isn't related to the flattened-series issue in the companion doc.
- **Not our downstream decode.** `evanalyzer_core`'s own byte-decode path (endianness/bit-depth/interleave handling) is verified byte-exact against real Java Bio-Formats reference output for a different, non-VSI fixture (`multi-channel-4D-series.ome.tif`, 30 passing regression tests including direct `.raw` comparisons) - it isn't touched or reconfigured differently for VSI, so the bug is upstream of it, in the bytes `open_bytes_region` itself returns.
- **Not "always the same tile returned".** Two adjacent tile-sized reads return visibly different content at the coarse scale, so `find_tile`'s row/col lookup does appear to select different source tiles for different requests. The repeat is *within* what one call returns, at a finer period than a single expected tile.

## Where to look

`CellSensReader::open_bytes_region` (`src/formats/flim2.rs`, `impl FormatReader for CellSensReader`) delegates the `CellSensTarget::Ets` case straight to:

```rust
CellSensTarget::Ets { volume, resolution } => {
    ...
    vol.assemble_region(resolution, z, c, t, x, y, w, h)
}
```

`EtsVolume::assemble_region` (same file, `impl EtsVolume`) walks the tile grid, and for each tile intersecting the request calls `self.decode_tile(resolution, row, col, z, c, t)`, then copies rows out of the decoded tile using a manually computed `src_stride = self.tile_x as usize * pixel`:

```rust
let src_stride = self.tile_x as usize * pixel;
for copy_row in 0..(oy1 - oy0) as usize {
    let src = (src_y + copy_row) * src_stride + src_x * pixel;
    let dst = (dst_y + copy_row) * out_row_len + dst_x * pixel;
    if src + copy_len <= tile.len() && dst + copy_len <= out.len() {
        out[dst..dst + copy_len].copy_from_slice(&tile[src..src + copy_len]);
    }
}
```

`decode_tile` itself dispatches by codec (`ETS_RAW`/`ETS_JPEG`/`ETS_JPEG_2000`/`ETS_PNG`/`ETS_BMP`) and always resizes/truncates the decoded buffer to exactly `self.tile_size()` bytes before returning:

```rust
if buf.len() < tile_size {
    buf.resize(tile_size, 0);
} else if buf.len() > tile_size {
    buf.truncate(tile_size);
}
```

Given the internal-repeat symptom (correct content, wrong period), the most likely places for the actual defect:

1. `self.tile_size()`/`self.tile_x`/`self.tile_y` not matching the *decoded* dimensions of a compressed (JPEG/JPEG2000) tile for this codec/level - if the decoder returns a tile at a different width than `tile_x` assumes, `src_stride` above would misalign every row after the first, and a naive `resize`/`truncate` to `tile_size` bytes (rather than to `tile_x`-wide rows) would silently accept the wrong byte count without erroring - producing exactly a fine-grained, position-shifted repeat rather than a clean crop or a hard failure.
2. `find_tile`'s resolution/row/col -> byte-offset lookup for JPEG-family codecs, if this file's tiles are compressed at a different resolution than `resolution` implies.

This wasn't narrowed further without deeper access to the `.ets` chunk-table structures for this specific file; flagged here as the two most likely spots for whoever fixes it.

## Which other formats were checked

Asked to check whether this is VSI-specific or affects other tile-based formats in the crate. The crate has (at least) three **independently implemented** tile-stitching code paths, not one shared implementation - a bug in one doesn't imply the same bug in another:

| Format | Tile-stitching code | Empirically verified here? |
|---|---|---|
| Olympus VSI/CellSens (`.ets`) | `EtsVolume::assemble_region`/`decode_tile` in `flim2.rs` | **Yes - confirmed broken** (this doc) |
| Ventana/BIF | `VentanaReader::assemble_region` in `tiff_wrappers.rs` - a separate, independently-written implementation with its own AOI-clipping/resolution-scaling geometry | No - no Ventana/BIF fixture available to test |
| TissueFaxs (HCS) | `decode_tile` in `hcs2.rs` - a much simpler per-tile codec dispatch, not a full stride-based stitcher | No - no fixture available to test |
| Plain pyramidal OME-TIFF, SVS, NDPI, Leica SCN (SubIFD-based pyramids) | Shared generic code in `tiff/reader.rs` (`resolve_ifd_index_at` + `get_samples`), not touched by any of the three format-specific implementations above | Indirectly - the *non-pyramidal* path through this same generic TIFF reader passes 30 byte-exact regression tests against real Java Bio-Formats reference output in this app's own test suite (`multi-channel-4D-series.ome.tif`). The multi-resolution/SubIFD branch of the same reader wasn't separately exercised against a real pyramidal fixture. |

**Conclusion**: confirmed broken for VSI/CellSens only. No evidence either way for Ventana/TissueFaxs (no fixtures to test). The generic SubIFD pyramid path used by several other formats is architecturally unrelated code, with at least partial (non-pyramidal) empirical verification already in place, so it's a much lower suspicion than the three format-specific hand-rolled stitchers - but this isn't a substitute for a real pyramidal fixture test against one of those formats.

## Why this can't be worked around downstream

The pyramid-flattening issue (companion doc) was a **metadata** problem: the crate reported correct pixel data, just organized into the wrong series/resolution shape, so the app could detect and re-map it without touching any bytes. This is a **pixel data** problem: `open_bytes_region` itself returns the wrong bytes for the region requested. There is no metadata-level signal to detect or correct for this from the calling app - it needs a fix in the tile decode/stitch itself.
