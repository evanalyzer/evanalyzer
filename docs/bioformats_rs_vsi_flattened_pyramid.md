# `bioformats` crate: VSI/CellSens reader always flattens pyramid resolutions into series

- **Crate**: [`bioformats`](https://github.com/henriksson-lab/bioformats-rs) `0.1.8`
- **Affected reader**: `CellSensReader` (Olympus VSI / cellSens), `crates/formats/flim2.rs` in the crate source (yes — the VSI reader lives in a file named `flim2.rs`, not an `olympus`/`vsi`-named file)
- **Found while**: porting an app off the old Java Bio-Formats JNI binding onto this crate
- **Status**: **fixed** upstream, on [`joda01/bioformats-rs@flatten-pyramid`](https://github.com/joda01/bioformats-rs/tree/flatten-pyramid), commit `81518d0` ("fix: pyramid flatten for vsi"). The downstream app-side workaround this doc originally described has been reverted in favor of consuming that fork directly.
- **Correction (after the fix landed)**: re-testing against the fix revealed `test_image_slide_scanner_rgb.vsi` (this doc's original repro file) actually has **two separate ETS volumes**, not one two-level pyramid - after the fix it correctly reports `series_count: 2`, each with `resolution_count: 1`, which is the *correct* shape for that file, not a remaining bug. The root cause described below (and the fix) is still real and still needed: the *old* code's `format!("{filename} #{}", flattened_index + 1)` naming was identical whether "the next entry" meant "the next resolution level of this volume" or "the first level of the next volume", so a name like `"...#2"` alone can't distinguish the two cases - this file happened to be an instance of the latter, not the former. The reproduction below is left as originally written since it accurately shows the *symptom* (and the fix's series/resolution model change is correct either way); readers looking for a single-volume multi-resolution-level repro will need a different fixture than the one used here.

## Summary

For pyramidal VSI files (Olympus `.vsi` + external `.ets` tile volumes), `CellSensReader` reports every pyramid resolution level as a **separate top-level series** instead of as a resolution level (`resolution_count()`/`set_resolution()`) within one series. Every other pyramidal reader in the crate (plain pyramidal OME-TIFF, SVS, etc.) exposes true multi-resolution series via `resolution_count() > 1`; VSI is the exception.

## Reproduction

```rust
// Cargo.toml: bioformats = "0.1.8"
use bioformats::common::reader::FormatReader;

fn main() {
    let path = std::path::Path::new("test_image_slide_scanner_rgb.vsi");
    let mut reader = bioformats::registry::open_reader_boxed(path).unwrap();

    println!("series_count: {}", reader.series_count());
    for s in 0..reader.series_count() {
        reader.set_series(s).unwrap();
        let m = reader.metadata();
        let name = reader
            .ome_metadata()
            .and_then(|o| o.images.get(s).and_then(|i| i.name.clone()));
        println!(
            "series {s}: name={:?} w={} h={} resolution_count={}",
            name, m.size_x, m.size_y, reader.resolution_count()
        );
    }
}
```

### Actual output

```
series_count: 2
series 0: name="test_image_slide_scanner_rgb.vsi"     w=512  h=375  resolution_count=1
series 1: name="test_image_slide_scanner_rgb.vsi #2"  w=4096 h=3000 resolution_count=1
```

### Expected output

One series covering the full pyramid:

```
series_count: 1
series 0: name="test_image_slide_scanner_rgb.vsi"  resolution_count=2
  resolution 0: w=4096 h=3000  (full res)
  resolution 1: w=512  h=375   (downsampled level)
```

(The `#2` suffix and swapped ordering above are exactly the naming scheme the reader itself generates for flattened levels — see below.)

## Root cause

`CellSensReader::resolution_count`/`set_resolution` are hard-coded:

```rust
// Flattened resolutions: every logical series is a single resolution level.
fn resolution_count(&self) -> usize {
    1
}
fn set_resolution(&mut self, level: usize) -> Result<()> {
    if level == 0 {
        Ok(())
    } else {
        Err(BioFormatsError::PlaneOutOfRange(level as u32))
    }
}
```

And in `set_id`, when building the flattened series list from a parsed `.ets` pyramid, the code's own comment states this is intentional:

```rust
// Build the flattened logical-series ordering. Mirrors Java with
// setFlattenedResolutions(true): each ETS pyramid resolution level is a
// distinct series, followed by one embedded TIFF image (the overview, the
// first IFD of the .vsi). ...
...
for (vi, vol) in self.ets.iter().enumerate() {
    for res in 0..vol.levels.len() {
        self.series_map.push(CellSensTarget::Ets { volume: vi, resolution: res });
        // Image 0 of the first pyramid takes the pyramid (stack) name;
        // later resolution levels get the default "filename #N" ...
        ...
        self.series_names.push(format!("{filename} #{}", series_idx + 1));
        ...
    }
}
```

Java Bio-Formats' `CellSensReader` supports **both** modes via `IFormatReader.setFlattenedResolutions(bool)` — `true` (flattened, what this port implements) and `false` (true nested pyramid, what most other apps built on Bio-Formats actually want and what this crate's own generic pyramidal-TIFF reader already does correctly via `resolution_count()`/`set_resolution()`). This Rust port only implements the `true` behavior, with no way to opt into the `false` behavior, and no `FormatReader`-level flag to select it either.

## Suggested fix

`CellSensTarget::Ets { volume, resolution }` already carries everything needed to group by `volume` instead of flattening by `(volume, resolution)`. The fix is essentially:

- One series per `volume` (`self.ets[vi]`), not one series per `(volume, resolution)`.
- `resolution_count()` returns `self.ets[current_volume].levels.len()`.
- `set_resolution(level)` re-derives `ets_meta` the same way `set_series` currently does for `CellSensTarget::Ets { resolution, .. }`, just keyed by the explicit `level` argument instead of a flattened series index.
- The one embedded-TIFF "macro image" overview stays a separate series (that part is correct — it's a genuinely different image, not a pyramid level, and isn't part of this issue).

This would bring VSI in line with every other pyramidal reader in the crate and match the `FormatReader` trait's existing `resolution_count`/`set_resolution` contract, which the crate already documents/uses correctly elsewhere.

## Impact

Without this, an app that expects "one series, multiple resolutions" for viewport LOD switching on whole-slide VSI images instead sees many series, each a full standalone image at one zoom level — no automatic downsampled preview, and picking the wrong "series" means loading full-resolution pixel data unnecessarily.

## Related gaps found in the same reader (same file, different symptoms — already worked around downstream, noted here for completeness)

Found while testing `G7_03.vsi` (3-channel fluorescence, no pyramid) — not the same bug as above, but the same reader:

1. **OME channel list shorter than the declared channel count.** `size_c=3`/`image_count=3` (3 real, non-RGB channels), but `reader.ome_metadata()`'s channel list has only 1 entry. Anything trusting `ome_image.channels.len()` to size a per-channel structure silently drops channels above the first.
2. **Channel `name`/`emission_wavelength` never populated**, even for the one channel that is listed. The real values (channel names `CY7`/`CY5`/`FITC`, wavelengths 767/670/518 nm) are present in `ImageMetadata.series_metadata` under vendor-prefixed keys (`cellsens.ets.channel_name.{i}`, `cellsens.ets.channel_wavelength.{i}` — nm already, no unit conversion needed) instead of the standard `OmeChannel` fields. The reader's own FV1000/OIF path (same file, different struct) does populate these correctly, for contrast.
