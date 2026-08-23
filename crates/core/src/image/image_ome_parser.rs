use crate::image::image_meta::{
    ChannelInfo, ImageInfo, ImageMeta, Objective, PixelSizes, PyramidInfo,
};
use crate::image::image_reader::ReadMode;
use bioformats::common::metadata::{ImageMetadata, MetadataValue};
use bioformats::common::reader::FormatReader;
use evanalyzer_cfg::core_types::InternalErrors;
use std::collections::HashMap;
use std::path::Path;

/// Physical pixel sizes on [`bioformats::common::ome_metadata::OmeImage`] are
/// always reported in micrometres (see that struct's doc comments); the rest
/// of this app stores lengths in nanometres.
const MICROMETER_TO_NANOMETER: f32 = 1_000.0;

fn bf_err(e: bioformats::common::error::BioFormatsError) -> InternalErrors {
    InternalErrors::ImageReadError(e.to_string())
}

/// Samples packed into one pixel of the *current* series/resolution - 1 for
/// a plain (grayscale or multi-channel) series, otherwise however many
/// samples `is_rgb` packs together (almost always 3). Mirrors Java
/// Bio-Formats' `IFormatReader.getRGBChannelCount()`; the `bioformats` crate
/// computes the same value internally (see `wrappers::ChannelSeparator`) but
/// doesn't expose it publicly, so it's re-derived here from the documented
/// `ImageMetadata` fields.
fn rgb_channel_count(meta: &ImageMetadata) -> u32 {
    if !meta.is_rgb {
        return 1;
    }
    let zt = meta.size_z.max(1).saturating_mul(meta.size_t.max(1));
    if zt > 0 && meta.image_count >= zt {
        let effective_c = (meta.image_count / zt).max(1);
        if effective_c > 0 && meta.size_c >= effective_c && meta.size_c % effective_c == 0 {
            return (meta.size_c / effective_c).max(1);
        }
    }
    meta.size_c.max(1)
}

/// The number of *distinct* channel planes/stacks, excluding RGB-packed
/// samples that live inside one plane together. Mirrors Java's
/// `getEffectiveSizeC()` - see [`rgb_channel_count`]. `pub(crate)` since
/// `image_reader.rs` needs the same value to compute a plane index for
/// `open_bytes_region`.
pub(crate) fn effective_size_c(meta: &ImageMetadata) -> u32 {
    if meta.is_rgb {
        (meta.size_c / rgb_channel_count(meta)).max(1)
    } else {
        meta.size_c.max(1)
    }
}

/// Reads `key` from `series_metadata` as a display string, regardless of
/// which [`MetadataValue`] variant it was stored as.
fn metadata_string(series_metadata: &HashMap<String, MetadataValue>, key: &str) -> Option<String> {
    Some(series_metadata.get(key)?.to_string())
}

/// Reads `key` from `series_metadata` as an `f64`, regardless of whether the
/// reader stored it as a genuine float or (as some readers do for
/// less-common tags) a plain string.
fn metadata_f64(series_metadata: &HashMap<String, MetadataValue>, key: &str) -> Option<f64> {
    match series_metadata.get(key)? {
        MetadataValue::Float(f) => Some(*f),
        MetadataValue::Int(i) => Some(*i as f64),
        MetadataValue::String(s) => s.parse().ok(),
        MetadataValue::Bool(_) | MetadataValue::Bytes(_) => None,
    }
}

/// Builds this app's [`ImageMeta`] by walking every series/resolution of an
/// already-opened `bioformats` reader (`reader.set_id`/`ImageReader::open`
/// must have already run - this only reads state, it never opens a file).
///
/// Replaces the old hand-rolled OME-XML string parser: `bioformats` already
/// parses every supported format's metadata into typed Rust structs
/// (`ImageMetadata`/`OmeMetadata`), so there is no XML left to parse here -
/// this is now a straight field-by-field mapping from the crate's model to
/// this app's own `ImageMeta`/`ImageInfo`/`PyramidInfo` shape.
pub(crate) fn build_image_meta(
    reader: &mut dyn FormatReader,
    path: &Path,
    read_mode: ReadMode,
) -> Result<ImageMeta, InternalErrors> {
    let mut meta = ImageMeta {
        name: path
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("Unknown")
            .to_string(),
        ..Default::default()
    };

    let mut objective_set = false;
    let series_count = reader.series_count();

    for series_idx in 0..series_count {
        reader.set_series(series_idx).map_err(bf_err)?;

        let ome = reader.ome_metadata();

        let (mut info, series_metadata) = {
            let base = reader.metadata();
            (
                ImageInfo {
                    nr_c_stacks: effective_size_c(base) as i32,
                    nr_z_stacks: base.size_z.max(1) as i32,
                    nr_t_stacks: base.size_t.max(1) as i32,
                    ..Default::default()
                },
                base.series_metadata.clone(),
            )
        };

        let ome_image = ome.as_ref().and_then(|o| o.images.get(series_idx));

        // Populated from `0..nr_c_stacks` (the already-computed real channel
        // count), not from `ome_image.channels.len()` - some readers (e.g.
        // VSI) report a `size_c`/`image_count` that implies N real channels
        // but only populate the OME channel list with 1 entry. Trusting the
        // OME list's own length there silently drops every channel above
        // the first, which then fails downstream with "Channel <i> does not
        // exist" the moment anything asks to read it.
        for i in 0..info.nr_c_stacks {
            let channel = ome_image.and_then(|img| img.channels.get(i as usize));

            // The Olympus CellSens/VSI reader doesn't populate the standard
            // OME channel fields at all (unlike e.g. its own FV1000/OIF
            // reader, which does) - the same data is there, just under its
            // own vendor-prefixed `series_metadata` keys instead. Only
            // consulted when the OME channel itself didn't already have an
            // answer, so a reader that *does* populate the standard fields
            // is never second-guessed.
            let name = channel
                .and_then(|c| c.name.clone())
                .or_else(|| {
                    metadata_string(&series_metadata, &format!("cellsens.ets.channel_name.{i}"))
                })
                .unwrap_or_default();
            let emission_wave_length = channel
                .and_then(|c| c.emission_wavelength)
                .or_else(|| {
                    metadata_f64(
                        &series_metadata,
                        &format!("cellsens.ets.channel_wavelength.{i}"),
                    )
                })
                .unwrap_or(0.0) as f32;

            info.channels.insert(
                i,
                ChannelInfo {
                    id: format!("Channel:{series_idx}:{i}"),
                    name,
                    emission_wave_length,
                    contrast_method: channel
                        .and_then(|c| c.contrast_method.clone())
                        .unwrap_or_default(),
                },
            );
        }

        if let Some(ome_image) = ome_image {
            info.pixel_sizes = PixelSizes {
                px_size_x: ome_image.physical_size_x.unwrap_or(0.0) as f32
                    * MICROMETER_TO_NANOMETER,
                px_size_y: ome_image.physical_size_y.unwrap_or(0.0) as f32
                    * MICROMETER_TO_NANOMETER,
                px_size_z: ome_image.physical_size_z.unwrap_or(0.0) as f32
                    * MICROMETER_TO_NANOMETER,
            };

            if !objective_set
                && let Some(ome_full) = ome.as_ref()
                && let Some(instrument_idx) = ome_image.instrument_ref
                && let Some(objective_idx) = ome_image.objective_ref
                && let Some(instrument) = ome_full.instruments.get(instrument_idx)
                && let Some(objective) = instrument.objectives.get(objective_idx)
            {
                meta.objective = Objective {
                    manufacturer: objective.manufacturer.clone().unwrap_or_default(),
                    model: objective.model.clone().unwrap_or_default(),
                    magnification: objective.nominal_magnification.unwrap_or(0.0) as f32,
                };
                objective_set = true;
            }
        }

        let resolution_count = reader.resolution_count();
        for res in 0..resolution_count {
            reader.set_resolution(res).map_err(bf_err)?;
            let m = reader.metadata();
            let mut pyramid_info = PyramidInfo {
                nr_bits: m.bits_per_pixel as u16,
                color_channels: rgb_channel_count(m) as u8,
                is_rgb: m.is_rgb,
                width: m.size_x as u64,
                height: m.size_y as u64,
                // `bioformats` handles its own internal tiling
                // transparently (`open_bytes_region` accepts an
                // arbitrary region, not a fixed native tile) - unlike
                // the old Java wrapper's `getOptimalTileWidth/Height`,
                // there is no native tile size to report here, and
                // nothing downstream reads it as more than a display
                // value, so it's just the resolution's own dimensions.
                tile_width: m.size_x as u64,
                tile_height: m.size_y as u64,
                is_interleaved: m.is_interleaved,
                is_little_endian: m.is_little_endian,
            };

            // `ReadMode::SplitChannels` reports each RGB sample as its own
            // single-component channel, so `is_rgb` reads `false` by the
            // time it gets here - Bio-Formats' `ChannelSeparator` has the
            // identical limitation in Java, which the old Java-backed
            // version of this parser worked around with this exact
            // signature (see git history, pre-port `image_ome_parser.rs`:
            // "TODO: This is a trick to find RGB images if we use split
            // channel"). Ported as-is for parity, imprecision included -
            // it can't perfectly distinguish a split RGB image from a
            // genuine 3-channel 8-bit non-interleaved fluorescence image,
            // same as the Java-era heuristic couldn't - which is exactly
            // why the old code (and this) only ever applies it in
            // `SplitChannels` mode: a plain `ReadMode::Default` open of a
            // genuine 3-channel fluorescence file must never be
            // reclassified as RGB just because it happens to match this
            // shape (see `build_image_meta_reads_series_channel_and_resolution_shape_from_a_real_file`,
            // a real non-RGB 3-channel 8-bit fixture that does).
            if read_mode == ReadMode::SplitChannels
                && pyramid_info.color_channels == 1
                && !pyramid_info.is_interleaved
                && info.nr_c_stacks == 3
                && pyramid_info.nr_bits == 8
            {
                pyramid_info.is_rgb = true;
                const RGB_NAMES: [&str; 3] = ["Red", "Green", "Blue"];
                const RGB_EMISSION_NM: [f32; 3] = [635.0, 532.0, 450.0];
                for i in 0..3 {
                    if let Some(channel) = info.channels.get_mut(&i) {
                        channel.name = RGB_NAMES[i as usize].to_string();
                        channel.emission_wave_length = RGB_EMISSION_NM[i as usize];
                    }
                }
            }

            info.resolutions.insert(res as i32, pyramid_info);
        }
        // `set_series` always resets the active resolution back to 0 (see
        // e.g. the TIFF reader), but reset explicitly here too so a series
        // with zero resolutions (shouldn't happen, but not guaranteed by
        // the trait) can't leave a later series reading at a stale level.
        reader.set_resolution(0).map_err(bf_err)?;

        meta.series.insert(series_idx as i32, info);
    }

    Ok(meta)
}
#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn meta(overrides: impl FnOnce(&mut ImageMetadata)) -> ImageMetadata {
        let mut m = ImageMetadata::default();
        overrides(&mut m);
        m
    }

    // -- rgb_channel_count / effective_size_c ------------------------------------

    #[test]
    fn rgb_channel_count_is_always_one_for_a_non_rgb_series() {
        let m = meta(|m| {
            m.is_rgb = false;
            m.size_c = 5;
        });
        assert_eq!(rgb_channel_count(&m), 1);
        assert_eq!(effective_size_c(&m), 5);
    }

    #[test]
    fn rgb_channel_count_derives_the_split_factor_from_image_count_when_available() {
        // A single-timepoint, single-Z RGB image: 1 real plane on disk
        // (image_count=1) holding all 3 samples together, reported as
        // size_c=3 in Bio-Formats' convention.
        let m = meta(|m| {
            m.is_rgb = true;
            m.size_c = 3;
            m.size_z = 1;
            m.size_t = 1;
            m.image_count = 1;
        });
        assert_eq!(rgb_channel_count(&m), 3);
        assert_eq!(
            effective_size_c(&m),
            1,
            "one packed-RGB plane is one effective channel"
        );
    }

    #[test]
    fn rgb_channel_count_supports_multiple_effective_channels_of_packed_rgb() {
        // Two real fluorescence-like channels, each itself RGB: size_c=6,
        // image_count=2 (one plane per effective channel).
        let m = meta(|m| {
            m.is_rgb = true;
            m.size_c = 6;
            m.size_z = 1;
            m.size_t = 1;
            m.image_count = 2;
        });
        assert_eq!(rgb_channel_count(&m), 3);
        assert_eq!(effective_size_c(&m), 2);
    }

    #[test]
    fn rgb_channel_count_falls_back_to_raw_size_c_when_image_count_is_inconsistent() {
        // image_count doesn't cleanly divide by z*t, or size_c isn't a
        // multiple of the derived effective channel count - falls back to
        // treating size_c itself as the split factor rather than panicking
        // or dividing unevenly.
        let m = meta(|m| {
            m.is_rgb = true;
            m.size_c = 3;
            m.size_z = 1;
            m.size_t = 1;
            m.image_count = 0;
        });
        assert_eq!(rgb_channel_count(&m), 3);
    }

    // -- build_image_meta (real fixture) -----------------------------------------

    /// The real `multi-channel-4D-series.ome.tif` fixture several other
    /// tests in `image_reader.rs` also open for real - a 3-channel, 5-Z,
    /// 7-T, single-series, non-pyramidal 8-bit grayscale TIFF.
    fn fixture_path() -> PathBuf {
        PathBuf::from(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/multi-channel-4D-series.ome.tif"
        ))
    }

    #[test]
    fn build_image_meta_reads_series_channel_and_resolution_shape_from_a_real_file() {
        let path = fixture_path();
        let mut reader =
            bioformats::registry::open_reader_boxed(&path).expect("real fixture must open");

        let image_meta = build_image_meta(reader.as_mut(), &path, ReadMode::Default).unwrap();

        assert_eq!(image_meta.name, "multi-channel-4D-series.ome.tif");
        assert_eq!(image_meta.series.len(), 1);

        let series = image_meta.series.get(&0).expect("series 0 must exist");
        assert_eq!(series.nr_c_stacks, 3);
        assert_eq!(series.nr_z_stacks, 5);
        assert_eq!(series.nr_t_stacks, 7);
        assert_eq!(series.channels.len(), 3);
        for c in 0..3 {
            assert!(
                series.channels.contains_key(&c),
                "channel {c} must be present"
            );
        }

        assert_eq!(series.resolutions.len(), 1);
        let res = series.resolutions.get(&0).expect("resolution 0 must exist");
        assert_eq!(res.width, 439);
        assert_eq!(res.height, 167);
        assert_eq!(res.nr_bits, 8);
        assert_eq!(
            res.color_channels, 1,
            "a plain (non-RGB) 3-channel series packs one sample per pixel"
        );
        assert!(!res.is_rgb);
        assert!(!res.is_interleaved);
        assert!(!res.is_little_endian);
        assert_eq!(
            (res.tile_width, res.tile_height),
            (res.width, res.height),
            "no native tile size is exposed - tile dims fall back to the resolution's own"
        );
    }

    /// Regression test for two real-world gaps found in `G7_03.vsi`'s
    /// Olympus CellSens/VSI reader:
    ///
    /// 1. it declares `size_c=3`/`image_count=3` (3 real channels, non-RGB)
    ///    but only populates 1 entry in its own OME channel list - trusting
    ///    that list's length to decide how many `ChannelInfo` entries to
    ///    create silently dropped channels 1 and 2, which then failed with
    ///    "Channel 1 does not exist in series 0" the moment anything tried
    ///    to read them;
    /// 2. it never populates the OME channel's `name`/`emission_wavelength`
    ///    fields at all, even for the one channel it does list - the same
    ///    data is present, just under its own `cellsens.ets.channel_name.N`/
    ///    `cellsens.ets.channel_wavelength.N` `series_metadata` keys.
    #[test]
    #[ignore = "requires tests/G7_03.vsi, a real VSI fixture too large to commit to git \
                (see .gitignore) - run locally with `cargo test -- --ignored` when present"]
    fn build_image_meta_populates_every_channel_even_when_ome_metadata_under_reports_them() {
        let path = PathBuf::from(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../tests/G7_03.vsi"
        ));
        let mut reader =
            bioformats::registry::open_reader_boxed(&path).expect("real fixture must open");

        let image_meta = build_image_meta(reader.as_mut(), &path, ReadMode::Default).unwrap();

        let series = image_meta.series.get(&0).expect("series 0 must exist");
        assert_eq!(series.nr_c_stacks, 3);
        assert_eq!(
            series.channels.len(),
            3,
            "every declared channel must get an entry, not just the ones the \
             OME metadata happened to describe"
        );
        for c in 0..3 {
            assert!(
                series.channels.contains_key(&c),
                "channel {c} must be present"
            );
        }

        // Names/wavelengths recovered from the vendor-prefixed
        // `series_metadata` keys, since the OME channel list itself never
        // carries them for this reader.
        let expected = [("CY7", 767.0_f32), ("CY5", 670.0), ("FITC", 518.0)];
        for (c, (name, wavelength)) in expected.into_iter().enumerate() {
            let channel = &series.channels[&(c as i32)];
            assert_eq!(channel.name, name, "channel {c}");
            assert_eq!(channel.emission_wave_length, wavelength, "channel {c}");
        }
    }

    #[test]
    fn build_image_meta_leaves_the_reader_at_resolution_zero_of_the_last_series() {
        let path = fixture_path();
        let mut reader =
            bioformats::registry::open_reader_boxed(&path).expect("real fixture must open");

        build_image_meta(reader.as_mut(), &path, ReadMode::Default).unwrap();

        assert_eq!(reader.resolution(), 0);
    }
}
