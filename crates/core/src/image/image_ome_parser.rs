//! # image_ome_parser
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-06-05
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.

use crate::image::image_meta::{
    ChannelInfo, DeviceInfo, EnvironmentInfo, ExperimenterInfo, FilterInfo, ImageInfo, ImageMeta,
    LightSourceInfo, Objective, PixelSizes, PyramidInfo,
};
use crate::image::image_reader::ReadMode;
use evanalyzer_cfg::core_types::InternalErrors;
use scopeio::{ChannelSelect, PixelType, Plane, SeriesMeta};
use std::path::Path;

pub(crate) fn scopeio_err(e: scopeio::Error) -> InternalErrors {
    InternalErrors::ImageReadError(e.to_string())
}

/// Channels one scopeio channel becomes in this app: an RGB channel is split
/// into its red, green and blue samples in [`ReadMode::SplitChannels`]
/// (alpha dropped), everything else stays one channel.
fn channels_per_source(series: &SeriesMeta, read_mode: ReadMode) -> u32 {
    if read_mode == ReadMode::SplitChannels && series.samples_per_pixel >= 3 {
        3
    } else {
        1
    }
}

/// What scopeio reads for this app's channel index `c`.
pub(crate) fn channel_select(series: &SeriesMeta, read_mode: ReadMode, c: u32) -> ChannelSelect {
    match channels_per_source(series, read_mode) {
        1 => ChannelSelect::Channel(c),
        n => ChannelSelect::Sample {
            channel: c / n,
            sample: (c % n) as u8,
        },
    }
}

/// Bits per sample: the significant bits if the file declares them (e.g. 12
/// for 12-bit data stored as u16), else the full width of the sample type.
fn bit_depth(series: &SeriesMeta) -> u16 {
    series
        .significant_bits
        .filter(|&bits| bits > 0)
        .map(u16::from)
        .unwrap_or(series.pixel_type.bytes() as u16 * 8)
}

/// Signed integer samples, which are shifted by the type's minimum when
/// normalized.
pub(crate) fn is_signed(pixel_type: PixelType) -> bool {
    matches!(pixel_type, PixelType::I8 | PixelType::I16 | PixelType::I32)
}

/// The time between two frames [s]: the series' time increment, else
/// derived from the planes' time stamps - the mean step between the first
/// and the last frame of channel 0, z 0 (`delta_t(t)`). `None` without
/// either, or for a single frame.
fn frame_interval(
    time_increment_s: Option<f64>,
    nr_t_stacks: i32,
    delta_t: impl Fn(u32) -> Option<f64>,
) -> Option<f32> {
    let positive = |v: f64| (v.is_finite() && v > 0.0).then_some(v as f32);
    if let Some(v) = time_increment_s.and_then(positive) {
        return Some(v);
    }
    if nr_t_stacks < 2 {
        return None;
    }
    let last = (nr_t_stacks - 1) as u32;
    positive((delta_t(last)? - delta_t(0)?) / last as f64)
}

/// Builds this app's [`ImageMeta`] from an opened scopeio reader: a
/// field-by-field mapping of each series' [`SeriesMeta`] (scopeio already
/// reports lengths and wavelengths in nanometres, like this app).
/// A text of the file, trimmed; "" when it has none.
fn text(value: &Option<String>) -> String {
    value.as_deref().unwrap_or("").trim().to_string()
}

fn device(
    manufacturer: &Option<String>,
    model: &Option<String>,
    serial_number: &Option<String>,
    kind: &Option<String>,
) -> DeviceInfo {
    DeviceInfo {
        manufacturer: text(manufacturer),
        model: text(model),
        serial_number: text(serial_number),
        kind: text(kind),
    }
}

fn filter(f: &scopeio::FilterMeta) -> FilterInfo {
    FilterInfo {
        manufacturer: text(&f.manufacturer),
        model: text(&f.model),
        kind: text(&f.kind),
        filter_wheel: text(&f.filter_wheel),
        cut_in_nm: f.cut_in_nm.map(|v| v as f32),
        cut_out_nm: f.cut_out_nm.map(|v| v as f32),
        transmittance: f.transmittance.map(|v| v as f32),
    }
}

pub(crate) fn build_image_meta(
    reader: &dyn scopeio::ImageReader,
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
    for series_idx in 0..reader.series_count() {
        let series = reader.series_meta(series_idx).map_err(scopeio_err)?;
        let split = channels_per_source(series, read_mode);
        let is_rgb = series.samples_per_pixel >= 3;

        let mut info = ImageInfo {
            name: series.name.clone().unwrap_or_default(),
            description: series.description.clone().unwrap_or_default(),
            acquisition_date: series.acquisition_date.clone().unwrap_or_default(),
            significant_bits: series.significant_bits,
            stage_position_nm: series
                .stage_position
                .as_ref()
                .map(|p| [p.x_nm, p.y_nm, p.z_nm])
                .unwrap_or([None; 3]),
            nr_c_stacks: (series.size_c.max(1) * split) as i32,
            nr_z_stacks: series.size_z.max(1) as i32,
            nr_t_stacks: series.size_t.max(1) as i32,
            pixel_sizes: PixelSizes {
                px_size_x: series.physical_size.x_nm.unwrap_or(0.0) as f32,
                px_size_y: series.physical_size.y_nm.unwrap_or(0.0) as f32,
                px_size_z: series.physical_size.z_nm.unwrap_or(0.0) as f32,
            },
            ..Default::default()
        };

        const RGB_NAMES: [&str; 3] = ["Red", "Green", "Blue"];
        const RGB_EMISSION_NM: [f32; 3] = [635.0, 532.0, 450.0];
        for i in 0..info.nr_c_stacks {
            let source = series.channels.get(i as usize / split as usize);
            let (name, emission_wave_length) = if split == 3 {
                let sample = i as usize % 3;
                (RGB_NAMES[sample].to_string(), RGB_EMISSION_NM[sample])
            } else {
                (
                    source.and_then(|c| c.name.clone()).unwrap_or_default(),
                    source.and_then(|c| c.emission_wavelength_nm).unwrap_or(0.0) as f32,
                )
            };
            info.channels.insert(
                i,
                ChannelInfo {
                    id: format!("Channel:{series_idx}:{i}"),
                    name,
                    emission_wave_length,
                    contrast_method: source
                        .and_then(|c| c.contrast_method.clone())
                        .unwrap_or_default(),
                    // The acquisition values of the channel (an RGB
                    // image's split samples share the one channel's).
                    fluor: source.and_then(|c| c.fluor.clone()).unwrap_or_default(),
                    excitation_wave_length: source
                        .and_then(|c| c.excitation_wavelength_nm)
                        .unwrap_or(0.0) as f32,
                    exposure_s: source.and_then(|c| c.exposure_s).map(|v| v as f32),
                    detector_gain: source.and_then(|c| c.detector_gain).map(|v| v as f32),
                    detector_offset: source.and_then(|c| c.detector_offset).map(|v| v as f32),
                    binning: source.and_then(|c| c.binning),
                    pinhole_nm: source.and_then(|c| c.pinhole_size_nm).map(|v| v as f32),
                    zoom: source.and_then(|c| c.zoom).map(|v| v as f32),
                    detector: source
                        .and_then(|c| c.detector.as_ref())
                        .map(|d| device(&d.manufacturer, &d.model, &d.serial_number, &d.kind))
                        .unwrap_or_default(),
                    light_source: source
                        .and_then(|c| c.light_source.as_ref())
                        .map(|l| LightSourceInfo {
                            device: device(&l.manufacturer, &l.model, &l.serial_number, &l.kind),
                            subtype: text(&l.subtype),
                            laser_medium: text(&l.laser_medium),
                            wavelength_nm: l.wavelength_nm.map(|v| v as f32),
                            power_mw: l.power_mw.map(|v| v as f32),
                            attenuation: l.attenuation.map(|v| v as f32),
                        })
                        .unwrap_or_default(),
                    excitation_filters: source
                        .map(|c| c.excitation_filters.iter().map(filter).collect())
                        .unwrap_or_default(),
                    emission_filters: source
                        .map(|c| c.emission_filters.iter().map(filter).collect())
                        .unwrap_or_default(),
                    dichroic: source
                        .and_then(|c| c.dichroic.as_ref())
                        .map(|d| device(&d.manufacturer, &d.model, &None, &None))
                        .unwrap_or_default(),
                },
            );
        }

        info.microscope = series
            .microscope
            .as_ref()
            .map(|m| device(&m.manufacturer, &m.model, &m.serial_number, &m.kind))
            .unwrap_or_default();
        info.environment = series
            .environment
            .as_ref()
            .map(|e| EnvironmentInfo {
                temperature_c: e.temperature_c.map(|v| v as f32),
                co2_fraction: e.co2_fraction.map(|v| v as f32),
                humidity_fraction: e.humidity_fraction.map(|v| v as f32),
                air_pressure_mbar: e.air_pressure_mbar.map(|v| v as f32),
            })
            .unwrap_or_default();
        info.experimenter = series
            .experimenter
            .as_ref()
            .map(|x| ExperimenterInfo {
                first_name: text(&x.first_name),
                last_name: text(&x.last_name),
                email: text(&x.email),
                institution: text(&x.institution),
                user_name: text(&x.user_name),
            })
            .unwrap_or_default();
        info.project = text(&series.project);
        info.dataset = text(&series.dataset);

        info.frame_interval = frame_interval(series.time_increment_s, info.nr_t_stacks, |t| {
            reader
                .plane_meta(series_idx, Plane::new(0, 0, t))
                .ok()
                .and_then(|p| p.delta_t_s)
        });

        if !objective_set && let Some(objective) = &series.objective {
            meta.objective = Objective {
                manufacturer: objective.manufacturer.clone().unwrap_or_default(),
                model: objective.model.clone().unwrap_or_default(),
                magnification: objective.magnification.unwrap_or(0.0) as f32,
                numerical_aperture: objective.numerical_aperture.unwrap_or(0.0) as f32,
                immersion: objective.immersion.clone().unwrap_or_default(),
                correction: text(&objective.correction),
                working_distance_nm: objective.working_distance_nm.map(|v| v as f32),
                refractive_index: objective.refractive_index.map(|v| v as f32),
            };
            objective_set = true;
        }

        for (res, level) in series.levels.iter().enumerate() {
            info.resolutions.insert(
                res as i32,
                PyramidInfo {
                    nr_bits: bit_depth(series),
                    // Alpha is dropped on read; split RGB samples are gray.
                    color_channels: if is_rgb && split == 1 { 3 } else { 1 },
                    is_rgb,
                    width: level.size_x,
                    height: level.size_y,
                    // Reads take any region, so the level itself is the tile.
                    tile_width: level.size_x,
                    tile_height: level.size_y,
                },
            );
        }

        meta.series.insert(series_idx as i32, info);
    }

    Ok(meta)
}
#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn series(samples_per_pixel: u8, size_c: u32) -> SeriesMeta {
        // `SeriesMeta` is `#[non_exhaustive]`: built from its default.
        let mut s = SeriesMeta::default();
        s.samples_per_pixel = samples_per_pixel;
        s.size_c = size_c;
        s.size_z = 1;
        s.size_t = 1;
        s
    }

    // -- channel_select / bit_depth ---------------------------------------------

    #[test]
    fn only_split_mode_splits_an_rgb_channel_into_its_samples() {
        let rgb = series(3, 1);
        assert_eq!(
            channel_select(&rgb, ReadMode::Default, 0),
            ChannelSelect::Channel(0)
        );
        assert_eq!(
            channel_select(&rgb, ReadMode::SplitChannels, 2),
            ChannelSelect::Sample {
                channel: 0,
                sample: 2
            }
        );
        // Two RGB channels: channel 4 is the green sample of the second.
        let two_rgb = series(4, 2);
        assert_eq!(
            channel_select(&two_rgb, ReadMode::SplitChannels, 4),
            ChannelSelect::Sample {
                channel: 1,
                sample: 1
            }
        );
        // A gray image is never split.
        let gray = series(1, 3);
        assert_eq!(
            channel_select(&gray, ReadMode::SplitChannels, 2),
            ChannelSelect::Channel(2)
        );
    }

    #[test]
    fn the_bit_depth_is_the_significant_bits_else_the_sample_width() {
        let mut s = series(1, 1);
        s.pixel_type = PixelType::U16;
        assert_eq!(bit_depth(&s), 16);
        s.significant_bits = Some(12);
        assert_eq!(bit_depth(&s), 12);
        s.significant_bits = Some(0);
        assert_eq!(bit_depth(&s), 16, "0 means not declared");
    }

    // -- build_image_meta: instrument metadata (scopeio 0.0.0-alpha.5) -----------

    /// A standalone OME-XML file with the instrument, environment and owner
    /// metadata (the example of scopeio's own test) and 4 × 4 pixels inline.
    const OME_XML_WITH_INSTRUMENT: &str = r#"<?xml version="1.0" encoding="UTF-8"?>
<OME xmlns="http://www.openmicroscopy.org/Schemas/OME/2016-06">
  <Project ID="Project:0" Name="Organoids"><DatasetRef ID="Dataset:0"/></Project>
  <Dataset ID="Dataset:0" Name="Day 3"><ImageRef ID="Image:0"/></Dataset>
  <Experimenter ID="Experimenter:0" FirstName="Ada" LastName="Lovelace" Email="ada@example.org"/>
  <Instrument ID="Instrument:0">
    <Microscope Manufacturer="Zeiss" Model="LSM 980" Type="Inverted"/>
    <Laser ID="LightSource:0" Type="SolidState" LaserMedium="Other" Wavelength="488" Power="0.03" PowerUnit="W"/>
    <Detector ID="Detector:0" Model="GaAsP" Type="PMT" Zoom="1"/>
    <Objective ID="Objective:0" Model="Plan-Apochromat 63x/1.40 Oil" Correction="PlanApo" Immersion="Oil"
               LensNA="1.4" NominalMagnification="63" WorkingDistance="0.19" WorkingDistanceUnit="mm"/>
    <FilterSet ID="FilterSet:0"><ExcitationFilterRef ID="Filter:0"/><DichroicRef ID="Dichroic:0"/><EmissionFilterRef ID="Filter:1"/></FilterSet>
    <Filter ID="Filter:0" Type="BandPass"><TransmittanceRange CutIn="470" CutOut="495"/></Filter>
    <Filter ID="Filter:1" Type="BandPass"><TransmittanceRange CutIn="500" CutOut="550" Transmittance="0.9"/></Filter>
    <Dichroic ID="Dichroic:0" Model="MBS 488"/>
  </Instrument>
  <Image ID="Image:0" Name="cells">
    <InstrumentRef ID="Instrument:0"/>
    <ObjectiveSettings ID="Objective:0" RefractiveIndex="1.518"/>
    <ImagingEnvironment Temperature="310.15" TemperatureUnit="K" CO2Percent="0.05" Humidity="0.9"/>
    <Pixels ID="Pixels:0" DimensionOrder="XYZCT" Type="uint8" SizeX="4" SizeY="4" SizeZ="1" SizeC="1" SizeT="1">
      <Channel ID="Channel:0:0" Name="GFP">
        <LightSourceSettings ID="LightSource:0" Wavelength="488" Attenuation="0.8"/>
        <DetectorSettings ID="Detector:0" Gain="650" Zoom="2.5"/>
        <FilterSetRef ID="FilterSet:0"/>
      </Channel>
      <BinData xmlns="http://www.openmicroscopy.org/Schemas/BinaryFile/2016-06" BigEndian="false" Length="16">AAAAAAAAAAAAAAAAAAAAAA==</BinData>
    </Pixels>
  </Image>
</OME>"#;

    /// The microscope, environment, owner, objective extras and each
    /// channel's detector, light source, filters and dichroic reach the
    /// app's metadata.
    #[test]
    fn the_instrument_metadata_reaches_the_image_meta() {
        let dir = tempfile::TempDir::new().unwrap();
        let path = dir.path().join("cells.ome.xml");
        std::fs::write(&path, OME_XML_WITH_INSTRUMENT).unwrap();
        let reader = scopeio::open(&path).expect("standalone OME-XML opens");
        let meta = build_image_meta(reader.as_ref(), &path, ReadMode::Default).unwrap();

        let series = &meta.series[&0];
        assert_eq!(series.microscope.manufacturer, "Zeiss");
        assert_eq!(series.microscope.model, "LSM 980");
        assert_eq!(series.microscope.kind, "Inverted");
        assert!((series.environment.temperature_c.unwrap() - 37.0).abs() < 1e-3);
        assert_eq!(series.environment.co2_fraction, Some(0.05));
        assert_eq!(series.environment.humidity_fraction, Some(0.9));
        assert_eq!(series.experimenter.first_name, "Ada");
        assert_eq!(series.experimenter.email, "ada@example.org");
        assert_eq!(
            (series.project.as_str(), series.dataset.as_str()),
            ("Organoids", "Day 3")
        );

        assert_eq!(meta.objective.correction, "PlanApo");
        assert_eq!(meta.objective.working_distance_nm, Some(190_000.0));
        assert_eq!(meta.objective.refractive_index, Some(1.518));

        let c = &series.channels[&0];
        assert_eq!(c.zoom, Some(2.5));
        assert_eq!(
            (c.detector.model.as_str(), c.detector.kind.as_str()),
            ("GaAsP", "PMT")
        );
        assert_eq!(c.light_source.device.kind, "Laser");
        assert_eq!(c.light_source.subtype, "SolidState");
        assert_eq!(c.light_source.wavelength_nm, Some(488.0));
        assert_eq!(c.light_source.power_mw, Some(30.0));
        assert_eq!(c.light_source.attenuation, Some(0.8));
        assert_eq!(c.excitation_filters.len(), 1);
        assert_eq!(c.excitation_filters[0].cut_in_nm, Some(470.0));
        assert_eq!(c.emission_filters[0].transmittance, Some(0.9));
        assert_eq!(c.dichroic.model, "MBS 488");
    }

    /// A file without them (the 4D fixture): everything empty, no error.
    #[test]
    fn a_file_without_instrument_metadata_leaves_it_empty() {
        let path = fixture_path();
        let reader = scopeio::open(&path).expect("real fixture must open");
        let meta = build_image_meta(reader.as_ref(), &path, ReadMode::Default).unwrap();
        let series = &meta.series[&0];
        assert!(series.microscope.is_empty());
        assert_eq!(series.environment, EnvironmentInfo::default());
        assert_eq!(series.project, "");
        let c = &series.channels[&0];
        assert!(c.detector.is_empty() && c.excitation_filters.is_empty() && c.dichroic.is_empty());
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
        let reader = scopeio::open(&path).expect("real fixture must open");

        let image_meta = build_image_meta(reader.as_ref(), &path, ReadMode::Default).unwrap();

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
        assert_eq!(
            (res.tile_width, res.tile_height),
            (res.width, res.height),
            "reads take any region - tile dims are the resolution's own"
        );

        // A genuine 3-channel 8-bit fluorescence image must not be taken
        // for a split RGB image.
        let split = build_image_meta(reader.as_ref(), &path, ReadMode::SplitChannels).unwrap();
        let split_series = &split.series[&0];
        assert_eq!(split_series.nr_c_stacks, 3);
        assert!(!split_series.resolutions[&0].is_rgb);
    }

    /// Regression test for `G7_03.vsi` (Olympus CellSens/VSI): it declares
    /// 3 real channels, each of which must get an entry with its name and
    /// emission wavelength.
    #[test]
    #[ignore = "requires tests/G7_03.vsi, a real VSI fixture too large to commit to git \
                (see .gitignore) - run locally with `cargo test -- --ignored` when present"]
    fn build_image_meta_populates_every_channel_even_when_ome_metadata_under_reports_them() {
        let path = PathBuf::from(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../tests/G7_03.vsi"
        ));
        let reader = scopeio::open(&path).expect("real fixture must open");

        let image_meta = build_image_meta(reader.as_ref(), &path, ReadMode::Default).unwrap();

        let series = image_meta.series.get(&0).expect("series 0 must exist");
        assert_eq!(series.nr_c_stacks, 3);
        assert_eq!(
            series.channels.len(),
            3,
            "every declared channel must get an entry"
        );

        let expected = [("CY7", 767.0_f32), ("CY5", 670.0), ("FITC", 518.0)];
        for (c, (name, wavelength)) in expected.into_iter().enumerate() {
            let channel = &series.channels[&(c as i32)];
            assert_eq!(channel.name, name, "channel {c}");
            assert_eq!(channel.emission_wave_length, wavelength, "channel {c}");
        }
    }

    #[test]
    fn the_frame_interval_comes_from_the_time_increment_else_the_plane_time_stamps() {
        // 0, 4.1, 7.9 s: 3.95 s on average.
        let stamps = |t: u32| [0.0, 4.1, 7.9].get(t as usize).copied();
        assert_eq!(frame_interval(Some(2.5), 3, stamps), Some(2.5));
        assert_eq!(frame_interval(None, 3, stamps), Some(3.95));

        assert_eq!(frame_interval(None, 1, stamps), None, "a single frame");
        assert_eq!(frame_interval(None, 3, |_| None), None, "no metadata");
        assert_eq!(
            frame_interval(Some(0.0), 3, |_| None),
            None,
            "zero means unknown"
        );
    }
}
