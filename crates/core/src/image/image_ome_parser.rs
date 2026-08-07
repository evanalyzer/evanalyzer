use crate::converters::LengthUnit;
use crate::image::image_meta::{ChannelInfo, ImageInfo, ImageMeta, PyramidInfo};
use crate::{ImageReader, ReadMode};
use evanalyzer_cfg::core_types::InternalErrors;
use quick_xml::XmlVersion;
use quick_xml::events::Event;
use quick_xml::reader::Reader;

impl ImageReader {
    /// Gets an XML string im OME format
    ///
    /// # Arguments
    ///
    /// - `&self` (`undefined`) - Describe this parameter.
    /// - `ome_xml` (`String`) - Describe this parameter.
    ///
    /// # Examples
    ///
    /// ```
    /// use crate::...;
    ///
    /// let _ = parse_ome_xml();
    /// ```
    pub(crate) fn parse_ome_xml(&self, xml_str: &str) -> Result<ImageMeta, InternalErrors> {
        let mut reader = Reader::from_str(xml_str);
        reader.config_mut().trim_text(true);

        let mut meta = ImageMeta::default();
        let mut buf = Vec::new();

        meta.name = self
            .current_path
            .file_name()
            .and_then(|os_str| os_str.to_str())
            .unwrap_or("Unknown")
            .into();

        // Tracking "State"
        let mut current_series_idx: i32 = -1;
        let mut in_joda = false;
        let mut in_pixels = false;
        let mut in_instrument = false;

        loop {
            match reader.read_event_into(&mut buf) {
                Ok(Event::Start(ref e)) | Ok(Event::Empty(ref e)) => {
                    match e.local_name().as_ref() {
                        b"Image" => {
                            let mut image_id = String::new();
                            let mut _image_name = String::new();
                            // Iterate through attributes efficiently
                            for attr in e.attributes().flatten() {
                                match attr.key.local_name().as_ref() {
                                    b"ID" => {
                                        // Use unescape_value() in case the ID has special chars
                                        image_id = attr
                                            .normalized_value(XmlVersion::Implicit1_0)?
                                            .into_owned();
                                    }
                                    b"Name" => {
                                        _image_name = attr
                                            .normalized_value(XmlVersion::Implicit1_0)?
                                            .into_owned();
                                    }
                                    _ => {}
                                }
                            }

                            let series_idx: i32 = image_id
                                .rsplit_once(':')
                                .map(|(_, val)| val.parse::<i32>())
                                .transpose()? // Propagates the parse error if it's not a number
                                .ok_or_else(|| {
                                    InternalErrors::ParseError(format!(
                                        "Invalid ID format: {}",
                                        image_id
                                    ))
                                })?;
                            current_series_idx = series_idx;
                            meta.series.insert(current_series_idx, ImageInfo::default());
                        }
                        b"Instrument" => {
                            in_instrument = true;
                        }
                        b"Objective" if in_instrument => {
                            for attr in e.attributes().flatten() {
                                match attr.key.local_name().as_ref() {
                                    b"ID" => {}
                                    b"Manufacturer" => {
                                        meta.objective.manufacturer = attr
                                            .normalized_value(XmlVersion::Implicit1_0)?
                                            .into_owned();
                                    }
                                    b"Model" => {
                                        meta.objective.model = attr
                                            .normalized_value(XmlVersion::Implicit1_0)?
                                            .into_owned();
                                    }
                                    b"LensNA" => {}
                                    b"NominalMagnification" => {
                                        meta.objective.magnification =
                                            self.parse_f32(attr.value.as_ref())?;
                                    }
                                    b"CalibratedMagnification" => {}
                                    b"WorkingDistance" => {}
                                    b"WorkingDistanceUnit" => {}
                                    _ => {}
                                }
                            }
                        }
                        b"Pixels" => {
                            //  <Pixels BigEndian="false" DimensionOrder="XYCZT" ID="Pixels:0" Interleaved="false" PhysicalSizeX="0.16250000000000006" PhysicalSizeXUnit="µm"
                            //   PhysicalSizeY="0.16250000000000006" PhysicalSizeYUnit="µm" PhysicalSizeZ="0.5" PhysicalSizeZUnit="µm"
                            //   SignificantBits="16" SizeC="5" SizeT="1" SizeX="2048" SizeY="2048" SizeZ="5" Type="uint16">

                            in_pixels = true;
                            if let Some(info) = meta.series.get_mut(&current_series_idx) {
                                // Extract attributes directly from the stream
                                let mut unit_x: LengthUnit = LengthUnit::Nanometer;
                                let mut unit_y: LengthUnit = LengthUnit::Nanometer;
                                let mut unit_z: LengthUnit = LengthUnit::Nanometer;

                                for attr in e.attributes().flatten() {
                                    match attr.key.local_name().as_ref() {
                                        b"PhysicalSizeX" => {
                                            info.pixel_sizes.px_size_x =
                                                self.parse_f32(attr.value.as_ref())?;
                                        }
                                        b"PhysicalSizeY" => {
                                            info.pixel_sizes.px_size_y =
                                                self.parse_f32(attr.value.as_ref())?;
                                        }
                                        b"PhysicalSizeZ" => {
                                            info.pixel_sizes.px_size_z =
                                                self.parse_f32(attr.value.as_ref())?;
                                        }
                                        b"SizeC" => {
                                            info.nr_c_stacks =
                                                self.parse_i32(attr.value.as_ref())?;
                                        }
                                        b"SizeT" => {
                                            info.nr_t_stacks =
                                                self.parse_i32(attr.value.as_ref())?;
                                        }
                                        b"SizeZ" => {
                                            info.nr_z_stacks =
                                                self.parse_i32(attr.value.as_ref())?;
                                        }
                                        b"PhysicalSizeXUnit" => {
                                            let unit_x_tmp = attr
                                                .normalized_value(XmlVersion::Implicit1_0)?
                                                .into_owned();
                                            unit_x = LengthUnit::try_from(unit_x_tmp.as_str())?;
                                        }
                                        b"PhysicalSizeYUnit" => {
                                            let unit_y_tmp = attr
                                                .normalized_value(XmlVersion::Implicit1_0)?
                                                .into_owned();
                                            unit_y = LengthUnit::try_from(unit_y_tmp.as_str())?;
                                        }
                                        b"PhysicalSizeZUnit" => {
                                            let unit_z_tmp = attr
                                                .normalized_value(XmlVersion::Implicit1_0)?
                                                .into_owned();
                                            unit_z = LengthUnit::try_from(unit_z_tmp.as_str())?;
                                        }
                                        _ => {}
                                    }
                                }
                                info.pixel_sizes.px_size_x =
                                    info.pixel_sizes.px_size_x * unit_x.to_nanometers_factor();
                                info.pixel_sizes.px_size_y =
                                    info.pixel_sizes.px_size_y * unit_y.to_nanometers_factor();
                                info.pixel_sizes.px_size_z =
                                    info.pixel_sizes.px_size_z * unit_z.to_nanometers_factor();
                            }
                        }
                        b"Channel" if in_pixels => {
                            if let Some(info) = meta.series.get_mut(&current_series_idx) {
                                let mut channel = ChannelInfo::default();
                                let mut emission_wave_length_unit: LengthUnit =
                                    LengthUnit::Nanometer;

                                for attr in e.attributes().flatten() {
                                    match attr.key.local_name().as_ref() {
                                        b"EmissionWavelength" => {
                                            channel.emission_wave_length =
                                                self.parse_f32(attr.value.as_ref())?;
                                        }
                                        b"EmissionWavelengthUnit" => {
                                            let unit_x_tmp = attr
                                                .normalized_value(XmlVersion::Implicit1_0)?
                                                .into_owned();
                                            emission_wave_length_unit =
                                                LengthUnit::try_from(unit_x_tmp.as_str())?;
                                        }
                                        b"ID" => {
                                            channel.id = attr
                                                .normalized_value(XmlVersion::Implicit1_0)?
                                                .into_owned();
                                        }
                                        b"Name" => {
                                            channel.name = attr
                                                .normalized_value(XmlVersion::Implicit1_0)?
                                                .into_owned();
                                        }
                                        b"ContrastMethod" => {
                                            channel.contrast_method = attr
                                                .normalized_value(XmlVersion::Implicit1_0)?
                                                .into_owned();
                                        }
                                        _ => {}
                                    }
                                }

                                let channel_nr = channel
                                    .id
                                    .rsplit_once(':')
                                    .map(|(_, last_part)| last_part.parse::<i32>())
                                    .transpose()? // This brings any parsing error to the surface
                                    .ok_or_else(|| {
                                        InternalErrors::ParseError(format!(
                                            "No colon found in ID: {}",
                                            channel.id
                                        ))
                                    })?;

                                channel.emission_wave_length = channel.emission_wave_length
                                    * emission_wave_length_unit.to_nanometers_factor();

                                info.channels.insert(channel_nr, channel);
                            }
                        }
                        b"JODA" => {
                            in_joda = true;
                        }
                        b"Series" if in_joda => {
                            // Extract JODA series index to match with OME series
                            current_series_idx = e
                                .try_get_attribute("idx")
                                .map_err(|e| InternalErrors::ParseError(e.to_string()))? // Convert XML error
                                .map(|a| self.parse_i32(a.value.as_ref())) // This returns Result<i32, InternalErrors>
                                .transpose()? // Lifts the inner Result out
                                .unwrap_or(-1);
                            // Now you can safely populate PyramidInfo for meta.series[idx]
                        }
                        b"PyramidResolution" if in_joda => {
                            if let Some(info) = meta.series.get_mut(&current_series_idx) {
                                let mut pyramid_info = PyramidInfo::default();
                                let mut idx: i32 = 0;
                                for attr in e.attributes().flatten() {
                                    match attr.key.local_name().as_ref() {
                                        b"idx" => {
                                            idx = self.parse_i32(attr.value.as_ref())?;
                                        }
                                        b"width" => {
                                            pyramid_info.width =
                                                self.parse_u64(attr.value.as_ref())?;
                                        }
                                        b"height" => {
                                            pyramid_info.height =
                                                self.parse_u64(attr.value.as_ref())?;
                                        }
                                        b"TileWidth" => {
                                            pyramid_info.tile_width =
                                                self.parse_u64(attr.value.as_ref())?;
                                        }
                                        b"TileHeight" => {
                                            pyramid_info.tile_height =
                                                self.parse_u64(attr.value.as_ref())?;
                                        }
                                        b"BitsPerPixel" => {
                                            pyramid_info.nr_bits =
                                                self.parse_u8(attr.value.as_ref())?;
                                        }
                                        b"RGBChannelCount" => {
                                            pyramid_info.color_channels =
                                                self.parse_u8(attr.value.as_ref())?;
                                        }
                                        b"IsInterleaved" => {
                                            pyramid_info.is_interleaved =
                                                self.parse_bool(attr.value.as_ref())?;
                                        }
                                        b"IsLittleEndian" => {
                                            pyramid_info.is_little_endian =
                                                self.parse_bool(attr.value.as_ref())?;
                                        }
                                        _ => {}
                                    }
                                }

                                // TODO: This is a trick to find RGB images if we use split channel
                                if self.read_mode == ReadMode::SplitChannels {
                                    if pyramid_info.color_channels == 1
                                        && !pyramid_info.is_interleaved
                                        && info.nr_c_stacks == 3
                                        && pyramid_info.nr_bits == 8
                                    {
                                        pyramid_info.is_rgb = true;

                                        let rgb_names = ["Red", "Green", "Blue"];
                                        let emission_wave_length = [635.0, 532.0, 450.0];
                                        for i in 0..3 {
                                            if let Some(info) = info.channels.get_mut(&i) {
                                                info.name = rgb_names[i as usize].to_string();
                                                info.emission_wave_length =
                                                    emission_wave_length[i as usize];
                                            }
                                        }
                                    }
                                } else {
                                    pyramid_info.is_rgb = pyramid_info.color_channels > 2;
                                }

                                info.resolutions.insert(idx, pyramid_info);
                            } else {
                                println!("No series yet");
                            }
                        }
                        _ => {}
                    }
                }
                Ok(Event::End(ref e)) => {
                    if e.local_name().as_ref() == b"JODA" {
                        in_joda = false;
                    }
                    if e.local_name().as_ref() == b"Pixels" {
                        in_pixels = false;
                    }
                    if e.local_name().as_ref() == b"Instrument" {
                        in_instrument = false;
                    }
                }
                Ok(Event::Eof) => break,
                Err(e) => return Err(InternalErrors::from(e)),
                _ => {}
            }
            buf.clear();
        }
        Ok(meta)
    }

    fn parse_f32(&self, bytes: &[u8]) -> Result<f32, InternalErrors> {
        let s = std::str::from_utf8(bytes)
            .map_err(|e| InternalErrors::ParseError(format!("UTF8 Error: {}", e)))?;

        s.parse::<f32>()
            .map_err(|e| InternalErrors::ParseError(format!("Float Parse Error: {}", e)))
    }

    fn parse_i32(&self, bytes: &[u8]) -> Result<i32, InternalErrors> {
        let s = std::str::from_utf8(bytes)
            .map_err(|e| InternalErrors::ParseError(format!("UTF8 Error: {}", e)))?;

        s.parse::<i32>()
            .map_err(|e| InternalErrors::ParseError(format!("i32 Parse Error: {}", e)))
    }

    fn parse_u8(&self, bytes: &[u8]) -> Result<u8, InternalErrors> {
        let s = std::str::from_utf8(bytes)
            .map_err(|e| InternalErrors::ParseError(format!("UTF8 Error: {}", e)))?;

        s.parse::<u8>()
            .map_err(|e| InternalErrors::ParseError(format!("u8 Parse Error: {}", e)))
    }

    fn parse_i64(&self, bytes: &[u8]) -> Result<i64, InternalErrors> {
        let s = std::str::from_utf8(bytes)
            .map_err(|e| InternalErrors::ParseError(format!("UTF8 Error: {}", e)))?;

        s.parse::<i64>()
            .map_err(|e| InternalErrors::ParseError(format!("i64 Parse Error: {}", e)))
    }

    fn parse_u64(&self, bytes: &[u8]) -> Result<u64, InternalErrors> {
        let s = std::str::from_utf8(bytes)
            .map_err(|e| InternalErrors::ParseError(format!("UTF8 Error: {}", e)))?;

        s.parse::<u64>()
            .map_err(|e| InternalErrors::ParseError(format!("i64 Parse Error: {}", e)))
    }

    fn parse_bool(&self, bytes: &[u8]) -> Result<bool, InternalErrors> {
        let s = std::str::from_utf8(bytes)
            .map_err(|e| InternalErrors::ParseError(format!("UTF8 Error: {}", e)))?
            .trim() // Remove potential whitespace
            .to_lowercase(); // Handle "True", "TRUE", etc.

        match s.as_str() {
            "true" | "1" => Ok(true),
            "false" | "0" => Ok(false),
            _ => Err(InternalErrors::ParseError(format!(
                "Invalid boolean value: '{}'. Expected true/false or 1/0",
                s
            ))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    /// Builds an `ImageReader` with no live JVM/Bio-Formats session -
    /// `parse_ome_xml` is pure string-in/struct-out and only reads
    /// `current_path`/`read_mode`, so it can be unit-tested without opening
    /// a real image.
    fn reader_at(path: &str, mode: ReadMode) -> ImageReader {
        ImageReader {
            wrapper_instance: None,
            read_mode: mode,
            image_meta: std::sync::Arc::new(ImageMeta::default()),
            current_path: PathBuf::from(path),
        }
    }

    /// The real OME-XML (+ JODA pyramid extension) Bio-Formats produces for
    /// `crates/core/tests/multi-channel-4D-series.ome.tif`, the same fixture
    /// several JNI-backed tests in `image_reader.rs` open for real.
    fn real_fixture_xml() -> String {
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/ome.xml");
        std::fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()))
    }

    #[test]
    fn real_fixture_parses_series_channels_and_pyramid_resolution() {
        let reader = reader_at(
            "/some/dir/multi-channel-4D-series.ome.tif",
            ReadMode::Default,
        );
        let meta = reader
            .parse_ome_xml(&real_fixture_xml())
            .expect("real fixture must parse");

        // Name comes from the reader's own path, not the XML's Image/@Name.
        assert_eq!(meta.name, "multi-channel-4D-series.ome.tif");

        assert_eq!(meta.series.len(), 1);
        let series = meta.series.get(&0).expect("series 0 must exist");
        assert_eq!(series.nr_c_stacks, 3);
        assert_eq!(series.nr_t_stacks, 7);
        assert_eq!(series.nr_z_stacks, 5);

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
        assert_eq!(res.tile_width, 439);
        assert_eq!(res.tile_height, 1);
        assert_eq!(res.nr_bits, 8);
        assert_eq!(res.color_channels, 1);
        assert!(!res.is_interleaved);
        assert!(!res.is_little_endian);
        // color_channels (1) is not > 2, so the ReadMode::Default RGB rule
        // (`color_channels > 2`) does not mark this - correctly - as RGB.
        assert!(!res.is_rgb);
    }

    /// Regression test pinning the exact trigger for the "Image decoder can
    /// panic on real files" issue: a `<PyramidResolution>` that omits
    /// `BitsPerPixel` (plausible for a pyramid sub-resolution level in a
    /// real-world file) leaves `nr_bits` at its `u8` default of 0, which
    /// `image_reader.rs`'s decode path then divides/chunks by. This test
    /// only pins the parser's actual behaviour (nr_bits stays 0, no error is
    /// raised) - it intentionally does not assert a fix, since the decoder
    /// fix is tracked separately.
    #[test]
    fn pyramid_resolution_missing_bits_per_pixel_leaves_nr_bits_at_zero() {
        let xml = r#"<?xml version="1.0" encoding="UTF-8"?>
<OME xmlns="http://www.openmicroscopy.org/Schemas/OME/2016-06">
  <Image ID="Image:0">
    <Pixels ID="Pixels:0" SizeC="1" SizeT="1" SizeZ="1" SizeX="10" SizeY="10"/>
  </Image>
</OME>
<JODA xmlns="https://www.imagec.org/" SeriesCount="1">
  <Series idx="0" ResolutionCount="1">
    <PyramidResolution idx="0" width="10" height="10" TileWidth="10" TileHeight="10" RGBChannelCount="1" IsInterleaved="0" IsLittleEndian="0"/>
  </Series>
</JODA>"#;

        let reader = reader_at("/some/dir/thumbnail.tif", ReadMode::Default);
        let meta = reader
            .parse_ome_xml(xml)
            .expect("missing BitsPerPixel must not itself be a parse error");

        let res = meta.series[&0]
            .resolutions
            .get(&0)
            .expect("resolution 0 must exist");
        assert_eq!(
            res.nr_bits, 0,
            "nr_bits must default to 0 when BitsPerPixel is absent"
        );
    }

    #[test]
    fn physical_pixel_sizes_and_channel_wavelengths_are_converted_to_nanometers() {
        let xml = r#"<?xml version="1.0" encoding="UTF-8"?>
<OME xmlns="http://www.openmicroscopy.org/Schemas/OME/2016-06">
  <Image ID="Image:0">
    <Pixels ID="Pixels:0" SizeC="2" SizeT="1" SizeZ="1"
        PhysicalSizeX="0.5" PhysicalSizeXUnit="µm"
        PhysicalSizeY="0.5" PhysicalSizeYUnit="µm"
        PhysicalSizeZ="1.2" PhysicalSizeZUnit="µm">
      <Channel ID="Channel:0:0" Name="DAPI" EmissionWavelength="461" EmissionWavelengthUnit="nm"/>
      <Channel ID="Channel:0:1" Name="GFP" EmissionWavelength="0.509" EmissionWavelengthUnit="µm"/>
    </Pixels>
  </Image>
</OME>"#;

        let reader = reader_at("/some/dir/img.tif", ReadMode::Default);
        let meta = reader.parse_ome_xml(xml).unwrap();
        let series = &meta.series[&0];

        assert_eq!(series.pixel_sizes.px_size_x, 500.0);
        assert_eq!(series.pixel_sizes.px_size_y, 500.0);
        assert_eq!(series.pixel_sizes.px_size_z, 1200.0);

        let ch0 = &series.channels[&0];
        assert_eq!(ch0.name, "DAPI");
        assert_eq!(ch0.emission_wave_length, 461.0);

        let ch1 = &series.channels[&1];
        assert_eq!(ch1.name, "GFP");
        assert_eq!(
            ch1.emission_wave_length, 509.0,
            "0.509 µm must convert to 509 nm"
        );
    }

    #[test]
    fn split_channels_rgb_heuristic_renames_channels_when_it_matches() {
        let xml = r#"<?xml version="1.0" encoding="UTF-8"?>
<OME xmlns="http://www.openmicroscopy.org/Schemas/OME/2016-06">
  <Image ID="Image:0">
    <Pixels ID="Pixels:0" SizeC="3" SizeT="1" SizeZ="1">
      <Channel ID="Channel:0:0"/>
      <Channel ID="Channel:0:1"/>
      <Channel ID="Channel:0:2"/>
    </Pixels>
  </Image>
</OME>
<JODA xmlns="https://www.imagec.org/" SeriesCount="1">
  <Series idx="0" ResolutionCount="1">
    <PyramidResolution idx="0" width="4" height="4" TileWidth="4" TileHeight="4" BitsPerPixel="8" RGBChannelCount="1" IsInterleaved="0" IsLittleEndian="0"/>
  </Series>
</JODA>"#;

        let reader = reader_at("/some/dir/rgb.tif", ReadMode::SplitChannels);
        let meta = reader.parse_ome_xml(xml).unwrap();
        let series = &meta.series[&0];

        let res = series.resolutions.get(&0).unwrap();
        assert!(
            res.is_rgb,
            "1 color channel + non-interleaved + 3 C-stacks + 8-bit must trip the RGB heuristic"
        );

        assert_eq!(series.channels[&0].name, "Red");
        assert_eq!(series.channels[&1].name, "Green");
        assert_eq!(series.channels[&2].name, "Blue");
        assert_eq!(series.channels[&0].emission_wave_length, 635.0);
        assert_eq!(series.channels[&1].emission_wave_length, 532.0);
        assert_eq!(series.channels[&2].emission_wave_length, 450.0);
    }

    #[test]
    fn split_channels_rgb_heuristic_does_not_trigger_in_default_read_mode() {
        // Identical pyramid metadata to the test above, but read in
        // ReadMode::Default: only `Self::color_channels > 2` decides `is_rgb`
        // there, and color_channels is 1, so no channel gets renamed.
        let xml = r#"<?xml version="1.0" encoding="UTF-8"?>
<OME xmlns="http://www.openmicroscopy.org/Schemas/OME/2016-06">
  <Image ID="Image:0">
    <Pixels ID="Pixels:0" SizeC="3" SizeT="1" SizeZ="1">
      <Channel ID="Channel:0:0" Name="Ch0"/>
    </Pixels>
  </Image>
</OME>
<JODA xmlns="https://www.imagec.org/" SeriesCount="1">
  <Series idx="0" ResolutionCount="1">
    <PyramidResolution idx="0" width="4" height="4" TileWidth="4" TileHeight="4" BitsPerPixel="8" RGBChannelCount="1" IsInterleaved="0" IsLittleEndian="0"/>
  </Series>
</JODA>"#;

        let reader = reader_at("/some/dir/rgb.tif", ReadMode::Default);
        let meta = reader.parse_ome_xml(xml).unwrap();
        let series = &meta.series[&0];

        assert!(!series.resolutions[&0].is_rgb);
        assert_eq!(
            series.channels[&0].name, "Ch0",
            "channel must keep its parsed name, not be renamed"
        );
    }

    #[test]
    fn multiple_images_are_tracked_as_independent_series() {
        let xml = r#"<?xml version="1.0" encoding="UTF-8"?>
<OME xmlns="http://www.openmicroscopy.org/Schemas/OME/2016-06">
  <Image ID="Image:0">
    <Pixels ID="Pixels:0" SizeC="1" SizeT="1" SizeZ="1"/>
  </Image>
  <Image ID="Image:1">
    <Pixels ID="Pixels:1" SizeC="4" SizeT="2" SizeZ="3"/>
  </Image>
</OME>"#;

        let reader = reader_at("/some/dir/multi-series.tif", ReadMode::Default);
        let meta = reader.parse_ome_xml(xml).unwrap();

        assert_eq!(meta.series.len(), 2);
        assert_eq!(meta.series[&0].nr_c_stacks, 1);
        assert_eq!(meta.series[&1].nr_c_stacks, 4);
        assert_eq!(meta.series[&1].nr_t_stacks, 2);
        assert_eq!(meta.series[&1].nr_z_stacks, 3);
    }

    #[test]
    fn image_id_without_a_colon_is_a_parse_error_not_a_panic() {
        let xml = r#"<OME><Image ID="NoColonHere"><Pixels ID="Pixels:0"/></Image></OME>"#;
        let reader = reader_at("/x.tif", ReadMode::Default);
        let err = reader
            .parse_ome_xml(xml)
            .err()
            .expect("missing ':' in Image/@ID must error");
        assert!(matches!(err, InternalErrors::ParseError(_)));
    }

    #[test]
    fn channel_id_without_a_colon_is_a_parse_error_not_a_panic() {
        let xml = r#"<OME><Image ID="Image:0"><Pixels ID="Pixels:0" SizeC="1"><Channel ID="NoColonHere"/></Pixels></Image></OME>"#;
        let reader = reader_at("/x.tif", ReadMode::Default);
        let err = reader
            .parse_ome_xml(xml)
            .err()
            .expect("missing ':' in Channel/@ID must error");
        assert!(matches!(err, InternalErrors::ParseError(_)));
    }

    #[test]
    fn non_numeric_image_id_suffix_is_a_parse_error_not_a_panic() {
        let xml = r#"<OME><Image ID="Image:not-a-number"><Pixels ID="Pixels:0"/></Image></OME>"#;
        let reader = reader_at("/x.tif", ReadMode::Default);
        let err = reader
            .parse_ome_xml(xml)
            .err()
            .expect("non-numeric series index must error");
        // A colon is present, so this goes through `val.parse::<i32>()?` rather
        // than the `ok_or_else(ParseError)` "no colon at all" branch above -
        // the `?` auto-converts the `ParseIntError` via `InternalErrors`'s
        // `#[from]` impl, so it surfaces as `ParseIntError`, not `ParseError`.
        assert!(matches!(err, InternalErrors::ParseIntError(_)));
    }

    #[test]
    fn unknown_physical_size_unit_is_a_parse_error_not_a_panic() {
        let xml = r#"<OME><Image ID="Image:0"><Pixels ID="Pixels:0" SizeC="1" PhysicalSizeX="1.0" PhysicalSizeXUnit="px"/></Image></OME>"#;
        let reader = reader_at("/x.tif", ReadMode::Default);
        let err = reader
            .parse_ome_xml(xml)
            .err()
            .expect("unsupported unit 'px' must error, not silently default");
        assert!(matches!(err, InternalErrors::ParseError(_)));
    }

    #[test]
    fn malformed_numeric_attribute_is_a_parse_error_not_a_panic() {
        let xml = r#"<OME><Image ID="Image:0"><Pixels ID="Pixels:0" SizeC="not-a-number"/></Image></OME>"#;
        let reader = reader_at("/x.tif", ReadMode::Default);
        let err = reader
            .parse_ome_xml(xml)
            .err()
            .expect("non-numeric SizeC must error");
        assert!(matches!(err, InternalErrors::ParseError(_)));
    }

    #[test]
    fn empty_document_yields_default_metadata_named_from_the_path() {
        let reader = reader_at("/some/dir/unnamed.tif", ReadMode::Default);
        let meta = reader.parse_ome_xml("").unwrap();
        assert_eq!(meta.name, "unnamed.tif");
        assert!(meta.series.is_empty());
    }
}
