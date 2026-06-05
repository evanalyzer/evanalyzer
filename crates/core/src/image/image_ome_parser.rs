use crate::converters::LengthUnit;
use crate::image::image_meta::{ChannelInfo, ImageInfo, ImageMeta, PyramidInfo};
use crate::{ImageReader, ReadMode};
use evanalyzer_cfg::core_types::InternalErrors;
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
                                        image_id = attr.unescape_value()?.into_owned();
                                    }
                                    b"Name" => {
                                        _image_name = attr.unescape_value()?.into_owned();
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
                                        meta.objective.manufacturer =
                                            attr.unescape_value()?.into_owned();
                                    }
                                    b"Model" => {
                                        meta.objective.model = attr.unescape_value()?.into_owned();
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
                                            let unit_x_tmp = attr.unescape_value()?.into_owned();
                                            unit_x = LengthUnit::try_from(unit_x_tmp.as_str())?;
                                        }
                                        b"PhysicalSizeYUnit" => {
                                            let unit_y_tmp = attr.unescape_value()?.into_owned();
                                            unit_y = LengthUnit::try_from(unit_y_tmp.as_str())?;
                                        }
                                        b"PhysicalSizeZUnit" => {
                                            let unit_z_tmp = attr.unescape_value()?.into_owned();
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
                                            let unit_x_tmp = attr.unescape_value()?.into_owned();
                                            emission_wave_length_unit =
                                                LengthUnit::try_from(unit_x_tmp.as_str())?;
                                        }
                                        b"ID" => {
                                            channel.id = attr.unescape_value()?.into_owned();
                                        }
                                        b"Name" => {
                                            channel.name = attr.unescape_value()?.into_owned();
                                        }
                                        b"ContrastMethod" => {
                                            channel.contrast_method =
                                                attr.unescape_value()?.into_owned();
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
