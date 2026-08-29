use crate::pipeline::pipeline_cache::GlobalPipelineCache;
use crate::storage::PipelineResultExporter;
use evanalyzer_cfg::core_types::{InternalErrors, ObjectClass};
use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::path::PathBuf;
use std::sync::Mutex;

/// Holds the open output file across `export()` calls, plus whether the
/// header row still needs to be written - `export()` is called once per
/// tile/z/t work unit (potentially thousands of times per job), so opening
/// the file fresh on every call would mean thousands of redundant
/// open/append/flush cycles for what is really one long-lived output stream.
struct CsvWriterState {
    writer: csv::Writer<File>,
    header_written: bool,
    /// The exact channel ids / colocalization classes this exporter's rows
    /// are shaped around - established once, by the first `export()` call,
    /// and reused for every row after, so the column count can never drift.
    /// Previously each `export()` call recomputed its own channel/class set
    /// from just that call's objects and sized its row to match - different
    /// tiles/images can have different channels/classes present, so a later
    /// call's row could end up with more or fewer columns than the header
    /// declared, silently misaligning every column after the first
    /// divergence. A call whose objects introduce a genuinely new
    /// channel/class after this is established has that data dropped from
    /// the row instead - there's no column for it to go in.
    ///
    /// `None` until established - kept separate from `header_written`
    /// because when appending to an already-existing file, `header_written`
    /// starts `true` but this process still has no way to know what columns
    /// that existing header actually has (it isn't re-parsed); this still
    /// gives every row *this process* writes a single consistent shape, even
    /// though it can't recover a previous process's exact column set.
    columns: Option<(Vec<i32>, Vec<ObjectClass>)>,
}

pub struct CsvExporter {
    // csv::Writer<File> is Send but not Sync; the Mutex makes the struct Sync
    // so it can satisfy the `PipelineResultExporter: Send + Sync` bound.
    state: Mutex<CsvWriterState>,
    /// Maps ObjectClass → human-readable name from project classification settings.
    pub class_names: HashMap<ObjectClass, String>,
}

impl CsvExporter {
    /// Opens (or creates, for appending) the output file once and returns a
    /// ready exporter. If the file already existed and had content, the
    /// header row is assumed to already be present and won't be rewritten.
    pub fn new(
        output_path: impl Into<PathBuf>,
        class_names: HashMap<ObjectClass, String>,
    ) -> Result<Self, InternalErrors> {
        let output_path: PathBuf = output_path.into();
        let header_written = output_path.exists();

        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&output_path)
            .map_err(|e| InternalErrors::Io(format!("IO Error: {}", e)))?;

        Ok(Self {
            state: Mutex::new(CsvWriterState {
                writer: csv::Writer::from_writer(file),
                header_written,
                columns: None,
            }),
            class_names,
        })
    }

    fn class_label(&self, class: &ObjectClass) -> String {
        match class {
            ObjectClass::Unset => "unset".to_string(),
            ObjectClass::Valid(n) => {
                if let Some(name) = self.class_names.get(class) {
                    format!("{} ({})", name, n)
                } else {
                    format!("class_{}", n)
                }
            }
        }
    }

    fn coloc_col_name(&self, class: &ObjectClass) -> String {
        format!("coloc_with_{}", self.class_label(class))
    }
}

impl PipelineResultExporter for CsvExporter {
    fn export(&self, cache: &GlobalPipelineCache) -> Result<(), InternalErrors> {
        let mut state = self
            .state
            .lock()
            .map_err(|_| InternalErrors::Io("Failed to acquire CsvExporter lock".to_string()))?;

        let px = &cache.image_meta.pixel_sizes;

        // --- Phase 1/2: establish (once) the channel/coloc-class set every
        // row this exporter writes will be shaped around - see the
        // `columns` field doc comment for why this only happens once.
        let (channel_ids, coloc_classes) = match &state.columns {
            Some(columns) => columns.clone(),
            None => {
                let mut channel_ids: Vec<i32> = cache
                    .object_cache
                    .values()
                    .flat_map(|object| object.intensities.keys().cloned())
                    .collect();
                channel_ids.sort();
                channel_ids.dedup();

                let mut coloc_classes: Vec<ObjectClass> = cache
                    .object_cache
                    .values()
                    .flat_map(|object| object.colocalized_with.keys().cloned())
                    .collect();
                coloc_classes.sort_by_key(|c| format!("{:?}", c));
                coloc_classes.dedup();

                state.columns = Some((channel_ids.clone(), coloc_classes.clone()));
                (channel_ids, coloc_classes)
            }
        };

        // --- Phase 3: Header Assembly ---
        if !state.header_written {
            let mut header = vec![
                // Image & Plane Info
                "image".to_string(),
                "channel".to_string(),
                "z_stack".to_string(),
                "t_stack".to_string(),
                // Object Identity & Lineage
                "object_id".to_string(),
                "segmentation_class".to_string(),
                "object_class".to_string(),
                "parent_id".to_string(),
                "children".to_string(),
                "track_id".to_string(),
                // Centroid
                "centroid_x_px".to_string(),
                "centroid_y_px".to_string(),
                "centroid_x_nm".to_string(),
                "centroid_y_nm".to_string(),
                // Bounding Box
                "bbox_xmin_px".to_string(),
                "bbox_ymin_px".to_string(),
                "bbox_xmax_px".to_string(),
                "bbox_ymax_px".to_string(),
                "bbox_xmin_nm".to_string(),
                "bbox_ymin_nm".to_string(),
                "bbox_xmax_nm".to_string(),
                "bbox_ymax_nm".to_string(),
                // Area
                "area_px".to_string(),
                "area_nm2".to_string(),
                // Perimeter
                "perimeter_px".to_string(),
                "perimeter_nm".to_string(),
                // Shape Descriptors
                "circularity".to_string(),
                "solidity".to_string(),
                "aspect_ratio".to_string(),
                "roundness".to_string(),
                "compactness".to_string(),
                // Ellipse Fitting
                "major_axis_px".to_string(),
                "minor_axis_px".to_string(),
                "major_axis_nm".to_string(),
                "minor_axis_nm".to_string(),
                "major_axis_angle".to_string(),
                "eccentricity".to_string(),
                // Feret Diameter
                "feret_diameter_px".to_string(),
                "min_feret_diameter_px".to_string(),
                "feret_diameter_nm".to_string(),
                "min_feret_diameter_nm".to_string(),
                // Boundary
                "touches_edge".to_string(),
                // Pixel Sizes (nm/pixel)
                "pixel_size_x_nm".to_string(),
                "pixel_size_y_nm".to_string(),
                "pixel_size_z_nm".to_string(),
                // Image bit depth
                "image_bit_depth".to_string(),
            ];

            for ch in &channel_ids {
                // Raw values are stored in [0, 1]; scaled values are in [0, 2^bit_depth - 1]
                header.push(format!("ch{}_integrated_density_raw", ch));
                header.push(format!("ch{}_integrated_density_scaled", ch));
                header.push(format!("ch{}_mean_intensity_raw", ch));
                header.push(format!("ch{}_mean_intensity_scaled", ch));
                header.push(format!("ch{}_min_intensity_raw", ch));
                header.push(format!("ch{}_min_intensity_scaled", ch));
                header.push(format!("ch{}_max_intensity_raw", ch));
                header.push(format!("ch{}_max_intensity_scaled", ch));
            }

            for class in &coloc_classes {
                header.push(self.coloc_col_name(class));
            }

            state
                .writer
                .write_record(&header)
                .map_err(|e| InternalErrors::Io(e.to_string()))?;
            state.header_written = true;
        }

        // --- Phase 4: Data Row Serialization ---
        let px_len = (px.px_size_x * px.px_size_y).sqrt();
        let nr_of_bits = cache.image_meta.nr_of_bits;
        // Same implausible-bit-depth guard as image_reader.rs's read path -
        // unguarded, `1u64 << nr_of_bits` for nr_of_bits > 63 is a
        // shift-by-too-large, silently producing a wrong scale factor
        // instead of an error.
        if !(1..=32).contains(&nr_of_bits) {
            return Err(InternalErrors::Generic(format!(
                "cannot export {}: implausible bit depth {nr_of_bits} (expected 1-32)",
                cache.image_rel_path.display()
            )));
        }
        // Max pixel value for the bit depth (e.g. 65535 for 16-bit)
        let bit_max = ((1u64 << nr_of_bits) - 1) as f64;

        for object in cache.object_cache.values() {
            let perimeter = object.get_perimeter();
            let ellipse = object.get_ellipse();
            let solidity = object.get_solidity();
            let centroid = object.get_centroid();
            let feret = object.get_feret_diameter();
            let min_feret = object.get_min_feret_diameter();

            let parent_id = object
                .parent_id
                .as_ref()
                .map(|id| id.to_string())
                .unwrap_or_default();

            let children = object
                .children
                .iter()
                .map(|id| id.to_string())
                .collect::<Vec<_>>()
                .join(",");

            let mut row = vec![
                format!("{:?}", cache.image_rel_path),
                format!("{}", object.plane.c),
                format!("{}", object.plane.z),
                format!("{}", object.plane.t),
                // Identity & Lineage
                object.id.to_string(),
                object.segmentation_class.to_string(),
                object
                    .object_class
                    .iter()
                    .map(|c| self.class_label(c))
                    .collect::<Vec<_>>()
                    .join(","),
                parent_id,
                children,
                object.track.id.0.to_string(),
                // Centroid px & nm
                format!("{:.2}", centroid.0),
                format!("{:.2}", centroid.1),
                format!("{:.2}", centroid.0 as f64 * px.px_size_x as f64),
                format!("{:.2}", centroid.1 as f64 * px.px_size_y as f64),
                // Bounding box px & nm
                object.bbox[0].to_string(),
                object.bbox[1].to_string(),
                object.bbox[2].to_string(),
                object.bbox[3].to_string(),
                format!("{:.2}", object.bbox[0] as f32 * px.px_size_x),
                format!("{:.2}", object.bbox[1] as f32 * px.px_size_y),
                format!("{:.2}", object.bbox[2] as f32 * px.px_size_x),
                format!("{:.2}", object.bbox[3] as f32 * px.px_size_y),
                // Area px & nm²
                object.area.to_string(),
                format!("{:.2}", object.area as f32 * px.px_size_x * px.px_size_y),
                // Perimeter px & nm
                format!("{:.2}", perimeter),
                format!("{:.2}", perimeter * px_len),
                // Shape descriptors
                format!("{:.4}", object.circularity()),
                format!("{:.4}", solidity),
                format!("{:.4}", object.get_aspect_ratio()),
                format!("{:.4}", object.get_roundness(perimeter)),
                format!("{:.4}", object.get_compactness(perimeter)),
                // Ellipse px & nm
                format!("{:.2}", ellipse.major),
                format!("{:.2}", ellipse.minor),
                format!("{:.2}", ellipse.major * px_len),
                format!("{:.2}", ellipse.minor * px_len),
                format!("{:.2}", ellipse.angle),
                format!("{:.4}", ellipse.eccentricity),
                // Feret px & nm
                format!("{:.2}", feret),
                format!("{:.2}", min_feret),
                format!("{:.2}", feret * px_len),
                format!("{:.2}", min_feret * px_len),
                // Boundary
                object.touches_edge.to_string(),
                // Pixel sizes
                format!("{:.4}", px.px_size_x),
                format!("{:.4}", px.px_size_y),
                format!("{:.4}", px.px_size_z),
                nr_of_bits.to_string(),
            ];

            for ch in &channel_ids {
                if let Some(intensity) = object.intensities.get(ch) {
                    let mean_raw = intensity.sum_intensity / (object.area as f64).max(1.0);
                    let min_raw = intensity.min_intensity as f64;
                    let max_raw = intensity.max_intensity as f64;

                    row.push(format!("{:.6}", intensity.sum_intensity));
                    row.push(format!("{:.2}", intensity.sum_intensity * bit_max));
                    row.push(format!("{:.6}", mean_raw));
                    row.push(format!("{:.2}", mean_raw * bit_max));
                    row.push(format!("{:.6}", min_raw));
                    row.push(format!("{:.2}", min_raw * bit_max));
                    row.push(format!("{:.6}", max_raw));
                    row.push(format!("{:.2}", max_raw * bit_max));
                } else {
                    // 8 empty cells (4 metrics × 2 scales)
                    row.extend(std::iter::repeat_n("".to_string(), 8));
                }
            }

            for class in &coloc_classes {
                if let Some(ids) = object.colocalized_with.get(class) {
                    let id_strings: Vec<String> = ids.iter().map(|id| id.to_string()).collect();
                    row.push(id_strings.join(","));
                } else {
                    row.push("".to_string());
                }
            }

            state
                .writer
                .write_record(&row)
                .map_err(|e| InternalErrors::ImageReadError(e.to_string()))?;
        }

        state
            .writer
            .flush()
            .map_err(|e| InternalErrors::ImageReadError(e.to_string()))?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::object::{Intensity, Track};
    use evanalyzer_cfg::core_types::{ObjectId, TrackId};
    use indexmap::IndexMap;
    use std::collections::HashSet;
    use std::fs;
    use tempfile::TempDir;

    /// Parses `content` as a CSV with a header row, returning the header record
    /// plus every data record. Small local helper so value-by-column-name
    /// lookups below don't rely on brittle positional/string matching.
    fn parse_csv(content: &str) -> (csv::StringRecord, Vec<csv::StringRecord>) {
        let mut reader = csv::Reader::from_reader(content.as_bytes());
        let headers = reader
            .headers()
            .expect("CSV should have a header row")
            .clone();
        let rows = reader
            .records()
            .map(|r| r.expect("valid CSV record"))
            .collect();
        (headers, rows)
    }

    /// Looks up `name`'s value in `row` by header name.
    fn column<'a>(headers: &csv::StringRecord, row: &'a csv::StringRecord, name: &str) -> &'a str {
        let idx = headers
            .iter()
            .position(|h| h == name)
            .unwrap_or_else(|| panic!("column `{name}` not found; headers: {:?}", headers));
        row.get(idx).unwrap_or_default()
    }

    /// Finds the single data row whose `object_id` column matches `object_id`.
    fn row_by_object_id<'a>(
        headers: &csv::StringRecord,
        rows: &'a [csv::StringRecord],
        object_id: &str,
    ) -> &'a csv::StringRecord {
        rows.iter()
            .find(|row| column(headers, row, "object_id") == object_id)
            .unwrap_or_else(|| panic!("no row found with object_id = {object_id}"))
    }

    #[test]
    fn test_csv_export_creates_file() {
        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let output_path = temp_dir.path().join("test_export.csv");

        let exporter =
            CsvExporter::new(output_path.clone(), HashMap::new()).expect("exporter init failed");

        let mut cache = GlobalPipelineCache::default();
        cache.image_rel_path = PathBuf::from("test_image.tif");

        let result = exporter.export(&cache);
        assert!(result.is_ok(), "Export should succeed");
        assert!(output_path.exists(), "CSV file should be created");
    }

    #[test]
    fn test_csv_export_includes_morphological_metrics() {
        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let output_path = temp_dir.path().join("test_morphology.csv");

        let exporter =
            CsvExporter::new(output_path.clone(), HashMap::new()).expect("exporter init failed");

        let cache = GlobalPipelineCache::default();
        let _ = exporter.export(&cache);

        let content = fs::read_to_string(&output_path).expect("Failed to read CSV");
        assert!(
            content.contains("circularity"),
            "Should contain circularity"
        );
        assert!(content.contains("solidity"), "Should contain solidity");
        assert!(
            content.contains("aspect_ratio"),
            "Should contain aspect_ratio"
        );
        assert!(
            content.contains("feret_diameter"),
            "Should contain feret_diameter"
        );
        assert!(content.contains("centroid_x"), "Should contain centroid_x");
    }

    #[test]
    fn test_csv_export_across_multiple_calls_writes_header_once_and_appends_every_row() {
        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let output_path = temp_dir.path().join("test_multi_call.csv");

        let exporter =
            CsvExporter::new(output_path.clone(), HashMap::new()).expect("exporter init failed");

        let mut cache_a = GlobalPipelineCache::default();
        cache_a.image_rel_path = PathBuf::from("image_a.tif");
        let mut object_a = crate::object::Object::new(crate::object::ObjectInit {
            id: evanalyzer_cfg::core_types::ObjectId::next(),
            area: 10,
            bbox: [0, 0, 3, 3],
            ..Default::default()
        });
        object_a.plane.t = 0;
        cache_a.object_cache.insert(object_a.id.clone(), object_a);

        let mut cache_b = GlobalPipelineCache::default();
        cache_b.image_rel_path = PathBuf::from("image_b.tif");
        let mut object_b = crate::object::Object::new(crate::object::ObjectInit {
            id: evanalyzer_cfg::core_types::ObjectId::next(),
            area: 20,
            bbox: [1, 1, 4, 4],
            ..Default::default()
        });
        object_b.plane.t = 0;
        cache_b.object_cache.insert(object_b.id.clone(), object_b);

        exporter
            .export(&cache_a)
            .expect("first export should succeed");
        exporter
            .export(&cache_b)
            .expect("second export should succeed");

        let content = fs::read_to_string(&output_path).expect("Failed to read CSV");
        let lines: Vec<&str> = content.lines().collect();
        assert_eq!(
            lines.len(),
            3,
            "expected one header row plus one data row per export() call, got: {content}"
        );
        assert_eq!(
            lines
                .iter()
                .filter(|l| l.starts_with("image,channel"))
                .count(),
            1,
            "the header row must be written exactly once across multiple export() calls"
        );
    }

    #[test]
    fn test_csv_export_keeps_every_row_aligned_with_the_header_even_when_later_calls_have_a_different_channel_set()
     {
        // Regression test: the header (and its channel/coloc-class columns)
        // used to be fixed from the first export() call, but every row's
        // *column count* was independently recomputed per call from just
        // that call's objects - a later call with a different channel set
        // than the first would silently misalign every column after the
        // channel columns.
        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let output_path = temp_dir.path().join("test_ragged_columns.csv");

        let exporter =
            CsvExporter::new(output_path.clone(), HashMap::new()).expect("exporter init failed");

        // First call: an object with intensities on channels 0 AND 1 -
        // establishes the header with both channels' columns.
        let mut cache_a = GlobalPipelineCache::default();
        cache_a.image_rel_path = PathBuf::from("image_a.tif");
        let mut object_a = crate::object::Object::new(crate::object::ObjectInit {
            id: evanalyzer_cfg::core_types::ObjectId::next(),
            area: 10,
            bbox: [0, 0, 3, 3],
            ..Default::default()
        });
        object_a.intensities.insert(0, Intensity::default());
        object_a.intensities.insert(1, Intensity::default());
        cache_a.object_cache.insert(object_a.id.clone(), object_a);

        // Second call: an object with intensities on channel 0 only - if
        // this were (incorrectly) used to size the row instead of the
        // header's established channel set, this row would be 8 columns
        // short.
        let mut cache_b = GlobalPipelineCache::default();
        cache_b.image_rel_path = PathBuf::from("image_b.tif");
        let mut object_b = crate::object::Object::new(crate::object::ObjectInit {
            id: evanalyzer_cfg::core_types::ObjectId::next(),
            area: 20,
            bbox: [1, 1, 4, 4],
            ..Default::default()
        });
        object_b.intensities.insert(0, Intensity::default());
        cache_b.object_cache.insert(object_b.id.clone(), object_b);

        exporter
            .export(&cache_a)
            .expect("first export should succeed");
        exporter
            .export(&cache_b)
            .expect("second export should succeed");

        let mut reader = csv::Reader::from_path(&output_path).expect("failed to open CSV");
        let header_len = reader.headers().expect("failed to read header").len();

        let mut row_count = 0;
        for record in reader.records() {
            let record = record.expect("failed to read row");
            row_count += 1;
            assert_eq!(
                record.len(),
                header_len,
                "row {row_count} has {} columns, header has {header_len} - columns are misaligned",
                record.len()
            );
        }
        assert_eq!(row_count, 2, "expected exactly one row per export() call");
    }

    #[test]
    fn test_class_label_uses_class_name_when_present_falls_back_otherwise_and_handles_unset() {
        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let output_path = temp_dir.path().join("test_class_label.csv");

        // Only class Valid(5) has a human-readable name configured.
        let mut class_names = HashMap::new();
        class_names.insert(ObjectClass::Valid(5), "Positive".to_string());

        let exporter =
            CsvExporter::new(output_path.clone(), class_names).expect("exporter init failed");

        let mut cache = GlobalPipelineCache::default();
        cache.image_rel_path = PathBuf::from("classes.tif");

        let named_id = ObjectId::next();
        let named = crate::object::Object::new(crate::object::ObjectInit {
            id: named_id.clone(),
            object_class: HashSet::from([ObjectClass::Valid(5)]),
            area: 10,
            bbox: [0, 0, 3, 3],
            ..Default::default()
        });

        let unnamed_id = ObjectId::next();
        let unnamed = crate::object::Object::new(crate::object::ObjectInit {
            id: unnamed_id.clone(),
            object_class: HashSet::from([ObjectClass::Valid(7)]),
            area: 10,
            bbox: [0, 0, 3, 3],
            ..Default::default()
        });

        let unset_id = ObjectId::next();
        let unset = crate::object::Object::new(crate::object::ObjectInit {
            id: unset_id.clone(),
            object_class: HashSet::from([ObjectClass::Unset]),
            area: 10,
            bbox: [0, 0, 3, 3],
            ..Default::default()
        });

        cache.object_cache.insert(named.id.clone(), named);
        cache.object_cache.insert(unnamed.id.clone(), unnamed);
        cache.object_cache.insert(unset.id.clone(), unset);

        exporter.export(&cache).expect("export should succeed");

        let content = fs::read_to_string(&output_path).expect("Failed to read CSV");
        let (headers, rows) = parse_csv(&content);

        assert_eq!(
            column(
                &headers,
                row_by_object_id(&headers, &rows, &named_id.to_string()),
                "object_class"
            ),
            "Positive (5)",
            "a class present in class_names should render as '<name> (<n>)'"
        );
        assert_eq!(
            column(
                &headers,
                row_by_object_id(&headers, &rows, &unnamed_id.to_string()),
                "object_class"
            ),
            "class_7",
            "a class absent from class_names should fall back to 'class_<n>'"
        );
        assert_eq!(
            column(
                &headers,
                row_by_object_id(&headers, &rows, &unset_id.to_string()),
                "object_class"
            ),
            "unset",
            "ObjectClass::Unset should render as 'unset'"
        );
    }

    #[test]
    fn test_colocalization_columns_are_named_and_comma_joined() {
        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let output_path = temp_dir.path().join("test_coloc.csv");

        let mut class_names = HashMap::new();
        class_names.insert(ObjectClass::Valid(9), "Nuclei".to_string());

        let exporter =
            CsvExporter::new(output_path.clone(), class_names).expect("exporter init failed");

        let mut cache = GlobalPipelineCache::default();
        cache.image_rel_path = PathBuf::from("coloc.tif");

        let partner_a = ObjectId::next();
        let partner_b = ObjectId::next();

        let colocalized_id = ObjectId::next();
        let colocalized = crate::object::Object::new(crate::object::ObjectInit {
            id: colocalized_id.clone(),
            area: 10,
            bbox: [0, 0, 3, 3],
            colocalized_with: IndexMap::from([(
                ObjectClass::Valid(9),
                vec![partner_a.clone(), partner_b.clone()],
            )]),
            ..Default::default()
        });

        let solo_id = ObjectId::next();
        let solo = crate::object::Object::new(crate::object::ObjectInit {
            id: solo_id.clone(),
            area: 10,
            bbox: [0, 0, 3, 3],
            ..Default::default()
        });

        cache
            .object_cache
            .insert(colocalized.id.clone(), colocalized);
        cache.object_cache.insert(solo.id.clone(), solo);

        exporter.export(&cache).expect("export should succeed");

        let content = fs::read_to_string(&output_path).expect("Failed to read CSV");
        let (headers, rows) = parse_csv(&content);

        let coloc_col = "coloc_with_Nuclei (9)";
        assert!(
            headers.iter().any(|h| h == coloc_col),
            "expected a '{coloc_col}' column; headers were: {:?}",
            headers
        );

        let expected_ids = format!("{},{}", partner_a, partner_b);
        assert_eq!(
            column(
                &headers,
                row_by_object_id(&headers, &rows, &colocalized_id.to_string()),
                coloc_col
            ),
            expected_ids,
            "multiple colocalization partners should be comma-joined"
        );
        assert_eq!(
            column(
                &headers,
                row_by_object_id(&headers, &rows, &solo_id.to_string()),
                coloc_col
            ),
            "",
            "an object with no colocalization for that class should have an empty cell"
        );
    }

    #[test]
    fn test_multi_channel_intensity_columns_are_computed_correctly() {
        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let output_path = temp_dir.path().join("test_intensities.csv");

        let exporter =
            CsvExporter::new(output_path.clone(), HashMap::new()).expect("exporter init failed");

        let mut cache = GlobalPipelineCache::default();
        cache.image_rel_path = PathBuf::from("intensities.tif");
        // PipelineCache::default() sets nr_of_bits = 16 (see PipelineImageMeta's test Default impl).
        let bit_max = ((1u64 << 16) - 1) as f64;

        let area: usize = 5;
        let ch0 = Intensity {
            sum_intensity: 2.5,
            min_intensity: 0.1,
            max_intensity: 0.9,
            avg_intensity: 0.5,
            pixel_values: vec![],
        };
        let ch3 = Intensity {
            sum_intensity: 1.2,
            min_intensity: 0.05,
            max_intensity: 0.4,
            avg_intensity: 0.24,
            pixel_values: vec![],
        };

        let object_id = ObjectId::next();
        let object = crate::object::Object::new(crate::object::ObjectInit {
            id: object_id.clone(),
            area,
            bbox: [0, 0, 3, 3],
            intensities: IndexMap::from([(0, ch0.clone()), (3, ch3.clone())]),
            ..Default::default()
        });
        cache.object_cache.insert(object.id.clone(), object);

        exporter.export(&cache).expect("export should succeed");

        let content = fs::read_to_string(&output_path).expect("Failed to read CSV");
        let (headers, rows) = parse_csv(&content);
        let row = row_by_object_id(&headers, &rows, &object_id.to_string());

        for (ch, intensity) in [(0, &ch0), (3, &ch3)] {
            let mean_raw = intensity.sum_intensity / (area as f64).max(1.0);
            let min_raw = intensity.min_intensity as f64;
            let max_raw = intensity.max_intensity as f64;

            assert_eq!(
                column(&headers, row, &format!("ch{ch}_integrated_density_raw")),
                format!("{:.6}", intensity.sum_intensity)
            );
            assert_eq!(
                column(&headers, row, &format!("ch{ch}_integrated_density_scaled")),
                format!("{:.2}", intensity.sum_intensity * bit_max)
            );
            assert_eq!(
                column(&headers, row, &format!("ch{ch}_mean_intensity_raw")),
                format!("{:.6}", mean_raw)
            );
            assert_eq!(
                column(&headers, row, &format!("ch{ch}_mean_intensity_scaled")),
                format!("{:.2}", mean_raw * bit_max)
            );
            assert_eq!(
                column(&headers, row, &format!("ch{ch}_min_intensity_raw")),
                format!("{:.6}", min_raw)
            );
            assert_eq!(
                column(&headers, row, &format!("ch{ch}_min_intensity_scaled")),
                format!("{:.2}", min_raw * bit_max)
            );
            assert_eq!(
                column(&headers, row, &format!("ch{ch}_max_intensity_raw")),
                format!("{:.6}", max_raw)
            );
            assert_eq!(
                column(&headers, row, &format!("ch{ch}_max_intensity_scaled")),
                format!("{:.2}", max_raw * bit_max)
            );
        }
    }

    #[test]
    fn test_new_against_preexisting_file_does_not_rewrite_header() {
        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let output_path = temp_dir.path().join("test_preexisting.csv");

        // First, independent exporter instance creates the file and writes header + one row.
        let exporter_1 =
            CsvExporter::new(output_path.clone(), HashMap::new()).expect("exporter init failed");
        let mut cache_a = GlobalPipelineCache::default();
        cache_a.image_rel_path = PathBuf::from("image_a.tif");
        let object_a = crate::object::Object::new(crate::object::ObjectInit {
            id: ObjectId::next(),
            area: 10,
            bbox: [0, 0, 3, 3],
            ..Default::default()
        });
        cache_a.object_cache.insert(object_a.id.clone(), object_a);
        exporter_1
            .export(&cache_a)
            .expect("first export should succeed");
        drop(exporter_1);

        // A brand-new CsvExporter instance pointed at the now-existing file should treat the
        // header as already written (header_written = output_path.exists()) and only append.
        let exporter_2 =
            CsvExporter::new(output_path.clone(), HashMap::new()).expect("exporter init failed");
        let mut cache_b = GlobalPipelineCache::default();
        cache_b.image_rel_path = PathBuf::from("image_b.tif");
        let object_b = crate::object::Object::new(crate::object::ObjectInit {
            id: ObjectId::next(),
            area: 20,
            bbox: [1, 1, 4, 4],
            ..Default::default()
        });
        cache_b.object_cache.insert(object_b.id.clone(), object_b);
        exporter_2
            .export(&cache_b)
            .expect("second export (fresh instance) should succeed");

        let content = fs::read_to_string(&output_path).expect("Failed to read CSV");
        let lines: Vec<&str> = content.lines().collect();
        assert_eq!(
            lines.len(),
            3,
            "expected one header row plus one data row per export() call, got: {content}"
        );
        assert_eq!(
            lines
                .iter()
                .filter(|l| l.starts_with("image,channel"))
                .count(),
            1,
            "a second CsvExporter instance over a pre-existing file must not rewrite the header"
        );
    }

    #[test]
    fn test_lineage_touches_edge_and_track_id_fields() {
        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let output_path = temp_dir.path().join("test_lineage.csv");

        let exporter =
            CsvExporter::new(output_path.clone(), HashMap::new()).expect("exporter init failed");

        let mut cache = GlobalPipelineCache::default();
        cache.image_rel_path = PathBuf::from("lineage.tif");

        let parent_id = ObjectId::next();
        let child_1 = ObjectId::next();
        let child_2 = ObjectId::next();

        let object_id = ObjectId::next();
        let object = crate::object::Object::new(crate::object::ObjectInit {
            id: object_id.clone(),
            area: 10,
            bbox: [0, 0, 3, 3],
            parent_id: Some(parent_id.clone()),
            children: vec![child_1.clone(), child_2.clone()],
            touches_edge: true,
            track: Track {
                id: TrackId(42),
                object_ids: vec![],
                parent_track: None,
            },
            ..Default::default()
        });
        cache.object_cache.insert(object.id.clone(), object);

        exporter.export(&cache).expect("export should succeed");

        let content = fs::read_to_string(&output_path).expect("Failed to read CSV");
        let (headers, rows) = parse_csv(&content);
        let row = row_by_object_id(&headers, &rows, &object_id.to_string());

        assert_eq!(column(&headers, row, "parent_id"), parent_id.to_string());
        assert_eq!(
            column(&headers, row, "children"),
            format!("{},{}", child_1, child_2)
        );
        assert_eq!(column(&headers, row, "touches_edge"), "true");
        assert_eq!(column(&headers, row, "track_id"), "42");
    }
}
