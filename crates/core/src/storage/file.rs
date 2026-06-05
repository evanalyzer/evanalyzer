use crate::pipeline::pipeline_cache::PipelineCache;
use crate::storage::PipelineResultExporter;
use evanalyzer_cfg::core_types::{InternalErrors, ObjectClass};
use std::collections::HashMap;
use std::fs::OpenOptions;
use std::path::PathBuf;

pub struct CsvExporter {
    pub output_path: PathBuf,
    /// Maps ObjectClass → human-readable name from project classification settings.
    pub class_names: HashMap<ObjectClass, String>,
}

impl CsvExporter {
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
    fn export(&self, cache: &PipelineCache) -> Result<(), InternalErrors> {
        let file_exists = self.output_path.exists();

        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.output_path)
            .map_err(|e| InternalErrors::Io(format!("IO Error: {}", e)))?;

        let mut writer = csv::Writer::from_writer(file);

        let px = &cache.image_cache.image_meta.pixel_sizes;

        // --- Phase 1: Dynamic Channel Extraction ---
        let mut channel_ids: Vec<i32> = cache
            .roi_cache
            .values()
            .flat_map(|roi| roi.intensities.keys().cloned())
            .collect();
        channel_ids.sort();
        channel_ids.dedup();

        // --- Phase 2: Dynamic Colocalization Class Extraction ---
        let mut coloc_classes: Vec<ObjectClass> = cache
            .roi_cache
            .values()
            .flat_map(|roi| roi.colocalized_with.keys().cloned())
            .collect();
        coloc_classes.sort_by_key(|c| format!("{:?}", c));
        coloc_classes.dedup();

        // --- Phase 3: Header Assembly ---
        if !file_exists {
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
                header.push(format!("ch{}_median_intensity_raw", ch));
                header.push(format!("ch{}_median_intensity_scaled", ch));
                header.push(format!("ch{}_std_dev_raw", ch));
                header.push(format!("ch{}_std_dev_scaled", ch));
                header.push(format!("ch{}_min_intensity_raw", ch));
                header.push(format!("ch{}_min_intensity_scaled", ch));
                header.push(format!("ch{}_max_intensity_raw", ch));
                header.push(format!("ch{}_max_intensity_scaled", ch));
            }

            for class in &coloc_classes {
                header.push(self.coloc_col_name(class));
            }

            writer
                .write_record(&header)
                .map_err(|e| InternalErrors::Io(e.to_string()))?;
        }

        // --- Phase 4: Data Row Serialization ---
        let px_len = (px.px_size_x * px.px_size_y).sqrt();
        let nr_of_bits = cache.image_cache.image_meta.nr_of_bits;
        // Max pixel value for the bit depth (e.g. 65535 for 16-bit)
        let bit_max = ((1u64 << nr_of_bits) - 1) as f64;

        for roi in cache.roi_cache.values() {
            let perimeter = roi.get_perimeter();
            let ellipse = roi.get_ellipse();
            let solidity = roi.get_solidity();
            let centroid = roi.get_centroid();
            let feret = roi.get_feret_diameter();
            let min_feret = roi.get_min_feret_diameter();

            let parent_id = roi
                .parent_id
                .as_ref()
                .map(|id| id.to_string())
                .unwrap_or_default();

            let children = roi
                .children
                .iter()
                .map(|id| id.to_string())
                .collect::<Vec<_>>()
                .join(",");

            let mut row = vec![
                format!("{:?}", cache.image_rel_path),
                format!("{}", roi.plane.c),
                format!("{}", roi.plane.z),
                format!("{}", roi.plane.t),
                // Identity & Lineage
                roi.id.to_string(),
                roi.segmentation_class.to_string(),
                roi.object_class.iter().map(|c| self.class_label(c)).collect::<Vec<_>>().join(","),
                parent_id,
                children,
                roi.track.id.0.to_string(),
                // Centroid px & nm
                format!("{:.2}", centroid.0),
                format!("{:.2}", centroid.1),
                format!("{:.2}", centroid.0 as f64 * px.px_size_x as f64),
                format!("{:.2}", centroid.1 as f64 * px.px_size_y as f64),
                // Bounding box px & nm
                roi.bbox[0].to_string(),
                roi.bbox[1].to_string(),
                roi.bbox[2].to_string(),
                roi.bbox[3].to_string(),
                format!("{:.2}", roi.bbox[0] as f32 * px.px_size_x),
                format!("{:.2}", roi.bbox[1] as f32 * px.px_size_y),
                format!("{:.2}", roi.bbox[2] as f32 * px.px_size_x),
                format!("{:.2}", roi.bbox[3] as f32 * px.px_size_y),
                // Area px & nm²
                roi.area.to_string(),
                format!("{:.2}", roi.area as f32 * px.px_size_x * px.px_size_y),
                // Perimeter px & nm
                format!("{:.2}", perimeter),
                format!("{:.2}", perimeter * px_len),
                // Shape descriptors
                format!("{:.4}", roi.circularity()),
                format!("{:.4}", solidity),
                format!("{:.4}", roi.get_aspect_ratio()),
                format!("{:.4}", roi.get_roundness(perimeter)),
                format!("{:.4}", roi.get_compactness(perimeter)),
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
                roi.touches_edge.to_string(),
                // Pixel sizes
                format!("{:.4}", px.px_size_x),
                format!("{:.4}", px.px_size_y),
                format!("{:.4}", px.px_size_z),
                nr_of_bits.to_string(),
            ];

            for ch in &channel_ids {
                if let Some(intensity) = roi.intensities.get(ch) {
                    let mean_raw = intensity.sum_intensity / (roi.area as f64).max(1.0);
                    let median_raw = intensity.median_intensity.unwrap_or(0.0) as f64;
                    let std_raw = intensity.std_dev.unwrap_or(0.0) as f64;
                    let min_raw = intensity.min_intensity as f64;
                    let max_raw = intensity.max_intensity as f64;

                    row.push(format!("{:.6}", intensity.sum_intensity));
                    row.push(format!("{:.2}", intensity.sum_intensity * bit_max));
                    row.push(format!("{:.6}", mean_raw));
                    row.push(format!("{:.2}", mean_raw * bit_max));
                    row.push(format!("{:.6}", median_raw));
                    row.push(format!("{:.2}", median_raw * bit_max));
                    row.push(format!("{:.6}", std_raw));
                    row.push(format!("{:.2}", std_raw * bit_max));
                    row.push(format!("{:.6}", min_raw));
                    row.push(format!("{:.2}", min_raw * bit_max));
                    row.push(format!("{:.6}", max_raw));
                    row.push(format!("{:.2}", max_raw * bit_max));
                } else {
                    // 12 empty cells (6 metrics × 2 scales)
                    row.extend(std::iter::repeat_n("".to_string(), 12));
                }
            }

            for class in &coloc_classes {
                if let Some(ids) = roi.colocalized_with.get(class) {
                    let id_strings: Vec<String> = ids.iter().map(|id| id.to_string()).collect();
                    row.push(id_strings.join(","));
                } else {
                    row.push("".to_string());
                }
            }

            writer
                .write_record(&row)
                .map_err(|e| InternalErrors::ImageReadError(e.to_string()))?;
        }

        writer
            .flush()
            .map_err(|e| InternalErrors::ImageReadError(e.to_string()))?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;

    #[test]
    fn test_csv_export_creates_file() {
        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let output_path = temp_dir.path().join("test_export.csv");

        let exporter = CsvExporter {
            output_path: output_path.clone(),
            class_names: HashMap::new(),
        };

        let mut cache = PipelineCache::default();
        cache.image_rel_path = PathBuf::from("test_image.tif");

        let result = exporter.export(&cache);
        assert!(result.is_ok(), "Export should succeed");
        assert!(output_path.exists(), "CSV file should be created");
    }

    #[test]
    fn test_csv_export_includes_morphological_metrics() {
        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let output_path = temp_dir.path().join("test_morphology.csv");

        let exporter = CsvExporter {
            output_path: output_path.clone(),
            class_names: HashMap::new(),
        };

        let cache = PipelineCache::default();
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
}
