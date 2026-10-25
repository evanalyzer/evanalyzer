//! The AI training panel's feature tables, apart from Slint: the pixel
//! classifier's features (what a feature sees × its size) and the object
//! classifier's (shape metrics, brightness statistics × channels), their
//! presets, and the conversion to and from what a model stores.

use evanalyzer_cfg::settings::ai_learning_object_settings::ObjectMetric;
use evanalyzer_cfg::settings::ai_learning_pixel_settings::PreprocessingSteps;
use evanalyzer_cfg::settings::pipeline_command_settings::{
    EdgeDetectionSobelSettings, FiltersHessianHessianModeSettings,
    FiltersRankFilterRankFilterTypeSettings, FiltersStructureTensorTensorModeSettings,
    GaussianBlurSettings, HessianSettings, LaplacianSettings, RankFilterSettings,
    StructureTensorSettings,
};

// -- Pixel features ----------------------------------------------------------

/// What a pixel feature sees: (name, what it means, has an "as is" column,
/// its options - the first is the default).
pub(crate) const PIXEL_FEATURES: [(&str, &str, bool, &[&str]); 7] = [
    (
        "Raw image (as it is)",
        "the pixel's own intensity, unfiltered",
        true,
        &[],
    ),
    (
        "Gaussian blur",
        "the intensity smoothed over the size",
        false,
        &[],
    ),
    (
        "Gaussian gradient (Sobel)",
        "edges: where the intensity changes",
        false,
        &[],
    ),
    (
        "Laplacian of Gaussian",
        "spots and blobs: round bright or dark specks",
        false,
        &[],
    ),
    (
        "Hessian of Gaussian",
        "lines and ridges: fibres, membranes, cell borders",
        false,
        &["determinant", "eigenvalue 1", "eigenvalue 2"],
    ),
    (
        "Structure tensor",
        "texture: how aligned the pattern is",
        false,
        &["coherence", "eigenvalue 1", "eigenvalue 2"],
    ),
    (
        "Rank filter",
        "median / mean / min / max around the pixel",
        false,
        &["median", "mean", "min", "max"],
    ),
];

/// The raw image: its one checkbox, no sizes.
const RAW: usize = 0;
const BRIGHTNESS: usize = 1;
const EDGES: usize = 2;
const SPOTS: usize = 3;
const RIDGES: usize = 4;
const TEXTURE: usize = 5;
const NEIGHBOURHOOD: usize = 6;

/// The size columns a new table starts with, in pixels.
pub(crate) const DEFAULT_SIZES: [f32; 5] = [1.0, 2.0, 4.0, 8.0, 16.0];

/// The largest size a column may have (Gaussian sigma 8, kernel 49).
pub(crate) const MAX_SIZE: f32 = 16.0;

/// At most this many size columns (they must fit the panel).
pub(crate) const MAX_SIZES: usize = 10;

/// One row of the pixel feature table.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct PixelFeatureRow {
    /// The "as is" column (Brightness only): the raw pixel value.
    pub raw: bool,
    /// One per size column.
    pub cells: Vec<bool>,
    /// Index into the row's options.
    pub option: usize,
}

/// The pixel feature table: rows in [`PIXEL_FEATURES`] order × `sizes`.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct PixelFeatureTable {
    pub sizes: Vec<f32>,
    pub rows: Vec<PixelFeatureRow>,
}

/// The presets offered above the table.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Preset {
    Simple,
    Recommended,
    All,
}

impl PixelFeatureTable {
    /// Nothing ticked, the default sizes.
    pub fn empty() -> Self {
        Self::empty_with(DEFAULT_SIZES.to_vec())
    }

    fn empty_with(sizes: Vec<f32>) -> Self {
        Self {
            rows: PIXEL_FEATURES
                .iter()
                .enumerate()
                .map(|(kind, _)| PixelFeatureRow {
                    raw: false,
                    cells: vec![false; if kind == RAW { 0 } else { sizes.len() }],
                    option: 0,
                })
                .collect(),
            sizes,
        }
    }

    /// A preset on the default sizes.
    pub fn preset(preset: Preset) -> Self {
        let mut table = Self::empty();
        let mut tick = |row: usize, sizes: &[f32]| {
            for size in sizes {
                if let Some(col) = DEFAULT_SIZES.iter().position(|s| s == size) {
                    table.rows[row].cells[col] = true;
                }
            }
        };
        match preset {
            Preset::Simple => {
                tick(BRIGHTNESS, &[1.0, 4.0]);
                tick(EDGES, &[1.0]);
            }
            Preset::Recommended => {
                tick(BRIGHTNESS, &[1.0, 4.0, 16.0]);
                tick(EDGES, &[1.0, 4.0]);
                tick(SPOTS, &[2.0, 4.0]);
                tick(RIDGES, &[2.0]);
                tick(NEIGHBOURHOOD, &[4.0]);
            }
            Preset::All => {
                for row in RAW + 1..PIXEL_FEATURES.len() {
                    tick(row, &DEFAULT_SIZES);
                }
            }
        }
        table.rows[RAW].raw = true;
        table
    }

    /// Which preset this table is, if any.
    pub fn matches(&self) -> Option<Preset> {
        [Preset::Simple, Preset::Recommended, Preset::All]
            .into_iter()
            .find(|p| Self::preset(*p) == *self)
    }

    /// Adds a size column (sorted in; nothing if it's there already or out
    /// of 0 < size <= [`MAX_SIZE`]); whether it was added.
    pub fn add_size(&mut self, size: f32) -> bool {
        if !(size > 0.0 && size <= MAX_SIZE)
            || self.sizes.len() >= MAX_SIZES
            || self.sizes.iter().any(|s| *s == size)
        {
            return false;
        }
        let at = self.sizes.partition_point(|s| *s < size);
        self.sizes.insert(at, size);
        for row in self.rows.iter_mut().skip(RAW + 1) {
            row.cells.insert(at, false);
        }
        true
    }

    /// Removes size column `col` (not the last one left); whether it did.
    pub fn remove_size(&mut self, col: usize) -> bool {
        if col >= self.sizes.len() || self.sizes.len() == 1 {
            return false;
        }
        self.sizes.remove(col);
        for row in self.rows.iter_mut().skip(RAW + 1) {
            row.cells.remove(col);
        }
        true
    }

    /// Ticked features per image channel.
    pub fn count(&self) -> usize {
        self.rows
            .iter()
            .map(|row| usize::from(row.raw) + row.cells.iter().filter(|c| **c).count())
            .sum()
    }

    /// What a model stores: one filter chain per ticked cell, row by row,
    /// left to right.
    pub fn to_recipe(&self) -> Vec<Vec<PreprocessingSteps>> {
        let mut recipe = Vec::new();
        for (kind, row) in self.rows.iter().enumerate() {
            if row.raw && PIXEL_FEATURES[kind].2 {
                recipe.push(Vec::new());
            }
            for (size, ticked) in self.sizes.iter().zip(&row.cells) {
                if *ticked {
                    recipe.push(chain(kind, *size, row.option));
                }
            }
        }
        recipe
    }

    /// The table a model's recipe was made from - the sizes it uses next to
    /// the default ones. `None` if a chain isn't one the table makes (a
    /// model from an older dialog).
    pub fn from_recipe(recipe: &[Vec<PreprocessingSteps>]) -> Option<Self> {
        let parsed: Vec<(usize, Option<f32>, usize)> = recipe
            .iter()
            .map(|steps| parse_chain(steps))
            .collect::<Option<_>>()?;
        let mut sizes = DEFAULT_SIZES.to_vec();
        for (_, size, _) in &parsed {
            if let Some(size) = size
                && !sizes.iter().any(|s| (s - size).abs() < 1e-3)
            {
                sizes.push(*size);
            }
        }
        sizes.sort_by(f32::total_cmp);
        let mut table = Self::empty_with(sizes);
        for (kind, size, option) in parsed {
            let row = &mut table.rows[kind];
            row.option = option;
            match size {
                None => row.raw = true,
                Some(size) => {
                    let col = table.sizes.iter().position(|s| (s - size).abs() < 1e-3)?;
                    row.cells[col] = true;
                }
            }
        }
        Some(table)
    }
}

/// Size `size` px as a Gaussian: sigma = size / 2, a kernel covering
/// ±3 sigma (odd, 3..=49).
fn gaussian(size: f32) -> GaussianBlurSettings {
    let sigma = size / 2.0;
    let half = (3.0 * sigma).ceil() as usize;
    GaussianBlurSettings {
        kernel_size: (2 * half + 1).clamp(3, 49),
        sigma,
    }
}

fn chain(kind: usize, size: f32, option: usize) -> Vec<PreprocessingSteps> {
    let blur = PreprocessingSteps::GaussianBlur(gaussian(size));
    match kind {
        BRIGHTNESS => vec![blur],
        EDGES => vec![
            blur,
            PreprocessingSteps::EdgeDetectionSobel(EdgeDetectionSobelSettings { kernel_size: 3 }),
        ],
        SPOTS => vec![
            blur,
            PreprocessingSteps::Laplacian(LaplacianSettings { kernel_size: 3 }),
        ],
        RIDGES => vec![
            blur,
            PreprocessingSteps::Hessian(HessianSettings {
                mode: match option {
                    1 => FiltersHessianHessianModeSettings::EigenvaluesX,
                    2 => FiltersHessianHessianModeSettings::EigenvaluesY,
                    _ => FiltersHessianHessianModeSettings::Determinant,
                },
            }),
        ],
        TEXTURE => {
            let g = gaussian(size);
            vec![PreprocessingSteps::StructureTensor(
                StructureTensorSettings {
                    mode: match option {
                        1 => FiltersStructureTensorTensorModeSettings::EigenvaluesX,
                        2 => FiltersStructureTensorTensorModeSettings::EigenvaluesY,
                        _ => FiltersStructureTensorTensorModeSettings::Coherence,
                    },
                    kernel_size: g.kernel_size,
                    sigma: g.sigma,
                },
            )]
        }
        _ => vec![PreprocessingSteps::RankFilter(RankFilterSettings {
            radius: size as f64,
            filter_type: match option {
                1 => FiltersRankFilterRankFilterTypeSettings::Mean,
                2 => FiltersRankFilterRankFilterTypeSettings::Min,
                3 => FiltersRankFilterRankFilterTypeSettings::Max,
                _ => FiltersRankFilterRankFilterTypeSettings::Median,
            },
        })],
    }
}

/// (row, size - `None`: as is, option) of a chain [`chain`] makes.
fn parse_chain(steps: &[PreprocessingSteps]) -> Option<(usize, Option<f32>, usize)> {
    use PreprocessingSteps as S;
    let size = |g: &GaussianBlurSettings| g.sigma * 2.0;
    Some(match steps {
        [] => (RAW, None, 0),
        [S::GaussianBlur(g)] => (BRIGHTNESS, Some(size(g)), 0),
        [S::GaussianBlur(g), S::EdgeDetectionSobel(_)] => (EDGES, Some(size(g)), 0),
        [S::GaussianBlur(g), S::Laplacian(_)] => (SPOTS, Some(size(g)), 0),
        [S::GaussianBlur(g), S::Hessian(h)] => (
            RIDGES,
            Some(size(g)),
            match h.mode {
                FiltersHessianHessianModeSettings::Determinant => 0,
                FiltersHessianHessianModeSettings::EigenvaluesX => 1,
                FiltersHessianHessianModeSettings::EigenvaluesY => 2,
            },
        ),
        [S::StructureTensor(t)] => (
            TEXTURE,
            Some(t.sigma * 2.0),
            match t.mode {
                FiltersStructureTensorTensorModeSettings::Coherence => 0,
                FiltersStructureTensorTensorModeSettings::EigenvaluesX => 1,
                FiltersStructureTensorTensorModeSettings::EigenvaluesY => 2,
            },
        ),
        [S::RankFilter(r)] => (
            NEIGHBOURHOOD,
            Some(r.radius as f32),
            match r.filter_type {
                FiltersRankFilterRankFilterTypeSettings::Median => 0,
                FiltersRankFilterRankFilterTypeSettings::Mean => 1,
                FiltersRankFilterRankFilterTypeSettings::Min => 2,
                FiltersRankFilterRankFilterTypeSettings::Max => 3,
                FiltersRankFilterRankFilterTypeSettings::Outliers(_) => return None,
            },
        ),
        _ => return None,
    })
}

/// A size as its column header ("1", "2.5").
pub(crate) fn size_label(size: f32) -> String {
    if size.fract() == 0.0 {
        format!("{size:.0}")
    } else {
        format!("{size}")
    }
}

// -- Object features ---------------------------------------------------------

/// The shape metrics, in the order the panel shows them.
pub(crate) const SHAPE_METRICS: [(&str, ObjectMetric); 14] = [
    ("Area", ObjectMetric::Area),
    ("Perimeter", ObjectMetric::Perimeter),
    ("Circularity", ObjectMetric::Circularity),
    ("Solidity", ObjectMetric::Solidity),
    ("Aspect ratio", ObjectMetric::AspectRatio),
    ("Roundness", ObjectMetric::Roundness),
    ("Max Feret", ObjectMetric::FeretDiameter),
    ("Min Feret", ObjectMetric::MinFeretDiameter),
    ("Eccentricity", ObjectMetric::Eccentricity),
    ("Major axis", ObjectMetric::EllipseMajor),
    ("Minor axis", ObjectMetric::EllipseMinor),
    ("Compactness", ObjectMetric::Compactness),
    ("Orientation", ObjectMetric::EllipseAngle),
    ("Touches edge", ObjectMetric::TouchesEdge),
];

/// The shape metrics the "Recommended" and "Shape only" presets tick.
const RECOMMENDED_SHAPE: [usize; 7] = [0, 1, 2, 3, 4, 6, 8];

/// A row of the brightness table: (name, hint, the metrics of a channel it
/// stands for).
pub(crate) const BRIGHTNESS_STATS: [(&str, &str, fn(i32) -> Vec<ObjectMetric>); 9] = [
    ("Mean", "", |c| vec![ObjectMetric::IntensityAvg(c)]),
    ("Median", "", |c| vec![ObjectMetric::IntensityMedian(c)]),
    ("Std dev", "how uneven", |c| {
        vec![ObjectMetric::IntensityStd(c)]
    }),
    ("Min / Max", "", |c| {
        vec![ObjectMetric::IntensityMin(c), ObjectMetric::IntensityMax(c)]
    }),
    ("Quartiles", "", |c| {
        vec![
            ObjectMetric::IntensityLowerQuartile(c),
            ObjectMetric::IntensityUpperQuartile(c),
        ]
    }),
    ("MAD", "uneven, robust to specks", |c| {
        vec![ObjectMetric::IntensityMad(c)]
    }),
    ("Sum", "grows with size", |c| {
        vec![ObjectMetric::IntensitySum(c)]
    }),
    ("Ring mean", "the surroundings", |c| {
        vec![ObjectMetric::RingIntensityAvg(c)]
    }),
    ("Contrast to ring", "stands out", |c| {
        vec![ObjectMetric::RingContrast(c)]
    }),
];

/// The rows "Recommended" ticks for every channel: mean, median, std dev.
const RECOMMENDED_STATS: [usize; 3] = [0, 1, 2];

/// The object feature tables: shape metrics, brightness statistic × channel.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct ObjectFeatureTable {
    pub shape: Vec<bool>,
    /// `[statistic][channel]`.
    pub brightness: Vec<Vec<bool>>,
}

impl ObjectFeatureTable {
    pub fn preset(preset: Preset, channels: usize) -> Self {
        let mut table = Self {
            shape: vec![false; SHAPE_METRICS.len()],
            brightness: vec![vec![false; channels]; BRIGHTNESS_STATS.len()],
        };
        match preset {
            Preset::Simple => {
                for i in RECOMMENDED_SHAPE {
                    table.shape[i] = true;
                }
            }
            Preset::Recommended => {
                for i in RECOMMENDED_SHAPE {
                    table.shape[i] = true;
                }
                for row in RECOMMENDED_STATS {
                    table.brightness[row].fill(true);
                }
            }
            Preset::All => {
                table.shape.fill(true);
                for row in &mut table.brightness {
                    row.fill(true);
                }
            }
        }
        table
    }

    pub fn matches(&self, channels: usize) -> Option<Preset> {
        [Preset::Simple, Preset::Recommended, Preset::All]
            .into_iter()
            .find(|p| Self::preset(*p, channels) == *self)
    }

    /// The same ticks for `channels` channels (new ones as the first
    /// channel, if any, else unticked).
    pub fn with_channels(&self, channels: usize) -> Self {
        let mut table = self.clone();
        for row in &mut table.brightness {
            let fill = row.first().copied().unwrap_or(false);
            row.resize(channels, fill);
        }
        table
    }

    pub fn to_metrics(&self) -> Vec<ObjectMetric> {
        let mut metrics: Vec<ObjectMetric> = SHAPE_METRICS
            .iter()
            .zip(&self.shape)
            .filter(|(_, on)| **on)
            .map(|((_, m), _)| m.clone())
            .collect();
        for ((_, _, of), row) in BRIGHTNESS_STATS.iter().zip(&self.brightness) {
            for (channel, on) in row.iter().enumerate() {
                if *on {
                    metrics.extend(of(channel as i32));
                }
            }
        }
        metrics
    }

    /// A model's metrics as ticks (a row counts as ticked when all its
    /// metrics are there).
    pub fn from_metrics(metrics: &[ObjectMetric], channels: usize) -> Self {
        Self {
            shape: SHAPE_METRICS
                .iter()
                .map(|(_, m)| metrics.contains(m))
                .collect(),
            brightness: BRIGHTNESS_STATS
                .iter()
                .map(|(_, _, of)| {
                    (0..channels)
                        .map(|c| of(c as i32).iter().all(|m| metrics.contains(m)))
                        .collect()
                })
                .collect(),
        }
    }

    pub fn count(&self) -> usize {
        self.to_metrics().len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_recommended_preset_has_ten_features_and_is_recognised() {
        let table = PixelFeatureTable::preset(Preset::Recommended);
        assert_eq!(table.count(), 10);
        assert_eq!(table.to_recipe().len(), 10);
        assert_eq!(table.matches(), Some(Preset::Recommended));
        let mut changed = table.clone();
        changed.rows[EDGES].cells[0] = false;
        assert_eq!(changed.matches(), None, "edited: custom");
    }

    #[test]
    fn a_size_is_a_gaussian_of_half_its_sigma_with_a_fitting_kernel() {
        assert_eq!(gaussian(1.0).sigma, 0.5);
        assert_eq!(gaussian(1.0).kernel_size, 5);
        assert_eq!(gaussian(4.0).kernel_size, 13);
        assert_eq!(gaussian(16.0).sigma, 8.0);
        assert_eq!(gaussian(16.0).kernel_size, 49);
    }

    #[test]
    fn every_ticked_cell_becomes_its_chain_and_back() {
        let mut table = PixelFeatureTable::preset(Preset::All);
        table.add_size(3.0);
        for row in &mut table.rows {
            row.cells.fill(true);
        }
        table.rows[RIDGES].option = 2;
        table.rows[TEXTURE].option = 1;
        table.rows[NEIGHBOURHOOD].option = 3;
        let recipe = table.to_recipe();
        assert_eq!(recipe.len(), 1 + 6 * 6);
        let back = PixelFeatureTable::from_recipe(&recipe).unwrap();
        assert_eq!(back, table);
        assert_eq!(back.to_recipe().len(), recipe.len());
    }

    #[test]
    fn a_chain_the_table_does_not_make_is_not_read() {
        let foreign = vec![vec![PreprocessingSteps::EdgeDetectionSobel(
            EdgeDetectionSobelSettings { kernel_size: 5 },
        )]];
        assert!(PixelFeatureTable::from_recipe(&foreign).is_none());
    }

    #[test]
    fn sizes_are_added_sorted_once_and_within_bounds() {
        let mut table = PixelFeatureTable::empty();
        assert!(table.add_size(3.0));
        assert!(!table.add_size(3.0));
        assert!(!table.add_size(0.0));
        assert!(!table.add_size(32.0));
        assert_eq!(table.sizes, [1.0, 2.0, 3.0, 4.0, 8.0, 16.0]);
        assert!(table.rows[RAW + 1..].iter().all(|r| r.cells.len() == 6));
        assert!(
            table.rows[RAW].cells.is_empty(),
            "the raw image has no sizes"
        );
        for size in [5.0, 6.0, 7.0, 9.0, 10.0] {
            table.add_size(size);
        }
        assert_eq!(table.sizes.len(), MAX_SIZES, "at most {MAX_SIZES}");
        table.rows[BRIGHTNESS].cells[3] = true; // 4 px
        assert!(table.remove_size(2)); // 3 px
        assert_eq!(table.sizes[2], 4.0);
        assert!(
            table.rows[BRIGHTNESS].cells[2],
            "the other columns keep their ticks"
        );
        let mut one = PixelFeatureTable::empty_with(vec![2.0]);
        assert!(!one.remove_size(0), "the last size stays");
        assert_eq!(size_label(3.0), "3");
        assert_eq!(size_label(2.5), "2.5");
    }

    #[test]
    fn object_features_round_trip_and_follow_the_channels() {
        let table = ObjectFeatureTable::preset(Preset::Recommended, 2);
        let metrics = table.to_metrics();
        assert_eq!(metrics.len(), 7 + 3 * 2);
        assert!(metrics.contains(&ObjectMetric::IntensityMedian(1)));
        assert_eq!(ObjectFeatureTable::from_metrics(&metrics, 2), table);
        assert_eq!(table.matches(2), Some(Preset::Recommended));

        let mut quartiles = ObjectFeatureTable::preset(Preset::Simple, 2);
        quartiles.brightness[4][1] = true;
        let m = quartiles.to_metrics();
        assert!(m.contains(&ObjectMetric::IntensityLowerQuartile(1)));
        assert!(m.contains(&ObjectMetric::IntensityUpperQuartile(1)));
        assert_eq!(ObjectFeatureTable::from_metrics(&m, 2), quartiles);

        let wider = table.with_channels(3);
        assert_eq!(wider.brightness[0], [true, true, true]);
        assert_eq!(wider.brightness[6], [false, false, false]);
    }

    #[test]
    fn the_ring_rows_tick_the_ring_features() {
        let ring = |label| {
            BRIGHTNESS_STATS
                .iter()
                .position(|(l, _, _)| *l == label)
                .unwrap()
        };
        let mut table = ObjectFeatureTable::preset(Preset::Simple, 2);
        table.brightness[ring("Ring mean")][0] = true;
        table.brightness[ring("Contrast to ring")][1] = true;
        let metrics = table.to_metrics();
        assert!(metrics.contains(&ObjectMetric::RingIntensityAvg(0)));
        assert!(metrics.contains(&ObjectMetric::RingContrast(1)));
        assert_eq!(ObjectFeatureTable::from_metrics(&metrics, 2), table);
        assert!(
            !ObjectFeatureTable::preset(Preset::Recommended, 2)
                .to_metrics()
                .iter()
                .any(ObjectMetric::uses_ring),
            "Recommended stays without the ring"
        );
    }
}
