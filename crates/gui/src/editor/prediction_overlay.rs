//! # prediction_overlay
//!
//! The prediction overlay (design canvas "AI Training Dialog", board U6):
//! every pixel in the colour of the class its pixel classifier predicts,
//! the stronger the more probable - pale where the model is unsure, which
//! is where to paint next. Keeps the preview's tiles of it (each
//! `ProgressEvent::PredictionReached`) and draws them into the object
//! layer, under the objects. For an object classifier
//! (`ProgressEvent::ObjectPredictionReached`) each object is as strong as
//! the model is sure of its class instead.
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-10-27
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.

use crate::editor::object_render::blend;
use evanalyzer_app::analysis::{ObjectPredictionCapture, PredictionCapture};
use evanalyzer_cfg::core_types::ObjectId;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::RwLock;

/// The colour of the background class (predicted classes the step maps to
/// no project class).
pub(crate) const BACKGROUND_RGB: [u8; 3] = [0x97, 0x97, 0x8f];

/// The colour of `class` on the overlay and in its legend: its project
/// class's, the background grey (also a class the project no longer has).
pub(crate) fn class_rgb<P: evanalyzer_app::project::ProjectExt + ?Sized>(
    project: &P,
    class: &evanalyzer_app::analysis::PredictedClass,
) -> [u8; 3] {
    if class.segmentation_class == evanalyzer_cfg::core_types::SegmentationClass::BACKGROUND {
        return BACKGROUND_RGB;
    }
    project
        .get_class_from_id(&evanalyzer_cfg::core_types::ObjectClass::Global(
            class.segmentation_class.as_u32(),
        ))
        .map_or(BACKGROUND_RGB, |class| {
            let rgb = class.color;
            [(rgb >> 16) as u8, (rgb >> 8) as u8, rgb as u8]
        })
}

/// The legend: each class of the last preview's prediction (the model's,
/// in its order) with its colour - empty without one.
impl PredictionOverlay {
    pub(crate) fn legend<P: evanalyzer_app::project::ProjectExt + ?Sized>(
        &self,
        project: &P,
    ) -> Vec<(String, [u8; 3])> {
        if let Some(objects) = &*self.objects.read().expect("Poisoned") {
            return objects
                .classes
                .iter()
                .map(|class| {
                    let rgb = project.get_class_from_id(&class.object_class).map_or(
                        BACKGROUND_RGB,
                        |class| {
                            let rgb = class.color;
                            [(rgb >> 16) as u8, (rgb >> 8) as u8, rgb as u8]
                        },
                    );
                    (class.name.clone(), rgb)
                })
                .collect();
        }
        let tiles = self.tiles.read().expect("Poisoned");
        tiles.first().map_or_else(Vec::new, |tile| {
            tile.capture
                .classes
                .iter()
                .map(|class| (class.name.clone(), class_rgb(project, class)))
                .collect()
        })
    }
}

/// What the overlay shows and how.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct PredictionOverlaySettings {
    pub(crate) on: bool,
    /// The model whose pixel classifier step is shown (see
    /// `evanalyzer_app::ai_learning::prediction_target`); `None`: the first
    /// pixel classifier.
    pub(crate) model: Option<PathBuf>,
    /// 0..=1, of a pixel the model is sure of.
    pub(crate) opacity: f32,
    /// The background class in its colour too, else left clear.
    pub(crate) show_background: bool,
    /// The class and its probability under the cursor (the pixel info).
    pub(crate) show_values: bool,
}

impl Default for PredictionOverlaySettings {
    fn default() -> Self {
        Self {
            on: false,
            model: None,
            opacity: 0.7,
            show_background: true,
            show_values: true,
        }
    }
}

/// One preview tile's prediction, at its place in the image.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct PredictionTile {
    pub(crate) tile_offset_x: usize,
    pub(crate) tile_offset_y: usize,
    pub(crate) capture: PredictionCapture,
}

impl PredictionTile {
    /// Index of image pixel `(x, y)` in this tile, if it lies in it.
    fn index_of(&self, x: i64, y: i64) -> Option<usize> {
        let (lx, ly) = (x - self.tile_offset_x as i64, y - self.tile_offset_y as i64);
        let (w, h) = (self.capture.width as i64, self.capture.height as i64);
        (lx >= 0 && ly >= 0 && lx < w && ly < h).then(|| (ly * w + lx) as usize)
    }
}

/// An object classifier's prediction: each scored object's class (index
/// into `classes`) and probability (0..=255).
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct ObjectPrediction {
    pub(crate) classes: Vec<evanalyzer_app::analysis::PredictedObjectClass>,
    pub(crate) objects: HashMap<ObjectId, (u8, u8)>,
}

impl From<ObjectPredictionCapture> for ObjectPrediction {
    fn from(capture: ObjectPredictionCapture) -> Self {
        Self {
            classes: capture.classes,
            objects: capture
                .objects
                .into_iter()
                .map(|(id, class, p)| (id, (class, p)))
                .collect(),
        }
    }
}

/// The overlay's settings and the last preview's prediction - its tiles
/// or its objects - kept until the next preview's, a preview without any
/// or another image.
#[derive(Default)]
pub(crate) struct PredictionOverlay {
    settings: RwLock<PredictionOverlaySettings>,
    tiles: RwLock<Vec<PredictionTile>>,
    objects: RwLock<Option<ObjectPrediction>>,
}

impl PredictionOverlay {
    pub(crate) fn settings(&self) -> PredictionOverlaySettings {
        self.settings.read().expect("Poisoned").clone()
    }

    pub(crate) fn set_settings(&self, settings: PredictionOverlaySettings) {
        if !settings.on {
            self.clear();
        }
        *self.settings.write().expect("Poisoned") = settings;
    }

    /// Stores `tile`; the `first` of a preview replaces the last preview's
    /// tiles, a later one only the tile at its place.
    pub(crate) fn add_tile(&self, tile: PredictionTile, first: bool) {
        let mut tiles = self.tiles.write().expect("Poisoned");
        if first {
            tiles.clear();
            *self.objects.write().expect("Poisoned") = None;
        }
        tiles.retain(|t| {
            (t.tile_offset_x, t.tile_offset_y) != (tile.tile_offset_x, tile.tile_offset_y)
        });
        tiles.push(tile);
    }

    /// Stores an object classifier's prediction in place of the last one
    /// (tiles or objects).
    pub(crate) fn set_objects(&self, capture: ObjectPredictionCapture) {
        self.tiles.write().expect("Poisoned").clear();
        *self.objects.write().expect("Poisoned") = Some(capture.into());
    }

    pub(crate) fn clear(&self) {
        self.tiles.write().expect("Poisoned").clear();
        *self.objects.write().expect("Poisoned") = None;
    }

    /// Whether there is a prediction (tiles or objects) to show.
    pub(crate) fn has_prediction(&self) -> bool {
        !self.tiles.read().expect("Poisoned").is_empty()
            || self.objects.read().expect("Poisoned").is_some()
    }

    /// Whether the prediction is an object classifier's.
    pub(crate) fn has_objects(&self) -> bool {
        self.objects.read().expect("Poisoned").is_some()
    }

    /// How opaque each object is drawn (see [`ObjectAlphas::get`]) - read
    /// once per redraw; `None`: the overlay is off or shows no objects.
    pub(crate) fn object_alphas(&self) -> Option<ObjectAlphas<'_>> {
        let settings = self.settings.read().expect("Poisoned");
        if !settings.on {
            return None;
        }
        let opacity = settings.opacity;
        drop(settings);
        let objects = self.objects.read().expect("Poisoned");
        objects
            .is_some()
            .then_some(ObjectAlphas { opacity, objects })
    }

    /// The class object `id` is predicted as and how probable it is
    /// (0..=100 %).
    pub(crate) fn object_value(&self, id: &ObjectId) -> Option<(String, u8)> {
        let objects = self.objects.read().expect("Poisoned");
        let objects = objects.as_ref()?;
        let (class, p) = objects.objects.get(id)?;
        let class = objects.classes.get(usize::from(*class))?;
        Some((class.name.clone(), percent(*p)))
    }

    /// The `n` objects the model is least sure of, least sure first: id,
    /// predicted class (index into the legend) and %.
    pub(crate) fn most_unsure_objects(&self, n: usize) -> Vec<(ObjectId, usize, u8)> {
        let objects = self.objects.read().expect("Poisoned");
        let Some(objects) = objects.as_ref() else {
            return Vec::new();
        };
        let mut unsure: Vec<(&ObjectId, &(u8, u8))> = objects.objects.iter().collect();
        // Ties: by id, so the list doesn't shuffle between previews.
        unsure.sort_unstable_by(|a, b| a.1.1.cmp(&b.1.1).then_with(|| a.0.cmp(b.0)));
        unsure
            .into_iter()
            .take(n)
            .map(|(id, (class, p))| (id.clone(), usize::from(*class), percent(*p)))
            .collect()
    }

    /// The class predicted at image pixel `(x, y)` and how probable it is
    /// (0..=100 %) - for the values under the cursor.
    pub(crate) fn value_at(&self, x: f32, y: f32) -> Option<(String, u8)> {
        let tiles = self.tiles.read().expect("Poisoned");
        let (x, y) = (x.floor() as i64, y.floor() as i64);
        tiles.iter().find_map(|tile| {
            let i = tile.index_of(x, y)?;
            let class = tile
                .capture
                .classes
                .get(usize::from(tile.capture.class_of_pixel[i]))?;
            Some((class.name.clone(), percent(tile.capture.probability[i])))
        })
    }

    /// Draws the tiles into `buffer` (see [`draw_prediction`]) - with the
    /// settings, if the overlay is on.
    pub(crate) fn draw(
        &self,
        buffer: &mut [slint::Rgba8Pixel],
        view: ScreenTransform,
        colour_of: impl Fn(&evanalyzer_app::analysis::PredictedClass) -> [u8; 3],
    ) {
        let settings = self.settings();
        if !settings.on {
            return;
        }
        let tiles = self.tiles.read().expect("Poisoned");
        draw_prediction(buffer, view, &tiles, settings.opacity, |class| {
            let background = class.segmentation_class
                == evanalyzer_cfg::core_types::SegmentationClass::BACKGROUND;
            match (background, settings.show_background) {
                (true, false) => None,
                (true, true) => Some(BACKGROUND_RGB),
                (false, _) => Some(colour_of(class)),
            }
        });
    }
}

/// The object prediction while a redraw reads it.
pub(crate) struct ObjectAlphas<'a> {
    opacity: f32,
    objects: std::sync::RwLockReadGuard<'a, Option<ObjectPrediction>>,
}

impl ObjectAlphas<'_> {
    /// How opaque object `id` is drawn, 0..=1 - the opacity × [`strength`]
    /// of its prediction; `None`: it wasn't scored (drawn as usual).
    pub(crate) fn get(&self, id: &ObjectId) -> Option<f32> {
        let objects = self.objects.as_ref()?;
        let (_, p) = objects.objects.get(id)?;
        Some(self.opacity * strength(*p, objects.classes.len()))
    }
}

/// A probability byte (0..=255) as 0..=100 %, rounded.
fn percent(probability: u8) -> u8 {
    ((u32::from(probability) * 100 + 127) / 255) as u8
}

/// Where image pixels land in a screen-space buffer: image pixel `p` at
/// `p * zoom + off`, the buffer `width` × `height`.
#[derive(Clone, Copy, Debug)]
pub(crate) struct ScreenTransform {
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) zoom: f32,
    pub(crate) off_x: f32,
    pub(crate) off_y: f32,
}

/// How strongly a pixel predicted with `probability` (0..=255) is
/// coloured, 0..=1: transparent where the classes tie (1/n of n classes -
/// the least a winning class can have) up to fully at 100 %.
pub(crate) fn strength(probability: u8, n_classes: usize) -> f32 {
    let floor = 1.0 / n_classes.max(2) as f32;
    ((f32::from(probability) / 255.0 - floor) / (1.0 - floor)).clamp(0.0, 1.0)
}

/// Blends every screen pixel of the buffer a tile covers in the colour of
/// its predicted class (`colour_of`; `None` leaves it clear), at
/// `opacity` × [`strength`]. Each screen pixel takes the image pixel under
/// its centre, so zoomed in a pixel is a block, zoomed out pixels drop out
/// as in the image layer.
pub(crate) fn draw_prediction(
    buffer: &mut [slint::Rgba8Pixel],
    view: ScreenTransform,
    tiles: &[PredictionTile],
    opacity: f32,
    colour_of: impl Fn(&evanalyzer_app::analysis::PredictedClass) -> Option<[u8; 3]>,
) {
    let ScreenTransform {
        width,
        height,
        zoom,
        off_x,
        off_y,
    } = view;
    if zoom <= 0.0 || opacity <= 0.0 {
        return;
    }
    for tile in tiles {
        let capture = &tile.capture;
        let colours: Vec<Option<[u8; 3]>> = capture.classes.iter().map(&colour_of).collect();
        let n = capture.classes.len();
        // The tile's screen rectangle, cut to the buffer.
        let screen = |image: usize, len: usize, off: f32, max: u32| {
            let start = (image as f32 * zoom + off).floor().max(0.0) as u32;
            let end = ((image + len) as f32 * zoom + off)
                .ceil()
                .clamp(0.0, max as f32) as u32;
            (start.min(max), end)
        };
        let (sx0, sx1) = screen(tile.tile_offset_x, capture.width, off_x, width);
        let (sy0, sy1) = screen(tile.tile_offset_y, capture.height, off_y, height);
        for sy in sy0..sy1 {
            let iy = ((sy as f32 + 0.5 - off_y) / zoom).floor() as i64;
            let row = &mut buffer[(sy * width) as usize..((sy + 1) * width) as usize];
            for sx in sx0..sx1 {
                let ix = ((sx as f32 + 0.5 - off_x) / zoom).floor() as i64;
                let Some(i) = tile.index_of(ix, iy) else {
                    continue;
                };
                let class = usize::from(capture.class_of_pixel[i]);
                let Some(Some(rgb)) = colours.get(class) else {
                    continue;
                };
                let alpha = opacity * strength(capture.probability[i], n);
                if alpha > 0.0 {
                    blend(&mut row[sx as usize], *rgb, alpha);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use evanalyzer_app::analysis::PredictedClass;
    use evanalyzer_cfg::core_types::SegmentationClass;

    /// A 2 × 1 tile at (`x`, `y`): Background (sure) left, Nucleus right
    /// at `p`.
    fn tile(x: usize, y: usize, p: u8) -> PredictionTile {
        PredictionTile {
            tile_offset_x: x,
            tile_offset_y: y,
            capture: PredictionCapture {
                width: 2,
                height: 1,
                classes: vec![
                    PredictedClass {
                        name: "Background".into(),
                        segmentation_class: SegmentationClass::BACKGROUND,
                    },
                    PredictedClass {
                        name: "Nucleus".into(),
                        segmentation_class: SegmentationClass(7),
                    },
                ],
                class_of_pixel: vec![0, 1],
                probability: vec![255, p],
            },
        }
    }

    const BLUE: [u8; 3] = [0x2a, 0x6b, 0xdf];

    fn draw(tiles: &[PredictionTile], width: u32, zoom: f32, off_x: f32) -> Vec<slint::Rgba8Pixel> {
        let mut buffer = vec![slint::Rgba8Pixel::default(); (width * 1) as usize];
        let view = ScreenTransform {
            width,
            height: 1,
            zoom,
            off_x,
            off_y: 0.0,
        };
        draw_prediction(&mut buffer, view, tiles, 1.0, |class| {
            (class.segmentation_class != SegmentationClass::BACKGROUND).then_some(BLUE)
        });
        buffer
    }

    /// Time to draw a full-HD object layer (plus a 64 px margin) covered by
    /// 512² prediction tiles of 3 classes, at zoom 1, 0.25 and 4. Run with
    /// `cargo test --release -p evanalyzer_gui --lib bench_draw_prediction -- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn bench_draw_prediction() {
        let (width, height) = (1920 + 128, 1080 + 128);
        let classes: Vec<PredictedClass> = (0..3)
            .map(|i| PredictedClass {
                name: format!("c{i}"),
                segmentation_class: SegmentationClass(i),
            })
            .collect();
        let tiles: Vec<PredictionTile> = (0..8 * 4)
            .map(|i| PredictionTile {
                tile_offset_x: (i % 8) * 512,
                tile_offset_y: (i / 8) * 512,
                capture: PredictionCapture {
                    width: 512,
                    height: 512,
                    classes: classes.clone(),
                    class_of_pixel: (0..512 * 512).map(|p| (p % 3) as u8).collect(),
                    probability: (0..512 * 512).map(|p| (p % 256) as u8).collect(),
                },
            })
            .collect();
        for zoom in [1.0, 0.25, 4.0] {
            let mut times = Vec::new();
            for _ in 0..5 {
                let mut buffer = vec![slint::Rgba8Pixel::default(); width * height];
                let start = std::time::Instant::now();
                draw_prediction(
                    &mut buffer,
                    ScreenTransform {
                        width: width as u32,
                        height: height as u32,
                        zoom,
                        off_x: 0.0,
                        off_y: 0.0,
                    },
                    &tiles,
                    0.7,
                    |_| Some(BLUE),
                );
                times.push(start.elapsed().as_secs_f64() * 1000.0);
            }
            times.sort_by(f64::total_cmp);
            println!(
                "zoom {zoom}: median {:.1} ms (min {:.1}, max {:.1})",
                times[2], times[0], times[4]
            );
        }
    }

    #[test]
    fn the_strength_runs_from_a_tie_to_certain() {
        assert_eq!(strength(255, 3), 1.0);
        assert_eq!(strength(85, 3), 0.0, "1/3 of 3 classes: a tie");
        assert_eq!(strength(0, 2), 0.0);
        assert!((strength(191, 2) - 0.498).abs() < 0.01, "75 % of 2 classes");
    }

    /// Each pixel in its class colour as strong as it is probable; a class
    /// without a colour (background hidden) stays clear.
    #[test]
    fn pixels_take_their_class_colour_as_strong_as_they_are_probable() {
        let buffer = draw(&[tile(0, 0, 191)], 2, 1.0, 0.0);
        assert_eq!(buffer[0].a, 0, "background left clear");
        assert_eq!([buffer[1].r, buffer[1].g, buffer[1].b], BLUE);
        assert_eq!(buffer[1].a, 127, "75 % of 2 classes: half strength");
        let sure = draw(&[tile(0, 0, 255)], 2, 1.0, 0.0);
        assert_eq!(sure[1].a, 255);
    }

    /// Zoomed in, an image pixel covers several screen pixels; panned, the
    /// tile moves with the image and what lies outside the buffer is cut.
    #[test]
    fn the_tile_follows_zoom_and_pan() {
        let zoomed = draw(&[tile(0, 0, 255)], 4, 2.0, 0.0);
        let alpha: Vec<u8> = zoomed.iter().map(|p| p.a).collect();
        assert_eq!(alpha, [0, 0, 255, 255]);
        let panned = draw(&[tile(1, 0, 255)], 2, 1.0, -1.0);
        let alpha: Vec<u8> = panned.iter().map(|p| p.a).collect();
        assert_eq!(alpha, [0, 255]);
        let outside = draw(&[tile(0, 0, 255)], 2, 1.0, -10.0);
        assert!(outside.iter().all(|p| p.a == 0));
    }

    #[test]
    fn a_new_preview_replaces_the_tiles_and_a_tile_its_place() {
        let overlay = PredictionOverlay::default();
        overlay.add_tile(tile(0, 0, 200), true);
        overlay.add_tile(tile(2, 0, 200), false);
        overlay.add_tile(tile(2, 0, 100), false);
        assert_eq!(overlay.tiles.read().unwrap().len(), 2);
        assert_eq!(overlay.value_at(3.5, 0.2), Some(("Nucleus".into(), 39)));
        overlay.add_tile(tile(4, 0, 255), true);
        assert_eq!(overlay.tiles.read().unwrap().len(), 1, "a new preview");
        assert_eq!(overlay.value_at(0.0, 0.0), None);
        assert_eq!(overlay.value_at(4.0, 0.0), Some(("Background".into(), 100)));
    }

    /// Off, nothing is drawn and the tiles go; the background follows its
    /// switch.
    #[test]
    fn the_settings_decide_what_is_drawn() {
        let overlay = PredictionOverlay::default();
        overlay.add_tile(tile(0, 0, 255), true);
        let view = ScreenTransform {
            width: 2,
            height: 1,
            zoom: 1.0,
            off_x: 0.0,
            off_y: 0.0,
        };
        let drawn = |overlay: &PredictionOverlay| {
            let mut buffer = vec![slint::Rgba8Pixel::default(); 2];
            overlay.draw(&mut buffer, view, |_| BLUE);
            buffer.iter().map(|p| p.a).collect::<Vec<_>>()
        };
        assert_eq!(drawn(&overlay), [0, 0], "off by default");
        overlay.set_settings(PredictionOverlaySettings {
            on: true,
            opacity: 1.0,
            ..Default::default()
        });
        overlay.add_tile(tile(0, 0, 255), true);
        assert_eq!(drawn(&overlay), [255, 255]);
        overlay.set_settings(PredictionOverlaySettings {
            on: true,
            opacity: 1.0,
            show_background: false,
            ..Default::default()
        });
        assert_eq!(drawn(&overlay), [0, 255]);
        overlay.set_settings(PredictionOverlaySettings::default());
        assert!(!overlay.has_prediction(), "off frees the tiles");
    }

    /// Three objects of a two-class object classifier: sure Mitotic, a tie,
    /// fairly sure Interphase.
    fn object_capture() -> ObjectPredictionCapture {
        use evanalyzer_app::analysis::PredictedObjectClass;
        use evanalyzer_cfg::core_types::ObjectClass;
        ObjectPredictionCapture {
            classes: vec![
                PredictedObjectClass {
                    name: "Interphase".into(),
                    object_class: ObjectClass::Global(1),
                },
                PredictedObjectClass {
                    name: "Mitotic".into(),
                    object_class: ObjectClass::Global(2),
                },
            ],
            objects: vec![
                (ObjectId(1), 1, 255),
                (ObjectId(2), 0, 128),
                (ObjectId(3), 0, 204),
            ],
        }
    }

    /// Each object is as opaque as the model is sure of it (a tie:
    /// clear), with the overlay on only; unscored objects keep their look.
    #[test]
    fn an_object_is_as_strong_as_the_model_is_sure_of_it() {
        let overlay = PredictionOverlay::default();
        overlay.set_objects(object_capture());
        assert!(overlay.object_alphas().is_none(), "off");
        overlay.set_settings(PredictionOverlaySettings {
            on: true,
            opacity: 0.5,
            ..Default::default()
        });
        let alphas = overlay.object_alphas().unwrap();
        assert_eq!(alphas.get(&ObjectId(1)), Some(0.5));
        let tie = alphas.get(&ObjectId(2)).unwrap();
        assert!(tie < 0.01, "{tie}");
        let fair = alphas.get(&ObjectId(3)).unwrap();
        assert!((fair - 0.5 * 0.6).abs() < 0.01, "{fair}");
        assert_eq!(alphas.get(&ObjectId(99)), None, "not scored");
        drop(alphas);
        assert_eq!(
            overlay.object_value(&ObjectId(3)),
            Some(("Interphase".to_string(), 80))
        );
    }

    /// Time for the per-object work of one redraw (an `object_alpha` per
    /// object) and for the "Most unsure" list, with 100 000 scored objects.
    /// Run with `cargo test --release -p evanalyzer_gui --lib
    /// bench_object_prediction -- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn bench_object_prediction() {
        use evanalyzer_app::analysis::PredictedObjectClass;
        use evanalyzer_cfg::core_types::ObjectClass;
        let n = 100_000u128;
        let capture = ObjectPredictionCapture {
            classes: (0..3)
                .map(|i| PredictedObjectClass {
                    name: format!("c{i}"),
                    object_class: ObjectClass::Global(i),
                })
                .collect(),
            objects: (0..n)
                .map(|i| (ObjectId(i * 7919), (i % 3) as u8, (i % 256) as u8))
                .collect(),
        };
        let overlay = PredictionOverlay::default();
        overlay.set_settings(PredictionOverlaySettings {
            on: true,
            ..Default::default()
        });
        let start = std::time::Instant::now();
        overlay.set_objects(capture);
        let stored = start.elapsed();
        let start = std::time::Instant::now();
        let mut sum = 0.0;
        let alphas = overlay.object_alphas().unwrap();
        for i in 0..n {
            sum += alphas.get(&ObjectId(i * 7919)).unwrap_or(0.0);
        }
        drop(alphas);
        let redraw = start.elapsed();
        let start = std::time::Instant::now();
        let unsure = overlay.most_unsure_objects(5);
        let list = start.elapsed();
        println!(
            "100k objects: store {stored:?}, alphas of a redraw {redraw:?}, unsure list {list:?} ({sum:.0}, {})",
            unsure.len()
        );
    }

    /// The least sure objects come first, as many as asked for.
    #[test]
    fn the_most_unsure_objects_come_first() {
        let overlay = PredictionOverlay::default();
        assert!(overlay.most_unsure_objects(8).is_empty());
        overlay.set_objects(object_capture());
        assert_eq!(
            overlay.most_unsure_objects(2),
            [(ObjectId(2), 0, 50), (ObjectId(3), 0, 80)]
        );
    }

    /// Objects and tiles replace each other: one prediction at a time.
    #[test]
    fn objects_and_tiles_replace_each_other() {
        let overlay = PredictionOverlay::default();
        overlay.add_tile(tile(0, 0, 255), true);
        overlay.set_objects(object_capture());
        assert!(overlay.has_objects());
        assert_eq!(overlay.value_at(1.0, 0.0), None, "the tiles are gone");
        overlay.add_tile(tile(0, 0, 255), true);
        assert!(!overlay.has_objects());
        assert_eq!(overlay.object_value(&ObjectId(1)), None);
        overlay.clear();
        assert!(!overlay.has_prediction());
    }

    /// The legend of an object prediction: the model's classes in their
    /// project classes' colours.
    #[test]
    fn the_legend_of_an_object_prediction_has_its_classes() {
        use evanalyzer_cfg::settings::classification_settings::{Class, ClassificationSettings};
        let mut project = evanalyzer_app::project::ProjectWithRuntime::default();
        project.classification = ClassificationSettings::new_from_existing(vec![Class {
            id: 2,
            name: "Mitotic".into(),
            color: 0x2a6bdf,
            notes: String::new(),
        }]);
        let overlay = PredictionOverlay::default();
        overlay.set_objects(object_capture());
        assert_eq!(
            overlay.legend(&project),
            [
                ("Interphase".to_string(), BACKGROUND_RGB),
                ("Mitotic".to_string(), BLUE)
            ]
        );
    }
}
