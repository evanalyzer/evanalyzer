//! Messages exchanged between client and server, and the conversions between
//! in-memory types and their wire form (pixels split off into frame blobs).

use crate::ai_learning::TrainingItems;
use crate::backend::{
    AnalysisRequest, DirEntry, Place, TemplateFolders, TileRequest, TrainingRequest,
};
use crate::images::{ImageChannel, ImageMeta, RawImageInfo, image_from_raw, image_to_raw};
use crate::job::{JobOutput, PreviewRequest, ProgressEvent};
use crate::result::{
    BoxplotFilter, BoxplotResult, ColumnEntry, DatabaseResult, GroupedByImageFilter,
    HistogramFilter, HistogramResult, ImageEntry, ImageHeatmapFilter, ListFilter, PlateFilter,
    ScatterFilter, ScatterResult, View, WellFilter,
};
use evanalyzer_cfg::core_types::{InternalErrors, TrainingProgressEvent};
use evanalyzer_cfg::settings::classification_settings::Class;
use evanalyzer_cfg::settings::object_settings::ObjectMetricSettings;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::sync::Arc;

/// Bumped on every incompatible change to the messages below. Client and
/// server must also run the same app version, since requests carry the
/// app's own settings types.
pub const PROTOCOL_VERSION: u32 = 3;

pub(crate) const APP_VERSION: &str = env!("CARGO_PKG_VERSION");

#[derive(Serialize, Deserialize)]
pub(crate) enum ClientMsg {
    Hello {
        protocol_version: u32,
        app_version: String,
        token: String,
    },
    Request {
        id: u64,
        request: Request,
    },
    Cancel {
        id: u64,
    },
    CloseImage {
        handle: u64,
    },
    CloseResults {
        handle: u64,
    },
}

#[derive(Serialize, Deserialize)]
pub(crate) enum Request {
    StartAnalysis(AnalysisRequest),
    StartPreview(PreviewRequest),
    StartTraining(TrainingRequest),
    OpenImage {
        path: PathBuf,
    },
    ReadTile {
        handle: u64,
        tile: TileRequest,
    },
    ReadImageMeta {
        path: PathBuf,
    },
    OpenResults {
        path: PathBuf,
    },
    /// A postcard-encoded [`ResultsQuery`] follows as blob 0.
    QueryResults {
        handle: u64,
    },
    /// A postcard-encoded `ResultExport` follows as blob 0; progress
    /// streams back until `ExportDone`.
    ExportResults {
        handle: u64,
    },
    TemplateFolders,
    Places,
    ListDir {
        path: PathBuf,
    },
    Stat {
        path: PathBuf,
    },
    ReadFile {
        path: PathBuf,
    },
    /// The file's contents follow as blob 0.
    WriteFile {
        path: PathBuf,
    },
    CreateDir {
        path: PathBuf,
    },
    Rename {
        from: PathBuf,
        to: PathBuf,
    },
    RemoveAll {
        path: PathBuf,
    },
}

/// One results-database operation. Travels as postcard (not JSON) because
/// results carry `NaN`/`±inf` floats, which JSON can't represent.
#[derive(Serialize, Deserialize)]
pub(crate) enum ResultsQuery {
    ObjectList(ListFilter),
    GroupedByImage(GroupedByImageFilter),
    GroupByPlate(PlateFilter, View),
    GroupByWell(WellFilter, View),
    ImageHeatmap(ImageHeatmapFilter, View),
    Images,
    EnableImage {
        image_rel_path: String,
        disable: bool,
    },
    ObjectClasses,
    AvailableColumns,
    ZStacks,
    TStacks,
    Boxplot(BoxplotFilter),
    Histogram(HistogramFilter),
    Scatter(ScatterFilter),
}

/// The successful result of a [`ResultsQuery`] (failures travel as
/// `Reply::Failed`).
#[derive(Serialize, Deserialize)]
pub(crate) enum ResultsAnswer {
    Table(DatabaseResult),
    Images(Vec<ImageEntry>),
    Classes(Vec<Class>),
    Columns(Vec<ColumnEntry>),
    Count(u32),
    Boxplot(BoxplotResult),
    Histogram(HistogramResult),
    Scatter(ScatterResult),
    Done,
}

pub(crate) fn to_postcard<T: Serialize>(value: &T) -> Result<Vec<u8>, InternalErrors> {
    postcard::to_allocvec(value)
        .map_err(|e| InternalErrors::Internal(format!("failed to encode message: {e}")))
}

pub(crate) fn from_postcard<T: serde::de::DeserializeOwned>(
    bytes: Option<&Vec<u8>>,
) -> Result<T, InternalErrors> {
    let bytes = bytes.ok_or_else(|| InternalErrors::Internal("message payload missing".into()))?;
    postcard::from_bytes(bytes)
        .map_err(|e| InternalErrors::Internal(format!("malformed message payload: {e}")))
}

#[derive(Serialize, Deserialize)]
pub(crate) enum ServerMsg {
    Welcome { app_version: String },
    Rejected { reason: String },
    Reply { id: u64, reply: Reply },
}

#[derive(Serialize, Deserialize)]
pub(crate) enum Reply {
    JobStarted {
        output_path: PathBuf,
        parallelism: usize,
    },
    PreviewTooManyTiles {
        tiles: usize,
    },
    /// Pixels of a `BreakpointReached` event follow as blobs.
    JobEvent(WireProgressEvent),
    JobDone(Result<JobOutput, WireError>),
    TrainingStarted {
        items: TrainingItems,
    },
    NoTrainingData,
    TrainingEvent(TrainingProgressEvent),
    /// On success the trained model follows as blob 0.
    TrainingDone(Result<(), WireError>),
    ImageOpened {
        handle: u64,
        meta: ImageMeta,
    },
    /// One blob of pixels per channel, in order.
    Tile(Vec<WireChannel>),
    ImageMeta(ImageMeta),
    ResultsOpened {
        handle: u64,
    },
    /// A postcard-encoded [`ResultsAnswer`] follows as blob 0.
    ResultsAnswer,
    ExportProgress {
        message: String,
        current: usize,
        total: usize,
    },
    ExportDone(Result<(), WireError>),
    TemplateFolders(TemplateFolders),
    Places(Vec<Place>),
    DirEntries(Vec<DirEntry>),
    Stat(Option<DirEntry>),
    /// The file's contents follow as blob 0.
    FileData,
    /// A request without a result value succeeded.
    Done,
    /// The request failed without producing its normal reply.
    Failed(WireError),
}

/// An `InternalErrors` reduced to what survives the trip: whether it was a
/// cancel (front ends treat that differently) and its message.
#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct WireError {
    cancelled: bool,
    message: String,
}

impl From<&InternalErrors> for WireError {
    fn from(e: &InternalErrors) -> Self {
        WireError {
            cancelled: matches!(e, InternalErrors::Cancelled),
            message: e.to_string(),
        }
    }
}

impl WireError {
    pub(crate) fn message(message: impl Into<String>) -> Self {
        WireError {
            cancelled: false,
            message: message.into(),
        }
    }

    pub(crate) fn into_internal(self) -> InternalErrors {
        if self.cancelled {
            InternalErrors::Cancelled
        } else {
            InternalErrors::Internal(format!("Server: {}", self.message))
        }
    }
}

#[derive(Serialize, Deserialize)]
pub(crate) enum WireProgressEvent {
    Started {
        total: usize,
    },
    TilesScheduled {
        total_tiles: usize,
    },
    TileCompleted {
        tile_index: usize,
        total_tiles: usize,
        objects: Vec<ObjectMetricSettings>,
    },
    WholeImagePhaseCompleted {
        completed: usize,
        total_tiles: usize,
    },
    ImageCompleted {
        index: usize,
        total: usize,
        path: PathBuf,
    },
    ImageFailed {
        path: PathBuf,
    },
    Finished,
    /// Blobs: `image`, then `segmentation` and `instances` if present.
    BreakpointReached {
        image: RawImageInfo,
        segmentation: Option<RawImageInfo>,
        instances: Option<RawImageInfo>,
        tile_offset_x: usize,
        tile_offset_y: usize,
        tile_width: usize,
        tile_height: usize,
        nr_bits: u16,
        channel_idx: Option<i32>,
    },
}

pub(crate) fn event_to_wire(event: ProgressEvent) -> (WireProgressEvent, Vec<Vec<u8>>) {
    let wire = match event {
        ProgressEvent::Started { total } => WireProgressEvent::Started { total },
        ProgressEvent::TilesScheduled { total_tiles } => {
            WireProgressEvent::TilesScheduled { total_tiles }
        }
        ProgressEvent::TileCompleted {
            tile_index,
            total_tiles,
            objects,
        } => WireProgressEvent::TileCompleted {
            tile_index,
            total_tiles,
            objects,
        },
        ProgressEvent::WholeImagePhaseCompleted {
            completed,
            total_tiles,
        } => WireProgressEvent::WholeImagePhaseCompleted {
            completed,
            total_tiles,
        },
        ProgressEvent::ImageCompleted { index, total, path } => {
            WireProgressEvent::ImageCompleted { index, total, path }
        }
        ProgressEvent::ImageFailed { path } => WireProgressEvent::ImageFailed { path },
        ProgressEvent::Finished => WireProgressEvent::Finished,
        ProgressEvent::BreakpointReached {
            image,
            segmentation,
            instances,
            tile_offset_x,
            tile_offset_y,
            tile_width,
            tile_height,
            nr_bits,
            channel_idx,
        } => {
            let mut blobs = Vec::new();
            let mut push = |img: &crate::images::ImageContainer| {
                let (info, bytes) = image_to_raw(img);
                blobs.push(bytes);
                info
            };
            let image = push(&image);
            let segmentation = segmentation.as_ref().map(&mut push);
            let instances = instances.as_ref().map(&mut push);
            return (
                WireProgressEvent::BreakpointReached {
                    image,
                    segmentation,
                    instances,
                    tile_offset_x,
                    tile_offset_y,
                    tile_width,
                    tile_height,
                    nr_bits,
                    channel_idx,
                },
                blobs,
            );
        }
    };
    (wire, Vec::new())
}

pub(crate) fn event_from_wire(
    event: WireProgressEvent,
    blobs: Vec<Vec<u8>>,
) -> Result<ProgressEvent, InternalErrors> {
    Ok(match event {
        WireProgressEvent::Started { total } => ProgressEvent::Started { total },
        WireProgressEvent::TilesScheduled { total_tiles } => {
            ProgressEvent::TilesScheduled { total_tiles }
        }
        WireProgressEvent::TileCompleted {
            tile_index,
            total_tiles,
            objects,
        } => ProgressEvent::TileCompleted {
            tile_index,
            total_tiles,
            objects,
        },
        WireProgressEvent::WholeImagePhaseCompleted {
            completed,
            total_tiles,
        } => ProgressEvent::WholeImagePhaseCompleted {
            completed,
            total_tiles,
        },
        WireProgressEvent::ImageCompleted { index, total, path } => {
            ProgressEvent::ImageCompleted { index, total, path }
        }
        WireProgressEvent::ImageFailed { path } => ProgressEvent::ImageFailed { path },
        WireProgressEvent::Finished => ProgressEvent::Finished,
        WireProgressEvent::BreakpointReached {
            image,
            segmentation,
            instances,
            tile_offset_x,
            tile_offset_y,
            tile_width,
            tile_height,
            nr_bits,
            channel_idx,
        } => {
            let mut blobs = blobs.into_iter();
            let mut pull = |info: &RawImageInfo| {
                let bytes = blobs
                    .next()
                    .ok_or_else(|| InternalErrors::Internal("missing breakpoint image".into()))?;
                image_from_raw(info, &bytes)
            };
            ProgressEvent::BreakpointReached {
                image: pull(&image)?,
                segmentation: segmentation.as_ref().map(&mut pull).transpose()?,
                instances: instances.as_ref().map(&mut pull).transpose()?,
                tile_offset_x,
                tile_offset_y,
                tile_width,
                tile_height,
                nr_bits,
                channel_idx,
            }
        }
    })
}

/// An `ImageChannel` without its pixels (those travel as a blob).
#[derive(Serialize, Deserialize)]
pub(crate) struct WireChannel {
    image: RawImageInfo,
    color: [f32; 3],
    is_visible: bool,
    c_stack: i32,
    name: String,
    is_rgb: bool,
}

pub(crate) fn channels_to_wire(channels: &[ImageChannel]) -> (Vec<WireChannel>, Vec<Vec<u8>>) {
    channels
        .iter()
        .map(|channel| {
            let (image, bytes) = image_to_raw(&channel.image);
            let wire = WireChannel {
                image,
                color: channel.color,
                is_visible: channel.is_visible,
                c_stack: channel.c_stack,
                name: channel.name.clone(),
                is_rgb: channel.is_rgb,
            };
            (wire, bytes)
        })
        .unzip()
}

pub(crate) fn channels_from_wire(
    channels: Vec<WireChannel>,
    blobs: Vec<Vec<u8>>,
) -> Result<Vec<ImageChannel>, InternalErrors> {
    if channels.len() != blobs.len() {
        return Err(InternalErrors::Internal(format!(
            "tile has {} channels but {} pixel blobs",
            channels.len(),
            blobs.len()
        )));
    }
    channels
        .into_iter()
        .zip(blobs)
        .map(|(channel, bytes)| {
            Ok(ImageChannel {
                image: Arc::new(image_from_raw(&channel.image, &bytes)?),
                color: channel.color,
                is_visible: channel.is_visible,
                c_stack: channel.c_stack,
                name: channel.name,
                is_rgb: channel.is_rgb,
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::images::{ImageContainer, ManagedImage, Point2d};
    use evanalyzer_cfg::core_types::ImagePlane;

    fn gray(values: Vec<f32>) -> ImageContainer {
        ImageContainer::F32Gray(ManagedImage {
            data: kornia_image::Image::new(
                kornia_image::ImageSize {
                    width: 2,
                    height: 2,
                },
                values,
            )
            .unwrap(),
            tile_offset: Point2d { x: 256, y: 0 },
            plane: Some(ImagePlane { z: 0, c: 1, t: 0 }),
        })
    }

    fn labels(values: Vec<u32>) -> ImageContainer {
        ImageContainer::U32(ManagedImage {
            data: kornia_image::Image::new(
                kornia_image::ImageSize {
                    width: 2,
                    height: 2,
                },
                values,
            )
            .unwrap(),
            tile_offset: Point2d::default(),
            plane: None,
        })
    }

    #[test]
    fn breakpoint_images_survive_the_wire_in_the_right_slots() {
        let event = ProgressEvent::BreakpointReached {
            image: gray(vec![0.0, 0.25, 0.5, 1.0]),
            segmentation: None,
            instances: Some(labels(vec![0, 1, 1, 2])),
            tile_offset_x: 256,
            tile_offset_y: 0,
            tile_width: 2,
            tile_height: 2,
            nr_bits: 12,
            channel_idx: Some(1),
        };
        let (wire, blobs) = event_to_wire(event);
        assert_eq!(blobs.len(), 2);
        let ProgressEvent::BreakpointReached {
            image,
            segmentation,
            instances,
            tile_offset_x,
            nr_bits,
            channel_idx,
            ..
        } = event_from_wire(wire, blobs).unwrap()
        else {
            panic!("expected a breakpoint event");
        };
        assert_eq!(image.as_f32_slice(), Some(&[0.0, 0.25, 0.5, 1.0][..]));
        assert_eq!(image.plane(), Some(ImagePlane { z: 0, c: 1, t: 0 }));
        assert!(segmentation.is_none());
        let Some(ImageContainer::U32(instances)) = instances else {
            panic!("expected the instance labels");
        };
        assert_eq!(instances.as_slice(), &[0, 1, 1, 2]);
        assert_eq!((tile_offset_x, nr_bits, channel_idx), (256, 12, Some(1)));
    }

    #[test]
    fn a_breakpoint_event_missing_its_pixels_is_an_error() {
        let (wire, _) = event_to_wire(ProgressEvent::BreakpointReached {
            image: gray(vec![0.0; 4]),
            segmentation: None,
            instances: None,
            tile_offset_x: 0,
            tile_offset_y: 0,
            tile_width: 2,
            tile_height: 2,
            nr_bits: 8,
            channel_idx: None,
        });
        assert!(event_from_wire(wire, Vec::new()).is_err());
    }

    #[test]
    fn results_answers_keep_nan_and_infinity_which_json_cannot_carry() {
        let answer = ResultsAnswer::Scatter(ScatterResult {
            points: vec![crate::result::ScatterPoint { x: f64::NAN, y: 1.5 }],
            x_min: f64::INFINITY,
            x_max: f64::NEG_INFINITY,
            y_min: -0.0,
            y_max: f64::MAX,
            total_object_count: 1,
        });
        let bytes = to_postcard(&answer).unwrap();
        let ResultsAnswer::Scatter(back) = from_postcard(Some(&bytes)).unwrap() else {
            panic!("expected a scatter answer");
        };
        assert!(back.points[0].x.is_nan());
        assert_eq!(back.points[0].y, 1.5);
        assert_eq!(back.x_min, f64::INFINITY);
        assert_eq!(back.x_max, f64::NEG_INFINITY);
        assert_eq!(back.y_max, f64::MAX);
        // JSON would have turned those into `null` and failed to read them.
        assert!(serde_json::from_str::<f64>(&serde_json::to_string(&f64::NAN).unwrap()).is_err());
    }

    #[test]
    fn a_missing_or_garbled_payload_is_an_error_not_a_panic() {
        assert!(from_postcard::<ResultsAnswer>(None).is_err());
        assert!(from_postcard::<ResultsAnswer>(Some(&vec![255, 255, 255])).is_err());
    }

    #[test]
    fn cancelled_errors_stay_cancelled_and_others_name_the_server() {
        let cancelled = WireError::from(&InternalErrors::Cancelled).into_internal();
        assert!(matches!(cancelled, InternalErrors::Cancelled));
        let other = WireError::from(&InternalErrors::Io("disk full".into())).into_internal();
        assert!(other.to_string().contains("Server"), "{other}");
        assert!(other.to_string().contains("disk full"), "{other}");
    }
}
