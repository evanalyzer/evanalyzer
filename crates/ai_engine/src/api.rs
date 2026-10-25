//! # api
//!
//! **Author:** Joachim Danmayr
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.
//!
//! The interface between the analysis and an AI inference engine.
//!
//! The analysis (`evanalyzer_core`) prepares an image and the parameters of
//! one model run - a [`Request`] - and turns the plain [`Response`] into
//! segmentation maps and objects. How the model runs is up to an
//! [`InferenceEngine`]: today libtorch, loaded as a plugin library on the
//! first AI step (the [`torch`](crate) engine in this crate built as a
//! cdylib, through the C interface in [`ffi`]); later possibly a pure-Rust engine (e.g. Burn) running in the
//! process. Requests describe *what* to compute per model kind, never
//! framework tensors, so an engine can be swapped without touching the
//! algorithms.

use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Bumped on every change to the types or the C functions below: a plugin
/// built against another version is refused.
pub const API_VERSION: u32 = 2;

/// Which devices AI inference may use.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum DeviceSelection {
    /// Every GPU there is, else the CPU.
    #[default]
    Auto,
    Cpu,
    /// These GPUs (indexes); ones that don't exist are ignored.
    Gpus(Vec<usize>),
}

/// How an engine sets itself up, once per process.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EngineConfig {
    pub devices: DeviceSelection,
    /// Inferences run at once per GPU (at least 1).
    pub gpu_slots: usize,
    /// CPU cores the engine may use.
    pub cores: usize,
}

/// A device inferences run on, and how many at once.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DeviceInfo {
    /// E.g. "CPU", "CUDA:0".
    pub name: String,
    pub slots: usize,
}

/// What an engine is, after it set itself up.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EngineInfo {
    /// E.g. "libtorch 2.11.0 (CUDA)".
    pub name: String,
    pub devices: Vec<DeviceInfo>,
}

/// One model run: which model file, and what to compute with it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Request {
    /// The model file (loaded once per device and kept while unchanged).
    pub model: PathBuf,
    pub task: Task,
}

/// What to compute, per model kind. The image comes with the request as an
/// [`Image`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Task {
    /// Semantic segmentation (`[1, 1, H, W]` in, `[1, C, H, W]` out): the
    /// tile is padded to the model's needs and the result cropped back.
    UNet {
        /// Channel of the foreground probability (when the output has more
        /// than one; out-of-range values are clamped).
        foreground_channel: i32,
        /// The channels are class scores: softmax over them first. Else
        /// they are independent probability maps.
        softmax: bool,
        /// A boundary-probability channel (boundary-aware models), `-1` none.
        boundary_channel: i32,
    },
    /// Cellpose(-SAM) flows: run in 256 x 256 tiles, blended.
    Cellpose {
        /// The size the model runs at (the image is resized to it).
        net_width: usize,
        net_height: usize,
        /// Channels the model takes (at least 1).
        input_channels: i32,
        /// Copy the gray image into every channel instead of zero-filling.
        replicate_gray_channel: bool,
    },
    /// StarDist probability and ray distances.
    StarDist,
    /// YOLOv5 detection (and instance masks for segmentation exports), run
    /// in overlapping 640 x 640 windows; detections in image coordinates,
    /// before the global NMS.
    Yolov5 {
        /// The size the model runs at (the RGB image is resized to it).
        width: usize,
        height: usize,
        window_overlap: i32,
        confidence_threshold: f32,
        iou_threshold: f32,
        mask_threshold: f32,
    },
}

/// The image of a request: `channels` planes of `width * height` values
/// (row-major), one after the other.
#[derive(Debug, Clone, Copy)]
pub struct Image<'a> {
    pub width: usize,
    pub height: usize,
    pub channels: usize,
    pub data: &'a [f32],
}

/// What a model run gives back - plain data, no framework types.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Response {
    UNet {
        /// Foreground probability per pixel (`width * height`).
        #[serde(with = "floats")]
        probabilities: Vec<f32>,
        /// Boundary probability per pixel, if a boundary channel was asked
        /// for and exists.
        #[serde(with = "optional_floats")]
        boundary: Option<Vec<f32>>,
    },
    Cellpose {
        /// Each `net_width * net_height`; `cell_prob` after the sigmoid.
        #[serde(with = "floats")]
        flow_y: Vec<f32>,
        #[serde(with = "floats")]
        flow_x: Vec<f32>,
        #[serde(with = "floats")]
        cell_prob: Vec<f32>,
    },
    StarDist {
        /// `grid_height * grid_width` probabilities, `n_rays` planes of ray
        /// distances of that size.
        #[serde(with = "floats")]
        prob: Vec<f32>,
        #[serde(with = "floats")]
        dist: Vec<f32>,
        grid_height: usize,
        grid_width: usize,
        n_rays: usize,
    },
    Yolov5 {
        detections: Vec<Detection>,
    },
}

/// Float planes as one block of bytes (little-endian) instead of float by
/// float: a 2048 x 2048 Cellpose answer is 48 MB, which postcard's
/// per-element encoding made cost ~60 ms on each side of the plugin
/// boundary; as bytes it is a copy.
mod floats {
    use serde::{Deserializer, Serializer, de};

    pub fn serialize<S: Serializer>(values: &[f32], serializer: S) -> Result<S::Ok, S::Error> {
        #[cfg(target_endian = "little")]
        {
            // SAFETY: f32 has no padding and any byte is a valid u8; the
            // slice covers exactly the values' memory.
            let bytes = unsafe {
                std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), size_of_val(values))
            };
            serializer.serialize_bytes(bytes)
        }
        #[cfg(not(target_endian = "little"))]
        {
            let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
            serializer.serialize_bytes(&bytes)
        }
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(deserializer: D) -> Result<Vec<f32>, D::Error> {
        struct Bytes;
        impl de::Visitor<'_> for Bytes {
            type Value = Vec<f32>;
            fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                f.write_str("float plane bytes")
            }
            fn visit_bytes<E: de::Error>(self, bytes: &[u8]) -> Result<Vec<f32>, E> {
                if bytes.len() % 4 != 0 {
                    return Err(E::custom("float plane bytes not a multiple of 4"));
                }
                Ok(bytes
                    .chunks_exact(4)
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect())
            }
        }
        deserializer.deserialize_bytes(Bytes)
    }
}

/// [`floats`] for an optional plane.
mod optional_floats {
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    #[derive(Serialize, Deserialize)]
    struct Plane(#[serde(with = "super::floats")] Vec<f32>);

    pub fn serialize<S: Serializer>(
        values: &Option<Vec<f32>>,
        serializer: S,
    ) -> Result<S::Ok, S::Error> {
        // A borrowed plane would need its own wrapper; this clones only
        // when there is a boundary plane, a copy next to the encoding.
        values.clone().map(Plane).serialize(serializer)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(
        deserializer: D,
    ) -> Result<Option<Vec<f32>>, D::Error> {
        Ok(Option::<Plane>::deserialize(deserializer)?.map(|p| p.0))
    }
}

/// One detection, in the coordinates of the image the model ran at.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Detection {
    pub score: f32,
    pub class: usize,
    /// `[x1, y1, x2, y2]`, exclusive upper bounds.
    pub bbox: [f32; 4],
    /// Pixel mask over `mask_rect` (row-major), `None` for a filled box.
    pub mask: Option<Vec<bool>>,
    /// Integer rectangle `[x, y, width, height]` the mask covers.
    pub mask_rect: [usize; 4],
}

/// Intersection over union of two `[x1, y1, x2, y2]` boxes (`0` for boxes
/// without area).
pub fn box_iou(a: &[f32; 4], b: &[f32; 4]) -> f32 {
    let iw = (a[2].min(b[2]) - a[0].max(b[0])).max(0.0);
    let ih = (a[3].min(b[3]) - a[1].max(b[1])).max(0.0);
    let intersection = iw * ih;
    let union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - intersection;
    if union <= 0.0 {
        0.0
    } else {
        intersection / union
    }
}

/// Runs AI models. Must be usable from many threads at once: an engine
/// queues inferences on its devices itself.
pub trait InferenceEngine: Send + Sync {
    fn info(&self) -> EngineInfo;
    /// Errors are the messages shown to the user.
    fn infer(&self, request: &Request, image: Image<'_>) -> Result<Response, String>;
}

/// The C interface of an engine plugin library - stable across compilers,
/// unlike Rust's own. Requests and answers cross it as postcard bytes; the
/// image by pointer (no copy). Every byte buffer a plugin returns is given
/// back to its [`FreeFn`].
pub mod ffi {
    use super::*;

    /// A byte buffer handed across the C interface.
    #[repr(C)]
    #[derive(Debug)]
    pub struct Buf {
        pub ptr: *mut u8,
        pub len: usize,
        pub cap: usize,
    }

    impl Buf {
        pub const EMPTY: Buf = Buf {
            ptr: std::ptr::null_mut(),
            len: 0,
            cap: 0,
        };

        /// Hands `bytes` over; only the side that made it may free it.
        pub fn from_vec(bytes: Vec<u8>) -> Buf {
            let mut bytes = std::mem::ManuallyDrop::new(bytes);
            Buf {
                ptr: bytes.as_mut_ptr(),
                len: bytes.len(),
                cap: bytes.capacity(),
            }
        }

        /// Takes a buffer back that [`Buf::from_vec`] of this same library
        /// made.
        ///
        /// # Safety
        /// `self` must come from `from_vec` in this library and be taken
        /// back once.
        pub unsafe fn into_vec(self) -> Vec<u8> {
            if self.ptr.is_null() {
                return Vec::new();
            }
            unsafe { Vec::from_raw_parts(self.ptr, self.len, self.cap) }
        }

        /// The bytes, borrowed.
        ///
        /// # Safety
        /// `self` must point at `len` valid bytes.
        pub unsafe fn as_slice(&self) -> &[u8] {
            if self.ptr.is_null() {
                &[]
            } else {
                unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
            }
        }
    }

    /// Log lines from the plugin into the app's log: level (1 error … 5
    /// trace), target and message as UTF-8.
    pub type LogFn = extern "C" fn(
        level: u8,
        target: *const u8,
        target_len: usize,
        message: *const u8,
        message_len: usize,
    );

    /// `u32 eva_ai_api_version(void)` - [`API_VERSION`] of the plugin.
    pub type ApiVersionFn = unsafe extern "C" fn() -> u32;
    /// `i32 eva_ai_init(config, log, max_log_level, out_info, out_err)` -
    /// once: postcard [`EngineConfig`] in, postcard [`EngineInfo`] out; 0 on
    /// success, else `out_err` holds the message (UTF-8).
    pub type InitFn = unsafe extern "C" fn(
        config: *const u8,
        config_len: usize,
        log: LogFn,
        max_log_level: u8,
        out_info: *mut Buf,
        out_err: *mut Buf,
    ) -> i32;
    /// `i32 eva_ai_infer(request, width, height, channels, data, out, out_err)`
    /// - thread-safe: postcard [`Request`] and the image planes in,
    ///   postcard [`Response`] out; 0 on success, else `out_err`.
    pub type InferFn = unsafe extern "C" fn(
        request: *const u8,
        request_len: usize,
        width: usize,
        height: usize,
        channels: usize,
        data: *const f32,
        data_len: usize,
        out: *mut Buf,
        out_err: *mut Buf,
    ) -> i32;
    /// `void eva_ai_free(Buf)` - gives a buffer back to the plugin.
    pub type FreeFn = unsafe extern "C" fn(buf: Buf);

    pub const SYMBOL_API_VERSION: &[u8] = b"eva_ai_api_version\0";
    pub const SYMBOL_INIT: &[u8] = b"eva_ai_init\0";
    pub const SYMBOL_INFER: &[u8] = b"eva_ai_infer\0";
    pub const SYMBOL_FREE: &[u8] = b"eva_ai_free\0";

    /// postcard bytes of `value`.
    pub fn encode<T: Serialize>(value: &T) -> Result<Vec<u8>, String> {
        postcard::to_stdvec(value).map_err(|e| format!("AI interface: could not encode: {e}"))
    }

    /// `T` from postcard bytes.
    pub fn decode<'a, T: Deserialize<'a>>(bytes: &'a [u8]) -> Result<T, String> {
        postcard::from_bytes(bytes).map_err(|e| format!("AI interface: could not decode: {e}"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn requests_and_responses_cross_the_interface_unchanged() {
        let request = Request {
            model: PathBuf::from("/models/yolo.pt"),
            task: Task::Yolov5 {
                width: 700,
                height: 500,
                window_overlap: 128,
                confidence_threshold: 0.25,
                iou_threshold: 0.45,
                mask_threshold: 0.5,
            },
        };
        let bytes = ffi::encode(&request).unwrap();
        assert_eq!(ffi::decode::<Request>(&bytes).unwrap(), request);

        let response = Response::Yolov5 {
            detections: vec![Detection {
                score: 0.9,
                class: 2,
                bbox: [1.0, 2.0, 30.0, 40.0],
                mask: Some(vec![true, false, true]),
                mask_rect: [1, 2, 3, 1],
            }],
        };
        let bytes = ffi::encode(&response).unwrap();
        assert_eq!(ffi::decode::<Response>(&bytes).unwrap(), response);

        // Float planes, also odd values and an optional plane.
        for response in [
            Response::UNet {
                probabilities: vec![0.0, -1.5, f32::MAX, f32::MIN_POSITIVE, 1e-30],
                boundary: Some(vec![0.25, 0.75]),
            },
            Response::UNet {
                probabilities: vec![],
                boundary: None,
            },
            Response::Cellpose {
                flow_y: vec![1.0; 7],
                flow_x: vec![-2.0; 7],
                cell_prob: vec![0.5; 7],
            },
            Response::StarDist {
                prob: vec![0.1, 0.9],
                dist: vec![3.0; 8],
                grid_height: 1,
                grid_width: 2,
                n_rays: 4,
            },
        ] {
            let bytes = ffi::encode(&response).unwrap();
            assert_eq!(ffi::decode::<Response>(&bytes).unwrap(), response);
        }
    }

    #[test]
    fn a_float_plane_is_its_bytes_not_one_encoding_per_float() {
        let response = Response::Cellpose {
            flow_y: vec![1.0; 1000],
            flow_x: vec![1.0; 1000],
            cell_prob: vec![1.0; 1000],
        };
        // 3 x 4000 bytes plus the variant and three lengths.
        assert!(ffi::encode(&response).unwrap().len() < 3 * 4000 + 16);
    }

    #[test]
    fn iou_of_boxes() {
        assert_eq!(
            box_iou(&[0.0, 0.0, 10.0, 10.0], &[0.0, 0.0, 10.0, 10.0]),
            1.0
        );
        assert_eq!(
            box_iou(&[0.0, 0.0, 10.0, 10.0], &[20.0, 20.0, 30.0, 30.0]),
            0.0
        );
        assert!(
            (box_iou(&[0.0, 0.0, 10.0, 10.0], &[5.0, 0.0, 15.0, 10.0]) - 1.0 / 3.0).abs() < 1e-6
        );
    }

    #[test]
    fn iou_of_zero_area_boxes_is_zero_not_nan() {
        let point = [5.0, 5.0, 5.0, 5.0];
        assert_eq!(box_iou(&point, &point), 0.0);
        let line = [0.0, 0.0, 10.0, 0.0];
        assert_eq!(box_iou(&line, &line), 0.0);
    }

    #[test]
    fn a_buffer_is_handed_over_and_taken_back_whole() {
        let buf = ffi::Buf::from_vec(vec![1, 2, 3]);
        assert_eq!(unsafe { buf.as_slice() }, &[1, 2, 3]);
        assert_eq!(unsafe { buf.into_vec() }, vec![1, 2, 3]);
        assert!(unsafe { ffi::Buf::EMPTY.into_vec() }.is_empty());
    }
}
