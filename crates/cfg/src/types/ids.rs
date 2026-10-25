// # ids
//
// **Author:** Joachim Danmayr
// **Date:** 2026-06-05
//
// ## License
// Copyright 2026 Joachim Danmayr.
// Licensed under the **PolyForm Noncommercial License 1.0.0**.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::{
    fmt::{self},
    sync::atomic::{AtomicU64, Ordering},
};

pub type MemorySlot = u32;

// Image addressins -----------
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub enum MemoryId {
    #[serde(alias = "PIPELINE_CONTEXT", alias = "pipeline_context")]
    PipelineContext(MemorySlot),
    #[serde(alias = "PROJECT_CACHE", alias = "project_cache")]
    ProjectCache(MemorySlot),
    #[serde(alias = "MODULE_CONTEXT", alias = "module_context")]
    ModuleContext(MemorySlot),
}

impl Default for MemoryId {
    fn default() -> Self {
        MemoryId::PipelineContext(1)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub enum ImageAddress {
    #[serde(rename = "SCRATCHPAD", alias = "scratchpad", alias = "Scratchpad")]
    Scratchpad,
    #[serde(alias = "MEMORY", alias = "memory")]
    Memory(MemoryId), // Memory slot
    #[serde(alias = "CHANNEL", alias = "channel")]
    Channel(i32), // Initial based on image channel
    /// No image - only as a pipeline's start: its first step loads one
    /// (Load Image Channel, Image Cache Load).
    #[serde(rename = "NONE", alias = "none")]
    None,
}

impl Default for ImageAddress {
    fn default() -> Self {
        ImageAddress::Memory(MemoryId::PipelineContext(1))
    }
}

// The build script also compiles this file, without using these.
#[allow(dead_code)]
impl ImageAddress {
    /// Kind and number as the pipeline UI edits them: ("scratchpad", 0),
    /// ("memory", slot), ("cache", slot), ("module", slot) or ("channel",
    /// channel).
    pub fn param_parts(&self) -> (&'static str, i64) {
        match *self {
            ImageAddress::Scratchpad => ("scratchpad", 0),
            ImageAddress::Memory(MemoryId::PipelineContext(slot)) => ("memory", slot as i64),
            ImageAddress::Memory(MemoryId::ProjectCache(slot)) => ("cache", slot as i64),
            ImageAddress::Memory(MemoryId::ModuleContext(slot)) => ("module", slot as i64),
            ImageAddress::Channel(channel) => ("channel", channel as i64),
            ImageAddress::None => ("none", 0),
        }
    }

    /// The pipeline UI's value string, e.g. "channel:2" or "scratchpad".
    /// Round-trips through [`ImageAddress::from_param_value`].
    pub fn to_param_value(&self) -> String {
        match self.param_parts() {
            ("scratchpad", _) => "scratchpad".to_string(),
            ("none", _) => "none".to_string(),
            (kind, nr) => format!("{kind}:{nr}"),
        }
    }

    /// Parses [`ImageAddress::to_param_value`]'s format; `None` for anything
    /// else (unknown kind, missing or out-of-range number).
    pub fn from_param_value(value: &str) -> Option<Self> {
        if value == "scratchpad" {
            return Some(ImageAddress::Scratchpad);
        }
        if value == "none" {
            return Some(ImageAddress::None);
        }
        let (kind, nr) = value.split_once(':')?;
        match kind {
            "memory" => Some(ImageAddress::Memory(MemoryId::PipelineContext(
                nr.parse().ok()?,
            ))),
            "cache" => Some(ImageAddress::Memory(MemoryId::ProjectCache(
                nr.parse().ok()?,
            ))),
            "module" => Some(ImageAddress::Memory(MemoryId::ModuleContext(
                nr.parse().ok()?,
            ))),
            "channel" => Some(ImageAddress::Channel(nr.parse().ok()?)),
            _ => None,
        }
    }
}

// Pipeline ID -----------------
#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
    Eq,
    Hash,
    Serialize,
    Deserialize,
    Default,
    JsonSchema,
    Ord,
    PartialOrd,
)]
pub struct PipelineId(pub u32);

impl fmt::Display for PipelineId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Pipeline({})", self.0) // Or just self.0
    }
}

#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
    Eq,
    Hash,
    Serialize,
    Deserialize,
    Default,
    JsonSchema,
    Ord,
    PartialOrd,
)]
pub struct ModuleId(pub u32);

impl fmt::Display for ModuleId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Module({})", self.0) // Or just self.0
    }
}

/// A module parameter's key, unique within its module definition and
/// readable in the file (`"min_size"`). The values a module instance holds
/// are keyed by it, so they survive reordering and relabelling the
/// definition's parameters, and updating it from the library. Made from the
/// label when the parameter is made, then kept.
#[derive(
    Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize, Default, JsonSchema, Ord, PartialOrd,
)]
#[serde(transparent)]
pub struct ModuleParamKey(pub String);

impl ModuleParamKey {
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl From<&str> for ModuleParamKey {
    fn from(key: &str) -> Self {
        Self(key.to_string())
    }
}

impl fmt::Display for ModuleParamKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

/// A pipeline step's id: a small number, unique within its list of steps
/// (a pipeline, or a module definition), so it reads well in a hand-written
/// project file (`"id": 7`) and in references to the step. Assigned by
/// whoever adds a step - the next number after the list's highest, never
/// reused within the list. `0` ([`StepId::UNASSIGNED`]) means "not
/// assigned yet": a step written without an id gets one when the project is
/// loaded.
#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
    Eq,
    Hash,
    Serialize,
    Deserialize,
    Default,
    JsonSchema,
    Ord,
    PartialOrd,
)]
pub struct StepId(pub u32);

impl StepId {
    /// Not assigned yet - also the id of a command that isn't a project
    /// step (e.g. tile merging), which no breakpoint can target.
    pub const UNASSIGNED: StepId = StepId(0);
}

impl fmt::Display for StepId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

// Object ID -----------------
/// An object's id - a UUIDv7. Stored in JSON as the UUID string (as in the
/// results database), in binary formats as the number: project files are loaded through `serde_json::Value`, which
/// keeps numbers above 2^64 only as `f64`, so as a number the id came back
/// wrong. Numbers are still read - from files written before; a rounded one
/// rounds the same way everywhere in the file, so references between
/// objects (parent, children, coloc partners) still match.
#[derive(Debug, Clone, Default, PartialEq, Eq, JsonSchema, Hash, PartialOrd, Ord)]
pub struct ObjectId(#[schemars(with = "String")] pub u128);

impl Serialize for ObjectId {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        // Binary formats (e.g. the tile scratch files) keep the compact number.
        if serializer.is_human_readable() {
            serializer.collect_str(self)
        } else {
            serializer.serialize_u128(self.0)
        }
    }
}

impl<'de> Deserialize<'de> for ObjectId {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct Visitor;
        impl serde::de::Visitor<'_> for Visitor {
            type Value = ObjectId;
            fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
                f.write_str("an object id: a UUID string")
            }
            fn visit_str<E: serde::de::Error>(self, v: &str) -> Result<ObjectId, E> {
                u128::from_str_radix(&v.replace('-', ""), 16)
                    .map(ObjectId)
                    .map_err(E::custom)
            }
            fn visit_u64<E: serde::de::Error>(self, v: u64) -> Result<ObjectId, E> {
                Ok(ObjectId(v as u128))
            }
            fn visit_u128<E: serde::de::Error>(self, v: u128) -> Result<ObjectId, E> {
                Ok(ObjectId(v))
            }
            // Files written before: an id above 2^64 came through as `f64`.
            fn visit_f64<E: serde::de::Error>(self, v: f64) -> Result<ObjectId, E> {
                if v.is_finite() && v >= 0.0 {
                    Ok(ObjectId(v as u128))
                } else {
                    Err(E::custom(format!("invalid object id {v}")))
                }
            }
        }
        if deserializer.is_human_readable() {
            deserializer.deserialize_any(Visitor)
        } else {
            u128::deserialize(deserializer).map(ObjectId)
        }
    }
}

#[allow(dead_code)]
impl ObjectId {
    pub fn next() -> Self {
        // Atomic increment ensures every ID is unique across all threads
        Self(fast_uuid_v7::gen_id_u128())
    }

    pub fn to_string(&self) -> String {
        format!("{}", self)
    }
}

impl fmt::Display for ObjectId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let v = self.0;
        write!(
            f,
            "{:08x}-{:04x}-{:04x}-{:04x}-{:012x}",
            (v >> 96) as u32,
            ((v >> 80) & 0xFFFF) as u16,
            ((v >> 64) & 0xFFFF) as u16,
            ((v >> 48) & 0xFFFF) as u16,
            v & 0x0000_FFFF_FFFF_FFFF_u128
        )
    }
}

// Track ID -----------------
#[allow(dead_code)]
static GLOBAL_TRACK_ID_COUNTER: AtomicU64 = AtomicU64::new(1);

#[derive(Serialize, Deserialize, Debug, Clone, Default, PartialEq, Eq, JsonSchema)]
pub struct TrackId(pub u64);

#[allow(dead_code)]
impl TrackId {
    pub fn next() -> Self {
        // Atomic increment ensures every ID is unique across all threads
        Self(GLOBAL_TRACK_ID_COUNTER.fetch_add(1, Ordering::SeqCst))
    }

    pub fn to_string(&self) -> String {
        let tmp: String = format!("{}", self.0);
        tmp
    }
}

// ImageChannel Idx -----------------
/// 0-based index of an image channel, as a command setting. Shown in the
/// pipeline UI as the channel picker (channel name + color of the open
/// image). Serializes as a plain number, so it can replace an existing
/// integer channel field without breaking saved projects.
#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
    Eq,
    Hash,
    Serialize,
    Deserialize,
    Default,
    JsonSchema,
    Ord,
    PartialOrd,
)]
pub struct ImageChannelIdx(pub u32);

impl fmt::Display for ImageChannelIdx {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn image_address_param_value_round_trips_every_variant() {
        for address in [
            ImageAddress::Scratchpad,
            ImageAddress::Memory(MemoryId::PipelineContext(3)),
            ImageAddress::Memory(MemoryId::ProjectCache(7)),
            ImageAddress::Memory(MemoryId::ModuleContext(2)),
            ImageAddress::None,
            ImageAddress::Channel(0),
            ImageAddress::Channel(15),
        ] {
            let value = address.to_param_value();
            assert_eq!(
                ImageAddress::from_param_value(&value),
                Some(address),
                "{value}"
            );
        }
        assert_eq!(ImageAddress::Channel(2).to_param_value(), "channel:2");
        assert_eq!(ImageAddress::Scratchpad.to_param_value(), "scratchpad");
        assert_eq!(
            ImageAddress::Memory(MemoryId::ModuleContext(2)).to_param_value(),
            "module:2"
        );
        assert_eq!(
            serde_json::to_string(&MemoryId::ModuleContext(2)).unwrap(),
            r#"{"moduleContext":2}"#
        );
    }

    #[test]
    fn image_address_from_param_value_rejects_malformed_values() {
        for value in [
            "",
            "channel",
            "channel:",
            "channel:x",
            "memory:-1",
            "foo:1",
            "Channel 2",
        ] {
            assert_eq!(ImageAddress::from_param_value(value), None, "{value:?}");
        }
    }

    #[test]
    fn memory_id_default_is_pipeline_context_slot_one() {
        assert_eq!(MemoryId::default(), MemoryId::PipelineContext(1));
    }

    #[test]
    fn memory_id_serializes_as_camel_case() {
        let json = serde_json::to_value(MemoryId::PipelineContext(3)).unwrap();
        assert_eq!(json, serde_json::json!({"pipelineContext": 3}));
    }

    /// `camelCase` is the on-disk shape (see `memory_id_serializes_as_camel_case`),
    /// but SCREAMING_SNAKE_CASE and snake_case must still deserialize - e.g.
    /// older documents, or ones hand-authored using either casing.
    #[test]
    fn memory_id_deserializes_every_variant_alias_casing() {
        for (json, expected) in [
            (
                serde_json::json!({"pipelineContext": 2}),
                MemoryId::PipelineContext(2),
            ),
            (
                serde_json::json!({"PIPELINE_CONTEXT": 2}),
                MemoryId::PipelineContext(2),
            ),
            (
                serde_json::json!({"pipeline_context": 2}),
                MemoryId::PipelineContext(2),
            ),
            (
                serde_json::json!({"projectCache": 5}),
                MemoryId::ProjectCache(5),
            ),
            (
                serde_json::json!({"PROJECT_CACHE": 5}),
                MemoryId::ProjectCache(5),
            ),
            (
                serde_json::json!({"project_cache": 5}),
                MemoryId::ProjectCache(5),
            ),
        ] {
            let parsed: MemoryId = serde_json::from_value(json.clone())
                .unwrap_or_else(|e| panic!("{json} failed to deserialize: {e}"));
            assert_eq!(parsed, expected, "for input {json}");
        }
    }

    #[test]
    fn image_address_default_is_memory_pipeline_context_slot_one() {
        assert_eq!(
            ImageAddress::default(),
            ImageAddress::Memory(MemoryId::PipelineContext(1))
        );
    }

    /// `Scratchpad` deliberately serializes as SCREAMING_SNAKE_CASE
    /// (`"SCRATCHPAD"`), unlike `Memory`/`Channel` which follow the enum's
    /// `camelCase` default - a per-variant `rename` override.
    #[test]
    fn image_address_scratchpad_serializes_as_screaming_snake_case() {
        let json = serde_json::to_value(ImageAddress::Scratchpad).unwrap();
        assert_eq!(json, serde_json::json!("SCRATCHPAD"));
    }

    #[test]
    fn image_address_scratchpad_deserializes_both_casings() {
        assert_eq!(
            serde_json::from_value::<ImageAddress>(serde_json::json!("SCRATCHPAD")).unwrap(),
            ImageAddress::Scratchpad
        );
        assert_eq!(
            serde_json::from_value::<ImageAddress>(serde_json::json!("scratchpad")).unwrap(),
            ImageAddress::Scratchpad
        );
    }

    #[test]
    fn image_address_memory_and_channel_still_serialize_as_camel_case() {
        assert_eq!(
            serde_json::to_value(ImageAddress::Memory(MemoryId::PipelineContext(1))).unwrap(),
            serde_json::json!({"memory": {"pipelineContext": 1}})
        );
        assert_eq!(
            serde_json::to_value(ImageAddress::Channel(3)).unwrap(),
            serde_json::json!({"channel": 3})
        );
    }

    #[test]
    fn image_channel_idx_serializes_as_a_plain_number() {
        assert_eq!(
            serde_json::to_value(ImageChannelIdx(3)).unwrap(),
            serde_json::json!(3)
        );
        assert_eq!(
            serde_json::from_value::<ImageChannelIdx>(serde_json::json!(3)).unwrap(),
            ImageChannelIdx(3)
        );
    }

    #[test]
    fn pipeline_id_display_wraps_the_number() {
        assert_eq!(PipelineId(42).to_string(), "Pipeline(42)");
    }

    #[test]
    fn pipeline_id_default_is_zero() {
        assert_eq!(PipelineId::default(), PipelineId(0));
    }

    #[test]
    fn object_id_next_generates_distinct_ids() {
        let a = ObjectId::next();
        let b = ObjectId::next();
        assert_ne!(a, b);
    }

    #[test]
    fn object_id_display_formats_as_a_uuid() {
        // 0x0123456789abcdef0123456789abcdef split into the standard
        // 8-4-4-4-12 hex components.
        let id = ObjectId(0x0123_4567_89ab_cdef_0123_4567_89ab_cdef);
        assert_eq!(id.to_string(), "01234567-89ab-cdef-0123-456789abcdef");
    }

    #[test]
    fn object_id_default_is_zero() {
        assert_eq!(ObjectId::default(), ObjectId(0));
    }

    #[test]
    fn track_id_next_increments_and_stays_distinct_across_calls() {
        let a = TrackId::next();
        let b = TrackId::next();
        assert_ne!(a.0, b.0);
        assert!(b.0 > a.0);
    }

    #[test]
    fn track_id_to_string_formats_the_inner_value() {
        assert_eq!(TrackId(7).to_string(), "7");
    }

    #[test]
    fn track_id_default_is_zero() {
        assert_eq!(TrackId::default(), TrackId(0));
    }
}

#[cfg(test)]
mod object_id_tests {
    use super::*;

    #[test]
    fn object_ids_survive_loading_through_a_json_value() {
        let id = ObjectId::next();
        let json = serde_json::to_string(&id).unwrap();
        assert_eq!(json, format!("\"{id}\""), "the UUID string");
        let value: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(serde_json::from_value::<ObjectId>(value).unwrap(), id);
        assert_eq!(
            serde_json::from_str::<ObjectId>(&format!("\"{:032x}\"", id.0)).unwrap(),
            id,
            "without dashes too"
        );
    }

    #[test]
    fn object_ids_of_files_written_before_still_load() {
        assert_eq!(
            serde_json::from_str::<ObjectId>("42").unwrap(),
            ObjectId(42)
        );
        // A UUIDv7 written as a number arrives as f64: rounded, but the same
        // number always rounds to the same id, so references still match.
        let old = format!("{}", ObjectId::next().0);
        let value: serde_json::Value = serde_json::from_str(&old).unwrap();
        let a = serde_json::from_value::<ObjectId>(value.clone()).unwrap();
        let b = serde_json::from_value::<ObjectId>(value).unwrap();
        assert_eq!(a, b);
    }
}
