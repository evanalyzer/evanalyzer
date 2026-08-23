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
}

impl Default for ImageAddress {
    fn default() -> Self {
        ImageAddress::Memory(MemoryId::PipelineContext(1))
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

// Object ID -----------------
#[derive(
    Serialize, Deserialize, Debug, Clone, Default, PartialEq, Eq, JsonSchema, Hash, PartialOrd, Ord,
)]
pub struct ObjectId(pub u128);

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

// Tracking ID -----------------
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

#[cfg(test)]
mod tests {
    use super::*;

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
        // 8-4-4-4-12 hex groups.
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
