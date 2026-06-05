pub mod hex_to_u32 {
    use serde::{self, Deserialize, Deserializer, Serializer};

    pub fn serialize<S>(color: &u32, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        // Using lowercase x for #ffffff or uppercase X for #FFFFFF
        serializer.serialize_str(&format!("#{:06x}", color))
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<u32, D::Error>
    where
        D: Deserializer<'de>, // This now recognizes the trait correctly
    {
        let s = String::deserialize(deserializer)?;
        u32::from_str_radix(s.trim_start_matches('#'), 16).map_err(serde::de::Error::custom)
    }
}

#[cfg(test)]
mod tests {
    use serde::{Deserialize, Serialize};

    #[derive(Serialize, Deserialize, PartialEq, Debug)]
    struct Colorful {
        #[serde(with = "super::hex_to_u32")]
        color: u32,
    }

    #[test]
    fn serializes_to_lowercase_hex_with_hash() {
        let c = Colorful { color: 0xFF8800 };
        let s = serde_json::to_string(&c).unwrap();
        assert_eq!(s, r##"{"color":"#ff8800"}"##);
    }

    #[test]
    fn deserializes_from_hex_with_hash() {
        let c: Colorful = serde_json::from_str(r##"{"color":"#ff8800"}"##).unwrap();
        assert_eq!(c.color, 0xFF8800);
    }

    #[test]
    fn round_trip_is_identity() {
        for &v in &[0u32, 0xFF0000, 0x00FF00, 0x0000FF, 0xFFFFFF, 0x123456] {
            let serialized = serde_json::to_string(&Colorful { color: v }).unwrap();
            let back: Colorful = serde_json::from_str(&serialized).unwrap();
            assert_eq!(back.color, v);
        }
    }

    #[test]
    fn zero_pads_to_six_digits() {
        let c = Colorful { color: 0x0000FF };
        let s = serde_json::to_string(&c).unwrap();
        assert_eq!(s, r##"{"color":"#0000ff"}"##);
    }

    #[test]
    fn deserialize_rejects_invalid_hex() {
        assert!(serde_json::from_str::<Colorful>(r##"{"color":"#zzzzzz"}"##).is_err());
    }
}
