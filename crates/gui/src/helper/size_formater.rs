pub fn format_bits(bits: u64) -> String {
    let bytes = bits as f64 / 8.0;
    let units = ["B", "KB", "MB", "GB", "TB", "PB"];

    if bytes < 1.0 {
        return format!("{} bits", bits);
    }

    let mut size = bytes;
    let mut unit_idx = 0;

    while size >= 1024.0 && unit_idx < units.len() - 1 {
        size /= 1024.0;
        unit_idx += 1;
    }

    format!("{:.2} {}", size, units[unit_idx])
}

pub fn format_bytes(bytes: f64) -> String {
    let units = ["B", "KB", "MB", "GB", "TB", "PB"];

    let mut size = bytes;
    let mut unit_idx = 0;

    while size >= 1024.0 && unit_idx < units.len() - 1 {
        size /= 1024.0;
        unit_idx += 1;
    }

    format!("{:.2} {}", size, units[unit_idx])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn format_bits_below_one_byte_stays_in_bits() {
        assert_eq!(format_bits(0), "0 bits");
        assert_eq!(format_bits(7), "7 bits");
    }

    #[test]
    fn format_bits_converts_to_bytes_and_up() {
        assert_eq!(format_bits(8), "1.00 B");
        assert_eq!(format_bits(8 * 1024), "1.00 KB");
        assert_eq!(format_bits(8 * 1024 * 1024), "1.00 MB");
        assert_eq!(format_bits(8 * 1024 * 1024 * 1024), "1.00 GB");
    }

    #[test]
    fn format_bits_clamps_at_the_largest_unit_instead_of_overflowing_the_table() {
        // Comfortably past PB (1024^5 bytes = 8*1024^5 bits) - the loop must
        // stop advancing `unit_idx` once it reaches "PB" rather than
        // indexing past the end of `units`.
        let huge_bits = 8u64.saturating_mul(1024u64.pow(6));
        let result = format_bits(huge_bits);
        assert!(
            result.ends_with("PB"),
            "expected a PB-suffixed result, got {result:?}"
        );
    }

    #[test]
    fn format_bytes_basic_boundaries() {
        assert_eq!(format_bytes(0.0), "0.00 B");
        assert_eq!(format_bytes(500.0), "500.00 B");
        assert_eq!(format_bytes(1024.0), "1.00 KB");
        assert_eq!(format_bytes(1536.0), "1.50 KB");
        assert_eq!(format_bytes(1024.0 * 1024.0), "1.00 MB");
    }

    #[test]
    fn format_bytes_clamps_at_the_largest_unit() {
        let huge = 1024f64.powi(7); // way past PB
        let result = format_bytes(huge);
        assert!(
            result.ends_with("PB"),
            "expected a PB-suffixed result, got {result:?}"
        );
    }

    #[test]
    fn format_bytes_just_under_a_unit_boundary_stays_in_the_lower_unit() {
        assert_eq!(format_bytes(1023.0), "1023.00 B");
    }
}
