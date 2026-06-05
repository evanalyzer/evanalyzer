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
