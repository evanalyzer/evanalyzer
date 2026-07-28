//! Minimal ASCII table printer for the `view` command - no extra dependency
//! for what is just a quick terminal preview (full data goes through `export`).
use evanalyzer_app::result::{ColumnSpec, ObjectRow, to_display_row};
use std::io::Write;

const MAX_COL_WIDTH: usize = 28;

pub fn print_object_table(specs: &[ColumnSpec], objects: &[ObjectRow]) {
    let headers: Vec<String> = specs.iter().map(|c| c.label.clone()).collect();
    let rows: Vec<Vec<String>> = objects
        .iter()
        .enumerate()
        .map(|(i, object)| to_display_row(i, object, specs).values)
        .collect();
    write_grid(&mut std::io::stdout(), &headers, &rows).ok();
}

fn truncate(s: &str, width: usize) -> String {
    if s.chars().count() <= width {
        return s.to_string();
    }
    let mut out: String = s.chars().take(width.saturating_sub(1)).collect();
    out.push('\u{2026}');
    out
}

/// Does the actual layout/formatting, writing to `out` instead of directly to
/// stdout so it's testable against an in-memory buffer.
fn write_grid(
    out: &mut impl Write,
    headers: &[String],
    rows: &[Vec<String>],
) -> std::io::Result<()> {
    let mut widths: Vec<usize> = headers.iter().map(|h| h.chars().count().min(MAX_COL_WIDTH)).collect();
    for row in rows {
        for (i, cell) in row.iter().enumerate() {
            if let Some(w) = widths.get_mut(i) {
                *w = (*w).max(cell.chars().count().min(MAX_COL_WIDTH));
            }
        }
    }

    let print_row = |out: &mut dyn Write, cells: &[String]| -> std::io::Result<()> {
        let line: Vec<String> = cells
            .iter()
            .zip(&widths)
            .map(|(c, w)| format!("{:<width$}", truncate(c, *w), width = *w))
            .collect();
        writeln!(out, "{}", line.join(" | "))
    };

    print_row(out, headers)?;
    let separator: Vec<String> = widths.iter().map(|w| "-".repeat(*w)).collect();
    writeln!(out, "{}", separator.join("-+-"))?;
    for row in rows {
        print_row(out, row)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn truncate_leaves_short_strings_untouched() {
        assert_eq!(truncate("abc", 10), "abc");
        assert_eq!(truncate("abc", 3), "abc");
    }

    #[test]
    fn truncate_replaces_the_last_char_with_an_ellipsis_when_too_long() {
        assert_eq!(truncate("abcdef", 4), "abc\u{2026}");
    }

    #[test]
    fn truncate_counts_unicode_scalars_not_bytes() {
        // 5 non-ASCII scalars, more bytes than that - would panic/mis-slice
        // if this indexed by byte offset instead of `.chars()`.
        assert_eq!(truncate("héllo", 3), "hé\u{2026}");
    }

    #[test]
    fn write_grid_pads_columns_to_the_widest_cell_in_each() {
        let mut out = Vec::new();
        write_grid(
            &mut out,
            &["ID".to_string(), "Name".to_string()],
            &[vec!["1".to_string(), "Nucleus".to_string()]],
        )
        .unwrap();

        let text = String::from_utf8(out).unwrap();
        let lines: Vec<&str> = text.lines().collect();
        assert_eq!(lines[0], "ID | Name   ");
        assert_eq!(lines[1], "---+--------");
        assert_eq!(lines[2], "1  | Nucleus");
    }

    #[test]
    fn write_grid_clamps_column_width_at_the_max_and_truncates_wider_cells() {
        let mut out = Vec::new();
        let long = "x".repeat(MAX_COL_WIDTH + 10);
        write_grid(&mut out, &["Col".to_string()], &[vec![long]]).unwrap();

        let text = String::from_utf8(out).unwrap();
        let header_width = text.lines().next().unwrap().chars().count();
        assert_eq!(header_width, MAX_COL_WIDTH);
    }

    #[test]
    fn write_grid_with_no_rows_still_prints_header_and_separator() {
        let mut out = Vec::new();
        write_grid(&mut out, &["Only".to_string()], &[]).unwrap();

        let text = String::from_utf8(out).unwrap();
        assert_eq!(text.lines().count(), 2);
    }
}
