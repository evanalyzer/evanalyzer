//! Minimal ASCII table printer for the `view` command - no extra dependency
//! for what is just a quick terminal preview (full data goes through `export`).
use evanalyzer_app::result::{ColumnSpec, RoiRow, to_display_row};

const MAX_COL_WIDTH: usize = 28;

pub fn print_roi_table(specs: &[ColumnSpec], rois: &[RoiRow]) {
    let headers: Vec<String> = specs.iter().map(|c| c.label.clone()).collect();
    let rows: Vec<Vec<String>> = rois
        .iter()
        .enumerate()
        .map(|(i, roi)| to_display_row(i, roi, specs).values)
        .collect();
    print_grid(&headers, &rows);
}

fn truncate(s: &str, width: usize) -> String {
    if s.chars().count() <= width {
        return s.to_string();
    }
    let mut out: String = s.chars().take(width.saturating_sub(1)).collect();
    out.push('\u{2026}');
    out
}

fn print_grid(headers: &[String], rows: &[Vec<String>]) {
    let mut widths: Vec<usize> = headers.iter().map(|h| h.chars().count().min(MAX_COL_WIDTH)).collect();
    for row in rows {
        for (i, cell) in row.iter().enumerate() {
            if let Some(w) = widths.get_mut(i) {
                *w = (*w).max(cell.chars().count().min(MAX_COL_WIDTH));
            }
        }
    }

    let print_row = |cells: &[String]| {
        let line: Vec<String> = cells
            .iter()
            .zip(&widths)
            .map(|(c, w)| format!("{:<width$}", truncate(c, *w), width = *w))
            .collect();
        println!("{}", line.join(" | "));
    };

    print_row(headers);
    let separator: Vec<String> = widths.iter().map(|w| "-".repeat(*w)).collect();
    println!("{}", separator.join("-+-"));
    for row in rows {
        print_row(row);
    }
}
