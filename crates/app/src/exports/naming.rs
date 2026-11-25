//! # naming
//!
//! The names of the files and folders an export writes (`docs/exports.md`
//! §2): automatic per entry, the export's prefix in front, a part per
//! image, well, folder or class where a table is split, the plane where the
//! results have several. Pure functions - the runner joins them onto the
//! output folder.
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-11-24
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.

use crate::exports::jobs::{PlateGrid, PlateLayoutExport, TableExport, TableRows};
use evanalyzer_cfg::settings::export_settings::{ExportEntry, ExportSettings, ScriptExport};
use std::collections::{BTreeMap, HashMap};

/// A part of a file name made from user text: letters, digits, `-` and `_`
/// stay, anything else becomes `_`.
pub fn sanitize(part: &str) -> String {
    part.chars()
        .map(|c| {
            if c.is_alphanumeric() || c == '-' || c == '_' {
                c
            } else {
                '_'
            }
        })
        .collect()
}

/// The names an entry writes without the export's prefix: a data export's
/// per output (`objects`, `images`, `plate`, ... - its `name` in front),
/// drawn images' own name or `drawn_images`. None for a script, which writes
/// into its `output` folder instead.
pub fn entry_names(entry: &ExportEntry) -> Vec<String> {
    match entry {
        ExportEntry::Data(data) => crate::exports::jobs::jobs_of(data)
            .iter()
            .map(|(_, job)| job_name(job))
            .collect(),
        ExportEntry::DrawnImages(drawn) => vec![own_or(&drawn.name, "drawn_images")],
        ExportEntry::Script(_) | ExportEntry::Unknown(_) => Vec::new(),
    }
}

/// The name an output's job writes (without the export's prefix).
pub fn job_name(job: &crate::exports::jobs::Job) -> String {
    match job {
        crate::exports::jobs::Job::Table(table) => table_name(table),
        crate::exports::jobs::Job::PlateLayout(plate) => plate_name(plate),
    }
}

/// A table's own name, else the automatic one by its rows.
pub fn table_name(table: &TableExport) -> String {
    own_or(&table.name, rows_name(table.rows))
}

/// A plate layout's own name, else the automatic one by its grid.
pub fn plate_name(plate: &PlateLayoutExport) -> String {
    own_or(
        &plate.name,
        match plate.grid {
            PlateGrid::Plate => "plate",
            PlateGrid::Well => "well",
            PlateGrid::WellList => "well_list",
            PlateGrid::Image => "heatmap",
        },
    )
}

fn own_or(own: &str, automatic: &str) -> String {
    if own.trim().is_empty() {
        automatic.to_string()
    } else {
        sanitize(own.trim())
    }
}

/// The automatic name of a table by what its rows are.
pub fn rows_name(rows: TableRows) -> &'static str {
    match rows {
        TableRows::Object => "objects",
        TableRows::Image => "images",
        TableRows::Well => "wells",
        TableRows::Folder => "folders",
        TableRows::ColocPair => "coloc_pairs",
        TableRows::Track => "tracks",
        TableRows::Measurement => "measurements",
        TableRows::TimePoint => "time_course",
        TableRows::ProfilePosition => "profile",
    }
}

/// `name` with the export's prefix in front: `{prefix}_{name}`, or `name`
/// without a prefix.
pub fn with_prefix(export: &ExportSettings, name: &str) -> String {
    let prefix = export.prefix.trim();
    if prefix.is_empty() {
        name.to_string()
    } else {
        format!("{}_{name}", sanitize(prefix))
    }
}

/// The folder a script writes into, below the export's output folder: its
/// `output`, else its file's name without the extension.
pub fn script_output(script: &ScriptExport) -> String {
    let own = script.output.trim();
    if !own.is_empty() {
        return sanitize(own);
    }
    let file = script.file.rsplit(['/', '\\']).next().unwrap_or("");
    let stem = file.rsplit_once('.').map(|(stem, _)| stem).unwrap_or(file);
    if stem.is_empty() {
        "script".to_string()
    } else {
        sanitize(stem)
    }
}

/// The part of a file name of each image of a run (relative paths, `/` or
/// `\` separated): the shortest unique one. The file name without its
/// extension; where images share it, their folders are added from the
/// inside out until each name is unique; where that isn't enough (no
/// folders left), 6 characters derived from the relative path. Depends on
/// which images there are, not on their order.
pub fn unique_image_parts(images: &[String]) -> BTreeMap<String, String> {
    struct Image {
        path: String,
        parts: Vec<String>,
        depth: usize,
    }
    let mut list: Vec<Image> = images
        .iter()
        .map(|path| {
            let mut parts: Vec<String> = path
                .split(['/', '\\'])
                .filter(|part| !part.is_empty())
                .map(String::from)
                .collect();
            if let Some(file) = parts.last_mut() {
                if let Some((stem, _)) = file.rsplit_once('.')
                    && !stem.is_empty()
                {
                    *file = stem.to_string();
                }
            }
            let parts = parts.iter().map(|part| sanitize(part)).collect();
            Image {
                path: path.clone(),
                parts,
                depth: 1,
            }
        })
        .collect();
    let name = |image: &Image| {
        let from = image.parts.len().saturating_sub(image.depth);
        image.parts[from..].join("_")
    };

    loop {
        let mut by_name: HashMap<String, Vec<usize>> = HashMap::new();
        for (i, image) in list.iter().enumerate() {
            by_name.entry(name(image)).or_default().push(i);
        }
        let mut deeper = false;
        for clash in by_name.values().filter(|group| group.len() > 1) {
            for &i in clash {
                if list[i].depth < list[i].parts.len() {
                    list[i].depth += 1;
                    deeper = true;
                }
            }
        }
        if !deeper {
            return list
                .iter()
                .map(|image| {
                    let candidate = name(image);
                    let clashes = by_name.get(&candidate).is_some_and(|group| group.len() > 1);
                    let part = if clashes {
                        format!("{candidate}_{}", short_code(&image.path))
                    } else {
                        candidate
                    };
                    (image.path.clone(), part)
                })
                .collect();
        }
    }
}

/// 6 hex characters derived from `text` (FNV-1a) - the same on every
/// system and in every version.
fn short_code(text: &str) -> String {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for byte in text.replace('\\', "/").bytes() {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x0100_0000_01b3);
    }
    format!("{:06x}", hash & 0xff_ffff)
}

/// The parts of names made from user text (classes, folders), each unique:
/// sanitized; where two would be the same, the given id is added to each of
/// them (`name_3`).
pub fn unique_parts(items: &[(u32, String)]) -> BTreeMap<u32, String> {
    let mut count: HashMap<String, usize> = HashMap::new();
    for (_, name) in items {
        *count.entry(sanitize(name)).or_default() += 1;
    }
    items
        .iter()
        .map(|(id, name)| {
            let part = sanitize(name);
            if count[&part] > 1 {
                (*id, format!("{part}_{id}"))
            } else {
                (*id, part)
            }
        })
        .collect()
}

/// `_z0002` and `_t0013` for a plane (counted from 0) - only where the
/// results have several z-planes or time points.
pub fn plane_suffix((z, t): (u32, u32), several_z: bool, several_t: bool) -> String {
    let mut suffix = String::new();
    if several_z {
        suffix.push_str(&format!("_z{z:04}"));
    }
    if several_t {
        suffix.push_str(&format!("_t{t:04}"));
    }
    suffix
}

/// The name for a file where `name.ext` is taken (`IfExists::Rename`):
/// `name_2.ext`, `name_3.ext`, ... the first `taken` says isn't.
pub fn free_name(file_name: &str, taken: impl Fn(&str) -> bool) -> String {
    if !taken(file_name) {
        return file_name.to_string();
    }
    let (stem, ext) = match file_name.rsplit_once('.') {
        Some((stem, ext)) if !stem.is_empty() => (stem, format!(".{ext}")),
        _ => (file_name, String::new()),
    };
    (2..)
        .map(|n| format!("{stem}_{n}{ext}"))
        .find(|candidate| !taken(candidate))
        .expect("a free name")
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn entry(value: serde_json::Value) -> ExportEntry {
        serde_json::from_value(value).unwrap()
    }

    fn parts(images: &[&str]) -> Vec<String> {
        let images: Vec<String> = images.iter().map(|s| s.to_string()).collect();
        let names = unique_image_parts(&images);
        images.iter().map(|image| names[image].clone()).collect()
    }

    #[test]
    fn entries_have_their_automatic_names_with_their_name_in_front() {
        assert_eq!(
            entry_names(&entry(json!({"kind": "DATA",
                "outputs": ["OBJECTS", "COLOC_PAIRS", "WELL_GRID", "WELL_LIST", "IMAGE_HEATMAP"]}))),
            ["objects", "coloc_pairs", "well", "well_list", "heatmap"]
        );
        assert_eq!(
            entry_names(&entry(
                json!({"kind": "DATA", "name": "ctrl", "outputs": ["IMAGES"]})
            )),
            ["ctrl_images"]
        );
        assert_eq!(
            entry_names(&entry(json!({"kind": "DRAWN_IMAGES"}))),
            ["drawn_images"]
        );
        assert!(
            entry_names(&entry(
                json!({"kind": "SCRIPT", "language": "R", "file": "scripts/a.R"})
            ))
            .is_empty()
        );
    }

    #[test]
    fn the_prefix_goes_in_front_sanitized() {
        let mut export: ExportSettings =
            serde_json::from_value(json!({"id": 1, "name": "x"})).unwrap();
        assert_eq!(with_prefix(&export, "objects"), "objects");
        export.prefix = "exp 1/a".into();
        assert_eq!(with_prefix(&export, "objects"), "exp_1_a_objects");
    }

    #[test]
    fn a_script_writes_into_its_output_or_its_file_name() {
        let script = |file: &str, output: &str| ScriptExport {
            language: evanalyzer_cfg::settings::export_settings::ScriptLanguage::R,
            file: file.into(),
            output: output.into(),
        };
        assert_eq!(
            script_output(&script("scripts/uptake_stats.R", "")),
            "uptake_stats"
        );
        assert_eq!(script_output(&script("scripts/a.R", "stats")), "stats");
    }

    /// Plates: every well's folder holds the same file names.
    #[test]
    fn images_sharing_a_name_get_their_folder_added() {
        assert_eq!(
            parts(&["A1/img.tif", "B1/img.tif", "C1/other.tif"]),
            ["A1_img", "B1_img", "other"]
        );
    }

    #[test]
    fn folders_are_added_from_the_inside_out_until_unique() {
        assert_eq!(
            parts(&["p1/A1/img.tif", "p2/A1/img.tif", "p2/B1/img.tif"]),
            ["p1_A1_img", "p2_A1_img", "B1_img"]
        );
    }

    /// `A1/img.tif` has to become `A1_img` (beside `B1/img.tif`), which a
    /// file called `A1_img.tif` already is: no folders left to add.
    #[test]
    fn a_clash_no_folder_solves_gets_a_short_code() {
        let names = parts(&["A1/img.tif", "B1/img.tif", "A1_img.tif"]);
        assert_eq!(names[1], "B1_img");
        assert_ne!(names[0], names[2]);
        for name in [&names[0], &names[2]] {
            assert!(
                name.starts_with("A1_img_") && name.len() == "A1_img_".len() + 6,
                "{name}"
            );
        }
        // The same every time and in every order.
        assert_eq!(
            parts(&["A1_img.tif", "A1/img.tif", "B1/img.tif"]),
            [names[2].clone(), names[0].clone(), names[1].clone()]
        );
        // Alone they don't clash.
        assert_eq!(parts(&["A1/img.tif", "A1_img.tif"]), ["img", "A1_img"]);
    }

    #[test]
    fn image_names_keep_inner_dots_and_lose_unsafe_characters() {
        assert_eq!(parts(&["img2.ome.tif"]), ["img2_ome"]);
        assert_eq!(parts(&["sub dir\\a b.tif"]), ["a_b"]);
    }

    #[test]
    fn class_names_that_would_be_equal_get_their_id() {
        let parts = unique_parts(&[(1, "Nuclei".into()), (2, "a/b".into()), (3, "a b".into())]);
        assert_eq!(parts[&1], "Nuclei");
        assert_eq!(parts[&2], "a_b_2");
        assert_eq!(parts[&3], "a_b_3");
    }

    #[test]
    fn the_plane_only_where_there_are_several() {
        assert_eq!(plane_suffix((2, 13), true, true), "_z0002_t0013");
        assert_eq!(plane_suffix((2, 13), false, true), "_t0013");
        assert_eq!(plane_suffix((2, 13), false, false), "");
    }

    #[test]
    fn a_taken_name_gets_the_next_number() {
        let taken = ["per_image.csv", "per_image_2.csv"];
        assert_eq!(
            free_name("per_image.csv", |n| taken.contains(&n)),
            "per_image_3.csv"
        );
        assert_eq!(free_name("new.csv", |n| taken.contains(&n)), "new.csv");
        assert_eq!(free_name("noext", |n| n == "noext"), "noext_2");
    }
}
