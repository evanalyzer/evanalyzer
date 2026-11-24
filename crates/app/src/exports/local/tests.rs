//! Export runs on a seeded results database, the files read back.

use super::*;
use crate::exports::types::ScriptPermission;
use crate::files::LocalFileSystem;
use crate::results::local::LocalResults;
use crate::results::local::test_support::{ObjectSpec, seed_db};
use evanalyzer_cfg::settings::export_settings::ExportSettings;
use serde_json::json;

/// Two wells with an image of the same name each, and one more image; two
/// classes.
fn results(dir: &Path) -> LocalResults {
    let path = dir.join("results.evadb");
    seed_db(
        &path,
        &[
            ObjectSpec::new("A1/img.tif", "Nucleus", 1, 100),
            ObjectSpec::new("A1/img.tif", "Nucleus", 1, 300),
            ObjectSpec::new("A1/img.tif", "Spot", 2, 5),
            ObjectSpec::new("B1/img.tif", "Nucleus", 1, 200),
            ObjectSpec::new("other.tif", "Spot", 2, 7),
        ],
    );
    LocalResults::open(path, LocalFileSystem::default()).expect("the seeded results")
}

fn run_of(dir: &Path, export: serde_json::Value) -> ExportRun {
    ExportRun {
        export: serde_json::from_value::<ExportSettings>(export).unwrap(),
        project_dir: dir.join("project"),
        results_dir: dir.to_path_buf(),
        output: None,
        scripts: ScriptPermission::Skip,
        drawing: None,
    }
}

fn run(results: &LocalResults, run: &ExportRun) -> ExportReport {
    let cancel = AtomicBool::new(false);
    results
        .run_export(run, &cancel, &mut |_, _, _| {})
        .expect("the export runs")
}

fn lines(path: &Path) -> Vec<String> {
    std::fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("{}: {e}", path.display()))
        .lines()
        .map(String::from)
        .collect()
}

#[test]
fn an_object_table_has_the_leading_columns_and_lengths_in_um() {
    let dir = tempfile::tempdir().unwrap();
    let results = results(dir.path());
    let run_ = run_of(
        dir.path(),
        json!({ "id": 1, "name": "Objects", "entries": [
            { "kind": "TABLE", "rows": "OBJECT", "columns": ["area_px", "area_nm2"] }
        ]}),
    );
    let report = run(&results, &run_);

    assert_eq!(report.folder, dir.path());
    assert!(report.all_written(), "{report:?}");
    assert_eq!(report.entries[0].files, [PathBuf::from("objects.csv")]);
    let lines = lines(&dir.path().join("objects.csv"));
    assert_eq!(lines[0], "image,class,object_id,area_px,area_um2");
    assert_eq!(lines.len(), 6, "{lines:?}");
    let first: Vec<&str> = lines[1].split(',').collect();
    assert_eq!(first[0], "A1/img.tif");
    assert_eq!(first[1], "Nucleus");
    assert_eq!(first[3], "100");
    assert_eq!(first[4], "0.0001", "100 nm² in µm²");
}

#[test]
fn image_rows_aggregate_and_count_once() {
    let dir = tempfile::tempdir().unwrap();
    let results = results(dir.path());
    let run_ = run_of(
        dir.path(),
        json!({ "id": 1, "name": "Images", "lengthUnit": "NM", "entries": [
            { "kind": "TABLE", "rows": "IMAGE", "columns": ["count", "area_px"],
              "aggregations": ["AVG", "MAX"], "classes": [1] }
        ]}),
    );
    let report = run(&results, &run_);
    assert!(report.all_written(), "{report:?}");
    let lines = lines(&dir.path().join("images.csv"));
    assert_eq!(lines[0], "image,class,count,area_px_avg,area_px_max");
    assert!(
        lines.contains(&"A1/img.tif,Nucleus,2,200,300".to_string()),
        "{lines:?}"
    );
    assert!(
        lines.contains(&"B1/img.tif,Nucleus,1,200,200".to_string()),
        "{lines:?}"
    );
    assert_eq!(lines.len(), 3, "only the asked-for class: {lines:?}");
}

#[test]
fn a_file_per_image_has_the_shortest_unique_names_and_the_prefix() {
    let dir = tempfile::tempdir().unwrap();
    let results = results(dir.path());
    let run_ = run_of(
        dir.path(),
        json!({ "id": 1, "name": "Split", "prefix": "exp1",
            "output": { "location": "RESULTS_FOLDER", "folder": "tables" },
            "entries": [{ "kind": "TABLE", "rows": "OBJECT", "split": "IMAGE" }] }),
    );
    let report = run(&results, &run_);
    assert_eq!(report.folder, dir.path().join("tables"));
    let mut files = report.entries[0].files.clone();
    files.sort();
    assert_eq!(
        files,
        [
            PathBuf::from("exp1_objects_A1_img.csv"),
            PathBuf::from("exp1_objects_B1_img.csv"),
            PathBuf::from("exp1_objects_other.csv"),
        ]
    );
    let a1 = lines(&dir.path().join("tables/exp1_objects_A1_img.csv"));
    assert_eq!(a1.len(), 4, "three objects of A1: {a1:?}");
    assert!(a1[1..].iter().all(|line| line.starts_with("A1/img.tif,")));
}

#[test]
fn a_file_per_class_and_left_out_images() {
    let dir = tempfile::tempdir().unwrap();
    let results = results(dir.path());
    results.enable_image("other.tif", true).unwrap();
    let run_ = run_of(
        dir.path(),
        json!({ "id": 1, "name": "Classes",
            "entries": [{ "kind": "TABLE", "rows": "OBJECT", "split": "CLASS" }] }),
    );
    let report = run(&results, &run_);
    let mut files = report.entries[0].files.clone();
    files.sort();
    assert_eq!(
        files,
        [
            PathBuf::from("objects_Nucleus.csv"),
            PathBuf::from("objects_Spot.csv")
        ]
    );
    let spots = lines(&dir.path().join("objects_Spot.csv"));
    assert_eq!(
        spots.len(),
        2,
        "the left-out image's spot is not included: {spots:?}"
    );
}

#[test]
fn rename_keeps_the_old_file() {
    let dir = tempfile::tempdir().unwrap();
    let results = results(dir.path());
    let run_ = run_of(
        dir.path(),
        json!({ "id": 1, "name": "Again",
            "output": { "location": "RESULTS_FOLDER", "folder": "", "ifExists": "RENAME" },
            "entries": [{ "kind": "TABLE", "rows": "IMAGE" }] }),
    );
    run(&results, &run_);
    let second = run(&results, &run_);
    assert_eq!(second.entries[0].files, [PathBuf::from("images_2.csv")]);
    assert!(dir.path().join("images.csv").exists());
}

#[test]
fn an_xlsx_workbook_has_a_sheet_per_part_and_the_about_sheet() {
    let dir = tempfile::tempdir().unwrap();
    let results = results(dir.path());
    let run_ = run_of(
        dir.path(),
        json!({ "id": 1, "name": "Workbook", "entries": [
            { "kind": "TABLE", "rows": "OBJECT", "format": "XLSX", "split": "CLASS", "splitInto": "SHEETS" }
        ]}),
    );
    let report = run(&results, &run_);
    assert!(report.all_written(), "{report:?}");
    assert_eq!(report.entries[0].files, [PathBuf::from("objects.xlsx")]);
    let bytes = std::fs::read(dir.path().join("objects.xlsx")).unwrap();
    assert!(bytes.starts_with(b"PK"), "a zip, as every xlsx");
}

#[test]
fn a_fixed_folder_is_below_the_project_and_output_overrides_it() {
    let dir = tempfile::tempdir().unwrap();
    let results = results(dir.path());
    let mut run_ = run_of(
        dir.path(),
        json!({ "id": 1, "name": "Fixed",
            "output": { "location": "FIXED_FOLDER", "folder": "exports" },
            "entries": [{ "kind": "TABLE", "rows": "IMAGE" }] }),
    );
    assert_eq!(
        run(&results, &run_).folder,
        dir.path().join("project/exports")
    );
    run_.output = Some(dir.path().join("elsewhere"));
    let report = run(&results, &run_);
    assert_eq!(report.folder, dir.path().join("elsewhere"));
    assert!(dir.path().join("elsewhere/images.csv").exists());
}

#[test]
fn missing_columns_are_noted_and_a_failing_entry_doesnt_stop_the_others() {
    let dir = tempfile::tempdir().unwrap();
    let results = results(dir.path());
    let run_ = run_of(
        dir.path(),
        json!({ "id": 1, "name": "Mixed", "entries": [
            { "kind": "TABLE", "rows": "OBJECT", "split": "WELL" },
            { "kind": "TABLE", "rows": "IMAGE", "columns": ["count", "measurement:gone"] }
        ]}),
    );
    let report = run(&results, &run_);
    assert!(matches!(report.entries[0].status, EntryStatus::Failed(_)));
    assert_eq!(report.entries[1].status, EntryStatus::Written);
    assert!(report.entries[1].notes[0].contains("measurement:gone"));
}

#[test]
fn an_export_with_errors_doesnt_run() {
    let dir = tempfile::tempdir().unwrap();
    let results = results(dir.path());
    let run_ = run_of(
        dir.path(),
        json!({ "id": 1, "name": "", "entries": [{ "kind": "TABLE", "rows": "IMAGE" }] }),
    );
    let cancel = AtomicBool::new(false);
    let refused = results.run_export(&run_, &cancel, &mut |_, _, _| {});
    assert!(matches!(refused, Err(InternalErrors::InvalidArgument(_))));
}
