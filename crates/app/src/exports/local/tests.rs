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
        interpreters: Default::default(),
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

/// Transposed image rows: a row per image, the classes side by side - the
/// count once per class, empty where an image has none of a class.
#[test]
fn transposed_image_rows_have_a_column_per_value_and_class() {
    let dir = tempfile::tempdir().unwrap();
    let results = results(dir.path());
    let run_ = run_of(
        dir.path(),
        json!({ "id": 1, "name": "Images", "entries": [
            { "kind": "TABLE", "rows": "IMAGE", "columns": ["count", "area_px"],
              "aggregations": ["AVG", "MAX"], "classes": [1, 2], "transposed": true,
              "format": "XLSX" },
            { "kind": "TABLE", "rows": "IMAGE", "columns": ["count", "area_px"],
              "aggregations": ["AVG", "MAX"], "classes": [1, 2], "transposed": true,
              "name": "wide" }
        ]}),
    );
    let report = run(&results, &run_);
    assert!(report.all_written(), "{report:?}");
    let lines = lines(&dir.path().join("wide.csv"));
    assert_eq!(
        lines[0],
        "image,count_nucleus,area_px_avg_nucleus,area_px_max_nucleus,\
         count_spot,area_px_avg_spot,area_px_max_spot"
    );
    assert_eq!(lines.len(), 4, "a row per image: {lines:?}");
    assert!(
        lines.contains(&"A1/img.tif,2,200,300,1,5,5".to_string()),
        "{lines:?}"
    );
    assert!(
        lines.contains(&"B1/img.tif,1,200,200,0,,".to_string()),
        "{lines:?}"
    );
    assert!(
        lines.contains(&"other.tif,0,,,1,7,7".to_string()),
        "{lines:?}"
    );
}

/// Transposed object rows: per image the n-th object of each class side by
/// side.
#[test]
fn transposed_object_rows_put_the_classes_side_by_side() {
    let dir = tempfile::tempdir().unwrap();
    let results = results(dir.path());
    let run_ = run_of(
        dir.path(),
        json!({ "id": 1, "name": "Objects", "entries": [
            { "kind": "TABLE", "rows": "OBJECT", "columns": ["area_px"],
              "classes": [1, 2], "transposed": true }
        ]}),
    );
    let report = run(&results, &run_);
    assert!(report.all_written(), "{report:?}");
    let lines = lines(&dir.path().join("objects.csv"));
    assert_eq!(
        lines[0],
        "image,object_id_nucleus,area_px_nucleus,object_id_spot,area_px_spot"
    );
    // A1: two nuclei and a spot - two rows; B1 and other one each.
    assert_eq!(lines.len(), 5, "{lines:?}");
    let a1: Vec<Vec<&str>> = lines
        .iter()
        .filter(|l| l.starts_with("A1/"))
        .map(|l| l.split(',').collect())
        .collect();
    assert_eq!(a1.len(), 2);
    assert_eq!(a1[0][4], "5");
    assert_eq!(a1[1][3], "", "no second spot");
}

/// Transposed well rows can't be: refused by the checks.
#[test]
fn transposed_well_rows_are_refused() {
    let problems = crate::exports::check::check_export(
        &serde_json::from_value(json!({ "id": 1, "name": "W", "entries": [
            { "kind": "TABLE", "rows": "WELL", "transposed": true }
        ]}))
        .unwrap(),
    );
    assert!(
        problems
            .iter()
            .any(|p| p.severity == crate::modules::Severity::Error
                && p.message.contains("transposed")),
        "{problems:?}"
    );
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
    // A folder where the objects' file should go: that entry can't write.
    std::fs::create_dir_all(dir.path().join("objects.csv")).unwrap();
    let run_ = run_of(
        dir.path(),
        json!({ "id": 1, "name": "Mixed", "entries": [
            { "kind": "TABLE", "rows": "OBJECT" },
            { "kind": "TABLE", "rows": "IMAGE", "columns": ["count", "measurement:gone"] }
        ]}),
    );
    let report = run(&results, &run_);
    assert!(
        matches!(report.entries[0].status, EntryStatus::Failed(_)),
        "{report:?}"
    );
    assert_eq!(report.entries[1].status, EntryStatus::Written);
    assert!(report.entries[1].notes[0].contains("measurement:gone"));
}

/// The wells as the results group the images (the well pattern on the
/// image names); images it doesn't fit go into `_`.
#[test]
fn a_file_per_well() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("results.evadb");
    seed_db(
        &path,
        &[
            ObjectSpec::new("A01_1.tif", "Cell", 1, 1),
            ObjectSpec::new("A01_2.tif", "Cell", 1, 2),
            ObjectSpec::new("B03_1.tif", "Cell", 1, 3),
        ],
    );
    let results = LocalResults::open(path, LocalFileSystem::default()).unwrap();
    let run_ = run_of(
        dir.path(),
        json!({ "id": 1, "name": "Wells", "entries": [
            { "kind": "TABLE", "rows": "OBJECT", "split": "WELL" }
        ]}),
    );
    let report = run(&results, &run_);
    assert!(report.all_written(), "{report:?}");
    let mut files = report.entries[0].files.clone();
    files.sort();
    assert_eq!(
        files,
        [
            PathBuf::from("objects_A01.csv"),
            PathBuf::from("objects_B03.csv")
        ]
    );
    assert_eq!(lines(&dir.path().join("objects_A01.csv")).len(), 3);
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

/// A track over three time points: where, when, its leading columns.
#[test]
fn a_track_table_has_a_row_per_track() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("results.evadb");
    seed_db(
        &path,
        &[
            ObjectSpec::new("A1/img.tif", "Cell", 1, 100)
                .in_track(7)
                .at_plane(0, 0),
            ObjectSpec::new("A1/img.tif", "Cell", 1, 120)
                .in_track(7)
                .at_plane(0, 1),
            ObjectSpec::new("A1/img.tif", "Cell", 1, 140)
                .in_track(7)
                .at_plane(0, 2),
            ObjectSpec::new("B1/img.tif", "Cell", 1, 50)
                .in_track(3)
                .at_plane(0, 1),
        ],
    );
    let results = LocalResults::open(path, LocalFileSystem::default()).unwrap();
    let run_ = run_of(
        dir.path(),
        json!({ "id": 1, "name": "Tracks", "entries": [
            { "kind": "TABLE", "rows": "TRACK", "columns": ["area_px"], "aggregations": ["AVG", "MAX"] }
        ]}),
    );
    let report = run(&results, &run_);
    assert!(report.all_written(), "{report:?}");
    let lines = lines(&dir.path().join("tracks.csv"));
    let header: Vec<&str> = lines[0].split(',').collect();
    assert_eq!(&header[..4], ["image", "well", "class", "track_id"]);
    for column in [
        "first_t",
        "last_t",
        "time_points",
        "touches_edge",
        "from_start",
        "area_px_avg",
        "area_px_max",
    ] {
        assert!(header.contains(&column), "{column} in {header:?}");
    }
    let at = |name: &str| header.iter().position(|h| *h == name).unwrap();
    let track = lines
        .iter()
        .map(|line| line.split(',').collect::<Vec<_>>())
        .find(|cells| cells[at("track_id")] == "7")
        .expect("track 7");
    assert_eq!(track[at("image")], "A1/img.tif");
    assert_eq!(track[at("first_t")], "0");
    assert_eq!(track[at("last_t")], "2");
    assert_eq!(track[at("time_points")], "3");
    assert_eq!(track[at("area_px_avg")], "120");
    assert_eq!(track[at("area_px_max")], "140");
    assert_eq!(lines.len(), 3, "two tracks: {lines:?}");
}

/// The measurement values one by one, only the asked-for keys, the unit as
/// a column.
#[test]
fn a_measurement_table_has_a_row_per_value() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("results.evadb");
    seed_db(&path, &[ObjectSpec::new("A1/img.tif", "Cell", 1, 100)]);
    {
        let conn = duckdb::Connection::open(&path).unwrap();
        conn.execute_batch(
            "CREATE TABLE measurements (image_name VARCHAR, image_rel_path VARCHAR, \
             z_stack INTEGER, t_stack INTEGER, region VARCHAR, object_id UUID, \
             related_object_id UUID, object_class_name VARCHAR, object_class_id INTEGER[], \
             step VARCHAR, measurement VARCHAR, unit VARCHAR, key VARCHAR, value DOUBLE);
             INSERT INTO measurements VALUES
               ('img.tif', 'A1/img.tif', 0, 0, 'image', NULL, NULL, NULL, NULL, 'threshold_01', 'Threshold', 'gray value', 'threshold_01/threshold', 42.0),
               ('img.tif', 'A1/img.tif', 0, 0, 'image', NULL, NULL, NULL, NULL, 'noise_01', 'Noise', '', 'noise_01/noise', 3.5);",
        )
        .unwrap();
    }
    let results = LocalResults::open(path, LocalFileSystem::default()).unwrap();
    let run_ = run_of(
        dir.path(),
        json!({ "id": 1, "name": "Measurements", "entries": [
            { "kind": "TABLE", "rows": "MEASUREMENT", "columns": ["measurement:threshold_01/threshold", "count"] }
        ]}),
    );
    let report = run(&results, &run_);
    assert!(report.all_written(), "{report:?}");
    assert!(
        report.entries[0].notes[0].contains("count"),
        "{:?}",
        report.entries[0].notes
    );
    let lines = lines(&dir.path().join("measurements.csv"));
    assert_eq!(
        lines[0],
        "image,region,object_id,related_object_id,class,step,measurement,key,value,unit"
    );
    assert_eq!(
        lines[1..],
        ["A1/img.tif,image,,,,threshold_01,Threshold,threshold_01/threshold,42,gray value"]
    );
}

/// The plate grid of the results' export, under the entry's name.
#[test]
fn a_plate_layout_is_a_workbook_under_its_name() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("results.evadb");
    seed_db(
        &path,
        &[
            ObjectSpec::new("A01_1.tif", "Cell", 1, 1),
            ObjectSpec::new("B03_1.tif", "Cell", 1, 3),
        ],
    );
    let results = LocalResults::open(path, LocalFileSystem::default()).unwrap();
    let run_ = run_of(
        dir.path(),
        json!({ "id": 1, "name": "Plate", "prefix": "p",
            "entries": [{ "kind": "PLATE_LAYOUT" }] }),
    );
    let report = run(&results, &run_);
    assert!(report.all_written(), "{report:?}");
    assert_eq!(report.entries[0].files, [PathBuf::from("p_plate.xlsx")]);
    assert!(dir.path().join("p_plate.xlsx").exists());
    let left: Vec<String> = std::fs::read_dir(dir.path())
        .unwrap()
        .flatten()
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .filter(|name| name.starts_with(".evanalyzer-export"))
        .collect();
    assert!(left.is_empty(), "the scratch folder is gone: {left:?}");
}

/// Without the project's image settings (no project) drawing is skipped;
/// with them a PNG per image, flat, by its shortest unique name.
#[test]
fn drawn_images_are_flat_pngs_by_image_name() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("results.evadb");
    let image = "multi-channel-4D-series.ome.tif";
    seed_db(&path, &[ObjectSpec::new(image, "Cell", 1, 50)]);
    let results = LocalResults::open(path, LocalFileSystem::default()).unwrap();
    let mut run_ = run_of(
        dir.path(),
        json!({ "id": 1, "name": "Drawn", "entries": [{ "kind": "DRAWN_IMAGES", "maxSide": 64 }] }),
    );
    let report = run(&results, &run_);
    assert!(matches!(report.entries[0].status, EntryStatus::Skipped(_)));

    run_.drawing = Some(crate::results::DrawnImages {
        image_root: Path::new(env!("CARGO_MANIFEST_DIR")).join("../core/tests"),
        max_side: 2048,
        ..Default::default()
    });
    let report = run(&results, &run_);
    assert!(report.all_written(), "{report:?}");
    assert_eq!(
        report.entries[0].files,
        [PathBuf::from(
            "drawn_images_multi-channel-4D-series_ome.png"
        )]
    );
    let png = std::fs::read(
        dir.path()
            .join("drawn_images_multi-channel-4D-series_ome.png"),
    )
    .unwrap();
    assert!(png.starts_with(b"\x89PNG"));
}

/// Python as the tests' interpreter - `None` (the test is skipped) where
/// there is none.
fn python() -> Option<&'static str> {
    let name = if cfg!(windows) { "python" } else { "python3" };
    std::process::Command::new(name)
        .arg("--version")
        .output()
        .ok()
        .filter(|out| out.status.success())
        .map(|_| name)
}

/// A project folder with `scripts/<name>`, and a run of an image table
/// then that script.
fn script_run(dir: &Path, name: &str, source: &str, scripts: ScriptPermission) -> ExportRun {
    let project = dir.join("project");
    std::fs::create_dir_all(project.join("scripts")).unwrap();
    std::fs::write(project.join("scripts").join(name), source).unwrap();
    let mut run_ = run_of(
        dir,
        json!({ "id": 1, "name": "With script", "entries": [
            { "kind": "TABLE", "rows": "IMAGE" },
            { "kind": "SCRIPT", "language": "PYTHON", "file": format!("scripts/{name}") }
        ]}),
    );
    run_.scripts = scripts;
    run_
}

const COUNT_LINES: &str = r#"
import sys, os
args = dict(zip(sys.argv[1::2], sys.argv[2::2]))
with open(os.path.join(args["--input"], "images.csv")) as f:
    n = len(f.read().splitlines()) - 1
os.makedirs(args["--output"], exist_ok=True)
with open(os.path.join(args["--output"], "n.txt"), "w") as f:
    f.write(str(n))
print("rows:", n, "cwd:", os.path.basename(os.getcwd()))
"#;

/// Allowed, the script runs after the table, on its folder: its files and
/// its output come back; it runs in the project's folder.
#[test]
fn an_allowed_script_runs_on_the_tables_before_it() {
    if python().is_none() {
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let results = results(dir.path());
    let run_ = script_run(dir.path(), "count.py", COUNT_LINES, ScriptPermission::All);
    let report = run(&results, &run_);
    assert!(report.all_written(), "{report:?}");
    assert_eq!(report.entries[1].files, [PathBuf::from("count/n.txt")]);
    assert_eq!(
        std::fs::read_to_string(dir.path().join("count/n.txt")).unwrap(),
        "4"
    );
    assert!(
        report.entries[1].log.contains("rows: 4 cwd: project"),
        "{}",
        report.entries[1].log
    );
}

/// The OK is per content: the hash allows it, a changed script asks again;
/// not allowed or scripts off, it is skipped - the table is still written.
#[test]
fn a_script_runs_only_with_the_ok_for_its_content() {
    if python().is_none() {
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let results = results(dir.path());
    let hash = crate::exports::script_hash(COUNT_LINES.as_bytes());

    let trusted = script_run(
        dir.path(),
        "count.py",
        COUNT_LINES,
        ScriptPermission::Trusted(vec![hash.clone()]),
    );
    assert!(run(&results, &trusted).all_written());

    let changed = script_run(
        dir.path(),
        "count.py",
        &format!("{COUNT_LINES}\n# changed"),
        ScriptPermission::Trusted(vec![hash]),
    );
    let report = run(&results, &changed);
    assert_eq!(report.entries[0].status, EntryStatus::Written);
    assert!(matches!(report.entries[1].status, EntryStatus::Skipped(_)));
    assert!(report.entries[1].notes[0].starts_with("script hash "));

    let off = script_run(dir.path(), "count.py", COUNT_LINES, ScriptPermission::Skip);
    assert!(matches!(
        run(&results, &off).entries[1].status,
        EntryStatus::Skipped(_)
    ));
}

/// A script that fails says its exit code and keeps its output; an
/// interpreter that isn't there says so.
#[test]
fn a_failing_script_or_a_missing_interpreter_fail_the_entry() {
    if python().is_none() {
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let results = results(dir.path());
    let failing = script_run(
        dir.path(),
        "fail.py",
        "import sys\nprint('oops', file=sys.stderr)\nsys.exit(3)\n",
        ScriptPermission::All,
    );
    let report = run(&results, &failing);
    assert_eq!(
        report.entries[1].status,
        EntryStatus::Failed("scripts/fail.py stopped with exit code 3".into())
    );
    assert!(report.entries[1].log.contains("oops"));

    let mut missing = script_run(dir.path(), "count.py", COUNT_LINES, ScriptPermission::All);
    missing.interpreters.python = Some(dir.path().join("no-such-python"));
    let report = run(&results, &missing);
    let EntryStatus::Failed(why) = &report.entries[1].status else {
        panic!("{report:?}");
    };
    assert!(why.contains("Could not start"), "{why}");
}
