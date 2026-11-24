# Exports: project file and table format 1

Status: specification, agreed 2026-11-24. Not built yet.

An **export** is a named recipe kept in the project file: which tables,
images and scripts to produce from a run's results, where to write them and
whether to do it after every run. Whoever gets the project gets the same
files after their run, so an R or Python script written against them keeps
working.

This document is the contract for two audiences:

- **The project file** (`exports` in `ProjectSettings`, project schema 5):
  what the app reads and writes, and must keep reading.
- **The files written** (table format 1): what a script can rely on.

Anything not listed here (how the exporter works inside, the UI) may change
freely.

---

## 1. In the project file

`exports` is a list at the top level of the project file, next to
`resultsViews`. Missing means no exports. Keys are camelCase and values
SCREAMING_SNAKE_CASE, like the rest of the project file.

```json
"exports": [
  {
    "id": 3,
    "name": "Per image for R",
    "prefix": "exp1",
    "lengthUnit": "UM",
    "output": { "location": "RESULTS_FOLDER", "folder": "", "ifExists": "OVERWRITE" },
    "runAfterEveryRun": true,
    "tableFormat": 1,
    "entries": [
      { "kind": "TABLE", "rows": "OBJECT", "format": "CSV", "split": "IMAGE" },
      { "kind": "TABLE", "rows": "IMAGE", "format": "CSV", "name": "per_image",
        "classes": [3, 5], "columns": ["count", "area_nm2", "intensity_mean_gray_ch1"],
        "aggregations": ["AVG", "MEDIAN"] },
      { "kind": "DRAWN_IMAGES", "classes": [3], "style": "OUTLINE" },
      { "kind": "SCRIPT", "language": "R", "file": "scripts/uptake_stats.R", "output": "stats" }
    ]
  }
]
```

### 1.1 Export

| Key | Type | Default | Meaning and rules |
|---|---|---|---|
| `id` | integer | required | Stable while the export exists, unique among the exports. |
| `name` | text | required | Shown in the app; `--export <name>` in the CLI. Not empty, unique ignoring case. |
| `prefix` | text | none | Put in front of every file and folder the export writes: `{prefix}_{name}`. Sanitized like every name part (§2.1). |
| `lengthUnit` | `"UM"` \| `"NM"` | `"UM"` | Unit of lengths, areas and volumes in the tables. Shows in the column names (§3.3). |
| `output` | object | see 1.2 | Where the files go. |
| `runAfterEveryRun` | bool | `false` | Run the export when an analysis run finishes. The run dialog can switch it off for one run. |
| `tableFormat` | integer | `1` | The table format the files are written in (§3). Pinned: a newer app keeps writing this format until the user changes it. |
| `entries` | list | `[]` | What to write, in this order. A script entry sees everything written by the entries before it. |

### 1.2 `output`

| Key | Type | Default | Meaning and rules |
|---|---|---|---|
| `location` | `"RESULTS_FOLDER"` \| `"FIXED_FOLDER"` | `"RESULTS_FOLDER"` | `resultsFolder`: the folder of the results database (`.evadb`) the export runs for. `fixedFolder`: the same folder for every run. |
| `folder` | text | `""` | `resultsFolder`: a subfolder inside it, `""` = the results folder itself. `fixedFolder`: a path relative to the project folder, e.g. `exports`. Absolute paths are allowed but warned about (not portable). Never `..`. `/` as separator. |
| `ifExists` | `"OVERWRITE"` \| `"RENAME"` | `"OVERWRITE"` | Per file: replace it, or keep it and write the new one as `name_2.csv`, `name_3.csv`, … |

### 1.3 Entries

Every entry has a `kind`. Entries that write files have an optional `name`
that replaces their automatic name (§2.2) - needed when two entries of the
same kind would write the same name.

#### `table`

| Key | Type | Default | Meaning and rules |
|---|---|---|---|
| `rows` | see below | required | What a row is. |
| `format` | `"CSV"` \| `"XLSX"` | `"CSV"` | |
| `classes` | list of class ids | all classes | The classes in the table. |
| `columns` | list of column keys | the default columns | The value columns, by the column keys the results views use (`count`, `area_nm2`, `intensity_mean_gray_ch1`, `measurement:<key>`, `children:…`, `parent:…`, `coloc:…`). The keys name the value; the unit in the written file follows `lengthUnit`. |
| `aggregations` | list of `"AVG"`, `"MEDIAN"`, `"STDDEV"`, `"SUM"`, `"MIN"`, `"MAX"`, `"SKEWNESS"` | `["AVG"]` | Grouped rows only (`image`, `well`, `folder`, `track`). |
| `images` | `"ALL"` \| `"INCLUDED"` \| `"LEFT_OUT"` | `"INCLUDED"` | Which images of the run. |
| `emptyClasses` | bool | `false` | Grouped rows: a row also for classes without objects. |
| `split` | `"NONE"` \| `"IMAGE"` \| `"WELL"` \| `"FOLDER"` \| `"CLASS"` | `"NONE"` | One file (or sheet) per image, well, folder or class. |
| `splitInto` | `"FILES"` \| `"SHEETS"` | `"FILES"` | `sheets`: one workbook, a sheet each. `xlsx` only. |
| `tracks` | object | see below | `rows: "TRACK"` only. |

`rows`:

| Value | A row is |
|---|---|
| `object` | an object |
| `image` | an image and class |
| `well` | a well and class |
| `folder` | an image folder and class |
| `colocPair` | a pair of overlapping objects (Colocalize Objects' partners) |
| `track` | a track |
| `measurement` | a measurement value (image, plane, object or region, step, measurement, value, unit) |

`tracks`: `minPoints` (integer, default `1`), `fromStart` (bool, `false`),
`noEdge` (bool, `false`), `speedUnit` (`"PX_FRAME"` \| `"UM_MIN"` \| `"UM_S"`,
default `"UM_MIN"`).

#### `plateLayout`

The values laid out as the plate (or well) grid, for Excel.

| Key | Type | Default | Meaning and rules |
|---|---|---|---|
| `grid` | `"PLATE"` \| `"WELL"` | `"PLATE"` | Plate: a cell per well. Well: a cell per image position inside the wells. |
| `classes` | list of class ids | all classes | |
| `columns` | list of column keys | `["count"]` | |
| `aggregation` | one aggregation (as above) | `"AVG"` | |
| `images` | as in `table` | `"INCLUDED"` | |

Always `xlsx`, a sheet per class and value.

#### `drawnImages`

A PNG per image (and plane) with the objects drawn on it.

| Key | Type | Default | Meaning and rules |
|---|---|---|---|
| `classes` | list of class ids | all classes | The classes drawn. |
| `style` | `"OUTLINE"` \| `"FILLED"` \| `"BOX"` | `"OUTLINE"` | |
| `brightness` | `"VIEWER"` \| `"AUTO"` | `"VIEWER"` | Each channel's display range: as set in the viewer, or automatic per image. |
| `maxSide` | integer (px) | `2048` | The longer side at most. |
| `images` | as in `table` | `"INCLUDED"` | |

Channel colours and display settings are the project's when the export runs.

#### `script`

| Key | Type | Default | Meaning and rules |
|---|---|---|---|
| `language` | `"R"` \| `"PYTHON"` | required | Which interpreter. Where it is installed is a setting of the computer (or server), never of the project. |
| `file` | text | required | Path relative to the project folder, inside `scripts/`. |
| `output` | text | the file name without extension | Subfolder of the export's output folder the script writes into. |

The call is fixed:

```
<interpreter> <file> --input <output folder> --output <output folder>/<output>
```

`<output folder>` is the export's output folder (§1.2) - where the tables
of this export were just written. The script's working folder is the
project folder. Nothing else is passed.

**Trust.** A script from a project someone else made never runs without the
user's OK. The OK is kept per script content (a hash) in the user settings,
never in the project - a project can't declare itself trusted. A changed
script asks again. Tables and images are written either way.

**Remote.** With `--remote` the export runs on the server, next to the
results. Scripts run there only if the server's administrator allows it in
the server configuration, with the interpreters' paths there; off by
default. The user's OKs are kept in their settings on the server.

### 1.4 Checks (`project validate`, and when the project is loaded)

Errors:

- `id` or `name` missing, duplicate (`name` ignoring case) or empty.
- An unknown `kind`, `rows`, `format`, `split`, `location`, … value.
- `folder` with `..`; a `name`, `prefix` or script `output` that isn't a
  single safe name.
- `aggregations` with non-grouped `rows`; `splitInto: "SHEETS"` with `"CSV"`;
  `tracks` with `rows` other than `track`.
- Two entries of one export, or two exports writing into the same folder,
  that would write the same file name.
- A script outside `scripts/`, or missing.

Warnings:

- An absolute `folder` (not portable).
- Class ids or column keys the project doesn't have (a class may be absent
  from some runs; the table then has no such column or rows).

### 1.5 Command line

`evanalyzer cli export` works with a project's exports:

| Command | Does |
|---|---|
| `export list --project <file.evaproj>` | Lists the exports: id, name, entries, output, run after every run. |
| `export view <id or name> --project <file.evaproj>` | Shows one export's settings. |
| `export run <id or name> --project <file.evaproj> [--results <.evadb or its folder>] [--output <folder>] [--allow-scripts \| --no-scripts]` | Runs the export on a run's results - the newest run of the project without `--results`. `--output` writes into that folder for this once instead of the export's `output`. A script someone else wrote needs `--allow-scripts` (there is nobody to ask); `--no-scripts` leaves scripts out. |

`analyze` runs the exports with `runAfterEveryRun` when it is done;
`--export <id or name>` (repeatable) runs those instead, `--no-exports`
none.

### 1.6 Versions

Exports come with project schema 5. The step from 4 to 5 changes nothing
in the file; it exists so that an app that doesn't know exports refuses the
file instead of saving it back without them.

Later additions (new `rows` values such as time course or profiles, new
entry kinds, new optional keys) are added with defaults, so older files keep
their meaning. Renaming or removing a key or value needs a schema version
and a migration step.

---

## 2. Files and folders written

### 2.1 Name parts

Every part of a file name made from user text (prefix, class name, image
name, entry name) is sanitized: letters, digits, `-` and `_` stay, anything
else becomes `_`.

### 2.2 Automatic names

| Entry | Name |
|---|---|
| `table` | by `rows`: `objects`, `images`, `wells`, `folders`, `coloc_pairs`, `tracks`, `measurements` |
| `plateLayout` | `plate` or `well` |
| `drawnImages` | `images` |
| `script` | (its `output` folder) |

With a `prefix`: `{prefix}_{name}`. An entry's own `name` replaces the
automatic one. The extension follows the format: `.csv`, `.xlsx`, `.png`.

### 2.3 One file per image, well, folder or class

Flat, in the output folder: `{name}_{part}.{ext}`, e.g.
`exp1_objects_A1_img.csv`.

- **Image** - the shortest unique name: the image's file name without its
  extension. Where two images of the run share it, their folders are added
  from the inside out until each name is unique (`A1/img.tif`, `B1/img.tif`
  → `A1_img`, `B1_img`; `C1/other.tif` → `other`). If that still clashes
  (e.g. `A1/img.tif` and a file `A1_img.tif`), 6 characters derived from the
  image's relative path are added. The names depend only on which images
  are in the run, not on their order. Every per-image file also has the
  `image` column with the full relative path.
- **Well** - the well, e.g. `A01`.
- **Folder** - the image folder, sanitized; `_` for the top folder.
- **Class** - the class name, sanitized; the class id is added only where
  two classes would get the same name.
- **Planes** - where the results have several z-planes or time points,
  `_z0002` and `_t0013` (4 digits, counted from 0) are added.

`drawnImages` always writes one file per image (and plane), named the same
way: `{name}_{image}.png`.

With `splitInto: "SHEETS"` the workbook is `{name}.xlsx` and the sheets are
named by the same parts, shortened and made unique to Excel's limits.

---

## 3. Table format 1

What a script can rely on. Written by every `table` entry with
`tableFormat: 1`. There is no separate metadata file: the column names say
what a column is and in which unit.

### 3.1 CSV

UTF-8 without BOM, `,` as separator, `.` as decimal point, `"` quoting
where needed, `\n` line ends, one header line. Empty cell = no value (a
class without objects, a value not measured). Numbers without thousands
separators.

### 3.2 Columns, in this order

1. Where the row belongs: `image` (relative path; not in `well` and
   `folder` rows), `well` and `folder` where the rows are grouped by them,
   `class` (the class name; left out in a file split by class), `object_id`
   for objects, `track_id` for tracks.
2. `t` and `z` (counted from 0), only where the results have several.
3. The value columns, in the format's fixed order of column types (not the
   order they were ticked in).

### 3.3 Column names

Made for scripts: lower case, only `a-z`, `0-9` and `_`.

- The column key with every other character turned into `_`:
  `measurement:threshold_dapi_nucleus` → `measurement_threshold_dapi_nucleus`.
- Lengths, areas and volumes carry the unit chosen by `lengthUnit`:
  `area_um2`, `perimeter_um`, `volume_um3`, `feret_diameter_um`
  (`…_nm2`, `…_nm` with `"NM"`). Pixel values end in `_px`, intensities in
  `_gray`.
- Grouped rows add the aggregation: `area_um2_avg`, `area_um2_median`.
  The count of objects is `count`.
- Measurement columns have no unit in their name (it isn't stored per
  column); a `rows: "MEASUREMENT"` table has the unit as a column.

### 3.4 XLSX

The same table on the first sheet (or a sheet per split part), plus a sheet
`About` in the same file: each column's name, label as shown in the app,
and unit.

### 3.5 Stability

Within table format 1 the column names, their order rules and the file
names above don't change. New columns appear only for values that are new
to the app (a new measurement type) and only where an export asks for them;
existing columns keep their names and order - a script reading columns by
name keeps working. Anything else becomes table format 2, which an export
only uses when the user switches it.
