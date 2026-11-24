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
      { "kind": "DATA", "outputs": ["OBJECTS", "IMAGES", "PLATE"],
        "format": "CSV", "classes": [3, 5],
        "columns": ["count", "area_nm2", "intensity_mean_gray_ch1"],
        "aggregations": ["AVG", "MEDIAN"], "heatmap": { "colors": "VIRIDIS" } },
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

#### `data`

Data of the results - tables and heatmaps - with **one filter**: each
output in `outputs` is written with the same columns, values, classes,
planes and images. An output that can't take one of them leaves it out
and says so once in the run's report (never an error). Different filters
for different outputs: a second `data` entry.

| Key | Type | Default | Meaning and rules |
|---|---|---|---|
| `outputs` | list, see below | none | What it writes - at least one, each once. |
| `name` | text | none | A name part in front of the automatic file names (`ctrl` → `ctrl_objects.csv`). Two data entries with the same outputs need different names. |
| `format` | `"CSV"` \| `"XLSX"` | `"CSV"` | The tables' format; heatmaps are always `xlsx`. |
| `classes` | list of class ids | all classes | |
| `columns` | list of column keys | each output's default columns | The value columns, by the column keys the results views use (`count`, `area_nm2`, `intensity_mean_gray_ch1`, `measurement:<key>`, `coloc:…`). The keys name the value; the unit in the written file follows `lengthUnit`. Measurements take only measurement columns, the time course only values (not the count). |
| `aggregations` | list of `"AVG"`, `"MEDIAN"`, `"STDDEV"`, `"SUM"`, `"MIN"`, `"MAX"`, `"SKEWNESS"` | `["AVG"]` | Where rows group objects: per image, well, folder, track, time point, heatmap cell (a sheet each). The time course has no skewness. |
| `images` | `"ALL"` \| `"INCLUDED"` \| `"LEFT_OUT"` | `"INCLUDED"` | Which images of the run. |
| `planes` | `{ "z": [from, to], "t": [from, to] }` | all | Which z-planes and time points (both ends included, from 0). Not for tracks and profiles (they span them). |
| `emptyClasses` | bool | `false` | Per image, well, folder: a row also for classes without objects. |
| `transposed` | bool | `false` | Objects, images, wells and folders: the classes side by side (§3.2). |
| `children` | list of `[parent class id, child class id]` | none | The children linked to each object (Relate Objects), as the Results List shows them: per link their count and each `childAggregations` value of each column. Objects, per image, well, folder, coloc pairs. |
| `childAggregations` | list of aggregations | none | How the children's values are summed up, besides their count. |
| `parents` | list of class ids | none | The linked objects (parents): their id and each column. |
| `split` | `"NONE"` \| `"IMAGE"` \| `"WELL"` \| `"FOLDER"` \| `"CLASS"` | `"NONE"` | One file (or sheet) per image, well, folder or class - for the tables. |
| `splitInto` | `"FILES"` \| `"SHEETS"` | `"FILES"` | `sheets`: one workbook, a sheet each. `xlsx` only. |
| `tracks` | object | see below | The `TRACKS` output. |
| `timeCourse` | object | see below | The `TIME_COURSE` output. |
| `profile` | object | see below | The `PROFILE` output (needed there). |
| `heatmap` | object | see below | The heatmap outputs. |

`outputs` (the old export dialog's ticks):

| Value | Writes |
|---|---|
| `OBJECTS` | `objects`: a row per object |
| `IMAGES` | `images`: a row per image and class |
| `WELLS` | `wells`: a row per well and class (the plate as a list) |
| `FOLDERS` | `folders`: a row per image folder and class |
| `COLOC_PAIRS` | `coloc_pairs`: a row per pair of overlapping objects (Colocalize Objects' partners) |
| `MEASUREMENTS` | `measurements`: a row per measurement value (image, plane, object or region, step, measurement, value, unit) |
| `TRACKS` | `tracks`: a row per track |
| `TIME_COURSE` | `time_course`: a row per time point and line, its values summed up over the line's objects there |
| `PROFILE` | `profile`: a row per position of a profile (e.g. a Sholl ring's radius), of a line or of an object |
| `PLATE` | `plate.xlsx`: the plate as a heatmap, a cell per well |
| `WELL_GRID` | `well.xlsx`: the wells as a heatmap, a cell per image position |
| `WELL_LIST` | `well_list.xlsx`: the wells as a list, a row per image position |
| `IMAGE_HEATMAP` | `heatmap_<image>.xlsx`: a workbook per image, its area as a grid of squares |

`tracks`: `minPoints` (integer, default `1`), `fromStart` (bool, `false`),
`noEdge` (bool, `false`), `speedUnit` (`"PX_FRAME"` \| `"UM_MIN"` \| `"UM_S"`,
default `"UM_MIN"`).

`timeCourse`: `lines` (`"CLASS"` \| `"IMAGE"` \| `"WELL"` \| `"TRACK"`,
default `"CLASS"`), `layout` (`"LONG"` \| `"WIDE"` \| `"BOTH"`, default
`"LONG"`; wide: a table per value - a row per time point, a column per line
- in `<name>_wide.xlsx` or `<name>_wide_<value>.csv`; not per track, not
split), `relative` (bool, `false`: the values in % of the line's first time
point with objects), `withParents` (bool, `false`: track lines with their
parents' history).

`profile`: `key` (the profile measurement, e.g.
`"sholl_01/intersections"`), `lines` (`"CLASS"` \| `"IMAGE"` \| `"WELL"` \|
`"ALL"`, default `"CLASS"`), `content` (`"LINES"` \| `"OBJECTS"`, default
`"LINES"`: per line and position n, mean, SD, median, Q1, Q3 - an object
without a value at a position counts 0; objects: each object's value per
position).

`heatmap`: `colors` (`"EXCEL"` \| `"VIRIDIS"` \| `"PLASMA"` \| `"INFERNO"` \|
`"CIVIDIS"` \| `"COOLWARM"` \| `"RED_BLUE"` \| `"YL_GN_BU"` \| `"HALINE"` \|
`"ALGAE"` \| `"THERMAL"`, default `"EXCEL"`), `squareSize` (the image
heatmap's square side in px, default `64`). Heatmaps write a sheet per
class, column and aggregation; without `columns` the count.

#### `drawnImages`

A PNG per image (and plane) with the objects drawn on it.

| Key | Type | Default | Meaning and rules |
|---|---|---|---|
| `classes` | list of class ids | all classes | The classes drawn. |
| `style` | `"OUTLINE"` \| `"FILLED"` \| `"BOX"` | `"OUTLINE"` | |
| `brightness` | `"VIEWER"` \| `"AUTO"` | `"VIEWER"` | Each channel's display range: as set in the viewer, or automatic per image. |
| `maxSide` | integer (px) | `2048` | The longer side at most. |
| `images` | as in `data` | `"INCLUDED"` | |
| `planes` | as in `data` | all | |

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
- An unknown `kind`, output, `format`, `split`, `location`, … value.
- `folder` with `..`; a `name`, `prefix` or script `output` that isn't a
  single safe name.
- A `data` entry without outputs, or with an output twice. A problem of
  several outputs is said once, naming them.
- Outputs that sum values up without any `aggregations`; `splitInto:
  "SHEETS"` with `"CSV"`; a `split` finer than an output's rows
  (`"IMAGE"` for wells or folders, `"WELL"` for folders and back,
  `"CLASS"` for measurements).
- `PROFILE` without `profile.key`; a wide time course per track or split;
  a plane range whose `from` is after its `to`.
- `classes` given but empty (it would write nothing; left out it means
  all classes).
- Two entries of one export, or two exports writing into the same folder,
  that would write the same file name.
- A script outside `scripts/`, or missing.

Warnings:

- An absolute `folder` (not portable).
- `transposed` with `split: "CLASS"` (each file has a single class).
- Settings an output can't use are left out for it and said in the
  report - never an error, so ticking one more output never breaks an
  export.
- Class ids or column keys the project doesn't have (a class may be absent
  from some runs; the table then has no such column or rows).

### 1.5 Command line

`evanalyzer cli export` works with a project's exports:

| Command | Does |
|---|---|
| `export list --project <file.evaproj>` | Lists the exports: id, name, entries, output, run after every run. |
| `export view <id or name> --project <file.evaproj>` | Shows one export's settings. |
| `export run <id or name> --project <file.evaproj> [--results <.evadb or its folder>] [--output <folder>] [--allow-scripts \| --no-scripts]` | Runs the export on a run's results - the newest run of the project without `--results`. `--output` writes into that folder for this once instead of the export's `output`. A script someone else wrote needs `--allow-scripts` (there is nobody to ask); `--no-scripts` leaves scripts out. |

`analyze` runs the exports with `runAfterEveryRun` when it is done, into
its results folder; `--export <id or name>` (repeatable) runs those
instead, `--no-exports` none, `--allow-scripts` lets their scripts run
without asking. The analysis is kept even where an export fails; the
command then fails, naming the export.

### 1.6 In the app

- **Exports tab** (Ctrl+4): the project's exports, the one picked and the
  settings of its entry; Run runs it on the project's newest analysis.
- **After a run**: the run dialog lists the exports with
  `runAfterEveryRun` under "Afterwards"; one switched off there stays off
  for that run only. They run on this computer when the analysis is done,
  into its results folder (also in remote mode: the results stay on the
  server, the export runs where they are). A note says what they wrote;
  "Open in Exports" shows an export's report.
- **Scripts**: before an export runs a script the user hasn't allowed yet,
  the app shows it and asks: don't run it, run it once, or always allow
  this version (its content's SHA-256, kept in the user's settings - a
  changed file asks again). Its tables and images are written either way.
- **Results › Export…** on any view: "Export this view" writes the view as
  it is shown into a folder, once: a List as a table, a Plate as a plate
  layout, Tracks and Paths as a table of tracks, a chart as the table behind
  it (over time: a time course; a profile; a histogram, scatter or boxplot:
  its values), Trees as the measurements - a file of that name stays, the new one gets
  `_2`. Ticked, it is also kept in the project: as a new export or added
  to one there is. A view's ⋯ › "Save as export…" keeps it without writing,
  "Add to export…" opens the dialog with the tick on. A chart's picture is
  Charts › "Save chart…".

### 1.7 Versions

Exports come with project schema 5. The step from 4 to 5 changes nothing
in the file; it exists so that an app that doesn't know exports refuses the
file instead of saving it back without them.

Later additions (new outputs, new entry kinds, new optional keys) are added with defaults, so older files keep
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
| `data` | per output: `objects`, `images`, `wells`, `folders`, `coloc_pairs`, `measurements`, `tracks`, `time_course`, `profile`, `plate`, `well`, `well_list`, `heatmap_<image>` |
| `drawnImages` | `drawn_images` |
| `script` | (its `output` folder) |

With a `prefix`: `{prefix}_{name}`. A `data` entry's `name` goes in front
of its automatic names (`{prefix}_{name}_objects`); drawn images' `name`
replaces theirs. The extension follows the format: `.csv`, `.xlsx`, `.png`.

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

What a script can rely on. Written by every table of a `data` entry with
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
   for objects, `track_id` for tracks - and for objects too when the column
   `track_id` is picked. `image`, `class` and `object_id` are always
   written, picked or not (the Exports tab shows them ticked and greyed).
2. `t` and `z` (counted from 0), only where the results have several.
3. The value columns, in the format's fixed order of column types (not the
   order they were ticked in).

`transposed`: no `class` column - the classes are side by side, in the
order of their ids, each with all value columns. The classes are those
picked, else every class but Background - a class without objects
included (`emptyClasses` doesn't take it out), so every file has the same
columns. An `image` (`well`, `folder`) row holds an image's (well's,
folder's) values of every class; an `object` row the n-th object of each
class of an image (`object_id` per class; a class with fewer objects is
empty in the rows past its last).

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
- A time course: `class`, the line (`image`, `well`, or `track_id` and
  `image`), `t`, `z` (several planes), `time_s` (empty without a frame
  interval), `n_images`, `n_objects`, `n_tracks`, then per column and
  aggregation `area_um2_avg`, … (`…_pct` with `relative`). Always the same
  columns, so a script needn't check for them.
- A profile: the line (`class`, `image` or `well`), `position`,
  `n_objects`, `mean`, `sd`, `median`, `q1`, `q3`; of objects: `image`,
  `class`, `object_id`, `position`, `value`.
- Children and linked objects are named by their key, class ids
  included: `children_1_2_count`, `children_1_2_max_area_um2`,
  `parent_1_id`, `parent_1_area_um2`.
- `transposed` adds the class at the end, its name made a name part
  (with its id where two classes would give the same): `count_nucleus`,
  `area_um2_avg_nucleus`, `object_id_cell`.
- Measurement columns have no unit in their name (it isn't stored per
  column); the `MEASUREMENTS` table has the unit as a column.

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
