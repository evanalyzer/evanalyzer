# EVAnalyzer – Vision

Where EVAnalyzer stands compared with **QuPath** and **CellProfiler**, and
which features could close the gaps. This is a list of options, **not a
commitment**: each item is to be decided on its own.

Status as of 2026-10-12. For EVAnalyzer it is based on the code. For QuPath
and CellProfiler it is based on their well-known feature sets; their exact
current versions were not checked.

Effort: **S** = days, **M** = 1–3 weeks, **L** = more than a month.

---

## 1. Where EVAnalyzer is already strong

| Area | EVAnalyzer | QuPath | CellProfiler |
|---|---|---|---|
| Speed and RAM | Rust, compact caches, 512 px grid viewer; usable on a 2-core / 4 GB machine | Java; needs a lot of RAM | Python; slow, RAM-heavy |
| Whole slide images | Viewer and tiled analysis, with tile-object merging | Yes (its core strength) | No real WSI support |
| Interactive viewer | Yes, with annotations and live intensity measurement | Yes | Minimal (only for checking a pipeline step) |
| Plate / well screening | Plate layout, well grouping (regex), per-well export, matrix view | No | Yes (via metadata) |
| Results | DuckDB database, tables, charts, plate heat maps, CSV/XLSX/Parquet | Measurement tables | Spreadsheets or a database; analysis needs CellProfiler Analyst |
| AI | StarDist, Cellpose, U-Net, YOLO, SAM; pixel and object classifier training built in | StarDist, Cellpose, InstanSeg via extensions | Via plugins (RunCellpose, RunStarDist) |
| Headless / remote | CLI plus a remote analysis server | Command line | Headless and cluster batch |
| Installation | Single binary | Java bundle | Python environment, often difficult |

---

## 2. Gaps compared with QuPath

### Q1 – Brightfield / IHC pathology (L)
- Colour deconvolution with stain vectors (H&E, H-DAB), plus estimating those vectors from the image.
- Positive cell detection: DAB scoring in 1+/2+/3+ classes, H-score, Allred score.
- *Needed only if pathology labs are a target group.* This is QuPath's core.

### Q2 – Cell = nucleus + cytoplasm (S, mostly there)
- Most of this can already be built: nuclei → `Voronoi` → `ObjectMath Subtract`. See C1 for what is still missing.

### Q3 – Annotation tools and hierarchy (M)
- Design: see section 7.
- Tools: brush, wand (snaps to edges), freehand, line/polyline, boolean operations (merge, subtract, intersect).
- Parent/child hierarchy, for example "all cells inside tumour region A".
- Per-region summaries: cells per mm², percent positive.
- `Object` already has `parent_id` / `children`, but no step fills them yet.

### Q4 – Spatial analysis (S–M)
- Distance between objects (already in `TODO`: "Add measure distance command").
- Distance from an object to an annotation border.
- Nearest-neighbour distances, number of neighbours, Delaunay clustering.

### Q5 – Interoperability (S each)
- GeoJSON import/export of objects and annotations. This is the exchange format with QuPath, napari and Python.
- Writing pyramidal OME-TIFF (a crop or a processed image).
- Sending a region to ImageJ/Fiji.
- OMERO access (common in core facilities). *(M)*

### Q6 – Texture and tissue features (M)
- Haralick texture features.
- SLIC superpixels with feature tables.
- Tissue detection as one step: threshold on the low-resolution level, producing an annotation.

### Q7 – Usability (S each)
- Measurement maps: colour objects by a measurement value directly in the viewer.
- Open recent files (already in `TODO`).
- Create a pipeline step from a viewer action. QuPath does this with its "workflow → script" feature.

### Q8 – Niche (M–L)
- TMA dearraying (tissue microarray cores).
- Interactive image alignment (overlaying two slides).
- An extension / plugin system.

---

## 3. Comparison with CellProfiler

### Where EVAnalyzer is better
- **Speed, RAM and installation:** a single Rust binary, versus a slow, memory-hungry Python app that is often hard to install.
- **Whole-slide images:** viewer and tiled analysis with object merging across tiles. CellProfiler has no real whole-slide support.
- **Interactive viewer:** annotations and live intensity measurement on the image. CellProfiler only shows images to check a pipeline step.
- **Built-in results:** database, tables, charts, plate heat maps and matrix view. CellProfiler needs the separate CellProfiler Analyst.
- **AI:** StarDist, Cellpose, U-Net, YOLO and SAM built in, plus pixel and object classifier training. CellProfiler needs plugins for this.
- **Remote server:** run analyses on a server. CellProfiler offers only cluster batch files.

### What is missing

### C1 – Secondary / tertiary objects (S, mostly there)
CellProfiler's most used screening pattern:
- **Primary objects:** nuclei.
- **Secondary objects:** the cell around each nucleus.
- **Tertiary objects:** cytoplasm = cell minus nucleus.

QuPath's "cell expansion" is the same idea.

**What EVAnalyzer can already do:**

```
Nucleus segmentation → Voronoi (centers: nuclei, mask: cell stain, max_radius) → ObjectMath Subtract (cell − nucleus)
```

- Each Voronoi region is linked to its nucleus.
- `ObjectMath Subtract` keeps each cytoplasm tied to its cell.
- So the topology is the same as CellProfiler's distance-based secondary objects.

**Where it differs:**

| | EVAnalyzer `Voronoi` | CellProfiler / QuPath |
|---|---|---|
| Border between touching cells | Geometric: halfway between the two nucleus **centres** | *Propagation*: follows the membrane/cytoplasm stain intensity, so the border lies where the real membrane is |
| Expansion limit | `max_radius` measured from the nucleus **centre point** | Distance measured from the nucleus **border** (CellProfiler Distance-N, QuPath expansion) |
| Large or elongated nuclei | A circle around the centre can cut off part of the nucleus. The midline to a small neighbour can cut through a big nucleus. | The cell always contains the whole nucleus and grows evenly around it |
| Result | Fine for round, evenly sized nuclei, which is often enough | Exact cell shapes in dense or irregular tissue |

**What would close the gap:**
1. **Seed from the nucleus mask instead of its centre point.** Grow the cell from the nucleus pixels. The distance is then measured from the nucleus border, and the nucleus is always fully inside its own cell. *(S)*
2. **Intensity-guided expansion ("propagation").** A seeded watershed on a membrane or cytoplasm channel, starting at the nuclei and limited by the mask and the maximum distance. `Watershed` already exists; it would need nucleus seeds plus a distance limit. *(M)*
3. **Optional:** a single "Cell expansion" step that runs nucleus → cell → cytoplasm in one go and gives nucleus/cell ratios, for users who know QuPath or CellProfiler. *(S, convenience only)*

### C2 – Relate objects (S)
- Design: see section 7.
- Link children to parents (spots → cell) and count children per parent.
- Give parent-level means of child measurements.
- Colocalization covers part of this already. The `parent_id` / `children` fields exist but are not filled.

### C3 – More measurements (S–M each)
- **Pixel-based colocalization:** Pearson, Manders, Costes, rank-weighted. Design sketch in `docs/idea.md`.
- **Image-level metrics:** whole-image intensity. Also in `docs/idea.md`.
- **Texture:** Haralick (same as Q6).
- **Granularity:** a spectrum of spot sizes.
- **Radial intensity distribution:** intensity from the object centre to its border.
- **Zernike moments:** advanced shape features.
- **Neighbours:** same as Q4.

### C4 – Image quality control (S)
- Focus / blur score, saturation percentage, intensity statistics per image.
- Used to flag bad wells or fields automatically. Very valuable in high-content screening.

### C5 – Object tracking over time (M)
- Design: see section 7.
- Link objects between frames, detect divisions, measure velocity and lineage.
- `Track` / `TrackId` / `TrackSettings` already exist in the object model, but no pipeline step does the tracking yet.

### C6 – 3D segmentation and measurement (L)
- Volumetric objects: 3D connected components and watershed, volume and surface area.
- EVAnalyzer currently works on 2D planes and Z projections.

### C7 – Metadata from file and folder names (S)
- CellProfiler extracts any field (plate, well, site, treatment, dose) with a regex and makes it available as a result column.
- EVAnalyzer has a grouping regex for plates and wells. Extending it to free metadata fields would allow, for example, dose–response charts.

### C8 – Result-level machine learning and gating (M)
- CellProfiler Analyst does interactive cell classification on the measurement table and gating in scatter plots.
- EVAnalyzer already has an object classifier and charts. Missing: selecting a population in a scatter plot and showing those cells in the image.

---

## 4. Suggested priorities

For the current target group (fluorescence microscopy and high-content
screening), ordered by benefit per effort:

1. **C1 / Q2** – seed `Voronoi` from the nucleus mask; later, intensity-guided expansion (the basic pattern already works)
2. **C2** – relate objects / child counts
3. **C4** – image quality control
4. **C3** – pixel-based colocalization and image-level metrics (design exists in `docs/idea.md`)
5. **Q4** – distances and neighbours
6. **Q5** – GeoJSON import/export
7. **C7** – free metadata fields from file names

Bigger strategic decisions, to be made separately:

- **Pathology users:** Q1, Q3, Q6.
- **Live-cell imaging:** C5.
- **3D:** C6.

---

## 5. Easy to do

Small steps that build on what already exists.

- **Grow from the nucleus pixels instead of the centre point (S).** Seed `Voronoi` from the nucleus mask instead of its centre point. The nucleus is then always fully inside its own cell, and the distance counts from its border. See C1.
- **Grow along the stain (M).** Use the existing `Watershed` on a membrane or cytoplasm channel, starting from the nuclei and stopping at a maximum distance. See C1.

---

## 6. Pipeline scopes: image steps and dataset steps

Status 2026-10-12: concept, not implemented. The architecture is still to be proposed and reviewed.

### The problem
Some analyses need the **whole data set**, not one image. Tracking is the first, but more will follow:
- **Tracking** across the frames of a movie.
- **Linking objects** across z-slices (3D without 3D segmentation) and across imaging rounds (cyclic multiplexing).
- **Thresholds set by controls**, e.g. "positive = above the 95th percentile of the negative-control wells".
- **Per-plate normalisation:** Z-score, percent of control, B-score.
- **Plate quality and dose response:** Z′-factor, dose–response fits (IC50).
- **Population analysis:** hit calling, outlier removal, clustering or UMAP, classifiers on measurements.

"Some things in the pipeline, some in the results" is hard to explain and gets messier with every new feature. A clear rule is needed.

### The model
**Everything that creates data is a pipeline step. The results view only shows data.** Steps differ only in *what they see*:

| Scope | Sees | Example steps | Runs |
|---|---|---|---|
| **Tile** (exists) | Pixels of one tile | Filters, threshold, segmentation | In parallel |
| **Image** (exists as `WholeImage`) | All objects and pixels of one image plane | Tile merge, Voronoi, relate objects | Once per image |
| **Dataset** (new) | The objects of **all** images, frames and wells, but **no pixels** | Tracking, normalisation, control thresholds, Z′, IC50, hit calling | Once, after all images |

For the user:

> **Image steps turn pixels into objects. Dataset steps connect and compare objects across images, frames and wells.**

Decision rule for developers:
- Needs pixels? → **image step**.
- Needs objects from more than one image or frame? → **dataset step**.
- Needs both? → the image step stores what is needed later (e.g. a compact mask or link candidates to the previous frame), and the dataset step uses it.

### Why steps, not result functions
- **Reproducible:** the project file fully describes how a number was made, which matters for publications.
- **Headless:** runs in the CLI and on the remote server. Code in the results window would exist only in the GUI.
- **Exportable:** the output lands in the database, so tables, charts and exports work automatically.
- **Same editor and concepts:** users learn nothing new.

The results view keeps everything that is only a **way of looking at the data**: filters, grouping per well, charts, drill-down, child counts per parent. Nothing there is saved as new data.

### User experience
- **The pipeline editor gets a second area,** "After all images" / "Dataset steps", below the image pipelines. It uses the same step cards, picker and settings.
- **One "Run" does both,** with a progress phase "Dataset steps".
- **"Re-run dataset steps only" (the key convenience).** The objects are already in the database, so changing the tracking distance or a normalisation takes seconds instead of a full re-analysis.
- **Clear ownership in the results:** each dataset step writes its own tables (links, tracks, normalised values), tagged with its run. A re-run replaces only those.

### How other tools do it

| Tool | Approach | Downside |
|---|---|---|
| CellProfiler | Image analysis in CellProfiler, data analysis in the separate CellProfiler Analyst or in R/Python | Two tools, split workflow |
| Harmony / Columbus (commercial screening software) | One "analysis sequence" that ends with per-well result steps | Closest to this model, but re-running only the end is not really possible |
| QuPath | Everything is a command, recorded into a script; no clear dataset stage | Cross-image analysis means scripting |
| TrackMate | Detection and linking are separate stages in one wizard | Only for tracking |

A unified pipeline with a dataset stage that can be re-run on its own would be cleaner than all four. That is a selling point.

### Starting point in the code
- `ExecutionScope` already has `Tile` and `WholeImage`. A dataset scope is the natural third level.
- Dataset steps need their own trait, because they work on the results database (DuckDB rows) instead of `GlobalPipelineCache`. Otherwise they plug into the same configuration and pipeline-editor system.
- The first dataset step: "Link tracks" (section 7).

---

## 7. Hierarchy and tracking – design

Status 2026-10-12: no pipeline step fills these fields yet.

### Starting point in the code
- `Object` already has `parent_id`, `children` and `track`. The results table has `parent_id`, `children` and `track_id` columns.
- `ExecutionScope::WholeImage` steps run once per image after all tiles (used by Voronoi and `TileMerge`).
- The executor analyses **one plane (t, z) at a time** and discards its objects afterwards.
- The results database stores centroid, bounding box, area, shape and intensities, but **no masks**.

### 7.0 Decided: relations data model (2026-10-23)

- **Explicit only.** Only the "Relate objects" step writes parent links. Voronoi, Propagation, ObjectMath, coloc intersections and TransformObjects stop writing `parent_id`: one responsibility per command, and a new command never has to know about links. "Largest overlap" finds the source exactly for Propagation, ObjectMath and intersections, and nearly always for Voronoi. A ready-made module (Voronoi/Propagation → Relate → ObjectMath) covers the CellProfiler pattern.
- **At most one parent per class.** `Object.parent_id` and `children` become `parents: IndexMap<ObjectClass, ObjectId>`; in DuckDB, `objects.parent_id`/`children` become `parent_ids MAP(INTEGER, UUID)`. The key is the parent class set in Relate. Children are derived with `GROUP BY parent_ids[class]` (no `children` column, no join).
- **Relate owns its key:** for every object of its child class it sets `parent_ids[parent class]` or clears it.
- **Separate from Coloc.** Different algorithm (directed, one per class, centroid rule); both only use `BboxGrid` and `Object::overlaps`.
- **Two meanings, two link columns** (like CellProfiler's RelateObjects / MaskObjects split). `parent_ids` = "belongs to" (Relate; every count and mean per parent). `coloc_partner_ids MAP(INTEGER, UUID[])` = "touches / at the same place" (Coloc; equal objects, all partners on both sides, "only the best partner" as an option). Coloc can also mark the colocalizing objects (of one class, or of every class) with a class ("ch1 with ch2"), for later steps - optional; the results show partners as ids, a count and an "Overlapping pairs" row type, never as means. ClassifyObjects' "Intersecting With" and Coloc's `MultiFor` stay in Advanced until they are removed. Details: TODO "Relate objects and Coloc".
- **Aggregates** (child count, mean/median/sum of a child measurement) are computed by the results and export queries, not stored.
- **Distances** (to parent centroid/edge, nearest neighbour, neighbour count, to a border) are per-object rows in the existing `measurements` table, possibly with an optional `related_object_id`. No pairwise distance matrix.

### 7.1 How to build them

**Hierarchy: rule-based containment (industry standard)**

| Tool | How a child is linked to its parent |
|---|---|
| CellProfiler `RelateObjects` | The child overlaps the parent, or its centre lies inside it; the parent gets child counts and child means |
| QuPath | Containment by centroid: annotation → cell → subcellular spots |
| Harmony / MetaXpress | "Find spots inside the cell region": well → field → cell → spot |

Best practice:
- **Matching rules:** "centre inside the parent" (default, most robust), "largest overlap", "at least X % overlap".
- **One parent per class pair**, so the result stays a tree. It can still have several levels: region → cell → nucleus → spot.
- **A parent in each class.** A single `parent_id` is too limited: a spot can belong to a cell *and* to a tissue region. Store parents per class, like `colocalized_with`.
- **Run after `TileMerge`**, otherwise objects cut at tile borders get wrong parents.

**Tracking: detect per frame, then link (industry standard)**
- **TrackMate (Fiji)** is the most used tool. Its LAP linker (Jaqaman 2008) links frame to frame by a cost (distance², plus optional penalties for differences in area or intensity), within a maximum distance. A second pass closes gaps and detects divisions.
- **CellProfiler `TrackObjects`:** overlap, distance or LAP linking.
- **Others:** btrack (Bayesian), trackpy (particles), ilastik tracking, Ultrack.
- **Newer learned linkers:** Trackastra (a transformer).
- **Data model:** a **graph** of objects and links, not a flat list per track, so divisions, merges and gaps fit naturally. The current `Track` struct (an ordered list plus `parent_track`) can't hold merges or gaps well.
- **Export formats:** Cell Tracking Challenge format, TrackMate XML, and the newer community graph-exchange format (GEFF).
- **Recommended order:** correct drift → LAP linking by centroid and features → close gaps → detect divisions. Mask overlap (IoU) as a linking cost is optional.

### 7.2 Where to run them

| | Where | Why |
|---|---|---|
| **Hierarchy links** | **Pipeline step** "Relate objects" (`WholeImage` scope, after `TileMerge`) | Needs the masks, which only the pipeline has |
| **Hierarchy aggregates** (child count, mean child intensity, children per mm²) | **In the results view**, computed from `parent_ids` | Only a way of looking at the data; change it without re-running. DuckDB does it in milliseconds. Aggregates that should be stored or exported as new data become a dataset step. |
| **Tracking** | **Dataset step** "Link tracks" (see section 6), on the objects in the results database | Linking needs all frames, but the pipeline sees only one. Centroid, area and intensities are already stored. Re-linking with other parameters takes seconds instead of re-segmenting the movie. TrackMate also keeps detection and linking separate for this reason. |

Limits of this split:
- **Linking by mask overlap** can't run on the database, because it has no masks. Either use bounding-box overlap as an approximation, or later add a pipeline step that links each frame to the previous one. The executor already processes time points in order, so that would cost one frame of objects in RAM.
- **Distances from a child to its parent's edge** also need masks, so they belong in the pipeline step too.

### 7.3 How to show them

**Standard on the market**

| Hierarchy | Tracking |
|---|---|
| A tree panel (QuPath) | Track overlay with a tail of the last N frames, coloured by track or by a measurement (TrackMate, napari) |
| Parent table with child counts and means (CellProfiler) | Lineage tree with time on one axis (TrackMate TrackScheme, Mastodon) |
| Selecting a parent highlights its children | Track table with speed, displacement, MSD and directionality |
| | Plots of a measurement over time per track; spider/rose plots for migration |

**What users would like but rarely get**
- **Linked selection everywhere.** Click a cell in the image → its row, its lineage branch and its points in the chart light up, and the other way round. Mastodon has this, but as a separate expert tool. EVAnalyzer has viewer, results and charts in one app.
- **Re-linking in seconds.** Change the maximum distance and see the new tracks immediately, without re-segmenting.
- **Uncertain links to review.** Each link gets a confidence, and the user checks only the uncertain ones (divisions, gaps, crowded areas) instead of watching the whole movie. Almost no tool offers this.
- **Manual corrections that survive a re-run.** A corrected link becomes a fixed constraint. TrackMate and CellProfiler lose edits on a re-run.
- **Hierarchy and tracking combined.** Track cells, and follow their spots and children over time, e.g. "spot count per cell along its lineage". Very rare.
- **Statistics per lineage.** Division time per generation and sister-cell comparisons out of the box.
- **Tracking at screening scale.** Motility per well as a plate heat map. Commercial screening tools do some of this, open-source tools hardly do. The plate matrix already exists.
- **Hierarchy on whole-slide images with millions of objects.** QuPath gets slow there; SQL aggregation in DuckDB doesn't.

### 7.4 Suggested order
1. **"Relate objects" pipeline step,** with parents per class and the three matching rules. Child aggregates are computed in the results.
2. **In the viewer:** select a parent to highlight its children. **In the results:** a parent table you can drill down into.
3. **"Link tracks" as a dataset step:** LAP with gap closing and divisions, stored as a links table (a graph). `track_id` stays as a shortcut.
4. **Tracks in the viewer** (tails, colouring), a track table, and plots over time.
5. **Lineage tree view** with linked selection. Later, review of uncertain links and corrections that survive a re-run.
