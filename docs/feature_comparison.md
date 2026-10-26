# EVAnalyzer vs. ilastik, CellProfiler and QuPath

EVAnalyzer features were checked in the code (2026-10-26). Features of the
other tools are from general knowledge and were not checked against their
latest releases.

**Legend**
- ✅ has it, ⚠️ partial or limited, ❌ missing
- **Verdict** column:
  - 🟰 **Equal**: all tools have it about equally well.
  - ⭐ **Only X**: only one tool has it.
  - 🏆 **X best**: several have it, one is clearly better.

## Feature comparison

| Area / Feature | EVAnalyzer | ilastik | CellProfiler | QuPath | Verdict |
|---|---|---|---|---|---|
| **Image input** | | | | | |
| Microscope formats (CZI, ND2, LIF, VSI…) | ✅ scopeio, built in | ⚠️ via conversion / limited | ✅ via Bio-Formats | ✅ via Bio-Formats | 🟰 Equal (EVA, CP, QuPath) |
| Whole-slide images with pyramids (SVS, NDPI, MRXS) | ✅ | ❌ | ❌ | ✅ | 🏆 QuPath ≈ EVA |
| Multi-well plates / HCS | ✅ plate heat maps | ❌ | ✅ | ❌ | 🏆 CellProfiler ≈ EVA |
| Z-stacks: viewing | ✅ ortho + volume view | ✅ | ⚠️ | ⚠️ plane by plane | 🟰 Equal (EVA, ilastik) |
| Z-stacks: true 3D analysis | ❌ planned | ✅ | ✅ | ❌ | 🏆 ilastik |
| **Workflow** | | | | | |
| Pipeline of steps, live preview | ✅ live tuning, breakpoints | ⚠️ fixed workflows | ✅ test mode, not live | ⚠️ scripts / workflow log | 🏆 EVA |
| Batch processing of thousands of images | ✅ GUI + CLI | ✅ headless | ✅ incl. clusters | ⚠️ project batch via scripts | 🏆 CellProfiler ≈ EVA |
| Command-line use | ✅ | ✅ | ✅ | ⚠️ | 🟰 Equal |
| Client/server, multi-user remote compute | ✅ `serve` / `--remote`, TLS, users | ❌ | ❌ | ❌ | ⭐ **Only EVA** |
| Scripting | ⚠️ Rhai | ⚠️ Python API | ✅ Python plugins | ✅ Groovy, very rich | 🏆 QuPath |
| Plugin ecosystem / community | ❌ new | ⚠️ | ✅ large | ✅ large | 🏆 CellProfiler ≈ QuPath |
| **Segmentation (classic)** | | | | | |
| Threshold, watershed, filters, morphology | ✅ | ⚠️ minimal | ✅ | ✅ | 🟰 Equal (EVA, CP, QuPath) |
| Illumination correction, rolling ball | ✅ | ❌ | ✅ | ⚠️ | 🟰 Equal (EVA, CP) |
| Carving (interactive 3D) | ❌ | ✅ | ❌ | ❌ | ⭐ **Only ilastik** |
| Multicut / boundary-based segmentation | ❌ | ✅ | ❌ | ❌ | ⭐ **Only ilastik** |
| Density counting | ❌ | ✅ | ❌ | ❌ | ⭐ **Only ilastik** |
| **Machine learning** | | | | | |
| Interactive pixel classifier (draw → train) | ✅ RF/MLP/KNN, brush + polygon | ✅ RF, the reference | ⚠️ via ilastik | ✅ RF/ANN/KNN | 🏆 ilastik (UX, feature set) |
| Object classifier | ✅ | ✅ | ⚠️ via CP Analyst | ✅ | 🟰 Equal (EVA, ilastik, QuPath) |
| Cellpose / StarDist built in | ✅ both, GPU (CUDA 13) | ⚠️ via NN workflow | ⚠️ plugins | ⚠️ StarDist extension, InstanSeg | 🏆 EVA |
| U-Net / YOLO inference | ✅ | ⚠️ bioimage.io | ❌ | ⚠️ | 🏆 EVA |
| Training deep-learning models (e.g. fine-tuning Cellpose) | ❌ | ⚠️ | ❌ | ❌ | 🏆 ilastik (none is strong) |
| **Measurements** | | | | | |
| Intensity, shape, area occupied | ✅ checked against CP/ImageJ | ⚠️ basic object features | ✅ | ✅ | 🟰 Equal (EVA, CP, QuPath) |
| Texture (Haralick) | ✅ | ⚠️ | ✅ | ✅ | 🟰 Equal (EVA, CP, QuPath) |
| Zernike, granularity, radial distribution | ❌ | ❌ | ✅ | ❌ | ⭐ **Only CellProfiler** |
| Colocalization (Pearson, Manders, Costes, RWC) | ✅ image + object level | ❌ | ✅ | ⚠️ | 🏆 CellProfiler ≈ EVA |
| Neighbors, parent/child relations, distances | ✅ explicit Relate, Voronoi | ❌ | ✅ | ✅ spatial analysis | 🟰 Equal (EVA, CP, QuPath) |
| Image quality metrics | ✅ | ❌ | ✅ | ❌ | 🟰 Equal (EVA, CP) |
| **Time-lapse** | | | | | |
| Object tracking | ⚠️ greedy nearest neighbour, no divisions or gaps | ✅ divisions, conservation tracking | ✅ several methods | ❌ | 🏆 ilastik |
| **Pathology** | | | | | |
| Manual annotation tools | ✅ rectangle, ellipse, polygon, brush, eraser, wand, markers | ⚠️ labels only | ❌ | ✅ full toolset (+ vertex editing, lines) | 🟰 Equal (EVA, QuPath) |
| TMA dearraying | ❌ | ❌ | ❌ | ✅ | ⭐ **Only QuPath** |
| GeoJSON annotation exchange | ✅ | ❌ | ❌ | ✅ | 🟰 Equal (EVA, QuPath) |
| Brightfield stain separation (H-DAB) | ✅ Color Deconvolution (H-DAB, H&E, H-E-DAB, custom) | ❌ | ✅ UnmixColors | ✅ | 🟰 Equal (EVA, CP, QuPath) |
| **Results** | | | | | |
| Built-in results database | ✅ DuckDB, one database per project | ❌ | ⚠️ SQLite/MySQL export | ⚠️ in the project | 🏆 EVA |
| Charts, plate heat maps, result views | ✅ | ❌ | ⚠️ via CP Analyst | ⚠️ | 🏆 EVA |
| CSV / Excel export | ✅ both | ✅ CSV/HDF5 | ✅ | ✅ | 🟰 Equal |
| **Footprint & licensing** | | | | | |
| Low RAM, fast startup on cheap PCs | ✅ ~207 MB idle, Rust | ⚠️ | ⚠️ Python/Java | ⚠️ JVM, needs lots of RAM | ⭐ **EVA's design goal** |
| Reproducibility (one versioned project file) | ✅ schema migrations | ✅ | ✅ `.cppipe` | ⚠️ scripts | 🟰 Equal |
| License | PolyForm Noncommercial | GPL | BSD-3 | GPL-3 | Only CP is permissive. EVA and the GPL tools both limit commercial use, in different ways. |
| Track record (citations, validation) | ❌ new | ✅ | ✅ very large | ✅ very large | 🏆 CellProfiler / QuPath |

## Can EVAnalyzer replace them?

| Replace… | Verdict | Why |
|---|---|---|
| **CellProfiler** | **Mostly yes**, for 2D HCS work: plates, batch, segmentation, intensity/shape/texture/coloc/relations measurements | Gaps: true 3D, Zernike/granularity/radial distribution, the plugin ecosystem, and its established standing as a citable reference. EVA gains a live GUI, WSI support, built-in DL models and a results database. |
| **QuPath** | **Partially**, for automated detection and measurement on WSI | Gaps: TMA dearraying, Groovy scripting, polygon vertex editing and line tools. It is a strong pick for headless/batch WSI quantification where QuPath is heavy. |
| **ilastik** | **Partially**, for the common case (2D pixel/object classification → measure) | Gaps: carving, multicut, density counting, 3D pixel classification, and tracking with divisions. ilastik's interactive pixel-classification UX is still the benchmark. |

## Biggest gaps to close for a replacement claim

1. 3D analysis (already planned in TODO)
2. Better tracking: divisions and gap closing
3. TMA dearraying
4. Missing CellProfiler measurement families: Zernike, granularity, radial distribution

## What only EVAnalyzer offers

All three use cases in one app (HCS + WSI + ML), multi-user remote compute, a
DuckDB-backed results database with views, and a low RAM footprint.
