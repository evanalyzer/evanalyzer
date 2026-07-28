# Image-level metrics (whole-image object)

Discussion notes on adding image-level metrics in addition to the existing
object-based ones:

- Average intensity of the whole image
- Pixel-based colocalization analysis across multiple channels (a coefficient)

Not implemented yet — this is a design sketch to align on before building it.

## Core idea: model it as a Object

`Object`/`ObjectRow` already *is* "a mask + a bbox + per-channel intensities + a
coloc slot" (`crates/core/src/object.rs:38-99`, `crates/core/src/storage/duckdb.rs:445-480`).
A whole-image row is just a degenerate `Object`: `bbox = [0,0,w,h]`, mask =
"everything," no object class.

If we go this route, the entire results stack works on it for free:

- `results_loader.rs`
- `results_table.slint` / `results_chart.slint`
- CSV/XLSX export
- The DuckDB `objects` table

No new table, no new results screen. That's the strongest argument for this
shape over a parallel `image_metrics` table.

`Object::measure_intensities()` (`crates/core/src/object.rs:702`) already computes
exactly "avg/min/max/sum per channel" for any mask — for a whole-image `Object`
that's the average-intensity metric with zero new math.

## The one real wrinkle: tiling

Large images are processed in independent 4096px tiles
(`crates/core/src/job/job_executor.rs:455`), each tile gets its own
`PipelineCache`, runs the *entire* pipeline, and is appended to DuckDB and
discarded — in parallel, with **no cross-tile merge step**
(`crates/core/src/storage/duckdb.rs:290-295`, a plain `Appender`).

Object ROIs mostly get away with this because a nucleus is tiny next to
4096px; a *whole-image* metric can't — if it's dropped in as a normal
`ImageAlgorithm` command, it only computes the truly-whole-image answer for
images that fit in one tile. Anything bigger silently becomes "per-tile
average," not "per-image average."

**The real fork:** does this need to reflect a specific point in the user's
configured pipeline (e.g. "after background subtraction"), or is it always
the raw/original image? For "average intensity" and "pixel coloc," the bet is
on the latter — these are usually meant as fixed QC/normalization numbers,
not something that should change based on where someone drops a step.

### Recommendation

Don't make it a draggable pipeline command at all. Compute it once per image,
outside the tiled loop, directly from the full-resolution channel data — a
small new function called from `analyze_image`/`analyze_image_tiles_parallel`
after (or independent of) the tile loop, building one synthetic `Object` and
exporting exactly one row per image through the existing exporter path.

This sidesteps the tiling problem entirely and is much less invasive than
threading a shared cross-tile accumulator through the parallel
`try_for_each`.

If pipeline-position-dependence turns out to be needed later, that's the
harder version: real accumulator state shared across tiles, finalized when
the last tile for an image lands. Worth doing only if the raw-image approach
turns out insufficient.

## The coefficient

Almost certainly **Pearson's Correlation Coefficient** — the headline number
in ImageJ's Coloc2, and the standard "coefficient" people mean by
pixel-based coloc.

Worth pairing with **Manders' M1/M2** (thresholded overlap fractions) since
they reuse the same paired-channel pixel loop and are the other half of what
Coloc2 reports.

Both computed pairwise across every channel pair over the image.

### Storage

Give this its own JSON field (e.g. `pixel_coloc_json`) rather than reusing
`coloc_json` — that field's current shape is
`object class → [colocalized object ids]` (spatial overlap between segmented
objects), semantically different from a numeric per-channel-pair coefficient.

## Display

- Tag the row distinctly: a reserved `SegmentationClass` value, or a small
  `is_image_summary` flag — cleaner than overloading class semantics.
- Lean on the **existing** "Group by: Image" mode in
  `crates/app/src/results/results_loader.rs:677` plus column visibility
  toggles rather than building new UI.
- Geometry columns (perimeter, circularity, etc.) just show as 0/blank for
  these rows, which is harmless as long as the class filter lets people
  exclude them from object-level views easily.

## Open question

Confirm: are these metrics always computed on the raw/original image, or do
they need to reflect a specific point in the configured pipeline? This
decides between the simple (recommended) approach and the harder
cross-tile-accumulator version.
