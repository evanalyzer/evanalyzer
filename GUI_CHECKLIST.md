# GUI check – new steps and columns (2026-10-24)

How to use: tick each box `[x]` that works. Where something is wrong, write it
under **Notes** (what you did, what you saw, what you expected; a screenshot
path helps). Then send the file back.

Test project: an image with nuclei and spots (e.g. the 81-image project), plus
the whole slide for section 9.

---

## 0. Basics for every new step

For each of: Measure Colocalization, Measure Object Neighbors, Measure
Distance To Parent, Measure Object Shape, Measure Texture

- [ ] Found in the step list under the *measure* category with that name
- [ ] Settings show readable names (no internal names like `channel_a`)
- [ ] The **ID** field is filled in automatically (e.g. `coloc_01`) and stays unique when adding the step twice
- [ ] Hover help / description texts are readable and make sense
- [ ] The step's summary line in the pipeline list makes sense
- [ ] Saving and reopening the project keeps every setting

Notes:

---

## 1. Measure Colocalization

Settings: two channels, *Threshold (% of maximum)* 15, *Costes thresholds*
Faster, *Per object of* = your cell class.

- [ ] Runs without error
- [ ] Results › per image: Pearson, Slope, Manders M1/M2, Overlap, K1, K2, RWC 1/2, Manders M1/M2 (Costes) – 11 values
- [ ] Results › per object: the same without Slope ("… per object")
- [ ] Value names show the channels ("channel 0 × channel 1")
- [ ] Ranges plausible: Pearson −1…1, Manders/RWC 0…1
- [ ] *Costes thresholds* = Off → the two Costes values disappear
- [ ] *Per object of* unset → no per-object values
- [ ] *Costes* = Fast on a 16-bit image: takes noticeably longer but finishes
- [ ] Exclude Image: the rule editor's measurement list offers these values (e.g. Pearson)

Notes:

---

## 2. Measure Object Neighbors

Settings: *Objects* = nuclei, *Neighbors* unset, try each *Neighbors are*:
Expand until adjacent / Adjacent / Within a distance (*Distance [px]* 10).

- [ ] Runs with each method
- [ ] Per object: Neighbors, Percent touching, First closest distance, Second closest distance, Angle between neighbors
- [ ] An isolated object: 0 neighbors; closest distances **empty** (not 0)
- [ ] Results list: an extra column "… · First closest distance · object" (the object the distance points to)
- [ ] Clicking that object column opens/jumps to that object
- [ ] *Neighbors* = another class (e.g. spots) works
- [ ] Whole slide or big image with "Within a distance": finishes in reasonable time

Notes:

---

## 3. Measure Distance To Parent

Needs a **Relate Objects** step before it (spots → cells).

- [ ] Runs; per child: distance to the parent's centre and to its edge
- [ ] Children without a parent: empty values
- [ ] The "… · object" column shows the parent; a click opens it
- [ ] Without a Relate step: clear message or empty values (no crash)

Notes:

---

## 4. Measure Object Shape

- [ ] Runs; per object: Convex area, Solidity (convex area), Euler number, Max/Mean/Median radius
- [ ] A ring-shaped object (hole) has Euler number 0, a filled one 1
- [ ] Solidity between 0 and 1

Notes:

---

## 5. Measure Texture

Settings: a channel, *Scale [px]* 3, *Gray levels* 256, *Directions* = Mean
of the directions, *Per object of* = cells.

- [ ] Runs; 13 values per image and 13 per object (Angular second moment … Information measure of correlation 2)
- [ ] *Directions* = Each direction → 52 values per image/object, names end with 0° / 45° / 90° / 135°
- [ ] *Gray levels* 8 → runs, values change
- [ ] Tiny objects (1–2 px wide): empty values, no crash
- [ ] Value names show channel and scale

Notes:

---

## 6. Transform Objects › Expand Without Overlap

Settings: function *Expand Without Overlap* (check how it is named in the
dropdown), margin 5 px, then 50 px.

- [ ] Function is offered in the dropdown with a readable name
- [ ] Viewer: objects grow, but neighbours never overlap; the gap between two objects is split between them
- [ ] Large margin: objects grow until they touch
- [ ] Unit nm/µm works too

Notes:

---

## 7. Results – new columns

In the Results list (and Plate/Charts where a column can be chosen):

- [ ] Max Feret [px], Min Feret [px], Major Axis [px], Minor Axis [px], Equivalent Diameter [px], Extent are offered and filled
- [ ] Per channel: Std Dev / Median / Lower Quartile / Upper Quartile / MAD Intensity (Ch n) offered and filled
- [ ] Plausible: Min ≤ Lower Quartile ≤ Median ≤ Upper Quartile ≤ Max
- [ ] Measurement-step values (sections 1–5) can be added as columns, aggregated per image, shown in Plate and Charts
- [ ] Parent id column: a click opens the parent
- [ ] Children / partner drill-down still works
- [ ] CSV export contains the new columns

Notes:

---

## 8. AI object classifier training

- [ ] The training dialog offers the new per-channel features: Std Dev, Median, Lower/Upper Quartile, MAD
- [ ] Training with them runs
- [ ] An annotation drawn before today: training skips it with a readable reason

Notes:

---

## 9. Whole slide

Run the whole-slide project.

- [ ] Finishes; time roughly 70 s on your laptop (was 130–170 s)
- [ ] Viewer: objects crossing tile seams are one object (no cut at the seams)
- [ ] Merged objects have intensities (not 0 / empty)
- [ ] Voronoi cells: no gaps or overlaps at tile seams

Notes:

---

## 10. Same result every time

- [ ] Run the same project twice (whole slide or the 81 images): the results (object counts, a few values, the Coloc/Voronoi objects) are identical

Notes:

---

## 11. Changes testers should notice

- [ ] Feret values differ from older runs (expected – they were wrong before)
- [ ] An old results file (`.duckdb`/`.evadb` from before today) doesn't open – the message is understandable
- [ ] A project with the removed "Pearson Correlation" step: the message is understandable

Notes:

---

## Anything else

Things that looked odd, were slow or confusing:

