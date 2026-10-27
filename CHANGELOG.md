# Changelog

## Unreleased

### Automatic thresholds: rewritten from the original papers

The automatic threshold methods were reimplemented from their original
publications (Otsu 1979, Li & Lee 1993 / Li & Tam 1998, Kittler & Illingworth
1986, Zack 1977, Tsai 1985, Huang & Wang 1995, Prewitt & Mendelsohn 1966,
Ridler & Calvard 1978, Kapur et al. 1985, Glasbey 1993, Doyle 1962, Sahoo et
al. 1997, Shanbhag 1994, Yen et al. 1995), and each method now cites its
paper. Where the papers leave a detail open, the choice is documented.

**Your results may change slightly.** If a project uses one of the methods
below, re-running it can give a slightly different mask than before:

| Method | What changes |
|---|---|
| Otsu, Mean, MaxEntropy, Shanbhag, Yen, Percentile | No change. |
| Li | Usually one gray level lower. Some images where the old version failed (threshold far off) now get a sensible threshold. |
| MinError | Often a few gray levels different: the threshold now always settles in a minimum of the method's error criterion (before: about 3 in 4 images). |
| Triangle, IsoData, Renyi, Moments | Usually one gray level up or down. |
| Huang, Minimum | Rarely different; for Minimum only inside empty valleys of the histogram, where no pixel changes. |

One gray level is 1/255 of the image's intensity range. For Li, Triangle,
IsoData, Renyi and Moments the foreground area typically changes by less than
1 %, and by up to about 4 % in 1 of 20 images; object counts are rarely
affected. MinError can move further, as it may now settle in a different
minimum - check projects that use it. To compare old and new results
exactly, keep using the previous version for projects already analysed.

### See what the AI classifiers predict

The AI training panel has a new "Show on image" card for pixel classifiers:
**Prediction** colours every pixel of the preview in the colour of the class
it is predicted as - the stronger, the more probable. Pale areas are where the
model is unsure: paint a few examples there and train again. A legend sits at
the top left of the image, the class and its probability under the cursor
appear in the pixel info, and opacity and the background's colour can be
set. It shows the pipeline step that applies the model ("Add to pipeline"
adds one).

For an object classifier the card shows each object as strong as the model
is sure of its class, the class and its probability of the object under the
cursor, and the **most unsure objects**: click one to jump to it, copy it to
manual and give it its class, then train again. The AI Object Classifier step
also got a **Minimum certainty**: objects the model is less sure about keep
their class(es) as they are.

The pixel classifier also got faster: the class and its certainty now come
from one pass through the trees, about 3 × faster on one core with "Minimum
certainty" and 2 × without, and the work is spread over all cores.

### Trained classifiers need far less memory

Pipelines with a random forest or MLP from the AI training panel could need
several GB of extra RAM. Now a model is loaded once per run (not once per
worker thread), random forests no longer store their out-of-bag samples (a
model trained on 200,000 pixels: 57 MB → 6 MB; existing models shrink in
memory right away and on disk when trained again; the unused "Accuracy from
out-of-bag samples" checkbox is gone), and MLPs predict in small blocks
without training overhead. Example with 8 threads: random forest +5.9 GB →
+50 MB, MLP +3.9 GB → +50 MB; both faster (forest 1.9 s → 1.3 s, MLP
0.73 s → 0.24 s). Results are unchanged.

### Large projects: undo no longer fills the memory

Each undo step kept a full copy of the project - with 100,000 images about
240 MB per step, up to 100 steps. Now unchanged images are shared between
steps: 100 edits need about 1 MB instead of 24 GB (edits of an image's
annotations: under 1 GB), and an edit no longer pauses for ~0.1 s.

### Fixes

- Results list and export: choosing images in subfolders showed no objects.
- Selecting an object no longer clears what can be redone.
- The image list could highlight the wrong image (e.g. "1.tif" for "11.tif").

### Faster YOLOv5 segmentation

Masks are now assembled only around each detected object instead of over the
whole 640 × 640 window: mask building is about 100 × faster and needs far less
memory (e.g. ~120 MB instead of ~870 MB with 300 objects). Results are
unchanged.

### Licenses

- The About dialog lists every third-party component with its license text:
  all libraries, plus code adapted from other open-source projects, the fonts
  and icons, and the native libraries shipped with the app (libtorch, DuckDB,
  NVIDIA CUDA).
- Every download now contains `LICENSE` and `THIRD-PARTY-NOTICES.txt`.
