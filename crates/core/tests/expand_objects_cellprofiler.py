"""Writes expand_objects_cellprofiler.json: the labels CellProfiler's
ExpandOrShrinkObjects gives when expanding objects by a number of pixels
(`expand` in cellprofiler/library/functions/object_processing.py,
CellProfiler 4.2.8, copied below - SciPy's distance_transform_edt with its
indices) for a few label images and distances, to check EVA's "Expand
Without Overlap" against.

Run with numpy and scipy installed:
    python expand_objects_cellprofiler.py
"""

import json

import numpy
import scipy.ndimage


def expand(labels, distance):
    """
    Expand labels by a specified distance.
    """

    background = labels == 0

    distances, (i, j) = scipy.ndimage.distance_transform_edt(
        background, return_indices=True
    )

    out_labels = labels.copy()

    mask = background & (distances <= distance)

    out_labels[mask] = labels[i[mask], j[mask]]

    return out_labels


def blobs(rng, width, height, count, max_radius):
    """Random discs that don't touch (a later disc overlapping an earlier
    one is left out), labelled 1.. in painting order."""
    labels = numpy.zeros((height, width), dtype=numpy.int32)
    yy, xx = numpy.mgrid[0:height, 0:width]
    label = 0
    for _ in range(count):
        cx, cy = rng.uniform(0, width), rng.uniform(0, height)
        r = rng.uniform(0.6, max_radius)
        disc = (xx - cx) ** 2 + (yy - cy) ** 2 <= r * r
        grown = scipy.ndimage.binary_dilation(disc)
        if not disc.any() or (labels[grown] != 0).any():
            continue
        label += 1
        labels[disc] = label
    return labels


def main():
    rng = numpy.random.default_rng(20261024)
    cases = []
    inputs = [
        ("sparse discs", blobs(rng, 64, 48, 25, 5.0)),
        ("crowded discs", blobs(rng, 50, 50, 80, 4.0)),
    ]
    # Single pixels on a grid: many equally near pixels (ties).
    grid = numpy.zeros((21, 21), dtype=numpy.int32)
    n = 0
    for y in range(2, 21, 6):
        for x in range(2, 21, 6):
            n += 1
            grid[y, x] = n
    inputs.append(("grid of points", grid))
    for name, labels in inputs:
        for distance in [1, 2.5, 4, 1000]:
            cases.append(
                {
                    "name": f"{name}, {distance} px",
                    "width": int(labels.shape[1]),
                    "height": int(labels.shape[0]),
                    "distance": float(distance),
                    "labels": labels.ravel().astype(int).tolist(),
                    "expanded": expand(labels, distance).ravel().astype(int).tolist(),
                }
            )
    with open("expand_objects_cellprofiler.json", "w") as f:
        json.dump(cases, f)
    print(f"{len(cases)} cases")


if __name__ == "__main__":
    main()
