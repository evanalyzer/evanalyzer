# Copyright 2026 Joachim Danmayr.
# Licensed under the PolyForm Noncommercial License 1.0.0.
"""Writes distance_to_parent_cellprofiler.json: CellProfiler's RelateObjects
child-parent distances - centroid to centroid and centroid to the parent's
edge (calculate_centroid_distances / calculate_minimum_distances in
cellprofiler/modules/relateobjects.py, CellProfiler 4.2.8, copied below
for label images and a given parent per child) - to check EVA's Measure
Distance To Parent against.

Run with numpy, scipy, scikit-image and centrosome installed:
    python distance_to_parent_cellprofiler.py
"""

import json

import numpy
import scipy.ndimage
import skimage.segmentation
from centrosome.cpmorphology import centers_of_labels


def center_of_mass(labels):
    """cellprofiler_core Objects.center_of_mass: (i, j) per label."""
    return centers_of_labels(labels).transpose()


def centroid_distances(parent_labels, child_labels, parents_of):
    pcenters = center_of_mass(parent_labels)
    ccenters = center_of_mass(child_labels)
    parents_of = parents_of - 1
    mask = (parents_of != -1) | (parents_of > pcenters.shape[0])
    dist = numpy.array([numpy.nan] * ccenters.shape[0])
    dist[mask] = numpy.sqrt(numpy.sum((ccenters[mask, :] - pcenters[parents_of[mask], :]) ** 2, 1))
    return dist


def minimum_distances(parent_labels, child_labels, parents_of):
    mask = parents_of > 0
    ccenters = center_of_mass(child_labels)
    ccenters = ccenters[mask, :]
    parents_of_masked = parents_of[mask] - 1
    pperim = skimage.segmentation.find_boundaries(parent_labels, mode="inner") * parent_labels
    perim_loc = numpy.argwhere(pperim != 0)
    perim_loc_t = tuple(map(tuple, perim_loc.transpose()))
    perim_idx = pperim[perim_loc_t]
    reverse_column_order = list(range(2))[::-1]
    coordinates = perim_loc[:, reverse_column_order].transpose().tolist()
    coordinates.append(perim_idx)
    idx = numpy.lexsort(coordinates)
    perim_loc = perim_loc[idx, :]
    perim_idx = perim_idx[idx]
    counts = scipy.ndimage.sum(
        numpy.ones(len(perim_idx)), perim_idx, numpy.arange(1, perim_idx[-1] + 1)
    ).astype(numpy.int32)
    indexes = numpy.cumsum(counts) - counts
    ccounts = counts[parents_of_masked]
    cindexes = indexes[parents_of_masked]
    clabel = numpy.zeros(numpy.sum(ccounts), int)
    cfirst = numpy.cumsum(ccounts) - ccounts
    clabel[cfirst[1:]] += 1
    clabel = numpy.cumsum(clabel)
    cp_index = numpy.arange(len(clabel)) - cfirst[clabel]
    cp_index += cindexes[clabel]
    dist = numpy.sqrt(numpy.sum((perim_loc[cp_index, :] - ccenters[clabel, :]) ** 2, 1))
    # Finally, find the minimum distance per child
    min_dist = scipy.ndimage.minimum(dist, clabel, numpy.arange(len(ccounts)))
    # Account for unparented children
    dist = numpy.array([numpy.nan] * len(mask))
    dist[mask] = min_dist
    return dist


def main():
    rng = numpy.random.default_rng(20261024)
    h, w = 50, 70
    # Parents: a Voronoi of random seeds (touching cells, some at the edge).
    seeds = numpy.zeros((h, w), numpy.int32)
    for k in range(12):
        seeds[rng.integers(0, h), rng.integers(0, w)] = k + 1
    _, (i, j) = scipy.ndimage.distance_transform_edt(seeds == 0, return_indices=True)
    parents = seeds[i, j]
    present = numpy.unique(parents)
    parents = numpy.searchsorted(present, parents) + 1
    # Children: small discs; their parent is the one under their centre.
    children = numpy.zeros((h, w), numpy.int32)
    yy, xx = numpy.mgrid[0:h, 0:w]
    n = 0
    for _ in range(40):
        cx, cy, r = rng.uniform(0, w), rng.uniform(0, h), rng.uniform(1.0, 2.5)
        disc = (xx - cx) ** 2 + (yy - cy) ** 2 <= r * r
        if not disc.any() or (children[scipy.ndimage.binary_dilation(disc)] != 0).any():
            continue
        n += 1
        children[disc] = n
    centers = center_of_mass(children)
    parents_of = numpy.array([parents[int(round(c[0])), int(round(c[1]))] for c in centers])
    parents_of[::7] = 0  # some without a parent
    case = {
        "width": w,
        "height": h,
        "parents": parents.ravel().astype(int).tolist(),
        "children": children.ravel().astype(int).tolist(),
        "parent_of_child": parents_of.astype(int).tolist(),
        "centroid": [None if not numpy.isfinite(v) else float(v) for v in centroid_distances(parents, children, parents_of)],
        "minimum": [None if not numpy.isfinite(v) else float(v) for v in minimum_distances(parents, children, parents_of)],
    }
    with open("distance_to_parent_cellprofiler.json", "w") as f:
        json.dump([case], f)
    print(f"{n} children, {int((parents_of > 0).sum())} with a parent")


if __name__ == "__main__":
    main()
