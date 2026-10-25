# Copyright 2026 Joachim Danmayr.
# Licensed under the PolyForm Noncommercial License 1.0.0.
"""Writes object_neighbors_cellprofiler.json: CellProfiler's
MeasureObjectNeighbors values per object - number of neighbors, percent
touching, first/second closest object and distance, angle between
neighbors (MeasureObjectNeighbors.run in
cellprofiler/modules/measureobjectneighbors.py, CellProfiler 4.2.8, copied
below for 2D label images where every label is kept: its renumbering
between "small removed" and kept labels is the identity then) - for a few
label images and methods, to check EVA's Measure Object Neighbors against.

Run with numpy, scipy and centrosome installed:
    python object_neighbors_cellprofiler.py
"""

import json

import numpy
import scipy.ndimage
import scipy.signal
from centrosome.cpmorphology import centers_of_labels, strel_disk
from centrosome.cpmorphology import fixup_scipy_ndimage_result as fix
from centrosome.outline import outline

D_ADJACENT = "Adjacent"
D_EXPAND = "Expand until adjacent"
D_WITHIN = "Within a specified distance"


def run(labels, neighbor_labels, neighbors_are_objects, distance_method, within):
    """The 2D path of MeasureObjectNeighbors.run; labels 1..n all present."""
    if neighbors_are_objects:
        neighbor_labels = labels.copy()
    original_labels = labels.copy()
    original_neighbor_labels = neighbor_labels.copy()
    nobjects = numpy.max(labels)
    nkept_objects = nobjects
    nneighbors = numpy.max(neighbor_labels)
    object_numbers = numpy.arange(nobjects) + 1
    neighbor_numbers = numpy.arange(nneighbors) + 1
    has_pixels = numpy.ones(nobjects, bool)
    neighbor_has_pixels = numpy.ones(nneighbors, bool)

    neighbor_count = numpy.zeros((nobjects,))
    pixel_count = numpy.zeros((nobjects,))
    first_object_number = numpy.zeros((nobjects,), int)
    second_object_number = numpy.zeros((nobjects,), int)
    first_x_vector = numpy.zeros((nobjects,))
    second_x_vector = numpy.zeros((nobjects,))
    first_y_vector = numpy.zeros((nobjects,))
    second_y_vector = numpy.zeros((nobjects,))
    angle = numpy.zeros((nobjects,))
    percent_touching = numpy.zeros((nobjects,))
    if distance_method == D_EXPAND:
        i, j = scipy.ndimage.distance_transform_edt(
            labels == 0, return_distances=False, return_indices=True
        )
        labels = labels[i, j]
        distance = 1  # dilate once to make touching edges overlap
        if neighbors_are_objects:
            neighbor_labels = labels.copy()
    elif distance_method == D_WITHIN:
        distance = within
    elif distance_method == D_ADJACENT:
        distance = 1
    if nneighbors > (1 if neighbors_are_objects else 0):
        object_indexes = numpy.arange(nobjects, dtype=numpy.int32) + 1
        ocenters = centers_of_labels(original_labels).transpose()
        ncenters = centers_of_labels(original_neighbor_labels).transpose()
        perimeter_outlines = outline(labels)
        perimeters = fix(scipy.ndimage.sum(numpy.ones(labels.shape), perimeter_outlines, object_indexes))

        order = numpy.zeros((nobjects, min(nneighbors, 3)), dtype=numpy.uint32)
        j = numpy.arange(nneighbors)
        partition_keys = tuple(range(min(nneighbors, 3)))
        for i in range(nobjects):
            dr = numpy.sqrt((ocenters[i, 0] - ncenters[j, 0]) ** 2 + (ocenters[i, 1] - ncenters[j, 1]) ** 2)
            order[i, :] = numpy.argpartition(dr, partition_keys)[:3]

        first_neighbor = 1 if neighbors_are_objects else 0
        first_object_index = order[:, first_neighbor]
        first_x_vector = ncenters[first_object_index, 1] - ocenters[:, 1]
        first_y_vector = ncenters[first_object_index, 0] - ocenters[:, 0]
        if nneighbors > first_neighbor + 1:
            second_object_index = order[:, first_neighbor + 1]
            second_x_vector = ncenters[second_object_index, 1] - ocenters[:, 1]
            second_y_vector = ncenters[second_object_index, 0] - ocenters[:, 0]
            v1 = numpy.array((first_x_vector, first_y_vector))
            v2 = numpy.array((second_x_vector, second_y_vector))
            dot = numpy.sum(v1 * v2, 0) / numpy.sqrt(numpy.sum(v1 ** 2, 0) * numpy.sum(v2 ** 2, 0))
            angle = numpy.arccos(dot) * 180.0 / numpy.pi

        strel = strel_disk(distance)
        strel_touching = strel_disk(distance + 0.5)
        i, j = numpy.mgrid[0 : labels.shape[0], 0 : labels.shape[1]]
        minimums_i, maximums_i, _, _ = scipy.ndimage.extrema(i, labels, object_indexes)
        minimums_j, maximums_j, _, _ = scipy.ndimage.extrema(j, labels, object_indexes)
        minimums_i = numpy.maximum(fix(minimums_i) - distance, 0).astype(int)
        maximums_i = numpy.minimum(fix(maximums_i) + distance + 1, labels.shape[0]).astype(int)
        minimums_j = numpy.maximum(fix(minimums_j) - distance, 0).astype(int)
        maximums_j = numpy.minimum(fix(maximums_j) + distance + 1, labels.shape[1]).astype(int)
        for object_number in object_numbers:
            index = object_number - 1
            patch = labels[minimums_i[index] : maximums_i[index], minimums_j[index] : maximums_j[index]]
            npatch = neighbor_labels[minimums_i[index] : maximums_i[index], minimums_j[index] : maximums_j[index]]
            patch_mask = patch == (index + 1)
            if distance <= 5:
                extended = scipy.ndimage.binary_dilation(patch_mask, strel)
            else:
                extended = scipy.signal.fftconvolve(patch_mask, strel, mode="same") > 0.5
            neighbors = numpy.unique(npatch[extended])
            neighbors = neighbors[neighbors != 0]
            if neighbors_are_objects:
                neighbors = neighbors[neighbors != object_number]
            nc = len(neighbors)
            neighbor_count[index] = nc
            outline_patch = (
                perimeter_outlines[minimums_i[index] : maximums_i[index], minimums_j[index] : maximums_j[index]]
                == object_number
            )
            if neighbors_are_objects:
                extendme = (patch != 0) & (patch != object_number)
                if distance <= 5:
                    extended = scipy.ndimage.binary_dilation(extendme, strel_touching)
                else:
                    extended = scipy.signal.fftconvolve(extendme, strel_touching, mode="same") > 0.5
            else:
                if distance <= 5:
                    extended = scipy.ndimage.binary_dilation((npatch != 0), strel_touching)
                else:
                    extended = scipy.signal.fftconvolve((npatch != 0), strel_touching, mode="same") > 0.5
            overlap = numpy.sum(outline_patch & extended)
            pixel_count[index] = overlap
        percent_touching = pixel_count * 100 / perimeters
        object_indexes = object_numbers - 1
        neighbor_indexes = neighbor_numbers - 1
        first_object_number = numpy.zeros(nkept_objects, int)
        second_object_number = numpy.zeros(nkept_objects, int)
        if nkept_objects > (1 if neighbors_are_objects else 0):
            di = ocenters[object_indexes[:, numpy.newaxis], 0] - ncenters[neighbor_indexes[numpy.newaxis, :], 0]
            dj = ocenters[object_indexes[:, numpy.newaxis], 1] - ncenters[neighbor_indexes[numpy.newaxis, :], 1]
            distance_matrix = numpy.sqrt(di * di + dj * dj)
            distance_matrix[~has_pixels, :] = numpy.inf
            distance_matrix[:, ~neighbor_has_pixels] = numpy.inf
            order = numpy.lexsort([distance_matrix]).astype(first_object_number.dtype)
            if neighbors_are_objects:
                first_object_number[has_pixels] = order[has_pixels, 1] + 1
                if nkept_objects > 2:
                    second_object_number[has_pixels] = order[has_pixels, 2] + 1
            else:
                first_object_number[has_pixels] = order[has_pixels, 0] + 1
                if order.shape[1] > 1:
                    second_object_number[has_pixels] = order[has_pixels, 1] + 1
    return {
        "number_of_neighbors": neighbor_count.tolist(),
        "percent_touching": percent_touching.tolist(),
        "first_closest_object": first_object_number.tolist(),
        "first_closest_distance": numpy.sqrt(first_x_vector ** 2 + first_y_vector ** 2).tolist(),
        "second_closest_object": second_object_number.tolist(),
        "second_closest_distance": numpy.sqrt(second_x_vector ** 2 + second_y_vector ** 2).tolist(),
        "angle_between_neighbors": [float(a) for a in numpy.atleast_1d(angle)],
    }


def blobs(rng, width, height, count, max_radius, touching):
    """Random discs; `touching`: later ones may border earlier ones (not
    overlap), else they keep a pixel apart. Labelled 1..n."""
    labels = numpy.zeros((height, width), dtype=numpy.int32)
    yy, xx = numpy.mgrid[0:height, 0:width]
    n = 0
    for _ in range(count):
        cx, cy = rng.uniform(0, width), rng.uniform(0, height)
        r = rng.uniform(1.5, max_radius)
        disc = (xx - cx) ** 2 + (yy - cy) ** 2 <= r * r
        test = disc if touching else scipy.ndimage.binary_dilation(disc)
        if not disc.any() or (labels[test] != 0).any():
            continue
        n += 1
        labels[disc] = n
    return labels


def main():
    rng = numpy.random.default_rng(20261024)
    cases = []
    crowded = blobs(rng, 60, 50, 120, 5.0, True)
    sparse = blobs(rng, 70, 50, 30, 4.0, False)
    spots = blobs(rng, 60, 50, 40, 2.0, False)
    for name, labels, neighbor_labels, method, within in [
        ("crowded, expand", crowded, None, D_EXPAND, 0),
        ("crowded, adjacent", crowded, None, D_ADJACENT, 0),
        ("sparse, within 4", sparse, None, D_WITHIN, 4),
        ("sparse, within 7 (fft)", sparse, None, D_WITHIN, 7),
        ("sparse, expand", sparse, None, D_EXPAND, 0),
        ("crowded vs spots, within 3", crowded, spots, D_WITHIN, 3),
        ("crowded vs spots, expand", crowded, spots, D_EXPAND, 0),
    ]:
        same = neighbor_labels is None
        result = run(labels, labels if same else neighbor_labels, same, method, within)
        cases.append(
            {
                "name": name,
                "width": int(labels.shape[1]),
                "height": int(labels.shape[0]),
                "method": method,
                "distance": within,
                "labels": labels.ravel().astype(int).tolist(),
                "neighbor_labels": None if same else neighbor_labels.ravel().astype(int).tolist(),
                **result,
            }
        )
    with open("object_neighbors_cellprofiler.json", "w") as f:
        json.dump(cases, f)
    print(f"{len(cases)} cases, {sum(len(c['number_of_neighbors']) for c in cases)} objects")


if __name__ == "__main__":
    main()
