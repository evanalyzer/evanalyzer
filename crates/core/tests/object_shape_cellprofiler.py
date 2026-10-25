# Copyright 2026 Joachim Danmayr.
# Licensed under the PolyForm Noncommercial License 1.0.0.
"""Writes object_shape_cellprofiler.json: CellProfiler's MeasureObjectSizeShape
values per object (MeasureObjectSizeShape.analyze_objects in
cellprofiler/modules/measureobjectsizeshape.py, CellProfiler 4.2.8, copied
below for 2D without Zernike features: scikit-image's regionprops, the
radii from the distance transform, centrosome's Feret diameters) for a
label image of irregular shapes - to check EVA's shape values against.

Run with numpy, scipy, scikit-image and centrosome installed:
    python object_shape_cellprofiler.py
"""

import json

import centrosome.cpmorphology
import numpy
import scipy.ndimage
import skimage.measure


def analyze(labels):
    """analyze_objects, 2D, for labels 1..n without overlaps."""
    indices = numpy.arange(1, labels.max() + 1)
    nobjects = len(indices)
    desired_properties = [
        "label", "image", "area", "perimeter", "bbox", "bbox_area",
        "major_axis_length", "minor_axis_length", "orientation", "centroid",
        "equivalent_diameter", "extent", "eccentricity", "convex_area",
        "solidity", "euler_number",
    ]
    props = skimage.measure.regionprops_table(labels, properties=desired_properties)

    formfactor = 4.0 * numpy.pi * props["area"] / props["perimeter"] ** 2
    denom = [max(x, 1) for x in 4.0 * numpy.pi * props["area"]]
    compactness = props["perimeter"] ** 2 / denom

    max_radius = numpy.zeros(nobjects)
    median_radius = numpy.zeros(nobjects)
    mean_radius = numpy.zeros(nobjects)
    for index, mini_image in enumerate(props["image"]):
        # Pad image to assist distance tranform
        mini_image = numpy.pad(mini_image, 1)
        distances = scipy.ndimage.distance_transform_edt(mini_image)
        max_radius[index] = centrosome.cpmorphology.fixup_scipy_ndimage_result(
            scipy.ndimage.maximum(distances, mini_image)
        )
        mean_radius[index] = centrosome.cpmorphology.fixup_scipy_ndimage_result(
            scipy.ndimage.mean(distances, mini_image)
        )
        median_radius[index] = centrosome.cpmorphology.median_of_labels(
            distances, mini_image.astype("int"), [1]
        )[0]

    i, j = numpy.nonzero(labels)
    ijv = numpy.column_stack((i, j, labels[i, j]))
    chulls, chull_counts = centrosome.cpmorphology.convex_hull_ijv(ijv, indices)
    min_feret_diameter, max_feret_diameter = centrosome.cpmorphology.feret_diameter(
        chulls, chull_counts, indices
    )
    return {
        "area": props["area"].tolist(),
        "perimeter": props["perimeter"].tolist(),
        "major_axis_length": props["major_axis_length"].tolist(),
        "minor_axis_length": props["minor_axis_length"].tolist(),
        "eccentricity": props["eccentricity"].tolist(),
        "form_factor": formfactor.tolist(),
        "extent": props["extent"].tolist(),
        "solidity": props["solidity"].tolist(),
        "compactness": numpy.asarray(compactness).tolist(),
        "euler_number": props["euler_number"].tolist(),
        "max_radius": max_radius.tolist(),
        "mean_radius": mean_radius.tolist(),
        "median_radius": median_radius.tolist(),
        "convex_area": props["convex_area"].tolist(),
        "min_feret_diameter": numpy.asarray(min_feret_diameter).tolist(),
        "max_feret_diameter": numpy.asarray(max_feret_diameter).tolist(),
        "equivalent_diameter": props["equivalent_diameter"].tolist(),
    }


def shapes(rng, width, height):
    """Irregular shapes that don't touch: unions of discs, rings (holes),
    rotated bars, single pixels and lines; labelled 1..n."""
    labels = numpy.zeros((height, width), numpy.int32)
    yy, xx = numpy.mgrid[0:height, 0:width]
    n = 0
    for k in range(70):
        cx, cy = rng.uniform(4, width - 4), rng.uniform(4, height - 4)
        kind = k % 5
        if kind == 0:
            shape = numpy.zeros_like(labels, bool)
            for _ in range(3):
                ox, oy, r = rng.normal(0, 2.5), rng.normal(0, 2.5), rng.uniform(1.5, 4)
                shape |= (xx - cx - ox) ** 2 + (yy - cy - oy) ** 2 <= r * r
        elif kind == 1:
            r = rng.uniform(3, 6)
            d2 = (xx - cx) ** 2 + (yy - cy) ** 2
            shape = (d2 <= r * r) & (d2 >= (r * 0.45) ** 2)
        elif kind == 2:
            a = rng.uniform(0, numpy.pi)
            u = (xx - cx) * numpy.cos(a) + (yy - cy) * numpy.sin(a)
            v = -(xx - cx) * numpy.sin(a) + (yy - cy) * numpy.cos(a)
            shape = (abs(u) <= rng.uniform(3, 8)) & (abs(v) <= rng.uniform(0.6, 2.5))
        elif kind == 3:
            shape = (xx == int(cx)) & (yy == int(cy))
        else:
            length = int(rng.integers(2, 7))
            shape = (yy == int(cy)) & (xx >= int(cx)) & (xx < int(cx) + length)
        if not shape.any() or (labels[scipy.ndimage.binary_dilation(shape, iterations=2)] != 0).any():
            continue
        # One piece only (8-connected).
        pieces, count = scipy.ndimage.label(shape, structure=numpy.ones((3, 3)))
        if count != 1:
            continue
        n += 1
        labels[shape] = n
    return labels


def main():
    rng = numpy.random.default_rng(20261024)
    labels = shapes(rng, 120, 90)
    case = {
        "width": int(labels.shape[1]),
        "height": int(labels.shape[0]),
        "labels": labels.ravel().astype(int).tolist(),
        **{
            k: [v if numpy.isfinite(v) else None for v in values]
            for k, values in analyze(labels).items()
        },
    }
    with open("object_shape_cellprofiler.json", "w") as f:
        json.dump([case], f)
    print(f"{labels.max()} objects")


if __name__ == "__main__":
    main()
