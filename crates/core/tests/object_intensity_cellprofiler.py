"""Writes object_intensity_cellprofiler.json: CellProfiler's per-object
intensity statistics (MeasureObjectIntensity: integrated, mean, std, min,
max, lower quartile, median, upper quartile, MAD - the label-wise code of
cellprofiler/modules/measureobjectintensity.py, CellProfiler 4.2.8, copied
below) for a few label images, to check EVA's object intensities against.

Pixel values are float32 (as EVA reads them), computed in float64 like
CellProfiler does.

Run with numpy, scipy and centrosome installed:
    python object_intensity_cellprofiler.py
"""

import json

import centrosome.cpmorphology
import numpy
import scipy.ndimage


def measure(labels, img):
    """measureobjectintensity.py, MeasureObjectIntensity.run (the 2D part
    without a mask): the statistics per label 1..max."""
    nobjects = int(labels.max())
    lindexes = numpy.arange(nobjects) + 1
    integrated_intensity = numpy.zeros((nobjects,))
    mean_intensity = numpy.zeros((nobjects,))
    std_intensity = numpy.zeros((nobjects,))
    min_intensity = numpy.zeros((nobjects,))
    max_intensity = numpy.zeros((nobjects,))
    lower_quartile_intensity = numpy.zeros((nobjects,))
    median_intensity = numpy.zeros((nobjects,))
    mad_intensity = numpy.zeros((nobjects,))
    upper_quartile_intensity = numpy.zeros((nobjects,))
    dimensions = 2

    lmask = labels > 0
    limg = img[lmask]
    llabels = labels[lmask]

    lcount = centrosome.cpmorphology.fixup_scipy_ndimage_result(
        scipy.ndimage.sum(numpy.ones(len(limg)), llabels, lindexes)
    )
    integrated_intensity[lindexes - 1] = centrosome.cpmorphology.fixup_scipy_ndimage_result(
        scipy.ndimage.sum(limg, llabels, lindexes)
    )
    mean_intensity[lindexes - 1] = integrated_intensity[lindexes - 1] / lcount
    std_intensity[lindexes - 1] = numpy.sqrt(
        centrosome.cpmorphology.fixup_scipy_ndimage_result(
            scipy.ndimage.mean(
                (limg - mean_intensity[llabels - 1]) ** 2,
                llabels,
                lindexes,
            )
        )
    )
    min_intensity[lindexes - 1] = centrosome.cpmorphology.fixup_scipy_ndimage_result(
        scipy.ndimage.minimum(limg, llabels, lindexes)
    )
    max_intensity[lindexes - 1] = centrosome.cpmorphology.fixup_scipy_ndimage_result(
        scipy.ndimage.maximum(limg, llabels, lindexes)
    )

    #
    # Sort the intensities by label, then intensity.
    # For each label, find the index above and below
    # the 25%, 50% and 75% mark and take the weighted
    # average.
    #
    order = numpy.lexsort((limg, llabels))
    areas = lcount.astype(int)
    indices = numpy.cumsum(areas) - areas
    for dest, fraction in (
        (lower_quartile_intensity, 1.0 / 4.0),
        (median_intensity, 1.0 / 2.0),
        (upper_quartile_intensity, 3.0 / 4.0),
    ):
        qindex = indices.astype(float) + areas * fraction
        qfraction = qindex - numpy.floor(qindex)
        qindex = qindex.astype(int)
        qmask = qindex < indices + areas - 1
        qi = qindex[qmask]
        qf = qfraction[qmask]
        dest[lindexes[qmask] - 1] = limg[order[qi]] * (1 - qf) + limg[order[qi + 1]] * qf

        #
        # In some situations (e.g., only 3 points), there may
        # not be an upper bound.
        #
        qmask = (~qmask) & (areas > 0)
        dest[lindexes[qmask] - 1] = limg[order[qindex[qmask]]]

    #
    # Once again, for the MAD
    #
    madimg = numpy.abs(limg - median_intensity[llabels - 1])
    order = numpy.lexsort((madimg, llabels))
    qindex = indices.astype(float) + areas / dimensions
    qfraction = qindex - numpy.floor(qindex)
    qindex = qindex.astype(int)
    qmask = qindex < indices + areas - 1
    qi = qindex[qmask]
    qf = qfraction[qmask]
    mad_intensity[lindexes[qmask] - 1] = madimg[order[qi]] * (1 - qf) + madimg[order[qi + 1]] * qf
    qmask = (~qmask) & (areas > 0)
    mad_intensity[lindexes[qmask] - 1] = madimg[order[qindex[qmask]]]

    return {
        "integrated": integrated_intensity.tolist(),
        "mean": mean_intensity.tolist(),
        "std": std_intensity.tolist(),
        "min": min_intensity.tolist(),
        "max": max_intensity.tolist(),
        "lower_quartile": lower_quartile_intensity.tolist(),
        "median": median_intensity.tolist(),
        "upper_quartile": upper_quartile_intensity.tolist(),
        "mad": mad_intensity.tolist(),
    }


def blobs(rng, width, height, count, max_radius):
    """Random overlapping discs, labelled 1..count in painting order (a
    later disc paints over an earlier one); labels renumbered densely."""
    labels = numpy.zeros((height, width), dtype=numpy.int32)
    yy, xx = numpy.mgrid[0:height, 0:width]
    for label in range(1, count + 1):
        cx, cy = rng.uniform(0, width), rng.uniform(0, height)
        r = rng.uniform(0.6, max_radius)
        labels[(xx - cx) ** 2 + (yy - cy) ** 2 <= r * r] = label
    present = numpy.unique(labels[labels > 0])
    dense = numpy.zeros(count + 1, dtype=numpy.int32)
    dense[present] = numpy.arange(1, len(present) + 1)
    return dense[labels]


def case(name, labels, img32):
    img = img32.astype(numpy.float64)
    return {
        "name": name,
        "width": int(labels.shape[1]),
        "height": int(labels.shape[0]),
        "labels": labels.ravel().astype(int).tolist(),
        "image": img32.ravel().astype(float).tolist(),
        **measure(labels, img),
    }


def main():
    rng = numpy.random.default_rng(20261024)
    cases = []

    labels = blobs(rng, 64, 48, 40, 7.0)
    cases.append(case("random discs", labels, rng.random((48, 64), dtype=numpy.float32)))

    # Objects of 1, 2, 3, 4 and 5 pixels: the quantile's top end.
    labels = numpy.zeros((5, 12), dtype=numpy.int32)
    labels[0, 0] = 1
    labels[0, 2:4] = 2
    labels[0, 5:8] = 3
    labels[2, 0:4] = 4
    labels[4, 0:5] = 5
    cases.append(case("tiny objects", labels, rng.random((5, 12), dtype=numpy.float32)))

    # 8-bit values: many ties.
    labels = blobs(rng, 40, 40, 15, 9.0)
    img = (rng.integers(0, 6, (40, 40)) / 255.0).astype(numpy.float32)
    cases.append(case("ties", labels, img))

    with open("object_intensity_cellprofiler.json", "w") as f:
        json.dump(cases, f)
    print(f"{len(cases)} cases, {sum(len(c['mean']) for c in cases)} objects")


if __name__ == "__main__":
    main()
