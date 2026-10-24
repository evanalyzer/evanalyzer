"""Writes colocalization_cellprofiler.json: CellProfiler's MeasureColocalization
values - over the whole image and per object: correlation, slope, Manders,
overlap, K, RWC and Manders with Costes' thresholds (the arithmetic of
run_image_pair_images, run_image_pair_objects, linear_costes and
bisection_costes in cellprofiler/modules/measurecolocalization.py,
CellProfiler 4.2.8, copied below without the module plumbing) - for a few
pairs of images, to check EVA's Measure Colocalization against.

Pixel values are float32 (as EVA reads them), computed in float64 like
CellProfiler does.

Run with numpy, scipy and centrosome installed:
    python colocalization_cellprofiler.py
"""

import json

import numpy
import scipy.ndimage
import scipy.stats
from centrosome.cpmorphology import fixup_scipy_ndimage_result as fix
from numpy.linalg import lstsq

M_FAST = "Fast"
M_FASTER = "Faster"
M_ACCURATE = "Accurate"


def linear_costes(fi, si, scale_max, fast_costes):
    """
    Finds the Costes Automatic Threshold for colocalization using a linear algorithm.
    Candiate thresholds are gradually decreased until Pearson R falls below 0.
    If "Fast" mode is enabled the "steps" between tested thresholds will be increased
    when Pearson R is much greater than 0.
    """
    i_step = 1 / scale_max
    non_zero = (fi > 0) | (si > 0)
    xvar = numpy.var(fi[non_zero], axis=0, ddof=1)
    yvar = numpy.var(si[non_zero], axis=0, ddof=1)

    xmean = numpy.mean(fi[non_zero], axis=0)
    ymean = numpy.mean(si[non_zero], axis=0)

    z = fi[non_zero] + si[non_zero]
    zvar = numpy.var(z, axis=0, ddof=1)

    covar = 0.5 * (zvar - (xvar + yvar))

    denom = 2 * covar
    num = (yvar - xvar) + numpy.sqrt(
        (yvar - xvar) * (yvar - xvar) + 4 * (covar * covar)
    )
    a = num / denom
    b = ymean - a * xmean

    # Start at 1 step above the maximum value
    img_max = max(fi.max(), si.max())
    i = i_step * ((img_max // i_step) + 1)

    num_true = None
    fi_max = fi.max()
    si_max = si.max()

    # Initialise without a threshold
    costReg, _ = scipy.stats.pearsonr(fi, si)
    thr_fi_c = i
    thr_si_c = (a * i) + b
    while i > fi_max and (a * i) + b > si_max:
        i -= i_step
    while i > i_step:
        thr_fi_c = i
        thr_si_c = (a * i) + b
        combt = (fi < thr_fi_c) | (si < thr_si_c)
        try:
            # Only run pearsonr if the input has changed.
            if (positives := numpy.count_nonzero(combt)) != num_true:
                costReg, _ = scipy.stats.pearsonr(fi[combt], si[combt])
                num_true = positives

            if costReg <= 0:
                break
            elif fast_costes == M_ACCURATE or i < i_step * 10:
                i -= i_step
            elif costReg > 0.45:
                # We're way off, step down 10x
                i -= i_step * 10
            elif costReg > 0.35:
                # Still far from 0, step 5x
                i -= i_step * 5
            elif costReg > 0.25:
                # Step 2x
                i -= i_step * 2
            else:
                i -= i_step
        except ValueError:
            break
    return thr_fi_c, thr_si_c


def bisection_costes(fi, si, scale_max):
    """
    Finds the Costes Automatic Threshold for colocalization using a bisection algorithm.
    Candidate thresholds are selected from within a window of possible intensities,
    this window is narrowed based on the R value of each tested candidate.
    We're looking for the first point below 0, and R value can become highly variable
    at lower thresholds in some samples. Therefore the candidate tested in each
    loop is 1/6th of the window size below the maximum value (as opposed to the midpoint).
    """

    non_zero = (fi > 0) | (si > 0)
    xvar = numpy.var(fi[non_zero], axis=0, ddof=1)
    yvar = numpy.var(si[non_zero], axis=0, ddof=1)

    xmean = numpy.mean(fi[non_zero], axis=0)
    ymean = numpy.mean(si[non_zero], axis=0)

    z = fi[non_zero] + si[non_zero]
    zvar = numpy.var(z, axis=0, ddof=1)

    covar = 0.5 * (zvar - (xvar + yvar))

    denom = 2 * covar
    num = (yvar - xvar) + numpy.sqrt(
        (yvar - xvar) * (yvar - xvar) + 4 * (covar * covar)
    )
    a = num / denom
    b = ymean - a * xmean

    # Initialise variables
    left = 1
    right = scale_max
    mid = ((right - left) // (6 / 5)) + left
    lastmid = 0
    # Marks the value with the last positive R value.
    valid = 1

    while lastmid != mid:
        thr_fi_c = mid / scale_max
        thr_si_c = (a * thr_fi_c) + b
        combt = (fi < thr_fi_c) | (si < thr_si_c)
        if numpy.count_nonzero(combt) <= 2:
            # Can't run pearson with only 2 values.
            left = mid - 1
        else:
            try:
                costReg, _ = scipy.stats.pearsonr(fi[combt], si[combt])
                if costReg < 0:
                    left = mid - 1
                elif costReg >= 0:
                    right = mid + 1
                    valid = mid
            except ValueError:
                # Catch misc Pearson errors with low sample numbers
                left = mid - 1
        lastmid = mid
        if right - left > 6:
            mid = ((right - left) // (6 / 5)) + left
        else:
            mid = ((right - left) // 2) + left

    thr_fi_c = (valid - 1) / scale_max
    thr_si_c = (a * thr_fi_c) + b

    return thr_fi_c, thr_si_c


def costes_thresholds(fi, si, scale, fast_costes):
    if fast_costes == M_FASTER:
        return bisection_costes(fi, si, scale)
    return linear_costes(fi, si, scale, fast_costes)


def image_values(fi, si, thr, scale, fast_costes):
    """run_image_pair_images: the whole image's values."""
    corr = numpy.corrcoef((fi, si))[1, 0]
    coeffs = lstsq(numpy.array((fi, numpy.ones_like(fi))).transpose(), si, rcond=None)[0]
    slope = coeffs[0]

    # Threshold as percentage of maximum intensity in each channel
    thr_fi = thr * numpy.max(fi) / 100
    thr_si = thr * numpy.max(si) / 100
    combined_thresh = (fi > thr_fi) & (si > thr_si)
    fi_thresh = fi[combined_thresh]
    si_thresh = si[combined_thresh]
    tot_fi_thr = fi[(fi > thr_fi)].sum()
    tot_si_thr = si[(si > thr_si)].sum()

    M1 = fi_thresh.sum() / tot_fi_thr
    M2 = si_thresh.sum() / tot_si_thr

    Rank1 = numpy.lexsort([fi])
    Rank2 = numpy.lexsort([si])
    Rank1_U = numpy.hstack([[False], fi[Rank1[:-1]] != fi[Rank1[1:]]])
    Rank2_U = numpy.hstack([[False], si[Rank2[:-1]] != si[Rank2[1:]]])
    Rank1_S = numpy.cumsum(Rank1_U)
    Rank2_S = numpy.cumsum(Rank2_U)
    Rank_im1 = numpy.zeros(fi.shape, dtype=int)
    Rank_im2 = numpy.zeros(si.shape, dtype=int)
    Rank_im1[Rank1] = Rank1_S
    Rank_im2[Rank2] = Rank2_S

    R = max(Rank_im1.max(), Rank_im2.max()) + 1
    Di = abs(Rank_im1 - Rank_im2)
    weight = ((R - Di) * 1.0) / R
    weight_thresh = weight[combined_thresh]
    RWC1 = (fi_thresh * weight_thresh).sum() / tot_fi_thr
    RWC2 = (si_thresh * weight_thresh).sum() / tot_si_thr

    overlap = (fi_thresh * si_thresh).sum() / numpy.sqrt(
        (fi_thresh ** 2).sum() * (si_thresh ** 2).sum()
    )
    K1 = (fi_thresh * si_thresh).sum() / (fi_thresh ** 2).sum()
    K2 = (fi_thresh * si_thresh).sum() / (si_thresh ** 2).sum()

    thr_fi_c, thr_si_c = costes_thresholds(fi, si, scale, fast_costes)
    combined_thresh_c = (fi > thr_fi_c) & (si > thr_si_c)
    fi_thresh_c = fi[combined_thresh_c]
    si_thresh_c = si[combined_thresh_c]
    tot_fi_thr_c = fi[(fi > thr_fi_c)].sum()
    tot_si_thr_c = si[(si > thr_si_c)].sum()
    C1 = fi_thresh_c.sum() / tot_fi_thr_c
    C2 = si_thresh_c.sum() / tot_si_thr_c

    return [corr, slope, M1, M2, overlap, K1, K2, RWC1, RWC2, C1, C2]


def object_values(first_pixels, second_pixels, labels, n_objects, thr, fi, si, scale, fast_costes):
    """run_image_pair_objects: per object (labels 1..n_objects) - the
    pixels of all objects with their labels; fi, si the whole image's."""
    lrange = numpy.arange(n_objects, dtype=numpy.int32) + 1

    mean1 = fix(scipy.ndimage.mean(first_pixels, labels, lrange))
    mean2 = fix(scipy.ndimage.mean(second_pixels, labels, lrange))
    std1 = numpy.sqrt(fix(scipy.ndimage.sum((first_pixels - mean1[labels - 1]) ** 2, labels, lrange)))
    std2 = numpy.sqrt(fix(scipy.ndimage.sum((second_pixels - mean2[labels - 1]) ** 2, labels, lrange)))
    x = first_pixels - mean1[labels - 1]  # x - mean(x)
    y = second_pixels - mean2[labels - 1]  # y - mean(y)
    corr = fix(scipy.ndimage.sum(x * y / (std1[labels - 1] * std2[labels - 1]), labels, lrange))
    corr[scipy.ndimage.sum(1, labels, lrange) == 0] = numpy.nan  # numpy.NaN in CellProfiler (numpy 1)

    tff = (thr / 100) * fix(scipy.ndimage.maximum(first_pixels, labels, lrange))
    tss = (thr / 100) * fix(scipy.ndimage.maximum(second_pixels, labels, lrange))
    combined_thresh = (first_pixels >= tff[labels - 1]) & (second_pixels >= tss[labels - 1])
    fi_thresh = first_pixels[combined_thresh]
    si_thresh = second_pixels[combined_thresh]
    tot_fi_thr = scipy.ndimage.sum(
        first_pixels[first_pixels >= tff[labels - 1]],
        labels[first_pixels >= tff[labels - 1]],
        lrange,
    )
    tot_si_thr = scipy.ndimage.sum(
        second_pixels[second_pixels >= tss[labels - 1]],
        labels[second_pixels >= tss[labels - 1]],
        lrange,
    )

    M1 = numpy.zeros(len(lrange))
    M2 = numpy.zeros(len(lrange))
    if numpy.any(combined_thresh):
        M1 = numpy.array(scipy.ndimage.sum(fi_thresh, labels[combined_thresh], lrange)) / numpy.array(tot_fi_thr)
        M2 = numpy.array(scipy.ndimage.sum(si_thresh, labels[combined_thresh], lrange)) / numpy.array(tot_si_thr)

    RWC1 = numpy.zeros(len(lrange))
    RWC2 = numpy.zeros(len(lrange))
    [Rank1] = numpy.lexsort(([labels], [first_pixels]))
    [Rank2] = numpy.lexsort(([labels], [second_pixels]))
    Rank1_U = numpy.hstack([[False], first_pixels[Rank1[:-1]] != first_pixels[Rank1[1:]]])
    Rank2_U = numpy.hstack([[False], second_pixels[Rank2[:-1]] != second_pixels[Rank2[1:]]])
    Rank1_S = numpy.cumsum(Rank1_U)
    Rank2_S = numpy.cumsum(Rank2_U)
    Rank_im1 = numpy.zeros(first_pixels.shape, dtype=int)
    Rank_im2 = numpy.zeros(second_pixels.shape, dtype=int)
    Rank_im1[Rank1] = Rank1_S
    Rank_im2[Rank2] = Rank2_S
    R = max(Rank_im1.max(), Rank_im2.max()) + 1
    Di = abs(Rank_im1 - Rank_im2)
    weight = (R - Di) * 1.0 / R
    weight_thresh = weight[combined_thresh]
    if numpy.any(combined_thresh):
        RWC1 = numpy.array(scipy.ndimage.sum(fi_thresh * weight_thresh, labels[combined_thresh], lrange)) / numpy.array(tot_fi_thr)
        RWC2 = numpy.array(scipy.ndimage.sum(si_thresh * weight_thresh, labels[combined_thresh], lrange)) / numpy.array(tot_si_thr)

    if numpy.any(combined_thresh):
        fpsq = scipy.ndimage.sum(first_pixels[combined_thresh] ** 2, labels[combined_thresh], lrange)
        spsq = scipy.ndimage.sum(second_pixels[combined_thresh] ** 2, labels[combined_thresh], lrange)
        pdt = numpy.sqrt(numpy.array(fpsq) * numpy.array(spsq))
        overlap = fix(scipy.ndimage.sum(first_pixels[combined_thresh] * second_pixels[combined_thresh], labels[combined_thresh], lrange) / pdt)
        K1 = fix((scipy.ndimage.sum(first_pixels[combined_thresh] * second_pixels[combined_thresh], labels[combined_thresh], lrange)) / (numpy.array(fpsq)))
        K2 = fix(scipy.ndimage.sum(first_pixels[combined_thresh] * second_pixels[combined_thresh], labels[combined_thresh], lrange) / numpy.array(spsq))
    else:
        overlap = K1 = K2 = numpy.zeros(len(lrange))

    thr_fi_c, thr_si_c = costes_thresholds(fi, si, scale, fast_costes)
    fi_above_thr = first_pixels > thr_fi_c
    si_above_thr = second_pixels > thr_si_c
    combined_thresh_c = fi_above_thr & si_above_thr
    fi_thresh_c = first_pixels[combined_thresh_c]
    si_thresh_c = second_pixels[combined_thresh_c]
    if numpy.any(fi_above_thr):
        tot_fi_thr_c = scipy.ndimage.sum(first_pixels[first_pixels >= thr_fi_c], labels[first_pixels >= thr_fi_c], lrange)
    else:
        tot_fi_thr_c = numpy.zeros(len(lrange))
    if numpy.any(si_above_thr):
        tot_si_thr_c = scipy.ndimage.sum(second_pixels[second_pixels >= thr_si_c], labels[second_pixels >= thr_si_c], lrange)
    else:
        tot_si_thr_c = numpy.zeros(len(lrange))
    C1 = numpy.zeros(len(lrange))
    C2 = numpy.zeros(len(lrange))
    if numpy.any(combined_thresh_c):
        C1 = numpy.array(scipy.ndimage.sum(fi_thresh_c, labels[combined_thresh_c], lrange)) / numpy.array(tot_fi_thr_c)
        C2 = numpy.array(scipy.ndimage.sum(si_thresh_c, labels[combined_thresh_c], lrange)) / numpy.array(tot_si_thr_c)

    slope = numpy.full(n_objects, numpy.nan)  # none per object
    return numpy.array([corr, slope, M1, M2, overlap, K1, K2, RWC1, RWC2, C1, C2]).T


def discs(rng, width, height, count, max_radius):
    """Non-touching random discs, labelled 1..n."""
    labels = numpy.zeros((height, width), dtype=numpy.int32)
    yy, xx = numpy.mgrid[0:height, 0:width]
    n = 0
    for _ in range(count):
        cx, cy = rng.uniform(0, width), rng.uniform(0, height)
        r = rng.uniform(1.5, max_radius)
        disc = (xx - cx) ** 2 + (yy - cy) ** 2 <= r * r
        if not disc.any() or (labels[scipy.ndimage.binary_dilation(disc)] != 0).any():
            continue
        n += 1
        labels[disc] = n
    return labels, n


def channels(rng, labels, levels, coupling):
    """Two channels on `levels` gray levels: bright in the objects, the
    second following the first by `coupling` (negative: against), noise."""
    h, w = labels.shape
    base = scipy.ndimage.gaussian_filter((labels > 0).astype(float) * rng.uniform(0.3, 1.0, labels.shape), 1.5)
    a = base + 0.15 * rng.random((h, w))
    b = coupling * base + 0.5 * (1 - abs(coupling)) * rng.random((h, w)) + 0.1 * rng.random((h, w))
    b = b - b.min()
    out = []
    for c in (a, b):
        c = c / c.max() * (levels * 0.9)
        out.append((numpy.round(c) / levels).astype(numpy.float32))
    return out


def case(name, labels, n, a32, b32, thr, scale, fast_costes):
    fi = a32.astype(numpy.float64).ravel()
    si = b32.astype(numpy.float64).ravel()
    mask = labels > 0
    objects = object_values(
        a32.astype(numpy.float64)[mask], b32.astype(numpy.float64)[mask], labels[mask], n,
        thr, fi, si, scale, fast_costes,
    )
    return {
        "name": name,
        "width": int(labels.shape[1]),
        "height": int(labels.shape[0]),
        "threshold_percent": thr,
        "scale": scale,
        "costes": fast_costes,
        "labels": labels.ravel().astype(int).tolist(),
        "a": a32.ravel().astype(float).tolist(),
        "b": b32.ravel().astype(float).tolist(),
        "image": [None if not numpy.isfinite(v) else float(v) for v in image_values(fi, si, thr, scale, fast_costes)],
        "objects": [[None if not numpy.isfinite(v) else float(v) for v in row] for row in objects],
    }


def main():
    rng = numpy.random.default_rng(20261024)
    cases = []
    labels, n = discs(rng, 64, 48, 30, 6.0)
    a, b = channels(rng, labels, 255, 0.8)
    for mode in (M_FASTER, M_FAST, M_ACCURATE):
        cases.append(case(f"8-bit together, {mode}", labels, n, a, b, 15.0, 255, mode))
    cases.append(case("8-bit together, 30 %", labels, n, a, b, 30.0, 255, M_FASTER))
    a, b = channels(rng, labels, 255, -0.6)
    cases.append(case("8-bit against, Fast", labels, n, a, b, 15.0, 255, M_FAST))
    labels, n = discs(rng, 50, 40, 20, 7.0)
    a, b = channels(rng, labels, 65535 // 16, 0.5)
    cases.append(case("16-bit, Faster", labels, n, a, b, 15.0, 65535, M_FASTER))
    with open("colocalization_cellprofiler.json", "w") as f:
        json.dump(cases, f)
    print(f"{len(cases)} cases, {sum(len(c['objects']) for c in cases)} objects")


if __name__ == "__main__":
    main()
