"""Writes image_quality_cellprofiler.json: CellProfiler's image quality
values (MeasureImageQuality: focus score, power log-log slope, percent
maximal/minimal - the code of cellprofiler_library/functions/measurement.py
with centrosome's radial power spectrum) and Immerkær's noise estimate, for
a few images, to check EVA's Image Quality against.

Run with numpy, scipy and centrosome installed:
    python image_quality_cellprofiler.py
"""

import json

import numpy as np
import scipy.linalg
import scipy.ndimage
import scipy.signal
from centrosome.radial_power_spectrum import rps


def focus_score(pixel_data):
    """CellProfiler's get_focus_score_for_scale_group (the whole-image part)."""
    mean_image_value = np.mean(pixel_data)
    squared_normalized_image = (pixel_data - mean_image_value) ** 2
    if mean_image_value > 0:
        return np.sum(squared_normalized_image) / (np.prod(pixel_data.shape) * mean_image_value)
    return 0.0


def saturation(pixel_data):
    """CellProfiler's get_saturation_value: percent minimal, percent maximal."""
    pixel_count = np.prod(pixel_data.shape)
    maximal = np.sum(pixel_data == np.max(pixel_data))
    minimal = np.sum(pixel_data == np.min(pixel_data))
    return 100.0 * minimal / pixel_count, 100.0 * maximal / pixel_count


class WithPtp(np.ndarray):
    """centrosome's rps calls ndarray.ptp, which numpy 2 removed."""

    def ptp(self, *args, **kwargs):
        return np.ptp(np.asarray(self), *args, **kwargs)


def power_slope(pixel_data):
    """CellProfiler's get_power_spectrum_measurement_values."""
    radii, magnitude, power = rps(pixel_data.view(WithPtp))
    if sum(magnitude) > 0 and len(np.unique(pixel_data)) > 1:
        valid = magnitude > 0
        radii = radii[valid].reshape((-1, 1))
        power = power[valid].reshape((-1, 1))
        if radii.shape[0] > 1:
            idx = np.isfinite(np.log(power))
            return float(
                scipy.linalg.lstsq(
                    np.hstack(
                        (
                            np.log(radii)[idx][:, np.newaxis],
                            np.ones(radii.shape)[idx][:, np.newaxis],
                        )
                    ),
                    np.log(power)[idx][:, np.newaxis],
                )[0][0][0]
            )
    return None


def noise(pixel_data):
    """Immerkær 1996: sqrt(pi/2) / (6 (W-2)(H-2)) * sum |I * N|."""
    mask = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], float)
    response = scipy.signal.convolve2d(pixel_data, mask, mode="valid")
    h, w = pixel_data.shape
    return float(np.sqrt(np.pi / 2) * np.sum(np.abs(response)) / (6 * (w - 2) * (h - 2)))


def cells(rng, height, width):
    yy, xx = np.mgrid[0:height, 0:width]
    image = np.zeros((height, width))
    for cy, cx in rng.uniform((0, 0), (height, width), size=(25, 2)):
        image += np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * rng.uniform(1.5, 4.0) ** 2))
    return image


cases = []
rng = np.random.default_rng(7)
sharp = cells(rng, 96, 128) + rng.normal(0, 0.05, (96, 128))
sharp = np.clip(sharp / sharp.max(), 0, 1)
images = {
    "sharp": sharp,
    "blurred": scipy.ndimage.gaussian_filter(sharp, 3.0),
    # 8-bit steps, a third of it clipped at the top: percent maximal.
    "saturated_8_bit": np.round(np.clip(sharp * 3, 0, 1) * 255) / 255,
    "odd_size": np.clip(cells(rng, 70, 90) / 2 + rng.normal(0, 0.1, (70, 90)), 0, 1),
}
for name, image in images.items():
    image = image.astype(np.float32).astype(np.float64)
    percent_minimal, percent_maximal = saturation(image)
    cases.append(
        {
            "name": name,
            "width": image.shape[1],
            "height": image.shape[0],
            "image": [float(v) for v in image.ravel()],
            "focus_score": float(focus_score(image)),
            "power_log_log_slope": power_slope(image),
            "percent_maximal": float(percent_maximal),
            "percent_minimal": float(percent_minimal),
            "noise": noise(image),
        }
    )

with open("image_quality_cellprofiler.json", "w") as f:
    json.dump({"source": "CellProfiler MeasureImageQuality, centrosome rps", "cases": cases}, f)
