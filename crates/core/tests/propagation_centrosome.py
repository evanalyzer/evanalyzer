"""Writes propagation_centrosome.json: CellProfiler's Propagation results
(centrosome.propagate) for a few images, to check EVA's port against.

Run with numpy and centrosome installed:
    python propagation_centrosome.py
"""

import json

import numpy as np
from centrosome.propagate import propagate


def cells_image(rng, height, width, count, eight_bit):
    """Gaussian "cells" plus noise, 0..1, float32 (8-bit steps if asked)."""
    yy, xx = np.mgrid[0:height, 0:width]
    centers = rng.uniform((4, 4), (height - 4, width - 4), size=(count, 2))
    image = np.zeros((height, width))
    for cy, cx in centers:
        sigma = rng.uniform(3.0, 6.0)
        image += np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * sigma**2))
    image += rng.normal(0, 0.05, image.shape)
    image = np.clip(image / image.max(), 0, 1)
    if eight_bit:
        image = np.round(image * 255) / 255
    return image.astype(np.float32), centers


def seeds_at(centers, height, width):
    """A 3x3 seed per center (later ones don't overwrite earlier ones)."""
    seeds = np.zeros((height, width), np.int32)
    for label, (cy, cx) in enumerate(centers, start=1):
        cy, cx = int(cy), int(cx)
        window = seeds[cy - 1 : cy + 2, cx - 1 : cx + 2]
        window[window == 0] = label
    return seeds


cases = []
for name, seed, eight_bit, masked, weight in [
    ("continuous", 1, False, False, 0.05),
    ("eight_bit", 2, True, False, 0.05),
    ("eight_bit_masked", 3, True, True, 0.05),
    ("continuous_masked_distance", 4, False, True, 1.0),
]:
    rng = np.random.default_rng(seed)
    height, width = 40, 56
    image, centers = cells_image(rng, height, width, 7, eight_bit)
    seeds = seeds_at(centers, height, width)
    mask = image > 0.15 if masked else np.ones(image.shape, bool)
    labels, _ = propagate(image.astype(np.float64), seeds, mask, weight)
    cases.append(
        {
            "name": name,
            "width": width,
            "height": height,
            "weight": weight,
            "image": [float(v) for v in image.ravel()],
            "seeds": [int(v) for v in seeds.ravel()],
            "mask": [bool(v) for v in mask.ravel()],
            "expected": [int(v) for v in labels.ravel()],
        }
    )

# Exact ties: a flat image, the column halfway between two seeds is as
# cheap from both - who gets it shows the order of equally cheap pixels.
height, width = 12, 20
image = np.zeros((height, width), np.float32)
seeds = np.zeros((height, width), np.int32)
seeds[5, 5] = 1
seeds[5, 11] = 2
labels, _ = propagate(image.astype(np.float64), seeds, np.ones(image.shape, bool), 1.0)
cases.append(
    {
        "name": "flat_ties",
        "width": width,
        "height": height,
        "weight": 1.0,
        "image": [float(v) for v in image.ravel()],
        "seeds": [int(v) for v in seeds.ravel()],
        "mask": [True] * (height * width),
        "expected": [int(v) for v in labels.ravel()],
    }
)

with open("propagation_centrosome.json", "w") as f:
    json.dump({"source": "centrosome.propagate", "cases": cases}, f)
