"""Writes texture_cellprofiler.json: CellProfiler's MeasureTexture values
(MeasureTexture.run_image and run_one in cellprofiler/modules/measuretexture.py,
CellProfiler 4.2.8, copied below for 2D: the image as 8 bits, rescaled to
the gray levels, mahotas' Haralick features per direction - over the whole
image, and per object with the pixels outside it and the zeros ignored) for
a 16-bit image with textured objects - to check EVA's texture values against.

Run with numpy, scikit-image and mahotas installed:
    python texture_cellprofiler.py
"""

import json

import mahotas.features
import numpy
import scipy.ndimage
import skimage.exposure
import skimage.measure
import skimage.util


def as_levels(image, gray_levels):
    """The image as CellProfiler loads it (0..1, float32) in 8 bits, rescaled
    to 0..gray_levels-1."""
    pixel_data = skimage.util.img_as_ubyte(image.astype(numpy.float32) / 65535)
    if gray_levels != 256:
        pixel_data = skimage.exposure.rescale_intensity(
            pixel_data, in_range=(0, 255), out_range=(0, gray_levels - 1)
        ).astype(numpy.uint8)
    return pixel_data


def run_image(image, scale, gray_levels):
    """run_image: 4 directions x 13 features (NaN: none)."""
    try:
        return mahotas.features.haralick(as_levels(image, gray_levels), distance=scale)
    except ValueError:
        return numpy.full((4, 13), numpy.nan)


def run_one(image, labels, scale, gray_levels):
    """run_one: objects x 4 directions x 13 features (NaN where mahotas
    gives up; CellProfiler then stores 0)."""
    pixel_data = as_levels(image, gray_levels)
    props = skimage.measure.regionprops(labels, pixel_data)
    features = numpy.empty((len(props), 4, 13))
    for index, prop in enumerate(props):
        label_data = prop["intensity_image"]
        try:
            features[index] = mahotas.features.haralick(
                label_data, distance=scale, ignore_zeros=True
            )
        except ValueError:
            features[index] = numpy.nan
    return features


def scene(rng, width, height):
    """Textured objects that don't touch on a dim background: discs and
    bars filled with stripes, speckles or smooth noise; plus lines and
    single pixels (no pairs) and a dark object (zeros); labelled 1..n."""
    image = rng.normal(800, 150, (height, width)).clip(0, 65535)
    labels = numpy.zeros((height, width), numpy.int32)
    yy, xx = numpy.mgrid[0:height, 0:width]
    smooth = scipy.ndimage.gaussian_filter(rng.normal(0, 1, (height, width)), 2)
    smooth /= smooth.std()
    n = 0
    for k in range(80):
        cx, cy = rng.uniform(6, width - 6), rng.uniform(6, height - 6)
        kind = k % 6
        if kind in (0, 1):
            r = rng.uniform(3, 9)
            shape = (xx - cx) ** 2 + (yy - cy) ** 2 <= r * r
        elif kind == 2:
            a = rng.uniform(0, numpy.pi)
            u = (xx - cx) * numpy.cos(a) + (yy - cy) * numpy.sin(a)
            v = -(xx - cx) * numpy.sin(a) + (yy - cy) * numpy.cos(a)
            shape = (abs(u) <= rng.uniform(4, 10)) & (abs(v) <= rng.uniform(1.5, 4))
        elif kind == 3:
            shape = (xx == int(cx)) & (yy >= int(cy)) & (yy < int(cy) + int(rng.integers(1, 6)))
        elif kind == 4:
            r = rng.uniform(2, 5)
            shape = (xx - cx) ** 2 + (yy - cy) ** 2 <= r * r
        else:
            shape = (xx - cx) ** 2 + (yy - cy) ** 2 <= rng.uniform(2, 6) ** 2
        if not shape.any() or (labels[scipy.ndimage.binary_dilation(shape, iterations=2)] != 0).any():
            continue
        pieces, count = scipy.ndimage.label(shape, structure=numpy.ones((3, 3)))
        if count != 1:
            continue
        n += 1
        labels[shape] = n
        base = rng.uniform(5000, 40000)
        texture = n % 4
        if texture == 0:
            period = rng.uniform(2, 6)
            values = base * (1 + 0.5 * numpy.sin(2 * numpy.pi * (xx + yy * rng.uniform(-1, 1)) / period))
        elif texture == 1:
            values = base * (1 + 0.4 * rng.normal(0, 1, (height, width)))
        elif texture == 2:
            values = base * (1 + 0.5 * smooth)
        else:
            values = numpy.full((height, width), base)
        if k % 13 == 5:
            values = rng.uniform(0, 120, (height, width))  # rounds to 0 in 8 bits
        image[shape] = values[shape]
    return numpy.rint(image.clip(0, 65535)).astype(numpy.uint16), labels


def nulls(values):
    return [v if numpy.isfinite(v) else None for v in numpy.asarray(values).ravel().tolist()]


def main():
    rng = numpy.random.default_rng(20261024)
    image, labels = scene(rng, 130, 100)
    cases = []
    for scale, gray_levels in [(3, 256), (1, 8), (2, 64)]:
        cases.append(
            {
                "width": int(image.shape[1]),
                "height": int(image.shape[0]),
                "scale": scale,
                "gray_levels": gray_levels,
                "image": image.ravel().astype(int).tolist(),
                "labels": labels.ravel().astype(int).tolist(),
                # direction-major, 13 features each
                "image_features": nulls(run_image(image, scale, gray_levels)),
                # object-major, then direction, then feature
                "object_features": nulls(run_one(image, labels, scale, gray_levels)),
            }
        )
    with open("texture_cellprofiler.json", "w") as f:
        json.dump(cases, f)
    print(f"{labels.max()} objects")


if __name__ == "__main__":
    main()
