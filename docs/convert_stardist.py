#!/usr/bin/env python3
# Copyright 2026 Joachim Danmayr.
# Licensed under the PolyForm Noncommercial License 1.0.0.
"""Convert a StarDist 2D model into EVAnalyzer's model file (dev tool).

EVAnalyzer runs StarDist 2D in its own implementation (crates/nn,
`model::stardist`) and reads only its own model files: safetensors with the
architecture in `__metadata__` (docs/nn_burn_migration_plan.md, D2). This
script converts a StarDist model folder (`config.json` + `weights_best.h5`,
e.g. the official `2D_versatile_fluo` from github.com/stardist/stardist-models)
once, by hand, through the `stardist` package (BSD-3):

  * every convolution as `<Keras layer name>.weight` `[out, in, kh, kw]`
    (PyTorch's layout) and `.bias`, f32 - the grid's convolutions (unnamed in
    Keras) as `grid.0`, `grid.1`, ...;
  * `__metadata__`: `evanalyzer_arch` = stardist2d, `arch_version` = 1,
    `config` = the architecture (crates/nn/src/model/stardist/config.rs, from
    StarDist's `config.json`), StarDist's thresholds (`prob_threshold`,
    `nms_threshold`), `source`, `license`, `citation`.

Only what EVAnalyzer implements is converted: one class, ReLU, no batch
norm, odd square kernels, pooling 2x2 - anything else is refused.

With `--references DIR` it also writes StarDist's own `predict` (TensorFlow,
CPU; its end-padding and crop, distances at least 1e-3) on synthetic cell
images for crates/nn's comparison test: `<case>.input.f32` and
`<case>.output.f32` (probability plane, then the ray planes, at the grid
cells StarDist keeps), raw little-endian f32, and `<case>.shape.json`.

    pip install tensorflow-cpu stardist safetensors
    python convert_stardist.py tests/stardist 2D_versatile_fluo \
        tests/stardist/2D_versatile_fluo.evanet --references tests/stardist/reference
"""

import argparse
import json
import os

import numpy as np
from safetensors.numpy import save_file


def cells(width: int, height: int) -> np.ndarray:
    """Bright discs of varying size and brightness on a textured background."""
    y, x = np.mgrid[0:height, 0:width].astype(np.float32)
    cx, cy = np.floor(x / 23.0), np.floor(y / 19.0)
    dx, dy = x - cx * 23.0 - 11.0, y - cy * 19.0 - 9.0
    radius = 4.0 + (cx * 3.0 + cy * 5.0) % 5.0
    img = np.where(dx * dx + dy * dy < radius * radius, 0.6 + ((cx + cy) % 4.0) * 0.1, 0.05)
    return (img + 0.04 * np.sin(x * 0.31) * np.cos(y * 0.17)).astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("basedir")
    parser.add_argument("name")
    parser.add_argument("output")
    parser.add_argument("--references")
    parser.add_argument("--source", default=None)
    args = parser.parse_args()

    from stardist.models import StarDist2D

    model = StarDist2D(None, name=args.name, basedir=args.basedir)
    c = model.config
    refuse = lambda what: SystemExit(f"not supported: {what}")
    if c.n_dim != 2 or c.backbone != "unet":
        raise refuse("only 2D U-Net StarDist")
    if getattr(c, "n_classes", None):
        raise refuse("multi-class models")
    if c.unet_batch_norm or c.unet_activation != "relu" or c.unet_last_activation != "relu":
        raise refuse("batch norm or another activation than relu")
    if list(c.unet_pool) != [2, 2] or c.unet_kernel_size[0] != c.unet_kernel_size[1] or c.unet_kernel_size[0] % 2 == 0:
        raise refuse("pooling other than 2x2 or a non-square / even kernel")
    config = {
        "in_channels": int(c.n_channel_in),
        "n_rays": int(c.n_rays),
        "grid": [int(g) for g in c.grid],
        "unet_depth": int(c.unet_n_depth),
        "unet_filter_base": int(c.unet_n_filter_base),
        "unet_conv_per_depth": int(c.unet_n_conv_per_depth),
        "kernel_size": int(c.unet_kernel_size[0]),
        "conv_after_unet": int(c.net_conv_after_unet),
    }

    tensors = {}
    grid = 0
    for layer in model.keras_model.layers:
        if type(layer).__name__ != "Conv2D":
            continue
        kernel, bias = layer.get_weights()
        if layer.name.startswith("conv2d"):
            name = f"grid.{grid}"
            grid += 1
        else:
            name = layer.name
        tensors[f"{name}.weight"] = np.ascontiguousarray(kernel.transpose(3, 2, 0, 1), dtype=np.float32)
        tensors[f"{name}.bias"] = np.ascontiguousarray(bias, dtype=np.float32)
    metadata = {
        "evanalyzer_arch": "stardist2d",
        "arch_version": "1",
        "config": json.dumps(config, separators=(",", ":")),
        "prob_threshold": repr(float(model.thresholds.prob)),
        "nms_threshold": repr(float(model.thresholds.nms)),
        "source": args.source or f"github.com/stardist/stardist-models {args.name}",
        "license": "BSD-3-Clause (StarDist)",
        "citation": "Schmidt, Weigert, Broaddus, Myers (2018) Cell Detection with Star-convex Polygons, MICCAI; Weigert, Schmidt, Haase, Sugawara, Myers (2020) Star-convex Polyhedra for 3D Object Detection and Segmentation in Microscopy, WACV",
    }
    save_file(tensors, args.output, metadata=metadata)
    print(f"wrote {args.output}: {len(tensors)} tensors, {config}")

    if args.references:
        os.makedirs(args.references, exist_ok=True)
        for name, (w, h) in {
            "cells256": (256, 256),
            "cells101x77": (101, 77),
            "cells512x384": (512, 384),
            "cells1100x600": (1100, 600),
        }.items():
            img = cells(w, h)
            prob, dist = model.predict(img)
            out = np.concatenate([prob[None], dist.transpose(2, 0, 1)]).astype("<f4")
            img.astype("<f4").tofile(os.path.join(args.references, f"{name}.input.f32"))
            out.tofile(os.path.join(args.references, f"{name}.output.f32"))
            with open(os.path.join(args.references, f"{name}.shape.json"), "w") as f:
                json.dump({"width": w, "height": h, "grid_height": prob.shape[0], "grid_width": prob.shape[1]}, f)
            print(f"{name}: {w}x{h} -> grid {prob.shape[1]}x{prob.shape[0]}, {dist.shape[2]} rays")


if __name__ == "__main__":
    main()
