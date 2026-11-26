#!/usr/bin/env python3
# Copyright 2026 Joachim Danmayr.
# Licensed under the PolyForm Noncommercial License 1.0.0.
"""Convert a Cellpose-SAM checkpoint into EVAnalyzer's model file (dev tool).

EVAnalyzer runs Cellpose-SAM in its own implementation (crates/nn,
`model::cellpose_sam`) and reads only its own model files: safetensors with
the architecture in `__metadata__` (docs/nn_burn_migration_plan.md, D2). This
script converts a Cellpose-SAM state dict once, by hand:

  * the tensors keep Cellpose's PyTorch names (`encoder.blocks.0.attn.qkv.weight`,
    ...), as f32;
  * the relative-position tables (`rel_pos_h`/`rel_pos_w`) keep SAM's sizes
    (27 rows in the windowed blocks, 127 in the global ones) - the file loads
    into Cellpose's own `CPSAM` unchanged; crates/nn resizes them to the 32 x 32
    token grid of a 256 x 256 tile once on loading, as SAM's `get_rel_pos` does
    at every forward pass;
  * `W2` (a fixed identity: the output's pixel shuffle) is checked and left out,
    as are `diam_labels` / `diam_mean` (not used by the network);
  * `__metadata__`: `evanalyzer_arch` = cellpose_sam, `arch_version` = 1,
    `config` = the architecture (crates/nn/src/model/cellpose_sam/config.rs),
    plus `source`, `license`, `citation`.

With `--references DIR` it also writes the network's output on fixed inputs
(Cellpose's own `CPSAM` module, f32, CPU) for crates/nn's comparison test:
`tile{i}.input.f32` ([C, 256, 256]) and `tile{i}.output.f32` ([3, 256, 256],
dY, dX, cell logit), raw little-endian f32.

    pip install "cellpose>=4" torch safetensors
    python convert_cellpose_sam.py tests/cellpose-sam/cpsam_v2 \
        tests/cellpose-sam/cpsam_v2.evanet --references tests/cellpose-sam/reference
"""

import argparse
import json
import os

import numpy as np
import torch
from safetensors.torch import save_file

CONFIG = {
    "in_channels": 3,
    "tile_size": 256,
    "patch_size": 8,
    "embed_dim": 1024,
    "depth": 24,
    "num_heads": 16,
    "mlp_dim": 4096,
    "neck_dim": 256,
    "out_channels": 3,
}


def cells(side: int, seed: int) -> np.ndarray:
    """Bright discs of varying size on a textured background, normalized to
    about 0..1 (as Cellpose's percentile normalization would)."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:side, 0:side].astype(np.float32)
    img = 0.05 * np.sin(x * 0.31) * np.cos(y * 0.17) + 0.1
    for _ in range(40):
        cy, cx = rng.uniform(0, side, 2)
        r = rng.uniform(6, 18)
        img += 0.8 * ((x - cx) ** 2 + (y - cy) ** 2 < r * r)
    return np.clip(img, 0, 1.2).astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("output")
    parser.add_argument("--references")
    parser.add_argument("--source", default="huggingface.co/mouseland/cellpose-sam cpsam_v2")
    args = parser.parse_args()

    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    state = {k[7:] if k.startswith("module.") else k: v for k, v in state.items()}
    w2 = state.pop("W2")
    eye = torch.eye(w2.shape[0]).reshape(w2.shape)
    assert torch.equal(w2.float(), eye), "W2 is not the identity pixel shuffle"
    for key in ("diam_labels", "diam_mean"):
        state.pop(key, None)
    tensors = {}
    for name, value in state.items():
        tensors[name] = value.float().contiguous()
    metadata = {
        "evanalyzer_arch": "cellpose_sam",
        "arch_version": "1",
        "config": json.dumps(CONFIG, separators=(",", ":")),
        "source": args.source,
        "license": "BSD-3-Clause (Cellpose, HHMI); SAM encoder Apache-2.0 (Meta)",
        "citation": "Pachitariu, Rariden, Stringer (2025) Cellpose-SAM, bioRxiv 10.1101/2025.04.28.651001",
    }
    save_file(tensors, args.output, metadata=metadata)
    print(f"wrote {args.output}: {len(tensors)} tensors")

    if args.references:
        from cellpose.vit import CPSAM

        os.makedirs(args.references, exist_ok=True)
        net = CPSAM()
        missing, unexpected = net.load_state_dict(
            {**tensors, "W2": w2.float()}, strict=False
        )
        assert not unexpected, unexpected
        assert set(missing) <= {"diam_labels", "diam_mean"}, missing
        net.eval()
        img = cells(256, 1)
        inputs = [
            np.stack([img, np.zeros_like(img), np.zeros_like(img)]),  # gray, zero-filled
            np.stack([img, img, img]),  # gray, replicated
            img[None],  # one channel
        ]
        for i, x in enumerate(inputs):
            with torch.no_grad():
                y, _ = net(torch.from_numpy(x)[None])
            x.astype("<f4").tofile(os.path.join(args.references, f"tile{i}.input.f32"))
            y[0].numpy().astype("<f4").tofile(os.path.join(args.references, f"tile{i}.output.f32"))
            print(f"tile{i}: input {x.shape}, output {tuple(y.shape)}")


if __name__ == "__main__":
    main()
